//! A deterministic stand-in for a real engine, so the transport can be tested and benchmarked
//! without model weights. The "model" is a hash chain over layers: splitting the layers across
//! any number of stages must produce exactly the same tokens as running them in one stage, which
//! makes the pipeline's correctness checkable end to end.

use crate::frame::{Acts, DType};
use crate::stage::{role_of, run_stage, StageConfig, StageExecutor, StageHandle, StageInput, StageOutput};
use anyhow::{bail, Result};
use bytes::Bytes;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(a << 6).wrapping_add(a >> 2);
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 31)
}

const SEED: u64 = 0x1234;

pub struct MockExecutor {
    pub layers: Range<u32>,
    pub is_tail: bool,
    pub hidden: u32,
    pub per_layer: Duration,
    pub vocab: u32,
    pub eos: Option<u32>,
    /// Fail any forward that starts at or past this position (simulates a full KV cache).
    pub fail_from_pos: Option<u32>,
    hist: HashMap<u64, u64>,
    pub released: Arc<Mutex<Vec<u64>>>,
}

impl MockExecutor {
    pub fn new(layers: Range<u32>, is_tail: bool, hidden: u32, per_layer: Duration, vocab: u32) -> Self {
        MockExecutor { layers, is_tail, hidden, per_layer, vocab, eos: None, fail_from_pos: None, hist: HashMap::new(), released: Arc::new(Mutex::new(Vec::new())) }
    }
}

impl StageExecutor for MockExecutor {
    fn forward(&mut self, req: u64, pos: u32, input: StageInput<'_>) -> Result<StageOutput> {
        if let Some(limit) = self.fail_from_pos {
            if pos >= limit {
                bail!("context full at position {pos}");
            }
        }
        let (mut h, hist, n_tokens) = match input {
            StageInput::Tokens(t) => {
                let mut hist = *self.hist.get(&req).unwrap_or(&SEED);
                for &x in t {
                    hist = mix(hist, x as u64);
                }
                self.hist.insert(req, hist);
                (mix(hist, 0xA11CE), hist, t.len() as u32)
            }
            StageInput::Acts(a) => {
                if a.data.len() < 16 {
                    bail!("activation frame too short");
                }
                let h = u64::from_le_bytes(a.data[0..8].try_into().unwrap());
                let hist = u64::from_le_bytes(a.data[8..16].try_into().unwrap());
                (h, hist, a.n_tokens)
            }
        };
        if !self.per_layer.is_zero() {
            std::thread::sleep(self.per_layer * self.layers.len() as u32);
        }
        for l in self.layers.clone() {
            h = mix(h, l as u64);
        }
        if self.is_tail {
            let id = (h % self.vocab as u64) as u32;
            return Ok(StageOutput::Token { id, eos: self.eos == Some(id) });
        }
        // Size the payload like real f16 hidden states so link traffic is realistic.
        let len = (n_tokens as usize * self.hidden as usize * 2).max(16);
        let mut data = vec![0u8; len];
        data[0..8].copy_from_slice(&h.to_le_bytes());
        data[8..16].copy_from_slice(&hist.to_le_bytes());
        Ok(StageOutput::Acts(Acts { dtype: DType::F16, scale: 1.0, n_tokens, data: Bytes::from(data) }))
    }

    fn release(&mut self, request: u64) {
        self.hist.remove(&request);
        self.released.lock().unwrap().push(request);
    }
}

/// What a correct pipeline must produce, computed without any transport.
pub fn reference_tokens(prompt: &[u32], max_new: u32, total_layers: u32, vocab: u32) -> Vec<u32> {
    let mut hist = SEED;
    for &x in prompt {
        hist = mix(hist, x as u64);
    }
    let mut out = Vec::new();
    for i in 0..max_new {
        let mut h = mix(hist, 0xA11CE);
        for l in 0..total_layers {
            h = mix(h, l as u64);
        }
        let id = (h % vocab as u64) as u32;
        out.push(id);
        if i + 1 < max_new {
            hist = mix(hist, id as u64);
        }
    }
    out
}

pub struct MockRing {
    pub stages: Vec<StageHandle>,
    pub released: Vec<Arc<Mutex<Vec<u64>>>>,
}

impl MockRing {
    pub fn head(&self) -> &crate::stage::HeadHandle {
        self.stages[0].head.as_ref().expect("stage 0 is the head")
    }
    pub fn stop(&self) {
        for s in &self.stages {
            s.stop();
        }
    }
}

/// Build a loopback ring. `bounds` are the layer boundaries, e.g. `[0, 10, 25, 32]` makes
/// three stages holding layers 0..10, 10..25 and 25..32.
pub async fn spawn_mock_ring(bounds: &[u32], hidden: u32, per_layer: Duration, vocab: u32, window: u32, epoch: u32) -> MockRing {
    let n = bounds.len() - 1;
    let mut listeners = Vec::new();
    let mut addrs: Vec<SocketAddr> = Vec::new();
    for _ in 0..n {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        addrs.push(l.local_addr().unwrap());
        listeners.push(l);
    }
    let mut stages = Vec::new();
    let mut released = Vec::new();
    for (i, listener) in listeners.into_iter().enumerate() {
        let is_tail = i + 1 == n;
        let exec = MockExecutor::new(bounds[i]..bounds[i + 1], is_tail, hidden, per_layer, vocab);
        released.push(exec.released.clone());
        let cfg = StageConfig {
            stage_index: i as u32,
            stage_count: n as u32,
            epoch,
            session: 7,
            downstream: if n == 1 { None } else { Some(addrs[(i + 1) % n]) },
            window,
            connect_timeout: Duration::from_secs(5),
            stall_timeout: crate::stage::DEFAULT_STALL_TIMEOUT,
        };
        debug_assert_eq!(role_of(cfg.stage_index, cfg.stage_count) == crate::stage::Role::Solo, n == 1);
        stages.push(run_stage(cfg, listener, exec));
    }
    MockRing { stages, released }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::{DoneReason, Event};
    use tokio::sync::mpsc::UnboundedReceiver;

    async fn collect(mut rx: UnboundedReceiver<Event>) -> (Vec<u32>, DoneReason) {
        let mut toks = Vec::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.expect("timed out waiting for the pipeline") {
                Some(Event::Token { id, .. }) => toks.push(id),
                Some(Event::Done(r)) => return (toks, r),
                None => panic!("event channel closed without Done"),
            }
        }
    }

    const VOCAB: u32 = 32_000;

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn three_stage_ring_matches_single_stage_reference() {
        let ring = spawn_mock_ring(&[0, 10, 25, 32], 64, Duration::ZERO, VOCAB, 8, 1).await;
        let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![42], vec![7, 7, 7, 7, 7, 7], vec![100, 200]];
        let runs: Vec<_> = prompts.iter().map(|p| ring.head().submit(p.clone(), 20)).collect();
        for ((_, rx), p) in runs.into_iter().zip(&prompts) {
            let (toks, reason) = collect(rx).await;
            assert_eq!(reason, DoneReason::MaxTokens);
            assert_eq!(toks, reference_tokens(p, 20, 32, VOCAB), "prompt {p:?}");
        }
        ring.stop();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn two_stage_and_solo_agree_with_reference() {
        for bounds in [vec![0u32, 16, 32], vec![0, 32]] {
            let ring = spawn_mock_ring(&bounds, 64, Duration::ZERO, VOCAB, 4, 1).await;
            let (_, rx) = ring.head().submit(vec![5, 6, 7], 12);
            let (toks, _) = collect(rx).await;
            assert_eq!(toks, reference_tokens(&[5, 6, 7], 12, 32, VOCAB), "bounds {bounds:?}");
            ring.stop();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn many_concurrent_requests_stay_independent() {
        let ring = spawn_mock_ring(&[0, 8, 16, 24, 32], 32, Duration::ZERO, VOCAB, 4, 1).await;
        let runs: Vec<_> = (0..24u32).map(|i| (i, ring.head().submit(vec![i, i + 1], 10 + i % 5))).collect();
        for (i, (_, rx)) in runs {
            let (toks, _) = collect(rx).await;
            assert_eq!(toks, reference_tokens(&[i, i + 1], 10 + i % 5, 32, VOCAB), "request {i}");
        }
        ring.stop();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn cancel_stops_generation_and_releases_every_stage() {
        let ring = spawn_mock_ring(&[0, 10, 20, 30], 32, Duration::from_micros(300), VOCAB, 4, 1).await;
        let (req, mut rx) = ring.head().submit(vec![1, 2], 100_000);
        for _ in 0..3 {
            assert!(matches!(rx.recv().await, Some(Event::Token { .. })));
        }
        ring.head().cancel(req);
        let mut reason = None;
        while let Some(ev) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap() {
            if let Event::Done(r) = ev {
                reason = Some(r);
                break;
            }
        }
        assert_eq!(reason, Some(DoneReason::Cancelled));
        // The cancel ripples down the ring; every executor must free the request.
        for _ in 0..100 {
            if ring.released.iter().all(|r| r.lock().unwrap().contains(&req)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        for (i, r) in ring.released.iter().enumerate() {
            assert!(r.lock().unwrap().contains(&req), "stage {i} did not release {req}");
        }
        ring.stop();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn eos_ends_a_request_early() {
        // Tail treats the token this prompt produces first as EOS.
        let first = reference_tokens(&[9, 9], 1, 30, VOCAB)[0];
        let n = 3usize;
        let bounds = [0u32, 10, 20, 30];
        let mut listeners = Vec::new();
        let mut addrs = Vec::new();
        for _ in 0..n {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            addrs.push(l.local_addr().unwrap());
            listeners.push(l);
        }
        let mut handles = Vec::new();
        for (i, l) in listeners.into_iter().enumerate() {
            let mut exec = MockExecutor::new(bounds[i]..bounds[i + 1], i + 1 == n, 16, Duration::ZERO, VOCAB);
            if i + 1 == n {
                exec.eos = Some(first);
            }
            let cfg = StageConfig { stage_index: i as u32, stage_count: n as u32, epoch: 1, session: 7, downstream: Some(addrs[(i + 1) % n]), window: 4, connect_timeout: Duration::from_secs(5), stall_timeout: crate::stage::DEFAULT_STALL_TIMEOUT };
            handles.push(run_stage(cfg, l, exec));
        }
        let (_, rx) = handles[0].head.as_ref().unwrap().submit(vec![9, 9], 50);
        let (toks, reason) = collect(rx).await;
        assert_eq!(reason, DoneReason::Eos);
        assert_eq!(toks, vec![first]);
        for h in &handles {
            h.stop();
        }
    }

    /// Regression: an engine error used to end the stage loop, which broke the ring and failed
    /// every request. Now it fails only that request, wherever in the ring it happens.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn engine_error_fails_one_request_not_the_ring() {
        let bounds = [0u32, 10, 20, 30];
        for failing in 0..3usize {
            let n = 3usize;
            let mut listeners = Vec::new();
            let mut addrs = Vec::new();
            for _ in 0..n {
                let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
                addrs.push(l.local_addr().unwrap());
                listeners.push(l);
            }
            let mut handles = Vec::new();
            let mut released = Vec::new();
            for (i, l) in listeners.into_iter().enumerate() {
                let mut exec = MockExecutor::new(bounds[i]..bounds[i + 1], i + 1 == n, 16, Duration::ZERO, VOCAB);
                if i == failing {
                    exec.fail_from_pos = Some(6);
                }
                released.push(exec.released.clone());
                let cfg = StageConfig { stage_index: i as u32, stage_count: n as u32, epoch: 1, session: 7, downstream: Some(addrs[(i + 1) % n]), window: 4, connect_timeout: Duration::from_secs(5), stall_timeout: crate::stage::DEFAULT_STALL_TIMEOUT };
                handles.push(run_stage(cfg, l, exec));
            }
            let head = handles[0].head.as_ref().unwrap();

            // Forwards start at 0 (the 2-token prompt), then 2, 3, 4, 5 (one token each): each yields
            // a token, so 5 tokens come back before the forward starting at position 6 fails.
            let (req, rx) = head.submit(vec![1, 2], 50);
            let (toks, reason) = collect(rx).await;
            match &reason {
                DoneReason::Failed(msg) => assert!(msg.contains(&format!("stage {failing}")) && msg.contains("context full"), "{msg}"),
                r => panic!("stage {failing}: expected Failed, got {r:?}"),
            }
            assert_eq!(toks, reference_tokens(&[1, 2], 5, 30, VOCAB), "stage {failing}: tokens before the failure");
            for _ in 0..100 {
                if released.iter().all(|r| r.lock().unwrap().contains(&req)) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            for (i, r) in released.iter().enumerate() {
                assert!(r.lock().unwrap().contains(&req), "failing {failing}: stage {i} did not release {req}");
            }

            // The ring survived: short requests still complete correctly.
            let (_, rx) = head.submit(vec![3], 4);
            assert_eq!(collect(rx).await, (reference_tokens(&[3], 4, 30, VOCAB), DoneReason::MaxTokens), "stage {failing}: ring dead");
            for h in &handles {
                h.stop();
            }
        }
    }

    /// A stage that stops making progress (frozen, or a half-dead link) must not leave requests
    /// hanging: the head fails them after `stall_timeout`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn a_stalled_stage_fails_requests_instead_of_hanging() {
        let n = 3usize;
        let bounds = [0u32, 10, 20, 30];
        let mut listeners = Vec::new();
        let mut addrs = Vec::new();
        for _ in 0..n {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            addrs.push(l.local_addr().unwrap());
            listeners.push(l);
        }
        let mut handles = Vec::new();
        for (i, l) in listeners.into_iter().enumerate() {
            // The middle stage takes 3 s per step, against a 0.5 s stall timeout: frozen.
            let per_layer = if i == 1 { Duration::from_millis(300) } else { Duration::ZERO };
            let exec = MockExecutor::new(bounds[i]..bounds[i + 1], i + 1 == n, 16, per_layer, VOCAB);
            let cfg = StageConfig { stage_index: i as u32, stage_count: n as u32, epoch: 1, session: 7, downstream: Some(addrs[(i + 1) % n]), window: 4, connect_timeout: Duration::from_secs(5), stall_timeout: Duration::from_millis(500) };
            handles.push(run_stage(cfg, l, exec));
        }
        let t0 = std::time::Instant::now();
        let (_, rx) = handles[0].head.as_ref().unwrap().submit(vec![1, 2], 10);
        let (toks, reason) = collect(rx).await;
        assert!(toks.is_empty());
        match reason {
            DoneReason::Failed(m) => assert!(m.contains("no progress"), "{m}"),
            r => panic!("expected a stall failure, got {r:?}"),
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "failed after {:?}, not promptly", t0.elapsed());
        for h in &handles {
            h.stop();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn empty_prompt_and_zero_budget_are_handled() {
        let ring = spawn_mock_ring(&[0, 16, 32], 16, Duration::ZERO, VOCAB, 4, 1).await;
        let (_, rx) = ring.head().submit(vec![], 5);
        assert!(matches!(collect(rx).await.1, DoneReason::Failed(_)));
        let (_, rx) = ring.head().submit(vec![1], 0);
        assert_eq!(collect(rx).await, (vec![], DoneReason::MaxTokens));
        ring.stop();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn losing_a_stage_fails_requests_instead_of_hanging() {
        let ring = spawn_mock_ring(&[0, 10, 20, 30], 16, Duration::from_micros(500), VOCAB, 4, 1).await;
        let (_, mut rx) = ring.head().submit(vec![1], 1_000_000);
        assert!(matches!(rx.recv().await, Some(Event::Token { .. })));
        ring.stages[1].stop(); // a device vanishes mid-generation
        let mut failed = false;
        while let Some(ev) = tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.expect("request must not hang") {
            if let Event::Done(DoneReason::Failed(_)) = ev {
                failed = true;
                break;
            }
        }
        assert!(failed);
        ring.stop();
    }
}
