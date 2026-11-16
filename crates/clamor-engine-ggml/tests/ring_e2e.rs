//! A real model (the tiny random Llama) running on the ring transport: GgmlExecutor stages on
//! loopback TCP, driven through the head exactly as a node would drive them.
//!
//! Two kinds of check:
//! * against the numpy reference's greedy generation (tools/ref_forward.py --generate), and
//! * bit-for-bit against the same executors called in-process with no network ("direct"),
//!   which isolates the transport: any difference is the ring's fault, not the engine's.
//!
//! Needs python3 with numpy + gguf (skips without, unless CLAMOR_REQUIRE_PY=1).
use clamor_engine_ggml::*;
use clamor_transport::frame::{Acts, DType};
use clamor_transport::stage::{run_stage, DoneReason, Event, HeadHandle, StageConfig, StageExecutor, StageHandle, StageInput, StageOutput};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::mpsc::UnboundedReceiver;

const PROMPT: [u32; 6] = [1, 5, 9, 23, 42, 7];
const N_GEN: u32 = 16;

struct Fixture {
    model: String,
    /// Numpy greedy continuation of PROMPT.
    greedy: Vec<u32>,
}

fn fixture() -> Option<&'static Fixture> {
    static F: OnceLock<Option<Fixture>> = OnceLock::new();
    F.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("clamor-ring-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("tiny.gguf").display().to_string();
        let tools = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools");
        let py = |args: Vec<String>| std::process::Command::new("python3").args(&args).output().map(|o| o.status.success()).unwrap_or(false);
        let ok = py(vec![format!("{tools}/gen_tiny_llama.py"), model.clone(), "--layers".into(), "4".into()])
            && py(vec![
                format!("{tools}/ref_forward.py"),
                model.clone(),
                dir.join("ref.npz").display().to_string(),
                "--tokens".into(),
                PROMPT.map(|t| t.to_string()).join(","),
                "--generate".into(),
                N_GEN.to_string(),
                "--raw".into(),
                dir.display().to_string(),
            ]);
        if !ok {
            if std::env::var("CLAMOR_REQUIRE_PY").as_deref() == Ok("1") {
                panic!("python3 with numpy+gguf is required (CLAMOR_REQUIRE_PY=1)");
            }
            eprintln!("skipping: python3 with numpy+gguf not available");
            return None;
        }
        let greedy = std::fs::read_to_string(dir.join("generated.txt")).unwrap().split(',').map(|t| t.parse().unwrap()).collect();
        Some(Fixture { model, greedy })
    })
    .as_ref()
}

fn cpu() -> usize {
    list_devices().iter().find(|d| d.kind == DeviceKind::Cpu).expect("a CPU device").index
}

fn opts(wire: DType, n_ctx: u32) -> ExecutorOptions {
    ExecutorOptions { stage: StageOptions { device: cpu(), n_ctx, n_threads: 2, ..Default::default() }, wire, sampling: Sampling::default(), eos: Some(vec![]) }
}

fn executors(f: &Fixture, bounds: &[u32], o: &ExecutorOptions) -> Vec<GgmlExecutor> {
    bounds.windows(2).map(|w| GgmlExecutor::load(&f.model, w[0]..w[1], o.clone()).unwrap()).collect()
}

struct Ring {
    stages: Vec<StageHandle>,
    live: Vec<Arc<AtomicUsize>>,
}
impl Ring {
    fn head(&self) -> &HeadHandle {
        self.stages[0].head.as_ref().unwrap()
    }
    /// Wait until no stage holds a KV cache (cancels ripple asynchronously).
    async fn assert_all_released(&self) {
        for _ in 0..200 {
            if self.live.iter().all(|l| l.load(Ordering::Relaxed) == 0) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let live: Vec<usize> = self.live.iter().map(|l| l.load(Ordering::Relaxed)).collect();
        panic!("KV caches still held per stage: {live:?}");
    }
}
impl Drop for Ring {
    fn drop(&mut self) {
        for s in &self.stages {
            s.stop();
        }
    }
}

async fn spawn_ring(execs: Vec<GgmlExecutor>, window: u32) -> Ring {
    let n = execs.len();
    let mut listeners = Vec::new();
    let mut addrs = Vec::new();
    for _ in 0..n {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        addrs.push(l.local_addr().unwrap());
        listeners.push(l);
    }
    let live = execs.iter().map(|e| e.live_requests()).collect();
    let stages = execs
        .into_iter()
        .zip(listeners)
        .enumerate()
        .map(|(i, (exec, l))| {
            let cfg = StageConfig {
                stage_index: i as u32,
                stage_count: n as u32,
                epoch: 1,
                session: 42,
                downstream: if n == 1 { None } else { Some(addrs[(i + 1) % n]) },
                window,
                connect_timeout: Duration::from_secs(5),
                stall_timeout: clamor_transport::stage::DEFAULT_STALL_TIMEOUT,
                tls: None,
            };
            run_stage(cfg, l, exec)
        })
        .collect();
    Ring { stages, live }
}

async fn collect(mut rx: UnboundedReceiver<Event>) -> (Vec<u32>, DoneReason) {
    let mut toks = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(60), rx.recv()).await.expect("timed out waiting for the ring") {
            Some(Event::Token { id, .. }) => toks.push(id),
            Some(Event::Done(r)) => return (toks, r),
            None => panic!("event channel closed without Done"),
        }
    }
}

/// The same executors, called in-process in ring order with no network: what the ring must
/// reproduce bit for bit.
fn direct(execs: &mut [GgmlExecutor], req: u64, prompt: &[u32], max_new: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let (mut pos, mut toks) = (0u32, prompt.to_vec());
    while out.len() < max_new as usize {
        let mut acts: Option<Acts> = None;
        let mut token = None;
        for e in execs.iter_mut() {
            let input = match acts.take() {
                None => StageInput::Tokens(&toks),
                Some(a) => StageInput::Acts(a),
            };
            match e.forward(req, pos, input).unwrap() {
                StageOutput::Acts(a) => acts = Some(a),
                StageOutput::Token { id, .. } => token = Some(id),
            }
        }
        let id = token.expect("the last stage samples");
        out.push(id);
        pos += toks.len() as u32;
        toks = vec![id];
    }
    for e in execs.iter_mut() {
        e.release(req);
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn solo_ring_matches_numpy_greedy() {
    let Some(f) = fixture() else { return };
    let ring = spawn_ring(executors(f, &[0, 4], &opts(DType::F16, 64)), 4).await;
    let (_, rx) = ring.head().submit(PROMPT.to_vec(), N_GEN);
    assert_eq!(collect(rx).await, (f.greedy.clone(), DoneReason::MaxTokens));
    ring.assert_all_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn split_rings_match_numpy_greedy() {
    let Some(f) = fixture() else { return };
    // f16 activations between stages add ~1e-3 rounding. That is far below this model's
    // smallest top-1/top-2 logit gap over the generation, so greedy text must not change.
    for bounds in [vec![0u32, 2, 4], vec![0, 1, 3, 4], vec![0, 1, 2, 3, 4]] {
        let ring = spawn_ring(executors(f, &bounds, &opts(DType::F16, 64)), 4).await;
        let (_, rx) = ring.head().submit(PROMPT.to_vec(), N_GEN);
        assert_eq!(collect(rx).await, (f.greedy.clone(), DoneReason::MaxTokens), "bounds {bounds:?}");
        ring.assert_all_released().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn ring_is_bit_exact_with_direct_calls_f16_and_i8() {
    let Some(f) = fixture() else { return };
    let bounds = [0u32, 1, 3, 4];
    for wire in [DType::F16, DType::I8] {
        let want = direct(&mut executors(f, &bounds, &opts(wire, 64)), 1, &PROMPT, 24);
        let ring = spawn_ring(executors(f, &bounds, &opts(wire, 64)), 4).await;
        let (_, rx) = ring.head().submit(PROMPT.to_vec(), 24);
        assert_eq!(collect(rx).await, (want, DoneReason::MaxTokens), "{wire:?}");
        ring.assert_all_released().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_requests_stay_independent() {
    let Some(f) = fixture() else { return };
    let bounds = [0u32, 2, 4];
    let prompts: Vec<Vec<u32>> = (0..10u32).map(|i| (0..1 + i % 4).map(|k| (i * 13 + k * 7) % 128).collect()).collect();
    let mut solo = executors(f, &bounds, &opts(DType::F16, 64));
    let want: Vec<Vec<u32>> = prompts.iter().map(|p| direct(&mut solo, 99, p, 12)).collect();
    // Window 4 with 10 requests: the head queues the rest and admits them as slots free.
    let ring = spawn_ring(executors(f, &bounds, &opts(DType::F16, 64)), 4).await;
    let runs: Vec<_> = prompts.iter().map(|p| ring.head().submit(p.clone(), 12)).collect();
    for (i, (_, rx)) in runs.into_iter().enumerate() {
        assert_eq!(collect(rx).await, (want[i].clone(), DoneReason::MaxTokens), "request {i}");
    }
    ring.assert_all_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cancel_frees_every_stage() {
    let Some(f) = fixture() else { return };
    let ring = spawn_ring(executors(f, &[0, 1, 3, 4], &opts(DType::F16, 4096)), 4).await;
    let (req, mut rx) = ring.head().submit(PROMPT.to_vec(), 3000);
    for _ in 0..3 {
        assert!(matches!(rx.recv().await, Some(Event::Token { .. })));
    }
    ring.head().cancel(req);
    let (_, reason) = collect(rx).await;
    assert_eq!(reason, DoneReason::Cancelled);
    ring.assert_all_released().await;
}

/// A request that outgrows its KV cache fails on its own; the ring keeps serving.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn context_overflow_fails_only_that_request() {
    let Some(f) = fixture() else { return };
    let bounds = [0u32, 2, 4];
    let ring = spawn_ring(executors(f, &bounds, &opts(DType::F16, 12)), 4).await;
    // 6 prompt tokens + 20 more needs 26 cells; there are 12.
    let (_, rx) = ring.head().submit(PROMPT.to_vec(), 20);
    let (toks, reason) = collect(rx).await;
    match reason {
        DoneReason::Failed(msg) => assert!(msg.contains("context full"), "{msg}"),
        r => panic!("expected Failed, got {r:?}"),
    }
    // Cells 0..12 hold the prompt plus 6 decode steps, each of which yields a token: 7 tokens.
    assert_eq!(toks, f.greedy[..7].to_vec());
    let (_, rx) = ring.head().submit(vec![3, 4], 5);
    let want = direct(&mut executors(f, &bounds, &opts(DType::F16, 12)), 1, &[3, 4], 5);
    assert_eq!(collect(rx).await, (want, DoneReason::MaxTokens), "ring must survive");
    ring.assert_all_released().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn eos_stops_generation() {
    let Some(f) = fixture() else { return };
    let eos = f.greedy[2];
    let first = f.greedy.iter().position(|&t| t == eos).unwrap();
    let mut o = opts(DType::F16, 64);
    o.eos = Some(vec![eos]);
    let ring = spawn_ring(executors(f, &[0, 2, 4], &o), 4).await;
    let (_, rx) = ring.head().submit(PROMPT.to_vec(), N_GEN);
    assert_eq!(collect(rx).await, (f.greedy[..=first].to_vec(), DoneReason::Eos));
    ring.assert_all_released().await;
}

#[test]
fn sampling_is_seeded_and_top_k_1_is_greedy() {
    let Some(f) = fixture() else { return };
    let bounds = [0u32, 4];
    let with = |s: Sampling| {
        let mut o = opts(DType::F16, 64);
        o.sampling = s;
        direct(&mut executors(f, &bounds, &o), 5, &PROMPT, N_GEN)
    };
    let hot = Sampling { temperature: 1.5, top_k: 0, seed: 11 };
    let a = with(hot);
    assert_eq!(a, with(hot), "same seed, same tokens");
    assert_ne!(a, f.greedy, "temperature 1.5 should wander from greedy on a random model");
    assert_ne!(a, with(Sampling { seed: 12, ..hot }), "different seed, different tokens");
    assert_eq!(with(Sampling { temperature: 1.5, top_k: 1, seed: 3 }), f.greedy, "top-k 1 is greedy");
}

#[test]
fn last_token_logits_match_full_logits() {
    let Some(f) = fixture() else { return };
    let o = StageOptions { device: cpu(), n_ctx: 32, ..Default::default() };
    let mut a = LlamaStage::load(&f.model, 0..4, o.clone()).unwrap();
    let mut b = LlamaStage::load(&f.model, 0..4, o).unwrap();
    let (LlamaOutput::Logits(all), LlamaOutput::Logits(last)) = (
        a.forward_logits(1, 0, LlamaInput::Tokens(&PROMPT), Logits::All).unwrap(),
        b.forward_logits(1, 0, LlamaInput::Tokens(&PROMPT), Logits::Last).unwrap(),
    ) else {
        panic!("tail returns logits")
    };
    let v = a.hparams().n_vocab as usize;
    assert_eq!(last.len(), v);
    let max_err = all[all.len() - v..].iter().zip(&last).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    assert!(max_err < 1e-5, "{max_err}");
}

#[test]
fn executor_rejects_malformed_activations() {
    let Some(f) = fixture() else { return };
    let mut tail = GgmlExecutor::load(&f.model, 2..4, opts(DType::F16, 64)).unwrap();
    let short = Acts { dtype: DType::F16, scale: 1.0, n_tokens: 2, data: vec![0u8; 64 * 2].into() }; // one token's worth
    assert!(tail.forward(1, 0, StageInput::Acts(short)).is_err());
    let untyped = Acts { dtype: DType::None, scale: 1.0, n_tokens: 1, data: vec![0u8; 64 * 2].into() };
    assert!(tail.forward(1, 0, StageInput::Acts(untyped)).is_err());
    assert!(GgmlExecutor::load(&f.model, 0..4, ExecutorOptions { wire: DType::None, ..opts(DType::F16, 64) }).is_err());
}

#[test]
fn wire_encoding_is_what_was_asked_for() {
    let Some(f) = fixture() else { return };
    for (wire, bytes_per_value) in [(DType::F16, 2usize), (DType::I8, 1)] {
        let mut head = GgmlExecutor::load(&f.model, 0..2, opts(wire, 64)).unwrap();
        let StageOutput::Acts(a) = head.forward(1, 0, StageInput::Tokens(&PROMPT)).unwrap() else { panic!("head sends activations") };
        assert_eq!(a.dtype, wire);
        assert_eq!(a.n_tokens, PROMPT.len() as u32);
        assert_eq!(a.data.len(), PROMPT.len() * 64 * bytes_per_value, "{wire:?}");
        if wire == DType::I8 {
            assert!(a.scale > 0.0 && a.scale != 1.0, "int8 carries a real scale: {}", a.scale);
        }
    }
}
