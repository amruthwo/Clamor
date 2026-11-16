//! Ring pipeline stages.
//!
//! ```text
//!   head (stage 0) --acts--> mid --acts--> ... --acts--> tail
//!      ^                                                   |
//!      +------------------------- token -------------------+
//! ```
//!
//! The head embeds tokens and owns the request table; the tail samples the next token and sends
//! it straight back to the head. Each stage listens for its upstream and connects to its
//! downstream, so any start order works. With a single stage there is no network at all.

use crate::frame::{Acts, Frame, Kind, FLAG_EOS};
use crate::link::{Link, LinkParams};
use anyhow::{anyhow, bail, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Solo,
    Head,
    Mid,
    Tail,
}

pub fn role_of(index: u32, count: u32) -> Role {
    match (index, count) {
        (_, 1) => Role::Solo,
        (0, _) => Role::Head,
        (i, n) if i + 1 == n => Role::Tail,
        _ => Role::Mid,
    }
}

pub struct StageConfig {
    pub stage_index: u32,
    pub stage_count: u32,
    /// Bumped by the coordinator on every re-plan. Frames from another epoch are dropped.
    pub epoch: u32,
    pub session: u64,
    /// Next stage in the ring (the head, for the tail). `None` only for a single stage.
    pub downstream: Option<SocketAddr>,
    /// Data frames buffered per link.
    pub window: u32,
    pub connect_timeout: Duration,
    /// The head fails a request that makes no progress for this long (a stage froze, or a
    /// link half-died). Generous: a slow device can take minutes over a long prompt.
    pub stall_timeout: Duration,
    /// Encrypt both links (pinned neighbour certificates). None: plain TCP (tests only).
    pub tls: Option<Arc<crate::link::LinkTls>>,
}

/// Default for `StageConfig::stall_timeout`.
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(300);

pub enum StageInput<'a> {
    Tokens(&'a [u32]),
    Acts(Acts),
}

pub enum StageOutput {
    Acts(Acts),
    Token { id: u32, eos: bool },
}

/// The inference engine behind a stage. Called from a blocking-safe context, one call at a time.
pub trait StageExecutor: Send + 'static {
    /// Run this stage's layers. `pos` is the position of the first token in the input.
    fn forward(&mut self, request: u64, pos: u32, input: StageInput<'_>) -> Result<StageOutput>;
    /// Free per-request state (KV cache). Must tolerate unknown ids.
    fn release(&mut self, request: u64);
}

#[derive(Default)]
pub struct StageStats {
    pub frames_in: AtomicU64,
    pub frames_out: AtomicU64,
    pub bytes_in: AtomicU64,
    pub bytes_out: AtomicU64,
    pub compute_ns: AtomicU64,
    pub dropped_stale: AtomicU64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DoneReason {
    MaxTokens,
    Eos,
    Cancelled,
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Token { id: u32, pos: u32 },
    Done(DoneReason),
}

enum HeadCmd {
    Submit { req: u64, prompt: Vec<u32>, max_new: u32, out: mpsc::UnboundedSender<Event> },
    Cancel(u64),
}

#[derive(Clone)]
pub struct HeadHandle {
    cmd: mpsc::UnboundedSender<HeadCmd>,
    next_req: Arc<AtomicU64>,
}

impl HeadHandle {
    /// Start generating. Events stream on the returned channel and end with `Event::Done`.
    pub fn submit(&self, prompt: Vec<u32>, max_new: u32) -> (u64, mpsc::UnboundedReceiver<Event>) {
        let req = self.next_req.fetch_add(1, Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        if let Err(e) = self.cmd.send(HeadCmd::Submit { req, prompt, max_new, out: tx }) {
            if let HeadCmd::Submit { out, .. } = e.0 {
                let _ = out.send(Event::Done(DoneReason::Failed("pipeline stopped".into())));
            }
        }
        (req, rx)
    }

    pub fn cancel(&self, req: u64) {
        let _ = self.cmd.send(HeadCmd::Cancel(req));
    }
}

pub struct StageHandle {
    pub role: Role,
    pub head: Option<HeadHandle>,
    pub stats: Arc<StageStats>,
    pub task: JoinHandle<Result<()>>,
}

impl StageHandle {
    pub fn stop(&self) {
        self.task.abort();
    }
}

/// Start a stage. Returns immediately; the links come up in the background.
pub fn run_stage<E: StageExecutor>(cfg: StageConfig, listener: TcpListener, exec: E) -> StageHandle {
    let role = role_of(cfg.stage_index, cfg.stage_count);
    let stats = Arc::new(StageStats::default());
    let (head, cmd_rx) = match role {
        Role::Head | Role::Solo => {
            let (tx, rx) = mpsc::unbounded_channel();
            (Some(HeadHandle { cmd: tx, next_req: Arc::new(AtomicU64::new(1)) }), Some(rx))
        }
        _ => (None, None),
    };
    let task = tokio::spawn(stage_main(cfg, listener, exec, role, stats.clone(), cmd_rx));
    StageHandle { role, head, stats, task }
}

async fn stage_main<E: StageExecutor>(cfg: StageConfig, listener: TcpListener, exec: E, role: Role, stats: Arc<StageStats>, cmd_rx: Option<mpsc::UnboundedReceiver<HeadCmd>>) -> Result<()> {
    let (up, down) = if role == Role::Solo {
        (None, None)
    } else {
        let params = LinkParams { epoch: cfg.epoch, session: cfg.session, stage: cfg.stage_index, window: cfg.window };
        let addr = cfg.downstream.ok_or_else(|| anyhow!("stage {} has no downstream address", cfg.stage_index))?;
        // Accept and connect concurrently: every stage does both, so a sequential order would deadlock.
        let tls = cfg.tls.as_deref();
        let (up, down) = tokio::try_join!(Link::accept(&listener, params, tls), Link::connect(addr, params, cfg.connect_timeout, tls))?;
        (Some(up), Some(down))
    };
    match role {
        Role::Solo | Role::Head => head_loop(cfg, exec, up, down, cmd_rx.expect("head has a command channel"), stats).await,
        Role::Mid | Role::Tail => worker_loop(cfg, exec, role, up.expect("worker has upstream"), down.expect("worker has downstream"), stats).await,
    }
}

struct ReqState {
    generated: u32,
    max_new: u32,
    out: mpsc::UnboundedSender<Event>,
    /// Last time this request moved forward (sent on, or a token back).
    progress: Instant,
}

struct Pending {
    req: u64,
    prompt: Vec<u32>,
    max_new: u32,
    out: mpsc::UnboundedSender<Event>,
}

/// Every active request has exactly one frame in flight (activations going round, or a token
/// coming back). Capping active requests at the link window therefore guarantees no link ever
/// needs more credit than it has, so the ring cannot deadlock with the head blocked sending
/// while the tail is blocked sending to the head. Extra requests wait here.
struct Head<E> {
    cfg: StageConfig,
    exec: E,
    down: Option<Link>,
    reqs: HashMap<u64, ReqState>,
    pending: VecDeque<Pending>,
    max_active: usize,
    stats: Arc<StageStats>,
}

impl<E: StageExecutor> Head<E> {
    /// Run the head's layers for `tokens` and push the result downstream. In solo mode the
    /// output is a token, which loops back into the next decode step without any network.
    async fn forward(&mut self, req: u64, pos: u32, tokens: Vec<u32>) -> Result<()> {
        if let Some(st) = self.reqs.get_mut(&req) {
            st.progress = Instant::now();
        }
        let mut next = Some((pos, tokens));
        while let Some((pos, toks)) = next.take() {
            let t0 = Instant::now();
            let res = tokio::task::block_in_place(|| self.exec.forward(req, pos, StageInput::Tokens(&toks)));
            self.stats.compute_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            // An engine error fails this request only; the ring keeps serving the others.
            let out = match res {
                Ok(o) => o,
                Err(e) => {
                    self.fail(req, format!("stage {}: {e:#}", self.cfg.stage_index));
                    return Ok(());
                }
            };
            match out {
                StageOutput::Acts(a) => {
                    let f = Frame::acts(self.cfg.epoch, req, pos, &a);
                    self.stats.frames_out.fetch_add(1, Relaxed);
                    self.stats.bytes_out.fetch_add(f.payload.len() as u64, Relaxed);
                    self.down.as_ref().ok_or_else(|| anyhow!("head has no downstream"))?.send_data(f).await?;
                }
                StageOutput::Token { id, eos } => {
                    let token_pos = pos + toks.len() as u32;
                    next = self.on_token(req, token_pos, id, eos);
                }
            }
        }
        Ok(())
    }

    /// Record a generated token; returns the next decode step if the request continues.
    fn on_token(&mut self, req: u64, pos: u32, id: u32, eos: bool) -> Option<(u32, Vec<u32>)> {
        let st = self.reqs.get_mut(&req)?; // cancelled or finished: ignore late tokens
        st.generated += 1;
        st.progress = Instant::now();
        let _ = st.out.send(Event::Token { id, pos });
        let reason = if eos {
            Some(DoneReason::Eos)
        } else if st.generated >= st.max_new {
            Some(DoneReason::MaxTokens)
        } else {
            None
        };
        match reason {
            Some(r) => {
                let _ = st.out.send(Event::Done(r));
                self.finish(req);
                None
            }
            None => Some((pos, vec![id])),
        }
    }

    /// End a request with `Failed(why)` and free it on every stage.
    fn fail(&mut self, req: u64, why: String) {
        if let Some(st) = self.reqs.get(&req) {
            let _ = st.out.send(Event::Done(DoneReason::Failed(why)));
        }
        self.finish(req);
    }

    /// Drop a request everywhere: local state, then a cancel that ripples down the ring.
    fn finish(&mut self, req: u64) {
        self.reqs.remove(&req);
        self.exec.release(req);
        if let Some(d) = &self.down {
            d.send_ctrl(Frame::ctrl(Kind::Cancel, self.cfg.epoch, req, 0));
        }
    }

    /// Start queued requests while there is room in the ring.
    async fn drain_pending(&mut self) -> Result<()> {
        while self.reqs.len() < self.max_active {
            let Some(p) = self.pending.pop_front() else { break };
            self.reqs.insert(p.req, ReqState { generated: 0, max_new: p.max_new, out: p.out, progress: Instant::now() });
            self.forward(p.req, 0, p.prompt).await?;
        }
        Ok(())
    }

    fn fail_all(&mut self, why: &str) {
        for (_, st) in self.reqs.drain() {
            let _ = st.out.send(Event::Done(DoneReason::Failed(why.to_string())));
        }
        for p in self.pending.drain(..) {
            let _ = p.out.send(Event::Done(DoneReason::Failed(why.to_string())));
        }
    }
}

async fn recv_opt(l: &mut Option<Link>) -> Option<Frame> {
    match l {
        Some(l) => l.recv().await,
        None => std::future::pending().await,
    }
}

async fn head_loop<E: StageExecutor>(cfg: StageConfig, exec: E, mut up: Option<Link>, down: Option<Link>, mut cmd_rx: mpsc::UnboundedReceiver<HeadCmd>, stats: Arc<StageStats>) -> Result<()> {
    let max_active = (cfg.window as usize).max(1);
    let mut head = Head { cfg, exec, down, reqs: HashMap::new(), pending: VecDeque::new(), max_active, stats };
    let mut watchdog = tokio::time::interval(Duration::from_secs(1).min(head.cfg.stall_timeout));
    let result: Result<()> = async {
        loop {
            tokio::select! {
                _ = watchdog.tick() => {
                    let limit = head.cfg.stall_timeout;
                    let stalled: Vec<u64> = head.reqs.iter().filter(|(_, st)| st.progress.elapsed() > limit).map(|(r, _)| *r).collect();
                    for r in stalled {
                        head.fail(r, format!("no progress for {} s: a stage may have stalled", limit.as_secs()));
                    }
                    head.drain_pending().await?;
                }
                cmd = cmd_rx.recv() => match cmd {
                    None => return Ok(()),
                    Some(HeadCmd::Submit { req, prompt, max_new, out }) => {
                        if prompt.is_empty() {
                            let _ = out.send(Event::Done(DoneReason::Failed("empty prompt".into())));
                        } else if max_new == 0 {
                            let _ = out.send(Event::Done(DoneReason::MaxTokens));
                        } else {
                            head.pending.push_back(Pending { req, prompt, max_new, out });
                            head.drain_pending().await?;
                        }
                    }
                    Some(HeadCmd::Cancel(req)) => {
                        if let Some(st) = head.reqs.get(&req) {
                            let _ = st.out.send(Event::Done(DoneReason::Cancelled));
                            head.finish(req);
                        } else if let Some(i) = head.pending.iter().position(|p| p.req == req) {
                            if let Some(p) = head.pending.remove(i) {
                                let _ = p.out.send(Event::Done(DoneReason::Cancelled));
                            }
                        }
                        head.drain_pending().await?;
                    }
                },
                frame = recv_opt(&mut up) => {
                    let Some(f) = frame else { bail!("ring broken: tail link closed") };
                    head.stats.frames_in.fetch_add(1, Relaxed);
                    if f.header.epoch != head.cfg.epoch {
                        head.stats.dropped_stale.fetch_add(1, Relaxed);
                        continue;
                    }
                    if f.is_failure() {
                        // A downstream stage's engine failed this request.
                        if head.reqs.contains_key(&f.header.request) {
                            head.fail(f.header.request, f.failure_message());
                        }
                        head.drain_pending().await?;
                    } else if f.header.kind == Kind::Token {
                        let eos = f.header.flags & FLAG_EOS != 0;
                        if let Some((pos, toks)) = head.on_token(f.header.request, f.header.pos, f.header.aux, eos) {
                            head.forward(f.header.request, pos, toks).await?;
                        }
                        head.drain_pending().await?;
                    }
                }
            }
        }
    }
    .await;
    if let Err(e) = &result {
        head.fail_all(&e.to_string());
    }
    result
}

async fn worker_loop<E: StageExecutor>(cfg: StageConfig, mut exec: E, role: Role, mut up: Link, down: Link, stats: Arc<StageStats>) -> Result<()> {
    // Requests cancelled or failed here; late activations for them must not resurrect their KV cache.
    let mut released: HashSet<u64> = HashSet::new();
    fn release<E: StageExecutor>(exec: &mut E, released: &mut HashSet<u64>, req: u64) {
        exec.release(req);
        if released.len() > 65_536 {
            released.clear();
        }
        released.insert(req);
    }
    while let Some(f) = up.recv().await {
        stats.frames_in.fetch_add(1, Relaxed);
        stats.bytes_in.fetch_add(f.payload.len() as u64, Relaxed);
        if f.header.epoch != cfg.epoch {
            stats.dropped_stale.fetch_add(1, Relaxed);
            continue;
        }
        let req = f.header.request;
        match f.header.kind {
            Kind::Activations => {
                if released.contains(&req) {
                    continue;
                }
                let (pos, n_tokens) = (f.header.pos, f.header.n_tokens);
                let acts = f.into_acts();
                let t0 = Instant::now();
                let res = tokio::task::block_in_place(|| exec.forward(req, pos, StageInput::Acts(acts)));
                stats.compute_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
                let frame = match res {
                    Ok(StageOutput::Acts(a)) => Frame::acts(cfg.epoch, req, pos, &a),
                    Ok(StageOutput::Token { id, eos }) => Frame::token(cfg.epoch, req, pos + n_tokens, id, eos),
                    // Fail this request only: free it here and tell the head, via the rest of the ring.
                    Err(e) => {
                        release(&mut exec, &mut released, req);
                        Frame::failed(cfg.epoch, req, pos, &format!("stage {}: {e:#}", cfg.stage_index))
                    }
                };
                stats.frames_out.fetch_add(1, Relaxed);
                stats.bytes_out.fetch_add(frame.payload.len() as u64, Relaxed);
                down.send_data(frame).await?;
            }
            // An upstream stage failed this request: free it and pass the news on toward the head.
            Kind::Token if f.is_failure() => {
                release(&mut exec, &mut released, req);
                down.send_data(f).await?;
            }
            Kind::Cancel => {
                release(&mut exec, &mut released, req);
                if role == Role::Mid {
                    down.send_ctrl(Frame::ctrl(Kind::Cancel, cfg.epoch, req, 0));
                }
            }
            _ => {}
        }
    }
    bail!("upstream link closed")
}
