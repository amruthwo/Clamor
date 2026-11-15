//! Coordinator: accepts nodes, keeps the roster, runs the muster on each join, and reaps
//! nodes that stop sending heartbeats. A departed node stays in the roster as `Away` with its
//! benchmark results, so rejoining is instant.

use crate::pipeline::Pipeline;
use crate::stage::EngineConfig;
use crate::wire::{read_msg, write_msg};
use anyhow::Result;
use clamor_muster::{bench, net, probe};
use clamor_proto::*;
use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::time::{sleep, timeout};

pub const HEARTBEAT_SECS: u32 = 5;

pub struct NodeEntry {
    pub report: NodeReport,
    pub state: NodeState,
    pub last_seen: u64,
    pub tx: Option<UnboundedSender<Message>>,
    pub cpu_load: Option<f32>,
    pub running_tasks: u32,
    pub note: Option<String>,
    /// Bumped on every (re)join so a stale connection cannot overwrite a newer one.
    pub generation: u64,
    /// The node's address, as seen on its control connection.
    pub peer_ip: Option<IpAddr>,
    /// The coordinator's own address on that connection: how this node reaches us.
    pub coord_ip: Option<IpAddr>,
    /// The node's binary (None for nodes too old to say).
    pub build: Option<BuildInfo>,
    /// A newer release the node could update to.
    pub update_available: Option<String>,
    /// Models the node holds, and the port it shares them on.
    pub models: Vec<ModelInfo>,
    pub share_port: Option<u16>,
    /// Why the node is not taking new work (on battery, owner busy, paused), if it isn't.
    pub paused: Option<String>,
    /// About the last settings change (e.g. it needs a restart to take effect).
    pub settings_note: Option<String>,
}

pub struct Swarm {
    pub name: String,
    pub token: Option<String>,
    pub coordinator: NodeId,
    pub nodes: BTreeMap<NodeId, NodeEntry>,
    next_generation: u64,
    pub engine: EngineConfig,
    /// Stage `i` of a pipeline listens on `data_port + i`.
    pub data_port: u16,
    /// Ask joining nodes for the quick benchmark instead of the standard one.
    pub quick_bench: bool,
    pub pipeline: Option<Pipeline>,
    pub next_epoch: u32,
    /// Port the coordinator shares its own models on.
    pub share_port: Option<u16>,
}

impl Swarm {
    pub fn new(name: String, token: Option<String>, coordinator: NodeId, engine: EngineConfig, data_port: u16, quick_bench: bool, share_port: Option<u16>) -> Self {
        Swarm { name, token, coordinator, nodes: BTreeMap::new(), next_generation: 1, engine, data_port, quick_bench, pipeline: None, next_epoch: 1, share_port }
    }

    /// A node that held a pipeline stage is gone: that pipeline cannot run any more.
    fn node_lost(&mut self, id: &NodeId, why: &str) {
        if let Some(p) = self.pipeline.as_mut() {
            let epoch = p.epoch;
            if let Some(i) = p.stages.iter().position(|s| &s.node == id) {
                p.mark_failed(epoch, i as u32, Some(id), why.to_string(), true);
            }
        }
    }
}

pub type Shared = Arc<Mutex<Swarm>>;

/// Add the coordinator's own hardware to the roster so it takes part in planning.
pub async fn register_self(shared: Shared, id: NodeId, owner_policy: Option<Policy>) {
    let models_dir = shared.lock().unwrap().engine.models_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut info = probe::probe_node(id);
        info.perf_cores = Some(clamor_engine_ggml::default_threads() as u32);
        let report = bench::run(&bench::BenchOptions::quick(), info.ram_available_bytes);
        (info, report, crate::share::scan(&models_dir))
    })
    .await;
    let Ok((info, bench_report, models)) = result else { return };
    let mut sw = shared.lock().unwrap();
    let share_port = sw.share_port;
    let policy = owner_policy.unwrap_or_else(|| Policy::default_for(info.class));
    let generation = sw.next_generation;
    sw.next_generation += 1;
    let id = info.id.clone();
    sw.nodes.insert(
        id,
        NodeEntry {
            report: NodeReport { info, policy, bench: Some(bench_report), link: None },
            state: NodeState::Ready,
            last_seen: unix_now(),
            tx: None,
            cpu_load: None,
            running_tasks: 0,
            note: Some("coordinator".into()),
            generation,
            peer_ip: None,
            coord_ip: None,
            build: Some(crate::build_info::current()),
            update_available: None,
            models,
            share_port,
            paused: None,
            settings_note: crate::neighbour::set_priority(policy.low_priority).err(),
        },
    );
}

/// The coordinator's own device: keep its load, memory, power and pause state current, as a
/// member's heartbeats do for theirs.
pub async fn monitor_self(shared: Shared) {
    let mut sampler = probe::LoadSampler::new();
    let mut pause = crate::neighbour::PauseState::default();
    loop {
        sleep(Duration::from_secs(HEARTBEAT_SECS as u64)).await;
        let load = sampler.sample_load();
        let power = probe::read_power();
        let mut sw = shared.lock().unwrap();
        let me = sw.coordinator.clone();
        let Some(e) = sw.nodes.get_mut(&me) else { continue };
        let paused = pause.update(&e.report.policy, &load, power.as_ref());
        if paused != e.paused {
            match &paused {
                Some(why) => println!("coordinator device not taking new work: {why}"),
                None => println!("coordinator device taking work again"),
            }
        }
        e.paused = paused;
        e.cpu_load = Some(load.total);
        e.report.info.ram_available_bytes = load.ram_available;
        e.report.info.power = power;
        e.last_seen = unix_now();
    }
}

pub async fn run_listener(bind: &str, shared: Shared) -> Result<()> {
    let listener = TcpListener::bind(bind).await?;
    println!("control plane listening on {bind}");
    loop {
        let (stream, addr) = listener.accept().await?;
        let shared = shared.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(stream, addr, shared).await {
                eprintln!("connection {addr}: {e}");
            }
        });
    }
}

enum Admission {
    Ok { generation: u64, swarm: String, coordinator: NodeId, cached_bench: bool, quick: bool },
    Rejected(String),
}

async fn handle_conn(stream: TcpStream, addr: SocketAddr, shared: Shared) -> Result<()> {
    stream.set_nodelay(true)?;
    let coord_ip = stream.local_addr().ok().map(|a| a.ip());
    let (mut rd, mut wr) = stream.into_split();

    let hello = timeout(Duration::from_secs(10), read_msg(&mut rd)).await??;
    let Message::Hello { proto, info, bench_port, join_token, build, models, share_port, policy: owner_policy } = hello else {
        write_msg(&mut wr, &Message::Reject { reason: "expected hello".into() }).await?;
        return Ok(());
    };
    if proto != PROTOCOL_VERSION {
        write_msg(&mut wr, &Message::Reject { reason: format!("protocol {proto} not supported (coordinator speaks {PROTOCOL_VERSION})") }).await?;
        return Ok(());
    }

    let (tx, mut rx) = unbounded_channel::<Message>();
    let id = info.id.clone();
    let admission = {
        let mut sw = shared.lock().unwrap();
        if sw.token.is_some() && sw.token != join_token {
            Admission::Rejected("bad join token".into())
        } else if id == sw.coordinator {
            Admission::Rejected("that id belongs to the coordinator".into())
        } else if sw.nodes.get(&id).map(|e| e.tx.is_some()).unwrap_or(false) {
            Admission::Rejected(format!("{id} is already connected"))
        } else {
            let generation = sw.next_generation;
            sw.next_generation += 1;
            let (swarm, coordinator) = (sw.name.clone(), sw.coordinator.clone());
            let previous = sw.nodes.remove(&id);
            let (policy, bench, link) = match previous {
                Some(p) => (p.report.policy, p.report.bench, p.report.link),
                None => (Policy::default_for(info.class), None, None),
            };
            // The device owner's own settings win.
            let policy = owner_policy.unwrap_or(policy);
            let cached = bench.is_some();
            sw.nodes.insert(
                id.clone(),
                NodeEntry {
                    report: NodeReport { info, policy, bench, link },
                    state: if cached { NodeState::Ready } else { NodeState::Joining },
                    last_seen: unix_now(),
                    tx: Some(tx.clone()),
                    cpu_load: None,
                    running_tasks: 0,
                    note: if cached { Some("rejoined with cached benchmark".into()) } else { None },
                    generation,
                    peer_ip: Some(addr.ip()),
                    coord_ip,
                    build,
                    update_available: None,
                    models,
                    share_port,
                    paused: None,
                    settings_note: None,
                },
            );
            Admission::Ok { generation, swarm, coordinator, cached_bench: cached, quick: sw.quick_bench }
        }
    };
    let (generation, swarm, coordinator, cached_bench, quick) = match admission {
        Admission::Ok { generation, swarm, coordinator, cached_bench, quick } => (generation, swarm, coordinator, cached_bench, quick),
        Admission::Rejected(reason) => {
            write_msg(&mut wr, &Message::Reject { reason }).await?;
            return Ok(());
        }
    };
    println!("{id} joined from {addr}{}", if cached_bench { " (cached benchmark)" } else { "" });

    let _ = tx.send(Message::Welcome { swarm, coordinator, heartbeat_secs: HEARTBEAT_SECS });
    // The node starts from its class defaults; send the policy we hold for it (it may have been
    // changed in the dashboard, and survives the node reconnecting).
    if let Some(policy) = shared.lock().unwrap().nodes.get(&id).map(|e| e.report.policy) {
        let _ = tx.send(Message::SetPolicy { policy });
    }
    if !cached_bench {
        set_state(&shared, &id, generation, NodeState::Benchmarking);
        let _ = tx.send(Message::BenchRequest { quick });
    }
    let writer = tokio::spawn(async move {
        while let Some(m) = rx.recv().await {
            if write_msg(&mut wr, &m).await.is_err() {
                break;
            }
        }
    });

    let mut leave_note = "connection lost".to_string();
    while let Ok(msg) = read_msg(&mut rd).await {
        {
            let mut sw = shared.lock().unwrap();
            match sw.nodes.get_mut(&id) {
                Some(e) if e.generation == generation => e.last_seen = unix_now(),
                _ => break, // replaced by a newer connection
            }
        }
        match msg {
            Message::Heartbeat { cpu_load, ram_available_bytes, power, running_tasks, update_available, paused, note } => {
                let mut sw = shared.lock().unwrap();
                if let Some(e) = sw.nodes.get_mut(&id) {
                    e.update_available = update_available;
                    e.settings_note = note;
                    e.paused = paused;
                    e.cpu_load = Some(cpu_load);
                    e.running_tasks = running_tasks;
                    e.report.info.ram_available_bytes = ram_available_bytes;
                    if power.is_some() {
                        e.report.info.power = power;
                    }
                }
            }
            Message::BenchResult { report } => {
                {
                    let mut sw = shared.lock().unwrap();
                    if let Some(e) = sw.nodes.get_mut(&id) {
                        e.report.bench = Some(report);
                    }
                }
                // Link measurement blocks on sockets, so keep it off the async threads.
                let (shared2, id2, ip) = (shared.clone(), id.clone(), addr.ip());
                tokio::task::spawn_blocking(move || {
                    let link = net::measure((ip, bench_port), 40, 32 << 20).ok();
                    let mut sw = shared2.lock().unwrap();
                    if let Some(e) = sw.nodes.get_mut(&id2) {
                        if e.generation == generation {
                            e.report.link = link;
                            if matches!(e.state, NodeState::Joining | NodeState::Benchmarking) {
                                e.state = NodeState::Ready;
                            }
                        }
                    }
                });
            }
            Message::Drained => {
                leave_note = "drained (left gracefully)".into();
                break;
            }
            Message::Leave => {
                leave_note = "left at the owner's request".into();
                break;
            }
            Message::Inventory { models } => {
                if let Some(e) = shared.lock().unwrap().nodes.get_mut(&id) {
                    e.models = models;
                }
            }
            Message::ShardProgress { epoch, stage_index, done_bytes, total_bytes } => {
                if let Some(p) = shared.lock().unwrap().pipeline.as_mut() {
                    p.mark_progress(epoch, stage_index, Some(&id), done_bytes, total_bytes);
                }
            }
            Message::ShardReady { epoch, stage_index, load_ms, .. } => {
                println!("{id}: stage {stage_index} ready (epoch {epoch}, loaded in {load_ms} ms)");
                if let Some(p) = shared.lock().unwrap().pipeline.as_mut() {
                    p.mark_ready(epoch, stage_index, Some(&id), load_ms);
                }
            }
            Message::ShardFailed { epoch, stage_index, error, stopped, .. } => {
                eprintln!("{id}: stage {stage_index} failed (epoch {epoch}): {error}");
                if let Some(p) = shared.lock().unwrap().pipeline.as_mut() {
                    p.mark_failed(epoch, stage_index, Some(&id), error, !stopped);
                }
            }
            Message::TaskUpdate { .. } | Message::TaskResult { .. } => {
                // Task bookkeeping lands with the scheduler.
            }
            _ => {}
        }
    }

    {
        let mut sw = shared.lock().unwrap();
        let mut lost = false;
        if let Some(e) = sw.nodes.get_mut(&id) {
            if e.generation == generation {
                e.state = NodeState::Away;
                e.tx = None;
                e.cpu_load = None;
                e.paused = None;
                e.note = Some(leave_note.clone());
                lost = true;
            }
        }
        if lost {
            sw.node_lost(&id, &leave_note);
        }
    }
    writer.abort();
    println!("{id}: {leave_note}");
    Ok(())
}

fn set_state(shared: &Shared, id: &NodeId, generation: u64, state: NodeState) {
    let mut sw = shared.lock().unwrap();
    if let Some(e) = sw.nodes.get_mut(id) {
        if e.generation == generation {
            e.state = state;
        }
    }
}

/// The coordinator's own update checks: show them in its roster entry and, with
/// `--auto-update`, install when no pipeline is running (members reconnect on their own).
pub async fn updates(shared: Shared, policy: crate::update::Policy) {
    if !(policy.check || policy.auto) {
        return;
    }
    let flag: crate::update::UpdateFlag = Default::default();
    crate::update::spawn_checker(flag.clone());
    loop {
        sleep(Duration::from_secs(300)).await;
        let available = flag.lock().unwrap().clone();
        let idle = {
            let mut sw = shared.lock().unwrap();
            let me = sw.coordinator.clone();
            if let Some(e) = sw.nodes.get_mut(&me) {
                e.update_available = available.clone();
            }
            sw.pipeline.is_none()
        };
        if policy.auto && available.is_some() && idle {
            if let Err(e) = crate::update::update_and_restart().await {
                eprintln!("update failed, staying on this version: {e:#}");
                *flag.lock().unwrap() = None;
            }
        }
    }
}

/// Mark nodes that stopped heartbeating as away.
pub async fn reaper(shared: Shared) {
    loop {
        sleep(Duration::from_secs(5)).await;
        let now = unix_now();
        let mut sw = shared.lock().unwrap();
        let mut lost = Vec::new();
        for (id, e) in sw.nodes.iter_mut() {
            if e.tx.is_some() && now.saturating_sub(e.last_seen) > 3 * HEARTBEAT_SECS as u64 {
                e.tx = None;
                e.state = NodeState::Away;
                e.note = Some("heartbeat timeout".into());
                println!("{id}: heartbeat timeout");
                lost.push(id.clone());
            }
        }
        for id in lost {
            sw.node_lost(&id, "heartbeat timeout");
        }
    }
}
