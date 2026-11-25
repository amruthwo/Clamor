//! Member node: joins a coordinator, runs benchmarks on request, heartbeats, and leaves
//! gracefully on `Drain` or Ctrl-C. Reconnects with backoff if the connection drops.

use crate::stage::{self, Assignment, EngineConfig};
use crate::{share, update};
use crate::wire::{read_msg, write_msg};
use anyhow::{bail, Result};
use clamor_muster::{bench, net, probe};
use clamor_proto::*;
use std::net::TcpListener;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::AbortHandle;
use tokio::time::{interval, sleep, timeout};

/// Progress of this node's pipeline stage, reported back into the session loop.
enum StageEvent {
    Loaded { model: String, epoch: u32, stage_index: u32, load_ms: u64, abort: AbortHandle, exec: stage::ComputeThread },
    Failed { model: String, epoch: u32, stage_index: u32, error: String, stopped: bool },
    /// Fetching the stage's slice from another node.
    Progress { epoch: u32, stage_index: u32, done: u64, total: u64 },
}

/// This node's (at most one) pipeline stage. Dropping it stops the stage.
#[derive(Default)]
struct Slot {
    epoch: Option<u32>,
    loading: Option<AbortHandle>,
    running: Option<AbortHandle>,
    /// The running stage's engine, which private rings run part of.
    exec: Option<stage::ComputeThread>,
    /// Private rings this stage helps run (their middle), by ring id.
    rings: std::collections::HashMap<u32, AbortHandle>,
}
impl Slot {
    fn stop(&mut self) {
        for h in [self.loading.take(), self.running.take()].into_iter().flatten().chain(self.rings.drain().map(|(_, h)| h)) {
            h.abort();
        }
        self.exec = None;
        self.epoch = None;
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.stop();
    }
}

fn spawn_stage(engine: EngineConfig, a: Assignment, tx: UnboundedSender<StageEvent>) -> AbortHandle {
    tokio::spawn(async move {
        let (model, epoch, stage_index) = (a.model.clone(), a.epoch, a.stage_index);
        let ptx = tx.clone();
        let progress = move |done, total| {
            let _ = ptx.send(StageEvent::Progress { epoch, stage_index, done, total });
        };
        match stage::start(&engine, &a, progress).await {
            Ok(s) => {
                let _ = tx.send(StageEvent::Loaded { model: model.clone(), epoch, stage_index, load_ms: s.load_ms, abort: s.abort.clone(), exec: s.exec.clone() });
                let error = match s.task.await {
                    Ok(Ok(())) => "stage stopped".to_string(),
                    Ok(Err(e)) => format!("{e:#}"),
                    Err(e) if e.is_cancelled() => return, // released on purpose
                    Err(e) => e.to_string(),
                };
                let _ = tx.send(StageEvent::Failed { model, epoch, stage_index, error, stopped: true });
            }
            Err(e) => {
                let _ = tx.send(StageEvent::Failed { model, epoch, stage_index, error: format!("{e:#}"), stopped: false });
            }
        }
    })
    .abort_handle()
}

enum Exit {
    Drained,
    Interrupted,
    Rejected(String),
    /// Left the swarm to install an update.
    Update,
}

/// How this device proves itself: its master secret (for per-swarm keys) and, on a first join,
/// the swarm's passphrase.
pub struct Credentials {
    pub device: crate::auth::Device,
    pub passphrase: Option<String>,
    /// A one-time code from "Add device" (instead of the passphrase).
    pub code: Option<String>,
    /// Your name, on a first join (`--name`).
    pub person_name: Option<String>,
    /// Where to serve the swarm's dashboard on this device (normally 127.0.0.1:7421).
    pub dashboard: Option<String>,
    /// Ask to join without a secret: an admin approves it (`--ask`).
    pub ask: bool,
    pub config_dir: std::path::PathBuf,
}

/// `rediscover`: when the coordinator cannot be reached, look for the same swarm on the local
/// network (it may have a new address).
#[allow(clippy::too_many_arguments)]
pub async fn join(mut addr: String, creds: Credentials, id: Option<String>, mut engine: EngineConfig, updates: update::Policy, overrides: crate::neighbour::Overrides, rediscover: bool) -> Result<()> {
    // This process's TLS certificate: neighbours pin it on data links.
    engine.tls = Some(clamor_transport::tls::Identity::generate()?);
    let update_flag: update::UpdateFlag = Default::default();
    if updates.check || updates.auto {
        update::spawn_checker(update_flag.clone());
    }
    // Network-benchmark server the coordinator measures this device against.
    let listener = TcpListener::bind(("0.0.0.0", DEFAULT_BENCH_PORT)).or_else(|_| TcpListener::bind("0.0.0.0:0"))?;
    let bench_port = listener.local_addr()?.port();
    std::thread::spawn(move || {
        let _ = net::serve(listener);
    });
    // Serve our models to the rest of the swarm (layer slices), to the swarm's devices only.
    let allowed: share::AllowedKeys = Default::default();
    // The swarm's dashboard on this device, signed as this device (see dashproxy.rs), and the
    // private chats it keeps here (private.rs).
    let dash: crate::dashproxy::Shared = Default::default();
    let private = crate::private::new(creds.config_dir.clone());
    if let Some(bind) = &creds.dashboard {
        if crate::dashproxy::serve(bind, dash.clone(), private.clone()).await? {
            println!("the swarm's dashboard is at http://{bind}/ on this device");
        } else {
            println!("{bind} is taken: not serving the swarm's dashboard on this device");
        }
    }
    let share_port = share::serve(DEFAULT_SHARE_PORT, engine.models_dir.clone(), allowed.clone()).await?;

    let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
    let mut backoff = 2u64;
    let mut swarm_id: Option<String> = None;
    loop {
        match session(&addr, &creds, &allowed, &dash, &private, &mut swarm_id, &id, bench_port, share_port, &mut backoff, &engine, &update_flag, updates.auto, overrides).await {
            Ok(Exit::Update) => {
                // Returns only if the update failed; then carry on with this version.
                if let Err(e) = update::update_and_restart().await {
                    eprintln!("update failed, staying on this version: {e:#}");
                }
                *update_flag.lock().unwrap() = None;
            }
            Ok(Exit::Drained) => {
                println!("drained: left the swarm. Run `clamor-node join` to rejoin.");
                return Ok(());
            }
            Ok(Exit::Interrupted) => {
                println!("left the swarm.");
                return Ok(());
            }
            Ok(Exit::Rejected(reason)) => {
                eprintln!("coordinator rejected us: {reason}");
                // A coordinator on a newer protocol: catch up and rejoin.
                if updates.auto && reason.contains("protocol") {
                    println!("trying an update to match the coordinator");
                    if let Err(e) = update::update_and_restart().await {
                        eprintln!("update failed: {e:#}");
                    }
                }
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("disconnected: {e}; retrying in {backoff}s");
                if let (true, Some(sid)) = (rediscover, swarm_id.clone()) {
                    let found = tokio::task::spawn_blocking(move || {
                        let found = crate::discovery::browse(Duration::from_secs(2)).unwrap_or_default();
                        found.into_iter().find(|f| f.swarm_id == sid).and_then(|f| f.reachable())
                    })
                    .await
                    .ok()
                    .flatten();
                    if let Some(a) = found.map(|a| a.to_string()).filter(|a| *a != addr) {
                        println!("the swarm is now at {a}");
                        addr = a;
                    }
                }
            }
        }
        sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(30);
    }
}

#[allow(clippy::too_many_arguments)]
async fn session(addr: &str, creds: &Credentials, allowed: &share::AllowedKeys, dash: &crate::dashproxy::Shared, private: &crate::private::Shared, swarm_id: &mut Option<String>, id: &NodeId, bench_port: u16, share_port: u16, backoff: &mut u64, engine: &EngineConfig, update_flag: &update::UpdateFlag, auto_update: bool, overrides: crate::neighbour::Overrides) -> Result<Exit> {
    let scan = |dir: std::path::PathBuf| async move { tokio::task::spawn_blocking(move || share::scan(&dir)).await.unwrap_or_default() };
    let mut models = scan(engine.models_dir.clone()).await;
    let id2 = id.clone();
    let mut info = tokio::task::spawn_blocking(move || probe::probe_node(id2)).await?;
    info.perf_cores = Some(clamor_engine_ggml::default_threads() as u32);
    let class = info.class;

    let stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    // Encrypted. Any certificate is accepted here: the join handshake below signs this TLS
    // session's exporter, which is what proves who is on the other end.
    let connector = tokio_rustls::TlsConnector::from(clamor_transport::tls::client_config(None, None)?);
    let tls = timeout(Duration::from_secs(10), connector.connect(clamor_transport::tls::server_name(), stream)).await??;
    let binding = clamor_transport::tls::exporter_client(tls.get_ref().1)?;
    let (mut rd, mut wr) = tokio::io::split(tls);
    let nonce_m = hex::encode(crate::auth::random_bytes::<16>());
    // A one-time code is good for the first join only; reconnects use this device's key.
    let code = creds.code.as_deref().filter(|_| swarm_id.is_none());
    let asking = creds.ask && swarm_id.is_none() && code.is_none() && creds.passphrase.is_none();
    write_msg(&mut wr, &Message::Hello { proto: PROTOCOL_VERSION, info, bench_port, build: Some(crate::build_info::current()), auth_nonce: Some(nonce_m.clone()), invite: code.is_some(), person_name: creds.person_name.clone(), ask: asking, tls_fp: engine.tls.as_ref().map(|t| t.fingerprint_hex()), models: models.clone(), share_port: Some(share_port) }).await?;

    // The coordinator's challenge: prove we are approved (or know the passphrase), and check
    // that it is the coordinator we joined before (or knows the passphrase).
    let ch = match timeout(Duration::from_secs(10), read_msg(&mut rd)).await?? {
        Message::AuthChallenge { swarm_id, swarm_name, coordinator_key, nonce, spake } => crate::auth::Challenge { swarm_id, swarm_name, coordinator_key, nonce, spake },
        Message::Reject { reason } => return Ok(Exit::Rejected(reason)),
        other => bail!("unexpected reply to hello: {other:?}"),
    };
    let profile = crate::auth::MemberProfile::load(&creds.config_dir, &ch.swarm_id);
    let key = creds.device.swarm_key(&ch.swarm_id);
    let (hs, proof) = match crate::auth::MemberHandshake::answer(&binding, &ch, &nonce_m, &key, profile.as_ref().map(|p| p.coordinator_key.as_str()), code.or(creds.passphrase.as_deref()), asking) {
        Ok(v) => v,
        Err(e) => {
            let _ = write_msg(&mut wr, &Message::Leave).await;
            return Ok(Exit::Rejected(format!("{e:#}")));
        }
    };
    if asking && profile.is_none() {
        let code = crate::auth::check_code(&binding, &ch.coordinator_key, &proof.device_key);
        println!("asking to join swarm '{}'. Check code: {code}", ch.swarm_name);
        println!("an admin approves this in the swarm's dashboard; make sure it shows the same code.");
    }
    write_msg(&mut wr, &Message::AuthProve { device_key: proof.device_key, signature: proof.signature, spake: proof.spake, confirm: proof.confirm }).await?;

    // Asking to join can take a while: someone has to approve it.
    let wait = if asking { crate::coordinator::ASK_SECS + 30 } else { 30 };
    let heartbeat_secs = match timeout(Duration::from_secs(wait), read_msg(&mut rd)).await?? {
        Message::Welcome { swarm, coordinator, heartbeat_secs, auth } => {
            let Some(auth) = auth else { bail!("the coordinator did not prove who it is") };
            if let Err(e) = hs.finish(&crate::auth::CoordinatorProof { signature: auth.signature, confirm: auth.confirm }) {
                return Ok(Exit::Rejected(format!("{e:#}")));
            }
            set_allowed(allowed, &auth.approved_keys, &ch.coordinator_key);
            if let (Some(port), Some(fp)) = (auth.portal_tls_port, auth.portal_tls_fp.as_deref().and_then(|f| clamor_transport::tls::parse_fingerprint(f).ok())) {
                let host = addr.rsplit_once(':').map(|(h, _)| h).unwrap_or(addr).trim_matches(['[', ']']).to_string();
                *dash.write().unwrap() = Some(crate::dashproxy::Target { host, port, fingerprint: fp, key: key.clone() });
            }
            *swarm_id = Some(ch.swarm_id.clone());
            // Remember the swarm: its coordinator's key, and where it was reached.
            let mut p = profile.unwrap_or_else(|| crate::auth::MemberProfile { swarm_id: ch.swarm_id.clone(), name: ch.swarm_name.clone(), coordinator_key: String::new(), addresses: Vec::new(), last_joined_unix: 0 });
            let first = p.coordinator_key.is_empty();
            p.name = ch.swarm_name.clone();
            p.coordinator_key = ch.coordinator_key.clone();
            p.addresses.retain(|a| a != addr);
            p.addresses.insert(0, addr.to_string());
            p.addresses.truncate(8);
            p.last_joined_unix = unix_now();
            if let Err(e) = p.save(&creds.config_dir) {
                eprintln!("could not remember this swarm: {e:#}");
            }
            println!("joined swarm '{swarm}' (coordinator {coordinator}){}", if first { "; remembered, so next time `clamor-node join` is enough" } else { "" });
            *backoff = 2;
            heartbeat_secs
        }
        Message::Reject { reason } => return Ok(Exit::Rejected(reason)),
        other => bail!("unexpected reply to auth: {other:?}"),
    };
    // Fetch slices from other devices as ourselves.
    let mut engine = engine.clone();
    engine.share_key = Some(key);
    engine.swarm_id = Some(ch.swarm_id.clone());
    let engine = &engine;

    // Reader runs in its own task so the select loop below never cancels a half-read frame.
    let (mtx, mut mrx) = unbounded_channel();
    tokio::spawn(async move {
        loop {
            let m = read_msg(&mut rd).await;
            let failed = m.is_err();
            if mtx.send(m).is_err() || failed {
                break;
            }
        }
    });

    // The device's own dashboard asks for private rings through this.
    let (ctl_tx, mut ctl_rx) = unbounded_channel::<Message>();
    {
        let mut p = private.lock().unwrap();
        p.control = Some(ctl_tx);
        p.swarm(&ch.swarm_id);
    }
    let mut hb = interval(Duration::from_secs(heartbeat_secs.max(1) as u64));
    // Notice models added to or removed from --models.
    let mut rescan = interval(Duration::from_secs(30));
    rescan.tick().await;
    let mut sampler = probe::LoadSampler::new();
    // Until the coordinator sends this device's stored policy (right after Welcome).
    // This swarm's settings, as the owner wants them; the coordinator answers with SetPolicy.
    let sid = ch.swarm_id.clone();
    let owner_policy = overrides.for_swarm(&sid, class);
    write_msg(&mut wr, &Message::OwnerPolicy { policy: owner_policy }).await?;
    let mut policy = owner_policy.unwrap_or_else(|| Policy::default_for(class));
    let mut pause = crate::neighbour::PauseState::default();
    let mut paused_reason: Option<String> = None;
    // Something the owner should know about the last settings change (e.g. needs a restart).
    let settings_note: Option<String> = None;
    let (stx, mut srx) = unbounded_channel::<StageEvent>();
    let (btx, mut brx) = unbounded_channel::<BenchReport>();
    let mut slot = Slot::default();

    loop {
        tokio::select! {
            _ = rescan.tick() => {
                let now = scan(engine.models_dir.clone()).await;
                if now != models {
                    models = now;
                    write_msg(&mut wr, &Message::Inventory { models: models.clone() }).await?;
                }
            }
            Some(report) = brx.recv() => {
                write_msg(&mut wr, &Message::BenchResult { report }).await?;
            }
            Some(ev) = srx.recv() => match ev {
                StageEvent::Progress { epoch, stage_index, done, total } => {
                    if slot.epoch == Some(epoch) {
                        write_msg(&mut wr, &Message::ShardProgress { epoch, stage_index, done_bytes: done, total_bytes: total }).await?;
                    }
                }
                StageEvent::Loaded { model, epoch, stage_index, load_ms, abort, exec } => {
                    if slot.epoch == Some(epoch) {
                        println!("stage {stage_index} of {model} ready (epoch {epoch}, loaded in {load_ms} ms)");
                        slot.loading = None;
                        slot.running = Some(abort);
                        slot.exec = Some(exec);
                        write_msg(&mut wr, &Message::ShardReady { model, epoch, stage_index, load_ms }).await?;
                    } else {
                        abort.abort(); // superseded or released while loading
                    }
                }
                StageEvent::Failed { model, epoch, stage_index, error, stopped } => {
                    if slot.epoch == Some(epoch) {
                        eprintln!("stage {stage_index} of {model} failed: {error}");
                        slot.stop();
                        write_msg(&mut wr, &Message::ShardFailed { model, epoch, stage_index, error, stopped }).await?;
                    }
                }
            },
            _ = hb.tick() => {
                let load = sampler.sample_load();
                let power = probe::read_power();
                let now = pause.update(&policy, &load, power.as_ref());
                if now != paused_reason {
                    match &now {
                        Some(why) => println!("not taking new work: {why}"),
                        None => println!("taking work again"),
                    }
                    paused_reason = now.clone();
                }
                let (cpu, ram) = (load.total, load.ram_available);
                let update_available = update_flag.lock().unwrap().clone();
                // Update only between pipelines, never under a running stage.
                if auto_update && update_available.is_some() && slot.epoch.is_none() {
                    println!("leaving the swarm to install clamor {}", update_available.unwrap_or_default());
                    let _ = write_msg(&mut wr, &Message::Leave).await;
                    return Ok(Exit::Update);
                }
                write_msg(&mut wr, &Message::Heartbeat { cpu_load: cpu, ram_available_bytes: ram, power, running_tasks: 0, update_available, paused: now, note: settings_note.clone() }).await?;
            }
            Some(m) = ctl_rx.recv() => {
                write_msg(&mut wr, &m).await?;
            }
            _ = tokio::signal::ctrl_c() => {
                let _ = write_msg(&mut wr, &Message::Leave).await;
                return Ok(Exit::Interrupted);
            }
            m = mrx.recv() => {
                let msg = match m {
                    None => bail!("connection closed"),
                    Some(Err(e)) => return Err(e.into()),
                    Some(Ok(msg)) => msg,
                };
                match msg {
                    Message::BenchRequest { quick } => {
                        // Off this loop: the standard benchmark takes about a minute, and
                        // heartbeats must keep going meanwhile (the coordinator gives up on a
                        // device after 30 s of silence).
                        println!("benchmarking at the coordinator's request...");
                        let (_, ram) = sampler.sample();
                        let opts = if quick { bench::BenchOptions::quick() } else { bench::BenchOptions::standard() };
                        let btx = btx.clone();
                        tokio::task::spawn_blocking(move || {
                            // Test hook: make the benchmark outlast the heartbeat timeout.
                            if let Some(s) = std::env::var("CLAMOR_TEST_BENCH_DELAY_SECS").ok().and_then(|s| s.parse().ok()) {
                                std::thread::sleep(Duration::from_secs(s));
                            }
                            let _ = btx.send(bench::run(&opts, ram));
                        });
                    }
                    Message::SetPolicy { policy: p } => {
                        println!("settings updated: {}", describe_policy(&p));
                        crate::neighbour::save_settings(&sid, &p);
                        policy = p;
                    }
                    Message::Approved { keys } => set_allowed(allowed, &keys, &ch.coordinator_key),
                    Message::Ping { seq, sent_unix_ms } => {
                        write_msg(&mut wr, &Message::Pong { seq, sent_unix_ms }).await?;
                    }
                    Message::ShardAssign { model, epoch, session, stage_index, stage_count, layer_start, layer_end, listen_port, downstream, act_dtype, n_ctx, embed, sampling, source, model_id, max_requests, need_bytes, upstream_tls, downstream_tls } => {
                        println!("assigned stage {stage_index}/{stage_count}: layers {layer_start}..{layer_end} of {model} (epoch {epoch})");
                        slot.stop(); // a new epoch supersedes whatever ran before
                        if let Some(src) = &source {
                            println!("  fetching the slice from {src}");
                        }
                        let a = Assignment { model, epoch, session, stage_index, stage_count, layer_start, layer_end, listen_port, downstream, act_dtype, n_ctx, embed, sampling, source, model_id, max_requests, need_bytes, upstream_tls, downstream_tls, policy };
                        slot.epoch = Some(epoch);
                        slot.loading = Some(spawn_stage(engine.clone(), a, stx.clone()));
                    }
                    Message::RingJoin { ring, epoch, session, stage_index, stage_count, layers, listen_port, downstream, upstream_tls, downstream_tls } => {
                        // Part of the stage this device already runs, in the middle of someone's
                        // private ring: no loading.
                        match (&slot.exec, slot.epoch == Some(epoch)) {
                            (Some(exec), true) => {
                                let part = stage::RingPart { ring, epoch, session, stage_index, stage_count, layers: layers[0]..layers[1], listen_port, downstream, upstream_tls, downstream_tls };
                                let (engine, exec) = (engine.clone(), exec.clone());
                                let h = tokio::spawn(async move {
                                    match stage::start_part(&engine, &exec, part).await {
                                        Ok(h) => {
                                            let _ = h.task.await;
                                        }
                                        Err(e) => eprintln!("private ring {ring}: {e:#}"),
                                    }
                                });
                                println!("helping run a private chat (ring {ring}, layers {}..{})", layers[0], layers[1]);
                                if let Some(old) = slot.rings.insert(ring, h.abort_handle()) {
                                    old.abort();
                                }
                            }
                            _ => eprintln!("asked to join private ring {ring}, but this device runs no stage of epoch {epoch}"),
                        }
                    }
                    Message::RingLeave { ring } => {
                        if let Some(h) = slot.rings.remove(&ring) {
                            h.abort();
                        }
                    }
                    Message::PrivateAssign { ring, model, model_id, source, epoch, session, n_ctx, act_dtype, sampling, stage_count, head_layers, tail_layers, head_port, tail_port, head_downstream, head_downstream_tls, tail_upstream_tls } => {
                        println!("setting up private chats: loading this device's ends of {model}");
                        {
                            let mut p = private.lock().unwrap();
                            p.stop();
                            p.ring = Some(ring);
                            p.state = crate::private::State::Starting;
                        }
                        let ends = crate::private::Ends { ring, model, model_id, source, epoch, session, n_ctx, act_dtype, sampling, stage_count, head_layers, tail_layers, head_port, tail_port, head_downstream, head_downstream_tls, tail_upstream_tls };
                        crate::private::start_ends(engine.clone(), policy, ends, private.clone());
                    }
                    Message::PrivateFailed { ring, message } => {
                        let mut p = private.lock().unwrap();
                        if ring.is_none() || p.ring == ring {
                            p.stop();
                            p.state = crate::private::State::Failed(message);
                        }
                    }
                    Message::ShardRelease { model, epoch } => {
                        if slot.epoch.is_some_and(|e| e <= epoch) {
                            println!("released {model} (epoch {epoch})");
                            slot.stop();
                        }
                    }
                    Message::Drain { grace_secs } => {
                        slot.stop();
                        println!("drain requested (grace {grace_secs}s); no work in flight, leaving");
                        // Leave even if the coordinator already closed the connection (a
                        // removed device's acknowledgement can arrive after it is gone).
                        let _ = write_msg(&mut wr, &Message::Drained).await;
                        return Ok(Exit::Drained);
                    }
                    Message::TaskAssign { task } => {
                        let output = serde_json::json!({ "error": "task execution is not implemented in v0.1" });
                        write_msg(&mut wr, &Message::TaskResult { task_id: task.id, ok: false, output }).await?;
                    }
                    other => eprintln!("ignoring {other:?} (policy {:?})", policy.participation),
                }
            }
        }
    }
}

/// Devices this node's share server answers: the swarm's approved devices and its coordinator.
fn set_allowed(allowed: &share::AllowedKeys, keys: &[String], coordinator_key: &str) {
    let mut a = allowed.write().unwrap();
    a.clear();
    a.extend(keys.iter().cloned());
    a.insert(coordinator_key.to_string());
}

fn describe_policy(p: &Policy) -> String {
    format!(
        "{:?}, {} battery, RAM {:.0}%, CPU {:.0}%, {} priority",
        p.participation,
        if p.require_plugged_in { "not on" } else { "also on" },
        p.max_ram_fraction * 100.0,
        p.max_cpu_fraction * 100.0,
        if p.low_priority { "low" } else { "normal" }
    )
}
