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
    /// The node's TLS certificate fingerprint (hex): its neighbours pin it on data links.
    pub tls_fp: Option<String>,
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
    /// The device's key in this swarm (None for the coordinator itself).
    pub key: Option<String>,
}

/// Who may join: the swarm's profile (id, passphrase, approved device keys) and the
/// coordinator's own key in this swarm.
pub struct SwarmAuth {
    pub profile: crate::auth::CoordinatorProfile,
    pub key: ed25519_dalek::SigningKey,
    pub config_dir: std::path::PathBuf,
    /// Keys the share servers accept (approved devices and the coordinator).
    pub allowed: crate::share::AllowedKeys,
    /// Browser sessions for the dashboard.
    pub sessions: crate::people::Sessions,
    pub pins: crate::people::PinGuard,
    /// Browsers waiting for a signed-in person to approve their sign-in code.
    pub signins: Vec<crate::people::PendingSignIn>,
}

impl SwarmAuth {
    pub fn new(profile: crate::auth::CoordinatorProfile, key: ed25519_dalek::SigningKey, config_dir: std::path::PathBuf) -> Self {
        let allowed: crate::share::AllowedKeys = Default::default();
        let sessions = crate::people::Sessions::load(&config_dir, &profile.swarm_id);
        let me = SwarmAuth { profile, key, config_dir, allowed, sessions, pins: Default::default(), signins: Vec::new() };
        me.refresh_allowed();
        me
    }
    pub fn approved_keys(&self) -> Vec<String> {
        self.profile.approved.iter().map(|d| d.key.clone()).collect()
    }
    fn refresh_allowed(&self) {
        let mut a = self.allowed.write().unwrap();
        a.clear();
        a.extend(self.approved_keys());
        a.insert(crate::auth::key_hex(&self.key.verifying_key()));
    }
    /// Withdraw a device's approval: it needs the passphrase to join again. Returns whether
    /// it was approved.
    pub fn revoke(&mut self, key: &str) -> bool {
        let before = self.profile.approved.len();
        self.profile.approved.retain(|d| d.key != key);
        let removed = self.profile.approved.len() != before;
        if removed {
            if let Err(e) = self.profile.save(&self.config_dir) {
                eprintln!("could not save the swarm profile: {e:#}");
            }
            self.refresh_allowed();
        }
        removed
    }

    /// Approve a device that proved it knows the passphrase (or a code). It belongs to
    /// `person` ("Add my device"), or else to a new member called `person_name`.
    fn approve(&mut self, key: &str, name: &str, person: Option<&str>, person_name: Option<&str>) {
        if self.profile.approved.iter().any(|d| d.key == key) {
            return;
        }
        let person = match person.filter(|p| self.profile.person(p).is_some()) {
            Some(p) => p.to_string(),
            None => {
                let id = crate::people::new_person_id();
                let pname = person_name.map(str::trim).filter(|n| !n.is_empty()).unwrap_or(name);
                self.profile.people.push(crate::people::Person { id: id.clone(), name: pname.chars().take(40).collect(), role: crate::people::Role::Member, pin: None, created_unix: unix_now(), reply_style: None });
                id
            }
        };
        self.profile.approved.push(crate::auth::ApprovedDevice { key: key.into(), name: name.into(), approved_unix: unix_now(), person });
        self.save();
        self.refresh_allowed();
    }

    pub fn save(&self) {
        if let Err(e) = self.profile.save(&self.config_dir) {
            eprintln!("could not save the swarm profile: {e:#}");
        }
    }

    /// Remove a person: their devices lose their approval, their sessions end. Returns the
    /// device keys revoked.
    pub fn remove_person(&mut self, id: &str) -> Vec<String> {
        let keys: Vec<String> = self.profile.approved.iter().filter(|d| d.person == id).map(|d| d.key.clone()).collect();
        self.profile.approved.retain(|d| d.person != id);
        self.profile.people.retain(|p| p.id != id);
        self.sessions.end_all(id, &self.config_dir, &self.profile.swarm_id);
        self.save();
        self.refresh_allowed();
        keys
    }
}

/// A one-time code for adding a device ("Add device" in the dashboard).
pub struct Invite {
    pub code: String,
    pub expires_unix: u64,
    pub failures: u32,
    /// "Add my device": the device joins as this person. None: a new person.
    pub person: Option<String>,
}

/// A device asking to join, waiting for an admin.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PendingJoin {
    pub id: String,
    pub device: String,
    pub person_name: String,
    pub from: String,
    /// The check code both screens show.
    pub code: String,
    pub created_unix: u64,
    /// None: waiting; Some(true): approved; Some(false): declined.
    pub decision: Option<bool>,
}

/// How long a device asking to join waits for approval.
pub const ASK_SECS: u64 = 600;

/// How long a one-time code works, and how many wrong tries it survives.
pub const INVITE_SECS: u64 = 600;
pub const INVITE_TRIES: u32 = 3;

impl Invite {
    pub fn new() -> Self {
        let n = u32::from_le_bytes(crate::auth::random_bytes::<4>()) % 1_000_000;
        Invite { code: format!("{:03}-{:03}", n / 1000, n % 1000), expires_unix: unix_now() + INVITE_SECS, failures: 0, person: None }
    }
}

pub struct Swarm {
    /// This process's TLS certificate (control connections, and stage 0's data links).
    pub tls: Arc<clamor_transport::tls::Identity>,
    tls_server: Arc<tokio_rustls::rustls::ServerConfig>,
    pub name: String,
    pub auth: SwarmAuth,
    /// The one-time code in force, if any.
    pub invite: Option<Invite>,
    /// How devices reach the coordinator (`host` or `host:port`), for join commands.
    pub join_hint: String,
    /// Port of the encrypted (HTTPS) dashboard, once listening.
    pub portal_tls_port: Option<u16>,
    /// Devices asking to join.
    pub joins: Vec<PendingJoin>,
    /// Conversations (personal and group).
    pub chats: crate::chats::Chats,
    /// Conversations the model is answering right now.
    pub generating: std::collections::HashMap<String, crate::generate::Generation>,
    /// Recent replies: (unix time, tokens per second), newest last.
    pub speeds: std::collections::VecDeque<(u64, f32)>,
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
    pub fn new(name: String, auth: SwarmAuth, coordinator: NodeId, mut engine: EngineConfig, data_port: u16, quick_bench: bool, share_port: Option<u16>) -> Self {
        let tls = coordinator_identity(&auth.config_dir, &auth.profile.swarm_id);
        let chats = crate::chats::Chats::load(&auth.config_dir, &auth.profile.swarm_id);
        let tls_server = clamor_transport::tls::server_config(&tls, None).expect("TLS config");
        engine.tls = Some(tls.clone());
        Swarm {
            tls,
            tls_server, name, auth, invite: None, join_hint: String::new(), portal_tls_port: None, joins: Vec::new(), chats, generating: Default::default(), speeds: Default::default(), coordinator, nodes: BTreeMap::new(), next_generation: 1, engine, data_port, quick_bench, pipeline: None, next_epoch: 1, share_port }
    }

    /// A node that held a pipeline stage is gone: that pipeline cannot run any more.
    pub fn node_lost(&mut self, id: &NodeId, why: &str) {
        if let Some(p) = self.pipeline.as_mut() {
            let epoch = p.epoch;
            if let Some(i) = p.stages.iter().position(|s| &s.node == id) {
                p.mark_failed(epoch, i as u32, Some(id), why.to_string(), true);
            }
        }
    }
}

pub type Shared = Arc<Mutex<Swarm>>;

impl Swarm {
    /// Tell every connected device which keys are approved now.
    pub fn broadcast_approved(&self) {
        let keys = self.auth.approved_keys();
        for e in self.nodes.values() {
            if let Some(t) = &e.tx {
                let _ = t.send(Message::Approved { keys: keys.clone() });
            }
        }
    }
}

/// Add the coordinator's own hardware to the roster so it takes part in planning.
/// Add the coordinator's own device to the roster before the portal opens (probe and model scan
/// take well under a second), so a pipeline can use it at once; the benchmark follows in the
/// background (`bench_self`). Registering only after the benchmark left a window in which the
/// coordinator's own models were unknown.
pub async fn register_self(shared: Shared, id: NodeId, overrides: crate::neighbour::Overrides) {
    let models_dir = shared.lock().unwrap().engine.models_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut info = probe::probe_node(id);
        info.perf_cores = Some(clamor_engine_ggml::default_threads() as u32);
        (info, crate::share::scan(&models_dir))
    })
    .await;
    let Ok((info, models)) = result else { return };
    let mut sw = shared.lock().unwrap();
    let share_port = sw.share_port;
    let policy = overrides.for_swarm(&sw.auth.profile.swarm_id, info.class).unwrap_or_else(|| Policy::default_for(info.class));
    let generation = sw.next_generation;
    sw.next_generation += 1;
    let my_fp = sw.tls.fingerprint_hex();
    let id = info.id.clone();
    sw.nodes.insert(
        id,
        NodeEntry {
            report: NodeReport { info, policy, bench: None, link: None },
            state: NodeState::Benchmarking,
            last_seen: unix_now(),
            tx: None,
            cpu_load: None,
            running_tasks: 0,
            note: Some("coordinator".into()),
            generation,
            tls_fp: Some(my_fp),
            peer_ip: None,
            coord_ip: None,
            build: Some(crate::build_info::current()),
            update_available: None,
            models,
            share_port,
            paused: None,
            settings_note: None,
            key: None,
        },
    );
}

/// Benchmark the coordinator's own device, then mark it ready.
pub async fn bench_self(shared: Shared) {
    let (me, ram) = {
        let sw = shared.lock().unwrap();
        let me = sw.coordinator.clone();
        let ram = sw.nodes.get(&me).map(|e| e.report.info.ram_available_bytes).unwrap_or(0);
        (me, ram)
    };
    let Ok(report) = tokio::task::spawn_blocking(move || bench::run(&bench::BenchOptions::quick(), ram)).await else { return };
    if let Some(e) = shared.lock().unwrap().nodes.get_mut(&me) {
        e.report.bench = Some(report);
        e.state = NodeState::Ready;
    }
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
    // Everything on the control connection is encrypted. A connection that does not complete
    // TLS is a reachability check (or something that is not Clamor 0.5+): drop it quietly.
    let acceptor = tokio_rustls::TlsAcceptor::from(shared.lock().unwrap().tls_server.clone());
    let Ok(Ok(tls)) = timeout(Duration::from_secs(10), acceptor.accept(stream)).await else {
        return Ok(());
    };
    let binding = clamor_transport::tls::exporter_server(tls.get_ref().1)?;
    let (mut rd, mut wr) = tokio::io::split(tls);

    let hello = match timeout(Duration::from_secs(10), read_msg(&mut rd)).await? {
        Ok(m) => m,
        // Closed before saying anything: a device checking that we are reachable.
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let Message::Hello { proto, info, bench_port, build, models, share_port, auth_nonce, invite, tls_fp, person_name, ask } = hello else {
        write_msg(&mut wr, &Message::Reject { reason: "expected hello".into() }).await?;
        return Ok(());
    };
    if proto != PROTOCOL_VERSION {
        write_msg(&mut wr, &Message::Reject { reason: format!("protocol {proto} not supported (coordinator speaks {PROTOCOL_VERSION})") }).await?;
        return Ok(());
    }

    let Some(nonce_m) = auth_nonce else {
        write_msg(&mut wr, &Message::Reject { reason: "this device's clamor is too old to join by passphrase: update it".into() }).await?;
        return Ok(());
    };

    // Authenticate: the device proves it is approved (its key) or knows the passphrase (SPAKE2).
    // The shared secret: the passphrase, or the one-time code from "Add device".
    let secret = {
        let mut sw = shared.lock().unwrap();
        if sw.invite.as_ref().is_some_and(|i| i.expires_unix < unix_now()) {
            sw.invite = None;
        }
        match (invite, &sw.invite) {
            (false, _) => Some((sw.auth.profile.passphrase.clone(), None)),
            (true, Some(i)) => Some((i.code.clone(), i.person.clone())),
            (true, None) => None,
        }
    };
    let Some((secret, invite_person)) = secret else {
        write_msg(&mut wr, &Message::Reject { reason: "that code has expired or was already used: ask for a new one (Add device in the dashboard)".into() }).await?;
        return Ok(());
    };
    let (hs, ch) = {
        let sw = shared.lock().unwrap();
        crate::auth::CoordinatorHandshake::start(&binding, &sw.auth.profile.swarm_id, &sw.name, &sw.auth.key, &secret)
    };
    write_msg(&mut wr, &Message::AuthChallenge { swarm_id: ch.swarm_id, swarm_name: ch.swarm_name, coordinator_key: ch.coordinator_key, nonce: ch.nonce, spake: ch.spake }).await?;
    let proof = match timeout(Duration::from_secs(30), read_msg(&mut rd)).await?? {
        Message::AuthProve { device_key, signature, spake, confirm } => crate::auth::Proof { device_key, signature, spake, confirm },
        Message::Leave => return Ok(()), // the device declined (e.g. the coordinator's key changed)
        _ => {
            write_msg(&mut wr, &Message::Reject { reason: "expected auth_prove".into() }).await?;
            return Ok(());
        }
    };
    let verified = {
        let mut sw = shared.lock().unwrap();
        let key = sw.auth.key.clone();
        let r = hs.verify(&key, &nonce_m, &proof, |k| sw.auth.profile.approved.iter().any(|d| d.key == k), ask && !invite);
        if invite && proof.spake.is_some() {
            // One-time: used up by a success; a few wrong tries cancel it.
            match &r {
                Ok(_) => sw.invite = None,
                Err(_) => {
                    if let Some(i) = sw.invite.as_mut() {
                        i.failures += 1;
                        if i.failures >= INVITE_TRIES {
                            println!("one-time code cancelled after {INVITE_TRIES} wrong tries");
                            sw.invite = None;
                        }
                    }
                }
            }
        }
        if let Ok((crate::auth::Verdict::Approve, _)) = &r {
            sw.auth.approve(&proof.device_key, &info.id.0, invite_person.as_deref(), person_name.as_deref());
            println!("{} approved with the {} (key {}…)", info.id, if invite { "one-time code" } else { "passphrase" }, &proof.device_key[..16]);
            // Everyone's share servers must now accept the new device.
            sw.broadcast_approved();
        }
        r
    };
    // Asking to join: wait for an admin to approve it in the dashboard.
    let verified = match verified {
        Ok((crate::auth::Verdict::Ask, cp)) => {
            let code = crate::auth::check_code(&binding, &crate::auth::key_hex(&shared.lock().unwrap().auth.key.verifying_key()), &proof.device_key);
            let id = hex::encode(crate::auth::random_bytes::<8>());
            {
                let mut sw = shared.lock().unwrap();
                sw.joins.retain(|j| j.created_unix + ASK_SECS > unix_now());
                sw.joins.push(PendingJoin { id: id.clone(), device: info.id.0.clone(), person_name: person_name.clone().unwrap_or_else(|| info.id.0.clone()), from: addr.ip().to_string(), code: code.clone(), created_unix: unix_now(), decision: None });
            }
            println!("{} ({}) asks to join; check code {code}: approve it in the dashboard", info.id, addr.ip());
            let t0 = std::time::Instant::now();
            let decision = loop {
                let d = shared.lock().unwrap().joins.iter().find(|j| j.id == id).map(|j| j.decision);
                match d {
                    Some(Some(d)) => break Some(d),
                    None => break None, // dropped
                    Some(None) if t0.elapsed() > Duration::from_secs(ASK_SECS) => break None,
                    Some(None) => sleep(Duration::from_millis(300)).await,
                }
            };
            shared.lock().unwrap().joins.retain(|j| j.id != id);
            match decision {
                Some(true) => {
                    let mut sw = shared.lock().unwrap();
                    sw.auth.approve(&proof.device_key, &info.id.0, None, person_name.as_deref());
                    sw.broadcast_approved();
                    println!("{} approved in the dashboard", info.id);
                    Ok(cp)
                }
                Some(false) => Err(anyhow::anyhow!("your request to join was declined")),
                None => Err(anyhow::anyhow!("nobody approved the request to join in time: ask again")),
            }
        }
        other => other.map(|(_, cp)| cp),
    };
    let cproof = match verified {
        Ok(v) => v,
        Err(e) => {
            let reason = if invite { format!("{e:#}").replace("wrong passphrase", "wrong code") } else { format!("{e:#}") };
            eprintln!("{} from {addr} refused: {reason}", info.id);
            write_msg(&mut wr, &Message::Reject { reason }).await?;
            return Ok(());
        }
    };
    let device_key = proof.device_key.clone();
    // approved_keys is filled in when the device enters the roster (same lock): from then on it
    // receives every `Approved` broadcast, so no approval can fall between the two.
    let mut welcome_auth = WelcomeAuth { signature: cproof.signature, confirm: cproof.confirm, approved_keys: Vec::new(), portal_tls_port: None, portal_tls_fp: None };

    let (tx, mut rx) = unbounded_channel::<Message>();
    let id = info.id.clone();
    let admission = {
        let mut sw = shared.lock().unwrap();
        if id == sw.coordinator {
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
                    tls_fp: tls_fp.clone(),
                    peer_ip: Some(addr.ip()),
                    coord_ip,
                    build,
                    update_available: None,
                    models,
                    share_port,
                    paused: None,
                    settings_note: None,
                    key: Some(device_key.clone()),
                },
            );
            welcome_auth.approved_keys = sw.auth.approved_keys();
            welcome_auth.portal_tls_port = sw.portal_tls_port;
            welcome_auth.portal_tls_fp = Some(sw.tls.fingerprint_hex());
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

    let _ = tx.send(Message::Welcome { swarm, coordinator, heartbeat_secs: HEARTBEAT_SECS, auth: Some(welcome_auth) });
    // The device answers Welcome with its owner's settings for this swarm (OwnerPolicy); until
    // then it keeps what we hold for it (from before it reconnected) or the defaults.
    if !cached_bench {
        set_state(&shared, &id, generation, NodeState::Benchmarking);
        let _ = tx.send(Message::BenchRequest { quick });
    }
    let mut writer = tokio::spawn(async move {
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
            Message::OwnerPolicy { policy } => {
                // The owner's settings win; with none, the device takes what we hold (it may have
                // been changed in the dashboard while the device was away).
                let mut sw = shared.lock().unwrap();
                if let Some(e) = sw.nodes.get_mut(&id) {
                    if let Some(p) = policy.filter(|p| crate::neighbour::validate(p).is_ok()) {
                        e.report.policy = p;
                    }
                    let _ = tx.send(Message::SetPolicy { policy: e.report.policy });
                }
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
    // Let the writer send what is queued (a Drain to a device that was just removed) before the
    // connection closes: aborting it could drop that message, and the device would reconnect
    // instead of leaving. It ends once every sender is gone (ours, and the roster's).
    drop(tx);
    if timeout(Duration::from_secs(2), &mut writer).await.is_err() {
        writer.abort();
    }
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
            // Six missed heartbeats (30 s). Tolerant on purpose: a device under heavy load can be
            // slow to answer, and a truly vanished one is caught sooner by the data links'
            // TCP keepalive.
            if e.tx.is_some() && now.saturating_sub(e.last_seen) > 6 * HEARTBEAT_SECS as u64 {
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

/// The coordinator's TLS certificate, kept per swarm so it stays the same across restarts (a
/// browser told to trust the dashboard's certificate keeps trusting it).
fn coordinator_identity(config_dir: &std::path::Path, swarm_id: &str) -> std::sync::Arc<clamor_transport::tls::Identity> {
    use clamor_transport::tls::Identity;
    let path = config_dir.join("swarms").join(format!("tls-{swarm_id}.json"));
    let load = || -> Option<std::sync::Arc<Identity>> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).ok()?).ok()?;
        Identity::from_parts(hex::decode(v["cert"].as_str()?).ok()?, hex::decode(v["key"].as_str()?).ok()?).ok()
    };
    if let Some(id) = load() {
        return id;
    }
    let id = Identity::generate().expect("TLS certificate");
    let (cert, key) = id.to_parts();
    let doc = serde_json::json!({ "cert": hex::encode(cert), "key": hex::encode(key) });
    if let Err(e) = crate::auth::write_private(&path, doc.to_string().as_bytes()) {
        eprintln!("could not save the TLS certificate: {e:#}");
    }
    id
}
