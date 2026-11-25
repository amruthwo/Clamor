//! `clamor-node`: run a coordinator (roster + web portal) or join one as a member. A library so
//! the apps can run a node in-process (`run_args`, `shutdown`); src/main.rs is the command line.

mod access;
mod auth;
mod build_info;
mod cache;
mod chats;
mod coordinator;
mod dashproxy;
mod discovery;
mod generate;
mod member;
mod neighbour;
mod people;
mod pipeline;
mod portal;
mod private;
mod share;
mod stage;
mod update;
mod wire;

use anyhow::Result;
use clamor_muster::probe;
use clamor_proto::*;
use clap::{Parser, Subcommand};
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(name = "clamor-node", version, about = "Pool the idle computers on your network")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// CPU priority for Clamor's model computation. `low` (default) lets your own apps always go
    /// first, so a busy swarm never makes this machine unresponsive; `normal` competes with them
    /// equally. Heartbeats and networking always run at normal priority.
    #[arg(long, global = true, value_enum)]
    priority: Option<Priority>,
    /// Where this device keeps its settings (default: the system config folder, e.g.
    /// ~/.config/clamor).
    #[arg(long, global = true)]
    config_dir: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Priority {
    Low,
    Normal,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show this build: version, platform, GPU variant, CPU level.
    Version {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// List the compute devices this build can use, and check each computes correctly.
    Devices,
    /// Update to the latest release from GitHub (only builds signed with the Clamor release key,
    /// matching this platform and GPU variant).
    Update {
        /// Only report whether an update is available.
        #[arg(long)]
        check: bool,
        /// Put back the version the last update replaced.
        #[arg(long, conflicts_with = "check")]
        rollback: bool,
    },
    /// Start a swarm: control plane, muster, and web portal.
    Coordinator {
        #[arg(long, default_value = "clamor")]
        swarm: String,
        #[arg(long, default_value_t = format!("0.0.0.0:{DEFAULT_PORT}"))]
        bind: String,
        #[arg(long, default_value_t = format!("0.0.0.0:{DEFAULT_PORTAL_PORT}"))]
        portal: String,
        /// The encrypted (HTTPS) dashboard; members' apps use it. Another free port if taken.
        #[arg(long, default_value = "0.0.0.0:7424")]
        portal_tls: String,
        /// The swarm's passphrase. Default: the one this swarm already has, or five random
        /// words for a new swarm (printed at start). Devices need it only for their first join.
        #[arg(long, alias = "token", env = "CLAMOR_PASSPHRASE", hide_env_values = true)]
        passphrase: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[command(flatten)]
        engine: EngineArgs,
        /// First data-plane port; pipeline stage i listens on this + i.
        #[arg(long, default_value_t = DEFAULT_DATA_PORT)]
        data_port: u16,
        /// Ask joining devices for the quick benchmark (seconds instead of a minute).
        #[arg(long)]
        quick_bench: bool,
        #[command(flatten)]
        updates: UpdateArgs,
        #[command(flatten)]
        policy: PolicyArgs,
        /// Do not advertise the swarm on the local network (devices then need its address).
        #[arg(long)]
        no_mdns: bool,
    },
    /// Join a swarm. Without an address, find it on the local network: a swarm this device
    /// joined before, or else the only swarm there (or the one named with --swarm).
    Join {
        /// Coordinator address, host:port (default port 7420).
        addr: Option<String>,
        /// A one-time code from "Add device" in the dashboard (or `clamor-node invite`), instead
        /// of the passphrase.
        #[arg(long, conflicts_with = "passphrase")]
        code: Option<String>,
        /// Ask to join without a code or passphrase: an admin approves this device in the
        /// dashboard, after checking that both screens show the same check code.
        #[arg(long, conflicts_with_all = ["passphrase", "code"])]
        ask: bool,
        /// Your name, shown to the others in the swarm (first join; default: this device's name).
        #[arg(long)]
        name: Option<String>,
        /// Serve the swarm's dashboard on this device, signed in as you ("off" to not).
        #[arg(long, default_value_t = format!("127.0.0.1:{DEFAULT_PORTAL_PORT}"))]
        dashboard: String,
        /// Without an address: join the swarm with this name.
        #[arg(long)]
        swarm: Option<String>,
        /// Never look for swarms on the network.
        #[arg(long)]
        no_mdns: bool,
        /// The swarm's passphrase: needed for the first join only (afterwards this device is
        /// recognised by its key).
        #[arg(long, alias = "token", env = "CLAMOR_PASSPHRASE", hide_env_values = true)]
        passphrase: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[command(flatten)]
        engine: EngineArgs,
        #[command(flatten)]
        updates: UpdateArgs,
        #[command(flatten)]
        policy: PolicyArgs,
    },
    /// Forget a swarm this device joined: it will not rejoin it without the passphrase.
    /// (To leave a running swarm for now, stop clamor-node with Ctrl-C or drain the device in
    /// the dashboard.)
    Leave {
        /// The swarm's name (may be left out if this device joined only one).
        swarm: Option<String>,
        /// Also delete the cached model slices only that swarm used.
        #[arg(long)]
        purge: bool,
        /// Delete a swarm this device coordinates: its passphrase and approved devices. Every
        /// device will have to join the new swarm with a new passphrase.
        #[arg(long)]
        delete_swarm: bool,
        /// The slice cache folder, if not the default.
        #[arg(long)]
        cache_dir: Option<std::path::PathBuf>,
    },
    /// On the coordinator: make a one-time code for adding a device (valid 10 minutes, once).
    Invite {
        /// The coordinator's web portal on this machine.
        #[arg(long, default_value_t = format!("127.0.0.1:{DEFAULT_PORTAL_PORT}"))]
        portal: String,
    },
    /// List the swarms this device coordinates or has joined.
    Swarms {
        /// Also list the swarms advertised on the local network now.
        #[arg(long)]
        scan: bool,
        /// Machine-readable output (used by the desktop app). Passphrases are left out.
        #[arg(long)]
        json: bool,
    },
}

/// How much of this device Clamor may use. Unset options keep the defaults for the kind of
/// device (laptops and phones: idle-only, plugged in only, half the RAM and CPU; desktops: full,
/// 70% RAM, 80% CPU).
#[derive(clap::Args)]
struct PolicyArgs {
    /// `full`: always take work. `idle-only`: step aside while you are using the device.
    #[arg(long, value_enum)]
    participation: Option<ParticipationArg>,
    /// Keep working on battery (laptops and phones otherwise pause until plugged in).
    #[arg(long)]
    allow_battery: bool,
    /// Share of RAM Clamor may use, 0.05-0.95.
    #[arg(long)]
    ram_share: Option<f32>,
    /// Share of the performance cores Clamor may use, 0.05-1.0.
    #[arg(long)]
    cpu_share: Option<f32>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ParticipationArg {
    Full,
    IdleOnly,
}

impl PolicyArgs {
    /// The owner's command-line settings; they win in every swarm, over what the device
    /// remembers for that swarm, over the defaults.
    fn overrides(&self, priority: Option<Priority>) -> neighbour::Overrides {
        neighbour::Overrides {
            participation: self.participation.map(|x| match x {
                ParticipationArg::Full => Participation::Full,
                ParticipationArg::IdleOnly => Participation::IdleOnly,
            }),
            allow_battery: self.allow_battery,
            ram_share: self.ram_share,
            cpu_share: self.cpu_share,
            low_priority: priority.map(|p| p == Priority::Low),
        }
    }
}

#[derive(clap::Args)]
struct UpdateArgs {
    /// Install new Clamor releases automatically when idle (no pipeline stage running), then
    /// restart with the same options. Only builds signed with the Clamor release key are used.
    #[arg(long)]
    auto_update: bool,
    /// Never contact GitHub to look for new releases.
    #[arg(long, conflicts_with = "auto_update")]
    no_update_check: bool,
}

impl UpdateArgs {
    fn policy(&self) -> update::Policy {
        update::Policy { check: !self.no_update_check, auto: self.auto_update }
    }
}

#[derive(clap::Args)]
struct EngineArgs {
    /// Directory holding model files (.gguf). Pipelines name a file in it.
    #[arg(long, default_value = "models")]
    models: std::path::PathBuf,
    /// Compute device for pipeline stages: cpu, a device name such as Vulkan0, or an index.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// CPU threads per stage. Default: the performance cores (efficiency cores and
    /// hyperthreads slow inference down).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Where layer slices fetched from other devices are kept (default: the system cache
    /// folder, e.g. ~/.cache/clamor). Delete it any time to reclaim space.
    #[arg(long)]
    cache_dir: Option<std::path::PathBuf>,
    /// Most disk space the cache may use, in GB. The least recently used slices are removed
    /// beyond that.
    #[arg(long, default_value_t = 32.0)]
    cache_max_gb: f64,
}

fn default_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("clamor").join("slices")
}

impl EngineArgs {
    /// `share_key`: signs this device's requests for model slices.
    fn config(self, share_key: Option<ed25519_dalek::SigningKey>, swarm_id: Option<String>) -> stage::EngineConfig {
        let cache_dir = self.cache_dir.unwrap_or_else(default_cache_dir);
        let cache_max = (self.cache_max_gb.max(0.0) * 1e9) as u64;
        stage::EngineConfig { models_dir: self.models, device: self.device, threads: self.threads, cache_dir, share_key, swarm_id, cache_max, tls: None }
    }
}

/// The `clamor-node` command line (the binary is just this).
pub fn main_cli() -> Result<()> {
    run_cli(Cli::parse())
}

/// Run `clamor-node` inside another program (the apps): `args` as on the command line, without
/// the program name. Blocks until the node stops: `shutdown()` asks it to (a member leaves the
/// swarm gracefully first).
pub fn run_args(args: Vec<String>) -> Result<()> {
    SHUTDOWN_FLAG.store(false, std::sync::atomic::Ordering::SeqCst);
    let cli = Cli::try_parse_from(std::iter::once("clamor-node".to_string()).chain(args)).map_err(|e| anyhow::anyhow!("{e}"))?;
    run_cli(cli)
}

fn run_cli(cli: Cli) -> Result<()> {
    neighbour::set_config_dir(cli.config_dir.clone());
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run(cli.cmd, cli.priority))
}

static SHUTDOWN: tokio::sync::Notify = tokio::sync::Notify::const_new();
static SHUTDOWN_FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ask a node started with `run_args` to stop (like Ctrl-C).
pub fn shutdown() {
    SHUTDOWN_FLAG.store(true, std::sync::atomic::Ordering::SeqCst);
    SHUTDOWN.notify_waiters();
}

/// Resolves when the node should stop: Ctrl-C, or `shutdown()`.
pub(crate) async fn stop_requested() {
    let asked = async {
        loop {
            let n = SHUTDOWN.notified();
            if SHUTDOWN_FLAG.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            n.await;
        }
    };
    tokio::select! {
        _ = asked => {}
        r = tokio::signal::ctrl_c() => {
            // Where signals cannot be watched (some embedded hosts), only shutdown() stops it.
            if r.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }
}

async fn run(cmd: Cmd, priority: Option<Priority>) -> Result<()> {
    match cmd {
        Cmd::Version { json } => {
            let b = build_info::current();
            if json {
                println!("{}", serde_json::to_string_pretty(&b)?);
            } else {
                println!("{}", build_info::describe(&b));
            }
            Ok(())
        }
        Cmd::Update { check, rollback } => {
            let exe = std::env::current_exe()?;
            if rollback {
                update::rollback(&exe)?;
                println!("rolled back {}", exe.display());
                return Ok(());
            }
            let me = build_info::current();
            println!("this build: {}", build_info::describe(&me));
            // Network and disk work: keep it off the async runtime.
            let found = tokio::task::spawn_blocking(move || update::check(&update::Source::github(), &me)).await??;
            match found {
                update::Check::UpToDate { latest } => println!("up to date (latest release: {latest})"),
                update::Check::NoReleases => println!("no releases have been published yet"),
                update::Check::NoMatchingBuild { latest } => {
                    println!("release {latest} is out, but has no build for this platform and GPU variant")
                }
                update::Check::Available(av) => {
                    println!("update available: {} ({}, {} MB)", av.version, av.asset.file, av.asset.size >> 20);
                    if let Some(page) = &av.release_page {
                        println!("release notes: {page}");
                    }
                    if !check {
                        let backup = tokio::task::spawn_blocking(move || update::apply(&av, &exe)).await??;
                        println!("updated. The previous version is kept at {} (clamor-node update --rollback).", backup.display());
                        println!("Restart clamor-node to run the new version.");
                    }
                }
            }
            Ok(())
        }
        Cmd::Devices => {
            println!("{}", build_info::describe(&build_info::current()));
            println!("default CPU threads: {} (the performance cores; override with --threads)", clamor_engine_ggml::default_threads());
            for d in clamor_engine_ggml::list_devices() {
                println!("[{}] {}: {} ({:?}, {} MiB)", d.index, d.name, d.description, d.kind, d.memory_total >> 20);
                match clamor_engine_ggml::matmul_check(d.index) {
                    Ok(r) => println!("    matmul ok (max error {:.1e})", r.max_abs_err),
                    Err(e) => println!("    cannot be used: {e}"),
                }
            }
            Ok(())
        }
        Cmd::Coordinator { swarm, bind, portal, portal_tls, passphrase, id, engine, data_port, quick_bench, updates, policy, no_mdns } => {
            let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
            let config_dir = neighbour::config_dir();
            let profile = auth::CoordinatorProfile::load_or_create(&config_dir, &swarm, passphrase.as_deref())?;
            let key = auth::Device::load_or_create(&config_dir)?.swarm_key(&profile.swarm_id);
            let engine = engine.config(Some(key.clone()), Some(profile.swarm_id.clone()));
            engine.validate()?; // fail fast on a bad --device
            println!("swarm '{swarm}' passphrase: {}", profile.passphrase);
            println!("add a device: clamor-node join {} --passphrase {}", join_hint(&bind), profile.passphrase);
            if !profile.approved.is_empty() {
                println!("{} device(s) approved; they rejoin without the passphrase", profile.approved.len());
            }
            // Advertise on the network, if listening on it.
            if let Ok(sa) = bind.parse::<std::net::SocketAddr>() {
                if sa.ip().is_unspecified() && !no_mdns {
                    match discovery::advertise(&profile.swarm_id, &swarm, sa.port()) {
                        Ok(()) => println!("advertising swarm '{swarm}' on the local network (devices can run `clamor-node join`)"),
                        Err(e) => eprintln!("could not advertise on the local network: {e:#}"),
                    }
                }
            }
            let swarm_auth = coordinator::SwarmAuth::new(profile, key, config_dir);
            // Share our models with the swarm's devices (layer slices).
            let share_port = share::serve(DEFAULT_SHARE_PORT, engine.models_dir.clone(), swarm_auth.allowed.clone()).await?;
            let shared = Arc::new(Mutex::new(coordinator::Swarm::new(swarm, swarm_auth, id.clone(), engine, data_port, quick_bench, Some(share_port))));
            shared.lock().unwrap().join_hint = join_hint(&bind);
            coordinator::register_self(shared.clone(), id, policy.overrides(priority)).await;
            tokio::spawn(coordinator::bench_self(shared.clone()));
            tokio::spawn(coordinator::reaper(shared.clone()));
            tokio::spawn(coordinator::monitor_self(shared.clone()));
            tokio::spawn(coordinator::updates(shared.clone(), updates.policy()));
            let portal_shared = shared.clone();
            tokio::spawn(async move {
                if let Err(e) = portal::serve(portal, portal_tls, portal_shared).await {
                    eprintln!("portal stopped: {e}");
                }
            });
            coordinator::run_listener(&bind, shared).await
        }
        Cmd::Join { addr, swarm, no_mdns, passphrase, code, ask, name, dashboard, id, engine, updates, policy } => {
            let config_dir = neighbour::config_dir();
            let addr = match addr {
                Some(a) if a.contains(':') => a,
                Some(a) => format!("{a}:{DEFAULT_PORT}"),
                None => find_swarm(&config_dir, swarm.as_deref(), no_mdns).await?,
            };
            let engine = engine.config(None, None); // the key is per swarm: set once joined
            engine.validate()?;
            let creds = member::Credentials { device: auth::Device::load_or_create(&config_dir)?, passphrase, code, person_name: name, ask, dashboard: (dashboard != "off").then_some(dashboard), config_dir };
            member::join(addr, creds, id, engine, updates.policy(), policy.overrides(priority), !no_mdns).await
        }
        Cmd::Leave { swarm, purge, delete_swarm, cache_dir } => {
            let config_dir = neighbour::config_dir();
            let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
            let (id, name) = if delete_swarm {
                let all = auth::CoordinatorProfile::all(&config_dir);
                let p = pick(all.iter().map(|p| (&p.swarm_id, &p.name)), swarm.as_deref(), "coordinates")?;
                auth::CoordinatorProfile::delete(&config_dir, &p.0)?;
                println!("deleted swarm '{}': its passphrase and approvals are gone", p.1);
                p
            } else {
                let all = auth::MemberProfile::all(&config_dir);
                let p = pick(all.iter().map(|p| (&p.swarm_id, &p.name)), swarm.as_deref(), "has joined")?;
                auth::MemberProfile::delete(&config_dir, &p.0)?;
                println!("forgot swarm '{}': joining it again needs its passphrase", p.1);
                p
            };
            if purge {
                let (n, bytes) = cache::purge(&cache_dir, &id);
                println!("removed {n} cached slice(s) of '{name}' ({:.1} GB)", bytes as f64 / 1e9);
            }
            Ok(())
        }
        Cmd::Invite { portal } => {
            let url = format!("http://{portal}/api/invite");
            let r = tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
                let mut resp = ureq::post(&url).config().http_status_as_error(false).build().send_empty()?;
                let text = resp.body_mut().read_to_string()?;
                if !resp.status().is_success() {
                    anyhow::bail!("{text}");
                }
                Ok(serde_json::from_str(&text)?)
            })
            .await?
            .map_err(|e| anyhow::anyhow!("could not reach the coordinator on this machine ({portal}): {e:#}"))?;
            println!("one-time code: {} (works once, for {} minutes)", r["code"].as_str().unwrap_or("?"), r["expires_in"].as_u64().unwrap_or(0) / 60);
            println!("on the new device: {}", r["command"].as_str().unwrap_or(""));
            Ok(())
        }
        Cmd::Swarms { scan, json } => {
            let config_dir = neighbour::config_dir();
            let coordinated = auth::CoordinatorProfile::all(&config_dir);
            let joined = auth::MemberProfile::all(&config_dir);
            if json {
                let found = if scan { tokio::task::spawn_blocking(|| discovery::browse(std::time::Duration::from_secs(3))).await?? } else { Vec::new() };
                let remembered: Vec<&str> = joined.iter().map(|p| p.swarm_id.as_str()).chain(coordinated.iter().map(|p| p.swarm_id.as_str())).collect();
                let out = serde_json::json!({
                    "coordinating": coordinated.iter().map(|p| serde_json::json!({ "id": p.swarm_id, "name": p.name, "approved_devices": p.approved.len() })).collect::<Vec<_>>(),
                    "joined": joined.iter().map(|p| serde_json::json!({ "id": p.swarm_id, "name": p.name, "address": p.addresses.first(), "last_joined_unix": p.last_joined_unix })).collect::<Vec<_>>(),
                    "on_network": found.iter().map(|f| serde_json::json!({ "id": f.swarm_id, "name": f.name, "addresses": f.addrs.iter().map(|a| a.to_string()).collect::<Vec<_>>(), "known": remembered.contains(&f.swarm_id.as_str()) })).collect::<Vec<_>>(),
                    "cache_bytes": cache::size(&default_cache_dir()),
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }
            if coordinated.is_empty() && joined.is_empty() {
                println!("no swarms yet. Start one with `clamor-node coordinator`, or join one with `clamor-node join <coordinator> --passphrase <words>`.");
            }
            for p in coordinated {
                println!("coordinating '{}': passphrase {}, {} device(s) approved", p.name, p.passphrase, p.approved.len());
                for d in &p.approved {
                    println!("    {} (key {}…)", d.name, &d.key[..16]);
                }
            }
            for p in joined {
                println!("joined '{}' at {} (last joined {})", p.name, p.addresses.first().map(String::as_str).unwrap_or("?"), ago(p.last_joined_unix));
            }
            let cache = default_cache_dir();
            println!("model slice cache: {:.1} GB in {}", cache::size(&cache) as f64 / 1e9, cache.display());
            if scan {
                let found = tokio::task::spawn_blocking(|| discovery::browse(std::time::Duration::from_secs(3))).await??;
                if found.is_empty() {
                    println!("no swarms advertised on the local network");
                }
                for f in found {
                    let at: Vec<String> = f.addrs.iter().map(|a| a.to_string()).collect();
                    println!("on the network: '{}' at {}", f.name, at.join(", "));
                }
            }
            Ok(())
        }
    }
}

/// How devices reach this coordinator, for the printed join command.
fn join_hint(bind: &str) -> String {
    let port = bind.rsplit(':').next().unwrap_or("7420");
    let host = if bind.starts_with("0.0.0.0") || bind.starts_with("[::]") { local_ip().unwrap_or_else(|| "<this-computer>".into()) } else { bind.rsplit_once(':').map(|(h, _)| h.to_string()).unwrap_or_default() };
    if port == DEFAULT_PORT.to_string() { host } else { format!("{host}:{port}") }
}

/// This machine's LAN address (the one used to reach the internet; no packet is sent).
fn local_ip() -> Option<String> {
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:9").ok()?;
    Some(s.local_addr().ok()?.ip().to_string())
}

fn ago(unix: u64) -> String {
    let s = unix_now().saturating_sub(unix);
    match s {
        0..=119 => "just now".into(),
        120..=7199 => format!("{} minutes ago", s / 60),
        7200..=172_799 => format!("{} hours ago", s / 3600),
        _ => format!("{} days ago", s / 86400),
    }
}

/// Where to join when no address was given: a remembered swarm found on the network (by id, so
/// a changed IP address does not matter), the only swarm on the network for a first join, or
/// else the address a remembered swarm was last reached at.
async fn find_swarm(config_dir: &std::path::Path, want: Option<&str>, no_mdns: bool) -> Result<String> {
    let profiles: Vec<auth::MemberProfile> = auth::MemberProfile::all(config_dir).into_iter().filter(|p| want.is_none_or(|w| p.name == w)).collect();
    let mut why = String::new();
    if !no_mdns {
        println!("looking for swarms on the local network...");
        let found = tokio::task::spawn_blocking(|| discovery::browse(std::time::Duration::from_secs(3))).await?.unwrap_or_else(|e| {
            eprintln!("cannot search the local network: {e:#}");
            Vec::new()
        });
        let ids: Vec<String> = profiles.iter().map(|p| p.swarm_id.clone()).collect();
        match discovery::choose(&found, &ids, want) {
            Ok(f) => {
                let f = f.clone();
                let name = f.name.clone();
                if let Some(a) = tokio::task::spawn_blocking(move || f.reachable()).await? {
                    println!("found swarm '{name}' at {a}");
                    return Ok(a.to_string());
                }
                why = format!("swarm '{name}' is advertised but cannot be reached");
            }
            Err(e) => why = e.to_string(),
        }
    }
    if let Some(p) = profiles.iter().find(|p| !p.addresses.is_empty()) {
        if !why.is_empty() {
            println!("{why}; trying where swarm '{}' was last", p.name);
        }
        println!("rejoining swarm '{}' at {}", p.name, p.addresses[0]);
        return Ok(p.addresses[0].clone());
    }
    if why.is_empty() {
        why = "this device has not joined a swarm yet".into();
    }
    anyhow::bail!("{why}. Give the coordinator's address: clamor-node join <coordinator> --passphrase <words>")
}

/// One (id, name) of `all`: the one called `want`, or the only one.
fn pick<'a>(all: impl Iterator<Item = (&'a String, &'a String)>, want: Option<&str>, verb: &str) -> Result<(String, String)> {
    let all: Vec<(String, String)> = all.map(|(i, n)| (i.clone(), n.clone())).collect();
    let names = || all.iter().map(|p| p.1.as_str()).collect::<Vec<_>>().join(", ");
    match want {
        Some(w) => all.iter().find(|p| p.1 == w).cloned().ok_or_else(|| anyhow::anyhow!("this device {verb} no swarm called '{w}' (it {verb}: {})", if all.is_empty() { "none".into() } else { names() })),
        None if all.len() == 1 => Ok(all[0].clone()),
        None if all.is_empty() => anyhow::bail!("this device {verb} no swarms"),
        None => anyhow::bail!("which swarm? this device {verb}: {}", names()),
    }
}
