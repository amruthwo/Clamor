//! `clamor-node`: run a coordinator (roster + web portal) or join one as a member.

mod build_info;
mod coordinator;
mod member;
mod neighbour;
mod pipeline;
mod portal;
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
    /// CPU priority for Clamor's work. `low` (default) lets your own apps always go first, so a
    /// busy swarm never makes this machine unresponsive; `normal` competes with them equally.
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

/// Lower this process's CPU priority. Called before the async runtime starts, because on Linux
/// priority is per thread and new threads inherit their creator's.
fn lower_priority() {
    if neighbour::set_priority(true).is_ok() {
        return;
    }
    #[cfg(unix)]
    unsafe {
        // nice 10: below interactive apps, above batch jobs.
        if libc::setpriority(libc::PRIO_PROCESS, 0, 10) != 0 {
            eprintln!("could not lower CPU priority (continuing at normal priority)");
        }
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass, BELOW_NORMAL_PRIORITY_CLASS};
        if SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS) == 0 {
            eprintln!("could not lower CPU priority (continuing at normal priority)");
        }
    }
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
        /// Shared secret nodes must present to join.
        #[arg(long)]
        token: Option<String>,
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
    },
    /// Join a swarm.
    Join {
        /// Coordinator address, host:port (default port 7420).
        addr: String,
        #[arg(long)]
        token: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[command(flatten)]
        engine: EngineArgs,
        #[command(flatten)]
        updates: UpdateArgs,
        #[command(flatten)]
        policy: PolicyArgs,
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
    /// The owner's settings on top of the defaults for this kind of device, or None if the
    /// owner set nothing (the coordinator then applies its defaults).
    /// Command-line flags, on top of the settings saved from the dashboard, on top of the
    /// defaults for this kind of device. None if there are neither flags nor saved settings.
    fn policy(&self, priority: Option<Priority>) -> Option<Policy> {
        let saved = neighbour::saved_settings();
        if saved.is_none() && self.participation.is_none() && !self.allow_battery && self.ram_share.is_none() && self.cpu_share.is_none() && priority.is_none() {
            return None;
        }
        let mut p = saved.unwrap_or_else(|| Policy::default_for(clamor_muster::probe::probe_node(NodeId("local".into())).class));
        if let Some(x) = self.participation {
            p.participation = match x {
                ParticipationArg::Full => Participation::Full,
                ParticipationArg::IdleOnly => Participation::IdleOnly,
            };
        }
        if self.allow_battery {
            p.require_plugged_in = false;
        }
        if let Some(r) = self.ram_share {
            p.max_ram_fraction = r.clamp(0.05, 0.95);
        }
        if let Some(c) = self.cpu_share {
            p.max_cpu_fraction = c.clamp(0.05, 1.0);
        }
        if let Some(pr) = priority {
            p.low_priority = pr == Priority::Low;
        }
        Some(p)
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
}

impl EngineArgs {
    /// `token`: the swarm's join token, also required to fetch model slices.
    fn config(self, token: Option<String>) -> stage::EngineConfig {
        let cache_dir = self.cache_dir.unwrap_or_else(|| dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("clamor").join("slices"));
        stage::EngineConfig { models_dir: self.models, device: self.device, threads: self.threads, cache_dir, share_token: token }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    neighbour::set_config_dir(cli.config_dir.clone());
    // --priority, else the setting saved from the dashboard, else low.
    let low = match cli.priority {
        Some(p) => p == Priority::Low,
        None => neighbour::saved_settings().map(|p| p.low_priority).unwrap_or(true),
    };
    if low && matches!(cli.cmd, Cmd::Coordinator { .. } | Cmd::Join { .. }) {
        lower_priority();
    }
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run(cli.cmd, cli.priority))
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
        Cmd::Coordinator { swarm, bind, portal, token, id, engine, data_port, quick_bench, updates, policy } => {
            let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
            let engine = engine.config(token.clone());
            engine.validate()?; // fail fast on a bad --device
            // Share our models with members that lack them (layer slices), behind the join token.
            let share_port = share::serve(DEFAULT_SHARE_PORT, engine.models_dir.clone(), token.clone()).await?;
            let shared = Arc::new(Mutex::new(coordinator::Swarm::new(swarm, token, id.clone(), engine, data_port, quick_bench, Some(share_port))));
            tokio::spawn(coordinator::register_self(shared.clone(), id, policy.policy(priority)));
            tokio::spawn(coordinator::reaper(shared.clone()));
            tokio::spawn(coordinator::monitor_self(shared.clone()));
            tokio::spawn(coordinator::updates(shared.clone(), updates.policy()));
            let portal_shared = shared.clone();
            tokio::spawn(async move {
                if let Err(e) = portal::serve(portal, portal_shared).await {
                    eprintln!("portal stopped: {e}");
                }
            });
            coordinator::run_listener(&bind, shared).await
        }
        Cmd::Join { addr, token, id, engine, updates, policy } => {
            let addr = if addr.contains(':') { addr } else { format!("{addr}:{DEFAULT_PORT}") };
            let engine = engine.config(token.clone());
            engine.validate()?;
            member::join(addr, token, id, engine, updates.policy(), policy.policy(priority)).await
        }
    }
}
