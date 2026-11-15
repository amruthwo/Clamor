//! `clamor-node`: run a coordinator (roster + web portal) or join one as a member.

mod build_info;
mod coordinator;
mod member;
mod pipeline;
mod portal;
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
    },
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
}

impl EngineArgs {
    fn config(self) -> stage::EngineConfig {
        stage::EngineConfig { models_dir: self.models, device: self.device, threads: self.threads }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
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
        Cmd::Coordinator { swarm, bind, portal, token, id, engine, data_port, quick_bench, updates } => {
            let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
            let engine = engine.config();
            engine.validate()?; // fail fast on a bad --device
            let shared = Arc::new(Mutex::new(coordinator::Swarm::new(swarm, token, id.clone(), engine, data_port, quick_bench)));
            tokio::spawn(coordinator::register_self(shared.clone(), id));
            tokio::spawn(coordinator::reaper(shared.clone()));
            tokio::spawn(coordinator::updates(shared.clone(), updates.policy()));
            let portal_shared = shared.clone();
            tokio::spawn(async move {
                if let Err(e) = portal::serve(portal, portal_shared).await {
                    eprintln!("portal stopped: {e}");
                }
            });
            coordinator::run_listener(&bind, shared).await
        }
        Cmd::Join { addr, token, id, engine, updates } => {
            let addr = if addr.contains(':') { addr } else { format!("{addr}:{DEFAULT_PORT}") };
            let engine = engine.config();
            engine.validate()?;
            member::join(addr, token, id, engine, updates.policy()).await
        }
    }
}
