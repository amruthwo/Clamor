//! `clamor-node`: run a coordinator (roster + web portal) or join one as a member.

mod coordinator;
mod member;
mod pipeline;
mod portal;
mod stage;
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
    },
}

#[derive(clap::Args)]
struct EngineArgs {
    /// Directory holding model files (.gguf). Pipelines name a file in it.
    #[arg(long, default_value = "models")]
    models: std::path::PathBuf,
    /// Compute device for pipeline stages: cpu, a device name such as Vulkan0, or an index.
    #[arg(long, default_value = "cpu")]
    device: String,
}

impl EngineArgs {
    fn config(self) -> stage::EngineConfig {
        stage::EngineConfig { models_dir: self.models, device: self.device }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Coordinator { swarm, bind, portal, token, id, engine, data_port, quick_bench } => {
            let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
            let engine = engine.config();
            engine.validate()?; // fail fast on a bad --device
            let shared = Arc::new(Mutex::new(coordinator::Swarm::new(swarm, token, id.clone(), engine, data_port, quick_bench)));
            tokio::spawn(coordinator::register_self(shared.clone(), id));
            tokio::spawn(coordinator::reaper(shared.clone()));
            let portal_shared = shared.clone();
            tokio::spawn(async move {
                if let Err(e) = portal::serve(portal, portal_shared).await {
                    eprintln!("portal stopped: {e}");
                }
            });
            coordinator::run_listener(&bind, shared).await
        }
        Cmd::Join { addr, token, id, engine } => {
            let addr = if addr.contains(':') { addr } else { format!("{addr}:{DEFAULT_PORT}") };
            let engine = engine.config();
            engine.validate()?;
            member::join(addr, token, id, engine).await
        }
    }
}
