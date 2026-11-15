//! `clamor-node`: run a coordinator (roster + web portal) or join one as a member.

mod coordinator;
mod member;
mod portal;
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
    },
    /// Join a swarm.
    Join {
        /// Coordinator address, host:port (default port 7420).
        addr: String,
        #[arg(long)]
        token: Option<String>,
        #[arg(long)]
        id: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Coordinator { swarm, bind, portal, token, id } => {
            let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
            let shared = Arc::new(Mutex::new(coordinator::Swarm::new(swarm, token, id.clone())));
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
        Cmd::Join { addr, token, id } => {
            let addr = if addr.contains(':') { addr } else { format!("{addr}:{DEFAULT_PORT}") };
            member::join(addr, token, id).await
        }
    }
}
