//! `clamor-muster`: benchmark this device, measure links, and plan which models the group can run.

use anyhow::{bail, Context, Result};
use clamor_muster::{bench, net, plan, probe};
use clamor_proto::*;
use clap::{Parser, Subcommand};
use std::net::TcpListener;

#[derive(Parser)]
#[command(name = "clamor-muster", version, about = "Muster, benchmark and plan for a Clamor swarm")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Probe and benchmark this device; optionally measure the link to a coordinator.
    Probe {
        /// Write the NodeReport JSON here.
        #[arg(long)]
        out: Option<String>,
        /// Coordinator bench server (host:port) to measure latency and throughput against.
        #[arg(long)]
        coordinator: Option<String>,
        /// Shorter tests, no disk benchmark.
        #[arg(long)]
        quick: bool,
        /// Override the node id (defaults to the hostname).
        #[arg(long)]
        id: Option<String>,
    },
    /// Run the network benchmark server other devices measure against.
    Serve {
        #[arg(long, default_value = "0.0.0.0:7422")]
        bind: String,
    },
    /// Measure latency and throughput to a bench server.
    Net {
        addr: String,
        #[arg(long, default_value_t = 50)]
        pings: u32,
        /// Bulk transfer size in MiB (0 = latency only).
        #[arg(long, default_value_t = 64)]
        mib: u64,
    },
    /// Combine NodeReport files and show which models the group can run.
    Plan {
        files: Vec<String>,
        /// Node id of the coordinator (defaults to the first file's node).
        #[arg(long)]
        coordinator: Option<String>,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long, default_value = "q4")]
        quant: String,
        /// Smallest acceptable decode speed when recommending a model.
        #[arg(long, default_value_t = 2.0)]
        target_tok_s: f64,
        /// Plan one custom ModelSpec JSON file instead of the built-in size classes.
        #[arg(long)]
        model_file: Option<String>,
        /// ring (stage-to-stage, clamor-transport) or hub (via coordinator, llama.cpp RPC style).
        #[arg(long, default_value = "ring")]
        topology: String,
        /// Print machine-readable JSON instead of tables.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Probe { out, coordinator, quick, id } => cmd_probe(out, coordinator, quick, id),
        Cmd::Serve { bind } => {
            let l = TcpListener::bind(&bind).with_context(|| format!("bind {bind}"))?;
            eprintln!("bench server listening on {bind}");
            net::serve(l)?;
            Ok(())
        }
        Cmd::Net { addr, pings, mib } => {
            let l = net::measure(&addr, pings, mib << 20)?;
            print_link(&l);
            Ok(())
        }
        Cmd::Plan { files, coordinator, ctx, quant, target_tok_s, model_file, topology, json } => {
            cmd_plan(files, coordinator, ctx, &quant, target_tok_s, model_file, &topology, json)
        }
    }
}

fn cmd_probe(out: Option<String>, coordinator: Option<String>, quick: bool, id: Option<String>) -> Result<()> {
    let id = id.map(NodeId).unwrap_or_else(probe::default_node_id);
    let info = probe::probe_node(id);
    println!("{}  ({:?}, {:?}/{})", info.id, info.class, info.os, info.arch);
    println!("  cpu   {} ({} cores / {} threads)", info.cpu_model, info.cpu_cores_physical, info.cpu_threads);
    println!("  ram   {:.1} GiB total, {:.1} GiB free", gib(info.ram_total_bytes), gib(info.ram_available_bytes));
    for a in &info.accelerators {
        println!("  accel {} ({:?}, {:.1} GiB{})", a.name, a.backend, gib(a.mem_bytes), if a.unified { ", unified" } else { "" });
    }

    let opts = if quick { bench::BenchOptions::quick() } else { bench::BenchOptions::standard() };
    println!("benchmarking ({}s per test)...", opts.test_secs);
    let report = bench::run(&opts, info.ram_available_bytes);
    println!("  cpu   {:.1} GFLOP/s single-thread, {:.1} GFLOP/s all threads", report.cpu_gflops_1t, report.cpu_gflops_mt);
    println!("  mem   {:.1} GB/s read", report.mem_bw_gbps);
    if let (Some(r), Some(w)) = (report.disk_read_mbps, report.disk_write_mbps) {
        println!("  disk  {r:.0} MB/s read, {w:.0} MB/s write");
    }
    if report.build == "debug" {
        eprintln!("warning: debug build, numbers are not representative. Use `cargo run --release`.");
    }

    let link = match coordinator {
        Some(addr) => {
            println!("measuring link to {addr}...");
            let l = net::measure(&addr, 50, if quick { 8 << 20 } else { 64 << 20 })?;
            print_link(&l);
            Some(l)
        }
        None => None,
    };

    let policy = Policy::default_for(info.class);
    let rep = NodeReport { info, policy, bench: Some(report), link };
    if let Some(path) = out {
        std::fs::write(&path, serde_json::to_string_pretty(&rep)?)?;
        println!("wrote {path}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_plan(files: Vec<String>, coordinator: Option<String>, ctx: u32, quant: &str, target: f64, model_file: Option<String>, topology: &str, json: bool) -> Result<()> {
    if files.is_empty() {
        bail!("give at least one NodeReport JSON file (from `clamor-muster probe --out`)");
    }
    let mut reports = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).with_context(|| format!("read {f}"))?;
        reports.push(serde_json::from_str::<NodeReport>(&text).with_context(|| format!("parse {f}"))?);
    }
    let coord = coordinator.map(NodeId).or_else(|| reports.first().map(|r| r.info.id.clone()));
    let nodes = plan::plan_nodes(&reports, coord.as_ref());
    let q = plan::Quant::parse(quant).context("quant must be q4, q6, q8 or f16")?;
    let models = match model_file {
        Some(p) => vec![serde_json::from_str::<plan::ModelSpec>(&std::fs::read_to_string(&p)?)?],
        None => plan::catalog(q),
    };
    let summary = plan::summarize(&reports, coord.as_ref());
    let topo = plan::Topology::parse(topology).context("topology must be ring or hub")?;
    let (plans, pick) = plan::recommend_with(&models, ctx, &nodes, target, topo);

    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "summary": summary, "plans": plans, "recommended": pick }))?);
        return Ok(());
    }

    println!("GROUP  {} devices, {} threads, {:.0} GiB RAM ({:.0} GiB free)", summary.devices, summary.cpu_threads, gib(summary.ram_total_bytes), gib(summary.ram_available_bytes));
    println!("       sum of memory bandwidth {:.0} GB/s, sum of compute {:.0} GFLOP/s", summary.mem_bw_sum_gbps, summary.gflops_sum);
    for w in &summary.warnings {
        println!("  ! {w}");
    }
    println!("\ncontext {ctx} tokens, quant {quant}, topology {topology}, coordinator {}\n", coord.map(|c| c.to_string()).unwrap_or_else(|| "-".into()));
    println!("{:<12} {:>8} {:>9} {:>9}  {:<7} {}", "model", "needs", "decode", "prefill", "stages", "bottleneck / notes");
    for (i, p) in plans.iter().enumerate() {
        if p.feasible {
            let stages: Vec<String> = p.stages.iter().map(|s| format!("{}[{}-{})", s.node, s.layer_start, s.layer_end)).collect();
            println!(
                "{:<12} {:>6.1}G {:>6.1}t/s {:>6.0}t/s  {:<7} {} | {}{}",
                p.model,
                p.need_bytes / (1u64 << 30) as f64,
                p.decode_tok_s,
                p.prefill_tok_s,
                p.stages.len(),
                p.bottleneck,
                stages.join(" "),
                if Some(i) == pick { "   <== recommended" } else { "" }
            );
        } else {
            println!("{:<12} {:>6.1}G  does not fit  ({})", p.model, p.need_bytes / (1u64 << 30) as f64, p.notes.join("; "));
        }
    }
    match pick {
        Some(i) => println!("\nrecommended: {} (>= {target} tok/s decode)", plans[i].model),
        None => println!("\nno model meets {target} tok/s decode; lower --target-tok-s or add devices"),
    }
    println!("estimates assume {:.0}% memory-bandwidth and {:.0}% compute efficiency; calibrate with llama-bench.", plan::MEM_EFFICIENCY * 100.0, plan::COMPUTE_EFFICIENCY * 100.0);
    Ok(())
}

fn print_link(l: &LinkStat) {
    println!("  rtt   p50 {:.2} ms, p95 {:.2} ms ({} samples)", l.rtt_ms_p50, l.rtt_ms_p95, l.samples);
    if let (Some(u), Some(d)) = (l.up_mbps, l.down_mbps) {
        println!("  link  {u:.0} Mbit/s up, {d:.0} Mbit/s down");
    }
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}
