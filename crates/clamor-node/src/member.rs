//! Member node: joins a coordinator, runs benchmarks on request, heartbeats, and leaves
//! gracefully on `Drain` or Ctrl-C. Reconnects with backoff if the connection drops.

use crate::stage::{self, Assignment, EngineConfig};
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
    Loaded { model: String, epoch: u32, stage_index: u32, load_ms: u64, abort: AbortHandle },
    Failed { model: String, epoch: u32, stage_index: u32, error: String, stopped: bool },
}

/// This node's (at most one) pipeline stage. Dropping it stops the stage.
#[derive(Default)]
struct Slot {
    epoch: Option<u32>,
    loading: Option<AbortHandle>,
    running: Option<AbortHandle>,
}
impl Slot {
    fn stop(&mut self) {
        for h in [self.loading.take(), self.running.take()].into_iter().flatten() {
            h.abort();
        }
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
        match stage::start(&engine, &a).await {
            Ok(s) => {
                let _ = tx.send(StageEvent::Loaded { model: model.clone(), epoch, stage_index, load_ms: s.load_ms, abort: s.abort.clone() });
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
}

pub async fn join(addr: String, token: Option<String>, id: Option<String>, engine: EngineConfig) -> Result<()> {
    // Network-benchmark server the coordinator measures this device against.
    let listener = TcpListener::bind(("0.0.0.0", DEFAULT_BENCH_PORT)).or_else(|_| TcpListener::bind("0.0.0.0:0"))?;
    let bench_port = listener.local_addr()?.port();
    std::thread::spawn(move || {
        let _ = net::serve(listener);
    });

    let id = NodeId(id.unwrap_or_else(|| probe::default_node_id().0));
    let mut backoff = 2u64;
    loop {
        match session(&addr, &token, &id, bench_port, &mut backoff, &engine).await {
            Ok(Exit::Drained) => {
                println!("drained: left the swarm. Run `clamor-node join {addr}` to rejoin.");
                return Ok(());
            }
            Ok(Exit::Interrupted) => {
                println!("left the swarm.");
                return Ok(());
            }
            Ok(Exit::Rejected(reason)) => {
                eprintln!("coordinator rejected us: {reason}");
                std::process::exit(1);
            }
            Err(e) => eprintln!("disconnected: {e}; retrying in {backoff}s"),
        }
        sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(30);
    }
}

async fn session(addr: &str, token: &Option<String>, id: &NodeId, bench_port: u16, backoff: &mut u64, engine: &EngineConfig) -> Result<Exit> {
    let id2 = id.clone();
    let info = tokio::task::spawn_blocking(move || probe::probe_node(id2)).await?;
    let class = info.class;

    let stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    let (mut rd, mut wr) = stream.into_split();
    write_msg(&mut wr, &Message::Hello { proto: PROTOCOL_VERSION, info, bench_port, join_token: token.clone() }).await?;

    let heartbeat_secs = match timeout(Duration::from_secs(10), read_msg(&mut rd)).await?? {
        Message::Welcome { swarm, coordinator, heartbeat_secs } => {
            println!("joined swarm '{swarm}' (coordinator {coordinator})");
            *backoff = 2;
            heartbeat_secs
        }
        Message::Reject { reason } => return Ok(Exit::Rejected(reason)),
        other => bail!("unexpected reply to hello: {other:?}"),
    };

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

    let mut hb = interval(Duration::from_secs(heartbeat_secs.max(1) as u64));
    let mut sampler = probe::LoadSampler::new();
    let mut policy = Policy::default_for(class);
    let (stx, mut srx) = unbounded_channel::<StageEvent>();
    let mut slot = Slot::default();

    loop {
        tokio::select! {
            Some(ev) = srx.recv() => match ev {
                StageEvent::Loaded { model, epoch, stage_index, load_ms, abort } => {
                    if slot.epoch == Some(epoch) {
                        println!("stage {stage_index} of {model} ready (epoch {epoch}, loaded in {load_ms} ms)");
                        slot.loading = None;
                        slot.running = Some(abort);
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
                let (cpu, ram) = sampler.sample();
                write_msg(&mut wr, &Message::Heartbeat { cpu_load: cpu, ram_available_bytes: ram, power: None, running_tasks: 0 }).await?;
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
                        println!("benchmarking at the coordinator's request...");
                        let (_, ram) = sampler.sample();
                        let opts = if quick { bench::BenchOptions::quick() } else { bench::BenchOptions::standard() };
                        let report = tokio::task::spawn_blocking(move || bench::run(&opts, ram)).await?;
                        write_msg(&mut wr, &Message::BenchResult { report }).await?;
                    }
                    Message::SetPolicy { policy: p } => {
                        println!("policy updated: {p:?}");
                        policy = p;
                    }
                    Message::Ping { seq, sent_unix_ms } => {
                        write_msg(&mut wr, &Message::Pong { seq, sent_unix_ms }).await?;
                    }
                    Message::ShardAssign { model, epoch, session, stage_index, stage_count, layer_start, layer_end, listen_port, downstream, act_dtype, n_ctx, embed, sampling } => {
                        println!("assigned stage {stage_index}/{stage_count}: layers {layer_start}..{layer_end} of {model} (epoch {epoch})");
                        slot.stop(); // a new epoch supersedes whatever ran before
                        let a = Assignment { model, epoch, session, stage_index, stage_count, layer_start, layer_end, listen_port, downstream, act_dtype, n_ctx, embed, sampling };
                        slot.epoch = Some(epoch);
                        slot.loading = Some(spawn_stage(engine.clone(), a, stx.clone()));
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
                        write_msg(&mut wr, &Message::Drained).await?;
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
