//! Web portal and JSON API served by the coordinator. Any device on the network can open it
//! in a browser, so a phone or tablet needs no app to monitor and steer the swarm.

use crate::coordinator::{NodeEntry, Shared};
use crate::pipeline::{self, PipelineRequest};
use clamor_transport::stage::{DoneReason, Event, HeadHandle};
use clamor_engine_ggml::{encode_chat, ChatMessage, Tokenizer};
use std::sync::Arc;
use std::time::{Duration, Instant};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use clamor_muster::{bench, plan, probe};
use clamor_proto::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

type ApiResult = Result<Json<Value>, (StatusCode, String)>;

pub async fn serve(bind: String, shared: Shared) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/roster", get(roster))
        .route("/api/plan", get(plan_handler))
        .route("/api/muster", post(muster))
        .route("/api/nodes/{id}/drain", post(drain))
        .route("/api/nodes/{id}/remove", post(remove))
        .route("/api/nodes/{id}/policy", post(set_policy))
        .route("/api/pipeline", get(pipeline_status).post(pipeline_start).delete(pipeline_stop))
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .with_state(shared);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    println!("web portal on http://{bind}/");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("portal.html"))
}

#[derive(Serialize)]
struct RosterRow {
    id: String,
    state: NodeState,
    is_coordinator: bool,
    last_seen_unix: u64,
    cpu_load: Option<f32>,
    running_tasks: u32,
    note: Option<String>,
    build: Option<BuildInfo>,
    update_available: Option<String>,
    models: Vec<ModelInfo>,
    share_port: Option<u16>,
    paused: Option<String>,
    settings_note: Option<String>,
    report: NodeReport,
}

fn row(entry: &NodeEntry, id: &NodeId, coordinator: &NodeId) -> RosterRow {
    RosterRow {
        id: id.to_string(),
        state: entry.state,
        is_coordinator: id == coordinator,
        last_seen_unix: entry.last_seen,
        cpu_load: entry.cpu_load,
        running_tasks: entry.running_tasks,
        note: entry.note.clone(),
        build: entry.build.clone(),
        update_available: entry.update_available.clone(),
        models: entry.models.clone(),
        share_port: entry.share_port,
        paused: entry.paused.clone(),
        settings_note: entry.settings_note.clone(),
        report: entry.report.clone(),
    }
}

async fn roster(State(shared): State<Shared>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let rows: Vec<RosterRow> = sw.nodes.iter().map(|(id, e)| row(e, id, &sw.coordinator)).collect();
    Json(json!({ "swarm": sw.name, "coordinator": sw.coordinator, "now": unix_now(), "nodes": rows }))
}

#[derive(Deserialize)]
struct PlanQuery {
    quant: Option<String>,
    ctx: Option<u32>,
    target: Option<f64>,
    topology: Option<String>,
}

/// Plan over the devices that are in the swarm right now and fully mustered.
async fn plan_handler(State(shared): State<Shared>, Query(q): Query<PlanQuery>) -> ApiResult {
    let quant = plan::Quant::parse(q.quant.as_deref().unwrap_or("q4")).ok_or((StatusCode::BAD_REQUEST, "quant must be q4, q6, q8 or f16".to_string()))?;
    let ctx = q.ctx.unwrap_or(8192).clamp(512, 1 << 20);
    let target = q.target.unwrap_or(2.0);
    let topo = plan::Topology::parse(q.topology.as_deref().unwrap_or("ring")).ok_or((StatusCode::BAD_REQUEST, "topology must be ring or hub".to_string()))?;

    let (reports, coordinator) = {
        let sw = shared.lock().unwrap();
        let reports: Vec<NodeReport> = sw
            .nodes
            .values()
            .filter(|e| matches!(e.state, NodeState::Ready | NodeState::Busy))
            .map(|e| e.report.clone())
            .collect();
        (reports, sw.coordinator.clone())
    };
    let nodes = plan::plan_nodes(&reports, Some(&coordinator));
    let (plans, pick) = plan::recommend_with(&plan::catalog(quant), ctx, &nodes, target, topo);
    Ok(Json(json!({
        "summary": plan::summarize(&reports),
        "plans": plans,
        "recommended": pick,
        "ctx": ctx,
        "target_tok_s": target,
        "assumptions": { "mem_efficiency": plan::MEM_EFFICIENCY, "compute_efficiency": plan::COMPUTE_EFFICIENCY },
    })))
}

/// Re-run the benchmarks and link measurements on every connected device, coordinator included.
async fn muster(State(shared): State<Shared>) -> ApiResult {
    let mut requested = 0;
    let coordinator = {
        let mut sw = shared.lock().unwrap();
        let coord = sw.coordinator.clone();
        let quick_bench = sw.quick_bench;
        for (id, e) in sw.nodes.iter_mut() {
            if let Some(tx) = &e.tx {
                if tx.send(Message::BenchRequest { quick: quick_bench }).is_ok() {
                    e.state = NodeState::Benchmarking;
                    requested += 1;
                }
            } else if *id == coord {
                e.state = NodeState::Benchmarking;
                requested += 1;
            }
        }
        coord
    };

    let shared2 = shared.clone();
    tokio::task::spawn_blocking(move || {
        let ram = probe::probe_node(coordinator.clone()).ram_available_bytes;
        let report = bench::run(&bench::BenchOptions::standard(), ram);
        let mut sw = shared2.lock().unwrap();
        if let Some(e) = sw.nodes.get_mut(&coordinator) {
            e.report.bench = Some(report);
            e.state = NodeState::Ready;
        }
    });
    Ok(Json(json!({ "requested": requested })))
}

async fn drain(State(shared): State<Shared>, Path(id): Path<String>) -> ApiResult {
    let id = NodeId(id);
    let mut sw = shared.lock().unwrap();
    if id == sw.coordinator {
        return Err((StatusCode::BAD_REQUEST, "the coordinator cannot drain itself".into()));
    }
    let entry = sw.nodes.get_mut(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown node {id}")))?;
    let tx = entry.tx.as_ref().ok_or((StatusCode::CONFLICT, format!("{id} is not connected")))?;
    tx.send(Message::Drain { grace_secs: 30 }).map_err(|_| (StatusCode::CONFLICT, "node connection closed".to_string()))?;
    entry.state = NodeState::Draining;
    entry.report.policy.participation = Participation::Draining;
    Ok(Json(json!({ "draining": id })))
}

/// Remove a device from the swarm: withdraw its approval (it needs the passphrase to join
/// again), let it finish its work and leave, and drop it from the roster.
async fn remove(State(shared): State<Shared>, Path(id): Path<String>) -> ApiResult {
    let id = NodeId(id);
    let mut sw = shared.lock().unwrap();
    if id == sw.coordinator {
        return Err((StatusCode::BAD_REQUEST, "the coordinator cannot remove itself".into()));
    }
    let entry = sw.nodes.get(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown node {id}")))?;
    let (key, tx) = (entry.key.clone(), entry.tx.clone());
    let revoked = key.as_deref().is_some_and(|k| sw.auth.revoke(k));
    if let Some(tx) = tx {
        let _ = tx.send(Message::Drain { grace_secs: 30 });
    }
    sw.node_lost(&id, "removed from the swarm");
    sw.nodes.remove(&id);
    sw.broadcast_approved();
    println!("{id} removed from the swarm{}", if revoked { "; its key is no longer approved" } else { "" });
    Ok(Json(json!({ "removed": id, "revoked": revoked })))
}

/// Change a device's settings. Pausing, battery and priority take effect within a heartbeat;
/// RAM and CPU shares apply from the next pipeline.
async fn set_policy(State(shared): State<Shared>, Path(id): Path<String>, Json(policy): Json<Policy>) -> ApiResult {
    crate::neighbour::validate(&policy).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let id = NodeId(id);
    let mut sw = shared.lock().unwrap();
    let is_coordinator = id == sw.coordinator;
    let entry = sw.nodes.get_mut(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown node {id}")))?;
    entry.report.policy = policy;
    if is_coordinator {
        // This device: remember here (pausing is picked up by monitor_self; priority and shares
        // apply to the next pipeline).
        crate::neighbour::save_settings(&policy);
        return Ok(Json(json!({ "ok": true, "note": entry.settings_note })));
    }
    if let Some(tx) = &entry.tx {
        let _ = tx.send(Message::SetPolicy { policy });
    }
    // A remote device reports any note with its next heartbeat; the stored one is stale.
    Ok(Json(json!({ "ok": true, "note": null })))
}

fn pipeline_json(p: &pipeline::Pipeline) -> Value {
    json!({ "pipeline": p, "ready": p.ready(), "failure": p.failure() })
}

async fn pipeline_status(State(shared): State<Shared>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    Json(sw.pipeline.as_ref().map(pipeline_json).unwrap_or(json!({ "pipeline": null, "ready": false })))
}

/// Set up a pipeline: `{"model": "x.gguf", "stages": [{"node": "a", "layers": [0, 16]}, ...]}`.
/// Without `stages`, the planner picks a layout from mustered devices. Returns at once; poll
/// GET /api/pipeline until `ready` (or `failure`).
async fn pipeline_start(State(shared): State<Shared>, Json(req): Json<PipelineRequest>) -> ApiResult {
    pipeline::start(&shared, req).await.map_err(|e| (StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let sw = shared.lock().unwrap();
    Ok(Json(sw.pipeline.as_ref().map(pipeline_json).unwrap_or(Value::Null)))
}

async fn pipeline_stop(State(shared): State<Shared>) -> Json<Value> {
    let mut sw = shared.lock().unwrap();
    let released = sw.pipeline.take().map(|p| {
        let epoch = p.epoch;
        pipeline::release(&mut sw, p);
        epoch
    });
    Json(json!({ "released_epoch": released }))
}

#[derive(Deserialize)]
struct GenerateRequest {
    /// Prompt token ids (a text interface arrives with the tokenizer).
    tokens: Vec<u32>,
    max_new: Option<u32>,
}

/// Cancels the request in the ring if the HTTP client goes away mid-generation.
struct CancelOnDrop {
    head: HeadHandle,
    req: u64,
    done: bool,
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.done {
            self.head.cancel(self.req);
        }
    }
}

/// The running pipeline's head, its epoch, context length and tokenizer, if it can take requests.
fn ready_pipeline(shared: &Shared) -> Result<(HeadHandle, u32, u32, Option<Arc<Tokenizer>>), (StatusCode, String)> {
    let sw = shared.lock().unwrap();
    let p = sw.pipeline.as_ref().ok_or((StatusCode::SERVICE_UNAVAILABLE, "no pipeline: POST /api/pipeline first".to_string()))?;
    if let Some(f) = p.failure() {
        return Err((StatusCode::SERVICE_UNAVAILABLE, format!("pipeline failed: {f}")));
    }
    if !p.ready() {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "pipeline is still loading".to_string()));
    }
    // A paused device (on battery, owner busy) takes no new requests; running ones finish.
    for s in &p.stages {
        if let Some(why) = sw.nodes.get(&s.node).and_then(|e| e.paused.as_ref()) {
            return Err((StatusCode::SERVICE_UNAVAILABLE, format!("paused: {} is not taking work right now ({why})", s.node)));
        }
    }
    Ok((p.head.clone().unwrap(), p.epoch, p.n_ctx, p.tokenizer.clone()))
}

struct Run {
    tokens: Vec<u32>,
    done: &'static str,
    error: Option<String>,
    ms: u64,
    first_token_ms: Option<u64>,
    decode_tok_s: Option<f64>,
}

/// Submit a prompt to the head and collect the reply.
async fn run(head: HeadHandle, prompt: Vec<u32>, max_new: u32) -> Run {
    let t0 = Instant::now();
    let (req, mut rx) = head.submit(prompt, max_new);
    let mut guard = CancelOnDrop { head, req, done: false };
    let mut tokens = Vec::new();
    let mut first_ms = None;
    let reason = loop {
        match tokio::time::timeout(Duration::from_secs(600), rx.recv()).await {
            Err(_) => break DoneReason::Failed("timed out after 600 s without a token".into()),
            Ok(None) => break DoneReason::Failed("pipeline stopped".into()),
            Ok(Some(Event::Token { id, .. })) => {
                first_ms.get_or_insert(t0.elapsed().as_millis() as u64);
                tokens.push(id);
            }
            Ok(Some(Event::Done(r))) => break r,
        }
    };
    guard.done = !matches!(reason, DoneReason::Failed(_));
    let ms = t0.elapsed().as_millis() as u64;
    let (done, error) = match reason {
        DoneReason::MaxTokens => ("max_tokens", None),
        DoneReason::Eos => ("eos", None),
        DoneReason::Cancelled => ("cancelled", None),
        DoneReason::Failed(e) => ("failed", Some(e)),
    };
    // Decode rate excludes the first token, which includes the whole prompt (prefill).
    let decode_tok_s = match (first_ms, tokens.len()) {
        (Some(f), n) if n > 1 && ms > f => Some((n - 1) as f64 / ((ms - f) as f64 / 1000.0)),
        _ => None,
    };
    Run { tokens, done, error, ms, first_token_ms: first_ms, decode_tok_s }
}

async fn generate(State(shared): State<Shared>, Json(g): Json<GenerateRequest>) -> ApiResult {
    let (head, epoch, _, _) = ready_pipeline(&shared)?;
    let r = run(head, g.tokens, g.max_new.unwrap_or(64).min(1 << 16)).await;
    Ok(Json(json!({ "tokens": r.tokens, "done": r.done, "error": r.error, "epoch": epoch, "ms": r.ms, "first_token_ms": r.first_token_ms, "decode_tok_s": r.decode_tok_s })))
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    max_new: Option<u32>,
}

/// `{"messages": [{"role": "user", "content": "Hi"}]}` -> the assistant's reply as text, using
/// the model's own chat template and tokenizer.
async fn chat(State(shared): State<Shared>, Json(c): Json<ChatRequest>) -> ApiResult {
    let (head, epoch, n_ctx, tok) = ready_pipeline(&shared)?;
    let tok = tok.ok_or((StatusCode::NOT_IMPLEMENTED, "this model's tokenizer is not supported yet; use /api/generate with token ids".to_string()))?;
    if c.messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages is empty".into()));
    }
    let prompt = encode_chat(&tok, &c.messages).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let room = n_ctx.saturating_sub(prompt.len() as u32);
    if room == 0 {
        return Err((StatusCode::BAD_REQUEST, format!("the conversation is {} tokens; the context holds {n_ctx}", prompt.len())));
    }
    let r = run(head, prompt.clone(), c.max_new.unwrap_or(512).min(room)).await;
    let text = tok.decode(&r.tokens, true);
    Ok(Json(json!({
        "message": { "role": "assistant", "content": text },
        "done": r.done, "error": r.error, "epoch": epoch,
        "prompt_tokens": prompt.len(), "completion_tokens": r.tokens.len(),
        "ms": r.ms, "first_token_ms": r.first_token_ms, "decode_tok_s": r.decode_tok_s,
    })))
}
