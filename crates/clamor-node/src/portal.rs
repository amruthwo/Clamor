//! Web portal and JSON API served by the coordinator. Any device on the network can open it
//! in a browser, so a phone or tablet needs no app to monitor and steer the swarm.

use crate::coordinator::{NodeEntry, Shared};
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
        .route("/api/nodes/{id}/policy", post(set_policy))
        .route("/api/chat", post(chat_stub))
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
        for (id, e) in sw.nodes.iter_mut() {
            if let Some(tx) = &e.tx {
                if tx.send(Message::BenchRequest { quick: false }).is_ok() {
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

async fn set_policy(State(shared): State<Shared>, Path(id): Path<String>, Json(policy): Json<Policy>) -> ApiResult {
    let id = NodeId(id);
    let mut sw = shared.lock().unwrap();
    let entry = sw.nodes.get_mut(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown node {id}")))?;
    entry.report.policy = policy;
    if let Some(tx) = &entry.tx {
        let _ = tx.send(Message::SetPolicy { policy });
    }
    Ok(Json(json!({ "ok": true })))
}

async fn chat_stub() -> (StatusCode, Json<Value>) {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({ "error": "chat and task console arrive in v0.2" })))
}
