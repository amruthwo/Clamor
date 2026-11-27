//! Web portal and JSON API served by the coordinator. Any device on the network can open it
//! in a browser, so a phone or tablet needs no app to monitor and steer the swarm.

use crate::coordinator::{NodeEntry, Shared};
use crate::pipeline::{self, PipelineRequest};
use clamor_transport::stage::HeadHandle;
use clamor_engine_ggml::{encode_chat, ChatMessage, Tokenizer};
use std::sync::Arc;
use std::time::Duration;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use crate::access::Caller;
use crate::people::Role;
use clamor_muster::{bench, plan, probe};
use clamor_proto::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

type ApiResult = Result<Json<Value>, (StatusCode, String)>;

/// Serve the dashboard on `bind` (plain HTTP) and on `tls_bind` (HTTPS with the coordinator's
/// certificate; any free port if that one is taken). Members' apps use the HTTPS one.
pub async fn serve(bind: String, tls_bind: String, shared: Shared) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/icon.svg", get(icon))
        .route("/api/roster", get(roster))
        .route("/api/plan", get(plan_handler))
        .route("/api/catalog", get(catalog_handler))
        .route("/api/models/download", post(download_model))
        .route("/api/models/cancel", post(cancel_download))
        .route("/api/muster", post(muster))
        .route("/api/nodes/{id}/drain", post(drain))
        .route("/api/nodes/{id}/remove", post(remove))
        .route("/api/invite", post(invite))
        .route("/api/nodes/{id}/policy", post(set_policy))
        .route("/api/nodes/{id}/name", post(set_device_name))
        .route("/api/pipeline", get(pipeline_status).post(pipeline_start).delete(pipeline_stop))
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/me", get(me))
        .route("/api/me/pin", post(set_pin))
        .route("/api/logout", post(logout))
        .route("/api/login/start", post(login_start))
        .route("/api/login/poll", get(login_poll))
        .route("/api/login/pin", post(login_pin))
        .route("/api/login/pending", get(login_pending))
        .route("/api/login/approve", post(login_approve))
        .route("/api/people", get(people))
        .route("/api/people/{id}/role", post(person_role))
        .route("/api/people/{id}/remove", post(person_remove))
        .route("/api/nodes/{id}/person", post(node_person))
        .route("/api/joins", get(joins))
        .route("/api/overview", get(overview))
        .route("/api/chats", get(list_chats).post(new_chat))
        .route("/api/chats/{id}", get(get_chat).delete(delete_chat))
        .route("/api/chats/{id}/messages", post(send_chat))
        .route("/api/chats/{id}/stop", post(stop_chat))
        .route("/api/me/prefs", post(set_prefs))
        .route("/api/joins/{id}/{decision}", post(decide_join))
        .layer(axum::middleware::from_fn_with_state(shared.clone(), crate::access::layer))
        .with_state(shared.clone());

    // HTTPS, with the coordinator's own certificate.
    let tls_listener = match tokio::net::TcpListener::bind(&tls_bind).await {
        Ok(l) => l,
        Err(_) => tokio::net::TcpListener::bind((tls_bind.rsplit_once(':').map(|(h, _)| h).unwrap_or("0.0.0.0"), 0)).await?,
    };
    let tls_port = tls_listener.local_addr()?.port();
    let server_cfg = {
        let mut sw = shared.lock().unwrap();
        sw.portal_tls_port = Some(tls_port);
        clamor_transport::tls::server_config(&sw.tls, None)?
    };
    let tls_app = app.clone().layer(axum::Extension(crate::access::Via::Tls));
    tokio::spawn(async move {
        let l = TlsListener { tcp: tls_listener, acceptor: tokio_rustls::TlsAcceptor::from(server_cfg) };
        if let Err(e) = axum::serve(l, tls_app.into_make_service_with_connect_info::<TlsPeer>()).await {
            eprintln!("encrypted dashboard stopped: {e}");
        }
    });

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    println!("web portal on http://{bind}/ (encrypted: https, port {tls_port})");
    axum::serve(listener, app.layer(axum::Extension(crate::access::Via::Plain)).into_make_service_with_connect_info::<std::net::SocketAddr>()).await?;
    Ok(())
}

/// The client's address on the HTTPS dashboard (axum's own covers only plain TCP listeners).
#[derive(Clone, Copy, Debug)]
pub struct TlsPeer(pub std::net::SocketAddr);

impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, TlsListener>> for TlsPeer {
    fn connect_info(s: axum::serve::IncomingStream<'_, TlsListener>) -> Self {
        TlsPeer(*s.remote_addr())
    }
}

/// The dashboard's HTTPS listener: TCP accept, then TLS. A client that never finishes the TLS
/// handshake is dropped after 10 s without holding up the others for longer.
pub struct TlsListener {
    tcp: tokio::net::TcpListener,
    acceptor: tokio_rustls::TlsAcceptor,
}

impl axum::serve::Listener for TlsListener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = std::net::SocketAddr;
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((s, addr)) = self.tcp.accept().await else {
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            };
            if let Ok(Ok(t)) = tokio::time::timeout(Duration::from_secs(10), self.acceptor.accept(s)).await {
                return (t, addr);
            }
        }
    }
    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.tcp.local_addr()
    }
}

/// The dashboard, with its icons (Material Symbols, shipped inside Clamor: no network needed).
async fn index() -> Html<&'static str> {
    static PAGE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| include_str!("portal.html").replacen("<!--ICONS-->", include_str!("icons.svg"), 1));
    Html(PAGE.as_str())
}

/// The Clamor icon (header and browser tab).
async fn icon() -> impl axum::response::IntoResponse {
    ([(axum::http::header::CONTENT_TYPE, "image/svg+xml"), (axum::http::header::CACHE_CONTROL, "max-age=86400")], include_str!("../../../assets/clamor-icon.svg"))
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
    /// The person the device belongs to (id, name).
    person: Option<(String, String)>,
    /// The name someone gave it (shown instead of the id).
    label: Option<String>,
}

fn row(entry: &NodeEntry, id: &NodeId, coordinator: &NodeId, sw: &crate::coordinator::Swarm) -> RosterRow {
    let person = device_person(sw, id, entry).and_then(|p| sw.auth.profile.person(&p)).map(|p| (p.id.clone(), p.name.clone()));
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
        person,
        label: sw.auth.profile.device_names.get(&id.0).cloned(),
    }
}

/// The person a device belongs to: the owner for the coordinator itself, else by its key.
fn device_person(sw: &crate::coordinator::Swarm, id: &NodeId, entry: &NodeEntry) -> Option<String> {
    if *id == sw.coordinator {
        return sw.auth.profile.people.iter().find(|p| p.role == crate::people::Role::Owner).map(|p| p.id.clone());
    }
    entry.key.as_deref().and_then(|k| sw.auth.profile.approved.iter().find(|d| d.key == k)).map(|d| d.person.clone())
}

async fn roster(State(shared): State<Shared>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let rows: Vec<RosterRow> = sw.nodes.iter().map(|(id, e)| row(e, id, &sw.coordinator, &sw)).collect();
    Json(json!({ "swarm": sw.name, "coordinator": sw.coordinator, "now": unix_now(), "nodes": rows }))
}

#[derive(Deserialize)]
struct PlanQuery {
    quant: Option<String>,
    ctx: Option<u32>,
    target: Option<f64>,
    topology: Option<String>,
}

#[derive(Deserialize)]
struct CatalogQuery {
    quant: Option<String>,
    ctx: Option<u32>,
}

/// The model picker: every model in Clamor's catalog, whether Clamor runs it, how it would run
/// on this swarm right now (fits? how fast? over how many devices?), whether the swarm already
/// has it, and downloads under way.
async fn catalog_handler(State(shared): State<Shared>, Query(q): Query<CatalogQuery>) -> Json<Value> {
    let quant = q.quant.unwrap_or_else(|| "q8".into());
    let ctx = q.ctx.unwrap_or(8192).clamp(512, 1 << 20);
    let (reports, coordinator, files) = {
        let sw = shared.lock().unwrap();
        let reports: Vec<NodeReport> = sw.nodes.values().filter(|e| matches!(e.state, NodeState::Ready | NodeState::Busy)).map(|e| e.report.clone()).collect();
        let files: Vec<(String, String)> = sw.nodes.iter().flat_map(|(id, e)| e.models.iter().map(move |m| (m.name.clone(), id.0.clone()))).collect();
        (reports, sw.coordinator.clone(), files)
    };
    let nodes = plan::plan_nodes(&reports, Some(&coordinator));
    let downloads = crate::models::downloads();
    let models: Vec<Value> = crate::models::catalog()
        .iter()
        .map(|m| {
            let f = m.file(&quant);
            let p = plan::plan_model_with(&m.spec(f), ctx, &nodes, plan::Topology::Ring);
            let have: Vec<Value> = m.files.iter().filter_map(|cf| {
                let on: Vec<&String> = files.iter().filter(|(n, _)| *n == cf.file).map(|(_, id)| id).collect();
                (!on.is_empty()).then(|| json!({ "file": cf.file, "quant": cf.quant, "on": on }))
            }).collect();
            let why = m.support().err();
            json!({
                "id": m.id, "name": m.name, "maker": m.maker, "license": m.license, "about": m.about, "repo": m.repo,
                "arch": m.arch, "params_b": m.params_b, "context": m.context, "caps": m.caps,
                "runs": why.is_none(), "why": why,
                "file": f, "quants": m.files.iter().map(|f| &f.quant).collect::<Vec<_>>(),
                "fit": { "feasible": p.feasible, "decode_tok_s": p.decode_tok_s, "prefill_tok_s": p.prefill_tok_s, "devices": p.stages.len(), "need_bytes": p.need_bytes, "notes": p.notes },
                "have": have,
                "download": downloads.iter().find(|d| d.model == m.id),
            })
        })
        .collect();
    Json(json!({ "quant": quant, "ctx": ctx, "models": models }))
}

#[derive(Deserialize)]
struct DownloadReq {
    id: String,
    quant: Option<String>,
}

/// Download a catalog model to the coordinator (admins). The other devices fetch the layers they
/// run from it, as for any model.
async fn download_model(State(shared): State<Shared>, Json(req): Json<DownloadReq>) -> ApiResult {
    let m = crate::models::find(&req.id).ok_or((StatusCode::NOT_FOUND, format!("no model {:?} in the catalog", req.id)))?;
    if let Err(why) = m.support() {
        return Err((StatusCode::BAD_REQUEST, format!("Clamor cannot run {} yet: {why}", m.name)));
    }
    let (dir, coordinator) = {
        let sw = shared.lock().unwrap();
        (sw.engine.models_dir.clone(), sw.coordinator.clone())
    };
    let (dir2, shared2) = (dir.clone(), shared.clone());
    let quant = req.quant.unwrap_or_else(|| "q8".into());
    let f = crate::models::start(m, &quant, &dir, move || {
        // The coordinator now has it: tell the swarm (and the dashboard).
        let found = crate::share::scan(&dir2);
        if let Some(e) = shared2.lock().unwrap().nodes.get_mut(&coordinator) {
            e.models = found;
        }
    })
    .map_err(|e| (StatusCode::CONFLICT, format!("{e:#}")))?;
    Ok(Json(json!({ "ok": true, "file": f.file, "size": f.size })))
}

#[derive(Deserialize)]
struct CancelReq {
    file: String,
}

async fn cancel_download(Json(req): Json<CancelReq>) -> ApiResult {
    if crate::models::cancel(&req.file) {
        Ok(Json(json!({ "ok": true })))
    } else {
        Err((StatusCode::NOT_FOUND, format!("{} is not downloading", req.file)))
    }
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
        "summary": plan::summarize(&reports, Some(&coordinator)),
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

#[derive(Deserialize, Default)]
struct InviteReq {
    /// "device": another device of the caller ("Add my device"); "person": someone new (admins).
    #[serde(default)]
    kind: Option<String>,
}

/// A one-time code for adding a device: one of the caller's own ("Add my device", anyone), or
/// a new person's (admins).
async fn invite(State(shared): State<Shared>, Extension(caller): Extension<Caller>, body: axum::body::Bytes) -> ApiResult {
    // An empty body (`clamor-node invite`) means a new person.
    let req: InviteReq = if body.iter().all(u8::is_ascii_whitespace) { InviteReq::default() } else { serde_json::from_slice(&body).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))? };
    let kind = req.kind.unwrap_or_else(|| "person".into());
    let person = match kind.as_str() {
        "device" => Some(caller.person.clone()),
        "person" if caller.role >= Role::Admin => None,
        "person" => return Err((StatusCode::FORBIDDEN, "only an admin can invite a new person; you can add your own devices".into())),
        _ => return Err((StatusCode::BAD_REQUEST, "kind is device or person".into())),
    };
    let mut sw = shared.lock().unwrap();
    let mut inv = crate::coordinator::Invite::new();
    inv.person = person;
    let code = inv.code.clone();
    sw.invite = Some(inv);
    let hint = if sw.join_hint.is_empty() { String::new() } else { format!("{} ", sw.join_hint) };
    println!("one-time code made for adding a device (valid {} minutes)", crate::coordinator::INVITE_SECS / 60);
    let link = join_link(&sw.name, &sw.auth.profile.swarm_id, &sw.join_hint, &code);
    Ok(Json(json!({ "code": code, "expires_in": crate::coordinator::INVITE_SECS, "swarm": sw.name, "command": format!("clamor-node join {hint}--code {code}"), "link": link, "qr_svg": qr_svg(&link) })))
}

/// What a QR code from "Add device" holds: the swarm, where its coordinator is, and the one-time
/// code (the code proves both sides to each other, as when it is typed).
pub fn join_link(swarm: &str, swarm_id: &str, addr: &str, code: &str) -> String {
    let enc = |s: &str| s.bytes().map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b':') { (b as char).to_string() } else { format!("%{b:02X}") }).collect::<String>();
    format!("clamor://join?swarm={}&id={}&addr={}&code={}", enc(swarm), enc(swarm_id), enc(addr), enc(code))
}

/// A QR code of `text`, as an SVG (black on white, with the quiet zone scanners need).
fn qr_svg(text: &str) -> Option<String> {
    use qrcode::render::svg;
    let code = qrcode::QrCode::with_error_correction_level(text.as_bytes(), qrcode::EcLevel::M).ok()?;
    Some(code.render::<svg::Color>().min_dimensions(220, 220).dark_color(svg::Color("#000000")).light_color(svg::Color("#ffffff")).quiet_zone(true).build())
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
#[derive(Deserialize)]
struct NameReq {
    name: String,
}

/// Give a device a name to show (its owner, or an admin). An empty name goes back to the id.
async fn set_device_name(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>, Json(req): Json<NameReq>) -> ApiResult {
    let name: String = req.name.trim().chars().filter(|c| !c.is_control()).collect();
    if name.chars().count() > 40 {
        return Err((StatusCode::BAD_REQUEST, "a device name is at most 40 characters".into()));
    }
    let id = NodeId(id);
    let mut sw = shared.lock().unwrap();
    let entry = sw.nodes.get(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown device {id}")))?;
    if caller.role < Role::Admin && device_person(&sw, &id, entry).as_deref() != Some(caller.person.as_str()) {
        return Err((StatusCode::FORBIDDEN, "only the device's owner (or an admin) can rename it".into()));
    }
    let names = &mut sw.auth.profile.device_names;
    if name.is_empty() || name == id.0 {
        names.remove(&id.0);
    } else {
        names.insert(id.0.clone(), name.clone());
    }
    sw.auth.save();
    Ok(Json(json!({ "ok": true, "name": if name.is_empty() { id.0.clone() } else { name } })))
}

async fn set_policy(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>, Json(policy): Json<Policy>) -> ApiResult {
    crate::neighbour::validate(&policy).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let id = NodeId(id);
    let mut sw = shared.lock().unwrap();
    // A device's settings are its owner's to change (or an admin's).
    if caller.role < Role::Admin {
        let entry = sw.nodes.get(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown node {id}")))?;
        if device_person(&sw, &id, entry).as_deref() != Some(caller.person.as_str()) {
            return Err((StatusCode::FORBIDDEN, "only the device's owner (or an admin) can change its settings".into()));
        }
    }
    let is_coordinator = id == sw.coordinator;
    let swarm_id = sw.auth.profile.swarm_id.clone();
    let entry = sw.nodes.get_mut(&id).ok_or((StatusCode::NOT_FOUND, format!("unknown node {id}")))?;
    entry.report.policy = policy;
    if is_coordinator {
        // This device: remember here (pausing is picked up by monitor_self; priority and shares
        // apply to the next pipeline).
        crate::neighbour::save_settings(&swarm_id, &policy);
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




async fn generate(State(shared): State<Shared>, Json(g): Json<GenerateRequest>) -> ApiResult {
    let (head, epoch, _, _) = ready_pipeline(&shared)?;
    let r = crate::generate::run(head, g.tokens, g.max_new.unwrap_or(64).min(1 << 16)).await;
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
    let r = crate::generate::run(head, prompt.clone(), c.max_new.unwrap_or(512).min(room)).await;
    let text = tok.decode(&r.tokens, true);
    Ok(Json(json!({
        "message": { "role": "assistant", "content": text },
        "done": r.done, "error": r.error, "epoch": epoch,
        "prompt_tokens": prompt.len(), "completion_tokens": r.tokens.len(),
        "ms": r.ms, "first_token_ms": r.first_token_ms, "decode_tok_s": r.decode_tok_s,
    })))
}

// ---------------------------------------------------------------- signing in, and people

fn person_json(sw: &crate::coordinator::Swarm, p: &crate::people::Person) -> Value {
    let devices: Vec<String> = sw.auth.profile.approved.iter().filter(|d| d.person == p.id).map(|d| d.name.clone()).collect();
    let devices = if p.role == Role::Owner { std::iter::once(sw.coordinator.0.clone()).chain(devices).collect() } else { devices };
    json!({ "id": p.id, "name": p.name, "role": p.role, "has_pin": p.pin.is_some(), "devices": devices })
}

async fn me(State(shared): State<Shared>, Extension(caller): Extension<Caller>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let pending = if caller.how != "session" || true { sw.auth.signins.iter().filter(|s| s.approved_by.is_none() && s.created_unix + crate::people::SIGNIN_SECS > unix_now()).count() } else { 0 };
    let has_pin = sw.auth.profile.person(&caller.person).is_some_and(|p| p.pin.is_some());
    let reply_style = sw.auth.profile.person(&caller.person).and_then(|p| p.reply_style.clone()).unwrap_or_else(|| "stream".into());
    Json(json!({ "id": caller.person, "name": caller.name, "role": caller.role, "how": caller.how, "device": caller.device, "has_pin": has_pin, "pending_signins": pending, "swarm": sw.name, "reply_style": reply_style }))
}

#[derive(Deserialize)]
struct PinReq {
    pin: String,
}

async fn set_pin(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Json(req): Json<PinReq>) -> ApiResult {
    crate::people::valid_pin(&req.pin).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let hashed = tokio::task::spawn_blocking(move || crate::people::hash_pin(&req.pin)).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut sw = shared.lock().unwrap();
    let auth = &mut sw.auth;
    let p = auth.profile.people.iter_mut().find(|p| p.id == caller.person).ok_or((StatusCode::NOT_FOUND, "no such person".to_string()))?;
    p.pin = Some(hashed);
    auth.save();
    Ok(Json(json!({ "ok": true })))
}

fn cookie_header(token: &str, via: crate::access::Via) -> String {
    let secure = if via == crate::access::Via::Tls { "; Secure" } else { "" };
    format!("{}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={}{secure}", crate::access::SESSION_COOKIE, crate::people::SESSION_DAYS * 86_400)
}

async fn logout(State(shared): State<Shared>, headers: axum::http::HeaderMap) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(t) = crate::access::session_token(&headers) {
        let mut sw = shared.lock().unwrap();
        let (dir, id) = (sw.auth.config_dir.clone(), sw.auth.profile.swarm_id.clone());
        sw.auth.sessions.end(&t, &dir, &id);
    }
    ([(axum::http::header::SET_COOKIE, format!("{}=; Path=/; Max-Age=0", crate::access::SESSION_COOKIE))], Json(json!({ "ok": true }))).into_response()
}

/// A browser asks to sign in: it gets a code to show, which someone signed in approves.
async fn login_start(State(shared): State<Shared>, Extension(peer): Extension<std::net::SocketAddr>) -> Json<Value> {
    let mut sw = shared.lock().unwrap();
    let now = unix_now();
    sw.auth.signins.retain(|s| s.created_unix + crate::people::SIGNIN_SECS > now);
    sw.auth.signins.truncate(20);
    let s = crate::people::PendingSignIn { id: hex::encode(crate::auth::random_bytes::<16>()), code: crate::people::six_digits(), created_unix: now, approved_by: None, from: peer.ip().to_string() };
    let out = json!({ "id": s.id, "code": s.code, "expires_in": crate::people::SIGNIN_SECS });
    sw.auth.signins.push(s);
    Json(out)
}

#[derive(Deserialize)]
struct PollQuery {
    id: String,
}

/// The browser waits for approval; once approved it gets its session cookie (once).
async fn login_poll(State(shared): State<Shared>, Extension(via): Extension<crate::access::Via>, Query(q): Query<PollQuery>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let mut sw = shared.lock().unwrap();
    let now = unix_now();
    let Some(i) = sw.auth.signins.iter().position(|s| s.id == q.id && s.created_unix + crate::people::SIGNIN_SECS > now) else {
        return (StatusCode::GONE, "this sign-in code has expired: start again").into_response();
    };
    let Some(person) = sw.auth.signins[i].approved_by.clone() else {
        return Json(json!({ "approved": false })).into_response();
    };
    sw.auth.signins.remove(i);
    let (dir, sid) = (sw.auth.config_dir.clone(), sw.auth.profile.swarm_id.clone());
    let token = sw.auth.sessions.create(&person, now, &dir, &sid);
    ([(axum::http::header::SET_COOKIE, cookie_header(&token, via))], Json(json!({ "approved": true }))).into_response()
}

#[derive(Deserialize)]
struct PinLogin {
    name: String,
    pin: String,
}

async fn login_pin(State(shared): State<Shared>, Extension(via): Extension<crate::access::Via>, Json(req): Json<PinLogin>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let now = unix_now();
    let found = {
        let sw = shared.lock().unwrap();
        sw.auth.profile.people.iter().find(|p| p.name.eq_ignore_ascii_case(req.name.trim())).map(|p| (p.id.clone(), p.pin.clone(), sw.auth.pins.locked(&p.id, now)))
    };
    let Some((person, Some(stored), locked)) = found else {
        return (StatusCode::UNAUTHORIZED, "no one by that name has a PIN: sign in with a code instead").into_response();
    };
    if locked {
        return (StatusCode::TOO_MANY_REQUESTS, "too many wrong PINs: wait 15 minutes, or sign in with a code").into_response();
    }
    let pin = req.pin.clone();
    let ok = tokio::task::spawn_blocking(move || crate::people::check_pin(&stored, &pin)).await.unwrap_or(false);
    let mut sw = shared.lock().unwrap();
    if !ok {
        sw.auth.pins.failed(&person, now);
        return (StatusCode::UNAUTHORIZED, "wrong PIN").into_response();
    }
    sw.auth.pins.succeeded(&person);
    let (dir, sid) = (sw.auth.config_dir.clone(), sw.auth.profile.swarm_id.clone());
    let token = sw.auth.sessions.create(&person, now, &dir, &sid);
    ([(axum::http::header::SET_COOKIE, cookie_header(&token, via))], Json(json!({ "ok": true }))).into_response()
}

/// Sign-in codes waiting for approval (for the "Approve sign-in" prompt).
async fn login_pending(State(shared): State<Shared>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let now = unix_now();
    let list: Vec<Value> = sw.auth.signins.iter().filter(|s| s.approved_by.is_none() && s.created_unix + crate::people::SIGNIN_SECS > now).map(|s| json!({ "id": s.id, "code": s.code, "from": s.from, "age_secs": now - s.created_unix })).collect();
    Json(json!({ "pending": list }))
}

#[derive(Deserialize)]
struct ApproveReq {
    /// The code the browser shows (what the person compares).
    code: String,
}

/// Approve a browser's sign-in: it signs in as the person approving.
async fn login_approve(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Json(req): Json<ApproveReq>) -> ApiResult {
    let mut sw = shared.lock().unwrap();
    let now = unix_now();
    let want: String = req.code.chars().filter(|c| c.is_ascii_digit()).collect();
    let s = sw
        .auth
        .signins
        .iter_mut()
        .find(|s| s.approved_by.is_none() && s.created_unix + crate::people::SIGNIN_SECS > now && s.code.chars().filter(|c| c.is_ascii_digit()).collect::<String>() == want)
        .ok_or((StatusCode::NOT_FOUND, "no browser is waiting with that code".to_string()))?;
    s.approved_by = Some(caller.person.clone());
    println!("{} approved a sign-in from {}", caller.name, s.from);
    Ok(Json(json!({ "ok": true })))
}

async fn people(State(shared): State<Shared>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let list: Vec<Value> = sw.auth.profile.people.iter().map(|p| person_json(&sw, p)).collect();
    Json(json!({ "people": list }))
}

#[derive(Deserialize)]
struct RoleReq {
    role: Role,
}

/// The owner makes someone an admin, or a member again. There is one owner.
async fn person_role(State(shared): State<Shared>, Path(id): Path<String>, Json(req): Json<RoleReq>) -> ApiResult {
    if req.role == Role::Owner {
        return Err((StatusCode::BAD_REQUEST, "a swarm has one owner: whoever runs its coordinator".into()));
    }
    let mut sw = shared.lock().unwrap();
    let p = sw.auth.profile.people.iter_mut().find(|p| p.id == id).ok_or((StatusCode::NOT_FOUND, "no such person".to_string()))?;
    if p.role == Role::Owner {
        return Err((StatusCode::BAD_REQUEST, "the owner's role cannot change".into()));
    }
    p.role = req.role;
    let name = p.name.clone();
    sw.auth.save();
    println!("{name} is now {:?}", req.role);
    Ok(Json(json!({ "ok": true })))
}

/// Remove a person: all their devices lose their approval and leave; their sessions end.
async fn person_remove(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult {
    let mut sw = shared.lock().unwrap();
    let target = sw.auth.profile.person(&id).cloned().ok_or((StatusCode::NOT_FOUND, "no such person".to_string()))?;
    if target.role == Role::Owner {
        return Err((StatusCode::BAD_REQUEST, "the owner cannot be removed".into()));
    }
    if target.role == Role::Admin && caller.role < Role::Owner {
        return Err((StatusCode::FORBIDDEN, "only the owner can remove an admin".into()));
    }
    let keys = sw.auth.remove_person(&id);
    sw.chats.remove_person(&id);
    let gone: Vec<NodeId> = sw.nodes.iter().filter(|(_, e)| e.key.as_ref().is_some_and(|k| keys.contains(k))).map(|(n, _)| n.clone()).collect();
    for n in &gone {
        if let Some(tx) = sw.nodes.get(n).and_then(|e| e.tx.clone()) {
            let _ = tx.send(Message::Drain { grace_secs: 30 });
        }
        sw.node_lost(n, "its owner was removed from the swarm");
        sw.nodes.remove(n);
    }
    sw.broadcast_approved();
    println!("{} removed from the swarm ({} device(s))", target.name, keys.len());
    Ok(Json(json!({ "removed": id, "devices": gone })))
}

#[derive(Deserialize)]
struct NodePersonReq {
    person: String,
}

/// Move a device to another person (e.g. it joined with the passphrase as a "new person").
async fn node_person(State(shared): State<Shared>, Path(id): Path<String>, Json(req): Json<NodePersonReq>) -> ApiResult {
    let mut sw = shared.lock().unwrap();
    if sw.auth.profile.person(&req.person).is_none() {
        return Err((StatusCode::NOT_FOUND, "no such person".into()));
    }
    let key = sw.nodes.get(&NodeId(id.clone())).and_then(|e| e.key.clone()).ok_or((StatusCode::BAD_REQUEST, "only a member device can be moved".to_string()))?;
    let auth = &mut sw.auth;
    let d = auth.profile.approved.iter_mut().find(|d| d.key == key).ok_or((StatusCode::NOT_FOUND, "device not approved".to_string()))?;
    let old = std::mem::replace(&mut d.person, req.person.clone());
    // A person left with no devices (and no PIN) disappears.
    if !auth.profile.approved.iter().any(|d| d.person == old) && auth.profile.person(&old).is_some_and(|p| p.role == Role::Member && p.pin.is_none()) {
        auth.profile.people.retain(|p| p.id != old);
    }
    auth.save();
    Ok(Json(json!({ "ok": true })))
}

/// Devices asking to join (admins approve them after comparing the check code).
async fn joins(State(shared): State<Shared>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let now = unix_now();
    let list: Vec<&crate::coordinator::PendingJoin> = sw.joins.iter().filter(|j| j.decision.is_none() && j.created_unix + crate::coordinator::ASK_SECS > now).collect();
    Json(json!({ "joins": list }))
}

async fn decide_join(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path((id, decision)): Path<(String, String)>) -> ApiResult {
    let yes = match decision.as_str() {
        "approve" => true,
        "decline" => false,
        _ => return Err((StatusCode::NOT_FOUND, "approve or decline".into())),
    };
    let mut sw = shared.lock().unwrap();
    let j = sw.joins.iter_mut().find(|j| j.id == id && j.decision.is_none()).ok_or((StatusCode::NOT_FOUND, "no such request (it may have timed out)".to_string()))?;
    j.decision = Some(yes);
    println!("{} {} {}'s request to join", caller.name, if yes { "approved" } else { "declined" }, j.device);
    Ok(Json(json!({ "ok": true })))
}

// ---------------------------------------------------------------- conversations

async fn list_chats(State(shared): State<Shared>, Extension(caller): Extension<Caller>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let g = sw.chats.get(crate::chats::GROUP);
    Json(json!({ "conversations": sw.chats.list(&caller.person), "group": { "id": crate::chats::GROUP, "updated": g.map(|c| c.updated), "count": g.map(|c| c.messages.len()) } }))
}

async fn new_chat(State(shared): State<Shared>, Extension(caller): Extension<Caller>) -> Json<Value> {
    let id = shared.lock().unwrap().chats.create(&caller.person, unix_now());
    Json(json!({ "id": id }))
}

fn open_chat(sw: &crate::coordinator::Swarm, id: &str, caller: &Caller) -> Result<crate::chats::Conversation, (StatusCode, String)> {
    if !sw.chats.can_open(id, &caller.person) {
        return Err((StatusCode::NOT_FOUND, "no such conversation".into()));
    }
    Ok(sw.chats.get(id).cloned().expect("checked"))
}

async fn get_chat(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult {
    let sw = shared.lock().unwrap();
    let mut v = serde_json::to_value(open_chat(&sw, &id, &caller)?).unwrap();
    // A reply being written right now (e.g. the page was reloaded, or someone else asked).
    if let Some(g) = sw.generating.get(&id) {
        v["live"] = json!({ "text": g.progress.borrow().text, "asker": g.asker });
    }
    Ok(Json(v))
}

async fn delete_chat(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult {
    let mut sw = shared.lock().unwrap();
    let c = open_chat(&sw, &id, &caller)?;
    if c.owner.is_none() {
        return Err((StatusCode::BAD_REQUEST, "the group conversation cannot be deleted".into()));
    }
    sw.chats.delete(&id);
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct SendReq {
    content: String,
}




/// Add a message to a conversation; the model's reply comes back as it is written, one JSON
/// object per line (see `generate::progress_body`). The reply is written (and saved) even if
/// the asker goes away; POST .../stop ends it early.
async fn send_chat(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>, Json(req): Json<SendReq>) -> Result<axum::response::Response, (StatusCode, String)> {
    use crate::generate::{chat_prompt, failed, progress_body, spawn_reply, Generation, Progress};
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "say something".into()));
    }
    let conv = {
        let mut sw = shared.lock().unwrap();
        open_chat(&sw, &id, &caller)?;
        if sw.generating.contains_key(&id) {
            return Err((StatusCode::CONFLICT, "the model is still answering the last message".into()));
        }
        sw.chats.append(&id, crate::chats::ChatMsg { role: "user".into(), content, by: Some(caller.name.clone()), at: unix_now(), tok_s: None, error: None, stopped: false });
        sw.chats.get(&id).cloned().expect("exists")
    };
    let (tx, rx) = tokio::sync::watch::channel(Progress::default());
    let stop = Arc::new(tokio::sync::Notify::new());
    let setup = ready_pipeline(&shared).map_err(|(_, e)| e).and_then(|(head, _, n_ctx, tok)| {
        let tok = tok.ok_or_else(|| "this model's tokenizer is not supported yet".to_string())?;
        let prompt = chat_prompt(&tok, &conv, n_ctx)?;
        Ok((head, tok, n_ctx, prompt))
    });
    match setup {
        Err(e) => {
            let msg = failed(e);
            shared.lock().unwrap().chats.append(&id, msg.clone());
            let _ = tx.send(Progress { done: Some(msg), ..Default::default() });
        }
        Ok((head, tok, n_ctx, prompt)) => {
            shared.lock().unwrap().generating.insert(id.clone(), Generation { asker: caller.person.clone(), stop: stop.clone(), progress: rx.clone() });
            let (shared2, id2) = (shared.clone(), id.clone());
            spawn_reply(head, tok, n_ctx, prompt, conv, stop, tx, move |msg| {
                let mut sw = shared2.lock().unwrap();
                if let Some(s) = msg.tok_s {
                    sw.speeds.push_back((msg.at, s));
                    while sw.speeds.len() > 30 {
                        sw.speeds.pop_front();
                    }
                }
                sw.chats.append(&id2, msg.clone());
                sw.generating.remove(&id2);
            });
        }
    }
    Ok(progress_body(rx))
}

/// Stop the reply being written in a conversation (what is written so far is kept).
async fn stop_chat(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Path(id): Path<String>) -> ApiResult {
    let sw = shared.lock().unwrap();
    open_chat(&sw, &id, &caller)?;
    let g = sw.generating.get(&id).ok_or((StatusCode::NOT_FOUND, "nothing is being written".to_string()))?;
    if g.asker != caller.person && caller.role < Role::Admin {
        return Err((StatusCode::FORBIDDEN, "only whoever asked (or an admin) can stop this reply".into()));
    }
    g.stop.notify_one();
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct PrefsReq {
    reply_style: String,
}

/// The caller's own preferences.
async fn set_prefs(State(shared): State<Shared>, Extension(caller): Extension<Caller>, Json(req): Json<PrefsReq>) -> ApiResult {
    if !matches!(req.reply_style.as_str(), "stream" | "whole") {
        return Err((StatusCode::BAD_REQUEST, "reply_style is stream or whole".into()));
    }
    let mut sw = shared.lock().unwrap();
    let auth = &mut sw.auth;
    let p = auth.profile.people.iter_mut().find(|p| p.id == caller.person).ok_or((StatusCode::NOT_FOUND, "no such person".to_string()))?;
    p.reply_style = Some(req.reply_style);
    auth.save();
    Ok(Json(json!({ "ok": true })))
}

// ---------------------------------------------------------------- the home screen at a glance

/// Everything the Home widgets show, in one call.
async fn overview(State(shared): State<Shared>, Extension(caller): Extension<Caller>) -> Json<Value> {
    let sw = shared.lock().unwrap();
    let online = |e: &NodeEntry| e.state != NodeState::Away;
    let total = sw.nodes.len();
    let up = sw.nodes.values().filter(|e| online(e)).count();
    let paused = sw.nodes.values().filter(|e| online(e) && e.paused.is_some()).count();
    // People with at least one device here.
    let mut here: Vec<String> = Vec::new();
    let mut mine = Vec::new();
    let mut pooled: u64 = 0;
    for (id, e) in &sw.nodes {
        let person = device_person(&sw, id, e);
        if online(e) {
            if let Some(p) = person.as_deref().and_then(|p| sw.auth.profile.person(p)) {
                if !here.contains(&p.name) {
                    here.push(p.name.clone());
                }
            }
            if e.paused.is_none() {
                pooled += crate::neighbour::ram_budget(&e.report.policy, e.report.info.ram_available_bytes, e.report.info.ram_total_bytes);
            }
        }
        if person.as_deref() == Some(caller.person.as_str()) {
            let in_model = sw.pipeline.as_ref().is_some_and(|p| p.stages.iter().any(|s| &s.node == id));
            mine.push(json!({ "id": id, "state": e.state, "paused": e.paused, "in_model": in_model, "policy": e.report.policy }));
        }
    }
    let model = sw.pipeline.as_ref().map(|p| {
        let state = if p.failure().is_some() { "failed" } else if p.ready() { "ready" } else { "loading" };
        json!({ "name": p.model, "state": state, "failure": p.failure(), "devices": p.stages.iter().map(|s| s.node.0.clone()).collect::<Vec<_>>(), "need_bytes": p.stages.iter().map(|s| s.need_bytes).sum::<u64>() })
    });
    let status = match sw.pipeline.as_ref() {
        Some(p) if p.failure().is_some() => "problem",
        _ if up == 0 => "problem",
        _ if paused > 0 || up < total => "partly",
        _ => "good",
    };
    Json(json!({
        "swarm": sw.name,
        "health": { "status": status, "online": up, "total": total, "paused": paused },
        "people": { "here": here, "total": sw.auth.profile.people.len() },
        "model": model,
        "speeds": sw.speeds.iter().map(|(_, s)| s).collect::<Vec<_>>(),
        "memory": { "pooled_bytes": pooled },
        "mine": mine,
    }))
}

#[cfg(test)]
mod tests {
    /// Every icon the dashboard and the app use (`#i-NAME`, also built in JavaScript as
    /// `ic('NAME')`) is in the icon sprite that ships with them; a missing one shows as a blank.
    #[test]
    fn every_icon_used_is_shipped() {
        let sprite = include_str!("icons.svg");
        let app_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/desktop/ui");
        let app_sprite = std::fs::read_to_string(app_dir.join("icons.svg")).unwrap();
        assert_eq!(sprite, app_sprite, "apps/desktop/ui/icons.svg differs: run tools/gen-icons.py");
        let pages = [("portal.html", include_str!("portal.html").to_string()), ("app", std::fs::read_to_string(app_dir.join("index.html")).unwrap())];
        let names = |re_start: &str, page: &str| -> Vec<String> {
            page.match_indices(re_start).map(|(i, _)| page[i + re_start.len()..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect::<String>()).filter(|n| !n.is_empty()).collect()
        };
        let mut checked = 0;
        for (which, page) in &pages {
            for n in names("#i-", page).into_iter().chain(names("ic('", page)) {
                assert!(sprite.contains(&format!("id=\"i-{n}\"")), "{which} uses icon {n:?}, which is not in icons.svg (add it to tools/gen-icons.py)");
                checked += 1;
            }
        }
        assert!(checked > 40, "found only {checked} icon uses: has the markup changed?");
    }

    #[test]
    fn join_links_carry_the_swarm_and_code() {
        let l = super::join_link("Bea & Cal's home", "1a7e71cf", "192.168.1.40", "482-913");
        assert_eq!(l, "clamor://join?swarm=Bea%20%26%20Cal%27s%20home&id=1a7e71cf&addr=192.168.1.40&code=482-913");
        let svg = super::qr_svg(&l).unwrap();
        assert!(svg.starts_with("<?xml") || svg.contains("<svg"), "{}", &svg[..60]);
    }

    /// The dashboard is served with its icons inlined (no network: swarms run offline on a LAN).
    #[tokio::test]
    async fn the_page_carries_its_icons() {
        let page = super::index().await.0;
        assert!(!page.contains("<!--ICONS-->"));
        assert!(page.contains("<symbol id=\"i-crab-fill\""));
    }
}
