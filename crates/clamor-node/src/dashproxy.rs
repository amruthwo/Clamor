//! The swarm's dashboard, on a member device: `http://127.0.0.1:7421/` forwards to the
//! coordinator's encrypted dashboard, signing every request with this device's key. So the app
//! (or a browser) on a member device needs no sign-in: it acts as the device's person.
//!
//! Only this machine can use it (it listens on 127.0.0.1), and the connection to the
//! coordinator is TLS with the certificate pinned (learned in the authenticated join).
use anyhow::Result;
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Where to forward to (None until the device has joined).
#[derive(Clone)]
pub struct Target {
    pub host: String,
    pub port: u16,
    pub fingerprint: clamor_transport::tls::Fingerprint,
    pub key: ed25519_dalek::SigningKey,
}

pub type Shared = Arc<RwLock<Option<Target>>>;

const MAX_BODY: usize = 8 << 20;

/// Listen on `bind` (normally 127.0.0.1:7421). Returns false if the port is taken (e.g. the
/// coordinator runs on this machine too, and its own dashboard is there).
pub async fn serve(bind: &str, target: Shared, private: crate::private::Shared) -> Result<bool> {
    let Ok(listener) = tokio::net::TcpListener::bind(bind).await else {
        return Ok(false);
    };
    use axum::routing::{get, post};
    // Private chats are answered here, on this device: never forwarded.
    let local = axum::Router::new()
        .route("/api/private/status", get(private_status))
        .route("/api/private/open", post(private_open))
        .route("/api/private/close", post(private_close))
        .route("/api/private/chats", get(private_list).post(private_new))
        .route("/api/private/chats/{id}", get(private_get).delete(private_delete))
        .route("/api/private/chats/{id}/messages", post(private_send))
        .route("/api/private/chats/{id}/stop", post(private_stop))
        .with_state(private);
    let app = axum::Router::new().fallback(forward).with_state(target).merge(local);
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("dashboard proxy stopped: {e}");
        }
    });
    Ok(true)
}

async fn forward(State(target): State<Shared>, req: Request) -> Response {
    let Some(t) = target.read().unwrap().clone() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "not connected to the swarm yet").into_response();
    };
    match send(&t, req).await {
        Ok(r) => r,
        Err(e) => (StatusCode::BAD_GATEWAY, format!("cannot reach the coordinator's dashboard: {e:#}")).into_response(),
    }
}

/// One request over its own HTTP/1.0 connection (no chunked encoding to undo).
async fn send(t: &Target, req: Request) -> Result<Response> {
    let method = req.method().clone();
    let path_q = req.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".into());
    let content_type = req.headers().get(axum::http::header::CONTENT_TYPE).cloned();
    let body = to_bytes(req.into_body(), MAX_BODY).await?;

    let tcp = tokio::time::timeout(Duration::from_secs(10), tokio::net::TcpStream::connect((t.host.as_str(), t.port))).await??;
    let cfg = clamor_transport::tls::client_config(None, Some(t.fingerprint))?;
    let mut tls = tokio_rustls::TlsConnector::from(cfg).connect(clamor_transport::tls::server_name(), tcp).await?;

    let auth = crate::auth::sign_share_request(&t.key, &format!("{method} {path_q}"));
    let mut head = format!("{method} {path_q} HTTP/1.0\r\nHost: clamor\r\n{}: {auth}\r\nContent-Length: {}\r\n", crate::auth::SHARE_AUTH_HEADER, body.len());
    if let Some(ct) = content_type.as_ref().and_then(|v| v.to_str().ok()) {
        head.push_str(&format!("Content-Type: {ct}\r\n"));
    }
    head.push_str("\r\n");
    tls.write_all(head.as_bytes()).await?;
    tls.write_all(&body).await?;
    tls.flush().await?;

    // Read the status line and headers, then pass the body on as it arrives (a chat reply is
    // streamed while the model writes it).
    let mut raw = Vec::new();
    let mut buf = vec![0u8; 16 << 10];
    let split = loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let n = tokio::time::timeout(Duration::from_secs(900), tls.read(&mut buf)).await??;
        if n == 0 {
            anyhow::bail!("the coordinator closed the connection");
        }
        raw.extend_from_slice(&buf[..n]);
        if raw.len() > 64 << 10 {
            anyhow::bail!("bad response");
        }
    };
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let first = raw[split + 4..].to_vec();
    let mut lines = head.lines();
    let status: u16 = lines.next().and_then(|l| l.split(' ').nth(1)).and_then(|c| c.parse().ok()).ok_or_else(|| anyhow::anyhow!("bad status line"))?;
    let (btx, brx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(16);
    tokio::spawn(async move {
        if !first.is_empty() && btx.send(Ok(first.into())).await.is_err() {
            return;
        }
        loop {
            // Generous: a chat reply can take minutes.
            match tokio::time::timeout(Duration::from_secs(900), tls.read(&mut buf)).await {
                Ok(Ok(0)) | Err(_) => return,
                Ok(Ok(n)) => {
                    if btx.send(Ok(bytes::Bytes::copy_from_slice(&buf[..n]))).await.is_err() {
                        return;
                    }
                }
                Ok(Err(e)) => {
                    let _ = btx.send(Err(e)).await;
                    return;
                }
            }
        }
    });
    let mut resp = Response::new(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(brx)));
    *resp.status_mut() = StatusCode::from_u16(status)?;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            // Cookies belong to the coordinator's own dashboard, not to this proxy.
            // A known length passes on (otherwise the body goes out chunked, as it arrives).
            if k.eq_ignore_ascii_case("content-type") || k.eq_ignore_ascii_case("cache-control") || k.eq_ignore_ascii_case("content-length") {
                if let (Ok(name), Ok(v)) = (axum::http::HeaderName::from_bytes(k.trim().as_bytes()), HeaderValue::from_str(v.trim())) {
                    resp.headers_mut().insert(name, v);
                }
            }
        }
    }
    Ok(resp)
}

// ---------------------------------------------------------------- private chats, on this device

type ApiError = (StatusCode, String);
type P = State<crate::private::Shared>;

async fn private_status(State(p): P) -> axum::Json<serde_json::Value> {
    let p = p.lock().unwrap();
    let (state, message) = match &p.state {
        crate::private::State::Failed(m) => ("failed", Some(m.clone())),
        s => (match s { crate::private::State::Off => "off", crate::private::State::Starting => "starting", _ => "ready" }, None),
    };
    axum::Json(serde_json::json!({ "available": true, "state": state, "message": message, "ring": p.ring, "layers_here": p.layers }))
}

async fn private_open(State(p): P) -> Result<axum::Json<serde_json::Value>, ApiError> {
    let mut p = p.lock().unwrap();
    let ctl = p.control.clone().ok_or((StatusCode::SERVICE_UNAVAILABLE, "not connected to the swarm".to_string()))?;
    if matches!(p.state, crate::private::State::Ready | crate::private::State::Starting) {
        return Ok(axum::Json(serde_json::json!({ "ok": true })));
    }
    ctl.send(clamor_proto::Message::PrivateOpen).map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "not connected to the swarm".to_string()))?;
    p.state = crate::private::State::Starting;
    Ok(axum::Json(serde_json::json!({ "ok": true })))
}

async fn private_close(State(p): P) -> axum::Json<serde_json::Value> {
    let mut p = p.lock().unwrap();
    if let Some(ctl) = &p.control {
        let _ = ctl.send(clamor_proto::Message::PrivateClose);
    }
    p.stop();
    p.state = crate::private::State::Off;
    axum::Json(serde_json::json!({ "ok": true }))
}

fn chats_of(p: &crate::private::Private) -> Result<&crate::chats::Chats, ApiError> {
    p.chats.as_ref().ok_or((StatusCode::SERVICE_UNAVAILABLE, "not connected to the swarm yet".to_string()))
}

async fn private_list(State(p): P) -> Result<axum::Json<serde_json::Value>, ApiError> {
    let p = p.lock().unwrap();
    Ok(axum::Json(serde_json::json!({ "conversations": chats_of(&p)?.list(crate::private::ME) })))
}

async fn private_new(State(p): P) -> Result<axum::Json<serde_json::Value>, ApiError> {
    let mut p = p.lock().unwrap();
    let c = p.chats.as_mut().ok_or((StatusCode::SERVICE_UNAVAILABLE, "not connected to the swarm yet".to_string()))?;
    Ok(axum::Json(serde_json::json!({ "id": c.create(crate::private::ME, clamor_proto::unix_now()) })))
}

fn open(p: &crate::private::Private, id: &str) -> Result<crate::chats::Conversation, ApiError> {
    let c = chats_of(p)?;
    if !c.can_open(id, crate::private::ME) || id == crate::chats::GROUP {
        return Err((StatusCode::NOT_FOUND, "no such conversation".into()));
    }
    Ok(c.get(id).cloned().expect("checked"))
}

async fn private_get(State(p): P, axum::extract::Path(id): axum::extract::Path<String>) -> Result<axum::Json<serde_json::Value>, ApiError> {
    let p = p.lock().unwrap();
    let mut v = serde_json::to_value(open(&p, &id)?).unwrap();
    if let Some(g) = p.generating.get(&id) {
        v["live"] = serde_json::json!({ "text": g.progress.borrow().text });
    }
    Ok(axum::Json(v))
}

async fn private_delete(State(p): P, axum::extract::Path(id): axum::extract::Path<String>) -> Result<axum::Json<serde_json::Value>, ApiError> {
    let mut p = p.lock().unwrap();
    open(&p, &id)?;
    p.chats.as_mut().unwrap().delete(&id);
    Ok(axum::Json(serde_json::json!({ "ok": true })))
}

#[derive(serde::Deserialize)]
struct SendReq {
    content: String,
}

/// Like the swarm's chats (see portal.rs), but the reply is written by this device's private
/// ring and the conversation stays on this device.
async fn private_send(State(p): P, axum::extract::Path(id): axum::extract::Path<String>, axum::Json(req): axum::Json<SendReq>) -> Result<Response, ApiError> {
    use crate::generate::{chat_prompt, failed, progress_body, spawn_reply, Generation, Progress};
    let content = req.content.trim().to_string();
    if content.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "say something".into()));
    }
    let shared = p.clone();
    let (conv, ring) = {
        let mut p = p.lock().unwrap();
        open(&p, &id)?;
        if p.generating.contains_key(&id) {
            return Err((StatusCode::CONFLICT, "the model is still answering the last message".into()));
        }
        let chats = p.chats.as_mut().unwrap();
        chats.append(&id, crate::chats::ChatMsg { role: "user".into(), content, by: None, at: clamor_proto::unix_now(), tok_s: None, error: None, stopped: false });
        let conv = chats.get(&id).cloned().expect("exists");
        let ring = match (&p.state, &p.head, &p.tokenizer) {
            (crate::private::State::Ready, Some(h), Some(t)) => Ok((h.clone(), t.clone(), p.n_ctx)),
            (crate::private::State::Starting, ..) => Err("private chats are still starting".to_string()),
            (crate::private::State::Failed(m), ..) => Err(m.clone()),
            _ => Err("private chats are off: turn them on first".to_string()),
        };
        (conv, ring)
    };
    let (tx, rx) = tokio::sync::watch::channel(Progress::default());
    let stop = Arc::new(tokio::sync::Notify::new());
    match ring.and_then(|(head, tok, n_ctx)| chat_prompt(&tok, &conv, n_ctx).map(|pr| (head, tok, n_ctx, pr))) {
        Err(e) => {
            let msg = failed(e);
            if let Some(c) = shared.lock().unwrap().chats.as_mut() {
                c.append(&id, msg.clone());
            }
            let _ = tx.send(Progress { done: Some(msg), ..Default::default() });
        }
        Ok((head, tok, n_ctx, prompt)) => {
            shared.lock().unwrap().generating.insert(id.clone(), Generation { asker: crate::private::ME.into(), stop: stop.clone(), progress: rx.clone() });
            let (s2, id2) = (shared.clone(), id.clone());
            spawn_reply(head, tok, n_ctx, prompt, conv, stop, tx, move |msg| {
                let mut p = s2.lock().unwrap();
                if let Some(c) = p.chats.as_mut() {
                    c.append(&id2, msg.clone());
                }
                p.generating.remove(&id2);
            });
        }
    }
    Ok(progress_body(rx))
}

async fn private_stop(State(p): P, axum::extract::Path(id): axum::extract::Path<String>) -> Result<axum::Json<serde_json::Value>, ApiError> {
    let p = p.lock().unwrap();
    let g = p.generating.get(&id).ok_or((StatusCode::NOT_FOUND, "nothing is being written".to_string()))?;
    g.stop.notify_one();
    Ok(axum::Json(serde_json::json!({ "ok": true })))
}
