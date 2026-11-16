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
pub async fn serve(bind: &str, target: Shared) -> Result<bool> {
    let Ok(listener) = tokio::net::TcpListener::bind(bind).await else {
        return Ok(false);
    };
    let app = axum::Router::new().fallback(forward).with_state(target);
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

    // Generous: a chat reply can take minutes to generate.
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(900), tls.read_to_end(&mut raw)).await??;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or_else(|| anyhow::anyhow!("bad response"))?;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let body = raw[split + 4..].to_vec();
    let mut lines = head.lines();
    let status: u16 = lines.next().and_then(|l| l.split(' ').nth(1)).and_then(|c| c.parse().ok()).ok_or_else(|| anyhow::anyhow!("bad status line"))?;
    let mut resp = Response::new(Body::from(body));
    *resp.status_mut() = StatusCode::from_u16(status)?;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            // Cookies belong to the coordinator's own dashboard, not to this proxy.
            if k.eq_ignore_ascii_case("content-type") {
                if let Ok(v) = HeaderValue::from_str(v.trim()) {
                    resp.headers_mut().insert(axum::http::header::CONTENT_TYPE, v);
                }
            }
        }
    }
    Ok(resp)
}
