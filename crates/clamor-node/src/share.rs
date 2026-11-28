//! Model sharing: one copy of a model on the network, not one per device.
//!
//! Every node serves the models in its `--models` directory over HTTP. A stage that runs on a
//! node without the model fetches a *layer slice* (a GGUF with all the metadata but only that
//! stage's tensors, see `clamor_engine_ggml::plan_stage_slice`) from a node that has it, keeps
//! it in its cache, and loads it like any model file. A 12-layer share of an 80-layer model
//! downloads roughly 12/80 of the file.
//!
//! Every request is signed by the requesting device's swarm key (`x-clamor-auth`, see
//! `auth::sign_share_request`), and a server answers only keys its swarm approved, so other
//! machines on the network cannot download your models.
use anyhow::{anyhow, bail, Context, Result};
use axum::body::Body;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use bytes::Bytes;
use clamor_engine_ggml::{plan_stage_slice, GgufFile};
use clamor_proto::ModelInfo;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

pub const MODEL_ID_HEADER: &str = "x-clamor-model-id";

/// SHA-256 of the GGUF header (everything before the tensor data) plus the file size.
pub fn model_id(path: &Path) -> Result<String> {
    let g = GgufFile::open(&path.display().to_string()).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let header_len = g.data_offset();
    let size = std::fs::metadata(path)?.len();
    let mut f = std::fs::File::open(path)?;
    let mut header = vec![0u8; header_len.min(size) as usize];
    f.read_exact(&mut header)?;
    let mut h = Sha256::new();
    h.update(&header);
    h.update(size.to_le_bytes());
    Ok(hex::encode(h.finalize()))
}

type IdCache = Mutex<HashMap<(PathBuf, u64, SystemTime), String>>;

/// The `.gguf` files in `dir`, with ids (cached by path, size and modification time).
pub fn scan(dir: &Path) -> Vec<ModelInfo> {
    static IDS: std::sync::OnceLock<IdCache> = std::sync::OnceLock::new();
    let ids = IDS.get_or_init(Default::default);
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()).map(str::to_string) else { continue };
        if crate::stage::resolve_model(dir, &name).is_err() {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        let key = (p.clone(), meta.len(), meta.modified().unwrap_or(SystemTime::UNIX_EPOCH));
        let cached = ids.lock().unwrap().get(&key).cloned();
        let id = match cached {
            Some(id) => id,
            None => match model_id(&p) {
                Ok(id) => {
                    ids.lock().unwrap().insert(key, id.clone());
                    id
                }
                Err(e) => {
                    eprintln!("skipping {name}: {e:#}");
                    continue;
                }
            },
        };
        out.push(ModelInfo { name, size: meta.len(), id });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Device keys (hex) allowed to fetch from this node: the swarm's approved devices and its
/// coordinator. Updated as devices are approved.
pub type AllowedKeys = Arc<RwLock<std::collections::HashSet<String>>>;

#[derive(Clone)]
struct Shared {
    models_dir: PathBuf,
    allowed: AllowedKeys,
}

#[derive(Deserialize)]
struct SliceQuery {
    start: u32,
    end: u32,
    #[serde(default)]
    embed: bool,
}

/// Why a request may not be served, if it may not.
fn refused(s: &Shared, uri: &Uri, headers: &HeaderMap) -> Option<Response> {
    let path = uri.path_and_query().map(|p| p.as_str()).unwrap_or("");
    let Some(h) = headers.get(crate::auth::SHARE_AUTH_HEADER).and_then(|v| v.to_str().ok()) else {
        return Some((StatusCode::UNAUTHORIZED, "only devices of this swarm may fetch models (signed request required)").into_response());
    };
    let allowed = s.allowed.read().unwrap();
    match crate::auth::verify_share_request(h, path, |k| allowed.contains(k)) {
        Ok(_) => None,
        Err(e) => Some((StatusCode::FORBIDDEN, format!("{e}")).into_response()),
    }
}

/// `x-clamor-auth` for a request to `url`, signed over its path and query.
fn sign(key: Option<&ed25519_dalek::SigningKey>, url: &str) -> Option<String> {
    let path = url.find("://").and_then(|i| url[i + 3..].find('/').map(|j| &url[i + 3 + j..])).unwrap_or("/");
    key.map(|k| crate::auth::sign_share_request(k, path))
}

async fn list(State(s): State<Shared>, uri: Uri, headers: HeaderMap) -> Response {
    if let Some(r) = refused(&s, &uri, &headers) {
        return r;
    }
    let dir = s.models_dir.clone();
    match tokio::task::spawn_blocking(move || scan(&dir)).await {
        Ok(models) => Json(models).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// A `Write` that hands chunks to the HTTP response stream.
struct ChannelWriter(tokio::sync::mpsc::Sender<std::io::Result<Bytes>>);
impl Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.blocking_send(Ok(Bytes::copy_from_slice(buf))).map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "client went away"))?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn slice(State(s): State<Shared>, UrlPath(name): UrlPath<String>, Query(q): Query<SliceQuery>, uri: Uri, headers: HeaderMap) -> Response {
    if let Some(r) = refused(&s, &uri, &headers) {
        return r;
    }
    let path = match crate::stage::resolve_model(&s.models_dir, &name) {
        Ok(p) => p,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let planned = tokio::task::spawn_blocking({
        let path = path.clone();
        move || -> Result<(clamor_engine_ggml::SlicePlan, String)> {
            let g = GgufFile::open(&path.display().to_string()).map_err(|e| anyhow!("{e}"))?;
            let plan = plan_stage_slice(&g, q.start..q.end, q.embed).map_err(|e| anyhow!("{e}"))?;
            Ok((plan, model_id(&path)?))
        }
    })
    .await;
    let (plan, id) = match planned {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => return (StatusCode::BAD_REQUEST, format!("{e:#}")).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let len = plan.total_len;
    let (tx, rx) = tokio::sync::mpsc::channel::<std::io::Result<Bytes>>(8);
    tokio::task::spawn_blocking(move || {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, ChannelWriter(tx.clone()));
        let r = std::fs::File::open(&path).and_then(|mut f| plan.write_to(&mut f, &mut out)).and_then(|_| out.flush());
        if let Err(e) = r {
            let _ = tx.blocking_send(Err(e));
        }
    });
    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Response::builder()
        .header("content-type", "application/octet-stream")
        .header("content-length", len)
        .header(MODEL_ID_HEADER, id)
        .body(body)
        .unwrap()
}

/// The whole model file (a handover: this device is leaving with the swarm's only copy).
async fn whole_file(State(s): State<Shared>, UrlPath(name): UrlPath<String>, uri: Uri, headers: HeaderMap) -> Response {
    if let Some(r) = refused(&s, &uri, &headers) {
        return r;
    }
    let path = match crate::stage::resolve_model(&s.models_dir, &name) {
        Ok(p) => p,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let id = match tokio::task::spawn_blocking({
        let path = path.clone();
        move || model_id(&path)
    })
    .await
    {
        Ok(Ok(id)) => id,
        Ok(Err(e)) => return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let len = match std::fs::metadata(&path) {
        Ok(m) => m.len(),
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<std::io::Result<Bytes>>(8);
    tokio::task::spawn_blocking(move || {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, ChannelWriter(tx.clone()));
        let r = std::fs::File::open(&path).and_then(|mut f| std::io::copy(&mut f, &mut out)).and_then(|_| out.flush());
        if let Err(e) = r {
            let _ = tx.blocking_send(Err(e));
        }
    });
    Response::builder()
        .header("content-type", "application/octet-stream")
        .header("content-length", len)
        .header(MODEL_ID_HEADER, id)
        .body(Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)))
        .unwrap()
}

/// Every tensor's name and size in bytes: enough to estimate a stage's memory without the file.
async fn tensors(State(s): State<Shared>, UrlPath(name): UrlPath<String>, uri: Uri, headers: HeaderMap) -> Response {
    if let Some(r) = refused(&s, &uri, &headers) {
        return r;
    }
    let path = match crate::stage::resolve_model(&s.models_dir, &name) {
        Ok(p) => p,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    match tokio::task::spawn_blocking(move || tensor_table(&path)).await {
        Ok(Ok(t)) => Json(t).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, format!("{e:#}")).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// (name, bytes) for every tensor in a GGUF file.
pub fn tensor_table(path: &Path) -> Result<Vec<(String, u64)>> {
    let g = GgufFile::open(&path.display().to_string()).map_err(|e| anyhow!("{e}"))?;
    Ok(g.tensors().into_iter().map(|t| (t.name, t.n_bytes)).collect())
}

/// Fetch another node's tensor table for `name` (checking it is model `model_id`).
pub fn fetch_tensor_table(source: &str, name: &str, key: Option<&ed25519_dalek::SigningKey>) -> Result<Vec<(String, u64)>> {
    let url = format!("{}/models/{}/tensors", source.trim_end_matches('/'), url_escape(name));
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(30))).http_status_as_error(false).build().into();
    let mut req = agent.get(&url);
    if let Some(h) = sign(key, &url) {
        req = req.header(crate::auth::SHARE_AUTH_HEADER, h);
    }
    let mut resp = req.call().with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        let msg = resp.body_mut().read_to_string().unwrap_or_default();
        bail!("{source} refused the tensor list ({}): {msg}", resp.status());
    }
    Ok(serde_json::from_reader(resp.body_mut().as_reader())?)
}

/// Serve the models in `models_dir`, on `port` if free, else on any free port. Returns the port.
pub async fn serve(port: u16, models_dir: PathBuf, allowed: AllowedKeys) -> Result<u16> {
    let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
        Ok(l) => l,
        Err(_) => tokio::net::TcpListener::bind(("0.0.0.0", 0)).await?,
    };
    let port = listener.local_addr()?.port();
    let app = Router::new()
        .route("/models", get(list))
        .route("/models/{name}/slice", get(slice))
        .route("/models/{name}/tensors", get(tensors))
        .route("/models/{name}/file", get(whole_file))
        .with_state(Shared { models_dir, allowed });
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("model sharing stopped: {e}");
        }
    });
    Ok(port)
}

/// Where a fetched slice is cached.
pub fn cached_slice_path(cache_dir: &Path, model_id: &str, name: &str, layers: std::ops::Range<u32>, embed: bool) -> PathBuf {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let short = &model_id[..model_id.len().min(16)];
    cache_dir.join(short).join(format!("{stem}.layers-{}-{}{}.gguf", layers.start, layers.end, if embed { "-embed" } else { "" }))
}

fn url_escape(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// Download a slice from `source` (`http://host:port`) to `dest`, unless it is already there.
/// Checks that the server's copy is the model `model_id` and that every byte arrived.
/// `progress(done, total)` is called as it goes.
#[allow(clippy::too_many_arguments)]
pub fn fetch_slice(source: &str, name: &str, model_id: &str, layers: std::ops::Range<u32>, embed: bool, key: Option<&ed25519_dalek::SigningKey>, dest: &Path, mut progress: impl FnMut(u64, u64)) -> Result<()> {
    if dest.is_file() {
        return Ok(());
    }
    let url = format!("{}/models/{}/slice?start={}&end={}&embed={}", source.trim_end_matches('/'), url_escape(name), layers.start, layers.end, embed);
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(10))).http_status_as_error(false).build().into();
    let mut req = agent.get(&url);
    if let Some(h) = sign(key, &url) {
        req = req.header(crate::auth::SHARE_AUTH_HEADER, h);
    }
    let mut resp = req.call().with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let msg = resp.body_mut().read_to_string().unwrap_or_default();
        bail!("{source} refused the slice ({status}): {msg}");
    }
    let got_id = resp.headers().get(MODEL_ID_HEADER).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if got_id != model_id {
        bail!("{source} has a different {name} (model id {got_id}, expected {model_id})");
    }
    let total: u64 = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).ok_or_else(|| anyhow!("no content-length from {source}"))?;
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let part = dest.with_extension("gguf.part");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&part)?);
    let mut body = resp.body_mut().with_config().limit(total + 1).reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut done, mut last) = (0u64, std::time::Instant::now());
    progress(0, total);
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if last.elapsed() > Duration::from_millis(500) {
            progress(done, total);
            last = std::time::Instant::now();
        }
    }
    out.flush()?;
    drop(out);
    if done != total {
        let _ = std::fs::remove_file(&part);
        bail!("{source} sent {done} of {total} bytes of {name}");
    }
    progress(done, total);
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// Copy the whole model file `name` from `source` into `models_dir` (a handover), checking that
/// the server has the model `model_id` and that the copy is that model before it is put in place.
pub fn fetch_file(source: &str, name: &str, model_id_want: &str, key: Option<&ed25519_dalek::SigningKey>, models_dir: &Path, mut progress: impl FnMut(u64, u64)) -> Result<()> {
    crate::stage::validate_model_name(name)?;
    let dest = models_dir.join(name);
    if dest.is_file() && model_id(&dest).is_ok_and(|id| id == model_id_want) {
        return Ok(());
    }
    let url = format!("{}/models/{}/file", source.trim_end_matches('/'), url_escape(name));
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(10))).http_status_as_error(false).build().into();
    let mut req = agent.get(&url);
    if let Some(h) = sign(key, &url) {
        req = req.header(crate::auth::SHARE_AUTH_HEADER, h);
    }
    let mut resp = req.call().with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let msg = resp.body_mut().read_to_string().unwrap_or_default();
        bail!("{source} refused {name} ({status}): {msg}");
    }
    let got_id = resp.headers().get(MODEL_ID_HEADER).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if got_id != model_id_want {
        bail!("{source} has a different {name} (model id {got_id}, expected {model_id_want})");
    }
    let total: u64 = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).ok_or_else(|| anyhow!("no content-length from {source}"))?;
    if let Some(free) = clamor_muster::probe::free_space(models_dir) {
        if free < total + (256 << 20) {
            bail!("not enough disk space here for {name}: {:.1} GB needed, {:.1} GB free", total as f64 / 1e9, free as f64 / 1e9);
        }
    }
    let part = models_dir.join(format!(".{name}.handover.part"));
    let result = (|| -> Result<()> {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&part)?);
        let mut body = resp.body_mut().with_config().limit(total + 1).reader();
        let mut buf = vec![0u8; 1 << 20];
        let (mut done, mut last) = (0u64, std::time::Instant::now());
        progress(0, total);
        loop {
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            if last.elapsed() > Duration::from_millis(500) {
                progress(done, total);
                last = std::time::Instant::now();
            }
        }
        out.flush()?;
        drop(out);
        if done != total {
            bail!("{source} sent {done} of {total} bytes of {name}");
        }
        // The copy must be the same model (header and size), as the original is about to go.
        let got = model_id(&part)?;
        if got != model_id_want {
            bail!("the copy of {name} is not the same model ({got})");
        }
        std::fs::rename(&part, &dest)?;
        progress(done, total);
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_paths_separate_models_layers_and_roles() {
        let d = Path::new("/c");
        let p = |id, l: std::ops::Range<u32>, e| cached_slice_path(d, id, "m.gguf", l, e);
        assert_eq!(p("abcdef0123456789ffff", 0..16, false), Path::new("/c/abcdef0123456789/m.layers-0-16.gguf"));
        assert_ne!(p("aaaa", 0..16, false), p("bbbb", 0..16, false), "different models");
        assert_ne!(p("aaaa", 0..16, false), p("aaaa", 0..16, true), "embedding or not");
    }

    #[test]
    fn requests_are_signed_over_path_and_query() {
        let k = crate::auth::Device::for_tests([5; 32]).swarm_key("s");
        let h = sign(Some(&k), "http://10.0.0.2:7423/models/m%20x.gguf/slice?start=0&end=2&embed=false").unwrap();
        let kh = crate::auth::key_hex(&k.verifying_key());
        assert!(crate::auth::verify_share_request(&h, "/models/m%20x.gguf/slice?start=0&end=2&embed=false", |x| x == kh).is_ok());
        assert!(sign(None, "http://h/x").is_none());
    }

    #[test]
    fn names_are_escaped_in_urls() {
        assert_eq!(url_escape("my model (q4).gguf"), "my%20model%20%28q4%29.gguf");
    }
}
