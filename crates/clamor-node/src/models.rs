//! The model picker: Clamor's catalog of known models (`catalog.json`, built from the real files
//! by tools/build-catalog.py), what this swarm could run of each, and downloading one.
//!
//! A download goes to the coordinator's models folder (the other devices fetch the layers they
//! run from there, as for any model). It is checked against the size and SHA-256 in the catalog,
//! which ships inside Clamor's signed releases, so a changed or broken file is refused. An
//! interrupted download resumes where it stopped.
use anyhow::{anyhow, bail, Context, Result};
use clamor_muster::plan;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogFile {
    pub quant: String,
    pub file: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub maker: String,
    pub license: String,
    pub about: String,
    pub repo: String,
    pub arch: String,
    pub tokenizer_pre: String,
    pub params_b: f64,
    pub n_layers: u32,
    pub hidden_dim: u32,
    pub kv_ratio: f64,
    pub context: u32,
    pub caps: Vec<String>,
    pub files: Vec<CatalogFile>,
}

#[derive(Deserialize)]
struct CatalogDoc {
    models: Vec<CatalogModel>,
}

pub fn catalog() -> &'static [CatalogModel] {
    static C: LazyLock<Vec<CatalogModel>> = LazyLock::new(|| serde_json::from_str::<CatalogDoc>(include_str!("catalog.json")).expect("catalog.json is valid").models);
    &C
}

pub fn find(id: &str) -> Option<&'static CatalogModel> {
    catalog().iter().find(|m| m.id == id)
}

impl CatalogModel {
    /// Whether this Clamor can run it, and why not.
    pub fn support(&self) -> std::result::Result<(), String> {
        clamor_engine_ggml::supports_model(&self.arch, &self.tokenizer_pre)
    }

    /// The file for a quantization ("q8"/"Q8_0", "q4"/"Q4_K_M"), else the first one there is.
    pub fn file(&self, quant: &str) -> &CatalogFile {
        let want = quant.to_ascii_lowercase();
        self.files.iter().find(|f| f.quant.to_ascii_lowercase().starts_with(&want)).unwrap_or(&self.files[0])
    }

    /// The planner's view of one file of this model.
    pub fn spec(&self, f: &CatalogFile) -> plan::ModelSpec {
        plan::ModelSpec {
            name: f.file.clone(),
            params_b: self.params_b,
            bits_per_weight: f.size as f64 * 8.0 / (self.params_b * 1e9).max(1.0),
            n_layers: self.n_layers.max(1),
            hidden_dim: self.hidden_dim.max(1),
            kv_ratio: self.kv_ratio,
        }
    }

    pub fn url(&self, f: &CatalogFile) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, f.file)
    }
}

// ---------------------------------------------------------------- deleting

/// Delete a model from this device: the file `name` in `models_dir` (only if it is the model
/// `model_id`, not another file that happens to have the same name) and the cached slices of it.
/// Returns the bytes freed.
pub fn delete_local(models_dir: &Path, cache_dir: &Path, name: &str, model_id: &str) -> Result<u64> {
    let mut freed = 0;
    if let Ok(path) = crate::stage::resolve_model(models_dir, name) {
        if path.exists() {
            let id = crate::share::model_id(&path)?;
            if id == model_id {
                freed += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                std::fs::remove_file(&path).with_context(|| format!("deleting {}", path.display()))?;
            }
        }
    }
    freed += crate::cache::purge_model(cache_dir, model_id).1;
    Ok(freed)
}

// ---------------------------------------------------------------- downloads

#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    pub model: String,
    pub quant: String,
    pub file: String,
    pub done: u64,
    pub total: u64,
    /// "downloading", "verifying", "done", "failed" or "cancelled".
    pub state: String,
    pub error: Option<String>,
    #[serde(skip)]
    cancel: Arc<AtomicBool>,
}

static DOWNLOADS: LazyLock<Mutex<Vec<Progress>>> = LazyLock::new(Default::default);

/// Downloads of this process, newest first.
pub fn downloads() -> Vec<Progress> {
    DOWNLOADS.lock().unwrap().iter().rev().cloned().collect()
}

fn set(file: &str, f: impl FnOnce(&mut Progress)) {
    if let Some(p) = DOWNLOADS.lock().unwrap().iter_mut().rev().find(|p| p.file == file) {
        f(p);
    }
}

pub fn cancel(file: &str) -> bool {
    let d = DOWNLOADS.lock().unwrap();
    match d.iter().rev().find(|p| p.file == file && matches!(p.state.as_str(), "downloading" | "verifying")) {
        Some(p) => {
            p.cancel.store(true, Ordering::Relaxed);
            true
        }
        None => false,
    }
}

/// Start downloading `model` (one quantization) into `dir`. Returns at once; `on_done` runs
/// (on the download's thread) when the file is in place.
pub fn start(model: &CatalogModel, quant: &str, dir: &Path, on_done: impl FnOnce() + Send + 'static) -> Result<CatalogFile> {
    let f = model.file(quant).clone();
    if dir.join(&f.file).exists() {
        bail!("{} is already in the models folder", f.file);
    }
    {
        let mut d = DOWNLOADS.lock().unwrap();
        if d.iter().any(|p| p.file == f.file && matches!(p.state.as_str(), "downloading" | "verifying")) {
            bail!("{} is already downloading", f.file);
        }
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let have = std::fs::metadata(part_path(dir, &f.file)).map(|m| m.len()).unwrap_or(0);
        if let Some(free) = clamor_muster::probe::free_space(dir) {
            // Room for the rest of the file, and a little to spare.
            let need = f.size.saturating_sub(have) + (256 << 20);
            if free < need {
                bail!("not enough disk space on this device: {} needs {:.1} GB more, {:.1} GB free", f.file, need as f64 / 1e9, free as f64 / 1e9);
            }
        }
        d.push(Progress { model: model.id.clone(), quant: f.quant.clone(), file: f.file.clone(), done: have, total: f.size, state: "downloading".into(), error: None, cancel: Arc::new(AtomicBool::new(false)) });
    }
    let cancel = DOWNLOADS.lock().unwrap().iter().rev().find(|p| p.file == f.file).unwrap().cancel.clone();
    let (url, dir, file) = (model.url(&f), dir.to_path_buf(), f.clone());
    std::thread::Builder::new().name("clamor-download".into()).spawn(move || {
        match fetch(&url, &dir, &file, &cancel) {
            Ok(()) => {
                set(&file.file, |p| p.state = "done".into());
                println!("downloaded {} (verified)", file.file);
                on_done();
            }
            Err(e) => {
                let cancelled = cancel.load(Ordering::Relaxed);
                if !cancelled {
                    eprintln!("download of {} failed: {e:#}", file.file);
                }
                set(&file.file, |p| {
                    p.state = if cancelled { "cancelled".into() } else { "failed".into() };
                    p.error = (!cancelled).then(|| format!("{e:#}"));
                });
            }
        }
    })?;
    Ok(f)
}

fn part_path(dir: &Path, file: &str) -> PathBuf {
    dir.join(format!(".{file}.part"))
}

/// Download `f` from `url` into `dir`, resuming a `.part` file left by an earlier try, then
/// check its size and SHA-256 and move it into place.
fn fetch(url: &str, dir: &Path, f: &CatalogFile, cancel: &AtomicBool) -> Result<()> {
    let part = part_path(dir, &f.file);
    let mut out = std::fs::OpenOptions::new().create(true).read(true).append(true).open(&part).with_context(|| format!("opening {}", part.display()))?;
    let mut have = out.metadata()?.len();
    if have > f.size {
        out.set_len(0)?;
        have = 0;
    }
    // Hash what is already there (a resumed download).
    let mut hasher = Sha256::new();
    if have > 0 {
        let mut r = std::fs::File::open(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        let mut left = have;
        while left > 0 {
            let n = r.read(&mut buf[..left.min(1 << 20) as usize])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            left -= n as u64;
        }
    }
    if have < f.size {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).timeout_recv_response(Some(Duration::from_secs(60))).user_agent(concat!("clamor-node/", env!("CARGO_PKG_VERSION"))).build().into();
        let mut req = agent.get(url);
        if have > 0 {
            req = req.header("Range", &format!("bytes={have}-"));
        }
        let mut resp = req.call().with_context(|| format!("GET {url}"))?;
        if have > 0 && resp.status() != 206 {
            // The server sent the whole file: start over.
            out.set_len(0)?;
            out.seek(std::io::SeekFrom::Start(0))?;
            hasher = Sha256::new();
            have = 0;
        }
        let mut body = resp.body_mut().with_config().limit(f.size + 1).reader();
        let mut buf = vec![0u8; 1 << 20];
        let mut since_flush = 0u64;
        loop {
            if cancel.load(Ordering::Relaxed) {
                out.flush()?;
                bail!("cancelled");
            }
            let n = body.read(&mut buf).context("reading the download")?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            have += n as u64;
            since_flush += n as u64;
            if since_flush > 64 << 20 {
                out.flush()?;
                since_flush = 0;
            }
            set(&f.file, |p| p.done = have);
        }
        out.flush()?;
    }
    drop(out);
    set(&f.file, |p| p.state = "verifying".into());
    if have != f.size {
        bail!("got {have} bytes, the catalog says {} (try again: it resumes)", f.size);
    }
    let sha = hex::encode(hasher.finalize());
    if !sha.eq_ignore_ascii_case(&f.sha256) {
        let _ = std::fs::remove_file(&part);
        return Err(anyhow!("the file's SHA-256 does not match Clamor's catalog ({sha}): it was removed"));
    }
    std::fs::rename(&part, dir.join(&f.file))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalog parses, every model has files with sizes and hashes, and the models the
    /// engine can run include the ones Clamor was checked with.
    #[test]
    fn the_catalog_is_complete() {
        let c = catalog();
        assert!(c.len() >= 10);
        for m in c {
            assert!(!m.files.is_empty() && m.n_layers > 0 && m.params_b > 0.0, "{}", m.id);
            for f in &m.files {
                assert!(f.size > 0 && f.sha256.len() == 64 && f.file.ends_with(".gguf"), "{} {}", m.id, f.file);
                assert!(!f.file.contains('/'), "{}", f.file);
            }
        }
        for id in ["llama-3.2-1b", "qwen2.5-0.5b", "smollm2-360m"] {
            assert!(find(id).unwrap().support().is_ok(), "{id}");
        }
        assert!(find("qwen3-8b").unwrap().support().is_ok(), "Qwen3 runs");
        assert!(find("gemma-3-4b").unwrap().support().is_err(), "Gemma 3 does not yet");
        assert_eq!(find("llama-3.2-3b").unwrap().file("q4").quant, "Q4_K_M");
        assert_eq!(find("llama-3.2-3b").unwrap().file("q8").quant, "Q8_0");
    }

    /// A download is checked against the catalog: the right bytes land in place (also when
    /// resumed from a partial file); wrong bytes are refused and removed.
    #[test]
    fn downloads_resume_and_are_verified() {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;
        let body: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/m.gguf", l.local_addr().unwrap());
        let served = body.clone();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { continue };
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut from = 0usize;
                loop {
                    let mut line = String::new();
                    if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        from = v.trim().trim_end_matches('-').parse().unwrap();
                    }
                }
                let part = &served[from..];
                let status = if from > 0 { "206 Partial Content" } else { "200 OK" };
                let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", part.len());
                let _ = s.write_all(part);
            }
        });
        let dir = std::env::temp_dir().join(format!("clamor-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = CatalogFile { quant: "Q8_0".into(), file: "m.gguf".into(), size: body.len() as u64, sha256: hex::encode(Sha256::digest(&body)) };
        // A partial file from an earlier try.
        std::fs::write(part_path(&dir, "m.gguf"), &body[..100_000]).unwrap();
        DOWNLOADS.lock().unwrap().push(Progress { model: "t".into(), quant: "Q8_0".into(), file: "m.gguf".into(), done: 0, total: f.size, state: "downloading".into(), error: None, cancel: Default::default() });
        fetch(&url, &dir, &f, &AtomicBool::new(false)).unwrap();
        assert_eq!(std::fs::read(dir.join("m.gguf")).unwrap(), body);
        assert!(!part_path(&dir, "m.gguf").exists());
        // Wrong contents (the catalog expects other bytes): refused, nothing left behind.
        let bad = CatalogFile { file: "n.gguf".into(), sha256: "00".repeat(32), ..f.clone() };
        let e = fetch(&url, &dir, &bad, &AtomicBool::new(false)).unwrap_err();
        assert!(format!("{e:#}").contains("does not match"), "{e:#}");
        assert!(!dir.join("n.gguf").exists() && !part_path(&dir, "n.gguf").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
