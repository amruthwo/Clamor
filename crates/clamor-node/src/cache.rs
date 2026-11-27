//! The slice cache: layer slices fetched from other devices (see share.rs), kept for next time.
//!
//! An index (`index.json` in the cache folder) remembers when each slice was last used and for
//! which swarms, so that:
//! * the cache stays under a size cap (`--cache-max`), dropping the least recently used slices;
//! * leaving a swarm with `--purge` deletes the slices only that swarm used.
//!
//! Files the index does not know (an older cache) count as last used at their modification time.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub last_used_unix: u64,
    pub swarms: Vec<String>,
}

/// Slice path (relative to the cache folder, with `/`) -> entry.
type Index = BTreeMap<String, Entry>;

static LOCK: Mutex<()> = Mutex::new(());

fn index_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("index.json")
}

fn load(cache_dir: &Path) -> Index {
    std::fs::read(index_path(cache_dir)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn save(cache_dir: &Path, idx: &Index) {
    let p = index_path(cache_dir);
    let tmp = p.with_extension("tmp");
    let r = std::fs::create_dir_all(cache_dir).and_then(|_| std::fs::write(&tmp, serde_json::to_vec_pretty(idx).unwrap())).and_then(|_| std::fs::rename(&tmp, &p));
    if let Err(e) = r {
        eprintln!("could not update the cache index: {e}");
    }
}

fn rel(cache_dir: &Path, p: &Path) -> Option<String> {
    Some(p.strip_prefix(cache_dir).ok()?.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/"))
}

/// Every slice file in the cache: (relative path, size, modification time).
fn files(cache_dir: &Path) -> Vec<(String, u64, u64)> {
    let mut out = Vec::new();
    for d in std::fs::read_dir(cache_dir).into_iter().flatten().flatten() {
        if !d.path().is_dir() {
            continue;
        }
        for f in std::fs::read_dir(d.path()).into_iter().flatten().flatten() {
            let p = f.path();
            if p.extension().is_some_and(|e| e == "gguf") {
                let Ok(m) = f.metadata() else { continue };
                let mtime = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                if let Some(r) = rel(cache_dir, &p) {
                    out.push((r, m.len(), mtime));
                }
            }
        }
    }
    out
}

/// A slice was fetched or reused for `swarm`.
pub fn used(cache_dir: &Path, slice: &Path, swarm: Option<&str>) {
    let _g = LOCK.lock().unwrap();
    let Some(r) = rel(cache_dir, slice) else { return };
    let mut idx = load(cache_dir);
    let e = idx.entry(r).or_default();
    e.last_used_unix = clamor_proto::unix_now();
    if let Some(s) = swarm {
        if !e.swarms.iter().any(|x| x == s) {
            e.swarms.push(s.to_string());
        }
    }
    save(cache_dir, &idx);
}

/// Delete least recently used slices until the cache is at most `max_bytes`, never `keep`.
/// Returns the bytes freed.
pub fn enforce_cap(cache_dir: &Path, max_bytes: u64, keep: &Path) -> u64 {
    let _g = LOCK.lock().unwrap();
    let mut idx = load(cache_dir);
    let keep = rel(cache_dir, keep);
    let mut all: Vec<(u64, String, u64)> = files(cache_dir).into_iter().map(|(r, size, mtime)| (idx.get(&r).map(|e| e.last_used_unix).unwrap_or(mtime), r, size)).collect();
    let mut total: u64 = all.iter().map(|x| x.2).sum();
    all.sort();
    let mut freed = 0;
    for (_, r, size) in all {
        if total <= max_bytes {
            break;
        }
        if Some(&r) == keep.as_ref() {
            continue;
        }
        // A slice a running stage still has open cannot be deleted on Windows: skip it.
        if std::fs::remove_file(cache_dir.join(&r)).is_ok() {
            println!("cache over {} GB: removed {r} ({:.1} GB, least recently used)", max_bytes as f64 / 1e9, size as f64 / 1e9);
            idx.remove(&r);
            total -= size;
            freed += size;
        }
    }
    save(cache_dir, &idx);
    freed
}

/// Forget `swarm`: delete the slices only it used. Returns (files, bytes) deleted.
pub fn purge(cache_dir: &Path, swarm: &str) -> (usize, u64) {
    let _g = LOCK.lock().unwrap();
    let mut idx = load(cache_dir);
    let (mut n, mut bytes) = (0, 0);
    for (r, size, _) in files(cache_dir) {
        let Some(e) = idx.get_mut(&r) else { continue };
        e.swarms.retain(|s| s != swarm);
        if e.swarms.is_empty() && std::fs::remove_file(cache_dir.join(&r)).is_ok() {
            idx.remove(&r);
            n += 1;
            bytes += size;
        }
    }
    save(cache_dir, &idx);
    (n, bytes)
}

/// Delete every cached slice of one model (its folder is the model id's first 16 characters).
/// Returns (files, bytes) deleted.
pub fn purge_model(cache_dir: &Path, model_id: &str) -> (usize, u64) {
    let _g = LOCK.lock().unwrap();
    let short = &model_id[..model_id.len().min(16)];
    if short.is_empty() || short.contains(['/', '\\', '.']) {
        return (0, 0);
    }
    let mut idx = load(cache_dir);
    let (mut n, mut bytes) = (0, 0);
    for (r, size, _) in files(cache_dir) {
        if r.starts_with(&format!("{short}/")) && std::fs::remove_file(cache_dir.join(&r)).is_ok() {
            idx.remove(&r);
            n += 1;
            bytes += size;
        }
    }
    let _ = std::fs::remove_dir(cache_dir.join(short));
    save(cache_dir, &idx);
    (n, bytes)
}

/// Total size of the cache in bytes.
pub fn size(cache_dir: &Path) -> u64 {
    files(cache_dir).iter().map(|f| f.1).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(dir: &Path, name: &str, bytes: usize) -> PathBuf {
        let p = dir.join("abcd").join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, vec![0u8; bytes]).unwrap();
        p
    }

    #[test]
    fn cap_drops_least_recently_used_and_purge_only_one_swarms_slices() {
        let dir = std::env::temp_dir().join(format!("clamor-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (a, b, c) = (slice(&dir, "a.gguf", 100), slice(&dir, "b.gguf", 100), slice(&dir, "c.gguf", 100));
        used(&dir, &a, Some("home"));
        used(&dir, &b, Some("home"));
        used(&dir, &c, Some("friends"));
        // Make a the oldest, then b, then c.
        let mut idx = load(&dir);
        for (i, k) in ["abcd/a.gguf", "abcd/b.gguf", "abcd/c.gguf"].iter().enumerate() {
            idx.get_mut(*k).unwrap().last_used_unix = 1000 + i as u64;
        }
        save(&dir, &idx);
        assert_eq!(size(&dir), 300);
        // Room for two: a goes. Never the one being loaded, even if it is the oldest.
        assert_eq!(enforce_cap(&dir, 250, &b), 100);
        assert!(!a.exists() && b.exists() && c.exists());
        assert_eq!(enforce_cap(&dir, 150, &b), 100, "c is newer but b is in use");
        assert!(b.exists() && !c.exists());

        // b used by both swarms: purging one keeps it.
        let d = slice(&dir, "d.gguf", 50);
        used(&dir, &b, Some("friends"));
        used(&dir, &d, Some("friends"));
        assert_eq!(purge(&dir, "friends"), (1, 50));
        assert!(b.exists() && !d.exists());
        assert_eq!(purge(&dir, "home"), (1, 100));
        assert!(!b.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
