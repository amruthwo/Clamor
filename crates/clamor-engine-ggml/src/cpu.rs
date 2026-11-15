//! How many CPU threads a stage should use.
//!
//! All logical CPUs is the wrong default for inference. ggml splits every step evenly across its
//! threads and waits for the slowest, so threads on efficiency cores (Apple M-series, Intel
//! 12th gen+, phones) or on hyperthread siblings drag the whole step down. Measured on an M1
//! under Asahi Linux, SmolLM2-360M decode: 4 threads (the performance cores) 70-77 tok/s;
//! 8 threads 4.5-53 tok/s and erratic.

/// Threads to use: the performance cores where they can be told apart, else physical cores.
pub fn default_threads() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    detect().unwrap_or(logical).clamp(1, logical.max(1))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect() -> Option<usize> {
    use std::fs;
    let cpus: Vec<std::path::PathBuf> = fs::read_dir("/sys/devices/system/cpu")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.strip_prefix("cpu").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))))
        .filter(|p| fs::read_to_string(p.join("online")).map(|s| s.trim() == "1").unwrap_or(true))
        .collect();
    let read = |p: &std::path::Path, f: &str| fs::read_to_string(p.join(f)).ok().and_then(|s| s.trim().parse::<u64>().ok());
    // Arm big.LITTLE (Apple, phones): the kernel's per-CPU capacity.
    let caps: Vec<u64> = cpus.iter().filter_map(|p| read(p, "cpu_capacity")).collect();
    if caps.len() == cpus.len() {
        if let Some(n) = fast_cores_by_capacity(&caps) {
            return Some(n);
        }
    }
    // Intel hybrid: performance cores are listed separately.
    if let Ok(list) = fs::read_to_string("/sys/devices/cpu_core/cpus") {
        if let Some(n) = count_cpu_list(&list) {
            return Some(physical(&cpus, Some(&list)).unwrap_or(n));
        }
    }
    physical(&cpus, None)
}

/// Physical cores (one per core, not per hyperthread), optionally only those in `only`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn physical(cpus: &[std::path::PathBuf], only: Option<&str>) -> Option<usize> {
    let keep: Option<std::collections::HashSet<usize>> = only.map(expand_cpu_list);
    let mut cores = std::collections::HashSet::new();
    for p in cpus {
        let id: usize = p.file_name()?.to_str()?.strip_prefix("cpu")?.parse().ok()?;
        if keep.as_ref().is_some_and(|k| !k.contains(&id)) {
            continue;
        }
        let pkg = std::fs::read_to_string(p.join("topology/physical_package_id")).ok()?;
        let core = std::fs::read_to_string(p.join("topology/core_id")).ok()?;
        cores.insert((pkg.trim().to_string(), core.trim().to_string()));
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(target_os = "macos")]
fn detect() -> Option<usize> {
    let q = |k: &str| std::process::Command::new("sysctl").args(["-n", k]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).and_then(|s| s.trim().parse().ok());
    q("hw.perflevel0.physicalcpu").or_else(|| q("hw.physicalcpu"))
}

#[cfg(target_os = "windows")]
fn detect() -> Option<usize> {
    // Heuristic until we query the processor topology: x86 Windows machines almost always
    // have two hyperthreads per core.
    let n = std::thread::available_parallelism().ok()?.get();
    Some(if cfg!(target_arch = "x86_64") && n >= 4 { n / 2 } else { n })
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "windows")))]
fn detect() -> Option<usize> {
    None
}

/// Cores with at least 60% of the fastest core's capacity, if capacities differ at all.
/// The cut keeps phone "mid" cores (close to the prime core) and drops efficiency cores
/// (about half or less).
pub(crate) fn fast_cores_by_capacity(caps: &[u64]) -> Option<usize> {
    let max = *caps.iter().max()?;
    if caps.iter().all(|&c| c == max) {
        return None;
    }
    Some(caps.iter().filter(|&&c| c * 10 >= max * 6).count())
}

/// "0-3,8,10-11" -> the set of CPU numbers.
pub(crate) fn expand_cpu_list(s: &str) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    out.extend(a..=b);
                }
            }
            None => {
                if let Ok(a) = part.parse() {
                    out.insert(a);
                }
            }
        }
    }
    out
}

pub(crate) fn count_cpu_list(s: &str) -> Option<usize> {
    let n = expand_cpu_list(s).len();
    (n > 0).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_tiers() {
        // Apple M1 under Asahi: 4 efficiency (493) + 4 performance (1024).
        assert_eq!(fast_cores_by_capacity(&[493, 493, 493, 493, 1024, 1024, 1024, 1024]), Some(4));
        // Phone with prime, mid and little cores: keep prime + mid.
        assert_eq!(fast_cores_by_capacity(&[280, 280, 900, 900, 960, 960, 960, 1024]), Some(6));
        // Uniform cores: no opinion (fall back to physical cores).
        assert_eq!(fast_cores_by_capacity(&[1024; 8]), None);
        assert_eq!(fast_cores_by_capacity(&[]), None);
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(count_cpu_list("0-3,8,10-11"), Some(7));
        assert_eq!(count_cpu_list("\n"), None);
        assert_eq!(expand_cpu_list("4-7"), (4..=7).collect());
    }

    #[test]
    fn default_is_sane() {
        let n = default_threads();
        assert!(n >= 1 && n <= std::thread::available_parallelism().unwrap().get());
    }
}
