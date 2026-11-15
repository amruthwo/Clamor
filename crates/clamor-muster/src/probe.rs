//! Static hardware discovery. Best effort and dependency-light: anything we cannot detect
//! on a platform is reported as absent rather than guessed.

use clamor_proto::*;
use std::process::Command;
use sysinfo::{Disks, System};

pub fn default_node_id() -> NodeId {
    NodeId(System::host_name().unwrap_or_else(|_| "unknown-host".into()))
}

pub fn probe_node(id: NodeId) -> NodeInfo {
    let mut sys = System::new().expect("system information is not available on this platform");
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let threads = sys.cpus().len().max(1) as u32;
    let cores = System::physical_core_count().map(|c| c as u32).unwrap_or(threads);
    let cpu_model = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
    let ram_total = sys.total_memory();

    let os = match std::env::consts::OS {
        "linux" => Os::Linux,
        "macos" => Os::Macos,
        "windows" => Os::Windows,
        "android" => Os::Android,
        _ => Os::Other,
    };
    let arch = std::env::consts::ARCH.to_string();
    let power = read_power();

    let disk_free = Disks::new_with_refreshed_list()
        .ok()
        .and_then(|d| d.iter().map(|x| x.available_space()).max())
        .unwrap_or(0);

    let accelerators = detect_accelerators(os, &arch, ram_total);
    let class = classify(os, &arch, ram_total, power.is_some());

    NodeInfo {
        id,
        hostname: System::host_name().unwrap_or_default(),
        os,
        arch,
        class,
        cpu_model,
        cpu_cores_physical: cores,
        cpu_threads: threads,
        ram_total_bytes: ram_total,
        ram_available_bytes: sys.available_memory(),
        accelerators,
        disk_free_bytes: disk_free,
        power,
        clamor_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Cheap repeated sampling of CPU load and free RAM, for heartbeats.
pub struct LoadSampler {
    sys: System,
}

impl LoadSampler {
    pub fn new() -> Self {
        let mut sys = System::new().expect("system information is not available on this platform");
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        LoadSampler { sys }
    }

    /// Returns (cpu load 0.0..1.0, available RAM in bytes).
    pub fn sample(&mut self) -> (f32, u64) {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        (self.sys.global_cpu_usage() / 100.0, self.sys.available_memory())
    }
}

impl Default for LoadSampler {
    fn default() -> Self {
        Self::new()
    }
}

fn classify(os: Os, arch: &str, ram_total: u64, has_battery: bool) -> DeviceClass {
    const GIB: u64 = 1 << 30;
    match os {
        Os::Android => DeviceClass::Phone,
        _ if has_battery => DeviceClass::Laptop,
        Os::Linux if arch.starts_with("aarch64") && ram_total <= 8 * GIB => DeviceClass::Sbc,
        _ => DeviceClass::Desktop,
    }
}

fn detect_accelerators(os: Os, arch: &str, ram_total: u64) -> Vec<Accelerator> {
    let mut out = Vec::new();

    // NVIDIA via nvidia-smi (Linux and Windows).
    if let Ok(o) = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                let mut parts = line.split(',');
                let name = parts.next().unwrap_or("").trim().to_string();
                let mib: u64 = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                if !name.is_empty() && mib > 0 {
                    out.push(Accelerator { name, backend: AccelBackend::Cuda, mem_bytes: mib << 20, unified: false });
                }
            }
        }
    }

    // Apple Silicon: unified memory. The GPU can use roughly three quarters of RAM.
    if os == Os::Macos && arch.starts_with("aarch64") {
        out.push(Accelerator {
            name: "Apple GPU".into(),
            backend: AccelBackend::Metal,
            mem_bytes: ram_total / 4 * 3,
            unified: true,
        });
    }
    out
}

#[cfg(target_os = "linux")]
fn read_power() -> Option<PowerState> {
    let dir = std::fs::read_dir("/sys/class/power_supply").ok()?;
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("BAT") {
            continue;
        }
        let p = entry.path();
        let pct = std::fs::read_to_string(p.join("capacity")).ok().and_then(|s| s.trim().parse::<u8>().ok());
        let status = std::fs::read_to_string(p.join("status")).unwrap_or_default();
        return Some(PowerState { on_battery: status.trim() == "Discharging", battery_pct: pct });
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn read_power() -> Option<PowerState> {
    // macOS, Windows and Android battery state need platform APIs; the Android app
    // reports its own PowerState, and desktop support is on the roadmap.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_sane_values() {
        let info = probe_node(NodeId("t".into()));
        assert!(info.cpu_threads >= 1);
        assert!(info.ram_total_bytes > 0);
    }

    #[test]
    fn android_is_phone() {
        assert_eq!(classify(Os::Android, "aarch64", 8 << 30, true), DeviceClass::Phone);
        assert_eq!(classify(Os::Linux, "x86_64", 32 << 30, false), DeviceClass::Desktop);
        assert_eq!(classify(Os::Windows, "x86_64", 16 << 30, true), DeviceClass::Laptop);
    }
}
