//! Static hardware discovery. Best effort and dependency-light: anything we cannot detect
//! on a platform is reported as absent rather than guessed.

use clamor_proto::*;
use std::process::Command;
use sysinfo::{Disks, System};

/// This device's name in a swarm: `CLAMOR_DEVICE_NAME` (the Android app sets the phone's name from
/// its settings), else the host name. Android's host name is "localhost" for every phone, so there
/// the model is used instead.
pub fn default_node_id() -> NodeId {
    let from_env = std::env::var("CLAMOR_DEVICE_NAME").ok().map(|n| device_name(&n)).filter(|n| !n.is_empty());
    let host = System::host_name().ok().map(|h| device_name(&h)).filter(|h| !h.is_empty() && h != "localhost");
    let model = || {
        if cfg!(target_os = "android") {
            Command::new("/system/bin/getprop").arg("ro.product.model").output().ok().map(|o| device_name(&String::from_utf8_lossy(&o.stdout))).filter(|m| !m.is_empty())
        } else {
            None
        }
    };
    NodeId(from_env.or(host).or_else(model).unwrap_or_else(|| "unknown-host".into()))
}

/// A device name as an id: letters, digits, '.', '_' and '-' (anything else becomes '-'), at most 48.
pub fn device_name(raw: &str) -> String {
    let mut out = String::new();
    for c in raw.trim().chars() {
        let c = if c.is_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '-' };
        if !(c == '-' && out.ends_with('-')) {
            out.push(c);
        }
    }
    out.trim_matches('-').chars().take(48).collect()
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
        // Filled in by clamor-node, which knows how it picks threads.
        perf_cores: None,
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
        let l = self.sample_load();
        (l.total, l.ram_available)
    }

    /// Whole-machine load, and how much of it is this process.
    pub fn sample_load(&mut self) -> Load {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let pid = sysinfo::Pid::from_u32(std::process::id());
        self.sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let ncpu = self.sys.cpus().len().max(1) as f32;
        // A process's usage is in percent of one CPU.
        let own = self.sys.process(pid).map(|p| p.cpu_usage() / 100.0 / ncpu).unwrap_or(0.0);
        let total = self.sys.global_cpu_usage() / 100.0;
        Load { total, own: own.min(total), ram_available: self.sys.available_memory(), ram_total: self.sys.total_memory() }
    }
}

/// One load sample. Fractions are of the whole machine (all CPUs).
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    pub total: f32,
    /// This process's share (Clamor's own work).
    pub own: f32,
    pub ram_available: u64,
    pub ram_total: u64,
}

impl Load {
    /// What everything except Clamor is using: how busy the owner keeps the machine.
    pub fn others(&self) -> f32 {
        (self.total - self.own).max(0.0)
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

/// Battery state, or None on a machine without a battery.
pub fn read_power() -> Option<PowerState> {
    // Set by the Android app, which reads the battery through Android (apps may not read the
    // kernel's power-supply files there) and keeps this file current.
    if let Some(p) = std::env::var_os("CLAMOR_POWER_FILE").and_then(|f| std::fs::read_to_string(f).ok()) {
        return parse_power_file(&p);
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        read_power_sysfs(std::path::Path::new("/sys/class/power_supply"))
    }
    #[cfg(target_os = "windows")]
    {
        read_power_windows()
    }
    #[cfg(target_os = "macos")]
    {
        read_power_macos()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows", target_os = "macos")))]
    {
        None
    }
}

/// `<percent> <plugged in: 0 or 1>`, e.g. "85 1".
pub(crate) fn parse_power_file(s: &str) -> Option<PowerState> {
    let mut it = s.split_whitespace();
    let pct: u8 = it.next()?.parse().ok()?;
    let plugged = it.next()? == "1";
    Some(PowerState { on_battery: !plugged, battery_pct: Some(pct.min(100)) })
}

/// Linux power supplies, identified by their `type` file, not their name: batteries are called
/// BAT0 on most PCs but `macsmc-battery` on Apple silicon under Asahi, and differently again on
/// phones. On mains when any Mains or USB supply is online, or the battery says it is charging.
#[cfg(any(target_os = "linux", target_os = "android", test))]
pub(crate) fn read_power_sysfs(root: &std::path::Path) -> Option<PowerState> {
    let read = |p: &std::path::Path, f: &str| std::fs::read_to_string(p.join(f)).map(|s| s.trim().to_string()).unwrap_or_default();
    let mut battery: Option<(Option<u8>, String)> = None;
    let mut mains_online = false;
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let p = entry.path();
        match read(&p, "type").as_str() {
            "Battery" if read(&p, "scope") != "Device" => {
                // A device battery (a mouse, a pen) is not this machine's battery.
                if battery.is_none() {
                    battery = Some((read(&p, "capacity").parse().ok(), read(&p, "status")));
                }
            }
            "Mains" | "USB" | "USB_C" | "USB_PD" => {
                if read(&p, "online") == "1" {
                    mains_online = true;
                }
            }
            _ => {}
        }
    }
    let (pct, status) = battery?;
    let charging = matches!(status.as_str(), "Charging" | "Full" | "Not charging");
    Some(PowerState { on_battery: !(mains_online || charging), battery_pct: pct })
}

#[cfg(target_os = "windows")]
fn read_power_windows() -> Option<PowerState> {
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    let mut s: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
    if unsafe { GetSystemPowerStatus(&mut s) } == 0 {
        return None;
    }
    // BatteryFlag 128: no system battery; 255: unknown.
    if s.BatteryFlag == 128 || s.BatteryFlag == 255 {
        return None;
    }
    Some(PowerState { on_battery: s.ACLineStatus == 0, battery_pct: (s.BatteryLifePercent <= 100).then_some(s.BatteryLifePercent) })
}

#[cfg(target_os = "macos")]
fn read_power_macos() -> Option<PowerState> {
    // `pmset -g batt`: "Now drawing from 'Battery Power'" / "'AC Power'", then " 85%; ...".
    let out = std::process::Command::new("pmset").args(["-g", "batt"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    if !text.contains("InternalBattery") {
        return None;
    }
    let pct = text.split('%').next().and_then(|s| s.rsplit(|c: char| !c.is_ascii_digit()).next()).and_then(|d| d.parse().ok());
    Some(PowerState { on_battery: text.contains("'Battery Power'"), battery_pct: pct })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_supply(root: &std::path::Path, name: &str, files: &[(&str, &str)]) {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        for (f, v) in files {
            std::fs::write(d.join(f), format!("{v}\n")).unwrap();
        }
    }

    /// Regression: only supplies named BAT* were checked, so an Apple silicon laptop under
    /// Asahi (battery `macsmc-battery`) looked like a desktop with no battery.
    #[test]
    fn batteries_are_found_by_type_not_name() {
        let root = std::env::temp_dir().join(format!("clamor-psy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Asahi M1, plugged in and full.
        fake_supply(&root, "macsmc-ac", &[("type", "Mains"), ("online", "1")]);
        fake_supply(&root, "macsmc-battery", &[("type", "Battery"), ("status", "Full"), ("capacity", "100")]);
        fake_supply(&root, "tps6598x-source-psy-0-0038", &[("type", "USB"), ("online", "1")]);
        assert_eq!(read_power_sysfs(&root), Some(PowerState { on_battery: false, battery_pct: Some(100) }));
        // Unplugged.
        fake_supply(&root, "macsmc-ac", &[("type", "Mains"), ("online", "0")]);
        fake_supply(&root, "tps6598x-source-psy-0-0038", &[("type", "USB"), ("online", "0")]);
        fake_supply(&root, "macsmc-battery", &[("type", "Battery"), ("status", "Discharging"), ("capacity", "57")]);
        assert_eq!(read_power_sysfs(&root), Some(PowerState { on_battery: true, battery_pct: Some(57) }));
        // A wireless mouse's battery is not the machine's.
        let desk = root.join("desk");
        fake_supply(&desk, "hidpp_battery_0", &[("type", "Battery"), ("scope", "Device"), ("status", "Discharging"), ("capacity", "40")]);
        assert_eq!(read_power_sysfs(&desk), None, "a desktop with a wireless mouse has no battery");
    }

    #[test]
    fn the_android_apps_power_file() {
        assert_eq!(parse_power_file("85 1\n"), Some(PowerState { on_battery: false, battery_pct: Some(85) }));
        assert_eq!(parse_power_file("40 0"), Some(PowerState { on_battery: true, battery_pct: Some(40) }));
        assert_eq!(parse_power_file(""), None);
        assert_eq!(parse_power_file("x 1"), None);
    }

    #[test]
    fn probe_reports_sane_values() {
        let info = probe_node(NodeId("t".into()));
        assert!(info.cpu_threads >= 1);
        assert!(info.ram_total_bytes > 0);
    }

    #[test]
    fn device_names_are_tidy_ids() {
        assert_eq!(device_name("  Galaxy S24 Ultra "), "Galaxy-S24-Ultra");
        assert_eq!(device_name("Bea's laptop"), "Bea-s-laptop");
        assert_eq!(device_name("ryzen-mini.local"), "ryzen-mini.local");
        assert_eq!(device_name("SM-S928U1\n"), "SM-S928U1");
        assert_eq!(device_name("   "), "");
    }

    #[test]
    fn android_is_phone() {
        assert_eq!(classify(Os::Android, "aarch64", 8 << 30, true), DeviceClass::Phone);
        assert_eq!(classify(Os::Linux, "x86_64", 32 << 30, false), DeviceClass::Desktop);
        assert_eq!(classify(Os::Windows, "x86_64", 16 << 30, true), DeviceClass::Laptop);
    }
}
