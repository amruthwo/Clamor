//! Being a good neighbour on someone's computer: Clamor uses what the device's policy allows
//! and steps aside when the owner needs the machine.
//!
//! * memory: a stage is only loaded if it fits in the policy's RAM share *and* in what is
//!   actually free, with a margin, so the machine never has to swap;
//! * CPU: stages use a policy share of the performance cores (and the process runs at low
//!   priority, see main.rs);
//! * pausing: a device stops taking new work when its owner paused it, when it is on battery
//!   and set to plugged-in only, or, in idle-only mode, while the owner keeps it busy. Work
//!   already running finishes.
use clamor_muster::probe::Load;
use clamor_proto::{Participation, Policy, PowerState};

/// Memory always left free, on top of the policy share.
pub const RAM_MARGIN: u64 = 512 << 20;

/// Bytes a stage may use on this device now.
pub fn ram_budget(policy: &Policy, ram_available: u64, ram_total: u64) -> u64 {
    let share = (ram_total as f64 * policy.max_ram_fraction.clamp(0.0, 1.0) as f64) as u64;
    share.min(ram_available.saturating_sub(RAM_MARGIN))
}

/// CPU threads for a stage: an explicit `--threads`, or the policy's share of the performance
/// cores. A device always keeps one of them for its owner unless it only has one.
pub fn threads_for(policy: &Policy, explicit: usize, performance_cores: usize) -> usize {
    if explicit > 0 {
        return explicit;
    }
    let n = performance_cores.max(1);
    let share = ((n as f32) * policy.max_cpu_fraction.clamp(0.0, 1.0)).round() as usize;
    share.clamp(1, if n > 1 { n - 1 } else { 1 })
}

/// Owner-busy detection for idle-only devices, with hysteresis so a brief burst of activity
/// does not flap the swarm: busy after 3 samples above 50% load (from anything but Clamor),
/// idle again after 6 samples below 25%. With 5 s heartbeats: 15 s and 30 s.
#[derive(Default, Debug)]
pub struct PauseState {
    busy_streak: u32,
    quiet_streak: u32,
    owner_busy: bool,
}

impl PauseState {
    /// Why this device should not take new work right now, if it should not.
    pub fn update(&mut self, policy: &Policy, load: &Load, power: Option<&PowerState>) -> Option<String> {
        let others = load.others();
        if others > 0.5 {
            self.busy_streak += 1;
            self.quiet_streak = 0;
        } else if others < 0.25 {
            self.quiet_streak += 1;
            self.busy_streak = 0;
        }
        if self.busy_streak >= 3 {
            self.owner_busy = true;
        }
        if self.quiet_streak >= 6 {
            self.owner_busy = false;
        }
        match policy.participation {
            Participation::Paused => return Some("paused by its owner".into()),
            Participation::Draining => return Some("leaving the swarm".into()),
            _ => {}
        }
        if policy.require_plugged_in && power.is_some_and(|p| p.on_battery) {
            return Some("on battery (set to work only when plugged in)".into());
        }
        if policy.participation == Participation::IdleOnly && self.owner_busy {
            return Some("its owner is using it (idle-only)".into());
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clamor_proto::DeviceClass;

    const GIB: u64 = 1 << 30;

    fn load(total: f32, own: f32) -> Load {
        Load { total, own, ram_available: 8 * GIB, ram_total: 16 * GIB }
    }

    #[test]
    fn memory_budget_respects_both_the_policy_and_what_is_free() {
        let laptop = Policy::default_for(DeviceClass::Laptop); // 50%
        assert_eq!(ram_budget(&laptop, 12 * GIB, 16 * GIB), 8 * GIB, "policy share");
        assert_eq!(ram_budget(&laptop, 3 * GIB, 16 * GIB), 3 * GIB - RAM_MARGIN, "what is free, minus a margin");
        assert_eq!(ram_budget(&laptop, 100 << 20, 16 * GIB), 0, "nearly full: nothing");
    }

    #[test]
    fn threads_follow_the_cpu_share_and_leave_one_core() {
        let laptop = Policy::default_for(DeviceClass::Laptop); // 50%
        let desktop = Policy::default_for(DeviceClass::Desktop); // 80%
        assert_eq!(threads_for(&laptop, 0, 4), 2);
        assert_eq!(threads_for(&desktop, 0, 8), 6);
        assert_eq!(threads_for(&desktop, 0, 4), 3, "always one core for the owner");
        assert_eq!(threads_for(&desktop, 0, 1), 1);
        assert_eq!(threads_for(&laptop, 7, 4), 7, "--threads wins");
    }

    #[test]
    fn pause_reasons() {
        let mut laptop = Policy::default_for(DeviceClass::Laptop); // idle-only, plugged-in only
        let mut s = PauseState::default();
        let mains = PowerState { on_battery: false, battery_pct: Some(80) };
        let battery = PowerState { on_battery: true, battery_pct: Some(80) };
        assert_eq!(s.update(&laptop, &load(0.1, 0.0), Some(&mains)), None);
        assert!(s.update(&laptop, &load(0.1, 0.0), Some(&battery)).unwrap().contains("battery"));
        laptop.participation = Participation::Paused;
        assert!(s.update(&laptop, &load(0.1, 0.0), Some(&mains)).unwrap().contains("owner"));
        // A desktop never pauses for battery or load.
        let desktop = Policy::default_for(DeviceClass::Desktop);
        let mut d = PauseState::default();
        for _ in 0..10 {
            assert_eq!(d.update(&desktop, &load(0.95, 0.0), None), None);
        }
    }

    #[test]
    fn owner_busy_has_hysteresis_and_ignores_clamors_own_load() {
        let laptop = Policy::default_for(DeviceClass::Laptop);
        let mains = PowerState { on_battery: false, battery_pct: None };
        let mut s = PauseState::default();
        // Clamor itself at 90%: not the owner.
        for _ in 0..10 {
            assert_eq!(s.update(&laptop, &load(0.95, 0.9), Some(&mains)), None);
        }
        // The owner at 70% for two samples: not yet.
        assert_eq!(s.update(&laptop, &load(0.7, 0.0), Some(&mains)), None);
        assert_eq!(s.update(&laptop, &load(0.7, 0.0), Some(&mains)), None);
        // Third sample: paused.
        assert!(s.update(&laptop, &load(0.7, 0.0), Some(&mains)).is_some());
        // Quiet for five samples: still paused; the sixth resumes.
        for _ in 0..5 {
            assert!(s.update(&laptop, &load(0.1, 0.0), Some(&mains)).is_some());
        }
        assert_eq!(s.update(&laptop, &load(0.1, 0.0), Some(&mains)), None);
    }
}

/// Lower the *calling thread's* CPU priority. Each stage computes on its own thread that calls
/// this, so only the number-crunching runs at low priority: heartbeats, the API and networking
/// stay at normal priority and cannot be starved when the owner's apps load the machine.
/// ggml's worker threads inherit it on Linux (nice is per thread) and macOS (QoS); on Windows
/// they are lowered by ggml itself (`StageOptions::low_priority`).
pub fn lower_this_thread() -> Result<(), String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        if libc::setpriority(libc::PRIO_PROCESS, tid, 10) != 0 {
            return Err("could not lower the compute thread's priority".into());
        }
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    unsafe {
        if libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) != 0 {
            return Err("could not lower the compute thread's priority".into());
        }
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        if SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) == 0 {
            return Err("could not lower the compute thread's priority".into());
        }
    }
    Ok(())
}

/// Reject policies that make no sense instead of storing them.
pub fn validate(p: &clamor_proto::Policy) -> Result<(), String> {
    if !(0.05..=0.95).contains(&p.max_ram_fraction) {
        return Err(format!("RAM share must be between 5% and 95%, not {:.0}%", p.max_ram_fraction * 100.0));
    }
    if !(0.05..=1.0).contains(&p.max_cpu_fraction) {
        return Err(format!("CPU share must be between 5% and 100%, not {:.0}%", p.max_cpu_fraction * 100.0));
    }
    Ok(())
}

// ---------------------------------------------------------------- remembered settings

static CONFIG_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// Where this device keeps its settings (`--config-dir`, default the OS config folder).
pub fn set_config_dir(dir: Option<std::path::PathBuf>) {
    let d = dir.unwrap_or_else(|| dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("clamor"));
    let _ = CONFIG_DIR.set(d);
}

fn settings_path() -> std::path::PathBuf {
    CONFIG_DIR.get_or_init(|| dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("clamor")).join("device-settings.json")
}

/// The settings this device last received (from the dashboard), if any. Read at startup, before
/// the priority is set: once lowered, Linux and macOS will not let it be raised again.
pub fn saved_settings() -> Option<clamor_proto::Policy> {
    let p = std::fs::read(settings_path()).ok()?;
    serde_json::from_slice(&p).ok().filter(|p| validate(p).is_ok())
}

/// Remember this device's settings for the next start.
pub fn save_settings(p: &clamor_proto::Policy) {
    let path = settings_path();
    let r = std::fs::create_dir_all(path.parent().unwrap()).and_then(|_| std::fs::write(&path, serde_json::to_vec_pretty(p).unwrap()));
    if let Err(e) = r {
        eprintln!("could not save settings to {}: {e}", path.display());
    }
}
