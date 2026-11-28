//! What this binary is: version, platform, GPU variant, CPU level. Used by `clamor-node version`,
//! reported to the coordinator, and matched against release manifests by the updater.
pub use clamor_proto::BuildInfo;

/// Messages this build understands beyond its protocol version (see `BuildInfo::features`).
pub const FEATURES: &[&str] = &["handover"];

pub fn current() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").into(),
        target: env!("CLAMOR_TARGET").into(),
        variant: env!("CLAMOR_VARIANT").into(),
        cpu: env!("CLAMOR_CPU_LEVEL").into(),
        commit: env!("CLAMOR_COMMIT").into(),
        features: FEATURES.iter().map(|f| f.to_string()).collect(),
    }
}

/// One line for humans.
pub fn describe(b: &BuildInfo) -> String {
    let gpu = if b.variant == "cpu" { "none" } else { &b.variant };
    format!("clamor {} for {} (GPU backends: {gpu}; CPU level: {}; commit {})", b.version, b.target, b.cpu, b.commit)
}
