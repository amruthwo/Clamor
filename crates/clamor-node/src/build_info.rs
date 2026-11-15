//! What this binary is: version, platform, GPU variant, CPU level. Used by `clamor-node version`,
//! reported to the coordinator, and matched against release manifests by the updater.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Clamor version, e.g. "0.4.0".
    pub version: String,
    /// Rust target triple, e.g. "x86_64-unknown-linux-gnu".
    pub target: String,
    /// GPU backends compiled in: "cpu", "vulkan", "metal", "cuda", ...
    pub variant: String,
    /// CPU instruction-set level: "x86-64-v3", "armv8.2-a+dotprod+fp16", "generic", "native".
    pub cpu: String,
    /// Git commit (first 12 hex digits) or "unknown".
    pub commit: String,
}

pub fn current() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").into(),
        target: env!("CLAMOR_TARGET").into(),
        variant: env!("CLAMOR_VARIANT").into(),
        cpu: env!("CLAMOR_CPU_LEVEL").into(),
        commit: env!("CLAMOR_COMMIT").into(),
    }
}

impl std::fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpu = if self.variant == "cpu" { "none" } else { &self.variant };
        write!(f, "clamor {} for {} (GPU backends: {gpu}; CPU level: {}; commit {})", self.version, self.target, self.cpu, self.commit)
    }
}
