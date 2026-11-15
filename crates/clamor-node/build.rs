//! Records what this binary is, so it can tell the coordinator and pick matching updates.
use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    // GPU backends compiled in, from the cargo features (cpu if none).
    let gpus: Vec<&str> = ["cuda", "vulkan", "metal", "hip", "opencl"].into_iter().filter(|f| env::var(format!("CARGO_FEATURE_{}", f.to_uppercase())).is_ok()).collect();
    let variant = if gpus.is_empty() { "cpu".to_string() } else { gpus.join("+") };
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    // Mirrors the default in clamor-engine-ggml's build.rs.
    let cpu = env::var("CLAMOR_GGML_CPU").unwrap_or_else(|_| match env::var("CLAMOR_GGML_NATIVE").as_deref() {
        Ok("1") => "native".into(),
        _ if arch == "x86_64" => "x86-64-v3".into(),
        _ => "generic".into(),
    });
    let commit = env::var("GITHUB_SHA").ok().or_else(|| {
        std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    });
    println!("cargo:rustc-env=CLAMOR_TARGET={target}");
    println!("cargo:rustc-env=CLAMOR_VARIANT={variant}");
    println!("cargo:rustc-env=CLAMOR_CPU_LEVEL={cpu}");
    println!("cargo:rustc-env=CLAMOR_COMMIT={}", commit.as_deref().map(|c| &c[..c.len().min(12)]).unwrap_or("unknown"));
    for v in ["CLAMOR_GGML_CPU", "CLAMOR_GGML_NATIVE", "GITHUB_SHA"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
