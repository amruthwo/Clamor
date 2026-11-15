//! Builds the vendored ggml with CMake and links it statically.
//! GPU backends are opt-in cargo features (cuda, vulkan, metal, hip, opencl).
use std::env;
use std::path::PathBuf;

fn on(feature: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", feature.to_uppercase())).is_ok()
}

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../vendor/ggml");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let mut cfg = cmake::Config::new(&root);
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .profile("Release");

    // CPU instruction sets. With GGML_NATIVE off, ggml enables *no* SIMD extensions on x86 (no
    // AVX/AVX2/FMA), which is several times slower. CLAMOR_GGML_CPU picks a level:
    //   native        tune for this machine (don't copy the binary elsewhere)
    //   x86-64-v3     AVX2, FMA, F16C, BMI2: Intel Haswell (2013) / AMD Zen (2017) and newer
    //   generic       baseline only (SSE2 on x86, NEON on arm64)
    //   armv8.x-a...  passed to -march on arm64, e.g. armv8.2-a+dotprod+fp16
    // Default: x86-64-v3 on x86_64, generic elsewhere. CLAMOR_GGML_NATIVE=1 still means native.
    println!("cargo:rerun-if-env-changed=CLAMOR_GGML_CPU");
    println!("cargo:rerun-if-env-changed=CLAMOR_GGML_NATIVE");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let level = match (env::var("CLAMOR_GGML_CPU"), env::var("CLAMOR_GGML_NATIVE").as_deref()) {
        (Ok(l), _) => l,
        (Err(_), Ok("1")) => "native".to_string(),
        _ if arch == "x86_64" => "x86-64-v3".to_string(),
        _ => "generic".to_string(),
    };
    cfg.define("GGML_NATIVE", if level == "native" { "ON" } else { "OFF" });
    match level.as_str() {
        "native" | "generic" => {}
        "x86-64-v3" if arch == "x86_64" => {
            for o in ["GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"] {
                cfg.define(o, "ON");
            }
        }
        l if l.starts_with("armv") && arch == "aarch64" => {
            cfg.define("GGML_CPU_ARM_ARCH", l);
        }
        l => panic!("CLAMOR_GGML_CPU={l} does not apply to target arch {arch}"),
    }

    let mut libs: Vec<&str> = vec!["ggml", "ggml-cpu", "ggml-base"];
    if on("cuda") {
        cfg.define("GGML_CUDA", "ON");
        libs.push("ggml-cuda");
    }
    if on("vulkan") {
        cfg.define("GGML_VULKAN", "ON");
        libs.push("ggml-vulkan");
    }
    if on("hip") {
        cfg.define("GGML_HIP", "ON");
        libs.push("ggml-hip");
    }
    if on("opencl") {
        cfg.define("GGML_OPENCL", "ON");
        libs.push("ggml-opencl");
    }
    // ggml turns its BLAS backend on by default for Apple targets (Accelerate). Decide explicitly so
    // the libraries we link always match what CMake built.
    if target_os == "macos" || target_os == "ios" {
        cfg.define("GGML_BLAS", "ON").define("GGML_BLAS_VENDOR", "Apple");
        libs.push("ggml-blas");
    } else {
        cfg.define("GGML_BLAS", "OFF");
    }
    if on("metal") {
        cfg.define("GGML_METAL", "ON").define("GGML_METAL_EMBED_LIBRARY", "ON");
        libs.push("ggml-metal");
    }

    if target_os == "android" {
        // cargo-ndk sets ANDROID_NDK_HOME; hand the toolchain file to CMake.
        if let Ok(ndk) = env::var("ANDROID_NDK_HOME").or_else(|_| env::var("ANDROID_NDK_ROOT")) {
            cfg.define("CMAKE_TOOLCHAIN_FILE", format!("{ndk}/build/cmake/android.toolchain.cmake"));
            cfg.define("ANDROID_ABI", "arm64-v8a").define("ANDROID_PLATFORM", "android-28");
        }
    }

    let dst = cfg.build();
    for sub in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={}", dst.join(sub).display());
    }
    for l in &libs {
        println!("cargo:rustc-link-lib=static={l}");
    }

    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            if on("metal") {
                for f in ["Foundation", "Metal", "MetalKit"] {
                    println!("cargo:rustc-link-lib=framework={f}");
                }
            }
        }
        // ggml-cpu reads CPU info from the registry (RegOpenKeyExA etc.).
        "windows" => println!("cargo:rustc-link-lib=advapi32"),
        "android" => {
            println!("cargo:rustc-link-lib=c++_shared");
            println!("cargo:rustc-link-lib=log");
            if on("vulkan") {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        _ => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
            if on("vulkan") {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
    }
}
