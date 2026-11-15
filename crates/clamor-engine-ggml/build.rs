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

    // Native tuning is wrong for CI artifacts that run on other machines.
    let native = env::var("CLAMOR_GGML_NATIVE").map(|v| v == "1").unwrap_or(false);
    cfg.define("GGML_NATIVE", if native { "ON" } else { "OFF" });

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
        "windows" => {}
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
