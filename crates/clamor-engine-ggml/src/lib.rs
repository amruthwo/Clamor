//! Clamor's ggml engine: raw bindings plus a small safe layer.
//!
//! Enumerate compute devices, run a matmul on any backend (smoke test + micro-benchmark),
//! read GGUF metadata, run a layer range of a Llama model (`LlamaStage`), and
//! run one in a ring pipeline stage (`GgmlExecutor`, a `clamor_transport` `StageExecutor`).
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
#[allow(warnings)]
pub mod bindings;
pub use bindings as sys;

mod devices;
mod executor;
mod gguf;
mod llama;
mod smoke;

pub use devices::{list_devices, DeviceInfo, DeviceKind};
pub use executor::{ExecutorOptions, GgmlExecutor, Sampling};
pub use gguf::{GgufFile, GgufTensorInfo};
pub use llama::{LlamaHparams, LlamaInput, LlamaOutput, LlamaStage, Logits, StageOptions};
pub use smoke::{matmul_check, matmul_gflops, MatmulReport};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ggml: {0}")]
    Ggml(String),
    #[error("no such device: {0}")]
    NoDevice(String),
    #[error("gguf: {0}")]
    Gguf(String),
    /// The device cannot run an op this graph needs (e.g. a virtual GPU missing a feature).
    /// Plan the stage onto another device.
    #[error("{device} does not support op {op}")]
    Unsupported { device: String, op: String },
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn cstr(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}
