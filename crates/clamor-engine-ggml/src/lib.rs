//! Clamor's ggml engine: raw bindings plus a small safe layer.
//!
//! Scope of this scaffold: enumerate compute devices, run a matmul on any
//! backend (smoke test + micro-benchmark), and read GGUF metadata. The layer-range
//! Llama graph and the `StageExecutor` implementation come next (see CLAUDE.md).
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
#[allow(warnings)]
pub mod bindings;
pub use bindings as sys;

mod devices;
mod gguf;
mod smoke;

pub use devices::{list_devices, DeviceInfo, DeviceKind};
pub use gguf::{GgufFile, GgufTensorInfo};
pub use smoke::{matmul_check, matmul_gflops, MatmulReport};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ggml: {0}")]
    Ggml(String),
    #[error("no such device: {0}")]
    NoDevice(String),
    #[error("gguf: {0}")]
    Gguf(String),
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn cstr(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}
