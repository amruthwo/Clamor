//! Minimal GGUF reader built on ggml's own parser (no tensor data loaded).
use crate::{cstr, sys, Error, Result};
use std::ffi::CString;

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_bytes: u64,
    pub offset: u64,
}

pub struct GgufFile {
    ctx: *mut sys::gguf_context,
}
impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe { sys::gguf_free(self.ctx) }
    }
}

impl GgufFile {
    pub fn open(path: &str) -> Result<Self> {
        let c = CString::new(path).map_err(|e| Error::Gguf(e.to_string()))?;
        let params = sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() };
        let ctx = unsafe { sys::gguf_init_from_file(c.as_ptr(), params) };
        if ctx.is_null() {
            return Err(Error::Gguf(format!("cannot parse {path}")));
        }
        Ok(Self { ctx })
    }

    fn key(&self, key: &str) -> Option<i64> {
        let c = CString::new(key).ok()?;
        let id = unsafe { sys::gguf_find_key(self.ctx, c.as_ptr()) };
        (id >= 0).then_some(id)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.key(key).map(|i| unsafe { sys::gguf_get_val_u32(self.ctx, i) })
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.key(key).map(|i| unsafe { sys::gguf_get_val_f32(self.ctx, i) })
    }
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.key(key).map(|i| cstr(unsafe { sys::gguf_get_val_str(self.ctx, i) }))
    }

    pub fn tensors(&self) -> Vec<GgufTensorInfo> {
        let n = unsafe { sys::gguf_get_n_tensors(self.ctx) };
        (0..n)
            .map(|i| unsafe {
                GgufTensorInfo {
                    name: cstr(sys::gguf_get_tensor_name(self.ctx, i)),
                    n_bytes: sys::gguf_get_tensor_size(self.ctx, i) as u64,
                    offset: sys::gguf_get_tensor_offset(self.ctx, i) as u64,
                }
            })
            .collect()
    }
}
