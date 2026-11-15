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
    /// Tensor shapes and types only (no data), owned by us.
    meta: *mut sys::ggml_context,
}
impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            sys::gguf_free(self.ctx);
            if !self.meta.is_null() {
                sys::ggml_free(self.meta);
            }
        }
    }
}

impl GgufFile {
    pub fn open(path: &str) -> Result<Self> {
        let c = CString::new(path).map_err(|e| Error::Gguf(e.to_string()))?;
        let mut meta: *mut sys::ggml_context = std::ptr::null_mut();
        let params = sys::gguf_init_params { no_alloc: true, ctx: &mut meta };
        let ctx = unsafe { sys::gguf_init_from_file(c.as_ptr(), params) };
        if ctx.is_null() {
            return Err(Error::Gguf(format!("cannot parse {path}")));
        }
        Ok(Self { ctx, meta })
    }

    /// Same as `open`; kept as the explicit name for callers that need tensor shapes.
    pub fn open_with_meta(path: &str) -> Result<Self> {
        Self::open(path)
    }

    /// The shape/type-only tensor for `name` (data pointer is null).
    pub(crate) fn meta_tensor(&self, name: &str) -> Option<*mut sys::ggml_tensor> {
        let c = CString::new(name).ok()?;
        let t = unsafe { sys::ggml_get_tensor(self.meta, c.as_ptr()) };
        (!t.is_null()).then_some(t)
    }

    /// ggml shape (`ne`, innermost first) of tensor `name`.
    pub fn tensor_ne(&self, name: &str) -> Option<[i64; 4]> {
        self.meta_tensor(name).map(|t| unsafe { (*t).ne })
    }

    /// Absolute byte offset of tensor `name`'s data in the file.
    pub fn tensor_file_offset(&self, name: &str) -> Option<u64> {
        let c = CString::new(name).ok()?;
        let id = unsafe { sys::gguf_find_tensor(self.ctx, c.as_ptr()) };
        (id >= 0).then(|| unsafe { (sys::gguf_get_data_offset(self.ctx) + sys::gguf_get_tensor_offset(self.ctx, id)) as u64 })
    }

    fn key(&self, key: &str) -> Option<i64> {
        let c = CString::new(key).ok()?;
        let id = unsafe { sys::gguf_find_key(self.ctx, c.as_ptr()) };
        (id >= 0).then_some(id)
    }

    fn kv_type(&self, id: i64) -> sys::gguf_type {
        unsafe { sys::gguf_get_kv_type(self.ctx, id) }
    }

    // ggml aborts the process on a type mismatch, so every accessor checks the type first and
    // returns None for the wrong one.

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        let i = self.key(key)?;
        match self.kv_type(i) {
            sys::gguf_type::GGUF_TYPE_UINT32 => Some(unsafe { sys::gguf_get_val_u32(self.ctx, i) }),
            sys::gguf_type::GGUF_TYPE_INT32 => u32::try_from(unsafe { sys::gguf_get_val_i32(self.ctx, i) }).ok(),
            _ => None,
        }
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        let i = self.key(key)?;
        (self.kv_type(i) == sys::gguf_type::GGUF_TYPE_FLOAT32).then(|| unsafe { sys::gguf_get_val_f32(self.ctx, i) })
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        let i = self.key(key)?;
        (self.kv_type(i) == sys::gguf_type::GGUF_TYPE_BOOL).then(|| unsafe { sys::gguf_get_val_bool(self.ctx, i) })
    }
    pub fn get_str(&self, key: &str) -> Option<String> {
        let i = self.key(key)?;
        (self.kv_type(i) == sys::gguf_type::GGUF_TYPE_STRING).then(|| cstr(unsafe { sys::gguf_get_val_str(self.ctx, i) }))
    }

    fn arr(&self, key: &str, elem: sys::gguf_type) -> Option<(i64, usize)> {
        let i = self.key(key)?;
        if self.kv_type(i) != sys::gguf_type::GGUF_TYPE_ARRAY || unsafe { sys::gguf_get_arr_type(self.ctx, i) } != elem {
            return None;
        }
        Some((i, unsafe { sys::gguf_get_arr_n(self.ctx, i) }))
    }
    /// A string array, e.g. `tokenizer.ggml.tokens`. Strings are raw bytes in GGUF; invalid
    /// UTF-8 is replaced.
    pub fn get_arr_str(&self, key: &str) -> Option<Vec<String>> {
        let (i, n) = self.arr(key, sys::gguf_type::GGUF_TYPE_STRING)?;
        Some((0..n).map(|k| cstr(unsafe { sys::gguf_get_arr_str(self.ctx, i, k) })).collect())
    }
    /// An int32 array, e.g. `tokenizer.ggml.token_type`.
    pub fn get_arr_i32(&self, key: &str) -> Option<Vec<i32>> {
        let (i, n) = self.arr(key, sys::gguf_type::GGUF_TYPE_INT32)?;
        let p = unsafe { sys::gguf_get_arr_data(self.ctx, i) } as *const i32;
        Some(unsafe { std::slice::from_raw_parts(p, n) }.to_vec())
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
