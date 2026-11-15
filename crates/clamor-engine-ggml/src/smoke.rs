//! Matmul on a chosen ggml device: correctness check against a CPU reference
//! and a rough GFLOP/s number (for the muster tool to consume later).
use crate::{sys, Error, Result};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MatmulReport {
    pub device: String,
    pub max_abs_err: f32,
    pub gflops: f64,
}

struct Run {
    ctx: *mut sys::ggml_context,
    buf: sys::ggml_backend_buffer_t,
    backend: sys::ggml_backend_t,
    gf: *mut sys::ggml_cgraph,
    out: *mut sys::ggml_tensor,
}
impl Drop for Run {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_buffer_free(self.buf);
            sys::ggml_free(self.ctx);
            sys::ggml_backend_free(self.backend);
        }
    }
}

/// out[n x m] = a[k x m]^T-style ggml mul_mat: a is (k, m), b is (k, n), result (m, n).
fn setup(device: usize, m: i64, n: i64, k: i64, a: &[f32], b: &[f32]) -> Result<Run> {
    unsafe {
        let dev = sys::ggml_backend_dev_get(device);
        if dev.is_null() {
            return Err(Error::NoDevice(device.to_string()));
        }
        let backend = sys::ggml_backend_dev_init(dev, std::ptr::null());
        if backend.is_null() {
            return Err(Error::Ggml("backend init failed".into()));
        }
        let params = sys::ggml_init_params { mem_size: 1 << 20, mem_buffer: std::ptr::null_mut(), no_alloc: true };
        let ctx = sys::ggml_init(params);
        let ta = sys::ggml_new_tensor_2d(ctx, sys::ggml_type::GGML_TYPE_F32, k, m);
        let tb = sys::ggml_new_tensor_2d(ctx, sys::ggml_type::GGML_TYPE_F32, k, n);
        let out = sys::ggml_mul_mat(ctx, ta, tb);
        let gf = sys::ggml_new_graph(ctx);
        sys::ggml_build_forward_expand(gf, out);
        let buf = sys::ggml_backend_alloc_ctx_tensors(ctx, backend);
        if buf.is_null() {
            sys::ggml_free(ctx);
            sys::ggml_backend_free(backend);
            return Err(Error::Ggml("alloc failed".into()));
        }
        sys::ggml_backend_tensor_set(ta, a.as_ptr().cast(), 0, a.len() * 4);
        sys::ggml_backend_tensor_set(tb, b.as_ptr().cast(), 0, b.len() * 4);
        Ok(Run { ctx, buf, backend, gf, out })
    }
}

fn compute(r: &Run) -> Result<()> {
    let st = unsafe { sys::ggml_backend_graph_compute(r.backend, r.gf) };
    if !matches!(st, sys::ggml_status::GGML_STATUS_SUCCESS) {
        return Err(Error::Ggml(format!("graph compute failed: {st:?}")));
    }
    Ok(())
}

fn fill(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

pub fn matmul_check(device: usize) -> Result<MatmulReport> {
    let (m, n, k) = (48usize, 16usize, 64usize);
    let a = fill(m * k, 1);
    let b = fill(n * k, 2);
    let r = setup(device, m as i64, n as i64, k as i64, &a, &b)?;
    compute(&r)?;
    let mut out = vec![0f32; m * n];
    unsafe { sys::ggml_backend_tensor_get(r.out, out.as_mut_ptr().cast(), 0, out.len() * 4) };
    let mut max_err = 0f32;
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0f32;
            for x in 0..k {
                acc += a[i * k + x] * b[j * k + x];
            }
            max_err = max_err.max((acc - out[j * m + i]).abs());
        }
    }
    let name = crate::list_devices().get(device).map(|d| d.name.clone()).unwrap_or_default();
    Ok(MatmulReport { device: name, max_abs_err: max_err, gflops: 0.0 })
}

pub fn matmul_gflops(device: usize, dim: usize, iters: usize) -> Result<f64> {
    let a = fill(dim * dim, 3);
    let b = fill(dim * dim, 4);
    let r = setup(device, dim as i64, dim as i64, dim as i64, &a, &b)?;
    compute(&r)?; // warm-up
    let t = Instant::now();
    for _ in 0..iters {
        compute(&r)?;
    }
    let secs = t.elapsed().as_secs_f64();
    Ok(2.0 * (dim as f64).powi(3) * iters as f64 / secs / 1e9)
}
