//! A Llama pipeline stage: layers `[a, b)` of a GGUF model on one ggml device.
//!
//! The stage that holds layer 0 is the head: it owns the token embedding and takes token ids.
//! The stage that holds the last layer is the tail: it owns the output norm and lm_head and
//! returns logits. Every other stage takes and returns hidden states `[n_tokens, n_embd]`.
//! A stage holding every layer is both.
//!
//! Only the weights for the stage's own layers are read from the file. Each request id gets
//! its own KV cache (allocated on first use, freed by `release`).
//!
//! Conventions are ggml's: weights are `[out, in]` row-major (ggml `ne = [in, out]`), and RoPE
//! rotates adjacent pairs (`GGML_ROPE_TYPE_NORMAL`). `tools/ref_forward.py` is the reference.
use crate::{gguf::GgufFile, sys, Error, Result};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::ptr::null_mut;

type T = *mut sys::ggml_tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaHparams {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_rot: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
}

impl LlamaHparams {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.get_str("general.architecture").unwrap_or_default();
        if arch != "llama" {
            return Err(Error::Gguf(format!("unsupported architecture {arch:?} (want llama)")));
        }
        let req = |k: &str| g.get_u32(&format!("llama.{k}")).ok_or_else(|| Error::Gguf(format!("missing llama.{k}")));
        let n_embd = req("embedding_length")?;
        let n_head = req("attention.head_count")?;
        let n_vocab = match g.get_u32("llama.vocab_size") {
            Some(v) => v,
            None => g.tensor_ne("token_embd.weight").map(|ne| ne[1] as u32).ok_or_else(|| Error::Gguf("no vocab size".into()))?,
        };
        Ok(Self {
            n_layer: req("block_count")?,
            n_embd,
            n_head,
            n_head_kv: g.get_u32("llama.attention.head_count_kv").unwrap_or(n_head),
            n_ff: req("feed_forward_length")?,
            n_vocab,
            n_ctx_train: req("context_length")?,
            n_rot: g.get_u32("llama.rope.dimension_count").unwrap_or(n_embd / n_head),
            rms_eps: g.get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            rope_freq_base: g.get_f32("llama.rope.freq_base").unwrap_or(10000.0),
        })
    }
    pub fn head_dim(&self) -> u32 {
        self.n_embd / self.n_head
    }
    /// Width of one layer's K (or V) row: all KV heads for one position.
    pub fn kv_dim(&self) -> u32 {
        self.head_dim() * self.n_head_kv
    }
}

#[derive(Debug, Clone)]
pub struct StageOptions {
    /// ggml device index (see `list_devices`). Order depends on the compiled-in backends:
    /// GPUs come first, so index 0 is the CPU only in a CPU-only build.
    pub device: usize,
    /// KV cache length per request. 0 means the model's training context.
    pub n_ctx: u32,
    /// CPU threads (CPU device only). 0 means all available.
    pub n_threads: usize,
}
impl Default for StageOptions {
    fn default() -> Self {
        Self { device: 0, n_ctx: 0, n_threads: 0 }
    }
}

pub enum LlamaInput<'a> {
    /// Head only.
    Tokens(&'a [u32]),
    /// Non-head stages: `n_tokens * n_embd` f32, token-major.
    Hidden(&'a [f32]),
}

#[derive(Debug, Clone)]
pub enum LlamaOutput {
    /// Non-tail stages: `n_tokens * n_embd` f32, token-major.
    Hidden(Vec<f32>),
    /// Tail only: `n_tokens * n_vocab` f32, token-major.
    Logits(Vec<f32>),
}

struct LayerW {
    attn_norm: T,
    wq: T,
    wk: T,
    wv: T,
    wo: T,
    ffn_norm: T,
    w_gate: T,
    w_up: T,
    w_down: T,
}

struct KvCache {
    ctx: *mut sys::ggml_context,
    buf: sys::ggml_backend_buffer_t,
    /// Per local layer: `[kv_dim, n_ctx]`.
    k: Vec<T>,
    v: Vec<T>,
    n_past: u32,
}
impl Drop for KvCache {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_buffer_free(self.buf);
            sys::ggml_free(self.ctx);
        }
    }
}

pub struct LlamaStage {
    hp: LlamaHparams,
    layers: Range<u32>,
    n_ctx: u32,
    backend: sys::ggml_backend_t,
    galloc: sys::ggml_gallocr_t,
    w_ctx: *mut sys::ggml_context,
    w_buf: sys::ggml_backend_buffer_t,
    tok_embd: Option<T>,
    blocks: Vec<LayerW>,
    out_norm: Option<T>,
    output: Option<T>,
    caches: HashMap<u64, KvCache>,
}

// The raw pointers are owned by this struct and only touched through &mut self.
unsafe impl Send for LlamaStage {}

impl Drop for LlamaStage {
    fn drop(&mut self) {
        self.caches.clear();
        unsafe {
            sys::ggml_gallocr_free(self.galloc);
            sys::ggml_backend_buffer_free(self.w_buf);
            sys::ggml_free(self.w_ctx);
            sys::ggml_backend_free(self.backend);
        }
    }
}

fn cname(s: &str) -> CString {
    CString::new(s).expect("tensor names have no NUL")
}

impl LlamaStage {
    /// Load layers `layers` of the GGUF model at `path` onto `opts.device`.
    pub fn load(path: &str, layers: Range<u32>, opts: StageOptions) -> Result<Self> {
        let g = GgufFile::open_with_meta(path)?;
        let hp = LlamaHparams::from_gguf(&g)?;
        if layers.start >= layers.end || layers.end > hp.n_layer {
            return Err(Error::Ggml(format!("bad layer range {layers:?} for a {}-layer model", hp.n_layer)));
        }
        if hp.n_embd % hp.n_head != 0 || hp.n_head % hp.n_head_kv != 0 {
            return Err(Error::Gguf("head counts do not divide evenly".into()));
        }
        let n_ctx = if opts.n_ctx == 0 { hp.n_ctx_train } else { opts.n_ctx };
        let is_head = layers.start == 0;
        let is_tail = layers.end == hp.n_layer;

        let mut names: Vec<String> = Vec::new();
        if is_head {
            names.push("token_embd.weight".into());
        }
        for l in layers.clone() {
            for s in ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"] {
                names.push(format!("blk.{l}.{s}.weight"));
            }
        }
        if is_tail {
            names.push("output_norm.weight".into());
            // Some models tie lm_head to the embedding and omit output.weight.
            let lm_head = if g.tensor_ne("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
            if !names.iter().any(|n| n == lm_head) {
                names.push(lm_head.into());
            }
        }

        unsafe {
            let dev = sys::ggml_backend_dev_get(opts.device);
            if dev.is_null() {
                return Err(Error::NoDevice(opts.device.to_string()));
            }
            let backend = sys::ggml_backend_dev_init(dev, std::ptr::null());
            if backend.is_null() {
                return Err(Error::Ggml("backend init failed".into()));
            }
            if sys::ggml_backend_is_cpu(backend) {
                let n = if opts.n_threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) } else { opts.n_threads };
                sys::ggml_backend_cpu_set_n_threads(backend, n as i32);
            }
            // From here on, `stage` owns everything, so an early return frees it.
            let mut stage = LlamaStage {
                hp: hp.clone(),
                layers: layers.clone(),
                n_ctx,
                backend,
                galloc: sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(backend)),
                w_ctx: sys::ggml_init(sys::ggml_init_params {
                    mem_size: sys::ggml_tensor_overhead() * (names.len() + 1),
                    mem_buffer: null_mut(),
                    no_alloc: true,
                }),
                w_buf: null_mut(),
                tok_embd: None,
                blocks: Vec::new(),
                out_norm: None,
                output: None,
                caches: HashMap::new(),
            };

            let mut sources: Vec<(T, u64)> = Vec::new();
            let mut by_name: HashMap<String, T> = HashMap::new();
            for n in &names {
                let meta = g.meta_tensor(n).ok_or_else(|| Error::Gguf(format!("missing tensor {n}")))?;
                let t = sys::ggml_dup_tensor(stage.w_ctx, meta);
                let c = cname(n);
                sys::ggml_set_name(t, c.as_ptr());
                sources.push((t, g.tensor_file_offset(n).unwrap()));
                by_name.insert(n.clone(), t);
            }
            stage.w_buf = sys::ggml_backend_alloc_ctx_tensors(stage.w_ctx, backend);
            if stage.w_buf.is_null() {
                return Err(Error::Ggml("cannot allocate weights on device".into()));
            }

            let mut f = File::open(path).map_err(|e| Error::Gguf(format!("{path}: {e}")))?;
            let mut scratch = Vec::new();
            for (t, off) in sources {
                let nbytes = sys::ggml_nbytes(t);
                scratch.resize(nbytes, 0u8);
                f.seek(SeekFrom::Start(off)).and_then(|_| f.read_exact(&mut scratch)).map_err(|e| Error::Gguf(format!("reading weights: {e}")))?;
                sys::ggml_backend_tensor_set(t, scratch.as_ptr().cast(), 0, nbytes);
            }

            let get = |n: String| by_name[&n];
            if is_head {
                stage.tok_embd = Some(get("token_embd.weight".into()));
            }
            for l in layers.clone() {
                let w = |s: &str| get(format!("blk.{l}.{s}.weight"));
                stage.blocks.push(LayerW {
                    attn_norm: w("attn_norm"),
                    wq: w("attn_q"),
                    wk: w("attn_k"),
                    wv: w("attn_v"),
                    wo: w("attn_output"),
                    ffn_norm: w("ffn_norm"),
                    w_gate: w("ffn_gate"),
                    w_up: w("ffn_up"),
                    w_down: w("ffn_down"),
                });
            }
            if is_tail {
                stage.out_norm = Some(get("output_norm.weight".into()));
                // Lazily: a tail-only stage has output.weight and never loads token_embd.
                stage.output = Some(by_name.get("output.weight").copied().unwrap_or_else(|| get("token_embd.weight".into())));
            }
            Ok(stage)
        }
    }

    pub fn hparams(&self) -> &LlamaHparams {
        &self.hp
    }
    pub fn layers(&self) -> Range<u32> {
        self.layers.clone()
    }
    pub fn is_head(&self) -> bool {
        self.layers.start == 0
    }
    pub fn is_tail(&self) -> bool {
        self.layers.end == self.hp.n_layer
    }
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }
    /// Tokens already in `request`'s KV cache (0 if unknown).
    pub fn n_past(&self, request: u64) -> u32 {
        self.caches.get(&request).map_or(0, |c| c.n_past)
    }

    /// Free `request`'s KV cache. Unknown ids are ignored.
    pub fn release(&mut self, request: u64) {
        self.caches.remove(&request);
    }

    /// Run this stage's layers on a batch of tokens starting at position `pos`.
    /// `pos` must equal the number of tokens already processed for `request`.
    pub fn forward(&mut self, request: u64, pos: u32, input: LlamaInput<'_>) -> Result<LlamaOutput> {
        self.run(request, pos, input, false).map(|(o, _)| o)
    }

    /// Like `forward`, but also returns the hidden state after every local layer
    /// (each `n_tokens * n_embd`). For tests and debugging.
    pub fn forward_traced(&mut self, request: u64, pos: u32, input: LlamaInput<'_>) -> Result<(LlamaOutput, Vec<Vec<f32>>)> {
        self.run(request, pos, input, true)
    }

    fn kv_for(&mut self, request: u64) -> Result<&mut KvCache> {
        if !self.caches.contains_key(&request) {
            let n = self.blocks.len();
            unsafe {
                let ctx = sys::ggml_init(sys::ggml_init_params { mem_size: sys::ggml_tensor_overhead() * 2 * n, mem_buffer: null_mut(), no_alloc: true });
                let mut k = Vec::with_capacity(n);
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    k.push(sys::ggml_new_tensor_2d(ctx, sys::ggml_type::GGML_TYPE_F32, self.hp.kv_dim() as i64, self.n_ctx as i64));
                    v.push(sys::ggml_new_tensor_2d(ctx, sys::ggml_type::GGML_TYPE_F32, self.hp.kv_dim() as i64, self.n_ctx as i64));
                }
                let buf = sys::ggml_backend_alloc_ctx_tensors(ctx, self.backend);
                if buf.is_null() {
                    sys::ggml_free(ctx);
                    return Err(Error::Ggml("cannot allocate KV cache".into()));
                }
                // Unwritten cells are masked out, but NaN garbage times a zero weight is still NaN.
                sys::ggml_backend_buffer_clear(buf, 0);
                self.caches.insert(request, KvCache { ctx, buf, k, v, n_past: 0 });
            }
        }
        Ok(self.caches.get_mut(&request).unwrap())
    }

    fn run(&mut self, request: u64, pos: u32, input: LlamaInput<'_>, trace: bool) -> Result<(LlamaOutput, Vec<Vec<f32>>)> {
        let hp = self.hp.clone();
        let (e, hd, nh, nkv) = (hp.n_embd as i64, hp.head_dim() as i64, hp.n_head as i64, hp.n_head_kv as i64);
        let n = match &input {
            LlamaInput::Tokens(t) => {
                if !self.is_head() {
                    return Err(Error::Ggml(format!("stage {:?} is not the head; it takes hidden states, not tokens", self.layers)));
                }
                if let Some(&bad) = t.iter().find(|&&x| x >= hp.n_vocab) {
                    return Err(Error::Ggml(format!("token {bad} out of vocab ({})", hp.n_vocab)));
                }
                t.len()
            }
            LlamaInput::Hidden(h) => {
                if self.is_head() {
                    return Err(Error::Ggml("the head stage takes tokens, not hidden states".into()));
                }
                if h.len() % hp.n_embd as usize != 0 {
                    return Err(Error::Ggml(format!("hidden length {} is not a multiple of n_embd {}", h.len(), hp.n_embd)));
                }
                h.len() / hp.n_embd as usize
            }
        } as u32;
        if n == 0 {
            return Err(Error::Ggml("empty batch".into()));
        }
        let n_ctx = self.n_ctx;
        let past = self.n_past(request);
        if pos != past {
            return Err(Error::Ggml(format!("request {request}: got position {pos}, but {past} tokens are cached")));
        }
        if pos + n > n_ctx {
            return Err(Error::Ggml(format!("request {request}: context full ({} > {n_ctx})", pos + n)));
        }
        let n_kv = (pos + n) as i64;
        let (kc, vc) = {
            let kv = self.kv_for(request)?;
            (kv.k.clone(), kv.v.clone())
        };

        let graph_size = 64 * self.blocks.len() + 64;
        unsafe {
            let ctx = sys::ggml_init(sys::ggml_init_params {
                mem_size: sys::ggml_tensor_overhead() * graph_size + sys::ggml_graph_overhead_custom(graph_size, false),
                mem_buffer: null_mut(),
                no_alloc: true,
            });
            struct Ctx(*mut sys::ggml_context);
            impl Drop for Ctx {
                fn drop(&mut self) {
                    unsafe { sys::ggml_free(self.0) }
                }
            }
            let _guard = Ctx(ctx);
            let gf = sys::ggml_new_graph_custom(ctx, graph_size, false);
            let n64 = n as i64;

            let input_t = |ty: sys::ggml_type, ne0: i64, ne1: i64| {
                let t = sys::ggml_new_tensor_2d(ctx, ty, ne0, ne1);
                sys::ggml_set_input(t);
                t
            };
            let (in_x, mut x) = match &input {
                LlamaInput::Tokens(_) => {
                    let ids = input_t(sys::ggml_type::GGML_TYPE_I32, n64, 1);
                    (ids, sys::ggml_get_rows(ctx, self.tok_embd.unwrap(), ids))
                }
                LlamaInput::Hidden(_) => {
                    let h = input_t(sys::ggml_type::GGML_TYPE_F32, e, n64);
                    (h, h)
                }
            };
            let in_pos = input_t(sys::ggml_type::GGML_TYPE_I32, n64, 1);
            let in_rows = input_t(sys::ggml_type::GGML_TYPE_I64, n64, 1);
            let in_mask = input_t(sys::ggml_type::GGML_TYPE_F32, n_kv, n64);
            let kq_scale = 1.0 / (hd as f32).sqrt();
            let rope = |t: T| {
                sys::ggml_rope_ext(ctx, t, in_pos, null_mut(), hp.n_rot as i32, sys::GGML_ROPE_TYPE_NORMAL as i32, hp.n_ctx_train as i32, hp.rope_freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };

            let mut traced = Vec::new();
            for (li, w) in self.blocks.iter().enumerate() {
                // Attention.
                let h = sys::ggml_mul(ctx, sys::ggml_rms_norm(ctx, x, hp.rms_eps), w.attn_norm);
                let q = rope(sys::ggml_reshape_3d(ctx, sys::ggml_mul_mat(ctx, w.wq, h), hd, nh, n64));
                let k = rope(sys::ggml_reshape_3d(ctx, sys::ggml_mul_mat(ctx, w.wk, h), hd, nkv, n64));
                let v = sys::ggml_mul_mat(ctx, w.wv, h);
                // Write this batch's K/V into rows [pos, pos+n) of the cache. set_rows returns a
                // view of the whole cache that depends on the write, so reads are ordered after it.
                let k_all = sys::ggml_set_rows(ctx, kc[li], sys::ggml_reshape_2d(ctx, k, hd * nkv, n64), in_rows);
                let v_all = sys::ggml_set_rows(ctx, vc[li], v, in_rows);
                let row = (hd * nkv) as usize * 4;
                // K: [hd, nkv, n_kv] -> [hd, n_kv, nkv]
                let kk = sys::ggml_permute(ctx, sys::ggml_view_3d(ctx, k_all, hd, nkv, n_kv, hd as usize * 4, row, 0), 0, 2, 1, 3);
                // Q: [hd, nh, n] -> [hd, n, nh]
                let qq = sys::ggml_permute(ctx, q, 0, 2, 1, 3);
                // scores [n_kv, n, nh]; mul_mat broadcasts kv heads over query heads (GQA).
                let kq = sys::ggml_soft_max_ext(ctx, sys::ggml_mul_mat(ctx, kk, qq), in_mask, kq_scale, 0.0);
                // V: [hd, nkv, n_kv] -> [n_kv, hd, nkv]
                let vv = sys::ggml_cont(ctx, sys::ggml_permute(ctx, sys::ggml_view_3d(ctx, v_all, hd, nkv, n_kv, hd as usize * 4, row, 0), 1, 2, 0, 3));
                // [hd, n, nh] -> [hd, nh, n] -> [e, n]
                let att = sys::ggml_mul_mat(ctx, vv, kq);
                let att = sys::ggml_reshape_2d(ctx, sys::ggml_cont(ctx, sys::ggml_permute(ctx, att, 0, 2, 1, 3)), e, n64);
                x = sys::ggml_add(ctx, x, sys::ggml_mul_mat(ctx, w.wo, att));

                // Feed-forward (SwiGLU).
                let h = sys::ggml_mul(ctx, sys::ggml_rms_norm(ctx, x, hp.rms_eps), w.ffn_norm);
                let ff = sys::ggml_swiglu_split(ctx, sys::ggml_mul_mat(ctx, w.w_gate, h), sys::ggml_mul_mat(ctx, w.w_up, h));
                x = sys::ggml_add(ctx, x, sys::ggml_mul_mat(ctx, w.w_down, ff));
                if trace {
                    sys::ggml_set_output(x);
                    sys::ggml_build_forward_expand(gf, x);
                    traced.push(x);
                }
            }
            let out = if self.is_tail() {
                let h = sys::ggml_mul(ctx, sys::ggml_rms_norm(ctx, x, hp.rms_eps), self.out_norm.unwrap());
                sys::ggml_mul_mat(ctx, self.output.unwrap(), h)
            } else {
                x
            };
            sys::ggml_set_output(out);
            sys::ggml_build_forward_expand(gf, out);

            if !sys::ggml_gallocr_alloc_graph(self.galloc, gf) {
                return Err(Error::Ggml("cannot allocate compute graph".into()));
            }

            match &input {
                LlamaInput::Tokens(t) => {
                    let ids: Vec<i32> = t.iter().map(|&x| x as i32).collect();
                    sys::ggml_backend_tensor_set(in_x, ids.as_ptr().cast(), 0, ids.len() * 4);
                }
                LlamaInput::Hidden(h) => sys::ggml_backend_tensor_set(in_x, h.as_ptr().cast(), 0, h.len() * 4),
            }
            let positions: Vec<i32> = (pos..pos + n).map(|p| p as i32).collect();
            sys::ggml_backend_tensor_set(in_pos, positions.as_ptr().cast(), 0, positions.len() * 4);
            let rows: Vec<i64> = (pos..pos + n).map(|p| p as i64).collect();
            sys::ggml_backend_tensor_set(in_rows, rows.as_ptr().cast(), 0, rows.len() * 8);
            // Causal mask: token i (absolute position pos+i) sees cache cells 0..=pos+i.
            let mut mask = vec![0f32; (n_kv * n64) as usize];
            for i in 0..n as usize {
                for j in (pos as usize + i + 1)..n_kv as usize {
                    mask[i * n_kv as usize + j] = f32::NEG_INFINITY;
                }
            }
            sys::ggml_backend_tensor_set(in_mask, mask.as_ptr().cast(), 0, mask.len() * 4);

            let st = sys::ggml_backend_graph_compute(self.backend, gf);
            if !matches!(st, sys::ggml_status::GGML_STATUS_SUCCESS) {
                return Err(Error::Ggml(format!("graph compute failed: {st:?}")));
            }

            let read = |t: T| {
                let mut v = vec![0f32; sys::ggml_nbytes(t) / 4];
                sys::ggml_backend_tensor_get(t, v.as_mut_ptr().cast(), 0, v.len() * 4);
                v
            };
            let result = read(out);
            let traced: Vec<Vec<f32>> = traced.into_iter().map(read).collect();
            self.caches.get_mut(&request).unwrap().n_past = pos + n;
            let out = if self.is_tail() { LlamaOutput::Logits(result) } else { LlamaOutput::Hidden(result) };
            Ok((out, traced))
        }
    }
}
