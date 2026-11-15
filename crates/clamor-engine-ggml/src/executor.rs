//! `GgmlExecutor`: a `LlamaStage` behind the ring transport's `StageExecutor` trait.
//!
//! * Head: token ids in. Every other stage: activations in, decoded from the wire dtype.
//! * Non-tail stages: hidden states out, encoded as f16 (default) or int8.
//! * Tail: samples the next token from the last position's logits and flags end-of-sequence.
//!
//! One KV cache per request id lives inside the `LlamaStage`; `release` frees it.
use crate::gguf::GgufFile;
use crate::llama::{LlamaInput, LlamaOutput, LlamaStage, Logits, StageOptions};
use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use clamor_transport::codec;
use clamor_transport::frame::{Acts, DType};
use clamor_transport::stage::{StageExecutor, StageInput, StageOutput};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// How the tail picks the next token. `temperature == 0` is greedy (argmax).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// Keep only the k most likely tokens before sampling. 0 means no limit.
    pub top_k: u32,
    /// Base seed; each request's generator is seeded from this and its id, so runs repeat.
    pub seed: u64,
}
impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.0, top_k: 0, seed: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutorOptions {
    pub stage: StageOptions,
    /// Encoding for activations this stage sends: `F16` (default) or `I8` (a quarter of f32
    /// traffic, lossy: validate before using).
    pub wire: DType,
    pub sampling: Sampling,
    /// End-of-sequence token ids. `None` reads them from the GGUF tokenizer metadata.
    pub eos: Option<Vec<u32>>,
}
impl Default for ExecutorOptions {
    fn default() -> Self {
        Self { stage: StageOptions::default(), wire: DType::F16, sampling: Sampling::default(), eos: None }
    }
}

pub struct GgmlExecutor {
    stage: LlamaStage,
    wire: DType,
    sampling: Sampling,
    eos: Vec<u32>,
    rng: HashMap<u64, u64>,
    live: Arc<AtomicUsize>,
}

impl GgmlExecutor {
    /// Load layers `layers` of the GGUF model at `path`.
    pub fn load(path: &str, layers: Range<u32>, opts: ExecutorOptions) -> Result<Self> {
        if !matches!(opts.wire, DType::F16 | DType::I8) {
            bail!("wire dtype must be F16 or I8, not {:?}", opts.wire);
        }
        let eos = match opts.eos {
            Some(e) => e,
            None => {
                let g = GgufFile::open(path)?;
                ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"].iter().filter_map(|k| g.get_u32(k)).collect()
            }
        };
        let stage = LlamaStage::load(path, layers, opts.stage)?;
        Ok(Self { stage, wire: opts.wire, sampling: opts.sampling, eos, rng: HashMap::new(), live: Arc::new(AtomicUsize::new(0)) })
    }

    pub fn stage(&self) -> &LlamaStage {
        &self.stage
    }
    pub fn eos(&self) -> &[u32] {
        &self.eos
    }
    /// Live count of requests holding a KV cache here. Stays readable after the executor has
    /// been moved into a running stage (for tests and monitoring).
    pub fn live_requests(&self) -> Arc<AtomicUsize> {
        self.live.clone()
    }

    fn update_live(&self) {
        self.live.store(self.stage.live_requests(), Ordering::Relaxed);
    }

    fn encode(&self, h: &[f32], n_tokens: u32) -> Acts {
        let (dtype, scale, data) = match self.wire {
            DType::I8 => {
                let (q, s) = codec::quantize_i8(h);
                (DType::I8, s, q)
            }
            _ => (DType::F16, 1.0, codec::f32_to_f16(h)),
        };
        Acts { dtype, scale, n_tokens, data: Bytes::from(data) }
    }

    fn decode(&self, a: &Acts) -> Result<Vec<f32>> {
        let h = match a.dtype {
            DType::F16 => codec::f16_to_f32(&a.data),
            DType::I8 => codec::dequantize_i8(&a.data, a.scale),
            DType::None => bail!("activations without a dtype"),
        };
        let want = a.n_tokens as usize * self.stage.hparams().n_embd as usize;
        if h.len() != want {
            bail!("activation frame has {} values, expected {} ({} tokens x {} embd)", h.len(), want, a.n_tokens, self.stage.hparams().n_embd);
        }
        Ok(h)
    }

    fn sample(&mut self, request: u64, logits: &[f32]) -> u32 {
        let argmax = || logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        let s = self.sampling;
        if s.temperature <= 0.0 {
            return argmax();
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        if s.top_k > 0 && (s.top_k as usize) < idx.len() {
            idx.select_nth_unstable_by(s.top_k as usize - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            idx.truncate(s.top_k as usize);
        }
        let max = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f64> = idx.iter().map(|&i| (((logits[i] - max) / s.temperature) as f64).exp()).collect();
        let total: f64 = w.iter().sum();
        let state = self.rng.entry(request).or_insert_with(|| splitmix(s.seed ^ request.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1);
        *state = xorshift(*state);
        let mut r = (*state >> 11) as f64 / (1u64 << 53) as f64 * total;
        for (k, &wk) in w.iter().enumerate() {
            r -= wk;
            if r <= 0.0 {
                return idx[k] as u32;
            }
        }
        idx[idx.len() - 1] as u32
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^ (x << 17)
}

impl StageExecutor for GgmlExecutor {
    fn forward(&mut self, request: u64, pos: u32, input: StageInput<'_>) -> Result<StageOutput> {
        let res = match input {
            StageInput::Tokens(t) => self.stage.forward_logits(request, pos, LlamaInput::Tokens(t), Logits::Last),
            StageInput::Acts(a) => {
                let h = self.decode(&a)?;
                self.stage.forward_logits(request, pos, LlamaInput::Hidden(&h), Logits::Last)
            }
        };
        self.update_live();
        match res.map_err(|e| anyhow!(e))? {
            LlamaOutput::Hidden(h) => {
                let n = (h.len() / self.stage.hparams().n_embd as usize) as u32;
                Ok(StageOutput::Acts(self.encode(&h, n)))
            }
            LlamaOutput::Logits(l) => {
                let id = self.sample(request, &l);
                Ok(StageOutput::Token { id, eos: self.eos.contains(&id) })
            }
        }
    }

    fn release(&mut self, request: u64) {
        self.stage.release(request);
        self.rng.remove(&request);
        self.update_live();
    }
}
