//! Running one pipeline stage on this machine: shared by members (on `ShardAssign`) and by the
//! coordinator, which always hosts stage 0 so it owns the request table behind the API.

use anyhow::{anyhow, bail, Context, Result};
use clamor_engine_ggml::{list_devices, DeviceKind, ExecutorOptions, GgmlExecutor, Sampling, StageOptions};
use clamor_proto::{ActDType, SamplingParams};
use clamor_transport::frame::DType;
use clamor_transport::stage::{run_stage, HeadHandle, StageConfig};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::net::{lookup_host, TcpListener};
use tokio::task::{AbortHandle, JoinHandle};

/// Where models live and which compute device stages run on.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub models_dir: PathBuf,
    /// `cpu`, a ggml device name such as `Vulkan0`, or a device index.
    pub device: String,
}

impl EngineConfig {
    pub fn device_index(&self) -> Result<usize> {
        let devs = list_devices();
        let d = self.device.trim();
        let found = if d.eq_ignore_ascii_case("cpu") {
            devs.iter().find(|x| x.kind == DeviceKind::Cpu)
        } else if let Ok(i) = d.parse::<usize>() {
            devs.iter().find(|x| x.index == i)
        } else {
            devs.iter().find(|x| x.name.eq_ignore_ascii_case(d))
        };
        let names: Vec<String> = devs.iter().map(|x| format!("{} ({})", x.name, x.index)).collect();
        found.map(|x| x.index).ok_or_else(|| anyhow!("no device {d:?}; this build has: {}", names.join(", ")))
    }
}

/// A model is named by its bare file name inside the models directory. Anything that could
/// point elsewhere (separators, `..`, absolute paths) is refused, so a coordinator cannot make a
/// member open arbitrary files.
pub fn resolve_model(dir: &Path, name: &str) -> Result<PathBuf> {
    let ok = !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
        && !name.starts_with('.')
        && name.ends_with(".gguf");
    if !ok {
        bail!("model must be a plain .gguf file name, got {name:?}");
    }
    let p = dir.join(name);
    if !p.is_file() {
        bail!("model {name} not found in {}", dir.display());
    }
    Ok(p)
}

/// Everything a stage needs; built from `ShardAssign` (members) or the pipeline plan (coordinator).
#[derive(Clone, Debug)]
pub struct Assignment {
    pub model: String,
    pub epoch: u32,
    pub session: u64,
    pub stage_index: u32,
    pub stage_count: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub listen_port: u16,
    pub downstream: String,
    pub act_dtype: ActDType,
    pub n_ctx: u32,
    pub embed: bool,
    pub sampling: SamplingParams,
}

pub struct Started {
    /// Present on stage 0.
    pub head: Option<HeadHandle>,
    pub abort: AbortHandle,
    /// Ends when the stage stops; `Err` says why.
    pub task: JoinHandle<Result<()>>,
    pub load_ms: u64,
}

/// Load this stage's layers, bind its data port and join the ring. Returns once loaded; the
/// links to the neighbours come up in the background as they start too.
pub async fn start(cfg: &EngineConfig, a: &Assignment) -> Result<Started> {
    let path = resolve_model(&cfg.models_dir, &a.model)?;
    let device = cfg.device_index()?;
    let opts = ExecutorOptions {
        stage: StageOptions { device, n_ctx: a.n_ctx, n_threads: 0, embed: Some(a.embed) },
        wire: match a.act_dtype {
            ActDType::F16 => DType::F16,
            ActDType::I8 => DType::I8,
        },
        sampling: Sampling { temperature: a.sampling.temperature, top_k: a.sampling.top_k, seed: a.sampling.seed },
        eos: None,
    };
    let (layers, p) = (a.layer_start..a.layer_end, path.display().to_string());
    let t0 = Instant::now();
    let exec = tokio::task::spawn_blocking(move || GgmlExecutor::load(&p, layers, opts))
        .await?
        .with_context(|| format!("loading layers {}..{} of {}", a.layer_start, a.layer_end, a.model))?;
    let load_ms = t0.elapsed().as_millis() as u64;

    let listener = TcpListener::bind(("0.0.0.0", a.listen_port)).await.with_context(|| format!("binding data port {}", a.listen_port))?;
    let downstream = if a.stage_count == 1 {
        None
    } else {
        Some(lookup_host(&a.downstream).await?.next().ok_or_else(|| anyhow!("cannot resolve {}", a.downstream))?)
    };
    let stage_cfg = StageConfig {
        stage_index: a.stage_index,
        stage_count: a.stage_count,
        epoch: a.epoch,
        session: a.session,
        downstream,
        window: 8,
        // Neighbours may still be loading their layers.
        connect_timeout: Duration::from_secs(300),
    };
    let h = run_stage(stage_cfg, listener, exec);
    Ok(Started { head: h.head, abort: h.task.abort_handle(), task: h.task, load_ms })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_cannot_escape_the_models_dir() {
        let dir = std::env::temp_dir().join(format!("clamor-models-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ok.gguf"), b"x").unwrap();
        assert!(resolve_model(&dir, "ok.gguf").is_ok());
        for bad in ["", ".", "..", "../ok.gguf", "/etc/passwd", "sub/ok.gguf", "..\\ok.gguf", ".hidden.gguf", "ok.txt", "a\0b.gguf"] {
            assert!(resolve_model(&dir, bad).is_err(), "{bad:?} should be refused");
        }
        assert!(resolve_model(&dir, "missing.gguf").is_err());
    }
}
