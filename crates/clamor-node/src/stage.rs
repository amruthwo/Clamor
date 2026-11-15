//! Running one pipeline stage on this machine: shared by members (on `ShardAssign`) and by the
//! coordinator, which always hosts stage 0 so it owns the request table behind the API.

use anyhow::{anyhow, bail, Context, Result};
use clamor_engine_ggml::{list_devices, DeviceKind, ExecutorOptions, GgmlExecutor, Sampling, StageOptions};
use clamor_proto::{ActDType, SamplingParams};
use clamor_transport::frame::DType;
use clamor_transport::frame::Acts;
use clamor_transport::stage::{run_stage, HeadHandle, StageConfig, StageExecutor, StageInput, StageOutput};
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
    /// CPU threads per stage; 0 picks the performance cores.
    pub threads: usize,
    /// Where layer slices fetched from other nodes are kept.
    pub cache_dir: PathBuf,
    /// Signs requests for slices from other devices (this device's key in the swarm; set once
    /// it has joined).
    pub share_key: Option<ed25519_dalek::SigningKey>,
}

impl EngineConfig {
    /// Check `--device` at startup. `cpu` needs no check, and skipping it matters: enumerating
    /// devices initialises every GPU backend, and Metal compiles its shaders on first use
    /// (21 s on a CI Mac), which would delay the coordinator's portal for nothing.
    pub fn validate(&self) -> Result<()> {
        if self.device.trim().eq_ignore_ascii_case("cpu") {
            return Ok(());
        }
        self.device_index().map(|_| ())
    }

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
/// Model names are bare `.gguf` file names, never paths.
pub fn validate_model_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
        && !name.starts_with('.')
        && name.ends_with(".gguf");
    if !ok {
        bail!("model must be a plain .gguf file name, got {name:?}");
    }
    Ok(())
}

pub fn resolve_model(dir: &Path, name: &str) -> Result<PathBuf> {
    validate_model_name(name)?;
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
    /// Fetch this stage's slice from here (`http://host:port`) instead of a local file.
    pub source: Option<String>,
    pub model_id: Option<String>,
    /// Most requests in flight at once (sized by the coordinator to fit memory).
    pub max_requests: Option<u32>,
    /// Expected memory use; refused if it does not fit this device's budget.
    pub need_bytes: Option<u64>,
    /// This device's policy (RAM and CPU shares). Not sent: each node knows its own.
    pub policy: clamor_proto::Policy,
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
pub async fn start(cfg: &EngineConfig, a: &Assignment, progress: impl FnMut(u64, u64) + Send + 'static) -> Result<Started> {
    // Check memory before downloading or loading anything: swapping would freeze the machine.
    if let Some(need) = a.need_bytes {
        let load = tokio::task::spawn_blocking(|| clamor_muster::probe::LoadSampler::new().sample_load()).await?;
        let budget = crate::neighbour::ram_budget(&a.policy, load.ram_available, load.ram_total);
        if need > budget {
            bail!(
                "not enough memory: this stage needs {:.1} GB, and this device may use {:.1} GB right now ({}% of its {:.1} GB, keeping {} MB free)",
                need as f64 / 1e9,
                budget as f64 / 1e9,
                (a.policy.max_ram_fraction * 100.0).round(),
                load.ram_total as f64 / 1e9,
                crate::neighbour::RAM_MARGIN >> 20
            );
        }
    }
    let path = match (&a.source, &a.model_id) {
        // No local copy: fetch just this stage's slice (or reuse it from the cache).
        (Some(src), Some(id)) => {
            validate_model_name(&a.model)?;
            let dest = crate::share::cached_slice_path(&cfg.cache_dir, id, &a.model, a.layer_start..a.layer_end, a.embed);
            let (src, id, name, layers, embed, key, d) = (src.clone(), id.clone(), a.model.clone(), a.layer_start..a.layer_end, a.embed, cfg.share_key.clone(), dest.clone());
            tokio::task::spawn_blocking(move || crate::share::fetch_slice(&src, &name, &id, layers, embed, key.as_ref(), &d, progress))
                .await?
                .with_context(|| format!("fetching layers {}..{} of {} from {}", a.layer_start, a.layer_end, a.model, a.source.as_deref().unwrap_or("")))?;
            dest
        }
        _ => resolve_model(&cfg.models_dir, &a.model)?,
    };
    let mut opts = ExecutorOptions {
        stage: StageOptions {
            device: 0,
            n_ctx: a.n_ctx,
            n_threads: crate::neighbour::threads_for(&a.policy, cfg.threads, clamor_engine_ggml::default_threads()),
            embed: Some(a.embed),
            ..Default::default()
        },
        wire: match a.act_dtype {
            ActDType::F16 => DType::F16,
            ActDType::I8 => DType::I8,
        },
        sampling: Sampling { temperature: a.sampling.temperature, top_k: a.sampling.top_k, seed: a.sampling.seed },
        eos: None,
    };
    let (layers, p) = (a.layer_start..a.layer_end, path.display().to_string());
    let t0 = Instant::now();
    let cfg2 = cfg.clone();
    // Looking up the device starts every compiled-in GPU backend, which can be slow (Metal
    // compiles its shaders on first use). Keep it on the blocking pool, next to the load.
    let low = a.policy.low_priority;
    opts.stage.low_priority = low;
    let exec = ComputeThread::spawn(low, move || -> Result<GgmlExecutor> {
        opts.stage.device = cfg2.device_index()?;
        Ok(GgmlExecutor::load(&p, layers, opts)?)
    })
    .await
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
        window: a.max_requests.unwrap_or(8).clamp(1, 64),
        // Neighbours may still be loading their layers.
        connect_timeout: Duration::from_secs(300),
        stall_timeout: clamor_transport::stage::DEFAULT_STALL_TIMEOUT,
    };
    let h = run_stage(stage_cfg, listener, exec);
    Ok(Started { head: h.head, abort: h.task.abort_handle(), task: h.task, load_ms })
}

/// A stage's engine on its own thread, named `clamor-compute`. Loading and every forward pass
/// run there, so only this thread (and ggml's workers, which it creates) runs at low priority;
/// the async runtime that sends heartbeats and serves the API stays at normal priority. A new
/// thread per stage also means a priority change applies from the next pipeline, on every OS.
/// Dropping it (the stage stopped) ends the thread, which frees the model.
pub struct ComputeThread {
    jobs: std::sync::mpsc::Sender<Job>,
}

enum Job {
    Forward { request: u64, pos: u32, input: OwnedInput, reply: std::sync::mpsc::SyncSender<Result<StageOutput>> },
    Release(u64),
}

enum OwnedInput {
    Tokens(Vec<u32>),
    Acts(Acts),
}

impl ComputeThread {
    pub async fn spawn<E: StageExecutor>(low_priority: bool, load: impl FnOnce() -> Result<E> + Send + 'static) -> Result<Self> {
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        let (loaded_tx, loaded_rx) = tokio::sync::oneshot::channel::<Result<()>>();
        std::thread::Builder::new()
            .name("clamor-compute".into())
            .spawn(move || {
                if low_priority {
                    if let Err(e) = crate::neighbour::lower_this_thread() {
                        eprintln!("{e} (computing at normal priority)");
                    }
                }
                let mut exec = match load() {
                    Ok(e) => {
                        let _ = loaded_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = loaded_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Forward { request, pos, input, reply } => {
                            let input = match &input {
                                OwnedInput::Tokens(t) => StageInput::Tokens(t),
                                OwnedInput::Acts(a) => StageInput::Acts(a.clone()),
                            };
                            let _ = reply.send(exec.forward(request, pos, input));
                        }
                        Job::Release(r) => exec.release(r),
                    }
                }
            })?;
        loaded_rx.await.map_err(|_| anyhow!("the compute thread stopped while loading"))??;
        Ok(Self { jobs })
    }
}

impl StageExecutor for ComputeThread {
    fn forward(&mut self, request: u64, pos: u32, input: StageInput<'_>) -> Result<StageOutput> {
        let input = match input {
            StageInput::Tokens(t) => OwnedInput::Tokens(t.to_vec()),
            StageInput::Acts(a) => OwnedInput::Acts(a),
        };
        let (reply, answer) = std::sync::mpsc::sync_channel(1);
        self.jobs.send(Job::Forward { request, pos, input, reply }).map_err(|_| anyhow!("the compute thread has stopped"))?;
        answer.recv().map_err(|_| anyhow!("the compute thread has stopped"))?
    }

    fn release(&mut self, request: u64) {
        let _ = self.jobs.send(Job::Release(request));
    }
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
