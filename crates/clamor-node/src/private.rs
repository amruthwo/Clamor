//! Private chats, on the asker's device.
//!
//! The device runs both ends of the model (the embedding and the first layers, the last layers
//! and the output) as stages 0 and `n - 1` of a private ring; the swarm's devices run the
//! layers in between with the stages they already have loaded (`RingJoin`). Words go in and
//! come out only here: the other devices, the coordinator included, see numbers in the middle
//! of the model. The conversations are kept on this device too, served by its own dashboard
//! (`dashproxy.rs`), never sent to the coordinator.
//!
//! What it does not hide: that you chatted, when, and roughly how long the messages were. And
//! the middle layers' numbers could, with deliberate effort, be turned back into approximate
//! text; more layers on this device make that harder.
use crate::chats::Chats;
use crate::generate::Generation;
use crate::stage::{self, Assignment, EngineConfig};
use anyhow::{anyhow, Result};
use clamor_engine_ggml::{GgufFile, Tokenizer};
use clamor_proto::*;
use clamor_transport::stage::HeadHandle;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::AbortHandle;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// Not asked for.
    Off,
    /// Asked; the ring is being set up (this device loads its ends).
    Starting,
    Ready,
    Failed(String),
}

pub struct Private {
    pub state: State,
    pub ring: Option<u32>,
    pub head: Option<HeadHandle>,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub n_ctx: u32,
    /// Layers this device runs: (head end, tail start, total).
    pub layers: Option<(u32, u32, u32)>,
    /// This device's two stages (and their loading task).
    pub stages: Vec<AbortHandle>,
    /// Conversations kept on this device (None until the swarm is known).
    pub chats: Option<Chats>,
    pub generating: HashMap<String, Generation>,
    /// To the coordinator, while connected.
    pub control: Option<UnboundedSender<Message>>,
    pub config_dir: std::path::PathBuf,
}

pub type Shared = Arc<Mutex<Private>>;

/// The owner of every conversation kept on this device.
pub const ME: &str = "me";

pub fn new(config_dir: std::path::PathBuf) -> Shared {
    Arc::new(Mutex::new(Private { state: State::Off, ring: None, head: None, tokenizer: None, n_ctx: 0, layers: None, stages: Vec::new(), chats: None, generating: HashMap::new(), control: None, config_dir }))
}

impl Private {
    /// Stop this device's ends of the ring.
    pub fn stop(&mut self) {
        for h in self.stages.drain(..) {
            h.abort();
        }
        self.head = None;
        self.ring = None;
        self.layers = None;
    }

    /// Joined (or rejoined) swarm `swarm_id`: its private conversations.
    pub fn swarm(&mut self, swarm_id: &str) {
        self.chats = Some(Chats::load(&self.config_dir.join("private"), swarm_id));
    }
}

/// The two ends of a private ring, as the coordinator described them.
pub struct Ends {
    pub ring: u32,
    pub model: String,
    pub model_id: Option<String>,
    pub source: Option<String>,
    pub epoch: u32,
    pub session: u64,
    pub n_ctx: u32,
    pub act_dtype: ActDType,
    pub sampling: SamplingParams,
    pub stage_count: u32,
    pub head_layers: [u32; 2],
    pub tail_layers: [u32; 2],
    pub head_port: u16,
    pub tail_port: u16,
    pub head_downstream: String,
    pub head_downstream_tls: String,
    pub tail_upstream_tls: String,
}

/// Load and start this device's two stages; when they are up, the private ring is ready (its
/// links finish connecting in the background as the middle joins).
pub fn start_ends(engine: EngineConfig, policy: Policy, e: Ends, private: Shared) {
    let p2 = private.clone();
    let task = tokio::spawn(async move {
        let r = load_ends(&engine, policy, &e).await;
        let mut p = p2.lock().unwrap();
        if p.ring != Some(e.ring) {
            return; // superseded
        }
        match r {
            Ok((head, tok, stages)) => {
                p.head = Some(head);
                p.tokenizer = Some(tok);
                p.n_ctx = e.n_ctx;
                p.layers = Some((e.head_layers[1], e.tail_layers[0], e.tail_layers[1]));
                p.stages.extend(stages);
                p.state = State::Ready;
                println!("private chats ready: this device runs layers 0..{} and {}..{}", e.head_layers[1], e.tail_layers[0], e.tail_layers[1]);
            }
            Err(err) => {
                eprintln!("private chats: {err:#}");
                p.state = State::Failed(format!("{err:#}"));
            }
        }
    });
    let mut p = private.lock().unwrap();
    p.stages.push(task.abort_handle());
}

async fn load_ends(engine: &EngineConfig, policy: Policy, e: &Ends) -> Result<(HeadHandle, Arc<Tokenizer>, Vec<AbortHandle>)> {
    let me = engine.tls.as_ref().ok_or_else(|| anyhow!("no TLS certificate"))?.fingerprint_hex();
    let base = |stage_index: u32, layers: [u32; 2], listen_port: u16, downstream: String, embed: bool, up: String, down: String| Assignment {
        model: e.model.clone(),
        epoch: e.epoch,
        session: e.session,
        stage_index,
        stage_count: e.stage_count,
        layer_start: layers[0],
        layer_end: layers[1],
        listen_port,
        downstream,
        act_dtype: e.act_dtype,
        n_ctx: e.n_ctx,
        embed,
        sampling: e.sampling,
        source: e.source.clone(),
        model_id: e.model_id.clone(),
        max_requests: Some(4),
        need_bytes: None,
        upstream_tls: Some(up),
        downstream_tls: Some(down),
        policy,
    };
    // Head: its upstream is this device's own tail. Tail: its downstream is this device's head.
    let head = base(0, e.head_layers, e.head_port, e.head_downstream.clone(), true, me.clone(), e.head_downstream_tls.clone());
    let tail = base(e.stage_count - 1, e.tail_layers, e.tail_port, format!("127.0.0.1:{}", e.head_port), false, e.tail_upstream_tls.clone(), me);
    let (h, t) = tokio::try_join!(stage::start(engine, &head, |_, _| {}), stage::start(engine, &tail, |_, _| {}))?;
    let head_handle = h.head.ok_or_else(|| anyhow!("the head stage has no head"))?;
    // The tokenizer comes with the head's slice (or this device's own copy of the model).
    let path = match (&e.source, &e.model_id) {
        (Some(_), Some(id)) => crate::share::cached_slice_path(&engine.cache_dir, id, &e.model, e.head_layers[0]..e.head_layers[1], true),
        _ => stage::resolve_model(&engine.models_dir, &e.model)?,
    };
    let tok = tokio::task::spawn_blocking(move || -> Result<Tokenizer> {
        let g = GgufFile::open(&path.display().to_string()).map_err(|e| anyhow!("{e}"))?;
        Tokenizer::from_gguf(&g).map_err(|e| anyhow!("this model's tokenizer is not supported yet: {e}"))
    })
    .await??;
    Ok((head_handle, Arc::new(tok), vec![h.abort, t.abort]))
}
