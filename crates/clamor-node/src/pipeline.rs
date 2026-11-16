//! Coordinator side of the big-model lane: turn a layout into `ShardAssign`s, host stage 0
//! locally (so the API can submit prompts to the head), and track every stage until it is
//! ready, fails, or is released.

use crate::coordinator::Shared;
use crate::stage::{self, Assignment};
use anyhow::{anyhow, bail, Result};
use anyhow::Context;
use clamor_engine_ggml::{GgufFile, LlamaHparams, Tokenizer};
use std::collections::HashMap;
use std::sync::Arc;
use clamor_muster::plan;
use clamor_proto::*;
use clamor_transport::stage::HeadHandle;
use serde::{Deserialize, Serialize};
use tokio::task::AbortHandle;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum StageStatus {
    Loading,
    /// Downloading its layer slice from a node that has the model.
    Fetching { done_bytes: u64, total_bytes: u64 },
    Ready { load_ms: u64 },
    Failed(String),
}

#[derive(Clone, Debug, Serialize)]
pub struct PipeStage {
    pub node: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
    /// `host:port` other stages use to reach this one.
    pub addr: String,
    pub status: StageStatus,
    /// Memory planned for this stage (0 when limits were ignored).
    pub need_bytes: u64,
}

#[derive(Serialize)]
pub struct Pipeline {
    pub model: String,
    pub epoch: u32,
    pub n_ctx: u32,
    pub n_layer: u32,
    pub act_dtype: ActDType,
    pub stages: Vec<PipeStage>,
    /// What to blame: the first *cause* (a device left, a stage failed to load), else the first
    /// knock-on failure (a running stage stopped, usually because a neighbour's link closed).
    /// Time order alone is not enough: a vanished device's data links close at the same moment
    /// as its control connection, so its neighbours' "link closed" can arrive first.
    pub first_failure: Option<String>,
    #[serde(skip)]
    pub failure_is_cause: bool,
    /// Text in and out (`/api/chat`). `None` if the model's tokenizer is not supported yet;
    /// token ids still work through `/api/generate`.
    #[serde(skip)]
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Why there is no tokenizer, if there isn't one.
    pub tokenizer_note: Option<String>,
    #[serde(skip)]
    pub head: Option<HeadHandle>,
    #[serde(skip)]
    pub local: Option<AbortHandle>,
}

impl Pipeline {
    pub fn ready(&self) -> bool {
        self.head.is_some() && self.stages.iter().all(|s| matches!(s.status, StageStatus::Ready { .. }))
    }
    pub fn failure(&self) -> Option<String> {
        self.first_failure.clone()
    }
    pub fn mark_progress(&mut self, epoch: u32, stage_index: u32, node: Option<&NodeId>, done_bytes: u64, total_bytes: u64) {
        if epoch != self.epoch {
            return;
        }
        if let Some(s) = self.stages.get_mut(stage_index as usize) {
            if node.is_none_or(|n| *n == s.node) && matches!(s.status, StageStatus::Loading | StageStatus::Fetching { .. }) {
                s.status = StageStatus::Fetching { done_bytes, total_bytes };
            }
        }
    }

    pub fn mark_ready(&mut self, epoch: u32, stage_index: u32, node: Option<&NodeId>, load_ms: u64) {
        self.set(epoch, stage_index, node, StageStatus::Ready { load_ms }, false);
    }

    /// `cause`: the device left or the stage could not load. Otherwise a running stage stopped.
    pub fn mark_failed(&mut self, epoch: u32, stage_index: u32, node: Option<&NodeId>, error: String, cause: bool) {
        self.set(epoch, stage_index, node, StageStatus::Failed(error), cause);
    }

    fn set(&mut self, epoch: u32, stage_index: u32, node: Option<&NodeId>, status: StageStatus, cause: bool) {
        if epoch != self.epoch {
            return;
        }
        let Some(s) = self.stages.get_mut(stage_index as usize) else { return };
        if node.is_some_and(|n| *n != s.node) {
            return;
        }
        if let StageStatus::Failed(e) = &status {
            if self.first_failure.is_none() || (cause && !self.failure_is_cause) {
                self.first_failure = Some(format!("stage {stage_index} ({}): {e}", s.node));
                self.failure_is_cause = cause;
            }
        }
        // A failure is final for this epoch; a later cause still updates what we blame (above).
        if !matches!(s.status, StageStatus::Failed(_)) {
            s.status = status;
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct StageSpec {
    pub node: String,
    /// `[start, end)` layer range.
    pub layers: [u32; 2],
}

#[derive(Clone, Debug, Deserialize)]
pub struct PipelineRequest {
    /// Model file name, present in every participating node's models directory.
    pub model: String,
    pub ctx: Option<u32>,
    pub act_dtype: Option<ActDType>,
    /// Explicit layout. Omitted: the planner chooses from mustered devices.
    pub stages: Option<Vec<StageSpec>>,
    #[serde(default)]
    pub sampling: SamplingParams,
    /// Skip the memory check (for experiments: the device may swap and become unresponsive).
    #[serde(default)]
    pub ignore_limits: bool,
}

/// Fixed working memory per stage, besides weights, KV caches and attention scratch.
const STAGE_OVERHEAD: u64 = 256 << 20;

/// Memory plan for one stage.
#[derive(Clone, Copy, Debug)]
pub struct StageMemory {
    pub weights: u64,
    /// KV cache for one request at full context.
    pub kv_per_request: u64,
    /// Scratch for one prefill chunk plus fixed overhead.
    pub working: u64,
}

impl StageMemory {
    pub fn need(&self, requests: u32) -> u64 {
        self.weights + self.working + self.kv_per_request * requests as u64
    }
}

/// What a stage holding `layers` needs. Uses the engine's own tensor selection, so the estimate
/// covers exactly what the loader will load.
pub fn stage_memory(sizes: &HashMap<String, u64>, hp: &LlamaHparams, layers: std::ops::Range<u32>, embed: bool, n_ctx: u32) -> Result<StageMemory> {
    let names = clamor_engine_ggml::stage_tensors_with(|n| sizes.contains_key(n), hp, layers.clone(), embed).map_err(|e| anyhow!("{e}"))?;
    let weights = names.iter().map(|n| sizes[n]).sum();
    let n_layers = (layers.end - layers.start) as u64;
    // f32 K and V per layer per position.
    let kv_per_request = 2 * n_layers * hp.kv_dim() as u64 * n_ctx as u64 * 4;
    // Attention scores and softmax for one prefill chunk against the full context.
    let chunk = clamor_engine_ggml::DEFAULT_PREFILL_CHUNK.min(n_ctx) as u64;
    let scratch = if n_layers > 0 { 2 * chunk * n_ctx as u64 * hp.n_head as u64 * 4 } else { 0 };
    Ok(StageMemory { weights, kv_per_request, working: STAGE_OVERHEAD + scratch })
}

/// Most requests every stage can hold at once within its node's budget (capped at 8), or an
/// explanation naming the stage that cannot fit even one.
pub fn fit_requests(stages: &[(NodeId, StageMemory, u64)]) -> std::result::Result<u32, String> {
    let mut best = 8u32;
    for (i, (node, m, budget)) in stages.iter().enumerate() {
        let fixed = m.weights + m.working;
        let fits = if m.kv_per_request == 0 { u32::MAX as u64 } else { budget.saturating_sub(fixed) / m.kv_per_request };
        if fixed > *budget || fits == 0 {
            return Err(format!(
                "stage {i} on {node} needs {:.1} GB ({:.1} GB of weights, {:.2} GB of context per request) but may use {:.1} GB: give it fewer layers, use a smaller context (ctx), or raise its RAM share",
                m.need(1) as f64 / 1e9,
                m.weights as f64 / 1e9,
                m.kv_per_request as f64 / 1e9,
                *budget as f64 / 1e9
            ));
        }
        best = best.min(fits.min(8) as u32);
    }
    Ok(best)
}

/// Planner input for a real GGUF: size from the actual tensor bytes, KV width from the heads.
fn model_spec(name: &str, file_size: u64, hp: &LlamaHparams) -> plan::ModelSpec {
    // The file size (from the holder's inventory): the coordinator may only have a
    // metadata-only slice, which has no tensors to add up.
    let bytes = file_size;
    plan::ModelSpec {
        name: name.to_string(),
        params_b: bytes as f64 / 1e9,
        bits_per_weight: 8.0,
        n_layers: hp.n_layer,
        hidden_dim: hp.n_embd,
        // The engine's KV cache is f32 (the planner assumes f16), hence the factor 2.
        kv_ratio: 2.0 * hp.n_head_kv as f64 / hp.n_head as f64,
    }
}

/// The nodes holding `name`. If the coordinator has it, its copy defines the model; otherwise the
/// first holder's does. Nodes with a different file under the same name are not holders.
fn holders(shared: &Shared, name: &str) -> Result<(ModelInfo, Vec<NodeId>)> {
    let sw = shared.lock().unwrap();
    let reachable = |id: &NodeId, e: &crate::coordinator::NodeEntry| *id == sw.coordinator || e.tx.is_some();
    let has = |e: &crate::coordinator::NodeEntry| e.models.iter().find(|m| m.name == name).cloned();
    let chosen = sw.nodes.get(&sw.coordinator).and_then(has).or_else(|| sw.nodes.iter().filter(|(id, e)| reachable(id, e)).find_map(|(_, e)| has(e)));
    let Some(info) = chosen else {
        bail!("no device in the swarm has {name}: put it in one device's --models folder");
    };
    let holders = sw.nodes.iter().filter(|(id, e)| reachable(id, e) && e.models.iter().any(|m| m.name == name && m.id == info.id)).map(|(id, _)| id.clone()).collect();
    Ok((info, holders))
}

/// `http://host:port` at which `fetcher` can reach `holder`'s model sharing.
fn share_url(sw: &crate::coordinator::Swarm, holder: &NodeId, fetcher: &NodeId) -> Option<String> {
    let (ip, port) = if *holder == sw.coordinator {
        // How the fetcher reaches the coordinator (it is on the fetcher's control connection).
        let ip = if *fetcher == sw.coordinator { std::net::IpAddr::from([127, 0, 0, 1]) } else { sw.nodes.get(fetcher)?.coord_ip? };
        (ip, sw.share_port?)
    } else {
        let e = sw.nodes.get(holder)?;
        (e.peer_ip?, e.share_port?)
    };
    Some(format!("http://{ip}:{port}"))
}

/// Validate a layout and make the coordinator stage 0 (embedding-only if the layout leaves it out).
pub fn normalize(layout: Vec<(NodeId, u32, u32)>, coordinator: &NodeId, n_layer: u32) -> Result<Vec<(NodeId, u32, u32)>> {
    let mut l = layout;
    if l.is_empty() {
        bail!("empty layout");
    }
    if l[0].0 != *coordinator {
        l.insert(0, (coordinator.clone(), 0, 0));
    }
    let mut next = 0;
    for (i, (node, a, b)) in l.iter().enumerate() {
        if i > 0 && node == coordinator {
            bail!("the coordinator must be stage 0 (it hosts the request table)");
        }
        if l[..i].iter().any(|(n, _, _)| n == node) {
            bail!("{node} appears twice; each device runs one stage");
        }
        if *a != next {
            bail!("stage {i} ({node}) starts at layer {a}, expected {next}: ranges must be contiguous");
        }
        if b < a || (b == a && i > 0) {
            bail!("stage {i} ({node}) has an empty range {a}..{b}");
        }
        next = *b;
    }
    if next != n_layer {
        bail!("layout covers layers 0..{next}, but the model has {n_layer}");
    }
    Ok(l)
}

/// Build and launch a pipeline. Replaces (and releases) any existing one.
pub async fn start(shared: &Shared, req: PipelineRequest) -> Result<()> {
    stage::validate_model_name(&req.model)?;
    let (engine, data_port, coordinator) = {
        let sw = shared.lock().unwrap();
        (sw.engine.clone(), sw.data_port, sw.coordinator.clone())
    };
    // Which file is "the model", and who has it.
    let (info, holders) = holders(shared, &req.model)?;
    // Shape, tokenizer and chat template: from the coordinator's own copy, or a metadata-only
    // slice (a few MB) from a node that has the model.
    let meta_path = if holders.contains(&coordinator) {
        stage::resolve_model(&engine.models_dir, &req.model)?
    } else {
        let src = {
            let sw = shared.lock().unwrap();
            share_url(&sw, &holders[0], &coordinator).ok_or_else(|| anyhow!("cannot reach {} for {}", holders[0], req.model))?
        };
        let dest = crate::share::cached_slice_path(&engine.cache_dir, &info.id, &req.model, 0..0, false);
        let (name, id, key, d) = (req.model.clone(), info.id.clone(), engine.share_key.clone(), dest.clone());
        tokio::task::spawn_blocking(move || crate::share::fetch_slice(&src, &name, &id, 0..0, false, key.as_ref(), &d, |_, _| {}))
            .await?
            .with_context(|| format!("fetching {}'s metadata from {}", req.model, holders[0]))?;
        crate::cache::used(&engine.cache_dir, &dest, engine.swarm_id.as_deref());
        crate::cache::enforce_cap(&engine.cache_dir, engine.cache_max, &dest);
        dest
    };
    // The header is a C handle (not Send), so it lives only in this block, never across an await.
    let (hp, tokenizer, tokenizer_note) = {
        let g = GgufFile::open(&meta_path.display().to_string()).map_err(|e| anyhow!("{e}"))?;
        let hp = LlamaHparams::from_gguf(&g).map_err(|e| anyhow!("{e}"))?;
        match Tokenizer::from_gguf(&g) {
            Ok(t) => (hp, Some(Arc::new(t)), None),
            Err(e) => (hp, None, Some(e.to_string())),
        }
    };
    let n_ctx = req.ctx.unwrap_or(2048).clamp(16, hp.n_ctx_train.max(16));

    let layout: Vec<(NodeId, u32, u32)> = match &req.stages {
        Some(specs) => specs.iter().map(|s| (NodeId(s.node.clone()), s.layers[0], s.layers[1])).collect(),
        None => {
            let (reports, coord) = {
                let sw = shared.lock().unwrap();
                let r: Vec<NodeReport> = sw
                    .nodes
                    .values()
                    .filter(|e| matches!(e.state, NodeState::Ready | NodeState::Busy) && e.paused.is_none())
                    .map(|e| e.report.clone())
                    .collect();
                (r, sw.coordinator.clone())
            };
            let nodes = plan::plan_nodes(&reports, Some(&coord));
            let p = plan::plan_model_with(&model_spec(&req.model, info.size, &hp), n_ctx, &nodes, plan::Topology::Ring);
            if !p.feasible {
                bail!("the planner found no layout that fits: {}", p.notes.join("; "));
            }
            p.stages.iter().map(|s| (s.node.clone(), s.layer_start, s.layer_end)).collect()
        }
    };
    let layout = normalize(layout, &coordinator, hp.n_layer)?;
    let n = layout.len() as u32;

    // Memory: size every stage, then allow as many simultaneous requests as all of them fit.
    let (max_requests, needs) = if req.ignore_limits {
        (8, vec![None; layout.len()])
    } else {
        let table = if holders.contains(&coordinator) {
            let p = meta_path.clone();
            tokio::task::spawn_blocking(move || crate::share::tensor_table(&p)).await??
        } else {
            let src = {
                let sw = shared.lock().unwrap();
                share_url(&sw, &holders[0], &coordinator).ok_or_else(|| anyhow!("cannot reach {}", holders[0]))?
            };
            let (name, key) = (req.model.clone(), engine.share_key.clone());
            tokio::task::spawn_blocking(move || crate::share::fetch_tensor_table(&src, &name, key.as_ref())).await??
        };
        let sizes: HashMap<String, u64> = table.into_iter().collect();
        let plans: Vec<(NodeId, StageMemory, u64)> = {
            let sw = shared.lock().unwrap();
            let mut v = Vec::new();
            for (i, (node, a, b)) in layout.iter().enumerate() {
                let m = stage_memory(&sizes, &hp, *a..*b, i == 0, n_ctx)?;
                let e = sw.nodes.get(node).ok_or_else(|| anyhow!("unknown node {node}"))?;
                // Memory this node's stage in the pipeline being replaced will give back.
                let freed: u64 = sw.pipeline.as_ref().map(|p| p.stages.iter().filter(|s| &s.node == node).map(|s| s.need_bytes).sum()).unwrap_or(0);
                let budget = crate::neighbour::ram_budget(&e.report.policy, e.report.info.ram_available_bytes + freed, e.report.info.ram_total_bytes);
                v.push((node.clone(), m, budget));
            }
            v
        };
        let r = fit_requests(&plans).map_err(|e| anyhow!(e))?;
        (r, plans.iter().map(|(_, m, _)| Some(m.need(r))).collect())
    };

    // Resolve how stages reach each other, and claim the epoch, under one lock.
    let (epoch, assigns, senders) = {
        let mut sw = shared.lock().unwrap();
        let mut hosts = Vec::new();
        let mut fps = Vec::new();
        let mut senders = Vec::new();
        for (i, (node, _, _)) in layout.iter().enumerate() {
            if i == 0 {
                // Stage 0 is reached by the last stage, at the address it reaches the coordinator on.
                let host = match layout.last() {
                    Some((last, _, _)) if n > 1 => sw.nodes.get(last).and_then(|e| e.coord_ip).ok_or_else(|| anyhow!("{last} is not connected"))?,
                    _ => std::net::IpAddr::from([127, 0, 0, 1]),
                };
                hosts.push(host);
                fps.push(sw.tls.fingerprint_hex());
                senders.push(None);
                continue;
            }
            let e = sw.nodes.get(node).ok_or_else(|| anyhow!("unknown node {node}"))?;
            let tx = e.tx.clone().ok_or_else(|| anyhow!("{node} is not connected"))?;
            if matches!(e.state, NodeState::Draining | NodeState::Away) || matches!(e.report.policy.participation, Participation::Paused | Participation::Draining) {
                bail!("{node} is not available (state {:?}, policy {:?})", e.state, e.report.policy.participation);
            }
            if let Some(why) = &e.paused {
                bail!("{node} is not taking work right now: {why}");
            }
            hosts.push(e.peer_ip.ok_or_else(|| anyhow!("{node} has no address"))?);
            fps.push(e.tls_fp.clone().ok_or_else(|| anyhow!("{node} runs a Clamor without encryption: update it"))?);
            senders.push(Some(tx));
        }
        let addr = |i: usize| format!("{}:{}", hosts[i], data_port + i as u16);
        let fp_of = |i: usize| fps[i].clone();
        let epoch = sw.next_epoch;
        sw.next_epoch += 1;
        let session = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        let act_dtype = req.act_dtype.unwrap_or(ActDType::F16);
        let assigns: Vec<Assignment> = layout
            .iter()
            .enumerate()
            .map(|(i, (node, a, b))| Assignment {
                model: req.model.clone(),
                epoch,
                session,
                stage_index: i as u32,
                stage_count: n,
                layer_start: *a,
                layer_end: *b,
                listen_port: data_port + i as u16,
                downstream: addr((i + 1) % n as usize),
                act_dtype,
                n_ctx,
                embed: i == 0,
                sampling: req.sampling,
                // A node without the model fetches its slice, preferably from the coordinator.
                source: if holders.contains(node) {
                    None
                } else {
                    let from = if holders.contains(&coordinator) { &coordinator } else { &holders[0] };
                    share_url(&sw, from, node)
                },
                model_id: Some(info.id.clone()),
                max_requests: Some(max_requests),
                need_bytes: needs[i],
                upstream_tls: Some(fp_of((i + n as usize - 1) % n as usize)),
                downstream_tls: Some(fp_of((i + 1) % n as usize)),
                policy: sw.nodes.get(node).map(|e| e.report.policy).unwrap_or_else(|| Policy::default_for(DeviceClass::Desktop)),
            })
            .collect();
        if let Some((i, (node, _, _))) = layout.iter().enumerate().find(|(i, (node, _, _))| !holders.contains(node) && assigns[*i].source.is_none()) {
            bail!("stage {i} ({node}) has no copy of {} and no holder it can reach", req.model);
        }
        let old = sw.pipeline.take();
        if let Some(old) = old {
            release(&mut sw, old);
        }
        sw.pipeline = Some(Pipeline {
            model: req.model.clone(),
            epoch,
            n_ctx,
            n_layer: hp.n_layer,
            act_dtype,
            first_failure: None,
            failure_is_cause: false,
            tokenizer,
            tokenizer_note,
            stages: layout.iter().enumerate().map(|(i, (node, a, b))| PipeStage { node: node.clone(), layer_start: *a, layer_end: *b, addr: addr(i), status: StageStatus::Loading, need_bytes: needs[i].unwrap_or(0) }).collect(),
            head: None,
            local: None,
        });
        (epoch, assigns, senders)
    };

    for (a, tx) in assigns.iter().zip(&senders) {
        if let Some(tx) = tx {
            let _ = tx.send(Message::ShardAssign {
                model: a.model.clone(),
                epoch: a.epoch,
                session: a.session,
                stage_index: a.stage_index,
                stage_count: a.stage_count,
                layer_start: a.layer_start,
                layer_end: a.layer_end,
                listen_port: a.listen_port,
                downstream: a.downstream.clone(),
                act_dtype: a.act_dtype,
                n_ctx: a.n_ctx,
                embed: a.embed,
                sampling: a.sampling,
                source: a.source.clone(),
                model_id: a.model_id.clone(),
                max_requests: a.max_requests,
                need_bytes: a.need_bytes,
                upstream_tls: a.upstream_tls.clone(),
                downstream_tls: a.downstream_tls.clone(),
            });
        }
    }

    // Stage 0 runs here.
    let local = assigns[0].clone();
    let shared2 = shared.clone();
    tokio::spawn(async move {
        let progress = {
            let shared = shared2.clone();
            move |done, total| {
                if let Some(p) = shared.lock().unwrap().pipeline.as_mut() {
                    p.mark_progress(epoch, 0, None, done, total);
                }
            }
        };
        match stage::start(&engine, &local, progress).await {
            Ok(s) => {
                {
                    let mut sw = shared2.lock().unwrap();
                    match sw.pipeline.as_mut() {
                        Some(p) if p.epoch == epoch => {
                            p.head = s.head;
                            p.local = Some(s.abort.clone());
                            p.mark_ready(epoch, 0, None, s.load_ms);
                        }
                        _ => {
                            s.abort.abort(); // superseded while loading
                            return;
                        }
                    }
                }
                let why = match s.task.await {
                    Ok(Ok(())) => "stopped".to_string(),
                    Ok(Err(e)) => format!("{e:#}"),
                    Err(e) if e.is_cancelled() => return, // released
                    Err(e) => e.to_string(),
                };
                if let Some(p) = shared2.lock().unwrap().pipeline.as_mut() {
                    p.mark_failed(epoch, 0, None, why, false);
                }
            }
            Err(e) => {
                if let Some(p) = shared2.lock().unwrap().pipeline.as_mut() {
                    p.mark_failed(epoch, 0, None, format!("{e:#}"), true);
                }
            }
        }
    });
    Ok(())
}

/// Stop a pipeline: tell members to drop their stages and stop the local head.
pub fn release(sw: &mut crate::coordinator::Swarm, p: Pipeline) {
    if let Some(a) = &p.local {
        a.abort();
    }
    for s in p.stages.iter().skip(1) {
        if let Some(tx) = sw.nodes.get(&s.node).and_then(|e| e.tx.as_ref()) {
            let _ = tx.send(Message::ShardRelease { model: p.model.clone(), epoch: p.epoch });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> NodeId {
        NodeId(s.into())
    }

    fn hp() -> LlamaHparams {
        LlamaHparams { n_layer: 4, n_embd: 64, n_head: 4, n_head_kv: 2, n_ff: 128, n_vocab: 128, n_ctx_train: 64, n_rot: 16, rms_eps: 1e-5, rope_freq_base: 1e4, arch: "llama".into(), rope_neox: false }
    }

    fn sizes() -> HashMap<String, u64> {
        let mut s = HashMap::new();
        s.insert("token_embd.weight".into(), 1000);
        s.insert("output_norm.weight".into(), 10);
        s.insert("output.weight".into(), 900);
        for l in 0..4 {
            for t in ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"] {
                s.insert(format!("blk.{l}.{t}.weight"), 100);
            }
        }
        s
    }

    #[test]
    fn stage_memory_counts_exactly_the_stage_tensors() {
        let h = hp();
        let head = stage_memory(&sizes(), &h, 0..0, true, 64).unwrap();
        assert_eq!((head.weights, head.kv_per_request), (1000, 0), "embedding only, no KV");
        let mid = stage_memory(&sizes(), &h, 0..2, false, 64).unwrap();
        assert_eq!(mid.weights, 2 * 9 * 100);
        assert_eq!(mid.kv_per_request, 2 * 2 * 32 * 64 * 4, "K+V, 2 layers, kv_dim 32, 64 positions, f32");
        let tail = stage_memory(&sizes(), &h, 2..4, false, 64).unwrap();
        assert_eq!(tail.weights, 2 * 9 * 100 + 10 + 900, "plus output norm and lm_head");
        assert!(mid.need(2) > mid.need(1));
    }

    #[test]
    fn requests_are_capped_to_what_every_stage_fits() {
        let m = StageMemory { weights: 1000, kv_per_request: 100, working: 0 };
        let st = |b| (id("a"), m, b);
        assert_eq!(fit_requests(&[st(1350)]), Ok(3), "room for 3 KV caches");
        assert_eq!(fit_requests(&[st(1_000_000)]), Ok(8), "never more than 8");
        assert_eq!(fit_requests(&[st(1_000_000), st(1250)]), Ok(2), "the tightest stage decides");
        let e = fit_requests(&[st(1050)]).unwrap_err();
        assert!(e.contains("stage 0 on a") && e.contains("may use"), "{e}");
        assert!(fit_requests(&[st(900)]).is_err(), "weights alone do not fit");
    }

    #[test]
    fn coordinator_becomes_an_embedding_only_head_when_left_out() {
        let l = normalize(vec![(id("a"), 0, 2), (id("b"), 2, 4)], &id("c"), 4).unwrap();
        assert_eq!(l, vec![(id("c"), 0, 0), (id("a"), 0, 2), (id("b"), 2, 4)]);
        let l = normalize(vec![(id("c"), 0, 1), (id("a"), 1, 4)], &id("c"), 4).unwrap();
        assert_eq!(l, vec![(id("c"), 0, 1), (id("a"), 1, 4)]);
        assert_eq!(normalize(vec![(id("c"), 0, 4)], &id("c"), 4).unwrap().len(), 1);
    }

    /// Regression: the status blamed whichever stage failed first. When member b left, its
    /// neighbours' links closed at the same moment, and their "link closed" (a knock-on effect)
    /// could arrive before the coordinator saw b's control connection drop.
    #[test]
    fn failure_blames_the_cause_not_the_knock_on_effects() {
        let st = |n: &str| PipeStage { node: id(n), layer_start: 0, layer_end: 0, addr: String::new(), status: StageStatus::Loading, need_bytes: 0 };
        let mut p = Pipeline { model: "m".into(), epoch: 2, n_ctx: 8, n_layer: 4, act_dtype: ActDType::F16, stages: vec![st("c"), st("a"), st("b")], first_failure: None, failure_is_cause: false, tokenizer: None, tokenizer_note: None, head: None, local: None };
        p.mark_failed(2, 0, None, "ring broken: tail link closed".into(), false);
        p.mark_failed(2, 1, Some(&id("a")), "upstream link closed".into(), false);
        assert_eq!(p.failure().as_deref(), Some("stage 0 (c): ring broken: tail link closed"), "only knock-on effects so far: the first");
        p.mark_failed(2, 2, Some(&id("b")), "connection lost".into(), true);
        assert_eq!(p.failure().as_deref(), Some("stage 2 (b): connection lost"), "the cause wins");
        p.mark_failed(2, 1, Some(&id("a")), "later cause".into(), true);
        assert_eq!(p.failure().as_deref(), Some("stage 2 (b): connection lost"), "the first cause stays");
        // Stale epochs and the wrong node are ignored; failures are final.
        p.mark_failed(1, 0, None, "old".into(), true);
        p.mark_failed(2, 0, Some(&id("x")), "wrong node".into(), true);
        p.mark_ready(2, 2, Some(&id("b")), 1);
        assert!(matches!(p.stages[2].status, StageStatus::Failed(_)));
        assert_eq!(p.failure().as_deref(), Some("stage 2 (b): connection lost"));
        assert!(!p.ready());
    }

    #[test]
    fn bad_layouts_are_refused() {
        let c = id("c");
        for (l, why) in [
            (vec![(id("a"), 0, 2), (id("b"), 3, 4)], "gap"),
            (vec![(id("a"), 0, 3), (id("b"), 2, 4)], "overlap"),
            (vec![(id("a"), 0, 2)], "short"),
            (vec![(id("a"), 0, 2), (id("a"), 2, 4)], "duplicate"),
            (vec![(id("a"), 0, 2), (id("c"), 2, 4)], "coordinator not first"),
            (vec![(id("a"), 0, 2), (id("b"), 2, 2), (id("d"), 2, 4)], "empty middle"),
            (vec![], "empty"),
        ] {
            assert!(normalize(l, &c, 4).is_err(), "{why}");
        }
    }
}


