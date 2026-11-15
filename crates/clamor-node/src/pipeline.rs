//! Coordinator side of the big-model lane: turn a layout into `ShardAssign`s, host stage 0
//! locally (so the API can submit prompts to the head), and track every stage until it is
//! ready, fails, or is released.

use crate::coordinator::Shared;
use crate::stage::{self, Assignment};
use anyhow::{anyhow, bail, Result};
use clamor_engine_ggml::{GgufFile, LlamaHparams, Tokenizer};
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
}

/// Planner input for a real GGUF: size from the actual tensor bytes, KV width from the heads.
fn model_spec(name: &str, g: &GgufFile, hp: &LlamaHparams) -> plan::ModelSpec {
    let bytes: u64 = g.tensors().iter().map(|t| t.n_bytes).sum();
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
    let (engine, data_port, coordinator) = {
        let sw = shared.lock().unwrap();
        (sw.engine.clone(), sw.data_port, sw.coordinator.clone())
    };
    // The coordinator embeds tokens, so it needs the model file too; it also tells us the shape.
    let path = stage::resolve_model(&engine.models_dir, &req.model)?;
    let g = GgufFile::open(&path.display().to_string()).map_err(|e| anyhow!("{e}"))?;
    let hp = LlamaHparams::from_gguf(&g).map_err(|e| anyhow!("{e}"))?;
    let n_ctx = req.ctx.unwrap_or(2048).clamp(16, hp.n_ctx_train.max(16));
    let (tokenizer, tokenizer_note) = match Tokenizer::from_gguf(&g) {
        Ok(t) => (Some(Arc::new(t)), None),
        Err(e) => (None, Some(e.to_string())),
    };

    let layout: Vec<(NodeId, u32, u32)> = match &req.stages {
        Some(specs) => specs.iter().map(|s| (NodeId(s.node.clone()), s.layers[0], s.layers[1])).collect(),
        None => {
            let (reports, coord) = {
                let sw = shared.lock().unwrap();
                let r: Vec<NodeReport> = sw
                    .nodes
                    .values()
                    .filter(|e| matches!(e.state, NodeState::Ready | NodeState::Busy))
                    .map(|e| e.report.clone())
                    .collect();
                (r, sw.coordinator.clone())
            };
            let nodes = plan::plan_nodes(&reports, Some(&coord));
            let p = plan::plan_model_with(&model_spec(&req.model, &g, &hp), n_ctx, &nodes, plan::Topology::Ring);
            if !p.feasible {
                bail!("the planner found no layout that fits: {}", p.notes.join("; "));
            }
            p.stages.iter().map(|s| (s.node.clone(), s.layer_start, s.layer_end)).collect()
        }
    };
    let layout = normalize(layout, &coordinator, hp.n_layer)?;
    let n = layout.len() as u32;

    // Resolve how stages reach each other, and claim the epoch, under one lock.
    let (epoch, assigns, senders) = {
        let mut sw = shared.lock().unwrap();
        let mut hosts = Vec::new();
        let mut senders = Vec::new();
        for (i, (node, _, _)) in layout.iter().enumerate() {
            if i == 0 {
                // Stage 0 is reached by the last stage, at the address it reaches the coordinator on.
                let host = match layout.last() {
                    Some((last, _, _)) if n > 1 => sw.nodes.get(last).and_then(|e| e.coord_ip).ok_or_else(|| anyhow!("{last} is not connected"))?,
                    _ => std::net::IpAddr::from([127, 0, 0, 1]),
                };
                hosts.push(host);
                senders.push(None);
                continue;
            }
            let e = sw.nodes.get(node).ok_or_else(|| anyhow!("unknown node {node}"))?;
            let tx = e.tx.clone().ok_or_else(|| anyhow!("{node} is not connected"))?;
            if matches!(e.state, NodeState::Draining | NodeState::Away) || matches!(e.report.policy.participation, Participation::Paused | Participation::Draining) {
                bail!("{node} is not available (state {:?}, policy {:?})", e.state, e.report.policy.participation);
            }
            hosts.push(e.peer_ip.ok_or_else(|| anyhow!("{node} has no address"))?);
            senders.push(Some(tx));
        }
        let addr = |i: usize| format!("{}:{}", hosts[i], data_port + i as u16);
        let epoch = sw.next_epoch;
        sw.next_epoch += 1;
        let session = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        let act_dtype = req.act_dtype.unwrap_or(ActDType::F16);
        let assigns: Vec<Assignment> = layout
            .iter()
            .enumerate()
            .map(|(i, (_, a, b))| Assignment {
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
            })
            .collect();
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
            stages: layout.iter().enumerate().map(|(i, (node, a, b))| PipeStage { node: node.clone(), layer_start: *a, layer_end: *b, addr: addr(i), status: StageStatus::Loading }).collect(),
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
            });
        }
    }

    // Stage 0 runs here.
    let local = assigns[0].clone();
    let shared2 = shared.clone();
    tokio::spawn(async move {
        match stage::start(&engine, &local).await {
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
        let st = |n: &str| PipeStage { node: id(n), layer_start: 0, layer_end: 0, addr: String::new(), status: StageStatus::Loading };
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
