//! Group planning: given what the muster measured, which models can this swarm run, how
//! should the layers be split, and how fast should it be?
//!
//! The model is deliberately simple and conservative:
//!  * Decode (token generation) is memory-bandwidth bound: each stage must read its weights
//!    once per token, so stage time = stage_bytes / (mem_bw * MEM_EFFICIENCY).
//!  * Stages run in sequence for a single request, so per-token time is the SUM of stage times
//!    plus one network round trip per remote stage (hub-and-spoke through the coordinator,
//!    which is how llama.cpp's RPC backend moves activations).
//!  * Prefill is compute-bound and uses measured CPU GFLOP/s.
//! The efficiency constants are assumptions to be calibrated against real `llama-bench` runs.

use clamor_proto::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MEM_EFFICIENCY: f64 = 0.7;
pub const COMPUTE_EFFICIENCY: f64 = 0.5;
const RUNTIME_OVERHEAD: f64 = 1.03;
const DEFAULT_RTT_MS: f64 = 5.0;
const DEFAULT_LINK_MBPS: f64 = 100.0;
/// Stand-in for "no network" on the coordinator itself.
const LOCAL_LINK_MBPS: f64 = 1e9;
const MAX_SUBSET_NODES: usize = 12;
/// A sampled token returning from the tail to the head.
const TOKEN_BYTES: f64 = 4.0;

/// How stages exchange data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    /// Stage i sends straight to stage i+1; the tail returns the token to the head.
    /// One one-way hop per stage per token. This is what `clamor-transport` implements.
    #[default]
    Ring,
    /// Every stage talks to the coordinator (llama.cpp RPC style): a full round trip per
    /// remote stage per token.
    Hub,
}

impl Topology {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ring" => Some(Topology::Ring),
            "hub" => Some(Topology::Hub),
            _ => None,
        }
    }
}

/// Seconds of network time charged to stage `i` per decoded token.
///
/// Ring hops use `max(rtt_a, rtt_b) / 2` and the slower of the two links, because the muster
/// only measures each device against the coordinator, not every pair.
fn hop_seconds(topo: Topology, stages: &[(&PlanNode, u32)], i: usize, act_bytes: f64) -> f64 {
    let n = stages.len();
    let a = stages[i].0;
    match topo {
        Topology::Hub => {
            if a.is_coordinator {
                0.0
            } else {
                a.rtt_ms / 1000.0 + act_bytes * 8.0 / (a.link_mbps * 1e6)
            }
        }
        Topology::Ring => {
            if n == 1 {
                return 0.0;
            }
            let b = stages[(i + 1) % n].0;
            let bytes = if i + 1 == n { TOKEN_BYTES } else { act_bytes };
            a.rtt_ms.max(b.rtt_ms) / 2000.0 + bytes * 8.0 / (a.link_mbps.min(b.link_mbps) * 1e6)
        }
    }
}

// ---------------------------------------------------------------- models

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quant {
    Q4,
    Q6,
    Q8,
    F16,
}

impl Quant {
    pub fn bits(self) -> f64 {
        match self {
            Quant::Q4 => 4.8,
            Quant::Q6 => 6.6,
            Quant::Q8 => 8.5,
            Quant::F16 => 16.0,
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "q4" => Some(Quant::Q4),
            "q6" => Some(Quant::Q6),
            "q8" => Some(Quant::Q8),
            "f16" => Some(Quant::F16),
            _ => None,
        }
    }
}

/// Architecture-level description of a dense transformer. Enough to size memory and traffic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub params_b: f64,
    pub bits_per_weight: f64,
    pub n_layers: u32,
    pub hidden_dim: u32,
    /// KV-cache width relative to `hidden_dim` (grouped-query attention makes this small).
    pub kv_ratio: f64,
}

impl ModelSpec {
    pub fn weight_bytes(&self) -> f64 {
        self.params_b * 1e9 * self.bits_per_weight / 8.0
    }
    pub fn kv_bytes(&self, ctx: u32) -> f64 {
        2.0 * self.n_layers as f64 * ctx as f64 * self.hidden_dim as f64 * self.kv_ratio * 2.0
    }
    pub fn per_layer_weight_bytes(&self) -> f64 {
        self.weight_bytes() / self.n_layers as f64
    }
    /// Weights plus KV cache plus runtime overhead, per layer.
    pub fn per_layer_total_bytes(&self, ctx: u32) -> f64 {
        (self.weight_bytes() + self.kv_bytes(ctx)) / self.n_layers as f64 * RUNTIME_OVERHEAD
    }
    pub fn activation_bytes(&self) -> f64 {
        self.hidden_dim as f64 * 2.0
    }
    pub fn total_bytes(&self, ctx: u32) -> f64 {
        self.per_layer_total_bytes(ctx) * self.n_layers as f64
    }
}

/// Generic dense size classes. Dimensions are typical for each class, not tied to one release;
/// pass a custom `ModelSpec` for an exact model.
pub fn catalog(q: Quant) -> Vec<ModelSpec> {
    const CLASSES: [(&str, f64, u32, u32); 7] = [
        ("3B", 3.0, 28, 3072),
        ("8B", 8.0, 32, 4096),
        ("14B", 14.0, 40, 5120),
        ("32B", 32.0, 64, 5120),
        ("70B", 70.0, 80, 8192),
        ("123B", 123.0, 88, 12288),
        ("405B", 405.0, 126, 16384),
    ];
    CLASSES
        .iter()
        .map(|&(name, params_b, n_layers, hidden_dim)| ModelSpec {
            name: format!("{name}-{q:?}").to_lowercase(),
            params_b,
            bits_per_weight: q.bits(),
            n_layers,
            hidden_dim,
            kv_ratio: 0.125,
        })
        .collect()
}

// ---------------------------------------------------------------- nodes

/// A device reduced to what the planner needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanNode {
    pub id: NodeId,
    pub mem_budget_bytes: f64,
    pub mem_bw_gbps: f64,
    pub gflops: f64,
    pub rtt_ms: f64,
    pub link_mbps: f64,
    pub is_coordinator: bool,
    pub link_measured: bool,
    pub uses_accelerator: bool,
}

/// Turn muster reports into planner inputs. Devices that are paused, draining, or have not
/// been benchmarked are left out. Accelerators are used only once their bandwidth is measured.
pub fn plan_nodes(reports: &[NodeReport], coordinator: Option<&NodeId>) -> Vec<PlanNode> {
    reports
        .iter()
        .filter_map(|r| {
            let bench = r.bench.as_ref()?;
            if matches!(r.policy.participation, Participation::Paused | Participation::Draining) {
                return None;
            }
            let accel = r
                .info
                .accelerators
                .iter()
                .max_by_key(|a| a.mem_bytes)
                .filter(|_| bench.accel_mem_bw_gbps.is_some());
            let (base_mem, bw, gflops) = match accel {
                Some(a) => {
                    let mem = if a.unified { r.info.ram_available_bytes.min(a.mem_bytes) } else { a.mem_bytes };
                    (mem, bench.accel_mem_bw_gbps.unwrap_or(bench.mem_bw_gbps), bench.accel_tflops.map(|t| t * 1000.0).unwrap_or(bench.cpu_gflops_mt))
                }
                None => (r.info.ram_available_bytes, bench.mem_bw_gbps, bench.cpu_gflops_mt),
            };
            let is_coord = coordinator.map(|c| c == &r.info.id).unwrap_or(false);
            let (rtt, mbps, measured) = if is_coord { (0.0, LOCAL_LINK_MBPS, true) } else { match &r.link {
                Some(l) => {
                    let m = match (l.up_mbps, l.down_mbps) {
                        (Some(u), Some(d)) => u.min(d),
                        (Some(x), None) | (None, Some(x)) => x,
                        (None, None) => DEFAULT_LINK_MBPS,
                    };
                    (l.rtt_ms_p50, m, true)
                }
                None => (DEFAULT_RTT_MS, DEFAULT_LINK_MBPS, false),
            } };
            Some(PlanNode {
                id: r.info.id.clone(),
                mem_budget_bytes: base_mem as f64 * r.policy.max_ram_fraction as f64,
                mem_bw_gbps: bw,
                gflops,
                rtt_ms: if is_coord { 0.0 } else { rtt },
                link_mbps: mbps,
                is_coordinator: is_coord,
                link_measured: measured || is_coord,
                uses_accelerator: accel.is_some(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------- group summary

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroupSummary {
    pub devices: u32,
    pub benchmarked: u32,
    pub cpu_threads: u32,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub mem_bw_sum_gbps: f64,
    pub gflops_sum: f64,
    pub by_class: BTreeMap<String, u32>,
    /// Devices whose numbers cannot be trusted for planning (and why).
    pub warnings: Vec<String>,
}

/// `coordinator`: the device the others' links are measured against (it has no link of its own).
pub fn summarize(reports: &[NodeReport], coordinator: Option<&NodeId>) -> GroupSummary {
    let mut g = GroupSummary::default();
    for r in reports {
        g.devices += 1;
        g.cpu_threads += r.info.cpu_threads;
        g.ram_total_bytes += r.info.ram_total_bytes;
        g.ram_available_bytes += r.info.ram_available_bytes;
        *g.by_class.entry(format!("{:?}", r.info.class).to_lowercase()).or_default() += 1;
        match &r.bench {
            Some(b) => {
                g.benchmarked += 1;
                g.mem_bw_sum_gbps += b.mem_bw_gbps;
                g.gflops_sum += b.cpu_gflops_mt;
                if b.build == "debug" {
                    g.warnings.push(format!("{}: benchmark came from a debug build; rerun with --release", r.info.id));
                }
            }
            None => g.warnings.push(format!("{}: not benchmarked yet", r.info.id)),
        }
        if r.link.is_none() && r.bench.is_some() && coordinator != Some(&r.info.id) {
            g.warnings.push(format!("{}: no link measurement, assuming {DEFAULT_RTT_MS} ms / {DEFAULT_LINK_MBPS} Mbit/s", r.info.id));
        }
        if !r.info.accelerators.is_empty() && r.bench.as_ref().map(|b| b.accel_mem_bw_gbps.is_none()).unwrap_or(false) {
            g.warnings.push(format!("{}: accelerator not benchmarked yet, planning on CPU/RAM only", r.info.id));
        }
    }
    g
}

// ---------------------------------------------------------------- planning

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage {
    pub node: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub weight_bytes: f64,
    pub est_ms_per_token: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub model: String,
    pub topology: Topology,
    pub params_b: f64,
    pub need_bytes: f64,
    pub feasible: bool,
    pub stages: Vec<Stage>,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub bottleneck: String,
    pub notes: Vec<String>,
}

struct Evaluated<'a> {
    stages: Vec<(&'a PlanNode, u32)>,
    decode_s: f64,
}

fn evaluate<'a>(model: &ModelSpec, ctx: u32, subset: &[&'a PlanNode], topo: Topology) -> Option<Evaluated<'a>> {
    let per_layer = model.per_layer_total_bytes(ctx);
    let caps: Vec<u32> = subset
        .iter()
        .map(|n| ((n.mem_budget_bytes / per_layer).floor() as u32).min(model.n_layers))
        .collect();
    if caps.iter().map(|&c| c as u64).sum::<u64>() < model.n_layers as u64 {
        return None;
    }

    // Single-stream decode time is additive across stages, so filling the fastest memory first
    // is optimal for a given subset.
    let mut order: Vec<usize> = (0..subset.len()).collect();
    order.sort_by(|&a, &b| subset[b].mem_bw_gbps.partial_cmp(&subset[a].mem_bw_gbps).unwrap_or(std::cmp::Ordering::Equal));
    let mut left = model.n_layers;
    let mut stages: Vec<(&PlanNode, u32)> = Vec::new();
    for i in order {
        if left == 0 {
            break;
        }
        let take = caps[i].min(left);
        if take > 0 {
            stages.push((subset[i], take));
            left -= take;
        }
    }
    // Coordinator hosts the first layers, remote stages follow fastest-first.
    stages.sort_by_key(|(n, _)| !n.is_coordinator);

    let act = model.activation_bytes();
    let decode_s: f64 = stages
        .iter()
        .enumerate()
        .map(|(i, (n, layers))| {
            let read = *layers as f64 * model.per_layer_weight_bytes() / (n.mem_bw_gbps * 1e9 * MEM_EFFICIENCY);
            read + hop_seconds(topo, &stages, i, act)
        })
        .sum();
    Some(Evaluated { stages, decode_s })
}

pub fn plan_model(model: &ModelSpec, ctx: u32, nodes: &[PlanNode]) -> Plan {
    plan_model_with(model, ctx, nodes, Topology::default())
}

pub fn plan_model_with(model: &ModelSpec, ctx: u32, nodes: &[PlanNode], topo: Topology) -> Plan {
    let need = model.total_bytes(ctx);
    let mut notes = Vec::new();
    let mut cand: Vec<&PlanNode> = nodes.iter().collect();
    if cand.len() > MAX_SUBSET_NODES {
        cand.sort_by(|a, b| b.mem_bw_gbps.partial_cmp(&a.mem_bw_gbps).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(MAX_SUBSET_NODES);
        notes.push(format!("{} devices: searched the {MAX_SUBSET_NODES} with the fastest memory", nodes.len()));
    }

    let mut best: Option<Evaluated> = None;
    for mask in 1u32..(1u32 << cand.len()) {
        let subset: Vec<&PlanNode> = cand.iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, n)| *n).collect();
        if let Some(ev) = evaluate(model, ctx, &subset, topo) {
            let better = match &best {
                None => true,
                Some(b) => ev.decode_s < b.decode_s - 1e-12 || (ev.decode_s <= b.decode_s + 1e-12 && ev.stages.len() < b.stages.len()),
            };
            if better {
                best = Some(ev);
            }
        }
    }

    let Some(ev) = best else {
        let have: f64 = nodes.iter().map(|n| n.mem_budget_bytes).sum();
        notes.push(format!("needs {:.1} GiB but the group offers {:.1} GiB", need / GIB, have / GIB));
        return Plan {
            model: model.name.clone(),
            topology: topo,
            params_b: model.params_b,
            need_bytes: need,
            feasible: false,
            stages: vec![],
            decode_tok_s: 0.0,
            prefill_tok_s: 0.0,
            bottleneck: "memory".into(),
            notes,
        };
    };

    let act = model.activation_bytes();
    let mut layer = 0u32;
    let mut stages = Vec::new();
    let (mut read_total, mut hop_total) = (0.0, 0.0);
    let mut worst: (f64, String) = (0.0, String::new());
    let mut prefill_s = 0.0;
    for (i, (n, layers)) in ev.stages.iter().enumerate() {
        let weight = *layers as f64 * model.per_layer_weight_bytes();
        let read = weight / (n.mem_bw_gbps * 1e9 * MEM_EFFICIENCY);
        let hop = hop_seconds(topo, &ev.stages, i, act);
        read_total += read;
        hop_total += hop;
        if read + hop > worst.0 {
            worst = (read + hop, n.id.to_string());
        }
        prefill_s += (*layers as f64 / model.n_layers as f64) * 2.0 * model.params_b * 1e9 / (n.gflops.max(0.001) * 1e9 * COMPUTE_EFFICIENCY);
        stages.push(Stage {
            node: n.id.clone(),
            layer_start: layer,
            layer_end: layer + layers,
            weight_bytes: weight,
            est_ms_per_token: (read + hop) * 1000.0,
        });
        layer += layers;
    }
    if topo == Topology::Ring && ev.stages.len() > 1 {
        notes.push("ring hops estimated from each device's link to the coordinator (pairwise links not measured yet)".into());
    }
    if ev.stages.iter().any(|(n, _)| !n.link_measured) {
        notes.push("some links were not measured; using pessimistic defaults".into());
    }
    let bottleneck = if hop_total > read_total { "network".to_string() } else { format!("memory bandwidth of {}", worst.1) };
    Plan {
        model: model.name.clone(),
        topology: topo,
        params_b: model.params_b,
        need_bytes: need,
        feasible: true,
        stages,
        decode_tok_s: 1.0 / ev.decode_s,
        prefill_tok_s: 1.0 / prefill_s,
        bottleneck,
        notes,
    }
}

const GIB: f64 = (1u64 << 30) as f64;

/// Plan every model in `models`; return the plans and the index of the largest model that is
/// feasible and meets `min_decode_tok_s`.
pub fn recommend(models: &[ModelSpec], ctx: u32, nodes: &[PlanNode], min_decode_tok_s: f64) -> (Vec<Plan>, Option<usize>) {
    recommend_with(models, ctx, nodes, min_decode_tok_s, Topology::default())
}

pub fn recommend_with(models: &[ModelSpec], ctx: u32, nodes: &[PlanNode], min_decode_tok_s: f64, topo: Topology) -> (Vec<Plan>, Option<usize>) {
    let plans: Vec<Plan> = models.iter().map(|m| plan_model_with(m, ctx, nodes, topo)).collect();
    let pick = plans
        .iter()
        .enumerate()
        .filter(|(_, p)| p.feasible && p.decode_tok_s >= min_decode_tok_s)
        .max_by(|(_, a), (_, b)| a.params_b.partial_cmp(&b.params_b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    (plans, pick)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coordinator has no link to itself: no "no link measurement" warning for it, but still
    /// for a member whose link was not measured.
    #[test]
    fn the_coordinator_needs_no_link_measurement() {
        let bench = BenchReport { cpu_gflops_1t: 10.0, cpu_gflops_mt: 40.0, mem_bw_gbps: 40.0, disk_read_mbps: None, disk_write_mbps: None, accel_mem_bw_gbps: None, accel_tflops: None, took_ms: 1, measured_at_unix: 0, build: "release".into() };
        let report = |id: &str| {
            let mut info = crate::probe::probe_node(NodeId(id.into()));
            info.accelerators.clear();
            NodeReport { policy: Policy::default_for(info.class), info, bench: Some(bench.clone()), link: None }
        };
        let g = summarize(&[report("laptop"), report("phone")], Some(&NodeId("laptop".into())));
        assert_eq!(g.warnings.len(), 1, "{:?}", g.warnings);
        assert!(g.warnings[0].starts_with("phone: no link measurement"), "{:?}", g.warnings);
    }

    fn node(id: &str, mem_gib: f64, bw: f64, rtt: f64, mbps: f64, coord: bool) -> PlanNode {
        PlanNode {
            id: NodeId(id.into()),
            mem_budget_bytes: mem_gib * GIB,
            mem_bw_gbps: bw,
            gflops: 200.0,
            rtt_ms: if coord { 0.0 } else { rtt },
            link_mbps: mbps,
            is_coordinator: coord,
            link_measured: true,
            uses_accelerator: false,
        }
    }

    fn model(name: &str) -> ModelSpec {
        catalog(Quant::Q4).into_iter().find(|m| m.name.starts_with(name)).unwrap()
    }

    #[test]
    fn small_model_stays_on_one_node() {
        let nodes = vec![node("a", 24.0, 60.0, 0.0, 1000.0, true), node("b", 24.0, 60.0, 1.0, 1000.0, false)];
        let p = plan_model(&model("8b"), 8192, &nodes);
        assert!(p.feasible);
        assert_eq!(p.stages.len(), 1);
    }

    #[test]
    fn big_model_needs_the_swarm() {
        // 70B at Q4 is ~42 GB of weights: no single 32 GB budget holds it, two do.
        let nodes = vec![node("a", 32.0, 60.0, 0.0, 1000.0, true), node("b", 32.0, 60.0, 1.0, 1000.0, false)];
        let m = model("70b");
        let p = plan_model(&m, 4096, &nodes);
        assert!(p.feasible);
        assert_eq!(p.stages.len(), 2);
        let layers: u32 = p.stages.iter().map(|s| s.layer_end - s.layer_start).sum();
        assert_eq!(layers, m.n_layers);
        assert_eq!(p.stages[0].layer_start, 0);
        assert_eq!(p.stages[1].layer_start, p.stages[0].layer_end);
    }

    #[test]
    fn infeasible_reports_the_shortfall() {
        let nodes = vec![node("a", 8.0, 40.0, 0.0, 1000.0, true)];
        let p = plan_model(&model("70b"), 4096, &nodes);
        assert!(!p.feasible);
        assert!(p.notes.iter().any(|n| n.contains("needs")));
    }

    #[test]
    fn slow_network_is_the_bottleneck() {
        // Very fast memory (GPU-class) makes the weight reads cheap, so a slow link dominates.
        let nodes = vec![node("a", 32.0, 2000.0, 0.0, 1000.0, true), node("b", 32.0, 2000.0, 40.0, 20.0, false)];
        let p = plan_model(&model("70b"), 4096, &nodes);
        assert!(p.feasible);
        assert_eq!(p.bottleneck, "network");
    }

    #[test]
    fn more_bandwidth_means_faster_decode() {
        let slow = vec![node("a", 64.0, 30.0, 0.0, 1000.0, true)];
        let fast = vec![node("a", 64.0, 90.0, 0.0, 1000.0, true)];
        let m = model("32b");
        assert!(plan_model(&m, 4096, &fast).decode_tok_s > plan_model(&m, 4096, &slow).decode_tok_s);
    }

    #[test]
    fn recommend_picks_largest_that_meets_target() {
        let nodes = vec![node("a", 32.0, 60.0, 0.0, 1000.0, true), node("b", 32.0, 60.0, 1.0, 1000.0, false)];
        let models = catalog(Quant::Q4);
        let (plans, pick) = recommend(&models, 4096, &nodes, 1.0);
        let pick = pick.expect("something should fit");
        assert!(plans[pick].feasible);
        assert!(plans[pick].decode_tok_s >= 1.0);
        // Nothing larger that also qualifies.
        assert!(plans.iter().all(|p| !(p.feasible && p.decode_tok_s >= 1.0 && p.params_b > plans[pick].params_b)));
    }

    #[test]
    fn ring_beats_hub_when_the_network_matters() {
        // 70B needs all three 16 GiB budgets. The ring pays 3 one-way hops, the hub 4.
        let nodes = vec![
            node("a", 16.0, 2000.0, 0.0, 1000.0, true),
            node("b", 16.0, 2000.0, 4.0, 1000.0, false),
            node("c", 16.0, 2000.0, 4.0, 1000.0, false),
        ];
        let m = model("70b");
        let ring = plan_model_with(&m, 4096, &nodes, Topology::Ring);
        let hub = plan_model_with(&m, 4096, &nodes, Topology::Hub);
        assert!(ring.feasible && hub.feasible);
        assert_eq!(ring.stages.len(), 3);
        assert!(ring.decode_tok_s > hub.decode_tok_s, "ring {} hub {}", ring.decode_tok_s, hub.decode_tok_s);
    }

    #[test]
    fn single_stage_has_no_network_cost_in_either_topology() {
        let nodes = vec![node("a", 64.0, 80.0, 0.0, 1000.0, true)];
        let m = model("14b");
        let ring = plan_model_with(&m, 4096, &nodes, Topology::Ring);
        let hub = plan_model_with(&m, 4096, &nodes, Topology::Hub);
        assert!((ring.decode_tok_s - hub.decode_tok_s).abs() < 1e-9);
    }

    #[test]
    fn ring_plans_say_how_hops_were_estimated() {
        let nodes = vec![node("a", 32.0, 60.0, 0.0, 1000.0, true), node("b", 32.0, 60.0, 1.0, 1000.0, false)];
        let p = plan_model(&model("70b"), 4096, &nodes);
        assert!(p.notes.iter().any(|n| n.contains("pairwise")));
    }

    #[test]
    fn kv_cache_grows_with_context() {
        let m = model("8b");
        assert!(m.kv_bytes(32768) > m.kv_bytes(4096) * 7.9);
    }
}
