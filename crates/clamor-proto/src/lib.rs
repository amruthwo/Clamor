//! Clamor wire protocol (v2).
//!
//! Every message is a JSON object with a `"type"` tag, sent as a length-prefixed frame
//! (4-byte big-endian length, then the UTF-8 JSON body). JSON keeps v1 debuggable with
//! `nc` and `jq`. Bulk tensor traffic never uses this channel: it goes over the
//! data-plane connections negotiated by `ShardAssign`, so the control plane stays small.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::{self, Read, Write};

/// v2: `ShardAssign` gained `n_ctx`, `embed` and `sampling`; added `ShardFailed`.
pub const PROTOCOL_VERSION: u16 = 2;
/// Model sharing: every node serves layer slices of the models it holds on this HTTP port.
pub const DEFAULT_SHARE_PORT: u16 = 7423;
/// First data-plane port: stage `i` listens on `DEFAULT_DATA_PORT + i`.
pub const DEFAULT_DATA_PORT: u16 = 7430;
/// Control plane (node <-> coordinator).
pub const DEFAULT_PORT: u16 = 7420;
/// Web portal and JSON API served by the coordinator.
pub const DEFAULT_PORTAL_PORT: u16 = 7421;
/// Raw network benchmark server (see `clamor-muster`).
pub const DEFAULT_BENCH_PORT: u16 = 7422;
pub const MAX_FRAME_BYTES: u32 = 4 * 1024 * 1024;

// ---------------------------------------------------------------- identity

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceClass {
    Desktop,
    Laptop,
    Server,
    Phone,
    Tablet,
    Sbc,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Os {
    Linux,
    Macos,
    Windows,
    Android,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelBackend {
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Npu,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Accelerator {
    pub name: String,
    pub backend: AccelBackend,
    pub mem_bytes: u64,
    /// True when the accelerator shares system RAM (Apple Silicon, most SoCs).
    pub unified: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerState {
    pub on_battery: bool,
    pub battery_pct: Option<u8>,
}

/// Static description of a device, sent in `Hello`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub hostname: String,
    pub os: Os,
    pub arch: String,
    pub class: DeviceClass,
    pub cpu_model: String,
    pub cpu_cores_physical: u32,
    pub cpu_threads: u32,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub accelerators: Vec<Accelerator>,
    pub disk_free_bytes: u64,
    pub power: Option<PowerState>,
    pub clamor_version: String,
}

// ---------------------------------------------------------------- policy

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Participation {
    /// Accept any work that fits the limits below.
    Full,
    /// Work only while the device looks idle (no foreground load).
    IdleOnly,
    /// Stay in the swarm but take no new work.
    Paused,
    /// Finishing current work, then leaving. Set by the coordinator on `Drain`.
    Draining,
}

/// What the device owner allows Clamor to use. Enforced by the node, trusted by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub participation: Participation,
    pub max_ram_fraction: f32,
    pub max_cpu_fraction: f32,
    pub require_plugged_in: bool,
}

impl Policy {
    /// Conservative defaults for a device class: phones and laptops only work on mains power.
    pub fn default_for(class: DeviceClass) -> Self {
        let portable = matches!(class, DeviceClass::Phone | DeviceClass::Tablet | DeviceClass::Laptop);
        Policy {
            participation: if portable { Participation::IdleOnly } else { Participation::Full },
            max_ram_fraction: if portable { 0.5 } else { 0.7 },
            max_cpu_fraction: if portable { 0.5 } else { 0.8 },
            require_plugged_in: portable,
        }
    }
}

// ---------------------------------------------------------------- measurements

/// Results of the on-device micro-benchmarks (`clamor-muster`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    pub cpu_gflops_1t: f64,
    pub cpu_gflops_mt: f64,
    /// Aggregate sequential read bandwidth across all threads. Token generation is
    /// memory-bound, so this is the number that predicts decode speed.
    pub mem_bw_gbps: f64,
    pub disk_read_mbps: Option<f64>,
    pub disk_write_mbps: Option<f64>,
    /// Filled in once accelerator benchmarks exist; the planner ignores accelerators until then.
    pub accel_mem_bw_gbps: Option<f64>,
    pub accel_tflops: Option<f64>,
    pub took_ms: u64,
    pub measured_at_unix: u64,
    /// "release" or "debug": debug numbers are far too low to plan with.
    pub build: String,
}

/// Network quality between a node and the coordinator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkStat {
    pub rtt_ms_p50: f64,
    pub rtt_ms_p95: f64,
    pub up_mbps: Option<f64>,
    pub down_mbps: Option<f64>,
    pub samples: u32,
}

/// Everything the muster knows about one device. Also the on-disk roster file format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeReport {
    pub info: NodeInfo,
    pub policy: Policy,
    pub bench: Option<BenchReport>,
    pub link: Option<LinkStat>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeState {
    Joining,
    Benchmarking,
    Ready,
    Busy,
    Draining,
    /// Heartbeats stopped; kept in the roster so it can rejoin without re-benchmarking.
    Away,
}

// ---------------------------------------------------------------- tasks

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Embed,
    Summarize,
    Transcribe,
    Ocr,
    /// Generic LLM call on a model small enough for one node.
    Infer,
    /// A tool invocation requested by an agent (OpenClaw, Hermes, ...).
    ToolCall,
    Custom(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Requirements {
    pub min_ram_bytes: u64,
    pub needs_accelerator: bool,
    pub min_mem_bw_gbps: f64,
    pub needs_plugged_in: bool,
    /// Restrict to these operating systems (empty = any).
    pub os: Vec<Os>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub kind: TaskKind,
    pub requirements: Requirements,
    pub priority: u8,
    pub deadline_unix: Option<u64>,
    /// Kind-specific input. Large blobs travel out of band (content-addressed store).
    pub payload: serde_json::Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Assigned,
    Running,
    Done,
    Failed,
    Retrying,
    Cancelled,
}

// ---------------------------------------------------------------- messages

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActDType {
    F16,
    /// Per-frame symmetric int8: a quarter of the f32 traffic, small accuracy cost.
    I8,
}

/// What a node's binary is: reported on join, shown in the dashboard, matched by the updater.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Clamor version, e.g. "0.4.0".
    pub version: String,
    /// Rust target triple, e.g. "x86_64-unknown-linux-gnu".
    pub target: String,
    /// GPU backends compiled in: "cpu", "vulkan", "metal", "cuda", ...
    pub variant: String,
    /// CPU instruction-set level, e.g. "x86-64-v3".
    pub cpu: String,
    /// Git commit (12 hex digits) or "unknown".
    pub commit: String,
}

/// A model file a node holds and can share.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// File name in the node's models directory.
    pub name: String,
    /// File size in bytes.
    pub size: u64,
    /// SHA-256 of the GGUF header plus the file size: tells two copies of the same model apart
    /// from two different files with the same name, without hashing gigabytes.
    pub id: String,
}

/// How the tail stage picks tokens. `temperature == 0` is greedy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Keep only the k most likely tokens. 0 means no limit.
    pub top_k: u32,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    // --- join handshake ---
    Hello {
        proto: u16,
        info: NodeInfo,
        /// Port where this node runs the raw network-benchmark server.
        bench_port: u16,
        join_token: Option<String>,
        /// Optional so older nodes still parse.
        #[serde(default)]
        build: Option<BuildInfo>,
        /// Models this node holds.
        #[serde(default)]
        models: Vec<ModelInfo>,
        /// Port serving layer slices of those models (see `DEFAULT_SHARE_PORT`).
        #[serde(default)]
        share_port: Option<u16>,
    },
    /// The node's models changed (a file was added or removed).
    Inventory {
        models: Vec<ModelInfo>,
    },
    Welcome {
        swarm: String,
        coordinator: NodeId,
        heartbeat_secs: u32,
    },
    Reject {
        reason: String,
    },

    // --- liveness and link measurement ---
    Heartbeat {
        cpu_load: f32,
        ram_available_bytes: u64,
        power: Option<PowerState>,
        running_tasks: u32,
        /// A newer release this node could update to.
        #[serde(default)]
        update_available: Option<String>,
    },
    Ping {
        seq: u64,
        sent_unix_ms: u64,
    },
    Pong {
        seq: u64,
        sent_unix_ms: u64,
    },

    // --- muster ---
    BenchRequest {
        quick: bool,
    },
    BenchResult {
        report: BenchReport,
    },
    SetPolicy {
        policy: Policy,
    },

    // --- leaving and rejoining ---
    /// Coordinator asks the node to finish up; node replies `Drained` when idle.
    Drain {
        grace_secs: u32,
    },
    Drained,
    /// Node-initiated departure (the user needs the machine back).
    Leave,

    // --- task lane ---
    TaskAssign {
        task: Task,
    },
    TaskUpdate {
        task_id: String,
        state: TaskState,
        progress: Option<f32>,
        message: Option<String>,
    },
    TaskResult {
        task_id: String,
        ok: bool,
        output: serde_json::Value,
    },

    // --- big-model pipeline lane (activations travel on the data plane, see clamor-transport) ---
    /// Give a node its place in the ring. A new `epoch` supersedes every earlier assignment.
    ShardAssign {
        /// Model file name in the node's models directory (a bare file name, never a path).
        model: String,
        epoch: u32,
        session: u64,
        stage_index: u32,
        stage_count: u32,
        layer_start: u32,
        layer_end: u32,
        /// Port this stage listens on for its upstream neighbour.
        listen_port: u16,
        /// `host:port` of the next stage (the head, for the last stage).
        downstream: String,
        /// Encoding for activations leaving this stage.
        act_dtype: ActDType,
        /// KV cache length per request.
        n_ctx: u32,
        /// This stage owns the token embedding (stage 0). Usually it also holds layer 0, but an
        /// embedding-only head has `layer_start == layer_end == 0`.
        embed: bool,
        /// Used by the last stage.
        sampling: SamplingParams,
        /// Where to fetch this stage's slice when the node does not hold the model:
        /// `http://host:port` of a node that does. `None`: load the local file.
        #[serde(default)]
        source: Option<String>,
        /// The model's `ModelInfo::id`, so a node never mixes up two files with one name.
        #[serde(default)]
        model_id: Option<String>,
    },
    /// Download progress of a stage's slice.
    ShardProgress {
        epoch: u32,
        stage_index: u32,
        done_bytes: u64,
        total_bytes: u64,
    },
    ShardReady {
        model: String,
        epoch: u32,
        stage_index: u32,
        load_ms: u64,
    },
    /// A stage could not load, or stopped. That epoch's pipeline cannot run any more.
    ShardFailed {
        model: String,
        epoch: u32,
        stage_index: u32,
        error: String,
        /// The stage had been running and then stopped. Usually a knock-on effect (a neighbour's
        /// link closed), so the coordinator prefers to blame a failure that is not.
        stopped: bool,
    },
    ShardRelease {
        model: String,
        epoch: u32,
    },
}

// ---------------------------------------------------------------- framing

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Serialize a message into a complete frame (header + body).
pub fn encode(msg: &Message) -> io::Result<Vec<u8>> {
    let body = serde_json::to_vec(msg).map_err(io::Error::other)?;
    if body.len() > MAX_FRAME_BYTES as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame too large"));
    }
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Validate a 4-byte header and return the body length.
pub fn frame_len(header: [u8; 4]) -> io::Result<usize> {
    let n = u32::from_be_bytes(header);
    if n > MAX_FRAME_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame exceeds MAX_FRAME_BYTES"));
    }
    Ok(n as usize)
}

pub fn decode(body: &[u8]) -> io::Result<Message> {
    serde_json::from_slice(body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write_frame<W: Write>(w: &mut W, msg: &Message) -> io::Result<()> {
    w.write_all(&encode(msg)?)?;
    w.flush()
}

pub fn read_frame<R: Read>(r: &mut R) -> io::Result<Message> {
    let mut header = [0u8; 4];
    r.read_exact(&mut header)?;
    let mut body = vec![0u8; frame_len(header)?];
    r.read_exact(&mut body)?;
    decode(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_info() -> NodeInfo {
        NodeInfo {
            id: NodeId("test-node".into()),
            hostname: "test".into(),
            os: Os::Linux,
            arch: "x86_64".into(),
            class: DeviceClass::Desktop,
            cpu_model: "cpu".into(),
            cpu_cores_physical: 8,
            cpu_threads: 16,
            ram_total_bytes: 32 << 30,
            ram_available_bytes: 24 << 30,
            accelerators: vec![],
            disk_free_bytes: 500 << 30,
            power: None,
            clamor_version: "0.1.0".into(),
        }
    }

    #[test]
    fn hello_roundtrip() {
        let msg = Message::Hello { proto: PROTOCOL_VERSION, info: sample_info(), bench_port: 7422, join_token: None, build: None, models: vec![ModelInfo { name: "m.gguf".into(), size: 9, id: "ab".into() }], share_port: Some(7423) };
        let mut buf = Vec::new();
        write_frame(&mut buf, &msg).unwrap();
        let back = read_frame(&mut buf.as_slice()).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn unit_variants_roundtrip() {
        for msg in [Message::Drained, Message::Leave] {
            let mut buf = Vec::new();
            write_frame(&mut buf, &msg).unwrap();
            assert_eq!(read_frame(&mut buf.as_slice()).unwrap(), msg);
        }
    }

    #[test]
    fn task_payload_roundtrip() {
        let task = Task {
            id: "t1".into(),
            kind: TaskKind::Custom("thumbnail".into()),
            requirements: Requirements { os: vec![Os::Android], ..Default::default() },
            priority: 5,
            deadline_unix: None,
            payload: serde_json::json!({"blob": "sha256:abc", "w": 256}),
        };
        let msg = Message::TaskAssign { task };
        let mut buf = Vec::new();
        write_frame(&mut buf, &msg).unwrap();
        assert_eq!(read_frame(&mut buf.as_slice()).unwrap(), msg);
    }

    /// Nodes from before build/update reporting send Hello and Heartbeat without those fields.
    #[test]
    fn older_hello_and_heartbeat_still_parse() {
        let hb = r#"{"type":"heartbeat","cpu_load":0.5,"ram_available_bytes":1,"power":null,"running_tasks":0}"#;
        assert!(matches!(decode(hb.as_bytes()).unwrap(), Message::Heartbeat { update_available: None, .. }));
        let mut hello = serde_json::to_value(Message::Hello { proto: PROTOCOL_VERSION, info: sample_info(), bench_port: 1, join_token: None, build: None, models: vec![], share_port: None }).unwrap();
        for k in ["build", "models", "share_port"] {
            hello.as_object_mut().unwrap().remove(k);
        }
        assert!(matches!(decode(hello.to_string().as_bytes()).unwrap(), Message::Hello { build: None, .. }));
    }

    #[test]
    fn shard_assign_roundtrip() {
        let msg = Message::ShardAssign {
            model: "70b-q4".into(),
            epoch: 3,
            session: 99,
            stage_index: 1,
            stage_count: 3,
            layer_start: 27,
            layer_end: 54,
            listen_port: 7430,
            downstream: "192.168.1.12:7430".into(),
            act_dtype: ActDType::I8,
            n_ctx: 4096,
            embed: false,
            sampling: SamplingParams { temperature: 0.7, top_k: 40, seed: 5 },
            source: Some("http://192.168.1.20:7423".into()),
            model_id: Some("ab12".into()),
        };
        let mut buf = Vec::new();
        write_frame(&mut buf, &msg).unwrap();
        assert_eq!(read_frame(&mut buf.as_slice()).unwrap(), msg);
        let msg = Message::ShardFailed { model: "tiny.gguf".into(), epoch: 3, stage_index: 1, error: "out of memory".into(), stopped: false };
        let mut buf = Vec::new();
        write_frame(&mut buf, &msg).unwrap();
        assert_eq!(read_frame(&mut buf.as_slice()).unwrap(), msg);
    }

    #[test]
    fn oversized_header_rejected() {
        let header = (MAX_FRAME_BYTES + 1).to_be_bytes();
        assert!(frame_len(header).is_err());
    }

    #[test]
    fn portable_devices_default_to_plugged_in_only() {
        let p = Policy::default_for(DeviceClass::Phone);
        assert!(p.require_plugged_in);
        assert_eq!(p.participation, Participation::IdleOnly);
        assert_eq!(Policy::default_for(DeviceClass::Desktop).participation, Participation::Full);
    }
}
