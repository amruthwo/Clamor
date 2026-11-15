//! Clamor wire protocol (v1).
//!
//! Every message is a JSON object with a `"type"` tag, sent as a length-prefixed frame
//! (4-byte big-endian length, then the UTF-8 JSON body). JSON keeps v1 debuggable with
//! `nc` and `jq`. Bulk tensor traffic never uses this channel: it goes over the
//! data-plane connections negotiated by `ShardAssign`, so the control plane stays small.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::{self, Read, Write};

pub const PROTOCOL_VERSION: u16 = 1;
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
    },
    ShardReady {
        model: String,
        epoch: u32,
        stage_index: u32,
        load_ms: u64,
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
        let msg = Message::Hello { proto: PROTOCOL_VERSION, info: sample_info(), bench_port: 7422, join_token: None };
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
        };
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
