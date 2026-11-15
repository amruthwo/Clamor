//! End to end with real processes: a coordinator and two members on localhost, a pipeline set up
//! through the HTTP API, and generation compared with the numpy reference.
//!
//! Needs python3 with numpy + gguf (skips without, unless CLAMOR_REQUIRE_PY=1).
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PROMPT: [u32; 6] = [1, 5, 9, 23, 42, 7];
const N_GEN: usize = 16;

fn python_fixture(dir: &Path) -> Option<Vec<u32>> {
    let tools = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools");
    let model = dir.join("tiny.gguf");
    let py = |args: Vec<String>| Command::new("python3").args(&args).output().map(|o| o.status.success()).unwrap_or(false);
    let ok = py(vec![format!("{tools}/gen_tiny_llama.py"), model.display().to_string(), "--layers".into(), "4".into()])
        && py(vec![
            format!("{tools}/ref_forward.py"),
            model.display().to_string(),
            dir.join("ref.npz").display().to_string(),
            "--tokens".into(),
            PROMPT.map(|t| t.to_string()).join(","),
            "--generate".into(),
            N_GEN.to_string(),
            "--raw".into(),
            dir.join("ref").display().to_string(),
        ]);
    if !ok {
        if std::env::var("CLAMOR_REQUIRE_PY").as_deref() == Ok("1") {
            panic!("python3 with numpy+gguf is required (CLAMOR_REQUIRE_PY=1)");
        }
        eprintln!("skipping: python3 with numpy+gguf not available");
        return None;
    }
    Some(std::fs::read_to_string(dir.join("ref/generated.txt")).unwrap().split(',').map(|t| t.parse().unwrap()).collect())
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A run of `n` consecutive free ports (data ports for the stages).
fn free_port_run(n: u16) -> u16 {
    for _ in 0..200 {
        let base = 20_000 + (std::process::id() as u16 % 20_000) + (Instant::now().elapsed().subsec_nanos() % 997) as u16;
        if (0..n).all(|i| TcpListener::bind(("0.0.0.0", base + i)).is_ok()) {
            return base;
        }
    }
    panic!("no free port run");
}

struct Proc(Child, &'static str, PathBuf);
impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
impl Proc {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.2).unwrap_or_default()
    }
}

fn spawn(dir: &Path, name: &'static str, args: &[&str]) -> Proc {
    let log = dir.join(format!("{name}.log"));
    let f = std::fs::File::create(&log).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_clamor-node")).args(args).stdout(Stdio::from(f.try_clone().unwrap())).stderr(Stdio::from(f)).spawn().unwrap();
    Proc(child, name, log)
}

/// Minimal HTTP/1.1 client: returns (status, JSON body or {"text": body}).
fn http(port: u16, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status: u16 = out.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let payload = out.split_once("\r\n\r\n").map(|x| x.1).unwrap_or("");
    (status, serde_json::from_str(payload).unwrap_or_else(|_| json!({ "text": payload })))
}

fn wait_for(what: &str, timeout: Duration, procs: &[&Proc], mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while t0.elapsed() < timeout {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let logs: Vec<String> = procs.iter().map(|p| format!("--- {} ---\n{}", p.1, p.log())).collect();
    panic!("timed out waiting for {what}\n{}", logs.join("\n"));
}

#[test]
fn coordinator_and_two_members_generate_through_the_api() {
    let dir = std::env::temp_dir().join(format!("clamor-node-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let Some(greedy) = python_fixture(&dir) else { return };
    let models = dir.display().to_string();
    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(3));
    let (ctrl_s, portal_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());

    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", "t", "--token", "T", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench"]);
    wait_for("the portal", Duration::from_secs(30), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let a = spawn(&dir, "member-a", &["join", &ctrl_s, "--token", "T", "--id", "a", "--models", &models]);
    let b = spawn(&dir, "member-b", &["join", &ctrl_s, "--token", "T", "--id", "b", "--models", &models]);
    let all = [&coord, &a, &b];
    wait_for("members to finish the muster", Duration::from_secs(120), &all, || {
        let (_, r) = http(portal, "GET", "/api/roster", None);
        ["a", "b"].iter().all(|id| r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == *id && n["state"] == "ready")))
    });

    let start = |stages: Value| {
        let (st, r) = http(portal, "POST", "/api/pipeline", Some(json!({ "model": "tiny.gguf", "ctx": 64, "stages": stages })));
        assert_eq!(st, 200, "{r}");
        // Generous: the first GPU-backend start in a new binary can compile shaders (macOS Metal
        // took 21 s on a CI runner) even when the stages run on the CPU.
        wait_for("the pipeline", Duration::from_secs(180), &all, || {
            let (_, r) = http(portal, "GET", "/api/pipeline", None);
            assert!(r["failure"].is_null(), "pipeline failed: {r}");
            r["ready"] == true
        });
        let (_, r) = http(portal, "GET", "/api/pipeline", None);
        r
    };
    let generate = || {
        let (st, r) = http(portal, "POST", "/api/generate", Some(json!({ "tokens": PROMPT, "max_new": N_GEN })));
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["done"], "max_tokens", "{r}");
        r["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect::<Vec<u32>>()
    };

    // 1. Members hold every layer; the coordinator is an embedding-only head.
    let p = start(json!([{ "node": "a", "layers": [0, 2] }, { "node": "b", "layers": [2, 4] }]));
    let stages = p["pipeline"]["stages"].as_array().unwrap();
    assert_eq!(stages.len(), 3, "{p}");
    assert_eq!((stages[0]["node"].as_str(), stages[0]["layer_end"].as_u64()), (Some("coord"), Some(0)));
    assert_eq!(generate(), greedy, "embedding-only head + 2 members");

    // 2. A new pipeline replaces the old one (new epoch); the coordinator holds layer 0 itself.
    let p2 = start(json!([{ "node": "coord", "layers": [0, 1] }, { "node": "a", "layers": [1, 3] }, { "node": "b", "layers": [3, 4] }]));
    assert!(p2["pipeline"]["epoch"].as_u64() > p["pipeline"]["epoch"].as_u64());
    assert_eq!(generate(), greedy, "coordinator with layers + 2 members");

    // 3. Requests that cannot work are refused up front.
    for (bad, why) in [
        (json!({ "model": "../tiny.gguf" }), "path in model name"),
        (json!({ "model": "missing.gguf" }), "missing model"),
        (json!({ "model": "tiny.gguf", "stages": [{ "node": "a", "layers": [0, 3] }] }), "short layout"),
        (json!({ "model": "tiny.gguf", "stages": [{ "node": "ghost", "layers": [0, 4] }] }), "unknown node"),
    ] {
        let (st, r) = http(portal, "POST", "/api/pipeline", Some(bad));
        assert_eq!(st, 400, "{why}: {r}");
    }
    // ...and the running pipeline survives a refused request.
    assert_eq!(generate(), greedy, "pipeline still serves after refused requests");

    // 4. Losing a member fails the pipeline visibly instead of hanging requests.
    drop(b);
    let mut last = Value::Null;
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(30) {
        let (_, r) = http(portal, "GET", "/api/pipeline", None);
        last = r;
        if last["failure"].as_str().is_some_and(|f| f.contains("(b)")) {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(last["failure"].as_str().is_some_and(|f| f.contains("(b)")), "pipeline should blame member b: {last}\n{}", coord.log());
    let (st, r) = http(portal, "POST", "/api/generate", Some(json!({ "tokens": PROMPT, "max_new": 4 })));
    assert_eq!(st, 503, "{r}");

    let (st, r) = http(portal, "DELETE", "/api/pipeline", None);
    assert_eq!(st, 200, "{r}");
    assert!(r["released_epoch"].is_u64(), "{r}");
    let (_, r) = http(portal, "GET", "/api/pipeline", None);
    assert!(r["pipeline"].is_null(), "{r}");
    drop(a);
}
