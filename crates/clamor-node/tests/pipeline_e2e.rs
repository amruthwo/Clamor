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
    // A huge declared context (16M tokens) lets a test request a pipeline that cannot fit in
    // memory; the other tests ask for 64 tokens.
    let ok = py(vec![format!("{tools}/gen_tiny_llama.py"), model.display().to_string(), "--layers".into(), "4".into(), "--bytes-vocab".into(), "--ctx".into(), "16777216".into()])
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

/// A run of `n` consecutive free ports (data ports for the stages). Chosen from 20000-32000,
/// below every OS's ephemeral range: ports from that range get reused for the test processes'
/// own outgoing connections (seen on macOS and Windows CI). Tests in this binary run in
/// parallel, so ranges handed out here are remembered and never given out twice.
fn free_port_run(n: u16) -> u16 {
    static TAKEN: std::sync::Mutex<Vec<u16>> = std::sync::Mutex::new(Vec::new());
    let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64 ^ (std::process::id() as u64) << 20;
    for _ in 0..500 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let base = 20_000 + (seed % 12_000) as u16;
        let mut taken = TAKEN.lock().unwrap();
        if taken.iter().any(|&t| base < t + n && t < base + n) {
            continue;
        }
        if (0..n).all(|i| TcpListener::bind(("0.0.0.0", base + i)).is_ok()) {
            taken.push(base);
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

/// A per-process settings folder, so test nodes never touch the real ~/.config/clamor.
fn cfg_dir(dir: &Path, name: &str) -> String {
    dir.join(format!("{name}-config")).display().to_string()
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

    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", "t", "--token", "T", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(30), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let a = spawn(&dir, "member-a", &["join", &ctrl_s, "--token", "T", "--id", "a", "--models", &models, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-a"), "--participation", "full", "--allow-battery"]);
    let b = spawn(&dir, "member-b", &["join", &ctrl_s, "--token", "T", "--id", "b", "--models", &models, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-b"), "--participation", "full", "--allow-battery"]);
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

    // Text chat: the reply must equal rendering + tokenizing the conversation ourselves, generating
    // from those ids, and decoding. (The model is random, so the text itself is gibberish.)
    let messages = json!([{ "role": "system", "content": "Be brief." }, { "role": "user", "content": "Héllo 😀" }]);
    let (st, chat) = http(portal, "POST", "/api/chat", Some(json!({ "messages": messages, "max_new": 12 })));
    assert_eq!(st, 200, "{chat}");
    let tok = clamor_engine_ggml::Tokenizer::from_gguf(&clamor_engine_ggml::GgufFile::open(dir.join("tiny.gguf").to_str().unwrap()).unwrap()).unwrap();
    let msgs: Vec<clamor_engine_ggml::ChatMessage> = serde_json::from_value(messages).unwrap();
    let prompt = clamor_engine_ggml::encode_chat(&tok, &msgs).unwrap();
    assert_eq!(chat["prompt_tokens"].as_u64(), Some(prompt.len() as u64), "{chat}");
    let (_, generated) = http(portal, "POST", "/api/generate", Some(json!({ "tokens": prompt, "max_new": 12 })));
    let ids: Vec<u32> = generated["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    assert_eq!(chat["message"]["content"].as_str(), Some(tok.decode(&ids, true).as_str()), "{chat}\n{generated}");
    assert_eq!(chat["done"], generated["done"]);
    let (st, r) = http(portal, "POST", "/api/chat", Some(json!({ "messages": [] })));
    assert_eq!(st, 400, "empty conversation: {r}");

    // 2. A new pipeline replaces the old one (new epoch); the coordinator holds layer 0 itself.
    let p2 = start(json!([{ "node": "coord", "layers": [0, 1] }, { "node": "a", "layers": [1, 3] }, { "node": "b", "layers": [3, 4] }]));
    assert!(p2["pipeline"]["epoch"].as_u64() > p["pipeline"]["epoch"].as_u64());
    assert_eq!(generate(), greedy, "coordinator with layers + 2 members");

    // Limits: a at the smallest RAM share (5%), asked for a 16M-token context (~8 GB of KV cache
    // per request for its two layers): refused up front with the numbers, not loaded until the
    // machine swaps.
    let policy = |id: &str, p: Value| {
        let (st, r) = http(portal, "POST", &format!("/api/nodes/{id}/policy"), Some(p));
        assert_eq!(st, 200, "{r}");
    };
    let full = json!({ "participation": "full", "max_ram_fraction": 0.7, "max_cpu_fraction": 0.8, "require_plugged_in": false });
    policy("a", json!({ "participation": "full", "max_ram_fraction": 0.05, "max_cpu_fraction": 0.8, "require_plugged_in": false }));
    let (st, r) = http(portal, "POST", "/api/pipeline", Some(json!({ "model": "tiny.gguf", "ctx": 16777216, "stages": [{ "node": "a", "layers": [0, 2] }, { "node": "b", "layers": [2, 4] }] })));
    assert_eq!(st, 400, "{r}");
    let msg = r["text"].as_str().unwrap_or("");
    assert!(msg.contains("on a") && msg.contains("may use"), "explains which device and why: {msg}");
    policy("a", full.clone());
    // The refused request left the running pipeline alone.
    assert_eq!(generate(), greedy, "still serving after a refused pipeline");

    // Pausing: a paused device takes no new requests (within one heartbeat), then resumes.
    policy("a", json!({ "participation": "paused", "max_ram_fraction": 0.7, "max_cpu_fraction": 0.8, "require_plugged_in": false }));
    wait_for("a's pause to reach the coordinator", Duration::from_secs(20), &all, || {
        let (st, r) = http(portal, "POST", "/api/generate", Some(json!({ "tokens": PROMPT, "max_new": 2 })));
        st == 503 && r["text"].as_str().unwrap_or("").contains("paused")
    });
    policy("a", full.clone());
    wait_for("a to resume", Duration::from_secs(20), &all, || http(portal, "POST", "/api/generate", Some(json!({ "tokens": PROMPT, "max_new": 2 }))).0 == 200);

    // Settings: nonsense is refused; a device remembers what it was given (its own folder).
    let (st, r) = http(portal, "POST", "/api/nodes/a/policy", Some(json!({ "participation": "full", "max_ram_fraction": 1.5, "max_cpu_fraction": 0.5, "require_plugged_in": false })));
    assert_eq!(st, 400, "{r}");
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("member-a-config/device-settings.json")).expect("a saved its settings")).unwrap();
    assert_eq!(saved["participation"], "full", "{saved}");
    // Normal priority cannot be applied while running on Linux/macOS: the device says so.
    if cfg!(unix) {
        policy("a", json!({ "participation": "full", "max_ram_fraction": 0.7, "max_cpu_fraction": 0.8, "require_plugged_in": false, "low_priority": false }));
        wait_for("a's priority note", Duration::from_secs(20), &all, || {
            let (_, r) = http(portal, "GET", "/api/roster", None);
            r["nodes"].as_array().unwrap().iter().any(|n| n["id"] == "a" && n["settings_note"].as_str().is_some_and(|s| s.contains("restart")))
        });
        policy("a", full);
    }

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

/// One copy of the model, on member b only. The coordinator (head) and member a fetch just
/// their layer slices from b, cache them, and the pipeline answers exactly as before.
#[test]
fn a_model_on_one_device_is_shared_with_the_others() {
    let dir = std::env::temp_dir().join(format!("clamor-node-share-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let Some(greedy) = python_fixture(&dir) else { return };
    let d = |n: &str| {
        let p = dir.join(n);
        std::fs::create_dir_all(&p).unwrap();
        p.display().to_string()
    };
    let (coord_models, a_models, b_models) = (d("coord-models"), d("a-models"), d("b-models"));
    let (coord_cache, a_cache, b_cache) = (d("coord-cache"), d("a-cache"), d("b-cache"));
    std::fs::rename(dir.join("tiny.gguf"), Path::new(&b_models).join("tiny.gguf")).unwrap();

    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(3));
    let (ctrl_s, portal_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());
    let coord = spawn(&dir, "coordinator", &["coordinator", "--token", "T", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &coord_models, "--cache-dir", &coord_cache, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(30), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let a = spawn(&dir, "member-a", &["join", &ctrl_s, "--token", "T", "--id", "a", "--models", &a_models, "--cache-dir", &a_cache, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-a"), "--participation", "full", "--allow-battery"]);
    let b = spawn(&dir, "member-b", &["join", &ctrl_s, "--token", "T", "--id", "b", "--models", &b_models, "--cache-dir", &b_cache, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-b"), "--participation", "full", "--allow-battery"]);
    let all = [&coord, &a, &b];
    let mut b_share = 0;
    wait_for("members to finish the muster", Duration::from_secs(120), &all, || {
        let (_, r) = http(portal, "GET", "/api/roster", None);
        let nodes = r["nodes"].as_array().cloned().unwrap_or_default();
        if let Some(bn) = nodes.iter().find(|n| n["id"] == "b") {
            b_share = bn["share_port"].as_u64().unwrap_or(0) as u16;
        }
        ["a", "b"].iter().all(|id| nodes.iter().any(|n| n["id"] == *id && n["state"] == "ready"))
    });
    let (_, r) = http(portal, "GET", "/api/roster", None);
    let holds = |id: &str| r["nodes"].as_array().unwrap().iter().find(|n| n["id"] == id).unwrap()["models"].as_array().unwrap().len();
    assert_eq!((holds("coord"), holds("a"), holds("b")), (0, 0, 1), "only b has the model: {r}");

    // b's model server wants the swarm token.
    let (st, _) = http(b_share, "GET", "/models", None);
    assert_eq!(st, 401, "model sharing must require the token");

    // Nobody has this one.
    let (st, r) = http(portal, "POST", "/api/pipeline", Some(json!({ "model": "nope.gguf" })));
    assert_eq!(st, 400, "{r}");
    assert!(r["text"].as_str().unwrap_or("").contains("no device"), "{r}");

    let start = || {
        let (st, r) = http(portal, "POST", "/api/pipeline", Some(json!({ "model": "tiny.gguf", "ctx": 64, "stages": [{ "node": "a", "layers": [0, 2] }, { "node": "b", "layers": [2, 4] }] })));
        assert_eq!(st, 200, "{r}");
        wait_for("the pipeline", Duration::from_secs(180), &all, || {
            let (_, r) = http(portal, "GET", "/api/pipeline", None);
            assert!(r["failure"].is_null(), "pipeline failed: {r}");
            r["ready"] == true
        });
    };
    start();
    let (st, r) = http(portal, "POST", "/api/generate", Some(json!({ "tokens": PROMPT, "max_new": N_GEN })));
    assert_eq!(st, 200, "{r}");
    let tokens: Vec<u32> = r["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    assert_eq!(tokens, greedy, "shared slices must compute exactly what the full file does");
    let (st, r) = http(portal, "POST", "/api/chat", Some(json!({ "messages": [{ "role": "user", "content": "Hi" }], "max_new": 4 })));
    assert_eq!(st, 200, "chat works from the fetched metadata: {r}");

    // The slices landed in the caches: a has layers 0..2, the coordinator its embedding slice
    // and the metadata; b, which holds the file, cached nothing.
    let files = |p: &str| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(p).unwrap().flatten().flat_map(|d| std::fs::read_dir(d.path()).unwrap().flatten().map(|f| f.file_name().to_string_lossy().into_owned())).collect();
        v.sort();
        v
    };
    assert_eq!(files(&a_cache), vec!["tiny.layers-0-2.gguf"]);
    assert_eq!(files(&coord_cache), vec!["tiny.layers-0-0-embed.gguf", "tiny.layers-0-0.gguf"]);
    assert!(files(&b_cache).is_empty(), "b used its own file");

    // A second pipeline with the same layout loads from the cache: no new download.
    let a_slice = std::fs::read_dir(&a_cache).unwrap().next().unwrap().unwrap().path().join("tiny.layers-0-2.gguf");
    let before = std::fs::metadata(&a_slice).unwrap().modified().unwrap();
    std::thread::sleep(Duration::from_millis(1100));
    start();
    assert_eq!(std::fs::metadata(&a_slice).unwrap().modified().unwrap(), before, "cached slice reused");
    drop((a, b));
}
