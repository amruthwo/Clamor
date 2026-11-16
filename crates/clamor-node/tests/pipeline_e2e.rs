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

/// A free port for a listener, outside the OS's ephemeral range (see `free_port_run`: ports
/// from that range get reused as the source ports of other tests' outgoing connections, which
/// once made a coordinator's dashboard unreachable mid-test on CI).
fn free_port() -> u16 {
    free_port_run(1)
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
        // A failing test shows what every process it started was doing.
        if std::thread::panicking() {
            let log = self.log();
            let tail: Vec<&str> = log.lines().rev().take(40).collect();
            eprintln!("--- {} (last lines) ---\n{}", self.1, tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
        }
    }
}
impl Proc {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.2).unwrap_or_default()
    }
}

/// The settings a device saved for the (only) swarm it is in.
fn swarm_settings(config_dir: &Path) -> Option<String> {
    let e = std::fs::read_dir(config_dir.join("swarms")).ok()?.flatten().find(|e| e.file_name().to_string_lossy().starts_with("settings-"))?;
    std::fs::read_to_string(e.path()).ok()
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

/// Timeouts are generous: hosted CI runners (macOS especially) run these tests in parallel with
/// others that compile GPU shaders, and a coordinator once took over 30 s to open its portal.
/// HTTP with an optional cookie; returns (status, Set-Cookie value if any, JSON body).
fn http_c(port: u16, method: &str, path: &str, body: Option<Value>, cookie: Option<&str>) -> (u16, Option<String>, Value) {
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let cookie = cookie.map(|c| format!("Cookie: {c}\r\n")).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{cookie}Connection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status: u16 = out.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let (head, payload) = out.split_once("\r\n\r\n").unwrap_or((&out, ""));
    let set_cookie = head.lines().find_map(|l| l.strip_prefix("set-cookie: ").or_else(|| l.strip_prefix("Set-Cookie: "))).and_then(|v| v.split(';').next()).map(str::to_string);
    (status, set_cookie, serde_json::from_str(payload).unwrap_or_else(|_| json!({ "text": payload })))
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

    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", "t", "--passphrase", "tiny-test-words", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery", "--priority", "normal"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let a = spawn(&dir, "member-a", &["join", &ctrl_s, "--passphrase", "tiny-test-words", "--id", "a", "--models", &models, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-a"), "--participation", "full", "--allow-battery", "--priority", "normal"]);
    let b = spawn(&dir, "member-b", &["join", &ctrl_s, "--passphrase", "tiny-test-words", "--id", "b", "--models", &models, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-b"), "--participation", "full", "--allow-battery", "--priority", "normal"]);
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

    // Conversations: the group gets the message and the model's reply; the Home overview
    // shows the model running and the reply's speed.
    let (st, r) = http(portal, "POST", "/api/chats/group/messages", Some(json!({ "content": "Hello" })));
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["message"]["role"], "assistant");
    assert!(r["message"]["error"].is_null() && r["message"]["tok_s"].as_f64().is_some_and(|t| t > 0.0), "{r}");
    let (_, g) = http(portal, "GET", "/api/chats/group", None);
    let msgs = g["messages"].as_array().unwrap();
    assert_eq!((msgs.len(), msgs[0]["role"].as_str(), msgs[0]["content"].as_str()), (2, Some("user"), Some("Hello")), "{g}");
    let (_, o) = http(portal, "GET", "/api/overview", None);
    assert_eq!((o["model"]["state"].as_str(), o["health"]["online"].as_u64(), o["health"]["status"].as_str()), (Some("ready"), Some(3), Some("good")), "{o}");
    assert_eq!(o["speeds"].as_array().map(|s| s.len()), Some(1), "{o}");
    assert_eq!(http(portal, "POST", "/api/chats/group/messages", Some(json!({ "content": "  " }))).0, 400);

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
    let saved: Value = serde_json::from_str(&swarm_settings(&dir.join("member-a-config")).expect("a saved its settings")).unwrap();
    assert_eq!(saved["participation"], "full", "{saved}");
    // Priority is just another setting now (it applies from the next pipeline, on every OS).
    policy("a", json!({ "participation": "full", "max_ram_fraction": 0.7, "max_cpu_fraction": 0.8, "require_plugged_in": false, "low_priority": false }));
    wait_for("a to save normal priority", Duration::from_secs(20), &all, || {
        swarm_settings(&dir.join("member-a-config")).is_some_and(|s| s.contains("\"low_priority\": false"))
    });
    policy("a", full);

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
    let coord = spawn(&dir, "coordinator", &["coordinator", "--passphrase", "tiny-test-words", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &coord_models, "--cache-dir", &coord_cache, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery", "--priority", "normal"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let a = spawn(&dir, "member-a", &["join", &ctrl_s, "--passphrase", "tiny-test-words", "--id", "a", "--models", &a_models, "--cache-dir", &a_cache, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-a"), "--participation", "full", "--allow-battery", "--priority", "normal"]);
    let b = spawn(&dir, "member-b", &["join", &ctrl_s, "--passphrase", "tiny-test-words", "--id", "b", "--models", &b_models, "--cache-dir", &b_cache, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member-b"), "--participation", "full", "--allow-battery", "--priority", "normal"]);
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

    // b's model server answers only signed requests from the swarm's devices.
    let (st, _) = http(b_share, "GET", "/models", None);
    assert_eq!(st, 401, "model sharing must require a device of the swarm");

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
        let mut v: Vec<String> = std::fs::read_dir(p).unwrap().flatten().filter(|d| d.path().is_dir()).flat_map(|d| std::fs::read_dir(d.path()).unwrap().flatten().map(|f| f.file_name().to_string_lossy().into_owned())).collect();
        v.sort();
        v
    };
    assert_eq!(files(&a_cache), vec!["tiny.layers-0-2.gguf"]);
    assert_eq!(files(&coord_cache), vec!["tiny.layers-0-0-embed.gguf", "tiny.layers-0-0.gguf"]);
    assert!(files(&b_cache).is_empty(), "b used its own file");
    // The cache index records which swarm each slice was fetched for (for `leave --purge`).
    let index: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(&a_cache).join("index.json")).unwrap()).unwrap();
    let swarms = index.as_object().unwrap().values().next().unwrap()["swarms"].clone();
    assert_eq!(swarms.as_array().map(|a| a.len()), Some(1), "{index}");

    // A second pipeline with the same layout loads from the cache: no new download.
    let a_slice = std::fs::read_dir(&a_cache).unwrap().flatten().find(|d| d.path().is_dir()).unwrap().path().join("tiny.layers-0-2.gguf");
    let before = std::fs::metadata(&a_slice).unwrap().modified().unwrap();
    std::thread::sleep(Duration::from_millis(1100));
    start();
    assert_eq!(std::fs::metadata(&a_slice).unwrap().modified().unwrap(), before, "cached slice reused");
    drop((a, b));
}

/// Regression: priority used to be lowered for the whole process, so on a loaded machine the
/// threads sending heartbeats starved and the coordinator timed the device out (seen on CI).
/// Now only the compute threads run at low priority.
#[cfg(target_os = "linux")]
#[test]
fn only_compute_threads_run_at_low_priority() {
    let dir = std::env::temp_dir().join(format!("clamor-node-prio-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    if python_fixture(&dir).is_none() {
        return;
    }
    let models = dir.display().to_string();
    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(1));
    let (ctrl_s, portal_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());
    // Default priority (low), unlike the other tests.
    let coord = spawn(&dir, "coordinator", &["coordinator", "--id", "c", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let (st, r) = http(portal, "POST", "/api/pipeline", Some(json!({ "model": "tiny.gguf", "ctx": 64, "stages": [{ "node": "c", "layers": [0, 4] }] })));
    assert_eq!(st, 200, "{r}");
    wait_for("the pipeline", Duration::from_secs(180), &[&coord], || http(portal, "GET", "/api/pipeline", None).1["ready"] == true);
    let (st, r) = http(portal, "POST", "/api/generate", Some(json!({ "tokens": PROMPT, "max_new": 4 })));
    assert_eq!(st, 200, "{r}");

    // (thread name, nice) for every thread of the coordinator.
    // Threads come and go (tokio's blocking pool): skip one that exits while we look.
    let threads: Vec<(String, i64)> = std::fs::read_dir(format!("/proc/{}/task", coord.0.id()))
        .unwrap()
        .flatten()
        .filter_map(|t| {
            let name = std::fs::read_to_string(t.path().join("comm")).ok()?.trim().to_string();
            let stat = std::fs::read_to_string(t.path().join("stat")).ok()?;
            // Fields after the ")" that closes the name; nice is field 19 overall.
            let after = &stat[stat.rfind(')').unwrap() + 2..];
            let nice = after.split(' ').nth(16)?.parse().ok()?;
            Some((name, nice))
        })
        .collect();
    let compute: Vec<i64> = threads.iter().filter(|(n, _)| n == "clamor-compute").map(|t| t.1).collect();
    let runtime: Vec<i64> = threads.iter().filter(|(n, _)| n.starts_with("tokio")).map(|t| t.1).collect();
    assert!(!compute.is_empty() && compute.iter().all(|&n| n == 10), "compute threads at low priority: {threads:?}");
    assert!(!runtime.is_empty() && runtime.iter().all(|&n| n == 0), "heartbeats and networking at normal priority: {threads:?}");
}

/// Waits for a process to exit; returns whether it succeeded.
fn exits(p: &mut Proc, timeout: Duration) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < timeout {
        if let Some(st) = p.0.try_wait().unwrap() {
            return st.success();
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("{} did not exit\n{}", p.1, p.log());
}

/// Joining by passphrase: a wrong one is refused, the right one approves the device, and from
/// then on `clamor-node join` alone rejoins (the device is known by its key), also after the
/// coordinator restarts. A different coordinator at the same address is not trusted without
/// the passphrase.
#[test]
fn devices_join_by_passphrase_and_rejoin_by_key() {
    let dir = std::env::temp_dir().join(format!("clamor-node-join-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let models = dir.join("models").display().to_string();
    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(1));
    let (ctrl_s, portal_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());
    let (coord_cfg, member_cfg) = (cfg_dir(&dir, "coordinator"), cfg_dir(&dir, "member"));
    let coordinator = |name: &'static str, cfg: &str| {
        let c = spawn(&dir, name, &["coordinator", "--swarm", "home", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", cfg, "--participation", "full", "--allow-battery"]);
        wait_for("the portal", Duration::from_secs(120), &[&c], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
        c
    };
    let member = |name: &'static str, extra: &[&str]| {
        let mut args = vec!["join"];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["--id", "m", "--models", &models, "--no-update-check", "--config-dir", &member_cfg, "--participation", "full", "--allow-battery"]);
        spawn(&dir, name, &args)
    };
    // The new process must say it joined: right after the previous member process is killed, the
    // roster can still show it as ready for a moment.
    let joined = |procs: &[&Proc]| {
        wait_for("the member to join", Duration::from_secs(60), procs, || {
            let (_, r) = http(portal, "GET", "/api/roster", None);
            procs.last().unwrap().log().contains("joined swarm") && r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == "m" && n["state"] == "ready"))
        })
    };

    // A new swarm gets five random words.
    let coord = coordinator("coordinator", &coord_cfg);
    wait_for("the passphrase", Duration::from_secs(10), &[&coord], || coord.log().contains("passphrase: "));
    let log = coord.log();
    let pass = log.split("passphrase: ").nth(1).unwrap().lines().next().unwrap().trim().to_string();
    assert_eq!(pass.split('-').count(), 5, "{pass}");
    assert!(!std::fs::read_dir(format!("{coord_cfg}/swarms")).unwrap().next().is_none());

    // The control connection is TLS only: a plaintext hello gets no plaintext answer.
    {
        let mut raw = TcpStream::connect(("127.0.0.1", ctrl)).unwrap();
        raw.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let hello = br#"{"type":"hello","proto":5}"#;
        raw.write_all(&(hello.len() as u32).to_be_bytes()).unwrap();
        raw.write_all(hello).unwrap();
        let mut got = Vec::new();
        let _ = raw.read_to_end(&mut got);
        assert!(!String::from_utf8_lossy(&got).contains("reject"), "plaintext answer: {got:?}");
    }

    // First join: no passphrase, or a wrong one, is refused.
    let mut m = member("member-nopass", &[&ctrl_s]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("passphrase"), "{}", m.log());
    let mut m = member("member-wrong", &[&ctrl_s, "--passphrase", "not-the-right-words"]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("wrong passphrase"), "{}", m.log());
    assert!(coord.log().contains("wrong passphrase"));

    // The right one (case and separators do not matter).
    let spaced = pass.replace('-', " ").to_uppercase();
    let m = member("member-first", &[&ctrl_s, "--passphrase", &spaced]);
    joined(&[&coord, &m]);
    assert!(coord.log().contains("approved"), "{}", coord.log());
    drop(m);

    // Rejoin with nothing but `join`: the address is remembered, the key is approved.
    let mut m = member("member-rejoin", &[]);
    joined(&[&coord, &m]);
    assert!(m.log().contains("rejoining swarm 'home'"), "{}", m.log());

    // Removed in the dashboard: it leaves, and its key is no longer enough to come back.
    let (st, r) = http(portal, "POST", "/api/nodes/m/remove", None);
    assert_eq!((st, r["revoked"].as_bool()), (200, Some(true)), "{r}");
    assert!(exits(&mut m, Duration::from_secs(30)), "a removed device leaves gracefully\n{}", m.log());
    let mut m = member("member-removed", &[]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("not approved"), "{}", m.log());
    let m = member("member-readded", &["--passphrase", &pass]);
    joined(&[&coord, &m]);
    // Settings given in the dashboard are remembered by the device, for this swarm.
    let (st, r) = http(portal, "POST", "/api/nodes/m/policy", Some(json!({ "participation": "full", "max_ram_fraction": 0.3, "max_cpu_fraction": 0.5, "require_plugged_in": false })));
    assert_eq!(st, 200, "{r}");
    let member_dir = dir.join("member-config");
    wait_for("m to save its settings", Duration::from_secs(20), &[&coord, &m], || swarm_settings(&member_dir).is_some_and(|s| s.contains("0.3")));
    drop(m);

    // The coordinator restarts: same swarm, same passphrase, same approvals.
    drop(coord);
    let coord = coordinator("coordinator-again", &coord_cfg);
    assert!(coord.log().contains(&pass) && coord.log().contains("1 device(s) approved"), "{}", coord.log());
    let m = member("member-after-restart", &[]);
    joined(&[&coord, &m]);
    // The restarted coordinator knew nothing of m's settings: the device brought them.
    wait_for("m's settings to reach the coordinator", Duration::from_secs(20), &[&coord, &m], || {
        let (_, r) = http(portal, "GET", "/api/roster", None);
        r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == "m" && n["report"]["policy"]["max_ram_fraction"].as_f64().is_some_and(|f| (f - 0.3).abs() < 1e-6)))
    });
    drop(m);
    drop(coord);

    // Someone else's coordinator at the same address, even with the same swarm name: the device
    // does not know it, so it will not join without that swarm's passphrase.
    let other = coordinator("coordinator-other", &cfg_dir(&dir, "other"));
    let mut m = member("member-other", &[]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("passphrase"), "{}", m.log());
    drop(other);

    // `swarms` lists what the member remembers.
    let out = Command::new(env!("CARGO_BIN_EXE_clamor-node")).args(["swarms", "--config-dir", &member_cfg]).output().unwrap();
    let out = String::from_utf8_lossy(&out.stdout);
    assert!(out.contains("joined 'home'") && out.contains(&ctrl_s), "{out}");

    // `leave` forgets it: joining again needs the passphrase.
    let st = Command::new(env!("CARGO_BIN_EXE_clamor-node")).args(["leave", "home", "--purge", "--config-dir", &member_cfg, "--cache-dir", &dir.join("cache").display().to_string()]).status().unwrap();
    assert!(st.success());
    let out = Command::new(env!("CARGO_BIN_EXE_clamor-node")).args(["swarms", "--config-dir", &member_cfg]).output().unwrap();
    assert!(!String::from_utf8_lossy(&out.stdout).contains("joined 'home'"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `clamor-node join` without an address finds the coordinator on the local network (mDNS),
/// for a first join (by `--swarm` name) and a rejoin (by the remembered swarm id). Required on
/// Linux; elsewhere it is skipped if the test machine's network blocks multicast.
#[test]
fn devices_find_the_swarm_on_the_local_network() {
    let dir = std::env::temp_dir().join(format!("clamor-node-mdns-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let models = dir.join("models").display().to_string();
    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(1));
    let (ctrl_s, portal_s, data_s) = (format!("0.0.0.0:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());
    let swarm = format!("e2e-{}", std::process::id());
    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", &swarm, "--passphrase", "find-me-on-the-lan", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    wait_for("the coordinator to advertise the swarm", Duration::from_secs(10), &[&coord], || coord.log().contains("advertising"));
    let member_cfg = cfg_dir(&dir, "member");
    let member = |name: &'static str, extra: &[&str]| {
        let mut args = vec!["join"];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["--id", "m", "--models", &models, "--no-update-check", "--config-dir", &member_cfg, "--participation", "full", "--allow-battery"]);
        spawn(&dir, name, &args)
    };
    // The new process must say it joined: right after the previous member process is killed, the
    // roster can still show it as ready for a moment.
    let joined = |procs: &[&Proc]| {
        wait_for("the member to join", Duration::from_secs(60), procs, || {
            let (_, r) = http(portal, "GET", "/api/roster", None);
            procs.last().unwrap().log().contains("joined swarm") && r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == "m" && n["state"] == "ready"))
        })
    };
    let mut m = member("member-first", &["--swarm", &swarm, "--passphrase", "find-me-on-the-lan"]);
    std::thread::sleep(Duration::from_secs(6));
    if !m.log().contains("found swarm") {
        if cfg!(target_os = "linux") {
            panic!("discovery failed on Linux:\n{}", m.log());
        }
        eprintln!("skipping: multicast discovery does not work on this machine\n{}", m.log());
        let _ = m.0.kill();
        return;
    }
    joined(&[&coord, &m]);
    drop(m);
    let m = member("member-rejoin", &[]);
    joined(&[&coord, &m]);
    assert!(m.log().contains(&format!("found swarm '{swarm}'")), "{}", m.log());
    let _ = std::fs::remove_dir_all(&dir);
}

/// "Add device": a one-time code lets one device in, once; wrong codes are refused and, after a
/// few, cancel it; the device then rejoins by its key. Codes come only from the coordinator
/// itself (the test's requests come from localhost).
#[test]
fn devices_join_with_a_one_time_code() {
    let dir = std::env::temp_dir().join(format!("clamor-node-code-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let models = dir.join("models").display().to_string();
    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(1));
    let (ctrl_s, portal_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());
    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", "codes", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let member = |name: &'static str, id: &str, extra: &[&str]| {
        let mut args = vec!["join", &ctrl_s, "--id", id, "--models", &models, "--no-update-check", "--participation", "full", "--allow-battery"];
        args.extend_from_slice(extra);
        let cfg = cfg_dir(&dir, id);
        args.extend_from_slice(&["--config-dir", &cfg]);
        spawn(&dir, name, &args)
    };
    let joined = |id: &str, procs: &[&Proc]| {
        wait_for("the member to join", Duration::from_secs(60), procs, || {
            if !procs.last().unwrap().log().contains("joined swarm") {
                return false; // the roster may still show a just-killed process as ready
            }
            let (_, r) = http(portal, "GET", "/api/roster", None);
            r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == id && n["state"] == "ready"))
        })
    };

    let (st, inv) = http(portal, "POST", "/api/invite", None);
    assert_eq!(st, 200, "{inv}");
    let code = inv["code"].as_str().unwrap().to_string();
    assert!(inv["command"].as_str().unwrap().ends_with(&format!("--code {code}")), "{inv}");
    let mut m = member("wrong-code", "x", &["--code", "000-000"]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    if code != "000-000" {
        assert!(m.log().contains("wrong code"), "{}", m.log());
    }
    let first = member("right-code", "first", &["--code", &code.replace('-', " ")]);
    joined("first", &[&coord, &first]);
    // Used up.
    let mut m = member("used-code", "second", &["--code", &code]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("already used"), "{}", m.log());
    // Rejoins by key, without a code.
    drop(first);
    let again = member("rejoin", "first", &[]);
    joined("first", &[&coord, &again]);

    // Three wrong tries cancel a code.
    let (_, inv) = http(portal, "POST", "/api/invite", None);
    let code = inv["code"].as_str().unwrap().to_string();
    let wrong = if code == "111-111" { "222-222" } else { "111-111" };
    for _ in 0..3 {
        let mut m = member("wrong-again", "y", &["--code", wrong]);
        assert!(!exits(&mut m, Duration::from_secs(30)));
    }
    let mut m = member("too-late", "third", &["--code", &code]);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("already used"), "{}", m.log());
    drop(again);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Regression (seen on the macOS CI runner): the member ran its benchmark inside its message
/// loop, so it sent no heartbeats meanwhile and the coordinator timed it out after 30 s. The
/// standard benchmark takes about a minute on real devices.
#[test]
fn a_long_benchmark_does_not_stop_heartbeats() {
    let dir = std::env::temp_dir().join(format!("clamor-node-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let models = dir.join("models").display().to_string();
    let (ctrl, portal, data) = (free_port(), free_port(), free_port_run(1));
    let (ctrl_s, portal_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), data.to_string());
    let coord = spawn(&dir, "coordinator", &["coordinator", "--passphrase", "slow-bench", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let log = dir.join("member.log");
    let f = std::fs::File::create(&log).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_clamor-node"))
        .args(["join", &ctrl_s, "--passphrase", "slow-bench", "--id", "m", "--models", &models, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member"), "--participation", "full", "--allow-battery"])
        .env("CLAMOR_TEST_BENCH_DELAY_SECS", "40")
        .stdout(Stdio::from(f.try_clone().unwrap()))
        .stderr(Stdio::from(f))
        .spawn()
        .unwrap();
    let m = Proc(child, "member", log);
    wait_for("the member to finish a 40 s benchmark", Duration::from_secs(150), &[&coord, &m], || {
        let (_, r) = http(portal, "GET", "/api/roster", None);
        r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == "m" && n["state"] == "ready"))
    });
    assert!(!coord.log().contains("heartbeat timeout"), "{}", coord.log());
    let _ = std::fs::remove_dir_all(&dir);
}

/// People, roles and signing in. A member's app reaches the dashboard through its device's
/// local proxy (signed as the device, over TLS): it is that person, with that person's rights.
/// A browser signs in with a code approved by a signed-in person, or a PIN. The owner appoints
/// admins and removes people. Without any of this, the dashboard says "sign in".
#[test]
fn people_roles_and_signing_in() {
    let dir = std::env::temp_dir().join(format!("clamor-node-people-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let models = dir.join("models").display().to_string();
    let (ctrl, portal, tls_port, data, dash) = (free_port(), free_port(), free_port(), free_port_run(1), free_port());
    let (ctrl_s, portal_s, tls_s, data_s, dash_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), format!("127.0.0.1:{tls_port}"), data.to_string(), format!("127.0.0.1:{dash}"));
    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", "home", "--passphrase", "people-test", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--portal-tls", &tls_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let mut m = spawn(&dir, "member", &["join", &ctrl_s, "--passphrase", "people-test", "--name", "Bea", "--dashboard", &dash_s, "--id", "m", "--models", &models, "--no-update-check", "--config-dir", &cfg_dir(&dir, "member"), "--participation", "full", "--allow-battery"]);
    wait_for("the member to join", Duration::from_secs(60), &[&coord, &m], || {
        let (_, r) = http(portal, "GET", "/api/roster", None);
        r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == "m" && n["state"] == "ready"))
    });

    // The member's own dashboard: it is Bea, a member.
    let (st, me) = http(dash, "GET", "/api/me", None);
    assert_eq!((st, me["name"].as_str(), me["role"].as_str(), me["how"].as_str()), (200, Some("Bea"), Some("member"), Some("device")), "{me}");
    let bea = me["id"].as_str().unwrap().to_string();
    let (st, _) = http(dash, "GET", "/api/roster", None);
    assert_eq!(st, 200, "members see the swarm");
    let (st, r) = http(dash, "POST", "/api/muster", None);
    assert_eq!(st, 403, "members cannot run the muster: {r}");
    let own = json!({ "participation": "full", "max_ram_fraction": 0.4, "max_cpu_fraction": 0.5, "require_plugged_in": false });
    assert_eq!(http(dash, "POST", "/api/nodes/m/policy", Some(own.clone())).0, 200, "their own device's settings");
    assert_eq!(http(dash, "POST", "/api/nodes/coord/policy", Some(own)).0, 403, "not someone else's");
    assert_eq!(http(dash, "POST", "/api/invite", Some(json!({ "kind": "person" }))).0, 403, "members cannot invite people");
    let (st, inv) = http(dash, "POST", "/api/invite", Some(json!({ "kind": "device" })));
    assert_eq!(st, 200, "but can add their own devices: {inv}");

    // Bea's own conversation is hers alone; without a model, the reply says why.
    let (_, c) = http(dash, "POST", "/api/chats", None);
    let chat = c["id"].as_str().unwrap().to_string();
    let (st, r) = http(dash, "POST", &format!("/api/chats/{chat}/messages"), Some(json!({ "content": "Hello?" })));
    assert_eq!(st, 200, "{r}");
    assert!(r["message"]["error"].as_str().is_some_and(|e| e.contains("pipeline")), "{r}");
    let (_, list) = http(dash, "GET", "/api/chats", None);
    assert_eq!(list["conversations"][0]["title"], "Hello?", "{list}");
    assert_eq!(http(portal, "GET", &format!("/api/chats/{chat}"), None).0, 404, "not even the owner opens someone's personal chat");
    assert_eq!(http(portal, "DELETE", &format!("/api/chats/{chat}"), None).0, 404);
    assert_eq!(http(dash, "GET", "/api/chats/group", None).0, 200, "the group is everyone's");

    // The owner (the coordinator machine) sees both, and makes Bea an admin.
    let (_, ppl) = http(portal, "GET", "/api/people", None);
    let roles: Vec<(String, String)> = ppl["people"].as_array().unwrap().iter().map(|p| (p["name"].as_str().unwrap().to_string(), p["role"].as_str().unwrap().to_string())).collect();
    assert!(roles.contains(&("Bea".into(), "member".into())) && roles.iter().any(|r| r.1 == "owner"), "{ppl}");
    assert_eq!(http(dash, "POST", &format!("/api/people/{bea}/role"), Some(json!({ "role": "admin" }))).0, 403, "only the owner appoints admins");
    assert_eq!(http(portal, "POST", &format!("/api/people/{bea}/role"), Some(json!({ "role": "admin" }))).0, 200);
    assert_eq!(http(dash, "POST", "/api/muster", None).0, 200, "admins can run the muster");

    // A browser signs in with a code that Bea approves.
    let (_, _, start) = http_c(portal, "POST", "/api/login/start", None, None);
    let (id, code) = (start["id"].as_str().unwrap().to_string(), start["code"].as_str().unwrap().to_string());
    let (_, _, waiting) = http_c(portal, "GET", &format!("/api/login/poll?id={id}"), None, None);
    assert_eq!(waiting["approved"], false);
    assert_eq!(http(dash, "POST", "/api/login/approve", Some(json!({ "code": "000-000" }))).0, 404, "a code nobody shows");
    assert_eq!(http(dash, "POST", "/api/login/approve", Some(json!({ "code": code }))).0, 200);
    let (_, cookie, ok) = http_c(portal, "GET", &format!("/api/login/poll?id={id}"), None, None);
    assert_eq!(ok["approved"], true);
    let cookie = cookie.expect("a session cookie");
    let (_, _, me) = http_c(portal, "GET", "/api/me", None, Some(&cookie));
    assert_eq!((me["name"].as_str(), me["how"].as_str()), (Some("Bea"), Some("session")), "{me}");
    assert_eq!(http_c(portal, "GET", &format!("/api/login/poll?id={id}"), None, None).0, 410, "the code works once");

    // PIN: Bea sets one, signs in with it; wrong PINs lock it.
    assert_eq!(http(dash, "POST", "/api/me/pin", Some(json!({ "pin": "12" }))).0, 400);
    assert_eq!(http(dash, "POST", "/api/me/pin", Some(json!({ "pin": "4821" }))).0, 200);
    let (st, pin_cookie, _) = http_c(portal, "POST", "/api/login/pin", Some(json!({ "name": "bea", "pin": "4821" })), None);
    assert_eq!((st, pin_cookie.is_some()), (200, true));
    for _ in 0..5 {
        assert_eq!(http_c(portal, "POST", "/api/login/pin", Some(json!({ "name": "Bea", "pin": "0000" })), None).0, 401);
    }
    assert_eq!(http_c(portal, "POST", "/api/login/pin", Some(json!({ "name": "Bea", "pin": "4821" })), None).0, 429, "locked after 5 wrong PINs");

    // The owner removes Bea: her device leaves, her sessions end.
    assert_eq!(http(portal, "POST", &format!("/api/people/{bea}/remove"), None).0, 200);
    assert!(exits(&mut m, Duration::from_secs(30)), "the removed person's device leaves\n{}", m.log());
    let (_, _, me) = http_c(portal, "GET", "/api/me", None, Some(&cookie));
    assert_ne!(me["name"].as_str(), Some("Bea"), "her session ended: {me}");
    let (_, ppl) = http(portal, "GET", "/api/people", None);
    assert!(!ppl.to_string().contains("Bea"), "{ppl}");
    drop(coord);

    // From anywhere but the coordinator machine, nobody signed in gets nothing but the page.
    let (ctrl2, portal2, tls2, data2) = (free_port(), free_port(), free_port(), free_port_run(1));
    let log = dir.join("anon.log");
    let f = std::fs::File::create(&log).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_clamor-node"))
        .args(["coordinator", "--passphrase", "x-y-z", "--bind", &format!("127.0.0.1:{ctrl2}"), "--portal", &format!("127.0.0.1:{portal2}"), "--portal-tls", &format!("127.0.0.1:{tls2}"), "--models", &models, "--data-port", &data2.to_string(), "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "anon")])
        .env("CLAMOR_TEST_NO_LOOPBACK_OWNER", "1")
        .stdout(Stdio::from(f.try_clone().unwrap()))
        .stderr(Stdio::from(f))
        .spawn()
        .unwrap();
    let anon = Proc(child, "anon", log);
    wait_for("the portal", Duration::from_secs(120), &[&anon], || TcpStream::connect(("127.0.0.1", portal2)).is_ok());
    assert_eq!(http(portal2, "GET", "/", None).0, 200);
    for (method, path) in [("GET", "/api/roster"), ("POST", "/api/chat"), ("POST", "/api/invite"), ("GET", "/api/people")] {
        assert_eq!(http(portal2, method, path, None).0, 401, "{method} {path}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Asking to join: no code or passphrase. The device shows a check code; an admin sees the same
/// code in the dashboard and approves (or declines). Approved, it is a member like any other.
#[test]
fn asking_to_join_waits_for_an_admin() {
    let dir = std::env::temp_dir().join(format!("clamor-node-ask-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let models = dir.join("models").display().to_string();
    let (ctrl, portal, tls_port, data) = (free_port(), free_port(), free_port(), free_port_run(1));
    let (ctrl_s, portal_s, tls_s, data_s) = (format!("127.0.0.1:{ctrl}"), format!("127.0.0.1:{portal}"), format!("127.0.0.1:{tls_port}"), data.to_string());
    let coord = spawn(&dir, "coordinator", &["coordinator", "--swarm", "home", "--passphrase", "ask-test", "--id", "coord", "--bind", &ctrl_s, "--portal", &portal_s, "--portal-tls", &tls_s, "--models", &models, "--data-port", &data_s, "--quick-bench", "--no-update-check", "--config-dir", &cfg_dir(&dir, "coordinator"), "--participation", "full", "--allow-battery"]);
    wait_for("the portal", Duration::from_secs(120), &[&coord], || TcpStream::connect(("127.0.0.1", portal)).is_ok());
    let cfg = cfg_dir(&dir, "member");
    let ask = |name: &'static str| spawn(&dir, name, &["join", &ctrl_s, "--ask", "--name", "Cal", "--dashboard", "off", "--id", "cal", "--models", &models, "--no-update-check", "--config-dir", &cfg, "--participation", "full", "--allow-battery"]);
    let pending = |m: &Proc| -> (String, String) {
        let mut found = (String::new(), String::new());
        wait_for("the request to join", Duration::from_secs(30), &[&coord, m], || {
            let (_, r) = http(portal, "GET", "/api/joins", None);
            match r["joins"].as_array().and_then(|j| j.first()) {
                Some(j) => {
                    found = (j["id"].as_str().unwrap().to_string(), j["code"].as_str().unwrap().to_string());
                    true
                }
                None => false,
            }
        });
        found
    };
    let shown = |m: &Proc| m.log().split("Check code: ").nth(1).map(|s| s.lines().next().unwrap().trim().to_string());

    // Declined: the device is told, and leaves.
    let mut m = ask("ask-declined");
    let (id, code) = pending(&m);
    assert_eq!(shown(&m).as_deref(), Some(code.as_str()), "both screens show the same check code\n{}", m.log());
    assert_eq!(http(portal, "POST", &format!("/api/joins/{id}/decline"), None).0, 200);
    assert!(!exits(&mut m, Duration::from_secs(30)));
    assert!(m.log().contains("declined"), "{}", m.log());

    // Approved: it joins as Cal, a member.
    let m = ask("ask-approved");
    let (id, code) = pending(&m);
    assert_eq!(shown(&m).as_deref(), Some(code.as_str()));
    assert_eq!(http(portal, "POST", &format!("/api/joins/{id}/approve"), None).0, 200);
    wait_for("Cal to join", Duration::from_secs(60), &[&coord, &m], || {
        let (_, r) = http(portal, "GET", "/api/roster", None);
        m.log().contains("joined swarm") && r["nodes"].as_array().is_some_and(|ns| ns.iter().any(|n| n["id"] == "cal" && n["state"] == "ready"))
    });
    let (_, ppl) = http(portal, "GET", "/api/people", None);
    assert!(ppl["people"].as_array().unwrap().iter().any(|p| p["name"] == "Cal" && p["role"] == "member"), "{ppl}");
    drop(m);
    // And rejoins by its key, without asking again.
    let m = spawn(&dir, "rejoin", &["join", &ctrl_s, "--dashboard", "off", "--id", "cal", "--models", &models, "--no-update-check", "--config-dir", &cfg, "--participation", "full", "--allow-battery"]);
    wait_for("Cal to rejoin", Duration::from_secs(60), &[&coord, &m], || m.log().contains("joined swarm"));
    assert!(!m.log().contains("Check code"), "{}", m.log());
    let _ = std::fs::remove_dir_all(&dir);
}
