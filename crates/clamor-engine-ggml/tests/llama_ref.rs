//! The layer-range Llama graph against the numpy reference (tools/ref_forward.py).
//!
//! Needs python3 with numpy + gguf. Without them the tests skip, unless
//! CLAMOR_REQUIRE_PY=1 (set in CI), in which case they fail.
use clamor_engine_ggml::*;
use std::path::PathBuf;
use std::sync::OnceLock;

const TOKENS: [u32; 6] = [1, 5, 9, 23, 42, 7];
const N_LAYER: u32 = 4;
const E: usize = 64;
const V: usize = 128;

struct Fixture {
    model: String,
    /// [layer][token * E]
    hidden: Vec<Vec<f32>>,
    /// [token * V]
    logits: Vec<f32>,
}

fn read_f32(p: PathBuf) -> Vec<f32> {
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn fixture() -> Option<&'static Fixture> {
    static F: OnceLock<Option<Fixture>> = OnceLock::new();
    F.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("clamor-llama-ref-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("tiny.gguf");
        let tools = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools");
        let toks = TOKENS.map(|t| t.to_string()).join(",");
        let run = |args: Vec<String>| {
            std::process::Command::new("python3").args(&args).output().map(|o| {
                if !o.status.success() {
                    eprintln!("{}", String::from_utf8_lossy(&o.stderr));
                }
                o.status.success()
            })
        };
        let ok = run(vec![format!("{tools}/gen_tiny_llama.py"), model.display().to_string(), "--layers".into(), N_LAYER.to_string()]).unwrap_or(false)
            && run(vec![
                format!("{tools}/ref_forward.py"),
                model.display().to_string(),
                dir.join("ref.npz").display().to_string(),
                "--tokens".into(),
                toks,
                "--raw".into(),
                dir.display().to_string(),
            ])
            .unwrap_or(false);
        if !ok {
            if std::env::var("CLAMOR_REQUIRE_PY").as_deref() == Ok("1") {
                panic!("python3 with numpy+gguf is required (CLAMOR_REQUIRE_PY=1)");
            }
            eprintln!("skipping: python3 with numpy+gguf not available");
            return None;
        }
        let flat = read_f32(dir.join("hidden.f32"));
        let per = TOKENS.len() * E;
        assert_eq!(flat.len(), per * N_LAYER as usize);
        let hidden = flat.chunks_exact(per).map(|c| c.to_vec()).collect();
        let logits = read_f32(dir.join("logits.f32"));
        assert_eq!(logits.len(), TOKENS.len() * V);
        Some(Fixture { model: model.display().to_string(), hidden, logits })
    })
    .as_ref()
}

/// Max error scaled so large logits are not held to a tighter relative bound than small ones.
fn max_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter().zip(want).map(|(g, w)| (g - w).abs() / w.abs().max(1.0)).fold(0.0, f32::max)
}

fn tol(d: &DeviceInfo) -> f32 {
    // GPUs may accumulate in reduced precision.
    if d.kind == DeviceKind::Cpu {
        1e-4
    } else {
        2e-2
    }
}

/// Device order depends on which backends are compiled in (with Vulkan, CPU is not index 0).
fn cpu() -> usize {
    list_devices().iter().find(|d| d.kind == DeviceKind::Cpu).expect("a CPU device").index
}

/// Devices that can run the Llama graph. A device that reports `Unsupported` (e.g. the
/// paravirtual GPU on hosted macOS runners, which lacks simdgroup reduction) is skipped with a
/// note; any other error fails the test. The CPU must always be supported.
fn usable_devices(f: &Fixture) -> Vec<DeviceInfo> {
    list_devices()
        .into_iter()
        .filter(|d| {
            let mut s = stage(f, 0..N_LAYER, d.index);
            match s.forward(0, 0, LlamaInput::Tokens(&TOKENS[..1])) {
                Ok(_) => true,
                Err(Error::Unsupported { device, op }) if d.kind != DeviceKind::Cpu => {
                    eprintln!("NOTE: skipping {device}: does not support {op}");
                    false
                }
                Err(e) => panic!("{}: {e}", d.name),
            }
        })
        .collect()
}

fn stage(f: &Fixture, layers: std::ops::Range<u32>, device: usize) -> LlamaStage {
    LlamaStage::load(&f.model, layers, StageOptions { device, n_ctx: 32, ..Default::default() }).unwrap()
}

fn hidden(o: LlamaOutput) -> Vec<f32> {
    match o {
        LlamaOutput::Hidden(h) => h,
        LlamaOutput::Logits(_) => panic!("expected hidden states, got logits"),
    }
}
fn logits(o: LlamaOutput) -> Vec<f32> {
    match o {
        LlamaOutput::Logits(l) => l,
        LlamaOutput::Hidden(_) => panic!("expected logits, got hidden states"),
    }
}

fn rows<'a>(x: &'a [f32], width: usize, r: std::ops::Range<usize>) -> &'a [f32] {
    &x[r.start * width..r.end * width]
}

#[test]
fn hparams_from_gguf() {
    let Some(f) = fixture() else { return };
    let s = stage(f, 0..N_LAYER, cpu());
    let hp = s.hparams();
    assert_eq!((hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_ff, hp.n_vocab), (4, 64, 4, 2, 128, 128));
    assert!(s.is_head() && s.is_tail());
}

#[test]
fn full_model_matches_reference_per_layer() {
    let Some(f) = fixture() else { return };
    for d in usable_devices(f) {
        let mut s = stage(f, 0..N_LAYER, d.index);
        let (out, traced) = s.forward_traced(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap();
        assert_eq!(traced.len(), N_LAYER as usize);
        for (l, h) in traced.iter().enumerate() {
            let err = max_err(h, &f.hidden[l]);
            assert!(err < tol(&d), "{}: layer {l} err {err}", d.name);
        }
        let err = max_err(&logits(out), &f.logits);
        eprintln!("{}: per-layer max err {:?}, logits {err:.2e}", d.name, traced.iter().enumerate().map(|(l, h)| format!("{:.1e}", max_err(h, &f.hidden[l]))).collect::<Vec<_>>());
        assert!(err < tol(&d), "{}: logits err {err}", d.name);
    }
}

#[test]
fn split_equals_full() {
    let Some(f) = fixture() else { return };
    for d in usable_devices(f) {
        let mut full = stage(f, 0..N_LAYER, d.index);
        let want = logits(full.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());

        let mut a = stage(f, 0..2, d.index);
        let mut b = stage(f, 2..N_LAYER, d.index);
        assert!(a.is_head() && !a.is_tail() && !b.is_head() && b.is_tail());
        let mid = hidden(a.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
        assert!(max_err(&mid, &f.hidden[1]) < tol(&d), "{}: stage-a boundary", d.name);
        let got = logits(b.forward(1, 0, LlamaInput::Hidden(&mid)).unwrap());
        let err = max_err(&got, &want);
        assert!(err < tol(&d), "{}: split vs full err {err}", d.name);

        // Three stages, including a middle one that neither embeds nor samples.
        let mut s0 = stage(f, 0..1, d.index);
        let mut s1 = stage(f, 1..3, d.index);
        let mut s2 = stage(f, 3..N_LAYER, d.index);
        assert!(!s1.is_head() && !s1.is_tail());
        let h0 = hidden(s0.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
        let h1 = hidden(s1.forward(1, 0, LlamaInput::Hidden(&h0)).unwrap());
        let got = logits(s2.forward(1, 0, LlamaInput::Hidden(&h1)).unwrap());
        assert!(max_err(&got, &f.logits) < tol(&d), "{}: 3-stage", d.name);
    }
}

/// Private chats: the asker's device runs the ends (head 0..1 and tail 3..4) and a stage that
/// has loaded the middle of the model (here all of it, as a normal ring's full stage would)
/// runs only layers 1..3 for that ring. Same logits as the full model, prefill and decode, while
/// the same loaded stage keeps serving its own ring's requests in between.
#[test]
fn a_loaded_stage_can_run_part_of_its_layers_for_another_ring() {
    let Some(f) = fixture() else { return };
    for d in usable_devices(f) {
        let mut shared = stage(f, 0..N_LAYER, d.index); // the normal ring's stage, all layers
        let mut head = stage(f, 0..1, d.index); // the asker's device
        let mut tail = stage(f, 3..N_LAYER, d.index);
        // A request of the normal ring, interleaved below; ids must not clash with the part's.
        let own = shared.forward(1, 0, LlamaInput::Tokens(&TOKENS[..2])).unwrap();
        assert!(matches!(own, LlamaOutput::Logits(_)), "the shared stage still ends its own ring");
        let private = 1u64 << 48 | 1;
        // Prefill three tokens, then decode the rest one at a time.
        let h = hidden(head.forward(1, 0, LlamaInput::Tokens(&TOKENS[..3])).unwrap());
        let m = shared.forward_part(private, 0, &h, 1..3).unwrap();
        assert_eq!(m.len(), h.len(), "hidden states in, hidden states out");
        let got = logits(tail.forward(1, 0, LlamaInput::Hidden(&m)).unwrap());
        assert!(max_err(&got, rows(&f.logits, V, 0..3)) < tol(&d), "{}: part prefill", d.name);
        for p in 3..TOKENS.len() {
            // The normal ring moves on too.
            if p == 3 {
                let own = logits(shared.forward(1, 2, LlamaInput::Tokens(&TOKENS[2..3])).unwrap());
                assert!(max_err(&own, rows(&f.logits, V, 2..3)) < tol(&d), "{}: own ring unaffected", d.name);
            }
            let h = hidden(head.forward(1, p as u32, LlamaInput::Tokens(&TOKENS[p..p + 1])).unwrap());
            let m = shared.forward_part(private, p as u32, &h, 1..3).unwrap();
            let got = logits(tail.forward(1, p as u32, LlamaInput::Hidden(&m)).unwrap());
            let err = max_err(&got, rows(&f.logits, V, p..p + 1));
            assert!(err < tol(&d), "{}: part decode pos {p} err {err}", d.name);
        }
        assert!(shared.forward_part(9, 0, &h, 3..5).is_err(), "outside the stage's layers");
        assert!(shared.forward_part(9, 0, &h, 2..2).is_err(), "empty");
        shared.release(private);
        assert_eq!(shared.n_past(private), 0);
    }
}

/// Prefill part of the prompt, then decode one token at a time through the KV cache: every
/// step must match the reference's full-prompt row for that position.
#[test]
fn kv_cache_decode_matches_prefill() {
    let Some(f) = fixture() else { return };
    for d in usable_devices(f) {
        // Single stage.
        let mut s = stage(f, 0..N_LAYER, d.index);
        let got = logits(s.forward(7, 0, LlamaInput::Tokens(&TOKENS[..3])).unwrap());
        assert!(max_err(&got, rows(&f.logits, V, 0..3)) < tol(&d));
        for p in 3..TOKENS.len() {
            let got = logits(s.forward(7, p as u32, LlamaInput::Tokens(&TOKENS[p..p + 1])).unwrap());
            let err = max_err(&got, rows(&f.logits, V, p..p + 1));
            assert!(err < tol(&d), "{}: decode pos {p} err {err}", d.name);
        }
        assert_eq!(s.n_past(7), TOKENS.len() as u32);

        // Split, decoding one token at a time from the start.
        let mut a = stage(f, 0..2, d.index);
        let mut b = stage(f, 2..N_LAYER, d.index);
        for p in 0..TOKENS.len() {
            let h = hidden(a.forward(7, p as u32, LlamaInput::Tokens(&TOKENS[p..p + 1])).unwrap());
            let got = logits(b.forward(7, p as u32, LlamaInput::Hidden(&h)).unwrap());
            let err = max_err(&got, rows(&f.logits, V, p..p + 1));
            assert!(err < tol(&d), "{}: split decode pos {p} err {err}", d.name);
        }
    }
}

#[test]
fn requests_have_independent_caches() {
    let Some(f) = fixture() else { return };
    let mut s = stage(f, 0..N_LAYER, cpu());
    // Interleave two requests; a different prefix in request 2 must not leak into request 1.
    s.forward(2, 0, LlamaInput::Tokens(&[99, 98, 97])).unwrap();
    s.forward(1, 0, LlamaInput::Tokens(&TOKENS[..2])).unwrap();
    s.forward(2, 3, LlamaInput::Tokens(&[96])).unwrap();
    for p in 2..TOKENS.len() {
        let got = logits(s.forward(1, p as u32, LlamaInput::Tokens(&TOKENS[p..p + 1])).unwrap());
        assert!(max_err(&got, rows(&f.logits, V, p..p + 1)) < 1e-4, "pos {p}");
    }
}

#[test]
fn release_frees_and_allows_restart() {
    let Some(f) = fixture() else { return };
    let mut s = stage(f, 0..N_LAYER, cpu());
    s.forward(5, 0, LlamaInput::Tokens(&[3, 4, 5])).unwrap();
    assert_eq!(s.n_past(5), 3);
    s.release(5);
    s.release(5); // unknown ids are fine
    s.release(12345);
    assert_eq!(s.n_past(5), 0);
    let got = logits(s.forward(5, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
    assert!(max_err(&got, &f.logits) < 1e-4);
}

#[test]
fn rejects_bad_inputs() {
    let Some(f) = fixture() else { return };
    let opts = || StageOptions { device: cpu(), n_ctx: 8, ..Default::default() };
    assert!(LlamaStage::load(&f.model, 2..2, opts()).is_err(), "empty range past the start");
    assert!(LlamaStage::load(&f.model, 0..5, opts()).is_err(), "past last layer");

    let mut head = LlamaStage::load(&f.model, 0..2, opts()).unwrap();
    let mut tail = LlamaStage::load(&f.model, 2..4, opts()).unwrap();
    assert!(head.forward(1, 0, LlamaInput::Hidden(&[0.0; E])).is_err(), "head takes tokens");
    assert!(tail.forward(1, 0, LlamaInput::Tokens(&[1])).is_err(), "tail takes hidden");
    assert!(tail.forward(1, 0, LlamaInput::Hidden(&[0.0; E - 1])).is_err(), "ragged hidden");
    assert!(head.forward(1, 0, LlamaInput::Tokens(&[])).is_err(), "empty batch");
    assert!(head.forward(1, 0, LlamaInput::Tokens(&[V as u32])).is_err(), "token out of vocab");
    assert!(head.forward(1, 1, LlamaInput::Tokens(&[1])).is_err(), "position gap");
    head.forward(1, 0, LlamaInput::Tokens(&[1, 2, 3, 4, 5, 6])).unwrap();
    assert!(head.forward(1, 3, LlamaInput::Tokens(&[1])).is_err(), "position rewind");
    assert!(head.forward(1, 6, LlamaInput::Tokens(&[1, 2, 3])).is_err(), "context overflow");
    head.forward(1, 6, LlamaInput::Tokens(&[1, 2])).unwrap(); // exactly fills n_ctx = 8
}

/// Regression: lm_head tied to token_embd (no output.weight). The tail must load token_embd
/// even when it is not the head, and a single stage must not load it twice.
#[test]
fn tied_embeddings() {
    if fixture().is_none() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("clamor-llama-tied-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("tied.gguf").display().to_string();
    let tools = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools");
    let py = |args: &[&str]| assert!(std::process::Command::new("python3").args(args).status().unwrap().success());
    py(&[&format!("{tools}/gen_tiny_llama.py"), &model, "--layers", "4", "--tied"]);
    let raw = dir.display().to_string();
    let npz = dir.join("ref.npz").display().to_string();
    let toks = TOKENS.map(|t| t.to_string()).join(",");
    py(&[&format!("{tools}/ref_forward.py"), &model, &npz, "--tokens", &toks, "--raw", &raw]);
    let want = read_f32(dir.join("logits.f32"));
    assert!(GgufFile::open(&model).unwrap().tensor_ne("output.weight").is_none());

    let opts = || StageOptions { device: cpu(), n_ctx: 32, ..Default::default() };
    let mut full = LlamaStage::load(&model, 0..4, opts()).unwrap();
    assert!(max_err(&logits(full.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap()), &want) < 1e-4);
    let mut a = LlamaStage::load(&model, 0..2, opts()).unwrap();
    let mut b = LlamaStage::load(&model, 2..4, opts()).unwrap();
    let h = hidden(a.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
    assert!(max_err(&logits(b.forward(1, 0, LlamaInput::Hidden(&h)).unwrap()), &want) < 1e-4);
}

/// An embedding-only head (layers 0..0) feeds a stage that holds every layer but does not
/// embed (`embed: Some(false)`): together they must equal the reference. This is the layout the
/// coordinator uses when it hosts the head without holding layers.
#[test]
fn embedding_only_head_composes() {
    let Some(f) = fixture() else { return };
    let o = |embed| StageOptions { device: cpu(), n_ctx: 32, embed, ..Default::default() };
    let mut head = LlamaStage::load(&f.model, 0..0, o(None)).unwrap();
    assert!(head.is_head() && !head.is_tail());
    let mut rest = LlamaStage::load(&f.model, 0..N_LAYER, o(Some(false))).unwrap();
    assert!(!rest.is_head() && rest.is_tail());

    let h = hidden(head.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
    assert_eq!(h.len(), TOKENS.len() * E);
    let (out, traced) = rest.forward_traced(1, 0, LlamaInput::Hidden(&h)).unwrap();
    for (l, t) in traced.iter().enumerate() {
        assert!(max_err(t, &f.hidden[l]) < 1e-4, "layer {l}");
    }
    assert!(max_err(&logits(out), &f.logits) < 1e-4);

    // The head tracks positions with no KV cache, and still enforces order.
    assert_eq!(head.n_past(1), TOKENS.len() as u32);
    assert!(head.forward(1, 0, LlamaInput::Tokens(&[3])).is_err(), "position rewind");
    head.release(1);
    assert_eq!(head.live_requests(), 0);

    // Roles that cannot work are refused.
    assert!(LlamaStage::load(&f.model, 0..0, o(Some(false))).is_err(), "no layers and no embedding");
    assert!(LlamaStage::load(&f.model, 2..4, o(Some(true))).is_err(), "embedding needs layer 0");
}

/// Qwen2 (q/k/v biases, NEOX RoPE) and Llama 3 (rope_freqs scaling), alone and together:
/// every layer, a two-stage split, and token-by-token decode through the KV cache must all
/// match the numpy reference.
#[test]
fn architecture_variants_match_reference() {
    if fixture().is_none() {
        return;
    }
    let tools = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools");
    let toks = TOKENS.map(|t| t.to_string()).join(",");
    for (label, extra) in [("qwen2", vec!["--arch", "qwen2"]), ("llama3 rope_freqs", vec!["--rope-freqs"]), ("qwen2 + rope_freqs", vec!["--arch", "qwen2", "--rope-freqs"])] {
        let dir = std::env::temp_dir().join(format!("clamor-arch-{}-{}", std::process::id(), label.replace([' ', '+'], "_")));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.gguf").display().to_string();
        let mut gen_args = vec![format!("{tools}/gen_tiny_llama.py"), model.clone(), "--layers".into(), N_LAYER.to_string()];
        gen_args.extend(extra.iter().map(|s| s.to_string()));
        let py = |args: &[String]| assert!(std::process::Command::new("python3").args(args).status().unwrap().success());
        py(&gen_args);
        py(&[format!("{tools}/ref_forward.py"), model.clone(), dir.join("r.npz").display().to_string(), "--tokens".into(), toks.clone(), "--raw".into(), dir.display().to_string()]);
        let ref_hidden = read_f32(dir.join("hidden.f32"));
        let want = read_f32(dir.join("logits.f32"));
        let per = TOKENS.len() * E;
        let o = || StageOptions { device: cpu(), n_ctx: 32, ..Default::default() };

        let mut full = LlamaStage::load(&model, 0..N_LAYER, o()).unwrap();
        assert_eq!(full.hparams().rope_neox, label.starts_with("qwen2"), "{label}");
        let (out, traced) = full.forward_traced(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap();
        for (l, h) in traced.iter().enumerate() {
            let err = max_err(h, &ref_hidden[l * per..(l + 1) * per]);
            assert!(err < 1e-4, "{label}: layer {l} err {err}");
        }
        assert!(max_err(&logits(out), &want) < 1e-4, "{label}: logits");

        // Split: the second stage needs the biases and rope_freqs too.
        let mut a = LlamaStage::load(&model, 0..2, o()).unwrap();
        let mut b = LlamaStage::load(&model, 2..N_LAYER, o()).unwrap();
        let h = hidden(a.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
        assert!(max_err(&logits(b.forward(1, 0, LlamaInput::Hidden(&h)).unwrap()), &want) < 1e-4, "{label}: split");

        // Decode one token at a time: positions must reach RoPE correctly.
        let mut s = LlamaStage::load(&model, 0..N_LAYER, o()).unwrap();
        for p in 0..TOKENS.len() {
            let got = logits(s.forward(9, p as u32, LlamaInput::Tokens(&TOKENS[p..p + 1])).unwrap());
            assert!(max_err(&got, rows(&want, V, p..p + 1)) < 1e-4, "{label}: decode pos {p}");
        }
    }
}

/// Prompts longer than `prefill_chunk` go through in chunks (bounding working memory); the
/// result must match the reference, through a split, and decoding must carry on from it.
#[test]
fn long_prompts_are_processed_in_chunks() {
    let Some(f) = fixture() else { return };
    let o = |embed: Option<bool>| StageOptions { device: cpu(), n_ctx: 32, embed, prefill_chunk: 2, ..Default::default() };
    // One stage, all logits and last-only.
    let mut s = LlamaStage::load(&f.model, 0..N_LAYER, o(None)).unwrap();
    assert!(max_err(&logits(s.forward_logits(1, 0, LlamaInput::Tokens(&TOKENS), Logits::All).unwrap()), &f.logits) < 1e-4);
    let mut s2 = LlamaStage::load(&f.model, 0..N_LAYER, o(None)).unwrap();
    let last = logits(s2.forward_logits(1, 0, LlamaInput::Tokens(&TOKENS), Logits::Last).unwrap());
    assert!(max_err(&last, rows(&f.logits, V, TOKENS.len() - 1..TOKENS.len())) < 1e-4);
    assert_eq!(s2.n_past(1), TOKENS.len() as u32, "every chunk landed in the KV cache");
    // A split: the hidden states of every chunk are passed on.
    let mut a = LlamaStage::load(&f.model, 0..2, o(None)).unwrap();
    let mut b = LlamaStage::load(&f.model, 2..N_LAYER, o(None)).unwrap();
    let h = hidden(a.forward(1, 0, LlamaInput::Tokens(&TOKENS)).unwrap());
    assert_eq!(h.len(), TOKENS.len() * E);
    assert!(max_err(&logits(b.forward(1, 0, LlamaInput::Hidden(&h)).unwrap()), &f.logits) < 1e-4, "split + chunks");
    // Decode continues from a chunked prefill exactly as from a single pass.
    let mut one = stage(f, 0..N_LAYER, cpu());
    one.forward(2, 0, LlamaInput::Tokens(&TOKENS[..5])).unwrap();
    let want = logits(one.forward(2, 5, LlamaInput::Tokens(&TOKENS[5..])).unwrap());
    let mut ch = LlamaStage::load(&f.model, 0..N_LAYER, o(None)).unwrap();
    ch.forward(2, 0, LlamaInput::Tokens(&TOKENS[..5])).unwrap();
    let got = logits(ch.forward(2, 5, LlamaInput::Tokens(&TOKENS[5..])).unwrap());
    assert!(max_err(&got, &want) < 1e-5);
}
