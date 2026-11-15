//! Layer slices: a smaller GGUF with all the metadata but only one stage's tensors. A stage
//! loaded from its slice must compute exactly what it computes from the full file.
//!
//! Needs python3 with numpy + gguf (skips without, unless CLAMOR_REQUIRE_PY=1).
use clamor_engine_ggml::*;
use std::path::{Path, PathBuf};

const TOKENS: [u32; 6] = [1, 5, 9, 23, 42, 7];

fn cpu() -> usize {
    list_devices().iter().find(|d| d.kind == DeviceKind::Cpu).unwrap().index
}

fn tiny(dir: &Path, name: &str, extra: &[&str]) -> Option<String> {
    let path = dir.join(name).display().to_string();
    let tool = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/gen_tiny_llama.py");
    let mut args = vec![tool.to_string(), path.clone(), "--layers".into(), "4".into(), "--bytes-vocab".into()];
    args.extend(extra.iter().map(|s| s.to_string()));
    let ok = std::process::Command::new("python3").args(&args).output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        if std::env::var("CLAMOR_REQUIRE_PY").as_deref() == Ok("1") {
            panic!("python3 with numpy+gguf is required (CLAMOR_REQUIRE_PY=1)");
        }
        eprintln!("skipping: python3 with numpy+gguf not available");
        return None;
    }
    Some(path)
}

fn write_slice(src: &str, layers: std::ops::Range<u32>, embed: bool, out: PathBuf) -> (String, SlicePlan) {
    let g = GgufFile::open(src).unwrap();
    let plan = plan_stage_slice(&g, layers, embed).unwrap();
    let mut f = std::fs::File::create(&out).unwrap();
    plan.write_to(&mut std::fs::File::open(src).unwrap(), &mut f).unwrap();
    (out.display().to_string(), plan)
}

fn tensor_bytes(path: &str, name: &str) -> Vec<u8> {
    let g = GgufFile::open(path).unwrap();
    let off = g.tensor_file_offset(name).unwrap();
    let len = g.tensors().iter().find(|t| t.name == name).unwrap().n_bytes;
    let data = std::fs::read(path).unwrap();
    data[off as usize..(off + len) as usize].to_vec()
}

fn run(stages: &mut [LlamaStage]) -> Vec<f32> {
    let mut hidden: Option<Vec<f32>> = None;
    for s in stages.iter_mut() {
        let input = match &hidden {
            None => LlamaInput::Tokens(&TOKENS),
            Some(h) => LlamaInput::Hidden(h),
        };
        match s.forward(1, 0, input).unwrap() {
            LlamaOutput::Hidden(h) => hidden = Some(h),
            LlamaOutput::Logits(l) => return l,
        }
    }
    panic!("no tail stage")
}

#[test]
fn stages_from_slices_compute_exactly_what_the_full_file_does() {
    let dir = std::env::temp_dir().join(format!("clamor-slice-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // "q8 + align 64": Q8_0 weights and a 32-byte rope_freqs tensor under 64-byte alignment, so
    // tensors need padding between them (the f32 variants happen to be multiples of 32 bytes).
    for (label, extra) in [
        ("llama", vec![]),
        ("qwen2", vec!["--arch", "qwen2", "--rope-freqs"]),
        ("tied", vec!["--tied"]),
        ("q8_align64", vec!["--q8", "--rope-freqs", "--alignment", "64"]),
    ] {
        let Some(full) = tiny(&dir, &format!("{label}.gguf"), &extra) else { return };
        // (layers, embed): an embedding-only head, a middle stage, and a tail.
        let layout = [(0..0, true), (0..2, false), (2..4, false)];
        let o = |embed| StageOptions { device: cpu(), n_ctx: 32, embed: Some(embed), ..Default::default() };
        let mut from_full: Vec<LlamaStage> = layout.iter().map(|(l, e)| LlamaStage::load(&full, l.clone(), o(*e)).unwrap()).collect();
        let want = run(&mut from_full);

        let mut from_slices = Vec::new();
        for (i, (layers, embed)) in layout.iter().enumerate() {
            let (path, plan) = write_slice(&full, layers.clone(), *embed, dir.join(format!("{label}-{i}.gguf")));
            assert_eq!(std::fs::metadata(&path).unwrap().len(), plan.total_len, "{label} stage {i}: planned size");
            let g = GgufFile::open(&path).unwrap_or_else(|e| panic!("{label} stage {i}: slice does not parse: {e}"));
            let src = GgufFile::open(&full).unwrap();
            let hp = LlamaHparams::from_gguf(&src).unwrap();
            let want_names = stage_tensors(&src, &hp, layers.clone(), *embed).unwrap();
            let mut got_names: Vec<String> = g.tensors().into_iter().map(|t| t.name).collect();
            got_names.sort();
            let mut sorted = want_names.clone();
            sorted.sort();
            assert_eq!(got_names, sorted, "{label} stage {i}: exactly this stage's tensors");
            for n in &want_names {
                assert_eq!(tensor_bytes(&path, n), tensor_bytes(&full, n), "{label} stage {i}: {n} bytes");
            }
            assert_eq!(LlamaHparams::from_gguf(&g).unwrap(), hp, "{label} stage {i}: same hyperparameters");
            from_slices.push(LlamaStage::load(&path, layers.clone(), o(*embed)).unwrap());
        }
        let got = run(&mut from_slices);
        assert_eq!(got, want, "{label}: slices must give bit-identical logits");
    }
}

#[test]
fn a_metadata_only_slice_carries_shape_tokenizer_and_template() {
    let dir = std::env::temp_dir().join(format!("clamor-slice-meta-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let Some(full) = tiny(&dir, "m.gguf", &["--arch", "qwen2"]) else { return };
    let (path, plan) = write_slice(&full, 0..0, false, dir.join("meta.gguf"));
    let full_len = std::fs::metadata(&full).unwrap().len();
    assert!(plan.total_len < full_len / 10, "metadata only: {} of {full_len} bytes", plan.total_len);
    let g = GgufFile::open(&path).unwrap();
    assert_eq!(g.n_tensors(), 0);
    let src = GgufFile::open(&full).unwrap();
    assert_eq!(LlamaHparams::from_gguf(&g).unwrap(), LlamaHparams::from_gguf(&src).unwrap());
    let (a, b) = (Tokenizer::from_gguf(&g).unwrap(), Tokenizer::from_gguf(&src).unwrap());
    let text = "<|im_start|>user\nHé 😀<|im_end|>";
    assert_eq!(a.encode(text, true), b.encode(text, true));
    assert_eq!(a.chat_template, b.chat_template);
}

#[test]
fn bad_slice_requests_are_refused() {
    let dir = std::env::temp_dir().join(format!("clamor-slice-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let Some(full) = tiny(&dir, "b.gguf", &[]) else { return };
    let g = GgufFile::open(&full).unwrap();
    assert!(plan_stage_slice(&g, 0..9, false).is_err(), "past the last layer");
    assert!(plan_stage_slice(&g, 2..4, true).is_err(), "embedding without layer 0");
    assert!(g.plan_slice(&["blk.0.attn_q.weight".into(), "blk.0.attn_q.weight".into()], &[]).is_err(), "duplicate tensor");
    assert!(g.plan_slice(&["no.such.tensor".into()], &[]).is_err(), "missing tensor");
}
