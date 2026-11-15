use clamor_engine_ggml::*;

#[test]
fn has_a_cpu_device() {
    let d = list_devices();
    assert!(d.iter().any(|x| x.kind == DeviceKind::Cpu), "{d:?}");
}

#[test]
fn matmul_matches_reference_on_every_device() {
    for d in list_devices() {
        let r = matmul_check(d.index).unwrap_or_else(|e| panic!("{}: {e}", d.name));
        // GPUs may compute in reduced precision internally.
        let tol = if d.kind == DeviceKind::Cpu { 1e-4 } else { 5e-2 };
        assert!(r.max_abs_err < tol, "{}: err {}", d.name, r.max_abs_err);
    }
}

#[test]
fn reads_tiny_gguf_if_python_available() {
    let dir = std::env::temp_dir().join("clamor-gguf-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/gen_tiny_llama.py");
    let ok = std::process::Command::new("python3")
        .args([script, path.to_str().unwrap(), "--layers", "3"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: python3 with numpy+gguf not available");
        return;
    }
    let g = GgufFile::open(path.to_str().unwrap()).unwrap();
    assert_eq!(g.get_u32("llama.block_count"), Some(3));
    assert_eq!(g.get_u32("llama.embedding_length"), Some(64));
    assert_eq!(g.get_str("general.architecture").as_deref(), Some("llama"));
    let t = g.tensors();
    assert_eq!(t.len(), 1 + 9 * 3 + 2); // embd + 9 per layer + out_norm + out
    assert!(t.iter().any(|x| x.name == "blk.2.ffn_down.weight"));
}
