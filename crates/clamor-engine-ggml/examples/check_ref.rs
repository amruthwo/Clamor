//! Compare the engine with `tools/ref_forward.py --raw DIR` on any Llama GGUF.
//!
//!     python3 tools/ref_forward.py MODEL.gguf /tmp/r.npz --tokens 1,2,3 --generate 8 --raw /tmp/ref
//!     cargo run --release -p clamor-engine-ggml --example check_ref -- MODEL.gguf /tmp/ref 1,2,3 [device]
//!
//! Prints per-layer and logit error (relative to max(|ref|, 1)) and checks greedy generation.
use clamor_engine_ggml::*;

fn read_f32(p: std::path::PathBuf) -> Vec<f32> {
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn max_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(g, w)| (g - w).abs() / w.abs().max(1.0)).fold(0.0, f32::max)
}

/// ||got - want|| / ||want||: scale-free, so a few huge activations don't dominate.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut d, mut w) = (0f64, 0f64);
    for (g, x) in got.iter().zip(want) {
        d += ((g - x) as f64).powi(2);
        w += (*x as f64).powi(2);
    }
    (d / w.max(1e-30)).sqrt()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: check_ref MODEL.gguf REF_DIR TOKENS [DEVICE_INDEX|-] [THREADS]");
        std::process::exit(2);
    }
    let (model, dir) = (&a[1], std::path::PathBuf::from(&a[2]));
    let toks: Vec<u32> = a[3].split(',').map(|t| t.trim().parse().unwrap()).collect();
    let device = a.get(4).filter(|d| d.as_str() != "-").map(|d| d.parse().unwrap()).unwrap_or_else(|| list_devices().iter().find(|d| d.kind == DeviceKind::Cpu).unwrap().index);
    let n_threads = a.get(5).map(|t| t.parse().unwrap()).unwrap_or(0);
    let dev_name = list_devices()[device].name.clone();

    let t0 = std::time::Instant::now();
    let mut s = LlamaStage::load(model, 0..u32::MAX.min(GgufFile::open(model).unwrap().get_u32("llama.block_count").unwrap()), StageOptions { device, n_ctx: 512, n_threads, ..Default::default() }).unwrap();
    println!("{dev_name}: loaded {} layers in {:.2?}", s.hparams().n_layer, t0.elapsed());
    let (e, v, n) = (s.hparams().n_embd as usize, s.hparams().n_vocab as usize, toks.len());

    let hidden = read_f32(dir.join("hidden.f32"));
    let logits = read_f32(dir.join("logits.f32"));
    let (out, traced) = s.forward_traced(1, 0, LlamaInput::Tokens(&toks)).unwrap();
    let errs: Vec<f32> = traced.iter().enumerate().map(|(l, h)| max_err(h, &hidden[l * n * e..(l + 1) * n * e])).collect();
    let worst = errs.iter().cloned().fold(0.0, f32::max);
    let rels: Vec<f64> = traced.iter().enumerate().map(|(l, h)| rel_l2(h, &hidden[l * n * e..(l + 1) * n * e])).collect();
    let worst_rel = rels.iter().cloned().fold(0.0, f64::max);
    println!("hidden-state max element error: first layer {:.2e}, last {:.2e}, worst {worst:.2e}", errs[0], errs[errs.len() - 1]);
    println!("hidden-state relative L2 error: first layer {:.2e}, last {:.2e}, worst {worst_rel:.2e}", rels[0], rels[rels.len() - 1]);
    let LlamaOutput::Logits(l) = out else { unreachable!() };
    println!("logit error: max element {:.2e}, relative L2 {:.2e}", max_err(&l, &logits), rel_l2(&l, &logits));
    let argmax = |x: &[f32]| x.iter().enumerate().fold((0, f32::MIN), |b, (i, &y)| if y > b.1 { (i, y) } else { b }).0 as u32;
    let agree = (0..n).filter(|&i| argmax(&l[i * v..(i + 1) * v]) == argmax(&logits[i * v..(i + 1) * v])).count();
    println!("top-1 agreement over prompt positions: {agree}/{n}");

    if let Ok(g) = std::fs::read_to_string(dir.join("generated.txt")) {
        let want: Vec<u32> = g.split(',').filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
        let mut got = Vec::new();
        let (mut pos, mut next) = (n as u32, argmax(&l[(n - 1) * v..]));
        let t1 = std::time::Instant::now();
        for _ in 0..want.len() {
            got.push(next);
            let LlamaOutput::Logits(l) = s.forward_logits(1, pos, LlamaInput::Tokens(&[next]), Logits::Last).unwrap() else { unreachable!() };
            pos += 1;
            next = argmax(&l);
        }
        let secs = t1.elapsed().as_secs_f64();
        println!("greedy: engine {got:?}\n        numpy  {want:?}  -> {}", if got == want { "MATCH" } else { "DIFFERENT" });
        println!("decode: {:.1} tok/s on {dev_name}", want.len() as f64 / secs);
    }
}
