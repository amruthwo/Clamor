//! Cut a layer slice out of a GGUF: all the metadata, only one stage's tensors.
//!
//!     cargo run --release -p clamor-engine-ggml --example slice -- MODEL.gguf START END [--embed] OUT.gguf
use clamor_engine_ggml::{plan_stage_slice, GgufFile};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let embed = a.iter().any(|x| x == "--embed");
    let pos: Vec<&String> = a.iter().skip(1).filter(|x| !x.starts_with("--")).collect();
    if pos.len() != 4 {
        eprintln!("usage: slice MODEL.gguf START END [--embed] OUT.gguf");
        std::process::exit(2);
    }
    let (src, start, end, out) = (pos[0], pos[1].parse().unwrap(), pos[2].parse().unwrap(), pos[3]);
    let plan = plan_stage_slice(&GgufFile::open(src).unwrap(), start..end, embed).unwrap_or_else(|e| panic!("{e}"));
    plan.write_to(&mut std::fs::File::open(src).unwrap(), &mut std::fs::File::create(out).unwrap()).unwrap();
    println!("wrote {out}: {} MB (of {} MB)", plan.total_len >> 20, std::fs::metadata(src).unwrap().len() >> 20);
}
