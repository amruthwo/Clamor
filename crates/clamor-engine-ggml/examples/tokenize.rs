//! Tokenize text with a GGUF's own tokenizer.
//!
//!     cargo run --release -p clamor-engine-ggml --example tokenize -- MODEL.gguf "some text" [--special]
use clamor_engine_ggml::{GgufFile, Tokenizer};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: tokenize MODEL.gguf TEXT [--special]");
        std::process::exit(2);
    }
    let tok = Tokenizer::from_gguf(&GgufFile::open(&a[1]).unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let ids = tok.encode(&a[2], a.iter().any(|x| x == "--special"));
    println!("{ids:?}");
    eprintln!("{} tokens; decodes to {:?}", ids.len(), tok.decode(&ids, false));
}
