//! `cargo run -p clamor-engine-ggml --example devices [--features vulkan]`
fn main() {
    for d in clamor_engine_ggml::list_devices() {
        println!("[{}] {} — {} ({:?}, {} MiB total, {} MiB free)", d.index, d.name, d.description, d.kind, d.memory_total >> 20, d.memory_free >> 20);
        match clamor_engine_ggml::matmul_check(d.index) {
            Ok(r) => {
                let g = clamor_engine_ggml::matmul_gflops(d.index, 512, 20).unwrap_or(0.0);
                println!("     matmul max_abs_err={:.2e}  ~{:.1} GFLOP/s (512^3)", r.max_abs_err, g);
            }
            Err(e) => println!("     matmul failed: {e}"),
        }
    }
}
