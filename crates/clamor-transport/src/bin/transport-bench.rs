//! Loopback pipeline benchmark with the mock engine. Compute is simulated with sleeps, so this
//! measures what the transport adds (framing, flow control, scheduling) and how well several
//! concurrent requests keep all stages busy.

use clap::Parser;
use clamor_transport::mock::{reference_tokens, spawn_mock_ring};
use clamor_transport::stage::Event;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "Benchmark the Clamor ring transport with a mock engine")]
struct Args {
    /// Number of pipeline stages.
    #[arg(long, default_value_t = 3)]
    stages: u32,
    #[arg(long, default_value_t = 60)]
    layers: u32,
    /// Hidden size; sets activation frame size (f16).
    #[arg(long, default_value_t = 8192)]
    hidden: u32,
    /// Simulated compute per layer, microseconds.
    #[arg(long, default_value_t = 40)]
    layer_us: u64,
    #[arg(long, default_value_t = 64)]
    tokens: u32,
    /// Comma-separated numbers of concurrent requests to try.
    #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
    concurrency: Vec<u32>,
    #[arg(long, default_value_t = 8)]
    window: u32,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    let a = Args::parse();
    let per = a.layers / a.stages;
    let mut bounds: Vec<u32> = (0..a.stages).map(|i| i * per).collect();
    bounds.push(a.layers);
    let per_layer = Duration::from_micros(a.layer_us);
    let vocab = 32_000;
    let ring = spawn_mock_ring(&bounds, a.hidden, per_layer, vocab, a.window, 1).await;

    let stage_ms: Vec<f64> = bounds.windows(2).map(|w| (w[1] - w[0]) as f64 * a.layer_us as f64 / 1000.0).collect();
    let single = stage_ms.iter().sum::<f64>();
    let bottleneck = stage_ms.iter().cloned().fold(0.0, f64::max);
    println!("stages {:?} layers, hidden {}, {} us/layer", bounds.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>(), a.hidden, a.layer_us);
    println!("compute per token: {single:.2} ms total, slowest stage {bottleneck:.2} ms");
    println!("ideal: {:.0} tok/s single stream, up to {:.0} tok/s when the ring is full\n", 1000.0 / single, 1000.0 / bottleneck);
    println!("{:>5} {:>10} {:>12} {:>14}", "reqs", "tok/s", "ms/token", "vs ideal");

    // Warm up connections.
    let (_, mut rx) = ring.head().submit(vec![1], 4);
    while let Some(ev) = rx.recv().await {
        if matches!(ev, Event::Done(_)) {
            break;
        }
    }

    for &c in &a.concurrency {
        let t0 = Instant::now();
        let runs: Vec<_> = (0..c).map(|i| (i, ring.head().submit(vec![i + 1, 2, 3], a.tokens))).collect();
        let mut total = 0u64;
        for (i, (_, mut rx)) in runs {
            let mut got = Vec::new();
            while let Some(ev) = rx.recv().await {
                match ev {
                    Event::Token { id, .. } => got.push(id),
                    Event::Done(_) => break,
                }
            }
            assert_eq!(got, reference_tokens(&[i + 1, 2, 3], a.tokens, a.layers, vocab), "pipeline output diverged from reference");
            total += got.len() as u64;
        }
        let secs = t0.elapsed().as_secs_f64();
        let tps = total as f64 / secs;
        // Ideal: c requests overlap until the slowest stage saturates.
        let ideal_tps = (c as f64 * 1000.0 / single).min(1000.0 / bottleneck);
        let ideal_ms = 1000.0 / ideal_tps;
        let per_token_ms = 1000.0 / tps;
        println!("{:>5} {:>10.0} {:>12.2} {:>13.0}%", c, tps, per_token_ms, (per_token_ms / ideal_ms - 1.0) * 100.0);
    }
    ring.stop();
}
