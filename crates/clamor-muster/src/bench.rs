//! On-device micro-benchmarks. They measure the two things that decide local-LLM speed:
//! memory read bandwidth (token generation is memory-bound) and floating-point throughput
//! (prompt processing is compute-bound). Run in release mode; debug builds read far too low.

use clamor_proto::{unix_now, BenchReport};
use std::hint::black_box;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct BenchOptions {
    /// Wall time spent on each CPU/memory test.
    pub test_secs: f64,
    pub disk: bool,
    pub disk_mb: usize,
    /// Cap on the memory-bandwidth buffer.
    pub mem_buffer_mb: usize,
}

impl BenchOptions {
    pub fn quick() -> Self {
        BenchOptions { test_secs: 0.4, disk: false, disk_mb: 64, mem_buffer_mb: 128 }
    }
    pub fn standard() -> Self {
        BenchOptions { test_secs: 1.0, disk: true, disk_mb: 256, mem_buffer_mb: 512 }
    }
}

pub fn run(opts: &BenchOptions, ram_available_bytes: u64) -> BenchReport {
    let start = Instant::now();
    let dur = Duration::from_secs_f64(opts.test_secs);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let cpu_1t = cpu_gflops(1, dur);
    let cpu_mt = cpu_gflops(threads, dur);

    // Never let the buffer take more than an eighth of free RAM.
    let cap = (ram_available_bytes / 8 / (1 << 20)) as usize;
    let mb = opts.mem_buffer_mb.min(cap.max(32));
    let mem_bw = mem_bandwidth_gbps(mb, threads, dur);

    let (disk_r, disk_w) = if opts.disk {
        match disk_speed(opts.disk_mb) {
            Some((r, w)) => (Some(r), Some(w)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    BenchReport {
        cpu_gflops_1t: cpu_1t,
        cpu_gflops_mt: cpu_mt,
        mem_bw_gbps: mem_bw,
        disk_read_mbps: disk_r,
        disk_write_mbps: disk_w,
        accel_mem_bw_gbps: None,
        accel_tflops: None,
        took_ms: start.elapsed().as_millis() as u64,
        measured_at_unix: unix_now(),
        build: if cfg!(debug_assertions) { "debug".into() } else { "release".into() },
    }
}

/// f32 matrix-multiply kernel. Returns GFLOP/s summed over `threads` workers.
pub fn cpu_gflops(threads: usize, dur: Duration) -> f64 {
    const N: usize = 160;
    let wall = Instant::now();
    let total_flops: f64 = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(move || {
                    let a = vec![1.0001f32; N * N];
                    let b = vec![0.9999f32; N * N];
                    let mut c = vec![0.0f32; N * N];
                    let t0 = Instant::now();
                    let mut reps = 0u64;
                    while t0.elapsed() < dur {
                        for i in 0..N {
                            for k in 0..N {
                                let aik = a[i * N + k];
                                let brow = &b[k * N..(k + 1) * N];
                                let crow = &mut c[i * N..(i + 1) * N];
                                for j in 0..N {
                                    crow[j] += aik * brow[j];
                                }
                            }
                        }
                        reps += 1;
                    }
                    black_box(&c);
                    reps as f64 * 2.0 * (N * N * N) as f64
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or(0.0)).sum()
    });
    total_flops / wall.elapsed().as_secs_f64() / 1e9
}

/// Aggregate sequential read bandwidth in GB/s.
pub fn mem_bandwidth_gbps(buffer_mb: usize, threads: usize, dur: Duration) -> f64 {
    let words = buffer_mb * (1 << 20) / 8;
    let buf: Vec<u64> = (0..words as u64).collect();
    let chunk = (words / threads.max(1)).max(1);
    let wall = Instant::now();
    let bytes: f64 = std::thread::scope(|s| {
        let handles: Vec<_> = buf
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let t0 = Instant::now();
                    let mut acc = 0u64;
                    let mut passes = 0u64;
                    while t0.elapsed() < dur {
                        acc = acc.wrapping_add(part.iter().fold(0u64, |a, &x| a.wrapping_add(x)));
                        passes += 1;
                    }
                    black_box(acc);
                    passes as f64 * part.len() as f64 * 8.0
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or(0.0)).sum()
    });
    bytes / wall.elapsed().as_secs_f64() / 1e9
}

/// Returns (read MB/s, write MB/s). Read speed may be inflated by the OS page cache.
pub fn disk_speed(size_mb: usize) -> Option<(f64, f64)> {
    let path = std::env::temp_dir().join(format!("clamor-bench-{}.tmp", std::process::id()));
    let block = vec![0xA5u8; 1 << 20];

    let t0 = Instant::now();
    {
        let mut f = std::fs::File::create(&path).ok()?;
        for _ in 0..size_mb {
            f.write_all(&block).ok()?;
        }
        f.sync_all().ok()?;
    }
    let write = size_mb as f64 / t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    {
        let mut f = std::fs::File::open(&path).ok()?;
        let mut buf = vec![0u8; 1 << 20];
        while f.read(&mut buf).ok()? > 0 {}
    }
    let read = size_mb as f64 / t1.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&path);
    Some((read, write))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmarks_return_positive_numbers() {
        let d = Duration::from_millis(100);
        assert!(cpu_gflops(1, d) > 0.0);
        assert!(mem_bandwidth_gbps(16, 2, d) > 0.0);
        assert!(disk_speed(4).is_some());
    }
}
