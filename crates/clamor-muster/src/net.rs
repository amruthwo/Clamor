//! Network benchmark between two nodes: latency distribution and bulk throughput in both
//! directions. A tiny raw TCP protocol (no JSON) so the numbers reflect the link, not the codec.
//!
//!   'P' + 8 bytes            -> echoes the 8 bytes            (ping)
//!   'U' + u64 n + n bytes    -> replies with 1 byte           (upload test)
//!   'D' + u64 n              -> sends n bytes                 (download test)

use clamor_proto::LinkStat;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

const CHUNK: usize = 256 * 1024;

/// Run the benchmark server until the listener fails. One thread per connection.
pub fn serve(listener: TcpListener) -> io::Result<()> {
    for conn in listener.incoming() {
        let stream = conn?;
        std::thread::spawn(move || {
            let _ = handle(stream);
        });
    }
    Ok(())
}

fn handle(mut s: TcpStream) -> io::Result<()> {
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let mut op = [0u8; 1];
        if s.read_exact(&mut op).is_err() {
            return Ok(());
        }
        match op[0] {
            b'P' => {
                let mut b = [0u8; 8];
                s.read_exact(&mut b)?;
                s.write_all(&b)?;
            }
            b'U' => {
                let mut n = read_u64(&mut s)?;
                while n > 0 {
                    let take = (n as usize).min(CHUNK);
                    s.read_exact(&mut buf[..take])?;
                    n -= take as u64;
                }
                s.write_all(&[1])?;
            }
            b'D' => {
                let mut n = read_u64(&mut s)?;
                while n > 0 {
                    let take = (n as usize).min(CHUNK);
                    s.write_all(&buf[..take])?;
                    n -= take as u64;
                }
            }
            _ => return Ok(()),
        }
    }
}

fn read_u64(s: &mut TcpStream) -> io::Result<u64> {
    let mut b = [0u8; 8];
    s.read_exact(&mut b)?;
    Ok(u64::from_be_bytes(b))
}

/// Measure the link to a benchmark server. `bulk_bytes == 0` skips the throughput tests.
pub fn measure(addr: impl ToSocketAddrs, pings: u32, bulk_bytes: u64) -> io::Result<LinkStat> {
    let mut s = TcpStream::connect(addr)?;
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(Duration::from_secs(30)))?;

    let mut rtts = Vec::with_capacity(pings as usize);
    for i in 0..pings {
        let t0 = Instant::now();
        s.write_all(b"P")?;
        s.write_all(&(i as u64).to_be_bytes())?;
        let mut echo = [0u8; 8];
        s.read_exact(&mut echo)?;
        rtts.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let (mut up, mut down) = (None, None);
    if bulk_bytes > 0 {
        let chunk = vec![0x5Au8; CHUNK];

        let t0 = Instant::now();
        s.write_all(b"U")?;
        s.write_all(&bulk_bytes.to_be_bytes())?;
        let mut left = bulk_bytes;
        while left > 0 {
            let take = (left as usize).min(CHUNK);
            s.write_all(&chunk[..take])?;
            left -= take as u64;
        }
        let mut ack = [0u8; 1];
        s.read_exact(&mut ack)?;
        up = Some(mbps(bulk_bytes, t0.elapsed()));

        let t1 = Instant::now();
        s.write_all(b"D")?;
        s.write_all(&bulk_bytes.to_be_bytes())?;
        let mut buf = vec![0u8; CHUNK];
        let mut left = bulk_bytes;
        while left > 0 {
            let take = (left as usize).min(CHUNK);
            s.read_exact(&mut buf[..take])?;
            left -= take as u64;
        }
        down = Some(mbps(bulk_bytes, t1.elapsed()));
    }

    Ok(LinkStat {
        rtt_ms_p50: percentile(&rtts, 0.50),
        rtt_ms_p95: percentile(&rtts, 0.95),
        up_mbps: up,
        down_mbps: down,
        samples: pings,
    })
}

fn mbps(bytes: u64, d: Duration) -> f64 {
    bytes as f64 * 8.0 / d.as_secs_f64() / 1e6
}

pub fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basics() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 1.0), 5.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn loopback_measure() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let _ = serve(listener);
        });
        let stat = measure(addr, 20, 4 << 20).unwrap();
        assert_eq!(stat.samples, 20);
        assert!(stat.rtt_ms_p50 >= 0.0);
        assert!(stat.up_mbps.unwrap() > 0.0);
        assert!(stat.down_mbps.unwrap() > 0.0);
    }
}
