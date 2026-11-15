//! One connection between two adjacent stages.
//!
//! Handshake: the connecting side sends `Hello` (epoch, session, its stage index, and the
//! receive window it grants); the accepting side checks epoch and session, then answers with
//! its own `Hello`. A mismatch closes the link, so a stage that missed a re-plan can never
//! splice into the new pipeline.
//!
//! Flow control: the sender starts with as many credits as the peer's window and spends one per
//! data frame; the receiver returns one credit each time the application actually consumes a
//! frame. A slow stage therefore stalls its upstream within a window's worth of frames, instead
//! of letting memory balloon. Control frames (cancel, credit, close) never wait for credit.

use crate::frame::{read_frame, write_frame, Frame, Kind};
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Semaphore};

#[derive(Clone, Copy, Debug)]
pub struct LinkParams {
    pub epoch: u32,
    pub session: u64,
    /// This side's stage index (informational, shown to the peer).
    pub stage: u32,
    /// Data frames this side is willing to buffer for the peer.
    pub window: u32,
}

pub struct Link {
    out: mpsc::UnboundedSender<Frame>,
    credits: Arc<Semaphore>,
    inbound: mpsc::Receiver<Frame>,
    pub peer_stage: u32,
}

fn err(kind: io::ErrorKind, msg: impl Into<String>) -> io::Error {
    io::Error::new(kind, msg.into())
}

/// No Nagle delay, and TCP keepalive: a neighbour that vanishes without closing the connection
/// (power loss, Wi-Fi gone) is noticed in about 25 s instead of whenever the OS gets round to it.
fn tune(stream: &TcpStream) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let ka = socket2::TcpKeepalive::new().with_time(Duration::from_secs(10)).with_interval(Duration::from_secs(5));
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    let ka = ka.with_retries(3);
    socket2::SockRef::from(stream).set_tcp_keepalive(&ka)
}

impl Link {
    /// Connect to `addr`, retrying until `timeout` (neighbours start in any order).
    pub async fn connect(addr: SocketAddr, p: LinkParams, timeout: Duration) -> io::Result<Link> {
        let deadline = Instant::now() + timeout;
        let mut stream = loop {
            match TcpStream::connect(addr).await {
                Ok(s) => break s,
                Err(e) if Instant::now() >= deadline => return Err(e),
                Err(_) => tokio::time::sleep(Duration::from_millis(40)).await,
            }
        };
        tune(&stream)?;
        write_frame(&mut stream, &Frame::ctrl(Kind::Hello, p.epoch, p.session, p.stage).with_window(p.window)).await?;
        stream.flush().await?;
        let ack = tokio::time::timeout(timeout, read_frame(&mut stream)).await.map_err(|_| err(io::ErrorKind::TimedOut, "no hello ack"))??;
        if ack.header.kind != Kind::Hello {
            return Err(err(io::ErrorKind::InvalidData, "peer refused the link (epoch or session mismatch)"));
        }
        Ok(spawn_io(stream, p.window, ack.header.n_tokens, ack.header.aux))
    }

    /// Accept one upstream connection and verify its epoch and session.
    pub async fn accept(listener: &TcpListener, p: LinkParams) -> io::Result<Link> {
        loop {
            let (mut stream, _) = listener.accept().await?;
            tune(&stream)?;
            let hello = match tokio::time::timeout(Duration::from_secs(10), read_frame(&mut stream)).await {
                Ok(Ok(f)) if f.header.kind == Kind::Hello => f,
                _ => continue, // stray connection; keep waiting for the real neighbour
            };
            if hello.header.epoch != p.epoch || hello.header.request != p.session {
                let _ = write_frame(&mut stream, &Frame::ctrl(Kind::Close, p.epoch, p.session, p.stage)).await;
                let _ = stream.flush().await;
                return Err(err(io::ErrorKind::PermissionDenied, format!("hello from epoch {} session {} rejected (expected {} / {})", hello.header.epoch, hello.header.request, p.epoch, p.session)));
            }
            write_frame(&mut stream, &Frame::ctrl(Kind::Hello, p.epoch, p.session, p.stage).with_window(p.window)).await?;
            stream.flush().await?;
            return Ok(spawn_io(stream, p.window, hello.header.n_tokens, hello.header.aux));
        }
    }

    /// Send a data frame, waiting for credit if the peer's window is full.
    pub async fn send_data(&self, f: Frame) -> io::Result<()> {
        debug_assert!(f.header.kind.is_data());
        let permit = self.credits.acquire().await.map_err(|_| err(io::ErrorKind::BrokenPipe, "link closed"))?;
        permit.forget();
        self.out.send(f).map_err(|_| err(io::ErrorKind::BrokenPipe, "link closed"))
    }

    /// Send a control frame immediately (never blocks on credit).
    pub fn send_ctrl(&self, f: Frame) {
        let _ = self.out.send(f);
    }

    /// Next frame from the peer. Returns the credit as a side effect for data frames.
    pub async fn recv(&mut self) -> Option<Frame> {
        let f = self.inbound.recv().await?;
        if f.header.kind.is_data() {
            let _ = self.out.send(Frame::ctrl(Kind::Credit, 0, 0, 1));
        }
        Some(f)
    }
}

impl Frame {
    /// Hello frames advertise the receive window in `n_tokens`.
    fn with_window(mut self, window: u32) -> Frame {
        self.header.n_tokens = window;
        self
    }
}

fn spawn_io(stream: TcpStream, my_window: u32, peer_window: u32, peer_stage: u32) -> Link {
    let (rd, wr) = stream.into_split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Frame>();
    let credits = Arc::new(Semaphore::new(peer_window as usize));
    let (in_tx, in_rx) = mpsc::channel::<Frame>(my_window as usize + 16);

    tokio::spawn(async move {
        let mut w = BufWriter::new(wr);
        while let Some(f) = out_rx.recv().await {
            if write_frame(&mut w, &f).await.is_err() {
                return;
            }
            // Coalesce whatever is already queued into one flush.
            while let Ok(f) = out_rx.try_recv() {
                if write_frame(&mut w, &f).await.is_err() {
                    return;
                }
            }
            if w.flush().await.is_err() {
                return;
            }
        }
        let _ = w.flush().await;
    });

    let c2 = credits.clone();
    tokio::spawn(async move {
        let mut r = BufReader::new(rd);
        while let Ok(f) = read_frame(&mut r).await {
            match f.header.kind {
                Kind::Credit => c2.add_permits(f.header.aux as usize),
                Kind::Close => break,
                _ => {
                    if in_tx.send(f).await.is_err() {
                        break;
                    }
                }
            }
        }
        c2.close(); // wake any sender waiting for credit
    });

    Link { out: out_tx, credits, inbound: in_rx, peer_stage }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Acts, DType};
    use bytes::Bytes;

    fn acts(n: u32) -> Frame {
        Frame::acts(1, 1, n, &Acts { dtype: DType::F16, scale: 1.0, n_tokens: 1, data: Bytes::from(vec![0u8; 64]) })
    }

    async fn pair(window: u32) -> (Link, Link) {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let p = LinkParams { epoch: 1, session: 42, stage: 0, window };
        let q = LinkParams { stage: 1, ..p };
        let (a, b) = tokio::join!(Link::connect(addr, p, Duration::from_secs(2)), Link::accept(&l, q));
        (a.unwrap(), b.unwrap())
    }

    #[tokio::test]
    async fn frames_flow_and_stages_are_exchanged() {
        let (a, mut b) = pair(4).await;
        assert_eq!(a.peer_stage, 1);
        assert_eq!(b.peer_stage, 0);
        a.send_data(acts(5)).await.unwrap();
        let f = b.recv().await.unwrap();
        assert_eq!(f.header.pos, 5);
    }

    #[tokio::test]
    async fn credit_window_applies_backpressure() {
        let (a, mut b) = pair(2).await;
        a.send_data(acts(1)).await.unwrap();
        a.send_data(acts(2)).await.unwrap();
        // Window is full: the third send must wait until the receiver consumes a frame.
        let blocked = tokio::time::timeout(Duration::from_millis(150), a.send_data(acts(3))).await;
        assert!(blocked.is_err(), "third send should block");
        b.recv().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), a.send_data(acts(4))).await.expect("credit should free the sender").unwrap();
    }

    #[tokio::test]
    async fn control_frames_bypass_credit() {
        let (a, mut b) = pair(1).await;
        a.send_data(acts(1)).await.unwrap();
        a.send_ctrl(Frame::ctrl(Kind::Cancel, 1, 77, 0));
        let first = b.recv().await.unwrap();
        let second = b.recv().await.unwrap();
        assert_eq!(first.header.kind, Kind::Activations);
        assert_eq!(second.header.kind, Kind::Cancel);
    }

    #[tokio::test]
    async fn epoch_mismatch_is_rejected_on_both_sides() {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let mine = LinkParams { epoch: 2, session: 42, stage: 1, window: 4 };
        let stale = LinkParams { epoch: 1, session: 42, stage: 0, window: 4 };
        let (c, a) = tokio::join!(Link::connect(addr, stale, Duration::from_secs(2)), Link::accept(&l, mine));
        assert!(c.is_err());
        assert!(a.is_err());
    }

    #[tokio::test]
    async fn sender_unblocks_with_error_when_peer_dies() {
        let (a, b) = pair(1).await;
        a.send_data(acts(1)).await.unwrap();
        drop(b);
        let r = tokio::time::timeout(Duration::from_secs(3), a.send_data(acts(2))).await.expect("must not hang");
        assert!(r.is_err());
    }
}
