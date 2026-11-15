//! Async framing helpers for the control plane (same format as `clamor_proto::write_frame`).

use clamor_proto::{decode, encode, frame_len, Message};
use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub async fn read_msg<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Message> {
    let mut header = [0u8; 4];
    r.read_exact(&mut header).await?;
    let mut body = vec![0u8; frame_len(header)?];
    r.read_exact(&mut body).await?;
    decode(&body)
}

pub async fn write_msg<W: AsyncWrite + Unpin>(w: &mut W, msg: &Message) -> io::Result<()> {
    w.write_all(&encode(msg)?).await?;
    w.flush().await
}
