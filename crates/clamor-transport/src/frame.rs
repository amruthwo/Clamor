//! Binary frame format. Header is 40 bytes, little-endian:
//!
//!   0..4   magic "CLMR"      4 version      5 kind       6 dtype      7 reserved
//!   8..12  epoch             12..20 request id           20..24 pos   24..28 n_tokens
//!   28..32 aux               32..36 payload_len          36..40 flags
//!
//! `aux` is kind-specific: stage index (hello), credits returned (credit), f32 scale bits
//! (int8 activations) or the token id (token).

use bytes::Bytes;
use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const MAGIC: [u8; 4] = *b"CLMR";
pub const WIRE_VERSION: u8 = 1;
pub const HEADER_LEN: usize = 40;
pub const MAX_PAYLOAD: u32 = 128 << 20;
/// Token frame flag: the tail sampled an end-of-sequence token.
pub const FLAG_EOS: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Hello = 1,
    Activations = 2,
    Token = 3,
    Cancel = 4,
    Credit = 5,
    Close = 6,
}

impl Kind {
    fn from_u8(b: u8) -> Option<Kind> {
        Some(match b {
            1 => Kind::Hello,
            2 => Kind::Activations,
            3 => Kind::Token,
            4 => Kind::Cancel,
            5 => Kind::Credit,
            6 => Kind::Close,
            _ => return None,
        })
    }
    /// Data frames consume flow-control credit; control frames never block.
    pub fn is_data(self) -> bool {
        matches!(self, Kind::Activations | Kind::Token)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    None = 0,
    F16 = 1,
    I8 = 2,
}

impl DType {
    fn from_u8(b: u8) -> Option<DType> {
        Some(match b {
            0 => DType::None,
            1 => DType::F16,
            2 => DType::I8,
            _ => return None,
        })
    }
    pub fn bytes_per_value(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::I8 => 1,
            DType::None => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub dtype: DType,
    pub epoch: u32,
    pub request: u64,
    pub pos: u32,
    pub n_tokens: u32,
    pub aux: u32,
    pub payload_len: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub header: Header,
    pub payload: Bytes,
}

/// Hidden-state activations for `n_tokens` tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Acts {
    pub dtype: DType,
    /// Dequantization scale for int8; 1.0 otherwise.
    pub scale: f32,
    pub n_tokens: u32,
    pub data: Bytes,
}

impl Frame {
    fn base(kind: Kind) -> Header {
        Header { kind, dtype: DType::None, epoch: 0, request: 0, pos: 0, n_tokens: 0, aux: 0, payload_len: 0, flags: 0 }
    }

    pub fn ctrl(kind: Kind, epoch: u32, request: u64, aux: u32) -> Frame {
        Frame { header: Header { epoch, request, aux, ..Self::base(kind) }, payload: Bytes::new() }
    }

    pub fn token(epoch: u32, request: u64, pos: u32, id: u32, eos: bool) -> Frame {
        Frame {
            header: Header { epoch, request, pos, n_tokens: 1, aux: id, flags: if eos { FLAG_EOS } else { 0 }, ..Self::base(Kind::Token) },
            payload: Bytes::new(),
        }
    }

    pub fn acts(epoch: u32, request: u64, pos: u32, a: &Acts) -> Frame {
        Frame {
            header: Header {
                epoch,
                request,
                pos,
                n_tokens: a.n_tokens,
                dtype: a.dtype,
                aux: a.scale.to_bits(),
                payload_len: a.data.len() as u32,
                ..Self::base(Kind::Activations)
            },
            payload: a.data.clone(),
        }
    }

    pub fn into_acts(self) -> Acts {
        Acts { dtype: self.header.dtype, scale: f32::from_bits(self.header.aux), n_tokens: self.header.n_tokens, data: self.payload }
    }
}

pub fn encode_header(h: &Header) -> [u8; HEADER_LEN] {
    let mut b = [0u8; HEADER_LEN];
    b[0..4].copy_from_slice(&MAGIC);
    b[4] = WIRE_VERSION;
    b[5] = h.kind as u8;
    b[6] = h.dtype as u8;
    b[8..12].copy_from_slice(&h.epoch.to_le_bytes());
    b[12..20].copy_from_slice(&h.request.to_le_bytes());
    b[20..24].copy_from_slice(&h.pos.to_le_bytes());
    b[24..28].copy_from_slice(&h.n_tokens.to_le_bytes());
    b[28..32].copy_from_slice(&h.aux.to_le_bytes());
    b[32..36].copy_from_slice(&h.payload_len.to_le_bytes());
    b[36..40].copy_from_slice(&h.flags.to_le_bytes());
    b
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

pub fn decode_header(b: &[u8; HEADER_LEN]) -> io::Result<Header> {
    if b[0..4] != MAGIC {
        return Err(bad("bad magic"));
    }
    if b[4] != WIRE_VERSION {
        return Err(bad("unsupported wire version"));
    }
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let payload_len = u32_at(32);
    if payload_len > MAX_PAYLOAD {
        return Err(bad("payload too large"));
    }
    Ok(Header {
        kind: Kind::from_u8(b[5]).ok_or_else(|| bad("unknown kind"))?,
        dtype: DType::from_u8(b[6]).ok_or_else(|| bad("unknown dtype"))?,
        epoch: u32_at(8),
        request: u64::from_le_bytes(b[12..20].try_into().unwrap()),
        pos: u32_at(20),
        n_tokens: u32_at(24),
        aux: u32_at(28),
        payload_len,
        flags: u32_at(36),
    })
}

pub async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, f: &Frame) -> io::Result<()> {
    let mut h = f.header;
    h.payload_len = f.payload.len() as u32;
    w.write_all(&encode_header(&h)).await?;
    if !f.payload.is_empty() {
        w.write_all(&f.payload).await?;
    }
    Ok(())
}

pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Frame> {
    let mut hb = [0u8; HEADER_LEN];
    r.read_exact(&mut hb).await?;
    let header = decode_header(&hb)?;
    let mut body = vec![0u8; header.payload_len as usize];
    r.read_exact(&mut body).await?;
    Ok(Frame { header, payload: Bytes::from(body) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn roundtrip_all_kinds() {
        let a = Acts { dtype: DType::I8, scale: 0.25, n_tokens: 3, data: Bytes::from(vec![1u8, 2, 3, 4]) };
        let frames = vec![
            Frame::ctrl(Kind::Hello, 7, 99, 2),
            Frame::acts(7, 5, 40, &a),
            Frame::token(7, 5, 43, 1234, true),
            Frame::ctrl(Kind::Cancel, 7, 5, 0),
            Frame::ctrl(Kind::Credit, 0, 0, 4),
        ];
        let (mut w, mut r) = tokio::io::duplex(1024);
        for f in &frames {
            write_frame(&mut w, f).await.unwrap();
        }
        for f in &frames {
            let got = read_frame(&mut r).await.unwrap();
            assert_eq!(got.header.kind, f.header.kind);
            assert_eq!(got.header.request, f.header.request);
            assert_eq!(got.header.aux, f.header.aux);
            assert_eq!(got.payload, f.payload);
        }
        let back = Frame::acts(1, 1, 0, &a);
        assert_eq!(back.into_acts(), a);
    }

    #[test]
    fn header_is_40_bytes_and_validated() {
        let h = Frame::token(1, 2, 3, 4, false).header;
        let mut b = encode_header(&h);
        assert_eq!(decode_header(&b).unwrap(), Header { payload_len: 0, ..h });
        b[0] = b'X';
        assert!(decode_header(&b).is_err());
        let mut b = encode_header(&h);
        b[4] = 9;
        assert!(decode_header(&b).is_err());
        let mut b = encode_header(&h);
        b[5] = 77;
        assert!(decode_header(&b).is_err());
        let mut b = encode_header(&h);
        b[32..36].copy_from_slice(&(MAX_PAYLOAD + 1).to_le_bytes());
        assert!(decode_header(&b).is_err());
    }
}
