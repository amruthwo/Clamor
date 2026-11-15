//! Activation codecs. f16 halves traffic versus f32; symmetric per-frame int8 quarters it.
//! The loss is small for hidden states but not zero: planners should treat int8 as an opt-in
//! for slow links, and the engine should validate perplexity before enabling it by default.

use half::f16;

pub fn f32_to_f16(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * 2);
    for &v in x {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

pub fn f16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

/// Returns (int8 bytes, scale) where value ~= byte as i8 * scale.
pub fn quantize_i8(x: &[f32]) -> (Vec<u8>, f32) {
    let max = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    let q = x.iter().map(|v| ((v / scale).round().clamp(-127.0, 127.0) as i8) as u8).collect();
    (q, scale)
}

pub fn dequantize_i8(b: &[u8], scale: f32) -> Vec<f32> {
    b.iter().map(|&v| (v as i8) as f32 * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f32> {
        (0..512).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect()
    }

    #[test]
    fn f16_roundtrip_is_close() {
        let x = sample();
        let y = f16_to_f32(&f32_to_f16(&x));
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() <= a.abs() * 1e-3 + 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn i8_error_is_within_half_a_step() {
        let x = sample();
        let (q, s) = quantize_i8(&x);
        assert_eq!(q.len(), x.len());
        let y = dequantize_i8(&q, s);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() <= s / 2.0 + 1e-6);
        }
    }

    #[test]
    fn zeros_survive() {
        let (q, s) = quantize_i8(&[0.0; 8]);
        assert!(dequantize_i8(&q, s).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sizes_shrink() {
        let x = sample();
        assert_eq!(f32_to_f16(&x).len(), x.len() * 2);
        assert_eq!(quantize_i8(&x).0.len(), x.len());
    }
}
