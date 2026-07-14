//! Decoders for the GGUF/GGML block-quantized formats.
//!
//! Every format here packs a fixed run of elements ([`DType::block_size`],
//! 32 for all five formats we support) into a fixed number of bytes
//! ([`DType::block_bytes`]) sharing one `f16` scale (and, for the `_1`
//! variants, one `f16` minimum). The block layouts below were read from the
//! reference C structs and dequantize loops in
//! `crates/kopitiam-ai/vendor/ggml/src/ggml-common.h` and `ggml-quants.c`
//! (MIT licensed) to get the byte offsets and nibble ordering right, then
//! reimplemented from scratch here — no C was transliterated.
//!
//! # Block layouts
//!
//! All fields are little-endian, matching GGUF's on-disk byte order.
//!
//! * **Q4_0** (18 bytes = 2 + 16): `f16 d` (scale), then 16 bytes of packed
//!   4-bit nibbles for 32 elements. Byte `j` holds elements `j` (low
//!   nibble) and `j + 16` (high nibble) — *not* elements `2j`/`2j+1` — so
//!   the first half of the block lives in the low nibbles and the second
//!   half in the high nibbles. Each nibble is a signed magnitude in
//!   `[0, 15]` biased by 8, i.e. decode as `(nibble - 8) * d`.
//! * **Q4_1** (20 bytes = 2 + 2 + 16): `f16 d`, `f16 m` (min), then the same
//!   16-byte nibble packing as Q4_0. Asymmetric: decode as
//!   `nibble * d + m` (no `-8` bias, since `m` carries the offset).
//! * **Q5_0** (22 bytes = 2 + 4 + 16): `f16 d`, a 32-bit `qh` field holding
//!   the 5th (high) bit of every element, then the same 16-byte nibble
//!   packing as Q4_0. Element `j`'s high bit is `qh` bit `j`; element
//!   `j + 16`'s high bit is `qh` bit `j + 16` — read out as
//!   `(qh >> (j + 12)) & 0x10`, which looks like it should select bit
//!   `j + 12` but does not: masking bit 4 (`0x10`) of a value already
//!   shifted right by `j + 12` selects original bit `(j + 12) + 4 = j + 16`.
//!   Getting this off by 4 (i.e. mis-deriving it as bit `j + 12`) was the
//!   one real bug this module shipped with — caught by the round-trip
//!   tests below, not by inspection. Decode as
//!   `((nibble | high_bit) - 16) * d`.
//! * **Q5_1** (24 bytes = 2 + 2 + 4 + 16): `f16 d`, `f16 m`, the same `qh`
//!   field as Q5_0, then nibbles. Asymmetric like Q4_1: decode as
//!   `(nibble | high_bit) * d + m`.
//! * **Q8_0** (34 bytes = 2 + 32): `f16 d`, then 32 signed `i8` values.
//!   Decode as `qs[j] * d`. The simplest format — no packing, just a scale.
//!
//! # Encoding and fused dot products
//!
//! This module also has the other half of the quantized story:
//! [`quantize_row_q8_0`] (encoding, but *only* to Q8_0, and *only* for
//! activations — see its docs for why that is a narrower scope than "a
//! general f32 -> quantized encoder") and [`q4_0_dot_q8_0`] /
//! [`q8_0_dot_q8_0`] (fused integer dot products between one quantized
//! weight block and one quantized activation block). Together these are
//! what [`crate::tensor::Tensor::quantized_matmul`] is built from — see
//! that method's docs for the algorithm they compose into.

use kopitiam_core::{DType, Error, Result};

use crate::half::f16_to_f32;

/// Decodes every block in `bytes` (assumed to already be validated as a
/// whole number of `dtype`-sized blocks, per [`crate::storage::Storage::new_quantized`])
/// into a flat `f32` vector in the tensor's original element order.
pub(crate) fn dequantize(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>> {
    let block_bytes = dtype.block_bytes();
    let block_size = dtype.block_size();
    let num_blocks = bytes.len() / block_bytes;
    let mut out = Vec::with_capacity(num_blocks * block_size);

    let decode_block: fn(&[u8], &mut Vec<f32>) = match dtype {
        DType::Q4_0 => dequant_q4_0,
        DType::Q4_1 => dequant_q4_1,
        DType::Q5_0 => dequant_q5_0,
        DType::Q5_1 => dequant_q5_1,
        DType::Q8_0 => dequant_q8_0,
        _ => return Err(Error::UnsupportedDType { op: "dequantize", dtype }),
    };

    for block in bytes.chunks_exact(block_bytes) {
        decode_block(block, &mut out);
    }
    Ok(out)
}

fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn dequant_q4_0(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let qs = &block[2..18];
    let mut values = [0f32; 32];
    for (j, &byte) in qs.iter().enumerate() {
        values[j] = (i32::from(byte & 0x0F) - 8) as f32 * d;
        values[j + 16] = (i32::from(byte >> 4) - 8) as f32 * d;
    }
    out.extend_from_slice(&values);
}

fn dequant_q4_1(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    let qs = &block[4..20];
    let mut values = [0f32; 32];
    for (j, &byte) in qs.iter().enumerate() {
        values[j] = f32::from(byte & 0x0F) * d + m;
        values[j + 16] = f32::from(byte >> 4) * d + m;
    }
    out.extend_from_slice(&values);
}

fn dequant_q5_0(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    let mut values = [0f32; 32];
    for (j, &byte) in qs.iter().enumerate() {
        let high_0 = (((qh >> j) << 4) & 0x10) as u8;
        let high_1 = ((qh >> (j + 12)) & 0x10) as u8;
        values[j] = (i32::from((byte & 0x0F) | high_0) - 16) as f32 * d;
        values[j + 16] = (i32::from((byte >> 4) | high_1) - 16) as f32 * d;
    }
    out.extend_from_slice(&values);
}

fn dequant_q5_1(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let m = read_f16(block, 2);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let qs = &block[8..24];
    let mut values = [0f32; 32];
    for (j, &byte) in qs.iter().enumerate() {
        let high_0 = (((qh >> j) << 4) & 0x10) as u8;
        let high_1 = ((qh >> (j + 12)) & 0x10) as u8;
        values[j] = f32::from((byte & 0x0F) | high_0) * d + m;
        values[j + 16] = f32::from((byte >> 4) | high_1) * d + m;
    }
    out.extend_from_slice(&values);
}

fn dequant_q8_0(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let qs = &block[2..34];
    out.extend(qs.iter().map(|&b| f32::from(b as i8) * d));
}

/// Quantizes one row of `f32` activations to Q8_0 blocks: for every
/// 32-element block, `d = max(|x|) / 127`, and each element becomes
/// `round(x / d)` clamped to `[-127, 127]` — the inverse of
/// [`dequant_q8_0`]'s `qs[j] * d`.
///
/// # Why this exists, and why it is *not* a general encoder
///
/// [`crate::tensor::Tensor::to_dtype`]'s docs say plainly that converting
/// `f32` to a quantized dtype is out of scope for this crate: "it requires
/// choosing a quantization *scheme* ... not a forward-pass inference
/// concern". This function does not contradict that — it is not a public
/// `Tensor -> Tensor` conversion at all, and it only ever quantizes
/// *activations*, never weights. An activation is different from a weight
/// in exactly the way that matters here: it is quantized fresh on every
/// single forward pass (there is no "calibration" or "export step" to get
/// right), so there is only one honest scheme to choose — per-block
/// symmetric, matching whatever scheme produced the weight it is about to
/// be dotted against. That is a narrow, mechanical decision belonging to
/// [`crate::tensor::Tensor::quantized_matmul`]'s implementation, not a
/// general-purpose feature.
///
/// `x.len()` must be a whole multiple of 32; the sole caller
/// ([`crate::tensor::Tensor::quantized_matmul`]) only reaches this after
/// already checking that its `in_features` is.
///
/// Returns `(block_scales, quantized_values)`: one `f32` scale per block
/// (`0.0` for an all-zero block, matching `id = 0` rather than dividing by
/// zero), and `x.len()` signed bytes in the same block order
/// [`dequant_q8_0`] expects.
pub(crate) fn quantize_row_q8_0(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    debug_assert!(x.len().is_multiple_of(32), "quantize_row_q8_0 requires a whole number of 32-element blocks");
    let num_blocks = x.len() / 32;
    let mut scales = Vec::with_capacity(num_blocks);
    let mut q = vec![0i8; x.len()];
    for (b, chunk) in x.chunks_exact(32).enumerate() {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (j, &v) in chunk.iter().enumerate() {
            q[b * 32 + j] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
        scales.push(d);
    }
    (scales, q)
}

/// Integer dot product of one Q4_0 weight block against one already
/// Q8_0-quantized activation block (`x_q`, exactly 32 values, sharing one
/// `x_scale`), without ever materializing either side as `f32`.
///
/// This is [`dequant_q4_0`]'s block layout, read directly: nibble `j`
/// decodes to weight `(nibble_j - 8) * d_w`, so
/// `sum_j weight[j] * activation[j]` factors into
/// `d_w * x_scale * sum_j (nibble_j - 8) * x_q[j]` — one `i32` accumulator
/// over 32 small integer multiply-adds, then exactly one `f32` multiply at
/// the very end. That deferral (scaling *after* the accumulation, not
/// per-element) is what makes this "matmul in integer space" a genuinely
/// different algorithm from "dequantize then multiply" rather than merely
/// a reordering of it: the 32 products and their sum use only `i32`
/// arithmetic (exact — no rounding at all, for values this small), so the
/// only floating-point rounding in this whole block is that unavoidable
/// final scale multiply.
///
/// `weight_block` must be exactly [`DType::Q4_0`]'s
/// [`DType::block_bytes`] (18) long; `x_q` exactly 32 elements.
pub(crate) fn q4_0_dot_q8_0(weight_block: &[u8], x_q: &[i8], x_scale: f32) -> f32 {
    debug_assert_eq!(weight_block.len(), DType::Q4_0.block_bytes());
    debug_assert_eq!(x_q.len(), 32);
    let d_w = read_f16(weight_block, 0);
    let qs = &weight_block[2..18];
    let mut acc: i32 = 0;
    for (j, &byte) in qs.iter().enumerate() {
        let w_lo = i32::from(byte & 0x0F) - 8;
        let w_hi = i32::from(byte >> 4) - 8;
        acc += w_lo * i32::from(x_q[j]);
        acc += w_hi * i32::from(x_q[j + 16]);
    }
    acc as f32 * d_w * x_scale
}

/// Integer dot product of one Q8_0 weight block against one already
/// Q8_0-quantized activation block — the `_1`-free companion to
/// [`q4_0_dot_q8_0`].
///
/// Simpler than the Q4_0 case: both sides are already plain signed bytes,
/// so there is no nibble unpacking or zero-bias offset to undo — just an
/// `i32` accumulator over 32 `i8 * i8` products, scaled once at the end by
/// both sides' block scales.
///
/// `weight_block` must be exactly [`DType::Q8_0`]'s
/// [`DType::block_bytes`] (34) long; `x_q` exactly 32 elements.
pub(crate) fn q8_0_dot_q8_0(weight_block: &[u8], x_q: &[i8], x_scale: f32) -> f32 {
    debug_assert_eq!(weight_block.len(), DType::Q8_0.block_bytes());
    debug_assert_eq!(x_q.len(), 32);
    let d_w = read_f16(weight_block, 0);
    let qs = &weight_block[2..34];
    let mut acc: i32 = 0;
    for (j, &byte) in qs.iter().enumerate() {
        acc += i32::from(byte as i8) * i32::from(x_q[j]);
    }
    acc as f32 * d_w * x_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a Q4_0 block by hand: `d` as its f16 bit pattern, plus 32
    /// values already known to fit `[-8, 7]`.
    fn q4_0_block(d_bits: u16, values: [i8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for j in 0..16 {
            let low = (values[j] + 8) as u8 & 0x0F;
            let high = (values[j + 16] + 8) as u8 & 0x0F;
            block[2 + j] = low | (high << 4);
        }
        block
    }

    #[test]
    fn q4_0_decodes_known_nibbles_with_the_i_and_i_plus_16_byte_sharing_rule() {
        // d = 2.0 (f16 0x4000, exact). Values sweep the full [-8, 7] range
        // so every nibble pattern 0x0..0xF is exercised at least once.
        let mut values = [0i8; 32];
        for (j, v) in values.iter_mut().enumerate() {
            *v = ((j % 16) as i8) - 8;
        }
        let block = q4_0_block(0x4000, values);
        let mut out = Vec::new();
        dequant_q4_0(&block, &mut out);
        assert_eq!(out.len(), 32);
        for j in 0..32 {
            assert_eq!(out[j], f32::from(values[j]) * 2.0, "mismatch at index {j}");
        }
        // Directly confirm the "byte j holds elements j and j+16" rule:
        // byte 0 packs values[0] (low nibble) and values[16] (high nibble).
        let byte0 = block[2];
        assert_eq!((byte0 & 0x0F) as i32 - 8, i32::from(values[0]));
        assert_eq!((byte0 >> 4) as i32 - 8, i32::from(values[16]));
    }

    #[test]
    fn q4_0_all_zero_block_decodes_to_all_zero() {
        let block = q4_0_block(0x4000, [0i8; 32]); // nibble 8 (bias) = value 0
        let mut out = Vec::new();
        dequant_q4_0(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    fn q4_1_block(d_bits: u16, m_bits: u16, values: [u8; 32]) -> Vec<u8> {
        // Q4_1 is unsigned/asymmetric: values are the raw 4-bit nibble
        // (0..15), decoded as nibble*d + m.
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block[2..4].copy_from_slice(&m_bits.to_le_bytes());
        for j in 0..16 {
            block[4 + j] = (values[j] & 0x0F) | ((values[j + 16] & 0x0F) << 4);
        }
        block
    }

    #[test]
    fn q4_1_decodes_with_scale_and_min() {
        // d = 1.0 (0x3C00), m = 10.0 (0x4900, exact power-of-two-friendly value).
        let mut values = [0u8; 32];
        for (j, v) in values.iter_mut().enumerate() {
            *v = (j % 16) as u8;
        }
        let block = q4_1_block(0x3C00, 0x4900, values);
        let mut out = Vec::new();
        dequant_q4_1(&block, &mut out);
        for j in 0..32 {
            let expected = f32::from(values[j]) * 1.0 + 10.0;
            assert_eq!(out[j], expected, "mismatch at index {j}");
        }
    }

    fn q5_0_block(d_bits: u16, values: [i8; 32]) -> Vec<u8> {
        // values in [-16, 15].
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        let mut qh: u32 = 0;
        for j in 0..16 {
            let raw_low = (values[j] + 16) as u8; // 5-bit unsigned
            let raw_high = (values[j + 16] + 16) as u8;
            if raw_low & 0x10 != 0 {
                qh |= 1 << j;
            }
            if raw_high & 0x10 != 0 {
                // Element `j + 16`'s 5th bit lives at qh bit `j + 16`, not
                // `j + 12`: the dequant formula reads it as
                // `(qh >> (j+12)) & 0x10`, and masking bit 4 (0x10) of a
                // value already shifted right by `j+12` selects original
                // bit `(j+12)+4 = j+16` — see the format-layout doc comment
                // at the top of this module.
                qh |= 1 << (j + 16);
            }
        }
        block[2..6].copy_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            let raw_low = ((values[j] + 16) as u8) & 0x0F;
            let raw_high = ((values[j + 16] + 16) as u8) & 0x0F;
            block[6 + j] = raw_low | (raw_high << 4);
        }
        block
    }

    #[test]
    fn q5_0_decodes_the_5th_bit_from_the_qh_field() {
        // d = 1.0. Sweep the full 5-bit signed range [-16, 15], which
        // requires every value of the high bit to be exercised.
        let mut values = [0i8; 32];
        for (j, v) in values.iter_mut().enumerate() {
            *v = ((j % 32) as i8) - 16;
        }
        let block = q5_0_block(0x3C00, values);
        let mut out = Vec::new();
        dequant_q5_0(&block, &mut out);
        for j in 0..32 {
            assert_eq!(out[j], f32::from(values[j]), "mismatch at index {j}");
        }
    }

    fn q5_1_block(d_bits: u16, m_bits: u16, values: [u8; 32]) -> Vec<u8> {
        // values in [0, 31] (unsigned 5-bit).
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block[2..4].copy_from_slice(&m_bits.to_le_bytes());
        let mut qh: u32 = 0;
        for j in 0..16 {
            if values[j] & 0x10 != 0 {
                qh |= 1 << j;
            }
            if values[j + 16] & 0x10 != 0 {
                qh |= 1 << (j + 16); // see the Q5_0 fixture comment above.
            }
        }
        block[4..8].copy_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            block[8 + j] = (values[j] & 0x0F) | ((values[j + 16] & 0x0F) << 4);
        }
        block
    }

    #[test]
    fn q5_1_decodes_with_scale_min_and_5th_bit() {
        let mut values = [0u8; 32];
        for (j, v) in values.iter_mut().enumerate() {
            *v = (j % 32) as u8;
        }
        let block = q5_1_block(0x3C00, 0x3800, values); // d=1.0, m=0.5
        let mut out = Vec::new();
        dequant_q5_1(&block, &mut out);
        for j in 0..32 {
            let expected = f32::from(values[j]) * 1.0 + 0.5;
            assert_eq!(out[j], expected, "mismatch at index {j}");
        }
    }

    fn q8_0_block(d_bits: u16, values: [i8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for (j, &v) in values.iter().enumerate() {
            block[2 + j] = v as u8;
        }
        block
    }

    #[test]
    fn q8_0_decodes_signed_bytes_scaled() {
        let mut values = [0i8; 32];
        for (j, v) in values.iter_mut().enumerate() {
            *v = (j as i8) - 16; // sweep [-16, 15]
        }
        let block = q8_0_block(0x4000, values); // d = 2.0
        let mut out = Vec::new();
        dequant_q8_0(&block, &mut out);
        for j in 0..32 {
            assert_eq!(out[j], f32::from(values[j]) * 2.0, "mismatch at index {j}");
        }
    }

    #[test]
    fn q8_0_extreme_values_do_not_overflow_i8() {
        // -128 and 127 are the i8 extremes; a `u8 as i8` cast must not panic
        // or wrap incorrectly.
        let block = q8_0_block(0x3C00, [i8::MIN, i8::MAX].repeat(16).try_into().unwrap());
        let mut out = Vec::new();
        dequant_q8_0(&block, &mut out);
        assert_eq!(out[0], f32::from(i8::MIN));
        assert_eq!(out[1], f32::from(i8::MAX));
    }

    #[test]
    fn dequantize_dispatches_by_dtype_and_handles_multiple_blocks() {
        let block_a = q8_0_block(0x3C00, [1i8; 32]); // d=1.0
        let block_b = q8_0_block(0x4000, [2i8; 32]); // d=2.0
        let mut bytes = block_a;
        bytes.extend(block_b);
        let out = dequantize(DType::Q8_0, &bytes).unwrap();
        assert_eq!(out.len(), 64);
        assert!(out[0..32].iter().all(|&v| v == 1.0));
        assert!(out[32..64].iter().all(|&v| v == 4.0));
    }

    // -- quantize_row_q8_0 / q4_0_dot_q8_0 / q8_0_dot_q8_0 --

    #[test]
    fn quantize_row_q8_0_round_trips_within_one_quantization_step() {
        let x: Vec<f32> = (0..32).map(|j| (j as f32 - 16.0) * 0.37).collect();
        let (scales, q) = quantize_row_q8_0(&x);
        assert_eq!(scales.len(), 1);
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        assert_eq!(scales[0], d);
        for (j, &orig) in x.iter().enumerate() {
            let decoded = f32::from(q[j]) * d;
            // Nearest-multiple-of-d rounding can be off by at most d/2.
            assert!((decoded - orig).abs() <= d / 2.0 + 1e-6, "index {j}: {decoded} vs {orig} (d={d})");
        }
    }

    #[test]
    fn quantize_row_q8_0_all_zero_block_has_zero_scale_and_zero_values() {
        let (scales, q) = quantize_row_q8_0(&[0.0f32; 32]);
        assert_eq!(scales, vec![0.0]);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn quantize_row_q8_0_handles_multiple_blocks_independently() {
        let mut x = vec![1.0f32; 32];
        x.extend(vec![100.0f32; 32]);
        let (scales, q) = quantize_row_q8_0(&x);
        assert_eq!(scales.len(), 2);
        // Block 0's max magnitude is 1.0, so its every value quantizes to
        // the top of the i8 range; block 1's scale is a hundred times
        // larger. A shared scale (the classic "one scale for the whole
        // row" bug) would make these equal.
        assert_ne!(scales[0], scales[1]);
        assert_eq!(q[0], 127);
        assert_eq!(q[32], 127);
    }

    #[test]
    fn quantize_row_q8_0_extreme_magnitude_does_not_overflow_i8() {
        let x = [f32::MAX, -f32::MAX].repeat(16);
        let (_scales, q) = quantize_row_q8_0(&x);
        assert!(q.iter().all(|&v| v == 127 || v == -127));
    }

    /// The correctness gate this whole module's "dot in integer space"
    /// claim rests on: for a Q4_0 block whose values are exact multiples
    /// of its scale (so *decoding* introduces zero error), the fused
    /// integer dot product must equal, bit-for-bit up to ordinary `f32`
    /// summation slop, an independently computed dequantize-then-multiply
    /// dot product over the same two rows. Any bug in the nibble bias, the
    /// `j`/`j+16` split, or the accumulation order would show up here.
    #[test]
    fn q4_0_dot_q8_0_matches_dequantize_then_dot() {
        // d_w = 1.0 (0x3C00, exact): weight values are exactly the signed
        // nibble range [-8, 7].
        let mut w_values = [0i8; 32];
        for (j, v) in w_values.iter_mut().enumerate() {
            *v = ((j % 16) as i8) - 8;
        }
        let w_block = q4_0_block(0x3C00, w_values);

        // A Q8_0 activation block with a non-trivial scale.
        let x: Vec<f32> = (0..32).map(|j| ((j as f32) - 16.0) * 0.5).collect();
        let (x_scales, x_q) = quantize_row_q8_0(&x);

        let fused = q4_0_dot_q8_0(&w_block, &x_q, x_scales[0]);

        let mut w_decoded = Vec::new();
        dequant_q4_0(&w_block, &mut w_decoded);
        let x_decoded: Vec<f32> = x_q.iter().map(|&q| f32::from(q) * x_scales[0]).collect();
        let reference: f32 = w_decoded.iter().zip(&x_decoded).map(|(a, b)| a * b).sum();

        assert!((fused - reference).abs() < 1e-3, "fused={fused}, reference={reference}");
    }

    #[test]
    fn q8_0_dot_q8_0_matches_dequantize_then_dot() {
        let mut w_values = [0i8; 32];
        for (j, v) in w_values.iter_mut().enumerate() {
            *v = ((j as i32 * 7) % 256 - 128) as i8;
        }
        let w_block = q8_0_block(0x3800, w_values); // d_w = 0.5

        let x: Vec<f32> = (0..32).map(|j| (j as f32 - 16.0) * 0.5).collect();
        let (x_scales, x_q) = quantize_row_q8_0(&x);

        let fused = q8_0_dot_q8_0(&w_block, &x_q, x_scales[0]);

        let mut w_decoded = Vec::new();
        dequant_q8_0(&w_block, &mut w_decoded);
        let x_decoded: Vec<f32> = x_q.iter().map(|&q| f32::from(q) * x_scales[0]).collect();
        let reference: f32 = w_decoded.iter().zip(&x_decoded).map(|(a, b)| a * b).sum();

        assert!((fused - reference).abs() < 1e-3, "fused={fused}, reference={reference}");
    }

    #[test]
    fn q4_0_dot_q8_0_of_an_all_zero_weight_block_is_zero() {
        // THE Q4_0 ZERO-POINT TRAP. A nibble of 0 does NOT mean a weight of
        // zero: Q4_0 is symmetric around a zero-point of 8, so a nibble decodes
        // as `(nibble - 8) * d`. Nibble 0 therefore decodes to `-8 * d`, and the
        // value zero is nibble *8*.
        //
        // `q4_0_block` takes dequantized VALUES and encodes them (`v + 8`), so
        // the all-zeros weight block is `[0i8; 32]` — passing `[-8i8; 32]` here
        // (which is what a reading of "nibble 0 everywhere" invites) builds a
        // block of weights that are all genuinely -8, and the dot product with
        // an all-5.0 activation is then 32 * -8 * 5 = -1280, not 0.
        //
        // That is not a hypothetical: this test was originally written that way
        // and correctly failed with -1280. The kernel was right; the fixture was
        // wrong. Preserved here because the same mistake, made in the kernel
        // rather than a test, produces a model that emits fluent nonsense.
        let w_block = q4_0_block(0x3C00, [0i8; 32]); // value 0 == nibble 8
        let x = vec![5.0f32; 32];
        let (x_scales, x_q) = quantize_row_q8_0(&x);
        assert_eq!(q4_0_dot_q8_0(&w_block, &x_q, x_scales[0]), 0.0);
    }

    #[test]
    fn q4_0_nibble_zero_decodes_to_minus_eight_not_zero() {
        // The positive statement of the trap above, asserted directly so that
        // anyone who "fixes" the zero-point later breaks this loudly.
        let w_block = q4_0_block(0x3C00, [-8i8; 32]); // scale 1.0, all nibbles 0
        let x = vec![5.0f32; 32];
        let (x_scales, x_q) = quantize_row_q8_0(&x);

        let dot = q4_0_dot_q8_0(&w_block, &x_q, x_scales[0]);
        assert!((dot - -1280.0).abs() < 1e-2, "expected 32 * -8 * 5 = -1280, got {dot}");
    }
}
