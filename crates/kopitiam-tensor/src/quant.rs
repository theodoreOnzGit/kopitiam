//! Decoders for the GGUF/GGML block-quantized formats.
//!
//! Two families live here. The **classic** formats (Q4_0/Q4_1/Q5_0/Q5_1/
//! Q8_0) pack 32 elements ([`DType::block_size`]) into a small block sharing
//! one `f16` scale (and, for the `_1` variants, one `f16` minimum). The
//! **K-quant** formats (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, plus the intermediate
//! Q8_K) pack 256 elements into a "super-block" with a two-level scale: a
//! floating-point super-scale over the whole block times small integer
//! sub-block scales — see the K-quant section further down. The block
//! layouts below were read from the reference C structs and dequantize loops
//! in `crates/kopitiam-ai/vendor/llama.cpp/ggml/src/ggml-common.h` and
//! `ggml-quants.c` (MIT licensed) to get the byte offsets, packing and
//! nibble ordering right, then reimplemented from scratch here — no C was
//! transliterated.
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
        DType::Q2_K => dequant_q2_k,
        DType::Q3_K => dequant_q3_k,
        DType::Q4_K => dequant_q4_k,
        DType::Q5_K => dequant_q5_k,
        DType::Q6_K => dequant_q6_k,
        DType::Q8_K => dequant_q8_k,
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

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

// ---- ggml "K-quant" super-block formats (QK_K = 256) ----
//
// The five weight formats below (Q2_K..Q6_K) and the intermediate Q8_K each
// pack 256 elements into one super-block. The block layouts and decode
// arithmetic follow ggml's `block_qX_K` structs (`ggml-common.h`) and
// `dequantize_row_qX_K` routines (`ggml-quants.c`), both MIT — read for the
// byte layout and the exact integer arithmetic, then reimplemented here from
// scratch. Bit-exactness to ggml is a hard requirement (a K-quant weight
// misread by even one ULP corrupts every downstream activation), so each
// function mirrors ggml's field offsets and per-element formula precisely;
// see the per-function docs for the layout of each.
//
// A K-quant super-block carries two levels of scale: a floating-point
// *super-scale* (`d`, and for the asymmetric formats a super-min `dmin`)
// shared by the whole 256-element block, and a set of small integer
// *sub-block* scales (one per 16- or 32-element sub-block) that the
// super-scale multiplies. `get_scale_min_k4` is the fiddly part: it unpacks
// the 6-bit sub-block scales/mins that Q4_K and Q5_K pack into 12 bytes.

/// Unpacks the `j`-th 6-bit sub-block scale (`d`) and min (`m`) from a Q4_K /
/// Q5_K 12-byte `scales` field. This is ggml's `get_scale_min_k4` (MIT): the
/// first four scales and mins live in the low 6 bits of bytes 0..8, while
/// scales/mins 4..8 are split — their low 4 bits in bytes 8..12 and their
/// high 2 bits stolen from the top 2 bits of bytes 0..8.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Q2_K (84 bytes): `scales[16]` (4-bit scale + 4-bit min per sub-block),
/// `qs[64]` (2-bit quants), `f16 d`, `f16 dmin`. 16 sub-blocks of 16
/// elements; each element is `d*scale*q - dmin*min`. Follows ggml's Q2_K
/// (MIT).
fn dequant_q2_k(block: &[u8], out: &mut Vec<f32>) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = read_f16(block, 80);
    let min = read_f16(block, 82);

    let mut is = 0usize;
    // Two 128-element halves; `qs` advances 32 bytes per half.
    for half in 0..2 {
        let q = &qs[half * 32..half * 32 + 32];
        let mut shift = 0u32;
        for _j in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * f32::from(sc & 0x0F);
            let ml = min * f32::from(sc >> 4);
            for &b in &q[0..16] {
                out.push(dl * f32::from((b >> shift) & 3) - ml);
            }
            let sc = scales[is];
            is += 1;
            let dl = d * f32::from(sc & 0x0F);
            let ml = min * f32::from(sc >> 4);
            for &b in &q[16..32] {
                out.push(dl * f32::from((b >> shift) & 3) - ml);
            }
            shift += 2;
        }
    }
}

/// Q3_K (110 bytes): `hmask[32]` (the 3rd, high bit of each quant), `qs[64]`
/// (low 2 bits), `scales[12]` (16 six-bit signed sub-block scales, packed),
/// `f16 d`. Each element is `d * scale * (low2bits - (highbit ? 0 : 4))` —
/// note the *inverted* high-bit convention. Follows ggml's Q3_K (MIT).
fn dequant_q3_k(block: &[u8], out: &mut Vec<f32>) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scale_bytes = &block[96..108];
    let d_all = read_f16(block, 108);

    // Unpack the 12 packed bytes into 16 six-bit scales, exactly as ggml
    // does via a u32 aux array reinterpreted as i8. `aux0..2` are the 12
    // bytes read as three little-endian u32; the shuffle below spreads the
    // stolen high 2 bits of each scale back into place.
    let aux0 = u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]);
    let aux1 = u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]);
    let aux2 = u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]);
    let tmp = aux2;
    let out_aux = [
        (aux0 & KMASK2) | ((tmp & KMASK1) << 4),
        (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut scales = [0i8; 16];
    for (w, chunk) in out_aux.iter().enumerate() {
        for (b, byte) in chunk.to_le_bytes().iter().enumerate() {
            scales[w * 4 + b] = *byte as i8;
        }
    }

    let mut m = 1u8;
    let mut is = 0usize;
    for half in 0..2 {
        let q = &qs[half * 32..half * 32 + 32];
        let mut shift = 0u32;
        for _j in 0..4 {
            let dl = d_all * (i32::from(scales[is]) - 32) as f32;
            is += 1;
            for l in 0..16 {
                let high = if hmask[l] & m != 0 { 0 } else { 4 };
                let v = i32::from((q[l] >> shift) & 3) - high;
                out.push(dl * v as f32);
            }
            let dl = d_all * (i32::from(scales[is]) - 32) as f32;
            is += 1;
            for l in 0..16 {
                let high = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                let v = i32::from((q[l + 16] >> shift) & 3) - high;
                out.push(dl * v as f32);
            }
            shift += 2;
            // Mirrors ggml's `uint8_t m <<= 1`: after the 8th sub-block m
            // wraps to 0, but that value is never read again.
            m <<= 1;
        }
    }
}

/// Q4_K (144 bytes): `f16 d`, `f16 dmin`, `scales[12]` (8 six-bit scales +
/// 8 six-bit mins, packed via [`get_scale_min_k4`]), `qs[128]` (4-bit
/// quants). 8 sub-blocks of 32; each element is `d*scale*nibble -
/// dmin*min`. The `Q4_K_M` workhorse format. Follows ggml's Q4_K (MIT).
fn dequant_q4_k(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let min = read_f16(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut is = 0usize;
    // Four 64-element groups; each consumes 32 `qs` bytes (low nibbles then
    // high nibbles) and two sub-block scale/min pairs.
    for g in 0..4 {
        let q = &qs[g * 32..g * 32 + 32];
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = min * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = min * f32::from(m);
        for &byte in q {
            out.push(d1 * f32::from(byte & 0x0F) - m1);
        }
        // Ordering matters: ggml emits all 32 low nibbles, then all 32 high
        // nibbles — not interleaved — so these are two separate passes.
        for &byte in q {
            out.push(d2 * f32::from(byte >> 4) - m2);
        }
        is += 2;
    }
}

/// Q5_K (176 bytes): like [`dequant_q4_k`] plus `qh[32]`, a high-bit field
/// promoting each 4-bit quant to 5 bits. Layout: `f16 d`, `f16 dmin`,
/// `scales[12]`, `qh[32]`, `qs[128]`. Each element is
/// `d*scale*(nibble + 16*highbit) - dmin*min`. Follows ggml's Q5_K (MIT).
fn dequant_q5_k(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f16(block, 0);
    let min = read_f16(block, 2);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let ql = &block[48..176];

    let mut is = 0usize;
    for g in 0..4 {
        let q = &ql[g * 32..g * 32 + 32];
        // The high-bit field is shared across the whole super-block, read via
        // a different bit pair per group: (bit0,bit1),(bit2,bit3),... .
        let u1 = 1u8 << (2 * g);
        let u2 = 2u8 << (2 * g);
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = min * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = min * f32::from(m);
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out.push(d1 * (i32::from(q[l] & 0x0F) + hi) as f32 - m1);
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out.push(d2 * (i32::from(q[l] >> 4) + hi) as f32 - m2);
        }
        is += 2;
    }
}

/// Q6_K (210 bytes): `ql[128]` (low 4 bits), `qh[64]` (high 2 bits),
/// `scales[16]` (signed `i8` sub-block scales), `f16 d`. 16 sub-blocks of
/// 16; each 6-bit quant is `(low4 | high2<<4) - 32` (a signed value in
/// `[-32, 31]`), scaled by `d * scale`. Follows ggml's Q6_K (MIT).
fn dequant_q6_k(block: &[u8], out: &mut Vec<f32>) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208]; // interpreted as i8 below
    let d = read_f16(block, 208);

    // Two 128-element halves. Within a half ggml writes the four quadrants to
    // y[l], y[l+32], y[l+64], y[l+96] as it walks l=0..32, so we stage a
    // 128-slot buffer and emit it in natural order.
    for half in 0..2 {
        let ql = &ql[half * 64..half * 64 + 64];
        let qh = &qh[half * 32..half * 32 + 32];
        let sc = &scales[half * 8..half * 8 + 8];
        let mut vals = [0f32; 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = (i32::from(ql[l] & 0x0F) | (i32::from(qh[l] & 3) << 4)) - 32;
            let q2 = (i32::from(ql[l + 32] & 0x0F) | (i32::from((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = (i32::from(ql[l] >> 4) | (i32::from((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = (i32::from(ql[l + 32] >> 4) | (i32::from((qh[l] >> 6) & 3) << 4)) - 32;
            vals[l] = d * f32::from(sc[is] as i8) * q1 as f32;
            vals[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
            vals[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
            vals[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
        }
        out.extend_from_slice(&vals);
    }
}

/// Q8_K (292 bytes): `f32 d` (note: a full `f32` super-scale, unlike every
/// other quant's `f16`), `qs[256]` signed `i8` quants, then `i16 bsums[16]`
/// (group sums used only by ggml's fused dot products, ignored on decode).
/// Each element is simply `d * q`. Follows ggml's Q8_K (MIT).
fn dequant_q8_k(block: &[u8], out: &mut Vec<f32>) {
    let d = read_f32(block, 0);
    let qs = &block[4..260];
    out.extend(qs.iter().map(|&b| d * f32::from(b as i8)));
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

/// Integer dot product of one Q4_K **super-block** (256 weights) against the
/// eight Q8_0 activation blocks covering the same 256 positions.
///
/// # Why a K-quant can be dotted at all, and why the algebra is exact
///
/// A Q4_K element is `d * sc_s * nibble - dmin * m_s`, where `s` is its
/// 32-element sub-block ([`dequant_q4_k`]). That is *affine*, not merely
/// scaled, so unlike Q4_0 the min term must be carried separately — but it
/// still factors out of the sum. Per sub-block `s`, with the activation
/// written as `x_scale_s * xq`:
///
/// ```text
/// sum_j (d*sc_s*nib_j - dmin*m_s) * x_scale_s * xq_j
///   = x_scale_s * ( d*sc_s * SUM(nib_j * xq_j)  -  dmin*m_s * SUM(xq_j) )
/// ```
///
/// Both sums are pure `i32` — exact, no rounding — and the whole sub-block
/// then costs a handful of `f32` multiplies. Same deal as
/// [`q4_0_dot_q8_0`], just with the extra `SUM(xq_j)` term that the min
/// needs. Confirm hor: `SUM(xq_j)` is ggml's `y[i].bsums`, precomputed
/// there because Q8_K stores it; we sum it inline instead since Q8_0 has no
/// such field.
///
/// # Provenance, and one deliberate divergence
///
/// Follows ggml's `ggml_vec_dot_q4_K_q8_K_generic`
/// (`crates/kopitiam-ai/vendor/llama.cpp/ggml/src/ggml-cpu/quants.c:696`,
/// MIT) — same factorisation, same `sumf -= dmin * sumi` min term (line
/// 765), same per-sub-block scale walk (line 747).
///
/// **We diverge on the activation format, on purpose.** ggml dots Q4_K
/// against **Q8_K**, which carries ONE `f32` scale for all 256 elements;
/// this crate quantizes activations to **Q8_0**, which carries one `f16`-
/// range scale per 32. Q8_0's blocks line up exactly with Q4_K's 32-element
/// sub-blocks, so the factorisation above still holds term for term — the
/// only difference is that `x_scale` sits inside the per-sub-block sum
/// instead of being hoisted to the super-block. That makes our activation
/// *more* finely scaled than ggml's (8 scales per 256 instead of 1), so it
/// cannot lose precision relative to the reference; it costs 7 extra `f32`
/// multiplies per super-block. The reason is not precision though, it is
/// reuse: [`quantize_row_q8_0`] is the crate's one activation encoder and
/// every fused kernel here consumes it, so adding Q4_K needed no second
/// activation format. What WOULD make this wrong: if a caller ever passed
/// `x_scales` that did not correspond 1:1 with the 32-element sub-blocks
/// (e.g. one scale per 256), every min term would be mis-weighted.
///
/// `weight_block` must be exactly [`DType::Q4_K`]'s [`DType::block_bytes`]
/// (144) long; `x_q` exactly 256 elements; `x_scales` exactly 8, in
/// sub-block order.
pub(crate) fn q4_k_dot_q8_0(weight_block: &[u8], x_q: &[i8], x_scales: &[f32]) -> f32 {
    debug_assert_eq!(weight_block.len(), DType::Q4_K.block_bytes());
    debug_assert_eq!(x_q.len(), 256);
    debug_assert_eq!(x_scales.len(), 8);

    let d = read_f16(weight_block, 0);
    let dmin = read_f16(weight_block, 2);
    let scales = &weight_block[4..16];
    let qs = &weight_block[16..144];

    let mut acc = 0f32;
    // Four 64-element groups, each 32 `qs` bytes: low nibbles are sub-block
    // 2g, high nibbles sub-block 2g+1 — the same walk `dequant_q4_k` does,
    // so element `e` lands in sub-block `e / 32` on both paths.
    for g in 0..4 {
        let q = &qs[g * 32..g * 32 + 32];
        for half in 0..2 {
            let s = 2 * g + half;
            let (sc, m) = get_scale_min_k4(s, scales);
            let xq = &x_q[s * 32..s * 32 + 32];
            let mut dot: i32 = 0;
            let mut xsum: i32 = 0;
            for (l, &xv) in xq.iter().enumerate() {
                let nib = if half == 0 { q[l] & 0x0F } else { q[l] >> 4 };
                dot += i32::from(nib) * i32::from(xv);
                xsum += i32::from(xv);
            }
            acc += x_scales[s]
                * (d * f32::from(sc) * dot as f32 - dmin * f32::from(m) * xsum as f32);
        }
    }
    acc
}

/// Integer dot product of one Q6_K **super-block** (256 weights) against the
/// eight Q8_0 activation blocks covering the same 256 positions.
///
/// Simpler than [`q4_k_dot_q8_0`]: Q6_K is symmetric (`d * sc_s * (q - 32)`,
/// no min), so there is no `SUM(xq_j)` correction term. The fiddly bit is
/// the *ordering* instead — Q6_K's sub-blocks are 16 elements, not 32, and
/// its bytes are laid out in four interleaved quadrants per 128-element
/// half rather than in natural order ([`dequant_q6_k`]). So this walks the
/// same quadrant pattern the dequantizer does and files each product into
/// an accumulator indexed by `element / 16`, which is that element's real
/// sub-block. Getting that index wrong would apply a neighbouring
/// sub-block's scale — plausible-looking output, silently wrong weights.
///
/// Two 16-element sub-blocks share one 32-element Q8_0 activation block, so
/// `x_scales[s / 2]` is the scale for sub-block `s`; the two are combined
/// under a single multiply per activation block.
///
/// # Provenance
///
/// Follows ggml's `ggml_vec_dot_q6_K_q8_K_generic`
/// (`crates/kopitiam-ai/vendor/llama.cpp/ggml/src/ggml-cpu/quants.c:851`,
/// MIT): the quadrant unpack at lines 879-882 and the `scales[is++]` walk
/// advancing once per 16 elements at line 890. The scales are **signed**
/// `i8` here, unlike Q4_K's unsigned 6-bit — read them wrong and the sign
/// flips on roughly half the sub-blocks. Same deliberate Q8_0-instead-of-
/// Q8_K activation divergence as [`q4_k_dot_q8_0`]; see its docs.
///
/// `weight_block` must be exactly [`DType::Q6_K`]'s [`DType::block_bytes`]
/// (210) long; `x_q` exactly 256 elements; `x_scales` exactly 8.
pub(crate) fn q6_k_dot_q8_0(weight_block: &[u8], x_q: &[i8], x_scales: &[f32]) -> f32 {
    debug_assert_eq!(weight_block.len(), DType::Q6_K.block_bytes());
    debug_assert_eq!(x_q.len(), 256);
    debug_assert_eq!(x_scales.len(), 8);

    let ql_all = &weight_block[0..128];
    let qh_all = &weight_block[128..192];
    let scales = &weight_block[192..208]; // signed i8, read as such below
    let d = read_f16(weight_block, 208);

    // One `i32` accumulator per 16-element sub-block. Exact: each product is
    // at most 32 * 127 in magnitude and 16 of them cannot overflow `i32`.
    let mut sub = [0i32; 16];
    for half in 0..2 {
        let ql = &ql_all[half * 64..half * 64 + 64];
        let qh = &qh_all[half * 32..half * 32 + 32];
        let base = half * 128;
        for l in 0..32 {
            let q1 = (i32::from(ql[l] & 0x0F) | (i32::from(qh[l] & 3) << 4)) - 32;
            let q2 = (i32::from(ql[l + 32] & 0x0F) | (i32::from((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = (i32::from(ql[l] >> 4) | (i32::from((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = (i32::from(ql[l + 32] >> 4) | (i32::from((qh[l] >> 6) & 3) << 4)) - 32;
            for (e, q) in [(base + l, q1), (base + l + 32, q2), (base + l + 64, q3), (base + l + 96, q4)] {
                sub[e / 16] += q * i32::from(x_q[e]);
            }
        }
    }

    let mut acc = 0f32;
    for (b, &x_scale) in x_scales.iter().enumerate() {
        let lo = f32::from(scales[2 * b] as i8) * sub[2 * b] as f32;
        let hi = f32::from(scales[2 * b + 1] as i8) * sub[2 * b + 1] as f32;
        acc += x_scale * (lo + hi);
    }
    acc * d
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

    // ---- K-quant super-block tests ----
    //
    // Each test hand-encodes a super-block using the exact ggml packing
    // (the encoders below are the literal inverse of ggml's `block_qX_K`
    // layout / `get_scale_min_k4` / the Q3_K scale shuffle), then asserts the
    // decoder produces bit-exact `f32`. All chosen super-scales are exact f16
    // (1.0 = 0x3C00, 0.5 = 0x3800) and every quant/scale is a small integer,
    // so each expected value is an exact integer-or-half product with no
    // rounding: `assert_eq!` on `f32` is legitimate, not fragile. The
    // expected values are recomputed by an independent per-element formula
    // (different code path from the decoder's group-walking loops), so a bug
    // in either the packing or the decode arithmetic surfaces here.

    const F16_ONE: u16 = 0x3C00; // 1.0
    const F16_HALF: u16 = 0x3800; // 0.5

    /// Inverse of ggml's `get_scale_min_k4`: packs 8 six-bit sub-block scales
    /// and 8 six-bit mins into the 12-byte Q4_K/Q5_K `scales` field.
    fn pack_k4_scales(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for j in 0..4 {
            q[j] = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6);
            q[j + 4] = (mins[j] & 63) | ((mins[j + 4] >> 4) << 6);
            q[j + 8] = (scales[j + 4] & 0x0F) | ((mins[j + 4] & 0x0F) << 4);
        }
        // Round-trip guard: what we pack must unpack to what we asked for.
        for j in 0..8 {
            assert_eq!(get_scale_min_k4(j, &q), (scales[j], mins[j]), "pack/unpack disagree at {j}");
        }
        q
    }

    // -- Q4_K --

    fn build_q4_k(d: u16, dmin: u16, scales: &[u8; 8], mins: &[u8; 8], nib: &[u8; 256]) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        block[4..16].copy_from_slice(&pack_k4_scales(scales, mins));
        // qs: group g byte l holds sub-block 2g element l (low nibble) and
        // sub-block 2g+1 element l (high nibble).
        for g in 0..4 {
            for l in 0..32 {
                let low = nib[2 * g * 32 + l] & 0x0F;
                let high = nib[(2 * g + 1) * 32 + l] & 0x0F;
                block[16 + g * 32 + l] = low | (high << 4);
            }
        }
        block
    }

    #[test]
    fn q4_k_is_bit_exact_across_both_get_scale_min_k4_branches() {
        // Sub-block scales 4..8 and mins 4..8 exercise the *else* branch of
        // get_scale_min_k4 (the 6-bit split-packing); 63 hits the field max.
        let scales = [1u8, 2, 3, 4, 5, 6, 7, 63];
        let mins = [0u8, 1, 2, 3, 63, 10, 20, 30];
        let mut nib = [0u8; 256];
        for (e, n) in nib.iter_mut().enumerate() {
            *n = (e % 16) as u8; // sweep every nibble value 0..15
        }
        let block = build_q4_k(F16_ONE, F16_HALF, &scales, &mins, &nib);
        let mut out = Vec::new();
        dequant_q4_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        for e in 0..256 {
            let s = e / 32; // sub-block index
            let expected = 1.0 * f32::from(scales[s]) * f32::from(nib[e]) - 0.5 * f32::from(mins[s]);
            assert_eq!(out[e], expected, "Q4_K mismatch at element {e} (sub-block {s})");
        }
    }

    #[test]
    fn q4_k_all_zero_superblock_decodes_to_all_zero() {
        let block = build_q4_k(F16_ONE, F16_HALF, &[0; 8], &[0; 8], &[0; 256]);
        let mut out = Vec::new();
        dequant_q4_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // -- Q5_K (Q4_K plus a shared high-bit field) --

    fn build_q5_k(
        d: u16,
        dmin: u16,
        scales: &[u8; 8],
        mins: &[u8; 8],
        nib: &[u8; 256],
        hi: &[u8; 256],
    ) -> Vec<u8> {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        block[4..16].copy_from_slice(&pack_k4_scales(scales, mins));
        // qh[l] bit s carries the high bit of sub-block s at position l.
        for l in 0..32 {
            let mut byte = 0u8;
            for s in 0..8 {
                byte |= (hi[s * 32 + l] & 1) << s;
            }
            block[16 + l] = byte;
        }
        for g in 0..4 {
            for l in 0..32 {
                let low = nib[2 * g * 32 + l] & 0x0F;
                let high = nib[(2 * g + 1) * 32 + l] & 0x0F;
                block[48 + g * 32 + l] = low | (high << 4);
            }
        }
        block
    }

    #[test]
    fn q5_k_is_bit_exact_including_the_fifth_bit() {
        let scales = [2u8, 4, 6, 8, 10, 12, 63, 1];
        let mins = [0u8, 2, 4, 6, 8, 63, 1, 3];
        let mut nib = [0u8; 256];
        let mut hi = [0u8; 256];
        for e in 0..256 {
            nib[e] = (e % 16) as u8;
            hi[e] = ((e / 16) % 2) as u8; // alternate the 5th bit in stripes
        }
        let block = build_q5_k(F16_ONE, F16_HALF, &scales, &mins, &nib, &hi);
        let mut out = Vec::new();
        dequant_q5_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        for e in 0..256 {
            let s = e / 32;
            let val = i32::from(nib[e]) + if hi[e] != 0 { 16 } else { 0 };
            let expected = 1.0 * f32::from(scales[s]) * val as f32 - 0.5 * f32::from(mins[s]);
            assert_eq!(out[e], expected, "Q5_K mismatch at element {e} (sub-block {s})");
        }
    }

    // -- Q6_K --

    fn build_q6_k(d: u16, scales: &[i8; 16], v: &[u8; 256]) -> Vec<u8> {
        // v[e] is the raw 6-bit quant (0..63); the decoder subtracts 32.
        let mut block = vec![0u8; 210];
        for h in 0..2 {
            for l in 0..32 {
                let vq1 = v[h * 128 + l];
                let vq2 = v[h * 128 + l + 32];
                let vq3 = v[h * 128 + l + 64];
                let vq4 = v[h * 128 + l + 96];
                block[h * 64 + l] = (vq1 & 0x0F) | ((vq3 & 0x0F) << 4);
                block[h * 64 + l + 32] = (vq2 & 0x0F) | ((vq4 & 0x0F) << 4);
                block[128 + h * 32 + l] =
                    (vq1 >> 4) | ((vq2 >> 4) << 2) | ((vq3 >> 4) << 4) | ((vq4 >> 4) << 6);
            }
        }
        for (i, &s) in scales.iter().enumerate() {
            block[192 + i] = s as u8;
        }
        block[208..210].copy_from_slice(&d.to_le_bytes());
        block
    }

    #[test]
    fn q6_k_is_bit_exact_with_signed_scales_and_the_minus_32_offset() {
        // Signed sub-block scales, both signs; raw quants sweep 0..63 so the
        // decoded (q - 32) sweeps the full signed [-32, 31].
        let scales: [i8; 16] = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];
        let mut v = [0u8; 256];
        for (e, x) in v.iter_mut().enumerate() {
            *x = (e % 64) as u8;
        }
        let block = build_q6_k(F16_ONE, &scales, &v);
        let mut out = Vec::new();
        dequant_q6_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        for e in 0..256 {
            let h = e / 128;
            let within = e % 128;
            let l = within % 32;
            let quadrant = within / 32;
            let scale_index = h * 8 + l / 16 + 2 * quadrant;
            let expected = 1.0 * f32::from(scales[scale_index]) * (i32::from(v[e]) - 32) as f32;
            assert_eq!(out[e], expected, "Q6_K mismatch at element {e}");
        }
    }

    // -- Q2_K --

    fn build_q2_k(d: u16, dmin: u16, scales: &[u8; 16], mins: &[u8; 16], q2: &[u8; 256]) -> Vec<u8> {
        let mut block = vec![0u8; 84];
        for i in 0..16 {
            block[i] = (scales[i] & 0x0F) | ((mins[i] & 0x0F) << 4);
        }
        // qs[h*32 + p] packs the four shift-planes (j = 0..4) at position p.
        for h in 0..2 {
            for p in 0..32 {
                let mut byte = 0u8;
                for j in 0..4 {
                    byte |= (q2[h * 128 + j * 32 + p] & 3) << (2 * j);
                }
                block[16 + h * 32 + p] = byte;
            }
        }
        block[80..82].copy_from_slice(&d.to_le_bytes());
        block[82..84].copy_from_slice(&dmin.to_le_bytes());
        block
    }

    #[test]
    fn q2_k_is_bit_exact() {
        let mut scales = [0u8; 16];
        let mut mins = [0u8; 16];
        for i in 0..16 {
            scales[i] = (i as u8 % 15) + 1; // 1..15
            mins[i] = (i as u8) % 16; // 0..15
        }
        let mut q2 = [0u8; 256];
        for (e, x) in q2.iter_mut().enumerate() {
            *x = (e % 4) as u8; // sweep 0..3
        }
        let block = build_q2_k(F16_ONE, F16_HALF, &scales, &mins, &q2);
        let mut out = Vec::new();
        dequant_q2_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        for e in 0..256 {
            let h = e / 128;
            let within = e % 128;
            let r = within % 32;
            let j = within / 32;
            let s = h * 8 + 2 * j + usize::from(r >= 16);
            let expected = 1.0 * f32::from(scales[s]) * f32::from(q2[e]) - 0.5 * f32::from(mins[s]);
            assert_eq!(out[e], expected, "Q2_K mismatch at element {e} (sub-block {s})");
        }
    }

    // -- Q3_K --

    /// Inverse of ggml's Q3_K 6-bit scale shuffle: packs 16 six-bit scales
    /// into the 12-byte `scales` field.
    fn pack_q3_scales(sc: &[u8; 16]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for b in 0..4 {
            q[b] = (sc[b] & 0x0F) | ((sc[8 + b] & 0x0F) << 4);
            q[4 + b] = (sc[4 + b] & 0x0F) | ((sc[12 + b] & 0x0F) << 4);
            q[8 + b] = (sc[b] >> 4)
                | ((sc[4 + b] >> 4) << 2)
                | ((sc[8 + b] >> 4) << 4)
                | ((sc[12 + b] >> 4) << 6);
        }
        q
    }

    fn build_q3_k(d: u16, sc: &[u8; 16], q2: &[u8; 256], hbit: &[u8; 256]) -> Vec<u8> {
        let mut block = vec![0u8; 110];
        // hmask[r] bit (h*4 + j) carries the high bit at position r.
        for r in 0..32 {
            let mut byte = 0u8;
            for h in 0..2 {
                for j in 0..4 {
                    byte |= (hbit[h * 128 + j * 32 + r] & 1) << (h * 4 + j);
                }
            }
            block[r] = byte;
        }
        for h in 0..2 {
            for p in 0..32 {
                let mut byte = 0u8;
                for j in 0..4 {
                    byte |= (q2[h * 128 + j * 32 + p] & 3) << (2 * j);
                }
                block[32 + h * 32 + p] = byte;
            }
        }
        block[96..108].copy_from_slice(&pack_q3_scales(sc));
        block[108..110].copy_from_slice(&d.to_le_bytes());
        block
    }

    #[test]
    fn q3_k_is_bit_exact_including_the_aux_scale_unpack_and_inverted_high_bit() {
        // 6-bit scales spanning the whole range, so the 12->16 scale shuffle
        // (the trickiest part of Q3_K) is fully exercised, including scales
        // whose high 2 bits are set.
        let mut sc = [0u8; 16];
        for (i, s) in sc.iter_mut().enumerate() {
            *s = (i as u8 * 4) & 63; // 0,4,8,...,60
        }
        let mut q2 = [0u8; 256];
        let mut hbit = [0u8; 256];
        for e in 0..256 {
            q2[e] = (e % 4) as u8;
            hbit[e] = ((e / 32) % 2) as u8; // toggle the high bit per sub-block plane
        }
        let block = build_q3_k(F16_ONE, &sc, &q2, &hbit);
        let mut out = Vec::new();
        dequant_q3_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        for e in 0..256 {
            let h = e / 128;
            let within = e % 128;
            let r = within % 32;
            let j = within / 32;
            let s = h * 8 + 2 * j + usize::from(r >= 16);
            // Inverted high-bit convention: subtract 0 if the bit is set, 4 if not.
            let val = i32::from(q2[e]) - if hbit[e] != 0 { 0 } else { 4 };
            let expected = 1.0 * (i32::from(sc[s]) - 32) as f32 * val as f32;
            assert_eq!(out[e], expected, "Q3_K mismatch at element {e} (sub-block {s})");
        }
    }

    // -- Q8_K --

    #[test]
    fn q8_k_is_bit_exact_with_an_f32_scale() {
        // Q8_K's scale is a full f32 (not f16) — 0.25 is exact.
        let d: f32 = 0.25;
        let mut block = vec![0u8; 292];
        block[0..4].copy_from_slice(&d.to_le_bytes());
        let mut qs = [0i8; 256];
        for (j, q) in qs.iter_mut().enumerate() {
            *q = (j as i32 - 128) as i8; // sweep the full i8 range
            block[4 + j] = *q as u8;
        }
        // bsums are ignored on decode; fill with garbage to prove it.
        block[260..292].fill(0xAB);
        let mut out = Vec::new();
        dequant_q8_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        for j in 0..256 {
            assert_eq!(out[j], 0.25 * f32::from(qs[j]), "Q8_K mismatch at {j}");
        }
    }

    // -- dispatch --

    #[test]
    fn dequantize_dispatches_k_quants_and_handles_multiple_superblocks() {
        let scales = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mins = [0u8; 8];
        let nib_a = [1u8; 256];
        let nib_b = [2u8; 256];
        let mut bytes = build_q4_k(F16_ONE, F16_HALF, &scales, &mins, &nib_a);
        bytes.extend(build_q4_k(F16_ONE, F16_HALF, &scales, &mins, &nib_b));
        let out = dequantize(DType::Q4_K, &bytes).unwrap();
        assert_eq!(out.len(), 512);
        // First superblock, sub-block 0 (scale 1, min 0), nibble 1 -> 1.0.
        assert_eq!(out[0], 1.0);
        // Second superblock, element 0 -> scale 1 * nibble 2 = 2.0.
        assert_eq!(out[256], 2.0);
    }
}
