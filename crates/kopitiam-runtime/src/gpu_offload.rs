//! Optional GPU offload for the **output projection**, the single most
//! expensive matmul in a decode step.
//!
//! # Why only this one matmul
//!
//! Measured on an Intel integrated GPU against 14 CPU cores at SmolLM2-360M's
//! real shapes (`kopitiam-gpu/tests/matmul_timing.rs`):
//!
//! ```text
//! decode  attn q/o     (1 tok)   cpu   443µs   gpu 1.88ms   0.24x   LOSS
//! decode  mlp gate/up  (1 tok)   cpu  1.28ms   gpu 3.20ms   0.40x   LOSS
//! output head, quantized + resident        cpu 26.12ms  gpu 8.55ms   3.06x  WIN
//! ```
//!
//! A decode step is a single row of activations — far too little arithmetic to
//! pay for moving a weight matrix across the bus. Offloading the small
//! projections would make chat 2–4x *slower*. The output head is different on
//! two counts: it is `vocab x hidden` (49152 x 960 for SmolLM2-360M), so it
//! costs more per token than every other decode matmul combined; and its weight
//! can stay **resident** on the device, so only the activation row crosses the
//! bus per token.
//!
//! # Why the weight has to stay quantized, not merely resident
//!
//! As `f32` that weight is 188 MB, against a 128 MB
//! `max_storage_buffer_binding_size` on the maintainer's adapter — it cannot be
//! bound at all. Held as block-scaled int8 it is ~47 MB of quants plus ~6 MB of
//! scales. So "quantized" is not an optimisation here, it is the difference
//! between running on the GPU and not.
//!
//! # The tied-embedding wrinkle, which is also a CPU-side finding
//!
//! Neither SmolLM2 release ships a separate `output.weight`; both tie it to
//! `token_embd.weight`. [`crate::weights::ModelWeights`] loads `token_embd`
//! through `load_tensor_f32` (the embedding lookup needs real floats), and the
//! tied fallback clones that — so **the biggest matmul in the model has been
//! running against a fully dequantized f32 weight**, skipping the fused
//! quantized kernel every other projection gets. This module reads the original
//! quantized bytes back off the model instead.

use kopitiam_core::{Error, Result};
use kopitiam_gpu::{Executor, ResidentBlockQ8Weight, BLOCK};
use kopitiam_loader::LoadedModel;

/// Bytes per Q8_0 block: one `f16` scale followed by 32 `i8` quants.
/// Matches `kopitiam_tensor`'s `dequant_q8_0`, which is itself verified against
/// ggml's `ggml-quants.c`.
const Q8_0_BLOCK_BYTES: usize = 2 + BLOCK;

/// Splits raw GGUF Q8_0 bytes into the separated `(scales, quants)` layout
/// `kopitiam-gpu` wants.
///
/// `kopitiam-gpu` is domain-agnostic parallel-compute infrastructure and does
/// not know what a GGUF is — it takes generic block-scaled int8. Translating
/// one to the other is format knowledge, so it lives here.
///
/// `rows * cols` elements are expected, `cols` a multiple of [`BLOCK`].
/// Returns `(scales, quants)` with `rows * cols / BLOCK` scales and
/// `rows * cols` quants, both in row-major order.
///
/// # Errors
///
/// [`Error::MalformedModel`] if `cols` is not a whole number of blocks or the
/// byte length does not match — both of which mean the tensor is not the Q8_0
/// shape claimed, and guessing would produce a plausible wrong weight.
pub fn split_q8_0(bytes: &[u8], rows: usize, cols: usize) -> Result<(Vec<f32>, Vec<i8>)> {
    if cols == 0 || !cols.is_multiple_of(BLOCK) {
        return Err(Error::MalformedModel {
            format: "q8_0-split",
            reason: format!("cols ({cols}) must be a non-zero multiple of {BLOCK}"),
        });
    }
    let blocks = rows * (cols / BLOCK);
    let expected = blocks * Q8_0_BLOCK_BYTES;
    if bytes.len() < expected {
        return Err(Error::MalformedModel {
            format: "q8_0-split",
            reason: format!("expected {expected} bytes for {rows}x{cols} Q8_0, got {}", bytes.len()),
        });
    }

    let mut scales = Vec::with_capacity(blocks);
    let mut quants = Vec::with_capacity(rows * cols);
    for b in 0..blocks {
        let block = &bytes[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
        scales.push(f16_to_f32(u16::from_le_bytes([block[0], block[1]])));
        quants.extend(block[2..].iter().map(|&byte| byte as i8));
    }
    Ok((scales, quants))
}

/// Minimal IEEE-754 half -> single conversion, matching `kopitiam_tensor`'s
/// `read_f16`. Duplicated rather than exported from there because it is four
/// lines and a cross-crate `pub` for it would widen that crate's API surface
/// for one caller.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let out = match exp {
        0 if frac == 0 => sign << 31,
        // Subnormal: normalise into a single-precision exponent.
        0 => {
            let mut e = -1i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + e) as u32;
            (sign << 31) | (exp32 << 23) | ((f & 0x3FF) << 13)
        }
        0x1F => (sign << 31) | (0xFF << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

/// The output projection, resident on the GPU.
///
/// `None` from [`Self::try_build`] is the ordinary case, not a failure: no GPU,
/// a weight the kernel does not handle, or a device that cannot bind it. The
/// caller keeps using the CPU path.
pub struct GpuOutputHead {
    weight: ResidentBlockQ8Weight,
}

impl GpuOutputHead {
    /// Uploads the model's output projection if — and only if — every condition
    /// for it to be a win holds.
    ///
    /// Returns `None` (never an error) when offload is simply not available:
    /// there is no GPU, the tensor is not Q8_0 (the 1.7B ships Q6_K, which this
    /// kernel does not decode), the tensor is missing, or the adapter cannot
    /// bind it. Every one of those is a normal machine configuration, not a
    /// fault, and the CPU path is always correct.
    #[must_use]
    pub fn try_build(model: &LoadedModel, executor: &Executor) -> Option<Self> {
        let ctx = executor.gpu_context()?;

        // Tied embeddings are the norm: neither SmolLM2 ships `output.weight`.
        let name = if model.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let entry = model.tensor(name)?;
        if entry.dtype != kopitiam_core::DType::Q8_0 {
            return None;
        }
        let dims = entry.shape.dims();
        if dims.len() != 2 {
            return None;
        }
        let (rows, cols) = (dims[0], dims[1]);

        let bytes = model.tensor_bytes(name).ok()?;
        let (scales, quants) = split_q8_0(bytes, rows, cols).ok()?;
        let weight = ResidentBlockQ8Weight::upload(ctx, &quants, &scales, rows, cols).ok()?;
        Some(Self { weight })
    }

    /// `logits = x @ w^T` for `m` rows of hidden states.
    ///
    /// # Errors
    ///
    /// Any GPU failure, which the caller should treat as "use the CPU path this
    /// time" rather than as fatal.
    pub fn forward(&self, executor: &Executor, x: &[f32], m: usize) -> Option<Vec<f32>> {
        let ctx = executor.gpu_context()?;
        self.weight.matmul_nt(ctx, x, m).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert!((f16_to_f32(0x3555) - 0.333_251).abs() < 1e-5);
    }

    /// The split must agree with the dequantizer that is already verified
    /// against ggml: rebuilding `scale * quant` has to reproduce what
    /// `kopitiam_tensor` produces from the same bytes. This is the check that
    /// makes the GPU weight trustworthy — the kernel can only be as right as
    /// the numbers handed to it.
    #[test]
    fn split_then_recombine_matches_the_verified_dequantizer() {
        // Two blocks: scale 1.0 and 0.5, quants spanning the signed range.
        let mut bytes = Vec::new();
        for (scale_bits, base) in [(0x3C00u16, 0i32), (0x3800u16, 1)] {
            bytes.extend_from_slice(&scale_bits.to_le_bytes());
            for t in 0..BLOCK {
                bytes.push((((t as i32 * 17 + base) % 255) - 128) as i8 as u8);
            }
        }
        let (scales, quants) = split_q8_0(&bytes, 1, BLOCK * 2).expect("split");
        assert_eq!(scales.len(), 2);
        assert_eq!(quants.len(), BLOCK * 2);
        assert_eq!(scales[0], 1.0);
        assert_eq!(scales[1], 0.5);

        let dequantized = kopitiam_tensor::Tensor::from_quantized(
            kopitiam_core::DType::Q8_0,
            bytes.clone(),
            [1, BLOCK * 2],
        )
        .expect("reference tensor")
        .to_dtype(kopitiam_core::DType::F32)
        .expect("reference dequantize")
        .to_vec_f32()
        .expect("reference values");
        for i in 0..BLOCK * 2 {
            let ours = scales[i / BLOCK] * f32::from(quants[i]);
            assert!(
                (ours - dequantized[i]).abs() < 1e-6,
                "element {i}: split gives {ours}, reference dequantizer gives {}",
                dequantized[i]
            );
        }
    }

    #[test]
    fn a_column_count_that_is_not_a_whole_number_of_blocks_is_refused() {
        let err = split_q8_0(&[0u8; 100], 1, 40).unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }

    #[test]
    fn truncated_bytes_are_refused_rather_than_padded() {
        let err = split_q8_0(&[0u8; 10], 1, BLOCK).unwrap_err();
        assert!(matches!(err, Error::MalformedModel { .. }));
    }
}
