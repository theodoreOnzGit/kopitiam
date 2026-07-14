//! Rotary position embeddings (Su et al., 2021, "RoFormer").
//!
//! # Which pairing convention this implements: split-half, not interleaved
//!
//! RoPE rotates pairs of dimensions within each attention head by an angle
//! proportional to sequence position. The *original* RoFormer paper (and
//! GPT-J) pairs *adjacent* dimensions: `(x[0], x[1])`, `(x[2], x[3])`, ...,
//! each pair rotated by its own frequency. LLaMA and every model derived
//! from it — including every Qwen release — uses a *different* pairing that
//! is easy to confuse with the first: dimension `i` is paired with
//! dimension `i + rope_dim/2`, i.e. the rotary width is split into two
//! halves and corresponding elements across the halves are rotated
//! together. `llama.cpp`/ggml calls this variant `GGML_ROPE_TYPE_NEOX`.
//!
//! This module implements the **split-half** ("NEOX") convention
//! exclusively, because that is what every GGUF Qwen/LLaMA export this
//! crate loads was trained with. Getting this backwards is exactly the
//! "silent-wrongness" failure mode the type system cannot catch: both
//! conventions produce a full head_dim-sized output that *looks* like a
//! valid rotation (same shape, same norm — rotation preserves vector
//! length regardless of which pairing is used), so a wrong pairing does not
//! fail to load or fail to run. It produces a model that emits fluent
//! grammar and wrong facts, because every downstream attention score is
//! computed from a subtly mispositioned query/key. See
//! `apply_matches_split_half_hand_computation_and_differs_from_interleaved`
//! below for the test that pins this down: it constructs a case where the
//! two conventions provably disagree and asserts this implementation lands
//! on the split-half answer.
//!
//! # The math
//!
//! For head dimension `d` (only the leading `rope_dim <= d` dimensions are
//! rotated; any trailing dimensions pass through unchanged — see
//! [`crate::config::QwenConfig::rope_dimension_count`]), position `m`, and
//! `half = rope_dim / 2`:
//!
//! ```text
//! freq[i]  = theta^(-2i / rope_dim)                  for i in 0..half
//! angle    = m * freq[i]
//! x1, x2   = x[0..half], x[half..rope_dim]            (the split halves)
//! out[i]        = x1[i] * cos(angle) - x2[i] * sin(angle)
//! out[half + i] = x2[i] * cos(angle) + x1[i] * sin(angle)
//! ```
//!
//! which is the standard 2D rotation matrix `[[cos, -sin], [sin, cos]]`
//! applied to the pair `(x1[i], x2[i])` — hence "rotary": each pair moves
//! along a circle of radius `sqrt(x1[i]^2 + x2[i]^2)` as position advances,
//! which is exactly why applying it never changes a vector's norm (see the
//! `rotation_preserves_vector_norm` test).

use kopitiam_core::{Error, Result};
use kopitiam_tensor::Tensor;

/// Precomputed `cos`/`sin` tables for every position up to `max_position`,
/// so [`RotaryEmbedding::apply`] is a table lookup per (position, pair)
/// rather than a `cos`/`sin` call — cheap enough to matter once this runs
/// once per token per layer per head during decode.
pub struct RotaryEmbedding {
    rope_dim: usize,
    half: usize,
    /// `cos[pos * half + i]` / `sin[pos * half + i]`.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RotaryEmbedding {
    /// Precomputes rotation tables for every position in `0..max_position`.
    ///
    /// `rope_dim` must be even (each pair needs two dimensions; see
    /// [`crate::config::QwenConfig::from_metadata`], which rejects an odd
    /// `rope_dimension_count` before a [`RotaryEmbedding`] is ever built).
    pub fn new(rope_dim: usize, theta: f32, max_position: usize) -> Self {
        debug_assert!(rope_dim.is_multiple_of(2), "rope_dim must be even");
        let half = rope_dim / 2;
        let freqs: Vec<f32> = (0..half).map(|i| theta.powf(-2.0 * i as f32 / rope_dim as f32)).collect();

        let mut cos = vec![0.0f32; max_position * half];
        let mut sin = vec![0.0f32; max_position * half];
        for pos in 0..max_position {
            for i in 0..half {
                let angle = pos as f32 * freqs[i];
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Self { rope_dim, half, cos, sin }
    }

    /// Rotates `x`, shaped `[n_heads, seq, head_dim]`, in place per
    /// position. `positions[s]` is the absolute sequence position of
    /// `x`'s `s`-th row — not necessarily `s` itself, since a KV-cache
    /// decode step's single query token sits at `cache_len`, not at `0`
    /// (see [`crate::kv_cache::KvCache`]).
    ///
    /// Only the leading `rope_dim` elements of each head are rotated;
    /// `head_dim - rope_dim` trailing elements (when `rope_dim < head_dim`)
    /// pass through unchanged, matching `llama.cpp`'s partial-rotary
    /// support.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if `x` is not rank 3, if its last dimension
    /// is smaller than `rope_dim`, or if `positions.len()` does not match
    /// `x`'s sequence dimension.
    pub fn apply(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let dims = x.shape().dims();
        if dims.len() != 3 {
            return Err(shape_err(x));
        }
        let (n_heads, seq, head_dim) = (dims[0], dims[1], dims[2]);
        if head_dim < self.rope_dim || seq != positions.len() {
            return Err(shape_err(x));
        }

        let data = x.to_vec_f32()?; // row-major [n_heads, seq, head_dim]
        let mut out = data.clone();

        for h in 0..n_heads {
            for (s, &pos) in positions.iter().enumerate() {
                let base = (h * seq + s) * head_dim;
                let table_base = pos * self.half;
                for i in 0..self.half {
                    let cos = self.cos[table_base + i];
                    let sin = self.sin[table_base + i];
                    let x1 = data[base + i];
                    let x2 = data[base + self.half + i];
                    out[base + i] = x1 * cos - x2 * sin;
                    out[base + self.half + i] = x2 * cos + x1 * sin;
                }
                // Dimensions [rope_dim..head_dim] were already copied into
                // `out` via `data.clone()` above and are left untouched.
            }
        }

        Tensor::from_f32(out, x.shape().clone())
    }
}

fn shape_err(x: &Tensor) -> Error {
    Error::ShapeMismatch { expected: x.shape().clone(), actual: x.shape().clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }

    /// Hand-computed (via an independent Python reference, not this
    /// module's own formula transcribed) for `head_dim = rope_dim = 4`,
    /// `theta = 10000`, position `1`, input `[1, 2, 3, 4]`.
    #[test]
    fn apply_matches_split_half_hand_computation_and_differs_from_interleaved() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 4);
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]).unwrap();
        let out = rope.apply(&x, &[1]).unwrap().to_vec_f32().unwrap();

        // Split-half ("NEOX") reference: pairs (x[0], x[2]) and (x[1], x[3]).
        let expected_split_half = [-1.984_110_6, 1.959_900_7, 2.462_378, 4.019_799_7];
        for (o, e) in out.iter().zip(expected_split_half) {
            assert_close(*o, e);
        }

        // The interleaved ("GPT-J") convention pairs adjacent dimensions
        // instead -- (x[0], x[1]) and (x[2], x[3]) -- and provably gives a
        // *different* answer for this same input. This is the assertion
        // that actually distinguishes "implemented the right convention"
        // from "implemented *a* rotation": if this module accidentally
        // paired adjacent dimensions, `out` would match this vector
        // instead, and the assertion above would already have failed, but
        // this pins down that the two conventions are not coincidentally
        // equal for this input either.
        let interleaved_would_give = [-1.142_639_7, 1.922_075_6, 2.959_850_7, 4.029_799_5];
        assert_ne!(
            out, interleaved_would_give,
            "split-half and interleaved RoPE must disagree on this input"
        );
    }

    /// Rotation is an orthogonal linear map, so it can never change a
    /// vector's Euclidean norm -- true for *any* position, not just the
    /// hand-computed one above. A norm-changing bug (e.g. a sign error
    /// that makes the "rotation" actually a shear) would be caught here
    /// even if it happened to leave the hand-computed case's norm intact
    /// by coincidence.
    #[test]
    fn rotation_preserves_vector_norm() {
        let rope = RotaryEmbedding::new(8, 10_000.0, 16);
        let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.37 - 1.1).collect();
        let norm_in: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt();

        for pos in 0..16 {
            let x = Tensor::from_f32(input.clone(), [1, 1, 8]).unwrap();
            let out = rope.apply(&x, &[pos]).unwrap().to_vec_f32().unwrap();
            let norm_out: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm_in - norm_out).abs() < 1e-4, "pos {pos}: {norm_in} vs {norm_out}");
        }
    }

    #[test]
    fn position_zero_is_the_identity() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 4);
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]).unwrap();
        let out = rope.apply(&x, &[0]).unwrap().to_vec_f32().unwrap();
        assert_close(out[0], 1.0);
        assert_close(out[1], 2.0);
        assert_close(out[2], 3.0);
        assert_close(out[3], 4.0);
    }

    /// Partial rotary: `rope_dim < head_dim` leaves the trailing
    /// dimensions untouched.
    #[test]
    fn dimensions_beyond_rope_dim_pass_through_unchanged() {
        let rope = RotaryEmbedding::new(2, 10_000.0, 4);
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]).unwrap();
        let out = rope.apply(&x, &[1]).unwrap().to_vec_f32().unwrap();
        // Dims [2, 3] (0-indexed) are beyond rope_dim=2, so unchanged.
        assert_close(out[2], 3.0);
        assert_close(out[3], 4.0);
        // Dims [0, 1] did rotate (angle != 0 at position 1).
        assert!((out[0] - 1.0).abs() > 1e-3 || (out[1] - 2.0).abs() > 1e-3);
    }

    #[test]
    fn different_positions_in_the_same_batch_get_different_rotations() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 8);
        let x = Tensor::from_f32(
            vec![
                1.0, 2.0, 3.0, 4.0, // seq index 0
                1.0, 2.0, 3.0, 4.0, // seq index 1 -- same input, different position
            ],
            [1, 2, 4],
        )
        .unwrap();
        let out = rope.apply(&x, &[0, 5]).unwrap().to_vec_f32().unwrap();
        // Position 0 is the identity (see position_zero_is_the_identity);
        // position 5 is not, so the two rows must differ despite identical
        // input.
        assert_eq!(&out[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_ne!(&out[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn wrong_rank_is_rejected() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 4);
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [4]).unwrap();
        assert!(matches!(rope.apply(&x, &[0]), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn mismatched_positions_length_is_rejected() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 4);
        let x = Tensor::from_f32(vec![1.0; 8], [1, 2, 4]).unwrap();
        assert!(matches!(rope.apply(&x, &[0]), Err(Error::ShapeMismatch { .. })));
    }
}
