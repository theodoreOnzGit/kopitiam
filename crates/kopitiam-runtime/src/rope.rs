//! Rotary position embeddings (Su et al., 2021, "RoFormer").
//!
//! # Two pairing conventions, and the architecture decides which
//!
//! RoPE rotates pairs of dimensions within each attention head by an angle
//! proportional to sequence position. Which dimensions pair up is **not**
//! universal, and getting it wrong is the "silent-wrongness" failure the type
//! system cannot catch: both conventions produce a full `head_dim`-sized
//! output that looks like a valid rotation (same shape, same norm — rotation
//! preserves length either way), so the model loads, runs, and emits fluent
//! grammar built on mispositioned queries and keys.
//!
//! * [`RopeKind::Interleaved`] pairs *adjacent* dimensions — `(x[0], x[1])`,
//!   `(x[2], x[3])`, ... This is the original RoFormer/GPT-J pairing, and
//!   ggml calls it the "normal" RoPE (`LLAMA_ROPE_TYPE_NORM`, mode 0).
//! * [`RopeKind::SplitHalf`] pairs dimension `i` with `i + rope_dim/2`, i.e.
//!   the rotary width is split in half and corresponding elements across the
//!   halves rotate together. ggml calls this `GGML_ROPE_TYPE_NEOX`.
//!
//! **This crate got the mapping backwards and it cost a long hunt (`bd-b9x`).**
//! The previous docs here claimed "LLaMA and every model derived from it —
//! including every Qwen release — uses split-half". Only the Qwen half is
//! true. `llama.cpp`'s `llama_model_rope_type`
//! (`src/llama-model.cpp`, the switch commented *"use what we call a normal
//! RoPE, operating on pairs of consecutive head values"*) puts
//! `LLM_ARCH_LLAMA` — and Deci, Baichuan, StarCoder, InternLM2, MiniCPM,
//! Command-R, OLMo — in the **NORM/interleaved** group, while `LLM_ARCH_QWEN2`,
//! Phi and Gemma sit in the **NEOX/split-half** group.
//!
//! Why HF and GGUF disagree, which is the trap underneath the trap:
//! HuggingFace's `LlamaAttention` really does use `rotate_half` (split-half),
//! but `convert_hf_to_gguf.py` **permutes the Q and K weight rows** on the way
//! into a GGUF precisely so that ggml's cheaper interleaved rotation
//! reproduces the same result. So a GGUF LLaMA file expects interleaved RoPE:
//! reading the HF modelling code and concluding "split-half" is correct about
//! the *architecture* and wrong about the *file*.
//!
//! Symptomatically this is nasty: at position 0 the rotation angle is 0, so
//! both conventions are the identity and short prompts still answer correctly.
//! The error grows with position, so quality degrades the longer the context —
//! which reads exactly like "the model is small and weak" rather than like a
//! bug.
//!
//! Element indexing is transcribed from ggml's `rotate_pairs`
//! (`ggml/src/ggml-cpu/ops.cpp`, MIT): pair index `j` uses frequency
//! `theta^(-2j/rope_dim)` in *both* conventions — only the two elements it
//! rotates differ.
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
/// Which dimensions RoPE pairs up. See this module's docs — the choice is
/// per-architecture and picking wrong is silently, progressively wrong rather
/// than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// Adjacent pairs `(2j, 2j+1)`. ggml's `LLAMA_ROPE_TYPE_NORM` (mode 0).
    /// **This is what GGUF `llama`-architecture files want**, including every
    /// SmolLM2 release.
    Interleaved,
    /// Pairs `(j, j + rope_dim/2)`. ggml's `GGML_ROPE_TYPE_NEOX`. What
    /// `qwen2`, Phi and Gemma files want.
    SplitHalf,
}

pub struct RotaryEmbedding {
    rope_dim: usize,
    half: usize,
    kind: RopeKind,
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
    pub fn new(rope_dim: usize, theta: f32, max_position: usize, kind: RopeKind) -> Self {
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
        Self { rope_dim, half, kind, cos, sin }
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
                // Pair index `j` carries the same frequency in both
                // conventions; only which two elements it rotates differs.
                // Transcribed from ggml's `rotate_pairs`.
                for j in 0..self.half {
                    let cos = self.cos[table_base + j];
                    let sin = self.sin[table_base + j];
                    let (lo, hi) = match self.kind {
                        RopeKind::Interleaved => (base + 2 * j, base + 2 * j + 1),
                        RopeKind::SplitHalf => (base + j, base + self.half + j),
                    };
                    let x1 = data[lo];
                    let x2 = data[hi];
                    out[lo] = x1 * cos - x2 * sin;
                    out[hi] = x2 * cos + x1 * sin;
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
        let rope = RotaryEmbedding::new(4, 10_000.0, 4, RopeKind::SplitHalf);
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

    /// The other half of the pair above — and the regression guard for
    /// `bd-b9x`, where `llama`-architecture models were silently given the
    /// split-half rotation. The expected vector is the very one the test above
    /// names as "what interleaved would give", so the two tests pin each
    /// convention against the same hand-computed input from opposite sides.
    #[test]
    fn interleaved_matches_the_hand_computed_adjacent_pairing() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 4, RopeKind::Interleaved);
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]).unwrap();
        let out = rope.apply(&x, &[1]).unwrap().to_vec_f32().unwrap();

        // Interleaved ("NORM"/GPT-J): pairs (x[0], x[1]) and (x[2], x[3]).
        let expected = [-1.142_639_7, 1.922_075_6, 2.959_850_7, 4.029_799_5];
        for (o, e) in out.iter().zip(expected) {
            assert_close(*o, e);
        }
    }

    /// Why this bug hid for so long, encoded as a test: at position 0 the
    /// angle is 0, so cos=1/sin=0 and **both conventions are the identity**.
    /// A wrong choice therefore costs nothing on the first token and grows
    /// with position — which reads as "small model, weak answers" rather than
    /// as a bug. Anyone tempted to validate RoPE with a single-token prompt
    /// should read this first.
    #[test]
    fn both_conventions_are_the_identity_at_position_zero() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]).unwrap();
        for kind in [RopeKind::Interleaved, RopeKind::SplitHalf] {
            let rope = RotaryEmbedding::new(4, 10_000.0, 4, kind);
            let out = rope.apply(&x, &[0]).unwrap().to_vec_f32().unwrap();
            assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0], "{kind:?} must be identity at position 0");
        }
    }

    /// Both conventions preserve each rotated pair's norm, so "the output
    /// looks like a valid rotation" can never distinguish them — the reason
    /// this class of bug is invisible to shape and sanity checks alike.
    #[test]
    fn both_conventions_preserve_norm_so_neither_looks_wrong() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = Tensor::from_f32(v.clone(), [1, 1, 4]).unwrap();
        let norm = |s: &[f32]| s.iter().map(|a| a * a).sum::<f32>().sqrt();
        for kind in [RopeKind::Interleaved, RopeKind::SplitHalf] {
            let rope = RotaryEmbedding::new(4, 10_000.0, 8, kind);
            let out = rope.apply(&x, &[5]).unwrap().to_vec_f32().unwrap();
            assert!(
                (norm(&out) - norm(&v)).abs() < 1e-4,
                "{kind:?} changed the vector norm"
            );
        }
    }

    /// Rotation is an orthogonal linear map, so it can never change a
    /// vector's Euclidean norm -- true for *any* position, not just the
    /// hand-computed one above. A norm-changing bug (e.g. a sign error
    /// that makes the "rotation" actually a shear) would be caught here
    /// even if it happened to leave the hand-computed case's norm intact
    /// by coincidence.
    #[test]
    fn rotation_preserves_vector_norm() {
        let rope = RotaryEmbedding::new(8, 10_000.0, 16, RopeKind::SplitHalf);
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
        let rope = RotaryEmbedding::new(4, 10_000.0, 4, RopeKind::SplitHalf);
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
        let rope = RotaryEmbedding::new(2, 10_000.0, 4, RopeKind::SplitHalf);
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
        let rope = RotaryEmbedding::new(4, 10_000.0, 8, RopeKind::SplitHalf);
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
        let rope = RotaryEmbedding::new(4, 10_000.0, 4, RopeKind::SplitHalf);
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [4]).unwrap();
        assert!(matches!(rope.apply(&x, &[0]), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn mismatched_positions_length_is_rejected() {
        let rope = RotaryEmbedding::new(4, 10_000.0, 4, RopeKind::SplitHalf);
        let x = Tensor::from_f32(vec![1.0; 8], [1, 2, 4]).unwrap();
        assert!(matches!(rope.apply(&x, &[0]), Err(Error::ShapeMismatch { .. })));
    }
}
