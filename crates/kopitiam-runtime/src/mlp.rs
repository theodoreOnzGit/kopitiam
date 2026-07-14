//! The SwiGLU feed-forward block (Shazeer, 2020, "GLU Variants Improve
//! Transformer").
//!
//! # Why three weight matrices, not two
//!
//! A classic transformer FFN is `down(activation(up(x)))` — two matrices.
//! SwiGLU replaces the single "up" projection with a *gated* pair: one
//! projection (`gate`) is passed through SiLU and used to gate a second,
//! separate projection (`up`) before the down-projection:
//!
//! ```text
//! swiglu(x) = down(silu(gate(x)) * up(x))
//! ```
//!
//! Every current Qwen/LLaMA GGUF export ships exactly these three matrices
//! per layer (`ffn_gate`, `ffn_up`, `ffn_down`), so treating this as "the
//! MLP with three weights" rather than trying to force it through a
//! two-matrix abstraction is what keeps [`crate::weights::LayerWeights`]
//! honest about what the file actually contains.

use kopitiam_core::Result;
use kopitiam_tensor::Tensor;

use crate::linear::linear;

/// Applies the SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
///
/// `x` is `[seq, hidden]`; `gate_weight`/`up_weight` are
/// `[ffn_hidden, hidden]` (GGUF's `nn.Linear`-style `[out, in]` layout —
/// see [`crate::linear::linear`]); `down_weight` is `[hidden, ffn_hidden]`.
/// Returns `[seq, hidden]`.
pub(crate) fn swiglu_mlp(x: &Tensor, gate_weight: &Tensor, up_weight: &Tensor, down_weight: &Tensor) -> Result<Tensor> {
    let gate = linear(x, gate_weight, None)?;
    let up = linear(x, up_weight, None)?;
    let gated = gate.silu()?.mul(&up)?;
    linear(&gated, down_weight, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    /// Hand-computed (independently, via Python) with deliberately
    /// *distinct* gate/up matrices, so a bug that swaps which projection
    /// feeds `silu` and which feeds the elementwise gate (`silu(up) * gate`
    /// instead of `silu(gate) * up`) would produce a different, wrong
    /// answer rather than accidentally matching by symmetry.
    #[test]
    fn swiglu_matches_hand_computation() {
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        // [out, in] layout, matching GGUF's on-disk convention.
        let gate_w = Tensor::from_f32(vec![2.0, 0.0, 0.0, 3.0], [2, 2]).unwrap();
        let up_w = Tensor::from_f32(vec![1.0, 1.0, 1.0, -1.0], [2, 2]).unwrap();
        let down_w = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap(); // identity

        let out = swiglu_mlp(&x, &gate_w, &up_w, &down_w).unwrap().to_vec_f32().unwrap();

        // gate = [2, 6], up = [3, -1]
        // silu(2) = 1.7615942, silu(6) = 5.9851643
        // h = [silu(2)*3, silu(6)*-1] = [5.2847826, -5.9851643]
        // down = identity, so out = h.
        assert_close(out[0], 5.284_782_5);
        assert_close(out[1], -5.985_164);
    }

    #[test]
    fn a_zero_gate_projection_zeroes_the_whole_block() {
        // silu(0) = 0, so gate(x) = 0 forces the elementwise product to 0
        // regardless of what up(x) computes, and down(0) = 0.
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let zero_gate = Tensor::from_f32(vec![0.0; 4], [2, 2]).unwrap();
        let up_w = Tensor::from_f32(vec![1.0, 1.0, 1.0, -1.0], [2, 2]).unwrap();
        let down_w = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap();

        let out = swiglu_mlp(&x, &zero_gate, &up_w, &down_w).unwrap().to_vec_f32().unwrap();
        assert_close(out[0], 0.0);
        assert_close(out[1], 0.0);
    }
}
