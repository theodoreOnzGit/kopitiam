//! One pre-norm transformer block: `x + Attn(RMSNorm(x))`, then
//! `x + MLP(RMSNorm(x))`.
//!
//! This is the "pre-norm" residual arrangement every current LLaMA/Qwen
//! checkpoint uses (normalize *before* the sub-layer, not after) — as
//! opposed to the original Transformer paper's post-norm. Getting the
//! order backwards (`RMSNorm(x + Attn(x))`) still type-checks and still
//! runs; it just does not match how the weights were trained, so — like
//! the RoPE convention and the GQA head mapping — this is a silent
//! correctness bug rather than a crash, which is why it gets its own named
//! function with a doc comment pinning down the order, instead of being
//! inlined into [`crate::model::QwenModel::forward`]'s loop.

use kopitiam_core::Result;
use kopitiam_tensor::Tensor;

use crate::attention::attention_forward;
use crate::kv_cache::KvCache;
use crate::mlp::swiglu_mlp;
use crate::rope::RotaryEmbedding;
use crate::weights::LayerWeights;

/// Runs one transformer block over `x` (`[seq, hidden]`), returning the
/// updated residual stream, also `[seq, hidden]`.
///
/// `position_offset` is the absolute sequence position of `x`'s first row,
/// computed *once* by the caller ([`crate::model::QwenModel::forward`])
/// before the layer loop starts and passed unchanged to every layer — see
/// [`crate::attention::attention_forward`]'s docs for why re-deriving it
/// per layer from `cache.len()` is a KV-cache correctness bug, not a
/// harmless simplification.
#[allow(clippy::too_many_arguments)] // one argument per genuinely distinct piece of per-layer state; bundling them into a struct would not reduce what the caller has to know, only rename it.
pub(crate) fn block_forward(
    x: &Tensor,
    weights: &LayerWeights,
    rope: &RotaryEmbedding,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    norm_eps: f32,
    layer_index: usize,
    position_offset: usize,
    cache: &mut KvCache,
) -> Result<Tensor> {
    let normed = x.rms_norm(&weights.attn_norm, norm_eps)?;
    let attn_out =
        attention_forward(&normed, weights, rope, n_heads, n_kv_heads, head_dim, layer_index, position_offset, cache)?;
    let x = x.add(&attn_out)?;

    let normed = x.rms_norm(&weights.ffn_norm, norm_eps)?;
    let mlp_out = swiglu_mlp(&normed, &weights.w_gate, &weights.w_up, &weights.w_down)?;
    x.add(&mlp_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(n: usize) -> Tensor {
        Tensor::from_f32((0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect(), [n, n]).unwrap()
    }

    /// With every weight matrix set to identity and every norm weight set
    /// to `1.0`, a block still isn't the identity function (RMSNorm
    /// rescales, attention mixes across positions, SiLU is nonlinear), but
    /// it must still produce a *finite*, correctly-shaped result and must
    /// still add the residual — i.e. output must differ from the
    /// zero-weight-attention case where the whole sub-layer contributes
    /// nothing. This is a wiring smoke test, not a numerical one; block.rs
    /// has no arithmetic of its own; attention_forward/swiglu_mlp/rms_norm
    /// each carry their own hand-computed tests.
    #[test]
    fn block_forward_produces_a_finite_correctly_shaped_residual_update() {
        let hidden = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = hidden / n_heads;
        let weights = LayerWeights {
            attn_norm: Tensor::from_f32(vec![1.0; hidden], [hidden]).unwrap(),
            wq: eye(hidden),
            bq: None,
            wk: Tensor::from_f32(vec![0.1; n_kv_heads * head_dim * hidden], [n_kv_heads * head_dim, hidden]).unwrap(),
            bk: None,
            wv: Tensor::from_f32(vec![0.1; n_kv_heads * head_dim * hidden], [n_kv_heads * head_dim, hidden]).unwrap(),
            bv: None,
            wo: eye(hidden),
            ffn_norm: Tensor::from_f32(vec![1.0; hidden], [hidden]).unwrap(),
            w_gate: eye(hidden),
            w_up: eye(hidden),
            w_down: eye(hidden),
        };
        let rope = RotaryEmbedding::new(head_dim, 10_000.0, 16);
        let mut cache = KvCache::new(1, 16);

        let x = Tensor::from_f32(vec![0.5, -0.3, 0.8, 0.1, 0.2, 0.4, -0.6, 0.9], [2, hidden]).unwrap();
        let out = block_forward(&x, &weights, &rope, n_heads, n_kv_heads, head_dim, 1e-6, 0, 0, &mut cache).unwrap();

        assert_eq!(out.shape().dims(), &[2, hidden]);
        for v in out.to_vec_f32().unwrap() {
            assert!(v.is_finite());
        }
        // The block must have changed x, not returned it unmodified.
        assert_ne!(out.to_vec_f32().unwrap(), x.to_vec_f32().unwrap());
    }
}
