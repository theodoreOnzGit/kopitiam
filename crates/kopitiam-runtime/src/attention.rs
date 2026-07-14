//! Grouped-query causal self-attention.
//!
//! # Grouped-query attention (GQA), in one picture
//!
//! Ordinary multi-head attention gives every query head its own key/value
//! head: `n_heads` queries, `n_heads` keys, `n_heads` values. GQA instead
//! gives every *group* of `n_heads / n_kv_heads` query heads a single
//! *shared* key/value head — Qwen2's actual configuration. Skipping this
//! (silently treating `n_kv_heads` as if it equalled `n_heads`, or vice
//! versa) does not fail to load: the tensor shapes for `n_kv_heads == 1`
//! group of size `n_heads` and for `n_kv_heads == n_heads` are both
//! internally consistent, just wrong, so a model that never checks this
//! attends to the wrong keys for every group but 0 and produces
//! fluent-looking garbage. [`repeat_kv_heads`] is the one place that
//! expansion happens; see its docs for the exact head-to-group mapping and
//! [`crate::attention::tests::repeat_kv_heads_makes_grouped_query_heads_genuinely_share_kv`]
//! for the test that pins down "genuinely shares", not merely "same shape".
//!
//! # Causal masking
//!
//! A query at absolute position `p` may attend to keys at positions
//! `0..=p`, never `p+1..`. [`causal_mask`] returns an additive bias tensor
//! (`0` where attention is allowed, `-inf` where it is not) sized for
//! however many *new* query positions this forward pass covers against
//! however many keys are in the cache after this pass's K/V were appended
//! — which lets the exact same mask-and-softmax code serve both a
//! multi-token prompt prefill (`seq_q == seq_kv`, `position_offset == 0`)
//! and a single-token KV-cache decode step (`seq_q == 1`,
//! `seq_kv == cache_len_before + 1`, `position_offset == cache_len_before`)
//! without a separate code path for either.

use kopitiam_core::Result;
use kopitiam_tensor::Tensor;

use crate::kv_cache::KvCache;
use crate::linear::linear;
use crate::rope::RotaryEmbedding;
use crate::weights::LayerWeights;

/// Repeats each of `x`'s `n_kv_heads` (`x` is `[n_kv_heads, seq, head_dim]`)
/// `group_size` times, producing `[n_kv_heads * group_size, seq, head_dim]`.
///
/// Query head `g` (`0`-indexed) reads key/value head `g / group_size` —
/// i.e. heads `0..group_size` share KV head `0`, heads
/// `group_size..2*group_size` share KV head `1`, and so on. That is exactly
/// what this function produces: repeating each KV head's data
/// `group_size` times *contiguously* before moving to the next KV head
/// (via a broadcast over a new size-`group_size` axis immediately after the
/// head axis, then a reshape that collapses the two into one) lines up with
/// query heads laid out the same way by [`attention_forward`]'s own
/// `reshape([seq, n_heads, head_dim])` — both derive their head ordering
/// from the same "heads = a reshape of a flat `hidden` axis" convention, so
/// group `g`'s query heads and repeated-KV head `g` agree on which
/// original KV head they came from.
///
/// `group_size == 1` (ordinary multi-head attention, no GQA) returns `x`
/// unchanged via a cheap `Arc` clone.
pub(crate) fn repeat_kv_heads(x: &Tensor, group_size: usize) -> Result<Tensor> {
    if group_size == 1 {
        return Ok(x.clone());
    }
    let dims = x.shape().dims();
    let (n_kv, seq, head_dim) = (dims[0], dims[1], dims[2]);
    x.reshape([n_kv, 1, seq, head_dim])?
        .broadcast_to([n_kv, group_size, seq, head_dim])?
        .reshape([n_kv * group_size, seq, head_dim])
}

/// An additive attention bias, shape `[1, seq_q, seq_kv]` (broadcasts over
/// the head axis), that is `0.0` where a query may attend to a key and
/// `f32::NEG_INFINITY` where it may not.
///
/// Query row `i` (`0`-indexed within this forward pass) sits at absolute
/// position `position_offset + i`; it may attend to key column `j`
/// (absolute position `j`, since the key axis always starts at position 0
/// of the whole cached sequence) exactly when `j <= position_offset + i`.
pub(crate) fn causal_mask(seq_q: usize, seq_kv: usize, position_offset: usize) -> Tensor {
    let mut data = vec![0f32; seq_q * seq_kv];
    for i in 0..seq_q {
        let last_visible = position_offset + i;
        for (j, cell) in data[i * seq_kv..(i + 1) * seq_kv].iter_mut().enumerate() {
            if j > last_visible {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_f32(data, [1, seq_q, seq_kv]).expect("mask data length matches its own shape by construction")
}

/// One layer's grouped-query causal self-attention, including the RoPE
/// application and the KV-cache read/append.
///
/// `x` is `[seq, hidden]` — this layer's already RMS-normalized input (see
/// [`crate::block`]). `position_offset` is the absolute sequence position
/// of `x`'s first row — the cache's length *before this entire forward
/// pass*, not before this specific layer's append.
///
/// # Why `position_offset` is a parameter here, not `cache.len()`
///
/// [`KvCache::len`] reports layer *0*'s cached length specifically (see
/// that method's docs on the "same for every layer" invariant). Within one
/// [`crate::model::QwenModel::forward`] call, every layer's
/// [`attention_forward`] runs in sequence, and layer 0's call already
/// appends to layer 0's cache before layer 1 ever runs — so a naive
/// `cache.len()` call made *inside* this function would read the
/// already-updated layer-0 length while computing layer 1's positions,
/// silently off-by-`seq` for every layer after the first. This was a real
/// bug caught by
/// `crate::model::tests::decoding_with_a_kv_cache_matches_a_full_forward_pass_without_one_bit_for_bit`
/// during development: it manifested as small but nonzero logit
/// differences between step-by-step decoding and a single full-sequence
/// forward pass, exactly the class of "looks almost right" error a KV
/// cache invites. The fix is architectural, not a cache-internals patch:
/// the caller ([`crate::model::QwenModel::forward`]) reads `cache.len()`
/// exactly once, before touching any layer, and threads that one value
/// through every layer's [`attention_forward`] call.
///
/// Returns `[seq, hidden]`, the attention output *before* the residual add
/// (the caller adds it back onto the un-normalized residual stream).
#[allow(clippy::too_many_arguments)] // one argument per genuinely distinct piece of per-layer state; see block::block_forward's identical rationale.
pub(crate) fn attention_forward(
    x: &Tensor,
    weights: &LayerWeights,
    rope: &RotaryEmbedding,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    layer_index: usize,
    position_offset: usize,
    cache: &mut KvCache,
) -> Result<Tensor> {
    let seq = x.shape().dims()[0];
    let positions: Vec<usize> = (position_offset..position_offset + seq).collect();

    // Project, then reshape [seq, n*head_dim] -> [seq, n, head_dim] ->
    // [n, seq, head_dim] so the head axis is the batch axis every
    // subsequent batched matmul (Tensor::matmul broadcasts leading "batch"
    // dims) treats independently.
    let q = linear(x, &weights.wq, weights.bq.as_ref())?.reshape([seq, n_heads, head_dim])?.transpose(0, 1)?;
    let k = linear(x, &weights.wk, weights.bk.as_ref())?.reshape([seq, n_kv_heads, head_dim])?.transpose(0, 1)?;
    let v = linear(x, &weights.wv, weights.bv.as_ref())?.reshape([seq, n_kv_heads, head_dim])?.transpose(0, 1)?;

    let q = rope.apply(&q, &positions)?;
    let k = rope.apply(&k, &positions)?;

    let (k_full, v_full) = cache.append(layer_index, k, v)?;
    let seq_kv = k_full.shape().dims()[1];

    let group_size = n_heads / n_kv_heads;
    let k_rep = repeat_kv_heads(&k_full, group_size)?;
    let v_rep = repeat_kv_heads(&v_full, group_size)?;

    let scale = Tensor::from_f32(vec![(head_dim as f32).sqrt()], []).expect("scalar tensor");
    let scores = q.matmul(&k_rep.transpose(1, 2)?)?.div(&scale)?; // [n_heads, seq, seq_kv]

    let mask = causal_mask(seq, seq_kv, position_offset);
    let scores = scores.add(&mask)?;
    let probs = scores.softmax(2)?;

    let attn_out = probs.matmul(&v_rep)?; // [n_heads, seq, head_dim]
    let attn_out = attn_out.transpose(0, 1)?.reshape([seq, n_heads * head_dim])?;

    linear(&attn_out, &weights.wo, None)
}

/// Sanity check the loader/config layer relies on but this module does not
/// itself enforce (callers, i.e. [`crate::config::QwenConfig::from_metadata`],
/// already reject a non-dividing `n_kv_heads` before any of this runs) —
/// kept here only so a future direct caller of [`repeat_kv_heads`] gets a
/// clear panic message instead of a confusing shape error deep inside
/// `reshape`.
#[allow(dead_code)]
fn debug_assert_divides(n_heads: usize, n_kv_heads: usize) {
    debug_assert!(n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads));
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_core::DType;

    #[test]
    fn repeat_kv_heads_makes_grouped_query_heads_genuinely_share_kv() {
        // n_kv_heads = 2, group_size = 2 -> n_heads = 4. KV head 0 has all
        // 1.0s, KV head 1 has all 2.0s, seq=1, head_dim=1 for readability.
        let kv = Tensor::from_f32(vec![1.0, 2.0], [2, 1, 1]).unwrap();
        let expanded = repeat_kv_heads(&kv, 2).unwrap();
        assert_eq!(expanded.shape().dims(), &[4, 1, 1]);
        let data = expanded.to_vec_f32().unwrap();

        // Heads 0 and 1 must see the SAME (KV head 0's) data...
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 1.0);
        // ...and heads 2 and 3 the same (KV head 1's) data...
        assert_eq!(data[2], 2.0);
        assert_eq!(data[3], 2.0);
        // ...which is the property that distinguishes genuine GQA sharing
        // from an (incorrect) implementation that silently produces 4
        // independent-looking values as though n_kv_heads == n_heads: this
        // asserts head 0 and head 1 are not just shaped alike but hold the
        // literal same values, and likewise for 2 and 3.
        assert_eq!(data[0], data[1]);
        assert_eq!(data[2], data[3]);
        assert_ne!(data[0], data[2]);
    }

    #[test]
    fn repeat_kv_heads_group_size_one_is_the_identity() {
        let kv = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [2, 2, 1]).unwrap();
        let same = repeat_kv_heads(&kv, 1).unwrap();
        assert_eq!(same.to_vec_f32().unwrap(), kv.to_vec_f32().unwrap());
    }

    #[test]
    fn causal_mask_blocks_only_strictly_future_positions() {
        // seq_q = seq_kv = 3, position_offset = 0: ordinary prefill mask.
        let mask = causal_mask(3, 3, 0).to_vec_f32().unwrap();
        // Row 0 (position 0): only column 0 visible.
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite() && mask[1] < 0.0);
        assert!(mask[2].is_infinite() && mask[2] < 0.0);
        // Row 1 (position 1): columns 0 and 1 visible, 2 masked.
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[4], 0.0);
        assert!(mask[5].is_infinite() && mask[5] < 0.0);
        // Row 2 (position 2): everything visible.
        assert_eq!(mask[6], 0.0);
        assert_eq!(mask[7], 0.0);
        assert_eq!(mask[8], 0.0);
    }

    #[test]
    fn causal_mask_with_a_position_offset_accounts_for_already_cached_positions() {
        // One new query token (seq_q=1) attending over a cache that
        // already holds 3 positions, so seq_kv=4 and position_offset=3:
        // the new token is at absolute position 3 and may see keys 0..=3.
        let mask = causal_mask(1, 4, 3).to_vec_f32().unwrap();
        assert_eq!(mask, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_softmax_assigns_exactly_zero_weight_to_future_positions() {
        let scores = Tensor::from_f32(vec![5.0, 5.0, 5.0], [1, 1, 3]).unwrap();
        let mask = causal_mask(1, 3, 0); // position 0: only column 0 visible.
        let masked = scores.add(&mask).unwrap();
        let probs = masked.softmax(2).unwrap().to_vec_f32().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-6);
        assert_eq!(probs[1], 0.0);
        assert_eq!(probs[2], 0.0);
    }

    /// Not `f32::MIN` or a large negative number: an actually-infinite mask
    /// entry is what guarantees `exp(masked_score) == 0.0` exactly rather
    /// than merely "very small", however large the unmasked scores get.
    #[test]
    fn masked_logits_are_exactly_negative_infinity() {
        let mask = causal_mask(2, 2, 0).to_vec_f32().unwrap();
        assert_eq!(mask[1], f32::NEG_INFINITY);
    }

    fn dummy_weights(hidden: usize, kv_dim: usize) -> LayerWeights {
        let eye = |n: usize| Tensor::from_f32((0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect(), [n, n]).unwrap();
        LayerWeights {
            attn_norm: Tensor::from_f32(vec![1.0; hidden], [hidden]).unwrap(),
            wq: eye(hidden),
            bq: None,
            wk: Tensor::from_f32(vec![0.0; kv_dim * hidden], [kv_dim, hidden]).unwrap(),
            bk: None,
            wv: Tensor::from_f32(vec![0.0; kv_dim * hidden], [kv_dim, hidden]).unwrap(),
            bv: None,
            wo: eye(hidden),
            ffn_norm: Tensor::from_f32(vec![1.0; hidden], [hidden]).unwrap(),
            w_gate: eye(hidden),
            w_up: eye(hidden),
            w_down: eye(hidden),
        }
    }

    /// A minimal end-to-end exercise of [`attention_forward`] itself
    /// (rather than its sub-pieces): with all-zero K/V weights, every
    /// query attends to all-zero keys, softmax is therefore uniform over
    /// whatever is causally visible, and V is all zero, so the attention
    /// output before the output projection is exactly zero everywhere —
    /// a cheap end-to-end sanity check that the plumbing (reshape/
    /// transpose axis order, RoPE application, mask application, KV-cache
    /// append) produces the right *shape* and a value that is easy to
    /// predict by hand.
    #[test]
    fn attention_forward_with_zeroed_kv_weights_produces_zero_output() {
        let hidden = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = hidden / n_heads;
        let weights = dummy_weights(hidden, n_kv_heads * head_dim);
        let rope = RotaryEmbedding::new(head_dim, 10_000.0, 16);
        let mut cache = KvCache::new(1, 16);

        let x = Tensor::from_f32((0..2 * hidden).map(|i| i as f32 * 0.1).collect(), [2, hidden]).unwrap();
        let out = attention_forward(&x, &weights, &rope, n_heads, n_kv_heads, head_dim, 0, 0, &mut cache).unwrap();
        assert_eq!(out.shape().dims(), &[2, hidden]);
        for v in out.to_vec_f32().unwrap() {
            assert_eq!(v, 0.0);
        }
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn attention_forward_rejects_non_f32_input_the_same_way_matmul_does() {
        // hidden = 32 so a one-block Q4_0 tensor ([1, 32], 32 elements) is
        // constructible at all -- the point of this test is exercising
        // error *propagation* out of attention_forward, not quantized
        // tensor construction.
        let hidden = 32;
        let weights = dummy_weights(hidden, hidden);
        let rope = RotaryEmbedding::new(hidden, 10_000.0, 16);
        let mut cache = KvCache::new(1, 16);
        let x = Tensor::from_quantized(DType::Q4_0, vec![0u8; 18], [1, hidden]).unwrap();
        assert!(attention_forward(&x, &weights, &rope, 1, 1, hidden, 0, 0, &mut cache).is_err());
    }
}
