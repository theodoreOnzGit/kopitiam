//! Matrix multiplication: the hot path of every transformer layer
//! (attention projections, the FFN's two big matmuls) and therefore the one
//! op in this crate most worth getting right before getting fast.
//!
//! Phase 1 targets *correct* `f32` matmul; a blocked/SIMD/threaded kernel
//! is a follow-up once there is a real model to benchmark against (see the
//! crate docs' "what's deliberately not here" list).

use kopitiam_core::{DType, Error, Result, Shape};

use crate::quant;
use crate::storage::Storage;

use super::Tensor;

/// Whether [`Tensor::quantized_matmul`] has a *fused*, dequantize-free kernel
/// for `dtype` in this build — the single source of truth every layer that
/// must choose "compute on the quantized bytes directly" vs. "dequantize to
/// `f32` first" shares, so the two decisions can never drift apart.
///
/// Today [`DType::Q4_0`], [`DType::Q8_0`], [`DType::Q4_K`] and
/// [`DType::Q6_K`] have such a kernel (the integer per-block dot products in
/// [`crate::quant`]). The remaining quantized formats —
/// `Q4_1`/`Q5_0`/`Q5_1`, and the K-quants `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K` —
/// dequantize correctly via [`Tensor::to_dtype`] but have no fused path here,
/// so a caller wanting to matmul against one must dequantize it to `f32` and
/// take the ordinary [`Tensor::matmul`] route.
///
/// # Why Q4_K and Q6_K specifically, and not the rest of the K-quants
///
/// Those two are exactly what a `Q4_K_M` file is made of — every matmul
/// weight in one is Q4_K or Q6_K (SmolLM2-1.7B-Instruct-Q4_K_M: 144 Q4_K
/// tensors, 25 Q6_K, 49 `f32` norms/biases, and nothing else). Leaving them
/// unfused made `Q4_K_M` the one real-world configuration where *nothing*
/// stayed quantized, so a 1.06 GB file expanded to ~6.8 GB of `f32` at load
/// and the model became too slow to finish an answer. See
/// [`Tensor::quantized_matmul`]'s "Why this is not just a speed optimisation"
/// for the measurement. The other K-quants are not absent for any deep
/// reason — no model the maintainer runs ships them as matmul weights yet, so
/// adding them would be untested code.
///
/// # Why this is a predicate, not a hard-coded match in each caller
///
/// Three places must agree on this set: [`Tensor::quantized_matmul`] (which
/// rejects a weight it cannot fuse), `kopitiam-runtime`'s `load_matmul_weight`
/// (which keeps a weight quantized only when it can be fused, and otherwise
/// dequantizes it at load so the forward pass never dead-ends in an
/// `UnsupportedDType`), and that crate's `linear` dispatch. If any of them
/// spelled the set out independently, adding a format to one and forgetting
/// another would silently reintroduce the load-time/compute-time mismatch this
/// function exists to prevent.
///
/// # Why it lives here and not on [`DType`]
///
/// "Which formats have a fused kernel" is a fact about *this crate's* kernels,
/// not about what the bytes mean — [`DType`] in `kopitiam-core` deliberately
/// says nothing about how a format is computed (see its docs). A future
/// wgpu/GPU backend may fuse more formats (or a CPU kernel here may be replaced
/// by one), and when it does, only this crate changes; `kopitiam-core` stays a
/// pure description of the byte layout.
pub const fn has_fused_matmul_kernel(dtype: DType) -> bool {
    matches!(dtype, DType::Q4_0 | DType::Q8_0 | DType::Q4_K | DType::Q6_K)
}

impl Tensor {
    /// Batched matrix multiplication: `self` is `[..batch, m, k]`, `other`
    /// is `[..batch, k, n]`, and the result is `[..batch, m, n]`.
    ///
    /// Both operands must be rank >= 2 and `f32`. The leading ("batch")
    /// dimensions are broadcast against each other with the same
    /// right-aligned rule as [`kopitiam_core::Shape::broadcast`] (so a
    /// single `[k, n]` weight matrix multiplies against a `[batch, m, k]`
    /// activation without the caller manually replicating it) — this is
    /// what makes a single implementation handle both the plain 2D case
    /// (empty batch shape) and the batched case identically, rather than
    /// special-casing 2D.
    ///
    /// # Errors
    ///
    /// Shape problems (`rank < 2`, or the inner dimension `k` disagreeing
    /// between operands) are reported as [`Error::ShapeMismatch`]. That
    /// variant's field names (`expected`/`actual`) were designed for
    /// reshape; here they are reused to mean "these two shapes are not
    /// compatible for matrix multiplication" — the closest fit among
    /// `kopitiam-core`'s error variants, which this crate cannot modify.
    /// Incompatible batch dimensions surface as
    /// [`Error::NotBroadcastable`], propagated directly from
    /// [`Shape::broadcast`].
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        other.require_dtype(DType::F32)?;

        let (ra, rb) = (self.rank(), other.rank());
        if ra < 2 || rb < 2 {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        let (m, k_a) = (self.shape.dims()[ra - 2], self.shape.dims()[ra - 1]);
        let (k_b, n) = (other.shape.dims()[rb - 2], other.shape.dims()[rb - 1]);
        if k_a != k_b {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        let batch_a = Shape::new(self.shape.dims()[..ra - 2].to_vec());
        let batch_b = Shape::new(other.shape.dims()[..rb - 2].to_vec());
        let batch = batch_a.broadcast(&batch_b)?;
        let batch_count = batch.elem_count();

        let a_full_shape = Shape::new([batch.dims(), &[m, k_a]].concat());
        let b_full_shape = Shape::new([batch.dims(), &[k_b, n]].concat());
        let a_view = self.broadcast_to(a_full_shape)?;
        let b_view = other.broadcast_to(b_full_shape)?;

        // Materialize both operands contiguously in [batch, m, k] / [batch,
        // k, n] order so the inner loop can use plain slice indexing
        // instead of re-deriving strided offsets per element.
        let Storage::F32(a_raw) = a_view.storage.as_ref() else { unreachable!() };
        let Storage::F32(b_raw) = b_view.storage.as_ref() else { unreachable!() };
        let a_mat: Vec<f32> = a_view.logical_offsets().map(|i| a_raw[i]).collect();
        let b_mat: Vec<f32> = b_view.logical_offsets().map(|i| b_raw[i]).collect();

        let mut out = vec![0f32; batch_count * m * n];
        for batch_idx in 0..batch_count {
            let a_off = batch_idx * m * k_a;
            let b_off = batch_idx * k_a * n;
            let o_off = batch_idx * m * n;
            // i-k-j loop order: the inner loop walks both `b_mat` and `out`
            // contiguously, which is friendlier to the cache than the
            // naive i-j-k order that strides through `b_mat` by `n` on
            // every multiply-add.
            for i in 0..m {
                for p in 0..k_a {
                    let a_val = a_mat[a_off + i * k_a + p];
                    if a_val == 0.0 {
                        continue;
                    }
                    let b_row = &b_mat[b_off + p * n..b_off + p * n + n];
                    let out_row = &mut out[o_off + i * n..o_off + i * n + n];
                    for (o, &b_val) in out_row.iter_mut().zip(b_row) {
                        *o += a_val * b_val;
                    }
                }
            }
        }

        let out_shape = Shape::new([batch.dims(), &[m, n]].concat());
        Tensor::from_f32(out, out_shape)
    }

    /// `self @ weight^T` — the same semantics [`crate::linear::linear`]'s
    /// (in `kopitiam-runtime`) transpose-then-[`Tensor::matmul`] gives an
    /// `f32` weight — but for a block-quantized `weight` ([`DType::Q4_0`]
    /// or [`DType::Q8_0`]), computed *without* ever dequantizing `weight`
    /// to `f32`.
    ///
    /// # The algorithm: quantize the activation, dot in integer space
    ///
    /// The "correct before fast" version of a quantized matmul is
    /// `weight.to_dtype(DType::F32)` once, then plain [`Tensor::matmul`] —
    /// that remains this crate's default and its permanent reference
    /// implementation (see "Correctness" below). It is also why a
    /// 7B-parameter Q4_0 model, genuinely about 4GB on disk, balloons to
    /// roughly 28GB of resident `f32` the instant it is loaded that way:
    /// dequantizing trades away the entire point of shipping a quantized
    /// model (fitting in memory) for arithmetic convenience.
    ///
    /// This method is the technique `ggml`/`llama.cpp` use instead — the
    /// reason quantized inference is *fast*, not merely *small* — and it
    /// never produces an `f32` copy of `weight` at all. Every row of
    /// `self` is quantized to Q8_0 blocks on the fly
    /// ([`quant::quantize_row_q8_0`]), and each output element becomes a
    /// per-block *integer* dot product ([`quant::q4_0_dot_q8_0`] /
    /// [`quant::q8_0_dot_q8_0`]) between that freshly-quantized activation
    /// block and `weight`'s already-quantized block, with both sides'
    /// scales multiplied in once at the very end of each block. `weight`'s
    /// bytes are read exactly as they sit in [`Storage::Quantized`] —
    /// nothing about `weight` is ever expanded to a wider type. This is a
    /// genuinely different algorithm from "dequantize then multiply", not
    /// an optimization of it: the former spends `2 * out_features *
    /// in_features` `f32` multiply-adds; this spends that many `i32`
    /// multiply-adds (integer arithmetic a CPU — including a phone-class
    /// one — does at several times `f32`'s throughput) plus only `2 *
    /// out_features * (in_features / 32)` `f32` multiplies for the
    /// per-block scale combination.
    ///
    /// `self` is `[..., in_features]`, `f32`, any rank >= 1; `weight` is
    /// `[out_features, in_features]`, exactly rank 2, quantized. Returns
    /// `[..., out_features]`. `in_features` must be a whole number of
    /// `weight`'s own blocks — 32 for Q4_0/Q8_0, **256** for the Q4_K/Q6_K
    /// super-block formats — which is true of every real GGUF export, since a
    /// block may never straddle a row boundary.
    ///
    /// # Why this is not just a speed optimisation
    ///
    /// It is what keeps a model in RAM at all. `kopitiam-runtime`'s
    /// `load_matmul_weight` dequantizes any weight this method cannot fuse, so
    /// an unfused format is not merely a slow path — it is a **~5x memory
    /// expansion at load**, and past a certain model size that expansion is
    /// what kills you, not the arithmetic.
    ///
    /// MEASURED on the maintainer's 15 GB Windows box, SmolLM2-1.7B-Instruct-
    /// Q4_K_M (1.06 GB on disk), prompt "What is the capital of France?",
    /// same binary, toggled only by `KOPITIAM_FORCE_F32_WEIGHTS` (which
    /// reproduces the pre-fix behaviour exactly — every matmul weight in this
    /// file is Q4_K or Q6_K, so "no fused kernel" and "force f32" are the same
    /// thing here):
    ///
    /// ```text
    ///                              wall     peak private commit
    ///   dequantized f32          392.9s                 6.82 GB
    ///   fused K-quant kernels     45.2s                 1.23 GB
    ///   llama.cpp, same file       4.0s                       -
    /// ```
    ///
    /// The very first observation of this was worse still — 19m04s — because
    /// only 0.42 GB was free at the time and the 6.8 GB of `f32` had to page
    /// to disk. Same work, 3x the wall clock, purely from not fitting.
    ///
    /// The output was never *wrong* at any point: all three rows above answer
    /// "The capital of France is Paris." It was unreachably slow, and the
    /// maintainer — watching a token-by-token stream — saw the first token
    /// appear and reasonably reported it as "emits one token then stops". That
    /// is the failure mode an unfused format actually produces, and it is why
    /// [`has_fused_matmul_kernel`] covering the formats a real file ships
    /// matters more than covering them elegantly.
    ///
    /// # Correctness
    ///
    /// This method's own tests assert, for both Q4_0 and Q8_0, that its
    /// output agrees with `weight.to_dtype(DType::F32)` followed by
    /// [`Tensor::matmul`] within the error the *activation's* Q8_0
    /// quantization alone can introduce (both paths read the identical
    /// weight bits, so any difference comes only from `self` going through
    /// [`quant::quantize_row_q8_0`] on this path and not on the reference
    /// path) — see this module's `quantized_matmul_agrees_with_...` tests.
    /// That reference path is why dequantizing to `f32` stays in this
    /// crate permanently: it is not just a fallback, it is the oracle this
    /// method is checked against.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] if `self` is not `f32`.
    /// [`Error::UnsupportedDType`] if `weight`'s dtype is not one of the
    /// formats this method has a fused kernel for (see
    /// [`has_fused_matmul_kernel`]). The remaining quantized formats —
    /// `Q4_1`/`Q5_0`/`Q5_1` and the K-quants `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K` —
    /// dequantize fine via [`Tensor::to_dtype`] but have no fused path here
    /// yet, so a caller wanting to matmul against one must dequantize it to
    /// `f32` first. Also errors if `weight` is not quantized at all (use
    /// [`Tensor::matmul`] directly for a plain `f32` weight).
    /// [`Error::ShapeMismatch`] if `weight` is not rank 2, if `self`'s last
    /// dimension does not equal `weight`'s `in_features`, or if
    /// `in_features` is not a multiple of `weight`'s block size (32, or 256
    /// for a K-quant).
    pub fn quantized_matmul(&self, weight: &Tensor) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        if !has_fused_matmul_kernel(weight.dtype()) {
            return Err(Error::UnsupportedDType { op: "quantized_matmul", dtype: weight.dtype() });
        }
        if weight.rank() != 2 {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: weight.shape.clone() });
        }
        let wdims = weight.shape.dims();
        let (out_features, in_features) = (wdims[0], wdims[1]);

        let Storage::Quantized { dtype, bytes } = weight.storage.as_ref() else {
            unreachable!("has_fused_matmul_kernel is true only for quantized dtypes, so the guard above guarantees Storage::Quantized")
        };
        let dtype = *dtype;
        // The weight's own block length: 32 for the classic formats, 256 for a
        // K-quant super-block. `in_features` must be a whole number of THESE,
        // not of 32 — a K-quant row that is a multiple of 32 but not of 256
        // would slice super-blocks in half.
        let weight_block_size = dtype.block_size();

        let xdims = self.shape.dims();
        let Some((&last, leading)) = xdims.split_last() else {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: weight.shape.clone() });
        };
        if last != in_features
            || in_features == 0
            || !in_features.is_multiple_of(weight_block_size)
        {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: weight.shape.clone() });
        }

        // logical_offsets()-ordered, so this is correct for a non-contiguous
        // `self` (a transposed or narrowed activation view) too, not just
        // the common contiguous case.
        let x_data = self.to_vec_f32()?;

        let block_bytes = dtype.block_bytes();
        let blocks_per_row = in_features / weight_block_size;
        let row_bytes = blocks_per_row * block_bytes;
        // Activations are always Q8_0 (32 per block), whatever the weight's
        // block length — so one weight block lines up with this many of them.
        // `quantize_row_q8_0` is the crate's single activation encoder; see
        // [`quant::q4_k_dot_q8_0`] on why the K-quant kernels consume Q8_0
        // rather than ggml's Q8_K.
        let x_blocks_per_weight_block = weight_block_size / 32;

        let rows = leading.iter().product::<usize>();
        let mut out = vec![0f32; rows * out_features];
        for r in 0..rows {
            let x_row = &x_data[r * in_features..(r + 1) * in_features];
            let (x_scales, x_q) = quant::quantize_row_q8_0(x_row);
            for o in 0..out_features {
                let w_row = &bytes[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0f32;
                for (b, w_block) in w_row.chunks_exact(block_bytes).enumerate() {
                    let xq_block = &x_q[b * weight_block_size..(b + 1) * weight_block_size];
                    acc += match dtype {
                        DType::Q4_0 => quant::q4_0_dot_q8_0(w_block, xq_block, x_scales[b]),
                        DType::Q8_0 => quant::q8_0_dot_q8_0(w_block, xq_block, x_scales[b]),
                        DType::Q4_K | DType::Q6_K => {
                            let s = &x_scales[b * x_blocks_per_weight_block
                                ..(b + 1) * x_blocks_per_weight_block];
                            if dtype == DType::Q4_K {
                                quant::q4_k_dot_q8_0(w_block, xq_block, s)
                            } else {
                                quant::q6_k_dot_q8_0(w_block, xq_block, s)
                            }
                        }
                        _ => unreachable!("has_fused_matmul_kernel admits only Q4_0, Q8_0, Q4_K and Q6_K"),
                    };
                }
                out[r * out_features + o] = acc;
            }
        }

        let mut out_shape = leading.to_vec();
        out_shape.push(out_features);
        Tensor::from_f32(out, out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x3_by_3x2_matches_hand_computation() {
        // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]]
        // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let b = Tensor::from_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.to_vec_f32().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_by_identity_is_the_identity() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let identity = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap();
        let c = a.matmul(&identity).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_rejects_incompatible_inner_dimensions() {
        let a = Tensor::from_f32(vec![1.0; 6], [2, 3]).unwrap();
        let b = Tensor::from_f32(vec![1.0; 8], [4, 2]).unwrap();
        assert!(matches!(a.matmul(&b), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn matmul_rejects_rank_1_operands() {
        let a = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        let b = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        assert!(matches!(a.matmul(&b), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn batched_matmul_broadcasts_a_shared_weight_matrix_over_the_batch() {
        // Two batches of a [1,2] "activation" times a shared [2,2] weight.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [2, 1, 2]).unwrap(); // [batch=2, m=1, k=2]
        let w = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap(); // [k=2, n=2], no batch dim
        let c = a.matmul(&w).unwrap();
        assert_eq!(c.shape().dims(), &[2, 1, 2]);
        // Multiplying by the identity should reproduce each batch's row.
        assert_eq!(c.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn batched_matmul_matches_per_batch_hand_computation() {
        // batch=2, m=2,k=2,n=2. Batch 0 = identity-ish, batch 1 = doubling.
        let a = Tensor::from_f32(
            vec![
                1.0, 2.0, 3.0, 4.0, // batch 0
                1.0, 0.0, 0.0, 1.0, // batch 1
            ],
            [2, 2, 2],
        )
        .unwrap();
        let b = Tensor::from_f32(
            vec![
                1.0, 0.0, 0.0, 1.0, // batch 0: identity
                2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
            ],
            [2, 2, 2],
        )
        .unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(
            c.to_vec_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0]
        );
    }

    #[test]
    fn matmul_only_accepts_f32() {
        let a = Tensor::from_i32(vec![1, 2, 3, 4], [2, 2]).unwrap();
        let b = Tensor::from_i32(vec![1, 2, 3, 4], [2, 2]).unwrap();
        assert!(matches!(a.matmul(&b), Err(Error::DTypeMismatch { .. })));
    }

    // -- quantized_matmul --

    /// A small, dependency-free deterministic PRNG (xorshift64*), purely for
    /// generating non-degenerate test data — the same pattern this
    /// workspace's other test fixtures use (see e.g.
    /// `kopitiam-runtime`'s `test_support::synthetic_gguf::Xorshift64`) so a
    /// fixed seed makes every test below reproducible without a `rand`
    /// dependency.
    fn random_f32s(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut state = seed | 1; // xorshift requires a nonzero state.
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as u32 as f32 / (1u32 << 24) as f32; // [0, 1)
                (unit - 0.5) * 2.0 * scale
            })
            .collect()
    }

    /// Encodes `values` (row-major, `shape = [rows, cols]`, `cols` a
    /// multiple of 32) as a Q8_0 tensor, matching [`crate::quant`]'s
    /// `qs[j] * d` decode formula exactly (this is a from-scratch,
    /// test-only encoder — `kopitiam-tensor` deliberately has no *public*
    /// f32 -> quantized encoder; see [`crate::quant::quantize_row_q8_0`]'s
    /// docs for why activation-only quantization is a narrower thing).
    fn encode_q8_0(values: &[f32], shape: [usize; 2]) -> Tensor {
        let mut bytes = Vec::with_capacity(values.len() / 32 * 34);
        for chunk in values.chunks_exact(32) {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            bytes.extend_from_slice(&crate::half::f32_to_f16(d).to_le_bytes());
            for &v in chunk {
                bytes.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        Tensor::from_quantized(DType::Q8_0, bytes, shape).unwrap()
    }

    /// Same idea as [`encode_q8_0`] but for Q4_0: the decode formula is
    /// `(nibble - 8) * d`, so the encoder picks `d = amax / 7` (the
    /// largest magnitude the signed-nibble range `[-8, 7]` can represent
    /// without saturating on the positive side) and packs two elements per
    /// byte exactly as [`crate::quant::dequant_q4_0`] expects: byte `j`
    /// holds elements `j` (low nibble) and `j + 16` (high nibble).
    fn encode_q4_0(values: &[f32], shape: [usize; 2]) -> Tensor {
        let mut bytes = Vec::with_capacity(values.len() / 32 * 18);
        for chunk in values.chunks_exact(32) {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / 7.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            bytes.extend_from_slice(&crate::half::f32_to_f16(d).to_le_bytes());
            let nibble = |v: f32| -> u8 { ((v * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8 & 0x0F };
            for j in 0..16 {
                bytes.push(nibble(chunk[j]) | (nibble(chunk[j + 16]) << 4));
            }
        }
        Tensor::from_quantized(DType::Q4_0, bytes, shape).unwrap()
    }

    /// `x @ weight^T` computed by dequantizing `weight` to `f32` first —
    /// the reference [`Tensor::quantized_matmul`] is checked against (see
    /// that method's "Correctness" docs).
    fn reference_linear(x: &Tensor, weight: &Tensor) -> Tensor {
        let weight_f32 = weight.to_dtype(DType::F32).unwrap();
        x.matmul(&weight_f32.transpose(0, 1).unwrap()).unwrap()
    }

    fn assert_allclose(actual: &[f32], expected: &[f32], rel: f32, abs_eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let tol = abs_eps + rel * e.abs();
            assert!((a - e).abs() <= tol, "index {i}: expected {e}, got {a} (tolerance {tol})");
        }
    }

    /// The exact worst-case bound on [`Tensor::quantized_matmul`]'s output
    /// error against [`reference_linear`]'s, derived from first principles
    /// rather than picked to make a test pass. Both paths read *identical*
    /// weight bits (`reference_linear` dequantizes the same already-quantized
    /// `weight` the fast path reads directly), so the only error source is
    /// the fast path's on-the-fly Q8_0 quantization of the activation row.
    /// Round-to-nearest guarantees every quantized activation element
    /// differs from its true value by at most half its block's step
    /// (`x_scale / 2`), so by the triangle inequality the whole row's dot
    /// product error is at most `sum_j |weight_j| * (x_scale_of_j's_block / 2)`
    /// — computed here per output row/column pair from the *dequantized*
    /// weight (what both paths agree the weight "means") and the real
    /// per-block Q8_0 scales [`quant::quantize_row_q8_0`] actually chose.
    /// A small additive slack accounts for ordinary `f32` summation-order
    /// noise (both paths sum 32-or-more terms in different groupings).
    fn max_activation_quantization_error(weight_row: &[f32], x_row: &[f32]) -> f32 {
        let (x_scales, _) = quant::quantize_row_q8_0(x_row);
        let bound: f32 = weight_row
            .chunks_exact(32)
            .zip(&x_scales)
            .map(|(w_block, &scale)| (scale / 2.0) * w_block.iter().map(|v| v.abs()).sum::<f32>())
            .sum();
        bound + 1e-3
    }

    /// Runs [`Tensor::quantized_matmul`] against [`reference_linear`] and
    /// checks every output element against
    /// [`max_activation_quantization_error`]'s bound, not a flat tolerance.
    fn assert_agrees_within_activation_quantization_bound(
        x_rows: &[Vec<f32>],
        weight: &Tensor,
        weight_rows_f32: &[Vec<f32>],
    ) {
        let in_features = x_rows[0].len();
        let x = Tensor::from_f32(x_rows.concat(), [x_rows.len(), in_features]).unwrap();
        let fast = x.quantized_matmul(weight).unwrap().to_vec_f32().unwrap();
        let reference = reference_linear(&x, weight).to_vec_f32().unwrap();

        let out_features = weight_rows_f32.len();
        for (idx, (&f, &r)) in fast.iter().zip(&reference).enumerate() {
            let row = idx / out_features;
            let col = idx % out_features;
            let bound = max_activation_quantization_error(&weight_rows_f32[col], &x_rows[row]);
            assert!(
                (f - r).abs() <= bound,
                "row {row} col {col}: expected {r}, got {f} (bound {bound})"
            );
        }
    }

    #[test]
    fn quantized_matmul_q8_0_agrees_with_dequantize_then_matmul_reference() {
        let (out_features, in_features) = (6, 64); // 2 blocks per row.
        let weight_vals = random_f32s(1, out_features * in_features, 3.0);
        let weight = encode_q8_0(&weight_vals, [out_features, in_features]);
        let weight_rows_f32: Vec<Vec<f32>> = weight
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32()
            .unwrap()
            .chunks_exact(in_features)
            .map(<[f32]>::to_vec)
            .collect();

        let x_rows: Vec<Vec<f32>> = (0..3).map(|r| random_f32s(2 + r as u64 * 100, in_features, 2.0)).collect();

        assert_agrees_within_activation_quantization_bound(&x_rows, &weight, &weight_rows_f32);
    }

    #[test]
    fn quantized_matmul_q4_0_agrees_with_dequantize_then_matmul_reference() {
        let (out_features, in_features) = (5, 96); // 3 blocks per row.
        let weight_vals = random_f32s(3, out_features * in_features, 3.0);
        let weight = encode_q4_0(&weight_vals, [out_features, in_features]);
        let weight_rows_f32: Vec<Vec<f32>> = weight
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32()
            .unwrap()
            .chunks_exact(in_features)
            .map(<[f32]>::to_vec)
            .collect();

        let x_rows: Vec<Vec<f32>> = (0..2).map(|r| random_f32s(4 + r as u64 * 100, in_features, 2.0)).collect();

        assert_agrees_within_activation_quantization_bound(&x_rows, &weight, &weight_rows_f32);
    }

    #[test]
    fn quantized_matmul_exact_when_activation_needs_no_rounding() {
        // A Q4_0 block built directly from chosen nibble bytes (not via
        // `encode_q4_0`'s amax-derived scale, which -- like any single
        // shared per-block scale -- exactly represents only the extreme
        // value and rounds every other one) so weight decoding is exact by
        // construction. The activation is likewise chosen to be an exact
        // multiple of its own Q8_0 step (amax = 127 -> d = 1.0, and every
        // value below is an integer). With both sources of rounding
        // eliminated, the fast and reference paths must agree exactly, not
        // just within a bound.
        let in_features = 32;
        let d_w = 1.0f32;
        let mut weight_bytes = crate::half::f32_to_f16(d_w).to_le_bytes().to_vec();
        for j in 0..16 {
            let lo = (j % 16) as u8; // nibble -> decoded value (j - 8) exactly.
            let hi = ((j + 3) % 16) as u8;
            weight_bytes.push(lo | (hi << 4));
        }
        let weight = Tensor::from_quantized(DType::Q4_0, weight_bytes, [1, in_features]).unwrap();

        // THE SUBTLETY THIS TEST EXISTS TO PIN, and which its first draft got
        // wrong: "integers, comfortably inside +/-127" is NOT enough to make the
        // activation round-free.
        //
        // Q8_0 derives its step from the block's own amax: `d = amax / 127`.
        // The step is 1.0 -- and integers are therefore exactly representable --
        // only when amax is EXACTLY 127. The original fixture used values
        // spanning -60..=64, giving amax = 64 and d = 64/127 ~= 0.504, under
        // which those integers are *not* multiples of the step at all. It then
        // asserted exactness to 1e-3 and failed by ~0.6%, which was Q8_0
        // rounding behaving perfectly correctly.
        //
        // So: integers, and amax pinned to exactly 127.
        let mut x_vals = [0f32; 32];
        for (j, v) in x_vals.iter_mut().enumerate() {
            *v = (j as f32) * 8.0 - 127.0; // -127 ..= 121; amax == 127 exactly -> d == 1.0
        }
        let amax = x_vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_eq!(amax, 127.0, "the whole point of this fixture is d == amax/127 == 1.0");

        let x = Tensor::from_f32(x_vals.to_vec(), [1, in_features]).unwrap();

        let fast = x.quantized_matmul(&weight).unwrap().to_vec_f32().unwrap();
        let reference = reference_linear(&x, &weight).to_vec_f32().unwrap();

        // With BOTH sources of rounding eliminated by construction, the fused
        // integer path and the dequantize-then-f32 path must now agree to within
        // f32 accumulation noise -- not merely within a quantization bound.
        assert_allclose(&fast, &reference, 0.0, 1e-3);
    }

    // ---- K-quant fused kernels (Q4_K / Q6_K): the `Q4_K_M` workhorses ----
    //
    // These need no `f32 -> K-quant` encoder, and deliberately do not have
    // one. EVERY byte pattern is a structurally valid K-quant super-block —
    // the scales, mins and packed quants are just integers, and only the f16
    // super-scales need to be sane — so the fixtures below emit arbitrary
    // deterministic bytes and let [`crate::quant`]'s ggml-verified
    // dequantizer say what they mean. That makes the oracle the *decoder*
    // (already checked bit-exact against `ggml-quants.c`) rather than a
    // hand-rolled encoder that could share a bug with the thing under test.

    /// `n_superblocks` of arbitrary-but-valid Q4_K bytes (144 each): `f16 d`,
    /// `f16 dmin`, 12 packed scale/min bytes, 128 nibble-pair bytes. The
    /// super-scales are kept small so the decoded weights land in a normal
    /// activation range instead of overflowing into f32 noise.
    fn arbitrary_q4_k(seed: u64, n_superblocks: usize) -> Vec<u8> {
        let mut state = seed | 1;
        let mut byte = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u8
        };
        let mut bytes = Vec::with_capacity(n_superblocks * 144);
        for _ in 0..n_superblocks {
            bytes.extend_from_slice(&crate::half::f32_to_f16(0.01).to_le_bytes());
            bytes.extend_from_slice(&crate::half::f32_to_f16(0.005).to_le_bytes());
            for _ in 0..(12 + 128) {
                bytes.push(byte());
            }
        }
        bytes
    }

    /// `n_superblocks` of arbitrary-but-valid Q6_K bytes (210 each):
    /// `ql[128]`, `qh[64]`, 16 SIGNED `i8` scales, `f16 d`.
    fn arbitrary_q6_k(seed: u64, n_superblocks: usize) -> Vec<u8> {
        let mut state = seed | 1;
        let mut byte = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u8
        };
        let mut bytes = Vec::with_capacity(n_superblocks * 210);
        for _ in 0..n_superblocks {
            for _ in 0..(128 + 64 + 16) {
                bytes.push(byte());
            }
            bytes.extend_from_slice(&crate::half::f32_to_f16(0.002).to_le_bytes());
        }
        bytes
    }

    /// The property that makes the Q4_K fused kernel safe to switch on:
    /// dotting the packed super-block directly must agree with dequantizing
    /// it and doing an ordinary `f32` matmul, to within the error the
    /// *activation's* Q8_0 rounding alone can introduce. Q4_K is the harder
    /// of the two because it is affine — `d*sc*nib - dmin*min` — so the min
    /// term has to be carried through the integer factorisation separately;
    /// drop it and the output is quietly biased rather than obviously wrong.
    #[test]
    fn quantized_matmul_q4_k_agrees_with_dequantize_then_matmul_reference() {
        let (out_features, in_features) = (5, 512); // 2 super-blocks per row.
        let bytes = arbitrary_q4_k(11, out_features * in_features / 256);
        let weight = Tensor::from_quantized(DType::Q4_K, bytes, [out_features, in_features]).unwrap();
        let weight_rows_f32: Vec<Vec<f32>> = weight
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32()
            .unwrap()
            .chunks_exact(in_features)
            .map(<[f32]>::to_vec)
            .collect();

        let x_rows: Vec<Vec<f32>> =
            (0..3).map(|r| random_f32s(12 + r as u64 * 100, in_features, 2.0)).collect();

        assert_agrees_within_activation_quantization_bound(&x_rows, &weight, &weight_rows_f32);
    }

    /// Same property for Q6_K. Its trap is ordering, not arithmetic: 16-element
    /// sub-blocks laid out in four interleaved quadrants per 128-element half,
    /// with SIGNED scales. Using the wrong sub-block's scale still produces
    /// finite, plausible numbers, so only a comparison against the decoder
    /// catches it.
    #[test]
    fn quantized_matmul_q6_k_agrees_with_dequantize_then_matmul_reference() {
        let (out_features, in_features) = (4, 512);
        let bytes = arbitrary_q6_k(21, out_features * in_features / 256);
        let weight = Tensor::from_quantized(DType::Q6_K, bytes, [out_features, in_features]).unwrap();
        let weight_rows_f32: Vec<Vec<f32>> = weight
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32()
            .unwrap()
            .chunks_exact(in_features)
            .map(<[f32]>::to_vec)
            .collect();

        let x_rows: Vec<Vec<f32>> =
            (0..3).map(|r| random_f32s(22 + r as u64 * 100, in_features, 2.0)).collect();

        assert_agrees_within_activation_quantization_bound(&x_rows, &weight, &weight_rows_f32);
    }

    /// THE REGRESSION TEST FOR THE LIVE BUG — and the gap the existing Q4_K
    /// tests could not close.
    ///
    /// `quant.rs` already proved `dequant_q4_k` bit-exact against ggml, and it
    /// was right: the decode was never wrong. But nothing anywhere asserted
    /// that a Q4_K weight could be *computed on without being expanded*, and
    /// that is the property a `Q4_K_M` model actually needs. With no fused
    /// kernel, `kopitiam-runtime` dequantized every matmul weight at load, so
    /// SmolLM2-1.7B-Instruct-Q4_K_M went from 1.06 GB on disk to 6.82 GB of
    /// private commit and answered in 392.9 s (19m04s the first time, when the
    /// box had to page) against llama.cpp's 4.0 s on the same file — correct
    /// output nobody would ever wait for.
    ///
    /// So this test states the invariant in the terms that broke: **every
    /// quantized dtype a real `Q4_K_M` file is built from must have a fused
    /// kernel.** The two dtypes below are what SmolLM2-1.7B-Instruct-Q4_K_M
    /// actually ships (144 Q4_K tensors + 25 Q6_K, verified by reading the
    /// GGUF's tensor table). A future refactor that drops either from
    /// `has_fused_matmul_kernel` reintroduces the memory blowup, and this
    /// fails instead of the maintainer discovering it as a hang.
    #[test]
    fn every_dtype_a_q4_k_m_file_ships_has_a_fused_kernel() {
        for dtype in [DType::Q4_K, DType::Q6_K] {
            assert!(
                has_fused_matmul_kernel(dtype),
                "{dtype} is a Q4_K_M matmul weight format; without a fused kernel the whole \
                 model is dequantized to f32 at load (~4x memory) and becomes unusably slow"
            );
        }
    }

    #[test]
    fn quantized_matmul_supports_leading_batch_dimensions() {
        let in_features = 32;
        let weight_vals = random_f32s(5, 4 * in_features, 3.0);
        let weight = encode_q8_0(&weight_vals, [4, in_features]);

        let x_vals = random_f32s(6, 2 * 3 * in_features, 2.0);
        let x = Tensor::from_f32(x_vals, [2, 3, in_features]).unwrap();

        let out = x.quantized_matmul(&weight).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3, 4]);

        // Must match doing it one flattened row at a time.
        let flat = x.reshape([6, in_features]).unwrap();
        let flat_out = flat.quantized_matmul(&weight).unwrap();
        assert_eq!(out.to_vec_f32().unwrap(), flat_out.to_vec_f32().unwrap());
    }

    #[test]
    fn quantized_matmul_rejects_a_non_quantized_weight() {
        let x = Tensor::from_f32(vec![1.0; 32], [1, 32]).unwrap();
        let weight = Tensor::from_f32(vec![1.0; 32], [1, 32]).unwrap();
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::UnsupportedDType { .. })));
    }

    #[test]
    fn quantized_matmul_rejects_a_quantized_format_with_no_fused_kernel() {
        // Q4_1 dequantizes fine (see quant::dequantize) but has no fused
        // integer dot product implemented here yet.
        let x = Tensor::from_f32(vec![1.0; 32], [1, 32]).unwrap();
        let weight = Tensor::from_quantized(DType::Q4_1, vec![0u8; 20], [1, 32]).unwrap();
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::UnsupportedDType { .. })));
    }

    #[test]
    fn quantized_matmul_rejects_non_f32_activations() {
        let x = Tensor::from_quantized(DType::Q8_0, vec![0u8; 34], [1, 32]).unwrap();
        let weight = encode_q8_0(&[0.0; 32], [1, 32]);
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::DTypeMismatch { .. })));
    }

    #[test]
    fn quantized_matmul_rejects_an_in_features_mismatch() {
        let x = Tensor::from_f32(vec![1.0; 32], [1, 32]).unwrap();
        let weight = encode_q8_0(&[0.0; 64], [1, 64]);
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn quantized_matmul_rejects_an_in_features_not_a_multiple_of_32() {
        // A Q4_0 tensor can only exist at all with a whole number of
        // 32-element blocks, but a rank-2 [out, in] shape can still smuggle
        // a non-block-aligned `in` past `Tensor::from_quantized` as long as
        // `out * in` is itself a whole number of blocks (e.g. out=2, in=16
        // -> 32 total elements, one block, but rows straddle the block).
        let weight = Tensor::from_quantized(DType::Q4_0, vec![0u8; 18], [2, 16]).unwrap();
        let x = Tensor::from_f32(vec![1.0; 16], [1, 16]).unwrap();
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn quantized_matmul_rejects_a_rank_1_weight() {
        let x = Tensor::from_f32(vec![1.0; 32], [1, 32]).unwrap();
        let weight = Tensor::from_quantized(DType::Q8_0, vec![0u8; 34], [32]).unwrap();
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn quantized_matmul_rejects_a_k_quant_weight_with_no_fused_kernel() {
        // Q5_K dequantizes fine (see quant::dequant_q5_k) but has no fused
        // integer dot product here — it must be dequantized to f32 and go
        // through `matmul` instead. One Q5_K super-block is 256 elements /
        // 176 bytes. (Q4_K and Q6_K, its neighbours, DO have kernels now —
        // they are what a real Q4_K_M file is made of.)
        let x = Tensor::from_f32(vec![1.0; 256], [1, 256]).unwrap();
        let weight = Tensor::from_quantized(DType::Q5_K, vec![0u8; 176], [1, 256]).unwrap();
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::UnsupportedDType { .. })));
    }

    /// A K-quant super-block is 256 elements, so an `in_features` that is a
    /// whole number of 32s but NOT of 256 must be refused. Slicing a
    /// super-block in half would read another row's scales as this row's —
    /// wrong numbers, no error, exactly the class of bug that is invisible
    /// downstream.
    #[test]
    fn quantized_matmul_rejects_a_k_quant_in_features_not_a_multiple_of_256() {
        // out=4, in=64: 256 elements total = one whole super-block, so
        // `from_quantized` accepts it, but each row straddles the block.
        let weight = Tensor::from_quantized(DType::Q4_K, vec![0u8; 144], [4, 64]).unwrap();
        let x = Tensor::from_f32(vec![1.0; 64], [1, 64]).unwrap();
        assert!(matches!(x.quantized_matmul(&weight), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn has_fused_matmul_kernel_is_true_for_exactly_q4_0_q8_0_q4_k_and_q6_k() {
        // The single source of truth every layer keys off. Anything NOT in
        // the first list is dequantized to f32 at load by `kopitiam-runtime`'s
        // `load_matmul_weight`, which costs ~4x that tensor's memory — so this
        // set is a memory-budget decision, not just a dispatch table.
        for dtype in [DType::Q4_0, DType::Q8_0, DType::Q4_K, DType::Q6_K] {
            assert!(has_fused_matmul_kernel(dtype), "{dtype} should have a fused kernel");
        }
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I8,
            DType::I32,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q2_K,
            DType::Q3_K,
            DType::Q5_K,
            DType::Q8_K,
        ] {
            assert!(!has_fused_matmul_kernel(dtype), "{dtype} should not have a fused kernel");
        }
    }
}
