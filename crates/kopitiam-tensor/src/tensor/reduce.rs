//! `sum` and `max` reductions along a single axis, plus `argmax`.

use kopitiam_core::{DType, Error, Result, Shape};

use crate::storage::Storage;

use super::Tensor;

impl Tensor {
    /// Sums along `axis`. If `keepdim`, `axis` becomes length 1 in the
    /// result instead of being removed (useful for immediately
    /// broadcasting the result back against the input, e.g. normalizing by
    /// a row sum).
    pub fn sum(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        self.reduce(axis, keepdim, 0.0, |acc, v| acc + v)
    }

    /// Reduces along `axis` by taking the maximum.
    pub fn max(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        self.reduce(axis, keepdim, f32::NEG_INFINITY, f32::max)
    }

    /// Index of the maximum along `axis`, returned as an [`DType::I32`]
    /// tensor of positions (not the max *values* — that's [`Tensor::max`]).
    ///
    /// This is the last step of greedy decoding: run the forward pass, get
    /// the final `[.., vocab]` logits, then `argmax(last_axis, false)` gives
    /// you the chosen token id per position. Read the ids out with
    /// [`Tensor::to_vec_i32`].
    ///
    /// The output dtype is `I32` on purpose — these are *indices* into a
    /// dimension, the same currency [`Tensor::gather_rows`] consumes as token
    /// ids, so the next embedding lookup can eat this straight without a cast.
    /// The result's shape is `self`'s shape with `axis` dropped, or set to
    /// length 1 if `keepdim` (same rule as [`Tensor::sum`]).
    ///
    /// **Tie-break contract (load-bearing, so it's deterministic):** on a tie
    /// the *first* (lowest-index) maximum wins. We only overwrite the running
    /// best on a strict `>`, never on `==`, so re-running on the same input
    /// always yields the same ids. NaN never wins a comparison (`v > best` is
    /// false for NaN), so a NaN element is skipped rather than selected —
    /// don't lean on that for correctness though; NaN logits mean the forward
    /// pass already went wrong upstream.
    pub fn argmax(&self, axis: usize, keepdim: bool) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        let rank = self.rank();
        if axis >= rank {
            return Err(Error::IndexOutOfBounds { dim: axis, index: axis, len: rank });
        }
        let Storage::F32(data) = self.storage.as_ref() else { unreachable!() };
        let contiguous: Vec<f32> = self.logical_offsets().map(|i| data[i]).collect();

        let dims = self.shape.dims();
        let axis_len = dims[axis];
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product();

        let mut out = vec![0i32; outer * inner];
        for o in 0..outer {
            for inn in 0..inner {
                let mut best = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for a in 0..axis_len {
                    let v = contiguous[o * axis_len * inner + a * inner + inn];
                    // Strict `>` only: first occurrence wins ties (see the
                    // tie-break contract in the doc comment above).
                    if v > best {
                        best = v;
                        best_idx = a;
                    }
                }
                out[o * inner + inn] = best_idx as i32;
            }
        }

        let mut new_dims = dims.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        Tensor::from_i32(out, Shape::new(new_dims))
    }

    fn reduce(&self, axis: usize, keepdim: bool, init: f32, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        let rank = self.rank();
        if axis >= rank {
            return Err(Error::IndexOutOfBounds { dim: axis, index: axis, len: rank });
        }
        let Storage::F32(data) = self.storage.as_ref() else { unreachable!() };
        let contiguous: Vec<f32> = self.logical_offsets().map(|i| data[i]).collect();

        let dims = self.shape.dims();
        let axis_len = dims[axis];
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product();

        let mut out = vec![init; outer * inner];
        for o in 0..outer {
            for inn in 0..inner {
                let mut acc = init;
                for a in 0..axis_len {
                    acc = f(acc, contiguous[o * axis_len * inner + a * inner + inn]);
                }
                out[o * inner + inn] = acc;
            }
        }

        let mut new_dims = dims.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        Tensor::from_f32(out, Shape::new(new_dims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_along_last_axis_matches_hand_computation() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let s = t.sum(1, false).unwrap();
        assert_eq!(s.shape().dims(), &[2]);
        assert_eq!(s.to_vec_f32().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn sum_keepdim_preserves_rank() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let s = t.sum(1, true).unwrap();
        assert_eq!(s.shape().dims(), &[2, 1]);
        assert_eq!(s.to_vec_f32().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn sum_along_axis_zero_matches_hand_computation() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let s = t.sum(0, false).unwrap();
        assert_eq!(s.to_vec_f32().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn max_along_last_axis_matches_hand_computation() {
        let t = Tensor::from_f32(vec![1.0, 5.0, 3.0, 9.0, 2.0, 4.0], [2, 3]).unwrap();
        let m = t.max(1, false).unwrap();
        assert_eq!(m.to_vec_f32().unwrap(), vec![5.0, 9.0]);
    }

    #[test]
    fn reduce_rejects_an_axis_beyond_rank() {
        let t = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        assert!(matches!(t.sum(1, false), Err(Error::IndexOutOfBounds { .. })));
    }

    #[test]
    fn argmax_last_axis_matches_hand_computation_and_is_i32() {
        // Row 0 peak is 5.0 at col 1; row 1 peak is 9.0 at col 0.
        let t = Tensor::from_f32(vec![1.0, 5.0, 3.0, 9.0, 2.0, 4.0], [2, 3]).unwrap();
        let idx = t.argmax(1, false).unwrap();
        assert_eq!(idx.dtype(), DType::I32);
        assert_eq!(idx.shape().dims(), &[2]);
        assert_eq!(idx.to_vec_i32().unwrap(), vec![1, 0]);
    }

    #[test]
    fn argmax_keepdim_preserves_rank() {
        let t = Tensor::from_f32(vec![1.0, 5.0, 3.0, 9.0, 2.0, 4.0], [2, 3]).unwrap();
        let idx = t.argmax(1, true).unwrap();
        assert_eq!(idx.shape().dims(), &[2, 1]);
        assert_eq!(idx.to_vec_i32().unwrap(), vec![1, 0]);
    }

    #[test]
    fn argmax_along_axis_zero_matches_hand_computation() {
        // Down each column: col0 max is 4>1 at row1; col1 max is 5>2 at row1;
        // col2 max is 6>3 at row1.
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let idx = t.argmax(0, false).unwrap();
        assert_eq!(idx.to_vec_i32().unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn argmax_breaks_ties_towards_the_first_occurrence() {
        // Three equal maxima: the contract says the lowest index wins.
        let t = Tensor::from_f32(vec![7.0, 7.0, 7.0], [3]).unwrap();
        let idx = t.argmax(0, false).unwrap();
        assert_eq!(idx.to_vec_i32().unwrap(), vec![0]);
    }

    #[test]
    fn argmax_picks_the_greedy_token_from_a_vocab_row() {
        // The realistic shape: a single position's logits over a vocab of 5.
        // Token 3 has the highest logit, so greedy decoding must pick id 3.
        let logits = Tensor::from_f32(vec![0.1, -2.0, 0.5, 3.7, 1.2], [5]).unwrap();
        let id = logits.argmax(0, false).unwrap();
        assert_eq!(id.to_vec_i32().unwrap(), vec![3]);
    }

    #[test]
    fn argmax_rejects_non_f32_input() {
        let t = Tensor::from_i32(vec![1, 2, 3], [3]).unwrap();
        assert!(matches!(t.argmax(0, false), Err(Error::DTypeMismatch { .. })));
    }

    #[test]
    fn argmax_rejects_an_axis_beyond_rank() {
        let t = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        assert!(matches!(t.argmax(1, false), Err(Error::IndexOutOfBounds { .. })));
    }

    #[test]
    fn to_vec_i32_rejects_f32_input() {
        let t = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        assert!(matches!(t.to_vec_i32(), Err(Error::DTypeMismatch { .. })));
    }
}
