//! `sum` and `max` reductions along a single axis.

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
}
