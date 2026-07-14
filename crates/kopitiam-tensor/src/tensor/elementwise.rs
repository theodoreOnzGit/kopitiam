//! Broadcasting elementwise arithmetic: `add`, `sub`, `mul`, `div`.
//!
//! All four share one implementation — the only thing that differs between
//! them is which `f32, f32 -> f32` function to apply — so they are thin
//! wrappers around [`broadcast_binary`] rather than four copies of the same
//! broadcast-then-zip-then-collect logic.

use kopitiam_core::{DType, Result};

use crate::storage::Storage;

use super::Tensor;

impl Tensor {
    /// Elementwise `self + other`, broadcasting shapes per
    /// [`kopitiam_core::Shape::broadcast`].
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        broadcast_binary(self, other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        broadcast_binary(self, other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        broadcast_binary(self, other, |a, b| a * b)
    }

    /// Elementwise `self / other`. Division by zero follows ordinary IEEE
    /// 754 float semantics (`+/-inf` or `NaN`) rather than erroring — a
    /// tensor op has no way to know whether that is a bug or an expected
    /// edge case (e.g. a masked-out attention score), so it is left to the
    /// caller to detect if it matters.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        broadcast_binary(self, other, |a, b| a / b)
    }
}

fn broadcast_binary(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    a.require_dtype(DType::F32)?;
    b.require_dtype(DType::F32)?;
    let out_shape = a.shape.broadcast(&b.shape)?;
    let a_view = a.broadcast_to(out_shape.clone())?;
    let b_view = b.broadcast_to(out_shape.clone())?;
    let Storage::F32(a_data) = a_view.storage.as_ref() else { unreachable!() };
    let Storage::F32(b_data) = b_view.storage.as_ref() else { unreachable!() };
    let result: Vec<f32> = a_view
        .logical_offsets()
        .zip(b_view.logical_offsets())
        .map(|(ia, ib)| f(a_data[ia], b_data[ib]))
        .collect();
    Tensor::from_f32(result, out_shape)
}

#[cfg(test)]
mod tests {
    use kopitiam_core::Error;

    use super::*;

    #[test]
    fn add_same_shape_matches_hand_computation() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let b = Tensor::from_f32(vec![10.0, 20.0, 30.0], [3]).unwrap();
        assert_eq!(a.add(&b).unwrap().to_vec_f32().unwrap(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn mul_broadcasts_a_row_vector_over_a_matrix() {
        // [[1,2,3],[4,5,6]] * [10,20,30] (broadcast over rows)
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let b = Tensor::from_f32(vec![10.0, 20.0, 30.0], [3]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        assert_eq!(c.to_vec_f32().unwrap(), vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    }

    #[test]
    fn sub_broadcasts_a_column_vector_over_a_matrix() {
        // [[1,2],[3,4]] - [[10],[20]] -> [[-9,-8],[-17,-16]]
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::from_f32(vec![10.0, 20.0], [2, 1]).unwrap();
        let c = a.sub(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![-9.0, -8.0, -17.0, -16.0]);
    }

    #[test]
    fn div_by_a_scalar_broadcasts_against_rank_zero() {
        let a = Tensor::from_f32(vec![2.0, 4.0, 6.0], [3]).unwrap();
        let scalar = Tensor::from_f32(vec![2.0], []).unwrap();
        assert_eq!(a.div(&scalar).unwrap().to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcasting_incompatible_shapes_is_rejected() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let b = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        assert!(matches!(a.add(&b), Err(Error::NotBroadcastable { .. })));
    }

    #[test]
    fn elementwise_ops_reject_non_f32_operands() {
        let a = Tensor::from_i32(vec![1, 2, 3], [3]).unwrap();
        let b = Tensor::from_i32(vec![1, 2, 3], [3]).unwrap();
        assert!(matches!(a.add(&b), Err(Error::DTypeMismatch { .. })));
    }
}
