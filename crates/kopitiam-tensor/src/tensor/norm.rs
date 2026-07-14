//! `rms_norm` and `layer_norm`, applied along the last dimension.
//!
//! These live in `kopitiam-tensor` rather than a later transformer-graph
//! crate because every transformer architecture this runtime will ever
//! load uses one of them at every layer boundary — duplicating the
//! reduction-and-scale logic in each model's graph-building code would be
//! exactly the "duplicated logic" `CLAUDE.md` asks to avoid.

use kopitiam_core::{DType, Error, Result};

use crate::storage::Storage;

use super::Tensor;

impl Tensor {
    /// RMSNorm (Zhang & Sennrich, 2019): `x / rms(x) * weight`, where
    /// `rms(x) = sqrt(mean(x^2) + eps)`, computed independently for every
    /// vector along the last dimension.
    ///
    /// `weight` must have exactly `hidden` elements, where `hidden` is
    /// `self`'s last dimension.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        let hidden = self.last_dim()?;
        let w = self.matching_weight(weight, hidden)?;

        let Storage::F32(data) = self.storage.as_ref() else { unreachable!() };
        let contiguous: Vec<f32> = self.logical_offsets().map(|i| data[i]).collect();

        let mut out = vec![0f32; contiguous.len()];
        for (row_out, row_in) in out.chunks_mut(hidden).zip(contiguous.chunks(hidden)) {
            let mean_square = row_in.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0 / (mean_square + eps).sqrt();
            for (o, (&x, &w)) in row_out.iter_mut().zip(row_in.iter().zip(&w)) {
                *o = x * inv_rms * w;
            }
        }
        Tensor::from_f32(out, self.shape.clone())
    }

    /// Layer normalization (Ba, Kiros & Hinton, 2016):
    /// `(x - mean(x)) / sqrt(var(x) + eps) * weight + bias`, with `mean`
    /// and the (population, i.e. divide-by-N) `var` computed independently
    /// for every vector along the last dimension.
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        let hidden = self.last_dim()?;
        let w = self.matching_weight(weight, hidden)?;
        let b = self.matching_weight(bias, hidden)?;

        let Storage::F32(data) = self.storage.as_ref() else { unreachable!() };
        let contiguous: Vec<f32> = self.logical_offsets().map(|i| data[i]).collect();

        let mut out = vec![0f32; contiguous.len()];
        for (row_out, row_in) in out.chunks_mut(hidden).zip(contiguous.chunks(hidden)) {
            let mean = row_in.iter().sum::<f32>() / hidden as f32;
            let variance = row_in.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();
            for (i, o) in row_out.iter_mut().enumerate() {
                *o = (row_in[i] - mean) * inv_std * w[i] + b[i];
            }
        }
        Tensor::from_f32(out, self.shape.clone())
    }

    fn last_dim(&self) -> Result<usize> {
        if self.rank() == 0 {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: self.shape.clone() });
        }
        Ok(self.shape.dims()[self.rank() - 1])
    }

    /// Validates that `weight` is a plain `f32` vector of exactly `hidden`
    /// elements and materializes it, so the per-row loop can index it
    /// directly instead of re-deriving strided offsets per element.
    fn matching_weight(&self, weight: &Tensor, hidden: usize) -> Result<Vec<f32>> {
        weight.require_dtype(DType::F32)?;
        if weight.elem_count() != hidden {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: weight.shape.clone() });
        }
        weight.to_vec_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_hand_computation() {
        // x = [3, 4], mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5).
        // With weight = [1, 1] and eps = 0: out = x / sqrt(12.5).
        let x = Tensor::from_f32(vec![3.0, 4.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![1.0, 1.0], [2]).unwrap();
        let out = x.rms_norm(&w, 0.0).unwrap().to_vec_f32().unwrap();
        let expected_scale = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * expected_scale).abs() < 1e-6);
        assert!((out[1] - 4.0 * expected_scale).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_applies_the_per_channel_weight() {
        let x = Tensor::from_f32(vec![1.0, 1.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![2.0, 3.0], [2]).unwrap();
        // rms([1,1]) = 1, so out = x * w directly.
        let out = x.rms_norm(&w, 0.0).unwrap().to_vec_f32().unwrap();
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_matches_hand_computation() {
        // x = [1, 2, 3, 4], mean=2.5, var = ((1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2)/4 = 1.25
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 4]).unwrap();
        let w = Tensor::from_f32(vec![1.0; 4], [4]).unwrap();
        let b = Tensor::from_f32(vec![0.0; 4], [4]).unwrap();
        let out = x.layer_norm(&w, &b, 0.0).unwrap().to_vec_f32().unwrap();
        let inv_std = 1.0 / 1.25f32.sqrt();
        let expected = [
            (1.0 - 2.5) * inv_std,
            (2.0 - 2.5) * inv_std,
            (3.0 - 2.5) * inv_std,
            (4.0 - 2.5) * inv_std,
        ];
        for (o, e) in out.iter().zip(expected) {
            assert!((o - e).abs() < 1e-5, "got {o}, expected {e}");
        }
        // Normalized output has zero mean.
        assert!(out.iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn layer_norm_applies_weight_and_bias_after_normalizing() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 4]).unwrap();
        let unit_w = Tensor::from_f32(vec![1.0; 4], [4]).unwrap();
        let zero_b = Tensor::from_f32(vec![0.0; 4], [4]).unwrap();
        let normalized = x.layer_norm(&unit_w, &zero_b, 0.0).unwrap().to_vec_f32().unwrap();

        let w = Tensor::from_f32(vec![2.0; 4], [4]).unwrap();
        let b = Tensor::from_f32(vec![10.0; 4], [4]).unwrap();
        let out = x.layer_norm(&w, &b, 0.0).unwrap().to_vec_f32().unwrap();

        for (o, n) in out.iter().zip(&normalized) {
            assert!((o - (n * 2.0 + 10.0)).abs() < 1e-5, "got {o}, expected {}", n * 2.0 + 10.0);
        }
    }

    #[test]
    fn norm_ops_reject_a_mismatched_weight_length() {
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![1.0, 1.0, 1.0], [3]).unwrap();
        assert!(matches!(x.rms_norm(&w, 1e-5), Err(Error::ShapeMismatch { .. })));
    }
}
