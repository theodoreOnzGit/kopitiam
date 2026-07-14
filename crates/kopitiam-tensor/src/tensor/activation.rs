//! `silu` and `gelu` activations.

use kopitiam_core::{DType, Result};

use crate::storage::Storage;

use super::Tensor;

/// `sqrt(2 / pi)`, the constant in the tanh approximation of GELU.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

impl Tensor {
    /// SiLU / swish (Elfwing, Uchibe & Doya, 2017; also Hendrycks & Gimpel):
    /// `x * sigmoid(x) = x / (1 + exp(-x))`. Used by LLaMA-family FFNs.
    pub fn silu(&self) -> Result<Tensor> {
        self.unary_f32(|x| x / (1.0 + (-x).exp()))
    }

    /// GELU (Hendrycks & Gimpel, 2016), tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    ///
    /// This crate implements the tanh approximation rather than the exact
    /// `0.5 * x * (1 + erf(x / sqrt(2)))` form because `erf` is not in
    /// Rust's `std` — computing it exactly would mean shipping a rational
    /// or Chebyshev `erf` approximation of our own, which is *more* code
    /// and *more* numerical risk than the tanh form for a difference that
    /// is below `1e-3` everywhere and is what GPT-2/GPT-NeoX-family models
    /// were themselves trained and evaluated with (the tanh form is
    /// sometimes called "gelu_new" in that lineage). If a model that
    /// specifically requires the exact erf form shows up, add it as a
    /// second method rather than replacing this one.
    pub fn gelu(&self) -> Result<Tensor> {
        self.unary_f32(|x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh()))
    }

    fn unary_f32(&self, f: impl Fn(f32) -> f32) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        let Storage::F32(data) = self.storage.as_ref() else { unreachable!() };
        let out: Vec<f32> = self.logical_offsets().map(|i| f(data[i])).collect();
        Tensor::from_f32(out, self.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use kopitiam_core::Error;

    use super::*;

    fn assert_close(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "expected {b}, got {a}");
    }

    #[test]
    fn silu_matches_hand_computation() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let t = Tensor::from_f32(vec![0.0, 1.0, -1.0], [3]).unwrap();
        let out = t.silu().unwrap().to_vec_f32().unwrap();
        assert_close(out[0], 0.0, 1e-6);
        // silu(1) = 1 / (1 + e^-1) ~= 0.7310586
        assert_close(out[1], 0.731_058_6, 1e-5);
        // silu(-1) = -1 / (1 + e^1) ~= -0.2689414
        assert_close(out[2], -0.268_941_4, 1e-5);
    }

    #[test]
    fn silu_matches_the_reference_formula_across_a_range_of_inputs() {
        // silu is *not* globally monotonic (it dips to a minimum around
        // x ~= -1.278 before rising) so the meaningful correctness check is
        // "matches x * sigmoid(x) pointwise", not a monotonicity claim.
        let xs = vec![-5.0, -3.0, -1.278, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0];
        let t = Tensor::from_f32(xs.clone(), [xs.len()]).unwrap();
        let out = t.silu().unwrap().to_vec_f32().unwrap();
        for (x, o) in xs.iter().zip(&out) {
            let expected = x / (1.0 + (-x).exp());
            assert_close(*o, expected, 1e-5);
        }
    }

    #[test]
    fn silu_is_monotonically_increasing_on_its_increasing_branch() {
        // For x >= -1.278ish, silu is monotonically increasing; this is the
        // range every real activation input in a trained model's residual
        // stream overwhelmingly falls into after the first few layers.
        let t = Tensor::from_f32(vec![-1.0, 0.0, 1.0, 2.0, 3.0], [5]).unwrap();
        let out = t.silu().unwrap().to_vec_f32().unwrap();
        for pair in out.windows(2) {
            assert!(pair[1] > pair[0], "silu should be increasing here: {out:?}");
        }
    }

    #[test]
    fn gelu_matches_hand_computation_at_zero_and_matches_known_values() {
        // gelu(0) = 0.
        let t = Tensor::from_f32(vec![0.0, 1.0, -1.0], [3]).unwrap();
        let out = t.gelu().unwrap().to_vec_f32().unwrap();
        assert_close(out[0], 0.0, 1e-6);
        // Known reference values for the tanh-approximation GELU (matches
        // the widely used "gelu_new" implementation to ~1e-6).
        assert_close(out[1], 0.841_192, 1e-4);
        assert_close(out[2], -0.158_808, 1e-4);
    }

    #[test]
    fn gelu_approaches_the_identity_for_large_positive_x_and_zero_for_large_negative_x() {
        let t = Tensor::from_f32(vec![10.0, -10.0], [2]).unwrap();
        let out = t.gelu().unwrap().to_vec_f32().unwrap();
        assert_close(out[0], 10.0, 1e-3);
        assert_close(out[1], 0.0, 1e-3);
    }

    #[test]
    fn activations_reject_non_f32_input() {
        let t = Tensor::from_i32(vec![1, 2, 3], [3]).unwrap();
        assert!(matches!(t.silu(), Err(Error::DTypeMismatch { .. })));
        assert!(matches!(t.gelu(), Err(Error::DTypeMismatch { .. })));
    }
}
