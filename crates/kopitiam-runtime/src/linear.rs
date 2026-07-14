//! The one linear-projection helper every weight matrix in this crate goes
//! through.
//!
//! # Why the transpose
//!
//! GGUF (and SafeTensors, mirroring the PyTorch `nn.Linear` convention it
//! was exported from) stores a projection's weight as `[out_features,
//! in_features]` — see `kopitiam_loader::gguf`'s dimension-order docs for
//! why that is `Shape`'s convention after the loader corrects ggml's
//! on-disk `ne[]` order. Computing `y = x @ W^T + b` (not `x @ W`) is what
//! reproduces `nn.Linear(x)` exactly; this helper is the one place that
//! transpose happens, so every call site (attention Q/K/V/O, the SwiGLU
//! MLP's three matrices, the output projection) writes `linear(x, w, b)`
//! instead of independently getting the transpose direction right (or
//! wrong).
//!
//! # Dtype dispatch: the one place a quantized weight takes a different path
//!
//! [`crate::weights::ModelWeights`] keeps a matmul-operand weight
//! block-quantized whenever the GGUF file shipped it that way (see
//! [`crate::bridge::load_matmul_weight`]'s docs). A quantized tensor
//! cannot be [`kopitiam_tensor::Tensor::transpose`]d at all (block data
//! addresses no finer than a whole block — see that crate's docs), so this
//! function, not each individual caller, is where the two cases split:
//! `weight.dtype().is_quantized()` routes through
//! [`kopitiam_tensor::Tensor::quantized_matmul`] (no transpose needed —
//! that method already takes `weight` in its on-disk `[out, in]` layout
//! directly, computing the equivalent of `x @ weight^T` without ever
//! forming `weight^T`); anything else takes the original transpose-then-
//! [`kopitiam_tensor::Tensor::matmul`] path. Every call site keeps writing
//! `linear(x, w, b)` either way and never needs to know or care which
//! dtype `w` turned out to be.

use kopitiam_core::Result;
use kopitiam_tensor::Tensor;

/// `x @ weight^T`, plus `bias` if given, broadcast over `x`'s rows.
///
/// `x` is `[.., in_features]`; `weight` is `[out_features, in_features]`
/// (`f32`, or block-quantized — see this module's docs); `bias`, if
/// present, is `[out_features]` (always `f32`: biases are never
/// quantized, see [`crate::weights::LayerWeights`]). Returns
/// `[.., out_features]`.
pub(crate) fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let y = if weight.dtype().is_quantized() {
        x.quantized_matmul(weight)?
    } else {
        let weight_t = weight.transpose(0, 1)?;
        x.matmul(&weight_t)?
    };
    match bias {
        Some(b) => y.add(b),
        None => Ok(y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_without_bias_matches_matmul_by_the_transposed_weight() {
        // x = [1, 2], weight (out=2, in=2) = [[1, 0], [0, 1]] (identity) -> y = x.
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.to_vec_f32().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn linear_applies_bias_after_the_matmul() {
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap();
        let b = Tensor::from_f32(vec![10.0, 20.0], [2]).unwrap();
        let y = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(y.to_vec_f32().unwrap(), vec![11.0, 22.0]);
    }

    #[test]
    fn linear_matches_hand_computation_for_a_non_trivial_weight() {
        // weight rows are the output features: out0 = 2*in0 + 0*in1, out1 = 1*in0 + 3*in1.
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![2.0, 0.0, 1.0, 3.0], [2, 2]).unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.to_vec_f32().unwrap(), vec![2.0, 7.0]);
    }

    /// A Q8_0 weight cannot go through `Tensor::transpose` at all (a
    /// block-quantized tensor's strides/offset are fixed — see
    /// `kopitiam-tensor`'s docs), so this is the test that would fail
    /// loudly (a `QuantizedElementAccess` error) if `linear`'s dtype
    /// dispatch (see this module's docs) were ever removed or broken:
    /// every element in each row is built to quantize losslessly (an
    /// all-`127` Q8_0 row decodes back to exactly its chosen scale), so the
    /// expected dot product is exact arithmetic, not an approximation.
    #[test]
    fn linear_dispatches_to_quantized_matmul_for_a_quantized_weight() {
        let in_features = 32;
        let row_values = [1.0f32, 2.0];
        let mut bytes = Vec::new();
        for &row_value in &row_values {
            let d = row_value / 127.0;
            bytes.extend_from_slice(&kopitiam_tensor::f32_to_f16(d).to_le_bytes());
            bytes.extend(std::iter::repeat_n(127u8, in_features));
        }
        let weight = Tensor::from_quantized(kopitiam_tensor::DType::Q8_0, bytes, [row_values.len(), in_features]).unwrap();

        let x = Tensor::from_f32(vec![1.0; in_features], [1, in_features]).unwrap();
        let y = linear(&x, &weight, None).unwrap();
        let out = y.to_vec_f32().unwrap();

        assert_eq!(out.len(), row_values.len());
        for (got, &row_value) in out.iter().zip(&row_values) {
            let expected = in_features as f32 * row_value;
            assert!((got - expected).abs() < 1e-2, "expected {expected}, got {got}");
        }
    }

    #[test]
    fn linear_applies_bias_after_a_quantized_matmul_too() {
        let in_features = 32;
        let d = 1.0f32 / 127.0;
        let mut bytes = kopitiam_tensor::f32_to_f16(d).to_le_bytes().to_vec();
        bytes.extend(std::iter::repeat_n(127u8, in_features));
        let weight = Tensor::from_quantized(kopitiam_tensor::DType::Q8_0, bytes, [1, in_features]).unwrap();

        let x = Tensor::from_f32(vec![1.0; in_features], [1, in_features]).unwrap();
        let bias = Tensor::from_f32(vec![100.0], [1]).unwrap();
        let y = linear(&x, &weight, Some(&bias)).unwrap();
        let out = y.to_vec_f32().unwrap();

        assert_eq!(out.len(), 1);
        assert!((out[0] - (in_features as f32 + 100.0)).abs() < 1e-2, "got {}", out[0]);
    }
}
