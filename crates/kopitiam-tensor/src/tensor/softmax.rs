//! Numerically stable softmax.

use kopitiam_core::{DType, Error, Result};

use crate::storage::Storage;

use super::Tensor;

impl Tensor {
    /// Softmax along `axis`: for every slice along that axis,
    /// `out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`.
    ///
    /// # Why subtract the max before `exp`
    ///
    /// Real transformer logits routinely reach the hundreds or thousands in
    /// magnitude. `exp(1000.0)` overflows `f32` to `+inf`, and
    /// `inf / inf` is `NaN` — softmax computed the naive way silently
    /// poisons the rest of the forward pass on real inputs, not just
    /// adversarial ones. Subtracting the row's max first is mathematically
    /// a no-op (it cancels between numerator and denominator) but keeps
    /// every exponent `<= 0`, so the largest possible input to `exp` is
    /// `exp(0) = 1`. This is why the max-subtraction step is not an
    /// optional optimization here — see the `large_magnitude_input_does_not_overflow`
    /// test below for the exact failure this avoids.
    pub fn softmax(&self, axis: usize) -> Result<Tensor> {
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

        let mut out = vec![0f32; contiguous.len()];
        for o in 0..outer {
            for inn in 0..inner {
                let base = o * axis_len * inner + inn;
                let at = |a: usize| contiguous[base + a * inner];

                let max = (0..axis_len).map(at).fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for a in 0..axis_len {
                    let exp = (at(a) - max).exp();
                    out[base + a * inner] = exp;
                    sum += exp;
                }
                for a in 0..axis_len {
                    out[base + a * inner] /= sum;
                }
            }
        }
        Tensor::from_f32(out, self.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let out = t.softmax(1).unwrap().to_vec_f32().unwrap();
        assert_close(out[0] + out[1] + out[2], 1.0);
        assert_close(out[3] + out[4] + out[5], 1.0);
    }

    #[test]
    fn softmax_matches_hand_computation_for_a_simple_row() {
        // softmax([0, 0]) = [0.5, 0.5]
        let t = Tensor::from_f32(vec![0.0, 0.0], [2]).unwrap();
        let out = t.softmax(0).unwrap().to_vec_f32().unwrap();
        assert_close(out[0], 0.5);
        assert_close(out[1], 0.5);
    }

    #[test]
    fn softmax_preserves_relative_order() {
        let t = Tensor::from_f32(vec![1.0, 3.0, 2.0], [3]).unwrap();
        let out = t.softmax(0).unwrap().to_vec_f32().unwrap();
        assert!(out[1] > out[2]);
        assert!(out[2] > out[0]);
    }

    /// The whole point of subtracting the row max first: without it,
    /// `exp(1000.0)` and `exp(1001.0)` both overflow `f32` to infinity, and
    /// `inf / inf` is `NaN`. With the max-subtraction trick this input is
    /// completely unremarkable (it reduces to `softmax([-1.0, 0.0])`).
    #[test]
    fn large_magnitude_input_does_not_overflow_to_nan() {
        let t = Tensor::from_f32(vec![1000.0, 1001.0], [2]).unwrap();
        let out = t.softmax(0).unwrap().to_vec_f32().unwrap();
        assert!(!out[0].is_nan() && !out[1].is_nan(), "softmax produced NaN: {out:?}");
        assert_close(out[0] + out[1], 1.0);
        // exp(-1)/(exp(-1)+1) ~= 0.2689414
        assert_close(out[0], 0.268_941_4);
    }

    #[test]
    fn softmax_operates_along_an_arbitrary_axis_of_a_3d_tensor() {
        // shape [1, 2, 2]; softmax along axis 1 (not the last axis).
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]).unwrap();
        let out = t.softmax(1).unwrap().to_vec_f32().unwrap();
        // Column 0: [1, 3], column 1: [2, 4], each independently normalized.
        assert_close(out[0] + out[2], 1.0);
        assert_close(out[1] + out[3], 1.0);
    }
}
