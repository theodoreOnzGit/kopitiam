//! 1-D convolution and max-pooling: [`Tensor::conv1d`] and
//! [`Tensor::maxpool1d`].
//!
//! Some LSTM OCR network configurations put a small convolution and/or a
//! max-pool at the input stage, ahead of the recurrent layers, to reduce the
//! along-width resolution and mix neighbouring columns before the LSTM sees
//! them. These are the standard signal-processing ops (valid, no-padding
//! cross-correlation; windowed max) — clean-room standard math, not a port of
//! Tesseract's image-stacking `Convolve`/reconfig `Maxpool`, which are shaped
//! around its `StrideMap`/`NetworkIO` plumbing rather than a plain 1-D layer.

use kopitiam_core::{DType, Error, Result, Shape};

use super::Tensor;

impl Tensor {
    /// 1-D convolution (cross-correlation, the ML convention — the kernel is
    /// *not* flipped), no padding ("valid"), with a configurable `stride`.
    ///
    /// `self` is the input `[in_channels, length]`; `weight` is
    /// `[out_channels, in_channels, kernel]`; `bias`, if given, is
    /// `[out_channels]`. The result is `[out_channels, out_length]` where
    /// `out_length = (length - kernel) / stride + 1`. Each output element is
    ///
    /// ```text
    /// out[oc][t] = bias[oc] + sum_ic sum_k input[ic][t*stride + k] * weight[oc][ic][k]
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] if any operand is not `f32`.
    /// [`Error::ShapeMismatch`] if `self` is not rank 2, `weight` is not
    /// rank 3, the channel counts disagree, `stride` is 0, or `kernel` is
    /// larger than `length` (which would make `out_length` empty or
    /// ill-defined).
    pub fn conv1d(&self, weight: &Tensor, bias: Option<&Tensor>, stride: usize) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        weight.require_dtype(DType::F32)?;

        if self.rank() != 2 || weight.rank() != 3 || stride == 0 {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: weight.shape.clone() });
        }
        let (in_channels, length) = (self.shape.dims()[0], self.shape.dims()[1]);
        let wdims = weight.shape.dims();
        let (out_channels, w_in_channels, kernel) = (wdims[0], wdims[1], wdims[2]);
        if w_in_channels != in_channels || kernel == 0 || kernel > length {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: weight.shape.clone() });
        }
        let out_length = (length - kernel) / stride + 1;

        // Materialize both operands in logical row-major order so the inner
        // loops can index flat slices directly (correct for a transposed or
        // narrowed input view too), the same approach matmul/norm take.
        let input = self.to_vec_f32()?;
        let w = weight.to_vec_f32()?;
        let bias_vec = match bias {
            Some(b) => {
                b.require_dtype(DType::F32)?;
                if b.elem_count() != out_channels {
                    return Err(Error::ShapeMismatch { expected: weight.shape.clone(), actual: b.shape.clone() });
                }
                b.to_vec_f32()?
            }
            None => vec![0.0f32; out_channels],
        };

        let mut out = vec![0f32; out_channels * out_length];
        for oc in 0..out_channels {
            for t in 0..out_length {
                let start = t * stride;
                let mut acc = bias_vec[oc];
                for ic in 0..in_channels {
                    let in_row = &input[ic * length + start..ic * length + start + kernel];
                    let w_row = &w[(oc * in_channels + ic) * kernel..(oc * in_channels + ic) * kernel + kernel];
                    for (x, wv) in in_row.iter().zip(w_row) {
                        acc += x * wv;
                    }
                }
                out[oc * out_length + t] = acc;
            }
        }
        Tensor::from_f32(out, Shape::new([out_channels, out_length]))
    }

    /// 1-D max-pooling over a sliding window, no padding, with a configurable
    /// `stride`. Applied independently per channel.
    ///
    /// `self` is `[channels, length]`; the result is `[channels, out_length]`
    /// where `out_length = (length - kernel) / stride + 1` and
    /// `out[c][t] = max(input[c][t*stride .. t*stride + kernel])`.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] if `self` is not `f32`.
    /// [`Error::ShapeMismatch`] if `self` is not rank 2, `stride` or `kernel`
    /// is 0, or `kernel` exceeds `length`.
    pub fn maxpool1d(&self, kernel: usize, stride: usize) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        if self.rank() != 2 || stride == 0 || kernel == 0 {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: self.shape.clone() });
        }
        let (channels, length) = (self.shape.dims()[0], self.shape.dims()[1]);
        if kernel > length {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: self.shape.clone() });
        }
        let out_length = (length - kernel) / stride + 1;

        let input = self.to_vec_f32()?;

        let mut out = vec![0f32; channels * out_length];
        for c in 0..channels {
            for t in 0..out_length {
                let start = t * stride;
                let window = &input[c * length + start..c * length + start + kernel];
                out[c * out_length + t] = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }
        }
        Tensor::from_f32(out, Shape::new([channels, out_length]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_single_channel_matches_hand_computation() {
        // input [1,5] = [1,2,3,4,5], kernel [1,1,3] = [1,0,-1], stride 1.
        // valid positions: out_len = (5-3)/1 + 1 = 3.
        // out[0] = 1*1 + 2*0 + 3*(-1) = -2
        // out[1] = 2*1 + 3*0 + 4*(-1) = -2
        // out[2] = 3*1 + 4*0 + 5*(-1) = -2
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], [1, 5]).unwrap();
        let w = Tensor::from_f32(vec![1.0, 0.0, -1.0], [1, 1, 3]).unwrap();
        let out = x.conv1d(&w, None, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![-2.0, -2.0, -2.0]);
    }

    #[test]
    fn conv1d_applies_bias_and_stride() {
        // input [1,5] = [1,2,3,4,5], kernel [1,1,2] = [1,1], bias [10], stride 2.
        // out_len = (5-2)/2 + 1 = 2. positions t=0 (idx 0..2), t=1 (idx 2..4).
        // out[0] = (1+2) + 10 = 13 ; out[1] = (3+4) + 10 = 17
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], [1, 5]).unwrap();
        let w = Tensor::from_f32(vec![1.0, 1.0], [1, 1, 2]).unwrap();
        let bias = Tensor::from_f32(vec![10.0], [1]).unwrap();
        let out = x.conv1d(&w, Some(&bias), 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![13.0, 17.0]);
    }

    #[test]
    fn conv1d_sums_over_input_channels_and_across_output_channels() {
        // 2 in-channels, length 3; 2 out-channels; kernel 2. stride 1 -> out_len 2.
        // input: ch0 = [1,2,3], ch1 = [4,5,6]
        // oc0 weights: ic0=[1,0], ic1=[0,1]; oc1 weights: ic0=[1,1], ic1=[1,1]
        // oc0,t0 = (1*1+2*0) + (4*0+5*1) = 1 + 5 = 6
        // oc0,t1 = (2*1+3*0) + (5*0+6*1) = 2 + 6 = 8
        // oc1,t0 = (1+2) + (4+5) = 12 ; oc1,t1 = (2+3) + (5+6) = 16
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let w = Tensor::from_f32(
            vec![
                1.0, 0.0, 0.0, 1.0, // oc0: ic0=[1,0], ic1=[0,1]
                1.0, 1.0, 1.0, 1.0, // oc1: ic0=[1,1], ic1=[1,1]
            ],
            [2, 2, 2],
        )
        .unwrap();
        let out = x.conv1d(&w, None, 1).unwrap();
        assert_eq!(out.shape().dims(), &[2, 2]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![6.0, 8.0, 12.0, 16.0]);
    }

    #[test]
    fn conv1d_rejects_a_channel_count_mismatch() {
        let x = Tensor::from_f32(vec![1.0; 5], [1, 5]).unwrap();
        let w = Tensor::from_f32(vec![1.0; 6], [1, 2, 3]).unwrap(); // wants 2 in-channels.
        assert!(matches!(x.conv1d(&w, None, 1), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn conv1d_rejects_a_kernel_wider_than_the_input() {
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let w = Tensor::from_f32(vec![1.0; 3], [1, 1, 3]).unwrap();
        assert!(matches!(x.conv1d(&w, None, 1), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn conv1d_rejects_non_f32_input() {
        let x = Tensor::from_i32(vec![1; 5], [1, 5]).unwrap();
        let w = Tensor::from_f32(vec![1.0; 3], [1, 1, 3]).unwrap();
        assert!(matches!(x.conv1d(&w, None, 1), Err(Error::DTypeMismatch { .. })));
    }

    #[test]
    fn maxpool1d_matches_hand_computation() {
        // input [1,6] = [1,3,2,5,4,0], kernel 2, stride 2.
        // out_len = (6-2)/2 + 1 = 3. windows: [1,3]->3, [2,5]->5, [4,0]->4
        let x = Tensor::from_f32(vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0], [1, 6]).unwrap();
        let out = x.maxpool1d(2, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![3.0, 5.0, 4.0]);
    }

    #[test]
    fn maxpool1d_overlapping_windows_per_channel() {
        // 2 channels, length 4, kernel 2, stride 1 -> out_len 3 (overlapping).
        // ch0 = [1,3,2,4]: [1,3]->3, [3,2]->3, [2,4]->4
        // ch1 = [9,0,7,7]: [9,0]->9, [0,7]->7, [7,7]->7
        let x = Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0, 9.0, 0.0, 7.0, 7.0], [2, 4]).unwrap();
        let out = x.maxpool1d(2, 1).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![3.0, 3.0, 4.0, 9.0, 7.0, 7.0]);
    }

    #[test]
    fn maxpool1d_handles_negative_values() {
        // All-negative window: the max must be the least-negative, not 0.
        let x = Tensor::from_f32(vec![-3.0, -1.0, -5.0, -2.0], [1, 4]).unwrap();
        let out = x.maxpool1d(2, 2).unwrap();
        assert_eq!(out.to_vec_f32().unwrap(), vec![-1.0, -2.0]);
    }

    #[test]
    fn maxpool1d_rejects_a_kernel_wider_than_the_input() {
        let x = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        assert!(matches!(x.maxpool1d(3, 1), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn maxpool1d_rejects_non_f32_input() {
        let x = Tensor::from_i32(vec![1; 4], [1, 4]).unwrap();
        assert!(matches!(x.maxpool1d(2, 2), Err(Error::DTypeMismatch { .. })));
    }
}
