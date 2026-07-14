//! Concatenating tensors along an axis.
//!
//! Unlike `narrow`/`transpose`/`broadcast_to`, concatenation cannot be
//! zero-copy — the output genuinely interleaves bytes from multiple
//! inputs — so this is the one shape op in the `tensor` module family that
//! always allocates.

use kopitiam_core::{DType, Error, Result, Shape};

use crate::storage::Storage;

use super::Tensor;

impl Tensor {
    /// Concatenates `tensors` along `dim`. All tensors must share a dtype
    /// (any non-quantized dtype — this is pure data movement, not
    /// arithmetic, so it is not restricted to `f32` the way the math ops
    /// are) and must agree on every dimension except `dim`.
    pub fn concat(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        let Some(first) = tensors.first() else {
            // No tensor to take a dtype/rank from; there is no sensible
            // shape for an empty concatenation.
            return Err(Error::ShapeMismatch { expected: Shape::scalar(), actual: Shape::scalar() });
        };
        let dtype = first.dtype();
        if dtype.is_quantized() {
            return Err(Error::QuantizedElementAccess { dtype, block_size: dtype.block_size() });
        }
        let rank = first.rank();
        if dim >= rank {
            return Err(Error::IndexOutOfBounds { dim, index: dim, len: rank });
        }
        for t in tensors {
            if t.dtype() != dtype {
                return Err(Error::DTypeMismatch { expected: dtype, actual: t.dtype() });
            }
            if t.rank() != rank {
                return Err(Error::ShapeMismatch { expected: first.shape.clone(), actual: t.shape.clone() });
            }
            for (d, (&a, &b)) in first.shape.dims().iter().zip(t.shape.dims()).enumerate() {
                if d != dim && a != b {
                    return Err(Error::ShapeMismatch { expected: first.shape.clone(), actual: t.shape.clone() });
                }
            }
        }

        let mut out_dims = first.shape.dims().to_vec();
        out_dims[dim] = tensors.iter().map(|t| t.shape.dims()[dim]).sum();
        let out_shape = Shape::new(out_dims.clone());
        let outer: usize = out_dims[..dim].iter().product();
        let inner: usize = out_dims[dim + 1..].iter().product();

        let contiguous: Vec<Tensor> = tensors.iter().map(Tensor::contiguous).collect::<Result<_>>()?;

        // One macro arm per `Storage` variant instead of five hand-written
        // copies of the same "slice out each input's chunk along `dim`,
        // append to `out`" loop, which differ only in element type.
        macro_rules! concat_variant {
            ($variant:ident, $ty:ty) => {{
                let mut out: Vec<$ty> = Vec::with_capacity(out_shape.elem_count());
                for o in 0..outer {
                    for t in &contiguous {
                        let Storage::$variant(src) = t.storage.as_ref() else { unreachable!() };
                        let len_d = t.shape.dims()[dim];
                        let start = o * len_d * inner;
                        out.extend_from_slice(&src[start..start + len_d * inner]);
                    }
                }
                Storage::$variant(out)
            }};
        }

        let storage = match dtype {
            DType::F32 => concat_variant!(F32, f32),
            DType::F16 => concat_variant!(F16, u16),
            DType::BF16 => concat_variant!(BF16, u16),
            DType::I8 => concat_variant!(I8, i8),
            DType::I32 => concat_variant!(I32, i32),
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0 => {
                unreachable!("quantized dtypes are rejected above")
            }
        };

        Ok(Tensor {
            storage: std::sync::Arc::new(storage),
            strides: out_shape.strides(),
            shape: out_shape,
            offset: 0,
            device: first.device,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_along_the_last_axis_interleaves_rows_correctly() {
        // [[1,2],[3,4]] concat [[5,6],[7,8]] along axis 1 -> [[1,2,5,6],[3,4,7,8]]
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();
        let c = Tensor::concat(&[a, b], 1).unwrap();
        assert_eq!(c.shape().dims(), &[2, 4]);
        assert_eq!(c.to_vec_f32().unwrap(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn concat_along_the_first_axis_stacks_rows() {
        let a = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let b = Tensor::from_f32(vec![3.0, 4.0], [1, 2]).unwrap();
        let c = Tensor::concat(&[a, b], 0).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_three_tensors_matches_hand_computation() {
        let a = Tensor::from_f32(vec![1.0], [1]).unwrap();
        let b = Tensor::from_f32(vec![2.0], [1]).unwrap();
        let c = Tensor::from_f32(vec![3.0], [1]).unwrap();
        let out = Tensor::concat(&[a, b, c], 0).unwrap();
        assert_eq!(out.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_rejects_mismatched_non_concat_dimensions() {
        let a = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let b = Tensor::from_f32(vec![1.0, 2.0, 3.0], [1, 3]).unwrap();
        assert!(matches!(Tensor::concat(&[a, b], 0), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn concat_of_non_contiguous_views_still_produces_correct_data() {
        let base = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let transposed = base.transpose(0, 1).unwrap(); // [3, 2]: [[1,4],[2,5],[3,6]]
        let other = Tensor::from_f32(vec![9.0, 9.0], [1, 2]).unwrap();
        let out = Tensor::concat(&[transposed, other], 0).unwrap();
        assert_eq!(out.to_vec_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 9.0, 9.0]);
    }
}
