//! Owned CPU element buffers.
//!
//! `Storage` is the layer below [`crate::Tensor`] that actually owns bytes.
//! It knows its [`DType`] and how many elements it holds, but nothing about
//! shape, strides, or views — that is `Tensor`'s job, layered on top so that
//! multiple tensors (a reshape, a transpose, a narrowed slice) can share one
//! `Storage` via `Arc` without copying.
//!
//! One variant per non-quantized [`DType`], plus a single [`Storage::Quantized`]
//! variant for every block-quantized format. Quantized formats share a
//! variant — rather than getting one each — because none of them can be
//! addressed as a `Vec<T>` of individually meaningful elements; they are all
//! "a scale (and maybe a min) plus packed sub-byte weights", so the only
//! thing that differs between them is how [`crate::quant`] decodes a block,
//! not how the bytes are stored.

use kopitiam_core::{DType, Error, Result, Shape};

/// An owned CPU buffer of tensor elements.
#[derive(Debug, Clone)]
pub enum Storage {
    F32(Vec<f32>),
    /// Raw IEEE 754 binary16 bits. Stored as `u16` rather than a float type
    /// because Rust's `f32`/`f64` are the only floats with hardware and
    /// library support; every element must go through
    /// [`crate::half::f16_to_f32`] to be computed on.
    F16(Vec<u16>),
    /// Raw bfloat16 bits, likewise `u16`; see [`crate::half::bf16_to_f32`].
    BF16(Vec<u16>),
    I8(Vec<i8>),
    I32(Vec<i32>),
    /// Any block-quantized format ([`DType::Q4_0`], [`DType::Q4_1`],
    /// [`DType::Q5_0`], [`DType::Q5_1`], [`DType::Q8_0`]): raw on-disk block
    /// bytes plus the tag saying how to decode them. Never indexed
    /// elementwise directly — see [`DType::is_quantized`] and
    /// [`crate::quant`].
    Quantized { dtype: DType, bytes: Vec<u8> },
}

impl Storage {
    /// The element type this storage holds.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::I8(_) => DType::I8,
            Self::I32(_) => DType::I32,
            Self::Quantized { dtype, .. } => *dtype,
        }
    }

    /// Number of logical elements this storage holds.
    ///
    /// For quantized storage this is derived from the byte length (every
    /// byte belongs to some block, and every block decodes to exactly
    /// [`DType::block_size`] elements) rather than tracked separately,
    /// which is only sound because [`Self::new_quantized`] is the sole
    /// constructor and it guarantees the byte length is a whole number of
    /// blocks.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) | Self::BF16(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::Quantized { dtype, bytes } => bytes.len() / dtype.block_bytes() * dtype.block_size(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Builds quantized storage for `elem_count` elements of `dtype` from
    /// raw block bytes.
    ///
    /// Rejects two distinct failure modes with distinct errors, both
    /// already defined by `kopitiam-core` for exactly this purpose:
    ///
    /// * `elem_count` is not a whole number of blocks (e.g. 33 elements of
    ///   a 32-wide format) -> [`Error::PartialQuantizedBlock`]. This is not
    ///   a rounding question — there is no valid byte layout for a partial
    ///   block, so it is rejected before ever looking at `bytes`.
    /// * `bytes` does not contain exactly the number of bytes that
    ///   `elem_count` blocks require -> [`Error::StorageTooSmall`]. This
    ///   catches truncated reads and mismatched shape/data pairs at
    ///   construction time instead of an out-of-bounds panic on first use.
    pub fn new_quantized(dtype: DType, bytes: Vec<u8>, elem_count: usize) -> Result<Self> {
        let expected = dtype
            .storage_bytes(elem_count)
            .ok_or(Error::PartialQuantizedBlock {
                dtype,
                count: elem_count,
                block_size: dtype.block_size(),
            })?;
        if bytes.len() != expected {
            return Err(Error::StorageTooSmall {
                shape: Shape::new([elem_count]),
                dtype,
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self::Quantized { dtype, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_and_len_match_the_variant() {
        assert_eq!(Storage::F32(vec![1.0, 2.0, 3.0]).dtype(), DType::F32);
        assert_eq!(Storage::F32(vec![1.0, 2.0, 3.0]).len(), 3);
        assert_eq!(Storage::I32(vec![1, 2]).dtype(), DType::I32);
    }

    #[test]
    fn quantized_storage_reports_decoded_element_count() {
        // 2 blocks of Q4_0 (18 bytes each) = 64 elements.
        let storage = Storage::new_quantized(DType::Q4_0, vec![0u8; 36], 64).unwrap();
        assert_eq!(storage.len(), 64);
        assert_eq!(storage.dtype(), DType::Q4_0);
    }

    #[test]
    fn a_partial_block_element_count_is_rejected() {
        // 33 is not a multiple of Q4_0's 32-element block.
        let err = Storage::new_quantized(DType::Q4_0, vec![0u8; 18], 33).unwrap_err();
        assert!(matches!(err, Error::PartialQuantizedBlock { count: 33, .. }));
    }

    #[test]
    fn mismatched_byte_length_is_rejected() {
        // 64 elements of Q4_0 need 36 bytes; give it 18.
        let err = Storage::new_quantized(DType::Q4_0, vec![0u8; 18], 64).unwrap_err();
        assert!(matches!(
            err,
            Error::StorageTooSmall {
                expected: 36,
                actual: 18,
                ..
            }
        ));
    }

    #[test]
    fn zero_elements_is_a_valid_empty_quantized_tensor() {
        let storage = Storage::new_quantized(DType::Q8_0, vec![], 0).unwrap();
        assert!(storage.is_empty());
    }
}
