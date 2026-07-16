//! [`Tensor`]: the shared-storage, strided, CPU tensor at the center of the
//! Kopitiam Runtime.
//!
//! This module owns the struct, its constructors, its accessors, and the
//! data-movement operations that need direct access to `storage`/`strides`/
//! `offset` (dtype conversion, reshape, transpose, narrow, broadcast,
//! concat). The math operations (matmul, elementwise arithmetic, softmax,
//! reductions, normalization, activations, embedding gather) live in
//! sibling modules — `matmul.rs`, `elementwise.rs`, and so on — each adding
//! another `impl Tensor` block. They are child modules of this one
//! specifically so they can see `Tensor`'s private fields directly, the
//! same way methods in a single large `impl` block would, without forcing
//! every field to be `pub(crate)` (and therefore visible to the rest of the
//! crate, not just the tensor family of modules).

mod matmul;
mod elementwise;
mod softmax;
mod reduce;
mod norm;
mod activation;
mod gather;
mod concat;

use std::sync::Arc;

use kopitiam_core::{DType, Device, Error, Result, Shape};

use crate::half;
use crate::quant;
use crate::storage::Storage;

/// A CPU tensor: a [`Shape`] plus a strided view into a shared, ref-counted
/// [`Storage`] buffer.
///
/// # Why `Arc<Storage>` plus separate strides and an offset
///
/// Model weights are hundreds of megabytes; every op that produces a new
/// logical view of existing data — `reshape`, `transpose`, `narrow`,
/// `broadcast_to` — must not copy that data. Sharing `storage` via `Arc`
/// makes `Tensor::clone()` and every pure view operation an O(1) pointer
/// bump plus a small `Shape`/`Vec<usize>` copy, never an O(n) memcpy of
/// element data. `strides` (in elements, one per dimension) and `offset`
/// (in elements, into the shared buffer) are what let a view disagree with
/// its storage about layout: `narrow` changes `offset`, `transpose`
/// reorders `strides`, `broadcast_to` introduces zero strides.
///
/// # Why quantized tensors don't get real views
///
/// A [`DType::is_quantized`] tensor's `strides` are always its shape's
/// canonical strides and its `offset` is always `0` — block-quantized data
/// cannot be addressed at anything finer than a whole block, so "a strided
/// view into the middle of a Q4_0 tensor" is not a representable concept.
/// Every op that would need one ([`Tensor::narrow`], [`Tensor::transpose`],
/// [`Tensor::broadcast_to`], the arithmetic and normalization ops) rejects
/// quantized tensors with [`Error::QuantizedElementAccess`] via
/// [`Tensor::require_elementwise`] and [`Tensor::require_dtype`]. Call
/// [`Tensor::to_dtype`] with [`DType::F32`] first.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(super) storage: Arc<Storage>,
    pub(super) shape: Shape,
    /// Element strides, one per dimension of `shape` (same rank).
    pub(super) strides: Vec<usize>,
    /// Element offset into `storage`'s flat buffer.
    pub(super) offset: usize,
    pub(super) device: Device,
}

impl Tensor {
    // -- Constructors --------------------------------------------------

    /// Builds a contiguous `f32` tensor from `data`, which must hold
    /// exactly `shape.elem_count()` elements.
    pub fn from_f32(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        check_len(&shape, DType::F32, data.len())?;
        Ok(Self::new_contiguous(Storage::F32(data), shape))
    }

    /// Builds a tensor from raw IEEE 754 binary16 bits. Values are decoded
    /// lazily by ops via [`Tensor::to_dtype`]; nothing here interprets them.
    pub fn from_f16(data: Vec<u16>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        check_len(&shape, DType::F16, data.len())?;
        Ok(Self::new_contiguous(Storage::F16(data), shape))
    }

    /// Builds a tensor from raw bfloat16 bits.
    pub fn from_bf16(data: Vec<u16>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        check_len(&shape, DType::BF16, data.len())?;
        Ok(Self::new_contiguous(Storage::BF16(data), shape))
    }

    pub fn from_i8(data: Vec<i8>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        check_len(&shape, DType::I8, data.len())?;
        Ok(Self::new_contiguous(Storage::I8(data), shape))
    }

    /// Builds an `i32` tensor — the dtype [`kopitiam_core::DType`] documents
    /// as "mostly for token ids and indices", e.g. the input to
    /// [`Tensor::gather_rows`].
    pub fn from_i32(data: Vec<i32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        check_len(&shape, DType::I32, data.len())?;
        Ok(Self::new_contiguous(Storage::I32(data), shape))
    }

    /// Builds a block-quantized tensor from raw on-disk block bytes.
    ///
    /// `bytes` must be exactly the number of bytes `shape.elem_count()`
    /// elements of `dtype` require; see [`Storage::new_quantized`] for the
    /// precise failure modes ([`Error::PartialQuantizedBlock`],
    /// [`Error::StorageTooSmall`]).
    pub fn from_quantized(dtype: DType, bytes: Vec<u8>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::new_quantized(dtype, bytes, shape.elem_count()).map_err(|e| {
            match e {
                // Re-attach the tensor's real (possibly multi-dimensional)
                // shape rather than the synthetic 1-D shape Storage used,
                // since Storage doesn't know about Shape's dimensions.
                Error::StorageTooSmall { dtype, expected, actual, .. } => Error::StorageTooSmall {
                    shape: shape.clone(),
                    dtype,
                    expected,
                    actual,
                },
                other => other,
            }
        })?;
        Ok(Self::new_contiguous(storage, shape))
    }

    /// A tensor of `f32` zeros. Convenience for tests and for initializing
    /// accumulators; not meant as a general "any dtype" factory, since
    /// every op in this crate computes in `f32` (see the crate docs).
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let data = vec![0.0f32; shape.elem_count()];
        Self::new_contiguous(Storage::F32(data), shape)
    }

    fn new_contiguous(storage: Storage, shape: Shape) -> Self {
        let strides = shape.strides();
        Self {
            storage: Arc::new(storage),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    // -- Accessors --------------------------------------------------------

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// This view's element strides. Not necessarily the canonical strides
    /// of `shape()` — see [`Tensor::is_contiguous`].
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Whether this view's elements sit sequentially in storage in
    /// row-major order with no gaps, i.e. whether `strides()` equals
    /// `shape().strides()`. A fresh tensor from any `from_*` constructor is
    /// always contiguous; `narrow`, `transpose`, and `broadcast_to` can all
    /// produce non-contiguous views.
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.strides()
    }

    // -- Element access helpers (used throughout this module family) -----

    /// Storage offsets, in this tensor's row-major logical element order.
    ///
    /// This is the one place broadcasting (zero strides), slicing (nonzero
    /// offset), and transposition (permuted strides) all reduce to the same
    /// formula, so every op that needs to walk a tensor's elements —
    /// materializing to a `Vec`, elementwise arithmetic, reductions,
    /// gather — shares this instead of hand-rolling nested loops per rank.
    pub(super) fn logical_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let dims = self.shape.dims();
        // Canonical (contiguous) strides for this tensor's own shape are
        // exactly the divisors needed to decompose a flat row-major index
        // `flat` into per-dimension coordinates: `coord[d] = (flat /
        // canonical[d]) % dims[d]`.
        let canonical = self.shape.strides();
        let total = self.shape.elem_count();
        (0..total).map(move |flat| {
            dims.iter().enumerate().fold(self.offset, |acc, (d, &dim)| {
                if dim == 0 {
                    return acc;
                }
                let coord = (flat / canonical[d]) % dim;
                acc + coord * self.strides[d]
            })
        })
    }

    /// Errors unless this tensor's dtype is exactly `dtype`.
    ///
    /// Used by every arithmetic/normalization/activation op, all of which
    /// are implemented for `f32` only (see the crate docs for why). Because
    /// every quantized dtype differs from `DType::F32`, this doubles as the
    /// "reject quantized input" check for those ops — callers see a single
    /// clear [`Error::DTypeMismatch`] rather than two different error paths
    /// depending on *which* wrong dtype they passed.
    pub(super) fn require_dtype(&self, dtype: DType) -> Result<()> {
        if self.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                actual: self.dtype(),
            });
        }
        Ok(())
    }

    /// Errors if this tensor's dtype is block-quantized.
    ///
    /// Used by the pure data-movement ops (`narrow`, `transpose`,
    /// `broadcast_to`, `contiguous`) that work for *any* addressable dtype,
    /// not just `f32` — unlike [`Tensor::require_dtype`], which pins to one
    /// specific type.
    pub(super) fn require_elementwise(&self) -> Result<()> {
        if self.dtype().is_quantized() {
            return Err(Error::QuantizedElementAccess {
                dtype: self.dtype(),
                block_size: self.dtype().block_size(),
            });
        }
        Ok(())
    }

    // -- Materialization ----------------------------------------------------

    /// Copies this view's elements out as a plain `Vec<f32>` in row-major
    /// order. Requires `dtype() == DType::F32`; call [`Tensor::to_dtype`]
    /// first for any other dtype, including the quantized ones.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.require_dtype(DType::F32)?;
        let Storage::F32(data) = self.storage.as_ref() else {
            unreachable!("require_dtype(F32) guarantees Storage::F32")
        };
        Ok(self.logical_offsets().map(|i| data[i]).collect())
    }

    /// Copies this view's elements out as a plain `Vec<i32>` in row-major
    /// order. The `i32` twin of [`Tensor::to_vec_f32`]: needs
    /// `dtype() == DType::I32`, errors [`Error::DTypeMismatch`] otherwise.
    ///
    /// This is how you read the *output* of an index-producing op — namely
    /// [`Tensor::argmax`], whose result is the token ids a greedy sampler
    /// hands back to the runtime. Without it argmax would be a dead end: you
    /// could compute the winning indices but never get them out as plain
    /// numbers. (There's deliberately no `i32 <-> f32` cast anywhere in this
    /// crate — see [`Tensor::to_dtype`] — so this accessor is the *only* way
    /// I32 data leaves a tensor, same as `to_vec_f32` is for F32.)
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.require_dtype(DType::I32)?;
        let Storage::I32(data) = self.storage.as_ref() else {
            unreachable!("require_dtype(I32) guarantees Storage::I32")
        };
        Ok(self.logical_offsets().map(|i| data[i]).collect())
    }

    /// Returns a tensor equal to `self` but guaranteed contiguous, copying
    /// only if `self` is not already contiguous (in which case it is a
    /// cheap `Arc` clone).
    ///
    /// This is the shared implementation behind [`Tensor::reshape`] and
    /// [`Tensor::concat`]: both need row-major-sequential data to work with
    /// and neither cares whether that came for free or required a copy.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        self.require_elementwise()?;
        let offsets: Vec<usize> = self.logical_offsets().collect();
        let storage = match self.storage.as_ref() {
            Storage::F32(d) => Storage::F32(offsets.iter().map(|&i| d[i]).collect()),
            Storage::F16(d) => Storage::F16(offsets.iter().map(|&i| d[i]).collect()),
            Storage::BF16(d) => Storage::BF16(offsets.iter().map(|&i| d[i]).collect()),
            Storage::I8(d) => Storage::I8(offsets.iter().map(|&i| d[i]).collect()),
            Storage::I32(d) => Storage::I32(offsets.iter().map(|&i| d[i]).collect()),
            Storage::Quantized { .. } => unreachable!("require_elementwise excludes quantized"),
        };
        Ok(Self::new_contiguous(storage, self.shape.clone()))
    }

    // -- Dtype conversion -------------------------------------------------

    /// Converts to `dtype`, decoding as needed.
    ///
    /// Converting to `dtype() == dtype` is a cheap `Arc` clone. Converting
    /// *to* [`DType::F32`] is supported from every dtype this crate knows,
    /// including every quantized format (this is the dequantization entry
    /// point — see [`crate::quant`] for the block layouts) and `f16`/`bf16`
    /// (see [`crate::half`]). Converting *from* `f32` to `f16` or `bf16` is
    /// also supported, for storing activations compactly.
    ///
    /// Not supported: converting `f32` back to a quantized format.
    /// Requantization is a model-export/training-time concern — it
    /// requires choosing a quantization *scheme* (per-tensor vs per-block
    /// scale, calibration, error-diffusion) — not a forward-pass inference
    /// concern, which is this crate's whole scope. `i8`/`i32` casts to or
    /// from `f32` are likewise not implemented: nothing in the ops this
    /// crate provides needs them (token ids are consumed as `i32` directly
    /// by [`Tensor::gather_rows`], never mixed arithmetically with `f32`).
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        match (self.storage.as_ref(), dtype) {
            (Storage::Quantized { dtype: source, bytes }, DType::F32) => {
                let data = quant::dequantize(*source, bytes)?;
                Tensor::from_f32(data, self.shape.clone())
            }
            (Storage::F16(_), DType::F32) => {
                let data: Vec<f32> = self.raw_f16().iter().copied().map(half::f16_to_f32).collect();
                let data = self.reorder(&data);
                Tensor::from_f32(data, self.shape.clone())
            }
            (Storage::BF16(_), DType::F32) => {
                let data: Vec<f32> = self.raw_bf16().iter().copied().map(half::bf16_to_f32).collect();
                let data = self.reorder(&data);
                Tensor::from_f32(data, self.shape.clone())
            }
            (Storage::F32(d), DType::F16) => {
                let data: Vec<u16> = self.logical_offsets().map(|i| half::f32_to_f16(d[i])).collect();
                Tensor::from_f16(data, self.shape.clone())
            }
            (Storage::F32(d), DType::BF16) => {
                let data: Vec<u16> = self.logical_offsets().map(|i| half::f32_to_bf16(d[i])).collect();
                Tensor::from_bf16(data, self.shape.clone())
            }
            _ => Err(Error::UnsupportedDType { op: "to_dtype", dtype }),
        }
    }

    fn raw_f16(&self) -> &[u16] {
        match self.storage.as_ref() {
            Storage::F16(d) => d,
            _ => unreachable!(),
        }
    }

    fn raw_bf16(&self) -> &[u16] {
        match self.storage.as_ref() {
            Storage::BF16(d) => d,
            _ => unreachable!(),
        }
    }

    /// Reorders an already-decoded, storage-order `Vec<f32>` (one entry per
    /// raw storage element) into this view's logical row-major order.
    fn reorder(&self, decoded: &[f32]) -> Vec<f32> {
        self.logical_offsets().map(|i| decoded[i]).collect()
    }

    // -- Shape-only views (zero-copy) --------------------------------------

    /// Returns a view with `dim0` and `dim1` swapped. Zero-copy: only the
    /// shape and strides change.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        self.require_elementwise()?;
        let rank = self.rank();
        if dim0 >= rank {
            return Err(Error::IndexOutOfBounds { dim: dim0, index: dim0, len: rank });
        }
        if dim1 >= rank {
            return Err(Error::IndexOutOfBounds { dim: dim1, index: dim1, len: rank });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        dims.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: Shape::new(dims),
            strides,
            offset: self.offset,
            device: self.device,
        })
    }

    /// Returns the sub-view `[start, start + len)` along `dim`. Zero-copy:
    /// only the shape and offset change.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        self.require_elementwise()?;
        let rank = self.rank();
        if dim >= rank {
            return Err(Error::IndexOutOfBounds { dim, index: dim, len: rank });
        }
        let dim_len = self.shape.dims()[dim];
        if start + len > dim_len {
            return Err(Error::IndexOutOfBounds { dim, index: start + len, len: dim_len });
        }
        let mut dims = self.shape.dims().to_vec();
        dims[dim] = len;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: Shape::new(dims),
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[dim],
            device: self.device,
        })
    }

    /// Returns a view broadcast to `shape`, following [`Shape::broadcast`]'s
    /// NumPy right-aligned rule. Zero-copy: dimensions being stretched from
    /// size 1 get stride 0, so every "broadcast" element reads the same
    /// underlying value.
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        self.require_elementwise()?;
        let target = shape.into();
        let result = self.shape.broadcast(&target)?;
        if result != target {
            return Err(Error::NotBroadcastable { left: self.shape.clone(), right: target });
        }
        let rank = target.rank();
        let mut strides = vec![0usize; rank];
        let offset_in_rank = rank - self.rank();
        for i in 0..self.rank() {
            if self.shape.dims()[i] != 1 {
                strides[offset_in_rank + i] = self.strides[i];
            }
            // else: dim is being stretched from 1, so it keeps stride 0
            // (already the vec's default).
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: target,
            strides,
            offset: self.offset,
            device: self.device,
        })
    }

    /// Reinterprets this tensor as `dims`, which must describe the same
    /// number of elements. Zero-copy when `self` is already contiguous;
    /// otherwise materializes a contiguous copy first (see
    /// [`Tensor::contiguous`]).
    pub fn reshape(&self, dims: impl Into<Vec<usize>>) -> Result<Tensor> {
        let new_shape = self.shape.reshape(dims)?;
        let base = self.contiguous()?;
        Ok(Tensor {
            storage: base.storage.clone(),
            strides: new_shape.strides(),
            shape: new_shape,
            offset: base.offset,
            device: base.device,
        })
    }
}

/// Checks that `actual_elems` matches what `shape` demands for `dtype`,
/// producing an [`Error::StorageTooSmall`] (expressed in bytes, matching
/// the error's own documented units) if not.
fn check_len(shape: &Shape, dtype: DType, actual_elems: usize) -> Result<()> {
    let expected_elems = shape.elem_count();
    if actual_elems != expected_elems {
        let expected = dtype.storage_bytes(expected_elems).unwrap_or(expected_elems * dtype.block_bytes());
        let actual = dtype.storage_bytes(actual_elems).unwrap_or(actual_elems * dtype.block_bytes());
        return Err(Error::StorageTooSmall {
            shape: shape.clone(),
            dtype,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f32_rejects_a_length_mismatch() {
        let err = Tensor::from_f32(vec![1.0, 2.0], [3]).unwrap_err();
        assert!(matches!(err, Error::StorageTooSmall { .. }));
    }

    #[test]
    fn basic_accessors_report_shape_and_dtype() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.elem_count(), 6);
        assert!(t.is_contiguous());
    }

    #[test]
    fn clone_shares_storage_without_copying() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let cloned = t.clone();
        assert!(Arc::ptr_eq(&t.storage, &cloned.storage));
    }

    #[test]
    fn transpose_is_zero_copy_and_produces_correct_logical_order() {
        // [[1, 2, 3], [4, 5, 6]] -> transposed -> [[1, 4], [2, 5], [3, 6]]
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        assert!(Arc::ptr_eq(&t.storage, &tt.storage), "transpose must not copy");
        assert_eq!(tt.shape().dims(), &[3, 2]);
        assert!(!tt.is_contiguous());
        assert_eq!(tt.to_vec_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn narrow_is_zero_copy_and_slices_the_requested_range() {
        let t = Tensor::from_f32((0..12).map(|v| v as f32).collect(), [4, 3]).unwrap();
        let n = t.narrow(0, 1, 2).unwrap();
        assert!(Arc::ptr_eq(&t.storage, &n.storage), "narrow must not copy");
        assert_eq!(n.shape().dims(), &[2, 3]);
        assert_eq!(n.to_vec_f32().unwrap(), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn narrow_out_of_range_is_rejected() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0], [3]).unwrap();
        assert!(matches!(t.narrow(0, 1, 5), Err(Error::IndexOutOfBounds { .. })));
    }

    #[test]
    fn broadcast_to_is_zero_copy_and_repeats_the_stretched_dimension() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0], [1, 3]).unwrap();
        let b = t.broadcast_to([4, 3]).unwrap();
        assert!(Arc::ptr_eq(&t.storage, &b.storage), "broadcast_to must not copy");
        assert_eq!(b.strides()[0], 0);
        assert_eq!(b.to_vec_f32().unwrap(), vec![
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
        ]);
    }

    #[test]
    fn reshape_is_zero_copy_when_contiguous() {
        let t = Tensor::from_f32((0..6).map(|v| v as f32).collect(), [2, 3]).unwrap();
        let r = t.reshape([3, 2]).unwrap();
        assert!(Arc::ptr_eq(&t.storage, &r.storage), "reshape of a contiguous tensor must not copy");
        assert_eq!(r.to_vec_f32().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reshape_after_transpose_materializes_a_copy_but_stays_correct() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let r = t.transpose(0, 1).unwrap().reshape([6]).unwrap();
        assert_eq!(r.to_vec_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn reshape_rejects_an_element_count_mismatch() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        assert!(matches!(t.reshape([3]), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn quantized_tensors_reject_view_operations() {
        let bytes = vec![0u8; 18]; // one Q4_0 block, all-zero
        let t = Tensor::from_quantized(DType::Q4_0, bytes, [32]).unwrap();
        assert!(matches!(t.narrow(0, 0, 1), Err(Error::QuantizedElementAccess { .. })));
        assert!(matches!(t.transpose(0, 0), Err(Error::QuantizedElementAccess { .. })));
        assert!(matches!(t.broadcast_to([2, 32]), Err(Error::QuantizedElementAccess { .. })));
        assert!(matches!(t.to_vec_f32(), Err(Error::DTypeMismatch { .. })));
    }

    #[test]
    fn to_dtype_identity_is_a_cheap_clone() {
        let t = Tensor::from_f32(vec![1.0, 2.0], [2]).unwrap();
        let same = t.to_dtype(DType::F32).unwrap();
        assert!(Arc::ptr_eq(&t.storage, &same.storage));
    }

    #[test]
    fn to_dtype_f16_round_trips_through_f32() {
        let t = Tensor::from_f32(vec![1.0, 2.5, -3.0, 0.0], [4]).unwrap();
        let as_f16 = t.to_dtype(DType::F16).unwrap();
        assert_eq!(as_f16.dtype(), DType::F16);
        let back = as_f16.to_dtype(DType::F32).unwrap();
        assert_eq!(back.to_vec_f32().unwrap(), vec![1.0, 2.5, -3.0, 0.0]);
    }

    #[test]
    fn to_dtype_bf16_round_trips_through_f32() {
        let t = Tensor::from_f32(vec![1.0, 2.5, -3.0, 0.0], [4]).unwrap();
        let as_bf16 = t.to_dtype(DType::BF16).unwrap();
        let back = as_bf16.to_dtype(DType::F32).unwrap();
        assert_eq!(back.to_vec_f32().unwrap(), vec![1.0, 2.5, -3.0, 0.0]);
    }

    #[test]
    fn to_dtype_respects_a_transposed_views_logical_order() {
        // Regression check for the f16/bf16 conversion path: it must read
        // through `logical_offsets`, not raw storage order, or a converted
        // transposed tensor would silently come out in the wrong order.
        let t = Tensor::from_f16(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].map(half::f32_to_f16).to_vec(),
            [2, 3],
        )
        .unwrap();
        let transposed = t.transpose(0, 1).unwrap();
        let f32_transposed = transposed.to_dtype(DType::F32).unwrap();
        assert_eq!(f32_transposed.to_vec_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn to_dtype_unsupported_pair_is_rejected() {
        let t = Tensor::from_i32(vec![1, 2, 3], [3]).unwrap();
        assert!(matches!(t.to_dtype(DType::F32), Err(Error::UnsupportedDType { .. })));
    }
}
