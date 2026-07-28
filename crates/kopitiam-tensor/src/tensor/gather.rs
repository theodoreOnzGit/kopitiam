//! Embedding-style row gather.

use kopitiam_core::{DType, Error, Result, Shape};

use crate::storage::Storage;

use super::Tensor;

impl Tensor {
    /// Gathers rows of a `[vocab, hidden]` embedding table by token id.
    ///
    /// `indices` (dtype [`DType::I32`], any shape) holds token ids in
    /// `[0, vocab)`. The result has shape `indices.shape() + [hidden]` —
    /// every index becomes one full row of `self`.
    ///
    /// # Why this is not `torch.gather`
    ///
    /// PyTorch's `gather(dim, index)` requires `index` to have the *same
    /// rank* as the source and picks one scalar per output position along
    /// `dim`. That is a general primitive this crate deliberately does not
    /// implement: nothing in a forward pass needs it. What every
    /// transformer *does* need is embedding lookup — pick whole rows out
    /// of a `[vocab, hidden]` table by a batch of token ids — which is a
    /// different (simpler, more common) shape of operation with its own
    /// name here rather than a special case of a more general `gather`
    /// that this crate does not otherwise use.
    ///
    /// # Quantized tables are read in place
    ///
    /// A block-quantized table is gathered from directly, decoding only the
    /// requested rows — see [`Tensor::gather_rows_quantized`]. The result is
    /// always `f32` either way, so callers cannot tell the difference except in
    /// memory use.
    pub fn gather_rows(&self, indices: &Tensor) -> Result<Tensor> {
        indices.require_dtype(DType::I32)?;
        if self.rank() != 2 {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: self.shape.clone() });
        }
        // Decode only the requested rows when the table is still quantized —
        // see `gather_rows_quantized` for why that matters.
        if let Storage::Quantized { dtype, bytes } = self.storage.as_ref() {
            return self.gather_rows_quantized(indices, *dtype, bytes);
        }
        self.require_dtype(DType::F32)?;
        self.gather_rows_f32(indices)
    }

    /// Gathers rows out of a **block-quantized** table, decoding only the rows
    /// actually asked for.
    ///
    /// # Why this exists
    ///
    /// An embedding table is the largest tensor in a small model — SmolLM2-360M's
    /// is `49152 x 960` — and a forward pass reads a handful of its rows: one per
    /// prompt token, often exactly one during decode. Dequantizing the whole
    /// table to `f32` just to read those rows costs 188 MB of resident memory
    /// against 47 MB of Q8_0, a 4x expansion of the single biggest allocation in
    /// the model, and it is pure waste on a phone.
    ///
    /// Rows are addressable because a row is a whole number of blocks: with
    /// `hidden` a multiple of the dtype's block size, row `r` occupies exactly
    /// `blocks_per_row` consecutive blocks starting at `r * blocks_per_row`. So
    /// a row decodes on its own, with no need to touch its neighbours.
    ///
    /// Requires a contiguous, unoffset view — quantized bytes have no strides to
    /// walk, so an arbitrary view could not be honoured, and silently ignoring
    /// one would return the wrong rows.
    fn gather_rows_quantized(
        &self,
        indices: &Tensor,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<Tensor> {
        let vocab = self.shape.dims()[0];
        let hidden = self.shape.dims()[1];
        let block_size = dtype.block_size();
        let block_bytes = dtype.block_bytes();

        // A row that straddles a block boundary is not independently decodable,
        // and neither is a strided view. Refuse rather than return wrong rows.
        if !hidden.is_multiple_of(block_size)
            || !self.is_contiguous()
            || self.offset != 0
        {
            return Err(Error::UnsupportedDType { op: "gather_rows", dtype });
        }
        let blocks_per_row = hidden / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        let Storage::I32(idx_data) = indices.storage.as_ref() else { unreachable!() };
        let mut out = Vec::with_capacity(indices.elem_count() * hidden);
        for offset in indices.logical_offsets() {
            let id = idx_data[offset];
            if id < 0 || id as usize >= vocab {
                return Err(Error::IndexOutOfBounds {
                    dim: 0,
                    index: id.max(0) as usize,
                    len: vocab,
                });
            }
            let start = id as usize * row_bytes;
            let row = bytes
                .get(start..start + row_bytes)
                .ok_or(Error::IndexOutOfBounds { dim: 0, index: id as usize, len: vocab })?;
            out.extend(crate::quant::dequantize(dtype, row)?);
        }

        let mut out_dims = indices.shape.dims().to_vec();
        out_dims.push(hidden);
        Tensor::from_f32(out, Shape::new(out_dims))
    }

    /// The `f32` path: honours strides and offset, so a transposed or sliced
    /// view of a table gathers correctly.
    fn gather_rows_f32(&self, indices: &Tensor) -> Result<Tensor> {
        let vocab = self.shape.dims()[0];
        let hidden = self.shape.dims()[1];

        let Storage::F32(data) = self.storage.as_ref() else { unreachable!() };
        let Storage::I32(idx_data) = indices.storage.as_ref() else { unreachable!() };

        let mut out = Vec::with_capacity(indices.elem_count() * hidden);
        for offset in indices.logical_offsets() {
            let id = idx_data[offset];
            if id < 0 || id as usize >= vocab {
                return Err(Error::IndexOutOfBounds {
                    dim: 0,
                    index: id.max(0) as usize,
                    len: vocab,
                });
            }
            let row_start = self.offset + id as usize * self.strides[0];
            for c in 0..hidden {
                out.push(data[row_start + c * self.strides[1]]);
            }
        }

        let mut out_dims = indices.shape.dims().to_vec();
        out_dims.push(hidden);
        Tensor::from_f32(out, Shape::new(out_dims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_rows_picks_the_requested_embedding_rows() {
        // vocab=4, hidden=2: rows are [0,1], [2,3], [4,5], [6,7].
        let table = Tensor::from_f32((0..8).map(|v| v as f32).collect(), [4, 2]).unwrap();
        let ids = Tensor::from_i32(vec![2, 0, 3], [3]).unwrap();
        let out = table.gather_rows(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![4.0, 5.0, 0.0, 1.0, 6.0, 7.0]);
    }

    #[test]
    fn gather_rows_preserves_the_indices_shape_for_batched_lookup() {
        // A [batch=2, seq=2] block of token ids gathers into [2, 2, hidden].
        let table = Tensor::from_f32((0..6).map(|v| v as f32).collect(), [3, 2]).unwrap();
        let ids = Tensor::from_i32(vec![0, 1, 2, 0], [2, 2]).unwrap();
        let out = table.gather_rows(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[2, 2, 2]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn gather_rows_works_through_a_transposed_view_of_the_table() {
        // table stored as [hidden, vocab], transposed to [vocab, hidden]
        // before gathering — exercises the strides[]/offset-aware path.
        let table = Tensor::from_f32(vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0], [2, 4]).unwrap();
        let transposed = table.transpose(0, 1).unwrap(); // [4, 2]: rows [0,1],[2,3],[4,5],[6,7]
        let ids = Tensor::from_i32(vec![1], [1]).unwrap();
        let out = transposed.gather_rows(&ids).unwrap();
        assert_eq!(out.to_vec_f32().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn gather_rows_rejects_an_out_of_range_id() {
        let table = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let ids = Tensor::from_i32(vec![5], [1]).unwrap();
        assert!(matches!(table.gather_rows(&ids), Err(Error::IndexOutOfBounds { .. })));
    }

    #[test]
    fn gather_rows_requires_i32_indices() {
        let table = Tensor::from_f32(vec![1.0, 2.0], [1, 2]).unwrap();
        let bad_ids = Tensor::from_f32(vec![0.0], [1]).unwrap();
        assert!(matches!(table.gather_rows(&bad_ids), Err(Error::DTypeMismatch { .. })));
    }
}

#[cfg(test)]
mod quantized_gather_tests {
    use super::*;

    /// Builds a `[rows, cols]` Q8_0 table whose blocks have distinct scales and
    /// quants, so a row-addressing bug shifts values visibly instead of
    /// returning something plausible.
    fn q8_table(rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for r in 0..rows {
            for b in 0..(cols / 32) {
                // scale = 1.0 for even blocks, 0.5 for odd (f16 bit patterns).
                let scale: u16 = if (r + b) % 2 == 0 { 0x3C00 } else { 0x3800 };
                bytes.extend_from_slice(&scale.to_le_bytes());
                for t in 0..32 {
                    bytes.push((((r * 7 + b * 3 + t) % 255) as i32 - 128) as i8 as u8);
                }
            }
        }
        bytes
    }

    /// The property that makes this safe to switch on: gathering from the
    /// quantized table must equal gathering from the same table dequantized in
    /// full. If it does not, embeddings silently change and the model degrades
    /// in a way no shape check would catch.
    #[test]
    fn quantized_gather_matches_dequantize_then_gather() {
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_table(rows, cols);
        let quantized = Tensor::from_quantized(DType::Q8_0, bytes, [rows, cols]).unwrap();
        let full = quantized.to_dtype(DType::F32).unwrap();

        // Deliberately out of order, repeated, and including the last row.
        let ids = Tensor::from_i32(vec![3, 0, 7, 3, 5], [5]).unwrap();
        let from_quant = quantized.gather_rows(&ids).unwrap().to_vec_f32().unwrap();
        let from_f32 = full.gather_rows(&ids).unwrap().to_vec_f32().unwrap();
        assert_eq!(from_quant.len(), 5 * cols);
        assert_eq!(from_quant, from_f32, "quantized gather diverged from the f32 path");
    }

    #[test]
    fn quantized_gather_rejects_an_out_of_range_id() {
        let bytes = q8_table(4, 32);
        let t = Tensor::from_quantized(DType::Q8_0, bytes, [4, 32]).unwrap();
        let ids = Tensor::from_i32(vec![4], [1]).unwrap();
        assert!(matches!(
            t.gather_rows(&ids),
            Err(Error::IndexOutOfBounds { .. })
        ));
    }

    /// A row that is not a whole number of blocks cannot be decoded in
    /// isolation, so the gather must refuse rather than read across into the
    /// next row's data.
    #[test]
    fn a_row_that_straddles_a_block_boundary_is_refused() {
        // cols = 48 is not a multiple of Q8_0's 32-element block.
        let bytes = vec![0u8; 3 * 34];
        let t = Tensor::from_quantized(DType::Q8_0, bytes, [2, 48]);
        // Construction may already reject this; if it does not, the gather must.
        if let Ok(t) = t {
            let ids = Tensor::from_i32(vec![0], [1]).unwrap();
            assert!(t.gather_rows(&ids).is_err(), "a straddling row must not be gathered");
        }
    }
}
