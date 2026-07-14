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
    pub fn gather_rows(&self, indices: &Tensor) -> Result<Tensor> {
        self.require_dtype(DType::F32)?;
        indices.require_dtype(DType::I32)?;
        if self.rank() != 2 {
            return Err(Error::ShapeMismatch { expected: self.shape.clone(), actual: self.shape.clone() });
        }
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
