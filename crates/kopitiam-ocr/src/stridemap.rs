//! Ported from Tesseract `src/lstm/stridemap.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2016 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the packed 4-D→2-D indexing arithmetic (`t_increments_`, carry/borrow in
//! `Increment`/`Decrement`) follows Tesseract exactly; re-expressed in idiomatic
//! Rust. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! A [`NetworkIO`](crate::networkio::NetworkIO) holds a 4-D tensor
//! `[batch, height, width, depth]` flattened so that `batch*height*width` is the
//! first ("time", `t`) index of an underlying 2-D `[t, depth]` array. [`StrideMap`]
//! is the index arithmetic that maps a `(batch, y, x)` coordinate to its flat
//! `t`, and iterates over all valid `t` in raster order — the piece the
//! convolutional / pooling / reconfig / reversed plumbing nodes use to walk the
//! grid. For a 1-D text line (`batch = height = 1`) `t` is simply the x-position.

// Tesseract: enum FlexDimensions (stridemap.h:32)
/// Index of multiple images in a batch.
pub const FD_BATCH: usize = 0;
/// y-coordinate in the image.
pub const FD_HEIGHT: usize = 1;
/// x-coordinate in the image.
pub const FD_WIDTH: usize = 2;
/// Number of flexible (non-depth) dimensions.
pub const FD_DIMSIZE: usize = 3;

/// Encapsulation of the mapping from `[batch][y][x]` to the first index into the
/// 2-D array underlying a `NetworkIO`.
///
/// Tesseract: `class StrideMap` (stridemap.h:41).
#[derive(Debug, Clone, Default)]
pub struct StrideMap {
    /// The size of each non-depth dimension (`shape_`).
    shape: [i32; FD_DIMSIZE],
    /// Precomputed `t` increments for each dimension (`t_increments_`).
    t_increments: [i32; FD_DIMSIZE],
    /// Per-batch image heights (`heights_`).
    heights: Vec<i32>,
    /// Per-batch image widths (`widths_`).
    widths: Vec<i32>,
}

impl StrideMap {
    /// A stride map for a single `height × width` image (batch of one) — the
    /// common text-line case. Convenience not present verbatim in Tesseract
    /// (which always builds from `SetStride`), but equivalent to
    /// `SetStride({{height, width}})`.
    pub fn single(height: i32, width: i32) -> Self {
        let mut map = StrideMap::default();
        map.set_stride(&[(height, width)]);
        map
    }

    /// Sets up the stride for the given `(height, width)` pairs, one per batch
    /// element. Tesseract: `StrideMap::SetStride` (stridemap.cpp:131).
    pub fn set_stride(&mut self, h_w_pairs: &[(i32, i32)]) {
        let mut max_height = 0;
        let mut max_width = 0;
        self.heights.clear();
        self.widths.clear();
        for &(height, width) in h_w_pairs {
            self.heights.push(height);
            self.widths.push(width);
            max_height = max_height.max(height);
            max_width = max_width.max(width);
        }
        self.shape[FD_BATCH] = self.heights.len() as i32;
        self.shape[FD_HEIGHT] = max_height;
        self.shape[FD_WIDTH] = max_width;
        self.compute_t_increments();
    }

    /// Scales width and height dimensions by the given factors (integer
    /// division). Tesseract: `StrideMap::ScaleXY` (stridemap.cpp:153).
    pub fn scale_xy(&mut self, x_factor: i32, y_factor: i32) {
        for h in &mut self.heights {
            *h /= y_factor;
        }
        for w in &mut self.widths {
            *w /= x_factor;
        }
        self.shape[FD_HEIGHT] /= y_factor;
        self.shape[FD_WIDTH] /= x_factor;
        self.compute_t_increments();
    }

    /// Reduces the width to 1 across the batch. Tesseract:
    /// `StrideMap::ReduceWidthTo1` (stridemap.cpp:166).
    pub fn reduce_width_to_1(&mut self) {
        for w in &mut self.widths {
            *w = 1;
        }
        self.shape[FD_WIDTH] = 1;
        self.compute_t_increments();
    }

    /// Transposes the width and height dimensions. Tesseract:
    /// `StrideMap::TransposeXY` (stridemap.cpp:173).
    pub fn transpose_xy(&mut self) {
        self.shape.swap(FD_HEIGHT, FD_WIDTH);
        std::mem::swap(&mut self.heights, &mut self.widths);
        self.compute_t_increments();
    }

    /// The size of the given flexible dimension. Tesseract: `StrideMap::Size`.
    pub fn size(&self, dimension: usize) -> i32 {
        self.shape[dimension]
    }

    /// The total flattened width (`t` range). Tesseract: `StrideMap::Width`.
    pub fn width(&self) -> i32 {
        self.t_increments[FD_BATCH] * self.shape[FD_BATCH]
    }

    // Tesseract: StrideMap::ComputeTIncrements (stridemap.cpp:180)
    fn compute_t_increments(&mut self) {
        self.t_increments[FD_DIMSIZE - 1] = 1;
        for d in (0..FD_DIMSIZE - 1).rev() {
            self.t_increments[d] = self.t_increments[d + 1] * self.shape[d + 1];
        }
    }

    /// A fresh index positioned at the first valid location.
    pub fn index(&self) -> Index<'_> {
        Index::new(self)
    }

    /// An index at an explicit `(batch, y, x)` coordinate.
    pub fn index_at(&self, batch: i32, y: i32, x: i32) -> Index<'_> {
        let mut idx = Index {
            stride_map: self,
            t: 0,
            indices: [batch, y, x],
        };
        idx.set_t_from_indices();
        idx
    }
}

/// A concrete `(batch, y, x)` position within a [`StrideMap`], carrying the
/// derived flat index `t`.
///
/// Tesseract: `class StrideMap::Index` (stridemap.h:44).
#[derive(Clone)]
pub struct Index<'a> {
    stride_map: &'a StrideMap,
    t: i32,
    indices: [i32; FD_DIMSIZE],
}

impl<'a> Index<'a> {
    fn new(stride_map: &'a StrideMap) -> Self {
        Index {
            stride_map,
            t: 0,
            indices: [0; FD_DIMSIZE],
        }
    }

    /// The flat index into the underlying 2-D array.
    pub fn t(&self) -> i32 {
        self.t
    }

    /// The coordinate along the given flexible dimension.
    pub fn index(&self, dimension: usize) -> i32 {
        self.indices[dimension]
    }

    /// Positions this index at the last valid location. Tesseract:
    /// `Index::InitToLast` (stridemap.h:68).
    pub fn init_to_last(&mut self) {
        self.init_to_last_of_batch(self.max_index_of_dim(FD_BATCH));
    }

    /// True if `*this` is a valid index. Tesseract: `Index::IsValid`.
    pub fn is_valid(&self) -> bool {
        if self.indices.iter().any(|&i| i < 0) {
            return false;
        }
        (0..FD_DIMSIZE).all(|d| self.indices[d] <= self.max_index_of_dim(d))
    }

    /// True if the given dimension is at its last index. Tesseract:
    /// `Index::IsLast`.
    pub fn is_last(&self, dimension: usize) -> bool {
        self.max_index_of_dim(dimension) == self.indices[dimension]
    }

    /// Given that dimensions up to `dim-1` are valid, the maximum index for
    /// `dim`. Tesseract: `Index::MaxIndexOfDim` (stridemap.cpp:46).
    pub fn max_index_of_dim(&self, dim: usize) -> i32 {
        let max_index = self.stride_map.shape[dim] - 1;
        if dim == FD_BATCH {
            return max_index;
        }
        let batch = self.indices[FD_BATCH] as usize;
        if dim == FD_HEIGHT {
            if batch >= self.stride_map.heights.len() || self.stride_map.heights[batch] > max_index {
                return max_index;
            }
            return self.stride_map.heights[batch] - 1;
        }
        if batch >= self.stride_map.widths.len() || self.stride_map.widths[batch] > max_index {
            return max_index;
        }
        self.stride_map.widths[batch] - 1
    }

    /// Adds `offset` to `dimension`; returns true if the result is a valid
    /// index. Tesseract: `Index::AddOffset` (stridemap.cpp:67).
    pub fn add_offset(&mut self, offset: i32, dimension: usize) -> bool {
        self.indices[dimension] += offset;
        self.set_t_from_indices();
        self.is_valid()
    }

    /// Advances to the next valid location in raster order; returns false when
    /// iteration is complete. Tesseract: `Index::Increment` (stridemap.cpp:75).
    pub fn increment(&mut self) -> bool {
        for d in (0..FD_DIMSIZE).rev() {
            if !self.is_last(d) {
                self.t += self.stride_map.t_increments[d];
                self.indices[d] += 1;
                return true;
            }
            self.t -= self.stride_map.t_increments[d] * self.indices[d];
            self.indices[d] = 0;
            // Carry to the next dimension.
        }
        false
    }

    /// Steps back to the previous valid location (paired with
    /// [`Index::init_to_last`]); returns false when complete. Tesseract:
    /// `Index::Decrement` (stridemap.cpp:92).
    pub fn decrement(&mut self) -> bool {
        for d in (0..FD_DIMSIZE).rev() {
            if self.indices[d] > 0 {
                self.indices[d] -= 1;
                if d == FD_BATCH {
                    // The other dimensions' upper limits may have changed.
                    self.init_to_last_of_batch(self.indices[FD_BATCH]);
                } else {
                    self.t -= self.stride_map.t_increments[d];
                }
                return true;
            }
            self.indices[d] = self.max_index_of_dim(d);
            self.t += self.stride_map.t_increments[d] * self.indices[d];
            // Borrow from the next dimension.
        }
        false
    }

    // Tesseract: Index::InitToLastOfBatch (stridemap.cpp:114)
    fn init_to_last_of_batch(&mut self, batch: i32) {
        self.indices[FD_BATCH] = batch;
        for d in (FD_BATCH + 1)..FD_DIMSIZE {
            self.indices[d] = self.max_index_of_dim(d);
        }
        self.set_t_from_indices();
    }

    // Tesseract: Index::SetTFromIndices (stridemap.cpp:123)
    fn set_t_from_indices(&mut self) {
        self.t = 0;
        for d in 0..FD_DIMSIZE {
            self.t += self.stride_map.t_increments[d] * self.indices[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_iterates_left_to_right() {
        let map = StrideMap::single(1, 5);
        assert_eq!(map.width(), 5);
        let mut idx = map.index();
        let mut seen = Vec::new();
        loop {
            seen.push(idx.t());
            if !idx.increment() {
                break;
            }
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn two_d_grid_rasterizes_in_row_major_t_order() {
        // 2 rows x 3 cols. t = y*3 + x.
        let map = StrideMap::single(2, 3);
        assert_eq!(map.width(), 6);
        let mut idx = map.index();
        let mut coords = Vec::new();
        loop {
            coords.push((idx.index(FD_HEIGHT), idx.index(FD_WIDTH), idx.t()));
            if !idx.increment() {
                break;
            }
        }
        assert_eq!(
            coords,
            vec![
                (0, 0, 0),
                (0, 1, 1),
                (0, 2, 2),
                (1, 0, 3),
                (1, 1, 4),
                (1, 2, 5),
            ]
        );
    }

    #[test]
    fn add_offset_detects_out_of_bounds() {
        let map = StrideMap::single(1, 3);
        let mut idx = map.index();
        assert!(idx.add_offset(2, FD_WIDTH)); // x=2 valid
        assert_eq!(idx.t(), 2);
        assert!(!idx.add_offset(1, FD_WIDTH)); // x=3 invalid
    }

    #[test]
    fn decrement_walks_back_from_last() {
        let map = StrideMap::single(1, 4);
        let mut idx = map.index();
        idx.init_to_last();
        assert_eq!(idx.t(), 3);
        let mut seen = vec![idx.t()];
        while idx.decrement() {
            seen.push(idx.t());
        }
        assert_eq!(seen, vec![3, 2, 1, 0]);
    }

    #[test]
    fn scale_and_reduce_change_the_width() {
        let mut map = StrideMap::single(4, 8);
        map.scale_xy(2, 2);
        assert_eq!(map.size(FD_WIDTH), 4);
        assert_eq!(map.size(FD_HEIGHT), 2);
        map.reduce_width_to_1();
        assert_eq!(map.size(FD_WIDTH), 1);
        assert_eq!(map.width(), 2);
    }
}
