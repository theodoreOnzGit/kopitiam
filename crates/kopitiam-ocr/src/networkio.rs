//! Ported from Tesseract `src/lstm/networkio.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the `[timesteps × features]` activation-buffer layout, the `Resize*` family,
//! the packing/reversal helpers, and their use of [`StrideMap`] follow Tesseract;
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`NetworkIO`] is the activation buffer that flows between network layers: a
//! 2-D `[t, num_features]` array of `f32`, where `t` ranges over the flattened
//! `batch × height × width` grid described by an owned [`StrideMap`]. Tesseract's
//! `NetworkIO` can also hold `int8_t` activations (`int_mode_`) as an inference
//! speed optimization for its fused-int8 matmul path; this port keeps activations
//! **float only** — the recognizer dequantizes int8 *weights* to `f32` up front
//! (see [`crate::weightmatrix`]) and computes every layer in `f32`, so an int
//! activation buffer is never needed. `int_mode()` is therefore always `false`.
//! The float path is the reference path Tesseract itself validates against.
//!
//! Only the read/forward-eval subset is ported: the training-time helpers
//! (`ZeroInvalidElements` beyond padding, `Randomize`, `ToPix`, the backward
//! delta ops) are omitted; each is a training/graphics concern outside this
//! forward-recognition phase.

use crate::stridemap::{FD_WIDTH, StrideMap};

/// The activation buffer passed between network layers: `[t, num_features]`
/// `f32` laid out row-major over the [`StrideMap`]'s flattened `t` axis.
///
/// Tesseract: `class NetworkIO` (networkio.h:37), float-mode subset.
#[derive(Debug, Clone)]
pub struct NetworkIO {
    stride_map: StrideMap,
    num_features: usize,
    /// Row-major `[width * num_features]`; row `t` is
    /// `data[t*num_features .. (t+1)*num_features]`.
    data: Vec<f32>,
}

impl NetworkIO {
    /// A zeroed buffer shaped to the given stride map and feature depth.
    /// Tesseract: `NetworkIO::ResizeToMap` (networkio.cpp), float branch.
    pub fn new(stride_map: StrideMap, num_features: usize) -> Self {
        let width = stride_map.width().max(0) as usize;
        NetworkIO {
            stride_map,
            num_features,
            data: vec![0.0; width * num_features],
        }
    }

    /// A single-line (`batch = height = 1`) buffer of `width` timesteps built
    /// from row-major `data` of length `width * num_features`. Convenience for
    /// constructing network input and for tests.
    pub fn from_rows(width: usize, num_features: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * num_features, "row-major length mismatch");
        NetworkIO {
            stride_map: StrideMap::single(1, width as i32),
            num_features,
            data,
        }
    }

    /// A buffer matching `src`'s stride map but with a new feature depth
    /// (zeroed). Tesseract: `NetworkIO::Resize` / `ResizeFloat` (networkio.h:44).
    pub fn resize_like(src: &NetworkIO, num_features: usize) -> Self {
        NetworkIO::new(src.stride_map.clone(), num_features)
    }

    /// The flattened timestep count (`t` range). Tesseract: `NetworkIO::Width`.
    pub fn width(&self) -> usize {
        self.stride_map.width().max(0) as usize
    }

    /// The feature depth. Tesseract: `NetworkIO::NumFeatures`.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Always `false` in this float-only port (see the module docs).
    /// Tesseract: `NetworkIO::int_mode`.
    pub fn int_mode(&self) -> bool {
        false
    }

    /// The owned stride map. Tesseract: `NetworkIO::stride_map`.
    pub fn stride_map(&self) -> &StrideMap {
        &self.stride_map
    }

    /// Read access to timestep `t`'s feature row. Tesseract: `NetworkIO::f(t)`.
    pub fn f(&self, t: usize) -> &[f32] {
        &self.data[t * self.num_features..(t + 1) * self.num_features]
    }

    /// Mutable access to timestep `t`'s feature row.
    pub fn f_mut(&mut self, t: usize) -> &mut [f32] {
        &mut self.data[t * self.num_features..(t + 1) * self.num_features]
    }

    /// Copies `input` into timestep `t`. Tesseract: `NetworkIO::WriteTimeStep`.
    pub fn write_time_step(&mut self, t: usize, input: &[f32]) {
        self.f_mut(t).copy_from_slice(input);
    }

    /// Copies `input` into `[offset, offset+num)` of timestep `t`.
    /// Tesseract: `NetworkIO::WriteTimeStepPart` (float branch).
    pub fn write_time_step_part(&mut self, t: usize, offset: usize, num: usize, input: &[f32]) {
        let base = t * self.num_features + offset;
        self.data[base..base + num].copy_from_slice(&input[..num]);
    }

    /// Copies a `[dest_offset, dest_offset+num)` slice of timestep `dest_t` from
    /// `[src_offset, src_offset+num)` of `src`'s timestep `src_t`. Tesseract:
    /// `NetworkIO::CopyTimeStepGeneral` (float branch).
    pub fn copy_time_step_general(
        &mut self,
        dest_t: usize,
        dest_offset: usize,
        num: usize,
        src: &NetworkIO,
        src_t: usize,
        src_offset: usize,
    ) {
        let dbase = dest_t * self.num_features + dest_offset;
        let sbase = src_t * src.num_features + src_offset;
        self.data[dbase..dbase + num].copy_from_slice(&src.data[sbase..sbase + num]);
    }

    /// Copies a whole timestep from `src` to `dest_t`. Tesseract:
    /// `NetworkIO::CopyTimeStepFrom`.
    pub fn copy_time_step_from(&mut self, dest_t: usize, src: &NetworkIO, src_t: usize) {
        let n = self.num_features;
        self.copy_time_step_general(dest_t, 0, n, src, src_t, 0);
    }

    /// Packs `src`'s features into `[feature_offset, ..)` of every timestep of
    /// `self`, returning `feature_offset + src.num_features` (the next free
    /// offset). This is how [`Parallel`](crate::parallel::Parallel) concatenates
    /// its sub-networks' outputs along the depth axis. Tesseract:
    /// `NetworkIO::CopyPacking` (networkio.cpp, float branch).
    pub fn copy_packing(&mut self, src: &NetworkIO, feature_offset: usize) -> usize {
        let width = src.width().min(self.width());
        for t in 0..width {
            let dbase = t * self.num_features + feature_offset;
            let sbase = t * src.num_features;
            self.data[dbase..dbase + src.num_features]
                .copy_from_slice(&src.data[sbase..sbase + src.num_features]);
        }
        feature_offset + src.num_features
    }

    /// Fills `self` with `src` reversed along the width (x/time) axis, per
    /// height row and batch. Drives the backward-direction LSTM of a BiLSTM
    /// (see [`Reversed`](crate::reversed::Reversed)). Tesseract:
    /// `NetworkIO::CopyWithXReversal` (networkio.cpp).
    pub fn copy_with_x_reversal(&mut self, src: &NetworkIO) {
        // Iterate batch × height, reversing each row's width run.
        let mut b_index = src.stride_map.index();
        loop {
            let mut y_index = b_index.clone();
            loop {
                let mut fwd = y_index.clone();
                let mut rev = y_index.clone();
                rev.add_offset(rev.max_index_of_dim(FD_WIDTH), FD_WIDTH);
                loop {
                    self.copy_time_step_from(rev.t() as usize, src, fwd.t() as usize);
                    if !(fwd.add_offset(1, FD_WIDTH) && rev.add_offset(-1, FD_WIDTH)) {
                        break;
                    }
                }
                if !y_index.add_offset(1, crate::stridemap::FD_HEIGHT) {
                    break;
                }
            }
            if !b_index.add_offset(1, crate::stridemap::FD_BATCH) {
                break;
            }
        }
    }

    /// Flattens the buffer to a row-major `[width][num_features]` matrix of
    /// `f32`. The recognizer's per-timestep softmax logits, for the beam
    /// decoder (phase 5).
    pub fn to_rows(&self) -> Vec<Vec<f32>> {
        (0..self.width()).map(|t| self.f(t).to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rows_indexes_timesteps() {
        let io = NetworkIO::from_rows(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(io.width(), 3);
        assert_eq!(io.num_features(), 2);
        assert_eq!(io.f(0), &[1.0, 2.0]);
        assert_eq!(io.f(2), &[5.0, 6.0]);
    }

    #[test]
    fn copy_packing_concatenates_features() {
        let a = NetworkIO::from_rows(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = NetworkIO::from_rows(2, 3, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let mut out = NetworkIO::new(a.stride_map().clone(), 5);
        let off = out.copy_packing(&a, 0);
        assert_eq!(off, 2);
        let off = out.copy_packing(&b, off);
        assert_eq!(off, 5);
        assert_eq!(out.f(0), &[1.0, 2.0, 10.0, 20.0, 30.0]);
        assert_eq!(out.f(1), &[3.0, 4.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn x_reversal_flips_the_time_axis() {
        let src = NetworkIO::from_rows(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut dst = NetworkIO::resize_like(&src, 2);
        dst.copy_with_x_reversal(&src);
        assert_eq!(dst.f(0), &[5.0, 6.0]);
        assert_eq!(dst.f(1), &[3.0, 4.0]);
        assert_eq!(dst.f(2), &[1.0, 2.0]);
    }
}
