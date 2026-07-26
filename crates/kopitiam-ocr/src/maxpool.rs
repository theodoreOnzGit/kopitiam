//! Ported from Tesseract `src/lstm/maxpool.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! `Maxpool::DeSerialize`/`Forward` (maxpool.cpp:29,37) — a max over each
//! `x_scale × y_scale` window, keeping the depth unchanged — follow Tesseract,
//! re-expressed over the ported [`StrideMap`](crate::stridemap). See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Maxpool`] is a [`Reconfig`](crate::reconfig::Reconfig) subtype whose output
//! depth equals its input depth (`no = ni`): instead of stacking the window's
//! cells it takes their per-feature maximum. It shares Reconfig's serialized
//! payload (`x_scale`, `y_scale`).

use crate::error::Result;
use crate::network::{NetworkHeader, NetworkNode};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;
use crate::stridemap::{FD_BATCH, FD_HEIGHT, FD_WIDTH};

/// Standard max-pooling layer. Tesseract: `class Maxpool` (maxpool.h).
#[derive(Debug)]
pub struct Maxpool {
    header: NetworkHeader,
    x_scale: i32,
    y_scale: i32,
}

impl Maxpool {
    /// Deserializes a `Maxpool`: the same `x_scale`/`y_scale` payload as
    /// [`Reconfig`](crate::reconfig::Reconfig), but with `no = ni`. Tesseract:
    /// `Maxpool::DeSerialize` (maxpool.cpp:29).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Maxpool> {
        let reconfig = crate::reconfig::Reconfig::deserialize(header, fp)?;
        let (x_scale, y_scale) = reconfig.scales();
        let mut header = reconfig.header().clone();
        // Maxpool keeps depth: no_ = ni_ (maxpool.cpp:31).
        header.no = header.ni;
        Ok(Maxpool {
            header,
            x_scale,
            y_scale,
        })
    }
}

impl NetworkNode for Maxpool {
    fn header(&self) -> &NetworkHeader {
        &self.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Tesseract: Maxpool::Forward (maxpool.cpp:37).
        let ni = self.header.ni as usize;
        let mut out_stride = input.stride_map().clone();
        out_stride.scale_xy(self.x_scale, self.y_scale);
        let mut output = NetworkIO::new(out_stride, ni);

        let out_stride_owned = output.stride_map().clone();
        let mut dest_index = out_stride_owned.index();
        loop {
            let out_t = dest_index.t() as usize;
            let src_index = input.stride_map().index_at(
                dest_index.index(FD_BATCH),
                dest_index.index(FD_HEIGHT) * self.y_scale,
                dest_index.index(FD_WIDTH) * self.x_scale,
            );
            // Seed the output row with the window's first cell.
            output.copy_time_step_from(out_t, input, src_index.t() as usize);
            for x in 0..self.x_scale {
                for y in 0..self.y_scale {
                    let mut src_xy = src_index.clone();
                    if src_xy.add_offset(x, FD_WIDTH) && src_xy.add_offset(y, FD_HEIGHT) {
                        let src_row = input.f(src_xy.t() as usize).to_vec();
                        let dst = output.f_mut(out_t);
                        for i in 0..ni {
                            if src_row[i] > dst[i] {
                                dst[i] = src_row[i];
                            }
                        }
                    }
                }
            }
            if !dest_index.increment() {
                break;
            }
        }
        Ok(output)
    }
}
