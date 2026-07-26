//! Ported from Tesseract `src/lstm/reconfig.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! `Reconfig::DeSerialize`/`Forward` (reconfig.cpp:60,87) — the x/y downscale
//! that trades time/height resolution for feature depth — follow Tesseract,
//! re-expressed over the ported [`StrideMap`](crate::stridemap). See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Reconfig`] shrinks the width by `x_scale` and height by `y_scale`, stacking
//! each `x_scale × y_scale` block of input cells into a single deeper output cell
//! (`depth × x_scale × y_scale`). Its forward is a pure regrouping — no weights.

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;
use crate::stridemap::{FD_BATCH, FD_HEIGHT, FD_WIDTH};

/// A time/height downscaler that deepens the feature axis. Tesseract:
/// `class Reconfig` (reconfig.h).
#[derive(Debug)]
pub struct Reconfig {
    header: NetworkHeader,
    x_scale: i32,
    y_scale: i32,
}

impl Reconfig {
    /// Deserializes a `Reconfig`: `int32 x_scale`, `int32 y_scale`, then
    /// recomputes `no = ni * x_scale * y_scale`. Tesseract:
    /// `Reconfig::DeSerialize` (reconfig.cpp:60).
    pub fn deserialize(mut header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Reconfig> {
        let x_scale = fp.read_i32()?;
        let y_scale = fp.read_i32()?;
        if x_scale <= 0 || y_scale <= 0 || header.ni <= 0 {
            return Err(Error::format(format!(
                "Reconfig: invalid parameters ni={} x_scale={x_scale} y_scale={y_scale}",
                header.ni
            )));
        }
        let no = (header.ni as i64) * (x_scale as i64) * (y_scale as i64);
        if no > i32::MAX as i64 {
            return Err(Error::format("Reconfig: output-channel count overflows"));
        }
        header.no = no as i32;
        Ok(Reconfig {
            header,
            x_scale,
            y_scale,
        })
    }

    /// The `(x_scale, y_scale)` downscale factors. Used by
    /// [`Maxpool`](crate::maxpool::Maxpool), which shares this payload.
    pub(crate) fn scales(&self) -> (i32, i32) {
        (self.x_scale, self.y_scale)
    }

    pub(crate) fn forward_impl(
        header: &NetworkHeader,
        x_scale: i32,
        y_scale: i32,
        input: &NetworkIO,
    ) -> Result<NetworkIO> {
        let ni = header.ni as usize;
        let no = header.no as usize;
        let mut out_stride = input.stride_map().clone();
        out_stride.scale_xy(x_scale, y_scale);
        let mut output = NetworkIO::new(out_stride, no);

        // Own a copy of the output stride to iterate while mutating `output`.
        let out_stride_owned = output.stride_map().clone();
        let mut dest_index = out_stride_owned.index();
        loop {
            let out_t = dest_index.t() as usize;
            let src_index = input.stride_map().index_at(
                dest_index.index(FD_BATCH),
                dest_index.index(FD_HEIGHT) * y_scale,
                dest_index.index(FD_WIDTH) * x_scale,
            );
            for x in 0..x_scale {
                for y in 0..y_scale {
                    let mut src_xy = src_index.clone();
                    if src_xy.add_offset(x, FD_WIDTH) && src_xy.add_offset(y, FD_HEIGHT) {
                        output.copy_time_step_general(
                            out_t,
                            ((x * y_scale + y) as usize) * ni,
                            ni,
                            input,
                            src_xy.t() as usize,
                            0,
                        );
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

impl NetworkNode for Reconfig {
    fn header(&self) -> &NetworkHeader {
        &self.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Tesseract: Reconfig::Forward (reconfig.cpp:87).
        Reconfig::forward_impl(&self.header, self.x_scale, self.y_scale, input)
    }
}
