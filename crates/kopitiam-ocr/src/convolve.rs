//! Ported from Tesseract `src/lstm/convolve.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! `Convolve::DeSerialize`/`Forward` (convolve.cpp:45,77) — a sliding
//! `(2·half_x+1) × (2·half_y+1)` window that stacks its neighborhood into a
//! deeper output of the *same* width — follow Tesseract, re-expressed over the
//! ported [`StrideMap`](crate::stridemap). See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Convolve`] copies each input cell's spatial neighborhood into a single,
//! deeper output cell (`depth × kernel_x × kernel_y`), leaving the width/height
//! unchanged. It has no weights — the learned convolution is the
//! [`FullyConnected`](crate::fullyconnected::FullyConnected) layer that follows
//! it in the series. Out-of-image window positions are filled with random noise
//! in Tesseract's *training* forward; at recognition there is no randomizer, so
//! this port zero-fills them (documented deviation, immaterial to the border
//! cells of a scaled text line).

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;
use crate::stridemap::{FD_HEIGHT, FD_WIDTH};

/// A neighborhood-stacking convolutional layer. Tesseract: `class Convolve`
/// (convolve.h).
#[derive(Debug)]
pub struct Convolve {
    header: NetworkHeader,
    half_x: i32,
    half_y: i32,
}

impl Convolve {
    /// Deserializes a `Convolve`: `int32 half_x`, `int32 half_y`, then
    /// recomputes `no = ni · (2·half_x+1) · (2·half_y+1)`. Tesseract:
    /// `Convolve::DeSerialize` (convolve.cpp:45).
    pub fn deserialize(mut header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Convolve> {
        let half_x = fp.read_i32()?;
        let half_y = fp.read_i32()?;
        if half_x < 0 || half_y < 0 || header.ni <= 0 {
            return Err(Error::format(format!(
                "Convolve: invalid parameters ni={} half_x={half_x} half_y={half_y}",
                header.ni
            )));
        }
        let kx = 2i64 * half_x as i64 + 1;
        let ky = 2i64 * half_y as i64 + 1;
        let no = header.ni as i64 * kx * ky;
        if no > i32::MAX as i64 {
            return Err(Error::format("Convolve: output-channel count overflows"));
        }
        header.no = no as i32;
        Ok(Convolve {
            header,
            half_x,
            half_y,
        })
    }
}

impl NetworkNode for Convolve {
    fn header(&self) -> &NetworkHeader {
        &self.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Tesseract: Convolve::Forward (convolve.cpp:77). Same width, deeper.
        let ni = self.header.ni as usize;
        let no = self.header.no as usize;
        let y_scale = 2 * self.half_y + 1;
        let mut output = NetworkIO::new(input.stride_map().clone(), no);

        let out_stride_owned = output.stride_map().clone();
        let mut dest_index = out_stride_owned.index();
        loop {
            let t = dest_index.t() as usize;
            let mut out_ix = 0usize;
            for x in -self.half_x..=self.half_x {
                let mut x_index = dest_index.clone();
                if !x_index.add_offset(x, FD_WIDTH) {
                    // This x is outside the image: zero-fill the y_scale*ni block.
                    zero_fill(&mut output, t, out_ix, (y_scale as usize) * ni);
                } else {
                    let mut out_iy = out_ix;
                    for y in -self.half_y..=self.half_y {
                        let mut y_index = x_index.clone();
                        if !y_index.add_offset(y, FD_HEIGHT) {
                            zero_fill(&mut output, t, out_iy, ni);
                        } else {
                            output.copy_time_step_general(
                                t,
                                out_iy,
                                ni,
                                input,
                                y_index.t() as usize,
                                0,
                            );
                        }
                        out_iy += ni;
                    }
                }
                out_ix += (y_scale as usize) * ni;
            }
            if !dest_index.increment() {
                break;
            }
        }
        Ok(output)
    }
}

/// Zeroes `[offset, offset+num)` of timestep `t` (out-of-image window fill).
fn zero_fill(output: &mut NetworkIO, t: usize, offset: usize, num: usize) {
    let row = output.f_mut(t);
    for v in &mut row[offset..offset + num] {
        *v = 0.0;
    }
}
