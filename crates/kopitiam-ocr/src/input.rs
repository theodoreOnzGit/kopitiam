//! Ported from Tesseract `src/lstm/input.{cpp,h}` + the `StaticShape`
//! (de)serialization from `src/lstm/static_shape.h` (commit db0ec62,
//! Apache-2.0, © 2014/2016 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! `Input::DeSerialize` reads a `StaticShape` (five `int32`), and
//! `Input::Forward` is the identity (input.cpp:45,64); re-expressed in Rust. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Input`] is the network's entry layer. In the forward pass it simply passes
//! the input activation buffer straight through (the image→buffer conversion it
//! also owns in Tesseract, `PreparePixInput`, is upstream of this phase). Its
//! only serialized payload is the [`StaticShape`] describing the expected input
//! tensor.

use crate::error::Result;
use crate::network::{NetworkHeader, NetworkNode};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;

/// The build-time 4-D tensor shape `[batch, height, width, depth]` plus the loss
/// type. Tesseract: `class StaticShape` (static_shape.h:38).
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticShape {
    /// Number of images in a batch (0 = runtime-variable).
    pub batch: i32,
    /// Image height (0 = runtime-variable).
    pub height: i32,
    /// Image width (0 = runtime-variable).
    pub width: i32,
    /// Feature depth (number of nodes).
    pub depth: i32,
    /// Loss/decoding type ordinal (`LossType`).
    pub loss_type: i32,
}

impl StaticShape {
    /// Reads a `StaticShape`: five `int32` (batch, height, width, depth,
    /// loss_type). Tesseract: `StaticShape::DeSerialize` (static_shape.h:83).
    pub fn deserialize(fp: &mut TFile<'_>) -> Result<StaticShape> {
        Ok(StaticShape {
            batch: fp.read_i32()?,
            height: fp.read_i32()?,
            width: fp.read_i32()?,
            depth: fp.read_i32()?,
            loss_type: fp.read_i32()?,
        })
    }
}

/// The network's input layer. Tesseract: `class Input` (input.h).
#[derive(Debug)]
pub struct Input {
    header: NetworkHeader,
    shape: StaticShape,
}

impl Input {
    /// Deserializes an `Input`: just its [`StaticShape`]. Tesseract:
    /// `Input::DeSerialize` (input.cpp:45).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Input> {
        let shape = StaticShape::deserialize(fp)?;
        Ok(Input { header, shape })
    }

    /// The declared input tensor shape.
    pub fn shape(&self) -> StaticShape {
        self.shape
    }
}

impl NetworkNode for Input {
    fn header(&self) -> &NetworkHeader {
        &self.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Identity: `*output = input`. Tesseract: Input::Forward (input.cpp:64).
        Ok(input.clone())
    }
}
