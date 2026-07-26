//! Ported from Tesseract `src/lstm/series.{cpp,h}` (commit db0ec62, Apache-2.0,
//! © 2013 Google Inc., Author: Ray Smith; part of the Tesseract project),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the
//! chain-forward (output of layer *n* is the input of layer *n+1*) follows
//! Tesseract's `Series::Forward` (series.cpp:107); re-expressed in Rust. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Series`] runs its sub-networks in sequence on the same data stream: the
//! first layer consumes the network input, each subsequent layer consumes the
//! previous layer's output, and the last layer's output is the series' output.
//! The top level of a recognizer is a `Series` (e.g. Input → Convolve → Maxpool
//! → LSTMs → Softmax).

use crate::error::Result;
use crate::network::{NetworkHeader, NetworkNode, NetworkType};
use crate::networkio::NetworkIO;
use crate::plumbing::Plumbing;
use crate::serialis::TFile;

/// A sequence of layers executed one after another. Tesseract: `class Series`
/// (series.h), a `Plumbing` with `type_ == NT_SERIES`.
#[derive(Debug)]
pub struct Series {
    inner: Plumbing,
}

impl Series {
    /// Deserializes a `Series`: its stack, then recomputes `ni`/`no` from the
    /// stack (input of the first layer, output of the last), matching
    /// `Plumbing::AddToStack` (plumbing.cpp:84).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Series> {
        let mut inner = Plumbing::deserialize(header, fp)?;
        if let (Some(first), Some(last)) = (inner.stack.first(), inner.stack.last()) {
            inner.header.ni = first.num_inputs();
            inner.header.no = last.num_outputs();
        }
        Ok(Series { inner })
    }

    /// The sub-networks, in order.
    pub fn stack(&self) -> &[Box<dyn NetworkNode>] {
        &self.inner.stack
    }
}

impl NetworkNode for Series {
    fn header(&self) -> &NetworkHeader {
        &self.inner.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Chain: run the first layer on `input`, then feed each output forward.
        // Tesseract: Series::Forward (series.cpp:107).
        let mut current = self.inner.stack[0].forward(input)?;
        for layer in &self.inner.stack[1..] {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn network_type(&self) -> NetworkType {
        NetworkType::Series
    }
}
