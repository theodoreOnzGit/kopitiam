//! Ported from Tesseract `src/lstm/reversed.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2013 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! `Reversed::Forward` (reversed.cpp:52) — reverse the input, run the single
//! sub-network, reverse the output back — follows Tesseract; re-expressed in
//! Rust. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Reversed`] wraps a single sub-network so it sees time-reversed input, then
//! un-reverses its output. This is what turns a left-to-right LSTM into the
//! right-to-left half of a bidirectional LSTM (`NT_XREVERSED`). `NT_YREVERSED`
//! and `NT_XYTRANSPOSE` reverse/transpose the y axis instead; their *forward* is
//! deferred here (they need a true 2-D input, absent on a reduced text line),
//! while all three deserialize normally.

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode, NetworkType};
use crate::networkio::NetworkIO;
use crate::plumbing::Plumbing;
use crate::serialis::TFile;

/// A single sub-network run on reversed input. Tesseract: `class Reversed`
/// (reversed.h), a `Plumbing` holding exactly one layer.
#[derive(Debug)]
pub struct Reversed {
    inner: Plumbing,
}

impl Reversed {
    /// Deserializes a `Reversed`: its (single-layer) stack, inheriting `ni`/`no`
    /// from that layer. Tesseract: `Plumbing::AddToStack` for the first element.
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Reversed> {
        let mut inner = Plumbing::deserialize(header, fp)?;
        if let Some(sub) = inner.stack.first() {
            inner.header.ni = sub.num_inputs();
            inner.header.no = sub.num_outputs();
        }
        Ok(Reversed { inner })
    }

    /// The wrapped sub-network.
    pub fn sub(&self) -> &dyn NetworkNode {
        self.inner.stack[0].as_ref()
    }
}

impl NetworkNode for Reversed {
    fn header(&self) -> &NetworkHeader {
        &self.inner.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Tesseract: Reversed::Forward + ReverseData (reversed.cpp:52,78).
        match self.inner.header.ntype {
            NetworkType::XReversed => {
                let mut rev_input = NetworkIO::resize_like(input, input.num_features());
                rev_input.copy_with_x_reversal(input);
                let rev_output = self.inner.stack[0].forward(&rev_input)?;
                let mut output = NetworkIO::resize_like(&rev_output, rev_output.num_features());
                output.copy_with_x_reversal(&rev_output);
                Ok(output)
            }
            NetworkType::YReversed | NetworkType::XyTranspose => Err(Error::format(
                "Reversed: y-reversal / xy-transpose forward is deferred \
                 (2-D input path not implemented in this phase)",
            )),
            other => Err(Error::format(format!(
                "Reversed: unexpected wrapped type {other:?}"
            ))),
        }
    }
}
