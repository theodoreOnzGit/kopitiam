//! Ported from Tesseract `src/lstm/plumbing.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the sub-network stack, the `DeSerialize` framing (`uint32` count + that many
//! recursively-deserialized layers + optional per-layer learning rates), and the
//! `AddToStack` dimension bookkeeping follow Tesseract; re-expressed in Rust. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Plumbing`] is the shared base of the composite layers — the ones that hold
//! *other* layers rather than weights: [`Series`](crate::series::Series) (chain),
//! [`Parallel`](crate::parallel::Parallel) (fan/concat), and
//! [`Reversed`](crate::reversed::Reversed) (single sub-net on reversed input).
//! It owns the `stack` of sub-networks and the deserialization of that stack.
//! The layer-specific forward pass lives in each wrapper.

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode, create_from_file};
use crate::serialis::TFile;

/// Bit 6 (`NF_LAYER_SPECIFIC_LR`) of a layer's network flags: a per-layer
/// learning-rate vector trails the stack in serialization. A training concern;
/// this port reads past it. Tesseract: `enum NetworkFlags` (network.h:82).
const NF_LAYER_SPECIFIC_LR: i32 = 64;

/// A collection of sub-networks and the shared header. Tesseract:
/// `class Plumbing` (plumbing.h:28), read/forward subset.
#[derive(Debug)]
pub struct Plumbing {
    /// The composite's own header (its `ni`/`no` are recomputed by the wrapping
    /// layer from the stack, matching `AddToStack`).
    pub header: NetworkHeader,
    /// The sub-networks, in serialization order.
    pub stack: Vec<Box<dyn NetworkNode>>,
}

impl Plumbing {
    /// Deserializes the sub-network stack (and skips any trailing per-layer
    /// learning rates). Tesseract: `Plumbing::DeSerialize` (plumbing.cpp:215).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Plumbing> {
        let size = fp.read_u32()?;
        // Reject unreasonably large network stacks (as Tesseract does).
        if size > 10_000 {
            return Err(Error::limit(format!(
                "Plumbing: stack size {size} exceeds 10000 guard"
            )));
        }
        let mut stack = Vec::with_capacity(size as usize);
        for _ in 0..size {
            stack.push(create_from_file(fp)?);
        }
        if (header.network_flags & NF_LAYER_SPECIFIC_LR) != 0 {
            // learning_rates_: a vector<float> (uint32 count + count f32). A
            // training-time value, discarded — skip its bytes.
            let count = fp.read_u32()?;
            if !fp.skip(count as usize * 4) {
                return Err(Error::unexpected_eof(
                    "Plumbing: truncated per-layer learning rates",
                ));
            }
        }
        Ok(Plumbing { header, stack })
    }
}
