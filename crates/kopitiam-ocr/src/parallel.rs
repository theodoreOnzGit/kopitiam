//! Ported from Tesseract `src/lstm/parallel.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2013 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the fan-out forward — every sub-network runs on the *same* input and their
//! outputs are concatenated along the feature/depth axis — follows Tesseract's
//! `Parallel::Forward` (parallel.cpp:52); re-expressed in Rust. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Parallel`] runs several sub-networks on the same input and packs their
//! outputs together, producing an output whose depth is the sum of the
//! sub-networks' depths (and whose width matches — the sub-networks must all keep
//! the same timestep count). The canonical use is a bidirectional LSTM: a
//! forward LSTM in parallel with a [`Reversed`](crate::reversed::Reversed)
//! backward LSTM (`NT_PAR_RL_LSTM`). All parallel flavors (`NT_PARALLEL`,
//! `NT_REPLICATED`, `NT_PAR_*_LSTM`) share this concatenating forward.

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode};
use crate::networkio::NetworkIO;
use crate::plumbing::Plumbing;
use crate::serialis::TFile;

/// Several sub-networks run on the same input, outputs concatenated by depth.
/// Tesseract: `class Parallel` (parallel.h).
#[derive(Debug)]
pub struct Parallel {
    inner: Plumbing,
}

impl Parallel {
    /// Deserializes a `Parallel`: its stack, then recomputes `ni`/`no` (all
    /// sub-networks share `ni`; `no` is the sum of their outputs), matching
    /// `Plumbing::AddToStack` for the parallel case (plumbing.cpp:92).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Parallel> {
        let mut inner = Plumbing::deserialize(header, fp)?;
        if let Some(first) = inner.stack.first() {
            inner.header.ni = first.num_inputs();
            inner.header.no = inner.stack.iter().map(|n| n.num_outputs()).sum();
        }
        Ok(Parallel { inner })
    }

    /// The sub-networks, in order.
    pub fn stack(&self) -> &[Box<dyn NetworkNode>] {
        &self.inner.stack
    }
}

impl NetworkNode for Parallel {
    fn header(&self) -> &NetworkHeader {
        &self.inner.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        // Run each sub-network on the same input; pack outputs along depth.
        // Tesseract: Parallel::Forward (parallel.cpp:52), non-2D branch.
        let total_features = self.inner.header.no as usize;
        let mut result: Option<NetworkIO> = None;
        let mut offset = 0;
        for layer in &self.inner.stack {
            let part = layer.forward(input)?;
            let out = result.get_or_insert_with(|| {
                NetworkIO::new(part.stride_map().clone(), total_features)
            });
            if part.width() != out.width() {
                return Err(Error::format(format!(
                    "Parallel: sub-network width {} disagrees with {}",
                    part.width(),
                    out.width()
                )));
            }
            offset = out.copy_packing(&part, offset);
        }
        result.ok_or_else(|| Error::format("Parallel: empty stack"))
    }
}
