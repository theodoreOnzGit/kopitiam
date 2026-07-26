//! Ported from Tesseract `src/lstm/fullyconnected.{cpp,h}` + the nonlinearity
//! function-objects in `src/lstm/functions.h` (commit db0ec62, Apache-2.0,
//! © 2014 Google Inc., Author: Ray Smith; part of the Tesseract project),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the dense
//! `weights · input + bias` step and the per-type nonlinearity selection
//! (`FullyConnected::Forward`/`ForwardTimeStep`, fullyconnected.cpp:129,203)
//! follow Tesseract; computed over the Kopitiam Runtime tensor ops
//! ([`Tensor::matmul`], [`Tensor::tanh`]/[`sigmoid`](Tensor::sigmoid)/
//! [`softmax`](Tensor::softmax)). See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`FullyConnected`] is the dense layer: one [`WeightMatrix`] applied to every
//! timestep, followed by a nonlinearity chosen by the layer's [`NetworkType`]
//! (tanh, logistic, relu, positive/symmetric clip, linear, or **softmax** — the
//! recognizer's output head, whose per-timestep distribution over the recoded
//! alphabet feeds the beam decoder).
//!
//! # Nonlinearity fidelity
//!
//! Tesseract's `Tanh`/`Logistic` use a 4096-entry lookup table with linear
//! interpolation (functions.h:44); this port uses the exact `f32`
//! `tanh`/`sigmoid` from `kopitiam-tensor` — a deliberate, more-accurate
//! deviation (max table error ≈ 2⁻¹², far below int8 weight quantization noise).
//! Its `SoftmaxInPlace` clips the shifted exponent to `[-86, 0]`; the exact
//! `Tensor::softmax` used here is numerically identical for any realistic logit
//! range.

use kopitiam_tensor::Tensor;

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode, NetworkType};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;
use crate::weightmatrix::WeightMatrix;

/// A dense feed-forward layer with a per-type nonlinearity. Tesseract:
/// `class FullyConnected` (fullyconnected.h).
#[derive(Debug)]
pub struct FullyConnected {
    header: NetworkHeader,
    weights: WeightMatrix,
}

impl FullyConnected {
    /// Deserializes a `FullyConnected`: just its [`WeightMatrix`]
    /// (`[no, ni+1]`). Tesseract: `FullyConnected::DeSerialize`
    /// (fullyconnected.cpp:123).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<FullyConnected> {
        let weights = WeightMatrix::deserialize(header.training, fp)?;
        Ok(FullyConnected { header, weights })
    }

    /// Builds a `FullyConnected` directly from a weight matrix — for tests and
    /// hand-built networks.
    pub fn from_parts(header: NetworkHeader, weights: WeightMatrix) -> FullyConnected {
        FullyConnected { header, weights }
    }

    /// The layer's weight matrix.
    pub fn weights(&self) -> &WeightMatrix {
        &self.weights
    }

    /// Applies this layer's nonlinearity to the `[width, no]` pre-activation
    /// tensor. Tesseract: `FullyConnected::ForwardTimeStep(output_line)`
    /// (fullyconnected.cpp:203), batched here over all timesteps.
    fn activate(&self, pre: Tensor) -> Result<Tensor> {
        use NetworkType::*;
        Ok(match self.header.ntype {
            Tanh => pre.tanh()?,
            Logistic => pre.sigmoid()?,
            Relu => map_f32(&pre, |x| if x > 0.0 { x } else { 0.0 })?,
            Posclip => map_f32(&pre, |x| x.clamp(0.0, 1.0))?,
            Symclip => map_f32(&pre, |x| x.clamp(-1.0, 1.0))?,
            Softmax | SoftmaxNoCtc => pre.softmax(1)?,
            Linear => pre,
            other => {
                return Err(Error::format(format!(
                    "FullyConnected: invalid nonlinearity type {other:?}"
                )));
            }
        })
    }
}

impl NetworkNode for FullyConnected {
    fn header(&self) -> &NetworkHeader {
        &self.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        let ni = self.header.ni as usize;
        let no = self.header.no as usize;
        let width = input.width();
        let mut output = NetworkIO::new(input.stride_map().clone(), no);
        if width == 0 {
            return Ok(output);
        }
        debug_assert_eq!(input.num_features(), ni, "FC input depth must equal ni");
        debug_assert_eq!(self.weights.num_inputs_with_bias(), ni + 1);

        // Pack the input as a [width, ni] matrix.
        let mut input_flat = Vec::with_capacity(width * ni);
        for t in 0..width {
            input_flat.extend_from_slice(input.f(t));
        }
        let inputs = Tensor::from_f32(input_flat, [width, ni])?;

        // weights are [no, ni+1]: split the ni weights from the trailing bias.
        let w = self.weights.weights_tensor()?;
        let w_main = w.narrow(1, 0, ni)?; // [no, ni]
        let bias = w.narrow(1, ni, 1)?.reshape([1, no])?; // [1, no]

        // pre = inputs @ w_mainᵀ + bias  -> [width, no].
        let pre = inputs.matmul(&w_main.transpose(0, 1)?)?;
        let pre = pre.add(&bias.broadcast_to([width, no])?)?;

        let out = self.activate(pre)?;
        let out_vec = out.to_vec_f32()?;
        for t in 0..width {
            output.write_time_step(t, &out_vec[t * no..(t + 1) * no]);
        }
        Ok(output)
    }
}

/// Elementwise `f32` map (for relu/clip, which `kopitiam-tensor` does not expose
/// as named ops), preserving shape.
fn map_f32(t: &Tensor, f: impl Fn(f32) -> f32) -> Result<Tensor> {
    let data: Vec<f32> = t.to_vec_f32()?.into_iter().map(f).collect();
    Ok(Tensor::from_f32(data, t.shape().dims().to_vec())?)
}
