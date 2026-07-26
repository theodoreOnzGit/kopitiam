//! Ported from Tesseract `src/lstm/lstm.{cpp,h}` (commit db0ec62, Apache-2.0,
//! © 2013 Google Inc., Author: Ray Smith; part of the Tesseract project),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the gate
//! weight-set layout (`CI/GI/GF1/GO/GFS`), the `DeSerialize` framing
//! (lstm.cpp:253), and the 1-D forward recurrence (lstm.cpp:291) follow
//! Tesseract; the timestep recurrence is driven by the Kopitiam Runtime's
//! [`Tensor::lstm_cell`], which composes the same gate assignment
//! (`sigmoid` on the input/forget/output gates, `tanh` on the cell candidate,
//! `tanh` on the squashed state — lstm.cpp:386,442,455). See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Lstm`] is the recurrent core of the recognizer. Each timestep it forms a
//! padded input `[image_features | previous_output]`, projects it through four
//! gate weight matrices (candidate `CI`, input gate `GI`, forget gate `GF1`,
//! output gate `GO`), and runs one LSTM cell to produce the new state and the
//! emitted output. Two forward-facing `Lstm`s (one direct, one wrapped in
//! [`Reversed`](crate::reversed::Reversed)) inside a
//! [`Parallel`](crate::parallel::Parallel) make a bidirectional LSTM.
//!
//! # Scope: forward for 1-D `NT_LSTM`
//!
//! All four LSTM type tags **deserialize** (so any real recognizer graph loads).
//! Forward evaluation is implemented for the plain 1-D `NT_LSTM` — the workhorse
//! of every modern line recognizer. The 2-D (`GFS`) LSTM, the summary LSTM
//! (`NT_LSTM_SUMMARY`), and the softmax-feedback variants
//! (`NT_LSTM_SOFTMAX`/`..._ENCODED`) surface a clear "deferred" error on forward
//! while still loading fully.
//!
//! # State-clip deviation
//!
//! Tesseract clips the cell state to ±100 (`kStateClip`) *before* the output
//! squash. [`Tensor::lstm_cell`] omits that clamp (it is the plain mathematical
//! LSTM); this port applies the ±100 clamp to the carried cell after each step.
//! For realistic activations `|state| < 100` always holds, so a step's emitted
//! output is identical to Tesseract's.

use kopitiam_tensor::Tensor;

use crate::error::{Error, Result};
use crate::network::{NetworkHeader, NetworkNode, NetworkType, create_from_file};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;
use crate::weightmatrix::WeightMatrix;

// Tesseract: enum WeightType (lstm.h:32). Order is the serialization order.
const CI: usize = 0; // Cell inputs (candidate).
const GI: usize = 1; // Input gate.
const GF1: usize = 2; // 1-D forget gate.
const GO: usize = 3; // Output gate.
// GFS (2-D forget gate) is index 4, present only for 2-D LSTMs.

/// Tesseract's cell-state clamp, `kStateClip` (lstm.cpp:71).
const STATE_CLIP: f32 = 100.0;

/// A Long-Short-Term-Memory layer. Tesseract: `class LSTM` (lstm.h:27).
#[derive(Debug)]
pub struct Lstm {
    header: NetworkHeader,
    /// Padded input width to the gate matrices (`na_`): `ni + ns` (+ `ns` again
    /// for 2-D, + `nf` for softmax feedback). The gate matrices are `[ns, na+1]`.
    na: i32,
    /// Number of internal states (`ns_`), derived from the `CI` matrix.
    ns: i32,
    /// Additional feedback states for the softmax variants (`nf_`).
    nf: i32,
    /// True for a 2-D LSTM (has the extra `GFS` gate).
    is_2d: bool,
    /// The gate weight matrices, in `[CI, GI, GF1, GO]` order (`GFS` appended
    /// for 2-D).
    gates: Vec<WeightMatrix>,
    /// The built-in softmax head, for the softmax-LSTM variants.
    softmax: Option<Box<dyn NetworkNode>>,
}

impl Lstm {
    /// Deserializes an `Lstm`: `int32 na`, then the gate matrices (`CI`, `GI`,
    /// `GF1`, `GO`, and `GFS` iff 2-D), then the softmax head for the softmax
    /// variants. `ns`/`is_2d` are derived from the `CI` matrix, exactly as
    /// `LSTM::DeSerialize` does (lstm.cpp:253).
    pub fn deserialize(header: NetworkHeader, fp: &mut TFile<'_>) -> Result<Lstm> {
        let na = fp.read_i32()?;
        let nf = match header.ntype {
            NetworkType::LstmSoftmax => header.no,
            NetworkType::LstmSoftmaxEncoded => ceil_log2(header.no as u32) as i32,
            _ => 0,
        };
        let training = header.training;
        let mut is_2d = false;
        let mut ns = 0;
        let mut gates = Vec::with_capacity(5);
        for w in [CI, GI, GF1, GO, 4 /* GFS */] {
            if w == 4 && !is_2d {
                continue;
            }
            let wm = WeightMatrix::deserialize(training, fp)?;
            if w == CI {
                ns = wm.num_outputs() as i32;
                is_2d = na - nf == header.ni + 2 * ns;
            }
            gates.push(wm);
        }
        let softmax = match header.ntype {
            NetworkType::LstmSoftmax | NetworkType::LstmSoftmaxEncoded => {
                Some(create_from_file(fp)?)
            }
            _ => None,
        };
        Ok(Lstm {
            header,
            na,
            ns,
            nf,
            is_2d,
            gates,
            softmax,
        })
    }

    /// Builds a plain 1-D `NT_LSTM` from four gate matrices — for tests and
    /// hand-built networks. `na` must equal each gate matrix's `num_inputs`
    /// (`cols - 1`), and every gate must have the same `num_outputs` (`ns`).
    pub fn from_gates(header: NetworkHeader, na: i32, gates: Vec<WeightMatrix>) -> Lstm {
        let ns = gates[CI].num_outputs() as i32;
        Lstm {
            header,
            na,
            ns,
            nf: 0,
            is_2d: false,
            gates,
            softmax: None,
        }
    }

    /// Number of internal states (`ns_`).
    pub fn num_states(&self) -> i32 {
        self.ns
    }

    /// Number of additional softmax-feedback states (`nf_`); non-zero only for
    /// the `NT_LSTM_SOFTMAX`/`..._ENCODED` variants.
    pub fn num_feedback(&self) -> i32 {
        self.nf
    }

    /// Whether this is a 2-D LSTM.
    pub fn is_2d(&self) -> bool {
        self.is_2d
    }

    /// The built-in softmax head, if any.
    pub fn softmax(&self) -> Option<&dyn NetworkNode> {
        self.softmax.as_deref()
    }
}

impl NetworkNode for Lstm {
    fn header(&self) -> &NetworkHeader {
        &self.header
    }

    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        if self.is_2d
            || self.softmax.is_some()
            || self.header.ntype != NetworkType::Lstm
        {
            return Err(Error::format(
                "Lstm: forward is implemented for plain 1-D NT_LSTM only \
                 (2-D / summary / softmax-feedback variants deserialize but their \
                 forward is deferred to a later phase)",
            ));
        }
        let ns = self.ns as usize;
        let ni = self.header.ni as usize;
        let na = self.na as usize;
        let width = input.width();
        let mut output = NetworkIO::new(input.stride_map().clone(), self.header.no as usize);

        // Recurrent carry: cell state and previous emitted output.
        let mut state = vec![0.0f32; ns];
        let mut prev_out = vec![0.0f32; ns];
        // Padded gate input: [image features (ni) | previous output (ns)].
        let mut curr_input = vec![0.0f32; na];

        for t in 0..width {
            curr_input[..ni].copy_from_slice(input.f(t));
            // nf_ == 0 for plain LSTM, so previous output sits at [ni, ni+ns).
            curr_input[ni..ni + ns].copy_from_slice(&prev_out);

            // Gate pre-activations (before nonlinearity — lstm_cell applies them).
            let ci = self.gates[CI].matrix_dot_vector(&curr_input);
            let gi = self.gates[GI].matrix_dot_vector(&curr_input);
            let gf = self.gates[GF1].matrix_dot_vector(&curr_input);
            let go = self.gates[GO].matrix_dot_vector(&curr_input);

            let ci_t = Tensor::from_f32(ci, [ns])?;
            let gi_t = Tensor::from_f32(gi, [ns])?;
            let gf_t = Tensor::from_f32(gf, [ns])?;
            let go_t = Tensor::from_f32(go, [ns])?;
            let state_t = Tensor::from_f32(state.clone(), [ns])?;

            // input_gate=GI, forget_gate=GF1, cell_candidate=CI, output_gate=GO.
            let step = Tensor::lstm_cell(&gi_t, &gf_t, &ci_t, &go_t, &state_t)?;

            // Carry the clamped cell state (kStateClip); emit the hidden output.
            state = step.cell.to_vec_f32()?;
            for c in &mut state {
                *c = c.clamp(-STATE_CLIP, STATE_CLIP);
            }
            prev_out = step.hidden.to_vec_f32()?;
            output.write_time_step(t, &prev_out);
        }
        Ok(output)
    }
}

/// `ceil(log2(n))`, matching Tesseract's `ceil_log2` (lstm.cpp:76): `0` for
/// `n <= 1`, else the number of bits needed to index `n` values.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    let l2 = 31 - n.leading_zeros();
    if n == (1u32 << l2) { l2 } else { l2 + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches_reference() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
    }
}
