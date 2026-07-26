//! One LSTM timestep: [`Tensor::lstm_cell`].
//!
//! This is the recurrence a Tesseract-style LSTM line recognizer runs at
//! every timestep of a text-line's width. The four gate *pre-activations*
//! (the summed input + recurrent linear contributions) are computed by the
//! recognizer with matmuls — [`Tensor::matmul`] or the int8 path in
//! `tessdata.rs` — and this helper turns them into the next cell state and
//! hidden output. Keeping the recurrence here (rather than open-coding it in
//! the recognizer) means the gate math lives next to the `sigmoid`/`tanh`
//! activations it is built from, and is unit-tested against hand arithmetic
//! independently of any model.
//!
//! The equations are the standard LSTM (Hochreiter & Schmidhuber, 1997;
//! Gers, Schmidhuber & Cummins, 2000) — clean-room standard math, not a port:
//!
//! ```text
//! i = sigmoid(i_pre)          (input gate)
//! f = sigmoid(f_pre)          (forget gate)
//! o = sigmoid(o_pre)          (output gate)
//! g = tanh(g_pre)             (cell candidate)
//! c' = f * c_prev + i * g     (new cell state)
//! h  = o * tanh(c')           (new hidden output)
//! ```
//!
//! This is exactly the gate assignment Tesseract's `lstm.cpp` uses (its `CI`
//! candidate takes `GFunc = Tanh`, its `GI`/`GF1`/`GO` gates take
//! `FFunc = Logistic`, then `curr_state = f*state + i*g` and
//! `curr_output = Tanh(state) * o`), but the recurrence itself is textbook —
//! nothing format- or Tesseract-specific is translated here. Tesseract
//! additionally clips the cell state to +/-100 before the output squash; that
//! clamp is a numerical-stability guard, deliberately omitted from this pure
//! recurrence so the operation stays the plain mathematical LSTM. If a real
//! model needs it, apply it to the returned [`LstmState::cell`].

use kopitiam_core::{Error, Result};

use super::Tensor;

/// The two things one LSTM timestep produces: the new cell state to carry to
/// the next timestep, and the hidden output emitted at this one.
///
/// A named pair rather than a bare `(Tensor, Tensor)` so a recognizer driving
/// a sequence cannot silently swap the carried state and the emitted output —
/// the two have the same shape, so the compiler could not catch a mix-up.
#[derive(Debug, Clone)]
pub struct LstmState {
    /// `c'`, the cell state carried into the next timestep as `cell_prev`.
    pub cell: Tensor,
    /// `h`, the hidden output this timestep emits (to a softmax/head, and as
    /// the recurrent input feeding the next timestep's gate pre-activations).
    pub hidden: Tensor,
}

impl Tensor {
    /// Computes one LSTM timestep from the four gate pre-activations and the
    /// previous cell state, returning the new [`LstmState`].
    ///
    /// Each argument is the *pre-activation* of one gate — the summed input
    /// and recurrent linear contributions, before any nonlinearity — so this
    /// helper composes cleanly after the recognizer's gate matmuls. The
    /// activations (`sigmoid` on the three gates, `tanh` on the candidate and
    /// on the squashed state) are applied here.
    ///
    /// All five tensors must be `f32` and share one shape (the hidden size,
    /// optionally with leading batch/time dims); the result carries that same
    /// shape. Broadcasting is intentionally *not* offered: the gates and the
    /// state of a single LSTM layer are always the same width, so a shape
    /// disagreement is a bug to surface, not a broadcast to paper over.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] if any operand is not `f32` (propagated from
    /// the underlying [`Tensor::sigmoid`]/[`Tensor::tanh`]/[`Tensor::mul`]).
    /// [`Error::ShapeMismatch`] if the operands' shapes are not all identical.
    pub fn lstm_cell(
        input_gate: &Tensor,
        forget_gate: &Tensor,
        cell_candidate: &Tensor,
        output_gate: &Tensor,
        cell_prev: &Tensor,
    ) -> Result<LstmState> {
        let shape = input_gate.shape();
        for other in [forget_gate, cell_candidate, output_gate, cell_prev] {
            if other.shape() != shape {
                return Err(Error::ShapeMismatch {
                    expected: shape.clone(),
                    actual: other.shape().clone(),
                });
            }
        }

        let i = input_gate.sigmoid()?;
        let f = forget_gate.sigmoid()?;
        let o = output_gate.sigmoid()?;
        let g = cell_candidate.tanh()?;

        // c' = f * c_prev + i * g
        let cell = f.mul(cell_prev)?.add(&i.mul(&g)?)?;
        // h = o * tanh(c')
        let hidden = o.mul(&cell.tanh()?)?;

        Ok(LstmState { cell, hidden })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    #[test]
    fn lstm_cell_matches_hand_computation_for_a_single_unit() {
        // One unit, chosen pre-activations, previous cell state c = 0.5.
        // i = sigmoid(1.0), f = sigmoid(0.5), o = sigmoid(-0.5), g = tanh(2.0)
        let i_pre = Tensor::from_f32(vec![1.0], [1]).unwrap();
        let f_pre = Tensor::from_f32(vec![0.5], [1]).unwrap();
        let g_pre = Tensor::from_f32(vec![2.0], [1]).unwrap();
        let o_pre = Tensor::from_f32(vec![-0.5], [1]).unwrap();
        let c_prev = Tensor::from_f32(vec![0.5], [1]).unwrap();

        let out = Tensor::lstm_cell(&i_pre, &f_pre, &g_pre, &o_pre, &c_prev).unwrap();

        let i = 1.0f32 / (1.0 + (-1.0f32).exp());
        let f = 1.0f32 / (1.0 + (-0.5f32).exp());
        let o = 1.0f32 / (1.0 + 0.5f32.exp());
        let g = 2.0f32.tanh();
        let c = f * 0.5 + i * g;
        let h = o * c.tanh();

        assert_close(out.cell.to_vec_f32().unwrap()[0], c);
        assert_close(out.hidden.to_vec_f32().unwrap()[0], h);
    }

    #[test]
    fn lstm_cell_with_zero_input_and_zero_state_stays_at_zero_cell() {
        // All pre-activations zero, c_prev = 0: g = tanh(0) = 0 and c_prev = 0,
        // so c' = f*0 + i*0 = 0 exactly. h = o * tanh(0) = 0 too.
        let z = Tensor::zeros([4]);
        let out = Tensor::lstm_cell(&z, &z, &z, &z, &z).unwrap();
        assert_eq!(out.cell.to_vec_f32().unwrap(), vec![0.0; 4]);
        assert_eq!(out.hidden.to_vec_f32().unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn forget_open_and_input_shut_preserves_the_cell_state_exactly() {
        // f -> 1 (large positive forget pre-activation) and i -> 0 (large
        // negative input pre-activation) is the LSTM's "carry the memory
        // unchanged" regime: c' should equal c_prev to within the gates'
        // saturation error.
        let big = 30.0f32; // sigmoid(30) ~= 1, sigmoid(-30) ~= 0.
        let f_pre = Tensor::from_f32(vec![big, big, big], [3]).unwrap();
        let i_pre = Tensor::from_f32(vec![-big, -big, -big], [3]).unwrap();
        let g_pre = Tensor::from_f32(vec![5.0, -5.0, 1.0], [3]).unwrap(); // irrelevant: i ~= 0.
        let o_pre = Tensor::from_f32(vec![0.0, 0.0, 0.0], [3]).unwrap();
        let c_prev = Tensor::from_f32(vec![0.3, -0.7, 1.2], [3]).unwrap();

        let out = Tensor::lstm_cell(&i_pre, &f_pre, &g_pre, &o_pre, &c_prev).unwrap();
        let cell = out.cell.to_vec_f32().unwrap();
        for (got, want) in cell.iter().zip([0.3, -0.7, 1.2]) {
            assert!((got - want).abs() < 1e-6, "cell state not preserved: got {got}, want {want}");
        }
    }

    #[test]
    fn lstm_cell_preserves_a_multi_dimensional_shape() {
        let a = Tensor::from_f32(vec![0.1, 0.2, 0.3, 0.4], [2, 2]).unwrap();
        let out = Tensor::lstm_cell(&a, &a, &a, &a, &a).unwrap();
        assert_eq!(out.cell.shape().dims(), &[2, 2]);
        assert_eq!(out.hidden.shape().dims(), &[2, 2]);
    }

    #[test]
    fn lstm_cell_rejects_a_shape_disagreement() {
        let a = Tensor::from_f32(vec![0.0; 3], [3]).unwrap();
        let b = Tensor::from_f32(vec![0.0; 4], [4]).unwrap();
        assert!(matches!(
            Tensor::lstm_cell(&a, &a, &a, &a, &b),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn lstm_cell_rejects_non_f32_input() {
        let f32t = Tensor::from_f32(vec![0.0; 3], [3]).unwrap();
        let i32t = Tensor::from_i32(vec![0; 3], [3]).unwrap();
        assert!(matches!(
            Tensor::lstm_cell(&i32t, &f32t, &f32t, &f32t, &f32t),
            Err(Error::DTypeMismatch { .. })
        ));
    }
}
