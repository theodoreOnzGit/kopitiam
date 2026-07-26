//! Decoding Tesseract's int8 LSTM weight matrices to `f32`:
//! [`Tensor::tessdata_int8_to_f32`].
//!
//! # Provenance (translation, not clean-room)
//!
//! Unlike the activations, the LSTM cell, and the conv/pool ops in this crate
//! — which are standard math written from scratch — the int8 weight layout
//! this module decodes is a **specific on-disk format** defined by Tesseract,
//! so this is a *translation* of that format, credited here at the point of
//! use per the project's attribution rules (`docs/ACKNOWLEDGEMENTS.md`,
//! `docs/ai-decisions/AID-0051`). The layout follows
//! `src/lstm/weightmatrix.cpp` — `WeightMatrix::ConvertToInt` (the encoder)
//! and `Debug2D`/`MatrixDotVector` (the `wi * scale` recovery) — from
//! Tesseract at commit `db0ec62`, Apache-2.0, (C) Google Inc., Author: Ray
//! Smith. Apache-2.0 is one-way compatible with KOPITIAM's AGPL-3.0-only.
//! No Tesseract code is copied; the byte-layout algorithm is reimplemented.
//!
//! # The layout
//!
//! A Tesseract int8 weight matrix is `[num_outputs, num_inputs]` (row-major,
//! one row per output unit). Every *row* is quantized independently:
//!
//! * `ConvertToInt` finds `max_abs`, the largest absolute weight in the row,
//!   sets `scale = max_abs / INT8_MAX` (`INT8_MAX == 127`), and stores each
//!   weight as `IntCastRounded(w / scale)` — round-half-away-from-zero, which
//!   is exactly Rust's `f32::round` — clamped to the int8 range.
//! * The per-row `scale` is what recovers the weight: `w ~= i8 * scale`.
//!   (`Debug2D` reconstructs weights as `wi_[i][j] * scales_[i]`; the fused
//!   integer `MatrixDotVector` multiplies each row's integer dot product by
//!   that same per-row scale.)
//!
//! This function takes that recovery scale directly — one `scale` per row,
//! the value for which `w ~= i8 * scale` — and returns the dequantized
//! `[rows, cols]` `f32` [`Tensor`], which the recognizer can then feed to the
//! ordinary [`Tensor::matmul`]. Recovering to `f32` (rather than a fused int8
//! matmul) is the "correct before fast" choice this crate makes everywhere
//! else too (see [`Tensor::quantized_matmul`]'s docs for the same reasoning);
//! a fused int8 path can be added later against a real model if the memory
//! saving proves to matter for tessdata's (comparatively small) LSTM weights.
//!
//! # Uncertainty flagged for the recognizer phase
//!
//! The int8 -> f32 recovery here (`w = i8 * scale`, one scale per row, with
//! `ConvertToInt`'s exact rounding/clamping validated by the round-trip test
//! below) is the unambiguous part. What a caller wiring this to a *real*
//! tessdata file must still pin down — out of scope for this tensor-op phase
//! because it needs a real model to check against — is precisely **which
//! scale value the deserializer hands us**. Tesseract keeps two different
//! scalings of the same number: the on-disk value is `max_abs / 127` (its
//! `Serialize` multiplies the in-memory scale back up by `INT8_MAX` before
//! writing), while the in-memory `scales_` carries an extra `/ 127` factor
//! (`max_abs / 127^2`) so its *fused* int8 path can multiply an int8-quantized
//! *activation* dot product without re-dividing. This function wants the
//! **on-disk** `max_abs / 127` scale — the one that reconstructs a weight on
//! its own. Two further tessdata details are likewise deferred to that phase,
//! as they are serializer/SIMD concerns, not part of the per-row decode:
//! (1) `MatrixDotVector` treats each weight row as having a trailing **bias**
//! column (an extra input pinned to 1.0), so the recognizer must decide
//! whether a given matrix's last column is a bias; and (2) the SIMD path pads
//! `num_outputs` up to a rounded count (`IntSimdMatrix::Init`) and can leave
//! extra scale entries — the caller passes the true, unpadded `rows`/`cols`.

use kopitiam_core::{Error, Result, Shape};

use super::Tensor;

impl Tensor {
    /// Decodes a Tesseract int8 LSTM weight matrix to an `f32` [`Tensor`].
    ///
    /// `weights` is the row-major `[rows, cols]` int8 matrix (`rows`
    /// = `num_outputs`, `cols` = `num_inputs`), and `scales` is the per-row
    /// recovery scale — one entry per row, the value for which
    /// `w ~= weights[i][j] * scales[i]` (Tesseract's on-disk `max_abs / 127`;
    /// see the module docs for the two-scaling subtlety). Returns the
    /// dequantized `[rows, cols]` tensor `out[i][j] = weights[i*cols + j] as
    /// f32 * scales[i]`, ready for [`Tensor::matmul`].
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if `weights.len() != rows * cols` or
    /// `scales.len() != rows`.
    pub fn tessdata_int8_to_f32(
        weights: &[i8],
        scales: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        if weights.len() != rows * cols {
            return Err(Error::ShapeMismatch {
                expected: Shape::new([rows, cols]),
                actual: Shape::new([weights.len()]),
            });
        }
        if scales.len() != rows {
            return Err(Error::ShapeMismatch {
                expected: Shape::new([rows]),
                actual: Shape::new([scales.len()]),
            });
        }

        let mut out = vec![0f32; rows * cols];
        for i in 0..rows {
            let scale = scales[i];
            for j in 0..cols {
                out[i * cols + j] = weights[i * cols + j] as f32 * scale;
            }
        }
        Tensor::from_f32(out, Shape::new([rows, cols]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The int8 range endpoint; `INT8_MAX` in `weightmatrix.cpp`.
    const INT8_MAX: f32 = 127.0;

    /// Encodes one weight row exactly as `WeightMatrix::ConvertToInt` does,
    /// returning `(int8 weights, on-disk scale)`. Test-only, mirroring the
    /// reference encoder so the round-trip can be checked against it — this
    /// crate has no public `f32 -> int8` weight encoder (weight quantization
    /// is a model-export concern, same stance as the GGUF side).
    fn encode_row(row: &[f32]) -> (Vec<i8>, f32) {
        let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = max_abs / INT8_MAX; // the on-disk recovery scale.
        // ConvertToInt substitutes 1.0 for a zero scale to avoid /0; the row is
        // then all zeros anyway.
        let div = if scale == 0.0 { 1.0 } else { scale };
        let q: Vec<i8> = row
            .iter()
            // IntCastRounded == round-half-away-from-zero == f32::round.
            .map(|&w| (w / div).round().clamp(-INT8_MAX, INT8_MAX) as i8)
            .collect();
        (q, scale)
    }

    #[test]
    fn decode_round_trips_a_known_row_within_quantization_error() {
        let row = vec![0.10, -0.40, 0.25, 0.80, -0.80, 0.05, 0.55, -0.30];
        let (q, scale) = encode_row(&row);

        // max_abs = 0.80 -> scale = 0.80/127. The extreme values (+/-0.80) are
        // representable exactly (they map to +/-127); every other weight is
        // within half a step, i.e. scale/2, of its original.
        let decoded = Tensor::tessdata_int8_to_f32(&q, &[scale], 1, row.len())
            .unwrap()
            .to_vec_f32()
            .unwrap();

        let half_step = scale / 2.0;
        for (got, want) in decoded.iter().zip(&row) {
            assert!(
                (got - want).abs() <= half_step + 1e-7,
                "decoded {got} too far from {want} (half-step {half_step})"
            );
        }
        // The two saturating extremes recover exactly.
        assert!((decoded[3] - 0.80).abs() < 1e-6);
        assert!((decoded[4] + 0.80).abs() < 1e-6);
    }

    #[test]
    fn decode_recovers_an_exact_multiple_row_bit_for_bit() {
        // A row whose values are exact integer multiples of a chosen scale:
        // no rounding, so decode reproduces it exactly. scale = 0.5, values
        // = {-2,-1,0,1,2} * 0.5 -> int8 {-127? no}. Pick max_abs = 1.0 so
        // scale = 1/127 and values are k/127 for integer k in [-127,127].
        let scale = 1.0 / INT8_MAX;
        let q: Vec<i8> = vec![-127, -64, 0, 33, 127];
        let expected: Vec<f32> = q.iter().map(|&k| k as f32 * scale).collect();
        let decoded = Tensor::tessdata_int8_to_f32(&q, &[scale], 1, q.len())
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn decode_uses_an_independent_scale_per_row() {
        // Two rows, different scales; each row's int8 values recover against
        // its own scale.
        let weights: Vec<i8> = vec![10, -20, 30, 1, -2, 3];
        let scales = [0.5f32, 2.0f32];
        let decoded = Tensor::tessdata_int8_to_f32(&weights, &scales, 2, 3)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(decoded, vec![5.0, -10.0, 15.0, 2.0, -4.0, 6.0]);
    }

    #[test]
    fn decoded_weight_matmul_matches_a_known_small_matrix() {
        // Decode a small [2,3] int8 weight matrix, then matmul it against a
        // known [3,2] input the way the recognizer would (W . x).
        //   scale row0 = 0.5, row1 = 1.0
        //   W = [[5, -10, 15], [2, -4, 6]]  (after decode)
        let weights: Vec<i8> = vec![10, -20, 30, 2, -4, 6];
        let scales = [0.5f32, 1.0f32];
        let w = Tensor::tessdata_int8_to_f32(&weights, &scales, 2, 3).unwrap();

        // x is [3, 2]; result W @ x is [2, 2].
        // row0 . col0 = 5*1 + (-10)*0 + 15*1 = 20 ; row0 . col1 = 5*2 + (-10)*1 + 15*0 = 0
        // row1 . col0 = 2*1 + (-4)*0 + 6*1 = 8    ; row1 . col1 = 2*2 + (-4)*1 + 6*0 = 0
        let x = Tensor::from_f32(vec![1.0, 2.0, 0.0, 1.0, 1.0, 0.0], [3, 2]).unwrap();
        let out = w.matmul(&x).unwrap();
        assert_eq!(out.shape().dims(), &[2, 2]);
        assert_eq!(out.to_vec_f32().unwrap(), vec![20.0, 0.0, 8.0, 0.0]);
    }

    #[test]
    fn decode_rejects_a_weight_length_mismatch() {
        let weights = vec![1i8, 2, 3];
        assert!(matches!(
            Tensor::tessdata_int8_to_f32(&weights, &[1.0, 1.0], 2, 3),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn decode_rejects_a_scale_count_mismatch() {
        let weights = vec![1i8, 2, 3, 4, 5, 6];
        assert!(matches!(
            Tensor::tessdata_int8_to_f32(&weights, &[1.0], 2, 3),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn decode_handles_an_all_zero_row() {
        // encode_row gives scale 0 for an all-zero row; decoding 0 * anything
        // is 0 regardless, so the row round-trips exactly.
        let row = vec![0.0f32; 4];
        let (q, scale) = encode_row(&row);
        assert_eq!(scale, 0.0);
        let decoded = Tensor::tessdata_int8_to_f32(&q, &[scale], 1, 4)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(decoded, vec![0.0; 4]);
    }
}
