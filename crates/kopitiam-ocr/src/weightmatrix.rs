//! Ported from Tesseract `src/lstm/weightmatrix.{cpp,h}` (commit db0ec62,
//! Apache-2.0, © 2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project), translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the serialized layout (`mode` flag byte, the `GENERIC_2D_ARRAY` size/empty/
//! data framing from `matrix.h`, the per-row int8 scale vector) and the
//! `MatrixDotVector` trailing-bias convention follow Tesseract exactly; the
//! dequantization + dot product are re-expressed in Rust over the Kopitiam
//! Runtime tensor ops. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`WeightMatrix`] is the perf-critical primitive under every dense layer: a
//! `[num_outputs, num_inputs+1]` weight matrix (the trailing `+1` column is the
//! bias) stored either as `f32` or as row-quantized `int8` + a per-row scale.
//! This port loads it from a [`TFile`] and computes `v = W·u` with the implied
//! bias, dequantizing int8 weights to `f32` up front via
//! [`kopitiam_tensor::Tensor::tessdata_int8_to_f32`] (correct-before-fast; a
//! fused int8 path can come later).
//!
//! # Serialized layout (`WeightMatrix::DeSerialize`, weightmatrix.cpp:280)
//!
//! * `uint8 mode` — bit 0 (`kInt8Flag`) int8 vs float; bit 2 (`kAdamFlag`)
//!   Adam; bit 7 (`kDoubleFlag`) "new" double format. Recognition models always
//!   set `kDoubleFlag`; the pre-double `DeSerializeOld` path is **not** ported
//!   (it is a backward-compat concern for models a decade old).
//! * **int8 branch:** a `GENERIC_2D_ARRAY<int8_t>` (`int32 dim1`, `int32 dim2`,
//!   `int8 empty`, then `dim1*dim2` `int8`), then `uint32 n` and `n` `double`
//!   scales.
//! * **float branch:** a `GENERIC_2D_ARRAY<double>` (`int32 dim1`, `int32 dim2`,
//!   `double empty`, then `dim1*dim2` `double`). If the layer was serialized in
//!   training mode, `updates` (and, iff Adam, `dw_sq_sum`) double matrices
//!   follow; this port reads and discards them.
//!
//! # The int8 scale factor — pinned (phase-3 flag resolved)
//!
//! Tesseract keeps the same number at two scalings. `ConvertToInt` stores the
//! in-memory `scales_` as `max_abs / 127²`, but `Serialize` writes `scale *
//! INT8_MAX == max_abs / 127` to disc (weightmatrix.cpp:257). The **on-disk**
//! value `max_abs / 127` is exactly the per-row recovery scale for which
//! `w ≈ i8 * scale`, which is what [`Tensor::tessdata_int8_to_f32`] wants. So
//! this loader keeps the on-disk double **verbatim** — it does *not* divide by
//! 127 (the C `DeSerialize` divides only because its fused int8 `MatrixDotVector`
//! re-applies an `INT8_MAX` factor to the int8-quantized *activations*, which the
//! float path here never does). Confirmed against the round-trip in
//! `tessdata.rs`. The SIMD path can pad the on-disk scale count above
//! `num_outputs` (`IntSimdMatrix::Init` → `scales_.resize(rounded_num_out)`); the
//! extra entries are ignored and the true `num_outputs` rows are used.
//!
//! # The trailing-bias column — pinned
//!
//! `MatrixDotVectorInternal` (weightmatrix.cpp:99) computes, for output row `i`,
//! `v[i] = Σ_j w[i][j]·u[j] + w[i][extent]` where `extent = dim2 - 1`. I.e. the
//! **last column of every row is the bias**, and the input vector `u` is one
//! shorter than the row (an implicit `u[extent] = 1`). This is honored below.

use kopitiam_tensor::Tensor;

use crate::error::{Error, Result};
use crate::serialis::TFile;

/// Bit 0 of `mode`: the matrix stores `int8` weights + per-row scales.
const K_INT8_FLAG: u8 = 1;
/// Bit 2 of `mode`: the matrix was trained with Adam (adds `dw_sq_sum` in
/// training serialization).
const K_ADAM_FLAG: u8 = 4;
/// Bit 7 of `mode`: the "new" (post-2018) format where scales/weights are
/// doubles. Absent ⇒ legacy `DeSerializeOld`, unsupported here.
const K_DOUBLE_FLAG: u8 = 128;
/// `matrix.h` rejects a `GENERIC_2D_ARRAY` dimension above `UINT16_MAX`.
const MAX_DIM: i32 = u16::MAX as i32;

/// A dense network weight matrix `[num_outputs, num_inputs+1]`, dequantized to
/// `f32`. The final column of each row is the bias.
///
/// Tesseract: `class WeightMatrix` (weightmatrix.h:70), read/forward subset.
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    /// `true` if loaded from the int8 branch (informational; the stored weights
    /// are always dequantized `f32`).
    int_mode: bool,
    /// `num_outputs` (matrix `dim1`).
    rows: usize,
    /// `num_inputs + 1` including the bias column (matrix `dim2`).
    cols: usize,
    /// Row-major `[rows*cols]` dequantized weights; `dense[i*cols + j]`.
    dense: Vec<f32>,
}

impl WeightMatrix {
    /// Reads a weight matrix from `fp`. `is_training` selects whether the float
    /// branch expects the trailing `updates`/`dw_sq_sum` matrices (a recognition
    /// model is not in training mode, so it does not).
    ///
    /// Tesseract: `WeightMatrix::DeSerialize` (weightmatrix.cpp:280).
    pub fn deserialize(is_training: bool, fp: &mut TFile<'_>) -> Result<Self> {
        let mode = fp.read_u8()?;
        let int_mode = (mode & K_INT8_FLAG) != 0;
        let use_adam = (mode & K_ADAM_FLAG) != 0;
        if (mode & K_DOUBLE_FLAG) == 0 {
            return Err(Error::format(
                "WeightMatrix: legacy (pre-double) format unsupported; \
                 only kDoubleFlag models are read in this phase",
            ));
        }
        if int_mode {
            let (wi, rows, cols) = read_i8_matrix(fp)?;
            let size = fp.read_u32()?;
            if size > 100_000_000 {
                return Err(Error::limit(format!(
                    "WeightMatrix: scale count {size} exceeds guard"
                )));
            }
            let raw_scales = fp.read_f64_vec(size as usize)?;
            if (size as usize) < rows {
                return Err(Error::format(format!(
                    "WeightMatrix: {size} scales for {rows} output rows"
                )));
            }
            // On-disk scale is `max_abs / 127`, the direct recovery factor —
            // kept verbatim (see the module docs). SIMD padding may leave extra
            // trailing scales; use the first `rows`.
            let scales: Vec<f32> = raw_scales[..rows].iter().map(|&s| s as f32).collect();
            // Dequantize through the Kopitiam Runtime, then keep the dense f32.
            let dense = Tensor::tessdata_int8_to_f32(&wi, &scales, rows, cols)?.to_vec_f32()?;
            Ok(WeightMatrix {
                int_mode,
                rows,
                cols,
                dense,
            })
        } else {
            let (wf, rows, cols) = read_f64_matrix(fp)?;
            if is_training {
                // Discard the training-only `updates` and (iff Adam) `dw_sq_sum`.
                let _ = read_f64_matrix(fp)?;
                if use_adam {
                    let _ = read_f64_matrix(fp)?;
                }
            }
            Ok(WeightMatrix {
                int_mode,
                rows,
                cols,
                dense: wf,
            })
        }
    }

    /// Builds a weight matrix directly from dense `f32` values — for tests and
    /// for hand-constructing tiny networks. `dense` is row-major `[rows*cols]`,
    /// the last column of each row being the bias.
    pub fn from_dense(rows: usize, cols: usize, dense: Vec<f32>) -> Self {
        assert_eq!(dense.len(), rows * cols, "dense length must be rows*cols");
        WeightMatrix {
            int_mode: false,
            rows,
            cols,
            dense,
        }
    }

    /// The number of output rows (`num_outputs`, `dim1`). Tesseract:
    /// `WeightMatrix::NumOutputs`.
    pub fn num_outputs(&self) -> usize {
        self.rows
    }

    /// The number of input columns *including* the trailing bias column
    /// (`dim2 = num_inputs + 1`).
    pub fn num_inputs_with_bias(&self) -> usize {
        self.cols
    }

    /// `true` if this matrix was loaded from the int8 branch.
    pub fn is_int_mode(&self) -> bool {
        self.int_mode
    }

    /// The dequantized `[rows, cols]` weights as a [`Tensor`], bias column
    /// included. Used by the batched [`crate::fullyconnected`] matmul path.
    pub fn weights_tensor(&self) -> Result<Tensor> {
        Ok(Tensor::from_f32(
            self.dense.clone(),
            [self.rows, self.cols],
        )?)
    }

    /// Computes `v = W·u` with the implied bias: `u` has length
    /// `num_inputs_with_bias() - 1`, and each output is
    /// `Σ_j W[i][j]·u[j] + W[i][last]`. Returns a fresh vector of length
    /// `num_outputs()`.
    ///
    /// Tesseract: `WeightMatrix::MatrixDotVector` via `MatrixDotVectorInternal`
    /// with `add_bias_fwd = true` (weightmatrix.cpp:99,393).
    pub fn matrix_dot_vector(&self, u: &[f32]) -> Vec<f32> {
        let extent = self.cols - 1;
        debug_assert_eq!(u.len(), extent, "input length must be num_inputs (cols-1)");
        self.dense
            .chunks_exact(self.cols)
            .map(|row| {
                // Σ_j w[j]·u[j] + bias (the trailing column).
                let dot: f32 = row[..extent].iter().zip(u).map(|(&w, &x)| w * x).sum();
                dot + row[extent]
            })
            .collect()
    }
}

/// Reads a `GENERIC_2D_ARRAY<int8_t>`: `int32 dim1`, `int32 dim2`, `int8 empty`,
/// then `dim1*dim2` `int8`. Returns `(data, dim1, dim2)`.
///
/// Tesseract: `GENERIC_2D_ARRAY::DeSerialize(TFile*)` + `DeSerializeSize`
/// (matrix.h:197,567).
fn read_i8_matrix(fp: &mut TFile<'_>) -> Result<(Vec<i8>, usize, usize)> {
    let (dim1, dim2) = read_matrix_size(fp)?;
    let _empty = fp.read_i8()?;
    let data = fp.read_i8_vec(dim1 * dim2)?;
    Ok((data, dim1, dim2))
}

/// Reads a `GENERIC_2D_ARRAY<double>` as `f32`: `int32 dim1`, `int32 dim2`,
/// `double empty`, then `dim1*dim2` `double`. Returns `(data_as_f32, dim1, dim2)`.
fn read_f64_matrix(fp: &mut TFile<'_>) -> Result<(Vec<f32>, usize, usize)> {
    let (dim1, dim2) = read_matrix_size(fp)?;
    let _empty = fp.read_f64()?;
    let data = fp.read_f64_vec(dim1 * dim2)?;
    Ok((data.into_iter().map(|d| d as f32).collect(), dim1, dim2))
}

/// Reads and validates the `int32 dim1`, `int32 dim2` header of a
/// `GENERIC_2D_ARRAY`. Tesseract: `DeSerializeSize(TFile*)` (matrix.h:567).
fn read_matrix_size(fp: &mut TFile<'_>) -> Result<(usize, usize)> {
    let dim1 = fp.read_i32()?;
    let dim2 = fp.read_i32()?;
    if !(0..=MAX_DIM).contains(&dim1) || !(0..=MAX_DIM).contains(&dim2) {
        return Err(Error::format(format!(
            "WeightMatrix: implausible matrix dims {dim1}x{dim2}"
        )));
    }
    Ok((dim1 as usize, dim2 as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The int8 range endpoint used as the quantization pivot (`INT8_MAX`).
    const INT8_MAX: f32 = 127.0;

    /// Serialize one row exactly as `WeightMatrix::ConvertToInt` +
    /// `Serialize` would: int8 line, on-disk scale = `max_abs / 127`.
    fn encode_row_disk(row: &[f32]) -> (Vec<i8>, f64) {
        let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = max_abs / INT8_MAX; // on-disk recovery scale.
        let div = if scale == 0.0 { 1.0 } else { scale };
        let q = row
            .iter()
            .map(|&w| (w / div).round().clamp(-INT8_MAX, INT8_MAX) as i8)
            .collect();
        (q, scale as f64)
    }

    /// Little-endian test-buffer builder (matches the round-trip: we write with
    /// native bytes and read with swap off on a little-endian host; on a
    /// big-endian host `to_ne_bytes` still round-trips, as `TFile` reads native).
    #[derive(Default)]
    struct Buf(Vec<u8>);
    impl Buf {
        fn u8(&mut self, v: u8) -> &mut Self {
            self.0.push(v);
            self
        }
        fn u32(&mut self, v: u32) -> &mut Self {
            self.0.extend_from_slice(&v.to_ne_bytes());
            self
        }
        fn i32(&mut self, v: i32) -> &mut Self {
            self.0.extend_from_slice(&v.to_ne_bytes());
            self
        }
        fn i8(&mut self, v: i8) -> &mut Self {
            self.0.push(v as u8);
            self
        }
        fn f64(&mut self, v: f64) -> &mut Self {
            self.0.extend_from_slice(&v.to_ne_bytes());
            self
        }
    }

    #[test]
    fn matrix_dot_vector_honors_the_trailing_bias_column() {
        // 2 outputs, 2 inputs + bias. W = [[1,2, 0.5], [3,4, -1.0]].
        // u = [10, 20] => v0 = 1*10+2*20+0.5 = 50.5 ; v1 = 3*10+4*20-1 = 109.
        let w = WeightMatrix::from_dense(2, 3, vec![1.0, 2.0, 0.5, 3.0, 4.0, -1.0]);
        let v = w.matrix_dot_vector(&[10.0, 20.0]);
        assert_eq!(v, vec![50.5, 109.0]);
    }

    #[test]
    fn deserialize_int8_matrix_then_dot_matches_hand_computation() {
        // Build a synthetic int8 WeightMatrix TFile buffer: 1 output row, 3
        // inputs + bias (cols = 4). Choose weights, encode as ConvertToInt does.
        let row = vec![0.20f32, -0.40, 0.10, 0.40]; // last entry is the bias weight
        let (q, scale) = encode_row_disk(&row);

        let mut b = Buf::default();
        b.u8(K_INT8_FLAG | K_DOUBLE_FLAG); // mode: int8 + double scales
        // GENERIC_2D_ARRAY<int8_t>: dim1=1, dim2=4, empty=0, then 4 int8.
        b.i32(1).i32(4).i8(0);
        for &qi in &q {
            b.i8(qi);
        }
        // scales: uint32 count=1, then 1 double (on-disk max_abs/127).
        b.u32(1).f64(scale);

        let mut fp = TFile::new(&b.0);
        let w = WeightMatrix::deserialize(false, &mut fp).unwrap();
        assert!(w.is_int_mode());
        assert_eq!(w.num_outputs(), 1);
        assert_eq!(w.num_inputs_with_bias(), 4);

        // v = Σ_j w[j]*u[j] + bias, with dequantized weights ≈ q[j]*scale.
        let u = [1.0f32, 2.0, 3.0];
        let v = w.matrix_dot_vector(&u);
        let dq: Vec<f32> = q.iter().map(|&qi| qi as f32 * scale as f32).collect();
        let expected = dq[0] * u[0] + dq[1] * u[1] + dq[2] * u[2] + dq[3];
        assert!(
            (v[0] - expected).abs() < 1e-5,
            "got {}, expected {expected}",
            v[0]
        );
        // And within quantization error of the *original* intended weights.
        let ideal = row[0] * u[0] + row[1] * u[1] + row[2] * u[2] + row[3];
        assert!((v[0] - ideal).abs() < scale as f32 * 4.0);
    }

    #[test]
    fn deserialize_float_matrix_reads_doubles() {
        // Float branch: mode = kDoubleFlag only. 1x2 matrix [3.5, -2.0].
        let mut b = Buf::default();
        b.u8(K_DOUBLE_FLAG);
        b.i32(1).i32(2).f64(0.0); // dim1, dim2, empty
        b.f64(3.5).f64(-2.0);
        let mut fp = TFile::new(&b.0);
        let w = WeightMatrix::deserialize(false, &mut fp).unwrap();
        assert!(!w.is_int_mode());
        // u has length cols-1 = 1: v = 3.5*u + (-2.0 bias).
        assert_eq!(w.matrix_dot_vector(&[2.0]), vec![3.5 * 2.0 - 2.0]);
    }

    #[test]
    fn legacy_format_without_double_flag_is_rejected() {
        let b = [K_INT8_FLAG]; // no kDoubleFlag
        let mut fp = TFile::new(&b);
        assert_eq!(
            WeightMatrix::deserialize(false, &mut fp)
                .unwrap_err()
                .kind(),
            crate::error::ErrorKind::Format
        );
    }
}
