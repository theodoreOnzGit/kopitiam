//! Test-only helpers that write synthetic, hermetic `.traineddata`-style LSTM
//! byte buffers, so the network loader can be exercised end-to-end (deserialize
//! + forward) without downloading a real model.
//!
//! The layout mirrors Tesseract's `Network::Serialize` (network.cpp:155) and
//! `WeightMatrix::Serialize` (weightmatrix.cpp:238) at commit db0ec62. Values are
//! written with native-endian bytes and read back on the same host with the swap
//! flag off — the same round-trip strategy the `serialis` tests use, correct on
//! both little- and big-endian hosts.

use crate::network::{NetworkType, TYPE_NAMES};

/// The `kDoubleFlag` bit that marks a "new format" (double) weight matrix.
const K_DOUBLE_FLAG: u8 = 128;

/// A growable little-helper for building test byte buffers.
#[derive(Default)]
pub struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer::default()
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.bytes.push(v);
        self
    }

    pub fn i8(&mut self, v: i8) -> &mut Self {
        self.bytes.push(v as u8);
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_ne_bytes());
        self
    }

    /// A length-prefixed string (`uint32` len + bytes), Tesseract's
    /// `TFile::Serialize(std::string)`.
    pub fn string(&mut self, s: &str) -> &mut Self {
        self.u32(s.len() as u32);
        self.bytes.extend_from_slice(s.as_bytes());
        self
    }

    /// A network layer header, mirroring `Network::Serialize`: type tag `0`,
    /// the serialized type name, training/needs-backprop bytes, flags, `ni`,
    /// `no`, `num_weights`, and the layer name. `training = false` (a
    /// recognition model) and `flags = 0` (no per-layer LR).
    pub fn header(&mut self, ntype: NetworkType, ni: i32, no: i32, name: &str) -> &mut Self {
        self.i8(0); // NT_NONE tag => the type follows as a name string.
        self.string(TYPE_NAMES[ntype as usize]);
        self.i8(0); // training: TS_DISABLED (not TS_ENABLED).
        self.i8(0); // needs_to_backprop: false.
        self.i32(0); // network_flags.
        self.i32(ni);
        self.i32(no);
        self.i32(0); // num_weights (unused by the loader).
        self.string(name);
        self
    }

    /// A float `WeightMatrix`: `mode = kDoubleFlag`, then a
    /// `GENERIC_2D_ARRAY<double>` (`dim1`, `dim2`, `empty`, `dim1*dim2` doubles).
    /// `dense` is row-major `[rows*cols]`, the last column of each row the bias.
    pub fn weight_matrix_float(&mut self, rows: i32, cols: i32, dense: &[f64]) -> &mut Self {
        assert_eq!(dense.len() as i32, rows * cols);
        self.u8(K_DOUBLE_FLAG);
        self.i32(rows); // dim1
        self.i32(cols); // dim2
        self.f64(0.0); // empty_
        for &v in dense {
            self.f64(v);
        }
        self
    }
}
