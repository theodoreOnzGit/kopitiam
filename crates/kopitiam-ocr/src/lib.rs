//! # `kopitiam-ocr` — the KOPITIAM OCR engine
//!
//! An OCR engine for KOPITIAM, **translated to Rust from Tesseract** (the C++ OCR
//! engine, © Google Inc. and contributors, Author: Ray Smith et al., Apache-2.0,
//! vendored read-only at `crates/kopitiam-ocr/vendor/tesseract`, commit
//! `db0ec62`). KOPITIAM is **AGPL-3.0-only**; Apache-2.0 is one-way compatible
//! with AGPLv3, so Tesseract's algorithms adapt into this AGPLv3 work provided
//! their copyright notices travel with the code. This is a **close adaptation /
//! translation**, not clean-room study: each module records its exact upstream
//! source, commit, license, and copyright in a per-file provenance header, and
//! carries `// Tesseract: <fn> (<file>:<line>)` breadcrumbs at the point of use.
//! Attribution is recorded per `docs/ACKNOWLEDGEMENTS.md` (the same
//! translation/provenance discipline as the MuPDF port; see
//! `docs/ai-decisions/AID-0051`).
//!
//! ## Phase 1: the `.traineddata` container parser
//!
//! This first phase ports the **read path** of Tesseract's binary container so a
//! KOPITIAM crate can open a `.traineddata` file and locate its components
//! (especially the LSTM model, unicharset, and recoder) without linking Tesseract:
//!
//! * [`serialis`] — [`TFile`], Tesseract's portable little/big-endian binary
//!   reader over a byte buffer (`src/ccutil/serialis.{cpp,h}`).
//! * [`tessdata`] — [`TessdataManager`], the `.traineddata` container: a tagged
//!   offset table of [`TessdataType`] components (`src/ccutil/tessdatamanager.{cpp,h}`).
//! * [`error`] — the crate's local [`Error`]/[`Result`], into which Tesseract's
//!   `return false` read-path failures are mapped.
//!
//! Only the read/parse path is ported; the write/combine path is deferred (each
//! module's header says what and why).
//!
//! ## Phase 2: the character set + CJK recoder
//!
//! This phase ports the read path of Tesseract's character machinery, so a
//! KOPITIAM crate can turn the LSTM model's numeric outputs back into text:
//!
//! * [`unichar`] — [`Unichar`], Tesseract's single-codepoint UTF-8 handling
//!   (`src/ccutil/unichar.{cpp,h}`).
//! * [`unicharmap`] — [`Unicharmap`], the UTF-8 → id trie (`src/ccutil/unicharmap.{cpp,h}`).
//! * [`unicharset`] — [`Unicharset`], the id↔UTF-8 table with per-char
//!   properties and the text-format load path (`src/ccutil/unicharset.{cpp,h}`).
//! * [`unicharcompress`] — [`UnicharCompress`]/[`RecodedCharID`], the **CJK
//!   recoder** that maps each unichar id to a short sequence of small codes and
//!   back — why chi_sim/jpn work (`src/ccutil/unicharcompress.{cpp,h}`).
//!
//! ## Phase 4: the LSTM recognizer network (forward evaluation)
//!
//! This phase ports Tesseract's neural-network engine (`src/lstm/*`) so a
//! KOPITIAM crate can build the recognizer graph from a `.traineddata` `Lstm`
//! component and run a forward pass, producing per-timestep softmax logits over
//! the recoded alphabet — the input to the beam decoder (phase 5, later). The
//! numeric work runs on the [`kopitiam_tensor`] runtime (matmul, `tanh`/
//! `sigmoid`, `lstm_cell`, `tessdata_int8_to_f32`, `softmax`).
//!
//! * [`network`] — the [`Network`] graph loader: the [`NetworkType`] tags, the
//!   [`NetworkNode`] trait, and [`create_from_file`], the node-type dispatch.
//! * [`stridemap`]/[`networkio`] — the 4-D→2-D index arithmetic and the
//!   `[timesteps × features]` float activation buffer passed between layers.
//! * [`weightmatrix`] — the [`WeightMatrix`] (int8 + per-row scale, or float)
//!   and `MatrixDotVector` with the trailing-bias convention.
//! * [`plumbing`]/[`series`]/[`parallel`]/[`reversed`] — the composite layers.
//! * [`fullyconnected`]/[`lstm`] — the dense + recurrent functional layers.
//! * [`input`]/[`reconfig`]/[`maxpool`]/[`convolve`] — the I/O and 2-D plumbing.
//!
//! Every node type deserializes; forward evaluation targets the 1-D text-line
//! path (each layer's module documents what its forward defers).

pub mod error;
pub mod serialis;
pub mod tessdata;
pub mod unichar;
pub mod unicharcompress;
pub mod unicharmap;
pub mod unicharset;

// Phase 4: the LSTM recognizer network — graph loader + forward evaluation.
pub mod convolve;
pub mod fullyconnected;
pub mod input;
pub mod lstm;
pub mod maxpool;
pub mod network;
pub mod networkio;
pub mod parallel;
pub mod plumbing;
pub mod reconfig;
pub mod reversed;
pub mod series;
pub mod stridemap;
pub mod weightmatrix;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod network_tests;

pub use error::{Error, ErrorKind, Result};
pub use network::{Network, NetworkHeader, NetworkNode, NetworkType, TYPE_NAMES, create_from_file};
pub use networkio::NetworkIO;
pub use serialis::TFile;
pub use tessdata::{FILE_SUFFIXES, MAX_NUM_TESSDATA_ENTRIES, NUM_ENTRIES, TessdataManager, TessdataType};
pub use weightmatrix::WeightMatrix;
pub use unichar::{
    Char32, INVALID_UNICHAR, INVALID_UNICHAR_ID, UNICHAR_LEN, Unichar, UnicharId, utf8_step,
    utf8_to_utf32, utf32_to_utf8,
};
pub use unicharcompress::{RecodedCharID, UnicharCompress};
pub use unicharmap::Unicharmap;
pub use unicharset::{
    CharFragment, SPECIAL_UNICHAR_CODES, SPECIAL_UNICHAR_CODES_COUNT, UNICHAR_BROKEN,
    UNICHAR_JOINED, UNICHAR_SPACE, Unicharset,
};
