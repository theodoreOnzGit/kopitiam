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

pub mod error;
pub mod serialis;
pub mod tessdata;

pub use error::{Error, ErrorKind, Result};
pub use serialis::TFile;
pub use tessdata::{FILE_SUFFIXES, MAX_NUM_TESSDATA_ENTRIES, NUM_ENTRIES, TessdataManager, TessdataType};
