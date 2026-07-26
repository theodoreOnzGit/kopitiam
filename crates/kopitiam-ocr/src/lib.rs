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
//!
//! ## Phase 5: the recode/CTC beam-search decoder
//!
//! This phase ports Tesseract's beam decoder (`src/lstm/recodebeam.{cpp,h}`), the
//! last recognition step: it turns the network's per-timestep softmax over the
//! recoded alphabet into recognized text.
//!
//! * [`recodebeam`] — [`RecodeBeamSearch`]/[`RecodeNode`], the beam over the
//!   **recoded** alphabet. It tracks the CTC blank (`null_char`), folds duplicate
//!   frames, and extends beam nodes only by the recoder's valid next codes, so a
//!   multi-unit CJK code sequence reassembles (via [`UnicharCompress`]) into one
//!   unichar id. [`RecodeBeamSearch::decode`] runs the beam across all timesteps;
//!   [`RecodeBeamSearch::extract_best_path_as_unichar_ids`] /
//!   [`RecodeBeamSearch::best_path_string`] emit the best path (with the
//!   [`decode_to_unichar_ids`]/[`decode_to_string`] convenience wrappers).
//!
//! The non-dictionary (raw CTC beam) path is ported in full; the dawg/dictionary
//! beam and char-box/`WERD_RES` extraction are deferred (see the module docs).
//!
//! ## Phase 6: the LSTM recognizer line driver
//!
//! This phase ties phases 1–5 together (`src/lstm/lstmrecognizer.{cpp,h}` +
//! `src/lstm/input.cpp`): load a recognizer from a `.traineddata` and recognize
//! a single normalized grayscale line image into text.
//!
//! * [`lstmrecognizer`] — [`LstmRecognizer`], the line driver.
//!   [`LstmRecognizer::load`] reads the `LstmUnicharset`, `LstmRecoder`, and
//!   `Lstm` components (network tree + scalar params, of which only `null_char_`
//!   matters at inference) and enforces the `num_outputs == code_range + 1`
//!   CTC-null invariant. [`LstmRecognizer::recognize_line`] normalizes a
//!   [`GrayLine`] (scale-to-height + the adaptive black/white contrast stretch
//!   from `input.cpp`/`networkio.cpp`, *not* a plain pixel/255), runs the forward
//!   pass, and decodes with the raw beam from phase 5.
//!
//! The image side that *produces* the [`GrayLine`] — the page rasterizer,
//! Leptonica preprocessing, and line-finding — is a later phase; `pixScale` is
//! approximated with bilinear resampling for now (see the module docs).
//!
//! ## Phase 7: the Leptonica image-preprocessing subset
//!
//! This phase ports the **Leptonica** (not Tesseract) image operations between
//! a rendered page raster and line-finding: grayscale conversion, binarization,
//! and scaling. Leptonica is BSD-2-Clause (© 2001-2020 Leptonica, Dan
//! Bloomberg), one-way compatible with AGPLv3; sources are vendored read-only
//! at `crates/kopitiam-ocr/vendor/leptonica` (commit `10bdea2`). Same raw
//! 8-bit-buffer style as [`GrayLine`] — no `image`-crate dependency.
//!
//! * [`image`] — the page-image types ([`GrayImage`], [`RgbImage`],
//!   [`RgbaImage`], the [`BinaryImage`] alias) and [`to_gray`], RGB(A)→gray with
//!   Leptonica's perceptual luma weights (`pixConvertRGBToGray`).
//! * [`binarize`] — [`otsu_binarize`] (global Otsu histogram split,
//!   `numaSplitDistribution`) and [`sauvola_binarize`] (adaptive local-mean/std
//!   threshold via summed-area tables, `pixSauvolaBinarize`).
//! * [`scale`] — [`scale_gray`]/[`scale_gray_li`], Leptonica's fixed-point
//!   grayscale linear-interpolation resampler (`pixScaleGrayLI`), the faithful
//!   form of the bilinear approximation Phase 6 uses to normalize line height.
//!
//! Only the minimal LSTM-path subset is ported; deskew/rotate, morphology,
//! color, the packed 1-bpp `Pix` API, tiled/adaptive thresholding, and image
//! I/O are deferred (each module documents what and why).

pub mod error;
pub mod serialis;

// Phase 7: the Leptonica image-preprocessing subset — grayscale conversion,
// binarization, and scaling (the steps between a rendered page raster and
// line-finding). Ported from Leptonica (BSD-2-Clause), not Tesseract.
pub mod binarize;
pub mod image;
pub mod scale;
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
pub mod lstmrecognizer;
pub mod maxpool;
pub mod network;
pub mod networkio;
pub mod parallel;
pub mod plumbing;
pub mod recodebeam;
pub mod reconfig;
pub mod reversed;
pub mod series;
pub mod stridemap;
pub mod weightmatrix;

#[cfg(test)]
mod network_tests;
#[cfg(test)]
mod test_support;

pub use binarize::{otsu_binarize, otsu_threshold, sauvola_binarize};
pub use error::{Error, ErrorKind, Result};
pub use image::{
    BINARY_BG, BINARY_FG, BinaryImage, GrayImage, RgbImage, RgbSource, RgbaImage, to_gray,
};
pub use lstmrecognizer::{GrayLine, LstmRecognizer};
pub use scale::{scale_gray, scale_gray_li};
pub use network::{Network, NetworkHeader, NetworkNode, NetworkType, TYPE_NAMES, create_from_file};
pub use networkio::NetworkIO;
pub use recodebeam::{
    BestPath, RecodeBeamSearch, RecodeNode, decode_to_string, decode_to_unichar_ids,
};
pub use serialis::TFile;
pub use tessdata::{
    FILE_SUFFIXES, MAX_NUM_TESSDATA_ENTRIES, NUM_ENTRIES, TessdataManager, TessdataType,
};
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
pub use weightmatrix::WeightMatrix;
