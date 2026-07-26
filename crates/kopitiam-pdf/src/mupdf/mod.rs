//! # KOPITIAM MuPDF port
//!
//! This module is a **faithful Rust port of MuPDF's text-extraction vertical** --
//! the `fitz` geometry primitives, and (in later phases) the structured-text
//! (`stext`) device that walks a page and hands back blocks / lines / spans with
//! bounding boxes. MuPDF is the C engine PyMuPDF binds; we translate the parts
//! KOPITIAM needs for `kopitiam-pdf` / `kopitiam-document` into idiomatic Rust,
//! instead of shelling out to a C library. It lives here as the `mupdf` module
//! inside `kopitiam-pdf`, not as a standalone crate.
//!
//! ## Provenance, stated plainly
//!
//! MuPDF is **AGPL-3.0**, © Artifex Software, Inc. KOPITIAM is **AGPL-3.0-only**,
//! and that relicense is exactly what makes this port permissible: an AGPL
//! upstream can be adapted into an AGPLv3 work, it cannot be absorbed into a
//! permissive one. This is a **close adaptation / translation**, not clean-room
//! study -- the algorithms and numeric behaviour follow MuPDF, only re-expressed
//! in Rust. So the rules from `docs/ACKNOWLEDGEMENTS.md` ("PDF &
//! document-extraction references") apply hard here:
//!
//! * Every ported file carries a provenance header naming the MuPDF source file,
//!   the pinned commit (`19f1284`), the AGPL-3.0 licence, and the Artifex
//!   copyright.
//! * Where a Rust function tracks a specific C function, a `// MuPDF: fz_<name>`
//!   comment sits at that function so the 1:1 mapping stays discoverable.
//! * The vendored C under `crates/kopitiam-pdf/vendor/mupdf` is read-only
//!   reference material -- never built, linked, or shipped.
//!
//! The translation conventions for the whole port (how `fz_matrix` becomes
//! `Matrix`, how MuPDF's rect empty/infinite/invalid conventions are preserved,
//! how `fz_context` / `fz_try` / `fz_throw` will map to Rust ownership and
//! `Result` in later modules) live in
//! **`docs/ai-decisions/AID-0051-mupdf-port-conventions.md`**. Read that before
//! porting the next module, so later phases follow the same pattern this one
//! sets.
//!
//! ## What's here now
//!
//! The FOUNDATION layer (`fitz` primitives everything else builds on):
//! * `geometry` -- `Point`, `Rect`, `IRect`, `Matrix`, `Quad` and their transforms.
//! * `error` -- MuPDF's `fz_error_type` taxonomy as `Error` / `ErrorKind` /
//!   `Result`; the setjmp/longjmp exception machinery is mapped to Rust `Result`,
//!   and this is the port-wide error type later modules return.
//! * `string_util` -- the UTF-8 rune codec (`chartorune`/`runetochar`) and the
//!   bounded string helpers, preserving MuPDF's exact invalid-sequence handling.
//! * `encodings` -- the static base-encoding tables (Standard / WinAnsi /
//!   MacRoman / MacExpert / PdfDoc), entry-for-entry from `encodings.h`.
//! * `hash` -- MuPDF's fixed-key open-addressing hash table (exact hash fn).
//! * `pool` -- the block-chained bump arena (the `stext` page lives in one).
//!
//! Still ahead: buffers/streams, the `filter-*` decoders, the PDF object model /
//! parser / xref, the content interpreter, fonts/CMaps/ToUnicode, and the `stext`
//! device + layout analysis (`boxer`/`para`) -- the parts that actually fix the
//! two-column reading order and the spurious inter-glyph spaces. Not built yet, hor.

pub mod buffer;
pub mod encodings;
pub mod error;
pub mod filter_basic;
pub mod filter_flate;
pub mod filter_lzw;
pub mod filter_predict;
pub mod geometry;
pub mod hash;
pub mod pool;
pub mod stream;
pub mod string_util;

pub use buffer::Buffer;
pub use encodings::BaseEncoding;
pub use error::{Error, ErrorKind, Result};
pub use geometry::{IRect, Matrix, Point, Quad, Rect};
pub use hash::HashTable;
pub use pool::{Handle, Pool};
pub use stream::{Stream, StreamSource, Whence};
