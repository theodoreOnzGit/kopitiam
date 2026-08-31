//! Extraction layer: recovers physical layout (text, position, font size)
//! from PDF files. Produces `Page`/`TextSpan` values only; no semantic
//! reconstruction (headings, paragraphs, tables, ...) happens in this crate
//! -- that is `kopitiam-document`'s job.

mod extractor;
mod font;
mod font_resources;
mod geometry;
mod mupdf_extract;
mod page;
mod textnorm;

// Faithful Rust port of MuPDF's text-extraction vertical (fitz geometry, and
// later the stext device). Kept as a namespaced module -- its `Rect` is MuPDF's
// f32 geometry rect, distinct from this crate's own `geometry::Rect` for text
// spans, so it is deliberately NOT re-exported at the crate root. Reach it as
// `kopitiam_pdf::mupdf::geometry::{Point, Rect, IRect, Matrix, Quad}`.
pub mod mupdf;

// Reusable egui-based PDF-viewer building blocks (page layout, zoom,
// hit-testing, forms UI, ...), lifted out of the `kpdf` binary so other
// KOPITIAM front ends can reuse them.
//
// Gated on `egui`, NOT on `kpdf` -- that distinction is the point (gh-96
// Phase 11). `kpdf` means "the standalone viewer application", and pulls in
// an eframe event loop and the `rfd` native file picker on top. An embedding
// application supplies both of those itself, so it takes `egui` alone and
// compiles neither. `kpdf` implies `egui`, so the binary still sees all of
// this.
//
// Named `gui_frontend` rather than `egui` on purpose, so it never shadows
// the external `egui` crate inside its own files.
#[cfg(feature = "egui")]
pub mod gui_frontend;

// `kpdf` is defined as a superset of `egui` (Cargo.toml `[features]`). If that
// ever stops holding -- someone edits the feature list, or a consumer manages
// to select `kpdf` without `egui` -- the binary would compile against a crate
// with no `gui_frontend` at all and fail deep inside kpdf.rs with a pile of
// unresolved-import errors. Say it here instead, once, in a sentence.
#[cfg(all(feature = "kpdf", not(feature = "egui")))]
compile_error!(
    "feature `kpdf` requires feature `egui` (kpdf = the reusable reader PLUS \
     the standalone eframe/rfd shell). Fix Cargo.toml's [features] so `kpdf` \
     lists `egui`."
);

pub use extractor::{ExtractError, extract, extract_from_bytes};
pub use font::FontStyle;
pub use geometry::Rect;
pub use mupdf_extract::{extract_mupdf, extract_mupdf_from_bytes, extract_mupdf_page};
pub use page::{Page, TextSpan};
