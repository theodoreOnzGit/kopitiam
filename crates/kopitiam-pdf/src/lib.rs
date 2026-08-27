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
// hit-testing, forms UI, ...), lifted out of the `kpdf` example binary so
// other KOPITIAM front ends can reuse them. Gated on the same `kpdf`
// feature that turns on the optional eframe/egui/rfd dependencies -- see
// Cargo.toml's `[features]` section. Named `gui_frontend` rather than
// `egui` on purpose, so it never shadows the external `egui` crate inside
// its own files.
#[cfg(feature = "kpdf")]
pub mod gui_frontend;

pub use extractor::{ExtractError, extract, extract_from_bytes};
pub use font::FontStyle;
pub use geometry::Rect;
pub use mupdf_extract::{extract_mupdf, extract_mupdf_from_bytes};
pub use page::{Page, TextSpan};
