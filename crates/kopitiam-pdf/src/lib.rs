//! Extraction layer: recovers physical layout (text, position, font size)
//! from PDF files. Produces `Page`/`TextSpan` values only; no semantic
//! reconstruction (headings, paragraphs, tables, ...) happens in this crate
//! -- that is `kopitiam-document`'s job.

mod extractor;
mod font;
mod font_resources;
mod geometry;
mod page;

// Faithful Rust port of MuPDF's text-extraction vertical (fitz geometry, and
// later the stext device). Kept as a namespaced module -- its `Rect` is MuPDF's
// f32 geometry rect, distinct from this crate's own `geometry::Rect` for text
// spans, so it is deliberately NOT re-exported at the crate root. Reach it as
// `kopitiam_pdf::mupdf::geometry::{Point, Rect, IRect, Matrix, Quad}`.
pub mod mupdf;

pub use extractor::{ExtractError, extract, extract_from_bytes};
pub use font::FontStyle;
pub use geometry::Rect;
pub use page::{Page, TextSpan};
