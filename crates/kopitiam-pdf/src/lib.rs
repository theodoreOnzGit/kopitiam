//! Extraction layer: recovers physical layout (text, position, font size)
//! from PDF files. Produces `Page`/`TextSpan` values only; no semantic
//! reconstruction (headings, paragraphs, tables, ...) happens in this crate
//! -- that is `kopitiam-document`'s job.

mod extractor;
mod font;
mod font_resources;
mod geometry;
mod page;

pub use extractor::{ExtractError, extract, extract_from_bytes};
pub use font::FontStyle;
pub use geometry::Rect;
pub use page::{Page, TextSpan};
