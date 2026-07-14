//! Document AST and reconstruction: turns `kopitiam_pdf::Page` layout data
//! into a semantic `Document` (headings, paragraphs, lists, tables, figures),
//! plus a validation report auditing the conversion.

mod block;
mod citation;
mod document;
mod figure;
mod heading;
mod list;
mod paragraph;
mod reconstruction;
mod table;
mod validation;

pub use block::{Block, CodeBlock, Quote};
pub use citation::Citation;
pub use document::{Document, Metadata};
pub use figure::Figure;
pub use heading::Heading;
pub use list::List;
pub use paragraph::Paragraph;
pub use reconstruction::reconstruct;
pub use table::Table;
pub use validation::{ConversionReport, validate};
