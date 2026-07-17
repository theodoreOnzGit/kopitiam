//! kvim's text engine: the rope-backed [`Buffer`], its branching undo
//! history, and the marks that survive edits.
//!
//! # Layout
//!
//! * [`buffer`] — [`Buffer`], the type every other `kvim` module actually
//!   uses. Orchestrates the pieces below; see its module docs for how they
//!   fit together.
//! * [`grapheme`] — `Position::col` <-> rope offset conversion. See its
//!   module docs for why grapheme segmentation runs per-line rather than
//!   over the whole rope.
//! * [`undo`] — the branching undo tree and edit grouping.
//! * [`mark`] — how a mark's rope offset moves when an edit lands before,
//!   after, or on top of it.
//! * [`line_ending`] — LF/CRLF detection and preservation.
//!
//! [`Buffer`] and [`LineEdit`] are the only exports. The rest are
//! implementation details other `kvim` modules should never need to name
//! directly — depending on `text::grapheme` or `text::undo` from outside this
//! module would be a sign the abstraction boundary is in the wrong place.
//! [`LineEdit`] is public because folds live one layer up (on the editor) and
//! must be told how each edit moved the lines; see its docs.

mod buffer;
mod grapheme;
mod line_ending;
mod mark;
mod undo;

pub use buffer::{Buffer, LineEdit};
