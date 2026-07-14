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
//! Only [`Buffer`] is exported. The rest are implementation details other
//! `kvim` modules should never need to name directly — depending on
//! `text::grapheme` or `text::undo` from outside this module would be a
//! sign the abstraction boundary is in the wrong place.

mod buffer;
mod grapheme;
mod line_ending;
mod mark;
mod undo;

pub use buffer::Buffer;
