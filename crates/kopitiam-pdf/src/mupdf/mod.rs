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
//! Only the FOUNDATION geometry module -- `Point`, `Rect`, `IRect`, `Matrix`,
//! `Quad` and their transforms. It is the pattern-setter; no other MuPDF module
//! is ported yet, hor. Don't reach for `stext` here -- not built.

pub mod geometry;

pub use geometry::{IRect, Matrix, Point, Quad, Rect};
