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
//! The PDF OBJECT layer (bytes -> PDF objects), on top of the streams:
//! * `object` -- MuPDF's polymorphic `pdf_obj` as the Rust [`Object`] enum
//!   (null/bool/int/real/string/name/array/dict/indirect), with the `pdf_is_*`
//!   predicates and `pdf_to_*` / dict / array accessors. Streams stay dicts (as
//!   in MuPDF); the raw-byte seam lives on the parser's indirect-object result.
//! * `lex` -- the `pdf_lex` tokenizer over a `Stream` ([`Token`]).
//! * `parse` -- `pdf_parse_array`/`dict`/`ind_obj`: tokens -> [`Object`] trees,
//!   returning an [`IndirectObject`] that records the stream body's start offset
//!   ([`StreamRange`]) for the xref layer to resolve.
//!
//! Still ahead: the `filter-*` decoders' remaining wiring, the xref / document
//! layer (indirect resolution, `/ObjStm` object streams, stream decode,
//! encryption), the content interpreter, fonts/CMaps/ToUnicode, and the `stext`
//! device + layout analysis (`boxer`/`para`) -- the parts that actually fix the
//! two-column reading order and the spurious inter-glyph spaces. Not built yet, hor.

pub mod agl;
pub mod agl_data;
pub mod annot_appearance;
pub mod annot_edit;
pub mod annot_run;
pub mod buffer;
pub mod cmap;
pub mod doc_info;
pub mod doc_stream;
pub mod draw_device;
pub mod draw_edge;
pub mod draw_path;
pub mod encodings;
pub mod error;
pub mod filter_basic;
pub mod filter_flate;
pub mod filter_lzw;
pub mod filter_predict;
pub mod font;
pub mod form;
pub mod geometry;
pub mod glyph;
pub mod glyph_cff;
pub mod glyph_skrifa;
pub mod glyph_truetype;
pub mod glyph_type1;
pub mod hash;
pub mod hayro_fallback;
pub mod interpret;
pub mod lex;
pub mod object;
pub mod op_run;
pub mod page_edit;
pub mod page_image;
pub mod page_run;
pub mod parse;
pub mod pixmap;
pub mod pool;
pub mod resources;
pub mod standard_font;
pub mod stext_boxer;
pub mod stext_classify;
pub mod stext_device;
pub mod stext_iterator;
pub mod stext_para;
pub mod stream;
pub mod string_util;
pub mod structured_text;
pub mod text_device;
pub mod write;
pub mod xref;

pub use agl::{unicode_from_glyph_name, unicode_from_glyph_name_strict};
pub use annot_appearance::{
    SynthAp, annot_border_width, annot_color_rgb, annot_opacity, synthesize_ap,
};
pub use annot_edit::{
    AnnotRef, EditHistory, InkAnnotSpec, InkStroke, add_ink_annot, delete_annot, page_annot_refs,
};
pub use annot_run::{AnnotPass, run_page_annots, run_page_annots_with};
pub use buffer::Buffer;
pub use cmap::CMap;
pub use doc_stream::decode_stream;
pub use draw_device::{
    DrawDevice, cmyk_to_rgb, gray_to_rgb, rasterize_page_ex, rasterize_page_native,
};
pub use form::{
    FieldKind, FormField, has_acroform, page_form_fields, set_field_value, toggle_checkbox,
};
pub use write::{NewObject, incremental_update, next_object_number, write_object};
// The crate's `rasterize_page` is the graceful (hayro-fallback-aware) one --
// see hayro_fallback.rs's module docs for why the plain-kopitiam-engine
// render (`rasterize_page_native`) isn't the default name. Every existing
// caller across the workspace already calls `mupdf::rasterize_page`, so this
// re-export is what makes the fallback apply everywhere with no call-site
// changes.
pub use draw_edge::{FillRule, fill_polygons};
pub use draw_path::Path;
pub use encodings::BaseEncoding;
pub use error::{Error, ErrorKind, Result};
pub use font::{Decoded, Font};
pub use geometry::{IRect, Matrix, Point, Quad, Rect};
pub use hash::HashTable;
pub use hayro_fallback::rasterize_page_graceful as rasterize_page;
pub use hayro_fallback::rasterize_page_with_fallback;
pub use interpret::Processor;
pub use lex::{Token, lex};
pub use object::Object;
pub use page_image::{DecodedImage, page_full_image, page_images};
pub use page_run::{run_page, run_page_dict};
pub use parse::{IndirectObject, StreamRange, parse_array, parse_dict, parse_ind_obj};
pub use pixmap::Pixmap;
pub use pool::{Handle, Pool};
pub use stext_boxer::{page_to_stext_segmented, segment_stext_page};
pub use stext_device::{StextDevice, page_to_stext};
pub use stext_iterator::{block_bbox, blocks_dfs, line_bbox};
pub use stext_para::paragraph_break;
pub use stream::{Stream, StreamSource, Whence};
pub use structured_text::{
    StextBlock, StextChar, StextImageBlock, StextLine, StextOptions, StextPage, StextTextBlock,
};
pub use text_device::TextDevice;
pub use xref::PdfDocument;
