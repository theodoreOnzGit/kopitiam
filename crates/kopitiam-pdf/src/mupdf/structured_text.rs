//! Ported from MuPDF `include/mupdf/fitz/structured-text.h` -- the structured
//! text (`stext`) data model: the `fz_stext_page` / `fz_stext_block` /
//! `fz_stext_line` / `fz_stext_char` tree and the `fz_stext_option_flags`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the data shapes and flag values
//! follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # The model
//!
//! A [`StextPage`] is a list of [`StextBlock`]s over a `mediabox`. The block
//! type that matters for text extraction is [`StextBlock::Text`], a
//! [`StextTextBlock`] holding a list of [`StextLine`]s (typically a paragraph);
//! each line is a run of [`StextChar`]s sharing a baseline, with a normalised
//! direction vector and a bounding box. Each char carries its Unicode value,
//! device-space origin, a [`Quad`] (built from the glyph origin + advance and
//! the font's scalar ascender/descender), its size, and a reference (by index
//! into [`StextPage::fonts`]) to the font it was drawn from.
//!
//! MuPDF stores this tree in a bump [`super::pool::Pool`] as intrusive linked
//! lists (`first_block`/`next`, `first_char`/`next`, …). This port uses owned
//! `Vec`s instead -- the ownership is simpler in Rust and the traversal order is
//! identical (document/draw order). The `prev`/`last_*` back-pointers MuPDF
//! keeps only to *build* the page collapse into "push onto the last element".
//!
//! ## Ported here vs. deferred
//!
//! The stext option flags are ported entry-for-entry as constants on
//! [`StextOptions`] so the values match MuPDF's `enum fz_stext_option_flags`.
//! The ones that affect **char/line assembly** are honoured by the device
//! ([`super::stext_device`]): [`StextOptions::PRESERVE_LIGATURES`],
//! [`StextOptions::PRESERVE_WHITESPACE`], [`StextOptions::INHIBIT_SPACES`],
//! [`StextOptions::DEHYPHENATE`], [`StextOptions::PRESERVE_SPANS`]. The rest
//! (segmentation, table hunting, structure collection, styles, accurate
//! bboxes, vectors, clipping, actualtext, CID/GID-for-unknown) are recognised
//! but not acted upon -- see the per-constant docs and the device module.
//!
//! The [`StextBlock::Image`] variant is kept as a documented stub: the
//! interpreter is not on the image path yet, so no image blocks are produced,
//! but the variant exists so a later image wave can extend the model without a
//! breaking change. `Struct`/`Vector`/`Grid` block types are omitted entirely
//! (deferred to the structure/table waves).

use super::font::Font;
use super::geometry::{Point, Quad, Rect};

/// Options controlling structured-text extraction. Wraps MuPDF's
/// `fz_stext_options.flags` bitfield (the `scale`/`clip` fields are deferred).
///
/// Construct with [`StextOptions::default`] (flags `0`, i.e. MuPDF's default:
/// ligatures expanded, whitespace normalised to spaces, missing spaces
/// synthesised) and OR in the constants below.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StextOptions {
    /// The OR of the `FZ_STEXT_*` flag bits (see the associated constants).
    pub flags: u32,
}

impl StextOptions {
    // MuPDF: enum fz_stext_option_flags (structured-text.h:194). Values copied
    // verbatim so a flag set here means the same bit as in MuPDF.

    /// Pass ligatures through unchanged instead of expanding e.g. `ﬃ` -> `f f i`.
    /// **Honoured** by the device. (`FZ_STEXT_PRESERVE_LIGATURES`)
    pub const PRESERVE_LIGATURES: u32 = 1;
    /// Keep original whitespace instead of collapsing every horizontal
    /// whitespace code to `U+0020`. **Honoured** by the device.
    /// (`FZ_STEXT_PRESERVE_WHITESPACE`)
    pub const PRESERVE_WHITESPACE: u32 = 2;
    /// Store images in the page. **Deferred** (no image path yet).
    /// (`FZ_STEXT_PRESERVE_IMAGES`)
    pub const PRESERVE_IMAGES: u32 = 4;
    /// Do **not** synthesise spaces across wide gaps. **Honoured** by the device
    /// (suppresses the gap->space rule). (`FZ_STEXT_INHIBIT_SPACES`)
    pub const INHIBIT_SPACES: u32 = 8;
    /// Record end-of-line hyphens as soft hyphens so lines can be re-joined.
    /// **Honoured** at the line-flag level (the join flag is set); the actual
    /// text flattening/join is a downstream concern. (`FZ_STEXT_DEHYPHENATE`)
    pub const DEHYPHENATE: u32 = 16;
    /// Do not merge spans on the same line: force a new line at each span start.
    /// **Honoured** by the device (via `force_new_line`).
    /// (`FZ_STEXT_PRESERVE_SPANS`)
    pub const PRESERVE_SPANS: u32 = 32;
    /// Drop fully clipped characters. **Deferred** (no clip stack on the text
    /// path). (`FZ_STEXT_CLIP`)
    pub const CLIP: u32 = 64;
    /// Use the CID as the Unicode value on a mapping miss. **Deferred**.
    /// (`FZ_STEXT_USE_CID_FOR_UNKNOWN_UNICODE`)
    pub const USE_CID_FOR_UNKNOWN_UNICODE: u32 = 128;
    /// Collect PDF structure tree markup. **Deferred** (structure wave).
    /// (`FZ_STEXT_COLLECT_STRUCTURE`)
    pub const COLLECT_STRUCTURE: u32 = 256;
    /// Compute char bboxes from glyph outlines (needs FreeType). **Deferred** --
    /// this port is scalar-metrics only. (`FZ_STEXT_ACCURATE_BBOXES`)
    pub const ACCURATE_BBOXES: u32 = 512;
    /// Collect vector-graphics bboxes. **Deferred**. (`FZ_STEXT_COLLECT_VECTORS`)
    pub const COLLECT_VECTORS: u32 = 1024;
    /// Do not apply `/ActualText` replacements. **Deferred** (no actualtext).
    /// (`FZ_STEXT_IGNORE_ACTUALTEXT`)
    pub const IGNORE_ACTUALTEXT: u32 = 2048;
    /// Segment the page into regions. **Deferred** (layout/`boxer` wave).
    /// (`FZ_STEXT_SEGMENT`)
    pub const SEGMENT: u32 = 4096;
    /// Break blocks at paragraph boundaries. **Deferred** (layout wave).
    /// (`FZ_STEXT_PARAGRAPH_BREAK`)
    pub const PARAGRAPH_BREAK: u32 = 8192;
    /// Hunt for tables. **Deferred** (table wave). (`FZ_STEXT_TABLE_HUNT`)
    pub const TABLE_HUNT: u32 = 16384;
    /// Detect text features (fake bold, underline, …). **Deferred**.
    /// (`FZ_STEXT_COLLECT_STYLES`)
    pub const COLLECT_STYLES: u32 = 32768;
    /// Use the GID as the Unicode value on a mapping miss. **Deferred**.
    /// (`FZ_STEXT_USE_GID_FOR_UNKNOWN_UNICODE`)
    pub const USE_GID_FOR_UNKNOWN_UNICODE: u32 = 65536;

    /// True if every bit in `flag` is set.
    #[inline]
    pub fn has(self, flag: u32) -> bool {
        self.flags & flag == flag
    }
}

// MuPDF: enum fz_stext_char_flags (structured-text.h:489). Only the bits this
// port can currently set are carried on [`StextChar::flags`]; the rest are
// deferred with the features that produce them (styles, clipping, CID/GID).

/// `FZ_STEXT_SYNTHETIC` -- the character was inserted by the extractor (a
/// synthesised inter-word space), not drawn by the content stream.
pub const FZ_STEXT_SYNTHETIC: u16 = 4;
/// `FZ_STEXT_SYNTHETIC_LARGE` -- a synthesised space spanning a large gap
/// (MuPDF sets this when the gap exceeds twice the space threshold).
pub const FZ_STEXT_SYNTHETIC_LARGE: u16 = 512;

// MuPDF: fz_stext_line_flags (structured-text.h:454).
/// `FZ_STEXT_LINE_FLAGS_JOINED` -- this line ends in a (soft) hyphen and, under
/// [`StextOptions::DEHYPHENATE`], should be joined to the next.
pub const FZ_STEXT_LINE_FLAGS_JOINED: u8 = 1;

/// A single extracted character: a Unicode value, the style it appears in, and
/// where it sits. Ports `struct fz_stext_char`.
#[derive(Clone, Copy, Debug)]
pub struct StextChar {
    // MuPDF: fz_stext_char.c
    /// The Unicode character value.
    pub c: char,
    // MuPDF: fz_stext_char.origin
    /// The glyph origin (usually the bottom-left) in device space.
    pub origin: Point,
    // MuPDF: fz_stext_char.quad
    /// The glyph's quad: origin/advance horizontally, ascender/descender
    /// vertically (scalar, non-accurate metrics).
    pub quad: Quad,
    // MuPDF: fz_stext_char.size
    /// The font size (matrix expansion of the text-rendering matrix).
    pub size: f32,
    // MuPDF: fz_stext_char.font -- here an index into [`StextPage::fonts`].
    /// Index of the drawing font in [`StextPage::fonts`].
    pub font: usize,
    // MuPDF: fz_stext_char.flags -- subset (see the FZ_STEXT_* char constants).
    /// Char flags (currently only the synthetic-space bits are ever set).
    pub flags: u16,
    /// The code's CID (glyph identifier within the font); `0` for a synthesised
    /// space. Not present in MuPDF's `fz_stext_char` (it stores the resolved
    /// glyph); kept here because the port has no glyph ids.
    pub cid: u32,
    /// Writing mode: 0 = horizontal, 1 = vertical.
    pub wmode: u8,
}

/// A line of text: characters sharing a common baseline. Ports
/// `struct fz_stext_line`.
#[derive(Clone, Debug)]
pub struct StextLine {
    // MuPDF: fz_stext_line.wmode
    /// Writing mode: 0 = horizontal, 1 = vertical.
    pub wmode: u8,
    // MuPDF: fz_stext_line.flags
    /// Line flags (e.g. [`FZ_STEXT_LINE_FLAGS_JOINED`]).
    pub flags: u8,
    // MuPDF: fz_stext_line.dir
    /// Normalised direction of the baseline.
    pub dir: Point,
    // MuPDF: fz_stext_line.bbox -- union of the char quads (set on finalise).
    /// Bounding box: the union of every char quad's bounding rect.
    pub bbox: Rect,
    // MuPDF: fz_stext_line.first_char/next
    /// The characters, in draw order.
    pub chars: Vec<StextChar>,
}

impl StextLine {
    /// The line's text as a `String` (concatenated char values).
    pub fn text(&self) -> String {
        self.chars.iter().map(|ch| ch.c).collect()
    }
}

/// A text block: a list of lines (typically a paragraph). Ports the
/// `FZ_STEXT_BLOCK_TEXT` arm of `struct fz_stext_block` (`u.t`).
#[derive(Clone, Debug)]
pub struct StextTextBlock {
    // MuPDF: fz_stext_block.bbox -- union of the line bboxes (set on finalise).
    /// Bounding box: the union of the contained line bboxes.
    pub bbox: Rect,
    // MuPDF: fz_stext_block.u.t.first_line/next
    /// The lines, in draw order.
    pub lines: Vec<StextLine>,
}

/// An image block. Ports the `FZ_STEXT_BLOCK_IMAGE` arm (`u.i`). **Stub**: the
/// interpreter does not yet walk the image path, so no image blocks are
/// produced; the variant exists for a later image wave.
#[derive(Clone, Copy, Debug)]
pub struct StextImageBlock {
    // MuPDF: fz_stext_block.bbox
    /// Bounding box of the image on the page.
    pub bbox: Rect,
    // MuPDF: fz_stext_block.u.i.transform
    /// The image-space -> device-space transform.
    pub transform: super::geometry::Matrix,
}

/// A page block. Ports `struct fz_stext_block`'s type union, restricted to the
/// two variants this wave models (`Struct`/`Vector`/`Grid` are deferred).
#[derive(Clone, Debug)]
pub enum StextBlock {
    // MuPDF: FZ_STEXT_BLOCK_TEXT (structured-text.h:373)
    /// A run of text lines.
    Text(StextTextBlock),
    // MuPDF: FZ_STEXT_BLOCK_IMAGE (structured-text.h:374) -- stub, see above.
    /// An image (never produced yet; see [`StextImageBlock`]).
    Image(StextImageBlock),
}

/// A structured-text page: blocks over a mediabox. Ports `struct fz_stext_page`
/// (the `refs`/`pool`/`id_list` bookkeeping fields are not needed in Rust).
#[derive(Clone, Debug)]
pub struct StextPage {
    // MuPDF: fz_stext_page.mediabox
    /// The page mediabox (device space).
    pub mediabox: Rect,
    // MuPDF: fz_stext_page.first_block/next
    /// The page's blocks, in draw order.
    pub blocks: Vec<StextBlock>,
    /// The distinct fonts referenced by [`StextChar::font`]. MuPDF stores an
    /// `fz_font*` per char; this port interns fonts here and stores an index,
    /// so the (potentially large) `Font` is not cloned per glyph.
    pub fonts: Vec<Font>,
}

impl StextPage {
    /// The page's full text, blocks and lines joined by `'\n'` (a plain-text
    /// dump in draw order). A convenience for callers and tests -- MuPDF's
    /// `fz_print_stext_page_as_text` does the equivalent traversal.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for block in &self.blocks {
            if let StextBlock::Text(tb) = block {
                for line in &tb.lines {
                    out.push_str(&line.text());
                    out.push('\n');
                }
            }
        }
        out
    }
}
