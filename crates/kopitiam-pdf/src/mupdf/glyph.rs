//! Embedded-font **glyph-outline** decoding: the pure-Rust replacement for the
//! FreeType outline extraction MuPDF performs in `source/fitz/font.c`
//! (`fz_outline_ft_glyph`, and the `FT_Outline_Decompose` callbacks `move_to` /
//! `line_to` / `conic_to` / `cubic_to`, font.c:1418-1545) (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the *outline-to-[`Path`] callback shape*
//! follows MuPDF; the *font-program parsing* MuPDF delegates to FreeType is
//! re-implemented here from the OpenType (`glyf`) and Adobe CFF / Type2
//! charstring specifications (the same formats FreeType decodes), because the
//! port avoids FreeType entirely (see [`super::font`]). See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this module gives the rasterizer
//!
//! [`Font`](super::font) previously answered only *code -> (Unicode, advance)*
//! and the draw device painted a filled advance box per glyph. This module adds
//! the missing piece: **the vector outline of a glyph**, so
//! [`show_glyph`](super::draw_device) can fill real letterforms.
//!
//! A [`FontProgram`] is parsed once from the descriptor's embedded font file
//! (`/FontFile2` = sfnt TrueType, `/FontFile3` = CFF / OpenType) and cached on the
//! [`Font`]. [`FontProgram::outline`] turns a **glyph index** (GID) into a
//! device-independent [`Path`] in **em space** (1 em = 1.0 unit, y-up, matching
//! the font's own glyph space after the units-per-em / FontMatrix scale). The
//! caller ([`show_glyph`](super::draw_device)) applies the glyph's
//! text-rendering matrix + CTM, exactly as it already does for the advance box,
//! and fills the result.
//!
//! ## Coverage vs fallback (precision over completeness)
//!
//! * **TrueType `glyf`** (`/FontFile2`, and OpenType-`glyf` `/FontFile3`):
//!   simple + composite glyphs, decoded here ([`glyph_truetype`](super::glyph_truetype)).
//! * **CFF / Type2 charstrings** (`/FontFile3` `Type1C` / `CIDFontType0C` /
//!   OpenType-`CFF `): a Type2 charstring interpreter
//!   ([`glyph_cff`](super::glyph_cff)), including the predefined-Standard-encoding
//!   `code -> name -> gid` fallback for simple CFF fonts.
//! * **Type1 (`/FontFile`)**: a Type1 charstring interpreter
//!   ([`glyph_type1`](super::glyph_type1)) -- selected **by glyph name**, not
//!   GID, so it is not part of this module's `gid -> outline` [`FontProgram`]
//!   shape; [`Font::glyph_outline`](super::font::Font) resolves the name and
//!   calls [`Type1Program::outline`](super::glyph_type1::Type1Program::outline)
//!   directly.
//! * Any font program that still fails to parse: **no** outline is produced;
//!   the draw device keeps its advance-box fallback. Never a panic, never a
//!   corrupted pixmap.

use super::draw_path::Path;
use super::geometry::Matrix;
use super::glyph_cff::CffProgram;
use super::glyph_truetype::TrueTypeProgram;

/// A parsed embedded font program able to produce glyph outlines.
///
/// Built by [`FontProgram::parse`] from the raw (already filter-decoded) bytes of
/// a `/FontFile2` or `/FontFile3` stream, then cached on the owning
/// [`Font`](super::font) (behind an `Arc`, shared across the font's clones).
#[derive(Clone, Debug)]
pub enum FontProgram {
    /// A TrueType (`glyf`) outline source.
    TrueType(TrueTypeProgram),
    /// A CFF / Type2 charstring outline source. Boxed: `CffProgram` (charset /
    /// encoding / FDArray tables) is much larger than `TrueTypeProgram`, and an
    /// unboxed variant would size every `FontProgram` -- including the common
    /// TrueType case -- to the larger one.
    Cff(Box<CffProgram>),
}

impl FontProgram {
    /// Parse a `/FontFile2` or `/FontFile3` embedded font program, sniffing the
    /// format from its leading bytes. Returns `None` for anything not decodable
    /// here (corrupt or unsupported programs) so the caller falls back to the
    /// advance box.
    ///
    /// **Not Type1**: `/FontFile` (Type1, `%!` PostScript / PFB `0x80`) is a
    /// different, name-keyed shape with no numeric GID (a `CharStrings` dict,
    /// not an array) and is decoded through
    /// [`Type1Program`](super::glyph_type1::Type1Program) directly by
    /// [`Font::load`](super::font::Font::load) — never through this function or
    /// wrapped in [`FontProgram`]. This function still rejects a `%!`/`0x80`
    /// tag defensively (returning `None`) in case it's ever called on one, but
    /// that path is not exercised in practice.
    ///
    /// * sfnt magic (`0x00010000`, `true`, `ttcf`) -> TrueType `glyf` (or the
    ///   `CFF ` table inside an OpenType wrapper).
    /// * `OTTO` -> OpenType with a `CFF ` table.
    /// * otherwise -> a bare CFF (`/FontFile3` `Type1C` / `CIDFontType0C`).
    pub fn parse(bytes: &[u8]) -> Option<FontProgram> {
        if bytes.len() < 4 {
            return None;
        }
        let tag = &bytes[0..4];
        match tag {
            // sfnt wrappers: could carry either `glyf` or a `CFF ` table.
            b"\x00\x01\x00\x00" | b"true" | b"ttcf" | b"OTTO" => {
                if let Some(cff) = TrueTypeProgram::cff_table(bytes) {
                    return CffProgram::parse(&cff).map(|c| FontProgram::Cff(Box::new(c)));
                }
                TrueTypeProgram::parse(bytes).map(FontProgram::TrueType)
            }
            // A bare CFF font program (typ. version 1.0 -> first byte 0x01), or a
            // Type1/PostScript program (handled elsewhere -- see above).
            _ => {
                if tag[0] == b'%' || tag[0] == 0x80 {
                    return None; // Type1 / PFB: not this function's job.
                }
                CffProgram::parse(bytes).map(|c| FontProgram::Cff(Box::new(c)))
            }
        }
    }

    /// The outline of glyph `gid` in **em space** (y-up, 1 em = 1.0), or `None`
    /// when the glyph is empty / out of range / undecodable. The caller applies
    /// the text-rendering matrix + CTM.
    pub fn outline(&self, gid: u16) -> Option<Path> {
        match self {
            FontProgram::TrueType(t) => t.outline(gid),
            FontProgram::Cff(c) => c.outline(gid),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers used by both decoders
// ---------------------------------------------------------------------------

/// A minimal big-endian byte-slice reader with bounds-checked accessors. All
/// reads past the end return `0` / `None` rather than panicking (a malformed
/// embedded font must never crash the rasterizer).
#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn at(data: &'a [u8], pos: usize) -> Reader<'a> {
        Reader { data, pos }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// The backing byte slice (for direct offset indexing, e.g. the cmap
    /// format-4 glyphIdArray).
    pub(crate) fn data(&self) -> &'a [u8] {
        self.data
    }

    pub(crate) fn u8(&mut self) -> u8 {
        let v = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        v
    }

    pub(crate) fn u16(&mut self) -> u16 {
        let hi = self.u8() as u16;
        let lo = self.u8() as u16;
        (hi << 8) | lo
    }

    pub(crate) fn i16(&mut self) -> i16 {
        self.u16() as i16
    }

    pub(crate) fn u32(&mut self) -> u32 {
        let hi = self.u16() as u32;
        let lo = self.u16() as u32;
        (hi << 16) | lo
    }

    /// Read a big-endian unsigned integer of `size` bytes (1..=4), as used by the
    /// CFF INDEX offset arrays.
    pub(crate) fn offset(&mut self, size: u8) -> u32 {
        let mut v = 0u32;
        for _ in 0..size {
            v = (v << 8) | self.u8() as u32;
        }
        v
    }
}

// MuPDF: fz_quadto (path.c) -- FreeType's conic_to (font.c:1450) converts a
// TrueType quadratic to a cubic before handing it to fz_curveto. Same elevation
// here, since [`Path`] stores cubics only.
/// Append a quadratic bezier (control `c`, endpoint `p`) to `path`, elevating it
/// to the equivalent cubic. `from` is the current point.
pub(crate) fn quad_to(path: &mut Path, from: (f32, f32), c: (f32, f32), p: (f32, f32)) {
    // Cubic controls of the degree-elevated quadratic:
    // c1 = from + 2/3 (c - from), c2 = p + 2/3 (c - p).
    let c1x = from.0 + 2.0 / 3.0 * (c.0 - from.0);
    let c1y = from.1 + 2.0 / 3.0 * (c.1 - from.1);
    let c2x = p.0 + 2.0 / 3.0 * (c.0 - p.0);
    let c2y = p.1 + 2.0 / 3.0 * (c.1 - p.1);
    path.curve_to(c1x, c1y, c2x, c2y, p.0, p.1);
}

/// A 2x3 affine used inside the decoders (glyph-unit space -> em space, and
/// composite-component transforms). Kept separate from [`Matrix`] only so the
/// decoders can accumulate integer-ish component transforms without importing the
/// device matrix conventions; converts to [`Matrix`] on demand.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Affine {
    pub(crate) fn scale(s: f32) -> Affine {
        Affine { a: s, b: 0.0, c: 0.0, d: s, e: 0.0, f: 0.0 }
    }

    pub(crate) fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)
    }

    /// `self` applied first, then `other` (matching [`Matrix::concat`] order).
    pub(crate) fn concat(&self, other: &Affine) -> Affine {
        Affine {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            e: self.e * other.a + self.f * other.c + other.e,
            f: self.e * other.b + self.f * other.d + other.f,
        }
    }
}

impl From<Affine> for Matrix {
    fn from(a: Affine) -> Matrix {
        Matrix::new(a.a, a.b, a.c, a.d, a.e, a.f)
    }
}

/// True if `path` has no drawable geometry (an empty / whitespace glyph, or a
/// decode that produced nothing) — reported as `None` by the outline decoders so
/// the caller skips or falls back cleanly.
pub(crate) fn path_is_empty(path: &Path) -> bool {
    path.flatten(Matrix::IDENTITY).is_empty()
}
