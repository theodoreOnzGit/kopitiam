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
//!
//! ## Rescue order: skrifa first, our decoders second, hayro last (gh-91)
//!
//! [`FontProgram`] additionally carries an optional
//! [`SkrifaProgram`](super::glyph_skrifa::SkrifaProgram), parsed from the same
//! bytes, and [`FontProgram::outline`] asks **skrifa first**, falling to the
//! from-spec decoder above only when skrifa has nothing, and to the advance box
//! (which is what hands the page to hayro) only when neither does:
//!
//! ```text
//!   skrifa  ->  ours  ->  advance box (=> hayro)
//! ```
//!
//! **This order was deliberately inverted** — gh-90 shipped ours-first, gh-91
//! flipped it on the maintainer's instruction — so the reasoning is worth
//! keeping, not just the outcome. Ours-first was the conservative call: it
//! cannot change any glyph that already renders, so skrifa could only ever
//! *fill a gap* (the documented gh-67 ceilings: predefined-Expert CFF encoding,
//! CID-keyed CFF edge cases, the `seac` operator). Skrifa-first is the
//! *most-correct-engine-first* call instead: skrifa (Google's `fontations`) is
//! a far more battle-tested OpenType implementation than decoders we wrote from
//! the spec, so where the two disagree about a real glyph, skrifa is the better
//! bet. The cost is that it changes the rendering of glyphs that already
//! worked, which is why gh-91 carried a pixel-diff verification bar rather than
//! just a green test suite.
//!
//! What is *not* negotiable in that chain: hayro stays last, and Type1
//! (`/FontFile`) never enters it at all. See
//! [`glyph_skrifa`](super::glyph_skrifa) for why skrifa is scoped to
//! `/FontFile2` / `/FontFile3` (OpenType `glyf` / `CFF`), never `/FontFile`.
//!
//! A note on why [`FontProgram`] still looks like a plain two-variant enum
//! rather than the more obvious `{ primary, skrifa }` struct: `font.rs`'s
//! `select_gid` (GID resolution -- a different concern from outline decoding,
//! entirely about the font's own encoding/charset tables) pattern-matches on
//! `FontProgram::TrueType(ttf)` / `FontProgram::Cff(cff)` directly and calls
//! primary-decoder methods (`gid_for_code`, `is_cid_keyed`, `gid_for_cid`) on
//! the binding. Reshaping the enum would require touching that match arm too.
//! Instead, [`TrueTypeWithSkrifa`] / [`CffWithSkrifa`] wrap the primary decoder
//! plus an optional skrifa program and `Deref` to the primary type, so
//! `select_gid`'s existing code keeps compiling and calling primary-decoder
//! methods exactly as it does today, while [`FontProgram::outline`] (defined
//! here) has the additional skrifa data it needs for the per-glyph fallback.

use super::draw_path::Path;
use super::geometry::Matrix;
use super::glyph_cff::CffProgram;
use super::glyph_skrifa::SkrifaProgram;
use super::glyph_truetype::TrueTypeProgram;

/// A [`TrueTypeProgram`] plus an optional skrifa second opinion parsed from the
/// same bytes (see [`super::glyph_skrifa`]). `Deref`s to [`TrueTypeProgram`] so
/// that code written against the primary decoder's API (e.g.
/// [`select_gid`](super::font::Font)'s GID resolution, which is a different
/// concern from outline decoding and has no need to know skrifa exists) keeps
/// compiling and behaving unchanged.
#[derive(Clone, Debug)]
pub struct TrueTypeWithSkrifa {
    primary: TrueTypeProgram,
    skrifa: Option<SkrifaProgram>,
}

impl std::ops::Deref for TrueTypeWithSkrifa {
    type Target = TrueTypeProgram;
    fn deref(&self) -> &TrueTypeProgram {
        &self.primary
    }
}

/// A [`CffProgram`] plus an optional skrifa second opinion parsed from the same
/// bytes. See [`TrueTypeWithSkrifa`] for why this wraps rather than replaces the
/// primary decoder.
#[derive(Clone, Debug)]
pub struct CffWithSkrifa {
    primary: CffProgram,
    skrifa: Option<SkrifaProgram>,
}

impl std::ops::Deref for CffWithSkrifa {
    type Target = CffProgram;
    fn deref(&self) -> &CffProgram {
        &self.primary
    }
}

/// A parsed embedded font program able to produce glyph outlines.
///
/// Built by [`FontProgram::parse`] from the raw (already filter-decoded) bytes of
/// a `/FontFile2` or `/FontFile3` stream, then cached on the owning
/// [`Font`](super::font) (behind an `Arc`, shared across the font's clones).
#[derive(Clone, Debug)]
pub enum FontProgram {
    /// A TrueType (`glyf`) outline source, plus an optional skrifa second
    /// opinion for glyphs the primary decoder can't handle.
    TrueType(TrueTypeWithSkrifa),
    /// A CFF / Type2 charstring outline source, plus an optional skrifa second
    /// opinion. Boxed: `CffWithSkrifa` (charset / encoding / FDArray tables)
    /// is much larger than `TrueTypeWithSkrifa`, and an unboxed variant would
    /// size every `FontProgram` -- including the common TrueType case -- to
    /// the larger one.
    Cff(Box<CffWithSkrifa>),
    /// **skrifa alone**: a program neither from-spec decoder could parse at
    /// all, that skrifa nevertheless reads. There is no primary decoder behind
    /// this variant, so both the outline *and* the GID resolution come from
    /// skrifa (see [`select_gid`](super::font::Font) and
    /// [`SkrifaProgram::gid_for_code`](super::glyph_skrifa::SkrifaProgram::gid_for_code)).
    ///
    /// Reaching this variant is **strictly better than the alternative**, and
    /// that is the whole argument for it: before gh-91 a font our decoders
    /// rejected produced no [`FontProgram`] at all, so every one of its glyphs
    /// painted an advance box and the page went to hayro. Nothing that renders
    /// today can regress into this variant — it is only ever constructed where
    /// [`FontProgram::parse`] used to return `None`.
    Skrifa(SkrifaProgram),
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
    /// that path is not exercised in practice. skrifa has no Type1 support
    /// either (see [`super::glyph_skrifa`]), so this rejection also keeps
    /// Type1 bytes away from the skrifa second opinion.
    ///
    /// * sfnt magic (`0x00010000`, `true`, `ttcf`) -> TrueType `glyf` (or the
    ///   `CFF ` table inside an OpenType wrapper).
    /// * `OTTO` -> OpenType with a `CFF ` table.
    /// * otherwise -> a bare CFF (`/FontFile3` `Type1C` / `CIDFontType0C`).
    ///
    /// **A primary decoder is no longer required** (gh-91). When the from-spec
    /// decoder rejects the program outright but skrifa reads it, this returns
    /// [`FontProgram::Skrifa`] instead of `None`, and GID resolution for that
    /// font comes from skrifa's own `cmap` / CFF `Encoding` / CFF charset (see
    /// [`Font::select_gid`](super::font::Font)). Before gh-91 that case yielded
    /// `None`, which meant an advance box per glyph and the whole page handed
    /// to hayro — so this widening cannot regress a font that already rendered,
    /// it can only rescue one that didn't. `None` now means *neither* engine
    /// could make sense of the bytes.
    pub fn parse(bytes: &[u8]) -> Option<FontProgram> {
        if bytes.len() < 4 {
            return None;
        }
        let tag = &bytes[0..4];
        // Type1 / PFB: not this function's job, and out of skrifa's scope too.
        if tag[0] == b'%' || tag[0] == 0x80 {
            return None;
        }
        let skrifa = SkrifaProgram::parse(bytes);
        // `skrifa_only` is the last resort: it is reached only where every
        // branch below would have returned `None` before gh-91.
        let skrifa_only = || skrifa.clone().map(FontProgram::Skrifa);
        match tag {
            // sfnt wrappers: could carry either `glyf` or a `CFF ` table.
            b"\x00\x01\x00\x00" | b"true" | b"ttcf" | b"OTTO" => {
                if let Some(cff) = TrueTypeProgram::cff_table(bytes) {
                    let Some(primary) = CffProgram::parse(&cff) else {
                        return skrifa_only();
                    };
                    return Some(FontProgram::Cff(Box::new(CffWithSkrifa {
                        primary,
                        skrifa,
                    })));
                }
                let Some(primary) = TrueTypeProgram::parse(bytes) else {
                    return skrifa_only();
                };
                Some(FontProgram::TrueType(TrueTypeWithSkrifa {
                    primary,
                    skrifa,
                }))
            }
            // A bare CFF font program (typ. version 1.0 -> first byte 0x01).
            _ => {
                let Some(primary) = CffProgram::parse(bytes) else {
                    return skrifa_only();
                };
                Some(FontProgram::Cff(Box::new(CffWithSkrifa {
                    primary,
                    skrifa,
                })))
            }
        }
    }

    /// The outline of glyph `gid` in **em space** (y-up, 1 em = 1.0), or `None`
    /// when the glyph is empty / out of range / undecodable by either the
    /// primary decoder or (when present) the skrifa second opinion. The caller
    /// applies the text-rendering matrix + CTM.
    ///
    /// **skrifa first, the from-spec decoder second** (gh-91): skrifa is asked
    /// for every glyph, and the primary decoder only answers the ones skrifa
    /// has nothing for. `None` from both is the advance-box signal, which is
    /// what puts the page on hayro — so the full chain is
    /// `skrifa -> ours -> hayro`, hayro always last.
    ///
    /// The order matters and was flipped on purpose; see this module's
    /// "Rescue order" section for why ours-first came first and why
    /// most-correct-engine-first replaced it.
    pub fn outline(&self, gid: u16) -> Option<Path> {
        match self {
            FontProgram::TrueType(t) => t
                .skrifa
                .as_ref()
                .and_then(|s| s.outline(gid))
                .or_else(|| t.primary.outline(gid)),
            FontProgram::Cff(c) => c
                .skrifa
                .as_ref()
                .and_then(|s| s.outline(gid))
                .or_else(|| c.primary.outline(gid)),
            FontProgram::Skrifa(s) => s.outline(gid),
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
        Affine {
            a: s,
            b: 0.0,
            c: 0.0,
            d: s,
            e: 0.0,
            f: 0.0,
        }
    }

    pub(crate) fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
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

#[cfg(test)]
mod tests {
    use super::super::glyph_skrifa::build_ttf_with_metrics;
    use super::super::glyph_truetype::{box_font, build_ttf, simple_glyf};
    use super::*;

    fn bbox(path: &Path) -> (f32, f32, f32, f32) {
        let polys = path.flatten(Matrix::IDENTITY);
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for poly in &polys {
            for p in poly {
                b = (b.0.min(p.x), b.1.min(p.y), b.2.max(p.x), b.3.max(p.y));
            }
        }
        b
    }

    /// A ring glyph (outer square (100,100)-(900,900), inner hole
    /// (300,300)-(700,700), opposite winding) at gid 1 in a 1000-upm font --
    /// deliberately a different shape/bbox from `glyph_truetype::box_font`'s
    /// gid 1, so tests below can tell "primary's box" from "skrifa's ring"
    /// apart by bounding box alone. Built WITH the `hhea`/`hmtx` skrifa needs
    /// (`glyph_truetype::ring_font` doesn't carry them, since our own
    /// decoder never needs them -- see `glyph_skrifa`'s module docs).
    fn ring_font_with_metrics() -> Vec<u8> {
        let notdef = simple_glyf(&[]);
        let ring = simple_glyf(&[
            vec![
                (100, 100, true),
                (100, 900, true),
                (900, 900, true),
                (900, 100, true),
            ],
            vec![
                (300, 300, true),
                (700, 300, true),
                (700, 700, true),
                (300, 700, true),
            ],
        ]);
        build_ttf_with_metrics(1000, &[notdef, ring])
    }

    /// Proves the ordering that is the whole point of this design: when BOTH
    /// skrifa and the primary decoder can produce an outline for the same GID,
    /// **skrifa's** result wins. Deliberately mismatched inputs (box_font as
    /// primary, a ring as the skrifa side) so a wrong ordering would be caught
    /// by a bounding-box mismatch, not just silently identical output.
    ///
    /// This test is the **record that the order changed on purpose** (gh-91).
    /// It was `outline_prefers_primary_over_skrifa` and asserted the opposite
    /// bounding box; it was inverted and renamed rather than deleted, so a
    /// `git log -S` on either name lands on the decision instead of on a
    /// vanished test.
    #[test]
    fn outline_prefers_skrifa_over_primary() {
        let primary = TrueTypeProgram::parse(&box_font()).expect("parse box font");
        let skrifa =
            SkrifaProgram::parse(&ring_font_with_metrics()).expect("parse ring font via skrifa");
        let program = FontProgram::TrueType(TrueTypeWithSkrifa {
            primary,
            skrifa: Some(skrifa),
        });
        let path = program.outline(1).expect("outline present");
        let (x0, y0, x1, y1) = bbox(&path);
        // box_font gid 1 is (0,0)-(500,700) -> em (0,0)-(0.5,0.7). The ring's
        // gid 1 outer square is (100,100)-(900,900) -> em (0.1,0.1)-(0.9,0.9).
        // These bounding boxes are disjoint enough that using the wrong source
        // cannot pass this assertion by accident.
        assert!(
            (x0 - 0.1).abs() < 1e-3 && (y0 - 0.1).abs() < 1e-3,
            "expected skrifa's ring origin, got {x0},{y0} (looks like the primary's box instead)"
        );
        assert!(
            (x1 - 0.9).abs() < 1e-3 && (y1 - 0.9).abs() < 1e-3,
            "expected skrifa's ring extent, got {x1},{y1} (looks like the primary's box instead)"
        );
    }

    /// The mirror of the test above, and the reason skrifa-first is not simply
    /// "skrifa-only": a glyph **skrifa** has nothing for still falls through to
    /// the from-spec decoder. Same fixtures, roles swapped -- skrifa gets the
    /// notdef-only font (gid 1 out of range for it), the primary gets the ring.
    #[test]
    fn outline_falls_back_to_primary_for_a_glyph_skrifa_lacks() {
        let notdef_only = build_ttf_with_metrics(1000, &[simple_glyf(&[])]);
        let skrifa = SkrifaProgram::parse(&notdef_only).expect("parse notdef-only via skrifa");
        assert!(
            skrifa.outline(1).is_none(),
            "skrifa should have nothing for gid 1"
        );
        let primary =
            TrueTypeProgram::parse(&ring_font_with_metrics()).expect("parse ring font as primary");
        let program = FontProgram::TrueType(TrueTypeWithSkrifa {
            primary,
            skrifa: Some(skrifa),
        });
        let path = program.outline(1).expect("primary should fill the gap");
        let (x0, y0, x1, y1) = bbox(&path);
        assert!(
            (x0 - 0.1).abs() < 1e-3 && (y0 - 0.1).abs() < 1e-3,
            "{x0},{y0}"
        );
        assert!(
            (x1 - 0.9).abs() < 1e-3 && (y1 - 0.9).abs() < 1e-3,
            "{x1},{y1}"
        );
    }

    /// The gh-67 shape: a font whose primary decoder parses fine but has no
    /// outline for a specific GID (out of range here, standing in for "decode
    /// failed"). skrifa is consulted and fills the gap.
    #[test]
    fn outline_falls_back_to_skrifa_for_a_glyph_the_primary_lacks() {
        // Only gid 0 (.notdef): gid 1 is out of range for the primary decoder.
        let notdef_only = build_ttf(1000, &[simple_glyf(&[])], None);
        let primary = TrueTypeProgram::parse(&notdef_only).expect("parse notdef-only font");
        assert!(
            primary.outline(1).is_none(),
            "primary should have nothing for gid 1"
        );
        let skrifa =
            SkrifaProgram::parse(&ring_font_with_metrics()).expect("parse ring font via skrifa");
        let program = FontProgram::TrueType(TrueTypeWithSkrifa {
            primary,
            skrifa: Some(skrifa),
        });
        let path = program.outline(1).expect("skrifa should fill the gap");
        let (x0, y0, x1, y1) = bbox(&path);
        assert!(
            (x0 - 0.1).abs() < 1e-3 && (y0 - 0.1).abs() < 1e-3,
            "{x0},{y0}"
        );
        assert!(
            (x1 - 0.9).abs() < 1e-3 && (y1 - 0.9).abs() < 1e-3,
            "{x1},{y1}"
        );
    }

    /// No skrifa side at all: behaviour must be identical to the pre-skrifa
    /// world (primary-only), never a panic.
    #[test]
    fn outline_without_skrifa_behaves_as_primary_alone() {
        let primary = TrueTypeProgram::parse(&box_font()).expect("parse box font");
        let program = FontProgram::TrueType(TrueTypeWithSkrifa {
            primary,
            skrifa: None,
        });
        assert!(program.outline(1).is_some());
        assert!(program.outline(99).is_none());
    }
}
