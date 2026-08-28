//! A **second opinion** glyph-outline source, backed by the
//! [`skrifa`](https://docs.rs/skrifa) crate (part of Google's `fontations`
//! project, Apache-2.0 / MIT dual-licensed -- see
//! docs/ACKNOWLEDGEMENTS.md), consulted only when the crate's own
//! [`glyph_truetype`](super::glyph_truetype) / [`glyph_cff`](super::glyph_cff)
//! decoders fail to produce an outline for a *specific* glyph.
//!
//! # Why this exists
//!
//! [`glyph_truetype`](super::glyph_truetype) and [`glyph_cff`](super::glyph_cff)
//! are from-spec decoders written to avoid a FreeType dependency (see
//! [`super::glyph`]'s module docs). They cover the overwhelming majority of
//! embedded fonts correctly, but a handful of ceilings are documented and known
//! (gh-67): predefined-Expert CFF encoding, CID-keyed CFF edge cases, and the
//! `seac` accented-composite charstring operator. In every one of those cases
//! the *font* parses fine -- [`FontProgram::parse`](super::glyph::FontProgram::parse)
//! succeeds -- but *one particular glyph*'s outline decode fails, and
//! [`FontProgram::outline`](super::glyph::FontProgram::outline) returns `None`,
//! which today means the draw device paints a filled advance box instead of the
//! real letterform.
//!
//! [`SkrifaProgram`] is parsed from the exact same font-program bytes and kept
//! alongside the primary decoder (see the `TrueTypeWithSkrifa` / `CffWithSkrifa`
//! wrappers in [`super::glyph`]). [`FontProgram::outline`](super::glyph::FontProgram::outline)
//! tries the primary decoder first, and only asks [`SkrifaProgram::outline`] when
//! that returns `None` -- so every glyph the primary decoders already handle
//! renders byte-for-byte as before; skrifa only ever fills a gap.
//!
//! # Scope: OpenType only, not Type1
//!
//! skrifa reads OpenType outline formats: `glyf` (TrueType) and `CFF`/`CFF2`
//! (PostScript charstrings inside an OpenType wrapper). That covers
//! `/FontFile2` and `/FontFile3`. **`/FontFile` (Type 1, `%!` PostScript or PFB
//! `0x80`) is out of scope** -- skrifa has no Type1 charstring interpreter, and
//! Type1 selection is by glyph *name* rather than GID (see
//! [`glyph_type1`](super::glyph_type1)), a different shape entirely. This module
//! defensively rejects a `%!`/`0x80` leading byte for the same reason
//! [`FontProgram::parse`](super::glyph::FontProgram::parse) does, even though in
//! practice `Font::load` never routes Type1 bytes here.
//!
//! # The bare-CFF wrapping problem
//!
//! skrifa's [`FontRef::new`] requires an OpenType **table directory** (an sfnt
//! header) to locate tables by tag. A `/FontFile2` TrueType program and an
//! OpenType-wrapped `/FontFile3` (`OTTO` magic) already have one and are handed
//! to skrifa unchanged. But the *common* `/FontFile3` case -- `Type1C` /
//! `CIDFontType0C`, a **bare** CFF byte stream with no sfnt wrapper at all -- has
//! no table directory for skrifa to walk. [`wrap_bare_cff`] builds a minimal
//! synthetic sfnt (`OTTO` + synthetic `head`/`hhea`/`hmtx` + the CFF bytes as the
//! `CFF ` table) around it purely so skrifa has something to parse; nothing
//! about the original CFF data is modified.
//!
//! ## Why `hhea`/`hmtx` too, not just `head`
//!
//! This one cost real debugging time, so it's recorded here: skrifa's outline
//! machinery unconditionally requires `hmtx` for **both** `glyf` and `CFF`
//! outlines, not only `head` -- `GlyphHMetrics::new` (`skrifa`
//! `src/outline/metrics.rs`) calls `font.hmtx().ok()?` with no fallback, and
//! both `glyf::Outlines::new` and `cff::Outlines::from_cff` call it before
//! they'll hand back an outline collection at all. A bare CFF program never
//! carries `hmtx`/`hhea` -- they don't exist in the PostScript/CFF world -- so
//! without synthesizing them, `wrap_bare_cff`'s output would parse fine (the
//! table directory and `head` are readable) but [`SkrifaProgram::outline`]
//! would silently return `None` for *every* glyph, forever, since
//! `font.outline_glyphs()` degrades to an empty collection whenever
//! `GlyphHMetrics::new` fails. `synthetic_hhea_hmtx` supplies the smallest
//! input that satisfies the read (`numberOfHMetrics = 1`, one all-zero
//! `(advance, lsb)` record); the actual numbers are never used for anything
//! this module cares about (only outlines, never advance widths, are
//! extracted), and `Hmtx::advance`/`side_bearing` clamp any gid beyond the
//! supplied record to the last one rather than erroring.
//!
//! An already sfnt-wrapped program (TrueType `/FontFile2`, or an
//! OpenType-CFF `/FontFile3`) is expected to carry a real `hhea`/`hmtx` --
//! every practical embedded font does, since any real rendering pipeline
//! needs them too -- so that case is handed to skrifa unchanged, no
//! synthesis performed.
//!
//! One more wrinkle worth recording since it cost real debugging time: for
//! **`glyf` (TrueType)** outlines specifically, the `lsb` half of an `hmtx`
//! record isn't inert. skrifa's unhinted scaler computes a "left phantom
//! point" as `xMin - lsb` (`skrifa` `src/outline/glyf/mod.rs`,
//! `setup_phantom_points`, following FreeType) and every x coordinate it
//! emits is relative to that point -- so a wrong `lsb` silently *shifts the
//! whole glyph horizontally* by `xMin - lsb` font units, with no error and no
//! `None`. `wrap_bare_cff`'s placeholder `lsb = 0` is fine specifically
//! because **CFF/PostScript outlines have no phantom-point concept at all**
//! (there is no TrueType hinting model to feed); this only bites the `glyf`
//! path. It doesn't bite real embedded `/FontFile2` fonts either, since they
//! carry their own already-self-consistent `hmtx` (`lsb` authored to match
//! each glyph's real `xMin`). It bit this module's *own test suite*, whose
//! synthetic `glyf` fixtures need a matching per-glyph `lsb` for exactly this
//! reason -- see `build_ttf_with_metrics`'s doc comment.
//!
//! ## Why the synthetic `head.unitsPerEm` value doesn't need to be "correct"
//!
//! `skrifa`'s CFF outline builder reads `units_per_em` from `head`
//! unconditionally, even for a pure PostScript-outline font with its own
//! `FontMatrix`: `let units_per_em = font.head().ok()?.units_per_em();` --
//! (`skrifa` `src/outline/cff/mod.rs`, `Outlines::new`). We declare
//! [`SYNTHETIC_UNITS_PER_EM`] (1000, the CFF-spec default 0.001 `FontMatrix`)
//! regardless of the font's actual `FontMatrix`. This is safe, not merely
//! convenient: skrifa's own `Subfont::new` (`src/outline/cff/mod.rs`, around the
//! "adjust our scale factor" comment) compares the CFF's own normalized
//! `FontMatrix` scale against the declared `units_per_em` and, if they differ,
//! **forces a corrective scale** so that unscaled output always comes out in
//! units of the *declared* `units_per_em` -- regardless of what the font's
//! `FontMatrix` actually says. So any nonzero value we declare is self-consistent
//! as long as we divide by that same value afterwards, which is exactly what
//! [`SkrifaProgram::outline`] does.
//!
//! # Coordinate space
//!
//! [`SkrifaProgram::outline`] returns a [`Path`] in the same **em space** as the
//! rest of this crate's decoders: y-up, 1 em = 1.0. skrifa's own "unscaled" draw
//! mode ([`Size::unscaled`]) yields raw font design units (`head.unitsPerEm` per
//! em), so [`PathPen`] divides every coordinate by `units_per_em` on the way into
//! the [`Path`].
//!
//! skrifa's [`OutlinePen`] hands us TrueType-style quadratic segments
//! (`quad_to`) as well as PostScript cubic segments (`curve_to`); [`Path`] only
//! stores cubics, so `quad_to` is degree-elevated to the equivalent cubic via
//! [`super::glyph::quad_to`] -- the exact same elevation
//! [`glyph_truetype`](super::glyph_truetype) already uses for its own `glyf`
//! decoding, so both paths produce identical curve geometry for the same input.

use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::ps::cff::CffFontRef;
use skrifa::raw::ps::string::Sid;
use skrifa::raw::TableProvider;
use skrifa::{FontRef, GlyphId, MetadataProvider};

use super::draw_path::Path;

/// The `units_per_em` we declare in the synthetic `head` table built for a bare
/// (unwrapped) CFF program -- see the module docs' "bare-CFF wrapping problem"
/// section for why the exact value doesn't matter as long as it's nonzero and
/// consistent with the division in [`SkrifaProgram::outline`].
const SYNTHETIC_UNITS_PER_EM: u16 = 1000;

/// True when `bytes` begins with a recognized sfnt/OpenType wrapper magic
/// (`0x00010000`, `true`, `ttcf`, `OTTO`) -- i.e. it already carries a table
/// directory skrifa can walk directly. Mirrors the tag sniff in
/// [`FontProgram::parse`](super::glyph::FontProgram::parse); kept as an
/// independent check here (rather than a shared helper) since the two callers
/// have different fallback behaviour on a "no" (our decoders fail outright vs.
/// this module still has the bare-CFF wrapping path to try).
fn is_sfnt_wrapped(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && matches!(
            &bytes[0..4],
            b"\x00\x01\x00\x00" | b"true" | b"ttcf" | b"OTTO"
        )
}

/// Append one sfnt table record's worth of bytes (`tag` + zero checksum +
/// `offset` + `length`) to `out`. Shared by [`wrap_bare_cff`]'s table
/// directory.
fn push_table_record(out: &mut Vec<u8>, tag: &[u8; 4], offset: u32, length: u32) {
    out.extend_from_slice(tag);
    out.extend_from_slice(&0u32.to_be_bytes()); // checksum: unused, never verified by read-fonts
    out.extend_from_slice(&offset.to_be_bytes());
    out.extend_from_slice(&length.to_be_bytes());
}

/// A minimal 54-byte OpenType `head` table declaring only `unitsPerEm` (at byte
/// offset 18, per the `head` table spec); every other field is left zeroed,
/// since skrifa's outline path reads nothing else from `head`.
fn synthetic_head_table(units_per_em: u16) -> [u8; 54] {
    let mut head = [0u8; 54];
    head[18..20].copy_from_slice(&units_per_em.to_be_bytes());
    head
}

/// A minimal 36-byte `hhea` table declaring `numberOfHMetrics = 1` (the only
/// field skrifa's outline machinery reads from it, via `hmtx`'s parse -- see
/// the module docs' "why hhea/hmtx too" section) and a matching 4-byte `hmtx`
/// table with a single all-zero `(advanceWidth, lsb)` record. Neither number
/// is ever used for anything this module cares about -- only glyph
/// *outlines*, never advance widths, are extracted here -- and `Hmtx`'s own
/// lookups clamp any GID beyond the single supplied record to it rather than
/// erroring, so one record covers every glyph in the font.
fn synthetic_hhea_hmtx() -> ([u8; 36], [u8; 4]) {
    let mut hhea = [0u8; 36];
    hhea[34..36].copy_from_slice(&1u16.to_be_bytes()); // numberOfHMetrics
    let hmtx = [0u8, 0, 0, 0]; // advanceWidth = 0, lsb = 0
    (hhea, hmtx)
}

/// Wrap a bare `/FontFile3` CFF program (`Type1C` / `CIDFontType0C`, no sfnt
/// wrapper) in a minimal synthetic sfnt so skrifa's [`FontRef::new`] -- which
/// requires a table directory to locate tables by tag, and (via
/// `GlyphHMetrics`) an `hmtx`/`hhea` pair before it will build an outline
/// collection at all -- has something to parse. The original CFF bytes are
/// copied byte-for-byte into a `CFF ` table; nothing about them is
/// reinterpreted or modified.
///
/// Returns `None` only if `cff` is empty (nothing to wrap).
fn wrap_bare_cff(cff: &[u8]) -> Option<Vec<u8>> {
    if cff.is_empty() {
        return None;
    }
    let head = synthetic_head_table(SYNTHETIC_UNITS_PER_EM);
    let (hhea, hmtx) = synthetic_hhea_hmtx();
    const NUM_TABLES: u16 = 4;
    let header_len = 12 + (NUM_TABLES as usize) * 16;

    let mut offset = header_len as u32;
    let cff_offset = offset;
    offset += cff.len() as u32;
    let head_offset = offset;
    offset += head.len() as u32;
    let hhea_offset = offset;
    offset += hhea.len() as u32;
    let hmtx_offset = offset;
    offset += hmtx.len() as u32;

    let mut out = Vec::with_capacity(offset as usize);
    out.extend_from_slice(b"OTTO"); // sfntVersion: signals a CFF/OpenType outline font.
    out.extend_from_slice(&NUM_TABLES.to_be_bytes());
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // searchRange/entrySelector/rangeShift: unused by read-fonts.
    // Table records don't strictly need to be sorted by tag -- read-fonts
    // falls back to a linear scan when they aren't (see
    // `FontRef::table_directory_sorted`) -- but "CFF " < "head" < "hhea" <
    // "hmtx" happens to sort correctly anyway, so we get the (irrelevant
    // here) binary-search fast path for free.
    push_table_record(&mut out, b"CFF ", cff_offset, cff.len() as u32);
    push_table_record(&mut out, b"head", head_offset, head.len() as u32);
    push_table_record(&mut out, b"hhea", hhea_offset, hhea.len() as u32);
    push_table_record(&mut out, b"hmtx", hmtx_offset, hmtx.len() as u32);
    out.extend_from_slice(cff);
    out.extend_from_slice(&head);
    out.extend_from_slice(&hhea);
    out.extend_from_slice(&hmtx);
    Some(out)
}

/// A parsed skrifa-backed outline source: the *second opinion* for embedded
/// font glyph outlines described in the module docs. Built once from a font
/// program's raw bytes (the same bytes handed to
/// [`FontProgram::parse`](super::glyph::FontProgram::parse)) and cached
/// alongside the primary decoder.
///
/// Holds owned bytes (rather than a borrowed [`FontRef`]) so it has no
/// lifetime parameter and can be stored in the same `'static`-shaped structures
/// as the rest of [`super::glyph`]'s decoders; [`FontRef::new`] is
/// re-parsed from those bytes on each [`outline`](SkrifaProgram::outline) call.
/// This re-parse only walks the (tiny) table directory, not glyph data, so the
/// cost is a small constant per glyph, not proportional to font size.
#[derive(Clone, Debug)]
pub struct SkrifaProgram {
    /// Bytes in a form skrifa's [`FontRef`] can parse directly: either the
    /// original sfnt-wrapped program, or (for a bare CFF program) the
    /// synthetic wrapper built by [`wrap_bare_cff`].
    bytes: Vec<u8>,
    /// `head.unitsPerEm`, read once at parse time (real value for sfnt-wrapped
    /// input, [`SYNTHETIC_UNITS_PER_EM`] for a wrapped bare CFF program) so
    /// [`outline`](SkrifaProgram::outline) doesn't need to re-read it per glyph.
    units_per_em: u16,
    /// True when this program carries a **CID-keyed** `CFF ` table (its Top
    /// DICT has `ROS`). Read once at parse time because
    /// [`super::font::Font::select_gid`] needs it on *every* glyph to choose
    /// between the charset (`cid -> gid`) and `/CIDToGIDMap`, and re-deriving
    /// it per glyph would re-walk the Top DICT for nothing. `false` for a
    /// `glyf` program, and for anything whose CFF table won't parse.
    cff_is_cid_keyed: bool,
}

impl SkrifaProgram {
    /// Parse a `/FontFile2` or `/FontFile3` embedded font program as a skrifa
    /// second opinion. Returns `None` when skrifa can't make sense of the bytes
    /// either -- this is not an error condition for the caller, just "no second
    /// opinion available"; the primary decoder's result (or lack of one) stands
    /// on its own.
    pub fn parse(bytes: &[u8]) -> Option<SkrifaProgram> {
        if bytes.len() < 4 {
            return None;
        }
        // Type1 / PFB: out of scope for skrifa (see module docs). Same
        // defensive check as `FontProgram::parse`.
        if bytes[0] == b'%' || bytes[0] == 0x80 {
            return None;
        }
        let wrapped = if is_sfnt_wrapped(bytes) {
            bytes.to_vec()
        } else {
            wrap_bare_cff(bytes)?
        };
        let font = FontRef::new(&wrapped).ok()?;
        let units_per_em = font.head().ok()?.units_per_em();
        if units_per_em == 0 {
            return None; // Would divide by zero in `outline`; treat as unusable.
        }
        // Reject anything skrifa can't actually build an outline source for
        // (e.g. a `CFF ` table that parsed as *present* but whose Top DICT is
        // garbage): `format()` is `None` whenever `OutlineGlyphCollection`
        // degraded to its empty `OutlineCollectionKind::None` variant, which
        // is also exactly what happens if `hmtx`/`hhea` are missing or
        // malformed. Catching that here means `outline()` never has to.
        font.outline_glyphs().format()?;
        let cff_is_cid_keyed = cff_font_ref(&wrapped).is_some_and(|c| c.is_cid());
        Some(SkrifaProgram {
            bytes: wrapped,
            units_per_em,
            cff_is_cid_keyed,
        })
    }

    /// Whether this program's `CFF ` table is **CID-keyed** (Top DICT `ROS`).
    ///
    /// [`super::font::Font`] uses this exactly the way it uses
    /// [`CffProgram::is_cid_keyed`](super::glyph_cff::CffProgram::is_cid_keyed):
    /// a `CIDFontType0` whose CFF is CID-keyed selects glyphs through the CFF
    /// **charset** ([`gid_for_cid`](SkrifaProgram::gid_for_cid)), and one whose
    /// CFF is *not* CID-keyed goes through the PDF's own `/CIDToGIDMap`
    /// instead. Always `false` for a `glyf` (TrueType) program.
    pub fn is_cid_keyed_cff(&self) -> bool {
        self.cff_is_cid_keyed
    }

    /// `code -> gid` for a **simple** (non-CID) font, or `None` when the code
    /// resolves to nothing (caller then falls back to `code == gid` identity,
    /// same as the primary decoders' callers do).
    ///
    /// Two sources, tried in that order, because a font program is one shape or
    /// the other and never both:
    ///
    /// 1. the sfnt **`cmap`**, via skrifa's [`Charmap`](skrifa::charmap::Charmap)
    ///    — which already implements the `(3,0)` symbol-font convention of
    ///    duplicating `U+F000..F0FF` at `U+0000..U+00FF`
    ///    (`skrifa` `src/charmap.rs`, `CodepointSubtable::map`, following
    ///    HarfBuzz), i.e. exactly what
    ///    [`TrueTypeProgram::gid_for_code`](super::glyph_truetype::TrueTypeProgram::gid_for_code)
    ///    hand-rolls;
    /// 2. the **CFF `Encoding`**, via read-fonts' [`CffFontRef::encoding`] —
    ///    which covers the font's custom Encoding *and* the predefined
    ///    Standard/Expert ones. The Expert case is one of our own CFF decoder's
    ///    documented ceilings (gh-67), so this path is strictly wider than
    ///    [`CffProgram::gid_for_code`](super::glyph_cff::CffProgram::gid_for_code),
    ///    not merely a duplicate of it.
    ///
    /// **Codes, not Unicode.** `code` is the raw PDF character code, handed to
    /// the `cmap` as-is. That is deliberate and matches what the primary
    /// TrueType decoder does: for the symbolic embedded fonts this path
    /// actually serves, the `cmap` *is* keyed by the code. A font whose glyphs
    /// are reachable only by going code -> glyph name -> Unicode -> `cmap`
    /// is out of scope here, same as it is for the primary decoder.
    pub fn gid_for_code(&self, code: u32) -> Option<u16> {
        if let Ok(font) = FontRef::new(&self.bytes)
            && let Some(gid) = font.charmap().map(code)
        {
            return u16::try_from(gid.to_u32()).ok();
        }
        let code = u8::try_from(code).ok()?;
        let cff = cff_font_ref(&self.bytes)?;
        let gid = cff.encoding()?.map(code)?;
        u16::try_from(gid.to_u32()).ok()
    }

    /// `cid -> gid` through a **CID-keyed** CFF's charset, falling back to the
    /// identity `cid == gid` when the charset doesn't list the CID — the same
    /// identity-ordering fallback
    /// [`CffProgram::gid_for_cid`](super::glyph_cff::CffProgram::gid_for_cid)
    /// makes, so both decoders answer alike for a font neither has an entry
    /// for.
    ///
    /// In a CID-keyed CFF the charset's SIDs *are* CIDs (CFF spec, `ROS`
    /// operator), which is why the CID goes straight in as a [`Sid`]. Calling
    /// this on a program that is **not** CID-keyed
    /// ([`is_cid_keyed_cff`](SkrifaProgram::is_cid_keyed_cff) is `false`) is a
    /// caller bug rather than a panic: it just returns the identity.
    pub fn gid_for_cid(&self, cid: u32) -> u16 {
        let identity = cid as u16;
        let Ok(sid) = u16::try_from(cid) else {
            return identity;
        };
        cff_font_ref(&self.bytes)
            .and_then(|c| c.charset())
            .and_then(|cs| cs.glyph_id(Sid::new(sid)).ok())
            .and_then(|g| u16::try_from(g.to_u32()).ok())
            .unwrap_or(identity)
    }

    /// The outline of glyph `gid`, in **em space** (y-up, 1 em = 1.0), or
    /// `None` when skrifa has no outline for this glyph either (out of range,
    /// empty, or a decode error). Never panics on malformed/adversarial font
    /// data: every skrifa call here is `Result`/`Option`-propagated with `?`.
    pub fn outline(&self, gid: u16) -> Option<Path> {
        // Re-parse: cheap (table-directory only, see the struct docs), and
        // keeps `SkrifaProgram` free of a self-referential `FontRef` lifetime.
        let font = FontRef::new(&self.bytes).ok()?;
        let glyph = font.outline_glyphs().get(GlyphId::from(gid))?;
        let settings = DrawSettings::unhinted(Size::unscaled(), LocationRef::default());
        let mut pen = PathPen::new(1.0 / self.units_per_em as f32);
        glyph.draw(settings, &mut pen).ok()?;
        if super::glyph::path_is_empty(&pen.path) {
            None
        } else {
            Some(pen.path)
        }
    }
}

/// The `CFF ` table of an sfnt-wrapped program, re-read as a read-fonts
/// [`CffFontRef`] so the CFF's own **charset** and **Encoding** tables are
/// reachable for GID resolution (they are not exposed on skrifa's higher-level
/// outline API, which only ever wants a GID it already has).
///
/// `bytes` must be the *wrapped* form [`SkrifaProgram`] stores — the original
/// sfnt for a `/FontFile2` / OpenType `/FontFile3`, or [`wrap_bare_cff`]'s
/// synthetic `OTTO` for a bare CFF. Returns `None` for a `glyf`-only program
/// (no `CFF ` table at all) or a CFF whose header/Top DICT won't read.
///
/// Top DICT index `0`: "The Name INDEX in the CFF data must contain only one
/// entry; that is, there must be only one font in the CFF FontSet" for a CFF
/// inside an OpenType wrapper (OpenType spec, `CFF ` table), which is the shape
/// everything reaching here has — skrifa's own `cff::Outlines::from_cff`
/// (`src/outline/cff/mod.rs`) passes `0` for the same reason. `upem = None`
/// makes read-fonts assume the CFF-spec default 1000, which is irrelevant to
/// charset/Encoding lookups (they are pure table reads, no scaling).
fn cff_font_ref(bytes: &[u8]) -> Option<CffFontRef<'_>> {
    let font = FontRef::new(bytes).ok()?;
    let cff = font.cff().ok()?;
    CffFontRef::new(cff.offset_data().as_bytes(), 0, None).ok()
}

/// An [`OutlinePen`] that scales skrifa's raw font-unit coordinates into em
/// space and appends the resulting commands to a [`Path`].
struct PathPen {
    path: Path,
    /// `1.0 / units_per_em`: applied to every coordinate skrifa hands us.
    scale: f32,
    /// The last point emitted (in **already-scaled** em space), needed to
    /// degree-elevate a `quad_to` into a cubic via [`super::glyph::quad_to`],
    /// which takes the current point explicitly rather than tracking it itself.
    cur: (f32, f32),
}

impl PathPen {
    fn new(scale: f32) -> PathPen {
        PathPen {
            path: Path::new(),
            scale,
            cur: (0.0, 0.0),
        }
    }

    fn scaled(&self, x: f32, y: f32) -> (f32, f32) {
        (x * self.scale, y * self.scale)
    }
}

impl OutlinePen for PathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.scaled(x, y);
        self.path.move_to(p.0, p.1);
        self.cur = p;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.scaled(x, y);
        self.path.line_to(p.0, p.1);
        self.cur = p;
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        // skrifa's `glyf` (TrueType) outlines are natively quadratic; `Path`
        // only stores cubics, so elevate degree with the same construction
        // `glyph_truetype.rs` uses for its own `glyf` decoding (see
        // `super::glyph::quad_to`'s doc comment for the control-point formula
        // and its MuPDF/FreeType provenance).
        let c = self.scaled(cx0, cy0);
        let p = self.scaled(x, y);
        super::glyph::quad_to(&mut self.path, self.cur, c, p);
        self.cur = p;
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let c0 = self.scaled(cx0, cy0);
        let c1 = self.scaled(cx1, cy1);
        let p = self.scaled(x, y);
        self.path.curve_to(c0.0, c0.1, c1.0, c1.1, p.0, p.1);
        self.cur = p;
    }

    fn close(&mut self) {
        self.path.close();
    }
}

/// Build a complete sfnt TrueType font, **including** the synthetic
/// `hhea`/`hmtx` skrifa requires (see this module's "why hhea/hmtx too"
/// docs) that `glyph_truetype.rs`'s own `build_ttf`/`box_font`/`ring_font`
/// test fixtures don't carry, since our own decoder never needs them. Shared
/// by this module's tests and [`super::glyph`]'s (which needs a skrifa-side
/// fixture distinguishable from its primary-decoder fixture to prove
/// [`super::glyph::FontProgram::outline`]'s primary-first ordering).
#[cfg(test)]
pub(crate) fn build_ttf_with_metrics(units_per_em: u16, glyphs: &[Vec<u8>]) -> Vec<u8> {
    use super::glyph_truetype::assemble_sfnt;

    let mut glyf = Vec::new();
    let mut loca = Vec::new();
    loca.extend_from_slice(&0u32.to_be_bytes());
    for g in glyphs {
        glyf.extend_from_slice(g);
        loca.extend_from_slice(&(glyf.len() as u32).to_be_bytes());
    }
    let mut head = vec![0u8; 54];
    head[18..20].copy_from_slice(&units_per_em.to_be_bytes());
    head[50..52].copy_from_slice(&1i16.to_be_bytes()); // indexToLocFormat = long
    let mut maxp = vec![0x00, 0x00, 0x50, 0x00, 0x00, 0x00]; // version 0.5
    maxp[4..6].copy_from_slice(&(glyphs.len() as u16).to_be_bytes());

    // One (advanceWidth, lsb) hmtx record per glyph, `lsb` set to that
    // glyph's own encoded xMin (bytes 2..4 of `simple_glyf`'s header; 0 for
    // an empty/notdef glyph, which has no bbox worth matching).
    //
    // This has to be correct, not a placeholder, or skrifa's *unhinted*
    // TrueType scaler still shifts the whole outline horizontally: its
    // "phantom point" for the left origin is computed as
    // `xMin - lsb` (`skrifa` `src/outline/glyf/mod.rs`,
    // `setup_phantom_points`), and every x coordinate skrifa emits is
    // relative to that phantom point. A placeholder `lsb = 0` on a glyph
    // whose real xMin isn't 0 (e.g. the ring fixture below, xMin = 100)
    // silently shifts the whole glyph left by `xMin` font units -- found by
    // this module's own test suite failing with a suspiciously-exactly-100-
    // unit-off bounding box. `lsb = xMin` makes the phantom point `(0, ...)`,
    // i.e. no shift, matching how real font tools author `hmtx` for an
    // unhinted glyph. Real embedded PDF fonts don't need this care -- they
    // carry their own already-correct `hmtx` -- this only matters because
    // *this test helper* synthesizes one from scratch.
    let hhea_num_metrics = glyphs.len().max(1) as u16;
    let mut hmtx = Vec::with_capacity(glyphs.len() * 4);
    for g in glyphs {
        let xmin = if g.len() >= 4 {
            i16::from_be_bytes([g[2], g[3]])
        } else {
            0
        };
        hmtx.extend_from_slice(&0u16.to_be_bytes()); // advanceWidth: unused here
        hmtx.extend_from_slice(&xmin.to_be_bytes()); // lsb
    }
    let mut hhea = [0u8; 36];
    hhea[34..36].copy_from_slice(&hhea_num_metrics.to_be_bytes());

    assemble_sfnt(&[
        (b"glyf", glyf),
        (b"head", head),
        (b"hhea", hhea.to_vec()),
        (b"hmtx", hmtx),
        (b"loca", loca),
        (b"maxp", maxp),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::Matrix;
    use crate::mupdf::glyph_truetype::simple_glyf;

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

    /// A box glyph (0,0)-(500,700) at gid 1 in a 1000-upm font, WITH the
    /// hhea/hmtx skrifa needs (unlike `glyph_truetype::box_font`).
    fn box_font_with_metrics() -> Vec<u8> {
        let notdef = simple_glyf(&[]);
        let boxg = simple_glyf(&[vec![
            (0, 0, true),
            (500, 0, true),
            (500, 700, true),
            (0, 700, true),
        ]]);
        build_ttf_with_metrics(1000, &[notdef, boxg])
    }

    // ------------------------------------------------------------------
    // Em-space scaling: the single most important assertion in this module.
    // A units_per_em mixup shows up as a bounding box in the hundreds/
    // thousands instead of 0..1, which is exactly what this test would catch.
    // ------------------------------------------------------------------
    #[test]
    fn truetype_box_scales_into_em_space() {
        let bytes = box_font_with_metrics();
        let prog = SkrifaProgram::parse(&bytes).expect("skrifa should parse a plain sfnt TrueType");
        let path = prog.outline(1).expect("box outline");
        let (x0, y0, x1, y1) = bbox(&path);
        // 500/1000 = 0.5, 700/1000 = 0.7 -- comfortably inside 0..1, not the
        // hundreds/thousands a units-per-em mixup would produce.
        assert!((x0 - 0.0).abs() < 1e-4, "x0 = {x0}");
        assert!((y0 - 0.0).abs() < 1e-4, "y0 = {y0}");
        assert!((x1 - 0.5).abs() < 1e-4, "x1 = {x1}");
        assert!((y1 - 0.7).abs() < 1e-4, "y1 = {y1}");
        assert!(
            x1 < 1.0 && y1 < 1.0,
            "bbox must land within 0..1 em, got {x1}x{y1}"
        );
    }

    // ------------------------------------------------------------------
    // Quadratic -> cubic conversion.
    // ------------------------------------------------------------------
    #[test]
    fn quadratic_segment_elevates_to_expected_cubic() {
        // One contour: on-curve (0,0) -> off-curve control (500,0) -> on-curve
        // endpoint (500,500). The `(500,0)` point is TrueType's off-curve
        // quadratic control point, so skrifa hands us a `quad_to`.
        let notdef = simple_glyf(&[]);
        let curve = simple_glyf(&[vec![(0, 0, true), (500, 0, false), (500, 500, true)]]);
        let bytes = build_ttf_with_metrics(1000, &[notdef, curve]);
        let prog = SkrifaProgram::parse(&bytes).expect("parse curve font");
        let path = prog.outline(1).expect("curve outline");

        let (x0, y0, x1, y1) = bbox(&path);
        assert!(
            x0 >= -1e-4 && y0 >= -1e-4,
            "unexpected negative bbox: {x0},{y0}"
        );
        // A quadratic from (0,0) to (0.5,0.5) via control (0.5,0) bulges
        // towards the control point -- concretely the bbox should reach the
        // control's x (0.5), which a degenerate straight-line stand-in
        // (a bug that dropped the control point) would not produce any
        // differently here, but *would* show up as a wrong y for x near 0.5
        // in a point-by-point check; the bbox reaching (0.5, 0.5) at least
        // confirms the curve was walked all the way to its endpoint.
        assert!((x1 - 0.5).abs() < 1e-3, "x1 = {x1}");
        assert!((y1 - 0.5).abs() < 1e-3, "y1 = {y1}");

        // Stronger check: flatten and confirm the path actually bulges away
        // from the straight line (0,0)-(0.5,0.5) -- i.e. it is genuinely a
        // curve, not a `line_to` standing in for the elevation.
        let polys = path.flatten(Matrix::IDENTITY);
        let on_bulge_side = polys.iter().flatten().any(|p| p.x > p.y + 1e-3);
        assert!(
            on_bulge_side,
            "expected the curve to bulge towards the (0.5,0) control point"
        );
    }

    // ------------------------------------------------------------------
    // Malformed input: never panic, always None.
    // ------------------------------------------------------------------
    #[test]
    fn malformed_input_never_panics() {
        assert!(SkrifaProgram::parse(&[]).is_none());
        assert!(SkrifaProgram::parse(&[0u8; 3]).is_none());
        assert!(SkrifaProgram::parse(&[0u8; 4]).is_none());
        assert!(SkrifaProgram::parse(b"%!PS-AdobeFont-Type1").is_none());
        assert!(SkrifaProgram::parse(&[0x80, 1, 2, 3, 4, 5]).is_none());
        // sfnt magic but truncated / garbage table directory.
        assert!(SkrifaProgram::parse(b"OTTO").is_none());
        assert!(SkrifaProgram::parse(b"OTTOxxxxxxxxxxxxxxxxxxxx").is_none());
        // Something that sniffs as "bare CFF" but is just noise -- wraps
        // "successfully" (a table directory results) but has no real CFF Top
        // DICT for skrifa to build outlines from.
        assert!(SkrifaProgram::parse(&[0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3]).is_none());
        // Out-of-range gid on a font that does parse: no panic, `None`.
        let prog = SkrifaProgram::parse(&box_font_with_metrics()).expect("parse box font");
        assert!(prog.outline(u16::MAX).is_none());
    }

    // ------------------------------------------------------------------
    // Bare-CFF wrapping round trip: a real (if minimal) CFF program, with no
    // sfnt wrapper of its own -- the common `/FontFile3` `Type1C` shape --
    // decoded correctly by skrifa after `wrap_bare_cff`.
    // ------------------------------------------------------------------

    /// Encode a CFF INDEX from its element byte-strings (same technique as
    /// `glyph_cff.rs`'s own test helper of the same name; duplicated here
    /// rather than imported since that one is private to `glyph_cff.rs`'s test
    /// module).
    fn cff_index(items: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(items.len() as u16).to_be_bytes());
        if items.is_empty() {
            return out;
        }
        let mut offsets = vec![1u32];
        for it in items {
            offsets.push(offsets.last().unwrap() + it.len() as u32);
        }
        let max = *offsets.last().unwrap();
        let off_size: u8 = if max <= 0xff {
            1
        } else if max <= 0xffff {
            2
        } else if max <= 0xff_ffff {
            3
        } else {
            4
        };
        out.push(off_size);
        for o in &offsets {
            out.extend_from_slice(&o.to_be_bytes()[(4 - off_size as usize)..]);
        }
        for it in items {
            out.extend_from_slice(it);
        }
        out
    }

    /// A Type2 integer operand in the 3-byte `28 hi lo` form.
    fn num(v: i32) -> Vec<u8> {
        vec![28, (v >> 8) as u8, v as u8]
    }

    /// Assemble a minimal bare CFF (gid 0 = `.notdef`, gid 1 = `charstring`).
    fn build_minimal_cff(charstring: Vec<u8>) -> Vec<u8> {
        let header = vec![1u8, 0, 4, 1];
        let name = cff_index(&[b"KOPITEST".to_vec()]);
        let strings = cff_index(&[]);
        let gsubr = cff_index(&[]);
        let make_top = |cs_off: u32| -> Vec<u8> {
            let mut d = vec![29u8];
            d.extend_from_slice(&cs_off.to_be_bytes());
            d.push(17); // CharStrings operator
            cff_index(&[d])
        };
        let top_len = make_top(0).len();
        let cs_off = (header.len() + name.len() + top_len + strings.len() + gsubr.len()) as u32;
        let top = make_top(cs_off);
        let cs_index = cff_index(&[vec![14], charstring]); // gid 0: endchar only

        let mut out = Vec::new();
        out.extend_from_slice(&header);
        out.extend_from_slice(&name);
        out.extend_from_slice(&top);
        out.extend_from_slice(&strings);
        out.extend_from_slice(&gsubr);
        out.extend_from_slice(&cs_index);
        out
    }

    /// `rmoveto(100,0) rlineto -> (400,0) rlineto -> (250,600) endchar` -- a
    /// triangle, default FontMatrix (0.001 -> 1000 upm).
    fn triangle_charstring() -> Vec<u8> {
        let mut cs = Vec::new();
        cs.extend(num(100));
        cs.extend(num(0));
        cs.push(21); // rmoveto
        cs.extend(num(300));
        cs.extend(num(0));
        cs.push(5); // rlineto -> (400,0)
        cs.extend(num(-150));
        cs.extend(num(600));
        cs.push(5); // rlineto -> (250,600)
        cs.push(14); // endchar
        cs
    }

    #[test]
    fn bare_cff_wraps_and_decodes_via_skrifa() {
        let cff = build_minimal_cff(triangle_charstring());
        assert!(
            !is_sfnt_wrapped(&cff),
            "a bare CFF program has no sfnt magic"
        );
        let prog = SkrifaProgram::parse(&cff).expect("skrifa should parse a wrapped bare CFF");
        let path = prog.outline(1).expect("triangle outline");
        let (x0, y0, x1, y1) = bbox(&path);
        // Font units (100..400, 0..600) / 1000 upm -> em (0.1..0.4, 0..0.6).
        assert!((x0 - 0.1).abs() < 1e-3, "x0 = {x0}");
        assert!((y0 - 0.0).abs() < 1e-3, "y0 = {y0}");
        assert!((x1 - 0.4).abs() < 1e-3, "x1 = {x1}");
        assert!((y1 - 0.6).abs() < 1e-3, "y1 = {y1}");
    }
}
