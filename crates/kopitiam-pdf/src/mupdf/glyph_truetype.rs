//! TrueType (`glyf`) glyph-outline decoding: the pure-Rust stand-in for the
//! FreeType `sfnt`/`truetype` driver MuPDF loads embedded `/FontFile2` (and
//! OpenType-`glyf` `/FontFile3`) programs through in `source/fitz/font.c`
//! (`fz_load_glyph` -> `FT_Load_Glyph` -> the outline decompose of font.c:1418,
//! commit 19f1284, AGPL-3.0, © Artifex Software, Inc.). Translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the outline-to-[`Path`] shape
//! follows MuPDF's FreeType callbacks; the `glyf`/`loca`/`cmap` table parsing and
//! the on-/off-curve point reconstruction are re-implemented from the OpenType
//! specification (what FreeType's `truetype` driver decodes). See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What is decoded
//!
//! * `head` (unitsPerEm, `indexToLocFormat`), `maxp` (numGlyphs), `loca`, `glyf`.
//! * **Simple glyphs**: flag/coordinate decoding + the quadratic on-/off-curve
//!   contour reconstruction (implied midpoints between consecutive off-curve
//!   points), each quadratic elevated to a cubic ([`super::glyph::quad_to`]).
//! * **Composite glyphs**: `MORE_COMPONENTS` loop, `ARGS_ARE_XY_VALUES`
//!   translation, and the `WE_HAVE_A_SCALE` / `X_AND_Y_SCALE` / `TWO_BY_TWO`
//!   component transforms, decoded recursively (depth-guarded).
//! * `cmap` (formats 0/4/6/12): the character-code -> GID map a **simple**
//!   TrueType font needs (the port has no FreeType charmap). CID fonts select
//!   glyphs through `CIDToGIDMap` instead and do not consult this.

use super::draw_path::Path;
use super::glyph::{Affine, Reader, quad_to};
use std::collections::HashMap;

/// Recursion cap on composite-glyph component nesting.
const MAX_COMPONENT_DEPTH: u32 = 8;

/// A parsed TrueType outline source.
#[derive(Clone, Debug)]
pub struct TrueTypeProgram {
    /// The whole sfnt (or bare TrueType) byte image; tables index into it.
    data: Vec<u8>,
    /// Byte offset + length of the `glyf` table.
    glyf: (usize, usize),
    /// Parsed glyph offsets from `loca` (`numGlyphs + 1` entries into `glyf`).
    loca: Vec<u32>,
    /// Font design units per em (the outline scale denominator).
    units_per_em: f32,
    /// `code -> gid` for simple fonts (from the best `cmap` subtable). Empty when
    /// the font has no usable cmap (CID fonts, or symbol fonts we key differently).
    cmap: HashMap<u32, u16>,
}

impl TrueTypeProgram {
    /// Parse an sfnt / bare-TrueType image. Returns `None` if the required
    /// `glyf`/`loca`/`head`/`maxp` tables are missing or malformed.
    pub fn parse(bytes: &[u8]) -> Option<TrueTypeProgram> {
        let dir = TableDir::parse(bytes)?;
        let (head_off, _) = dir.find(b"head")?;
        let (maxp_off, _) = dir.find(b"maxp")?;
        let glyf = dir.find(b"glyf")?;
        let loca_tbl = dir.find(b"loca")?;

        // head: unitsPerEm @18 (u16), indexToLocFormat @50 (i16).
        let mut r = Reader::at(bytes, head_off + 18);
        let mut units_per_em = r.u16() as f32;
        if units_per_em == 0.0 {
            units_per_em = 1000.0;
        }
        let mut r = Reader::at(bytes, head_off + 50);
        let loc_format = r.i16();

        // maxp: numGlyphs @4 (u16).
        let mut r = Reader::at(bytes, maxp_off + 4);
        let num_glyphs = r.u16() as usize;

        let loca = parse_loca(bytes, loca_tbl, loc_format, num_glyphs)?;
        let cmap = dir
            .find(b"cmap")
            .map(|(off, _)| parse_cmap(bytes, off))
            .unwrap_or_default();

        Some(TrueTypeProgram {
            data: bytes.to_vec(),
            glyf: (glyf.0, glyf.1),
            loca,
            units_per_em,
            cmap,
        })
    }

    /// Extract the raw `CFF ` table bytes from an OpenType (`OTTO` / sfnt) image,
    /// or `None` if there is no `CFF ` table (so the caller decodes `glyf`
    /// instead). Used by [`super::glyph::FontProgram::parse`] to route
    /// OpenType-CFF fonts to the CFF decoder.
    pub fn cff_table(bytes: &[u8]) -> Option<Vec<u8>> {
        let dir = TableDir::parse(bytes)?;
        let (off, len) = dir.find(b"CFF ")?;
        bytes.get(off..off + len).map(|s| s.to_vec())
    }

    /// The `code -> gid` mapping for a simple TrueType font. Tries the code as-is,
    /// then the symbol-font `0xF000 | code` convention. `None` when unmapped.
    pub fn gid_for_code(&self, code: u32) -> Option<u16> {
        if let Some(&g) = self.cmap.get(&code) {
            return Some(g);
        }
        if code < 0x100
            && let Some(&g) = self.cmap.get(&(0xF000 | code))
        {
            return Some(g);
        }
        None
    }

    /// The number of glyphs (from `loca`).
    pub fn num_glyphs(&self) -> u16 {
        self.loca.len().saturating_sub(1).min(u16::MAX as usize) as u16
    }

    /// Decode glyph `gid` to an em-space [`Path`] (y-up, 1 em = 1.0), or `None`
    /// for an empty / out-of-range glyph.
    pub fn outline(&self, gid: u16) -> Option<Path> {
        let mut path = Path::new();
        let xform = Affine::scale(1.0 / self.units_per_em);
        self.decode_into(&mut path, gid, xform, 0);
        // An empty path (whitespace glyph, or a decode that produced nothing) is
        // reported as None so the caller can skip / fall back cleanly.
        if super::glyph::path_is_empty(&path) {
            None
        } else {
            Some(path)
        }
    }

    /// Append glyph `gid`'s contours to `path`, transformed by `xform`
    /// (font-unit glyph space -> em space, plus any composite-component transform).
    fn decode_into(&self, path: &mut Path, gid: u16, xform: Affine, depth: u32) {
        if depth > MAX_COMPONENT_DEPTH {
            return;
        }
        let g = gid as usize;
        if g + 1 >= self.loca.len() {
            return;
        }
        let start = self.glyf.0 + self.loca[g] as usize;
        let end = self.glyf.0 + self.loca[g + 1] as usize;
        if end <= start || end > self.glyf.0 + self.glyf.1 || end > self.data.len() {
            return; // empty glyph (e.g. space) or out of range
        }

        let mut r = Reader::at(&self.data, start);
        let ncont = r.i16();
        // Skip the glyph bbox (xMin/yMin/xMax/yMax).
        r.seek(r.pos() + 8);

        if ncont >= 0 {
            self.decode_simple(path, &mut r, ncont as usize, &xform);
        } else {
            self.decode_composite(path, &mut r, &xform, depth);
        }
    }

    // OpenType `glyf` simple-glyph body.
    fn decode_simple(&self, path: &mut Path, r: &mut Reader, ncont: usize, xform: &Affine) {
        if ncont == 0 {
            return;
        }
        // endPtsOfContours -> point count.
        let mut ends = Vec::with_capacity(ncont);
        for _ in 0..ncont {
            ends.push(r.u16() as usize);
        }
        let npoints = ends.last().map(|e| e + 1).unwrap_or(0);
        if npoints == 0 || npoints > 20_000 {
            return; // empty, or absurd (guard against corruption)
        }

        // Skip the hinting instructions.
        let instr_len = r.u16() as usize;
        r.seek(r.pos() + instr_len);

        // Flags (with the REPEAT run-length byte).
        const ON_CURVE: u8 = 0x01;
        const X_SHORT: u8 = 0x02;
        const Y_SHORT: u8 = 0x04;
        const REPEAT: u8 = 0x08;
        const X_SAME_OR_POS: u8 = 0x10; // when X_SHORT: sign; else: dx==0
        const Y_SAME_OR_POS: u8 = 0x20;

        let mut flags = Vec::with_capacity(npoints);
        while flags.len() < npoints {
            let f = r.u8();
            flags.push(f);
            if f & REPEAT != 0 {
                let mut n = r.u8();
                while n > 0 && flags.len() < npoints {
                    flags.push(f);
                    n -= 1;
                }
            }
        }

        // X then Y coordinate deltas -> absolute font-unit coords.
        let mut xs = Vec::with_capacity(npoints);
        let mut acc = 0i32;
        for &f in &flags {
            if f & X_SHORT != 0 {
                let d = r.u8() as i32;
                acc += if f & X_SAME_OR_POS != 0 { d } else { -d };
            } else if f & X_SAME_OR_POS == 0 {
                acc += r.i16() as i32;
            }
            xs.push(acc as f32);
        }
        let mut ys = Vec::with_capacity(npoints);
        acc = 0;
        for &f in &flags {
            if f & Y_SHORT != 0 {
                let d = r.u8() as i32;
                acc += if f & Y_SAME_OR_POS != 0 { d } else { -d };
            } else if f & Y_SAME_OR_POS == 0 {
                acc += r.i16() as i32;
            }
            ys.push(acc as f32);
        }

        // Emit each contour.
        let mut begin = 0usize;
        for &e in &ends {
            if e >= npoints || e < begin {
                break;
            }
            let pts: Vec<(f32, f32, bool)> = (begin..=e)
                .map(|i| (xs[i], ys[i], flags[i] & ON_CURVE != 0))
                .collect();
            emit_contour(path, xform, &pts);
            begin = e + 1;
        }
    }

    // OpenType `glyf` composite-glyph body: recurse into each component.
    fn decode_composite(&self, path: &mut Path, r: &mut Reader, xform: &Affine, depth: u32) {
        const ARG_WORDS: u16 = 0x0001;
        const ARGS_XY: u16 = 0x0002;
        const HAVE_SCALE: u16 = 0x0008;
        const MORE: u16 = 0x0020;
        const XY_SCALE: u16 = 0x0040;
        const TWO_BY_TWO: u16 = 0x0080;

        loop {
            let flags = r.u16();
            let comp_gid = r.u16();

            // Arguments: dx/dy (XY) or point indices (ignored -> no offset).
            let (dx, dy);
            if flags & ARG_WORDS != 0 {
                let a1 = r.i16() as f32;
                let a2 = r.i16() as f32;
                (dx, dy) = if flags & ARGS_XY != 0 {
                    (a1, a2)
                } else {
                    (0.0, 0.0)
                };
            } else {
                let a1 = r.u8() as i8 as f32;
                let a2 = r.u8() as i8 as f32;
                (dx, dy) = if flags & ARGS_XY != 0 {
                    (a1, a2)
                } else {
                    (0.0, 0.0)
                };
            }

            // 2x2 component transform (F2Dot14).
            let (mut a, mut b, mut c, mut d) = (1.0f32, 0.0f32, 0.0f32, 1.0f32);
            if flags & HAVE_SCALE != 0 {
                let s = f2dot14(r.i16());
                a = s;
                d = s;
            } else if flags & XY_SCALE != 0 {
                a = f2dot14(r.i16());
                d = f2dot14(r.i16());
            } else if flags & TWO_BY_TWO != 0 {
                a = f2dot14(r.i16());
                b = f2dot14(r.i16());
                c = f2dot14(r.i16());
                d = f2dot14(r.i16());
            }

            // Component space -> parent space -> (parent's) em space.
            let comp = Affine {
                a,
                b,
                c,
                d,
                e: dx,
                f: dy,
            };
            let child = comp.concat(xform);
            self.decode_into(path, comp_gid, child, depth + 1);

            if flags & MORE == 0 {
                break;
            }
        }
    }
}

/// Emit one contour (a slice of `(x, y, on_curve)` font-unit points) into `path`,
/// transformed by `xform`. Reconstructs the quadratic outline: consecutive
/// off-curve points imply an on-curve midpoint, and the contour is rotated to
/// start on an on-curve point.
fn emit_contour(path: &mut Path, xform: &Affine, raw: &[(f32, f32, bool)]) {
    let n = raw.len();
    if n < 2 {
        return;
    }
    // Insert the implied on-curve midpoints between consecutive off-curve points
    // (cyclically), so every off-curve control is followed by an on-curve point.
    let mut q: Vec<(f32, f32, bool)> = Vec::with_capacity(n + 4);
    for i in 0..n {
        let cur = raw[i];
        q.push(cur);
        let next = raw[(i + 1) % n];
        if !cur.2 && !next.2 {
            q.push(((cur.0 + next.0) * 0.5, (cur.1 + next.1) * 0.5, true));
        }
    }
    // Rotate to begin on an on-curve point.
    let Some(start_idx) = q.iter().position(|p| p.2) else {
        return; // no on-curve point even after midpoint insertion (degenerate)
    };
    let m = q.len();
    let pt = |k: usize| -> (f32, f32, bool) { q[(start_idx + k) % m] };

    let start = pt(0);
    let s = xform.apply(start.0, start.1);
    path.move_to(s.0, s.1);
    let mut cur = s;
    let mut i = 1usize;
    while i <= m {
        let p = pt(i);
        if p.2 {
            let e = xform.apply(p.0, p.1);
            path.line_to(e.0, e.1);
            cur = e;
            i += 1;
        } else {
            let ctrl = xform.apply(p.0, p.1);
            let ep = pt(i + 1);
            let e = xform.apply(ep.0, ep.1);
            quad_to(path, cur, ctrl, e);
            cur = e;
            i += 2;
        }
    }
    path.close();
}

/// Decode an F2Dot14 fixed-point value (composite-transform scales).
fn f2dot14(v: i16) -> f32 {
    v as f32 / 16384.0
}

// ---------------------------------------------------------------------------
// Test font construction (shared by the glyf, draw-device, and font tests)
// ---------------------------------------------------------------------------

/// Assemble a minimal sfnt image from `(tag, table_bytes)` pairs (the checksum
/// field is left zero -- the decoders here do not verify it).
#[cfg(test)]
pub(crate) fn assemble_sfnt(tables: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let n = tables.len();
    let mut out = Vec::new();
    // Offset table: sfntVersion 1.0, numTables, then the binary-search fields.
    out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    out.extend_from_slice(&(n as u16).to_be_bytes());
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // searchRange/entrySelector/rangeShift
    // Table records start after the 12-byte header; body follows the records.
    let body_start = 12 + n * 16;
    let mut off = body_start;
    for (tag, data) in tables {
        out.extend_from_slice(*tag);
        out.extend_from_slice(&0u32.to_be_bytes()); // checksum (unused)
        out.extend_from_slice(&(off as u32).to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        off += data.len();
    }
    for (_, data) in tables {
        out.extend_from_slice(data);
    }
    out
}

/// A 54-byte `head` table with `units_per_em` (@18) and long `loca` (@50 = 1).
#[cfg(test)]
fn head_table(units_per_em: u16) -> Vec<u8> {
    let mut h = vec![0u8; 54];
    h[18..20].copy_from_slice(&units_per_em.to_be_bytes());
    h[50..52].copy_from_slice(&1i16.to_be_bytes()); // indexToLocFormat = long
    h
}

/// A 6-byte `maxp` (version 0.5) declaring `num_glyphs`.
#[cfg(test)]
fn maxp_table(num_glyphs: u16) -> Vec<u8> {
    let mut m = vec![0x00, 0x00, 0x50, 0x00, 0x00, 0x00];
    m[4..6].copy_from_slice(&num_glyphs.to_be_bytes());
    m
}

/// Encode one simple `glyf` entry from its contours (each a list of
/// `(x, y, on_curve)` font-unit points), using plain i16 coordinate deltas.
#[cfg(test)]
pub(crate) fn simple_glyf(contours: &[Vec<(i16, i16, bool)>]) -> Vec<u8> {
    let mut g = Vec::new();
    let all: Vec<(i16, i16, bool)> = contours.iter().flatten().copied().collect();
    let (xmin, ymin, xmax, ymax) = all.iter().fold(
        (i16::MAX, i16::MAX, i16::MIN, i16::MIN),
        |(a, b, c, d), p| (a.min(p.0), b.min(p.1), c.max(p.0), d.max(p.1)),
    );
    g.extend_from_slice(&(contours.len() as i16).to_be_bytes());
    g.extend_from_slice(&xmin.to_be_bytes());
    g.extend_from_slice(&ymin.to_be_bytes());
    g.extend_from_slice(&xmax.to_be_bytes());
    g.extend_from_slice(&ymax.to_be_bytes());
    let mut end = 0usize;
    for c in contours {
        end += c.len();
        g.extend_from_slice(&((end - 1) as u16).to_be_bytes());
    }
    g.extend_from_slice(&0u16.to_be_bytes()); // instructionLength
    for p in &all {
        g.push(if p.2 { 0x01 } else { 0x00 }); // ON_CURVE flag only (i16 deltas)
    }
    let mut px = 0i16;
    for p in &all {
        g.extend_from_slice(&(p.0 - px).to_be_bytes());
        px = p.0;
    }
    let mut py = 0i16;
    for p in &all {
        g.extend_from_slice(&(p.1 - py).to_be_bytes());
        py = p.1;
    }
    g
}

/// A format-0 `cmap` mapping single-byte `code -> gid` (`map` is 256 entries).
#[cfg(test)]
fn cmap_format0(map: &[u8; 256]) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&0u16.to_be_bytes()); // version
    c.extend_from_slice(&1u16.to_be_bytes()); // numTables
    c.extend_from_slice(&1u16.to_be_bytes()); // platformID = Macintosh
    c.extend_from_slice(&0u16.to_be_bytes()); // encodingID = Roman
    c.extend_from_slice(&12u32.to_be_bytes()); // subtable offset
    c.extend_from_slice(&0u16.to_be_bytes()); // format 0
    c.extend_from_slice(&262u16.to_be_bytes()); // length
    c.extend_from_slice(&0u16.to_be_bytes()); // language
    c.extend_from_slice(map);
    c
}

/// Build a complete TrueType font (`glyphs[0]` is `.notdef`) with an optional
/// `code -> gid` cmap. `glyphs` are pre-encoded `glyf` entries.
#[cfg(test)]
pub(crate) fn build_ttf(
    units_per_em: u16,
    glyphs: &[Vec<u8>],
    cmap_map: Option<[u8; 256]>,
) -> Vec<u8> {
    // loca (long): running offsets into the concatenated glyf table.
    let mut glyf = Vec::new();
    let mut loca = Vec::new();
    loca.extend_from_slice(&0u32.to_be_bytes());
    for g in glyphs {
        glyf.extend_from_slice(g);
        loca.extend_from_slice(&(glyf.len() as u32).to_be_bytes());
    }
    let mut tables: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"head", head_table(units_per_em)),
        (b"maxp", maxp_table(glyphs.len() as u16)),
        (b"loca", loca),
        (b"glyf", glyf),
    ];
    if let Some(m) = cmap_map {
        tables.push((b"cmap", cmap_format0(&m)));
    }
    assemble_sfnt(&tables)
}

/// A single-glyph (`.notdef` + a box) font: gid 1 is the box `(0,0)-(500,700)`.
#[cfg(test)]
pub(crate) fn box_font() -> Vec<u8> {
    let notdef = simple_glyf(&[]);
    let boxg = simple_glyf(&[vec![
        (0, 0, true),
        (500, 0, true),
        (500, 700, true),
        (0, 700, true),
    ]]);
    build_ttf(1000, &[notdef, boxg], None)
}

/// A ring-glyph (`O`-like) font: gid 1 has an outer + oppositely-wound inner
/// square, so a nonzero fill leaves the centre a hole. cmap maps code `0x41` ->
/// gid 1. Used by the end-to-end "letterform has interior white" test.
#[cfg(test)]
pub(crate) fn ring_font() -> Vec<u8> {
    let notdef = simple_glyf(&[]);
    let ring = simple_glyf(&[
        // Outer square, clockwise (in y-up).
        vec![
            (100, 100, true),
            (100, 900, true),
            (900, 900, true),
            (900, 100, true),
        ],
        // Inner square, counter-clockwise -> a hole under nonzero winding.
        vec![
            (300, 300, true),
            (700, 300, true),
            (700, 700, true),
            (300, 700, true),
        ],
    ]);
    let mut map = [0u8; 256];
    map[0x41] = 1;
    build_ttf(1000, &[notdef, ring], Some(map))
}

// ---------------------------------------------------------------------------
// sfnt table directory + loca + cmap
// ---------------------------------------------------------------------------

/// The sfnt table directory: `tag -> (offset, length)`.
struct TableDir {
    tables: Vec<([u8; 4], usize, usize)>,
}

impl TableDir {
    fn parse(bytes: &[u8]) -> Option<TableDir> {
        if bytes.len() < 12 {
            return None;
        }
        // A TrueType Collection: use the first font's table directory.
        let mut base = 0usize;
        if &bytes[0..4] == b"ttcf" {
            let mut r = Reader::at(bytes, 12);
            base = r.u32() as usize;
        }
        let mut r = Reader::at(bytes, base);
        let _sfnt_version = r.u32();
        let num_tables = r.u16() as usize;
        r.seek(base + 12); // skip searchRange/entrySelector/rangeShift
        let mut tables = Vec::with_capacity(num_tables.min(64));
        for _ in 0..num_tables {
            if r.remaining() < 16 {
                break;
            }
            let mut tag = [0u8; 4];
            tag[0] = r.u8();
            tag[1] = r.u8();
            tag[2] = r.u8();
            tag[3] = r.u8();
            let _checksum = r.u32();
            let off = r.u32() as usize;
            let len = r.u32() as usize;
            if off <= bytes.len() {
                tables.push((tag, off, len.min(bytes.len().saturating_sub(off))));
            }
        }
        Some(TableDir { tables })
    }

    fn find(&self, tag: &[u8; 4]) -> Option<(usize, usize)> {
        self.tables.iter().find(|t| &t.0 == tag).map(|t| (t.1, t.2))
    }
}

/// Parse the `loca` table into `numGlyphs + 1` byte offsets into `glyf`.
fn parse_loca(
    bytes: &[u8],
    loca: (usize, usize),
    format: i16,
    num_glyphs: usize,
) -> Option<Vec<u32>> {
    let (off, len) = loca;
    let count = num_glyphs + 1;
    let mut r = Reader::at(bytes, off);
    let mut out = Vec::with_capacity(count);
    if format == 0 {
        // Short format: u16 offsets, doubled.
        if len < count * 2 {
            return None;
        }
        for _ in 0..count {
            out.push(r.u16() as u32 * 2);
        }
    } else {
        // Long format: u32 offsets.
        if len < count * 4 {
            return None;
        }
        for _ in 0..count {
            out.push(r.u32());
        }
    }
    Some(out)
}

/// Parse the `cmap` table, choosing the most useful subtable, into a
/// `code -> gid` map. Returns an empty map on any problem (a simple font with no
/// cmap then falls back to identity code==gid at the call site).
fn parse_cmap(bytes: &[u8], cmap_off: usize) -> HashMap<u32, u16> {
    let mut r = Reader::at(bytes, cmap_off);
    let _version = r.u16();
    let n = r.u16() as usize;

    // Rank the encoding records; prefer Unicode/BMP, then symbol, then Mac.
    let mut best: Option<(i32, usize)> = None;
    for _ in 0..n {
        if r.remaining() < 8 {
            break;
        }
        let platform = r.u16();
        let encoding = r.u16();
        let sub_off = r.u32() as usize;
        let score = match (platform, encoding) {
            (3, 10) => 5, // Windows UCS-4
            (3, 1) => 4,  // Windows Unicode BMP
            (0, _) => 3,  // Unicode
            (3, 0) => 2,  // Windows Symbol
            (1, 0) => 1,  // Mac Roman
            _ => 0,
        };
        if best.map(|(s, _)| score > s).unwrap_or(true) {
            best = Some((score, cmap_off + sub_off));
        }
    }

    let Some((_, sub)) = best else {
        return HashMap::new();
    };
    parse_cmap_subtable(bytes, sub)
}

/// Parse a single cmap subtable (formats 0, 4, 6, 12) into `code -> gid`.
fn parse_cmap_subtable(bytes: &[u8], off: usize) -> HashMap<u32, u16> {
    let mut map = HashMap::new();
    let mut r = Reader::at(bytes, off);
    let format = r.u16();
    match format {
        0 => {
            // Byte encoding: 256 single-byte glyph indices.
            let _len = r.u16();
            let _lang = r.u16();
            for code in 0u32..256 {
                let g = r.u8();
                if g != 0 {
                    map.insert(code, g as u16);
                }
            }
        }
        6 => {
            // Trimmed table mapping.
            let _len = r.u16();
            let _lang = r.u16();
            let first = r.u16() as u32;
            let count = r.u16() as usize;
            for i in 0..count {
                let g = r.u16();
                if g != 0 {
                    map.insert(first + i as u32, g);
                }
            }
        }
        4 => parse_cmap_format4(&mut r, &mut map),
        12 => {
            // Segmented coverage (UCS-4).
            let _reserved = r.u16();
            let _len = r.u32();
            let _lang = r.u32();
            let ngroups = r.u32() as usize;
            for _ in 0..ngroups.min(100_000) {
                let start = r.u32();
                let end = r.u32();
                let start_gid = r.u32();
                if end < start || end - start > 65_535 {
                    continue;
                }
                for c in start..=end {
                    map.insert(c, (start_gid + (c - start)) as u16);
                }
            }
        }
        _ => {}
    }
    map
}

/// Parse a format-4 (segment mapping to delta values) cmap subtable.
fn parse_cmap_format4(r: &mut Reader, map: &mut HashMap<u32, u16>) {
    let _len = r.u16();
    let _lang = r.u16();
    let segx2 = r.u16() as usize;
    let segcount = segx2 / 2;
    let _search_range = r.u16();
    let _entry_selector = r.u16();
    let _range_shift = r.u16();

    let end_codes: Vec<u16> = (0..segcount).map(|_| r.u16()).collect();
    let _reserved_pad = r.u16();
    let start_codes: Vec<u16> = (0..segcount).map(|_| r.u16()).collect();
    let id_deltas: Vec<i16> = (0..segcount).map(|_| r.i16()).collect();
    // idRangeOffset entries: record the file position of each so the glyphIdArray
    // (which follows contiguously) can be indexed relative to it, per the spec.
    let id_range_base = r.pos();
    let id_range_offsets: Vec<u16> = (0..segcount).map(|_| r.u16()).collect();

    for s in 0..segcount {
        let start = start_codes[s] as u32;
        let end = end_codes[s] as u32;
        if start == 0xFFFF {
            continue;
        }
        for c in start..=end.min(0xFFFF) {
            let gid = if id_range_offsets[s] == 0 {
                (c as i32 + id_deltas[s] as i32) as u16
            } else {
                // glyphIdArray index per the OpenType format-4 addressing.
                let ro = id_range_offsets[s] as usize;
                let gid_pos = id_range_base + s * 2 + ro + (c - start) as usize * 2;
                let mut gr = Reader::at(r.data(), gid_pos);
                let g = gr.u16();
                if g == 0 {
                    0
                } else {
                    (g as i32 + id_deltas[s] as i32) as u16
                }
            };
            if gid != 0 {
                map.insert(c, gid);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::Matrix;

    /// The device-space bbox (min_x, min_y, max_x, max_y) of a flattened path.
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

    #[test]
    fn box_glyph_decodes_to_em_space_bbox() {
        let prog = TrueTypeProgram::parse(&box_font()).expect("parse box ttf");
        assert_eq!(prog.num_glyphs(), 2);
        let path = prog.outline(1).expect("box outline");
        // 1000 units/em: box (0,0)-(500,700) -> em (0,0)-(0.5,0.7).
        let (x0, y0, x1, y1) = bbox(&path);
        assert!(
            x0.abs() < 1e-4 && y0.abs() < 1e-4,
            "origin corner {x0},{y0}"
        );
        assert!((x1 - 0.5).abs() < 1e-4, "max x {x1}");
        assert!((y1 - 0.7).abs() < 1e-4, "max y {y1}");
    }

    #[test]
    fn box_glyph_has_four_corners_one_contour() {
        let prog = TrueTypeProgram::parse(&box_font()).unwrap();
        let path = prog.outline(1).unwrap();
        let polys = path.flatten(Matrix::IDENTITY);
        assert_eq!(polys.len(), 1, "one contour");
        // move + 3 line + closing return: >= 4 distinct corners.
        assert!(polys[0].len() >= 4, "corners {}", polys[0].len());
    }

    #[test]
    fn empty_glyph_is_none() {
        let prog = TrueTypeProgram::parse(&box_font()).unwrap();
        // gid 0 is .notdef with no contours -> no drawable outline.
        assert!(prog.outline(0).is_none());
        // Out-of-range gid never panics.
        assert!(prog.outline(99).is_none());
    }

    #[test]
    fn cmap_maps_code_to_gid() {
        let prog = TrueTypeProgram::parse(&ring_font()).unwrap();
        assert_eq!(prog.gid_for_code(0x41), Some(1));
        assert_eq!(prog.gid_for_code(0x42), None);
    }

    #[test]
    fn ring_glyph_has_two_contours() {
        let prog = TrueTypeProgram::parse(&ring_font()).unwrap();
        let path = prog.outline(1).unwrap();
        let polys = path.flatten(Matrix::IDENTITY);
        assert_eq!(polys.len(), 2, "outer + inner contour");
    }

    #[test]
    fn composite_glyph_translates_component() {
        // gid1 = a box; gid2 = a composite placing gid1 translated by (500,0).
        let notdef = simple_glyf(&[]);
        let boxg = simple_glyf(&[vec![
            (0, 0, true),
            (400, 0, true),
            (400, 400, true),
            (0, 400, true),
        ]]);
        // Composite: flags = ARG_WORDS|ARGS_XY|MORE_off, component gid 1, dx=500 dy=0.
        let mut comp = Vec::new();
        comp.extend_from_slice(&(-1i16).to_be_bytes()); // numberOfContours < 0 -> composite
        comp.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // bbox (ignored)
        let flags: u16 = 0x0001 | 0x0002; // ARG_1_AND_2_ARE_WORDS | ARGS_ARE_XY_VALUES
        comp.extend_from_slice(&flags.to_be_bytes());
        comp.extend_from_slice(&1u16.to_be_bytes()); // component gid 1
        comp.extend_from_slice(&500i16.to_be_bytes()); // dx
        comp.extend_from_slice(&0i16.to_be_bytes()); // dy
        let ttf = build_ttf(1000, &[notdef, boxg, comp], None);
        let prog = TrueTypeProgram::parse(&ttf).unwrap();
        let path = prog.outline(2).expect("composite outline");
        let (x0, _, x1, _) = bbox(&path);
        // Component box 0..400 translated +500 -> font 500..900 -> em 0.5..0.9.
        assert!((x0 - 0.5).abs() < 1e-4, "x0 {x0}");
        assert!((x1 - 0.9).abs() < 1e-4, "x1 {x1}");
    }
}
