//! CFF / Type2 charstring glyph-outline decoding: the pure-Rust stand-in for the
//! FreeType `cff` driver MuPDF loads embedded `/FontFile3` (`Type1C` /
//! `CIDFontType0C` / OpenType-`CFF `) programs through in `source/fitz/font.c`
//! (`fz_load_glyph` -> `FT_Load_Glyph` -> the outline decompose of font.c:1418,
//! commit 19f1284, AGPL-3.0, © Artifex Software, Inc.). The CFF container parse
//! (INDEX / DICT / charset / FDSelect and the `subr_bias`) is a close adaptation
//! of MuPDF's own non-FreeType CFF reader in `source/fitz/subset-cff.c` (the
//! `parse_index` / `parse_dict` / `subr_bias` / `execute_charstring` shape);
//! the Type2 charstring **interpreter** builds a [`Path`] following the Adobe
//! Type2 Charstring Format (Technical Note #5177), which is what FreeType's `cff`
//! driver executes. Translated to Rust for KOPITIAM (AGPL-3.0-only). See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What is decoded
//!
//! * The CFF header, Name / Top-DICT / String / Global-Subr INDEXes.
//! * Top DICT: `CharStrings`, `charset`, `Encoding`, `Private` (-> local subrs,
//!   `defaultWidthX` / `nominalWidthX`), `FontMatrix`, `CharstringType`, and the
//!   CID keys `ROS` / `FDArray` / `FDSelect`.
//! * The **Type2 charstring interpreter**: `rmoveto`/`rlineto`/`rrcurveto` and the
//!   whole move/line/curve family (`h`/`v`/`hh`/`vv`/`hv`/`vh`, `rcurveline`,
//!   `rlinecurve`), the flex operators (`flex`/`hflex`/`hflex1`/`flex1`), the
//!   hint operators (`hstem`…`cntrmask`, whose mask bytes are consumed), `callsubr`
//!   / `callgsubr` / `return` with the correct subr bias, `endchar`, and the
//!   leading-width parsing.
//!
//! # GID selection
//!
//! * **CID-keyed** CFF (`CIDFontType0C`): the `charset` maps GID -> CID; inverted
//!   here to select a glyph by CID ([`CffProgram::gid_for_cid`]).
//! * **Simple** CFF (`Type1C`): the CFF's own custom `Encoding` maps character
//!   code -> GID directly ([`CffProgram::gid_for_code`]) — the common subset-font
//!   case. When the encoding is instead the **predefined Standard** encoding
//!   (`Encoding` offset `0`, or absent) -- common for Type1-converted-to-CFF
//!   subset fonts with an unmodified encoding -- `gid_for_code` resolves
//!   `code -> name` via [`super::encodings::STANDARD`] (identical to Adobe
//!   StandardEncoding, CFF spec Appendix B) and then `name -> GID` via the
//!   charset (SIDs below 391 resolve through [`CFF_STANDARD_STRINGS`]; SIDs at
//!   or above 391 through the font's own String INDEX). The predefined
//!   **Expert**/**ExpertSubset** encodings (offset `1`) are not resolved this
//!   wave — rare in practice — and such a glyph falls back to the advance box
//!   at the call site. Documented ceiling, never a crash.

use super::draw_path::Path;
use super::encodings::BaseEncoding;
use super::glyph::{Affine, Reader};
use std::collections::HashMap;

/// Recursion / call-depth guard for `callsubr` / `callgsubr`.
const MAX_CALL_DEPTH: u32 = 60;

/// A byte range `[start, end)` into the CFF image (an INDEX object or subr).
type Span = (usize, usize);

/// A parsed CFF outline source.
#[derive(Clone, Debug)]
pub struct CffProgram {
    data: Vec<u8>,
    /// Per-GID charstring byte ranges.
    charstrings: Vec<Span>,
    /// Global subroutine byte ranges + bias.
    gsubrs: Vec<Span>,
    gbias: i32,
    /// Font-wide (non-CID) local subroutines + bias.
    lsubrs: Vec<Span>,
    lbias: i32,
    nominal_width: f32,
    default_width: f32,
    /// Font-unit -> em transform (the `FontMatrix`, default 0.001 scale).
    matrix: Affine,
    /// `code -> gid` from a custom CFF Encoding (simple fonts). Empty for
    /// predefined encodings / CID fonts.
    encoding: HashMap<u32, u16>,
    /// `name -> gid` from the charset (simple fonts only, built whenever
    /// `encoding` is empty -- i.e. a predefined encoding -- so
    /// [`CffProgram::gid_for_code`] can resolve `code -> name -> gid`). Empty
    /// for CID fonts and for fonts with a custom `Encoding`.
    name_to_gid: HashMap<String, u16>,
    /// `cid -> gid` from the charset (CID-keyed fonts only).
    cid_to_gid: HashMap<u16, u16>,
    /// Whether this is a CID-keyed CFF (`ROS` present).
    is_cid: bool,
    /// Per-GID local subrs for CID fonts (indexed via FDSelect -> FDArray).
    fd_select: Vec<u8>,
    fd_subrs: Vec<(Vec<Span>, i32, f32, f32)>, // (lsubrs, lbias, nominalW, defaultW)
}

impl CffProgram {
    /// Parse a bare CFF font program. Returns `None` on malformed input.
    pub fn parse(bytes: &[u8]) -> Option<CffProgram> {
        if bytes.len() < 4 {
            return None;
        }
        // Header: major, minor, hdrSize, offSize. Everything starts at hdrSize.
        let hdr_size = bytes[2] as usize;
        let mut pos = hdr_size;

        // Name INDEX (skipped), Top DICT INDEX, String INDEX, Global Subr INDEX.
        let (_names, p) = parse_index(bytes, pos)?;
        pos = p;
        let (top_dicts, p) = parse_index(bytes, pos)?;
        pos = p;
        let (strings, p) = parse_index(bytes, pos)?;
        pos = p;
        let (gsubrs, _p) = parse_index(bytes, pos)?;

        let top_span = *top_dicts.first()?;
        let top = parse_dict(&bytes[top_span.0..top_span.1]);

        let cs_off = *top.get(&17)?.first()? as usize;
        let (charstrings, _) = parse_index(bytes, cs_off)?;
        let nglyphs = charstrings.len();
        if nglyphs == 0 {
            return None;
        }

        let charstring_type = top
            .get(&1206)
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(2.0) as i32;
        let matrix = top
            .get(&1207)
            .filter(|m| m.len() == 6)
            .map(|m| Affine {
                a: m[0] as f32,
                b: m[1] as f32,
                c: m[2] as f32,
                d: m[3] as f32,
                e: m[4] as f32,
                f: m[5] as f32,
            })
            .unwrap_or_else(|| Affine::scale(0.001));

        let gbias = subr_bias(gsubrs.len(), charstring_type);

        // Font-wide Private DICT -> local subrs + widths.
        let (lsubrs, nominal_width, default_width) =
            parse_private(bytes, top.get(&18), charstring_type);
        let lbias = subr_bias(lsubrs.len(), charstring_type);

        let is_cid = top.contains_key(&1230); // ROS

        // charset: GID -> SID/CID.
        let charset_off = top.get(&15).and_then(|v| v.first()).copied().unwrap_or(0.0) as usize;
        let gid_to_sid = parse_charset(bytes, charset_off, nglyphs);
        let cid_to_gid = if is_cid {
            let mut m = HashMap::with_capacity(gid_to_sid.len());
            for (gid, &cid) in gid_to_sid.iter().enumerate() {
                m.entry(cid).or_insert(gid as u16);
            }
            m
        } else {
            HashMap::new()
        };

        // Simple-font custom Encoding: code -> GID.
        let encoding = if is_cid {
            HashMap::new()
        } else {
            let enc_off = top.get(&16).and_then(|v| v.first()).copied().unwrap_or(0.0) as usize;
            parse_encoding(bytes, enc_off)
        };

        // `name -> GID` via the charset's SIDs. Two callers need it:
        //
        // * [`CffProgram::gid_for_code`], to chain `code -> name -> GID` for
        //   the (common) predefined-Standard-encoding case.
        // * [`CffProgram::gid_for_name`], used when this program is a
        //   **substitute** for a non-embedded font (see
        //   [`super::standard_font`]) -- there the PDF's own `/Encoding`
        //   names the glyph and the document's GIDs are meaningless, so name
        //   lookup is the only correct selection path.
        //
        // Built for every non-CID font, not only ones with a predefined
        // encoding: a substitute with its own custom `Encoding` table still
        // has to be addressable by name, and it used to come back empty.
        let name_to_gid = if !is_cid {
            let mut m = HashMap::with_capacity(gid_to_sid.len());
            for (gid, &sid) in gid_to_sid.iter().enumerate() {
                if let Some(name) = sid_name(sid, &strings, bytes) {
                    m.entry(name).or_insert(gid as u16);
                }
            }
            m
        } else {
            HashMap::new()
        };

        // CID fonts: FDSelect (GID -> fd) + FDArray (per-fd Private -> local subrs).
        let (fd_select, fd_subrs) = if is_cid {
            let fd_select = top
                .get(&1237)
                .and_then(|v| v.first())
                .map(|&o| parse_fdselect(bytes, o as usize, nglyphs))
                .unwrap_or_default();
            let fd_subrs = top
                .get(&1236)
                .and_then(|v| v.first())
                .map(|&o| parse_fdarray(bytes, o as usize, charstring_type))
                .unwrap_or_default();
            (fd_select, fd_subrs)
        } else {
            (Vec::new(), Vec::new())
        };

        Some(CffProgram {
            data: bytes.to_vec(),
            charstrings,
            gsubrs,
            gbias,
            lsubrs,
            lbias,
            nominal_width,
            default_width,
            matrix,
            encoding,
            name_to_gid,
            cid_to_gid,
            is_cid,
            fd_select,
            fd_subrs,
        })
    }

    /// Whether this CFF is CID-keyed (used by [`super::font`] to route selection).
    pub fn is_cid_keyed(&self) -> bool {
        self.is_cid
    }

    /// Map a CID to a GID (CID-keyed fonts). Falls back to `cid == gid` when the
    /// charset does not list the CID (identity ordering).
    pub fn gid_for_cid(&self, cid: u32) -> u16 {
        if let Some(&g) = self.cid_to_gid.get(&(cid as u16)) {
            g
        } else {
            cid as u16
        }
    }

    /// Map a character code to a GID: through the CFF's custom Encoding when it
    /// has one, otherwise through the predefined-Standard-encoding fallback
    /// chain `code -> name -> gid` (see the module docs). `None` when neither
    /// resolves the code (e.g. a predefined Expert encoding) so the caller can
    /// fall back to the advance box.
    /// `name -> GID` through the charset.
    ///
    /// The selection path for a **substitute** font: when a PDF names a font
    /// it does not embed, the document's own GIDs refer to a program we do
    /// not have, so the only meaningful link between the document and the
    /// substitute is the glyph *name* from the PDF `/Encoding`. Empty for
    /// CID-keyed fonts, whose charset maps to CIDs rather than names.
    pub fn gid_for_name(&self, name: &str) -> Option<u16> {
        self.name_to_gid.get(name).copied()
    }

    pub fn gid_for_code(&self, code: u32) -> Option<u16> {
        if let Some(&g) = self.encoding.get(&code) {
            return Some(g);
        }
        if code < 256 {
            let name = BaseEncoding::Standard.glyph_name(code as u8)?;
            return self.name_to_gid.get(name).copied();
        }
        None
    }

    /// Decode glyph `gid` to an em-space [`Path`] (y-up, 1 em = 1.0), or `None`
    /// for an empty / out-of-range glyph.
    pub fn outline(&self, gid: u16) -> Option<Path> {
        self.run_charstring(gid).and_then(|(path, _)| {
            if super::glyph::path_is_empty(&path) {
                None
            } else {
                Some(path)
            }
        })
    }

    /// The glyph's horizontal advance in **1/1000 em**, as the charstring
    /// itself declares it (the optional leading width operand, relative to the
    /// Private DICT's `nominalWidthX`, defaulting to `defaultWidthX` —
    /// CFF spec §16 / Type2 charstring spec §3.1).
    ///
    /// This is the metric source of last resort for a **substituted** font.
    /// A PDF that names a standard-14 font usually omits `/Widths` entirely
    /// (§9.6.2.2 lets it), which leaves the viewer with no advance at all —
    /// and a zero advance means every glyph is skipped, so the text is not
    /// merely mis-spaced but *invisible*. The bundled faces are
    /// metric-compatible stand-ins for the standard 14, so their own declared
    /// widths are the right answer, and reading them here avoids porting the
    /// AFM tables to get the same numbers.
    pub fn advance_width(&self, gid: u16) -> Option<f32> {
        self.run_charstring(gid).map(|(_, w)| w)
    }

    /// Interpret one glyph's charstring, returning its em-space path and the
    /// advance width the charstring declared.
    fn run_charstring(&self, gid: u16) -> Option<(Path, f32)> {
        let g = gid as usize;
        let span = *self.charstrings.get(g)?;

        // CID fonts pick local subrs per glyph via FDSelect -> FDArray.
        let (lsubrs, lbias, nominal_width, default_width) =
            if self.is_cid && !self.fd_subrs.is_empty() {
                let fd = self.fd_select.get(g).copied().unwrap_or(0) as usize;
                match self.fd_subrs.get(fd) {
                    Some((ls, lb, nw, dw)) => (ls.as_slice(), *lb, *nw, *dw),
                    None => (
                        self.lsubrs.as_slice(),
                        self.lbias,
                        self.nominal_width,
                        self.default_width,
                    ),
                }
            } else {
                (
                    self.lsubrs.as_slice(),
                    self.lbias,
                    self.nominal_width,
                    self.default_width,
                )
            };

        let mut ctx = T2Ctx {
            data: &self.data,
            path: Path::new(),
            x: 0.0,
            y: 0.0,
            matrix: self.matrix,
            open: false,
            have_width: false,
            nstems: 0,
            width: default_width,
            nominal_width,
            gsubrs: &self.gsubrs,
            gbias: self.gbias,
            lsubrs,
            lbias,
            stack: Vec::with_capacity(48),
            ended: false,
        };
        ctx.run(span, 0);
        if ctx.open {
            ctx.path.close();
        }
        Some((ctx.path, ctx.width))
    }
}

// ---------------------------------------------------------------------------
// The Type2 charstring interpreter
// ---------------------------------------------------------------------------

/// Execution context for one glyph's Type2 charstring (+ its subrs).
struct T2Ctx<'a> {
    data: &'a [u8],
    path: Path,
    x: f32,
    y: f32,
    matrix: Affine,
    open: bool,
    have_width: bool,
    nstems: u32,
    width: f32,
    nominal_width: f32,
    gsubrs: &'a [Span],
    gbias: i32,
    lsubrs: &'a [Span],
    lbias: i32,
    stack: Vec<f32>,
    ended: bool,
}

impl T2Ctx<'_> {
    /// Emit the current point (font units) transformed to em space.
    fn tx(&self, x: f32, y: f32) -> (f32, f32) {
        self.matrix.apply(x, y)
    }

    fn moveto(&mut self, dx: f32, dy: f32) {
        if self.open {
            self.path.close();
        }
        self.x += dx;
        self.y += dy;
        let (px, py) = self.tx(self.x, self.y);
        self.path.move_to(px, py);
        self.open = true;
    }

    fn lineto(&mut self, dx: f32, dy: f32) {
        self.x += dx;
        self.y += dy;
        let (px, py) = self.tx(self.x, self.y);
        self.path.line_to(px, py);
    }

    fn curveto(&mut self, dx1: f32, dy1: f32, dx2: f32, dy2: f32, dx3: f32, dy3: f32) {
        let x1 = self.x + dx1;
        let y1 = self.y + dy1;
        let x2 = x1 + dx2;
        let y2 = y1 + dy2;
        self.x = x2 + dx3;
        self.y = y2 + dy3;
        let (c1x, c1y) = self.tx(x1, y1);
        let (c2x, c2y) = self.tx(x2, y2);
        let (ex, ey) = self.tx(self.x, self.y);
        self.path.curve_to(c1x, c1y, c2x, c2y, ex, ey);
    }

    /// Take the leading width operand if present, for a stack-clearing operator
    /// that expects `even_args` arguments (moves/endchar pass their exact count).
    fn take_width(&mut self, expected: usize) {
        if !self.have_width && self.stack.len() > expected {
            // The extra leading operand is the (nominal-relative) width.
            self.width = self.nominal_width + self.stack.remove(0);
        }
        self.have_width = true;
    }

    fn take_width_stem(&mut self) {
        if !self.have_width && self.stack.len() % 2 == 1 {
            self.width = self.nominal_width + self.stack.remove(0);
        }
        self.have_width = true;
    }

    // MuPDF: execute_charstring (subset-cff.c:935) — same operator dispatch and
    // subr-bias handling; here it *builds* the outline instead of marking usage.
    fn run(&mut self, span: Span, depth: u32) {
        if depth > MAX_CALL_DEPTH || self.ended {
            return;
        }
        let (mut pc, end) = span;
        let end = end.min(self.data.len());
        while pc < end && !self.ended {
            let b0 = self.data[pc];
            pc += 1;
            if b0 >= 32 || b0 == 28 {
                // An operand.
                let (val, np) = parse_operand(self.data, pc - 1);
                self.stack.push(val);
                pc = np;
                continue;
            }
            match b0 {
                1 | 3 | 18 | 23 => {
                    // hstem/vstem/hstemhm/vstemhm.
                    self.take_width_stem();
                    self.nstems += (self.stack.len() / 2) as u32;
                    self.stack.clear();
                }
                19 | 20 => {
                    // hintmask/cntrmask: trailing implicit vstem args, then mask.
                    self.take_width_stem();
                    self.nstems += (self.stack.len() / 2) as u32;
                    self.stack.clear();
                    pc += self.nstems.div_ceil(8) as usize;
                }
                21 => {
                    // rmoveto.
                    self.take_width(2);
                    let dy = self.stack.pop().unwrap_or(0.0);
                    let dx = self.stack.pop().unwrap_or(0.0);
                    self.moveto(dx, dy);
                    self.stack.clear();
                }
                22 => {
                    // hmoveto.
                    self.take_width(1);
                    let dx = self.stack.pop().unwrap_or(0.0);
                    self.moveto(dx, 0.0);
                    self.stack.clear();
                }
                4 => {
                    // vmoveto.
                    self.take_width(1);
                    let dy = self.stack.pop().unwrap_or(0.0);
                    self.moveto(0.0, dy);
                    self.stack.clear();
                }
                5 => {
                    // rlineto: pairs.
                    let n = self.stack.len() / 2;
                    for i in 0..n {
                        self.lineto(self.stack[i * 2], self.stack[i * 2 + 1]);
                    }
                    self.stack.clear();
                }
                6 => self.hv_lineto(true),  // hlineto
                7 => self.hv_lineto(false), // vlineto
                8 => {
                    // rrcurveto: sextuples.
                    let n = self.stack.len() / 6;
                    for i in 0..n {
                        let s = &self.stack[i * 6..i * 6 + 6];
                        self.curveto(s[0], s[1], s[2], s[3], s[4], s[5]);
                    }
                    self.stack.clear();
                }
                24 => {
                    // rcurveline: curves, then a final line.
                    let ncurves = (self.stack.len().saturating_sub(2)) / 6;
                    for i in 0..ncurves {
                        let s = &self.stack[i * 6..i * 6 + 6];
                        self.curveto(s[0], s[1], s[2], s[3], s[4], s[5]);
                    }
                    let base = ncurves * 6;
                    if base + 1 < self.stack.len() {
                        self.lineto(self.stack[base], self.stack[base + 1]);
                    }
                    self.stack.clear();
                }
                25 => {
                    // rlinecurve: lines, then a final curve.
                    let nlines = (self.stack.len().saturating_sub(6)) / 2;
                    for i in 0..nlines {
                        self.lineto(self.stack[i * 2], self.stack[i * 2 + 1]);
                    }
                    let base = nlines * 2;
                    if base + 5 < self.stack.len() {
                        let s = &self.stack[base..base + 6];
                        self.curveto(s[0], s[1], s[2], s[3], s[4], s[5]);
                    }
                    self.stack.clear();
                }
                26 => self.vv_curveto(),
                27 => self.hh_curveto(),
                30 => self.vh_curveto(true),  // vhcurveto
                31 => self.vh_curveto(false), // hvcurveto
                10 => {
                    // callsubr (local).
                    if let Some(i) = self.stack.pop() {
                        let idx = i as i32 + self.lbias;
                        if let Some(&sp) = self.lsubrs.get(idx.max(0) as usize) {
                            self.run(sp, depth + 1);
                        }
                    }
                }
                29 => {
                    // callgsubr (global).
                    if let Some(i) = self.stack.pop() {
                        let idx = i as i32 + self.gbias;
                        if let Some(&sp) = self.gsubrs.get(idx.max(0) as usize) {
                            self.run(sp, depth + 1);
                        }
                    }
                }
                11 => return, // return
                14 => {
                    // endchar (width parsing; deprecated seac args ignored).
                    if !self.have_width && (self.stack.len() == 1 || self.stack.len() == 5) {
                        self.width = self.nominal_width + self.stack.remove(0);
                    }
                    self.have_width = true;
                    self.ended = true;
                    return;
                }
                12 => {
                    // Two-byte escape operators.
                    if pc >= end {
                        return;
                    }
                    let b1 = self.data[pc];
                    pc += 1;
                    self.escape(b1);
                }
                _ => {
                    // Reserved / unhandled: clear and continue safely.
                    self.stack.clear();
                }
            }
        }
    }

    // hlineto (start_horizontal=true) / vlineto: alternate axis each segment.
    fn hv_lineto(&mut self, mut horizontal: bool) {
        for i in 0..self.stack.len() {
            let d = self.stack[i];
            if horizontal {
                self.lineto(d, 0.0);
            } else {
                self.lineto(0.0, d);
            }
            horizontal = !horizontal;
        }
        self.stack.clear();
    }

    // vvcurveto: {dx1?} {dya dxb dyb dyc}+ — vertical start & end tangents.
    fn vv_curveto(&mut self) {
        let mut i = 0;
        let mut dx1 = 0.0;
        if self.stack.len() % 4 == 1 {
            dx1 = self.stack[0];
            i = 1;
        }
        while i + 3 < self.stack.len() {
            let dya = self.stack[i];
            let dxb = self.stack[i + 1];
            let dyb = self.stack[i + 2];
            let dyc = self.stack[i + 3];
            self.curveto(dx1, dya, dxb, dyb, 0.0, dyc);
            dx1 = 0.0;
            i += 4;
        }
        self.stack.clear();
    }

    // hhcurveto: {dy1?} {dxa dxb dyb dxc}+ — horizontal start & end tangents.
    fn hh_curveto(&mut self) {
        let mut i = 0;
        let mut dy1 = 0.0;
        if self.stack.len() % 4 == 1 {
            dy1 = self.stack[0];
            i = 1;
        }
        while i + 3 < self.stack.len() {
            let dxa = self.stack[i];
            let dxb = self.stack[i + 1];
            let dyb = self.stack[i + 2];
            let dxc = self.stack[i + 3];
            self.curveto(dxa, dy1, dxb, dyb, dxc, 0.0);
            dy1 = 0.0;
            i += 4;
        }
        self.stack.clear();
    }

    // vhcurveto (start_vertical=true) / hvcurveto: alternating tangent curves,
    // with an optional trailing 5th argument on the last curve.
    fn vh_curveto(&mut self, mut vertical: bool) {
        let n = self.stack.len();
        let mut i = 0;
        while i + 3 < n {
            let remaining = n - i;
            // The last group may carry a 5th value (df), used on the free axis.
            let last = remaining < 8;
            let df = if last && remaining == 5 {
                self.stack[i + 4]
            } else {
                0.0
            };
            if vertical {
                // vertical start: (0, dy1) (dx2, dy2) (dx3, df)
                self.curveto(
                    0.0,
                    self.stack[i],
                    self.stack[i + 1],
                    self.stack[i + 2],
                    self.stack[i + 3],
                    df,
                );
            } else {
                // horizontal start: (dx1, 0) (dx2, dy2) (df, dy3)
                self.curveto(
                    self.stack[i],
                    0.0,
                    self.stack[i + 1],
                    self.stack[i + 2],
                    df,
                    self.stack[i + 3],
                );
            }
            vertical = !vertical;
            i += 4;
        }
        self.stack.clear();
    }

    /// Two-byte (escape) operators — the flex family; arithmetic/logic ops are
    /// not needed for outline construction and clear the stack safely.
    fn escape(&mut self, op: u8) {
        match op {
            34 => {
                // hflex: dx1 dx2 dy2 dx3 dx4 dx5 dx6.
                let s = &self.stack;
                if s.len() >= 7 {
                    let (dx1, dx2, dy2, dx3, dx4, dx5, dx6) =
                        (s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
                    self.curveto(dx1, 0.0, dx2, dy2, dx3, 0.0);
                    self.curveto(dx4, 0.0, dx5, -dy2, dx6, 0.0);
                }
            }
            36 => {
                // hflex1: dx1 dy1 dx2 dy2 dx3 dx4 dx5 dy5 dx6.
                let s = &self.stack;
                if s.len() >= 9 {
                    let (dx1, dy1, dx2, dy2, dx3) = (s[0], s[1], s[2], s[3], s[4]);
                    let (dx4, dx5, dy5, dx6) = (s[5], s[6], s[7], s[8]);
                    self.curveto(dx1, dy1, dx2, dy2, dx3, 0.0);
                    self.curveto(dx4, 0.0, dx5, dy5, dx6, -(dy1 + dy2 + dy5));
                }
            }
            35 => {
                // flex: two curves + a flex-depth arg (ignored).
                if self.stack.len() >= 12 {
                    let s: [f32; 12] = std::array::from_fn(|i| self.stack[i]);
                    self.curveto(s[0], s[1], s[2], s[3], s[4], s[5]);
                    self.curveto(s[6], s[7], s[8], s[9], s[10], s[11]);
                }
            }
            37 if self.stack.len() >= 11 => {
                // flex1: dx1 dy1 dx2 dy2 dx3 dy3 dx4 dy4 dx5 dy5 d6.
                let s: [f32; 11] = std::array::from_fn(|i| self.stack[i]);
                let dx = s[0] + s[2] + s[4] + s[6] + s[8];
                let dy = s[1] + s[3] + s[5] + s[7] + s[9];
                self.curveto(s[0], s[1], s[2], s[3], s[4], s[5]);
                if dx.abs() > dy.abs() {
                    self.curveto(s[6], s[7], s[8], s[9], s[10], -dy);
                } else {
                    self.curveto(s[6], s[7], s[8], s[9], -dx, s[10]);
                }
            }
            _ => {}
        }
        self.stack.clear();
    }
}

/// Parse one Type2 numeric operand at `data[pos]`, returning `(value, next_pos)`.
fn parse_operand(data: &[u8], pos: usize) -> (f32, usize) {
    let b0 = data[pos];
    let g = |i: usize| data.get(i).copied().unwrap_or(0) as i32;
    match b0 {
        28 => {
            let v = ((g(pos + 1) << 8) | g(pos + 2)) as i16;
            (v as f32, pos + 3)
        }
        32..=246 => (b0 as f32 - 139.0, pos + 1),
        247..=250 => (((b0 as i32 - 247) * 256 + g(pos + 1) + 108) as f32, pos + 2),
        251..=254 => (
            -(b0 as f32 - 251.0) * 256.0 - g(pos + 1) as f32 - 108.0,
            pos + 2,
        ),
        255 => {
            let v = (g(pos + 1) << 24) | (g(pos + 2) << 16) | (g(pos + 3) << 8) | g(pos + 4);
            (v as f32 / 65536.0, pos + 5)
        }
        _ => (0.0, pos + 1),
    }
}

// ---------------------------------------------------------------------------
// CFF container structures (adapted from subset-cff.c)
// ---------------------------------------------------------------------------

// MuPDF: subr_bias (subset-cff.c:245).
/// The subroutine index bias for Type2 (Type1 uses 0).
fn subr_bias(count: usize, charstring_type: i32) -> i32 {
    if charstring_type == 1 {
        0
    } else if count < 1240 {
        107
    } else if count < 33900 {
        1131
    } else {
        32768
    }
}

// MuPDF: the INDEX reader inside subset-cff.c (parse_index).
/// Parse a CFF INDEX starting at `pos`, returning each element's byte range and
/// the position just past the INDEX.
fn parse_index(bytes: &[u8], pos: usize) -> Option<(Vec<Span>, usize)> {
    let mut r = Reader::at(bytes, pos);
    if r.remaining() < 2 {
        return None;
    }
    let count = r.u16() as usize;
    if count == 0 {
        return Some((Vec::new(), pos + 2));
    }
    let off_size = r.u8();
    if !(1..=4).contains(&off_size) {
        return None;
    }
    let mut offsets = Vec::with_capacity(count + 1);
    for _ in 0..=count {
        offsets.push(r.offset(off_size));
    }
    // Object data is addressed relative to the byte before the first object.
    let data_base = r.pos() as i64 - 1;
    let mut spans = Vec::with_capacity(count);
    for i in 0..count {
        let s = data_base + offsets[i] as i64;
        let e = data_base + offsets[i + 1] as i64;
        if s < 0 || e < s || e as usize > bytes.len() {
            return None;
        }
        spans.push((s as usize, e as usize));
    }
    let end = (data_base + *offsets.last().unwrap() as i64) as usize;
    Some((spans, end))
}

/// Parse a CFF DICT into `operator -> operands`. Escape operators (`12 b`) key as
/// `1200 + b`. Reals (op 30) are decoded from packed BCD.
fn parse_dict(bytes: &[u8]) -> HashMap<u16, Vec<f64>> {
    let mut dict = HashMap::new();
    let mut operands: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        if b0 <= 21 {
            // Operator.
            let key = if b0 == 12 {
                i += 1;
                1200 + *bytes.get(i).unwrap_or(&0) as u16
            } else {
                b0 as u16
            };
            i += 1;
            dict.insert(key, std::mem::take(&mut operands));
        } else if b0 == 28 {
            let v = (((*bytes.get(i + 1).unwrap_or(&0) as i32) << 8)
                | *bytes.get(i + 2).unwrap_or(&0) as i32) as i16;
            operands.push(v as f64);
            i += 3;
        } else if b0 == 29 {
            let v = ((*bytes.get(i + 1).unwrap_or(&0) as i32) << 24)
                | ((*bytes.get(i + 2).unwrap_or(&0) as i32) << 16)
                | ((*bytes.get(i + 3).unwrap_or(&0) as i32) << 8)
                | *bytes.get(i + 4).unwrap_or(&0) as i32;
            operands.push(v as f64);
            i += 5;
        } else if b0 == 30 {
            // Real number: packed BCD nibbles.
            let (v, ni) = parse_real(bytes, i + 1);
            operands.push(v);
            i = ni;
        } else if (32..=246).contains(&b0) {
            operands.push(b0 as f64 - 139.0);
            i += 1;
        } else if (247..=250).contains(&b0) {
            operands.push(
                ((b0 as i32 - 247) * 256 + *bytes.get(i + 1).unwrap_or(&0) as i32 + 108) as f64,
            );
            i += 2;
        } else if (251..=254).contains(&b0) {
            operands.push(
                (-(b0 as i32 - 251) * 256 - *bytes.get(i + 1).unwrap_or(&0) as i32 - 108) as f64,
            );
            i += 2;
        } else {
            i += 1;
        }
    }
    dict
}

/// Decode a CFF DICT real number (packed BCD) starting at `pos`, returning
/// `(value, next_pos)`.
fn parse_real(bytes: &[u8], mut pos: usize) -> (f64, usize) {
    let mut s = String::new();
    'outer: while pos < bytes.len() {
        let byte = bytes[pos];
        pos += 1;
        for nibble in [byte >> 4, byte & 0x0f] {
            match nibble {
                0..=9 => s.push((b'0' + nibble) as char),
                0x0a => s.push('.'),
                0x0b => s.push('E'),
                0x0c => s.push_str("E-"),
                0x0e => s.push('-'),
                0x0f => break 'outer,
                _ => {}
            }
        }
    }
    (s.parse::<f64>().unwrap_or(0.0), pos)
}

/// Parse a Private DICT (`[size, offset]` from the Top DICT) into
/// `(local_subrs, nominalWidthX, defaultWidthX)`.
fn parse_private(
    bytes: &[u8],
    priv_entry: Option<&Vec<f64>>,
    charstring_type: i32,
) -> (Vec<Span>, f32, f32) {
    let empty = (Vec::new(), 0.0, 0.0);
    let Some(p) = priv_entry else { return empty };
    if p.len() < 2 {
        return empty;
    }
    let size = p[0] as usize;
    let off = p[1] as usize;
    let Some(pd) = bytes.get(off..off + size) else {
        return empty;
    };
    let dict = parse_dict(pd);
    let nominal_width = dict
        .get(&21)
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(0.0) as f32;
    let default_width = dict
        .get(&20)
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(0.0) as f32;
    // Local Subrs offset (op 19) is relative to the Private DICT start.
    let lsubrs = dict
        .get(&19)
        .and_then(|v| v.first())
        .and_then(|&rel| parse_index(bytes, off + rel as usize).map(|(s, _)| s))
        .unwrap_or_default();
    let _ = charstring_type;
    (lsubrs, nominal_width, default_width)
}

/// Parse the charset into `gid -> SID/CID` (index 0 is `.notdef` = 0).
fn parse_charset(bytes: &[u8], off: usize, nglyphs: usize) -> Vec<u16> {
    let mut out = vec![0u16; nglyphs];
    // Predefined charsets (0/1/2) are not expanded here; identity is a safe
    // fallback for CID fonts (they normally embed an explicit charset).
    if off <= 2 {
        for (gid, slot) in out.iter_mut().enumerate() {
            *slot = gid as u16;
        }
        return out;
    }
    let mut r = Reader::at(bytes, off);
    let format = r.u8();
    let mut gid = 1usize;
    match format {
        0 => {
            while gid < nglyphs {
                out[gid] = r.u16();
                gid += 1;
            }
        }
        1 => {
            while gid < nglyphs && r.remaining() >= 3 {
                let first = r.u16();
                let nleft = r.u8() as usize;
                for k in 0..=nleft {
                    if gid >= nglyphs {
                        break;
                    }
                    out[gid] = first + k as u16;
                    gid += 1;
                }
            }
        }
        2 => {
            while gid < nglyphs && r.remaining() >= 4 {
                let first = r.u16();
                let nleft = r.u16() as usize;
                for k in 0..=nleft {
                    if gid >= nglyphs {
                        break;
                    }
                    out[gid] = first + k as u16;
                    gid += 1;
                }
            }
        }
        _ => {}
    }
    out
}

/// The CFF Standard Strings (Adobe Technical Note #5176, Appendix A): SID 0
/// (`.notdef`) through SID 390 (`Semibold`). A charset SID at or above 391
/// instead indexes the font's own String INDEX ([`sid_name`]). Cross-checked
/// against fontTools' `cffLib.cffStandardStrings` (BSD-3-Clause, the same list
/// verbatim -- these are the fixed strings the CFF spec itself defines, not
/// fontTools' own expression).
static CFF_STANDARD_STRINGS: [&str; 391] = [
    ".notdef",
    "space",
    "exclam",
    "quotedbl",
    "numbersign",
    "dollar",
    "percent",
    "ampersand",
    "quoteright",
    "parenleft",
    "parenright",
    "asterisk",
    "plus",
    "comma",
    "hyphen",
    "period",
    "slash",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "colon",
    "semicolon",
    "less",
    "equal",
    "greater",
    "question",
    "at",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "bracketleft",
    "backslash",
    "bracketright",
    "asciicircum",
    "underscore",
    "quoteleft",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "braceleft",
    "bar",
    "braceright",
    "asciitilde",
    "exclamdown",
    "cent",
    "sterling",
    "fraction",
    "yen",
    "florin",
    "section",
    "currency",
    "quotesingle",
    "quotedblleft",
    "guillemotleft",
    "guilsinglleft",
    "guilsinglright",
    "fi",
    "fl",
    "endash",
    "dagger",
    "daggerdbl",
    "periodcentered",
    "paragraph",
    "bullet",
    "quotesinglbase",
    "quotedblbase",
    "quotedblright",
    "guillemotright",
    "ellipsis",
    "perthousand",
    "questiondown",
    "grave",
    "acute",
    "circumflex",
    "tilde",
    "macron",
    "breve",
    "dotaccent",
    "dieresis",
    "ring",
    "cedilla",
    "hungarumlaut",
    "ogonek",
    "caron",
    "emdash",
    "AE",
    "ordfeminine",
    "Lslash",
    "Oslash",
    "OE",
    "ordmasculine",
    "ae",
    "dotlessi",
    "lslash",
    "oslash",
    "oe",
    "germandbls",
    "onesuperior",
    "logicalnot",
    "mu",
    "trademark",
    "Eth",
    "onehalf",
    "plusminus",
    "Thorn",
    "onequarter",
    "divide",
    "brokenbar",
    "degree",
    "thorn",
    "threequarters",
    "twosuperior",
    "registered",
    "minus",
    "eth",
    "multiply",
    "threesuperior",
    "copyright",
    "Aacute",
    "Acircumflex",
    "Adieresis",
    "Agrave",
    "Aring",
    "Atilde",
    "Ccedilla",
    "Eacute",
    "Ecircumflex",
    "Edieresis",
    "Egrave",
    "Iacute",
    "Icircumflex",
    "Idieresis",
    "Igrave",
    "Ntilde",
    "Oacute",
    "Ocircumflex",
    "Odieresis",
    "Ograve",
    "Otilde",
    "Scaron",
    "Uacute",
    "Ucircumflex",
    "Udieresis",
    "Ugrave",
    "Yacute",
    "Ydieresis",
    "Zcaron",
    "aacute",
    "acircumflex",
    "adieresis",
    "agrave",
    "aring",
    "atilde",
    "ccedilla",
    "eacute",
    "ecircumflex",
    "edieresis",
    "egrave",
    "iacute",
    "icircumflex",
    "idieresis",
    "igrave",
    "ntilde",
    "oacute",
    "ocircumflex",
    "odieresis",
    "ograve",
    "otilde",
    "scaron",
    "uacute",
    "ucircumflex",
    "udieresis",
    "ugrave",
    "yacute",
    "ydieresis",
    "zcaron",
    "exclamsmall",
    "Hungarumlautsmall",
    "dollaroldstyle",
    "dollarsuperior",
    "ampersandsmall",
    "Acutesmall",
    "parenleftsuperior",
    "parenrightsuperior",
    "twodotenleader",
    "onedotenleader",
    "zerooldstyle",
    "oneoldstyle",
    "twooldstyle",
    "threeoldstyle",
    "fouroldstyle",
    "fiveoldstyle",
    "sixoldstyle",
    "sevenoldstyle",
    "eightoldstyle",
    "nineoldstyle",
    "commasuperior",
    "threequartersemdash",
    "periodsuperior",
    "questionsmall",
    "asuperior",
    "bsuperior",
    "centsuperior",
    "dsuperior",
    "esuperior",
    "isuperior",
    "lsuperior",
    "msuperior",
    "nsuperior",
    "osuperior",
    "rsuperior",
    "ssuperior",
    "tsuperior",
    "ff",
    "ffi",
    "ffl",
    "parenleftinferior",
    "parenrightinferior",
    "Circumflexsmall",
    "hyphensuperior",
    "Gravesmall",
    "Asmall",
    "Bsmall",
    "Csmall",
    "Dsmall",
    "Esmall",
    "Fsmall",
    "Gsmall",
    "Hsmall",
    "Ismall",
    "Jsmall",
    "Ksmall",
    "Lsmall",
    "Msmall",
    "Nsmall",
    "Osmall",
    "Psmall",
    "Qsmall",
    "Rsmall",
    "Ssmall",
    "Tsmall",
    "Usmall",
    "Vsmall",
    "Wsmall",
    "Xsmall",
    "Ysmall",
    "Zsmall",
    "colonmonetary",
    "onefitted",
    "rupiah",
    "Tildesmall",
    "exclamdownsmall",
    "centoldstyle",
    "Lslashsmall",
    "Scaronsmall",
    "Zcaronsmall",
    "Dieresissmall",
    "Brevesmall",
    "Caronsmall",
    "Dotaccentsmall",
    "Macronsmall",
    "figuredash",
    "hypheninferior",
    "Ogoneksmall",
    "Ringsmall",
    "Cedillasmall",
    "questiondownsmall",
    "oneeighth",
    "threeeighths",
    "fiveeighths",
    "seveneighths",
    "onethird",
    "twothirds",
    "zerosuperior",
    "foursuperior",
    "fivesuperior",
    "sixsuperior",
    "sevensuperior",
    "eightsuperior",
    "ninesuperior",
    "zeroinferior",
    "oneinferior",
    "twoinferior",
    "threeinferior",
    "fourinferior",
    "fiveinferior",
    "sixinferior",
    "seveninferior",
    "eightinferior",
    "nineinferior",
    "centinferior",
    "dollarinferior",
    "periodinferior",
    "commainferior",
    "Agravesmall",
    "Aacutesmall",
    "Acircumflexsmall",
    "Atildesmall",
    "Adieresissmall",
    "Aringsmall",
    "AEsmall",
    "Ccedillasmall",
    "Egravesmall",
    "Eacutesmall",
    "Ecircumflexsmall",
    "Edieresissmall",
    "Igravesmall",
    "Iacutesmall",
    "Icircumflexsmall",
    "Idieresissmall",
    "Ethsmall",
    "Ntildesmall",
    "Ogravesmall",
    "Oacutesmall",
    "Ocircumflexsmall",
    "Otildesmall",
    "Odieresissmall",
    "OEsmall",
    "Oslashsmall",
    "Ugravesmall",
    "Uacutesmall",
    "Ucircumflexsmall",
    "Udieresissmall",
    "Yacutesmall",
    "Thornsmall",
    "Ydieresissmall",
    "001.000",
    "001.001",
    "001.002",
    "001.003",
    "Black",
    "Bold",
    "Book",
    "Light",
    "Medium",
    "Regular",
    "Roman",
    "Semibold",
];

/// Resolve a charset SID to a glyph name: the fixed [`CFF_STANDARD_STRINGS`]
/// table for SID < 391, otherwise the font's own String INDEX at
/// `sid - 391`.
fn sid_name(sid: u16, strings: &[Span], data: &[u8]) -> Option<String> {
    if (sid as usize) < CFF_STANDARD_STRINGS.len() {
        return Some(CFF_STANDARD_STRINGS[sid as usize].to_string());
    }
    let (s, e) = *strings.get(sid as usize - CFF_STANDARD_STRINGS.len())?;
    Some(String::from_utf8_lossy(&data[s..e]).into_owned())
}

/// Parse a custom CFF Encoding (`code -> gid`). Predefined encodings (offset
/// 0/1) return an empty map (the caller falls back — see the module ceiling).
fn parse_encoding(bytes: &[u8], off: usize) -> HashMap<u32, u16> {
    let mut map = HashMap::new();
    if off <= 1 {
        return map; // predefined Standard/Expert encoding: not expanded
    }
    let mut r = Reader::at(bytes, off);
    let format = r.u8();
    match format & 0x7f {
        0 => {
            let ncodes = r.u8() as usize;
            for gid in 1..=ncodes {
                let code = r.u8() as u32;
                map.entry(code).or_insert(gid as u16);
            }
        }
        1 => {
            let nranges = r.u8() as usize;
            let mut gid = 1u16;
            for _ in 0..nranges {
                let first = r.u8() as u32;
                let nleft = r.u8() as u32;
                for c in first..=first + nleft {
                    map.entry(c).or_insert(gid);
                    gid += 1;
                }
            }
        }
        _ => {}
    }
    map
}

/// Parse FDSelect into `gid -> fd index`.
fn parse_fdselect(bytes: &[u8], off: usize, nglyphs: usize) -> Vec<u8> {
    let mut out = vec![0u8; nglyphs];
    let mut r = Reader::at(bytes, off);
    match r.u8() {
        0 => {
            for slot in out.iter_mut() {
                *slot = r.u8();
            }
        }
        3 => {
            let nranges = r.u16() as usize;
            let mut ranges = Vec::with_capacity(nranges);
            for _ in 0..nranges {
                let first = r.u16() as usize;
                let fd = r.u8();
                ranges.push((first, fd));
            }
            let sentinel = r.u16() as usize;
            for w in 0..ranges.len() {
                let (first, fd) = ranges[w];
                let next = if w + 1 < ranges.len() {
                    ranges[w + 1].0
                } else {
                    sentinel
                };
                for slot in out.iter_mut().take(next.min(nglyphs)).skip(first) {
                    *slot = fd;
                }
            }
        }
        _ => {}
    }
    out
}

/// Parse the FDArray (an INDEX of font DICTs) into per-fd
/// `(local_subrs, lbias, nominalWidthX, defaultWidthX)`.
fn parse_fdarray(
    bytes: &[u8],
    off: usize,
    charstring_type: i32,
) -> Vec<(Vec<Span>, i32, f32, f32)> {
    let Some((dicts, _)) = parse_index(bytes, off) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(dicts.len());
    for (s, e) in dicts {
        let fd = parse_dict(&bytes[s..e]);
        let (lsubrs, nominal, default) = parse_private(bytes, fd.get(&18), charstring_type);
        let lbias = subr_bias(lsubrs.len(), charstring_type);
        out.push((lsubrs, lbias, nominal, default));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::Matrix;

    /// Encode a CFF INDEX from its element byte-strings.
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

    /// Assemble a bare CFF with the given per-GID charstrings (gid 0 = `.notdef`).
    fn build_cff(charstrings: &[Vec<u8>]) -> Vec<u8> {
        let header = vec![1u8, 0, 4, 1];
        let name = cff_index(&[b"KOPITEST".to_vec()]);
        let strings = cff_index(&[]);
        let gsubr = cff_index(&[]);
        // Top DICT: `<offset> 17` (CharStrings), offset in the fixed 5-byte op-29
        // 32-bit form so the DICT length is independent of the offset value.
        let make_top = |cs_off: u32| -> Vec<u8> {
            let mut d = vec![29u8];
            d.extend_from_slice(&cs_off.to_be_bytes());
            d.push(17);
            cff_index(&[d])
        };
        let top_len = make_top(0).len();
        let cs_off = (header.len() + name.len() + top_len + strings.len() + gsubr.len()) as u32;
        let top = make_top(cs_off);
        let cs_index = cff_index(charstrings);

        let mut out = Vec::new();
        out.extend_from_slice(&header);
        out.extend_from_slice(&name);
        out.extend_from_slice(&top);
        out.extend_from_slice(&strings);
        out.extend_from_slice(&gsubr);
        out.extend_from_slice(&cs_index);
        out
    }

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

    /// A triangle charstring: moveto(100,0), line to (400,0), line to (250,600).
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
    fn type2_triangle_decodes_to_expected_bbox() {
        let cff = build_cff(&[vec![14], triangle_charstring()]);
        let prog = CffProgram::parse(&cff).expect("parse cff");
        assert!(!prog.is_cid_keyed());
        let path = prog.outline(1).expect("triangle outline");
        // FontMatrix default 0.001: font (100..400, 0..600) -> em (0.1..0.4, 0..0.6).
        let (x0, y0, x1, y1) = bbox(&path);
        assert!((x0 - 0.1).abs() < 1e-4, "x0 {x0}");
        assert!(y0.abs() < 1e-4, "y0 {y0}");
        assert!((x1 - 0.4).abs() < 1e-4, "x1 {x1}");
        assert!((y1 - 0.6).abs() < 1e-4, "y1 {y1}");
    }

    #[test]
    fn type2_triangle_is_single_closed_contour() {
        let cff = build_cff(&[vec![14], triangle_charstring()]);
        let prog = CffProgram::parse(&cff).unwrap();
        let path = prog.outline(1).unwrap();
        let polys = path.flatten(Matrix::IDENTITY);
        assert_eq!(polys.len(), 1, "one contour");
        assert!(polys[0].len() >= 3, "triangle has >= 3 vertices");
    }

    #[test]
    fn empty_charstring_is_none_no_panic() {
        let cff = build_cff(&[vec![14], vec![14]]);
        let prog = CffProgram::parse(&cff).unwrap();
        assert!(prog.outline(0).is_none());
        assert!(prog.outline(1).is_none());
        assert!(prog.outline(50).is_none()); // out of range, no panic
    }

    #[test]
    fn predefined_standard_encoding_resolves_via_charset() {
        // No custom Encoding (Top DICT omits op 16 -> predefined Standard) and
        // an explicit charset mapping gid 1 -> SID 34 ("A", per
        // CFF_STANDARD_STRINGS). `gid_for_code(0x41)` should chain
        // code 'A' -> name "A" -> SID 34 -> gid 1 through the charset.
        let header = vec![1u8, 0, 4, 1];
        let name = cff_index(&[b"KOPITEST".to_vec()]);
        let strings = cff_index(&[]);
        let gsubr = cff_index(&[]);
        let cs_index = cff_index(&[vec![14], triangle_charstring()]);
        let charset_bytes = vec![0u8, 0x00, 34]; // format 0, gid1 -> SID 34

        let make_top = |charset_off: u32, cs_off: u32| -> Vec<u8> {
            let mut d = Vec::new();
            d.push(29u8);
            d.extend_from_slice(&charset_off.to_be_bytes());
            d.push(15); // charset
            d.push(29u8);
            d.extend_from_slice(&cs_off.to_be_bytes());
            d.push(17); // CharStrings
            d
        };
        let top_len = cff_index(&[make_top(0, 0)]).len();
        let base = (header.len() + name.len() + top_len + strings.len() + gsubr.len()) as u32;
        let charset_off = base;
        let cs_off = charset_off + charset_bytes.len() as u32;
        let top = cff_index(&[make_top(charset_off, cs_off)]);

        let mut cff = Vec::new();
        cff.extend_from_slice(&header);
        cff.extend_from_slice(&name);
        cff.extend_from_slice(&top);
        cff.extend_from_slice(&strings);
        cff.extend_from_slice(&gsubr);
        cff.extend_from_slice(&charset_bytes);
        cff.extend_from_slice(&cs_index);

        let prog = CffProgram::parse(&cff).expect("parse cff");
        assert_eq!(prog.gid_for_code(0x41), Some(1), "'A' -> SID 34 -> gid 1");
        let path = prog
            .outline(prog.gid_for_code(0x41).unwrap())
            .expect("outline");
        let (x0, y0, x1, y1) = bbox(&path);
        assert!(
            (x0 - 0.1).abs() < 1e-4
                && y0.abs() < 1e-4
                && (x1 - 0.4).abs() < 1e-4
                && (y1 - 0.6).abs() < 1e-4
        );
    }

    #[test]
    fn rrcurveto_produces_a_curve() {
        // moveto(0,0) then one rrcurveto -> a cubic; flatten yields many points.
        let mut cs = Vec::new();
        cs.extend(num(0));
        cs.extend(num(0));
        cs.push(21); // rmoveto
        for v in [0, 300, 300, 300, 300, 0] {
            cs.extend(num(v));
        }
        cs.push(8); // rrcurveto
        cs.push(14);
        let cff = build_cff(&[vec![14], cs]);
        let prog = CffProgram::parse(&cff).unwrap();
        let path = prog.outline(1).unwrap();
        // Flatten at a glyph-rendering scale so the cubic actually subdivides
        // (em-space coords are sub-pixel and would flatten to a single segment).
        let polys = path.flatten(Matrix::scale(1000.0, 1000.0));
        assert!(
            polys[0].len() > 8,
            "curve flattened to {} pts",
            polys[0].len()
        );
    }
}
