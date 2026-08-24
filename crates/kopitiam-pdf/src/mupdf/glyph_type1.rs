//! Type 1 (`/FontFile`) glyph-outline decoding: PFA/PFB unwrapping, `eexec` /
//! charstring decryption, and a Type 1 charstring interpreter, implemented from
//! the **Adobe Type 1 Font Format specification** (Adobe Systems Inc., 1990,
//! "the Black Book") -- the document that also underlies FreeType's `psaux`/
//! `type1` driver and `t1lib`, which is what MuPDF loads `/FontFile` through
//! (`source/pdf/pdf-font.c` -> `fz_new_type1_font` -> FreeType, commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.). This is a from-specification
//! implementation, not a translation of MuPDF's or FreeType's C -- the port
//! avoids FreeType entirely (see [`super::font`]) and MuPDF itself never
//! implements a Type 1 interpreter of its own to translate from. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What is decoded
//!
//! * **Container**: raw PFA (`%!`-prefixed cleartext + `eexec`) and PFB
//!   (`0x80`-segmented) font programs. `/Length1`/`/Length2` from the PDF
//!   `/FontFile` stream dict are used as a boundary hint when available
//!   ([`Type1Program::parse`]); otherwise the `eexec` keyword is located by
//!   search.
//! * **Decryption**: the `eexec` cipher (R=55665, discarding the fixed 4
//!   leading garbage bytes) over the Private dict + `CharStrings`/`Subrs`, then
//!   the per-charstring cipher (R=4330, discarding `/lenIV` bytes, default 4)
//!   for each individual charstring/subroutine (Type 1 Font Format spec,
//!   sections 7.3 "eexec Encryption" and 7.2 "Charstring Encryption" -- the
//!   same two Vigenere-style ciphers, different keys and skip counts).
//! * **The Type 1 charstring interpreter**: the move/line/curve operators
//!   (`hsbw`/`sbw` establish the starting point; `rmoveto`/`hmoveto`/`vmoveto`,
//!   `rlineto`/`hlineto`/`vlineto`, `rrcurveto`/`vhcurveto`/`hvcurveto`,
//!   `closepath`), `callsubr`/`return` (no subr bias -- that is a Type 2
//!   refinement), `div`, the **flex** and **hint-replacement** `OtherSubrs`
//!   convention via `callothersubr`/`pop` (spec section 8.3), and `seac`
//!   (accented-character composition against the Adobe StandardEncoding, spec
//!   section 8.7). Hint operators (`hstem`/`vstem`/`vstem3`/`hstem3`/
//!   `dotsection`) are recognised and discarded -- this module builds outlines,
//!   not hints.
//! * **The font's own built-in `/Encoding`** (code -> glyph name), read from
//!   the *cleartext* portion (this array always precedes `eexec` in a Type 1
//!   program) -- used as the fallback selection when the PDF's own `/Encoding`
//!   doesn't name a glyph for a code (see [`super::font::Font::glyph_outline`]).
//!
//! # Selection is by **name**, not GID
//!
//! Unlike TrueType/CFF, a Type 1 font has no numeric glyph index the PDF can
//! address -- `CharStrings` is a `name -> charstring` dictionary. So
//! [`Type1Program`] is *not* wrapped in [`super::glyph::FontProgram`]'s
//! `gid -> outline` shape; [`Font::glyph_outline`](super::font::Font) resolves
//! a glyph **name** first (from the PDF `/Encoding`/`/Differences`, falling
//! back to this font's own built-in encoding) and calls
//! [`Type1Program::outline`] directly.

use super::draw_path::Path;
use super::encodings::BaseEncoding;
use super::glyph::{path_is_empty, Affine};
use std::collections::HashMap;

/// Recursion / call-depth guard for `callsubr` (mirrors [`super::glyph_cff`]'s
/// `MAX_CALL_DEPTH`; Type 1 has no subr bias but the same infinite-recursion
/// risk from a malformed font).
const MAX_CALL_DEPTH: u32 = 30;

/// A parsed Type 1 (`/FontFile`) outline source: decrypted `CharStrings` (by
/// glyph name) + `Subrs`, the font's own built-in `/Encoding`, and its
/// `FontMatrix`.
#[derive(Clone, Debug)]
pub struct Type1Program {
    charstrings: HashMap<String, Vec<u8>>,
    subrs: Vec<Vec<u8>>,
    /// The font's built-in `code -> glyph name` encoding (spec section 7.1);
    /// `None` for a code this font's `/Encoding` array doesn't define.
    encoding: Box<[Option<String>; 256]>,
    matrix: Affine,
}

impl Type1Program {
    /// Parse a `/FontFile` (Type 1) program. `length1`/`length2` are the PDF
    /// stream dict's `/Length1`/`/Length2` (cleartext / encrypted byte counts)
    /// when known -- an exact boundary, preferred over searching for `eexec`.
    /// Returns `None` on anything unparseable (never a panic; the caller falls
    /// back to the advance box).
    pub fn parse(bytes: &[u8], length1: Option<usize>, length2: Option<usize>) -> Option<Type1Program> {
        if bytes.first() == Some(&0x80) {
            return Type1Program::parse_pfb(bytes);
        }
        let (cleartext, enc_region): (&[u8], &[u8]) = match length1.filter(|&l| l <= bytes.len()) {
            Some(l1) => {
                let rest = &bytes[l1..];
                match length2.filter(|&l2| l2 <= rest.len()) {
                    Some(l2) => (&bytes[..l1], &rest[..l2]),
                    None => (&bytes[..l1], rest),
                }
            }
            None => {
                let pos = find(bytes, b"eexec")?;
                (&bytes[..pos], &bytes[pos + 5..])
            }
        };
        Type1Program::from_parts(cleartext, enc_region)
    }

    /// Unwrap a PFB (`0x80`-segmented) program: concatenate the ASCII (type 1)
    /// segments as the cleartext header and the binary (type 2) segments as the
    /// already-unencoded `eexec` ciphertext.
    fn parse_pfb(bytes: &[u8]) -> Option<Type1Program> {
        let mut cleartext = Vec::new();
        let mut cipher = Vec::new();
        let mut pos = 0usize;
        while pos + 6 <= bytes.len() && bytes[pos] == 0x80 {
            let kind = bytes[pos + 1];
            if kind == 3 {
                break; // EOF marker segment
            }
            let len = u32::from_le_bytes([
                bytes[pos + 2],
                bytes[pos + 3],
                bytes[pos + 4],
                bytes[pos + 5],
            ]) as usize;
            let start = pos + 6;
            let end = start.checked_add(len)?.min(bytes.len());
            match kind {
                1 => cleartext.extend_from_slice(&bytes[start..end]),
                2 => cipher.extend_from_slice(&bytes[start..end]),
                _ => {}
            }
            pos = end;
        }
        Type1Program::from_cipher(&cleartext, &cipher)
    }

    /// `enc_region` is the still-encoded payload right after `eexec` (or the
    /// PDF's declared `/Length2` slice): skip leading whitespace, detect
    /// ASCII-hex vs. raw binary, then decrypt.
    fn from_parts(cleartext: &[u8], enc_region: &[u8]) -> Option<Type1Program> {
        let payload = skip_ws(enc_region);
        let cipher = if is_hex_lead(payload) {
            decode_hex(payload)
        } else {
            payload.to_vec()
        };
        Type1Program::from_cipher(cleartext, &cipher)
    }

    fn from_cipher(cleartext: &[u8], cipher: &[u8]) -> Option<Type1Program> {
        // eexec: R=55665, discard the fixed 4 leading (garbage) plaintext bytes.
        let priv_text = decrypt(cipher, 55665, 4);
        let len_iv = find_len_iv(&priv_text).unwrap_or(4);
        let charstrings = parse_charstrings(&priv_text, len_iv)?;
        if charstrings.is_empty() {
            return None;
        }
        let subrs = parse_subrs(&priv_text, len_iv);
        let matrix = find_font_matrix(cleartext).unwrap_or_else(|| Affine::scale(0.001));
        let encoding = find_encoding(cleartext);
        Some(Type1Program {
            charstrings,
            subrs,
            encoding: Box::new(encoding),
            matrix,
        })
    }

    /// The font's own built-in glyph name for `code` (spec section 7.1),
    /// `None` if this font's `/Encoding` leaves that code undefined.
    pub fn encoding_name(&self, code: u8) -> Option<&str> {
        self.encoding[code as usize].as_deref()
    }

    /// Decode the glyph named `name` to an em-space [`Path`] (y-up, 1 em =
    /// 1.0), or `None` for an undefined name / empty glyph.
    pub fn outline(&self, name: &str) -> Option<Path> {
        let cs = self.charstrings.get(name)?;
        let mut ctx = T1Ctx::new(&self.subrs, self.matrix, None);
        ctx.run(cs, 0);
        if let Some((asb, adx, ady, bchar, achar)) = ctx.seac {
            return self.build_seac(asb, adx, ady, bchar, achar);
        }
        if ctx.open {
            ctx.path.close();
        }
        if path_is_empty(&ctx.path) {
            None
        } else {
            Some(ctx.path)
        }
    }

    // MuPDF: no equivalent (FreeType's `t1_decoder_parse_charstrings` seac
    // path); Adobe Type 1 Font Format spec section 8.7 "seac".
    /// Compose an accented character: the base glyph at its own position, plus
    /// the accent glyph translated so its (spec-computed) origin lands at
    /// `(adx - asb + base_sbx, ady)` -- the accent's *own* `hsbw`/`sbw`
    /// sidebearing is discarded per spec and this computed origin used instead.
    fn build_seac(&self, asb: f32, adx: f32, ady: f32, bchar: u8, achar: u8) -> Option<Path> {
        let bname = BaseEncoding::Standard.glyph_name(bchar)?;
        let aname = BaseEncoding::Standard.glyph_name(achar)?;
        let base_cs = self.charstrings.get(bname)?;
        let accent_cs = self.charstrings.get(aname)?;

        let mut base_ctx = T1Ctx::new(&self.subrs, self.matrix, None);
        base_ctx.run(base_cs, 0);
        if base_ctx.open {
            base_ctx.path.close();
        }
        let base_sbx = base_ctx.sbx;

        let origin = (adx - asb + base_sbx, ady);
        let mut accent_ctx = T1Ctx::new(&self.subrs, self.matrix, Some(origin));
        accent_ctx.run(accent_cs, 0);
        if accent_ctx.open {
            accent_ctx.path.close();
        }

        let mut path = base_ctx.path;
        path.append(accent_ctx.path);
        if path_is_empty(&path) {
            None
        } else {
            Some(path)
        }
    }
}

// ---------------------------------------------------------------------------
// The Type 1 charstring interpreter
// ---------------------------------------------------------------------------

/// Execution context for one glyph's Type 1 charstring (+ its subrs).
struct T1Ctx<'a> {
    subrs: &'a [Vec<u8>],
    path: Path,
    x: f32,
    y: f32,
    /// The left sidebearing x set by `hsbw`/`sbw` (spec section 8.7's `sbx`,
    /// needed by [`Type1Program::build_seac`]).
    sbx: f32,
    matrix: Affine,
    open: bool,
    stack: Vec<f32>,
    /// The separate PostScript operand stack `callothersubr`/`pop` move values
    /// through (spec section 8.3) -- distinct from the charstring stack.
    ps_stack: Vec<f32>,
    in_flex: bool,
    /// Absolute (post-`hsbw`-space) points accumulated by the `rmoveto`s inside
    /// a flex sequence (spec section 8.3: 7 points -- a reference point + two
    /// curves' worth of control/end points).
    flex_pts: Vec<(f32, f32)>,
    /// When set (accent glyphs under `seac`), overrides the starting point
    /// `hsbw`/`sbw` would otherwise compute from its own operands.
    override_start: Option<(f32, f32)>,
    /// Set by `seac`; the caller ([`Type1Program::outline`]) composes the
    /// accented character instead of using `path`.
    seac: Option<(f32, f32, f32, u8, u8)>,
    ended: bool,
}

impl<'a> T1Ctx<'a> {
    fn new(subrs: &'a [Vec<u8>], matrix: Affine, override_start: Option<(f32, f32)>) -> T1Ctx<'a> {
        T1Ctx {
            subrs,
            path: Path::new(),
            x: 0.0,
            y: 0.0,
            sbx: 0.0,
            matrix,
            open: false,
            stack: Vec::with_capacity(32),
            ps_stack: Vec::with_capacity(8),
            in_flex: false,
            flex_pts: Vec::with_capacity(8),
            override_start,
            seac: None,
            ended: false,
        }
    }

    fn tx(&self, x: f32, y: f32) -> (f32, f32) {
        self.matrix.apply(x, y)
    }

    fn moveto(&mut self, dx: f32, dy: f32) {
        self.x += dx;
        self.y += dy;
        if self.in_flex {
            self.flex_pts.push((self.x, self.y));
            return;
        }
        if self.open {
            self.path.close();
        }
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

    /// Emit a cubic from three **absolute** (post-`hsbw`-space) points -- the
    /// two flex curves, whose endpoints were accumulated via [`T1Ctx::moveto`]
    /// while `in_flex`.
    fn curveto_abs(&mut self, c1: (f32, f32), c2: (f32, f32), end: (f32, f32)) {
        if !self.open {
            let (px, py) = self.tx(self.x, self.y);
            self.path.move_to(px, py);
            self.open = true;
        }
        let (c1x, c1y) = self.tx(c1.0, c1.1);
        let (c2x, c2y) = self.tx(c2.0, c2.1);
        let (ex, ey) = self.tx(end.0, end.1);
        self.path.curve_to(c1x, c1y, c2x, c2y, ex, ey);
        self.x = end.0;
        self.y = end.1;
    }

    fn set_start(&mut self, sbx: f32, sby: f32) {
        self.sbx = sbx;
        match self.override_start.take() {
            Some((ox, oy)) => {
                self.x = ox;
                self.y = oy;
            }
            None => {
                self.x = sbx;
                self.y = sby;
            }
        }
    }

    fn run(&mut self, cs: &[u8], depth: u32) {
        if depth > MAX_CALL_DEPTH || self.ended {
            return;
        }
        let mut pc = 0usize;
        while pc < cs.len() && !self.ended {
            let b0 = cs[pc];
            pc += 1;
            if b0 >= 32 {
                let (val, np) = parse_num(cs, pc - 1);
                self.stack.push(val);
                pc = np;
                continue;
            }
            match b0 {
                1 | 3 => self.stack.clear(), // hstem / vstem (hints, not drawn)
                4 => {
                    // vmoveto
                    let dy = self.stack.pop().unwrap_or(0.0);
                    self.moveto(0.0, dy);
                    self.stack.clear();
                }
                5 => {
                    // rlineto
                    if self.stack.len() >= 2 {
                        let n = self.stack.len();
                        self.lineto(self.stack[n - 2], self.stack[n - 1]);
                    }
                    self.stack.clear();
                }
                6 => {
                    // hlineto
                    let dx = self.stack.pop().unwrap_or(0.0);
                    self.lineto(dx, 0.0);
                    self.stack.clear();
                }
                7 => {
                    // vlineto
                    let dy = self.stack.pop().unwrap_or(0.0);
                    self.lineto(0.0, dy);
                    self.stack.clear();
                }
                8 => {
                    // rrcurveto
                    if self.stack.len() >= 6 {
                        let n = self.stack.len();
                        let s = &self.stack[n - 6..n];
                        self.curveto(s[0], s[1], s[2], s[3], s[4], s[5]);
                    }
                    self.stack.clear();
                }
                9 => {
                    // closepath
                    if self.open {
                        self.path.close();
                    }
                    self.stack.clear();
                }
                10 => {
                    // callsubr (no bias in Type 1)
                    if let Some(i) = self.stack.pop() {
                        let idx = i as i32;
                        if idx >= 0
                            && let Some(sp) = self.subrs.get(idx as usize).cloned()
                        {
                            self.run(&sp, depth + 1);
                        }
                    }
                }
                11 => return, // return
                13 => {
                    // hsbw: sbx wx
                    if self.stack.len() >= 2 {
                        let n = self.stack.len();
                        self.set_start(self.stack[n - 2], 0.0);
                    }
                    self.stack.clear();
                }
                14 => {
                    // endchar
                    self.ended = true;
                    return;
                }
                21 => {
                    // rmoveto
                    if self.stack.len() >= 2 {
                        let n = self.stack.len();
                        self.moveto(self.stack[n - 2], self.stack[n - 1]);
                    }
                    self.stack.clear();
                }
                22 => {
                    // hmoveto
                    let dx = self.stack.pop().unwrap_or(0.0);
                    self.moveto(dx, 0.0);
                    self.stack.clear();
                }
                30 => {
                    // vhcurveto: dy1 dx2 dy2 dx3
                    if self.stack.len() >= 4 {
                        let n = self.stack.len();
                        let s = &self.stack[n - 4..n];
                        self.curveto(0.0, s[0], s[1], s[2], s[3], 0.0);
                    }
                    self.stack.clear();
                }
                31 => {
                    // hvcurveto: dx1 dx2 dy2 dy3
                    if self.stack.len() >= 4 {
                        let n = self.stack.len();
                        let s = &self.stack[n - 4..n];
                        self.curveto(s[0], 0.0, s[1], s[2], 0.0, s[3]);
                    }
                    self.stack.clear();
                }
                12 => {
                    if pc >= cs.len() {
                        return;
                    }
                    let b1 = cs[pc];
                    pc += 1;
                    self.escape(b1);
                }
                _ => self.stack.clear(),
            }
        }
    }

    /// Two-byte (`12 b`) operators.
    fn escape(&mut self, op: u8) {
        match op {
            0..=2 => self.stack.clear(), // dotsection / vstem3 / hstem3 (hints)
            6 => {
                // seac: asb adx ady bchar achar
                if self.stack.len() >= 5 {
                    let n = self.stack.len();
                    let s = &self.stack[n - 5..n];
                    self.seac = Some((s[0], s[1], s[2], s[3] as u8, s[4] as u8));
                }
                self.stack.clear();
                self.ended = true;
            }
            7 => {
                // sbw: sbx sby wx wy
                if self.stack.len() >= 4 {
                    let n = self.stack.len();
                    self.set_start(self.stack[n - 4], self.stack[n - 3]);
                }
                self.stack.clear();
            }
            12 => {
                // div: a b -> a/b (leaves the result on the stack; not a
                // stack-clearing operator).
                let b = self.stack.pop().unwrap_or(1.0);
                let a = self.stack.pop().unwrap_or(0.0);
                self.stack.push(if b != 0.0 { a / b } else { 0.0 });
            }
            16 => self.callothersubr(),
            17 => {
                // pop: move one value from the PS stack to the charstring stack.
                let v = self.ps_stack.pop().unwrap_or(0.0);
                self.stack.push(v);
            }
            33 => {
                // setcurrentpoint: x y (used after the flex pop pop).
                if self.stack.len() >= 2 {
                    let n = self.stack.len();
                    self.x = self.stack[n - 2];
                    self.y = self.stack[n - 1];
                }
                self.stack.clear();
            }
            _ => self.stack.clear(),
        }
    }

    // Adobe Type 1 Font Format spec, section 8.3 "Flex" and "Hint
    // Replacement": the standard `OtherSubrs` 0-3 convention. Everything else
    // (a font-specific OtherSubr) degrades to echoing its args back onto the PS
    // stack, so a following `pop` sequence gets plausible values rather than
    // nothing.
    fn callothersubr(&mut self) {
        let othersubr = self.stack.pop().unwrap_or(0.0) as i32;
        let n = self.stack.pop().unwrap_or(0.0).max(0.0) as usize;
        let mut args = Vec::with_capacity(n);
        for _ in 0..n {
            args.push(self.stack.pop().unwrap_or(0.0));
        }
        args.reverse(); // args[0] = the first (leftmost) argument.

        match othersubr {
            1 => {
                // Flex start.
                self.in_flex = true;
                self.flex_pts.clear();
            }
            2 => {
                // Flex reference-point marker: no-op (the point itself was
                // already recorded by the rmoveto that preceded this call).
            }
            0 => {
                // Flex end: args = [flex_height, x, y]. Emit the two curves
                // from the 7 accumulated reference points (point 0 is the
                // reference point used only for the height test, ignored for
                // geometry).
                self.in_flex = false;
                if self.flex_pts.len() >= 7 {
                    let p = &self.flex_pts;
                    let (c1, c2, e1) = (p[1], p[2], p[3]);
                    let (c3, c4, e2) = (p[4], p[5], p[6]);
                    self.curveto_abs(c1, c2, e1);
                    self.curveto_abs(c3, c4, e2);
                }
                if args.len() >= 3 {
                    self.ps_stack.push(args[2]); // y (popped second)
                    self.ps_stack.push(args[1]); // x (popped first)
                } else if let Some(&(lx, ly)) = self.flex_pts.last() {
                    self.ps_stack.push(ly);
                    self.ps_stack.push(lx);
                }
                self.flex_pts.clear();
            }
            3 => {
                // Hint replacement: echo the subr# back for the following
                // `pop callsubr`. The replacement subr sets hints only (no
                // path ops in a well-formed font), so simply calling it is
                // harmless for outline purposes.
                self.ps_stack.push(args.first().copied().unwrap_or(0.0));
            }
            _ => {
                for &a in args.iter().rev() {
                    self.ps_stack.push(a);
                }
            }
        }
    }
}

/// Parse one Type 1 numeric operand at `data[pos]` (spec section 6.2 -- no
/// `28` two-byte-int marker and no 16.16-fixed `255`, unlike Type 2).
fn parse_num(data: &[u8], pos: usize) -> (f32, usize) {
    let b0 = data[pos];
    let g = |i: usize| data.get(i).copied().unwrap_or(0) as i32;
    match b0 {
        32..=246 => (b0 as f32 - 139.0, pos + 1),
        247..=250 => (((b0 as i32 - 247) * 256 + g(pos + 1) + 108) as f32, pos + 2),
        251..=254 => ((-(b0 as i32 - 251) * 256 - g(pos + 1) - 108) as f32, pos + 2),
        255 => {
            let v = (g(pos + 1) << 24) | (g(pos + 2) << 16) | (g(pos + 3) << 8) | g(pos + 4);
            (v as f32, pos + 5)
        }
        _ => (0.0, pos + 1),
    }
}

// ---------------------------------------------------------------------------
// Container: PFA/PFB unwrapping, decryption, the Private-dict text scan
// ---------------------------------------------------------------------------

// Adobe Type 1 Font Format spec, section 7.3 "eexec Encryption" / 7.2
// "Charstring Encryption": both use the same Vigenere-style cipher (constants
// C1=52845, C2=22719 either way), keyed by a starting `r` and discarding
// `skip` leading plaintext bytes (4, fixed, for eexec; `/lenIV` -- default 4
// -- per charstring).
fn decrypt(cipher: &[u8], mut r: u16, skip: usize) -> Vec<u8> {
    const C1: u16 = 52845;
    const C2: u16 = 22719;
    let mut out = Vec::with_capacity(cipher.len().saturating_sub(skip));
    for (i, &c) in cipher.iter().enumerate() {
        let p = c ^ (r >> 8) as u8;
        r = (c as u16).wrapping_add(r).wrapping_mul(C1).wrapping_add(C2);
        if i >= skip {
            out.push(p);
        }
    }
    out
}

fn decrypt_charstring(raw: &[u8], len_iv: usize) -> Vec<u8> {
    decrypt(raw, 4330, len_iv)
}

/// A small cursor over the decrypted Private-dict text (itself a mix of
/// PostScript tokens and raw embedded binary charstring runs).
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Cursor<'a> {
        Cursor { data, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    /// Move past the next occurrence of `needle`, or to EOF if absent.
    fn skip_to(&mut self, needle: &[u8]) -> bool {
        match find(&self.data[self.pos..], needle) {
            Some(off) => {
                self.pos += off + needle.len();
                true
            }
            None => {
                self.pos = self.data.len();
                false
            }
        }
    }

    fn read_word(&mut self) -> &'a [u8] {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if !c.is_ascii_whitespace()) {
            self.pos += 1;
        }
        &self.data[start..self.pos]
    }

    fn read_int(&mut self) -> Option<i64> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return None;
        }
        std::str::from_utf8(&self.data[start..self.pos]).ok()?.parse().ok()
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return None;
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }
}

/// Parse the `N array ... dup <idx> <len> RD <bytes> NP ...` `/Subrs` block:
/// `N` (read right after `/Subrs`) bounds the loop, so a malformed or
/// truncated block degrades to a partial (or empty) table rather than hanging.
fn parse_subrs(data: &[u8], len_iv: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let Some(start) = find(data, b"/Subrs") else { return out };
    let mut cur = Cursor::new(data);
    cur.pos = start + "/Subrs".len();
    let Some(n) = cur.read_int() else { return out };
    if !(0..=65536).contains(&n) {
        return out;
    }
    out.resize(n as usize, Vec::new());
    for _ in 0..n {
        if !cur.skip_to(b"dup") {
            break;
        }
        let Some(idx) = cur.read_int() else { break };
        let Some(len) = cur.read_int() else { break };
        cur.skip_ws();
        let _rd = cur.read_word();
        if cur.peek() == Some(b' ') {
            cur.pos += 1;
        }
        let Some(len) = usize::try_from(len).ok() else { break };
        let Some(raw) = cur.take(len) else { break };
        if idx >= 0 && (idx as usize) < out.len() {
            out[idx as usize] = decrypt_charstring(raw, len_iv);
        }
    }
    out
}

/// Parse the `/CharStrings N dict dup begin /name len RD <bytes> ND ... end`
/// block into `name -> decrypted charstring`. Each iteration finds whichever
/// comes first, the next `/` (a glyph-name entry) or the block-closing `end`
/// -- so it terminates even with unconventional `RD`/`ND` token spellings, and
/// never scans into already-consumed binary charstring bytes.
fn parse_charstrings(data: &[u8], len_iv: usize) -> Option<HashMap<String, Vec<u8>>> {
    let start = find(data, b"/CharStrings")?;
    let mut cur = Cursor::new(data);
    cur.pos = start + "/CharStrings".len();
    if !cur.skip_to(b"begin") {
        return None;
    }
    let mut map = HashMap::new();
    loop {
        let rest = &data[cur.pos..];
        let next_slash = find(rest, b"/");
        let next_end = find(rest, b"end");
        let slash_off = match (next_slash, next_end) {
            (Some(s), Some(e)) if e < s => break,
            (Some(s), _) => s,
            (None, _) => break,
        };
        cur.pos += slash_off + 1;
        let name = String::from_utf8_lossy(cur.read_word()).into_owned();
        let Some(len) = cur.read_int() else { break };
        cur.skip_ws();
        let _rd = cur.read_word();
        if cur.peek() == Some(b' ') {
            cur.pos += 1;
        }
        let Some(len) = usize::try_from(len).ok() else { break };
        let Some(raw) = cur.take(len) else { break };
        map.insert(name, decrypt_charstring(raw, len_iv));
    }
    Some(map)
}

fn find_len_iv(data: &[u8]) -> Option<usize> {
    let pos = find(data, b"/lenIV")?;
    let mut cur = Cursor::new(data);
    cur.pos = pos + "/lenIV".len();
    cur.read_int().and_then(|v| usize::try_from(v).ok())
}

/// Read `/FontMatrix [ a b c d e f ]` from the cleartext header, if present.
fn find_font_matrix(cleartext: &[u8]) -> Option<Affine> {
    let pos = find(cleartext, b"/FontMatrix")?;
    let bracket = find(&cleartext[pos..], b"[")? + pos + 1;
    let close = find(&cleartext[bracket..], b"]")? + bracket;
    let text = std::str::from_utf8(&cleartext[bracket..close]).ok()?;
    let nums: Vec<f32> = text
        .split_ascii_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .collect();
    if nums.len() != 6 {
        return None;
    }
    Some(Affine {
        a: nums[0],
        b: nums[1],
        c: nums[2],
        d: nums[3],
        e: nums[4],
        f: nums[5],
    })
}

/// Read the font's built-in `/Encoding` from the cleartext header: either the
/// name `StandardEncoding` (the common case -- copy [`super::encodings::STANDARD`]),
/// or a custom `256 array ... dup <code> /<name> put ...` block.
fn find_encoding(cleartext: &[u8]) -> [Option<String>; 256] {
    const NONE: Option<String> = None;
    let mut table = [NONE; 256];
    let Some(pos) = find(cleartext, b"/Encoding") else { return table };
    let mut cur = Cursor::new(cleartext);
    cur.pos = pos + "/Encoding".len();
    cur.skip_ws();
    let save = cur.pos;
    let word = cur.read_word();
    if word == b"StandardEncoding" {
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = BaseEncoding::Standard.glyph_name(i as u8).map(str::to_owned);
        }
        return table;
    }
    cur.pos = save;
    // Custom array: `dup <code> /<name> put` entries. Bounded by a generous
    // iteration cap (never more than 256 codes exist) so a font whose
    // `/Encoding` block never closes cleanly still terminates.
    for _ in 0..2000 {
        if !cur.skip_to(b"dup") {
            break;
        }
        let Some(code) = cur.read_int() else { break };
        cur.skip_ws();
        if cur.peek() != Some(b'/') {
            continue;
        }
        cur.pos += 1;
        let name = String::from_utf8_lossy(cur.read_word()).into_owned();
        if (0..256).contains(&code) {
            table[code as usize] = Some(name);
        }
    }
    table
}

fn skip_ws(data: &[u8]) -> &[u8] {
    let mut i = 0;
    while matches!(data.get(i), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        i += 1;
    }
    &data[i..]
}

/// True if `data`'s first few bytes look like ASCII-hex (the common PFA
/// encoding of the `eexec` payload) rather than raw binary ciphertext.
fn is_hex_lead(data: &[u8]) -> bool {
    let probe = &data[..data.len().min(4)];
    !probe.is_empty() && probe.iter().all(|b| b.is_ascii_hexdigit())
}

/// Decode ASCII-hex (whitespace-tolerant) up to the first non-hex,
/// non-whitespace byte (the `cleartomark` trailer after the 512-zero padding
/// line, or simply EOF).
fn decode_hex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let mut hi: Option<u8> = None;
    for &b in data {
        let nibble = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            _ => break,
        };
        match hi.take() {
            Some(h) => out.push((h << 4) | nibble),
            None => hi = Some(nibble),
        }
    }
    out
}

/// A plain (non-regex) substring search -- `data` is typically a few KB of
/// PostScript header text, so this is cheap and keeps the module dependency-free.
fn find(data: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > data.len() {
        return None;
    }
    data.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Test-only fixture builders (also used by draw_device's end-to-end tests)
// ---------------------------------------------------------------------------

/// eexec-encrypt (or charstring-encrypt) `plain`, matching the decoder's own
/// cipher, for building synthetic fixtures.
#[cfg(test)]
fn encrypt(plain: &[u8], mut r: u16, pad: &[u8]) -> Vec<u8> {
    const C1: u16 = 52845;
    const C2: u16 = 22719;
    let mut out = Vec::with_capacity(pad.len() + plain.len());
    for &p in pad.iter().chain(plain.iter()) {
        let c = p ^ (r >> 8) as u8;
        out.push(c);
        r = (c as u16).wrapping_add(r).wrapping_mul(C1).wrapping_add(C2);
    }
    out
}

/// A PFA Type1 font whose glyph `A` (code `0x41`, resolved through the font's
/// **built-in** `StandardEncoding` -- the PDF font dict need not specify one)
/// is a ring: an outer 100..900 square plus an oppositely-wound inner
/// 300..700 square, so a correct nonzero fill leaves the centre a hole (the
/// same discriminator [`super::glyph_truetype::ring_font`] uses for
/// TrueType). Used by the end-to-end "letterform has interior white" test.
#[cfg(test)]
pub(crate) fn ring_type1_pfa() -> Vec<u8> {
    fn n(v: i32) -> Vec<u8> {
        let mut out = vec![255u8];
        out.extend_from_slice(&v.to_be_bytes());
        out
    }
    let mut cs = Vec::new();
    // hsbw(0, 1000): start point (0,0), width 1000.
    cs.extend(n(0));
    cs.extend(n(1000));
    cs.push(13);
    // Outer square 100..900, clockwise (in y-up).
    cs.extend(n(100));
    cs.extend(n(100));
    cs.push(21); // rmoveto -> (100,100)
    cs.extend(n(800));
    cs.extend(n(0));
    cs.push(5); // rlineto -> (900,100)
    cs.extend(n(0));
    cs.extend(n(800));
    cs.push(5); // rlineto -> (900,900)
    cs.extend(n(-800));
    cs.extend(n(0));
    cs.push(5); // rlineto -> (100,900)
    cs.push(9); // closepath
    // Inner square 300..700, counter-clockwise -> a hole under nonzero winding.
    cs.extend(n(200));
    cs.extend(n(-600));
    cs.push(21); // rmoveto -> (300,300)
    cs.extend(n(0));
    cs.extend(n(400));
    cs.push(5); // rlineto -> (300,700)
    cs.extend(n(400));
    cs.extend(n(0));
    cs.push(5); // rlineto -> (700,700)
    cs.extend(n(0));
    cs.extend(n(-400));
    cs.push(5); // rlineto -> (700,300)
    cs.push(9); // closepath
    cs.push(14); // endchar

    let cs_enc = encrypt(&cs, 4330, &[0, 0, 0, 0]);
    let mut priv_plain = Vec::new();
    priv_plain.extend_from_slice(b"dup /Private 9 dict dup begin\n/lenIV 4 def\n");
    priv_plain.extend_from_slice(b"/CharStrings 1 dict dup begin\n");
    priv_plain.extend_from_slice(format!("/A {} RD ", cs_enc.len()).as_bytes());
    priv_plain.extend_from_slice(&cs_enc);
    priv_plain.extend_from_slice(b" ND\nend\nend\n");
    let priv_enc = encrypt(&priv_plain, 55665, &[0, 0, 0, 0]);

    let mut pfa = Vec::new();
    pfa.extend_from_slice(b"%!PS-AdobeFont-1.0\n/FontMatrix [0.001 0 0 0.001 0 0] readonly def\n");
    pfa.extend_from_slice(b"/Encoding StandardEncoding def\n");
    pfa.extend_from_slice(b"currentfile eexec\n");
    for byte in &priv_enc {
        pfa.extend_from_slice(format!("{byte:02x}").as_bytes());
    }
    pfa.extend_from_slice(b"\n");
    pfa
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::Matrix;

    fn charstring_num(v: i32) -> Vec<u8> {
        // The 255-prefixed 4-byte-int form covers the whole i32 range and
        // keeps the fixture builder simple.
        let mut out = vec![255u8];
        out.extend_from_slice(&v.to_be_bytes());
        out
    }

    /// Build a minimal PFA Type 1 program (cleartext header + eexec-encrypted
    /// Private dict) with one glyph, `name`, whose decrypted charstring is
    /// `charstring` (charstring-encrypted with `lenIV`=4 here).
    fn build_type1(name: &str, charstring: &[u8]) -> Vec<u8> {
        let cs_enc = encrypt(charstring, 4330, &[0, 0, 0, 0]);
        let mut priv_plain = Vec::new();
        priv_plain.extend_from_slice(b"dup /Private 9 dict dup begin\n/lenIV 4 def\n");
        priv_plain.extend_from_slice(b"/CharStrings 1 dict dup begin\n");
        priv_plain.extend_from_slice(format!("/{name} {} RD ", cs_enc.len()).as_bytes());
        priv_plain.extend_from_slice(&cs_enc);
        priv_plain.extend_from_slice(b" ND\nend\nend\n");

        let priv_enc = encrypt(&priv_plain, 55665, &[0, 0, 0, 0]);

        let mut pfa = Vec::new();
        pfa.extend_from_slice(b"%!PS-AdobeFont-1.0\n/FontMatrix [0.001 0 0 0.001 0 0] readonly def\n");
        pfa.extend_from_slice(b"/Encoding StandardEncoding def\n");
        pfa.extend_from_slice(b"currentfile eexec\n");
        for byte in &priv_enc {
            pfa.extend_from_slice(format!("{byte:02x}").as_bytes());
        }
        pfa.extend_from_slice(b"\n");
        pfa
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

    /// hsbw(100,0) rmoveto(0,0) rlineto(300,0) rlineto(-150,600) closepath endchar.
    fn triangle_charstring() -> Vec<u8> {
        let mut cs = Vec::new();
        cs.extend(charstring_num(100));
        cs.extend(charstring_num(500));
        cs.push(13); // hsbw
        cs.extend(charstring_num(0));
        cs.extend(charstring_num(0));
        cs.push(21); // rmoveto (no-op move to sbx,sby)
        cs.extend(charstring_num(300));
        cs.extend(charstring_num(0));
        cs.push(5); // rlineto -> (400,0)
        cs.extend(charstring_num(-150));
        cs.extend(charstring_num(600));
        cs.push(5); // rlineto -> (250,600)
        cs.push(9); // closepath
        cs.push(14); // endchar
        cs
    }

    #[test]
    fn pfa_triangle_decodes_to_expected_bbox() {
        let pfa = build_type1("A", &triangle_charstring());
        let prog = Type1Program::parse(&pfa, None, None).expect("parse type1");
        let path = prog.outline("A").expect("triangle outline");
        // hsbw sets the start point to (100,0); FontMatrix 0.001: font
        // (100..400, 0..600) -> em (0.1..0.4, 0..0.6).
        let (x0, y0, x1, y1) = bbox(&path);
        assert!((x0 - 0.1).abs() < 1e-4, "x0 {x0}");
        assert!(y0.abs() < 1e-4, "y0 {y0}");
        assert!((x1 - 0.4).abs() < 1e-4, "x1 {x1}");
        assert!((y1 - 0.6).abs() < 1e-4, "y1 {y1}");
    }

    #[test]
    fn builtin_standard_encoding_resolves_names() {
        let pfa = build_type1("A", &triangle_charstring());
        let prog = Type1Program::parse(&pfa, None, None).unwrap();
        // code 0x41 ('A' in StandardEncoding) -> glyph name "A".
        assert_eq!(prog.encoding_name(0x41), Some("A"));
        assert_eq!(prog.encoding_name(0x00), None); // undefined slot
    }

    #[test]
    fn missing_glyph_name_is_none_no_panic() {
        let pfa = build_type1("A", &triangle_charstring());
        let prog = Type1Program::parse(&pfa, None, None).unwrap();
        assert!(prog.outline("nonexistent").is_none());
    }

    #[test]
    fn length1_length2_hint_matches_eexec_search() {
        let pfa = build_type1("A", &triangle_charstring());
        let eexec_kw = find(&pfa, b"eexec").unwrap();
        let length1 = eexec_kw + 5; // right after the "eexec" keyword
        let length2 = pfa.len() - length1;
        let prog = Type1Program::parse(&pfa, Some(length1), Some(length2)).unwrap();
        let path = prog.outline("A").unwrap();
        assert!(!path_is_empty(&path));
    }

    #[test]
    fn hex_lead_detection() {
        assert!(is_hex_lead(b"4b6a"));
        assert!(!is_hex_lead(&[0xffu8, 0x02, 0x9a, 0x11]));
    }

    #[test]
    fn div_computes_fraction() {
        // A charstring using `div` to build a coordinate: 300 2 div -> 150.
        let mut cs = Vec::new();
        cs.extend(charstring_num(100));
        cs.extend(charstring_num(500));
        cs.push(13); // hsbw
        cs.extend(charstring_num(0));
        cs.extend(charstring_num(0));
        cs.push(21); // rmoveto -> (100,0)
        cs.extend(charstring_num(300));
        cs.extend(charstring_num(2));
        cs.push(12);
        cs.push(12); // div -> 150
        cs.extend(charstring_num(0));
        cs.push(5); // rlineto(150,0) -> (250,0)
        cs.push(14); // endchar
        let pfa = build_type1("B", &cs);
        let prog = Type1Program::parse(&pfa, None, None).unwrap();
        let path = prog.outline("B").unwrap();
        let (x0, _, x1, _) = bbox(&path);
        assert!((x0 - 0.1).abs() < 1e-4);
        assert!((x1 - 0.25).abs() < 1e-4, "x1 {x1}");
    }
}
