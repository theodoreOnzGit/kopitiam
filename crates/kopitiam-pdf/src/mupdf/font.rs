//! Ported from MuPDF `source/pdf/pdf-font.c` (simple + CID font loading),
//! `source/pdf/pdf-metrics.c` (scalar advance-width cache), the
//! `source/pdf/pdf-unicode.c` `pdf_load_to_unicode` path, and the glyph-code ->
//! Unicode fallback chain of `source/pdf/pdf-op-run.c` (`pdf_show_char`,
//! lines 1414-1435) (+ `include/mupdf/pdf/font.h`) (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # The glyph-code -> (Unicode, advance) layer
//!
//! A [`Font`] loads from a PDF font dictionary and answers the two questions text
//! extraction needs for each character code: **what Unicode does it stand for**,
//! and **how wide is it** (so the layout engine can tell a real space from mere
//! glyph advance -- the source of the spurious inter-glyph spaces). It is the
//! extraction-relevant slice of MuPDF's `pdf_font_desc`.
//!
//! ## No FreeType
//!
//! MuPDF reads embedded font programs through FreeType to build the code -> glyph
//! map and, as a last resort, glyph widths and the builtin encoding. Per the port
//! manifest this port **avoids FreeType entirely**: Unicode comes from
//! `/ToUnicode` and the `/Encoding` glyph names (via the AGL, [`super::agl`]), and
//! advances come from the PDF's `/Widths` (simple fonts) or `/W` + `/DW` (CID
//! fonts). Where MuPDF would consult FreeType -- reading an embedded font's own
//! cmap, reversing builtin glyph names, or falling back to glyph metrics when
//! `/Widths` is absent -- this port falls back gracefully to the PDF-declared
//! default (`/MissingWidth` or `/DW`, and [`super::agl::REPLACEMENT_CHARACTER`]
//! for Unicode), never panics, and never rasterises a glyph.
//!
//! ## The fallback chain (the "missing unicode map" bug fix)
//!
//! [`Font::decode`] reproduces `pdf_show_char`'s exact chain for a code's CID:
//!
//! 1. `/ToUnicode` lookup (the remapped CID -> Unicode CMap).
//! 2. Control characters 8..=13 become a space; values `< 32` or `127..160` are
//!    rejected (treated as "no mapping").
//! 3. The `/Encoding`-derived `cid_to_ucs` table (base encoding + `/Differences`
//!    glyph names resolved through the AGL) -- simple fonts only.
//! 4. Otherwise **U+FFFD**. This is the graceful fallback that replaces the old
//!    "missing unicode map" panic.
//!
//! ## Deferred
//!
//! Type3 `/FontMatrix` width scaling (Type3 fonts are loaded through the simple
//! path, `/Widths` taken verbatim), vertical writing-mode metrics (`/W2` / `/DW2`
//! -> `vmtx`; horizontal `hmtx` is always built), the named Adobe CJK CMap
//! resource set (see `cmap.rs`), and embedded-font glyph reading (FreeType).

use super::agl::{unicode_from_glyph_name, REPLACEMENT_CHARACTER};
use super::cmap::{self, CMap};
use super::encodings::BaseEncoding;
use super::error::Result;
use super::object::Object;
use super::xref::PdfDocument;

/// One horizontal-metrics entry: codes/CIDs in `lo..=hi` advance by `w` (in
/// 1/1000 em). Mirrors `pdf_hmtx`.
// MuPDF: pdf_hmtx (font.h:45)
#[derive(Clone, Copy, Debug)]
struct Hmtx {
    lo: u32,
    hi: u32,
    w: i32,
}

/// The result of decoding one character code: its Unicode value, advance width
/// (in 1/1000 em, i.e. glyph-space units matching the PDF `/Widths` / `/W`), and
/// CID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoded {
    /// The Unicode code point this code maps to (U+FFFD on a total miss).
    pub unicode: char,
    /// The advance width in 1/1000 em (multiply by fontsize/1000 for text space).
    pub advance: f32,
    /// The CID the code maps to through the encoding CMap.
    pub cid: u32,
}

/// A loaded PDF font: the extraction-relevant slice of `pdf_font_desc`.
///
/// See the module docs for the code -> Unicode fallback chain ([`Font::decode`])
/// and the no-FreeType policy.
#[derive(Clone, Debug)]
pub struct Font {
    /// `code -> CID` (Identity 1-byte for simple fonts; the `/Encoding` CMap for
    /// CID fonts). Also carries the codespace used to split a string into codes.
    encoding: CMap,
    /// `CID -> Unicode`, built by remapping the `/ToUnicode` CMap through the
    /// encoding (`pdf_remap_cmap`). `None` when the font has no usable ToUnicode.
    to_unicode: Option<CMap>,
    /// `code -> Unicode` from the `/Encoding` glyph names via the AGL (simple
    /// fonts only; length 256). `None` for CID fonts.
    cid_to_ucs: Option<Vec<u16>>,
    /// Writing mode: 0 = horizontal, 1 = vertical.
    wmode: i32,
    /// Horizontal advance ranges, sorted by `lo` (binary-searched).
    hmtx: Vec<Hmtx>,
    /// Default advance for CIDs outside every `hmtx` range (`/MissingWidth` for
    /// simple fonts, `/DW` -- default 1000 -- for CID fonts).
    dhmtx_w: i32,
}

impl Font {
    // -----------------------------------------------------------------------
    // Loading
    // -----------------------------------------------------------------------

    // MuPDF: pdf_load_font (pdf-font.c:1583) -- dispatch on /Subtype.
    /// Load a font from its PDF font dictionary `dict` (already resolved),
    /// resolving `/ToUnicode` / CMap / `CIDFont` sub-objects through `doc`.
    pub fn load(doc: &PdfDocument, dict: &Object) -> Result<Font> {
        let subtype = doc.resolve_get(dict, "Subtype")?;
        match subtype.to_name() {
            b"Type0" => Font::load_cid(doc, dict),
            b"Type1" | b"MMType1" | b"TrueType" | b"Type3" => Font::load_simple(doc, dict),
            _ => {
                // Unknown subtype: a font with descendants is a Type0 (MuPDF's
                // fallback), otherwise treat it as a simple font.
                if !doc.resolve_get(dict, "DescendantFonts")?.is_null() {
                    Font::load_cid(doc, dict)
                } else {
                    Font::load_simple(doc, dict)
                }
            }
        }
    }

    // MuPDF: pdf_load_simple_font (pdf-font.c:781) -- the no-FreeType slice.
    fn load_simple(doc: &PdfDocument, dict: &Object) -> Result<Font> {
        let descriptor = doc.resolve_get(dict, "FontDescriptor")?;
        let flags = doc.resolve_get(&descriptor, "Flags")?.to_int() as i32;
        let missing_width = doc.resolve_get(&descriptor, "MissingWidth")?.to_int() as i32;
        let is_embedded = has_font_file(&descriptor);

        // symbolic = flags & FIXED... actually FD_SYMBOLIC (bit 2); bit 5
        // (nonsymbolic) forces non-symbolic (bug 703273).
        let mut symbolic = flags & 4 != 0;
        if flags & 32 != 0 {
            symbolic = false;
        }

        // Build the per-code glyph-name table (`estrings`).
        let mut estrings: Vec<Option<String>> = vec![None; 256];
        let encoding = doc.resolve_get(dict, "Encoding")?;
        if encoding.is_name() {
            load_encoding(&mut estrings, encoding.to_name());
        } else if encoding.is_dict() {
            let base = doc.resolve_get(&encoding, "BaseEncoding")?;
            if base.is_name() {
                load_encoding(&mut estrings, base.to_name());
            } else if !is_embedded && !symbolic {
                load_encoding(&mut estrings, b"StandardEncoding");
            }
            // /Differences overrides individual codes.
            let diff = doc.resolve_get(&encoding, "Differences")?;
            if diff.is_array() {
                let mut k: usize = 0;
                for i in 0..diff.array_len() {
                    match diff.array_get(i) {
                        Some(Object::Int(n)) => {
                            if *n >= 0 && *n < 256 {
                                k = *n as usize;
                            }
                        }
                        Some(Object::Name(nm)) if k < 256 => {
                            estrings[k] = Some(String::from_utf8_lossy(nm).into_owned());
                            k += 1;
                        }
                        _ => {}
                    }
                }
            }
        } else if !is_embedded && !symbolic {
            load_encoding(&mut estrings, b"StandardEncoding");
        }

        let mut font = Font {
            // Simple fonts use an Identity 1-byte encoding: code == CID.
            encoding: CMap::new_identity(0, 1),
            to_unicode: None,
            cid_to_ucs: None,
            wmode: 0,
            hmtx: Vec::new(),
            dhmtx_w: missing_width,
        };

        // cid_to_ucs from the glyph names (pdf_load_to_unicode's `strings` path).
        let mut cid_to_ucs = vec![REPLACEMENT_CHARACTER as u16; 256];
        for (c, slot) in cid_to_ucs.iter_mut().enumerate() {
            *slot = match &estrings[c] {
                Some(name) => unicode_from_glyph_name(name) as u16,
                None => REPLACEMENT_CHARACTER as u16,
            };
        }
        font.cid_to_ucs = Some(cid_to_ucs);

        // /ToUnicode stream (remapped through the identity encoding).
        font.to_unicode = load_to_unicode(doc, dict.dict_gets("ToUnicode"), &font.encoding);

        // Widths -> hmtx.
        let widths = doc.resolve_get(dict, "Widths")?;
        if widths.is_array() {
            let mut first = doc.resolve_get(dict, "FirstChar")?.to_int();
            let mut last = doc.resolve_get(dict, "LastChar")?.to_int();
            if first < 0 || last > 255 || first > last {
                first = 0;
                last = 0;
            }
            for i in 0..(last - first + 1) {
                let wid = widths
                    .array_get(i as usize)
                    .map(|o| resolve(doc, o).to_int())
                    .unwrap_or(0) as i32;
                font.add_hmtx((i + first) as u32, (i + first) as u32, wid);
            }
        }
        // No /Widths: MuPDF falls back to FreeType glyph widths; we avoid
        // FreeType, so codes fall through to the default (missing_width) advance.
        font.end_hmtx();

        Ok(font)
    }

    // MuPDF: pdf_load_type0_font + load_cid_font (pdf-font.c:1425, 1171).
    fn load_cid(doc: &PdfDocument, dict: &Object) -> Result<Font> {
        let dfonts = doc.resolve_get(dict, "DescendantFonts")?;
        let dfont = match dfonts.array_get(0) {
            Some(d) => resolve(doc, d),
            None => Object::Null,
        };

        // Encoding: a predefined name, or an embedded CMap stream.
        let enc_raw = dict.dict_gets("Encoding").cloned().unwrap_or(Object::Null);
        let encoding = match &enc_raw {
            Object::Name(n) => {
                let name = String::from_utf8_lossy(n);
                // Only Identity-* is implemented; the named Adobe CJK resource
                // set is deferred, so fall back to Identity-H (2-byte) gracefully.
                CMap::load_predefined(&name).unwrap_or_else(|| CMap::new_identity(0, 2))
            }
            Object::Ref { .. } => doc
                .open_stream(&enc_raw)
                .ok()
                .and_then(|b| cmap::load(&b).ok())
                .unwrap_or_else(|| CMap::new_identity(0, 2)),
            // MuPDF throws "font missing encoding"; a Type0 font without one is
            // unusable, but we degrade to Identity rather than fail the page.
            _ => CMap::new_identity(0, 2),
        };

        let wmode = encoding.wmode();

        let mut font = Font {
            encoding,
            to_unicode: None,
            cid_to_ucs: None,
            wmode,
            hmtx: Vec::new(),
            dhmtx_w: 1000,
        };

        // /ToUnicode, remapped from code->Unicode to CID->Unicode.
        font.to_unicode = load_to_unicode(doc, dict.dict_gets("ToUnicode"), &font.encoding);

        // Horizontal metrics: /DW (default 1000) + /W.
        let dw = doc.resolve_get(&dfont, "DW")?;
        font.dhmtx_w = if dw.is_null() { 1000 } else { dw.to_int() as i32 };

        let w = doc.resolve_get(&dfont, "W")?;
        if w.is_array() {
            font.parse_cid_widths(doc, &w);
        }
        font.end_hmtx();

        // Vertical (/W2, /DW2) writing-mode metrics are deferred; horizontal
        // advances are always available for the common wmode == 0 case.

        Ok(font)
    }

    // MuPDF: the /W parser inside load_cid_font (pdf-font.c:1323).
    fn parse_cid_widths(&mut self, doc: &PdfDocument, w: &Object) {
        let n = w.array_len();
        let mut i = 0;
        while i < n {
            let c0 = match w.array_get(i) {
                Some(o) => resolve(doc, o).to_int(),
                None => break,
            };
            let obj = match w.array_get(i + 1) {
                Some(o) => resolve(doc, o),
                None => break,
            };
            if obj.is_array() {
                // [ c [w1 w2 ...] ]: consecutive CIDs c, c+1, ...
                for k in 0..obj.array_len() {
                    let wid = obj
                        .array_get(k)
                        .map(|o| resolve(doc, o).to_int())
                        .unwrap_or(0) as i32;
                    self.add_hmtx((c0 + k as i64) as u32, (c0 + k as i64) as u32, wid);
                }
                i += 2;
            } else {
                // [ c0 c1 w ]: the range c0..=c1 all advance by w.
                let c1 = obj.to_int();
                let wid = match w.array_get(i + 2) {
                    Some(o) => resolve(doc, o).to_int() as i32,
                    None => break,
                };
                self.add_hmtx(c0 as u32, c1 as u32, wid);
                i += 3;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Metrics (pdf-metrics.c)
    // -----------------------------------------------------------------------

    // MuPDF: pdf_add_hmtx (pdf-metrics.c:47)
    fn add_hmtx(&mut self, lo: u32, hi: u32, w: i32) {
        self.hmtx.push(Hmtx { lo, hi, w });
    }

    // MuPDF: pdf_end_hmtx (pdf-metrics.c:95) -- sort for binary search.
    fn end_hmtx(&mut self) {
        self.hmtx.sort_by_key(|h| h.lo);
    }

    // MuPDF: pdf_lookup_hmtx (pdf-metrics.c:113)
    /// The advance width (1/1000 em) for `cid`, or the default width if no
    /// `hmtx` range contains it.
    fn lookup_hmtx(&self, cid: u32) -> i32 {
        let mut l = 0isize;
        let mut r = self.hmtx.len() as isize - 1;
        while l <= r {
            let m = ((l + r) >> 1) as usize;
            let h = &self.hmtx[m];
            if cid < h.lo {
                r = m as isize - 1;
            } else if cid > h.hi {
                l = m as isize + 1;
            } else {
                return h.w;
            }
        }
        self.dhmtx_w
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// The writing mode (0 = horizontal, 1 = vertical).
    pub fn wmode(&self) -> i32 {
        self.wmode
    }

    // MuPDF: fz_font_ascender (font.c:281) -- FZ_ASCDESC_DEFAULT path.
    /// The font's ascender in em units (glyph box top). Because this port
    /// avoids FreeType, per-font face metrics are unavailable, so this returns
    /// MuPDF's default ascender `0.8` -- exactly the value MuPDF itself falls
    /// back to (`FZ_ASCDESC_DEFAULT`) whenever a face reports no trustworthy
    /// ascent (`fz_new_font_from_buffer`, font.c:785). The `stext` device uses
    /// it to build the scalar (non-accurate) glyph quad. Reading `/Ascent` from
    /// the PDF `FontDescriptor` is deliberately *not* done here: MuPDF's
    /// `fz_font_ascender` never consults it either.
    pub fn ascender(&self) -> f32 {
        0.8
    }

    // MuPDF: fz_font_descender (font.c:286) -- FZ_ASCDESC_DEFAULT path.
    /// The font's descender in em units (glyph box bottom, negative). Returns
    /// MuPDF's default `-0.2` for the same no-FreeType reason as
    /// [`Font::ascender`].
    pub fn descender(&self) -> f32 {
        -0.2
    }

    // MuPDF: pdf_decode_cmap over fontdesc->encoding (pdf-op-run.c:1577).
    /// Split the front of `buf` into one character code, returning
    /// `(code, bytes_consumed)`. Use this to iterate a PDF string's codes (1- or
    /// 2-byte, per the encoding CMap's codespace).
    pub fn next_code(&self, buf: &[u8]) -> (u32, usize) {
        self.encoding.decode(buf)
    }

    /// Decode a whole PDF string into a sequence of [`Decoded`] characters,
    /// splitting it into codes and mapping each (`show_string`'s loop).
    pub fn decode_string(&self, buf: &[u8]) -> Vec<Decoded> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < buf.len() {
            let (code, n) = self.next_code(&buf[i..]);
            out.push(self.decode(code));
            i += n.max(1);
        }
        out
    }

    // MuPDF: pdf_lookup_cmap(encoding) + pdf_lookup_hmtx + the ucs fallback chain
    // (pdf-op-run.c:1580, pdf-interpret.c:2034, pdf-op-run.c:1414-1435).
    /// Decode one character `code` into its Unicode, advance, and CID.
    ///
    /// Reproduces MuPDF's graceful fallback chain exactly (see the module docs):
    /// ToUnicode, then control-char/bad-value filtering, then the encoding-derived
    /// `cid_to_ucs` table, then U+FFFD. **Never panics** -- this is the fix for
    /// the "missing unicode map" panic.
    pub fn decode(&self, code: u32) -> Decoded {
        // code -> CID (identity for simple fonts). A code the CMap cannot map
        // falls back to the code itself rather than being dropped.
        let cid = self.encoding.lookup(code).unwrap_or(code);
        let advance = self.lookup_hmtx(cid) as f32;
        let unicode = self.decode_unicode(cid);
        Decoded {
            unicode,
            advance,
            cid,
        }
    }

    // MuPDF: pdf_show_char's ucs resolution (pdf-op-run.c:1414-1435).
    fn decode_unicode(&self, cid: u32) -> char {
        let mut ucsbuf: Vec<u32> = Vec::new();
        let mut ucslen = 0usize;

        // (1) ToUnicode.
        if let Some(tu) = &self.to_unicode {
            ucslen = tu.lookup_full(cid, &mut ucsbuf);
        }

        // (2) ASCII whitespace control chars 8..=13 become a space.
        if ucslen == 1 && (8..=13).contains(&ucsbuf[0]) {
            ucsbuf[0] = ' ' as u32;
        }
        // Reject obviously bad ToUnicode values, falling back to cid_to_ucs.
        if ucslen == 1 && (ucsbuf[0] < 32 || (127..160).contains(&ucsbuf[0])) {
            ucslen = 0;
        }

        // (3) The Encoding-derived cid_to_ucs table.
        if ucslen == 0
            && let Some(c2u) = &self.cid_to_ucs
            && (cid as usize) < c2u.len()
        {
            ucsbuf.clear();
            ucsbuf.push(c2u[cid as usize] as u32);
            ucslen = 1;
        }

        // (4) Otherwise U+FFFD (the graceful, non-panicking fallback).
        if ucslen == 0 || (ucslen == 1 && ucsbuf[0] == 0) {
            return '\u{FFFD}';
        }

        char::from_u32(ucsbuf[0]).unwrap_or('\u{FFFD}')
    }
}

// MuPDF: pdf_load_encoding (pdf-font.c:42) -- fill estrings from a base encoding.
fn load_encoding(estrings: &mut [Option<String>], name: &[u8]) {
    let enc = match name {
        b"StandardEncoding" => BaseEncoding::Standard,
        b"MacRomanEncoding" => BaseEncoding::MacRoman,
        b"MacExpertEncoding" => BaseEncoding::MacExpert,
        b"WinAnsiEncoding" => BaseEncoding::WinAnsi,
        _ => return,
    };
    for (i, slot) in estrings.iter_mut().enumerate().take(256) {
        *slot = enc.glyph_name(i as u8).map(|s| s.to_string());
    }
}

// MuPDF: pdf_load_to_unicode's stream/name branches (pdf-unicode.c:93).
/// Load a `/ToUnicode` value into a CID -> Unicode CMap by remapping the raw
/// code -> Unicode CMap through `encoding`. Returns `None` (graceful) on any
/// failure -- a missing, malformed, or unsupported ToUnicode is never fatal.
fn load_to_unicode(doc: &PdfDocument, raw: Option<&Object>, encoding: &CMap) -> Option<CMap> {
    let obj = raw?;
    let ucs_from_cpt = match obj {
        Object::Name(n) => {
            let name = String::from_utf8_lossy(n);
            CMap::load_predefined(&name)?
        }
        Object::Ref { .. } => {
            let bytes = doc.open_stream(obj).ok()?;
            cmap::load(&bytes).ok()?
        }
        _ => return None,
    };
    Some(encoding.remap(&ucs_from_cpt))
}

/// True if the font descriptor embeds a font program (`/FontFile*`).
fn has_font_file(descriptor: &Object) -> bool {
    descriptor.dict_gets("FontFile").is_some()
        || descriptor.dict_gets("FontFile2").is_some()
        || descriptor.dict_gets("FontFile3").is_some()
}

/// Resolve `obj` against `doc` if it is an indirect reference, else clone it.
/// A convenience over [`PdfDocument::resolve`] that swallows resolution errors
/// into [`Object::Null`] (accessors then degrade silently, as MuPDF's do).
fn resolve(doc: &PdfDocument, obj: &Object) -> Object {
    doc.resolve(obj).unwrap_or(Object::Null)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hand-built PDF fixtures ------------------------------------------------

    /// Build a minimal, valid one-page PDF whose objects 1..=3 are a
    /// catalog/pages/page, with `extra` object bodies appended as objects 4, 5,
    /// … (each `body` is wrapped in `N 0 obj … endobj`; a body may itself contain
    /// a `stream … endstream`). Uses a classic xref table.
    fn build_pdf(extra: &[&[u8]]) -> Vec<u8> {
        let mut bodies: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec(),
        ];
        for e in extra {
            bodies.push(e.to_vec());
        }

        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref_ofs = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }

    /// A ToUnicode stream object body mapping <41>->'A', <42>..<44>->'1'..'3'.
    fn tounicode_body() -> Vec<u8> {
        let cmap = b"/CIDInit /ProcSet findresource begin\n\
12 dict begin\nbegincmap\n\
1 begincodespacerange\n<00> <ff>\nendcodespacerange\n\
1 beginbfchar\n<41> <0041>\nendbfchar\n\
1 beginbfrange\n<42> <44> <0031>\nendbfrange\n\
endcmap\nend\nend";
        let mut body = Vec::new();
        body.extend_from_slice(format!("<< /Length {} >>\nstream\n", cmap.len()).as_bytes());
        body.extend_from_slice(cmap);
        body.extend_from_slice(b"\nendstream");
        body
    }

    // -- Tests ------------------------------------------------------------------

    #[test]
    fn tounicode_stream_decodes() {
        // Simple font (object 4) with a ToUnicode stream (object 5).
        let font_dict = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 5 0 R >>";
        let pdf = build_pdf(&[font_dict, &tounicode_body()]);
        let doc = PdfDocument::open(pdf).unwrap();
        let dict = doc.resolve(&Object::new_indirect(4, 0)).unwrap();
        let font = Font::load(&doc, &dict).unwrap();

        // bfchar and bfrange both resolve through ToUnicode.
        assert_eq!(font.decode(0x41).unicode, 'A');
        assert_eq!(font.decode(0x42).unicode, '1');
        assert_eq!(font.decode(0x44).unicode, '3');
    }

    #[test]
    fn simple_font_winansi_and_widths() {
        // WinAnsiEncoding + /Widths: code 0x41 ('A') width 700, 0x80 (Euro) 556.
        let font_dict = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
/Encoding /WinAnsiEncoding /FirstChar 65 /LastChar 128 \
/Widths [700 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 556] >>";
        let pdf = build_pdf(&[font_dict]);
        let doc = PdfDocument::open(pdf).unwrap();
        let dict = doc.resolve(&Object::new_indirect(4, 0)).unwrap();
        let font = Font::load(&doc, &dict).unwrap();

        // Unicode via WinAnsi base encoding (no ToUnicode present).
        let a = font.decode(0x41);
        assert_eq!(a.unicode, 'A');
        assert_eq!(a.advance, 700.0);
        // 0x80 is the Euro sign in WinAnsi, width 556 (the last /Widths entry).
        let euro = font.decode(0x80);
        assert_eq!(euro.unicode, '\u{20AC}');
        assert_eq!(euro.advance, 556.0);
    }

    #[test]
    fn simple_font_differences_override() {
        // Base WinAnsi, but /Differences remaps code 0x41 to the "bullet" glyph.
        let font_dict = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
/Encoding << /BaseEncoding /WinAnsiEncoding /Differences [65 /bullet] >> >>";
        let pdf = build_pdf(&[font_dict]);
        let doc = PdfDocument::open(pdf).unwrap();
        let dict = doc.resolve(&Object::new_indirect(4, 0)).unwrap();
        let font = Font::load(&doc, &dict).unwrap();

        // 0x41 now decodes to U+2022 (bullet), not 'A'.
        assert_eq!(font.decode(0x41).unicode, '\u{2022}');
        // 0x42 is untouched -> 'B'.
        assert_eq!(font.decode(0x42).unicode, 'B');
    }

    #[test]
    fn cid_font_identity_h_with_w_and_dw() {
        // Type0 (obj 4) -> DescendantFonts [CIDFont (obj 5)], Identity-H.
        let type0 = b"<< /Type /Font /Subtype /Type0 /BaseFont /Sub /Encoding /Identity-H /DescendantFonts [5 0 R] >>";
        let cidfont = b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Sub \
/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
/DW 1000 /W [ 1 [ 500 600 ] 10 20 800 ] >>";
        let pdf = build_pdf(&[type0, cidfont]);
        let doc = PdfDocument::open(pdf).unwrap();
        let dict = doc.resolve(&Object::new_indirect(4, 0)).unwrap();
        let font = Font::load(&doc, &dict).unwrap();

        // 2-byte code splits correctly: <0001> is one code, CID 1.
        let (code, n) = font.next_code(&[0x00, 0x01]);
        assert_eq!((code, n), (0x0001, 2));

        // /W: CID 1 -> 500, CID 2 -> 600 (array form).
        assert_eq!(font.decode(0x0001).advance, 500.0);
        assert_eq!(font.decode(0x0002).advance, 600.0);
        // /W range form: CIDs 10..=20 -> 800.
        assert_eq!(font.decode(15).advance, 800.0);
        // Outside every range -> /DW default 1000.
        assert_eq!(font.decode(500).advance, 1000.0);
        // CID == code under Identity-H.
        assert_eq!(font.decode(0x0002).cid, 2);
    }

    #[test]
    fn missing_unicode_map_yields_replacement_not_panic() {
        // The bug fix, at the unit level: a font with no ToUnicode and no
        // cid_to_ucs must return U+FFFD for any code, never panic or error.
        let font = Font {
            encoding: CMap::new_identity(0, 1),
            to_unicode: None,
            cid_to_ucs: None,
            wmode: 0,
            hmtx: Vec::new(),
            dhmtx_w: 500,
        };
        let d = font.decode(0x41);
        assert_eq!(d.unicode, '\u{FFFD}');
        assert_eq!(d.advance, 500.0);
        assert_eq!(d.cid, 0x41);
    }

    #[test]
    fn cid_font_without_tounicode_is_graceful() {
        // A Type0/Identity-H font with NO /ToUnicode: decode must not panic and
        // every code maps to U+FFFD (no cid_to_ucs for CID fonts).
        let type0 = b"<< /Type /Font /Subtype /Type0 /BaseFont /Sub /Encoding /Identity-H /DescendantFonts [5 0 R] >>";
        let cidfont = b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Sub \
/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 1000 >>";
        let pdf = build_pdf(&[type0, cidfont]);
        let doc = PdfDocument::open(pdf).unwrap();
        let dict = doc.resolve(&Object::new_indirect(4, 0)).unwrap();
        let font = Font::load(&doc, &dict).unwrap();

        assert_eq!(font.decode(0x1234).unicode, '\u{FFFD}');
        assert_eq!(font.decode(0x1234).advance, 1000.0); // /DW, not a spurious 0
    }

    #[test]
    fn control_char_tounicode_becomes_space() {
        // ToUnicode mapping code 1 -> U+0009 (tab, a control char 8..13) must be
        // normalised to a space by the fallback chain.
        let cmap = b"begincmap\n1 begincodespacerange\n<00> <ff>\nendcodespacerange\n\
1 beginbfchar\n<01> <0009>\nendbfchar\nendcmap";
        let mut body = Vec::new();
        body.extend_from_slice(format!("<< /Length {} >>\nstream\n", cmap.len()).as_bytes());
        body.extend_from_slice(cmap);
        body.extend_from_slice(b"\nendstream");

        let font_dict = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /ToUnicode 5 0 R >>";
        let pdf = build_pdf(&[font_dict, &body]);
        let doc = PdfDocument::open(pdf).unwrap();
        let dict = doc.resolve(&Object::new_indirect(4, 0)).unwrap();
        let font = Font::load(&doc, &dict).unwrap();

        assert_eq!(font.decode(0x01).unicode, ' ');
    }
}
