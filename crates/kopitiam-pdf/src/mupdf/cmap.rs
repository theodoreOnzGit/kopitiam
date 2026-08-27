//! Ported from MuPDF `source/pdf/pdf-cmap.c`, `source/pdf/pdf-cmap-parse.c`,
//! and the ToUnicode-relevant parts of `source/pdf/pdf-cmap-load.c` +
//! `source/pdf/pdf-unicode.c` (+ `include/mupdf/pdf/cmap.h`) (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithms and numeric behaviour
//! follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # The CMap: code -> CID / code -> Unicode
//!
//! A [`CMap`] is MuPDF's `pdf_cmap`: a set of *codespace ranges* (used to split a
//! PDF string into 1- or 2-byte codes) plus the mappings themselves -- one-to-one
//! contiguous *ranges* (`begincidrange` / `beginbfrange`) and one-to-many
//! *mranges* (`beginbfchar` / `beginbfrange` with a multi-character
//! destination). `usecmap` chains a fallback CMap.
//!
//! ## Representation vs. the C
//!
//! MuPDF builds the map into a self-balancing **splay tree** (`cmap_splay`) for
//! O(log n) insertion with overlap resolution, then `pdf_sort_cmap` flattens the
//! tree into three flat, sorted arrays (`ranges` for values that fit in
//! `unsigned short`, `xranges` for wider values, `mranges` for one-to-many) that
//! `pdf_lookup_cmap` binary-searches. This port keeps the **observable lookup
//! behaviour** but drops the splay tree: mappings are pushed into plain `Vec`s
//! and [`CMap::sort`] sorts them by `low` so [`CMap::lookup`] /
//! [`CMap::lookup_full`] binary-search exactly as the C does. MuPDF's
//! `ranges`/`xranges` split is purely a storage-size optimisation (both are
//! searched identically), so they are unified here into one [`Range`] vector. The
//! splay tree's incremental **overlap splitting / adjacent-range merging** is not
//! reproduced -- predefined, ToUnicode, and CID CMaps do not carry overlapping
//! ranges -- so a later-added range does not rewrite an earlier overlap; this is
//! noted as the one deliberate simplification.
//!
//! ## Not ported (deliberately)
//!
//! The named Adobe predefined CMap **resource set** (`pdf-cmap-load.c`'s vendored
//! `cmaps/*.h` tables: GBK/UniGB/UniJIS/…) is *not* vendored -- that is tens of
//! thousands of lines of CJK data. [`CMap::new_identity`] implements Identity-H /
//! Identity-V directly (the overwhelmingly common Type0 encoding), and
//! [`CMap::load_predefined`] returns `None` for any other name so the caller can
//! fall back gracefully (see `font.rs`). `wmode` (vertical writing) is parsed and
//! stored but the vertical-metrics path is handled in `font.rs`.

use super::error::Result;
use super::lex::{Token, lex};
use super::stream::Stream;

/// The maximum 1-to-many mapping length (`PDF_MRANGE_CAP`). The PDF reference
/// caps ToUnicode CMaps at 512 bytes -> up to 256 characters.
// MuPDF: PDF_MRANGE_CAP (cmap.h:32)
pub const PDF_MRANGE_CAP: usize = 256;

/// One codespace range: an `n`-byte code in `[low, high]` (`pdf_cmap.codespace`).
// MuPDF: pdf_cmap.codespace[] (cmap.h:62)
#[derive(Clone, Copy, Debug)]
struct Codespace {
    /// Byte width of codes in this range.
    n: usize,
    low: u32,
    high: u32,
}

/// A contiguous one-to-one mapping: `low..=high` maps to `out..out+(high-low)`.
///
/// Unifies MuPDF's `pdf_range` (16-bit) and `pdf_xrange` (32-bit): both are
/// searched identically, so the width split is elided.
// MuPDF: pdf_range / pdf_xrange (cmap.h:34, 39)
#[derive(Clone, Copy, Debug)]
struct Range {
    low: u32,
    high: u32,
    out: u32,
}

/// A one-to-many mapping: the single code `low` maps to the sequence `out`.
// MuPDF: pdf_mrange + the `dict` payload (cmap.h:44)
#[derive(Clone, Debug)]
struct MRange {
    low: u32,
    out: Vec<u32>,
}

/// A character map: `pdf_cmap`. Decodes PDF-string bytes into codes via its
/// codespace ranges and maps each code to a CID or Unicode sequence.
#[derive(Clone, Debug, Default)]
pub struct CMap {
    /// The CMap's own name (`/CMapName`), e.g. `"Identity-H"`.
    pub cmap_name: String,
    /// The name given by a `usecmap` operator before the fallback is resolved.
    pub usecmap_name: String,
    /// The fallback CMap consulted when a lookup misses (`pdf_cmap.usecmap`).
    usecmap: Option<Box<CMap>>,
    /// Writing mode: 0 = horizontal, 1 = vertical (`/WMode`).
    wmode: i32,
    codespace: Vec<Codespace>,
    /// One-to-one ranges (sorted by `low` after [`CMap::sort`]).
    ranges: Vec<Range>,
    /// One-to-many ranges (sorted by `low` after [`CMap::sort`]).
    mranges: Vec<MRange>,
    /// Set once [`CMap::sort`] has run; guards the binary-search invariant.
    sorted: bool,
}

impl CMap {
    /// A fresh, empty CMap (`pdf_new_cmap`).
    // MuPDF: pdf_new_cmap (pdf-cmap.c:49)
    pub fn new() -> CMap {
        CMap::default()
    }

    /// The writing mode (0 = horizontal, 1 = vertical).
    // MuPDF: pdf_cmap_wmode (pdf-cmap.c:90)
    pub fn wmode(&self) -> i32 {
        self.wmode
    }

    // MuPDF: pdf_set_cmap_wmode (pdf-cmap.c:96)
    fn set_wmode(&mut self, wmode: i32) {
        self.wmode = wmode;
    }

    // MuPDF: pdf_add_codespace (pdf-cmap.c:102)
    fn add_codespace(&mut self, low: u32, high: u32, n: usize) {
        if self.codespace.len() + 1 == 40 {
            return; // "too many code space ranges"
        }
        self.codespace.push(Codespace { n, low, high });
    }

    // MuPDF: pdf_map_range_to_range -> add_range (pdf-cmap.c:687, 508)
    /// Add a contiguous one-to-one mapping `srclo..=srchi -> dstlo..`.
    fn map_range_to_range(&mut self, low: u32, high: u32, out: u32) {
        if low > high {
            return; // "range limits out of range"
        }
        if self.codespace.is_empty() {
            // "CMap is missing codespace range" -- default to 2-byte.
            self.add_codespace(0, 65535, 2);
        }
        self.ranges.push(Range { low, high, out });
        self.sorted = false;
    }

    // MuPDF: pdf_map_one_to_many (pdf-cmap.c:693)
    /// Add a one-to-many mapping for the single code `low`. Decodes UTF-16
    /// surrogate pairs in `values` (MuPDF: only the `*-UCS2` CMaps use these).
    fn map_one_to_many(&mut self, low: u32, values: &[u32]) {
        let mut vals: Vec<u32> = Vec::with_capacity(values.len());
        if values.len() >= 2 {
            // Merge surrogate pairs into astral code points (bug 706131).
            let mut i = 0;
            while i < values.len() {
                let mut hi = values[i];
                if (0xd800..0xdc00).contains(&hi) && i < values.len() - 1 {
                    let lo = values[i + 1];
                    if (0xdc00..0xe000).contains(&lo) {
                        hi = ((hi - 0xd800) << 10) + (lo - 0xdc00) + 0x10000;
                        i += 1;
                    }
                }
                vals.push(hi);
                i += 1;
            }
        } else {
            vals.extend_from_slice(values);
        }

        if vals.len() == 1 {
            self.map_range_to_range(low, low, vals[0]);
            return;
        }
        if vals.len() > PDF_MRANGE_CAP {
            return; // "ignoring one to many mapping"
        }
        if self.codespace.is_empty() {
            self.add_codespace(0, 65535, 2);
        }
        self.mranges.push(MRange { low, out: vals });
        self.sorted = false;
    }

    // MuPDF: pdf_sort_cmap (pdf-cmap.c:789) -- flatten the splay tree to sorted
    // arrays. Here the ranges are already flat; we only need the sort order the
    // binary-search lookups rely on.
    /// Sort the mapping tables by `low` so [`CMap::lookup`] can binary-search.
    fn sort(&mut self) {
        self.ranges.sort_by_key(|r| r.low);
        self.mranges.sort_by_key(|m| m.low);
        self.sorted = true;
    }

    // MuPDF: pdf_set_usecmap (pdf-cmap.c:74)
    /// Chain `usecmap` as the fallback; inherit its codespace if we have none.
    pub fn set_usecmap(&mut self, usecmap: CMap) {
        if self.codespace.is_empty() {
            self.codespace = usecmap.codespace.clone();
        }
        self.usecmap = Some(Box::new(usecmap));
    }

    // MuPDF: pdf_lookup_cmap (pdf-cmap.c:815)
    /// Look up the single-valued mapping of `cpt` (code point -> CID / Unicode),
    /// or `None` if unmapped (following `usecmap`).
    pub fn lookup(&self, cpt: u32) -> Option<u32> {
        // Binary search the sorted one-to-one ranges.
        let mut l = 0isize;
        let mut r = self.ranges.len() as isize - 1;
        while l <= r {
            let m = ((l + r) >> 1) as usize;
            let rg = &self.ranges[m];
            if cpt < rg.low {
                r = m as isize - 1;
            } else if cpt > rg.high {
                l = m as isize + 1;
            } else {
                return Some(cpt - rg.low + rg.out);
            }
        }
        if let Some(uc) = &self.usecmap {
            return uc.lookup(cpt);
        }
        None
    }

    // MuPDF: pdf_lookup_cmap_full (pdf-cmap.c:854)
    /// Look up the full (possibly multi-character) mapping of `cpt`, appending
    /// the result to `out`, and return the number of characters produced.
    pub fn lookup_full(&self, cpt: u32, out: &mut Vec<u32>) -> usize {
        // One-to-one ranges first.
        let mut l = 0isize;
        let mut r = self.ranges.len() as isize - 1;
        while l <= r {
            let m = ((l + r) >> 1) as usize;
            let rg = &self.ranges[m];
            if cpt < rg.low {
                r = m as isize - 1;
            } else if cpt > rg.high {
                l = m as isize + 1;
            } else {
                out.push(cpt - rg.low + rg.out);
                return 1;
            }
        }

        // One-to-many ranges (each a single point `low`).
        let mut l = 0isize;
        let mut r = self.mranges.len() as isize - 1;
        while l <= r {
            let m = ((l + r) >> 1) as usize;
            let mr = &self.mranges[m];
            if cpt < mr.low {
                r = m as isize - 1;
            } else if cpt > mr.low {
                l = m as isize + 1;
            } else {
                out.extend_from_slice(&mr.out);
                return mr.out.len();
            }
        }

        if let Some(uc) = &self.usecmap {
            return uc.lookup_full(cpt, out);
        }
        0
    }

    // MuPDF: pdf_decode_cmap (pdf-cmap.c:920)
    /// Use the codespace ranges to extract one code point from the front of
    /// `buf`, returning `(code, bytes_consumed)`. An unmatched byte consumes one
    /// byte and yields code 0 (matching MuPDF's fallthrough).
    pub fn decode(&self, buf: &[u8]) -> (u32, usize) {
        let len = buf.len().min(4);
        let mut c: u32 = 0;
        for (n, &byte) in buf.iter().take(len).enumerate() {
            c = (c << 8) | byte as u32;
            for cs in &self.codespace {
                if cs.n == n + 1 && c >= cs.low && c <= cs.high {
                    return (c, n + 1);
                }
            }
        }
        (0, 1)
    }

    // MuPDF: pdf_new_identity_cmap (pdf-cmap-load.c:114)
    /// Build an Identity-H (`wmode` 0) or Identity-V (`wmode` 1) CMap over
    /// `bytes`-byte codes (1 or 2).
    pub fn new_identity(wmode: i32, bytes: usize) -> CMap {
        let mut cmap = CMap::new();
        let high = (1u64 << (bytes * 8)) as u32 - 1;
        cmap.cmap_name = if wmode != 0 {
            "Identity-V"
        } else {
            "Identity-H"
        }
        .to_string();
        cmap.add_codespace(0, high, bytes);
        cmap.map_range_to_range(0, high, 0);
        cmap.sort();
        cmap.set_wmode(wmode);
        cmap
    }

    // MuPDF: pdf_load_builtin_cmap for Identity-* (pdf-cmap-load.c:298); the
    // named Adobe CJK resource set is deferred (see the module docs).
    /// Resolve a predefined CMap by name. Only Identity-H / Identity-V are
    /// implemented directly; every other name (the Adobe CJK resource set)
    /// returns `None` so the caller can fall back gracefully.
    pub fn load_predefined(name: &str) -> Option<CMap> {
        match name {
            "Identity-H" | "Identity" => Some(CMap::new_identity(0, 2)),
            "Identity-V" => Some(CMap::new_identity(1, 2)),
            _ => None,
        }
    }

    // MuPDF: pdf_remap_cmap (pdf-unicode.c:48)
    /// Compose this CMap (`code -> gid/cid`) with `ucs_from_cpt`
    /// (`code -> Unicode`) into a `gid/cid -> Unicode` CMap. This is how a font's
    /// ToUnicode is built: the raw ToUnicode stream maps *codes* to Unicode, and
    /// remapping it through the encoding yields a CID-keyed map.
    pub fn remap(&self, ucs_from_cpt: &CMap) -> CMap {
        let mut out = CMap::new();
        if let Some(uc) = &self.usecmap {
            out.usecmap = Some(Box::new(uc.remap(ucs_from_cpt)));
        }
        out.add_codespace(0, 0x7fff_ffff, 4);
        // Snapshot the ranges: map_range_to_range mutates self would-be, but we
        // read from `self` and write to `out`.
        let ranges = self.ranges.clone();
        for rg in &ranges {
            remap_range(&mut out, rg.low, rg.out, rg.high - rg.low, ucs_from_cpt);
        }
        // Font encoding CMaps have no one-to-many mappings, so mranges are
        // ignored (matching MuPDF's comment).
        out.sort();
        out
    }
}

// MuPDF: pdf_remap_cmap_range (pdf-unicode.c:30)
/// Map `gid+k -> Unicode(cpt+k)` for `k` in `0..=n`, into `out`.
fn remap_range(out: &mut CMap, cpt: u32, gid: u32, n: u32, ucs_from_cpt: &CMap) {
    for k in 0..=n {
        let mut ucsbuf: Vec<u32> = Vec::new();
        let ucslen = ucs_from_cpt.lookup_full(cpt.wrapping_add(k), &mut ucsbuf);
        if ucslen == 1 {
            out.map_range_to_range(gid + k, gid + k, ucsbuf[0]);
        } else if ucslen > 1 {
            out.map_one_to_many(gid + k, &ucsbuf);
        }
    }
}

// ---------------------------------------------------------------------------
// CMap stream parser (pdf-cmap-parse.c)
// ---------------------------------------------------------------------------

// MuPDF: pdf_code_from_string (pdf-cmap-parse.c:71)
/// Interpret a byte string as a big-endian integer code.
fn code_from_string(bytes: &[u8]) -> u32 {
    let mut a: u32 = 0;
    for &b in bytes {
        a = (a << 8) | b as u32;
    }
    a
}

/// Does `tok` match keyword `word` (allowing trailing garbage, like MuPDF's
/// prefix `strncmp`)?
// MuPDF: is_keyword (pdf-cmap-parse.c:32)
fn is_keyword(tok: &Token, word: &str) -> bool {
    match tok {
        Token::Keyword(k) => k.starts_with(word.as_bytes()),
        _ => false,
    }
}

/// Load and parse a CMap from a decoded CMap stream body (`pdf_load_cmap`).
///
/// Recognises `begincodespacerange`, `begincidrange`, `begincidchar`,
/// `beginbfrange`, `beginbfchar`, `/CMapName`, `/WMode`, and `usecmap`;
/// everything else (the PostScript preamble) is ignored, as in MuPDF.
// MuPDF: pdf_load_cmap (pdf-cmap-parse.c:372)
pub fn load(bytes: &[u8]) -> Result<CMap> {
    let mut f = Stream::from_slice(bytes);
    let mut cmap = CMap::new();
    let mut key: Vec<u8> = b".notdef".to_vec();

    loop {
        let tok = lex(&mut f)?;
        match &tok {
            Token::Eof => break,
            Token::Name(name) => {
                if name == b"CMapName" {
                    if let Token::Name(n) = lex(&mut f)? {
                        cmap.cmap_name = String::from_utf8_lossy(&n).into_owned();
                    }
                } else if name == b"WMode" {
                    if let Token::Int(i) = lex(&mut f)? {
                        cmap.set_wmode(i as i32);
                    }
                } else {
                    key = name.clone();
                }
            }
            Token::Keyword(_) => {
                if is_keyword(&tok, "endcmap") {
                    break;
                } else if is_keyword(&tok, "usecmap") {
                    cmap.usecmap_name = String::from_utf8_lossy(&key).into_owned();
                } else if is_keyword(&tok, "begincodespacerange") {
                    parse_codespace_range(&mut f, &mut cmap)?;
                } else if is_keyword(&tok, "beginbfchar") {
                    parse_bf_char(&mut f, &mut cmap)?;
                } else if is_keyword(&tok, "begincidchar") {
                    parse_cid_char(&mut f, &mut cmap)?;
                } else if is_keyword(&tok, "beginbfrange") {
                    parse_bf_range(&mut f, &mut cmap)?;
                } else if is_keyword(&tok, "begincidrange") {
                    parse_cid_range(&mut f, &mut cmap)?;
                }
                // else: ignore
            }
            _ => { /* ignore everything else */ }
        }
    }

    cmap.sort();
    Ok(cmap)
}

/// Consume tokens up to and including a keyword matching `end` (or EOF/Error).
// MuPDF: skip_to_keyword (pdf-cmap-parse.c:39)
fn skip_to_keyword(f: &mut Stream, end: &str) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if is_keyword(&tok, end) || matches!(tok, Token::Error | Token::Eof) {
            return Ok(());
        }
    }
}

/// Consume tokens up to and including one equal to `end` (or EOF/Error).
// MuPDF: skip_to_token (pdf-cmap-parse.c:55)
fn skip_to_close_array(f: &mut Stream) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if matches!(tok, Token::CloseArray | Token::Error | Token::Eof) {
            return Ok(());
        }
    }
}

// MuPDF: pdf_parse_codespace_range (pdf-cmap-parse.c:106)
fn parse_codespace_range(f: &mut Stream, cmap: &mut CMap) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if is_keyword(&tok, "endcodespacerange") {
            return Ok(());
        }
        match tok {
            Token::String(lo) => {
                let low = code_from_string(&lo);
                match lex(f)? {
                    Token::String(hi) => {
                        let high = code_from_string(&hi);
                        cmap.add_codespace(low, high, hi.len());
                    }
                    _ => return skip_to_keyword(f, "endcodespacerange"),
                }
            }
            _ => return skip_to_keyword(f, "endcodespacerange"),
        }
    }
}

// MuPDF: pdf_parse_cid_range (pdf-cmap-parse.c:142)
fn parse_cid_range(f: &mut Stream, cmap: &mut CMap) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if is_keyword(&tok, "endcidrange") {
            return Ok(());
        }
        let lo = match tok {
            Token::String(s) => code_from_string(&s),
            _ => return skip_to_keyword(f, "endcidrange"),
        };
        let hi = match lex(f)? {
            Token::String(s) => code_from_string(&s),
            _ => return skip_to_keyword(f, "endcidrange"),
        };
        let dst = match lex(f)? {
            Token::Int(i) => i as u32,
            _ => return skip_to_keyword(f, "endcidrange"),
        };
        cmap.map_range_to_range(lo, hi, dst);
    }
}

// MuPDF: pdf_parse_cid_char (pdf-cmap-parse.c:185)
fn parse_cid_char(f: &mut Stream, cmap: &mut CMap) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if is_keyword(&tok, "endcidchar") {
            return Ok(());
        }
        let src = match tok {
            Token::String(s) => code_from_string(&s),
            _ => return skip_to_keyword(f, "endcidchar"),
        };
        let dst = match lex(f)? {
            Token::Int(i) => i as u32,
            _ => return skip_to_keyword(f, "endcidchar"),
        };
        cmap.map_range_to_range(src, src, dst);
    }
}

// MuPDF: pdf_parse_bf_char (pdf-cmap-parse.c:331)
fn parse_bf_char(f: &mut Stream, cmap: &mut CMap) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if is_keyword(&tok, "endbfchar") {
            return Ok(());
        }
        let src = match tok {
            Token::String(s) => code_from_string(&s),
            _ => return skip_to_keyword(f, "endbfchar"),
        };
        // Note: does not handle /dstName.
        let dst = match lex(f)? {
            Token::String(s) => s,
            _ => return skip_to_keyword(f, "endbfchar"),
        };
        if dst.len() / 2 != 0 {
            let vals = utf16be_units(&dst);
            cmap.map_one_to_many(src, &vals);
        }
    }
}

// MuPDF: pdf_parse_bf_range (pdf-cmap-parse.c:253)
fn parse_bf_range(f: &mut Stream, cmap: &mut CMap) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if is_keyword(&tok, "endbfrange") {
            return Ok(());
        }
        let lo = match tok {
            Token::String(s) => code_from_string(&s),
            _ => return skip_to_keyword(f, "endbfrange"),
        };
        let hi = match lex(f)? {
            Token::String(s) => code_from_string(&s),
            _ => return skip_to_keyword(f, "endbfrange"),
        };
        if lo > 65535 || hi > 65535 || lo > hi {
            return skip_to_keyword(f, "endbfrange");
        }

        match lex(f)? {
            Token::String(dst) => {
                if dst.len() == 2 {
                    let d = code_from_string(&dst);
                    cmap.map_range_to_range(lo, hi, d);
                } else if dst.len() / 2 != 0 {
                    // Multi-character destination: increment the last unit for
                    // each successive source code.
                    let mut units = utf16be_units(&dst);
                    let last = units.len() - 1;
                    let mut cur = lo;
                    while cur <= hi {
                        cmap.map_one_to_many(cur, &units);
                        units[last] = units[last].wrapping_add(1);
                        cur += 1;
                    }
                }
            }
            Token::OpenArray => parse_bf_range_array(f, cmap, lo)?,
            _ => return skip_to_keyword(f, "endbfrange"),
        }
    }
}

// MuPDF: pdf_parse_bf_range_array (pdf-cmap-parse.c:219)
fn parse_bf_range_array(f: &mut Stream, cmap: &mut CMap, mut lo: u32) -> Result<()> {
    loop {
        let tok = lex(f)?;
        if matches!(tok, Token::CloseArray) {
            return Ok(());
        }
        // Note: does not handle [ /Name /Name ... ].
        let s = match tok {
            Token::String(s) => s,
            _ => return skip_to_close_array(f),
        };
        if s.len() / 2 != 0 {
            let vals = utf16be_units(&s);
            cmap.map_one_to_many(lo, &vals);
        }
        lo += 1;
    }
}

/// Split a big-endian byte string into 2-byte code units (capped at
/// [`PDF_MRANGE_CAP`]), matching MuPDF's `pdf_code_from_string(&buf[i*2], 2)`
/// loop.
fn utf16be_units(bytes: &[u8]) -> Vec<u32> {
    let len = (bytes.len() / 2).min(PDF_MRANGE_CAP);
    (0..len)
        .map(|i| code_from_string(&bytes[i * 2..i * 2 + 2]))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_h_maps_code_to_itself() {
        let cmap = CMap::new_identity(0, 2);
        assert_eq!(cmap.lookup(0x0041), Some(0x0041));
        assert_eq!(cmap.lookup(0xABCD), Some(0xABCD));
        assert_eq!(cmap.wmode(), 0);
        // Identity-V flips the writing mode.
        assert_eq!(CMap::new_identity(1, 2).wmode(), 1);
    }

    #[test]
    fn codespace_decode_picks_one_or_two_bytes() {
        // A CMap with a 1-byte codespace [00,80] and a 2-byte [8140,9FFC]
        // (Shift-JIS-like): the decoder must pick the right width per lead byte.
        let mut cmap = CMap::new();
        cmap.add_codespace(0x00, 0x80, 1);
        cmap.add_codespace(0x8140, 0x9ffc, 2);

        // 0x41 is a single-byte code.
        let (c, n) = cmap.decode(&[0x41, 0x42]);
        assert_eq!((c, n), (0x41, 1));
        // 0x81 0x40 is a two-byte code (0x81 alone matches no 1-byte range).
        let (c, n) = cmap.decode(&[0x81, 0x40]);
        assert_eq!((c, n), (0x8140, 2));
    }

    #[test]
    fn identity_codespace_is_two_bytes() {
        let cmap = CMap::new_identity(0, 2);
        let (c, n) = cmap.decode(&[0x12, 0x34, 0x56]);
        assert_eq!((c, n), (0x1234, 2));
    }

    #[test]
    fn parse_tounicode_bfchar_and_bfrange() {
        // A minimal ToUnicode CMap: one bfchar and one bfrange.
        let src = b"/CIDInit /ProcSet findresource begin\n\
12 dict begin\nbegincmap\n\
1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n\
1 beginbfchar\n<0041> <0041>\nendbfchar\n\
1 beginbfrange\n<0060> <0062> <0030>\nendbfrange\n\
endcmap\nend\nend";
        let cmap = super::load(src).unwrap();

        // bfchar: code 0x41 -> U+0041 'A'.
        let mut out = Vec::new();
        assert_eq!(cmap.lookup_full(0x41, &mut out), 1);
        assert_eq!(out, vec![0x0041]);

        // bfrange: 0x60->0x30, 0x61->0x31, 0x62->0x32.
        assert_eq!(cmap.lookup(0x60), Some(0x30));
        assert_eq!(cmap.lookup(0x61), Some(0x31));
        assert_eq!(cmap.lookup(0x62), Some(0x32));
    }

    #[test]
    fn parse_bfchar_multi_character() {
        // code 0x01 maps to the two-char sequence "fi" (U+0066 U+0069).
        let src = b"begincmap\n1 begincodespacerange\n<00> <ff>\nendcodespacerange\n\
1 beginbfchar\n<01> <00660069>\nendbfchar\nendcmap";
        let cmap = super::load(src).unwrap();
        let mut out = Vec::new();
        assert_eq!(cmap.lookup_full(0x01, &mut out), 2);
        assert_eq!(out, vec![0x0066, 0x0069]);
    }

    #[test]
    fn parse_cidrange_and_cidchar() {
        let src = b"begincmap\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n\
1 begincidrange\n<0000> <00ff> 0\nendcidrange\n\
1 begincidchar\n<0100> 300\nendcidchar\nendcmap";
        let cmap = super::load(src).unwrap();
        // cidrange 0000..00ff -> CID 0.., so 0x0041 -> 0x41.
        assert_eq!(cmap.lookup(0x0041), Some(0x0041));
        // cidchar 0x0100 -> 300.
        assert_eq!(cmap.lookup(0x0100), Some(300));
    }

    #[test]
    fn remap_composes_encoding_with_tounicode() {
        // encoding: 2-byte code -> CID (Identity here). tounicode: code -> ucs.
        let mut encoding = CMap::new();
        encoding.add_codespace(0, 0xffff, 2);
        encoding.map_range_to_range(0x00, 0x02, 0x10); // codes 0..2 -> CIDs 16..18
        encoding.sort();

        let mut tounicode = CMap::new();
        tounicode.add_codespace(0, 0xffff, 2);
        tounicode.map_range_to_range(0x00, 0x00, 0x41); // code 0 -> 'A'
        tounicode.map_range_to_range(0x01, 0x01, 0x42); // code 1 -> 'B'
        tounicode.sort();

        // Composed: CID 16 -> 'A', CID 17 -> 'B'.
        let cid_to_ucs = encoding.remap(&tounicode);
        assert_eq!(cid_to_ucs.lookup(0x10), Some(0x41));
        assert_eq!(cid_to_ucs.lookup(0x11), Some(0x42));
    }

    #[test]
    fn usecmap_fallback_is_consulted() {
        let mut base = CMap::new();
        base.add_codespace(0, 0xffff, 2);
        base.map_range_to_range(0x00, 0xff, 0x00);
        base.sort();

        let mut top = CMap::new();
        top.map_range_to_range(0x1000, 0x1000, 0x99);
        top.sort();
        top.set_usecmap(base);

        // Own mapping.
        assert_eq!(top.lookup(0x1000), Some(0x99));
        // Falls through to usecmap.
        assert_eq!(top.lookup(0x0041), Some(0x0041));
        // Inherited the codespace from usecmap.
        assert_eq!(top.decode(&[0x00, 0x41]), (0x0041, 2));
    }

    #[test]
    fn load_predefined_identity_only() {
        assert!(CMap::load_predefined("Identity-H").is_some());
        assert!(CMap::load_predefined("Identity-V").is_some());
        // Named Adobe CJK CMaps are deferred -> None (graceful fallback).
        assert!(CMap::load_predefined("UniGB-UCS2-H").is_none());
    }
}
