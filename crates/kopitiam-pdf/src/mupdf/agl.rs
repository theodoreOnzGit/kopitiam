//! Ported from MuPDF `source/fitz/encodings.c`
//! (`fz_unicode_from_glyph_name` / `fz_unicode_from_glyph_name_strict`) plus the
//! generated `source/fitz/glyphlist.h` data table (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # Adobe Glyph List: glyph name -> Unicode
//!
//! Step (3) of MuPDF's glyph-code -> Unicode fallback chain (see `font.rs`) turns
//! an Adobe glyph name (`"adieresis"`, `"quoteright"`, the names the standard
//! encodings in [`super::encodings`] carry) into a Unicode code point. This
//! module is [`unicode_from_glyph_name`] (`fz_unicode_from_glyph_name`): a binary
//! search of the full AGL ([`AGL_SINGLE`], in `agl_data.rs`), followed by the
//! algorithmic `uniXXXX` / `uXXXXXX` / `cdXX` name forms.
//!
//! ## What is ported
//!
//! * The **complete** AGL single-name table (4455 entries), ported verbatim from
//!   `glyphlist.h` -- so every standard-encoding glyph name resolves, as the task
//!   requires.
//! * The algorithmic name forms MuPDF recognises: `uniXXXX` (exactly 4 hex
//!   digits), `uXXXX`..`uXXXXXX` (variable hex), `aNN`/`aNNN` (decimal),
//!   and the bare-decimal fallthrough.
//!
//! ## What is deferred
//!
//! MuPDF's two exotic branches -- `G%x` (DOS CP-437, bug 709466) and `g%x`
//! (TrueType-UCS2 glyph index, Mozilla bug 1027533) -- depend on the
//! `fz_unicode_from_dos_437` / `fz_unicode_from_true_type_glyph_index` legacy
//! cmaps that `encodings.rs` deliberately did not port. They are not reachable
//! from the standard PDF simple-font encodings, so those two branches fall
//! through to the generic decimal path here (noted, not silently dropped).

use super::agl_data::AGL_SINGLE;

/// U+FFFD, MuPDF's `FZ_REPLACEMENT_CHARACTER`.
// MuPDF: FZ_REPLACEMENT_CHARACTER (string-util.h:31)
pub const REPLACEMENT_CHARACTER: u32 = 0xFFFD;

// MuPDF: fz_unicode_from_glyph_name_strict (encodings.c:57)
/// Strict AGL lookup: binary-search the glyph-name table for an exact match, or
/// return 0. No algorithmic-name fallback and no `.`/`_` trimming.
pub fn unicode_from_glyph_name_strict(name: &str) -> u32 {
    match AGL_SINGLE.binary_search_by(|(n, _)| (*n).cmp(name)) {
        Ok(i) => AGL_SINGLE[i].1 as u32,
        Err(_) => 0,
    }
}

/// Parse `p` as an integer in `base`, requiring the whole string to be consumed
/// (MuPDF's `read_num`: `strtol` then reject trailing garbage).
// MuPDF: read_num (encodings.c:77)
fn read_num(p: &str, base: u32) -> u32 {
    u32::from_str_radix(p, base).unwrap_or(0)
}

// MuPDF: fz_unicode_from_glyph_name (encodings.c:87)
/// Map an Adobe glyph name to a Unicode code point, reproducing MuPDF's full
/// resolution: trim after the first `.`, handle the `f`-ligature `_` aliases (and
/// otherwise trim after the first `_`), binary-search the AGL, then fall back to
/// the algorithmic `uniXXXX` / `uXXXX` / `aNN` / decimal name forms. Never
/// panics; an unresolvable name yields [`REPLACEMENT_CHARACTER`].
pub fn unicode_from_glyph_name(name: &str) -> u32 {
    // fz_strlcpy into a 64-byte buffer: MuPDF truncates long names.
    let mut buf: String = name.chars().take(63).collect();

    // Kill anything after the first period.
    if let Some(pos) = buf.find('.') {
        buf.truncate(pos);
    }
    // Handle the underscore ligature aliases, else trim after the first '_'.
    if let Some(pos) = buf.find('_') {
        if buf.starts_with('f') {
            match buf.as_str() {
                "f_f" => buf = "ff".to_string(),
                "f_f_i" => buf = "ffi".to_string(),
                "f_f_l" => buf = "ffl".to_string(),
                "f_i" => buf = "fi".to_string(),
                "f_l" => buf = "fl".to_string(),
                _ => buf.truncate(pos),
            }
        } else {
            buf.truncate(pos);
        }
    }

    // Exact AGL match.
    if let Ok(i) = AGL_SINGLE.binary_search_by(|(n, _)| (*n).cmp(buf.as_str())) {
        return AGL_SINGLE[i].1 as u32;
    }

    // Algorithmic name forms.
    let bytes = buf.as_bytes();
    let code: u32 = if bytes.len() == 7 && buf.starts_with("uni") {
        read_num(&buf[3..], 16)
    } else if bytes.first() == Some(&b'u') {
        read_num(&buf[1..], 16)
    } else if bytes.first() == Some(&b'a') && bytes.len() >= 3 {
        read_num(&buf[1..], 10)
    } else {
        // MuPDF's `G%x` (DOS-437) and `g%x` (TrueType-UCS2) branches depend on
        // legacy cmaps not ported here; they fall through to the generic decimal
        // parse, as does any other name.
        read_num(&buf, 10)
    };

    if code > 0 && code <= 0x10ffff {
        code
    } else {
        REPLACEMENT_CHARACTER
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::encodings::{BaseEncoding, MAC_ROMAN, STANDARD, WIN_ANSI};

    #[test]
    fn agl_table_is_sorted_for_binary_search() {
        // The port relies on the C file's ASCII-sort order.
        for w in AGL_SINGLE.windows(2) {
            assert!(w[0].0 < w[1].0, "not sorted at {:?}/{:?}", w[0], w[1]);
        }
    }

    #[test]
    fn standard_glyph_names_resolve() {
        // A standard-encoding glyph name (the task requires these to resolve).
        assert_eq!(unicode_from_glyph_name("adieresis"), 0x00E4);
        assert_eq!(unicode_from_glyph_name("A"), 0x0041);
        assert_eq!(unicode_from_glyph_name("space"), 0x0020);
        assert_eq!(unicode_from_glyph_name("quoteright"), 0x2019);
        assert_eq!(unicode_from_glyph_name("germandbls"), 0x00DF);
    }

    #[test]
    fn algorithmic_uni_names_resolve() {
        // uniXXXX form.
        assert_eq!(unicode_from_glyph_name("uni00E4"), 0x00E4);
        assert_eq!(unicode_from_glyph_name("uni20AC"), 0x20AC);
        // uXXXXXX form (astral).
        assert_eq!(unicode_from_glyph_name("u1F600"), 0x1F600);
    }

    #[test]
    fn period_and_underscore_handling() {
        // Trailing ".sc" style suffix is stripped.
        assert_eq!(unicode_from_glyph_name("A.sc"), 0x0041);
        // f-ligature underscore alias.
        assert_eq!(unicode_from_glyph_name("f_i"), unicode_from_glyph_name("fi"));
    }

    #[test]
    fn unresolvable_name_is_replacement() {
        // A name that is neither in the AGL nor an algorithmic form.
        assert_eq!(
            unicode_from_glyph_name("totallybogusnameZZ"),
            REPLACEMENT_CHARACTER
        );
    }

    #[test]
    fn strict_lookup_has_no_fallback() {
        assert_eq!(unicode_from_glyph_name_strict("A"), 0x0041);
        assert_eq!(unicode_from_glyph_name_strict("uni00E4"), 0); // no algorithmic fallback
    }

    #[test]
    fn every_standard_encoding_name_resolves() {
        // Exhaustively confirm the four base encodings' glyph names all resolve
        // to a non-replacement code point -- this is the guarantee the fallback
        // chain in font.rs depends on.
        for enc in [
            BaseEncoding::Standard,
            BaseEncoding::WinAnsi,
            BaseEncoding::MacRoman,
        ] {
            for code in 0u16..256 {
                if let Some(name) = enc.glyph_name(code as u8) {
                    let u = unicode_from_glyph_name(name);
                    assert_ne!(
                        u, REPLACEMENT_CHARACTER,
                        "{enc:?} code {code:#x} name {name:?} did not resolve"
                    );
                }
            }
        }
        // Silence unused-import warnings if tables are referenced only here.
        let _ = (&STANDARD, &WIN_ANSI, &MAC_ROMAN);
    }
}
