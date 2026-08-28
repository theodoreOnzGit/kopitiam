//! Substituting a bundled face for a font the PDF **does not embed** — the
//! reason Word- and LibreOffice-produced documents render as text rather than
//! solid boxes.
//!
//! # The problem this solves
//!
//! A PDF may name a font and leave the viewer to supply it (PDF 32000-1:2008
//! §9.6.2.2 for the standard 14, §9.8.2 for descriptor-based substitution).
//! Word and LibreOffice do this constantly: `/BaseFont /ArialMT`, `/Subtype
//! /TrueType`, and no `/FontFile*` anywhere. Before this module, such a font
//! reached [`super::font::Font`] with nothing to decode, `outline_source()`
//! returned [`OutlineSource::None`](super::font::OutlineSource::None), and the
//! draw device emitted a filled advance box per glyph. A maintainer-supplied
//! workbook had 51 such font instances against 29 embedded ones — the whole
//! document was unreadable.
//!
//! # Provenance
//!
//! The **selection heuristic** below is ported from `hayro-interpret 0.7.0`,
//! `src/font/standard_font.rs`'s `select_standard_font` (Apache-2.0 OR MIT, ©
//! Laurenz Stampfl — <https://github.com/LaurenzV/hayro>). Ported rather than
//! called because it is `pub(crate)` there and so unreachable from outside
//! that crate; [`StandardFont`] and its `get_font_data()` *are* public, so the
//! **font data is used directly rather than re-bundled**. See
//! `docs/ACKNOWLEDGEMENTS.md`.
//!
//! The bundled faces are the **Foxit** base-14 set that ships inside
//! `hayro-interpret`'s `assets/` under its default `embed-fonts` feature,
//! extracted from PDFium (BSD-3-Clause, © 2014 PDFium Authors; original code ©
//! Foxit Software Inc.). They are metric-compatible stand-ins for the standard
//! 14, which is what makes them safe here: `/Widths` from the PDF still drives
//! layout, and a face with the wrong metrics would reflow the text even once
//! glyphs appeared.
//!
//! Despite their `.pfb` extension the payloads are **bare CFF** (they begin
//! `01 00 04 02` — a CFF header, not a PFB `80 01` segment marker), which is
//! why [`super::glyph_cff::CffProgram`] parses them directly and no Type1
//! path is involved.
//!
//! # Why not bundle our own face
//!
//! CLAUDE.md's hard rule: prefer an existing, actively-maintained, pure-Rust
//! crate with a genuinely usable public API over writing or shipping an
//! equivalent ourselves. `hayro` is already a mandatory dependency (it is the
//! cross-engine fallback renderer), its `embed-fonts` feature is **on by
//! default**, so these bytes are already in the binary. Bundling a second set
//! would add megabytes and a second licence obligation to gain nothing.
//!
//! The reachability precondition that same rule demands was checked, not
//! assumed — `hayro` re-exports `hayro_interpret` wholesale, `pub mod font`
//! re-exports `StandardFont`, and `get_font_data()` is `pub`. (CLAUDE.md
//! records the opposite outcome for `hayro-font`, whose API turned out to be
//! `pub(crate)`-private; this is the same check, passed.)
//!
//! # Known ceiling: no AFM widths
//!
//! When a PDF omits `/Widths` entirely — legal for the standard 14 — the
//! correct widths come from the font's AFM metrics, which hayro ships as
//! generated tables and this module does not port. Such a font gets
//! `/MissingWidth` for every glyph and will lay out wrongly even though the
//! glyphs now appear. Word/LibreOffice always write `/Widths`, so the common
//! case is unaffected; see the bead/issue for the gap.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use hayro::hayro_interpret::font::StandardFont;

use super::glyph_cff::CffProgram;
use super::object::Object;
use super::xref::PdfDocument;

/// Strip a subset prefix: six uppercase letters and a `+`, as in
/// `ABCDEF+ArialMT` (PDF 32000-1:2008 §9.6.4). hayro:
/// `font::strip_subset_prefix`.
fn strip_subset_prefix(name: &str) -> &str {
    let bytes = name.as_bytes();
    if bytes.len() > 7
        && bytes[6] == b'+'
        && bytes[..6].iter().all(|b| b.is_ascii_uppercase())
    {
        &name[7..]
    } else {
        name
    }
}

/// The three families the standard 14 collapse into, before bold/italic.
enum Family {
    Helvetica,
    Courier,
    Times,
}

/// Pick a standard-14 face for a non-embedded font.
///
/// Ported from hayro-interpret 0.7.0 `select_standard_font`
/// (`src/font/standard_font.rs`). Two stages, in this order:
///
/// 1. **Literal match** against the 14 standard PostScript names.
/// 2. **Keyword heuristic** over the lowercased base name, combined with the
///    descriptor's `/FontWeight` and `/ItalicAngle`. This is the half that
///    makes Word and LibreOffice output readable: `ArialMT` contains "arial"
///    and maps to Helvetica, `Arial-BoldMT` adds bold, `TimesNewRomanPSMT`
///    contains "times" and maps to Times.
///
/// Returns `None` when nothing matches, which is deliberate: an unrecognised
/// symbolic font substituted with Helvetica would render confident nonsense,
/// and the advance-box fallback is the more honest answer.
pub fn select_standard_font(base_font: &str, descriptor: &Object, doc: &PdfDocument) -> Option<StandardFont> {
    let name = strip_subset_prefix(base_font);

    // Stage 1: the literal standard-14 names (§9.6.2.2, Table 109).
    match name {
        "Helvetica" => return Some(StandardFont::Helvetica),
        "Helvetica-Bold" => return Some(StandardFont::HelveticaBold),
        "Helvetica-Oblique" => return Some(StandardFont::HelveticaOblique),
        "Helvetica-BoldOblique" => return Some(StandardFont::HelveticaBoldOblique),
        "Courier" => return Some(StandardFont::Courier),
        "Courier-Bold" => return Some(StandardFont::CourierBold),
        "Courier-Oblique" => return Some(StandardFont::CourierOblique),
        "Courier-BoldOblique" => return Some(StandardFont::CourierBoldOblique),
        "Times-Roman" => return Some(StandardFont::TimesRoman),
        "Times-Bold" => return Some(StandardFont::TimesBold),
        "Times-Italic" => return Some(StandardFont::TimesItalic),
        "Times-BoldItalic" => return Some(StandardFont::TimesBoldItalic),
        "Symbol" => return Some(StandardFont::Symbol),
        "ZapfDingbats" => return Some(StandardFont::ZapfDingBats),
        _ => {}
    }

    // Stage 2: keywords in the name, plus what the descriptor asserts.
    let lower = name.to_ascii_lowercase();

    let weight = descriptor
        .dict_gets("FontWeight")
        .and_then(|o| doc.resolve(o).ok())
        .map(|o| o.to_real());
    let italic_angle = descriptor
        .dict_gets("ItalicAngle")
        .and_then(|o| doc.resolve(o).ok())
        .map(|o| o.to_real());

    let is_bold = weight.is_some_and(|w| w >= 700.0)
        || lower.contains("bold")
        || lower.contains("demi");
    let is_italic = italic_angle.is_some_and(|a| a != 0.0)
        || lower.contains("italic")
        || lower.contains("oblique");

    // Deliberate divergence from hayro: it pairs each family with an `exact`
    // flag (true for "helvetica"/"courier"/"times", false for the generic
    // "sans"/"mono"/"serif"), which it uses elsewhere to decide how much to
    // trust the match. Nothing here consumes that flag, so the exact and
    // generic keywords are folded into one arm per family. Behaviour is
    // identical; only the discarded flag differs.
    //
    // "arial" -> Helvetica is the Word/LibreOffice case, and it is safe
    // precisely because Arial was drawn metric-compatible with Helvetica.
    let family = if lower.contains("helvetica")
        || lower.contains("arial")
        || lower.contains("sans")
    {
        Family::Helvetica
    } else if lower.contains("courier") || lower.contains("mono") {
        Family::Courier
    } else if lower.contains("times") || lower.contains("serif") {
        Family::Times
    } else if lower.contains("zapfdingbats") || lower.contains("dingbats") {
        return Some(StandardFont::ZapfDingBats);
    } else {
        return None;
    };

    Some(match (family, is_bold, is_italic) {
        (Family::Helvetica, false, false) => StandardFont::Helvetica,
        (Family::Helvetica, true, false) => StandardFont::HelveticaBold,
        (Family::Helvetica, false, true) => StandardFont::HelveticaOblique,
        (Family::Helvetica, true, true) => StandardFont::HelveticaBoldOblique,
        (Family::Courier, false, false) => StandardFont::Courier,
        (Family::Courier, true, false) => StandardFont::CourierBold,
        (Family::Courier, false, true) => StandardFont::CourierOblique,
        (Family::Courier, true, true) => StandardFont::CourierBoldOblique,
        (Family::Times, false, false) => StandardFont::TimesRoman,
        (Family::Times, true, false) => StandardFont::TimesBold,
        (Family::Times, false, true) => StandardFont::TimesItalic,
        (Family::Times, true, true) => StandardFont::TimesBoldItalic,
    })
}

/// Parsed substitute programs, keyed by face.
///
/// Parsing a CFF is not free and the same handful of faces recur on every page
/// of a document (and across documents in one session), so each is parsed at
/// most once per process. `Arc` because [`super::font::Font`] is cloned into a
/// font cache and must not duplicate the program bytes — the same reasoning as
/// `Font::program`.
fn cache() -> &'static Mutex<HashMap<usize, Option<Arc<CffProgram>>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Option<Arc<CffProgram>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A stable key for a [`StandardFont`], which is `Copy` but not `Hash`.
fn key(f: StandardFont) -> usize {
    match f {
        StandardFont::Helvetica => 0,
        StandardFont::HelveticaBold => 1,
        StandardFont::HelveticaOblique => 2,
        StandardFont::HelveticaBoldOblique => 3,
        StandardFont::Courier => 4,
        StandardFont::CourierBold => 5,
        StandardFont::CourierOblique => 6,
        StandardFont::CourierBoldOblique => 7,
        StandardFont::TimesRoman => 8,
        StandardFont::TimesBold => 9,
        StandardFont::TimesItalic => 10,
        StandardFont::TimesBoldItalic => 11,
        StandardFont::ZapfDingBats => 12,
        StandardFont::Symbol => 13,
    }
}

/// The parsed CFF program for a standard face, or `None` if it will not parse.
///
/// A parse failure is cached too: it cannot succeed on a later call (the bytes
/// are compiled in and never change), so retrying per glyph would burn time to
/// reach the same answer.
pub fn program_for(face: StandardFont) -> Option<Arc<CffProgram>> {
    let k = key(face);
    let mut cache = cache().lock().ok()?;
    cache
        .entry(k)
        .or_insert_with(|| {
            let (data, _index) = face.get_font_data();
            let bytes: &[u8] = (*data).as_ref();
            CffProgram::parse(bytes).map(Arc::new)
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_descriptor() -> Object {
        Object::new_dict()
    }

    /// A tiny document, only so `select_standard_font` has something to
    /// resolve indirect descriptor entries against. Offsets are computed, not
    /// hand-written -- the first cut of this fixture had them wrong and every
    /// test that opened it failed for a reason that had nothing to do with
    /// font selection.
    fn doc() -> PdfDocument {
        let bodies = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 10 10] >>",
            "<< /Type /Page /Parent 2 0 R >>",
        ];
        let mut pdf: Vec<u8> = b"%PDF-1.5\n".to_vec();
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n{body}\nendobj\n").as_bytes());
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
        PdfDocument::open(pdf).unwrap()
    }

    #[test]
    fn subset_prefixes_are_stripped() {
        assert_eq!(strip_subset_prefix("ABCDEF+ArialMT"), "ArialMT");
        assert_eq!(strip_subset_prefix("ArialMT"), "ArialMT");
        // Not six uppercase letters -> not a subset prefix.
        assert_eq!(strip_subset_prefix("Abcdef+ArialMT"), "Abcdef+ArialMT");
        assert_eq!(strip_subset_prefix("ABC+X"), "ABC+X");
    }

    #[test]
    fn the_literal_standard_14_names_match() {
        let d = doc();
        for (name, want) in [
            ("Helvetica", StandardFont::Helvetica),
            ("Times-Roman", StandardFont::TimesRoman),
            ("Courier-BoldOblique", StandardFont::CourierBoldOblique),
            ("Symbol", StandardFont::Symbol),
            ("ZapfDingbats", StandardFont::ZapfDingBats),
        ] {
            let got = select_standard_font(name, &no_descriptor(), &d).expect(name);
            assert_eq!(key(got), key(want), "{name}");
        }
    }

    /// The regression this module exists for: the exact `/BaseFont` names from
    /// the maintainer's Word-produced workbook, which rendered as boxes.
    #[test]
    fn word_and_libreoffice_names_resolve() {
        let d = doc();
        for (name, want) in [
            ("ArialMT", StandardFont::Helvetica),
            ("Arial-BoldMT", StandardFont::HelveticaBold),
            ("Arial-ItalicMT", StandardFont::HelveticaOblique),
            ("Arial-BoldItalicMT", StandardFont::HelveticaBoldOblique),
            ("TimesNewRomanPSMT", StandardFont::TimesRoman),
            ("TimesNewRomanPS-BoldMT", StandardFont::TimesBold),
            ("CourierNewPSMT", StandardFont::Courier),
            ("LiberationSans", StandardFont::Helvetica),
            ("DejaVuSerif", StandardFont::TimesRoman),
        ] {
            let got = select_standard_font(name, &no_descriptor(), &d).expect(name);
            assert_eq!(key(got), key(want), "{name}");
        }
    }

    /// A subset-prefixed Word name must resolve the same as the bare one.
    #[test]
    fn a_subset_prefixed_name_still_resolves() {
        let d = doc();
        let got = select_standard_font("ABCDEF+Arial-BoldMT", &no_descriptor(), &d).unwrap();
        assert_eq!(key(got), key(StandardFont::HelveticaBold));
    }

    /// The descriptor can assert bold/italic that the name does not say.
    #[test]
    fn the_descriptor_supplies_bold_and_italic() {
        let d = doc();
        let mut desc = Object::new_dict();
        desc.dict_put("FontWeight", Object::new_int(700));
        assert_eq!(
            key(select_standard_font("ArialMT", &desc, &d).unwrap()),
            key(StandardFont::HelveticaBold)
        );

        let mut desc = Object::new_dict();
        desc.dict_put("ItalicAngle", Object::new_real(-12.0));
        assert_eq!(
            key(select_standard_font("ArialMT", &desc, &d).unwrap()),
            key(StandardFont::HelveticaOblique)
        );

        // A weight below 700 must NOT read as bold.
        let mut desc = Object::new_dict();
        desc.dict_put("FontWeight", Object::new_int(400));
        assert_eq!(
            key(select_standard_font("ArialMT", &desc, &d).unwrap()),
            key(StandardFont::Helvetica)
        );
    }

    /// An unrecognised font yields None rather than a confident wrong guess —
    /// substituting Helvetica for an unknown symbolic font would render
    /// plausible nonsense, which is worse than an honest box.
    #[test]
    fn an_unrecognised_name_is_refused() {
        let d = doc();
        assert!(select_standard_font("Wingdings", &no_descriptor(), &d).is_none());
        assert!(select_standard_font("SomeCorporateIcons", &no_descriptor(), &d).is_none());
    }

    /// Every one of the 14 faces must actually parse with our own CFF decoder,
    /// and give a real outline. If hayro ever changes the bundled payloads,
    /// this is what catches it.
    #[test]
    fn all_fourteen_faces_parse_and_yield_outlines() {
        for face in [
            StandardFont::Helvetica,
            StandardFont::HelveticaBold,
            StandardFont::HelveticaOblique,
            StandardFont::HelveticaBoldOblique,
            StandardFont::Courier,
            StandardFont::CourierBold,
            StandardFont::CourierOblique,
            StandardFont::CourierBoldOblique,
            StandardFont::TimesRoman,
            StandardFont::TimesBold,
            StandardFont::TimesItalic,
            StandardFont::TimesBoldItalic,
            StandardFont::ZapfDingBats,
            StandardFont::Symbol,
        ] {
            let p = program_for(face).expect("face parses as CFF");
            assert!(!p.is_cid_keyed(), "a standard face must not be CID-keyed");
            // Every text face names 'A'; the two symbol fonts do not, so they
            // are checked by their own glyph names instead.
            let probe = match key(face) {
                12 => "a9",   // ZapfDingbats
                13 => "alpha", // Symbol
                _ => "A",
            };
            let gid = p
                .gid_for_name(probe)
                .unwrap_or_else(|| panic!("{probe:?} missing from face {}", key(face)));
            assert!(
                p.outline(gid).is_some(),
                "no outline for {probe:?} in face {}",
                key(face)
            );
        }
    }

    /// The bug that made typed form text invisible: a PDF naming a standard-14
    /// font may omit `/Widths` entirely, leaving the advance at zero — and the
    /// draw device skips any glyph whose advance is `<= 0`, so the text does
    /// not merely mis-space, it vanishes. The substituted face must therefore
    /// declare usable widths of its own.
    #[test]
    fn substituted_faces_declare_nonzero_advances() {
        for face in [
            StandardFont::Helvetica,
            StandardFont::HelveticaBold,
            StandardFont::TimesRoman,
            StandardFont::Courier,
        ] {
            let p = program_for(face).expect("face parses");
            for name in ["A", "space", "period", "zero"] {
                let gid = p.gid_for_name(name).unwrap_or_else(|| panic!("{name} missing"));
                let w = p.advance_width(gid).unwrap_or_else(|| panic!("no width for {name}"));
                assert!(w > 0.0, "{name} advance must be > 0, got {w}");
                assert!(w < 2000.0, "{name} advance {w} is implausible for 1/1000 em");
            }
        }
    }

    /// Courier is genuinely monospaced at 600/1000 em, so it is the one face
    /// whose widths can be checked against a known constant rather than a
    /// range — a cheap guard that the numbers are real metrics and not a
    /// default being echoed back.
    #[test]
    fn courier_widths_are_the_known_monospace_value() {
        let p = program_for(StandardFont::Courier).unwrap();
        for name in ["A", "i", "W", "period"] {
            let gid = p.gid_for_name(name).unwrap();
            let w = p.advance_width(gid).unwrap();
            assert!(
                (w - 600.0).abs() < 1.0,
                "Courier {name} should advance 600/1000 em, got {w}"
            );
        }
    }

    /// The cache must hand back the same allocation, not re-parse.
    #[test]
    fn programs_are_cached() {
        let a = program_for(StandardFont::Helvetica).unwrap();
        let b = program_for(StandardFont::Helvetica).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }
}
