//! `kpdf-doctor` — say **why** a PDF renders badly in kpdf, without needing
//! to send anyone the file.
//!
//! Two of kpdf's recurring dogfooding complaints are invisible from the
//! outside, and both have been blocked on "we need a reproduction":
//!
//! * **Solid blue/black boxes instead of text** (bd-515 / gh-67). The boxes
//!   are the draw device's advance-box fallback: the glyph had no decodable
//!   outline. Which font, and why, is not something a screenshot can answer.
//! * **A checkbox that will not tick** (this session's report). A form widget
//!   only toggles if it is classified as a checkbox/radio, is not read-only,
//!   and has an `/AP` `/N` dict naming a non-`Off` on-state.
//!
//! This prints exactly those facts, per page, so a document that cannot leave
//! the maintainer's machine (a dissertation, an internal report) can still be
//! diagnosed by pasting the output.
//!
//! ```text
//! cargo run --release -p kopitiam-pdf --bin kpdf-doctor -- thesis.pdf
//! cargo run --release -p kopitiam-pdf --bin kpdf-doctor -- thesis.pdf --pages 1-5
//! ```
//!
//! It is read-only: it opens the document, inspects it, and writes nothing.

use kopitiam_pdf::mupdf::font::{Font, OutlineSource};
use kopitiam_pdf::mupdf::form::{self, FieldKind};
use kopitiam_pdf::mupdf::object::Object;
use kopitiam_pdf::mupdf::xref::PdfDocument;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: kpdf-doctor <file.pdf> [--pages A-B]");
        std::process::exit(2);
    };
    let mut range: Option<(usize, usize)> = None;
    while let Some(a) = args.next() {
        if a == "--pages" {
            range = args.next().and_then(|s| parse_range(&s));
        }
    }

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            std::process::exit(1);
        }
    };
    let doc = match PdfDocument::open(bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open {path}: {e}");
            std::process::exit(1);
        }
    };

    let n = doc.page_count();
    let (first, last) = range.unwrap_or((0, n.saturating_sub(1)));
    let last = last.min(n.saturating_sub(1));
    println!("{path}: {n} pages, inspecting {}..={}", first + 1, last + 1);

    let mut boxed_fonts = 0usize;
    let mut togglable = 0usize;
    let mut stuck = 0usize;

    for page_index in first..=last {
        let Ok(page) = doc.page(page_index) else {
            continue;
        };
        let page = page.clone();
        let fonts = report_fonts(&doc, &page, page_index, &mut boxed_fonts);
        let fields = report_fields(&doc, page_index, &mut togglable, &mut stuck);
        if fonts || fields {
            println!();
        }
    }

    println!("== summary ==");
    println!(
        "  fonts with NO decodable outline (these draw as boxes): {boxed_fonts}",
    );
    println!("  form fields kpdf can toggle: {togglable}");
    println!("  form fields kpdf will NOT toggle: {stuck}");
    if boxed_fonts > 0 {
        println!(
            "\n  Boxes are the advance-box fallback. Paste the FONT lines above\n  \
             into bd-515 / gh-67 -- Subtype + FontFile key + the load error is\n  \
             exactly what that bug has been missing."
        );
    }
    if stuck > 0 {
        println!(
            "\n  A field kpdf will not toggle is read-only, is not classified\n  \
             checkbox/radio, or has no non-Off on-state in its /AP /N dict.\n  \
             The FIELD lines above say which."
        );
    }
}

fn parse_range(s: &str) -> Option<(usize, usize)> {
    let (a, b) = s.split_once('-')?;
    let a: usize = a.trim().parse().ok()?;
    let b: usize = b.trim().parse().ok()?;
    (a >= 1 && b >= a).then(|| (a - 1, b - 1))
}

/// Per-font facts for one page. Returns whether anything was printed.
fn report_fonts(doc: &PdfDocument, page: &Object, page_index: usize, boxed: &mut usize) -> bool {
    let Ok(res) = doc.resolve_get(page, "Resources") else {
        return false;
    };
    let Ok(fonts) = doc.resolve_get(&res, "Font") else {
        return false;
    };
    if !fonts.is_dict() {
        return false;
    }

    let mut printed = false;
    for i in 0..fonts.dict_len() {
        let (Some(key), Some(val)) = (fonts.dict_get_key(i), fonts.dict_get_val(i)) else {
            continue;
        };
        let Ok(dict) = doc.resolve(val) else { continue };
        let name = String::from_utf8_lossy(key).into_owned();
        let base = dict
            .dict_gets("BaseFont")
            .map(|o| String::from_utf8_lossy(o.to_name()).into_owned())
            .unwrap_or_else(|| "-".into());
        let subtype = dict
            .dict_gets("Subtype")
            .map(|o| String::from_utf8_lossy(o.to_name()).into_owned())
            .unwrap_or_else(|| "-".into());

        // For a Type0 font the interesting descriptor is the descendant's.
        let (desc_subtype, descriptor) = descendant_info(doc, &dict);
        let embedded = embedded_kind(doc, &descriptor);

        let (source, err) = match Font::load(doc, &dict) {
            Ok(f) => (Some(f.outline_source()), None),
            Err(e) => (None, Some(e.to_string())),
        };

        let verdict = match (source, &err) {
            (Some(OutlineSource::Program), _) => "ok (embedded program)".to_string(),
            (Some(OutlineSource::Type1), _) => "ok (Type1 program)".to_string(),
            (Some(OutlineSource::None), _) => {
                *boxed += 1;
                "NO OUTLINES -> draws as boxes".to_string()
            }
            (None, Some(e)) => {
                *boxed += 1;
                format!("LOAD FAILED -> draws as boxes: {e}")
            }
            (None, None) => "?".to_string(),
        };

        println!(
            "p{:<4} FONT {name:<8} {base:<34} {subtype}{}{} embed={embedded:<12} {verdict}",
            page_index + 1,
            if desc_subtype.is_empty() { "" } else { "/" },
            desc_subtype,
        );
        printed = true;
    }
    printed
}

/// `(descendant subtype, font descriptor dict)` — for a simple font the
/// descriptor is its own; for a Type0 it lives on the descendant CIDFont,
/// which is where CIDFontType0C/CIDFontType2 actually shows up.
fn descendant_info(doc: &PdfDocument, dict: &Object) -> (String, Object) {
    if let Ok(desc_fonts) = doc.resolve_get(dict, "DescendantFonts")
        && let Some(first) = desc_fonts.array_get(0)
        && let Ok(d) = doc.resolve(first)
    {
        let sub = d
            .dict_gets("Subtype")
            .map(|o| String::from_utf8_lossy(o.to_name()).into_owned())
            .unwrap_or_default();
        let descriptor = doc.resolve_get(&d, "FontDescriptor").unwrap_or(Object::Null);
        return (sub, descriptor);
    }
    (
        String::new(),
        doc.resolve_get(dict, "FontDescriptor").unwrap_or(Object::Null),
    )
}

/// Which `/FontFile*` the descriptor embeds, with `/FontFile3`'s own
/// `/Subtype` — that is where `CIDFontType0C` vs `Type1C` vs `OpenType` is
/// distinguished, and it is the top hypothesis in bd-515.
fn embedded_kind(doc: &PdfDocument, descriptor: &Object) -> String {
    if !descriptor.is_dict() {
        return "none".into();
    }
    for key in ["FontFile", "FontFile2", "FontFile3"] {
        let Some(ff) = descriptor.dict_gets(key) else {
            continue;
        };
        let sub = doc
            .resolve(ff)
            .ok()
            .and_then(|s| {
                s.dict_gets("Subtype")
                    .map(|o| String::from_utf8_lossy(o.to_name()).into_owned())
            })
            .unwrap_or_default();
        return if sub.is_empty() {
            key.to_string()
        } else {
            format!("{key}:{sub}")
        };
    }
    "none".into()
}

/// Per-form-field facts for one page. Returns whether anything was printed.
fn report_fields(doc: &PdfDocument, page_index: usize, ok: &mut usize, stuck: &mut usize) -> bool {
    let fields = form::page_form_fields(doc, page_index);
    let mut printed = false;
    for f in &fields {
        let togglable = matches!(f.kind, FieldKind::Checkbox | FieldKind::Radio)
            && !f.read_only
            && f.on_state.is_some();
        let why = if f.read_only {
            "NO: read-only".to_string()
        } else if !matches!(f.kind, FieldKind::Checkbox | FieldKind::Radio) {
            format!("n/a: {:?} (kpdf toggles Checkbox/Radio only)", f.kind)
        } else if f.on_state.is_none() {
            "NO: no non-Off on-state in /AP /N -- toggling would guess 'Yes'".to_string()
        } else {
            "yes".to_string()
        };
        if matches!(f.kind, FieldKind::Checkbox | FieldKind::Radio) {
            if togglable {
                *ok += 1;
            } else {
                *stuck += 1;
            }
        }
        println!(
            "p{:<4} FIELD {:<28} {:?}{} on_state={:?} value={:?} togglable={why}",
            page_index + 1,
            f.name,
            f.kind,
            if f.read_only { " [ro]" } else { "" },
            f.on_state,
            f.value,
        );
        printed = true;
    }
    printed
}
