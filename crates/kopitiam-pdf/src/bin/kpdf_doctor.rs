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
//! # `--render`: the checks that catch *invisible* content
//!
//! The structural checks above answer "is the document shaped correctly".
//! They found none of the four bugs in the 2026-08-28 dogfooding round,
//! because in every one of them the structure was already correct and only
//! the *drawing* was lost:
//!
//! * chart images rendered as solid black (`/SMask` never composited);
//! * typed form text stored fine and drew nothing (no `/Widths`, so every
//!   advance was 0, and a glyph with advance 0 is skipped);
//! * checkbox ticks drew nothing (ZapfDingbats `4` resolved through
//!   StandardEncoding to `four`, a glyph the face has not got);
//! * a 506-page document froze the window for 38 s before its first frame.
//!
//! Each was invisible to a structural check and obvious to a pixel count. So
//! `--render` rasterizes, and asks the questions a person would:
//!
//! * **Does this page draw anything at all?** A blank page in a document that
//!   is not blank.
//! * **Is it suspiciously, solidly black?** The signature of an image drawn
//!   without its soft mask.
//! * **Does ticking a checkbox change any pixels?** If not, the tick is
//!   invisible however correct the `/V` and `/AS` values are.
//! * **Does typing into a text field change any pixels?** Same question for
//!   the widget appearance kpdf regenerates.
//! * **How long would the whole document take?** Per-page render cost,
//!   projected, so a document that will hang the reader says so.
//!
//! Field probes are done **entirely in memory** — `toggle_checkbox` and
//! `set_field_value` return new bytes and never touch the file on disk, so
//! this stays read-only even while testing edits.
//!
//! ```text
//! kpdf-doctor thesis.pdf --render
//! kpdf-doctor form.pdf --render --pages 7-9
//! ```
//!
//! It is read-only: it opens the document, inspects it, and writes nothing.

use std::time::{Duration, Instant};

use kopitiam_pdf::mupdf::draw_device::rasterize_page_ex;
use kopitiam_pdf::mupdf::font::{Font, OutlineSource};
use kopitiam_pdf::mupdf::form::{self, FieldKind, FormField};
use kopitiam_pdf::mupdf::object::Object;
use kopitiam_pdf::mupdf::pixmap::Pixmap;
use kopitiam_pdf::mupdf::xref::PdfDocument;

/// Rasterization dpi for the render checks. Low enough to keep a long
/// document tolerable, high enough that a checkbox tick (a few points across)
/// still covers several pixels — at 72 dpi a tick can land on so few pixels
/// that "did anything change" gets genuinely ambiguous.
const CHECK_DPI: f32 = 100.0;

/// How many fields of each kind to probe per page. Every probe costs a full
/// re-render, so a 30-field page would otherwise take half a minute on its
/// own. Three is enough to catch a systemic "no tick ever appears" fault,
/// which is what this looks for — it is not a per-field audit.
const FIELD_PROBES_PER_PAGE: usize = 3;

/// A page darker than this is reported as suspect. Real pages of text run
/// ~1-2% ink; the unmasked-image bug put pages 8 and 9 of the workbook at
/// 16.7% and 8.6% pure black, and a scanned or deliberately dark page is rare
/// enough that a false positive costs only a line of output.
const BLACK_PAGE_FRACTION: f64 = 0.05;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: kpdf-doctor <file.pdf> [--pages A-B] [--render]");
        std::process::exit(2);
    };
    let mut range: Option<(usize, usize)> = None;
    let mut render = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pages" => range = args.next().and_then(|s| parse_range(&s)),
            "--render" => render = true,
            _ => {}
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
    let mut substituted = 0usize;
    let mut togglable = 0usize;
    let mut stuck = 0usize;
    let mut render_findings = 0usize;
    let mut render_time = Duration::ZERO;
    let mut rendered_pages = 0usize;

    for page_index in first..=last {
        let Ok(page) = doc.page(page_index) else {
            continue;
        };
        let page = page.clone();
        let fonts = report_fonts(&doc, &page, page_index, &mut boxed_fonts, &mut substituted);
        let fields = report_fields(&doc, page_index, &mut togglable, &mut stuck);
        let mut drew = false;
        if render {
            let r = render_checks(&doc, page_index);
            render_time += r.elapsed;
            rendered_pages += 1;
            render_findings += r.findings;
            drew = r.findings > 0;
        }
        if fonts || fields || drew {
            println!();
        }
    }

    println!("== summary ==");
    println!(
        "  fonts with NO decodable outline (these draw as boxes): {boxed_fonts}",
    );
    println!(
        "  fonts NOT embedded, rendered via a substituted standard-14 face: {substituted}",
    );
    println!("  form fields kpdf can toggle: {togglable}");
    println!("  form fields kpdf will NOT toggle: {stuck}");
    if render {
        println!("  render findings: {render_findings}");
        if rendered_pages > 0 {
            let per = render_time / rendered_pages as u32;
            println!(
                "  render cost: {:?}/page over {rendered_pages} page(s) -> whole document ({n} pages) ~{:.1}s",
                per,
                per.as_secs_f64() * n as f64
            );
            // kpdf renders on the UI thread, so a document whose pages cost
            // this much in total is one a reader will feel as a freeze.
            if per.as_secs_f64() * n as f64 > 10.0 {
                println!(
                    "\n  This document is slow enough to feel like a hang if anything\n  \
                     ever renders all its pages up front. Page sizes come from\n  \
                     /MediaBox rather than a render for exactly this reason -- keep\n  \
                     any new all-pages pass off the UI thread."
                );
            }
        }
    }
    if boxed_fonts > 0 {
        println!(
            "\n  Boxes are the advance-box fallback. Paste the FONT lines above\n  \
             into bd-515 / gh-67 -- Subtype + FontFile key + the load error is\n  \
             exactly what that bug has been missing."
        );
    }
    if render && render_findings > 0 {
        println!(
            "\n  A RENDER line means the document is structurally fine and the\n  \
             DRAWING is what went missing -- the class of bug the checks above\n  \
             cannot see. Each line says which question failed."
        );
    }
    if !render {
        println!(
            "\n  Structural checks only. Re-run with --render to catch content that\n  \
             is present but never drawn (unmasked images, invisible ticks and\n  \
             field text), and to measure what the document costs to render."
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


// ---------------------------------------------------------------------------
// Render checks (--render)
// ---------------------------------------------------------------------------

/// Ink statistics for one rasterized page.
struct Ink {
    /// Pixels darker than mid-grey — "something was drawn here".
    dark: usize,
    /// Pixels that are essentially pure black **and achromatic** — see
    /// [`Ink::of`]'s chroma check for why both conditions matter.
    black: usize,
    total: usize,
}

/// How far apart an RGB pixel's channels may be and still count as
/// "neutral" (grey/black) rather than a saturated colour.
///
/// # Why luma alone was wrong
///
/// A real government form the maintainer opened has navy-blue (`RGB(0, 42,
/// 122)`) section banners. Luma alone: `0.299*0 + 0.587*42 + 0.114*122 =
/// 38.5` — under the old `l < 40` cutoff, so 106,586 pixels of a perfectly
/// normal, deliberately-coloured banner were counted as "black", tripping
/// [`BLACK_PAGE_FRACTION`] and printing "suspect an image drawn without its
/// /SMask" on a page that has **no image XObject on it at all**. A vivid
/// colour can have low luminance; that is not what a missing `/SMask` looks
/// like. A composited-without-its-mask image is neutral -- the base image's
/// stored "hide me" background is black or a dark grey, not a saturated hue,
/// because a producer has no reason to paint the throwaway background in a
/// colour.
///
/// So a pixel must be dark *and* have R, G and B within this many levels of
/// each other. 24 is generous enough to admit ordinary anti-aliasing and
/// JPEG ringing around a genuinely black/grey region while still rejecting
/// navy blue's channel spread of 122.
const BLACK_CHROMA_TOLERANCE: u8 = 24;

impl Ink {
    fn of(pix: &Pixmap) -> Ink {
        let (w, h) = (pix.width(), pix.height());
        let mut dark = 0;
        let mut black = 0;
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                if let Some(l) = pix.luma(x, y) {
                    if l < 128 {
                        dark += 1;
                    }
                    if l < 40 && pix.pixel(x, y).is_some_and(is_neutral) {
                        black += 1;
                    }
                }
            }
        }
        Ink {
            dark,
            black,
            total: (w as usize) * (h as usize),
        }
    }

    fn black_fraction(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.black as f64 / self.total as f64
        }
    }
}

/// Whether an RGB(+) pixel's channels are close enough to call it grey/black
/// rather than a saturated colour -- see [`BLACK_CHROMA_TOLERANCE`].
///
/// A 1-component (grayscale) pixmap is trivially neutral; only the first
/// three bytes are read, since that is all [`Pixmap::luma`] itself looks at.
fn is_neutral(sample: &[u8]) -> bool {
    if sample.len() < 3 {
        return true;
    }
    let (r, g, b) = (sample[0], sample[1], sample[2]);
    let hi = r.max(g).max(b);
    let lo = r.min(g).min(b);
    hi - lo <= BLACK_CHROMA_TOLERANCE
}

/// Count dark pixels on one page of `doc`, or `None` if it will not render.
fn ink_of_page(doc: &PdfDocument, page_index: usize) -> Option<usize> {
    rasterize_page_ex(doc, page_index, CHECK_DPI)
        .ok()
        .map(|(pix, _)| Ink::of(&pix).dark)
}

/// What a render check concluded for one page.
struct RenderReport {
    elapsed: Duration,
    findings: usize,
}

/// Rasterize `page_index` and ask the questions a person would.
///
/// Returns how long the render took (for the whole-document projection) and
/// how many findings were reported.
fn render_checks(doc: &PdfDocument, page_index: usize) -> RenderReport {
    let started = Instant::now();
    let rendered = rasterize_page_ex(doc, page_index, CHECK_DPI);
    let elapsed = started.elapsed();
    let p = page_index + 1;

    let (pix, fallback_glyphs) = match rendered {
        Ok(v) => v,
        Err(e) => {
            println!("p{p:<4} RENDER  FAILED: {e}");
            return RenderReport {
                elapsed,
                findings: 1,
            };
        }
    };
    let ink = Ink::of(&pix);
    let mut findings = 0;

    // A page that draws nothing. Legitimately blank pages exist, so this is
    // reported rather than treated as an error -- but a whole document of
    // them is the signature of a rendering fault, not of blank paper.
    if ink.dark == 0 {
        println!("p{p:<4} RENDER  page is completely blank (0 dark pixels)");
        findings += 1;
    }

    // Solid black: an image drawn without its /SMask, which stores the
    // background as opaque black and relies on the mask to hide it.
    if ink.black_fraction() > BLACK_PAGE_FRACTION {
        println!(
            "p{p:<4} RENDER  {:.1}% of the page is solid black -- suspect an image \
             drawn without its /SMask",
            ink.black_fraction() * 100.0
        );
        findings += 1;
    }

    // Advance boxes: glyphs with no decodable outline. The structural font
    // check predicts these; this confirms they actually reached the page.
    if fallback_glyphs > 0 {
        println!(
            "p{p:<4} RENDER  {fallback_glyphs} glyphs drew as advance boxes (no outline)"
        );
        findings += 1;
    }

    findings += probe_fields(doc, page_index, ink.dark);

    RenderReport { elapsed, findings }
}

/// Spread a sample of at most `n` items evenly across `items`.
///
/// Taking the first `n` is what a first cut of this did, and it silently
/// missed the very bug it was written for: page 7 of the workbook lists its
/// radios before its checkboxes, and only the checkboxes were broken. Radios
/// draw a vector circle and checkboxes draw a ZapfDingbats glyph — different
/// mechanisms, different failure modes — so a sample that never reaches the
/// second kind proves nothing. Hence spreading, *and* sampling each kind
/// separately below.
fn spread<T>(items: Vec<T>, n: usize) -> Vec<T> {
    if items.len() <= n || n == 0 {
        return items;
    }
    let step = items.len() as f64 / n as f64;
    let mut out = Vec::with_capacity(n);
    for (i, item) in items.into_iter().enumerate() {
        if out.len() < n && (i as f64) >= out.len() as f64 * step {
            out.push(item);
        }
    }
    out
}

/// Probe a sample of this page's fields: change one, re-render, and check the
/// page actually looks different.
///
/// This is the check that catches an *invisible* edit — a value stored
/// correctly whose appearance draws nothing. Every probe is in memory; the
/// file on disk is never written.
fn probe_fields(doc: &PdfDocument, page_index: usize, baseline_ink: usize) -> usize {
    let fields = form::page_form_fields(doc, page_index);
    let p = page_index + 1;
    let mut findings = 0;

    // Checkbox and Radio are sampled SEPARATELY, not as one "togglable" pool:
    // a radio's on-appearance is usually a drawn circle while a checkbox's is
    // a ZapfDingbats glyph, so one working says nothing about the other.
    let mut toggles: Vec<&FormField> = Vec::new();
    for kind in [FieldKind::Checkbox, FieldKind::Radio] {
        let of_kind: Vec<&FormField> = fields
            .iter()
            .filter(|f| {
                std::mem::discriminant(&f.kind) == std::mem::discriminant(&kind)
                    && !f.read_only
                    && f.on_state.is_some()
                    // Excluded from SAMPLING, not just from the verdict: a
                    // `/F` Hidden/NoView widget is invisible on screen by the
                    // document's own design (kpdf's renderer already skips it
                    // unconditionally, matching MuPDF -- see
                    // `FormField::hidden`'s doc comment). Spending one of
                    // FIELD_PROBES_PER_PAGE on a field that is GUARANTEED to
                    // show zero pixel change wastes the sample budget on a
                    // question with a known answer, on top of the false
                    // "invisible" finding this used to print.
                    && !f.hidden
            })
            .collect();
        toggles.extend(spread(of_kind, FIELD_PROBES_PER_PAGE));
    }
    for f in toggles {
        let Ok(bytes) = form::toggle_checkbox(doc, f) else {
            continue;
        };
        let Ok(edited) = PdfDocument::open(bytes) else {
            continue;
        };
        if ink_of_page(&edited, page_index) == Some(baseline_ink) {
            println!(
                "p{p:<4} RENDER  ticking {:?} changes NO pixels -- the tick is invisible",
                f.name
            );
            findings += 1;
        }
    }

    let texts: Vec<&FormField> = spread(
        fields
            .iter()
            .filter(|f| matches!(f.kind, FieldKind::Text) && !f.read_only && !f.hidden)
            .collect(),
        FIELD_PROBES_PER_PAGE,
    );
    for f in texts {
        // A probe string of wide, unambiguous glyphs, so "no pixels changed"
        // cannot be blamed on the text being thin or the box being tiny.
        let Ok(bytes) = form::set_field_value(doc, f, "WWW") else {
            continue;
        };
        let Ok(edited) = PdfDocument::open(bytes) else {
            continue;
        };
        if ink_of_page(&edited, page_index) == Some(baseline_ink) {
            println!(
                "p{p:<4} RENDER  typing into {:?} changes NO pixels -- the text is invisible",
                f.name
            );
            findings += 1;
        }
    }

    findings
}

fn parse_range(s: &str) -> Option<(usize, usize)> {
    let (a, b) = s.split_once('-')?;
    let a: usize = a.trim().parse().ok()?;
    let b: usize = b.trim().parse().ok()?;
    (a >= 1 && b >= a).then(|| (a - 1, b - 1))
}

/// Per-font facts for one page. Returns whether anything was printed.
fn report_fonts(
    doc: &PdfDocument,
    page: &Object,
    page_index: usize,
    boxed: &mut usize,
    substituted: &mut usize,
) -> bool {
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
            (Some(OutlineSource::Substitute), _) => {
                *substituted += 1;
                "ok (SUBSTITUTED standard-14 face -- font not embedded)".to_string()
            }
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

#[cfg(test)]
mod ink_tests {
    use super::*;

    /// THE REGRESSION. A real government form (the maintainer's) has navy-blue
    /// (`RGB(0, 42, 122)`) section banners. Luma alone: `0.299*0 + 0.587*42 +
    /// 0.114*122 = 38.5`, which the old `l < 40` cutoff counted as "black" --
    /// 106,586 pixels of it, on a page with no image XObject at all, so
    /// `render_checks` printed "suspect an image drawn without its /SMask" on
    /// a page that could not possibly have that bug.
    ///
    /// A vivid, saturated colour is not what a missing `/SMask` looks like --
    /// that failure mode is a NEUTRAL black/grey background a producer had no
    /// reason to paint in a hue. So a pixel this dark but this far from grey
    /// must not count.
    #[test]
    fn a_saturated_navy_banner_is_not_counted_as_black() {
        assert!(
            !is_neutral(&[0, 42, 122]),
            "R=0 G=42 B=122 spans 122 levels -- a vivid colour, not a shade of \
             black or grey"
        );
    }

    /// The check this exists to keep working: a genuinely neutral near-black
    /// pixel -- exactly what an image's opaque "hide me" background looks
    /// like when its `/SMask` is never composited -- must still count.
    #[test]
    fn a_neutral_near_black_pixel_still_counts() {
        for rgb in [[0u8, 0, 0], [3, 3, 3], [12, 10, 14], [37, 37, 37]] {
            assert!(
                is_neutral(&rgb),
                "{rgb:?} is a shade of black/grey and must still be flagged"
            );
        }
    }

    /// The boundary is a real design choice (generous enough for
    /// anti-aliasing and JPEG ringing, tight enough to reject a real hue), so
    /// it is pinned at both edges rather than left to whatever the constant
    /// happens to be.
    #[test]
    fn the_chroma_tolerance_boundary_is_where_it_is_meant_to_be() {
        let base = 10u8;
        assert!(is_neutral(&[base, base, base + BLACK_CHROMA_TOLERANCE]));
        assert!(!is_neutral(&[base, base, base + BLACK_CHROMA_TOLERANCE + 1]));
    }

    /// A grayscale (1-component) pixmap has no colour to be saturated, so a
    /// dark pixel there must never be excluded on chroma grounds -- `is_neutral`
    /// must treat anything shorter than 3 bytes as trivially neutral.
    #[test]
    fn a_grayscale_sample_is_always_neutral() {
        assert!(is_neutral(&[5]));
        assert!(is_neutral(&[]));
    }

    /// End-to-end version of the regression: build a page whose only content
    /// is a full-bleed navy rectangle (no image, no /SMask anywhere), and
    /// confirm `Ink::of` does NOT count it as "black" ink -- the false
    /// positive was exactly this shape, and the fix must hold at the level
    /// `render_checks` actually calls, not only in the unit-tested helper.
    #[test]
    fn a_page_filled_with_a_saturated_colour_reports_no_black_ink() {
        let mut pix = Pixmap::new_rgb(100, 100);
        let (chunks, _) = pix.samples.as_chunks_mut::<3>();
        for chunk in chunks {
            *chunk = [0, 42, 122]; // the maintainer's navy
        }
        let ink = Ink::of(&pix);
        assert_eq!(ink.black, 0, "a saturated fill must not read as black ink");
        assert_eq!(
            ink.dark,
            ink.total,
            "it is still legitimately DARK -- luma 38.5 is well under 128 -- \
             just not black. Only the black/SMask heuristic changes."
        );
    }

    /// And the case this file was fixing FOR must still work: a page that
    /// really is mostly opaque neutral black (the un-composited-mask shape)
    /// must still cross the threshold.
    #[test]
    fn a_page_filled_with_true_black_still_reports_black_ink() {
        let pix = Pixmap::new_rgb(100, 100); // zeroed = (0,0,0) everywhere
        let ink = Ink::of(&pix);
        assert_eq!(ink.black, ink.total);
        assert!(ink.black_fraction() > BLACK_PAGE_FRACTION);
    }
}
