//! Regression tests for the "scanned paper opens blank in kovan" bug
//! (2026-09-28): an Acrobat Capture 3.0 "searchable image" PDF -- CCITT strip
//! images for the visible page, plus an invisible (`3 Tr`) OCR text layer in
//! **non-embedded** TrueType fonts (`TimesNewRoman`, `Arial,Bold`,
//! `BookAntiqua,BoldItalic`, ...).
//!
//! Two separate defects were stacked on top of each other, and each test
//! below pins one of them:
//!
//! 1. **`/Contents N 0 R` naming an ARRAY.** Capture writes the page's
//!    content as an indirect array of dozens of tiny streams. The port only
//!    concatenated a *direct* array; an indirect one went to `open_stream`,
//!    failed, and the page ran with zero bytes -- blank, no error. MuPDF's
//!    `pdf_open_contents_stream` resolves before it tests `pdf_is_array`.
//! 2. **Invisible text got painted.** MuPDF's `pdf_flush_text_imp` fills
//!    nothing for render mode 3 (`doinvisible`) or 7 (`doclip`). The draw
//!    device used to paint every glyph whatever the mode, so the OCR layer was
//!    drawn over the scan -- as substituted outlines, or as solid advance
//!    boxes for faces with no standard-14 substitute (which then also tripped
//!    the hayro fallback for the whole page).
//!
//! Fixtures are hand-built synthetic PDFs so the bytes under test stay
//! readable. The real paper that exposed this is restricted literature and is
//! deliberately NOT a fixture.

use kopitiam_pdf::mupdf::font::Font;
use kopitiam_pdf::mupdf::geometry::Matrix;
use kopitiam_pdf::mupdf::text_device::TextDevice;
use kopitiam_pdf::mupdf::xref::PdfDocument;
use kopitiam_pdf::mupdf::{rasterize_page_ex, run_page};

/// Assemble objects `1..` from `bodies` into a PDF with a classic xref table.
/// Offsets are computed, never hand-written.
fn build_pdf(bodies: &[Vec<u8>]) -> Vec<u8> {
    let mut pdf: Vec<u8> = b"%PDF-1.4\n".to_vec();
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

fn stream(ops: &str) -> Vec<u8> {
    format!("<< /Length {} >>\nstream\n{ops}\nendstream", ops.len()).into_bytes()
}

/// A 100x100 pt page whose `/Contents 4 0 R` is an **indirect array** of the
/// given content streams (objects 6..), with one font `/F1` = object 5 --
/// exactly the Capture layout. `font` is the font dict body.
fn capture_style_doc(font: &str, parts: &[&str]) -> PdfDocument {
    let first_part = 6;
    let refs: Vec<String> = (0..parts.len())
        .map(|i| format!("{} 0 R", first_part + i))
        .collect();
    let mut bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        format!("[ {} ]", refs.join(" ")).into_bytes(),
        font.as_bytes().to_vec(),
    ];
    bodies.extend(parts.iter().map(|p| stream(p)));
    PdfDocument::open(build_pdf(&bodies)).expect("fixture opens")
}

/// What Capture writes for its OCR fonts: non-embedded TrueType, WinAnsi, no
/// `/Widths`, no descriptor.
const TIMES_NEW_ROMAN: &str =
    "<< /Type /Font /Subtype /TrueType /BaseFont /TimesNewRoman /Encoding /WinAnsiEncoding >>";
const ARIAL_BOLD: &str =
    "<< /Type /Font /Subtype /TrueType /BaseFont /Arial,Bold /Encoding /WinAnsiEncoding >>";
/// Build a WinAnsi TrueType font dict over codes 32..=126, every advance
/// `w`/1000 em, with an optional inline `/FontDescriptor` body.
fn widths_font(base: &str, w: u32, descriptor: Option<&str>) -> String {
    let widths: Vec<String> = (32..=126).map(|_| w.to_string()).collect();
    let desc = descriptor
        .map(|d| format!(" /FontDescriptor {d}"))
        .unwrap_or_default();
    format!(
        "<< /Type /Font /Subtype /TrueType /BaseFont /{base} /Encoding /WinAnsiEncoding \
/FirstChar 32 /LastChar 126 /Widths [ {} ]{desc} >>",
        widths.join(" ")
    )
}

/// No family keyword in the name and no descriptor: nothing to substitute,
/// so a *visible* glyph falls back to an advance box.
fn book_antiqua_no_descriptor() -> String {
    widths_font("BookAntiqua,BoldItalic", 500, None)
}

/// The same name with the descriptor Capture really writes for it:
/// `/Flags 16482` = Serif | Nonsymbolic | Italic (+ a non-standard bit 15).
fn book_antiqua_capture_descriptor() -> String {
    widths_font(
        "BookAntiqua,BoldItalic",
        500,
        Some(
            "<< /Type /FontDescriptor /FontName /BookAntiqua,BoldItalic /Flags 16482 \
/FontBBox [ -250 -276 1200 931 ] /ItalicAngle -11 /Ascent 931 /Descent 276 \
/CapHeight 931 /StemV 143 >>",
        ),
    )
}

fn dark_pixels(doc: &PdfDocument) -> (usize, usize) {
    let (pix, fallback) = rasterize_page_ex(doc, 0, 72.0).expect("rasterizes");
    let n = pix.n as usize;
    let dark = pix.samples.chunks(n).filter(|p| p[0] < 128).count();
    (dark, fallback)
}

/// Defect 1: an indirect `/Contents` array must run, and its parts must be
/// concatenated -- the `q` in part one and the `Q` in part three are one
/// graphics-state pair, like Capture's `Q \r q \r ... Do` splitting.
#[test]
fn indirect_contents_array_is_run_and_concatenated() {
    let doc = capture_style_doc(TIMES_NEW_ROMAN, &["q 0 0 0 rg", "20 20 60 60 re f", "Q"]);
    let (dark, _) = dark_pixels(&doc);
    // 60x60 pt at 72 dpi = 3600 px. Before the fix: 0 (page ran empty).
    assert!(dark > 3000, "indirect /Contents array drew {dark} dark px, want ~3600");
}

/// Visible (`0 Tr`) non-embedded TimesNewRoman must paint real substituted
/// outlines, not boxes -- the control that proves the `3 Tr` test below can fail.
#[test]
fn visible_non_embedded_truetype_paints_substituted_outlines() {
    let doc = capture_style_doc(TIMES_NEW_ROMAN, &["BT /F1 24 Tf 5 40 Td 0 Tr (HHHH) Tj ET"]);
    let (dark, fallback) = dark_pixels(&doc);
    assert!(dark > 100, "visible TimesNewRoman drew only {dark} dark px");
    assert_eq!(fallback, 0, "TimesNewRoman must substitute Times, not box");
}

/// Defect 2: the same text in `3 Tr` paints NOTHING (MuPDF `doinvisible`).
#[test]
fn render_mode_3_paints_nothing() {
    let doc = capture_style_doc(TIMES_NEW_ROMAN, &["BT /F1 24 Tf 5 40 Td 3 Tr (HHHH) Tj ET"]);
    let (dark, fallback) = dark_pixels(&doc);
    assert_eq!(dark, 0, "invisible (3 Tr) text painted {dark} px");
    assert_eq!(fallback, 0);
}

/// Mode 7 (clip only) also neither fills nor strokes.
#[test]
fn render_mode_7_paints_nothing() {
    let doc = capture_style_doc(TIMES_NEW_ROMAN, &["BT /F1 24 Tf 5 40 Td 7 Tr (HHHH) Tj ET"]);
    assert_eq!(dark_pixels(&doc).0, 0);
}

/// Invisible text in a face with no substitute must not draw boxes, and must
/// not be counted as a fallback glyph -- that count is what sends a page to
/// the hayro re-render. With `0 Tr` the same text DOES box (control).
#[test]
fn invisible_unsubstitutable_font_draws_no_boxes_and_no_fallback() {
    let font = book_antiqua_no_descriptor();
    let invisible = capture_style_doc(&font, &["BT /F1 24 Tf 5 40 Td 3 Tr (Graphite) Tj ET"]);
    assert_eq!(dark_pixels(&invisible), (0, 0));

    let visible = capture_style_doc(&font, &["BT /F1 24 Tf 5 40 Td 0 Tr (Graphite) Tj ET"]);
    let (dark, fallback) = dark_pixels(&visible);
    assert!(fallback > 0 && dark > 0, "control: visible BookAntiqua should box ({dark}, {fallback})");
}

/// MuPDF's last-resort substitution (`pdf_lookup_substitute_font`): an
/// unknown name whose descriptor says Serif + Nonsymbolic draws with Times
/// outlines, not boxes -- compare the control in the test above.
#[test]
fn nonsymbolic_descriptor_flags_substitute_an_unknown_name() {
    let font = book_antiqua_capture_descriptor();
    let doc = capture_style_doc(&font, &["BT /F1 24 Tf 5 40 Td 0 Tr (Graphite) Tj ET"]);
    let (dark, fallback) = dark_pixels(&doc);
    assert_eq!(fallback, 0, "Serif|Nonsymbolic BookAntiqua should substitute Times");
    assert!(dark > 50, "substituted BookAntiqua drew only {dark} px");
}

/// Records `(unicode, adv_em)` per glyph -- the extraction side of the seam.
#[derive(Default)]
struct Recorder {
    glyphs: Vec<(char, f32)>,
}

impl TextDevice for Recorder {
    fn show_glyph(&mut self, _font: &Font, _trm: Matrix, adv: f32, unicode: char, _cid: u32, _wmode: u8) {
        self.glyphs.push((unicode, adv));
    }
}

/// The invisible OCR layer must still reach an extraction device -- it is the
/// only text a scanned paper has.
#[test]
fn invisible_text_is_still_extracted() {
    let doc = capture_style_doc(TIMES_NEW_ROMAN, &["BT /F1 12 Tf 5 40 Td 3 Tr (Graphite) Tj ET"]);
    let mut rec = Recorder::default();
    run_page(&doc, 0, &mut rec).unwrap();
    let text: String = rec.glyphs.iter().map(|g| g.0).collect();
    assert_eq!(text, "Graphite");
}

/// With no `/Widths`, advances come from the substituted face's own metrics
/// (Times-Roman / Helvetica-Bold AFM values, 1/1000 em), not zero -- so glyph
/// positions and extracted word boxes are right.
#[test]
fn missing_widths_use_substitute_face_advances() {
    for (font, expect) in [
        // Times-Roman AFM: A=722, i=278, space=250.
        (TIMES_NEW_ROMAN, [('A', 0.722), ('i', 0.278), (' ', 0.250)]),
        // Helvetica-Bold AFM: A=722, i=278, space=278.
        (ARIAL_BOLD, [('A', 0.722), ('i', 0.278), (' ', 0.278)]),
    ] {
        let doc = capture_style_doc(font, &["BT /F1 10 Tf 5 40 Td 3 Tr (Ai ) Tj ET"]);
        let mut rec = Recorder::default();
        run_page(&doc, 0, &mut rec).unwrap();
        assert_eq!(rec.glyphs.len(), 3, "{font}");
        for ((got_c, got_adv), (want_c, want_adv)) in rec.glyphs.iter().zip(expect) {
            assert_eq!(*got_c, want_c);
            assert!(
                (got_adv - want_adv).abs() < 0.002,
                "{font}: {want_c:?} advance {got_adv}, want {want_adv}"
            );
        }
    }
}
