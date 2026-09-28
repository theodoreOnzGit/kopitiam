//! MuPDF-parity regression tests -- one per divergence the code-to-code
//! harness (`examples/mupdf_oracle.rs`, methodology + measured results in
//! `docs/mupdf-code-to-code.md`) caught between kopitiam-pdf and real MuPDF
//! (`19f1284`) on the same input.
//!
//! Every test here was run against the tree BEFORE its fix and failed, so each
//! one can fail again if the divergence ever comes back. The fixtures are
//! hand-built synthetic PDFs (no third-party documents), small enough to read.
//!
//! Sections follow the 0.4.2 tranches:
//! * **Tranche 1 -- structured text** (`stext-device.c`, `pdf-op-run.c`).
//! * **Tranche 2 -- images** (`draw-device.c` / `draw-scale-simple.c` /
//!   `draw-affine.c` image path; JPX + JBIG2 codecs).

use kopitiam_pdf::mupdf::structured_text::{StextBlock, StextChar, StextOptions};
use kopitiam_pdf::mupdf::xref::PdfDocument;
use kopitiam_pdf::mupdf::{page_images, page_to_stext, rasterize_page_native};

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

fn stream(dict_extra: &str, data: &str) -> Vec<u8> {
    format!("<< /Length {}{dict_extra} >>\nstream\n{data}\nendstream", data.len()).into_bytes()
}

/// A ToUnicode CMap body mapping each `(code, utf16be-hex)` pair.
fn to_unicode(pairs: &[(&str, &str)]) -> String {
    let body: Vec<String> = pairs.iter().map(|(c, u)| format!("<{c}> <{u}>")).collect();
    format!(
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n\
1 begincodespacerange <00> <FF> endcodespacerange\n\
{} beginbfchar\n{}\nendbfchar\n\
endcmap CMapName currentdict /CMap defineresource pop end end",
        pairs.len(),
        body.join("\n")
    )
}

/// One 200x100 pt page, content `ops`, font `/F1` = Helvetica with every
/// advance 500/1000 em and (optionally) the given ToUnicode mappings.
fn one_page(ops: &str, tounicode: Option<&[(&str, &str)]>) -> PdfDocument {
    let widths = vec!["500"; 256].join(" ");
    let tu_ref = if tounicode.is_some() { " /ToUnicode 6 0 R" } else { "" };
    let mut bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream("", ops),
        format!(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /FirstChar 0 /LastChar 255 \
/Widths [ {widths} ]{tu_ref} >>"
        )
        .into_bytes(),
    ];
    if let Some(pairs) = tounicode {
        bodies.push(stream("", &to_unicode(pairs)));
    }
    PdfDocument::open(build_pdf(&bodies)).expect("fixture opens")
}

/// Every char of page 0, in stext order, extracted with MuPDF's API-default
/// options (flags 0).
fn chars(doc: &PdfDocument) -> Vec<StextChar> {
    let page = page_to_stext(doc, 0, StextOptions::default()).expect("stext");
    page.blocks
        .iter()
        .filter_map(|b| match b {
            StextBlock::Text(t) => Some(t),
            _ => None,
        })
        .flat_map(|t| t.lines.iter())
        .flat_map(|l| l.chars.iter().cloned())
        .collect()
}

fn text(cs: &[StextChar]) -> String {
    cs.iter().map(|c| c.c).collect()
}

// ---------------------------------------------------------------------------
// Tranche 1 -- structured text
// ---------------------------------------------------------------------------

/// MuPDF `pdf_show_char` (pdf-op-run.c:1449): a one-to-many ToUnicode entry
/// shows its 2nd+ code points as zero-advance filler glyphs, which
/// `fz_add_stext_char_imp` (stext-device.c:850) parks on the pen. A TeX "fi"
/// ligature (`<0C> -> "fi"`) must therefore extract as `f`,`i` with the `i`
/// at the END of the ligature glyph, not dropped (0.4.1 dropped it).
#[test]
fn one_to_many_tounicode_emits_filler_chars_on_the_pen() {
    let doc = one_page(
        "BT /F1 10 Tf 10 50 Td <0C41> Tj ET",
        Some(&[("0C", "00660069"), ("41", "0041")]),
    );
    let cs = chars(&doc);
    assert_eq!(text(&cs), "fiA");
    // f at the glyph origin, i on the pen = origin + 0.5 em * 10 pt.
    assert!((cs[0].origin.x - 10.0).abs() < 1e-3, "f at {}", cs[0].origin.x);
    assert!((cs[1].origin.x - 15.0).abs() < 1e-3, "i at {}", cs[1].origin.x);
    assert!((cs[2].origin.x - 15.0).abs() < 1e-3, "A at {}", cs[2].origin.x);
}

/// MuPDF's fake-bold overprint check (stext-device.c:870-874) drops a char
/// drawn again on top of itself -- but only when `glyph >= 0`. The port used
/// to pass `-1` for every glyph, which silently disabled it, so a producer's
/// "print twice for bold" came out as doubled letters.
#[test]
fn overprinted_same_glyph_is_dropped_as_fake_bold() {
    let doc = one_page(
        "BT /F1 10 Tf 10 50 Td (A) Tj ET BT /F1 10 Tf 10 50 Td (A) Tj ET",
        None,
    );
    assert_eq!(text(&chars(&doc)), "A");
}

/// A non-spacing mark (general category Mn) rides on the pen and does not
/// move it (stext-device.c:850-858), even when the content stream kerned its
/// glyph backwards over the base letter -- which is exactly how producers
/// position a combining accent.
#[test]
fn combining_mark_sits_on_the_pen_not_its_kerned_origin() {
    // `A`, then kern back 500/1000 em, then the accent glyph <42>.
    let doc = one_page(
        "BT /F1 10 Tf 10 50 Td [<41> 500 <42>] TJ ET",
        Some(&[("41", "0041"), ("42", "0301")]),
    );
    let cs = chars(&doc);
    assert_eq!(text(&cs), "A\u{301}");
    assert!((cs[1].origin.x - 15.0).abs() < 1e-3, "mark at {}", cs[1].origin.x);
}

/// "Alphabetic and arabic presentation forms" decompose
/// (stext-device.c:1097-1104, `ucdn_compat_decompose`) unless
/// PRESERVE_LIGATURES. U+FB13 ARMENIAN SMALL LIGATURE MEN NOW -> U+0574 U+0576.
#[test]
fn presentation_form_ligature_decomposes() {
    let doc = one_page("BT /F1 10 Tf 10 50 Td <43> Tj ET", Some(&[("43", "FB13")]));
    assert_eq!(text(&chars(&doc)), "\u{574}\u{576}");
}

// ---------------------------------------------------------------------------
// Tranche 2 -- images
// ---------------------------------------------------------------------------

/// A raw (binary-safe) stream object: `dict` must end in `/Length`'s value
/// being filled in here.
fn raw_stream(dict_extra: &str, data: &[u8]) -> Vec<u8> {
    let mut v = format!("<< /Length {}{dict_extra} >>\nstream\n", data.len()).into_bytes();
    v.extend_from_slice(data);
    v.extend_from_slice(b"\nendstream");
    v
}

/// One `w x h` pt page that draws image XObject `/Im1` (object 5) over the
/// whole page: `q w 0 0 h 0 0 cm /Im1 Do Q`.
fn image_page(w: u32, h: u32, image_dict: &str, data: &[u8]) -> PdfDocument {
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] \
/Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream("", &format!("q {w} 0 0 {h} 0 0 cm /Im1 Do Q")),
        raw_stream(&format!(" /Type /XObject /Subtype /Image {image_dict}"), data),
    ];
    PdfDocument::open(build_pdf(&bodies)).expect("fixture opens")
}

/// MuPDF shrinks an image with `fz_subsample_pixblock` + the "simple" filter of
/// `fz_scale_pixmap` (draw-device.c:1912), so 1-px black/white stripes drawn
/// at a quarter of their resolution come out as an even mid-grey. 0.4.1
/// nearest-neighbour sampled one source pixel per device pixel and painted
/// pure black columns -- the "bolder, ragged scan" of bd-6lx.
///
/// Expected value measured, not reasoned: `mutool draw -r 72` (19f1284) on
/// this exact PDF paints **127 in every one of the 64 pixels**. (Both axes must
/// shrink: MuPDF's `fz_default_image_scale` only scales when width AND height
/// get smaller -- a 64x4 image drawn 16x4 is point-sampled by MuPDF too, which
/// is what the first draft of this test wrongly assumed otherwise.)
#[test]
fn downscaled_image_is_filtered_not_point_sampled() {
    // 64x16 8-bit gray, columns alternating 0/255, drawn onto 16x4 pt @ 72 dpi.
    let mut px = Vec::new();
    for _y in 0..16 {
        for x in 0..64 {
            px.push(if x % 2 == 0 { 0u8 } else { 255 });
        }
    }
    let doc = image_page(
        16,
        4,
        "/Width 64 /Height 16 /ColorSpace /DeviceGray /BitsPerComponent 8",
        &px,
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!((pix.w, pix.h), (16, 4));
    for (i, v) in pix.samples.iter().enumerate() {
        assert_eq!(*v, 127, "sample {i} = {v}; MuPDF paints 127 everywhere");
    }
}

/// JPEG 2000 used to be a "deferred codec": Unsupported, figure blank. MuPDF
/// decodes it (openjpeg); we now do too (hayro-jpeg2000, AID-0052). Fixture:
/// lossless 8x8 codestream, left half red, right half blue
/// (tests/fixtures/make-image-codecs.py; mutool renders it the same).
#[test]
fn jpx_image_decodes() {
    let j2k = include_bytes!("fixtures/jpx-8x8-red-blue.j2k");
    let doc = image_page(8, 8, "/Width 8 /Height 8 /Filter /JPXDecode", j2k);
    let imgs = page_images(&doc, 0).expect("JPX decodes");
    let im = &imgs[0];
    assert_eq!((im.width, im.height, im.components), (8, 8, 3));
    assert_eq!(&im.pixels[0..3], &[255, 0, 0]);
    assert_eq!(&im.pixels[7 * 3..8 * 3], &[0, 0, 255]);
}

/// JBIG2 likewise: embedded stream, MMR generic region, left 8 columns black
/// (jbig2dec's 1 = black inverted to PDF's 0 = black, filter-jbig2.c:119).
#[test]
fn jbig2_image_decodes() {
    let jb2 = include_bytes!("fixtures/jbig2-16x8-left-black.jb2");
    let doc = image_page(
        16,
        8,
        "/Width 16 /Height 8 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode",
        jb2,
    );
    let imgs = page_images(&doc, 0).expect("JBIG2 decodes");
    let im = &imgs[0];
    assert_eq!((im.width, im.height, im.components), (16, 8, 1));
    assert_eq!(im.pixels[0], 0, "left half black");
    assert_eq!(im.pixels[15], 255, "right half white");
}
