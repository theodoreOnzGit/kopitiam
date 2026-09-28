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
//! * **Tranche 3 -- graphics state** (`gs`, `d`/`J`/`j`/`M`, stroked text,
//!   CMYK conversion).
//! * **Tranche 4 -- page boxes** (`pdf_page_obj_transform_box`: the CropBox
//!   is the page).

use kopitiam_pdf::mupdf::structured_text::{StextBlock, StextChar, StextOptions};
use kopitiam_pdf::mupdf::xref::PdfDocument;
use kopitiam_pdf::mupdf::page_geom::{page_media_box_points, page_size_points};
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

// ---------------------------------------------------------------------------
// Tranche 3 -- graphics state
// ---------------------------------------------------------------------------

/// A 200x200 pt page with the given resources and content (no fonts etc.
/// unless the caller puts them in `extra` as objects 5..).
fn page_with(resources: &str, content: &str, extra: &[&str]) -> PdfDocument {
    let mut bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources {resources} /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream("", content),
    ];
    bodies.extend(extra.iter().map(|e| e.as_bytes().to_vec()));
    PdfDocument::open(build_pdf(&bodies)).expect("fixture opens")
}

/// Red channel of device pixel `(x, y)` of a 72-dpi native render.
fn red_at(pix: &kopitiam_pdf::mupdf::Pixmap, x: u32, y: u32) -> u8 {
    pix.samples[((y * pix.w + x) * pix.n as u32) as usize]
}

/// `d` was parsed-and-ignored, so every dashed line drew solid; and every
/// stroke got round caps whatever `J` said. MuPDF (draw-path.c dash walker,
/// butt caps by default) on `12 w [30 15] 0 d 20 100 m 180 100 l S`, measured
/// with mutool at y = 100: x = 25, 40 ink; 55 (gap 50..65) paper; 70 ink;
/// 183 (past the butt end at 180) paper.
#[test]
fn dash_pattern_and_butt_caps_follow_mupdf() {
    let doc = page_with("<< >>", "0 0 0 RG 12 w [30 15] 0 d 20 100 m 180 100 l S", &[]);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (x, ink) in [(25, true), (40, true), (55, false), (70, true), (183, false), (10, false)] {
        let v = red_at(&pix, x, 100);
        assert_eq!(v < 64, ink, "x = {x}: red {v}, want ink = {ink}");
    }
}

/// ExtGState `/ca` was ignored, so translucent fills painted opaque. MuPDF:
/// blue at ca 0.5 over red = (128, 0, 126) (mutool, 19f1284).
#[test]
fn extgstate_fill_alpha_blends() {
    let doc = page_with(
        "<< /ExtGState << /A << /ca 0.5 >> >> >>",
        "1 0 0 rg 0 0 200 200 re f /A gs 0 0 1 rg 0 0 100 200 re f",
        &[],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    let o = ((50 * pix.w + 50) * pix.n as u32) as usize;
    let rgb = &pix.samples[o..o + 3];
    for (got, want) in rgb.iter().zip([128u8, 0, 126]) {
        assert!((*got as i32 - want as i32).abs() <= 2, "got {rgb:?}, MuPDF (128, 0, 126)");
    }
}

/// Text render mode 1 is stroke-only: the glyph interior stays paper. The
/// port used to FILL every glyph in modes 0..=6 except 3.
#[test]
fn text_render_mode_1_strokes_without_filling() {
    let doc = page_with(
        "<< /Font << /F 5 0 R >> >>",
        "0 0 1 RG 1 w 1 0 0 rg BT /F 150 Tf 1 Tr 20 40 Td (I) Tj ET",
        &["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    // Middle of the I's stem: MuPDF leaves it white (mutool: (255,255,255)).
    let o = ((100 * pix.w + 45) * pix.n as u32) as usize;
    assert_eq!(&pix.samples[o..o + 3], &[255, 255, 255], "stem interior must stay unfilled");
    // ...while the outline itself is inked in the stroke colour somewhere on
    // the row's left edge.
    let edge = (20..40).any(|x| {
        let o = ((100 * pix.w + x) * pix.n as u32) as usize;
        pix.samples[o + 2] > 200 && pix.samples[o] < 128
    });
    assert!(edge, "no blue outline found on the stem's left edge");
}

/// MuPDF's no-ICC CMYK conversion is `1 - min(1, c + k)` (color-fast.c:117),
/// not `(1 - c)(1 - k)`.
#[test]
fn cmyk_conversion_is_mupdfs_fast_path() {
    use kopitiam_pdf::mupdf::cmyk_to_rgb;
    assert_eq!(cmyk_to_rgb(0.5, 0.5, 0.5, 0.5), [0.0, 0.0, 0.0]);
    assert_eq!(cmyk_to_rgb(0.25, 0.0, 1.0, 0.25), [0.5, 0.75, 0.0]);
}

// ---------------------------------------------------------------------------
// Tranche 4 -- page boxes: the CropBox is the page
// ---------------------------------------------------------------------------

/// A one-page document with free-form `/Pages` and `/Page` extras (boxes,
/// /Rotate) and content, font `/F` = Helvetica (object 5).
fn boxed_page(pages_extra: &str, page_extra: &str, content: &str) -> PdfDocument {
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!("<< /Type /Pages /Kids [3 0 R] /Count 1 {pages_extra} >>").into_bytes(),
        format!(
            "<< /Type /Page /Parent 2 0 R {page_extra} \
/Resources << /Font << /F 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream("", content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ];
    PdfDocument::open(build_pdf(&bodies)).expect("fixture opens")
}

/// The IAEA-TECDOC cover shape (maintainer report 2026-09-28, "my iaea tecdoc
/// document is unusually wide"; the real files are restricted and are NOT
/// fixtures): MediaBox a two-page spread, CropBox the A4 right half. A red
/// square and a word inside the crop, a blue square and a word outside it.
const SPREAD: &str = "/MediaBox [0 0 1340 898] /CropBox [717 28 1312 870]";
const SPREAD_CONTENT: &str = "1 0 0 rg 800 100 100 100 re f 0 0 1 rg 100 100 100 100 re f 0 g \
BT /F 20 Tf 800 600 Td (Inside) Tj ET BT /F 20 Tf 100 600 Td (Outside) Tj ET";

/// Size: MuPDF's pdf_bound_page is the CropBox (clipped to the MediaBox),
/// 595 x 842 -- poppler says the same. 0.4.1 said 1340 x 898.
#[test]
fn cropbox_is_the_page_size() {
    let doc = boxed_page("", SPREAD, SPREAD_CONTENT);
    assert_eq!(page_size_points(&doc, 0), (595.0, 842.0));
    let mb = page_media_box_points(&doc, 0);
    assert_eq!((mb.x0, mb.y0, mb.x1, mb.y1), (717.0, 28.0, 1312.0, 870.0));
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!((pix.w, pix.h), (595, 842));
}

/// Raster: the CropBox corner is the origin (the red square at user x 800
/// lands at device x 83), and content outside the crop is not drawn at all
/// (MuPDF clips to the CropBox, pdf-run.c:179).
#[test]
fn cropbox_offsets_and_clips_the_raster() {
    let doc = boxed_page("", SPREAD, SPREAD_CONTENT);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    // user (850, 150) -> device (850 - 717, 870 - 150) = (133, 720).
    let o = ((720 * pix.w + 133) * pix.n as u32) as usize;
    assert_eq!(&pix.samples[o..o + 3], &[255, 0, 0], "red square not at the crop-relative spot");
    let blue = pix
        .samples
        .chunks(pix.n as usize)
        .any(|p| p[2] > 200 && p[0] < 50 && p[1] < 50);
    assert!(!blue, "the blue square outside the CropBox must not be drawn");
}

/// Structured text lives in the same cropped space as the raster, so text
/// selection lines up: "Inside" starts at x = 800 - 717 = 83, baseline y =
/// 870 - 600 = 270. With FZ_STEXT_CLIP, "Outside" (x < 0) is dropped, as
/// `mutool draw -F stext` does; without it MuPDF keeps it (at negative x).
#[test]
fn cropbox_is_the_stext_coordinate_space() {
    let doc = boxed_page("", SPREAD, SPREAD_CONTENT);
    let page = page_to_stext(&doc, 0, StextOptions { flags: StextOptions::CLIP }).expect("stext");
    assert_eq!(page.text().trim(), "Inside");
    let first = page
        .blocks
        .iter()
        .find_map(|b| match b {
            StextBlock::Text(t) => t.lines.first().and_then(|l| l.chars.first()).cloned(),
            _ => None,
        })
        .expect("a char");
    assert!((first.origin.x - 83.0).abs() < 1e-3 && (first.origin.y - 270.0).abs() < 1e-3, "{:?}", first.origin);
    assert_eq!((page.mediabox.x1 - page.mediabox.x0, page.mediabox.y1 - page.mediabox.y0), (595.0, 842.0));

    let all = page_to_stext(&doc, 0, StextOptions::default()).expect("stext");
    assert!(all.text().contains("Outside"), "without CLIP the hidden text is still extracted");
}

/// `/CropBox` is inheritable (§7.7.3.4): set on the /Pages node, it applies.
#[test]
fn cropbox_is_inherited_from_the_page_tree() {
    let doc = boxed_page("/CropBox [717 28 1312 870]", "/MediaBox [0 0 1340 898]", SPREAD_CONTENT);
    assert_eq!(page_size_points(&doc, 0), (595.0, 842.0));
}

/// A CropBox bigger than the paper is clipped to the MediaBox ("never use a
/// box larger than fits the paper", pdf-page.c:766).
///
/// Honest note: this one also passed on 0.4.1, which ignored the CropBox
/// altogether. It is here to pin the INTERSECTION rule of the new code -- a
/// CropBox-aware change that used the raw CropBox would fail it (1050 x 1050).
#[test]
fn cropbox_larger_than_mediabox_is_clipped() {
    let doc = boxed_page("", "/MediaBox [0 0 300 400] /CropBox [-50 -50 1000 1000]", "");
    assert_eq!(page_size_points(&doc, 0), (300.0, 400.0));
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!((pix.w, pix.h), (300, 400));
}

/// /Rotate 90 turns the CROP, not the MediaBox, on its side.
#[test]
fn cropbox_with_rotate_90_swaps_the_crop_extents() {
    let doc = boxed_page("", &format!("{SPREAD} /Rotate 90"), SPREAD_CONTENT);
    assert_eq!(page_size_points(&doc, 0), (842.0, 595.0));
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!((pix.w, pix.h), (842, 595));
    // Rotated 90° clockwise: user (850, 150) -> device (870 - ... ); rather
    // than re-derive the matrix, check the red square is still on the page
    // and the blue one still is not.
    let red = pix.samples.chunks(pix.n as usize).any(|p| p[0] > 200 && p[1] < 50 && p[2] < 50);
    let blue = pix.samples.chunks(pix.n as usize).any(|p| p[2] > 200 && p[0] < 50 && p[1] < 50);
    assert!(red && !blue);
}
