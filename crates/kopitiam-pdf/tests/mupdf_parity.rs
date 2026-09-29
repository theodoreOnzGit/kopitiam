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
//! * **Tranche 5 -- stencils, inline images, encrypted streams**.
//! * **Tranche 6 -- clipping** (`fz_clip_path` masks, text clip, Form
//!   `/BBox`, `gbot`).
//! * **Tranche 7 -- functions, colour spaces, shadings, optional content,
//!   Type3** (`pdf-function.c`, `pdf-colorspace.c`, `pdf-shade.c` / `shade.c` /
//!   `draw-mesh.c`, `pdf-layer.c`, `pdf-type3.c`).
//! * **Tranche 8 -- tiling patterns** (`pdf_show_pattern`,
//!   `fz_draw_begin_tile` / `fz_draw_end_tile`).
//! * Then, unnumbered: image `/Mask` (colour keys, stencil streams), AES-256.
//! * **Tranche 9 -- transparency** (blend modes, soft masks, groups).
//! * **ActualText** (`do_extract_within_actualtext`, MCID structure lookup).
//!
//! From tranche 7 on the expected values are measured with `mutool draw -N -M
//! 0` -- MuPDF's no-ICC, no-spot-simulation mode, which is the mode this port
//! targets (it has no CMS and no overprint simulation; see the coverage map).

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

/// DCTDecode colour handling, against MuPDF's own pixels (gh-116). The bug
/// shape: a 4-component CMYK JPEG with an Adobe APP14 marker (transform 0),
/// samples stored plain, no `/Decode` -- a real paper's figure. 0.4.2 painted
/// it solid BLACK: zune-jpeg converted CMYK to RGB itself assuming
/// Adobe-inverted storage (`r = c * k / 255`), so plain white came out 0.
/// MuPDF takes libjpeg's raw CMYK (never inverted inside a PDF,
/// pdf-stream.c:164), applies `/Decode` (image.c:726) and converts after.
///
/// The fixture (tests/fixtures/make-dct-colour.py; six 16x16 JPEGs drawn 1:1)
/// also covers Adobe-inverted CMYK + `/Decode [1 0 ...]`, YCCK (transform 2),
/// a partial CMYK `/Decode`, a 3-component JPEG stored as RGB with
/// `/ColorTransform 0`, and a gray JPEG with `/Decode [1 0]`. The expected
/// image is `mutool draw -N -M 0 -r 72` at 19f1284, committed beside it.
///
/// Tolerance: every channel of every pixel within 3 levels. zune-jpeg's IDCT
/// and colour rounding are not libjpeg's (AID-0052); the quadrants are single
/// flat 8x8 blocks at quality 100, so a real divergence (a missing inversion,
/// a missing `/Decode`, a wrong colour transform) is >= 64 levels.
#[test]
fn dct_colour_transforms_and_decode_follow_mupdf() {
    let pdf = include_bytes!("fixtures/dct-colour.pdf").to_vec();
    let ppm = include_bytes!("fixtures/dct-colour.mutool.ppm");
    // Binary PPM: "P6\n96 16\n255\n" + RGB rows.
    let mut parts = ppm.splitn(4, |b| *b == b'\n');
    assert_eq!(parts.next(), Some(&b"P6"[..]));
    let dims = std::str::from_utf8(parts.next().unwrap()).unwrap();
    let mut dims = dims.split_whitespace().map(|v| v.parse::<u32>().unwrap());
    let (w, h) = (dims.next().unwrap(), dims.next().unwrap());
    parts.next(); // maxval
    let want = parts.next().unwrap();
    assert_eq!(want.len(), (w * h * 3) as usize);

    let doc = PdfDocument::open(pdf).expect("fixture opens");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!((pix.w, pix.h), (w, h));
    let mut worst = (0i32, 0u32, 0u32);
    let mut bad_images = std::collections::BTreeSet::new();
    for y in 0..h {
        for x in 0..w {
            let got = rgb_at(&pix, x, y);
            let o = ((y * w + x) * 3) as usize;
            for (c, g) in got.iter().enumerate() {
                let d = (*g as i32 - want[o + c] as i32).abs();
                if d > 3 {
                    bad_images.insert(x / 16);
                }
                if d > worst.0 {
                    worst = (d, x, y);
                }
            }
        }
    }
    let (d, x, y) = worst;
    let o = ((y * w + x) * 3) as usize;
    eprintln!("dct-colour: max |d| = {d} vs mutool");
    assert!(
        bad_images.is_empty(),
        "images {bad_images:?} differ from mutool; max |d| = {d} at ({x}, {y}): got {:?}, mutool {:?}",
        rgb_at(&pix, x, y),
        &want[o..o + 3]
    );
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

// ---------------------------------------------------------------------------
// Tranche 5 -- stencils, inline images, encrypted streams
// ---------------------------------------------------------------------------

fn rgb_at(pix: &kopitiam_pdf::mupdf::Pixmap, x: u32, y: u32) -> [u8; 3] {
    let o = ((y * pix.w + x) * pix.n as u32) as usize;
    [pix.samples[o], pix.samples[o + 1], pix.samples[o + 2]]
}

fn near(a: [u8; 3], b: [u8; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (*x as i32 - y as i32).abs() <= 2)
}

/// `/ImageMask true` is a stencil: MuPDF paints the FILL COLOUR where a
/// sample is 0 and leaves everything else alone (fz_fill_image_mask). 0.4.1
/// drew it as an opaque black-and-white picture, wiping out what was under
/// the "transparent" half. mutool on this page: x = 60 -> (254, 0, 0), x = 5
/// -> the blue underneath (0, 0, 255).
#[test]
fn image_mask_paints_the_fill_colour_through_the_stencil() {
    let mask: Vec<u8> = [0xF0u8, 0x0F].repeat(16);
    let mut img = format!(
        "<< /Length {} /Type /XObject /Subtype /Image /Width 16 /Height 16 /ImageMask true /BitsPerComponent 1 >>\nstream\n",
        mask.len()
    )
    .into_bytes();
    img.extend_from_slice(&mask);
    img.extend_from_slice(b"\nendstream");
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /XObject << /M 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        stream("", "0 0 1 rg 0 0 200 200 re f 1 0 0 rg q 200 0 0 200 0 0 cm /M Do Q"),
        img,
    ];
    let doc = PdfDocument::open(build_pdf(&bodies)).expect("fixture opens");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert!(near(rgb_at(&pix, 60, 5), [254, 0, 0]), "{:?}", rgb_at(&pix, 60, 5));
    assert!(near(rgb_at(&pix, 5, 5), [0, 0, 255]), "{:?}", rgb_at(&pix, 5, 5));
}

/// Inline images (`BI … ID … EI`) were skipped. MuPDF decodes and paints
/// them; an 8x8 gray ramp stretched over the page gives, across a row, the
/// column values 0, 32, …, 224 (mutool, point-sampled: it is an upscale
/// beyond 2x, so no bilinear).
#[test]
fn inline_image_is_decoded_and_painted() {
    let ramp: Vec<u8> = (0..8).flat_map(|_| (0..8u8).map(|x| x * 32)).collect();
    let mut content = b"q 200 0 0 200 0 0 cm BI /W 8 /H 8 /CS /G /BPC 8 ID\n".to_vec();
    content.extend_from_slice(&ramp);
    content.extend_from_slice(b"\nEI Q");
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>".to_vec(),
        raw_stream("", &content),
    ];
    let doc = PdfDocument::open(build_pdf(&bodies)).expect("fixture opens");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (x, want) in [(5u32, 0u8), (30, 32), (55, 64), (80, 96), (105, 128), (130, 160), (155, 192), (180, 224)] {
        assert_eq!(rgb_at(&pix, x, 100)[0], want, "x = {x}");
    }
}

/// A FILTERED inline image (ASCIIHex, abbreviated `/F /AHx`, `/CS /RGB`):
/// its data has no length, so the end is found by decoding up to each
/// candidate `EI`. Two pixels, red then green (mutool agrees).
#[test]
fn filtered_inline_image_with_abbreviations_decodes() {
    let doc = page_with(
        "<< >>",
        "q 200 0 0 200 0 0 cm BI /W 2 /H 1 /CS /RGB /BPC 8 /F /AHx ID\nFF000000FF00>\nEI Q",
        &[],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 50, 100), [255, 0, 0]);
    assert_eq!(rgb_at(&pix, 150, 100), [0, 255, 0]);
}

/// An encrypted stream in an object with generation 1 was decrypted with
/// generation 0 -- a different RC4 key -- so the page came out blank. The
/// fixture (RC4-40, content stream `4 1 obj`) is made by MuPDF itself from a
/// synthetic plaintext (tests/fixtures/make-encrypted-gen1.py); mutool shows
/// the red square at (50, 50).
#[test]
fn encrypted_stream_uses_its_objects_generation() {
    let bytes = include_bytes!("fixtures/encrypted-rc4-gen1.pdf").to_vec();
    let doc = PdfDocument::open(bytes).expect("opens");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 50, 50), [255, 0, 0]);
}

// ---------------------------------------------------------------------------
// Tranche 6 -- clipping
// ---------------------------------------------------------------------------

/// A circular clip (four beziers) then a full-page fill: MuPDF masks the fill
/// to the circle (fz_draw_clip_path's non-rectangular branch). 0.4.1 clipped
/// to the circle's BOUNDING BOX, so the corners of that box came out red.
/// mutool: (30,30) and (170,170) -- inside the box, outside the circle --
/// white; centre red; (25,100) -- inside the circle's left edge -- red.
#[test]
fn non_rectangular_clip_masks_to_the_path() {
    let k = 0.5523 * 80.0;
    let circ = format!(
        "100 180 m {a} 180 180 {a} 180 100 c 180 {b} {a} 20 100 20 c {b} 20 20 {b} 20 100 c 20 {a} {b} 180 100 180 c h",
        a = 100.0 + k,
        b = 100.0 - k
    );
    let doc = page_with("<< >>", &format!("q {circ} W n 1 0 0 rg 0 0 200 200 re f Q"), &[]);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 30, 30), [255, 255, 255], "box corner outside the circle");
    assert_eq!(rgb_at(&pix, 170, 170), [255, 255, 255], "box corner outside the circle");
    assert_eq!(rgb_at(&pix, 100, 100), [255, 0, 0]);
    assert_eq!(rgb_at(&pix, 25, 100), [255, 0, 0]);
}

/// A Form XObject is clipped to its /BBox (pdf_run_xobject). The form here
/// paints red far outside its 100x100 box; mutool shows red only inside it.
#[test]
fn form_xobject_is_clipped_to_its_bbox() {
    let doc = page_with(
        "<< /XObject << /F 5 0 R >> >>",
        "/F Do",
        &["<< /Length 30 /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Matrix [1 0 0 1 50 50] >>\nstream\n1 0 0 rg -50 -50 300 300 re f\nendstream"],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 20, 20), [255, 255, 255]);
    assert_eq!(rgb_at(&pix, 100, 100), [255, 0, 0]);
    assert_eq!(rgb_at(&pix, 160, 160), [255, 255, 255]);
}

/// Text render mode 7 adds the glyphs to the clip (flushed at ET). The green
/// fill that follows shows only through "II"; between and after the letters
/// it is paper. 0.4.1 ignored the text clip and flooded the page green.
#[test]
fn text_render_mode_7_clips_to_the_glyphs() {
    let doc = page_with(
        "<< /Font << /F 5 0 R >> >>",
        "q BT /F 100 Tf 7 Tr 10 50 Td (II) Tj ET 0 0.6 0 rg 0 0 200 200 re f Q",
        &["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    // mutool row y = 120: green at x = 24 and 56 (the stems), white at 40
    // (between them) and 120 (past the text).
    assert_eq!(rgb_at(&pix, 24, 120)[1] > 100 && rgb_at(&pix, 24, 120)[0] < 100, true, "{:?}", rgb_at(&pix, 24, 120));
    assert_eq!(rgb_at(&pix, 120, 120), [255, 255, 255]);
    assert_eq!(rgb_at(&pix, 40, 120), [255, 255, 255]);
}

/// A stray `Q` inside a Form XObject must not pop the CALLER's graphics
/// state (MuPDF raises gbot around the form: "gstate underflow"). The page
/// sets a blue fill, runs a form whose content is just `Q Q Q`, then fills:
/// mutool paints blue. 0.4.1 let the form pop the caller's `q` and painted
/// in the default black.
#[test]
fn stray_q_in_a_form_cannot_pop_the_callers_state() {
    let doc = page_with(
        "<< /XObject << /F 5 0 R >> >>",
        "q 0 0 1 rg /F Do 50 50 100 100 re f Q",
        &["<< /Length 5 /Type /XObject /Subtype /Form /BBox [0 0 200 200] >>\nstream\nQ Q Q\nendstream"],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 100, 100), [0, 0, 255]);
}

// ---------------------------------------------------------------------------
// Tranche 7 -- functions, colour spaces, shadings
// ---------------------------------------------------------------------------

/// A 200x200 page from raw bodies: resources + content, extra objects 5...
fn raw_page(resources: &str, content: &[u8], extra: Vec<Vec<u8>>) -> PdfDocument {
    let mut bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources {resources} /Contents 4 0 R >>")
            .into_bytes(),
        raw_stream("", content),
    ];
    bodies.extend(extra);
    PdfDocument::open(build_pdf(&bodies)).expect("fixture opens")
}

/// Indexed fills look the index up in the palette; Separation fills run the
/// tint transform into the alternate space; Lab fills use MuPDF's no-ICC
/// `lab_to_rgb`. 0.4.1: the index as a gray (white), `1 - max(tint)` gray,
/// Lab-as-RGB. mutool -N: (0,255,0), (255,127,127), (201,45,49).
#[test]
fn fill_colour_spaces_convert_like_mupdf() {
    let doc = page_with(
        "<< /ColorSpace << /I [/Indexed /DeviceRGB 1 <FF000000FF00>] \
/S [/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 1 0] /N 1 >>] \
/L [/Lab << /WhitePoint [0.9505 1 1.089] /Range [-100 100 -100 100] >>] >> >>",
        "/I cs 1 sc 0 0 100 100 re f /S cs 0.5 sc 100 0 100 100 re f /L cs 50 60 40 sc 0 100 100 100 re f",
        &[],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert!(near(rgb_at(&pix, 50, 150), [0, 255, 0]), "indexed {:?}", rgb_at(&pix, 50, 150));
    assert!(near(rgb_at(&pix, 150, 150), [255, 127, 127]), "separation {:?}", rgb_at(&pix, 150, 150));
    assert!(near(rgb_at(&pix, 50, 50), [201, 45, 49]), "lab {:?}", rgb_at(&pix, 50, 50));
}

/// `sh` with an axial shading was parsed and ignored (blank page). MuPDF
/// samples the function 256 times and Gouraud-fills an extended quad; mutool
/// -N along y = 100: (255,0,0) (192,0,63) (127,0,127) (63,0,191) (1,0,253).
#[test]
fn axial_shading_sh_paints_the_gradient() {
    let doc = page_with(
        "<< /Shading << /Ax << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 200 0] /Extend [true true] \
/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >> >> >>",
        "/Ax sh",
        &[],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (x, want) in [(0u32, [255u8, 0, 0]), (50, [192, 0, 63]), (100, [127, 0, 127]), (150, [63, 0, 191]), (199, [1, 0, 253])] {
        assert!(near(rgb_at(&pix, x, 100), want), "x = {x}: {:?} want {want:?}", rgb_at(&pix, x, 100));
    }
}

/// A shading PATTERN as the fill colour (`/Pattern cs /P scn ... f`): the
/// path becomes a clip around fz_fill_shade in the pattern space. 0.4.1
/// kept the previous colour. mutool -N: (100,30) 216 gray, centre 127, (100,170)
/// 38, outside the rectangle white.
#[test]
fn shading_pattern_fills_through_the_path() {
    let doc = page_with(
        "<< /Pattern << /P << /PatternType 2 /Shading << /ShadingType 2 /ColorSpace /DeviceGray \
/Coords [0 0 0 200] /Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >> >> >> >>",
        "/Pattern cs /P scn 20 20 160 160 re f",
        &[],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (p, want) in [((100u32, 30u32), 216u8), ((100, 100), 127), ((100, 170), 38), ((10, 100), 255)] {
        let v = rgb_at(&pix, p.0, p.1)[0];
        assert!((v as i32 - want as i32).abs() <= 2, "{p:?}: {v}, want {want}");
    }
}

/// A type 4 free-form triangle mesh read from the bit-packed stream
/// (fz_process_shade_type4) and Gouraud-filled (draw-mesh.c). mutool -N
/// samples: (20,20) (0,25,229), (100,100) (0,127,127), (150,50) (127,191,63).
#[test]
fn type4_mesh_shading_is_gouraud_filled() {
    let mesh: Vec<u8> = vec![
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 1, 255, 255, 255, 255, 0,
    ];
    let mut sh = format!(
        "<< /Length {} /ShadingType 4 /ColorSpace /DeviceRGB /BitsPerCoordinate 8 /BitsPerComponent 8 \
/BitsPerFlag 8 /Decode [0 200 0 200 0 1 0 1 0 1] >>\nstream\n",
        mesh.len()
    )
    .into_bytes();
    sh.extend_from_slice(&mesh);
    sh.extend_from_slice(b"\nendstream");
    let doc = raw_page("<< /Shading << /M 5 0 R >> >>", b"/M sh", vec![sh]);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (p, want) in [((20u32, 20u32), [0u8, 25, 229]), ((100, 100), [0, 127, 127]), ((150, 50), [127, 191, 63])] {
        assert!(near(rgb_at(&pix, p.0, p.1), want), "{p:?}: {:?} want {want:?}", rgb_at(&pix, p.0, p.1));
    }
}

/// A DeviceN image (two inks, a sampled type 0 tint function into RGB) used
/// to be "unsupported colorspace" and drew nothing. mutool -N, 2x2 image
/// stretched over the left half: (25,50) black, (75,50) red, (25,150) green,
/// (75,150) blue.
#[test]
fn devicen_image_goes_through_its_tint_transform() {
    let img = [0u8, 0, 255, 0, 0, 255, 255, 255];
    let samp = [0u8, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255];
    let mut im = format!(
        "<< /Length {} /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 \
/ColorSpace [/DeviceN [/A /B] /DeviceRGB 6 0 R] >>\nstream\n",
        img.len()
    )
    .into_bytes();
    im.extend_from_slice(&img);
    im.extend_from_slice(b"\nendstream");
    let mut f = format!(
        "<< /Length {} /FunctionType 0 /Domain [0 1 0 1] /Range [0 1 0 1 0 1] /Size [2 2] /BitsPerSample 8 >>\nstream\n",
        samp.len()
    )
    .into_bytes();
    f.extend_from_slice(&samp);
    f.extend_from_slice(b"\nendstream");
    let doc = raw_page("<< /XObject << /D 5 0 R >> >>", b"q 100 0 0 200 0 0 cm /D Do Q", vec![im, f]);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (p, want) in [((25u32, 50u32), [0u8, 0, 0]), ((75, 50), [255, 0, 0]), ((25, 150), [0, 255, 0]), ((75, 150), [0, 0, 255])] {
        assert!(near(rgb_at(&pix, p.0, p.1), want), "{p:?}: {:?} want {want:?}", rgb_at(&pix, p.0, p.1));
    }
}

/// Optional content: `/OC /L BDC ... EMC` where `/L` is OFF in the default
/// configuration paints nothing and extracts nothing (pdf_process_BDC +
/// pdf_is_ocg_hidden); an XObject whose /OC is hidden is skipped. 0.4.1 drew
/// every layer.
#[test]
fn optional_content_off_layers_are_hidden() {
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R /OCProperties << /OCGs [5 0 R] /D << /OFF [5 0 R] >> >> >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Properties << /L 5 0 R >> \
/Font << /F 6 0 R >> /XObject << /X 7 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream(
            "",
            "0 1 0 rg 0 0 200 200 re f /OC /L BDC 1 0 0 rg 50 50 100 100 re f BT /F 20 Tf 20 20 Td (Secret) Tj ET EMC \
0 g BT /F 20 Tf 20 170 Td (Shown) Tj ET /X Do",
        ),
        b"<< /Type /OCG /Name (Hidden) >>".to_vec(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        b"<< /Length 22 /Type /XObject /Subtype /Form /BBox [0 0 200 200] /OC 5 0 R >>\nstream\n0 0 1 rg 0 0 30 30 re f\nendstream".to_vec(),
    ];
    let doc = PdfDocument::open(build_pdf(&bodies)).expect("opens");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 100, 100), [0, 255, 0], "the OFF layer's red square must not paint");
    assert_eq!(rgb_at(&pix, 10, 190), [0, 255, 0], "the hidden form's blue square must not paint");
    let text = page_to_stext(&doc, 0, StextOptions::default()).expect("stext").text();
    assert!(text.contains("Shown") && !text.contains("Secret"), "{text:?}");
}

/// Type3 glyphs are content-stream procedures (pdf-type3.c): MuPDF runs them
/// under `FontMatrix · trm`. 0.4.1 drew each glyph as an advance box (and
/// counted it as a fallback glyph); widths ignored the FontMatrix. Here the
/// FontMatrix is 0.002, so /Widths 500 is a 1-em advance: "AB" at 60 pt puts
/// B at x = 80 (mutool stext agrees), and the square glyph fills (40, 130).
#[test]
fn type3_glyph_procedures_run_and_widths_use_the_font_matrix() {
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /T3 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        stream("", "0 0 1 rg BT /T3 60 Tf 20 40 Td (AB) Tj ET"),
        b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 500 500] /FontMatrix [0.002 0 0 0.002 0 0] \
/CharProcs << /sq 6 0 R /tri 7 0 R >> /Encoding << /Type /Encoding /Differences [65 /sq /tri] >> \
/FirstChar 65 /LastChar 66 /Widths [500 500] /Resources << >> >>"
            .to_vec(),
        stream("", "500 0 0 0 500 500 d1 0 0 500 500 re f"),
        stream("", "500 0 0 0 500 500 d1 0 0 m 500 0 l 250 500 l f"),
    ];
    let doc = PdfDocument::open(build_pdf(&bodies)).expect("opens");
    let (pix, fallback) = kopitiam_pdf::mupdf::rasterize_page_ex(&doc, 0, 72.0).expect("renders");
    assert_eq!(fallback, 0, "a Type3 glyph is not a fallback box");
    assert_eq!(rgb_at(&pix, 40, 130), [0, 0, 255], "inside the square glyph (text fill colour, d1 mask)");
    assert_eq!(rgb_at(&pix, 85, 105), [255, 255, 255], "outside the triangle's left edge");
    let cs = chars(&doc);
    assert_eq!(text(&cs), "AB");
    assert!((cs[1].origin.x - 80.0).abs() < 1e-3, "B at {}", cs[1].origin.x);
}

// ---------------------------------------------------------------------------
// Tranche 8 -- tiling patterns (pdf-op-run.c pdf_show_pattern, pdf-pattern.c,
// draw-device.c fz_draw_begin_tile / fz_draw_end_tile).
// ---------------------------------------------------------------------------

/// A coloured tiling pattern (checkerboard cell, 20 pt step) filling the
/// page. 0.4.1 ignored PatternType 1 and kept the previous colour (black
/// everywhere). mutool -N -M 0: black at cell-local (5,5) and (15,15), white
/// at (15,5) and (5,15), in every repeat; pixel-identical page.
#[test]
fn coloured_tiling_pattern_repeats_its_cell() {
    let doc = raw_page(
        "<< /Pattern << /T 5 0 R >> >>",
        b"/Pattern cs /T scn 0 0 200 200 re f",
        vec![raw_stream(
            " /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 20 20] /XStep 20 /YStep 20 /Resources << >>",
            b"0 0 0 rg 0 0 10 10 re f 10 10 10 10 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for (cx, cy) in [(0u32, 0u32), (60, 100), (180, 180)] {
        // device y = 199 - pdf y
        let dev = |x: u32, y: u32| rgb_at(&pix, cx + x, 199 - (cy + y));
        assert_eq!(dev(5, 5), [0, 0, 0], "cell ({cx},{cy}) lower-left");
        assert_eq!(dev(15, 15), [0, 0, 0], "cell ({cx},{cy}) upper-right");
        assert_eq!(dev(15, 5), [255, 255, 255], "cell ({cx},{cy}) lower-right");
        assert_eq!(dev(5, 15), [255, 255, 255], "cell ({cx},{cy}) upper-left");
    }
}

/// An UNCOLOURED (PaintType 2) pattern through `[/Pattern /DeviceRGB]`,
/// under a rotating + scaling /Matrix: painted in the `scn` colour (red),
/// with the cell's own `0 0 0 rg` ignored (gstate->ismask), each repeat
/// placed at MuPDF's truncated integer tile offset. 0.4.1: solid black.
/// Sample points measured with mutool -N -M 0 (3x3-uniform neighbourhoods).
#[test]
fn uncoloured_tiling_pattern_uses_the_scn_colour_and_matrix() {
    let doc = raw_page(
        "<< /Pattern << /U 5 0 R >> /ColorSpace << /PU [/Pattern /DeviceRGB] >> >>",
        b"/PU cs 1 0 0 /U scn 0 0 200 200 re f",
        vec![raw_stream(
            " /Type /Pattern /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 12 12] /XStep 12 /YStep 12 \
/Matrix [1.2 0.7 -0.7 1.2 3 5] /Resources << >>",
            b"0 0 6 6 re f 0 0 0 rg 6 6 6 6 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for p in [(49u32, 66u32), (78, 89), (78, 135), (20, 181), (165, 181)] {
        assert_eq!(rgb_at(&pix, p.0, p.1), [255, 0, 0], "red at {p:?}");
    }
    for p in [(78u32, 20u32), (78, 66), (165, 66), (107, 89), (49, 135), (49, 181)] {
        assert_eq!(rgb_at(&pix, p.0, p.1), [255, 255, 255], "white at {p:?}");
    }
}

/// Text whose fill colour is a tiling pattern: each glyph becomes a clip
/// around the pattern (pdf_flush_text_imp's PDF_MAT_PATTERN). 0.4.1 filled
/// the glyph black. mutool -N -M 0 on the stem of a 120 pt Helvetica-Bold
/// "I": 2-pt-wide blue / yellow stripes (x 19..20 blue, 21..24 yellow).
#[test]
fn text_filled_with_a_tiling_pattern_shows_the_pattern() {
    let doc = raw_page(
        "<< /Pattern << /C 5 0 R >> /Font << /F 6 0 R >> >>",
        b"/Pattern cs /C scn BT /F 120 Tf 10 50 Td (I) Tj ET",
        vec![
            raw_stream(
                " /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 8 8] /XStep 8 /YStep 8 /Resources << >>",
                b"0 0 1 rg 0 0 4 8 re f 1 1 0 rg 4 0 4 8 re f",
            ),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>".to_vec(),
        ],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    for y in [72u32, 100, 128] {
        assert_eq!(rgb_at(&pix, 19, y), [0, 0, 255], "blue stripe, y = {y}");
        assert_eq!(rgb_at(&pix, 22, y), [255, 255, 0], "yellow stripe, y = {y}");
        assert_eq!(rgb_at(&pix, 10, y), [255, 255, 255], "left of the glyph, y = {y}");
    }
}

/// A pattern whose cell fills with ITSELF. MuPDF itself dies with
/// "exception stack overflow" and draws nothing, so there is no oracle
/// value; the port must simply terminate (nesting cap) and keep the page.
#[test]
fn self_referencing_tiling_pattern_terminates() {
    let doc = raw_page(
        "<< /Pattern << /S 5 0 R >> >>",
        b"0 1 0 rg 0 0 50 50 re f /Pattern cs /S scn 100 100 60 60 re f",
        vec![raw_stream(
            " /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 10 10] /XStep 10 /YStep 10 \
/Resources << /Pattern << /S 5 0 R >> >>",
            b"0 0 1 rg 0 0 5 5 re f /Pattern cs /S scn 5 5 5 5 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("terminates and renders");
    assert_eq!(rgb_at(&pix, 25, 175), [0, 255, 0], "content before the pattern survives");
}

/// Image `/Mask`: a colour-key ARRAY (fz_mask_color_key, image.c:166 -- on
/// the raw samples, so a 4-bit key range compares 4-bit values) and an
/// explicit stencil `/Mask` STREAM (pdf-image.c:146, inverted 1-bit
/// samples). 0.4.1 drew all three images opaque over the blue background.
/// mutool -N -M 0: pixel-identical page; the probes below are its values.
#[test]
fn image_mask_colour_key_and_stencil_stream_let_the_background_through() {
    let ck_rgb: &[u8] = &[255, 255, 255, 255, 0, 0, 0, 255, 0, 255, 255, 255];
    let ck_g4: &[u8] = &[0x05, 0x9F, 0xF7, 0x31];
    let st_img: Vec<u8> = [200u8, 30, 30].repeat(4);
    let st_mask: &[u8] = &[0x40, 0x80];
    let doc = raw_page(
        "<< /XObject << /A 5 0 R /B 6 0 R /C 7 0 R >> >>",
        b"0 0 1 rg 0 0 200 200 re f q 100 0 0 100 0 100 cm /A Do Q \
q 200 0 0 100 0 0 cm /B Do Q q 100 0 0 100 100 100 cm /C Do Q",
        vec![
            raw_stream(
                " /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceRGB \
/Mask [250 255 250 255 250 255]",
                ck_rgb,
            ),
            raw_stream(
                " /Type /XObject /Subtype /Image /Width 4 /Height 2 /BitsPerComponent 4 /ColorSpace /DeviceGray /Mask [5 9]",
                ck_g4,
            ),
            raw_stream(
                " /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceRGB /Mask 8 0 R",
                &st_img,
            ),
            raw_stream(" /Type /XObject /Subtype /Image /Width 2 /Height 2 /ImageMask true", st_mask),
        ],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    let blue = [0u8, 0, 255];
    for (p, want) in [
        ((25u32, 25u32), blue),          // RGB white keyed out
        ((75, 25), [255, 0, 0]),         // RGB red kept
        ((75, 75), blue),                // RGB white keyed out
        ((25, 150), [255, 255, 255]),    // gray 15 kept
        ((75, 150), blue),               // gray 7 in [5, 9]: keyed
        ((125, 150), [51, 51, 51]),      // gray 3 kept
        ((125, 25), [200, 30, 30]),      // stencil bit 0: painted
        ((175, 25), blue),               // stencil bit 1: masked out
        ((125, 75), blue),
        ((175, 75), [200, 30, 30]),
    ] {
        assert_eq!(rgb_at(&pix, p.0, p.1), want, "at {p:?}");
    }
}

/// AES-256 (`/V 5 /R 6`, `/AESV3`), made by mutool itself
/// (`tests/fixtures/make-encrypted-aes256.py`). 0.4.1 refused R5/R6 by name,
/// so this Acrobat X+-style owner-restricted file (empty user password) did
/// not open at all. mutool renders the red square at (50, 50); the Info
/// title string decrypts too. The second fixture has user password
/// "secret" and an EMPTY owner password: the empty password authenticates
/// as the owner, but MuPDF refuses that "to match Acrobat"
/// (pdf-crypt.c:817) and mutool will not open it without `-p`; so must we.
#[test]
fn aes_256_r6_file_opens_with_the_empty_user_password() {
    let doc = PdfDocument::open(include_bytes!("fixtures/encrypted-aes256-r6.pdf").to_vec())
        .expect("R6 with an empty user password opens");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 50, 50), [255, 0, 0], "the red square decrypts");
    assert_eq!(rgb_at(&pix, 5, 5), [255, 255, 255]);
    let info = doc.resolve(doc.trailer().dict_gets("Info").expect("/Info")).expect("resolves");
    let title = doc.resolve_get(&info, "Title").expect("title");
    assert_eq!(title.to_string_bytes(), b"AES-256 fixture");
    let refused = PdfDocument::open(include_bytes!("fixtures/encrypted-aes256-r6-empty-owner.pdf").to_vec());
    assert!(refused.is_err(), "an empty owner password must not open the file");
}

/// gh-98's exact shape, end to end: `/Standard /V 4 /R 4 /AESV2`, empty user
/// password, owner-restricted (`/P -4`), made by mutool itself
/// (`tests/fixtures/make-encrypted-aes128.py`). The content stream is
/// deflated, so it must decrypt BEFORE it inflates (the other order is what
/// produced gh-98's "corrupt object stream"), and the page tree sits in an
/// object stream whose members must not be decrypted a second time. mutool
/// renders the red square at (50, 50); the direct-object /Info title string
/// decrypts too. The crypt.rs unit tests pin R4's key derivation and /U check;
/// until this test nothing opened a real AESV2 file.
#[test]
fn aes_128_r4_owner_restricted_form_opens_with_the_empty_user_password() {
    let doc = PdfDocument::open(include_bytes!("fixtures/encrypted-aes128-r4.pdf").to_vec())
        .expect("V4 R4 AESV2 with an empty user password opens");
    assert_eq!(doc.page_count(), 1, "the page tree inside the object stream resolves");
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_eq!(rgb_at(&pix, 50, 50), [255, 0, 0], "the deflated content stream decrypts");
    assert_eq!(rgb_at(&pix, 5, 5), [255, 255, 255]);
    let info = doc.resolve(doc.trailer().dict_gets("Info").expect("/Info")).expect("resolves");
    let title = doc.resolve_get(&info, "Title").expect("title");
    assert_eq!(title.to_string_bytes(), b"AES-128 fixture");
}

// ---------------------------------------------------------------------------
// Tranche 9 -- transparency (draw-blend.c, fz_draw_begin/end_group and
// _mask, pdf-op-run.c begin_softmask / pdf_begin_group / pdf_run_xobject)
// ---------------------------------------------------------------------------
//
// Every value below is `mutool draw -N -M 0 -r 72 -c rgb` on the same page
// (the fixtures are the transparency entries of
// scripts/mupdf-feature-corpus.py). The port matches most of these pages
// pixel for pixel; `near` allows MuPDF's +-2 rounding slack anyway.

/// The four mid-tone stripes the transparency fixtures draw everything over.
const STRIPES: &[u8] = b"1 0.2 0.2 rg 0 0 50 200 re f 0.2 0.8 0.3 rg 50 0 50 200 re f \
0.3 0.4 0.9 rg 100 0 50 200 re f 0.9 0.9 0.5 rg 150 0 50 200 re f\n";

fn with_stripes(ops: &[u8]) -> Vec<u8> {
    let mut v = STRIPES.to_vec();
    v.extend_from_slice(ops);
    v
}

fn assert_probes(pix: &kopitiam_pdf::mupdf::Pixmap, probes: &[((u32, u32), [u8; 3])]) {
    for &((x, y), want) in probes {
        let got = rgb_at(pix, x, y);
        assert!(near(got, want), "at ({x}, {y}): got {got:?}, mutool {want:?}");
    }
}

/// The original feature file (#15): a Multiply rect and a luminosity soft
/// mask from ExtGState. 0.4.1 ignored `/BM` and `/SMask`: the yellow rect
/// was painted plain and the red one unmasked (62.5 % of the page grossly
/// off). mutool: (25,150) cyan x yellow multiplied = (0,255,0) only where
/// they overlap -- (25,150) is plain cyan (0,255,255), (100,150) the
/// multiply (0,255,0); the mask lets red through only in its white square:
/// (100,50) (255,0,0), (25,50) white (255,255,255).
#[test]
fn blend_mode_and_luminosity_smask_from_extgstate() {
    let doc = raw_page(
        "<< /ExtGState << /M << /BM /Multiply >> /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> >> >>",
        b"0 1 1 rg 0 0 200 100 re f q /M gs 1 1 0 rg 50 0 100 100 re f Q \
q /S gs 1 0 0 rg 0 100 200 100 re f Q",
        vec![raw_stream(
            " /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency /CS /DeviceGray >>",
            b"0 0 200 200 re 0 g f 1 g 50 50 100 100 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((25, 150), [0, 255, 255]), ((100, 150), [0, 255, 0]), ((100, 50), [255, 0, 0]), ((25, 50), [255, 255, 255])]);
}

/// Separable blend modes (fz_blend_separable, draw-blend.c:350), source
/// (0.6, 0.3, 0.8) over the stripes. 0.4.1 painted every band as the plain
/// source (153, 76, 204). mutool: Multiply over red (153,15,41), Screen over
/// green (173,219,219), Darken over blue (76,76,204), ColorBurn over yellow
/// (212,168,95), Difference over green (102,128,128), Exclusion over blue
/// (137,118,67), Normal at ca 0.5 over red (204,63,126).
#[test]
fn separable_blend_modes_follow_mupdf() {
    let modes = ["Multiply", "Screen", "Overlay", "Darken", "Lighten", "ColorDodge", "ColorBurn", "HardLight", "SoftLight", "Difference", "Exclusion", "Normal"];
    let mut res = String::from("<< /ExtGState << ");
    let mut ops = Vec::new();
    for (i, m) in modes.iter().enumerate() {
        let ca = if *m == "Normal" { " /ca 0.5" } else { "" };
        res.push_str(&format!("/B{i} << /BM /{m}{ca} >> "));
        ops.extend_from_slice(format!("q /B{i} gs 0.6 0.3 0.8 rg 0 {} 200 14 re f Q\n", 200 - 16 * (i + 1)).as_bytes());
    }
    res.push_str(">> >>");
    let doc = raw_page(&res, &with_stripes(&ops), vec![]);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(
        &pix,
        &[
            ((25, 9), [153, 15, 41]),
            ((75, 25), [173, 219, 219]),
            ((125, 57), [76, 76, 204]),
            ((175, 105), [212, 168, 95]),
            ((75, 153), [102, 128, 128]),
            ((125, 169), [137, 118, 67]),
            ((25, 185), [204, 63, 126]),
        ],
    );
}

/// Non-separable blend modes (fz_hue_rgb / fz_saturation_rgb /
/// fz_color_rgb / fz_luminosity_rgb). 0.4.1: plain source colour --
/// (230,51,127) left, (51,153,229) right. mutool: Hue over red
/// (246,42,129), Saturation over yellow (237,237,59), Color over blue
/// (28,130,206), Luminosity over yellow (142,142,40).
#[test]
fn nonseparable_blend_modes_follow_mupdf() {
    let modes = ["Hue", "Saturation", "Color", "Luminosity"];
    let mut res = String::from("<< /ExtGState << ");
    let mut ops = Vec::new();
    for (i, m) in modes.iter().enumerate() {
        res.push_str(&format!("/B{i} << /BM /{m} >> "));
        let y = 200 - 50 * (i + 1) + 5;
        ops.extend_from_slice(format!("q /B{i} gs 0.9 0.2 0.5 rg 0 {y} 100 40 re f 0.2 0.6 0.9 rg 100 {y} 100 40 re f Q\n").as_bytes());
    }
    res.push_str(">> >>");
    let doc = raw_page(&res, &with_stripes(&ops), vec![]);
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((25, 25), [246, 42, 129]), ((175, 75), [237, 237, 59]), ((125, 125), [28, 130, 206]), ((175, 175), [142, 142, 40])]);
}

/// A luminosity soft mask with `/BC`: the mask group only paints inside its
/// `/BBox`; outside, the mask is the backdrop colour's gray (fz_draw_begin_mask
/// clears to it). `/BC [0.2 0.5 0.8]` in the group's `/CS /DeviceRGB` is
/// gray 0.443. 0.4.1: the red fill unmasked, (255,0,0) everywhere. mutool
/// (red over blue): outside the BBox (111,0,143); under the 0.2 gray rect
/// (50,0,204); under the red rect, luminosity 0.3, (75,0,179); under the
/// 0.8 gray rect (204,0,50); under the blue rect (27,0,227).
#[test]
fn luminosity_smask_uses_the_backdrop_colour_outside_its_group() {
    let doc = raw_page(
        "<< /ExtGState << /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R /BC [0.2 0.5 0.8] >> >> >> >>",
        b"0 0 1 rg 0 0 200 200 re f q /S gs 1 0 0 rg 0 0 200 200 re f Q",
        vec![raw_stream(
            " /Type /XObject /Subtype /Form /BBox [20 20 180 180] /Group << /S /Transparency /CS /DeviceRGB >>",
            b"0.2 g 20 20 50 160 re f 1 0 0 rg 70 20 50 160 re f 0 0 1 rg 120 20 60 80 re f 0.8 g 120 100 60 80 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(
        &pix,
        &[((10, 10), [111, 0, 143]), ((45, 100), [50, 0, 204]), ((95, 100), [75, 0, 179]), ((150, 50), [204, 0, 50]), ((150, 150), [27, 0, 227])],
    );
}

/// An `/S /Alpha` soft mask is the ALPHA its group paints: an opaque rect
/// (full), a `ca 0.5` rect (half), nothing (none). 0.4.1: the green fill
/// covered the whole page. mutool: (50,100) green (0,127,0); (150,100)
/// half green over the yellow stripe (116,178,64); (150,30) the stripe
/// (229,229,127).
#[test]
fn alpha_smask_is_the_groups_alpha() {
    let doc = raw_page(
        "<< /ExtGState << /S << /SMask << /Type /Mask /S /Alpha /G 5 0 R >> >> >> >>",
        &with_stripes(b"q /S gs 0 0.5 0 rg 0 0 200 200 re f Q"),
        vec![raw_stream(
            " /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >> \
/Resources << /ExtGState << /H << /ca 0.5 >> >> >>",
            b"0 0 1 rg 10 10 90 180 re f /H gs 1 1 0 rg 60 60 130 80 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((50, 100), [0, 127, 0]), ((150, 100), [116, 178, 64]), ((150, 30), [229, 229, 127])]);
}

/// A soft mask's `/TR` (here `1 - x`) maps the mask -- but MuPDF's
/// begin_softmask DROPS the transfer function from the graphics state the
/// first time it is used, so only the FIRST object painted after the `gs`
/// gets it; the port replicates that. 0.4.1: no mask at all, the page
/// black. mutool: top (first object, inverted mask) (62,62,62) left /
/// (192,192,192) right; bottom (second object, plain mask) (192,192,192) /
/// (63,63,63).
#[test]
fn smask_transfer_function_applies_to_the_first_object_only() {
    let doc = raw_page(
        "<< /ExtGState << /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R \
/TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> >> >> >> >>",
        b"1 1 1 rg 0 0 200 200 re f q /S gs 0 0 0 rg 0 100 200 100 re f 0 0 0 rg 0 0 200 100 re f Q",
        vec![raw_stream(
            " /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >>",
            b"0.25 g 0 0 100 200 re f 0.75 g 100 0 100 200 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((50, 50), [62, 62, 62]), ((150, 50), [192, 192, 192]), ((50, 150), [192, 192, 192]), ((150, 150), [63, 63, 63])]);
}

/// Transparency groups (fz_draw_begin_group with ATTEMPT_KNOCKOUT_AND_ISOLATED):
/// a white `/Difference` rect inside a NON-isolated group inverts the page
/// under it; inside an ISOLATED group it differences against nothing and
/// stays white. Both groups are drawn at `ca 0.6` (the group alpha of the
/// `Do`). 0.4.1 ignored `/Group` and `/BM`: plain white at 0.6. mutool:
/// non-isolated over red (102,142,142), over green (142,113,137);
/// isolated over blue (183,193,244), over yellow (244,244,203); a Normal
/// group at ca 0.5 over green (153,102,165).
#[test]
fn nonisolated_and_isolated_groups_with_group_alpha() {
    let grp: &[u8] = b"/M gs 1 1 1 rg 10 10 80 80 re f 0 0 0 rg 30 30 40 40 re f";
    let res = " /Resources << /ExtGState << /M << /BM /Difference >> >> >>";
    let doc = raw_page(
        "<< /XObject << /N 5 0 R /I 6 0 R /P 7 0 R >> /ExtGState << /GA << /ca 0.6 >> /HA << /ca 0.5 >> >> >>",
        &with_stripes(b"q /GA gs 1 0 0 1 0 100 cm /N Do Q q /GA gs 1 0 0 1 100 100 cm /I Do Q q /HA gs 1 0 0 1 50 0 cm /P Do Q"),
        vec![
            raw_stream(&format!(" /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Group << /S /Transparency >>{res}"), grp),
            raw_stream(&format!(" /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Group << /S /Transparency /I true >>{res}"), grp),
            raw_stream(
                " /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Group << /S /Transparency >>",
                b"1 0 1 rg 0 0 100 100 re f 0 0 0 rg 25 25 50 50 re f",
            ),
        ],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(
        &pix,
        &[((20, 50), [102, 142, 142]), ((50, 50), [142, 113, 137]), ((120, 50), [183, 193, 244]), ((150, 50), [244, 244, 203]), ((60, 190), [153, 102, 165])],
    );
}

/// Images and transparency (pdf_show_image): an image under a gstate soft
/// mask is masked; an image with its OWN `/SMask` under `/BM /Multiply`
/// gets a blend group but no gstate mask. 0.4.1: the top image opaque, the
/// bottom one merely alpha-blended. mutool: top, mask 0.2, over green
/// (91,163,60); mask 0.9 over blue... (150,25) (23,252,12); bottom,
/// multiply at smask 255 over green (51,0,0), at smask 128 over yellow
/// (113,228,63).
#[test]
fn image_under_gstate_smask_and_image_smask_under_blend() {
    let img: &[u8] = &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
    let doc = raw_page(
        "<< /XObject << /A 6 0 R /B 7 0 R >> /ExtGState << /S << /SMask << /Type /Mask /S /Luminosity /G 5 0 R >> >> \
/M << /BM /Multiply /SMask /None >> >> >>",
        &with_stripes(b"q /S gs 200 0 0 100 0 100 cm /A Do Q q /M gs 200 0 0 100 0 0 cm /B Do Q"),
        vec![
            raw_stream(
                " /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >>",
                b"0.2 g 0 0 100 200 re f 0.9 g 100 0 100 200 re f",
            ),
            raw_stream(" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceRGB", img),
            raw_stream(" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceRGB /SMask 8 0 R", img),
            raw_stream(" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceGray", &[255, 128, 64, 0]),
        ],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((50, 25), [91, 163, 60]), ((150, 25), [23, 252, 12]), ((50, 125), [51, 0, 0]), ((150, 125), [113, 228, 63])]);
}

/// Text under a blend mode and under a soft mask (pdf_flush_text_imp's
/// pdf_begin_group). 0.4.1: white text on the stripes, and the masked word
/// drawn whole in black. mutool: the white `/Difference` "D" stem over the
/// red stripe is cyan (0,204,204); the "M" of the masked word is black at
/// (12,150); the "k", right of the mask's edge at x = 100, is not drawn:
/// the blue stripe (76,102,229) at (139,150).
#[test]
fn text_under_blend_mode_and_soft_mask() {
    let doc = raw_page(
        "<< /Font << /F 5 0 R >> /ExtGState << /D << /BM /Difference >> /S << /SMask << /Type /Mask /S /Alpha /G 6 0 R >> >> >> >>",
        &with_stripes(b"q /D gs 1 1 1 rg BT /F 64 Tf 5 120 Td (Diff) Tj ET Q q /S gs 0 0 0 rg BT /F 64 Tf 5 30 Td (Mask) Tj ET Q"),
        vec![
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>".to_vec(),
            raw_stream(" /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >>", b"0 0 0 rg 0 0 100 200 re f"),
        ],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((12, 60), [0, 204, 204]), ((12, 150), [0, 0, 0]), ((139, 150), [76, 102, 229])]);
}

/// A tiling-pattern fill at `ca 0.5`: pdf_show_path wraps the pattern in a
/// Normal group at the fill alpha (the cell itself runs at the PARENT
/// state's alpha, 1). 0.4.1 drew the checkerboard opaque: (0,0,255) and
/// (255,255,0). mutool: blue half over red (128,26,153), yellow half over
/// red (255,153,26), the empty cells the stripe (51,204,76).
#[test]
fn tiling_pattern_fill_at_fill_alpha() {
    let doc = raw_page(
        "<< /Pattern << /P 5 0 R >> /ExtGState << /H << /ca 0.5 >> >> >>",
        &with_stripes(b"q /H gs /Pattern cs /P scn 20 20 160 160 re f Q"),
        vec![raw_stream(
            " /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 20 20] /XStep 20 /YStep 20 /Resources << >>",
            b"0 0 1 rg 0 0 10 10 re f 1 1 0 rg 10 10 10 10 re f",
        )],
    );
    let pix = rasterize_page_native(&doc, 0, 72.0).expect("renders");
    assert_probes(&pix, &[((25, 175), [128, 26, 153]), ((35, 165), [255, 153, 26]), ((75, 175), [51, 204, 76])]);
}

/// MuPDF's stext device has no begin_mask: a soft mask's group is run
/// through it like any content, so glyphs INSIDE the mask are extracted.
/// 0.4.1 never ran the mask, so the mask's word was missing from the text.
/// `mutool draw -F text` on this page prints "Hidden" (from the mask group)
/// and then "Shown". (MuPDF also drops a mask glyph that falls outside the
/// device scissor -- the masked object's bbox -- via FZ_STEXT_CLIP; this
/// port's stext device has no scissor yet, so here the mask word sits on
/// top of the shown one, where both agree.)
#[test]
fn soft_mask_group_text_reaches_stext_like_mupdf() {
    let doc = raw_page(
        "<< /Font << /F 5 0 R >> /ExtGState << /S << /SMask << /Type /Mask /S /Luminosity /G 6 0 R >> >> >> >>",
        b"q /S gs 0 0 0 rg BT /F 24 Tf 10 100 Td (Shown) Tj ET Q",
        vec![
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            raw_stream(
                " /Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency >> /Resources << /Font << /F 5 0 R >> >>",
                b"1 g BT /F 24 Tf 10 100 Td (Hidden) Tj ET",
            ),
        ],
    );
    let t = text(&chars(&doc));
    assert!(t.contains("Shown"), "page text: {t:?}");
    assert!(t.contains("Hidden"), "mask-group text reaches stext: {t:?}");
}

/// ActualText (stext-device.c `do_extract_within_actualtext` /
/// `fz_stext_end_metatext`, pdf-op-run.c `begin_metatext` + `lookup_mcid`):
/// an inline `/ActualText (lie)` replaces "fib" inside "Politicians fib,
/// always." (matching prefix and suffix kept); a span whose MCID's
/// STRUCTURE ELEMENT says `/ActualText ()` vanishes; an ActualText over an
/// image is placed at the image. 0.4.1 ignored marked content: "fib",
/// "Hidden words", and no "E=mc2". mutool `-F text` on the same bytes
/// (scripts/mupdf-feature-corpus.py #34): "Politicians lie, always." /
/// "Before" / "E=mc2"; the harness matches all 33 chars in place.
#[test]
fn actualtext_replaces_glyphs_including_via_the_structure_tree() {
    let img: &[u8] = &[0, 128, 255, 64];
    let bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R /MarkInfo << /Marked true >> >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /StructParents 0 \
/Resources << /Font << /F 5 0 R >> /XObject << /Im 9 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream(
            "",
            "BT /F 14 Tf 20 170 Td (Politicians ) Tj /Span << /ActualText (lie) >> BDC (fib) Tj EMC (, always.) Tj ET\n\
/Span << /MCID 0 >> BDC BT /F 14 Tf 20 120 Td (Hidden words) Tj ET EMC\n\
BT /F 14 Tf 20 80 Td (Before) Tj ET\n\
/Figure << /ActualText (E=mc2) >> BDC q 60 0 0 20 20 40 cm /Im Do Q EMC",
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        b"<< /Type /StructTreeRoot /ParentTree 7 0 R /K [8 0 R] >>".to_vec(),
        b"<< /Nums [0 [8 0 R]] >>".to_vec(),
        b"<< /Type /StructElem /S /Span /P 6 0 R /Pg 3 0 R /K 0 /ActualText () >>".to_vec(),
        raw_stream(" /Type /XObject /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceGray", img),
    ];
    let doc = PdfDocument::open(build_pdf(&bodies)).expect("opens");
    let t = text(&chars(&doc));
    assert!(t.contains("Politicians lie, always."), "{t:?}");
    assert!(!t.contains("fib"), "{t:?}");
    assert!(!t.contains("Hidden"), "the empty structure ActualText swallows the span: {t:?}");
    assert!(t.contains("Before"), "{t:?}");
    assert!(t.contains("E=mc2"), "ActualText over an image is placed at it: {t:?}");
}
