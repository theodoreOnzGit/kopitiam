//! Ported from MuPDF `source/fitz/draw-device.c` (`fz_draw_fill_path`,
//! `fz_draw_stroke_path`, `fz_draw_fill_image`, and the DeviceRGB colour
//! conversion the draw device forces) and the inverse-map image blit of
//! `source/fitz/draw-affine.c` (`fz_paint_image` / `fz_paint_affine`), with the
//! glyph sink following `source/fitz/draw-glyph.c` / `fz_draw_fill_text`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the device call shapes and the
//! CTM/colour composition follow MuPDF; the code is re-expressed in idiomatic
//! Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # The rasterizing device
//!
//! [`DrawDevice`] is the pixmap-rendering counterpart to the extraction
//! [`StextDevice`](super::stext_device): where that device records text, this one
//! *paints*. It implements the interpreter's [`TextDevice`] glyph sink (so
//! [`run_page`](super::run_page) drives it straight away) and exposes the fuller
//! MuPDF device surface -- [`fill_path`](DrawDevice::fill_path),
//! [`stroke_path`](DrawDevice::stroke_path),
//! [`draw_image`](DrawDevice::draw_image) -- as inherent methods a later
//! interpreter wave (or the viewer) can call once the content interpreter emits
//! path/colour/image operators (today it emits only glyphs; see
//! [`interpret`](super::interpret)).
//!
//! Everything composes through the CTM (a [`Matrix`]) onto a device-space
//! [`Pixmap`]; the working colourspace is **DeviceRGB** (gray and CMYK inputs are
//! converted on the way in, matching MuPDF forcing the destination colourspace).
//!
//! ## The fidelity ceiling (what is stubbed -- read before trusting a render)
//!
//! * **Glyph shapes.** This port has *no glyph outlines* -- [`Font`] deliberately
//!   avoids FreeType (see [`font`](super::font)), so there is nothing to fill.
//!   [`show_glyph`](DrawDevice::show_glyph) therefore paints a **filled advance
//!   box** per visible glyph (whitespace is skipped): text renders as legible
//!   *blocks* at the right positions and sizes, not letterforms. True glyph
//!   rendering needs an embedded-font outline decoder (a FreeType-equivalent),
//!   which is out of scope. `// TODO(draw): glyph outlines`.
//! * **Invisible text.** The glyph sink has no render-mode, so `Tr 3/7`
//!   (invisible, e.g. an OCR text layer over a scan) still paints a box. A future
//!   render-mode-aware sink should suppress it.
//! * **Colour.** The content interpreter does not emit colour operators yet, so
//!   fills default to black; the colour plumbing (gray/RGB/CMYK -> RGB) is here
//!   and correct, just not yet fed by the interpreter.
//! * **Not implemented at all** (safe no-ops / skips, never corruption): mesh &
//!   gradient shadings (`draw-mesh.c`), blend modes beyond Normal
//!   (`draw-blend.c`), soft masks, clip masks beyond a rectangular clip,
//!   knockout / transparency groups. Image blitting is **nearest-neighbour**
//!   (no bilinear/mip smoothing) and honours only a rectangular clip.

use super::draw_edge::{fill_polygons, FillRule};
use super::draw_path::Path;
use super::font::Font;
use super::geometry::{IRect, Matrix, Point};
use super::object::Object;
use super::page_image::DecodedImage;
use super::pixmap::Pixmap;
use super::text_device::TextDevice;
use super::xref::PdfDocument;

/// A device that rasterizes onto a [`Pixmap`] in DeviceRGB.
///
/// Construct with [`new`](DrawDevice::new), drive it (via
/// [`run_page`](super::run_page), or by calling the fill/stroke/image methods
/// directly), then take the result with [`into_pixmap`](DrawDevice::into_pixmap).
pub struct DrawDevice {
    /// The render target (DeviceRGB, `n = 3`).
    pix: Pixmap,
    /// A device-space transform applied *after* every incoming CTM -- used to
    /// scale a 72-dpi page transform up to the requested output resolution
    /// ([`rasterize_page`]).
    base: Matrix,
    /// The rectangular clip (device pixels). Defaults to the whole pixmap.
    clip: IRect,
    /// The current fill colour (DeviceRGB, 0..=255). Defaults to black.
    fill: [u8; 3],
}

impl DrawDevice {
    /// Create a device over a fresh white `w×h` DeviceRGB pixmap. `base` is the
    /// post-CTM device transform (use [`Matrix::IDENTITY`] for 1:1).
    pub fn new(w: u32, h: u32, base: Matrix) -> DrawDevice {
        let mut pix = Pixmap::new_rgb(w.max(1), h.max(1));
        pix.clear_with_value(0xff); // white page
        let clip = pix.bbox();
        DrawDevice { pix, base, clip, fill: [0, 0, 0] }
    }

    /// Borrow the current pixmap.
    pub fn pixmap(&self) -> &Pixmap {
        &self.pix
    }

    /// Consume the device, returning the painted pixmap.
    pub fn into_pixmap(self) -> Pixmap {
        self.pix
    }

    /// Set the fill colour from DeviceRGB components (0..=1).
    pub fn set_fill_rgb(&mut self, r: f32, g: f32, b: f32) {
        self.fill = rgb_to_bytes([r, g, b]);
    }

    // MuPDF: fz_draw_fill_path (draw-device.c:672).
    /// Fill `path` (path space) with `color` (DeviceRGB 0..=1) at `alpha`,
    /// transformed by `ctm`, using the winding rule `rule`.
    pub fn fill_path(&mut self, path: &Path, rule: FillRule, ctm: Matrix, color: [f32; 3], alpha: f32) {
        let m = ctm.concat(self.base);
        let polys = path.flatten(m);
        let c = rgb_to_bytes(color);
        fill_polygons(&mut self.pix, &polys, rule, &c, alpha, self.clip);
    }

    // MuPDF: fz_draw_stroke_path (draw-device.c:766).
    /// Stroke `path` with `color` at `alpha`, `line_width` in *path* units,
    /// transformed by `ctm`. Uses the width-expansion approximation
    /// ([`Path::stroke_to_polygons`]).
    pub fn stroke_path(&mut self, path: &Path, ctm: Matrix, line_width: f32, color: [f32; 3], alpha: f32) {
        let m = ctm.concat(self.base);
        // Convert the path-space width to device pixels via the CTM expansion.
        let dev_w = line_width * m.max_expansion();
        let polys = path.stroke_to_polygons(m, dev_w);
        let c = rgb_to_bytes(color);
        fill_polygons(&mut self.pix, &polys, FillRule::NonZero, &c, alpha, self.clip);
    }

    // MuPDF: fz_draw_fill_image (draw-device.c) -> fz_paint_image_imp ->
    // fz_paint_affine (draw-affine.c): inverse-map each destination pixel into the
    // image's unit square and sample.
    /// Blit `img` at `alpha` under `ctm`, which maps the image's unit square
    /// `[0,1]²` (fitz image space, with `(0,0)` top-left) onto the page. Nearest-
    /// neighbour sampling; grayscale (`components == 1`) is expanded to RGB.
    pub fn draw_image(&mut self, img: &DecodedImage, ctm: Matrix, alpha: f32) {
        if img.width == 0 || img.height == 0 || alpha <= 0.0 {
            return;
        }
        let m = ctm.concat(self.base);
        let Some(inv) = m.try_invert() else { return };

        // Device bounds = unit square transformed by m, clamped to clip ∩ pixmap.
        let corners = [
            Point::new(0.0, 0.0).transform(m),
            Point::new(1.0, 0.0).transform(m),
            Point::new(1.0, 1.0).transform(m),
            Point::new(0.0, 1.0).transform(m),
        ];
        let mut x0 = f32::INFINITY;
        let mut y0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut y1 = f32::NEG_INFINITY;
        for p in corners {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        let bounds = IRect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32)
            .intersect(self.pix.bbox())
            .intersect(self.clip);
        if bounds.is_empty() {
            return;
        }

        let iw = img.width;
        let ih = img.height;
        for py in bounds.y0..bounds.y1 {
            for px in bounds.x0..bounds.x1 {
                // Pixel centre -> image space.
                let ip = Point::new(px as f32 + 0.5, py as f32 + 0.5).transform(inv);
                if ip.x < 0.0 || ip.y < 0.0 || ip.x >= 1.0 || ip.y >= 1.0 {
                    continue;
                }
                let sx = ((ip.x * iw as f32) as usize).min(iw - 1);
                let sy = ((ip.y * ih as f32) as usize).min(ih - 1);
                let si = (sy * iw + sx) * img.components as usize;
                let rgb = match img.components {
                    1 => {
                        let v = img.pixels[si];
                        [v, v, v]
                    }
                    _ => [img.pixels[si], img.pixels[si + 1], img.pixels[si + 2]],
                };
                if let Some(o) = self.pix.offset(px, py) {
                    // Straight source-over at the image's alpha.
                    blend_rgb(&mut self.pix.samples[o..o + 3], rgb, alpha);
                }
            }
        }
    }
}

// MuPDF: fz_draw_fill_text -> fz_draw_glyph (draw-device.c / draw-glyph.c). Here,
// with no outline source, reduced to the advance-box placeholder (see the module
// header's fidelity-ceiling note).
impl TextDevice for DrawDevice {
    fn show_glyph(&mut self, font: &Font, trm: Matrix, adv: f32, unicode: char, _cid: u32, _wmode: u8) {
        // Whitespace and zero-width glyphs paint nothing (keeps inter-word gaps).
        if adv <= 0.0 || unicode.is_whitespace() {
            return;
        }
        // TODO(draw): fill the real glyph outline. No outline source exists in
        // this port (font.rs is FreeType-free), so paint the glyph's advance box:
        // a legible block at the correct position/size, inset so adjacent glyphs
        // stay visually distinct and lines do not merge vertically.
        let asc = font.ascender().min(0.9); // ~cap height mass
        let x0 = adv * 0.08;
        let x1 = adv * 0.92;
        let mut path = Path::new();
        path.rect(x0, 0.0, x1, asc);

        let m = trm.concat(self.base);
        let polys = path.flatten(m);
        fill_polygons(&mut self.pix, &polys, FillRule::NonZero, &self.fill, 1.0, self.clip);
    }
}

/// Clamp a DeviceRGB float triple (0..=1) to bytes.
fn rgb_to_bytes(c: [f32; 3]) -> [u8; 3] {
    [
        (c[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (c[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (c[2] * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Straight (non-premultiplied) source-over of `src` at `a` (0..=1) onto a 3-byte
/// RGB destination slice.
fn blend_rgb(dst: &mut [u8], src: [u8; 3], a: f32) {
    let ia = 1.0 - a;
    for k in 0..3 {
        dst[k] = (src[k] as f32 * a + dst[k] as f32 * ia).round().clamp(0.0, 255.0) as u8;
    }
}

// MuPDF: DeviceGray -> DeviceRGB (fz_convert_color): replicate the single value.
/// Convert a DeviceGray value (0..=1) to DeviceRGB.
pub fn gray_to_rgb(g: f32) -> [f32; 3] {
    [g, g, g]
}

// MuPDF: DeviceCMYK -> DeviceRGB (the naive conversion fz_cmyk_to_rgb uses when no
// ICC profile applies).
/// Convert DeviceCMYK (each 0..=1) to DeviceRGB.
pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> [f32; 3] {
    [(1.0 - c) * (1.0 - k), (1.0 - m) * (1.0 - k), (1.0 - y) * (1.0 - k)]
}

// ---------------------------------------------------------------------------
// The public rasterize_page entry point
// ---------------------------------------------------------------------------

// MuPDF: fz_run_page + fz_new_draw_device onto a fresh fz_pixmap sized by the
// requested resolution (the pattern in the `mutool draw` tool / fz_new_pixmap).
/// Rasterize page `page_index` (0-based) of `doc` at `dpi` into a fresh white
/// DeviceRGB [`Pixmap`]. The scale is `dpi / 72`.
///
/// This is the public entry the terminal PDF viewer (`tdf-parity`) and the kmux
/// LaTeX live-preview call. It drives the existing content interpreter
/// ([`run_page`](super::run_page)) over a [`DrawDevice`]; today that emits glyph
/// boxes (see the fidelity-ceiling note on [`DrawDevice`]). Path/image operators
/// will paint once the interpreter forwards them.
pub fn rasterize_page(doc: &PdfDocument, page_index: usize, dpi: f32) -> super::error::Result<Pixmap> {
    let scale = (dpi / 72.0).max(0.01);
    let page = doc.page(page_index)?.clone();

    // Page size in points from the MediaBox, with a US-Letter fallback.
    let (mut w_pt, mut h_pt) = mediabox_size(doc, &page);
    // A 90/270 rotation swaps the device-space width/height (page_ctm applies the
    // rotation; we only need matching pixmap dimensions).
    if page_is_quarter_turned(doc, &page) {
        std::mem::swap(&mut w_pt, &mut h_pt);
    }

    let iw = (w_pt * scale).ceil().max(1.0) as u32;
    let ih = (h_pt * scale).ceil().max(1.0) as u32;

    let mut dev = DrawDevice::new(iw, ih, Matrix::scale(scale, scale));
    super::run_page(doc, page_index, &mut dev)?;
    Ok(dev.into_pixmap())
}

/// The MediaBox width/height in points (normalised), falling back to US Letter
/// (612×792) for a missing or degenerate box -- mirrors `page_ctm`'s guard.
fn mediabox_size(doc: &PdfDocument, page: &Object) -> (f32, f32) {
    let mb = doc.resolve_get(page, "MediaBox").unwrap_or(Object::Null);
    let v = |i: usize| -> f32 {
        mb.array_get(i)
            .and_then(|o| doc.resolve(o).ok())
            .map(|o| o.to_real() as f32)
            .unwrap_or(0.0)
    };
    if mb.array_len() >= 4 {
        let w = (v(0) - v(2)).abs();
        let h = (v(1) - v(3)).abs();
        if w >= 1.0 && h >= 1.0 {
            return (w, h);
        }
    }
    (612.0, 792.0)
}

/// True if the page's `/Rotate` (snapped to a multiple of 90) is 90° or 270°.
fn page_is_quarter_turned(doc: &PdfDocument, page: &Object) -> bool {
    let mut r = doc.resolve_get(page, "Rotate").map(|o| o.to_int()).unwrap_or(0);
    r = ((r % 360) + 360) % 360;
    r = 90 * ((r + 45) / 90);
    r %= 360;
    r == 90 || r == 270
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_path_black_rect_luma() {
        let mut dev = DrawDevice::new(40, 40, Matrix::IDENTITY);
        let mut p = Path::new();
        p.rect(10.0, 10.0, 30.0, 30.0);
        dev.fill_path(&p, FillRule::NonZero, Matrix::IDENTITY, [0.0, 0.0, 0.0], 1.0);
        let pix = dev.pixmap();
        assert!(pix.luma(20, 20).unwrap() < 5, "inside not black");
        assert_eq!(pix.luma(2, 2).unwrap(), 255, "margin not white");
    }

    #[test]
    fn base_scale_enlarges() {
        // base = 2x: a unit-ish rect fills 2x the pixels.
        let mut dev = DrawDevice::new(40, 40, Matrix::scale(2.0, 2.0));
        let mut p = Path::new();
        p.rect(0.0, 0.0, 10.0, 10.0);
        dev.fill_path(&p, FillRule::NonZero, Matrix::IDENTITY, [0.0, 0.0, 0.0], 1.0);
        // (10,10) path -> (20,20) device: still black inside.
        assert!(dev.pixmap().luma(15, 15).unwrap() < 5);
        assert!(dev.pixmap().luma(25, 25).unwrap() == 255);
    }

    #[test]
    fn cmyk_and_gray_conversions() {
        assert_eq!(gray_to_rgb(0.5), [0.5, 0.5, 0.5]);
        // Pure cyan -> (0,1,1).
        assert_eq!(cmyk_to_rgb(1.0, 0.0, 0.0, 0.0), [0.0, 1.0, 1.0]);
        // Pure black key -> (0,0,0).
        assert_eq!(cmyk_to_rgb(0.0, 0.0, 0.0, 1.0), [0.0, 0.0, 0.0]);
    }

    /// Build a one-page PDF from `bodies` (objects 1..) with a classic xref.
    fn build_pdf(bodies: &[&[u8]]) -> Vec<u8> {
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
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n").as_bytes(),
        );
        pdf
    }

    fn text_page_doc() -> PdfDocument {
        // MediaBox 200x200, a big word near the top-left. FirstChar..LastChar
        // spans E(69)..O(79) so every letter in "HELLO" has a 600-unit width.
        let font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
/Encoding /WinAnsiEncoding /FirstChar 69 /LastChar 79 \
/Widths [600 600 600 600 600 600 600 600 600 600 600] >>";
        let content = b"<< /Length 36 >>\nstream\nBT /F1 24 Tf 20 100 Td (HELLO) Tj ET\nendstream";
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>";
        let bodies: [&[u8]; 5] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            content,
            font,
        ];
        PdfDocument::open(build_pdf(&bodies)).unwrap()
    }

    #[test]
    fn rasterize_page_dims_scale_with_dpi() {
        let doc = text_page_doc();
        let p72 = rasterize_page(&doc, 0, 72.0).unwrap();
        assert_eq!((p72.w, p72.h), (200, 200), "72 dpi -> 1:1 with the 200pt MediaBox");
        let p144 = rasterize_page(&doc, 0, 144.0).unwrap();
        assert_eq!((p144.w, p144.h), (400, 400), "144 dpi -> 2x");
    }

    #[test]
    fn rasterize_page_has_content_and_clean_margins() {
        let doc = text_page_doc();
        let pix = rasterize_page(&doc, 0, 72.0).unwrap();

        // The page is not blank: some pixel is darkened by the text.
        let any_dark = (0..pix.h as i32).any(|y| (0..pix.w as i32).any(|x| pix.luma(x, y).unwrap() < 200));
        assert!(any_dark, "rasterized text page should not be all white");

        // The text sits near the top-left (device y ~ 78..100, x ~ 20..90). The
        // bottom-right quadrant is margin: it must stay white.
        for y in 150..200 {
            for x in 150..200 {
                assert_eq!(pix.luma(x, y).unwrap(), 255, "margin pixel ({x},{y}) not white");
            }
        }
    }

    #[test]
    fn draw_image_fills_unit_square() {
        let mut dev = DrawDevice::new(20, 20, Matrix::IDENTITY);
        // A 2x2 solid red image mapped onto a 10x10 device square at (5,5).
        let img = DecodedImage { width: 2, height: 2, components: 3, pixels: vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0] };
        let ctm = Matrix::new(10.0, 0.0, 0.0, 10.0, 5.0, 5.0);
        dev.draw_image(&img, ctm, 1.0);
        let px = dev.pixmap().pixel(10, 10).unwrap();
        assert_eq!(px, &[255, 0, 0], "image centre should be red");
        // Outside the mapped square stays white.
        assert_eq!(dev.pixmap().pixel(1, 1).unwrap(), &[255, 255, 255]);
    }
}
