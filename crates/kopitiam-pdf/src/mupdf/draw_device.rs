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
//! [`draw_image`](DrawDevice::draw_image) -- both as inherent methods (for the
//! viewer / direct callers) and as [`TextDevice`] callbacks the content
//! interpreter now drives for path/colour/image operators (see
//! [`interpret`](super::interpret) / [`op_run`](super::op_run)).
//!
//! Everything composes through the CTM (a [`Matrix`]) onto a device-space
//! [`Pixmap`]; the working colourspace is **DeviceRGB** (gray and CMYK inputs are
//! converted on the way in, matching MuPDF forcing the destination colourspace).
//!
//! ## The fidelity ceiling (what is stubbed -- read before trusting a render)
//!
//! * **Glyph shapes.** [`Font`] deliberately avoids FreeType (see
//!   [`font`](super::font)); instead pure-Rust outline decoders fill real
//!   letterforms for TrueType `glyf` ([`glyph_truetype`](super::glyph_truetype)),
//!   CFF / Type2 charstrings including the predefined-Standard-encoding
//!   fallback ([`glyph_cff`](super::glyph_cff)), and Type1 charstrings
//!   ([`glyph_type1`](super::glyph_type1)). [`show_glyph`](DrawDevice::show_glyph)
//!   falls back to a **filled advance box** only when [`Font::glyph_outline`]
//!   returns `None` -- a non-embedded font, an undecodable/predefined-Expert
//!   program, or a genuinely empty glyph (whitespace is skipped outright).
//! * **Invisible text.** ~~The glyph sink has no render-mode, so `Tr 3/7`
//!   (invisible, e.g. an OCR text layer over a scan) still paints a box.~~ --
//!   **CORRECTED 2026-09-28** (checked against `show_glyph`): the sink gets
//!   the mode through [`TextDevice::set_text_render_mode`] since 0.4.1 and
//!   paints only the fill modes 0/2/4/6; 1/2/5/6 are stroked through
//!   `stroke_glyph` (0.4.2). Clip modes 4..=7 do not clip yet.
//! * **Colour.** The content interpreter now emits colour operators (`g/rg/k`,
//!   `cs/sc/scn`), tracked in the graphics state and converted to DeviceRGB
//!   ([`resources`](super::resources)); fills/strokes and glyph boxes pick up the
//!   fill colour. Separation/DeviceN and Indexed spaces are approximated (see
//!   [`ColorSpace`](super::resources)); ICCBased maps by component count.
//! * **Not implemented at all** (safe no-ops / skips, never corruption): mesh &
//!   gradient shadings (`draw-mesh.c`), blend modes beyond Normal
//!   (`draw-blend.c`), soft masks, clip masks beyond a rectangular clip,
//!   knockout / transparency groups. ~~Image blitting is **nearest-neighbour**
//!   (no bilinear/mip smoothing)~~ -- **CORRECTED 2026-09-28 (0.4.2)**: images
//!   go through MuPDF's subsample + smooth-scale + near/lerp paint pipeline
//!   (`draw_scale`, `draw_affine`); they still honour only a rectangular clip.

use super::draw_affine;
use super::draw_edge::{FillRule, fill_polygons};
use super::draw_scale::{self, ScalePix};
use super::draw_path::Path;
use super::font::Font;
use super::geometry::{IRect, Matrix, Rect};
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
    /// ([`rasterize_page_native`]).
    base: Matrix,
    /// The rectangular clip (device pixels). Defaults to the whole pixmap.
    clip: IRect,
    /// The current fill colour (DeviceRGB, 0..=255). Defaults to black.
    fill: [u8; 3],
    /// How many glyphs on this page fell back to the solid advance box
    /// ([`show_glyph`](DrawDevice::show_glyph)'s fallback branch) because
    /// [`Font::glyph_outline`] returned `None`. `0` means every glyph got a
    /// real outline. See [`rasterize_page_ex`] / the `hayro_fallback` module
    /// for what this is used for.
    fallback_glyphs: usize,
    /// The current text render mode (`Tr`), pushed by the interpreter through
    /// [`TextDevice::set_text_render_mode`] before each glyph. Modes 3
    /// (invisible) and 7 (clip only) paint nothing -- see `show_glyph`.
    text_render_mode: i32,
    /// The current fill alpha for text (`/ca`), 0..=1.
    fill_alpha: f32,
}

impl DrawDevice {
    /// Create a device over a fresh white `w×h` DeviceRGB pixmap. `base` is the
    /// post-CTM device transform (use [`Matrix::IDENTITY`] for 1:1).
    pub fn new(w: u32, h: u32, base: Matrix) -> DrawDevice {
        let mut pix = Pixmap::new_rgb(w.max(1), h.max(1));
        pix.clear_with_value(0xff); // white page
        let clip = pix.bbox();
        DrawDevice {
            pix,
            base,
            clip,
            fill: [0, 0, 0],
            fallback_glyphs: 0,
            text_render_mode: 0,
            fill_alpha: 1.0,
        }
    }

    /// Create a device that draws **on top of an existing pixmap** instead of a
    /// fresh white page.
    ///
    /// This is what lets two engines contribute to one page. When a page hits
    /// the glyph fallback, [`super::hayro_fallback`] renders it with `hayro`
    /// and gets back a finished raster -- but hayro gates annotations behind
    /// `/AP` -> `/N`, so any annotation the file stores without an appearance
    /// stream is simply missing from it. Wrapping hayro's pixmap in a device
    /// lets the annotation pass paint those over the top, rather than the page
    /// silently losing its annotations whenever the fallback engages.
    ///
    /// `base` must be the same post-CTM device transform the pixmap was
    /// rendered under (`Matrix::scale(dpi / 72.0, dpi / 72.0)` for the
    /// rasterizer's own convention), or the overlay lands in the wrong place.
    pub fn over_pixmap(pix: Pixmap, base: Matrix) -> DrawDevice {
        let clip = pix.bbox();
        DrawDevice {
            pix,
            base,
            clip,
            fill: [0, 0, 0],
            fallback_glyphs: 0,
            text_render_mode: 0,
            fill_alpha: 1.0,
        }
    }

    /// Borrow the current pixmap.
    pub fn pixmap(&self) -> &Pixmap {
        &self.pix
    }

    /// How many glyphs painted so far fell back to the solid advance box
    /// instead of a real outline. See the `fallback_glyphs` field doc.
    pub fn fallback_glyph_count(&self) -> usize {
        self.fallback_glyphs
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
    /// transformed by `ctm`, using the winding rule `rule`. Clipped to the whole
    /// pixmap (use the [`TextDevice`] path for content-stream `W`/`W*` clipping).
    pub fn fill_path(
        &mut self,
        path: &Path,
        rule: FillRule,
        ctm: Matrix,
        color: [f32; 3],
        alpha: f32,
    ) {
        let clip = self.clip;
        self.fill_path_clipped(path, rule, ctm, color, alpha, clip);
    }

    /// [`fill_path`](DrawDevice::fill_path) with an explicit device-pixel clip.
    fn fill_path_clipped(
        &mut self,
        path: &Path,
        rule: FillRule,
        ctm: Matrix,
        color: [f32; 3],
        alpha: f32,
        clip: IRect,
    ) {
        let m = ctm.concat(self.base);
        let polys = path.flatten(m);
        let c = rgb_to_bytes(color);
        fill_polygons(&mut self.pix, &polys, rule, &c, alpha, clip);
    }

    // MuPDF: fz_draw_stroke_path (draw-device.c:766).
    /// Stroke `path` with `color` at `alpha`, `line_width` in *path* units,
    /// transformed by `ctm`. Uses the width-expansion approximation
    /// ([`Path::stroke_to_polygons`]).
    pub fn stroke_path(
        &mut self,
        path: &Path,
        ctm: Matrix,
        line_width: f32,
        color: [f32; 3],
        alpha: f32,
    ) {
        let clip = self.clip;
        self.stroke_path_clipped(path, ctm, line_width, color, alpha, clip);
    }

    /// [`stroke_path`](DrawDevice::stroke_path) with an explicit device-pixel clip.
    fn stroke_path_clipped(
        &mut self,
        path: &Path,
        ctm: Matrix,
        line_width: f32,
        color: [f32; 3],
        alpha: f32,
        clip: IRect,
    ) {
        let m = ctm.concat(self.base);
        // Convert the path-space width to device pixels via the CTM expansion.
        let dev_w = line_width * m.max_expansion();
        let polys = path.stroke_to_polygons(m, dev_w);
        let c = rgb_to_bytes(color);
        fill_polygons(&mut self.pix, &polys, FillRule::NonZero, &c, alpha, clip);
    }

    /// Resolve an optional content-space (pre-`base`) rectangular clip against the
    /// device's own pixel clip. `None` leaves the device clip unchanged. A non-rect
    /// clip is bbox-approximated upstream (see the interpreter's `W`/`W*`).
    fn resolve_clip(&self, clip: Option<Rect>) -> IRect {
        match clip {
            None => self.clip,
            // `base` is the device output transform (dpi scale); an axis-aligned
            // scale keeps the transformed rect axis-aligned, so its bbox is exact.
            Some(r) => self
                .clip
                .intersect(r.transform(self.base).irect_from_rect()),
        }
    }

    // MuPDF: fz_draw_fill_image (draw-device.c) -> fz_paint_image_imp ->
    // MuPDF: fz_draw_fill_image (draw-device.c:1836)
    /// Paint `img` at `alpha` under `ctm`, which maps the image's unit square
    /// `[0,1]²` (fitz image space, with `(0,0)` top-left) onto the page.
    ///
    /// Since 0.4.2 this follows MuPDF's pipeline instead of nearest-neighbour
    /// sampling every device pixel: grid-fit the matrix, box-subsample by a
    /// power of two (`l2factor`, [`draw_scale::subsample_pixmap`]), smooth-scale
    /// to the exact device footprint ([`draw_scale::transform_pixmap`]), then
    /// paint 1:1 -- bilinear only where MuPDF would ([`draw_affine::paint_image`]).
    pub fn draw_image(&mut self, img: &DecodedImage, ctm: Matrix, alpha: f32) {
        let clip = self.clip;
        self.draw_image_clipped(img, ctm, alpha, clip);
    }

    /// [`draw_image`](DrawDevice::draw_image) with an explicit device-pixel clip.
    fn draw_image_clipped(&mut self, img: &DecodedImage, ctm: Matrix, alpha: f32, clip: IRect) {
        if img.width == 0 || img.height == 0 || alpha <= 0.0 {
            return;
        }
        let clip = clip.intersect(self.pix.bbox());
        if clip.is_empty() {
            return;
        }
        // `local_ctm = fz_concat(in_ctm, dev->transform)`, then grid-fitted
        // (not a Type3 glyph device, so always).
        let local = draw_scale::gridfit_matrix(false, ctm.concat(self.base));

        let base = ScalePix {
            x: 0,
            y: 0,
            w: img.width as i32,
            h: img.height as i32,
            n: img.components as i32,
            alpha: false,
            samples: img.pixels.clone(),
        };
        let Some((pix, pix_ctm)) = prepare_image_pixmap(base, local, alpha, clip) else {
            return;
        };
        let pix = gray_to_rgb_pix(pix);

        // The /SMask goes through the same subsample + scale path as its own
        // image (MuPDF: fz_clip_image_mask -> fz_get_pixmap_mask_from_image),
        // then becomes a device-space coverage mask (see draw_affine's docs).
        let mask = img.smask.as_ref().filter(|m| m.width > 0 && m.height > 0).and_then(|m| {
            let mpix = ScalePix {
                x: 0,
                y: 0,
                w: m.width as i32,
                h: m.height as i32,
                n: 1,
                alpha: false,
                samples: m.alpha.clone(),
            };
            let (mpix, mctm) = prepare_image_pixmap(mpix, local, alpha, clip)?;
            let mpix = gray_to_rgb_pix(mpix);
            let mut layer = Pixmap::new_rgb((clip.x1 - clip.x0) as u32, (clip.y1 - clip.y0) as u32);
            layer.x = clip.x0;
            layer.y = clip.y0;
            draw_affine::paint_image(&mut layer, clip, &mpix, mctm, 255, true, false, None);
            let data = layer.samples.chunks(3).map(|p| p[0]).collect();
            Some(draw_affine::DevMask { bbox: clip, data })
        });
        if img.smask.is_some() && mask.is_none() {
            // A mask that scales to nothing covers nothing.
            return;
        }

        draw_affine::paint_image(
            &mut self.pix,
            clip,
            &pix,
            pix_ctm,
            (alpha * 255.0) as i32,
            true,
            false,
            mask.as_ref(),
        );
    }
}

// MuPDF: fz_get_pixmap_from_image (image.c:1019) + the scaling block of
// fz_draw_fill_image (draw-device.c:1912-1929).
/// Subsample `pix` by MuPDF's `l2factor` for the device footprint of `local`,
/// then (when shrinking) smooth-scale it. Returns the pixmap to paint and the
/// matrix that maps the unit square onto it, or `None` when nothing is left.
fn prepare_image_pixmap(mut pix: ScalePix, local: Matrix, alpha: f32, clip: IRect) -> Option<(ScalePix, Matrix)> {
    let (iw, ih) = (pix.w, pix.h);
    // "Figure out the extent" -- C `int w = sqrtf(...)`, i.e. truncation.
    let dx = (local.a * local.a + local.b * local.b).sqrt() as i32;
    let dy = (local.c * local.c + local.d * local.d).sqrt() as i32;
    // Clamp: we never magnify here.
    let w = dx.min(iw);
    let h = dy.min(ih);
    // The largest power-of-two shrink that stays >= the needed size (+2 for
    // grid-fit growth). FZ_IMAGE_RENDERING_BALANCE is MuPDF's default tuning.
    let mut l2factor = 0;
    if w > 0 && h > 0 {
        while (iw >> (l2factor + 1)) >= w + 2 && (ih >> (l2factor + 1)) >= h + 2 && l2factor < 6 {
            l2factor += 1;
        }
    }
    // MuPDF consumes l2factor inside the decoder for compressed non-indexed
    // images (subsample_stream -> fz_subsample_pixblock) and falls back to
    // fz_subsample_pixmap (Bresenham for ragged sizes) for everything else.
    // Our images arrive fully decoded, so the plain block subsample is the
    // decoder-path equivalent; see draw_scale's docs.
    if l2factor > 0 {
        let (nw, nh) = draw_scale::subsample_pixblock(&mut pix.samples, pix.w, pix.h, pix.n, l2factor);
        pix.w = nw;
        pix.h = nh;
        pix.samples.truncate((nw * nh * pix.n) as usize);
    }

    let mut m = local;
    // ctx->tuning->image_scale == fz_default_image_scale: only when shrinking.
    if dx < pix.w && dy < pix.h {
        let gridfit = alpha == 1.0;
        let scaled = match draw_scale::transform_pixmap(&pix, &mut m, dx, dy, gridfit, Some(clip)) {
            Some(s) => Some(s),
            None => draw_scale::scale_pixmap(&pix, pix.x as f32, pix.y as f32, dx.max(1) as f32, dy.max(1) as f32, None),
        };
        if let Some(s) = scaled {
            pix = s;
        }
    }
    Some((pix, m))
}

/// Gray (+alpha) -> RGB (+alpha) by replication, done AFTER scaling exactly
/// where MuPDF converts "images with fewer components" (and what its `g2rgb`
/// painter computes).
fn gray_to_rgb_pix(pix: ScalePix) -> ScalePix {
    let cn = pix.n - i32::from(pix.alpha);
    if cn != 1 {
        return pix;
    }
    let step = pix.n as usize;
    let mut out = Vec::with_capacity(pix.samples.len() / step * (3 + usize::from(pix.alpha)));
    for px in pix.samples.chunks(step) {
        out.extend_from_slice(&[px[0], px[0], px[0]]);
        if pix.alpha {
            out.push(px[1]);
        }
    }
    ScalePix { n: 3 + i32::from(pix.alpha), samples: out, ..pix }
}

// MuPDF: fz_draw_fill_text -> fz_draw_glyph (draw-device.c / draw-glyph.c). Here,
// with no outline source, reduced to the advance-box placeholder (see the module
// header's fidelity-ceiling note).
impl TextDevice for DrawDevice {
    fn show_glyph(
        &mut self,
        font: &Font,
        trm: Matrix,
        adv: f32,
        unicode: char,
        cid: u32,
        _wmode: u8,
    ) {
        // Whitespace and zero-width glyphs paint nothing (keeps inter-word gaps).
        if adv <= 0.0 || unicode.is_whitespace() {
            return;
        }

        // MuPDF pdf_flush_text_imp: mode 3 sets only `doinvisible`, mode 7 only
        // `doclip` -- neither fills nor strokes, so nothing reaches the pixels.
        // This is the OCR layer of every Acrobat Capture / ocrmypdf "searchable
        // image" scan: the words sit in `3 Tr` right over the scanned glyphs.
        // Painting them (as the port used to) doubled every letter with a
        // substituted face slightly off the scan -- or, for a font with no
        // substitute (BookAntiqua, CenturyGothic...), covered the scan with
        // solid advance boxes AND counted those boxes as fallback glyphs, which
        // kicked the whole page over to the hayro fallback for no reason.
        // Clip-text (4..=7) is not modelled by this device; 4..=6 still paint.
        //
        // More generally (0.4.2): only the FILL modes paint here -- 0, 2, 4, 6
        // (pdf_flush_text_imp's `dofill`). Mode 1 / 5 is stroke-only, drawn by
        // `stroke_glyph`; mode 2 / 6 gets both calls.
        if !matches!(self.text_render_mode, 0 | 2 | 4 | 6) {
            return;
        }

        let m = trm.concat(self.base);

        // Preferred path: fill the glyph's real outline from the embedded font
        // program (TrueType `glyf` / CFF Type2). The outline is in em space
        // (y-up, 1 em = 1.0), exactly the space `trm` maps -- the same space the
        // advance-box fallback below uses. `glyph_outline` returns `None` for a
        // non-embedded / undecodable font or an empty glyph, in which case we fall
        // back to the advance box (never a panic, never a solid block over a real
        // letterform).
        if let Some(path) = font.glyph_outline(cid) {
            let polys = path.flatten(m);
            // Nonzero winding: a glyph's counter (the hole in 'o'/'A') is wound
            // opposite its outer contour, so nonzero leaves it unfilled -- the
            // interior white that distinguishes a real letterform from the box.
            fill_polygons(
                &mut self.pix,
                &polys,
                FillRule::NonZero,
                &self.fill,
                self.fill_alpha,
                self.clip,
            );
            return;
        }

        // Fallback: no decodable outline, so paint the glyph's advance box -- a
        // legible block at the correct position/size, inset so adjacent glyphs
        // stay visually distinct and lines do not merge vertically. Lands here
        // for a non-embedded font, a predefined-Expert-encoding simple CFF (see
        // the glyph_cff module ceiling), or a genuinely undecodable program.
        self.fallback_glyphs += 1;
        let asc = font.ascender().min(0.9); // ~cap height mass
        let x0 = adv * 0.08;
        let x1 = adv * 0.92;
        let mut path = Path::new();
        path.rect(x0, 0.0, x1, asc);

        let polys = path.flatten(m);
        fill_polygons(
            &mut self.pix,
            &polys,
            FillRule::NonZero,
            &self.fill,
            self.fill_alpha,
            self.clip,
        );
    }

    // The interpreter's path/colour/image operators drive these (see
    // [`interpret`](super::interpret) / [`op_run`](super::op_run)); each forwards to
    // the inherent painter with the content-space clip resolved to device pixels.
    fn fill_path(
        &mut self,
        path: &Path,
        rule: FillRule,
        ctm: Matrix,
        color: [f32; 3],
        alpha: f32,
        clip: Option<Rect>,
    ) {
        let ic = self.resolve_clip(clip);
        self.fill_path_clipped(path, rule, ctm, color, alpha, ic);
    }

    fn stroke_path(
        &mut self,
        path: &Path,
        ctm: Matrix,
        line_width: f32,
        color: [f32; 3],
        alpha: f32,
        clip: Option<Rect>,
    ) {
        let ic = self.resolve_clip(clip);
        self.stroke_path_clipped(path, ctm, line_width, color, alpha, ic);
    }

    fn draw_image(&mut self, img: &DecodedImage, ctm: Matrix, alpha: f32, clip: Option<Rect>) {
        let ic = self.resolve_clip(clip);
        self.draw_image_clipped(img, ctm, alpha, ic);
    }

    fn set_fill_color(&mut self, color: [f32; 3]) {
        self.fill = rgb_to_bytes(color);
    }

    // MuPDF: fz_draw_fill_image_mask (draw-device.c:1957) -- the stencil goes
    // through the same l2factor subsample + smooth scale as any image
    // (fz_get_pixmap_mask_from_image), then fz_paint_image_with_color.
    fn draw_image_mask(&mut self, img: &DecodedImage, ctm: Matrix, color: [f32; 3], alpha: f32, clip: Option<Rect>) {
        if img.width == 0 || img.height == 0 || alpha <= 0.0 {
            return;
        }
        let clip = self.resolve_clip(clip).intersect(self.pix.bbox());
        if clip.is_empty() {
            return;
        }
        let local = draw_scale::gridfit_matrix(false, ctm.concat(self.base));
        // An image mask's decoded pixmap is alpha with 0 = opaque inverted to
        // 255 (image.c: "Invert 1-bit image masks"). Our decode kept the gray
        // convention (0 = paint), so invert here.
        let n = img.components.max(1) as usize;
        let samples = img.pixels.iter().step_by(n).map(|v| 255 - *v).collect();
        let mpix = ScalePix {
            x: 0,
            y: 0,
            w: img.width as i32,
            h: img.height as i32,
            n: 1,
            alpha: false,
            samples,
        };
        let Some((mpix, mctm)) = prepare_image_pixmap(mpix, local, alpha, clip) else {
            return;
        };
        draw_affine::paint_image_color(
            &mut self.pix,
            clip,
            &mpix,
            mctm,
            rgb_to_bytes(color),
            (alpha * 255.0) as i32,
            true,
            false,
        );
    }

    fn set_text_render_mode(&mut self, mode: i32) {
        self.text_render_mode = mode;
    }

    fn set_fill_alpha(&mut self, alpha: f32) {
        self.fill_alpha = alpha;
    }

    // MuPDF: fz_draw_stroke_path_aux (draw-device.c:746) with the whole
    // stroke state.
    fn stroke_path_styled(
        &mut self,
        path: &Path,
        ctm: Matrix,
        style: &super::draw_path::StrokeStyle,
        color: [f32; 3],
        alpha: f32,
        clip: Option<Rect>,
    ) {
        let ic = self.resolve_clip(clip);
        let m = ctm.concat(self.base);
        let polys = stroke_polygons(path, m, style);
        let c = rgb_to_bytes(color);
        fill_polygons(&mut self.pix, &polys, FillRule::NonZero, &c, alpha, ic);
    }

    fn stroke_glyph(
        &mut self,
        font: &Font,
        trm: Matrix,
        ctm: Matrix,
        cid: u32,
        style: &super::draw_path::StrokeStyle,
        color: [f32; 3],
        alpha: f32,
    ) {
        // MuPDF fz_draw_stroke_text -> fz_render_stroked_glyph: the glyph
        // outline goes through trm, the line width through the user CTM.
        let Some(outline) = font.glyph_outline(cid) else { return };
        let user = ctm.concat(self.base);
        // Bring the em-space outline into user space (trm = text-matrix·ctm,
        // so em->user is trm·ctm⁻¹), then stroke under the user CTM.
        let Some(ctm_inv) = ctm.try_invert() else { return };
        let em_to_user = trm.concat(ctm_inv);
        let user_path = outline.transformed(em_to_user);
        let polys = stroke_polygons(&user_path, user, style);
        let c = rgb_to_bytes(color);
        fill_polygons(&mut self.pix, &polys, FillRule::NonZero, &c, alpha, self.clip);
    }
}

// MuPDF: fz_draw_stroke_path_aux's width floor + fz_flatten_stroke_path's
// dash walk (draw-device.c:772-776, draw-path.c:1566).
/// Device-space stroke polygons for `path` under device matrix `m`.
fn stroke_polygons(path: &Path, m: Matrix, style: &super::draw_path::StrokeStyle) -> Vec<Vec<super::geometry::Point>> {
    let mut expansion = m.expansion();
    if expansion < f32::EPSILON {
        expansion = 1.0;
    }
    // aa_level = 2 / (8 + 2) at MuPDF's default 8-bit graphics AA.
    let aa_level = 0.2f32;
    let mut lw = style.line_width;
    if lw * expansion < aa_level {
        lw = aa_level / expansion;
    }
    let dev_w = lw * expansion;
    match path.dashed(&style.dash, style.dash_phase, m) {
        Some(d) => d.stroke_to_polygons_styled(m, dev_w, style.cap, style.join, style.miter_limit),
        None => path.stroke_to_polygons_styled(m, dev_w, style.cap, style.join, style.miter_limit),
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

// MuPDF: DeviceGray -> DeviceRGB (fz_convert_color): replicate the single value.
/// Convert a DeviceGray value (0..=1) to DeviceRGB.
pub fn gray_to_rgb(g: f32) -> [f32; 3] {
    [g, g, g]
}

// MuPDF: cmyk_to_rgb (color-fast.c:117) -- the conversion MuPDF uses when no
// ICC colour management applies.
/// Convert DeviceCMYK (each 0..=1) to DeviceRGB: `1 - min(1, c + k)` per
/// channel.
///
/// ~~`(1 - c)(1 - k)`, "the naive conversion fz_cmyk_to_rgb uses"~~ --
/// **CORRECTED 2026-09-28 (0.4.2)**: that product is not what MuPDF computes;
/// color-fast.c is additive-and-clamp, as above. The two agree when `k = 0`
/// and differ for mixed inks (c = m = y = k = 0.5 gives 0.25 before, 0 now).
/// Note that a default `mutool` build has lcms2 ICC management ON, which
/// converts through its default CMYK profile instead; this port has no CMS
/// (see docs/mupdf-port-coverage.md), so fast conversion is the parity target.
pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> [f32; 3] {
    [
        1.0 - (c + k).min(1.0),
        1.0 - (m + k).min(1.0),
        1.0 - (y + k).min(1.0),
    ]
}

// ---------------------------------------------------------------------------
// The rasterize_page entry points
// ---------------------------------------------------------------------------

// MuPDF: fz_run_page + fz_new_draw_device onto a fresh fz_pixmap sized by the
// requested resolution (the pattern in the `mutool draw` tool / fz_new_pixmap).
/// Rasterize page `page_index` (0-based) of `doc` at `dpi` into a fresh white
/// DeviceRGB [`Pixmap`], using **only** kopitiam-pdf's own engine -- no
/// cross-engine fallback. The scale is `dpi / 72`.
///
/// Most callers want the crate-level `rasterize_page`
/// ([`hayro_fallback::rasterize_page_graceful`](super::hayro_fallback::rasterize_page_graceful),
/// re-exported at `kopitiam_pdf::mupdf::rasterize_page`) instead, which wraps
/// this and transparently re-renders with `hayro` when this engine had to
/// fall back to an advance box. This native-only entry point still exists
/// for anyone who specifically wants kopitiam's own engine (e.g. comparing
/// the two, or a context where the extra dependency isn't wanted) and as the
/// building block `rasterize_page_ex` / the graceful wrapper are built on.
///
/// It drives the content interpreter ([`run_page`](super::run_page)) over a
/// [`DrawDevice`], which paints vector fills/strokes, colour, images and real
/// glyph letterforms wherever an embedded font program decodes (see the
/// fidelity-ceiling note on [`DrawDevice`] for the remaining stubs and the
/// advance-box fallback).
pub fn rasterize_page_native(
    doc: &PdfDocument,
    page_index: usize,
    dpi: f32,
) -> super::error::Result<Pixmap> {
    rasterize_page_ex(doc, page_index, dpi).map(|(pix, _fallback_glyphs)| pix)
}

/// [`rasterize_page_native`], additionally reporting how many glyphs on the
/// page fell back to the solid advance box
/// ([`DrawDevice::fallback_glyph_count`]) -- `0` means the page rendered with
/// real outlines throughout. This is the seam
/// [`hayro_fallback`](super::hayro_fallback) uses to decide whether a
/// second, cross-engine render is worth attempting.
pub fn rasterize_page_ex(
    doc: &PdfDocument,
    page_index: usize,
    dpi: f32,
) -> super::error::Result<(Pixmap, usize)> {
    let scale = (dpi / 72.0).max(0.01);
    let page = doc.page(page_index)?.clone();

    // MuPDF: fz_bound_page -> pdf_bound_page (pdf-page.c:666) -- the CROP box
    // (clipped to the MediaBox), rotated and /UserUnit-scaled. Before 0.4.2
    // this was the MediaBox, so a cover page whose MediaBox is a two-page
    // spread rendered at the spread's full width (IAEA TECDOC covers).
    let bounds = super::page_run::page_bounds(doc, &page);
    let (w_pt, h_pt) = (bounds.x1 - bounds.x0, bounds.y1 - bounds.y0);

    // The MediaBox is attacker-controlled; `dpi` is caller-controlled. Bound the
    // resulting raster so a pathological page (a huge MediaBox, optionally at a
    // high dpi) is rejected with an error instead of requesting a multi-GB `Vec`
    // and aborting the process on OOM. Checked in f64 first so the f32->u32 cast
    // below can never silently saturate. MAX_PIXELS ~ 100 MP (RGB => ~300 MB).
    const MAX_DIM: u32 = 30_000;
    const MAX_PIXELS: u64 = 100_000_000;
    let wf = (w_pt * scale).ceil() as f64;
    let hf = (h_pt * scale).ceil() as f64;
    if !(wf.is_finite() && hf.is_finite())
        || wf > MAX_DIM as f64
        || hf > MAX_DIM as f64
        || wf.max(0.0) * hf.max(0.0) > MAX_PIXELS as f64
    {
        return Err(super::error::Error::limit(format!(
            "page {page_index} rasterizes to {wf}x{hf}px at {dpi}dpi, over the \
             {MAX_DIM}px / {MAX_PIXELS}px limit"
        )));
    }
    let iw = (wf as u32).max(1);
    let ih = (hf as u32).max(1);

    let mut dev = DrawDevice::new(iw, ih, Matrix::scale(scale, scale));
    super::run_page(doc, page_index, &mut dev)?;

    // MuPDF: fz_run_page = pdf_run_page_contents + pdf_run_page_annots
    // (pdf-run.c:353). A page's /Annots is a *sibling* of /Contents, so the
    // interpreter run above draws none of them -- annotations need their own
    // pass, under the same base CTM, painted *after* the content because PDF
    // draws them on top.
    //
    // Errors are swallowed deliberately: annotations are decoration, and a
    // single malformed annot must never stop an otherwise readable page from
    // rendering. `run_page_annots` already contains per-annot error handling;
    // this is the outer belt-and-braces for a failure to even read /Annots.
    let base_ctm = super::page_run::page_ctm(doc, &page);
    let _ = super::annot_run::run_page_annots(doc, &page, base_ctm, &mut dev);

    let fallback_glyphs = dev.fallback_glyph_count();
    Ok((dev.into_pixmap(), fallback_glyphs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_path_black_rect_luma() {
        let mut dev = DrawDevice::new(40, 40, Matrix::IDENTITY);
        let mut p = Path::new();
        p.rect(10.0, 10.0, 30.0, 30.0);
        dev.fill_path(
            &p,
            FillRule::NonZero,
            Matrix::IDENTITY,
            [0.0, 0.0, 0.0],
            1.0,
        );
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
        dev.fill_path(
            &p,
            FillRule::NonZero,
            Matrix::IDENTITY,
            [0.0, 0.0, 0.0],
            1.0,
        );
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
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n")
                .as_bytes(),
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
        let p72 = rasterize_page_native(&doc, 0, 72.0).unwrap();
        assert_eq!(
            (p72.w, p72.h),
            (200, 200),
            "72 dpi -> 1:1 with the 200pt MediaBox"
        );
        let p144 = rasterize_page_native(&doc, 0, 144.0).unwrap();
        assert_eq!((p144.w, p144.h), (400, 400), "144 dpi -> 2x");
    }

    #[test]
    fn rasterize_page_rejects_pathological_mediabox() {
        // A hostile MediaBox must be rejected with an error, not allocate a
        // multi-GB pixmap and abort on OOM (pre-publish audit hardening).
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100000000 100000000] >>";
        let bodies: [&[u8]; 3] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
        ];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let err = rasterize_page_native(&doc, 0, 72.0).unwrap_err();
        assert!(
            err.message().contains("limit"),
            "expected a limit error, got: {}",
            err.message()
        );
        // A sane page at a sane dpi still succeeds (guard doesn't over-reject).
        assert!(rasterize_page_native(&text_page_doc(), 0, 300.0).is_ok());
    }

    #[test]
    fn rasterize_page_has_content_and_clean_margins() {
        let doc = text_page_doc();
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // The page is not blank: some pixel is darkened by the text.
        let any_dark =
            (0..pix.h as i32).any(|y| (0..pix.w as i32).any(|x| pix.luma(x, y).unwrap() < 200));
        assert!(any_dark, "rasterized text page should not be all white");

        // The text sits near the top-left (device y ~ 78..100, x ~ 20..90). The
        // bottom-right quadrant is margin: it must stay white.
        for y in 150..200 {
            for x in 150..200 {
                assert_eq!(
                    pix.luma(x, y).unwrap(),
                    255,
                    "margin pixel ({x},{y}) not white"
                );
            }
        }
    }

    #[test]
    fn draw_image_fills_unit_square() {
        let mut dev = DrawDevice::new(20, 20, Matrix::IDENTITY);
        // A 2x2 solid red image mapped onto a 10x10 device square at (5,5).
        let img = DecodedImage {
            smask: None,
            width: 2,
            height: 2,
            components: 3,
            pixels: vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
        };
        let ctm = Matrix::new(10.0, 0.0, 0.0, 10.0, 5.0, 5.0);
        dev.draw_image(&img, ctm, 1.0);
        let px = dev.pixmap().pixel(10, 10).unwrap();
        assert_eq!(px, &[255, 0, 0], "image centre should be red");
        // Outside the mapped square stays white.
        assert_eq!(dev.pixmap().pixel(1, 1).unwrap(), &[255, 255, 255]);
    }

    // -- end-to-end: content-stream operators -> painted pixels --------------

    /// Wrap content-stream operator bytes `ops` as a `/Length`-correct stream object.
    fn content_stream(ops: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(format!("<< /Length {} >>\nstream\n", ops.len()).as_bytes());
        v.extend_from_slice(ops);
        v.extend_from_slice(b"\nendstream");
        v
    }

    /// A one-page (100x100) PDF whose content is `ops`, with an optional page
    /// `/Resources` body and extra objects (numbered from 5).
    fn op_doc(ops: &[u8], resources: &str, extra: &[&[u8]]) -> PdfDocument {
        let content = content_stream(ops);
        let page = format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources {resources} /Contents 4 0 R >>"
        );
        let mut bodies: Vec<&[u8]> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page.as_bytes(),
            &content,
        ];
        bodies.extend_from_slice(extra);
        PdfDocument::open(build_pdf(&bodies)).unwrap()
    }

    #[test]
    fn rasterize_fills_red_rectangle_via_interpreter() {
        // `1 0 0 rg` -> red fill; `re f` fills the path-space (20,20)..(80,80) box.
        let doc = op_doc(b"1 0 0 rg 20 20 60 60 re f", "<< >>", &[]);
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // Inside the rectangle: red (high R, low G/B) -- not the old default black.
        let c = pix.pixel(50, 50).unwrap();
        assert!(
            c[0] > 200 && c[1] < 60 && c[2] < 60,
            "centre should be red, got {c:?}"
        );
        // Outside the rectangle: still white.
        assert_eq!(
            pix.pixel(5, 5).unwrap(),
            &[255, 255, 255],
            "margin not white"
        );
    }

    #[test]
    fn colour_operator_changes_fill_from_black() {
        // The same geometry filled black (default) vs green must differ.
        let black = op_doc(b"20 20 60 60 re f", "<< >>", &[]);
        let green = op_doc(b"0 1 0 rg 20 20 60 60 re f", "<< >>", &[]);
        let pb = rasterize_page_native(&black, 0, 72.0).unwrap();
        let pg = rasterize_page_native(&green, 0, 72.0).unwrap();

        let cb = pb.pixel(50, 50).unwrap();
        let cg = pg.pixel(50, 50).unwrap();
        assert!(
            cb.iter().all(|&v| v < 40),
            "default fill should be black, got {cb:?}"
        );
        assert!(
            cg[1] > 200 && cg[0] < 60 && cg[2] < 60,
            "rg fill should be green, got {cg:?}"
        );
        assert_ne!(cb, cg, "colour operator must change the output colour");
    }

    #[test]
    fn rasterize_strokes_a_line() {
        // A 2pt-wide horizontal line at path y=50 (device y=50 after the flip).
        let doc = op_doc(b"2 w 10 50 m 90 50 l S", "<< >>", &[]);
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // On the line: dark pixels.
        assert!(
            pix.luma(50, 50).unwrap() < 150,
            "stroke should darken the line"
        );
        assert!(
            pix.luma(30, 50).unwrap() < 150,
            "stroke should darken along the line"
        );
        // Well away from the line: white.
        assert_eq!(
            pix.luma(50, 10).unwrap(),
            255,
            "off-line pixel should be white"
        );
    }

    #[test]
    fn nonzero_vs_even_odd_differ_end_to_end() {
        // A self-intersecting pentagram (5 vertices connected star-wise). Under the
        // nonzero rule its central pentagon is filled (winding 2); under even-odd it
        // is a hole. Routed through the interpreter's `f` vs `f*`.
        let star = b"50 85 m 29.4 21.7 l 83.3 60.8 l 16.7 60.8 l 70.6 21.7 l ";
        let mut nz_ops = star.to_vec();
        nz_ops.extend_from_slice(b"f");
        let mut eo_ops = star.to_vec();
        eo_ops.extend_from_slice(b"f*");

        let nz = rasterize_page_native(&op_doc(&nz_ops, "<< >>", &[]), 0, 72.0).unwrap();
        let eo = rasterize_page_native(&op_doc(&eo_ops, "<< >>", &[]), 0, 72.0).unwrap();

        // Centre pentagon: filled under nonzero, a hole under even-odd.
        assert!(
            nz.luma(50, 50).unwrap() < 100,
            "nonzero centre should be filled"
        );
        assert!(
            eo.luma(50, 50).unwrap() > 200,
            "even-odd centre should be a hole"
        );
        // Even-odd still paints the five spikes -- it is not blank.
        let eo_has_dark = (0..100).any(|y| (0..100).any(|x| eo.luma(x, y).unwrap() < 100));
        assert!(eo_has_dark, "even-odd star should still paint its spikes");
    }

    #[test]
    fn rasterize_paints_image_xobject_via_do() {
        // A 4x4 solid-red raw DeviceRGB image, placed by `cm` on a 60x60 square at
        // user (20,20) and painted with `Do`.
        let pixels: Vec<u8> = std::iter::repeat_n([255u8, 0, 0], 16).flatten().collect();
        let mut img: Vec<u8> = Vec::new();
        img.extend_from_slice(
            format!(
                "<< /Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB \
/BitsPerComponent 8 /Length {} >>\nstream\n",
                pixels.len()
            )
            .as_bytes(),
        );
        img.extend_from_slice(&pixels);
        img.extend_from_slice(b"\nendstream");

        let doc = op_doc(
            b"q 60 0 0 60 20 20 cm /Im0 Do Q",
            "<< /XObject << /Im0 5 0 R >> >>",
            &[&img],
        );
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // The image covers device (20,20)..(80,80): its centre is red.
        let c = pix.pixel(50, 50).unwrap();
        assert!(
            c[0] > 200 && c[1] < 60 && c[2] < 60,
            "image centre should be red, got {c:?}"
        );
        // Outside the placement: white.
        assert_eq!(
            pix.pixel(5, 5).unwrap(),
            &[255, 255, 255],
            "outside image not white"
        );
    }

    #[test]
    fn clip_confines_fill_via_interpreter() {
        // A rectangular clip (`re W n`) to (40,40)..(60,60), then a full-page red
        // fill: only the clipped window is painted.
        let doc = op_doc(
            b"40 40 20 20 re W n 1 0 0 rg 0 0 100 100 re f",
            "<< >>",
            &[],
        );
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // Inside the clip window: red.
        let c = pix.pixel(50, 50).unwrap();
        assert!(
            c[0] > 200 && c[1] < 60 && c[2] < 60,
            "clip window should be red, got {c:?}"
        );
        // Outside the clip window (but inside the fill rect): unpainted white.
        assert_eq!(
            pix.pixel(20, 20).unwrap(),
            &[255, 255, 255],
            "clip should mask the fill"
        );
    }

    // -- end-to-end: embedded-font glyph OUTLINES (not the advance box) --------

    /// Wrap raw font-program bytes as a `/FontFile2` stream object.
    fn fontfile2_body(ttf: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(
            format!(
                "<< /Length {} /Length1 {} >>\nstream\n",
                ttf.len(),
                ttf.len()
            )
            .as_bytes(),
        );
        v.extend_from_slice(ttf);
        v.extend_from_slice(b"\nendstream");
        v
    }

    #[test]
    fn embedded_truetype_renders_letterform_with_interior_white() {
        // A simple TrueType font whose one glyph (code 0x41) is a ring: an outer
        // square with an oppositely-wound inner square, so a correct nonzero fill
        // leaves the centre a HOLE. This is the acceptance discriminator: the
        // advance-box fallback would fill the centre solid; a real outline does not.
        let ttf = super::super::glyph_truetype::ring_font();
        let font = b"<< /Type /Font /Subtype /TrueType /BaseFont /Ring \
/FirstChar 65 /LastChar 65 /Widths [1000] /FontDescriptor 6 0 R >>";
        let descriptor = b"<< /Type /FontDescriptor /FontName /Ring /Flags 4 /FontFile2 7 0 R >>";
        let content = b"<< /Length 32 >>\nstream\nBT /F1 150 Tf 20 30 Td (A) Tj ET\nendstream";
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>";
        let ff = fontfile2_body(&ttf);
        let bodies: [&[u8]; 7] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            content,
            font,
            descriptor,
            &ff,
        ];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // The glyph outer ring spans device ~(35,35)-(155,155); its counter (hole)
        // spans device ~(65,65)-(125,125). Centre of the hole must stay WHITE --
        // the real letterform's interior, impossible under the solid advance box.
        assert!(
            pix.luma(95, 95).unwrap() > 200,
            "glyph counter should be white (interior), got {}",
            pix.luma(95, 95).unwrap()
        );
        // The ring wall (left of the hole) must be painted dark.
        assert!(
            pix.luma(40, 95).unwrap() < 120,
            "ring wall should be dark, got {}",
            pix.luma(40, 95).unwrap()
        );
        // Sanity: the page is not blank.
        let any_dark = (0..200).any(|y| (0..200).any(|x| pix.luma(x, y).unwrap() < 100));
        assert!(any_dark, "glyph should paint some dark pixels");
    }

    /// Wrap raw font-program bytes as a `/FontFile` (Type1) stream object, with
    /// no `/Length1`/`/Length2` so the decoder exercises its `eexec`-keyword
    /// search fallback rather than the length-hint fast path.
    fn fontfile_body(pfa: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(format!("<< /Length {} >>\nstream\n", pfa.len()).as_bytes());
        v.extend_from_slice(pfa);
        v.extend_from_slice(b"\nendstream");
        v
    }

    #[test]
    fn embedded_type1_renders_letterform_with_interior_white() {
        // Issue #31: Type1 (`/FontFile`) glyphs used to render as solid filled
        // boxes. Same ring-glyph discriminator as the TrueType test above: a
        // real outline leaves the ring's centre a white hole; the old
        // advance-box fallback would fill it solid. The font has no PDF-level
        // /Encoding, so code 0x41 resolves through the Type1 program's own
        // built-in StandardEncoding (the common symbolic-Type1 case).
        let pfa = super::super::glyph_type1::ring_type1_pfa();
        let font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Ring \
/FirstChar 65 /LastChar 65 /Widths [1000] /FontDescriptor 6 0 R >>";
        let descriptor = b"<< /Type /FontDescriptor /FontName /Ring /Flags 4 /FontFile 7 0 R >>";
        let content = b"<< /Length 32 >>\nstream\nBT /F1 150 Tf 20 30 Td (A) Tj ET\nendstream";
        let page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>";
        let ff = fontfile_body(&pfa);
        let bodies: [&[u8]; 7] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            content,
            font,
            descriptor,
            &ff,
        ];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();

        // Same device geometry as the TrueType ring test (identical Tf/Td and
        // em-space square coordinates).
        assert!(
            pix.luma(95, 95).unwrap() > 200,
            "glyph counter should be white (interior), got {}",
            pix.luma(95, 95).unwrap()
        );
        assert!(
            pix.luma(40, 95).unwrap() < 120,
            "ring wall should be dark, got {}",
            pix.luma(40, 95).unwrap()
        );
        let any_dark = (0..200).any(|y| (0..200).any(|x| pix.luma(x, y).unwrap() < 100));
        assert!(any_dark, "glyph should paint some dark pixels");
    }

    #[test]
    fn non_embedded_font_falls_back_to_filled_box() {
        // A standard-14 font with NO embedded program: show_glyph must fall back to
        // the advance box (a solid block), never panic. The centre is filled.
        let doc = text_page_doc(); // Helvetica, no FontFile
        let pix = rasterize_page_native(&doc, 0, 72.0).unwrap();
        // "HELLO" at size 24 near the top-left paints solid blocks (no outlines).
        let any_dark =
            (0..pix.h as i32).any(|y| (0..pix.w as i32).any(|x| pix.luma(x, y).unwrap() < 100));
        assert!(any_dark, "advance-box fallback should still paint text");
    }
}
