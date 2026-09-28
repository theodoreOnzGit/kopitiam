//! The text-device seam: a collapsed stand-in for MuPDF's two interpretation
//! vtables. Modelled on `fz_device`'s glyph sink (`fz_show_glyph_aux` in
//! `source/pdf/pdf-op-run.c`, and `include/mupdf/fitz/device.h`'s `fill_text` /
//! the structured-text device `source/fitz/stext-device.c`) fused with the
//! `pdf_processor` text-showing operators (`source/pdf/pdf-interpret.c`)
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Two vtables collapsed into one
//!
//! MuPDF's run processor is a **`pdf_processor`** (one function pointer per PDF
//! content operator: `op_Tj`, `op_TJ`, `op_Tf`, `op_cm`, …) that, for the
//! text-showing operators, accumulates glyphs into an `fz_text` object and then
//! *flushes* it to an **`fz_device`** via `dev->fill_text` / `dev->clip_text`.
//! The device (for extraction, the `stext` device) walks that `fz_text` and, for
//! each glyph item, recovers the per-glyph text-rendering matrix by composing the
//! item's stored matrix with the fill `ctm`.
//!
//! For text **extraction** the intermediate `fz_text` buffer, the render-mode
//! flush boundaries, and the device/processor split add nothing: every glyph is
//! wanted, positioned, in device space. So this port collapses the two vtables
//! into a single sink. The interpreter ([`super::interpret`] /
//! [`super::op_run`]) computes each glyph's **device-space** text-rendering
//! matrix directly (`trm = [size·Tz, 0, 0, size, 0, rise] · Tm · CTM`, i.e.
//! MuPDF's `pdf_tos_make_trm` result already post-multiplied by the fill `ctm`
//! the `stext` device would have applied) and calls [`TextDevice::show_glyph`]
//! straight away. There is no buffered `fz_text`, no `q`/`Q`-style device stack.
//!
//! The next wave's structured-text builder implements this trait; nothing else
//! consumes it. Paths, colours, shadings, clips, blends and images are **not**
//! on the text-extraction path and are parsed-and-ignored by the interpreter, so
//! this trait exposes only the glyph sink (plus an image hook kept as a
//! documented no-op default for a later wave).

use super::draw_edge::FillRule;
use super::draw_path::Path;
use super::font::Font;
use super::geometry::{Matrix, Rect};
use super::page_image::DecodedImage;

/// The sink the content-stream interpreter emits positioned glyphs to.
///
/// This is the WAVE-5 seam the structured-text (`stext`) device will implement.
/// It fuses MuPDF's `pdf_processor` text operators with the `fz_device`
/// `fill_text`/`clip_text` glyph callbacks (see the module docs for why the two
/// vtables are collapsed).
pub trait TextDevice {
    // MuPDF: fz_show_glyph_aux (pdf-op-run.c:1441) feeding fz_fill_text ->
    // the stext device's fz_stext_fill_text glyph loop (stext-device.c).
    /// Emit one positioned glyph.
    ///
    /// * `font` -- the loaded font the glyph is drawn from.
    /// * `trm` -- the glyph's **device-space** text-rendering matrix:
    ///   `[size·Tz, 0, 0, size, 0, rise] · Tm · CTM`. Its `(e, f)` are the
    ///   glyph origin in device space; its `a`/`d` carry `size` (× horizontal
    ///   scale on `a`), so the on-page glyph box is `trm` applied to the font's
    ///   1-em glyph box.
    /// * `adv` -- the glyph's nominal advance width in **em** units (the PDF
    ///   `/Widths` value ÷ 1000), i.e. MuPDF's `w0`. Multiply by the font size
    ///   for the text-space advance. (The full inter-glyph step, including
    ///   `Tc`/`Tw`/`Tz`, is applied to `Tm` by the interpreter; this is the raw
    ///   per-glyph width the layout analyser needs to tell a real space from mere
    ///   advance.)
    /// * `unicode` -- the code's Unicode value (U+FFFD on a total miss).
    /// * `cid` -- the code's CID (the glyph identifier within the font).
    /// * `wmode` -- writing mode: 0 = horizontal, 1 = vertical.
    fn show_glyph(
        &mut self,
        font: &Font,
        trm: Matrix,
        adv: f32,
        unicode: char,
        cid: u32,
        wmode: u8,
    );

    // MuPDF: the fz_device fill_path callback (fz_fill_path -> dev->fill_path,
    // device.h). Wired by the content interpreter's path-painting operators.
    /// Fill `path` (path space) with `color` (DeviceRGB 0..=1) at `alpha`, under
    /// `ctm`, using winding rule `rule`. `clip` is an optional device-space (before
    /// any device output transform) rectangular clip from `W`/`W*`. Default no-op:
    /// the extraction sinks ignore vector fills.
    fn fill_path(
        &mut self,
        _path: &Path,
        _rule: FillRule,
        _ctm: Matrix,
        _color: [f32; 3],
        _alpha: f32,
        _clip: Option<Rect>,
    ) {
    }

    // MuPDF: the fz_device stroke_path callback (fz_stroke_path -> dev->stroke_path).
    /// Stroke `path` with `color` at `alpha`, `line_width` in *path* units, under
    /// `ctm`, honouring an optional rectangular `clip`. Default no-op.
    fn stroke_path(
        &mut self,
        _path: &Path,
        _ctm: Matrix,
        _line_width: f32,
        _color: [f32; 3],
        _alpha: f32,
        _clip: Option<Rect>,
    ) {
    }

    // MuPDF: the fz_device fill_image callback (fz_fill_image -> dev->fill_image).
    /// Paint the decoded image `img` under `ctm` (which maps the fitz image unit
    /// square onto the page) at `alpha`, honouring an optional rectangular `clip`.
    /// Default no-op: extraction sinks handle images out of band.
    fn draw_image(&mut self, _img: &DecodedImage, _ctm: Matrix, _alpha: f32, _clip: Option<Rect>) {}

    // MuPDF: the fz_device fill_image_mask callback (fz_fill_image_mask,
    // pdf-op-run.c:883) -- a stencil `/ImageMask` painted in the fill colour.
    /// Paint `color` (DeviceRGB 0..=1) at `alpha` through the stencil `img`
    /// (as decoded: 1 component, **0 = paint**, 255 = leave alone -- the
    /// `/ImageMask` sample convention after `/Decode`), under `ctm`, clipped
    /// to `clip`. Default no-op: extraction sinks ignore images.
    fn draw_image_mask(
        &mut self,
        _img: &DecodedImage,
        _ctm: Matrix,
        _color: [f32; 3],
        _alpha: f32,
        _clip: Option<Rect>,
    ) {
    }

    // MuPDF: the fill material of pdf_gstate carried into fz_fill_text.
    /// Set the current fill colour (DeviceRGB 0..=1). Used so the placeholder glyph
    /// boxes pick up the content stream's fill colour; extraction sinks ignore it.
    fn set_fill_color(&mut self, _color: [f32; 3]) {}

    // MuPDF: the fill alpha of pdf_gstate carried into fz_fill_text
    // (`gstate->fill.alpha`, set by an ExtGState `/ca`).
    /// Set the current fill alpha (0..=1) for the glyphs that follow. Called
    /// right after [`set_fill_color`](TextDevice::set_fill_color); extraction
    /// sinks ignore it.
    fn set_fill_alpha(&mut self, _alpha: f32) {}

    // MuPDF: fz_stroke_text (pdf_flush_text_imp's `dostroke`, render modes
    // 1 / 2 / 5 / 6).
    /// Stroke one glyph's outline: `trm` is its device-space text-rendering
    /// matrix (as for [`show_glyph`](TextDevice::show_glyph)), `ctm` the user
    /// CTM the line width is measured in. Default no-op: extraction already
    /// got the glyph from `show_glyph`.
    #[allow(clippy::too_many_arguments)]
    fn stroke_glyph(
        &mut self,
        _font: &Font,
        _trm: Matrix,
        _ctm: Matrix,
        _cid: u32,
        _style: &super::draw_path::StrokeStyle,
        _color: [f32; 3],
        _alpha: f32,
    ) {
    }

    // MuPDF: dev->stroke_path with the whole fz_stroke_state.
    /// Stroke `path` with the full line style (width, caps, join, miter limit,
    /// dash). The default forwards to [`stroke_path`](TextDevice::stroke_path)
    /// with just the width, so a sink that only cares about geometry keeps
    /// working unchanged; the draw device overrides it.
    fn stroke_path_styled(
        &mut self,
        path: &Path,
        ctm: Matrix,
        style: &super::draw_path::StrokeStyle,
        color: [f32; 3],
        alpha: f32,
        clip: Option<Rect>,
    ) {
        self.stroke_path(path, ctm, style.line_width, color, alpha, clip);
    }

    // MuPDF: pdf_flush_text_imp (pdf-op-run.c:1121) -- the `tos.text_mode`
    // switch that picks dofill / dostroke / doclip / doinvisible per text run.
    /// Set the text render mode (`Tr`, PDF 32000-1:2008 §9.3.6, Table 106) that
    /// applies to the glyphs emitted after this call. The interpreter calls it
    /// just before every [`show_glyph`](TextDevice::show_glyph), same as
    /// [`set_fill_color`](TextDevice::set_fill_color).
    ///
    /// Why a separate hook and not a `show_glyph` argument: only a *painting*
    /// device cares. Mode 3 ("neither fill nor stroke") and mode 7 ("add to
    /// clip only") are **invisible** -- MuPDF hands them to the device as
    /// `fz_ignore_text` / clip text, never `fz_fill_text` -- but an
    /// **extraction** sink still wants every one of those glyphs, because that
    /// invisible layer is exactly the OCR text of a scanned "searchable image"
    /// PDF. So extraction sinks keep this default no-op and still see all the
    /// glyphs; the draw device overrides it and paints nothing for 3 and 7.
    fn set_text_render_mode(&mut self, _mode: i32) {}

    // MuPDF: the `gid = -1` glyphs pdf_show_char adds "for one-to-many unicode
    // mapping" (pdf-op-run.c:1449), which fz_stext_extract turns into chars
    // with zero advance (stext-device.c:1191) and the draw device never paints.
    /// A zero-advance **filler** char at the same `trm` as the glyph just shown:
    /// the second and later code points of a one-to-many `/ToUnicode` entry
    /// (the `i` of an "fi" ligature, say). There is no glyph to draw -- the real
    /// glyph already went through [`show_glyph`](TextDevice::show_glyph) -- so a
    /// painting device keeps this default no-op; an extraction sink adds the
    /// char so "fi" does not come out as "f".
    fn show_filler_char(&mut self, _font: &Font, _trm: Matrix, _unicode: char, _wmode: u8) {}
}
