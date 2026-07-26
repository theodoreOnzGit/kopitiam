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

use super::font::Font;
use super::geometry::Matrix;

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
    fn show_glyph(&mut self, font: &Font, trm: Matrix, adv: f32, unicode: char, cid: u32, wmode: u8);

    // MuPDF: fz_fill_image / op_Do_image (pdf-op-run.c). Not on the text path.
    /// Emit an image (an XObject `/Image` or inline `BI` image). Images are **not**
    /// on the text-extraction path, so the interpreter never calls this yet and the
    /// default is a no-op; the hook exists so a later (image) wave can extend the
    /// same sink without breaking existing implementors.
    fn fill_image(&mut self) {}
}
