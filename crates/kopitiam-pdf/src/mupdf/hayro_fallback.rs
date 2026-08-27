//! Cross-engine graceful fallback for [`super::draw_device::rasterize_page_native`],
//! folded directly into the crate's public `rasterize_page` (see `mod.rs`'s
//! re-export).
//!
//! kopitiam-pdf's own glyph decoders (`glyph_truetype.rs` / `glyph_cff.rs` /
//! `glyph_type1.rs`) are from-spec, self-contained, and cover the large
//! majority of embedded fonts -- but they have documented ceilings
//! (predefined-Expert CFF encoding, an as-yet-unconfirmed CID-keyed-CFF or
//! `seac` edge case, see GitHub issue #67) where [`DrawDevice::show_glyph`]
//! falls back to painting a solid advance box instead of a real letterform.
//! That is the honest, never-a-crash behaviour -- but it is not what a reader
//! wants to see.
//!
//! [`rasterize_page_graceful`] upgrades that: render with kopitiam's own
//! engine first (the common case, and the fast path -- most pages never
//! touch the fallback at all), and only when that run actually hit the box
//! fallback for at least one glyph, re-render the *same* page with
//! [`hayro`](https://github.com/LaurenzV/hayro) (Apache-2.0 OR MIT, pure
//! Rust) -- "currently the most comprehensive and feature-complete
//! implementation of a PDF rasterizer in pure Rust" per its own docs, with
//! over 1000 PDFs in its own regression suite -- and return *that* pixmap
//! instead. A real letterform from a different engine beats a page of
//! blue/black boxes from ours. If hayro can't parse the page either, this
//! keeps kopitiam's own (box-fallback) output rather than erroring: never
//! worse than what
//! [`rasterize_page_native`](super::draw_device::rasterize_page_native)
//! already gave.
//!
//! # Why this is a whole extra render, not a per-glyph patch
//!
//! hayro's own public surface is page-level ([`hayro::render`] drives its
//! `Device` trait over an entire page), not a callable "decode this one
//! glyph" API -- its font decoders are `pub(crate)`-private even to crates
//! that depend on `hayro-interpret` directly. So the achievable granularity
//! here is "re-render the whole page," not "patch in just the failing
//! glyphs." For a page that hit the fallback at all, that is still a strict
//! improvement.
//!
//! # Why this is mandatory, not a Cargo feature
//!
//! Per the maintainer's explicit instruction, `hayro` is a plain, unconditional
//! dependency (see the Cargo.toml comment next to it) and this fallback is
//! folded into the crate's default `rasterize_page` name -- every existing
//! caller across the workspace (`apps/cli`, the `kpdf` example, and anything
//! written against kopitiam-pdf in the future) gets the graceful behaviour
//! automatically, with no opt-in and no call-site changes. A consumer that
//! only extracts text and never calls `rasterize_page` still pays hayro's
//! build cost in its dependency graph, but not its runtime cost -- the whole
//! point of the fast path above is that hayro is never invoked unless a page
//! actually hit the box fallback.
//!
//! # Provenance
//!
//! `hayro` is depended on directly (a real Cargo dependency, not a
//! translation) per `CLAUDE.md`'s Pure Rust Core hard rule: prefer an
//! existing, actively-maintained pure-Rust crate over writing a new
//! implementation, when one exists with a genuinely usable public API. Its
//! own `render`/`RenderCache`/`RenderSettings`/`InterpreterSettings` types
//! are used as documented; no code is copied from it. See
//! `docs/ACKNOWLEDGEMENTS.md` and `docs/ai-decisions/AID-0055.md` (which
//! covers the earlier, narrower `hayro-font` question this supersedes).

use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{RenderCache, RenderSettings, render};

use super::draw_device::rasterize_page_ex;
use super::error::Result;
use super::pixmap::Pixmap;
use super::xref::PdfDocument;

/// Rasterize page `page_index` (0-based) of `doc` at `dpi`, the same contract
/// as [`rasterize_page_native`](super::draw_device::rasterize_page_native),
/// except that a page which would otherwise show one or more advance-box
/// glyphs is transparently re-rendered with `hayro` instead. hayro parses the
/// PDF independently via its own `hayro-syntax` crate (it can't share
/// kopitiam's already-open [`PdfDocument`]), so this reaches for
/// [`PdfDocument::raw_bytes`] to hand it the same file.
pub fn rasterize_page_graceful(doc: &PdfDocument, page_index: usize, dpi: f32) -> Result<Pixmap> {
    rasterize_page_with_fallback(doc, page_index, dpi, true)
}

/// [`rasterize_page_graceful`], with the cross-engine fallback switchable at
/// runtime.
///
/// `fallback = true` is the normal behaviour and what plain `rasterize_page`
/// does. `fallback = false` pins the render to kopitiam's own engine, so a page
/// that would have been re-rendered by `hayro` instead shows this engine's real
/// output -- advance boxes and all.
///
/// # Why a viewer wants this switch
///
/// The fallback is deliberately invisible: it swaps engines mid-document
/// whenever a glyph fails to decode, which is exactly right for *reading* and
/// exactly wrong for *diagnosing*. Someone looking at a suspicious page cannot
/// otherwise tell "kopitiam rendered this correctly" from "kopitiam gave up and
/// hayro rendered it", and the two have real behavioural differences -- most
/// sharply, hayro draws no annotation that lacks an `/AP`
/// (`hayro-interpret/src/interpret/mod.rs:157`), which is why
/// [`overlay_missing_annots`] exists at all.
///
/// So this is a debugging and bug-reporting affordance first: flip it off, and
/// what you see is unambiguously ours. It is also the honest answer for a user
/// who would rather see our imperfect glyphs consistently than have the page
/// silently change engine underneath them.
pub fn rasterize_page_with_fallback(
    doc: &PdfDocument,
    page_index: usize,
    dpi: f32,
    fallback: bool,
) -> Result<Pixmap> {
    let (pix, fallback_glyphs) = rasterize_page_ex(doc, page_index, dpi)?;
    if fallback_glyphs == 0 || !fallback {
        return Ok(pix);
    }
    match render_with_hayro(doc.raw_bytes(), page_index, dpi) {
        Some(hayro_pix) => Ok(overlay_missing_annots(doc, page_index, dpi, hayro_pix)),
        None => Ok(pix),
    }
}

/// Paint onto `hayro_pix` the annotations `hayro` did not draw.
///
/// hayro gates every annotation behind `/AP` -> `/N`
/// (`hayro-interpret/src/interpret/mod.rs:157`), so a file that stores an
/// annotation as pure data -- `/InkList`, `/C`, `/Border`, no appearance
/// stream, which is exactly what Okular writes -- comes back from hayro with
/// that annotation missing. Without this step, taking the fallback for a
/// *glyph* problem would silently cost the page its *annotations*: the reader
/// would watch ink disappear on precisely the pages where the fallback
/// engaged, which is worse than either engine alone.
///
/// Only [`AnnotPass::SynthesizedOnly`] is painted, so annotations hayro
/// already drew from a real `/AP` are not drawn a second time -- double
/// compositing would darken any annot with `/CA` < 1.
///
/// Failure here is never fatal: annotations are decoration, and hayro's page
/// is already a valid render. Any error leaves `hayro_pix` exactly as it came.
fn overlay_missing_annots(
    doc: &PdfDocument,
    page_index: usize,
    dpi: f32,
    hayro_pix: Pixmap,
) -> Pixmap {
    let Ok(page) = doc.page(page_index).cloned() else {
        return hayro_pix;
    };
    let scale = (dpi / 72.0).max(0.01);
    let base_ctm = super::page_run::page_ctm(doc, &page);

    let mut dev = super::draw_device::DrawDevice::over_pixmap(
        hayro_pix,
        super::geometry::Matrix::scale(scale, scale),
    );
    let _ = super::annot_run::run_page_annots_with(
        doc,
        &page,
        base_ctm,
        &mut dev,
        super::annot_run::AnnotPass::SynthesizedOnly,
    );
    dev.into_pixmap()
}

/// Render `page_index` of `bytes` with `hayro`, or `None` if hayro itself
/// can't (an unparseable file, an out-of-range page, ...) -- graceful by
/// design, since the caller already has kopitiam's own output to fall back
/// to.
fn render_with_hayro(bytes: &[u8], page_index: usize, dpi: f32) -> Option<Pixmap> {
    let pdf = Pdf::new(bytes.to_vec()).ok()?;
    let page = pdf.pages().get(page_index)?;
    let cache = RenderCache::new();
    let settings = InterpreterSettings::default();
    let scale = (dpi / 72.0).max(0.01);
    let render_settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: WHITE,
        ..Default::default()
    };
    Some(convert_pixmap(render(
        page,
        &cache,
        &settings,
        &render_settings,
    )))
}

/// `hayro::vello_cpu::Pixmap` (premultiplied RGBA8) to kopitiam-pdf's own
/// DeviceRGB [`Pixmap`] (straight RGB8, no alpha). `RenderSettings::bg_color`
/// above always paints the page background opaque white, so dropping alpha
/// here loses nothing a caller of `rasterize_page` would have seen anyway --
/// its own [`DrawDevice`](super::draw_device::DrawDevice) is DeviceRGB-only
/// too.
fn convert_pixmap(vpix: hayro::vello_cpu::Pixmap) -> Pixmap {
    let w = vpix.width() as u32;
    let h = vpix.height() as u32;
    let mut out = Pixmap::new_rgb(w.max(1), h.max(1));
    for (dst, src) in out
        .samples
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(vpix.take_unpremultiplied())
    {
        dst[0] = src.r;
        dst[1] = src.g;
        dst[2] = src.b;
    }
    out
}
