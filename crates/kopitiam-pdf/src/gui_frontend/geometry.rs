//! Page layout and the screen-space <-> PDF-space coordinate mapping.
//!
//! # The coordinate trap
//!
//! A GUI toolkit reports pointer positions in **screen space** (ui points,
//! origin top-left, y **down**). PDF annotation/form geometry
//! ([`crate::mupdf::annot_edit::InkStroke`],
//! [`crate::mupdf::annot_edit::AnnotRef::rect`],
//! [`crate::mupdf::form::FormField::rect`]) all speak **default user space**
//! (PDF points, origin bottom-left, y **up**). Getting the y-flip or the
//! dpi/zoom unscaling wrong here puts ink in the wrong place, or mirrored --
//! and it would look plausible enough to ship without a careful eye.
//! [`screen_to_page`]/[`page_to_screen`]/[`page_size_pts`] isolate exactly
//! that seam as pure, unit-tested functions (see the `tests` module) so the
//! mapping can be checked without a display.

use super::zoom::DPI_DEFAULT;
use crate::mupdf::Rect;

/// Compute the on-screen size (in ui points) to draw the current page
/// texture at, given the dpi it was rasterized at.
///
/// # The bug this replaces
///
/// The obvious approach -- `egui::Image::new(tex).shrink_to_fit()` -- always
/// scales the texture to exactly fill the available panel, preserving
/// aspect ratio (`egui`'s own `ImageFit::Fraction` resolves to
/// `scale_to_fit(image_source_size, available_size, ..)`, i.e. `ratio =
/// min(available.x / tex.x, available.y / tex.y)`, computed straight from
/// the *current* texture's pixel size). Since a caller's texture cache
/// re-rasterizes at a higher pixel density every time `dpi` rises, that fit
/// ratio shrinks by almost exactly the amount `dpi` grew by -- the two
/// cancel out, the page gets sharper but never bigger, and "zoom" is
/// silently a no-op on screen. This was true of `kpdf`'s `+`/`-` keys from
/// day one; on-screen buttons only made it obvious.
///
/// # The fix
///
/// 1. Undo dpi's contribution to the texture's pixel size, recovering
///    `base` -- what the texture's pixel size would be at [`DPI_DEFAULT`].
///    This is dpi-invariant: it depends only on the page's own physical
///    dimensions, which don't change with dpi.
/// 2. Contain-fit `base` into `available` the same way `shrink_to_fit`
///    would (preserve aspect ratio, scale by
///    `min(available.w/base.w, available.h/base.h)`) -- this reproduces
///    the on-screen size at `dpi == DPI_DEFAULT` exactly.
/// 3. Re-apply the zoom factor (`dpi / DPI_DEFAULT`) on top of that fitted
///    size, so raising dpi now visibly grows the displayed page instead of
///    cancelling out against step 2's shrink.
///
/// Returns `(0.0, 0.0)` for degenerate input (a non-positive texture
/// dimension or dpi) rather than dividing by zero / propagating a NaN into
/// a layout.
///
/// Pure arithmetic over plain `f32`s -- no GUI types -- so it is
/// unit-tested directly; see the `tests` module below.
pub fn page_display_size(
    tex_w: f32,
    tex_h: f32,
    dpi: f32,
    available_w: f32,
    available_h: f32,
) -> (f32, f32) {
    if tex_w <= 0.0 || tex_h <= 0.0 || dpi <= 0.0 {
        return (0.0, 0.0);
    }
    let zoom = dpi / DPI_DEFAULT;
    let base_w = tex_w / zoom;
    let base_h = tex_h / zoom;
    let ratio = (available_w / base_w).min(available_h / base_h);
    let ratio = if ratio.is_finite() { ratio } else { 1.0 };
    (base_w * ratio * zoom, base_h * ratio * zoom)
}

/// After a zoom changes the page's displayed size, compute the scroll
/// offset (along one axis) that keeps whatever content point was centred in
/// the viewport *before* the resize still centred *after* it -- rather than
/// leaving the same raw pixel offset pointing at an arbitrary spot on the
/// now differently-sized page, which reads as the page randomly jumping.
///
/// Works per-axis (plain `f32`s, not a vector type) so the same function
/// covers both width and height and is trivial to unit-test; a caller
/// applies it twice.
///
/// `content_size` and `viewport_size` describe the layout *before* the
/// resize (i.e. as of the last frame); `new_content_size` is the page's
/// freshly-computed [`page_display_size`] for this frame. Anchors to the
/// viewport centre rather than the cursor position -- simpler, and "nicer
/// but do not over-engineer it" was an explicit non-goal for the cursor
/// variant.
pub fn recentred_scroll_offset(
    offset: f32,
    content_size: f32,
    viewport_size: f32,
    new_content_size: f32,
) -> f32 {
    if content_size <= 0.0 {
        return 0.0;
    }
    // Fraction of the old content that was centred under the viewport
    // (0.0 = the content's top/left edge is centred, 1.0 = its
    // bottom/right edge is) -- clamped because a viewport bigger than the
    // content (nothing to scroll) would otherwise push this outside [0, 1].
    let center_fraction = ((offset + viewport_size / 2.0) / content_size).clamp(0.0, 1.0);
    (center_fraction * new_content_size - viewport_size / 2.0).max(0.0)
}

/// The page's physical size in **PDF points** (default user space),
/// recovered from a rasterized texture's pixel size and the dpi it was
/// rasterized at: `points = pixels / dpi * 72` (72 points per inch is the
/// PDF spec's own unit definition, not a viewer convention).
///
/// This is **dpi-invariant** by construction -- rasterizing the same page at
/// a different dpi changes `tex_w`/`tex_h` but this function's return value
/// stays the same, which is exactly what makes it safe to use as the fixed
/// reference frame in [`screen_to_page`]/[`page_to_screen`] no matter what
/// zoom level is currently on screen.
pub fn page_size_pts(tex_w: f32, tex_h: f32, dpi: f32) -> (f32, f32) {
    if dpi <= 0.0 {
        return (0.0, 0.0);
    }
    (tex_w / dpi * 72.0, tex_h / dpi * 72.0)
}

/// Everything [`screen_to_page`]/[`page_to_screen`] need about the current
/// frame's layout: where the page image is drawn on screen, and the page's
/// own physical size. Bundled into one small `Copy` struct purely to keep
/// those two functions' argument count under clippy's `too_many_arguments`
/// lint -- it carries no behaviour beyond being a parameter bag, and every
/// field is still named at each construction site so nothing is positionally
/// ambiguous.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageLayout {
    /// Top-left of the on-screen image rect (screen space).
    pub image_x: f32,
    pub image_y: f32,
    /// Displayed size of the page image (screen space) -- already fitted and
    /// zoomed by [`page_display_size`], **not** the texture's raw pixel size.
    pub image_w: f32,
    pub image_h: f32,
    /// The page's physical size in PDF points -- from [`page_size_pts`].
    pub page_w_pts: f32,
    pub page_h_pts: f32,
    /// The page's `/MediaBox` origin in PDF points -- the user-space
    /// coordinates of its **bottom-left corner**.
    ///
    /// Not always `(0, 0)`, which is the entire reason this field exists.
    /// §7.7.3.3 places the media box's lower-left at `(x0, y0)`, so on a page
    /// boxed `[9 9 621 801]` the visual bottom-left corner is user-space
    /// `(9, 9)`. Assuming zero puts every annotation and form-field highlight
    /// off by exactly that much -- not obviously broken, just persistently
    /// *slightly wrong*, which is how it survived until a maintainer drew on a
    /// real paper and said the ink landed in the wrong place.
    pub page_x0: f32,
    pub page_y0: f32,
}

/// Screen space (ui points, origin top-left, y **down** -- what a pointer
/// event reports) to default user space (PDF points, origin bottom-left, y
/// **up** -- what [`crate::mupdf::annot_edit::InkStroke`]/
/// [`crate::mupdf::annot_edit::AnnotRef::rect`]/
/// [`crate::mupdf::form::FormField::rect`] all expect).
///
/// Pure (plain `f32`s and [`PageLayout`], no GUI/mupdf types), so it is
/// unit-tested directly without a display -- see the `tests` module's
/// round-trip and y-flip cases, which is the one part of this coordinate
/// mapping a gate can actually check.
pub fn screen_to_page(screen_x: f32, screen_y: f32, layout: PageLayout) -> (f32, f32) {
    if layout.image_w <= 0.0 || layout.image_h <= 0.0 {
        return (0.0, 0.0);
    }
    let fx = (screen_x - layout.image_x) / layout.image_w;
    let fy = (screen_y - layout.image_y) / layout.image_h;
    // Offset by the media box origin, NOT just scaled: default user space
    // starts at the box's lower-left corner, which is only (0, 0) when the
    // box says so. See `PageLayout::page_x0`.
    let page_x = layout.page_x0 + fx * layout.page_w_pts;
    // The y-flip: screen y grows downward from the image's top; page y
    // grows upward from the page's bottom. fy=0 (image top) must map to
    // page_y = page_h_pts (page top); fy=1 (image bottom) must map to
    // page_y = 0 (page bottom).
    let page_y = layout.page_y0 + (1.0 - fy) * layout.page_h_pts;
    (page_x, page_y)
}

/// The inverse of [`screen_to_page`] -- default user space back to screen
/// space, e.g. to paint an in-progress ink preview or to position something
/// over a `/Rect`. See [`screen_to_page`]'s docs for the parameters and the
/// y-flip this undoes.
/// Map a point in **fitz page space** to the screen.
///
/// Not the same space as [`page_to_screen`], and mixing them up is a real bug
/// rather than a theoretical one — it shipped in the first cut of search
/// highlighting and put every highlight in the wrong place.
///
/// * [`page_to_screen`] takes **PDF user space**: y-up, origin bottom-left.
///   That is what annotation `/Rect`s and destination coordinates use, so it
///   flips y.
/// * This takes **fitz page space**: y-down, origin top-left, box already
///   translated to `(0, 0)` — what `page_ctm` produces and therefore what
///   every `StextChar::quad` from the structured-text device is in
///   (`page_run::page_ctm`: "left-handed (flip y), then rotate", then
///   translate the box origin to the origin).
///
/// Because that space already matches the rendered image's own orientation,
/// this is a plain scale with **no flip**. Passing stext coordinates through
/// `page_to_screen` flips them a second time and mirrors the result
/// vertically.
pub fn stext_to_screen(x: f32, y: f32, layout: PageLayout) -> (f32, f32) {
    if layout.page_w_pts <= 0.0 || layout.page_h_pts <= 0.0 {
        return (layout.image_x, layout.image_y);
    }
    (
        layout.image_x + (x / layout.page_w_pts) * layout.image_w,
        layout.image_y + (y / layout.page_h_pts) * layout.image_h,
    )
}

pub fn page_to_screen(page_x: f32, page_y: f32, layout: PageLayout) -> (f32, f32) {
    if layout.page_w_pts <= 0.0 || layout.page_h_pts <= 0.0 {
        return (layout.image_x, layout.image_y);
    }
    // The inverse of `screen_to_page`: subtract the origin before scaling.
    let fx = (page_x - layout.page_x0) / layout.page_w_pts;
    let fy = 1.0 - (page_y - layout.page_y0) / layout.page_h_pts;
    (
        layout.image_x + fx * layout.image_w,
        layout.image_y + fy * layout.image_h,
    )
}

/// Map a form field's `/Rect` (default user space) to an on-screen rect --
/// for painting the fillable-area highlight or positioning an in-place text
/// editor over the field's own box. Delegates both corners to the
/// already-tested [`page_to_screen`] and only then normalises min/max: the
/// y-flip means a PDF rect's "bottom" corner (`y0`, the smaller value in
/// PDF's y-up space) becomes the on-screen *top*, so a naive corner-to-
/// corner copy without normalising would swap top and bottom on screen (the
/// same trap [`screen_to_page`]/[`page_to_screen`]'s own docs call out).
/// Returns `(min_x, min_y, max_x, max_y)` rather than a GUI-toolkit rect
/// type so it stays composable with the other pure functions here and
/// testable without constructing GUI geometry.
pub fn field_rect_to_screen(rect: Rect, layout: PageLayout) -> (f32, f32, f32, f32) {
    let (sx0, sy0) = page_to_screen(rect.x0, rect.y0, layout);
    let (sx1, sy1) = page_to_screen(rect.x1, rect.y1, layout);
    (sx0.min(sx1), sy0.min(sy1), sx0.max(sx1), sy0.max(sy1))
}

/// Widen `rect` (default user space) about its own centre so that, once
/// mapped to screen space via `layout`, neither side is smaller than
/// `min_screen_size` -- a comfortable click target for a field whose
/// `/Rect` renders as a hairline or a near-zero box at low zoom (not
/// unheard of; some form authors size the widget to a printed underline
/// rather than the real input box). Used both before hit-testing
/// ([`crate::gui_frontend::hit_test::hit_test_field_expanded`]) and before painting
/// the highlight overlay, deliberately: the highlighted area and the
/// clickable area must always agree, or a field would visibly highlight
/// smaller than where a click actually registers.
///
/// Computed per-axis from `layout`'s own page-points-to-screen-pixels ratio
/// -- [`page_display_size`] currently keeps zoom uniform across both axes,
/// so in practice `scale_x == scale_y`, but this function does not assume
/// that.
pub fn min_hit_rect(rect: Rect, layout: PageLayout, min_screen_size: f32) -> Rect {
    if layout.page_w_pts <= 0.0
        || layout.page_h_pts <= 0.0
        || layout.image_w <= 0.0
        || layout.image_h <= 0.0
    {
        return rect;
    }
    let scale_x = layout.image_w / layout.page_w_pts;
    let scale_y = layout.image_h / layout.page_h_pts;
    let min_w = min_screen_size / scale_x;
    let min_h = min_screen_size / scale_y;
    let cx = (rect.x0 + rect.x1) / 2.0;
    let cy = (rect.y0 + rect.y1) / 2.0;
    let w = (rect.x1 - rect.x0).abs().max(min_w);
    let h = (rect.y1 - rect.y0).abs().max(min_h);
    Rect {
        x0: cx - w / 2.0,
        y0: cy - h / 2.0,
        x1: cx + w / 2.0,
        y1: cy + h / 2.0,
    }
}

/// Whether `(x, y)` (default user space) falls within `r` -- an inclusive
/// bounds check, since a click exactly on an annotation's edge should still
/// hit it.
pub fn rect_contains(r: Rect, x: f32, y: f32) -> bool {
    x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1
}

// -- Continuous (all-pages-in-one-column) scroll layout ----------------------
//
// Everything below this point supports a *continuous* viewer -- every page
// stacked in one scrollable column, rather than one page filling the whole
// viewport at a time (the model [`PageLayout`]/[`screen_to_page`] above
// assume). The overall shape -- learn each page's on-screen size from its
// already-cheap thumbnail render rather than a full rasterization, only
// full-render pages actually intersecting the viewport, track "current
// page" by vertical viewport overlap -- is ported from a sibling project
// (`kovan`, a different KOPITIAM-adjacent app by the same maintainer) that
// solved continuous PDF scrolling first; see `kpdf.rs`'s module docs for
// exactly what was taken and from where. Unlike that source, this version
// *does* resolve pen/eraser/forms clicks to a page while scrolling (see
// [`screen_to_page_at`]) rather than leaving continuous mode view-only --
// kopitiam-pdf already has the single-page hit-testing this needs, so the
// extra resolution step is a binary search plus delegation, not a
// re-derivation.
//
// See [`ContinuousSlot`]'s docs for the coordinate system this introduces,
// and [`screen_to_page_at`] for the seam every interactive feature (pen,
// eraser, forms clicks, field highlighting) now routes through.

/// One page's already-computed on-screen size and physical (PDF-point) size
/// -- the input [`layout_continuous_pages`] stacks into a
/// [`ContinuousSlot`] per page. Deliberately carries no dpi/zoom logic of
/// its own: the caller (`kpdf.rs`) is the one that knows the current dpi and
/// derives `display_w`/`display_h` from a page's cached thumbnail texture
/// size scaled up to render dpi (see that file's module docs) -- keeping
/// that arithmetic out of this pure-layout function is what lets
/// [`layout_continuous_pages`] stay a plain "stack these rectangles"
/// operation, testable with hand-picked numbers instead of a real render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageSize {
    /// This page's on-screen size at the caller's current zoom.
    pub display_w: f32,
    pub display_h: f32,
    /// The page's physical size in PDF points -- carried through so
    /// [`screen_to_page_at`] can build a [`PageLayout`] without a second
    /// document lookup.
    pub page_w_pts: f32,
    pub page_h_pts: f32,
    /// The page's `/MediaBox` origin in PDF points -- the user-space
    /// coordinates of its **bottom-left corner**.
    ///
    /// Not always `(0, 0)`, which is the entire reason this field exists.
    /// §7.7.3.3 places the media box's lower-left at `(x0, y0)`, so on a page
    /// boxed `[9 9 621 801]` the visual bottom-left corner is user-space
    /// `(9, 9)`. Assuming zero puts every annotation and form-field highlight
    /// off by exactly that much -- not obviously broken, just persistently
    /// *slightly wrong*, which is how it survived until a maintainer drew on a
    /// real paper and said the ink landed in the wrong place.
    pub page_x0: f32,
    pub page_y0: f32,
}

/// One page's placement within a **continuous** (all-pages-stacked-
/// vertically) scroll layout, in **content space**: content `y == 0` is the
/// top of the very first page and increases monotonically down the whole
/// document, entirely independent of the current scroll offset -- scrolling
/// only changes which slice of this space the viewport currently shows, it
/// never changes these numbers. Built by [`layout_continuous_pages`] (one
/// per page, for the whole document) and consumed by [`screen_to_page_at`]
/// (resolving a content-space point back to a page) and by a caller's own
/// per-frame painting (converting each visible slot to an on-screen
/// [`PageLayout`] by adding the content column's current screen-space
/// origin -- see `kpdf.rs`'s `Mode::Image` rendering for the one place that
/// addition happens).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousSlot {
    /// 0-based, matching every other page index in this crate.
    pub page_index: usize,
    /// Top edge of this page's image, content-space y.
    pub top: f32,
    pub width: f32,
    pub height: f32,
    /// The page's physical size in PDF points -- see [`PageSize`].
    pub page_w_pts: f32,
    pub page_h_pts: f32,
    /// The page's `/MediaBox` origin in PDF points -- the user-space
    /// coordinates of its **bottom-left corner**.
    ///
    /// Not always `(0, 0)`, which is the entire reason this field exists.
    /// §7.7.3.3 places the media box's lower-left at `(x0, y0)`, so on a page
    /// boxed `[9 9 621 801]` the visual bottom-left corner is user-space
    /// `(9, 9)`. Assuming zero puts every annotation and form-field highlight
    /// off by exactly that much -- not obviously broken, just persistently
    /// *slightly wrong*, which is how it survived until a maintainer drew on a
    /// real paper and said the ink landed in the wrong place.
    pub page_x0: f32,
    pub page_y0: f32,
}

impl ContinuousSlot {
    /// The bottom edge, content-space y -- `top + height`, named so a
    /// caller doesn't have to re-derive it (and risk an off-by-`height`
    /// mistake) at every use site.
    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }
}

/// Stack `sizes` top-to-bottom, `gap` screen points apart, into one
/// [`ContinuousSlot`] per page -- pure arithmetic (no document, no dpi/zoom
/// awareness; see [`PageSize`]'s docs for why that lives in the caller
/// instead), which is what makes this testable with hand-picked sizes
/// rather than a real render. `gap` is clamped to `>= 0.0` so a caller
/// passing a negative value by mistake cannot make slots overlap.
pub fn layout_continuous_pages(sizes: &[PageSize], gap: f32) -> Vec<ContinuousSlot> {
    let gap = gap.max(0.0);
    let mut slots = Vec::with_capacity(sizes.len());
    let mut top = 0.0f32;
    for (i, size) in sizes.iter().enumerate() {
        let width = size.display_w.max(0.0);
        let height = size.display_h.max(0.0);
        slots.push(ContinuousSlot {
            page_index: i,
            top,
            width,
            height,
            page_w_pts: size.page_w_pts,
            page_h_pts: size.page_h_pts,
            page_x0: size.page_x0,
            page_y0: size.page_y0,
        });
        top += height + gap;
    }
    slots
}

/// The continuous-view analogue of [`screen_to_page`]: resolve a
/// **content-space** point (see [`ContinuousSlot`]'s docs -- a caller
/// converts a raw pointer position to content space by subtracting the
/// content column's own on-screen origin for this frame, which already
/// reflects the current scroll offset, since that origin comes from
/// wherever the GUI toolkit actually placed the column widget) to
/// `(page_index, page_x_pts, page_y_pts)`.
///
/// Returns `None` for a point above the first page, below the last page, or
/// inside the gap between two pages -- there is no page there, and a caller
/// (annotation drawing, the eraser, forms clicks) must treat that exactly
/// like today's single-page "click missed the image" case rather than
/// guessing at a nearby page. See the `tests` module below for exactly
/// these boundary cases.
///
/// `slots` must be in top-to-bottom document order (exactly what
/// [`layout_continuous_pages`] produces) -- this binary-searches it
/// (`partition_point`, O(log n)) for the first slot whose bottom edge is
/// past `content_y`, rather than scanning linearly, since a long document
/// can carry hundreds of slots and this runs on every pointer event a tool
/// handles.
///
/// Delegates the actual page-space mapping to the already-tested
/// [`screen_to_page`], via a [`PageLayout`] built from the resolved slot
/// (`image_x = 0.0`, `image_y = slot.top`) -- so the y-flip this module's
/// top-of-file docs warn about is *inherited*, not re-derived a second time
/// for the continuous case.
pub fn screen_to_page_at(
    content_x: f32,
    content_y: f32,
    slots: &[ContinuousSlot],
) -> Option<(usize, f32, f32)> {
    // `partition_point`'s predicate must hold for a prefix and then never
    // again -- true here because `top` (hence `bottom()`) increases
    // monotonically across `slots`, so "this slot ends at or before
    // content_y" starts true and flips to false exactly once.
    let idx = slots.partition_point(|s| s.bottom() <= content_y);
    let slot = slots
        .get(idx)
        .filter(|s| content_y >= s.top && content_y < s.bottom())?;
    let layout = PageLayout {
        image_x: 0.0,
        image_y: slot.top,
        image_w: slot.width,
        image_h: slot.height,
        page_w_pts: slot.page_w_pts,
        page_h_pts: slot.page_h_pts,
        page_x0: slot.page_x0,
        page_y0: slot.page_y0,
    };
    let (px, py) = screen_to_page(content_x, content_y, layout);
    Some((slot.page_index, px, py))
}

/// Which page counts as "current" for a continuous view whose viewport
/// (content-space) currently spans `[viewport_top, viewport_bottom)` -- the
/// page with the **most vertical overlap** with that range, ported from
/// `kovan`'s own continuous-scroll reader (see this module's top-of-section
/// docs): more robust than a single centre-point test when a very tall page
/// spans the whole viewport, or several short pages are all partly visible
/// at once. Falls back to the nearest page (by distance to `viewport_top`)
/// when `viewport_top`/`viewport_bottom` fall entirely outside every slot
/// (before the first page or past the last) -- only an empty `slots`
/// returns `None`.
pub fn current_page_in_view(
    slots: &[ContinuousSlot],
    viewport_top: f32,
    viewport_bottom: f32,
) -> Option<usize> {
    slots
        .iter()
        .map(|s| {
            let overlap = s.bottom().min(viewport_bottom) - s.top.max(viewport_top);
            (s, overlap.max(0.0))
        })
        .max_by(|(a, ov_a), (b, ov_b)| {
            ov_a.partial_cmp(ov_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    // Tie-break (including the "no slot overlaps at all" case,
                    // where every overlap is 0.0) by proximity to the top of
                    // the viewport, so "past the last page" resolves to the
                    // last page and "before the first" to the first, rather
                    // than an arbitrary `max_by` tie winner.
                    let da = (a.top - viewport_top).abs();
                    let db = (b.top - viewport_top).abs();
                    db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .map(|(s, _)| s.page_index)
}

/// Whether `slot` intersects `[viewport_top, viewport_bottom]` widened by
/// `margin` on both edges -- the "should this page be full-resolution
/// rendered this frame" test a continuous view's paint loop runs per page,
/// per frame. Widening by a margin (rather than the exact viewport) means a
/// page one screen-height below the fold is already rasterized by the time
/// a fast scroll reaches it, instead of the reader seeing a bare placeholder
/// flash by first.
pub fn continuous_slot_visible(
    slot: &ContinuousSlot,
    viewport_top: f32,
    viewport_bottom: f32,
    margin: f32,
) -> bool {
    slot.bottom() >= viewport_top - margin && slot.top <= viewport_bottom + margin
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- page_display_size ---------------------------------------------

    #[test]
    fn at_default_dpi_matches_plain_contain_fit() {
        // zoom == 1.0, so this must reduce to an ordinary "contain fit"
        // (preserve aspect ratio, shrink to the tighter axis) exactly like
        // egui's own `shrink_to_fit` -- this is the "look the way it does
        // today" requirement at DPI_DEFAULT.
        //
        // Portrait texture (1000x2000), wide-ish viewport (500x1500):
        // width ratio 500/1000=0.5, height ratio 1500/2000=0.75 -> width
        // is the tighter axis, so the result is (500, 1000).
        let (w, h) = page_display_size(1000.0, 2000.0, DPI_DEFAULT, 500.0, 1500.0);
        assert!((w - 500.0).abs() < 1e-3);
        assert!((h - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn doubling_dpi_doubles_the_displayed_size() {
        // A page that exactly fills a 900x1200 window at DPI_DEFAULT (i.e.
        // its DPI_DEFAULT-rasterized texture is already 900x1200, a 1:1
        // contain fit). Doubling dpi doubles the texture's pixel size to
        // 1800x2400 (what a real texture cache would actually produce); the
        // displayed size must double too, to 1800x2400, overflowing the
        // still-900x1200 window -- that overflow is exactly what makes the
        // page pannable, and what was missing before this fix (previously
        // it would still have come out as 900x1200, unchanged).
        let (w, h) = page_display_size(1800.0, 2400.0, DPI_DEFAULT * 2.0, 900.0, 1200.0);
        assert!((w - 1800.0).abs() < 1e-2);
        assert!((h - 2400.0).abs() < 1e-2);
    }

    #[test]
    fn halving_dpi_halves_the_displayed_size() {
        // Same fixture as above, but zooming out: half the pixel density,
        // half the resolution -- the shown page should also shrink to half
        // of the fitted 900x1200 baseline, not stay put.
        let (w, h) = page_display_size(450.0, 600.0, DPI_DEFAULT / 2.0, 900.0, 1200.0);
        assert!((w - 450.0).abs() < 1e-2);
        assert!((h - 600.0).abs() < 1e-2);
    }

    #[test]
    fn zoom_preserves_aspect_ratio_when_container_constrains_the_other_axis() {
        // Landscape texture (2000x1000) into a viewport where height, not
        // width, is the tighter constraint (available 3000x500): contain
        // fit at DPI_DEFAULT gives ratio = 500/1000 = 0.5 -> (1000, 500).
        // At 2x zoom both dimensions must double to (2000, 1000) --
        // doubling the *texture* pixel size to 4000x2000 first, per a real
        // texture cache's actual behaviour.
        let (w, h) = page_display_size(4000.0, 2000.0, DPI_DEFAULT * 2.0, 3000.0, 500.0);
        assert!((w - 2000.0).abs() < 1e-2);
        assert!((h - 1000.0).abs() < 1e-2);
    }

    #[test]
    fn degenerate_input_returns_zero_not_nan() {
        assert_eq!(
            page_display_size(0.0, 100.0, DPI_DEFAULT, 500.0, 500.0),
            (0.0, 0.0)
        );
        assert_eq!(
            page_display_size(100.0, 0.0, DPI_DEFAULT, 500.0, 500.0),
            (0.0, 0.0)
        );
        assert_eq!(
            page_display_size(100.0, 100.0, 0.0, 500.0, 500.0),
            (0.0, 0.0)
        );
        assert_eq!(
            page_display_size(100.0, 100.0, -10.0, 500.0, 500.0),
            (0.0, 0.0)
        );
    }

    #[test]
    fn stays_finite_across_the_whole_dpi_range() {
        use super::super::zoom::{DPI_MAX, DPI_MIN};
        for dpi in [DPI_MIN, DPI_DEFAULT, DPI_MAX] {
            let (w, h) = page_display_size(1000.0, 1400.0, dpi, 900.0, 1100.0);
            assert!(w.is_finite() && w > 0.0);
            assert!(h.is_finite() && h > 0.0);
        }
    }

    // -- recentred_scroll_offset -----------------------------------------

    #[test]
    fn recentre_tracks_the_same_relative_content_point_after_doubling() {
        // Viewport 500pt wide, old content 1000pt wide, scrolled to the
        // very start (offset 0) -- the viewport's centre sits at content
        // position 250, i.e. 25% into the content. After the content
        // doubles to 2000pt, that same 25%-in point is at absolute
        // position 500; centring it means an offset of 500 - 250 = 250.
        let new_offset = recentred_scroll_offset(0.0, 1000.0, 500.0, 2000.0);
        assert!((new_offset - 250.0).abs() < 1e-3);
    }

    #[test]
    fn recentre_is_a_no_op_when_content_size_is_unchanged() {
        let old_offset = 137.0;
        let new_offset = recentred_scroll_offset(old_offset, 1000.0, 500.0, 1000.0);
        assert!((new_offset - old_offset).abs() < 1e-3);
    }

    #[test]
    fn recentre_never_goes_negative() {
        // Viewport bigger than the content -- nothing to scroll -- must
        // clamp to a sane (non-negative) offset instead of going negative.
        let new_offset = recentred_scroll_offset(0.0, 100.0, 400.0, 100.0);
        assert!(new_offset >= 0.0);
    }

    #[test]
    fn recentre_handles_zero_content_size_without_panicking() {
        assert_eq!(recentred_scroll_offset(0.0, 0.0, 500.0, 800.0), 0.0);
    }

    // -- page_size_pts ---------------------------------------------------

    /// The regression that shipped: stext coordinates are y-DOWN, so putting
    /// them through `page_to_screen` (which flips for y-up user space) mirrors
    /// every highlight vertically. A point near the top of the page must map
    /// near the top of the image.
    #[test]
    fn stext_to_screen_does_not_flip_y() {
        let layout = PageLayout {
            image_x: 0.0,
            image_y: 0.0,
            image_w: 100.0,
            image_h: 200.0,
            page_w_pts: 50.0,
            page_h_pts: 100.0,
            page_x0: 0.0,
            page_y0: 0.0,
        };
        // Top of the page in fitz space (y small) -> top of the image.
        let (_, top) = stext_to_screen(0.0, 10.0, layout);
        assert!(top < 40.0, "y=10/100 should land near the top, got {top}");
        // Bottom of the page -> bottom of the image.
        let (_, bottom) = stext_to_screen(0.0, 90.0, layout);
        assert!(bottom > 160.0, "y=90/100 should land near the bottom, got {bottom}");
        // And it is the OPPOSITE of what page_to_screen would have produced,
        // which is precisely why the two must not be interchanged.
        let (_, flipped) = page_to_screen(0.0, 10.0, layout);
        assert!(
            (flipped - bottom).abs() < 1.0,
            "page_to_screen flips: {flipped} vs stext {bottom}"
        );
    }

    #[test]
    fn stext_to_screen_scales_x_and_respects_the_image_origin() {
        let layout = PageLayout {
            image_x: 7.0,
            image_y: 11.0,
            image_w: 100.0,
            image_h: 200.0,
            page_w_pts: 50.0,
            page_h_pts: 100.0,
            page_x0: 0.0,
            page_y0: 0.0,
        };
        assert_eq!(stext_to_screen(0.0, 0.0, layout), (7.0, 11.0));
        assert_eq!(stext_to_screen(50.0, 100.0, layout), (107.0, 211.0));
        assert_eq!(stext_to_screen(25.0, 50.0, layout), (57.0, 111.0));
    }

    #[test]
    fn stext_to_screen_survives_a_degenerate_page() {
        let layout = PageLayout {
            image_x: 3.0,
            image_y: 4.0,
            image_w: 10.0,
            image_h: 10.0,
            page_w_pts: 0.0,
            page_h_pts: 0.0,
            page_x0: 0.0,
            page_y0: 0.0,
        };
        assert_eq!(stext_to_screen(5.0, 5.0, layout), (3.0, 4.0));
    }

    #[test]
    fn page_size_pts_at_72_dpi_is_pixels_unchanged() {
        // 72 points per inch is the PDF spec's own definition of a "point",
        // so rasterizing at exactly 72 dpi must make pixels and points
        // numerically identical.
        assert_eq!(page_size_pts(612.0, 792.0, 72.0), (612.0, 792.0));
    }

    #[test]
    fn page_size_pts_is_dpi_invariant() {
        // The same page rasterized at two different dpis must recover the
        // *same* physical size -- this is the property that makes it safe
        // to use as screen_to_page/page_to_screen's fixed reference frame
        // regardless of the current zoom level.
        let at_150 = page_size_pts(1275.0, 1650.0, 150.0);
        let at_300 = page_size_pts(2550.0, 3300.0, 300.0);
        assert!((at_150.0 - at_300.0).abs() < 1e-3);
        assert!((at_150.1 - at_300.1).abs() < 1e-3);
        assert!((at_150.0 - 612.0).abs() < 1e-3); // US Letter width in points
    }

    #[test]
    fn page_size_pts_degenerate_dpi_returns_zero() {
        assert_eq!(page_size_pts(100.0, 100.0, 0.0), (0.0, 0.0));
        assert_eq!(page_size_pts(100.0, 100.0, -1.0), (0.0, 0.0));
    }

    // -- screen_to_page / page_to_screen (the coordinate trap) -----------

    /// A representative layout: a 612x792pt (US Letter) page displayed in a
    /// 300x400 screen-point image rect, offset from the window's origin (so
    /// a bug that forgets to subtract `image_x`/`image_y` shows up).
    fn sample_layout() -> PageLayout {
        PageLayout {
            image_x: 20.0,
            image_y: 40.0,
            image_w: 300.0,
            image_h: 400.0,
            page_w_pts: 612.0,
            page_h_pts: 792.0,
            page_x0: 0.0,
            page_y0: 0.0,
        }
    }

    #[test]
    fn y_flip_top_of_image_is_top_of_page() {
        // The image's top-left corner (screen space) must map to the
        // page's top-left corner in default user space -- i.e. x=0, and
        // y = page_h_pts (page y grows *up* from the bottom, so the top
        // edge is at the maximum y, not 0). Getting this backwards is
        // exactly "the coordinate trap" the module docs call out.
        let layout = sample_layout();
        let (px, py) = screen_to_page(layout.image_x, layout.image_y, layout);
        assert!((px - 0.0).abs() < 1e-3);
        assert!((py - layout.page_h_pts).abs() < 1e-3);
    }

    #[test]
    fn y_flip_bottom_of_image_is_bottom_of_page() {
        let layout = sample_layout();
        let (px, py) = screen_to_page(
            layout.image_x + layout.image_w,
            layout.image_y + layout.image_h,
            layout,
        );
        assert!((px - layout.page_w_pts).abs() < 1e-3);
        assert!((py - 0.0).abs() < 1e-3);
    }

    #[test]
    fn screen_to_page_center_maps_to_page_center() {
        let layout = sample_layout();
        let (px, py) = screen_to_page(
            layout.image_x + layout.image_w / 2.0,
            layout.image_y + layout.image_h / 2.0,
            layout,
        );
        assert!((px - layout.page_w_pts / 2.0).abs() < 1e-2);
        assert!((py - layout.page_h_pts / 2.0).abs() < 1e-2);
    }

    /// THE "SLIGHTLY OFF" BUG. A page boxed `[9 9 621 801]` puts its visual
    /// bottom-left corner at user-space `(9, 9)`, not `(0, 0)` (§7.7.3.3).
    /// Ignoring that offsets every annotation and form highlight by exactly
    /// the origin -- which is why ink drawn on a real paper landed near, but
    /// not on, the pen.
    ///
    /// Both directions, and the corners specifically: the corners are where
    /// the offset is unambiguous, and a scale-only bug would still get the
    /// centre roughly right.
    #[test]
    fn a_non_zero_media_box_origin_shifts_page_coordinates() {
        // 100x200 image showing a 612x792 page boxed at (9, 9).
        let layout = PageLayout {
            image_x: 0.0,
            image_y: 0.0,
            image_w: 100.0,
            image_h: 200.0,
            page_w_pts: 612.0,
            page_h_pts: 792.0,
            page_x0: 9.0,
            page_y0: 9.0,
        };

        // Bottom-left of the image is the page's origin corner.
        let (px, py) = screen_to_page(0.0, 200.0, layout);
        assert!(
            (px - 9.0).abs() < 0.01 && (py - 9.0).abs() < 0.01,
            "bottom-left must be user-space (9, 9), got ({px}, {py}) -- \
             (0, 0) means the media box origin was dropped"
        );

        // Top-right is origin + extent.
        let (px, py) = screen_to_page(100.0, 0.0, layout);
        assert!(
            (px - 621.0).abs() < 0.01 && (py - 801.0).abs() < 0.01,
            "top-right must be (621, 801), got ({px}, {py})"
        );

        // And the inverse agrees, or ink would be drawn at one place and
        // painted back at another.
        let (sx, sy) = page_to_screen(9.0, 9.0, layout);
        assert!(
            sx.abs() < 0.01 && (sy - 200.0).abs() < 0.01,
            "page (9, 9) must paint back at the image's bottom-left, got ({sx}, {sy})"
        );
    }

    /// A zero origin must behave exactly as before -- the overwhelming
    /// majority of pages, and the case every other test in this module
    /// assumes.
    #[test]
    fn a_zero_origin_is_unchanged_by_the_offset() {
        let layout = PageLayout {
            image_x: 0.0,
            image_y: 0.0,
            image_w: 100.0,
            image_h: 200.0,
            page_w_pts: 612.0,
            page_h_pts: 792.0,
            page_x0: 0.0,
            page_y0: 0.0,
        };
        let (px, py) = screen_to_page(0.0, 200.0, layout);
        assert!(px.abs() < 0.01 && py.abs() < 0.01);
    }

    /// The round trip must hold with an origin too, not only without one.
    #[test]
    fn round_trips_survive_a_non_zero_origin() {
        let layout = PageLayout {
            image_x: 17.0,
            image_y: 23.0,
            image_w: 300.0,
            image_h: 400.0,
            page_w_pts: 612.0,
            page_h_pts: 792.0,
            page_x0: 9.0,
            page_y0: 9.0,
        };
        for (sx, sy) in [(17.0, 23.0), (317.0, 423.0), (100.0, 200.0)] {
            let (px, py) = screen_to_page(sx, sy, layout);
            let (bx, by) = page_to_screen(px, py, layout);
            assert!(
                (bx - sx).abs() < 0.01 && (by - sy).abs() < 0.01,
                "({sx}, {sy}) -> ({px}, {py}) -> ({bx}, {by})"
            );
        }
    }

    #[test]
    fn round_trip_page_to_screen_to_page() {
        // page_to_screen(screen_to_page(p)) ~= p, for a handful of points
        // spread across the image, not just the center -- the explicit
        // round-trip property the task calls for.
        let layout = sample_layout();
        for (sx, sy) in [
            (layout.image_x, layout.image_y),
            (
                layout.image_x + layout.image_w,
                layout.image_y + layout.image_h,
            ),
            (layout.image_x + 10.0, layout.image_y + 380.0),
            (layout.image_x + 150.0, layout.image_y + 200.0),
            (layout.image_x + 299.0, layout.image_y + 1.0),
        ] {
            let (px, py) = screen_to_page(sx, sy, layout);
            let (sx2, sy2) = page_to_screen(px, py, layout);
            assert!(
                (sx2 - sx).abs() < 1e-2,
                "x round-trip: {sx} -> {px} -> {sx2}"
            );
            assert!(
                (sy2 - sy).abs() < 1e-2,
                "y round-trip: {sy} -> {py} -> {sy2}"
            );
        }
    }

    #[test]
    fn round_trip_screen_to_page_to_screen() {
        // The other direction: starting from a page-space point (as if
        // reading a `/Rect` back), converting to screen and back must
        // recover it, across points that are not simply the corners.
        let layout = sample_layout();
        for (px, py) in [
            (0.0, 0.0),
            (612.0, 792.0),
            (100.0, 700.0),
            (306.0, 396.0),
            (611.0, 1.0),
        ] {
            let (sx, sy) = page_to_screen(px, py, layout);
            let (px2, py2) = screen_to_page(sx, sy, layout);
            assert!(
                (px2 - px).abs() < 1e-2,
                "x round-trip: {px} -> {sx} -> {px2}"
            );
            assert!(
                (py2 - py).abs() < 1e-2,
                "y round-trip: {py} -> {sy} -> {py2}"
            );
        }
    }

    #[test]
    fn screen_to_page_degenerate_image_size_returns_zero_not_nan() {
        let mut layout = sample_layout();
        layout.image_w = 0.0;
        assert_eq!(screen_to_page(50.0, 50.0, layout), (0.0, 0.0));
        layout = sample_layout();
        layout.image_h = -5.0;
        assert_eq!(screen_to_page(50.0, 50.0, layout), (0.0, 0.0));
    }

    #[test]
    fn page_to_screen_degenerate_page_size_returns_image_origin_not_nan() {
        let mut layout = sample_layout();
        layout.page_w_pts = 0.0;
        let (sx, sy) = page_to_screen(10.0, 10.0, layout);
        assert!(sx.is_finite() && sy.is_finite());
        assert_eq!((sx, sy), (layout.image_x, layout.image_y));
    }

    // -- field_rect_to_screen ------------------------------------------------

    #[test]
    fn field_rect_to_screen_full_page_matches_image_bounds() {
        let layout = sample_layout();
        // The whole page's rect must map back to exactly the image's
        // on-screen bounds -- this is the same y-flip `page_to_screen`
        // already covers, exercised here through the field-rect wrapper.
        let (x0, y0, x1, y1) = field_rect_to_screen(
            Rect::new(0.0, 0.0, layout.page_w_pts, layout.page_h_pts),
            layout,
        );
        assert!((x0 - layout.image_x).abs() < 1e-3);
        assert!((y0 - layout.image_y).abs() < 1e-3);
        assert!((x1 - (layout.image_x + layout.image_w)).abs() < 1e-3);
        assert!((y1 - (layout.image_y + layout.image_h)).abs() < 1e-3);
    }

    #[test]
    fn field_rect_to_screen_normalises_the_y_flip() {
        let layout = sample_layout();
        // A rect near the *bottom* of the page (small y0/y1, PDF y-up) must
        // still come out with `y0 < y1` on screen (y-down) -- i.e. this
        // must not silently hand back an inverted rect just because the
        // corner mapping flips which corner is which.
        let (_, y0, _, y1) = field_rect_to_screen(Rect::new(0.0, 0.0, 50.0, 20.0), layout);
        assert!(y0 < y1);
    }

    // -- min_hit_rect ---------------------------------------------------------

    /// Test-only stand-in for `kpdf`'s own `MIN_HIT_AREA_SCREEN` default
    /// (also 16.0 there) -- the value itself is an application UX choice
    /// (see the binary's doc comment on that constant), not something this
    /// library should own; these tests just need *some* representative
    /// minimum to exercise the widen behaviour.
    const TEST_MIN_HIT_AREA: f32 = 16.0;

    #[test]
    fn min_hit_rect_widens_a_field_smaller_than_the_minimum() {
        let layout = sample_layout();
        // scale_x = image_w / page_w_pts = 300/612, scale_y = 400/792 --
        // a 10x10pt field is well under 16 screen px on both axes here.
        let tiny = Rect::new(100.0, 100.0, 110.0, 110.0);
        let widened = min_hit_rect(tiny, layout, TEST_MIN_HIT_AREA);
        let (sx0, sy0, sx1, sy1) = field_rect_to_screen(widened, layout);
        assert!(
            sx1 - sx0 >= TEST_MIN_HIT_AREA - 1e-3,
            "widened width {} px, wanted >= {}",
            sx1 - sx0,
            TEST_MIN_HIT_AREA
        );
        assert!(
            sy1 - sy0 >= TEST_MIN_HIT_AREA - 1e-3,
            "widened height {} px, wanted >= {}",
            sy1 - sy0,
            TEST_MIN_HIT_AREA
        );
        // Widening must not move the field's centre.
        let cx = (tiny.x0 + tiny.x1) / 2.0;
        let cy = (tiny.y0 + tiny.y1) / 2.0;
        assert!(((widened.x0 + widened.x1) / 2.0 - cx).abs() < 1e-3);
        assert!(((widened.y0 + widened.y1) / 2.0 - cy).abs() < 1e-3);
    }

    #[test]
    fn min_hit_rect_leaves_an_already_comfortable_field_alone() {
        let layout = sample_layout();
        // 200x200pt is already far larger than any sane minimum -- no
        // reason to touch its bounds.
        let big = Rect::new(0.0, 0.0, 200.0, 200.0);
        assert_eq!(min_hit_rect(big, layout, TEST_MIN_HIT_AREA), big);
    }

    #[test]
    fn min_hit_rect_degenerate_layout_returns_rect_unchanged() {
        let mut layout = sample_layout();
        layout.image_w = 0.0;
        let r = Rect::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(min_hit_rect(r, layout, TEST_MIN_HIT_AREA), r);
    }

    // -- layout_continuous_pages / screen_to_page_at / current_page_in_view --
    //
    // Three US-Letter-shaped pages (612x792pt, displayed at 300x400 screen
    // points -- the same proportions `sample_layout` above uses), stacked
    // with a 10pt gap: page 0 spans content y [0, 400), the gap spans
    // [400, 410), page 1 spans [410, 810), the gap spans [810, 820), page 2
    // spans [820, 1220).

    fn three_page_sizes() -> Vec<PageSize> {
        vec![
            PageSize {
                display_w: 300.0,
                display_h: 400.0,
                page_w_pts: 612.0,
                page_h_pts: 792.0,
                page_x0: 0.0,
                page_y0: 0.0,
            };
            3
        ]
    }

    #[test]
    fn layout_continuous_pages_stacks_with_the_given_gap() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].top, 0.0);
        assert_eq!(slots[0].bottom(), 400.0);
        assert_eq!(slots[1].top, 410.0);
        assert_eq!(slots[1].bottom(), 810.0);
        assert_eq!(slots[2].top, 820.0);
        assert_eq!(slots[2].bottom(), 1220.0);
        for (i, s) in slots.iter().enumerate() {
            assert_eq!(s.page_index, i);
            assert_eq!(s.width, 300.0);
            assert_eq!(s.height, 400.0);
        }
    }

    #[test]
    fn layout_continuous_pages_negative_gap_is_clamped_to_zero() {
        let slots = layout_continuous_pages(&three_page_sizes(), -50.0);
        // A negative gap must never make slots overlap -- clamp to 0, i.e.
        // back-to-back, not negative spacing.
        assert_eq!(slots[1].top, slots[0].bottom());
    }

    #[test]
    fn layout_continuous_pages_mixed_sizes_stack_independently() {
        let sizes = vec![
            PageSize {
                display_w: 300.0,
                display_h: 400.0,
                page_w_pts: 612.0,
                page_h_pts: 792.0,
                page_x0: 0.0,
                page_y0: 0.0,
            },
            PageSize {
                display_w: 300.0,
                display_h: 200.0,
                page_w_pts: 612.0,
                page_h_pts: 396.0,
                page_x0: 0.0,
                page_y0: 0.0,
            },
        ];
        let slots = layout_continuous_pages(&sizes, 5.0);
        assert_eq!(slots[0].bottom(), 400.0);
        assert_eq!(slots[1].top, 405.0);
        assert_eq!(slots[1].bottom(), 605.0);
    }

    #[test]
    fn screen_to_page_at_above_the_first_page_is_none() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        assert_eq!(screen_to_page_at(150.0, -1.0, &slots), None);
    }

    #[test]
    fn screen_to_page_at_below_the_last_page_is_none() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        let last_bottom = slots.last().unwrap().bottom();
        assert_eq!(screen_to_page_at(150.0, last_bottom + 1.0, &slots), None);
        assert_eq!(screen_to_page_at(150.0, last_bottom, &slots), None);
    }

    #[test]
    fn screen_to_page_at_in_the_gap_between_pages_is_none() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        // The gap between page 0 (ends at 400) and page 1 (starts at 410).
        assert_eq!(screen_to_page_at(150.0, 405.0, &slots), None);
    }

    #[test]
    fn screen_to_page_at_exact_top_boundary_is_inclusive_to_that_page() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        // content_y == 410.0 is page 1's exact top edge.
        let hit = screen_to_page_at(150.0, 410.0, &slots);
        assert_eq!(hit.map(|(p, ..)| p), Some(1));
    }

    #[test]
    fn screen_to_page_at_exact_bottom_boundary_is_exclusive() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        // content_y == 400.0 is page 0's exact bottom edge -- half-open, so
        // this belongs to neither page 0 nor the gap's own page.
        assert_eq!(screen_to_page_at(150.0, 400.0, &slots), None);
    }

    #[test]
    fn screen_to_page_at_resolves_the_correct_page_and_maps_within_it() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        // The top-left corner of page 1's slot (content x=0, y=410) must
        // map to that page's own top-left in default user space: x=0,
        // y=page_h_pts (the y-flip `screen_to_page` already covers).
        let (page, px, py) = screen_to_page_at(0.0, 410.0, &slots).expect("hits page 1");
        assert_eq!(page, 1);
        assert!((px - 0.0).abs() < 1e-2);
        assert!((py - 792.0).abs() < 1e-2);
    }

    #[test]
    fn screen_to_page_at_empty_slots_is_none() {
        assert_eq!(screen_to_page_at(0.0, 0.0, &[]), None);
    }

    #[test]
    fn current_page_in_view_picks_the_most_overlapping_page() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        // A viewport mostly over page 1 (410..810) with only a sliver of
        // page 0 showing (390..400) must resolve to page 1.
        assert_eq!(current_page_in_view(&slots, 390.0, 420.0), Some(1));
    }

    #[test]
    fn current_page_in_view_before_the_first_page_falls_back_to_it() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        assert_eq!(current_page_in_view(&slots, -500.0, -100.0), Some(0));
    }

    #[test]
    fn current_page_in_view_past_the_last_page_falls_back_to_it() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        assert_eq!(current_page_in_view(&slots, 5000.0, 5400.0), Some(2));
    }

    #[test]
    fn current_page_in_view_empty_slots_is_none() {
        assert_eq!(current_page_in_view(&[], 0.0, 100.0), None);
    }

    #[test]
    fn continuous_slot_visible_true_when_intersecting() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        assert!(continuous_slot_visible(&slots[1], 500.0, 700.0, 0.0));
    }

    #[test]
    fn continuous_slot_visible_false_when_far_off_screen() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        assert!(!continuous_slot_visible(&slots[0], 5000.0, 5400.0, 0.0));
    }

    #[test]
    fn continuous_slot_visible_margin_extends_the_range() {
        let slots = layout_continuous_pages(&three_page_sizes(), 10.0);
        // Page 2 starts at 820 -- a viewport ending at 700 misses it with no
        // margin, but a 200pt margin (one extra "screen or so" of lookahead)
        // reaches it.
        assert!(!continuous_slot_visible(&slots[2], 500.0, 700.0, 0.0));
        assert!(continuous_slot_visible(&slots[2], 500.0, 700.0, 200.0));
    }
}
