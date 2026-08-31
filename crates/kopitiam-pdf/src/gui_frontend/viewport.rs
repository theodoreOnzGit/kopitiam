//! The continuous-document viewport: where each page sits in one tall
//! scrolling column, which page the reader is looking at, and the scroll
//! intents that navigation queues up for the next paint.
//!
//! Lifted out of `src/bin/kpdf.rs` (gh-96 Phase 4). The coordinate
//! *mathematics* already lived in [`geometry`](super::geometry) and stays
//! there -- this module owns the **state** around it, which is the half that
//! was still trapped in the binary: the slot cache and when to invalidate it,
//! the current page, the queued scroll deltas, and the last measured viewport
//! height.
//!
//! # One authoritative coordinate model
//!
//! The brief's Phase 4 requirement, and worth stating plainly: a downstream
//! application must never re-derive page hit testing. Everything that maps a
//! screen position to a page -- and back -- goes through
//! [`geometry::screen_to_page_at`](super::geometry::screen_to_page_at) over
//! the slots this type produces. A second implementation is how two views of
//! the same document start disagreeing about what the reader clicked.
//!
//! # No egui in here, deliberately
//!
//! Scroll deltas are plain `(f32, f32)`, not `egui::Vec2`, so the whole model
//! is testable with no window -- which is exactly what the tests below do.
//! The host converts at the paint boundary.
//!
//! # Why navigation does not just assign the page
//!
//! The current page is **recomputed every frame from viewport overlap**
//! ([`Viewport::recompute_page`]), not assigned by nav keys. So `next_page`,
//! `goto`, `gg`/`G` and a sidebar click all record a *scroll intent* instead
//! ([`Viewport::take_scroll_target`]), which the paint pass consumes. Assigning
//! the page directly would simply be overwritten by the next frame's
//! recomputation before it ever painted -- a bug that looks like "the button
//! does nothing".

use super::geometry::{
    ContinuousSlot, PageSize, current_page_in_view, layout_continuous_pages,
};

/// Vertical gap between pages in the continuous column, in pixels.
///
/// Non-zero so consecutive pages read as separate sheets rather than one
/// unbroken scroll of text; small enough that it never looks like a missing
/// page.
pub const CONTINUOUS_GAP: f32 = 8.0;

/// What invalidates the cached slot list: dpi, the page count (a new
/// document), and the fallback toggle.
///
/// It used to carry the thumbnail count too, because page sizes were measured
/// from rendered thumbnails and so improved as more arrived. Sizes now come
/// from `/MediaBox`, which is exact on the first frame, so there is nothing to
/// refine and the layout is built exactly once per combination. `f32` has no
/// [`Eq`]/[`Hash`], hence `to_bits()`.
pub type SlotsCacheKey = (u32, usize, bool);

/// The continuous view's scroll/layout state for one document.
#[derive(Debug, Clone)]
pub struct Viewport {
    page: usize,
    page_count: usize,
    dpi: f32,
    scroll_to_page: Option<usize>,
    pending_scroll_delta: (f32, f32),
    viewport_h: f32,
    slots_cache: Option<(SlotsCacheKey, Vec<ContinuousSlot>)>,
}

impl Viewport {
    /// A viewport over a `page_count`-page document at `dpi`.
    ///
    /// `viewport_h` starts at a plausible 800 px because `Ctrl+d`/`Ctrl+u`'s
    /// half-viewport step needs a height before the scroll area has ever laid
    /// out. It is replaced by the real measurement on the first painted frame.
    pub fn new(page_count: usize, dpi: f32) -> Viewport {
        Viewport {
            page: 0,
            page_count,
            dpi,
            scroll_to_page: None,
            pending_scroll_delta: (0.0, 0.0),
            viewport_h: 800.0,
            slots_cache: None,
        }
    }

    pub fn page(&self) -> usize {
        self.page
    }

    pub fn page_count(&self) -> usize {
        self.page_count
    }

    pub fn dpi(&self) -> f32 {
        self.dpi
    }

    /// Change the dpi, invalidating the layout.
    pub fn set_dpi(&mut self, dpi: f32) {
        if dpi != self.dpi {
            self.dpi = dpi;
            self.slots_cache = None;
        }
    }

    /// Point this viewport at a different document, resetting everything
    /// position-shaped.
    ///
    /// The page is clamped rather than reset to 0: reloading a live-recompiled
    /// PDF should keep the reader where they were, which is the whole point of
    /// a preview loop. A document that got shorter clamps to its new last page.
    pub fn set_document(&mut self, page_count: usize) {
        self.page_count = page_count;
        self.page = self.page.min(page_count.saturating_sub(1));
        self.scroll_to_page = None;
        self.pending_scroll_delta = (0.0, 0.0);
        self.slots_cache = None;
    }

    /// The last measured height of the scrolling viewport, in pixels.
    pub fn viewport_height(&self) -> f32 {
        self.viewport_h
    }

    /// Record the viewport height measured during layout. Ignores
    /// non-positive values, which egui can report for a collapsed or
    /// not-yet-laid-out area and which would make a half-page step a no-op.
    pub fn set_viewport_height(&mut self, h: f32) {
        if h > 0.0 {
            self.viewport_h = h;
        }
    }

    /// The per-page slots of the continuous layout, rebuilt only when
    /// [`SlotsCacheKey`] actually changes.
    ///
    /// `sizes` is a closure rather than a slice because building it means
    /// reading `/MediaBox` for every page -- work worth skipping entirely on
    /// the frames where the cache is warm, which is nearly all of them. A long
    /// document's slot list is otherwise non-trivial `Vec` churn to rebuild
    /// many times a second for nothing.
    pub fn slots<F>(&mut self, fallback: bool, sizes: F) -> &[ContinuousSlot]
    where
        F: FnOnce() -> Vec<PageSize>,
    {
        let key: SlotsCacheKey = (self.dpi.to_bits(), self.page_count, fallback);
        let stale = !matches!(&self.slots_cache, Some((k, _)) if *k == key);
        if stale {
            self.slots_cache = Some((key, layout_continuous_pages(&sizes(), CONTINUOUS_GAP)));
        }
        // `stale` guarantees this is populated.
        &self.slots_cache.as_ref().expect("just populated").1
    }

    /// Drop the cached layout, forcing a rebuild on the next
    /// [`slots`](Self::slots) call. For changes the key does not capture --
    /// pages inserted or deleted at the same count, say.
    pub fn invalidate_layout(&mut self) {
        self.slots_cache = None;
    }

    /// Recompute the current page from what is actually on screen.
    ///
    /// Call once per frame with the painted scroll offsets. Returns `true` if
    /// the page changed, which a host can turn into a page-changed event.
    /// Leaves the page alone when nothing is visible (an empty document, or a
    /// scroll position past the end), rather than snapping to 0.
    pub fn recompute_page(&mut self, viewport_top: f32, viewport_bottom: f32) -> bool {
        let Some((_, slots)) = &self.slots_cache else {
            return false;
        };
        match current_page_in_view(slots, viewport_top, viewport_bottom) {
            Some(p) if p != self.page => {
                self.page = p;
                true
            }
            _ => false,
        }
    }

    // ---- navigation: queues a scroll intent, never assigns the page -----

    /// Ask to scroll to `page`, clamped into range. A no-op for an empty
    /// document.
    pub fn scroll_to(&mut self, page: usize) {
        if self.page_count == 0 {
            return;
        }
        self.scroll_to_page = Some(page.min(self.page_count - 1));
    }

    /// Ask to scroll to a **1-based** page number, as typed into `:N`.
    ///
    /// Separate from [`scroll_to`](Self::scroll_to) because off-by-one here is
    /// a bug the reader sees immediately, and it is worth having exactly one
    /// place that does the conversion. `0` clamps to the first page rather
    /// than underflowing.
    pub fn scroll_to_1based(&mut self, page_1based: usize) {
        self.scroll_to(page_1based.saturating_sub(1));
    }

    pub fn next_page(&mut self) {
        if self.page + 1 < self.page_count {
            self.scroll_to(self.page + 1);
        }
    }

    pub fn prev_page(&mut self) {
        if self.page > 0 {
            self.scroll_to(self.page - 1);
        }
    }

    /// `gg`.
    pub fn go_to_first_page(&mut self) {
        self.scroll_to(0);
    }

    /// `G`.
    pub fn go_to_last_page(&mut self) {
        self.scroll_to(self.page_count.saturating_sub(1));
    }

    /// Move to `page` **now** and queue the scroll there.
    ///
    /// The exception to "navigation never assigns the page", and it earns it:
    /// after a search jump or a link the reader must act on the new page
    /// *this* frame -- looking up hits, painting highlights -- and waiting for
    /// next frame's recompute would use the old page for all of it. Ordinary
    /// nav keys must still go through [`scroll_to`](Self::scroll_to).
    pub fn jump_to(&mut self, page: usize) {
        if self.page_count == 0 {
            return;
        }
        self.page = page.min(self.page_count - 1);
        self.scroll_to_page = Some(self.page);
    }

    /// Re-scroll to wherever the reader already is. Used after a change that
    /// rebuilt the layout (a zoom, an inserted page) so the view does not
    /// silently drift to a different part of the document.
    pub fn rescroll_to_current(&mut self) {
        self.scroll_to(self.page);
    }

    /// Take the pending scroll target, if any. Consuming, because the intent
    /// applies to exactly one frame -- leaving it set would re-scroll every
    /// frame and make the document impossible to scroll away from by hand.
    pub fn take_scroll_target(&mut self) -> Option<usize> {
        self.scroll_to_page.take()
    }

    /// Whether a scroll target is queued, without consuming it.
    pub fn has_scroll_target(&self) -> bool {
        self.scroll_to_page.is_some()
    }

    // ---- scroll nudges (vim h/j/k/l, Ctrl+d/u) --------------------------

    /// Queue a scroll nudge in screen pixels. Accumulates within a frame, so
    /// two `j`s in one frame scroll twice as far rather than once.
    pub fn nudge(&mut self, dx: f32, dy: f32) {
        self.pending_scroll_delta.0 += dx;
        self.pending_scroll_delta.1 += dy;
    }

    /// Take the accumulated nudge. Consuming, for the same reason as
    /// [`take_scroll_target`](Self::take_scroll_target): a delta left in place
    /// would re-apply every frame and the document would scroll forever.
    pub fn take_scroll_delta(&mut self) -> (f32, f32) {
        std::mem::replace(&mut self.pending_scroll_delta, (0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes(n: usize) -> Vec<PageSize> {
        (0..n)
            .map(|_| PageSize {
                display_w: 600.0,
                display_h: 800.0,
                page_w_pts: 612.0,
                page_h_pts: 792.0,
            })
            .collect()
    }

    fn vp(n: usize) -> Viewport {
        Viewport::new(n, 150.0)
    }

    /// The layout is built once per (dpi, page count, fallback) and reused.
    /// Rebuilding it every frame is measurable `Vec` churn on a long book.
    #[test]
    fn the_slot_layout_is_cached_until_its_key_changes() {
        let mut v = vp(10);
        // A Cell so the counter can be read between calls without the
        // closure holding a mutable borrow across them.
        let built = std::cell::Cell::new(0usize);
        let build = |v: &mut Viewport, fallback: bool| -> usize {
            v.slots(fallback, || {
                built.set(built.get() + 1);
                sizes(10)
            })
            .len()
        };
        assert_eq!(build(&mut v, false), 10);
        assert_eq!(built.get(), 1, "first call builds");
        build(&mut v, false);
        assert_eq!(built.get(), 1, "a warm cache must not rebuild");

        build(&mut v, true);
        assert_eq!(built.get(), 2, "the fallback toggle is part of the key");

        v.set_dpi(300.0);
        build(&mut v, true);
        assert_eq!(built.get(), 3, "dpi is part of the key");

        v.invalidate_layout();
        build(&mut v, true);
        assert_eq!(built.get(), 4, "an explicit invalidation forces a rebuild");
    }

    /// Setting the same dpi again must not throw the layout away -- a zoom
    /// handler that clamps to the maximum would otherwise rebuild the whole
    /// slot list on every further zoom-in keystroke.
    #[test]
    fn setting_an_unchanged_dpi_keeps_the_cache() {
        let mut v = vp(10);
        let built = std::cell::Cell::new(0usize);
        v.slots(false, || {
            built.set(built.get() + 1);
            sizes(10)
        });
        v.set_dpi(150.0);
        v.slots(false, || {
            built.set(built.get() + 1);
            sizes(10)
        });
        assert_eq!(built.get(), 1);
    }

    /// Navigation queues an intent; it must NOT assign the page, which is
    /// recomputed from viewport overlap each frame. Assigning here is the bug
    /// where the nav button appears to do nothing.
    #[test]
    fn navigation_queues_a_scroll_target_without_moving_the_page() {
        let mut v = vp(10);
        v.next_page();
        assert_eq!(v.page(), 0, "the page is not assigned by navigating");
        assert!(v.has_scroll_target());
        assert_eq!(v.take_scroll_target(), Some(1));
        assert_eq!(
            v.take_scroll_target(),
            None,
            "the intent applies to one frame only"
        );
    }

    #[test]
    fn navigation_stays_in_range_at_both_ends() {
        let mut v = vp(3);
        v.prev_page();
        assert_eq!(v.take_scroll_target(), None, "no previous page from 0");
        v.go_to_last_page();
        assert_eq!(v.take_scroll_target(), Some(2));
        v.scroll_to(999);
        assert_eq!(v.take_scroll_target(), Some(2), "clamped to the last page");
    }

    /// `:N` is 1-based. Off-by-one here is visible to the reader on every
    /// jump, so both ends are pinned.
    #[test]
    fn goto_is_one_based_and_clamps_at_both_ends() {
        let mut v = vp(10);
        v.scroll_to_1based(1);
        assert_eq!(v.take_scroll_target(), Some(0), "page 1 is index 0");
        v.scroll_to_1based(10);
        assert_eq!(v.take_scroll_target(), Some(9));
        v.scroll_to_1based(0);
        assert_eq!(v.take_scroll_target(), Some(0), "0 clamps, does not wrap");
        v.scroll_to_1based(99);
        assert_eq!(v.take_scroll_target(), Some(9));
    }

    /// An empty document must not panic or queue a target for a page that
    /// does not exist.
    #[test]
    fn an_empty_document_refuses_to_queue_navigation() {
        let mut v = vp(0);
        v.go_to_last_page();
        v.next_page();
        v.scroll_to_1based(1);
        assert_eq!(v.take_scroll_target(), None);
        assert_eq!(v.page(), 0);
    }

    /// `jump_to` is the deliberate exception: a search hit must be usable on
    /// the frame it is found, not the frame after.
    #[test]
    fn jump_to_moves_the_page_immediately_and_also_scrolls() {
        let mut v = vp(10);
        v.jump_to(6);
        assert_eq!(v.page(), 6, "usable this frame, unlike scroll_to");
        assert_eq!(v.take_scroll_target(), Some(6));
        v.jump_to(999);
        assert_eq!(v.page(), 9, "still clamped");
    }

    #[test]
    fn jump_to_is_a_no_op_on_an_empty_document() {
        let mut v = vp(0);
        v.jump_to(3);
        assert_eq!(v.page(), 0);
        assert_eq!(v.take_scroll_target(), None);
    }

    #[test]
    fn scroll_nudges_accumulate_and_are_taken_once() {
        let mut v = vp(5);
        v.nudge(0.0, -40.0);
        v.nudge(0.0, -40.0);
        v.nudge(10.0, 0.0);
        assert_eq!(v.take_scroll_delta(), (10.0, -80.0));
        assert_eq!(
            v.take_scroll_delta(),
            (0.0, 0.0),
            "taking must reset, or the view scrolls forever"
        );
    }

    /// The page follows what is on screen. Pages here are 800 tall with an
    /// 8 px gap, so page 1 spans y = 808..1608.
    #[test]
    fn the_page_is_recomputed_from_what_is_visible() {
        let mut v = vp(5);
        v.slots(false, || sizes(5));
        assert!(v.recompute_page(808.0, 1608.0), "moved onto page 1");
        assert_eq!(v.page(), 1);
        assert!(
            !v.recompute_page(808.0, 1608.0),
            "an unchanged page reports no change"
        );
    }

    /// Before any layout exists there is nothing to recompute against, and
    /// guessing would fight the first real frame.
    #[test]
    fn recomputing_without_a_layout_changes_nothing() {
        let mut v = vp(5);
        assert!(!v.recompute_page(0.0, 800.0));
        assert_eq!(v.page(), 0);
    }

    /// A hot reload that shortened the document must clamp, not leave the
    /// reader on a page that no longer exists.
    #[test]
    fn switching_documents_clamps_the_page_instead_of_resetting_it() {
        let mut v = vp(10);
        v.slots(false, || sizes(10));
        v.recompute_page(808.0 * 5.0, 808.0 * 5.0 + 800.0);
        let was = v.page();
        assert!(was > 0, "precondition: we are not on page 0");

        v.set_document(20);
        assert_eq!(v.page(), was, "a longer document keeps the position");

        v.set_document(3);
        assert_eq!(v.page(), 2, "a shorter one clamps to its last page");

        v.set_document(0);
        assert_eq!(v.page(), 0, "and an empty one does not underflow");
    }

    #[test]
    fn a_document_swap_drops_stale_scroll_intents() {
        let mut v = vp(10);
        v.scroll_to(7);
        v.nudge(0.0, -100.0);
        v.set_document(10);
        assert_eq!(v.take_scroll_target(), None);
        assert_eq!(v.take_scroll_delta(), (0.0, 0.0));
    }

    /// A zero or negative measurement (a collapsed panel, a frame before
    /// layout) must not become the half-page step, or Ctrl+d stops working.
    #[test]
    fn a_nonpositive_viewport_height_is_ignored() {
        let mut v = vp(5);
        v.set_viewport_height(1000.0);
        v.set_viewport_height(0.0);
        v.set_viewport_height(-5.0);
        assert_eq!(v.viewport_height(), 1000.0);
    }
}
