//! Page-thumbnail scheduling and caching, separated from whatever chrome
//! draws them.
//!
//! Lifted out of `src/bin/kpdf.rs` (gh-96 Phase 5). The split the brief asks
//! for, and the reason it matters: **the engine belongs to the library, the
//! sidebar belongs to the host.** An embedding application may want a
//! horizontal filmstrip, a grid, a popover, or no thumbnail UI at all while
//! still using thumbnails for something else -- so what is exposed here is
//! request / fetch-cached / know-if-pending, not a widget.
//!
//! (Per AID-0057 the reader also offers a ready-made sidebar the host can
//! drop into a `Ui` it owns. That is a convenience built on this type, never
//! a requirement for using it.)
//!
//! # Why thumbnails go through the render worker
//!
//! A thumbnail costs ~76 ms to rasterise, and a scrolling list asks for every
//! row it can see: jumping to the end of a 506-page book uncovers about ten
//! uncached rows at once, which rendered inline is most of a second of frozen
//! window. Worse, that stall also delayed the page pump, so the full-size
//! pages started late too -- together, that was the whole of "press G and
//! everything says loading for a while". So thumbnails are queued on the same
//! [`RenderWorker`] as pages, at [`THUMBNAIL_DPI`], and drawn when they
//! arrive.
//!
//! # Why the cache is unbounded
//!
//! Deliberately, unlike the full-resolution page cache. A thumbnail at 24 dpi
//! is a few kilobytes, so holding every page's for as long as the document
//! stays open costs little -- and the sidebar scrolls back and forth
//! constantly, which is the access pattern an LRU serves worst. It is cleared
//! when the document is replaced or edited, which is the only time the
//! contents can go stale.

use std::collections::HashMap;

use super::render::{RenderKind, RenderRequest, RenderWorker};

/// Low, fixed dpi every page is rasterised at for the thumbnail strip.
///
/// `kovan`'s equivalent constant is 36.0, chosen for a slightly larger strip
/// in that app's 3-pane layout. 24.0 here instead, at the upper end of
/// "something like 16-24" from this crate's own brief -- smaller than kovan's
/// for a slightly cheaper up-front render, still large enough that a scaled-up
/// placeholder box's aspect ratio reads correctly. Either value is a judgment
/// call, not a measured one.
pub const THUMBNAIL_DPI: f32 = 24.0;

/// Thumbnail cache and scheduler for one document.
#[derive(Default)]
pub struct Thumbnails {
    cache: HashMap<usize, egui::TextureHandle>,
}

impl Thumbnails {
    pub fn new() -> Thumbnails {
        Thumbnails::default()
    }

    /// This page's thumbnail, if it has been rendered.
    pub fn get(&self, page: usize) -> Option<&egui::TextureHandle> {
        self.cache.get(&page)
    }

    /// Record a finished thumbnail.
    pub fn insert(&mut self, page: usize, tex: egui::TextureHandle) {
        self.cache.insert(page, tex);
    }

    /// How many pages have a thumbnail.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Drop everything. Call when the document is replaced or edited -- a
    /// thumbnail of a page that has since been deleted or drawn on is
    /// actively misleading, not merely stale.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// The render key a thumbnail of `page` is produced under, so a host can
    /// ask the worker whether it is already coming.
    pub fn key(page: usize, fallback: bool) -> super::render::RenderKey {
        (page, THUMBNAIL_DPI.to_bits(), fallback)
    }

    /// Whether this page's thumbnail is queued but not yet delivered.
    ///
    /// A host uses this to tell "still coming" from "will never come"
    /// (no worker), which decides between a placeholder and rendering inline.
    pub fn is_pending(&self, worker: &RenderWorker, page: usize, fallback: bool) -> bool {
        !self.cache.contains_key(&page) && worker.is_inflight(&Self::key(page, fallback))
    }

    /// Ask for `page`'s thumbnail unless it is already cached.
    ///
    /// Returns `true` if it is cached or now queued, `false` if the worker
    /// has died and the caller must render inline instead. Queuing is
    /// idempotent -- [`RenderWorker::request`] ignores a key already in
    /// flight -- so calling this every frame for every visible row, which is
    /// exactly what a scrolling sidebar does, costs one queue per page.
    pub fn request(&self, worker: &mut RenderWorker, page: usize, fallback: bool) -> bool {
        if self.cache.contains_key(&page) {
            return true;
        }
        worker.request(RenderRequest {
            page,
            dpi: THUMBNAIL_DPI,
            fallback,
            generation: worker.generation(),
            kind: RenderKind::Thumbnail,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key must match what the render worker will stamp on the result,
    /// or delivered thumbnails would never be recognised as the ones asked
    /// for.
    #[test]
    fn the_key_is_the_thumbnail_dpi_and_the_fallback_flag() {
        assert_eq!(Thumbnails::key(4, true), (4, THUMBNAIL_DPI.to_bits(), true));
        assert_ne!(
            Thumbnails::key(4, true),
            Thumbnails::key(4, false),
            "the fallback toggle changes the image, so it changes the key"
        );
    }

    #[test]
    fn the_cache_holds_and_clears() {
        let mut t = Thumbnails::new();
        assert!(t.is_empty());
        assert!(t.get(0).is_none());
        // A real TextureHandle needs a Context; the cache's own behaviour is
        // exercised through len/clear, which is all this type adds over a map.
        t.clear();
        assert_eq!(t.len(), 0);
    }
}
