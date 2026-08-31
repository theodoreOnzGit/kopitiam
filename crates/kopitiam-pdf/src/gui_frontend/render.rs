//! Background page rasterisation: the render worker, its request/result
//! types, and the generation counter that makes stale work cheap to drop.
//!
//! Lifted out of `src/bin/kpdf.rs` (gh-96 Phase 2) so an embedding egui
//! application gets the same asynchronous rendering kpdf has, without copying
//! it. Behaviour is unchanged by the move -- this is an extraction, not a
//! redesign.
//!
//! # The problem it solves
//!
//! Rasterising a page costs a **median of 135 ms and up to 444 ms** at
//! 150 dpi (measured on a 506-page book). Done on the UI thread -- which is
//! how kpdf worked until this existed -- every page scrolling into view
//! stalls the window for that long.
//!
//! # The shape
//!
//! One UI thread, one worker. The host rasterises its first few pages
//! synchronously at open so the document is readable the instant it appears;
//! everything after that is the worker's, delivered **one page at a time** as
//! each finishes rather than batched, so pages fill in progressively instead
//! of arriving in a lump. A page still rendering should draw a labelled
//! placeholder, so "not ready yet" is never mistaken for "blank page".
//!
//! # Generations, and why the counter is shared
//!
//! Zoom changes the dpi, which invalidates every in-flight render. Rather
//! than try to cancel work already running, each request carries a
//! generation and results from an older one are dropped on arrival.
//!
//! That alone is not enough, and the difference is worth stating because it
//! was a real bug. With only a UI-side check the queue is strictly FIFO, so
//! jumping to the end of a 506-page book left the worker grinding through
//! every page queued near the *old* position -- at 135 ms each -- before it
//! reached the pages actually on screen. Hence [`RenderWorker`] shares its
//! generation with the worker thread through an `AtomicU64`: the worker
//! re-checks it immediately before rasterising and drops a stale request in
//! microseconds instead of minutes.
//!
//! # No egui in here, deliberately
//!
//! The worker traffics in raw RGBA plus dimensions. Texture upload belongs at
//! the UI boundary, in the host's paint code, because only the host has an
//! `egui::Context` -- and keeping `egui::TextureHandle` out of the worker is
//! what lets the whole module be tested headlessly (see this file's tests,
//! which render real pages with no window at all).
//!
//! # The memory this costs, stated plainly
//!
//! A [`PdfDocument`](crate::mupdf::PdfDocument) holds `RefCell`s -- `Send`,
//! not `Sync` -- so the worker needs its own, opened from a copy of the
//! file's bytes. That is a second full copy of the document in memory
//! (a third, alongside a search worker), which is real for a 100 MB book.
//! Making `PdfDocument` share its bytes behind an `Arc` would leave only the
//! parsed tables duplicated; that is a change the maintainer has not taken
//! yet, and it is the honest cost of this design until they do.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::gui_frontend::pixmap::rgb_to_rgba;
use crate::mupdf::{PdfDocument, rasterize_page_with_fallback};

/// What identifies one rendered image: `(page_index, dpi.to_bits(),
/// fallback_enabled)`.
///
/// `dpi` and the fallback toggle are both part of the key so changing either
/// always produces a freshly rendered image rather than reusing one from the
/// previous setting. `f32` has no [`Eq`]/[`Hash`], hence `to_bits()`.
///
/// Named `RenderKey` rather than the binary's old `PageTextureKey`: nothing
/// at this layer knows what a texture is, and the same key identifies a
/// render whether the host uploads it to the GPU or writes it to a file.
pub type RenderKey = (usize, u32, bool);

/// What a render request is for.
///
/// Thumbnails are the same rasteriser at a much lower dpi; they are
/// distinguished only so the result lands in the right cache on the host
/// side. Routing them through the same worker (rather than rendering them
/// inline) is deliberate -- inline thumbnails were half of the "press G and
/// everything says loading" stall.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderKind {
    Page,
    Thumbnail,
}

/// One unit of work for the rasteriser.
#[derive(Clone, Copy, Debug)]
pub struct RenderRequest {
    pub page: usize,
    pub dpi: f32,
    /// Whether to allow the cross-engine fallback rasteriser.
    pub fallback: bool,
    /// The generation this request belongs to; the worker drops it without
    /// rendering if the shared counter has moved on.
    pub generation: u64,
    pub kind: RenderKind,
}

impl RenderRequest {
    /// The key this request will produce a result under.
    pub fn key(&self) -> RenderKey {
        (self.page, self.dpi.to_bits(), self.fallback)
    }
}

/// A finished page, as raw RGBA ready for texture upload.
///
/// The pixmap is converted to RGBA on the worker thread, not the UI thread --
/// it is a per-pixel pass over a multi-megabyte buffer and belongs with the
/// rendering.
#[derive(Clone, Debug)]
pub struct RenderedPage {
    pub key: RenderKey,
    pub generation: u64,
    /// `[width, height]` in pixels, in the order `egui::ColorImage` wants.
    pub size: [usize; 2],
    pub rgba: Vec<u8>,
    pub kind: RenderKind,
}

/// A background page rasteriser over a private copy of a document.
pub struct RenderWorker {
    req: Sender<RenderRequest>,
    res: Receiver<RenderedPage>,
    /// Pages asked for and not yet delivered, so a page visible across many
    /// frames is requested once.
    inflight: HashSet<RenderKey>,
    /// Bumped whenever dpi, the fallback toggle, or the reading position
    /// changes enough to make queued work irrelevant.
    generation: u64,
    /// The same counter, visible to the worker -- see the module docs.
    shared: Arc<AtomicU64>,
}

impl RenderWorker {
    /// Start a rasteriser over a private copy of `bytes`.
    ///
    /// `None` if the thread cannot be spawned; the caller should then fall
    /// back to rendering inline, which is slow but always correct. A host
    /// that cannot render inline can treat `None` as fatal, but kpdf does
    /// not, and neither should an embedder if it can avoid it.
    pub fn spawn(bytes: Vec<u8>) -> Option<RenderWorker> {
        let (req_tx, req_rx) = channel::<RenderRequest>();
        let (res_tx, res_rx) = channel::<RenderedPage>();
        let shared = Arc::new(AtomicU64::new(0));
        let worker_generation = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("kopitiam-pdf-render".to_string())
            .spawn(move || {
                let Ok(doc) = PdfDocument::open(bytes) else {
                    return;
                };
                while let Ok(r) = req_rx.recv() {
                    // Drop work the UI has already moved on from BEFORE
                    // paying for it. This check is the difference between a
                    // jump costing one page's render and costing the whole
                    // queued backlog.
                    if r.generation != worker_generation.load(Ordering::Relaxed) {
                        continue;
                    }
                    let Ok(pix) = rasterize_page_with_fallback(&doc, r.page, r.dpi, r.fallback)
                    else {
                        continue;
                    };
                    let page = RenderedPage {
                        key: r.key(),
                        generation: r.generation,
                        size: [pix.w as usize, pix.h as usize],
                        rgba: rgb_to_rgba(&pix),
                        kind: r.kind,
                    };
                    // Send each page as it finishes: the UI shows progress
                    // rather than waiting for the batch.
                    if res_tx.send(page).is_err() {
                        return; // UI gone
                    }
                }
            })
            .ok()?;
        Some(RenderWorker {
            req: req_tx,
            res: res_rx,
            inflight: HashSet::new(),
            generation: 0,
            shared,
        })
    }

    /// The generation new requests should carry.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Invalidate everything queued or running: bump the generation, publish
    /// it to the worker, and forget what was in flight.
    ///
    /// Call after a zoom, a fallback toggle, or a jump far enough that the
    /// queued backlog is no longer worth finishing. Clearing `inflight` is
    /// part of the contract, not an optimisation: those requests will never
    /// produce a delivered result now, so leaving them recorded would make
    /// [`is_inflight`](Self::is_inflight) permanently true and the host would
    /// never re-request those pages.
    pub fn bump_generation(&mut self) -> u64 {
        self.generation += 1;
        self.shared.store(self.generation, Ordering::Relaxed);
        self.inflight.clear();
        self.generation
    }

    /// Whether this key has been requested and not yet delivered.
    pub fn is_inflight(&self, key: &RenderKey) -> bool {
        self.inflight.contains(key)
    }

    /// Anything requested and not yet delivered.
    pub fn busy(&self) -> bool {
        !self.inflight.is_empty()
    }

    /// Queue a render, recording it as in flight.
    ///
    /// Returns `false` if the worker thread has died, which the host should
    /// treat as "switch to the inline path" rather than as an error. A key
    /// already in flight is not re-sent -- a page visible across sixty frames
    /// must not queue sixty renders.
    pub fn request(&mut self, r: RenderRequest) -> bool {
        let key = r.key();
        if self.inflight.contains(&key) {
            return true;
        }
        if self.req.send(r).is_err() {
            return false;
        }
        self.inflight.insert(key);
        true
    }

    /// Take one finished page, if any is waiting. Never blocks.
    ///
    /// Clears the key's in-flight mark whatever the result's generation, so a
    /// stale arrival cannot leave the key stuck as "still coming". Callers
    /// still have to check [`RenderedPage::generation`] against
    /// [`generation`](Self::generation) before *using* the pixels.
    pub fn try_recv(&mut self) -> Option<RenderedPage> {
        let page = self.res.try_recv().ok()?;
        self.inflight.remove(&page.key);
        Some(page)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// The workspace's arXiv fixture, or `None` when it is not present (a
    /// packaged build, say) -- the test then skips rather than fails.
    fn fixture() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/arxiv-2608.17504v1.pdf");
        std::fs::read(path).ok()
    }

    fn req(page: usize, generation: u64) -> RenderRequest {
        RenderRequest {
            page,
            dpi: 72.0,
            fallback: true,
            generation,
            kind: RenderKind::Page,
        }
    }

    /// The key is what makes dpi and the fallback toggle part of a render's
    /// identity. Losing either would serve a stale image after a zoom.
    #[test]
    fn the_key_separates_dpi_and_fallback() {
        let a = req(3, 0);
        let mut b = a;
        b.dpi = 150.0;
        let mut c = a;
        c.fallback = false;
        assert_ne!(a.key(), b.key(), "a different dpi is a different render");
        assert_ne!(a.key(), c.key(), "a different fallback is a different render");
        assert_eq!(a.key(), req(3, 99).key(), "the generation is NOT part of it");
    }

    /// The background rasteriser must return **exactly** what rendering on
    /// the UI thread would have returned. It exists to move the cost, never
    /// to change the pixels, so the synchronous path is the correctness
    /// reference and this asserts equivalence against it.
    #[test]
    fn the_worker_matches_a_direct_render() {
        let Some(bytes) = fixture() else { return };
        let dpi = 72.0;
        let doc = PdfDocument::open(bytes.clone()).expect("fixture opens");
        let pages: Vec<usize> = (0..3).collect();
        let expected: Vec<(RenderKey, [usize; 2], Vec<u8>)> = pages
            .iter()
            .map(|&p| {
                let pix = rasterize_page_with_fallback(&doc, p, dpi, true).expect("renders");
                (
                    (p, dpi.to_bits(), true),
                    [pix.w as usize, pix.h as usize],
                    rgb_to_rgba(&pix),
                )
            })
            .collect();

        let mut worker = RenderWorker::spawn(bytes).expect("worker spawns");
        for &p in &pages {
            assert!(worker.request(req(p, 0)), "worker accepts requests");
        }

        let mut got: Vec<RenderedPage> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(120);
        while got.len() < pages.len() && Instant::now() < deadline {
            match worker.try_recv() {
                Some(p) => got.push(p),
                None => std::thread::yield_now(),
            }
        }
        assert_eq!(got.len(), pages.len(), "every requested page came back");
        got.sort_by_key(|p| p.key.0);
        for (g, (key, size, rgba)) in got.iter().zip(expected) {
            assert_eq!(g.key, key);
            assert_eq!(g.size, size, "page {} size", key.0);
            assert_eq!(g.rgba, rgba, "page {} pixels differ from a direct render", key.0);
        }
    }

    /// The stale-work guard, and the reason `G` on a long book is no longer a
    /// stall: a request whose generation the worker has moved past must be
    /// dropped **without rendering**. Proven by timing -- a real render of
    /// these pages takes far longer than this budget.
    #[test]
    fn stale_requests_are_skipped_without_rendering() {
        let Some(bytes) = fixture() else { return };
        let mut worker = RenderWorker::spawn(bytes).expect("worker spawns");
        // Publish generation 2, then queue a pile of generation-1 work.
        worker.bump_generation();
        worker.bump_generation();
        assert_eq!(worker.generation(), 2);

        let t0 = Instant::now();
        for page in 0..6 {
            let mut r = req(page, 1);
            r.dpi = 300.0; // expensive, if it were ever rendered
            assert!(worker.request(r));
        }
        // Now one live request, which must come back promptly because the
        // stale ones cost microseconds each rather than a render apiece.
        assert!(worker.request(req(0, 2)));
        let deadline = Instant::now() + Duration::from_secs(120);
        let mut live = None;
        while live.is_none() && Instant::now() < deadline {
            match worker.try_recv() {
                Some(p) => live = Some(p),
                None => std::thread::yield_now(),
            }
        }
        let p = live.expect("the live request came back");
        assert_eq!(p.generation, 2, "only the current generation is delivered");
        assert!(
            t0.elapsed() < Duration::from_secs(60),
            "took {:?} -- the stale generation-1 work was rendered instead of \
             skipped",
            t0.elapsed()
        );
    }

    /// A page visible across many frames must be queued once, not once per
    /// frame.
    #[test]
    fn an_inflight_page_is_not_requested_twice() {
        let Some(bytes) = fixture() else { return };
        let mut worker = RenderWorker::spawn(bytes).expect("worker spawns");
        let r = req(0, 0);
        assert!(!worker.is_inflight(&r.key()), "nothing in flight to start");
        assert!(worker.request(r));
        assert!(worker.is_inflight(&r.key()), "requesting marks it in flight");
        assert!(worker.busy());
        assert!(worker.request(r), "a repeat is accepted...");
        assert!(worker.is_inflight(&r.key()), "...and still counted once");
    }

    /// Bumping the generation must forget what was in flight. Those requests
    /// will never deliver now, so a host that kept believing they were coming
    /// would never re-request the page -- it would stay a placeholder
    /// forever.
    #[test]
    fn bumping_the_generation_clears_inflight() {
        let Some(bytes) = fixture() else { return };
        let mut worker = RenderWorker::spawn(bytes).expect("worker spawns");
        let r = req(0, 0);
        worker.request(r);
        assert!(worker.is_inflight(&r.key()));
        worker.bump_generation();
        assert!(
            !worker.is_inflight(&r.key()),
            "a generation bump must release the in-flight mark"
        );
        assert!(!worker.busy());
    }

    /// Dropping the worker must stop its thread rather than leaking it: the
    /// request channel closes, the `recv` loop ends.
    #[test]
    fn dropping_the_worker_stops_the_thread() {
        let Some(bytes) = fixture() else { return };
        let worker = RenderWorker::spawn(bytes).expect("worker spawns");
        let res = worker.res;
        drop(worker.req);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match res.recv_timeout(Duration::from_millis(200)) {
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                _ if Instant::now() > deadline => {
                    panic!("worker thread outlived its request channel")
                }
                _ => continue,
            }
        }
    }
}
