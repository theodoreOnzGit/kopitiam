//! Background full-text search: the per-query worker, and the resumable scan
//! state that lets `n`/`N` find a hit on a page nobody has searched yet.
//!
//! Lifted out of `src/bin/kpdf.rs` (gh-96 Phase 3) so an embedding egui
//! application gets the same non-blocking search kpdf has. Behaviour is
//! unchanged by the move.
//!
//! # Why the work cannot stay on the UI thread
//!
//! Highlights can only be drawn for pages that have been searched, and
//! searching a page means extracting its text. Measured on the arXiv fixture,
//! that costs a **median of 5.4 ms but a p90 of 300 ms and a worst case of
//! 349 ms** -- the plot-heavy pages. So neither obvious approach works:
//! extracting on scroll drops a third of a second whenever such a page comes
//! into view, and a per-frame time budget cannot help either, because the
//! cost is paid inside one indivisible extraction that cannot be preempted
//! partway.
//!
//! Worse is the whole-document case. Searching for a term that is not in a
//! 506-page book meant scanning every page inline: **16.3 seconds** of frozen
//! window before the reader was told "not found". Hence [`FindScan`] -- the
//! scan parks after a bounded lookahead, the worker keeps feeding it pages,
//! and the window stays live throughout.
//!
//! # The copy this costs
//!
//! The worker opens its own [`PdfDocument`](crate::mupdf::PdfDocument) from a
//! copy of the bytes, because a document is not shareable across threads.
//! That is another full copy of the file in memory -- real, and worth knowing
//! for a 100 MB document -- so a host should keep the worker alive only while
//! a search is active and drop it the moment the query is cleared or
//! replaced. kpdf does exactly that.

use std::collections::HashSet;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::mupdf::structured_text::StextOptions;
use crate::mupdf::stext_search::{SearchHit, search_page};
use crate::mupdf::{PdfDocument, page_to_stext};

/// A background text-search worker for one query.
///
/// One worker per query, deliberately: the query is baked in at
/// [`spawn`](Self::spawn) so a result can never be attributed to the wrong
/// search, and changing the query means dropping this worker (which stops its
/// thread and frees its copy of the document) and starting another.
pub struct SearchWorker {
    /// Pages the UI wants searched. Dropping this closes the channel, which
    /// is how the worker is told to exit.
    req: Sender<usize>,
    /// Finished pages coming back.
    res: Receiver<(usize, Vec<SearchHit>)>,
    /// Pages already asked for, so a page visible across many frames is
    /// requested once rather than every frame.
    requested: HashSet<usize>,
}

impl SearchWorker {
    /// Start a worker searching `query` over a private copy of `bytes`.
    ///
    /// Returns `None` if the thread cannot be spawned; the caller then simply
    /// has no prefetch, and on-demand searching still works.
    pub fn spawn(bytes: Vec<u8>, query: String) -> Option<SearchWorker> {
        let (req_tx, req_rx) = channel::<usize>();
        let (res_tx, res_rx) = channel::<(usize, Vec<SearchHit>)>();
        std::thread::Builder::new()
            .name("kopitiam-pdf-search".to_string())
            .spawn(move || {
                let Ok(doc) = PdfDocument::open(bytes) else {
                    return;
                };
                // Ends when the UI drops the request sender.
                while let Ok(page) = req_rx.recv() {
                    let hits = page_to_stext(&doc, page, StextOptions::default())
                        .map(|sp| search_page(&sp, &query))
                        .unwrap_or_default();
                    // A send error means the UI is gone; stop.
                    if res_tx.send((page, hits)).is_err() {
                        return;
                    }
                }
            })
            .ok()?;
        Some(SearchWorker {
            req: req_tx,
            res: res_rx,
            requested: HashSet::new(),
        })
    }

    /// Whether this page has already been asked for.
    pub fn is_requested(&self, page: usize) -> bool {
        self.requested.contains(&page)
    }

    /// Ask for a page, unless it has already been asked for.
    ///
    /// Returns `false` if the worker thread has died; the host should then
    /// drop the worker and fall back to on-demand searching.
    ///
    /// Note the difference from the render worker: a delivered page stays
    /// marked as requested forever, rather than being released on arrival.
    /// That is correct here because the result is a permanent fact about this
    /// query -- the host caches the hits and never needs the page searched
    /// again. A worker only lives as long as its query, so the set cannot
    /// grow unboundedly across searches.
    pub fn request(&mut self, page: usize) -> bool {
        if self.requested.contains(&page) {
            return true;
        }
        if self.req.send(page).is_err() {
            return false;
        }
        self.requested.insert(page);
        true
    }

    /// Take one finished page, if any is waiting. Never blocks.
    pub fn try_recv(&mut self) -> Option<(usize, Vec<SearchHit>)> {
        self.res.try_recv().ok()
    }
}

/// Where a hit-scan got to, so it can resume when more pages are searched.
///
/// A scan exists only while `n`/`N` is looking for the next hit on pages that
/// have not been searched yet. It is *parked* rather than looping: the host
/// asks for a bounded lookahead each frame, and resumes the scan as results
/// arrive. That is what keeps "search for a term that isn't in the book" from
/// freezing the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FindScan {
    /// Scan direction: `/`+`n` forward, `?`+`n` backward.
    pub forward: bool,
    /// The next page to examine, in scan order.
    pub next: usize,
    /// How many more pages to examine before concluding there is no match.
    /// Counts down so the scan wraps the document exactly once -- without it
    /// a term that is absent would scan forever.
    pub remaining: usize,
}

/// The next `lookahead` pages a parked scan wants searched, in scan order,
/// wrapping the document.
///
/// Pure, and separated out because the backward case's modular arithmetic is
/// the kind of thing that is quietly wrong for months: going backward past
/// page 0 has to wrap to the last page, and the step has to stay correct when
/// the lookahead exceeds the page count (a 3-page document with a lookahead
/// of 8). Both are pinned by tests below.
///
/// Returns an empty vector for a zero-page document rather than dividing by
/// zero.
pub fn scan_page_order(scan: &FindScan, page_count: usize, lookahead: usize) -> Vec<usize> {
    if page_count == 0 {
        return Vec::new();
    }
    let n = page_count;
    (0..lookahead.min(scan.remaining.max(1)))
        .map(|k| {
            if scan.forward {
                (scan.next + k) % n
            } else {
                // `k % n` first so the addition cannot overflow on a long
                // lookahead, and `+ n` so the subtraction never goes
                // negative in usize arithmetic.
                (scan.next + n - (k % n)) % n
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn fixture() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/arxiv-2608.17504v1.pdf");
        std::fs::read(path).ok()
    }

    fn scan(forward: bool, next: usize, remaining: usize) -> FindScan {
        FindScan {
            forward,
            next,
            remaining,
        }
    }

    #[test]
    fn a_forward_scan_walks_up_and_wraps_at_the_end() {
        assert_eq!(scan_page_order(&scan(true, 8, 100), 10, 4), vec![8, 9, 0, 1]);
    }

    #[test]
    fn a_backward_scan_walks_down_and_wraps_past_zero() {
        assert_eq!(scan_page_order(&scan(false, 1, 100), 10, 4), vec![1, 0, 9, 8]);
    }

    /// A lookahead longer than the document must keep stepping correctly
    /// rather than repeating or jumping -- the `k % n` in the backward branch
    /// is what makes this work.
    #[test]
    fn a_lookahead_longer_than_the_document_still_steps_by_one() {
        // Steps back one page each time, so with 3 pages the sequence cycles
        // with period 3: 0, 2, 1, 0, 2, 1, 0. (Revisiting pages past the
        // first lap is harmless -- `remaining` is what stops the scan, and
        // the host skips pages it has already searched.)
        assert_eq!(
            scan_page_order(&scan(false, 0, 100), 3, 7),
            vec![0, 2, 1, 0, 2, 1, 0],
            "each step must move back exactly one page, modulo the page count"
        );
        assert_eq!(
            scan_page_order(&scan(true, 0, 100), 3, 7),
            vec![0, 1, 2, 0, 1, 2, 0]
        );
    }

    /// `remaining` bounds the scan so an absent term terminates instead of
    /// looping the document forever.
    #[test]
    fn remaining_caps_how_far_the_scan_looks() {
        assert_eq!(scan_page_order(&scan(true, 0, 2), 100, 8), vec![0, 1]);
        assert_eq!(
            scan_page_order(&scan(true, 0, 0), 100, 8),
            vec![0],
            "a spent scan still offers the page it is sitting on, not nothing"
        );
    }

    #[test]
    fn an_empty_document_yields_no_pages_rather_than_dividing_by_zero() {
        assert!(scan_page_order(&scan(true, 0, 5), 0, 8).is_empty());
    }

    /// The background searcher must return **exactly** what searching on the
    /// UI thread would have returned. It exists to move the cost, never to
    /// change the answer, so the synchronous path is the correctness
    /// reference and this asserts equivalence against it.
    #[test]
    fn the_worker_agrees_with_a_direct_search() {
        let Some(bytes) = fixture() else { return };
        let query = "reactor";

        let doc = PdfDocument::open(bytes.clone()).expect("fixture opens");
        let pages: Vec<usize> = (0..6).collect();
        let expected: Vec<(usize, Vec<SearchHit>)> = pages
            .iter()
            .map(|&p| {
                let hits = page_to_stext(&doc, p, StextOptions::default())
                    .map(|sp| search_page(&sp, query))
                    .unwrap_or_default();
                (p, hits)
            })
            .collect();

        let mut worker =
            SearchWorker::spawn(bytes, query.to_string()).expect("worker thread spawns");
        for &p in &pages {
            assert!(worker.request(p), "worker accepts requests");
        }

        let mut got: Vec<(usize, Vec<SearchHit>)> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(120);
        while got.len() < pages.len() && Instant::now() < deadline {
            match worker.try_recv() {
                Some(v) => got.push(v),
                None => std::thread::yield_now(),
            }
        }
        assert_eq!(got.len(), pages.len(), "every requested page came back");
        got.sort_by_key(|(p, _)| *p);
        assert_eq!(got, expected, "background search must match a direct one");
    }

    /// A page visible across many frames must be asked for once.
    #[test]
    fn a_requested_page_is_not_requested_twice() {
        let Some(bytes) = fixture() else { return };
        let mut worker = SearchWorker::spawn(bytes, "reactor".into()).expect("spawns");
        assert!(!worker.is_requested(3));
        assert!(worker.request(3));
        assert!(worker.is_requested(3));
        assert!(worker.request(3), "a repeat is accepted and ignored");
    }

    /// Dropping the worker must stop its thread, so it stops holding its copy
    /// of the document.
    #[test]
    fn dropping_the_worker_stops_the_thread() {
        let Some(bytes) = fixture() else { return };
        let worker = SearchWorker::spawn(bytes, "reactor".to_string()).expect("spawns");
        let res = worker.res;
        drop(worker.req); // the UI going away
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match res.recv_timeout(Duration::from_millis(500)) {
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) if Instant::now() < deadline => {
                    continue;
                }
                Err(_) => panic!("worker did not exit within the timeout"),
                Ok(_) => continue, // drain anything already in flight
            }
        }
    }
}
