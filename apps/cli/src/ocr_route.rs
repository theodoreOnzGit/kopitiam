//! Page routing: *which engine should read this page, and why?*
//!
//! # The decision this replaces
//!
//! Until now the choice was a single bool ([`super::ocr_fallback::should_ocr_page`]):
//! either the PDF's own text layer produced the page, or tesseract OCR'd the
//! whole thing. That is enough while there are exactly two engines, but it has
//! three problems that get worse as engines are added:
//!
//! 1. **It cannot express a third option.** A bool has no room for "a vision
//!    model should read this one".
//! 2. **It throws away its reasoning.** A page silently comes back thin and
//!    nobody can tell whether OCR declined to run, ran and found nothing, or was
//!    never asked. That is the most common "is this a bug?" question in the
//!    pipeline.
//! 3. **It cannot be compared.** To know whether a smarter router is actually
//!    better, you need two plans over the same document to diff. A bool computed
//!    inline leaves nothing to diff.
//!
//! A [`RoutePlan`] fixes all three: it is a *value*, produced before any work
//! happens, that says what each page gets and on what evidence.
//!
//! # Why a VLM belongs at this seam, not deeper
//!
//! The intended next router asks a small vision model (SmolVLM-class) to look at
//! a page and dispatch it. Note carefully what that model is being trusted with
//! here: **the routing decision, not the characters.**
//!
//! That boundary is deliberate and it is a correctness argument, not a
//! performance one. A VLM asked to *read* a page emits fluent, plausible text
//! that may not be on the page, and nothing downstream can tell the difference —
//! exactly the "do not invent facts" failure this project cannot absorb, because
//! the wrong text becomes a permanent structured fact. A VLM asked to *route* a
//! page can only be wrong in ways that are visible and recoverable: it sends a
//! page to the wrong engine, and that engine still supplies the real glyphs.
//!
//! So the rule this module encodes: **models choose the reader; deterministic
//! engines do the reading.**
//!
//! # Status: the VLM router is not implemented yet, on purpose
//!
//! [`Engine::Vlm`] exists in the vocabulary but no router in this file ever
//! emits it, because `kopitiam-runtime` is GGUF **text-only** — no ViT, no
//! projector, no `mmproj` path — so a SmolVLM simply cannot execute here today.
//! Shipping a stub router that always fails would be worse than shipping none:
//! it reads as capability that isn't there. What ships now is the seam plus the
//! honest heuristic router, so the model drops in when the runtime can host one.

use kopitiam_pdf::Page;

use crate::ocr_fallback::{is_low_text_page, OcrMode};

/// Which engine should produce the text for a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// The PDF's own text layer, as extracted by the ported MuPDF `stext`
    /// engine. Free, exact, and always preferred when the page actually has one
    /// — no recognition step can beat glyphs the file already told us about.
    TextLayer,
    /// Rasterize the page and run tesseract's LSTM recognizer over it. The
    /// fallback for scanned/image-only pages.
    Tesseract,
    /// A vision model reads the page directly.
    ///
    /// **Not currently reachable.** No router emits this yet — see the module
    /// docs. It is in the vocabulary so the plan type does not have to change
    /// shape when a vision-capable runtime lands.
    ///
    /// The `allow` is deliberate and narrow: this is a declared seam, kept
    /// honest by `no_router_emits_vlm_yet`, which fails the build the moment
    /// anything starts producing it by accident. Deleting the variant to please
    /// the lint would remove the one piece of vocabulary the whole module exists
    /// to provide; faking a router that emits it would be worse still.
    #[allow(dead_code)]
    Vlm,
}

impl Engine {
    /// A short label for plan output.
    pub fn label(self) -> &'static str {
        match self {
            Engine::TextLayer => "text-layer",
            Engine::Tesseract => "tesseract",
            Engine::Vlm => "vlm",
        }
    }
}

/// What one page was routed to, and the evidence behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// 0-based index of the page within the document.
    pub page_index: usize,
    /// The engine chosen to read it.
    pub engine: Engine,
    /// Why — phrased for a human reading a plan dump. Never empty: a decision
    /// that cannot explain itself is the thing this type exists to abolish.
    pub why: String,
}

/// The routing decision for a whole document, computed **before** any
/// recognition runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoutePlan {
    /// Which router produced this plan ([`PageRouter::name`]). Recorded so a
    /// saved or printed plan is self-describing — when two plans are diffed to
    /// judge a new router, the output has to say which is which or the
    /// comparison is unattributable.
    pub router: String,
    pub dispatches: Vec<Dispatch>,
}

impl RoutePlan {
    /// Pages routed to `engine`, by index — the query the pipeline actually
    /// needs ("which pages do I hand to tesseract?").
    pub fn pages_for(&self, engine: Engine) -> Vec<usize> {
        self.dispatches
            .iter()
            .filter(|d| d.engine == engine)
            .map(|d| d.page_index)
            .collect()
    }

    /// Renders the plan one line per page, stable and diffable — so two routers
    /// over the same document can be compared with `diff`. That comparison is
    /// the only honest way to claim a smarter router is an improvement.
    pub fn to_report(&self) -> String {
        let mut out = format!("router: {}\n", self.router);
        for d in &self.dispatches {
            // 1-based in output: page numbers are a human-facing thing, and every
            // other page reference the user sees (viewers, `--pages`) is 1-based.
            out.push_str(&format!(
                "page {}: {} — {}\n",
                d.page_index + 1,
                d.engine.label(),
                d.why
            ));
        }
        out
    }
}

/// Decides which engine reads which page.
///
/// One method, taking the whole document at once rather than a page at a time:
/// a router is allowed to reason across pages (a document that is scanned
/// throughout should not be re-litigated per page), and a vision router will
/// want to batch. A per-page signature would foreclose both.
pub trait PageRouter {
    /// Stable identifier for this router, recorded in plan output so a saved
    /// plan says which router produced it.
    fn name(&self) -> &str;

    /// Produce the routing plan for `pages`.
    fn route(&self, pages: &[Page], mode: OcrMode) -> RoutePlan;
}

/// The deterministic router: today's behaviour, made explicit and explainable.
///
/// Emits exactly the decisions [`super::ocr_fallback::should_ocr_page`] already
/// made — this is deliberately **not** a behaviour change. The value added is
/// that each decision now carries its evidence, and that the whole thing is a
/// value you can print, diff and test. Getting cleverer is the next router's
/// job; this one's job is to be a faithful, boring baseline to measure against.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicRouter;

impl PageRouter for HeuristicRouter {
    fn name(&self) -> &str {
        "heuristic"
    }

    fn route(&self, pages: &[Page], mode: OcrMode) -> RoutePlan {
        let dispatches = pages
            .iter()
            .enumerate()
            .map(|(page_index, page)| {
                let (engine, why) = match mode {
                    OcrMode::Off => {
                        (Engine::TextLayer, "--ocr off: OCR disabled".to_string())
                    }
                    OcrMode::On => {
                        (Engine::Tesseract, "--ocr on: forced for every page".to_string())
                    }
                    OcrMode::Auto if is_low_text_page(page) => (
                        Engine::Tesseract,
                        "little/no extractable text — looks scanned".to_string(),
                    ),
                    OcrMode::Auto => (
                        Engine::TextLayer,
                        "text layer already has content".to_string(),
                    ),
                };
                Dispatch { page_index, engine, why }
            })
            .collect();
        RoutePlan { router: self.name().to_string(), dispatches }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_pdf::TextSpan;

    /// A page carrying `texts` as its spans — enough for the low-text trigger,
    /// which only counts non-whitespace characters. Page geometry is US Letter,
    /// matching the equivalent helper in `ocr_fallback`'s tests.
    fn page(number: usize, texts: &[&str]) -> Page {
        Page {
            number,
            width: 612.0,
            height: 792.0,
            spans: texts
                .iter()
                .map(|t| TextSpan { text: (*t).to_string(), ..TextSpan::default() })
                .collect(),
        }
    }

    fn scanned(number: usize) -> Page {
        page(number, &[])
    }

    fn typeset(number: usize) -> Page {
        page(number, &["This page has a real text layer with plenty of characters on it."])
    }

    #[test]
    fn auto_sends_only_the_scanned_pages_to_tesseract() {
        let pages = vec![typeset(1), scanned(2), typeset(3)];
        let plan = HeuristicRouter.route(&pages, OcrMode::Auto);
        assert_eq!(plan.pages_for(Engine::Tesseract), vec![1]);
        assert_eq!(plan.pages_for(Engine::TextLayer), vec![0, 2]);
    }

    #[test]
    fn off_never_routes_to_ocr_and_on_always_does() {
        let pages = vec![typeset(1), scanned(2)];
        assert!(HeuristicRouter.route(&pages, OcrMode::Off).pages_for(Engine::Tesseract).is_empty());
        assert_eq!(
            HeuristicRouter.route(&pages, OcrMode::On).pages_for(Engine::Tesseract),
            vec![0, 1]
        );
    }

    #[test]
    fn the_plan_matches_what_should_ocr_page_already_decided() {
        // The baseline router must be a FAITHFUL restatement of the existing
        // bool, not a quiet behaviour change smuggled in under a refactor. If
        // these ever diverge, the pipeline silently starts OCRing different
        // pages than before — invisible in output, expensive in wrong text.
        let pages = vec![typeset(1), scanned(2), typeset(3), scanned(4)];
        for mode in [OcrMode::Auto, OcrMode::On, OcrMode::Off] {
            let plan = HeuristicRouter.route(&pages, mode);
            for (i, p) in pages.iter().enumerate() {
                let routed_to_ocr = plan.dispatches[i].engine == Engine::Tesseract;
                assert_eq!(
                    routed_to_ocr,
                    crate::ocr_fallback::should_ocr_page(mode, p),
                    "page {i} under {mode:?} disagrees with should_ocr_page"
                );
            }
        }
    }

    #[test]
    fn every_dispatch_explains_itself() {
        // The whole point of the type: a decision with no evidence is the bug.
        let pages = vec![typeset(1), scanned(2)];
        for mode in [OcrMode::Auto, OcrMode::On, OcrMode::Off] {
            for d in HeuristicRouter.route(&pages, mode).dispatches {
                assert!(!d.why.trim().is_empty(), "{mode:?} produced a silent decision");
            }
        }
    }

    #[test]
    fn report_is_one_line_per_page_and_one_based() {
        let plan = HeuristicRouter.route(&[typeset(1), scanned(2)], OcrMode::Auto);
        let report = plan.to_report();
        // Header line naming the router, then one line per page.
        assert_eq!(report.lines().count(), 3);
        assert!(report.starts_with("router: heuristic\n"), "got: {report}");
        assert!(report.contains("page 1: text-layer"), "got: {report}");
        assert!(report.contains("page 2: tesseract"), "got: {report}");
    }

    #[test]
    fn no_router_emits_vlm_yet() {
        // Guards the honesty claim in the module docs: the variant exists as a
        // seam, but nothing pretends to route to a model that cannot run. If a
        // real VLM router lands, this test is the one to delete deliberately.
        let pages = vec![typeset(1), scanned(2)];
        for mode in [OcrMode::Auto, OcrMode::On, OcrMode::Off] {
            assert!(
                HeuristicRouter.route(&pages, mode).pages_for(Engine::Vlm).is_empty(),
                "no vision-capable runtime exists yet, so nothing may route to it"
            );
        }
    }

    #[test]
    fn an_empty_document_plans_nothing_rather_than_panicking() {
        let plan = HeuristicRouter.route(&[], OcrMode::Auto);
        assert!(plan.dispatches.is_empty());
        // Still names its router: an empty plan is a real answer ("nothing to
        // do"), not an absence of one.
        assert_eq!(plan.to_report(), "router: heuristic\n");
    }
}
