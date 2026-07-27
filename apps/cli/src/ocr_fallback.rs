//! Automatic per-page OCR fallback for the `pdf2md` pipeline (token-max Task #10).
//!
//! # Why this exists
//!
//! A scanned page carries its text as a raster image (typically a `DCTDecode`
//! JPEG), so the normal extract path — which reads a PDF's *text* operators —
//! comes back with little or nothing for that page. Without a fallback, such a
//! page silently converts to an empty (or near-empty) stretch of Markdown, and
//! the recovery report can only report the loss after the fact. This module
//! closes that gap: after normal text extraction it detects the low-text pages,
//! rasterizes each one, runs it through the ported Tesseract LSTM recognizer
//! ([`kopitiam_ocr`]) using the configured languages' `.traineddata` (pulled
//! from the [`kopitiam_models`] store), and splices the recognized text back into
//! the page as ordinary [`TextSpan`]s. Reconstruction, rendering, and validation
//! then run over those spans exactly as they do for a born-digital page, so
//! headers/figures/tables/anchors all still apply.
//!
//! # Recovery-ratio honesty (`kopitiam_token_max.md` §2.1)
//!
//! OCR-recognized text is **legitimate extracted content**: it is spliced in as
//! real `TextSpan`s, so it counts on the *extracted* side of the recovery ratio
//! normally and is rendered faithfully on the *rendered* side. No scaffolding
//! (front matter, anchors, placeholders) is invented here, so the ratio stays
//! honest — a page recovered by OCR is measured against its own recognized text,
//! neither inflated nor deflated.
//!
//! # The seam (testability)
//!
//! The real OCR path needs a heavy `.traineddata` download and a rasterizer, so
//! the recognition step sits behind the [`PageOcr`] trait. The deterministic
//! unit tests drive [`apply_ocr_fallback`] with a fake recognizer returning
//! canned lines — exercising the trigger, the flag/mode logic, the
//! language→model-spec resolution, and the spans→Document merge without running
//! real OCR or touching the network. The single end-to-end test that drives the
//! real [`TesseractPageOcr`] is `#[ignore]`d with a documented run command.

use std::path::Path;

use anyhow::Context;
use clap::ValueEnum;

use kopitiam_models::{Catalog, ModelSpec, ModelStore};
use kopitiam_ocr::{
    GrayImage, GrayLine, LstmRecognizer, TessdataManager, find_text_lines, otsu_binarize,
};
use kopitiam_pdf::mupdf::{PdfDocument, Pixmap, rasterize_page};
use kopitiam_pdf::{Page, TextSpan};

use crate::ocr_route::{Engine, HeuristicRouter, PageRouter, RoutePlan};

/// Below this many non-whitespace *extracted* characters, a page is treated as
/// scanned (image-only) and becomes eligible for the OCR fallback.
///
/// This is a deliberately small, near-zero absolute threshold rather than a
/// per-area ratio: a genuinely scanned page yields ~0 real characters (at most a
/// few strays from stamped vector artifacts), whereas even a very sparse
/// born-digital page — a section title and one line — carries far more than this.
/// So the two populations are cleanly separated and the trigger never fires on a
/// page that has real text to lose (no double-processing, no cost).
pub const LOW_TEXT_CHAR_THRESHOLD: usize = 24;

/// The resolution the fallback rasterizes a low-text page at before OCR. 300 dpi
/// is the standard sweet spot for LSTM OCR (enough detail for small CJK strokes
/// without ballooning the raster).
pub const OCR_DPI: f32 = 300.0;

/// The default `--ocr-lang` set: English + Simplified Chinese + Japanese, the
/// driving case for the CJK/English nuclear literature this tool converts.
pub const DEFAULT_OCR_LANGS: &str = "eng,chi_sim,jpn";

/// Nominal per-line vertical advance (PDF points) used when laying recognized
/// lines back onto a page. Line-finding yields lines in reading order but not
/// their page coordinates (see [`ocr_lines_to_spans`]), so a uniform advance is
/// synthesized.
const OCR_LINE_HEIGHT_PT: f32 = 12.0;

/// Nominal font size (PDF points) stamped on every recognized-line span. Uniform
/// so the reconstruction layer's font-ratio heading detection treats OCR output
/// as plain body text — OCR does not recover type sizes, so inventing a heading
/// hierarchy from it would be a lie.
const OCR_FONT_SIZE_PT: f32 = 10.0;

/// `--ocr <auto|on|off>`: how the fallback engages.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum OcrMode {
    /// Automatic fallback: OCR only the pages that yielded little or no text
    /// (the scanned pages). Pages with real text are untouched. This is the
    /// default — the fallback is on out of the box.
    #[default]
    Auto,
    /// Force OCR on *every* page, even ones that extracted fine. For a document
    /// whose text layer is present but garbled, or an all-scanned corpus.
    On,
    /// Disable the fallback entirely: behave exactly as the pre-OCR pipeline did.
    Off,
}

/// What the fallback did, for a one-line notice (and for the tests to assert on).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcrSummary {
    /// How many pages were recognized by OCR and had their spans replaced.
    pub pages_ocred: usize,
    /// Total recognized lines spliced across all OCR'd pages.
    pub lines_recognized: usize,
}

/// The recognition seam: turn the low-text page at 0-based document index
/// `page_index` into lines of text, top-to-bottom in reading order.
///
/// The real implementation ([`TesseractPageOcr`]) rasterizes the page, binarizes
/// and line-finds it, and runs each line through the LSTM recognizer. Tests
/// substitute a fake returning canned lines so every non-OCR concern — trigger,
/// mode, merge — is exercised deterministically.
pub trait PageOcr {
    /// Recognize the page at 0-based document index `page_index`. An empty vector
    /// means nothing legible was found (the page's spans are then left as-is).
    fn recognize_page(&self, page_index: usize) -> anyhow::Result<Vec<String>>;
}

/// The count of non-whitespace characters the extractor produced for `page`.
fn page_nonspace_chars(page: &Page) -> usize {
    page.spans
        .iter()
        .flat_map(|span| span.text.chars())
        .filter(|c| !c.is_whitespace())
        .count()
}

/// Whether `page` extracted so little text that it is almost certainly a scanned
/// (image-only) page — the [`LOW_TEXT_CHAR_THRESHOLD`] trigger.
pub fn is_low_text_page(page: &Page) -> bool {
    page_nonspace_chars(page) < LOW_TEXT_CHAR_THRESHOLD
}

/// Whether `page` should be OCR'd under `mode`: never for `Off`, always for `On`,
/// and only when [`is_low_text_page`] for `Auto`.
///
/// **Retained as the reference oracle, not as production code.** The pipeline
/// now routes through [`crate::ocr_route::HeuristicRouter`]; this is the
/// original decision kept so a test can assert the router still agrees with it
/// exactly. That equivalence is worth a permanent guard, because a divergence
/// would be invisible — the pipeline would just quietly start OCRing a
/// different set of pages, with no error and no output that looks wrong.
#[cfg(test)]
pub fn should_ocr_page(mode: OcrMode, page: &Page) -> bool {
    match mode {
        OcrMode::Off => false,
        OcrMode::On => true,
        OcrMode::Auto => is_low_text_page(page),
    }
}

/// Parse an `--ocr-lang` spec (`"eng,chi_sim,jpn"`) into a cleaned, ordered list:
/// whitespace trimmed, empties dropped, duplicates removed (first occurrence
/// wins, so the order the user wrote is preserved).
pub fn parse_langs(spec: &str) -> Vec<String> {
    let mut langs: Vec<String> = Vec::new();
    for raw in spec.split(',') {
        let lang = raw.trim();
        if lang.is_empty() {
            continue;
        }
        if !langs.iter().any(|existing| existing == lang) {
            langs.push(lang.to_string());
        }
    }
    langs
}

/// The catalog id of the `.traineddata` model for a Tesseract language code
/// (`"eng"` → `"tessdata-eng"`). The `tessdata-*` naming is the convention the
/// [`kopitiam_models`] catalog uses for its LSTM OCR entries.
pub fn model_id_for_lang(lang: &str) -> String {
    format!("tessdata-{lang}")
}

/// Every `tessdata-*` language code the built-in catalog knows, for a helpful
/// error when an unknown `--ocr-lang` is requested.
fn known_ocr_langs() -> Vec<String> {
    Catalog::builtin()
        .into_iter()
        .filter_map(|spec| spec.id.strip_prefix("tessdata-").map(str::to_string))
        .collect()
}

/// Resolve each requested language to its catalog [`ModelSpec`]. Pure catalog
/// lookup — no disk, no network — so it is unit-testable without any model
/// present. An unknown language is a clear error listing the supported codes.
pub fn resolve_lang_specs(langs: &[String]) -> anyhow::Result<Vec<(String, ModelSpec)>> {
    if langs.is_empty() {
        anyhow::bail!(
            "no OCR languages given: pass --ocr-lang with at least one of {}",
            known_ocr_langs().join(", ")
        );
    }
    let mut resolved = Vec::with_capacity(langs.len());
    for lang in langs {
        let id = model_id_for_lang(lang);
        match Catalog::find(&id) {
            Some(spec) => resolved.push((lang.clone(), spec)),
            None => anyhow::bail!(
                "no OCR model for language '{lang}' (looked for catalog id '{id}'). \
                 Supported languages: {}",
                known_ocr_langs().join(", ")
            ),
        }
    }
    Ok(resolved)
}

/// Lay recognized line texts back onto a page as ordinary [`TextSpan`]s.
///
/// Line-finding returns lines in reading order but *not* their page coordinates
/// (a [`GrayLine`] carries only pixels), so geometry is synthesized: each line
/// is stacked at a uniform left margin with a monotonically increasing baseline
/// and a uniform [`OCR_FONT_SIZE_PT`]. That is enough for the pre-ordered
/// reconstruction path, which trusts span order and only reads the font size (to
/// tell body text from headings — and OCR text is uniformly body text). The
/// recognized text itself is verbatim: this splices real content, never
/// scaffolding, keeping the recovery ratio honest (§2.1).
pub fn ocr_lines_to_spans(lines: &[String]) -> Vec<TextSpan> {
    let mut spans = Vec::with_capacity(lines.len());
    let mut y = 0.0f32;
    for raw in lines {
        let text = raw.trim();
        if text.is_empty() {
            continue;
        }
        // A rough width so the span has a non-degenerate box; the pre-ordered
        // reconstruction does not depend on it, but a plausible extent keeps any
        // geometry-aware consumer well-behaved.
        let width = text.chars().count() as f32 * (OCR_FONT_SIZE_PT * 0.5);
        spans.push(TextSpan {
            text: text.to_string(),
            x: 0.0,
            y,
            width,
            height: OCR_LINE_HEIGHT_PT,
            font_size: OCR_FONT_SIZE_PT,
            ..TextSpan::default()
        });
        y += OCR_LINE_HEIGHT_PT;
    }
    spans
}

/// Apply the OCR fallback to `pages` in place, driving `ocr` for each page that
/// [`should_ocr_page`] selects under `mode`. Each recognized page has its spans
/// replaced by the OCR'd spans (so downstream reconstruction sees them as normal
/// text); a page that recognizes to nothing is left untouched.
///
/// This is the testable core: with a fake [`PageOcr`] it exercises the trigger,
/// mode, and merge without any real OCR.
///
/// **Test-only convenience.** Production goes through [`run_ocr_fallback`],
/// which routes once and reuses the plan; this wrapper exists so the many
/// trigger/mode/merge tests can stay written in terms of a mode rather than
/// having to build a plan by hand.
#[cfg(test)]
pub fn apply_ocr_fallback(
    pages: &mut [Page],
    mode: OcrMode,
    ocr: &dyn PageOcr,
) -> anyhow::Result<OcrSummary> {
    let plan = HeuristicRouter.route(pages, mode);
    apply_route_plan(pages, &plan, ocr)
}

/// Applies an already-computed [`RoutePlan`]: OCR exactly the pages it routed to
/// [`Engine::Tesseract`], leave the rest to their text layer.
///
/// Taking the plan as a parameter (rather than deciding inline) is what lets the
/// routing decision be inspected, diffed and eventually made by a smarter
/// router, without this function changing at all.
///
/// # Two different "page indexes" meet here — do not conflate them
///
/// * `RoutePlan` dispatches are keyed by **position in the `pages` slice**,
///   because that is all a router is given.
/// * The recognizer addresses pages by **document index**, derived from
///   `page.number - 1`.
///
/// These coincide only when `pages` is the whole document in order. With a
/// subset — `pdf2md --pages 5-9` — slice position 0 is document page 4, and
/// using one where the other belongs would silently OCR (and splice) the WRONG
/// page: no error, just text from elsewhere in the document appearing on a page
/// that never had it. So the plan is consulted by slice position, and the
/// recognizer is always called with `page.number - 1`.
pub fn apply_route_plan(
    pages: &mut [Page],
    plan: &RoutePlan,
    ocr: &dyn PageOcr,
) -> anyhow::Result<OcrSummary> {
    let mut summary = OcrSummary::default();
    for (slot, page) in pages.iter_mut().enumerate() {
        // A page with no dispatch is left alone: a router that declined to say
        // anything about a page has not asked for it to be rewritten.
        if plan.dispatches.get(slot).map(|d| d.engine) != Some(Engine::Tesseract) {
            continue;
        }
        // The rasterizer indexes pages 0-based; the mupdf extractor numbers them
        // 1-based sequentially, so `number - 1` is the document page index.
        let page_index = page.number.saturating_sub(1);
        let lines = ocr.recognize_page(page_index)?;
        let spans = ocr_lines_to_spans(&lines);
        if spans.is_empty() {
            continue;
        }
        summary.pages_ocred += 1;
        summary.lines_recognized += spans.len();
        page.spans = spans;
    }
    Ok(summary)
}

/// Run the automatic OCR fallback over `pages` extracted from `input`.
///
/// Fast-exits (touching neither the model store nor the rasterizer) when the
/// mode is `Off` or when no page qualifies — so a born-digital PDF never pays any
/// OCR cost and its output stays byte-identical to the pre-OCR pipeline. Only
/// when at least one page qualifies does it resolve the languages' models, load
/// the recognizers, open the document, and recognize.
///
/// # Models absent
///
/// If a requested language's `.traineddata` is not in the local model store,
/// this returns a clear, actionable error naming the `kopitiam models pull …`
/// command (via [`TesseractPageOcr::new`]) rather than panicking. Auto-pull is
/// deliberately *not* done: silently downloading hundreds of megabytes mid-
/// conversion would be a surprising side effect, so — matching how
/// `kopitiam models path` behaves — the user is pointed at the explicit `pull`.
pub fn run_ocr_fallback(
    input: &Path,
    pages: &mut [Page],
    mode: OcrMode,
    langs: &[String],
) -> anyhow::Result<OcrSummary> {
    if mode == OcrMode::Off {
        return Ok(OcrSummary::default());
    }
    // Route once, up front, and reuse the same plan for the early-exit check and
    // the work itself — deciding twice invites the two answers to drift.
    let plan = HeuristicRouter.route(pages, mode);
    if plan.pages_for(Engine::Tesseract).is_empty() {
        // No scanned page: never load models or rasterize (no cost, and the
        // output is identical to the pre-OCR pipeline).
        return Ok(OcrSummary::default());
    }
    let store = ModelStore::with_default_root()
        .context("could not locate the local model store for OCR")?;
    let recognizer = TesseractPageOcr::new(input, langs, &store)?;
    apply_route_plan(pages, &plan, &recognizer)
}

/// The real recognizer: an opened [`PdfDocument`] plus one loaded
/// [`LstmRecognizer`] per configured language.
pub struct TesseractPageOcr {
    doc: PdfDocument,
    /// One recognizer per language, in the configured order.
    recognizers: Vec<LstmRecognizer>,
    dpi: f32,
}

impl TesseractPageOcr {
    /// Build the recognizer: resolve each language to its catalog model, verify it
    /// is present in `store` (else a clear "run `kopitiam models pull …`" error),
    /// load the LSTM recognizer from its `.traineddata`, and open `input` for
    /// rasterization.
    pub fn new(input: &Path, langs: &[String], store: &ModelStore) -> anyhow::Result<Self> {
        let specs = resolve_lang_specs(langs)?;
        let mut recognizers = Vec::with_capacity(specs.len());
        for (lang, spec) in &specs {
            let artifact = spec.artifacts.first().ok_or_else(|| {
                anyhow::anyhow!("OCR model '{}' has no artifact to load", spec.id)
            })?;
            if !store.is_present(spec) {
                anyhow::bail!(
                    "OCR model for language '{lang}' is not in the local model store. \
                     Run `kopitiam models pull {}` to fetch it (or drop `{}` at the path \
                     `kopitiam models path {}` prints).",
                    spec.id,
                    artifact.filename,
                    spec.id
                );
            }
            let path = store.artifact_path(spec, artifact);
            let bytes = std::fs::read(&path)
                .with_context(|| format!("could not read OCR model {}", path.display()))?;
            // `TessdataManager`/`LstmRecognizer::load` copy what they need into
            // owned structures, so the borrowed bytes can be dropped after load.
            let mgr = TessdataManager::from_bytes(&bytes).map_err(|e| {
                anyhow::anyhow!("could not parse OCR model {}: {e}", path.display())
            })?;
            let recognizer = LstmRecognizer::load(&mgr).map_err(|e| {
                anyhow::anyhow!("could not load OCR recognizer for '{lang}': {e}")
            })?;
            recognizers.push(recognizer);
        }

        let bytes = std::fs::read(input)
            .with_context(|| format!("could not read {} for OCR", input.display()))?;
        let doc = PdfDocument::open(bytes)
            .map_err(|e| anyhow::anyhow!("could not open {} for OCR: {e}", input.display()))?;

        Ok(TesseractPageOcr {
            doc,
            recognizers,
            dpi: OCR_DPI,
        })
    }
}

impl PageOcr for TesseractPageOcr {
    fn recognize_page(&self, page_index: usize) -> anyhow::Result<Vec<String>> {
        // Rasterize the whole page (rather than decoding a single embedded image
        // via `page_full_image`): rasterization handles every scanned-page shape
        // uniformly — a single full-page image, several stacked image strips, or
        // an image-plus-vector overlay — at a controlled DPI, whereas the direct
        // image-decode path only helps the exact single-full-page-image case.
        let pixmap = rasterize_page(&self.doc, page_index, self.dpi)
            .map_err(|e| anyhow::anyhow!("could not rasterize page {page_index} for OCR: {e}"))?;
        let gray = pixmap_to_gray(&pixmap);
        let binary = otsu_binarize(&gray);
        let lines = find_text_lines(&binary, &gray);

        let mut out = Vec::with_capacity(lines.len());
        for line in &lines {
            let text = best_recognition(&self.recognizers, line)?;
            if !text.trim().is_empty() {
                out.push(text);
            }
        }
        Ok(out)
    }
}

/// Recognize one line with each language's model and keep the result with the
/// most non-whitespace characters.
///
/// This is a pragmatic multi-language heuristic, documented as such: the port's
/// recognizer exposes only recognized text (no per-glyph confidence), so — absent
/// real per-line script detection — "the model that read the most characters
/// wins" is a reasonable proxy. It favors the correct script on dense CJK text (a
/// Chinese line run through the English model yields a handful of characters; run
/// through `chi_sim` it yields many). Full script routing is deferred.
fn best_recognition(recognizers: &[LstmRecognizer], line: &GrayLine) -> anyhow::Result<String> {
    let mut best = String::new();
    let mut best_score = 0usize;
    for recognizer in recognizers {
        let text = recognizer
            .recognize_line(line)
            .map_err(|e| anyhow::anyhow!("OCR recognition failed: {e}"))?;
        let score = text.chars().filter(|c| !c.is_whitespace()).count();
        if score > best_score {
            best_score = score;
            best = text;
        }
    }
    Ok(best)
}

/// Convert a rasterized [`Pixmap`] (DeviceGray or DeviceRGB) to the OCR engine's
/// [`GrayImage`], via the pixmap's Rec. 601 luma.
fn pixmap_to_gray(pixmap: &Pixmap) -> GrayImage {
    let w = pixmap.w as usize;
    let h = pixmap.h as usize;
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            // White (255) for any out-of-bounds read, which cannot happen here.
            pixels.push(pixmap.luma(x as i32, y as i32).unwrap_or(255));
        }
    }
    GrayImage::new(w, h, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A page with `spans` built from `(text)` lines, numbered `number`.
    fn page_with_texts(number: usize, texts: &[&str]) -> Page {
        let spans = texts
            .iter()
            .map(|t| TextSpan {
                text: (*t).to_string(),
                ..TextSpan::default()
            })
            .collect();
        Page {
            number,
            width: 612.0,
            height: 792.0,
            spans,
        }
    }

    /// A fake recognizer returning canned lines, recording the page indices it saw.
    struct FakeOcr {
        lines: Vec<String>,
        calls: RefCell<Vec<usize>>,
    }

    impl FakeOcr {
        fn new(lines: &[&str]) -> Self {
            FakeOcr {
                lines: lines.iter().map(|s| (*s).to_string()).collect(),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl PageOcr for FakeOcr {
        fn recognize_page(&self, page_index: usize) -> anyhow::Result<Vec<String>> {
            self.calls.borrow_mut().push(page_index);
            Ok(self.lines.clone())
        }
    }

    // ---- trigger (low-text detection) --------------------------------------

    #[test]
    fn trigger_fires_on_a_near_empty_page_not_a_text_rich_one() {
        // A scanned page: no spans at all -> zero non-whitespace chars -> triggers.
        let scanned = page_with_texts(1, &[]);
        assert!(is_low_text_page(&scanned));

        // A page with only a stray glyph or two is still below threshold.
        let almost_empty = page_with_texts(1, &["·", "  "]);
        assert!(is_low_text_page(&almost_empty));

        // A real text page (a sentence) is well above threshold -> never triggers.
        let text_rich = page_with_texts(
            2,
            &["The reactor coolant pump seal package was inspected in full."],
        );
        assert!(!is_low_text_page(&text_rich));
    }

    #[test]
    fn should_ocr_page_honors_the_mode() {
        let scanned = page_with_texts(1, &[]);
        let text_rich = page_with_texts(2, &["A whole line of genuine extracted prose here."]);

        // Off: never.
        assert!(!should_ocr_page(OcrMode::Off, &scanned));
        assert!(!should_ocr_page(OcrMode::Off, &text_rich));
        // On: always, even a text page.
        assert!(should_ocr_page(OcrMode::On, &scanned));
        assert!(should_ocr_page(OcrMode::On, &text_rich));
        // Auto: only the scanned page.
        assert!(should_ocr_page(OcrMode::Auto, &scanned));
        assert!(!should_ocr_page(OcrMode::Auto, &text_rich));
    }

    // ---- flag parsing ------------------------------------------------------

    #[test]
    fn ocr_mode_defaults_to_auto() {
        assert_eq!(OcrMode::default(), OcrMode::Auto);
    }

    #[test]
    fn parse_langs_trims_dedups_and_drops_empties() {
        assert_eq!(
            parse_langs("eng,chi_sim,jpn"),
            vec!["eng", "chi_sim", "jpn"]
        );
        // Whitespace tolerated, empties and duplicates dropped, order preserved.
        assert_eq!(
            parse_langs("  eng , , chi_sim ,eng "),
            vec!["eng", "chi_sim"]
        );
        assert!(parse_langs("").is_empty());
        assert!(parse_langs(" , ,").is_empty());
    }

    #[test]
    fn default_langs_parse_to_the_three_driving_languages() {
        assert_eq!(parse_langs(DEFAULT_OCR_LANGS), vec!["eng", "chi_sim", "jpn"]);
    }

    // ---- language -> model-spec resolution ---------------------------------

    #[test]
    fn resolve_lang_specs_maps_known_languages_to_tessdata_ids() {
        let specs = resolve_lang_specs(&parse_langs("eng,chi_sim,jpn")).unwrap();
        let ids: Vec<&str> = specs.iter().map(|(_, s)| s.id.as_str()).collect();
        assert_eq!(ids, vec!["tessdata-eng", "tessdata-chi_sim", "tessdata-jpn"]);
        // The language label is carried through alongside the spec, in order.
        assert_eq!(specs[1].0, "chi_sim");
    }

    #[test]
    fn resolve_lang_specs_rejects_an_unknown_language() {
        let err = resolve_lang_specs(&parse_langs("klingon")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("klingon"), "message was: {msg}");
        assert!(msg.contains("tessdata-klingon"), "message was: {msg}");
        // Empty language list is also an error.
        assert!(resolve_lang_specs(&[]).is_err());
    }

    #[test]
    fn model_id_for_lang_uses_the_tessdata_prefix() {
        assert_eq!(model_id_for_lang("eng"), "tessdata-eng");
        assert_eq!(model_id_for_lang("chi_sim"), "tessdata-chi_sim");
    }

    // ---- spans -> Document merge -------------------------------------------

    #[test]
    fn ocr_lines_to_spans_stacks_lines_as_uniform_body_text() {
        let spans = ocr_lines_to_spans(&[
            "First recognized line.".to_string(),
            "  ".to_string(), // blank -> dropped
            "Second recognized line.".to_string(),
        ]);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].text, "First recognized line.");
        assert_eq!(spans[1].text, "Second recognized line.");
        // Uniform font size (no invented headings) and a monotonically increasing
        // baseline so reading order is preserved.
        assert_eq!(spans[0].font_size, OCR_FONT_SIZE_PT);
        assert_eq!(spans[1].font_size, OCR_FONT_SIZE_PT);
        assert!(spans[1].y > spans[0].y);
    }

    #[test]
    fn apply_ocr_fallback_auto_replaces_only_the_scanned_page() {
        let mut pages = vec![
            page_with_texts(1, &[]), // scanned -> will be OCR'd
            page_with_texts(2, &["Genuine extracted prose that must be left alone."]),
        ];
        let fake = FakeOcr::new(&["Recognized page one text."]);

        let summary = apply_ocr_fallback(&mut pages, OcrMode::Auto, &fake).unwrap();

        assert_eq!(summary.pages_ocred, 1);
        assert_eq!(summary.lines_recognized, 1);
        // Page 1 (index 0) was rasterized; only it was recognized.
        assert_eq!(*fake.calls.borrow(), vec![0]);
        // The scanned page now carries the recognized text...
        assert_eq!(pages[0].spans.len(), 1);
        assert_eq!(pages[0].spans[0].text, "Recognized page one text.");
        // ...and the text-rich page is byte-for-byte untouched.
        assert_eq!(
            pages[1].spans[0].text,
            "Genuine extracted prose that must be left alone."
        );
    }

    #[test]
    fn apply_ocr_fallback_off_is_a_no_op_and_on_hits_every_page() {
        let base = || {
            vec![
                page_with_texts(1, &[]),
                page_with_texts(2, &["Real text on page two."]),
            ]
        };

        // Off: nothing recognized, spans unchanged.
        let mut pages = base();
        let fake = FakeOcr::new(&["x"]);
        let summary = apply_ocr_fallback(&mut pages, OcrMode::Off, &fake).unwrap();
        assert_eq!(summary.pages_ocred, 0);
        assert!(fake.calls.borrow().is_empty());
        assert!(pages[0].spans.is_empty());

        // On: both pages recognized (page 2 numbered 2 -> index 1).
        let mut pages = base();
        let fake = FakeOcr::new(&["forced"]);
        let summary = apply_ocr_fallback(&mut pages, OcrMode::On, &fake).unwrap();
        assert_eq!(summary.pages_ocred, 2);
        assert_eq!(*fake.calls.borrow(), vec![0, 1]);
        assert_eq!(pages[1].spans[0].text, "forced");
    }

    #[test]
    fn ocr_output_flows_through_the_document_reconstruction_path() {
        // The whole point: OCR spans feed the SAME reconstruction + render path,
        // so recognized text lands in the Markdown like any extracted text.
        let mut pages = vec![page_with_texts(1, &[])];
        let fake = FakeOcr::new(&[
            "Containment integrated leak rate test summary.",
            "The measured leakage was within the technical specification limit.",
        ]);
        apply_ocr_fallback(&mut pages, OcrMode::Auto, &fake).unwrap();

        let document = kopitiam_document::reconstruct_preordered(&pages);
        let markdown = kopitiam_markdown::render_document(&document);
        assert!(markdown.contains("Containment integrated leak rate test summary."));
        assert!(markdown.contains("within the technical specification limit."));

        // §2.1 honesty: the recognized text is measured as real extracted content,
        // so the recovery ratio stays sane (no scaffolding inflates it above 1.0).
        let report = kopitiam_document::validate(&pages, &document, &markdown);
        assert!(report.recovery_ratio() <= 1.0 + 1e-9);
    }

    // ---- live end-to-end (needs a real model + a scanned fixture) ----------

    /// End-to-end OCR over a real scanned PDF with a downloaded `.traineddata`.
    /// Ignored by default (the model is a heavy download and CI has no fixture).
    ///
    /// Run it with:
    /// ```text
    /// kopitiam models pull tessdata-eng
    /// KOPITIAM_OCR_FIXTURE=/path/to/scanned.pdf \
    ///   cargo test -p kopitiam -- --ignored --nocapture \
    ///   ocr_fallback::tests::live_ocr_recognizes_a_scanned_page
    /// ```
    #[test]
    #[ignore = "needs `kopitiam models pull tessdata-eng` + KOPITIAM_OCR_FIXTURE=<scanned.pdf>"]
    fn live_ocr_recognizes_a_scanned_page() {
        let fixture = std::env::var("KOPITIAM_OCR_FIXTURE")
            .expect("set KOPITIAM_OCR_FIXTURE to a scanned PDF path");
        let path = Path::new(&fixture);
        let mut pages = kopitiam_pdf::extract_mupdf(path).expect("extract");
        let langs = parse_langs("eng");
        let summary = run_ocr_fallback(path, &mut pages, OcrMode::Auto, &langs).expect("ocr");
        assert!(
            summary.pages_ocred > 0,
            "expected at least one scanned page to be OCR'd"
        );
    }
}
