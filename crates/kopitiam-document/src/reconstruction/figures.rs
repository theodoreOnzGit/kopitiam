//! Figure-caption detection **and** figure-region collapsing.
//!
//! # The problem this closes (`kopitiam_token_max.md` §7 card I-D)
//!
//! [`try_figure`] only ever matched a single line that *begins* `Figure N` /
//! `Fig. N`. Architecture and flow diagrams, however, are vector art whose box
//! labels are **real PDF text**: "Data Ingestion", "Control Unit", "Sensor
//! Array", scattered across the page at wildly different x positions. Each label
//! became its own one-line `Paragraph`, so a single diagram exploded into
//! hundreds of near-zero-information short lines -- measured at ~44% of the
//! tokens spent reading a diagram-heavy chapter. The renderer then piled on a
//! 37-char `[Figure omitted ...]` placeholder per figure.
//!
//! [`collapse_figure_regions`] runs as a page-level pass (ahead of column
//! splitting and `consume_paragraph`) that recognises a run of scattered short
//! label lines anchored to a real `Fig. N` caption and drops the labels,
//! keeping the caption alone. The caption span survives untouched, so it still
//! reaches [`try_figure`] downstream and becomes a `Figure` block exactly as
//! before -- only the label soup is gone.
//!
//! # Approach (clean-room study of two references, credited in the report /
//! `docs/ACKNOWLEDGEMENTS.md`)
//!
//! The "drop the lines that fall inside a figure's region" idea is from
//! pdf-to-markdown's `pdf2md/regions.py` (`drop_lines_in_boxes`, MIT) -- but we
//! have no image bounding boxes (the extraction layer recovers text spans
//! only), so the region is inferred purely from text geometry rather than read
//! from a detected image box. marker's `processors/ignoretext.py` and
//! `processors/marginalia.py` (Apache-2.0) were studied for the guard-rail
//! philosophy: be conservative, combine signals, never let one heuristic act
//! alone. This is original Rust, not a port.
//!
//! # Precision over recall, and the safer default
//!
//! Wrongly swallowing a paragraph of prose is far worse than leaving label soup
//! behind, so collapsing fires only when **every** signal agrees:
//!
//! 1. a run of consecutive *short* lines (< 5 words) with no terminal
//!    sentence punctuation and no sentence-like structure;
//! 2. that run is **spatially anchored to a matched `Fig. N` / `Figure N`
//!    caption** (the strongest single signal -- a scattered short-line run with
//!    no caption is left completely alone; that is the "safer default", and the
//!    more aggressive caption-less collapse is deliberately *not* enabled);
//! 3. the run's left edges are **scattered** across the page
//!    ([`MIN_X_SCATTER_FRACTION`] of page width), which a left-aligned prose
//!    list can never satisfy;
//! 4. the run is at least [`MIN_LABEL_LINES`] lines long.
//!
//! A terse list of full sentences (short, but terminally punctuated and
//! left-aligned, and with no caption) fails on three of the four and is never
//! touched. See the unit tests.
//!
//! # Keeping the recovery ratio honest (`kopitiam_token_max.md` §2.1)
//!
//! Dropping the label spans lowers `rendered_chars`; if the extracted side
//! still counted them, every diagram-bearing document's `recovery_ratio` would
//! sink below the 0.98 PASS threshold. [`collapse_figure_regions`] is therefore
//! a **pure, deterministic** function of the input pages, exactly like
//! `headers::strip_marginalia`, and `validation::validate` runs the *same*
//! function over the *same* pages (after the same `strip_marginalia`) before
//! counting `extracted_chars`. Because both sides drop byte-for-byte the same
//! label spans, the dropped labels are absent from `extracted_chars` and
//! `rendered_chars` alike, and the ratio measures only real body recovery --
//! staying honestly inside `[0.98, 1.0]`.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use kopitiam_pdf::{Page, TextSpan};

use super::{SAME_LINE_Y_TOLERANCE_RATIO, Line, build_line};
use crate::Figure;

static FIGURE_CAPTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(figure|fig\.)\s*\d+").unwrap());

/// A line with more words than this is prose, never a diagram label. "< 5
/// words" from the card, i.e. at most four.
const MAX_LABEL_WORDS: usize = 4;

/// A caption-anchored run must be at least this many label lines before it is
/// collapsed. A captioned photo with a one-line description (or a caption with
/// no internal labels at all) never reaches this and is left completely alone.
const MIN_LABEL_LINES: usize = 4;

/// The label run's left edges must span at least this fraction of the page
/// width to count as "scattered". A left-aligned prose list has ~zero spread
/// and can never satisfy this -- it is the signal that most cleanly separates
/// diagram labels from a terse vertical list of real text.
const MIN_X_SCATTER_FRACTION: f32 = 0.10;

/// Images themselves are not extracted (the extraction layer only recovers
/// text spans) -- only the caption is preserved.
pub(super) fn try_figure(line: &Line) -> Option<Figure> {
    let text = line.text.trim();
    if FIGURE_CAPTION.is_match(text) {
        Some(Figure {
            caption: Some(text.to_string()),
            image_path: None,
        })
    } else {
        None
    }
}

/// One visual line on a page, carrying everything the figure-region heuristic
/// needs plus the original span indices so exactly the right spans can be
/// dropped.
struct FigureLine {
    span_indices: Vec<usize>,
    /// Whether this line is a `Fig. N` / `Figure N` caption (an anchor, never
    /// itself dropped).
    is_caption: bool,
    /// Whether this line looks like a scattered diagram label (short, no
    /// terminal punctuation) -- a *candidate* for dropping, gated by the
    /// region-level checks in [`collapse_page`].
    is_label_like: bool,
    /// Left edge of the line (min span x), for the x-scatter test.
    left_x: f32,
}

/// Returns a copy of `pages` with scattered figure-label spans removed,
/// preserving original span order within each page (the pre-ordered
/// reconstruction path trusts that order). Caption spans, and every span
/// outside a collapsed figure region, are kept untouched.
///
/// Pure and deterministic in `pages`: the same input always yields the same
/// output, which is what lets `validation::validate` re-run it to keep the
/// recovery ratio honest (see module docs and `kopitiam_token_max.md` §2.1).
pub(crate) fn collapse_figure_regions(pages: &[Page]) -> Vec<Page> {
    pages.iter().map(collapse_page).collect()
}

fn collapse_page(page: &Page) -> Page {
    let lines = figure_lines(page);
    let drop = labels_to_drop(&lines, page.width);

    if drop.is_empty() {
        return page.clone();
    }

    let spans = page
        .spans
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, span)| span.clone())
        .collect();

    Page {
        number: page.number,
        width: page.width,
        height: page.height,
        spans,
    }
}

/// The set of span indices belonging to label lines in a collapsible figure
/// region. Empty unless every precision signal agrees (see module docs).
fn labels_to_drop(lines: &[FigureLine], page_width: f32) -> HashSet<usize> {
    let mut drop = HashSet::new();

    for (c, line) in lines.iter().enumerate() {
        if !line.is_caption {
            continue;
        }
        // Expand outward from the caption in both directions, collecting the
        // maximal contiguous run of label-like lines touching it. Diagram
        // labels sit directly above (caption below the figure) or directly
        // below (caption above it); anything non-label -- a heading, a real
        // paragraph line -- breaks the run and bounds the region, which is
        // what keeps prose out.
        let mut region: Vec<&FigureLine> = Vec::new();

        let mut k = c;
        while k > 0 && lines[k - 1].is_label_like {
            k -= 1;
            region.push(&lines[k]);
        }
        let mut k = c;
        while k + 1 < lines.len() && lines[k + 1].is_label_like {
            k += 1;
            region.push(&lines[k]);
        }

        if region.len() < MIN_LABEL_LINES {
            continue; // not enough scattered labels to be a diagram
        }

        // x-scatter: a genuine label cloud spans a wide horizontal range; a
        // left-aligned vertical list of short prose does not.
        let min_x = region.iter().map(|l| l.left_x).fold(f32::INFINITY, f32::min);
        let max_x = region
            .iter()
            .map(|l| l.left_x)
            .fold(f32::NEG_INFINITY, f32::max);
        if (max_x - min_x) < page_width * MIN_X_SCATTER_FRACTION {
            continue; // too neatly aligned to be a scattered diagram
        }

        for label in region {
            drop.extend(&label.span_indices);
        }
    }

    drop
}

/// Groups a page's spans into visual lines (shared baseline), preserving each
/// span's original index, and classifies each line. Mirrors
/// `reconstruction::group_spans_by_baseline` but keeps span indices, exactly as
/// `headers::zone_lines` does for the marginalia pass.
fn figure_lines(page: &Page) -> Vec<FigureLine> {
    let mut ordered: Vec<(usize, &TextSpan)> = page.spans.iter().enumerate().collect();
    ordered.sort_by(|a, b| b.1.y.partial_cmp(&a.1.y).unwrap_or(Ordering::Equal));

    let mut groups: Vec<Vec<(usize, &TextSpan)>> = Vec::new();
    for (i, span) in ordered {
        let joins_last = groups.last().is_some_and(|group| {
            let anchor = group[0].1;
            let tolerance = anchor.font_size.max(span.font_size) * SAME_LINE_Y_TOLERANCE_RATIO;
            (anchor.y - span.y).abs() <= tolerance
        });
        if joins_last {
            groups.last_mut().unwrap().push((i, span));
        } else {
            groups.push(vec![(i, span)]);
        }
    }

    groups
        .into_iter()
        .map(|mut group| {
            group.sort_by(|a, b| a.1.x.partial_cmp(&b.1.x).unwrap_or(Ordering::Equal));
            let refs: Vec<&TextSpan> = group.iter().map(|(_, span)| *span).collect();
            let text = build_line(&refs).text;
            let trimmed = text.trim();
            let is_caption = FIGURE_CAPTION.is_match(trimmed);
            FigureLine {
                span_indices: group.iter().map(|(i, _)| *i).collect(),
                is_caption,
                is_label_like: !is_caption && is_label_like(trimmed),
                left_x: group
                    .iter()
                    .map(|(_, span)| span.x)
                    .fold(f32::INFINITY, f32::min),
            }
        })
        .collect()
}

/// A line is label-like when it is short (< 5 words) and shows no sentence-like
/// structure -- no terminal sentence punctuation. Diagram labels ("Data
/// Ingestion Layer", "PID Controller") pass; a real sentence -- however short
/// -- ends in `.`/`!`/`?` and does not.
fn is_label_like(trimmed: &str) -> bool {
    if trimmed.is_empty() {
        return false;
    }
    let words = trimmed.split_whitespace().count();
    if words == 0 || words > MAX_LABEL_WORDS {
        return false;
    }
    !matches!(trimmed.chars().last(), Some('.' | '!' | '?'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, x: f32, y: f32, width: f32, font_size: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            x,
            y,
            width,
            height: font_size,
            font_size,
            font_name: None,
            ..TextSpan::default()
        }
    }

    fn texts(page: &Page) -> Vec<&str> {
        page.spans.iter().map(|s| s.text.as_str()).collect()
    }

    /// A synthetic architecture diagram: a caption plus ~30 short box labels
    /// scattered across the page at widely varying x positions, none ending in
    /// terminal punctuation -- exactly the vector-art-label archetype the card
    /// describes.
    fn diagram_page() -> Page {
        let labels = [
            "Sensor Array", "Data Ingestion", "Message Queue", "Stream Processor",
            "Control Unit", "PID Controller", "Actuator Bank", "Feedback Loop",
            "State Store", "Cache Layer", "Load Balancer", "API Gateway",
            "Auth Service", "Audit Log", "Metrics Sink", "Alert Manager",
            "Config Server", "Service Mesh", "Edge Node", "Core Cluster",
            "Batch Worker", "Scheduler", "Dispatch Table", "Retry Buffer",
            "Health Probe", "Trace Collector", "Span Exporter", "Model Registry",
            "Inference Pod", "Result Cache",
        ];
        let mut spans = Vec::new();
        // Scatter labels across the page: x cycles widely (50..500), y descends.
        for (i, label) in labels.iter().enumerate() {
            let x = 50.0 + ((i * 97) % 450) as f32; // wide, varied left edges
            let y = 720.0 - (i as f32) * 12.0;
            spans.push(span(label, x, y, 80.0, 10.0));
        }
        // Caption directly below the label cloud.
        spans.push(span(
            "Fig. 1 System architecture of the synthetic platform",
            50.0,
            340.0,
            400.0,
            10.0,
        ));
        Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans,
        }
    }

    #[test]
    fn scattered_labels_are_dropped_and_the_caption_survives() {
        let page = diagram_page();
        let before = page.spans.len();

        let collapsed = collapse_figure_regions(&[page]);
        let remaining = texts(&collapsed[0]);

        // Only the caption survives.
        assert_eq!(
            remaining,
            vec!["Fig. 1 System architecture of the synthetic platform"],
            "every scattered label must be dropped, only the caption kept"
        );
        assert!(
            collapsed[0].spans.len() < before / 2,
            "expected >50% span reduction, went {} -> {}",
            before,
            collapsed[0].spans.len()
        );
    }

    #[test]
    fn a_terse_list_of_real_sentences_is_not_collapsed() {
        // The negative fixture (precision proof): short lines, but each is a
        // full sentence (terminal punctuation) AND left-aligned AND there is no
        // figure caption. None of the collapse signals fire.
        let sentences = [
            "The reactor is stable.",
            "Pressure holds steady.",
            "Coolant flows freely.",
            "Sensors read normal.",
            "Alarms remain silent.",
            "The shift ends calmly.",
        ];
        let mut spans = Vec::new();
        for (i, s) in sentences.iter().enumerate() {
            spans.push(span(s, 50.0, 700.0 - (i as f32) * 14.0, 300.0, 10.0));
        }
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans,
        };

        let collapsed = collapse_figure_regions(&[page]);
        assert_eq!(
            collapsed[0].spans.len(),
            sentences.len(),
            "real prose must never be collapsed: {:?}",
            texts(&collapsed[0])
        );
    }

    #[test]
    fn scattered_short_labels_without_a_caption_are_left_alone() {
        // The safer default: the same scattered short labels, but with NO
        // caption to anchor them, are NOT collapsed. Caption-anchoring is the
        // strongest signal and aggressive caption-less collapse is deliberately
        // off (precision over recall).
        let mut page = diagram_page();
        page.spans.pop(); // remove the caption
        let before = page.spans.len();

        let collapsed = collapse_figure_regions(&[page]);
        assert_eq!(
            collapsed[0].spans.len(),
            before,
            "without a caption anchor nothing may be dropped"
        );
    }

    #[test]
    fn a_left_aligned_run_of_short_labels_below_a_caption_is_not_collapsed() {
        // Even with a caption and short, unpunctuated lines, a neatly
        // left-aligned run (zero x-scatter) is not a scattered diagram and must
        // survive -- this is what stops a caption-adjacent short list of prose
        // fragments from being swallowed.
        let mut spans = vec![span("Fig. 2 A tidy caption", 50.0, 700.0, 200.0, 10.0)];
        for i in 0..6 {
            spans.push(span("Aligned label item", 50.0, 686.0 - (i as f32) * 12.0, 100.0, 10.0));
        }
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans,
        };

        let collapsed = collapse_figure_regions(&[page]);
        assert_eq!(
            collapsed[0].spans.len(),
            7,
            "a left-aligned (unscattered) run must not be collapsed: {:?}",
            texts(&collapsed[0])
        );
    }

    #[test]
    fn is_label_like_rejects_sentences_and_long_lines() {
        assert!(is_label_like("Data Ingestion Layer"));
        assert!(is_label_like("Controller"));
        assert!(!is_label_like("This is a complete sentence."));
        assert!(!is_label_like("A run of clearly more than four words here"));
        assert!(!is_label_like(""));
    }

    #[test]
    fn empty_input_is_handled() {
        assert!(collapse_figure_regions(&[]).is_empty());
    }
}
