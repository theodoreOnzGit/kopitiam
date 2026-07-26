//! Strips running heads, running feet, and bare page numbers before the rest
//! of reconstruction ever sees them.
//!
//! # The problem this closes
//!
//! Nothing downstream read [`kopitiam_pdf::Page::height`], so a line sitting in
//! a page's top or bottom margin was reconstructed like any other line: a
//! running head became its own `Paragraph`, and a bare page number became a
//! one-line `Paragraph` of its own (because `consume_paragraph` breaks on the
//! vertical gap above it). Repeated on every page, that is pure noise in the
//! Markdown output -- the maintainer's original pdf2md complaint.
//!
//! # Approach (clean-room study of two references, both credited in
//! `docs/ACKNOWLEDGEMENTS.md`)
//!
//! The signature idea is from pdf-to-markdown's `pdf2md/headers.py` (MIT): a
//! candidate line in a margin *zone* gets a signature that normalizes digit
//! runs to `#`, so `Page 1` / `Page 2` collapse to one signature; a signature
//! recurring across enough pages is a running head/foot and is dropped, and a
//! bare page number in a zone is always dropped. marker's
//! `processors/marginalia.py` and `processors/ignoretext.py` (Apache-2.0) were
//! studied for the guard rails -- be conservative on short documents, and only
//! ever touch text in the margin zone. This is original Rust, not a port.
//!
//! # Keeping the recovery ratio honest (`kopitiam_token_max.md` §2.1)
//!
//! Removing content lowers `rendered_chars`; if the extracted side still counts
//! the removed text, every document's `recovery_ratio` sinks below the 0.98
//! PASS threshold. [`strip_marginalia`] is therefore a **pure, deterministic**
//! function of the input pages, and `validation::validate` calls the *same*
//! function on the *same* pages before counting `extracted_chars`. Because both
//! sides drop byte-for-byte the same spans, the removed header/footer text is
//! absent from `extracted_chars` and `rendered_chars` alike, and the ratio
//! measures only body recovery -- staying honestly inside `[0.98, 1.0]`.
//!
//! # Orientation independence
//!
//! The two extractors disagree on the y axis: the legacy `pdf-rs` path uses
//! PDF convention (origin bottom-left, y grows upward, so a header has a *large*
//! y), while the MuPDF `stext` path (`mupdf_extract.rs`) uses top-left origin
//! (a header has a *small* y). This module never assumes which edge is "top":
//! it classifies a line as being near the `y = 0` edge or the `y = height` edge
//! and treats *both* as margin zones. Whichever way the document is oriented, a
//! running head and a running foot land in the two different edge buckets
//! consistently across every page, so their signatures still align.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;

use kopitiam_pdf::{Page, TextSpan};

use super::{SAME_LINE_Y_TOLERANCE_RATIO, build_line};

/// Fraction of the page height, measured from each edge, that counts as a
/// header/footer margin zone. A line whose y falls within this fraction of
/// either the `y = 0` or the `y = height` edge is a stripping *candidate*; body
/// text in the middle 80% is never touched.
const MARGIN_ZONE_FRACTION: f32 = 0.10;

/// A digit-normalized signature must recur on at least this fraction of the
/// document's pages to be treated as a running head/foot.
const RECURRENCE_THRESHOLD_FRACTION: f32 = 0.5;

/// Absolute floor on the recurrence count, regardless of page count: a
/// signature seen on only one or two pages is never a running head. Mirrors
/// pdf2md's `max(3, ...)`.
const MIN_RECURRENCE_COUNT: usize = 3;

/// Signature-based stripping does not engage at all below this many pages: on a
/// 1-2 page document a legitimate heading that happens to repeat (or simply
/// sits in the top zone) would look "recurring" with nothing to distinguish it
/// from a running head. The bare-page-number rule still applies at any length.
const MIN_PAGES_FOR_RECURRENCE: usize = 4;

/// A line that is *only* a page number, optionally wrapped in punctuation:
/// `12`, `- 3 -`, `[ 4 ]`. Up to four digits so a year in prose (`2024`) as a
/// standalone line is caught, but a five-plus digit run (an id, a phone number)
/// is not. Always dropped inside a margin zone, at any document length.
static BARE_PAGE_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\W*\d{1,4}\W*$").unwrap());

/// Collapses each run of digits to a single `#` so `Page 12` and `Page 13`
/// share one signature (see `_norm` in pdf2md's `headers.py`).
static DIGIT_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());

/// Which page edge a margin-zone line sits against. Deliberately *not* named
/// "top"/"bottom": see the module docs on orientation independence. All that
/// matters is that the two edges are kept distinct, so a running head and a
/// running foot with coincidentally-equal text never share a signature bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Edge {
    /// Near the `y = 0` page edge.
    NearZero,
    /// Near the `y = height` page edge.
    NearHeight,
}

/// One visual line inside a margin zone, with everything needed to decide
/// whether to strip it and, if so, which of the page's spans to remove.
struct ZoneLine {
    edge: Edge,
    /// Digit-normalized, lowercased line text -- the recurrence signature.
    signature: String,
    /// Trimmed line text, for the bare-page-number test.
    raw_text: String,
    /// Indices into `page.spans` of every span that composes this line.
    span_indices: Vec<usize>,
}

/// Returns a copy of `pages` with running-head, running-foot, and bare
/// page-number spans removed, **preserving the original span order** within
/// each page (the pre-ordered reconstruction path trusts that order, so it must
/// not be shuffled). Page count and page geometry are unchanged; only offending
/// spans are dropped.
///
/// Pure and deterministic in `pages`: the same input always yields the same
/// output, which is what lets validation re-run it to keep the recovery ratio
/// honest (see module docs and `kopitiam_token_max.md` §2.1).
pub(crate) fn strip_marginalia(pages: &[Page]) -> Vec<Page> {
    if pages.is_empty() {
        return Vec::new();
    }

    // One pass to find every margin-zone line on every page.
    let per_page: Vec<Vec<ZoneLine>> = pages.iter().map(zone_lines).collect();

    // Count how often each (edge, signature) recurs across pages.
    let mut counts: HashMap<(Edge, &str), usize> = HashMap::new();
    for lines in &per_page {
        for line in lines {
            if line.signature.is_empty() {
                continue;
            }
            *counts
                .entry((line.edge, line.signature.as_str()))
                .or_default() += 1;
        }
    }

    let recurring = recurring_signatures(&counts, pages.len());

    pages
        .iter()
        .zip(&per_page)
        .map(|(page, lines)| strip_page(page, lines, &recurring))
        .collect()
}

/// The set of `(edge, signature)` pairs frequent enough to be running
/// heads/feet. Empty on short documents, where signature stripping is disabled
/// entirely (the bare-number rule still runs in [`strip_page`]).
fn recurring_signatures(
    counts: &HashMap<(Edge, &str), usize>,
    n_pages: usize,
) -> HashSet<(Edge, String)> {
    if n_pages < MIN_PAGES_FOR_RECURRENCE {
        return HashSet::new();
    }
    let threshold =
        MIN_RECURRENCE_COUNT.max((n_pages as f32 * RECURRENCE_THRESHOLD_FRACTION).ceil() as usize);
    counts
        .iter()
        .filter(|&(_, &count)| count >= threshold)
        .map(|(&(edge, sig), _)| (edge, sig.to_string()))
        .collect()
}

/// Rebuilds one page without the spans belonging to any stripped zone line.
fn strip_page(page: &Page, lines: &[ZoneLine], recurring: &HashSet<(Edge, String)>) -> Page {
    let mut drop: HashSet<usize> = HashSet::new();
    for line in lines {
        let is_bare_number = BARE_PAGE_NUMBER.is_match(&line.raw_text);
        let is_running =
            !line.signature.is_empty() && recurring.contains(&(line.edge, line.signature.clone()));
        if is_bare_number || is_running {
            drop.extend(&line.span_indices);
        }
    }

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

/// Groups a page's margin-zone spans into visual lines, preserving each span's
/// original index so the caller can remove exactly those spans.
fn zone_lines(page: &Page) -> Vec<ZoneLine> {
    if page.height <= 0.0 {
        return Vec::new();
    }

    // Only spans inside a margin zone are candidates; keep their original index.
    let mut candidates: Vec<(usize, &TextSpan, Edge)> = page
        .spans
        .iter()
        .enumerate()
        .filter_map(|(i, span)| edge_of(span.y, page.height).map(|edge| (i, span, edge)))
        .collect();

    // Group by shared baseline, top to bottom, exactly as `group_lines` does --
    // but never merge across the two edges (a header and a footer can never
    // share a baseline anyway; this is belt-and-braces).
    candidates.sort_by(|a, b| b.1.y.partial_cmp(&a.1.y).unwrap_or(Ordering::Equal));

    let mut groups: Vec<Vec<(usize, &TextSpan, Edge)>> = Vec::new();
    for candidate in candidates {
        let (_, span, edge) = candidate;
        let joins_last = groups.last().is_some_and(|group| {
            let (_, anchor, anchor_edge) = group[0];
            let tolerance = anchor.font_size.max(span.font_size) * SAME_LINE_Y_TOLERANCE_RATIO;
            anchor_edge == edge && (anchor.y - span.y).abs() <= tolerance
        });
        if joins_last {
            groups.last_mut().unwrap().push(candidate);
        } else {
            groups.push(vec![candidate]);
        }
    }

    groups
        .into_iter()
        .map(|mut group| {
            group.sort_by(|a, b| a.1.x.partial_cmp(&b.1.x).unwrap_or(Ordering::Equal));
            let refs: Vec<&TextSpan> = group.iter().map(|(_, span, _)| *span).collect();
            let text = build_line(&refs).text;
            let raw_text = text.trim().to_string();
            ZoneLine {
                edge: group[0].2,
                signature: normalize_signature(&raw_text),
                raw_text,
                span_indices: group.iter().map(|(i, _, _)| *i).collect(),
            }
        })
        .collect()
}

/// Classifies a y coordinate into a margin zone, or `None` for body text.
/// Symmetric in the two edges so it does not care which extractor's y
/// convention produced the coordinate (see module docs).
fn edge_of(y: f32, height: f32) -> Option<Edge> {
    let fraction = y / height;
    if fraction < MARGIN_ZONE_FRACTION {
        Some(Edge::NearZero)
    } else if fraction > 1.0 - MARGIN_ZONE_FRACTION {
        Some(Edge::NearHeight)
    } else {
        None
    }
}

fn normalize_signature(trimmed_text: &str) -> String {
    DIGIT_RUN.replace_all(trimmed_text, "#").to_lowercase()
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

    /// A page laid out in PDF convention (origin bottom-left, so a header has a
    /// *large* y): running head at the top, body in the middle, page number at
    /// the bottom. `head` and `page_number` may be empty to omit them.
    fn page_with(head: &str, body: &str, page_number: &str) -> Page {
        let mut spans = Vec::new();
        if !head.is_empty() {
            // y = 770 of 800 -> fraction 0.9625, inside the top 10% zone.
            spans.push(span(head, 50.0, 770.0, 300.0, 10.0));
        }
        // y = 400 -> fraction 0.5, squarely body.
        spans.push(span(body, 50.0, 400.0, 400.0, 10.0));
        if !page_number.is_empty() {
            // y = 30 of 800 -> fraction 0.0375, inside the bottom 10% zone.
            spans.push(span(page_number, 300.0, 30.0, 20.0, 10.0));
        }
        Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans,
        }
    }

    fn texts(page: &Page) -> Vec<&str> {
        page.spans.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn six_page_running_head_and_alternating_footer_numbers_all_stripped() {
        // The archetype from kopitiam_token_max.md's acceptance criteria: a
        // running head on every page plus a page number that changes each page.
        let pages: Vec<Page> = (1..=6)
            .map(|n| {
                page_with(
                    "A Study of Synthetic Widgets",
                    &format!("Body sentence number {n} carries the real content here."),
                    &n.to_string(),
                )
            })
            .collect();

        let stripped = strip_marginalia(&pages);

        for (i, page) in stripped.iter().enumerate() {
            let remaining = texts(page);
            assert_eq!(
                remaining.len(),
                1,
                "page {} should keep only its body line, got {:?}",
                i + 1,
                remaining
            );
            assert!(
                remaining[0].starts_with("Body sentence"),
                "the surviving line must be the body, got {:?}",
                remaining[0]
            );
        }
    }

    #[test]
    fn short_document_does_not_strip_a_repeating_heading() {
        // Two pages that both open with the same heading in the top zone. That
        // is exactly what signature stripping would flag -- but with only two
        // pages there is no way to tell a running head from a real section
        // heading, so the conservative guard must leave it alone.
        let pages = vec![
            page_with("Introduction", "First body paragraph of the document.", ""),
            page_with("Introduction", "Second body paragraph of the document.", ""),
        ];

        let stripped = strip_marginalia(&pages);

        for page in &stripped {
            assert!(
                texts(page).contains(&"Introduction"),
                "a heading on a short document must not be treated as a running head: {:?}",
                texts(page)
            );
        }
    }

    #[test]
    fn bare_number_in_zone_dropped_but_body_number_kept() {
        // One page, so signature stripping is off entirely; only the always-on
        // bare-page-number rule is under test. "12" in the bottom zone is a
        // page number; the identical "12" mid-page is body content (a figure
        // count, a value) and must survive.
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![
                span(
                    "Body text with the number 12 inside it.",
                    50.0,
                    400.0,
                    400.0,
                    10.0,
                ),
                span("12", 300.0, 400.0, 20.0, 10.0), // bare "12" but mid-page
                span("12", 300.0, 30.0, 20.0, 10.0),  // bare "12" in bottom zone
            ],
        };

        let stripped = strip_marginalia(&[page]);
        let remaining = texts(&stripped[0]);

        // The mid-page prose and the mid-page bare "12" survive; exactly one
        // span (the zone page number) is removed.
        assert_eq!(
            remaining.len(),
            2,
            "only the zone page number should go: {remaining:?}"
        );
        assert!(remaining.contains(&"Body text with the number 12 inside it."));
        assert!(remaining.contains(&"12"), "the mid-page 12 must be kept");
    }

    #[test]
    fn punctuation_wrapped_page_number_is_stripped() {
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![
                span("Real body content on the page.", 50.0, 400.0, 400.0, 10.0),
                span("- 7 -", 290.0, 25.0, 30.0, 10.0),
            ],
        };
        let stripped = strip_marginalia(&[page]);
        assert_eq!(texts(&stripped[0]), vec!["Real body content on the page."]);
    }

    #[test]
    fn works_with_top_left_origin_orientation() {
        // The MuPDF stext path reports a header with a *small* y (top-left
        // origin). Same six-page archetype, flipped: head at y = 30, footer
        // number at y = 770. Stripping must be just as complete.
        let pages: Vec<Page> = (1..=6)
            .map(|n| Page {
                number: n,
                width: 600.0,
                height: 800.0,
                spans: vec![
                    span("Running Head Flipped", 50.0, 30.0, 300.0, 10.0),
                    span(
                        "Body content line goes here as prose.",
                        50.0,
                        400.0,
                        400.0,
                        10.0,
                    ),
                    span(&n.to_string(), 300.0, 770.0, 20.0, 10.0),
                ],
            })
            .collect();

        let stripped = strip_marginalia(&pages);
        for page in &stripped {
            assert_eq!(
                texts(page),
                vec!["Body content line goes here as prose."],
                "orientation must not affect stripping"
            );
        }
    }

    #[test]
    fn digit_normalized_signature_collapses_page_numbered_running_heads() {
        // A running foot of the form "Chapter 3 - 45" whose page number climbs
        // every page: the digit-normalization is what makes these one signature.
        let pages: Vec<Page> = (1..=5)
            .map(|n| Page {
                number: n,
                width: 600.0,
                height: 800.0,
                spans: vec![
                    span(
                        "Body prose for this page of the report.",
                        50.0,
                        400.0,
                        400.0,
                        10.0,
                    ),
                    span(&format!("Chapter 3 - {}", 40 + n), 50.0, 30.0, 200.0, 10.0),
                ],
            })
            .collect();

        let stripped = strip_marginalia(&pages);
        for page in &stripped {
            assert_eq!(
                texts(page),
                vec!["Body prose for this page of the report."],
                "digit-varying running foot must be recognised as one signature"
            );
        }
    }

    #[test]
    fn empty_input_is_handled() {
        assert!(strip_marginalia(&[]).is_empty());
    }
}
