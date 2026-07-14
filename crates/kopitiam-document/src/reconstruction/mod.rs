mod citations;
mod figures;
mod headings;
mod lists;
mod paragraphs;
mod tables;

use std::cmp::Ordering;

use kopitiam_pdf::{Page, TextSpan};

use crate::{Block, Document, Heading, Metadata, Paragraph};

/// One visual line of text on a page: spans grouped by shared baseline and
/// sorted left to right.
struct Line {
    text: String,
    y: f32,
    font_size: f32,
    /// Sub-runs of this line separated by a gap wide enough to suggest a
    /// column boundary; used by table detection.
    cells: Vec<Cell>,
}

struct Cell {
    text: String,
    x: f32,
    x_end: f32,
}

const SAME_LINE_Y_TOLERANCE_RATIO: f32 = 0.4;
const WORD_GAP_RATIO: f32 = 0.15;
const COLUMN_GAP_RATIO: f32 = 2.5;
const STRADDLE_LINE_MAX_FRACTION: f32 = 0.15;
const FULL_WIDTH_CELL_MIN_RATIO: f32 = 0.66;


/// Turn a page's raw text spans into the semantic `Document` AST: split each
/// page into reading-order columns, group spans into lines, then classify
/// each line (or run of lines) as a heading, list, table, figure caption, or
/// paragraph. A final pass repairs the one join a per-page pipeline cannot
/// see by construction: a paragraph split across a page break (see
/// `merge_page_breaks` / kopitiam-d3n).
pub fn reconstruct(pages: &[Page]) -> Document {
    let body_font_size = estimate_body_font_size(pages);
    let mut citations = Vec::new();
    let mut pages_blocks: Vec<Vec<Block>> = Vec::with_capacity(pages.len());

    for page in pages {
        let mut page_blocks = Vec::new();
        for column_spans in split_columns(page) {
            let lines = group_lines(&column_spans);
            for block in build_blocks(&lines, body_font_size) {
                if let Block::Paragraph(paragraph) = &block {
                    citations.extend(citations::detect(&paragraph.text));
                }
                page_blocks.push(block);
            }
        }
        pages_blocks.push(page_blocks);
    }

    let (blocks, block_pages) = merge_page_breaks(pages_blocks);

    Document {
        title: infer_title(&blocks),
        metadata: Metadata {
            source_pages: pages.len(),
        },
        blocks,
        block_pages,
        citations,
    }
}

/// Joins each page's independently-reconstructed blocks into one stream,
/// repairing a paragraph that a page break cut in two (kopitiam-d3n).
///
/// Reconstruction runs per page (`split_columns` and `build_blocks` only see
/// one page's spans at a time), so a paragraph that runs from the bottom of
/// page N into the top of page N+1 comes out of the per-page loop as two
/// separate `Paragraph` blocks with no memory of each other. This pass is
/// the only place that sees both halves at once, so it is the only place
/// that can recognise and repair the split.
///
/// Only the immediate boundary between two pages is ever considered: the
/// last block produced for page N against the first block produced for page
/// N+1. That means a Heading/Table/Figure/List sitting at either boundary
/// blocks the merge automatically, without extra logic -- the merge check
/// only fires when *both* boundary blocks are `Block::Paragraph`. Blank
/// pages (no spans, e.g. an intentional page break) are skipped when
/// looking for a boundary, so a paragraph can still merge across a blank
/// page onto the next page with real content.
/// Returns the flattened blocks alongside the 1-based page each one **starts**
/// on — see [`crate::Document::block_pages`] for why that page number is worth
/// carrying rather than discarding, as this function used to.
///
/// A block merged across a page break keeps the *earlier* page, because that is
/// where a reader following the citation should begin looking.
fn merge_page_breaks(pages_blocks: Vec<Vec<Block>>) -> (Vec<Block>, Vec<usize>) {
    let mut blocks: Vec<Block> = Vec::new();
    let mut block_pages: Vec<usize> = Vec::new();

    for (page_index, page_blocks) in pages_blocks.into_iter().enumerate() {
        if page_blocks.is_empty() {
            continue;
        }
        // Pages are 1-based when a human is going to read the number.
        let page = page_index + 1;

        let mut page_blocks = page_blocks.into_iter();
        let leading = page_blocks.next();

        let merged_text = match (blocks.last(), &leading) {
            (Some(Block::Paragraph(trailing)), Some(Block::Paragraph(leading_paragraph))) => {
                paragraphs::merge_across_page_break(&trailing.text, &leading_paragraph.text)
            }
            _ => None,
        };

        match merged_text {
            Some(text) => {
                *blocks
                    .last_mut()
                    .expect("merged_text is only Some when blocks.last() matched") =
                    Block::Paragraph(Paragraph { text });
                // Deliberately do NOT touch this block's recorded page: the
                // merged paragraph began on the previous page, and that is the
                // page a citation must point at.
            }
            None => {
                if let Some(leading_block) = leading {
                    blocks.push(leading_block);
                    block_pages.push(page);
                }
            }
        }

        for block in page_blocks {
            blocks.push(block);
            block_pages.push(page);
        }
    }

    debug_assert_eq!(
        blocks.len(),
        block_pages.len(),
        "block_pages must stay parallel to blocks, or every citation this document produces is wrong"
    );

    (blocks, block_pages)
}

fn build_blocks(lines: &[Line], body_font_size: f32) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if let Some((table, consumed)) = tables::try_table(&lines[i..]) {
            blocks.push(Block::Table(table));
            i += consumed;
            continue;
        }

        if let Some(figure) = figures::try_figure(&lines[i]) {
            blocks.push(Block::Figure(figure));
            i += 1;
            continue;
        }

        if let Some(level) = headings::heading_level(&lines[i], body_font_size) {
            blocks.push(Block::Heading(Heading {
                level,
                text: lines[i].text.trim().to_string(),
            }));
            i += 1;
            continue;
        }

        if let Some((list, consumed)) = lists::try_list(&lines[i..]) {
            blocks.push(Block::List(list));
            i += consumed;
            continue;
        }

        let (paragraph, consumed) = paragraphs::consume_paragraph(&lines[i..]);
        blocks.push(Block::Paragraph(paragraph));
        i += consumed;
    }

    blocks
}

fn infer_title(blocks: &[Block]) -> Option<String> {
    blocks.iter().find_map(|block| match block {
        Block::Heading(Heading { level: 1, text }) => Some(text.clone()),
        _ => None,
    })
}

/// The most common font size across the document, used as the "body text"
/// baseline that heading detection compares against.
///
/// # Determinism, and the bug this used to have
///
/// This counted into a `HashMap` and picked the winner with `max_by_key`. When
/// two font sizes tie on frequency, `max_by_key` returns whichever the iterator
/// happened to yield last — and `HashMap`'s iteration order is **randomised per
/// process**. So `reconstruct()` could produce a *different document from the
/// same PDF on two runs*: a different body size means different headings, which
/// means different structure.
///
/// That is a direct violation of the Semantic Runtime's reproducibility
/// principle ("Indexes are reproducible, not synchronized" — CLAUDE.md), and it
/// was not theoretical: it was hit on a real 3-line endorsement page, where a
/// tie is entirely normal because there is barely any text to break it.
///
/// Ties are now broken **towards the smaller font size**, deterministically.
/// That is not an arbitrary choice: body text is the thing there is most of, and
/// when a document is too short to establish that by frequency, the smaller of
/// two equally-common sizes is far more likely to be the body than the heading.
/// Guessing "heading" would promote ordinary prose into headings and shred the
/// structure.
fn estimate_body_font_size(pages: &[Page]) -> f32 {
    use std::collections::BTreeMap;

    // BTreeMap, not HashMap: iteration is ordered by key, so the tie-break below
    // is reproducible across runs and machines.
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for page in pages {
        for span in &page.spans {
            // Bucket to the nearest half-point to absorb float noise.
            let bucket = (span.font_size * 2.0).round() as u32;
            *counts.entry(bucket).or_default() += 1;
        }
    }

    counts
        .into_iter()
        // Highest count wins; on a tie, the SMALLEST bucket wins. `min_by_key`
        // over (Reverse(count), bucket) picks max count, then min bucket — and
        // because BTreeMap yields buckets in ascending order, the result is the
        // same on every run.
        .min_by_key(|&(bucket, count)| (std::cmp::Reverse(count), bucket))
        .map(|(bucket, _)| bucket as f32 / 2.0)
        .unwrap_or(12.0)
}

/// Splits a page's spans into left-to-right, top-to-bottom reading-order
/// column groups.
///
/// A single-column page with normal margins routinely has lines whose text
/// crosses the page's geometric midpoint (most paragraph lines are wider
/// than half the page) -- so "spans exist on both sides of the midpoint" is
/// true for nearly every document and cannot be the two-column test.
///
/// Real two-column layouts instead have a genuine empty gutter at the
/// midpoint. But naively grouping all of a page's spans into y-based lines
/// first (as `group_lines` does) can still merge left- and right-column text
/// that happens to share a baseline (common: columns are typeset on a shared
/// line grid) into one "line" whose overall bounding box crosses the
/// midpoint -- even though neither column's text actually does. So the test
/// is per *cell*, not per line's overall extent: a cell is one uninterrupted
/// glyph run (see `build_line`'s gap detection), so a cell crossing the
/// midpoint means continuous prose was actually typeset across it, whereas
/// two same-baseline column fragments merged by `group_lines` show up as two
/// separate cells that individually stay on one side.
///
/// A confirmed two-column page can still contain a full-width element (a
/// spanning figure, table, or section heading) that interrupts the flow
/// partway down -- see `split_two_column_page_into_bands` (kopitiam-zay).
fn split_columns(page: &Page) -> Vec<Vec<TextSpan>> {
    if page.spans.is_empty() {
        return vec![Vec::new()];
    }

    let midpoint = page.width / 2.0;
    let full_lines = group_lines(&page.spans);

    let straddling = full_lines
        .iter()
        .filter(|line| is_full_width_line(line, page.width, midpoint))
        .count();
    let straddle_fraction = straddling as f32 / full_lines.len().max(1) as f32;

    if straddle_fraction > STRADDLE_LINE_MAX_FRACTION {
        return vec![page.spans.clone()];
    }

    let mut left = Vec::new();
    let mut right = Vec::new();
    for span in &page.spans {
        let center = span.x + span.width / 2.0;
        if center < midpoint {
            left.push(span.clone());
        } else {
            right.push(span.clone());
        }
    }

    if left.is_empty() || right.is_empty() {
        return vec![page.spans.clone()];
    }

    split_two_column_page_into_bands(page, midpoint)
}

/// A line counts as a full-width interruption of a two-column layout under
/// either of two independent signals:
///
/// - One of its cells (a single uninterrupted glyph run, see `build_line`)
///   literally straddles the column gutter at the page midpoint. This is
///   the same per-cell test `split_columns` uses to decide two-column vs.
///   single-column in the first place, for the same reason: a merged same-
///   baseline `Line` built from two unrelated column fragments must not be
///   judged by its combined bounding box (see the `split_columns` doc
///   comment), only by whether one continuous glyph run actually crosses
///   the midpoint.
/// - One of its cells is, by itself, wider than a plausible single column
///   (`FULL_WIDTH_CELL_MIN_RATIO` of the page width). This catches a
///   spanning element whose own internal layout (e.g. a table with its own
///   column gap) happens not to cross the exact page midpoint pixel, while
///   still being deliberately typeset wider than either page column. Like
///   the straddle test, this is evaluated per cell rather than over the
///   line's overall extent, so it cannot be fooled by two narrow same-
///   baseline column fragments that merely sit far apart.
fn is_full_width_line(line: &Line, page_width: f32, midpoint: f32) -> bool {
    line.cells.iter().any(|cell| {
        (cell.x < midpoint && cell.x_end > midpoint)
            || (cell.x_end - cell.x) > page_width * FULL_WIDTH_CELL_MIN_RATIO
    })
}

/// Reading order within a confirmed two-column page, in the presence of a
/// full-width element that interrupts the two-column flow partway down
/// (kopitiam-zay).
///
/// Without this, `split_columns` would bucket every span on the page into
/// "left" or "right" purely by which side of the midpoint its centre falls
/// on -- which is correct for genuine column text, but scrambles a spanning
/// figure/table/heading: half its spans land in the left group and half in
/// the right, and both halves get read in the wrong place (after all of the
/// real left/right column text, instead of at the full-width element's own
/// vertical position).
///
/// Instead this walks the page top to bottom and buckets each line into one
/// of three running accumulators -- left column, right column, or the
/// current full-width run -- flushing the other two whenever the mode
/// changes. Consecutive full-width lines are kept in one run (rather than
/// flushed line-by-line) so a multi-row full-width table or a multi-line
/// full-width caption still reaches `build_blocks` as consecutive `Line`s,
/// which multi-line detectors like `tables::try_table` require. The result
/// is a sequence of column groups in true reading order: left-then-right
/// within each vertical band, with full-width runs emitted as their own
/// single group exactly where they occur between bands.
fn split_two_column_page_into_bands(page: &Page, midpoint: f32) -> Vec<Vec<TextSpan>> {
    let mut result = Vec::new();
    let mut band_left: Vec<TextSpan> = Vec::new();
    let mut band_right: Vec<TextSpan> = Vec::new();
    let mut band_full_width: Vec<TextSpan> = Vec::new();

    for mut group in group_spans_by_baseline(&page.spans) {
        group.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));
        let refs: Vec<&TextSpan> = group.iter().collect();
        let line = build_line(&refs);

        if is_full_width_line(&line, page.width, midpoint) {
            if !band_left.is_empty() {
                result.push(std::mem::take(&mut band_left));
            }
            if !band_right.is_empty() {
                result.push(std::mem::take(&mut band_right));
            }
            band_full_width.extend(group);
        } else {
            if !band_full_width.is_empty() {
                result.push(std::mem::take(&mut band_full_width));
            }
            for span in group {
                let center = span.x + span.width / 2.0;
                if center < midpoint {
                    band_left.push(span);
                } else {
                    band_right.push(span);
                }
            }
        }
    }

    if !band_full_width.is_empty() {
        result.push(band_full_width);
    }
    if !band_left.is_empty() {
        result.push(band_left);
    }
    if !band_right.is_empty() {
        result.push(band_right);
    }

    result
}

fn group_lines(spans: &[TextSpan]) -> Vec<Line> {
    group_spans_by_baseline(spans)
        .into_iter()
        .map(|mut group| {
            group.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));
            let refs: Vec<&TextSpan> = group.iter().collect();
            build_line(&refs)
        })
        .collect()
}

/// Groups spans that share a baseline (within `SAME_LINE_Y_TOLERANCE_RATIO`
/// of font size) into per-line runs, sorted top to bottom.
///
/// Factored out of `group_lines` so `split_two_column_page_into_bands` can
/// reuse the same baseline-matching logic while keeping the original
/// `TextSpan`s: `group_lines`'s `Line` output only keeps merged, already-
/// concatenated `Cell` text, which is enough to classify a line but not
/// enough to re-partition its spans between page columns.
fn group_spans_by_baseline(spans: &[TextSpan]) -> Vec<Vec<TextSpan>> {
    let mut ordered: Vec<&TextSpan> = spans.iter().collect();
    ordered.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(Ordering::Equal));

    let mut groups: Vec<Vec<TextSpan>> = Vec::new();
    for span in ordered {
        let joins_last = groups.last().is_some_and(|group: &Vec<TextSpan>| {
            let anchor = &group[0];
            let tolerance = anchor.font_size.max(span.font_size) * SAME_LINE_Y_TOLERANCE_RATIO;
            (anchor.y - span.y).abs() <= tolerance
        });

        if joins_last {
            groups.last_mut().unwrap().push(span.clone());
        } else {
            groups.push(vec![span.clone()]);
        }
    }

    groups
}

fn build_line(spans: &[&TextSpan]) -> Line {
    let mut cells: Vec<Cell> = Vec::new();
    let mut text = String::new();
    let mut prev_end: Option<f32> = None;

    for span in spans {
        let gap = prev_end.map(|end| span.x - end);

        // A real inter-word space is a much smaller gap than a column/cell
        // boundary. Below `WORD_GAP_RATIO` the spans are contiguous glyphs
        // (e.g. an OCR text layer that split one word into several spans)
        // and must be concatenated with no space, or every such split would
        // otherwise render as a broken word ("Boo k" instead of "Book").
        let is_word_gap = gap.is_some_and(|gap| gap > span.font_size * WORD_GAP_RATIO);
        let starts_new_cell = match gap {
            Some(gap) => gap > span.font_size * COLUMN_GAP_RATIO,
            None => true,
        };

        if starts_new_cell {
            cells.push(Cell {
                text: span.text.clone(),
                x: span.x,
                x_end: span.x + span.width,
            });
        } else if let Some(cell) = cells.last_mut() {
            if is_word_gap {
                cell.text.push(' ');
            }
            cell.text.push_str(&span.text);
            cell.x_end = span.x + span.width;
        }

        if is_word_gap {
            text.push(' ');
        }
        text.push_str(&span.text);

        prev_end = Some(span.x + span.width);
    }

    let y = spans.first().map(|s| s.y).unwrap_or(0.0);
    let font_size = spans.iter().map(|s| s.font_size).fold(0.0_f32, f32::max);

    Line {
        text,
        y,
        font_size,
        cells,
    }
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

    #[test]
    fn build_line_merges_contiguous_glyph_runs_without_a_space() {
        // "Boo" then "k" with almost no gap simulates an OCR text layer that
        // split one word into two spans; it must read back as "Book".
        let boo = span("Boo", 0.0, 0.0, 18.0, 10.0);
        let k = span("k", 18.2, 0.0, 6.0, 10.0);
        let line = build_line(&[&boo, &k]);
        assert_eq!(line.text, "Book");
    }

    #[test]
    fn build_line_keeps_a_space_for_a_real_word_gap() {
        let book = span("Book", 0.0, 0.0, 24.0, 10.0);
        let reviews = span("Reviews", 27.0, 0.0, 40.0, 10.0);
        let line = build_line(&[&book, &reviews]);
        assert_eq!(line.text, "Book Reviews");
    }

    #[test]
    fn build_line_splits_a_wide_gap_into_a_new_cell() {
        let metric = span("Metric", 0.0, 0.0, 30.0, 10.0);
        let value = span("Value", 60.0, 0.0, 20.0, 10.0);
        let line = build_line(&[&metric, &value]);
        assert_eq!(line.cells.len(), 2);
    }

    #[test]
    fn single_column_page_is_not_split() {
        // Every line's text spans the full page width, crossing the
        // midpoint -- this must not be mistaken for a two-column layout.
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![
                span(
                    "Line one spans the whole page width here",
                    50.0,
                    700.0,
                    500.0,
                    10.0,
                ),
                span(
                    "Line two also spans the whole page width",
                    50.0,
                    686.0,
                    500.0,
                    10.0,
                ),
            ],
        };
        assert_eq!(split_columns(&page).len(), 1);
    }

    #[test]
    fn two_column_page_is_split() {
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![
                span("Left column text here", 40.0, 700.0, 220.0, 10.0),
                span("Left column text here", 40.0, 686.0, 220.0, 10.0),
                span("Right column text here", 340.0, 700.0, 220.0, 10.0),
                span("Right column text here", 340.0, 686.0, 220.0, 10.0),
            ],
        };
        assert_eq!(split_columns(&page).len(), 2);
    }

    /// Builds a two-column page with a full-width row (e.g. a spanning
    /// figure/table/heading) interrupting the flow partway down, per
    /// kopitiam-zay. Three rows of paired left/right text above and below
    /// the interruption keep the full-width line's share of all lines under
    /// `STRADDLE_LINE_MAX_FRACTION`, so the page is still correctly
    /// recognised as two-column rather than falling back to the single-
    /// column path.
    fn two_column_page_with_full_width_interruption() -> Page {
        let mut spans = Vec::new();
        for (i, y) in [760.0, 748.0, 736.0].into_iter().enumerate() {
            spans.push(span(&format!("Top left {i}"), 40.0, y, 220.0, 10.0));
            spans.push(span(&format!("Top right {i}"), 340.0, y, 220.0, 10.0));
        }
        spans.push(span(
            "Full width heading spanning both columns",
            40.0,
            700.0,
            520.0,
            10.0,
        ));
        for (i, y) in [660.0, 648.0, 636.0].into_iter().enumerate() {
            spans.push(span(&format!("Bottom left {i}"), 40.0, y, 220.0, 10.0));
            spans.push(span(&format!("Bottom right {i}"), 340.0, y, 220.0, 10.0));
        }

        Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans,
        }
    }

    #[test]
    fn full_width_element_splits_a_two_column_page_into_bands() {
        let page = two_column_page_with_full_width_interruption();
        let columns = split_columns(&page);

        // Top-left, top-right, the full-width run, bottom-left, bottom-right
        // -- five groups in true top-to-bottom, left-then-right order, not
        // "everything left of the midpoint, then everything right of it"
        // (which would scatter the full-width row's spans across both).
        assert_eq!(columns.len(), 5);
        assert!(columns[0].iter().all(|s| s.text.starts_with("Top left")));
        assert!(columns[1].iter().all(|s| s.text.starts_with("Top right")));
        assert_eq!(columns[2].len(), 1);
        assert_eq!(columns[2][0].text, "Full width heading spanning both columns");
        assert!(columns[3].iter().all(|s| s.text.starts_with("Bottom left")));
        assert!(columns[4].iter().all(|s| s.text.starts_with("Bottom right")));
    }

    #[test]
    fn plain_two_column_page_is_unaffected_by_band_splitting() {
        // Same shape as `two_column_page_is_split`, re-asserted through the
        // banding path to confirm a page with no full-width interruption
        // still produces exactly the original left-then-right column split.
        let page = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![
                span("Left column text here", 40.0, 700.0, 220.0, 10.0),
                span("Left column text here", 40.0, 686.0, 220.0, 10.0),
                span("Right column text here", 340.0, 700.0, 220.0, 10.0),
                span("Right column text here", 340.0, 686.0, 220.0, 10.0),
            ],
        };
        let columns = split_columns(&page);
        assert_eq!(columns.len(), 2);
        assert!(columns[0].iter().all(|s| s.text == "Left column text here"));
        assert!(columns[1].iter().all(|s| s.text == "Right column text here"));
    }

    #[test]
    fn paragraph_split_across_a_page_break_is_merged() {
        // Single-column pages (spans deliberately cross the page midpoint,
        // as in `single_column_page_is_not_split`) so column splitting is
        // not a confound for this test -- only the cross-page merge pass is
        // under test here.
        let page1 = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![span(
                "This paragraph is cut off at the bottom of the page and",
                50.0,
                700.0,
                500.0,
                10.0,
            )],
        };
        let page2 = Page {
            number: 2,
            width: 600.0,
            height: 800.0,
            spans: vec![span(
                "continues here after the page break.",
                50.0,
                700.0,
                500.0,
                10.0,
            )],
        };

        let document = reconstruct(&[page1, page2]);
        assert_eq!(document.blocks.len(), 1);
        match &document.blocks[0] {
            Block::Paragraph(paragraph) => assert_eq!(
                paragraph.text,
                "This paragraph is cut off at the bottom of the page and continues here after the page break."
            ),
            other => panic!("expected a merged Paragraph block, got {other:?}"),
        }

        // A merged paragraph must cite the page it STARTED on. It began on page
        // 1; a citation pointing at page 2 would send a reader to the middle of
        // a sentence and look authoritative doing it.
        assert_eq!(document.page_of(0), Some(1));
    }

    #[test]
    fn every_block_records_the_page_it_starts_on() {
        // The property every provenance-carrying consumer depends on: a
        // citation without a page is not one a reader can follow.
        let page1 = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![span("Sentence one finishes here.", 50.0, 700.0, 500.0, 10.0)],
        };
        let page2 = Page {
            number: 2,
            width: 600.0,
            height: 800.0,
            spans: vec![span("Sentence two begins the second page.", 50.0, 700.0, 500.0, 10.0)],
        };

        let document = reconstruct(&[page1, page2]);

        // Parallel, always. If these ever diverge, every citation the Document
        // Engine produces is silently wrong.
        assert_eq!(document.blocks.len(), document.block_pages.len());
        assert_eq!(document.page_of(0), Some(1));
        assert_eq!(document.page_of(1), Some(2));
        // Out of range is None, never a guessed page 1.
        assert_eq!(document.page_of(99), None);

        let paired: Vec<Option<usize>> = document.blocks_with_pages().map(|(_, page)| page).collect();
        assert_eq!(paired, vec![Some(1), Some(2)]);
    }

    #[test]
    fn body_font_size_is_deterministic_when_two_sizes_tie() {
        // reconstruct() counted font sizes into a HashMap and broke ties with
        // `max_by_key`, so a tie was resolved by RANDOMISED hash iteration order
        // -- meaning the same PDF could reconstruct differently on two runs. A
        // different body size means different headings, which means a different
        // document. This was hit on a real 3-line endorsement page, where a tie
        // is entirely normal because there is barely any text to break it.
        let tied = || Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![
                span("Alpha at ten point", 50.0, 700.0, 400.0, 10.0),
                span("Bravo at fourteen", 50.0, 660.0, 400.0, 14.0),
            ],
        };

        // Same input, many runs: the answer must never move.
        let first = estimate_body_font_size(&[tied()]);
        for _ in 0..64 {
            assert_eq!(estimate_body_font_size(&[tied()]), first, "font-size estimate is not deterministic");
        }

        // And the tie breaks towards the SMALLER size: when a document is too
        // short to establish the body size by frequency, the smaller of two
        // equally-common sizes is far likelier to be body text than a heading.
        // Guessing "heading" would promote ordinary prose and shred the structure.
        assert_eq!(first, 10.0);
    }

    #[test]
    fn a_document_with_no_page_information_reports_none_rather_than_guessing() {
        // `Default` (and any hand-built Document) has no page information. It
        // must say so, not silently attribute everything to page 1 -- a wrong
        // page in a citation is worse than an absent one.
        let document = Document {
            blocks: vec![Block::Paragraph(Paragraph { text: "orphan".to_string() })],
            ..Document::default()
        };
        assert_eq!(document.page_of(0), None);
        assert_eq!(document.blocks_with_pages().next().unwrap().1, None);
    }

    #[test]
    fn paragraph_ending_a_sentence_does_not_merge_across_a_page_break() {
        let page1 = Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans: vec![span(
                "This sentence finishes cleanly on the first page.",
                50.0,
                700.0,
                500.0,
                10.0,
            )],
        };
        let page2 = Page {
            number: 2,
            width: 600.0,
            height: 800.0,
            spans: vec![span(
                "New paragraph starts capitalized on the next page.",
                50.0,
                700.0,
                500.0,
                10.0,
            )],
        };

        let document = reconstruct(&[page1, page2]);
        assert_eq!(document.blocks.len(), 2);
        for block in &document.blocks {
            assert!(matches!(block, Block::Paragraph(_)));
        }
    }
}
