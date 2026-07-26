//! Ported from MuPDF `source/fitz/stext-para.c` -- paragraph splitting within a
//! block/column (`fz_paragraph_break`) (commit 19f1284, AGPL-3.0, © Artifex
//! Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close
//! adaptation: the line-grouping semantics follow MuPDF; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # What this does
//!
//! After segmentation ([`super::stext_boxer`]) each region is a run of lines.
//! `fz_paragraph_break` walks those lines and splits a block wherever a
//! paragraph boundary is detected. MuPDF applies a battery of heuristics
//! (underlined/bold titles, indents, line-height changes, trailing-gap
//! analysis, full-justification detection, bulleted lists), each implemented via
//! a shared `line_walker`. Most of those depend on features this port does not
//! model (underline/bold style flags, `FZ_STEXT_TEXT_JUSTIFY_*` block flags,
//! the struct DIV tree the splits are hung on).
//!
//! This port implements the load-bearing, model-independent case: **splitting a
//! block where the vertical gap between consecutive lines is large relative to
//! the running line pitch** -- MuPDF's notion (from the stext device's
//! `PARAGRAPH_DIST` and para.c's line-gap walker) that a blank-line-sized gap
//! ends a paragraph. Lines are grouped top-to-bottom (device space, y
//! increasing downward); when the whitespace between one line's bottom and the
//! next line's top exceeds [`PARAGRAPH_LEADING_FACTOR`] times the current line
//! height, a new block begins.
//!
//! # Deferred (noted, not done)
//!
//! * Title detection by underline/bold (`detect_underlined_titles`,
//!   `detect_titles_by_font_usage`) -- needs style flags.
//! * Indent / trailing-gap / justification / list-item breaking
//!   (`break_paragraphs_by_indent`, `_by_analysing_trailing_gaps`,
//!   `_within_justified_text`, `break_list_items`) -- need justify flags and the
//!   struct DIV tree. The line-gap break is the one that fires on plain prose.

use super::geometry::Rect;
use super::stext_iterator::line_bbox;
use super::structured_text::{StextBlock, StextLine, StextPage, StextTextBlock};

// MuPDF: 0.25f line-height slack drives `break_paragraphs_by_line_gap`
// (stext-para.c:640); the stext device treats a >1.5em baseline jump as a new
// paragraph (`PARAGRAPH_DIST`). Here a *gap* larger than half the line height --
// i.e. roughly a blank line's worth of leading -- ends a paragraph. Chosen to
// fire on paragraph spacing while ignoring ordinary single-spaced leading.
/// Inter-line whitespace, as a multiple of the line height, above which a
/// paragraph break is inserted.
pub const PARAGRAPH_LEADING_FACTOR: f32 = 0.5;

// MuPDF: fz_paragraph_break (stext-para.c:1631) -> do_para_break over the page's
// blocks. This port runs only the line-gap breaker.
/// Split every text block of `page` into paragraphs at large inter-line gaps,
/// in place. Non-text blocks pass through unchanged.
pub fn paragraph_break(page: &mut StextPage) {
    let blocks = std::mem::take(&mut page.blocks);
    let mut out: Vec<StextBlock> = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            StextBlock::Text(tb) => split_block_by_line_gap(tb, &mut out),
            other => out.push(other),
        }
    }
    page.blocks = out;
}

// MuPDF: break_paragraphs_by_line_gap (stext-para.c:663) via split_block_at_line
// (stext-para.c:171): accumulate lines into a paragraph, flushing a new block
// whenever the gap to the next line is large.
fn split_block_by_line_gap(tb: StextTextBlock, out: &mut Vec<StextBlock>) {
    if tb.lines.len() < 2 {
        out.push(StextBlock::Text(tb));
        return;
    }

    let mut group: Vec<StextLine> = Vec::new();
    let mut prev_bottom: Option<f32> = None;
    let mut prev_height = 0.0f32;

    for line in tb.lines {
        let lb = line_bbox(&line);
        let height = lb.y1 - lb.y0;

        if let Some(pb) = prev_bottom {
            let gap = lb.y0 - pb; // device space: next line sits below (larger y).
            let pitch = if prev_height > 0.0 { prev_height } else { height };
            if gap > PARAGRAPH_LEADING_FACTOR * pitch && !group.is_empty() {
                out.push(flush(std::mem::take(&mut group)));
            }
        }

        prev_bottom = Some(lb.y1);
        prev_height = height;
        group.push(line);
    }

    if !group.is_empty() {
        out.push(flush(group));
    }
}

// MuPDF: recalc_bbox (stext-para.c:31) -- a block's bbox is the union of its
// line bboxes.
fn flush(lines: Vec<StextLine>) -> StextBlock {
    let mut bbox = Rect::EMPTY;
    for line in &lines {
        bbox = bbox.union(line_bbox(line));
    }
    StextBlock::Text(StextTextBlock { bbox, lines })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::{Point, Quad};
    use crate::mupdf::structured_text::{StextChar, StextLine, StextTextBlock};

    fn line(s: &str, y_top: f32, size: f32) -> StextLine {
        let mut x = 40.0f32;
        let mut chars = Vec::new();
        for c in s.chars() {
            let w = size * 0.5;
            let quad = Quad {
                ul: Point::new(x, y_top),
                ur: Point::new(x + w, y_top),
                ll: Point::new(x, y_top + size),
                lr: Point::new(x + w, y_top + size),
            };
            chars.push(StextChar {
                c,
                origin: Point::new(x, y_top + size),
                quad,
                size,
                font: 0,
                flags: 0,
                cid: 0,
                wmode: 0,
            });
            x += w;
        }
        StextLine {
            wmode: 0,
            flags: 0,
            dir: Point::new(1.0, 0.0),
            bbox: Rect::new(40.0, y_top, x, y_top + size),
            chars,
        }
    }

    #[test]
    fn large_gap_splits_into_two_paragraphs() {
        // Two tightly-spaced lines, a big gap, then two more lines.
        // size 10: normal leading ~2 (< 5 threshold); paragraph gap 15 (> 5).
        let tb = StextTextBlock {
            bbox: Rect::EMPTY,
            lines: vec![
                line("first", 40.0, 10.0),
                line("line", 52.0, 10.0),  // gap 2 -> same paragraph
                line("second", 77.0, 10.0), // gap 15 -> new paragraph
                line("para", 89.0, 10.0),  // gap 2 -> same paragraph
            ],
        };
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 300.0, 300.0),
            blocks: vec![StextBlock::Text(tb)],
            fonts: Vec::new(),
        };
        paragraph_break(&mut page);
        assert_eq!(page.blocks.len(), 2, "the large gap splits the block in two");
        if let (StextBlock::Text(a), StextBlock::Text(b)) = (&page.blocks[0], &page.blocks[1]) {
            assert_eq!(a.lines.len(), 2);
            assert_eq!(b.lines.len(), 2);
            assert_eq!(a.lines[0].text(), "first");
            assert_eq!(b.lines[0].text(), "second");
        } else {
            panic!("expected two text blocks");
        }
    }

    #[test]
    fn tight_lines_stay_one_paragraph() {
        let tb = StextTextBlock {
            bbox: Rect::EMPTY,
            lines: vec![line("a", 40.0, 10.0), line("b", 52.0, 10.0), line("c", 64.0, 10.0)],
        };
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 300.0, 300.0),
            blocks: vec![StextBlock::Text(tb)],
            fonts: Vec::new(),
        };
        paragraph_break(&mut page);
        assert_eq!(page.blocks.len(), 1, "evenly-spaced lines stay one paragraph");
    }
}
