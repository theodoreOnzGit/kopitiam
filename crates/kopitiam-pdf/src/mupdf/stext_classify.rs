//! Ported from MuPDF `source/fitz/stext-classify.c` -- the step that takes the
//! regions discovered by the segmenter and assigns the page's blocks to them in
//! reading order (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the
//! reading-order semantics follow MuPDF; the code is re-expressed in idiomatic
//! Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What MuPDF does, and what this port does
//!
//! In MuPDF, `fz_classify_stext_rect` walks the block tree
//! ([`super::stext_iterator`]), and for each analysis region moves the blocks
//! (splitting text blocks line-by-line where they straddle a boundary, via
//! `split_text_by_rect`) into a new `FZ_STEXT_BLOCK_STRUCT` node. Because the
//! segmenter visits regions in DFS reading order (left column before right,
//! top band before bottom), the resulting struct tree, walked depth-first,
//! yields the blocks in reading order.
//!
//! This port has no struct-block tree (deferred; see
//! [`super::structured_text`]), so [`order_blocks_by_regions`] achieves the same
//! *observable* result on the flat block list: it buckets each block into the
//! reading-order region it best belongs to, orders the buckets in region order,
//! and within a bucket sorts top-to-bottom (then left-to-right). The net effect
//! -- a two-column page linearising column-by-column -- matches MuPDF's DFS walk
//! of the struct tree it would have built.
//!
//! # Deferred (noted, not done)
//!
//! * **`split_text_by_rect`**: line/char-level splitting of a block that
//!   straddles a region boundary. Here a straddling block is assigned whole to
//!   the region it overlaps most. Faithful splitting is deferred with the
//!   struct wave.
//! * **`fz_structure` classification tags** (H / DIV / list-item, …): the port
//!   has no structure field to carry them.

use super::geometry::Rect;
use super::stext_iterator::block_bbox;
use super::structured_text::{StextBlock, StextPage};

// MuPDF: the reading-order result of fz_classify_stext_rect over every analysis
// region (stext-classify.c:356), reduced to a reordering of the flat block list.
/// Re-order `page.blocks` into the reading order implied by `regions` (the
/// segmenter's leaf regions, already in DFS reading order).
///
/// Each block is bucketed into the region it overlaps most; buckets are emitted
/// in region order, and within a bucket blocks are sorted top-to-bottom then
/// left-to-right. Blocks overlapping no region keep their original relative
/// order at the end. A no-op when there are no regions or fewer than two blocks.
pub(crate) fn order_blocks_by_regions(page: &mut StextPage, regions: &[Rect]) {
    if regions.is_empty() || page.blocks.len() < 2 {
        return;
    }

    let blocks = std::mem::take(&mut page.blocks);
    let mut buckets: Vec<Vec<StextBlock>> = (0..regions.len()).map(|_| Vec::new()).collect();
    let mut leftover: Vec<StextBlock> = Vec::new();

    for b in blocks {
        let bb = block_bbox(&b);
        let mut best: Option<usize> = None;
        let mut best_area = 0.0f32;
        for (i, r) in regions.iter().enumerate() {
            let inter = bb.intersect(*r);
            if inter.is_empty() {
                continue;
            }
            // fz_rect_area, but the intersection is already finite/valid here.
            let area = (inter.x1 - inter.x0) * (inter.y1 - inter.y0);
            if area > best_area {
                best_area = area;
                best = Some(i);
            }
        }
        match best {
            Some(i) => buckets[i].push(b),
            None => leftover.push(b),
        }
    }

    let mut out: Vec<StextBlock> = Vec::with_capacity(page_len(&buckets, &leftover));
    for bucket in &mut buckets {
        // Within a region, read top-to-bottom (device y increases downward),
        // then left-to-right. total_cmp keeps a strict total order under NaN.
        bucket.sort_by(|a, b| {
            let ra = block_bbox(a);
            let rb = block_bbox(b);
            ra.y0.total_cmp(&rb.y0).then(ra.x0.total_cmp(&rb.x0))
        });
        out.append(bucket);
    }
    out.append(&mut leftover);

    page.blocks = out;
}

fn page_len(buckets: &[Vec<StextBlock>], leftover: &[StextBlock]) -> usize {
    buckets.iter().map(Vec::len).sum::<usize>() + leftover.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::{Point, Quad};
    use crate::mupdf::structured_text::{StextChar, StextLine, StextTextBlock};

    fn block_at(tag: &str, x0: f32, y0: f32) -> StextBlock {
        let quad = Quad {
            ul: Point::new(x0, y0),
            ur: Point::new(x0 + 20.0, y0),
            ll: Point::new(x0, y0 + 10.0),
            lr: Point::new(x0 + 20.0, y0 + 10.0),
        };
        let ch = StextChar {
            c: tag.chars().next().unwrap(),
            origin: Point::new(x0, y0 + 10.0),
            quad,
            size: 10.0,
            font: 0,
            flags: 0,
            cid: 0,
            wmode: 0,
        };
        let line = StextLine {
            wmode: 0,
            flags: 0,
            dir: Point::new(1.0, 0.0),
            bbox: Rect::new(x0, y0, x0 + 20.0, y0 + 10.0),
            chars: vec![ch],
        };
        StextBlock::Text(StextTextBlock {
            bbox: line.bbox,
            lines: vec![line],
        })
    }

    fn tags(page: &StextPage) -> String {
        page.blocks
            .iter()
            .filter_map(|b| match b {
                StextBlock::Text(tb) => tb.lines.first().and_then(|l| l.chars.first()).map(|c| c.c),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn buckets_in_region_order_then_top_to_bottom() {
        // Blocks in scrambled draw order; two regions (left then right).
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 200.0, 200.0),
            blocks: vec![
                block_at("2", 10.0, 60.0),  // left, lower
                block_at("b", 110.0, 10.0), // right, upper
                block_at("1", 10.0, 10.0),  // left, upper
                block_at("a", 110.0, 60.0), // right, lower
            ],
            fonts: Vec::new(),
        };
        let regions = vec![
            Rect::new(0.0, 0.0, 100.0, 200.0),
            Rect::new(100.0, 0.0, 200.0, 200.0),
        ];
        order_blocks_by_regions(&mut page, &regions);
        // Left region top-to-bottom: '1','2'; right region: 'b','a' -> but 'a'
        // is lower so should come after 'b'. Result: 1 2 b a.
        assert_eq!(tags(&page), "12ba");
    }

    #[test]
    fn no_regions_is_noop() {
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 200.0, 200.0),
            blocks: vec![block_at("x", 10.0, 10.0), block_at("y", 10.0, 60.0)],
            fonts: Vec::new(),
        };
        order_blocks_by_regions(&mut page, &[]);
        assert_eq!(tags(&page), "xy");
    }
}
