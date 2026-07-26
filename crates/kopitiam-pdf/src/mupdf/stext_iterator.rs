//! Ported from MuPDF `source/fitz/stext-iterator.c` -- the depth-first block
//! iterator used to walk an `fz_stext_page`'s block tree in reading order
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the traversal order follows
//! MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this is
//!
//! MuPDF's `fz_stext_page_block_iterator` walks the block list, and the `_dfs`
//! variants (`fz_stext_page_block_iterator_begin_dfs` /
//! `_next_dfs` / `_eod_dfs`) silently descend into `FZ_STEXT_BLOCK_STRUCT`
//! nodes and pop back up at the end of a run, yielding every leaf block in
//! document/reading order. The layout-analysis code
//! ([`super::stext_boxer`], [`super::stext_classify`]) uses this walk to visit
//! every block regardless of how deeply the segmentation nested it.
//!
//! # This port's data model
//!
//! Wave 6's [`StextPage`] is a **flat** `Vec<StextBlock>` -- the port does not
//! (yet) model `FZ_STEXT_BLOCK_STRUCT` nesting (deferred with the structure
//! wave; see [`super::structured_text`]). With no `Struct` blocks to descend
//! into, the DFS walk degenerates to a straight left-to-right scan of the block
//! vector, which is exactly the leaf order MuPDF's `_next_dfs` would produce
//! over the same (struct-free) list. The iterator is kept as a named type so a
//! later structure wave can add the descend/ascend arms without changing
//! callers.

use super::geometry::Rect;
use super::structured_text::{StextBlock, StextLine, StextPage};

// MuPDF: the bbox stored on every fz_stext_block (block->bbox), recomputed here
// from the block's content so hand-built fixtures need not pre-populate it.
/// The bounding box of a block: for a text block, the union of its line boxes;
/// for an image block, its stored transform-derived box.
pub fn block_bbox(block: &StextBlock) -> Rect {
    match block {
        StextBlock::Text(tb) => {
            let mut r = Rect::EMPTY;
            for line in &tb.lines {
                r = r.union(line_bbox(line));
            }
            r
        }
        StextBlock::Image(ib) => ib.bbox,
    }
}

// MuPDF: fz_stext_line.bbox (the union of the char quads). Prefers the value the
// device already rolled up; falls back to recomputing from the char quads when
// a fixture leaves it unset.
/// The bounding box of a line: its stored bbox if valid, else the union of its
/// char quads.
pub fn line_bbox(line: &StextLine) -> Rect {
    if line.bbox.is_valid() && !line.bbox.is_empty() {
        return line.bbox;
    }
    let mut r = Rect::EMPTY;
    for ch in &line.chars {
        r = r.union(Rect::from_quad(ch.quad));
    }
    r
}

/// A depth-first iterator over a page's blocks, in reading order.
///
/// Ports the traversal of `fz_stext_page_block_iterator_begin_dfs` /
/// `_next_dfs`. With no `Struct` blocks in this port's model (see the module
/// docs) it is a linear scan; the type exists so struct descent can be added
/// later without touching call sites.
pub struct BlockDfsIter<'a> {
    // MuPDF: fz_stext_page_block_iterator.block (advanced by _next_dfs).
    inner: std::slice::Iter<'a, StextBlock>,
}

impl<'a> Iterator for BlockDfsIter<'a> {
    type Item = &'a StextBlock;

    // MuPDF: fz_stext_page_block_iterator_next_dfs -- yields the next leaf block.
    fn next(&mut self) -> Option<&'a StextBlock> {
        self.inner.next()
    }
}

// MuPDF: fz_stext_page_block_iterator_begin_dfs (stext-iterator.c:73).
/// Begin a depth-first walk over `page`'s blocks in reading order.
pub fn blocks_dfs(page: &StextPage) -> BlockDfsIter<'_> {
    BlockDfsIter { inner: page.blocks.iter() }
}
