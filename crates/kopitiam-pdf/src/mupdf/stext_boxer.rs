//! Ported from MuPDF `source/fitz/stext-boxer.c` -- the whitespace-cover
//! column/region detector that drives `fz_segment_stext_page` (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithm and numeric behaviour
//! follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # The two-column reading-order fix
//!
//! A naive extractor emits blocks in draw order. On a two-column page the draw
//! order zig-zags across the gutter (`L1 R1 L2 R2 …`), so the linearised text
//! is garbage. [`segment_stext_page`] fixes this: it discovers the page's
//! **column/region structure** and re-orders the flat block list into true
//! reading order -- every left-column block (top→bottom) *then* every
//! right-column block -- so the page reads column-by-column.
//!
//! ## The whitespace-cover / X-Y-cut rule (what distinguishes columns)
//!
//! The detector maintains a **`rectlist` of candidate whitespace rectangles**
//! over the page. It starts with the whole mediabox as one big whitespace rect,
//! then, for every occupied glyph/word rectangle fed in ([`Boxer::feed`]),
//! *subtracts* that rectangle from the whitespace by replacing each whitespace
//! rect with its intersections against the four bands that lie outside the fed
//! box (left / right / above / below). After all words are fed, the rectlist
//! holds the maximal empty rectangles of the page.
//!
//! To split the page ([`boxer_subdivide`]) we look for a whitespace rectangle
//! that spans the region **from edge to edge** -- a full-height vertical
//! corridor (a *gutter*) or a full-width horizontal corridor -- and cut along
//! the **largest** such corridor, recursing into the two halves (an X-Y cut).
//! Crucially, only a corridor that runs *uninterrupted* the full height counts
//! as a column boundary. Wide inter-word spacing never produces one, because
//! the gaps on different lines do not line up vertically; and each word is
//! grown by a `size/4` margin before subtraction, so sub-`size/2` gaps are
//! swallowed entirely. That single rule -- *a column split requires an
//! uninterrupted full-height whitespace gutter, the widest one winning* -- is
//! what tells a genuine two-column layout apart from one column with loose
//! spacing.
//!
//! # Deferred (noted, not done)
//!
//! * **Struct DIV tree.** MuPDF's `page_subset` moves the blocks of each
//!   discovered region into a nested `FZ_STEXT_BLOCK_STRUCT` node; this port has
//!   no struct blocks (see [`super::structured_text`]), so instead of building a
//!   tree we collect the leaf region rectangles in DFS (reading) order and
//!   re-bucket the flat block list into them ([`super::stext_classify`]).
//! * **Page-fill removal** (`fz_stext_remove_page_fill`) and the
//!   vector-run collation (`fz_collate_small_vector_run`): the port produces no
//!   vector blocks, so there is nothing to strip or collate.
//! * **Table grids** (`stext-table.c`), nested/RTL regions: deferred.

use super::geometry::Rect;
use super::stext_classify::order_blocks_by_regions;
use super::stext_iterator::block_bbox;
use super::structured_text::{StextBlock, StextLine, StextPage};

// MuPDF: MAX_ANALYSIS_DEPTH (stext-boxer.c:734).
/// Cap on the X-Y-cut recursion depth.
const MAX_ANALYSIS_DEPTH: i32 = 6;

// ---------------------------------------------------------------------------
// rectlist_t / boxer_t
// ---------------------------------------------------------------------------

// MuPDF: struct boxer_s (stext-boxer.c:37) together with its rectlist_t
// (stext-boxer.c:30). The two are merged here: the rectlist's `fudge` lives on
// the boxer, and `list` is a plain Vec.
/// A whitespace-cover accumulator over a `mediabox`: the `list` is the set of
/// candidate empty rectangles remaining after the fed (occupied) boxes have
/// been subtracted.
struct Boxer {
    // MuPDF: boxer_s.mediabox
    mediabox: Rect,
    // MuPDF: boxer_s.list (rectlist_t.list)
    list: Vec<Rect>,
    // MuPDF: rectlist_t.fudge -- inclusion slack (0 tight, 4 points loose).
    fudge: f32,
    // MuPDF: boxer_s.tight
    tight: bool,
}

impl Boxer {
    // MuPDF: boxer_create_length (stext-boxer.c:102) -- an empty boxer.
    fn create_length(mediabox: Rect, tight: bool) -> Boxer {
        Boxer {
            mediabox,
            list: Vec::new(),
            fudge: if tight { 0.0 } else { 4.0 },
            tight,
        }
    }

    // MuPDF: boxer_create (stext-boxer.c:126) -- seeds the list with the whole
    // mediabox as the initial (single) whitespace rectangle.
    fn create(mediabox: Rect, tight: bool) -> Boxer {
        let mut boxer = Boxer::create_length(mediabox, tight);
        boxer.append(mediabox);
        boxer
    }

    // MuPDF: rectlist_append (stext-boxer.c:58). Push `box_` unless an existing
    // rect already encloses it (within `fudge`); drop any existing rects that
    // `box_` encloses (replacing them with it).
    fn append(&mut self, box_: Rect) {
        let f = self.fudge;
        let mut i = 0;
        while i < self.list.len() {
            let r = self.list[i];
            let smaller = Rect::new(r.x0 + f, r.y0 + f, r.x1 - f, r.y1 - f);
            let larger = Rect::new(r.x0 - f, r.y0 - f, r.x1 + f, r.y1 + f);

            if larger.contains(box_) {
                return; // box is enclosed! Nothing to do.
            }
            if box_.contains(smaller) {
                // box encloses r. Ditch r (swap the last entry into its slot and
                // reconsider that slot next).
                let last = self.list.len() - 1;
                self.list[i] = self.list[last];
                self.list.pop();
                continue;
            }
            i += 1;
        }
        self.list.push(box_);
    }

    // MuPDF: boxer_feed (stext-boxer.c:162). Mark `bbox` as occupied: rebuild
    // the whitespace list as the union of every existing rect intersected with
    // each of the four mediabox bands lying outside `bbox`.
    fn feed(&mut self, bbox: Rect) {
        let mb = self.mediabox;
        let mut newlist = Boxer {
            mediabox: mb,
            list: Vec::new(),
            fudge: self.fudge,
            tight: self.tight,
        };

        // Left (mb.x0, mb.y0) -> (bbox.x0, mb.y1)
        self.feed_intersect(&mut newlist, Rect::new(mb.x0, mb.y0, bbox.x0, mb.y1));
        // Right (bbox.x1, mb.y0) -> (mb.x1, mb.y1)
        self.feed_intersect(&mut newlist, Rect::new(bbox.x1, mb.y0, mb.x1, mb.y1));
        // Bottom (mb.x0, mb.y0) -> (mb.x1, bbox.y0)
        self.feed_intersect(&mut newlist, Rect::new(mb.x0, mb.y0, mb.x1, bbox.y0));
        // Top (mb.x0, bbox.y1) -> (mb.x1, mb.y1)
        self.feed_intersect(&mut newlist, Rect::new(mb.x0, bbox.y1, mb.x1, mb.y1));

        self.list = newlist.list;
    }

    // MuPDF: boxlist_feed_intersect (stext-boxer.c:152) + push_if_intersect_suitable
    // (stext-boxer.c:138): intersect every current rect with `box_` and append
    // the (valid) results to `dst`.
    fn feed_intersect(&self, dst: &mut Boxer, box_: Rect) {
        for &r in &self.list {
            let c = r.intersect(box_);
            // Keep valid (possibly zero-area) intersections; drop disjoint ones.
            if !c.is_valid() {
                continue;
            }
            dst.append(c);
        }
    }

    // MuPDF: boxer_margins (stext-boxer.c:270). Shrink the mediabox inward by any
    // full-span edge whitespace, returning the content bbox.
    fn margins(&self) -> Rect {
        let mut margins = self.mediabox;
        for &r in &self.list {
            if r.x0 <= margins.x0 && r.y0 <= margins.y0 && r.y1 >= margins.y1 {
                margins.x0 = r.x1; // Left margin
            } else if r.x1 >= margins.x1 && r.y0 <= margins.y0 && r.y1 >= margins.y1 {
                margins.x1 = r.x0; // Right margin
            } else if r.x0 <= margins.x0 && r.x1 >= margins.x1 && r.y0 <= margins.y0 {
                margins.y0 = r.y1; // Top margin
            } else if r.x0 <= margins.x0 && r.x1 >= margins.x1 && r.y1 >= margins.y1 {
                margins.y1 = r.y0; // Bottom margin
            }
        }
        margins
    }

    // MuPDF: boxer_subset (stext-boxer.c:293). A new boxer over `rect`, holding
    // every current rect clipped to `rect` (dropping empties).
    fn subset(&self, rect: Rect) -> Boxer {
        let mut new_boxer = Boxer::create_length(rect, self.tight);
        for &r in &self.list {
            let ri = r.intersect(rect);
            if ri.is_empty() {
                continue;
            }
            new_boxer.append(ri);
        }
        new_boxer
    }
}

// ---------------------------------------------------------------------------
// Feeding the page content into a boxer
// ---------------------------------------------------------------------------

// MuPDF: line_isnt_all_spaces (stext-boxer.c:781).
fn line_isnt_all_spaces(line: &StextLine) -> bool {
    line.chars
        .iter()
        .any(|ch| ch.c != ' ' && ch.c != '\u{00A0}')
}

// MuPDF: feed_line (stext-boxer.c:791). Feed each whitespace-delimited word run
// of the line (each grown by a `size/4` margin in loose mode) as one occupied
// box, so inter-word gaps narrower than ~size/2 get covered.
fn feed_line(boxer: &mut Boxer, line: &StextLine) {
    let chars = &line.chars;
    let mut i = 0;
    while i < chars.len() {
        if chars[i].c == ' ' {
            i += 1;
            continue;
        }
        let mut r = Rect::EMPTY;
        while i < chars.len() && chars[i].c != ' ' {
            let ch = &chars[i];
            let margin = if boxer.tight { 0.0 } else { ch.size / 4.0 };
            let bbox = Rect::from_quad(ch.quad).expand(margin);
            r = r.union(bbox);
            i += 1;
        }
        boxer.feed(r);
    }
}

// MuPDF: recurse_and_feed (stext-boxer.c:866). Feed every non-blank text line of
// every block. (The vector/struct arms are absent -- this port has no such
// blocks; image blocks are fed by their bbox.)
fn recurse_and_feed(boxer: &mut Boxer, blocks: &[StextBlock]) {
    for block in blocks {
        match block {
            StextBlock::Text(tb) => {
                for line in &tb.lines {
                    if line_isnt_all_spaces(line) {
                        feed_line(boxer, line);
                    }
                }
            }
            StextBlock::Image(ib) => boxer.feed(ib.bbox),
        }
    }
}

// ---------------------------------------------------------------------------
// The X-Y-cut recursion (analyse_sub / boxer_subdivide / analyse_subset)
// ---------------------------------------------------------------------------

// A region has content if any block box overlaps it. This stands in for MuPDF's
// `page_subset` returning NULL (nothing to collect) -- we do not move blocks
// during the recursion; we only need to know whether the region is worth
// emitting/subdividing.
fn region_has_content(region: Rect, boxes: &[Rect]) -> bool {
    boxes.iter().any(|b| !b.intersect(region).is_empty())
}

// MuPDF: analyse_sub (stext-boxer.c:737). Shrink to the content margins, then
// try to subdivide; if we cannot, emit this region as a reading-order leaf.
// Appends leaf regions to `out` in DFS (reading) order.
fn analyse_sub(boxer: &Boxer, boxes: &[Rect], depth: i32, out: &mut Vec<Rect>) -> bool {
    let margins = boxer.margins();
    let sub = boxer.subset(margins);

    // If nothing textual falls in this region, give up (MuPDF: div == NULL).
    if !region_has_content(margins, boxes) {
        return false;
    }

    if depth < MAX_ANALYSIS_DEPTH && boxer_subdivide(&sub, boxes, depth + 1, out) {
        return true;
    }

    // Leaf: this region is emitted as one reading-order unit.
    out.push(sub.mediabox);
    true
}

// MuPDF: boxer_subdivide (stext-boxer.c:331). Find the largest full-span
// whitespace corridor (full-width horizontal, or full-height vertical gutter)
// and cut along it, recursing into the two halves. Returns false (no cut) when
// no full-span corridor exists.
fn boxer_subdivide(boxer: &Boxer, boxes: &[Rect], depth: i32, out: &mut Vec<Rect>) -> bool {
    let mb = boxer.mediabox;
    let mut max_size = 0.0f64;
    let mut largest: Option<usize> = None;
    let mut horiz = false;

    for (i, r) in boxer.list.iter().enumerate() {
        if r.x0 <= mb.x0 && r.x1 >= mb.x1 {
            // Full-width => horizontal divider; its "size" is its height.
            let size = (r.y1 - r.y0) as f64;
            if size > max_size {
                max_size = size;
                largest = Some(i);
                horiz = true;
            }
        }
        if r.y0 <= mb.y0 && r.y1 >= mb.y1 {
            // Full-height => vertical divider (gutter); its "size" is its width.
            let size = (r.x1 - r.x0) as f64;
            if size > max_size {
                max_size = size;
                largest = Some(i);
                horiz = false;
            }
        }
    }

    let Some(li) = largest else {
        return false;
    };
    let div = boxer.list[li];

    let mut r = mb;
    if horiz {
        // Divider runs horizontally: top region first, then bottom.
        r.y1 = div.y0;
        analyse_sub(&boxer.subset(r), boxes, depth, out);

        r.y0 = div.y1;
        r.y1 = mb.y1;
        analyse_sub(&boxer.subset(r), boxes, depth, out);
    } else {
        // Divider runs vertically: left region first, then right.
        r.x1 = div.x0;
        analyse_sub(&boxer.subset(r), boxes, depth, out);

        r.x0 = div.x1;
        r.x1 = mb.x1;
        analyse_sub(&boxer.subset(r), boxes, depth, out);
    }

    true
}

// MuPDF: segment_rect (stext-boxer.c:904). Build a boxer over `box_`, feed the
// page content, and run the analysis, returning the discovered leaf regions in
// reading order.
fn segment_regions(page: &StextPage) -> Vec<Rect> {
    if page.blocks.is_empty() || !page.mediabox.is_valid() || page.mediabox.is_empty() {
        return Vec::new();
    }

    // tight = false for a whole page (loose inclusion + per-word size/4 margins).
    let mut boxer = Boxer::create(page.mediabox, false);
    recurse_and_feed(&mut boxer, &page.blocks);

    let boxes: Vec<Rect> = page
        .blocks
        .iter()
        .map(block_bbox)
        .filter(|r| r.is_valid() && !r.is_empty())
        .collect();

    let mut out = Vec::new();
    analyse_sub(&boxer, &boxes, 0, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

// MuPDF: fz_segment_stext_page (stext-boxer.c:939). THE two-column reading-order
// fix: segment the page into column/region boxes and re-order the flat block
// list so it reads column-by-column (per-column top→bottom, columns
// left→right).
/// Segment `page` into columns/regions and re-order its blocks into reading
/// order in place. On a two-column page this linearises the left column fully,
/// then the right -- never interleaving across the gutter.
///
/// A no-op when the page has fewer than two blocks or no full-span whitespace
/// corridor is found (a genuine single column is left untouched).
pub fn segment_stext_page(page: &mut StextPage) {
    let regions = segment_regions(page);
    order_blocks_by_regions(page, &regions);
}

// MuPDF: fz_new_stext_page_from_page + fz_segment_stext_page (the caller side).
/// Extract page `page_index` of `doc` into a [`StextPage`] and then segment it
/// into reading order -- [`super::page_to_stext`] followed by
/// [`segment_stext_page`].
pub fn page_to_stext_segmented(
    doc: &super::xref::PdfDocument,
    page_index: usize,
    opts: super::structured_text::StextOptions,
) -> super::error::Result<StextPage> {
    let mut page = super::stext_device::page_to_stext(doc, page_index, opts)?;
    segment_stext_page(&mut page);
    Ok(page)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::geometry::{Point, Quad};
    use crate::mupdf::structured_text::{StextChar, StextLine, StextPage, StextTextBlock};

    // -- Fixture builders -------------------------------------------------------

    // Build a single-line text block whose text is `s`, laid out left-to-right
    // starting at (x0, y_top) with the given glyph `size`. Device space: y
    // increases downward (top of page = small y), matching the page CTM's y-flip
    // and MuPDF's boxer convention.
    fn line_block(s: &str, x0: f32, y_top: f32, size: f32) -> StextBlock {
        let mut x = x0;
        let mut chars = Vec::new();
        for c in s.chars() {
            // Every glyph (space included) advances by 0.5em.
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
        let line = StextLine {
            wmode: 0,
            flags: 0,
            dir: Point::new(1.0, 0.0),
            bbox: Rect::new(x0, y_top, x, y_top + size),
            chars,
        };
        StextBlock::Text(StextTextBlock {
            bbox: line.bbox,
            lines: vec![line],
        })
    }

    fn text_of(page: &StextPage) -> String {
        page.blocks
            .iter()
            .filter_map(|b| match b {
                StextBlock::Text(tb) => Some(
                    tb.lines
                        .iter()
                        .map(|l| l.text())
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    // -- The headline two-column fix -------------------------------------------

    #[test]
    fn two_columns_linearize_column_by_column() {
        // Left column x in [50,120], right column x in [320,390]; a wide empty
        // gutter at ~[120,320]. Six one-line blocks, interleaved in draw (Y)
        // order: L1 R1 L2 R2 L3 R3.
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 450.0, 300.0),
            blocks: vec![
                line_block("L1", 50.0, 40.0, 10.0),
                line_block("R1", 320.0, 40.0, 10.0),
                line_block("L2", 50.0, 100.0, 10.0),
                line_block("R2", 320.0, 100.0, 10.0),
                line_block("L3", 50.0, 160.0, 10.0),
                line_block("R3", 320.0, 160.0, 10.0),
            ],
            fonts: Vec::new(),
        };
        segment_stext_page(&mut page);
        assert_eq!(
            text_of(&page),
            "L1 L2 L3 R1 R2 R3",
            "must read column-by-column, not interleaved"
        );
    }

    // -- Single column is not falsely split ------------------------------------

    #[test]
    fn single_column_not_split() {
        // Full-width lines that cross the horizontal midpoint (225): no
        // uninterrupted vertical gutter can exist, so no column split. The
        // words also do not align vertically, so no spurious corridor.
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 450.0, 300.0),
            blocks: vec![
                line_block("alpha beta gamma delta", 40.0, 40.0, 10.0),
                line_block("one two three four five", 40.0, 90.0, 10.0),
                line_block("kappa lambda mu nu xi", 40.0, 140.0, 10.0),
            ],
            fonts: Vec::new(),
        };
        let before = text_of(&page);
        segment_stext_page(&mut page);
        assert_eq!(
            text_of(&page),
            before,
            "single column order must be preserved top-to-bottom"
        );
    }

    // -- Full-width heading interrupts two columns into bands -------------------

    #[test]
    fn full_width_heading_bands() {
        // A full-width heading at the top, then two columns below it.
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 450.0, 300.0),
            blocks: vec![
                // Heading spans nearly the full width.
                line_block("HEADING SPANS THE WHOLE WIDTH HERE OK", 40.0, 30.0, 10.0),
                // Two columns below (draw order interleaved).
                line_block("L1", 50.0, 90.0, 10.0),
                line_block("R1", 320.0, 90.0, 10.0),
                line_block("L2", 50.0, 140.0, 10.0),
                line_block("R2", 320.0, 140.0, 10.0),
            ],
            fonts: Vec::new(),
        };
        segment_stext_page(&mut page);
        // Heading first, then left column, then right column.
        let t = text_of(&page);
        assert!(
            t.starts_with("HEADING"),
            "heading must come first, got {t:?}"
        );
        assert_eq!(
            t, "HEADING SPANS THE WHOLE WIDTH HERE OK L1 L2 R1 R2",
            "bands in order"
        );
    }

    // -- Gutter detection: genuine gutter splits, narrow gap does not -----------

    #[test]
    fn genuine_gutter_splits() {
        // Aligned full-height empty gutter at ~[120,320] => a column split.
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 450.0, 300.0),
            blocks: vec![
                line_block("aa", 50.0, 40.0, 10.0),
                line_block("bb", 320.0, 40.0, 10.0),
                line_block("cc", 50.0, 120.0, 10.0),
                line_block("dd", 320.0, 120.0, 10.0),
            ],
            fonts: Vec::new(),
        };
        segment_stext_page(&mut page);
        assert_eq!(
            text_of(&page),
            "aa cc bb dd",
            "aligned gutter triggers a column split"
        );
    }

    #[test]
    fn narrow_interword_gap_does_not_split() {
        // Single column: one wide-ish inter-word gap per line, but at DIFFERENT
        // x on each line, so no uninterrupted vertical corridor forms => stays
        // one column (top-to-bottom order preserved).
        let mut page = StextPage {
            mediabox: Rect::new(0.0, 0.0, 450.0, 300.0),
            blocks: vec![
                line_block("aaaa bbbb", 40.0, 40.0, 10.0),
                line_block("cccccc dddd", 40.0, 90.0, 10.0),
                line_block("ee ffffffff", 40.0, 140.0, 10.0),
            ],
            fonts: Vec::new(),
        };
        let before = text_of(&page);
        segment_stext_page(&mut page);
        assert_eq!(
            text_of(&page),
            before,
            "loose inter-word spacing is not a column boundary"
        );
    }

    // -- Boxer unit: a fed box carves the whitespace ----------------------------

    #[test]
    fn feed_carves_four_bands() {
        let mut boxer = Boxer::create(Rect::new(0.0, 0.0, 100.0, 100.0), true);
        assert_eq!(boxer.list.len(), 1);
        // Feed a central occupied box; the surrounding whitespace splits.
        boxer.feed(Rect::new(40.0, 40.0, 60.0, 60.0));
        // Full-height left band [0,40]x[0,100] and right band [60,100]x[0,100]
        // must both be present (they are the vertical corridors either side).
        let has_left = boxer
            .list
            .iter()
            .any(|r| r.x0 <= 0.0 && r.x1 >= 40.0 && r.y0 <= 0.0 && r.y1 >= 100.0);
        let has_right = boxer
            .list
            .iter()
            .any(|r| r.x0 <= 60.0 && r.x1 >= 100.0 && r.y0 <= 0.0 && r.y1 >= 100.0);
        assert!(has_left, "left full-height corridor present");
        assert!(has_right, "right full-height corridor present");
    }
}
