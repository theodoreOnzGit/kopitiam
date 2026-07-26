//! Pragmatic page → text-line finder (Phase 8): turn a binarized page into
//! individual grayscale text-line images ([`GrayLine`]) in reading order.
//!
//! # Provenance — an original pragmatic reimplementation (clean-room / adaptation)
//!
//! This module is **not** a translation of Tesseract's `textord`. Tesseract's
//! full column/table analysis (`src/textord/colfind.cpp`, `colpartition*.cpp`,
//! `makerow.cpp`, `bbgrid.cpp`, … ~35k LOC) models blob grids, column
//! partitions, tab-stops, and baselines; per the project's OCR manifest that is
//! out of scope. Instead this is a few-hundred-line, original Rust
//! reimplementation that is good enough for born-digital / cleanly scanned 1–2
//! column scientific pages.
//!
//! The *ideas* were studied from three read-only vendored references, but the
//! code below was written fresh (clean-room / adaptation, not a line-by-line
//! port); their copyright notices are recorded here per the crate-wide
//! provenance discipline (see docs/ACKNOWLEDGEMENTS.md, `docs/ai-decisions/
//! AID-0051`):
//!
//! * **Tesseract** `src/textord/makerow.cpp` (© 2013 Google Inc., Author: Ray
//!   Smith, Apache-2.0, vendored at `crates/kopitiam-ocr/vendor/tesseract`,
//!   commit `db0ec62`) — the idea of a horizontal ink-projection profile over
//!   rows, with adaptive line-height/gap estimation to group rows into text
//!   lines. We reimplement only the projection-profile shape, not Tesseract's
//!   blob-based row building.
//! * **Leptonica** `src/pix1.c` (`pixCountPixelsByRow` / `…ByColumn`), BSD-2-
//!   Clause, © 2001-2020 Leptonica (Dan Bloomberg), vendored at
//!   `crates/kopitiam-ocr/vendor/leptonica` — the idea of a per-row / per-column
//!   foreground-pixel projection as the primitive for band and gutter finding.
//! * The **MuPDF boxer** column-gutter principle (full-height whitespace splits
//!   a page into columns), applied here on binary pixels rather than glyph
//!   boxes.
//!
//! Both compatibly-licensed upstreams are one-way compatible with KOPITIAM's
//! AGPL-3.0-only license; their notices travel with this header.
//!
//! # What this is — the line-finding phase
//!
//! The recognizer (Phase 6) consumes one normalized [`GrayLine`] at a time. This
//! phase produces those lines from a page raster that Phase 7 preprocessing has
//! turned into a [`BinaryImage`] (0 = ink, 255 = paper) plus its source
//! [`GrayImage`]. (That page pair will, in a future phase, come from the
//! rasterizer's output run through [`crate::to_gray`] + [`crate::sauvola_binarize`]
//! / [`crate::otsu_binarize`].) The pipeline is:
//!
//! 1. **Column split** — a full-height vertical-whitespace projection finds
//!    gutters (columns whose whole height is ~blank) and splits the page into
//!    left-to-right column regions. A page with no interior gutter is a single
//!    column.
//! 2. **Line bands** — within each column, a horizontal ink-projection profile
//!    (ink pixels per row) gives runs of inked rows separated by low-ink gaps.
//!    An adaptive line-height estimate merges small gaps (descenders/diacritics)
//!    back into their line while keeping true inter-line gaps as splits.
//! 3. **Per-line crop** — each line band is trimmed to its horizontal ink extent
//!    and the corresponding rectangle is cropped from the **gray** image (the
//!    recognizer wants grayscale, not the binary) into a [`GrayLine`].
//!
//! Lines are emitted in reading order: within a column top-to-bottom, columns
//! left-to-right.
//!
//! # Deferred
//!
//! Deskew (this assumes roughly horizontal text — a skewed page projects across
//! line boundaries and blurs the bands), curved/warped baselines, Tesseract's
//! full `textord` column/table/tab-stop analysis, reading order across complex
//! multi-column/figure layouts, touching-line separation (two lines with no
//! blank row between them are not split), and speckle/noise removal (a single
//! stray ink pixel is treated as ink) are all out of scope for this pragmatic
//! finder. Each is a place a real `textord` port would go further.

use crate::image::{BinaryImage, GrayImage};
use crate::lstmrecognizer::GrayLine;

/// A pixel counts as ink (foreground) when it is darker than mid-gray. A
/// [`BinaryImage`] only ever holds `0`/`255`, so this is exact there; the `<128`
/// form also tolerates a gray input used directly.
#[inline]
fn is_ink(v: u8) -> bool {
    v < 128
}

/// Finds the page's text lines and returns them in reading order.
///
/// `binary` drives detection (0 = ink, 255 = paper); the returned [`GrayLine`]s
/// are cropped from `gray`, which must have the same dimensions as `binary`.
/// A blank page (or a dimension mismatch) yields an empty vector.
///
/// The algorithm is: **column split** (full-height whitespace gutters) →
/// **projection-profile line bands** per column (with adaptive small-gap merge)
/// → **gray crop** trimmed to each line's ink extent → **reading order**
/// (per column top-to-bottom, columns left-to-right).
pub fn find_text_lines(binary: &BinaryImage, gray: &GrayImage) -> Vec<GrayLine> {
    if binary.width == 0 || binary.height == 0 {
        return Vec::new();
    }
    // The crop reads `gray` at the coordinates `binary` selects; they must align.
    if gray.width != binary.width || gray.height != binary.height {
        return Vec::new();
    }
    let mut lines = Vec::new();
    for (x0, x1) in split_columns(binary) {
        lines.extend(find_text_lines_in_column(binary, gray, x0, x1));
    }
    lines
}

/// Finds the text lines inside one column region (`x0..x1`, half-open) of the
/// page, top-to-bottom. Used by [`find_text_lines`] once per column; exposed so
/// a caller with its own column geometry can drive line-finding directly.
///
/// `x1` is clamped to the image width; an empty or out-of-range range yields no
/// lines.
pub fn find_text_lines_in_column(
    binary: &BinaryImage,
    gray: &GrayImage,
    x0: usize,
    x1: usize,
) -> Vec<GrayLine> {
    let w = binary.width;
    let h = binary.height;
    let x1 = x1.min(w);
    if x0 >= x1 || h == 0 {
        return Vec::new();
    }
    let col_w = x1 - x0;
    // A row is "inked" when its ink count exceeds a tolerance that scales with
    // the column width, so a lone speckle in an otherwise blank row does not
    // start a band on a large page (0 for the small hermetic test images).
    let row_tol = col_w / 100;

    // Horizontal projection profile: ink pixels per row within the column.
    let mut row_ink = vec![0usize; h];
    for (y, ink) in row_ink.iter_mut().enumerate() {
        let row = &binary.pixels[y * w + x0..y * w + x1];
        *ink = row.iter().filter(|&&v| is_ink(v)).count();
    }

    // Raw bands: maximal runs of inked rows.
    let mut bands: Vec<(usize, usize)> = Vec::new();
    let mut y = 0;
    while y < h {
        if row_ink[y] > row_tol {
            let start = y;
            while y < h && row_ink[y] > row_tol {
                y += 1;
            }
            bands.push((start, y));
        } else {
            y += 1;
        }
    }
    if bands.is_empty() {
        return Vec::new();
    }

    // Adaptive small-gap merge: a gap up to half the typical line height is
    // treated as intra-line (a descender or diacritic dips into the blank rows),
    // while larger gaps stay as inter-line splits.
    let heights: Vec<usize> = bands.iter().map(|&(a, b)| b - a).collect();
    let line_h = median(&heights);
    let max_intra_gap = (line_h / 2).max(1);

    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut cur = bands[0];
    for &(bs, be) in &bands[1..] {
        let gap = bs - cur.1;
        if gap <= max_intra_gap {
            cur.1 = be;
        } else {
            merged.push(cur);
            cur = (bs, be);
        }
    }
    merged.push(cur);

    // Crop each band to its horizontal ink extent, out of the gray image.
    merged
        .into_iter()
        .filter_map(|(y0, y1)| crop_line(binary, gray, x0, x1, y0, y1))
        .collect()
}

/// Splits the page into left-to-right column regions on full-height vertical
/// whitespace gutters. Returns `(x0, x1)` half-open ranges; a page with no
/// interior gutter yields a single region spanning the inked content, and a
/// blank page yields an empty vector.
fn split_columns(binary: &BinaryImage) -> Vec<(usize, usize)> {
    let w = binary.width;
    let h = binary.height;
    // A column is "blank" when its full-height ink count is within tolerance,
    // which scales with page height to tolerate gutter speckle on large scans
    // (0 for the small hermetic test images).
    let col_tol = h / 100;

    // Vertical projection profile: ink pixels per column over the whole height.
    let mut col_ink = vec![0usize; w];
    for y in 0..h {
        let row = &binary.pixels[y * w..y * w + w];
        for (x, &v) in row.iter().enumerate() {
            if is_ink(v) {
                col_ink[x] += 1;
            }
        }
    }

    // Trim outer margins to the inked content.
    let first = col_ink.iter().position(|&c| c > col_tol);
    let last = col_ink.iter().rposition(|&c| c > col_tol);
    let (x_start, x_end) = match (first, last) {
        (Some(a), Some(b)) => (a, b + 1),
        _ => return Vec::new(), // blank page
    };

    // A gutter must be wider than an incidental blank stripe; scale a minimum
    // with page width (floor 2). Full-height blankness already excludes ordinary
    // inter-word spaces (other lines ink those columns), so this only guards
    // against a coincidental thin aligned gap.
    let min_gutter = (w / 25).max(2);

    let mut columns: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = x_start;
    let mut x = x_start;
    while x < x_end {
        if col_ink[x] <= col_tol {
            let run_start = x;
            while x < x_end && col_ink[x] <= col_tol {
                x += 1;
            }
            if x - run_start >= min_gutter {
                if run_start > seg_start {
                    columns.push((seg_start, run_start));
                }
                seg_start = x;
            }
        } else {
            x += 1;
        }
    }
    if x_end > seg_start {
        columns.push((seg_start, x_end));
    }
    columns
}

/// Crops the rectangle `[x0, x1) × [y0, y1)`, trimmed to its horizontal ink
/// extent, out of `gray` into a [`GrayLine`]. Returns `None` if the band holds
/// no ink.
fn crop_line(
    binary: &BinaryImage,
    gray: &GrayImage,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> Option<GrayLine> {
    let w = binary.width;

    // Left/right-most columns holding ink anywhere in the band.
    let mut left: Option<usize> = None;
    let mut right = x0;
    for x in x0..x1 {
        let mut has_ink = false;
        for y in y0..y1 {
            if is_ink(binary.pixels[y * w + x]) {
                has_ink = true;
                break;
            }
        }
        if has_ink {
            if left.is_none() {
                left = Some(x);
            }
            right = x;
        }
    }
    let left = left?;
    let cx0 = left;
    let cx1 = right + 1;
    let cw = cx1 - cx0;
    let ch = y1 - y0;

    let gw = gray.width;
    let mut pixels = vec![0u8; cw * ch];
    for (ry, y) in (y0..y1).enumerate() {
        for (rx, x) in (cx0..cx1).enumerate() {
            pixels[ry * cw + rx] = gray.pixels[y * gw + x];
        }
    }
    Some(GrayLine::new(cw, ch, pixels))
}

/// The median of a non-empty slice of band heights (average of the two middle
/// values, floored, for an even count). Returns `0` for an empty slice.
fn median(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{BINARY_BG, BINARY_FG};

    /// Build a `(binary, gray)` page pair from a closure that returns `true`
    /// where there is ink. The gray page encodes each pixel's `x` (so crops can
    /// be checked positionally): ink pixels get gray value `x.min(200)`, paper
    /// gets `255`.
    fn make_page(
        w: usize,
        h: usize,
        ink: impl Fn(usize, usize) -> bool,
    ) -> (BinaryImage, GrayImage) {
        let mut bpix = vec![BINARY_BG; w * h];
        let mut gpix = vec![255u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if ink(x, y) {
                    bpix[y * w + x] = BINARY_FG;
                    gpix[y * w + x] = x.min(200) as u8;
                }
            }
        }
        (
            BinaryImage::new(w, h, bpix),
            GrayImage::new(w, h, gpix),
        )
    }

    #[test]
    fn three_bands_give_three_lines_with_expected_y_extents() {
        // Three full-width bands of ink at rows [2,7), [12,17), [22,27),
        // separated by blank rows (gap 5 > line_h/2 = 2, so no merge).
        let bands = [(2usize, 7usize), (12, 17), (22, 27)];
        let (binary, gray) = make_page(30, 30, |_x, y| {
            bands.iter().any(|&(a, b)| y >= a && y < b)
        });
        let lines = find_text_lines(&binary, &gray);
        assert_eq!(lines.len(), 3);
        for (line, &(a, b)) in lines.iter().zip(bands.iter()) {
            assert_eq!(line.height, b - a);
            // Full-width ink -> crop spans the whole width.
            assert_eq!(line.width, 30);
        }
    }

    #[test]
    fn blank_page_gives_no_lines() {
        let (binary, gray) = make_page(20, 20, |_x, _y| false);
        assert!(find_text_lines(&binary, &gray).is_empty());
    }

    #[test]
    fn descender_below_main_band_stays_one_line() {
        // Main band [10,16) full width; a small detached mark at [17,19) in a
        // narrow x-range, one blank row (16) between. Small gap must merge.
        let (binary, gray) = make_page(30, 30, |x, y| {
            let main = (10..16).contains(&y);
            let mark = (17..19).contains(&y) && (12..15).contains(&x);
            main || mark
        });
        let lines = find_text_lines(&binary, &gray);
        assert_eq!(lines.len(), 1, "descender mark must not split the line");
        // The merged line spans from the main band's top through the mark.
        assert_eq!(lines[0].height, 19 - 10);
    }

    #[test]
    fn two_column_page_returns_lines_in_column_order() {
        // Left column x in [4,24), right column x in [36,56), blank gutter
        // [24,36). Left has lines at y[2,8) and y[20,26); right at y[10,16) and
        // y[28,34). Row order would interleave them; column order must not.
        let w = 60;
        let (binary, gray) = make_page(w, 40, |x, y| {
            let left = (4..24).contains(&x)
                && ((2..8).contains(&y) || (20..26).contains(&y));
            let right = (36..56).contains(&x)
                && ((10..16).contains(&y) || (28..34).contains(&y));
            left || right
        });
        let lines = find_text_lines(&binary, &gray);
        assert_eq!(lines.len(), 4);
        // First two lines belong to the left column (ink trimmed to x in [4,24)),
        // last two to the right column (x in [36,56)) — not row-interleaved.
        // The gray encoding stores x, so the first pixel's column is recoverable
        // via the crop's known left edge; check the crop widths/positions.
        // Left crops start at x=4 (leftmost ink); right crops start at x=36.
        // We verify via the minimum gray value in each crop (== leftmost x).
        let leftmost = |line: &GrayLine| *line.pixels.iter().min().unwrap();
        assert_eq!(leftmost(&lines[0]), 4);
        assert_eq!(leftmost(&lines[1]), 4);
        assert_eq!(leftmost(&lines[2]), 36);
        assert_eq!(leftmost(&lines[3]), 36);
    }

    #[test]
    fn per_line_crop_trims_leading_and_trailing_blank_columns() {
        // A single band [5,10) with ink only in x[8,15) of a 40-wide page.
        let (binary, gray) = make_page(40, 20, |x, y| {
            (5..10).contains(&y) && (8..15).contains(&x)
        });
        let lines = find_text_lines(&binary, &gray);
        assert_eq!(lines.len(), 1);
        // Crop trimmed to the ink extent: width 15-8 = 7, not the full 40.
        assert_eq!(lines[0].width, 7);
        assert_eq!(lines[0].height, 5);
        // Leftmost cropped column is x=8 (gray encodes x).
        assert_eq!(*lines[0].pixels.iter().min().unwrap(), 8);
    }

    #[test]
    fn dimension_mismatch_yields_no_lines() {
        let (binary, _) = make_page(10, 10, |x, y| x == y);
        let gray = GrayImage::filled(8, 8, 255);
        assert!(find_text_lines(&binary, &gray).is_empty());
    }

    #[test]
    fn single_column_when_no_full_height_gutter() {
        // Two side-by-side blocks but each row's blank middle is filled on some
        // other row is NOT the case here — instead a full-height gutter should
        // split. Confirm the negative: a page with content bridging the middle
        // on at least one row stays single-column.
        let (binary, gray) = make_page(40, 20, |x, y| {
            // A band that spans the full width on row 10 bridges any gutter.
            (2..6).contains(&y) && x < 15
                || (2..6).contains(&y) && x >= 25
                || y == 10 // full-width bridge row
        });
        let cols = split_columns(&binary);
        assert_eq!(cols.len(), 1, "a full-width bridge row prevents a gutter");
        // And it still finds the line(s).
        assert!(!find_text_lines(&binary, &gray).is_empty());
    }
}
