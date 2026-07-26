//! Ported from MuPDF `source/fitz/draw-edge.c` (the Global Edge List scan
//! converter: `fz_insert_gel`, `fz_scan_convert_aa`, `non_zero_winding_aa` /
//! `even_odd_aa`, `undelta_aa`) and the span compositing of
//! `source/fitz/draw-paint.c` (`fz_paint_span` -> the source-over painter)
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the *algorithm* (edge list,
//! per-subscanline winding accumulation with the nonzero and even-odd rules, and
//! anti-aliased coverage) follows MuPDF; the fixed-point Bresenham GEL is
//! re-expressed as a coverage-accumulating scanline sweep (analytic in x,
//! super-sampled in y) that is numerically simpler while producing the same
//! nonzero/even-odd anti-aliased fill. See docs/ACKNOWLEDGEMENTS.md
//! ("PDF & document-extraction references").
//!
//! # The core: anti-aliased polygon fill
//!
//! [`fill_polygons`] is the legibility primitive the whole draw device rests on.
//! Given a set of **already-flattened, device-space** closed polygons (produced
//! by [`super::draw_path`]), a winding rule, a colour and an alpha, it paints an
//! anti-aliased fill into a [`Pixmap`].
//!
//! MuPDF's `fz_scan_convert_aa` groups `fz_aa_vscale` sub-scanlines into one
//! output row and, per sub-scanline, walks the active edges left-to-right
//! accumulating a per-pixel *winding delta*; `undelta_aa` integrates the deltas
//! into 0..255 coverage. This port keeps the same shape -- edges, per-sub-scanline
//! crossings, winding accumulation, nonzero vs even-odd -- but:
//! * super-samples only in **y** ([`SUBSAMPLES`] sub-scanlines per row), and
//! * computes **x** coverage *analytically* (a span's exact fractional overlap of
//!   each pixel cell), rather than MuPDF's horizontal `hscale` super-sampling.
//!
//! The result is the same anti-aliased nonzero/even-odd fill; the numeric detail
//! differs (MuPDF's exact fixed-point GEL is not bit-reproduced -- see the module
//! header). A single [`fill_polygons`] call scans only its own bounding box, so
//! the per-glyph / per-path calls the device makes stay cheap; there is **no**
//! global active-edge table across calls (a deliberate simplicity/perf trade for
//! the small fills this device emits).

use super::geometry::{IRect, Point};
use super::pixmap::Pixmap;

/// The winding rule for a fill (`PDF` `f`/`F` vs `f*`, MuPDF's `even_odd` flag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    /// Nonzero winding (`f` / `B`): a point is inside when the signed crossing
    /// count is non-zero. `non_zero_winding_aa`.
    NonZero,
    /// Even-odd (`f*` / `B*`): inside when the crossing count is odd.
    /// `even_odd_aa`.
    EvenOdd,
}

/// Vertical super-sampling factor (sub-scanlines per output row). MuPDF's default
/// `fz_aa_vscale` is 15; 4 keeps the small per-glyph / per-path fills this device
/// emits fast while still smoothing the staircase. x coverage is exact
/// (analytic), so horizontal edges are already crisp regardless of this value.
pub const SUBSAMPLES: u32 = 4;

/// One monotonic-in-y edge: the segment from `(x_top, ytop)` sloping to `ybot`.
struct EdgeSeg {
    ytop: f32,
    ybot: f32,
    /// x at `y == ytop`.
    x_top: f32,
    /// dx/dy along the edge.
    dxdy: f32,
    /// +1 if the original segment ran downward (increasing y), else -1. This is
    /// MuPDF's `edge->ydir` -- the winding contribution.
    dir: i32,
}

// MuPDF: fz_insert_gel_raw (draw-edge.c:111) -- reject horizontal edges, orient
// top-to-bottom and record the winding sign.
/// Build the edge list from closed polygons. Each polygon is implicitly closed
/// (last vertex joined back to the first).
fn build_edges(subpaths: &[Vec<Point>]) -> Vec<EdgeSeg> {
    let mut edges = Vec::new();
    for poly in subpaths {
        if poly.len() < 2 {
            continue;
        }
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            if a.y == b.y {
                continue; // horizontal: contributes no crossing
            }
            let (top, bot, dir) = if a.y < b.y { (a, b, 1) } else { (b, a, -1) };
            let dxdy = (bot.x - top.x) / (bot.y - top.y);
            edges.push(EdgeSeg { ytop: top.y, ybot: bot.y, x_top: top.x, dxdy, dir });
        }
    }
    edges
}

/// The integer device-space bounds of `subpaths`, intersected with `clip` and
/// the pixmap, as a half-open `[x0, x1) × [y0, y1)` rectangle. Empty if there is
/// nothing to draw.
fn fill_bounds(subpaths: &[Vec<Point>], pix: &Pixmap, clip: IRect) -> IRect {
    let mut x0 = f32::INFINITY;
    let mut y0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for poly in subpaths {
        for p in poly {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
    }
    if !x0.is_finite() || x1 <= x0 || y1 <= y0 {
        return IRect::EMPTY;
    }
    // Floor the low edge, ceil the high edge, then clamp to pixmap ∩ clip.
    let b = IRect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32);
    b.intersect(pix.bbox()).intersect(clip)
}

/// Add a horizontal span `[a, b)` (device x, `a <= b`) to the coverage row
/// `cov` (indexed from `x0`) with weight `w`, using exact fractional per-pixel
/// overlap -- this is where the x anti-aliasing comes from.
fn add_span(cov: &mut [f32], x0: i32, a: f32, b: f32, w: f32) {
    if b <= a {
        return;
    }
    let lo = a.max(x0 as f32);
    let hi = b.min((x0 + cov.len() as i32) as f32);
    if hi <= lo {
        return;
    }
    let mut px = lo.floor() as i32;
    let last = (hi.ceil() as i32) - 1;
    while px <= last {
        let cell_lo = px as f32;
        let cell_hi = cell_lo + 1.0;
        let covered = hi.min(cell_hi) - lo.max(cell_lo);
        if covered > 0.0 {
            let idx = (px - x0) as usize;
            cov[idx] += covered * w;
        }
        px += 1;
    }
}

// MuPDF: the source-over span painter fz_paint_span_... (draw-paint.c) reduced to
// a single straight (non-premultiplied) source-over composite.
/// Composite `color` at coverage `a` (0..=1) onto the pixel at byte offset `o`.
fn composite(pix: &mut Pixmap, o: usize, color: &[u8], a: f32) {
    let n = pix.n as usize;
    let ia = 1.0 - a;
    // Colour components (all but a trailing alpha, if any).
    let ccount = if pix.alpha { n - 1 } else { n };
    for k in 0..ccount {
        let src = *color.get(k).unwrap_or(&0) as f32;
        let dst = pix.samples[o + k] as f32;
        pix.samples[o + k] = (src * a + dst * ia).round().clamp(0.0, 255.0) as u8;
    }
    if pix.alpha {
        let dst = pix.samples[o + n - 1] as f32;
        // source-over alpha: a + dst*(1-a)
        pix.samples[o + n - 1] = (a * 255.0 + dst * ia).round().clamp(0.0, 255.0) as u8;
    }
}

// MuPDF: fz_scan_convert_aa (draw-edge.c:555) fused with the nonzero/even-odd
// winding functions (draw-edge.c) and undelta_aa's coverage integration.
/// Paint an anti-aliased fill of `subpaths` (device-space closed polygons) into
/// `pix`, clipped to `clip`, with `color` (`pix.n` colour components, alpha
/// excluded) at overall `alpha` (0..=1).
///
/// This is the rasterizer entry the [`DrawDevice`](super::draw_device) drives for
/// every filled path and every glyph box.
pub fn fill_polygons(pix: &mut Pixmap, subpaths: &[Vec<Point>], rule: FillRule, color: &[u8], alpha: f32, clip: IRect) {
    if alpha <= 0.0 {
        return;
    }
    let bounds = fill_bounds(subpaths, pix, clip);
    if bounds.is_empty() {
        return;
    }
    let edges = build_edges(subpaths);
    if edges.is_empty() {
        return;
    }

    let x0 = bounds.x0;
    let width = bounds.width() as usize;
    let inv_ss = 1.0 / SUBSAMPLES as f32;

    let mut cov = vec![0.0f32; width];
    let mut xs: Vec<(f32, i32)> = Vec::new();

    for py in bounds.y0..bounds.y1 {
        cov.iter_mut().for_each(|c| *c = 0.0);

        for s in 0..SUBSAMPLES {
            let sy = py as f32 + (s as f32 + 0.5) * inv_ss;

            // Collect this sub-scanline's crossings with the active edges.
            xs.clear();
            for e in &edges {
                if sy >= e.ytop && sy < e.ybot {
                    let x = e.x_top + (sy - e.ytop) * e.dxdy;
                    xs.push((x, e.dir));
                }
            }
            if xs.len() < 2 {
                continue;
            }
            xs.sort_by(|p, q| p.0.total_cmp(&q.0));

            // Walk the sorted crossings, applying the winding rule to mark the
            // inside spans, and accumulate their coverage.
            let mut winding = 0i32;
            for i in 0..xs.len() - 1 {
                winding += xs[i].1;
                let inside = match rule {
                    FillRule::NonZero => winding != 0,
                    // even-odd: the i-th gap is inside when an odd number of
                    // edges lie to its left (i.e. i is even after 0-indexing the
                    // gap that opens at crossing i).
                    FillRule::EvenOdd => (i & 1) == 0,
                };
                if inside {
                    add_span(&mut cov, x0, xs[i].0, xs[i + 1].0, inv_ss);
                }
            }
        }

        // Composite the accumulated coverage row.
        for (i, &c) in cov.iter().enumerate() {
            if c <= 0.0 {
                continue;
            }
            let a = (c.min(1.0)) * alpha;
            if a <= 0.0 {
                continue;
            }
            if let Some(o) = pix.offset(x0 + i as i32, py) {
                composite(pix, o, color, a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
        vec![
            Point::new(x0, y0),
            Point::new(x1, y0),
            Point::new(x1, y1),
            Point::new(x0, y1),
        ]
    }

    #[test]
    fn filled_rect_is_black_inside_white_outside() {
        let mut pix = Pixmap::new_rgb(20, 20);
        pix.clear_with_value(0xff);
        let bb = pix.bbox();
        // Fill an axis-aligned rectangle black.
        fill_polygons(&mut pix, &[rect(5.0, 5.0, 15.0, 15.0)], FillRule::NonZero, &[0, 0, 0], 1.0, bb);
        // Deep inside -> near 0.
        assert!(pix.luma(10, 10).unwrap() < 5, "inside luma {}", pix.luma(10, 10).unwrap());
        // Well outside -> still white.
        assert_eq!(pix.luma(1, 1).unwrap(), 255);
        assert_eq!(pix.luma(18, 18).unwrap(), 255);
    }

    #[test]
    fn edges_are_antialiased() {
        // A rectangle with a fractional edge at x=5.5 should leave the boundary
        // column partially covered (not a hard 0/255 step).
        let mut pix = Pixmap::new_rgb(20, 20);
        pix.clear_with_value(0xff);
        let bb = pix.bbox();
        fill_polygons(&mut pix, &[rect(5.5, 5.0, 15.0, 15.0)], FillRule::NonZero, &[0, 0, 0], 1.0, bb);
        let edge = pix.luma(5, 10).unwrap();
        assert!(edge > 5 && edge < 250, "AA edge luma should be gray, got {edge}");
    }

    #[test]
    fn nonzero_and_even_odd_differ_on_nested_same_winding() {
        // Outer + inner rectangle wound the SAME direction. Nonzero fills the
        // whole outer (winding 2 in the middle); even-odd leaves an inner hole.
        let outer = rect(2.0, 2.0, 18.0, 18.0);
        let inner = rect(6.0, 6.0, 14.0, 14.0);

        let mut nz = Pixmap::new_rgb(20, 20);
        nz.clear_with_value(0xff);
        let bb = nz.bbox();
        fill_polygons(&mut nz, &[outer.clone(), inner.clone()], FillRule::NonZero, &[0, 0, 0], 1.0, bb);

        let mut eo = Pixmap::new_rgb(20, 20);
        eo.clear_with_value(0xff);
        let bb = eo.bbox();
        fill_polygons(&mut eo, &[outer, inner], FillRule::EvenOdd, &[0, 0, 0], 1.0, bb);

        // Centre pixel: nonzero -> filled (black), even-odd -> hole (white).
        assert!(nz.luma(10, 10).unwrap() < 5, "nonzero centre should be black");
        assert_eq!(eo.luma(10, 10).unwrap(), 255, "even-odd centre should be a hole");
        // A pixel in the outer ring is filled under both rules.
        assert!(nz.luma(4, 10).unwrap() < 5);
        assert!(eo.luma(4, 10).unwrap() < 5);
    }

    #[test]
    fn alpha_blends_halfway() {
        let mut pix = Pixmap::new_rgb(10, 10);
        pix.clear_with_value(0xff);
        let bb = pix.bbox();
        fill_polygons(&mut pix, &[rect(2.0, 2.0, 8.0, 8.0)], FillRule::NonZero, &[0, 0, 0], 0.5, bb);
        // Half-alpha black over white -> ~128.
        let v = pix.luma(5, 5).unwrap();
        assert!((v as i32 - 128).abs() <= 2, "half-alpha luma {v}");
    }
}
