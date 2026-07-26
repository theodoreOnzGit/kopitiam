//! Ported from MuPDF `source/fitz/path.c` (the `fz_path` moveto/lineto/curveto/
//! closepath builder and `fz_walk_path`) and `source/fitz/draw-path.c`
//! (`fz_flatten_fill_path`, the `bezier`/`quadratic` adaptive subdivision, and
//! the `line` primitive) (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the path
//! model, the bezier flattening (midpoint subdivision to a flatness tolerance),
//! and the fill-flatten walk follow MuPDF; the code is re-expressed in idiomatic
//! Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Path -> device-space polygons
//!
//! [`Path`] is the vector outline the [`DrawDevice`](super::draw_device) fills or
//! strokes: a list of sub-paths, each a run of moveto/lineto/curveto commands.
//! [`Path::flatten`] transforms every control point by the CTM (so flattening
//! happens in device pixels, as MuPDF's `line`/`bezier` do) and subdivides curves
//! to straight segments within [`FLATNESS`], returning closed device-space
//! polygons ready for [`fill_polygons`](super::draw_edge::fill_polygons).
//!
//! ## Stroking (approximate)
//!
//! MuPDF's stroker (`fz_flatten_stroke_path`) builds true offset outlines with
//! miter/round/bevel joins and butt/round/square caps. This port takes the
//! sanctioned **width-expansion** shortcut: [`Path::stroke_to_polygons`] emits one
//! rectangle per flattened segment plus a small square at each vertex to bridge
//! the joins. Good enough for legible thin rules and boxes; joins/caps are
//! approximate (see the `// TODO(draw)` markers).

use super::geometry::Matrix;
use super::geometry::Point;

/// The default fill flatness in **device pixels**. MuPDF uses `0.3 / expansion`
/// clamped to `>= 0.001` (draw-device.c:699); flattening here is done directly in
/// device space, so `0.3 px` is the constant tolerance.
pub const FLATNESS: f32 = 0.3;

/// Recursion cap on bezier subdivision (`MAX_DEPTH` in draw-path.c).
const MAX_DEPTH: u32 = 36;

/// One command of a sub-path, in path (user) space.
#[derive(Clone, Copy, Debug)]
enum Cmd {
    /// Start a new sub-path at this point.
    MoveTo(Point),
    /// Straight segment to this point.
    LineTo(Point),
    /// Cubic bezier with control points `c1`, `c2` and endpoint `to`.
    CurveTo(Point, Point, Point),
    /// Close the current sub-path (segment back to its start).
    Close,
}

/// A vector outline: a sequence of moveto / lineto / curveto / close commands,
/// mirroring `fz_path`. Build with the `move_to` / `line_to` / `curve_to` /
/// `close` / `rect` methods, then [`flatten`](Path::flatten) or
/// [`stroke_to_polygons`](Path::stroke_to_polygons).
#[derive(Clone, Debug, Default)]
pub struct Path {
    cmds: Vec<Cmd>,
}

impl Path {
    /// An empty path.
    pub fn new() -> Path {
        Path { cmds: Vec::new() }
    }

    // MuPDF: fz_moveto (path.c:290).
    pub fn move_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.cmds.push(Cmd::MoveTo(Point::new(x, y)));
        self
    }

    // MuPDF: fz_lineto (path.c:329).
    pub fn line_to(&mut self, x: f32, y: f32) -> &mut Self {
        self.cmds.push(Cmd::LineTo(Point::new(x, y)));
        self
    }

    // MuPDF: fz_curveto (path.c:376) -- cubic bezier.
    pub fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> &mut Self {
        self.cmds.push(Cmd::CurveTo(Point::new(x1, y1), Point::new(x2, y2), Point::new(x3, y3)));
        self
    }

    // MuPDF: fz_closepath (path.c:564).
    pub fn close(&mut self) -> &mut Self {
        self.cmds.push(Cmd::Close);
        self
    }

    // MuPDF: fz_rectto (path.c:606) -- a closed 4-point rectangle sub-path.
    /// Append a closed rectangle sub-path `[x0, y0]..[x1, y1]`.
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> &mut Self {
        self.move_to(x0, y0).line_to(x1, y0).line_to(x1, y1).line_to(x0, y1).close()
    }

    // MuPDF: fz_flatten_fill_path (draw-path.c) driving fz_walk_path with the
    // bezier() subdivision; each generated point is transformed by ctm ("in
    // device space") before it enters the polygon.
    /// Transform every control point by `ctm` and flatten curves to straight
    /// segments (within [`FLATNESS`] device pixels), returning one **closed**
    /// device-space polygon per sub-path. Empty sub-paths are dropped.
    ///
    /// For filling, every sub-path is implicitly closed (the rasterizer joins the
    /// last vertex back to the first), matching `fz_flatten_fill_path`.
    pub fn flatten(&self, ctm: Matrix) -> Vec<Vec<Point>> {
        let mut out: Vec<Vec<Point>> = Vec::new();
        let mut cur: Vec<Point> = Vec::new();
        // The current point and the start-of-sub-path point, in device space.
        let mut here = Point::new(0.0, 0.0);
        let mut start = here;

        let flush = |out: &mut Vec<Vec<Point>>, cur: &mut Vec<Point>| {
            if cur.len() >= 2 {
                out.push(std::mem::take(cur));
            } else {
                cur.clear();
            }
        };

        for cmd in &self.cmds {
            match *cmd {
                Cmd::MoveTo(p) => {
                    flush(&mut out, &mut cur);
                    here = p.transform(ctm);
                    start = here;
                    cur.push(here);
                }
                Cmd::LineTo(p) => {
                    here = p.transform(ctm);
                    cur.push(here);
                }
                Cmd::CurveTo(c1, c2, to) => {
                    let d1 = c1.transform(ctm);
                    let d2 = c2.transform(ctm);
                    let d3 = to.transform(ctm);
                    bezier(&mut cur, here, d1, d2, d3, 0);
                    here = d3;
                }
                Cmd::Close => {
                    // Filling closes implicitly; just return to the sub-path start.
                    here = start;
                    flush(&mut out, &mut cur);
                    cur.push(start);
                }
            }
        }
        flush(&mut out, &mut cur);
        out
    }

    // MuPDF: fz_flatten_stroke_path (draw-path.c) -- here reduced to the width
    // expansion the module header describes.
    /// Flatten and expand the path to filled polygons approximating a stroke of
    /// device-space `line_width`, using `ctm` for the geometry. One rectangle per
    /// segment plus a bridging square per vertex. Fill the result with
    /// [`FillRule::NonZero`](super::draw_edge::FillRule::NonZero).
    pub fn stroke_to_polygons(&self, ctm: Matrix, line_width: f32) -> Vec<Vec<Point>> {
        // A hairline (width 0 or sub-pixel) still draws ~1px, as MuPDF does.
        let hw = (line_width * 0.5).max(0.35);
        let polylines = self.flatten_open(ctm);
        let mut polys: Vec<Vec<Point>> = Vec::new();

        for line in &polylines {
            for w in line.windows(2) {
                let (a, b) = (w[0], w[1]);
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 {
                    continue;
                }
                // Unit normal, scaled to the half-width.
                let nx = -dy / len * hw;
                let ny = dx / len * hw;
                polys.push(vec![
                    Point::new(a.x + nx, a.y + ny),
                    Point::new(b.x + nx, b.y + ny),
                    Point::new(b.x - nx, b.y - ny),
                    Point::new(a.x - nx, a.y - ny),
                ]);
            }
            // TODO(draw): true miter/round/bevel joins and butt/round/square caps
            // (fz_flatten_stroke_path). Bridge each interior vertex with a small
            // square so segments visually connect.
            for v in line.iter().skip(1).take(line.len().saturating_sub(2)) {
                polys.push(vec![
                    Point::new(v.x - hw, v.y - hw),
                    Point::new(v.x + hw, v.y - hw),
                    Point::new(v.x + hw, v.y + hw),
                    Point::new(v.x - hw, v.y + hw),
                ]);
            }
        }
        polys
    }

    /// Like [`flatten`](Path::flatten) but keeps sub-paths **open** (does not add
    /// the closing segment unless an explicit `close` command was given). Used by
    /// the stroker, which walks each polyline segment.
    fn flatten_open(&self, ctm: Matrix) -> Vec<Vec<Point>> {
        let mut out: Vec<Vec<Point>> = Vec::new();
        let mut cur: Vec<Point> = Vec::new();
        let mut here = Point::new(0.0, 0.0);
        let mut start = here;

        for cmd in &self.cmds {
            match *cmd {
                Cmd::MoveTo(p) => {
                    if cur.len() >= 2 {
                        out.push(std::mem::take(&mut cur));
                    } else {
                        cur.clear();
                    }
                    here = p.transform(ctm);
                    start = here;
                    cur.push(here);
                }
                Cmd::LineTo(p) => {
                    here = p.transform(ctm);
                    cur.push(here);
                }
                Cmd::CurveTo(c1, c2, to) => {
                    let d1 = c1.transform(ctm);
                    let d2 = c2.transform(ctm);
                    let d3 = to.transform(ctm);
                    bezier(&mut cur, here, d1, d2, d3, 0);
                    here = d3;
                }
                Cmd::Close => {
                    here = start;
                    cur.push(start);
                }
            }
        }
        if cur.len() >= 2 {
            out.push(cur);
        }
        out
    }
}

// MuPDF: bezier (draw-path.c:80) -- midpoint (de Casteljau) subdivision to a
// flatness tolerance, emitting a `line` at the leaves. `pts` already holds the
// segment start; this pushes only the subdivided interior + endpoint.
fn bezier(pts: &mut Vec<Point>, a: Point, b: Point, c: Point, d: Point, depth: u32) {
    // Termination: control polygon flat enough, or recursion capped.
    let mut dmax = (a.x - b.x).abs();
    dmax = dmax.max((a.y - b.y).abs());
    dmax = dmax.max((d.x - c.x).abs());
    dmax = dmax.max((d.y - c.y).abs());
    if dmax < FLATNESS || depth >= MAX_DEPTH {
        pts.push(d);
        return;
    }

    // Subdivide (MuPDF's integer-avoiding sum-then-scale form).
    let ab = Point::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
    let bc = Point::new((b.x + c.x) * 0.5, (b.y + c.y) * 0.5);
    let cd = Point::new((c.x + d.x) * 0.5, (c.y + d.y) * 0.5);
    let abc = Point::new((ab.x + bc.x) * 0.5, (ab.y + bc.y) * 0.5);
    let bcd = Point::new((bc.x + cd.x) * 0.5, (bc.y + cd.y) * 0.5);
    let abcd = Point::new((abc.x + bcd.x) * 0.5, (abc.y + bcd.y) * 0.5);

    bezier(pts, a, ab, abc, abcd, depth + 1);
    bezier(pts, abcd, bcd, cd, d, depth + 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_flattens_to_four_points() {
        let mut p = Path::new();
        p.rect(0.0, 0.0, 10.0, 10.0);
        let polys = p.flatten(Matrix::IDENTITY);
        assert_eq!(polys.len(), 1);
        // move + 3 line + close-return -> 4 distinct corners (+ the returned
        // start point which coincides with the first corner).
        assert!(polys[0].len() >= 4);
        assert_eq!(polys[0][0], Point::new(0.0, 0.0));
    }

    #[test]
    fn ctm_transforms_flattened_points() {
        let mut p = Path::new();
        p.move_to(1.0, 1.0).line_to(2.0, 1.0);
        let polys = p.flatten(Matrix::scale(10.0, 10.0));
        assert_eq!(polys[0][0], Point::new(10.0, 10.0));
        assert_eq!(polys[0][1], Point::new(20.0, 10.0));
    }

    #[test]
    fn bezier_subdivides_to_many_segments() {
        let mut p = Path::new();
        // A wide curve should flatten to well more than 2 points.
        p.move_to(0.0, 0.0).curve_to(0.0, 100.0, 100.0, 100.0, 100.0, 0.0);
        let polys = p.flatten(Matrix::IDENTITY);
        assert!(polys[0].len() > 8, "curve produced {} pts", polys[0].len());
        // Endpoint preserved.
        let last = *polys[0].last().unwrap();
        assert!((last.x - 100.0).abs() < 0.5 && last.y.abs() < 0.5);
    }

    #[test]
    fn stroke_expands_segment_to_a_quad() {
        let mut p = Path::new();
        p.move_to(0.0, 5.0).line_to(10.0, 5.0);
        let polys = p.stroke_to_polygons(Matrix::IDENTITY, 2.0);
        // One horizontal segment -> at least one quad polygon.
        assert!(!polys.is_empty());
        // The quad should straddle y=5 by the half width (1.0).
        let ys: Vec<f32> = polys[0].iter().map(|pt| pt.y).collect();
        assert!(ys.iter().cloned().fold(f32::INFINITY, f32::min) <= 4.0 + 1e-3);
        assert!(ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max) >= 6.0 - 1e-3);
    }
}
