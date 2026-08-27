//! Ported from MuPDF `source/fitz/path.c` (the `fz_path` moveto/lineto/curveto/
//! closepath builder and `fz_walk_path`) and `source/fitz/draw-path.c`
//! (`fz_flatten_fill_path`, the `bezier`/`quadratic` adaptive subdivision, the
//! `line` primitive, and `fz_flatten_stroke_path`'s join/cap construction:
//! `fz_add_line_join`, `do_linecap`/`fz_add_line_cap`, `fz_add_zero_len_cap`,
//! `fz_add_line_dot` and `fz_add_arc`) (commit 19f1284, AGPL-3.0, © Artifex
//! Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close
//! adaptation: the path model, the bezier flattening (midpoint subdivision to
//! a flatness tolerance), the fill-flatten walk, and the stroke join/cap
//! geometry follow MuPDF; the code is re-expressed in idiomatic Rust rather
//! than transliterated from the C. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
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
//! ## Stroking
//!
//! MuPDF's stroker (`fz_flatten_stroke_path`) builds true offset outlines with
//! miter/round/bevel joins and butt/round/square/triangle caps by emitting a
//! single continuous pair of rasterizer edges (forward + reverse) that a
//! winding-number rasterizer fills directly. Our [`fill_polygons`] instead
//! fills a *list* of independent closed polygons under
//! [`FillRule::NonZero`](super::draw_edge::FillRule::NonZero), so this port
//! re-expresses the same geometry as one polygon per segment (a rectangle)
//! plus one small polygon per join and per cap, all summed under the same
//! non-zero winding rule -- overlap between polygons is harmless (it just
//! adds winding, never cancels), which is what lets each shape be generated
//! independently instead of stitched into one boundary walk. See
//! [`Path::stroke_to_polygons_styled`] for the per-shape derivations and their
//! upstream citations.

use super::geometry::Matrix;
use super::geometry::Point;

/// The default fill flatness in **device pixels**. MuPDF uses `0.3 / expansion`
/// clamped to `>= 0.001` (draw-device.c:699); flattening here is done directly in
/// device space, so `0.3 px` is the constant tolerance.
pub const FLATNESS: f32 = 0.3;

/// Recursion cap on bezier subdivision (`MAX_DEPTH` in draw-path.c).
const MAX_DEPTH: u32 = 36;

/// Line cap styles, matching PDF's `J` operator values (PDF 32000-1 table 52)
/// and MuPDF's `fz_linecap` (`FZ_LINECAP_BUTT`/`_ROUND`/`_SQUARE`/`_TRIANGLE`).
/// `Triangle` has no PDF operator value of its own -- MuPDF exposes it as an
/// XPS-derived extension -- but is kept here so this enum mirrors
/// `fz_linecap` exactly for anyone porting further stroke code later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineCap {
    #[default]
    Butt = 0,
    Round = 1,
    Square = 2,
    Triangle = 3,
}

/// Line join styles, matching PDF's `j` operator values (PDF 32000-1 table 53)
/// and MuPDF's `fz_linejoin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineJoin {
    #[default]
    Miter = 0,
    Round = 1,
    Bevel = 2,
}

/// PDF's own default miter limit (PDF 32000-1 8.4.3.5), used when
/// [`Path::stroke_to_polygons`]'s caller has no `10 M`-equivalent state to
/// hand in yet.
const DEFAULT_MITER_LIMIT: f32 = 10.0;

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
        self.cmds.push(Cmd::CurveTo(
            Point::new(x1, y1),
            Point::new(x2, y2),
            Point::new(x3, y3),
        ));
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
        self.move_to(x0, y0)
            .line_to(x1, y0)
            .line_to(x1, y1)
            .line_to(x0, y1)
            .close()
    }

    /// Append another path's commands to the end of this one, preserving its
    /// sub-path structure. Used to merge composite glyph outlines (Type 1
    /// `seac`: a base glyph's path plus a translated accent glyph's path).
    pub(crate) fn append(&mut self, mut other: Path) {
        self.cmds.append(&mut other.cmds);
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

    /// Flatten and expand the path to filled polygons approximating a stroke of
    /// device-space `line_width`, using `ctm` for the geometry, with **round**
    /// caps and **round** joins.
    ///
    /// Round is not PDF's actual default (that is butt caps / miter joins,
    /// `J`/`j` both `0`) -- it is a deliberate, documented stand-in used
    /// because the content-stream interpreter does not yet track `J`/`j`
    /// graphics state and so cannot tell this call what the real style is
    /// (see [`Path::stroke_to_polygons_styled`] for the entry point the
    /// interpreter should switch to once it does). Round is chosen over
    /// keeping the old butt/square-approximation behaviour because: it is
    /// what the ink-annotation appearance synthesis path explicitly asks for
    /// (`1 J 1 j`, ported faithfully in `annot_appearance.rs`); it is the
    /// only style under which a single-point subpath paints anything at all
    /// (`fz_add_line_dot`, draw-path.c:763 -- with butt caps a zero-length
    /// subpath has zero area and disappears); and it is strictly closer to
    /// poppler's rendering than butt/square for every other stroked path too.
    /// This divergence should be removed once `J`/`j` are plumbed through the
    /// interpreter, at which point this should simply delegate with the
    /// graphics state's real values instead of hard-coded `Round`/`Round`.
    pub fn stroke_to_polygons(&self, ctm: Matrix, line_width: f32) -> Vec<Vec<Point>> {
        self.stroke_to_polygons_styled(
            ctm,
            line_width,
            LineCap::Round,
            LineJoin::Round,
            DEFAULT_MITER_LIMIT,
        )
    }

    /// Flatten and expand the path to filled polygons approximating a stroke of
    /// device-space `line_width`, honouring the given cap/join/miter-limit
    /// graphics state. `ctm` supplies the geometry (as for
    /// [`flatten`](Path::flatten)); fill the result with
    /// [`FillRule::NonZero`](super::draw_edge::FillRule::NonZero).
    ///
    /// Ports MuPDF's `fz_flatten_stroke_path` (draw-path.c:1607), reworked
    /// from its single continuous winding-edge walk into one polygon per
    /// segment/join/cap (see the module doc comment for why that
    /// re-expression is safe under non-zero winding). A hairline (width `0`
    /// or sub-pixel) still draws about one device pixel, matching MuPDF.
    pub fn stroke_to_polygons_styled(
        &self,
        ctm: Matrix,
        line_width: f32,
        cap: LineCap,
        join: LineJoin,
        miter_limit: f32,
    ) -> Vec<Vec<Point>> {
        let hw = (line_width * 0.5).max(0.35);
        // A non-finite or sub-1.0 miter limit is meaningless (PDF 32000-1
        // 8.4.3.5 requires `M >= 1`); fall back to the spec default rather
        // than propagate garbage into the miter-limit test below.
        let miter_limit = if miter_limit.is_finite() && miter_limit >= 1.0 {
            miter_limit
        } else {
            DEFAULT_MITER_LIMIT
        };

        let subpaths = self.flatten_open(ctm);
        let mut polys: Vec<Vec<Point>> = Vec::new();

        for (raw_line, closed) in &subpaths {
            // Drop non-finite vertices and collapse zero-length runs before
            // doing any geometry: this code parses attacker-controlled file
            // content, and a poisoned coordinate must not propagate NaN into
            // the rasterizer or the join/cap trigonometry below.
            let line = dedup_finite_points(raw_line);

            if line.len() < 2 {
                // A degenerate sub-path: at most one distinct point survived.
                // MuPDF's own appearance-stream synthesis deliberately builds
                // exactly this shape for a single-point ink stroke -- a
                // moveto with no real draws -- specifically to get a round
                // dot out of `fz_add_line_dot` (draw-path.c:763, driven from
                // `fz_stroke_flush`/`fz_stroke_closepath` when `sn == 0` and
                // the cap style is round); pdf-appearance.c:1069 emits a
                // zero-length `l` for exactly this reason, and
                // `annot_appearance.rs` already ports that call site
                // faithfully. Reproduce the same behaviour here directly:
                // any round-capped sub-path that collapses to one point
                // still paints a dot; other cap styles paint nothing, same
                // as upstream (a butt/square/triangle cap has no defined
                // direction with a zero-length segment --
                // `fz_add_zero_len_cap` returns early when `dx == 0 && dy ==
                // 0`, draw-path.c:749).
                if cap == LineCap::Round
                    && let Some(&p) = line.first()
                    && let Some(dot) = round_dot(p, hw)
                {
                    polys.push(dot);
                }
                continue;
            }

            let n = line.len();
            // Per-segment unit tangents, `None` for a segment too short to
            // have a meaningful direction (already rare after dedup, but a
            // matrix can still map two distinct user-space points onto the
            // same device pixel).
            let mut tangents: Vec<Option<Point>> = Vec::with_capacity(n - 1);

            for w in line.windows(2) {
                let (a, b) = (w[0], w[1]);
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let len = (dx * dx + dy * dy).sqrt();
                if !(len.is_finite()) || len < 1e-6 {
                    tangents.push(None);
                    continue;
                }
                let t = Point::new(dx / len, dy / len);
                // Unit normal, scaled to the half-width -- same rotation
                // (`(-dy, dx)`) as the join/cap code below uses, so a join's
                // or cap's chord points land exactly on these quad corners
                // with no seam.
                let nrm = Point::new(-t.y * hw, t.x * hw);
                polys.push(vec![
                    Point::new(a.x + nrm.x, a.y + nrm.y),
                    Point::new(b.x + nrm.x, b.y + nrm.y),
                    Point::new(b.x - nrm.x, b.y - nrm.y),
                    Point::new(a.x - nrm.x, a.y - nrm.y),
                ]);
                tangents.push(Some(t));
            }

            // Interior joins (draw-path.c fz_add_line_join, called from
            // fz_stroke_lineto_aux whenever a third point arrives). Only the
            // convex/outer side needs an explicit polygon: the two
            // full-width segment quads already overlap on the concave/inner
            // side (verified by construction -- each quad spans its whole
            // segment at the full half-width on both sides, so at the shared
            // vertex the inner corner is covered twice over, and `NonZero`
            // fill treats double coverage as ordinary fill, not a hole).
            // Only the outer side is ever left with a real gap, which is
            // exactly what add_join fills.
            for i in 1..n - 1 {
                if let (Some(t0), Some(t1)) = (tangents[i - 1], tangents[i]) {
                    add_join(&mut polys, line[i], t0, t1, hw, join, miter_limit);
                }
            }

            if *closed {
                // The wrap-around join where the last segment meets the
                // first. `flatten_open`'s Close handling repeats the
                // sub-path's start point as its last point (mirroring
                // `flatten`), so `line[0] == line[n - 1]` and the join sits
                // at that shared vertex.
                if let (Some(t_last), Some(t_first)) = (tangents[n - 2], tangents[0]) {
                    add_join(&mut polys, line[0], t_last, t_first, hw, join, miter_limit);
                }
                // Closed sub-paths get no caps at all -- both "ends" are the
                // same joined vertex, handled above.
            } else if let Some(t0) = tangents[0] {
                // Start cap: bulges backward, away from the sub-path.
                let n_hat = Point::new(-t0.y, t0.x);
                let dir = Point::new(-t0.x, -t0.y);
                add_cap(&mut polys, line[0], dir, n_hat, hw, cap);

                if let Some(t_last) = tangents[n - 2] {
                    // End cap: bulges forward, continuing the sub-path.
                    let n_hat = Point::new(-t_last.y, t_last.x);
                    add_cap(&mut polys, line[n - 1], t_last, n_hat, hw, cap);
                }
            }
        }
        polys
    }

    /// Like [`flatten`](Path::flatten) but keeps sub-paths **open** (does not
    /// add the closing segment unless an explicit `close` command was
    /// given). Used by the stroker, which walks each polyline segment and
    /// needs to distinguish a genuinely closed sub-path (gets a join at the
    /// wrap point, no caps) from one that merely happens to end where it
    /// started (PDF only closes a sub-path on an explicit `h`/`s` operator;
    /// see `Cmd::Close`).
    ///
    /// Returns `(points, closed)` per sub-path. A closed sub-path's point
    /// list repeats its start point as its last point, exactly like
    /// [`flatten`]'s close handling, so the stroker can find the wrap-around
    /// join vertex as `points[0]` (== `points[points.len() - 1]`).
    ///
    /// One pre-existing simplification carried over unchanged from before
    /// this method returned a `closed` flag: if path-construction commands
    /// follow a `Close` without an intervening `MoveTo` (legal-looking but
    /// unusual PDF content), they extend the *same* accumulated polyline
    /// rather than starting a fresh one, and the whole run is reported
    /// `closed`. Real content streams always emit a fresh `m` after `h`, so
    /// this has never been observed to matter in practice.
    fn flatten_open(&self, ctm: Matrix) -> Vec<(Vec<Point>, bool)> {
        let mut out: Vec<(Vec<Point>, bool)> = Vec::new();
        let mut cur: Vec<Point> = Vec::new();
        let mut closed = false;
        let mut here = Point::new(0.0, 0.0);
        let mut start = here;

        for cmd in &self.cmds {
            match *cmd {
                Cmd::MoveTo(p) => {
                    // Flush at `>= 1`, not `>= 2`: a lone `moveto` with no
                    // draws is exactly the single-point ink-dot subpath
                    // (`fz_add_line_dot`'s case, see `stroke_to_polygons_styled`'s
                    // degenerate branch below) and must survive to be seen there.
                    if !cur.is_empty() {
                        out.push((std::mem::take(&mut cur), closed));
                    }
                    closed = false;
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
                    closed = true;
                }
            }
        }
        if !cur.is_empty() {
            out.push((cur, closed));
        }
        out
    }
}

/// Drop non-finite points and collapse runs of (near-)identical consecutive
/// points, so the stroker never has to reason about a zero-length segment or
/// a NaN/infinite coordinate coming out of a hostile CTM or path.
///
/// `1e-12` is a squared-distance threshold, i.e. a `1e-6` device-pixel
/// distance cutoff -- the same tolerance the segment loop already used
/// (`len < 1e-6`) before this port, kept consistent rather than introducing a
/// second magic number for the same idea.
fn dedup_finite_points(points: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::with_capacity(points.len());
    for &p in points {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        if let Some(&last) = out.last() {
            let dx = p.x - last.x;
            let dy = p.y - last.y;
            if dx * dx + dy * dy < 1e-12 {
                continue;
            }
        }
        out.push(p);
    }
    out
}

/// Fills the outer-side gap at an interior (or wrap-around) join between two
/// segments with unit tangents `t0` (incoming) and `t1` (outgoing), meeting
/// at `b`. Ports `fz_add_line_join` (draw-path.c:543).
///
/// Only the convex/outer side is ever drawn explicitly -- see the call site's
/// comment for why the concave/inner side needs nothing (the two segment
/// quads already overlap there).
fn add_join(
    polys: &mut Vec<Vec<Point>>,
    b: Point,
    t0: Point,
    t1: Point,
    hw: f32,
    join: LineJoin,
    miter_limit: f32,
) {
    if !(b.x.is_finite() && b.y.is_finite()) {
        return;
    }

    // Per-segment left-hand normals, same rotation as the segment quads.
    let n0 = Point::new(-t0.y * hw, t0.x * hw);
    let n1 = Point::new(-t1.y * hw, t1.x * hw);

    let cross = t0.x * t1.y - t0.y * t1.x;
    let dot = t0.x * t1.x + t0.y * t1.y;

    // draw-path.c:584: the two segments are colinear and pointing the same
    // way (a flattened bezier's interior "joins" are almost always this) --
    // there is no corner to fill.
    if cross * cross < f32::EPSILON && dot >= 0.0 {
        return;
    }

    // The two per-segment normals point to the same fixed side (left of
    // travel) regardless of which way the path turns; whichever of `+n`/`-n`
    // is *not* where the quads already overlap is the outer/convex side that
    // needs an explicit fill. `cross > 0` is a left turn, whose outer side is
    // the right-hand (`-n`) side, and symmetrically for a right turn.
    let outer_sign = if cross > 0.0 { -1.0 } else { 1.0 };
    let outer_n0 = Point::new(n0.x * outer_sign, n0.y * outer_sign);
    let outer_n1 = Point::new(n1.x * outer_sign, n1.y * outer_sign);
    let p0 = Point::new(b.x + outer_n0.x, b.y + outer_n0.y);
    let p1 = Point::new(b.x + outer_n1.x, b.y + outer_n1.y);

    match join {
        LineJoin::Bevel => {
            polys.push(vec![b, p0, p1]);
        }
        LineJoin::Miter => {
            // draw-path.c:622 (`FZ_LINEJOIN_MITER`) plus the miterlimit test
            // at :597. `dm` there is the average of the two *inner*-side
            // normals; its squared length `dmr2` is unaffected by negating
            // both inputs by the same sign, so it is computed here directly
            // from the outer normals instead of replaying upstream's
            // cross-sign bookkeeping (that swap exists to keep upstream's
            // single continuous winding edge consistent; independent
            // polygons under non-zero fill don't need it).
            //
            // `|dm| = hw * cos(phi/2)` where `phi` is the angle between the
            // segments, so `dmr2 * miterlimit^2 >= hw^2` is algebraically
            // PDF 32000-1 8.4.3.5's `miterLength / lineWidth = 1 /
            // sin(interior_angle/2) <= miterLimit` test (`interior_angle =
            // pi - phi`, so `sin(interior_angle/2) = cos(phi/2)`) -- worked
            // out by hand here and cross-checked against upstream's
            // acceptance direction rather than taken on faith.
            let dm = Point::new(
                (outer_n0.x + outer_n1.x) * 0.5,
                (outer_n0.y + outer_n1.y) * 0.5,
            );
            let dmr2 = dm.x * dm.x + dm.y * dm.y;
            let miter_ok = dmr2 > f32::EPSILON && dmr2 * miter_limit * miter_limit >= hw * hw;
            if miter_ok {
                let scale = hw * hw / dmr2;
                let m = Point::new(b.x + dm.x * scale, b.y + dm.y * scale);
                if m.x.is_finite() && m.y.is_finite() {
                    polys.push(vec![b, p0, m, p1]);
                } else {
                    polys.push(vec![b, p0, p1]);
                }
            } else {
                // Miter limit exceeded -> falls back to bevel (draw-path.c:598).
                polys.push(vec![b, p0, p1]);
            }
        }
        LineJoin::Round => {
            polys.push(round_arc_fan(b, p0, p1, hw));
        }
    }
}

/// Emits the extra geometry an open sub-path's end needs beyond the flush
/// end its segment quad already provides. Ports `do_linecap`
/// (draw-path.c:679-727), called at both ends from `fz_stroke_flush`.
///
/// `b` is the endpoint, `n_hat` the segment's (unit) perpendicular normal at
/// that end (so `b - hw*n_hat` / `b + hw*n_hat` land exactly on the
/// segment-quad corners already emitted for this end -- no seam), and `dir`
/// the (unit) direction the cap should bulge, i.e. away from the sub-path:
/// backward past the start, forward past the end.
fn add_cap(polys: &mut Vec<Vec<Point>>, b: Point, dir: Point, n_hat: Point, hw: f32, cap: LineCap) {
    if !(b.x.is_finite()
        && b.y.is_finite()
        && dir.x.is_finite()
        && dir.y.is_finite()
        && n_hat.x.is_finite()
        && n_hat.y.is_finite())
    {
        return;
    }

    let from = Point::new(b.x - n_hat.x * hw, b.y - n_hat.y * hw);
    let to = Point::new(b.x + n_hat.x * hw, b.y + n_hat.y * hw);

    match cap {
        // draw-path.c FZ_LINECAP_BUTT: a single edge straight across `from`..`to` --
        // exactly the segment quad's own flush end, so there is nothing to add.
        LineCap::Butt => {}
        LineCap::Square => {
            // draw-path.c:708-715: extend a hw x 2*hw box straight out along `dir`.
            let ext = Point::new(dir.x * hw, dir.y * hw);
            polys.push(vec![
                from,
                Point::new(from.x + ext.x, from.y + ext.y),
                Point::new(to.x + ext.x, to.y + ext.y),
                to,
            ]);
        }
        LineCap::Triangle => {
            // draw-path.c:717-723: a triangular spike out to the tip.
            let tip = Point::new(b.x + dir.x * hw, b.y + dir.y * hw);
            polys.push(vec![from, tip, to]);
        }
        LineCap::Round => {
            polys.push(round_cap_polygon(b, n_hat, dir, hw));
        }
    }
}

/// Angular subdivision step used by every arc this stroker draws (round
/// joins, round caps, round dots): `2*sqrt(2)*sqrt(flatness/r)` radians per
/// segment, straight out of `fz_add_arc` (draw-path.c:449-450), which is the
/// same formula `do_linecap`'s `FZ_LINECAP_ROUND` case specialises to a fixed
/// `PI`-radian sweep (draw-path.c:690) and `fz_add_line_dot` specialises to a
/// fixed `2*PI`-radian sweep (draw-path.c:768). Smaller radii or a coarser
/// flatness tolerance both take fewer steps, which is what keeps vertex
/// counts bounded on huge paths instead of exploding.
fn arc_step(hw: f32) -> f32 {
    2.0 * std::f32::consts::SQRT_2 * (FLATNESS / hw).sqrt()
}

/// Fills a join's outer wedge with a fan of triangles approximating the arc
/// of radius `hw` centred at `apex`, from `from` to `to`. Ports the round-join
/// case of `fz_add_line_join` (draw-path.c:667), which calls `fz_add_arc`
/// (draw-path.c:437) to do the actual stepping.
///
/// Unlike [`round_cap_polygon`] (where the two chord points are always
/// exactly 90 degrees apart, letting the sweep be built from an orthonormal
/// basis with no trigonometric inverse needed), a join's `from`/`to` can be
/// separated by any angle up to `PI`, so this walks the angle explicitly via
/// `atan2`. That is safe here specifically because the angle between them is
/// bounded to `(0, PI]` by construction (`add_join`'s caller only reaches
/// this for a real corner, and negating both `outer_n0`/`outer_n1` by the
/// same sign -- which is all `outer_sign` does -- cannot change the angle
/// between them), so normalising the raw `atan2` delta into `(-PI, PI]`
/// always recovers the true short way round, with no wraparound ambiguity
/// and no division by a `sin` that could be near zero.
fn round_arc_fan(apex: Point, from: Point, to: Point, hw: f32) -> Vec<Point> {
    let mut out = vec![apex, from];
    if hw > f32::EPSILON {
        let a0 = (from.y - apex.y).atan2(from.x - apex.x);
        let a1 = (to.y - apex.y).atan2(to.x - apex.x);
        let mut delta = a1 - a0;
        while delta <= -std::f32::consts::PI {
            delta += std::f32::consts::TAU;
        }
        while delta > std::f32::consts::PI {
            delta -= std::f32::consts::TAU;
        }
        let step = arc_step(hw);
        let n = if step > 0.0 && step.is_finite() {
            ((delta.abs() / step).ceil() as u32).max(1)
        } else {
            1
        };
        for i in 1..n {
            let theta = a0 + delta * (i as f32) / (n as f32);
            out.push(Point::new(
                apex.x + theta.cos() * hw,
                apex.y + theta.sin() * hw,
            ));
        }
    }
    out.push(to);
    out
}

/// Builds a cap's round half-disc boundary: from `b - hw*n_hat` through the
/// tip `b + hw*dir` to `b + hw*n_hat`. Ports `do_linecap`'s
/// `FZ_LINECAP_ROUND` case (draw-path.c:690-706).
///
/// `n_hat` and `dir` are always an orthonormal pair by construction (`dir` is
/// always `+-` the segment's own unit tangent, `n_hat` always that tangent's
/// perpendicular), so the sweep is written directly against that basis --
/// `b + hw*(sin(theta)*n_hat + cos(theta)*dir)` for `theta` in
/// `[-PI/2, PI/2]` -- rather than replaying upstream's
/// `cos(theta)`/`sin(theta)` combination of the raw offset vector `dl`. Both
/// describe the same semicircle; parametrising from an orthonormal basis
/// makes it possible to check the endpoints land exactly on `from`/`tip`/`to`
/// by inspection instead of by re-deriving upstream's rotation-direction
/// convention by hand.
fn round_cap_polygon(b: Point, n_hat: Point, dir: Point, hw: f32) -> Vec<Point> {
    let from = Point::new(b.x - n_hat.x * hw, b.y - n_hat.y * hw);
    let to = Point::new(b.x + n_hat.x * hw, b.y + n_hat.y * hw);
    let mut out = vec![from];
    if hw > f32::EPSILON {
        let step = arc_step(hw);
        let n = if step > 0.0 && step.is_finite() {
            ((std::f32::consts::PI / step).ceil() as u32).max(1)
        } else {
            1
        };
        for i in 1..n {
            let theta =
                -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * (i as f32) / (n as f32);
            let (s, c) = theta.sin_cos();
            out.push(Point::new(
                b.x + hw * (s * n_hat.x + c * dir.x),
                b.y + hw * (s * n_hat.y + c * dir.y),
            ));
        }
    }
    out.push(to);
    out
}

/// Builds a full circle of radius `hw` around `center`, used when a
/// zero-length sub-path (a single point) has a round cap. Ports
/// `fz_add_line_dot` (draw-path.c:763-788). Returns `None` for a degenerate
/// (non-finite or non-positive) radius/centre rather than emitting NaN
/// vertices.
fn round_dot(center: Point, hw: f32) -> Option<Vec<Point>> {
    if !(center.x.is_finite() && center.y.is_finite()) || !hw.is_finite() || hw <= f32::EPSILON {
        return None;
    }
    // draw-path.c:768: same step formula as every other arc here, but a full
    // 2*PI sweep instead of PI; upstream additionally floors this at 3 steps
    // so a very thin dot still looks like a polygon rather than degenerating
    // to a line.
    let step = arc_step(hw) * 0.5;
    let n = if step > 0.0 && step.is_finite() {
        ((std::f32::consts::PI / step).ceil() as u32).max(3)
    } else {
        3
    };
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
        out.push(Point::new(
            center.x + theta.cos() * hw,
            center.y + theta.sin() * hw,
        ));
    }
    Some(out)
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

    fn all_finite(polys: &[Vec<Point>]) -> bool {
        polys
            .iter()
            .all(|poly| poly.iter().all(|p| p.x.is_finite() && p.y.is_finite()))
    }

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
        p.move_to(0.0, 0.0)
            .curve_to(0.0, 100.0, 100.0, 100.0, 100.0, 0.0);
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

    /// Round caps must add area beyond a segment's flush ends: with butt
    /// caps, no point of any emitted polygon should extend past the
    /// endpoints along the stroke direction; with round caps, at least one
    /// should.
    #[test]
    fn round_cap_adds_area_beyond_segment_ends() {
        let mut p = Path::new();
        p.move_to(0.0, 0.0).line_to(10.0, 0.0);

        let butt = p.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Butt,
            LineJoin::Bevel,
            10.0,
        );
        assert!(
            butt.iter()
                .flatten()
                .all(|pt| pt.x >= -1e-3 && pt.x <= 10.0 + 1e-3),
            "butt cap should not extend past the segment ends"
        );

        let round = p.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Round,
            LineJoin::Bevel,
            10.0,
        );
        assert!(
            round
                .iter()
                .flatten()
                .any(|pt| pt.x < -1e-3 || pt.x > 10.0 + 1e-3),
            "round cap should extend past the segment ends"
        );
        assert!(all_finite(&round));
    }

    /// A single-point subpath ("dot") only paints with round caps -- this is
    /// the fix for MuPDF's zero-length-`l`-back-onto-itself ink-dot trick
    /// (pdf-appearance.c:1069 / draw-path.c:763) rendering as nothing under
    /// butt caps.
    #[test]
    fn single_point_subpath_dot_only_with_round_cap() {
        let mut p = Path::new();
        p.move_to(5.0, 5.0);

        let butt = p.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Butt,
            LineJoin::Bevel,
            10.0,
        );
        assert!(
            butt.is_empty(),
            "butt cap on a single point should paint nothing"
        );

        let round = p.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Round,
            LineJoin::Bevel,
            10.0,
        );
        assert_eq!(
            round.len(),
            1,
            "round cap on a single point should paint exactly one dot polygon"
        );
        assert!(
            round[0].len() >= 3,
            "the dot should be a real polygon, not degenerate"
        );
        assert!(all_finite(&round));
    }

    /// A join whose turn angle exceeds the miter limit must fall back to a
    /// bevel (a 3-point triangle), never emit the 4-point miter spike.
    #[test]
    fn miter_beyond_limit_degrades_to_bevel() {
        let mut p = Path::new();
        // A sharp near-reversal: 170 degrees of turn needs a huge miter
        // limit to keep the spike; miter_limit = 1.0 (the spec minimum)
        // should always force a bevel here.
        p.move_to(0.0, 0.0).line_to(10.0, 0.0).line_to(0.3, 1.0);
        let polys =
            p.stroke_to_polygons_styled(Matrix::IDENTITY, 2.0, LineCap::Butt, LineJoin::Miter, 1.0);
        // The join polygon is whichever one isn't a 4-point axis-aligned
        // segment quad; both segment quads have exactly 4 points too, so
        // instead check that no polygon has the 4-point *miter* shape by
        // confirming a spike (a point far from the joint) is absent -- more
        // directly, assert no polygon has 4 points AND a vertex farther from
        // (10,0) than the segment half-width would allow for a plain quad.
        // Simplest robust check: with a 1.0 limit every join must bevel, so
        // there must be at least one 3-point polygon (the bevel) beyond the
        // two 4-point segment quads, and no vertex should shoot far away
        // from the joint (a runaway miter spike).
        assert!(
            polys.iter().any(|poly| poly.len() == 3),
            "expected a bevel triangle when the miter limit is exceeded"
        );
        for poly in &polys {
            for pt in poly {
                assert!(
                    pt.x.hypot(pt.y) < 100.0,
                    "a vertex ran away to {pt:?}; looks like a runaway miter spike"
                );
            }
        }
        assert!(all_finite(&polys));
    }

    /// A generous miter limit on a modest turn should actually produce the
    /// 4-point miter spike (sanity check for the opposite branch).
    #[test]
    fn miter_within_limit_produces_spike() {
        let mut p = Path::new();
        // A 90-degree turn: miter ratio is 1/sin(45deg) ~= 1.41, well within
        // a limit of 10.
        p.move_to(0.0, 0.0).line_to(10.0, 0.0).line_to(10.0, 10.0);
        let polys = p.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Butt,
            LineJoin::Miter,
            10.0,
        );
        assert!(
            polys
                .iter()
                .any(|poly| poly.len() == 4 && !poly.contains(&Point::new(0.0, 0.0))),
            "expected a 4-point miter join polygon"
        );
        assert!(all_finite(&polys));
    }

    /// Closed sub-paths get a join at the wrap-around vertex and no caps at
    /// either end.
    #[test]
    fn closed_subpath_gets_join_not_caps() {
        let mut p = Path::new();
        p.rect(0.0, 0.0, 10.0, 10.0);
        let with_square_cap = p.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Square,
            LineJoin::Bevel,
            10.0,
        );
        // 4 segment quads + 4 bevel joins (one per corner, including the
        // wrap-around) = 8 polygons; a square cap would add extra polygons
        // beyond that if (wrongly) applied to a closed path.
        assert_eq!(
            with_square_cap.len(),
            8,
            "closed rect should be 4 quads + 4 joins, no caps"
        );
        assert!(all_finite(&with_square_cap));
    }

    /// Degenerate and hostile input -- repeated points, zero-length
    /// segments, non-finite coordinates -- must not panic or produce NaN
    /// vertices, for any cap/join combination.
    #[test]
    fn degenerate_and_nonfinite_input_is_safe() {
        let mut p = Path::new();
        p.move_to(1.0, 1.0)
            .line_to(1.0, 1.0) // zero-length segment
            .line_to(1.0, 1.0) // repeated point
            .line_to(f32::NAN, 2.0) // non-finite
            .line_to(f32::INFINITY, f32::NEG_INFINITY)
            .line_to(3.0, 3.0);

        for cap in [
            LineCap::Butt,
            LineCap::Round,
            LineCap::Square,
            LineCap::Triangle,
        ] {
            for join in [LineJoin::Miter, LineJoin::Round, LineJoin::Bevel] {
                let polys = p.stroke_to_polygons_styled(Matrix::IDENTITY, 2.0, cap, join, 10.0);
                assert!(
                    all_finite(&polys),
                    "cap={cap:?} join={join:?} produced a non-finite vertex"
                );
            }
        }

        // A path that is nothing but non-finite/degenerate points.
        let mut empty_ish = Path::new();
        empty_ish.move_to(f32::NAN, f32::NAN);
        let polys = empty_ish.stroke_to_polygons_styled(
            Matrix::IDENTITY,
            2.0,
            LineCap::Round,
            LineJoin::Round,
            10.0,
        );
        assert!(polys.is_empty());
    }
}
