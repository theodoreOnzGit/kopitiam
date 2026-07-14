//! Vector path extraction from a PDF content stream.
//!
//! This is the module the rest of the crate rests on. If it works, plot
//! digitisation is a mostly-exact geometric exercise; if it does not, the only
//! fallback is rasterising the page and tracing curves out of pixels, which is
//! lossy, needs anti-aliasing heuristics, and cannot recover a line's dash
//! pattern or a curve hidden under another curve.
//!
//! It works. A PDF plot is, in the overwhelming majority of cases, *vector
//! graphics*: the data curve is a path (`m`, `l`, `c`, `re`), the axes and ticks
//! are paths, and the tick labels are text. The numbers that generated the
//! figure are therefore still in the file, transformed by an affine map we can
//! recover, and no image processing is involved at any point.
//!
//! # Why we walk the stream ourselves
//!
//! `pdf-extract` -- which `kopitiam-pdf` already uses, and which we already
//! depend on transitively -- *does* offer [`pdf_extract::OutputDev::stroke`]
//! and `fill` callbacks carrying path geometry and colour. Using them would be
//! less code. We do not, for three reasons, all of which are fatal rather than
//! stylistic:
//!
//! 1. **No line width, no dash pattern.** The callbacks pass the CTM, the
//!    colour space, the colour, and the path -- and nothing else. `pdf-extract`
//!    tracks `line_width` in its internal graphics state but never surfaces it,
//!    and does not track the dash pattern at all. Colour, width and dash are
//!    precisely the three cues that separate one series from another
//!    ([`crate::style`]), so a digitiser built on those callbacks could not tell
//!    a solid 1pt black curve from a dashed 2pt black one. That is not a corner
//!    case; it is the normal shape of a two-series figure.
//!
//! 2. **Five paint operators are silently dropped.** `pdf-extract`'s content
//!    interpreter matches `"s" | "f*" | "B" | "B*" | "b"` and logs them as
//!    unhandled -- no callback fires, *and the path buffer is not cleared*, so
//!    the discarded path's segments leak into whatever path is constructed next.
//!    Only `S` and `f` reach a callback. `B` (fill-then-stroke) is how a solid
//!    scatter marker is normally painted and `f*` is the even-odd fill used by
//!    several producers, so this would lose entire series outright and corrupt
//!    their neighbours while doing it.
//!
//! 3. `kopitiam-pdf` already set this precedent for exactly this class of gap:
//!    `font_resources.rs` re-walks the same content stream with `lopdf` to
//!    recover font state that `OutputDev` does not expose. This module is the
//!    same manoeuvre applied to path state, and deliberately mirrors its shape.
//!
//! The cost is real and should be stated plainly: the workspace now contains a
//! second, partial PDF content-stream interpreter, which must be maintained.
//! See `docs/ai-decisions/AID-0017-plot-vector-extraction.md`.
//!
//! # Bézier curves are deliberately *not* flattened
//!
//! The obvious move -- subdivide every `c` operator into short line segments --
//! is wrong here, and understanding why is the difference between a digitiser
//! that reports data and one that reports its own interpolation.
//!
//! A line plot is emitted as `m` followed by one `l` per data point. **Those
//! anchor points *are* the data**, exactly, to the precision the producer wrote
//! them at. Flattening a curve would *add* points that were never in anyone's
//! dataset, and a caller has no way to tell an invented point from a measured
//! one. So we keep on-curve anchors only, and never synthesise a coordinate.
//!
//! When a path genuinely does contain `c` segments, that means one of two
//! things -- the producer drew a smoothed spline through the data (anchors are
//! still the data; the control points are cosmetic), or it drew a genuine
//! analytic curve (anchors are just knots, and the data between them is not
//! recoverable as samples). We cannot distinguish these from the file, so the
//! honest thing is to report the anchors *and say that we did*:
//! [`Subpath::has_curves`] drives a warning in [`crate::digitise`]. Silently
//! emitting 16 flattened points per Bézier and calling them data would be the
//! cardinal sin this crate is built to avoid.
//!
//! # What this module does not model
//!
//! Stated bluntly, because a digitiser that hides its blind spots is worse than
//! one that has none:
//!
//! * **Clipping** is tracked only as a bounding box, and only as a hint. A path
//!   clipped to a shape more complex than a rectangle is reported in full.
//! * **Type 3 / glyph-drawn markers.** A producer may draw scatter markers as
//!   characters of a font rather than as paths (matplotlib historically did).
//!   Those arrive as *text*, not paths, and this module will not see them at
//!   all. [`crate::digitise`] warns when it finds marker-like text inside a plot
//!   region so the failure is loud rather than silent.
//! * **Shading (`sh`) and image XObjects.** Heatmaps, contour fills and any
//!   raster-backed figure carry no path geometry. We report nothing for them,
//!   and say so.
//! * **Soft masks and transparency groups** are ignored; a fully-transparent
//!   path is reported as if it were painted.

use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object};

use crate::geometry::{Matrix, Point, Rect};
use crate::style::{Paint, Rgb, SeriesStyle};

/// Guard against self-referential or maliciously nested Form XObjects. Mirrors
/// the cap `kopitiam-pdf`'s `font_resources` applies, for the same reason:
/// this crate must not hang or blow the stack on hostile input.
const MAX_XOBJECT_RECURSION_DEPTH: u32 = 32;

/// One segment of a subpath. Endpoints are in page space (CTM already applied).
///
/// See the module docs for why `Curve` retains its control points instead of
/// being flattened into `Line`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment {
    Line(Point),
    Curve {
        c1: Point,
        c2: Point,
        to: Point,
    },
}

impl Segment {
    /// The on-curve endpoint. For a Bézier this is the anchor; the control
    /// points are not on the curve and are never data.
    pub fn end(&self) -> Point {
        match self {
            Segment::Line(p) => *p,
            Segment::Curve { to, .. } => *to,
        }
    }
}

/// A connected run of segments: one `m` and everything that followed it.
#[derive(Debug, Clone, PartialEq)]
pub struct Subpath {
    pub start: Point,
    pub segments: Vec<Segment>,
    /// Set by `h` (closepath), or implied by `re`.
    pub closed: bool,
}

impl Subpath {
    /// The on-curve anchor points, in draw order: the start point followed by
    /// each segment's endpoint.
    ///
    /// For the polyline that a line plot is actually made of, this is the data
    /// series verbatim. See the module docs.
    pub fn anchors(&self) -> Vec<Point> {
        let mut pts = Vec::with_capacity(self.segments.len() + 1);
        pts.push(self.start);
        pts.extend(self.segments.iter().map(|s| s.end()));
        pts
    }

    /// Whether any segment is a Bézier, meaning the anchors may be spline knots
    /// rather than data points. Drives an honest warning rather than a silent
    /// assumption.
    pub fn has_curves(&self) -> bool {
        self.segments
            .iter()
            .any(|s| matches!(s, Segment::Curve { .. }))
    }

    /// Bounding box of the anchors *and* the control points.
    ///
    /// Control points are included because a Bézier is contained within the
    /// convex hull of its control polygon: this is guaranteed to contain the
    /// curve, whereas the anchor box alone is not. For structural questions
    /// ("is this short enough to be a tick mark?") a box that might be too
    /// small is a box that lies.
    pub fn bbox(&self) -> Rect {
        let mut pts = vec![self.start];
        for seg in &self.segments {
            match seg {
                Segment::Line(p) => pts.push(*p),
                Segment::Curve { c1, c2, to } => pts.extend([*c1, *c2, *to]),
            }
        }
        Rect::bounding(pts).unwrap_or(Rect::from_corners(
            self.start.x,
            self.start.y,
            self.start.x,
            self.start.y,
        ))
    }

    /// The straight line segments of this subpath, as endpoint pairs, including
    /// the implicit closing segment when `closed`.
    ///
    /// Bézier segments are skipped: this exists to answer geometric questions
    /// about *straight* things (axis spines, tick marks, grid lines, legend
    /// keys), and a curve is none of them.
    pub fn line_segments(&self) -> Vec<(Point, Point)> {
        let mut out = Vec::new();
        let mut cursor = self.start;
        for seg in &self.segments {
            if let Segment::Line(p) = seg {
                out.push((cursor, *p));
            }
            cursor = seg.end();
        }
        if self.closed && !self.segments.is_empty() {
            out.push((cursor, self.start));
        }
        out
    }
}

/// A path as it was painted: geometry plus the graphics state in force at the
/// moment the paint operator ran.
///
/// The style is captured at *paint* time, not at construction time, because
/// that is what the renderer uses -- a stream may set the colour after building
/// the path and before stroking it, and both orders occur in the wild.
#[derive(Debug, Clone, PartialEq)]
pub struct PaintedPath {
    pub subpaths: Vec<Subpath>,
    pub style: SeriesStyle,
    /// The clip bounding box in force, if any. A hint only -- see module docs.
    pub clip: Option<Rect>,
}

impl PaintedPath {
    pub fn bbox(&self) -> Option<Rect> {
        self.subpaths
            .iter()
            .map(|s| s.bbox())
            .reduce(|acc, b| acc.union(&b))
    }
}

/// Mutable graphics state, saved and restored by `q`/`Q`.
///
/// Only the parts that affect path geometry or a path's visual identity are
/// modelled. Text state is absent by design: text is `kopitiam-pdf`'s job and
/// re-deriving it here would duplicate font decoding for no gain.
#[derive(Debug, Clone)]
struct GraphicsState {
    ctm: Matrix,
    stroke_color: Rgb,
    fill_color: Rgb,
    /// Line width in *user* space, as written by `w`. Converted to page space
    /// only at paint time, when the CTM that will scale it is known.
    line_width: f32,
    /// Dash `on`/`off` lengths in user space, as written by `d`.
    dash: Vec<f32>,
    clip: Option<Rect>,
    /// Number of colour components in the current stroke/fill colour space,
    /// when it was set by `CS`/`cs` to a space we recognise. `sc`/`scn` carry
    /// no colour-space information of their own, so this is how we know whether
    /// their operands are gray, RGB or CMYK.
    stroke_space_components: Option<usize>,
    fill_space_components: Option<usize>,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            ctm: Matrix::IDENTITY,
            // PDF's initial colour in any device space is black (ISO 32000-1
            // §8.6.8), and the initial line width is 1.0 (Table 52).
            stroke_color: Rgb::BLACK,
            fill_color: Rgb::BLACK,
            line_width: 1.0,
            dash: Vec::new(),
            clip: None,
            stroke_space_components: None,
            fill_space_components: None,
        }
    }
}

/// Path under construction, in *user* space until a paint operator fires.
#[derive(Debug, Default)]
struct PathBuilder {
    subpaths: Vec<Subpath>,
    current: Option<Subpath>,
    /// Current point in user space, needed for `v`/`y` which reference it.
    cursor: Point,
    start: Point,
}

impl PathBuilder {
    fn move_to(&mut self, p: Point, ctm: &Matrix) {
        self.flush();
        self.cursor = p;
        self.start = p;
        self.current = Some(Subpath {
            start: ctm.apply(p),
            segments: Vec::new(),
            closed: false,
        });
    }

    fn line_to(&mut self, p: Point, ctm: &Matrix) {
        // A `l` with no preceding `m` is malformed. Rather than drop it (which
        // would silently shorten a curve), treat the current point as the
        // implicit start -- which is what a lenient renderer does.
        if self.current.is_none() {
            self.move_to(self.cursor, ctm);
        }
        if let Some(sp) = self.current.as_mut() {
            sp.segments.push(Segment::Line(ctm.apply(p)));
        }
        self.cursor = p;
    }

    fn curve_to(&mut self, c1: Point, c2: Point, to: Point, ctm: &Matrix) {
        if self.current.is_none() {
            self.move_to(self.cursor, ctm);
        }
        if let Some(sp) = self.current.as_mut() {
            sp.segments.push(Segment::Curve {
                c1: ctm.apply(c1),
                c2: ctm.apply(c2),
                to: ctm.apply(to),
            });
        }
        self.cursor = to;
    }

    fn close(&mut self) {
        if let Some(sp) = self.current.as_mut() {
            sp.closed = true;
        }
        self.cursor = self.start;
    }

    /// `re x y w h`: a complete closed rectangular subpath.
    ///
    /// The corners are transformed individually rather than the rectangle being
    /// transformed as a box, because under a rotating or skewing CTM the result
    /// is a parallelogram, not an axis-aligned rectangle. Transforming a box
    /// would quietly straighten a rotated frame.
    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, ctm: &Matrix) {
        self.flush();
        let corners = [
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ];
        self.subpaths.push(Subpath {
            start: ctm.apply(corners[0]),
            segments: corners[1..]
                .iter()
                .map(|c| Segment::Line(ctm.apply(*c)))
                .collect(),
            closed: true,
        });
        // Per ISO 32000-1 Table 59, `re` leaves the current point at (x, y).
        self.cursor = Point::new(x, y);
        self.start = self.cursor;
    }

    fn flush(&mut self) {
        if let Some(sp) = self.current.take()
            && !sp.segments.is_empty()
        {
            self.subpaths.push(sp);
        }
    }

    /// Finish the path and hand back its subpaths, resetting for the next one.
    fn take(&mut self) -> Vec<Subpath> {
        self.flush();
        std::mem::take(&mut self.subpaths)
    }
}

/// Extract every painted path on one page, in page space.
///
/// `page_number` is 1-based, matching both `lopdf`'s `get_pages()` and
/// `kopitiam-pdf`'s `Page::number`, so the two extractions can be zipped by
/// page without an off-by-one.
pub fn paths_on_page(doc: &Document, page_number: u32) -> Vec<PaintedPath> {
    let empty = Dictionary::new();
    let Some(page_id) = doc.get_pages().get(&page_number).copied() else {
        return Vec::new();
    };
    let resources = doc
        .get_dictionary(page_id)
        .ok()
        .and_then(|d| inherited_resources(doc, d))
        .unwrap_or(&empty);

    let mut out = Vec::new();
    if let Ok(bytes) = doc.get_page_content(page_id) {
        walk(
            doc,
            &bytes,
            resources,
            GraphicsState::default(),
            &mut out,
            0,
        );
    }
    out
}

/// Resolve a page's effective `/Resources`, walking up `/Parent`.
///
/// Identical in behaviour to `kopitiam-pdf`'s helper of the same name: it
/// returns the *nearest* ancestor's dictionary rather than merging across
/// levels, which is what the spec's inheritance rule says and what every PDF
/// consumer in the tree already does.
fn inherited_resources<'a>(doc: &'a Document, page: &'a Dictionary) -> Option<&'a Dictionary> {
    if let Ok(obj) = page.get(b"Resources")
        && let Ok((_, resolved)) = doc.dereference(obj)
        && let Ok(dict) = resolved.as_dict()
    {
        return Some(dict);
    }
    let parent = page.get(b"Parent").ok()?.as_reference().ok()?;
    inherited_resources(doc, doc.get_dictionary(parent).ok()?)
}

fn walk(
    doc: &Document,
    bytes: &[u8],
    resources: &Dictionary,
    initial: GraphicsState,
    out: &mut Vec<PaintedPath>,
    depth: u32,
) {
    if depth > MAX_XOBJECT_RECURSION_DEPTH {
        return;
    }
    let Ok(content) = Content::decode(bytes) else {
        // A stream we cannot decode contributes no paths. That is an honest
        // "we saw nothing here", and the caller will notice the absence of a
        // plot rather than being handed a wrong one.
        return;
    };

    let mut gs = initial;
    let mut stack: Vec<GraphicsState> = Vec::new();
    let mut path = PathBuilder::default();
    // Set by `W`/`W*`, consumed by the next paint operator (ISO 32000-1 §8.5.4:
    // the clip takes effect *after* the path is painted).
    let mut pending_clip = false;

    for op in &content.operations {
        let n = &op.operands;
        match op.operator.as_str() {
            "q" => stack.push(gs.clone()),
            "Q" => {
                if let Some(restored) = stack.pop() {
                    gs = restored;
                }
            }
            "cm" => {
                if let Some(m) = matrix_from(n) {
                    gs.ctm = m.then(&gs.ctm);
                }
            }

            // ---- path construction (operands are in user space) ----
            "m" => {
                if let [x, y] = nums(n)[..] {
                    path.move_to(Point::new(x, y), &gs.ctm);
                }
            }
            "l" => {
                if let [x, y] = nums(n)[..] {
                    path.line_to(Point::new(x, y), &gs.ctm);
                }
            }
            "c" => {
                if let [x1, y1, x2, y2, x3, y3] = nums(n)[..] {
                    path.curve_to(
                        Point::new(x1, y1),
                        Point::new(x2, y2),
                        Point::new(x3, y3),
                        &gs.ctm,
                    );
                }
            }
            // `v`: the current point doubles as the first control point.
            "v" => {
                if let [x2, y2, x3, y3] = nums(n)[..] {
                    let c1 = path.cursor;
                    path.curve_to(c1, Point::new(x2, y2), Point::new(x3, y3), &gs.ctm);
                }
            }
            // `y`: the endpoint doubles as the second control point.
            "y" => {
                if let [x1, y1, x3, y3] = nums(n)[..] {
                    let to = Point::new(x3, y3);
                    path.curve_to(Point::new(x1, y1), to, to, &gs.ctm);
                }
            }
            "h" => path.close(),
            "re" => {
                if let [x, y, w, h] = nums(n)[..] {
                    path.rect(x, y, w, h, &gs.ctm);
                }
            }

            // ---- clipping ----
            "W" | "W*" => pending_clip = true,

            // ---- painting ----
            // Every one of these both paints and *ends* the path. Note that
            // `pdf-extract` handles only `S` and `f` and drops the rest --
            // see the module docs. Missing `B` alone would lose most scatter
            // plots.
            "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n" => {
                if matches!(op.operator.as_str(), "s" | "b" | "b*") {
                    path.close();
                }
                let subpaths = path.take();
                if let Some(paint) = paint_kind(op.operator.as_str()) {
                    emit(&gs, paint, &subpaths, out);
                }
                // `W` sets the clip to the path just built, whatever the paint
                // operator was -- including `n`, which is the usual idiom
                // (`W n` clips without painting).
                if pending_clip {
                    if let Some(bbox) = subpaths
                        .iter()
                        .map(|s| s.bbox())
                        .reduce(|acc, b| acc.union(&b))
                    {
                        gs.clip = Some(match gs.clip {
                            // Nested clips intersect. We approximate the
                            // intersection by the smaller box, which is
                            // conservative for our only use of it (a hint).
                            Some(prev) if prev.area() < bbox.area() => prev,
                            _ => bbox,
                        });
                    }
                    pending_clip = false;
                }
            }

            // ---- colour ----
            "G" => {
                if let [v] = nums(n)[..] {
                    gs.stroke_color = Rgb::gray(v);
                    gs.stroke_space_components = Some(1);
                }
            }
            "g" => {
                if let [v] = nums(n)[..] {
                    gs.fill_color = Rgb::gray(v);
                    gs.fill_space_components = Some(1);
                }
            }
            "RG" => {
                if let [r, g, b] = nums(n)[..] {
                    gs.stroke_color = Rgb::new(r, g, b);
                    gs.stroke_space_components = Some(3);
                }
            }
            "rg" => {
                if let [r, g, b] = nums(n)[..] {
                    gs.fill_color = Rgb::new(r, g, b);
                    gs.fill_space_components = Some(3);
                }
            }
            "K" => {
                if let [c, m, y, k] = nums(n)[..] {
                    gs.stroke_color = Rgb::cmyk(c, m, y, k);
                    gs.stroke_space_components = Some(4);
                }
            }
            "k" => {
                if let [c, m, y, k] = nums(n)[..] {
                    gs.fill_color = Rgb::cmyk(c, m, y, k);
                    gs.fill_space_components = Some(4);
                }
            }
            "CS" => gs.stroke_space_components = space_components(n),
            "cs" => gs.fill_space_components = space_components(n),
            // `sc`/`scn` set a colour in whatever space `CS`/`cs` selected.
            // With a named space we cannot resolve (Separation, ICCBased,
            // Indexed, ...) the operand count is the only signal available, and
            // for 1/3/4 numeric operands it is unambiguous in practice. Any
            // other shape (a pattern name, say) leaves the colour untouched at
            // its previous value rather than guessing.
            "SC" | "SCN" => {
                if let Some(c) = color_from_components(&nums(n), gs.stroke_space_components) {
                    gs.stroke_color = c;
                }
            }
            "sc" | "scn" => {
                if let Some(c) = color_from_components(&nums(n), gs.fill_space_components) {
                    gs.fill_color = c;
                }
            }

            // ---- stroke parameters ----
            "w" => {
                if let [v] = nums(n)[..] {
                    gs.line_width = v;
                }
            }
            "d" => {
                if let Some(Object::Array(items)) = n.first() {
                    gs.dash = items.iter().filter_map(as_f32).collect();
                }
            }
            // An ExtGState can carry line width (`/LW`) and dash (`/D`), and
            // some producers set them only there. Ignoring it would silently
            // merge a dashed series into a solid one.
            "gs" => {
                if let Some(name) = n.first().and_then(|o| o.as_name().ok())
                    && let Ok(states) = doc.get_dict_in_dict(resources, b"ExtGState")
                    && let Ok(state) = doc.get_dict_in_dict(states, name)
                {
                    apply_ext_gstate(doc, state, &mut gs);
                }
            }

            "Do" => {
                if let Some(name) = n.first().and_then(|o| o.as_name().ok())
                    && let Some((xres, bytes, form_matrix)) = form_xobject(doc, resources, name)
                {
                    // A Form XObject is drawn with its own `/Matrix` applied on
                    // top of the caller's CTM, and inherits the rest of the
                    // graphics state (ISO 32000-1 §8.10.2). Inheriting is what
                    // makes marker XObjects work: the same form is invoked once
                    // per data point with a different `cm` each time, and each
                    // invocation must land where its CTM puts it.
                    let mut inner = gs.clone();
                    inner.ctm = form_matrix.then(&gs.ctm);
                    walk(doc, &bytes, xres, inner, out, depth + 1);
                }
            }
            _ => {}
        }
    }
}

/// Push a painted path, dropping empty ones and normalising the style.
fn emit(gs: &GraphicsState, paint: Paint, subpaths: &[Subpath], out: &mut Vec<PaintedPath>) {
    if subpaths.is_empty() {
        return;
    }
    let scale = gs.ctm.mean_scale();
    let style = SeriesStyle {
        paint,
        // A filled marker's identity is its fill colour; a stroked curve's is
        // its stroke colour. For fill+stroke the fill dominates visually.
        color: if paint.fills() {
            gs.fill_color
        } else {
            gs.stroke_color
        },
        line_width: if paint.strokes() {
            // A `w` of 0 means "thinnest line the device can render" (ISO
            // 32000-1 Table 52), not "invisible". Report it as 0 and let
            // grouping treat it as its own width, which is exactly what a
            // renderer does.
            gs.line_width * scale
        } else {
            0.0
        },
        dash: if paint.strokes() {
            gs.dash.iter().map(|d| d * scale).collect()
        } else {
            Vec::new()
        },
    };
    out.push(PaintedPath {
        subpaths: subpaths.to_vec(),
        style,
        clip: gs.clip,
    });
}

/// Map a paint operator to how it inks the path. `n` (no-op) returns `None`:
/// it ends the path without painting, and a path that was never painted was
/// never visible, so it is not evidence of anything.
fn paint_kind(op: &str) -> Option<Paint> {
    match op {
        "S" | "s" => Some(Paint::Stroke),
        "f" | "F" | "f*" => Some(Paint::Fill),
        "B" | "B*" | "b" | "b*" => Some(Paint::FillStroke),
        _ => None,
    }
}

/// Read `/LW` and `/D` out of an ExtGState dictionary.
fn apply_ext_gstate(doc: &Document, state: &Dictionary, gs: &mut GraphicsState) {
    if let Ok(lw) = state.get(b"LW")
        && let Ok((_, lw)) = doc.dereference(lw)
        && let Some(v) = as_f32(lw)
    {
        gs.line_width = v;
    }
    // `/D` is `[[on off ...] phase]`.
    if let Ok(d) = state.get(b"D")
        && let Ok((_, d)) = doc.dereference(d)
        && let Ok(arr) = d.as_array()
        && let Some(Object::Array(pattern)) = arr.first()
    {
        gs.dash = pattern.iter().filter_map(as_f32).collect();
    }
}

/// Resolve a `Do` name to a *Form* XObject: its resources, content bytes, and
/// `/Matrix`.
///
/// Image XObjects are deliberately rejected here rather than being walked as if
/// they were content streams. An image contributes no path geometry by
/// definition, and attempting to decode its bytes as operators would at best
/// waste time and at worst manufacture nonsense paths out of pixel data.
fn form_xobject<'a>(
    doc: &'a Document,
    resources: &'a Dictionary,
    name: &[u8],
) -> Option<(&'a Dictionary, Vec<u8>, Matrix)> {
    let xobjects = doc.get_dict_in_dict(resources, b"XObject").ok()?;
    let (_, obj) = doc.dereference(xobjects.get(name).ok()?).ok()?;
    let stream = obj.as_stream().ok()?;

    if stream.dict.get(b"Subtype").ok()?.as_name().ok()? != b"Form" {
        return None;
    }

    let matrix = stream
        .dict
        .get(b"Matrix")
        .ok()
        .and_then(|m| doc.dereference(m).ok())
        .and_then(|(_, m)| m.as_array().ok().and_then(|a| matrix_from(a)))
        .unwrap_or(Matrix::IDENTITY);

    let xres = stream
        .dict
        .get(b"Resources")
        .ok()
        .and_then(|r| doc.dereference(r).ok())
        .and_then(|(_, o)| o.as_dict().ok())
        .unwrap_or(resources);

    let bytes = stream
        .decompressed_content()
        .unwrap_or_else(|_| stream.content.clone());
    Some((xres, bytes, matrix))
}

/// Number of colour components for the device colour spaces we recognise by
/// name. Anything else yields `None`, which makes `sc`/`scn` fall back to the
/// operand count.
fn space_components(operands: &[Object]) -> Option<usize> {
    match operands.first()?.as_name().ok()? {
        b"DeviceGray" | b"G" | b"CalGray" => Some(1),
        b"DeviceRGB" | b"RGB" | b"CalRGB" | b"Lab" => Some(3),
        b"DeviceCMYK" | b"CMYK" => Some(4),
        _ => None,
    }
}

/// Interpret `sc`/`scn` operands. The declared component count is preferred;
/// where the space was unrecognised, the operand count is the fallback.
fn color_from_components(values: &[f32], declared: Option<usize>) -> Option<Rgb> {
    let n = declared.unwrap_or(values.len());
    match (n, values) {
        (1, [v]) => Some(Rgb::gray(*v)),
        (3, [r, g, b]) => Some(Rgb::new(*r, *g, *b)),
        (4, [c, m, y, k]) => Some(Rgb::cmyk(*c, *m, *y, *k)),
        _ => None,
    }
}

fn matrix_from(operands: &[Object]) -> Option<Matrix> {
    match nums(operands)[..] {
        [a, b, c, d, e, f] => Some(Matrix::new(a, b, c, d, e, f)),
        _ => None,
    }
}

fn nums(operands: &[Object]) -> Vec<f32> {
    operands.iter().filter_map(as_f32).collect()
}

fn as_f32(o: &Object) -> Option<f32> {
    match o {
        Object::Integer(i) => Some(*i as f32),
        Object::Real(r) => Some(*r),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subpath_anchors_exclude_control_points() {
        let sp = Subpath {
            start: Point::new(0.0, 0.0),
            segments: vec![
                Segment::Line(Point::new(1.0, 1.0)),
                Segment::Curve {
                    c1: Point::new(50.0, 50.0),
                    c2: Point::new(60.0, 60.0),
                    to: Point::new(2.0, 2.0),
                },
            ],
            closed: false,
        };
        // The control points at (50,50)/(60,60) must NOT appear as data.
        assert_eq!(
            sp.anchors(),
            vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(2.0, 2.0)
            ]
        );
        assert!(sp.has_curves());
        // ...but they DO count towards the bounding box, because the curve can
        // bulge towards them.
        assert!(sp.bbox().right() >= 50.0);
    }

    #[test]
    fn closed_subpath_yields_closing_segment() {
        let sp = Subpath {
            start: Point::new(0.0, 0.0),
            segments: vec![
                Segment::Line(Point::new(1.0, 0.0)),
                Segment::Line(Point::new(1.0, 1.0)),
            ],
            closed: true,
        };
        let segs = sp.line_segments();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[2], (Point::new(1.0, 1.0), Point::new(0.0, 0.0)));
    }

    #[test]
    fn paint_kinds_cover_every_operator_pdf_extract_drops() {
        // The five operators `pdf-extract` silently discards. If any of these
        // regress to `None`, series will vanish from real figures.
        for op in ["s", "f*", "B", "B*", "b"] {
            assert!(paint_kind(op).is_some(), "{op} must paint");
        }
        assert_eq!(paint_kind("n"), None);
    }

    #[test]
    fn scn_falls_back_to_operand_count_for_unknown_spaces() {
        assert_eq!(
            color_from_components(&[1.0, 0.0, 0.0], None),
            Some(Rgb::new(1.0, 0.0, 0.0))
        );
        // A declared space wins over the operand count.
        assert_eq!(color_from_components(&[0.5], Some(1)), Some(Rgb::gray(0.5)));
        // A pattern name (no numeric operands) must not invent a colour.
        assert_eq!(color_from_components(&[], None), None);
    }

    #[test]
    fn rect_builds_a_closed_four_corner_subpath() {
        let mut pb = PathBuilder::default();
        pb.rect(10.0, 20.0, 30.0, 40.0, &Matrix::IDENTITY);
        let subpaths = pb.take();
        assert_eq!(subpaths.len(), 1);
        assert!(subpaths[0].closed);
        assert_eq!(subpaths[0].anchors().len(), 4);
        assert_eq!(subpaths[0].bbox(), Rect::from_corners(10.0, 20.0, 40.0, 60.0));
    }
}
