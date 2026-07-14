//! Plane geometry in **PDF user space**: the coordinate system every value in
//! this crate that is called a "page coordinate" lives in.
//!
//! # Which space is "page space", exactly
//!
//! This matters more than it looks, because plot digitisation is fundamentally
//! an act of *matching two independently extracted things* -- path geometry
//! (which we recover ourselves, in [`crate::content`]) against tick-label text
//! (which [`kopitiam_pdf`] recovers for us). If those two land in different
//! coordinate systems, every tick match is garbage and the calibration is
//! silently wrong. So the convention has to be pinned down, not assumed.
//!
//! PDF user space, per ISO 32000-1 §8.3.2.3, has its origin at the page's
//! **bottom-left** with **y increasing upward**. `kopitiam-pdf`'s
//! [`kopitiam_pdf::TextSpan`] coordinates are in exactly this space, and the
//! reason is worth writing down because it is not obvious and it is load-bearing:
//!
//! `pdf-extract` internally computes a `flip_ctm` that would convert to a
//! top-left-origin, y-down space -- but it only applies that flip inside its
//! *own* HTML/SVG output devices. The generic [`pdf_extract::OutputDev`]
//! callback path that `kopitiam-pdf` builds on passes the text-rendering matrix
//! straight through: `show_text` takes the flip matrix as an argument named
//! `_flip_ctm` and never uses it, and `Processor`'s CTM starts at the identity.
//! So `output_character` receives `Trm = Tsm x Tm x CTM` in raw user space,
//! y-up.
//!
//! Our own content-stream walk starts its CTM at the identity too, and applies
//! `cm` exactly as the spec says. Therefore **paths and text spans agree by
//! construction**, with no flip to reconcile. The round-trip test
//! (`tests/roundtrip.rs`) is what actually *proves* this, rather than us taking
//! it on faith: it draws a plot at coordinates it chose itself and checks the
//! numbers come back.
//!
//! One consequence to be aware of: because the initial CTM is the identity
//! rather than a translation by the MediaBox origin, a page whose MediaBox does
//! not start at `(0, 0)` yields coordinates relative to raw user space, not to
//! the visible page corner. This is harmless here -- axis calibration is an
//! affine fit, so a constant offset shared by ticks and curves cancels out
//! exactly -- and it keeps us bit-for-bit consistent with `kopitiam-pdf`, which
//! is what actually matters.

use serde::{Deserialize, Serialize};

/// A point in PDF user space (points; origin bottom-left, y up).
///
/// `Default` is the origin, which is also PDF's initial current point.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// An axis-aligned rectangle in PDF user space.
///
/// Deliberately mirrors [`kopitiam_pdf::Rect`]'s field layout rather than
/// re-using it, because this crate needs `Serialize` (a digitised plot is
/// persisted as knowledge) and constructors from corner pairs that the PDF
/// crate has no use for. Conversion is provided by [`Rect::from_pdf`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    /// Build a rectangle from any two opposite corners, normalising so that
    /// `width` and `height` are non-negative.
    pub fn from_corners(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            x: x0.min(x1),
            y: y0.min(y1),
            width: (x1 - x0).abs(),
            height: (y1 - y0).abs(),
        }
    }

    /// The bounding box of a point set, or `None` if the set is empty.
    pub fn bounding(points: impl IntoIterator<Item = Point>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let (mut min_x, mut min_y) = (first.x, first.y);
        let (mut max_x, mut max_y) = (first.x, first.y);
        for p in iter {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        Some(Self::from_corners(min_x, min_y, max_x, max_y))
    }

    pub fn from_pdf(rect: kopitiam_pdf::Rect) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }

    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn top(&self) -> f32 {
        self.y + self.height
    }

    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.x && p.x <= self.right() && p.y >= self.y && p.y <= self.top()
    }

    /// Grow the rectangle by `pad` on every side.
    pub fn padded(&self, pad: f32) -> Self {
        Self {
            x: self.x - pad,
            y: self.y - pad,
            width: self.width + 2.0 * pad,
            height: self.height + 2.0 * pad,
        }
    }

    /// The union of two rectangles.
    pub fn union(&self, other: &Rect) -> Self {
        Self::from_corners(
            self.x.min(other.x),
            self.y.min(other.y),
            self.right().max(other.right()),
            self.top().max(other.top()),
        )
    }

    /// Area of the intersection, zero if disjoint.
    pub fn intersection_area(&self, other: &Rect) -> f32 {
        let w = (self.right().min(other.right()) - self.x.max(other.x)).max(0.0);
        let h = (self.top().min(other.top()) - self.y.max(other.y)).max(0.0);
        w * h
    }

    /// Intersection over the *smaller* of the two areas.
    ///
    /// Used for de-duplicating candidate plot regions. Deliberately not the
    /// classic intersection-over-union: a boxed plot frame produces several
    /// candidate regions that are near-identical, but also occasionally one
    /// that is *nested* inside another (an inset, or a frame drawn twice at
    /// slightly different extents). IoU under-reports containment, whereas
    /// dividing by the smaller area reports a fully-contained region as `1.0`,
    /// which is what we want to collapse.
    pub fn overlap_ratio(&self, other: &Rect) -> f32 {
        let smaller = self.area().min(other.area());
        if smaller <= 0.0 {
            return 0.0;
        }
        self.intersection_area(other) / smaller
    }
}

/// A 2-D affine transform in PDF's row-vector convention.
///
/// PDF matrices are written `[a b c d e f]` and denote
///
/// ```text
///            | a  b  0 |
/// [x y 1]  x | c  d  0 |  =  [ a*x + c*y + e ,  b*x + d*y + f , 1 ]
///            | e  f  1 |
/// ```
///
/// (ISO 32000-1 §8.3.3). The row-vector convention is why concatenation reads
/// "backwards" relative to the column-vector convention most graphics code
/// uses -- see [`Matrix::then`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Matrix {
    pub const IDENTITY: Matrix = Matrix {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// Apply this transform to a point.
    pub fn apply(&self, p: Point) -> Point {
        Point::new(
            self.a * p.x + self.c * p.y + self.e,
            self.b * p.x + self.d * p.y + self.f,
        )
    }

    /// Compose: the transform that applies `self` first, then `outer`.
    ///
    /// This is exactly what the `cm` operator needs. `cm` *pre*-concatenates
    /// its matrix onto the CTM (`CTM' = m x CTM`), so a `cm` inside a `q ... Q`
    /// block composes on top of whatever transform is already in effect. It is
    /// also what a Form XObject's `/Matrix` needs, for the same reason.
    ///
    /// Getting this order backwards is the single easiest way to produce a plot
    /// that looks plausible and is entirely wrong, so it has a unit test below.
    pub fn then(&self, outer: &Matrix) -> Matrix {
        Matrix {
            a: self.a * outer.a + self.b * outer.c,
            b: self.a * outer.b + self.b * outer.d,
            c: self.c * outer.a + self.d * outer.c,
            d: self.c * outer.b + self.d * outer.d,
            e: self.e * outer.a + self.f * outer.c + outer.e,
            f: self.e * outer.b + self.f * outer.d + outer.f,
        }
    }

    /// The factor by which this transform scales lengths, on average.
    ///
    /// Used to convert a line width from the user space it was declared in
    /// (the `w` operator's operand) into the page space we report geometry in.
    /// For a non-uniform scale there is no single correct answer -- a stroke is
    /// genuinely wider in one direction than the other -- so we take
    /// `sqrt(|det|)`, the geometric mean of the principal scale factors, which
    /// is the standard choice and is exact for the uniform case that essentially
    /// every real plot uses.
    pub fn mean_scale(&self) -> f32 {
        (self.a * self.d - self.b * self.c).abs().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_leaves_points_alone() {
        let p = Point::new(3.0, 4.0);
        assert_eq!(Matrix::IDENTITY.apply(p), p);
    }

    #[test]
    fn applies_scale_then_translate_in_pdf_order() {
        // A `cm` of [2 0 0 2 10 20]: scale by 2, then translate by (10, 20).
        let m = Matrix::new(2.0, 0.0, 0.0, 2.0, 10.0, 20.0);
        assert_eq!(m.apply(Point::new(1.0, 1.0)), Point::new(12.0, 22.0));
    }

    #[test]
    fn composition_applies_self_before_outer() {
        // Inner scales by 2; outer translates by (100, 0). A point at x=1
        // must land at 2*1 + 100 = 102, NOT 2*(1 + 100) = 202. If `then` had
        // its arguments the wrong way round, this is the test that catches it.
        let inner = Matrix::new(2.0, 0.0, 0.0, 2.0, 0.0, 0.0);
        let outer = Matrix::new(1.0, 0.0, 0.0, 1.0, 100.0, 0.0);
        let composed = inner.then(&outer);
        assert_eq!(composed.apply(Point::new(1.0, 1.0)), Point::new(102.0, 2.0));
    }

    #[test]
    fn mean_scale_is_uniform_scale_factor() {
        let m = Matrix::new(3.0, 0.0, 0.0, 3.0, 0.0, 0.0);
        assert!((m.mean_scale() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_ratio_reports_containment_as_one() {
        let outer = Rect::from_corners(0.0, 0.0, 10.0, 10.0);
        let inner = Rect::from_corners(2.0, 2.0, 4.0, 4.0);
        assert!((outer.overlap_ratio(&inner) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bounding_box_of_points() {
        let r = Rect::bounding([Point::new(1.0, 5.0), Point::new(4.0, 2.0)]).unwrap();
        assert_eq!(r, Rect::from_corners(1.0, 2.0, 4.0, 5.0));
    }
}
