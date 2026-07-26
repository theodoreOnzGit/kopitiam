//! Ported from MuPDF `source/fitz/geometry.c` + `include/mupdf/fitz/geometry.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Translation conventions (this module sets the pattern for the whole port)
//!
//! The full write-up lives in
//! `docs/ai-decisions/AID-0051-mupdf-port-conventions.md`; the load-bearing bits,
//! kept here so the next porter doesn't have to go digging:
//!
//! * **`float` -> `f32`.** MuPDF's geometry runs in C `float`. We use `f32`
//!   everywhere so the numeric behaviour matches bit-for-bit where it can. The
//!   two places MuPDF deliberately widens to `double` -- matrix inversion and the
//!   `fmod` in `rotate` -- we widen to `f64` the same way, then narrow back.
//! * **`FZ_PI` is `3.14159265f`, NOT `std::f32::consts::PI`.** MuPDF hard-codes a
//!   truncated pi (see `fitz/system.h`). Using the "correct" pi would drift the
//!   last ULP of every rotation off MuPDF. We copy the truncated constant.
//! * **`fz_min`/`fz_max` are the ternary `a < b ? a : b`, not `f32::min`.**
//!   `f32::min(a, NaN)` returns the non-NaN operand; MuPDF's `fz_min` returns
//!   whatever the ternary picks (NaN propagates when it's the second operand).
//!   We reimplement the ternary so NaN behaviour matches -- it matters for quads.
//! * **Types.** `fz_matrix (a b c d e f)` -> [`Matrix`]; `fz_point` ->
//!   [`Point`]; `fz_rect` -> [`Rect`]; `fz_irect` -> [`IRect`]; `fz_quad` ->
//!   [`Quad`]. All `#[derive(Clone, Copy, Debug, PartialEq)]` (they are the same
//!   plain-old-data value types MuPDF passes by value).
//! * **Free functions -> methods where natural.** `fz_transform_rect(r, m)` ->
//!   `r.transform(m)`, `fz_concat(a, b)` -> `a.concat(b)` (also `a * b`), etc.
//!   Each keeps a `// MuPDF: fz_<name>` breadcrumb so the 1:1 map is findable.
//! * **No `fz_context` here** -- geometry is pure math. For later modules the
//!   convention (documented in the AID, not implemented here) is:
//!   `fz_context` -> Rust ownership / explicit allocators, and
//!   `fz_try`/`fz_catch`/`fz_throw` -> `Result`.
//!
//! ## Not ported from this file (deliberately)
//!
//! `geometry.c` also carries the checked-integer helpers (`fz_ckd_mul_*`,
//! `fz_ckd_add_*`, ...) and `geometry.h` the 0..255 pixel-blend macros
//! (`fz_mul255`, `FZ_BLEND`, ...). Those are not geometry and not on the
//! text-extraction path -- Rust has `checked_mul` / `saturating_add` in std when
//! we need them. `fz_gridfit_matrix` is *declared* in the header but has **no
//! body in geometry.c** (it lives in another translation unit), so it is out of
//! scope for this file too.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `FZ_MIN_INF_RECT` -- `(int)0x80000000`, i.e. `i32::MIN`. The smallest 32-bit
/// signed integer, used as the lower bound of an infinite rect.
const FZ_MIN_INF_RECT: i32 = i32::MIN; // 0x80000000
/// `FZ_MAX_INF_RECT` -- `0x7fffff80` (2147483520). MuPDF picks the *largest*
/// 32-bit signed int that survives round-tripping through `f32` (it is
/// `0xFFFFFF << 7`, so exactly representable), for the upper bound of an
/// infinite rect.
const FZ_MAX_INF_RECT: i32 = 0x7fff_ff80; // 2147483520

/// Biggest / smallest integers a float can represent perfectly (24 bits).
/// MuPDF clamps rounded rect coordinates to this range.
const MAX_SAFE_INT: f32 = 16_777_216.0;
const MIN_SAFE_INT: f32 = -16_777_216.0;

/// MuPDF's `FZ_PI` from `fitz/system.h` -- a **truncated** pi (`3.14159265f`),
/// deliberately not `std::f32::consts::PI`, so rotations match MuPDF exactly.
/// The clippy lints below are silenced ON PURPOSE: using the "real" pi constant
/// or trimming digits would defeat the whole point of copying MuPDF's literal.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const FZ_PI: f32 = 3.141_592_65;

/// `FLT_EPSILON` -- the C `<float.h>` value, which is exactly `f32::EPSILON`.
const FLT_EPSILON: f32 = f32::EPSILON;
/// `DBL_EPSILON` -- the C `<float.h>` value, which is exactly `f64::EPSILON`.
const DBL_EPSILON: f64 = f64::EPSILON;

// ---------------------------------------------------------------------------
// Small math helpers (fz_min / fz_max / fz_clamp), faithful to the C ternary.
// ---------------------------------------------------------------------------

// MuPDF: fz_min (geometry.h)
#[inline]
fn fz_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

// MuPDF: fz_max (geometry.h)
#[inline]
fn fz_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

// MuPDF: fz_clamp (geometry.h)
#[inline]
fn fz_clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

// MuPDF: MAX4 / MIN4 (geometry.c)
#[inline]
fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    fz_max(fz_max(a, b), fz_max(c, d))
}
#[inline]
fn min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    fz_min(fz_min(a, b), fz_min(c, d))
}

// ===========================================================================
// Point -- fz_point
// ===========================================================================

/// A point in two-dimensional space. Ports `fz_point`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    // MuPDF: fz_make_point (geometry.h)
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }

    // MuPDF: fz_transform_point (geometry.c:334)
    /// Apply a transform to this point (translation included).
    #[inline]
    pub fn transform(self, m: Matrix) -> Point {
        Point {
            x: self.x * m.a + self.y * m.c + m.e,
            y: self.x * m.b + self.y * m.d + m.f,
        }
    }

    // MuPDF: fz_transform_point_xy (geometry.c:343)
    /// Apply a transform to the raw coordinates `(x, y)` (translation included).
    #[inline]
    pub fn transform_xy(x: f32, y: f32, m: Matrix) -> Point {
        Point {
            x: x * m.a + y * m.c + m.e,
            y: x * m.b + y * m.d + m.f,
        }
    }

    // MuPDF: fz_transform_vector (geometry.c:352)
    /// Apply a transform to this point as a **vector** -- any translation
    /// (`e`, `f`) is ignored.
    #[inline]
    pub fn transform_vector(self, m: Matrix) -> Point {
        Point {
            x: self.x * m.a + self.y * m.c,
            y: self.x * m.b + self.y * m.d,
        }
    }

    // MuPDF: fz_normalize_vector (geometry.c:361)
    /// Scale this vector to length one. A zero-length vector is returned
    /// unchanged (matching MuPDF, which guards `len != 0`).
    #[inline]
    pub fn normalize(self) -> Point {
        let mut p = self;
        let len = p.x * p.x + p.y * p.y;
        if len != 0.0 {
            let len = len.sqrt();
            p.x /= len;
            p.y /= len;
        }
        p
    }
}

// ===========================================================================
// Matrix -- fz_matrix (row-major 3x3, stored as [a b c d e f])
// ===========================================================================

/// A row-major affine transform, `[ a b c d e f ]`, standing for
///
/// ```text
/// / a b 0 \
/// | c d 0 |
/// \ e f 1 /
/// ```
///
/// Points transform as row-vectors: `p' = p * M`. Ports `fz_matrix`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Matrix {
    // MuPDF: fz_identity (geometry.c:52)
    /// The identity transform, `[ 1 0 0 1 0 0 ]`.
    pub const IDENTITY: Matrix = Matrix { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: 0.0, f: 0.0 };

    // MuPDF: fz_make_matrix (geometry.h)
    #[inline]
    pub const fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        Matrix { a, b, c, d, e, f }
    }

    /// The identity transform. Prefer [`Matrix::IDENTITY`]; this mirrors MuPDF's
    /// `fz_identity` being usable as a value.
    #[inline]
    pub const fn identity() -> Self {
        Matrix::IDENTITY
    }

    // MuPDF: fz_is_identity (geometry.h)
    #[inline]
    pub fn is_identity(self) -> bool {
        self.a == 1.0 && self.b == 0.0 && self.c == 0.0 && self.d == 1.0 && self.e == 0.0 && self.f == 0.0
    }

    // MuPDF: fz_concat (geometry.c:54)
    /// Concatenate two transforms. `self.concat(other)` is the transform that
    /// applies `self` **first**, then `other` (`p * self * other`). Not
    /// commutative. Also available as `self * other`.
    #[inline]
    pub fn concat(self, other: Matrix) -> Matrix {
        let one = self;
        let two = other;
        Matrix {
            a: one.a * two.a + one.b * two.c,
            b: one.a * two.b + one.b * two.d,
            c: one.c * two.a + one.d * two.c,
            d: one.c * two.b + one.d * two.d,
            e: one.e * two.a + one.f * two.c + two.e,
            f: one.e * two.b + one.f * two.d + two.f,
        }
    }

    // MuPDF: fz_scale (geometry.c:67)
    #[inline]
    pub fn scale(sx: f32, sy: f32) -> Matrix {
        Matrix { a: sx, b: 0.0, c: 0.0, d: sy, e: 0.0, f: 0.0 }
    }

    // MuPDF: fz_pre_scale (geometry.c:77)
    /// Scale by premultiplication: `scale(sx, sy) * self`.
    #[inline]
    pub fn pre_scale(self, sx: f32, sy: f32) -> Matrix {
        let mut m = self;
        m.a *= sx;
        m.b *= sx;
        m.c *= sy;
        m.d *= sy;
        m
    }

    // MuPDF: fz_post_scale (geometry.c:87)
    /// Scale by postmultiplication: `self * scale(sx, sy)`.
    #[inline]
    pub fn post_scale(self, sx: f32, sy: f32) -> Matrix {
        let mut m = self;
        m.a *= sx;
        m.b *= sy;
        m.c *= sx;
        m.d *= sy;
        m.e *= sx;
        m.f *= sy;
        m
    }

    // MuPDF: fz_shear (geometry.c:99)
    /// Shearing matrix `[ 1 sy sx 1 0 0 ]`.
    #[inline]
    pub fn shear(sx: f32, sy: f32) -> Matrix {
        Matrix { a: 1.0, b: sy, c: sx, d: 1.0, e: 0.0, f: 0.0 }
    }

    // MuPDF: fz_pre_shear (geometry.c:109)
    /// Premultiply by a shearing matrix `[ 1 sy sx 1 0 0 ]`.
    #[inline]
    pub fn pre_shear(self, sx: f32, sy: f32) -> Matrix {
        let mut m = self;
        let a = m.a;
        let b = m.b;
        m.a += sy * m.c;
        m.b += sy * m.d;
        m.c += sx * a;
        m.d += sx * b;
        m
    }

    // MuPDF: fz_rotate (geometry.c:121)
    /// Rotation matrix for `degrees` of counter-clockwise rotation. The four
    /// axis-aligned angles (0/90/180/270) are special-cased to exact values,
    /// exactly as MuPDF does, so a 90-degree rotation has a clean `0`/`1` and no
    /// `sinf` round-off.
    pub fn rotate(degrees: f32) -> Matrix {
        // MuPDF uses `fmod` (double) then narrows; f32 `%` gives the same
        // truncated remainder for the values that reach the general branch.
        let mut theta = degrees % 360.0;
        if theta < 0.0 {
            theta += 360.0;
        }

        let (s, c);
        if (0.0 - theta).abs() < FLT_EPSILON {
            s = 0.0;
            c = 1.0;
        } else if (90.0 - theta).abs() < FLT_EPSILON {
            s = 1.0;
            c = 0.0;
        } else if (180.0 - theta).abs() < FLT_EPSILON {
            s = 0.0;
            c = -1.0;
        } else if (270.0 - theta).abs() < FLT_EPSILON {
            s = -1.0;
            c = 0.0;
        } else {
            s = (theta * FZ_PI / 180.0).sin();
            c = (theta * FZ_PI / 180.0).cos();
        }

        Matrix { a: c, b: s, c: -s, d: c, e: 0.0, f: 0.0 }
    }

    // MuPDF: fz_pre_rotate (geometry.c:164)
    /// Premultiply by a rotation of `degrees` (counter-clockwise):
    /// `rotate(degrees) * self`. Axis-aligned angles are handled without any
    /// trig, matching MuPDF.
    pub fn pre_rotate(self, degrees: f32) -> Matrix {
        let mut m = self;
        let mut theta = degrees % 360.0;
        if theta < 0.0 {
            theta += 360.0;
        }

        if (0.0 - theta).abs() < FLT_EPSILON {
            // Nothing to do.
        } else if (90.0 - theta).abs() < FLT_EPSILON {
            let a = m.a;
            let b = m.b;
            m.a = m.c;
            m.b = m.d;
            m.c = -a;
            m.d = -b;
        } else if (180.0 - theta).abs() < FLT_EPSILON {
            m.a = -m.a;
            m.b = -m.b;
            m.c = -m.c;
            m.d = -m.d;
        } else if (270.0 - theta).abs() < FLT_EPSILON {
            let a = m.a;
            let b = m.b;
            m.a = -m.c;
            m.b = -m.d;
            m.c = a;
            m.d = b;
        } else {
            let s = (theta * FZ_PI / 180.0).sin();
            let c = (theta * FZ_PI / 180.0).cos();
            let a = m.a;
            let b = m.b;
            m.a = c * a + s * m.c;
            m.b = c * b + s * m.d;
            m.c = -s * a + c * m.c;
            m.d = -s * b + c * m.d;
        }

        m
    }

    // MuPDF: fz_translate (geometry.c:215)
    #[inline]
    pub fn translate(tx: f32, ty: f32) -> Matrix {
        Matrix { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: tx, f: ty }
    }

    // MuPDF: fz_pre_translate (geometry.c:225)
    /// Translate by premultiplication: `translate(tx, ty) * self`.
    #[inline]
    pub fn pre_translate(self, tx: f32, ty: f32) -> Matrix {
        let mut m = self;
        m.e += tx * m.a + ty * m.c;
        m.f += tx * m.b + ty * m.d;
        m
    }

    // MuPDF: fz_transform_page (geometry.c:233)
    /// Build the transform that draws `mediabox` at `resolution` DPI and
    /// `rotate` degrees, snapping the page to a whole number of pixels and
    /// moving its origin to `(0, 0)`.
    pub fn transform_page(mediabox: Rect, resolution: f32, rotate: f32) -> Matrix {
        let user_w = mediabox.x1 - mediabox.x0;
        let user_h = mediabox.y1 - mediabox.y0;
        let pixel_w = (user_w * resolution / 72.0 + 0.5).floor();
        let pixel_h = (user_h * resolution / 72.0 + 0.5).floor();

        let mut matrix = Matrix::scale(pixel_w / user_w, pixel_h / user_h).pre_rotate(rotate);

        let pixel_box = mediabox.transform(matrix);
        matrix.e -= pixel_box.x0;
        matrix.f -= pixel_box.y0;

        matrix
    }

    // MuPDF: fz_invert_matrix (geometry.c:256)
    /// Invert the matrix. A degenerate (singular, `det ~= 0`) matrix cannot be
    /// inverted, and -- exactly like MuPDF -- the all-zero matrix is returned so
    /// the degeneracy is easy to spot when debugging. The determinant is
    /// computed in `f64`, as MuPDF does.
    // The `det >= -eps && det <= eps` form mirrors MuPDF's C 1:1; keep it that
    // way rather than clippy's `(-eps..=eps).contains(&det)` rewrite.
    #[allow(clippy::manual_range_contains)]
    pub fn invert(self) -> Matrix {
        let src = self;
        let sa = src.a as f64;
        let sb = src.b as f64;
        let sc = src.c as f64;
        let sd = src.d as f64;
        let mut det = sa * sd - sb * sc;
        if det >= -DBL_EPSILON && det <= DBL_EPSILON {
            return Matrix::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        det = 1.0 / det;
        let da = sd * det;
        let db = -sb * det;
        let dc = -sc * det;
        let dd = sa * det;
        let de = -(src.e as f64) * da - (src.f as f64) * dc;
        let df = -(src.e as f64) * db - (src.f as f64) * dd;
        Matrix::new(da as f32, db as f32, dc as f32, dd as f32, de as f32, df as f32)
    }

    // MuPDF: fz_try_invert_matrix (geometry.c:279)
    /// Try to invert. Returns `None` if the matrix is degenerate (singular),
    /// `Some(inverse)` otherwise. This is the `Result`-shaped sibling of
    /// [`Matrix::invert`] -- MuPDF returns a "1 == degenerate" flag; we return an
    /// `Option`. Note MuPDF's subtle bug-for-parity ordering (`e`/`f` reuse the
    /// already-overwritten `da`/`db`): we compute `de`/`df` from the original
    /// `da`..`dd` values, which is what the C ends up doing too.
    #[allow(clippy::manual_range_contains)]
    pub fn try_invert(self) -> Option<Matrix> {
        let src = self;
        let sa = src.a as f64;
        let sb = src.b as f64;
        let sc = src.c as f64;
        let sd = src.d as f64;
        let mut det = sa * sd - sb * sc;
        if det >= -DBL_EPSILON && det <= DBL_EPSILON {
            return None;
        }
        det = 1.0 / det;
        let da = sd * det;
        let db = -sb * det;
        let dc = -sc * det;
        let dd = sa * det;
        let de = -(src.e as f64) * da - (src.f as f64) * dc;
        let df = -(src.e as f64) * db - (src.f as f64) * dd;
        Some(Matrix::new(da as f32, db as f32, dc as f32, dd as f32, de as f32, df as f32))
    }

    // MuPDF: fz_is_rectilinear (geometry.c:305)
    /// True if the transform is rectilinear: no shear, and any rotation is a
    /// multiple of 90 degrees. Axis-aligned rects stay axis-aligned under it.
    #[inline]
    pub fn is_rectilinear(self) -> bool {
        (self.b.abs() < FLT_EPSILON && self.c.abs() < FLT_EPSILON)
            || (self.a.abs() < FLT_EPSILON && self.d.abs() < FLT_EPSILON)
    }

    // MuPDF: fz_matrix_expansion (geometry.c:312)
    /// Average scaling factor: `sqrt(|a*d - b*c|)`.
    #[inline]
    pub fn expansion(self) -> f32 {
        (self.a * self.d - self.b * self.c).abs().sqrt()
    }

    // MuPDF: fz_matrix_max_expansion (geometry.c:318)
    /// The largest expansion: `max(|a|, |b|, |c|, |d|)`.
    #[inline]
    pub fn max_expansion(self) -> f32 {
        let mut max = self.a.abs();
        let x = self.b.abs();
        if max < x {
            max = x;
        }
        let x = self.c.abs();
        if max < x {
            max = x;
        }
        let x = self.d.abs();
        if max < x {
            max = x;
        }
        max
    }
}

impl std::ops::Mul for Matrix {
    type Output = Matrix;
    // MuPDF: fz_concat (geometry.c:54) -- `left * right` applies `left` first.
    #[inline]
    fn mul(self, rhs: Matrix) -> Matrix {
        self.concat(rhs)
    }
}

// ===========================================================================
// Rect -- fz_rect
// ===========================================================================

/// An axis-aligned rectangle, two diagonally-opposite corners `(x0,y0)` /
/// `(x1,y1)`. Ports `fz_rect`.
///
/// MuPDF distinguishes three categories, and we preserve every one exactly:
/// * **valid** -- `x0 <= x1 && y0 <= y1` (zero-area rects are valid);
/// * **infinite** -- all four ordinates pinned to the `FZ_{MIN,MAX}_INF_RECT`
///   sentinels;
/// * **invalid** -- anything not valid (e.g. the `{0,0,-1,-1}` sentinel).
///
/// **Empty** ([`Rect::is_empty`]) means zero (or negative) area -- `x0 >= x1 ||
/// y0 >= y1` -- which includes all invalid rects. This is a *different* question
/// from validity, and the two must not be conflated (that distinction is the
/// whole reason MuPDF chose this representation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Rect {
    // MuPDF: fz_infinite_rect (geometry.c:380)
    pub const INFINITE: Rect = Rect {
        x0: FZ_MIN_INF_RECT as f32,
        y0: FZ_MIN_INF_RECT as f32,
        x1: FZ_MAX_INF_RECT as f32,
        y1: FZ_MAX_INF_RECT as f32,
    };
    // MuPDF: fz_empty_rect (geometry.c:381) -- note the corners are *swapped*
    // relative to INFINITE, giving a maximally-negative-area sentinel.
    pub const EMPTY: Rect = Rect {
        x0: FZ_MAX_INF_RECT as f32,
        y0: FZ_MAX_INF_RECT as f32,
        x1: FZ_MIN_INF_RECT as f32,
        y1: FZ_MIN_INF_RECT as f32,
    };
    // MuPDF: fz_invalid_rect (geometry.c:382)
    pub const INVALID: Rect = Rect { x0: 0.0, y0: 0.0, x1: -1.0, y1: -1.0 };
    // MuPDF: fz_unit_rect (geometry.c:383)
    pub const UNIT: Rect = Rect { x0: 0.0, y0: 0.0, x1: 1.0, y1: 1.0 };

    // MuPDF: fz_make_rect (geometry.h)
    #[inline]
    pub const fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Rect { x0, y0, x1, y1 }
    }

    // MuPDF: fz_is_empty_rect (geometry.h)
    /// Empty == zero-or-negative area. All invalid rects are empty.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    // MuPDF: fz_is_infinite_rect (geometry.h)
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.x0 == FZ_MIN_INF_RECT as f32
            && self.x1 == FZ_MAX_INF_RECT as f32
            && self.y0 == FZ_MIN_INF_RECT as f32
            && self.y1 == FZ_MAX_INF_RECT as f32
    }

    // MuPDF: fz_is_valid_rect (geometry.h)
    #[inline]
    pub fn is_valid(self) -> bool {
        self.x0 <= self.x1 && self.y0 <= self.y1
    }

    // MuPDF: fz_irect_from_rect (geometry.c:393)
    /// The minimal integer bounding box covering this rect: top-left rounds
    /// down/left (`floor`), bottom-right rounds up/right (`ceil`). Follows the
    /// numbers slavishly -- see [`Rect::round`] for the tolerant version.
    pub fn irect_from_rect(self) -> IRect {
        let r = self;
        if r.is_infinite() {
            return IRect::INFINITE;
        }
        if !r.is_valid() {
            return IRect::INVALID;
        }
        IRect {
            x0: fz_clamp(r.x0.floor(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
            y0: fz_clamp(r.y0.floor(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
            x1: fz_clamp(r.x1.ceil(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
            y1: fz_clamp(r.y1.ceil(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
        }
    }

    // MuPDF: fz_round_rect (geometry.c:425)
    /// Round to an integer bbox, allowing a small (0.001) tolerance so a hair of
    /// over/under-calculation doesn't add a whole extra pixel. Unlike
    /// [`Rect::irect_from_rect`], this does **not** special-case infinite rects
    /// (MuPDF doesn't either).
    pub fn round(self) -> IRect {
        let r = self;
        IRect {
            x0: fz_clamp((r.x0 + 0.001).floor(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
            y0: fz_clamp((r.y0 + 0.001).floor(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
            x1: fz_clamp((r.x1 - 0.001).ceil(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
            y1: fz_clamp((r.y1 - 0.001).ceil(), MIN_SAFE_INT, MAX_SAFE_INT) as i32,
        }
    }

    // MuPDF: fz_rect_from_irect (geometry.c:410)
    #[inline]
    pub fn from_irect(a: IRect) -> Rect {
        if a.is_infinite() {
            return Rect::INFINITE;
        }
        Rect { x0: a.x0 as f32, y0: a.y0 as f32, x1: a.x1 as f32, y1: a.y1 as f32 }
    }

    // MuPDF: fz_intersect_rect (geometry.c:443)
    /// Smallest rect covering the area common to both. An infinite operand is
    /// treated as "everything", so the intersection is the other rect. Note this
    /// does **not** pre-check emptiness -- intersecting disjoint rects yields a
    /// (possibly invalid/negative) rect that reports `is_empty()`, which is
    /// exactly what MuPDF relies on downstream.
    pub fn intersect(self, b: Rect) -> Rect {
        let mut a = self;
        if b.is_infinite() {
            return a;
        }
        if a.is_infinite() {
            return b;
        }
        if a.x0 < b.x0 {
            a.x0 = b.x0;
        }
        if a.y0 < b.y0 {
            a.y0 = b.y0;
        }
        if a.x1 > b.x1 {
            a.x1 = b.x1;
        }
        if a.y1 > b.y1 {
            a.y1 = b.y1;
        }
        a
    }

    // MuPDF: fz_union_rect (geometry.c:475)
    /// Smallest rect encompassing both. Emptiness is checked *before*
    /// infiniteness (MuPDF's ordering): an empty/invalid operand drops out and
    /// the other is returned; an infinite operand swallows the result.
    pub fn union(self, b: Rect) -> Rect {
        let mut a = self;
        // Check for empty (invalid) box before infinite box.
        if !b.is_valid() {
            return a;
        }
        if !a.is_valid() {
            return b;
        }
        if a.is_infinite() {
            return a;
        }
        if b.is_infinite() {
            return b;
        }
        if a.x0 > b.x0 {
            a.x0 = b.x0;
        }
        if a.y0 > b.y0 {
            a.y0 = b.y0;
        }
        if a.x1 < b.x1 {
            a.x1 = b.x1;
        }
        if a.y1 < b.y1 {
            a.y1 = b.y1;
        }
        a
    }

    // MuPDF: fz_translate_rect (geometry.c:494)
    /// Translate by `(xoff, yoff)`. An infinite rect is returned unchanged.
    #[inline]
    pub fn translate(self, xoff: f32, yoff: f32) -> Rect {
        let mut a = self;
        if a.is_infinite() {
            return a;
        }
        a.x0 += xoff;
        a.y0 += yoff;
        a.x1 += xoff;
        a.y1 += yoff;
        a
    }

    // MuPDF: fz_transform_rect (geometry.c:519)
    /// Transform this rect and return the smallest axis-aligned rect covering the
    /// (possibly sheared/rotated) image of all four corners. The two special
    /// cases matter and are preserved:
    /// * an **infinite** rect is returned unchanged;
    /// * for a rectilinear transform (only one of the b/c or a/d pairs is
    ///   non-zero) MuPDF takes a fast two-corner path, swapping corners first if
    ///   the relevant scale is negative, so an invalid rect stays invalid;
    /// * a general transform maps all four corners and takes the min/max, then
    ///   re-swaps to keep an invalid input invalid on output.
    pub fn transform(self, m: Matrix) -> Rect {
        let mut r = self;

        if r.is_infinite() {
            return r;
        }

        if m.b.abs() < FLT_EPSILON && m.c.abs() < FLT_EPSILON {
            if m.a < 0.0 {
                std::mem::swap(&mut r.x0, &mut r.x1);
            }
            if m.d < 0.0 {
                std::mem::swap(&mut r.y0, &mut r.y1);
            }
            let s = Point::transform_xy(r.x0, r.y0, m);
            let t = Point::transform_xy(r.x1, r.y1, m);
            r.x0 = s.x;
            r.y0 = s.y;
            r.x1 = t.x;
            r.y1 = t.y;
            // If r was invalid coming in, it'll still be invalid now.
            return r;
        } else if m.a.abs() < FLT_EPSILON && m.d.abs() < FLT_EPSILON {
            if m.b < 0.0 {
                std::mem::swap(&mut r.x0, &mut r.x1);
            }
            if m.c < 0.0 {
                std::mem::swap(&mut r.y0, &mut r.y1);
            }
            let s = Point::transform_xy(r.x0, r.y0, m);
            let t = Point::transform_xy(r.x1, r.y1, m);
            r.x0 = s.x;
            r.y0 = s.y;
            r.x1 = t.x;
            r.y1 = t.y;
            // If r was invalid coming in, it'll still be invalid now.
            return r;
        }

        let invalid = (r.x0 > r.x1) || (r.y0 > r.y1);
        let s = Point::new(r.x0, r.y0).transform(m);
        let t = Point::new(r.x0, r.y1).transform(m);
        let u = Point::new(r.x1, r.y1).transform(m);
        let v = Point::new(r.x1, r.y0).transform(m);
        r.x0 = min4(s.x, t.x, u.x, v.x);
        r.y0 = min4(s.y, t.y, u.y, v.y);
        r.x1 = max4(s.x, t.x, u.x, v.x);
        r.y1 = max4(s.y, t.y, u.y, v.y);

        // If we were called with an invalid rectangle, return an invalid
        // rectangle after transformation.
        if invalid {
            std::mem::swap(&mut r.x0, &mut r.x1);
            std::mem::swap(&mut r.y0, &mut r.y1);
        }
        r
    }

    // MuPDF: fz_expand_rect (geometry.c:608)
    /// Grow the rect by `expand` in every direction. Infinite and invalid rects
    /// are returned unchanged.
    #[inline]
    pub fn expand(self, expand: f32) -> Rect {
        let mut a = self;
        if a.is_infinite() {
            return a;
        }
        if !a.is_valid() {
            return a;
        }
        a.x0 -= expand;
        a.y0 -= expand;
        a.x1 += expand;
        a.y1 += expand;
        a
    }

    // MuPDF: fz_rect_area (geometry.c:620)
    /// Area, always non-negative. Empty/invalid rects return `0`; infinite
    /// rects return `+inf`.
    #[inline]
    pub fn area(self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        if self.is_infinite() {
            return f32::INFINITY;
        }
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }

    // MuPDF: fz_include_point_in_rect (geometry.c:635)
    /// Grow the rect to include `p`. Adding a point to an invalid rect makes the
    /// zero-area rect at that point (so, to bound a run of points, start from an
    /// empty rect at the first point). Infinite rects are returned unchanged.
    #[inline]
    pub fn include_point(self, p: Point) -> Rect {
        let mut r = self;
        if r.is_infinite() {
            return r;
        }
        if p.x < r.x0 {
            r.x0 = p.x;
        }
        if p.x > r.x1 {
            r.x1 = p.x;
        }
        if p.y < r.y0 {
            r.y0 = p.y;
        }
        if p.y > r.y1 {
            r.y1 = p.y;
        }
        r
    }

    // MuPDF: fz_contains_rect (geometry.c:645)
    /// True if `self` entirely contains `b`. An invalid `self` contains nothing;
    /// any valid rect contains all invalid rects.
    pub fn contains(self, b: Rect) -> bool {
        let a = self;
        if !a.is_valid() {
            return false;
        }
        if !b.is_valid() {
            return true;
        }
        a.x0 <= b.x0 && a.y0 <= b.y0 && a.x1 >= b.x1 && a.y1 >= b.y1
    }

    // MuPDF: fz_overlaps_rect (geometry.c:659)
    /// True if the overlap area of `self` and `b` is non-zero.
    #[inline]
    pub fn overlaps(self, b: Rect) -> bool {
        !self.intersect(b).is_empty()
    }

    // MuPDF: fz_is_point_inside_rect (geometry.c:784)
    /// Half-open inclusion: `p.x >= x0 && p.x < x1 && p.y >= y0 && p.y < y1`.
    /// The top/right edges are *excluded*.
    #[inline]
    pub fn is_point_inside(self, p: Point) -> bool {
        p.x >= self.x0 && p.x < self.x1 && p.y >= self.y0 && p.y < self.y1
    }

    // MuPDF: fz_is_rect_inside_rect (geometry.c:794)
    /// True if `inner` sits inside `outer` (both assumed open or both closed).
    /// No invalid rect includes or is included by anything.
    pub fn is_rect_inside(inner: Rect, outer: Rect) -> bool {
        if !outer.is_valid() {
            return false;
        }
        if !inner.is_valid() {
            return false;
        }
        inner.x0 >= outer.x0 && inner.x1 <= outer.x1 && inner.y0 >= outer.y0 && inner.y1 <= outer.y1
    }

    // MuPDF: fz_rect_from_quad (geometry.c:734)
    /// The smallest rect covering `q`. An invalid quad gives [`Rect::INVALID`],
    /// an infinite quad gives [`Rect::INFINITE`].
    pub fn from_quad(q: Quad) -> Rect {
        if !q.is_valid() {
            return Rect::INVALID;
        }
        if q.is_infinite() {
            return Rect::INFINITE;
        }
        Rect {
            x0: min4(q.ll.x, q.lr.x, q.ul.x, q.ur.x),
            y0: min4(q.ll.y, q.lr.y, q.ul.y, q.ur.y),
            x1: max4(q.ll.x, q.lr.x, q.ul.x, q.ur.x),
            y1: max4(q.ll.y, q.lr.y, q.ul.y, q.ur.y),
        }
    }
}

// ===========================================================================
// IRect -- fz_irect (integer rectangle)
// ===========================================================================

/// An integer axis-aligned rectangle, used for pixmap dimensions and bounding
/// boxes. Ports `fz_irect`. Same empty/infinite/invalid conventions as
/// [`Rect`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl IRect {
    // MuPDF: fz_infinite_irect (geometry.c:385)
    pub const INFINITE: IRect =
        IRect { x0: FZ_MIN_INF_RECT, y0: FZ_MIN_INF_RECT, x1: FZ_MAX_INF_RECT, y1: FZ_MAX_INF_RECT };
    // MuPDF: fz_empty_irect (geometry.c:386)
    pub const EMPTY: IRect =
        IRect { x0: FZ_MAX_INF_RECT, y0: FZ_MAX_INF_RECT, x1: FZ_MIN_INF_RECT, y1: FZ_MIN_INF_RECT };
    // MuPDF: fz_invalid_irect (geometry.c:387)
    pub const INVALID: IRect = IRect { x0: 0, y0: 0, x1: -1, y1: -1 };

    // MuPDF: fz_make_irect (geometry.h)
    #[inline]
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        IRect { x0, y0, x1, y1 }
    }

    // MuPDF: fz_is_empty_irect (geometry.h)
    #[inline]
    pub fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    // MuPDF: fz_is_infinite_irect (geometry.h)
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.x0 == FZ_MIN_INF_RECT
            && self.x1 == FZ_MAX_INF_RECT
            && self.y0 == FZ_MIN_INF_RECT
            && self.y1 == FZ_MAX_INF_RECT
    }

    // MuPDF: fz_is_valid_irect (geometry.h)
    #[inline]
    pub fn is_valid(self) -> bool {
        self.x0 <= self.x1 && self.y0 <= self.y1
    }

    // MuPDF: fz_irect_width (geometry.h)
    /// Width, or `0` for an invalid irect. Computed with wrapping unsigned
    /// subtraction and an overflow guard, exactly as MuPDF does.
    #[inline]
    pub fn width(self) -> u32 {
        if self.x0 >= self.x1 {
            return 0;
        }
        let w = (self.x1 as u32).wrapping_sub(self.x0 as u32);
        if (w as i32) < 0 {
            return 0;
        }
        w
    }

    // MuPDF: fz_irect_height (geometry.h)
    /// Height, or `0` for an invalid irect. (MuPDF's declared return type is
    /// `int`; the value is always non-negative, so we return `u32` to match
    /// [`IRect::width`].)
    #[inline]
    pub fn height(self) -> u32 {
        if self.y0 >= self.y1 {
            return 0;
        }
        let h = (self.y1 as u32).wrapping_sub(self.y0 as u32);
        if (h as i32) < 0 {
            return 0;
        }
        h
    }

    // MuPDF: fz_intersect_irect (geometry.c:459)
    pub fn intersect(self, b: IRect) -> IRect {
        let mut a = self;
        if b.is_infinite() {
            return a;
        }
        if a.is_infinite() {
            return b;
        }
        if a.x0 < b.x0 {
            a.x0 = b.x0;
        }
        if a.y0 < b.y0 {
            a.y0 = b.y0;
        }
        if a.x1 > b.x1 {
            a.x1 = b.x1;
        }
        if a.y1 > b.y1 {
            a.y1 = b.y1;
        }
        a
    }

    // MuPDF: fz_translate_irect (geometry.c:505)
    /// Translate by `(xoff, yoff)` with saturating (clamping) integer add, so an
    /// overflow pins to `i32::MIN`/`i32::MAX` rather than wrapping -- this is
    /// MuPDF's `ADD_WITH_SAT`. Empty and infinite irects are returned unchanged.
    pub fn translate(self, xoff: i32, yoff: i32) -> IRect {
        let mut a = self;
        if a.is_empty() {
            return a;
        }
        if a.is_infinite() {
            return a;
        }
        a.x0 = a.x0.saturating_add(xoff);
        a.y0 = a.y0.saturating_add(yoff);
        a.x1 = a.x1.saturating_add(xoff);
        a.y1 = a.y1.saturating_add(yoff);
        a
    }

    // MuPDF: fz_expand_irect (geometry.c:596)
    #[inline]
    pub fn expand(self, expand: i32) -> IRect {
        let mut a = self;
        if a.is_infinite() {
            return a;
        }
        if !a.is_valid() {
            return a;
        }
        a.x0 -= expand;
        a.y0 -= expand;
        a.x1 += expand;
        a.y1 += expand;
        a
    }

    // MuPDF: fz_is_point_inside_irect (geometry.c:789)
    /// Half-open inclusion: `x >= x0 && x < x1 && y >= y0 && y < y1`.
    #[inline]
    pub fn is_point_inside(self, x: i32, y: i32) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    // MuPDF: fz_is_irect_inside_irect (geometry.c:805)
    pub fn is_irect_inside(inner: IRect, outer: IRect) -> bool {
        if !outer.is_valid() {
            return false;
        }
        if !inner.is_valid() {
            return false;
        }
        inner.x0 >= outer.x0 && inner.x1 <= outer.x1 && inner.y0 >= outer.y0 && inner.y1 <= outer.y1
    }
}

// ===========================================================================
// Quad -- fz_quad (four points; edges need not be axis-aligned)
// ===========================================================================

/// A region defined by four corner points -- upper-left, upper-right,
/// lower-left, lower-right. Unlike a [`Rect`], a quad's edges need not be
/// axis-aligned, so it survives a rotation/shear without being widened to its
/// bounding box. Ports `fz_quad`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub ul: Point,
    pub ur: Point,
    pub ll: Point,
    pub lr: Point,
}

impl Quad {
    // MuPDF: fz_infinite_quad (geometry.c:390)
    pub const INFINITE: Quad = Quad {
        ul: Point { x: f32::NEG_INFINITY, y: f32::INFINITY },
        ur: Point { x: f32::INFINITY, y: f32::INFINITY },
        ll: Point { x: f32::NEG_INFINITY, y: f32::NEG_INFINITY },
        lr: Point { x: f32::INFINITY, y: f32::NEG_INFINITY },
    };
    // MuPDF: fz_invalid_quad (geometry.c:391) -- every ordinate NaN.
    pub const INVALID: Quad = Quad {
        ul: Point { x: f32::NAN, y: f32::NAN },
        ur: Point { x: f32::NAN, y: f32::NAN },
        ll: Point { x: f32::NAN, y: f32::NAN },
        lr: Point { x: f32::NAN, y: f32::NAN },
    };

    // MuPDF: fz_make_quad (geometry.h)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        ul_x: f32,
        ul_y: f32,
        ur_x: f32,
        ur_y: f32,
        ll_x: f32,
        ll_y: f32,
        lr_x: f32,
        lr_y: f32,
    ) -> Self {
        Quad {
            ul: Point { x: ul_x, y: ul_y },
            ur: Point { x: ur_x, y: ur_y },
            ll: Point { x: ll_x, y: ll_y },
            lr: Point { x: lr_x, y: lr_y },
        }
    }

    // MuPDF: fz_is_valid_quad (geometry.c:664)
    /// A quad is valid iff none of its eight ordinates is NaN.
    pub fn is_valid(self) -> bool {
        let q = self;
        !q.ll.x.is_nan()
            && !q.ll.y.is_nan()
            && !q.ul.x.is_nan()
            && !q.ul.y.is_nan()
            && !q.lr.x.is_nan()
            && !q.lr.y.is_nan()
            && !q.ur.x.is_nan()
            && !q.ur.y.is_nan()
    }

    // MuPDF: fz_is_infinite_quad (geometry.c:677)
    /// All eight ordinates must be infinite *and* arranged as one of the eight
    /// valid orderings of the corner-signs `(-,-) (-,+) (+,+) (+,-)`. A quad
    /// whose ordinates are all infinite but mis-signed is **not** infinite --
    /// MuPDF is deliberately strict here, and so are we.
    pub fn is_infinite(self) -> bool {
        let q = self;
        let all_inf = q.ll.x.is_infinite()
            && q.ll.y.is_infinite()
            && q.ul.x.is_infinite()
            && q.ul.y.is_infinite()
            && q.lr.x.is_infinite()
            && q.lr.y.is_infinite()
            && q.ur.x.is_infinite()
            && q.ur.y.is_infinite();
        if !all_inf {
            return false;
        }

        // INF_QUAD_TEST(A,B,C,D): A=(-,-) B=(-,+) C=(+,+) D=(+,-)
        let test = |a: Point, b: Point, c: Point, d: Point| -> bool {
            a.x < 0.0
                && a.y < 0.0
                && b.x < 0.0
                && b.y > 0.0
                && c.x > 0.0
                && c.y > 0.0
                && d.x > 0.0
                && d.y < 0.0
        };

        test(q.ll, q.ul, q.ur, q.lr)
            || test(q.ul, q.ur, q.lr, q.ll)
            || test(q.ur, q.lr, q.ll, q.ul)
            || test(q.lr, q.ll, q.ul, q.ur)
            || test(q.ll, q.lr, q.ur, q.ul)
            || test(q.lr, q.ur, q.ul, q.ll)
            || test(q.ur, q.ul, q.ll, q.lr)
            || test(q.ul, q.ll, q.lr, q.ur)
    }

    // MuPDF: fz_is_empty_quad (geometry.c:712)
    /// Empty == zero signed area (via the shoelace formula). Infinite quads are
    /// never empty; all invalid quads are empty.
    pub fn is_empty(self) -> bool {
        let q = self;
        if q.is_infinite() {
            return false;
        }
        if !q.is_valid() {
            return true; // All invalid quads are empty.
        }
        let area = q.ll.x * q.lr.y
            + q.lr.x * q.ur.y
            + q.ur.x * q.ul.y
            + q.ul.x * q.ll.y
            - q.lr.x * q.ll.y
            - q.ur.x * q.lr.y
            - q.ul.x * q.ur.y
            - q.ll.x * q.ul.y;
        area == 0.0
    }

    // MuPDF: fz_quad_from_rect (geometry.c:766)
    /// Losslessly widen a rect to a quad. Invalid rect -> [`Quad::INVALID`],
    /// infinite rect -> [`Quad::INFINITE`].
    pub fn from_rect(r: Rect) -> Quad {
        if !r.is_valid() {
            return Quad::INVALID;
        }
        if r.is_infinite() {
            return Quad::INFINITE;
        }
        Quad {
            ul: Point::new(r.x0, r.y0),
            ur: Point::new(r.x1, r.y0),
            ll: Point::new(r.x0, r.y1),
            lr: Point::new(r.x1, r.y1),
        }
    }

    /// The smallest [`Rect`] covering this quad. Sibling of
    /// [`Rect::from_quad`].
    #[inline]
    pub fn to_rect(self) -> Rect {
        Rect::from_quad(self)
    }

    // MuPDF: fz_transform_quad (geometry.c:751)
    /// Transform all four corners by `m`. Invalid and infinite quads are
    /// returned unchanged.
    pub fn transform(self, m: Matrix) -> Quad {
        let mut q = self;
        if !q.is_valid() {
            return q;
        }
        if q.is_infinite() {
            return q;
        }
        q.ul = q.ul.transform(m);
        q.ur = q.ur.transform(m);
        q.ll = q.ll.transform(m);
        q.lr = q.lr.transform(m);
        q
    }

    // MuPDF: fz_is_point_inside_quad (geometry.c:872)
    /// Is `p` inside this quad? Splits the quad into two triangles
    /// (ul-ur-lr and ul-lr-ll) and tests both. Invalid quad -> false, infinite
    /// quad -> true. May break down for non-'well-formed' quads (MuPDF's caveat).
    pub fn is_point_inside(self, p: Point) -> bool {
        let q = self;
        if !q.is_valid() {
            return false;
        }
        if q.is_infinite() {
            return true;
        }
        is_point_inside_triangle(p, q.ul, q.ur, q.lr) || is_point_inside_triangle(p, q.ul, q.lr, q.ll)
    }

    // MuPDF: fz_is_quad_inside_quad (geometry.c:884)
    /// Does `haystack` contain every corner of `needle`?
    pub fn is_quad_inside(needle: Quad, haystack: Quad) -> bool {
        if !needle.is_valid() || !haystack.is_valid() {
            return false;
        }
        if haystack.is_infinite() {
            return true;
        }
        haystack.is_point_inside(needle.ul)
            && haystack.is_point_inside(needle.ur)
            && haystack.is_point_inside(needle.ll)
            && haystack.is_point_inside(needle.lr)
    }

    // MuPDF: fz_is_quad_intersecting_quad (geometry.c:898)
    /// Cheap intersection test: do the quads' bounding rects overlap? (This is
    /// exactly MuPDF's approximation -- it can report intersection for quads
    /// whose bounding boxes touch but whose bodies do not.)
    pub fn is_quad_intersecting(a: Quad, b: Quad) -> bool {
        let ar = Rect::from_quad(a);
        let br = Rect::from_quad(b);
        !ar.intersect(br).is_empty()
    }
}

// MuPDF: cross (geometry.c:817) -- (b-a) x (p-a).
#[inline]
fn cross(a: Point, b: Point, p: Point) -> f32 {
    let bx = b.x - a.x;
    let by = b.y - a.y;
    let px = p.x - a.x;
    let py = p.y - a.y;
    bx * py - by * px
}

// MuPDF: fz_is_point_inside_triangle (geometry.c:827)
fn is_point_inside_triangle(p: Point, a: Point, b: Point, c: Point) -> bool {
    let crossa = cross(a, b, p);
    let crossb = cross(b, c, p);
    let crossc = cross(c, a, p);

    // Degenerate case: all three vertices the same.
    if crossa == 0.0 && crossb == 0.0 && crossc == 0.0 {
        return a.x == p.x && a.y == p.y;
    }

    if crossa >= 0.0 && crossb >= 0.0 && crossc >= 0.0 {
        return true;
    }
    if crossa <= 0.0 && crossb <= 0.0 && crossc <= 0.0 {
        return true;
    }
    false
}

// ===========================================================================
// Tests -- parity with MuPDF geometry semantics.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn identity_is_identity() {
        assert!(Matrix::IDENTITY.is_identity());
        assert_eq!(Matrix::identity(), Matrix::IDENTITY);
        // A point is unchanged by the identity transform.
        let p = Point::new(3.5, -7.25);
        assert_eq!(p.transform(Matrix::IDENTITY), p);
    }

    #[test]
    fn concat_with_identity_is_noop() {
        let m = Matrix::new(2.0, 0.5, -1.0, 3.0, 4.0, 5.0);
        assert_eq!(m.concat(Matrix::IDENTITY), m);
        assert_eq!(Matrix::IDENTITY.concat(m), m);
        // Mul operator agrees with concat.
        assert_eq!(m * Matrix::IDENTITY, m.concat(Matrix::IDENTITY));
    }

    #[test]
    fn concat_is_associative() {
        let a = Matrix::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let b = Matrix::new(0.5, -1.0, 2.0, 0.25, -3.0, 1.0);
        let c = Matrix::new(-1.0, 0.0, 0.0, -1.0, 7.0, -2.0);
        let left = a.concat(b).concat(c);
        let right = a.concat(b.concat(c));
        for (x, y) in [
            (left.a, right.a),
            (left.b, right.b),
            (left.c, right.c),
            (left.d, right.d),
            (left.e, right.e),
            (left.f, right.f),
        ] {
            assert!(approx(x, y, 1e-3), "{x} vs {y}");
        }
    }

    #[test]
    fn concat_order_matters_matches_point_transform() {
        // (p * a) * b == p * (a.concat(b))
        let a = Matrix::scale(2.0, 3.0);
        let b = Matrix::translate(10.0, -5.0);
        let p = Point::new(4.0, 6.0);
        let via_steps = p.transform(a).transform(b);
        let via_concat = p.transform(a.concat(b));
        assert!(approx(via_steps.x, via_concat.x, 1e-4));
        assert!(approx(via_steps.y, via_concat.y, 1e-4));
    }

    #[test]
    fn scale_transform_point() {
        // Hand-verified: fz_scale(2,3) maps (5,7) -> (10, 21).
        let m = Matrix::scale(2.0, 3.0);
        let p = Point::new(5.0, 7.0).transform(m);
        assert_eq!(p, Point::new(10.0, 21.0));
    }

    #[test]
    fn translate_transform_point() {
        // Hand-verified: fz_translate(3,-4) maps (5,7) -> (8, 3).
        let m = Matrix::translate(3.0, -4.0);
        let p = Point::new(5.0, 7.0).transform(m);
        assert_eq!(p, Point::new(8.0, 3.0));
    }

    #[test]
    fn pre_scale_composition() {
        // pre_scale(sx,sy) == scale(sx,sy).concat(m)
        let m = Matrix::translate(10.0, 20.0);
        assert_eq!(m.pre_scale(2.0, 3.0), Matrix::scale(2.0, 3.0).concat(m));
    }

    #[test]
    fn pre_translate_composition() {
        let m = Matrix::scale(2.0, 4.0);
        assert_eq!(m.pre_translate(3.0, 5.0), Matrix::translate(3.0, 5.0).concat(m));
    }

    #[test]
    fn pre_rotate_composition() {
        let m = Matrix::scale(2.0, 2.0);
        let via_pre = m.pre_rotate(90.0);
        let via_concat = Matrix::rotate(90.0).concat(m);
        for (x, y) in [
            (via_pre.a, via_concat.a),
            (via_pre.b, via_concat.b),
            (via_pre.c, via_concat.c),
            (via_pre.d, via_concat.d),
        ] {
            assert!(approx(x, y, 1e-4), "{x} vs {y}");
        }
    }

    #[test]
    fn rotate_90_is_exact() {
        // The 0/90/180/270 special cases must be exact (no sinf round-off).
        let m = Matrix::rotate(90.0);
        assert_eq!(m, Matrix::new(0.0, 1.0, -1.0, 0.0, 0.0, 0.0));
        // (1,0) rotates 90 CCW to (0,1).
        let p = Point::new(1.0, 0.0).transform(m);
        assert_eq!(p, Point::new(0.0, 1.0));
    }

    #[test]
    fn rotate_negative_and_wraparound() {
        // -90 wraps to 270; 450 wraps to 90.
        assert_eq!(Matrix::rotate(-90.0), Matrix::new(0.0, -1.0, 1.0, 0.0, 0.0, 0.0));
        assert_eq!(Matrix::rotate(450.0), Matrix::rotate(90.0));
    }

    #[test]
    fn invert_round_trip() {
        let m = Matrix::new(2.0, 1.0, 1.0, 3.0, 4.0, 5.0);
        let inv = m.invert();
        let back = m.concat(inv);
        assert!(approx(back.a, 1.0, 1e-5));
        assert!(approx(back.b, 0.0, 1e-5));
        assert!(approx(back.c, 0.0, 1e-5));
        assert!(approx(back.d, 1.0, 1e-5));
        assert!(approx(back.e, 0.0, 1e-4));
        assert!(approx(back.f, 0.0, 1e-4));
    }

    #[test]
    fn invert_degenerate_returns_zero_matrix() {
        // Singular matrix (det = 0): MuPDF returns the all-zero matrix.
        let singular = Matrix::new(1.0, 2.0, 2.0, 4.0, 0.0, 0.0);
        assert_eq!(singular.invert(), Matrix::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        assert!(singular.try_invert().is_none());
    }

    #[test]
    fn try_invert_matches_invert_when_invertible() {
        let m = Matrix::new(2.0, 1.0, 1.0, 3.0, 4.0, 5.0);
        assert_eq!(m.try_invert(), Some(m.invert()));
    }

    #[test]
    fn is_rectilinear_cases() {
        assert!(Matrix::scale(2.0, 3.0).is_rectilinear());
        assert!(Matrix::rotate(90.0).is_rectilinear());
        assert!(Matrix::rotate(180.0).is_rectilinear());
        assert!(!Matrix::rotate(45.0).is_rectilinear());
        assert!(!Matrix::shear(1.0, 0.0).is_rectilinear());
    }

    #[test]
    fn matrix_expansion() {
        // sqrt(|a*d - b*c|) for scale(2,8) = sqrt(16) = 4.
        assert!(approx(Matrix::scale(2.0, 8.0).expansion(), 4.0, 1e-5));
        // max(|a|,|b|,|c|,|d|).
        assert!(approx(Matrix::new(-3.0, 1.0, 2.0, -5.0, 0.0, 0.0).max_expansion(), 5.0, 1e-5));
    }

    #[test]
    fn rect_empty_infinite_valid() {
        assert!(Rect::EMPTY.is_empty());
        assert!(!Rect::EMPTY.is_valid());
        assert!(Rect::INFINITE.is_infinite());
        assert!(Rect::INFINITE.is_valid());
        assert!(Rect::UNIT.is_valid());
        assert!(!Rect::UNIT.is_empty());
        assert!(!Rect::INVALID.is_valid());
        assert!(Rect::INVALID.is_empty());
        // A zero-area rect is valid but empty.
        let line = Rect::new(1.0, 1.0, 1.0, 5.0);
        assert!(line.is_valid());
        assert!(line.is_empty());
    }

    #[test]
    fn rect_transform_all_corners() {
        // Rotating a rect 90 degrees is rectilinear, but exercise the general
        // path with a 45-degree rotation of the unit rect. All four corners
        // must be considered for the bounding box.
        let m = Matrix::rotate(45.0);
        let r = Rect::new(0.0, 0.0, 1.0, 1.0).transform(m);
        let s = (0.5f32).sqrt(); // sin/cos 45
        // Corners map to (0,0), (s,-s)?? Let's just check the bbox extents.
        // x ranges from min to max of {0, cos, -sin, cos-sin} etc.
        assert!(approx(r.x0, -s, 1e-4), "x0={}", r.x0);
        assert!(approx(r.x1, s, 1e-4), "x1={}", r.x1);
        assert!(approx(r.y0, 0.0, 1e-4), "y0={}", r.y0);
        assert!(approx(r.y1, 2.0 * s, 1e-4), "y1={}", r.y1);
    }

    #[test]
    fn rect_transform_rectilinear_negative_scale() {
        // Negative scale must swap corners so the result stays valid.
        let m = Matrix::scale(-1.0, 1.0);
        let r = Rect::new(2.0, 3.0, 5.0, 7.0).transform(m);
        assert!(r.is_valid());
        assert_eq!(r, Rect::new(-5.0, 3.0, -2.0, 7.0));
    }

    #[test]
    fn rect_transform_infinite_unchanged() {
        let m = Matrix::scale(2.0, 2.0);
        assert_eq!(Rect::INFINITE.transform(m), Rect::INFINITE);
    }

    #[test]
    fn rect_intersect() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 15.0, 15.0);
        assert_eq!(a.intersect(b), Rect::new(5.0, 5.0, 10.0, 10.0));
        // Infinite operand => the other rect.
        assert_eq!(a.intersect(Rect::INFINITE), a);
        assert_eq!(Rect::INFINITE.intersect(a), a);
        // Disjoint => empty result.
        let c = Rect::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersect(c).is_empty());
    }

    #[test]
    fn rect_union() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 15.0, 15.0);
        assert_eq!(a.union(b), Rect::new(0.0, 0.0, 15.0, 15.0));
        // Empty/invalid operand drops out.
        assert_eq!(a.union(Rect::EMPTY), a);
        assert_eq!(Rect::EMPTY.union(a), a);
        // Infinite operand swallows.
        assert_eq!(a.union(Rect::INFINITE), Rect::INFINITE);
    }

    #[test]
    fn rect_overlaps_and_contains() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 8.0, 8.0);
        assert!(a.contains(b));
        assert!(!b.contains(a));
        assert!(a.overlaps(b));
        let c = Rect::new(50.0, 50.0, 60.0, 60.0);
        assert!(!a.overlaps(c));
        // Invalid rect contains nothing; every valid rect contains invalid rects.
        assert!(!Rect::INVALID.contains(a));
        assert!(a.contains(Rect::INVALID));
    }

    #[test]
    fn point_in_rect_half_open() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!(r.is_point_inside(Point::new(0.0, 0.0))); // bottom-left included
        assert!(r.is_point_inside(Point::new(5.0, 5.0)));
        assert!(!r.is_point_inside(Point::new(10.0, 5.0))); // right edge excluded
        assert!(!r.is_point_inside(Point::new(5.0, 10.0))); // top edge excluded
    }

    #[test]
    fn include_point_builds_bbox() {
        // Start from an empty rect and fold in points -> tight bbox.
        let mut r = Rect::EMPTY;
        for p in [Point::new(3.0, 4.0), Point::new(-1.0, 8.0), Point::new(5.0, -2.0)] {
            r = r.include_point(p);
        }
        assert_eq!(r, Rect::new(-1.0, -2.0, 5.0, 8.0));
    }

    #[test]
    fn rect_area() {
        assert_eq!(Rect::new(0.0, 0.0, 3.0, 4.0).area(), 12.0);
        assert_eq!(Rect::EMPTY.area(), 0.0);
        assert_eq!(Rect::INFINITE.area(), f32::INFINITY);
    }

    #[test]
    fn irect_from_rect_rounds_out() {
        let r = Rect::new(0.2, 0.8, 3.1, 4.9);
        // floor(x0)/floor(y0), ceil(x1)/ceil(y1).
        assert_eq!(r.irect_from_rect(), IRect::new(0, 0, 4, 5));
    }

    #[test]
    fn round_rect_tolerant() {
        // 0.0009 slop must not bump a pixel outward.
        let r = Rect::new(1.0009, 2.0009, 2.9991, 3.9991);
        assert_eq!(r.round(), IRect::new(1, 2, 3, 4));
    }

    #[test]
    fn rect_irect_round_trip() {
        let ir = IRect::new(1, 2, 3, 4);
        assert_eq!(Rect::from_irect(ir), Rect::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(Rect::from_irect(IRect::INFINITE), Rect::INFINITE);
    }

    #[test]
    fn irect_width_height_and_translate() {
        let r = IRect::new(2, 3, 7, 10);
        assert_eq!(r.width(), 5);
        assert_eq!(r.height(), 7);
        assert_eq!(r.translate(10, 20), IRect::new(12, 23, 17, 30));
        // Saturating add: pushing past i32::MAX clamps, not wraps.
        let big = IRect::new(0, 0, i32::MAX - 1, 10);
        let t = big.translate(100, 0);
        assert_eq!(t.x1, i32::MAX);
    }

    #[test]
    fn quad_from_rect_and_back() {
        let r = Rect::new(1.0, 2.0, 5.0, 8.0);
        let q = Quad::from_rect(r);
        assert_eq!(q.ul, Point::new(1.0, 2.0));
        assert_eq!(q.ur, Point::new(5.0, 2.0));
        assert_eq!(q.ll, Point::new(1.0, 8.0));
        assert_eq!(q.lr, Point::new(5.0, 8.0));
        // Round-trips back to the same rect.
        assert_eq!(q.to_rect(), r);
        assert_eq!(Rect::from_quad(q), r);
    }

    #[test]
    fn quad_from_invalid_and_infinite_rect() {
        // An invalid rect gives the all-NaN invalid quad. Can't use == here:
        // NaN != NaN by IEEE (and by derived PartialEq), which is precisely the
        // correct behaviour -- so test via is_valid() instead.
        let q = Quad::from_rect(Rect::INVALID);
        assert!(!q.is_valid());
        assert!(q.ul.x.is_nan() && q.lr.y.is_nan());
        assert_eq!(Quad::from_rect(Rect::INFINITE), Quad::INFINITE);
    }

    #[test]
    fn quad_validity() {
        assert!(Quad::from_rect(Rect::UNIT).is_valid());
        assert!(!Quad::INVALID.is_valid());
        assert!(Quad::INFINITE.is_infinite());
        assert!(Quad::INVALID.is_empty());
        assert!(!Quad::INFINITE.is_empty());
        // A degenerate (zero-area) quad is empty.
        let flat = Quad::new(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0);
        assert!(flat.is_empty());
    }

    #[test]
    fn quad_transform_rotates_corners() {
        // Transform the unit-rect quad by a 90-degree rotation; corners move but
        // the quad stays a (rotated) unit square, NOT widened to a bbox.
        let q = Quad::from_rect(Rect::UNIT).transform(Matrix::rotate(90.0));
        assert!(approx(q.ul.x, 0.0, 1e-4) && approx(q.ul.y, 0.0, 1e-4));
        assert!(approx(q.ur.x, 0.0, 1e-4) && approx(q.ur.y, 1.0, 1e-4));
        assert!(approx(q.ll.x, -1.0, 1e-4) && approx(q.ll.y, 0.0, 1e-4));
        assert!(approx(q.lr.x, -1.0, 1e-4) && approx(q.lr.y, 1.0, 1e-4));
    }

    #[test]
    fn point_inside_quad() {
        let q = Quad::from_rect(Rect::new(0.0, 0.0, 10.0, 10.0));
        assert!(q.is_point_inside(Point::new(5.0, 5.0)));
        assert!(!q.is_point_inside(Point::new(15.0, 5.0)));
        // Infinite quad contains everything.
        assert!(Quad::INFINITE.is_point_inside(Point::new(1e9, -1e9)));
        // Invalid quad contains nothing.
        assert!(!Quad::INVALID.is_point_inside(Point::new(0.0, 0.0)));
    }

    #[test]
    fn quad_inside_and_intersecting() {
        let outer = Quad::from_rect(Rect::new(0.0, 0.0, 10.0, 10.0));
        let inner = Quad::from_rect(Rect::new(2.0, 2.0, 8.0, 8.0));
        assert!(Quad::is_quad_inside(inner, outer));
        assert!(!Quad::is_quad_inside(outer, inner));
        assert!(Quad::is_quad_intersecting(outer, inner));
        let far = Quad::from_rect(Rect::new(50.0, 50.0, 60.0, 60.0));
        assert!(!Quad::is_quad_intersecting(outer, far));
    }

    #[test]
    fn normalize_vector() {
        let v = Point::new(3.0, 4.0).normalize();
        assert!(approx(v.x, 0.6, 1e-6));
        assert!(approx(v.y, 0.8, 1e-6));
        // Zero vector stays zero (no divide-by-zero).
        assert_eq!(Point::new(0.0, 0.0).normalize(), Point::new(0.0, 0.0));
    }

    #[test]
    fn transform_vector_ignores_translation() {
        let m = Matrix::translate(100.0, 200.0).pre_scale(2.0, 3.0);
        let v = Point::new(1.0, 1.0).transform_vector(m);
        // Translation dropped: only the linear part (2,3) applies.
        assert_eq!(v, Point::new(2.0, 3.0));
    }

    #[test]
    fn transform_page_snaps_origin() {
        // A 100x200pt page at 72dpi with no rotation -> pixel box origin at 0,0.
        let mediabox = Rect::new(0.0, 0.0, 100.0, 200.0);
        let m = Matrix::transform_page(mediabox, 72.0, 0.0);
        let pixel = mediabox.transform(m);
        assert!(approx(pixel.x0, 0.0, 1e-3));
        assert!(approx(pixel.y0, 0.0, 1e-3));
        assert!(approx(pixel.x1, 100.0, 1e-3));
        assert!(approx(pixel.y1, 200.0, 1e-3));
    }
}
