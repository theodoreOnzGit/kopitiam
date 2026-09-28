//! Ported from MuPDF `source/fitz/draw-scale-simple.c` (`fz_scale_pixmap`, the
//! weight tables, the row scalers) + the box subsamplers of
//! `source/fitz/pixmap.c` (`fz_subsample_pixblock`,
//! `fz_subsample_pixblock_bresenham`, `fz_subsample_pixmap`) (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithms and numeric behaviour
//! follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Why this exists (0.4.2)
//!
//! Before 0.4.2 the draw device painted every image by **nearest-neighbour**
//! sampling at device-pixel centres. For a 300-dpi 1-bit scan drawn at 72 dpi
//! that throws away 15 of every 16 source pixels, so thin strokes break up and
//! the page looks bolder and ragged (bd-6lx; WASH-1400 failed the code-to-code
//! raster gate on 194 of 228 pages). MuPDF never does that: `fz_draw_fill_image`
//! first box-subsamples by a power of two (`l2factor`), then smooth-scales to
//! the exact device size with the "simple" filter below, and only then paints
//! 1:1. This module is those two stages.
//!
//! # Fixed-point conventions (copied, not "improved")
//!
//! Weights are integers summing to 256 per fully covered output pixel
//! (`check_weights` forces it); accumulators start at 128 and are shifted
//! `>> 8`, i.e. round-half-up. Every `f32` expression keeps MuPDF's `float`
//! evaluation order, because the weight table is where a last-ULP difference
//! turns into a whole-pixel difference.
//!
//! What is deliberately NOT here: the ARM assembler row scalers (same results
//! as the generic C), the per-device weight caches (`fz_scale_cache` -- a speed
//! optimisation, no effect on output), and `draw-scale.c`'s other filters
//! (MuPDF's draw device only ever uses `fz_scale_filter_simple`).

use super::geometry::{IRect, Matrix};

/// A plain 8-bit pixmap as the scaler sees it: `n` components per pixel
/// INCLUDING alpha when `alpha` (premultiplied, like every MuPDF pixmap), rows
/// tightly packed (`stride == w * n`). `x`/`y` is its device-space origin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalePix {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub n: i32,
    pub alpha: bool,
    pub samples: Vec<u8>,
}

impl ScalePix {
    fn stride(&self) -> usize {
        (self.w * self.n) as usize
    }
}

// ---------------------------------------------------------------------------
// fz_gridfit_matrix (draw-affine.c:3727)
// ---------------------------------------------------------------------------

/// `MY_EPSILON` in draw-affine.c: how far past a pixel boundary an edge may
/// sit before grid-fitting pushes it to the next boundary.
const MY_EPSILON: f32 = 0.001;

/// C's `(float)(int)x` -- truncation toward zero, through a 32-bit int.
fn trunc_i(x: f32) -> f32 {
    (x as i32) as f32
}

// MuPDF: fz_gridfit_matrix (draw-affine.c:3727)
/// Snap an image matrix so the image covers whole device pixels: the edges
/// move OUTWARD onto pixel boundaries (or to the nearest boundary when
/// `as_tiled`). Only rectilinear matrices change; anything rotated off-axis
/// comes back untouched.
pub fn gridfit_matrix(as_tiled: bool, mut m: Matrix) -> Matrix {
    if m.b.abs() < f32::EPSILON && m.c.abs() < f32::EPSILON {
        if as_tiled {
            let f = trunc_i(m.e + 0.5);
            m.a += m.e - f;
            m.e = f;
            m.a = trunc_i(m.a + 0.5);
        } else if m.a > 0.0 {
            let mut f = trunc_i(m.e);
            if f - m.e > MY_EPSILON {
                f -= 1.0;
            }
            m.a += m.e - f;
            m.e = f;
            let mut f = trunc_i(m.a);
            if m.a - f > MY_EPSILON {
                f += 1.0;
            }
            m.a = f;
        } else if m.a < 0.0 {
            let mut f = trunc_i(m.e);
            if m.e - f > MY_EPSILON {
                f += 1.0;
            }
            m.a += m.e - f;
            m.e = f;
            let mut f = trunc_i(m.a);
            if f - m.a > MY_EPSILON {
                f -= 1.0;
            }
            m.a = f;
        }
        if as_tiled {
            let f = trunc_i(m.f + 0.5);
            m.d += m.f - f;
            m.f = f;
            m.d = trunc_i(m.d + 0.5);
        } else if m.d > 0.0 {
            let mut f = trunc_i(m.f);
            if f - m.f > MY_EPSILON {
                f -= 1.0;
            }
            m.d += m.f - f;
            m.f = f;
            let mut f = trunc_i(m.d);
            if m.d - f > MY_EPSILON {
                f += 1.0;
            }
            m.d = f;
        } else if m.d < 0.0 {
            let mut f = trunc_i(m.f);
            if m.f - f > MY_EPSILON {
                f += 1.0;
            }
            m.d += m.f - f;
            m.f = f;
            let mut f = trunc_i(m.d);
            if f - m.d > MY_EPSILON {
                f -= 1.0;
            }
            m.d = f;
        }
    } else if m.a.abs() < f32::EPSILON && m.d.abs() < f32::EPSILON {
        if as_tiled {
            let f = trunc_i(m.e + 0.5);
            m.b += m.e - f;
            m.e = f;
            m.b = trunc_i(m.b + 0.5);
        } else if m.b > 0.0 {
            let mut f = trunc_i(m.f);
            if f - m.f > MY_EPSILON {
                f -= 1.0;
            }
            m.b += m.f - f;
            m.f = f;
            let mut f = trunc_i(m.b);
            if m.b - f > MY_EPSILON {
                f += 1.0;
            }
            m.b = f;
        } else if m.b < 0.0 {
            let mut f = trunc_i(m.f);
            if m.f - f > MY_EPSILON {
                f += 1.0;
            }
            m.b += m.f - f;
            m.f = f;
            let mut f = trunc_i(m.b);
            if f - m.b > MY_EPSILON {
                f -= 1.0;
            }
            m.b = f;
        }
        if as_tiled {
            let f = trunc_i(m.f + 0.5);
            m.c += m.f - f;
            m.f = f;
            m.c = trunc_i(m.c + 0.5);
        } else if m.c > 0.0 {
            let mut f = trunc_i(m.e);
            if f - m.e > MY_EPSILON {
                f -= 1.0;
            }
            m.c += m.e - f;
            m.e = f;
            let mut f = trunc_i(m.c);
            if m.c - f > MY_EPSILON {
                f += 1.0;
            }
            m.c = f;
        } else if m.c < 0.0 {
            let mut f = trunc_i(m.e);
            if m.e - f > MY_EPSILON {
                f += 1.0;
            }
            m.c += m.e - f;
            m.e = f;
            let mut f = trunc_i(m.c);
            if f - m.c > MY_EPSILON {
                f -= 1.0;
            }
            m.c = f;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Power-of-two box subsampling (pixmap.c)
// ---------------------------------------------------------------------------

// MuPDF: fz_subsample_pixblock (pixmap.c:1822), the portable (non-ARM) arm.
/// Average `2^factor x 2^factor` blocks in place; a partial block on the right
/// or bottom edge averages only the pixels it has. Returns the new size.
pub fn subsample_pixblock(s: &mut [u8], w: i32, h: i32, n: i32, factor: i32) -> (i32, i32) {
    let f = 1i32 << factor;
    let stride = (w * n) as usize;
    let nw = (w + f - 1) >> factor;
    let nh = (h + f - 1) >> factor;
    let mut out = Vec::with_capacity((nw * nh * n) as usize);
    let mut y0 = 0;
    while y0 < h {
        let bh = f.min(h - y0);
        let mut x0 = 0;
        while x0 < w {
            let bw = f.min(w - x0);
            for c in 0..n {
                let mut v: i32 = 0;
                for xx in 0..bw {
                    for yy in 0..bh {
                        v += s[(y0 + yy) as usize * stride + ((x0 + xx) * n + c) as usize] as i32;
                    }
                }
                // Full blocks shift by 2*factor; strays divide by their count
                // (`x * f`, `y * f`, `x * y` in the C).
                let o = if bw == f && bh == f { v >> (2 * factor) } else { v / (bw * bh) };
                out.push(o as u8);
            }
            x0 += f;
        }
        y0 += f;
    }
    s[..out.len()].copy_from_slice(&out);
    (nw, nh)
}

// MuPDF: fz_subsample_pixblock_bresenham (pixmap.c:1707)
/// The "smarter" subsample for sizes that are not a multiple of `2^factor`:
/// every output pixel averages a full `f x f` block, and `subx`/`suby` blocks
/// re-use a line so the stray pixels spread evenly instead of piling up on the
/// last row/column.
fn subsample_pixblock_bresenham(s: &mut [u8], w: i32, h: i32, n: i32, factor: i32, subx: i32, suby: i32) {
    let f = 1i32 << factor;
    let stride = (w * n) as isize;
    let bxd = ((w + subx) / f) - 1;
    let mut bxf = (bxd + 1) >> 1;
    let byd = ((h + suby) / f) - 1;
    let mut byf = (byd + 1) >> 1;
    bxf = bxd - bxf;
    byf = byd - byf;
    let fwd = stride;
    let back = f as isize * fwd - n as isize;
    let back2 = (f * n - 1) as isize;
    let fwd2 = ((f - 1) * n) as isize;
    let sh2 = 2 * factor;
    let mut d: usize = 0;
    let mut s2: isize = 0;
    let mut y = h;
    while y > 0 {
        let mut bxf2 = bxf;
        let mut sp: isize = s2;
        let mut x = w;
        while x > 0 {
            for _ in 0..n {
                let mut v: i32 = 0;
                for _ in 0..f {
                    for _ in 0..f {
                        v += s[sp as usize] as i32;
                        sp += fwd;
                    }
                    sp -= back;
                }
                s[d] = (v >> sh2) as u8;
                d += 1;
                sp -= back2;
            }
            sp += fwd2;
            bxf2 -= subx;
            while bxf2 < 0 {
                sp -= n as isize;
                bxf2 += bxd;
            }
            x -= f;
        }
        s2 += stride * f as isize;
        byf -= suby;
        while byf < 0 {
            s2 -= stride;
            byf += byd;
        }
        y -= f;
    }
}

// MuPDF: fz_subsample_pixmap (pixmap.c:1772)
/// Subsample `pix` in place by `2^factor`, picking the Bresenham variant when
/// the size is not a multiple of the block (and the image is at least one
/// block big), exactly as MuPDF does.
pub fn subsample_pixmap(pix: &mut ScalePix, factor: i32) {
    if factor <= 0 {
        return;
    }
    let f = 1i32 << factor;
    let mut subx = pix.w % f;
    let mut suby = pix.h % f;
    if (subx != 0 || suby != 0) && pix.w >= f && pix.h >= f {
        subx = if subx != 0 { f - subx } else { 0 };
        suby = if suby != 0 { f - suby } else { 0 };
        subsample_pixblock_bresenham(&mut pix.samples, pix.w, pix.h, pix.n, factor, subx, suby);
    } else {
        subsample_pixblock(&mut pix.samples, pix.w, pix.h, pix.n, factor);
    }
    pix.w = (pix.w + f - 1) >> factor;
    pix.h = (pix.h + f - 1) >> factor;
    pix.samples.truncate((pix.w * pix.h * pix.n) as usize);
}

// ---------------------------------------------------------------------------
// The weight tables (draw-scale-simple.c:203-548)
// ---------------------------------------------------------------------------

// MuPDF: simple (draw-scale-simple.c:110) -- fz_scale_filter_simple, width 1.
fn filter_simple(x: f32) -> f32 {
    if x >= 1.0 {
        return 0.0;
    }
    1.0 + (2.0 * x - 3.0) * x * x
}

/// The filter's support half-width (`fz_scale_filter_simple.width`).
const FILTER_WIDTH: i32 = 1;

// MuPDF: fz_weights (draw-scale-simple.c:203)
struct Weights {
    flip: bool,
    count: i32,
    max_len: i32,
    n: i32,
    new_line: bool,
    patch_l: i32,
    index: Vec<i32>,
}

impl Weights {
    // MuPDF: new_weights (draw-scale-simple.c:230)
    fn new(src_w: i32, dst_w: f32, patch_w: i32, n: i32, flip: bool, patch_l: i32) -> Weights {
        let max_len = if (src_w as f32) > dst_w {
            let m = ((2 * FILTER_WIDTH * src_w) as f32 / dst_w).ceil() as i32;
            m.min(src_w)
        } else {
            2 * FILTER_WIDTH
        };
        let mut index = vec![0i32; ((max_len + 3) * (patch_w + 1)) as usize];
        index[0] = patch_w;
        Weights { flip, count: -1, max_len, n, new_line: false, patch_l, index }
    }

    // MuPDF: init_weights (draw-scale-simple.c:270)
    fn init(&mut self, j: i32) {
        let j = j - self.patch_l;
        self.count += 1;
        self.new_line = true;
        let index = if j == 0 {
            self.index[0]
        } else {
            let prev = self.index[(j - 1) as usize];
            prev + 2 + self.index[(prev + 1) as usize]
        };
        self.index[j as usize] = index;
        self.index[index as usize] = 0;
        self.index[(index + 1) as usize] = 0;
    }

    // MuPDF: insert_weight (draw-scale-simple.c:291)
    fn insert(&mut self, j: i32, i: i32, weight: i32) {
        let j = (j - self.patch_l) as usize;
        if self.new_line {
            self.new_line = false;
            let index = self.index[j] as usize;
            self.index[index] = i;
            self.index[index + 1] = 0;
        }
        let mut index = self.index[j] as usize;
        let mut min = self.index[index];
        index += 1;
        let mut len = self.index[index];
        index += 1;
        while i < min {
            let mut k = len;
            while k > 0 {
                self.index[index + k as usize] = self.index[index + k as usize - 1];
                k -= 1;
            }
            self.index[index] = 0;
            min -= 1;
            len += 1;
            self.index[index - 2] = min;
            self.index[index - 1] = len;
        }
        if i - min >= len {
            loop {
                len += 1;
                if i - min < len {
                    break;
                }
                self.index[index + len as usize - 1] = 0;
            }
            self.index[index + (i - min) as usize] = weight;
            self.index[index - 1] = len;
        } else {
            self.index[index + (i - min) as usize] += weight;
        }
    }

    // MuPDF: add_weight (draw-scale-simple.c:345)
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, j: i32, i: i32, x: f32, f_: f32, g: f32, src_w: i32, dst_w: f32) {
        let mut dist = j as f32 - x + 0.5 - ((i as f32 + 0.5) * dst_w / src_w as f32);
        dist *= g;
        if dist < 0.0 {
            dist = -dist;
        }
        let f = filter_simple(dist) * f_;
        let weight = (256.0 * f + 0.5) as i32;
        if i < 0 || i >= src_w {
            return;
        }
        if weight != 0 {
            self.insert(j, i, weight);
        }
    }

    // MuPDF: reorder_weights (draw-scale-simple.c:366)
    fn reorder(&mut self, j: i32, src_w: i32) {
        let mut idx = self.index[(j - self.patch_l) as usize] as usize;
        let mut min = self.index[idx];
        idx += 1;
        let mut len = self.index[idx];
        idx += 1;
        let max = self.max_len;
        let tmp: Vec<i32> = {
            let mut t = self.index[idx..idx + len as usize].to_vec();
            t.resize(max as usize, 0);
            t
        };
        let mut off = 0;
        if len < max {
            len = max;
            if min + len > src_w {
                off = min + len - src_w;
                min = src_w - len;
                self.index[idx - 2] = min;
            }
            self.index[idx - 1] = len;
        }
        for (i, &t) in tmp.iter().enumerate().take(len as usize) {
            let at = ((min + i as i32 + off) % max) as usize;
            self.index[idx + at] = t;
        }
    }

    // MuPDF: check_weights (draw-scale-simple.c:407)
    fn check(&mut self, j: i32, w: i32, x: f32, wf: f32) {
        let mut idx = self.index[(j - self.patch_l) as usize] as usize;
        idx += 1; // min
        let len = self.index[idx];
        idx += 1;
        let mut sum = 0;
        let mut max = -256;
        let mut maxidx = 0usize;
        for _ in 0..len {
            let v = self.index[idx];
            idx += 1;
            sum += v;
            if v > max {
                max = v;
                maxidx = idx;
            }
        }
        if ((j != 0) && (j != w - 1)) || (sum > 256) {
            self.index[maxidx - 1] += 256 - sum;
        } else if (j == 0) && (x < 0.0001) && (sum != 256) {
            self.index[maxidx - 1] += 256 - sum;
        } else if (j == w - 1) && (w as f32 - wf < 0.0001) && (sum != 256) {
            self.index[maxidx - 1] += 256 - sum;
        }
    }
}

// MuPDF: window_fix (draw-scale-simple.c:444)
fn window_fix(mut l: i32, r: &mut i32, window: f32, centre: f32) -> i32 {
    while centre - l as f32 > window {
        l += 1;
    }
    while *r as f32 - centre > window {
        *r -= 1;
    }
    l
}

// MuPDF: make_weights (draw-scale-simple.c:456), without the cache.
#[allow(clippy::too_many_arguments)]
fn make_weights(
    src_w: i32,
    x: f32,
    dst_w: f32,
    vertical: bool,
    dst_w_int: i32,
    patch_l: i32,
    patch_r: i32,
    n: i32,
    flip: bool,
) -> Weights {
    let (f_, g) = if dst_w < src_w as f32 {
        (dst_w / src_w as f32, 1.0f32)
    } else {
        (1.0f32, src_w as f32 / dst_w)
    };
    let window = FILTER_WIDTH as f32 / f_;
    let mut weights = Weights::new(src_w, dst_w, patch_r - patch_l, n, flip, patch_l);
    for j in patch_l..patch_r {
        let centre = (j as f32 - x + 0.5) * src_w as f32 / dst_w - 0.5;
        let mut l = (centre - window).ceil() as i32;
        let mut r = (centre + window).floor() as i32;
        if (r - l) as f32 > 2.0 * window {
            l = window_fix(l, &mut r, window, centre);
        }
        weights.init(j);
        while l <= r {
            weights.add(j, l, x, f_, g, src_w, dst_w);
            l += 1;
        }
        if weights.new_line {
            // Bug 706764: no non-zero weight at all -- use the central pixel.
            let src_x = (centre.floor() as i32).clamp(0, src_w - 1);
            weights.insert(j, src_x, 1);
        }
        weights.check(j, dst_w_int, x, dst_w);
        if vertical {
            weights.reorder(j, src_w);
        }
    }
    weights.count += 1;
    weights
}

// MuPDF: scale_row_to_temp (draw-scale-simple.c:551) -- the generic arm; the
// n = 1..4 specialisations compute the same sums.
fn scale_row_to_temp(dst: &mut [u8], src: &[u8], w: &Weights) {
    let n = w.n as usize;
    let mut ci = w.index[0] as usize;
    let count = w.count as usize;
    for i in 0..count {
        let min = w.index[ci] as usize * n;
        let len = w.index[ci + 1] as usize;
        ci += 2;
        let out = if w.flip { count - 1 - i } else { i };
        for c in 0..n {
            let mut t: i32 = 128;
            for k in 0..len {
                t += src[min + k * n + c] as i32 * w.index[ci + k];
            }
            dst[out * n + c] = (t >> 8) as u8;
        }
        ci += len;
    }
}

// MuPDF: scale_row_from_temp / scale_row_from_temp_alpha (draw-scale-simple.c:1256, 1282)
fn scale_row_from_temp(dst: &mut [u8], temp: &[u8], w: &Weights, width_px: usize, n: usize, row: usize, forcealpha: bool) {
    let mut ci = w.index[row] as usize;
    ci += 1; // skip min
    let len = w.index[ci] as usize;
    ci += 1;
    let width = width_px * n;
    let mut d = 0usize;
    for x in 0..width_px {
        for c in 0..n {
            let col = x * n + c;
            let mut val: i32 = 128;
            for k in 0..len {
                val += temp[k * width + col] as i32 * w.index[ci + k];
            }
            dst[d] = (val >> 8) as u8;
            d += 1;
        }
        if forcealpha {
            dst[d] = 255;
            d += 1;
        }
    }
}

// MuPDF: scale_single_row (draw-scale-simple.c:1340)
fn scale_single_row(out: &mut ScalePix, src: &[u8], w: &Weights, forcealpha: bool) {
    let n = w.n as usize;
    let nf = n + usize::from(forcealpha);
    let count = w.count as usize;
    let mut row = vec![0u8; count * nf];
    let mut ci = w.index[0] as usize;
    for i in 0..count {
        let min = w.index[ci] as usize * n;
        let len = w.index[ci + 1] as usize;
        ci += 2;
        let mut tmp = [128i32; 33];
        for k in 0..len {
            let c = w.index[ci + k];
            for j in 0..n {
                tmp[j] += src[min + k * n + j] as i32 * c;
            }
            if forcealpha {
                tmp[n] += 255 * c;
            }
        }
        ci += len;
        let o = if w.flip { count - 1 - i } else { i };
        for j in 0..nf {
            row[o * nf + j] = (tmp[j] >> 8) as u8;
        }
    }
    for y in 0..out.h as usize {
        let st = out.stride();
        out.samples[y * st..y * st + row.len()].copy_from_slice(&row);
    }
}

// MuPDF: scale_single_col (draw-scale-simple.c:1409)
fn scale_single_col(out: &mut ScalePix, src: &[u8], src_h: i32, n: usize, w: &Weights, forcealpha: bool) {
    let nf = n + usize::from(forcealpha);
    let count = w.count as usize;
    let dw = out.w as usize;
    let st = out.stride();
    let mut ci = w.index[0] as usize;
    for i in 0..count {
        let min = w.index[ci];
        let len = w.index[ci + 1] as usize;
        ci += 2;
        let mut tmp = [128i32; 33];
        for k in 0..len {
            let c = w.index[ci + k];
            let sy = if w.flip { (src_h - 1) - (min + k as i32) } else { min + k as i32 } as usize;
            for j in 0..n {
                tmp[j] += src[sy * n + j] as i32 * c;
            }
            if forcealpha {
                tmp[n] += 255 * c;
            }
        }
        ci += len;
        for x in 0..dw {
            for j in 0..nf {
                out.samples[i * st + x * nf + j] = (tmp[j] >> 8) as u8;
            }
        }
    }
}

// MuPDF: get_alpha_edge_values (draw-scale-simple.c:1485)
fn alpha_edge_values(rows: &Weights) -> (i32, i32) {
    let mut ci = rows.index[0] as usize;
    ci += 1;
    let mut len = rows.index[ci];
    ci += 1;
    let mut t = 0;
    for _ in 0..len {
        t += rows.index[ci];
        ci += 1;
    }
    let mut i = rows.count - 2;
    while i > 0 {
        ci += 1;
        len = rows.index[ci];
        ci += 1;
        ci += len as usize;
        i -= 1;
    }
    let mut b = 0;
    if i == 0 {
        ci += 1;
        len = rows.index[ci];
        ci += 1;
        for _ in 0..len {
            b += rows.index[ci];
            ci += 1;
        }
    }
    if rows.flip && i == 0 { (b, t) } else { (t, b) }
}

// MuPDF: adjust_alpha_edges (draw-scale-simple.c:1523)
fn adjust_alpha_edges(pix: &mut ScalePix, rows: &Weights, cols: &Weights) {
    let (t, b) = alpha_edge_values(rows);
    let (l, r) = alpha_edge_values(cols);
    let w = pix.w;
    let n = pix.n as usize;
    let st = pix.stride();
    let l = (255 * l + 128) >> 8;
    let r = (255 * r + 128) >> 8;
    let tl = (l * t + 128) >> 8;
    let tr = (r * t + 128) >> 8;
    let bl = (l * b + 128) >> 8;
    let br = (r * b + 128) >> 8;
    let t = (255 * t + 128) >> 8;
    let b = (255 * b + 128) >> 8;
    let a = |x: i32, y: i32| y as usize * st + x as usize * n + n - 1;
    let s = &mut pix.samples;
    s[a(0, 0)] = tl as u8;
    for x in 1..w - 1 {
        s[a(x, 0)] = t as u8;
    }
    if w >= 2 {
        s[a(w - 1, 0)] = tr as u8;
    }
    for y in 1..pix.h - 1 {
        if w >= 2 {
            s[a(w - 1, y)] = r as u8;
        }
        s[a(0, y)] = l as u8;
    }
    if pix.h >= 2 {
        let y = pix.h - 1;
        s[a(0, y)] = bl as u8;
        for x in 1..w - 1 {
            s[a(x, y)] = b as u8;
        }
        if w >= 2 {
            s[a(w - 1, y)] = br as u8;
        }
    }
}

// MuPDF: fz_scale_pixmap_cached (draw-scale-simple.c:1586)
/// Smooth-scale `src` so its top-left lands at `(x, y)` with size `w x h`
/// (negative = flipped), cropped to `clip`. Returns the new pixmap with its
/// device origin set, or `None` where MuPDF returns NULL (extreme scale, or
/// nothing left after the clip).
pub fn scale_pixmap(src: &ScalePix, mut x: f32, mut y: f32, mut w: f32, mut h: f32, clip: Option<IRect>) -> Option<ScalePix> {
    const BIG: f32 = (1 << 24) as f32;
    if w > BIG || h > BIG || w < -BIG || h < -BIG {
        return None;
    }
    if x > BIG || y > BIG || x < -BIG || y < -BIG {
        return None;
    }
    if w <= -1.0 {
    } else if w < 0.0 {
        w = -1.0;
    } else if w < 1.0 {
        w = 1.0;
    }
    if h <= -1.0 {
    } else if h < 0.0 {
        h = -1.0;
    } else if h < 1.0 {
        h = 1.0;
    }
    let isint = |v: f32| v == (v as i32) as f32;
    let forcealpha = !src.alpha && (!isint(x) || !isint(y) || !isint(w) || !isint(h));

    let flip_x = w < 0.0;
    let (mut dst_x_int, dst_w_int);
    if flip_x {
        w = -w;
        dst_x_int = (x - w).floor() as i32;
        let tmp = x.ceil();
        let dwi = tmp as i32;
        x = tmp - x;
        dst_w_int = dwi - dst_x_int;
    } else {
        dst_x_int = x.floor() as i32;
        x -= dst_x_int as f32;
        dst_w_int = (x + w).ceil() as i32;
    }
    let flip_y = h < 0.0;
    let (mut dst_y_int, dst_h_int);
    if flip_y {
        h = -h;
        dst_y_int = (y - h).floor() as i32;
        let tmp = y.ceil();
        let dhi = tmp as i32;
        y = tmp - y;
        dst_h_int = dhi - dst_y_int;
    } else {
        dst_y_int = y.floor() as i32;
        y -= dst_y_int as f32;
        dst_h_int = (y + h).ceil() as i32;
    }

    // Step 0: the patch.
    let (mut px0, mut py0, mut px1, mut py1) = (0, 0, dst_w_int, dst_h_int);
    if let Some(clip) = clip {
        if flip_x {
            if dst_x_int + dst_w_int > clip.x1 {
                px0 = dst_x_int + dst_w_int - clip.x1;
            }
            if clip.x0 > dst_x_int {
                px1 = dst_w_int - (clip.x0 - dst_x_int);
                dst_x_int = clip.x0;
            }
        } else {
            if dst_x_int + dst_w_int > clip.x1 {
                px1 = clip.x1 - dst_x_int;
            }
            if clip.x0 > dst_x_int {
                px0 = clip.x0 - dst_x_int;
                dst_x_int += px0;
            }
        }
        if flip_y {
            if dst_y_int + dst_h_int > clip.y1 {
                py1 = clip.y1 - dst_y_int;
            }
            if clip.y0 > dst_y_int {
                py0 = clip.y0 - dst_y_int;
                dst_y_int = clip.y0;
            }
        } else {
            if dst_y_int + dst_h_int > clip.y1 {
                py1 = clip.y1 - dst_y_int;
            }
            if clip.y0 > dst_y_int {
                py0 = clip.y0 - dst_y_int;
                dst_y_int += py0;
            }
        }
    }
    if px0 >= px1 || py0 >= py1 {
        return None;
    }

    // Step 1: weights (SINGLE_PIXEL_SPECIALS: none for a 1-wide/1-high side).
    let cols = (src.w != 1).then(|| make_weights(src.w, x, w, false, dst_w_int, px0, px1, src.n, flip_x));
    let rows = (src.h != 1).then(|| make_weights(src.h, y, h, true, dst_h_int, py0, py1, src.n, flip_y));

    let out_alpha = src.alpha || forcealpha;
    let out_n = src.n + i32::from(forcealpha);
    let ow = px1 - px0;
    let oh = py1 - py0;
    let mut out = ScalePix {
        x: dst_x_int,
        y: dst_y_int,
        w: ow,
        h: oh,
        n: out_n,
        alpha: out_alpha,
        samples: vec![0u8; (ow * oh * out_n) as usize],
    };

    // Step 2: apply.
    match (&rows, &cols) {
        (None, None) => {
            // duplicate_single_pixel
            let mut px: Vec<u8> = src.samples[..src.n as usize].to_vec();
            if forcealpha {
                px.push(255);
            }
            for chunk in out.samples.chunks_mut(px.len()) {
                chunk.copy_from_slice(&px);
            }
        }
        (None, Some(cols)) => scale_single_row(&mut out, &src.samples, cols, forcealpha),
        (Some(rows), None) => scale_single_col(&mut out, &src.samples, src.h, src.n as usize, rows, forcealpha),
        (Some(rows), Some(cols)) => {
            let n = src.n as usize;
            let temp_span = cols.count as usize * n;
            let temp_rows = rows.max_len as usize;
            if temp_span == 0 {
                return Some(out);
            }
            let mut temp = vec![0u8; temp_span * temp_rows];
            let src_stride = src.stride();
            let out_stride = out.stride();
            let mut max_row = rows.index[rows.index[0] as usize];
            for row in 0..rows.count as usize {
                let ri = rows.index[row] as usize;
                let row_min = rows.index[ri];
                let row_len = rows.index[ri + 1];
                while max_row < row_min + row_len {
                    let sy = if flip_y { src.h - 1 - max_row } else { max_row } as usize;
                    let slot = (max_row as usize) % temp_rows;
                    scale_row_to_temp(
                        &mut temp[temp_span * slot..temp_span * (slot + 1)],
                        &src.samples[sy * src_stride..(sy + 1) * src_stride],
                        cols,
                    );
                    max_row += 1;
                }
                scale_row_from_temp(
                    &mut out.samples[row * out_stride..(row + 1) * out_stride],
                    &temp,
                    rows,
                    cols.count as usize,
                    n,
                    row,
                    forcealpha,
                );
            }
            if forcealpha {
                adjust_alpha_edges(&mut out, rows, cols);
            }
        }
    }
    Some(out)
}

// MuPDF: fz_transform_pixmap (draw-device.c:1688)
/// Scale `image` to its device footprint when the image matrix `ctm` is
/// rectilinear (grid-fitting first when `gridfit`), rewriting `ctm` to map the
/// unit square onto the scaled pixmap 1:1. A skewed/rotated matrix only gets
/// the downscale to `dx x dy` (the affine painter does the rest).
pub fn transform_pixmap(
    image: &ScalePix,
    ctm: &mut Matrix,
    dx: i32,
    dy: i32,
    gridfit: bool,
    clip: Option<IRect>,
) -> Option<ScalePix> {
    if let Some(c) = clip
        && c.is_empty()
    {
        return None;
    }
    if ctm.a != 0.0 && ctm.b == 0.0 && ctm.c == 0.0 && ctm.d != 0.0 {
        // Unrotated or X-flip or Y-flip or XY-flip.
        let m = if gridfit { gridfit_matrix(false, *ctm) } else { *ctm };
        let scaled = scale_pixmap(image, m.e, m.f, m.a, m.d, clip)?;
        ctm.a = scaled.w as f32;
        ctm.d = scaled.h as f32;
        ctm.e = scaled.x as f32;
        ctm.f = scaled.y as f32;
        return Some(scaled);
    }
    if ctm.a == 0.0 && ctm.b != 0.0 && ctm.c != 0.0 && ctm.d == 0.0 {
        // Other orthogonal flip/rotation cases.
        let m = if gridfit { gridfit_matrix(false, *ctm) } else { *ctm };
        let rclip = clip.map(|c| IRect::new(c.y0, c.x0, c.y1, c.x1));
        let scaled = scale_pixmap(image, m.f, m.e, m.b, m.c, rclip)?;
        ctm.b = scaled.w as f32;
        ctm.c = scaled.h as f32;
        ctm.f = scaled.x as f32;
        ctm.e = scaled.y as f32;
        return Some(scaled);
    }
    // Downscale, non rectilinear case.
    if dx > 0 && dy > 0 {
        return scale_pixmap(image, 0.0, 0.0, dx as f32, dy as f32, None);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsample_full_blocks_average() {
        // 4x2 gray, factor 1 -> 2x1: (0+255+0+255)/4 = 127, (255*4)/4 = 255.
        let mut s = vec![0, 255, 255, 255, 255, 0, 255, 255];
        let (w, h) = subsample_pixblock(&mut s, 4, 2, 1, 1);
        assert_eq!((w, h), (2, 1));
        assert_eq!(&s[..2], &[127, 255]);
    }

    #[test]
    fn scale_down_by_two_is_a_box_average_for_integral_sizes() {
        // 4x1 -> 2x1: the "simple" filter at F = 0.5 weights the two nearest
        // source pixels 128/128 (MuPDF's table for this case), plus zero
        // tails; interior sums forced to 256.
        let src = ScalePix { x: 0, y: 0, w: 4, h: 2, n: 1, alpha: false, samples: vec![0, 0, 255, 255, 0, 0, 255, 255] };
        let out = scale_pixmap(&src, 0.0, 0.0, 2.0, 1.0, None).expect("scaled");
        assert_eq!((out.w, out.h, out.n), (2, 1, 1));
        assert!(out.samples[0] < 64 && out.samples[1] > 191, "{:?}", out.samples);
    }

    #[test]
    fn gridfit_moves_edges_outward() {
        let m = gridfit_matrix(false, Matrix { a: 10.4, b: 0.0, c: 0.0, d: 5.2, e: 1.3, f: 2.6 });
        assert_eq!((m.e, m.f), (1.0, 2.0));
        assert_eq!((m.a, m.d), (11.0, 6.0));
    }
}
