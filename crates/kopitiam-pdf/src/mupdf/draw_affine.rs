//! Ported from MuPDF `source/fitz/draw-affine.c` (`fz_paint_image_imp` and the
//! `template_affine_*_near` / `_lerp` span painters it dispatches to) (commit
//! 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this paints
//!
//! An image pixmap (already subsampled + smooth-scaled by
//! [`draw_scale`](super::draw_scale) when it was being shrunk) onto the RGB
//! page, through the image matrix, in MuPDF's 14-bit fixed point (`PREC`).
//! Nearest sampling for 1:1 / downscaled rectilinear paints, bilinear (`lerp`)
//! for upscaled or rotated ones -- the same `dolerp` decision MuPDF makes.
//!
//! Only the arms the kopitiam draw device can reach are here: an RGB
//! destination with no alpha plane, no shape / group-alpha planes, no
//! overprint, no spot colours. The `g2rgb` special case is covered by
//! expanding gray to RGB *after* scaling, which is exactly what that painter
//! computes (it writes the same gray value into all three channels).
//!
//! One addition MuPDF does not have in this function: an optional per-pixel
//! device-space `mask`. MuPDF draws an `/SMask`ed image by pushing the mask as
//! a clip (`fz_clip_image_mask`), painting the image into a layer and blending
//! the layer back through the mask (`fz_paint_pixmap_with_mask`). For an
//! opaque layer that composite is `mul255(img, m) + mul255(dst, 255 - m)` per
//! channel -- the same formula as the constant-alpha painter with the alpha
//! replaced by `m` -- so the mask is folded in as a per-pixel alpha here
//! instead of building a layer stack.

use super::draw_scale::ScalePix;
use super::geometry::{IRect, Matrix, Rect};
use super::pixmap::Pixmap;

/// `PREC` / `ONE` / `MASK` / `HALF` / `LIMIT` (draw-affine.c:33-37).
const PREC: i64 = 14;
const ONE: i64 = 1 << PREC;
const MASK: i64 = ONE - 1;
const HALF: i64 = 1 << (PREC - 1);
const LIMIT: i64 = 1 << (63 - PREC);

// MuPDF: fz_mul255 (geometry.h:38)
#[inline]
fn mul255(a: i32, b: i32) -> i32 {
    let mut x = a * b + 128;
    x += x >> 8;
    x >> 8
}

// MuPDF: lerp / bilerp (draw-affine.c:43-50)
#[inline]
fn lerp(a: i32, b: i32, f: i32) -> i32 {
    a + (((b - a) * f) >> PREC)
}
#[inline]
fn bilerp(a: i32, b: i32, c: i32, d: i32, uf: i32, vf: i32) -> i32 {
    lerp(lerp(a, b, uf), lerp(c, d, uf), vf)
}

/// A device-space coverage mask (one byte per device pixel), the kopitiam
/// stand-in for MuPDF's image-mask clip layer. Pixels outside it are 0.
pub struct DevMask {
    pub bbox: IRect,
    pub data: Vec<u8>,
}

impl DevMask {
    fn at(&self, x: i32, y: i32) -> i32 {
        if x < self.bbox.x0 || y < self.bbox.y0 || x >= self.bbox.x1 || y >= self.bbox.y1 {
            return 0;
        }
        let w = (self.bbox.x1 - self.bbox.x0) as usize;
        self.data[(y - self.bbox.y0) as usize * w + (x - self.bbox.x0) as usize] as i32
    }
}

/// `fz_is_rectilinear`: axis-aligned, possibly flipped or quarter-turned.
fn is_rectilinear(m: Matrix) -> bool {
    (m.b.abs() < f32::EPSILON && m.c.abs() < f32::EPSILON)
        || (m.a.abs() < f32::EPSILON && m.d.abs() < f32::EPSILON)
}

// MuPDF: sample_nearest (draw-affine.c:52)
#[inline]
fn sample_nearest(w: i64, h: i64, u: i64, v: i64) -> (i64, i64) {
    let mut u = u.max(0);
    let mut v = v.max(0);
    if u >= (w >> PREC) {
        u = (w >> PREC) - 1;
    }
    if v >= (h >> PREC) {
        v = (h >> PREC) - 1;
    }
    (u, v)
}

// MuPDF: fz_paint_image_imp (draw-affine.c:3901), reduced as described in the
// module docs. `img` must carry exactly the destination's 3 colour components
// (+ alpha when `img.alpha`, premultiplied). `alpha` is 0..=255.
/// Paint `img` onto `dst` through `ctm` (which maps the unit square onto the
/// image's device footprint), clipped to `scissor`.
#[allow(clippy::too_many_arguments)]
pub fn paint_image(
    dst: &mut Pixmap,
    scissor: IRect,
    img: &ScalePix,
    ctm: Matrix,
    alpha: i32,
    lerp_allowed: bool,
    interpolate: bool,
    mask: Option<&DevMask>,
) {
    if alpha == 0 || img.w <= 0 || img.h <= 0 {
        return;
    }
    debug_assert_eq!(img.n - i32::from(img.alpha), 3, "paint_image wants RGB samples");

    // Turn on interpolation for upscaled and non-rectilinear transforms ...
    let mut dolerp = false;
    if !is_rectilinear(ctm) {
        dolerp = lerp_allowed;
    }
    let ew = (ctm.a * ctm.a + ctm.b * ctm.b).sqrt();
    let eh = (ctm.c * ctm.c + ctm.d * ctm.d).sqrt();
    if ew > img.w as f32 {
        dolerp = lerp_allowed;
    }
    if eh > img.h as f32 {
        dolerp = lerp_allowed;
    }
    // ... except at large magnifications.
    if !interpolate {
        if ew > (img.w * 2) as f32 {
            dolerp = false;
        }
        if eh > (img.h * 2) as f32 {
            dolerp = false;
        }
    }

    let bbox = Rect::new(0.0, 0.0, 1.0, 1.0).transform(ctm).irect_from_rect().intersect(scissor);
    let bbox = bbox.intersect(dst.bbox());
    if bbox.is_empty() {
        return;
    }
    let (x, y) = (bbox.x0, bbox.y0);
    let w = bbox.x1 - bbox.x0;
    let h = bbox.y1 - bbox.y0;

    // Map from screen space (x,y) to image space (u,v).
    let m = ctm.pre_scale(1.0 / img.w as f32, 1.0 / img.h as f32);
    let Some(mut m) = m.try_invert() else { return };
    m.a *= ONE as f32;
    m.b *= ONE as f32;
    m.c *= ONE as f32;
    m.d *= ONE as f32;
    m.e *= ONE as f32;
    m.f *= ONE as f32;
    let fa = m.a as i64;
    let fb = m.b as i64;
    let fc = m.c as i64;
    let fd = m.d as i64;
    // Half step to start; kept in float as long as possible (bug 693021).
    let mut u0 = ((m.a * x as f32) + (m.c * y as f32) + m.e + ((m.a + m.c) * 0.5)) as i32 as i64;
    let mut v0 = ((m.b * x as f32) + (m.d * y as f32) + m.f + ((m.b + m.d) * 0.5)) as i32 as i64;

    let mut sw = img.w as i64;
    let mut sh = img.h as i64;
    if sw >= LIMIT || sh >= LIMIT {
        return; // "image too large for fixed point math"
    }
    if dolerp {
        u0 -= HALF;
        v0 -= HALF;
        sw = (sw << PREC) + HALF;
        sh = (sh << PREC) + HALF;
    }

    let sn = 3usize;
    let sa = img.alpha;
    let spx = img.n as usize;
    let ss = (img.w * img.n) as usize;
    let dn = dst.n as usize;
    let sp = &img.samples;

    for row in 0..h {
        let py = y + row;
        let mut u = u0;
        let mut v = v0;
        for col in 0..w {
            let px = x + col;
            // Per-pixel paint alpha: the constant alpha, folded with the
            // smask coverage when there is one (see the module docs).
            let a_px = match mask {
                Some(mk) => mul255(mk.at(px, py), alpha),
                None => alpha,
            };
            if a_px != 0 {
                let o = dst.offset(px, py).expect("bbox is inside dst");
                let d = &mut dst.samples[o..o + dn];
                if dolerp {
                    // template_affine_N_lerp / template_affine_alpha_N_lerp
                    if u + HALF >= 0 && u + ONE < sw && v + HALF >= 0 && v + ONE < sh {
                        let ui = u >> PREC;
                        let vi = v >> PREC;
                        let uf = (u & MASK) as i32;
                        let vf = (v & MASK) as i32;
                        let at = |uu: i64, vv: i64| {
                            let (uu, vv) = sample_nearest(sw, sh, uu, vv);
                            vv as usize * ss + uu as usize * spx
                        };
                        let (ia, ib, ic, id) = (at(ui, vi), at(ui + 1, vi), at(ui, vi + 1), at(ui + 1, vi + 1));
                        let comp = |k: usize| bilerp(sp[ia + k] as i32, sp[ib + k] as i32, sp[ic + k] as i32, sp[id + k] as i32, uf, vf);
                        let yv = if sa { comp(sn) } else { 255 };
                        if a_px == 255 {
                            if yv != 0 {
                                let t = 255 - yv;
                                for (k, dk) in d.iter_mut().enumerate().take(sn) {
                                    *dk = (comp(k) + mul255(*dk as i32, t)) as u8;
                                }
                            }
                        } else {
                            let xa = if sa { mul255(yv, a_px) } else { a_px };
                            if xa != 0 {
                                let t = 255 - xa;
                                for (k, dk) in d.iter_mut().enumerate().take(sn) {
                                    *dk = (mul255(comp(k), a_px) + mul255(*dk as i32, t)) as u8;
                                }
                            }
                        }
                    }
                } else {
                    // template_affine_N_near / template_affine_alpha_N_near
                    let ui = u >> PREC;
                    let vi = v >> PREC;
                    if ui >= 0 && ui < sw && vi >= 0 && vi < sh {
                        let si = vi as usize * ss + ui as usize * spx;
                        let sample = &sp[si..si + spx];
                        let a = if sa { sample[sn] as i32 } else { 255 };
                        if a_px == 255 {
                            if a != 0 {
                                let t = 255 - a;
                                if t == 0 {
                                    d[..sn].copy_from_slice(&sample[..sn]);
                                } else {
                                    for (k, dk) in d.iter_mut().enumerate().take(sn) {
                                        *dk = (sample[k] as i32 + mul255(*dk as i32, t)) as u8;
                                    }
                                }
                            }
                        } else {
                            let aa = if sa { mul255(a, a_px) } else { a_px };
                            if aa != 0 {
                                let t = 255 - aa;
                                for (k, dk) in d.iter_mut().enumerate().take(sn) {
                                    *dk = (mul255(sample[k] as i32, a_px) + mul255(*dk as i32, t)) as u8;
                                }
                            }
                        }
                    }
                }
            }
            u += fa;
            v += fb;
        }
        u0 += fc;
        v0 += fd;
    }
}
