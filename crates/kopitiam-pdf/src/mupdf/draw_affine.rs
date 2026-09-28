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
#[derive(Clone, Debug)]
pub struct DevMask {
    pub bbox: IRect,
    pub data: Vec<u8>,
}

impl DevMask {
    /// A mask covering `bbox` fully (every pixel 255).
    pub fn full(bbox: IRect) -> DevMask {
        let n = ((bbox.x1 - bbox.x0).max(0) * (bbox.y1 - bbox.y0).max(0)) as usize;
        DevMask { bbox, data: vec![255; n] }
    }

    // MuPDF: nested clip masks multiply (each clip layer is painted through
    // its own mask into the layer below, fz_paint_pixmap_with_mask).
    /// The pixel-wise product of two masks, over their common bbox.
    pub fn intersect(&self, other: &DevMask) -> DevMask {
        let bbox = self.bbox.intersect(other.bbox);
        let w = (bbox.x1 - bbox.x0).max(0);
        let h = (bbox.y1 - bbox.y0).max(0);
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in bbox.y0..bbox.y0 + h {
            for x in bbox.x0..bbox.x0 + w {
                data.push(mul255(self.at(x, y), other.at(x, y)) as u8);
            }
        }
        DevMask { bbox, data }
    }

    /// Coverage 0..=255 at device pixel `(x, y)`; 0 outside the bbox.
    pub(crate) fn at(&self, x: i32, y: i32) -> i32 {
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

/// The fixed-point stepping state `fz_paint_image_imp` sets up before it
/// dispatches to a span painter.
struct Walk {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    u0: i64,
    v0: i64,
    fa: i64,
    fb: i64,
    fc: i64,
    fd: i64,
    sw: i64,
    sh: i64,
    dolerp: bool,
}

// MuPDF: the set-up half of fz_paint_image_imp (draw-affine.c:3926-4029).
fn walk(
    dst: &Pixmap,
    scissor: IRect,
    img_w: i32,
    img_h: i32,
    ctm: Matrix,
    lerp_allowed: bool,
    interpolate: bool,
) -> Option<Walk> {
    // Turn on interpolation for upscaled and non-rectilinear transforms ...
    let mut dolerp = false;
    if !is_rectilinear(ctm) {
        dolerp = lerp_allowed;
    }
    let ew = (ctm.a * ctm.a + ctm.b * ctm.b).sqrt();
    let eh = (ctm.c * ctm.c + ctm.d * ctm.d).sqrt();
    if ew > img_w as f32 {
        dolerp = lerp_allowed;
    }
    if eh > img_h as f32 {
        dolerp = lerp_allowed;
    }
    // ... except at large magnifications.
    if !interpolate {
        if ew > (img_w * 2) as f32 {
            dolerp = false;
        }
        if eh > (img_h * 2) as f32 {
            dolerp = false;
        }
    }

    let bbox = Rect::new(0.0, 0.0, 1.0, 1.0)
        .transform(ctm)
        .irect_from_rect()
        .intersect(scissor)
        .intersect(dst.bbox());
    if bbox.is_empty() {
        return None;
    }

    // Map from screen space (x,y) to image space (u,v).
    let m = ctm.pre_scale(1.0 / img_w as f32, 1.0 / img_h as f32);
    let mut m = m.try_invert()?;
    m.a *= ONE as f32;
    m.b *= ONE as f32;
    m.c *= ONE as f32;
    m.d *= ONE as f32;
    m.e *= ONE as f32;
    m.f *= ONE as f32;
    let (x, y) = (bbox.x0, bbox.y0);
    // Half step to start; kept in float as long as possible (bug 693021).
    let mut u0 = ((m.a * x as f32) + (m.c * y as f32) + m.e + ((m.a + m.c) * 0.5)) as i32 as i64;
    let mut v0 = ((m.b * x as f32) + (m.d * y as f32) + m.f + ((m.b + m.d) * 0.5)) as i32 as i64;
    let mut sw = img_w as i64;
    let mut sh = img_h as i64;
    if sw >= LIMIT || sh >= LIMIT {
        return None; // "image too large for fixed point math"
    }
    if dolerp {
        u0 -= HALF;
        v0 -= HALF;
        sw = (sw << PREC) + HALF;
        sh = (sh << PREC) + HALF;
    }
    Some(Walk {
        x,
        y,
        w: bbox.x1 - bbox.x0,
        h: bbox.y1 - bbox.y0,
        u0,
        v0,
        fa: m.a as i64,
        fb: m.b as i64,
        fc: m.c as i64,
        fd: m.d as i64,
        sw,
        sh,
        dolerp,
    })
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
    let Some(wk) = walk(dst, scissor, img.w, img.h, ctm, lerp_allowed, interpolate) else {
        return;
    };
    let (sw, sh, dolerp) = (wk.sw, wk.sh, wk.dolerp);

    let sn = 3usize;
    let sa = img.alpha;
    let spx = img.n as usize;
    let ss = (img.w * img.n) as usize;
    let dn = dst.n as usize;
    let sp = &img.samples;

    let (mut u0, mut v0) = (wk.u0, wk.v0);
    for row in 0..wk.h {
        let py = wk.y + row;
        let mut u = u0;
        let mut v = v0;
        for col in 0..wk.w {
            let px = wk.x + col;
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
            u += wk.fa;
            v += wk.fb;
        }
        u0 += wk.fc;
        v0 += wk.fd;
    }
}

// MuPDF: FZ_EXPAND / FZ_COMBINE / FZ_BLEND (geometry.h:57-76).
#[inline]
fn fz_expand(a: i32) -> i32 {
    a + (a >> 7)
}
#[inline]
fn fz_combine(a: i32, b: i32) -> i32 {
    (a * b) >> 8
}
#[inline]
fn fz_blend(src: i32, dst: i32, amount: i32) -> i32 {
    ((src - dst) * amount + (dst << 8)) >> 8
}

// MuPDF: fz_paint_image_with_color (draw-affine.c:4114) ->
// template_affine_color_N_near / _lerp (draw-affine.c:1068, 1155).
/// Paint `color` through a one-channel coverage pixmap `mask` (a stencil
/// `/ImageMask` after decode + scaling), at `alpha` (0..=255): MuPDF's
/// `fz_fill_image_mask` painter.
#[allow(clippy::too_many_arguments)]
pub fn paint_image_color(
    dst: &mut Pixmap,
    scissor: IRect,
    mask: &ScalePix,
    ctm: Matrix,
    color: [u8; 3],
    alpha: i32,
    lerp_allowed: bool,
    interpolate: bool,
    clip_mask: Option<&DevMask>,
) {
    if alpha == 0 || mask.w <= 0 || mask.h <= 0 {
        return;
    }
    debug_assert_eq!(mask.n, 1, "a stencil is one coverage channel");
    let Some(wk) = walk(dst, scissor, mask.w, mask.h, ctm, lerp_allowed, interpolate) else {
        return;
    };
    let (sw, sh) = (wk.sw, wk.sh);
    let ss = mask.w as usize;
    let sp = &mask.samples;
    let dn = dst.n as usize;
    let (mut u0, mut v0) = (wk.u0, wk.v0);
    for row in 0..wk.h {
        let py = wk.y + row;
        let mut u = u0;
        let mut v = v0;
        for col in 0..wk.w {
            let px = wk.x + col;
            let ma = if wk.dolerp {
                if u + HALF >= 0 && u + ONE < sw && v + HALF >= 0 && v + ONE < sh {
                    let ui = u >> PREC;
                    let vi = v >> PREC;
                    let uf = (u & MASK) as i32;
                    let vf = (v & MASK) as i32;
                    let at = |uu: i64, vv: i64| {
                        let (uu, vv) = sample_nearest(sw, sh, uu, vv);
                        sp[vv as usize * ss + uu as usize] as i32
                    };
                    Some(bilerp(at(ui, vi), at(ui + 1, vi), at(ui, vi + 1), at(ui + 1, vi + 1), uf, vf))
                } else {
                    None
                }
            } else {
                let ui = u >> PREC;
                let vi = v >> PREC;
                (ui >= 0 && ui < sw && vi >= 0 && vi < sh).then(|| sp[vi as usize * ss + ui as usize] as i32)
            };
            if let Some(ma) = ma {
                let alpha = match clip_mask {
                    Some(cm) => mul255(cm.at(px, py), alpha),
                    None => alpha,
                };
                let masa = fz_combine(fz_expand(ma), alpha);
                if masa != 0 {
                    let o = dst.offset(px, py).expect("bbox is inside dst");
                    for (k, dk) in dst.samples[o..o + dn].iter_mut().enumerate().take(3) {
                        *dk = fz_blend(color[k] as i32, *dk as i32, masa) as u8;
                    }
                }
            }
            u += wk.fa;
            v += wk.fb;
        }
        u0 += wk.fc;
        v0 += wk.fd;
    }
}
