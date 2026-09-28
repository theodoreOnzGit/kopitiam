//! Ported from MuPDF `source/fitz/draw-blend.c` (the 16 PDF blend modes,
//! `fz_blend_pixmap` and the isolated / non-isolated span blenders it
//! dispatches to) plus the pixmap-to-pixmap painters of
//! `source/fitz/draw-paint.c` the group and mask stack needs
//! (`fz_paint_pixmap`, `fz_paint_pixmap_alpha`, `fz_paint_pixmap_with_mask`,
//! `fz_paint_over_pixmap_with_mask`) and the fast RGB -> gray pixmap
//! conversion of `source/fitz/color-fast.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the algorithms and the integer
//! arithmetic follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # What this is for
//!
//! The draw device's transparency groups and soft masks
//! ([`DrawDevice`](super::draw_device::DrawDevice)'s `begin_group` /
//! `end_group` / `begin_mask` / `end_mask`) render into offscreen pixmaps
//! and composite them back with these. Only the arms the device can reach
//! are here: the working colour space is always DeviceRGB, so every pixmap
//! is `n = 3` (opaque) or `n = 4` (RGB + premultiplied alpha). No CMYK
//! `complement`, no spot colours -- those branches of the C are dead for
//! this device lah.
//!
//! A "shape" / group-alpha plane is represented by an RGBA pixmap whose
//! **alpha channel** is MuPDF's alpha-only `state->group_alpha` pixmap (the
//! colour channels are scratch). That lets every existing painter update it
//! without a second, alpha-only code path; only the last channel is ever read
//! here.

use super::draw_affine::DevMask;
use super::geometry::IRect;
use super::pixmap::Pixmap;

/// A PDF blend mode (`/BM`), in MuPDF's `FZ_BLEND_*` order.
// MuPDF: enum { FZ_BLEND_NORMAL, ... } (include/mupdf/fitz/device.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    // MuPDF: fz_lookup_blendmode (draw-blend.c:132) -- an unknown name is
    // Normal.
    /// The blend mode a `/BM` name selects.
    pub fn lookup(name: &[u8]) -> BlendMode {
        match name {
            b"Multiply" => BlendMode::Multiply,
            b"Screen" => BlendMode::Screen,
            b"Overlay" => BlendMode::Overlay,
            b"Darken" => BlendMode::Darken,
            b"Lighten" => BlendMode::Lighten,
            b"ColorDodge" => BlendMode::ColorDodge,
            b"ColorBurn" => BlendMode::ColorBurn,
            b"HardLight" => BlendMode::HardLight,
            b"SoftLight" => BlendMode::SoftLight,
            b"Difference" => BlendMode::Difference,
            b"Exclusion" => BlendMode::Exclusion,
            b"Hue" => BlendMode::Hue,
            b"Saturation" => BlendMode::Saturation,
            b"Color" => BlendMode::Color,
            b"Luminosity" => BlendMode::Luminosity,
            _ => BlendMode::Normal,
        }
    }

    /// `blendmode >= FZ_BLEND_HUE`: the four non-separable modes.
    fn is_nonseparable(self) -> bool {
        matches!(self, BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity)
    }
}

// MuPDF: fz_mul255 (geometry.h:38).
#[inline]
fn mul255(a: i32, b: i32) -> i32 {
    let mut x = a * b + 128;
    x += x >> 8;
    x >> 8
}

// MuPDF: FZ_EXPAND / FZ_COMBINE / FZ_BLEND (geometry.h).
#[inline]
fn expand(a: i32) -> i32 {
    a + (a >> 7)
}
#[inline]
fn combine(a: i32, b: i32) -> i32 {
    (a * b) >> 8
}
#[inline]
fn fz_blend(src: i32, dst: i32, amount: i32) -> i32 {
    ((src - dst) * amount + (dst << 8)) >> 8
}

// ---------------------------------------------------------------------------
// Separable blend modes (draw-blend.c:150-236)
// ---------------------------------------------------------------------------

// MuPDF: fz_screen_byte (draw-blend.c:150)
#[inline]
fn screen_byte(b: i32, s: i32) -> i32 {
    b + s - mul255(b, s)
}

// MuPDF: fz_hard_light_byte (draw-blend.c:155)
#[inline]
fn hard_light_byte(b: i32, s: i32) -> i32 {
    let s2 = s << 1;
    if s <= 127 { mul255(b, s2) } else { screen_byte(b, s2 - 255) }
}

// MuPDF: fz_color_dodge_byte (draw-blend.c:179)
#[inline]
fn color_dodge_byte(b: i32, s: i32) -> i32 {
    let s = 255 - s;
    if b <= 0 {
        0
    } else if b >= s {
        255
    } else {
        (0x1fe * b + s) / (s << 1)
    }
}

// MuPDF: fz_color_burn_byte (draw-blend.c:190)
#[inline]
fn color_burn_byte(b: i32, s: i32) -> i32 {
    let b = 255 - b;
    if b <= 0 {
        255
    } else if b >= s {
        0
    } else {
        0xff - (0x1fe * b + s) / (s << 1)
    }
}

// MuPDF: fz_soft_light_byte (draw-blend.c:201)
#[inline]
fn soft_light_byte(b: i32, s: i32) -> i32 {
    if s < 128 {
        b - mul255(mul255(255 - (s << 1), b), 255 - b)
    } else {
        let dbd = if b < 64 {
            mul255(mul255((b << 4) - 3060, b) + 1020, b)
        } else {
            (255.0f32 * b as f32).sqrt() as i32
        };
        b + mul255((s << 1) - 255, dbd - b)
    }
}

// MuPDF: the `switch (blendmode)` of fz_blend_separable (draw-blend.c:390).
/// `B(cb, cs)` for a separable mode (Normal for anything else).
#[inline]
fn separable(mode: BlendMode, bc: i32, sc: i32) -> i32 {
    match mode {
        BlendMode::Multiply => mul255(bc, sc),
        BlendMode::Screen => screen_byte(bc, sc),
        BlendMode::Overlay => hard_light_byte(sc, bc), // note swapped order
        BlendMode::Darken => bc.min(sc),
        BlendMode::Lighten => bc.max(sc),
        BlendMode::ColorDodge => color_dodge_byte(bc, sc),
        BlendMode::ColorBurn => color_burn_byte(bc, sc),
        BlendMode::HardLight => hard_light_byte(bc, sc),
        BlendMode::SoftLight => soft_light_byte(bc, sc),
        BlendMode::Difference => (bc - sc).abs(),
        BlendMode::Exclusion => bc + sc - (mul255(bc, sc) << 1),
        _ => sc,
    }
}

// ---------------------------------------------------------------------------
// Non-separable blend modes (draw-blend.c:240-350)
// ---------------------------------------------------------------------------

// MuPDF: fz_luminosity_rgb (draw-blend.c:241)
fn luminosity_rgb(rb: i32, gb: i32, bb: i32, rs: i32, gs: i32, bs: i32) -> [i32; 3] {
    // 0.3f, 0.59f, 0.11f in fixed point
    let delta = ((rs - rb) * 77 + (gs - gb) * 151 + (bs - bb) * 28 + 0x80) >> 8;
    let (mut r, mut g, mut b) = (rb + delta, gb + delta, bb + delta);
    if (r | g | b) & 0x100 != 0 {
        let y = (rs * 77 + gs * 151 + bs * 28 + 0x80) >> 8;
        let scale = if delta > 0 {
            let max = r.max(g.max(b));
            if max == y { 0 } else { ((255 - y) << 16) / (max - y) }
        } else {
            let min = r.min(g.min(b));
            if y == min { 0 } else { (y << 16) / (y - min) }
        };
        r = y + (((r - y) * scale + 0x8000) >> 16);
        g = y + (((g - y) * scale + 0x8000) >> 16);
        b = y + (((b - y) * scale + 0x8000) >> 16);
    }
    [r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255)]
}

// MuPDF: fz_saturation_rgb (draw-blend.c:279)
fn saturation_rgb(rb: i32, gb: i32, bb: i32, rs: i32, gs: i32, bs: i32) -> [i32; 3] {
    let minb = rb.min(gb.min(bb));
    let maxb = rb.max(gb.max(bb));
    if minb == maxb {
        // backdrop has zero saturation, avoid divide by 0
        let g = gb.clamp(0, 255);
        return [g, g, g];
    }
    let mins = rs.min(gs.min(bs));
    let maxs = rs.max(gs.max(bs));
    let mut scale = ((maxs - mins) << 16) / (maxb - minb);
    let y = (rb * 77 + gb * 151 + bb * 28 + 0x80) >> 8;
    let mut r = y + ((((rb - y) * scale) + 0x8000) >> 16);
    let mut g = y + ((((gb - y) * scale) + 0x8000) >> 16);
    let mut b = y + ((((bb - y) * scale) + 0x8000) >> 16);
    if (r | g | b) & 0x100 != 0 {
        let min = r.min(g.min(b));
        let max = r.max(g.max(b));
        let scalemin = if min < 0 { (y << 16) / (y - min) } else { 0x10000 };
        let scalemax = if max > 255 { ((255 - y) << 16) / (max - y) } else { 0x10000 };
        scale = scalemin.min(scalemax);
        r = y + (((r - y) * scale + 0x8000) >> 16);
        g = y + (((g - y) * scale + 0x8000) >> 16);
        b = y + (((b - y) * scale + 0x8000) >> 16);
    }
    [r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255)]
}

// MuPDF: the `switch (blendmode)` of fz_blend_nonseparable (draw-blend.c:547)
// with fz_color_rgb / fz_hue_rgb (draw-blend.c:333, 339).
/// `B(cb, cs)` for a non-separable mode, backdrop `b` and source `s`.
fn nonseparable(mode: BlendMode, b: [i32; 3], s: [i32; 3]) -> [i32; 3] {
    match mode {
        BlendMode::Saturation => saturation_rgb(b[0], b[1], b[2], s[0], s[1], s[2]),
        BlendMode::Color => luminosity_rgb(s[0], s[1], s[2], b[0], b[1], b[2]),
        BlendMode::Luminosity => luminosity_rgb(b[0], b[1], b[2], s[0], s[1], s[2]),
        // FZ_BLEND_HUE (and the `default:` arm)
        _ => {
            let t = luminosity_rgb(s[0], s[1], s[2], b[0], b[1], b[2]);
            saturation_rgb(t[0], t[1], t[2], b[0], b[1], b[2])
        }
    }
}

// ---------------------------------------------------------------------------
// Per-pixel blend loops (draw-blend.c:350-900), RGB only (n1 = 3, no spots)
// ---------------------------------------------------------------------------

// MuPDF: fz_blend_separable / fz_blend_nonseparable (draw-blend.c:350, 499),
// one pixel. `bp`/`sp` are the destination and source pixels, each with a
// trailing alpha iff `bal`/`sal`.
fn blend_isolated_px(bp: &mut [u8], bal: bool, sp: &[u8], sal: bool, mode: BlendMode) {
    let sa = if sal { sp[3] as i32 } else { 255 };
    if sa == 0 {
        return;
    }
    let ba = if bal { bp[3] as i32 } else { 255 };
    if ba == 0 {
        // memcpy(bp, sp, n1 + (sal && bal)). The C then writes
        // `bp[n1+1] = 255` for bal && !sal -- one byte PAST the pixel, into
        // the next pixel's second channel. The branch cannot be reached by
        // this device (an alpha destination always gets an alpha source),
        // and we set this pixel's own alpha instead of copying the overrun.
        bp[..3].copy_from_slice(&sp[..3]);
        if bal {
            bp[3] = if sal { sp[3] } else { 255 };
        }
        return;
    }
    let saba = mul255(sa, ba);
    // ugh, division to get non-premul components
    let invsa = 255 * 256 / sa;
    let invba = 255 * 256 / ba;
    let rc: [i32; 3] = if mode.is_nonseparable() {
        let s = [(sp[0] as i32 * invsa) >> 8, (sp[1] as i32 * invsa) >> 8, (sp[2] as i32 * invsa) >> 8];
        let b = [(bp[0] as i32 * invba) >> 8, (bp[1] as i32 * invba) >> 8, (bp[2] as i32 * invba) >> 8];
        nonseparable(mode, b, s)
    } else {
        let mut r = [0; 3];
        for (k, rk) in r.iter_mut().enumerate() {
            let sc = (sp[k] as i32 * invsa) >> 8;
            let bc = (bp[k] as i32 * invba) >> 8;
            *rk = separable(mode, bc, sc);
        }
        r
    };
    for k in 0..3 {
        // nonseparable stores through `unsigned char` (rr/rg/rb) and the
        // separable path through `byte`: both wrap to 8 bits.
        let v = mul255(255 - sa, bp[k] as i32) + mul255(255 - ba, sp[k] as i32) + mul255(saba, rc[k]);
        bp[k] = v as u8;
    }
    if bal {
        bp[3] = (ba + sa - saba) as u8;
    }
}

// MuPDF: fz_blend_separable_nonisolated (draw-blend.c:648), one pixel.
// `ha` is the group's shape (group alpha) at this pixel.
fn blend_sep_nonisolated_px(bp: &mut [u8], bal: bool, sp: &[u8], sal: bool, mode: BlendMode, ha: i32, alpha: i32) {
    let haa = mul255(ha, alpha); // ha = shape_alpha
    // If haa == 0 then leave everything unchanged
    if haa == 0 {
        return;
    }
    let sa = if sal { sp[3] as i32 } else { 255 };
    if sa == 0 {
        return; // No change!
    }
    let invsa = 255 * 256 / sa;
    let ba = if bal { bp[3] as i32 } else { 255 };
    if ba == 0 {
        // Just copy pixels (allowing for change in premultiplied alphas)
        for k in 0..3 {
            bp[k] = mul255((sp[k] as i32 * invsa) >> 8, haa) as u8;
        }
        if bal {
            bp[3] = haa as u8;
        }
        return;
    }
    let invba = 255 * 256 / ba;
    // The gs 'uncomposition' of pdf_reference17 section 7.3.3 (the C says as
    // much: copied from gs, understood by nobody -- we follow it exactly).
    let scale = (512 * ba + ha) / (ha * 2) - expand(ba);
    let bahaa = mul255(ba, haa);
    let ra0 = ba - bahaa;
    let ra = ra0 + haa;
    if bal {
        bp[3] = ra as u8;
    }
    if ra == 0 {
        return;
    }
    for k in 0..3 {
        let mut sc = (sp[k] as i32 * invsa) >> 8;
        let bc = (bp[k] as i32 * invba) >> 8;
        // Uncomposite
        sc += ((sc - bc) * scale) >> 8;
        sc = sc.clamp(0, 255);
        let mut rc = separable(mode, bc, sc);
        // ra.rc = bc * ra0 + haa * (255 - ba) * sc + bahaa * B(Cb, Cs)
        if bahaa != 255 {
            rc = mul255(bahaa, rc);
        }
        if ba != 255 {
            let t = mul255(255 - ba, haa);
            rc += mul255(t, sc);
        }
        if ra0 != 0 {
            rc += mul255(ra0, bc);
        }
        bp[k] = rc.clamp(0, ra) as u8;
    }
}

// MuPDF: fz_blend_nonseparable_nonisolated (draw-blend.c:917), one pixel.
fn blend_nonsep_nonisolated_px(bp: &mut [u8], bal: bool, sp: &[u8], sal: bool, mode: BlendMode, ha: i32, alpha: i32) {
    let haa = mul255(ha, alpha);
    if haa == 0 {
        return;
    }
    let sa = if sal { sp[3] as i32 } else { 255 };
    let ba = if bal { bp[3] as i32 } else { 255 };
    if ba == 0 && alpha == 255 {
        bp[..3].copy_from_slice(&sp[..3]);
        if bal {
            bp[3] = if sal { sp[3] } else { 255 };
        }
        return;
    }
    let bahaa = mul255(ba, haa);
    // Calculate result_alpha
    let ra0 = ba - bahaa;
    let ra = ra0 + haa;
    if bal {
        bp[3] = ra as u8;
    }
    if ra == 0 {
        return;
    }
    // "We assume that normal blending has been done inside the group, so:
    // ra.rc = (1-ha).bc + ha.sc" -- uncomposite with 1/ha.
    let invha = if ha != 0 { 255 * 256 / ha } else { 0 };
    let invsa = if sa != 0 { 255 * 256 / sa } else { 0 };
    let invba = if ba != 0 { 255 * 256 / ba } else { 0 };
    let b = [(bp[0] as i32 * invba) >> 8, (bp[1] as i32 * invba) >> 8, (bp[2] as i32 * invba) >> 8];
    let mut s = [0i32; 3];
    for k in 0..3 {
        let v = (sp[k] as i32 * invsa) >> 8;
        s[k] = ((((v - b[k]) * invha) >> 8) + b[k]).clamp(0, 255);
    }
    let r = nonseparable(mode, b, s);
    // rr/rg/rb are `unsigned char` in the C: every += wraps at 8 bits.
    let mut out = [r[0] as u8, r[1] as u8, r[2] as u8];
    for k in 0..3 {
        if bahaa != 255 {
            out[k] = mul255(bahaa, out[k] as i32) as u8;
        }
        if ba != 255 {
            let t = mul255(255 - ba, haa);
            out[k] = out[k].wrapping_add(mul255(t, s[k]) as u8);
        }
        if ra0 != 0 {
            out[k] = out[k].wrapping_add(mul255(ra0, b[k]) as u8);
        }
    }
    bp[..3].copy_from_slice(&out);
}

// MuPDF: fz_blend_pixmap (draw-blend.c:1095).
/// Blend `src` onto `dst` with `mode` at `alpha` (0..=255). `isolated`
/// groups composite with the plain PDF equation; non-isolated ones first
/// "uncomposite" the backdrop they were drawn over, using `shape` (the
/// group-alpha plane -- required when `!isolated`, read from its last
/// channel).
///
/// Like the C, an isolated blend at `alpha < 255` first multiplies `src`
/// through by `alpha` IN PLACE ("TODO: fix this hack!"), which the caller
/// relies on when it later paints `src`'s alpha into a parent group alpha.
pub fn blend_pixmap(dst: &mut Pixmap, src: &mut Pixmap, alpha: i32, mode: BlendMode, isolated: bool, shape: Option<&Pixmap>) {
    if isolated && alpha < 255 {
        for v in src.samples.iter_mut() {
            *v = mul255(*v as i32, alpha) as u8;
        }
    }
    let bbox = src.bbox().intersect(dst.bbox());
    if bbox.is_empty() {
        return;
    }
    let (sal, bal) = (src.alpha, dst.alpha);
    let (sn, dn) = (src.n as usize, dst.n as usize);
    if sn - usize::from(sal) != 3 || dn - usize::from(bal) != 3 {
        return; // only DeviceRGB reaches here (see the module docs)
    }
    for y in bbox.y0..bbox.y1 {
        for x in bbox.x0..bbox.x1 {
            let so = src.offset(x, y).expect("in src bbox");
            let o = dst.offset(x, y).expect("in dst bbox");
            let sp = &src.samples[so..so + sn];
            let bp = &mut dst.samples[o..o + dn];
            if isolated {
                blend_isolated_px(bp, bal, sp, sal, mode);
            } else {
                // A non-isolated group always has its shape; without one
                // there is nothing to uncomposite with, so treat it as empty.
                let ha = shape.and_then(|s| s.offset(x, y).map(|ho| s.samples[ho + s.n as usize - 1] as i32)).unwrap_or(0);
                if mode.is_nonseparable() {
                    blend_nonsep_nonisolated_px(bp, bal, sp, sal, mode, ha, alpha);
                } else if !sal && alpha == 255 && mode == BlendMode::Normal {
                    // "the uncompositing and the recompositing cancel one
                    // another out, and it's just a simple copy."
                    if mul255(ha, alpha) != 0 {
                        bp[..3].copy_from_slice(&sp[..3]);
                        if bal {
                            bp[3] = 255;
                        }
                    }
                } else {
                    blend_sep_nonisolated_px(bp, bal, sp, sal, mode, ha, alpha);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pixmap painters (draw-paint.c)
// ---------------------------------------------------------------------------

// MuPDF: fz_paint_pixmap (draw-paint.c:2460) -> template_span_N_general /
// template_span_N_with_alpha_general (draw-paint.c:1870, 1680).
/// Paint premultiplied `src` over `dst` at `alpha` (0..=255). Both must
/// carry the same number of colour components (MuPDF silently does nothing
/// otherwise -- its "FIXME").
pub fn paint_pixmap(dst: &mut Pixmap, src: &Pixmap, alpha: i32) {
    if alpha == 0 {
        return;
    }
    let (sa, da) = (src.alpha, dst.alpha);
    let (sn, dn) = (src.n as usize, dst.n as usize);
    let n1 = sn - usize::from(sa);
    if dn - usize::from(da) != n1 {
        return;
    }
    let bbox = src.bbox().intersect(dst.bbox());
    if bbox.is_empty() {
        return;
    }
    let alpha_x = if sa { expand(alpha) } else { alpha };
    for y in bbox.y0..bbox.y1 {
        for x in bbox.x0..bbox.x1 {
            let so = src.offset(x, y).expect("in src bbox");
            let o = dst.offset(x, y).expect("in dst bbox");
            let sp = &src.samples[so..so + sn];
            let dp = &mut dst.samples[o..o + dn];
            if alpha == 255 {
                let t = if sa { expand(sp[n1] as i32) } else { 256 };
                if t == 0 {
                    continue;
                }
                let t = 256 - t;
                if t == 0 {
                    dp[..n1].copy_from_slice(&sp[..n1]);
                    if da {
                        dp[n1] = if sa { sp[n1] } else { 255 };
                    }
                } else {
                    // sa != 0, as t != 0
                    for k in 0..n1 {
                        dp[k] = (sp[k] as i32 + combine(dp[k] as i32, t)) as u8;
                    }
                    if da {
                        dp[n1] = (sp[n1] as i32 + combine(dp[n1] as i32, t)) as u8;
                    }
                }
            } else {
                let masa = if sa { combine(sp[n1] as i32, alpha_x) } else { alpha_x };
                let t = expand(255 - masa);
                for k in 0..n1 {
                    dp[k] = (combine(sp[k] as i32, alpha_x) + combine(dp[k] as i32, t)) as u8;
                }
                if da {
                    dp[n1] = (masa + combine(dp[n1] as i32, t)) as u8;
                }
            }
        }
    }
}

// MuPDF: fz_paint_pixmap_alpha (draw-paint.c:2538) -> paint_span_alpha_solid /
// paint_span_alpha_not_solid; also what fz_paint_pixmap does between two
// alpha-only pixmaps. Writes only `dst`'s alpha channel (its group-alpha
// plane), from `src`'s alpha channel.
/// Paint `src`'s alpha over `dst`'s alpha at `alpha` (0..=255).
pub fn paint_alpha_plane(dst: &mut Pixmap, src: &Pixmap, alpha: i32) {
    if alpha == 0 || !src.alpha || !dst.alpha {
        return;
    }
    let bbox = src.bbox().intersect(dst.bbox());
    if bbox.is_empty() {
        return;
    }
    let (sl, dl) = (src.n as usize - 1, dst.n as usize - 1);
    let alpha_x = expand(alpha);
    for y in bbox.y0..bbox.y1 {
        for x in bbox.x0..bbox.x1 {
            let s = src.samples[src.offset(x, y).expect("in src bbox") + sl] as i32;
            let o = dst.offset(x, y).expect("in dst bbox") + dl;
            let d = dst.samples[o] as i32;
            dst.samples[o] = if alpha == 255 {
                let t = expand(255 - s);
                s + combine(d, t)
            } else {
                let masa = combine(s, alpha_x);
                let t = expand(255 - masa);
                masa + combine(d, t)
            } as u8;
        }
    }
}

// MuPDF: fz_paint_pixmap_with_mask (draw-paint.c:2626) ->
// template_span_with_mask_N_general (draw-paint.c:1451).
/// Lerp `src` onto `dst` through the coverage `mask` -- how a clip (or a soft
/// mask, which becomes a clip) pops. `src` and `dst` share a layout.
pub fn paint_pixmap_with_mask(dst: &mut Pixmap, src: &Pixmap, mask: &DevMask) {
    if src.n != dst.n {
        return;
    }
    let bbox = dst.bbox().intersect(src.bbox()).intersect(mask.bbox);
    if bbox.is_empty() {
        return;
    }
    let n = dst.n as usize;
    let a = src.alpha;
    let n1 = n - usize::from(a);
    for y in bbox.y0..bbox.y1 {
        for x in bbox.x0..bbox.x1 {
            let ma = mask.at(x, y);
            let so = src.offset(x, y).expect("in src bbox");
            if ma == 0 || (a && src.samples[so + n1] == 0) {
                continue;
            }
            let o = dst.offset(x, y).expect("in dst bbox");
            if ma == 255 {
                dst.samples[o..o + n].copy_from_slice(&src.samples[so..so + n]);
            } else {
                let ma = expand(ma);
                for k in 0..n {
                    dst.samples[o + k] = fz_blend(src.samples[so + k] as i32, dst.samples[o + k] as i32, ma) as u8;
                }
            }
        }
    }
}

// MuPDF: fz_paint_over_pixmap_with_mask (draw-paint.c:2700) ->
// paint_over_span_with_mask (draw-paint.c:2673): union of two alpha planes,
// the source attenuated by the mask. Channels as in paint_alpha_plane.
/// `dst.alpha = 255 - (255 - src.alpha * m)(255 - dst.alpha)`.
pub fn paint_over_alpha_with_mask(dst: &mut Pixmap, src: &Pixmap, mask: &DevMask) {
    if !src.alpha || !dst.alpha {
        return;
    }
    let bbox = dst.bbox().intersect(src.bbox()).intersect(mask.bbox);
    if bbox.is_empty() {
        return;
    }
    let (sl, dl) = (src.n as usize - 1, dst.n as usize - 1);
    for y in bbox.y0..bbox.y1 {
        for x in bbox.x0..bbox.x1 {
            let ma = expand(mask.at(x, y));
            let s = src.samples[src.offset(x, y).expect("in src bbox") + sl] as i32;
            if ma == 0 || s == 0 {
                continue;
            }
            // (sic) fz_mul255 of the EXPANDED mask value, as in the C.
            let a = if ma != 256 { mul255(ma, s) } else { s };
            let o = dst.offset(x, y).expect("in dst bbox") + dl;
            dst.samples[o] = (255 - mul255(255 - a, 255 - dst.samples[o] as i32)) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Pixmap utilities
// ---------------------------------------------------------------------------

// MuPDF: fast_rgb_to_gray (color-fast.c:440), the per-pixel formula.
/// DeviceRGB bytes -> DeviceGray byte, as `fz_convert_pixmap` computes it
/// without ICC.
#[inline]
pub fn rgb_to_gray_byte(r: u8, g: u8, b: u8) -> u8 {
    (((r as i32 + 1) * 77 + (g as i32 + 1) * 150 + (b as i32 + 1) * 28) >> 8) as u8
}

/// A pixmap of `n` components (`alpha`: last is alpha) covering `bbox`
/// (empty when `bbox` is), all samples zero -- `fz_new_pixmap_with_bbox`
/// followed by `fz_clear_pixmap`.
pub fn new_pixmap_with_bbox(bbox: IRect, n: u8, alpha: bool) -> Pixmap {
    let w = (bbox.x1 - bbox.x0).max(0) as u32;
    let h = (bbox.y1 - bbox.y0).max(0) as u32;
    let mut p = Pixmap::new(w, h, n, alpha);
    p.x = bbox.x0;
    p.y = bbox.y0;
    p
}

// MuPDF: fz_copy_pixmap_rect (pixmap.c:470), same-layout case.
/// A copy of `src` restricted to `bbox` (intersected with `src`'s bounds).
pub fn copy_pixmap_rect(src: &Pixmap, bbox: IRect) -> Pixmap {
    let bbox = bbox.intersect(src.bbox());
    let mut out = new_pixmap_with_bbox(bbox, src.n, src.alpha);
    if out.w == 0 || out.h == 0 {
        return out;
    }
    let n = src.n as usize;
    let row = out.w as usize * n;
    for y in bbox.y0..bbox.y1 {
        let so = src.offset(bbox.x0, y).expect("in src bbox");
        let o = out.offset(bbox.x0, y).expect("in out bbox");
        out.samples[o..o + row].copy_from_slice(&src.samples[so..so + row]);
    }
    out
}

/// Write `src` back into `dst` over their common area (same layout): the
/// inverse of [`copy_pixmap_rect`].
pub fn put_pixmap_rect(dst: &mut Pixmap, src: &Pixmap) {
    let bbox = src.bbox().intersect(dst.bbox());
    if bbox.is_empty() || src.n != dst.n {
        return;
    }
    let n = src.n as usize;
    let row = (bbox.x1 - bbox.x0) as usize * n;
    for y in bbox.y0..bbox.y1 {
        let so = src.offset(bbox.x0, y).expect("in src bbox");
        let o = dst.offset(bbox.x0, y).expect("in dst bbox");
        dst.samples[o..o + row].copy_from_slice(&src.samples[so..so + row]);
    }
}

// MuPDF: fz_draw_end_group's colour conversion (fz_convert_pixmap into the
// parent's space) for a group whose blending space is DeviceGray, drawn by
// this device in RGB: convert every pixel to gray with the fast formula and
// write it back to all three channels (gray -> RGB is a replication).
/// Knock an RGB(A) pixmap down to its gray values, in place.
pub fn rgb_to_gray_in_place(pix: &mut Pixmap) {
    if pix.n < 3 {
        return;
    }
    let n = pix.n as usize;
    for px in pix.samples.chunks_mut(n) {
        let g = rgb_to_gray_byte(px[0], px[1], px[2]);
        px[0] = g;
        px[1] = g;
        px[2] = g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_names_and_default() {
        assert_eq!(BlendMode::lookup(b"Multiply"), BlendMode::Multiply);
        assert_eq!(BlendMode::lookup(b"Luminosity"), BlendMode::Luminosity);
        assert_eq!(BlendMode::lookup(b"Compatible"), BlendMode::Normal);
    }

    #[test]
    fn separable_bytes_match_the_c() {
        assert_eq!(separable(BlendMode::Multiply, 255, 128), 128);
        assert_eq!(separable(BlendMode::Screen, 0, 128), 128);
        assert_eq!(separable(BlendMode::Difference, 200, 50), 150);
        assert_eq!(separable(BlendMode::Darken, 200, 50), 50);
        assert_eq!(separable(BlendMode::ColorDodge, 0, 200), 0);
        assert_eq!(separable(BlendMode::ColorBurn, 255, 10), 255);
    }

    #[test]
    fn isolated_multiply_onto_white_is_the_source() {
        // Multiply over white leaves the source colour.
        let mut dst = new_pixmap_with_bbox(IRect::new(0, 0, 1, 1), 3, false);
        dst.samples.copy_from_slice(&[255, 255, 255]);
        let mut src = new_pixmap_with_bbox(IRect::new(0, 0, 1, 1), 4, true);
        src.samples.copy_from_slice(&[255, 255, 0, 255]);
        blend_pixmap(&mut dst, &mut src, 255, BlendMode::Multiply, true, None);
        assert_eq!(dst.samples, vec![255, 255, 0]);
    }

    #[test]
    fn gray_formula() {
        assert_eq!(rgb_to_gray_byte(255, 255, 255), 255);
        assert_eq!(rgb_to_gray_byte(0, 0, 0), 0);
        assert_eq!(rgb_to_gray_byte(255, 0, 0), 77);
    }
}
