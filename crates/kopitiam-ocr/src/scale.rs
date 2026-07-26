//! Grayscale linear-interpolation scaling (Phase 7 preprocessing).
//!
//! Ported from Leptonica `src/scale1.c` (`pixScaleGrayLI` and its inner
//! `scaleGrayLILow`) at commit `10bdea2`, license **BSD-2-Clause**, © 2001-2020
//! Leptonica (Dan Bloomberg), vendored read-only at
//! `crates/kopitiam-ocr/vendor/leptonica`. Translated to Rust for KOPITIAM
//! (AGPL-3.0-only); BSD-2-Clause is one-way compatible with AGPLv3 (copyright
//! carried in this header). Close adaptation: the fixed-point 1/16-subpixel
//! bilinear weighting, the edge-replication cases, and the `+128`/256 rounding
//! follow Leptonica exactly. See docs/ACKNOWLEDGEMENTS.md (`AID-0051`).
//!
//! # What this is — the faithful gray resampler (Phase 7)
//!
//! Line normalization scales each text line to the network's input height. The
//! recognizer (Phase 6, `lstmrecognizer.rs`) currently approximates Leptonica's
//! `pixScale` with a float bilinear resample; this module exposes Leptonica's
//! actual `pixScaleGrayLI` — linear interpolation of the 4 neighboring source
//! pixels using 4-bit (1/16-pixel) fixed-point coordinates — so that path can
//! be made bit-faithful later.
//!
//! `pixScaleGrayLI` is appropriate for upscaling and modest downscaling (scale
//! factor >= 0.7); for large reductions Leptonica switches to area mapping,
//! which is **not** ported here (see the module-level deferral note). This is
//! the general LI resampler only.

use crate::image::GrayImage;

/// Scales a [`GrayImage`] to `dst_width` × `dst_height` with Leptonica's
/// grayscale linear interpolation (`pixScaleGrayLI`).
///
/// This is the target-dimension form of [`scale_gray_li`]: the scale factors
/// are `dst_width / src_width` and `dst_height / src_height`. Returns an
/// identity clone when the target equals the source, and an empty image when
/// any dimension is zero.
///
/// Leptonica: callers reach `scaleGrayLILow` via `pixScaleGrayLI(pixs, scalex,
/// scaley)` (`scale1.c:766`), which computes `wd = (int)(scalex*ws + 0.5)`.
/// Here the destination size is given directly, so no such rounding is needed.
///
/// # Note on the 0.7 lower bound
/// `pixScaleGrayLI` warns and falls back to area mapping below a 0.7 scale
/// factor (large reductions alias under pure LI). That fallback is not ported;
/// this function always applies LI, matching `scaleGrayLILow` itself. For the
/// line-height normalization on the OCR path the factor is at or above ~1.
pub fn scale_gray(src: &GrayImage, dst_width: usize, dst_height: usize) -> GrayImage {
    if dst_width == 0 || dst_height == 0 || src.width == 0 || src.height == 0 {
        return GrayImage::new(dst_width, dst_height, vec![0u8; dst_width * dst_height]);
    }
    // Leptonica: pixScaleGrayLI fast path — scalex == scaley == 1.0 returns a
    // copy (scale1.c:785).
    if dst_width == src.width && dst_height == src.height {
        return src.clone();
    }
    scale_gray_li_low(src, dst_width, dst_height)
}

/// Scales a [`GrayImage`] by independent `scale_x` / `scale_y` factors with
/// Leptonica's grayscale linear interpolation (`pixScaleGrayLI`).
///
/// The destination size is `round(scale_x·width)` × `round(scale_y·height)`
/// (each at least 1), matching `wd = (l_int32)(scalex*ws + 0.5)` in
/// `pixScaleGrayLI` (`scale1.c:796`). Delegates to [`scale_gray`].
pub fn scale_gray_li(src: &GrayImage, scale_x: f32, scale_y: f32) -> GrayImage {
    let dst_w = ((scale_x * src.width as f32) + 0.5) as usize;
    let dst_h = ((scale_y * src.height as f32) + 0.5) as usize;
    scale_gray(src, dst_w.max(1), dst_h.max(1))
}

/// The inner loop of Leptonica's `scaleGrayLILow` (`scale1.c:2355`).
///
/// Source coordinates are carried in 1/16-pixel fixed point: `scx = 16·ws/wd`
/// maps a destination column to `(xp, xf)` = (integer src column, 4-bit
/// fraction). The four bilinear weights are `(16−xf)(16−yf)`, `xf(16−yf)`,
/// `(16−xf)yf`, `xf·yf` (they sum to 256); the accumulated value is rounded with
/// `(sum + 128) / 256`. Near the right/bottom edges the missing neighbors are
/// replicated from `v00` exactly as Leptonica does, so no source read leaves the
/// `ws`×`hs` buffer.
fn scale_gray_li_low(src: &GrayImage, wd: usize, hd: usize) -> GrayImage {
    let ws = src.width;
    let hs = src.height;
    let s = &src.pixels;
    let mut out = vec![0u8; wd * hd];

    // scx, scy: dest coords -> src coords, in 1/16-pixel units (scale1.c:2376).
    let scx = 16.0f32 * ws as f32 / wd as f32;
    let scy = 16.0f32 * hs as f32 / hd as f32;
    // wm2/hm2 = ws-2/hs-2: the last index at which both forward neighbors exist.
    // Use i32 so a 1-pixel source (wm2 == -1) compares correctly.
    let wm2 = ws as i32 - 2;
    let hm2 = hs as i32 - 2;

    for i in 0..hd {
        let ypm = (scy * i as f32) as i32;
        let yp = (ypm >> 4) as usize;
        let yf = ypm & 0x0f;
        let row = yp * ws;
        let row_next = row + ws; // valid iff yp <= hm2 (guarded below)
        for j in 0..wd {
            let xpm = (scx * j as f32) as i32;
            let xp = (xpm >> 4) as usize;
            let xf = xpm & 0x0f;

            // Bilinear neighbor selection with edge replication (scale1.c:2397).
            let v00_val = s[row + xp] as i32;
            let (v10_val, v01_val, v11_val);
            if xp as i32 > wm2 || yp as i32 > hm2 {
                if yp as i32 > hm2 && xp as i32 <= wm2 {
                    // pixels near the bottom row
                    v01_val = v00_val;
                    v10_val = s[row + xp + 1] as i32;
                    v11_val = v10_val;
                } else if xp as i32 > wm2 && (yp as i32) <= hm2 {
                    // pixels near the right side
                    v01_val = s[row_next + xp] as i32;
                    v10_val = v00_val;
                    v11_val = v01_val;
                } else {
                    // lower-right corner
                    v10_val = v00_val;
                    v01_val = v00_val;
                    v11_val = v00_val;
                }
            } else {
                v10_val = s[row + xp + 1] as i32;
                v01_val = s[row_next + xp] as i32;
                v11_val = s[row_next + xp + 1] as i32;
            }

            let v00 = (16 - xf) * (16 - yf) * v00_val;
            let v10 = xf * (16 - yf) * v10_val;
            let v01 = (16 - xf) * yf * v01_val;
            let v11 = xf * yf * v11_val;

            out[i * wd + j] = ((v00 + v01 + v10 + v11 + 128) / 256) as u8;
        }
    }

    GrayImage {
        width: wd,
        height: hd,
        pixels: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_target_equals_source() {
        let img = GrayImage::new(3, 2, vec![10, 20, 30, 40, 50, 60]);
        let out = scale_gray(&img, 3, 2);
        assert_eq!(out, img);
    }

    #[test]
    fn scale_factor_one_is_identity() {
        let img = GrayImage::new(4, 3, (0..12).map(|v| v as u8 * 10).collect());
        assert_eq!(scale_gray_li(&img, 1.0, 1.0), img);
    }

    #[test]
    fn upscale_dimensions_are_correct() {
        let img = GrayImage::new(2, 2, vec![0, 0, 0, 0]);
        let out = scale_gray(&img, 5, 7);
        assert_eq!((out.width, out.height), (5, 7));
        assert_eq!(out.pixels.len(), 35);
    }

    #[test]
    fn constant_image_upscales_to_the_same_constant() {
        // A flat field must interpolate to the same value everywhere.
        let img = GrayImage::filled(3, 3, 128);
        let out = scale_gray(&img, 9, 5);
        assert!(out.pixels.iter().all(|&p| p == 128));
    }

    #[test]
    fn upscaled_ramp_interpolates_between_endpoints() {
        // A 1-D horizontal ramp 0,255 upscaled 2x1 -> the left dest pixels sit
        // over the left source pixel; interior values fall between 0 and 255 and
        // are monotonically non-decreasing.
        let img = GrayImage::new(2, 1, vec![0, 255]);
        let out = scale_gray(&img, 6, 1);
        assert_eq!((out.width, out.height), (6, 1));
        assert_eq!(out.pixels[0], 0); // sample over src col 0
        for w in out.pixels.windows(2) {
            assert!(w[1] >= w[0], "ramp must be non-decreasing: {:?}", out.pixels);
        }
        // The last dest pixel samples near the right source pixel (value 255).
        assert!(*out.pixels.last().unwrap() > 0);
    }

    #[test]
    fn downscale_produces_requested_size() {
        let img = GrayImage::new(8, 8, (0..64).map(|v| v as u8).collect());
        let out = scale_gray(&img, 6, 6); // factor 0.75, within LI's >= 0.7 range
        assert_eq!((out.width, out.height), (6, 6));
    }

    #[test]
    fn upscale_midpoint_of_two_pixels_is_the_average() {
        // 2x1 source {0, 200} scaled to 4x1: with scx = 16*2/4 = 8, dest col 1
        // maps to src coord 8/16 = 0.5 -> xp=0, xf=8, i.e. the exact midpoint,
        // giving (8*0 + 8*200 + 128)/256 = 1728/256 = 6 ... verify against the
        // fixed-point formula.
        let img = GrayImage::new(2, 1, vec![0, 200]);
        let out = scale_gray(&img, 4, 1);
        // col0: xf=0 -> 0; col1: xf=8 -> (8*16-... ) compute exactly.
        // col1: xpm = 8*1 = 8 -> xp=0, xf=8. yf=0.
        //   v00 = 16*16*0 = 0; v10 = 8*16*200 = 25600; v01=0; v11=0.
        //   (0+0+25600+0+128)/256 = 25728/256 = 100.
        assert_eq!(out.pixels[0], 0);
        assert_eq!(out.pixels[1], 100);
    }
}
