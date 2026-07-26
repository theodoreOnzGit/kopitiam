//! Global (Otsu) and adaptive (Sauvola) binarization (Phase 7 preprocessing).
//!
//! Ported from Leptonica `src/binarize.c` (`pixOtsuAdaptiveThreshold` /
//! `pixSauvolaBinarize` / `pixSauvolaGetThreshold` / `pixApplyLocalThreshold`),
//! `src/pix4.c` (`pixSplitDistributionFgBg`), `src/numafunc2.c`
//! (`numaSplitDistribution`, the Otsu histogram split), and `src/convolve.c`
//! (`pixWindowedMean` / `pixWindowedMeanSquare` and their `blockconvAccumLow` /
//! `pixMeanSquareAccum` summed-area tables) plus the mirrored border from
//! `src/pix2.c` (`pixAddMirroredBorder`), at commit `10bdea2`, license
//! **BSD-2-Clause**, © 2001-2020 Leptonica (Dan Bloomberg), vendored read-only
//! at `crates/kopitiam-ocr/vendor/leptonica`. Translated to Rust for KOPITIAM
//! (AGPL-3.0-only); BSD-2-Clause is one-way compatible with AGPLv3 (copyright
//! carried in this header). Close adaptation: the Otsu between-class-variance
//! score, the score-fraction valley search, the Sauvola threshold formula, and
//! the integral-image window statistics follow Leptonica; re-expressed in
//! idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md (`AID-0051`).
//!
//! # What this is — turning gray into ink/paper (Phase 7)
//!
//! Two thresholders, both producing a [`BinaryImage`] (a [`GrayImage`] of
//! [`BINARY_FG`]/[`BINARY_BG`] — see [`crate::image`]):
//!
//! * [`otsu_binarize`] — a single global threshold from the classic Otsu
//!   between-class-variance maximum over the gray histogram. Fast; correct when
//!   illumination is even.
//! * [`sauvola_binarize`] — a per-pixel threshold `t = m·(1 − k·(1 − s/128))`
//!   from the local mean `m` and standard deviation `s` over a sliding window,
//!   computed in `O(1)` per pixel with summed-area tables. The better default
//!   for scanned scientific pages (uneven lighting, show-through), where a
//!   single global threshold loses text in shaded regions.
//!
//! In Leptonica's 1-bpp convention a **set** bit is foreground, chosen by
//! `pixel < threshold` (dark ink); we map that to [`BINARY_FG`] (`0`).
//!
//! # Deferred
//! Only the two thresholding methods on the OCR path are ported. Leptonica's
//! *tiled/adaptive* Otsu (`pixOtsuAdaptiveThreshold` with `sx,sy` sub-tiles and
//! `pixBlockconv` threshold smoothing), background normalization
//! (`pixOtsuThreshOnBackgroundNorm`, `pixContrastNorm`), the tiled Sauvola
//! (`pixSauvolaBinarizeTiled`, `PIXTILING`), and other methods (Niblack, Otsu
//! on double-norm, `pixThresholdToBinary` for arbitrary depths) are **not**
//! ported. Both functions here use a single tile / whole-image window.

use crate::error::{Error, Result};
use crate::image::{BINARY_BG, BINARY_FG, BinaryImage, GrayImage};

// ---------------------------------------------------------------------------
// Otsu global threshold
// ---------------------------------------------------------------------------

/// The default score fraction for [`otsu_binarize`].
///
/// Leptonica's `numaSplitDistribution` (`numafunc2.c:1974`) first finds the
/// index of maximum Otsu score, then — over all contiguous indices whose score
/// is within `scorefract` of that maximum — picks the split at the histogram
/// *minimum* (the valley). `0.0` is the classic Otsu (split exactly at the
/// max-score index); a small positive fraction (Leptonica uses e.g. `0.1` for a
/// global threshold in `pixThreshOnDoubleNorm`) nudges the split into the
/// valley between the peaks. We default to `0.1`, matching Leptonica's own
/// whole-image usage; pass an explicit value to [`otsu_threshold`] to override.
pub const DEFAULT_OTSU_SCORE_FRACT: f32 = 0.1;

/// Binarizes a [`GrayImage`] with a single global Otsu threshold
/// ([`DEFAULT_OTSU_SCORE_FRACT`]).
///
/// Leptonica: the whole-image case of `pixOtsuAdaptiveThreshold` (one tile) →
/// `pixSplitDistributionFgBg` → `numaSplitDistribution` for the threshold, then
/// `pixThresholdToBinary` (`binarize.c` / `pix4.c`). A pixel is foreground
/// ([`BINARY_FG`]) iff its value is `< threshold`, exactly as
/// `pixApplyLocalThreshold`/`pixThresholdToBinary` set the fg bit.
pub fn otsu_binarize(src: &GrayImage) -> BinaryImage {
    let thresh = otsu_threshold(src, DEFAULT_OTSU_SCORE_FRACT);
    threshold_to_binary(src, thresh)
}

/// Computes the global Otsu threshold of a [`GrayImage`] for a given score
/// fraction (see [`DEFAULT_OTSU_SCORE_FRACT`]).
///
/// Leptonica: `pixGetGrayHistogram` (256 bins) → `numaSplitDistribution`
/// (`numafunc2.c:1974`). Returns the threshold `t` such that
/// `pixThresholdToBinary` marks `value < t` as foreground; for a bimodal
/// histogram `t` lands in the valley between the two modes. An empty image
/// yields `0`.
pub fn otsu_threshold(src: &GrayImage, score_fract: f32) -> u8 {
    if src.pixels.is_empty() {
        return 0;
    }
    // Leptonica: pixGetGrayHistogram(pixg, 1) — a 256-bin count histogram.
    let mut histo = [0.0f32; 256];
    for &p in &src.pixels {
        histo[p as usize] += 1.0;
    }
    split_distribution(&histo, score_fract)
}

/// The Otsu split of a 256-bin gray histogram.
///
/// Leptonica: `numaSplitDistribution` (`numafunc2.c:1974`). For each split index
/// `i` (lower part `[0..i]`, upper `[i+1..n-1]`) it computes the between-class
/// variance score `norm · f₁(1−f₁) · (μ₂−μ₁)²` incrementally; then, over the
/// contiguous run of indices scoring within `score_fract` of the maximum, it
/// selects the histogram minimum and returns that index `+ 1` (capped at 255),
/// because thresholding takes the set with values *below* the threshold.
fn split_distribution(histo: &[f32; 256], score_fract: f32) -> u8 {
    const N: usize = 256;
    let sum: f32 = histo.iter().sum();
    if sum <= 0.0 {
        return 0;
    }
    // norm = 4 / (n-1)^2  (numafunc2.c:2005).
    let norm = 4.0f32 / ((N - 1) as f32 * (N - 1) as f32);

    // ave2prev initialized to the mean of the whole histogram
    // (numaGetHistogramStats with startx=0, deltax=1: moment/sum).
    let mut moment = 0.0f32;
    for (i, &y) in histo.iter().enumerate() {
        moment += i as f32 * y;
    }
    let mut ave1prev = 0.0f32;
    let mut ave2prev = moment / sum;
    let mut num1prev = 0.0f32;
    let mut num2prev = sum;

    let mut scores = [0.0f32; N];
    let mut maxscore = 0.0f32;
    let mut maxindex = N / 2; // initialize with something (numafunc2.c:2010)

    for (i, &val) in histo.iter().enumerate() {
        let num1 = num1prev + val;
        let ave1 = if num1 == 0.0 {
            ave1prev
        } else {
            (num1prev * ave1prev + i as f32 * val) / num1
        };
        let num2 = num2prev - val;
        let ave2 = if num2 == 0.0 {
            ave2prev
        } else {
            (num2prev * ave2prev - i as f32 * val) / num2
        };
        let fract1 = num1 / sum;
        let score = norm * (fract1 * (1.0 - fract1)) * (ave2 - ave1) * (ave2 - ave1);
        scores[i] = score;
        if score > maxscore {
            maxscore = score;
            maxindex = i;
        }
        num1prev = num1;
        num2prev = num2;
        ave1prev = ave1;
        ave2prev = ave2;
    }

    // Contiguous range within score_fract of the max, then the histogram min in
    // that range (numafunc2.c:2054).
    let minscore = (1.0 - score_fract) * maxscore;
    let mut minrange = 0usize;
    for i in (0..maxindex).rev() {
        if scores[i] < minscore {
            minrange = i + 1;
            break;
        }
        // If the loop reaches i == 0 without breaking, minrange stays 0.
    }
    let mut maxrange = N - 1;
    for (i, &sc) in scores.iter().enumerate().take(N).skip(maxindex + 1) {
        if sc < minscore {
            maxrange = i - 1;
            break;
        }
    }

    let mut minval = histo[minrange];
    let mut bestsplit = minrange;
    for (i, &v) in histo
        .iter()
        .enumerate()
        .take(maxrange + 1)
        .skip(minrange + 1)
    {
        if v < minval {
            minval = v;
            bestsplit = i;
        }
    }

    // +1 to get the threshold, capped at 255 (numafunc2.c:2080).
    bestsplit.saturating_add(1).min(255) as u8
}

/// Thresholds a [`GrayImage`] to a [`BinaryImage`]: `value < thresh` ⇒
/// foreground.
///
/// Leptonica: `pixThresholdToBinary` / the inner test of
/// `pixApplyLocalThreshold` (`binarize.c:820`) — `if (vals < valt)` sets the
/// foreground bit.
fn threshold_to_binary(src: &GrayImage, thresh: u8) -> BinaryImage {
    let pixels = src
        .pixels
        .iter()
        .map(|&v| if v < thresh { BINARY_FG } else { BINARY_BG })
        .collect();
    GrayImage {
        width: src.width,
        height: src.height,
        pixels,
    }
}

// ---------------------------------------------------------------------------
// Sauvola adaptive threshold
// ---------------------------------------------------------------------------

/// The default window half-size for [`sauvola_binarize`]. The full window is
/// `2·whsize + 1` on a side; Leptonica's own regression test
/// (`prog/binarize_reg.c`) uses `whsize = 7`.
pub const DEFAULT_SAUVOLA_WHSIZE: usize = 7;

/// The default `k` factor for [`sauvola_binarize`]. Leptonica documents `k`
/// typically in `[0.2, 0.5]` (`binarize.c:598`) and uses `0.34` in its
/// regression test (`prog/binarize_reg.c`).
pub const DEFAULT_SAUVOLA_K: f32 = 0.34;

/// Adaptively binarizes a [`GrayImage`] with Sauvola's method.
///
/// For each pixel the threshold is
/// `t = m · (1 − k · (1 − s/128))`
/// where `m` and `s` are the mean and standard deviation over the
/// `(2·whsize+1)²` window centered on the pixel; a pixel is foreground
/// ([`BINARY_FG`]) iff its value is `< t`. This is Leptonica's exact formula
/// (`pixSauvolaGetThreshold`, `binarize.c:773`), equivalently
/// `m · (1 + k·(s/128 − 1))`. `s` is maximized at `127.5` when half the window
/// is `0` and half `255`, so a high-variance (edge/text) region keeps `t` near
/// `m`, while a flat region lowers `t` below `m`.
///
/// The window statistics use summed-area tables (`pixWindowedMean` /
/// `pixWindowedMeanSquare`), so cost is independent of `whsize`. Boundary
/// pixels are handled by a mirrored border of width `whsize + 1`
/// (`pixAddMirroredBorder`, always applied here — Leptonica's `addborder = 1`
/// path), so the output is the same size as the input.
///
/// Sensible defaults are [`DEFAULT_SAUVOLA_WHSIZE`] and [`DEFAULT_SAUVOLA_K`].
///
/// # Errors
/// Returns [`Error::format`] if `whsize < 2`, if `k < 0`, or if the image is
/// too small for the window (`width` or `height` `< 2·whsize + 3`) — the same
/// guards as `pixSauvolaBinarize` (`binarize.c:626`).
pub fn sauvola_binarize(src: &GrayImage, whsize: usize, k: f32) -> Result<BinaryImage> {
    // Leptonica guards (binarize.c:626).
    if whsize < 2 {
        return Err(Error::format("sauvola: whsize must be >= 2"));
    }
    if k < 0.0 {
        return Err(Error::format("sauvola: factor (k) must be >= 0"));
    }
    let (w, h) = (src.width, src.height);
    if w < 2 * whsize + 3 || h < 2 * whsize + 3 {
        return Err(Error::format(format!(
            "sauvola: whsize {whsize} too large for {w}x{h} image (need >= {})",
            2 * whsize + 3
        )));
    }

    // Leptonica: pixAddMirroredBorder(pixs, whsize+1, ...) then windowed stats
    // with hasborder=1 (binarize.c:634).
    let border = whsize + 1;
    let bordered = add_mirrored_border(src, border);

    let mean = windowed_mean(&bordered, whsize, w, h);
    let meansq = windowed_mean_square(&bordered, whsize, w, h);

    // pixSauvolaGetThreshold + pixApplyLocalThreshold (binarize.c:710/790),
    // fused: compute the per-pixel threshold and immediately apply it to src.
    let mut pixels = vec![BINARY_BG; w * h];
    for idx in 0..w * h {
        let mv = mean[idx] as i32;
        let ms = meansq[idx] as i32;
        // var = ms - mv*mv; s = sqrt(var). var is non-negative for a valid
        // window (mv is floor(mean), so mv*mv <= mean^2 <= ms); guard anyway.
        let var = (ms - mv * mv).max(0);
        let sd = (var as f32).sqrt();
        // t = m * (1 - k*(1 - s/128)); C truncates the float to l_int32 then to
        // a byte (SET_DATA_BYTE). Match with f32 -> i32 -> u8.
        let t = (mv as f32 * (1.0 - k * (1.0 - sd / 128.0))) as i32 as u8;
        if src.pixels[idx] < t {
            pixels[idx] = BINARY_FG;
        }
    }

    Ok(GrayImage {
        width: w,
        height: h,
        pixels,
    })
}

/// `pixAddMirroredBorder(pixs, b, b, b, b)` (`pix2.c:2122`): a reflected border
/// of width `b` on every side (edge pixel duplicated: `…c b a | a b c…`).
///
/// Fills the horizontal (left/right) borders across the content rows first,
/// then copies whole (already-bordered) rows for the top/bottom borders, so the
/// corners match Leptonica's fill order.
fn add_mirrored_border(src: &GrayImage, b: usize) -> GrayImage {
    let (w, h) = (src.width, src.height);
    let (pw, ph) = (w + 2 * b, h + 2 * b);
    let mut out = vec![0u8; pw * ph];

    // Content, plus left/right reflection, into rows [b .. b+h).
    for y in 0..h {
        let orow = (y + b) * pw;
        let srow = y * w;
        // content
        for x in 0..w {
            out[orow + b + x] = src.pixels[srow + x];
        }
        // left border col x holds original col (b-1-x); right border col
        // (b+w+j) holds original col (w-1-j).
        for x in 0..b {
            out[orow + x] = src.pixels[srow + (b - 1 - x)];
            out[orow + b + w + x] = src.pixels[srow + (w - 1 - x)];
        }
    }
    // Top border row (b-1-i) copies content-region row (b+i); bottom border row
    // (b+h+i) copies row (b+h-1-i). These copy the full width (with the
    // horizontal borders already filled), giving correct corners.
    for i in 0..b {
        let (top_dst, top_src) = ((b - 1 - i) * pw, (b + i) * pw);
        let (bot_dst, bot_src) = ((b + h + i) * pw, (b + h - 1 - i) * pw);
        out.copy_within(top_src..top_src + pw, top_dst);
        out.copy_within(bot_src..bot_src + pw, bot_dst);
    }

    GrayImage {
        width: pw,
        height: ph,
        pixels: out,
    }
}

/// `pixWindowedMean(pixg, wc, wc, hasborder=1, normflag=1)` (`convolve.c:1055`):
/// the mean over each `(2·wc+1)²` window, output stripped of the `wc+1` border,
/// so `out[i*w + j]` is the mean centered on original pixel `(j, i)`.
///
/// `bordered` already carries the `wc+1` border; `w`,`h` are the *output*
/// (original) dimensions. Uses a `u64` summed-area table (`blockconvAccumLow`),
/// then the `(l_uint8)(norm·sum)` truncation.
fn windowed_mean(bordered: &GrayImage, wc: usize, w: usize, h: usize) -> Vec<u8> {
    let sat = summed_area(bordered);
    let bw = bordered.width;
    let incr = 2 * wc + 1;
    let norm = 1.0f32 / (incr as f32 * incr as f32);
    let mut out = vec![0u8; w * h];
    // out(i,j) window sum = A[i+incr][j+incr] - A[i][j+incr] - A[i+incr][j] + A[i][j]
    for i in 0..h {
        for j in 0..w {
            let s = sat[(i + incr) * bw + (j + incr)] + sat[i * bw + j]
                - sat[i * bw + (j + incr)]
                - sat[(i + incr) * bw + j];
            out[i * w + j] = (norm * s as f32) as u8;
        }
    }
    out
}

/// `pixWindowedMeanSquare(pixg, wc, wc, hasborder=1)` (`convolve.c:1170`): the
/// mean of squared values over each `(2·wc+1)²` window (`pixMeanSquareAccum`
/// `f64` summed-area table), with the `(l_uint32)(norm·sum + 0.5)` rounding.
fn windowed_mean_square(bordered: &GrayImage, wc: usize, w: usize, h: usize) -> Vec<u32> {
    let bw = bordered.width;
    let bh = bordered.height;
    // f64 summed-area table of squared pixel values (pixMeanSquareAccum).
    let mut sat = vec![0.0f64; bw * bh];
    for i in 0..bh {
        for j in 0..bw {
            let v = bordered.pixels[i * bw + j] as f64;
            let sq = v * v;
            let up = if i > 0 { sat[(i - 1) * bw + j] } else { 0.0 };
            let left = if j > 0 { sat[i * bw + (j - 1)] } else { 0.0 };
            let ul = if i > 0 && j > 0 {
                sat[(i - 1) * bw + (j - 1)]
            } else {
                0.0
            };
            sat[i * bw + j] = sq + up + left - ul;
        }
    }
    let incr = 2 * wc + 1;
    let norm = 1.0f64 / (incr as f64 * incr as f64);
    let mut out = vec![0u32; w * h];
    for i in 0..h {
        for j in 0..w {
            let s = sat[(i + incr) * bw + (j + incr)] + sat[i * bw + j]
                - sat[i * bw + (j + incr)]
                - sat[(i + incr) * bw + j];
            out[i * w + j] = (norm * s + 0.5) as u32;
        }
    }
    out
}

/// A `u64` summed-area table of an 8-bit image (Leptonica's `blockconvAccumLow`
/// recursion `a(i,j) = v + a(i-1,j) + a(i,j-1) − a(i-1,j-1)`, `convolve.c:499`).
/// Leptonica accumulates into `l_uint32`; `u64` is identical for non-overflowing
/// sizes and cannot wrap on realistic pages.
fn summed_area(img: &GrayImage) -> Vec<u64> {
    let (w, h) = (img.width, img.height);
    let mut sat = vec![0u64; w * h];
    for i in 0..h {
        for j in 0..w {
            let v = img.pixels[i * w + j] as u64;
            let up = if i > 0 { sat[(i - 1) * w + j] } else { 0 };
            let left = if j > 0 { sat[i * w + (j - 1)] } else { 0 };
            let ul = if i > 0 && j > 0 {
                sat[(i - 1) * w + (j - 1)]
            } else {
                0
            };
            sat[i * w + j] = v + up + left - ul;
        }
    }
    sat
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Otsu ----

    #[test]
    fn otsu_threshold_lands_between_two_modes() {
        // A bimodal image: half the pixels at 60, half at 190. The Otsu split
        // must fall strictly between the two levels.
        let mut pixels = vec![60u8; 50];
        pixels.extend(std::iter::repeat_n(190u8, 50));
        let img = GrayImage::new(10, 10, pixels);
        let t = otsu_threshold(&img, DEFAULT_OTSU_SCORE_FRACT);
        assert!(t > 60 && t <= 190, "threshold {t} not between the modes");
    }

    #[test]
    fn otsu_binarize_splits_foreground_and_background() {
        // Dark text (30) on light paper (220): dark -> fg (0), light -> bg (255).
        let mut pixels = vec![220u8; 64];
        // stamp a block of dark pixels
        for y in 2..6 {
            for x in 2..6 {
                pixels[y * 8 + x] = 30;
            }
        }
        let img = GrayImage::new(8, 8, pixels);
        let bin = otsu_binarize(&img);
        assert_eq!(bin.get(3, 3), Some(BINARY_FG));
        assert_eq!(bin.get(0, 0), Some(BINARY_BG));
    }

    #[test]
    fn otsu_empty_image_is_safe() {
        let img = GrayImage::new(0, 0, vec![]);
        assert_eq!(otsu_threshold(&img, 0.0), 0);
        assert!(otsu_binarize(&img).pixels.is_empty());
    }

    // ---- Sauvola ----

    #[test]
    fn sauvola_rejects_too_small_or_bad_params() {
        let img = GrayImage::filled(10, 10, 128);
        assert!(sauvola_binarize(&img, 1, 0.34).is_err()); // whsize < 2
        assert!(sauvola_binarize(&img, 3, -0.1).is_err()); // k < 0
        assert!(sauvola_binarize(&img, 5, 0.34).is_err()); // window too big
    }

    #[test]
    fn sauvola_separates_text_under_uneven_illumination_where_otsu_fails() {
        // Build a horizontal brightness gradient (left bright ~230, right dark
        // ~120): the "paper" is shaded, so no single global threshold works.
        // Stamp a dark text patch (value 40 below the *local* paper) on the
        // dark-shaded right side, and a light-but-still-paper region on the
        // left. Sauvola must call the patch foreground and the shaded paper
        // background, whereas global Otsu misclassifies.
        let w = 60usize;
        let h = 30usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                // paper gradient: 230 at x=0 down to 120 at x=w-1
                let paper = 230.0 - (x as f32 / (w - 1) as f32) * 110.0;
                pixels[y * w + x] = paper as u8;
            }
        }
        // A dark text patch on the shaded (right) side: local paper there is
        // ~120; make the ink ~70 (clearly darker than local paper, but *lighter*
        // than the bright-side paper on the left, which is ~230 — this is what
        // defeats a global threshold).
        for y in 12..18 {
            for x in 44..50 {
                pixels[y * w + x] = 70;
            }
        }
        let img = GrayImage::new(w, h, pixels);

        let sauvola = sauvola_binarize(&img, DEFAULT_SAUVOLA_WHSIZE, DEFAULT_SAUVOLA_K).unwrap();
        // The dark patch on the shaded side is foreground under Sauvola.
        assert_eq!(
            sauvola.get(46, 15),
            Some(BINARY_FG),
            "Sauvola must mark the shaded-region ink as foreground"
        );
        // The bright-side paper (value ~230) is background under Sauvola.
        assert_eq!(
            sauvola.get(5, 15),
            Some(BINARY_BG),
            "Sauvola must keep bright paper as background"
        );
        // The shaded-side paper near the patch (but not on it) is background.
        assert_eq!(
            sauvola.get(20, 2),
            Some(BINARY_BG),
            "Sauvola must keep shaded paper as background"
        );

        // Now show global Otsu fails: its single threshold sits somewhere in the
        // gradient, so a large swath of the darker-but-still-paper gradient gets
        // misread as foreground. Count fg pixels off the text patch.
        let otsu = otsu_binarize(&img);
        let mut otsu_paper_fg = 0;
        let mut sauvola_paper_fg = 0;
        for y in 0..h {
            for x in 0..w {
                let on_patch = (12..18).contains(&y) && (44..50).contains(&x);
                if on_patch {
                    continue;
                }
                if otsu.pixels[y * w + x] == BINARY_FG {
                    otsu_paper_fg += 1;
                }
                if sauvola.pixels[y * w + x] == BINARY_FG {
                    sauvola_paper_fg += 1;
                }
            }
        }
        assert!(
            otsu_paper_fg > sauvola_paper_fg,
            "global Otsu should misclassify more shaded paper as fg \
             (otsu={otsu_paper_fg}, sauvola={sauvola_paper_fg})"
        );
    }

    #[test]
    fn sauvola_flat_field_is_all_background() {
        // A perfectly flat gray field: local std = 0, so t = m; no pixel is
        // strictly < its own local mean, so everything is background.
        let img = GrayImage::filled(40, 40, 128);
        let bin = sauvola_binarize(&img, DEFAULT_SAUVOLA_WHSIZE, DEFAULT_SAUVOLA_K).unwrap();
        assert!(bin.pixels.iter().all(|&p| p == BINARY_BG));
        assert_eq!((bin.width, bin.height), (40, 40));
    }

    #[test]
    fn mirrored_border_reflects_edges() {
        // 2x2 image, border 1: content preserved, edges reflected (edge dup).
        let img = GrayImage::new(2, 2, vec![10, 20, 30, 40]);
        let b = add_mirrored_border(&img, 1);
        assert_eq!((b.width, b.height), (4, 4));
        // content at (1,1)..(2,2)
        assert_eq!(b.get(1, 1), Some(10));
        assert_eq!(b.get(2, 1), Some(20));
        // left border of top content row reflects col 0 (value 10)
        assert_eq!(b.get(0, 1), Some(10));
        // right border reflects col 1 (value 20)
        assert_eq!(b.get(3, 1), Some(20));
        // top border reflects row 0; corner (0,0) reflects (0,1)->10
        assert_eq!(b.get(0, 0), Some(10));
        assert_eq!(b.get(3, 3), Some(40));
    }
}
