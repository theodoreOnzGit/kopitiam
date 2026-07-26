//! Page-image types and RGB→grayscale conversion (Phase 7 preprocessing).
//!
//! Ported from Leptonica `src/pixconv.c` (`pixConvertRGBToGray`) and the depth
//! definitions in `src/pix.h` at commit `10bdea2`, license **BSD-2-Clause**,
//! © 2001-2020 Leptonica (Dan Bloomberg). Leptonica is vendored read-only at
//! `crates/kopitiam-ocr/vendor/leptonica`. Translated to Rust for KOPITIAM
//! (AGPL-3.0-only): BSD-2-Clause is one-way compatible with AGPLv3, so
//! Leptonica's algorithms adapt into this AGPLv3 work provided the copyright
//! notice travels with the code (this header). Close adaptation, not clean-room:
//! the exact luma weights and the round-to-nearest packing follow Leptonica,
//! re-expressed idiomatically. See docs/ACKNOWLEDGEMENTS.md (`docs/ai-decisions/
//! AID-0051`) for the crate-wide provenance discipline.
//!
//! # What this is — the image types on the OCR preprocessing path (Phase 7)
//!
//! The recognizer (Phase 6) consumes a single normalized [`GrayLine`]. Getting
//! there from a rendered page raster is: (1) color → 8-bit gray, (2) binarize +
//! find lines, (3) scale each line to the network's input height. This module
//! holds the image containers and step (1); binarization lives in
//! [`crate::binarize`] and scaling in [`crate::scale`].
//!
//! Following the same discipline as [`GrayLine`]: **raw 8-bit buffers, no
//! `image`-crate dependency** (Android-safe). Leptonica's `Pix` is a packed,
//! reference-counted, colormap-carrying container; we port only the flat 8-bit
//! and RGB(A) rasters the LSTM path actually needs — not the `Pix` API.
//!
//! [`GrayLine`]: crate::lstmrecognizer::GrayLine

/// A row-major 8-bit grayscale image: `height` rows of `width` pixels
/// (`pixels[y * width + x]`, `0` = black … `255` = white).
///
/// This is Leptonica's 8-bpp `Pix` reduced to a flat buffer — the working
/// representation for binarization and scaling. It is the same shape as
/// [`GrayLine`](crate::lstmrecognizer::GrayLine) (which is the recognizer's
/// single-line input); a page `GrayImage` is cropped into `GrayLine`s by the
/// later line-finding phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrayImage {
    /// Image width in pixels (number of columns).
    pub width: usize,
    /// Image height in pixels (number of rows).
    pub height: usize,
    /// Row-major 8-bit gray pixels, length `width * height`.
    pub pixels: Vec<u8>,
}

impl GrayImage {
    /// A grayscale image from its dimensions and row-major pixel buffer.
    ///
    /// # Panics
    /// If `pixels.len() != width * height`.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            width * height,
            "GrayImage pixel buffer length must be width * height"
        );
        GrayImage {
            width,
            height,
            pixels,
        }
    }

    /// A `width * height` grayscale image filled with `value`.
    pub fn filled(width: usize, height: usize, value: u8) -> Self {
        GrayImage {
            width,
            height,
            pixels: vec![value; width * height],
        }
    }

    /// The pixel at `(x, y)`, or `None` if out of range.
    pub fn get(&self, x: usize, y: usize) -> Option<u8> {
        if x < self.width && y < self.height {
            Some(self.pixels[y * self.width + x])
        } else {
            None
        }
    }
}

/// The two levels of a [`BinaryImage`]: foreground (ink) and background (paper).
///
/// Leptonica binarizes to a 1-bpp `Pix` where a **set** bit (`1`) marks
/// foreground — the dark ink a threshold selects with `value < threshold`
/// (`pixThresholdToBinary`/`pixApplyLocalThreshold`, `binarize.c`). We keep a
/// [`GrayImage`] rather than a packed bitmap (see [`BinaryImage`]) and map that
/// set bit to [`BINARY_FG`] (`0`, black) and a clear bit to [`BINARY_BG`]
/// (`255`, white), so a binary result is also a valid grayscale image.
pub const BINARY_FG: u8 = 0;

/// The background (paper / clear-bit) level of a [`BinaryImage`]. See
/// [`BINARY_FG`].
pub const BINARY_BG: u8 = 255;

/// A bilevel image: a [`GrayImage`] whose pixels are all [`BINARY_FG`] (`0`,
/// ink) or [`BINARY_BG`] (`255`, paper).
///
/// # Why a `GrayImage`, not a packed 1-bpp bitmap
///
/// Leptonica's binary `Pix` is a packed MSB-first bitmap (32-bit words, 1 bit
/// per pixel). Packing is a space optimization that the downstream LSTM path
/// does not need — line-finding and normalization operate on byte rasters — so
/// we represent binary images as the `{0, 255}` subset of [`GrayImage`]. This
/// keeps one raster type through the pipeline and avoids porting the bit-packing
/// machinery (`arrayaccess.h`, the `SET_DATA_BIT` family). Packed 1-bpp is
/// deferred; if a morphology phase later needs it, it can be added without
/// changing these signatures.
pub type BinaryImage = GrayImage;

/// A source of per-pixel RGB triplets, so [`to_gray`] serves both packed RGB
/// and RGBA rasters without an intermediate copy.
///
/// Leptonica's `pixConvertRGBToGray` reads a 32-bpp word and masks the R/G/B
/// bytes; both [`RgbImage`] (3 bytes/pixel) and [`RgbaImage`] (4 bytes/pixel,
/// alpha ignored) present the same three channels here.
pub trait RgbSource {
    /// Image width in pixels.
    fn width(&self) -> usize;
    /// Image height in pixels.
    fn height(&self) -> usize;
    /// The `(r, g, b)` bytes at `(x, y)`. Callers stay within `width`×`height`.
    fn pixel_rgb(&self, x: usize, y: usize) -> (u8, u8, u8);
}

/// A row-major 24-bit RGB image: 3 interleaved bytes per pixel (`R, G, B`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Row-major `R, G, B` bytes, length `width * height * 3`.
    pub pixels: Vec<u8>,
}

impl RgbImage {
    /// An RGB image from its dimensions and interleaved `R, G, B` buffer.
    ///
    /// # Panics
    /// If `pixels.len() != width * height * 3`.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            width * height * 3,
            "RgbImage pixel buffer length must be width * height * 3"
        );
        RgbImage {
            width,
            height,
            pixels,
        }
    }
}

impl RgbSource for RgbImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_rgb(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let i = (y * self.width + x) * 3;
        (self.pixels[i], self.pixels[i + 1], self.pixels[i + 2])
    }
}

/// A row-major 32-bit RGBA image: 4 interleaved bytes per pixel (`R, G, B, A`).
/// The alpha channel is ignored by [`to_gray`] (Leptonica masks only R/G/B).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Row-major `R, G, B, A` bytes, length `width * height * 4`.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// An RGBA image from its dimensions and interleaved `R, G, B, A` buffer.
    ///
    /// # Panics
    /// If `pixels.len() != width * height * 4`.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            width * height * 4,
            "RgbaImage pixel buffer length must be width * height * 4"
        );
        RgbaImage {
            width,
            height,
            pixels,
        }
    }
}

impl RgbSource for RgbaImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_rgb(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let i = (y * self.width + x) * 4;
        (self.pixels[i], self.pixels[i + 1], self.pixels[i + 2])
    }
}

// Leptonica: the perceptual luma weights (pix.h:370). These are the exact
// constants `pixConvertRGBToGray` substitutes when the caller passes 0/0/0, and
// they sum to 1.0 so the weighted average cannot overflow a byte.
/// `L_RED_WEIGHT` — perceptual weight for the red channel (Leptonica `pix.h`).
pub const L_RED_WEIGHT: f32 = 0.3;
/// `L_GREEN_WEIGHT` — perceptual weight for the green channel.
pub const L_GREEN_WEIGHT: f32 = 0.5;
/// `L_BLUE_WEIGHT` — perceptual weight for the blue channel.
pub const L_BLUE_WEIGHT: f32 = 0.2;

/// Converts an RGB(A) image to 8-bit grayscale with Leptonica's perceptual luma
/// weights (`0.3·R + 0.5·G + 0.2·B`, rounded to nearest).
///
/// Leptonica: `pixConvertRGBToGray(pixs, 0, 0, 0)` (`pixconv.c:814`), the
/// default-weight path. Each output byte is
/// `(l_int32)(rwt*r + gwt*g + bwt*b + 0.5)` with the [`L_RED_WEIGHT`] /
/// [`L_GREEN_WEIGHT`] / [`L_BLUE_WEIGHT`] constants; the `+ 0.5` then truncation
/// is round-half-up. Because the weights sum to exactly `1.0`, the result is in
/// `[0, 255]` and needs no clamp. Alpha (for [`RgbaImage`]) is ignored, as in
/// Leptonica's `L_RED_SHIFT`/`L_GREEN_SHIFT`/`L_BLUE_SHIFT` byte masks.
pub fn to_gray<S: RgbSource>(src: &S) -> GrayImage {
    let (w, h) = (src.width(), src.height());
    let mut pixels = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = src.pixel_rgb(x, y);
            // Leptonica: pixConvertRGBToGray (pixconv.c:863) — weighted sum with
            // the +0.5 round, truncated to a byte.
            let val =
                L_RED_WEIGHT * r as f32 + L_GREEN_WEIGHT * g as f32 + L_BLUE_WEIGHT * b as f32 + 0.5;
            pixels[y * w + x] = val as u8;
        }
    }
    GrayImage {
        width: w,
        height: h,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_gray_matches_leptonica_weights_on_a_known_pixel() {
        // r=100, g=150, b=200: 0.3*100 + 0.5*150 + 0.2*200 + 0.5
        // = 30 + 75 + 40 + 0.5 = 145.5 -> truncates to 145.
        let img = RgbImage::new(1, 1, vec![100, 150, 200]);
        let gray = to_gray(&img);
        assert_eq!(gray.pixels, vec![145]);
    }

    #[test]
    fn to_gray_pure_channels_give_the_weights_times_255() {
        // Pure red -> 0.3*255 + 0.5 = 77.0 -> 77.
        // Pure green -> 0.5*255 + 0.5 = 128.0 -> 128.
        // Pure blue -> 0.2*255 + 0.5 = 51.5 -> 51.
        let img = RgbImage::new(3, 1, vec![255, 0, 0, 0, 255, 0, 0, 0, 255]);
        let gray = to_gray(&img);
        assert_eq!(gray.pixels, vec![77, 128, 51]);
    }

    #[test]
    fn to_gray_white_and_black_are_preserved() {
        let img = RgbImage::new(2, 1, vec![255, 255, 255, 0, 0, 0]);
        let gray = to_gray(&img);
        // white: 0.3*255+0.5*255+0.2*255+0.5 = 255.5 -> 255.
        assert_eq!(gray.pixels, vec![255, 0]);
    }

    #[test]
    fn to_gray_ignores_alpha_for_rgba() {
        let rgb = RgbImage::new(1, 1, vec![100, 150, 200]);
        let rgba = RgbaImage::new(1, 1, vec![100, 150, 200, 7]);
        assert_eq!(to_gray(&rgb).pixels, to_gray(&rgba).pixels);
    }

    #[test]
    fn weights_sum_to_one() {
        assert!((L_RED_WEIGHT + L_GREEN_WEIGHT + L_BLUE_WEIGHT - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gray_get_is_bounds_checked() {
        let img = GrayImage::new(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(img.get(1, 1), Some(4));
        assert_eq!(img.get(2, 0), None);
        assert_eq!(img.get(0, 2), None);
    }
}
