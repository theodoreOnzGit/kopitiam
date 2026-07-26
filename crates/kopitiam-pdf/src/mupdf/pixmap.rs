//! Ported from MuPDF `source/fitz/pixmap.c` (`fz_new_pixmap_with_data`,
//! `fz_clear_pixmap`, `fz_clear_pixmap_with_value`, `fz_pixmap_bbox`) and the
//! `fz_pixmap` layout in `include/mupdf/fitz/pixmap.h` (commit 19f1284,
//! AGPL-3.0, © Artifex Software, Inc.), translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the layout and clear/bbox semantics follow
//! MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # The raster target
//!
//! [`Pixmap`] is the pixel buffer the [`DrawDevice`](super::draw_device) paints
//! into and the terminal PDF viewer (`tdf-parity`) / kmux LaTeX live-preview
//! consume. It is the extraction-relevant slice of MuPDF's `fz_pixmap`: the
//! sample geometry (`x`/`y` origin, `w`/`h`, `n` components, `stride`), the
//! interleaved 8-bit `samples`, and the alpha flag. The spot-colour (`s`),
//! separations, ICC `colorspace` handle, and the `xres`/`yres` bookkeeping of
//! the C struct are dropped -- the draw device works in DeviceGray / DeviceRGB
//! only, with the last component optionally an alpha (see [`DrawDevice`]).
//!
//! [`DrawDevice`]: super::draw_device::DrawDevice

use super::geometry::IRect;

/// An 8-bit interleaved raster image: `n` components per pixel, row-major,
/// `stride` bytes per row. Mirrors the sample-carrying subset of `fz_pixmap`.
///
/// Component layouts the draw device uses:
/// * `n = 1`, `alpha = false` -- DeviceGray.
/// * `n = 3`, `alpha = false` -- DeviceRGB (the default draw target).
/// * `n = 4`, `alpha = true` -- DeviceRGB + alpha (premultiplied is *not*
///   assumed here; the painter composites straight source-over).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pixmap {
    /// The x origin of the pixmap in device space (`fz_pixmap.x`).
    pub x: i32,
    /// The y origin of the pixmap in device space (`fz_pixmap.y`).
    pub y: i32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
    /// Components per pixel, **including** the alpha component if `alpha`.
    pub n: u8,
    /// Whether the final component is an alpha channel.
    pub alpha: bool,
    /// Bytes per row (`w * n`; no extra padding in this port).
    pub stride: usize,
    /// The interleaved 8-bit samples (`stride * h` bytes).
    pub samples: Vec<u8>,
}

impl Pixmap {
    // MuPDF: fz_new_pixmap (pixmap.c:158) -- samples zeroed (transparent black).
    /// Create an `w×h` pixmap with `n` components (`alpha` = last is alpha),
    /// origin `(0, 0)`, all samples zeroed.
    pub fn new(w: u32, h: u32, n: u8, alpha: bool) -> Pixmap {
        assert!(n >= 1, "pixmap needs at least one component");
        let stride = w as usize * n as usize;
        Pixmap {
            x: 0,
            y: 0,
            w,
            h,
            n,
            alpha,
            stride,
            samples: vec![0u8; stride * h as usize],
        }
    }

    /// Create an opaque DeviceRGB (`n = 3`) pixmap, samples zeroed (black).
    pub fn new_rgb(w: u32, h: u32) -> Pixmap {
        Pixmap::new(w, h, 3, false)
    }

    /// Create a DeviceGray (`n = 1`) pixmap, samples zeroed.
    pub fn new_gray(w: u32, h: u32) -> Pixmap {
        Pixmap::new(w, h, 1, false)
    }

    // MuPDF: fz_clear_pixmap_with_value (pixmap.c:562) -- byte-fill every sample.
    /// Fill every sample byte with `value` (e.g. `0xff` for white on an opaque
    /// colour pixmap, `0x00` for black / transparent).
    pub fn clear_with_value(&mut self, value: u8) {
        self.samples.iter_mut().for_each(|s| *s = value);
    }

    // MuPDF: fz_clear_pixmap (pixmap.c:542) -- zero all samples.
    /// Zero every sample (black, or transparent if the pixmap has alpha).
    pub fn clear(&mut self) {
        self.clear_with_value(0);
    }

    // MuPDF: fz_pixmap_bbox (pixmap.c:238).
    /// The pixmap's bounding box in device space.
    pub fn bbox(&self) -> IRect {
        IRect::new(self.x, self.y, self.x + self.w as i32, self.y + self.h as i32)
    }

    /// Width in pixels (`fz_pixmap_width`).
    pub fn width(&self) -> u32 {
        self.w
    }

    /// Height in pixels (`fz_pixmap_height`).
    pub fn height(&self) -> u32 {
        self.h
    }

    /// Byte offset of pixel `(px, py)` (device coordinates) into `samples`, or
    /// `None` if it lies outside the pixmap.
    #[inline]
    pub fn offset(&self, px: i32, py: i32) -> Option<usize> {
        if px < self.x || py < self.y || px >= self.x + self.w as i32 || py >= self.y + self.h as i32 {
            return None;
        }
        let col = (px - self.x) as usize;
        let row = (py - self.y) as usize;
        Some(row * self.stride + col * self.n as usize)
    }

    /// The `n` component bytes of pixel `(px, py)`, or `None` if out of bounds.
    pub fn pixel(&self, px: i32, py: i32) -> Option<&[u8]> {
        let o = self.offset(px, py)?;
        Some(&self.samples[o..o + self.n as usize])
    }

    /// Rec. 601 luma (0..=255) of pixel `(px, py)`, treating the first up-to-3
    /// components as gray/RGB. Gray pixmaps return the single component; RGB use
    /// `0.299 R + 0.587 G + 0.114 B`. Out-of-bounds returns `None`.
    pub fn luma(&self, px: i32, py: i32) -> Option<u8> {
        let p = self.pixel(px, py)?;
        let v = match self.n {
            1 => p[0] as f32,
            _ => 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32,
        };
        Some(v.round().clamp(0.0, 255.0) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rgb_is_zeroed_and_sized() {
        let p = Pixmap::new_rgb(4, 3);
        assert_eq!((p.w, p.h, p.n, p.alpha), (4, 3, 3, false));
        assert_eq!(p.stride, 12);
        assert_eq!(p.samples.len(), 36);
        assert!(p.samples.iter().all(|&b| b == 0));
        assert_eq!(p.bbox(), IRect::new(0, 0, 4, 3));
    }

    #[test]
    fn clear_white_then_luma() {
        let mut p = Pixmap::new_rgb(2, 2);
        p.clear_with_value(0xff);
        assert_eq!(p.luma(0, 0), Some(255));
        assert_eq!(p.luma(1, 1), Some(255));
        // Out of bounds.
        assert_eq!(p.luma(2, 0), None);
        assert_eq!(p.pixel(-1, 0), None);
    }

    #[test]
    fn offset_matches_layout() {
        let p = Pixmap::new_rgb(3, 2);
        assert_eq!(p.offset(0, 0), Some(0));
        assert_eq!(p.offset(2, 0), Some(6));
        assert_eq!(p.offset(0, 1), Some(9));
        assert_eq!(p.offset(3, 0), None);
    }
}
