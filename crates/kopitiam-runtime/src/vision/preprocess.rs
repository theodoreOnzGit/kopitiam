//! Turning a rasterized page into the tensor a vision tower expects.
//!
//! This is step 1 of the vision path (see [`super`]): everything between "we
//! have pixels" and "the patch-embedding matmul can run". It is deliberately
//! **pure arithmetic over pixel buffers** — no model, no weights, no GGUF — so
//! it is fully testable today, before any vision-capable runtime exists.
//!
//! # The pipeline, and why each step is where it is
//!
//! ```text
//! RGB8 pixels ──resize──▶ square canvas ──normalize──▶ CHW f32 ──patchify──▶ [n_patches, patch_dim]
//! ```
//!
//! * **Resize** to a fixed square. Vision towers have a *fixed* input geometry
//!   baked into their learned position embeddings — there is one embedding per
//!   patch slot, so the patch grid cannot vary at inference without also
//!   interpolating those embeddings. Fixing the canvas keeps that whole problem
//!   out of scope.
//! * **Normalize** to the mean/std the tower was *trained* with. Getting these
//!   constants wrong does not error — it silently shifts every activation and
//!   quietly degrades the output, which is why they are named constants with
//!   their provenance recorded rather than magic numbers inline.
//! * **Patchify** into non-overlapping `patch × patch` tiles, each flattened to
//!   a row. That row-major matrix is exactly the operand a patch-embedding
//!   matmul wants.
//!
//! # Why there is no `conv2d` here
//!
//! ViT patch embedding is conventionally written as a `conv2d` with
//! `kernel = stride = patch_size`. With **non-overlapping** patches that
//! convolution is *identical* to flattening each tile and doing one matmul
//! against the reshaped kernel — every input element lands in exactly one
//! output window, so there is no sliding-window reuse for a convolution to
//! exploit. `kopitiam-tensor` has `matmul` but no `conv2d`, and this identity is
//! why that costs us nothing. (It stops being true for overlapping patches; a
//! tower using `stride < kernel` would genuinely need a convolution.)
//!
//! # Channel order: CHW, not HWC
//!
//! Pixels arrive interleaved (`RGBRGBRGB…`, i.e. HWC) and leave planar
//! (`RRR…GGG…BBB…`, i.e. CHW), because that is the layout the tower's weights
//! are stored against. Mixing these up is the classic silent bug in this area:
//! the shapes match, nothing errors, and the model sees colour noise.

use kopitiam_tensor::Tensor;

/// Per-channel mean used by SigLIP-family vision towers (the encoder SmolVLM
/// uses), in the 0..1 range, RGB order.
///
/// Provenance: SigLIP's published preprocessing normalizes with mean 0.5 and
/// std 0.5 per channel — i.e. it maps 0..1 to −1..1 — rather than the ImageNet
/// statistics that CLIP-family towers use. Recorded as a constant because a
/// wrong value here is invisible: no error, just degraded output.
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];

/// Per-channel standard deviation matching [`SIGLIP_MEAN`].
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// An 8-bit RGB image, row-major and interleaved (`RGBRGB…`), as a rasterizer
/// hands it over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb8 {
    pub width: usize,
    pub height: usize,
    /// `width · height · 3` bytes, row-major, channels interleaved.
    pub data: Vec<u8>,
}

impl Rgb8 {
    /// Wraps a buffer, checking it is exactly the size the dimensions imply.
    ///
    /// Checked rather than assumed because the alternative is an out-of-bounds
    /// panic deep inside resampling, where the message tells you nothing about
    /// the real mistake (a stride miscalculation at the call site).
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> std::result::Result<Self, String> {
        let want = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| format!("image dimensions {width}x{height} overflow"))?;
        if data.len() != want {
            return Err(format!(
                "RGB8 buffer is {} bytes, but {width}x{height}x3 needs {want}",
                data.len()
            ));
        }
        Ok(Self { width, height, data })
    }

    /// Reads the pixel at `(x, y)` as `[r, g, b]`, clamping coordinates into
    /// range so edge sampling never goes out of bounds.
    fn pixel_clamped(&self, x: usize, y: usize) -> [u8; 3] {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let i = (y * self.width + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

/// How an image is prepared for a specific vision tower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessConfig {
    /// Side length of the square canvas the tower expects, in pixels.
    pub image_size: usize,
    /// Side length of one square patch. Must divide [`Self::image_size`] — a
    /// tower's position-embedding table has exactly `(image_size / patch_size)²`
    /// entries, so a non-dividing pair describes a model that cannot exist.
    pub patch_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for PreprocessConfig {
    /// SigLIP-style 512px canvas with 16px patches — SmolVLM's shape.
    ///
    /// A default is offered only so tests and callers have a sane starting
    /// point; a real run must take these from the loaded model's metadata,
    /// because guessing them wrong produces plausible-looking garbage rather
    /// than an error.
    fn default() -> Self {
        Self { image_size: 512, patch_size: 16, mean: SIGLIP_MEAN, std: SIGLIP_STD }
    }
}

impl PreprocessConfig {
    /// Patches per side of the canvas.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total patches — the sequence length the vision tower sees.
    pub fn n_patches(&self) -> usize {
        self.grid() * self.grid()
    }

    /// Flattened length of one patch: `patch · patch · 3`.
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    /// Rejects a configuration that cannot describe a real tower.
    fn validate(&self) -> std::result::Result<(), String> {
        if self.image_size == 0 || self.patch_size == 0 {
            return Err("image_size and patch_size must both be non-zero".to_string());
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(format!(
                "patch_size {} must divide image_size {} exactly — the tower has one \
                 position embedding per patch slot, so a partial patch has no slot",
                self.patch_size, self.image_size
            ));
        }
        if self.std.contains(&0.0) {
            return Err("std must be non-zero in every channel (division by zero)".to_string());
        }
        Ok(())
    }
}

/// Resizes to a square `size × size` canvas by nearest-neighbour sampling.
///
/// # Why nearest-neighbour, and when that stops being good enough
///
/// Nearest-neighbour is exactly reproducible (integer arithmetic, no rounding
/// drift across platforms or float modes), which matters more than fidelity for
/// a first cut: the same page must yield the same tensor on Termux and on a
/// desktop, or nothing downstream is comparable.
///
/// It is genuinely worse than bilinear/Lanczos when **downsampling** a
/// high-DPI page render, because it point-samples and so aliases thin strokes —
/// exactly the strokes that carry text. A tower deciding *layout* (which is
/// what routing needs) tolerates that; a tower asked to *read* would not. Revisit
/// with an area-average resample before ever trusting this path for recognition.
pub fn resize_square(img: &Rgb8, size: usize) -> std::result::Result<Rgb8, String> {
    if size == 0 {
        return Err("target size must be non-zero".to_string());
    }
    if img.width == 0 || img.height == 0 {
        return Err("cannot resize an image with a zero dimension".to_string());
    }
    let mut out = vec![0u8; size * size * 3];
    for y in 0..size {
        // Sample at the CENTRE of each destination pixel (`+ 0.5`), not its
        // corner. Corner sampling biases the whole image up-left by half a
        // destination pixel — invisible on a photo, but a systematic shift on a
        // rendered page, and it compounds with the patch grid.
        let src_y = (((y as f64 + 0.5) * img.height as f64) / size as f64) as usize;
        for x in 0..size {
            let src_x = (((x as f64 + 0.5) * img.width as f64) / size as f64) as usize;
            let [r, g, b] = img.pixel_clamped(src_x, src_y);
            let i = (y * size + x) * 3;
            out[i] = r;
            out[i + 1] = g;
            out[i + 2] = b;
        }
    }
    Ok(Rgb8 { width: size, height: size, data: out })
}

/// Full preprocessing: resize, normalize, and patchify into the
/// `[n_patches, patch_dim]` matrix a patch-embedding matmul consumes.
///
/// Rows are patches in **row-major grid order** (left-to-right, then top-to-
/// bottom), which must match the order of the tower's position-embedding table;
/// column-major here would pair every patch with the wrong position and degrade
/// silently.
///
/// Within a row, values are ordered **channel-plane first** (all R of the patch,
/// then all G, then all B) — the CHW convention the weights are stored against.
pub fn preprocess(img: &Rgb8, cfg: &PreprocessConfig) -> std::result::Result<Tensor, String> {
    cfg.validate()?;
    let canvas = resize_square(img, cfg.image_size)?;

    let grid = cfg.grid();
    let patch = cfg.patch_size;
    let patch_dim = cfg.patch_dim();
    let mut rows = vec![0f32; cfg.n_patches() * patch_dim];

    for gy in 0..grid {
        for gx in 0..grid {
            let row_base = (gy * grid + gx) * patch_dim;
            for c in 0..3 {
                // Channel-plane offset within the row: CHW inside each patch.
                let plane = c * patch * patch;
                for py in 0..patch {
                    for px in 0..patch {
                        let sx = gx * patch + px;
                        let sy = gy * patch + py;
                        let s = (sy * cfg.image_size + sx) * 3 + c;
                        let v = f32::from(canvas.data[s]) / 255.0;
                        rows[row_base + plane + py * patch + px] =
                            (v - cfg.mean[c]) / cfg.std[c];
                    }
                }
            }
        }
    }

    Tensor::from_f32(rows, vec![cfg.n_patches(), patch_dim]).map_err(|e| e.to_string())
}

/// Convenience for the OCR/routing path: a grayscale page raster promoted to
/// RGB by replicating the single channel.
///
/// Vision towers are trained on three-channel input; handing them one channel
/// would mismatch the patch-embedding weight's shape. Replication is the
/// standard promotion and is what the towers saw for greyscale training data.
pub fn gray_to_rgb8(width: usize, height: usize, gray: &[u8]) -> std::result::Result<Rgb8, String> {
    if gray.len() != width * height {
        return Err(format!(
            "grayscale buffer is {} bytes, but {width}x{height} needs {}",
            gray.len(),
            width * height
        ));
    }
    let mut data = Vec::with_capacity(gray.len() * 3);
    for &g in gray {
        data.extend_from_slice(&[g, g, g]);
    }
    Rgb8::new(width, height, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w × h` image whose pixel `(x, y)` is a distinct, predictable colour,
    /// so a geometry mistake shows up as the wrong VALUE rather than needing a
    /// visual check.
    fn ramp(w: usize, h: usize) -> Rgb8 {
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[x as u8, y as u8, 7]);
            }
        }
        Rgb8::new(w, h, data).unwrap()
    }

    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Rgb8 {
        Rgb8::new(w, h, rgb.iter().cycle().take(w * h * 3).copied().collect()).unwrap()
    }

    #[test]
    fn rgb8_rejects_a_buffer_that_does_not_match_its_dimensions() {
        assert!(Rgb8::new(2, 2, vec![0; 11]).is_err());
        assert!(Rgb8::new(2, 2, vec![0; 12]).is_ok());
    }

    #[test]
    fn config_rejects_a_patch_size_that_does_not_divide_the_canvas() {
        // 512/17 leaves a partial patch, which has no position-embedding slot —
        // a model shaped like that cannot exist, so this must fail loudly rather
        // than silently truncate the last row and column.
        let bad = PreprocessConfig { patch_size: 17, ..Default::default() };
        assert!(bad.validate().is_err());
        assert!(PreprocessConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_zero_std_before_it_divides_by_it() {
        let bad = PreprocessConfig { std: [0.5, 0.0, 0.5], ..Default::default() };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn resize_is_identity_when_the_size_already_matches() {
        let img = ramp(8, 8);
        assert_eq!(resize_square(&img, 8).unwrap(), img);
    }

    #[test]
    fn resize_preserves_a_solid_colour_exactly() {
        // Any resampling scheme must leave a constant image constant; if this
        // fails the indexing is wrong, not the filter.
        let out = resize_square(&solid(3, 7, [9, 200, 30]), 16).unwrap();
        assert!(out.data.chunks(3).all(|p| p == [9, 200, 30]));
    }

    #[test]
    fn resize_samples_pixel_centres_rather_than_corners() {
        // Downscaling 4x4 -> 2x2 must take one pixel from each quadrant. Corner
        // sampling would take (0,0),(2,0),(0,2),(2,2) — biased up-left by half a
        // destination pixel. Centre sampling takes (1,1),(3,1),(1,3),(3,3).
        let out = resize_square(&ramp(4, 4), 2).unwrap();
        let px = |i: usize| [out.data[i * 3], out.data[i * 3 + 1]];
        assert_eq!(px(0), [1, 1], "top-left should sample source (1,1)");
        assert_eq!(px(1), [3, 1], "top-right should sample source (3,1)");
        assert_eq!(px(2), [1, 3], "bottom-left should sample source (1,3)");
        assert_eq!(px(3), [3, 3], "bottom-right should sample source (3,3)");
    }

    #[test]
    fn preprocess_produces_the_shape_the_patch_embedding_matmul_wants() {
        let cfg = PreprocessConfig { image_size: 32, patch_size: 16, ..Default::default() };
        let t = preprocess(&ramp(40, 20), &cfg).unwrap();
        // 32/16 = 2 -> 4 patches; each 16*16*3 = 768 long.
        assert_eq!(t.shape().dims(), &[4, 768]);
        assert_eq!(cfg.n_patches(), 4);
        assert_eq!(cfg.patch_dim(), 768);
    }

    #[test]
    fn normalization_maps_black_and_white_to_minus_one_and_one() {
        // With mean = std = 0.5 the 0..1 range maps to -1..1. A wrong constant
        // here never errors — it just shifts every activation — so pin it.
        let cfg = PreprocessConfig { image_size: 16, patch_size: 16, ..Default::default() };
        let black = preprocess(&solid(4, 4, [0, 0, 0]), &cfg).unwrap().to_vec_f32().unwrap();
        assert!(black.iter().all(|v| (*v + 1.0).abs() < 1e-6), "black must map to -1");
        let white = preprocess(&solid(4, 4, [255, 255, 255]), &cfg).unwrap().to_vec_f32().unwrap();
        assert!(white.iter().all(|v| (*v - 1.0).abs() < 1e-6), "white must map to +1");
    }

    #[test]
    fn each_patch_row_is_channel_planar_not_interleaved() {
        // CHW inside the row: all R, then all G, then all B. Interleaving here is
        // the classic silent bug — shapes still match, model sees colour noise.
        let cfg = PreprocessConfig {
            image_size: 2,
            patch_size: 2,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let img = solid(2, 2, [255, 0, 0]);
        let v = preprocess(&img, &cfg).unwrap().to_vec_f32().unwrap();
        assert_eq!(v.len(), 12);
        assert!(v[0..4].iter().all(|x| (*x - 1.0).abs() < 1e-6), "first plane is R (all 1.0)");
        assert!(v[4..12].iter().all(|x| x.abs() < 1e-6), "G and B planes are 0.0");
    }

    #[test]
    fn patch_rows_are_in_row_major_grid_order() {
        // Row order must match the tower's position-embedding table. Getting it
        // column-major pairs every patch with the wrong position — no error, just
        // quietly wrong. Two horizontally-split halves, distinct colours.
        let cfg = PreprocessConfig { image_size: 2, patch_size: 1, mean: [0.0; 3], std: [1.0; 3] };
        let img = Rgb8::new(2, 1, vec![255, 0, 0, /**/ 0, 0, 255]).unwrap();
        let v = preprocess(&img, &cfg).unwrap().to_vec_f32().unwrap();
        // 4 patches of dim 3. Row 0 = grid (0,0) = left = red.
        assert_eq!(&v[0..3], &[1.0, 0.0, 0.0], "patch 0 must be the LEFT half");
        assert_eq!(&v[3..6], &[0.0, 0.0, 1.0], "patch 1 must be the RIGHT half");
    }

    #[test]
    fn preprocessing_is_deterministic_for_the_same_input() {
        // The whole path must be reproducible: same page, same tensor, every run
        // and every platform. Integer-arithmetic resampling is chosen for this.
        let cfg = PreprocessConfig { image_size: 32, patch_size: 8, ..Default::default() };
        let img = ramp(37, 19);
        let a = preprocess(&img, &cfg).unwrap().to_vec_f32().unwrap();
        let b = preprocess(&img, &cfg).unwrap().to_vec_f32().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn gray_promotes_to_three_identical_channels() {
        let rgb = gray_to_rgb8(2, 1, &[10, 200]).unwrap();
        assert_eq!(rgb.data, vec![10, 10, 10, 200, 200, 200]);
        assert!(gray_to_rgb8(2, 1, &[10]).is_err(), "wrong-length buffer must be caught");
    }
}
