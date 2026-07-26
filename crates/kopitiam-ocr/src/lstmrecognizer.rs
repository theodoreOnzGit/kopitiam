//! Ported from Tesseract `src/lstm/lstmrecognizer.cpp` +
//! `src/lstm/lstmrecognizer.h` and the input-normalization from
//! `src/lstm/input.cpp` (+ the pixel packing in `src/lstm/networkio.cpp`
//! `FromPixes`/`ComputeBlackWhite`/`Copy1DGreyImage`/`SetPixel`, and the
//! percentile in `src/ccstruct/statistc.cpp` `STATS::ile`) at commit db0ec62,
//! Apache-2.0, © 2013/2014 Google Inc., Author: Ray Smith; part of the Tesseract
//! project. Translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation:
//! the `LSTMRecognizer::Load`/`DeSerialize`/`LoadCharsets` component wiring, the
//! `num_outputs == code_range + 1` CTC-null invariant, and the exact
//! image→`NetworkIO` normalization (adaptive black/white contrast stretch, not a
//! plain pixel/255) follow Tesseract; re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md (`docs/ai-decisions/AID-0051`).
//!
//! # What this is — the LSTM line driver (Phase 6)
//!
//! This ties phases 1–5 together into a single object that loads an LSTM
//! recognizer from a `.traineddata` and recognizes one normalized grayscale line
//! image into text:
//!
//! * [`LstmRecognizer::load`] = Tesseract's `LSTMRecognizer::Load` +
//!   `DeSerialize` + `LoadCharsets`: it reads the `LstmUnicharset`
//!   ([`Unicharset`]), the `LstmRecoder` ([`UnicharCompress`]), and the `Lstm`
//!   component (the [`Network`] tree plus the trailing scalar params, of which
//!   only `null_char_` matters at inference), and verifies the
//!   `num_outputs == code_range + 1` invariant the beam decoder relies on.
//! * [`LstmRecognizer::recognize_line`] = the recognition step of
//!   `LSTMRecognizer::RecognizeLine`: normalize the line image into a
//!   [`NetworkIO`], run [`Network::forward`], and decode with the raw
//!   (non-dictionary) [`RecodeBeamSearch`](crate::recodebeam::RecodeBeamSearch)
//!   from phase 5.
//!
//! # Input normalization (reproduced from input.cpp + networkio.cpp)
//!
//! Given a raw 8-bit grayscale [`GrayLine`], the pipeline is:
//! `PrepareLSTMInputs` scales the image to the network's input height
//! (`network.num_inputs()`, typically 36) preserving aspect ratio, then
//! `PreparePixInput`/`FromPixes` pack it. For the 1-D text-line path the image
//! becomes a `[width timesteps × height features]` [`NetworkIO`]
//! (`Copy1DGreyImage`), each feature the normalized value of one vertical pixel.
//! The per-pixel normalization is **not** `pixel/255`: `ComputeBlackWhite`
//! samples the local minima/maxima of the middle scanline to find an adaptive
//! `black`/`white` point, and `SetPixel` maps `[black, black + 2*contrast]` to
//! `[-1, 1]` via `float_pixel = (pixel - black) / contrast - 1`, where
//! `contrast = (white - black) / 2` (clamped to `1` if non-positive). The black
//! point is the 25th percentile of local minima; the white point the 75th of
//! local maxima (`STATS::ile`).
//!
//! # Deferred
//!
//! The image side that *produces* the [`GrayLine`] — the page rasterizer,
//! Leptonica preprocessing, and line-finding — are later phases; this driver
//! consumes an already-cropped, deskewed grayscale line. Consequently the exact
//! Leptonica `pixScale` resampling is approximated here with bilinear
//! interpolation (deterministic and aspect-preserving; it will not be
//! bit-identical to Leptonica on a real model — see the report). The two-stage
//! `PreScale`→`pixScale` (with the `kMaxInputHeight = 48` cap) collapses to one
//! scale-to-height since a line recognizer's input height is well under the cap.
//! Auto-inversion (`invert_threshold`/`OutputStats`), the dictionary beam,
//! char-box/`WERD_RES` extraction, the `LT_SOFTMAX` "simple text" output path,
//! and the `int_mode` fast path are deferred with their respective phases; the
//! raw CTC decode is driven directly.

use crate::error::{Error, Result};
use crate::network::Network;
use crate::networkio::NetworkIO;
use crate::recodebeam::decode_to_string;
use crate::tessdata::{TessdataManager, TessdataType};
use crate::unicharcompress::UnicharCompress;
use crate::unicharset::{UNICHAR_BROKEN, Unicharset};

/// `TF_COMPRESS_UNICHARSET`: the training flag that marks a model whose
/// `recoder_` re-encodes text to a smaller alphabet (all real LSTM models).
/// When set, the recoder is deserialized from the `LstmRecoder` component;
/// otherwise a pass-through recoder is built from the unicharset.
///
/// Tesseract: `TF_COMPRESS_UNICHARSET` (lstmrecognizer.h:46).
const TF_COMPRESS_UNICHARSET: i32 = 64;

/// A raw 8-bit grayscale text-line image: `height` rows of `width` pixels, in
/// row-major order (`pixels[y * width + x]`, `0` = black … `255` = white).
///
/// This is the input to [`LstmRecognizer::recognize_line`] — the product of the
/// (later-phase) page rasterizer + line-finding, kept deliberately minimal so
/// this crate takes no `image`-crate dependency. Tesseract's equivalent is the
/// single-line `Pix` handed to `Input::PreparePixInput`.
#[derive(Clone, Debug)]
pub struct GrayLine {
    /// Image width in pixels (number of columns).
    pub width: usize,
    /// Image height in pixels (number of rows).
    pub height: usize,
    /// Row-major 8-bit gray pixels, length `width * height`.
    pub pixels: Vec<u8>,
}

impl GrayLine {
    /// A grayscale line from its dimensions and row-major pixel buffer.
    ///
    /// # Panics
    /// If `pixels.len() != width * height`.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            width * height,
            "GrayLine pixel buffer length must be width * height"
        );
        GrayLine {
            width,
            height,
            pixels,
        }
    }
}

/// The loaded LSTM line recognizer: a [`Network`], its [`Unicharset`], the
/// [`UnicharCompress`] recoder, and the CTC null class index.
///
/// Tesseract: the inference-relevant members of `class LSTMRecognizer`
/// (lstmrecognizer.h:51). The training-only members (`randomizer_`,
/// `scratch_space_`, iteration counters, learning-rate/momentum/adam) are not
/// retained; `search_` is created per call rather than cached.
pub struct LstmRecognizer {
    /// The network hierarchy. Tesseract: `network_`.
    network: Network,
    /// The unicharset (id ↔ UTF-8). Tesseract: `ccutil_.unicharset`.
    unicharset: Unicharset,
    /// The recoder (unichar id ↔ code sequence). Tesseract: `recoder_`.
    recoder: UnicharCompress,
    /// The softmax class index of the CTC null/blank. Tesseract: `null_char_`.
    null_char: i32,
    /// Whether the model re-encodes (`TF_COMPRESS_UNICHARSET`). Retained for
    /// provenance; every real model is recoding.
    is_recoding: bool,
}

impl std::fmt::Debug for LstmRecognizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LstmRecognizer")
            .field("network_type", &self.network.network_type())
            .field("num_inputs", &self.network.num_inputs())
            .field("num_outputs", &self.network.num_outputs())
            .field("unicharset_size", &self.unicharset.size())
            .field("recoder_code_range", &self.recoder.code_range())
            .field("null_char", &self.null_char)
            .field("is_recoding", &self.is_recoding)
            .finish()
    }
}

impl LstmRecognizer {
    /// Loads a recognizer from a parsed `.traineddata` container.
    ///
    /// Tesseract: `LSTMRecognizer::Load` (lstmrecognizer.cpp:75) →
    /// `DeSerialize` (lstmrecognizer.cpp:133) → `LoadCharsets`
    /// (lstmrecognizer.cpp:180), restricted to the `mgr`-has-charsets path
    /// (`include_charsets == false`) that every real `.traineddata` takes: the
    /// unicharset and recoder are separate components, and the `Lstm` component
    /// holds the network followed by the scalar params. The dictionary load is
    /// omitted (the raw beam runs without a `Dict`).
    pub fn load(mgr: &TessdataManager) -> Result<Self> {
        // --- The Lstm component: network tree, then the scalar params. ---
        // Tesseract: DeSerialize -> Network::CreateFromFile, then the
        // network_str_/training_flags_/.../null_char_ block (lstmrecognizer.cpp:135).
        let mut lstm_fp = mgr
            .component_reader(TessdataType::Lstm)
            .ok_or_else(|| Error::format("traineddata has no LSTM component"))?;
        let network = Network::deserialize(&mut lstm_fp)?;

        // Read the trailing scalar params if serialized. For a real model these
        // always follow the network; a bare synthetic network may omit them, in
        // which case we default exactly as the `LSTMRecognizer()` constructor
        // does (`training_flags_ = 0`, `null_char_ = UNICHAR_BROKEN`,
        // lstmrecognizer.cpp:55). Only `training_flags_` (to pick the recoder
        // path) and `null_char_` matter at inference; adam/lr/momentum are
        // ignored and left unread.
        let (training_flags, null_char) = if lstm_fp.remaining_bytes() > 0 {
            let _network_str = lstm_fp.de_serialize_string()?;
            let training_flags = lstm_fp.read_i32()?;
            let _training_iteration = lstm_fp.read_i32()?;
            let _sample_iteration = lstm_fp.read_i32()?;
            let null_char = lstm_fp.read_i32()?;
            (training_flags, null_char)
        } else {
            (0, UNICHAR_BROKEN)
        };
        let is_recoding = (training_flags & TF_COMPRESS_UNICHARSET) != 0;

        // --- LoadCharsets: the unicharset, then the recoder. ---
        // Tesseract: LoadCharsets (lstmrecognizer.cpp:180).
        let mut uni_fp = mgr
            .component_reader(TessdataType::LstmUnicharset)
            .ok_or_else(|| Error::format("traineddata has no LSTM unicharset component"))?;
        let unicharset = Unicharset::load(&mut uni_fp, false)?;

        // Tesseract: LoadRecoder (lstmrecognizer.cpp:198): deserialize when
        // recoding, else build a pass-through from the unicharset.
        let mut recoder = UnicharCompress::new();
        if is_recoding {
            let mut rec_fp = mgr
                .component_reader(TessdataType::LstmRecoder)
                .ok_or_else(|| Error::format("recoding model has no LSTM recoder component"))?;
            recoder.deserialize(&mut rec_fp)?;
        } else {
            recoder.setup_pass_through(&unicharset);
        }

        // The softmax must have exactly one class per recoder code plus the CTC
        // null (the `+1` asserted by the phase-4 integration test). A mismatched
        // network/recoder pair cannot be decoded.
        let expected = recoder.code_range() + 1;
        if network.num_outputs() != expected {
            return Err(Error::format(format!(
                "network outputs {} != recoder code_range + 1 = {expected} (CTC-null invariant)",
                network.num_outputs()
            )));
        }

        Ok(LstmRecognizer {
            network,
            unicharset,
            recoder,
            null_char,
            is_recoding,
        })
    }

    /// The network's input feature dimension (the image height it expects).
    /// Tesseract: `LSTMRecognizer::NumInputs` (lstmrecognizer.h:213).
    pub fn num_inputs(&self) -> i32 {
        self.network.num_inputs()
    }

    /// The network's output feature dimension (softmax code range + CTC null).
    /// Tesseract: `LSTMRecognizer::NumOutputs` (lstmrecognizer.h:57).
    pub fn num_outputs(&self) -> i32 {
        self.network.num_outputs()
    }

    /// The CTC null/blank class index. Tesseract: `LSTMRecognizer::null_char`.
    pub fn null_char(&self) -> i32 {
        self.null_char
    }

    /// Whether the model re-encodes text through the recoder.
    pub fn is_recoding(&self) -> bool {
        self.is_recoding
    }

    /// The recognizer's unicharset.
    pub fn unicharset(&self) -> &Unicharset {
        &self.unicharset
    }

    /// Recognizes a single normalized grayscale line image into text.
    ///
    /// Tesseract: the recognition core of `LSTMRecognizer::RecognizeLine`
    /// (lstmrecognizer.cpp:320) + `LabelsViaReEncode` (lstmrecognizer.cpp:529):
    /// normalize the image into a [`NetworkIO`], run the forward pass, and decode
    /// with the raw (non-dictionary) beam
    /// (`Decode(output, 1.0, 0.0, kMinCertainty, nullptr)`), then reassemble the
    /// best path into a string. The `LT_SOFTMAX` "simple text" branch is not
    /// taken (recoding models use the recode/CTC path), so `simple_text = false`.
    pub fn recognize_line(&self, line: &GrayLine) -> Result<String> {
        let input = prepare_input(self.network.num_inputs().max(0) as usize, line);
        let output = self.network.forward(&input)?;
        Ok(decode_to_string(
            &self.recoder,
            self.null_char,
            false,
            &output,
            &self.unicharset,
        ))
    }
}

// ---------------------------------------------------------------------------
// Input normalization (input.cpp + networkio.cpp)
// ---------------------------------------------------------------------------

/// Normalizes a [`GrayLine`] into the network's input [`NetworkIO`]: scale to
/// `target_height` preserving aspect ratio, then pack as a
/// `[width × target_height]` buffer with the adaptive black/white contrast
/// stretch. Returns an empty buffer for a degenerate (zero-sized) input.
///
/// Tesseract: `Input::PrepareLSTMInputs` (input.cpp:82) +
/// `Input::PreparePixInput` (input.cpp:108) + `NetworkIO::FromPixes`
/// (networkio.cpp:171) + `Copy1DGreyImage` (networkio.cpp:257).
fn prepare_input(target_height: usize, line: &GrayLine) -> NetworkIO {
    if target_height == 0 || line.width == 0 || line.height == 0 {
        return NetworkIO::from_rows(0, target_height.max(1), Vec::new());
    }
    let (scaled_w, scaled_h, scaled) = scale_to_height(target_height, line);
    if scaled_w == 0 {
        return NetworkIO::from_rows(0, scaled_h.max(1), Vec::new());
    }

    // ComputeBlackWhite + the contrast clamp (networkio.cpp:122/197).
    let (black, white) = compute_black_white(&scaled, scaled_w, scaled_h);
    let mut contrast = (white - black) / 2.0;
    if contrast <= 0.0 {
        contrast = 1.0;
    }

    // Copy1DGreyImage: feature `y` of timestep `x` is the normalized pixel at
    // (row y, col x). The output depth equals the (scaled) image height.
    let num_features = scaled_h;
    let mut data = vec![0.0f32; scaled_w * num_features];
    for x in 0..scaled_w {
        for (y, slot) in data[x * num_features..(x + 1) * num_features]
            .iter_mut()
            .enumerate()
        {
            let pixel = scaled[y * scaled_w + x] as f32;
            // SetPixel (networkio.cpp:290): map [black, black+2*contrast] -> [-1,1].
            *slot = (pixel - black) / contrast - 1.0;
        }
    }
    NetworkIO::from_rows(scaled_w, num_features, data)
}

/// Bilinear resample of `line` to `target_height` rows, width scaled to preserve
/// aspect ratio. Returns `(width, height, row-major pixels)`.
///
/// Tesseract scales with Leptonica's `pixScale` (input.cpp:139); this bilinear
/// approximation is deterministic, aspect-preserving, and an identity when the
/// source is already at `target_height` — sufficient for the hermetic tests and
/// the (later-phase) real preprocessing that will supply properly scaled lines.
fn scale_to_height(target_height: usize, line: &GrayLine) -> (usize, usize, Vec<u8>) {
    let src_w = line.width;
    let src_h = line.height;
    let dst_h = target_height;
    // new_width = round(src_w * target_height / src_h), at least 1.
    let dst_w = (((src_w as f64) * (dst_h as f64) / (src_h as f64)).round() as usize).max(1);

    // Identity fast path (also what an already-normalized line hits).
    if dst_h == src_h && dst_w == src_w {
        return (src_w, src_h, line.pixels.clone());
    }

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let mut out = vec![0u8; dst_w * dst_h];
    let sample = |x: i32, y: i32| -> f32 {
        let xc = x.clamp(0, src_w as i32 - 1) as usize;
        let yc = y.clamp(0, src_h as i32 - 1) as usize;
        line.pixels[yc * src_w + xc] as f32
    };
    for oy in 0..dst_h {
        let fy = (oy as f32 + 0.5) * scale_y - 0.5;
        let y0 = fy.floor();
        let wy = fy - y0;
        let y0i = y0 as i32;
        for ox in 0..dst_w {
            let fx = (ox as f32 + 0.5) * scale_x - 0.5;
            let x0 = fx.floor();
            let wx = fx - x0;
            let x0i = x0 as i32;
            let p00 = sample(x0i, y0i);
            let p10 = sample(x0i + 1, y0i);
            let p01 = sample(x0i, y0i + 1);
            let p11 = sample(x0i + 1, y0i + 1);
            let top = p00 + (p10 - p00) * wx;
            let bot = p01 + (p11 - p01) * wx;
            let v = top + (bot - top) * wy;
            out[oy * dst_w + ox] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    (dst_w, dst_h, out)
}

/// Finds an adaptive black/white point from the middle scanline's local
/// minima/maxima, as the recognizer's contrast enhancer.
///
/// Tesseract: `ComputeBlackWhite` (networkio.cpp:127). `black` is the 25th
/// percentile of local minima; `white` the 75th of local maxima
/// ([`Stats::ile`]).
fn compute_black_white(img: &[u8], width: usize, height: usize) -> (f32, f32) {
    let mut mins = Stats::new();
    let mut maxes = Stats::new();
    if width >= 3 {
        let y = height / 2;
        let row = &img[y * width..y * width + width];
        let mut prev = row[0] as i32;
        let mut curr = row[1] as i32;
        // C: for (x = 1; x + 1 < width; ++x).
        for x in 1..width - 1 {
            let next = row[x + 1] as i32;
            if (curr < prev && curr <= next) || (curr <= prev && curr < next) {
                mins.add(curr);
            }
            if (curr > prev && curr >= next) || (curr >= prev && curr > next) {
                maxes.add(curr);
            }
            prev = curr;
            curr = next;
        }
    }
    if mins.total == 0 {
        mins.add(0);
    }
    if maxes.total == 0 {
        maxes.add(255);
    }
    (mins.ile(0.25) as f32, maxes.ile(0.75) as f32)
}

/// A minimal histogram over `[0, 255]` reproducing the subset of Tesseract's
/// `STATS` that `ComputeBlackWhite` uses (`add` + `ile`).
///
/// Tesseract: `class STATS` (statistc.h) with `rangemin_ = 0`, `rangemax_ = 255`.
struct Stats {
    buckets: [i64; 256],
    total: i64,
}

impl Stats {
    fn new() -> Self {
        Stats {
            buckets: [0; 256],
            total: 0,
        }
    }

    /// Tesseract: `STATS::add(value, 1)` (statistc.cpp:99) — clip to range, bump.
    fn add(&mut self, value: i32) {
        let v = value.clamp(0, 255) as usize;
        self.buckets[v] += 1;
        self.total += 1;
    }

    /// The interpolated fractile: the value below which `frac` of the samples
    /// lie. Tesseract: `STATS::ile` (statistc.cpp:172), `double` target branch.
    fn ile(&self, frac: f64) -> f64 {
        if self.total == 0 {
            return 0.0; // rangemin_
        }
        let mut target = frac * self.total as f64;
        target = target.clamp(1.0, self.total as f64);
        let mut sum: i64 = 0;
        let mut index: usize = 0;
        // C: for (index=0; index<=255 && sum<target; sum += buckets[index++]);
        while index <= 255 && (sum as f64) < target {
            sum += self.buckets[index];
            index += 1;
        }
        if index > 0 {
            let last = self.buckets[index - 1];
            if last == 0 {
                // The C asserts buckets_[index-1] > 0; guard against a divide by
                // zero (unreachable for a non-empty histogram + frac in (0,1)).
                return (index - 1) as f64;
            }
            // rangemin_ (0) + index - (sum - target) / buckets_[index-1].
            index as f64 - (sum as f64 - target) / last as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests;
