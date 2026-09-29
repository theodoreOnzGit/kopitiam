//! Ported from MuPDF's DCTDecode colour handling -- `source/fitz/filter-dct.c`
//! (`next_dctd`: the `/ColorTransform` vs Adobe-APP14 choice, lines 255-279),
//! `source/pdf/pdf-stream.c:160-165` (a PDF's DCT stream is never
//! CMYK-inverted), `source/fitz/image.c:726` + `source/fitz/draw-unpack.c:350`
//! (`fz_decode_tile`: `/Decode` on the decoded samples), and the libjpeg 9
//! bits MuPDF links (`thirdparty/libjpeg`: `jdapimin.c`
//! `default_decompress_parms` colour-space guess, `jdmarker.c`
//! `examine_app0`/`examine_app14`, `jdcolor.c` `build_ycc_rgb_table`,
//! `ycc_rgb_convert`, `ycck_cmyk_convert`) -- commit 19f1284, AGPL-3.0,
//! (c) Artifex Software, Inc.; libjpeg (c) Thomas G. Lane, Guido Vollbeding
//! (IJG licence). Translated to Rust for KOPITIAM (AGPL-3.0-only). See
//! docs/ACKNOWLEDGEMENTS.md and docs/ai-decisions/AID-0051 / AID-0052.
//!
//! # Wah, why this module exists (gh-116)
//!
//! The entropy decoding + IDCT is still `zune-jpeg` (AID-0052 substitution
//! for libjpeg). But the COLOUR part cannot be left to zune-jpeg, because its
//! defaults are not MuPDF's:
//!
//! * Left on its default output colour space (RGB), zune-jpeg converts a
//!   4-component JPEG to RGB itself as `r = c * k / 255` -- i.e. it *assumes*
//!   Photoshop-style inverted CMYK. Inside a PDF, MuPDF never inverts
//!   (`invert_cmyk = 0`, pdf-stream.c:164); the inversion, if any, is the
//!   image's `/Decode [1 0 1 0 1 0 1 0]`. So a plain-CMYK figure (Adobe
//!   transform 0, no `/Decode`) came out solid black in 0.4.2 lah.
//! * zune-jpeg ignores the PDF's `/DecodeParms /ColorTransform`, and the
//!   libjpeg component-id rules.
//!
//! So here we ask zune-jpeg for the RAW component samples (output colour
//! space = its input colour space: no conversion at all), decide the JPEG's
//! colour space exactly like libjpeg + filter-dct.c, convert YCbCr/YCCK with
//! libjpeg's own fixed-point tables, apply `/Decode` like `fz_decode_tile`,
//! and only then go CMYK -> RGB (MuPDF's no-ICC `fast_cmyk_to_rgb`).
//!
//! Known, recorded divergences (rare in PDFs): libjpeg 9's wide-gamut
//! `bg-sYCC` / `bg-RGB` component-id variants and its LSE colour transform
//! (`JCT_SUBTRACT_GREEN`) are treated as plain YCbCr / RGB.

use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use super::error::{Error, Result};

/// The colour space of the JPEG's *coded* components, as libjpeg's
/// `jpeg_color_space` ends up after filter-dct.c's override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JpegSpace {
    Gray,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
}

/// What the marker scan (up to SOS) found -- the bits of libjpeg's
/// `jpeg_read_header` state the colour-space choice depends on.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct JpegMarkers {
    /// `saw_JFIF_marker` (APP0 "JFIF\0", jdmarker.c `examine_app0`).
    pub jfif: bool,
    /// `Adobe_transform` when `saw_Adobe_marker` (APP14 "Adobe",
    /// jdmarker.c `examine_app14`: the 12th data byte).
    pub adobe_transform: Option<u8>,
    /// SOF component ids, in order (their count is `num_components`).
    pub component_ids: Vec<u8>,
}

/// Walk the JPEG markers from SOI to SOS, like `jpeg_read_header` does.
/// Stops at the first SOS (or EOI / truncated data); never panics.
pub(crate) fn scan_markers(bytes: &[u8]) -> JpegMarkers {
    let mut m = JpegMarkers::default();
    let mut i = 2usize; // past SOI
    while i + 1 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1; // garbage before a marker: libjpeg skips it too (next_marker)
            continue;
        }
        let code = bytes[i + 1];
        if code == 0xFF {
            i += 1; // fill byte
            continue;
        }
        i += 2;
        // Standalone markers carry no length.
        if code == 0x01 || (0xD0..=0xD8).contains(&code) {
            continue;
        }
        if code == 0xD9 || code == 0xDA || i + 1 >= bytes.len() {
            break; // EOI, SOS, or truncated
        }
        let len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
        let data = bytes.get(i + 2..(i + len).min(bytes.len())).unwrap_or(&[]);
        match code {
            // jdmarker.c:examine_app0 -- APP0_DATA_LEN = 14.
            0xE0 if data.len() >= 14 && data.starts_with(b"JFIF\0") => m.jfif = true,
            // jdmarker.c:examine_app14 -- APP14_DATA_LEN = 12.
            0xEE if data.len() >= 12 && data.starts_with(b"Adobe") => {
                m.adobe_transform = Some(data[11]);
            }
            // SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC).
            0xC0..=0xCF if !matches!(code, 0xC4 | 0xC8 | 0xCC) && data.len() >= 6 => {
                let n = data[5] as usize;
                m.component_ids = (0..n)
                    .filter_map(|c| data.get(6 + 3 * c).copied())
                    .collect();
            }
            _ => {}
        }
        i += len.max(2);
    }
    m
}

/// libjpeg's `default_decompress_parms` guess (jdapimin.c:115-199) followed by
/// filter-dct.c's `/ColorTransform` override (filter-dct.c:255-279).
/// `color_transform` is the PDF's `/DecodeParms /ColorTransform`, `-1` when
/// absent (pdf-stream.c:163).
pub(crate) fn jpeg_space(m: &JpegMarkers, color_transform: i64) -> Option<JpegSpace> {
    let ids = &m.component_ids;
    let n = ids.len();
    // jdapimin.c: the guess from component ids, then JFIF / Adobe markers.
    let guess = match n {
        1 => JpegSpace::Gray,
        3 => {
            if ids[..] == [0x01, 0x02, 0x03] || ids[..] == [0x01, 0x22, 0x23] {
                JpegSpace::YCbCr // (0x01 0x22 0x23 is bg-sYCC; see module docs)
            } else if ids[..] == [0x52, 0x47, 0x42] || ids[..] == [0x72, 0x67, 0x62] {
                JpegSpace::Rgb // 'R''G''B' (and bg-RGB 'r''g''b')
            } else if m.jfif {
                JpegSpace::YCbCr
            } else if let Some(t) = m.adobe_transform {
                if t == 0 {
                    JpegSpace::Rgb
                } else {
                    JpegSpace::YCbCr
                }
            } else {
                JpegSpace::YCbCr
            }
        }
        4 => {
            if ids[..] == [0x01, 0x02, 0x03, 0x04] {
                JpegSpace::Ycck
            } else if ids[..] == [0x43, 0x4D, 0x59, 0x4B] {
                JpegSpace::Cmyk // 'C''M''Y''K'
            } else if let Some(t) = m.adobe_transform {
                // transform 0 = CMYK; 2 = YCCK; anything else libjpeg warns
                // and assumes YCCK.
                if t == 0 {
                    JpegSpace::Cmyk
                } else {
                    JpegSpace::Ycck
                }
            } else {
                JpegSpace::Cmyk
            }
        }
        _ => return None,
    };
    // filter-dct.c:255-279: default 1 for 3 components, else 0; the Adobe
    // marker overrides the PDF; 0 switches libjpeg's transform off.
    let mut ct = color_transform;
    if ct < 0 {
        ct = if n == 3 { 1 } else { 0 };
    }
    if let Some(t) = m.adobe_transform {
        ct = t as i64;
    }
    Some(match (ct, n) {
        (0, 3) => JpegSpace::Rgb,
        (0, 4) => JpegSpace::Cmyk,
        _ => guess,
    })
}

// ---------------------------------------------------------------------------
// libjpeg jdcolor.c colour conversion (SCALEBITS = 16, exact integer tables)
// ---------------------------------------------------------------------------

const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

/// `FIX(x)`: `(INT32) ((x) * (1L<<SCALEBITS) + 0.5)`.
const fn fix(x: f64) -> i32 {
    (x * (1u32 << SCALEBITS) as f64 + 0.5) as i32
}

/// `build_ycc_rgb_table` (jdcolor.c:109): (Cr->R, Cb->B, Cr->G, Cb->G).
struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

fn ycc_tables() -> YccTables {
    let mut t = YccTables {
        cr_r: [0; 256],
        cb_b: [0; 256],
        cr_g: [0; 256],
        cb_g: [0; 256],
    };
    for i in 0..256 {
        let x = i as i32 - 128; // CENTERJSAMPLE
                                // DESCALE(v, SCALEBITS) = (v + ONE_HALF) >> SCALEBITS (arithmetic).
        t.cr_r[i] = (fix(1.402) * x + ONE_HALF) >> SCALEBITS;
        t.cb_b[i] = (fix(1.772) * x + ONE_HALF) >> SCALEBITS;
        t.cr_g[i] = -fix(0.714136286) * x;
        t.cb_g[i] = -fix(0.344136286) * x + ONE_HALF;
    }
    t
}

/// `range_limit[]`: clamp to a sample.
fn limit(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// `ycc_rgb_convert` (jdcolor.c:193), in place on 3-sample pixels.
fn ycc_to_rgb(px: &mut [u8]) {
    let t = ycc_tables();
    for p in px.chunks_exact_mut(3) {
        let (y, cb, cr) = (p[0] as i32, p[1] as usize, p[2] as usize);
        p[0] = limit(y + t.cr_r[cr]);
        p[1] = limit(y + ((t.cb_g[cb] + t.cr_g[cr]) >> SCALEBITS));
        p[2] = limit(y + t.cb_b[cb]);
    }
}

/// `ycck_cmyk_convert` (jdcolor.c:504), in place on 4-sample pixels: CMY are
/// `255 - RGB` of the YCC triple, K passes through unchanged.
fn ycck_to_cmyk(px: &mut [u8]) {
    let t = ycc_tables();
    for p in px.chunks_exact_mut(4) {
        let (y, cb, cr) = (p[0] as i32, p[1] as usize, p[2] as usize);
        p[0] = limit(255 - (y + t.cr_r[cr]));
        p[1] = limit(255 - (y + ((t.cb_g[cb] + t.cr_g[cr]) >> SCALEBITS)));
        p[2] = limit(255 - (y + t.cb_b[cb]));
    }
}

/// `fz_mul255` (geometry.h:38): Blinn's `a * b / 255`, rounded.
fn mul255(a: i32, b: i32) -> i32 {
    let mut x = a * b + 128;
    x += x >> 8;
    x >> 8
}

/// `fz_decode_tile` (draw-unpack.c:350) on `n`-sample pixels. `decode` is the
/// image's `/Decode`, padded like pdf-image.c:124-128 (a missing entry reads
/// as 0). Skipped when it is the identity (`use_decode`, image.c:1328-1333).
pub(crate) fn decode_tile(px: &mut [u8], n: usize, decode: Option<&[f32]>) {
    let Some(d) = decode else { return };
    let get = |i: usize| d.get(i).copied().unwrap_or(0.0);
    if (0..n).all(|k| get(2 * k) == 0.0 && get(2 * k + 1) == 1.0) {
        return;
    }
    let mut add = [0i32; 4];
    let mut mul = [0i32; 4];
    for k in 0..n.min(4) {
        // C float -> int conversion truncates toward zero.
        let min = (get(2 * k) * 255.0) as i32;
        let max = (get(2 * k + 1) * 255.0) as i32;
        add[k] = min;
        mul[k] = max - min;
    }
    for p in px.chunks_exact_mut(n) {
        for k in 0..n.min(4) {
            p[k] = (add[k] + mul255(p[k] as i32, mul[k])).clamp(0, 255) as u8;
        }
    }
}

/// Decoded, colour-resolved JPEG samples: `n` = 1 (gray), 3 (RGB) or 4
/// (CMYK, after `/Decode`, before any conversion to RGB).
pub(crate) struct DctSamples {
    pub width: usize,
    pub height: usize,
    pub n: usize,
    pub pixels: Vec<u8>,
}

// MuPDF: fz_open_dctd / next_dctd (filter-dct.c) + fz_decode_tile, with
// libjpeg's entropy decode + IDCT substituted by zune-jpeg (AID-0052).
/// Decode a DCTDecode stream to MuPDF's samples: gray, RGB or CMYK exactly as
/// libjpeg would hand them to MuPDF (no CMYK inversion), then `/Decode`.
/// `pdf_w`/`pdf_h` are the dict's size, used only if the codec reports none.
pub(crate) fn decode_dct(
    bytes: &[u8],
    pdf_w: usize,
    pdf_h: usize,
    color_transform: i64,
    decode: Option<&[f32]>,
) -> Result<DctSamples> {
    let space = jpeg_space(&scan_markers(bytes), color_transform);

    let mut probe = JpegDecoder::new(ZCursor::new(bytes));
    probe
        .decode_headers()
        .map_err(|e| Error::library(format!("JPEG decode failed: {e:?}")))?;
    let input = probe.input_colorspace().unwrap_or(ColorSpace::Unknown);
    // Raw samples: ask for the input colour space back, which zune-jpeg
    // copies through without converting (worker.rs color_convert).
    let raw_space = match input {
        ColorSpace::Luma
        | ColorSpace::YCbCr
        | ColorSpace::RGB
        | ColorSpace::CMYK
        | ColorSpace::YCCK => input,
        other => {
            return Err(Error::unsupported(format!(
                "unsupported JPEG colorspace: {other:?}"
            )))
        }
    };
    let mut decoder = JpegDecoder::new_with_options(
        ZCursor::new(bytes),
        DecoderOptions::default().jpeg_set_out_colorspace(raw_space),
    );
    let mut pixels = decoder
        .decode()
        .map_err(|e| Error::library(format!("JPEG decode failed: {e:?}")))?;
    let (w, h) = decoder
        .info()
        .map(|i| (i.width as usize, i.height as usize))
        .unwrap_or((pdf_w, pdf_h));
    let raw_n = raw_space.num_components();

    // If our marker scan could not decide (odd component count), fall back to
    // what zune-jpeg saw.
    let space = space.unwrap_or(match raw_space {
        ColorSpace::Luma => JpegSpace::Gray,
        ColorSpace::RGB => JpegSpace::Rgb,
        ColorSpace::CMYK => JpegSpace::Cmyk,
        ColorSpace::YCCK => JpegSpace::Ycck,
        _ => JpegSpace::YCbCr,
    });
    let n = match space {
        JpegSpace::Gray => 1,
        JpegSpace::YCbCr | JpegSpace::Rgb => 3,
        JpegSpace::Cmyk | JpegSpace::Ycck => 4,
    };
    if n != raw_n || pixels.len() < w * h * n {
        return Err(Error::library(format!(
            "JPEG: {raw_n} decoded components for a {space:?} image"
        )));
    }
    pixels.truncate(w * h * n);
    match space {
        JpegSpace::YCbCr => ycc_to_rgb(&mut pixels),
        JpegSpace::Ycck => ycck_to_cmyk(&mut pixels),
        JpegSpace::Gray | JpegSpace::Rgb | JpegSpace::Cmyk => {}
    }
    decode_tile(&mut pixels, n, decode);
    Ok(DctSamples {
        width: w,
        height: h,
        n,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn markers(jfif: bool, adobe: Option<u8>, ids: &[u8]) -> JpegMarkers {
        JpegMarkers {
            jfif,
            adobe_transform: adobe,
            component_ids: ids.to_vec(),
        }
    }

    #[test]
    fn adobe_transform_overrides_the_pdf_colortransform() {
        // The gh-116 shape: 'CMYK' ids + Adobe 0 -> plain CMYK.
        let m = markers(false, Some(0), b"CMYK");
        assert_eq!(jpeg_space(&m, -1), Some(JpegSpace::Cmyk));
        assert_eq!(jpeg_space(&m, 1), Some(JpegSpace::Cmyk));
        // Adobe 2 with 1..4 ids is YCCK, whatever the PDF says.
        let m = markers(false, Some(2), &[1, 2, 3, 4]);
        assert_eq!(jpeg_space(&m, 0), Some(JpegSpace::Ycck));
        // No Adobe marker: 4 components default to ColorTransform 0 -> CMYK,
        // even with YCCK-looking ids; /ColorTransform 1 lets libjpeg's id
        // guess stand.
        let m = markers(false, None, &[1, 2, 3, 4]);
        assert_eq!(jpeg_space(&m, -1), Some(JpegSpace::Cmyk));
        assert_eq!(jpeg_space(&m, 1), Some(JpegSpace::Ycck));
    }

    #[test]
    fn three_components_follow_colortransform() {
        let m = markers(true, None, &[1, 2, 3]);
        assert_eq!(jpeg_space(&m, -1), Some(JpegSpace::YCbCr));
        assert_eq!(jpeg_space(&m, 0), Some(JpegSpace::Rgb));
        assert_eq!(
            jpeg_space(&markers(false, Some(0), &[1, 2, 3]), 1),
            Some(JpegSpace::Rgb)
        );
        assert_eq!(
            jpeg_space(&markers(false, None, b"RGB"), -1),
            Some(JpegSpace::Rgb)
        );
    }

    #[test]
    fn decode_tile_matches_fz_decode_tile() {
        // [1 0] inverts; [0 0.5] halves (Blinn-rounded); identity is a no-op.
        let mut px = vec![0u8, 255, 128, 128];
        decode_tile(&mut px, 4, Some(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5]));
        assert_eq!(px, vec![255, 255, 128, 64]);
        let mut px = vec![7u8];
        decode_tile(&mut px, 1, Some(&[0.0, 1.0]));
        assert_eq!(px, vec![7]);
    }

    #[test]
    fn ycc_tables_match_libjpeg_neutral_and_extremes() {
        // Neutral chroma (128) is exact; libjpeg's Cr->R at Cr=255 is
        // DESCALE(FIX(1.402) * 127) = 178.
        let mut px = vec![100u8, 128, 128];
        ycc_to_rgb(&mut px);
        assert_eq!(px, vec![100, 100, 100]);
        assert_eq!(ycc_tables().cr_r[255], 178);
        // YCCK white ink: Y=255 neutral -> CMY 0, K kept.
        let mut px = vec![255u8, 128, 128, 42];
        ycck_to_cmyk(&mut px);
        assert_eq!(px, vec![0, 0, 0, 42]);
    }
}
