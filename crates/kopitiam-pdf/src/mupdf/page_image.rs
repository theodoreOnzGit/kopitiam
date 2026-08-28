//! Ported from MuPDF's image loaders -- `source/pdf/pdf-image.c`
//! (`pdf_load_image_imp`: the `/Width /Height /ColorSpace /BitsPerComponent
//! /Filter /Decode` reading and the sample layout), `source/fitz/image.c`
//! (`fz_decomp_image_from_stream` / `fz_unpack_stream`: unpacking 1/2/4/8/16-bpc
//! samples, the indexed-palette lookup, colour conversion), and
//! `source/fitz/load-jpeg.c` (`fz_load_jpeg`: the DCTDecode path) -- commit
//! 19f1284, AGPL-3.0, (c) Artifex Software, Inc., translated to Rust for KOPITIAM
//! (AGPL-3.0-only). Close adaptation: the field reading, sample unpacking and
//! colour handling follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references") and
//! docs/ai-decisions/AID-0051-mupdf-port-conventions.md.
//!
//! # Decoding a page's embedded images
//!
//! A PDF Image XObject is a stream whose dict carries the pixel geometry
//! (`/Width`, `/Height`, `/BitsPerComponent`), a `/ColorSpace`, an optional
//! `/Decode` remap, and a `/Filter` chain. This module walks a page's
//! `/Resources /XObject` sub-dictionary, keeps the `/Subtype /Image` entries, and
//! decodes each to a [`DecodedImage`] of 8-bit gray or RGB samples.
//!
//! It is the INPUT the OCR fallback consumes for scanned PDFs (a scanned page is
//! one big image; see [`page_full_image`]) and it lets a viewer paint image
//! pages.
//!
//! ## What is decoded here
//!
//! * **DCTDecode (JPEG)** -- decoded by the pure-Rust `zune-jpeg` crate
//!   (substituting MuPDF's `<jpeglib.h>`), which handles baseline and progressive
//!   JPEG. Grayscale -> gray, YCbCr -> RGB, CMYK/YCCK -> RGB.
//! * **Non-image filters** (Flate/LZW/ASCIIHex/ASCII85/RunLength, with
//!   predictors) -- decoded through the WAVE-2 filter layer
//!   ([`PdfDocument::open_stream`]) to raw samples, then interpreted per
//!   colorspace: 1/2/4/8/16-bpc unpacking, Indexed-palette lookup, CMYK->RGB,
//!   gray as-is, plus the `/Decode` remap.
//! * **Colorspaces**: DeviceGray/CalGray, DeviceRGB/CalRGB, DeviceCMYK,
//!   ICCBased (by its `/N`, falling back to `/Alternate`), and Indexed over any
//!   of those.
//!
//! ## What is deferred (a later codec wave)
//!
//! `JPXDecode` (JPEG 2000), `JBIG2Decode`, and `CCITTFaxDecode` return a clear
//! [`ErrorKind::Unsupported`](super::ErrorKind::Unsupported) error (never a
//! panic). Separation/DeviceN/Lab colorspaces are likewise unsupported. Soft
//! masks / `/SMask` / stencil-mask compositing are not applied -- each image is
//! returned as its own opaque sample buffer.

use super::error::{Error, Result};
use super::object::Object;
use super::xref::PdfDocument;

use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;

/// A decoded raster image: 8-bit samples, row-major, either 1 component
/// (grayscale) or 3 (RGB). This is the normalized form the OCR pipeline
/// (`kopitiam-ocr`: `to_gray` -> `binarize` -> `find_text_lines` -> `recognize`)
/// and the viewer's image mode consume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoftMask {
    /// Mask width in pixels — **independent of the base image's**. PDF
    /// 32000-1:2008 §11.6.5.3 lets an `/SMask` carry its own resolution and
    /// requires it to be scaled onto the image, so it is kept at its own size
    /// and sampled in normalized image space rather than resampled up front:
    /// no interpolation pass, and no precision lost when the mask is the
    /// coarser of the two.
    pub width: usize,
    pub height: usize,
    /// One alpha byte per mask pixel, row-major. 0 = fully transparent,
    /// 255 = fully opaque.
    pub alpha: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Samples per pixel: `1` = grayscale, `3` = RGB.
    pub components: u8,
    /// Row-major, 8-bit-per-component samples (`width * height * components`).
    pub pixels: Vec<u8>,
    /// The image's `/SMask` soft mask (§11.6.5.3), if it has one — per-pixel
    /// alpha to composite this image with.
    ///
    /// Ignoring this is not a cosmetic shortcut: a chart or logo exported by
    /// Word stores its background as *opaque black* in the base image and
    /// relies entirely on the mask to make it disappear. Drawn without the
    /// mask, such an image is a solid black rectangle over the page — which is
    /// exactly how the maintainer's workbook rendered before this existed.
    pub smask: Option<SoftMask>,
}

/// The interpreted `/ColorSpace` of an image, reduced to what the sample decoder
/// needs (`fz_colorspace` collapsed to its component count + indexed palette).
// MuPDF: fz_colorspace_type / fz_colorspace_n (image.c colour handling).
enum ColorKind {
    /// DeviceGray / CalGray / ICCBased N=1 -- one component.
    Gray,
    /// DeviceRGB / CalRGB / ICCBased N=3 -- three components.
    Rgb,
    /// DeviceCMYK / ICCBased N=4 -- four components, converted to RGB on output.
    Cmyk,
    /// `[/Indexed base hival lookup]` -- one index component into `palette`
    /// (row of `base` samples, 8-bit each), resolved to `base`'s colour.
    Indexed {
        base: Box<ColorKind>,
        palette: Vec<u8>,
    },
}

impl ColorKind {
    /// Components per pixel in the *sample data* (an Indexed image stores one
    /// index per pixel).
    fn source_components(&self) -> usize {
        match self {
            ColorKind::Gray => 1,
            ColorKind::Rgb => 3,
            ColorKind::Cmyk => 4,
            ColorKind::Indexed { .. } => 1,
        }
    }

    /// Components per pixel in the decoded [`DecodedImage`] (gray -> 1, everything
    /// else -> RGB 3).
    fn output_components(&self) -> u8 {
        match self {
            ColorKind::Gray => 1,
            ColorKind::Rgb | ColorKind::Cmyk => 3,
            ColorKind::Indexed { base, .. } => base.output_components(),
        }
    }
}

/// The image-only compression filters this module does not (yet) decode.
fn is_deferred_codec(name: &[u8]) -> bool {
    matches!(
        name,
        b"JPXDecode" | b"JBIG2Decode" | b"CCITTFaxDecode" | b"CCF"
    )
}

/// The JPEG filter names (`DCTDecode` / its inline abbreviation `DCT`).
fn is_dct(name: &[u8]) -> bool {
    matches!(name, b"DCTDecode" | b"DCT")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// MuPDF: pdf_load_image / the /XObject walk of pdf_run_xobject (image case).
/// Decode every `/Subtype /Image` XObject in page `page_index`'s resources, in
/// resource-dictionary order.
///
/// Returns an [`ErrorKind::Unsupported`](super::ErrorKind::Unsupported) error if
/// any image uses a deferred codec (JPX/JBIG2/CCITT) or colorspace; a page with
/// no images yields an empty vector.
pub fn page_images(doc: &PdfDocument, page_index: usize) -> Result<Vec<DecodedImage>> {
    let page = doc.page(page_index)?.clone();
    let images = image_xobjects(doc, &page)?;
    let mut out = Vec::with_capacity(images.len());
    for (dict, stream_ref) in &images {
        out.push(decode_image(doc, dict, stream_ref)?);
    }
    Ok(out)
}

/// The scanned-PDF helper: if page `page_index` is essentially a single image
/// covering the page, decode and return it; otherwise `None`.
///
/// ## Heuristic
///
/// A page is treated as one scanned image when its resources contain **exactly
/// one** Image XObject whose pixel aspect ratio matches the page's `/MediaBox`
/// aspect ratio within 15% (either orientation, to tolerate a `/Rotate`d page),
/// and whose smaller pixel dimension is at least 64 (so a lone small logo on an
/// otherwise-text page is not mistaken for a full-page scan). This is the common
/// scanned-document shape -- one full-bleed image per page -- and it deliberately
/// avoids interpreting the content stream to recover the image's exact placement
/// (`cm` matrix): the aspect match is a cheap, robust proxy for "covers the
/// page". A decode error (e.g. a deferred codec) propagates.
pub fn page_full_image(doc: &PdfDocument, page_index: usize) -> Result<Option<DecodedImage>> {
    let page = doc.page(page_index)?.clone();
    let images = image_xobjects(doc, &page)?;
    if images.len() != 1 {
        return Ok(None);
    }
    let (dict, stream_ref) = &images[0];

    let img_w = geta(doc, dict, "Width", "W").to_int();
    let img_h = geta(doc, dict, "Height", "H").to_int();
    if img_w < 64 || img_h < 64 {
        return Ok(None);
    }
    let img_aspect = img_w as f32 / img_h as f32;
    let page_aspect = page_media_aspect(doc, &page);

    let spans_page =
        aspect_close(img_aspect, page_aspect) || aspect_close(img_aspect, 1.0 / page_aspect);
    if !spans_page {
        return Ok(None);
    }
    Ok(Some(decode_image(doc, dict, stream_ref)?))
}

// ---------------------------------------------------------------------------
// Resource walking
// ---------------------------------------------------------------------------

/// The page's `/Resources /XObject` entries with `/Subtype /Image`, returned as
/// `(resolved dict, the entry's indirect reference)` pairs. The reference is kept
/// so the stream body can be opened.
fn image_xobjects(doc: &PdfDocument, page: &Object) -> Result<Vec<(Object, Object)>> {
    let resources = doc.resolve_get(page, "Resources")?;
    let xobjects = doc.resolve_get(&resources, "XObject")?;
    if !xobjects.is_dict() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for i in 0..xobjects.dict_len() {
        let Some(entry) = xobjects.dict_get_val(i) else {
            continue;
        };
        let dict = doc.resolve(entry)?;
        if !dict.is_dict() {
            continue;
        }
        if doc.resolve_get(&dict, "Subtype")?.to_name() == b"Image" {
            out.push((dict, entry.clone()));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Single-image decode
// ---------------------------------------------------------------------------

// MuPDF: pdf_load_image (pdf-image.c) as reached from `Do` in pdf_run_xobject.
/// Decode an Image XObject named by a content-stream `Do`, given its resolved
/// dict and the indirect reference to its stream. The crate-visible entry the
/// content interpreter ([`Processor::op_do`](super::interpret::Processor)) uses to
/// paint image XObjects; a deferred codec / colorspace returns an error the caller
/// treats as a safe skip.
pub(crate) fn decode_image_xobject(
    doc: &PdfDocument,
    dict: &Object,
    stream_ref: &Object,
) -> Result<DecodedImage> {
    decode_image(doc, dict, stream_ref)
}

// MuPDF: pdf_load_image_imp (pdf-image.c:33).
/// Decode one Image XObject to a [`DecodedImage`].
fn decode_image(doc: &PdfDocument, dict: &Object, stream_ref: &Object) -> Result<DecodedImage> {
    let mut img = decode_image_base(doc, dict, stream_ref)?;
    img.smask = decode_smask(doc, dict);
    Ok(img)
}

/// Decode an image's `/SMask` (§11.6.5.3) to per-pixel alpha.
///
/// The mask is itself an image XObject in `DeviceGray`, whose samples *are*
/// the alpha. Returns `None` when there is no mask, or when the mask cannot be
/// decoded — a failed mask must never fail the base image, since dropping the
/// picture entirely is worse than drawing it opaque.
///
/// `/Matte` (pre-multiplied alpha, §11.6.5.3) is not handled: it is rare, and
/// treating a matted mask as ordinary alpha merely mis-tints edge pixels
/// rather than producing the solid-black rectangle this function exists to
/// prevent.
fn decode_smask(doc: &PdfDocument, dict: &Object) -> Option<SoftMask> {
    let smask_ref = dict.dict_gets("SMask")?;
    let smask_dict = doc.resolve(smask_ref).ok()?;
    if !smask_dict.is_dict() {
        return None;
    }
    let decoded = decode_image_base(doc, &smask_dict, smask_ref).ok()?;
    if decoded.width == 0 || decoded.height == 0 {
        return None;
    }
    // The mask is DeviceGray by spec, but be defensive: if it decoded to RGB
    // (a malformed producer), take the first component rather than bailing.
    let n = decoded.components.max(1) as usize;
    let alpha: Vec<u8> = decoded.pixels.iter().step_by(n).copied().collect();
    if alpha.len() < decoded.width * decoded.height {
        return None;
    }
    Some(SoftMask {
        width: decoded.width,
        height: decoded.height,
        alpha,
    })
}

fn decode_image_base(
    doc: &PdfDocument,
    dict: &Object,
    stream_ref: &Object,
) -> Result<DecodedImage> {
    let width = geta(doc, dict, "Width", "W").to_int();
    let height = geta(doc, dict, "Height", "H").to_int();
    if width <= 0 || height <= 0 {
        return Err(Error::syntax("image has zero (or negative) dimensions"));
    }
    let (width, height) = (width as usize, height as usize);

    // Classify the filter chain: a deferred codec errors early; a DCT filter
    // routes to the JPEG decoder; anything else is the raw-sample path.
    let filters = filter_names(doc, dict);
    if let Some(name) = filters.iter().find(|n| is_deferred_codec(n)) {
        return Err(Error::unsupported(format!(
            "image codec not yet supported: {}",
            String::from_utf8_lossy(name)
        )));
    }

    if let Some(pos) = filters.iter().position(|n| is_dct(n)) {
        // DCTDecode: apply any *leading* non-image filters, then JPEG-decode.
        let (raw, filter, parms) = doc.stream_raw(stream_ref)?;
        let jpeg = apply_leading_filters(raw, &filter, &parms, pos)?;
        return decode_jpeg(&jpeg, width, height, read_decode(doc, dict).as_deref());
    }

    // Raw-sample path: BitsPerComponent, ColorSpace, Decode, then unpack.
    // MuPDF: bpc defaults to 8; an ImageMask forces bpc=1 (pdf-image.c:64,74).
    let image_mask = geta(doc, dict, "ImageMask", "IM").to_bool();
    let mut bpc = geta(doc, dict, "BitsPerComponent", "BPC").to_int();
    if bpc == 0 {
        bpc = 8;
    }
    if image_mask {
        bpc = 1;
    }
    if !matches!(bpc, 1 | 2 | 4 | 8 | 16) {
        return Err(Error::syntax(format!(
            "unsupported bits-per-component: {bpc}"
        )));
    }
    let bpc = bpc as u32;

    // An image mask is a 1-bit stencil with an implicit 1-component gray space.
    let cs = if image_mask {
        ColorKind::Gray
    } else {
        let cs_obj = geta(doc, dict, "ColorSpace", "CS");
        parse_colorspace(doc, &cs_obj)?
    };

    let samples = doc.open_stream(stream_ref)?;
    let decode = read_decode(doc, dict);
    Ok(decode_samples(
        width,
        height,
        bpc,
        &cs,
        decode.as_deref(),
        &samples,
    ))
}

/// Apply the filters *before* index `image_pos` in the chain (usually none), so
/// the bytes handed to the JPEG decoder are the codec's own input.
fn apply_leading_filters(
    raw: Vec<u8>,
    filter: &Object,
    parms: &Object,
    image_pos: usize,
) -> Result<Vec<u8>> {
    if image_pos == 0 {
        return Ok(raw);
    }
    // Rebuild a /Filter (+ /DecodeParms) covering just the leading entries.
    let mut lead_filter = Object::new_array();
    let mut lead_parms = Object::new_array();
    if let Object::Array(items) = filter {
        for f in items.iter().take(image_pos) {
            lead_filter.array_push(f.clone());
        }
    }
    if let Object::Array(items) = parms {
        for p in items.iter().take(image_pos) {
            lead_parms.array_push(p.clone());
        }
    }
    super::doc_stream::decode_stream(raw, &lead_filter, &lead_parms)
}

// ---------------------------------------------------------------------------
// JPEG (DCTDecode) path
// ---------------------------------------------------------------------------

// MuPDF: fz_load_jpeg (load-jpeg.c) -- here delegated to the pure-Rust zune-jpeg.
/// Decode baseline/progressive JPEG bytes to a [`DecodedImage`]. `pdf_w`/`pdf_h`
/// are the dict's declared dimensions, used only as a fallback if the codec does
/// not report them. `decode` is the image's `/Decode` array (used to invert
/// Adobe CMYK JPEGs when present).
fn decode_jpeg(
    bytes: &[u8],
    pdf_w: usize,
    pdf_h: usize,
    decode: Option<&[f32]>,
) -> Result<DecodedImage> {
    let mut decoder = JpegDecoder::new(ZCursor::new(bytes));
    let pixels = decoder
        .decode()
        .map_err(|e| Error::library(format!("JPEG decode failed: {e:?}")))?;
    let (w, h) = decoder
        .info()
        .map(|i| (i.width as usize, i.height as usize))
        .unwrap_or((pdf_w, pdf_h));
    let cs = decoder.output_colorspace().unwrap_or(ColorSpace::Unknown);

    match cs {
        ColorSpace::Luma => Ok(DecodedImage {
            smask: None,
            width: w,
            height: h,
            components: 1,
            pixels,
        }),
        ColorSpace::RGB => Ok(DecodedImage {
            smask: None,
            width: w,
            height: h,
            components: 3,
            pixels,
        }),
        ColorSpace::CMYK | ColorSpace::YCCK => {
            // Adobe CMYK JPEGs commonly store inverted samples, flagged by a
            // /Decode of [1 0 1 0 1 0 1 0]; honour it when present.
            let invert = decode
                .map(|d| d.first().copied().unwrap_or(0.0) > 0.5)
                .unwrap_or(false);
            let mut out = Vec::with_capacity(w * h * 3);
            for px in pixels.chunks_exact(4) {
                let f = |b: u8| {
                    if invert {
                        1.0 - b as f32 / 255.0
                    } else {
                        b as f32 / 255.0
                    }
                };
                let (r, g, b) = cmyk_to_rgb(f(px[0]), f(px[1]), f(px[2]), f(px[3]));
                out.extend_from_slice(&[r, g, b]);
            }
            Ok(DecodedImage {
                smask: None,
                width: w,
                height: h,
                components: 3,
                pixels: out,
            })
        }
        other => Err(Error::unsupported(format!(
            "unsupported JPEG output colorspace: {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Sample unpacking (raw / non-JPEG path)
// ---------------------------------------------------------------------------

// MuPDF: fz_unpack_stream + fz_decomp_image_from_stream (image.c) -- unpack
// bpc-bit samples row by row and resolve colour.
/// Interpret decoded stream `data` as `width * height` pixels of `cs`, unpacking
/// `bpc`-bit samples (rows are byte-aligned) and applying the optional `/Decode`
/// remap, producing an 8-bit gray/RGB [`DecodedImage`].
fn decode_samples(
    width: usize,
    height: usize,
    bpc: u32,
    cs: &ColorKind,
    decode: Option<&[f32]>,
    data: &[u8],
) -> DecodedImage {
    let src_n = cs.source_components();
    let out_n = cs.output_components() as usize;
    let maxval = ((1u32 << bpc) - 1) as f32;
    // Row stride in bytes (each row starts on a byte boundary, per PDF).
    let stride = (width * src_n * bpc as usize).div_ceil(8);

    let mut pixels = Vec::with_capacity(width * height * out_n);
    let mut comps = [0u32; 4];
    for y in 0..height {
        let row = data.get(y * stride..).unwrap_or(&[]);
        let mut bit = 0usize;
        for _x in 0..width {
            for c in comps.iter_mut().take(src_n) {
                *c = read_bits(row, &mut bit, bpc);
            }
            emit_pixel(cs, &comps[..src_n], maxval, decode, &mut pixels);
        }
    }
    DecodedImage {
        width,
        height,
        components: out_n as u8,
        pixels,
        smask: None,
    }
}

/// Read `bpc` bits (MSB-first) starting at `*bit` within `row`, advancing `*bit`.
/// Reads past the row's end yield 0 (a truncated final row is tolerated).
fn read_bits(row: &[u8], bit: &mut usize, bpc: u32) -> u32 {
    let mut v = 0u32;
    for _ in 0..bpc {
        let byte = row.get(*bit / 8).copied().unwrap_or(0);
        let b = (byte >> (7 - (*bit % 8))) & 1;
        v = (v << 1) | b as u32;
        *bit += 1;
    }
    v
}

/// Resolve one pixel's `src_n` raw samples through `cs` and push its output
/// bytes.
fn emit_pixel(
    cs: &ColorKind,
    comps: &[u32],
    maxval: f32,
    decode: Option<&[f32]>,
    out: &mut Vec<u8>,
) {
    match cs {
        ColorKind::Gray | ColorKind::Rgb | ColorKind::Cmyk => {
            let mut c01 = [0f32; 4];
            for (i, &raw) in comps.iter().enumerate() {
                c01[i] = component01(raw, maxval, decode, i);
            }
            push_base(cs, &c01[..comps.len()], out);
        }
        ColorKind::Indexed { base, palette } => {
            // The single sample is a palette index (optionally /Decode-remapped).
            let index = match decode {
                Some(d) if d.len() >= 2 => (d[0] + comps[0] as f32 * (d[1] - d[0]) / maxval)
                    .round()
                    .max(0.0) as usize,
                _ => comps[0] as usize,
            };
            let base_n = base.source_components();
            let start = index * base_n;
            let mut c01 = [0f32; 4];
            for (i, slot) in c01.iter_mut().take(base_n).enumerate() {
                *slot = palette.get(start + i).copied().unwrap_or(0) as f32 / 255.0;
            }
            push_base(base, &c01[..base_n], out);
        }
    }
}

/// Push a base-colorspace pixel (`comps` in `[0, 1]`) as 8-bit gray or RGB.
fn push_base(base: &ColorKind, comps: &[f32], out: &mut Vec<u8>) {
    match base {
        ColorKind::Gray => out.push(to8(comps[0])),
        ColorKind::Rgb => out.extend_from_slice(&[to8(comps[0]), to8(comps[1]), to8(comps[2])]),
        ColorKind::Cmyk => {
            let (r, g, b) = cmyk_to_rgb(comps[0], comps[1], comps[2], comps[3]);
            out.extend_from_slice(&[r, g, b]);
        }
        // An Indexed palette's base is never itself Indexed (PDF forbids it).
        ColorKind::Indexed { .. } => out.push(to8(comps[0])),
    }
}

/// Map raw sample `raw` (0..=maxval) for component `i` to `[0, 1]`, honouring the
/// `/Decode` array when present (a linear remap of the sample range).
fn component01(raw: u32, maxval: f32, decode: Option<&[f32]>, i: usize) -> f32 {
    match decode {
        Some(d) if d.len() >= 2 * (i + 1) => {
            let (dmin, dmax) = (d[2 * i], d[2 * i + 1]);
            (dmin + raw as f32 * (dmax - dmin) / maxval).clamp(0.0, 1.0)
        }
        _ => (raw as f32 / maxval).clamp(0.0, 1.0),
    }
}

/// Naive subtractive CMYK->RGB (each channel in `[0, 1]`), matching the common
/// `r = (1-c)(1-k)` device conversion.
fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (u8, u8, u8) {
    (
        to8((1.0 - c) * (1.0 - k)),
        to8((1.0 - m) * (1.0 - k)),
        to8((1.0 - y) * (1.0 - k)),
    )
}

/// Clamp `[0, 1]` and scale to an 8-bit sample (round to nearest).
fn to8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// ColorSpace parsing
// ---------------------------------------------------------------------------

// MuPDF: pdf_load_colorspace (pdf-colorspace.c) reduced to the component count
// (+ indexed palette) the sample decoder needs.
/// Interpret a `/ColorSpace` object (name or array) into a [`ColorKind`].
fn parse_colorspace(doc: &PdfDocument, obj: &Object) -> Result<ColorKind> {
    let obj = doc.resolve(obj)?;
    match &obj {
        Object::Name(name) => match name.as_slice() {
            b"DeviceGray" | b"G" | b"CalGray" => Ok(ColorKind::Gray),
            b"DeviceRGB" | b"RGB" | b"CalRGB" => Ok(ColorKind::Rgb),
            b"DeviceCMYK" | b"CMYK" => Ok(ColorKind::Cmyk),
            other => Err(Error::unsupported(format!(
                "unsupported colorspace: /{}",
                String::from_utf8_lossy(other)
            ))),
        },
        Object::Array(items) => {
            let head = items.first().map(|o| o.to_name()).unwrap_or(b"");
            match head {
                b"ICCBased" => {
                    let stream = items.get(1).cloned().unwrap_or(Object::Null);
                    let sdict = doc.resolve(&stream)?;
                    let n = doc.resolve_get(&sdict, "N")?.to_int();
                    match n {
                        1 => Ok(ColorKind::Gray),
                        3 => Ok(ColorKind::Rgb),
                        4 => Ok(ColorKind::Cmyk),
                        _ => match sdict.dict_gets("Alternate") {
                            Some(alt) => parse_colorspace(doc, &alt.clone()),
                            None => Err(Error::unsupported(format!(
                                "ICCBased colorspace with unsupported N={n}"
                            ))),
                        },
                    }
                }
                b"Indexed" | b"I" => {
                    let base_obj = items.get(1).cloned().unwrap_or(Object::Null);
                    let base = parse_colorspace(doc, &base_obj)?;
                    let lookup_obj = items.get(3).cloned().unwrap_or(Object::Null);
                    let resolved = doc.resolve(&lookup_obj)?;
                    let palette = match resolved {
                        Object::String(bytes) => bytes,
                        d if d.is_dict() => doc.open_stream(&lookup_obj)?,
                        _ => {
                            return Err(Error::syntax(
                                "Indexed colorspace lookup is not a string or stream",
                            ));
                        }
                    };
                    Ok(ColorKind::Indexed {
                        base: Box::new(base),
                        palette,
                    })
                }
                b"CalGray" => Ok(ColorKind::Gray),
                b"CalRGB" => Ok(ColorKind::Rgb),
                other => Err(Error::unsupported(format!(
                    "unsupported colorspace family: /{}",
                    String::from_utf8_lossy(other)
                ))),
            }
        }
        _ => Err(Error::unsupported("missing or malformed /ColorSpace")),
    }
}

// ---------------------------------------------------------------------------
// Small dict helpers
// ---------------------------------------------------------------------------

/// `pdf_dict_geta`: look up `primary`, else its inline `abbrev`, then resolve.
fn geta(doc: &PdfDocument, dict: &Object, primary: &str, abbrev: &str) -> Object {
    let entry = dict.dict_gets(primary).or_else(|| dict.dict_gets(abbrev));
    match entry {
        Some(o) => doc.resolve(o).unwrap_or(Object::Null),
        None => Object::Null,
    }
}

/// The image's `/Filter` (or `/F`) chain as a list of decoded filter names.
fn filter_names(doc: &PdfDocument, dict: &Object) -> Vec<Vec<u8>> {
    match geta(doc, dict, "Filter", "F") {
        Object::Name(n) => vec![n],
        Object::Array(items) => items
            .iter()
            .filter_map(|o| match doc.resolve(o).unwrap_or(Object::Null) {
                Object::Name(n) => Some(n),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The image's `/Decode` (or `/D`) array as floats, or `None` if absent/empty.
fn read_decode(doc: &PdfDocument, dict: &Object) -> Option<Vec<f32>> {
    let arr = geta(doc, dict, "Decode", "D");
    if arr.array_len() == 0 {
        return None;
    }
    Some(
        (0..arr.array_len())
            .map(|i| {
                arr.array_get(i)
                    .and_then(|o| doc.resolve(o).ok())
                    .map(|o| o.to_real() as f32)
                    .unwrap_or(0.0)
            })
            .collect(),
    )
}

/// The page's `/MediaBox` aspect ratio (width / height), defaulting to US Letter.
fn page_media_aspect(doc: &PdfDocument, page: &Object) -> f32 {
    let mb = doc.resolve_get(page, "MediaBox").unwrap_or(Object::Null);
    if mb.array_len() < 4 {
        return 612.0 / 792.0;
    }
    let v = |i: usize| -> f32 {
        mb.array_get(i)
            .and_then(|o| doc.resolve(o).ok())
            .map(|o| o.to_real() as f32)
            .unwrap_or(0.0)
    };
    let w = (v(2) - v(0)).abs();
    let h = (v(3) - v(1)).abs();
    if w < 1.0 || h < 1.0 {
        612.0 / 792.0
    } else {
        w / h
    }
}

/// Two aspect ratios are "close" when within 15% (relative to `b`).
fn aspect_close(a: f32, b: f32) -> bool {
    b > 0.0 && ((a - b).abs() / b) < 0.15
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_oxide::deflate::compress_to_vec_zlib;

    /// Build a PDF from raw object bodies (object `n` = `bodies[n-1]`), wrapping
    /// each in `N 0 obj ... endobj` and appending a classic xref + trailer. Bodies
    /// are byte vectors so stream objects can carry binary data.
    fn build_pdf(bodies: &[Vec<u8>]) -> Vec<u8> {
        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref_ofs = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }

    /// A stream object body: `<<dict /Length L>>\nstream\n<data>\nendstream`.
    /// `dict` must be the dict *contents* without the enclosing `<< >>` or Length.
    fn stream_body(dict: &str, data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(format!("<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes());
        b.extend_from_slice(data);
        b.extend_from_slice(b"\nendstream");
        b
    }

    fn plain(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    /// Standard catalog/pages/page objects (objects 1..=3). The page (object 3)
    /// takes a MediaBox string and a Resources string.
    fn scaffold(mediabox: &str, resources: &str) -> [Vec<u8>; 3] {
        [
            plain("<< /Type /Catalog /Pages 2 0 R >>"),
            plain("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
            plain(&format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox {mediabox} /Resources {resources} >>"
            )),
        ]
    }

    #[test]
    fn device_gray_8bpc_flate() {
        // A 2x2 gray grid with exact known values.
        let raw = [10u8, 20, 30, 40];
        let data = compress_to_vec_zlib(&raw, 6);
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode",
            &data,
        );
        let s = scaffold("[0 0 2 2]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let images = page_images(&doc, 0).unwrap();
        assert_eq!(images.len(), 1);
        let im = &images[0];
        assert_eq!((im.width, im.height, im.components), (2, 2, 1));
        assert_eq!(im.pixels, vec![10, 20, 30, 40]);
    }

    #[test]
    fn device_rgb_8bpc_flate() {
        // 2x1 RGB: red, then green.
        let raw = [255u8, 0, 0, 0, 255, 0];
        let data = compress_to_vec_zlib(&raw, 6);
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode",
            &data,
        );
        let s = scaffold("[0 0 2 1]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let images = page_images(&doc, 0).unwrap();
        let im = &images[0];
        assert_eq!((im.width, im.height, im.components), (2, 1, 3));
        assert_eq!(im.pixels, vec![255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn indexed_palette_lookup() {
        // Palette of 3 RGB colours: red, green, blue (hival 2). Indices pick
        // red, green, blue, red across a 2x2 grid.
        let indices = [0u8, 1, 2, 0];
        let data = compress_to_vec_zlib(&indices, 6);
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
             /ColorSpace [/Indexed /DeviceRGB 2 <FF000000FF000000FF>] \
             /BitsPerComponent 8 /Filter /FlateDecode",
            &data,
        );
        let s = scaffold("[0 0 2 2]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let im = &page_images(&doc, 0).unwrap()[0];
        assert_eq!((im.width, im.height, im.components), (2, 2, 3));
        assert_eq!(
            im.pixels,
            vec![
                255, 0, 0, /**/ 0, 255, 0, /**/ 0, 0, 255, /**/ 255, 0, 0
            ]
        );
    }

    #[test]
    fn one_bpc_unpacks_to_black_and_white() {
        // 8x1 DeviceGray, 1 bpc, no filter. 0b10101010 -> 255,0,255,0,...
        let data = vec![0b1010_1010u8];
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 8 /Height 1 \
             /ColorSpace /DeviceGray /BitsPerComponent 1",
            &data,
        );
        let s = scaffold("[0 0 8 1]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let im = &page_images(&doc, 0).unwrap()[0];
        assert_eq!((im.width, im.height, im.components), (8, 1, 1));
        assert_eq!(im.pixels, vec![255, 0, 255, 0, 255, 0, 255, 0]);
    }

    #[test]
    fn baseline_jpeg_dctdecode_decodes_dims() {
        // A tiny embedded baseline JPEG (JPEG_16X8): 16x8, 3-component (YCbCr),
        // a solid ~128 gray field. zune-jpeg converts YCbCr -> RGB.
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 16 /Height 8 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode",
            JPEG_16X8,
        );
        let s = scaffold("[0 0 16 8]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let im = &page_images(&doc, 0).unwrap()[0];
        // JPEG is lossy: assert exact dims + component count, pixels only sane.
        assert_eq!((im.width, im.height), (16, 8));
        assert_eq!(im.components, 3);
        assert_eq!(im.pixels.len(), 16 * 8 * 3);
        // The fixture is a solid mid-gray field; every sample is near 128.
        assert!(im.pixels.iter().all(|&p| (100..=160).contains(&p)));
    }

    #[test]
    fn page_full_image_some_for_single_full_page_scan() {
        // One image whose aspect (128x64 = 2.0) matches the page (400x200 = 2.0).
        let raw = vec![200u8; 128 * 64];
        let data = compress_to_vec_zlib(&raw, 6);
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 128 /Height 64 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode",
            &data,
        );
        let s = scaffold("[0 0 400 200]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let full = page_full_image(&doc, 0).unwrap();
        assert!(full.is_some());
        let im = full.unwrap();
        assert_eq!((im.width, im.height), (128, 64));
    }

    #[test]
    fn page_full_image_none_for_text_page() {
        // No XObjects at all -> not a scanned page.
        let s = scaffold("[0 0 400 200]", "<< /Font << >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone()];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        assert!(page_full_image(&doc, 0).unwrap().is_none());
    }

    #[test]
    fn page_full_image_none_for_multiple_images() {
        // Two images -> not a single full-page scan (heuristic requires exactly one).
        let raw = vec![128u8; 64 * 64];
        let data = compress_to_vec_zlib(&raw, 6);
        let dict = "/Type /XObject /Subtype /Image /Width 64 /Height 64 \
                    /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode";
        let img_a = stream_body(dict, &data);
        let img_b = stream_body(dict, &data);
        let s = scaffold("[0 0 64 64]", "<< /XObject << /Im0 4 0 R /Im1 5 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img_a, img_b];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        // Two images decode fine, but the full-page heuristic declines.
        assert_eq!(page_images(&doc, 0).unwrap().len(), 2);
        assert!(page_full_image(&doc, 0).unwrap().is_none());
    }

    #[test]
    fn jpxdecode_is_unsupported_not_panic() {
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 4 /Height 4 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /JPXDecode",
            b"not really jpeg2000",
        );
        let s = scaffold("[0 0 4 4]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();

        let err = page_images(&doc, 0).unwrap_err();
        assert_eq!(err.kind(), crate::mupdf::ErrorKind::Unsupported);
    }

    #[test]
    fn jbig2decode_is_unsupported_not_panic() {
        let img = stream_body(
            "/Type /XObject /Subtype /Image /Width 4 /Height 4 \
             /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode",
            b"not really jbig2",
        );
        let s = scaffold("[0 0 4 4]", "<< /XObject << /Im0 4 0 R >> >>");
        let bodies = vec![s[0].clone(), s[1].clone(), s[2].clone(), img];
        let doc = PdfDocument::open(build_pdf(&bodies)).unwrap();
        let err = page_images(&doc, 0).unwrap_err();
        assert_eq!(err.kind(), crate::mupdf::ErrorKind::Unsupported);
    }

    // The embedded 16x8 baseline JPEG fixture (`const JPEG_16X8`).
    include!("page_image_test_jpeg.rs");
}
