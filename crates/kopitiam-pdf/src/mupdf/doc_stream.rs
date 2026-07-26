//! Ported from MuPDF `source/pdf/pdf-stream.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the filter-chain wiring follows MuPDF; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # Decoding a PDF stream body
//!
//! A PDF stream is a dict plus a run of raw bytes; the dict's `/Filter` (and
//! `/DecodeParms`) name the decode pipeline. MuPDF's `pdf_open_filter` builds an
//! `fz_stream` chain -- a length-bounded "null" filter over the file, then one
//! decode filter per `/Filter` entry (`build_filter_chain` / `build_filter`) --
//! and reads it through. This port keeps that structure but starts from an
//! already-sliced `Vec<u8>` of raw body bytes (the xref layer resolves `/Length`
//! and slices `[stm_ofs, stm_ofs+Length)` before calling in), then applies the
//! WAVE-2 filters (`filter_basic`, `filter_flate`, `filter_lzw`,
//! `filter_predict`) in order via [`decode_stream`].
//!
//! ## The filter/parms pairing, matched to MuPDF
//!
//! `/Filter` is either a single name or an array; `/DecodeParms` (abbreviation
//! `/DP`) mirrors it. MuPDF's pairing is exact and slightly asymmetric:
//! `build_filter` (name form) hands the whole `/DecodeParms` dict to the one
//! filter, while `build_filter_chain_drop` (array form) indexes
//! `pdf_array_get(params, i)` per filter -- so a *single-dict* `/DecodeParms`
//! against an *array* `/Filter` yields no parms (a dict is not an array).
//! [`normalize`] reproduces that: a name filter takes the parms object directly;
//! an array filter takes `parms[i]` (or none if parms is not an array).
//!
//! ## Predictors
//!
//! For `FlateDecode`/`LZWDecode` MuPDF folds the `/Predictor` into the image
//! decompression path (`build_compression_params` → `fz_open_image_decomp_stream`,
//! pdf-stream.c:170); the observable effect is inflate/LZW followed by the
//! predictor undo. This port applies [`filter_predict::open_predict`] right after
//! the Flate/LZW filter when `/Predictor > 1`, reading `/Columns`, `/Colors`,
//! `/BitsPerComponent` (and `/EarlyChange` for LZW) from that filter's parms.
//!
//! ## Not supported for text (deliberately)
//!
//! The image-only compression filters -- `DCTDecode`/`DCT` (JPEG),
//! `JPXDecode` (JPEG 2000), `JBIG2Decode`, and `CCITTFaxDecode`/`CCF` -- are not
//! glyph/content decoders and are not ported here; [`decode_stream`] returns
//! `FZ_ERROR_UNSUPPORTED` for them (MuPDF opens a real image codec, or, for a
//! non-image caller, warns and yields an empty stream -- pdf-stream.c:237). The
//! `/Crypt` filter (encryption) and external-file streams (`/F` filespec) are
//! likewise out of scope for the WAVE-3b text path. An unrecognised filter name
//! is passed through unchanged, matching `build_filter`'s warn-and-keep-chain.

use super::error::{Error, Result};
use super::filter_basic::{open_a85d, open_ahxd, open_rld};
use super::filter_flate::open_flated;
use super::filter_lzw::open_lzwd_with;
use super::filter_predict::open_predict;
use super::object::Object;
use super::stream::Stream;

/// One resolved link in the decode chain: a filter name and its parameter object
/// (which is [`Object::Null`] when the filter has no `/DecodeParms`).
struct FilterSpec {
    name: Vec<u8>,
    parms: Object,
}

// MuPDF: pdf_dict_geta(Filter, F) / pdf_dict_geta(DecodeParms, DP) pairing in
// pdf_open_filter (pdf-stream.c:395,427) + build_filter_chain_drop indexing.
/// Pair each `/Filter` entry with its `/DecodeParms`, reproducing MuPDF's
/// name-vs-array asymmetry (see the module docs).
fn normalize(filter: &Object, parms: &Object) -> Vec<FilterSpec> {
    let mut out = Vec::new();
    match filter {
        Object::Name(name) => {
            // Name form: the whole parms object belongs to this one filter.
            out.push(FilterSpec {
                name: name.clone(),
                parms: if parms.is_dict() {
                    parms.clone()
                } else {
                    Object::Null
                },
            });
        }
        Object::Array(items) => {
            for (i, f) in items.iter().enumerate() {
                if let Object::Name(name) = f {
                    // Array form: parms indexed as an array (a lone dict → none).
                    let p = parms.array_get(i).cloned().unwrap_or(Object::Null);
                    out.push(FilterSpec {
                        name: name.clone(),
                        parms: p,
                    });
                }
            }
        }
        // No /Filter (null/absent) -- raw, uncompressed bytes.
        _ => {}
    }
    out
}

/// Read an `i32` param from a `/DecodeParms` dict, falling back to `default`.
fn parm_int(parms: &Object, key: &str, default: i32) -> i32 {
    parms.dict_gets(key).map_or(default, |o| o.to_int() as i32)
}

// MuPDF: build_compression_params predictor branch (pdf-stream.c:170) applied via
// fz_open_image_decomp_stream. Undo the /Predictor transform if one is present.
fn maybe_predict<'a>(chain: Stream<'a>, parms: &Object) -> Result<Stream<'a>> {
    let predictor = parm_int(parms, "Predictor", 1);
    if predictor > 1 {
        let columns = parm_int(parms, "Columns", 1);
        let colors = parm_int(parms, "Colors", 1);
        let bpc = parm_int(parms, "BitsPerComponent", 8);
        open_predict(chain, predictor, columns, colors, bpc)
    } else {
        Ok(chain)
    }
}

// MuPDF: build_filter (pdf-stream.c:220) -- one filter by name + param dict.
/// Wrap `chain` in the decode filter named `spec.name`.
fn build_filter<'a>(chain: Stream<'a>, spec: &FilterSpec) -> Result<Stream<'a>> {
    match spec.name.as_slice() {
        b"FlateDecode" | b"Fl" => maybe_predict(open_flated(chain), &spec.parms),
        b"LZWDecode" | b"LZW" => {
            let early = parm_int(&spec.parms, "EarlyChange", 1);
            maybe_predict(open_lzwd_with(chain, early, 9, false, false), &spec.parms)
        }
        b"ASCIIHexDecode" | b"AHx" => Ok(open_ahxd(chain)),
        b"ASCII85Decode" | b"A85" => Ok(open_a85d(chain)),
        b"RunLengthDecode" | b"RL" => Ok(open_rld(chain)),
        // Image-only codecs: not glyph decoders. MuPDF opens an image codec (or,
        // for a non-image caller, warns and yields empty); the text path refuses.
        b"DCTDecode" | b"DCT" | b"JPXDecode" | b"JBIG2Decode" | b"CCITTFaxDecode" | b"CCF" => {
            Err(Error::unsupported(format!(
                "image filter unsupported for text extraction: {}",
                String::from_utf8_lossy(&spec.name)
            )))
        }
        // Unknown filter name: pass through unchanged (build_filter warns and
        // returns the chain untouched, pdf-stream.c:288).
        _ => Ok(chain),
    }
}

// MuPDF: pdf_open_filter + build_filter_chain (pdf-stream.c:393, 332), reading
// the whole decoded stream (fz_read_all).
/// Decode a stream body: apply the `/Filter` (+ `/DecodeParms`) chain to the raw
/// bytes and return the decoded bytes. `filter` and `parms` are the already
/// resolved `/Filter` and `/DecodeParms` objects (either may be [`Object::Null`]);
/// `raw` is the exact `[stm_ofs, stm_ofs+Length)` slice the xref layer produced.
/// With no filters the raw bytes are returned unchanged.
pub fn decode_stream(raw: Vec<u8>, filter: &Object, parms: &Object) -> Result<Vec<u8>> {
    let specs = normalize(filter, parms);
    if specs.is_empty() {
        return Ok(raw);
    }
    let mut chain = Stream::from_vec(raw);
    for spec in &specs {
        chain = build_filter(chain, spec)?;
    }
    chain.read_all()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_oxide::deflate::compress_to_vec_zlib;

    fn name(n: &str) -> Object {
        Object::new_name(n.as_bytes().to_vec())
    }

    #[test]
    fn no_filter_returns_raw() {
        let out = decode_stream(b"hello world".to_vec(), &Object::Null, &Object::Null).unwrap();
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn flate_single_name_filter() {
        let original = b"BT /F1 12 Tf (Hello) Tj ET";
        let compressed = compress_to_vec_zlib(original, 6);
        let out = decode_stream(compressed, &name("FlateDecode"), &Object::Null).unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn flate_with_png_predictor() {
        // PNG-Up predicted rows, then zlib. columns=4, colors=1, bpc=8.
        let orig: [[u8; 4]; 3] = [[5, 10, 15, 20], [6, 12, 18, 24], [7, 14, 21, 28]];
        let mut encoded = Vec::new();
        let mut prev = [0u8; 4];
        for row in &orig {
            encoded.push(2u8); // Up filter
            for i in 0..4 {
                encoded.push(row[i].wrapping_sub(prev[i]));
            }
            prev = *row;
        }
        let compressed = compress_to_vec_zlib(&encoded, 6);

        let mut parms = Object::new_dict();
        parms.dict_put("Predictor", Object::new_int(15));
        parms.dict_put("Columns", Object::new_int(4));

        let out = decode_stream(compressed, &name("FlateDecode"), &parms).unwrap();
        let expected: Vec<u8> = orig.iter().flatten().copied().collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn filter_chain_a85_then_flate() {
        // Encode: original -> zlib -> ASCII85. Decode chain [ASCII85Decode, FlateDecode].
        let original = b"chained filters";
        let compressed = compress_to_vec_zlib(original, 6);
        let a85 = ascii85_encode(&compressed);

        let mut filter = Object::new_array();
        filter.array_push(name("ASCII85Decode"));
        filter.array_push(name("FlateDecode"));

        let out = decode_stream(a85, &filter, &Object::Null).unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn unsupported_image_filter_errors() {
        let err = decode_stream(vec![0u8; 4], &name("DCTDecode"), &Object::Null).unwrap_err();
        assert_eq!(err.kind(), crate::mupdf::ErrorKind::Unsupported);
    }

    #[test]
    fn unknown_filter_passes_through() {
        let out = decode_stream(b"raw".to_vec(), &name("MysteryDecode"), &Object::Null).unwrap();
        assert_eq!(out, b"raw");
    }

    /// Minimal ASCII85 encoder (with the `~>` terminator) for the chain test.
    fn ascii85_encode(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut chunks = data.chunks_exact(4);
        for chunk in &mut chunks {
            let word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            emit85(word, 5, &mut out);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 4];
            buf[..rem.len()].copy_from_slice(rem);
            let word = u32::from_be_bytes(buf);
            emit85(word, rem.len() + 1, &mut out);
        }
        out.extend_from_slice(b"~>");
        out
    }

    fn emit85(word: u32, count: usize, out: &mut Vec<u8>) {
        let mut digits = [0u8; 5];
        let mut w = word;
        for d in digits.iter_mut().rev() {
            *d = (w % 85) as u8 + b'!';
            w /= 85;
        }
        out.extend_from_slice(&digits[..count]);
    }
}
