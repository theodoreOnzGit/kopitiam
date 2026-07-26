//! Ported from MuPDF `source/fitz/filter-predict.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithm and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # Predictors (PNG + TIFF)
//!
//! A predictor is a reversible transform applied *before* Flate/LZW compression
//! to make image rows compress better; the decode filter undoes it. PDF's
//! `/Predictor` selects the family:
//!
//! * `1` -- no prediction (identity copy).
//! * `2` -- TIFF Predictor 2: horizontal differencing per colour component.
//! * `10`..`15` -- PNG predictors: each *row* is prefixed by a filter-type byte
//!   (0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth) chosen per row by the encoder;
//!   the `/Predictor` value only signals "PNG family".
//!
//! This is a direct translation of `next_predict`, `fz_predict_tiff`,
//! `fz_predict_png`, `getcomponent`/`putcomponent`, and `paeth`
//! (filter-predict.c). The parameters `columns`, `colors`, `bpc` come from the
//! stream's `/DecodeParms`.
//!
//! ## Row handling, preserved exactly
//!
//! * A row is `stride = ceil(bpc*colors*columns / 8)` bytes; PNG rows carry one
//!   extra leading filter-type byte, so the filter reads `stride + ispng` bytes
//!   per row (`ispng = predictor >= 10`).
//! * The PNG predictors reference the **previous decoded row** (`ref`), which
//!   starts zeroed; after decoding a row it is copied to `ref` for the next.
//!   `bpp = ceil(bpc*colors / 8)` is the per-pixel byte step used by Sub/Average/
//!   Paeth, clamped down to the row length for a short final row.
//! * TIFF differencing walks components left-to-right, `bpc == 8` taking a fast
//!   byte path and other depths going through `getcomponent`/`putcomponent`
//!   (which assume zeroed output for `bpc < 8`, so the row is `memset` to 0
//!   first).
//! * The Paeth predictor's `ac`/`bc` definitions are MuPDF's (the "not a typo"
//!   comment is preserved) and the tie-break order `pa <= pb && pa <= pc` is
//!   kept verbatim.
//!
//! The decoder serves output in bursts (bounded by the `max` hint, cap 4096, as
//! in the C's `state->buffer`), carrying a partly-consumed row across `refill`s.

use super::error::{Error, Result};
use super::stream::{Stream, StreamSource};

/// `FZ_MAX_COLORS` (color.h:102).
const FZ_MAX_COLORS: usize = 32;

// MuPDF: getcomponent (filter-predict.c:49)
fn getcomponent(line: &[u8], x: usize, bpc: i32) -> i32 {
    match bpc {
        1 => ((line[x >> 3] >> (7 - (x & 7))) & 1) as i32,
        2 => ((line[x >> 2] >> ((3 - (x & 3)) << 1)) & 3) as i32,
        4 => ((line[x >> 1] >> ((1 - (x & 1)) << 2)) & 15) as i32,
        8 => line[x] as i32,
        16 => ((line[x << 1] as i32) << 8) + line[(x << 1) + 1] as i32,
        _ => 0,
    }
}

// MuPDF: putcomponent (filter-predict.c:62)
fn putcomponent(buf: &mut [u8], x: usize, bpc: i32, value: i32) {
    let value = value as u32;
    match bpc {
        1 => buf[x >> 3] |= (value << (7 - (x & 7))) as u8,
        2 => buf[x >> 2] |= (value << ((3 - (x & 3)) << 1)) as u8,
        4 => buf[x >> 1] |= (value << ((1 - (x & 1)) << 2)) as u8,
        8 => buf[x] = value as u8,
        16 => {
            buf[x << 1] = (value >> 8) as u8;
            buf[(x << 1) + 1] = value as u8;
        }
        _ => {}
    }
}

// MuPDF: paeth (filter-predict.c:74)
fn paeth(a: i32, b: i32, c: i32) -> i32 {
    // The definitions of ac and bc are correct, not a typo.
    let ac = b - c;
    let bc = a - c;
    let abcc = ac + bc;
    let pa = ac.abs();
    let pb = bc.abs();
    let pc = abcc.abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

// MuPDF: fz_predict_tiff (filter-predict.c:84)
fn predict_tiff(out: &mut [u8], input: &[u8], colors: usize, columns: usize, bpc: i32, stride: usize) {
    let mut left = [0i32; FZ_MAX_COLORS];
    let mask = (1i32 << bpc) - 1;

    // special fast case
    if bpc == 8 {
        let mut o = 0;
        let mut i = 0;
        for _ in 0..columns {
            for k in left.iter_mut().take(colors) {
                *k = (input[i] as i32 + *k) & 0xFF;
                out[o] = *k as u8;
                o += 1;
                i += 1;
            }
        }
        return;
    }

    // putcomponent assumes zeroed memory for bpc < 8
    if bpc < 8 {
        for b in out.iter_mut().take(stride) {
            *b = 0;
        }
    }

    for i in 0..columns {
        for (k, leftk) in left.iter_mut().enumerate().take(colors) {
            let a = getcomponent(input, i * colors + k, bpc);
            let b = a + *leftk;
            let c = b & mask;
            putcomponent(out, i * colors + k, bpc, c);
            *leftk = c;
        }
    }
}

// MuPDF: fz_predict_png (filter-predict.c:120)
fn predict_png(out: &mut [u8], input: &[u8], ref_row: &[u8], bpp: usize, len: usize, predictor: i32) {
    let bpp = if bpp > len { len } else { bpp };

    match predictor {
        0 => {
            out[..len].copy_from_slice(&input[..len]);
        }
        1 => {
            out[..bpp].copy_from_slice(&input[..bpp]);
            for i in bpp..len {
                out[i] = input[i].wrapping_add(out[i - bpp]);
            }
        }
        2 => {
            for i in 0..len {
                out[i] = input[i].wrapping_add(ref_row[i]);
            }
        }
        3 => {
            for i in 0..bpp {
                out[i] = input[i].wrapping_add(ref_row[i] / 2);
            }
            for i in bpp..len {
                let avg = ((out[i - bpp] as i32 + ref_row[i] as i32) / 2) as u8;
                out[i] = input[i].wrapping_add(avg);
            }
        }
        4 => {
            for i in 0..bpp {
                out[i] = input[i].wrapping_add(paeth(0, ref_row[i] as i32, 0) as u8);
            }
            for i in bpp..len {
                let p = paeth(
                    out[i - bpp] as i32,
                    ref_row[i] as i32,
                    ref_row[i - bpp] as i32,
                );
                out[i] = input[i].wrapping_add(p as u8);
            }
        }
        _ => {
            // unknown png predictor, treat as none
            out[..len].copy_from_slice(&input[..len]);
        }
    }
}

struct Predict<'a> {
    chain: Stream<'a>,
    predictor: i32,
    columns: usize,
    colors: usize,
    bpc: i32,
    stride: usize,
    bpp: usize,
    in_row: Vec<u8>,
    out: Vec<u8>,
    ref_row: Vec<u8>,
    rp: usize,
    wp: usize,
    buffer: Vec<u8>,
}

impl<'a> StreamSource for Predict<'a> {
    // MuPDF: next_predict (filter-predict.c:185)
    fn refill(&mut self, len: usize) -> Result<&[u8]> {
        self.buffer.clear();
        let cap = len.clamp(1, 4096);
        let ispng = if self.predictor >= 10 { 1 } else { 0 };

        // Drain leftover from the previous row.
        while self.rp < self.wp && self.buffer.len() < cap {
            self.buffer.push(self.out[self.rp]);
            self.rp += 1;
        }

        while self.buffer.len() < cap {
            let want = self.stride + ispng;
            // Zero the input row so a short final read leaves determinate bytes.
            for b in self.in_row.iter_mut().take(want) {
                *b = 0;
            }
            let n = self.chain.read(&mut self.in_row[..want])?;
            if n == 0 {
                break;
            }

            if self.predictor == 1 {
                self.out[..n].copy_from_slice(&self.in_row[..n]);
            } else if self.predictor == 2 {
                predict_tiff(
                    &mut self.out,
                    &self.in_row,
                    self.colors,
                    self.columns,
                    self.bpc,
                    self.stride,
                );
            } else {
                let filter = self.in_row[0] as i32;
                predict_png(
                    &mut self.out,
                    &self.in_row[1..],
                    &self.ref_row,
                    self.bpp,
                    n - 1,
                    filter,
                );
                self.ref_row[..self.stride].copy_from_slice(&self.out[..self.stride]);
            }

            self.rp = 0;
            self.wp = n - ispng;

            while self.rp < self.wp && self.buffer.len() < cap {
                self.buffer.push(self.out[self.rp]);
                self.rp += 1;
            }
        }

        Ok(&self.buffer)
    }
}

// MuPDF: fz_open_predict (filter-predict.c:245)
/// Open a predictor-decode filter over `chain`. `predictor` is the PDF
/// `/Predictor` value (1 = none, 2 = TIFF, 10..15 = PNG); `columns`, `colors`,
/// `bpc` come from `/DecodeParms`. Returns `FZ_ERROR_ARGUMENT`/`FZ_ERROR_LIMIT`
/// on invalid parameters, exactly as MuPDF.
pub fn open_predict(
    chain: Stream<'_>,
    predictor: i32,
    columns: i32,
    colors: i32,
    bpc: i32,
) -> Result<Stream<'_>> {
    let mut predictor = predictor;
    let mut columns = columns;
    let mut colors = colors;
    let mut bpc = bpc;
    if predictor < 1 {
        predictor = 1;
    }
    if columns < 1 {
        columns = 1;
    }
    if colors < 1 {
        colors = 1;
    }
    if bpc < 1 {
        bpc = 8;
    }

    if bpc != 1 && bpc != 2 && bpc != 4 && bpc != 8 && bpc != 16 {
        return Err(Error::argument(format!(
            "invalid number of bits per component: {bpc}"
        )));
    }
    if colors as usize > FZ_MAX_COLORS {
        return Err(Error::argument(format!(
            "too many color components ({colors} > {FZ_MAX_COLORS})"
        )));
    }
    if columns >= i32::MAX / (bpc * colors) {
        return Err(Error::limit(format!(
            "too many columns lead to an integer overflow ({columns})"
        )));
    }

    if predictor != 1
        && predictor != 2
        && predictor != 10
        && predictor != 11
        && predictor != 12
        && predictor != 13
        && predictor != 14
        && predictor != 15
    {
        // invalid predictor -> treat as none
        predictor = 1;
    }

    let columns = columns as usize;
    let colors = colors as usize;
    // MuPDF: (bpc*colors*columns + 7) / 8 -- ceil-div to whole bytes.
    let stride = (bpc as usize * colors * columns).div_ceil(8);
    let bpp = (bpc as usize * colors).div_ceil(8);

    Ok(Stream::from_source(Box::new(Predict {
        chain,
        predictor,
        columns,
        colors,
        bpc,
        stride,
        bpp,
        in_row: vec![0u8; stride + 1],
        out: vec![0u8; stride],
        ref_row: vec![0u8; stride],
        rp: 0,
        wp: 0,
        buffer: Vec::with_capacity(4096),
    })))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mupdf::filter_flate::open_flated;
    use miniz_oxide::deflate::compress_to_vec_zlib;

    #[test]
    fn tiff_predictor_horizontal_differencing() {
        // Original single row [10,15,12,20], colors=1, bpc=8, columns=4.
        // TIFF-encoded (left differences): [10, 5, -3=253, 8].
        let encoded = [10u8, 5, 253, 8];
        let mut s = open_predict(Stream::from_slice(&encoded), 2, 4, 1, 8).unwrap();
        assert_eq!(s.read_all().unwrap(), &[10, 15, 12, 20]);
    }

    #[test]
    fn tiff_predictor_multi_component() {
        // colors=2, columns=2, bpc=8. Original interleaved [R0,G0, R1,G1]:
        // [10,20, 13,25]. Differences per component: [10,20, 3,5].
        let encoded = [10u8, 20, 3, 5];
        let mut s = open_predict(Stream::from_slice(&encoded), 2, 2, 2, 8).unwrap();
        assert_eq!(s.read_all().unwrap(), &[10, 20, 13, 25]);
    }

    #[test]
    fn png_sub_predictor() {
        // One row, Sub filter (byte 1). data [10,5,253,8] -> [10,15,12,20].
        let encoded = [1u8, 10, 5, 253, 8];
        let mut s = open_predict(Stream::from_slice(&encoded), 15, 4, 1, 8).unwrap();
        assert_eq!(s.read_all().unwrap(), &[10, 15, 12, 20]);
    }

    #[test]
    fn png_up_predictor_two_rows() {
        // Row0 Up (ref=0): filter 2, data = orig0 = [5,10,15,20].
        // Row1 Up: filter 2, data = orig1 - orig0 = [1,1,1,1] -> orig1 [6,11,16,21].
        let encoded = [2u8, 5, 10, 15, 20, 2u8, 1, 1, 1, 1];
        let mut s = open_predict(Stream::from_slice(&encoded), 12, 4, 1, 8).unwrap();
        assert_eq!(s.read_all().unwrap(), &[5, 10, 15, 20, 6, 11, 16, 21]);
    }

    #[test]
    fn png_none_predictor_passes_through() {
        // filter byte 0 = None: row copied verbatim.
        let encoded = [0u8, 1, 2, 3, 4];
        let mut s = open_predict(Stream::from_slice(&encoded), 10, 4, 1, 8).unwrap();
        assert_eq!(s.read_all().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn flate_plus_png_predictor_end_to_end() {
        // The real content-stream shape: PNG-Up-predicted rows, zlib compressed.
        let orig: [[u8; 4]; 3] = [[5, 10, 15, 20], [6, 12, 18, 24], [7, 14, 21, 28]];
        // Encode each row with the Up filter (2) against the previous row.
        let mut encoded = Vec::new();
        let mut prev = [0u8; 4];
        for row in &orig {
            encoded.push(2u8); // Up
            for i in 0..4 {
                encoded.push(row[i].wrapping_sub(prev[i]));
            }
            prev = *row;
        }
        let compressed = compress_to_vec_zlib(&encoded, 6);

        // Decode: Flate -> Predict(PNG, columns=4, colors=1, bpc=8).
        let flate = open_flated(Stream::from_slice(&compressed));
        let mut predict = open_predict(flate, 15, 4, 1, 8).unwrap();
        let out = predict.read_all().unwrap();

        let expected: Vec<u8> = orig.iter().flatten().copied().collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn invalid_bpc_errors() {
        assert!(open_predict(Stream::from_slice(b""), 2, 4, 1, 3).is_err());
    }

    #[test]
    fn too_many_colors_errors() {
        assert!(open_predict(Stream::from_slice(b""), 2, 4, 33, 8).is_err());
    }
}
