//! Ported from MuPDF `source/fitz/filter-lzw.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithm and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # `LZWDecode` (PDF / TIFF variant)
//!
//! LZW as PDF and TIFF use it: variable-width codes (min 9 bits, growing to 12),
//! a `CLEAR` code that resets the dictionary, an `EOD` code, and a string table
//! built incrementally. This is a direct translation of `next_lzwd`
//! (filter-lzw.c:71) and `fz_open_lzwd` -- the dictionary, the reverse-string
//! copy, and the code-width growth all follow the C step for step.
//!
//! ## The subtlety that bites: `EarlyChange`
//!
//! PDF's LZW has an `EarlyChange` parameter (default **1**). It shifts *when* the
//! code width grows by one bit: with early change, the width bumps as soon as
//! `next_code > (1 << code_bits) - EarlyChange - 1`, i.e. one code *before* the
//! table would otherwise force it. Encoders and decoders must agree or the whole
//! stream desyncs. MuPDF keeps `early_change` in the filter state and applies it
//! in exactly that comparison (filter-lzw.c:163); this port preserves it
//! verbatim, defaulting to 1 for PDF via [`open_lzwd`]. TIFF's LZW uses
//! `EarlyChange = 0`; [`open_lzwd_with`] exposes it.
//!
//! ## Bit order and the code layout
//!
//! PDF/TIFF LZW packs codes most-significant-bit-first, so this filter reads via
//! [`Stream::read_bits`] (the `reverse_bits` path, for the rare LSB-first TIFF
//! predecessor, is carried but defaults off). `CLEAR = 1 << (min_bits - 1)`
//! (256 for the standard 9-bit start), `EOD = CLEAR + 1` (257), and the first
//! assignable string code is `CLEAR + 2` (258).
//!
//! ## Recoverable-error handling
//!
//! Where MuPDF `fz_warn`s and recovers (missing clear code, a single
//! out-of-range code, premature EOF), this port takes the same recovery branch
//! silently (the warning stream is not ported -- see `error.rs`). The two
//! genuinely fatal cases (`code > next_code`, or an impossible table entry)
//! still return `FZ_ERROR_FORMAT`, matching `next_lzwd`'s `fz_throw`s.

use super::error::{Error, Result};
use super::stream::{Stream, StreamSource};

const MAX_BITS: i32 = 12;
const NUM_CODES: usize = 1 << MAX_BITS; // 4096
const MAX_LENGTH: usize = 4097;

// MuPDF: lzw_code (filter-lzw.c:40) -- one dictionary entry.
#[derive(Clone, Copy)]
struct LzwCode {
    prev: i32,
    length: u16,
    value: u8,
    first_char: u8,
}

struct Lzwd<'a> {
    chain: Stream<'a>,
    eod: bool,

    early_change: i32,
    reverse_bits: bool,
    old_tiff: bool,
    min_bits: i32,
    code_bits: i32,
    code: i32,
    old_code: i32,
    next_code: i32,

    table: Vec<LzwCode>,

    // The decoded-string scratch (`bp`), served out via rp..wp indices.
    bp: Vec<u8>,
    rp: usize,
    wp: usize,

    buffer: Vec<u8>,
}

impl<'a> Lzwd<'a> {
    #[inline]
    fn lzw_clear(&self) -> i32 {
        1 << (self.min_bits - 1)
    }
    #[inline]
    fn lzw_eod(&self) -> i32 {
        self.lzw_clear() + 1
    }
    #[inline]
    fn lzw_first(&self) -> i32 {
        self.lzw_clear() + 2
    }
}

impl<'a> StreamSource for Lzwd<'a> {
    // MuPDF: next_lzwd (filter-lzw.c:71)
    fn refill(&mut self, len: usize) -> Result<&[u8]> {
        self.buffer.clear();
        let cap = len.min(self.buffer.capacity().max(4096)).max(1);
        let cap = cap.min(4096);

        let clear = self.lzw_clear();
        let eod = self.lzw_eod();
        let first = self.lzw_first();

        let mut code_bits = self.code_bits;
        let mut code = self.code;
        let mut old_code = self.old_code;
        let mut next_code = self.next_code;

        // Drain any leftover decoded string from the previous call.
        while self.rp < self.wp && self.buffer.len() < cap {
            self.buffer.push(self.bp[self.rp]);
            self.rp += 1;
        }

        while self.buffer.len() < cap {
            if self.eod {
                break;
            }

            if self.chain.is_eof_bits()? {
                // "premature end in lzw decode" -> recover by ending.
                self.eod = true;
                break;
            }

            code = if self.reverse_bits {
                self.chain.read_rbits(code_bits)? as i32
            } else {
                self.chain.read_bits(code_bits)? as i32
            };

            if code == eod {
                self.eod = true;
                break;
            }

            // Old TIFFs may omit the clear code and overrun at the end.
            if !self.old_tiff && next_code > NUM_CODES as i32 && code != clear {
                // "missing clear code in lzw decode" -> force a clear.
                code = clear;
            }

            if code == clear {
                code_bits = self.min_bits;
                next_code = first;
                old_code = -1;
                continue;
            }

            // If the stream starts without a clear code, old_code is undefined.
            if old_code == -1 {
                old_code = code;
            } else if !self.old_tiff && next_code == NUM_CODES as i32 {
                // Tolerate a single out-of-range code (Ghostscript-like).
                next_code += 1;
            } else if code > next_code || (!self.old_tiff && next_code >= NUM_CODES as i32) {
                return Err(Error::format("out of range code encountered in lzw decode"));
            } else if next_code < NUM_CODES as i32 {
                // Add a new entry to the code table.
                let nc = next_code as usize;
                let oc = old_code as usize;
                self.table[nc].prev = old_code;
                self.table[nc].first_char = self.table[oc].first_char;
                self.table[nc].length = self.table[oc].length + 1;
                if code < next_code {
                    self.table[nc].value = self.table[code as usize].first_char;
                } else if code == next_code {
                    self.table[nc].value = self.table[nc].first_char;
                } else {
                    return Err(Error::format("out of range code encountered in lzw decode"));
                }

                next_code += 1;

                if next_code > (1 << code_bits) - self.early_change - 1 {
                    code_bits += 1;
                    if code_bits > MAX_BITS {
                        code_bits = MAX_BITS;
                    }
                }

                old_code = code;
            }

            // code maps to a string: copy to output in reverse into bp.
            if code >= clear {
                let codelen = self.table[code as usize].length as usize;
                debug_assert!(codelen < MAX_LENGTH);
                self.rp = 0;
                self.wp = codelen;
                let mut s = codelen;
                let mut c = code;
                loop {
                    s -= 1;
                    self.bp[s] = self.table[c as usize].value;
                    c = self.table[c as usize].prev;
                    if !(c >= 0 && s > 0) {
                        break;
                    }
                }
            } else {
                // ... or just a single character.
                self.bp[0] = code as u8;
                self.rp = 0;
                self.wp = 1;
            }

            // Copy the decoded string to output.
            while self.rp < self.wp && self.buffer.len() < cap {
                self.buffer.push(self.bp[self.rp]);
                self.rp += 1;
            }
        }

        self.code_bits = code_bits;
        self.code = code;
        self.old_code = old_code;
        self.next_code = next_code;

        Ok(&self.buffer)
    }
}

// MuPDF: fz_open_lzwd (filter-lzw.c:225)
/// Open an `LZWDecode` filter over `chain` with the PDF defaults
/// (`EarlyChange = 1`, 9-bit initial code size, MSB-first, non-TIFF).
pub fn open_lzwd(chain: Stream<'_>) -> Stream<'_> {
    open_lzwd_with(chain, 1, 9, false, false)
}

// MuPDF: fz_open_lzwd (filter-lzw.c:225) -- full parameter form.
/// Open an `LZWDecode` filter with explicit parameters:
/// `early_change` (PDF: 1, TIFF: 0), `min_bits` (initial code size, usually 9),
/// `reverse_bits` (LSB-first for old TIFF), and `old_tiff` (tolerate a missing
/// clear code and end overrun).
pub fn open_lzwd_with(
    chain: Stream<'_>,
    early_change: i32,
    min_bits: i32,
    reverse_bits: bool,
    old_tiff: bool,
) -> Stream<'_> {
    let mut min_bits = min_bits;
    if min_bits > MAX_BITS {
        // "out of range initial lzw code size".
        min_bits = MAX_BITS;
    }

    let mut table = vec![
        LzwCode {
            prev: -1,
            length: 0,
            value: 0,
            first_char: 0,
        };
        NUM_CODES
    ];
    let clear = 1usize << (min_bits - 1);
    for (i, entry) in table.iter_mut().enumerate().take(clear) {
        entry.value = i as u8;
        entry.first_char = i as u8;
        entry.length = 1;
        entry.prev = -1;
    }

    let first = (clear + 2) as i32;
    Stream::from_source(Box::new(Lzwd {
        chain,
        eod: false,
        early_change,
        reverse_bits,
        old_tiff,
        min_bits,
        code_bits: min_bits,
        code: -1,
        old_code: -1,
        next_code: first,
        table,
        bp: vec![0u8; MAX_LENGTH],
        rp: 0,
        wp: 0,
        buffer: Vec::with_capacity(4096),
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    /// Pack `width` low bits of `code` MSB-first into `out` (via `acc`/`nbits`).
    fn emit(out: &mut Vec<u8>, acc: &mut u32, nbits: &mut u32, code: u32, width: u32) {
        *acc = (*acc << width) | code;
        *nbits += width;
        while *nbits >= 8 {
            *nbits -= 8;
            out.push((*acc >> *nbits) as u8);
        }
    }

    /// Encode a byte sequence to PDF LZW (MSB-first, EarlyChange=1, 9-bit start)
    /// so the decoder can be cross-checked against an independent encoder. The
    /// width-bump condition mirrors the decoder's exactly
    /// (`next_code > (1<<width) - 2`), keeping the two in lockstep. Callers must
    /// keep the dictionary under 4096 entries (i.e. input length < 4096) so no
    /// table-full CLEAR is needed -- which this encoder does not emit.
    fn lzw_encode_pdf(data: &[u8]) -> Vec<u8> {
        const CLEAR: u32 = 256;
        const EOD: u32 = 257;
        assert!(
            data.len() < 4096,
            "test encoder does not emit table-full CLEAR"
        );

        let mut out: Vec<u8> = Vec::new();
        let mut acc: u32 = 0;
        let mut nbits: u32 = 0;

        let mut dict: HashMap<Vec<u8>, u32> = HashMap::new();
        for i in 0..256u32 {
            dict.insert(vec![i as u8], i);
        }
        let mut next_code: u32 = 258;
        let mut code_width: u32 = 9;

        emit(&mut out, &mut acc, &mut nbits, CLEAR, code_width);

        let mut w: Vec<u8> = Vec::new();
        for &b in data {
            let mut wb = w.clone();
            wb.push(b);
            if dict.contains_key(&wb) {
                w = wb;
            } else {
                emit(&mut out, &mut acc, &mut nbits, dict[&w], code_width);
                dict.insert(wb, next_code);
                next_code += 1;
                // EarlyChange = 1. The decoder's first post-CLEAR read adds no
                // dictionary entry (its `old_code == -1` branch), so the decoder
                // lags the encoder by exactly one entry. To bump the code width
                // on the same emit/read the decoder does, evaluate the decoder's
                // bump test (`next_code_dec > (1<<w) - 2`) against the decoder's
                // count `next_code - 1`, i.e. `next_code > (1<<w) - 1`.
                if next_code > (1u32 << code_width) - 1 && code_width < 12 {
                    code_width += 1;
                }
                w = vec![b];
            }
        }
        if !w.is_empty() {
            emit(&mut out, &mut acc, &mut nbits, dict[&w], code_width);
        }
        emit(&mut out, &mut acc, &mut nbits, EOD, code_width);
        if nbits > 0 {
            out.push((acc << (8 - nbits)) as u8);
        }
        out
    }

    fn decode(input: &[u8]) -> Result<Vec<u8>> {
        open_lzwd(Stream::from_slice(input)).read_all()
    }

    #[test]
    fn lzw_round_trip_repeated_runs() {
        // Repeated runs build multi-byte dictionary strings quickly -- the
        // classic "-----AAAAA" shape from the PDF spec's LZW discussion.
        let original: &[u8] = b"-----AAAAA-----AAAAA";
        let encoded = lzw_encode_pdf(original);
        assert_eq!(decode(&encoded).unwrap(), original);
    }

    #[test]
    fn lzw_round_trip_text() {
        let original = b"TOBEORNOTTOBEORTOBEORNOT";
        let encoded = lzw_encode_pdf(original);
        assert_eq!(decode(&encoded).unwrap(), original);
    }

    #[test]
    fn lzw_round_trip_forces_code_width_growth() {
        // 3000 (< 4096) high-entropy bytes add close to 3000 dictionary entries,
        // crossing the 9->10 (>510), 10->11 (>1022) and 11->12 (>2046)
        // EarlyChange width bumps without ever filling the table.
        let original: Vec<u8> = (0..3000u32).map(|i| ((i * 37) & 0xff) as u8).collect();
        let encoded = lzw_encode_pdf(&original);
        assert_eq!(decode(&encoded).unwrap(), original);
    }

    #[test]
    fn lzw_all_bytes_round_trip() {
        let original: Vec<u8> = (0..=255u8).collect();
        let encoded = lzw_encode_pdf(&original);
        assert_eq!(decode(&encoded).unwrap(), original);
    }

    #[test]
    fn lzw_empty_input() {
        // CLEAR then EOD only.
        let encoded = lzw_encode_pdf(b"");
        assert_eq!(decode(&encoded).unwrap(), b"");
    }
}
