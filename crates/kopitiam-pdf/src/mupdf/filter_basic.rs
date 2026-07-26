//! Ported from MuPDF `source/fitz/filter-basic.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the algorithms and numeric behaviour follow MuPDF; the code
//! is re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # The basic PDF decode filters
//!
//! Each filter is a [`StreamSource`] that wraps an inner [`Stream`] (the
//! "chain") and decodes bytes from it on demand -- the direct translation of
//! MuPDF's `fz_open_*` filter constructors, whose `next` callbacks pull from
//! `state->chain`. Ported here:
//!
//! * [`open_ahxd`] -- `ASCIIHexDecode` (`fz_open_ahxd`).
//! * [`open_a85d`] -- `ASCII85Decode` (`fz_open_a85d`).
//! * [`open_rld`] -- `RunLengthDecode` (`fz_open_rld`).
//! * [`open_null`] -- the length-bounded copy filter (`fz_open_null_filter`).
//!
//! ## Subtleties preserved
//!
//! * **ASCIIHex** treats an odd trailing nibble as if padded with 0 (`a << 4`),
//!   ends on `>`, ignores whitespace, and throws `FZ_ERROR_FORMAT` on any other
//!   byte -- exactly as `next_ahxd`.
//! * **ASCII85** keeps the `z` shortcut (a lone `z`, only mid-group, expands to
//!   four zero bytes), the `~>` end-of-data marker, and the partial-final-group
//!   arithmetic (`word * 85^k + 0xFF..` then take the high bytes) for group
//!   sizes 2/3/4. A group size of 1 is illegal per spec but tolerated (dropped),
//!   matching Adobe/Ghostscript and `next_a85d`'s comment.
//! * **RunLength** reads a length byte `l`: `l < 128` copies `l+1` literal bytes;
//!   `l > 128` repeats the next byte `257-l` times; `l == 128` is the
//!   end-of-data marker. Premature EOF inside a run throws `FZ_ERROR_FORMAT`.
//!
//! The MuPDF filters decode in 256-byte bursts into `state->buffer`; this port
//! decodes into a `Vec<u8>` window per `refill`, bounded by the `max` hint (cap
//! 256, as in the C).
//!
//! ## Not ported (deliberately)
//!
//! The `range`, `endstream`, and `concat` filters (xref/object-stream plumbing,
//! not glyph decoding) and the RC4/AES crypt filters (encryption, a separate
//! module) are out of scope for WAVE 2. `fz_open_null_filter`'s `offset`
//! argument (it re-seeks the chain to an absolute offset each refill, for
//! reading a slice of a shared stream) is dropped: [`open_null`] assumes the
//! chain is already positioned and simply bounds it to `len` bytes -- the "copy
//! filter" essence. Port the offset form when the xref layer needs it.

use super::error::{Error, Result};
use super::stream::{Stream, StreamSource};

/// Cap on a single decode burst, matching the C filters' 256-byte `state->buffer`.
const BURST: usize = 256;

// MuPDF: iswhite (filter-basic.c:415) -- the filter whitespace set (note it
// includes NUL, backspace, and 0177/DEL, which differs from PDF's usual set).
fn iswhite(a: i32) -> bool {
    matches!(
        a,
        0x0a | 0x0d | 0x09 | 0x20 | 0x00 | 0x0c | 0x08 | 0o177
    )
}

// ---------------------------------------------------------------------------
// ASCIIHexDecode (fz_ahxd)
// ---------------------------------------------------------------------------

// MuPDF: ishex (filter-basic.c:425)
fn ishex(a: i32) -> bool {
    (b'A' as i32..=b'F' as i32).contains(&a)
        || (b'a' as i32..=b'f' as i32).contains(&a)
        || (b'0' as i32..=b'9' as i32).contains(&a)
}

// MuPDF: unhex (filter-basic.c:432)
fn unhex(a: i32) -> i32 {
    if (b'A' as i32..=b'F' as i32).contains(&a) {
        a - b'A' as i32 + 0xA
    } else if (b'a' as i32..=b'f' as i32).contains(&a) {
        a - b'a' as i32 + 0xA
    } else if (b'0' as i32..=b'9' as i32).contains(&a) {
        a - b'0' as i32
    } else {
        0
    }
}

struct Ahxd<'a> {
    chain: Stream<'a>,
    eod: bool,
    buffer: Vec<u8>,
}

impl<'a> StreamSource for Ahxd<'a> {
    // MuPDF: next_ahxd (filter-basic.c:440)
    fn refill(&mut self, max: usize) -> Result<&[u8]> {
        let max = max.clamp(1, BURST);
        self.buffer.clear();
        let mut a = 0i32;
        let mut odd = false;

        while self.buffer.len() < max {
            if self.eod {
                break;
            }
            let c = self.chain.read_byte()?;
            let c = match c {
                Some(b) => b as i32,
                None => break,
            };

            if ishex(c) {
                if !odd {
                    a = unhex(c);
                    odd = true;
                } else {
                    let b = unhex(c);
                    self.buffer.push(((a << 4) | b) as u8);
                    odd = false;
                }
            } else if c == b'>' as i32 {
                if odd {
                    self.buffer.push((a << 4) as u8);
                }
                self.eod = true;
                break;
            } else if !iswhite(c) {
                return Err(Error::format(format!(
                    "bad data in ahxd: '{}'",
                    (c as u8) as char
                )));
            }
        }
        Ok(&self.buffer)
    }
}

// MuPDF: fz_open_ahxd (filter-basic.c:506)
/// Open an `ASCIIHexDecode` filter over `chain`.
pub fn open_ahxd(chain: Stream<'_>) -> Stream<'_> {
    Stream::from_source(Box::new(Ahxd {
        chain,
        eod: false,
        buffer: Vec::with_capacity(BURST),
    }))
}

// ---------------------------------------------------------------------------
// ASCII85Decode (fz_a85d)
// ---------------------------------------------------------------------------

struct A85d<'a> {
    chain: Stream<'a>,
    eod: bool,
    buffer: Vec<u8>,
}

impl<'a> StreamSource for A85d<'a> {
    // MuPDF: next_a85d (filter-basic.c:524)
    fn refill(&mut self, max: usize) -> Result<&[u8]> {
        self.buffer.clear();
        if self.eod {
            return Ok(&self.buffer);
        }
        let max = max.clamp(1, BURST);

        let mut count = 0u32;
        let mut word = 0u32;

        while self.buffer.len() < max {
            let c = match self.chain.read_byte()? {
                Some(b) => b as i32,
                None => break,
            };

            if (b'!' as i32..=b'u' as i32).contains(&c) {
                if count == 4 {
                    word = word.wrapping_mul(85).wrapping_add((c - b'!' as i32) as u32);
                    self.buffer.push((word >> 24) as u8);
                    self.buffer.push((word >> 16) as u8);
                    self.buffer.push((word >> 8) as u8);
                    self.buffer.push(word as u8);
                    word = 0;
                    count = 0;
                } else {
                    word = word.wrapping_mul(85).wrapping_add((c - b'!' as i32) as u32);
                    count += 1;
                }
            } else if c == b'z' as i32 && count == 0 {
                self.buffer.push(0);
                self.buffer.push(0);
                self.buffer.push(0);
                self.buffer.push(0);
            } else if c == b'~' as i32 {
                // End-of-data. The '~' should be followed by '>'.
                let _ = self.chain.read_byte()?; // consume/ignore the '>' (MuPDF warns on mismatch)
                match count {
                    0 => {}
                    1 => {
                        // Illegal per spec but tolerated (Adobe/gs cope).
                    }
                    2 => {
                        word = word
                            .wrapping_mul(85 * 85 * 85)
                            .wrapping_add(0x00ff_ffff);
                        self.buffer.push((word >> 24) as u8);
                    }
                    3 => {
                        word = word.wrapping_mul(85 * 85).wrapping_add(0x0000_ffff);
                        self.buffer.push((word >> 24) as u8);
                        self.buffer.push((word >> 16) as u8);
                    }
                    4 => {
                        word = word.wrapping_mul(85).wrapping_add(0x0000_00ff);
                        self.buffer.push((word >> 24) as u8);
                        self.buffer.push((word >> 16) as u8);
                        self.buffer.push((word >> 8) as u8);
                    }
                    _ => unreachable!(),
                }
                self.eod = true;
                break;
            } else if !iswhite(c) {
                return Err(Error::format(format!(
                    "bad data in a85d: '{}'",
                    (c as u8) as char
                )));
            }
        }
        Ok(&self.buffer)
    }
}

// MuPDF: fz_open_a85d (filter-basic.c:635)
/// Open an `ASCII85Decode` filter over `chain`.
pub fn open_a85d(chain: Stream<'_>) -> Stream<'_> {
    Stream::from_source(Box::new(A85d {
        chain,
        eod: false,
        buffer: Vec::with_capacity(BURST),
    }))
}

// ---------------------------------------------------------------------------
// RunLengthDecode (fz_rld)
// ---------------------------------------------------------------------------

struct Rld<'a> {
    chain: Stream<'a>,
    run: i32,
    n: i32,
    c: i32,
    buffer: Vec<u8>,
}

impl<'a> StreamSource for Rld<'a> {
    // MuPDF: next_rld (filter-basic.c:653)
    fn refill(&mut self, max: usize) -> Result<&[u8]> {
        self.buffer.clear();
        if self.run == 128 {
            return Ok(&self.buffer);
        }
        let max = max.clamp(1, BURST);

        while self.buffer.len() < max {
            if self.run == 128 {
                break;
            }

            if self.n == 0 {
                self.run = self.chain.read_byte()?.map_or(-1, |b| b as i32);
                if self.run < 0 {
                    self.run = 128;
                    break;
                }
                if self.run < 128 {
                    self.n = self.run + 1;
                }
                if self.run > 128 {
                    self.n = 257 - self.run;
                    self.c = self.chain.read_byte()?.map_or(-1, |b| b as i32);
                    if self.c < 0 {
                        return Err(Error::format(
                            "premature end of data in run length decode",
                        ));
                    }
                }
            }

            if self.run < 128 {
                while self.buffer.len() < max && self.n != 0 {
                    let c = self.chain.read_byte()?.map_or(-1, |b| b as i32);
                    if c < 0 {
                        return Err(Error::format(
                            "premature end of data in run length decode",
                        ));
                    }
                    self.buffer.push(c as u8);
                    self.n -= 1;
                }
            }

            if self.run > 128 {
                while self.buffer.len() < max && self.n != 0 {
                    self.buffer.push(self.c as u8);
                    self.n -= 1;
                }
            }
        }
        Ok(&self.buffer)
    }
}

// MuPDF: fz_open_rld (filter-basic.c:731)
/// Open a `RunLengthDecode` filter over `chain`.
pub fn open_rld(chain: Stream<'_>) -> Stream<'_> {
    Stream::from_source(Box::new(Rld {
        chain,
        run: 0,
        n: 0,
        c: 0,
        buffer: Vec::with_capacity(BURST),
    }))
}

// ---------------------------------------------------------------------------
// Null / copy filter (fz_open_null_filter, length-bounded form)
// ---------------------------------------------------------------------------

struct NullFilter<'a> {
    chain: Stream<'a>,
    remain: u64,
    buffer: Vec<u8>,
}

impl<'a> StreamSource for NullFilter<'a> {
    // MuPDF: next_null (filter-basic.c:37) -- minus the absolute-offset re-seek.
    fn refill(&mut self, max: usize) -> Result<&[u8]> {
        self.buffer.clear();
        if self.remain == 0 {
            return Ok(&self.buffer);
        }
        let chunk = self.chain.fill_buf(max)?;
        if chunk.is_empty() {
            return Ok(&self.buffer);
        }
        let mut n = chunk.len() as u64;
        if n > self.remain {
            n = self.remain;
        }
        let n = n as usize;
        self.buffer.extend_from_slice(&chunk[..n]);
        self.chain.consume(n);
        self.remain -= n as u64;
        Ok(&self.buffer)
    }
}

// MuPDF: fz_open_null_filter (filter-basic.c:73)
/// Open a length-bounded copy ("null") filter that passes at most `len` bytes
/// from `chain` unchanged. (The C `offset` argument, an absolute re-seek per
/// refill, is not modelled -- the chain must already be positioned.)
pub fn open_null(chain: Stream<'_>, len: u64) -> Stream<'_> {
    Stream::from_source(Box::new(NullFilter {
        chain,
        remain: len,
        buffer: Vec::new(),
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_ahxd(input: &[u8]) -> Result<Vec<u8>> {
        open_ahxd(Stream::from_slice(input)).read_all()
    }
    fn decode_a85d(input: &[u8]) -> Result<Vec<u8>> {
        open_a85d(Stream::from_slice(input)).read_all()
    }
    fn decode_rld(input: &[u8]) -> Result<Vec<u8>> {
        open_rld(Stream::from_slice(input)).read_all()
    }

    #[test]
    fn ahxd_basic_and_whitespace() {
        // "48 65 6C 6C 6F" = "Hello", with spaces/newlines, terminated by '>'.
        assert_eq!(decode_ahxd(b"48 65 6C 6C 6F>").unwrap(), b"Hello");
        assert_eq!(decode_ahxd(b"48\n65\t6c6C6f >").unwrap(), b"Hello");
    }

    #[test]
    fn ahxd_odd_nibble_is_zero_padded() {
        // "4" alone -> 0x40 = '@'.
        assert_eq!(decode_ahxd(b"4>").unwrap(), b"\x40");
        // "414" -> 0x41, 0x40.
        assert_eq!(decode_ahxd(b"414>").unwrap(), &[0x41, 0x40]);
    }

    #[test]
    fn ahxd_without_eod_marker_reads_all_hex() {
        assert_eq!(decode_ahxd(b"4869").unwrap(), b"Hi");
    }

    #[test]
    fn ahxd_bad_byte_errors() {
        assert!(decode_ahxd(b"48XY>").is_err());
    }

    #[test]
    fn a85d_known_vector() {
        // Wikipedia's canonical ASCII85 example: "Man " (0x4D616E20) -> "9jqo^".
        assert_eq!(decode_a85d(b"9jqo^~>").unwrap(), b"Man ");
        // A full 5-char group "87cUR" decodes the 4 bytes "Hell" (0x48656C6C).
        assert_eq!(decode_a85d(b"87cUR~>").unwrap(), b"Hell");
    }

    #[test]
    fn a85d_z_shortcut_is_four_zeros() {
        // A lone 'z' expands to four zero bytes; then a full group.
        let out = decode_a85d(b"z~>").unwrap();
        assert_eq!(out, &[0, 0, 0, 0]);
    }

    #[test]
    fn a85d_partial_final_group() {
        // Encode a 1-byte payload 0x00: "!!" then "~>"? Verify via round trip of
        // a known 2-char partial group. "<~z!!~>" style. Here test 2 leftover
        // chars decode to 1 byte. "!!~>": count=2 -> one output byte.
        // '!' - '!' = 0, so word stays 0; word*85^3 + 0xffffff = 0x00ffffff;
        // high byte = 0x00.
        assert_eq!(decode_a85d(b"!!~>").unwrap(), &[0x00]);
    }

    #[test]
    fn a85d_whitespace_ignored() {
        // Same "Man " vector with interspersed whitespace.
        assert_eq!(decode_a85d(b"9j\nq o\t^~>").unwrap(), b"Man ");
    }

    #[test]
    fn rld_literal_run() {
        // length 4 (=5 literal bytes) then 5 bytes "Hello".
        let mut input = vec![4u8];
        input.extend_from_slice(b"Hello");
        input.push(128); // EOD
        assert_eq!(decode_rld(&input).unwrap(), b"Hello");
    }

    #[test]
    fn rld_repeat_run() {
        // 257 - 250 = 7 copies of 'A' (0x41). length byte 250, value 0x41.
        let input = [250u8, 0x41, 128];
        assert_eq!(decode_rld(&input).unwrap(), b"AAAAAAA");
    }

    #[test]
    fn rld_mixed_and_eod() {
        // literal "ab" (len 1), repeat 'C' x3 (len 254), EOD.
        let input = [1u8, b'a', b'b', 254, b'C', 128, /* trailing ignored */ 9];
        assert_eq!(decode_rld(&input).unwrap(), b"abCCC");
    }

    #[test]
    fn rld_premature_eof_errors() {
        // Says 5 literal bytes but only 2 present.
        let input = [4u8, b'a', b'b'];
        assert!(decode_rld(&input).is_err());
    }

    #[test]
    fn null_filter_bounds_length() {
        let mut s = open_null(Stream::from_slice(b"abcdefgh"), 3);
        assert_eq!(s.read_all().unwrap(), b"abc");
    }

    #[test]
    fn null_filter_stops_at_chain_eof() {
        let mut s = open_null(Stream::from_slice(b"ab"), 100);
        assert_eq!(s.read_all().unwrap(), b"ab");
    }
}
