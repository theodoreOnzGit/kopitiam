//! Ported from MuPDF `source/fitz/filter-flate.c` (commit 19f1284, AGPL-3.0,
//! © Artifex Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the *filter wiring* follows MuPDF; the *inflate algorithm*
//! is not MuPDF's (MuPDF calls `<zlib.h>`) -- this port substitutes the
//! pure-Rust **`miniz_oxide`** crate, itself a translation of Mark Adler's
//! zlib/miniz. Both implement RFC 1950/1951 DEFLATE, so the decoded bytes are
//! identical; only the library providing the inflate loop differs. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # `FlateDecode` as an `fz_stream` filter
//!
//! MuPDF's `next_flated` (filter-flate.c:46) drives zlib's `inflate()` with
//! `Z_SYNC_FLUSH`, pulling compressed input from `state->chain` via
//! `fz_available` and pushing decompressed output into a 4096-byte buffer. This
//! port keeps that structure exactly, calling `miniz_oxide::inflate::stream::
//! inflate` with `MZFlush::Sync` in place of zlib's `inflate(Z_SYNC_FLUSH)` and
//! pulling chain bytes through [`Stream::fill_buf`]/[`Stream::consume`].
//!
//! ## Error handling, matched to MuPDF's tolerances
//!
//! `next_flated` is deliberately lenient with broken PDF streams -- it *warns
//! and stops* rather than failing on several zlib errors:
//!
//! | zlib code                         | miniz equivalent      | this port |
//! |-----------------------------------|-----------------------|-----------|
//! | `Z_STREAM_END`                    | `Ok(StreamEnd)`       | stop (done) |
//! | `Z_BUF_ERROR` (premature end)     | `Err(Buf)`            | stop (done) |
//! | `Z_DATA_ERROR` (bad data / adler) | `Err(Data)`           | stop (done) |
//! | `Z_OK`                            | `Ok(Ok)`              | continue |
//! | other                             | `Err(other)`          | `FZ_ERROR_LIBRARY` |
//!
//! MuPDF distinguishes two `Z_DATA_ERROR` sub-cases (trailing-garbage vs.
//! "incorrect data check") but *both* branches just `fz_warn` and `break`; this
//! port collapses them to a single "stop" (the warning stream is not ported --
//! see `error.rs`). A genuinely unexpected miniz status still throws
//! `FZ_ERROR_LIBRARY`, as zlib's fallthrough does.
//!
//! ## Zlib vs. raw DEFLATE
//!
//! MuPDF opens flate with a `window_bits` argument (positive = zlib-wrapped,
//! negative = raw). PDF `FlateDecode` is zlib-wrapped, so [`open_flated`]
//! defaults to that; [`open_flated_raw`] selects raw DEFLATE for the rare
//! header-less streams.

use super::error::{Error, Result};
use super::stream::{Stream, StreamSource};
use miniz_oxide::inflate::stream::{InflateState, inflate};
use miniz_oxide::{DataFormat, MZError, MZFlush, MZStatus};

/// Output burst size, matching `fz_inflate_state.buffer` (filter-flate.c:33).
const OUT_LEN: usize = 4096;

struct Flate<'a> {
    chain: Stream<'a>,
    state: Box<InflateState>,
    out: Vec<u8>,
    done: bool,
}

impl<'a> StreamSource for Flate<'a> {
    // MuPDF: next_flated (filter-flate.c:46)
    fn refill(&mut self, _max: usize) -> Result<&[u8]> {
        self.out.clear();
        if self.done {
            return Ok(&self.out);
        }
        self.out.resize(OUT_LEN, 0);
        let mut written = 0;

        while written < OUT_LEN {
            // zp->avail_in = fz_available(chain, 1); zp->next_in = chain->rp;
            let input = self.chain.fill_buf(1)?;
            let had_input = !input.is_empty();

            // code = inflate(zp, Z_SYNC_FLUSH);
            let result = inflate(
                &mut self.state,
                input,
                &mut self.out[written..],
                MZFlush::Sync,
            );

            // chain->rp = chain->wp - zp->avail_in;  (advance by what was read)
            self.chain.consume(result.bytes_consumed);
            written += result.bytes_written;

            match result.status {
                // Z_STREAM_END: the deflate stream ended cleanly.
                Ok(MZStatus::StreamEnd) => {
                    self.done = true;
                    break;
                }
                // Z_BUF_ERROR: "premature end of data in flate filter" -> stop.
                Err(MZError::Buf) => {
                    self.done = true;
                    break;
                }
                // Z_DATA_ERROR (both MuPDF sub-cases): warn-and-stop.
                Err(MZError::Data) => {
                    self.done = true;
                    break;
                }
                // Z_OK: keep going while there is output room.
                Ok(_) => {
                    // No progress and no more input means we are wedged at EOF;
                    // treat as end (mirrors zlib returning Z_BUF_ERROR here).
                    if !had_input && result.bytes_consumed == 0 && result.bytes_written == 0 {
                        self.done = true;
                        break;
                    }
                }
                // Anything else is a genuine library fault.
                Err(e) => {
                    return Err(Error::library(format!("zlib error: {e:?}")));
                }
            }
        }

        self.out.truncate(written);
        if self.out.is_empty() {
            self.done = true;
        }
        Ok(&self.out)
    }
}

// MuPDF: fz_open_flated (filter-flate.c:122) with zlib window bits.
/// Open a zlib-wrapped `FlateDecode` filter over `chain` (PDF's default).
pub fn open_flated(chain: Stream<'_>) -> Stream<'_> {
    Stream::from_source(Box::new(Flate {
        chain,
        state: InflateState::new_boxed(DataFormat::Zlib),
        out: Vec::with_capacity(OUT_LEN),
        done: false,
    }))
}

// MuPDF: fz_open_flated (filter-flate.c:122) with negative window bits.
/// Open a raw (header-less) DEFLATE filter over `chain`.
pub fn open_flated_raw(chain: Stream<'_>) -> Stream<'_> {
    Stream::from_source(Box::new(Flate {
        chain,
        state: InflateState::new_boxed(DataFormat::Raw),
        out: Vec::with_capacity(OUT_LEN),
        done: false,
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_oxide::deflate::{compress_to_vec, compress_to_vec_zlib};

    fn inflate_zlib(input: &[u8]) -> Result<Vec<u8>> {
        open_flated(Stream::from_slice(input)).read_all()
    }

    #[test]
    fn flate_decodes_zlib_compressed_string() {
        let original = b"The quick brown fox jumps over the lazy dog.";
        let compressed = compress_to_vec_zlib(original, 6);
        assert_eq!(inflate_zlib(&compressed).unwrap(), original);
    }

    #[test]
    fn flate_decodes_hardcoded_zlib_vector() {
        // A fixed zlib stream for "hello" (RFC 1950 wrapper, fixed Huffman):
        //   78 9c cb 48 cd c9 c9 07 00 06 2c 02 15
        let vector = [
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15,
        ];
        assert_eq!(inflate_zlib(&vector).unwrap(), b"hello");
    }

    #[test]
    fn flate_decodes_large_payload_over_many_bursts() {
        // Larger than OUT_LEN so refill runs multiple output bursts.
        let original: Vec<u8> = (0..20_000u32).map(|i| ((i * 31) & 0xff) as u8).collect();
        let compressed = compress_to_vec_zlib(&original, 9);
        assert_eq!(inflate_zlib(&compressed).unwrap(), original);
    }

    #[test]
    fn flate_raw_deflate() {
        let original = b"raw deflate, no zlib header";
        let compressed = compress_to_vec(original, 6); // raw DEFLATE
        let out = open_flated_raw(Stream::from_slice(&compressed))
            .read_all()
            .unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn flate_tolerates_truncated_stream() {
        // Chop the tail off a valid zlib stream: MuPDF warns and returns the
        // bytes it managed to inflate rather than erroring.
        let original = b"partial data that will be cut short in the middle";
        let compressed = compress_to_vec_zlib(original, 6);
        let truncated = &compressed[..compressed.len() - 4];
        // Should not panic/err; returns a (possibly short) prefix.
        let out = inflate_zlib(truncated).unwrap();
        assert!(original.starts_with(&out[..]) || out.is_empty());
    }
}
