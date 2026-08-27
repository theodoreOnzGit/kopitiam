//! Ported from MuPDF `source/fitz/stream-open.c`, `source/fitz/stream-read.c`,
//! and `include/mupdf/fitz/stream.h` (commit 19f1284, AGPL-3.0, © Artifex
//! Software, Inc.), translated to Rust for KOPITIAM (AGPL-3.0-only). Close
//! adaptation: the algorithms and numeric behaviour follow MuPDF; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md ("PDF &
//! document-extraction references").
//!
//! # The buffered-reader model
//!
//! `fz_stream` is a buffered reader that can seek in both directions. Its heart
//! is three fields -- `rp`, `wp`, and a `next` callback: only the bytes between
//! `rp` (read pointer) and `wp` (write pointer) are valid, and when the caller
//! reads past `wp` the `next` callback refills the window from the underlying
//! source (a file, a memory block, or another stream being filtered). Every
//! `fz_read_byte`/`fz_peek_byte`/`fz_read`/`fz_read_bits` is expressed against
//! that window.
//!
//! This port keeps the model but splits it in two, which is what makes it fit
//! Rust's borrow rules:
//!
//! * [`StreamSource`] is the `next`/`seek`/`drop` triple as a trait. Its
//!   `refill` returns the next chunk of bytes (an empty slice means EOF) --
//!   exactly the job of `fz_stream_next_fn`, minus the pointer bookkeeping.
//! * [`Stream`] owns the window (`rp`/`wp`) and the bit-reader state
//!   (`avail`/`bits`), driving a boxed `StreamSource`. It provides
//!   [`Stream::read_byte`], [`Stream::peek_byte`], [`Stream::read`],
//!   [`Stream::tell`]/[`Stream::seek`], [`Stream::read_bits`] &c.
//!
//! ## One deliberate difference from MuPDF: the window is copied in
//!
//! MuPDF avoids copying by aliasing: `next` points `rp`/`wp` straight into the
//! source's own buffer. That relies on the source outliving the borrow, which a
//! `Stream` holding both the source *and* a slice into it cannot express in safe
//! Rust (it would be self-referential). So [`Stream`] copies each refilled chunk
//! into its own `Vec<u8>` window. The behaviour a caller observes is identical;
//! the cost is one memcpy per refill -- which the C filters already pay anyway
//! (they `memcpy` into their own `state->buffer`). Chunks are bounded (4096
//! bytes for the in-memory source, matching `fz_file_stream`'s block size), so
//! this streams rather than slurping.
//!
//! ## Error handling
//!
//! `fz_read_byte` & friends *catch* read errors, warn, and report EOF (see the
//! `fz_catch` blocks in stream.h). This port instead propagates the [`Error`]
//! up through [`Result`] (the port-wide `fz_throw` -> `Result` convention). The
//! in-memory source never errors; filters that hit malformed data return an
//! `Err`, which is strictly more informative than MuPDF's silent-EOF-on-error.
//! `FZ_ERROR_TRYLATER` (progressive loading) is not modelled -- there is no
//! progressive source on the text path.
//!
//! ## Not ported (deliberately)
//!
//! File streams (`fz_open_file*`), the utf-16/utf-8 rune readers
//! (`fz_read_rune`/`fz_read_utf16_*`), the big/little-endian integer readers
//! (`fz_read_uint32` &c.), `fz_read_line`/`fz_skip_string`/`fz_skip_space`, the
//! `fz_read_best` compression-bomb guard, and the leech filter (`fz_open_leecher`)
//! are not needed by the WAVE-2 filters. `read_all` covers `fz_read_all` for
//! tests and whole-stream decoding. Port the rest when a later module needs it.

use super::error::{Error, Result};

/// Block size for refilling the window from an in-memory source. Matches the
/// 4096-byte buffer `fz_file_stream` reads in (`stream-open.c:120`).
const BLOCK: usize = 4096;

/// Where a seek offset is measured from -- MuPDF passes C's `SEEK_SET`/`_CUR`/
/// `_END` (0/1/2) as the `whence` argument to `fz_seek`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Whence {
    /// `SEEK_SET` (0): from the start of the stream.
    Set,
    /// `SEEK_CUR` (1): relative to the current position.
    Cur,
    /// `SEEK_END` (2): relative to the end of the stream.
    End,
}

/// The source behind a [`Stream`]: the `fz_stream_next_fn`/`fz_stream_seek_fn`
/// pair as a trait.
///
/// `refill` is `next`: it produces the next run of bytes, returning an empty
/// slice at end of data. A seekable source additionally implements [`seek`].
///
/// [`seek`]: StreamSource::seek
pub trait StreamSource {
    // MuPDF: fz_stream_next_fn (stream.h:297)
    /// Produce the next chunk of stream data. `max` is a hint for how many bytes
    /// the caller wants ready (safe to ignore, as MuPDF's file source does).
    /// Return an empty slice to signal end of data.
    fn refill(&mut self, max: usize) -> Result<&[u8]>;

    /// Whether this source supports [`seek`](StreamSource::seek).
    fn is_seekable(&self) -> bool {
        false
    }

    // MuPDF: fz_stream_seek_fn (stream.h:317)
    /// Seek to `offset` measured from `whence` (only `Set`/`End` are passed here;
    /// [`Stream::seek`] pre-resolves `Cur` to `Set`). Returns the new absolute
    /// position. The default is the non-seekable case.
    fn seek(&mut self, _offset: i64, _whence: Whence) -> Result<i64> {
        Err(Error::unsupported("stream is not seekable"))
    }
}

/// `fz_stream` -- a buffered reader over a [`StreamSource`].
pub struct Stream<'a> {
    source: Box<dyn StreamSource + 'a>,
    /// The current window: valid bytes are `buf[rp..]` (C's `rp`..`wp`).
    buf: Vec<u8>,
    /// Read position within `buf` (C's `rp`, as an index).
    rp: usize,
    /// End-of-data reached on the underlying source (C's `eof`).
    eof: bool,
    /// Absolute position of the *end* of the current window (C's `pos`).
    pos: i64,
    /// Bits available in `bits` for `read_bits` (C's `avail`).
    avail: i32,
    /// Bit reservoir for `read_bits`/`read_rbits` (C's `bits`, an `int`; `-1`
    /// after an EOF byte, matching `fz_read_byte`'s `EOF`).
    bits: i32,
}

impl<'a> Stream<'a> {
    // MuPDF: fz_new_stream (stream-open.c:55)
    /// Build a stream from a boxed source.
    pub fn from_source(source: Box<dyn StreamSource + 'a>) -> Self {
        Stream {
            source,
            buf: Vec::new(),
            rp: 0,
            eof: false,
            pos: 0,
            avail: 0,
            bits: 0,
        }
    }

    // MuPDF: fz_open_memory (stream-open.c:373) -- borrowed byte block.
    /// Open a borrowed byte block as a seekable stream. Ownership stays with the
    /// caller (this is `fz_open_memory`, and covers `fz_buffer`'s `shared` flag).
    pub fn from_slice(data: &'a [u8]) -> Self {
        Stream::from_source(Box::new(BytesSource { data, pos: 0 }))
    }

    // MuPDF: fz_open_buffer (stream-open.c:353) -- owned byte block.
    /// Open an owned byte vector as a seekable stream.
    pub fn from_vec(data: Vec<u8>) -> Stream<'static> {
        Stream::<'static>::from_source(Box::new(BytesSource { data, pos: 0 }))
    }

    // MuPDF: refill via stm->next (stream.h "fz_stream_next_fn" contract)
    /// Refill the window from the source. Sets `eof` and empties the window at
    /// end of data.
    fn fill(&mut self, max: usize) -> Result<()> {
        let chunk = self.source.refill(max)?;
        if chunk.is_empty() {
            self.eof = true;
            self.buf.clear();
            self.rp = 0;
            return Ok(());
        }
        self.pos += chunk.len() as i64;
        self.buf.clear();
        self.buf.extend_from_slice(chunk);
        self.rp = 0;
        Ok(())
    }

    // MuPDF: fz_available (stream.h:405)
    /// Bytes immediately available in the window, refilling if it is empty.
    /// Returns 0 only at EOF.
    fn available(&mut self, max: usize) -> Result<usize> {
        if self.eof {
            return Ok(0);
        }
        let len = self.buf.len() - self.rp;
        if len != 0 {
            return Ok(len);
        }
        self.fill(max)?;
        Ok(self.buf.len() - self.rp)
    }

    /// Borrow the currently-available window (refilling if empty), like
    /// `std::io::BufRead::fill_buf`. Used by filters that consume the chain's raw
    /// bytes (e.g. flate). Returns an empty slice at EOF.
    pub fn fill_buf(&mut self, max: usize) -> Result<&[u8]> {
        if self.rp >= self.buf.len() && !self.eof {
            self.fill(max)?;
        }
        Ok(&self.buf[self.rp..])
    }

    /// Advance the read pointer by `n` bytes previously observed via
    /// [`fill_buf`](Stream::fill_buf) (C's `chain->rp += n`). `n` must not exceed
    /// the available window.
    pub fn consume(&mut self, n: usize) {
        self.rp += n;
        debug_assert!(self.rp <= self.buf.len());
    }

    // MuPDF: fz_read_byte (stream.h:443)
    /// Read the next byte, or `None` at end of stream.
    pub fn read_byte(&mut self) -> Result<Option<u8>> {
        if self.rp < self.buf.len() {
            let b = self.buf[self.rp];
            self.rp += 1;
            return Ok(Some(b));
        }
        if self.eof {
            return Ok(None);
        }
        self.fill(1)?;
        if self.rp < self.buf.len() {
            let b = self.buf[self.rp];
            self.rp += 1;
            Ok(Some(b))
        } else {
            Ok(None)
        }
    }

    /// `fz_read_byte` with MuPDF's `int` convention: the byte value, or `-1`
    /// (`EOF`) at end of stream. Convenience for the bit-reader and filters that
    /// branch on `c < 0`.
    fn read_byte_c(&mut self) -> Result<i32> {
        Ok(match self.read_byte()? {
            Some(b) => b as i32,
            None => -1,
        })
    }

    // MuPDF: fz_peek_byte (stream.h:473)
    /// Peek at the next byte without consuming it, or `None` at end of stream.
    pub fn peek_byte(&mut self) -> Result<Option<u8>> {
        if self.rp < self.buf.len() {
            return Ok(Some(self.buf[self.rp]));
        }
        if self.eof {
            return Ok(None);
        }
        self.fill(1)?;
        Ok(self.buf.get(self.rp).copied())
    }

    // MuPDF: fz_read (stream-read.c:29)
    /// Read into `out`, returning the number of bytes read (short only at EOF).
    pub fn read(&mut self, out: &mut [u8]) -> Result<usize> {
        let mut count = 0;
        while count < out.len() {
            let want = out.len() - count;
            let n = self.available(want)?;
            if n == 0 {
                break;
            }
            let n = n.min(want);
            out[count..count + n].copy_from_slice(&self.buf[self.rp..self.rp + n]);
            self.rp += n;
            count += n;
        }
        Ok(count)
    }

    /// Read the entire remaining stream into a `Vec<u8>` (`fz_read_all`).
    pub fn read_all(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut tmp = [0u8; BLOCK];
        loop {
            let n = self.read(&mut tmp)?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&tmp[..n]);
        }
        Ok(out)
    }

    // MuPDF: fz_tell (stream-read.c:167)
    /// The current absolute read position.
    pub fn tell(&self) -> i64 {
        self.pos - (self.buf.len() - self.rp) as i64
    }

    // MuPDF: fz_seek (stream-read.c:173)
    /// Seek within the stream. Seekable sources jump directly; non-seekable ones
    /// support forward seeks only (by reading and discarding).
    pub fn seek(&mut self, offset: i64, whence: Whence) -> Result<()> {
        self.avail = 0; // reset bit reading, as fz_seek does
        if self.source.is_seekable() {
            // fz_seek pre-converts SEEK_CUR to an absolute SEEK_SET.
            let (off, wh) = match whence {
                Whence::Cur => (offset + self.tell(), Whence::Set),
                w => (offset, w),
            };
            let newpos = self.source.seek(off, wh)?;
            self.pos = newpos;
            self.buf.clear();
            self.rp = 0;
            self.eof = false;
            Ok(())
        } else {
            let target = match whence {
                Whence::Set => offset,
                Whence::Cur => offset + self.tell(),
                Whence::End => {
                    return Err(Error::unsupported(
                        "cannot seek relative to end on a non-seekable stream",
                    ));
                }
            };
            let delta = target - self.tell();
            if delta < 0 {
                return Err(Error::unsupported(
                    "cannot seek backwards on a non-seekable stream",
                ));
            }
            let mut delta = delta;
            while delta > 0 {
                if self.read_byte()?.is_none() {
                    break;
                }
                delta -= 1;
            }
            Ok(())
        }
    }

    // MuPDF: fz_is_eof (stream.h:520)
    /// Whether the stream has reached end of data (bytewise).
    pub fn is_eof(&mut self) -> Result<bool> {
        if self.rp < self.buf.len() {
            return Ok(false);
        }
        if self.eof {
            return Ok(true);
        }
        Ok(self.peek_byte()?.is_none())
    }

    // MuPDF: fz_read_bits (stream.h:542) -- MSB-first.
    /// Read the next `n` bits, most-significant-bit first.
    pub fn read_bits(&mut self, n: i32) -> Result<u32> {
        let mut n = n;
        let x: i32;
        if n <= self.avail {
            self.avail -= n;
            x = (self.bits >> self.avail) & ((1 << n) - 1);
        } else {
            let mut xx = self.bits & ((1 << self.avail) - 1);
            n -= self.avail;
            self.avail = 0;
            while n > 8 {
                xx = (xx << 8) | self.read_byte_c()?;
                n -= 8;
            }
            if n > 0 {
                self.bits = self.read_byte_c()?;
                self.avail = 8 - n;
                xx = (xx << n) | (self.bits >> self.avail);
            }
            x = xx;
        }
        Ok(x as u32)
    }

    // MuPDF: fz_read_rbits (stream.h:585) -- LSB-first.
    /// Read the next `n` bits, least-significant-bit first.
    pub fn read_rbits(&mut self, n: i32) -> Result<u32> {
        let mut n = n;
        let x: i32;
        if n <= self.avail {
            x = self.bits & ((1 << n) - 1);
            self.avail -= n;
            self.bits >>= n;
        } else {
            let mut xx = self.bits & ((1 << self.avail) - 1);
            n -= self.avail;
            let mut used = self.avail;
            self.avail = 0;
            while n > 8 {
                xx |= self.read_byte_c()? << used;
                n -= 8;
                used += 8;
            }
            if n > 0 {
                self.bits = self.read_byte_c()?;
                xx |= (self.bits & ((1 << n) - 1)) << used;
                self.avail = 8 - n;
                self.bits >>= n;
            }
            x = xx;
        }
        Ok(x as u32)
    }

    // MuPDF: fz_sync_bits (stream.h:628)
    /// Resync to a byte boundary before returning to bytewise reading.
    pub fn sync_bits(&mut self) {
        self.avail = 0;
    }

    // MuPDF: fz_is_eof_bits (stream.h:640)
    /// Whether the stream has reached end of data (bitwise).
    pub fn is_eof_bits(&mut self) -> Result<bool> {
        Ok(self.is_eof()? && (self.avail == 0 || self.bits == -1))
    }
}

// ---------------------------------------------------------------------------
// In-memory source (fz_open_memory / fz_open_buffer)
// ---------------------------------------------------------------------------

/// A source over an in-memory byte block (borrowed `&[u8]` or owned `Vec<u8>`),
/// backing `fz_open_memory` / `fz_open_buffer`. Seekable in both directions.
struct BytesSource<T> {
    data: T,
    pos: usize,
}

impl<T: AsRef<[u8]>> StreamSource for BytesSource<T> {
    fn refill(&mut self, _max: usize) -> Result<&[u8]> {
        let data = self.data.as_ref();
        if self.pos >= data.len() {
            return Ok(&[]);
        }
        // MuPDF's memory stream hands back the whole block at once; we bound the
        // window to BLOCK so the copy into the Stream window streams rather than
        // slurps. Behaviour to the reader is identical.
        let end = (self.pos + BLOCK).min(data.len());
        let start = self.pos;
        self.pos = end;
        Ok(&data[start..end])
    }

    fn is_seekable(&self) -> bool {
        true
    }

    // MuPDF: seek_buffer (stream-open.c:327) -- clamp into [0, len].
    fn seek(&mut self, offset: i64, whence: Whence) -> Result<i64> {
        let len = self.data.as_ref().len() as i64;
        let mut off = match whence {
            Whence::Set => offset,
            Whence::End => len + offset,
            // Stream::seek resolves Cur to Set before calling us; handle it
            // defensively as relative to the source cursor.
            Whence::Cur => self.pos as i64 + offset,
        };
        if off < 0 {
            off = 0;
        }
        if off > len {
            off = len;
        }
        self.pos = off as usize;
        Ok(off)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_byte_then_eof() {
        let mut s = Stream::from_slice(b"AB");
        assert_eq!(s.read_byte().unwrap(), Some(b'A'));
        assert_eq!(s.read_byte().unwrap(), Some(b'B'));
        assert_eq!(s.read_byte().unwrap(), None);
        // Idempotent at EOF.
        assert_eq!(s.read_byte().unwrap(), None);
    }

    #[test]
    fn peek_does_not_consume() {
        let mut s = Stream::from_slice(b"xyz");
        assert_eq!(s.peek_byte().unwrap(), Some(b'x'));
        assert_eq!(s.peek_byte().unwrap(), Some(b'x'));
        assert_eq!(s.read_byte().unwrap(), Some(b'x'));
        assert_eq!(s.peek_byte().unwrap(), Some(b'y'));
    }

    #[test]
    fn read_fills_slice_and_reports_count() {
        // Data larger than one BLOCK to exercise multiple refills.
        let data: Vec<u8> = (0..10_000u32).map(|i| (i & 0xff) as u8).collect();
        let mut s = Stream::from_vec(data.clone());
        let mut out = vec![0u8; 10_000];
        assert_eq!(s.read(&mut out).unwrap(), 10_000);
        assert_eq!(out, data);
        // Nothing left.
        let mut tail = [0u8; 4];
        assert_eq!(s.read(&mut tail).unwrap(), 0);
    }

    #[test]
    fn read_all_reads_everything() {
        let data: Vec<u8> = (0..5000u32).map(|i| ((i * 7) & 0xff) as u8).collect();
        let mut s = Stream::from_vec(data.clone());
        assert_eq!(s.read_all().unwrap(), data);
    }

    #[test]
    fn tell_and_seek_set_cur_end() {
        let mut s = Stream::from_slice(b"0123456789");
        assert_eq!(s.tell(), 0);
        s.read_byte().unwrap();
        s.read_byte().unwrap();
        assert_eq!(s.tell(), 2);

        s.seek(5, Whence::Set).unwrap();
        assert_eq!(s.tell(), 5);
        assert_eq!(s.read_byte().unwrap(), Some(b'5'));

        s.seek(2, Whence::Cur).unwrap(); // now at 6, +2 = 8
        assert_eq!(s.tell(), 8);
        assert_eq!(s.read_byte().unwrap(), Some(b'8'));

        s.seek(-3, Whence::End).unwrap(); // 10 - 3 = 7
        assert_eq!(s.tell(), 7);
        assert_eq!(s.read_byte().unwrap(), Some(b'7'));

        s.seek(0, Whence::Set).unwrap();
        assert_eq!(s.read_byte().unwrap(), Some(b'0'));
    }

    #[test]
    fn is_eof_transitions() {
        let mut s = Stream::from_slice(b"z");
        assert!(!s.is_eof().unwrap());
        assert_eq!(s.read_byte().unwrap(), Some(b'z'));
        assert!(s.is_eof().unwrap());
    }

    #[test]
    fn fill_buf_and_consume() {
        let mut s = Stream::from_slice(b"hello");
        let win = s.fill_buf(1).unwrap().to_vec();
        assert_eq!(win, b"hello");
        s.consume(2);
        assert_eq!(s.read_byte().unwrap(), Some(b'l'));
    }

    #[test]
    fn read_bits_msb_first() {
        // 0b1011_0010, 0b1100_0001
        let mut s = Stream::from_slice(&[0b1011_0010, 0b1100_0001]);
        assert_eq!(s.read_bits(1).unwrap(), 0b1);
        assert_eq!(s.read_bits(3).unwrap(), 0b011);
        assert_eq!(s.read_bits(4).unwrap(), 0b0010);
        assert_eq!(s.read_bits(8).unwrap(), 0b1100_0001);
    }

    #[test]
    fn read_bits_spanning_bytes() {
        // 0xF0 0x0F -> read 12 bits MSB first = 0xF00.
        let mut s = Stream::from_slice(&[0xF0, 0x0F]);
        assert_eq!(s.read_bits(12).unwrap(), 0xF00);
        assert_eq!(s.read_bits(4).unwrap(), 0x0F);
    }

    #[test]
    fn empty_stream_is_immediately_eof() {
        let mut s = Stream::from_slice(b"");
        assert!(s.is_eof().unwrap());
        assert_eq!(s.read_byte().unwrap(), None);
        assert_eq!(s.read_all().unwrap(), Vec::<u8>::new());
    }
}
