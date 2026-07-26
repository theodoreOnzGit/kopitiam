//! Ported from Tesseract `src/ccutil/serialis.cpp` + `src/ccutil/serialis.h`
//! (commit db0ec62, Apache-2.0, © 1990 Hewlett-Packard Ltd., Author: Phil
//! Cheatle; part of the Tesseract project, © Google Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the binary layout, the clamping
//! semantics, and the endianness/swap logic follow Tesseract exactly; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`TFile`] is Tesseract's portable binary reader: a cursor over an in-memory
//! byte buffer that reads little/big-endian scalars, arrays, and raw blocks, with
//! a `swap` flag that byte-reverses each multi-byte item when the file's
//! endianness differs from the host's. The `.traineddata` container
//! ([`crate::tessdata`]) is layered directly on top of it.
//!
//! # Endianness: native read, then optional reverse (faithful, host-dependent)
//!
//! Tesseract reads every scalar with a raw `memcpy` into a host-native integer
//! (`TFile::FRead`) and *then*, if `swap_` is set, reverses the bytes of each item
//! (`TFile::FReadEndian` → `ReverseN`, `helpers.h:198`). So the value obtained
//! before any swap is **host-endianness dependent** — that is the whole point of
//! the `swap_` flag, which higher layers set once they detect a mismatch (see
//! [`TFile::de_serialize_size`] and `TessdataManager::LoadMemBuffer`). This port
//! reproduces that precisely: scalars are decoded with `from_ne_bytes` (the
//! native `memcpy`) and [`TFile::set_swap`] applies the `ReverseN` correction. Do
//! not "fix" this to a fixed endianness — the container's auto-detection relies on
//! the native read.
//!
//! # Read path only
//!
//! Only the **deserialise / read** side of `TFile` is ported: it is all the
//! container parser needs. The write side (`OpenWrite`/`FWrite`/`CloseWrite`/the
//! `Serialize` family and `LoadDataFromFile`/`SaveDataToFile` file I/O) is
//! deliberately **not** ported; KOPITIAM reads `.traineddata`, it does not author
//! it. Port the write path only if a later phase needs to emit containers.
//!
//! Where Tesseract owns a heap `std::vector<char>` copy of the file
//! (`TFile::Open(data, size)` `memcpy`s the input), this port **borrows** the
//! caller's `&[u8]` instead — a read-only cursor needs no copy.

use crate::error::{Error, Result};

/// A portable binary reader over a borrowed byte buffer.
///
/// Mirrors Tesseract's `TFile` in read mode: a `data` slice, a byte `offset`
/// cursor into it, and a `swap` flag controlling per-item byte reversal.
///
/// Tesseract: `class TFile` (serialis.h:61)
#[derive(Debug)]
pub struct TFile<'a> {
    /// The buffered data (Tesseract: `data_`, an owned `std::vector<char>*`; here
    /// a borrow, since the read path never mutates it).
    data: &'a [u8],
    /// The number of bytes consumed so far (Tesseract: `offset_`). Invariant:
    /// `offset <= data.len()`.
    offset: usize,
    /// True when each multi-byte item must be byte-reversed on read (Tesseract:
    /// `swap_`).
    swap: bool,
}

impl<'a> TFile<'a> {
    /// Open a reader over an existing memory buffer.
    ///
    /// Tesseract: `TFile::Open(const char *data, size_t size)` (serialis.cpp:173).
    /// Unlike Tesseract, which `memcpy`s the bytes into an owned vector, this
    /// borrows `data` directly. `swap` starts `false`.
    pub fn new(data: &'a [u8]) -> Self {
        TFile {
            data,
            offset: 0,
            swap: false,
        }
    }

    /// Set the swap flag, so subsequent multi-byte reads byte-reverse each item.
    ///
    /// Tesseract: `TFile::set_swap` (serialis.h:75).
    pub fn set_swap(&mut self, value: bool) {
        self.swap = value;
    }

    /// The current swap flag.
    pub const fn swap(&self) -> bool {
        self.swap
    }

    /// The current read cursor, in bytes from the start of the buffer.
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// The number of bytes remaining to be read.
    ///
    /// Tesseract: `TFile::RemainingBytes` (serialis.h:79).
    pub const fn remaining_bytes(&self) -> usize {
        // The C guards `offset_ < data_->size()`; our invariant keeps
        // `offset <= data.len()`, so a plain subtraction is equivalent.
        self.data.len() - self.offset
    }

    /// Reset the cursor to the start, as if freshly opened.
    ///
    /// Tesseract: `TFile::Rewind` (serialis.cpp:259).
    pub fn rewind(&mut self) {
        self.offset = 0;
    }

    // -----------------------------------------------------------------------
    // Core: fread / fread_endian
    // -----------------------------------------------------------------------

    /// Replicate `fread`: copy up to `count` items of `size` bytes into `out`,
    /// returning the number of whole items actually copied and advancing the
    /// cursor by the copied byte count.
    ///
    /// Tesseract: `TFile::FRead` (serialis.cpp:239). A short buffer is **clamped**
    /// to the bytes remaining rather than treated as an error — the shortfall
    /// surfaces later as a `count` mismatch in the `DeSerialize` wrappers.
    fn fread(&mut self, out: &mut [u8], size: usize, count: usize) -> usize {
        debug_assert!(size > 0, "Tesseract: ASSERT_HOST(size > 0)");
        let remaining = self.data.len() - self.offset;
        // Tesseract avoids `size * count` overflow (`SIZE_MAX / size <= count`)
        // by clamping the request to the bytes remaining; `checked_mul` is the
        // faithful equivalent. In either case the request is capped at
        // `remaining`.
        let required = match size.checked_mul(count) {
            Some(bytes) => bytes.min(remaining),
            None => remaining,
        };
        if required > 0 {
            out[..required].copy_from_slice(&self.data[self.offset..self.offset + required]);
        }
        self.offset += required;
        required / size
    }

    /// Replicate `fread` followed by a byte-swap of each item if `swap` is set.
    ///
    /// Tesseract: `TFile::FReadEndian` (serialis.cpp:228). Items of `size == 1`
    /// are never swapped.
    fn fread_endian(&mut self, out: &mut [u8], size: usize, count: usize) -> usize {
        let num_read = self.fread(out, size, count);
        if self.swap && size != 1 {
            // Tesseract: ReverseN over each of the `num_read` items (helpers.h:198).
            for item in out[..num_read * size].chunks_exact_mut(size) {
                item.reverse();
            }
        }
        num_read
    }

    // -----------------------------------------------------------------------
    // Typed scalar reads: Tesseract's `DeSerialize<T>(T*, count=1)`
    // (serialis.h:89), i.e. `FReadEndian(data, sizeof(T), count) == count`.
    // -----------------------------------------------------------------------

    /// Read one `u8`. (Single bytes are never swapped.)
    pub fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        if self.fread_endian(&mut buf, 1, 1) != 1 {
            return Err(Error::unexpected_eof("read_u8: buffer exhausted"));
        }
        Ok(buf[0])
    }

    /// Read one `i8`. (Single bytes are never swapped.)
    ///
    /// Tesseract: `DeSerialize(int8_t*)` (serialis.h:89). Used by the LSTM
    /// network loader for the layer type tag and the int8 flag/training bytes
    /// (`network.cpp:191` `getNetworkType`).
    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    /// Read one `f64` (IEEE-754 double), honouring the swap flag.
    ///
    /// Tesseract weight matrices and per-row int8 scales are serialized as
    /// doubles (`weightmatrix.cpp:257`, `matrix.h` `GENERIC_2D_ARRAY<double>`),
    /// so the LSTM loader ([`crate::weightmatrix`]) reads them through this.
    pub fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        if self.fread_endian(&mut buf, 8, 1) != 1 {
            return Err(Error::unexpected_eof("read_f64: buffer exhausted"));
        }
        Ok(f64::from_ne_bytes(buf))
    }

    /// Read `count` `i8` values into a fresh `Vec`. (Single-byte items are
    /// never swapped.) The array form used to read an int8 weight matrix's data
    /// block (`matrix.h` `GENERIC_2D_ARRAY<int8_t>::DeSerialize`).
    pub fn read_i8_vec(&mut self, count: usize) -> Result<Vec<i8>> {
        let bytes = self.read_bytes(count)?;
        Ok(bytes.into_iter().map(|b| b as i8).collect())
    }

    /// Read `count` `f64` values into a fresh `Vec`, honouring the swap flag.
    /// The array form used to read a double weight matrix's data block and the
    /// per-row scale vector.
    pub fn read_f64_vec(&mut self, count: usize) -> Result<Vec<f64>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut bytes = vec![0u8; count * 8];
        if self.fread_endian(&mut bytes, 8, count) != count {
            return Err(Error::unexpected_eof("read_f64_vec: buffer exhausted"));
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Read a length-prefixed UTF-8 string: a `uint32` count followed by that
    /// many raw bytes, decoded lossily as UTF-8.
    ///
    /// Tesseract: `TFile::DeSerialize(std::string &)` (serialis.cpp:98),
    /// including the 50,000,000-byte guard. This is how a network layer's name
    /// and the serialized layer-type name are read (`network.cpp:198`,
    /// `network.cpp:246`).
    pub fn de_serialize_string(&mut self) -> Result<String> {
        let bytes = self.de_serialize_char_vec()?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Read one `u16`, honouring the swap flag.
    pub fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        if self.fread_endian(&mut buf, 2, 1) != 1 {
            return Err(Error::unexpected_eof("read_u16: buffer exhausted"));
        }
        Ok(u16::from_ne_bytes(buf))
    }

    /// Read one `u32`, honouring the swap flag.
    pub fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        if self.fread_endian(&mut buf, 4, 1) != 1 {
            return Err(Error::unexpected_eof("read_u32: buffer exhausted"));
        }
        Ok(u32::from_ne_bytes(buf))
    }

    /// Read one `i32`, honouring the swap flag.
    pub fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        if self.fread_endian(&mut buf, 4, 1) != 1 {
            return Err(Error::unexpected_eof("read_i32: buffer exhausted"));
        }
        Ok(i32::from_ne_bytes(buf))
    }

    /// Read one `i64`, honouring the swap flag.
    pub fn read_i64(&mut self) -> Result<i64> {
        let mut buf = [0u8; 8];
        if self.fread_endian(&mut buf, 8, 1) != 1 {
            return Err(Error::unexpected_eof("read_i64: buffer exhausted"));
        }
        Ok(i64::from_ne_bytes(buf))
    }

    /// Read `out.len()` `i64` values into `out`, honouring the swap flag.
    ///
    /// This is the array form `DeSerialize(&offset_table[0], num_entries)` the
    /// container's offset table is read with (`tessdatamanager.cpp:133`). Fails if
    /// fewer than `out.len()` whole values are available.
    pub fn read_i64_slice(&mut self, out: &mut [i64]) -> Result<()> {
        let count = out.len();
        if count == 0 {
            return Ok(());
        }
        let mut bytes = vec![0u8; count * 8];
        if self.fread_endian(&mut bytes, 8, count) != count {
            return Err(Error::unexpected_eof("read_i64_slice: buffer exhausted"));
        }
        for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(8)) {
            // `chunk` is exactly 8 bytes; the conversion cannot fail.
            *dst = i64::from_ne_bytes(chunk.try_into().unwrap());
        }
        Ok(())
    }

    /// Read `count` raw bytes into a fresh `Vec`.
    ///
    /// This is `DeSerialize<char>(data, count)` — the component-payload read of
    /// `TessdataManager::LoadMemBuffer` (`tessdatamanager.cpp:156`). Single-byte
    /// items are never swapped. Fails if fewer than `count` bytes are available.
    pub fn read_bytes(&mut self, count: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; count];
        if count > 0 && self.fread(&mut buf, 1, count) != count {
            return Err(Error::unexpected_eof("read_bytes: buffer exhausted"));
        }
        Ok(buf)
    }

    /// Read a `uint32` size and, if it looks byte-swapped relative to the buffer,
    /// flip the swap flag and correct it.
    ///
    /// Tesseract: `TFile::DeSerializeSize` (serialis.cpp:72). A size larger than
    /// `data.len() / 4` cannot be a valid element count for this buffer, so it is
    /// taken as evidence of an endianness mismatch: the swap flag is toggled and
    /// the value byte-reversed. Returned as `i32` to match the C signature.
    pub fn de_serialize_size(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        if self.fread_endian(&mut buf, 4, 1) != 1 {
            return Err(Error::unexpected_eof("de_serialize_size: buffer exhausted"));
        }
        let mut size = u32::from_ne_bytes(buf);
        if size as usize > self.data.len() / 4 {
            // Reverse endianness (Tesseract: `swap_ = !swap_; ReverseN(&size, 4)`).
            self.swap = !self.swap;
            size = size.swap_bytes();
        }
        Ok(size as i32)
    }

    /// Read a length-prefixed `vector<char>`: a `uint32` count followed by that
    /// many raw bytes.
    ///
    /// Tesseract: `TFile::DeSerialize(std::vector<char> &)` (serialis.cpp:119),
    /// including the arbitrary 50,000,000-byte guard against corrupt lengths.
    pub fn de_serialize_char_vec(&mut self) -> Result<Vec<u8>> {
        let size = self.read_u32()?;
        if size > 50_000_000 {
            // Arbitrarily limit the size to protect against bad data.
            return Err(Error::limit(format!(
                "de_serialize_char_vec: length {size} exceeds 50000000 guard"
            )));
        }
        self.read_bytes(size as usize)
    }

    /// Read one line, `fgets`-style: consume bytes up to and including the next
    /// `\n` (or end-of-buffer), returning at most `buffer_size - 1` bytes.
    ///
    /// Tesseract: `TFile::FGets` (serialis.cpp:213), the parsing primitive the
    /// text unicharset loader ([`crate::unicharset`]) reads through. Returns
    /// [`None`] at end-of-buffer (the C's `nullptr`); a line longer than
    /// `buffer_size - 1` bytes is truncated there, and its tail is returned by
    /// the following call — exactly as the fixed 256-byte C buffer behaves. The
    /// returned bytes include the trailing `\n` when one was reached. Does
    /// nothing and returns [`None`] if `buffer_size <= 1`.
    pub fn fgets(&mut self, buffer_size: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        while out.len() + 1 < buffer_size && self.offset < self.data.len() {
            let b = self.data[self.offset];
            self.offset += 1;
            out.push(b);
            if b == b'\n' {
                break;
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }

    /// Skip `count` bytes.
    ///
    /// Tesseract: `TFile::Skip` (serialis.cpp:145). On failure (not enough bytes
    /// remain) the cursor is parked at end-of-buffer and `false` is returned,
    /// matching the C.
    pub fn skip(&mut self, count: usize) -> bool {
        let data_size = self.data.len();
        // Subtraction-based bound check, avoiding `offset + count` overflow.
        if self.offset >= data_size || count > data_size - self.offset {
            self.offset = data_size;
            return false;
        }
        self.offset += count;
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Endianness is validated relative to the *host*: encode a known value with
    // `to_ne_bytes` (Tesseract's native `memcpy`) and read it back with swap off;
    // then feed the reversed bytes with swap on and expect the same value. This is
    // correct on both little- and big-endian hosts.

    #[test]
    fn read_u32_native_no_swap() {
        let v: u32 = 0x0A0B_0C0D;
        let bytes = v.to_ne_bytes();
        let mut f = TFile::new(&bytes);
        assert_eq!(f.read_u32().unwrap(), v);
        assert_eq!(f.offset(), 4);
        assert_eq!(f.remaining_bytes(), 0);
    }

    #[test]
    fn read_u32_reversed_with_swap() {
        let v: u32 = 0x0A0B_0C0D;
        let mut bytes = v.to_ne_bytes();
        bytes.reverse();
        let mut f = TFile::new(&bytes);
        f.set_swap(true);
        assert_eq!(f.read_u32().unwrap(), v);
    }

    #[test]
    fn read_i64_swap_roundtrip() {
        let v: i64 = -0x0102_0304_0506_0708;
        // no swap
        let bytes = v.to_ne_bytes();
        let mut f = TFile::new(&bytes);
        assert_eq!(f.read_i64().unwrap(), v);
        // swap
        let mut bytes = v.to_ne_bytes();
        bytes.reverse();
        let mut f = TFile::new(&bytes);
        f.set_swap(true);
        assert_eq!(f.read_i64().unwrap(), v);
    }

    #[test]
    fn read_u16_and_u8_and_sequential_offset() {
        let a: u16 = 0x1234;
        let b: u16 = 0xABCD;
        let mut buf = Vec::new();
        buf.extend_from_slice(&a.to_ne_bytes());
        buf.extend_from_slice(&b.to_ne_bytes());
        buf.push(0x5A);
        let mut f = TFile::new(&buf);
        assert_eq!(f.read_u16().unwrap(), a);
        assert_eq!(f.offset(), 2);
        assert_eq!(f.read_u16().unwrap(), b);
        assert_eq!(f.offset(), 4);
        assert_eq!(f.read_u8().unwrap(), 0x5A);
        assert_eq!(f.remaining_bytes(), 0);
    }

    #[test]
    fn single_bytes_are_not_swapped() {
        // With swap on, a u8 read must still return the raw byte.
        let buf = [0x42u8, 0x99];
        let mut f = TFile::new(&buf);
        f.set_swap(true);
        assert_eq!(f.read_u8().unwrap(), 0x42);
        assert_eq!(f.read_u8().unwrap(), 0x99);
    }

    #[test]
    fn read_i64_slice_reads_all() {
        let vals: [i64; 3] = [1, -1, 0x0011_2233_4455_6677];
        let mut buf = Vec::new();
        for v in vals {
            buf.extend_from_slice(&v.to_ne_bytes());
        }
        let mut f = TFile::new(&buf);
        let mut out = [0i64; 3];
        f.read_i64_slice(&mut out).unwrap();
        assert_eq!(out, vals);
    }

    #[test]
    fn short_read_is_unexpected_eof() {
        let buf = [0u8, 0, 0]; // only 3 bytes; a u32 needs 4
        let mut f = TFile::new(&buf);
        let err = f.read_u32().unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::UnexpectedEof);
        // The clamped short read still advanced the cursor to the end.
        assert_eq!(f.offset(), 3);
    }

    #[test]
    fn read_bytes_block() {
        let buf = b"hello world";
        let mut f = TFile::new(buf);
        assert_eq!(f.read_bytes(5).unwrap(), b"hello");
        assert_eq!(f.read_u8().unwrap(), b' ');
        assert_eq!(f.read_bytes(5).unwrap(), b"world");
        assert!(f.read_bytes(1).is_err());
    }

    #[test]
    fn skip_and_bounds() {
        let buf = [1u8, 2, 3, 4, 5];
        let mut f = TFile::new(&buf);
        assert!(f.skip(2));
        assert_eq!(f.read_u8().unwrap(), 3);
        // Skipping past the end fails and parks at end-of-buffer.
        assert!(!f.skip(100));
        assert_eq!(f.remaining_bytes(), 0);
    }

    #[test]
    fn de_serialize_size_detects_swap() {
        // A small count encoded natively in a buffer big enough that `size` stays
        // below `data.len()/4`: no swap needed. (The heuristic reads `size` as a
        // plausible element count, so the buffer must be at least `4*size` long.)
        let n: u32 = 3;
        let mut native = n.to_ne_bytes().to_vec();
        native.extend_from_slice(&[0u8; 16]); // len 20, len/4 == 5 > 3
        let mut f = TFile::new(&native);
        assert_eq!(f.de_serialize_size().unwrap(), 3);
        assert!(!f.swap());

        // The same count in the wrong endianness reads as an absurd value
        // (> data.len()/4), so DeSerializeSize flips swap and corrects it. Give a
        // longer buffer so data.len()/4 is small relative to the swapped value.
        let mut bytes = n.to_ne_bytes().to_vec();
        bytes.reverse();
        bytes.extend_from_slice(&[0u8; 16]); // pad so len/4 == 5 < swapped value
        let mut f = TFile::new(&bytes);
        assert_eq!(f.de_serialize_size().unwrap(), 3);
        assert!(f.swap());
    }

    #[test]
    fn de_serialize_char_vec_length_prefixed() {
        let payload = b"traineddata";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(payload.len() as u32).to_ne_bytes());
        buf.extend_from_slice(payload);
        let mut f = TFile::new(&buf);
        assert_eq!(f.de_serialize_char_vec().unwrap(), payload);
    }

    #[test]
    fn de_serialize_char_vec_rejects_absurd_length() {
        let bogus: u32 = 60_000_000;
        let bytes = bogus.to_ne_bytes();
        let mut f = TFile::new(&bytes);
        assert_eq!(
            f.de_serialize_char_vec().unwrap_err().kind(),
            crate::error::ErrorKind::Limit
        );
    }
}
