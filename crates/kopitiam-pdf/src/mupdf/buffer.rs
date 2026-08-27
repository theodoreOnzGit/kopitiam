//! Ported from MuPDF `source/fitz/buffer.c` + `include/mupdf/fitz/buffer.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # `fz_buffer` -> [`Buffer`]
//!
//! `fz_buffer` is a growable byte array with a `cap`/`len` split and a
//! reference count. This port keeps the growth *semantics* (the exact 1.5x
//! grow rule, the 256-byte first bump) while sitting on a `Vec<u8>`, which is
//! Rust's growable byte array. The two pieces of `fz_buffer` machinery that do
//! **not** translate 1:1 are recorded here so the next porter is not surprised:
//!
//! * **Reference counting (`fz_keep_buffer`/`fz_drop_buffer`, the `refs`
//!   field).** MuPDF shares one heap `fz_buffer` between many owners via a
//!   refcount; Rust expresses shared ownership with the type system. So there is
//!   no `refs` field: a [`Buffer`] is owned by its holder, `Clone` makes an
//!   independent copy (`fz_clone_buffer`), and if a later module needs *shared*
//!   ownership it wraps the buffer in `Rc`/`Arc` at that call site. Dropping is
//!   `Drop`.
//! * **The `shared` flag (`fz_new_buffer_from_shared_data`).** In C this marks a
//!   buffer whose storage is borrowed (not freed on drop, cannot be resized).
//!   The read/text path never resizes such a buffer -- it only reads it -- and
//!   in this port "read a borrowed byte block as a stream" is expressed directly
//!   by opening a stream over a `&[u8]` (see `stream::Stream::from_slice`), so
//!   the `shared` flag is **out of scope** for [`Buffer`]: every [`Buffer`] owns
//!   its storage. `fz_resize_buffer`'s "cannot resize shared storage" throw
//!   therefore cannot arise here.
//!
//! ## Not ported from this file (deliberately)
//!
//! `fz_append_bits`/`fz_append_bits_pad` (bit-packed *writing*, only used by the
//! writer/output side), the `printf`-family (`fz_append_printf`,
//! `fz_new_buffer_from_printf`), `fz_append_pdf_string`, base64
//! (`fz_new_buffer_from_base64`/`fz_append_base64`), `fz_md5_buffer`,
//! `fz_slice_buffer`, and `fz_terminate_buffer`/`fz_string_from_buffer` (the C
//! zero-terminator trick, unneeded when a `Vec<u8>` already carries its length)
//! are not on the text-extraction read path. Port them only if a later module
//! needs them.

use super::string_util::runetochar;

/// `fz_buffer` -- a growable byte buffer. Owns its storage (a `Vec<u8>`).
///
/// The `cap`/`len` split of the C struct maps onto `Vec`'s capacity/length; the
/// `unused_bits` field (only meaningful for `fz_append_bits`, not ported) and
/// `refs`/`shared` (see the module docs) are intentionally absent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Buffer {
    data: Vec<u8>,
}

impl Buffer {
    // MuPDF: fz_new_buffer (buffer.c:28)
    /// Create a new, empty buffer with room for at least `capacity` bytes.
    ///
    /// MuPDF bumps a requested size of 0 or 1 up to 16 (`size = size > 1 ? size
    /// : 16`); we preserve that minimum-reservation behaviour.
    pub fn new(capacity: usize) -> Self {
        let capacity = if capacity > 1 { capacity } else { 16 };
        Buffer {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Create an empty buffer with no reserved capacity.
    pub fn empty() -> Self {
        Buffer { data: Vec::new() }
    }

    // MuPDF: fz_new_buffer_from_data (buffer.c:53) -- takes ownership of `data`.
    /// Wrap an existing byte vector, taking ownership. `len == cap == data.len()`.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Buffer { data }
    }

    // MuPDF: fz_new_buffer_from_copied_data (buffer.c:92)
    /// Create a buffer containing a copy of `data`.
    pub fn from_copied_data(data: &[u8]) -> Self {
        Buffer {
            data: data.to_vec(),
        }
    }

    // MuPDF: fz_buffer_storage / buf->len (buffer.c:298)
    /// The number of bytes currently in the buffer (`fz_buffer_storage`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The buffer's current capacity (`buf->cap`).
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// The buffer contents as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// The buffer contents as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    // MuPDF: fz_clear_buffer (buffer.c:283) -- storage kept, len reset to 0.
    /// Empty the buffer. Capacity (storage) is retained for reuse.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    // MuPDF: fz_grow_buffer (buffer.c:240)
    /// Ensure there is room to append at least one more byte, growing capacity
    /// by MuPDF's exact rule: 256 bytes for an unallocated buffer, otherwise
    /// `cap + cap/2` (1.5x, rounding down).
    ///
    /// `Vec`'s own `push`/`extend` already grow on demand; this method exists
    /// for API parity and to expose MuPDF's growth *shape* for callers that
    /// pre-grow.
    pub fn grow(&mut self) {
        let cap = self.data.capacity();
        // check for integer overflow, mirroring buffer.c
        let newsize = if cap == 0 {
            256
        } else {
            let ns = cap + (cap >> 1);
            if ns < cap { usize::MAX } else { ns }
        };
        let len = self.data.len();
        if newsize > len {
            self.data.reserve_exact(newsize - len);
        }
    }

    // MuPDF: fz_append_byte (buffer.c:386)
    /// Append a single byte, growing as required.
    pub fn append_byte(&mut self, val: u8) {
        self.data.push(val);
    }

    // MuPDF: fz_append_data (buffer.c:365)
    /// Append a block of bytes, growing as required.
    pub fn append_data(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }

    // MuPDF: fz_append_string (buffer.c:375) -- appends the bytes, no NUL.
    /// Append the bytes of a string (no terminator is written).
    pub fn append_string(&mut self, data: &str) {
        self.data.extend_from_slice(data.as_bytes());
    }

    // MuPDF: fz_append_buffer (buffer.c:350)
    /// Append the contents of another buffer onto the end of this one.
    pub fn append_buffer(&mut self, extra: &Buffer) {
        self.data.extend_from_slice(&extra.data);
    }

    // MuPDF: fz_append_rune (buffer.c:395) -- UTF-8 encode `c` and append it.
    /// Append a Unicode code point as UTF-8 (via `fz_runetochar`).
    pub fn append_rune(&mut self, c: u32) {
        let mut tmp = [0u8; 10];
        let n = runetochar(&mut tmp, c);
        self.data.extend_from_slice(&tmp[..n]);
    }

    // MuPDF: fz_append_uint16_be (buffer.c:417)
    /// Append a big-endian `u16`.
    pub fn append_uint16_be(&mut self, x: u16) {
        self.data.extend_from_slice(&x.to_be_bytes());
    }

    // MuPDF: fz_append_uint32_be (buffer.c:408)
    /// Append a big-endian `u32`.
    pub fn append_uint32_be(&mut self, x: u32) {
        self.data.extend_from_slice(&x.to_be_bytes());
    }

    // MuPDF: fz_append_uint16_le (buffer.c:433)
    /// Append a little-endian `u16`.
    pub fn append_uint16_le(&mut self, x: u16) {
        self.data.extend_from_slice(&x.to_le_bytes());
    }

    // MuPDF: fz_append_uint32_le (buffer.c:424)
    /// Append a little-endian `u32`.
    pub fn append_uint32_le(&mut self, x: u32) {
        self.data.extend_from_slice(&x.to_le_bytes());
    }

    // MuPDF: fz_buffer_extract (buffer.c:315) -- take ownership, leaving empty.
    /// Consume the buffer, returning its backing byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

impl core::ops::Deref for Buffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data
    }
}

impl AsRef<[u8]> for Buffer {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl From<Vec<u8>> for Buffer {
    fn from(data: Vec<u8>) -> Self {
        Buffer::from_vec(data)
    }
}

impl From<&[u8]> for Buffer {
    fn from(data: &[u8]) -> Self {
        Buffer::from_copied_data(data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_reserves_at_least_the_minimum() {
        // fz_new_buffer bumps 0/1 up to 16.
        assert!(Buffer::new(0).capacity() >= 16);
        assert!(Buffer::new(1).capacity() >= 16);
        assert!(Buffer::new(100).capacity() >= 100);
        assert_eq!(Buffer::new(0).len(), 0);
        assert!(Buffer::new(0).is_empty());
    }

    #[test]
    fn append_byte_and_data_track_len() {
        let mut b = Buffer::empty();
        b.append_byte(0x41);
        b.append_byte(0x42);
        b.append_data(b"CDE");
        assert_eq!(b.as_slice(), b"ABCDE");
        assert_eq!(b.len(), 5);
        b.append_string("FG");
        assert_eq!(b.as_slice(), b"ABCDEFG");
    }

    #[test]
    fn grow_follows_mupdf_rule() {
        // Unallocated -> 256.
        let mut b = Buffer::empty();
        assert_eq!(b.capacity(), 0);
        b.grow();
        assert!(b.capacity() >= 256);

        // cap -> cap + cap/2 (>= 1.5x), truncating data invariants preserved.
        let mut c = Buffer::new(100);
        let before = c.capacity();
        c.grow();
        assert!(c.capacity() >= before + (before >> 1));
    }

    #[test]
    fn append_many_grows_and_preserves_content() {
        let mut b = Buffer::new(4);
        let payload: Vec<u8> = (0..300).map(|i| (i & 0xff) as u8).collect();
        b.append_data(&payload);
        assert_eq!(b.len(), 300);
        assert_eq!(b.as_slice(), payload.as_slice());
    }

    #[test]
    fn append_buffer_concatenates() {
        let mut a = Buffer::from_copied_data(b"foo");
        let extra = Buffer::from_copied_data(b"bar");
        a.append_buffer(&extra);
        assert_eq!(a.as_slice(), b"foobar");
        // Source is untouched (ownership unchanged, as in fz_append_buffer).
        assert_eq!(extra.as_slice(), b"bar");
    }

    #[test]
    fn append_rune_encodes_utf8() {
        let mut b = Buffer::empty();
        b.append_rune('A' as u32); // 1 byte
        b.append_rune(0x00E9); // é, 2 bytes
        b.append_rune(0x20AC); // €, 3 bytes
        b.append_rune(0x1F600); // 😀, 4 bytes
        assert_eq!(b.as_slice(), "Aé€😀".as_bytes());
    }

    #[test]
    fn endian_appends_match_byte_order() {
        let mut b = Buffer::empty();
        b.append_uint16_be(0x0102);
        b.append_uint16_le(0x0304);
        b.append_uint32_be(0x0A0B0C0D);
        b.append_uint32_le(0x0A0B0C0D);
        assert_eq!(
            b.as_slice(),
            &[
                0x01, 0x02, 0x04, 0x03, 0x0A, 0x0B, 0x0C, 0x0D, 0x0D, 0x0C, 0x0B, 0x0A
            ]
        );
    }

    #[test]
    fn clear_keeps_capacity() {
        let mut b = Buffer::new(64);
        b.append_data(b"hello");
        let cap = b.capacity();
        b.clear();
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
        assert_eq!(b.capacity(), cap);
    }

    #[test]
    fn into_bytes_and_from_vec_round_trip() {
        let v = vec![1u8, 2, 3, 4];
        let b = Buffer::from_vec(v.clone());
        assert_eq!(b.len(), 4);
        assert_eq!(b.into_bytes(), v);
    }

    #[test]
    fn clone_is_independent() {
        // fz_clone_buffer makes a copy, not a shared reference.
        let mut a = Buffer::from_copied_data(b"abc");
        let b = a.clone();
        a.append_byte(b'd');
        assert_eq!(a.as_slice(), b"abcd");
        assert_eq!(b.as_slice(), b"abc");
    }
}
