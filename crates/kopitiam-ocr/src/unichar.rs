//! Ported from Tesseract `src/ccutil/unichar.cpp` +
//! `include/tesseract/unichar.h` (commit db0ec62, Apache-2.0, © 2006 Google
//! Inc., Author: Ray Smith), translated to Rust for KOPITIAM (AGPL-3.0-only).
//! Close adaptation: the UTF-8 stepping table, the UCS-4 encode/decode
//! arithmetic, and the iteration helpers follow Tesseract exactly; the code is
//! re-expressed in idiomatic Rust. See docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! [`Unichar`] is Tesseract's single-character (or ligature) UTF-8 holder, and
//! the free functions here are its static helpers: [`utf8_step`] (the byte
//! length of the first UTF-8 character), [`utf8_to_utf32`] / [`utf32_to_utf8`],
//! and the [`Utf8Iter`] codepoint walk. These are the primitives the
//! [`crate::unicharset`] loader and the CJK recoder ([`crate::unicharcompress`])
//! build on, so KOPITIAM ports Tesseract's own UTF-8 handling rather than
//! depending on any other crate.
//!
//! # Faithfulness note
//!
//! Tesseract decodes UTF-8 with a 256-entry lookup table that classifies only
//! the *first* byte, then trusts the continuation bytes (it does no overlong or
//! surrogate rejection beyond the `0x80` continuation-bit check in the
//! multi-codepoint constructor). This port reproduces that table and that
//! leniency verbatim so ids parsed from a real unicharset line up exactly; it
//! is deliberately *not* Rust's stricter `str::chars`.

use crate::error::{Error, Result};

/// Maximum number of UTF-8 bytes a [`Unichar`] can hold.
///
/// Tesseract: `#define UNICHAR_LEN 30` (unichar.h:31). Must be at least 4 and
/// not exceed 31 (the length is coded in the last byte of the C array).
pub const UNICHAR_LEN: usize = 30;

/// A `UNICHAR_ID` is the unique id of a unichar within a `UNICHARSET`.
///
/// Tesseract: `using UNICHAR_ID = int` (unichar.h:34).
pub type UnicharId = i32;

/// A UCS-4 codepoint. Tesseract: `using char32 = signed int` (unichar.h:49).
pub type Char32 = i32;

/// An invalid or uninitialised unichar id.
///
/// Tesseract: `INVALID_UNICHAR_ID = -1` (unichar.h:37).
pub const INVALID_UNICHAR_ID: UnicharId = -1;

/// The special unichar string that corresponds to [`INVALID_UNICHAR_ID`].
///
/// Tesseract: `INVALID_UNICHAR[] = "__INVALID_UNICHAR__"` (unichar.h:39).
pub const INVALID_UNICHAR: &str = "__INVALID_UNICHAR__";

/// The largest legal UTF-32 codepoint. Tesseract: `UNI_MAX_LEGAL_UTF32`
/// (unichar.cpp:23).
const UNI_MAX_LEGAL_UTF32: i32 = 0x0010_FFFF;

/// Per-first-byte UTF-8 length table.
///
/// Tesseract: the `utf8_bytes[256]` table in `UNICHAR::utf8_step`
/// (unichar.cpp:144), reproduced verbatim (9 rows of 29/29/29/29/29/29/29/29/24
/// entries). Entry `b` is the number of bytes in a UTF-8 character starting with
/// byte `b`, or `0` if `b` cannot begin a character: 1 for `0x00..=0x7F`, 0 for
/// `0x80..=0xBF`, 2 for `0xC0..=0xDF`, 3 for `0xE0..=0xEF`, 4 for `0xF0..=0xF7`,
/// 0 for `0xF8..=0xFF`. Note that Tesseract leniently assigns length 2 to the
/// technically-invalid overlong lead bytes `0xC0`/`0xC1`; this is preserved so
/// ids parsed from a real unicharset match byte-for-byte.
#[rustfmt::skip]
const UTF8_BYTES: [u8; 256] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// UCS-4 reconstruction offsets, indexed by UTF-8 length.
///
/// Tesseract: `utf8_offsets[5]` in `UNICHAR::first_uni` (unichar.cpp:106).
const UTF8_OFFSETS: [i32; 5] = [0, 0, 0x3080, 0xE2080, 0x3C82080];

/// The number of bytes in the first character of the given UTF-8 buffer.
///
/// Tesseract: `UNICHAR::utf8_step` (unichar.cpp:143). Returns `0` if the first
/// byte cannot begin a UTF-8 character. An empty buffer is treated as a `NUL`
/// first byte (length 1), matching the C, which reads `*utf8_str` unconditionally
/// — callers that iterate always guard on a non-empty slice first.
pub fn utf8_step(utf8_str: &[u8]) -> usize {
    let first = utf8_str.first().copied().unwrap_or(0);
    UTF8_BYTES[first as usize] as usize
}

/// The UNICHAR class holds a single classification result: one Unicode character
/// (1–4 UTF-8 bytes) or the NFKC expansion of a ligature (also UTF-8).
///
/// Tesseract: `class UNICHAR` (unichar.h:55). This port stores the UTF-8 bytes
/// in a `Vec` rather than the C's fixed `char[UNICHAR_LEN]` with a length coded
/// in the final byte; the observable behaviour (`utf8`, `utf8_len`, `first_uni`)
/// is identical.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Unichar {
    chars: Vec<u8>,
}

impl Unichar {
    /// Construct from a UTF-8 byte slice, taking as many complete, legal UTF-8
    /// characters as fit in [`UNICHAR_LEN`] bytes.
    ///
    /// Tesseract: `UNICHAR::UNICHAR(const char *utf8_str, int len)`
    /// (unichar.cpp:31). Stops at the first illegal first byte, a continuation
    /// byte whose top bits are not `10`, or the point where the next character
    /// would overflow `UNICHAR_LEN`. The result may be empty.
    pub fn from_utf8(utf8_str: &[u8]) -> Self {
        let len = utf8_str.len();
        let mut total_len = 0usize;
        while total_len < len {
            let step = utf8_step(&utf8_str[total_len..]);
            if total_len + step > UNICHAR_LEN {
                break; // Too long.
            }
            if step == 0 {
                break; // Illegal first byte.
            }
            // Verify the continuation bytes are 10xxxxxx.
            let mut i = 1;
            while i < step {
                if total_len + i >= len || (utf8_str[total_len + i] & 0xc0) != 0x80 {
                    break;
                }
                i += 1;
            }
            if i < step {
                break; // Illegal / truncated sequence.
            }
            total_len += step;
        }
        Unichar {
            chars: utf8_str[..total_len].to_vec(),
        }
    }

    /// Construct from a single UCS-4 codepoint. Illegal values yield an empty
    /// [`Unichar`].
    ///
    /// Tesseract: `UNICHAR::UNICHAR(int unicode)` (unichar.cpp:68).
    pub fn from_unicode(mut unicode: i32) -> Self {
        const BYTEMASK: i32 = 0xBF;
        const BYTEMARK: i32 = 0x80;
        let mut chars = Vec::with_capacity(4);
        if unicode < 0x80 {
            chars.push(unicode as u8);
        } else if unicode < 0x800 {
            let b1 = ((unicode | BYTEMARK) & BYTEMASK) as u8;
            unicode >>= 6;
            chars.push((unicode | 0xc0) as u8);
            chars.push(b1);
        } else if unicode < 0x10000 {
            let b2 = ((unicode | BYTEMARK) & BYTEMASK) as u8;
            unicode >>= 6;
            let b1 = ((unicode | BYTEMARK) & BYTEMASK) as u8;
            unicode >>= 6;
            chars.push((unicode | 0xe0) as u8);
            chars.push(b1);
            chars.push(b2);
        } else if unicode <= UNI_MAX_LEGAL_UTF32 {
            let b3 = ((unicode | BYTEMARK) & BYTEMASK) as u8;
            unicode >>= 6;
            let b2 = ((unicode | BYTEMARK) & BYTEMASK) as u8;
            unicode >>= 6;
            let b1 = ((unicode | BYTEMARK) & BYTEMASK) as u8;
            unicode >>= 6;
            chars.push((unicode | 0xf0) as u8);
            chars.push(b1);
            chars.push(b2);
            chars.push(b3);
        }
        // else: illegal -> empty.
        Unichar { chars }
    }

    /// The first character decoded as UCS-4.
    ///
    /// Tesseract: `UNICHAR::first_uni` (unichar.cpp:105).
    pub fn first_uni(&self) -> Char32 {
        let len = utf8_step(&self.chars);
        let mut uni: i32 = 0;
        // Tesseract accumulates `len` bytes, shifting left 6 between each, then
        // subtracts the length-indexed offset.
        for i in 0..len {
            if i > 0 {
                uni <<= 6;
            }
            uni = uni.wrapping_add(self.chars[i] as i32);
        }
        uni - UTF8_OFFSETS.get(len).copied().unwrap_or(0)
    }

    /// The UTF-8 bytes (NOT NUL-terminated).
    ///
    /// Tesseract: `UNICHAR::utf8` (unichar.h:81).
    pub fn utf8(&self) -> &[u8] {
        &self.chars
    }

    /// The length of the UTF-8 representation, in bytes.
    ///
    /// Tesseract: `UNICHAR::utf8_len` (unichar.h:75).
    pub fn utf8_len(&self) -> usize {
        self.chars.len()
    }

    /// Whether the holder is empty (no legal characters were stored).
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }
}

/// An iterator over the codepoints of a UTF-8 buffer.
///
/// Tesseract: `UNICHAR::const_iterator` (unichar.h:105) plus [`Unichar::begin`]/
/// [`Unichar::end`]. Unlike the C iterator it borrows the buffer directly.
/// Illegal bytes are handled exactly as Tesseract does: [`Iterator::next`]
/// yields a space (`' '`) codepoint for an illegal position and advances by one
/// byte.
#[derive(Clone, Debug)]
pub struct Utf8Iter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Utf8Iter<'a> {
    /// Start iterating over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Utf8Iter { data, pos: 0 }
    }

    /// The byte length of the current codepoint, or `1` for an illegal position.
    ///
    /// Tesseract: `const_iterator::utf8_len` (unichar.cpp:195).
    pub fn utf8_len(&self) -> usize {
        let step = utf8_step(&self.data[self.pos..]);
        if step == 0 { 1 } else { step }
    }

    /// Whether the current position holds a legal UTF-8 character.
    ///
    /// Tesseract: `const_iterator::is_legal` (unichar.cpp:205).
    pub fn is_legal(&self) -> bool {
        utf8_step(&self.data[self.pos..]) > 0
    }
}

impl Iterator for Utf8Iter<'_> {
    type Item = Char32;

    /// Tesseract: `const_iterator::operator*` then `operator++` (unichar.cpp:172,
    /// 158). Returns `' '` for an illegal codepoint (and still advances).
    fn next(&mut self) -> Option<Char32> {
        if self.pos >= self.data.len() {
            return None;
        }
        let step = utf8_step(&self.data[self.pos..]);
        if step == 0 {
            // Illegal UTF-8: yield a space and advance one byte.
            self.pos += 1;
            return Some(b' ' as Char32);
        }
        let end = (self.pos + step).min(self.data.len());
        let uni = Unichar::from_utf8(&self.data[self.pos..end]).first_uni();
        self.pos = end;
        Some(uni)
    }
}

/// Convert a UTF-8 buffer to a vector of UCS-4 codepoints.
///
/// Tesseract: `UNICHAR::UTF8ToUTF32` (unichar.cpp:220). Returns [`None`]
/// (Tesseract's empty vector) if the input contains any illegal UTF-8.
pub fn utf8_to_utf32(utf8_str: &[u8]) -> Option<Vec<Char32>> {
    let mut unicodes = Vec::with_capacity(utf8_str.len());
    let mut it = Utf8Iter::new(utf8_str);
    while it.pos < it.data.len() {
        if !it.is_legal() {
            return None;
        }
        // is_legal() guarantees a codepoint is available.
        unicodes.push(it.next().unwrap());
    }
    Some(unicodes)
}

/// Convert a vector of UCS-4 codepoints to a UTF-8 byte vector.
///
/// Tesseract: `UNICHAR::UTF32ToUTF8` (unichar.cpp:237). Returns an
/// [`Error`]`(Format)` (Tesseract's empty string) if any codepoint is illegal.
pub fn utf32_to_utf8(str32: &[Char32]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for &ch in str32 {
        let uni = Unichar::from_unicode(ch);
        if uni.utf8_len() > 0 && utf8_step(uni.utf8()) > 0 {
            out.extend_from_slice(uni.utf8());
        } else {
            return Err(Error::format(format!(
                "utf32_to_utf8: illegal codepoint {ch}"
            )));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_step_classifies_first_byte() {
        assert_eq!(utf8_step(b"A"), 1); // 0x41
        assert_eq!(utf8_step("é".as_bytes()), 2); // 0xC3 ...
        assert_eq!(utf8_step("标".as_bytes()), 3); // 0xE6 ...
        assert_eq!(utf8_step("😀".as_bytes()), 4); // 0xF0 ...
        assert_eq!(utf8_step(&[0x80]), 0); // continuation byte can't start
        assert_eq!(utf8_step(&[0xFF]), 0);
    }

    #[test]
    fn ascii_roundtrip() {
        for cp in [0x00, 0x20, 0x41, 0x7F] {
            let u = Unichar::from_unicode(cp);
            assert_eq!(u.utf8_len(), 1);
            assert_eq!(Unichar::from_utf8(u.utf8()).first_uni(), cp);
        }
    }

    #[test]
    fn multibyte_roundtrip_including_cjk() {
        // 2-byte (Latin-1 supplement), 3-byte (CJK), 4-byte (emoji / SMP).
        let codepoints = [
            0xE9,    // é
            0x6807,  // 标  (a Han character used in Tesseract's own comment)
            0x4E2D,  // 中
            0xAC00,  // 가  (first Hangul syllable)
            0x1F600, // 😀
        ];
        for cp in codepoints {
            let u = Unichar::from_unicode(cp);
            // Cross-check against Rust's own UTF-8 encoder.
            let expected = char::from_u32(cp as u32).unwrap().to_string();
            assert_eq!(u.utf8(), expected.as_bytes(), "cp {cp:#x}");
            assert_eq!(Unichar::from_utf8(u.utf8()).first_uni(), cp, "cp {cp:#x}");
        }
    }

    #[test]
    fn from_unicode_rejects_above_max_utf32() {
        // Above U+10FFFF falls through to the empty case (Tesseract's memset).
        assert!(Unichar::from_unicode(0x11_0000).is_empty());
        assert!(Unichar::from_unicode(0x7FFF_FFFF).is_empty());
    }

    #[test]
    fn utf8_to_utf32_roundtrip() {
        let s = "A标가😀z".as_bytes();
        let u32s = utf8_to_utf32(s).unwrap();
        assert_eq!(u32s, vec![0x41, 0x6807, 0xAC00, 0x1F600, 0x7A]);
        let back = utf32_to_utf8(&u32s).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn utf8_to_utf32_rejects_illegal() {
        // 0x80 is a lone continuation byte.
        assert!(utf8_to_utf32(&[0x41, 0x80, 0x42]).is_none());
    }

    #[test]
    fn iterator_yields_space_for_illegal() {
        let mut it = Utf8Iter::new(&[0x41, 0xFF, 0x42]);
        assert_eq!(it.next(), Some(0x41));
        assert_eq!(it.next(), Some(b' ' as Char32)); // illegal -> space
        assert_eq!(it.next(), Some(0x42));
        assert_eq!(it.next(), None);
    }
}
