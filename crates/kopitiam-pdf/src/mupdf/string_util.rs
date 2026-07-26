//! Ported from MuPDF `source/fitz/string.c` + `include/mupdf/fitz/string-util.h`
//! (commit 19f1284, AGPL-3.0, © Artifex Software, Inc.), translated to Rust for
//! KOPITIAM (AGPL-3.0-only). Close adaptation: the algorithms and numeric
//! behaviour follow MuPDF; the code is re-expressed in idiomatic Rust. See
//! docs/ACKNOWLEDGEMENTS.md ("PDF & document-extraction references").
//!
//! # Scope
//!
//! This module ports the **UTF-8 rune codec and the size-bounded string helpers**
//! that the text-extraction path leans on. Following the port conventions in
//! `docs/ai-decisions/AID-0051-mupdf-port-conventions.md`, free functions keep
//! their `// MuPDF: fz_<name>` breadcrumb so the 1:1 map stays findable, and MuPDF
//! semantics are reproduced exactly where they differ from Rust's std.
//!
//! Ported here:
//!
//! * [`chartorune`] -- `fz_chartorune` / `fz_chartorunen` (UTF-8 decode).
//! * [`runetochar`] -- `fz_runetochar` (UTF-8 encode).
//! * [`runelen`] -- `fz_runelen` (encoded byte length of a rune).
//! * [`utflen`] -- `fz_utflen` (count runes).
//! * [`strlcpy`] / [`strlcat`] -- `fz_strlcpy` / `fz_strlcat` (bounded copy/cat).
//! * [`strsep`] -- `fz_strsep` (delimited tokeniser).
//! * [`tolower`], [`strcasecmp`], [`strncasecmp`] -- `fz_tolower`,
//!   `fz_strcasecmp`, `fz_strncasecmp` (see the Unicode caveat below).
//!
//! ## Runes are `u32`, not `char`
//!
//! MuPDF's rune type is a C `int`, and the decoder deliberately does **not**
//! reject the UTF-16 surrogate range or noncharacters in a 3-byte sequence -- the
//! bytes `ED A0 80` decode to `0xD800`, which is *not* a valid Rust `char`. So the
//! codec speaks `u32`, and [`runetochar`] accepts arbitrary `u32` (out-of-range
//! values fold to the replacement character exactly as MuPDF does), rather than
//! forcing the caller through `char`, which would silently reject those inputs.
//!
//! ## Deliberately NOT ported here
//!
//! * **Filesystem path helpers** -- `fz_cleanname`, `fz_realpath`, `fz_dirname`,
//!   `fz_basename`, `fz_format_output_path`: not on the text-extraction path;
//!   `std::path` covers this when we need it.
//! * **`printf`-family formatting** -- `fz_strtof`/`fz_atof`/`fz_grisu` and the
//!   number parsing belong with `printf.c`, not this module.
//! * **URI en/decode, `fz_memmem`, `fz_strstr(case)`, `fz_strverscmp`,
//!   `fz_wchar_*`, the `fz_pack_*`/`fz_unpack_*` inlines** from the header: not
//!   needed by extraction yet; port them into their own module if a later phase
//!   asks.
//! * **The full Unicode case-folding tables (`utfdata.h`).** `fz_tolower` folds
//!   ASCII inline (that part lives in `string.c`), then consults the
//!   `ucd_tolower1`/`ucd_tolower2` tables, which are a **separate generated data
//!   file** (`source/fitz/utfdata.h`, 600+ rows derived from `UnicodeData.txt`) --
//!   another translation unit, exactly the kind AID-0051 §6 defers. So [`tolower`]
//!   here folds only the ASCII range and returns other codepoints unchanged, and
//!   [`strcasecmp`]/[`strncasecmp`] are correspondingly case-insensitive for ASCII
//!   only. When a phase needs full Unicode folding, port `utfdata.h` as a
//!   `utfdata` module and route [`tolower`] through it.

/// `FZ_REPLACEMENT_CHARACTER` -- the rune substituted for input that is unknown or
/// unrepresentable (`0xFFFD`). MuPDF: `string-util.h`.
pub const FZ_REPLACEMENT_CHARACTER: u32 = 0xFFFD;

/// `FZ_UTFMAX` -- maximum number of bytes in a decoded rune (the largest value
/// [`chartorune`] can consume, and the buffer size [`runetochar`] can require).
pub const FZ_UTFMAX: usize = 4;

// ---------------------------------------------------------------------------
// UTF-8 codec constants (MuPDF: string.c:532..566, the `Bit*`/`T*`/`Rune*` enums)
// ---------------------------------------------------------------------------
//
// Kept as the exact derived values MuPDF computes so the decode/encode maths is a
// byte-for-byte transcription of the C. Decode works in `i32` (C `int`); encode
// works in `u32` (C casts the signed rune to `unsigned` up front).

const BITX: i32 = 6; // continuation-byte payload width

const TX: i32 = 0x80; // continuation-byte tag       1000 0000
const T2: i32 = 0xC0; // 2-byte lead tag             1100 0000
const T3: i32 = 0xE0; // 3-byte lead tag             1110 0000
const T4: i32 = 0xF0; // 4-byte lead tag             1111 0000
const T5: i32 = 0xF8; // (one past 4-byte lead)      1111 1000

const RUNE1: i32 = 0x7F; // max value in 1 byte
const RUNE2: i32 = 0x7FF; // max value in 2 bytes
const RUNE3: i32 = 0xFFFF; // max value in 3 bytes
const RUNE4: i32 = 0x1F_FFFF; // max value in 4 bytes

const MASKX: i32 = 0x3F; // continuation-byte payload mask   0011 1111
const TESTX: i32 = 0xC0; // detects a malformed continuation 1100 0000

/// `Runeerror`/`Bad` -- decode failure sentinel (`0xFFFD`).
const BAD: u32 = FZ_REPLACEMENT_CHARACTER;
/// `Runemax` -- largest legal rune (`0x10FFFF`).
const RUNEMAX: u32 = 0x10_FFFF;

// ---------------------------------------------------------------------------
// UTF-8 decode
// ---------------------------------------------------------------------------

/// Decode a single UTF-8 rune from the front of `s`, returning
/// `(rune, bytes_consumed)`.
///
/// This fuses MuPDF's `fz_chartorune` and `fz_chartorunen`: a Rust slice already
/// carries its length, so `s.len()` plays the role of `fz_chartorunen`'s `n`, and
/// we never read past the slice (the C `fz_chartorune` relies on the string's NUL
/// terminator and would read one byte beyond a truncated tail).
///
/// Semantics reproduced from MuPDF, and these differ from `str::chars`:
///
/// * The **overlong NUL** `C0 80` decodes to rune `0` consuming **2** bytes (this
///   is MuPDF's own "modified UTF-8" NUL; std treats `C0` as invalid).
/// * **Any** malformed / truncated / overlong sequence yields
///   `(0xFFFD, 1)` -- rune `0xFFFD`, exactly **one** byte consumed, so the caller
///   resynchronises at the next byte.
/// * Surrogate-range and noncharacter codepoints in a 3-byte sequence are **not**
///   rejected (e.g. `ED A0 80` -> `(0xD800, 3)`); MuPDF only rejects overlong
///   encodings via the `l <= Rune{n-1}` checks.
///
/// An empty slice yields `(0xFFFD, 1)` (MuPDF's `n < 1` -> bad path).
// MuPDF: fz_chartorune (string.c:568) / fz_chartorunen (string.c:649)
pub fn chartorune(s: &[u8]) -> (u32, usize) {
    let n = s.len();
    if n < 1 {
        return (BAD, 1);
    }

    // one character sequence: 00000-0007F => T1
    let c = s[0] as i32;
    if c < TX {
        return (c as u32, 1);
    }

    if n < 2 {
        return (BAD, 1);
    }

    // overlong null character
    if s[0] == 0xc0 && s[1] == 0x80 {
        return (0, 2);
    }

    // two character sequence: 0080-07FF => T2 Tx
    let c1 = (s[1] as i32) ^ TX;
    if c1 & TESTX != 0 {
        return (BAD, 1);
    }
    if c < T3 {
        if c < T2 {
            return (BAD, 1);
        }
        let l = ((c << BITX) | c1) & RUNE2;
        if l <= RUNE1 {
            return (BAD, 1);
        }
        return (l as u32, 2);
    }

    if n < 3 {
        return (BAD, 1);
    }

    // three character sequence: 0800-FFFF => T3 Tx Tx
    let c2 = (s[2] as i32) ^ TX;
    if c2 & TESTX != 0 {
        return (BAD, 1);
    }
    if c < T4 {
        let l = ((((c << BITX) | c1) << BITX) | c2) & RUNE3;
        if l <= RUNE2 {
            return (BAD, 1);
        }
        return (l as u32, 3);
    }

    if n < 4 {
        return (BAD, 1);
    }

    // four character sequence (21-bit value): 10000-1FFFFF => T4 Tx Tx Tx
    let c3 = (s[3] as i32) ^ TX;
    if c3 & TESTX != 0 {
        return (BAD, 1);
    }
    if c < T5 {
        let l = ((((((c << BITX) | c1) << BITX) | c2) << BITX) | c3) & RUNE4;
        if l <= RUNE3 {
            return (BAD, 1);
        }
        return (l as u32, 4);
    }

    // 5-byte-or-longer sequences are unsupported: fall through to bad.
    (BAD, 1)
}

// ---------------------------------------------------------------------------
// UTF-8 encode
// ---------------------------------------------------------------------------

/// Encode `rune` as UTF-8 into the front of `buf`, returning the byte count.
///
/// `buf` must be at least [`runelen`]`(rune)` bytes long; a `FZ_UTFMAX`-byte
/// buffer is always sufficient. Semantics reproduced from MuPDF:
///
/// * rune `0` encodes to the **overlong NUL** `C0 80` (2 bytes), the inverse of
///   [`chartorune`]'s special case (std would emit a single `00`).
/// * A rune greater than `0x10FFFF` is folded to `0xFFFD` before encoding (so it
///   emits 3 bytes), matching MuPDF's out-of-range handling.
// MuPDF: fz_runetochar (string.c:742)
pub fn runetochar(buf: &mut [u8], rune: u32) -> usize {
    let mut c = rune;

    // overlong null character
    if c == 0 {
        buf[0] = 0xc0;
        buf[1] = 0x80;
        return 2;
    }

    // one character sequence: 00000-0007F => 00-7F
    if c <= RUNE1 as u32 {
        buf[0] = c as u8;
        return 1;
    }

    // two character sequence: 0080-07FF => T2 Tx
    if c <= RUNE2 as u32 {
        buf[0] = (T2 as u32 | (c >> BITX)) as u8;
        buf[1] = (TX as u32 | (c & MASKX as u32)) as u8;
        return 2;
    }

    // Out-of-range runes become the error rune (which encodes to three bytes).
    if c > RUNEMAX {
        c = FZ_REPLACEMENT_CHARACTER;
    }

    // three character sequence: 0800-FFFF => T3 Tx Tx
    if c <= RUNE3 as u32 {
        buf[0] = (T3 as u32 | (c >> (2 * BITX))) as u8;
        buf[1] = (TX as u32 | ((c >> BITX) & MASKX as u32)) as u8;
        buf[2] = (TX as u32 | (c & MASKX as u32)) as u8;
        return 3;
    }

    // four character sequence (21-bit value): 10000-1FFFFF => T4 Tx Tx Tx
    buf[0] = (T4 as u32 | (c >> (3 * BITX))) as u8;
    buf[1] = (TX as u32 | ((c >> (2 * BITX)) & MASKX as u32)) as u8;
    buf[2] = (TX as u32 | ((c >> BITX) & MASKX as u32)) as u8;
    buf[3] = (TX as u32 | (c & MASKX as u32)) as u8;
    4
}

/// Number of bytes [`runetochar`] would emit for `rune`.
///
/// Mirrors the branch structure of [`runetochar`] (MuPDF computes this by encoding
/// into a scratch buffer). Note rune `0` -> `2` (the overlong NUL) and any rune
/// above `0x10FFFF` -> `3` (folded to the replacement character first).
// MuPDF: fz_runelen (string.c:805)
pub fn runelen(rune: u32) -> usize {
    if rune == 0 {
        return 2;
    }
    if rune <= RUNE1 as u32 {
        return 1;
    }
    if rune <= RUNE2 as u32 {
        return 2;
    }
    if rune > RUNEMAX {
        // folds to 0xFFFD, which is a 3-byte encoding
        return 3;
    }
    if rune <= RUNE3 as u32 {
        return 3;
    }
    4
}

/// Count how many runes the UTF-8 encoded `s` decodes to, stepping with
/// [`chartorune`] (so malformed bytes each count as one replacement rune).
///
/// MuPDF's `fz_utflen` stops at a NUL terminator; a Rust slice carries its own
/// length, so this counts every rune across the whole slice. If your data is
/// NUL-terminated, slice it to the NUL first.
// MuPDF: fz_utflen (string.c:843)
pub fn utflen(mut s: &[u8]) -> usize {
    let mut n = 0;
    while !s.is_empty() {
        let (_rune, len) = chartorune(s);
        s = &s[len..];
        n += 1;
    }
    n
}

// ---------------------------------------------------------------------------
// Size-bounded copy / concatenate
// ---------------------------------------------------------------------------
//
// `dst.len()` plays the role of MuPDF's `siz` (the destination buffer size). `src`
// is treated as a C string: its logical end is the first NUL byte, or the end of
// the slice if there is none. Both return the length the fully-copied string
// *would* have had (excluding the terminator) -- the value that lets a caller
// detect truncation -- which is the subtle part these share with `strlcpy(3)` and
// differ from `str`/`Vec` copies.

/// `src` as a C string: index of its first NUL, or `src.len()` if none.
#[inline]
fn cstrlen(src: &[u8]) -> usize {
    src.iter().position(|&b| b == 0).unwrap_or(src.len())
}

/// Byte of `src` at logical index `i`, with a virtual NUL at (and past) its end.
#[inline]
fn cbyte(src: &[u8], i: usize) -> u8 {
    if i < src.len() { src[i] } else { 0 }
}

/// Copy `src` into `dst` (capacity `dst.len()`), NUL-terminating, and return the
/// length of `src` (excluding terminator) -- i.e. what a full copy would need.
///
/// At most `dst.len() - 1` bytes of `src` are copied. If `dst.len() == 0` nothing
/// is written. The return value is the source length regardless of truncation, so
/// `ret >= dst.len()` signals the copy was truncated.
// MuPDF: fz_strlcpy (string.c:166)
pub fn strlcpy(dst: &mut [u8], src: &[u8]) -> usize {
    let siz = dst.len();
    let mut d = 0usize;
    let mut s = 0usize;
    let mut n = siz;

    // Copy as many bytes as will fit.
    if n != 0 && {
        n -= 1;
        n != 0
    } {
        loop {
            let b = cbyte(src, s);
            s += 1;
            dst[d] = b;
            d += 1;
            if b == 0 {
                break;
            }
            n -= 1;
            if n == 0 {
                break;
            }
        }
    }

    // Not enough room in dst: NUL-terminate and traverse the rest of src.
    if n == 0 {
        if siz != 0 {
            dst[d] = 0; // d == siz - 1 here
        }
        while cbyte(src, s) != 0 {
            s += 1;
        }
        s += 1; // consume the (virtual) NUL, mirroring C's `while (*s++);`
    }

    s - 1 // count excludes the NUL
}

/// Append `src` onto the C string already in `dst` (buffer capacity `dst.len()`),
/// NUL-terminating, and return the length the concatenation *would* have had
/// (excluding terminator).
///
/// If `dst` holds no NUL within its capacity, the destination is treated as full
/// and the return value is `dst.len() + cstrlen(src)` (the classic `strlcat`
/// behaviour), with nothing appended.
// MuPDF: fz_strlcat (string.c:192)
pub fn strlcat(dst: &mut [u8], src: &[u8]) -> usize {
    let siz = dst.len();

    // Find the end of dst, bounded by the buffer size.
    let mut d = 0usize;
    let mut n = siz;
    while d < siz && dst[d] != 0 && n != 0 {
        n -= 1;
        d += 1;
    }
    let dlen = d;

    let mut n = siz - dlen;
    if n == 0 {
        return dlen + cstrlen(src);
    }

    let mut s = 0usize;
    while cbyte(src, s) != 0 {
        if n != 1 {
            dst[d] = cbyte(src, s);
            d += 1;
            n -= 1;
        }
        s += 1;
    }
    dst[d] = 0;

    dlen + s // count excludes the NUL
}

// ---------------------------------------------------------------------------
// Delimited tokeniser
// ---------------------------------------------------------------------------

/// Split off the next `delim`-delimited token from `*stringp`, advancing the
/// cursor past the delimiter, and return the token (or `None` when the cursor is
/// exhausted).
///
/// `stringp: &mut Option<&[u8]>` is the idiomatic stand-in for MuPDF's
/// `char **stringp`: `None` is the C `NULL`. Faithful semantics:
///
/// * A run of the token's bytes up to (excluding) the first delimiter byte is
///   returned; the cursor is set to just after that delimiter.
/// * When no delimiter is present, the whole remaining string is returned and the
///   cursor becomes `None`.
/// * Consecutive delimiters therefore yield **empty** tokens (`Some(b"")`), and a
///   leading delimiter yields an empty first token -- exactly like `strsep(3)`.
/// * A `None` cursor returns `None`.
///
/// Unlike the C, which overwrites the delimiter with a NUL in place, this borrows
/// subslices of the original data and mutates only the cursor.
// MuPDF: fz_strsep (string.c:156)
pub fn strsep<'a>(stringp: &mut Option<&'a [u8]>, delim: &[u8]) -> Option<&'a [u8]> {
    let s = (*stringp)?;
    match s.iter().position(|b| delim.contains(b)) {
        Some(i) => {
            *stringp = Some(&s[i + 1..]);
            Some(&s[..i])
        }
        None => {
            *stringp = None;
            Some(s)
        }
    }
}

// ---------------------------------------------------------------------------
// Case folding and case-insensitive comparison (ASCII fold; see module caveat)
// ---------------------------------------------------------------------------

/// Fold `c` to lower case.
///
/// Only the ASCII fast path (which lives in `string.c`) is implemented: `A`..=`Z`
/// map to `a`..=`z`, everything else is returned unchanged. The Unicode branch
/// consults the `utfdata.h` case-folding tables, which are deferred (see the
/// module docs); until a `utfdata` module is ported, codepoints `>= 128` pass
/// through untouched.
// MuPDF: fz_tolower (string.c:61) -- ASCII fast path only.
pub fn tolower(c: u32) -> u32 {
    if c < 128 {
        if (b'A' as u32..=b'Z' as u32).contains(&c) {
            return c + (b'a' - b'A') as u32;
        }
        return c;
    }
    // TODO(utfdata): full Unicode case folding not yet ported.
    c
}

/// Read the next rune from a C-string-style slice, returning `(0, 0)` at the end
/// of the slice (mirroring the NUL that terminates MuPDF's loops).
#[inline]
fn next_rune(s: &[u8]) -> (u32, usize) {
    if s.is_empty() {
        (0, 0)
    } else {
        chartorune(s)
    }
}

/// Case-insensitive comparison of two UTF-8 strings, terminating at the end of
/// either slice (the C's NUL). Returns the sign of the first differing folded
/// rune, or `0` when equal.
///
/// Case-insensitive for ASCII only, per [`tolower`]'s current limitation.
// MuPDF: fz_strcasecmp (string.c:136)
pub fn strcasecmp(mut a: &[u8], mut b: &[u8]) -> i32 {
    loop {
        let (ra, na) = next_rune(a);
        let (rb, nb) = next_rune(b);
        let la = tolower(ra) as i32;
        let lb = tolower(rb) as i32;
        if la == lb {
            if la == 0 {
                return 0;
            }
        } else {
            return la - lb;
        }
        a = &a[na..];
        b = &b[nb..];
    }
}

/// Case-insensitive comparison of at most `n` bytes read from either slice.
///
/// Follows MuPDF's `fz_strncasecmp`: only lower-cases the runes when they differ,
/// and stops early at the byte budget or at the end of either string. An embedded
/// NUL (or the end of a slice) acts as a terminator. Case-insensitive for ASCII
/// only, per [`tolower`]'s current limitation.
// MuPDF: fz_strncasecmp (string.c:103)
pub fn strncasecmp(mut a: &[u8], mut b: &[u8], mut n: usize) -> i32 {
    while n > 0 {
        let (ra, na) = if a.is_empty() {
            (0u32, 0usize)
        } else {
            chartorune(&a[..a.len().min(n)])
        };
        let (rb, nb) = if b.is_empty() {
            (0u32, 0usize)
        } else {
            chartorune(&b[..b.len().min(n)])
        };

        // One or both strings ran out.
        if ra == 0 || rb == 0 {
            return ra as i32 - rb as i32;
        }

        let (ra, rb) = if ra != rb {
            (tolower(ra), tolower(rb))
        } else {
            (ra, rb)
        };
        if ra != rb {
            return ra as i32 - rb as i32;
        }

        a = &a[na..];
        b = &b[nb..];
        n -= na;
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a rune into a fresh `FZ_UTFMAX` buffer and return the used bytes.
    fn enc(rune: u32) -> Vec<u8> {
        let mut buf = [0u8; FZ_UTFMAX];
        let len = runetochar(&mut buf, rune);
        buf[..len].to_vec()
    }

    #[test]
    fn roundtrip_ascii() {
        for &r in &[0x41u32, 0x7F, b'z' as u32, b' ' as u32] {
            let bytes = enc(r);
            assert_eq!(bytes.len(), 1);
            assert_eq!(runelen(r), 1);
            assert_eq!(chartorune(&bytes), (r, 1));
        }
    }

    #[test]
    fn roundtrip_two_byte() {
        // U+00E9 é, U+07FF (max 2-byte).
        for &r in &[0xE9u32, 0x7FF, 0x80] {
            let bytes = enc(r);
            assert_eq!(bytes.len(), 2, "rune {r:#x}");
            assert_eq!(runelen(r), 2);
            assert_eq!(chartorune(&bytes), (r, 2));
        }
        // Spot-check exact encoding of U+07FF.
        assert_eq!(enc(0x7FF), vec![0xDF, 0xBF]);
    }

    #[test]
    fn roundtrip_three_byte() {
        // U+20AC €, U+0800 (min 3-byte), U+FFFF (max 3-byte).
        for &r in &[0x20ACu32, 0x800, 0xFFFF] {
            let bytes = enc(r);
            assert_eq!(bytes.len(), 3, "rune {r:#x}");
            assert_eq!(runelen(r), 3);
            assert_eq!(chartorune(&bytes), (r, 3));
        }
        assert_eq!(enc(0x20AC), vec![0xE2, 0x82, 0xAC]);
    }

    #[test]
    fn roundtrip_four_byte() {
        // U+1F600 😀, U+10000 (min 4-byte), U+10FFFF (max legal).
        for &r in &[0x1F600u32, 0x10000, 0x10FFFF] {
            let bytes = enc(r);
            assert_eq!(bytes.len(), 4, "rune {r:#x}");
            assert_eq!(runelen(r), 4);
            assert_eq!(chartorune(&bytes), (r, 4));
        }
        assert_eq!(enc(0x1F600), vec![0xF0, 0x9F, 0x98, 0x80]);
    }

    #[test]
    fn overlong_null_roundtrips_as_two_bytes() {
        // MuPDF's modified UTF-8: rune 0 <-> C0 80 (never a bare 00).
        assert_eq!(enc(0), vec![0xC0, 0x80]);
        assert_eq!(runelen(0), 2);
        assert_eq!(chartorune(&[0xC0, 0x80]), (0, 2));
        // A bare 00 still decodes as an ASCII NUL, one byte.
        assert_eq!(chartorune(&[0x00]), (0, 1));
    }

    #[test]
    fn out_of_range_rune_folds_to_replacement() {
        // > 0x10FFFF folds to 0xFFFD (a 3-byte encoding) on encode.
        assert_eq!(runelen(0x110000), 3);
        assert_eq!(enc(0x110000), enc(FZ_REPLACEMENT_CHARACTER));
        assert_eq!(enc(0x110000), vec![0xEF, 0xBF, 0xBD]);
    }

    #[test]
    fn invalid_sequences_yield_replacement_one_byte() {
        // Every malformed input: rune 0xFFFD, exactly one byte consumed.
        let cases: &[&[u8]] = &[
            &[],             // empty
            &[0x80],         // lone continuation byte (n < 2)
            &[0x80, 0x80],   // continuation byte as lead
            &[0xC0, 0xAF],   // overlong '/' (2-byte)
            &[0xC1, 0x81],   // overlong (2-byte)
            &[0xE2, 0x82],   // truncated 3-byte (n < 3)
            &[0xE0, 0x80, 0x80], // overlong 3-byte
            &[0xF0, 0x9F, 0x98], // truncated 4-byte (n < 4)
            &[0xF0, 0x80, 0x80, 0x80], // overlong 4-byte
            &[0xFF],         // invalid lead >= T5
            &[0xE2, 0xFF, 0xAC], // bad continuation in 3-byte
        ];
        for &c in cases {
            assert_eq!(
                chartorune(c),
                (FZ_REPLACEMENT_CHARACTER, 1),
                "input {c:02x?} should be replacement + 1 byte"
            );
        }
    }

    #[test]
    fn surrogates_are_not_rejected() {
        // MuPDF does not reject the surrogate range in a 3-byte sequence.
        assert_eq!(chartorune(&[0xED, 0xA0, 0x80]), (0xD800, 3));
        assert_eq!(chartorune(&[0xED, 0xBF, 0xBF]), (0xDFFF, 3));
    }

    #[test]
    fn utflen_counts_runes_including_replacements() {
        assert_eq!(utflen(b""), 0);
        assert_eq!(utflen(b"hello"), 5);
        // "a€😀" = 1 + 3 + 4 bytes, 3 runes.
        let mut s = Vec::new();
        s.extend_from_slice(b"a");
        s.extend_from_slice(&enc(0x20AC));
        s.extend_from_slice(&enc(0x1F600));
        assert_eq!(utflen(&s), 3);
        // Two lone continuation bytes -> two replacement runes.
        assert_eq!(utflen(&[0x80, 0x80]), 2);
    }

    #[test]
    fn strlcpy_fits() {
        let mut dst = [0u8; 10];
        let ret = strlcpy(&mut dst, b"hi");
        assert_eq!(ret, 2); // strlen(src)
        assert_eq!(&dst[..3], b"hi\0");
    }

    #[test]
    fn strlcpy_truncates_and_returns_source_length() {
        let mut dst = [0u8; 3];
        let ret = strlcpy(&mut dst, b"hello");
        assert_eq!(ret, 5); // full source length, not the copied length
        assert_eq!(&dst[..2], b"he");
        assert_eq!(dst[2], 0); // NUL-terminated within the 3-byte buffer
    }

    #[test]
    fn strlcpy_zero_size_writes_nothing() {
        let mut dst = [0u8; 0];
        let ret = strlcpy(&mut dst, b"x");
        assert_eq!(ret, 1);
    }

    #[test]
    fn strlcat_appends() {
        let mut dst = [0u8; 10];
        dst[0] = b'a';
        dst[1] = b'b';
        let ret = strlcat(&mut dst, b"cd");
        assert_eq!(ret, 4); // strlen("abcd")
        assert_eq!(&dst[..5], b"abcd\0");
    }

    #[test]
    fn strlcat_truncates_and_returns_intended_length() {
        let mut dst = [0u8; 4];
        dst[0] = b'a';
        dst[1] = b'b';
        let ret = strlcat(&mut dst, b"cdef");
        assert_eq!(ret, 6); // strlen("ab") + strlen("cdef")
        assert_eq!(&dst[..4], b"abc\0");
    }

    #[test]
    fn strlcat_unterminated_dst() {
        // dst has no NUL within its capacity -> treated as full.
        let mut dst = [b'x'; 3];
        let ret = strlcat(&mut dst, b"yz");
        assert_eq!(ret, 3 + 2); // dst.len() + strlen(src)
        assert_eq!(&dst, b"xxx"); // nothing appended
    }

    #[test]
    fn strsep_splits_with_empty_tokens() {
        let data = b"a,b,,c";
        let mut cur: Option<&[u8]> = Some(data);
        assert_eq!(strsep(&mut cur, b","), Some(&b"a"[..]));
        assert_eq!(strsep(&mut cur, b","), Some(&b"b"[..]));
        assert_eq!(strsep(&mut cur, b","), Some(&b""[..])); // consecutive delims
        assert_eq!(strsep(&mut cur, b","), Some(&b"c"[..]));
        assert_eq!(strsep(&mut cur, b","), None);
        assert_eq!(strsep(&mut cur, b","), None); // stays None
    }

    #[test]
    fn strsep_no_delimiter_returns_whole_string() {
        let mut cur: Option<&[u8]> = Some(&b"abc"[..]);
        assert_eq!(strsep(&mut cur, b","), Some(&b"abc"[..]));
        assert_eq!(cur, None);
    }

    #[test]
    fn strsep_leading_delimiter_yields_empty_first_token() {
        let mut cur: Option<&[u8]> = Some(&b",x"[..]);
        assert_eq!(strsep(&mut cur, b","), Some(&b""[..]));
        assert_eq!(strsep(&mut cur, b","), Some(&b"x"[..]));
        assert_eq!(strsep(&mut cur, b","), None);
    }

    #[test]
    fn tolower_folds_ascii() {
        assert_eq!(tolower(b'A' as u32), b'a' as u32);
        assert_eq!(tolower(b'Z' as u32), b'z' as u32);
        assert_eq!(tolower(b'a' as u32), b'a' as u32);
        assert_eq!(tolower(b'5' as u32), b'5' as u32);
        // Non-ASCII passes through (Unicode tables deferred).
        assert_eq!(tolower(0x00C0), 0x00C0);
    }

    #[test]
    fn strcasecmp_ascii() {
        assert_eq!(strcasecmp(b"Hello", b"hello"), 0);
        assert_eq!(strcasecmp(b"", b""), 0);
        assert!(strcasecmp(b"abc", b"abd") < 0);
        assert!(strcasecmp(b"abd", b"abc") > 0);
        assert!(strcasecmp(b"ab", b"abc") < 0); // shorter is less
        assert!(strcasecmp(b"abc", b"ab") > 0);
    }

    #[test]
    fn strncasecmp_ascii() {
        // First 5 bytes fold-equal even though the tails differ.
        assert_eq!(strncasecmp(b"HELLOxx", b"helloyy", 5), 0);
        assert_eq!(strncasecmp(b"abc", b"abd", 2), 0);
        assert!(strncasecmp(b"abc", b"abd", 3) < 0);
        assert!(strncasecmp(b"abc", b"abz", 3) < 0);
        // A zero budget compares nothing.
        assert_eq!(strncasecmp(b"abc", b"xyz", 0), 0);
    }
}
