//! The GPT-2 "byte-to-unicode" alphabet.
//!
//! # Why this exists
//!
//! Byte-level BPE's whole selling point is that the base alphabet is the 256
//! possible byte values, so *any* input -- valid UTF-8, invalid UTF-8, raw
//! binary -- is representable with no `<UNK>` token. But a `tokenizer.json`
//! is itself a JSON document, and JSON strings cannot contain raw control
//! bytes (0x00-0x1F), the byte 0x22 (`"`) needs escaping, whitespace bytes
//! collide with JSON's own whitespace handling in careless implementations,
//! and a lone unpaired continuation byte (0x80-0xBF) is not valid UTF-8 on
//! its own -- so it cannot appear literally inside a JSON string at all.
//!
//! GPT-2's `encoder.py` solves this with a reversible byte <-> `char`
//! bijection: bytes that are already "nice" printable characters (roughly
//! ASCII `!`..`~` plus a couple of Latin-1 supplement ranges) map to
//! themselves, and the remaining ~68 awkward bytes (controls, space, DEL,
//! and the unassigned/continuation bytes in 0x7F-0xA0/0xAD) are remapped to
//! otherwise-unused codepoints starting at U+0100. Every one of the 256
//! resulting characters is then representable as an ordinary JSON string
//! character, so a `tokenizer.json` vocab can hold `"Ġhello"` (where `Ġ`
//! stands in for the space byte 0x20) instead of needing binary escapes.
//!
//! This mapping is a *serialization* concern only. Once a vocab is loaded,
//! [`crate::vocab::Vocab`] stores each token's canonical raw bytes, and the
//! rest of this crate (encode, decode, merges) never looks at mapped
//! characters again -- see [`crate::loader`] for the one place this map is
//! actually used.
//!
//! Reference: <https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9>

use std::sync::LazyLock;

/// `BYTE_TO_UNICODE[b]` is the character byte `b` is mapped to.
static BYTE_TO_UNICODE: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut printable = [false; 256];
    for b in 0x21u16..=0x7E {
        printable[b as usize] = true; // '!'..='~'
    }
    for b in 0xA1u16..=0xAC {
        printable[b as usize] = true;
    }
    for b in 0xAEu16..=0xFF {
        printable[b as usize] = true;
    }

    let mut table = ['\u{0}'; 256];
    let mut next_extra: u32 = 0;
    for b in 0..256usize {
        table[b] = if printable[b] {
            // Safety net: every value 0..256 is a valid Unicode scalar
            // value, so this never hits the `None` branch below.
            char::from_u32(b as u32).expect("byte value is always a valid scalar value")
        } else {
            let c = char::from_u32(256 + next_extra)
                .expect("256..(256+256) is well within the Basic Multilingual Plane");
            next_extra += 1;
            c
        };
    }
    table
});

/// Reverse of [`BYTE_TO_UNICODE`], built once from it so the two can never
/// drift out of sync with each other.
static UNICODE_TO_BYTE: LazyLock<std::collections::HashMap<char, u8>> = LazyLock::new(|| {
    BYTE_TO_UNICODE
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect()
});

/// Maps a raw byte to its GPT-2 byte-level-alphabet character.
///
/// Used only when *writing* a byte-level vocab out in JSON-safe form; the
/// runtime encode/decode path never calls this (see the module docs).
pub fn byte_to_unicode(b: u8) -> char {
    BYTE_TO_UNICODE[b as usize]
}

/// Inverse of [`byte_to_unicode`]: recovers the raw byte a mapped character
/// stands for, or `None` if `c` is not one of the 256 characters in the
/// byte-level alphabet (which means the vocab this came from is not a valid
/// byte-level BPE vocab).
pub fn unicode_to_byte(c: char) -> Option<u8> {
    UNICODE_TO_BYTE.get(&c).copied()
}

/// Decodes a mapped-alphabet token string (as it appears verbatim in a
/// `tokenizer.json` vocab key) back into the raw bytes it represents.
///
/// Returns `None` if any character in `s` falls outside the 256-character
/// byte-level alphabet -- that indicates the caller handed this a token
/// string that was never byte-level-mapped in the first place.
pub fn decode_mapped_token(s: &str) -> Option<Vec<u8>> {
    s.chars().map(unicode_to_byte).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The single most important property of this module: the map must be
    /// a bijection over all 256 byte values, or byte-level BPE silently
    /// loses information round-tripping through a `tokenizer.json`.
    #[test]
    fn byte_to_unicode_round_trips_every_byte() {
        for b in 0u16..=255 {
            let b = b as u8;
            let c = byte_to_unicode(b);
            assert_eq!(
                unicode_to_byte(c),
                Some(b),
                "byte {b:#04x} did not round-trip through char {c:?}"
            );
        }
    }

    #[test]
    fn mapped_characters_are_pairwise_distinct() {
        let mut seen = std::collections::HashSet::new();
        for b in 0u16..=255 {
            let c = byte_to_unicode(b as u8);
            assert!(seen.insert(c), "char {c:?} produced by two different bytes");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        // Spot-check the identity part of the mapping.
        assert_eq!(byte_to_unicode(b'A'), 'A');
        assert_eq!(byte_to_unicode(b'!'), '!');
        assert_eq!(byte_to_unicode(b'~'), '~');
    }

    #[test]
    fn space_maps_to_the_conventional_gpt2_placeholder() {
        // 0x20 (space) is not in any of the "printable" ranges, so it gets
        // remapped. GPT-2/GPT-4/Qwen vocabs universally render it as 'Ġ'
        // (U+0120) -- this is the very first "extra" codepoint assigned,
        // since space is byte 0 in ascending scan order among the
        // non-printable bytes.
        assert_eq!(byte_to_unicode(b' '), 'Ġ');
    }

    #[test]
    fn decode_mapped_token_round_trips() {
        let bytes: Vec<u8> = b" hello\n".to_vec();
        let mapped: String = bytes.iter().map(|&b| byte_to_unicode(b)).collect();
        assert_eq!(decode_mapped_token(&mapped), Some(bytes));
    }

    #[test]
    fn decode_mapped_token_rejects_foreign_characters() {
        assert_eq!(decode_mapped_token("not-byte-level-\u{1F600}"), None);
    }
}
