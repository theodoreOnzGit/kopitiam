//! `decode(encode(x)) == x` over a broad corpus.
//!
//! This is called out explicitly in this crate's design brief as *the*
//! property that matters most for a byte-level tokenizer: because the base
//! alphabet is raw bytes rather than characters, there is no `<UNK>` and no
//! input that can fail to round-trip -- if this test ever fails, it means
//! something in the encode/decode pipeline is dropping or corrupting bytes,
//! which would otherwise show up only much later as garbled model output.

use kopitiam_tokenizer::{BpeTokenizer, Tokenizer};

/// Builds a tokenizer with the full 256-entry byte-level base vocab plus a
/// handful of hand-picked merges (so the merge loop is actually exercised,
/// not bypassed) and two Qwen-style special tokens.
///
/// Round-tripping does not actually depend on *which* merges fire --
/// `MergeTable::build` only ever accepts a merge rule whose output token's
/// bytes are the exact concatenation of its two inputs' bytes (see
/// `merges.rs`), so byte-exactness is an invariant of the vocab
/// construction itself, not something that happens to hold for this
/// particular merge set. The merges here exist so this test exercises the
/// real merge loop rather than a degenerate byte-only path.
fn tokenizer() -> BpeTokenizer {
    let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    let mut merges = Vec::new();

    let mut add_merge = |vocab: &mut Vec<Vec<u8>>, a: &[u8], b: &[u8]| {
        let mut merged = a.to_vec();
        merged.extend_from_slice(b);
        vocab.push(merged);
        merges.push((a.to_vec(), b.to_vec()));
    };

    add_merge(&mut vocab, b"t", b"h"); // "th"
    add_merge(&mut vocab, b"th", b"e"); // "the"
    add_merge(&mut vocab, b" ", b"t"); // " t"
    add_merge(&mut vocab, b"l", b"l"); // "ll"
    add_merge(&mut vocab, b"e", b"e"); // "ee"

    let mut tok = BpeTokenizer::from_vocab_and_merges(vocab, merges)
        .expect("hand-built round-trip test vocab is valid");
    tok.add_special_token("<|endoftext|>", 300).unwrap();
    tok.add_special_token("<|im_start|>", 301).unwrap();
    tok.add_special_token("<|im_end|>", 302).unwrap();
    tok
}

fn assert_round_trips(tok: &BpeTokenizer, s: &str) {
    let ids = tok.encode(s).expect("byte-level encode never fails");
    let decoded = tok
        .decode(&ids)
        .expect("every id encode produced is in the vocab");
    assert_eq!(decoded, s, "round-trip failed for {s:?} (ids: {ids:?})");
}

#[test]
fn round_trips_plain_ascii() {
    let tok = tokenizer();
    for s in [
        "the quick brown fox jumps over the lazy dog",
        "Hello, World!",
        "don't stop believing",
        "a1 b22 c333",
        "SHOUTING and whispering",
        "",
        " ",
        "   leading and trailing   ",
    ] {
        assert_round_trips(&tok, s);
    }
}

#[test]
fn round_trips_whitespace_variety() {
    let tok = tokenizer();
    for s in [
        "a\tb",
        "a\nb",
        "a\r\nb",
        "a\n\n\nb",
        "a\t\t\tb",
        "line one\nline two\nline three",
        "col1\tcol2\tcol3",
        "  multiple   internal   spaces  ",
        "\n\n\n",
        "\t \n \t",
    ] {
        assert_round_trips(&tok, s);
    }
}

#[test]
fn round_trips_accented_latin() {
    let tok = tokenizer();
    for s in [
        "café",
        "naïve résumé",
        "Zürich, Müller, Straße",
        "El niño está aquí",
        "Ā ā Ē ē Ī ī Ō ō Ū ū",
    ] {
        assert_round_trips(&tok, s);
    }
}

#[test]
fn round_trips_cjk() {
    let tok = tokenizer();
    for s in [
        "你好，世界",
        "日本語のテキストです",
        "한국어 텍스트입니다",
        "繁體中文測試",
        "混合 mixed 文字 text",
    ] {
        assert_round_trips(&tok, s);
    }
}

#[test]
fn round_trips_emoji_and_symbols() {
    let tok = tokenizer();
    for s in [
        "🎉🎊✨",
        "family: 👨‍👩‍👧‍👦",
        "flags: 🇸🇬🇺🇸",
        "math: ∑ ∫ √ ≠ ≤ ≥ ∞",
        "arrows: → ← ↑ ↓ ⭢",
        "combining: e\u{0301} (e + combining acute accent)",
    ] {
        assert_round_trips(&tok, s);
    }
}

#[test]
fn round_trips_raw_punctuation_runs() {
    let tok = tokenizer();
    for s in [
        "!!!???...",
        "***&&&%%%",
        "()[]{}<>",
        "~`!@#$%^&*()_+-=",
        "\"quoted 'nested' text\"",
        "path/to/some\\file.ext",
    ] {
        assert_round_trips(&tok, s);
    }
}

#[test]
fn round_trips_text_containing_special_token_strings() {
    let tok = tokenizer();
    for s in [
        "<|endoftext|>",
        "before<|endoftext|>after",
        "<|im_start|>user\nhello<|im_end|>",
        "<|im_start|><|im_end|>",
        // A special token's literal text appearing where it is *not*
        // registered still round-trips: it just gets pre-tokenized and
        // BPE-encoded like any other text, byte for byte.
        "<|not_a_real_special|>",
    ] {
        assert_round_trips(&tok, s);
    }
}

/// The same corpus as the other tests, concatenated into a handful of long
/// mixed strings, so the property is also checked across chunk and segment
/// boundaries rather than only on isolated short inputs.
#[test]
fn round_trips_long_mixed_strings() {
    let tok = tokenizer();
    let mut big = String::new();
    for _ in 0..50 {
        big.push_str("the quick brown fox 你好 🎉 café \t naïve\n");
    }
    assert_round_trips(&tok, &big);

    let mixed = "<|im_start|>system\n\
                 You are a helpful assistant. 你好! Emoji: 🚀🔥. \
                 Math: E=mc². Whitespace:   \t\n  end.<|im_end|>\
                 <|im_start|>user\nHello, world!<|im_end|>";
    assert_round_trips(&tok, mixed);
}

/// A broader, mostly-generated corpus: every printable ASCII byte, plus
/// every codepoint in a sampling of Unicode blocks, individually and in a
/// few combinations -- covers the "many strings" requirement without
/// hand-typing hundreds of literals.
#[test]
fn round_trips_generated_corpus() {
    let tok = tokenizer();

    // Every single printable ASCII character on its own.
    for b in 0x20u8..=0x7E {
        let s = (b as char).to_string();
        assert_round_trips(&tok, &s);
    }

    // Every byte value 0..=255 individually, reinterpreted through
    // `char::from_u32` where possible (covers Latin-1 supplement, which
    // includes several of the "awkward" bytes the byte-to-unicode map
    // treats specially -- though note this is about exercising *input
    // text* the tokenizer receives, an orthogonal concern from that map,
    // which only ever operates on already-loaded vocab strings).
    for cp in 0u32..=0xFF {
        if let Some(c) = char::from_u32(cp) {
            let s = c.to_string();
            assert_round_trips(&tok, &s);
        }
    }

    // A sampling of higher Unicode blocks: Greek, Cyrillic, Hiragana,
    // Hangul syllables, CJK unified ideographs, emoji.
    let blocks: &[std::ops::RangeInclusive<u32>] = &[
        0x0370..=0x03FF,   // Greek and Coptic
        0x0400..=0x04FF,   // Cyrillic
        0x3040..=0x309F,   // Hiragana
        0xAC00..=0xAC20,   // a slice of Hangul syllables
        0x4E00..=0x4E20,   // a slice of CJK unified ideographs
        0x1F600..=0x1F620, // a slice of emoticons
    ];
    for range in blocks {
        let mut s = String::new();
        for cp in range.clone() {
            if let Some(c) = char::from_u32(cp) {
                s.push(c);
            }
        }
        if !s.is_empty() {
            assert_round_trips(&tok, &s);
        }
    }
}
