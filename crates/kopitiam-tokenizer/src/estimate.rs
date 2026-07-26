//! Dependency-free, deterministic *token count estimation* -- the "make
//! cost visible" primitive from the token-max plan (§0.7): let an agent
//! choose read-vs-outline informed by an approximate token cost rather
//! than blind.
//!
//! # Why an estimate, not the real BPE
//!
//! This crate already contains a real, from-scratch byte-level BPE
//! ([`crate::BpeTokenizer`]) that returns the *exact* ids a GPT-2/Qwen
//! family model was trained on -- but it is inert without a loaded vocab,
//! and this workspace **bundles no standalone `tokenizer.json`** (only a
//! present GGUF model carries one, via
//! `kopitiam_runtime::tokenizer_from_gguf`). Shipping a real GPT-2/Qwen
//! merge table purely to *count* tokens would mean embedding a multi-MB
//! asset into every build for a number that only needs to be
//! approximately right. The token-max card for this task is explicit that
//! "a good BPE approximation suffices," so this module deliberately trades
//! the last ~20-30% of accuracy for zero dependencies, zero bundled
//! assets, and constant-time-per-char evaluation.
//!
//! **This is an estimate, not a billing oracle.** It exists to inform a
//! choice (is this file cheap enough to read whole, or should I outline
//! it?), where being within a quarter of the true count changes nothing
//! about the decision. It is *not* suitable for anything that must match a
//! provider's exact token accounting.
//!
//! # The model: per-script char weighting
//!
//! Real byte-level BPE compresses different scripts at very different
//! rates, and a single "chars / 4" rule (the usual English rule of thumb)
//! is wrong by up to ~4x on CJK text -- a Chinese ideograph is three UTF-8
//! bytes and costs on the order of one whole token, where four Latin
//! letters share one. So instead of one global ratio, each character
//! contributes a *weight in tokens* chosen per Unicode script/category,
//! and the estimate is the rounded sum. The weights are calibrated against
//! the well-documented behavior of the GPT-2/GPT-4/Qwen tokenizers:
//!
//! | Class                     | tokens/char | ~chars/token | rationale |
//! |---------------------------|-------------|--------------|-----------|
//! | ASCII letter              | 0.25        | 4.0          | the classic English "~4 chars/token" norm |
//! | Whitespace                | 0.25        | 4.0          | a lone inter-word space folds into the next word token, but runs/newlines cost; 0.25 keeps prose at ~chars/4 |
//! | ASCII digit               | 0.40        | 2.5          | tokenizers split numbers into 1-3 digit groups |
//! | ASCII punctuation/symbol  | 0.50        | 2.0          | often its own token, but runs merge |
//! | CJK ideograph / kana / hangul | 0.75    | ~1.33        | card's 1.0-1.5 chars/token band for ideographs |
//! | Other non-ASCII letter    | 0.50        | 2.0          | accented Latin/Cyrillic/Greek: multi-byte, mid density |
//! | Other non-ASCII symbol    | 1.00        | 1.0          | emoji/pictographs cost 1-3 tokens each |
//!
//! # Expected accuracy
//!
//! Against a real GPT-2/Qwen-family BPE tokenizer, this lands within
//! roughly **±25-30%** on ordinary prose (English, Chinese, or a mix) and
//! natural-language-heavy documents -- the regime this is built for. It is
//! looser on atypical inputs (dense source code, long digit runs, heavy
//! emoji, base64 blobs), which is acceptable for a read-vs-outline signal.
//! The three guarantees callers *can* rely on are exact: the estimate is
//! **deterministic** (same input -> same count), **monotonic** (appending
//! text never lowers the count), and **zero for the empty string**.
//!
//! # Relationship to Part I's measurements
//!
//! The token-max plan's Part I harness already reports output bytes, line
//! count, and non-whitespace char count for converted documents (§137).
//! This adds an orthogonal *token* axis (§268): bytes and lines measure
//! the artifact, tokens measure what it costs an LLM to actually read it,
//! and the two can diverge sharply (a CJK document is small in tokens
//! relative to its byte count; a wide table is the reverse).

/// The token-cost class of a single `char`, used to pick its weight.
///
/// Classification is by Unicode range/category and touches no per-model
/// state, so it is stable across builds and platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Whitespace,
    AsciiLetter,
    AsciiDigit,
    AsciiPunct,
    /// CJK ideographs, kana, hangul, and CJK punctuation/fullwidth forms.
    Cjk,
    /// Non-ASCII alphabetic that is not CJK (accented Latin, Cyrillic,
    /// Greek, ...).
    OtherLetter,
    /// Everything else non-ASCII: emoji, pictographs, box drawing, etc.
    OtherSymbol,
}

impl CharClass {
    /// The estimated token cost, in tokens, that a character of this class
    /// contributes. See the module docs for the calibration rationale.
    fn weight(self) -> f64 {
        match self {
            CharClass::Whitespace => 0.25,
            CharClass::AsciiLetter => 0.25,
            CharClass::AsciiDigit => 0.40,
            CharClass::AsciiPunct => 0.50,
            CharClass::Cjk => 0.75,
            CharClass::OtherLetter => 0.50,
            CharClass::OtherSymbol => 1.00,
        }
    }
}

/// True for the Unicode ranges this estimator treats as CJK-dense (one
/// character ~ one token): Han ideographs (incl. extensions), Japanese
/// kana, Korean hangul, and the CJK symbol/fullwidth blocks.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F   // CJK symbols and punctuation
        | 0x3040..=0x309F // Hiragana
        | 0x30A0..=0x30FF // Katakana
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF // CJK compatibility ideographs
        | 0xFF00..=0xFFEF // Halfwidth and fullwidth forms
        | 0x20000..=0x2FA1F // CJK Unified Ideographs Extensions B-F + compat supplement
    )
}

/// Assigns a single character to its [`CharClass`].
fn classify(c: char) -> CharClass {
    if c.is_whitespace() {
        CharClass::Whitespace
    } else if c.is_ascii() {
        if c.is_ascii_alphabetic() {
            CharClass::AsciiLetter
        } else if c.is_ascii_digit() {
            CharClass::AsciiDigit
        } else {
            // ASCII, non-whitespace, non-alphanumeric: punctuation/symbols
            // and the C0 control range (rare; treated as punctuation-cost).
            CharClass::AsciiPunct
        }
    } else if is_cjk(c) {
        CharClass::Cjk
    } else if c.is_alphabetic() {
        CharClass::OtherLetter
    } else {
        CharClass::OtherSymbol
    }
}

/// The raw (unrounded) estimated token cost of `text`, in tokens.
///
/// Kept separate from [`estimate_tokens`] so per-line/per-section helpers
/// can round once at the boundary they care about rather than accumulate
/// rounding error across many small pieces.
fn raw_cost(text: &str) -> f64 {
    text.chars().map(|c| classify(c).weight()).sum()
}

/// Estimates the number of BPE tokens `text` would encode to, using the
/// dependency-free per-script heuristic documented at the [module
/// level](self).
///
/// Guarantees: `estimate_tokens("") == 0`; deterministic; and monotonic
/// (for any `a` and `b`, `estimate_tokens(a) <= estimate_tokens(&(a + b))`,
/// since every character's weight is non-negative). Expect roughly
/// ±25-30% agreement with a real GPT-2/Qwen tokenizer on prose -- it is an
/// estimate for informing read-vs-outline, not exact billing.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    raw_cost(text).round() as usize
}

/// A per-line token estimate: the (1-based) line number and its estimated
/// token count. Returned by [`estimate_tokens_by_line`] so the CLI can
/// surface a breakdown (e.g. "which lines dominate this file's cost").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineEstimate {
    /// 1-based line number, matching how editors and `file:line`
    /// coordinates (§10.1 finding 4) count lines.
    pub line: usize,
    /// Estimated tokens for this line's content (its trailing `\n` is not
    /// counted -- newlines are cheap and would only add rounding noise per
    /// line).
    pub tokens: usize,
}

/// Estimates tokens per line, splitting on `\n` (a trailing `\r` on each
/// line is included in that line's content, so CRLF files behave sanely).
///
/// Each line is rounded independently, so the sum of the per-line `tokens`
/// may differ from [`estimate_tokens`] of the whole text by a few tokens
/// of rounding; treat [`estimate_tokens`] as the authoritative total and
/// this as a *relative* breakdown for spotting the expensive lines.
///
/// An empty input yields a single line-1 entry with `tokens == 0`, matching
/// `"".lines()`-style expectations for "one empty line."
#[must_use]
pub fn estimate_tokens_by_line(text: &str) -> Vec<LineEstimate> {
    text.split('\n')
        .enumerate()
        .map(|(i, line)| LineEstimate {
            line: i + 1,
            tokens: estimate_tokens(line),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn deterministic_same_input_same_count() {
        let s = "The quick brown fox jumps over the lazy dog. 你好世界 123!";
        let first = estimate_tokens(s);
        for _ in 0..5 {
            assert_eq!(estimate_tokens(s), first);
        }
    }

    #[test]
    fn monotonic_longer_text_never_fewer_tokens() {
        let base = "Lorem ipsum dolor sit amet";
        let mut acc = String::new();
        let mut prev = 0usize;
        for word in base.split(' ') {
            acc.push_str(word);
            acc.push(' ');
            let now = estimate_tokens(&acc);
            assert!(now >= prev, "estimate dropped from {prev} to {now} after adding {word:?}");
            prev = now;
        }
    }

    /// A real GPT-2/cl100k tokenizer encodes this 44-char pangram to ~10
    /// tokens (`"The"," quick"," brown"," fox"," jumps"," over"," the","
    /// lazy"," dog","."`). The heuristic must land in the documented ~4
    /// chars/token English band -- assert 3.3-5.0 chars/token, i.e. the
    /// estimate is within roughly a quarter of the real count.
    #[test]
    fn english_paragraph_within_documented_band() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let chars = text.chars().count() as f64; // 44
        let est = estimate_tokens(text) as f64;
        let chars_per_token = chars / est;
        assert!(
            (3.3..=5.0).contains(&chars_per_token),
            "English estimate {est} => {chars_per_token:.2} chars/token, want 3.3-5.0 \
             (real BPE ~10 tokens for {chars} chars)"
        );
    }

    /// A longer, more representative English paragraph, checked the same
    /// way to make sure the band holds beyond one short sentence.
    #[test]
    fn longer_english_prose_within_band() {
        let text = "Tokenization turns text into the discrete units a language model \
                    consumes. Estimating that count cheaply lets an agent decide whether \
                    to read a file whole or ask only for its outline, spending its budget \
                    where it matters most.";
        let chars = text.chars().count() as f64;
        let est = estimate_tokens(text) as f64;
        let chars_per_token = chars / est;
        assert!(
            (3.3..=5.0).contains(&chars_per_token),
            "prose estimate {est} => {chars_per_token:.2} chars/token, want 3.3-5.0"
        );
    }

    /// CJK must not be counted at the Latin ~4-chars/token rate (that would
    /// be ~4x *under* the truth). A real tokenizer spends on the order of
    /// one token per ideograph; assert 0.5-1.0 tokens/char, well above the
    /// naive `chars/4 == 0.25` a Latin-only heuristic would give.
    #[test]
    fn cjk_string_estimated_sensibly_not_4x_off() {
        // A run of Chinese ideographs (no ASCII), so every char is CJK.
        let text = "机器学习模型需要把文本转换成离散的标记单元";
        let chars = text.chars().count();
        let est = estimate_tokens(text);
        let naive_latin = chars / 4; // the wrong (4x-under) Latin answer
        assert!(
            est >= 2 * naive_latin,
            "CJK estimate {est} is too close to the Latin heuristic {naive_latin} (4x under)"
        );
        let tokens_per_char = est as f64 / chars as f64;
        assert!(
            (0.5..=1.0).contains(&tokens_per_char),
            "CJK estimate {est} => {tokens_per_char:.2} tokens/char, want 0.5-1.0"
        );
    }

    /// Mixed English + Chinese should sit between the two regimes, and in
    /// particular the CJK half must pull the total above what pure-Latin
    /// weighting of the same char count would give.
    #[test]
    fn mixed_script_between_regimes() {
        let text = "The model 模型 reads text 文本 as tokens 标记.";
        let est = estimate_tokens(text);
        let chars = text.chars().count();
        assert!(est > chars / 4, "mixed estimate {est} should exceed the pure-Latin chars/4");
        assert!(est < chars, "mixed estimate {est} should stay below 1 token/char");
    }

    #[test]
    fn whitespace_only_is_cheap_but_nonzero_for_runs() {
        assert_eq!(estimate_tokens(""), 0);
        // A single space rounds to zero cost; longer runs accrue.
        assert_eq!(estimate_tokens(" "), 0);
        assert!(estimate_tokens(&" ".repeat(40)) > 0);
    }

    #[test]
    fn digits_denser_than_letters() {
        // Same char count, digits should weigh more than letters.
        let letters = estimate_tokens(&"a".repeat(20));
        let digits = estimate_tokens(&"1".repeat(20));
        assert!(digits > letters, "digits {digits} should exceed letters {letters}");
    }

    #[test]
    fn by_line_numbers_lines_from_one() {
        let text = "alpha beta\ngamma delta epsilon\n";
        let lines = estimate_tokens_by_line(text);
        // "a\nb\n" splits to ["a", "b", ""] -> three entries.
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].line, 1);
        assert_eq!(lines[1].line, 2);
        assert_eq!(lines[2].line, 3);
        assert_eq!(lines[2].tokens, 0, "trailing empty line costs nothing");
        // The busier second line should not be cheaper than the first.
        assert!(lines[1].tokens >= lines[0].tokens);
    }

    #[test]
    fn by_line_empty_input_is_one_zero_line() {
        let lines = estimate_tokens_by_line("");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], LineEstimate { line: 1, tokens: 0 });
    }

    #[test]
    fn by_line_breakdown_roughly_sums_to_total() {
        let text = "one two three\nfour five six seven\neight nine";
        let total = estimate_tokens(text);
        let sum: usize = estimate_tokens_by_line(text).iter().map(|l| l.tokens).sum();
        // Per-line rounding drift is small; within a couple tokens.
        let diff = total.abs_diff(sum);
        assert!(diff <= 2, "per-line sum {sum} vs total {total} drifted by {diff}");
    }
}
