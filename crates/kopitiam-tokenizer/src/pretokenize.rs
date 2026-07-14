//! GPT-2/Qwen-style pre-tokenization: splitting text into chunks (words,
//! numbers, punctuation runs, contractions, whitespace runs) before BPE
//! merges ever run.
//!
//! Pre-tokenization exists so that BPE never merges *across* these
//! boundaries -- "dog." and "dog!" share the "dog" subword, and a merge
//! that fused a word with trailing punctuation would fragment the vocab
//! for no linguistic benefit. Getting this step wrong is uniquely
//! dangerous: a wrong chunk boundary still round-trips perfectly under
//! `decode(encode(x)) == x` (every byte is still accounted for, just
//! grouped differently), so the bug is invisible to the one test everyone
//! remembers to write, and only shows up as the model receiving different
//! token ids than it was trained on.
//!
//! # The lookahead problem
//!
//! The canonical GPT-2 pattern (see
//! <https://github.com/openai/gpt-2/blob/master/src/encoder.py#L98>) is:
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! The second-to-last alternative, `\s+(?!\S)`, uses a negative lookahead,
//! which the `regex` crate deliberately does not support (lookaround
//! requires backtracking, which is exactly what `regex` avoids in order to
//! guarantee linear-time matching). Simply dropping the lookahead and
//! keeping `\s+` changes behavior: for a multi-space run in the interior of
//! the text (e.g. `"a   b"`), the true pattern reserves the *last* space of
//! the run to prefix the following word (`" b"`, matched by ` ?\p{L}+`,
//! which the alternation tries before falling through to a whitespace
//! alternative), leaving a shorter whitespace-only chunk (`"  "`) before it.
//! A bare `\s+` alternative has no such carve-out and would greedily
//! swallow the entire run, shifting every subsequent chunk boundary by one
//! character relative to the reference tokenizer.
//!
//! ## The fix
//!
//! [`split`] runs the pattern with `\s+(?!\S)|\s+` collapsed to plain
//! `\s+`, then does a cheap post-pass: **any chunk that is made up
//! entirely of whitespace and is not the last chunk in the text must have
//! come from that trailing `\s+` alternative** (every other alternative
//! requires at least one non-whitespace character to match at all, so a
//! chunk made of nothing but whitespace could not have come from them).
//! Such a chunk gets its last character peeled off and handled specially,
//! reproducing `\s+(?!\S)` exactly. Note that "` ?`" in the word/number/
//! punctuation alternatives matches a *literal ASCII space* only -- not
//! `\s` in general -- so the peeled-off character's fate depends on what it
//! actually is:
//!
//! * A whitespace run at the very end of the string has no "next chunk" at
//!   all, so it is left alone -- matching `(?!\S)` being trivially
//!   satisfied by end-of-input.
//! * If the peeled-off character is a space (`' '`), it is folded onto the
//!   *front* of the following chunk, exactly like the classic GPT-2 `Ġ`
//!   prefix: `"a   b"` -> `["a", "  ", " b"]`.
//! * If the peeled-off character is any other whitespace (tab, newline,
//!   ...), it cannot be absorbed by `" ?"` at all, so it becomes a
//!   standalone one-character chunk of its own instead:
//!   `"a\t\tb"` -> `["a", "\t", "\t", "b"]`, `"a\nb"` -> `["a", "\n", "b"]`.
//!
//! This is easy to get backwards (assuming *any* trailing whitespace
//! character folds forward like a space does) and the mistake is
//! invisible to casual testing since decode still round-trips -- see
//! `tests::newline_runs_do_not_fold_into_the_next_word` below, which is
//! cross-checked against CPython's `regex` module (which does support
//! `(?!\S)` natively) to make sure this crate's workaround is not just
//! internally consistent but actually matches the reference pattern.

use regex::Regex;
use std::sync::LazyLock;

/// The GPT-2 pre-tokenization pattern with the unsupported trailing
/// `(?!\S)` lookahead removed. See the module docs for why this is safe:
/// the fix-up pass in [`split`] restores the original behavior.
static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .expect("pre-tokenization pattern is a fixed, tested constant")
});

/// Splits `text` into pre-tokenization chunks, returned as byte offsets
/// into `text` so callers can slice without an extra allocation per chunk.
pub fn split(text: &str) -> Vec<(usize, usize)> {
    let raw: Vec<(usize, usize)> = PATTERN
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();

    let mut out = Vec::with_capacity(raw.len());
    // When a peeled-off trailing space is folded onto the next chunk (see
    // the module docs), this carries the overridden start offset for that
    // next chunk instead of mutating `raw` in place -- a whitespace chunk
    // can never itself be the *target* of a fold (two adjacent `\s+`
    // matches never occur, since the pattern always consumes a maximal
    // whitespace run in one match), so this only ever needs to survive one
    // loop iteration.
    let mut pending_start: Option<usize> = None;

    for (i, &(start, end)) in raw.iter().enumerate() {
        let effective_start = pending_start.take().unwrap_or(start);
        let is_last = i + 1 == raw.len();
        let chunk = &text[start..end];

        if is_last || !is_all_whitespace(chunk) {
            out.push((effective_start, end));
            continue;
        }

        // `chunk` came from the bare `\s+` alternative (see module docs
        // for why that is the only way to get an all-whitespace match).
        // Peel off its last character; `char_indices` (not byte slicing)
        // keeps this correct for multi-byte whitespace like U+00A0.
        let (last_char_offset, last_char) = chunk
            .char_indices()
            .last()
            .expect("is_all_whitespace guarantees `chunk` is non-empty");
        let split_at = start + last_char_offset;

        if split_at > effective_start {
            out.push((effective_start, split_at));
        }
        if last_char == ' ' {
            pending_start = Some(split_at);
        } else {
            out.push((split_at, end));
        }
    }

    out
}

fn is_all_whitespace(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(text: &str) -> Vec<&str> {
        split(text).into_iter().map(|(s, e)| &text[s..e]).collect()
    }

    #[test]
    fn splits_words_and_punctuation() {
        assert_eq!(chunks("Hello, world!"), vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn single_space_attaches_to_the_following_word() {
        // The classic GPT-2 "Ġ" behavior: one leading space becomes part
        // of the word chunk, never its own whitespace chunk.
        assert_eq!(chunks("a b"), vec!["a", " b"]);
    }

    #[test]
    fn multi_space_run_holds_back_exactly_one_trailing_space() {
        // "a" + "  " (2 of the 3 spaces) + " b" (last space + word).
        assert_eq!(chunks("a   b"), vec!["a", "  ", " b"]);
    }

    #[test]
    fn trailing_whitespace_at_end_of_string_is_not_split() {
        assert_eq!(chunks("a   "), vec!["a", "   "]);
    }

    #[test]
    fn contractions_are_kept_whole() {
        assert_eq!(chunks("don't"), vec!["don", "'t"]);
    }

    #[test]
    fn numbers_and_letters_split_apart() {
        assert_eq!(chunks("room101"), vec!["room", "101"]);
    }

    /// A single space folds forward onto the next word, but a single
    /// non-space whitespace character (here, a newline) does *not* --
    /// `" ?"` in the word alternatives only ever matches a literal space.
    /// Getting this backwards is exactly the trap described in the module
    /// docs: `["a", "\nb"]` looks plausible and still round-trips under
    /// `decode(encode(x)) == x`, but is not what the reference tokenizer
    /// produces. Cross-checked against CPython's `regex` module running
    /// the *real* lookahead pattern (see the sibling
    /// `newline_runs_do_not_fold_into_the_next_word` test below for the
    /// three-newline case, which pins down the same rule for N >= 2).
    #[test]
    fn lone_newline_before_a_word_does_not_fold_forward() {
        assert_eq!(chunks("a\nb"), vec!["a", "\n", "b"]);
    }

    #[test]
    fn newline_runs_do_not_fold_into_the_next_word() {
        // Verified against `re.compile(r" ?[A-Za-z]+|\s+(?!\S)|\s+")`
        // (CPython's stdlib `re`, which does not support `\p{L}`/`\p{N}`
        // but does support the lookahead we care about here) run over
        // "a\n\n\nb": ['a', '\n\n', '\n', 'b'].
        assert_eq!(chunks("a\n\n\nb"), vec!["a", "\n\n", "\n", "b"]);
    }

    #[test]
    fn tab_run_splits_into_two_standalone_chunks() {
        // Verified the same way: "a\t\tb" -> ['a', '\t', '\t', 'b'].
        assert_eq!(chunks("a\t\tb"), vec!["a", "\t", "\t", "b"]);
    }

    #[test]
    fn mixed_run_ending_in_space_still_folds_the_final_space_forward() {
        // "a \n b" -> ['a', ' \n', ' b']: only the *last* character of the
        // run decides the fold, regardless of what precedes it.
        assert_eq!(chunks("a \n b"), vec!["a", " \n", " b"]);
    }

    #[test]
    fn empty_string_has_no_chunks() {
        assert!(chunks("").is_empty());
    }

    #[test]
    fn cjk_and_emoji_are_treated_as_letter_runs_or_punctuation() {
        // CJK ideographs are \p{L}; most emoji are \p{So} (Symbol, other),
        // which falls into the "everything else" punctuation-ish
        // alternative. Either way every byte must still show up in some
        // chunk -- exercised properly by the round-trip tests in
        // `tests/round_trip.rs`; this just checks nothing panics or drops
        // a chunk outright.
        let joined: String = chunks("你好 🎉 world").concat();
        assert_eq!(joined, "你好 🎉 world");
    }
}
