//! Label-jump navigation — the native replacement for hop.nvim.
//!
//! # Why this one is load-bearing
//!
//! The maintainer's `keymaps.lua` rebinds `f` itself (in every mode) to
//! `hop.hint_words`, shadowing vim's built-in find-char-on-line motion. That
//! is not a rarely-used extra — it is how they move within a line dozens of
//! times an hour. Getting the label algorithm wrong is not a papercut, it
//! breaks the editor's primary navigation method for them.
//!
//! # The algorithm
//!
//! 1. Find every word start in the visible lines (a maximal run of
//!    alphanumeric/underscore characters, at the position its first
//!    grapheme begins).
//! 2. Assign each target a label drawn from a small alphabet, home-row
//!    letters first (`asdghklqwertyuiopzxcvbnmfj`, matching hop.nvim's and
//!    easymotion's default) so the common case — a handful of visible words
//!    — never needs more than one keystroke past `f` itself.
//! 3. When there are more targets than letters, some targets need two-letter
//!    labels. The one property that makes a hop-style jump *usable* rather
//!    than maddening is: **no one-character label may be a prefix of a
//!    two-character label.** Without that guarantee, typing the single
//!    character `a` is ambiguous — does it select the target labelled `"a"`,
//!    or start typing the target labelled `"as"`? hop.nvim (and
//!    easymotion/vim-sneak before it) solve this by reserving a subset of
//!    the alphabet purely as *prefixes*: those letters are never handed out
//!    as a standalone label, only as the first character of a two-letter
//!    one. [`assign_labels`] implements exactly that reservation.
//!
//! # Why this doesn't take a cursor position (yet)
//!
//! Real hop.nvim (and easymotion) hand the *shortest* labels to the targets
//! nearest the cursor, on the theory that nearby jumps are more frequent.
//! That requires a cursor [`Position`] as an additional input purely to
//! choose *label assignment order* — it has no bearing on prefix-freedom,
//! which is the property this module is responsible for and the one the
//! test suite is asked to prove. The straightforward extension (sort targets
//! by distance from the cursor before calling [`assign_labels`]) is left to
//! whoever wires this into the editor's motion dispatch, where the cursor
//! position naturally lives; duplicating that state here would just be
//! another thing to keep in sync.

use unicode_segmentation::UnicodeSegmentation;

use crate::core::Position;

/// Home-row-first label alphabet, matching hop.nvim's and easymotion's
/// default. Home row first because those are the fastest keys to reach
/// without moving the hands, and hop labels are meant to be typed as fast as
/// the jump itself.
pub const DEFAULT_ALPHABET: &str = "asdghklqwertyuiopzxcvbnmfj";

/// One labelled jump target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hint {
    pub position: Position,
    pub label: String,
}

/// The result of typing one more character towards a hop label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HopResult {
    /// The typed input uniquely and completely identifies a target — move
    /// the cursor there.
    Jump(Position),
    /// The typed input is a prefix shared by more than one label; here are
    /// the remaining candidates (the UI keeps only their labels highlighted
    /// and waits for the next keystroke).
    Narrow(Vec<Hint>),
    /// The typed input doesn't prefix any label — the user missed, and hop
    /// should cancel.
    NoMatch,
}

/// Finds every word start in `lines`, using `first_line` as the buffer line
/// number of `lines[0]` (i.e. `lines` is already sliced to the visible
/// viewport — this module doesn't know how tall the terminal is, and
/// shouldn't).
///
/// A "word" is a maximal run of alphanumeric/underscore characters, matching
/// vim's `iskeyword` default closely enough for hop's purposes: it is where
/// `w` would land, which is exactly the set of places a user reaches for `f`
/// to jump to.
pub fn hint_words(lines: &[&str], first_line: usize) -> Vec<Hint> {
    hint_words_with_alphabet(lines, first_line, DEFAULT_ALPHABET)
}

/// [`hint_words`] with an explicit label alphabet. Exposed mainly so tests
/// can force the two-character-label path without needing 27+ words of
/// fixture text; production callers should use [`hint_words`].
pub fn hint_words_with_alphabet(lines: &[&str], first_line: usize, alphabet: &str) -> Vec<Hint> {
    let mut targets = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let mut in_word = false;
        for (grapheme_col, grapheme) in line.graphemes(true).enumerate() {
            let is_word_char =
                grapheme.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_');
            if is_word_char && !in_word {
                targets.push(Position::new(first_line + i, grapheme_col));
            }
            in_word = is_word_char;
        }
    }

    let labels = assign_labels(targets.len(), alphabet);
    targets.into_iter().zip(labels).map(|(position, label)| Hint { position, label }).collect()
}

/// Assigns `count` prefix-free labels drawn from `alphabet`, in the order
/// targets were found (the caller pre-sorts targets — by cursor distance,
/// document order, or otherwise — before this is called).
///
/// # The reservation scheme
///
/// Let `a = alphabet.len()`. If `count <= a`, every target gets a distinct
/// single-character label and there is nothing more to do.
///
/// Otherwise, some letters must be reserved as **prefixes**: a prefix letter
/// is never itself handed out as a label, only combined with a second
/// letter (drawn from the *entire* alphabet, prefixes included — a
/// two-character label's second character can safely reuse any letter,
/// because a one-character label `"x"` is never a prefix of a two-character
/// label unless that label also starts with `x`, and prefix letters are by
/// definition excluded from standalone use). With `k` reserved prefixes:
///
/// * `a - k` targets get single-character labels (the first `a - k`
///   targets, so they stay the shortest — i.e. fastest to type).
/// * the remaining targets get two-character labels, at a capacity of
///   `k * a` (each of the `k` prefixes combined with all `a` second
///   characters).
///
/// `k` is the smallest value for which `(a - k) + k * a >= count` — found by
/// a linear scan over `k` from 1 to `a` (at most 26 iterations for the
/// default alphabet; not worth a closed form). This is the same shape of
/// reservation hop.nvim and vim-easymotion use.
///
/// # Capacity limit
///
/// Two-character labels top out at `a * a` targets (all prefixes reserved,
/// `k = a`). If `count` exceeds that, this returns only `a * a` labels
/// rather than inventing three-character ones — [`hint_words`] then simply
/// leaves the excess targets unlabelled, the same way hop.nvim declines to
/// hint every word on a screen too dense to label unambiguously.
pub fn assign_labels(count: usize, alphabet: &str) -> Vec<String> {
    let letters: Vec<char> = alphabet.chars().collect();
    let a = letters.len();
    if a == 0 || count == 0 {
        return Vec::new();
    }

    if count <= a {
        return letters[..count].iter().map(|c| c.to_string()).collect();
    }

    let mut k = 1usize;
    while k < a && (a - k) + k * a < count {
        k += 1;
    }

    let standalone = a - k;
    let mut labels: Vec<String> = letters[..standalone].iter().map(|c| c.to_string()).collect();

    let prefixes = &letters[standalone..];
    'outer: for &prefix in prefixes {
        for &second in &letters {
            if labels.len() == count {
                break 'outer;
            }
            labels.push(format!("{prefix}{second}"));
        }
    }
    labels
}

/// Resolves accumulated typed input (`input`) against the candidate `hints`
/// produced by [`hint_words`]. The UI calls this after every keystroke,
/// passing the *whole* input typed so far — resolution is stateless because
/// the candidate set (`hints`) doesn't change mid-hop, so there is nothing
/// to gain from mutating in place.
pub fn resolve(hints: &[Hint], input: &str) -> HopResult {
    if input.is_empty() {
        return HopResult::Narrow(hints.to_vec());
    }

    let matching: Vec<&Hint> = hints.iter().filter(|h| h.label.starts_with(input)).collect();

    // Prefix-freedom (see the module docs) guarantees that if an exact match
    // exists, it is the *only* match — a one-character label can never also
    // be a prefix of some other label. Checking for it explicitly rather
    // than relying on `matching.len() == 1` keeps this correct even if that
    // invariant were ever violated by a future alphabet change: it fails
    // safe (jumps on the unambiguous case) rather than silently narrowing
    // forever.
    if let Some(exact) = matching.iter().find(|h| h.label == input) {
        return HopResult::Jump(exact.position);
    }

    if matching.is_empty() {
        HopResult::NoMatch
    } else {
        HopResult::Narrow(matching.into_iter().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_word_starts_on_one_line() {
        let hints = hint_words(&["let foo = bar_baz + 1;"], 0);
        let cols: Vec<usize> = hints.iter().map(|h| h.position.col).collect();
        // "let" @0, "foo" @4, "bar_baz" @10, "1" @20.
        assert_eq!(cols, vec![0, 4, 10, 20]);
    }

    #[test]
    fn positions_use_the_viewports_first_line_offset() {
        let hints = hint_words(&["alpha", "beta"], 40);
        assert_eq!(hints[0].position, Position::new(40, 0));
        assert_eq!(hints[1].position, Position::new(41, 0));
    }

    #[test]
    fn labels_are_prefix_free_within_the_alphabet() {
        // 26-letter default alphabet, force > 26 targets by repeating a
        // trivially tokenizable line.
        let line = "a ".repeat(60);
        let hints = hint_words(&[line.as_str()], 0);
        assert!(hints.len() > DEFAULT_ALPHABET.len());
        assert_prefix_free(&hints);
    }

    #[test]
    fn overflow_produces_valid_two_character_labels() {
        // A 3-letter alphabet forces the two-character path after the 3rd
        // target, and has a hard capacity of 3*3 = 9 — pick a target count
        // inside that so every target actually gets labelled.
        let alphabet = "abc";
        let words: Vec<&str> = std::iter::repeat_n("w ", 8).collect();
        let line: String = words.concat();
        let hints = hint_words_with_alphabet(&[line.as_str()], 0, alphabet);
        assert_eq!(hints.len(), 8);
        assert!(hints.iter().any(|h| h.label.chars().count() == 2));
        assert_prefix_free(&hints);
        // Every label is built only from the given alphabet.
        for hint in &hints {
            assert!(hint.label.chars().all(|c| alphabet.contains(c)));
        }
    }

    #[test]
    fn assign_labels_degrades_gracefully_past_alphabet_capacity() {
        // A 2-letter alphabet can address at most 2*2 = 4 targets with
        // one-or-two-character labels. Asking for more must not panic; it
        // should simply return as many labels as fit.
        let labels = assign_labels(10, "ab");
        assert_eq!(labels.len(), 4);
        assert_prefix_free_labels(&labels);
    }

    #[test]
    fn every_assigned_label_is_unique() {
        let labels = assign_labels(200, DEFAULT_ALPHABET);
        assert_eq!(labels.len(), 200);
        let mut sorted = labels.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 200, "duplicate label assigned");
    }

    #[test]
    fn resolve_jumps_on_a_single_character_label() {
        let hints = vec![
            Hint { position: Position::new(0, 0), label: "a".into() },
            Hint { position: Position::new(0, 5), label: "sa".into() },
        ];
        assert_eq!(resolve(&hints, "a"), HopResult::Jump(Position::new(0, 0)));
    }

    #[test]
    fn resolve_narrows_then_jumps_on_a_two_character_label() {
        let hints = vec![
            Hint { position: Position::new(0, 0), label: "sa".into() },
            Hint { position: Position::new(0, 5), label: "sd".into() },
        ];
        // Typing "s" alone doesn't (and, being a reserved prefix, can't)
        // match anything exactly, so it narrows.
        match resolve(&hints, "s") {
            HopResult::Narrow(candidates) => assert_eq!(candidates.len(), 2),
            other => panic!("expected Narrow, got {other:?}"),
        }
        // Typing the second character resolves the jump.
        assert_eq!(resolve(&hints, "sd"), HopResult::Jump(Position::new(0, 5)));
    }

    #[test]
    fn resolve_reports_no_match_for_input_matching_no_label() {
        let hints = vec![Hint { position: Position::new(0, 0), label: "a".into() }];
        assert_eq!(resolve(&hints, "z"), HopResult::NoMatch);
    }

    #[test]
    fn empty_viewport_hints_nothing() {
        assert!(hint_words(&[], 0).is_empty());
        assert!(hint_words(&[""], 0).is_empty());
    }

    fn assert_prefix_free(hints: &[Hint]) {
        let labels: Vec<String> = hints.iter().map(|h| h.label.clone()).collect();
        assert_prefix_free_labels(&labels);
    }

    fn assert_prefix_free_labels(labels: &[String]) {
        let one_char: Vec<&str> =
            labels.iter().map(String::as_str).filter(|l| l.chars().count() == 1).collect();
        let two_char: Vec<&str> =
            labels.iter().map(String::as_str).filter(|l| l.chars().count() == 2).collect();
        for short in &one_char {
            for long in &two_char {
                assert!(
                    !long.starts_with(short),
                    "one-char label {short:?} is a prefix of two-char label {long:?}"
                );
            }
        }
    }
}
