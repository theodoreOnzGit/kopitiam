//! Delimiter alignment — the native replacement for vim-easy-align's `ga`.
//!
//! # Scope
//!
//! vim-easy-align supports interactive mode switching, Nth-occurrence
//! selection, left/right/center alignment and per-line overrides — an
//! entire configuration language triggered by extra keystrokes after `ga`.
//! This module implements the 90% case the maintainer's config actually
//! exposes a keymap for: align a block of lines on the *first* occurrence of
//! a delimiter, left-justified (padding goes before the delimiter, matching
//! the `key = value` shape the docstring's own test table uses). Extending
//! to Nth-occurrence or right-alignment is a matter of adding parameters to
//! [`align`], not a redesign — deliberately not built until something
//! actually needs it (see CLAUDE.md's "avoid unnecessary abstraction").
//!
//! # Why this returns `Vec<Edit>` instead of mutating anything
//!
//! Every plugin module in [`crate::plugins`] is headless: it has no buffer
//! to mutate, no undo tree to push onto, and no LSP `didChange` to notify.
//! Only the caller (the editor's operator dispatch) has all three. Returning
//! [`Edit`]s keeps `align` pure and trivially testable, and means an
//! interrupted or rejected alignment (e.g. the user immediately undoes it)
//! is just an unappplied `Edit` — there's no engine-side state to roll back.

use unicode_segmentation::UnicodeSegmentation;

use crate::core::{Edit, Position, Range};

/// What to align on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    /// A specific character, e.g. `=`, `:`, `|`, `,`.
    Char(char),
    /// The first run of whitespace on the line — for aligning
    /// already-whitespace-separated columns (`vim-easy-align`'s `ga<space>`
    /// / `Tabularize /\s\+`).
    Whitespace,
}

/// Aligns `lines` (a contiguous slice of buffer lines, starting at buffer
/// line `first_line`) on `delimiter`.
///
/// Only the text *before* the delimiter is padded — the delimiter itself and
/// everything after it is left untouched. That is what makes this
/// idempotent: a line whose pre-delimiter text is already the target width
/// round-trips to an identical string, which means no [`Edit`] is emitted
/// for it at all. Lines that don't contain the delimiter are skipped
/// entirely (not counted towards the target width, not edited) — that is
/// what vim-easy-align does with a heterogeneous block (e.g. a comment line
/// mixed in with `key = value` lines), rather than mangling it.
pub fn align(lines: &[&str], first_line: usize, delimiter: Delimiter) -> Vec<Edit> {
    // First pass: split each line (that contains the delimiter) into the
    // part before it (right-trimmed of existing padding) and the part from
    // the delimiter onward, verbatim.
    let split: Vec<Option<(&str, &str)>> = lines.iter().map(|line| split_on(line, delimiter)).collect();

    let target_col = split
        .iter()
        .flatten()
        .map(|(before, _)| before.graphemes(true).count())
        .max();

    let Some(target_col) = target_col else {
        return Vec::new();
    };
    // One space of separation before the delimiter, matching vim-easy-align's
    // default padding.
    let target_col = target_col + 1;

    lines
        .iter()
        .zip(split)
        .enumerate()
        .filter_map(|(i, (&line, parts))| {
            let (before, from_delim) = parts?;
            let pad = target_col - before.graphemes(true).count();
            let new_line = format!("{before}{}{from_delim}", " ".repeat(pad));
            if new_line == line {
                return None; // Already aligned — idempotent, no-op.
            }
            let old_len = line.graphemes(true).count();
            let abs_line = first_line + i;
            Some(Edit::replace(
                Range::new(Position::new(abs_line, 0), Position::new(abs_line, old_len)),
                new_line,
            ))
        })
        .collect()
}

/// Splits `line` at the first occurrence of `delimiter`, returning
/// `(before, from_delimiter_onward)` with `before` right-trimmed of
/// whitespace. `None` if the delimiter isn't present on this line at all.
fn split_on(line: &str, delimiter: Delimiter) -> Option<(&str, &str)> {
    match delimiter {
        Delimiter::Char(c) => {
            let idx = line.find(c)?;
            Some((line[..idx].trim_end(), &line[idx..]))
        }
        Delimiter::Whitespace => {
            let idx = line.find(char::is_whitespace)?;
            let (before, rest) = line.split_at(idx);
            Some((before, rest.trim_start()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_key_value_lines_on_equals() {
        let lines = ["foo=1", "barbaz=2", "x=3"];
        let edits = align(&lines, 10, Delimiter::Char('='));
        assert_eq!(edits.len(), 3);
        assert_eq!(edits[0].text, "foo    =1");
        assert_eq!(edits[1].text, "barbaz =2");
        assert_eq!(edits[2].text, "x      =3");
        // Buffer-absolute line numbers, not 0-relative to the slice.
        assert_eq!(edits[0].range.anchor, Position::new(10, 0));
        assert_eq!(edits[2].range.anchor, Position::new(12, 0));
    }

    #[test]
    fn already_aligned_input_is_unchanged() {
        let aligned = ["foo    =1", "barbaz =2", "x      =3"];
        let edits = align(&aligned, 0, Delimiter::Char('='));
        assert!(edits.is_empty(), "re-aligning already-aligned input must be a no-op");
    }

    #[test]
    fn a_line_without_the_delimiter_is_left_alone() {
        let lines = ["foo=1", "// a comment, no equals sign here", "barbaz=2"];
        let edits = align(&lines, 0, Delimiter::Char('='));
        // Only the two lines that actually have '=' are touched.
        assert_eq!(edits.len(), 2);
        assert!(edits.iter().all(|e| e.range.anchor.line != 1));
    }

    #[test]
    fn no_delimiter_anywhere_produces_no_edits() {
        let lines = ["one", "two", "three"];
        assert!(align(&lines, 0, Delimiter::Char('=')).is_empty());
    }

    #[test]
    fn aligns_on_whitespace() {
        let lines = ["a bb", "ccc d"];
        let edits = align(&lines, 0, Delimiter::Whitespace);
        // Widest first column is "ccc" (3 graphemes) -> target column 4.
        // "ccc d" already sits at that column, so it's left as a no-op edit
        // (idempotence — see `already_aligned_input_is_unchanged`) while
        // "a bb" needs padding out to match.
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].range.anchor, Position::new(0, 0));
        assert_eq!(edits[0].text, "a   bb");
    }

    #[test]
    fn aligns_on_an_arbitrary_character() {
        let lines = ["a|1", "bb|2"];
        let edits = align(&lines, 0, Delimiter::Char('|'));
        // Widest pre-delimiter text is "bb" (2 graphemes) -> target column 3.
        assert_eq!(edits[0].text, "a  |1");
        assert_eq!(edits[1].text, "bb |2");
    }
}
