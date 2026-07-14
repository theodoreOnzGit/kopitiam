//! Grapheme cluster <-> rope offset conversion.
//!
//! This is the one place in the text engine allowed to know that a
//! [`Position::col`](crate::Position) is a grapheme index while a
//! [`ropey::Rope`] indexes by `char`. Every other module in [`text`](super)
//! calls through here rather than re-deriving the conversion, so there is
//! exactly one place that can get CJK/emoji/combining-mark handling wrong.
//!
//! # Why per-line, not whole-rope
//!
//! Grapheme segmentation here only ever runs over a single **line's**
//! content, materialized as a plain `&str`. `ropey`'s chunks are not
//! guaranteed to split on grapheme-cluster boundaries (only on `char`
//! boundaries), so segmenting a `RopeSlice` correctly in general would mean
//! driving `unicode_segmentation::GraphemeCursor` incrementally across chunk
//! boundaries and handling its `Incomplete` results. That complexity buys
//! nothing here: a [`Position::col`](crate::Position) is inherently
//! line-scoped — nothing in `Buffer`'s API asks "what's the four-millionth
//! grapheme of the whole file" — and a line is bounded by how much text a
//! human can usefully look at, even in a pathological minified-JS
//! single-line file. [`super::buffer::Buffer`] hands this module an owned
//! `String` for one line at a time, which keeps the hard problem (grapheme
//! correctness) testable in complete isolation from the rope. If a workload
//! with truly enormous single lines ever shows up in practice, revisit with
//! `GraphemeCursor` then — the call sites are exactly the four functions
//! below, so the blast radius of that change is small.

use unicode_segmentation::UnicodeSegmentation;

/// Number of grapheme clusters in `line`.
pub(crate) fn grapheme_len(line: &str) -> usize {
    line.graphemes(true).count()
}

/// The grapheme cluster at index `col` in `line`, or `None` if `col` is at
/// or past the end of the line (there is no cluster *at* end-of-line, only
/// a valid cursor position for one — see [`col_to_char`]).
pub(crate) fn grapheme_at(line: &str, col: usize) -> Option<&str> {
    line.graphemes(true).nth(col)
}

/// Byte offset in `line` where grapheme cluster `col` begins.
///
/// `col == grapheme_len(line)` (one past the last cluster) is a valid,
/// deliberately supported case: it is "end of line", the position a cursor
/// sits at after `$` or after appending with `A`. That case, and anything
/// further out, maps to `line.len()`.
fn col_to_byte(line: &str, col: usize) -> usize {
    line.grapheme_indices(true).nth(col).map(|(b, _)| b).unwrap_or(line.len())
}

/// Inverse of [`col_to_byte`]: the grapheme column whose cluster starts at
/// byte offset `byte`, or that immediately follows it if `byte` lands
/// mid-cluster (which should not happen for offsets this module produces,
/// but a mid-cluster byte offset must still resolve to *something* rather
/// than panic).
fn byte_to_col(line: &str, byte: usize) -> usize {
    line.grapheme_indices(true).take_while(|(b, _)| *b < byte).count()
}

/// Char offset within `line` corresponding to byte offset `byte`.
fn byte_to_char(line: &str, byte: usize) -> usize {
    line[..byte.min(line.len())].chars().count()
}

/// Byte offset within `line` corresponding to char offset `ch`.
fn char_to_byte(line: &str, ch: usize) -> usize {
    line.char_indices().nth(ch).map(|(b, _)| b).unwrap_or(line.len())
}

/// Grapheme column `col` in `line` -> `char` offset within that line (i.e.
/// within `ropey`'s indexing unit, but still relative to the start of the
/// line, not the whole rope).
pub(crate) fn col_to_char(line: &str, col: usize) -> usize {
    byte_to_char(line, col_to_byte(line, col))
}

/// `char` offset within `line` -> the grapheme column containing it (or
/// immediately following it, for a mid-cluster offset).
pub(crate) fn char_to_col(line: &str, ch: usize) -> usize {
    byte_to_col(line, char_to_byte(line, ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_counts_one_column_per_character() {
        let line = "日本語";
        assert_eq!(grapheme_len(line), 3);
        assert_eq!(grapheme_at(line, 0), Some("日"));
        assert_eq!(grapheme_at(line, 1), Some("本"));
        assert_eq!(grapheme_at(line, 2), Some("語"));
        assert_eq!(grapheme_at(line, 3), None);
    }

    #[test]
    fn zwj_emoji_family_is_one_grapheme_cluster() {
        let line = "👨\u{200D}👩\u{200D}👧"; // family: man, ZWJ, woman, ZWJ, girl
        assert_eq!(grapheme_len(line), 1);
        assert_eq!(grapheme_at(line, 0), Some(line));
        assert_eq!(grapheme_at(line, 1), None);
    }

    #[test]
    fn combining_acute_accent_joins_its_base_letter() {
        let line = "e\u{0301}"; // 'e' followed by COMBINING ACUTE ACCENT
        assert_eq!(grapheme_len(line), 1);
        assert_eq!(grapheme_at(line, 0), Some(line));
    }

    #[test]
    fn mixed_line_counts_clusters_not_chars_or_bytes() {
        // "a" (1 byte, 1 char) + CJK char (3 bytes, 1 char) + ZWJ family (many
        // bytes, many chars) + combining-mark 'e' (2 bytes, 2 chars) = 4
        // grapheme clusters total, despite wildly different byte/char counts.
        let line = "a語👨\u{200D}👩\u{200D}👧e\u{0301}";
        assert_eq!(grapheme_len(line), 4);
    }

    #[test]
    fn col_to_char_and_back_round_trip_at_every_boundary() {
        let line = "a語👨\u{200D}👩\u{200D}👧e\u{0301}";
        let n = grapheme_len(line);
        for col in 0..=n {
            let ch = col_to_char(line, col);
            assert_eq!(char_to_col(line, ch), col, "round trip failed at col {col}");
        }
    }

    #[test]
    fn end_of_line_column_is_supported_and_clamped() {
        let line = "abc";
        assert_eq!(col_to_char(line, 3), 3); // exactly at end
        assert_eq!(col_to_char(line, 100), 3); // past end clamps to end
        assert_eq!(grapheme_at(line, 3), None);
    }

    #[test]
    fn empty_line_has_zero_length_and_no_graphemes() {
        assert_eq!(grapheme_len(""), 0);
        assert_eq!(grapheme_at("", 0), None);
        assert_eq!(col_to_char("", 0), 0);
        assert_eq!(char_to_col("", 0), 0);
    }
}
