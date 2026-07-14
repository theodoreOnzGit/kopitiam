//! Pattern search: `/`, `?`, `n`, `N`, `*`, `#`.
//!
//! # Why line-by-line rather than one rope-wide regex
//!
//! kvim's cursor is addressed in `(line, grapheme column)` ([`Position`]),
//! never in byte offsets — that is the whole point of [`crate::core`]'s
//! addressing model (see its module docs). A regex, though, matches over a
//! `&str` and reports **byte** offsets. Searching the entire buffer as one
//! string would mean converting an arbitrary byte offset back into a
//! `(line, grapheme)` pair, which is exactly the rope arithmetic
//! [`crate::text::Buffer`] exists to keep out of every other module. Scanning
//! one line at a time keeps the byte→grapheme conversion local and cheap: it
//! only ever spans a single line's text, so a plain grapheme count of the
//! matched prefix is enough.
//!
//! The trade-off is that a pattern cannot match across a line break. Vim's
//! own `/` can (with `\n` in the pattern); kvim's cannot yet. That is a
//! documented scope cut, not an oversight — the overwhelming majority of
//! interactive searches are single-line, and lifting the limit later is a
//! change to this one module, not to the cursor model.

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

use crate::core::Position;
use crate::text::Buffer;

/// The grapheme column at which byte offset `byte` falls within `line`.
fn byte_to_col(line: &str, byte: usize) -> usize {
    line.grapheme_indices(true).take_while(|(b, _)| *b < byte).count()
}

/// Finds the next match of `pattern` starting strictly after `from` when
/// `forward`, or strictly before it when searching backward, wrapping around
/// the buffer once. Returns the match's start [`Position`], or `None` if the
/// pattern is invalid or matches nowhere.
///
/// "Strictly after/before the cursor" is what makes `n` advance rather than
/// re-finding the match the cursor is already sitting on; the wrap is what
/// makes a search from the last match cycle back to the first, exactly as
/// vim's `/` does (kvim does not implement `'wrapscan' off`).
pub fn find(buf: &Buffer, from: Position, pattern: &str, forward: bool) -> Option<Position> {
    let re = Regex::new(pattern).ok()?;
    let line_count = buf.line_count();
    if forward {
        find_forward(buf, &re, from, line_count)
    } else {
        find_backward(buf, &re, from, line_count)
    }
}

fn find_forward(buf: &Buffer, re: &Regex, from: Position, line_count: usize) -> Option<Position> {
    // Two passes: from the cursor line to the end, then wrapping from the top
    // back to (and including) the cursor line, so a match earlier on the
    // start line is still reachable on the wrap.
    for pass in 0..2 {
        let (first, last) = if pass == 0 { (from.line, line_count) } else { (0, from.line + 1) };
        for line in first..last.min(line_count) {
            let Some(text) = buf.line(line) else { continue };
            for m in re.find_iter(&text) {
                let col = byte_to_col(&text, m.start());
                let pos = Position::new(line, col);
                let after_cursor = line != from.line || col > from.col;
                let on_wrap = pass == 1 && (line < from.line || (line == from.line && col <= from.col));
                if after_cursor || on_wrap {
                    return Some(pos);
                }
            }
        }
    }
    None
}

fn find_backward(buf: &Buffer, re: &Regex, from: Position, line_count: usize) -> Option<Position> {
    // Mirror of `find_forward`: walk lines from the cursor upward, taking the
    // last match before the cursor on each line, then wrap from the bottom.
    let last_line = line_count.saturating_sub(1);
    for pass in 0..2 {
        let lines: Vec<usize> = if pass == 0 {
            (0..=from.line.min(last_line)).rev().collect()
        } else {
            (from.line..line_count).rev().collect()
        };
        for line in lines {
            let Some(text) = buf.line(line) else { continue };
            let mut best: Option<usize> = None;
            for m in re.find_iter(&text) {
                let col = byte_to_col(&text, m.start());
                let before_cursor = line != from.line || col < from.col;
                let on_wrap = pass == 1 && (line > from.line || (line == from.line && col >= from.col));
                if before_cursor || on_wrap {
                    best = Some(col); // keep the last (rightmost) qualifying match
                }
            }
            if let Some(col) = best {
                return Some(Position::new(line, col));
            }
        }
    }
    None
}

/// The keyword under (or after) the cursor on its line, for `*`/`#`. A
/// keyword is a maximal run of alphanumeric/underscore graphemes, matching
/// vim's default `iskeyword`. Returns `None` if there is no keyword at or
/// after the cursor on the line.
pub fn word_under_cursor(buf: &Buffer, pos: Position) -> Option<String> {
    let text = buf.line(pos.line)?;
    let graphemes: Vec<&str> = text.graphemes(true).collect();
    let is_word = |g: &str| g.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_');

    // Start at the cursor; if it is not on a word char, scan forward to the
    // next one on the same line (vim's behaviour when `*` is pressed off a
    // word).
    let mut start = pos.col.min(graphemes.len());
    while start < graphemes.len() && !is_word(graphemes[start]) {
        start += 1;
    }
    if start >= graphemes.len() {
        return None;
    }
    // Walk back to the run's beginning, then forward to its end.
    while start > 0 && is_word(graphemes[start - 1]) {
        start -= 1;
    }
    let mut end = start;
    while end < graphemes.len() && is_word(graphemes[end]) {
        end += 1;
    }
    Some(graphemes[start..end].concat())
}

/// Builds the anchored pattern `*`/`#` search with — `\bword\b` — with the
/// word's regex metacharacters escaped so a keyword like `a.b` (not that
/// `.` is a keyword char, but defensively) is matched literally.
pub fn word_pattern(word: &str) -> String {
    format!(r"\b{}\b", regex::escape(word))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_search_finds_the_next_match_after_the_cursor() {
        let buf = Buffer::from_str("foo bar foo baz");
        let p = find(&buf, Position::new(0, 0), "foo", true).unwrap();
        assert_eq!(p, Position::new(0, 8), "must skip the match the cursor is on");
    }

    #[test]
    fn forward_search_wraps_to_the_top() {
        let buf = Buffer::from_str("foo\nbar\nbaz");
        let p = find(&buf, Position::new(2, 0), "foo", true).unwrap();
        assert_eq!(p, Position::new(0, 0), "wraps around the end of the buffer");
    }

    #[test]
    fn backward_search_finds_the_previous_match() {
        let buf = Buffer::from_str("foo bar foo baz");
        // From col 12 ("baz"), the nearest match starting before the cursor is
        // the second "foo" at col 8 — not the first, which is further back.
        let p = find(&buf, Position::new(0, 12), "foo", false).unwrap();
        assert_eq!(p, Position::new(0, 8));
        // From the start of the second match, the previous one is the first.
        let p = find(&buf, Position::new(0, 8), "foo", false).unwrap();
        assert_eq!(p, Position::new(0, 0));
    }

    #[test]
    fn backward_search_wraps_to_the_bottom() {
        let buf = Buffer::from_str("foo\nbar\nfoo");
        let p = find(&buf, Position::new(0, 0), "foo", false).unwrap();
        assert_eq!(p, Position::new(2, 0));
    }

    #[test]
    fn word_under_cursor_reads_the_whole_keyword() {
        let buf = Buffer::from_str("let bar_baz = 1");
        assert_eq!(word_under_cursor(&buf, Position::new(0, 5)).as_deref(), Some("bar_baz"));
    }

    #[test]
    fn word_under_cursor_scans_forward_when_off_a_word() {
        let buf = Buffer::from_str("  hi");
        assert_eq!(word_under_cursor(&buf, Position::new(0, 0)).as_deref(), Some("hi"));
    }

    #[test]
    fn byte_to_col_counts_graphemes_not_bytes() {
        // "中" is 3 bytes, 1 grapheme; a match at byte 3 is at column 1.
        let line = "中x";
        assert_eq!(byte_to_col(line, 3), 1);
    }
}
