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

use regex::{Regex, RegexBuilder};
use unicode_segmentation::UnicodeSegmentation;

use crate::core::Position;
use crate::text::Buffer;

/// The grapheme column at which byte offset `byte` falls within `line`.
fn byte_to_col(line: &str, byte: usize) -> usize {
    line.grapheme_indices(true).take_while(|(b, _)| *b < byte).count()
}

/// Compile `pattern` into a [`Regex`], applying vim's `'ignorecase'` /
/// `'smartcase'` case rules, or `None` if the pattern is not a valid regex.
///
/// This is the one place case-folding is decided, so the motion (`n`/`N`/`/`)
/// and the search-match *highlight* (hlsearch) can never disagree about which
/// text match — both of them funnel through here. Vim's rule:
///
/// * `ignorecase` off → case-sensitive, always.
/// * `ignorecase` on, `smartcase` off → case-insensitive, always.
/// * `ignorecase` on **and** `smartcase` on → case-insensitive *only while the
///   pattern is all-lowercase*; the moment the user type an uppercase letter,
///   the search snap back to case-sensitive. That one is what make `/foo` match
///   `Foo` but `/Foo` match only `Foo`.
pub fn build_regex(pattern: &str, ignorecase: bool, smartcase: bool) -> Option<Regex> {
    let has_upper = pattern.chars().any(|c| c.is_uppercase());
    let case_insensitive = ignorecase && !(smartcase && has_upper);
    RegexBuilder::new(pattern).case_insensitive(case_insensitive).build().ok()
}

/// Every match of `re` on `line`, as inclusive-start / exclusive-end **grapheme
/// column** ranges (`(start, end)`), left to right.
///
/// Grapheme columns, not byte offsets, because that one is the coordinate kvim's
/// cursor and its renderer both speak (see this module header on why search work
/// line-by-line). The renderer convert these to *display* columns with the
/// line's text in hand — a tab is one grapheme and several cells — same like it
/// does for a visual selection.
///
/// Zero-width matches kena dropped: a pattern like `x*` match the empty string
/// between characters, and got nothing on screen to paint for that one. Keeping
/// them also can make `find_iter` no progress; skip them so the highlight only
/// cover spans a user can actually see.
pub fn line_match_cols(line: &str, re: &Regex) -> Vec<(usize, usize)> {
    re.find_iter(line)
        .filter(|m| m.start() != m.end())
        .map(|m| (byte_to_col(line, m.start()), byte_to_col(line, m.end())))
        .collect()
}

/// Finds the next match of `pattern` starting strictly after `from` when
/// `forward`, or strictly before it when searching backward, wrapping around
/// the buffer once. Returns the match's start [`Position`], or `None` if the
/// pattern is invalid or matches nowhere.
///
/// `ignorecase`/`smartcase` are threaded straight to [`build_regex`], so the
/// jump and the highlight share one notion of what match.
///
/// "Strictly after/before the cursor" is what makes `n` advance rather than
/// re-finding the match the cursor is already sitting on; the wrap is what
/// makes a search from the last match cycle back to the first, exactly as
/// vim's `/` does (kvim does not implement `'wrapscan' off`).
pub fn find(buf: &Buffer, from: Position, pattern: &str, forward: bool, ignorecase: bool, smartcase: bool) -> Option<Position> {
    let re = build_regex(pattern, ignorecase, smartcase)?;
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

/// The `g*`/`g#` counterpart to [`word_pattern`]: the escaped keyword with
/// **no** `\b` word boundaries, so the search matches the keyword as a
/// substring (`g*` on `foo` also lands on `foobar`). That missing anchor is
/// the whole difference between `*` and `g*`.
pub fn word_pattern_loose(word: &str) -> String {
    regex::escape(word)
}

/// The filename under (or after) the cursor on its line, for `gf`. A filename
/// grapheme is one of vim's default `isfname` set: alphanumerics plus the
/// path/URL punctuation `/._-~+#$%@` (kvim keeps it deliberately small — no
/// spaces, no shell metacharacters — so a stray word in prose does not read as
/// a path). Returns `None` when there is no such run at or after the cursor on
/// the line. Mirrors [`word_under_cursor`]'s scan-forward-when-off-a-token
/// behaviour so pressing `gf` with the cursor in the indent still finds the
/// name.
pub fn file_under_cursor(buf: &Buffer, pos: Position) -> Option<String> {
    let text = buf.line(pos.line)?;
    let graphemes: Vec<&str> = text.graphemes(true).collect();
    let is_fname = |g: &str| g.chars().next().is_some_and(|c| c.is_alphanumeric() || "/._-~+#$%@".contains(c));

    let mut start = pos.col.min(graphemes.len());
    while start < graphemes.len() && !is_fname(graphemes[start]) {
        start += 1;
    }
    if start >= graphemes.len() {
        return None;
    }
    while start > 0 && is_fname(graphemes[start - 1]) {
        start -= 1;
    }
    let mut end = start;
    while end < graphemes.len() && is_fname(graphemes[end]) {
        end += 1;
    }
    Some(graphemes[start..end].concat())
}

/// The match of `pattern` that `gn`/`gN` should select, as an inclusive-start /
/// exclusive-end grapheme-column [`Position`] pair (`(start, end)`), both on the
/// same line (kvim searches never cross a newline — see the module header).
///
/// vim's rule, and ours: if the cursor is sitting *on* a match, that match is
/// the one selected; otherwise move to the next one (`forward`) or the previous
/// one (`gN`). The wrap around the buffer works exactly like [`find`], so `gn`
/// at the last match cycles to the first.
///
/// This exists separately from [`find`] because `gn` needs the match's *extent*
/// (to paint / operate over the whole thing), whereas `n` only needs where to
/// put the cursor. `case_insensitive`/`smartcase` thread through [`build_regex`]
/// so a `gn` selection can never drift from what `n` would jump to.
pub fn match_range(buf: &Buffer, from: Position, pattern: &str, forward: bool, ignorecase: bool, smartcase: bool) -> Option<(Position, Position)> {
    let re = build_regex(pattern, ignorecase, smartcase)?;
    let line_count = buf.line_count();
    if forward {
        match_range_forward(buf, &re, from, line_count)
    } else {
        match_range_backward(buf, &re, from, line_count)
    }
}

fn match_range_forward(buf: &Buffer, re: &Regex, from: Position, line_count: usize) -> Option<(Position, Position)> {
    for pass in 0..2 {
        let (first, last) = if pass == 0 { (from.line, line_count) } else { (0, from.line + 1) };
        for line in first..last.min(line_count) {
            let Some(text) = buf.line(line) else { continue };
            for (start, end) in line_match_cols(&text, re) {
                // A match qualifies when it reaches past the cursor: on the
                // cursor line that means `end > cursor.col`, which is true both
                // when the cursor sits inside the match (select it) and when the
                // match starts after the cursor (move to it). On any other line,
                // or on the wrap, every match qualifies.
                let qualifies = line != from.line || end > from.col;
                let on_wrap = pass == 1 && (line < from.line || (line == from.line && start <= from.col));
                if (pass == 0 && qualifies) || on_wrap {
                    return Some((Position::new(line, start), Position::new(line, end)));
                }
            }
        }
    }
    None
}

fn match_range_backward(buf: &Buffer, re: &Regex, from: Position, line_count: usize) -> Option<(Position, Position)> {
    let last_line = line_count.saturating_sub(1);
    for pass in 0..2 {
        let lines: Vec<usize> = if pass == 0 { (0..=from.line.min(last_line)).rev().collect() } else { (from.line..line_count).rev().collect() };
        for line in lines {
            let Some(text) = buf.line(line) else { continue };
            let mut best: Option<(usize, usize)> = None;
            for (start, end) in line_match_cols(&text, re) {
                // Rightmost match starting at or before the cursor on the cursor
                // line (covers both "cursor inside the match" and "match before
                // the cursor"); every match on other lines / the wrap.
                let qualifies = line != from.line || start <= from.col;
                let on_wrap = pass == 1 && (line > from.line || (line == from.line && start >= from.col));
                if (pass == 0 && qualifies) || on_wrap {
                    best = Some((start, end));
                }
            }
            if let Some((start, end)) = best {
                return Some((Position::new(line, start), Position::new(line, end)));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_search_finds_the_next_match_after_the_cursor() {
        let buf = Buffer::from_str("foo bar foo baz");
        let p = find(&buf, Position::new(0, 0), "foo", true, false, false).unwrap();
        assert_eq!(p, Position::new(0, 8), "must skip the match the cursor is on");
    }

    #[test]
    fn forward_search_wraps_to_the_top() {
        let buf = Buffer::from_str("foo\nbar\nbaz");
        let p = find(&buf, Position::new(2, 0), "foo", true, false, false).unwrap();
        assert_eq!(p, Position::new(0, 0), "wraps around the end of the buffer");
    }

    #[test]
    fn backward_search_finds_the_previous_match() {
        let buf = Buffer::from_str("foo bar foo baz");
        // From col 12 ("baz"), the nearest match starting before the cursor is
        // the second "foo" at col 8 — not the first, which is further back.
        let p = find(&buf, Position::new(0, 12), "foo", false, false, false).unwrap();
        assert_eq!(p, Position::new(0, 8));
        // From the start of the second match, the previous one is the first.
        let p = find(&buf, Position::new(0, 8), "foo", false, false, false).unwrap();
        assert_eq!(p, Position::new(0, 0));
    }

    #[test]
    fn backward_search_wraps_to_the_bottom() {
        let buf = Buffer::from_str("foo\nbar\nfoo");
        let p = find(&buf, Position::new(0, 0), "foo", false, false, false).unwrap();
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

    #[test]
    fn line_match_cols_reports_every_match_in_grapheme_columns() {
        let re = build_regex("foo", false, false).unwrap();
        // Two matches: cols 0..3 and 8..11.
        assert_eq!(line_match_cols("foo bar foo baz", &re), vec![(0, 3), (8, 11)]);
    }

    #[test]
    fn line_match_cols_counts_graphemes_not_bytes() {
        let re = build_regex("x", false, false).unwrap();
        // "中" is 3 bytes, 1 grapheme, so the "x" after it is at grapheme col 1.
        assert_eq!(line_match_cols("中x中x", &re), vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn line_match_cols_drops_zero_width_matches() {
        // `o*` matches the empty string at every gap; only the real "oo" run
        // should be painted.
        let re = build_regex("o*", false, false).unwrap();
        assert_eq!(line_match_cols("foo", &re), vec![(1, 3)]);
    }

    #[test]
    fn ignorecase_folds_case_only_when_set() {
        // Case-sensitive by default: "foo" does not match "Foo".
        assert!(!build_regex("foo", false, false).unwrap().is_match("Foo"));
        // ignorecase on: it does.
        assert!(build_regex("foo", true, false).unwrap().is_match("Foo"));
    }

    #[test]
    fn word_pattern_loose_drops_the_word_boundaries() {
        assert_eq!(word_pattern("foo"), r"\bfoo\b");
        assert_eq!(word_pattern_loose("foo"), "foo");
        // The loose pattern matches inside a longer word; the anchored one does not.
        assert!(build_regex(&word_pattern_loose("foo"), false, false).unwrap().is_match("foobar"));
        assert!(!build_regex(&word_pattern("foo"), false, false).unwrap().is_match("foobar"));
    }

    #[test]
    fn file_under_cursor_reads_a_path_like_token() {
        let buf = Buffer::from_str("see src/main.rs here");
        assert_eq!(file_under_cursor(&buf, Position::new(0, 4)).as_deref(), Some("src/main.rs"));
        // Off a token, it scans forward to the next one on the line.
        assert_eq!(file_under_cursor(&buf, Position::new(0, 3)).as_deref(), Some("src/main.rs"));
    }

    #[test]
    fn match_range_selects_the_match_under_the_cursor_then_the_next() {
        let buf = Buffer::from_str("foo foo foo");
        // Cursor inside the middle match selects that whole match, not the next.
        let (s, e) = match_range(&buf, Position::new(0, 5), "foo", true, false, false).unwrap();
        assert_eq!((s, e), (Position::new(0, 4), Position::new(0, 7)));
        // Cursor before a match (in the gap) moves forward to it.
        let (s, e) = match_range(&buf, Position::new(0, 3), "foo", true, false, false).unwrap();
        assert_eq!((s, e), (Position::new(0, 4), Position::new(0, 7)));
    }

    #[test]
    fn match_range_backward_takes_the_previous_match() {
        let buf = Buffer::from_str("foo foo foo");
        let (s, e) = match_range(&buf, Position::new(0, 8), "foo", false, false, false).unwrap();
        // From inside the last match, gN selects that same match.
        assert_eq!((s, e), (Position::new(0, 8), Position::new(0, 11)));
        // From the gap before it, gN steps back to the middle match.
        let (s, e) = match_range(&buf, Position::new(0, 7), "foo", false, false, false).unwrap();
        assert_eq!((s, e), (Position::new(0, 4), Position::new(0, 7)));
    }

    #[test]
    fn smartcase_stays_sensitive_once_the_pattern_has_an_uppercase() {
        // ignorecase + smartcase: an all-lowercase pattern folds case...
        assert!(build_regex("foo", true, true).unwrap().is_match("FOO"));
        // ...but the moment the pattern carries an uppercase letter it does not.
        assert!(!build_regex("Foo", true, true).unwrap().is_match("foo"));
        assert!(build_regex("Foo", true, true).unwrap().is_match("Foo"));
    }
}
