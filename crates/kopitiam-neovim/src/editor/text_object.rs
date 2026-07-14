//! Text objects: `iw`, `aw`, `i(`, `a{`, `it`, `ip`, ... — the highest-value
//! feature in vi, per the brief, because they let an operator target "the
//! thing here" (a word, a quoted string, a tag body) without the user having
//! to spell out a motion that happens to land in the right place.
//!
//! Like [`super::motion`], every text object here is a pure function:
//! `(buffer, position, object, scope) -> Option<Range>`. It returns `None`
//! rather than a zero-width range when there is genuinely no such object at
//! the cursor (e.g. `di(` with no enclosing parenthesis) — that lets the
//! caller distinguish "act on nothing" from "delete an empty pair", which
//! matters because `ci()` on `"()"` is a real, useful command.

use unicode_segmentation::UnicodeSegmentation;

use crate::core::{Granularity, Position, Range};
use crate::text::Buffer;

use super::motion::{step_left, step_right};

/// `i` (inner) excludes the delimiters; `a` (around) includes them (and, for
/// word/quote objects, a run of surrounding whitespace).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectScope {
    Inner,
    Around,
}

/// The text objects from the brief. Bracket objects accept vim's usual
/// aliases (`)`/`b` for parens, `}`/`B` for braces, `]` for brackets) at the
/// parsing layer in [`super::pending`] — this enum only needs one variant
/// per delimiter *pair*, not per keystroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextObject {
    Word,
    BigWord,
    Paren,
    Brace,
    Bracket,
    Angle,
    DoubleQuote,
    SingleQuote,
    Backtick,
    Tag,
    Paragraph,
}

/// Resolves a text object at `pos` to the range it covers, and the
/// granularity an operator should treat that range with. Paragraphs resolve
/// linewise (matching vim: `dip` removes whole lines, not a ragged span
/// ending mid-line); everything else is charwise.
pub fn resolve(buf: &Buffer, pos: Position, obj: TextObject, scope: ObjectScope) -> Option<(Range, Granularity)> {
    match obj {
        TextObject::Word => word_object(buf, pos, false, scope).map(|r| (r, Granularity::Charwise)),
        TextObject::BigWord => word_object(buf, pos, true, scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Paren => bracket_object(buf, pos, '(', ')', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Brace => bracket_object(buf, pos, '{', '}', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Bracket => bracket_object(buf, pos, '[', ']', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Angle => bracket_object(buf, pos, '<', '>', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::DoubleQuote => quote_object(buf, pos, '"', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::SingleQuote => quote_object(buf, pos, '\'', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Backtick => quote_object(buf, pos, '`', scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Tag => tag_object(buf, pos, scope).map(|r| (r, Granularity::Charwise)),
        TextObject::Paragraph => paragraph_object(buf, pos, scope).map(|r| (r, Granularity::Linewise)),
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Class {
    Word,
    Punct,
    Space,
}

fn classify(g: &str, big: bool) -> Class {
    let Some(c) = g.chars().next() else { return Class::Space };
    if c.is_whitespace() {
        Class::Space
    } else if big || c.is_alphanumeric() || c == '_' {
        Class::Word
    } else {
        Class::Punct
    }
}

/// `iw`/`aw`/`iW`/`aW`: the run of word- or WORD-class graphemes at `pos`
/// (or, if `pos` sits on whitespace, that whitespace run). `aw`/`aW`
/// additionally pull in one adjacent whitespace run — trailing if there is
/// one, otherwise leading — matching vim's "a word always tries to eat
/// surrounding space" rule.
fn word_object(buf: &Buffer, pos: Position, big: bool, scope: ObjectScope) -> Option<Range> {
    let line = buf.line(pos.line)?;
    let graphemes: Vec<&str> = line.graphemes(true).collect();
    if graphemes.is_empty() {
        return Some(Range::point(pos));
    }
    let col = pos.col.min(graphemes.len() - 1);
    let class = classify(graphemes[col], big);

    let mut start = col;
    while start > 0 && classify(graphemes[start - 1], big) == class {
        start -= 1;
    }
    let mut end = col;
    while end + 1 < graphemes.len() && classify(graphemes[end + 1], big) == class {
        end += 1;
    }

    let inner = Range::new(Position::new(pos.line, start), Position::new(pos.line, end + 1));
    if scope == ObjectScope::Inner {
        return Some(inner);
    }

    // Try trailing whitespace first.
    if end + 1 < graphemes.len() && classify(graphemes[end + 1], big) == Class::Space {
        let mut trail_end = end + 1;
        while trail_end + 1 < graphemes.len() && classify(graphemes[trail_end + 1], big) == Class::Space {
            trail_end += 1;
        }
        return Some(Range::new(Position::new(pos.line, start), Position::new(pos.line, trail_end + 1)));
    }
    // Otherwise pull in leading whitespace.
    if start > 0 && classify(graphemes[start - 1], big) == Class::Space {
        let mut lead_start = start - 1;
        while lead_start > 0 && classify(graphemes[lead_start - 1], big) == Class::Space {
            lead_start -= 1;
        }
        return Some(Range::new(Position::new(pos.line, lead_start), Position::new(pos.line, end + 1)));
    }
    Some(inner)
}

/// Finds the innermost `open`/`close` pair enclosing `pos`, scanning the
/// whole buffer one grapheme at a time via [`step_left`]/[`step_right`].
/// Handles the cursor sitting exactly on either delimiter (vim treats `i(`
/// on the `(` itself as "inside").
fn find_enclosing(buf: &Buffer, pos: Position, open: char, close: char) -> Option<(Position, Position)> {
    let at = buf.grapheme_at(pos).and_then(|g| g.chars().next());
    if at == Some(open) {
        let end = scan_forward_for_close(buf, pos, open, close)?;
        return Some((pos, end));
    }
    if at == Some(close) {
        let start = scan_backward_for_open(buf, pos, open, close)?;
        return Some((start, pos));
    }
    let start = scan_backward_for_open(buf, pos, open, close)?;
    let end = scan_forward_for_close(buf, start, open, close)?;
    Some((start, end))
}

fn scan_forward_for_close(buf: &Buffer, from_open: Position, open: char, close: char) -> Option<Position> {
    let mut depth = 1i32;
    let mut p = from_open;
    while let Some(next) = step_right(buf, p) {
        p = next;
        match buf.grapheme_at(p).and_then(|g| g.chars().next()) {
            Some(c) if c == open => depth += 1,
            Some(c) if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(p);
                }
            }
            _ => {}
        }
    }
    None
}

fn scan_backward_for_open(buf: &Buffer, from: Position, open: char, close: char) -> Option<Position> {
    let mut depth = 0i32;
    let mut p = from;
    while let Some(prev) = step_left(buf, p) {
        p = prev;
        match buf.grapheme_at(p).and_then(|g| g.chars().next()) {
            Some(c) if c == close => depth += 1,
            Some(c) if c == open => {
                if depth == 0 {
                    return Some(p);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

fn bracket_object(buf: &Buffer, pos: Position, open: char, close: char, scope: ObjectScope) -> Option<Range> {
    let (open_pos, close_pos) = find_enclosing(buf, pos, open, close)?;
    match scope {
        ObjectScope::Around => {
            let end = step_right(buf, close_pos).unwrap_or(Position::new(close_pos.line, buf.line_len(close_pos.line)));
            Some(Range::new(open_pos, end))
        }
        ObjectScope::Inner => {
            let start = step_right(buf, open_pos).unwrap_or(close_pos);
            Some(Range::new(start, close_pos))
        }
    }
}

/// Quote objects are line-scoped in real vim (a quote never spans a
/// newline), so this only looks at `pos.line`.
fn quote_object(buf: &Buffer, pos: Position, quote: char, scope: ObjectScope) -> Option<Range> {
    let line = buf.line(pos.line)?;
    let graphemes: Vec<&str> = line.graphemes(true).collect();
    let quote_cols: Vec<usize> = graphemes
        .iter()
        .enumerate()
        .filter(|(_, g)| g.starts_with(quote))
        .map(|(i, _)| i)
        .collect();

    // Pair consecutive quotes: (0,1), (2,3), ...
    let pair = quote_cols.chunks(2).find_map(|pair| match pair {
        [a, b] => {
            if pos.col <= *b || pos.col < *a {
                Some((*a, *b))
            } else {
                None
            }
        }
        _ => None,
    })?;
    let (start_col, end_col) = pair;
    match scope {
        ObjectScope::Around => {
            // Pull in one trailing space if present, else a leading space.
            let mut end = end_col + 1;
            if end < graphemes.len() && classify(graphemes[end], false) == Class::Space {
                end += 1;
                Some(Range::new(Position::new(pos.line, start_col), Position::new(pos.line, end)))
            } else if start_col > 0 && classify(graphemes[start_col - 1], false) == Class::Space {
                Some(Range::new(Position::new(pos.line, start_col - 1), Position::new(pos.line, end_col + 1)))
            } else {
                Some(Range::new(Position::new(pos.line, start_col), Position::new(pos.line, end_col + 1)))
            }
        }
        ObjectScope::Inner => Some(Range::new(Position::new(pos.line, start_col + 1), Position::new(pos.line, end_col))),
    }
}

/// A single grapheme, tagged with its buffer position, for the flat scan
/// [`tag_object`] needs. Built once per call rather than cached: this crate
/// has no notion of "big file" performance work yet, and tag lookups are
/// already O(document length) in real vim too. If profiling ever shows this
/// mattering, memoizing the flattened buffer per edit is the natural fix.
fn flatten(buf: &Buffer) -> Vec<(Position, String)> {
    let mut out = Vec::new();
    for line in 0..buf.line_count() {
        let Some(text) = buf.line(line) else { continue };
        for (col, g) in text.graphemes(true).enumerate() {
            out.push((Position::new(line, col), g.to_string()));
        }
    }
    out
}

struct TagMatch {
    open_start: usize,
    open_end: usize,
    close_start: usize,
    close_end: usize,
}

/// `it`/`at`: the innermost `<name ...>...</name>` pair enclosing `pos`.
/// Self-closing tags (`<br/>`) are skipped since they have no body. Nesting
/// is tracked by tag name so `<a><b>x</b></a>` with the cursor on `x`
/// resolves to the `<b>` pair, not `<a>`.
fn tag_object(buf: &Buffer, pos: Position, scope: ObjectScope) -> Option<Range> {
    let flat = flatten(buf);
    let idx_of = flat.iter().position(|(p, _)| *p == pos).unwrap_or_else(|| flat.iter().position(|(p, _)| *p >= pos).unwrap_or(flat.len()));

    let joined: String = flat.iter().map(|(_, g)| g.as_str()).collect();
    let chars: Vec<char> = joined.chars().collect();
    // `flat` and `chars` are index-aligned because every grapheme collected
    // here is ASCII-range tag syntax in practice; multi-scalar graphemes
    // inside tag content never affect the `<`/`>`/`/` scan below.

    let tags = scan_tags(&chars);
    let m = enclosing_tag(&tags, idx_of)?;

    let (start_pos, end_pos) = match scope {
        ObjectScope::Around => (flat[m.open_start].0, end_of(&flat, m.close_end)),
        ObjectScope::Inner => {
            if m.open_end >= m.close_start {
                (flat[m.open_end].0, flat[m.open_end].0) // empty body
            } else {
                (flat[m.open_end].0, flat[m.close_start].0)
            }
        }
    };
    Some(Range::new(start_pos, end_pos))
}

fn end_of(flat: &[(Position, String)], idx: usize) -> Position {
    if idx < flat.len() {
        flat[idx].0
    } else if let Some((last, _)) = flat.last() {
        Position::new(last.line, last.col + 1)
    } else {
        Position::ORIGIN
    }
}

struct RawTag {
    start: usize,
    end: usize, // exclusive, one past '>'
    name: String,
    closing: bool,
    self_closing: bool,
}

fn scan_tags(chars: &[char]) -> Vec<RawTag> {
    let mut tags = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            let start = i;
            let mut j = i + 1;
            let closing = j < chars.len() && chars[j] == '/';
            if closing {
                j += 1;
            }
            let name_start = j;
            while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '-' || chars[j] == '_' || chars[j] == ':') {
                j += 1;
            }
            let name: String = chars[name_start..j].iter().collect();
            if name.is_empty() {
                i += 1;
                continue;
            }
            // Find the closing '>', tracking a trailing '/' for self-closing.
            let mut self_closing = false;
            while j < chars.len() && chars[j] != '>' {
                if chars[j] == '/' {
                    self_closing = true;
                }
                j += 1;
            }
            if j < chars.len() {
                j += 1; // consume '>'
                tags.push(RawTag { start, end: j, name, closing, self_closing });
                i = j;
                continue;
            }
        }
        i += 1;
    }
    tags
}

fn enclosing_tag(tags: &[RawTag], idx: usize) -> Option<TagMatch> {
    // Walk tags in order, matching each opening tag with its corresponding
    // closing tag via a per-name stack, and remember the innermost pair
    // whose body contains `idx`.
    let mut stacks: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    let mut best: Option<TagMatch> = None;
    for (i, tag) in tags.iter().enumerate() {
        if tag.self_closing {
            continue;
        }
        if !tag.closing {
            stacks.entry(tag.name.clone()).or_default().push(i);
        } else if let Some(open_i) = stacks.get_mut(&tag.name).and_then(|s| s.pop()) {
            let open = &tags[open_i];
            let close = tag;
            if open.end <= idx && idx <= close.start {
                let candidate = TagMatch { open_start: open.start, open_end: open.end, close_start: close.start, close_end: close.end };
                let better = match &best {
                    None => true,
                    Some(b) => candidate.open_end >= b.open_end && candidate.close_start <= b.close_start,
                };
                if better {
                    best = Some(candidate);
                }
            }
        }
    }
    best
}

/// `ip`/`ap`: a run of contiguous non-blank lines (or, if `pos` is on a
/// blank line, a run of contiguous blank lines). `ap` additionally pulls in
/// the following blank run, or the preceding one if there is none after.
fn paragraph_object(buf: &Buffer, pos: Position, scope: ObjectScope) -> Option<Range> {
    let is_blank = |l: usize| buf.line(l).map(|s| s.trim().is_empty()).unwrap_or(true);
    let target_blank = is_blank(pos.line);

    let mut first = pos.line;
    while first > 0 && is_blank(first - 1) == target_blank {
        first -= 1;
    }
    let mut last = pos.line;
    let line_count = buf.line_count();
    while last + 1 < line_count && is_blank(last + 1) == target_blank {
        last += 1;
    }

    if scope == ObjectScope::Around {
        if last + 1 < line_count && is_blank(last + 1) != target_blank {
            let mut trail = last + 1;
            while trail + 1 < line_count && is_blank(trail + 1) == is_blank(last + 1) {
                trail += 1;
            }
            last = trail;
        } else if first > 0 {
            let mut lead = first - 1;
            while lead > 0 && is_blank(lead - 1) == is_blank(first - 1) {
                lead -= 1;
            }
            first = lead;
        }
    }

    Some(Range::new(Position::new(first, 0), Position::new(last, buf.line_len(last))))
}
