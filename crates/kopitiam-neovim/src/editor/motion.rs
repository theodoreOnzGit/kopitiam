//! Motions: pure functions from `(buffer, position, count)` to a new
//! `position`, plus the datum that makes operators compose correctly.
//!
//! # Why `MotionKind` lives on the motion, not the operator
//!
//! `dw` deletes a different span than `de`, even though both are "`d` plus a
//! word motion". The difference is not a property of `d` — `d` always
//! deletes "the range the motion described". It is a property of `w` versus
//! `e`: `w` lands *after* the word it crossed (its landing grapheme does not
//! belong to what was crossed, so an operator must exclude it) while `e`
//! lands *on* the last grapheme of the word (which does belong, so an
//! operator must include it). Vim calls these "exclusive" and "inclusive"
//! motions. A third kind, "linewise", says the motion was never about
//! columns at all (`dd`, `dG`, `d}`... no — `}` is exclusive; `dG`, `dgg`,
//! `dj` are linewise).
//!
//! Encoding this on [`Motion::kind`] means every operator ([`super::operator`])
//! is a single generic "act on this range" function. Nothing in this crate
//! special-cases "if the operator is `d` and the motion is `w`" — that kind
//! of match is exactly the pile-of-special-cases this engine is built to
//! avoid.

use crate::core::Position;
use crate::text::Buffer;

/// How a motion's landing position combines with its start to form an
/// operator's range. See the module docs for why this exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionKind {
    /// The landing grapheme is *not* part of what the motion crossed.
    /// `dw` on `"foo bar"` from `f` deletes `"foo "`, not `"foo b"`.
    Exclusive,
    /// The landing grapheme *is* part of what the motion crossed.
    /// `de` on `"foo bar"` from `f` deletes `"foo"`, including the `o`.
    Inclusive,
    /// The motion is not about columns: it always spans whole lines.
    Linewise,
}

/// Which direction `;`/`,` should repeat the last `f`/`F`/`t`/`T` in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindKind {
    /// `f{c}`: land on the next occurrence of `c`.
    To,
    /// `F{c}`: land on the previous occurrence of `c`.
    ToBack,
    /// `t{c}`: land just before the next occurrence of `c`.
    Till,
    /// `T{c}`: land just after the previous occurrence of `c`.
    TillBack,
}

impl FindKind {
    fn is_forward(self) -> bool {
        matches!(self, Self::To | Self::Till)
    }

    /// The kind `,` should use to repeat this one backwards.
    fn reversed(self) -> Self {
        match self {
            Self::To => Self::ToBack,
            Self::ToBack => Self::To,
            Self::Till => Self::TillBack,
            Self::TillBack => Self::Till,
        }
    }
}

/// A motion: something that can move the cursor, and — when an operator is
/// pending — describes a range for that operator to act on.
///
/// `f`/`F`/`t`/`T` carry their target character; `;`/`,` are resolved to a
/// concrete [`Motion::FindChar`] by the caller (see
/// [`super::Editor::last_find`]) rather than existing as their own variant,
/// so that `Motion::apply` never needs access to "what did the user search
/// for three commands ago" — that memory lives in `Editor`, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    Left,
    Right,
    Up,
    Down,
    WordForward,
    WordForwardBig,
    WordBackward,
    WordBackwardBig,
    WordEnd,
    WordEndBig,
    /// `ge`: end of the *previous* word (vim's backward word-end motion,
    /// included for completeness since [`Motion::WordEnd`]'s inverse is
    /// otherwise inexpressible).
    WordEndBack,
    /// `gE`: end of the previous WORD (whitespace-delimited).
    WordEndBackBig,
    LineStart,
    FirstNonBlank,
    /// `g_`: the last non-blank character of the line.
    LastNonBlank,
    LineEnd,
    /// `gg`, or `{count}gg`/`{count}G` for "go to line count".
    FileStart,
    /// `G`, or `{count}G` for "go to line count".
    FileEnd,
    ParagraphForward,
    ParagraphBackward,
    /// `)`: start of the next sentence. Distinct from [`Motion::ParagraphForward`]
    /// (`}`) — vim has always kept sentence and paragraph motions separate,
    /// and the brief's motion list includes both `( )` and `{ }`.
    SentenceForward,
    /// `(`: start of the current/previous sentence.
    SentenceBackward,
    MatchPair,
    /// `H`: top of the buffer/viewport. This crate has no notion of a
    /// terminal viewport (that belongs to `ui`), so `H`/`M`/`L` approximate
    /// against the whole buffer. Faithful viewport-relative behaviour is the
    /// UI layer's job to layer on top, if it chooses to.
    ScreenHigh,
    ScreenMid,
    ScreenLow,
    FindChar { kind: FindKind, target: char },
}

impl Motion {
    /// Whether this motion, combined with an operator, is exclusive,
    /// inclusive or linewise. See the module docs.
    pub fn kind(self) -> MotionKind {
        use Motion::*;
        match self {
            Left | Right | WordForward | WordForwardBig | WordBackward | WordBackwardBig | LineStart | FirstNonBlank
            | ParagraphForward | ParagraphBackward | SentenceForward | SentenceBackward => MotionKind::Exclusive,
            WordEnd | WordEndBig | WordEndBack | WordEndBackBig | LastNonBlank | LineEnd | MatchPair => MotionKind::Inclusive,
            Up | Down | FileStart | FileEnd | ScreenHigh | ScreenMid | ScreenLow => {
                MotionKind::Linewise
            }
            FindChar { kind, .. } => match kind {
                FindKind::To | FindKind::ToBack => MotionKind::Inclusive,
                FindKind::Till | FindKind::TillBack => MotionKind::Inclusive,
            },
        }
    }

    /// Whether this motion can serve as the target of `f`/`F`/`t`/`T` repeat
    /// (`;`/`,`) bookkeeping, i.e. whether it *is* a find-motion.
    pub fn find_kind(self) -> Option<(FindKind, char)> {
        match self {
            Motion::FindChar { kind, target } => Some((kind, target)),
            _ => None,
        }
    }

    /// Applies this motion, returning the landing position.
    ///
    /// `count` is `None` when the user typed no digits at all, and `Some`
    /// otherwise — already the resolved, multiplied count (`2d3w` passes
    /// `Some(6)` to `WordForward`, not `Some(3)`; see [`super::pending`] for
    /// where counts are combined). The distinction matters for
    /// [`Motion::FileEnd`]: bare `G` and `1G` are different commands (last
    /// line vs. first line), so "count was omitted" cannot be collapsed into
    /// "count is 1" the way every other motion safely does.
    pub fn apply(self, buf: &Buffer, pos: Position, count: Option<usize>) -> Position {
        use Motion::*;
        if let FileEnd = self {
            let last = buf.line_count().saturating_sub(1);
            let target = match count {
                Some(n) => n.saturating_sub(1).min(last),
                None => last,
            };
            return Position::new(target, first_non_blank(buf, target));
        }
        let count = count.unwrap_or(1).max(1);
        match self {
            Left => buf.clamp(Position::new(pos.line, pos.col.saturating_sub(count))),
            Right => {
                let len = buf.line_len(pos.line);
                let max = len.saturating_sub(1).max(pos.col);
                Position::new(pos.line, (pos.col + count).min(max.max(pos.col)))
            }
            Up => {
                let line = pos.line.saturating_sub(count);
                buf.clamp(Position::new(line, pos.col))
            }
            Down => {
                let line = (pos.line + count).min(buf.line_count().saturating_sub(1));
                buf.clamp(Position::new(line, pos.col))
            }
            WordForward => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_forward(buf, p, false);
                }
                p
            }
            WordForwardBig => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_forward(buf, p, true);
                }
                p
            }
            WordBackward => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_backward(buf, p, false);
                }
                p
            }
            WordBackwardBig => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_backward(buf, p, true);
                }
                p
            }
            WordEnd => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_end(buf, p, false);
                }
                p
            }
            WordEndBig => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_end(buf, p, true);
                }
                p
            }
            WordEndBack => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_end_back(buf, p, false);
                }
                p
            }
            WordEndBackBig => {
                let mut p = pos;
                for _ in 0..count {
                    p = word_end_back(buf, p, true);
                }
                p
            }
            LineStart => Position::new(pos.line, 0),
            FirstNonBlank => Position::new(pos.line, first_non_blank(buf, pos.line)),
            LastNonBlank => Position::new(pos.line, last_non_blank(buf, pos.line)),
            LineEnd => {
                let len = buf.line_len(pos.line);
                Position::new(pos.line, len.saturating_sub(1))
            }
            FileStart => {
                let target = count.saturating_sub(1).min(buf.line_count().saturating_sub(1));
                Position::new(target, first_non_blank(buf, target))
            }
            FileEnd => unreachable!("handled above before `count` was defaulted"),
            ParagraphForward => {
                let mut p = pos;
                for _ in 0..count {
                    p = paragraph_forward(buf, p);
                }
                p
            }
            ParagraphBackward => {
                let mut p = pos;
                for _ in 0..count {
                    p = paragraph_backward(buf, p);
                }
                p
            }
            SentenceForward => {
                let mut p = pos;
                for _ in 0..count {
                    p = sentence_forward(buf, p);
                }
                p
            }
            SentenceBackward => {
                let mut p = pos;
                for _ in 0..count {
                    p = sentence_backward(buf, p);
                }
                p
            }
            MatchPair => match_pair(buf, pos).unwrap_or(pos),
            ScreenHigh => Position::new(0, first_non_blank(buf, 0)),
            ScreenMid => {
                let mid = (buf.line_count().saturating_sub(1)) / 2;
                Position::new(mid, first_non_blank(buf, mid))
            }
            ScreenLow => {
                let last = buf.line_count().saturating_sub(1);
                Position::new(last, first_non_blank(buf, last))
            }
            FindChar { kind, target } => {
                let mut p = pos;
                for _ in 0..count {
                    match find_char(buf, p, kind, target) {
                        Some(next) => p = next,
                        None => break,
                    }
                }
                p
            }
        }
    }
}

/// Vim's three lexical classes for word motions. Whitespace always
/// terminates a run; word-chars and punctuation are each their own run so
/// that `foo(bar)` is three words (`foo`, `(`, `bar`, `)` — four, actually),
/// not one.
#[derive(PartialEq, Eq, Clone, Copy)]
enum CharClass {
    Word,
    Punct,
    Space,
}

fn classify(g: &str, big: bool) -> CharClass {
    let Some(c) = g.chars().next() else { return CharClass::Space };
    if c.is_whitespace() {
        CharClass::Space
    } else if big {
        // A WORD (capital-W word) is any non-blank run: word-chars and
        // punctuation are not distinguished.
        CharClass::Word
    } else if c.is_alphanumeric() || c == '_' {
        CharClass::Word
    } else {
        CharClass::Punct
    }
}

fn grapheme_class(buf: &Buffer, pos: Position, big: bool) -> Option<CharClass> {
    buf.grapheme_at(pos).map(|g| classify(&g, big))
}

/// One grapheme to the right, crossing to the next line's column 0 at
/// end-of-line. Returns `None` at the very end of the buffer.
///
/// `pub(crate)` because [`super::operator`] and [`super::text_object`] need
/// the exact same "step across a line boundary" logic to build charwise
/// ranges (e.g. turning an inclusive motion's landing grapheme into a
/// half-open range end) — duplicating it there would be exactly the kind of
/// special-casing this crate is trying to avoid.
pub(crate) fn step_right(buf: &Buffer, pos: Position) -> Option<Position> {
    let len = buf.line_len(pos.line);
    if pos.col + 1 < len {
        Some(Position::new(pos.line, pos.col + 1))
    } else if pos.line + 1 < buf.line_count() {
        Some(Position::new(pos.line + 1, 0))
    } else {
        None
    }
}

/// One grapheme to the left, crossing to the previous line's last column.
/// See [`step_right`] for why this is `pub(crate)`.
pub(crate) fn step_left(buf: &Buffer, pos: Position) -> Option<Position> {
    if pos.col > 0 {
        Some(Position::new(pos.line, pos.col - 1))
    } else if pos.line > 0 {
        let prev = pos.line - 1;
        let len = buf.line_len(prev);
        Some(Position::new(prev, len.saturating_sub(1)))
    } else {
        None
    }
}

fn first_non_blank(buf: &Buffer, line: usize) -> usize {
    let Some(text) = buf.line(line) else { return 0 };
    for (i, g) in text.graphemes(true).enumerate() {
        if classify(g, false) != CharClass::Space {
            return i;
        }
    }
    0
}

fn word_forward(buf: &Buffer, pos: Position, big: bool) -> Position {
    let mut p = pos;
    let start_class = grapheme_class(buf, p, big);
    // An empty line counts as its own "word" for the purposes of `w`, so
    // that `w` over a blank line stops there rather than skipping it.
    if buf.line_len(p.line) == 0 {
        if let Some(next) = step_right(buf, Position::new(p.line, 0)) {
            return next;
        }
        return p;
    }
    if let Some(class) = start_class {
        if class != CharClass::Space {
            // Skip the rest of the current run.
            while let Some(next) = step_right(buf, p) {
                if buf.line_len(next.line) == 0 {
                    return next; // landed on a blank line: that's the word.
                }
                if grapheme_class(buf, next, big) != Some(class) {
                    p = next;
                    break;
                }
                p = next;
            }
        } else {
            match step_right(buf, p) {
                Some(next) => p = next,
                None => return p,
            }
        }
    }
    // Skip whitespace up to the next non-blank (or a blank line, which is
    // itself a stopping point).
    loop {
        if buf.line_len(p.line) == 0 {
            return p;
        }
        match grapheme_class(buf, p, big) {
            Some(CharClass::Space) => match step_right(buf, p) {
                Some(next) => p = next,
                None => return p,
            },
            _ => return p,
        }
    }
}

fn word_backward(buf: &Buffer, pos: Position, big: bool) -> Position {
    let Some(mut p) = step_left(buf, pos) else { return pos };
    // Skip whitespace backwards.
    loop {
        if buf.line_len(p.line) == 0 {
            return p;
        }
        match grapheme_class(buf, p, big) {
            Some(CharClass::Space) => match step_left(buf, p) {
                Some(prev) => p = prev,
                None => return p,
            },
            _ => break,
        }
    }
    // Walk to the start of this run.
    let class = grapheme_class(buf, p, big);
    loop {
        let Some(prev) = step_left(buf, p) else { return p };
        if buf.line_len(prev.line) == 0 {
            return p;
        }
        if grapheme_class(buf, prev, big) != class {
            return p;
        }
        p = prev;
    }
}

fn word_end(buf: &Buffer, pos: Position, big: bool) -> Position {
    let Some(mut p) = step_right(buf, pos) else { return pos };
    loop {
        if buf.line_len(p.line) == 0 {
            // A blank line has no "end"; keep going past it like whitespace.
            match step_right(buf, p) {
                Some(next) => {
                    p = next;
                    continue;
                }
                None => return p,
            }
        }
        match grapheme_class(buf, p, big) {
            Some(CharClass::Space) => match step_right(buf, p) {
                Some(next) => p = next,
                None => return p,
            },
            _ => break,
        }
    }
    let class = grapheme_class(buf, p, big);
    loop {
        let Some(next) = step_right(buf, p) else { return p };
        if buf.line_len(next.line) == 0 || grapheme_class(buf, next, big) != class {
            return p;
        }
        p = next;
    }
}

fn word_end_back(buf: &Buffer, pos: Position, big: bool) -> Position {
    // `ge`/`gE`: the end of the *previous* word/WORD. Mirror of `word_end`:
    // step left, skip out of the word the cursor started in (if any), skip
    // whitespace, and land on the first non-blank — which, approaching from
    // the right, is the end of the previous word.
    let Some(mut p) = step_left(buf, pos) else { return pos };
    // Phase 1: if the cursor started inside a word, walk left past the rest of
    // that same word first, so `ge` from the middle/end of a word lands on the
    // *previous* word's end, not an earlier character of the current one.
    if let Some(start_class) = grapheme_class(buf, pos, big)
        && start_class != CharClass::Space
    {
        while buf.line_len(p.line) != 0 && grapheme_class(buf, p, big) == Some(start_class) {
            match step_left(buf, p) {
                Some(prev) => p = prev,
                None => return p,
            }
        }
    }
    // Phase 2: skip whitespace; the first non-blank is the previous word's end.
    loop {
        if buf.line_len(p.line) == 0 {
            return p;
        }
        match grapheme_class(buf, p, big) {
            Some(CharClass::Space) => match step_left(buf, p) {
                Some(prev) => p = prev,
                None => return p,
            },
            _ => return p,
        }
    }
}

/// The position `<C-w>` (delete-word-back) in Insert mode should delete back
/// to: the start of the word (or whitespace-then-word) immediately before the
/// cursor, never crossing a line boundary. `pub(crate)` so the editor's
/// Insert-mode handler can reuse it.
pub(crate) fn word_back_for_delete(buf: &Buffer, pos: Position) -> Position {
    if pos.col == 0 {
        return pos;
    }
    let text = buf.line(pos.line).unwrap_or_default();
    let graphemes: Vec<&str> = text.graphemes(true).collect();
    let mut i = pos.col.min(graphemes.len());
    // Skip whitespace immediately before the cursor.
    while i > 0 && graphemes[i - 1].chars().next().map(char::is_whitespace).unwrap_or(false) {
        i -= 1;
    }
    if i == 0 {
        return Position::new(pos.line, 0);
    }
    // Then delete the run of one lexical class (word chars, or punctuation).
    let class = classify(graphemes[i - 1], false);
    while i > 0 && classify(graphemes[i - 1], false) == class {
        i -= 1;
    }
    Position::new(pos.line, i)
}

/// The column of the last non-blank grapheme on `line` (`g_`), or 0 for a
/// blank/empty line.
fn last_non_blank(buf: &Buffer, line: usize) -> usize {
    let Some(text) = buf.line(line) else { return 0 };
    let mut last = 0;
    for (i, g) in text.graphemes(true).enumerate() {
        if classify(g, false) != CharClass::Space {
            last = i;
        }
    }
    last
}

fn paragraph_forward(buf: &Buffer, pos: Position) -> Position {
    let mut line = pos.line;
    let is_blank = |l: usize| buf.line(l).map(|s| s.trim().is_empty()).unwrap_or(true);
    // Skip past the current blank run, if any, then find the next blank line.
    while line + 1 < buf.line_count() && is_blank(line) {
        line += 1;
    }
    while line + 1 < buf.line_count() && !is_blank(line + 1) {
        line += 1;
    }
    if line + 1 < buf.line_count() {
        line += 1;
        Position::new(line, 0)
    } else {
        let last = buf.line_count().saturating_sub(1);
        Position::new(last, buf.line_len(last).saturating_sub(1))
    }
}

fn paragraph_backward(buf: &Buffer, pos: Position) -> Position {
    let is_blank = |l: usize| buf.line(l).map(|s| s.trim().is_empty()).unwrap_or(true);
    if pos.line == 0 {
        return Position::new(0, 0);
    }
    let mut line = pos.line - 1;
    while line > 0 && is_blank(line) {
        line -= 1;
    }
    while line > 0 && !is_blank(line - 1) {
        line -= 1;
    }
    Position::new(line, 0)
}

/// `true` if the grapheme at `pos` is `.`/`!`/`?` and is followed (after any
/// closing quotes/brackets) by whitespace or the end of the buffer — vim's
/// definition of "this ends a sentence".
fn is_sentence_end(buf: &Buffer, pos: Position) -> bool {
    let Some(c) = buf.grapheme_at(pos).and_then(|g| g.chars().next()) else { return false };
    if !matches!(c, '.' | '!' | '?') {
        return false;
    }
    let mut p = pos;
    loop {
        let Some(next) = step_right(buf, p) else { return true };
        match buf.grapheme_at(next).and_then(|g| g.chars().next()) {
            Some(')' | ']' | '"' | '\'') => p = next,
            Some(c2) if c2.is_whitespace() => return true,
            None => return true,
            _ => return false,
        }
    }
}

/// Every sentence's starting position in the buffer, in document order.
/// Sentence motions (`(`/`)`) are implemented by scanning this list rather
/// than walking incrementally, trading O(document length) per keystroke for
/// a single, unified definition of "where sentences start" shared by both
/// directions — the same trade-off [`super::text_object`]'s tag object
/// makes, and for the same reason (this crate has no "large file" fast path
/// yet; see that module's docs).
fn sentence_starts(buf: &Buffer) -> Vec<Position> {
    let mut starts = vec![Position::ORIGIN];
    let mut p = Position::ORIGIN;
    while let Some(next) = step_right(buf, p) {
        p = next;
        if buf.line_len(p.line) == 0 {
            starts.push(p);
            continue;
        }
        if is_sentence_end(buf, p) {
            let mut q = p;
            while let Some(n) = step_right(buf, q) {
                if matches!(buf.grapheme_at(n).and_then(|g| g.chars().next()), Some(')') | Some(']') | Some('"') | Some('\'')) {
                    q = n;
                } else {
                    break;
                }
            }
            while let Some(n) = step_right(buf, q) {
                if buf.grapheme_at(n).and_then(|g| g.chars().next()).map(char::is_whitespace).unwrap_or(false) {
                    q = n;
                } else {
                    break;
                }
            }
            match step_right(buf, q) {
                Some(n) => {
                    starts.push(n);
                    p = n;
                }
                None => p = q,
            }
        }
    }
    starts
}

fn last_position(buf: &Buffer) -> Position {
    let last_line = buf.line_count().saturating_sub(1);
    Position::new(last_line, buf.line_len(last_line).saturating_sub(1))
}

fn sentence_forward(buf: &Buffer, pos: Position) -> Position {
    sentence_starts(buf).into_iter().find(|&s| s > pos).unwrap_or_else(|| last_position(buf))
}

fn sentence_backward(buf: &Buffer, pos: Position) -> Position {
    sentence_starts(buf).into_iter().rfind(|&s| s < pos).unwrap_or(Position::ORIGIN)
}

/// `%`: jump to the matching bracket for the nearest bracket at or after the
/// cursor on the current line.
fn match_pair(buf: &Buffer, pos: Position) -> Option<Position> {
    const OPEN: &[char] = &['(', '[', '{'];
    const CLOSE: &[char] = &[')', ']', '}'];
    let text = buf.line(pos.line)?;
    let graphemes: Vec<&str> = text.graphemes(true).collect();
    let mut start_col = None;
    for (i, g) in graphemes.iter().enumerate().skip(pos.col) {
        let c = g.chars().next()?;
        if OPEN.contains(&c) || CLOSE.contains(&c) {
            start_col = Some(i);
            break;
        }
    }
    let col = start_col?;
    let c = graphemes[col].chars().next()?;
    if let Some(idx) = OPEN.iter().position(|&o| o == c) {
        let close = CLOSE[idx];
        let open = c;
        let mut depth = 1i32;
        let mut p = Position::new(pos.line, col);
        while let Some(next) = step_right(buf, p) {
            p = next;
            if let Some(ch) = buf.grapheme_at(p).and_then(|g| g.chars().next()) {
                if ch == open {
                    depth += 1;
                } else if ch == close {
                    depth -= 1;
                    if depth == 0 {
                        return Some(p);
                    }
                }
            }
        }
        None
    } else {
        let idx = CLOSE.iter().position(|&cl| cl == c)?;
        let open = OPEN[idx];
        let close = c;
        let mut depth = 1i32;
        let mut p = Position::new(pos.line, col);
        while let Some(prev) = step_left(buf, p) {
            p = prev;
            if let Some(ch) = buf.grapheme_at(p).and_then(|g| g.chars().next()) {
                if ch == close {
                    depth += 1;
                } else if ch == open {
                    depth -= 1;
                    if depth == 0 {
                        return Some(p);
                    }
                }
            }
        }
        None
    }
}

/// One `f`/`F`/`t`/`T` application, searching from immediately next to
/// `pos` in the given direction.
///
/// `t`/`T` (Till) deliberately do **not** try to detect "the cursor is
/// already sitting right where the *previous* `t` left it" and skip an
/// extra character the way real vim's `;` repeat does — that adjustment is
/// a property of *repeating in the same direction*, not of `t` itself, and
/// baking it in here previously made even a *first* `t{c}` press wrong (it
/// would skip past an adjacent target). The trade-off: repeating a `t` via
/// `;` immediately afterwards can be a no-op in the narrow case where the
/// cursor already sits exactly where a fresh search would land again.
/// Documented scope cut rather than threading "is this a repeat" state
/// through a pure motion function.
fn find_char(buf: &Buffer, pos: Position, kind: FindKind, target: char) -> Option<Position> {
    let text = buf.line(pos.line)?;
    let graphemes: Vec<&str> = text.graphemes(true).collect();
    let matches = |g: &&str| g.starts_with(target);
    if kind.is_forward() {
        let search_from = (pos.col + 1).min(graphemes.len());
        let found = graphemes.iter().enumerate().skip(search_from).find(|(_, g)| matches(g))?;
        let col = match kind {
            FindKind::To => found.0,
            FindKind::Till => found.0.saturating_sub(1),
            _ => unreachable!(),
        };
        Some(Position::new(pos.line, col))
    } else {
        if pos.col == 0 {
            return None;
        }
        let found = graphemes[..pos.col].iter().enumerate().rev().find(|(_, g)| matches(g))?;
        let col = match kind {
            FindKind::ToBack => found.0,
            FindKind::TillBack => found.0 + 1,
            _ => unreachable!(),
        };
        Some(Position::new(pos.line, col))
    }
}

/// `,` repeats the last `f`/`F`/`t`/`T` in the *opposite* direction.
/// `;` repeats it as-is. Both are expressed here as producing the
/// [`Motion::FindChar`] to apply, given the remembered `(kind, target)`.
pub fn repeat_find(kind: FindKind, target: char, reverse: bool) -> Motion {
    let kind = if reverse { kind.reversed() } else { kind };
    Motion::FindChar { kind, target }
}

use unicode_segmentation::UnicodeSegmentation;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_forward_stops_at_the_next_word_start() {
        let buf = Buffer::from_str("foo bar baz");
        let p = Motion::WordForward.apply(&buf, Position::new(0, 0), None);
        assert_eq!(p, Position::new(0, 4));
    }

    #[test]
    fn word_forward_treats_punctuation_as_its_own_class() {
        let buf = Buffer::from_str("foo(bar)");
        let p = Motion::WordForward.apply(&buf, Position::new(0, 0), None);
        assert_eq!(p, Position::new(0, 3)); // lands on '('
    }

    #[test]
    fn word_end_lands_on_the_last_grapheme_of_the_word() {
        let buf = Buffer::from_str("foo bar");
        let p = Motion::WordEnd.apply(&buf, Position::new(0, 0), None);
        assert_eq!(p, Position::new(0, 2)); // the second 'o'
    }

    #[test]
    fn count_multiplies_word_motions() {
        let buf = Buffer::from_str("one two three four");
        let p = Motion::WordForward.apply(&buf, Position::new(0, 0), Some(2));
        assert_eq!(p, Position::new(0, 8)); // start of "three"
    }

    #[test]
    fn file_end_without_a_count_goes_to_the_last_line() {
        // No trailing newline: with one, `ropey` (and this text engine,
        // deliberately — see `Buffer::line_count`'s docs) counts a real,
        // addressable, empty final line after it, which is a different
        // buffer than "three lines" for this purpose.
        let buf = Buffer::from_str("a\nb\nc");
        let p = Motion::FileEnd.apply(&buf, Position::new(0, 0), None);
        assert_eq!(p.line, 2);
    }

    #[test]
    fn file_end_with_a_count_goes_to_that_line() {
        let buf = Buffer::from_str("a\nb\nc\n");
        let p = Motion::FileEnd.apply(&buf, Position::new(2, 0), Some(2));
        assert_eq!(p.line, 1);
    }

    #[test]
    fn sentence_forward_lands_after_a_period_and_space() {
        let buf = Buffer::from_str("One sentence. Another one. A third.");
        let p = Motion::SentenceForward.apply(&buf, Position::new(0, 0), None);
        assert_eq!(p, Position::new(0, 14)); // start of "Another"
    }

    #[test]
    fn fresh_till_lands_immediately_before_an_adjacent_target() {
        // Regression test: `find_char` previously special-cased `Till` to
        // always skip an extra character, which was only correct for a
        // *repeated* `t` via `;`, not a fresh one — it would skip straight
        // over a comma one character away.
        let buf = Buffer::from_str("a,b,c");
        let p = Motion::FindChar { kind: FindKind::Till, target: ',' }.apply(&buf, Position::new(0, 0), None);
        assert_eq!(p, Position::new(0, 0), "t, from 'a' should land just before the adjacent ',', i.e. not move");
    }

    #[test]
    fn match_pair_finds_the_partner_bracket() {
        let buf = Buffer::from_str("foo(bar(baz))");
        let p = Motion::MatchPair.apply(&buf, Position::new(0, 3), None);
        assert_eq!(p, Position::new(0, 12));
    }
}
