//! Operators: things that act on a range described by a motion or text
//! object.
//!
//! Every operator in vi ultimately reduces to "compute a range, then do one
//! of a handful of things to the text in it". [`Operator::apply`] is that
//! single generic function. It does not know or care whether the range came
//! from a motion (`dw`), a text object (`di(`), or a visual selection
//! (`vjd`) — by the time it runs, a [`crate::core::Range`] and a
//! [`crate::core::Granularity`] are all it needs. That uniformity is the
//! payoff of encoding exclusive/inclusive/linewise on the *motion*
//! ([`super::motion::MotionKind`]) instead of here.

use crate::core::{Edit, Granularity, Position, Range};
use crate::text::Buffer;

/// The operators from the brief. `Delete`/`Yank`/`Change` share the same
/// "grab this range" shape; `Indent`/`Dedent`/the three case operators are
/// linewise- and charwise-capable respectively; `x`/`X`/`s`/`r`/`~`/`J` are
/// modelled separately in `mod.rs` because they either take no motion at all
/// or (for `r`) take a replacement character instead of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Delete,
    Change,
    Yank,
    Indent,
    Dedent,
    LowerCase,
    UpperCase,
    ToggleCase,
}

impl Operator {
    /// `true` for operators that leave the editor in Insert mode afterwards.
    pub fn enters_insert(self) -> bool {
        matches!(self, Operator::Change)
    }

    /// `true` for operators that must be wrapped in a single undo group
    /// (i.e. they mutate the buffer at all — `Yank` does not).
    pub fn mutates(self) -> bool {
        !matches!(self, Operator::Yank)
    }
}

/// What applying an operator produced, for the caller ([`super::Editor`]) to
/// act on: where the cursor lands, and what (if anything) should be written
/// to a register.
pub struct OperatorOutcome {
    pub cursor: Position,
    /// Text to store in a register, and the granularity it should be
    /// remembered with — see [`crate::core::Granularity`]'s docs for why
    /// that distinction has to survive into the register.
    pub register_write: Option<(String, Granularity)>,
}

/// Resolves a motion/text-object's `(start, end)` and [`super::motion::MotionKind`]
/// into the concrete, half-open [`Range`] an operator should act on.
///
/// This is where vim's "exclusive motion landing at column 0 rolls back to
/// the end of the previous line" rule lives (see `:help exclusive`) — it is
/// a property of how a *motion's* result becomes an operator *range*, not of
/// any particular operator, so it belongs here rather than duplicated in
/// every operator.
pub fn charwise_range(buf: &Buffer, start: Position, end: Position, kind: super::motion::MotionKind) -> (Range, Granularity) {
    use super::motion::MotionKind;
    match kind {
        MotionKind::Linewise => {
            let (a, b) = if start <= end { (start.line, end.line) } else { (end.line, start.line) };
            // Note: this is the *content* span (`a..=b` inclusive, no
            // trailing newline) — not the wider span `Operator::apply`
            // actually deletes. Keeping this a content range means every
            // caller (this function, `Editor`'s `dd`/`guu`/`>>` path, text
            // objects) can treat `.line`/`range.normalized()` as "the real
            // first/last line", and only `Operator::apply` itself needs to
            // know about the newline-consuming widening — see
            // `linewise_delete_range`'s docs.
            (linewise_content_range(buf, a, b), Granularity::Linewise)
        }
        MotionKind::Exclusive => {
            let (a, mut b) = if start <= end { (start, end) } else { (end, start) };
            // vim: an exclusive motion whose end lands in column 0 (i.e. it
            // crossed a line boundary) is pulled back to the end of the
            // previous line instead of eating that line's leading newline.
            // Concretely, this is what makes `dw` on the last word of a line
            // delete the word but not merge it with the next line.
            if b.col == 0 && b.line > a.line {
                let prev = b.line - 1;
                b = Position::new(prev, buf.line_len(prev));
            }
            (Range::new(a, b), Granularity::Charwise)
        }
        MotionKind::Inclusive => {
            let (a, b) = if start <= end { (start, end) } else { (end, start) };
            // Extend one grapheme past `b` so the landing char is *included*
            // (that is what "inclusive" means). But when `b` is already the
            // last grapheme of its line, stop at end-of-line rather than
            // stepping onto the next line's column 0 — otherwise the range
            // would swallow the trailing newline and merge the two lines.
            // That mattered the day `D`/`C`/`Y` (`d$`/`c$`/`y$`) landed: on a
            // non-final line, `y$` used to yank `"foo\n"` and `D` used to pull
            // the next line up, neither of which is what vim does. This is the
            // one place that rule can live, since every inclusive motion
            // (`$`, `e`, `f`, `t`, `%`) funnels through here. Cross-line
            // inclusive motions like `%` are unaffected: their landing `b` is
            // not the last char of its line, so the `b.col + 1` branch runs.
            let line_len = buf.line_len(b.line);
            let after = if b.col + 1 < line_len { Position::new(b.line, b.col + 1) } else { Position::new(b.line, line_len) };
            (Range::new(a, after), Granularity::Charwise)
        }
    }
}

/// Builds the range to hand to `Buffer::apply` for a linewise deletion of
/// lines `first..=last`, correctly consuming a newline on one side so that
/// the line count actually shrinks.
///
/// Deleting `[first,0)..[last+1,0)` removes the lines and their trailing
/// newlines cleanly — except when `last` is the buffer's final line, which
/// has no trailing newline to consume. In that case the *leading* newline
/// (the one ending the previous line) is consumed instead. If `first == 0`
/// and `last` is also the last line — the whole buffer is being removed —
/// there is no newline on either side to borrow, so the raw content range is
/// returned and the text engine's "a buffer always has >= 1 line" invariant
/// is relied on to leave a single empty line behind.
pub(crate) fn linewise_delete_range(buf: &Buffer, first: usize, last: usize) -> Range {
    let line_count = buf.line_count();
    if last + 1 < line_count {
        Range::new(Position::new(first, 0), Position::new(last + 1, 0))
    } else if first > 0 {
        let prev = first - 1;
        Range::new(Position::new(prev, buf.line_len(prev)), Position::new(last, buf.line_len(last)))
    } else {
        Range::new(Position::new(first, 0), Position::new(last, buf.line_len(last)))
    }
}

/// The *content-only* span of lines `first..=last` (no trailing newline) —
/// what every caller outside this module works with. See
/// [`linewise_delete_range`]'s docs for the wider span actually removed from
/// the buffer, which only [`Operator::apply`] needs to compute.
pub(crate) fn linewise_content_range(buf: &Buffer, first: usize, last: usize) -> Range {
    Range::new(Position::new(first, 0), Position::new(last, buf.line_len(last)))
}

impl Operator {
    /// Applies this operator to `range` (of the given `granularity`),
    /// mutating `buf` as needed and reporting what happened.
    ///
    /// `range` must already be normalized to document order; callers get
    /// that for free from [`charwise_range`] or [`super::text_object::resolve`].
    ///
    /// `shiftwidth`/`expandtab` only matter to [`Operator::Indent`]/
    /// [`Operator::Dedent`] — every other variant ignores them. They are
    /// still parameters of this one generic function, rather than a second
    /// `indent`-only entry point, so that `Editor::apply_operator` (the
    /// single call site driving all of `d`/`c`/`y`/`>`/`<`/`gu`/`gU`/`g~`)
    /// does not need to know which operators care about indentation.
    pub fn apply(
        self,
        buf: &mut Buffer,
        range: Range,
        granularity: Granularity,
        shiftwidth: usize,
        expandtab: bool,
    ) -> crate::Result<OperatorOutcome> {
        let (start, end) = range.normalized();
        // `range` is always a *content* span (see `charwise_range`'s docs);
        // deleting a Linewise range has to widen it to also consume a
        // newline, or the line count would never actually shrink. That
        // widening is this function's own business — every other operator,
        // and every caller, only ever deals in content spans.
        let deletion_range = if granularity == Granularity::Linewise { linewise_delete_range(buf, start.line, end.line) } else { range };
        match self {
            Operator::Delete | Operator::Change => {
                let content = buf.slice(range);
                let register_text = if granularity == Granularity::Linewise { format!("{content}\n") } else { content };
                let cursor = buf.apply(Edit::delete(deletion_range))?;
                let cursor = match granularity {
                    Granularity::Linewise => {
                        let line = cursor.line.min(buf.line_count() - 1);
                        Position::new(line, first_non_blank_col(buf, line))
                    }
                    _ => cursor,
                };
                Ok(OperatorOutcome { cursor, register_write: Some((register_text, granularity)) })
            }
            Operator::Yank => {
                let content = buf.slice(range);
                let register_text = if granularity == Granularity::Linewise { format!("{content}\n") } else { content };
                let cursor = match granularity {
                    Granularity::Linewise => Position::new(start.line, first_non_blank_col(buf, start.line)),
                    _ => start,
                };
                Ok(OperatorOutcome { cursor: buf.clamp(cursor), register_write: Some((register_text, granularity)) })
            }
            Operator::Indent | Operator::Dedent => {
                let (first, last) = (start.line, end.line);
                for line in first..=last {
                    indent_line(buf, line, shiftwidth, expandtab, matches!(self, Operator::Indent))?;
                }
                let cursor = Position::new(first, first_non_blank_col(buf, first));
                Ok(OperatorOutcome { cursor: buf.clamp(cursor), register_write: None })
            }
            Operator::LowerCase | Operator::UpperCase | Operator::ToggleCase => {
                let text = buf.slice(range);
                let transformed = match self {
                    Operator::LowerCase => text.to_lowercase(),
                    Operator::UpperCase => text.to_uppercase(),
                    Operator::ToggleCase => toggle_case(&text),
                    _ => unreachable!(),
                };
                let cursor = buf.apply(Edit::replace(range, transformed))?;
                Ok(OperatorOutcome { cursor, register_write: None })
            }
        }
    }
}

fn toggle_case(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_uppercase() {
                c.to_lowercase().next().unwrap_or(c)
            } else if c.is_lowercase() {
                c.to_uppercase().next().unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Column of the first non-blank grapheme on `line`, defaulting to 0.
pub(crate) fn first_non_blank_col(buf: &Buffer, line: usize) -> usize {
    let Some(text) = buf.line(line) else { return 0 };
    for (i, g) in unicode_segmentation::UnicodeSegmentation::graphemes(text.as_str(), true).enumerate() {
        if !g.chars().next().map(char::is_whitespace).unwrap_or(true) {
            return i;
        }
    }
    0
}

/// Shifts a single `line` by one shiftwidth, in the given direction
/// (`>>`/`<<`/`>{motion}`/`<{motion}` all bottom out here, one call per line
/// in the range).
pub(crate) fn indent_line(buf: &mut Buffer, line: usize, shiftwidth: usize, expandtab: bool, indent: bool) -> crate::Result<()> {
    let Some(text) = buf.line(line) else { return Ok(()) };
    if indent {
        let fill = if expandtab { " ".repeat(shiftwidth) } else { "\t".to_string() };
        buf.apply(Edit::insert(Position::new(line, 0), fill))?;
    } else {
        // `removed_width` tracks *columns* (a tab is worth `shiftwidth` of
        // them, matching vim's own dedent rule) so the loop knows when to
        // stop; `removed_graphemes` tracks how many actual grapheme
        // clusters that consumed, since a tab is one grapheme wide
        // regardless of how many columns it counts for. Deleting
        // `removed_width` graphemes instead would eat far more text than
        // intended on any line using tabs.
        let mut removed_width = 0usize;
        let mut removed_graphemes = 0usize;
        for g in unicode_segmentation::UnicodeSegmentation::graphemes(text.as_str(), true) {
            if removed_width >= shiftwidth {
                break;
            }
            match g {
                "\t" => {
                    removed_width += shiftwidth;
                    removed_graphemes += 1;
                }
                " " => {
                    removed_width += 1;
                    removed_graphemes += 1;
                }
                _ => break,
            }
        }
        if removed_graphemes > 0 {
            let end = removed_graphemes.min(buf.line_len(line));
            buf.apply(Edit::delete(Range::new(Position::new(line, 0), Position::new(line, end))))?;
        }
    }
    Ok(())
}
