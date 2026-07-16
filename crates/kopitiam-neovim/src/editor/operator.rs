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
    /// `!{motion}` / `!!` — the shell-filter operator. Unlike every other
    /// operator it never edits text in [`Operator::apply`]: it only resolves a
    /// motion/object to a line range, which
    /// [`super::Editor::run_operator`] intercepts to open a prefilled
    /// `:{range}!` command line. Modelled as an operator purely so it inherits
    /// the operator-pending grammar — counts, motions, text objects and the
    /// `!!` doubled-key line form all compose for free.
    Filter,
    /// `zf{motion}` / visual `zf` — the manual-fold-create operator. Like
    /// [`Operator::Filter`] it never edits text in [`Operator::apply`]: it
    /// resolves a motion/object to a *line* range, which
    /// [`super::Editor::run_operator`] intercepts to create a closed fold over
    /// those lines (see [`super::fold::FoldSet::create`]). Modelled as an
    /// operator so `zf` inherits the whole operator-pending grammar — counts
    /// (`zf3j`), motions (`zfG`, `zf}`) and text objects (`zfip`, `zfa{`) all
    /// compose for free, exactly as they do for `d`/`y`/`!`.
    Fold,
    /// `={motion}` / `==` / visual `=` — the reindent/format operator. Always
    /// operates on whole *lines* (the lines the range touches), like vim's `=`.
    ///
    /// # Which path runs: fallback today, LSP later
    ///
    /// vim routes `=` to `equalprg`/`indentexpr` when set, else its built-in
    /// C-style indenter. kvim's intended routing is **LSP
    /// `textDocument/rangeFormatting` when a server is ready, else this
    /// deterministic fallback** (the "native Rust before local AI before cloud"
    /// order in `CLAUDE.md`). The LSP leg is *not yet wired*: it needs a
    /// `range_formatting` request on `kopitiam-semantic`'s session — a crate
    /// outside this change's owned directory, so the LSP leg was deferred with
    /// its full rationale recorded on bead `kopitiam-cj0.57` (an AI-decision
    /// review item: why deferred, where it goes, what would make it wrong).
    /// Until that lands, `=` **always** runs the fallback below. The fallback is
    /// applied here in [`Operator::apply`]; when the LSP leg arrives it belongs
    /// in [`super::Editor::run_operator`] (an `EditorResponse::Action`, as
    /// `<C-]>` does for go-to-definition), falling through to this same `apply`
    /// when the server is absent or still starting.
    ///
    /// # The fallback: a light, brace-depth C-style reindent
    ///
    /// Each line is aligned to `shiftwidth * depth`, where `depth` is the count
    /// of unclosed `{`/`(`/`[` seen so far (a line beginning with a closer is
    /// dedented one level, so a `}` lines up under its opener). It is
    /// deliberately simple and predictable — it does not parse strings,
    /// character literals, comments or continuation lines, so a brace inside a
    /// string will skew the depth. A real language-aware indenter is filed as
    /// bead `kopitiam-cj0.58`; this is the "keep it simple, respect
    /// `shiftwidth`/`expandtab`" brief.
    Format,
}

impl Operator {
    /// `true` for operators that leave the editor in Insert mode afterwards.
    pub fn enters_insert(self) -> bool {
        matches!(self, Operator::Change)
    }

    /// `true` for operators that must be wrapped in a single undo group
    /// (i.e. they mutate the buffer at all — `Yank`, and `Fold` which only
    /// touches the fold table, do not).
    pub fn mutates(self) -> bool {
        !matches!(self, Operator::Yank | Operator::Fold)
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
            Operator::Format => {
                let (first, last) = (start.line, end.line);
                reindent_c_style(buf, first, last, shiftwidth, expandtab)?;
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
            // `Filter` is intercepted in `Editor::run_operator` before `apply`
            // is ever reached (it opens a command line rather than editing), so
            // this arm exists only to keep the match exhaustive.
            Operator::Filter => unreachable!("the filter operator is handled in Editor::run_operator, never applied"),
            // Like `Filter`, `Fold` is intercepted in `Editor::run_operator`
            // before `apply` is reached (it edits the fold table, not the text).
            Operator::Fold => unreachable!("the fold operator is handled in Editor::run_operator, never applied"),
        }
    }
}

/// The net bracket depth change across `s`: `+1` per opening `{`/`(`/`[`, `-1`
/// per closing `}`/`)`/`]`. No string/comment awareness — see
/// [`Operator::Format`]'s docs for that deliberate simplification.
fn net_bracket_delta(s: &str) -> i32 {
    let mut d = 0;
    for c in s.chars() {
        match c {
            '{' | '(' | '[' => d += 1,
            '}' | ')' | ']' => d -= 1,
            _ => {}
        }
    }
    d
}

/// Reindents lines `first..=last` to a brace-nesting depth, the fallback path
/// of the `=` operator ([`Operator::Format`]). The starting depth is derived by
/// scanning every line *above* `first`, so a reindented block nests correctly
/// inside its surroundings. A line whose first non-blank grapheme is a closer
/// (`}`/`)`/`]`) is pulled out one level so it aligns under its opener. Blank
/// lines are emptied of trailing indent and do not affect depth.
fn reindent_c_style(buf: &mut Buffer, first: usize, last: usize, shiftwidth: usize, expandtab: bool) -> crate::Result<()> {
    let mut depth: i32 = 0;
    for line in 0..first {
        depth += net_bracket_delta(&buf.line(line).unwrap_or_default());
    }
    for line in first..=last {
        let content = buf.line(line).unwrap_or_default();
        let trimmed = content.trim_start();
        // Existing leading-whitespace width in *graphemes* — whitespace is
        // one grapheme per char, so a char count is exact here and lets us
        // replace exactly the current indent run.
        let lead = content.chars().take_while(|c| *c == ' ' || *c == '\t').count();
        if trimmed.is_empty() {
            // Strip a blank line's stray indentation; depth is unchanged.
            if lead > 0 {
                buf.apply(Edit::delete(Range::new(Position::new(line, 0), Position::new(line, lead))))?;
            }
            continue;
        }
        let closes_first = matches!(trimmed.chars().next(), Some('}') | Some(')') | Some(']'));
        let this_depth = if closes_first { (depth - 1).max(0) } else { depth.max(0) };
        let want = if expandtab {
            " ".repeat(this_depth as usize * shiftwidth)
        } else {
            "\t".repeat(this_depth as usize)
        };
        // Only rewrite when the indent actually differs, to avoid churning the
        // undo group with no-op edits on already-correct lines.
        let current_lead: String = content.chars().take(lead).collect();
        if current_lead != want {
            buf.apply(Edit::replace(Range::new(Position::new(line, 0), Position::new(line, lead)), want))?;
        }
        depth += net_bracket_delta(&content);
        if depth < 0 {
            depth = 0;
        }
    }
    Ok(())
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
