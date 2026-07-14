//! TeX/LaTeX lexer.
//!
//! # The awkward case: `%` comments
//!
//! TeX comments run from an unescaped `%` to end-of-line, but `\%`
//! produces a literal percent sign and starts nothing. The naive fix is
//! "check whether the character before `%` is a backslash," which is
//! wrong for `\\%` (a line-break command `\\` immediately followed by a
//! *real* comment) — the correct rule is the *parity* of the run of
//! backslashes immediately preceding the `%`.
//!
//! This lexer does not implement that parity check directly. It doesn't
//! need to: scanning strictly left to right, every `\` is consumed
//! *together* with whatever follows it (a command name, or — for a
//! non-letter — a single control-symbol character, see [`command_or_escape`])
//! in one step that advances past both characters atomically. So when the
//! main loop's cursor lands on a `%` byte on its own, that `%` was — by
//! construction — never part of a two-character unit consumed by the
//! previous iteration, which is exactly "an even (possibly zero) number
//! of backslashes precede it." `\%` is swallowed whole as one token by
//! the `\`-handling branch and the loop never re-examines its `%` in
//! isolation; `\\%` consumes `\\` as one token first, then sees the `%`
//! fresh and correctly starts a comment. Parity falls out of ordinary
//! left-to-right consumption for free.
//!
//! # Multi-line constructs
//!
//! TeX comments themselves never span lines (each is bounded by its own
//! EOL). What can span lines here: a `verbatim` environment (its content
//! must not be touched by any of this lexer's rules until `\end{verbatim}`)
//! and inline math (`$...$`) left open across a line break, which is rarer
//! in well-formed documents but handled the same "unterminated at EOL
//! carries state forward" way every other language in this crate handles
//! an open string.

use crate::util::scan_number;
use crate::{HighlightKind, HighlightSpan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TexState {
    #[default]
    Normal,
    /// Inside `\begin{verbatim} ... \end{verbatim}`: no highlighting is
    /// applied to its content at all (that's the point of `verbatim`),
    /// represented by returning no spans for these lines.
    Verbatim,
    /// Inside `$...$` math mode left open at end-of-line.
    Math,
}

const ESCAPABLE_SPECIALS: &[u8] = b"%$&#_{}";
const PUNCTUATION_CHARS: &[u8] = b"{}";
const OPERATOR_CHARS: &[u8] = b"&^_~";

pub fn highlight_line(line: &str, state: &mut TexState) -> Vec<HighlightSpan> {
    match *state {
        TexState::Normal => finish(line, 0, Vec::new(), state),
        TexState::Verbatim => {
            if let Some(end_at) = line.find(r"\end{verbatim}") {
                *state = TexState::Normal;
                // Everything from `\end{verbatim}` onward is ordinary TeX
                // again; the verbatim content before it on this line (a
                // rare same-line case) is left unhighlighted, matching
                // every other verbatim line.
                let close_end = end_at + r"\end{verbatim}".len();
                let mut spans = vec![span(end_at, close_end, HighlightKind::Keyword)];
                spans.extend(finish(line, close_end, Vec::new(), state));
                spans
            } else {
                Vec::new()
            }
        }
        TexState::Math => {
            let len = line.len();
            if let Some(end) = find_unescaped(line, 0, b'$') {
                *state = TexState::Normal;
                let mut spans = vec![span(0, end + 1, HighlightKind::String)];
                spans.extend(finish(line, end + 1, Vec::new(), state));
                spans
            } else {
                vec![span(0, len, HighlightKind::String)]
            }
        }
    }
}

fn finish(line: &str, mut i: usize, mut spans: Vec<HighlightSpan>, state: &mut TexState) -> Vec<HighlightSpan> {
    let bytes = line.as_bytes();
    let len = bytes.len();

    while i < len {
        let c = bytes[i];

        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }

        // See the module docs: this is always a genuine, unescaped
        // comment start by construction.
        if c == b'%' {
            spans.push(span(i, len, HighlightKind::Comment));
            return spans;
        }

        if c == b'\\' {
            let (end, kind, command_name) = command_or_escape(line, i);
            spans.push(span(i, end, kind));

            // `\begin{verbatim}` / `\begin{env}`: highlight the
            // environment name and, for `verbatim`, switch state so
            // subsequent lines are left untouched until `\end{verbatim}`.
            if (command_name == Some("begin") || command_name == Some("end"))
                && let Some(env_end) = brace_group_end(line, end)
            {
                spans.push(span(end, env_end, HighlightKind::Type));
                // `env_end` is just past the closing `}`; `brace_group_end`
                // guarantees at least `{}` was matched, so `env_end - 1`
                // (the `}`) is always `>= end + 1` (just past the `{`).
                let env_name = &line[end + 1..env_end - 1];
                if command_name == Some("begin") && env_name == "verbatim" {
                    // The rest of *this* line (if any trails past
                    // `\begin{verbatim}`) is still ordinary TeX;
                    // verbatim content starts on the next line.
                    *state = TexState::Verbatim;
                }
                i = env_end;
                continue;
            }
            i = end;
            continue;
        }

        // Math mode.
        if c == b'$' {
            let start = i;
            if let Some(end) = find_unescaped(line, i + 1, b'$') {
                spans.push(span(start, end + 1, HighlightKind::String));
                i = end + 1;
                continue;
            }
            spans.push(span(start, len, HighlightKind::String));
            *state = TexState::Math;
            return spans;
        }

        if c.is_ascii_digit() {
            let end = scan_number(line, i);
            spans.push(span(i, end, HighlightKind::Number));
            i = end;
            continue;
        }

        if PUNCTUATION_CHARS.contains(&c) {
            spans.push(span(i, i + 1, HighlightKind::Punctuation));
            i += 1;
            continue;
        }
        if OPERATOR_CHARS.contains(&c) {
            spans.push(span(i, i + 1, HighlightKind::Operator));
            i += 1;
            continue;
        }

        i += 1;
    }

    spans
}

/// Scans a `\command` (control word: letters) or `\x` (control symbol: one
/// non-letter) starting at the backslash. Returns the exclusive end
/// offset, the [`HighlightKind`] to tag it with, and — only for a control
/// word — the command name itself (used by the caller to special-case
/// `\begin`/`\end`).
///
/// This is the atomic "consume `\` together with what follows" step the
/// module docs describe: nothing outside this function ever looks at a
/// `%` (or any other escapable special) that immediately follows a `\`.
fn command_or_escape(line: &str, backslash_at: usize) -> (usize, HighlightKind, Option<&str>) {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let next = backslash_at + 1;
    match bytes.get(next) {
        None => (next, HighlightKind::Punctuation, None),
        Some(&b) if b.is_ascii_alphabetic() => {
            let mut end = next;
            while end < len && bytes[end].is_ascii_alphabetic() {
                end += 1;
            }
            let name = &line[next..end];
            (end, HighlightKind::Keyword, Some(name))
        }
        Some(&b) if ESCAPABLE_SPECIALS.contains(&b) => (next + 1, HighlightKind::Escape, None),
        Some(_) => (next + 1, HighlightKind::Keyword, None), // control symbol, e.g. `\\`, `\,`.
    }
}

/// From `start` (expected to be a `{`), finds the offset just past the
/// matching `}`, or `None` if `start` isn't `{` or there's no closing
/// brace on this line. Braces are assumed non-nested for this purpose —
/// an environment name (`\begin{verbatim}`) never itself contains `{}`.
fn brace_group_end(line: &str, start: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let close_rel = line[start + 1..].find('}')?;
    Some(start + 1 + close_rel + 1)
}

/// Finds the next occurrence of `target` at or after `start` that is not
/// immediately preceded by an unescaped backslash — same "atomic
/// consumption" reasoning as [`command_or_escape`], applied to scanning
/// forward for a math-mode closer without a full re-lex.
fn find_unescaped(line: &str, start: usize, target: u8) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start;
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2;
            continue;
        }
        if bytes[i] == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn span(start: usize, end: usize, kind: HighlightKind) -> HighlightSpan {
    HighlightSpan { start, end, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Highlighter, Language};

    fn kinds(line: &str) -> Vec<(String, HighlightKind)> {
        Highlighter::new(Language::Tex)
            .highlight_line(line)
            .into_iter()
            .map(|s| (line[s.start..s.end].to_string(), s.kind))
            .collect()
    }

    #[test]
    fn command_is_a_keyword() {
        let k = kinds(r"\section{Introduction}");
        assert!(k.contains(&(r"\section".to_string(), HighlightKind::Keyword)));
    }

    #[test]
    fn plain_comment_runs_to_eol() {
        let k = kinds(r"some text % a real comment");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Comment && t == "% a real comment"));
    }

    #[test]
    fn escaped_percent_is_not_a_comment() {
        let k = kinds(r"100\% complete, no comment here");
        assert!(!k.iter().any(|(_, kind)| *kind == HighlightKind::Comment));
        assert!(k.contains(&(r"\%".to_string(), HighlightKind::Escape)));
    }

    /// The genuinely awkward case: `\\%` is a line-break command followed
    /// by a REAL comment, not an escaped percent.
    #[test]
    fn double_backslash_then_percent_is_a_real_comment() {
        let k = kinds(r"end of line\\% now a real comment");
        assert!(k.contains(&(r"\\".to_string(), HighlightKind::Keyword)));
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Comment && t == "% now a real comment"));
    }

    #[test]
    fn inline_math_is_tagged() {
        let k = kinds(r"the equation $x^2 + y^2 = z^2$ holds");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t == "$x^2 + y^2 = z^2$"));
    }

    #[test]
    fn numbers_are_tagged() {
        let k = kinds(r"\SI{42}{\kelvin}");
        assert!(k.contains(&("42".to_string(), HighlightKind::Number)));
    }

    #[test]
    fn begin_verbatim_environment_name_is_tagged() {
        let k = kinds(r"\begin{verbatim}");
        assert!(k.contains(&(r"\begin".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("{verbatim}".to_string(), HighlightKind::Type)));
    }

    #[test]
    fn verbatim_environment_spans_multiple_lines_untouched() {
        let mut h = Highlighter::new(Language::Tex);
        h.highlight_line(r"\begin{verbatim}");
        assert_ne!(h.state(), crate::LineState::initial(Language::Tex));

        // Inside verbatim, a `%` must NOT start a comment and a `\foo`
        // must NOT be tagged a command -- this is real verbatim content.
        let l2 = h.highlight_line(r"\foo % not a comment, not a command either");
        assert!(l2.is_empty());

        let l3 = h.highlight_line(r"\end{verbatim}");
        assert!(l3.iter().any(|s| s.kind == HighlightKind::Keyword));
        assert_eq!(h.state(), crate::LineState::initial(Language::Tex));
    }

    #[test]
    fn multi_line_math_mode() {
        let mut h = Highlighter::new(Language::Tex);
        h.highlight_line(r"$ x^2 + y^2");
        assert_ne!(h.state(), crate::LineState::initial(Language::Tex));
        let l2 = h.highlight_line(r"= z^2 $ done");
        assert!(l2.iter().any(|s| s.kind == HighlightKind::String));
        assert_eq!(h.state(), crate::LineState::initial(Language::Tex));
    }

    #[test]
    fn pathological_input_does_not_hang_or_panic() {
        let long_line = "u".repeat(200_000);
        let _ = Highlighter::new(Language::Tex).highlight_line(&long_line);

        let mut h = Highlighter::new(Language::Tex);
        let unterminated = format!("$x {}", "a".repeat(100_000));
        let spans = h.highlight_line(&unterminated);
        assert!(spans.iter().any(|s| s.kind == HighlightKind::String));
        assert_ne!(h.state(), crate::LineState::initial(Language::Tex));

        // A lone trailing backslash must not panic on out-of-bounds access.
        let mut h2 = Highlighter::new(Language::Tex);
        let _ = h2.highlight_line(r"trailing backslash \");
    }
}
