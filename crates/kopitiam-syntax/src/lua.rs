//! Lua lexer.
//!
//! The multi-line hazard here is Lua's "long bracket" syntax: `[[...]]`,
//! or `[=[...]=]`, `[==[...]==]` with more `=` signs for arbitrarily
//! deeper nesting-safety, used for both long comments (`--[[ ... ]]`) and
//! long strings (`[[ ... ]]`). The number of `=` signs in the opener must
//! match exactly in the closer, which is exactly the kind of stateful
//! bookkeeping across a line boundary this crate exists to get right (see
//! the crate-level docs and `kopitiam-lua`, which is implementing a full
//! Lua VM elsewhere in this workspace and independently has to solve the
//! same lexical-grammar problem).

use crate::util::{is_ident_start, scan_ident, scan_number, scan_run, skip_inline_whitespace};
use crate::{HighlightKind, HighlightSpan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LuaState {
    #[default]
    Normal,
    /// Inside `--[=*[ ... ]=*]`, `level` equal-signs deep.
    LongComment { level: u32 },
    /// Inside `[=*[ ... ]=*]`, `level` equal-signs deep.
    LongString { level: u32 },
    /// Inside a `"..."`/`'...'` string left open at end-of-line (Lua
    /// allows this via a line-continuation backslash; treated leniently
    /// the same way the Rust lexer treats an unterminated string, see its
    /// module docs).
    String { quote: u8 },
}

const KEYWORDS: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if",
    "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while",
];

/// Lua's standard-library table names, highlighted as `Type` -- the
/// closest fit for "namespace-like builtin," since Lua otherwise has no
/// static type system to speak of.
const BUILTIN_MODULES: &[&str] =
    &["string", "table", "math", "io", "os", "coroutine", "debug", "utf8", "package"];

const OPERATOR_CHARS: &str = "=~<>+-*/%^#.:";
const PUNCTUATION_CHARS: &str = "(){}[],;";

pub fn highlight_line(line: &str, state: &mut LuaState) -> Vec<HighlightSpan> {
    let mut spans = Vec::new();
    let len = line.len();

    match *state {
        LuaState::Normal => {}
        LuaState::LongComment { level } => {
            if let Some(end) = find_long_bracket_close(line, 0, level) {
                spans.push(span(0, end, HighlightKind::Comment));
                *state = LuaState::Normal;
                return finish(line, end, spans, state);
            }
            spans.push(span(0, len, HighlightKind::Comment));
            return spans;
        }
        LuaState::LongString { level } => {
            if let Some(end) = find_long_bracket_close(line, 0, level) {
                spans.push(span(0, end, HighlightKind::String));
                *state = LuaState::Normal;
                return finish(line, end, spans, state);
            }
            spans.push(span(0, len, HighlightKind::String));
            return spans;
        }
        LuaState::String { quote } => match scan_quoted_body(line, 0, quote, &mut spans) {
            Some(end) => {
                *state = LuaState::Normal;
                return finish(line, end, spans, state);
            }
            None => {
                *state = LuaState::String { quote };
                return spans;
            }
        },
    }

    finish(line, 0, spans, state)
}

fn finish(line: &str, mut i: usize, mut spans: Vec<HighlightSpan>, state: &mut LuaState) -> Vec<HighlightSpan> {
    let bytes = line.as_bytes();
    let len = bytes.len();

    while i < len {
        let c = bytes[i];

        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }

        // Comment: `--`, possibly a long comment `--[=*[`.
        if bytes[i..].starts_with(b"--") {
            let start = i;
            if let Some(level) = long_bracket_open_level(line, i + 2) {
                let body_start = i + 2 + level as usize + 2;
                if let Some(end) = find_long_bracket_close(line, body_start, level) {
                    spans.push(span(start, end, HighlightKind::Comment));
                    i = end;
                    continue;
                }
                spans.push(span(start, len, HighlightKind::Comment));
                *state = LuaState::LongComment { level };
                return spans;
            }
            spans.push(span(start, len, HighlightKind::Comment));
            return spans;
        }

        // Long string: `[=*[`.
        if c == b'[' && let Some(level) = long_bracket_open_level(line, i) {
            let start = i;
            let body_start = i + level as usize + 2;
            if let Some(end) = find_long_bracket_close(line, body_start, level) {
                spans.push(span(start, end, HighlightKind::String));
                i = end;
                continue;
            }
            spans.push(span(start, len, HighlightKind::String));
            *state = LuaState::LongString { level };
            return spans;
        }

        // Ordinary quoted string.
        if c == b'"' || c == b'\'' {
            let start = i;
            spans.push(span(start, start + 1, HighlightKind::String));
            match scan_quoted_body(line, i + 1, c, &mut spans) {
                Some(end) => {
                    i = end;
                    continue;
                }
                None => {
                    *state = LuaState::String { quote: c };
                    return spans;
                }
            }
        }

        if c.is_ascii_digit() {
            let end = scan_number(line, i);
            spans.push(span(i, end, HighlightKind::Number));
            i = end;
            continue;
        }

        if is_ident_start(c as char) || !c.is_ascii() {
            let end = scan_ident(line, i);
            if end == i {
                let ch = line[i..].chars().next().unwrap();
                i += ch.len_utf8();
                continue;
            }
            let word = &line[i..end];
            let after_ws = skip_inline_whitespace(line, end);
            let is_call = bytes.get(after_ws) == Some(&b'(');

            let kind = if KEYWORDS.contains(&word) {
                HighlightKind::Keyword
            } else if BUILTIN_MODULES.contains(&word) {
                HighlightKind::Type
            } else if is_call {
                HighlightKind::Function
            } else {
                i = end;
                continue;
            };
            spans.push(span(i, end, kind));
            i = end;
            continue;
        }

        if PUNCTUATION_CHARS.as_bytes().contains(&c) {
            spans.push(span(i, i + 1, HighlightKind::Punctuation));
            i += 1;
            continue;
        }
        if OPERATOR_CHARS.as_bytes().contains(&c) {
            let end = scan_run(line, i, OPERATOR_CHARS);
            spans.push(span(i, end, HighlightKind::Operator));
            i = end;
            continue;
        }

        i += 1;
    }

    spans
}

/// If `line[i..]` opens a long bracket (`[`, then zero-or-more `=`, then
/// `[`), returns the equal-sign count (the "level"). Used both for
/// `[[...]]` long strings and, after a leading `--`, long comments.
fn long_bracket_open_level(line: &str, i: usize) -> Option<u32> {
    let bytes = line.as_bytes();
    if bytes.get(i) != Some(&b'[') {
        return None;
    }
    let mut j = i + 1;
    let mut level = 0u32;
    while bytes.get(j) == Some(&b'=') {
        level += 1;
        j += 1;
    }
    if bytes.get(j) == Some(&b'[') { Some(level) } else { None }
}

/// From `start`, finds the offset just past a long-bracket closer (`]`,
/// `level` `=`s, `]`) matching `level`. `start` is typically the position
/// right after the opener on the line that opened it, or 0 when resuming
/// a construct left open by a previous line.
fn find_long_bracket_close(line: &str, start: usize, level: u32) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start;
    while i < len {
        if bytes[i] == b']' {
            let mut j = i + 1;
            let mut count = 0u32;
            while bytes.get(j) == Some(&b'=') {
                count += 1;
                j += 1;
            }
            if count == level && bytes.get(j) == Some(&b']') {
                return Some(j + 1);
            }
        }
        i += 1;
    }
    None
}

/// Scans a `"..."`/`'...'` body (the matching `quote` byte closes it),
/// producing non-overlapping `String`/`Escape` spans -- same tiling
/// strategy as the Rust lexer's `scan_string_body`, see its docs,
/// including the same `Option` return contract: `Some(end)` means closed
/// (even if `end == line.len()`), `None` means unterminated.
fn scan_quoted_body(line: &str, start: usize, quote: u8, spans: &mut Vec<HighlightSpan>) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start;
    let mut run_start = start;
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            if run_start < i {
                spans.push(span(run_start, i, HighlightKind::String));
            }
            spans.push(span(i, i + 2, HighlightKind::Escape));
            i += 2;
            run_start = i;
        } else if bytes[i] == quote {
            spans.push(span(run_start, i + 1, HighlightKind::String));
            return Some(i + 1);
        } else {
            i += 1;
        }
    }
    if run_start < len {
        spans.push(span(run_start, len, HighlightKind::String));
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
        Highlighter::new(Language::Lua)
            .highlight_line(line)
            .into_iter()
            .map(|s| (line[s.start..s.end].to_string(), s.kind))
            .collect()
    }

    #[test]
    fn keywords_and_function_calls() {
        let k = kinds("local function add(a, b) return helper(a, b) end");
        assert!(k.contains(&("local".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("function".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("return".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("end".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("add".to_string(), HighlightKind::Function)));
        assert!(k.contains(&("helper".to_string(), HighlightKind::Function)));
    }

    #[test]
    fn line_comment() {
        let k = kinds("local x = 1 -- a comment");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Comment && t == "-- a comment"));
    }

    #[test]
    fn short_strings_with_escapes() {
        let k = kinds(r#"local s = "hi\nthere""#);
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t == "hi"));
        assert!(k.contains(&(r"\n".to_string(), HighlightKind::Escape)));
    }

    #[test]
    fn numbers_hex_and_float() {
        let k = kinds("local a = 0xFF local b = 3.14 local c = 1e10");
        assert!(k.contains(&("0xFF".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("3.14".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("1e10".to_string(), HighlightKind::Number)));
    }

    #[test]
    fn long_string_single_line() {
        let k = kinds("local s = [[hello world]]");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t == "[[hello world]]"));
    }

    #[test]
    fn long_string_with_level_disambiguates_close() {
        // `]]` inside a `[=[...]=]` must NOT close it -- only `]=]` does.
        let k = kinds("local s = [=[ has a ]] fake close ]=]");
        assert!(
            k.iter().any(|(t, kind)| *kind == HighlightKind::String
                && t == "[=[ has a ]] fake close ]=]")
        );
    }

    #[test]
    fn long_comment_spans_multiple_lines() {
        let mut h = Highlighter::new(Language::Lua);
        let l1 = h.highlight_line("local x = 1 --[[ start of comment");
        assert!(l1.iter().any(|s| s.kind == HighlightKind::Comment));
        assert_ne!(h.state(), crate::LineState::initial(Language::Lua));

        let line2 = "still inside, `local` here is not a keyword";
        let l2 = h.highlight_line(line2);
        assert_eq!(l2.len(), 1);
        assert_eq!(l2[0].kind, HighlightKind::Comment);
        assert_eq!(l2[0].start, 0);
        assert_eq!(l2[0].end, line2.len());

        let l3 = h.highlight_line("end of comment ]] local y = 2");
        assert!(l3.iter().any(|s| s.kind == HighlightKind::Comment));
        assert!(l3.iter().any(|s| s.kind == HighlightKind::Keyword));
        assert_eq!(h.state(), crate::LineState::initial(Language::Lua));
    }

    #[test]
    fn long_string_spans_multiple_lines_with_level() {
        let mut h = Highlighter::new(Language::Lua);
        h.highlight_line("local s = [==[ line one");
        assert!(matches!(h.state(), crate::LineState::Lua(LuaState::LongString { level: 2 })));

        // A `]]` (wrong level) in the middle must not close it.
        h.highlight_line("line two has ]] but not the real close");
        assert!(matches!(h.state(), crate::LineState::Lua(LuaState::LongString { level: 2 })));

        let l3 = h.highlight_line("line three ]==]");
        assert!(l3.iter().any(|s| s.kind == HighlightKind::String));
        assert_eq!(h.state(), crate::LineState::initial(Language::Lua));
    }

    /// Regression test: see the identical Rust-lexer test for why this
    /// matters -- closing on the line's last byte must not be confused
    /// with running off the end unterminated.
    #[test]
    fn string_closing_at_the_very_end_of_the_line_is_not_left_open() {
        let mut h = Highlighter::new(Language::Lua);
        h.highlight_line(r#"local s = "hello""#);
        assert_eq!(h.state(), crate::LineState::initial(Language::Lua));
    }

    #[test]
    fn pathological_input_does_not_hang_or_panic() {
        let long_line = "y".repeat(200_000);
        let _ = Highlighter::new(Language::Lua).highlight_line(&long_line);

        let mut h = Highlighter::new(Language::Lua);
        let unterminated = format!(r#"local s = "{}"#, "a".repeat(100_000));
        let spans = h.highlight_line(&unterminated);
        assert!(spans.iter().any(|s| s.kind == HighlightKind::String));
        assert_ne!(h.state(), crate::LineState::initial(Language::Lua));
    }
}
