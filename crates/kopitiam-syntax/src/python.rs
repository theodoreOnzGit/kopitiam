//! Python lexer.
//!
//! The multi-line hazard is the triple-quoted string (`"""..."""` /
//! `'''...'''`), used pervasively for docstrings. Unlike Rust and Lua,
//! Python's *single*-quoted strings deliberately do **not** carry across a
//! line boundary (an unterminated `"..."` is a syntax error in real
//! Python, full stop) so this lexer does not treat them as open state --
//! only the triple-quoted form does.

use crate::util::{is_ident_start, scan_ident, scan_number, scan_run, skip_inline_whitespace};
use crate::{HighlightKind, HighlightSpan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PythonState {
    #[default]
    Normal,
    /// Inside a `"""..."""` or `'''...'''` string (`quote` is `"` or `'`).
    TripleString { quote: u8 },
}

const KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
    "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
];

const BUILTIN_TYPES: &[&str] = &[
    "bool", "bytes", "complex", "dict", "float", "frozenset", "int", "list", "object", "set",
    "str", "tuple", "type",
];

const OPERATOR_CHARS: &str = "=!<>+-*/%^&|~:.";
const PUNCTUATION_CHARS: &str = "(){}[],;";

pub fn highlight_line(line: &str, state: &mut PythonState) -> Vec<HighlightSpan> {
    let mut spans = Vec::new();
    let len = line.len();

    if let PythonState::TripleString { quote } = *state {
        if let Some(end) = find_triple_close(line, 0, quote) {
            spans.push(span(0, end, HighlightKind::String));
            *state = PythonState::Normal;
            return finish(line, end, spans, state);
        }
        spans.push(span(0, len, HighlightKind::String));
        return spans;
    }

    finish(line, 0, spans, state)
}

fn finish(line: &str, mut i: usize, mut spans: Vec<HighlightSpan>, state: &mut PythonState) -> Vec<HighlightSpan> {
    let bytes = line.as_bytes();
    let len = bytes.len();

    while i < len {
        let c = bytes[i];

        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }

        if c == b'#' {
            spans.push(span(i, len, HighlightKind::Comment));
            return spans;
        }

        // Decorator: `@name`, only when `@` opens the line (mod leading
        // whitespace) -- otherwise this is the matrix-multiplication
        // operator (`a @ b`), which this lexer doesn't otherwise tag.
        // `spans.is_empty()` alone isn't a reliable proxy for "start of
        // line": tokens this lexer doesn't emit spans for (plain
        // identifiers) leave `spans` empty too, so `a @b` would otherwise
        // be misread as a decorator. Checking the untokenized prefix
        // directly is unambiguous.
        if c == b'@' && line[..i].trim().is_empty() {
            let start = i;
            let ident_end = scan_ident(line, i + 1);
            if ident_end > i + 1 {
                spans.push(span(start, ident_end, HighlightKind::Attribute));
                i = ident_end;
                continue;
            }
        }

        // Allow an optional string-prefix letter (r, b, f, u and
        // combinations like rb/fr) directly before the quote.
        let prefix_len = string_prefix_len(line, i);
        let quote_pos = i + prefix_len;
        if let Some(&quote) = bytes.get(quote_pos).filter(|b| **b == b'"' || **b == b'\'') {
            let is_raw = line[i..quote_pos].to_ascii_lowercase().contains('r');
            if bytes.get(quote_pos + 1) == Some(&quote) && bytes.get(quote_pos + 2) == Some(&quote) {
                // Triple-quoted.
                let start = i;
                let body_start = quote_pos + 3;
                if let Some(end) = find_triple_close(line, body_start, quote) {
                    spans.push(span(start, end, HighlightKind::String));
                    i = end;
                    continue;
                }
                spans.push(span(start, len, HighlightKind::String));
                *state = PythonState::TripleString { quote };
                return spans;
            }
            // Single-quoted: does NOT carry across lines (see module docs).
            let start = i;
            spans.push(span(start, quote_pos + 1, HighlightKind::String));
            let body_start = quote_pos + 1;
            let end = if is_raw {
                scan_raw_quoted_body(line, body_start, quote, &mut spans)
            } else {
                scan_quoted_body(line, body_start, quote, &mut spans)
            };
            i = end;
            continue;
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
            // `def name(` and a plain call `name(` share the same
            // "identifier immediately followed by `(`" shape, so one
            // heuristic covers both definition and call sites.
            let is_call = bytes.get(after_ws) == Some(&b'(');

            let kind = if KEYWORDS.contains(&word) {
                HighlightKind::Keyword
            } else if BUILTIN_TYPES.contains(&word) {
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

/// Length of a string-literal prefix (`r`, `b`, `f`, `u`, or a two-letter
/// combination like `rb`/`fr`) at `i`, or 0 if none. Case-insensitive, per
/// Python's grammar.
fn string_prefix_len(line: &str, i: usize) -> usize {
    let bytes = line.as_bytes();
    let is_prefix_char = |b: u8| matches!(b.to_ascii_lowercase(), b'r' | b'b' | b'f' | b'u');
    match (bytes.get(i), bytes.get(i + 1)) {
        (Some(&a), Some(&b)) if is_prefix_char(a) && is_prefix_char(b) => 2,
        (Some(&a), _) if is_prefix_char(a) => 1,
        _ => 0,
    }
}

fn find_triple_close(line: &str, start: usize, quote: u8) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start;
    while i < len {
        if bytes[i] == quote && bytes.get(i + 1) == Some(&quote) && bytes.get(i + 2) == Some(&quote) {
            return Some(i + 3);
        }
        i += 1;
    }
    None
}

/// Tiles a single-quoted body into non-overlapping `String`/`Escape`
/// spans, same strategy as the Rust and Lua lexers.
fn scan_quoted_body(line: &str, start: usize, quote: u8, spans: &mut Vec<HighlightSpan>) -> usize {
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
            return i + 1;
        } else {
            i += 1;
        }
    }
    if run_start < len {
        spans.push(span(run_start, len, HighlightKind::String));
    }
    len
}

/// Raw-string variant: backslashes have no escaping meaning, so the whole
/// body (up to the closing quote) is one `String` span.
fn scan_raw_quoted_body(line: &str, start: usize, quote: u8, spans: &mut Vec<HighlightSpan>) -> usize {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start;
    while i < len {
        if bytes[i] == quote {
            spans.push(span(start, i + 1, HighlightKind::String));
            return i + 1;
        }
        i += 1;
    }
    if start < len {
        spans.push(span(start, len, HighlightKind::String));
    }
    len
}

fn span(start: usize, end: usize, kind: HighlightKind) -> HighlightSpan {
    HighlightSpan { start, end, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Highlighter, Language};

    fn kinds(line: &str) -> Vec<(String, HighlightKind)> {
        Highlighter::new(Language::Python)
            .highlight_line(line)
            .into_iter()
            .map(|s| (line[s.start..s.end].to_string(), s.kind))
            .collect()
    }

    #[test]
    fn keywords_and_def() {
        let k = kinds("def add(a, b): return helper(a, b)");
        assert!(k.contains(&("def".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("return".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("add".to_string(), HighlightKind::Function)));
        assert!(k.contains(&("helper".to_string(), HighlightKind::Function)));
    }

    #[test]
    fn comment_runs_to_eol() {
        let k = kinds("x = 1  # trailing comment");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Comment && t == "# trailing comment"));
    }

    #[test]
    fn single_quoted_string_with_escape() {
        let k = kinds(r#"s = "hi\nthere""#);
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t == "hi"));
        assert!(k.contains(&(r"\n".to_string(), HighlightKind::Escape)));
    }

    #[test]
    fn raw_string_ignores_escapes() {
        let k = kinds(r#"path = r"C:\new\test""#);
        assert!(!k.iter().any(|(_, kind)| *kind == HighlightKind::Escape));
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t.contains("C:")));
    }

    #[test]
    fn numbers_including_float_and_complex() {
        let k = kinds("a = 0xFF; b = 3.14; c = 2j; d = 1_000");
        assert!(k.contains(&("0xFF".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("3.14".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("2j".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("1_000".to_string(), HighlightKind::Number)));
    }

    #[test]
    fn decorator_is_tagged() {
        let k = kinds("@staticmethod");
        assert!(k.contains(&("@staticmethod".to_string(), HighlightKind::Attribute)));
    }

    #[test]
    fn builtin_types_are_tagged() {
        let k = kinds("x: int = 0");
        assert!(k.contains(&("int".to_string(), HighlightKind::Type)));
    }

    #[test]
    fn single_quoted_string_does_not_carry_across_lines() {
        // Real Python semantics: an unterminated `'...'`/`"..."` is a
        // syntax error, not a continuation -- the next line starts fresh.
        let mut h = Highlighter::new(Language::Python);
        h.highlight_line(r#"s = "unterminated"#);
        assert_eq!(h.state(), crate::LineState::initial(Language::Python));
        let k = h.highlight_line("y = 1");
        assert!(k.iter().any(|s| s.kind == HighlightKind::Number && s.start == 4));
    }

    #[test]
    fn triple_quoted_docstring_spans_multiple_lines() {
        let mut h = Highlighter::new(Language::Python);
        let l1 = h.highlight_line(r#"def f():"#);
        assert!(l1.iter().any(|s| s.kind == HighlightKind::Keyword));
        h.highlight_line(r#"    """Docstring starts here"#);
        assert_ne!(h.state(), crate::LineState::initial(Language::Python));

        let line2 = "    still inside the docstring, `def` is not a keyword here";
        let l2 = h.highlight_line(line2);
        assert_eq!(l2.len(), 1);
        assert_eq!(l2[0].kind, HighlightKind::String);
        assert_eq!(l2[0].start, 0);
        assert_eq!(l2[0].end, line2.len());

        let l3 = h.highlight_line(r#"    end of docstring""" return 1"#);
        assert!(l3.iter().any(|s| s.kind == HighlightKind::String));
        assert!(l3.iter().any(|s| s.kind == HighlightKind::Keyword));
        assert_eq!(h.state(), crate::LineState::initial(Language::Python));
    }

    #[test]
    fn pathological_input_does_not_hang_or_panic() {
        let long_line = "z".repeat(200_000);
        let _ = Highlighter::new(Language::Python).highlight_line(&long_line);

        let mut h = Highlighter::new(Language::Python);
        let unterminated = format!(r#"s = """{}"#, "a".repeat(100_000));
        let spans = h.highlight_line(&unterminated);
        assert!(spans.iter().any(|s| s.kind == HighlightKind::String));
        assert_ne!(h.state(), crate::LineState::initial(Language::Python));
    }
}
