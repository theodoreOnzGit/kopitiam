//! TOML lexer.
//!
//! The multi-line hazard is the multi-line string forms: `"""..."""`
//! (basic) and `'''...'''` (literal, no escape processing). Everything
//! else in TOML -- keys, `=`, inline tables, arrays that happen to span
//! several lines -- is punctuation this crate doesn't need to track
//! nesting depth for for highlighting purposes (an unbalanced `[` doesn't
//! change how the *next* line's tokens should be coloured, unlike an open
//! string or comment).

use crate::util::scan_number;
use crate::{HighlightKind, HighlightSpan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TomlState {
    #[default]
    Normal,
    /// Inside `"""..."""` (escape sequences are processed, matching a
    /// basic string).
    MultilineBasicString,
    /// Inside `'''...'''` (no escape processing, matching a literal
    /// string).
    MultilineLiteralString,
}

const PUNCTUATION_CHARS: &str = "[]{},.=";

pub fn highlight_line(line: &str, state: &mut TomlState) -> Vec<HighlightSpan> {
    let mut spans = Vec::new();
    let len = line.len();

    match *state {
        TomlState::Normal => {}
        TomlState::MultilineBasicString => match scan_multiline_basic_close(line, 0, &mut spans) {
            Some(end) => {
                *state = TomlState::Normal;
                return finish(line, end, spans, state);
            }
            None => {
                *state = TomlState::MultilineBasicString;
                return spans;
            }
        },
        TomlState::MultilineLiteralString => {
            if let Some(end) = find_triple(line, 0, b'\'') {
                spans.push(span(0, end, HighlightKind::String));
                *state = TomlState::Normal;
                return finish(line, end, spans, state);
            }
            spans.push(span(0, len, HighlightKind::String));
            return spans;
        }
    }

    finish(line, 0, spans, state)
}

fn finish(line: &str, mut i: usize, mut spans: Vec<HighlightSpan>, state: &mut TomlState) -> Vec<HighlightSpan> {
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

        // Table header: `[section]` or `[[array.of.tables]]`, to EOL's
        // closing bracket(s) -- highlighted as a single Type span, the
        // closest existing category to "this names a namespace."
        if c == b'[' && spans.is_empty() && let Some(end) = find_table_header_end(line, i) {
            spans.push(span(i, end, HighlightKind::Type));
            i = end;
            continue;
        }

        if bytes[i..].starts_with(b"\"\"\"") {
            let start = i;
            let body_start = i + 3;
            spans.push(span(start, body_start, HighlightKind::String));
            match scan_multiline_basic_close(line, body_start, &mut spans) {
                Some(end) => {
                    i = end;
                    continue;
                }
                None => {
                    *state = TomlState::MultilineBasicString;
                    return spans;
                }
            }
        }
        if bytes[i..].starts_with(b"'''") {
            let start = i;
            let body_start = i + 3;
            if let Some(end) = find_triple(line, body_start, b'\'') {
                spans.push(span(start, end, HighlightKind::String));
                i = end;
                continue;
            }
            spans.push(span(start, len, HighlightKind::String));
            *state = TomlState::MultilineLiteralString;
            return spans;
        }

        if c == b'"' {
            let start = i;
            spans.push(span(start, start + 1, HighlightKind::String));
            let end = scan_basic_body(line, start + 1, &mut spans);
            i = end;
            continue;
        }
        if c == b'\'' {
            // Literal string: no escapes, runs to the next `'` or EOL.
            let start = i;
            if let Some(end) = memfind(bytes, i + 1, b'\'') {
                spans.push(span(start, end + 1, HighlightKind::String));
                i = end + 1;
            } else {
                spans.push(span(start, len, HighlightKind::String));
                i = len;
            }
            continue;
        }

        if c.is_ascii_digit() || ((c == b'+' || c == b'-') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            let digit_start = if c == b'+' || c == b'-' { i + 1 } else { i };
            let end = scan_number(line, digit_start);
            spans.push(span(i, end, HighlightKind::Number));
            i = end;
            continue;
        }

        if let Some(end) = keyword_at(line, i, "true").or_else(|| keyword_at(line, i, "false")) {
            spans.push(span(i, end, HighlightKind::Keyword));
            i = end;
            continue;
        }

        if PUNCTUATION_CHARS.as_bytes().contains(&c) {
            spans.push(span(i, i + 1, HighlightKind::Punctuation));
            i += 1;
            continue;
        }

        i += 1;
    }

    spans
}

/// Matches a bare keyword (`true`/`false`) at `i`, requiring a non-identifier
/// boundary on both sides so e.g. `truest` doesn't partially match `true`.
fn keyword_at(line: &str, i: usize, word: &str) -> Option<usize> {
    let rest = line.get(i..)?;
    if !rest.starts_with(word) {
        return None;
    }
    let end = i + word.len();
    let boundary_ok = line[end..].chars().next().is_none_or(|c| !c.is_alphanumeric() && c != '_');
    boundary_ok.then_some(end)
}

fn find_table_header_end(line: &str, start: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    let double = bytes.get(start + 1) == Some(&b'[');
    let mut i = start + if double { 2 } else { 1 };
    while i < bytes.len() && bytes[i] != b']' {
        i += 1;
    }
    if bytes.get(i) != Some(&b']') {
        return None;
    }
    i += 1;
    if double {
        if bytes.get(i) != Some(&b']') {
            return None;
        }
        i += 1;
    }
    Some(i)
}

fn find_triple(line: &str, start: usize, quote: u8) -> Option<usize> {
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

/// Tiles a `"""`-delimited body into non-overlapping `String`/`Escape`
/// spans, stopping (and reporting) the closing `"""` as its own trailing
/// String span.
///
/// Returns `Some(end)` if the closer was found (`end` may equal
/// `line.len()` when it lands exactly on the last byte -- still
/// "closed"), `None` if unterminated. See the identical contract on the
/// Rust lexer's `scan_string_body` for why this can't be a plain `usize`.
fn scan_multiline_basic_close(line: &str, start: usize, spans: &mut Vec<HighlightSpan>) -> Option<usize> {
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
        } else if bytes[i] == b'"' && bytes.get(i + 1) == Some(&b'"') && bytes.get(i + 2) == Some(&b'"') {
            spans.push(span(run_start, i + 3, HighlightKind::String));
            return Some(i + 3);
        } else {
            i += 1;
        }
    }
    if run_start < len {
        spans.push(span(run_start, len, HighlightKind::String));
    }
    None
}

/// Tiles a `"..."` basic-string body (single line by TOML's grammar --
/// unlike Rust/Lua, an unterminated basic string does not carry to the
/// next line) into non-overlapping `String`/`Escape` spans.
fn scan_basic_body(line: &str, start: usize, spans: &mut Vec<HighlightSpan>) -> usize {
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
        } else if bytes[i] == b'"' {
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

fn memfind(bytes: &[u8], start: usize, needle: u8) -> Option<usize> {
    bytes[start.min(bytes.len())..].iter().position(|&b| b == needle).map(|p| p + start)
}

fn span(start: usize, end: usize, kind: HighlightKind) -> HighlightSpan {
    HighlightSpan { start, end, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Highlighter, Language};

    fn kinds(line: &str) -> Vec<(String, HighlightKind)> {
        Highlighter::new(Language::Toml)
            .highlight_line(line)
            .into_iter()
            .map(|s| (line[s.start..s.end].to_string(), s.kind))
            .collect()
    }

    #[test]
    fn table_header_is_a_type() {
        let k = kinds("[package]");
        assert!(k.contains(&("[package]".to_string(), HighlightKind::Type)));
        let k2 = kinds("[[bin]]");
        assert!(k2.contains(&("[[bin]]".to_string(), HighlightKind::Type)));
    }

    #[test]
    fn comment_runs_to_eol() {
        let k = kinds("name = \"kopitiam\" # comment");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Comment && t == "# comment"));
    }

    #[test]
    fn basic_string_with_escape() {
        let k = kinds(r#"path = "C:\\new""#);
        assert!(k.contains(&(r"\\".to_string(), HighlightKind::Escape)));
    }

    #[test]
    fn literal_string_ignores_escapes() {
        let k = kinds(r#"path = 'C:\new\test'"#);
        assert!(!k.iter().any(|(_, kind)| *kind == HighlightKind::Escape));
    }

    #[test]
    fn numbers_are_tagged() {
        assert!(kinds("a = 42").contains(&("42".to_string(), HighlightKind::Number)));
        assert!(kinds("b = 3.14").contains(&("3.14".to_string(), HighlightKind::Number)));
    }

    #[test]
    fn booleans_are_keywords() {
        let k = kinds("enabled = true");
        assert!(k.contains(&("true".to_string(), HighlightKind::Keyword)));
    }

    #[test]
    fn multiline_basic_string_spans_multiple_lines() {
        let mut h = Highlighter::new(Language::Toml);
        h.highlight_line(r#"description = """This is"#);
        assert_ne!(h.state(), crate::LineState::initial(Language::Toml));

        let line2 = "a multi-line description";
        let l2 = h.highlight_line(line2);
        assert_eq!(l2.len(), 1);
        assert_eq!(l2[0].kind, HighlightKind::String);

        let l3 = h.highlight_line(r#"that ends here.""""#);
        assert!(l3.iter().any(|s| s.kind == HighlightKind::String));
        assert_eq!(h.state(), crate::LineState::initial(Language::Toml));
    }

    /// Regression test: see the identical Rust-lexer test for why this
    /// matters -- closing on the line's last byte must not be confused
    /// with running off the end unterminated.
    #[test]
    fn multiline_string_closing_at_the_very_end_of_the_line_is_not_left_open() {
        let mut h = Highlighter::new(Language::Toml);
        h.highlight_line(r#"d = """closes right here""""#);
        assert_eq!(h.state(), crate::LineState::initial(Language::Toml));
    }

    #[test]
    fn pathological_input_does_not_hang_or_panic() {
        let long_line = "w".repeat(200_000);
        let _ = Highlighter::new(Language::Toml).highlight_line(&long_line);

        let mut h = Highlighter::new(Language::Toml);
        let unterminated = format!(r#"s = """{}"#, "a".repeat(100_000));
        let spans = h.highlight_line(&unterminated);
        assert!(spans.iter().any(|s| s.kind == HighlightKind::String));
        assert_ne!(h.state(), crate::LineState::initial(Language::Toml));
    }
}
