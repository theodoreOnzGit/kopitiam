//! Rust lexer.
//!
//! Handles the two constructs that genuinely span multiple lines: nested
//! `/* */` block comments (Rust block comments nest, unlike C's) and both
//! plain and raw string literals left open at end-of-line. Everything else
//! (keywords, numbers, macros, attributes, function-call heuristics) is a
//! single-line concern.

use crate::util::{is_ident_start, scan_ident, scan_number, scan_run, skip_inline_whitespace};
use crate::{HighlightKind, HighlightSpan};

/// What the *previous* line left open. `Normal` is both the initial state
/// and the state after any line that closes everything it opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RustState {
    #[default]
    Normal,
    /// Inside a `/* ... */` comment, `depth` levels deep (Rust nests these,
    /// so `/* outer /* inner */ still commented */` is one comment, and
    /// `depth` is how many unmatched `/*` remain open).
    BlockComment { depth: u32 },
    /// Inside an ordinary `"..."` (or `b"..."`) string that did not close
    /// before end-of-line — Rust string literals are allowed to contain a
    /// literal newline, so this is valid, not an error.
    String,
    /// Inside a raw string `r#*"..."` opened with `hashes` `#` characters,
    /// so the closer must be `"` followed by exactly that many `#`s.
    RawString { hashes: u32 },
}

const KEYWORDS: &[&str] = &[
    "as", "async", "await", "box", "break", "const", "continue", "crate", "dyn", "else", "enum",
    "extern", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
    "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true",
    "false", "try", "type", "unsafe", "use", "where", "while", "yield", "union",
];

const PRIMITIVE_TYPES: &[&str] = &[
    "bool", "char", "str", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16",
    "u32", "u64", "u128", "usize",
];

const OPERATOR_CHARS: &str = "=!<>&|+-*/%^.:?~@";
const PUNCTUATION_CHARS: &str = "(){}[],;#";

pub fn highlight_line(line: &str, state: &mut RustState) -> Vec<HighlightSpan> {
    let mut spans = Vec::new();
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    // Resume whatever the previous line left open before doing anything
    // else -- these branches may consume the whole line and return early.
    match *state {
        RustState::Normal => {}
        RustState::BlockComment { mut depth } => {
            let start = i;
            while i < len {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        spans.push(span(start, i, HighlightKind::Comment));
                        *state = RustState::Normal;
                        return finish(line, i, spans, state);
                    }
                } else {
                    i += 1;
                }
            }
            spans.push(span(start, len, HighlightKind::Comment));
            *state = RustState::BlockComment { depth };
            return spans;
        }
        RustState::String => match scan_string_body(line, 0, &mut spans) {
            Some(end) => {
                *state = RustState::Normal;
                return finish(line, end, spans, state);
            }
            None => {
                *state = RustState::String;
                return spans;
            }
        },
        RustState::RawString { hashes } => {
            let start = i;
            if let Some(close_end) = find_raw_string_close(line, i, hashes) {
                spans.push(span(start, close_end, HighlightKind::String));
                *state = RustState::Normal;
                return finish(line, close_end, spans, state);
            }
            spans.push(span(start, len, HighlightKind::String));
            return spans;
        }
    }

    finish(line, i, spans, state)
}

/// Scans forward from byte offset `i` in `line` in `RustState::Normal`,
/// tokenizing until either the line ends or a multi-line construct is
/// opened (at which point `*state` is set accordingly and scanning stops
/// for this line, matching every other branch's contract of "state
/// reflects what's still open at end-of-line").
fn finish(line: &str, mut i: usize, mut spans: Vec<HighlightSpan>, state: &mut RustState) -> Vec<HighlightSpan> {
    let bytes = line.as_bytes();
    let len = bytes.len();

    while i < len {
        let c = bytes[i];

        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }

        // Line comment: everything to EOL, including `///` doc comments.
        if bytes[i..].starts_with(b"//") {
            spans.push(span(i, len, HighlightKind::Comment));
            return spans;
        }

        // Block comment start.
        if bytes[i..].starts_with(b"/*") {
            let start = i;
            let mut depth = 1u32;
            i += 2;
            while i < len {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            spans.push(span(start, i, HighlightKind::Comment));
            if depth > 0 {
                *state = RustState::BlockComment { depth };
                return spans;
            }
            continue;
        }

        // Raw string: `r`, `br`, `rb` followed by zero-or-more `#` and `"`.
        if let Some(prefix_len) = raw_string_prefix_len(line, i) {
            let hash_start = i + prefix_len;
            let hashes = count_hashes(line, hash_start);
            let quote_pos = hash_start + hashes;
            if bytes.get(quote_pos) == Some(&b'"') {
                let start = i;
                let body_start = quote_pos + 1;
                if let Some(close_end) = find_raw_string_close(line, body_start, hashes as u32) {
                    spans.push(span(start, close_end, HighlightKind::String));
                    i = close_end;
                    continue;
                }
                spans.push(span(start, len, HighlightKind::String));
                *state = RustState::RawString { hashes: hashes as u32 };
                return spans;
            }
        }

        // Ordinary / byte string.
        if c == b'"' || (c == b'b' && bytes.get(i + 1) == Some(&b'"')) {
            let start = i;
            let quote_at = if c == b'b' { i + 1 } else { i };
            let body_start = quote_at + 1;
            // Prefix (`b`, if present) plus the opening quote, as its own
            // span, so the tiling `scan_string_body` produces from
            // `body_start` onward stays contiguous and non-overlapping.
            spans.push(span(start, body_start, HighlightKind::String));
            match scan_string_body(line, body_start, &mut spans) {
                Some(end) => {
                    i = end;
                    continue;
                }
                None => {
                    *state = RustState::String;
                    return spans;
                }
            }
        }

        // Char literal / lifetime: `'`.
        if c == b'\'' {
            if let Some(end) = scan_char_literal(line, i) {
                spans.push(span(i, end, HighlightKind::String));
                i = end;
                continue;
            }
            // Not a char literal (e.g. a lifetime `'a`) -- leave
            // unhighlighted, see the crate docs on HighlightKind::Type.
            i += 1;
            continue;
        }

        // Attribute: `#[...]` or `#![...]`.
        if c == b'#' {
            let attr_start = i;
            let mut j = i + 1;
            if bytes.get(j) == Some(&b'!') {
                j += 1;
            }
            if bytes.get(j) == Some(&b'[') {
                let mut depth = 0u32;
                while j < len {
                    match bytes[j] {
                        b'[' => {
                            depth += 1;
                            j += 1;
                        }
                        b']' => {
                            depth -= 1;
                            j += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => j += 1,
                    }
                }
                spans.push(span(attr_start, j, HighlightKind::Attribute));
                i = j;
                continue;
            }
            spans.push(span(i, i + 1, HighlightKind::Punctuation));
            i += 1;
            continue;
        }

        if c.is_ascii_digit() {
            let end = scan_number(line, i);
            spans.push(span(i, end, HighlightKind::Number));
            i = end;
            continue;
        }

        if is_ident_start(c as char) || !c.is_ascii() {
            let ch = line[i..].chars().next().unwrap();
            let end = scan_ident(line, i);
            if end == i {
                // Non-ASCII, non-identifier character (rare) -- treat as
                // a single opaque unhighlighted char, not an infinite loop.
                i += ch.len_utf8();
                continue;
            }
            let word = &line[i..end];
            // Macro invocation: `name!`.
            let after_ws = skip_inline_whitespace(line, end);
            let is_macro = bytes.get(after_ws) == Some(&b'!') && bytes.get(after_ws + 1) != Some(&b'=');
            let is_call = bytes.get(after_ws) == Some(&b'(');

            let kind = if KEYWORDS.contains(&word) {
                HighlightKind::Keyword
            } else if PRIMITIVE_TYPES.contains(&word) {
                HighlightKind::Type
            } else if is_macro {
                HighlightKind::Macro
            } else if is_call {
                HighlightKind::Function
            } else if word.starts_with(|c: char| c.is_uppercase()) {
                HighlightKind::Type
            } else {
                i = end;
                continue; // plain identifier: no span, default rendering.
            };
            spans.push(span(i, end, kind));
            i = end;
            continue;
        }

        // Operators / punctuation.
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

/// Scans a `"..."` string body starting at `start` (the first byte after
/// the opening quote, or byte 0 when resuming a string left open by the
/// previous line), pushing non-overlapping `String` and `Escape` spans
/// that tile the body exactly -- a literal run, then an `Escape` span for
/// each `\x` sequence, repeated, with the closing quote folded into the
/// final `String` run. Returns the offset just past the closing quote, or
/// `line.len()` if the string runs off the end of the line unterminated
/// (the multi-line case).
///
/// Spans are deliberately non-overlapping (rather than one big `String`
/// span with `Escape` spans layered inside it) so a renderer can walk the
/// returned `Vec` in order and apply exactly one style per byte, with no
/// "which span wins" ambiguity.
///
/// Returns `Some(end)` if a closing quote was found (`end` may equal
/// `line.len()` when the string closes on the very last byte of the line
/// -- that is still "closed," not "unterminated"), or `None` if the body
/// ran off the end of the line with no closing quote. Callers must branch
/// on this `Option`, not on `end == line.len()`: those two outcomes both
/// legitimately reach `line.len()` and are only distinguishable through
/// this return value.
fn scan_string_body(line: &str, start: usize, spans: &mut Vec<HighlightSpan>) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start;
    let mut run_start = start;
    while i < len {
        match bytes[i] {
            b'\\' if i + 1 < len => {
                if run_start < i {
                    spans.push(span(run_start, i, HighlightKind::String));
                }
                spans.push(span(i, i + 2, HighlightKind::Escape));
                i += 2;
                run_start = i;
            }
            b'"' => {
                spans.push(span(run_start, i + 1, HighlightKind::String));
                return Some(i + 1);
            }
            _ => i += 1,
        }
    }
    if run_start < len {
        spans.push(span(run_start, len, HighlightKind::String));
    }
    None
}

/// Scans a `'x'` or `'\n'` char literal starting at the opening `'`.
/// Returns `None` (not a char literal -- likely a lifetime) if the
/// content between `'` and the next `'` isn't exactly one char or escape.
fn scan_char_literal(line: &str, start: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = start + 1;
    if i >= len {
        return None;
    }
    if bytes[i] == b'\\' {
        i += 1;
        // Consume the escape body up to (but not including) the closing
        // quote; unicode escapes like `\u{1F600}` are variable-length.
        while i < len && bytes[i] != b'\'' {
            i += 1;
        }
    } else {
        i += line[i..].chars().next()?.len_utf8();
    }
    if bytes.get(i) == Some(&b'\'') { Some(i + 1) } else { None }
}

/// If `line[i..]` starts with a raw-string prefix (`r`, `br`, `rb`)
/// immediately followed by `#`s and a `"`, returns the length of the
/// prefix (1 for `r`, 2 for `br`/`rb`) -- not including the hashes or
/// quote, which the caller scans separately.
fn raw_string_prefix_len(line: &str, i: usize) -> Option<usize> {
    let rest = line.as_bytes().get(i..)?;
    if rest.starts_with(b"br") || rest.starts_with(b"rb") {
        Some(2)
    } else if rest.starts_with(b"r") {
        Some(1)
    } else {
        None
    }
}

fn count_hashes(line: &str, start: usize) -> usize {
    line.as_bytes()[start..].iter().take_while(|&&b| b == b'#').count()
}

/// From `body_start` (just past the opening `"`), finds the end of a raw
/// string requiring exactly `hashes` `#` characters after the closing `"`.
/// Returns the offset just past the close, or `None` if not found on this
/// line.
fn find_raw_string_close(line: &str, body_start: usize, hashes: u32) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = body_start;
    while i < len {
        if bytes[i] == b'"' {
            let hash_end = i + 1 + hashes as usize;
            if hash_end <= len && bytes[i + 1..hash_end].iter().all(|&b| b == b'#') {
                return Some(hash_end);
            }
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

    fn spans_of(line: &str) -> Vec<HighlightSpan> {
        Highlighter::new(Language::Rust).highlight_line(line)
    }

    fn kinds(line: &str) -> Vec<(String, HighlightKind)> {
        spans_of(line).into_iter().map(|s| (line[s.start..s.end].to_string(), s.kind)).collect()
    }

    #[test]
    fn keywords_are_tagged() {
        let k = kinds("fn main() {}");
        assert!(k.contains(&("fn".to_string(), HighlightKind::Keyword)));
        assert!(k.contains(&("main".to_string(), HighlightKind::Function)));
    }

    #[test]
    fn strings_and_escapes() {
        let k = kinds(r#"let s = "hi\nthere";"#);
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t.contains("hi")));
        assert!(k.contains(&(r"\n".to_string(), HighlightKind::Escape)));
    }

    #[test]
    fn line_comment_runs_to_eol() {
        let k = kinds("let x = 1; // trailing comment");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Comment && t == "// trailing comment"));
    }

    #[test]
    fn numbers_with_suffix_and_hex() {
        let k = kinds("let a = 1u32; let b = 0xFFu8; let c = 1.5f64;");
        assert!(k.contains(&("1u32".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("0xFFu8".to_string(), HighlightKind::Number)));
        assert!(k.contains(&("1.5f64".to_string(), HighlightKind::Number)));
    }

    #[test]
    fn function_definitions_and_calls() {
        let k = kinds("fn add(a: i32) -> i32 { helper(a) }");
        assert!(k.contains(&("add".to_string(), HighlightKind::Function)));
        assert!(k.contains(&("helper".to_string(), HighlightKind::Function)));
        assert!(k.contains(&("i32".to_string(), HighlightKind::Type)));
    }

    #[test]
    fn types_are_upper_camel_case() {
        let k = kinds("let v: Vec<Option<String>> = Vec::new();");
        assert!(k.contains(&("Vec".to_string(), HighlightKind::Type)));
        assert!(k.contains(&("Option".to_string(), HighlightKind::Type)));
        assert!(k.contains(&("String".to_string(), HighlightKind::Type)));
    }

    #[test]
    fn macro_invocation_is_tagged() {
        let k = kinds(r#"println!("hi {}", name);"#);
        assert!(k.contains(&("println".to_string(), HighlightKind::Macro)));
    }

    #[test]
    fn attribute_is_tagged() {
        let k = kinds("#[derive(Debug, Clone)]");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::Attribute && t == "#[derive(Debug, Clone)]"));
    }

    #[test]
    fn raw_string_single_line() {
        let k = kinds(r##"let s = r#"hello "world""#;"##);
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t == r##"r#"hello "world""#"##));
    }

    #[test]
    fn block_comment_spans_multiple_lines() {
        let mut h = Highlighter::new(Language::Rust);
        let l1 = h.highlight_line("let x = 1; /* start of comment");
        assert!(l1.iter().any(|s| s.kind == HighlightKind::Comment));
        assert_ne!(h.state(), crate::LineState::initial(Language::Rust));

        let l2 = h.highlight_line("still inside the comment, no code here");
        // The ENTIRE second line must be one comment span -- this is
        // exactly the classic line-based-highlighter bug: without carried
        // state, line 2 would come back with zero spans (or worse, treat
        // its words as keywords/identifiers).
        assert_eq!(l2.len(), 1);
        assert_eq!(l2[0].kind, HighlightKind::Comment);
        assert_eq!(l2[0].start, 0);
        assert_eq!(l2[0].end, "still inside the comment, no code here".len());

        let l3 = h.highlight_line("end of comment */ let y = 2;");
        assert!(l3.iter().any(|s| s.kind == HighlightKind::Comment));
        assert!(l3.iter().any(|s| s.kind == HighlightKind::Keyword));
        assert_eq!(h.state(), crate::LineState::initial(Language::Rust));
    }

    #[test]
    fn nested_block_comments() {
        let mut h = Highlighter::new(Language::Rust);
        h.highlight_line("/* outer /* inner */ still in outer");
        assert_ne!(h.state(), crate::LineState::initial(Language::Rust));
        h.highlight_line("close outer */ let z = 1;");
        assert_eq!(h.state(), crate::LineState::initial(Language::Rust));
    }

    #[test]
    fn multi_line_plain_string() {
        let mut h = Highlighter::new(Language::Rust);
        let l1 = h.highlight_line(r#"let s = "line one"#);
        assert!(l1.iter().any(|s| s.kind == HighlightKind::String));
        assert_ne!(h.state(), crate::LineState::initial(Language::Rust));

        let l2 = h.highlight_line(r#"line two ends here";"#);
        assert!(l2.iter().any(|s| s.kind == HighlightKind::String && s.start == 0));
        assert_eq!(h.state(), crate::LineState::initial(Language::Rust));
    }

    #[test]
    fn multi_line_raw_string_with_hashes() {
        let mut h = Highlighter::new(Language::Rust);
        h.highlight_line(r##"let s = r#"line one"##);
        assert!(matches!(h.state(), crate::LineState::Rust(RustState::RawString { hashes: 1 })));
        // A bare `"` (no matching `#`) on the middle line must NOT close it.
        h.highlight_line(r#"has a " quote but not the real close"#);
        assert!(matches!(h.state(), crate::LineState::Rust(RustState::RawString { hashes: 1 })));
        let l3 = h.highlight_line(r##"line three"#;"##);
        assert!(l3.iter().any(|s| s.kind == HighlightKind::String));
        assert_eq!(h.state(), crate::LineState::initial(Language::Rust));
    }

    #[test]
    fn lifetime_is_not_confused_with_char_literal() {
        let k = kinds("fn f<'a>(x: &'a str) -> &'a str { x }");
        // `'a` should not appear as a String span.
        assert!(!k.iter().any(|(t, kind)| *kind == HighlightKind::String && t.contains('a')));
    }

    #[test]
    fn char_literal_with_escape() {
        let k = kinds(r"let c = '\n';");
        assert!(k.iter().any(|(t, kind)| *kind == HighlightKind::String && t == r"'\n'"));
    }

    /// Regression test: a string (or comment) closing on the exact last
    /// byte of the line must not be mistaken for one that ran off the end
    /// unterminated -- both cases reach `end == line.len()`, and only
    /// looking at *why* the scan stopped (found a closing quote vs. ran
    /// out of bytes) distinguishes them.
    #[test]
    fn string_closing_at_the_very_end_of_the_line_is_not_left_open() {
        let mut h = Highlighter::new(Language::Rust);
        h.highlight_line(r#"let s = "hello""#);
        assert_eq!(h.state(), crate::LineState::initial(Language::Rust));

        // Same check resuming mid-string: the closing quote is the last
        // byte of the continuation line.
        let mut h2 = Highlighter::new(Language::Rust);
        h2.highlight_line(r#"let s = "hello"#);
        assert_ne!(h2.state(), crate::LineState::initial(Language::Rust));
        h2.highlight_line("world\"");
        assert_eq!(h2.state(), crate::LineState::initial(Language::Rust));
    }

    /// Pathological input must not hang or panic: a very long line and an
    /// unterminated string.
    #[test]
    fn pathological_input_does_not_hang_or_panic() {
        let long_line = "x".repeat(200_000);
        let _ = spans_of(&long_line);

        let mut h = Highlighter::new(Language::Rust);
        let unterminated = format!(r#"let s = "{}"#, "a".repeat(100_000));
        let spans = h.highlight_line(&unterminated);
        assert!(spans.iter().any(|s| s.kind == HighlightKind::String));
        assert_ne!(h.state(), crate::LineState::initial(Language::Rust));
    }
}
