//! Markdown lexer.
//!
//! The multi-line hazard is the fenced code block: ` ```lang ` ... ` ``` `
//! (or `~~~`). Per CommonMark, the closing fence must use the same
//! character as the opener and be at least as long; this lexer tracks
//! both so a stray ` ``` ` of the *wrong* fence character inside a fenced
//! block (rare, but valid — e.g. a ` ~~~ `-fenced block containing
//! Markdown source that itself shows a ` ``` ` example) doesn't
//! prematurely close it.
//!
//! Markdown is otherwise a line-oriented format: headings, list markers
//! and blockquotes are recognised from the start of the line, so most of
//! this lexer does not need the same character-by-character token loop
//! the code-oriented languages use.

use crate::{HighlightKind, HighlightSpan};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarkdownState {
    #[default]
    Normal,
    /// Inside a fenced code block opened with `fence` (`` ` `` or `~`),
    /// `len` characters long — the closing fence must be `fence` repeated
    /// at least `len` times.
    FencedCode { fence: u8, len: u32 },
}

pub fn highlight_line(line: &str, state: &mut MarkdownState) -> Vec<HighlightSpan> {
    let bytes = line.as_bytes();
    let len = bytes.len();

    if let MarkdownState::FencedCode { fence, len: fence_len } = *state {
        if is_closing_fence(line, fence, fence_len) {
            *state = MarkdownState::Normal;
        }
        return vec![span(0, len, HighlightKind::CodeBlock)];
    }

    // Fence opener: up to 3 leading spaces, then 3+ of the same
    // backtick/tilde character (CommonMark's indentation allowance).
    let indent = leading_spaces(line).min(3);
    if let Some((fence, fence_len)) = fence_open(line, indent) {
        *state = MarkdownState::FencedCode { fence, len: fence_len };
        return vec![span(0, len, HighlightKind::CodeBlock)];
    }

    // ATX heading: 1-6 `#` then a space or EOL.
    if atx_heading_hashes(line).is_some() {
        return vec![span(0, len, HighlightKind::Heading)];
    }

    inline_spans(line)
}

fn leading_spaces(line: &str) -> usize {
    line.bytes().take_while(|&b| b == b' ').count()
}

fn fence_open(line: &str, indent: usize) -> Option<(u8, u32)> {
    let bytes = line.as_bytes();
    let fence = *bytes.get(indent)?;
    if fence != b'`' && fence != b'~' {
        return None;
    }
    let run = bytes[indent..].iter().take_while(|&&b| b == fence).count();
    if run < 3 {
        return None;
    }
    // A backtick fence's info string must not itself contain a backtick
    // (CommonMark); a tilde fence has no such restriction. Not enforced
    // here -- best-effort, matching the rest of this crate's philosophy.
    Some((fence, run as u32))
}

fn is_closing_fence(line: &str, fence: u8, min_len: u32) -> bool {
    let indent = leading_spaces(line).min(3);
    let bytes = line.as_bytes();
    if bytes.get(indent) != Some(&fence) {
        return false;
    }
    let run = bytes[indent..].iter().take_while(|&&b| b == fence).count();
    // A closing fence line contains nothing else but the fence itself
    // (and trailing whitespace).
    let rest_is_blank = line[indent + run..].trim().is_empty();
    run as u32 >= min_len && rest_is_blank
}

fn atx_heading_hashes(line: &str) -> Option<u32> {
    let bytes = line.as_bytes();
    let run = bytes.iter().take_while(|&&b| b == b'#').count();
    if run == 0 || run > 6 {
        return None;
    }
    match bytes.get(run) {
        None => Some(run as u32),                // `#` alone, or `######` at EOL.
        Some(b' ') | Some(b'\t') => Some(run as u32),
        _ => None, // e.g. `#foo` is not a heading.
    }
}

/// Scans inline constructs: code spans, bold/italic emphasis, links.
/// Deliberately simple (no nested-emphasis resolution, no reference-style
/// links) — Markdown's full inline grammar is famously ambiguous even in
/// the CommonMark spec; this covers what a theme actually needs to colour
/// differently.
fn inline_spans(line: &str) -> Vec<HighlightSpan> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut spans = Vec::new();
    let mut i = 0usize;

    while i < len {
        let c = bytes[i];

        // Inline code span: `` `code` ``, possibly using a longer run of
        // backticks as the delimiter (CommonMark allows this so the code
        // span's content can itself contain single backticks).
        if c == b'`' {
            let run = bytes[i..].iter().take_while(|&&b| b == b'`').count();
            let delim = &line[i..i + run];
            if let Some(close_rel) = line[i + run..].find(delim) {
                let end = i + run + close_rel + run;
                spans.push(span(i, end, HighlightKind::CodeBlock));
                i = end;
                continue;
            }
        }

        // Bold: `**text**` or `__text__`.
        if (c == b'*' || c == b'_') && bytes.get(i + 1) == Some(&c) {
            let marker = [c, c];
            let marker_str = std::str::from_utf8(&marker).unwrap();
            if let Some(close_rel) = line[i + 2..].find(marker_str) {
                let end = i + 2 + close_rel + 2;
                spans.push(span(i, end, HighlightKind::Emphasis));
                i = end;
                continue;
            }
        }

        // Italic: single `*text*` or `_text_`.
        if (c == b'*' || c == b'_')
            && let Some(close_rel) = line[i + 1..].find(c as char)
        {
            let end = i + 1 + close_rel + 1;
            // Require non-whitespace immediately inside both delimiters,
            // or this matches too eagerly on stray punctuation (e.g. a
            // bare `*` used as a list bullet).
            if close_rel > 0 && bytes.get(i + 1) != Some(&b' ') {
                spans.push(span(i, end, HighlightKind::Emphasis));
                i = end;
                continue;
            }
        }

        // Link: `[text](url)`.
        if c == b'['
            && let Some(close_bracket_rel) = line[i + 1..].find(']')
        {
            let close_bracket = i + 1 + close_bracket_rel;
            if bytes.get(close_bracket + 1) == Some(&b'(')
                && let Some(close_paren_rel) = line[close_bracket + 2..].find(')')
            {
                let end = close_bracket + 2 + close_paren_rel + 1;
                spans.push(span(i, end, HighlightKind::Link));
                i = end;
                continue;
            }
        }

        i += 1;
    }

    spans
}

fn span(start: usize, end: usize, kind: HighlightKind) -> HighlightSpan {
    HighlightSpan { start, end, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Highlighter, Language};

    fn spans(line: &str) -> Vec<HighlightSpan> {
        Highlighter::new(Language::Markdown).highlight_line(line)
    }

    #[test]
    fn atx_heading() {
        let s = spans("## A Heading");
        assert_eq!(s, vec![span(0, "## A Heading".len(), HighlightKind::Heading)]);
    }

    #[test]
    fn hash_without_space_is_not_a_heading() {
        let s = spans("#no-space-here");
        assert!(s.iter().all(|sp| sp.kind != HighlightKind::Heading));
    }

    #[test]
    fn inline_code_span() {
        let line = "Run `cargo test` now";
        let s = spans(line);
        assert!(s.iter().any(|sp| sp.kind == HighlightKind::CodeBlock && &line[sp.start..sp.end] == "`cargo test`"));
    }

    #[test]
    fn bold_and_italic_emphasis() {
        let line = "This is **bold** and *italic* text";
        let s = spans(line);
        assert!(s.iter().any(|sp| sp.kind == HighlightKind::Emphasis && &line[sp.start..sp.end] == "**bold**"));
        assert!(s.iter().any(|sp| sp.kind == HighlightKind::Emphasis && &line[sp.start..sp.end] == "*italic*"));
    }

    #[test]
    fn link_is_tagged() {
        let line = "See [the docs](https://example.com/docs) for more";
        let s = spans(line);
        assert!(s.iter().any(|sp| sp.kind == HighlightKind::Link
            && &line[sp.start..sp.end] == "[the docs](https://example.com/docs)"));
    }

    #[test]
    fn fenced_code_block_spans_multiple_lines_and_hides_markdown_syntax() {
        let mut h = Highlighter::new(Language::Markdown);
        let l1 = h.highlight_line("```rust");
        assert_eq!(l1, vec![span(0, "```rust".len(), HighlightKind::CodeBlock)]);
        assert_ne!(h.state(), crate::LineState::initial(Language::Markdown));

        // Inside the fence, a line that LOOKS like a heading must NOT be
        // treated as one -- this is exactly the "state carries across the
        // line boundary" requirement.
        let l2 = h.highlight_line("# this is Rust code, not a heading");
        assert_eq!(l2.len(), 1);
        assert_eq!(l2[0].kind, HighlightKind::CodeBlock);

        let l3 = h.highlight_line("```");
        assert_eq!(l3, vec![span(0, 3, HighlightKind::CodeBlock)]);
        assert_eq!(h.state(), crate::LineState::initial(Language::Markdown));
    }

    #[test]
    fn wrong_fence_character_does_not_close_the_block() {
        let mut h = Highlighter::new(Language::Markdown);
        h.highlight_line("~~~");
        assert_ne!(h.state(), crate::LineState::initial(Language::Markdown));
        // A backtick fence inside a tilde-fenced block must not close it.
        h.highlight_line("```");
        assert_ne!(h.state(), crate::LineState::initial(Language::Markdown));
        h.highlight_line("~~~");
        assert_eq!(h.state(), crate::LineState::initial(Language::Markdown));
    }

    #[test]
    fn pathological_input_does_not_hang_or_panic() {
        let long_line = "v".repeat(200_000);
        let _ = Highlighter::new(Language::Markdown).highlight_line(&long_line);

        let mut h = Highlighter::new(Language::Markdown);
        h.highlight_line("```");
        let unterminated_fenced = "a".repeat(100_000);
        let s = h.highlight_line(&unterminated_fenced);
        assert_eq!(s[0].kind, HighlightKind::CodeBlock);
    }
}
