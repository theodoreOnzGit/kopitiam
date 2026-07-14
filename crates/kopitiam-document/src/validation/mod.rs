mod report;

pub use report::ConversionReport;

use kopitiam_pdf::{Page, TextSpan};

use crate::{Block, Document};

/// Markdown text `Figure::render` emits in place of an image (see
/// `kopitiam-markdown`'s `renderer.rs`). It is renderer boilerplate, not
/// content recovered from the source PDF, so [`strip_rendered_markdown_syntax`]
/// removes it before counting. Kept as a literal here rather than a shared
/// constant because `kopitiam-document` does not depend on `kopitiam-markdown`
/// (dependencies flow the other way); if the renderer's wording changes this
/// constant must be updated too, which is a known, cheap-to-miss coupling.
const FIGURE_PLACEHOLDER: &str = "[Figure omitted from Markdown output]";

/// Compares what was extracted against what was rendered, and tallies the
/// block types found, so every conversion produces an auditable report
/// rather than a silent best-effort guess.
///
/// The headline recovery signal ([`ConversionReport::recovery_ratio`]) is a
/// non-whitespace character count, not a word count -- see that method's
/// rustdoc for why. Word counts are still gathered and reported alongside
/// it as an informational secondary signal (see kopitiam-wwr).
pub fn validate(pages: &[Page], document: &Document, rendered_markdown: &str) -> ConversionReport {
    let extracted_words = pages
        .iter()
        .flat_map(|page| &page.spans)
        .map(|span| word_count(&span.text))
        .sum();

    let mut headings_found = 0;
    let mut lists_found = 0;
    let mut tables_found = 0;

    for block in &document.blocks {
        match block {
            Block::Heading(_) => headings_found += 1,
            Block::List(_) => lists_found += 1,
            Block::Table(_) => tables_found += 1,
            _ => {}
        }
    }

    ConversionReport {
        pages: pages.len(),
        extracted_words,
        rendered_words: word_count(rendered_markdown),
        extracted_chars: extracted_content_chars(pages),
        rendered_chars: rendered_content_chars(rendered_markdown),
        headings_found,
        lists_found,
        tables_found,
        citations_found: document.citations.len(),
    }
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn content_char_count(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

/// Sums non-whitespace characters across every extracted `TextSpan`, on
/// every page, treating a soft line-wrap hyphen (see [`is_wrap_hyphen`]) as
/// not-content so it does not count against recovery once reconstruction
/// repairs it away.
///
/// Non-whitespace characters are used, rather than whitespace-delimited
/// words, because *how* the text is tokenized (one span per word, one span
/// per OCR glyph run, one span per table cell, ...) is an artifact of PDF
/// extraction and has nothing to do with whether any content was lost.
/// Concatenating spans with or without a separating space never changes a
/// non-whitespace character count, so this signal is naturally immune to
/// re-tokenization -- which is exactly the failure mode that made the old
/// word-count ratio unreliable (see kopitiam-wwr).
fn extracted_content_chars(pages: &[Page]) -> usize {
    let mut total = 0;
    for page in pages {
        let spans = &page.spans;
        for (i, span) in spans.iter().enumerate() {
            let text = span.text.as_str();
            let drop_trailing_hyphen = spans
                .get(i + 1)
                .is_some_and(|next| is_wrap_hyphen(span, next));
            let counted = if drop_trailing_hyphen {
                &text[..text.len() - 1]
            } else {
                text
            };
            total += content_char_count(counted);
        }
    }
    total
}

/// True when `current`'s trailing hyphen is a soft line-wrap artifact
/// rather than real content: `next` starts a new visual line -- its `y`
/// differs from `current`'s by more than "same line" tolerance -- and
/// begins with a lowercase letter.
///
/// This mirrors the rule `reconstruction::paragraphs::append_line` uses to
/// repair the same hyphen when assembling prose: a hyphen immediately
/// before a capitalized word (e.g. "Anglo-Saxon") is a real compound and is
/// left as content on both sides of the comparison, while a hyphen at a
/// justified line's right margin followed by the wrapped word's remainder
/// ("develop-" / "ment") is not -- reconstruction deletes that hyphen when
/// it rejoins the word, so counting it on the extracted side would be an
/// artifact mismatch, not lost content.
///
/// The two rules are independent implementations of the same idea rather
/// than shared code: reconstruction operates on already-grouped `Line`s,
/// while this operates directly on raw `TextSpan`s (validation must stay
/// usable even if reconstruction's internal grouping changes). Kept in sync
/// by the hyphenation unit tests below and in `reconstruction::paragraphs`.
fn is_wrap_hyphen(current: &TextSpan, next: &TextSpan) -> bool {
    let ends_with_hyphen = current.text.ends_with('-');
    let same_line_tolerance = current.font_size.max(next.font_size) * 0.4;
    let different_line = (current.y - next.y).abs() > same_line_tolerance;
    let continues_lowercase = next.text.chars().next().is_some_and(char::is_lowercase);
    ends_with_hyphen && different_line && continues_lowercase
}

fn rendered_content_chars(markdown: &str) -> usize {
    content_char_count(&strip_rendered_markdown_syntax(markdown))
}

/// Strips Markdown scaffolding syntax that `kopitiam-markdown`'s renderer
/// adds -- heading hashes, list markers, table pipes and separator rows,
/// blockquote markers, code fences, and the figure-omitted placeholder --
/// before counting rendered content.
///
/// This matters in both directions. If scaffolding were left in, a
/// document with many short table cells could push `recovery_ratio` above
/// 100% (every `|` and `---` the renderer adds counts as "recovered"
/// content that never existed in the source PDF), which would mask real
/// content loss elsewhere in the same document -- the opposite failure
/// mode from the old metric's false FAILs, but just as untrustworthy.
///
/// This is line-oriented, regex-free text surgery rather than a full
/// Markdown parser: it recognizes exactly the small, fixed vocabulary of
/// syntax `kopitiam-markdown`'s renderer (`renderer.rs`) is known to
/// produce, not arbitrary Markdown. It does not attempt to reparse or
/// validate the rendered output.
fn strip_rendered_markdown_syntax(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    for line in markdown.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            continue; // code fence delimiter, not content
        }
        if trimmed == FIGURE_PLACEHOLDER {
            continue; // renderer boilerplate, never present in the source PDF
        }
        if is_table_separator_line(trimmed) {
            continue; // e.g. "| --- | --- |"
        }

        let content = strip_heading_hashes(trimmed);
        let content = content
            .strip_prefix("> ")
            .or_else(|| content.strip_prefix('>'))
            .unwrap_or(content);
        let content = strip_list_marker(content);
        let content = strip_table_pipes(content);

        out.push_str(&content);
        out.push('\n');
    }
    out
}

/// Strips a leading `#`..`######` heading marker (`Heading::render` always
/// emits `"{hashes} {text}"`).
fn strip_heading_hashes(line: &str) -> &str {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ') {
        &line[hashes + 1..]
    } else {
        line
    }
}

/// Strips a leading unordered (`"- "`) or ordered (`"1. "`) list marker
/// (`List::render`'s two branches). A plain-prose line that coincidentally
/// starts the same way (a sentence beginning "12. " or a paragraph opening
/// with an en-dash rendered as `"- "`) is stripped too; this is a known,
/// deliberately conservative false-positive -- it can only ever remove a
/// few characters from the *rendered* side, which pushes the ratio down,
/// never up, so it cannot turn real content loss into a false PASS.
fn strip_list_marker(line: &str) -> &str {
    if let Some(rest) = line.strip_prefix("- ") {
        return rest;
    }
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0
        && let Some(rest) = line[digits..].strip_prefix(". ")
    {
        return rest;
    }
    line
}

/// Strips the leading `"| "` / trailing `" |"` a table row (`render_row`)
/// adds, splits cells on the `" | "` separator it joins them with, and
/// unescapes `"\|"` back to a literal `|` (the escaping `escape_cell`
/// applies to a cell containing a real pipe character, so unescaping keeps
/// that character counted as content rather than discarding it as syntax).
fn strip_table_pipes(line: &str) -> String {
    let Some(inner) = line.strip_prefix("| ").and_then(|s| s.strip_suffix(" |")) else {
        return line.to_string();
    };
    inner
        .split(" | ")
        .map(|cell| cell.replace("\\|", "|"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// True for a table separator row (`render_separator`'s `"| --- | --- |"`):
/// a line made up of only `|`, `-`, `:`, and spaces, containing at least one
/// dash. Real content never renders as a bare line of dashes and pipes, so
/// this cannot misfire on prose.
fn is_table_separator_line(line: &str) -> bool {
    line.starts_with('|')
        && line.ends_with('|')
        && line.contains('-')
        && line.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Heading, Metadata, Paragraph};

    fn span(text: &str, x: f32, y: f32, width: f32, font_size: f32) -> TextSpan {
        TextSpan {
            text: text.to_string(),
            x,
            y,
            width,
            height: font_size,
            font_size,
            font_name: None,
            ..TextSpan::default()
        }
    }

    fn page(spans: Vec<TextSpan>) -> Page {
        Page {
            number: 1,
            width: 600.0,
            height: 800.0,
            spans,
        }
    }

    fn empty_document(blocks: Vec<Block>) -> Document {
        Document {
            title: None,
            metadata: Metadata { source_pages: 1 },
            block_pages: vec![1; blocks.len()],
            blocks,
            citations: Vec::new(),
        }
    }

    // -- headline signal: content genuinely dropped => low ratio => FAIL --

    #[test]
    fn dropped_content_fails() {
        let pages = vec![page(vec![span(
            "This paragraph has plenty of words that never make it into the output.",
            50.0,
            700.0,
            500.0,
            10.0,
        )])];
        let document = empty_document(vec![Block::Paragraph(Paragraph {
            text: "This paragraph has".to_string(),
        })]);
        let report = validate(&pages, &document, "This paragraph has\n");

        assert!(
            report.recovery_ratio() < 0.5,
            "expected a low ratio for dropped content, got {}",
            report.recovery_ratio()
        );
        assert!(!report.passes(), "truncated content must not PASS");
    }

    // -- hyphenation repaired across a line break => still ~100% => PASS --

    #[test]
    fn repaired_hyphenation_still_passes() {
        // Two spans on different lines simulate a justified paragraph where
        // "development" wraps as "develop-" / "ment"; reconstruction joins
        // them back into one word and drops the hyphen.
        let pages = vec![page(vec![
            span("develop-", 50.0, 700.0, 60.0, 10.0),
            span("ment continues steadily.", 50.0, 688.0, 150.0, 10.0),
        ])];
        let document = empty_document(vec![Block::Paragraph(Paragraph {
            text: "development continues steadily.".to_string(),
        })]);
        let report = validate(&pages, &document, "development continues steadily.\n");

        assert!(
            report.recovery_ratio() >= 0.99,
            "hyphenation repair must not be penalized, got {}",
            report.recovery_ratio()
        );
        assert!(report.passes());
    }

    #[test]
    fn a_real_compound_hyphen_is_not_stripped_from_either_side() {
        // "Anglo-" / "Saxon" is a genuine compound, not a line-wrap; both
        // reconstruction and this metric must leave the hyphen as content.
        let pages = vec![page(vec![
            span("Anglo-", 50.0, 700.0, 40.0, 10.0),
            span("Saxon history.", 50.0, 688.0, 90.0, 10.0),
        ])];
        let document = empty_document(vec![Block::Paragraph(Paragraph {
            text: "Anglo-Saxon history.".to_string(),
        })]);
        let report = validate(&pages, &document, "Anglo-Saxon history.\n");

        assert!(
            report.recovery_ratio() >= 0.99,
            "expected ~100%, got {}",
            report.recovery_ratio()
        );
    }

    // -- a table rendered with pipe syntax => pipes don't inflate ratio => PASS --

    #[test]
    fn table_pipe_syntax_does_not_inflate_recovery() {
        let pages = vec![page(vec![
            span("Metric", 50.0, 700.0, 60.0, 10.0),
            span("Value", 200.0, 700.0, 60.0, 10.0),
            span("Speed", 50.0, 688.0, 60.0, 10.0),
            span("42", 200.0, 688.0, 60.0, 10.0),
        ])];
        let document = empty_document(vec![Block::Table(crate::Table {
            headers: vec!["Metric".to_string(), "Value".to_string()],
            rows: vec![vec!["Speed".to_string(), "42".to_string()]],
        })]);
        let rendered = "| Metric | Value |\n| --- | --- |\n| Speed | 42 |\n";
        let report = validate(&pages, &document, rendered);

        assert!(
            report.recovery_ratio() <= 1.0 + 1e-9,
            "table scaffolding must not push recovery above 100%, got {}",
            report.recovery_ratio()
        );
        assert!(
            report.recovery_ratio() >= 0.99,
            "expected ~100% once pipes/separator are stripped, got {}",
            report.recovery_ratio()
        );
        assert!(report.passes());
    }

    // -- OCR word-gap merge ("hel lo" -> "hello") => still PASS --

    #[test]
    fn ocr_word_gap_merge_still_passes() {
        // "Boo" and "k" simulate an OCR text layer that split one word
        // into two spans (see reconstruction::group_lines); reconstruction
        // reads them back as "Book" with no space.
        let pages = vec![page(vec![
            span("Boo", 50.0, 700.0, 18.0, 10.0),
            span("k", 68.2, 700.0, 6.0, 10.0),
            span("Reviews", 78.0, 700.0, 50.0, 10.0),
        ])];
        let document = empty_document(vec![Block::Heading(Heading {
            level: 1,
            text: "Book Reviews".to_string(),
        })]);
        let report = validate(&pages, &document, "# Book Reviews\n");

        assert!(
            report.recovery_ratio() >= 0.99,
            "word-gap re-tokenization must not be penalized, got {}",
            report.recovery_ratio()
        );
        assert!(report.passes());
    }

    // -- normalization building blocks --

    #[test]
    fn strip_rendered_markdown_syntax_removes_all_known_scaffolding() {
        let markdown = "# Title\n\n\
             Body paragraph.\n\n\
             - First item\n\
             1. Ordered item\n\n\
             | A | B |\n\
             | --- | --- |\n\
             | 1 | 2 |\n\n\
             > Quoted line\n\n\
             ```rust\n\
             fn main() {}\n\
             ```\n\n\
             Caption text.\n\n\
             [Figure omitted from Markdown output]\n";
        let stripped = strip_rendered_markdown_syntax(markdown);

        assert!(!stripped.contains('#'));
        assert!(!stripped.contains('|'));
        assert!(!stripped.contains('>'));
        assert!(!stripped.contains("```"));
        assert!(!stripped.contains("[Figure omitted"));
        assert!(stripped.contains("Title"));
        assert!(stripped.contains("Body paragraph."));
        assert!(stripped.contains("First item"));
        assert!(stripped.contains("Ordered item"));
        assert!(stripped.contains("Quoted line"));
        assert!(stripped.contains("fn main() {}"));
        assert!(stripped.contains("Caption text."));
    }

    #[test]
    fn table_pipes_are_stripped_but_a_literal_pipe_in_a_cell_survives_unescaped() {
        assert_eq!(strip_table_pipes("| A | B |"), "A B");
        assert_eq!(strip_table_pipes("| A\\|B | C |"), "A|B C");
    }

    #[test]
    fn empty_extraction_reports_full_recovery_by_convention() {
        let report = validate(&[], &empty_document(vec![]), "");
        assert_eq!(report.recovery_ratio(), 1.0);
        assert!(report.passes());
    }
}
