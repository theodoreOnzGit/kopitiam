mod report;

pub use report::{ConversionReport, PageRecovery};

use std::collections::BTreeMap;

use kopitiam_pdf::{Page, TextSpan};

use crate::reconstruction::{collapse_figure_regions, strip_marginalia};
use crate::{Block, Document};

/// Marker text `Figure::render` emits for a figure that has **no** caption (see
/// `kopitiam-markdown`'s `renderer.rs`). It is renderer boilerplate, not
/// content recovered from the source PDF, so [`strip_rendered_markdown_syntax`]
/// removes it before counting. Kept as a literal here rather than a shared
/// constant because `kopitiam-document` does not depend on `kopitiam-markdown`
/// (dependencies flow the other way); if the renderer's wording changes this
/// constant must be updated too, which is a known, cheap-to-miss coupling
/// (`kopitiam_token_max.md` §2.3). A *captioned* figure now renders only its
/// caption (real extracted content), so no placeholder is emitted for it.
const FIGURE_PLACEHOLDER: &str = "[figure]";

/// Compares what was extracted against what was rendered, and tallies the
/// block types found, so every conversion produces an auditable report
/// rather than a silent best-effort guess.
///
/// The headline recovery signal ([`ConversionReport::recovery_ratio`]) is a
/// non-whitespace character count, not a word count -- see that method's
/// rustdoc for why. Word counts are still gathered and reported alongside
/// it as an informational secondary signal (see kopitiam-wwr).
pub fn validate(pages: &[Page], document: &Document, rendered_markdown: &str) -> ConversionReport {
    // Recovery-ratio honesty (`kopitiam_token_max.md` §2.1). Reconstruction
    // deletes running heads, running feet, and bare page numbers, so those
    // characters are legitimately absent from `rendered_markdown`. If the
    // extracted side still counted them, `recovery_ratio` would sink below the
    // 0.98 PASS threshold on every document that has a running head -- masking
    // real content loss behind noise we intended to drop. `strip_marginalia` is
    // a pure, deterministic function of `pages`, and reconstruction ran it over
    // exactly these same pages, so rerunning it here removes byte-for-byte the
    // same spans. The removed header/footer text is then absent from *both*
    // `extracted_chars` and `rendered_chars`, and the ratio measures only body
    // recovery, staying honestly inside [0.98, 1.0].
    //
    // The identical reasoning applies to figure-region collapsing: reconstruction
    // drops scattered diagram-label spans (keeping only the caption), so those
    // labels are legitimately absent from `rendered_markdown`. `collapse_figure_regions`
    // is likewise pure and deterministic in its input pages, and reconstruction
    // ran it over exactly these same (already marginalia-stripped) pages in the
    // same order, so rerunning the same composition here drops byte-for-byte the
    // same label spans -- discounting them from `extracted_chars` too, exactly
    // as I-C does for running heads.
    let stripped = strip_marginalia(pages);
    let stripped = collapse_figure_regions(&stripped);

    let extracted_words = stripped
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
        extracted_chars: extracted_content_chars(&stripped),
        rendered_chars: rendered_content_chars(rendered_markdown),
        headings_found,
        lists_found,
        tables_found,
        citations_found: document.citations.len(),
        per_page: per_page_recovery(&stripped, document, rendered_markdown),
    }
}

/// Attributes character recovery to individual source pages using the
/// `<!-- page N -->` anchors the renderer emits at page boundaries
/// (`kopitiam_token_max.md` §8 card I-B).
///
/// # How pages are numbered
///
/// Both sides use the **same** 1-based page numbering: `reconstruct` labels a
/// block with `page_index + 1` (its position in the input `pages`, see
/// `reconstruction::merge_page_breaks`), and the renderer's anchors carry those
/// same `block_pages` numbers. `strip_marginalia`/`collapse_figure_regions`
/// preserve the one-page-per-input-page mapping, so `stripped[i]` is page
/// `i + 1` — the same number a block on that page records. That alignment is
/// what lets the extracted side (indexed by position) and the rendered side
/// (split on anchors) be compared page-for-page.
///
/// # Honesty (`kopitiam_token_max.md` §2.1)
///
/// The extracted side is the *same* `strip_marginalia` + `collapse_figure_regions`
/// pages the document-wide count uses, summed per page; the rendered side is the
/// *same* `strip_rendered_markdown_syntax` normalization, applied per anchor
/// segment. The anchor lines strip to nothing, so the per-page figures partition
/// the document-wide totals exactly (they sum back to them).
///
/// Returns an empty vec for a document with no page provenance
/// (`block_pages` empty): there is no page to attribute content to, and
/// guessing page 1 for everything would be dishonest.
fn per_page_recovery(
    stripped: &[Page],
    document: &Document,
    rendered_markdown: &str,
) -> Vec<PageRecovery> {
    let Some(&first_page) = document.block_pages.first() else {
        return Vec::new();
    };

    let rendered_by_page = rendered_chars_by_page(rendered_markdown, first_page);

    (0..stripped.len())
        .map(|i| {
            let page = i + 1;
            PageRecovery {
                page,
                extracted_chars: extracted_content_chars(std::slice::from_ref(&stripped[i])),
                rendered_chars: rendered_by_page.get(&page).copied().unwrap_or(0),
            }
        })
        .collect()
}

/// Splits the rendered Markdown on `<!-- page N -->` anchors and returns the
/// non-whitespace content-character count per page number.
///
/// The segment *before* the first anchor belongs to `first_page` (the page the
/// document's first block starts on — no anchor precedes the first page); each
/// segment after an anchor belongs to that anchor's page. Every segment is run
/// through [`strip_rendered_markdown_syntax`] before counting, exactly like the
/// document-wide figure, so the per-page counts sum back to it.
fn rendered_chars_by_page(markdown: &str, first_page: usize) -> BTreeMap<usize, usize> {
    let mut by_page: BTreeMap<usize, usize> = BTreeMap::new();
    let mut current_page = first_page;
    let mut segment = String::new();

    let mut flush = |page: usize, segment: &mut String| {
        let chars = content_char_count(&strip_rendered_markdown_syntax(segment));
        *by_page.entry(page).or_insert(0) += chars;
        segment.clear();
    };

    for line in markdown.lines() {
        if let Some(page) = parse_page_anchor(line.trim()) {
            flush(current_page, &mut segment);
            current_page = page;
        } else {
            segment.push_str(line);
            segment.push('\n');
        }
    }
    flush(current_page, &mut segment);

    by_page
}

/// Parses a `<!-- page N -->` page-boundary anchor line, returning `N`.
///
/// The literal form is **duplicated** from `kopitiam-markdown`'s renderer
/// (`page_anchor`), the same way [`FIGURE_PLACEHOLDER`] is: `kopitiam-document`
/// does not depend on `kopitiam-markdown` (dependencies flow the other way), so
/// the two are kept in sync by tests on both sides rather than a shared
/// constant (`kopitiam_token_max.md` §2.3). If the renderer's anchor wording
/// changes, this must change with it.
fn parse_page_anchor(line: &str) -> Option<usize> {
    line.strip_prefix("<!-- page ")?
        .strip_suffix(" -->")?
        .parse()
        .ok()
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
        if parse_page_anchor(trimmed).is_some() {
            // Page-boundary anchor (`<!-- page N -->`, I-B). The renderer adds it
            // for navigation/citation; it is not content from the source PDF, so
            // it must not inflate `rendered_chars` and push the ratio above 1.0
            // (`kopitiam_token_max.md` §2.1). Stripping it here also makes the
            // per-page split (`rendered_chars_by_page`) sum back to the
            // document-wide total, since the anchor lines count as zero.
            continue;
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
        page_n(1, spans)
    }

    fn page_n(number: usize, spans: Vec<TextSpan>) -> Page {
        Page {
            number,
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
    fn stripped_running_head_keeps_recovery_ratio_honest() {
        // The recovery-ratio trap (kopitiam_token_max.md §2.1). A 4-page doc
        // (enough to engage signature stripping) with a running head in the top
        // zone on every page. Reconstruction drops that head, so the rendered
        // Markdown never contains it. If `validate` still counted the head on
        // the extracted side, the ratio would fall below 0.98 and the doc would
        // FAIL purely for having a running head. Because `validate` reruns the
        // same `strip_marginalia`, the head is discounted on both sides and the
        // ratio stays honestly in [0.98, 1.0].
        let head = "Confidential Draft Running Header";
        let body = "Body sentence carrying the real content of the page.";
        let pages: Vec<Page> = (1..=4)
            .map(|n| Page {
                number: n,
                width: 600.0,
                height: 800.0,
                spans: vec![
                    // y = 770 / 800 -> top 10% zone.
                    span(head, 50.0, 770.0, 300.0, 10.0),
                    span(body, 50.0, 400.0, 400.0, 10.0),
                ],
            })
            .collect();

        // Rendered output is the body only (the head was stripped before render).
        let rendered = format!("{body}\n").repeat(4);
        let document = empty_document(vec![Block::Paragraph(Paragraph {
            text: body.to_string(),
        })]);
        let report = validate(&pages, &document, &rendered);

        assert!(
            report.recovery_ratio() <= 1.0 + 1e-9,
            "ratio must not exceed 1.0, got {}",
            report.recovery_ratio()
        );
        assert!(
            report.recovery_ratio() >= 0.98,
            "discounting the stripped head on the extracted side must keep the \
             ratio within PASS range, got {}",
            report.recovery_ratio()
        );
        assert!(report.passes(), "a doc whose only loss is a running head must PASS");

        // Sanity: the head really was removed from the extracted count. Had it
        // been left in, `extracted_chars` would carry the head's characters
        // (4 x 28 non-whitespace) on top of the body, dropping the ratio well
        // below 0.98.
        let body_chars = content_char_count(body) * 4;
        assert_eq!(
            report.extracted_chars, body_chars,
            "extracted side must count body only, not the stripped head"
        );
    }

    #[test]
    fn collapsed_figure_labels_keep_recovery_ratio_honest() {
        // The recovery-ratio trap (kopitiam_token_max.md §2.1) for I-D. A page
        // with a scattered cloud of short diagram labels anchored to a `Fig. 1`
        // caption. Reconstruction drops the labels and keeps only the caption,
        // so the rendered Markdown is the caption alone. If `validate` still
        // counted the dropped labels on the extracted side, the ratio would
        // crater far below 0.98 and the doc would FAIL purely for containing a
        // diagram. Because `validate` reruns the same `collapse_figure_regions`
        // (after the same `strip_marginalia`), the labels are discounted on both
        // sides and the ratio stays honestly in [0.98, 1.0].
        let caption = "Fig. 1 System architecture of the platform";
        let labels = [
            "Sensor Array", "Data Ingestion", "Message Queue", "Stream Processor",
            "Control Unit", "PID Controller", "Actuator Bank", "Feedback Loop",
            "State Store", "Cache Layer", "Load Balancer", "API Gateway",
        ];
        let mut spans = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            let x = 50.0 + ((i * 97) % 450) as f32;
            let y = 720.0 - (i as f32) * 12.0;
            spans.push(span(label, x, y, 80.0, 10.0));
        }
        spans.push(span(caption, 50.0, 340.0, 400.0, 10.0));
        let pages = vec![page(spans)];

        // Rendered output is the caption only (labels were collapsed away, and a
        // captioned figure now renders just its caption).
        let rendered = format!("{caption}\n");
        let document = empty_document(vec![Block::Figure(crate::Figure {
            caption: Some(caption.to_string()),
            image_path: None,
        })]);
        let report = validate(&pages, &document, &rendered);

        assert!(
            report.recovery_ratio() <= 1.0 + 1e-9,
            "ratio must not exceed 1.0, got {}",
            report.recovery_ratio()
        );
        assert!(
            report.recovery_ratio() >= 0.98,
            "discounting collapsed labels on the extracted side must keep the \
             ratio in PASS range, got {}",
            report.recovery_ratio()
        );
        assert!(report.passes(), "a doc whose only loss is diagram labels must PASS");

        // Sanity: the labels really were removed from the extracted count -- only
        // the caption's characters remain.
        assert_eq!(
            report.extracted_chars,
            content_char_count(caption),
            "extracted side must count the caption only, not the collapsed labels"
        );
    }

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
             [figure]\n\n\
             <!-- page 2 -->\n";
        let stripped = strip_rendered_markdown_syntax(markdown);

        assert!(!stripped.contains('#'));
        assert!(!stripped.contains('|'));
        assert!(!stripped.contains('>'));
        assert!(!stripped.contains("```"));
        // The page-boundary anchor (I-B, §2.1) is renderer scaffolding, not
        // source content, and must be stripped before counting.
        assert!(!stripped.contains("<!-- page"));
        // The caption-less figure marker (I-D shortened it to "[figure]", §2.3)
        // is renderer boilerplate and must be stripped before counting.
        assert!(!stripped.contains("[figure]"));
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

    // -- I-B: page anchors + per-page recovery --

    #[test]
    fn page_anchor_scaffolding_does_not_inflate_recovery() {
        // The §2.1 recovery-ratio trap for I-B: the `<!-- page N -->` anchors the
        // renderer now emits are NOT source content. If they counted toward
        // `rendered_chars`, a two-page doc would report >100% recovery and mask
        // real content loss. Because `strip_rendered_markdown_syntax` drops the
        // anchor lines, the ratio stays honestly in [0.98, 1.0]. NEVER weaken 0.98.
        let body1 = "First page body sentence carrying the real content.";
        let body2 = "Second page body sentence carrying the real content.";
        let pages = vec![
            page_n(1, vec![span(body1, 50.0, 400.0, 400.0, 10.0)]),
            page_n(2, vec![span(body2, 50.0, 400.0, 400.0, 10.0)]),
        ];
        let document = Document {
            title: None,
            metadata: Metadata { source_pages: 2 },
            block_pages: vec![1, 2],
            blocks: vec![
                Block::Paragraph(Paragraph { text: body1.to_string() }),
                Block::Paragraph(Paragraph { text: body2.to_string() }),
            ],
            citations: Vec::new(),
        };
        // Exactly what `render_document` produces: one anchor at the boundary.
        let rendered = format!("{body1}\n\n<!-- page 2 -->\n\n{body2}\n");
        let report = validate(&pages, &document, &rendered);

        assert!(
            report.recovery_ratio() <= 1.0 + 1e-9,
            "page anchors must not push recovery above 100%, got {}",
            report.recovery_ratio()
        );
        assert!(
            report.recovery_ratio() >= 0.98,
            "expected ~100% once anchors are stripped, got {}",
            report.recovery_ratio()
        );
        assert!(report.passes());
    }

    #[test]
    fn per_page_recovery_is_reported_and_sums_to_the_document_wide_figure() {
        // Two full pages plus a third page whose body was truncated in the
        // rendered output (content loss confined to page 3). The per-page split
        // localizes the loss to page 3 while pages 1 and 2 stay ~100%, and the
        // per-page char totals partition the document-wide totals exactly.
        let body1 = "Alpha content on the very first page here.";
        let body2 = "Beta content on the second page as well now.";
        let body3_full = "Gamma content on the third page that is mostly dropped downstream.";
        let body3_rendered = "Gamma content"; // truncated: real loss on page 3
        let pages = vec![
            page_n(1, vec![span(body1, 50.0, 400.0, 400.0, 10.0)]),
            page_n(2, vec![span(body2, 50.0, 400.0, 400.0, 10.0)]),
            page_n(3, vec![span(body3_full, 50.0, 400.0, 400.0, 10.0)]),
        ];
        let document = Document {
            title: None,
            metadata: Metadata { source_pages: 3 },
            block_pages: vec![1, 2, 3],
            blocks: vec![
                Block::Paragraph(Paragraph { text: body1.to_string() }),
                Block::Paragraph(Paragraph { text: body2.to_string() }),
                Block::Paragraph(Paragraph { text: body3_rendered.to_string() }),
            ],
            citations: Vec::new(),
        };
        let rendered =
            format!("{body1}\n\n<!-- page 2 -->\n\n{body2}\n\n<!-- page 3 -->\n\n{body3_rendered}\n");
        let report = validate(&pages, &document, &rendered);

        // One entry per source page, in page order.
        assert_eq!(report.per_page.len(), 3);
        assert_eq!(
            report.per_page.iter().map(|p| p.page).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        // Pages 1 and 2 are fully recovered; page 3 is not.
        assert!(report.per_page[0].recovery_ratio() >= 0.99);
        assert!(report.per_page[1].recovery_ratio() >= 0.99);
        assert!(
            report.per_page[2].recovery_ratio() < 0.5,
            "the truncated page 3 must show a low per-page ratio, got {}",
            report.per_page[2].recovery_ratio()
        );

        // Consistency: the per-page figures partition the document-wide totals.
        let sum_extracted: usize = report.per_page.iter().map(|p| p.extracted_chars).sum();
        let sum_rendered: usize = report.per_page.iter().map(|p| p.rendered_chars).sum();
        assert_eq!(sum_extracted, report.extracted_chars);
        assert_eq!(sum_rendered, report.rendered_chars);

        // The document-wide ratio lies within the span of the per-page ratios
        // (page 3 drags it down without the per-page view, we would not know
        // *which* page).
        let doc_ratio = report.recovery_ratio();
        assert!(doc_ratio < report.per_page[0].recovery_ratio());
        assert!(doc_ratio > report.per_page[2].recovery_ratio());
    }

    #[test]
    fn a_specific_page_anchor_is_literal_and_greppable_in_the_rendered_output() {
        // Acceptance: `rg -n "<!-- page 2 -->"` (conceptually) locates page 2.
        // The anchor is a plain literal string, not an escaped/encoded token.
        let rendered = "Page one.\n\n<!-- page 2 -->\n\nPage two.\n";
        assert!(rendered.contains("<!-- page 2 -->"));
        assert_eq!(parse_page_anchor("<!-- page 2 -->"), Some(2));
        assert_eq!(parse_page_anchor("<!-- page 717 -->"), Some(717));
        // Not every comment-shaped line is an anchor.
        assert_eq!(parse_page_anchor("<!-- note -->"), None);
        assert_eq!(parse_page_anchor("plain text"), None);
    }

    #[test]
    fn a_document_without_page_provenance_reports_no_per_page_breakdown() {
        // No `block_pages` -> no honest page attribution -> empty per_page.
        let document = Document {
            title: None,
            metadata: Metadata::default(),
            block_pages: Vec::new(),
            blocks: vec![Block::Paragraph(Paragraph { text: "Body.".to_string() })],
            citations: Vec::new(),
        };
        let pages = vec![page(vec![span("Body.", 50.0, 400.0, 100.0, 10.0)])];
        let report = validate(&pages, &document, "Body.\n");
        assert!(report.per_page.is_empty());
    }
}
