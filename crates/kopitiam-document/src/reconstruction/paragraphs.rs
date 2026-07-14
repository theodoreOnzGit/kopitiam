use super::Line;
use crate::Paragraph;

const PARAGRAPH_BREAK_GAP_RATIO: f32 = 1.8;

/// Merges consecutive lines into one paragraph, repairing end-of-line
/// hyphenation, until a vertical gap suggests a paragraph break (or the
/// slice runs out). Always consumes at least one line.
pub(super) fn consume_paragraph(lines: &[Line]) -> (Paragraph, usize) {
    let mut text = String::new();
    let mut consumed = 0;
    let mut prev_line: Option<&Line> = None;

    for line in lines {
        if let Some(prev) = prev_line {
            let gap = prev.y - line.y;
            if gap > prev.font_size * PARAGRAPH_BREAK_GAP_RATIO {
                break;
            }
        }

        append_line(&mut text, &line.text);
        consumed += 1;
        prev_line = Some(line);
    }

    (Paragraph { text }, consumed.max(1))
}

fn append_line(text: &mut String, line: &str) {
    let line = line.trim();
    let ends_with_hyphen = text.ends_with('-');
    let continues_word = line.chars().next().is_some_and(|c| c.is_lowercase());

    if ends_with_hyphen && continues_word {
        text.pop();
        text.push_str(line);
        return;
    }

    if !text.is_empty() {
        text.push(' ');
    }
    text.push_str(line);
}

/// Decides whether a trailing paragraph at the bottom of one page and a
/// leading paragraph at the top of the next are really one paragraph that a
/// page break split in two (see kopitiam-d3n), and if so, produces the
/// merged text.
///
/// `reconstruct` processes each page independently, so page-break splitting
/// is invisible to `consume_paragraph` -- this runs afterwards as a second
/// pass over the block stream. A missed merge just leaves two blocks a
/// reader can still visually connect across the page turn; a false merge
/// permanently glues two unrelated paragraphs together with no signal left
/// downstream to undo it. So this insists on two independent positive
/// signals rather than either alone: the trailing paragraph must not end a
/// sentence (a real paragraph break, coincidentally at a page boundary,
/// almost always ends in terminal punctuation), *and* the leading paragraph
/// must open lowercase (a new paragraph almost always opens with a
/// capitalized word or a numeral). A trailing hyphen is handled by the same
/// `append_line` logic that already repairs end-of-line hyphenation within a
/// page -- a word broken across a page boundary is the same phenomenon as
/// one broken across a line, just at a coarser granularity -- so it is not
/// treated as a separate signal here.
pub(super) fn merge_across_page_break(trailing: &str, leading: &str) -> Option<String> {
    let trailing = trailing.trim_end();
    let leading = leading.trim_start();

    if trailing.is_empty() || leading.is_empty() {
        return None;
    }

    let leading_starts_lowercase = leading.chars().next().is_some_and(|c| c.is_lowercase());
    if !leading_starts_lowercase {
        return None;
    }

    let trailing_ends_sentence = matches!(trailing.chars().last(), Some('.' | '!' | '?'));
    if trailing_ends_sentence {
        return None;
    }

    let mut merged = trailing.to_string();
    append_line(&mut merged, leading);
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, y: f32, font_size: f32) -> Line {
        Line {
            text: text.to_string(),
            y,
            font_size,
            cells: Vec::new(),
        }
    }

    #[test]
    fn repairs_end_of_line_hyphenation() {
        let lines = vec![
            line("develop-", 100.0, 10.0),
            line("ment continues.", 88.0, 10.0),
        ];
        let (paragraph, consumed) = consume_paragraph(&lines);
        assert_eq!(paragraph.text, "development continues.");
        assert_eq!(consumed, 2);
    }

    #[test]
    fn does_not_dehyphenate_across_a_capitalized_word() {
        // A trailing hyphen followed by a capitalized word is a real hyphen
        // (e.g. a compound name), not a wrapped word.
        let lines = vec![
            line("Anglo-", 100.0, 10.0),
            line("Saxon history.", 88.0, 10.0),
        ];
        let (paragraph, _) = consume_paragraph(&lines);
        assert_eq!(paragraph.text, "Anglo- Saxon history.");
    }

    #[test]
    fn stops_at_a_large_vertical_gap() {
        let lines = vec![
            line("First paragraph.", 100.0, 10.0),
            line("Still first paragraph.", 88.0, 10.0),
            line("Second paragraph after a big gap.", 50.0, 10.0),
        ];
        let (paragraph, consumed) = consume_paragraph(&lines);
        assert_eq!(paragraph.text, "First paragraph. Still first paragraph.");
        assert_eq!(consumed, 2);
    }

    #[test]
    fn merges_a_paragraph_split_by_a_page_break() {
        let merged = merge_across_page_break(
            "The result carries over to the next page and",
            "continues here without interruption.",
        );
        assert_eq!(
            merged,
            Some(
                "The result carries over to the next page and continues here without interruption."
                    .to_string()
            )
        );
    }

    #[test]
    fn repairs_hyphenation_split_by_a_page_break() {
        // Same phenomenon as `repairs_end_of_line_hyphenation`, just at a
        // page-turn instead of a line-wrap.
        let merged = merge_across_page_break("word wrapping at the page bound-", "ary");
        assert_eq!(merged, Some("word wrapping at the page boundary".to_string()));
    }

    #[test]
    fn does_not_merge_when_the_trailing_paragraph_ends_a_sentence() {
        // Ending in terminal punctuation is strong evidence the paragraph
        // genuinely finished, and just happened to land at a page boundary.
        let merged = merge_across_page_break(
            "This sentence is already complete.",
            "the next paragraph, lowercase or not, is unrelated",
        );
        assert_eq!(merged, None);
    }

    #[test]
    fn does_not_merge_when_the_leading_paragraph_starts_a_new_sentence() {
        // An uppercase start is strong evidence of a fresh paragraph, even
        // if the previous one has no terminal punctuation (e.g. it ends in
        // a list-style fragment, a colon, or an em dash).
        let merged = merge_across_page_break(
            "As shown in the results above",
            "New Findings About The Topic",
        );
        assert_eq!(merged, None);
    }

    #[test]
    fn does_not_merge_an_empty_trailing_or_leading_paragraph() {
        assert_eq!(merge_across_page_break("", "continues here"), None);
        assert_eq!(merge_across_page_break("Ends here.", ""), None);
    }
}
