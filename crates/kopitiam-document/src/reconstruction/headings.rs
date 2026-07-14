use std::sync::LazyLock;

use regex::Regex;

use super::Line;

static NUMBERED_SECTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+(\.\d+)*\.?\s+\S").unwrap());

const MAX_HEADING_CHARS: usize = 160;

/// Heading detection heuristics: font size significantly larger than body
/// text, or a numbered-section prefix (e.g. "3.4 Hyperfocus") at body size or
/// larger. Bold-text and centered-text cues are not available -- the PDF
/// extraction backend does not expose font style, and centering would
/// require reliable per-page text-column width, which multi-column layouts
/// don't have yet (see kopitiam-q4f).
pub(super) fn heading_level(line: &Line, body_font_size: f32) -> Option<usize> {
    let text = line.text.trim();
    if text.is_empty() || text.chars().count() > MAX_HEADING_CHARS {
        return None;
    }

    let ratio = line.font_size / body_font_size;
    if ratio >= 1.6 {
        return Some(1);
    }
    if ratio >= 1.3 {
        return Some(2);
    }
    if ratio >= 1.12 {
        return Some(3);
    }
    if ratio >= 0.98 && NUMBERED_SECTION.is_match(text) {
        return Some(section_depth(text));
    }

    None
}

fn section_depth(text: &str) -> usize {
    text.split_whitespace()
        .next()
        .map(|number| number.trim_end_matches('.').matches('.').count() + 1)
        .unwrap_or(1)
        .min(6)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, font_size: f32) -> Line {
        Line {
            text: text.to_string(),
            y: 0.0,
            font_size,
            cells: Vec::new(),
        }
    }

    #[test]
    fn large_font_is_a_top_level_heading() {
        assert_eq!(heading_level(&line("Chapter One", 20.0), 10.0), Some(1));
    }

    #[test]
    fn body_sized_text_is_not_a_heading() {
        assert_eq!(
            heading_level(&line("Just a normal sentence.", 10.0), 10.0),
            None
        );
    }

    #[test]
    fn numbered_section_at_body_size_is_a_heading() {
        assert_eq!(heading_level(&line("3.4 Hyperfocus", 10.0), 10.0), Some(2));
    }

    #[test]
    fn overly_long_line_is_never_a_heading_even_if_large() {
        let text = "word ".repeat(60);
        assert_eq!(heading_level(&line(&text, 20.0), 10.0), None);
    }
}
