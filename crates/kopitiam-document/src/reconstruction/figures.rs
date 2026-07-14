use std::sync::LazyLock;

use regex::Regex;

use super::Line;
use crate::Figure;

static FIGURE_CAPTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(figure|fig\.)\s*\d+").unwrap());

/// Images themselves are not extracted (the extraction layer only recovers
/// text spans) -- only the caption is preserved, matching the spec's
/// "[Figure omitted from Markdown output]" placeholder convention.
pub(super) fn try_figure(line: &Line) -> Option<Figure> {
    let text = line.text.trim();
    if FIGURE_CAPTION.is_match(text) {
        Some(Figure {
            caption: Some(text.to_string()),
            image_path: None,
        })
    } else {
        None
    }
}
