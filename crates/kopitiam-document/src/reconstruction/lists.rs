use std::sync::LazyLock;

use regex::Regex;

use super::Line;
use crate::List;

static UNORDERED_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[•\-\*\x{2022}\x{2023}\x{25E6}\x{2043}]\s+(.*)$").unwrap());
static ORDERED_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:\d+|[a-zA-Z])[.)]\s+(.*)$").unwrap());

/// One level of indent counts as nesting when the left edge moves right by at
/// least this fraction of the line's font size. Marker uses ~1% of the page
/// width (`ListProcessor.min_x_indent`); we do not have the page width inside
/// list detection, so we scale by the font size instead -- an indent step is
/// almost always on the order of one em or more, and jitter within a level is
/// well under it. Kept conservative so ordinary same-level items are not split
/// into spurious sub-lists.
const NEST_INDENT_TOLERANCE_RATIO: f32 = 0.75;

enum Marker {
    Ordered(String),
    /// `ascii` is true for the ambiguous ASCII bullets `-`/`*` (which prose
    /// also opens with) and false for genuine bullet glyphs (•, ‣, ◦, ⁃).
    Unordered { text: String, ascii: bool },
}

fn marker(text: &str) -> Option<Marker> {
    let text = text.trim();
    if let Some(caps) = UNORDERED_MARKER.captures(text) {
        let ascii = matches!(text.chars().next(), Some('-') | Some('*'));
        return Some(Marker::Unordered {
            text: caps[1].to_string(),
            ascii,
        });
    }
    if let Some(caps) = ORDERED_MARKER.captures(text) {
        return Some(Marker::Ordered(caps[1].to_string()));
    }
    None
}

/// Left edge (x of the first glyph run) of a line, used to infer list nesting.
/// A line with no cells (e.g. a hand-built test line) reports 0.0, which yields
/// a flat list -- the safe default.
fn left_edge(line: &Line) -> f32 {
    line.cells.first().map(|cell| cell.x).unwrap_or(0.0)
}

pub(super) fn try_list(lines: &[Line]) -> Option<(List, usize)> {
    let ordered = matches!(marker(&lines[0].text)?, Marker::Ordered(_));
    let mut items = Vec::new();
    let mut edges = Vec::new();
    let mut all_ascii = true;
    let mut consumed = 0;

    for line in lines {
        match marker(&line.text) {
            Some(Marker::Ordered(text)) if ordered => items.push(text),
            Some(Marker::Unordered { text, ascii }) if !ordered => {
                all_ascii &= ascii;
                items.push(text);
            }
            _ => break,
        }
        edges.push(left_edge(line));
        consumed += 1;
    }

    if items.is_empty() {
        return None;
    }

    // Guard (task #16): a *single* item led by an ASCII `-`/`*` is ambiguous --
    // ordinary prose routinely opens with a dash ("- but see the appendix ..."),
    // so a lone ascii-dash line is far more likely a sentence than a one-item
    // list. Require corroboration (>= 2 items) before treating it as a list; a
    // genuine bullet glyph (•, ‣, ...) or an ordered enumerator is unambiguous
    // and may stand alone. Precision over recall: never swallow a real paragraph
    // that merely starts with a dash.
    if !ordered && items.len() == 1 && all_ascii {
        return None;
    }

    let depths = infer_depths(&edges, lines[0].font_size);
    Some((
        List {
            ordered,
            items,
            depths,
        },
        consumed,
    ))
}

/// Infers each item's nesting depth from its left edge, adapting marker's
/// x-indentation stack (`ListProcessor.list_group_indentation`) to work without
/// a page-width reference.
///
/// A stack holds one representative left edge per currently-open depth level.
/// An item clearly further right than the current level (`> top + tol`) opens a
/// child level; an item clearly further left (`< top - tol`) closes levels back
/// out until it lines up with an ancestor; otherwise it is a sibling at the
/// current level. `tol` is a fraction of the font size (see
/// [`NEST_INDENT_TOLERANCE_RATIO`]).
///
/// Returns an **empty** vec when every item is top level, so a flat list stays
/// the flat shape (`depths` empty) and renders byte-identically to before.
fn infer_depths(edges: &[f32], font_size: f32) -> Vec<usize> {
    let tol = (font_size * NEST_INDENT_TOLERANCE_RATIO).max(1.0);
    let mut depths = Vec::with_capacity(edges.len());
    let mut stack: Vec<f32> = Vec::new();

    for &x in edges {
        // Close any deeper levels sitting to the right of x: the indent moved
        // back out toward an ancestor.
        while stack.last().is_some_and(|&top| x < top - tol) {
            stack.pop();
        }
        let depth = match stack.last() {
            None => {
                stack.push(x);
                0
            }
            Some(&top) if x > top + tol => {
                stack.push(x);
                stack.len() - 1
            }
            Some(_) => stack.len() - 1,
        };
        depths.push(depth);
    }

    if depths.iter().all(|&d| d == 0) {
        Vec::new()
    } else {
        depths
    }
}

#[cfg(test)]
mod tests {
    use super::super::Cell;
    use super::*;

    fn line(text: &str) -> Line {
        line_at(text, 0.0)
    }

    /// A list line whose single cell begins at `x` -- the left edge list
    /// nesting keys off. The whole line text (marker included) is the cell
    /// text, matching what `build_line` produces for a real bullet line.
    fn line_at(text: &str, x: f32) -> Line {
        Line {
            text: text.to_string(),
            y: 0.0,
            font_size: 10.0,
            cells: vec![Cell {
                text: text.to_string(),
                x,
                x_end: x + 100.0,
            }],
        }
    }

    #[test]
    fn unordered_bullets_become_an_unordered_list() {
        let lines = vec![line("- Item A"), line("- Item B"), line("Not a bullet")];
        let (list, consumed) = try_list(&lines).unwrap();
        assert!(!list.ordered);
        assert_eq!(list.items, vec!["Item A", "Item B"]);
        assert_eq!(consumed, 2);
        // Uniform indent -> flat list, no depth bookkeeping.
        assert!(list.depths.is_empty());
    }

    #[test]
    fn numbered_items_become_an_ordered_list() {
        let lines = vec![line("1. First"), line("2. Second")];
        let (list, consumed) = try_list(&lines).unwrap();
        assert!(list.ordered);
        assert_eq!(list.items, vec!["First", "Second"]);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn plain_prose_is_not_a_list() {
        let lines = vec![line("This is a normal paragraph.")];
        assert!(try_list(&lines).is_none());
    }

    #[test]
    fn a_lone_paragraph_starting_with_a_dash_is_not_a_list() {
        // Task #16 acceptance: a single ascii-dash line is ambiguous with prose
        // ("- The quarterly figures were revised upward ...") and must NOT be
        // swallowed into a one-item list.
        let lines = vec![
            line("- The quarterly figures were revised upward after the audit."),
            line("The next paragraph continues the discussion in full sentences."),
        ];
        assert!(
            try_list(&lines).is_none(),
            "a lone dash-led prose line must stay a paragraph"
        );
    }

    #[test]
    fn a_lone_genuine_bullet_glyph_item_is_still_a_list() {
        // A real bullet glyph is unambiguous, so a single-item list of one is
        // allowed -- only the ASCII dash/asterisk needs corroboration.
        let lines = vec![line("• Sole bullet item")];
        let (list, _) = try_list(&lines).unwrap();
        assert!(!list.ordered);
        assert_eq!(list.items, vec!["Sole bullet item"]);
    }

    #[test]
    fn two_level_nested_bullet_list_infers_depth_from_indentation() {
        // Task #16 acceptance: a 2-level nested bullet list. Top items at x=50,
        // nested items indented to x=80.
        let lines = vec![
            line_at("- Top one", 50.0),
            line_at("- Nested a", 80.0),
            line_at("- Nested b", 80.0),
            line_at("- Top two", 50.0),
        ];
        let (list, consumed) = try_list(&lines).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(list.items, vec!["Top one", "Nested a", "Nested b", "Top two"]);
        assert_eq!(list.depths, vec![0, 1, 1, 0]);
    }

    #[test]
    fn nested_ordered_list_infers_depth_from_indentation() {
        // Task #16 acceptance: a nested ordered list.
        let lines = vec![
            line_at("1. First", 50.0),
            line_at("2. Second", 50.0),
            line_at("1. Sub of second", 82.0),
            line_at("3. Third", 50.0),
        ];
        let (list, _) = try_list(&lines).unwrap();
        assert!(list.ordered);
        assert_eq!(list.depths, vec![0, 0, 1, 0]);
    }

    #[test]
    fn a_flat_ascii_list_keeps_empty_depths() {
        let lines = vec![line_at("- a", 50.0), line_at("- b", 50.0)];
        let (list, _) = try_list(&lines).unwrap();
        assert!(
            list.depths.is_empty(),
            "a flat list must not carry per-item depths"
        );
    }
}
