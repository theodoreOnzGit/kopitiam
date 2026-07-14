use std::sync::LazyLock;

use regex::Regex;

use super::Line;
use crate::List;

static UNORDERED_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[•\-\*\x{2022}\x{2023}\x{25E6}\x{2043}]\s+(.*)$").unwrap());
static ORDERED_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:\d+|[a-zA-Z])[.)]\s+(.*)$").unwrap());

enum Marker {
    Ordered(String),
    Unordered(String),
}

fn marker(text: &str) -> Option<Marker> {
    let text = text.trim();
    if let Some(caps) = UNORDERED_MARKER.captures(text) {
        return Some(Marker::Unordered(caps[1].to_string()));
    }
    if let Some(caps) = ORDERED_MARKER.captures(text) {
        return Some(Marker::Ordered(caps[1].to_string()));
    }
    None
}

pub(super) fn try_list(lines: &[Line]) -> Option<(List, usize)> {
    let ordered = matches!(marker(&lines[0].text)?, Marker::Ordered(_));
    let mut items = Vec::new();
    let mut consumed = 0;

    for line in lines {
        match marker(&line.text) {
            Some(Marker::Ordered(text)) if ordered => items.push(text),
            Some(Marker::Unordered(text)) if !ordered => items.push(text),
            _ => break,
        }
        consumed += 1;
    }

    if items.is_empty() {
        None
    } else {
        Some((List { ordered, items }, consumed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> Line {
        Line {
            text: text.to_string(),
            y: 0.0,
            font_size: 10.0,
            cells: Vec::new(),
        }
    }

    #[test]
    fn unordered_bullets_become_an_unordered_list() {
        let lines = vec![line("- Item A"), line("- Item B"), line("Not a bullet")];
        let (list, consumed) = try_list(&lines).unwrap();
        assert!(!list.ordered);
        assert_eq!(list.items, vec!["Item A", "Item B"]);
        assert_eq!(consumed, 2);
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
}
