//! Snippet **placement**: turning a `kopitiam-snippet` [`Expansion`] (literal
//! text + tabstops expressed in *char* offsets) into concrete buffer positions
//! the editor can drive `<Tab>`/`<S-Tab>` navigation against.
//!
//! # The seam this module owns
//!
//! `kopitiam-snippet` is deliberately UI-free (see `docs/ai-decisions/
//! AID-0024`): it hands back `Expansion { text, tabstops }` where every tabstop
//! range is a **char** offset into `text`, and says nothing about where that
//! text lands in a buffer. kvim's cursor is a **grapheme**-indexed
//! [`Position`], and the snippet is inserted at some arbitrary `(line, col)`
//! that may already be indented. Bridging those two — char offsets in a
//! standalone string ⇒ grapheme `(line, col)` in the buffer after insertion —
//! is this module's whole job, kept pure and unit-tested so the arithmetic is
//! verified without a running editor *or* a running snippet engine (its fields
//! are public, so a test builds an [`Expansion`] by hand).
//!
//! # Char offsets vs. grapheme columns
//!
//! The LSP snippet grammar counts in Unicode scalar values (`char`s), matching
//! the rest of KOPITIAM's LSP boundary. kvim counts columns in grapheme
//! clusters. For the overwhelmingly common ASCII snippet the two coincide; they
//! diverge only when a *placeholder* contains a combining sequence (`é` as
//! `e` + U+0301) or an astral character. [`offset_to_position`] converts by
//! slicing `text` at the char offset and counting graphemes in the result, so a
//! placeholder with such content still lands the cursor on the right cell — the
//! case AID-0024 flagged to test.
//!
//! # What this module does *not* do
//!
//! It does not mutate a buffer and it does not propagate mirrored edits as the
//! user types. Inserting `Expansion::text` and re-syncing mirrors are buffer
//! operations owned by [`crate::ui::app::App`] (which holds the editor handle);
//! this module only computes *where* things are. See `kopitiam-cj0.17`'s report
//! and the follow-up bead for the current depth of live mirror support.

use unicode_segmentation::UnicodeSegmentation;

use kopitiam_snippet::{CharRange, Expansion};

use crate::core::{Position, Range};

/// Converts a **char** offset into `text` to the buffer [`Position`] it occupies
/// once `text` is inserted at `at`.
///
/// A newline in `text` before the offset moves the result down a line and
/// resets the column to the snippet line's own start (column 0), because text
/// after a `\n` begins at the left margin, not indented under `at.col`. On the
/// *first* line the column is `at.col + <graphemes before the offset>`, since
/// that line continues from wherever the insertion began.
pub fn offset_to_position(text: &str, at: Position, char_offset: usize) -> Position {
    let chars: Vec<char> = text.chars().collect();
    let end = char_offset.min(chars.len());
    let prefix: String = chars[..end].iter().collect();
    let newlines = prefix.matches('\n').count();
    let last_segment = match prefix.rfind('\n') {
        Some(i) => &prefix[i + 1..],
        None => prefix.as_str(),
    };
    let col = last_segment.graphemes(true).count();
    if newlines == 0 {
        Position::new(at.line, at.col + col)
    } else {
        Position::new(at.line + newlines, col)
    }
}

/// Maps one snippet [`CharRange`] to a buffer [`Range`] anchored at `at`.
fn char_range_to_range(text: &str, at: Position, cr: CharRange) -> Range {
    Range::new(offset_to_position(text, at, cr.start), offset_to_position(text, at, cr.end))
}

/// A single tabstop, placed in the buffer: its number, every range it occupies
/// (the first is the primary edit site; any others are mirrors), and its
/// placeholder text if it had one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetStop {
    pub index: u32,
    pub ranges: Vec<Range>,
    pub placeholder: Option<String>,
}

impl SnippetStop {
    /// The primary (first) range — the one the cursor edits; the rest are
    /// mirrors that should track it.
    pub fn primary(&self) -> Range {
        self.ranges[0]
    }

    /// Where the cursor lands when this stop becomes active: the start of its
    /// primary range.
    pub fn target(&self) -> Position {
        self.primary().normalized().0
    }
}

/// An in-progress snippet expansion the editor is navigating: the placed
/// tabstops in visit order (ascending index, with the final `$0` stop last —
/// the order `kopitiam-snippet` already guarantees) and which one is current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetSession {
    pub stops: Vec<SnippetStop>,
    pub current: usize,
}

impl SnippetSession {
    /// Builds a session from a freshly-computed [`Expansion`] inserted at `at`,
    /// or `None` when the expansion has no tabstops to navigate (a plain string
    /// with no `$1`/`$0`, or — until the real engine lands — the scaffold
    /// stub, whose `expand` returns an empty tabstop list). `None` means "the
    /// text was inserted; there is nothing to drive with `<Tab>`", which the
    /// caller treats as an ordinary completion.
    pub fn from_expansion(exp: &Expansion, at: Position) -> Option<Self> {
        if exp.tabstops.is_empty() {
            return None;
        }
        let stops: Vec<SnippetStop> = exp
            .tabstops
            .iter()
            .filter(|t| !t.ranges.is_empty())
            .map(|t| SnippetStop {
                index: t.index,
                ranges: t.ranges.iter().map(|r| char_range_to_range(&exp.text, at, *r)).collect(),
                placeholder: t.placeholder.clone(),
            })
            .collect();
        if stops.is_empty() {
            return None;
        }
        Some(Self { stops, current: 0 })
    }

    /// The active stop.
    pub fn current_stop(&self) -> &SnippetStop {
        &self.stops[self.current]
    }

    /// Where the cursor should sit for the active stop.
    pub fn target(&self) -> Position {
        self.current_stop().target()
    }

    /// The primary range to *select* for the active stop when it carries a
    /// placeholder (so typing replaces it), or `None` for a bare stop.
    pub fn placeholder_range(&self) -> Option<Range> {
        let stop = self.current_stop();
        stop.placeholder.as_ref().map(|_| stop.primary())
    }

    /// Whether the active stop is the last one (the final `$0` / implicit end
    /// stop): a `<Tab>` from here ends the session.
    pub fn on_final(&self) -> bool {
        self.current + 1 >= self.stops.len()
    }

    /// Advances to the next stop, returning whether it moved (`false` means it
    /// was already on the final stop — the caller ends the session).
    pub fn advance(&mut self) -> bool {
        if self.current + 1 < self.stops.len() {
            self.current += 1;
            true
        } else {
            false
        }
    }

    /// Steps back to the previous stop, returning whether it moved (`false`
    /// means it was already on the first stop).
    pub fn retreat(&mut self) -> bool {
        if self.current > 0 {
            self.current -= 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_snippet::Tabstop;

    #[test]
    fn offset_on_the_first_line_is_relative_to_the_insertion_column() {
        // Insert "greet()" at column 4 of line 2; the offset of `)` (char 6)
        // must land at column 4 + 6 = 10 on the same line.
        let pos = offset_to_position("greet()", Position::new(2, 4), 6);
        assert_eq!(pos, Position::new(2, 10));
    }

    #[test]
    fn offset_after_a_newline_drops_a_line_and_resets_the_column() {
        // "fn f() {\n\tbody" — the 'b' of "body" is char offset 10, on the
        // second snippet line, column 1 (after the tab). The insertion column
        // must NOT bleed onto the wrapped line.
        let text = "fn f() {\n\tbody";
        let at = Position::new(0, 4);
        let pos = offset_to_position(text, at, 10);
        assert_eq!(pos, Position::new(1, 1), "line +1, column reset to the snippet line's own start");
    }

    #[test]
    fn offset_counts_graphemes_not_chars_after_an_astral_placeholder() {
        // "🚀x": the 'x' is char offset 1 but the rocket is a single grapheme,
        // so 'x' sits at grapheme column at.col + 1.
        let pos = offset_to_position("🚀x", Position::new(0, 3), 1);
        assert_eq!(pos, Position::new(0, 4), "🚀 is one grapheme column, so x is at 3 + 1");
    }

    fn tabstop(index: u32, ranges: &[(usize, usize)], placeholder: Option<&str>) -> Tabstop {
        Tabstop {
            index,
            ranges: ranges.iter().map(|&(start, end)| CharRange { start, end }).collect(),
            placeholder: placeholder.map(str::to_string),
            choices: Vec::new(),
        }
    }

    #[test]
    fn session_places_tabstops_and_navigates_forward_and_back() {
        // "fn ${1:name}() {\n\t$0\n}" expands to "fn name() {\n\t\n}" with a
        // tabstop 1 over "name" (chars 3..7) and a final $0 (char 12) — the
        // shape the real engine will produce.
        let exp = Expansion {
            text: "fn name() {\n\t\n}".to_string(),
            tabstops: vec![
                tabstop(1, &[(3, 7)], Some("name")),
                // $0 sits between the `\t` (char 12) and the trailing `\n`
                // (char 13), so its char offset is 13.
                tabstop(0, &[(13, 13)], None),
            ],
        };
        let at = Position::new(5, 0);
        let mut session = SnippetSession::from_expansion(&exp, at).expect("two tabstops -> a session");

        // Stop 1: cursor on "name", which is selectable (has a placeholder).
        assert_eq!(session.target(), Position::new(5, 3));
        assert_eq!(session.placeholder_range(), Some(Range::new(Position::new(5, 3), Position::new(5, 7))));
        assert!(!session.on_final());

        // Tab -> the final $0 on the (inserted) second line.
        assert!(session.advance(), "advancing off stop 1 moves to $0");
        assert_eq!(session.target(), Position::new(6, 1), "$0 sits after the tab on the wrapped line");
        assert_eq!(session.placeholder_range(), None, "$0 has no placeholder to select");
        assert!(session.on_final(), "and it is the last stop");
        assert!(!session.advance(), "a Tab from the final stop does not move; the caller ends the session");

        // S-Tab -> back to stop 1.
        assert!(session.retreat());
        assert_eq!(session.target(), Position::new(5, 3));
        assert!(!session.retreat(), "already on the first stop");
    }

    #[test]
    fn a_mirrored_tabstop_records_every_range_primary_first() {
        // "${1:x} = $1" -> "x = x": stop 1 mirrors, ranges (0..1) and (4..5).
        let exp = Expansion {
            text: "x = x".to_string(),
            tabstops: vec![tabstop(1, &[(0, 1), (4, 5)], Some("x")), tabstop(0, &[(5, 5)], None)],
        };
        let session = SnippetSession::from_expansion(&exp, Position::ORIGIN).unwrap();
        let stop = &session.stops[0];
        assert_eq!(stop.ranges.len(), 2, "a mirrored stop keeps every occurrence");
        assert_eq!(stop.primary(), Range::new(Position::new(0, 0), Position::new(0, 1)), "the first range is the primary");
        assert_eq!(stop.ranges[1], Range::new(Position::new(0, 4), Position::new(0, 5)), "the second is the mirror");
    }

    #[test]
    fn no_tabstops_yields_no_session() {
        // The scaffold stub (and any plain string) returns an empty tabstop
        // list: there is nothing to navigate, so no session starts.
        let exp = Expansion { text: "just text".to_string(), tabstops: Vec::new() };
        assert!(SnippetSession::from_expansion(&exp, Position::ORIGIN).is_none());
    }
}
