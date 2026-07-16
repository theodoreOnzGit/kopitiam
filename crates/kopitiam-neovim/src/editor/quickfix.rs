//! The quickfix and location lists: vim's model for "a list of file positions
//! you step through and jump to".
//!
//! # What lives here, and what deliberately does not
//!
//! This module owns the *model* — a list of [`QuickfixEntry`] positions plus a
//! cursor into it ([`QuickfixList`]), and the navigation grammar (`:cnext`,
//! `:cprev`, `:cfirst`, `:clast`, `:cc {nr}`) as pure index arithmetic. It is
//! headless and has no idea where entries come from or how a jump is performed:
//! populating the list is [`crate::plugins::grep`]'s job (the `:grep` walk), and
//! performing a jump is the UI's job (open the file, move the cursor — see
//! `crate::ui::app`). Keeping the model this thin is what lets it be unit-tested
//! with a handful of fake entries and no editor, no filesystem, and no terminal.
//!
//! # Quickfix vs. location list
//!
//! Vim carries two nearly-identical lists: one global *quickfix* list
//! (`:copen`, `:cnext`, …) and a *location* list that is local to a window
//! (`:lopen`, `:lnext`, …). They use the same machinery with different command
//! prefixes, so [`QuickfixList`] serves both and [`ListKind`] only names which
//! one a parsed command was aimed at — the executor picks the matching list.
//!
//! # Navigation semantics follow vim exactly
//!
//! `:cnext` at the last entry is an *error* (vim's `E553: No more items`), not a
//! wrap-around to the top; likewise `:cprev` at the first. `:cfirst`/`:clast`
//! jump to the ends unconditionally. `:cc {nr}` goes to an explicit 1-based
//! entry, erroring if it is out of range. Getting these edges wrong is the
//! difference between a quickfix list that feels like vim and one that merely
//! resembles it, so each edge is a test below.

use std::path::PathBuf;

/// Which of vim's two lists a command targets. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListKind {
    /// The global quickfix list: `:grep`, `:copen`, `:cnext`, `:cc`, …
    Quickfix,
    /// The window-local location list: `:lgrep`, `:lopen`, `:lnext`, `:ll`, …
    Location,
}

impl ListKind {
    /// The human word for this list, for messages (`"quickfix"` / `"location"`).
    pub fn label(self) -> &'static str {
        match self {
            ListKind::Quickfix => "quickfix",
            ListKind::Location => "location",
        }
    }
}

/// One position in a [`QuickfixList`]: a file, a 1-based line and column, and the
/// text of that line for display. Built from a [`crate::plugins::grep::GrepMatch`]
/// (and, in future, from LSP diagnostics or compiler errors — anything that is a
/// "here is a place worth jumping to").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuickfixEntry {
    pub path: PathBuf,
    /// 1-based line, as vim's quickfix list counts.
    pub line: usize,
    /// 1-based column.
    pub col: usize,
    /// The matching line's text, shown in the quickfix window after `file|line|`.
    pub text: String,
}

/// A quickfix or location list: the entries plus a cursor (`current`) into them.
///
/// `current` is only meaningful when `entries` is non-empty; on an empty list it
/// is `0` and every navigation is a no-op-returning-error. The list is populated
/// wholesale by [`Self::set`] (a fresh `:grep` replaces the old results, matching
/// vim, which pushes a new list rather than appending) and then walked by the
/// navigation methods.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuickfixList {
    entries: Vec<QuickfixEntry>,
    current: usize,
}

/// Why a navigation could not move — mapped by the caller to a vim-style message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavError {
    /// The list has no entries (`E42: No Errors`).
    Empty,
    /// Already at the last entry and asked to go further (`E553: No more items`).
    AtEnd,
    /// Already at the first entry and asked to go back (`E553`).
    AtStart,
    /// `:cc {nr}` named an entry outside `1..=len` (`E541`-ish).
    OutOfRange,
}

impl QuickfixList {
    /// Replaces the list's contents and resets the cursor to the first entry —
    /// the effect of a fresh `:grep`. Vim starts a new quickfix list at entry 1,
    /// so a following bare `:cc` or `:copen` lands on the first result.
    pub fn set(&mut self, entries: Vec<QuickfixEntry>) {
        self.entries = entries;
        self.current = 0;
    }

    /// Every entry, for the quickfix window to render.
    pub fn entries(&self) -> &[QuickfixEntry] {
        &self.entries
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty (no `:grep` has populated it, or it found
    /// nothing).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The 0-based index of the current entry, meaningful only when non-empty.
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// The current entry, or `None` on an empty list.
    pub fn current(&self) -> Option<&QuickfixEntry> {
        self.entries.get(self.current)
    }

    /// Sets the current entry to a 0-based index, clamped into range. Used by the
    /// quickfix window when the user moves the selection with `j`/`k` — the
    /// window and `:cnext` share one cursor, so a window move *is* a `current`
    /// move. A clamp (not an error) is right here: the window can never point
    /// past its own rows.
    pub fn select(&mut self, index: usize) -> Option<&QuickfixEntry> {
        if self.entries.is_empty() {
            return None;
        }
        self.current = index.min(self.entries.len() - 1);
        self.current()
    }

    /// `:cnext`/`:lnext` — advance to the next entry, erroring at the end
    /// (vim does not wrap).
    ///
    /// Named `advance`, not `next`, so it is never confused with
    /// [`Iterator::next`] — this is a cursor move on a fixed list, not iteration.
    pub fn advance(&mut self) -> Result<&QuickfixEntry, NavError> {
        if self.entries.is_empty() {
            return Err(NavError::Empty);
        }
        if self.current + 1 >= self.entries.len() {
            return Err(NavError::AtEnd);
        }
        self.current += 1;
        Ok(&self.entries[self.current])
    }

    /// `:cprev`/`:lprev` — step back to the previous entry, erroring at the top.
    /// Named `retreat` for symmetry with [`Self::advance`].
    pub fn retreat(&mut self) -> Result<&QuickfixEntry, NavError> {
        if self.entries.is_empty() {
            return Err(NavError::Empty);
        }
        if self.current == 0 {
            return Err(NavError::AtStart);
        }
        self.current -= 1;
        Ok(&self.entries[self.current])
    }

    /// `:cfirst`/`:lfirst` — jump to the first entry.
    pub fn first(&mut self) -> Result<&QuickfixEntry, NavError> {
        if self.entries.is_empty() {
            return Err(NavError::Empty);
        }
        self.current = 0;
        Ok(&self.entries[0])
    }

    /// `:clast`/`:llast` — jump to the last entry.
    pub fn last(&mut self) -> Result<&QuickfixEntry, NavError> {
        if self.entries.is_empty() {
            return Err(NavError::Empty);
        }
        self.current = self.entries.len() - 1;
        Ok(&self.entries[self.current])
    }

    /// `:cc [nr]`/`:ll [nr]` — go to the entry `nr` (1-based), or, when `nr` is
    /// `None`, re-select the *current* entry (vim's bare `:cc` re-displays the
    /// current error). Errors if `nr` is out of range.
    pub fn goto(&mut self, nr: Option<usize>) -> Result<&QuickfixEntry, NavError> {
        if self.entries.is_empty() {
            return Err(NavError::Empty);
        }
        match nr {
            None => Ok(&self.entries[self.current]),
            Some(n) => {
                if n == 0 || n > self.entries.len() {
                    return Err(NavError::OutOfRange);
                }
                self.current = n - 1;
                Ok(&self.entries[self.current])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, line: usize) -> QuickfixEntry {
        QuickfixEntry { path: PathBuf::from(name), line, col: 1, text: format!("{name}:{line}") }
    }

    fn three() -> QuickfixList {
        let mut l = QuickfixList::default();
        l.set(vec![entry("a.rs", 1), entry("b.rs", 2), entry("c.rs", 3)]);
        l
    }

    #[test]
    fn a_fresh_list_starts_on_the_first_entry() {
        let l = three();
        assert_eq!(l.len(), 3);
        assert_eq!(l.current_index(), 0);
        assert_eq!(l.current().unwrap().line, 1);
    }

    #[test]
    fn next_and_prev_walk_and_error_at_the_ends() {
        let mut l = three();
        assert_eq!(l.advance().unwrap().line, 2);
        assert_eq!(l.advance().unwrap().line, 3);
        // At the last entry: :cnext errors rather than wrapping (vim E553).
        assert_eq!(l.advance(), Err(NavError::AtEnd));
        assert_eq!(l.current_index(), 2, "a failed :cnext must not move");
        assert_eq!(l.retreat().unwrap().line, 2);
        assert_eq!(l.retreat().unwrap().line, 1);
        assert_eq!(l.retreat(), Err(NavError::AtStart));
        assert_eq!(l.current_index(), 0);
    }

    #[test]
    fn first_and_last_jump_to_the_ends() {
        let mut l = three();
        l.advance().unwrap();
        assert_eq!(l.last().unwrap().line, 3);
        assert_eq!(l.current_index(), 2);
        assert_eq!(l.first().unwrap().line, 1);
        assert_eq!(l.current_index(), 0);
    }

    #[test]
    fn cc_goes_to_an_explicit_one_based_entry() {
        let mut l = three();
        assert_eq!(l.goto(Some(3)).unwrap().line, 3);
        assert_eq!(l.current_index(), 2);
        // Bare :cc re-selects the current one.
        assert_eq!(l.goto(None).unwrap().line, 3);
        // Out of range errors and does not move.
        assert_eq!(l.goto(Some(9)), Err(NavError::OutOfRange));
        assert_eq!(l.goto(Some(0)), Err(NavError::OutOfRange));
        assert_eq!(l.current_index(), 2);
    }

    #[test]
    fn select_clamps_and_shares_the_cursor_with_navigation() {
        let mut l = three();
        // A window move to row 1 sets current to 1; a following :cnext goes to 2.
        l.select(1);
        assert_eq!(l.current_index(), 1);
        assert_eq!(l.advance().unwrap().line, 3);
        // Selecting past the end clamps to the last row.
        l.select(99);
        assert_eq!(l.current_index(), 2);
    }

    #[test]
    fn every_navigation_on_an_empty_list_errors() {
        let mut l = QuickfixList::default();
        assert!(l.is_empty());
        assert_eq!(l.advance(), Err(NavError::Empty));
        assert_eq!(l.retreat(), Err(NavError::Empty));
        assert_eq!(l.first(), Err(NavError::Empty));
        assert_eq!(l.last(), Err(NavError::Empty));
        assert_eq!(l.goto(Some(1)), Err(NavError::Empty));
        assert!(l.current().is_none());
    }
}
