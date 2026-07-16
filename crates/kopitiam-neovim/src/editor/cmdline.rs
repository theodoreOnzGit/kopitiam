//! The command-line line editor: the small editable buffer behind the `:`,
//! `/` and `?` prompts, plus the per-prompt history rings.
//!
//! # Why this is its own thing, not just a `String`
//!
//! The command line used to be a bare `String` that keys only ever *appended*
//! to (with a lone `Backspace`). That is enough to type `:wq` and no more — no
//! cursor to move, no history to recall, no `<C-w>` to rub out a word. A vim
//! user reaches for those constantly, so a write-only prompt feels broken.
//!
//! [`CmdlineBuffer`] gives the prompt a real insertion point: text plus a
//! **grapheme** cursor (the same unit every other cursor in kvim uses — see
//! [`crate::core::Position`] on why graphemes, not bytes or `char`s), with the
//! editing operations vim's command line supports. [`History`] is the recalled
//! `:`/`/` list, kept as a de-duplicated most-recent-last ring.
//!
//! What this module deliberately does *not* do: interpret what was typed
//! (that is [`super::ex`] and [`super::search`]), reach the filesystem, or read
//! the buffer table. Completion *candidates* need those, so the editor computes
//! them and hands them in — see [`CmdlineBuffer::begin_completion`]. This module
//! stays a pure, terminal-free string editor, testable without an `Editor`.

use unicode_segmentation::UnicodeSegmentation;

/// A bounded, de-duplicated command-line history ring for one prompt kind.
///
/// vim keeps the `:` history and the `/`?` history apart — recalling a search
/// must never surface an ex command and vice-versa — so the editor holds one
/// `History` per [`super::CommandKind`] family (see [`super::Editor`]).
///
/// Entries are stored oldest-first, newest-last. Pushing a line that already
/// exists moves it to the newest position rather than duplicating it, matching
/// vim: re-running `:w` twice leaves one `w` at the top of the history, not two.
#[derive(Debug, Clone)]
pub struct History {
    /// Oldest at index 0, most-recently-entered at the end.
    entries: Vec<String>,
    /// Cap on retained entries, oldest evicted first. vim's default
    /// `'history'` is 10000; 200 is plenty for a session and keeps the ring
    /// cheap to walk and filter.
    max: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    pub fn new() -> Self {
        Self { entries: Vec::new(), max: 200 }
    }

    /// Records a completed command line as the newest entry.
    ///
    /// Empty lines are never stored (pressing `:` then Enter records nothing).
    /// A line equal to an existing entry is *moved* to newest rather than
    /// duplicated — so the history reads as "distinct commands, most recent
    /// first" the way vim's does.
    pub fn push(&mut self, entry: impl Into<String>) {
        let entry = entry.into();
        if entry.is_empty() {
            return;
        }
        self.entries.retain(|e| e != &entry);
        self.entries.push(entry);
        if self.entries.len() > self.max {
            let excess = self.entries.len() - self.max;
            self.entries.drain(0..excess);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The entries starting with `prefix`, oldest-first, as `(index, text)`.
    /// An empty prefix matches every entry — that is bare `<Up>` cycling the
    /// whole history. A non-empty prefix drives vim's prefix-filtered recall
    /// (`:e<Up>` walks only past commands that began with `e`).
    fn matching(&self, prefix: &str) -> Vec<(usize, &str)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.starts_with(prefix))
            .map(|(i, e)| (i, e.as_str()))
            .collect()
    }
}

/// An in-flight walk through history, remembering the live draft so that
/// stepping back past the newest entry restores exactly what the user had
/// typed before they first pressed `<Up>`.
#[derive(Debug, Clone)]
struct HistoryWalk {
    /// The matching history entries (their text), oldest-first, captured once
    /// when the walk began — filtered by the command-line text at the moment
    /// the first `<Up>`/`<Down>` was pressed. History does not change mid-walk,
    /// so this snapshot is stable for the life of the walk.
    matches: Vec<String>,
    /// Position within `matches`; `matches.len()` is the sentinel meaning "back
    /// at the live draft" (past the newest entry).
    pos: usize,
    /// The text and cursor the user had before the walk started, restored when
    /// `pos` returns to the draft sentinel.
    draft: String,
    draft_cursor: usize,
}

/// One active completion cycle: the candidates for the token under the cursor
/// and where we are in cycling them. `<Tab>` advances, `<S-Tab>` retreats;
/// any other key ends the cycle (the `completion` field on [`CmdlineBuffer`]
/// goes back to `None`).
#[derive(Debug, Clone)]
struct CompletionCycle {
    /// Grapheme index in the text where the completed token begins. Everything
    /// from here to the cursor is the current candidate (or, before the first
    /// application, the originally-typed fragment).
    start: usize,
    /// The full replacement strings, in offer order.
    candidates: Vec<String>,
    /// Which candidate is currently shown.
    pos: usize,
}

/// The editable command-line buffer: the typed text and a grapheme cursor,
/// with the editing/history/completion state layered on top.
#[derive(Debug, Clone, Default)]
pub struct CmdlineBuffer {
    text: String,
    /// Cursor as a grapheme offset into `text` (`0..=grapheme_count`).
    cursor: usize,
    walk: Option<HistoryWalk>,
    completion: Option<CompletionCycle>,
}

impl CmdlineBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the buffer back to an empty prompt — what entering `Mode::Command`
    /// does. Drops any in-flight history walk or completion cycle too.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.walk = None;
        self.completion = None;
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The cursor as a grapheme offset — what the renderer needs to place the
    /// caret, and what the tests assert on.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Takes the current line out, leaving the buffer empty — used on Enter,
    /// where the editor needs to own the string to parse/run and record in
    /// history.
    pub fn take(&mut self) -> String {
        let line = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.walk = None;
        self.completion = None;
        line
    }

    fn graphemes(&self) -> Vec<&str> {
        self.text.graphemes(true).collect()
    }

    fn grapheme_count(&self) -> usize {
        self.text.graphemes(true).count()
    }

    /// Byte offset of grapheme index `g` (== text length when `g` is at or past
    /// the end), so edits can splice `text` without re-collecting it.
    fn byte_at(&self, g: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .nth(g)
            .map(|(b, _)| b)
            .unwrap_or(self.text.len())
    }

    // --- character input -------------------------------------------------

    /// Inserts one typed character at the cursor and steps the cursor over it.
    /// Ends any history walk or completion cycle: typing is an edit, and an
    /// edit invalidates both.
    pub fn insert_char(&mut self, c: char) {
        self.end_transients();
        let at = self.byte_at(self.cursor);
        self.text.insert(at, c);
        self.cursor += 1;
    }

    /// Inserts a whole string (a register's contents, `<C-r>{reg}`) at the
    /// cursor. A multi-line register is flattened — the command line is one
    /// line, so embedded newlines become nothing rather than breaking the
    /// prompt. Cursor lands after the inserted text.
    pub fn insert_str(&mut self, s: &str) {
        self.end_transients();
        let cleaned: String = s.chars().filter(|&c| c != '\n' && c != '\r').collect();
        let at = self.byte_at(self.cursor);
        self.text.insert_str(at, &cleaned);
        self.cursor += cleaned.graphemes(true).count();
    }

    // --- deletion --------------------------------------------------------

    /// `<BS>`/`<C-h>`: delete the grapheme before the cursor. At column 0 this
    /// is a no-op here; the caller decides whether an empty-line backspace
    /// leaves command mode (vim does).
    pub fn backspace(&mut self) {
        self.end_transients();
        if self.cursor == 0 {
            return;
        }
        let start = self.byte_at(self.cursor - 1);
        let end = self.byte_at(self.cursor);
        self.text.replace_range(start..end, "");
        self.cursor -= 1;
    }

    /// `<Del>`: delete the grapheme *at* the cursor (the one to its right).
    pub fn delete_forward(&mut self) {
        self.end_transients();
        if self.cursor >= self.grapheme_count() {
            return;
        }
        let start = self.byte_at(self.cursor);
        let end = self.byte_at(self.cursor + 1);
        self.text.replace_range(start..end, "");
    }

    /// `<C-w>`: delete the word before the cursor. "Word" here is vim's
    /// command-line `<C-w>`: skip any run of whitespace immediately left of the
    /// cursor, then delete the run of non-whitespace before that.
    pub fn delete_word_back(&mut self) {
        self.end_transients();
        let graphemes = self.graphemes();
        let mut i = self.cursor;
        while i > 0 && graphemes[i - 1].chars().all(char::is_whitespace) {
            i -= 1;
        }
        while i > 0 && !graphemes[i - 1].chars().all(char::is_whitespace) {
            i -= 1;
        }
        let start = self.byte_at(i);
        let end = self.byte_at(self.cursor);
        self.text.replace_range(start..end, "");
        self.cursor = i;
    }

    /// `<C-u>`: delete from the cursor back to the start of the line. (vim's
    /// command-line `<C-u>` clears to the start, unlike insert-mode `<C-u>`
    /// which stops at the first-inserted column; the command line has no such
    /// column, so to-start is the whole behaviour.)
    pub fn delete_to_start(&mut self) {
        self.end_transients();
        let end = self.byte_at(self.cursor);
        self.text.replace_range(0..end, "");
        self.cursor = 0;
    }

    // --- cursor movement -------------------------------------------------

    /// `<Left>`.
    pub fn move_left(&mut self) {
        self.end_transients();
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// `<Right>`.
    pub fn move_right(&mut self) {
        self.end_transients();
        if self.cursor < self.grapheme_count() {
            self.cursor += 1;
        }
    }

    /// `<Home>`/`<C-b>`.
    pub fn move_home(&mut self) {
        self.end_transients();
        self.cursor = 0;
    }

    /// `<End>`/`<C-e>`.
    pub fn move_end(&mut self) {
        self.end_transients();
        self.cursor = self.grapheme_count();
    }

    /// Overwrites the whole line and drops the cursor at `cursor` (clamped),
    /// without disturbing an in-flight history walk — the recall path uses this
    /// to show a recalled entry. Public callers that are *editing* should not
    /// use this; they want the methods above, which end the walk.
    fn set_line(&mut self, text: String, cursor: usize) {
        self.text = text;
        self.cursor = cursor.min(self.grapheme_count());
    }

    /// Ends any history walk and completion cycle. Called by every editing
    /// method: once you touch the text, the next `<Up>` should start a fresh
    /// walk from what is now on the line, and the completion cycle no longer
    /// matches what is there.
    fn end_transients(&mut self) {
        self.walk = None;
        self.completion = None;
    }

    // --- history ---------------------------------------------------------

    /// `<Up>`/`<C-p>`: step to an older history entry. On the first press it
    /// captures the current line as the walk's prefix filter and draft; further
    /// presses walk further back, stopping (a no-op) at the oldest match.
    pub fn history_prev(&mut self, history: &History) {
        self.completion = None;
        if self.walk.is_none() {
            let matches: Vec<String> = history.matching(&self.text).into_iter().map(|(_, e)| e.to_string()).collect();
            if matches.is_empty() {
                return;
            }
            self.walk = Some(HistoryWalk {
                pos: matches.len(),
                matches,
                draft: self.text.clone(),
                draft_cursor: self.cursor,
            });
        }
        let walk = self.walk.as_ref().unwrap();
        if walk.pos == 0 {
            return; // already at the oldest match
        }
        let new_pos = walk.pos - 1;
        let entry = walk.matches[new_pos].clone();
        self.walk.as_mut().unwrap().pos = new_pos;
        let end = entry.graphemes(true).count();
        self.set_line(entry, end);
    }

    /// `<Down>`/`<C-n>`: step to a newer history entry, and past the newest one
    /// back to the live draft the walk started from. A no-op when no walk is in
    /// progress (nothing newer than the draft to go to).
    pub fn history_next(&mut self, _history: &History) {
        self.completion = None;
        let Some(walk) = self.walk.as_ref() else { return };
        if walk.pos >= walk.matches.len() {
            return; // already at the draft
        }
        let new_pos = walk.pos + 1;
        if new_pos == walk.matches.len() {
            // Back to the draft.
            let (draft, cursor) = (walk.draft.clone(), walk.draft_cursor);
            self.walk.as_mut().unwrap().pos = new_pos;
            self.set_line(draft, cursor);
        } else {
            let entry = walk.matches[new_pos].clone();
            self.walk.as_mut().unwrap().pos = new_pos;
            let end = entry.graphemes(true).count();
            self.set_line(entry, end);
        }
    }

    // --- completion ------------------------------------------------------

    /// The token under the cursor that `<Tab>` completion should replace, as
    /// `(start_grapheme_index, prefix, kind)`.
    ///
    /// Splits the text-before-cursor into a command word and its argument:
    ///
    /// * no space yet → completing the **command name**; the prefix is the
    ///   whole word-before-cursor, offered against the command registry.
    /// * a space has been typed → completing the **argument**; the prefix is
    ///   the fragment after the last space, and the *kind* (file vs buffer)
    ///   depends on which command the first word names — the caller resolves
    ///   that via [`super::command::lookup`].
    ///
    /// Returns the raw split; the editor turns `kind` into concrete candidates
    /// because that needs the filesystem and the buffer table.
    pub fn completion_context(&self) -> CompletionContext {
        let graphemes = self.graphemes();
        let before: String = graphemes[..self.cursor].concat();

        match before.rfind(' ') {
            None => {
                // First word — completing the command name itself.
                CompletionContext { start: 0, prefix: before.clone(), command: None }
            }
            Some(space_byte) => {
                // Argument. The command name is the first whitespace-delimited
                // word; the prefix is the fragment after the last space.
                let command = before.split_whitespace().next().map(|s| s.to_string());
                // Grapheme index just past the last space.
                let arg_start_byte = space_byte + ' '.len_utf8();
                let arg_prefix = before[arg_start_byte..].to_string();
                let start = before[..arg_start_byte].graphemes(true).count();
                CompletionContext { start, prefix: arg_prefix, command }
            }
        }
    }

    /// Begins a completion cycle with candidates the editor computed, replacing
    /// the token at `start` with the first candidate. A single candidate is
    /// applied outright with no cycle kept (there is nothing to cycle). An
    /// empty `candidates` is a no-op (vim beeps; kvim just does nothing).
    pub fn begin_completion(&mut self, start: usize, candidates: Vec<String>) {
        if candidates.is_empty() {
            return;
        }
        self.completion = Some(CompletionCycle { start, candidates, pos: 0 });
        self.apply_current_candidate();
    }

    /// `<Tab>` again after a cycle is open: show the next candidate (wrapping).
    /// Returns `false` if no cycle is open, so the caller can start one.
    pub fn cycle_completion(&mut self, forward: bool) -> bool {
        let Some(cycle) = self.completion.as_mut() else { return false };
        let n = cycle.candidates.len();
        cycle.pos = if forward { (cycle.pos + 1) % n } else { (cycle.pos + n - 1) % n };
        self.apply_current_candidate();
        true
    }

    /// Replaces `start..cursor` with the current candidate and parks the cursor
    /// at its end. Because the previous candidate ran `start..cursor` too, this
    /// is idempotent to re-apply — which is exactly what cycling does.
    fn apply_current_candidate(&mut self) {
        let Some(cycle) = self.completion.as_ref() else { return };
        let start = cycle.start;
        let candidate = cycle.candidates[cycle.pos].clone();
        let start_byte = self.byte_at(start);
        let end_byte = self.byte_at(self.cursor);
        self.text.replace_range(start_byte..end_byte, &candidate);
        self.cursor = start + candidate.graphemes(true).count();
    }

    /// The candidate list currently being cycled and which is selected, for a
    /// wildmenu-style display. `None` when no cycle is open.
    pub fn active_completions(&self) -> Option<(&[String], usize)> {
        self.completion.as_ref().map(|c| (c.candidates.as_slice(), c.pos))
    }
}

/// The result of [`CmdlineBuffer::completion_context`] — where the token to
/// complete starts, what has been typed of it, and (for an argument) the
/// command name that governs what kind of completion applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionContext {
    /// Grapheme index where the token being completed begins.
    pub start: usize,
    /// What has been typed of the token so far.
    pub prefix: String,
    /// `None` when completing the command name itself; `Some(name)` when
    /// completing an argument of the command named `name`.
    pub command: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- History ---------------------------------------------------------

    #[test]
    fn history_dedups_moving_repeat_to_newest() {
        let mut h = History::new();
        h.push("w");
        h.push("q");
        h.push("w"); // repeat -> moves to newest, not duplicated
        let all: Vec<&str> = h.matching("").into_iter().map(|(_, e)| e).collect();
        assert_eq!(all, vec!["q", "w"]);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn history_never_stores_empty_lines() {
        let mut h = History::new();
        h.push("");
        assert!(h.is_empty());
    }

    #[test]
    fn history_prefix_filter_only_matches_that_prefix() {
        let mut h = History::new();
        h.push("edit a");
        h.push("write");
        h.push("edit b");
        let e: Vec<&str> = h.matching("e").into_iter().map(|(_, e)| e).collect();
        assert_eq!(e, vec!["edit a", "edit b"]);
    }

    // --- cursor + editing ------------------------------------------------

    fn buf(text: &str, cursor: usize) -> CmdlineBuffer {
        let mut b = CmdlineBuffer::new();
        for c in text.chars() {
            b.insert_char(c);
        }
        b.cursor = cursor;
        b
    }

    #[test]
    fn typing_appends_and_advances_the_cursor() {
        let mut b = CmdlineBuffer::new();
        b.insert_char('w');
        b.insert_char('q');
        assert_eq!(b.text(), "wq");
        assert_eq!(b.cursor(), 2);
    }

    #[test]
    fn insert_at_a_moved_cursor_lands_mid_line() {
        let mut b = buf("wq", 1);
        b.insert_char('X');
        assert_eq!(b.text(), "wXq");
        assert_eq!(b.cursor(), 2);
    }

    #[test]
    fn left_right_home_end_move_within_bounds() {
        let mut b = buf("abc", 3);
        b.move_left();
        assert_eq!(b.cursor(), 2);
        b.move_home();
        assert_eq!(b.cursor(), 0);
        b.move_left(); // clamps at 0
        assert_eq!(b.cursor(), 0);
        b.move_end();
        assert_eq!(b.cursor(), 3);
        b.move_right(); // clamps at end
        assert_eq!(b.cursor(), 3);
    }

    #[test]
    fn backspace_and_delete_forward() {
        let mut b = buf("abc", 2);
        b.backspace();
        assert_eq!((b.text(), b.cursor()), ("ac", 1));
        b.delete_forward();
        assert_eq!((b.text(), b.cursor()), ("a", 1));
    }

    #[test]
    fn ctrl_w_deletes_a_word_including_leading_space() {
        let mut b = buf("edit foo", 8);
        b.delete_word_back();
        assert_eq!((b.text(), b.cursor()), ("edit ", 5));
        // second <C-w> eats the trailing space and the word before it
        b.delete_word_back();
        assert_eq!((b.text(), b.cursor()), ("", 0));
    }

    #[test]
    fn ctrl_u_deletes_to_start_from_the_cursor() {
        let mut b = buf("wsplit", 3);
        b.delete_to_start();
        assert_eq!((b.text(), b.cursor()), ("lit", 0));
    }

    #[test]
    fn ctrl_r_inserts_a_register_flattening_newlines() {
        let mut b = buf("e ", 2);
        b.insert_str("path/to\nfile");
        assert_eq!(b.text(), "e path/tofile");
    }

    // --- history walk ----------------------------------------------------

    #[test]
    fn up_recalls_newest_then_older_then_down_returns_to_draft() {
        let mut h = History::new();
        h.push("first");
        h.push("second");
        let mut b = CmdlineBuffer::new();
        b.insert_char('x'); // a live draft
        b.history_prev(&h);
        assert_eq!(b.text(), "x"); // "x" is the prefix; nothing matches -> no move
                                   // (draft "x" filters; no history starts with x)
        // Now with an empty draft, the whole history is walkable.
        let mut b = CmdlineBuffer::new();
        b.history_prev(&h);
        assert_eq!(b.text(), "second");
        b.history_prev(&h);
        assert_eq!(b.text(), "first");
        b.history_prev(&h); // oldest already -> stays
        assert_eq!(b.text(), "first");
        b.history_next(&h);
        assert_eq!(b.text(), "second");
        b.history_next(&h); // past newest -> back to the (empty) draft
        assert_eq!(b.text(), "");
    }

    #[test]
    fn prefix_filtered_history_only_cycles_matching_entries() {
        let mut h = History::new();
        h.push("edit a");
        h.push("write");
        h.push("edit b");
        let mut b = CmdlineBuffer::new();
        b.insert_char('e'); // draft "e" -> only "edit *" entries recalled
        b.history_prev(&h);
        assert_eq!(b.text(), "edit b");
        b.history_prev(&h);
        assert_eq!(b.text(), "edit a");
        b.history_prev(&h); // "write" is skipped; oldest match reached
        assert_eq!(b.text(), "edit a");
    }

    #[test]
    fn editing_ends_the_history_walk() {
        let mut h = History::new();
        h.push("write");
        let mut b = CmdlineBuffer::new();
        b.history_prev(&h);
        assert_eq!(b.text(), "write");
        b.backspace(); // edit -> walk ends
        assert_eq!(b.text(), "writ");
        // a fresh <Up> now starts from "writ" and finds "write"
        b.history_prev(&h);
        assert_eq!(b.text(), "write");
    }

    // --- completion context + cycle -------------------------------------

    #[test]
    fn context_of_a_bare_word_is_name_completion() {
        let b = buf("wr", 2);
        let ctx = b.completion_context();
        assert_eq!(ctx, CompletionContext { start: 0, prefix: "wr".into(), command: None });
    }

    #[test]
    fn context_after_a_space_is_argument_completion() {
        let b = buf("e src/l", 7);
        let ctx = b.completion_context();
        assert_eq!(ctx.command.as_deref(), Some("e"));
        assert_eq!(ctx.prefix, "src/l");
        assert_eq!(ctx.start, 2);
    }

    #[test]
    fn begin_and_cycle_completion_replaces_the_token() {
        let mut b = buf("w", 1);
        b.begin_completion(0, vec!["w".into(), "wq".into(), "write".into()]);
        assert_eq!(b.text(), "w");
        assert!(b.cycle_completion(true));
        assert_eq!(b.text(), "wq");
        assert_eq!(b.cursor(), 2);
        b.cycle_completion(true);
        assert_eq!(b.text(), "write");
        b.cycle_completion(true); // wraps
        assert_eq!(b.text(), "w");
        b.cycle_completion(false); // backwards wraps to end
        assert_eq!(b.text(), "write");
    }

    #[test]
    fn cycle_returns_false_when_no_cycle_is_open() {
        let mut b = buf("w", 1);
        assert!(!b.cycle_completion(true));
    }

    #[test]
    fn a_keystroke_ends_the_completion_cycle() {
        let mut b = buf("e ", 2);
        b.begin_completion(2, vec!["alpha".into(), "bravo".into()]);
        assert_eq!(b.text(), "e alpha");
        b.insert_char('x'); // typing ends the cycle
        assert!(b.active_completions().is_none());
        assert_eq!(b.text(), "e alphax");
    }
}
