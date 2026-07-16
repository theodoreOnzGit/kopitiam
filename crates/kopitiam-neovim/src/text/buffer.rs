//! The rope-backed text buffer: kvim's single source of truth for "what is
//! in this file right now."
//!
//! [`Buffer`] itself is mostly orchestration. The three hard problems it
//! delegates:
//!
//! * [`grapheme`] — turning a [`Position`] (line, grapheme column) into a
//!   rope char offset and back, which is the only place `Position::col`'s
//!   unit (grapheme clusters, not bytes or `char`s) has to be handled.
//! * [`undo`] — the branching undo tree and edit grouping.
//! * [`mark`] — how a mark's rope offset moves when an edit lands before,
//!   after, or on top of it.
//! * [`line_ending`] — LF/CRLF detection and preserving it on save.
//!
//! See each module's docs for the reasoning; this file is the glue.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use ropey::Rope;

use crate::core::{Edit, Error, Position, Range, Result};

use super::grapheme;
use super::line_ending::LineEnding;
use super::mark;
use super::undo::{NodeId, UndoTree};

/// A rope-backed text buffer with branching undo and shift-on-edit marks.
///
/// `Buffer` is the text engine's entire public surface; every other `kvim`
/// subsystem that touches file content goes through it rather than the
/// underlying [`ropey::Rope`] directly, so this is the only place that has
/// to reconcile [`Position`]'s grapheme-column addressing with the rope's
/// char-offset addressing. See the [module docs](self) for how the pieces
/// fit together.
pub struct Buffer {
    rope: Rope,
    path: Option<PathBuf>,
    line_ending: LineEnding,
    /// Marks stored as rope **char** offsets, not `Position`s — an offset
    /// shifts with one integer add on every edit; a `Position` would need
    /// to be re-derived across any edit that changes the line count.
    /// Converted to/from `Position` only at [`Buffer::mark`] /
    /// [`Buffer::set_mark`].
    marks: HashMap<char, usize>,
    undo: UndoTree,
    /// The undo-tree node this buffer looked like the last time it was
    /// saved (or, for a buffer that has never been saved, the root — a
    /// fresh buffer is trivially "not modified" relative to itself).
    /// [`Buffer::is_modified`] is exactly `undo.current_id() != saved_at`.
    ///
    /// Comparing tree *positions* rather than flipping a `bool` on every
    /// edit means undoing back to exactly the state that was last saved
    /// correctly reports "not modified" again, instead of staying stuck on
    /// "modified" the way a naive dirty flag would.
    saved_at: NodeId,
}

impl Default for Buffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Buffer {
    /// An empty buffer with no associated file.
    pub fn new() -> Self {
        Self::from_str("")
    }

    /// A buffer over `text`, with no associated file. The line ending is
    /// autodetected from `text` itself — see [`LineEnding::detect`].
    // This name is part of the crate-wide `Buffer` API contract and is
    // infallible by design, so it deliberately does not implement
    // `std::str::FromStr` (which returns `Result`) despite the name match.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Self {
        let undo = UndoTree::new();
        Self {
            rope: Rope::from_str(text),
            path: None,
            line_ending: LineEnding::detect(text),
            marks: HashMap::new(),
            saved_at: undo.current_id(),
            undo,
        }
    }

    /// Reads `path` as UTF-8 and builds a buffer over its contents.
    ///
    /// Invalid UTF-8 surfaces as [`Error::Io`] (via `std::fs::read_to_string`,
    /// which reports it as [`std::io::ErrorKind::InvalidData`]) rather than
    /// a dedicated error variant — `core::Error` is owned by another part of
    /// this crate and this text engine must not add variants to it.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut buffer = Self::from_str(&text);
        buffer.path = Some(path.to_path_buf());
        Ok(buffer)
    }

    /// Writes the buffer back to the path it was opened or last saved with.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] wrapping an [`std::io::ErrorKind::InvalidInput`]
    /// error if the buffer has no associated path (a brand-new, never-saved
    /// buffer) — use [`Buffer::save_as`] instead. `core::Error` has no
    /// dedicated "no path" variant and this crate must not add one to it, so
    /// this reuses `Io`, which is a reasonable fit: the underlying problem
    /// really is "there is nowhere to write to."
    pub fn save(&mut self) -> Result<()> {
        let path = self
            .path
            .clone()
            .ok_or_else(|| Error::Io(io::Error::new(io::ErrorKind::InvalidInput, "buffer has no path; use save_as")))?;
        self.write_to(&path)?;
        self.saved_at = self.undo.current_id();
        Ok(())
    }

    /// Writes the buffer to `path` and adopts it as the buffer's path for
    /// future [`Buffer::save`] calls.
    pub fn save_as(&mut self, path: &Path) -> Result<()> {
        self.write_to(path)?;
        self.path = Some(path.to_path_buf());
        self.saved_at = self.undo.current_id();
        Ok(())
    }

    fn write_to(&self, path: &Path) -> Result<()> {
        // No line-ending conversion happens here: the rope already holds
        // whichever ending `self.line_ending` says it should (existing
        // bytes were never rewritten; newly inserted text was normalized to
        // match on the way in — see `LineEnding::normalize` and
        // `Buffer::raw_apply`). Writing the rope out verbatim is therefore
        // exactly "preserve on save."
        let file = fs::File::create(path)?;
        self.rope.write_to(io::BufWriter::new(file))?;
        Ok(())
    }

    /// The path this buffer was opened from or last saved to, if any.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Whether the buffer differs from its last-saved (or, if never saved,
    /// initial) state. See the doc comment on [`Buffer::saved_at`] for why
    /// this is exact even across undo.
    pub fn is_modified(&self) -> bool {
        self.undo.current_id() != self.saved_at
    }

    /// Number of lines in the buffer. Always at least 1, even for an empty
    /// buffer (an empty buffer is one empty line, matching every other
    /// operation's "there is always a line 0 to put the cursor on").
    ///
    /// This is `ropey`'s own line count, which means a file ending in a
    /// trailing newline (`"a\n"`) reports **one more** line than `wc -l`
    /// (2, not 1) — the trailing `\n` terminates line 0 and there is a real,
    /// addressable, empty line 1 after it. This matches how the rope
    /// actually stores the content (and how Helix, VS Code, and friends
    /// report it): a line is genuinely nothing more than "the text before
    /// the next line break, or before the end of the rope." Vim's own line
    /// count instead special-cases this via a separate `noeol` flag it
    /// tracks per buffer, so that typing `A<CR>` after `"a"` and loading a
    /// file that is literally `"a\n"` can be told apart even though both
    /// produce the same rope content. Replicating that would mean carrying
    /// an extra flag through every edit for a genuinely rare distinction;
    /// this text engine deliberately does not.
    pub fn line_count(&self) -> usize {
        self.rope.len_lines()
    }

    /// The content of `line`, without its trailing line break, or `None` if
    /// `line >= line_count()`.
    pub fn line(&self, line: usize) -> Option<String> {
        self.line_content(line)
    }

    /// Length of `line` in **grapheme clusters** (not bytes, not `char`s),
    /// or 0 if `line >= line_count()`.
    pub fn line_len(&self, line: usize) -> usize {
        self.line_content(line).map(|s| grapheme::grapheme_len(&s)).unwrap_or(0)
    }

    /// The buffer's full text, line endings included exactly as stored.
    pub fn text(&self) -> String {
        self.rope.to_string()
    }

    /// The text within `range`, which is normalized (so the caller doesn't
    /// need to worry about anchor/head order) and clamped to valid buffer
    /// positions rather than erroring, since this method's signature has no
    /// room to report an error. Use [`Buffer::clamp`] yourself first if you
    /// need to distinguish "clamped" from "was already valid."
    pub fn slice(&self, range: Range) -> String {
        let (start, end) = range.normalized();
        let start = self.clamp(start);
        let end = self.clamp(end);
        let start_c = self.position_to_char(start).expect("clamp() always produces a valid position");
        let end_c = self.position_to_char(end).expect("clamp() always produces a valid position");
        self.rope.slice(start_c..end_c).to_string()
    }

    /// The grapheme cluster at `pos`, or `None` if `pos` is out of bounds or
    /// names the (valid, but cluster-less) end-of-line position.
    pub fn grapheme_at(&self, pos: Position) -> Option<String> {
        let line = self.line_content(pos.line)?;
        grapheme::grapheme_at(&line, pos.col).map(str::to_string)
    }

    /// Applies `edit` and returns the cursor position immediately after it
    /// (the end of whatever text was inserted — an empty-text delete lands
    /// the cursor at the deletion's start, which is the same thing since
    /// then there is no inserted text to be "after").
    ///
    /// Handles insertion (`edit.range` empty), deletion (`edit.text`
    /// empty), and replacement uniformly, per [`Edit`]'s own contract — all
    /// three are the same rope operation (remove the range, insert the
    /// text) with different arguments.
    ///
    /// # Errors
    ///
    /// [`Error::PositionOutOfBounds`] if either end of `edit.range` names a
    /// line `>= line_count()`, or a column past the valid end-of-line
    /// position for its line. Callers that received a `Position` from user
    /// input rather than from this buffer itself should run it through
    /// [`Buffer::clamp`] first if they'd rather clamp than error.
    pub fn apply(&mut self, edit: Edit) -> Result<Position> {
        let (pos, forward, inverse) = self.raw_apply(&edit)?;
        self.undo.record(forward, inverse);
        Ok(pos)
    }

    /// Mutates the rope and fixes up marks for one edit, without touching
    /// the undo tree. Returns the cursor position after the edit, the
    /// *realized* forward edit (same range, but with `text` normalized to
    /// the buffer's line ending — see [`LineEnding::normalize`]), and its
    /// inverse. Shared by [`Buffer::apply`] (which records the pair into
    /// the undo tree) and [`Buffer::undo`]/[`Buffer::redo`] (which replay
    /// already-recorded edits and must not record them again).
    fn raw_apply(&mut self, edit: &Edit) -> Result<(Position, Edit, Edit)> {
        let (start_pos, end_pos) = edit.range.normalized();
        let start = self
            .position_to_char(start_pos)
            .ok_or(Error::PositionOutOfBounds { pos: start_pos, lines: self.line_count() })?;
        let end = self
            .position_to_char(end_pos)
            .ok_or(Error::PositionOutOfBounds { pos: end_pos, lines: self.line_count() })?;

        let old_text = self.rope.slice(start..end).to_string();
        let new_text = self.line_ending.normalize(&edit.text).into_owned();
        let new_len = new_text.chars().count();

        if end > start {
            self.rope.remove(start..end);
        }
        if !new_text.is_empty() {
            self.rope.insert(start, &new_text);
        }

        mark::shift_all(&mut self.marks, start, end, new_len);

        // Positions computed from here on read the rope in its POST-edit
        // state, which is exactly what both the returned cursor position
        // and the inverse edit's range need to be expressed in.
        let inverse_start = self.char_to_position(start);
        let inverse_end = self.char_to_position(start + new_len);

        let forward = Edit { range: Range::new(start_pos, end_pos), text: new_text };
        let inverse = Edit { range: Range::new(inverse_start, inverse_end), text: old_text };
        Ok((inverse_end, forward, inverse))
    }

    /// Undoes the most recent edit (or grouped edit session) and returns
    /// the cursor position afterward.
    pub fn undo(&mut self) -> Result<Position> {
        let (edits, _) = self.undo.undo().ok_or(Error::NothingToUndo)?;
        self.replay(&edits)
    }

    /// Redoes the most recently undone edit (or grouped session) and
    /// returns the cursor position afterward. Redo always follows the
    /// most-recently-touched branch of the undo tree — see the [`undo`]
    /// module docs for why an intervening edit doesn't destroy other
    /// branches, only stops being the one redo reaches by default.
    pub fn redo(&mut self) -> Result<Position> {
        let (edits, _) = self.undo.redo().ok_or(Error::NothingToRedo)?;
        self.replay(&edits)
    }

    /// Re-applies a sequence of already-recorded edits (an undo group's
    /// worth of inverses, or forwards) to the rope, without touching the
    /// undo tree — it already knows about them.
    fn replay(&mut self, edits: &[Edit]) -> Result<Position> {
        debug_assert!(!edits.is_empty(), "the undo tree never commits an empty node");
        let mut pos = Position::ORIGIN;
        for edit in edits {
            let (p, _forward, _inverse) = self.raw_apply(edit)?;
            pos = p;
        }
        Ok(pos)
    }

    /// Opens an undo group: edits applied until the matching
    /// [`Buffer::end_undo_group`] coalesce into a single `undo()`/`redo()`
    /// step, the way vim treats a whole insert-mode session as one change.
    /// Nests — see [`undo::UndoTree::begin_group`](super::undo::UndoTree::begin_group).
    pub fn begin_undo_group(&mut self) {
        self.undo.begin_group();
    }

    /// Closes one level of undo group opened by [`Buffer::begin_undo_group`].
    pub fn end_undo_group(&mut self) {
        self.undo.end_group();
    }

    /// Sets mark `name` to `pos`, clamped into the buffer. Marks named `'`
    /// and `` ` `` etc. carry vim-specific meaning at the editor layer;
    /// `Buffer` itself treats every `char` as an equally valid mark name.
    pub fn set_mark(&mut self, name: char, pos: Position) {
        let pos = self.clamp(pos);
        if let Some(offset) = self.position_to_char(pos) {
            self.marks.insert(name, offset);
        }
    }

    /// The current position of mark `name`, or `None` if it was never set.
    /// Automatically kept in sync with edits — see the [`mark`] module docs.
    pub fn mark(&self, name: char) -> Option<Position> {
        self.marks.get(&name).map(|&offset| self.char_to_position(offset))
    }

    /// Every set lowercase (`a`–`z`) mark, as `(name, position)` pairs in no
    /// particular order — the file-local marks the `['`/`]'`/`` [` ``/`` ]` ``
    /// motions navigate between. Uppercase (global) and special marks are
    /// excluded, matching vim's "next mark" commands, which only visit
    /// lowercase marks.
    pub fn lowercase_marks(&self) -> Vec<(char, Position)> {
        self.marks
            .iter()
            .filter(|(name, _)| name.is_ascii_lowercase())
            .map(|(&name, &offset)| (name, self.char_to_position(offset)))
            .collect()
    }

    /// Clamps `pos` into the buffer: the line is capped to the last valid
    /// line, and the column to that line's length in graphemes (i.e. the
    /// valid end-of-line position). Never panics, regardless of how far out
    /// of range `pos` is — including `usize::MAX` in either field.
    pub fn clamp(&self, pos: Position) -> Position {
        let last_line = self.line_count().saturating_sub(1);
        let line = pos.line.min(last_line);
        let col = pos.col.min(self.line_len(line));
        Position::new(line, col)
    }

    /// Content of `line`, without its trailing line break (`\n` or `\r\n`),
    /// as an owned `String`. `None` if `line >= line_count()`. Every other
    /// per-line accessor (`line`, `line_len`, `grapheme_at`,
    /// `position_to_char`, `char_to_position`) goes through this so the
    /// line-ending-stripping logic lives in exactly one place.
    fn line_content(&self, line: usize) -> Option<String> {
        if line >= self.line_count() {
            return None;
        }
        let raw = self.rope.line(line);
        let len = raw.len_chars();
        let content_len = if len == 0 {
            0
        } else if raw.char(len - 1) == '\n' {
            if len >= 2 && raw.char(len - 2) == '\r' { len - 2 } else { len - 1 }
        } else {
            len
        };
        Some(raw.slice(0..content_len).to_string())
    }

    /// Converts `pos` to a rope char offset, or `None` if `pos.line` is out
    /// of range or `pos.col` is past that line's valid end-of-line column.
    /// This is the only function in `Buffer` that combines a rope-level
    /// line lookup with a [`grapheme`] column conversion; every read/write
    /// path funnels through it or its inverse, [`Buffer::char_to_position`].
    fn position_to_char(&self, pos: Position) -> Option<usize> {
        let content = self.line_content(pos.line)?;
        if pos.col > grapheme::grapheme_len(&content) {
            return None;
        }
        let line_start = self.rope.line_to_char(pos.line);
        Some(line_start + grapheme::col_to_char(&content, pos.col))
    }

    /// Converts a rope char offset back to a `Position`. Infallible: any
    /// offset up to and including `rope.len_chars()` names a real position
    /// (the latter being the buffer's own end).
    fn char_to_position(&self, char_idx: usize) -> Position {
        let char_idx = char_idx.min(self.rope.len_chars());
        let line = self.rope.char_to_line(char_idx);
        let line_start = self.rope.line_to_char(line);
        let offset_in_line = char_idx - line_start;
        let content = self.line_content(line).unwrap_or_default();
        let clamped_offset = offset_in_line.min(content.chars().count());
        Position::new(line, grapheme::char_to_col(&content, clamped_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    // ---- Empty buffer -----------------------------------------------

    #[test]
    fn empty_buffer_has_one_empty_line() {
        let buf = Buffer::new();
        assert_eq!(buf.line_count(), 1);
        assert_eq!(buf.line(0), Some(String::new()));
        assert_eq!(buf.line_len(0), 0);
        assert_eq!(buf.text(), "");
    }

    // ---- Grapheme correctness (integration-level; see text::grapheme for
    // the exhaustive unit-level coverage) --------------------------------

    #[test]
    fn line_len_and_grapheme_at_use_grapheme_clusters() {
        let buf = Buffer::from_str("日本語\n👨\u{200D}👩\u{200D}👧\ne\u{0301}");
        assert_eq!(buf.line_len(0), 3);
        assert_eq!(buf.line_len(1), 1);
        assert_eq!(buf.line_len(2), 1);
        assert_eq!(buf.grapheme_at(Position::new(1, 0)).as_deref(), Some("👨\u{200D}👩\u{200D}👧"));
        assert_eq!(buf.grapheme_at(Position::new(1, 1)), None);
    }

    // ---- apply / edit uniformity ----------------------------------------

    #[test]
    fn insert_delete_and_replace_all_go_through_apply() {
        let mut buf = Buffer::from_str("hello");
        buf.apply(Edit::insert(Position::new(0, 5), " world")).unwrap();
        assert_eq!(buf.text(), "hello world");

        buf.apply(Edit::delete(Range::new(Position::new(0, 5), Position::new(0, 11)))).unwrap();
        assert_eq!(buf.text(), "hello");

        buf.apply(Edit::replace(Range::new(Position::new(0, 0), Position::new(0, 5)), "bye")).unwrap();
        assert_eq!(buf.text(), "bye");
    }

    #[test]
    fn apply_reports_out_of_bounds_positions() {
        let mut buf = Buffer::from_str("ab");
        let err = buf.apply(Edit::insert(Position::new(5, 0), "x")).unwrap_err();
        assert!(matches!(err, Error::PositionOutOfBounds { .. }));

        let err = buf.apply(Edit::insert(Position::new(0, 10), "x")).unwrap_err();
        assert!(matches!(err, Error::PositionOutOfBounds { .. }));
    }

    #[test]
    fn apply_returns_the_cursor_position_after_the_edit() {
        let mut buf = Buffer::from_str("");
        let pos = buf.apply(Edit::insert(Position::ORIGIN, "abc")).unwrap();
        assert_eq!(pos, Position::new(0, 3));

        let pos = buf.apply(Edit::insert(Position::new(0, 3), "\ndef")).unwrap();
        assert_eq!(pos, Position::new(1, 3));
    }

    // ---- undo / redo integration ----------------------------------------

    #[test]
    fn undo_and_redo_round_trip_through_apply() {
        let mut buf = Buffer::from_str("");
        buf.apply(Edit::insert(Position::ORIGIN, "hello")).unwrap();
        assert_eq!(buf.text(), "hello");

        let pos = buf.undo().unwrap();
        assert_eq!(buf.text(), "");
        assert_eq!(pos, Position::ORIGIN);

        let pos = buf.redo().unwrap();
        assert_eq!(buf.text(), "hello");
        assert_eq!(pos, Position::new(0, 5));

        assert!(matches!(buf.redo(), Err(Error::NothingToRedo)));
    }

    #[test]
    fn undo_on_a_pristine_buffer_errors() {
        let mut buf = Buffer::new();
        assert!(matches!(buf.undo(), Err(Error::NothingToUndo)));
    }

    #[test]
    fn an_insert_session_of_ten_chars_undoes_in_one_step() {
        let mut buf = Buffer::from_str("");
        buf.begin_undo_group();
        for ch in "1234567890".chars() {
            let at = Position::new(0, buf.line_len(0));
            buf.apply(Edit::insert(at, ch.to_string())).unwrap();
        }
        buf.end_undo_group();
        assert_eq!(buf.text(), "1234567890");

        buf.undo().unwrap();
        assert_eq!(buf.text(), "", "the whole grouped session must undo in a single step");
        assert!(matches!(buf.undo(), Err(Error::NothingToUndo)));
    }

    #[test]
    fn diverging_after_undo_does_not_destroy_the_old_branch() {
        let mut buf = Buffer::from_str("");
        buf.apply(Edit::insert(Position::ORIGIN, "A")).unwrap();
        buf.apply(Edit::insert(Position::new(0, 1), "B")).unwrap();
        assert_eq!(buf.text(), "AB");

        buf.undo().unwrap();
        assert_eq!(buf.text(), "A");

        buf.apply(Edit::insert(Position::new(0, 1), "C")).unwrap(); // diverge
        assert_eq!(buf.text(), "AC");

        buf.undo().unwrap();
        assert_eq!(buf.text(), "A");
        buf.redo().unwrap();
        assert_eq!(buf.text(), "AC", "redo follows the newest branch");
        // The (b) branch's continued existence as a sibling is proven at the
        // UndoTree level in text::undo's own tests, which can see node
        // structure that Buffer deliberately does not expose.
    }

    // ---- marks ------------------------------------------------------

    #[test]
    fn marks_shift_with_edits_before_and_after_them() {
        let mut buf = Buffer::from_str("hello world");
        buf.set_mark('a', Position::new(0, 6)); // 'w' in "world"
        buf.set_mark('b', Position::new(0, 0)); // 'h' in "hello"

        buf.apply(Edit::insert(Position::new(0, 0), "XXX")).unwrap();
        assert_eq!(buf.mark('a'), Some(Position::new(0, 9)), "mark after the insertion point shifts right");
        assert_eq!(buf.mark('b'), Some(Position::new(0, 0)), "mark at the insertion point does not shift");
    }

    #[test]
    fn mark_inside_a_deleted_range_clamps_to_the_deletion_start() {
        let mut buf = Buffer::from_str("hello world");
        buf.set_mark('a', Position::new(0, 8)); // inside "world"
        buf.apply(Edit::delete(Range::new(Position::new(0, 5), Position::new(0, 11)))).unwrap();
        assert_eq!(buf.mark('a'), Some(Position::new(0, 5)));
    }

    #[test]
    fn unset_mark_is_none() {
        let buf = Buffer::new();
        assert_eq!(buf.mark('z'), None);
    }

    // ---- clamp --------------------------------------------------------

    #[test]
    fn clamp_never_panics_and_always_returns_a_valid_position() {
        let buf = Buffer::from_str("a\nbb\nccc");
        for pos in [
            Position::new(usize::MAX, usize::MAX),
            Position::new(0, usize::MAX),
            Position::new(usize::MAX, 0),
            Position::ORIGIN,
            Position::new(1, 1),
        ] {
            let clamped = buf.clamp(pos);
            assert!(clamped.line < buf.line_count());
            assert!(clamped.col <= buf.line_len(clamped.line));
        }
    }

    // ---- save / load / line endings ------------------------------------

    #[test]
    fn save_without_a_path_errors() {
        let mut buf = Buffer::new();
        assert!(matches!(buf.save(), Err(Error::Io(_))));
    }

    #[test]
    fn save_as_then_save_round_trips_through_a_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");

        let mut buf = Buffer::from_str("hello");
        buf.save_as(&path).unwrap();
        assert_eq!(buf.path(), Some(path.as_path()));
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");

        buf.apply(Edit::insert(Position::new(0, 5), "!")).unwrap();
        buf.save().unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello!");
    }

    #[test]
    fn is_modified_tracks_edits_and_resets_on_save_or_undo_back_to_saved_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        let mut buf = Buffer::from_str("hello");
        assert!(!buf.is_modified());

        buf.apply(Edit::insert(Position::new(0, 5), "!")).unwrap();
        assert!(buf.is_modified());

        buf.save_as(&path).unwrap();
        assert!(!buf.is_modified());

        buf.apply(Edit::insert(Position::new(0, 6), "?")).unwrap();
        assert!(buf.is_modified());

        buf.undo().unwrap();
        assert!(!buf.is_modified(), "undoing back to the saved state must clear modified");
    }

    #[test]
    fn crlf_round_trips_byte_for_byte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crlf.txt");
        let original: &[u8] = b"line one\r\nline two\r\n";
        fs::write(&path, original).unwrap();

        let mut buf = Buffer::from_file(&path).unwrap();
        buf.save().unwrap();

        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn typing_enter_in_a_crlf_buffer_inserts_crlf_not_bare_lf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crlf.txt");
        fs::write(&path, b"ab\r\n").unwrap();

        let mut buf = Buffer::from_file(&path).unwrap();
        buf.apply(Edit::insert(Position::new(0, 1), "\n")).unwrap();
        buf.save().unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"a\r\nb\r\n");
    }

    #[test]
    fn crlf_file_is_not_touched_by_an_unrelated_edit() {
        // A CRLF file's EXISTING line endings must never be rewritten, even
        // though newly typed newlines get normalized to match.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crlf.txt");
        fs::write(&path, b"aaa\r\nbbb\r\n").unwrap();

        let mut buf = Buffer::from_file(&path).unwrap();
        buf.apply(Edit::insert(Position::new(0, 3), "!")).unwrap();
        buf.save().unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"aaa!\r\nbbb\r\n");
    }

    // ---- property test: Buffer vs. a naive String reference -------------
    //
    // Applies a long deterministic sequence of random inserts/deletes/
    // replaces to both a `Buffer` and a plain `String`, asserting they stay
    // identical after every single step. Restricted to ASCII content
    // (including bare `\n`, to exercise line-count changes) so that a
    // grapheme column, a `char` index, and a byte index all coincide —
    // that keeps the reference model's own position bookkeeping trivially
    // correct, so any divergence is a `Buffer` bug, not a test-harness bug.
    // Grapheme-cluster correctness itself is covered exhaustively and
    // separately in `text::grapheme`.

    /// A tiny deterministic PRNG (xorshift64*) — no new dependency, and
    /// reproducible across runs and machines per CLAUDE.md's determinism
    /// preference.
    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1) // must be nonzero for xorshift to cycle
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        /// A value in `0..bound`, or 0 if `bound == 0`.
        fn below(&mut self, bound: usize) -> usize {
            if bound == 0 {
                0
            } else {
                (self.next_u64() % bound as u64) as usize
            }
        }
    }

    /// Random ASCII text (letters, digits, space, newline) up to `max_len`
    /// chars, so generated edits sometimes split or merge lines.
    fn random_text(rng: &mut Xorshift64, max_len: usize) -> String {
        const ALPHABET: &[u8] = b"abcXYZ012 \n";
        let len = rng.below(max_len + 1);
        (0..len).map(|_| ALPHABET[rng.below(ALPHABET.len())] as char).collect()
    }

    /// The `Position` of char offset `at` within `reference`, computed by
    /// walking the reference string itself — a model of `Buffer`'s own
    /// line/col addressing, but derived independently of any `Buffer` code.
    fn position_at(reference: &str, at: usize) -> Position {
        let mut line = 0;
        let mut col = 0;
        for (i, ch) in reference.chars().enumerate() {
            if i == at {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        Position::new(line, col)
    }

    #[test]
    fn property_buffer_matches_a_naive_string_reference_under_random_edits() {
        let mut rng = Xorshift64::new(0xC0FFEE_D15EA5E);
        let mut reference = String::new();
        let mut buffer = Buffer::new();

        for step in 0..1000 {
            let len = reference.chars().count();
            let a = rng.below(len + 1);
            let b = rng.below(len + 1);
            let (mut start, end) = if a <= b { (a, b) } else { (b, a) };

            // op 0: pure insert (force a zero-width range); op 1: delete
            // (empty replacement text); op 2: replace (both non-trivial).
            let op = rng.below(3);
            if op == 0 {
                start = end; // zero-width: insertion
            }
            let text = if op == 1 { String::new() } else { random_text(&mut rng, 6) };

            let start_pos = position_at(&reference, start);
            let end_pos = position_at(&reference, end);
            let edit = Edit { range: Range::new(start_pos, end_pos), text: text.clone() };

            buffer.apply(edit).unwrap_or_else(|e| panic!("step {step}: apply failed: {e}"));
            reference.replace_range(start..end, &text);

            assert_eq!(
                buffer.text(),
                reference,
                "step {step}: mismatch after op={op} start={start} end={end} text={text:?}"
            );
        }
    }
}
