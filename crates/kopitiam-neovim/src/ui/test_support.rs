//! Test-only fakes for the [`crate::ui::event::EditorHost`] and
//! [`crate::ui::event::BufferView`] seam traits.
//!
//! Compiled only under `#[cfg(test)]` (see `ui/mod.rs`), so none of this
//! reaches the release binary. Centralized here, rather than duplicated
//! per-submodule, because several `ui/` submodules (`textarea`, `app`,
//! `window`) all need "a buffer with some lines" or "an editor that does
//! something predictable when fed a key" to test against, and per the
//! module docs on [`crate::ui`], writing every renderer against the trait
//! rather than a concrete `editor`/`text` type is the entire point of this
//! seam — these fakes are what makes that testable *today*, before
//! `crate::editor` exists.

use std::path::{Path, PathBuf};

use unicode_segmentation::UnicodeSegmentation;

use crate::core::{Mode, Position, Range};
use crate::ui::event::{BufferView, EditorHost, HostResponse, Key, KeyPress};

/// A minimal in-memory buffer implementing [`BufferView`], for tests.
#[derive(Debug, Clone)]
pub struct FakeBuffer {
    lines: Vec<String>,
    modified: bool,
    path: Option<PathBuf>,
}

impl FakeBuffer {
    pub fn new(lines: Vec<String>) -> Self {
        // A buffer is never zero lines, matching the frozen `text::Buffer`
        // contract this fake stands in for (an empty buffer is one empty
        // line, not no lines).
        let lines = if lines.is_empty() { vec![String::new()] } else { lines };
        Self { lines, modified: false, path: None }
    }

    pub fn with_modified(mut self, modified: bool) -> Self {
        self.modified = modified;
        self
    }

    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }
}

impl BufferView for FakeBuffer {
    fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn line(&self, n: usize) -> Option<String> {
        self.lines.get(n).cloned()
    }

    fn line_len(&self, n: usize) -> usize {
        self.lines.get(n).map(|l| unicode_segmentation::UnicodeSegmentation::graphemes(l.as_str(), true).count()).unwrap_or(0)
    }

    fn is_modified(&self) -> bool {
        self.modified
    }

    fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

/// A scripted [`EditorHost`] fake: `j`/`k` move the cursor, `q` quits,
/// anything else reports `Unchanged`. Enough behaviour to drive the event
/// loop and statusline tests without depending on `crate::editor`.
pub struct FakeHost {
    pub buffer: FakeBuffer,
    pub mode: Mode,
    pub cursor: Position,
    /// Records every key handed to `handle_key`, so tests can assert on
    /// what the event loop actually forwarded.
    pub received: Vec<KeyPress>,
    /// Records every path handed to `open` — how the overlay tests assert that
    /// pressing `o` on a file in the tree really did reach the editor, and did
    /// not merely close the sidebar and look plausible.
    pub opened: Vec<PathBuf>,
    /// The next `handle_key` returns this instead of its scripted response, once.
    /// Lets a test drive the `<leader>e` → [`HostResponse::Action`] path without
    /// reimplementing the editor's keymap engine in a fake.
    pub next_response: Option<HostResponse>,
    /// What the editor would report as typed at the `:` prompt. Set it, set
    /// [`FakeHost::mode`] to [`Mode::Command`], and the renderer must paint it —
    /// which is the assertion that was missing when `:Neotree` echoed nothing.
    pub command_line: Option<String>,
    /// The command-line caret position (grapheme offset) the editor would
    /// report. `None` falls back to end-of-text, so most tests can ignore it;
    /// the mid-line-cursor test sets it to prove the caret follows.
    pub command_cursor: Option<usize>,
    /// The `<Tab>` completion candidates + selected index the editor would
    /// report, for the wildmenu strip. `None` means no cycle is open.
    pub command_completions: Option<(Vec<String>, usize)>,
    /// The visual selection the editor would report. Set it alongside a visual
    /// [`FakeHost::mode`] to test the highlight.
    pub selection: Option<(Position, Position)>,
}

impl FakeHost {
    pub fn new(buffer: FakeBuffer) -> Self {
        Self {
            buffer,
            mode: Mode::Normal,
            cursor: Position::ORIGIN,
            received: Vec::new(),
            opened: Vec::new(),
            next_response: None,
            command_line: None,
            command_cursor: None,
            command_completions: None,
            selection: None,
        }
    }

    /// Scripts the response to the next key, whatever that key is.
    pub fn answer_next_with(&mut self, response: HostResponse) {
        self.next_response = Some(response);
    }

    /// Inserts `c` at the cursor and advances it — the insert-mode typing the
    /// completion tests drive. Single-line (grapheme-indexed); the fixtures only
    /// ever type on one line.
    fn insert_char(&mut self, c: char) {
        let Some(line) = self.buffer.lines.get_mut(self.cursor.line) else { return };
        let graphemes: Vec<String> = line.graphemes(true).map(str::to_string).collect();
        let col = self.cursor.col.min(graphemes.len());
        let mut rebuilt: String = graphemes[..col].concat();
        rebuilt.push(c);
        rebuilt.push_str(&graphemes[col..].concat());
        *line = rebuilt;
        self.cursor.col = col + 1;
        self.buffer.modified = true;
    }
}

impl EditorHost for FakeHost {
    type Buffer = FakeBuffer;

    fn handle_key(&mut self, key: KeyPress) -> HostResponse {
        self.received.push(key);
        if let Some(scripted) = self.next_response.take() {
            return scripted;
        }
        // While in Insert mode, a plain character types into the buffer — the
        // behaviour the completion menu's auto-trigger observes. Checked first
        // so `i`/`j`/`k` insert literally in Insert mode (as vim does) rather
        // than being read as motions.
        if self.mode == Mode::Insert
            && !key.mods.ctrl
            && !key.mods.alt
            && let Key::Char(c) = key.key
        {
            self.insert_char(c);
            return HostResponse::Changed;
        }
        match key.key {
            Key::Char('q') => HostResponse::Quit,
            Key::Char('j') if self.cursor.line + 1 < self.buffer.line_count() => {
                self.cursor.line += 1;
                HostResponse::Changed
            }
            Key::Char('k') if self.cursor.line > 0 => {
                self.cursor.line -= 1;
                HostResponse::Changed
            }
            Key::Char('i') => {
                self.mode = Mode::Insert;
                HostResponse::Changed
            }
            Key::Escape => {
                self.mode = Mode::Normal;
                HostResponse::Changed
            }
            _ => HostResponse::Unchanged,
        }
    }

    fn mode(&self) -> Mode {
        self.mode
    }

    fn cursor(&self) -> Position {
        self.cursor
    }

    fn buffer(&self) -> &Self::Buffer {
        &self.buffer
    }

    fn command_line(&self) -> Option<&str> {
        self.command_line.as_deref()
    }

    fn command_cursor(&self) -> Option<usize> {
        // Mirror the real editor: a caret only exists while a prompt is open.
        self.command_line.as_ref().map(|line| {
            self.command_cursor.unwrap_or_else(|| line.graphemes(true).count())
        })
    }

    fn command_completions(&self) -> Option<(Vec<String>, usize)> {
        self.command_completions.clone()
    }

    fn selection(&self) -> Option<(Position, Position)> {
        self.selection
    }

    /// Replaces a **single-line** range with `text` (which may itself contain
    /// newlines, splitting the line), moves the cursor to the end of the
    /// inserted text, and returns it — enough to exercise completion-accept and
    /// snippet expansion. The fixtures only ever accept on one line, so a
    /// multi-line *range* is not modelled.
    fn replace_range(&mut self, range: Range, text: &str) -> Position {
        let (start, end) = range.normalized();
        let Some(line) = self.buffer.lines.get(start.line).cloned() else { return self.cursor };
        let graphemes: Vec<String> = line.graphemes(true).map(str::to_string).collect();
        let s = start.col.min(graphemes.len());
        let e = end.col.min(graphemes.len()).max(s);
        let prefix: String = graphemes[..s].concat();
        let suffix: String = graphemes[e..].concat();
        let combined = format!("{prefix}{text}{suffix}");
        let new_lines: Vec<String> = combined.split('\n').map(str::to_string).collect();
        let inserted: Vec<&str> = text.split('\n').collect();
        let landed = if inserted.len() == 1 {
            Position::new(start.line, s + text.graphemes(true).count())
        } else {
            let last = inserted.last().copied().unwrap_or("");
            Position::new(start.line + inserted.len() - 1, last.graphemes(true).count())
        };
        self.buffer.lines.splice(start.line..=start.line, new_lines);
        self.cursor = landed;
        self.buffer.modified = true;
        landed
    }

    fn move_cursor(&mut self, pos: Position) {
        self.cursor = pos;
    }

    /// Reads the file for real, so that the failure path (opening a directory, a
    /// missing file) is exercised by tests rather than assumed away.
    fn open(&mut self, path: &Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.opened.push(path.to_path_buf());
        self.buffer = FakeBuffer::new(text.lines().map(str::to_string).collect()).with_path(path);
        self.cursor = Position::ORIGIN;
        Ok(())
    }
}
