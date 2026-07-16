//! Crossterm → editor key translation, and the trait seam the UI renders
//! against instead of `crate::editor` directly.
//!
//! # Why a local `KeyPress` type at all
//!
//! `crate::editor` deliberately defines its own `Key` type rather than using
//! `crossterm::event::KeyEvent`, so that the modal state machine stays
//! headlessly testable without a terminal. That means this crate ends up
//! with *two* notions of "a key was pressed": crossterm's (arrives from the
//! terminal) and the editor's (consumed by `handle_key`). Something has to
//! translate between them, and per `CLAUDE.md` that translation belongs in
//! the UI, not the editor — the editor should never need to know crossterm
//! exists.
//!
//! `crate::editor` was still a placeholder module when this file was
//! written, so its `Key` type's exact shape was not yet available to
//! compile against. [`KeyPress`] below is this module's best-effort
//! prediction of that shape (a `KeyCode`-like enum plus modifier flags,
//! which is the shape essentially every modal editor's key type takes,
//! kvim's likely included). [`map_crossterm_key`] is the single function
//! that would need to change — to return `editor::Key` instead of
//! [`KeyPress`], or to feed a `KeyPress → editor::Key` conversion — once the
//! real type lands. Nothing else in `ui/` depends on crossterm's key types
//! directly; they all go through [`KeyPress`] and the [`EditorHost`] trait
//! below.
//!
//! # Why `EditorHost` / `BufferView` instead of `crate::editor::Editor`
//!
//! For the same reason: `editor::Editor` and its `EditorResponse` were being
//! designed concurrently by another agent and were not yet compilable
//! against. [`EditorHost`] states only what the renderer and event loop
//! actually need — feed it a key, read back mode/cursor/buffer — as a trait,
//! so `ui/` compiles, renders, and is unit-tested today against small fakes
//! (see the tests in [`crate::ui::app`] and [`crate::ui::textarea`]), and so
//! that wiring in the real editor later is one `impl EditorHost for
//! editor::Editor` block rather than a rewrite of the renderer.
//! [`BufferView`] mirrors the frozen `text::Buffer` read API
//! (`line_count`, `line`, `line_len`, `is_modified`, `path`) so the same
//! reasoning applies to buffer access.

use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::core::{BufferId, Mode, Position, Range, ViewportScroll, WindowCommand};
use crate::ui::cmdline::PromptKind;

/// A single logical key, independent of crossterm.
///
/// Mirrors the shape of `crossterm::event::KeyCode` closely on purpose: that
/// keeps [`map_crossterm_key`] a near-mechanical translation, which is
/// exactly what you want from a mapping function whose job is to have as few
/// interesting decisions in it as possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    Char(char),
    Enter,
    Escape,
    Backspace,
    Tab,
    BackTab,
    Delete,
    Insert,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    F(u8),
}

/// Modifier keys held during a keypress.
///
/// `shift` is tracked even though `Key::Char` already arrives
/// shift-applied (crossterm hands us `'A'`, not `'a'` + shift) because some
/// bindings care about the *physical* shift key independent of the
/// resulting character — e.g. `<S-Tab>` (BackTab) or a future `<S-F3>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Modifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
}

/// A key together with the modifiers held while it was pressed — the unit
/// [`EditorHost::handle_key`] consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyPress {
    pub key: Key,
    pub mods: Modifiers,
}

impl KeyPress {
    pub const fn new(key: Key, mods: Modifiers) -> Self {
        Self { key, mods }
    }

    /// A bare key with no modifiers held — the common case in tests.
    pub const fn plain(key: Key) -> Self {
        Self { key, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }
}

/// Translates a crossterm key event into a [`KeyPress`], or `None` when the
/// event carries no new keypress to act on.
///
/// Returns `None` for [`KeyEventKind::Release`]: most Unix terminals never
/// emit release events at all (they require the Kitty keyboard protocol to
/// be explicitly enabled), but when they are present, treating a *release*
/// as a second keypress would run every command twice. [`KeyEventKind::Repeat`]
/// (an OS-level auto-repeat while a key is held) is treated the same as
/// `Press` — from the editor's point of view a repeated `j` is just `j`
/// pressed again.
pub fn map_crossterm_key(ev: KeyEvent) -> Option<KeyPress> {
    if ev.kind == KeyEventKind::Release {
        return None;
    }

    let key = match ev.code {
        KeyCode::Char(c) => Key::Char(c),
        KeyCode::Enter => Key::Enter,
        KeyCode::Esc => Key::Escape,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        KeyCode::BackTab => Key::BackTab,
        KeyCode::Delete => Key::Delete,
        KeyCode::Insert => Key::Insert,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::F(n) => Key::F(n),
        // Media keys, caps/scroll/num lock, keypad-specific codes, and
        // anything else crossterm may add: kvim has no binding surface for
        // these yet, and inventing a `Key::Other` variant would just push
        // the "what do we do with this" decision into the editor for no
        // benefit today.
        _ => return None,
    };

    let mods = Modifiers {
        ctrl: ev.modifiers.contains(KeyModifiers::CONTROL),
        alt: ev.modifiers.contains(KeyModifiers::ALT),
        shift: ev.modifiers.contains(KeyModifiers::SHIFT),
    };

    Some(KeyPress::new(key, mods))
}

/// What the UI needs to read from, and feed keys into, the editor.
///
/// See the module docs for why this trait exists instead of a direct
/// dependency on `crate::editor::Editor`. `Buffer` is an associated type
/// (rather than `&dyn BufferView`) so implementations can return a
/// concrete, statically-dispatched buffer reference — the renderer runs on
/// every redraw and should not pay virtual-dispatch overhead for something
/// as hot as "read the current line".
pub trait EditorHost {
    type Buffer: BufferView;

    /// Feeds one key to the editor and returns what happened.
    fn handle_key(&mut self, key: KeyPress) -> HostResponse;

    /// The current mode, for the statusline label and cursor shape.
    fn mode(&self) -> Mode;

    /// The cursor's current buffer position.
    fn cursor(&self) -> Position;

    /// Read-only access to the active buffer's text.
    fn buffer(&self) -> &Self::Buffer;

    /// The id of the active buffer, so the UI can record which buffer each
    /// window is showing.
    ///
    /// Defaults to `BufferId(0)` for fakes/placeholders that own exactly one
    /// buffer and never split; the real editor overrides it.
    fn active_buffer_id(&self) -> BufferId {
        BufferId(0)
    }

    /// Read-only access to *any* open buffer by id — the seam a split window
    /// renders its (possibly different) buffer through. See
    /// [`crate::editor::Editor::buffer_by_id`] and bug `kopitiam-cj0.10.3`.
    ///
    /// Defaults to the active buffer, which is correct for a single-buffer
    /// host: with only one buffer, every window shows it.
    fn buffer_by_id(&self, _id: BufferId) -> Option<&Self::Buffer> {
        Some(self.buffer())
    }

    /// Switch the active buffer/cursor to a window's saved state, when window
    /// focus moves. Default no-op (a single-window host never calls it).
    fn set_active(&mut self, _buffer: BufferId, _cursor: Position) {}

    /// Create a fresh empty buffer and return its id (`:new`/`:vnew`).
    /// Default returns the active id — a host that cannot make buffers simply
    /// reuses the one it has.
    fn new_buffer(&mut self) -> BufferId {
        self.active_buffer_id()
    }

    /// Tell the editor how many text rows the active window shows, so
    /// `<C-d>`/`<C-f>` scroll by the right amount. Default no-op.
    fn set_viewport_height(&mut self, _lines: usize) {}

    /// Which prompt (`:`/`/`/`?`) is open, for the command-line prefix.
    /// Defaults to `Command` whenever [`EditorHost::command_line`] is `Some`,
    /// which is correct for a host that only implements `:`.
    fn command_prompt(&self) -> PromptKind {
        if self.command_line().is_some() {
            PromptKind::Command
        } else {
            PromptKind::None
        }
    }

    /// Replaces the buffer text in `range` with `text`, moves the cursor to the
    /// end of the inserted text, and returns that new cursor position.
    ///
    /// This is the buffer-mutation primitive the insert-mode completion menu and
    /// the snippet expander need: accepting a candidate replaces the typed
    /// prefix with the full label (a `replace`), and expanding a snippet
    /// replaces that same prefix with the snippet's literal text before the
    /// editor drives `<Tab>` navigation over the tabstops. It goes through the
    /// host seam — rather than the UI reaching into a buffer directly — for the
    /// same reason [`EditorHost::open`] does: the editor owns text mutation, so
    /// the UI asks it to edit rather than editing behind its back (undo history,
    /// marks, and the modified flag all stay the editor's responsibility).
    ///
    /// Defaults to a no-op that returns the current cursor, so a fake host that
    /// never exercises completion need not implement it.
    fn replace_range(&mut self, _range: Range, _text: &str) -> Position {
        self.cursor()
    }

    /// Moves the cursor to `pos` (clamped to the buffer) without changing text —
    /// used to jump between snippet tabstops. Defaults to a no-op.
    fn move_cursor(&mut self, _pos: Position) {}

    /// Opens `path` and makes it the active buffer.
    ///
    /// The UI needs this because an overlay (the file tree today; the fuzzy
    /// pickers and harpoon tomorrow) selects a *path*, and something has to turn
    /// that into an open buffer. That "something" is the editor —
    /// `editor::Editor::open` already exists and is the crate's one sanctioned
    /// read-side I/O entry point — so the seam grows a method rather than the UI
    /// growing a second way to read a file.
    ///
    /// The error is a `String`, not `crate::Error`, for the same reason
    /// [`HostResponse`] is coarser than `EditorResponse`: this trait describes
    /// what the *UI* needs, and all the UI can do with a failed open is print it.
    fn open(&mut self, path: &Path) -> Result<(), String>;

    /// The text typed so far on the `:` command line, or `None` when the editor
    /// is not in [`Mode::Command`]. Excludes the `:` itself — the prompt
    /// character is chrome, and chrome is this layer's business.
    ///
    /// # This method's absence was a bug, and a instructive one
    ///
    /// Without it, the UI had no way to ask what the user had typed, so it drew
    /// an empty prompt: **`:Neotree` was invisible while you typed it, and the
    /// whole ex-command layer was unusable.** The editor's state was right, the
    /// renderer was right, and nothing joined them — a seam bug, which is the
    /// failure mode a trait seam like this one exists to make *impossible*, and
    /// instead made silent.
    ///
    /// Note the shape of the test that missed it: 305 tests asserted on editor
    /// *state* and on widget *inputs*. None asserted on the *painted cells*. The
    /// tests added alongside this method render through `TestBackend` and assert
    /// the literal string is on screen, because that is the only assertion that
    /// could have caught this.
    ///
    /// Defaults to `None` so that a host with no command line (the placeholder,
    /// a fake in a test that does not care) need not implement it.
    fn command_line(&self) -> Option<&str> {
        None
    }

    /// Where the caret sits within the command line, as a **grapheme** offset,
    /// or `None` when no prompt is open. Paired with [`Self::command_line`] to
    /// render the caret at the right column now that it can move (`<Left>`,
    /// `<C-w>`, history recall). Defaults to "end of the typed text" so a host
    /// that only appends need not implement it — which is exactly where an
    /// append-only prompt's caret is.
    fn command_cursor(&self) -> Option<usize> {
        use unicode_segmentation::UnicodeSegmentation;
        self.command_line().map(|line| line.graphemes(true).count())
    }

    /// The `<Tab>` completion candidates currently being cycled and the index of
    /// the selected one, for a wildmenu strip, or `None` when nothing is being
    /// completed. Defaults to `None` — a host with no completion shows no menu.
    fn command_completions(&self) -> Option<(Vec<String>, usize)> {
        None
    }

    /// The visual selection as `(start, end)` in document order, or `None` when
    /// not in a visual mode. Pair it with [`EditorHost::mode`] to know *which*
    /// visual mode: the three select genuinely different things, and expanding
    /// this pair into "which cells are highlighted" is a rendering question, so
    /// it is the renderer's job — see [`crate::ui::textarea::Selection`].
    ///
    /// Missing for the same reason [`EditorHost::command_line`] was, with the
    /// same consequence: **visual mode selected text without highlighting any of
    /// it.**
    fn selection(&self) -> Option<(Position, Position)> {
        None
    }

    /// The which-key rows for the key sequence buffered so far, or empty when
    /// no multi-key mapping is pending. See [`crate::ui::whichkey`] — the popup
    /// is a passive heads-up display, so this is a pure read of editor state
    /// with no effect on what the next key does.
    ///
    /// Defaults to empty so a host with no keymap engine (a fake) need not
    /// implement it.
    fn which_key(&self) -> Vec<crate::ui::whichkey::WhichKeyRow> {
        Vec::new()
    }
}

/// What the UI does after `handle_key` returns.
///
/// This is deliberately coarser than the editor's own forthcoming
/// `EditorResponse` — the UI does not need to know *why* something changed,
/// only whether it needs to redraw, show a message, or exit. When the real
/// `editor::EditorResponse` lands, the adapter that implements
/// [`EditorHost`] for `editor::Editor` is the one place that needs to map
/// its richer variants down to this enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostResponse {
    /// State changed; redraw. The overwhelmingly common case.
    Changed,
    /// Nothing observable changed (e.g. an unmapped key in normal mode);
    /// skip the redraw to avoid unnecessary terminal writes.
    Unchanged,
    /// Show an informational message in the command-line area.
    Message(String),
    /// Show an error message in the command-line area.
    Error(String),
    /// The user asked to quit the whole editor.
    Quit,
    /// `:q`/`:wq`/`<C-w>q`: close the active window, or quit the editor if it
    /// is the last one. The distinction (quit vs. close-a-split) is the UI's
    /// to make, because only the UI knows how many windows are open.
    QuitWindow,
    /// A window-management command (`:sp`, `:vs`, `:only`, `:close`) for the
    /// UI to carry out — see [`WindowCommand`].
    Window(WindowCommand),
    /// A viewport reposition (`zz`, `zt`, `zb`, `<C-e>`, `<C-y>`) — see
    /// [`ViewportScroll`].
    Scroll(ViewportScroll),
    /// A configured keymap resolved to an [`Action`] the *UI* owns — opening the
    /// file tree, a fuzzy picker, the hop overlay.
    ///
    /// The editor deliberately hands these back rather than performing them (see
    /// `editor::EditorResponse::Action`'s docs: the editor must not depend on
    /// `plugins` or `ui`). This variant carries them the last hop, from the
    /// `EditorHost` adapter to [`crate::ui::app::App`], which is the layer that
    /// owns overlays and focus. Actions with no UI yet are answered honestly on
    /// the command line rather than silently swallowed — see
    /// `App::handle_action`.
    ///
    /// [`Action`]: crate::config::Action
    Action(crate::config::Action),
}

/// Read-only view of a buffer's text, matching the frozen `text::Buffer`
/// API this UI is allowed to depend on (`line_count`, `line`, `line_len`,
/// `is_modified`, `path`). See the module docs for why this is a trait
/// rather than a direct `&text::Buffer` reference.
pub trait BufferView {
    /// Number of lines in the buffer. A buffer always has at least one line
    /// (an empty buffer is one empty line), matching vim's own model.
    fn line_count(&self) -> usize;

    /// The text of line `n` (0-based), or `None` if out of range.
    fn line(&self, n: usize) -> Option<String>;

    /// The length of line `n` in **graphemes** (not bytes, not `char`s).
    fn line_len(&self, n: usize) -> usize;

    /// Whether the buffer has unsaved changes.
    fn is_modified(&self) -> bool;

    /// The file this buffer is backed by, if any (`None` for a scratch
    /// buffer that has never been written).
    fn path(&self) -> Option<&Path>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventState;

    fn press(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent { code, modifiers: mods, kind: KeyEventKind::Press, state: KeyEventState::NONE }
    }

    #[test]
    fn plain_char_maps_through() {
        let kp = map_crossterm_key(press(KeyCode::Char('j'), KeyModifiers::NONE)).unwrap();
        assert_eq!(kp.key, Key::Char('j'));
        assert_eq!(kp.mods, Modifiers::default());
    }

    #[test]
    fn ctrl_modifier_is_captured() {
        let kp = map_crossterm_key(press(KeyCode::Char('r'), KeyModifiers::CONTROL)).unwrap();
        assert_eq!(kp.key, Key::Char('r'));
        assert!(kp.mods.ctrl);
    }

    #[test]
    fn escape_and_enter_map_to_named_variants() {
        assert_eq!(
            map_crossterm_key(press(KeyCode::Esc, KeyModifiers::NONE)).unwrap().key,
            Key::Escape
        );
        assert_eq!(
            map_crossterm_key(press(KeyCode::Enter, KeyModifiers::NONE)).unwrap().key,
            Key::Enter
        );
    }

    #[test]
    fn release_events_are_swallowed() {
        let ev = KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        };
        assert!(map_crossterm_key(ev).is_none());
    }

    #[test]
    fn repeat_events_map_the_same_as_press() {
        let ev = KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Repeat,
            state: KeyEventState::NONE,
        };
        assert_eq!(map_crossterm_key(ev).unwrap().key, Key::Char('x'));
    }

    #[test]
    fn function_keys_carry_their_number() {
        let kp = map_crossterm_key(press(KeyCode::F(5), KeyModifiers::NONE)).unwrap();
        assert_eq!(kp.key, Key::F(5));
    }
}
