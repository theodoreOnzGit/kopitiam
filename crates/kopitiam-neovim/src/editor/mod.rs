//! The modal editing engine: kvim's heart.
//!
//! This module owns the vi grammar end to end — modes, motions, operators,
//! text objects, registers, counts, dot-repeat, macros, and ex commands —
//! and exposes exactly one entry point the UI layer drives:
//! [`Editor::handle_key`]. Everything in here is buffer-and-keystroke logic;
//! nothing renders, nothing reads a terminal, nothing touches `lsp`/`ui`/
//! `plugins` (see [`EditorResponse`] for how those layers get triggered
//! *without* a dependency edge pointing at them).
//!
//! # Map of the submodules
//!
//! * [`key`] — [`Key`], independent of `crossterm` so this whole engine is
//!   testable headlessly (see this file's `tests` module).
//! * [`motion`] — pure `(buffer, position, count) -> position` functions,
//!   each carrying whether it is exclusive/inclusive/linewise.
//! * [`operator`] — `d`/`c`/`y`/`>`/`<`/`gu`/`gU`/`g~`, each a single
//!   generic "act on this range" function.
//! * [`text_object`] — `iw`/`i(`/`it`/`ip`/..., each resolving to a range an
//!   operator (or visual selection) can act on.
//! * [`register`] — named/unnamed/yank register storage, each remembering
//!   its [`crate::core::Granularity`] (the `dd`-then-`p`-pastes-a-line vs.
//!   `dw`-then-`p`-pastes-inline distinction).
//! * [`pending`] — the operator-pending grammar state machine. Read that
//!   module's docs first; it explains the one design decision everything
//!   else here depends on.
//! * [`ex`] — `:` command parsing and buffer-only execution (`:s`, `:g`,
//!   `:d`); effects requiring real I/O come back out through
//!   [`EditorResponse`] instead of happening inline.

pub mod cmdline;
pub mod command;
pub mod ex;
pub mod help;
pub mod key;
pub mod motion;
pub mod operator;
pub mod pending;
pub mod register;
pub mod search;
pub mod text_object;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use unicode_segmentation::UnicodeSegmentation;

use crate::core::{BufferId, Edit, Granularity, Mode, Position, Range, ViewportScroll, WindowCommand};
use crate::text::Buffer;

pub use key::{Key, KeyCode, Modifiers};
pub use pending::{GrammarCommand, InsertPos};

use motion::{FindKind, Motion};
use operator::Operator;
use cmdline::{CmdlineBuffer, History};
use pending::{FeedResult, Pending};
use register::Registers;
use text_object::ObjectScope;

/// Which shape the current visual selection has. A separate type from
/// [`Mode`] would be redundant — `Mode` already distinguishes
/// `Visual`/`VisualLine`/`VisualBlock` — but `pending::GrammarCommand::EnterVisual`
/// needs to name a *kind* to enter, before there is a `Mode` to read it back
/// from, so this exists as the small piece of vocabulary shared between "I
/// want to enter visual mode" and "which visual mode".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualKind {
    Charwise,
    Linewise,
    Blockwise,
}

/// Everything `Editor::handle_key` can ask its caller to do.
///
/// # Why this carries `crate::config::Action` instead of LSP/plugin types
///
/// `kopitiam-neovim`'s dependency graph is one-way: `editor` must not depend
/// on `lsp`, `plugins`, or `ui`, or this crate stops being testable
/// headlessly and stops being usable from a future TUI/GUI frontend that
/// wants a different LSP client. But a keymap like `<leader>gd` has to
/// *mean* "go to definition" somehow. The resolution is
/// [`crate::config::Action`] — pure data, already defined by `config` (a
/// module this crate already depends on for its own default keymaps) — so
/// `Editor` can look a key sequence up in the configured keymaps, find
/// `Action::LspDefinition`, and hand it back through
/// [`EditorResponse::Action`] without ever importing `kopitiam-lsp`. The
/// caller (ultimately `apps/cli` or a future TUI) is the one place that
/// knows how to turn `Action::LspDefinition` into a real LSP request.
///
/// The ex-command effects (`Write`, `Quit`, ...) follow the same shape for a
/// different reason: [`ex`]'s module docs explain why real I/O is kept out
/// of command execution.
#[derive(Debug, Clone, PartialEq)]
pub enum EditorResponse {
    /// The key was handled; nothing further is needed.
    Continue,
    /// `:q`/`:q!` with no unsaved-changes conflict (or `!` overriding one).
    Quit,
    /// `:qa`/`:qa!`: quit every window and exit the editor. Distinct from
    /// [`EditorResponse::Quit`], which closes the active window and only exits
    /// on the last one — quit-all exits unconditionally regardless of how many
    /// windows are open. The unsaved-changes guard has already run in
    /// [`Editor::execute_ex`], so reaching here means the exit is allowed.
    QuitAll,
    /// `:w`/`:w {file}`. The caller decides *how* to write — typically by
    /// calling [`Buffer::save`]/[`Buffer::save_as`] on
    /// [`Editor::buffer_mut`] — rather than this crate doing it inline; see
    /// [`ex`]'s module docs.
    Write { path: Option<PathBuf> },
    /// `:wq`/`:x`.
    WriteThenQuit { path: Option<PathBuf> },
    /// `:wa`/`:wall` (`then_quit == false`) and `:wqa`/`:xa` (`then_quit ==
    /// true`): write every modified buffer, then — for the quit-all forms —
    /// exit the editor. Like [`EditorResponse::Write`], the editor returns the
    /// intent and the caller performs the I/O across all buffers (see [`ex`]'s
    /// module docs on why writing is the caller's job).
    WriteAll { then_quit: bool },
    /// Feedback for the statusline/echo area (`:s` match counts, error
    /// text, ...).
    Message(String),
    /// A keymap resolved to a configured action. See this type's docs.
    Action(crate::config::Action),
    /// A window-management command the UI must carry out (`:sp`, `:vs`,
    /// `:only`, `:close`). See [`WindowCommand`] for why the editor cannot do
    /// this itself.
    Window(WindowCommand),
    /// A viewport reposition request (`zz`, `zt`, `zb`, `<C-e>`, `<C-y>`).
    /// See [`ViewportScroll`] for why this is a request, not an edit.
    Scroll(ViewportScroll),
}

/// Which of the three command-line prompts is currently open — `:` for ex
/// commands, `/`/`?` for searches. Kept in the editor (not the UI) because
/// *what* the typed text means is editor business; the UI only needs to know
/// which prefix character to draw, which it derives from this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKind {
    Ex,
    SearchForward,
    SearchBackward,
}

/// Text lines a window is assumed to show until the UI says otherwise.
///
/// Only affects Ctrl+D/U/F/B. A sane fallback matters because a headless
/// `Editor` (tests, scripts) has no window at all, and a zero here would make
/// a half-page scroll move zero lines — a silent no-op that looks like a bug in
/// the keymap rather than a missing viewport.
pub const DEFAULT_VIEWPORT_LINES: usize = 24;

/// The modal editing engine. One `Editor` owns every open buffer, the
/// current mode, registers, macros, and the operator-pending grammar state
/// — everything the brief asks for except rendering.
pub struct Editor {
    buffers: BTreeMap<BufferId, Buffer>,
    buffer_order: Vec<BufferId>,
    next_buffer_id: u32,
    current: BufferId,
    saved_cursor: HashMap<BufferId, Position>,

    mode: Mode,
    cursor: Position,

    visual_anchor: Position,
    visual_kind: VisualKind,
    // Visual mode's small amount of multi-key lookahead state. See
    // `handle_visual_key`'s docs for why visual mode does not reuse
    // `Pending` (its grammar genuinely differs: operators act immediately
    // on the selection rather than waiting for a motion).
    visual_g_pending: bool,
    visual_find_pending: Option<FindKind>,
    visual_object_pending: Option<ObjectScope>,

    pending: Pending,
    registers: Registers,

    macros: HashMap<char, Vec<Key>>,
    recording: Option<(char, Vec<Key>)>,
    last_played_macro: Option<char>,

    /// The keys of the change currently being composed, if any — see
    /// `commit_dot`/`discard_dot`.
    dot_recording: Option<Vec<Key>>,
    /// The last *completed change*'s keys, replayed verbatim by `.`.
    dot: Option<Vec<Key>>,
    /// Nonzero while replaying macro/dot-repeat keys, so that replayed
    /// keystrokes do not themselves get re-recorded into
    /// `recording`/`dot_recording` (which would duplicate or corrupt them —
    /// see `handle_key`'s docs).
    replaying: u32,

    last_find: Option<(FindKind, char)>,

    /// The editable command-line buffer (text + grapheme cursor + in-flight
    /// history/completion state). Meaningful only while `mode == Mode::Command`;
    /// entering command mode [`CmdlineBuffer::clear`]s it. See
    /// [`cmdline::CmdlineBuffer`] for why the prompt needs a real line editor
    /// and not the bare `String` this used to be.
    cmdline: CmdlineBuffer,
    /// Which prompt (`:`, `/`, `?`) the command line is serving — see
    /// [`CommandKind`]. Meaningful only while `mode == Mode::Command`.
    command_kind: CommandKind,
    /// vim keeps the `:` history and the `/`?` history apart; these are the two
    /// rings `<Up>`/`<Down>` walk, picked by `command_kind`. Session-scoped for
    /// now — cross-session persistence (vim's `viminfo`) is a filed follow-up.
    ex_history: History,
    search_history: History,
    /// Command-line `<C-r>{reg}` is a two-key sequence (`<C-r>` then a register
    /// name); this remembers we are between the two, mirroring
    /// [`Self::insert_register_pending`] for insert mode.
    command_register_pending: bool,
    /// The last `/`/`?`/`*`/`#` search, as `(pattern, forward)`, so `n`/`N`
    /// can repeat it. `n` reuses `forward`; `N` inverts it.
    last_search: Option<(String, bool)>,

    /// The jump history for `<C-o>`/`<C-i>` and `` `` `` — see the module
    /// docs on [`Self::record_jump`]. `jump_index == jumps.len()` means "at
    /// the present position" (nothing to redo forward into).
    jumps: Vec<(BufferId, Position)>,
    jump_index: usize,
    /// Where the most recent jump started, for the `` `` ``/`''` motion
    /// (jump back to the position before the last jump).
    last_jump_from: Option<(BufferId, Position)>,

    /// The last visual selection (`anchor`, `cursor`, kind), for `gv`.
    last_visual: Option<(Position, Position, VisualKind)>,

    /// Insert-mode `<C-r>` is a two-key sequence (`<C-r>` then a register
    /// name); this remembers we are between the two.
    insert_register_pending: bool,
    /// Insert-mode `<C-o>` runs exactly one Normal-mode command and then
    /// returns to Insert; this is set while that one command is in flight.
    insert_one_shot: bool,

    /// Text lines currently visible in the window, kept up to date by the UI
    /// via `set_viewport_height`. Only Ctrl+D/U/F/B need it — see that method.
    viewport_height: usize,

    /// Compiled from `Config::keymaps` (with `<leader>` substituted) — see
    /// `compile_keymaps`. Checked before the built-in grammar for any key
    /// sequence that could still become a configured mapping; see
    /// `handle_normal_key`'s docs on why keymaps take priority.
    keymaps: Vec<(Vec<Key>, crate::config::Action)>,
    /// The same compiled key sequences paired with their human-readable
    /// descriptions (`Keymap::desc`), used only to render the which-key popup
    /// — kept parallel to [`Self::keymaps`] rather than folded into it so that
    /// [`Self::match_keymap`]'s hot path stays a plain `(seq, action)` compare
    /// with no string it never reads.
    keymap_descs: Vec<(Vec<Key>, String)>,
    keymap_buffer: Vec<Key>,

    /// The last `:s` (substitute) as `(pattern, replacement, global)`, so `&`
    /// can repeat it on the current line. `None` until the first `:s` runs.
    last_substitution: Option<(String, String, bool)>,

    /// The "alternate file" (`#`): the buffer `<C-^>`/`<C-6>` toggles back to.
    /// Set to whatever buffer we just left whenever the active buffer changes
    /// (`:e`, `:b{n}`, `:bn`/`:bp`, window focus, `<C-^>` itself). `None` until
    /// a second buffer has ever been visited.
    alternate: Option<BufferId>,

    options: crate::config::Options,
}

impl Editor {
    /// A fresh editor with one empty buffer and the maintainer's default
    /// keymaps/options (see [`crate::config::Config::default`]).
    pub fn new() -> Self {
        Self::with_config(crate::config::Config::default())
    }

    /// Like [`Self::new`], but with a caller-supplied configuration (a
    /// loaded `~/.config/kvim/config.json`, or a fixture in tests).
    pub fn with_config(config: crate::config::Config) -> Self {
        let mut buffers = BTreeMap::new();
        let id = BufferId(0);
        buffers.insert(id, Buffer::new());
        let keymaps = compile_keymaps(&config);
        let keymap_descs = compile_keymap_descs(&config);
        Self {
            buffers,
            buffer_order: vec![id],
            next_buffer_id: 1,
            current: id,
            saved_cursor: HashMap::new(),
            mode: Mode::Normal,
            cursor: Position::ORIGIN,
            visual_anchor: Position::ORIGIN,
            visual_kind: VisualKind::Charwise,
            visual_g_pending: false,
            visual_find_pending: None,
            visual_object_pending: None,
            pending: Pending::new(),
            registers: Registers::new(),
            macros: HashMap::new(),
            recording: None,
            last_played_macro: None,
            dot_recording: None,
            dot: None,
            replaying: 0,
            last_find: None,
            cmdline: CmdlineBuffer::new(),
            command_kind: CommandKind::Ex,
            ex_history: History::new(),
            search_history: History::new(),
            command_register_pending: false,
            last_search: None,
            jumps: Vec::new(),
            jump_index: 0,
            last_jump_from: None,
            last_visual: None,
            insert_register_pending: false,
            insert_one_shot: false,
            viewport_height: DEFAULT_VIEWPORT_LINES,
            keymaps,
            keymap_descs,
            keymap_buffer: Vec::new(),
            last_substitution: None,
            alternate: None,
            options: config.options,
        }
    }

    /// Opens `path` as a new buffer, switches to it, and returns its id.
    /// Real I/O (via [`Buffer::from_file`]) — this is the crate's one
    /// sanctioned entry point for it on the read side; `:e {file}` in ex
    /// commands is implemented in terms of this same method, not a
    /// duplicate code path.
    pub fn open(&mut self, path: &Path) -> crate::Result<BufferId> {
        let buf = Buffer::from_file(path)?;
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.buffers.insert(id, buf);
        self.buffer_order.push(id);
        self.saved_cursor.insert(self.current, self.cursor);
        self.alternate = Some(self.current);
        self.current = id;
        self.cursor = Position::ORIGIN;
        Ok(id)
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The which-key rows to show for the key sequence buffered so far, or an
    /// empty vector when nothing is pending.
    ///
    /// This is the passive-UI half of which-key: it never changes what a key
    /// *does* (the keymap engine still resolves the full sequence), it only
    /// reports "given you have typed `<leader>`, here is what each next key
    /// leads to." Empty whenever [`Self::keymap_buffer`] is empty — i.e. the
    /// popup shows exactly while a multi-key mapping is mid-flight.
    pub fn which_key(&self) -> Vec<WhichKeyItem> {
        which_key_for(&self.keymap_buffer, &self.keymap_descs)
    }

    pub fn cursor(&self) -> Position {
        self.cursor
    }

    pub fn buffer(&self) -> &Buffer {
        self.current_buffer()
    }

    pub fn buffer_mut(&mut self) -> &mut Buffer {
        self.current_buffer_mut()
    }

    pub fn buffer_id(&self) -> BufferId {
        self.current
    }

    /// Read-only access to any open buffer by id — the seam that lets the UI
    /// render an *inactive* split showing a different buffer than the active
    /// one. Without it, `App::render_windows` had no way to ask for anything
    /// but the current buffer, so every split painted the same text (bug
    /// `kopitiam-cj0.10.3`).
    pub fn buffer_by_id(&self, id: BufferId) -> Option<&Buffer> {
        self.buffers.get(&id)
    }

    /// Whether *any* open buffer has unsaved changes — the widened form of
    /// [`Buffer::is_modified`] that `:qa` needs. `:q` asks only about the
    /// current buffer; quit-all must not discard an unsaved buffer sitting in
    /// another window, so it asks about all of them.
    pub fn any_buffer_modified(&self) -> bool {
        self.buffers.values().any(Buffer::is_modified)
    }

    /// Mutable access to every open buffer, in id order — the seam the UI's
    /// write-all (`:wa`/`:wqa`) uses to save each modified buffer. Like
    /// [`Editor::buffer_mut`], the editor exposes the buffers and lets the
    /// caller perform the actual disk I/O (see [`ex`]'s module docs on why
    /// writing is not done inside the editor).
    pub fn buffers_mut(&mut self) -> impl Iterator<Item = &mut Buffer> {
        self.buffers.values_mut()
    }

    /// Makes `buffer` the active buffer and moves the cursor to `cursor`
    /// (clamped) — the primitive the UI calls when window focus moves to a
    /// split showing a different buffer/position.
    ///
    /// # Why the editor keeps a single cursor even with splits
    ///
    /// Per-window cursor and buffer state lives in the UI's window tree (a
    /// window *is* a viewport — see `ui::window`), not here: the editor edits
    /// exactly one buffer with one cursor at a time, and the UI hands it the
    /// right one as focus moves. This keeps the whole modal engine unaware of
    /// windows, layout, and geometry — none of which belong below the UI
    /// line — while still giving each split its own independent view. The
    /// alternative (the editor owning N cursors) would drag window identity
    /// into every motion and operator for no gain.
    ///
    /// A no-op if `buffer` is not open, so a stale [`crate::core::WindowId`]
    /// mapping can never point the editor at a buffer that does not exist.
    pub fn set_active(&mut self, buffer: BufferId, cursor: Position) {
        if !self.buffers.contains_key(&buffer) {
            return;
        }
        if buffer != self.current {
            self.alternate = Some(self.current);
        }
        self.saved_cursor.insert(self.current, self.cursor);
        self.current = buffer;
        self.cursor = self.current_buffer().clamp(cursor);
        // Leaving whatever transient state the previous window was mid-typing
        // behind would let a half-typed operator "leak" into the newly
        // focused window; reset to a clean Normal-mode slate.
        self.mode = Mode::Normal;
        self.pending.reset();
    }

    /// Replaces the text in `range` with `text`, moves the cursor to the end of
    /// the inserted text, and returns that position.
    ///
    /// Backs the UI's insert-mode completion accept and snippet expansion (see
    /// [`crate::ui::event::EditorHost::replace_range`]): both replace the typed
    /// prefix with new text. It goes through [`Buffer::apply`], so the edit is
    /// recorded on the undo stack and marks are fixed up like any other change —
    /// accepting a completion is a normal, undoable edit, not a side channel. A
    /// malformed range (out of bounds) leaves the buffer untouched and returns
    /// the current cursor rather than panicking; the caller computed the range
    /// from live buffer state, so this is belt-and-braces.
    pub fn replace_range(&mut self, range: Range, text: &str) -> Position {
        match self.current_buffer_mut().apply(Edit::replace(range, text.to_string())) {
            Ok(landed) => {
                self.cursor = landed;
                landed
            }
            Err(_) => self.cursor,
        }
    }

    /// Moves the cursor to `pos`, clamped to the buffer — used to jump between
    /// snippet tabstops after an expansion. Text is untouched.
    pub fn move_cursor(&mut self, pos: Position) {
        self.cursor = self.current_buffer().clamp(pos);
    }

    /// Creates a fresh empty scratch buffer, makes it active, and returns its
    /// id — backs `:new`/`:vnew` and `<C-w>n`.
    pub fn new_buffer(&mut self) -> BufferId {
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.buffers.insert(id, Buffer::new());
        self.buffer_order.push(id);
        self.saved_cursor.insert(self.current, self.cursor);
        self.alternate = Some(self.current);
        self.current = id;
        self.cursor = Position::ORIGIN;
        id
    }

    /// `:bd`/`:bw`: chuck away the current buffer and land on a surviving one,
    /// returning `(deleted, replacement)` so the UI can repoint any window that
    /// was showing the deleted buffer (see
    /// [`WindowCommand::BufferDeleted`]).
    ///
    /// The rules follow vim:
    ///
    /// * **Unsaved guard.** Without `force`, a modified buffer refuses with
    ///   [`crate::Error::UnsavedChanges`] — the same guard `:q` uses — so you
    ///   cannot lose changes by a stray `:bd`. `:bd!` (`force == true`) deletes
    ///   anyway and discards the changes.
    /// * **Never zero buffers.** Deleting the *only* buffer would leave the
    ///   editor with none, which every accessor here assumes cannot happen
    ///   (`current` is always a live id). vim solves this by opening a fresh
    ///   empty buffer in the deleted one's place; so do we.
    /// * **Land on the alternate.** With more than one buffer open, we switch
    ///   to the next buffer in order, or the previous one if we were on the
    ///   last — the buffer vim leaves you on after a `:bd`.
    ///
    /// `wipe` (`:bw` vs `:bd`) makes no behavioural difference today: kvim does
    /// not yet track vim's unlisted/hidden-buffer state, so both forms remove
    /// the buffer outright. See [`ex::ExCommand::DeleteBuffer`] for why the flag
    /// is carried anyway.
    pub fn delete_buffer(&mut self, force: bool, _wipe: bool) -> crate::Result<(BufferId, BufferId)> {
        if !force && self.current_buffer().is_modified() {
            return Err(crate::Error::UnsavedChanges);
        }
        let deleted = self.current;
        let idx = self
            .buffer_order
            .iter()
            .position(|&id| id == deleted)
            .expect("current buffer id is always present in buffer_order");

        let replacement = if self.buffer_order.len() > 1 {
            let alt_idx = if idx + 1 < self.buffer_order.len() { idx + 1 } else { idx - 1 };
            self.buffer_order[alt_idx]
        } else {
            let new_id = BufferId(self.next_buffer_id);
            self.next_buffer_id += 1;
            self.buffers.insert(new_id, Buffer::new());
            self.buffer_order.push(new_id);
            new_id
        };

        self.buffers.remove(&deleted);
        self.buffer_order.retain(|&id| id != deleted);
        self.saved_cursor.remove(&deleted);
        self.current = replacement;
        let landed = *self.saved_cursor.get(&replacement).unwrap_or(&Position::ORIGIN);
        self.cursor = self.current_buffer().clamp(landed);
        Ok((deleted, replacement))
    }

    /// `:ls`/`:buffers`: one line per open buffer, in buffer order, echoing
    /// vim's layout closely enough to be familiar — the id, a `%a` marker on
    /// the active buffer, a `+` when the buffer got unsaved changes, and the
    /// file name (`[No Name]` for a scratch buffer with no path).
    pub fn buffer_list(&self) -> String {
        self.buffer_order
            .iter()
            .filter_map(|&id| self.buffers.get(&id).map(|buf| (id, buf)))
            .map(|(id, buf)| {
                let active = if id == self.current { "%a" } else { "  " };
                let modified = if buf.is_modified() { "+" } else { " " };
                let name = buf.path().map(|p| p.display().to_string()).unwrap_or_else(|| "[No Name]".to_string());
                format!("{:>3} {active} {modified} {name}", id.0)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Which prompt the command line is serving, or `None` when not in
    /// [`Mode::Command`]. Lets the UI draw `:` vs `/` vs `?`.
    pub fn command_line_kind(&self) -> Option<CommandKind> {
        (self.mode == Mode::Command).then_some(self.command_kind)
    }

    /// The text typed so far on the command line, or `None` when not in
    /// [`Mode::Command`].
    ///
    /// # Why this exists
    ///
    /// It did not, and the consequence was that **`:` commands were invisible
    /// while you typed them.** The editor accumulated `Neotree` in
    /// `self.command_line` perfectly well; the UI simply had no way to ask for
    /// it, so it rendered an empty prompt. The whole ex-command layer was
    /// unusable, and 305 passing tests said nothing about it — because every
    /// one of them checked editor *state*, and none checked what was on screen.
    ///
    /// A textbook seam bug: the editor half was right, the renderer half was
    /// right, and nothing joined them.
    pub fn command_line(&self) -> Option<&str> {
        (self.mode == Mode::Command).then_some(self.cmdline.text())
    }

    /// Where the caret sits within the command line, as a grapheme offset, or
    /// `None` when not in [`Mode::Command`].
    ///
    /// This is the other half of making the command line a real line editor:
    /// the text alone was enough while the cursor could only ever be at the end
    /// (append-only typing), but now that `<Left>`/`<C-w>`/`<Home>` move it, the
    /// renderer has to be told where it actually is rather than assuming "the
    /// end". Grapheme units, to match [`crate::core::Position`] and the
    /// renderer's own [`crate::ui::cmdline::CmdlineState::cursor`].
    pub fn command_cursor(&self) -> Option<usize> {
        (self.mode == Mode::Command).then(|| self.cmdline.cursor())
    }

    /// The `<Tab>` completion candidates currently being cycled and which one is
    /// selected, for a wildmenu-style display, or `None` when no cycle is open.
    pub fn command_completions(&self) -> Option<(&[String], usize)> {
        if self.mode != Mode::Command {
            return None;
        }
        self.cmdline.active_completions()
    }

    /// The current visual selection as `(start, end)` in document order, or
    /// `None` when not in a visual mode.
    ///
    /// Same story as [`Self::command_line`]: the editor tracked
    /// `visual_anchor` correctly the whole time, but the UI could not see it,
    /// so **visual mode selected text without highlighting any of it.**
    ///
    /// # The three visual modes select genuinely different things
    ///
    /// This returns the raw anchor/cursor pair, normalised into document
    /// order. It deliberately does **not** try to expand that pair into "the
    /// cells that are selected" — that expansion depends on the mode and is a
    /// *rendering* question, so it belongs to the renderer:
    ///
    /// * [`Mode::Visual`] — charwise. The span runs from `start` to `end`,
    ///   partial on the first and last lines.
    /// * [`Mode::VisualLine`] — linewise. **Whole lines**, columns ignored
    ///   entirely. A renderer that highlights only `start.col..end.col` here is
    ///   wrong, and it is the classic mistake.
    /// * [`Mode::VisualBlock`] — blockwise. A *rectangle*: the column range
    ///   `start.col..=end.col` on every line from `start.line` to `end.line`.
    ///
    /// Use [`Self::mode`] alongside this to decide which of the three you are
    /// painting.
    pub fn selection(&self) -> Option<(Position, Position)> {
        if !self.mode.is_visual() {
            return None;
        }
        let (a, b) = (self.visual_anchor, self.cursor);
        Some(if a <= b { (a, b) } else { (b, a) })
    }

    /// Tells the editor how many text lines the window currently shows.
    ///
    /// The UI must call this on startup and on every terminal resize.
    ///
    /// # Why the editor needs to know
    ///
    /// `Ctrl+D` scrolls down by **half a screen**, and `Ctrl+F` by a **full**
    /// one. "Screen" is a property of the *window*, not of the text — so these
    /// are the one place the editor genuinely cannot compute its own answer.
    /// Without this the editor would have to invent a number, and the cursor
    /// would jump by an amount that had nothing to do with what the user could
    /// see.
    ///
    /// Defaults to [`DEFAULT_VIEWPORT_LINES`] so that a headless `Editor` (in
    /// tests, or driving a script) still behaves sensibly rather than dividing
    /// by zero.
    pub fn set_viewport_height(&mut self, lines: usize) {
        self.viewport_height = lines.max(1);
    }

    /// Moves the cursor `lines` down (positive) or up (negative), clamped to
    /// the buffer — the motion behind `Ctrl+D`/`Ctrl+U`/`Ctrl+F`/`Ctrl+B`.
    ///
    /// Vim keeps the cursor's column across a scroll, so this does too, and
    /// clamps it to the destination line's length rather than pushing it past
    /// the end.
    fn scroll_lines(&mut self, lines: isize) -> EditorResponse {
        let line_count = self.buffer().line_count();
        let target = (self.cursor.line as isize + lines).clamp(0, line_count.saturating_sub(1) as isize) as usize;

        self.cursor = self.buffer().clamp(Position::new(target, self.cursor.col));
        EditorResponse::Continue
    }

    fn current_buffer(&self) -> &Buffer {
        self.buffers.get(&self.current).expect("current buffer id is always valid")
    }

    fn current_buffer_mut(&mut self) -> &mut Buffer {
        self.buffers.get_mut(&self.current).expect("current buffer id is always valid")
    }

    /// Feeds one key. This is the crate's single entry point — see the
    /// module docs.
    ///
    /// # Recording bookkeeping
    ///
    /// Macro recording ([`Self::recording`]) and dot-repeat recording
    /// ([`Self::dot_recording`]) both work by capturing raw keys rather than
    /// a semantic replay of "the command that ran" — see [`pending`]'s docs
    /// for why [`Pending`] itself stays buffer-free, and
    /// [`pending::GrammarCommand`] for the structured form used to *execute*
    /// a command once. Raw-key capture means both mechanisms have to be
    /// careful not to record keys that are *replaying* an earlier recording
    /// (macro-of-a-macro, or `.` re-running `.`'s own keys) — that is what
    /// [`Self::replaying`] guards against: it is incremented for the
    /// duration of any programmatic replay, and both recorders check it
    /// before appending.
    pub fn handle_key(&mut self, key: Key) -> crate::Result<EditorResponse> {
        if self.replaying == 0 {
            if key.code == KeyCode::Char('q') && !key.mods.ctrl && self.recording.is_some() && self.mode == Mode::Normal && self.pending.is_idle() {
                let (reg, keys) = self.recording.take().expect("checked is_some above");
                self.macros.insert(reg, keys);
                return Ok(EditorResponse::Continue);
            }
            if let Some((_, keys)) = self.recording.as_mut() {
                keys.push(key);
            }
            if self.mode == Mode::Normal && self.pending.is_idle() && self.dot_recording.is_none() {
                self.dot_recording = Some(vec![key]);
            } else if let Some(keys) = self.dot_recording.as_mut() {
                keys.push(key);
            }
        }

        let mode_before = self.mode;
        let result = match self.mode {
            Mode::Insert => self.handle_insert_key(key),
            Mode::Replace => self.handle_replace_key(key),
            Mode::Command => self.handle_command_key(key),
            Mode::Visual | Mode::VisualLine | Mode::VisualBlock => self.handle_visual_key(key),
            Mode::Normal | Mode::OperatorPending => self.handle_normal_key(key),
        };

        // Insert-mode `<C-o>` semantics: the keystroke that armed one-shot ran
        // as an Insert key (so `mode_before == Insert` — skip); the *next*
        // keystroke runs in Normal mode, and once it completes (pending idle,
        // still Normal), we drop back into Insert. If that one command itself
        // switched to a non-Normal mode (e.g. the user pressed `a`), we honour
        // that and simply clear the flag.
        if self.insert_one_shot && mode_before == Mode::Normal {
            if self.mode == Mode::Normal && self.pending.is_idle() {
                self.insert_one_shot = false;
                self.mode = Mode::Insert;
            } else if self.mode != Mode::Normal && self.mode != Mode::OperatorPending {
                self.insert_one_shot = false;
            }
        }

        result
    }

    fn commit_dot(&mut self) {
        if let Some(keys) = self.dot_recording.take() {
            self.dot = Some(keys);
        }
    }

    fn discard_dot(&mut self) {
        self.dot_recording = None;
    }

    // ---------------------------------------------------------------
    // Normal / operator-pending mode
    // ---------------------------------------------------------------

    /// Dispatches a key while in Normal or OperatorPending mode.
    ///
    /// # Why keymaps are checked before the vi grammar
    ///
    /// The maintainer's config remaps bare `f` (ordinarily "find character
    /// on line") to a hop-to-word plugin in *every* mode — see
    /// `config::default_keymaps`'s comment on that binding. That means a
    /// configured keymap can legitimately shadow a built-in single-key
    /// motion, so keymap resolution has to run first. It only runs when
    /// `pending` is idle, though: once a command like `d` or a count is
    /// already in flight, keys belong to *that* command's grammar (so
    /// `df<x>` still finds `<x>` on the line, rather than the shadowed `f`
    /// hijacking `d`'s motion) — see this crate's report for the trade-off
    /// this implies.
    fn handle_normal_key(&mut self, key: Key) -> crate::Result<EditorResponse> {
        // Scrolling commands, handled before the vi grammar sees them.
        //
        // These are NOT motions and must not be: `d<C-d>` is not a thing, and
        // routing them through the operator-pending machinery would make it
        // one. They are whole commands that move the cursor by a screenful,
        // which is why they need `viewport_height` — see `set_viewport_height`.
        //
        // They were simply absent before, so Ctrl+D and Ctrl+U did nothing at
        // all. The maintainer found that by using the editor; no unit test did,
        // because you cannot notice a missing keybinding by testing the
        // bindings that exist.
        // Handled UNCONDITIONALLY, not just when idle -- and that matters more
        // than it looks. The pending-command grammar matches on `KeyCode` and
        // **ignores the ctrl modifier entirely**, so before this guard existed
        // `<C-d>` was indistinguishable from a bare `d`. Consequences:
        //
        //   * plain `Ctrl+D` in normal mode silently started an operator-pending
        //     DELETE, which is precisely the "Ctrl+D doesn't behave right" the
        //     maintainer reported; and
        //   * `d<C-d>` was read as `dd` and deleted a line.
        //
        // Catching these here, ahead of the grammar, is what makes a
        // ctrl-modified key a different key from its unmodified twin.
        if key.mods.ctrl {
            // Ctrl-modified whole commands, caught ahead of the vi grammar for
            // the same reason `<C-d>` is (see below): `Pending` matches on
            // `KeyCode` and ignores the ctrl bit, so without this guard every
            // one of these would be mistaken for its unmodified twin —
            // `<C-a>` would start Insert (`a`), `<C-x>` would delete a
            // character, `<C-e>`/`<C-y>` would start a yank/word-motion.
            //
            // `<C-r>` (redo) and `<C-v>` (visual-block) are deliberately NOT
            // here: `Pending` guards those on the ctrl bit itself, so they are
            // safe to reach the grammar.
            let half = (self.viewport_height / 2).max(1) as isize;
            let full = self.viewport_height.max(1) as isize;
            enum CtrlCmd {
                Scroll(isize),
                Increment(i64),
                JumpBack,
                View(ViewportScroll),
                /// `<C-g>`: echo the file name, line count and cursor position.
                FileInfo,
                /// `<C-^>`/`<C-6>`: switch to the alternate (`#`) buffer.
                AlternateFile,
                /// `<C-]>`: jump to the definition of the symbol under the
                /// cursor. kvim has no ctags, so this routes to the LSP
                /// go-to-definition the editor already provides (the `gd`
                /// path), not a tag stack.
                GotoDefinition,
            }
            let cmd = match key.code {
                KeyCode::Char('d') => Some(CtrlCmd::Scroll(half)),
                KeyCode::Char('u') => Some(CtrlCmd::Scroll(-half)),
                KeyCode::Char('f') => Some(CtrlCmd::Scroll(full)),
                KeyCode::Char('b') => Some(CtrlCmd::Scroll(-full)),
                KeyCode::Char('a') => Some(CtrlCmd::Increment(1)),
                KeyCode::Char('x') => Some(CtrlCmd::Increment(-1)),
                KeyCode::Char('o') => Some(CtrlCmd::JumpBack),
                KeyCode::Char('e') => Some(CtrlCmd::View(ViewportScroll::LineDown)),
                KeyCode::Char('y') => Some(CtrlCmd::View(ViewportScroll::LineUp)),
                KeyCode::Char('g') => Some(CtrlCmd::FileInfo),
                // In a terminal Ctrl+^ and Ctrl+6 send the same byte; accept
                // both so either keyboard reflex reaches the alternate file.
                KeyCode::Char('^') | KeyCode::Char('6') => Some(CtrlCmd::AlternateFile),
                KeyCode::Char(']') => Some(CtrlCmd::GotoDefinition),
                _ => None,
            };
            if let Some(cmd) = cmd {
                // Abandon any half-typed command. None of these is a motion,
                // so none can complete a pending operator -- and leaving `d`
                // armed would make the NEXT keystroke delete something the
                // user never asked it to.
                if !self.mode.is_visual() {
                    self.pending.reset();
                    self.mode = Mode::Normal;
                }
                return Ok(match cmd {
                    CtrlCmd::Scroll(lines) => {
                        self.discard_dot();
                        self.scroll_lines(lines)
                    }
                    // Increment mutates the buffer, so it commits a dot-repeat.
                    CtrlCmd::Increment(delta) => self.increment_number(delta),
                    CtrlCmd::JumpBack => {
                        self.discard_dot();
                        self.jump_back();
                        EditorResponse::Continue
                    }
                    CtrlCmd::View(v) => {
                        self.discard_dot();
                        EditorResponse::Scroll(v)
                    }
                    CtrlCmd::FileInfo => {
                        self.discard_dot();
                        EditorResponse::Message(self.file_info())
                    }
                    CtrlCmd::AlternateFile => {
                        self.discard_dot();
                        self.edit_alternate()
                    }
                    CtrlCmd::GotoDefinition => {
                        self.discard_dot();
                        EditorResponse::Action(crate::config::Action::LspDefinition)
                    }
                });
            }
        }

        // <Tab> (== <C-i> in a terminal) jumps forward in the jumplist;
        // PageDown/PageUp scroll a full screen, matching `<C-f>`/`<C-b>`.
        if self.pending.is_idle() && !self.mode.is_visual() {
            match key.code {
                KeyCode::Tab => {
                    self.discard_dot();
                    self.jump_forward();
                    self.mode = Mode::Normal;
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::PageDown | KeyCode::PageUp => {
                    self.discard_dot();
                    let full = self.viewport_height.max(1) as isize;
                    let lines = if key.code == KeyCode::PageDown { full } else { -full };
                    self.mode = Mode::Normal;
                    return Ok(self.scroll_lines(lines));
                }
                _ => {}
            }
        }

        if self.pending.is_idle() {
            match self.match_keymap(key) {
                KeymapDispatch::Action(action) => {
                    self.discard_dot();
                    return Ok(EditorResponse::Action(action));
                }
                KeymapDispatch::Buffered => return Ok(EditorResponse::Continue),
                KeymapDispatch::Replay(buffered) => {
                    for k in buffered {
                        if let FeedResult::Complete(cmd) = self.pending.feed(k) {
                            return self.execute_grammar(cmd);
                        }
                    }
                }
                KeymapDispatch::None => {}
            }
        }

        match self.pending.feed(key) {
            FeedResult::Continue => {
                self.mode = if self.pending.is_idle() { Mode::Normal } else { Mode::OperatorPending };
                Ok(EditorResponse::Continue)
            }
            FeedResult::Invalid => {
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Continue)
            }
            FeedResult::Complete(cmd) => self.execute_grammar(cmd),
        }
    }

    /// Checks `key` against the compiled keymap table, given whatever
    /// partial sequence is already buffered. See [`KeymapDispatch`].
    fn match_keymap(&mut self, key: Key) -> KeymapDispatch {
        if self.keymaps.is_empty() {
            return KeymapDispatch::None;
        }
        let mut candidate = self.keymap_buffer.clone();
        candidate.push(normalize_for_keymap(key));

        if let Some((_, action)) = self.keymaps.iter().find(|(seq, _)| *seq == candidate) {
            self.keymap_buffer.clear();
            return KeymapDispatch::Action(action.clone());
        }
        if self.keymaps.iter().any(|(seq, _)| seq.len() > candidate.len() && seq.starts_with(&candidate)) {
            self.keymap_buffer = candidate;
            return KeymapDispatch::Buffered;
        }
        KeymapDispatch::Replay(std::mem::take(&mut self.keymap_buffer))
    }

    /// Runs a fully-parsed [`GrammarCommand`], mutating the buffer/mode/
    /// cursor as needed.
    fn execute_grammar(&mut self, cmd: GrammarCommand) -> crate::Result<EditorResponse> {
        match cmd {
            GrammarCommand::Move { count, motion } => {
                // `gg`/`G` are jumps: record where we left so `<C-o>` returns.
                if matches!(motion, Motion::FileStart | Motion::FileEnd) {
                    self.record_jump();
                }
                self.cursor = motion.apply(self.current_buffer(), self.cursor, count);
                self.remember_find(motion);
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::OperatorMotion { register, count, operator, motion } => {
                let motion = adjust_change_word_motion(operator, motion, self.current_buffer(), self.cursor);
                let end = motion.apply(self.current_buffer(), self.cursor, count);
                self.remember_find(motion);
                let (range, granularity) = operator::charwise_range(self.current_buffer(), self.cursor, end, motion.kind());
                self.run_operator(operator, range, granularity, register)
            }
            GrammarCommand::OperatorTextObject { register, operator, scope, object } => {
                match text_object::resolve(self.current_buffer(), self.cursor, object, scope) {
                    Some((range, granularity)) => self.run_operator(operator, range, granularity, register),
                    None => {
                        self.discard_dot();
                        self.mode = Mode::Normal;
                        Ok(EditorResponse::Continue)
                    }
                }
            }
            GrammarCommand::OperatorLines { register, count, operator } => {
                let n = count.unwrap_or(1).max(1);
                let last = (self.cursor.line + n - 1).min(self.current_buffer().line_count() - 1);
                let range = operator::linewise_content_range(self.current_buffer(), self.cursor.line, last);
                self.run_operator(operator, range, Granularity::Linewise, register)
            }
            GrammarCommand::RepeatFind { register, count, operator, reverse } => {
                let Some((kind, target)) = self.last_find else {
                    self.discard_dot();
                    self.mode = Mode::Normal;
                    return Ok(EditorResponse::Continue);
                };
                let motion = motion::repeat_find(kind, target, reverse);
                match operator {
                    Some(operator) => {
                        let end = motion.apply(self.current_buffer(), self.cursor, count);
                        let (range, granularity) = operator::charwise_range(self.current_buffer(), self.cursor, end, motion.kind());
                        self.run_operator(operator, range, granularity, register)
                    }
                    None => {
                        self.cursor = motion.apply(self.current_buffer(), self.cursor, count);
                        self.discard_dot();
                        self.mode = Mode::Normal;
                        Ok(EditorResponse::Continue)
                    }
                }
            }
            GrammarCommand::DeleteCharForward { register, count } => self.delete_chars(register, count, true),
            GrammarCommand::DeleteCharBackward { register, count } => self.delete_chars(register, count, false),
            GrammarCommand::SubstituteChar { register, count } => {
                let n = count.unwrap_or(1).max(1);
                let buf = self.current_buffer();
                let end = (self.cursor.col + n).min(buf.line_len(self.cursor.line));
                let range = Range::new(self.cursor, Position::new(self.cursor.line, end));
                let text = self.current_buffer().slice(range);
                self.begin_insert_group();
                let cursor = self.current_buffer_mut().apply(Edit::delete(range))?;
                self.registers.write_delete(register, text, Granularity::Charwise);
                self.cursor = cursor;
                self.mode = Mode::Insert;
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::ReplaceChar { count, ch } => self.replace_char(count, ch),
            GrammarCommand::ToggleCaseUnderCursor { count } => self.toggle_case_under_cursor(count),
            GrammarCommand::JoinLines { count } => self.join_lines(count),
            GrammarCommand::Put { register, count, before } => self.put(register, count.unwrap_or(1).max(1), before),
            GrammarCommand::EnterInsert(pos) => self.enter_insert_at(pos),
            GrammarCommand::Undo => {
                self.cursor = self.current_buffer_mut().undo()?;
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::Redo => {
                self.cursor = self.current_buffer_mut().redo()?;
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::RepeatLast => {
                self.discard_dot();
                self.mode = Mode::Normal;
                if let Some(keys) = self.dot.clone() {
                    self.replaying += 1;
                    let mut result = Ok(EditorResponse::Continue);
                    for k in keys {
                        result = self.handle_key(k);
                        if result.is_err() {
                            break;
                        }
                    }
                    self.replaying -= 1;
                    return result;
                }
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::EnterVisual(kind) => {
                self.visual_anchor = self.cursor;
                self.visual_kind = kind;
                self.mode = match kind {
                    VisualKind::Charwise => Mode::Visual,
                    VisualKind::Linewise => Mode::VisualLine,
                    VisualKind::Blockwise => Mode::VisualBlock,
                };
                self.discard_dot();
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::EnterCommandLine => {
                self.mode = Mode::Command;
                self.command_kind = CommandKind::Ex;
                self.cmdline.clear();
                self.command_register_pending = false;
                self.discard_dot();
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::StartRecording { register } => {
                self.recording = Some((register, Vec::new()));
                self.discard_dot();
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::PlayMacro { register, count } => {
                self.last_played_macro = Some(register);
                self.play_keys(self.macros.get(&register).cloned().unwrap_or_default(), count.unwrap_or(1))
            }
            GrammarCommand::ReplayLastMacro { count } => match self.last_played_macro {
                Some(register) => self.play_keys(self.macros.get(&register).cloned().unwrap_or_default(), count.unwrap_or(1)),
                None => {
                    self.discard_dot();
                    Ok(EditorResponse::Continue)
                }
            },
            GrammarCommand::SetMark { name } => {
                let at = self.cursor;
                self.current_buffer_mut().set_mark(name, at);
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::JumpMark { name, exact } => {
                self.discard_dot();
                self.mode = Mode::Normal;
                self.jump_to_mark(name, exact);
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::StartSearch { forward } => {
                self.mode = Mode::Command;
                self.command_kind = if forward { CommandKind::SearchForward } else { CommandKind::SearchBackward };
                self.cmdline.clear();
                self.command_register_pending = false;
                self.discard_dot();
                Ok(EditorResponse::Continue)
            }
            GrammarCommand::RepeatSearch { reverse } => {
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(self.repeat_search(reverse))
            }
            GrammarCommand::SearchWord { forward } => {
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(self.search_word_under_cursor(forward))
            }
            GrammarCommand::Scroll(req) => {
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Scroll(req))
            }
            GrammarCommand::ReselectVisual => {
                self.discard_dot();
                if let Some((anchor, cursor, kind)) = self.last_visual {
                    self.visual_anchor = self.current_buffer().clamp(anchor);
                    self.cursor = self.current_buffer().clamp(cursor);
                    self.visual_kind = kind;
                    self.mode = match kind {
                        VisualKind::Charwise => Mode::Visual,
                        VisualKind::Linewise => Mode::VisualLine,
                        VisualKind::Blockwise => Mode::VisualBlock,
                    };
                }
                Ok(EditorResponse::Continue)
            }
            // `ZZ` = `:x` (write if modified, then quit). The caller performs
            // the disk I/O, same as every other write path — see
            // `EditorResponse::WriteThenQuit`.
            GrammarCommand::WriteQuit => {
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::WriteThenQuit { path: None })
            }
            // `ZQ` = `:q!`: quit unconditionally, discarding changes. No
            // unsaved-changes guard, unlike a plain `:q`.
            GrammarCommand::QuitForce => {
                self.discard_dot();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Quit)
            }
            GrammarCommand::RepeatSubstitute => {
                self.mode = Mode::Normal;
                self.repeat_substitution()
            }
            GrammarCommand::JumpBracketMark { forward, exact } => {
                self.discard_dot();
                self.mode = Mode::Normal;
                self.jump_bracket_mark(forward, exact);
                Ok(EditorResponse::Continue)
            }
        }
    }

    /// `&`: re-run the last `:s` on the current line, dropping its flags (so
    /// only the first match on the line is replaced, matching vim). A no-op
    /// with a friendly note when no substitution has been run yet — the same
    /// thing vim's `E33: No previous substitute regular expression` guards
    /// against, phrased as a statusline message rather than an error.
    fn repeat_substitution(&mut self) -> crate::Result<EditorResponse> {
        self.discard_dot();
        let Some((pattern, replacement, _)) = self.last_substitution.clone() else {
            return Ok(EditorResponse::Message("no previous substitute".to_string()));
        };
        let line = self.cursor.line;
        let n = ex::substitute(self.current_buffer_mut(), line, line, &pattern, &replacement, false)?;
        self.cursor = self.current_buffer().clamp(self.cursor);
        Ok(EditorResponse::Message(format!("{n} substitution(s)")))
    }

    /// `['`/`` [` ``/`]'`/`` ]` ``: jump to the previous/next lowercase mark by
    /// buffer position. `exact` lands on the mark's column (back-tick forms);
    /// otherwise on the first non-blank of the mark's line (apostrophe forms).
    /// Records a jump so `<C-o>` returns here. A no-op when there is no mark in
    /// the requested direction.
    fn jump_bracket_mark(&mut self, forward: bool, exact: bool) {
        let cursor = self.cursor;
        let mut marks = self.current_buffer().lowercase_marks();
        marks.sort_by_key(|&(_, pos)| pos);
        let target = if forward {
            marks.iter().map(|&(_, p)| p).find(|&p| p > cursor)
        } else {
            marks.iter().map(|&(_, p)| p).rev().find(|&p| p < cursor)
        };
        let Some(pos) = target else { return };
        self.record_jump();
        self.cursor = if exact {
            self.current_buffer().clamp(pos)
        } else {
            Position::new(pos.line, operator::first_non_blank_col(self.current_buffer(), pos.line))
        };
    }

    // ---------------------------------------------------------------
    // Jumps, marks, search, increment
    // ---------------------------------------------------------------

    /// Records the current position as the origin of a jump, for `<C-o>` and
    /// `` `` ``. Called *before* a jump command (search, mark jump, `gg`/`G`)
    /// moves the cursor. Any forward history (positions reachable via
    /// `<C-i>`) is dropped, matching vim: making a new jump abandons the redo
    /// branch of the jumplist.
    fn record_jump(&mut self) {
        self.last_jump_from = Some((self.current, self.cursor));
        self.jumps.truncate(self.jump_index);
        self.jumps.push((self.current, self.cursor));
        self.jump_index = self.jumps.len();
    }

    /// `<C-o>`: move to an older position in the jumplist.
    fn jump_back(&mut self) {
        if self.jump_index == 0 {
            return;
        }
        // On the first step back from the present, append the present
        // position so `<C-i>` can return to it.
        if self.jump_index == self.jumps.len() {
            self.jumps.push((self.current, self.cursor));
        }
        self.jump_index -= 1;
        let (buffer, pos) = self.jumps[self.jump_index];
        self.goto_jump(buffer, pos);
    }

    /// `<C-i>`/`<Tab>`: move to a newer position in the jumplist.
    fn jump_forward(&mut self) {
        if self.jump_index + 1 >= self.jumps.len() {
            return;
        }
        self.jump_index += 1;
        let (buffer, pos) = self.jumps[self.jump_index];
        self.goto_jump(buffer, pos);
    }

    fn goto_jump(&mut self, buffer: BufferId, pos: Position) {
        if buffer != self.current && self.buffers.contains_key(&buffer) {
            self.saved_cursor.insert(self.current, self.cursor);
            self.current = buffer;
        }
        self.cursor = self.current_buffer().clamp(pos);
    }

    /// `` `{a-z} `` / `'{a-z}` (and `` `` ``/`''` for the pre-jump position).
    fn jump_to_mark(&mut self, name: char, exact: bool) {
        let target = if name == '`' || name == '\'' {
            self.last_jump_from
        } else {
            self.current_buffer().mark(name).map(|p| (self.current, p))
        };
        let Some((buffer, pos)) = target else { return };
        self.record_jump();
        let pos = if exact {
            pos
        } else {
            // `'{mark}` jumps to the first non-blank of the mark's line.
            let buf = self.buffers.get(&buffer).unwrap_or_else(|| self.current_buffer());
            Position::new(pos.line, operator::first_non_blank_col(buf, pos.line))
        };
        self.goto_jump(buffer, pos);
    }

    /// Runs a search, moving the cursor to the match (or reporting a miss),
    /// and remembers it for `n`/`N`. Records a jump first so `<C-o>` returns.
    fn do_search(&mut self, pattern: &str, forward: bool) -> EditorResponse {
        if pattern.is_empty() {
            return EditorResponse::Continue;
        }
        self.last_search = Some((pattern.to_string(), forward));
        self.record_jump();
        match search::find(self.current_buffer(), self.cursor, pattern, forward) {
            Some(pos) => {
                self.cursor = self.current_buffer().clamp(pos);
                EditorResponse::Continue
            }
            None => EditorResponse::Message(format!("pattern not found: {pattern}")),
        }
    }

    /// `n`/`N`: repeat the last search. `n` keeps its original direction;
    /// `N` (and `n` when `reverse`) inverts it.
    fn repeat_search(&mut self, reverse: bool) -> EditorResponse {
        let Some((pattern, forward)) = self.last_search.clone() else {
            return EditorResponse::Message("no previous search".to_string());
        };
        let dir = forward ^ reverse;
        self.record_jump();
        match search::find(self.current_buffer(), self.cursor, &pattern, dir) {
            Some(pos) => {
                self.cursor = self.current_buffer().clamp(pos);
                EditorResponse::Continue
            }
            None => EditorResponse::Message(format!("pattern not found: {pattern}")),
        }
    }

    /// `*`/`#`: search for the keyword under the cursor.
    fn search_word_under_cursor(&mut self, forward: bool) -> EditorResponse {
        let Some(word) = search::word_under_cursor(self.current_buffer(), self.cursor) else {
            return EditorResponse::Continue;
        };
        let pattern = search::word_pattern(&word);
        self.do_search(&pattern, forward)
    }

    /// `<C-a>`/`<C-x>`: increment/decrement the decimal number at or after the
    /// cursor on the current line, with carry (`99` -> `100`) and negative
    /// numbers handled by parsing the whole run (with any leading `-`) as an
    /// `i64`. Leaves the cursor on the last digit of the result, matching vim.
    fn increment_number(&mut self, delta: i64) -> EditorResponse {
        let line_idx = self.cursor.line;
        let Some(text) = self.current_buffer().line(line_idx) else {
            self.discard_dot();
            return EditorResponse::Continue;
        };
        let graphemes: Vec<&str> = text.graphemes(true).collect();
        let n = graphemes.len();
        let is_digit = |g: &str| g.chars().next().is_some_and(|c| c.is_ascii_digit());

        // Find the start of the digit run at or after the cursor.
        let mut i = self.cursor.col.min(n);
        if i < n && is_digit(graphemes[i]) {
            while i > 0 && is_digit(graphemes[i - 1]) {
                i -= 1;
            }
        } else {
            while i < n && !is_digit(graphemes[i]) {
                i += 1;
            }
        }
        if i >= n {
            self.discard_dot();
            return EditorResponse::Continue; // no number on the line at/after the cursor
        }
        let start = i;
        let mut end = start;
        while end < n && is_digit(graphemes[end]) {
            end += 1;
        }
        let has_minus = start > 0 && graphemes[start - 1] == "-";
        let num_start = if has_minus { start - 1 } else { start };
        let numstr: String = graphemes[num_start..end].concat();
        let Ok(value) = numstr.parse::<i64>() else {
            self.discard_dot();
            return EditorResponse::Continue;
        };
        let new = value.saturating_add(delta).to_string();
        let range = Range::new(Position::new(line_idx, num_start), Position::new(line_idx, end));
        self.begin_insert_group();
        let result = self.current_buffer_mut().apply(Edit::replace(range, new.clone()));
        self.current_buffer_mut().end_undo_group();
        if result.is_err() {
            self.discard_dot();
            return EditorResponse::Continue;
        }
        let last_col = num_start + new.graphemes(true).count();
        self.cursor = self.current_buffer().clamp(Position::new(line_idx, last_col.saturating_sub(1)));
        self.mode = Mode::Normal;
        self.commit_dot();
        EditorResponse::Continue
    }

    fn play_keys(&mut self, keys: Vec<Key>, count: usize) -> crate::Result<EditorResponse> {
        self.discard_dot();
        self.mode = Mode::Normal;
        self.replaying += 1;
        let mut result = Ok(EditorResponse::Continue);
        for _ in 0..count.max(1) {
            for k in &keys {
                result = self.handle_key(*k);
                if result.is_err() {
                    self.replaying -= 1;
                    return result;
                }
            }
        }
        self.replaying -= 1;
        result
    }

    /// `;`/`,` bookkeeping and standalone `f`/`F`/`t`/`T` both funnel
    /// through here so the "remember the last find" side effect lives in
    /// exactly one place.
    fn remember_find(&mut self, motion: Motion) {
        if let Some((kind, target)) = motion.find_kind() {
            self.last_find = Some((kind, target));
        }
    }

    fn run_operator(&mut self, operator: Operator, range: Range, granularity: Granularity, register: Option<char>) -> crate::Result<EditorResponse> {
        let enters_insert = operator.enters_insert();
        if operator.mutates() {
            self.begin_insert_group();
        }
        let sw = self.options.shiftwidth.resolve(self.options.tabstop);
        let expandtab = self.options.expandtab;
        let outcome = operator.apply(self.current_buffer_mut(), range, granularity, sw, expandtab);
        if operator.mutates() && !enters_insert {
            self.current_buffer_mut().end_undo_group();
        }
        let outcome = outcome?;

        if let Some((text, gran)) = outcome.register_write {
            if operator == Operator::Yank {
                self.registers.write_yank(register, text, gran);
            } else {
                self.registers.write_delete(register, text, gran);
            }
        }
        self.cursor = self.current_buffer().clamp(outcome.cursor);

        if enters_insert {
            self.mode = Mode::Insert;
            Ok(EditorResponse::Continue)
        } else {
            self.mode = Mode::Normal;
            if operator == Operator::Yank {
                self.discard_dot();
            } else {
                self.commit_dot();
            }
            Ok(EditorResponse::Continue)
        }
    }

    fn delete_chars(&mut self, register: Option<char>, count: Option<usize>, forward: bool) -> crate::Result<EditorResponse> {
        let n = count.unwrap_or(1).max(1);
        let buf = self.current_buffer();
        let line = self.cursor.line;
        let len = buf.line_len(line);
        let range = if forward {
            let end = (self.cursor.col + n).min(len);
            Range::new(Position::new(line, self.cursor.col), Position::new(line, end))
        } else {
            let start = self.cursor.col.saturating_sub(n);
            Range::new(Position::new(line, start), Position::new(line, self.cursor.col))
        };
        if len == 0 || range.is_empty() {
            self.mode = Mode::Normal;
            self.discard_dot();
            return Ok(EditorResponse::Continue);
        }
        let text = self.current_buffer().slice(range);
        self.begin_insert_group();
        let cursor = self.current_buffer_mut().apply(Edit::delete(range));
        self.current_buffer_mut().end_undo_group();
        let cursor = cursor?;
        self.registers.write_delete(register, text, Granularity::Charwise);
        self.cursor = self.current_buffer().clamp(cursor);
        self.mode = Mode::Normal;
        self.commit_dot();
        Ok(EditorResponse::Continue)
    }

    fn replace_char(&mut self, count: Option<usize>, ch: char) -> crate::Result<EditorResponse> {
        let n = count.unwrap_or(1).max(1);
        let buf = self.current_buffer();
        let line = self.cursor.line;
        let len = buf.line_len(line);
        if self.cursor.col + n > len {
            self.mode = Mode::Normal;
            self.discard_dot();
            return Ok(EditorResponse::Continue); // not enough characters: no-op, matching vim's beep.
        }
        let range = Range::new(Position::new(line, self.cursor.col), Position::new(line, self.cursor.col + n));
        let replacement = if ch == '\n' { "\n".repeat(n) } else { ch.to_string().repeat(n) };
        self.begin_insert_group();
        let cursor = self.current_buffer_mut().apply(Edit::replace(range, replacement));
        self.current_buffer_mut().end_undo_group();
        let cursor = cursor?;
        self.cursor = self.current_buffer().clamp(Position::new(cursor.line, cursor.col.saturating_sub(1)));
        self.mode = Mode::Normal;
        self.commit_dot();
        Ok(EditorResponse::Continue)
    }

    fn toggle_case_under_cursor(&mut self, count: Option<usize>) -> crate::Result<EditorResponse> {
        let n = count.unwrap_or(1).max(1);
        let buf = self.current_buffer();
        let line = self.cursor.line;
        let len = buf.line_len(line);
        let end = (self.cursor.col + n).min(len);
        if end <= self.cursor.col {
            self.mode = Mode::Normal;
            self.discard_dot();
            return Ok(EditorResponse::Continue);
        }
        let range = Range::new(Position::new(line, self.cursor.col), Position::new(line, end));
        let text = self.current_buffer().slice(range);
        let toggled: String = text
            .chars()
            .map(|c| {
                if c.is_uppercase() {
                    c.to_lowercase().next().unwrap_or(c)
                } else if c.is_lowercase() {
                    c.to_uppercase().next().unwrap_or(c)
                } else {
                    c
                }
            })
            .collect();
        self.begin_insert_group();
        let cursor = self.current_buffer_mut().apply(Edit::replace(range, toggled));
        self.current_buffer_mut().end_undo_group();
        let cursor = cursor?;
        self.cursor = self.current_buffer().clamp(cursor);
        self.mode = Mode::Normal;
        self.commit_dot();
        Ok(EditorResponse::Continue)
    }

    /// `J`/`{count}J`: join `count.saturating_sub(1)` following lines onto
    /// the current one (bare `J` joins one pair). Leading whitespace on the
    /// joined-in line is stripped and replaced with a single space, unless
    /// the current line is empty or the joined-in line has no content —
    /// vim's own simplified rule (real vim additionally special-cases a
    /// joined-in line starting with `)`; not implemented here).
    fn join_lines(&mut self, count: Option<usize>) -> crate::Result<EditorResponse> {
        let joins = count.map(|n| n.saturating_sub(1)).unwrap_or(1).max(1);
        self.begin_insert_group();
        let mut landed = self.cursor;
        let mut error = None;
        for _ in 0..joins {
            let line = landed.line;
            if line + 1 >= self.current_buffer().line_count() {
                break;
            }
            let this_len = self.current_buffer().line_len(line);
            let next_text = self.current_buffer().line(line + 1).unwrap_or_default();
            let next_graphemes: Vec<&str> = next_text.graphemes(true).collect();
            let mut skip = 0usize;
            while skip < next_graphemes.len() && next_graphemes[skip].chars().next().map(char::is_whitespace).unwrap_or(false) {
                skip += 1;
            }
            let needs_space = this_len > 0 && skip < next_graphemes.len();
            let joiner = if needs_space { " " } else { "" };
            let del_range = Range::new(Position::new(line, this_len), Position::new(line + 1, skip));
            match self.current_buffer_mut().apply(Edit::replace(del_range, joiner.to_string())) {
                Ok(_) => landed = Position::new(line, this_len),
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        self.current_buffer_mut().end_undo_group();
        if let Some(e) = error {
            return Err(e);
        }
        self.cursor = self.current_buffer().clamp(landed);
        self.mode = Mode::Normal;
        self.commit_dot();
        Ok(EditorResponse::Continue)
    }

    fn put(&mut self, register: Option<char>, count: usize, before: bool) -> crate::Result<EditorResponse> {
        let Some(content) = self.registers.read(register).cloned() else {
            self.mode = Mode::Normal;
            self.discard_dot();
            return Ok(EditorResponse::Continue);
        };
        if content.text.is_empty() {
            self.mode = Mode::Normal;
            self.discard_dot();
            return Ok(EditorResponse::Continue);
        }
        let repeated = content.text.repeat(count);
        self.begin_insert_group();
        let result = self.put_inner(&repeated, content.granularity, before);
        self.current_buffer_mut().end_undo_group();
        result?;
        self.mode = Mode::Normal;
        self.commit_dot();
        Ok(EditorResponse::Continue)
    }

    /// The mechanics of inserting register text at the cursor, honoring its
    /// remembered [`Granularity`] — see [`register`]'s module docs for why
    /// that distinction exists at all. Factored out of [`Self::put`] so
    /// visual-mode paste-over-selection can reuse it without also reopening
    /// an undo group around a single call (see `handle_visual_key`).
    fn put_inner(&mut self, text: &str, granularity: Granularity, before: bool) -> crate::Result<()> {
        let cur = self.cursor;
        match granularity {
            Granularity::Linewise | Granularity::Blockwise => {
                if before {
                    self.current_buffer_mut().apply(Edit::insert(Position::new(cur.line, 0), text.to_string()))?;
                    self.cursor = Position::new(cur.line, operator::first_non_blank_col(self.current_buffer(), cur.line));
                } else if cur.line + 1 < self.current_buffer().line_count() {
                    let target = cur.line + 1;
                    self.current_buffer_mut().apply(Edit::insert(Position::new(target, 0), text.to_string()))?;
                    self.cursor = Position::new(target, operator::first_non_blank_col(self.current_buffer(), target));
                } else {
                    let pos = Position::new(cur.line, self.current_buffer().line_len(cur.line));
                    let insertion = format!("\n{}", text.trim_end_matches('\n'));
                    self.current_buffer_mut().apply(Edit::insert(pos, insertion))?;
                    let target = cur.line + 1;
                    self.cursor = Position::new(target, operator::first_non_blank_col(self.current_buffer(), target));
                }
            }
            Granularity::Charwise => {
                let insert_at = if before {
                    cur
                } else {
                    let len = self.current_buffer().line_len(cur.line);
                    Position::new(cur.line, (cur.col + 1).min(len))
                };
                let landed = self.current_buffer_mut().apply(Edit::insert(insert_at, text.to_string()))?;
                self.cursor = self.current_buffer().clamp(Position::new(landed.line, landed.col.saturating_sub(1)));
            }
        }
        Ok(())
    }

    fn begin_insert_group(&mut self) {
        self.current_buffer_mut().begin_undo_group();
    }

    fn enter_insert_at(&mut self, pos: InsertPos) -> crate::Result<EditorResponse> {
        let buf = self.current_buffer();
        let line = self.cursor.line;
        self.cursor = match pos {
            InsertPos::Before => self.cursor,
            InsertPos::After => Position::new(line, (self.cursor.col + 1).min(buf.line_len(line))),
            InsertPos::LineStart => Position::new(line, operator::first_non_blank_col(buf, line)),
            InsertPos::LineEnd => Position::new(line, buf.line_len(line)),
            InsertPos::NewLineBelow | InsertPos::NewLineAbove => self.cursor, // resolved below, after the edit.
        };
        self.begin_insert_group();
        match pos {
            InsertPos::NewLineBelow => {
                let at = Position::new(line, self.current_buffer().line_len(line));
                self.current_buffer_mut().apply(Edit::insert(at, "\n".to_string()))?;
                self.cursor = Position::new(line + 1, 0);
            }
            InsertPos::NewLineAbove => {
                self.current_buffer_mut().apply(Edit::insert(Position::new(line, 0), "\n".to_string()))?;
                self.cursor = Position::new(line, 0);
            }
            _ => {}
        }
        self.mode = Mode::Insert;
        Ok(EditorResponse::Continue)
    }

    fn leave_insert(&mut self) {
        self.current_buffer_mut().end_undo_group();
        // vim moves the cursor one grapheme left when leaving Insert mode
        // (so it lands *on* the last typed character, not past it).
        if self.cursor.col > 0 {
            self.cursor = Position::new(self.cursor.line, self.cursor.col - 1);
        }
        self.cursor = self.current_buffer().clamp(self.cursor);
        self.mode = Mode::Normal;
        self.commit_dot();
    }

    // ---------------------------------------------------------------
    // Insert / Replace mode
    // ---------------------------------------------------------------

    fn handle_insert_key(&mut self, key: Key) -> crate::Result<EditorResponse> {
        // `<C-r>{reg}`: the register name arrives as the key *after* `<C-r>`.
        if self.insert_register_pending {
            self.insert_register_pending = false;
            if let Some(c) = key.as_char() {
                let reg = if c == '"' { None } else { Some(c) };
                if let Some(content) = self.registers.read(reg).cloned() {
                    let cur = self.cursor;
                    let landed = self.current_buffer_mut().apply(Edit::insert(cur, content.text))?;
                    self.cursor = landed;
                }
            }
            return Ok(EditorResponse::Continue);
        }

        // Ctrl-modified insert-mode editing shortcuts, checked before the
        // plain-character path so they are not inserted literally.
        if key.mods.ctrl {
            match key.code {
                KeyCode::Char('w') => return self.insert_delete_word_back(),
                KeyCode::Char('u') => return self.insert_delete_to_line_start(),
                KeyCode::Char('r') => {
                    self.insert_register_pending = true;
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('o') => {
                    // `<C-o>`: drop to Normal for exactly one command, then
                    // return to Insert (see the wrapper in `handle_key`).
                    self.insert_one_shot = true;
                    self.mode = Mode::Normal;
                    return Ok(EditorResponse::Continue);
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Esc => {
                self.leave_insert();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Enter => {
                let cur = self.cursor;
                let pos = self.current_buffer_mut().apply(Edit::insert(cur, "\n".to_string()))?;
                self.cursor = pos;
                Ok(EditorResponse::Continue)
            }
            KeyCode::Backspace => {
                let cur = self.cursor;
                if let Some(prev) = motion::step_left(self.current_buffer(), cur) {
                    let pos = self.current_buffer_mut().apply(Edit::delete(Range::new(prev, cur)))?;
                    self.cursor = pos;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Delete => {
                let cur = self.cursor;
                if let Some(next) = motion::step_right(self.current_buffer(), cur) {
                    let pos = self.current_buffer_mut().apply(Edit::delete(Range::new(cur, next)))?;
                    self.cursor = pos;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Tab => {
                let cur = self.cursor;
                let text = if self.options.expandtab { " ".repeat(self.options.shiftwidth.resolve(self.options.tabstop)) } else { "\t".to_string() };
                let pos = self.current_buffer_mut().apply(Edit::insert(cur, text))?;
                self.cursor = pos;
                Ok(EditorResponse::Continue)
            }
            // Arrow keys and Home/End move the insertion point, clamped to the
            // buffer. In Insert mode the cursor may sit one past the last
            // grapheme (unlike Normal mode), so end-of-line is `line_len`, not
            // `line_len - 1`.
            KeyCode::Left => {
                if let Some(prev) = motion::step_left(self.current_buffer(), self.cursor) {
                    self.cursor = prev;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Right => {
                let len = self.current_buffer().line_len(self.cursor.line);
                if self.cursor.col < len {
                    self.cursor = Position::new(self.cursor.line, self.cursor.col + 1);
                } else if let Some(next) = motion::step_right(self.current_buffer(), self.cursor) {
                    self.cursor = next;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Up => {
                let line = self.cursor.line.saturating_sub(1);
                let len = self.current_buffer().line_len(line);
                self.cursor = Position::new(line, self.cursor.col.min(len));
                Ok(EditorResponse::Continue)
            }
            KeyCode::Down => {
                let line = (self.cursor.line + 1).min(self.current_buffer().line_count().saturating_sub(1));
                let len = self.current_buffer().line_len(line);
                self.cursor = Position::new(line, self.cursor.col.min(len));
                Ok(EditorResponse::Continue)
            }
            KeyCode::Home => {
                self.cursor = Position::new(self.cursor.line, 0);
                Ok(EditorResponse::Continue)
            }
            KeyCode::End => {
                self.cursor = Position::new(self.cursor.line, self.current_buffer().line_len(self.cursor.line));
                Ok(EditorResponse::Continue)
            }
            _ => {
                if let Some(c) = key.as_char() {
                    let cur = self.cursor;
                    let pos = self.current_buffer_mut().apply(Edit::insert(cur, c.to_string()))?;
                    self.cursor = pos;
                }
                Ok(EditorResponse::Continue)
            }
        }
    }

    /// `<C-w>` in Insert mode: delete the word before the cursor.
    fn insert_delete_word_back(&mut self) -> crate::Result<EditorResponse> {
        let cur = self.cursor;
        let target = motion::word_back_for_delete(self.current_buffer(), cur);
        if target != cur {
            let pos = self.current_buffer_mut().apply(Edit::delete(Range::new(target, cur)))?;
            self.cursor = pos;
        }
        Ok(EditorResponse::Continue)
    }

    /// `<C-u>` in Insert mode: delete from the first non-blank (or line start)
    /// up to the cursor.
    fn insert_delete_to_line_start(&mut self) -> crate::Result<EditorResponse> {
        let line = self.cursor.line;
        let fnb = operator::first_non_blank_col(self.current_buffer(), line);
        // vim deletes back to the first non-blank; if already at/left of it,
        // deletes to column 0 instead.
        let cur = self.cursor;
        let target_col = if cur.col > fnb { fnb } else { 0 };
        if target_col < cur.col {
            let start = Position::new(line, target_col);
            let pos = self.current_buffer_mut().apply(Edit::delete(Range::new(start, cur)))?;
            self.cursor = pos;
        }
        Ok(EditorResponse::Continue)
    }

    /// `R`: overwrite mode. A simplified model of vim's Replace mode —
    /// typed characters overwrite what's under the cursor (or append past
    /// end of line) and Backspace steps back over them, but the *original*
    /// text is not restored on backspace the way real vim's replace stack
    /// does. Documented scope cut; see the crate-level report.
    fn handle_replace_key(&mut self, key: Key) -> crate::Result<EditorResponse> {
        match key.code {
            KeyCode::Esc => {
                self.leave_insert();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Backspace => {
                let cur = self.cursor;
                if let Some(prev) = motion::step_left(self.current_buffer(), cur) {
                    self.cursor = prev;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Enter => {
                let cur = self.cursor;
                let pos = self.current_buffer_mut().apply(Edit::insert(cur, "\n".to_string()))?;
                self.cursor = pos;
                Ok(EditorResponse::Continue)
            }
            _ => {
                if let Some(c) = key.as_char() {
                    let cur = self.cursor;
                    let len = self.current_buffer().line_len(cur.line);
                    if cur.col < len {
                        let range = Range::new(cur, Position::new(cur.line, cur.col + 1));
                        self.current_buffer_mut().apply(Edit::replace(range, c.to_string()))?;
                        self.cursor = Position::new(cur.line, cur.col + 1);
                    } else {
                        let pos = self.current_buffer_mut().apply(Edit::insert(cur, c.to_string()))?;
                        self.cursor = pos;
                    }
                }
                Ok(EditorResponse::Continue)
            }
        }
    }

    // ---------------------------------------------------------------
    // Command-line (`:`) mode
    // ---------------------------------------------------------------

    /// Handles one keystroke while the `:`/`/`/`?` prompt is open.
    ///
    /// This is the command line's full line editor. It supports the vim
    /// command-line keys — cursor movement, the word/line deletes, `<C-r>`
    /// register insertion, `<Up>`/`<Down>` history, and `<Tab>` completion —
    /// on top of the text-editing primitives in [`cmdline::CmdlineBuffer`]. The
    /// buffer owns the text-and-cursor mechanics; this method owns the *policy*
    /// that needs the wider editor (which history ring to walk, what candidates
    /// `<Tab>` should offer, what Enter *does* with the finished line).
    fn handle_command_key(&mut self, key: Key) -> crate::Result<EditorResponse> {
        // `<C-r>{reg}`: the register name arrives as the key *after* `<C-r>`,
        // exactly like insert mode's `<C-r>` (see `handle_insert_key`).
        if self.command_register_pending {
            self.command_register_pending = false;
            if let Some(c) = key.as_char() {
                let reg = if c == '"' { None } else { Some(c) };
                if let Some(content) = self.registers.read(reg) {
                    let text = content.text.clone();
                    self.cmdline.insert_str(&text);
                }
            }
            return Ok(EditorResponse::Continue);
        }

        // Ctrl-modified command-line shortcuts, matched before the plain-char
        // path so they are not typed literally.
        if key.mods.ctrl {
            match key.code {
                KeyCode::Char('w') => {
                    self.cmdline.delete_word_back();
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('u') => {
                    self.cmdline.delete_to_start();
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('h') => {
                    // `<C-h>` is Backspace; share its empty-line-cancels rule.
                    return Ok(self.command_backspace());
                }
                KeyCode::Char('b') => {
                    self.cmdline.move_home();
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('e') => {
                    self.cmdline.move_end();
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('p') => {
                    self.command_history_prev();
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('n') => {
                    self.command_history_next();
                    return Ok(EditorResponse::Continue);
                }
                KeyCode::Char('r') => {
                    self.command_register_pending = true;
                    return Ok(EditorResponse::Continue);
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Esc => {
                self.cmdline.clear();
                self.mode = Mode::Normal;
                Ok(EditorResponse::Continue)
            }
            KeyCode::Enter => {
                let line = self.cmdline.take();
                self.mode = Mode::Normal;
                match self.command_kind {
                    CommandKind::Ex => {
                        self.ex_history.push(line.clone());
                        self.execute_ex(&line)
                    }
                    CommandKind::SearchForward | CommandKind::SearchBackward => {
                        let forward = self.command_kind == CommandKind::SearchForward;
                        self.search_history.push(line.clone());
                        // An empty search line repeats the last pattern (vim's
                        // behaviour), in the direction just typed.
                        let pattern = if line.is_empty() {
                            self.last_search.as_ref().map(|(p, _)| p.clone()).unwrap_or_default()
                        } else {
                            line
                        };
                        Ok(self.do_search(&pattern, forward))
                    }
                }
            }
            KeyCode::Backspace => Ok(self.command_backspace()),
            KeyCode::Delete => {
                self.cmdline.delete_forward();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Left => {
                self.cmdline.move_left();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Right => {
                self.cmdline.move_right();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Home => {
                self.cmdline.move_home();
                Ok(EditorResponse::Continue)
            }
            KeyCode::End => {
                self.cmdline.move_end();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Up => {
                self.command_history_prev();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Down => {
                self.command_history_next();
                Ok(EditorResponse::Continue)
            }
            KeyCode::Tab => {
                // The editor models `<S-Tab>` as Tab + shift (see the UI's
                // key mapping), so the modifier picks the cycle direction.
                self.command_complete(!key.mods.shift);
                Ok(EditorResponse::Continue)
            }
            _ => {
                if let Some(c) = key.as_char() {
                    self.cmdline.insert_char(c);
                }
                Ok(EditorResponse::Continue)
            }
        }
    }

    /// Backspace on the command line, with vim's rule that a backspace on an
    /// already-empty prompt cancels the command line (leaves `Mode::Command`).
    fn command_backspace(&mut self) -> EditorResponse {
        if self.cmdline.text().is_empty() && self.cmdline.cursor() == 0 {
            self.cmdline.clear();
            self.mode = Mode::Normal;
        } else {
            self.cmdline.backspace();
        }
        EditorResponse::Continue
    }

    /// `<Up>`/`<C-p>`: walk the current prompt's history ring backward. `:` uses
    /// the ex ring, `/`?` the search ring — kept apart the way vim keeps them.
    fn command_history_prev(&mut self) {
        // Match on kind so `cmdline` and the chosen ring are borrowed as two
        // disjoint fields (a `&History` helper would borrow all of `self`).
        match self.command_kind {
            CommandKind::Ex => self.cmdline.history_prev(&self.ex_history),
            CommandKind::SearchForward | CommandKind::SearchBackward => self.cmdline.history_prev(&self.search_history),
        }
    }

    fn command_history_next(&mut self) {
        match self.command_kind {
            CommandKind::Ex => self.cmdline.history_next(&self.ex_history),
            CommandKind::SearchForward | CommandKind::SearchBackward => self.cmdline.history_next(&self.search_history),
        }
    }

    /// `<Tab>`/`<S-Tab>`: complete the token under the cursor, or advance an
    /// already-open completion cycle. Only the `:` prompt completes — `/`?`
    /// search has no command/file grammar to complete against, so `<Tab>` there
    /// is a no-op (vim inserts a literal tab; a no-op is the safer default and
    /// avoids a stray tab in a regex).
    fn command_complete(&mut self, forward: bool) {
        if self.command_kind != CommandKind::Ex {
            return;
        }
        // If a cycle is already running, just step it.
        if self.cmdline.cycle_completion(forward) {
            return;
        }
        let ctx = self.cmdline.completion_context();
        let candidates = self.command_completion_candidates(&ctx);
        self.cmdline.begin_completion(ctx.start, candidates);
    }

    /// Turns a [`cmdline::CompletionContext`] into the concrete candidate
    /// strings `<Tab>` should cycle. This is where the editor's own resources
    /// come in — the command registry, the filesystem (for file args) and the
    /// buffer table (for `:b`) — which is why candidate *generation* lives here
    /// rather than in the terminal-free `cmdline` module.
    fn command_completion_candidates(&self, ctx: &cmdline::CompletionContext) -> Vec<String> {
        match &ctx.command {
            // No command word yet -> completing the command name itself.
            None => command::complete_names(&ctx.prefix),
            Some(name) => match command::lookup(name).map(|spec| spec.arg) {
                Some(command::ArgKind::File) => {
                    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                    let mut items: Vec<String> = crate::lsp::completion::path_candidates(&ctx.prefix, &cwd)
                        .into_iter()
                        .map(|item| item.insert_text)
                        .collect();
                    items.sort();
                    items
                }
                Some(command::ArgKind::Buffer) => self.buffer_name_candidates(&ctx.prefix),
                _ => Vec::new(),
            },
        }
    }

    /// Open-buffer names (basenames) starting with `prefix`, for `:b`
    /// completion. Unnamed buffers contribute nothing (there is no name to
    /// offer). Sorted and de-duplicated for a stable cycle order.
    fn buffer_name_candidates(&self, prefix: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .buffer_order
            .iter()
            .filter_map(|id| self.buffers.get(id))
            .filter_map(|b| b.path())
            .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
            .filter(|name| name.starts_with(prefix))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Parses and runs one `:` command line (without its leading `:`). See
    /// [`ex`]'s module docs for the parse/execute split.
    pub fn execute_ex(&mut self, line: &str) -> crate::Result<EditorResponse> {
        let cmd = ex::parse(line);
        match cmd {
            ex::ExCommand::Empty => Ok(EditorResponse::Continue),
            ex::ExCommand::Write { path, then_quit, force: _ } => {
                let path = path.map(PathBuf::from);
                if then_quit {
                    Ok(EditorResponse::WriteThenQuit { path })
                } else {
                    Ok(EditorResponse::Write { path })
                }
            }
            ex::ExCommand::Quit { force } => {
                if !force && self.current_buffer().is_modified() {
                    return Err(crate::Error::UnsavedChanges);
                }
                Ok(EditorResponse::Quit)
            }
            ex::ExCommand::QuitAll { force } => {
                // The same guard `:q` uses, widened to every buffer: quit-all
                // must not silently discard an unsaved buffer in some other
                // window just because the current one is clean.
                if !force && self.any_buffer_modified() {
                    return Err(crate::Error::UnsavedChanges);
                }
                Ok(EditorResponse::QuitAll)
            }
            // `:wa`/`:wqa`/`:xa`: the write itself is the caller's I/O (see
            // `EditorResponse::WriteAll`); the guard `:q` needs does not apply to
            // a *write*-all, and the quit that follows `:wqa` is safe because the
            // write clears every buffer's modified flag first.
            ex::ExCommand::WriteAll { then_quit, force: _ } => {
                Ok(EditorResponse::WriteAll { then_quit })
            }
            ex::ExCommand::Edit { path } => {
                self.open(Path::new(&path))?;
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::NextBuffer => {
                self.switch_buffer(1);
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::PrevBuffer => {
                self.switch_buffer(-1);
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::GotoBuffer(n) => {
                if let Some(&id) = self.buffer_order.get(n.saturating_sub(1)) {
                    if id != self.current {
                        self.alternate = Some(self.current);
                    }
                    self.saved_cursor.insert(self.current, self.cursor);
                    self.current = id;
                    self.cursor = *self.saved_cursor.get(&id).unwrap_or(&Position::ORIGIN);
                }
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::GotoBufferName(name) => {
                // vim resolves `:b {name}` to the buffer whose name uniquely
                // contains `{name}`, preferring an exact basename match. kvim
                // does the same: try an exact file-name hit first, then fall
                // back to the first buffer whose path contains the string.
                let target = self
                    .buffer_order
                    .iter()
                    .find(|&&id| self.buffers.get(&id).and_then(|b| b.path()).and_then(|p| p.file_name()).map(|f| f.to_string_lossy() == name).unwrap_or(false))
                    .or_else(|| {
                        self.buffer_order
                            .iter()
                            .find(|&&id| self.buffers.get(&id).and_then(|b| b.path()).map(|p| p.to_string_lossy().contains(&name)).unwrap_or(false))
                    })
                    .copied();
                if let Some(id) = target {
                    if id != self.current {
                        self.alternate = Some(self.current);
                    }
                    self.saved_cursor.insert(self.current, self.cursor);
                    self.current = id;
                    self.cursor = *self.saved_cursor.get(&id).unwrap_or(&Position::ORIGIN);
                    Ok(EditorResponse::Continue)
                } else {
                    Err(crate::Error::UnknownCommand(format!("b {name}")))
                }
            }
            ex::ExCommand::DeleteBuffer { force, wipe } => {
                let (deleted, replacement) = self.delete_buffer(force, wipe)?;
                // The editor already switched to `replacement`; the UI still has
                // to repoint any window (active or split) that was showing the
                // deleted buffer. See `WindowCommand::BufferDeleted`.
                Ok(EditorResponse::Window(WindowCommand::BufferDeleted { deleted, replacement }))
            }
            ex::ExCommand::ListBuffers => Ok(EditorResponse::Message(self.buffer_list())),
            ex::ExCommand::Substitute { range, pattern, replacement, global } => {
                let (first, last) = range.resolve(self.cursor.line, self.current_buffer().line_count());
                let n = ex::substitute(self.current_buffer_mut(), first, last, &pattern, &replacement, global)?;
                self.cursor = self.current_buffer().clamp(self.cursor);
                // Remember it so `&` (and a future bare `:s`) can repeat it.
                self.last_substitution = Some((pattern, replacement, global));
                Ok(EditorResponse::Message(format!("{n} substitution(s)")))
            }
            ex::ExCommand::Global { pattern, cmd } => {
                let n = ex::global(self.current_buffer_mut(), &pattern, &cmd)?;
                self.cursor = self.current_buffer().clamp(self.cursor);
                Ok(EditorResponse::Message(format!("{n} line(s) changed")))
            }
            ex::ExCommand::Delete { range } => {
                let (first, last) = range.resolve(self.cursor.line, self.current_buffer().line_count());
                let pos = ex::delete_lines(self.current_buffer_mut(), first, last)?;
                self.cursor = self.current_buffer().clamp(pos);
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::NoHighlight => Ok(EditorResponse::Continue),
            ex::ExCommand::Set { key, value } => {
                self.apply_set_option(&key, value.as_deref());
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::GotoLine(spec) => {
                let (line, _) = ex::LineRange::Single(spec).resolve(self.cursor.line, self.current_buffer().line_count());
                self.record_jump();
                self.cursor = Position::new(line, operator::first_non_blank_col(self.current_buffer(), line));
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::Split { vertical, file, scratch } => Ok(EditorResponse::Window(WindowCommand::Split {
                vertical,
                file: file.map(PathBuf::from),
                scratch,
            })),
            ex::ExCommand::Only => Ok(EditorResponse::Window(WindowCommand::Only)),
            ex::ExCommand::Close => Ok(EditorResponse::Window(WindowCommand::Close)),
            // A real `:term` is a PTY-backed terminal emulator — a large
            // feature (ANSI grid parsing, keystroke forwarding, colours) that
            // is worse done badly than deferred. So this opens a scratch buffer
            // that says so plainly, rather than a broken terminal or a silent
            // no-op. Tracked as kopitiam-cj0.10.4; a future implementation may
            // embed `kopitiam-mux`.
            ex::ExCommand::Terminal => {
                self.new_buffer();
                let msg = "-- :term (terminal emulation) is not yet implemented in kvim (kopitiam-cj0.10.4) --";
                self.current_buffer_mut().apply(Edit::insert(Position::ORIGIN, msg.to_string()))?;
                self.cursor = Position::ORIGIN;
                Ok(EditorResponse::Continue)
            }
            // `:help [topic]` opens kvim's built-in Singlish manual in a fresh
            // scratch buffer — reusing the same "new_buffer + insert text"
            // machinery `:term` uses. The manual and its section line-index are
            // rendered together (see `help::render`), so `:help <topic>` can put
            // the cursor right on that section's heading. An unknown topic falls
            // back to the top of the manual rather than erroring: a typo should
            // still show *some* help, the way real vim does.
            ex::ExCommand::Help { topic } => {
                let rendered = help::render();
                self.new_buffer();
                self.current_buffer_mut().apply(Edit::insert(Position::ORIGIN, rendered.text))?;
                let line = topic
                    .as_deref()
                    .and_then(help::resolve)
                    .and_then(|id| help::section_line(&rendered.sections, id))
                    .unwrap_or(0);
                self.record_jump();
                self.cursor = self.current_buffer().clamp(Position::new(line, 0));
                Ok(EditorResponse::Continue)
            }
            ex::ExCommand::Unknown(s) => Err(crate::Error::UnknownCommand(s)),
        }
    }

    /// `<C-g>`: a one-line summary of the current buffer — its name, whether
    /// it has unsaved changes, its line count and the cursor's line as a
    /// percentage through the file. Roughly vim's own `<C-g>` echo,
    /// `"name" [Modified] N lines --P%--`.
    fn file_info(&self) -> String {
        let buf = self.current_buffer();
        let name = buf.path().map(|p| p.display().to_string()).unwrap_or_else(|| "[No Name]".to_string());
        let modified = if buf.is_modified() { " [Modified]" } else { "" };
        let lines = buf.line_count();
        let cur = self.cursor.line + 1;
        let pct = (cur * 100).checked_div(lines).unwrap_or(0);
        format!("\"{name}\"{modified} {lines} line(s) --{pct}%--")
    }

    /// `<C-^>`/`<C-6>`: swap to the alternate (`#`) buffer and make the buffer
    /// we left the new alternate, so the key toggles between the two. Restores
    /// the alternate's saved cursor. A friendly note (not an error) when there
    /// is no alternate yet or it has since been closed.
    fn edit_alternate(&mut self) -> EditorResponse {
        let Some(alt) = self.alternate else {
            return EditorResponse::Message("no alternate file".to_string());
        };
        if alt == self.current || !self.buffers.contains_key(&alt) {
            return EditorResponse::Message("no alternate file".to_string());
        }
        self.saved_cursor.insert(self.current, self.cursor);
        self.alternate = Some(self.current);
        self.current = alt;
        let restored = *self.saved_cursor.get(&alt).unwrap_or(&Position::ORIGIN);
        self.cursor = self.current_buffer().clamp(restored);
        self.mode = Mode::Normal;
        EditorResponse::Continue
    }

    fn switch_buffer(&mut self, delta: i32) {
        let Some(idx) = self.buffer_order.iter().position(|&id| id == self.current) else { return };
        let len = self.buffer_order.len() as i32;
        let next = (idx as i32 + delta).rem_euclid(len.max(1)) as usize;
        let Some(&id) = self.buffer_order.get(next) else { return };
        if id != self.current {
            self.alternate = Some(self.current);
        }
        self.saved_cursor.insert(self.current, self.cursor);
        self.current = id;
        self.cursor = *self.saved_cursor.get(&id).unwrap_or(&Position::ORIGIN);
    }

    fn apply_set_option(&mut self, key: &str, value: Option<&str>) {
        let on = value.map(|v| v != "false").unwrap_or(true);
        match key {
            "number" | "nu" => self.options.number = on,
            "relativenumber" | "rnu" => self.options.relativenumber = on,
            "wrap" => self.options.wrap = on,
            "spell" => self.options.spell = on,
            "expandtab" | "et" => self.options.expandtab = on,
            "tabstop" | "ts" => {
                if let Some(v) = value.and_then(|v| v.parse().ok()) {
                    self.options.tabstop = v;
                }
            }
            "shiftwidth" | "sw" => {
                if let Some(v) = value.and_then(|v| v.parse().ok()) {
                    self.options.shiftwidth = crate::config::bool_or_usize::ShiftWidth(v);
                }
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------
    // Visual / Visual-line / Visual-block mode
    // ---------------------------------------------------------------

    /// Dispatches a key while in any of the three visual modes.
    ///
    /// # Why this does not reuse `Pending`
    ///
    /// Visual mode's grammar is genuinely smaller and shaped differently
    /// from the operator-pending grammar `pending` implements: an operator
    /// key (`d`, `y`, `c`, ...) acts *immediately* on the selection — there
    /// is no "waiting for a motion" phase, because the selection already is
    /// the range. `i`/`a` mean *extend the selection to a text object*
    /// here, where in Normal mode they mean *enter Insert mode* — the same
    /// overload `pending` resolves by checking whether an operator is
    /// already pending, which has no equivalent concept in visual mode.
    /// Building one state machine that served both grammars correctly would
    /// need a mode parameter threaded through nearly every transition for a
    /// grammar that is, in the end, "optional count, then act": not enough
    /// shared structure to justify forcing them into the same type. Motion
    /// recognition ([`pending::simple_motion`]) *is* shared, so no motion
    /// table is duplicated — only the small amount of sequencing around
    /// `g`/`f`/`F`/`t`/`T`/text-objects is written twice, once here and once
    /// implicitly inside `Pending`.
    fn handle_visual_key(&mut self, key: Key) -> crate::Result<EditorResponse> {
        if key.code == KeyCode::Esc {
            self.exit_visual();
            return Ok(EditorResponse::Continue);
        }

        if let Some(scope) = self.visual_object_pending.take() {
            if let Some(obj) = pending::text_object_for(key)
                && let Some((range, gran)) = text_object::resolve(self.current_buffer(), self.cursor, obj, scope)
            {
                let (start, end) = range.normalized();
                self.visual_anchor = start;
                self.cursor = self.current_buffer().clamp(step_back_one(self.current_buffer(), end));
                if gran == Granularity::Linewise {
                    self.mode = Mode::VisualLine;
                    self.visual_kind = VisualKind::Linewise;
                }
            }
            return Ok(EditorResponse::Continue);
        }

        if self.visual_g_pending {
            self.visual_g_pending = false;
            match key.code {
                KeyCode::Char('g') => self.cursor = Motion::FileStart.apply(self.current_buffer(), self.cursor, None),
                KeyCode::Char('e') => self.cursor = Motion::WordEndBack.apply(self.current_buffer(), self.cursor, None),
                KeyCode::Char('u') => return self.run_visual_operator(Operator::LowerCase),
                KeyCode::Char('U') => return self.run_visual_operator(Operator::UpperCase),
                KeyCode::Char('~') => return self.run_visual_operator(Operator::ToggleCase),
                _ => {}
            }
            return Ok(EditorResponse::Continue);
        }

        if let Some(kind) = self.visual_find_pending.take() {
            if let Some(c) = key.as_char() {
                let motion = Motion::FindChar { kind, target: c };
                self.cursor = motion.apply(self.current_buffer(), self.cursor, None);
                self.last_find = Some((kind, c));
            }
            return Ok(EditorResponse::Continue);
        }

        match key.code {
            KeyCode::Char('v') => {
                if self.mode == Mode::Visual {
                    self.exit_visual();
                } else {
                    self.mode = Mode::Visual;
                    self.visual_kind = VisualKind::Charwise;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('V') => {
                if self.mode == Mode::VisualLine {
                    self.exit_visual();
                } else {
                    self.mode = Mode::VisualLine;
                    self.visual_kind = VisualKind::Linewise;
                }
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('g') => {
                self.visual_g_pending = true;
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('i') => {
                self.visual_object_pending = Some(ObjectScope::Inner);
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('a') => {
                self.visual_object_pending = Some(ObjectScope::Around);
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('f') => {
                self.visual_find_pending = Some(FindKind::To);
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('F') => {
                self.visual_find_pending = Some(FindKind::ToBack);
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('t') => {
                self.visual_find_pending = Some(FindKind::Till);
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('T') => {
                self.visual_find_pending = Some(FindKind::TillBack);
                Ok(EditorResponse::Continue)
            }
            KeyCode::Char('d') | KeyCode::Char('x') => self.run_visual_operator(Operator::Delete),
            KeyCode::Char('c') | KeyCode::Char('s') => self.run_visual_operator(Operator::Change),
            KeyCode::Char('y') => self.run_visual_operator(Operator::Yank),
            KeyCode::Char('>') => self.run_visual_operator(Operator::Indent),
            KeyCode::Char('<') => self.run_visual_operator(Operator::Dedent),
            KeyCode::Char('u') => self.run_visual_operator(Operator::LowerCase),
            KeyCode::Char('U') => self.run_visual_operator(Operator::UpperCase),
            KeyCode::Char('~') => self.run_visual_operator(Operator::ToggleCase),
            KeyCode::Char('J') => {
                self.exit_visual();
                self.join_lines(None)
            }
            KeyCode::Char('p') | KeyCode::Char('P') => self.visual_paste(),
            // `o`/`O`: swap the cursor and the anchor, so the *other* end of
            // the selection becomes the one you extend from.
            KeyCode::Char('o') | KeyCode::Char('O') => {
                std::mem::swap(&mut self.visual_anchor, &mut self.cursor);
                Ok(EditorResponse::Continue)
            }
            _ => {
                if let Some(motion) = pending::simple_motion(key) {
                    self.cursor = motion.apply(self.current_buffer(), self.cursor, None);
                }
                Ok(EditorResponse::Continue)
            }
        }
    }

    fn exit_visual(&mut self) {
        // Remember the selection so `gv` can restore it.
        self.last_visual = Some((self.visual_anchor, self.cursor, self.visual_kind));
        self.mode = Mode::Normal;
        self.visual_g_pending = false;
        self.visual_find_pending = None;
        self.visual_object_pending = None;
    }

    /// The selection's range and granularity, ready to hand to
    /// [`Operator::apply`] — computed the same way for every visual
    /// operator key, so `handle_visual_key`'s match arms stay one line each.
    fn visual_range(&self) -> (Range, Granularity) {
        let buf = self.current_buffer();
        let (a, b) = if self.visual_anchor <= self.cursor { (self.visual_anchor, self.cursor) } else { (self.cursor, self.visual_anchor) };
        match self.visual_kind {
            VisualKind::Charwise => {
                let end = motion::step_right(buf, b).unwrap_or(Position::new(b.line, buf.line_len(b.line)));
                (Range::new(a, end), Granularity::Charwise)
            }
            VisualKind::Linewise => (operator::linewise_content_range(buf, a.line, b.line), Granularity::Linewise),
            VisualKind::Blockwise => {
                // Simplified block range: the smallest charwise span
                // covering every selected line's column band, rather than a
                // true per-line rectangle. Real block-mode editing
                // (`I`/`A` across the block, ragged-line handling) is a
                // documented scope cut — see the crate-level report.
                let end = motion::step_right(buf, b).unwrap_or(Position::new(b.line, buf.line_len(b.line)));
                (Range::new(a, end), Granularity::Blockwise)
            }
        }
    }

    fn run_visual_operator(&mut self, operator: Operator) -> crate::Result<EditorResponse> {
        let (range, granularity) = self.visual_range();
        self.exit_visual();
        self.run_operator(operator, range, granularity, None)
    }

    /// `p`/`P` over a visual selection: replace it with the unnamed
    /// register's contents, as one undo step.
    fn visual_paste(&mut self) -> crate::Result<EditorResponse> {
        let Some(content) = self.registers.read(None).cloned() else {
            self.exit_visual();
            return Ok(EditorResponse::Continue);
        };
        let (range, granularity) = self.visual_range();
        self.begin_insert_group();
        let result = (|| -> crate::Result<()> {
            let outcome = Operator::Delete.apply(self.current_buffer_mut(), range, granularity, 0, false)?;
            self.cursor = self.current_buffer().clamp(outcome.cursor);
            self.put_inner(&content.text, content.granularity, true)
        })();
        self.current_buffer_mut().end_undo_group();
        result?;
        self.exit_visual();
        self.mode = Mode::Normal;
        self.commit_dot();
        Ok(EditorResponse::Continue)
    }
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

/// `cw`/`cW` behave like `ce`/`cE` when the cursor sits on a non-blank
/// character: vim's famous "`cw` doesn't eat trailing whitespace" quirk. It
/// is implemented here, at the point an operator and motion combine, rather
/// than inside [`Motion::apply`], because it is not a property of the
/// motion `w` — `dw` and `yw` are unaffected — it is a property of this one
/// operator's interaction with this one motion.
fn adjust_change_word_motion(operator: Operator, motion: Motion, buf: &Buffer, pos: Position) -> Motion {
    if operator != Operator::Change {
        return motion;
    }
    let on_blank = buf.grapheme_at(pos).map(|g| g.chars().next().map(char::is_whitespace).unwrap_or(true)).unwrap_or(true);
    if on_blank {
        return motion;
    }
    match motion {
        Motion::WordForward => Motion::WordEnd,
        Motion::WordForwardBig => Motion::WordEndBig,
        other => other,
    }
}

fn step_back_one(buf: &Buffer, pos: Position) -> Position {
    if pos.col > 0 {
        Position::new(pos.line, pos.col - 1)
    } else if pos.line > 0 {
        let prev = pos.line - 1;
        Position::new(prev, buf.line_len(prev).saturating_sub(1))
    } else {
        pos
    }
}

/// Normalises a key for keymap comparison: for a **character** key the case is
/// already carried by the `char` itself (`'K'` vs `'k'`), so the physical
/// `shift` bit is redundant — and, worse, distinguishing on it breaks matching,
/// because [`key::parse`] compiles `"K"` with `shift = false` while a terminal
/// delivers a typed `K` with `shift = true`. Stripping `shift` for `Char` keys
/// makes an uppercase-letter mapping (e.g. `K` → hover) actually fire. Non-char
/// keys (`<S-Tab>`) keep their modifiers untouched, where `shift` is meaningful.
fn normalize_for_keymap(key: Key) -> Key {
    match key.code {
        key::KeyCode::Char(_) => Key { code: key.code, mods: key::Modifiers { shift: false, ..key.mods } },
        _ => key,
    }
}

/// The result of checking a key against the compiled keymap table.
enum KeymapDispatch {
    /// The key completed a configured mapping.
    Action(crate::config::Action),
    /// The key extended a still-viable prefix; more keys are needed.
    Buffered,
    /// No keymap starts this way. Whatever was previously buffered belongs
    /// to the vi grammar after all and is returned so the caller can replay
    /// it through `Pending` before feeding the current key.
    Replay(Vec<Key>),
    /// There was nothing buffered and this key does not start any keymap.
    None,
}

/// Compiles [`crate::config::Config::keymaps`] into concrete key sequences,
/// substituting `<leader>` for the configured leader key before parsing —
/// see [`key::parse`]'s docs for why that substitution has to happen first.
fn compile_keymaps(config: &crate::config::Config) -> Vec<(Vec<Key>, crate::config::Action)> {
    let leader = config.leader.to_string();
    config
        .keymaps
        .iter()
        .filter(|k| k.mode.is_empty() || k.mode == "n")
        .map(|k| {
            let substituted = k.lhs.replace("<leader>", &leader);
            (key::parse(&substituted), k.action.clone())
        })
        .collect()
}

/// The which-key sibling of [`compile_keymaps`]: the same compiled sequences
/// paired with each mapping's [`crate::config::Keymap::desc`] instead of its
/// action.
fn compile_keymap_descs(config: &crate::config::Config) -> Vec<(Vec<Key>, String)> {
    let leader = config.leader.to_string();
    config
        .keymaps
        .iter()
        .filter(|k| k.mode.is_empty() || k.mode == "n")
        .map(|k| {
            let substituted = k.lhs.replace("<leader>", &leader);
            (key::parse(&substituted), k.desc.clone())
        })
        .collect()
}

/// One row of the which-key popup: the next key(s) after the pending prefix,
/// and where they lead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhichKeyItem {
    /// The label for the key that comes *next* after the pending prefix (e.g.
    /// `g`, `e`, `<Esc>`).
    pub keys: String,
    /// Either the mapping's description (for a leaf) or a `+group` marker (when
    /// this key only extends toward longer mappings).
    pub desc: String,
    /// Whether this row is a prefix leading to further mappings (`g` → `gd`,
    /// `gr`) rather than a mapping in its own right. which-key renders groups
    /// distinctly (a leading `+`).
    pub is_group: bool,
}

/// Computes the which-key rows for `prefix` against the compiled
/// `(sequence, desc)` table.
///
/// For every mapping whose sequence strictly extends `prefix`, the row is keyed
/// by the single key immediately after `prefix`. If some mapping's sequence is
/// exactly `prefix + [next]` it is a *leaf* (its own description); if `next`
/// only ever appears in longer sequences it is a *group* (`+3` continuations).
/// Rows are sorted by their key label for a stable popup.
fn which_key_for(prefix: &[Key], table: &[(Vec<Key>, String)]) -> Vec<WhichKeyItem> {
    if prefix.is_empty() {
        return Vec::new();
    }
    use std::collections::BTreeMap;
    // key label -> (leaf desc if any, count of deeper continuations)
    let mut groups: BTreeMap<String, (Option<String>, usize)> = BTreeMap::new();
    for (seq, desc) in table {
        if seq.len() <= prefix.len() || !seq.starts_with(prefix) {
            continue;
        }
        let next = seq[prefix.len()];
        let label = key_label(next);
        let entry = groups.entry(label).or_insert((None, 0));
        if seq.len() == prefix.len() + 1 {
            entry.0 = Some(desc.clone());
        } else {
            entry.1 += 1;
        }
    }
    groups
        .into_iter()
        .map(|(keys, (leaf, deeper))| match leaf {
            Some(desc) => WhichKeyItem { keys, desc, is_group: false },
            None => WhichKeyItem { keys, desc: format!("+{deeper} more"), is_group: true },
        })
        .collect()
}

/// A short, popup-friendly label for a single [`Key`]. Space is spelled out
/// (`<Space>`) because a literal blank would render as nothing; everything else
/// reuses [`Key`]'s own `Display`.
fn key_label(key: Key) -> String {
    match key.code {
        key::KeyCode::Char(' ') => "<Space>".to_string(),
        _ => key.to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! The keystroke-sequence harness the brief asks for: [`run`] feeds a
    //! vim-notation key sequence into a fresh [`Editor`] and returns the
    //! resulting buffer text, so every test below reads like the vim
    //! session it is asserting about instead of a wall of `handle_key`
    //! calls. Tests that need to inspect something other than the final
    //! text (a register's contents, the mode, the response variant) build
    //! their own [`Editor`] and call [`feed`] directly.
    use super::*;

    fn editor_with(text: &str) -> Editor {
        let mut ed = Editor::new();
        let id = ed.current;
        ed.buffers.insert(id, Buffer::from_str(text));
        ed.cursor = Position::ORIGIN;
        ed
    }

    fn feed(ed: &mut Editor, keys: &str) {
        for k in key::parse(keys) {
            ed.handle_key(k).unwrap_or_else(|e| panic!("key {k:?} (from {keys:?}) errored: {e}"));
        }
    }

    fn run(initial: &str, keys: &str) -> String {
        let mut ed = editor_with(initial);
        feed(&mut ed, keys);
        ed.buffer().text()
    }

    // -----------------------------------------------------------------
    // Motions + the delete operator: exclusive vs. inclusive, counts.
    // -----------------------------------------------------------------

    #[test]
    fn dw_deletes_exclusive_of_the_next_words_start() {
        assert_eq!(run("foo bar baz", "dw"), "bar baz");
    }

    #[test]
    fn d2w_and_2dw_both_delete_two_words() {
        assert_eq!(run("foo bar baz qux", "d2w"), "baz qux");
        assert_eq!(run("foo bar baz qux", "2dw"), "baz qux");
    }

    #[test]
    fn counts_multiply_2d3w_deletes_six_words() {
        assert_eq!(run("one two three four five six seven", "2d3w"), "seven");
    }

    #[test]
    fn de_deletes_inclusive_of_the_words_last_character() {
        assert_eq!(run("foo bar", "de"), " bar");
    }

    #[test]
    fn d_dollar_deletes_to_end_of_line_inclusive() {
        assert_eq!(run("foo bar", "d$"), "");
    }

    #[test]
    fn dd_and_3dd_delete_whole_lines() {
        assert_eq!(run("foo\nbar\nbaz", "dd"), "bar\nbaz");
        assert_eq!(run("a\nb\nc\nd\ne", "3dd"), "d\ne");
    }

    #[test]
    fn dg_deletes_from_cursor_to_end_of_file_linewise() {
        // move down to line "b" first, so the deletion doesn't start at
        // line 0 — exercises the "borrow the previous line's newline"
        // branch of `linewise_delete_range` (see `operator`'s docs).
        assert_eq!(run("a\nb\nc\nd", "jdG"), "a");
    }

    #[test]
    fn dgg_deletes_from_cursor_to_start_of_file_linewise() {
        assert_eq!(run("a\nb\nc\nd", "jjdgg"), "d");
    }

    // -----------------------------------------------------------------
    // Text objects.
    // -----------------------------------------------------------------

    // These position the cursor with counted `l` rather than bare `f{c}`:
    // the maintainer's config remaps bare `f` to a hop-to-word plugin in
    // every mode (see `handle_normal_key`'s docs and
    // `gg_motion_falls_back_correctly_after_starting_as_a_keymap_prefix`),
    // so a *standalone* `f(` here would be swallowed by that keymap before
    // it ever reaches the vi grammar — see
    // `operator_composed_find_still_works_despite_the_f_keymap_shadow`
    // below for the case where `f` *is* reachable (composed with a pending
    // operator).

    #[test]
    fn ci_paren_changes_inside_the_nearest_parens() {
        // cursor lands exactly ON the '(' — the classic edge case for a
        // bracket text object, since `find_enclosing` has to treat sitting
        // on the delimiter itself as "inside".
        assert_eq!(run("foo(bar)baz", "3lci(hi<Esc>"), "foo(hi)baz");
    }

    #[test]
    fn ci_quote_changes_inside_double_quotes() {
        assert_eq!(run("say \"hello\" now", "4lci\"bye<Esc>"), "say \"bye\" now");
    }

    #[test]
    fn ca_brace_changes_around_braces_including_them() {
        assert_eq!(run("x{body}y", "lca{new<Esc>"), "xnewy");
    }

    #[test]
    fn operator_composed_find_still_works_despite_the_f_keymap_shadow() {
        // Once `d` is pending, `Pending` is no longer idle, so the keymap
        // layer steps aside (see `handle_normal_key`'s docs) and `f`
        // behaves as vim's ordinary find-character motion — inclusive of
        // the found character itself, so `df(` removes "foo(" entirely.
        assert_eq!(run("foo(bar)baz", "df("), "bar)baz");
    }

    #[test]
    fn diw_deletes_the_inner_word_only_leaving_surrounding_space() {
        assert_eq!(run("foo bar baz", "wdiw"), "foo  baz");
    }

    #[test]
    fn daw_deletes_the_word_and_its_trailing_space() {
        assert_eq!(run("foo bar baz", "wdaw"), "foo baz");
    }

    #[test]
    fn cit_changes_inside_a_tag_body() {
        assert_eq!(run("<b>hello</b>", "3lcitworld<Esc>"), "<b>world</b>");
    }

    // -----------------------------------------------------------------
    // Registers + granularity: the dd/p-vs-dw/p distinction the brief
    // calls out by name.
    // -----------------------------------------------------------------

    #[test]
    fn yy_then_p_pastes_a_whole_line_below() {
        assert_eq!(run("foo\nbar\nbaz", "yyp"), "foo\nfoo\nbar\nbaz");
    }

    #[test]
    fn yw_then_p_pastes_inline_after_the_cursor() {
        let text = run("foo bar", "ywp");
        // Charwise: no new line is created, and the yanked text lands
        // immediately after the cursor's original character — vim's actual
        // (slightly surprising) rule, computed here rather than
        // hand-transcribed to avoid an off-by-one typo.
        let expected = format!("f{}oo bar", "foo ");
        assert_eq!(text, expected);
        assert_eq!(text.lines().count(), 1, "a charwise put must not create a new line");
    }

    #[test]
    fn named_register_survives_an_unrelated_unnamed_delete() {
        // "ayy stashes "foo" in register a; the later `dd` (no explicit
        // register) only touches the unnamed register, so "ap must still
        // paste what was yanked, not what was deleted.
        assert_eq!(run("foo\nbar\nbaz", "\"ayyjdd\"ap"), "foo\nbaz\nfoo");
    }

    // -----------------------------------------------------------------
    // Single-key commands: x/X, r, ~, J.
    // -----------------------------------------------------------------

    #[test]
    fn x_and_3x_delete_characters_forward() {
        assert_eq!(run("hello", "x"), "ello");
        assert_eq!(run("hello", "3x"), "lo");
    }

    #[test]
    fn r_replaces_the_character_under_the_cursor() {
        assert_eq!(run("hello", "rX"), "Xello");
    }

    #[test]
    fn tilde_toggles_case_and_advances() {
        assert_eq!(run("hello", "~"), "Hello");
        // Two `~` in a row toggle two different characters, because `~`
        // advances the cursor after each toggle (unlike `r`, which stays
        // put) — "hEllo" -> "HEllo" (h->H, cursor now on 'E') -> "Hello"
        // (E->e).
        assert_eq!(run("hEllo", "~~"), "Hello");
    }

    #[test]
    fn j_joins_lines_with_a_single_space() {
        assert_eq!(run("foo\nbar", "J"), "foo bar");
    }

    // -----------------------------------------------------------------
    // Indent operators.
    // -----------------------------------------------------------------

    #[test]
    fn shift_right_and_left_use_a_tab_by_default() {
        assert_eq!(run("foo", ">>"), "\tfoo");
        assert_eq!(run("\tfoo", "<<"), "foo");
    }

    #[test]
    fn a_count_indents_that_many_lines() {
        assert_eq!(run("a\nb\nc", "3>>"), "\ta\n\tb\n\tc");
    }

    // -----------------------------------------------------------------
    // Dot-repeat.
    // -----------------------------------------------------------------

    #[test]
    fn dot_repeats_the_last_change() {
        assert_eq!(run("one two three", "dw."), "three");
    }

    #[test]
    fn dot_repeats_an_insert_session_verbatim() {
        assert_eq!(run("", "ihi<Esc>."), "hhii");
    }

    // -----------------------------------------------------------------
    // Macros.
    // -----------------------------------------------------------------

    #[test]
    fn a_recorded_macro_replays_with_at_reg() {
        assert_eq!(run("one two three four", "qadwq@a"), "three four");
    }

    #[test]
    fn at_at_replays_the_last_played_macro() {
        assert_eq!(run("one two three four five", "qadwq@a@@"), "four five");
    }

    // -----------------------------------------------------------------
    // Undo: an insert session is one undo step.
    // -----------------------------------------------------------------

    #[test]
    fn an_insert_session_undoes_in_one_step() {
        assert_eq!(run("", "ihello<Esc>u"), "");
    }

    #[test]
    fn a_change_operator_groups_the_delete_and_insert_together() {
        assert_eq!(run("foo bar", "cwbaz<Esc>u"), "foo bar");
    }

    // -----------------------------------------------------------------
    // Visual mode.
    // -----------------------------------------------------------------

    #[test]
    fn charwise_visual_delete() {
        assert_eq!(run("foo\nbar\nbaz", "vjd"), "ar\nbaz");
    }

    #[test]
    fn linewise_visual_delete() {
        assert_eq!(run("foo\nbar\nbaz", "Vd"), "bar\nbaz");
    }

    #[test]
    fn visual_inner_word_change() {
        assert_eq!(run("foo bar baz", "wviwcXX<Esc>"), "foo XX baz");
    }

    // -----------------------------------------------------------------
    // Ex commands.
    // -----------------------------------------------------------------

    #[test]
    fn substitute_with_g_flag_replaces_every_match_on_the_line() {
        assert_eq!(run("foo foo", ":s/foo/bar/g<CR>"), "bar bar");
    }

    #[test]
    fn percent_s_with_empty_pattern_and_replacement_parses_and_is_a_no_op() {
        assert_eq!(run("abc", ":%s///<CR>"), "abc");
    }

    #[test]
    fn ranged_delete_removes_the_given_lines() {
        assert_eq!(run("a\nb\nc\nd\ne", ":2,4d<CR>"), "a\ne");
    }

    #[test]
    fn ex_write_and_quit_are_returned_as_effects_not_performed() {
        let mut ed = editor_with("hello");
        let resp = ed.execute_ex("w /tmp/should-not-be-created-by-this-test.kvimtest").unwrap();
        assert_eq!(resp, EditorResponse::Write { path: Some(PathBuf::from("/tmp/should-not-be-created-by-this-test.kvimtest")) });
        assert!(!Path::new("/tmp/should-not-be-created-by-this-test.kvimtest").exists(), "Editor must not perform I/O itself for :w");
    }

    #[test]
    fn quit_all_quits_when_clean_refuses_when_dirty_and_bang_forces() {
        let mut ed = editor_with("hello");
        // A clean buffer: `:qa` exits the whole editor.
        assert_eq!(ed.execute_ex("qa").unwrap(), EditorResponse::QuitAll);

        // Make it dirty, and `:qa` must refuse — the same guard `:q` uses.
        // The `<Esc>` closes the insert session so the edit commits to the undo
        // tree; `is_modified` is `current_id != saved_at`, which only advances
        // once the group ends (see `Buffer::saved_at`). This also mirrors how a
        // user reaches `:qa` — from Normal mode, not mid-insert.
        feed(&mut ed, "ix<Esc>");
        assert!(ed.any_buffer_modified());
        assert!(
            matches!(ed.execute_ex("qa"), Err(crate::Error::UnsavedChanges)),
            "`:qa` on a modified buffer must refuse"
        );
        // `!` overrides, discarding the changes.
        assert_eq!(ed.execute_ex("qa!").unwrap(), EditorResponse::QuitAll);
    }

    #[test]
    fn quit_all_checks_every_buffer_not_just_the_current_one() {
        let mut ed = editor_with("clean");
        let first = ed.buffer_id();
        // A second buffer, made dirty, then switch focus back to the clean one.
        let second = ed.new_buffer();
        ed.set_active(second, Position::ORIGIN);
        feed(&mut ed, "iDIRTY<Esc>"); // `<Esc>` commits the insert so the buffer reads modified
        ed.set_active(first, Position::ORIGIN);

        assert!(!ed.current_buffer().is_modified(), "the current buffer is clean");
        assert!(ed.any_buffer_modified(), "but another buffer is dirty");
        assert!(
            matches!(ed.execute_ex("qa"), Err(crate::Error::UnsavedChanges)),
            "`:qa` must refuse while any buffer is dirty, not just the current one"
        );
    }

    #[test]
    fn bd_deletes_current_buffer_and_lands_on_the_other() {
        let mut ed = editor_with("first");
        let first = ed.buffer_id();
        let second = ed.new_buffer(); // empty scratch, now current
        feed(&mut ed, "isecond<Esc>"); // give it its own text

        // `:bd!` (force, since we just made `second` dirty) removes it and
        // switches focus to the surviving buffer, telling the UI to repoint.
        let resp = ed.execute_ex("bd!").unwrap();
        assert_eq!(resp, EditorResponse::Window(WindowCommand::BufferDeleted { deleted: second, replacement: first }));
        assert_eq!(ed.buffer_id(), first, "focus must land on the surviving buffer");
        assert_eq!(ed.current_buffer().text(), "first", "and it must show the surviving buffer's text");
        assert!(ed.buffer_by_id(second).is_none(), "the deleted buffer must be gone from the table");
    }

    #[test]
    fn bd_refuses_a_modified_buffer_and_bang_forces() {
        let mut ed = editor_with("keep");
        let first = ed.buffer_id();
        let second = ed.new_buffer();
        feed(&mut ed, "idirty<Esc>"); // `<Esc>` commits so the buffer reads modified
        assert!(ed.current_buffer().is_modified());

        // Plain `:bd` must refuse — the same unsaved guard `:q` uses.
        assert!(
            matches!(ed.execute_ex("bd"), Err(crate::Error::UnsavedChanges)),
            "`:bd` on a modified buffer must refuse"
        );
        assert_eq!(ed.buffer_id(), second, "a refused `:bd` must not delete anything");

        // `:bd!` overrides, discarding the changes.
        let resp = ed.execute_ex("bd!").unwrap();
        assert_eq!(resp, EditorResponse::Window(WindowCommand::BufferDeleted { deleted: second, replacement: first }));
        assert_eq!(ed.buffer_id(), first);
    }

    #[test]
    fn bd_on_the_last_buffer_opens_a_fresh_empty_one() {
        // vim never leaves you with zero buffers; deleting the only buffer must
        // replace it with a fresh empty scratch rather than deleting into nothing.
        let mut ed = editor_with("only");
        let only_id = ed.buffer_id();
        let resp = ed.execute_ex("bd").unwrap();
        match resp {
            EditorResponse::Window(WindowCommand::BufferDeleted { deleted, replacement }) => {
                assert_eq!(deleted, only_id);
                assert_ne!(replacement, only_id, "the replacement must be a brand-new buffer");
            }
            other => panic!("expected a BufferDeleted window command, got {other:?}"),
        }
        assert_ne!(ed.buffer_id(), only_id, "the old buffer must no longer be current");
        assert_eq!(ed.current_buffer().text(), "", "and the fresh buffer must be empty");
        assert!(ed.buffer_by_id(only_id).is_none());
    }

    #[test]
    fn ls_lists_every_open_buffer_with_a_modified_flag() {
        let mut ed = editor_with("aaa");
        ed.new_buffer(); // a second, empty, now-active buffer
        feed(&mut ed, "ibbb<Esc>"); // make the active one modified

        let EditorResponse::Message(list) = ed.execute_ex("ls").unwrap() else {
            panic!("`:ls` must report a message");
        };
        assert_eq!(list.lines().count(), 2, "one line per open buffer");
        assert!(list.contains("%a"), "the active buffer must be marked");
        assert!(list.contains('+'), "the modified buffer must carry a `+` flag");
    }

    #[test]
    fn help_opens_a_buffer_of_singlish_manual_text() {
        let mut ed = editor_with("hello");
        let resp = ed.execute_ex("help").unwrap();
        assert_eq!(resp, EditorResponse::Continue);
        // A fresh scratch buffer now holds the manual, cursor at the top.
        assert_eq!(ed.cursor, Position::new(0, 0));
        let text = ed.current_buffer().text();
        assert!(text.contains("kvim :help"), "the help buffer must hold the manual");
        assert!(text.contains("<leader>e"), "and quote real key names verbatim");
    }

    #[test]
    fn help_topic_jumps_the_cursor_to_that_section() {
        let mut ed = editor_with("hello");
        ed.execute_ex("help lsp").unwrap();
        let line = ed.cursor.line;
        // The cursor's line must be the LSP section's heading, not the top.
        assert!(line > 0, "`:help lsp` must jump past the overview");
        let heading = ed.current_buffer().line(line).unwrap();
        assert!(heading.contains("LSP"), "landed on {heading:?}, expected the LSP heading");
    }

    #[test]
    fn help_with_an_unknown_topic_falls_back_to_the_top() {
        let mut ed = editor_with("hello");
        ed.execute_ex("help nonsense-topic-lah").unwrap();
        assert_eq!(ed.cursor, Position::new(0, 0), "an unknown topic still shows the manual, at the top");
    }

    #[test]
    fn write_all_variants_are_returned_as_effects() {
        let mut ed = editor_with("hello");
        assert_eq!(ed.execute_ex("wa").unwrap(), EditorResponse::WriteAll { then_quit: false });
        assert_eq!(ed.execute_ex("wqa").unwrap(), EditorResponse::WriteAll { then_quit: true });
        assert_eq!(ed.execute_ex("xa").unwrap(), EditorResponse::WriteAll { then_quit: true });
        // Write-all is not gated by the unsaved guard — writing is the point.
        feed(&mut ed, "ix<Esc>");
        assert_eq!(ed.execute_ex("wa").unwrap(), EditorResponse::WriteAll { then_quit: false });
    }

    // -----------------------------------------------------------------
    // Command-line line editor: history, editing keys, completion.
    // These drive the whole editor through `feed` (the real key path),
    // not `CmdlineBuffer` in isolation — that unit coverage lives in
    // `cmdline.rs`. Here we prove the wiring: which ring a prompt walks,
    // that `<Tab>` reaches the registry, that Enter records history.
    // -----------------------------------------------------------------

    #[test]
    fn command_line_history_recalls_previous_ex_commands_newest_first() {
        let mut ed = editor_with("hello");
        feed(&mut ed, ":noh<CR>");
        feed(&mut ed, ":ls<CR>");
        feed(&mut ed, ":");
        assert_eq!(ed.command_line(), Some(""), "fresh prompt");
        feed(&mut ed, "<Up>");
        assert_eq!(ed.command_line(), Some("ls"), "newest first");
        feed(&mut ed, "<Up>");
        assert_eq!(ed.command_line(), Some("noh"));
        feed(&mut ed, "<Up>"); // oldest already -> stays
        assert_eq!(ed.command_line(), Some("noh"));
        feed(&mut ed, "<Down>");
        assert_eq!(ed.command_line(), Some("ls"));
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn ctrl_p_and_ctrl_n_walk_history_like_the_arrows() {
        let mut ed = editor_with("hello");
        feed(&mut ed, ":noh<CR>");
        feed(&mut ed, ":<C-p>");
        assert_eq!(ed.command_line(), Some("noh"));
        feed(&mut ed, "<C-n>"); // back to the empty draft
        assert_eq!(ed.command_line(), Some(""));
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn ex_and_search_histories_stay_separate() {
        let mut ed = editor_with("foo bar");
        feed(&mut ed, ":noh<CR>");
        feed(&mut ed, "/bar<CR>"); // records "bar" in the SEARCH ring only
        // The `:` prompt recalls the ex command, never the search.
        feed(&mut ed, ":<Up>");
        assert_eq!(ed.command_line(), Some("noh"));
        feed(&mut ed, "<Esc>");
        // The `/` prompt recalls the search, never the ex command.
        feed(&mut ed, "/<Up>");
        assert_eq!(ed.command_line(), Some("bar"));
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn command_line_cursor_moves_and_inserts_mid_line() {
        let mut ed = editor_with("x");
        feed(&mut ed, ":abc");
        assert_eq!((ed.command_line(), ed.command_cursor()), (Some("abc"), Some(3)));
        feed(&mut ed, "<Left><Left>");
        assert_eq!(ed.command_cursor(), Some(1));
        feed(&mut ed, "Z");
        assert_eq!(ed.command_line(), Some("aZbc"));
        feed(&mut ed, "<Home>");
        assert_eq!(ed.command_cursor(), Some(0));
        feed(&mut ed, "<End>");
        assert_eq!(ed.command_cursor(), Some(4));
        feed(&mut ed, "<Del>"); // nothing to the right at end -> no-op
        assert_eq!(ed.command_line(), Some("aZbc"));
        feed(&mut ed, "<Home><Del>"); // delete the 'a'
        assert_eq!(ed.command_line(), Some("Zbc"));
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn ctrl_w_and_ctrl_u_delete_word_and_to_start() {
        let mut ed = editor_with("x");
        feed(&mut ed, ":edit foo");
        feed(&mut ed, "<C-w>");
        assert_eq!(ed.command_line(), Some("edit "));
        feed(&mut ed, "<C-u>");
        assert_eq!(ed.command_line(), Some(""));
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn backspace_on_an_empty_command_line_cancels_it() {
        let mut ed = editor_with("x");
        feed(&mut ed, ":");
        assert_eq!(ed.mode(), Mode::Command);
        feed(&mut ed, "<BS>");
        assert_eq!(ed.mode(), Mode::Normal, "backspace on an empty prompt leaves command mode");
        assert_eq!(ed.command_line(), None);
    }

    #[test]
    fn ctrl_r_inserts_a_register_into_the_command_line() {
        let mut ed = editor_with("hello world\n");
        feed(&mut ed, "yw"); // yank "hello " into the unnamed register
        feed(&mut ed, ":");
        feed(&mut ed, "<C-r>\"");
        assert!(ed.command_line().unwrap().contains("hello"), "got {:?}", ed.command_line());
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn tab_completes_and_cycles_ex_command_names() {
        let mut ed = editor_with("x");
        feed(&mut ed, ":w<Tab>");
        assert_eq!(ed.command_line(), Some("w"), "first candidate (sorted) is the bare name");
        assert!(ed.command_completions().is_some(), "the wildmenu list is exposed");
        feed(&mut ed, "<Tab>");
        assert_eq!(ed.command_line(), Some("wa"));
        // <S-Tab> is Tab+shift (the editor has no distinct BackTab code — the
        // UI maps it this way), so build it directly rather than via key::parse,
        // which has no notation for it. It cycles backward, wrapping to "w".
        let shift_tab = Key::new(KeyCode::Tab, Modifiers { shift: true, ..Default::default() });
        ed.handle_key(shift_tab).unwrap();
        assert_eq!(ed.command_line(), Some("w"));
        feed(&mut ed, "x"); // a keystroke ends the cycle
        assert!(ed.command_completions().is_none());
        feed(&mut ed, "<Esc>");
    }

    #[test]
    fn tab_completes_a_buffer_name_for_colon_b() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alpha.txt");
        std::fs::write(&path, "hi").unwrap();
        let mut ed = Editor::new();
        ed.open(&path).unwrap();
        feed(&mut ed, ":b al<Tab>");
        assert_eq!(ed.command_line(), Some("b alpha.txt"));
        feed(&mut ed, "<CR>"); // resolving the name goes to that buffer, no error
    }

    #[test]
    fn tab_is_inert_on_the_search_prompt() {
        let mut ed = editor_with("foo");
        feed(&mut ed, "/ba<Tab>");
        assert_eq!(ed.command_line(), Some("ba"), "search has nothing to complete against");
        feed(&mut ed, "<Esc>");
    }

    // -----------------------------------------------------------------
    // Keymaps: every entry in the maintainer's default config must
    // resolve to its configured Action.
    // -----------------------------------------------------------------

    #[test]
    fn which_key_lists_leader_continuations_after_pressing_space() {
        // The maintainer's default config: pressing the leader (space) should
        // surface every next key under it, with groups for `g`/`r` (which lead
        // to `gd`/`gr`/`rn`) and leaves for `e`/`b`/`q`.
        let mut ed = Editor::new();
        assert!(ed.which_key().is_empty(), "nothing pending yet");
        ed.handle_key(Key::char(' ')).unwrap();

        let rows = ed.which_key();
        assert!(!rows.is_empty(), "leader must raise which-key rows");
        let by_key = |k: &str| rows.iter().find(|r| r.keys == k).cloned();

        let e = by_key("e").expect("`e` continuation present");
        assert_eq!(e.desc, "Toggle file explorer");
        assert!(!e.is_group);

        let g = by_key("g").expect("`g` continuation present");
        assert!(g.is_group, "`g` leads to gd/gr, so it is a group");
    }

    #[test]
    fn which_key_clears_once_the_mapping_completes() {
        let mut ed = Editor::new();
        ed.handle_key(Key::char(' ')).unwrap();
        assert!(!ed.which_key().is_empty());
        ed.handle_key(Key::char('e')).unwrap(); // completes <leader>e
        assert!(ed.which_key().is_empty(), "a resolved mapping leaves nothing pending");
    }

    #[test]
    fn every_default_keymap_resolves_to_its_action() {
        let config = crate::config::Config::default();
        for km in &config.keymaps {
            if !(km.mode.is_empty() || km.mode == "n") {
                continue;
            }
            let mut ed = Editor::with_config(crate::config::Config::default());
            let substituted = km.lhs.replace("<leader>", &config.leader.to_string());
            let keys = key::parse(&substituted);
            assert!(!keys.is_empty(), "keymap {:?} parsed to no keys", km.lhs);
            let mut last = EditorResponse::Continue;
            for k in keys {
                last = ed.handle_key(k).unwrap_or_else(|e| panic!("keymap {:?} errored: {e}", km.lhs));
            }
            assert_eq!(last, EditorResponse::Action(km.action.clone()), "keymap {:?} did not resolve to its action", km.lhs);
        }
    }

    #[test]
    fn gg_motion_falls_back_correctly_after_starting_as_a_keymap_prefix() {
        // "g" is a viable prefix of the "ga" keymap, so bare "gg" must fall
        // through the keymap layer's replay path (see `match_keymap`'s
        // docs) and still resolve as the `gg` motion, not silently vanish
        // into the keymap buffer.
        assert_eq!(run("a\nb\nc", "jjggx"), "\nb\nc");
    }

    // -----------------------------------------------------------------
    // A few extras beyond the required list: `;`/`,` repeat, counts
    // combined with `p`, visual-line text-object promotion.
    // -----------------------------------------------------------------

    #[test]
    fn semicolon_repeats_the_last_find_and_comma_reverses_it() {
        // Seeded via `$` + bare `F` rather than bare `f` — `f` alone is
        // shadowed by the maintainer's hop-to-word keymap whenever `Pending`
        // is idle (see the comment above the `ci*`/`ca*` tests); `F` is not
        // remapped, and `;`/`,` bookkeeping doesn't care which direction
        // seeded it.
        let mut ed = editor_with("a,b,c,d");
        feed(&mut ed, "$"); // -> end of line, on 'd' (col 6)
        feed(&mut ed, "F,"); // -> nearest comma searching backward (col 5)
        assert_eq!(ed.cursor(), Position::new(0, 5));
        feed(&mut ed, ";"); // repeat backward: the next comma further back (col 3)
        assert_eq!(ed.cursor(), Position::new(0, 3));
        feed(&mut ed, ","); // reversed: forward again, back to col 5
        assert_eq!(ed.cursor(), Position::new(0, 5));
    }

    #[test]
    fn a_count_on_put_repeats_the_register_contents() {
        assert_eq!(run("foo\nbar", "yyj2p"), "foo\nbar\nfoo\nfoo");
    }

    // -----------------------------------------------------------------
    // The three bugs the maintainer found by USING the editor.
    //
    // All three survived a 305-test suite, and the reason is worth stating:
    // every existing test asserted editor STATE. None asserted what the user
    // could SEE, and none noticed a keybinding that was simply absent. A test
    // cannot catch a missing feature by exercising the features that exist.
    // -----------------------------------------------------------------

    #[test]
    fn the_command_line_is_visible_to_the_ui_while_being_typed() {
        // Typing `:Neotree` showed NOTHING on screen. The editor accumulated
        // the text correctly the whole time -- the UI just had no way to ask
        // for it, so it painted an empty prompt. This accessor is that way.
        let mut ed = editor_with("hello");
        assert_eq!(ed.command_line(), None, "not in command mode yet");

        feed(&mut ed, ":");
        assert_eq!(ed.mode(), Mode::Command);
        assert_eq!(ed.command_line(), Some(""), "prompt open, nothing typed");

        feed(&mut ed, "Neotree");
        assert_eq!(ed.command_line(), Some("Neotree"), "THIS is what the UI must render");

        // Backspace is reflected too -- a stale echo is its own bug.
        feed(&mut ed, "<BS>");
        assert_eq!(ed.command_line(), Some("Neotre"));

        // And it disappears when the prompt closes.
        feed(&mut ed, "<Esc>");
        assert_eq!(ed.command_line(), None);
    }

    #[test]
    fn the_visual_selection_is_visible_to_the_ui() {
        // Visual mode selected text and highlighted none of it, for exactly the
        // same reason: the editor knew, the UI could not ask.
        let mut ed = editor_with("hello world");
        assert_eq!(ed.selection(), None, "not in visual mode");

        feed(&mut ed, "v");
        assert_eq!(ed.selection(), Some((Position::new(0, 0), Position::new(0, 0))));

        feed(&mut ed, "ll");
        assert_eq!(
            ed.selection(),
            Some((Position::new(0, 0), Position::new(0, 2))),
            "anchor stays put, cursor moves"
        );

        feed(&mut ed, "<Esc>");
        assert_eq!(ed.selection(), None);
    }

    #[test]
    fn a_backwards_visual_selection_is_normalised_into_document_order() {
        // Select leftwards: the cursor is now BEFORE the anchor. A renderer
        // handed (anchor, cursor) unnormalised would paint a backwards range,
        // i.e. nothing at all.
        let mut ed = editor_with("hello world");
        feed(&mut ed, "$"); // end of line
        feed(&mut ed, "v");
        feed(&mut ed, "hhh"); // move left, behind the anchor

        let (start, end) = ed.selection().expect("in visual mode");
        assert!(start <= end, "selection must come back in document order, got {start:?}..{end:?}");
    }

    #[test]
    fn ctrl_d_and_ctrl_u_scroll_by_half_a_viewport() {
        // These were not implemented AT ALL -- no half-page scroll existed
        // anywhere in the editor, so both keys silently did nothing.
        let text = (0..100).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let mut ed = editor_with(&text);
        ed.set_viewport_height(20); // so half is exactly 10

        feed(&mut ed, "<C-d>");
        assert_eq!(ed.cursor().line, 10, "Ctrl+D moves down half a viewport");

        feed(&mut ed, "<C-d>");
        assert_eq!(ed.cursor().line, 20);

        feed(&mut ed, "<C-u>");
        assert_eq!(ed.cursor().line, 10, "Ctrl+U moves back up half a viewport");

        feed(&mut ed, "<C-f>");
        assert_eq!(ed.cursor().line, 30, "Ctrl+F moves a FULL viewport");

        feed(&mut ed, "<C-b>");
        assert_eq!(ed.cursor().line, 10, "Ctrl+B moves back a full viewport");
    }

    #[test]
    fn scrolling_clamps_at_both_ends_of_the_buffer_rather_than_overshooting() {
        let text = (0..5).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let mut ed = editor_with(&text);
        ed.set_viewport_height(20); // half a viewport is bigger than the file

        feed(&mut ed, "<C-d>");
        assert_eq!(ed.cursor().line, 4, "clamps to the last line, does not run off the end");

        feed(&mut ed, "<C-u>");
        assert_eq!(ed.cursor().line, 0, "clamps to the first line");

        // And a viewport nobody set must not make the scroll a silent no-op.
        let mut fresh = editor_with(&text);
        feed(&mut fresh, "<C-d>");
        assert!(fresh.cursor().line > 0, "a default viewport must still scroll");
    }

    #[test]
    fn ctrl_d_is_a_command_not_a_motion() {
        // `d<C-d>` must not delete half a screen. Routing the scroll keys
        // through the operator-pending grammar would have made them motions,
        // which they are not in vi.
        let text = (0..100).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let mut ed = editor_with(&text);
        let before = ed.buffer().line_count();

        feed(&mut ed, "d<C-d>");

        assert_eq!(ed.buffer().line_count(), before, "d<C-d> must not delete anything");
    }

    // -----------------------------------------------------------------
    // Search: `/`, `?`, `n`, `N`, `*`.
    // -----------------------------------------------------------------

    #[test]
    fn forward_search_jumps_to_the_next_match_then_n_repeats() {
        let mut ed = editor_with("foo bar foo baz foo");
        feed(&mut ed, "/foo<CR>");
        assert_eq!(ed.cursor(), Position::new(0, 8), "/foo jumps to the next match");
        feed(&mut ed, "n");
        assert_eq!(ed.cursor(), Position::new(0, 16), "n repeats forward");
        feed(&mut ed, "N");
        assert_eq!(ed.cursor(), Position::new(0, 8), "N repeats backward");
    }

    #[test]
    fn star_searches_the_word_under_the_cursor() {
        let mut ed = editor_with("cat dog cat");
        feed(&mut ed, "*");
        assert_eq!(ed.cursor(), Position::new(0, 8), "* jumps to the next 'cat'");
    }

    #[test]
    fn the_search_prompt_reports_its_kind_for_the_ui() {
        let mut ed = editor_with("abc");
        feed(&mut ed, "/");
        assert_eq!(ed.command_line_kind(), Some(CommandKind::SearchForward));
        feed(&mut ed, "ab");
        assert_eq!(ed.command_line(), Some("ab"), "the search text echoes like an ex command");
        feed(&mut ed, "<Esc>");
        assert_eq!(ed.command_line_kind(), None);
    }

    // -----------------------------------------------------------------
    // Marks and the jumplist.
    // -----------------------------------------------------------------

    #[test]
    fn a_mark_can_be_set_and_jumped_back_to() {
        let mut ed = editor_with("a\nb\nc\nd");
        feed(&mut ed, "jjma"); // cursor on line 2, mark a
        feed(&mut ed, "gg"); // back to the top
        assert_eq!(ed.cursor().line, 0);
        feed(&mut ed, "`a");
        assert_eq!(ed.cursor(), Position::new(2, 0), "`a returns to the mark");
    }

    #[test]
    fn ctrl_o_returns_to_the_position_before_a_jump() {
        let mut ed = editor_with((0..50).map(|i| format!("l{i}")).collect::<Vec<_>>().join("\n").as_str());
        feed(&mut ed, "jj"); // line 2
        feed(&mut ed, "G"); // jump to the end
        assert!(ed.cursor().line > 2);
        feed(&mut ed, "<C-o>");
        assert_eq!(ed.cursor().line, 2, "<C-o> returns to where G left from");
    }

    // -----------------------------------------------------------------
    // Increment / decrement.
    // -----------------------------------------------------------------

    #[test]
    fn ctrl_a_increments_and_carries() {
        assert_eq!(run("9", "<C-a>"), "10");
        assert_eq!(run("99", "<C-a>"), "100");
        assert_eq!(run("10", "<C-x>"), "9");
        assert_eq!(run("-1", "<C-a>"), "0");
    }

    #[test]
    fn ctrl_a_finds_the_number_after_the_cursor() {
        assert_eq!(run("x = 41;", "<C-a>"), "x = 42;");
    }

    #[test]
    fn ctrl_a_is_dot_repeatable() {
        assert_eq!(run("5", "<C-a>."), "7");
    }

    // -----------------------------------------------------------------
    // New motions: ge, gE, g_.
    // -----------------------------------------------------------------

    #[test]
    fn ge_moves_to_the_end_of_the_previous_word() {
        let mut ed = editor_with("foo bar");
        feed(&mut ed, "$ge");
        assert_eq!(ed.cursor(), Position::new(0, 2), "ge lands on the last char of 'foo'");
    }

    #[test]
    fn g_underscore_moves_to_the_last_non_blank() {
        let mut ed = editor_with("  hi  ");
        feed(&mut ed, "g_");
        assert_eq!(ed.cursor(), Position::new(0, 3), "g_ lands on the last non-blank");
    }

    // -----------------------------------------------------------------
    // Visual: gv reselect, o swap ends.
    // -----------------------------------------------------------------

    #[test]
    fn gv_reselects_the_last_visual_selection() {
        let mut ed = editor_with("hello world");
        feed(&mut ed, "vll<Esc>"); // select cols 0..=2, then leave
        assert_eq!(ed.selection(), None);
        feed(&mut ed, "gv");
        assert!(ed.mode().is_visual(), "gv re-enters visual mode");
        assert_eq!(ed.selection(), Some((Position::new(0, 0), Position::new(0, 2))));
    }

    #[test]
    fn o_swaps_the_visual_ends() {
        let mut ed = editor_with("hello");
        feed(&mut ed, "vll"); // anchor 0, cursor 2
        assert_eq!(ed.cursor(), Position::new(0, 2));
        feed(&mut ed, "o");
        assert_eq!(ed.cursor(), Position::new(0, 0), "o moves the cursor to the other end");
    }

    // -----------------------------------------------------------------
    // Insert-mode editing shortcuts.
    // -----------------------------------------------------------------

    #[test]
    fn insert_ctrl_w_deletes_the_word_before_the_cursor() {
        assert_eq!(run("", "ihello world<C-w><Esc>"), "hello ");
    }

    #[test]
    fn insert_ctrl_u_deletes_to_the_line_start() {
        assert_eq!(run("", "ihello<C-u>bye<Esc>"), "bye");
    }

    #[test]
    fn insert_ctrl_r_pastes_a_register() {
        // Yank "foo" into the unnamed register, then paste it in insert mode
        // via `<C-r>"`.
        assert_eq!(run("foo", "yiwA<C-r>\"<Esc>"), "foofoo");
    }

    // -----------------------------------------------------------------
    // zz emits a viewport request rather than editing.
    // -----------------------------------------------------------------

    #[test]
    fn zz_asks_the_ui_to_recentre_without_touching_the_buffer() {
        let mut ed = editor_with("a\nb\nc");
        let before = ed.buffer().text();
        let mut resp = EditorResponse::Continue;
        for k in key::parse("zz") {
            resp = ed.handle_key(k).unwrap();
        }
        assert_eq!(resp, EditorResponse::Scroll(crate::core::ViewportScroll::CenterCursor));
        assert_eq!(ed.buffer().text(), before, "zz must not change the buffer");
    }

    // -----------------------------------------------------------------
    // cj0.41: one-key reflexes C / D / Y / S and the ZZ / ZQ quit pair.
    // -----------------------------------------------------------------

    /// Feeds `keys` and returns the response of the *last* keystroke — the
    /// harness for the commands whose observable effect is a response variant
    /// (quit, message, LSP action) rather than a buffer edit.
    fn feed_last(ed: &mut Editor, keys: &str) -> EditorResponse {
        let mut resp = EditorResponse::Continue;
        for k in key::parse(keys) {
            resp = ed.handle_key(k).unwrap();
        }
        resp
    }

    #[test]
    fn big_d_deletes_to_end_of_line_like_d_dollar() {
        assert_eq!(run("hello world", "5lD"), "hello");
        assert_eq!(run("hello world", "D"), "");
    }

    #[test]
    fn inclusive_to_eol_motions_do_not_swallow_the_newline() {
        // `D`/`d$` on a non-final line must leave the line break intact
        // (leaving an empty line here), not pull the next line up — the EOL
        // fix in `operator::charwise_range`.
        assert_eq!(run("foo\nbar", "D"), "\nbar");
        assert_eq!(run("foo\nbar", "d$"), "\nbar");
        // Cross-line inclusive motions (`%`) are untouched by the fix: the
        // matching `)` is mid-line, so the range still extends onto it.
        assert_eq!(run("(a\nb)c", "d%"), "c");
    }

    #[test]
    fn big_c_changes_to_end_of_line_and_enters_insert() {
        assert_eq!(run("hello world", "5lCthere<Esc>"), "hellothere");
    }

    #[test]
    fn big_y_yanks_to_end_of_line_neovim_default_not_the_whole_line() {
        // Neovim's `Y` is `y$`, not `yy`: it grabs the line's text charwise,
        // so `P` pastes it inline before the cursor rather than opening a new
        // line. A linewise `yy` would instead have duplicated the whole line
        // (two lines). `Y` must behave exactly like `y$`.
        assert_eq!(run("foo", "YP"), "foofoo");
        assert_eq!(run("foo", "YP"), run("foo", "y$P"), "Y must equal y$");
    }

    #[test]
    fn big_s_substitutes_the_whole_line_like_cc() {
        // `S` is an exact alias of `cc` — same operator, same code path — so
        // the two must produce identical results whatever `cc` does.
        assert_eq!(run("foo\nbar\nbaz", "jShi<Esc>"), run("foo\nbar\nbaz", "jcchi<Esc>"));
    }

    #[test]
    fn zz_writes_then_quits_and_zq_quits_unconditionally() {
        let mut ed = editor_with("some text");
        assert_eq!(feed_last(&mut ed, "ZZ"), EditorResponse::WriteThenQuit { path: None });
        let mut ed = editor_with("dirty");
        feed(&mut ed, "xxx"); // make it modified
        assert!(ed.buffer().is_modified());
        // ZQ must quit even with unsaved changes — no guard, unlike `:q`.
        assert_eq!(feed_last(&mut ed, "ZQ"), EditorResponse::Quit);
    }

    // -----------------------------------------------------------------
    // cj0.41: column / line motions | + - _ <CR>.
    // -----------------------------------------------------------------

    #[test]
    fn bar_goes_to_column_count_one_based() {
        // `3|` -> column 2 (exclusive), so `d3|` removes the first two chars.
        assert_eq!(run("hello", "d3|"), "llo");
        // bare `|` is column 1 == start of line: `d|` from column 1 removes
        // just the first char.
        assert_eq!(run("hello", "ld|"), "ello");
    }

    #[test]
    fn plus_cr_minus_underscore_are_linewise_first_non_blank_motions() {
        // `+` / `<CR>`: this line plus the next, linewise.
        assert_eq!(run("a\nb\nc", "d+"), "c");
        assert_eq!(run("a\nb\nc", "d<CR>"), "c");
        // `-`: this line plus the previous.
        assert_eq!(run("a\nb\nc", "jjd-"), "a");
        // `_`: bare `_` is the current line (`dd`); `2_` reaches one down.
        assert_eq!(run("a\nb", "d_"), "b");
        assert_eq!(run("a\nb\nc", "d2_"), "c");
    }

    // -----------------------------------------------------------------
    // cj0.41: & repeats the last :s, <C-g> echoes file info,
    // <C-]> routes to LSP go-to-definition.
    // -----------------------------------------------------------------

    #[test]
    fn ampersand_repeats_the_last_substitution_on_the_current_line() {
        let mut ed = editor_with("foo\nfoo\nfoo");
        ed.execute_ex("s/foo/bar/").unwrap(); // line 0 -> bar
        feed(&mut ed, "j&"); // repeat on line 1
        assert_eq!(ed.buffer().text(), "bar\nbar\nfoo");
    }

    #[test]
    fn ampersand_without_a_prior_substitution_is_a_friendly_no_op() {
        let mut ed = editor_with("foo");
        assert_eq!(feed_last(&mut ed, "&"), EditorResponse::Message("no previous substitute".to_string()));
        assert_eq!(ed.buffer().text(), "foo");
    }

    #[test]
    fn ctrl_g_echoes_file_info_without_editing() {
        let mut ed = editor_with("a\nb\nc");
        let EditorResponse::Message(msg) = feed_last(&mut ed, "<C-g>") else {
            panic!("<C-g> must report a message");
        };
        assert!(msg.contains("[No Name]"), "unsaved buffer reports [No Name]: {msg}");
        assert!(msg.contains("line(s)"), "message reports a line count: {msg}");
    }

    #[test]
    fn ctrl_bracket_routes_to_lsp_go_to_definition() {
        let mut ed = editor_with("fn main() {}");
        assert_eq!(feed_last(&mut ed, "<C-]>"), EditorResponse::Action(crate::config::Action::LspDefinition));
    }

    // -----------------------------------------------------------------
    // cj0.35: bracket [ ] motion family.
    // -----------------------------------------------------------------

    #[test]
    fn unmatched_brace_forward_and_back_skip_balanced_pairs() {
        // Cursor inside the outer braces; `]}` lands on the outer close,
        // stepping over the inner balanced pair, and `[{` on the outer open.
        let mut ed = editor_with("{\n  { inner }\n  body\n}");
        feed(&mut ed, "jj]}"); // from "  body" to the final "}"
        assert_eq!(ed.cursor(), Position::new(3, 0));
        feed(&mut ed, "[{"); // back to the opening "{"
        assert_eq!(ed.cursor(), Position::new(0, 0));
    }

    #[test]
    fn unmatched_paren_motions_count_nesting_on_one_line() {
        // "(a(b)c)": from 'b' (index 3), `])` reaches the inner ')' at 4,
        // `[(` the enclosing '(' at 2.
        let mut ed = editor_with("(a(b)c)");
        feed(&mut ed, "3l])");
        assert_eq!(ed.cursor(), Position::new(0, 4));
        // Fresh cursor on 'b' again for the backward direction.
        let mut ed = editor_with("(a(b)c)");
        feed(&mut ed, "3l[(");
        assert_eq!(ed.cursor(), Position::new(0, 2));
    }

    #[test]
    fn section_motions_jump_between_braces_in_column_zero() {
        let mut ed = editor_with("{\n body\n{\n more\n}");
        feed(&mut ed, "j]]"); // from " body" to the next col-0 '{'
        assert_eq!(ed.cursor(), Position::new(2, 0));
        feed(&mut ed, "[["); // back to the first col-0 '{'
        assert_eq!(ed.cursor(), Position::new(0, 0));
    }

    #[test]
    fn bracket_motions_compose_with_an_operator() {
        // `d]}` deletes up to but not including the unmatched close brace
        // (the motion is charwise-exclusive, like neovim's).
        assert_eq!(run("(abc)", "l])"), "(abc)"); // sanity: motion alone doesn't edit
        assert_eq!(run("{ab}", "ld]}"), "{}");
    }

    #[test]
    fn method_motions_land_on_the_nearest_brace() {
        let mut ed = editor_with("fn a() {\n}\nfn b() {\n}");
        feed(&mut ed, "]m"); // next '{'
        assert_eq!(ed.cursor(), Position::new(0, 7));
        feed(&mut ed, "]M"); // next '}'
        assert_eq!(ed.cursor(), Position::new(1, 0));
    }

    #[test]
    fn bracket_mark_motions_jump_to_the_next_and_previous_lowercase_mark() {
        let mut ed = editor_with("a\nb\nc\nd\ne");
        feed(&mut ed, "jma"); // mark 'a' on line 1
        feed(&mut ed, "jjmb"); // mark 'b' on line 3
        feed(&mut ed, "gg"); // back to the top
        feed(&mut ed, "]'"); // next mark's line -> line 1
        assert_eq!(ed.cursor().line, 1);
        feed(&mut ed, "]'"); // next mark's line -> line 3
        assert_eq!(ed.cursor().line, 3);
        feed(&mut ed, "['"); // previous mark's line -> line 1
        assert_eq!(ed.cursor().line, 1);
    }

    // -----------------------------------------------------------------
    // cj0.41: <C-^> edits the alternate file.
    // -----------------------------------------------------------------

    #[test]
    fn ctrl_caret_toggles_between_the_two_most_recent_buffers() {
        let mut ed = editor_with("first");
        let first = ed.buffer_id();
        let second = ed.new_buffer();
        ed.buffer_mut().apply(Edit::insert(Position::ORIGIN, "second".to_string())).unwrap();
        assert_eq!(ed.buffer_id(), second);
        feed_last(&mut ed, "<C-^>"); // back to the alternate (first)
        assert_eq!(ed.buffer_id(), first);
        feed_last(&mut ed, "<C-6>"); // and forward again (same key, other byte)
        assert_eq!(ed.buffer_id(), second);
    }
}
