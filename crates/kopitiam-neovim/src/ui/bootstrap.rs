//! The [`run`] entry point `main.rs` calls, and the small amount of glue
//! needed to actually start an [`App`].
//!
//! # `impl BufferView for text::Buffer` lives here, permanently
//!
//! [`crate::text::Buffer`]'s read API is the "frozen" contract this UI is
//! allowed to depend on (`line_count`, `line`, `line_len`, `is_modified`,
//! `path`), and it is *exactly* [`BufferView`]'s shape. This impl is not a
//! stopgap — `editor::Editor::buffer()` is specified to return `&text::Buffer`
//! (see the architecture note this crate was built against), so once
//! `editor::Editor` exists, its own `EditorHost::Buffer` associated type is
//! simply `text::Buffer`, and it reuses this same impl.
//!
//! # `PlaceholderHost`, unlike the impl above, is temporary
//!
//! `crate::editor::Editor` — the real modal state machine — did not exist
//! yet when this file was written (see `ui/mod.rs` and `ui/event.rs` for the
//! full explanation of the seam this crate builds against). [`main`][crate]
//! still needs `kvim` to open a file and put something on screen, so
//! [`PlaceholderHost`] is the smallest possible [`EditorHost`] that makes
//! that true: it moves a cursor around a buffer and quits on `q`. It
//! deliberately implements **no modes, no operators, no registers, no ex
//! commands** — adding any of those here would be exactly the "business
//! logic inside the UI" `CLAUDE.md` prohibits. The moment `editor::Editor`
//! exists, this struct should be deleted and [`run`] should construct
//! `App::new(editor::Editor::open(files)?, ...)` instead — a one-function
//! change, which is the entire point of building against [`EditorHost`]
//! rather than a concrete type.

use std::io;
use std::path::{Path, PathBuf};

use ratatui::{backend::CrosstermBackend, Terminal};

use crate::config::Config;
use crate::core::{Mode, Position};
use crate::editor::{EditorResponse, Key as EditorKey};
use crate::text::Buffer;
use crate::ui::app::App;
use crate::ui::event::{BufferView, EditorHost, HostResponse, Key, KeyPress};
use crate::ui::terminal::TerminalGuard;
use crate::ui::theme::Theme;

/// Performs the write that [`EditorResponse::Write`] only *asks* for.
///
/// The editor deliberately returns the intent rather than touching the disk
/// itself (see `editor::ex`'s docs) — so that `:w` is testable without a
/// filesystem, and so the decision of *whether* to write stays with the caller.
/// This is that caller, and this is where the I/O actually happens.
fn write_buffer(editor: &mut crate::editor::Editor, path: Option<&Path>) -> crate::Result<String> {
    let buffer = editor.buffer_mut();
    match path {
        Some(path) => {
            buffer.save_as(path)?;
            Ok(format!("\"{}\" written", path.display()))
        }
        None => {
            buffer.save()?;
            let name = buffer.path().map(|p| p.display().to_string()).unwrap_or_else(|| "[No Name]".to_string());
            Ok(format!("\"{name}\" written"))
        }
    }
}

/// Performs the write-all that [`EditorResponse::WriteAll`] asks for: saves
/// every modified buffer that has a backing file, and reports how many were
/// written.
///
/// Returns `Err` if a modified buffer has no file name — vim's `E32`. That is
/// the one case `:wqa`/`:xa` must not quit through, since quitting would
/// silently drop the unsaved, unnamed buffer; the caller checks the result and
/// declines the quit. Buffers that are unmodified, or unmodified-and-unnamed,
/// are simply skipped (an unnamed *scratch* buffer with no edits is not a
/// reason to refuse a `:wqa`).
fn write_all_buffers(editor: &mut crate::editor::Editor) -> crate::Result<String> {
    let mut written = 0usize;
    let mut unnamed_modified = false;
    for buffer in editor.buffers_mut() {
        if !buffer.is_modified() {
            continue;
        }
        if buffer.path().is_none() {
            unnamed_modified = true;
            continue;
        }
        buffer.save()?;
        written += 1;
    }
    if unnamed_modified {
        // Matches vim's E32: a modified no-name buffer cannot be written by a
        // bare `:wa`, so say so rather than silently leaving it unsaved.
        return Err(crate::Error::Io(std::io::Error::other("E32: a modified buffer has no file name")));
    }
    Ok(format!("{written} buffer(s) written"))
}

impl BufferView for Buffer {
    fn line_count(&self) -> usize {
        Buffer::line_count(self)
    }

    fn line(&self, n: usize) -> Option<String> {
        Buffer::line(self, n)
    }

    fn line_len(&self, n: usize) -> usize {
        Buffer::line_len(self, n)
    }

    fn is_modified(&self) -> bool {
        Buffer::is_modified(self)
    }

    fn path(&self) -> Option<&Path> {
        Buffer::path(self)
    }
}

/// Joins the real modal engine to the UI's [`EditorHost`] seam.
///
/// The two halves of kvim were built against this trait precisely so they could
/// be written independently and meet here. The only real work is translating
/// key types: the UI speaks [`KeyPress`] (its own, so it can be tested with no
/// terminal) and the editor speaks [`crate::editor::Key`] (its own, so it can be
/// tested with no UI). Neither depends on the other, and neither depends on
/// crossterm — which is why this adapter is the *only* place the mapping lives.
impl EditorHost for crate::editor::Editor {
    type Buffer = Buffer;

    fn handle_key(&mut self, key: KeyPress) -> HostResponse {
        let Some(editor_key) = to_editor_key(key) else {
            return HostResponse::Unchanged;
        };

        match crate::editor::Editor::handle_key(self, editor_key) {
            Ok(response) => match response {
                EditorResponse::Continue => HostResponse::Changed,
                // `:q` closes the active *window* and only quits the editor on
                // the last one — a distinction only the UI can make, since the
                // editor has no window tree. The unsaved-changes check has
                // already run inside `execute_ex`, so reaching here means the
                // close is allowed.
                EditorResponse::Quit => HostResponse::QuitWindow,
                // `:qa`/`:qa!` exits the whole editor unconditionally — unlike
                // `:q`, it does not close one window and stay. `HostResponse::Quit`
                // already means "quit the editor" to `App`, so it is reused
                // rather than inventing a parallel variant.
                EditorResponse::QuitAll => HostResponse::Quit,
                EditorResponse::Message(msg) => HostResponse::Message(msg),

                // `:w` and `:wq` deliberately do not write from inside the
                // editor — it returns the intent and lets the caller perform the
                // I/O (see `editor::ex`'s docs). This is that caller.
                EditorResponse::Write { path } => match write_buffer(self, path.as_deref()) {
                    Ok(msg) => HostResponse::Message(msg),
                    Err(e) => HostResponse::Error(e.to_string()),
                },
                EditorResponse::WriteThenQuit { path } => match write_buffer(self, path.as_deref()) {
                    Ok(_) => HostResponse::QuitWindow,
                    Err(e) => HostResponse::Error(e.to_string()),
                },
                // `:wa` writes all and stays; `:wqa`/`:xa` write all and exit the
                // whole editor. A write failure (e.g. E32: a modified no-name
                // buffer) aborts the quit, so nothing unsaved is lost.
                EditorResponse::WriteAll { then_quit } => match write_all_buffers(self) {
                    Ok(msg) => {
                        if then_quit {
                            HostResponse::Quit
                        } else {
                            HostResponse::Message(msg)
                        }
                    }
                    Err(e) => HostResponse::Error(e.to_string()),
                },

                // Window and viewport commands the editor recognised but the UI
                // must carry out (it owns the window tree and the scroll
                // offsets — see `EditorResponse::Window`/`Scroll`).
                EditorResponse::Window(cmd) => HostResponse::Window(cmd),
                EditorResponse::Scroll(req) => HostResponse::Scroll(req),

                // A keymap resolved to one of the maintainer's configured
                // actions (`<leader>e` → file tree, `f` → hop, `\ff` → find
                // files, ...). The editor cannot perform these itself — it must
                // not depend on `plugins` or `ui` (see `EditorResponse::Action`)
                // — so they are forwarded one more hop, to `ui::app::App`, which
                // is the layer that owns overlays and focus. `App::handle_action`
                // decides what each one does, and still answers honestly for the
                // ones with no UI yet.
                EditorResponse::Action(action) => HostResponse::Action(action),

                // The quickfix / location-list family, parsed by the editor and
                // performed by `App` (see `EditorResponse::Quickfix`).
                EditorResponse::Quickfix(cmd) => HostResponse::Quickfix(cmd),
            },
            Err(e) => HostResponse::Error(e.to_string()),
        }
    }

    fn mode(&self) -> Mode {
        crate::editor::Editor::mode(self)
    }

    fn cursor(&self) -> Position {
        crate::editor::Editor::cursor(self)
    }

    fn buffer(&self) -> &Buffer {
        crate::editor::Editor::buffer(self)
    }

    fn open(&mut self, path: &Path) -> Result<(), String> {
        crate::editor::Editor::open(self, path).map(|_| ()).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn replace_range(&mut self, range: crate::core::Range, text: &str) -> Position {
        crate::editor::Editor::replace_range(self, range, text)
    }

    fn move_cursor(&mut self, pos: Position) {
        crate::editor::Editor::move_cursor(self, pos);
    }

    fn run_ex(&mut self, line: &str) -> Result<(), String> {
        // `:cdo` only cares that the buffer edit (its `:s`) lands; window/quit
        // effects are not meaningful mid-iteration, so they are dropped here.
        crate::editor::Editor::execute_ex(self, line).map(|_| ()).map_err(|e| e.to_string())
    }

    fn save(&mut self) -> Result<(), String> {
        write_buffer(self, None).map(|_| ()).map_err(|e| e.to_string())
    }

    // The two accessors below are the whole fix for "`:` commands were invisible
    // while you typed them" and "visual mode highlighted nothing". The editor had
    // both pieces of state all along; this seam simply had no way to ask for
    // them. Delegation, nothing more — which is exactly how much code the bug was
    // worth, and exactly why it was so easy to leave out.

    fn command_line(&self) -> Option<&str> {
        crate::editor::Editor::command_line(self)
    }

    fn command_cursor(&self) -> Option<usize> {
        crate::editor::Editor::command_cursor(self)
    }

    fn command_completions(&self) -> Option<(Vec<String>, usize)> {
        crate::editor::Editor::command_completions(self).map(|(items, sel)| (items.to_vec(), sel))
    }

    fn selection(&self) -> Option<(Position, Position)> {
        crate::editor::Editor::selection(self)
    }

    fn search_highlight(&self) -> Option<regex::Regex> {
        crate::editor::Editor::search_highlight(self)
    }

    /// Maps the editor's [`crate::editor::WhichKeyItem`]s to the UI's
    /// [`crate::ui::whichkey::WhichKeyRow`] — the one place these two
    /// vocabularies meet, kept in this adapter so neither `editor` nor the
    /// `whichkey` widget depends on the other's type.
    fn which_key(&self) -> Vec<crate::ui::whichkey::WhichKeyRow> {
        crate::editor::Editor::which_key(self)
            .into_iter()
            .map(|item| crate::ui::whichkey::WhichKeyRow {
                keys: item.keys,
                desc: item.desc,
                is_group: item.is_group,
            })
            .collect()
    }

    fn active_buffer_id(&self) -> crate::core::BufferId {
        crate::editor::Editor::buffer_id(self)
    }

    fn buffer_by_id(&self, id: crate::core::BufferId) -> Option<&Buffer> {
        crate::editor::Editor::buffer_by_id(self, id)
    }

    fn collapsed_folds(&self, id: crate::core::BufferId) -> Vec<(usize, usize)> {
        crate::editor::Editor::collapsed_folds_for(self, id)
    }

    fn set_active(&mut self, buffer: crate::core::BufferId, cursor: Position) {
        crate::editor::Editor::set_active(self, buffer, cursor);
    }

    fn buffers(&self) -> Vec<crate::ui::event::BufferEntry> {
        crate::editor::Editor::buffer_entries(self)
            .into_iter()
            .map(|(id, name, modified)| crate::ui::event::BufferEntry { id, name, modified })
            .collect()
    }

    fn focus_buffer(&mut self, id: crate::core::BufferId) {
        crate::editor::Editor::focus_buffer(self, id);
    }

    fn new_buffer(&mut self) -> crate::core::BufferId {
        crate::editor::Editor::new_buffer(self)
    }

    fn set_viewport_height(&mut self, lines: usize) {
        crate::editor::Editor::set_viewport_height(self, lines);
    }

    /// Translates the editor's own [`crate::editor::CommandKind`] into the
    /// UI's [`PromptKind`] — the one place these two vocabularies meet, kept
    /// here (in the adapter) so neither `editor` nor the `ui` widgets depend on
    /// the other's enum.
    fn command_prompt(&self) -> crate::ui::cmdline::PromptKind {
        use crate::editor::CommandKind;
        use crate::ui::cmdline::PromptKind;
        match crate::editor::Editor::command_line_kind(self) {
            Some(CommandKind::Ex) => PromptKind::Command,
            Some(CommandKind::SearchForward) => PromptKind::SearchForward,
            Some(CommandKind::SearchBackward) => PromptKind::SearchBackward,
            None => PromptKind::None,
        }
    }
}

/// Translates a UI [`KeyPress`] into the editor's own key type.
///
/// Returns `None` for keys the editor has no code for, which the caller treats
/// as "nothing changed" rather than as an error — an unmapped key in vi is a
/// no-op, not a failure.
fn to_editor_key(key: KeyPress) -> Option<EditorKey> {
    use crate::editor::key::KeyCode as E;

    let code = match key.key {
        Key::Char(c) => E::Char(c),
        Key::Enter => E::Enter,
        Key::Escape => E::Esc,
        Key::Backspace => E::Backspace,
        Key::Tab => E::Tab,
        Key::Delete => E::Delete,
        Key::Up => E::Up,
        Key::Down => E::Down,
        Key::Left => E::Left,
        Key::Right => E::Right,
        Key::Home => E::Home,
        Key::End => E::End,
        Key::PageUp => E::PageUp,
        Key::PageDown => E::PageDown,
        Key::F(n) => E::F(n),
        // The editor models Shift-Tab as plain Tab plus the shift modifier
        // rather than as a distinct code, so map it that way instead of
        // dropping it.
        Key::BackTab => E::Tab,
        Key::Insert => return None,
    };

    let shift = key.mods.shift || matches!(key.key, Key::BackTab);
    Some(EditorKey {
        code,
        mods: crate::editor::key::Modifiers { ctrl: key.mods.ctrl, alt: key.mods.alt, shift },
    })
}

/// See the module docs: a temporary stand-in for `editor::Editor`, kept
/// intentionally incapable of anything beyond "look at a file and quit".
///
/// Retained only because its tests pin the [`EditorHost`] seam itself,
/// independently of the real engine. The binary no longer uses it — see
/// [`run`].
#[allow(dead_code)]
struct PlaceholderHost {
    buffer: Buffer,
    cursor: Position,
}

impl EditorHost for PlaceholderHost {
    type Buffer = Buffer;

    fn handle_key(&mut self, key: KeyPress) -> HostResponse {
        let mut target = self.cursor;
        match key.key {
            Key::Char('q') => return HostResponse::Quit,
            Key::Char('h') | Key::Left => target.col = target.col.saturating_sub(1),
            Key::Char('l') | Key::Right => target.col = target.col.saturating_add(1),
            Key::Char('k') | Key::Up => target.line = target.line.saturating_sub(1),
            Key::Char('j') | Key::Down => target.line = target.line.saturating_add(1),
            _ => return HostResponse::Unchanged,
        }
        // Bounds-checking a cursor move is the buffer's own job (it already
        // knows its line count and each line's grapheme length) — delegated
        // via `Buffer::clamp` rather than re-implemented here, which is
        // exactly the kind of motion *policy* (does moving past the buffer
        // end wrap, clamp, or error?) that belongs to whichever module owns
        // "what a valid cursor position is," not to this placeholder.
        let clamped = self.buffer.clamp(target);
        if clamped == self.cursor {
            return HostResponse::Unchanged;
        }
        self.cursor = clamped;
        HostResponse::Changed
    }

    fn mode(&self) -> Mode {
        // No modal state machine exists yet to report; every real editor
        // key handler downstream of this placeholder should treat "always
        // Normal" as exactly as capable as this host actually is.
        Mode::Normal
    }

    fn cursor(&self) -> Position {
        self.cursor
    }

    fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn open(&mut self, _path: &Path) -> Result<(), String> {
        // Deliberately incapable, like everything else on this placeholder: it
        // has one buffer and no way to acquire a second. See the module docs.
        Err("the placeholder host cannot open files".to_string())
    }
}

/// Opens `files` (or an empty scratch buffer, if none were given) and runs
/// the editor until it quits.
///
/// This is the function `main.rs` hands control to after parsing CLI
/// arguments and loading [`Config`]. It owns the terminal for the process's
/// entire interactive lifetime: entering [`TerminalGuard`] here, rather
/// than in `main`, keeps the raw-mode/alternate-screen invariant scoped to
/// exactly "while `run` is on the stack," which is also exactly "while
/// there is a UI to draw."
pub fn run(config: Config, files: &[PathBuf]) -> anyhow::Result<()> {
    // Execute the discovered `init.lua` / `lua/*.lua` (if any) on top of the
    // loaded config, through the `vim.*` shim. This is where the Lua the
    // maintainer wrote actually takes effect — options, keymaps, leader, theme.
    // A config that binds a Lua *function* to a key hands back a live runtime,
    // kept alive for the session so the closure can fire on the keypress. A
    // broken or unsupported config degrades to warnings, never a failure to
    // start — see `crate::luaconfig`.
    let (config, lua_runtime, startup_note) = apply_lua_config(config);

    let theme = Theme::from_name(&config.theme);
    let options = config.options.clone();
    let leader = config.leader;
    // The resource-aware LSP guard's tuning, cloned out before `config` moves
    // into the editor. See `crate::lsp::resource_guard`.
    let lsp_guard_cfg = config.lsp_guard.clone();

    // Detected once, here, and threaded down: the statusline needs to know
    // whether Powerline separators are safe (a bar full of tofu boxes is worse
    // than one with plain separators) and the file tree needs the same answer for
    // its devicons. `IconSet::detect` defaults to the timid tier when it cannot
    // tell — see `crate::icons`.
    let icons = crate::icons::IconSet::detect();

    let mut editor = crate::editor::Editor::with_config(config);
    for path in files {
        editor.open(path)?;
    }

    let mut app = App::new(editor, options, theme, icons, leader);
    app.set_lsp_guard_config(lsp_guard_cfg);
    if let Some(runtime) = lua_runtime {
        app.set_lua_runtime(runtime);
    }
    if let Some(note) = startup_note {
        app.set_startup_message(note);
    }
    // Once the App exists but before the terminal is taken over: if kvim is
    // running inside a multiplexer, work out whether tmux would eat its
    // `<C-h/j/k/l>` (the vim-tmux-navigator `is_vim` problem) and arm the
    // consent popup if so. Reads the environment and the user's tmux.conf, so
    // it lives here — in `run`, next to the other startup detection — and not
    // in `App::new`, which must stay pure enough to build in a unit test. See
    // `crate::tmux`.
    app.apply_startup_advice(crate::tmux::startup_advice());

    let mut guard = TerminalGuard::new()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    let result = app.run(&mut terminal);
    // Stop any language servers before restoring the terminal, so kvim leaves
    // no orphaned rust-analyzer/texlab processes behind.
    app.shutdown_lsp();
    // Explicit rather than relying solely on `guard`'s `Drop`: restoring
    // before returning means a startup/runtime error prints to a terminal
    // that is back in its normal (non-raw, non-alternate-screen) state,
    // instead of being swallowed by the alternate screen `Drop` is about to
    // leave.
    guard.restore();
    result.map_err(anyhow::Error::from)
}

/// Runs the discovered Lua config on top of `base`, returning the merged
/// config, the live runtime (when the config bound Lua-function keymaps or we
/// want it kept anyway), and an optional one-line startup note summarising
/// anything that did not fully apply.
///
/// Split out of [`run`] so the wiring is unit-testable without a terminal: it
/// reads the real config directory (empty on most machines, in which case the
/// base config passes straight through untouched).
fn apply_lua_config(
    base: Config,
) -> (Config, Option<crate::luaconfig::LuaRuntime>, Option<String>) {
    let discovered = crate::luaconfig::Discovered::from_config_dir();
    if discovered.is_empty() {
        return (base, None, None);
    }

    let runtime = crate::luaconfig::LuaRuntime::load(base, &discovered);
    let config = runtime.config();

    // A startup note only when there is something the user should know: a
    // notification the config raised, or a count of vim.* items kvim could not
    // apply. Silence otherwise — a clean config should not nag.
    let warnings = runtime.warnings();
    let note = runtime
        .notifications()
        .into_iter()
        .next_back()
        .or_else(|| {
            (!warnings.is_empty()).then(|| {
                format!(
                    "kvim: {} item(s) in your Lua config are not supported yet (config still loaded)",
                    warnings.len()
                )
            })
        });

    (config, Some(runtime), note)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_lua_config_is_a_no_op_when_there_is_no_lua() {
        // On a machine with no ~/.kopitiam/kopitiam-neovim Lua files, the base
        // config must pass through byte-for-byte, with no runtime and no note.
        // (If the developer running the suite happens to HAVE such a config, the
        // merged config is still valid — so only assert the no-Lua invariant when
        // the directory is genuinely empty.)
        if crate::luaconfig::Discovered::from_config_dir().is_empty() {
            let base = Config::default();
            let (cfg, rt, note) = apply_lua_config(base.clone());
            assert_eq!(cfg, base);
            assert!(rt.is_none());
            assert!(note.is_none());
        }
    }

    #[test]
    fn placeholder_host_moves_the_cursor_and_clamps_to_the_buffer() {
        let buffer = Buffer::from_str("ab\ncd\n");
        let mut host = PlaceholderHost { buffer, cursor: Position::ORIGIN };

        assert_eq!(host.handle_key(KeyPress::plain(Key::Char('l'))), HostResponse::Changed);
        assert_eq!(host.cursor(), Position::new(0, 1));

        assert_eq!(host.handle_key(KeyPress::plain(Key::Char('j'))), HostResponse::Changed);
        assert_eq!(host.cursor().line, 1);
    }

    #[test]
    fn placeholder_host_reports_quit_on_q() {
        let buffer = Buffer::new();
        let mut host = PlaceholderHost { buffer, cursor: Position::ORIGIN };
        assert_eq!(host.handle_key(KeyPress::plain(Key::Char('q'))), HostResponse::Quit);
    }

    #[test]
    fn placeholder_host_reports_unchanged_at_the_buffer_edge() {
        let buffer = Buffer::from_str("a");
        let mut host = PlaceholderHost { buffer, cursor: Position::ORIGIN };
        // Already at (0, 0); moving left/up must clamp to the same spot.
        assert_eq!(host.handle_key(KeyPress::plain(Key::Char('h'))), HostResponse::Unchanged);
        assert_eq!(host.handle_key(KeyPress::plain(Key::Char('k'))), HostResponse::Unchanged);
    }

    #[test]
    fn text_buffer_implements_bufferview_matching_its_own_api() {
        let buffer = Buffer::from_str("hello\nworld\n");
        assert_eq!(BufferView::line_count(&buffer), 3); // trailing newline -> 3rd empty line.
        assert_eq!(BufferView::line(&buffer, 0).as_deref(), Some("hello"));
        assert!(!BufferView::is_modified(&buffer));
        assert_eq!(BufferView::path(&buffer), None);
    }
}
