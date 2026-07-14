//! The event loop: read a terminal event, translate it, feed the editor,
//! act on the response, redraw.
//!
//! # Redraw discipline
//!
//! `crossterm::event::poll` blocks (with a timeout) until a terminal event
//! actually arrives, so [`App::run`] only ever wakes up because something
//! happened — a keypress or a resize — never on a fixed tick. There is no
//! frame timer and nothing renders "just in case". That matters on the
//! Android target this crate is built for: a terminal UI repainting on a
//! 60fps clock burns battery for a screen that, between keystrokes, has not
//! changed a single cell. The poll timeout itself (see
//! [`App::EVENT_POLL_INTERVAL`]) exists only so the loop can periodically
//! check whether it's been asked to stop — it is not a redraw tick, and no
//! draw happens on a timeout with no event.
//!
//! Within a single keypress, [`event::HostResponse::Unchanged`] short-
//! circuits before reaching [`App::draw`] at all, so an unmapped key in
//! normal mode (which vim itself just beeps at) costs one `handle_key` call
//! and nothing else.
//!
//! # Focus
//!
//! Exactly one thing has focus at a time: the buffer, or an open overlay (see
//! [`crate::ui::overlay`]). [`App::handle_event`] branches on that *before* it
//! looks at the key, so when the file tree has focus the editor does not merely
//! ignore `j` — it never sees it. That is the difference between a sidebar that
//! feels right and one that moves the text cursor behind your back.
//!
//! Focus is stored, not inferred. "Is an overlay open" and "does the overlay
//! have focus" are two different questions: neo-tree stays visible after you
//! open a file from it, at which point it is drawn but inert.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::{self, Event};
use unicode_segmentation::UnicodeSegmentation;
use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction as LayoutDirection, Layout, Rect},
    Frame, Terminal,
};

use crate::config::{Action, Options};
use crate::core::{BufferId, Direction, Mode, Position, Range, ViewportScroll, WindowCommand};
use crate::icons::IconSet;
use crate::lsp::completion::{self, CompletionItem as CItem, CompletionSource};
use crate::lsp::{Location as LspLocation, LspClient};
use crate::ui::completion_menu::{menu_rect, CompletionMenu as CompletionMenuWidget};
use crate::ui::lsp_ui::{centered_rect, InfoBox};
use crate::ui::snippet::SnippetSession;
use crate::ui::cmdline::{Cmdline, CmdlineState, PromptKind, StatusMessage};
use crate::ui::event::{map_crossterm_key, BufferView, EditorHost, HostResponse, Key, KeyPress};
use crate::ui::filetree::FileTreePanel;
use crate::ui::gutter::LineNumberMode;
use crate::ui::hop::{HopFeed, HopState};
use crate::ui::overlay::{Focus, OpenTarget, Overlay, OverlayOutcome};
use crate::ui::scrolling;
use crate::ui::statusline::{Statusline, StatuslineData};
use crate::ui::textarea::{Scroll, Selection, TextArea};
use crate::ui::theme::Theme;
use crate::ui::window::{SplitKind, WindowTree};

use ratatui::style::{Modifier, Style};

/// What the event loop should do after processing one terminal event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    Continue,
    Redraw,
    Quit,
}

/// Ties the terminal, the window tree, the theme, and an [`EditorHost`]
/// together into a running application.
///
/// Generic over `H: EditorHost` rather than a concrete `editor::Editor` for
/// the reason explained in `ui/mod.rs` and `ui/event.rs`: at the time this
/// was written `editor::Editor` didn't exist to be generic over yet. Once it
/// lands, `App<editor::Editor>` (after writing the `impl EditorHost for
/// editor::Editor` adapter) is a fully working application with no changes
/// needed here.
pub struct App<H: EditorHost> {
    pub host: H,
    pub windows: WindowTree,
    pub theme: Theme,
    pub options: Options,
    /// The message shown on the bottom row when no prompt is active (`"written"`,
    /// an error, `"HopWords is not wired into the UI yet"`).
    ///
    /// # This is a message, not a command line
    ///
    /// It used to be a whole [`CmdlineState`] — an app-owned mirror of what the
    /// user was typing at the `:` prompt, which **nothing ever wrote to**. The
    /// result was that `:Neotree` echoed nothing at all: the editor accumulated
    /// the text, the renderer drew this empty mirror, and the two never met.
    ///
    /// The fix is not to feed the mirror. It is to delete it. A second copy of
    /// state that already exists in the editor is a bug waiting for someone to
    /// forget to sync it, and it duly was. The command line is now *derived* from
    /// the host every frame (see [`App::cmdline_state`]), the same way
    /// [`StatuslineData`] always has been, and the only thing the app still owns
    /// is the one piece of state the editor genuinely does not have: the message
    /// *it* was asked to display.
    pub message: StatusMessage,
    /// Which glyph vocabulary the terminal can render. Threaded into the
    /// statusline (which needs `glyphs: bool`, i.e. "are Powerline separators
    /// safe?") and into the file tree (which needs the whole three-tier set, for
    /// devicons). See [`crate::icons`] for why this is detected once, at
    /// startup, and never re-derived per widget.
    pub icons: IconSet,
    /// The configured leader key. The *editor* resolves `<leader>e` while the
    /// buffer has focus; an overlay with focus has to resolve it itself, because
    /// the editor never sees those keys. See [`crate::ui::filetree`].
    pub leader: char,
    /// Where the file tree roots itself when `<leader>e` first opens it — the
    /// process's working directory, matching neo-tree's default.
    pub tree_root: PathBuf,
    /// The open overlay, if any. See [`crate::ui::overlay`] for why the file tree
    /// is *not* a leaf in [`WindowTree`].
    pub overlay: Option<Overlay>,
    /// Where keys go. Never inferred from `overlay.is_some()` — see the module
    /// docs. Read it through [`App::focus`], which cannot report `Overlay` while
    /// no overlay is open.
    focus: Focus,
    /// The active hop (`f`), if any. While `Some`, keystrokes go to the hop
    /// and its labels are painted over the buffer — see [`crate::ui::hop`].
    hop: Option<HopState>,
    /// Set after `<C-w>` in Normal mode: the *next* key is a window command
    /// (`h`/`j`/`k`/`l`/`w`/`s`/`v`/…), not text. See [`App::handle_event`].
    awaiting_window_key: bool,
    /// The window area painted on the last frame, so spatial `<C-w>h/j/k/l`
    /// (which needs geometry) and hop (which needs the active window's rect)
    /// can be resolved between frames, when there is no live `Frame` to ask.
    last_windows_area: Rect,
    /// The live language-server registry (`(server, root)`-keyed — see
    /// [`crate::lsp::LspClient`]). Servers spawn lazily on the first LSP request
    /// for a file of their language.
    lsp: LspClient,
    /// An open hover popup's text, one screen line per element. Dismissed on the
    /// next keypress. See [`Action::LspHover`].
    lsp_hover: Option<Vec<String>>,
    /// An open references list (`<leader>gr`). While `Some`, `j`/`k` navigate,
    /// Enter jumps, `q`/`<Esc>` closes — keys never reach the editor.
    lsp_refs: Option<RefList>,
    /// An in-progress rename (`<leader>rn`): the captured symbol context plus
    /// the new name being typed. While `Some`, keys go to the prompt.
    lsp_rename: Option<RenameState>,
    /// The most recent diagnostics per file, polled from the running servers
    /// (diagnostics are *pushed* asynchronously, so they are refreshed on the
    /// event loop's idle tick — see [`App::refresh_diagnostics`]). Rendered as
    /// gutter signs + underlines over the active window (cj0.16).
    diagnostics: std::collections::HashMap<PathBuf, Vec<crate::lsp::Diagnostic>>,
    /// Files already announced to their server with `didOpen`, so a buffer of a
    /// served language gets per-file diagnostics without re-opening it every
    /// poll.
    lsp_opened: std::collections::HashSet<PathBuf>,
    /// Files whose language has no available server (none registered, or its
    /// binary is not on `PATH`). Remembered so the attach-on-open path in
    /// [`App::refresh_diagnostics`] does not rescan `PATH` on every idle tick of
    /// an unserved buffer.
    lsp_no_server: std::collections::HashSet<PathBuf>,
    /// Set after `]`/`[` in Normal mode: the next key (`d`) completes a
    /// diagnostic-navigation motion (`]d`/`[d`). See [`App::handle_event`].
    pending_bracket: Option<char>,
    /// The open insert-mode completion popup, if any. Driven by
    /// [`App::refresh_completion`] as the user types, navigated with
    /// `<C-n>`/`<C-p>`, accepted with `<CR>`/`<Tab>` — see
    /// [`App::completion_intercept`].
    completion: Option<CompletionMenu>,
    /// The active snippet expansion being navigated with `<Tab>`/`<S-Tab>`, if
    /// any. Set when a snippet completion is accepted; cleared on `<Tab>` past
    /// the final tabstop or when insert mode ends. See [`crate::ui::snippet`].
    snippet: Option<SnippetSession>,
    /// The active buffer cursor's screen cell on the last frame, so the
    /// completion popup can be anchored at the cursor (the popup is a render
    /// pass with no live `Frame` geometry of its own — the same between-frame
    /// trick as [`Self::last_windows_area`]).
    last_cursor_screen: Option<(u16, u16)>,
}

/// A navigable list of reference locations (`<leader>gr`).
struct RefList {
    locations: Vec<LspLocation>,
    selected: usize,
}

/// An in-flight rename: everything needed to issue `textDocument/rename` once
/// the user confirms a new name, captured at the moment `<leader>rn` was
/// pressed (so a later cursor move does not change what gets renamed).
struct RenameState {
    input: String,
    filetype: String,
    file: PathBuf,
    pos: Position,
    line_text: String,
}

/// The most rows the completion popup shows at once before it starts to scroll
/// to keep the selection visible — a blink.cmp-ish height.
const MAX_COMPLETION_ROWS: usize = 8;

/// The open insert-mode completion popup: the ranked candidates, which one is
/// selected, and where the accepted text replaces from.
///
/// `anchor` is the start of the token being completed (the identifier prefix, or
/// the whole path fragment in a path context). Accepting an item replaces
/// `anchor..cursor` with its `insert_text` (or its expanded snippet) — so the
/// menu owns "what span does this completion overwrite", which the buffer only
/// learns at accept time.
struct CompletionMenu {
    items: Vec<CItem>,
    selected: usize,
    scroll: usize,
    anchor: Position,
    /// Whether the menu was opened explicitly with `<C-Space>`. An explicit menu
    /// survives the prefix emptying (so `<C-Space>` on nothing lists everything);
    /// an auto-triggered one closes when there is no longer a prefix to filter.
    explicit: bool,
}

impl<H: EditorHost> App<H> {
    /// How long [`event::poll`] blocks before giving the loop a chance to
    /// notice `should_quit` and exit even with no terminal event pending.
    /// Not a redraw interval — see the module docs.
    const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(250);

    pub fn new(host: H, options: Options, theme: Theme, icons: IconSet, leader: char) -> Self {
        // Seed the sole window from the editor's *current* buffer, not a
        // hard-coded `BufferId(0)`: `run()` opens the files before building the
        // `App`, so the active buffer is already whatever was opened last, and
        // a window pointing at buffer 0 would render the empty scratch buffer
        // instead (or, once `render_windows` respects `window.buffer`, nothing
        // useful).
        let windows = WindowTree::single(host.active_buffer_id());
        Self {
            host,
            windows,
            theme,
            options,
            message: StatusMessage::None,
            icons,
            leader,
            // A process with no working directory (deleted out from under it) is
            // pathological but not a reason to refuse to start; `.` is what every
            // other tool falls back to.
            tree_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            overlay: None,
            focus: Focus::Buffer,
            hop: None,
            awaiting_window_key: false,
            last_windows_area: Rect::default(),
            lsp: LspClient::new(),
            lsp_hover: None,
            lsp_refs: None,
            lsp_rename: None,
            diagnostics: std::collections::HashMap::new(),
            lsp_opened: std::collections::HashSet::new(),
            lsp_no_server: std::collections::HashSet::new(),
            pending_bracket: None,
            completion: None,
            snippet: None,
            last_cursor_screen: None,
        }
    }

    /// Shuts down every running language server. Called once the event loop
    /// exits, so kvim does not leave orphaned `rust-analyzer`/`texlab`
    /// processes behind.
    pub fn shutdown_lsp(&mut self) {
        self.lsp.shutdown_all();
    }

    /// Where keystrokes are currently going.
    ///
    /// Guards the one combination the type system allows but the app must never
    /// observe: `Focus::Overlay` with no overlay open. Closing an overlay always
    /// restores `Focus::Buffer`, so this is belt-and-braces — but it means a
    /// future bug in that path degrades to "keys go to the editor" rather than
    /// "keys go nowhere and the editor appears frozen".
    pub fn focus(&self) -> Focus {
        match self.overlay {
            Some(_) => self.focus,
            None => Focus::Buffer,
        }
    }

    /// Whether Powerline separators can be drawn — see [`Statusline::glyphs`].
    fn glyphs(&self) -> bool {
        self.icons.needs_font()
    }

    /// Runs the event loop until the editor reports [`HostResponse::Quit`].
    ///
    /// Owns the terminal for its duration: callers are expected to have
    /// already entered raw mode / the alternate screen via
    /// [`crate::ui::terminal::TerminalGuard`] before calling this (and to
    /// hold that guard until after `run` returns), so this function only
    /// needs to know how to draw into whatever `Terminal` it's handed.
    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        terminal.draw(|frame| self.render(frame))?;
        loop {
            if !event::poll(Self::EVENT_POLL_INTERVAL)? {
                // Idle tick: diagnostics are pushed by the server asynchronously,
                // so poll for fresh ones here (never on a fixed redraw clock —
                // see the module docs) and repaint only if they changed.
                if self.refresh_diagnostics() {
                    terminal.draw(|frame| self.render(frame))?;
                }
                continue;
            }
            let action = self.handle_event(event::read()?);
            match action {
                LoopAction::Quit => return Ok(()),
                LoopAction::Redraw => {
                    terminal.draw(|frame| self.render(frame))?;
                }
                LoopAction::Continue => {}
            }
        }
    }

    /// Translates and dispatches one crossterm [`Event`], updating
    /// `self.host` / `self.message` as needed, and reports what the caller
    /// should do next. Pure with respect to the terminal (no drawing) so it
    /// can be unit tested without a `Terminal` at all — see the tests
    /// module.
    pub fn handle_event(&mut self, ev: Event) -> LoopAction {
        match ev {
            Event::Key(key_event) => {
                let Some(kp) = map_crossterm_key(key_event) else {
                    return LoopAction::Continue;
                };
                // The focus branch, and the reason `j` in the file tree does not
                // also move the text cursor: when an overlay has focus the editor
                // is not handed the key at all.
                match self.focus() {
                    Focus::Overlay => self.handle_overlay_key(kp),
                    Focus::Buffer => {
                        // An LSP prompt/list in flight owns the keyboard, like an
                        // overlay: rename input and reference navigation must not
                        // reach the editor. Checked before hop and window keys.
                        if self.lsp_rename.is_some() {
                            return self.handle_rename_key(kp);
                        }
                        if self.lsp_refs.is_some() {
                            return self.handle_refs_key(kp);
                        }
                        if self.lsp_hover.is_some() {
                            // Any key dismisses a hover popup (Neovim: the popup
                            // closes on the next action), and is then swallowed.
                            self.lsp_hover = None;
                            return LoopAction::Redraw;
                        }
                        // Insert-mode autocompletion. Intercepts the popup's own
                        // keys (navigate/accept/cancel/trigger, and snippet
                        // `<Tab>` jumps); anything it does not consume falls
                        // through to the editor and refreshes the menu after.
                        if let Some(action) = self.completion_intercept(kp) {
                            return action;
                        }
                        // A hop in flight owns the keyboard, exactly like an
                        // overlay: its label keystrokes must not reach the editor.
                        if self.hop.is_some() {
                            return self.handle_hop_key(kp);
                        }
                        // `<C-w>` in Normal mode begins a window command; the
                        // *next* key completes it. In Insert mode `<C-w>` is the
                        // editor's (delete-word-back), so only intercept in
                        // Normal.
                        if self.awaiting_window_key {
                            return self.handle_window_key(kp);
                        }
                        // `]d`/`[d` diagnostic navigation: `]`/`[` in Normal mode
                        // arms the motion, `d` completes it. A following key that
                        // isn't `d` falls through to the editor (kvim has no other
                        // `]`/`[` motions, so the bracket is simply dropped).
                        if let Some(bracket) = self.pending_bracket.take() {
                            if kp.key == Key::Char('d') {
                                return self.jump_diagnostic(bracket == ']');
                            }
                            return self.handle_host_key(kp);
                        }
                        if self.host.mode() == Mode::Normal
                            && matches!(kp.key, Key::Char(']') | Key::Char('['))
                            && !kp.mods.ctrl
                        {
                            self.pending_bracket = Some(match kp.key {
                                Key::Char(c) => c,
                                _ => unreachable!(),
                            });
                            return LoopAction::Continue;
                        }
                        if kp.mods.ctrl
                            && kp.key == Key::Char('w')
                            && self.host.mode() == Mode::Normal
                        {
                            self.awaiting_window_key = true;
                            return LoopAction::Continue;
                        }
                        self.host_key_then_refresh_completion(kp)
                    }
                }
            }
            Event::Resize(_, _) => LoopAction::Redraw,
            // Mouse events are unhandled while mouse capture defaults to
            // off (see `ui::terminal`); paste/focus events have no UI
            // meaning yet.
            Event::Mouse(_) | Event::Paste(_) | Event::FocusGained | Event::FocusLost => {
                LoopAction::Continue
            }
        }
    }

    fn handle_host_key(&mut self, kp: KeyPress) -> LoopAction {
        match self.host.handle_key(kp) {
            HostResponse::Quit => LoopAction::Quit,
            HostResponse::QuitWindow => self.quit_active_window(),
            HostResponse::Changed => {
                self.sync_active_window();
                LoopAction::Redraw
            }
            HostResponse::Message(m) => self.info(m),
            HostResponse::Error(e) => self.error(e),
            HostResponse::Unchanged => LoopAction::Continue,
            HostResponse::Action(action) => self.handle_action(action),
            HostResponse::Window(cmd) => self.handle_window_command(cmd),
            HostResponse::Scroll(req) => self.handle_scroll(req),
        }
    }

    /// Performs a configured [`Action`] — the last hop of a keymap like
    /// `<leader>e`.
    ///
    /// Only [`Action::FileTreeToggle`] has a UI today. The rest keep saying so,
    /// plainly, rather than being silently swallowed: an editor that ignores a
    /// key you bound is far worse than one that tells you the key is understood
    /// and not yet implemented. Each arm added below (the pickers, hop, harpoon)
    /// deletes one line of that message.
    fn handle_action(&mut self, action: Action) -> LoopAction {
        match action {
            Action::FileTreeToggle => self.toggle_file_tree(),
            Action::HopWords => self.start_hop(),
            Action::LspDefinition => self.lsp_definition(),
            Action::LspReferences => self.lsp_references(),
            Action::LspRename => self.lsp_start_rename(),
            Action::LspHover => self.lsp_hover(),
            other => self.info(format!("{other:?} is not wired into the UI yet")),
        }
    }

    // ---------------------------------------------------------------
    // LSP: go-to-definition, references, hover, rename.
    //
    // Each request is synchronous: pressing `<leader>gd` blocks the UI until
    // the server answers (the first request to a rust-analyzer also blocks
    // while it spawns and indexes). That is honest and acceptable for a first
    // cut — Neovim's first LSP call feels the same — and avoids threading an
    // async runtime into the event loop. Server binaries that are not installed
    // degrade to a statusline note, never a crash.
    // ---------------------------------------------------------------

    /// The `(filetype, path, cursor, cursor-line-text)` an LSP request needs,
    /// or `None` when the active buffer has no path or no known language server.
    fn lsp_context(&self) -> Option<(String, PathBuf, Position, String)> {
        let path = self.host.buffer().path()?.to_path_buf();
        let filetype = lsp_filetype(&path)?;
        let cursor = self.host.cursor();
        let line = self.host.buffer().line(cursor.line).unwrap_or_default();
        Some((filetype, path, cursor, line))
    }

    /// `<leader>gd`: jump to the definition of the symbol under the cursor.
    fn lsp_definition(&mut self) -> LoopAction {
        let Some((ft, file, pos, line)) = self.lsp_context() else {
            return self.info("no language server configured for this buffer".to_string());
        };
        if !LspClient::server_available(&ft) {
            return self.info(format!("{ft} language server is not installed"));
        }
        match self.lsp.definition(&ft, &file, pos, &line) {
            Ok(locs) if locs.is_empty() => self.info("no definition found".to_string()),
            Ok(locs) => {
                let target = locs[0].clone();
                self.jump_to_location(&target.file, target.range.anchor)
            }
            Err(e) => self.error(format!("LSP definition: {e}")),
        }
    }

    /// `<leader>gr`: open a navigable list of references to the symbol.
    fn lsp_references(&mut self) -> LoopAction {
        let Some((ft, file, pos, line)) = self.lsp_context() else {
            return self.info("no language server configured for this buffer".to_string());
        };
        if !LspClient::server_available(&ft) {
            return self.info(format!("{ft} language server is not installed"));
        }
        match self.lsp.references(&ft, &file, pos, &line) {
            Ok(locs) if locs.is_empty() => self.info("no references found".to_string()),
            Ok(locs) => {
                self.lsp_refs = Some(RefList { locations: locs, selected: 0 });
                LoopAction::Redraw
            }
            Err(e) => self.error(format!("LSP references: {e}")),
        }
    }

    /// `K`: show hover documentation for the symbol under the cursor.
    fn lsp_hover(&mut self) -> LoopAction {
        let Some((ft, file, pos, line)) = self.lsp_context() else {
            return self.info("no language server configured for this buffer".to_string());
        };
        if !LspClient::server_available(&ft) {
            return self.info(format!("{ft} language server is not installed"));
        }
        match self.lsp.hover(&ft, &file, pos, &line) {
            Ok(Some(text)) => {
                self.lsp_hover = Some(text.lines().map(str::to_string).collect());
                LoopAction::Redraw
            }
            Ok(None) => self.info("no hover information".to_string()),
            Err(e) => self.error(format!("LSP hover: {e}")),
        }
    }

    /// `<leader>rn`: begin renaming the symbol under the cursor. Captures the
    /// request context now (so a later cursor move can't change the target) and
    /// seeds the input with the current identifier.
    fn lsp_start_rename(&mut self) -> LoopAction {
        let Some((ft, file, pos, line)) = self.lsp_context() else {
            return self.info("no language server configured for this buffer".to_string());
        };
        if !LspClient::server_available(&ft) {
            return self.info(format!("{ft} language server is not installed"));
        }
        let seed = word_under_cursor(&line, pos.col);
        self.lsp_rename = Some(RenameState { input: seed, filetype: ft, file, pos, line_text: line });
        LoopAction::Redraw
    }

    /// Feeds a key to the rename prompt: type the new name, Enter to apply,
    /// `<Esc>` to cancel.
    fn handle_rename_key(&mut self, kp: KeyPress) -> LoopAction {
        match kp.key {
            Key::Escape => {
                self.lsp_rename = None;
                LoopAction::Redraw
            }
            Key::Enter => self.apply_rename(),
            Key::Backspace => {
                if let Some(r) = self.lsp_rename.as_mut() {
                    r.input.pop();
                }
                LoopAction::Redraw
            }
            Key::Char(c) => {
                if let Some(r) = self.lsp_rename.as_mut() {
                    r.input.push(c);
                }
                LoopAction::Redraw
            }
            _ => LoopAction::Continue,
        }
    }

    /// Issues the rename and writes the resulting edits to disk, then reloads
    /// the active buffer so the change is visible.
    fn apply_rename(&mut self) -> LoopAction {
        let Some(r) = self.lsp_rename.take() else { return LoopAction::Continue };
        if r.input.is_empty() {
            return self.info("rename cancelled (empty name)".to_string());
        }
        match self.lsp.rename(&r.filetype, &r.file, r.pos, &r.line_text, &r.input) {
            Ok(edits) if edits.is_empty() => self.info("rename produced no changes".to_string()),
            Ok(edits) => {
                let count = edits.len();
                for edit in &edits {
                    if let Err(e) = std::fs::write(&edit.path, &edit.updated) {
                        return self.error(format!("writing {}: {e}", edit.path.display()));
                    }
                }
                // Reload the active buffer from disk so the rename shows on
                // screen. (`Editor::open` reopens the path; the window's buffer
                // id follows via `sync_active_window`.)
                if let Some(path) = self.host.buffer().path().map(Path::to_path_buf) {
                    let cursor = self.host.cursor();
                    if self.host.open(&path).is_ok() {
                        self.host.set_active(self.host.active_buffer_id(), cursor);
                        self.sync_active_window();
                    }
                }
                self.info(format!("renamed to '{}' across {count} file(s)", r.input))
            }
            Err(e) => self.error(format!("LSP rename: {e}")),
        }
    }

    /// Feeds a key to the references list: `j`/`k` (or arrows) move, Enter
    /// jumps to the selected reference, `q`/`<Esc>` closes.
    fn handle_refs_key(&mut self, kp: KeyPress) -> LoopAction {
        let Some(refs) = self.lsp_refs.as_mut() else { return LoopAction::Continue };
        match kp.key {
            Key::Escape | Key::Char('q') => {
                self.lsp_refs = None;
                LoopAction::Redraw
            }
            Key::Char('j') | Key::Down => {
                if refs.selected + 1 < refs.locations.len() {
                    refs.selected += 1;
                }
                LoopAction::Redraw
            }
            Key::Char('k') | Key::Up => {
                refs.selected = refs.selected.saturating_sub(1);
                LoopAction::Redraw
            }
            Key::Enter => {
                let target = refs.locations[refs.selected].clone();
                self.lsp_refs = None;
                self.jump_to_location(&target.file, target.range.anchor)
            }
            _ => LoopAction::Continue,
        }
    }

    /// Opens `file` (if not already active) and moves the cursor to `pos` — the
    /// shared tail of go-to-definition and reference-jump.
    fn jump_to_location(&mut self, file: &Path, pos: Position) -> LoopAction {
        let already_here = self.host.buffer().path() == Some(file);
        if !already_here
            && let Err(e) = self.host.open(file)
        {
            return self.error(e);
        }
        self.host.set_active(self.host.active_buffer_id(), pos);
        self.sync_active_window();
        self.windows.active_mut().scroll = Scroll::default();
        self.info(format!("{}:{pos}", file.display()))
    }

    // ---------------------------------------------------------------
    // Insert-mode completion: the `blink.cmp` replacement.
    //
    // The headless engine (`lsp::completion`) already merges and ranks the four
    // sources; this layer decides *when* to (re)query, owns the popup state, and
    // turns an accepted item into a buffer edit (a plain insert, or a snippet
    // expansion driven by `kopitiam-snippet`). Frontend keys follow the
    // maintainer's `blink.cmp`/`LuaSnip`: `<C-Space>` triggers, `<C-n>`/`<C-p>`
    // (and Down/Up) move, `<CR>`/`<Tab>` accept, `<C-e>` cancels, `<Tab>`/
    // `<S-Tab>` drive snippet tabstops while a snippet is active.
    // ---------------------------------------------------------------

    /// Intercepts the completion popup's and snippet session's own keys, or
    /// returns `None` to let the key reach the editor (a typed character, which
    /// then refreshes the menu via [`Self::host_key_then_refresh_completion`]).
    fn completion_intercept(&mut self, kp: KeyPress) -> Option<LoopAction> {
        let insert = self.host.mode() == Mode::Insert;

        // A live snippet session claims `<Tab>`/`<S-Tab>` for tabstop jumps,
        // even with the menu closed — this is `LuaSnip`'s jump, and it must win
        // over the editor's own `<Tab>` (indent) while a snippet is active.
        if self.snippet.is_some() {
            match kp.key {
                Key::Tab if !kp.mods.shift => return Some(self.snippet_jump(true)),
                Key::Tab if kp.mods.shift => return Some(self.snippet_jump(false)),
                Key::BackTab => return Some(self.snippet_jump(false)),
                _ => {}
            }
        }

        // `<C-Space>`: open/refresh the menu. Terminals disagree on the byte for
        // Ctrl-Space (a literal space, or NUL / Ctrl-@), so accept all three.
        if insert
            && kp.mods.ctrl
            && matches!(kp.key, Key::Char(' ') | Key::Char('\0') | Key::Char('@'))
        {
            self.refresh_completion(true);
            return Some(LoopAction::Redraw);
        }

        // Everything below needs an open menu.
        self.completion.as_ref()?;
        match kp.key {
            Key::Char('n') if kp.mods.ctrl => Some(self.menu_move(1)),
            Key::Char('p') if kp.mods.ctrl => Some(self.menu_move(-1)),
            Key::Down => Some(self.menu_move(1)),
            Key::Up => Some(self.menu_move(-1)),
            // Page through a long list (Neovim's pmenu `<C-f>`/`<C-b>`).
            Key::Char('f') if kp.mods.ctrl => Some(self.menu_move(MAX_COMPLETION_ROWS as isize)),
            Key::Char('b') if kp.mods.ctrl => Some(self.menu_move(-(MAX_COMPLETION_ROWS as isize))),
            // `<C-e>`: dismiss, staying in insert mode with nothing inserted.
            Key::Char('e') if kp.mods.ctrl => {
                self.completion = None;
                Some(LoopAction::Redraw)
            }
            Key::Enter | Key::Tab => Some(self.accept_completion()),
            _ => None,
        }
    }

    /// Feeds a key to the editor, then — if still in insert mode — refreshes the
    /// completion menu against the new buffer state; if the key left insert mode
    /// (e.g. `<Esc>`), closes the menu and any snippet session.
    fn host_key_then_refresh_completion(&mut self, kp: KeyPress) -> LoopAction {
        let action = self.handle_host_key(kp);
        if action == LoopAction::Quit {
            return action;
        }
        if self.host.mode() == Mode::Insert {
            let changed = self.refresh_completion(false);
            if changed && action == LoopAction::Continue {
                return LoopAction::Redraw;
            }
        } else if self.completion.is_some() || self.snippet.is_some() {
            self.completion = None;
            self.snippet = None;
            return LoopAction::Redraw;
        }
        action
    }

    /// Moves the popup selection by `delta` (wrapping, like `<C-n>`/`<C-p>`) and
    /// scrolls so the selection stays visible.
    fn menu_move(&mut self, delta: isize) -> LoopAction {
        if let Some(menu) = self.completion.as_mut() {
            let n = menu.items.len();
            if n == 0 {
                return LoopAction::Continue;
            }
            menu.selected = (menu.selected as isize + delta).rem_euclid(n as isize) as usize;
            let visible = n.min(MAX_COMPLETION_ROWS);
            if menu.selected < menu.scroll {
                menu.scroll = menu.selected;
            } else if menu.selected >= menu.scroll + visible {
                menu.scroll = menu.selected + 1 - visible;
            }
        }
        LoopAction::Redraw
    }

    /// Accepts the selected candidate: replaces the completed token
    /// (`anchor..cursor`) with the item's text, expanding a snippet through
    /// `kopitiam-snippet` when the item is one.
    fn accept_completion(&mut self) -> LoopAction {
        let Some(menu) = self.completion.take() else { return LoopAction::Continue };
        let Some(item) = menu.items.get(menu.selected).cloned() else { return LoopAction::Redraw };
        let cursor = self.host.cursor();
        let range = Range::new(menu.anchor, cursor);
        match &item.snippet {
            Some(body) => self.expand_snippet(range, body),
            None => {
                self.host.replace_range(range, &item.insert_text);
                self.sync_active_window();
            }
        }
        LoopAction::Redraw
    }

    /// Expands `body` (LSP snippet grammar) over `range` and, if it has
    /// tabstops, starts a [`SnippetSession`] with the cursor on the first stop.
    fn expand_snippet(&mut self, range: Range, body: &str) {
        // Variable resolution: only the handful we can answer locally; anything
        // else falls back to the snippet's own `${VAR:default}` (or empty).
        let filename =
            self.host.buffer().path().and_then(|p| p.file_name()).and_then(|n| n.to_str()).map(str::to_string);
        let resolve = |var: &str| -> Option<String> {
            match var {
                "TM_FILENAME" => filename.clone(),
                "TM_FILENAME_BASE" => filename.as_deref().and_then(|f| f.rsplit_once('.').map(|(b, _)| b.to_string())),
                _ => None,
            }
        };
        let expansion = match kopitiam_snippet::Snippet::parse(body) {
            Ok(snippet) => snippet.expand(&resolve),
            // A snippet body that will not parse is inserted literally rather
            // than dropped — a broken snippet should still type its text.
            Err(_) => {
                self.host.replace_range(range, body);
                self.sync_active_window();
                return;
            }
        };
        let at = range.normalized().0;
        let landed = self.host.replace_range(range, &expansion.text);
        match SnippetSession::from_expansion(&expansion, at) {
            Some(session) => {
                let target = session.target();
                self.host.move_cursor(target);
                self.snippet = Some(session);
            }
            // No tabstops (a plain snippet, or the scaffold stub until the real
            // engine lands): leave the cursor at the end of the inserted text.
            None => self.host.move_cursor(landed),
        }
        self.sync_active_window();
    }

    /// Jumps to the next (`forward`) or previous snippet tabstop, ending the
    /// session on a `<Tab>` past the final stop.
    fn snippet_jump(&mut self, forward: bool) -> LoopAction {
        let target = {
            let Some(session) = self.snippet.as_mut() else { return LoopAction::Continue };
            let moved = if forward { session.advance() } else { session.retreat() };
            if forward && !moved {
                // Past the final `$0`: the snippet is done.
                self.snippet = None;
                return LoopAction::Redraw;
            }
            session.target()
        };
        self.host.move_cursor(target);
        self.sync_active_window();
        LoopAction::Redraw
    }

    /// Recomputes the completion popup against the cursor's current identifier
    /// (or path) prefix. `explicit` marks a `<C-Space>` trigger, which keeps the
    /// menu open even with an empty prefix. Returns whether the popup changed
    /// (so the caller repaints only when needed).
    fn refresh_completion(&mut self, explicit: bool) -> bool {
        if self.host.mode() != Mode::Insert {
            return self.close_completion_if_open();
        }
        let cursor = self.host.cursor();
        let line = self.host.buffer().line(cursor.line).unwrap_or_default();

        // A path-ish token (containing `/`) takes precedence and yields only
        // path candidates, replacing the whole fragment on accept.
        if let Some((anchor_col, prefix, path_items)) = self.path_context(&line, cursor.col) {
            let ranked = completion::merge_and_rank(&prefix, vec![], vec![], vec![], path_items);
            return self.set_completion(ranked, Position::new(cursor.line, anchor_col), explicit);
        }

        let (anchor_col, prefix) = identifier_prefix(&line, cursor.col);
        let explicit = explicit || self.completion.as_ref().is_some_and(|m| m.explicit);
        if prefix.is_empty() && !explicit {
            return self.close_completion_if_open();
        }
        let anchor = Position::new(cursor.line, anchor_col);
        let ranked = self.gather_completions(&prefix, anchor);
        self.set_completion(ranked, anchor, explicit)
    }

    /// Installs a freshly-ranked candidate list, preserving the selection index
    /// when it still fits, or clears the popup when the list is empty. Returns
    /// whether anything changed.
    fn set_completion(&mut self, items: Vec<CItem>, anchor: Position, explicit: bool) -> bool {
        if items.is_empty() {
            return self.close_completion_if_open();
        }
        let selected = self
            .completion
            .as_ref()
            .map(|m| m.selected.min(items.len() - 1))
            .unwrap_or(0);
        let scroll = selected.saturating_sub(MAX_COMPLETION_ROWS - 1);
        self.completion = Some(CompletionMenu { items, selected, scroll, anchor, explicit });
        true
    }

    /// Closes the popup if it is open, reporting whether it was.
    fn close_completion_if_open(&mut self) -> bool {
        if self.completion.is_some() {
            self.completion = None;
            true
        } else {
            false
        }
    }

    /// Gathers and ranks the four sources for identifier-context completion:
    /// LSP, built-in snippets, buffer words, and (empty here — path is its own
    /// context) paths.
    fn gather_completions(&mut self, prefix: &str, _anchor: Position) -> Vec<CItem> {
        let lsp_items = self.lsp_completion_items();
        let filetype = self.host.buffer().path().and_then(lsp_filetype);
        let snippet_items = filetype.map(|ft| completion::builtin_snippets(&ft)).unwrap_or_default();
        let buffer_items = {
            let buffer = self.host.buffer();
            let lines: Vec<String> = (0..buffer.line_count()).map(|i| buffer.line(i).unwrap_or_default()).collect();
            let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
            completion::buffer_words(&refs)
        };
        completion::merge_and_rank(prefix, lsp_items, snippet_items, buffer_items, vec![])
    }

    /// Fetches `textDocument/completion` for the active buffer and converts each
    /// item to the headless engine's shape. Empty (never an error to the user)
    /// when no server is *already running* for the buffer or the request fails —
    /// a completion source going quiet should narrow the menu, not interrupt
    /// typing.
    ///
    /// # Why `is_running`, not `server_available`
    ///
    /// The menu is refreshed on **every keystroke**. Gating on
    /// `server_available` (binary is on `PATH`) would make the *first* keystroke
    /// in a fresh buffer spawn rust-analyzer and block the UI on its initial
    /// indexing pass — seconds to minutes — which is unacceptable mid-typing.
    /// A server is instead spawned lazily by the attach-on-open path
    /// ([`Self::refresh_diagnostics`], AID-0023) on the event loop's idle tick;
    /// completion only *queries* a server that is already up. So a just-opened
    /// buffer's first few keystrokes may see LSP-less (buffer/snippet only)
    /// completions until indexing finishes, then LSP items join in — the same
    /// "warms up after a moment" behaviour Neovim's LSP completion has, and the
    /// reason the PTY test waits for indexing before it types.
    fn lsp_completion_items(&mut self) -> Vec<CItem> {
        let Some((ft, file, pos, line)) = self.lsp_context() else { return Vec::new() };
        if !self.lsp.is_running(&ft) {
            return Vec::new();
        }
        match self.lsp.completion(&ft, &file, pos, &line) {
            Ok(items) => items.into_iter().map(convert_completion_item).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Detects a path-completion context: a trailing run of path characters that
    /// contains a `/`. Returns `(token-start column, filename prefix, candidate
    /// list)`, or `None` when the cursor is not in one.
    fn path_context(&self, line: &str, col: usize) -> Option<(usize, String, Vec<CItem>)> {
        let graphemes: Vec<&str> = line.graphemes(true).collect();
        let is_path = |g: &str| {
            let c = g.chars().next().unwrap_or(' ');
            c.is_alphanumeric() || matches!(c, '_' | '/' | '.' | '-' | '~')
        };
        let end = col.min(graphemes.len());
        let mut start = end;
        while start > 0 && is_path(graphemes[start - 1]) {
            start -= 1;
        }
        let token: String = graphemes[start..end].concat();
        if !token.contains('/') {
            return None;
        }
        let base = self
            .host
            .buffer()
            .path()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.tree_root.clone());
        let items = completion::path_candidates(&token, &base);
        if items.is_empty() {
            return None;
        }
        let fname = token.rsplit('/').next().unwrap_or("").to_string();
        Some((start, fname, items))
    }

    /// Polls the running server for the active buffer's diagnostics, updating
    /// [`Self::diagnostics`]. Returns whether the set changed (so the caller
    /// repaints only when there is something new). A no-op when no server is
    /// running for the active buffer — diagnostics appear once any LSP action
    /// has spawned the server.
    fn refresh_diagnostics(&mut self) -> bool {
        let Some((ft, file, _, _)) = self.lsp_context() else { return false };
        // Attach-on-open: announce the buffer to its server the first time a
        // served file is shown. `did_open` lazily spawns the server (see
        // `LspClient::session`), so this is what brings the LSP -- diagnostics
        // included -- to life on open, instead of leaving it dormant until the
        // user happens to issue a gd/hover that spawns it. Without it, a file
        // you only read never gets diagnostics at all.
        //
        // Gated three ways so the idle tick stays cheap and honest: `lsp_opened`
        // announces a present server exactly once (never re-spawning it every
        // tick); `lsp_no_server` remembers a file whose language has no server
        // (or whose server binary is not installed) so we do not rescan `PATH`
        // on every idle tick of, say, a plain-text buffer; and `server_available`
        // is what makes the missing-binary case degrade silently rather than
        // spawn-and-fail.
        if !self.lsp_opened.contains(&file) {
            if self.lsp_no_server.contains(&file) {
                return false;
            }
            if !LspClient::server_available(&ft) {
                self.lsp_no_server.insert(file.clone());
                return false;
            }
            let text = self.active_buffer_text();
            if self.lsp.did_open(&ft, &file, &text).is_ok() {
                self.lsp_opened.insert(file.clone());
            } else {
                return false;
            }
        }
        match self.lsp.diagnostics(&ft, &file) {
            Ok(diags) => {
                let changed = self.diagnostics.get(&file) != Some(&diags);
                if changed {
                    if diags.is_empty() {
                        self.diagnostics.remove(&file);
                    } else {
                        self.diagnostics.insert(file, diags);
                    }
                }
                changed
            }
            Err(_) => false,
        }
    }

    /// The active buffer's full text, reconstructed from its lines — needed for
    /// `didOpen`. (`BufferView` exposes lines, not a whole-buffer string.)
    fn active_buffer_text(&self) -> String {
        let buffer = self.host.buffer();
        (0..buffer.line_count()).map(|i| buffer.line(i).unwrap_or_default()).collect::<Vec<_>>().join("\n")
    }

    /// `]d` / `[d`: move the cursor to the next / previous diagnostic in the
    /// active buffer, wrapping around. A no-op (with a note) when the buffer has
    /// none.
    fn jump_diagnostic(&mut self, forward: bool) -> LoopAction {
        let Some(file) = self.host.buffer().path().map(Path::to_path_buf) else {
            return LoopAction::Continue;
        };
        // Collect (position, first-line message) pairs, then drop the borrow of
        // `self.diagnostics` before any `&mut self` call below.
        let (target, msg) = {
            let Some(diags) = self.diagnostics.get(&file) else {
                return self.info("no diagnostics in this buffer".to_string());
            };
            let mut entries: Vec<(Position, String)> = diags
                .iter()
                .map(|d| (d.range.normalized().0, d.message.lines().next().unwrap_or_default().to_string()))
                .collect();
            entries.sort_by_key(|(p, _)| *p);
            entries.dedup_by_key(|(p, _)| *p);
            if entries.is_empty() {
                return self.info("no diagnostics in this buffer".to_string());
            }
            let cursor = self.host.cursor();
            if forward {
                entries.iter().find(|(p, _)| *p > cursor).cloned().unwrap_or_else(|| entries[0].clone())
            } else {
                entries.iter().rev().find(|(p, _)| *p < cursor).cloned().unwrap_or_else(|| entries.last().unwrap().clone())
            }
        };
        self.host.set_active(self.host.active_buffer_id(), target);
        self.sync_active_window();
        self.info(format!("{target}: {msg}"))
    }

    /// Paints the active buffer's diagnostics over its text: a gutter sign in
    /// the severity colour on each diagnostic's first line, and an underline in
    /// that colour across the flagged range. Painted after the text (and syntax)
    /// pass, like [`Self::paint_hop_labels`], so it layers on top.
    fn paint_diagnostics(&self, frame: &mut Frame, rect: Rect, gutter_w: u16, scroll: Scroll, buffer_id: BufferId) {
        let Some(buffer) = self.host.buffer_by_id(buffer_id) else { return };
        let Some(file) = buffer.path().map(Path::to_path_buf) else { return };
        let Some(diags) = self.diagnostics.get(&file) else { return };
        let text_x = rect.x + gutter_w;
        let text_width = rect.width.saturating_sub(gutter_w) as usize;
        // Paint least-severe first so a more-severe diagnostic's sign/underline
        // wins the cell: an error squiggle should never be hidden under a hint.
        let mut ordered: Vec<&crate::lsp::Diagnostic> = diags.iter().collect();
        ordered.sort_by_key(|d| severity_priority(d.severity));
        let buf = frame.buffer_mut();
        for diag in ordered {
            let colour = severity_color(diag.severity, &self.theme);
            let (start, end) = diag.range.normalized();
            // Gutter sign on the first line of the diagnostic.
            if let Some(row) = start.line.checked_sub(scroll.top)
                && row < rect.height as usize
                && let Some(cell) = buf.cell_mut((rect.x, rect.y + row as u16))
            {
                cell.set_symbol(severity_sign(diag.severity));
                cell.set_fg(colour);
            }
            // End-of-line virtual text: the (first line of the) message,
            // painted in the severity colour a couple of cells past the line's
            // end — Neovim's inline diagnostic style.
            if let Some(row) = start.line.checked_sub(scroll.top)
                && row < rect.height as usize
            {
                let line = buffer.line(start.line).unwrap_or_default();
                let line_w = crate::ui::textarea::display_width(&line, self.options.tabstop);
                if let Some(scol) = (line_w + 2).checked_sub(scroll.left)
                    && scol < text_width
                {
                    let message = diag.message.lines().next().unwrap_or_default();
                    let avail = text_width - scol;
                    // Pad to the line's end so this (highest-severity, painted
                    // last) message overwrites any shorter one a lower-severity
                    // diagnostic on the same line left behind.
                    let mut text = format!("■ {message}");
                    while text.chars().count() < avail {
                        text.push(' ');
                    }
                    let style = Style::default().fg(colour).bg(self.theme.bg);
                    buf.set_stringn(text_x + scol as u16, rect.y + row as u16, &text, avail, style);
                }
            }

            // Underline the flagged range on each visible line it covers.
            for line_idx in start.line..=end.line {
                let Some(row) = line_idx.checked_sub(scroll.top) else { continue };
                if row >= rect.height as usize {
                    break;
                }
                let line = buffer.line(line_idx).unwrap_or_default();
                let line_len = line.graphemes(true).count();
                let first_g = if line_idx == start.line { start.col } else { 0 };
                let last_g = if line_idx == end.line { end.col } else { line_len };
                let start_disp = crate::ui::textarea::display_col_of_grapheme(&line, first_g, self.options.tabstop);
                let end_disp = crate::ui::textarea::display_col_of_grapheme(&line, last_g, self.options.tabstop).max(start_disp + 1);
                for dc in start_disp..end_disp {
                    let Some(col) = dc.checked_sub(scroll.left) else { continue };
                    if col >= text_width {
                        break;
                    }
                    if let Some(cell) = buf.cell_mut((text_x + col as u16, rect.y + row as u16)) {
                        // `set_style` adds the modifier and underline colour
                        // while preserving the cell's existing fg/bg (its syntax
                        // colour), so a squiggle never erases the text under it.
                        cell.set_style(Style::default().add_modifier(Modifier::UNDERLINED).underline_color(colour));
                    }
                }
            }
        }
    }

    /// `<leader>e`. Faithful to neo-tree's `toggle`: open if closed, close if
    /// open — including when the tree is open but the *buffer* has focus, which
    /// is where the tree ends up after you open a file from it.
    ///
    /// (Neovim users get back to a visible-but-unfocused tree with `<C-w>h`.
    /// kvim has no window-motion keys at all yet — see [`crate::ui::window`] —
    /// so today the way back in is `<leader>e` twice. Adding `<C-w>` motions is
    /// tracked separately; it is a window-tree feature, not a sidebar one.)
    fn toggle_file_tree(&mut self) -> LoopAction {
        if matches!(self.overlay, Some(Overlay::FileTree(_))) {
            self.close_overlay();
            return LoopAction::Redraw;
        }
        match FileTreePanel::open(&self.tree_root, self.leader) {
            Ok(panel) => {
                self.overlay = Some(Overlay::FileTree(panel));
                self.focus = Focus::Overlay;
                LoopAction::Redraw
            }
            Err(e) => self.error(format!("{}: {e}", self.tree_root.display())),
        }
    }

    /// Feeds a key to the focused overlay and acts on what it asks for. The
    /// overlay never touches the editor or the window tree itself — see
    /// [`OverlayOutcome`].
    fn handle_overlay_key(&mut self, kp: KeyPress) -> LoopAction {
        let Some(overlay) = self.overlay.as_mut() else {
            // Unreachable via `focus()`, which cannot report `Overlay` with none
            // open. Fall back to the editor rather than dropping the key.
            return self.handle_host_key(kp);
        };
        match overlay.handle_key(kp) {
            OverlayOutcome::Ignored => LoopAction::Continue,
            OverlayOutcome::Consumed => LoopAction::Redraw,
            OverlayOutcome::Close => {
                self.close_overlay();
                LoopAction::Redraw
            }
            OverlayOutcome::Message(m) => self.info(m),
            OverlayOutcome::Error(e) => self.error(e),
            OverlayOutcome::OpenPath { path, target } => self.open_path(&path, target),
        }
    }

    /// Opens a path an overlay selected, and moves focus to the buffer.
    ///
    /// The overlay stays open — neo-tree keeps the tree visible when you open a
    /// file from it, and closing it here would make `i`/`s` (open in a split)
    /// useless for opening a second file.
    fn open_path(&mut self, path: &Path, target: OpenTarget) -> LoopAction {
        // Split *before* opening: `WindowTree::split` duplicates the active
        // window's view, and the new window is the one that should end up showing
        // the file.
        let mut note = None;
        match target {
            OpenTarget::Current => {}
            OpenTarget::HorizontalSplit => {
                self.windows.split(SplitKind::Horizontal);
            }
            OpenTarget::VerticalSplit => {
                self.windows.split(SplitKind::Vertical);
            }
            // kvim has no tab pages (see `ui::window`: one tree, one "tab"). Say
            // so, and do the closest useful thing, rather than pretending.
            OpenTarget::Tab => {
                note = Some("kvim has no tab pages yet — opened in the current window");
            }
        }

        if let Err(e) = self.host.open(path) {
            return self.error(e);
        }
        self.focus = Focus::Buffer;
        self.sync_active_window();
        // A fresh buffer starts at the origin, so the previous file's scroll
        // offset must not survive into it.
        self.windows.active_mut().scroll = Scroll::default();

        self.message = match note {
            Some(note) => StatusMessage::Info(note.to_string()),
            None => StatusMessage::Info(format!("\"{}\"", path.display())),
        };
        LoopAction::Redraw
    }

    /// Closes the open overlay and returns focus to the buffer. The layout is
    /// restored implicitly: the sidebar reserved its columns at render time, so
    /// giving them back is a matter of no longer reserving them.
    fn close_overlay(&mut self) {
        self.overlay = None;
        self.focus = Focus::Buffer;
    }

    fn info(&mut self, message: String) -> LoopAction {
        self.message = StatusMessage::Info(message);
        LoopAction::Redraw
    }

    fn error(&mut self, message: String) -> LoopAction {
        self.message = StatusMessage::Error(message);
        LoopAction::Redraw
    }

    /// Writes the editor's live cursor and active-buffer id back into the
    /// active window. Called after every buffer-changing key, and before any
    /// window operation, so the window tree's copy of "where the active window
    /// is looking" is current before focus moves elsewhere. The buffer id is
    /// synced too (not just the cursor) so that `:e`/`:bn`, which switch the
    /// editor's buffer, are reflected in the window that requested them.
    fn sync_active_window(&mut self) {
        let cursor = self.host.cursor();
        let buffer = self.host.active_buffer_id();
        let win = self.windows.active_mut();
        win.cursor = cursor;
        win.buffer = buffer;
    }

    /// Loads the active window's saved buffer/cursor into the editor — the
    /// other half of a focus change. After [`Self::sync_active_window`] has
    /// stored the outgoing window's state and the tree's `active` has moved,
    /// this points the single-cursor editor at the newly-focused window.
    fn load_active_window(&mut self) {
        let win = *self.windows.active();
        self.host.set_active(win.buffer, win.cursor);
    }

    // ---------------------------------------------------------------
    // Window commands (`<C-w>…` and `:sp`/`:vs`/`:only`/`:close`)
    // ---------------------------------------------------------------

    /// Handles the key *after* `<C-w>` in Normal mode. Anything unrecognised
    /// is dropped (vim beeps); the deferred motions (`H`/`J`/`K`/`L` move,
    /// `T` to a new tab) report honestly rather than doing something else.
    fn handle_window_key(&mut self, kp: KeyPress) -> LoopAction {
        self.awaiting_window_key = false;
        // `<C-w><C-h>` and `<C-w>h` mean the same thing, so the ctrl bit on
        // the second key is ignored for the letter commands.
        match kp.key {
            Key::Char('h') | Key::Left => self.focus_dir(Direction::Left),
            Key::Char('j') | Key::Down => self.focus_dir(Direction::Down),
            Key::Char('k') | Key::Up => self.focus_dir(Direction::Up),
            Key::Char('l') | Key::Right => self.focus_dir(Direction::Right),
            Key::Char('w') => self.cycle_window(true),
            Key::Char('W') => self.cycle_window(false),
            Key::Char('p') => self.focus_prev_window(),
            Key::Char('s') | Key::Char('S') => self.handle_window_command(WindowCommand::Split {
                vertical: false,
                file: None,
                scratch: false,
            }),
            Key::Char('v') => self.handle_window_command(WindowCommand::Split {
                vertical: true,
                file: None,
                scratch: false,
            }),
            Key::Char('n') => self.handle_window_command(WindowCommand::Split {
                vertical: false,
                file: None,
                scratch: true,
            }),
            Key::Char('c') => self.handle_window_command(WindowCommand::Close),
            Key::Char('q') => self.quit_active_window(),
            Key::Char('o') => self.handle_window_command(WindowCommand::Only),
            Key::Char('=') => {
                self.windows.equalize();
                LoopAction::Redraw
            }
            Key::Char('+') => self.resize_window(false, true),
            Key::Char('-') => self.resize_window(false, false),
            Key::Char('>') => self.resize_window(true, true),
            Key::Char('<') => self.resize_window(true, false),
            Key::Char('x') => self.exchange_window(),
            Key::Char('r') => self.rotate_windows(),
            // Deferred: moving a window to an edge (H/J/K/L) restructures the
            // tree, and `T` needs tab pages, which kvim does not have. Say so
            // rather than silently doing the wrong thing.
            Key::Char('H') | Key::Char('J') | Key::Char('K') | Key::Char('L') => {
                self.info("<C-w> move-to-edge is not implemented yet (kopitiam-cj0.10.5)".to_string())
            }
            Key::Char('T') => {
                self.info("kvim has no tab pages yet (kopitiam-cj0.10.6)".to_string())
            }
            _ => LoopAction::Continue,
        }
    }

    fn focus_dir(&mut self, dir: Direction) -> LoopAction {
        self.sync_active_window();
        let area = self.last_windows_area;
        if self.windows.focus_direction(area, dir).is_some() {
            self.load_active_window();
            LoopAction::Redraw
        } else {
            LoopAction::Continue
        }
    }

    fn cycle_window(&mut self, forward: bool) -> LoopAction {
        self.sync_active_window();
        self.windows.cycle(forward);
        self.load_active_window();
        LoopAction::Redraw
    }

    fn focus_prev_window(&mut self) -> LoopAction {
        self.sync_active_window();
        self.windows.focus_prev();
        self.load_active_window();
        LoopAction::Redraw
    }

    fn resize_window(&mut self, vertical: bool, grow: bool) -> LoopAction {
        self.windows.resize_active(vertical, grow);
        LoopAction::Redraw
    }

    fn exchange_window(&mut self) -> LoopAction {
        self.sync_active_window();
        self.windows.exchange();
        self.load_active_window();
        LoopAction::Redraw
    }

    fn rotate_windows(&mut self) -> LoopAction {
        self.sync_active_window();
        self.windows.rotate();
        self.load_active_window();
        LoopAction::Redraw
    }

    /// `<C-w>q` / `:q` / `:wq`: close the active window, or quit the editor if
    /// it is the only one.
    fn quit_active_window(&mut self) -> LoopAction {
        if self.windows.window_count() <= 1 {
            return LoopAction::Quit;
        }
        self.sync_active_window();
        self.windows.close_active();
        self.load_active_window();
        LoopAction::Redraw
    }

    fn handle_window_command(&mut self, cmd: WindowCommand) -> LoopAction {
        match cmd {
            WindowCommand::Split { vertical, file, scratch } => {
                self.sync_active_window();
                let kind = if vertical { SplitKind::Vertical } else { SplitKind::Horizontal };
                self.windows.split(kind);
                if scratch {
                    let id = self.host.new_buffer();
                    let win = self.windows.active_mut();
                    win.buffer = id;
                    win.cursor = Position::ORIGIN;
                    win.scroll = Scroll::default();
                    self.host.set_active(id, Position::ORIGIN);
                    return LoopAction::Redraw;
                }
                if let Some(file) = file {
                    if let Err(e) = self.host.open(&file) {
                        return self.error(e);
                    }
                    let id = self.host.active_buffer_id();
                    let cursor = self.host.cursor();
                    let win = self.windows.active_mut();
                    win.buffer = id;
                    win.cursor = cursor;
                    win.scroll = Scroll::default();
                }
                LoopAction::Redraw
            }
            WindowCommand::Only => {
                self.sync_active_window();
                self.windows.only();
                LoopAction::Redraw
            }
            WindowCommand::Close => {
                if self.windows.window_count() <= 1 {
                    return self.info("cannot close last window".to_string());
                }
                self.sync_active_window();
                self.windows.close_active();
                self.load_active_window();
                LoopAction::Redraw
            }
        }
    }

    fn handle_scroll(&mut self, req: ViewportScroll) -> LoopAction {
        // The scroll offset lives in the window; the cursor line lives in the
        // editor. `zz`/`zt`/`zb` reposition the *view* around the cursor;
        // `<C-e>`/`<C-y>` move the view and drag the cursor only if it would
        // fall outside. The text height was captured on the last frame.
        let cursor_line = self.host.cursor().line;
        let height = self.active_text_height().max(1);
        let line_count = self.host.buffer().line_count();
        let win = self.windows.active_mut();
        match req {
            ViewportScroll::CenterCursor => {
                win.scroll.top = cursor_line.saturating_sub(height / 2);
            }
            ViewportScroll::CursorToTop => {
                win.scroll.top = cursor_line;
            }
            ViewportScroll::CursorToBottom => {
                win.scroll.top = cursor_line.saturating_sub(height.saturating_sub(1));
            }
            ViewportScroll::LineDown => {
                let max_top = line_count.saturating_sub(1);
                win.scroll.top = (win.scroll.top + 1).min(max_top);
                // Keep the cursor on screen: if it fell off the top, nudge it.
                if cursor_line < win.scroll.top {
                    self.host.set_active(self.host.active_buffer_id(), Position::new(win.scroll.top, self.host.cursor().col));
                }
            }
            ViewportScroll::LineUp => {
                win.scroll.top = win.scroll.top.saturating_sub(1);
                let bottom = win.scroll.top + height.saturating_sub(1);
                if cursor_line > bottom {
                    self.host.set_active(self.host.active_buffer_id(), Position::new(bottom, self.host.cursor().col));
                }
            }
        }
        self.sync_active_window();
        LoopAction::Redraw
    }

    /// The text-row height of the active window on the last painted frame —
    /// what `zz`/`<C-e>` and `<C-d>` divide against.
    fn active_text_height(&self) -> usize {
        let active = self.windows.active_id();
        self.windows
            .layout(self.last_windows_area)
            .into_iter()
            .find(|(id, _)| *id == active)
            .map(|(_, r)| r.height as usize)
            .unwrap_or(self.last_windows_area.height as usize)
    }

    // ---------------------------------------------------------------
    // Hop (`f`)
    // ---------------------------------------------------------------

    /// Starts a hop: label every word-start visible in the active window and
    /// wait for the user to type a label. Reuses the overlay layer's focus
    /// discipline (keys go to the hop, not the editor) without being an
    /// `Overlay` — see [`crate::ui::hop`] for why.
    fn start_hop(&mut self) -> LoopAction {
        let win = *self.windows.active();
        let height = self.active_text_height().max(1);
        let first = win.scroll.top;
        let buffer = self.host.buffer();
        let last = (first + height).min(buffer.line_count());
        let lines: Vec<String> = (first..last).map(|l| buffer.line(l).unwrap_or_default()).collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let hints = crate::plugins::hop::hint_words(&refs, first);
        if hints.is_empty() {
            return LoopAction::Continue;
        }
        self.hop = Some(HopState::new(hints));
        LoopAction::Redraw
    }

    /// Feeds one key to the active hop.
    fn handle_hop_key(&mut self, kp: KeyPress) -> LoopAction {
        // Esc always cancels.
        if kp.key == Key::Escape {
            self.hop = None;
            return LoopAction::Redraw;
        }
        let Some(c) = (match kp.key {
            Key::Char(c) => Some(c),
            _ => None,
        }) else {
            return LoopAction::Continue;
        };
        let feed = self.hop.as_mut().map(|h| h.feed(c));
        match feed {
            Some(HopFeed::Narrowed) => LoopAction::Redraw,
            Some(HopFeed::Jump(pos)) => {
                self.hop = None;
                self.host.set_active(self.host.active_buffer_id(), pos);
                self.sync_active_window();
                LoopAction::Redraw
            }
            Some(HopFeed::Cancel) | None => {
                self.hop = None;
                LoopAction::Redraw
            }
        }
    }

    /// Renders one full frame: overlay (if any), window(s), statusline, and
    /// command line.
    ///
    /// The overlay's rectangle is carved out of the windows' area *first*, so
    /// [`WindowTree`] lays out splits inside whatever is left and never knows the
    /// sidebar exists. It is *painted* last, so that a floating overlay (the
    /// pickers, when they land) draws over the text rather than under it.
    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(LayoutDirection::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1), Constraint::Length(1)])
            .split(area);
        let (full_windows_area, statusline_area, cmdline_area) = (chunks[0], chunks[1], chunks[2]);

        let (overlay_area, windows_area) = match &self.overlay {
            Some(overlay) => {
                let (o, w) = overlay.placement().split(full_windows_area);
                (Some(o), w)
            }
            None => (None, full_windows_area),
        };

        self.render_windows(frame, windows_area);
        self.render_statusline(frame, statusline_area);
        self.render_cmdline(frame, cmdline_area);
        if let Some(rect) = overlay_area {
            self.render_overlay(frame, rect);
        }
        // which-key sits on top of everything else: it is a heads-up display
        // that appears the moment a multi-key prefix is buffered. Suppressed
        // while an overlay or hop owns the keyboard (their own keys are not the
        // editor's keymaps) or a `:` prompt is open.
        self.render_which_key(frame, windows_area);
        self.render_lsp_popups(frame, windows_area);
        // The completion popup sits on top of everything, anchored at the cursor
        // captured during `render_windows`.
        self.render_completion_menu(frame, windows_area);
    }

    /// Draws the insert-mode completion popup at the cursor, when one is open.
    /// A passive render pass — key handling lives in
    /// [`Self::completion_intercept`]. See [`crate::ui::completion_menu`].
    fn render_completion_menu(&self, frame: &mut Frame, area: Rect) {
        let Some(menu) = &self.completion else { return };
        let Some(cursor) = self.last_cursor_screen else { return };
        if menu.items.is_empty() {
            return;
        }
        let visible = menu.items.len().min(MAX_COMPLETION_ROWS);
        let width = CompletionMenuWidget::desired_width(&menu.items, area.width.min(64));
        let rect = menu_rect(area, cursor, visible, width);
        frame.render_widget(
            CompletionMenuWidget {
                items: &menu.items,
                selected: menu.selected,
                scroll: menu.scroll,
                theme: &self.theme,
            },
            rect,
        );
    }

    /// Draws whichever LSP popup is open: hover text, the references list, or
    /// the rename prompt. At most one is active at a time (they take focus
    /// mutually exclusively). See [`crate::ui::lsp_ui`].
    fn render_lsp_popups(&self, frame: &mut Frame, area: Rect) {
        if let Some(lines) = &self.lsp_hover {
            let rect = popup_rect_for(area, lines, 60, "hover");
            frame.render_widget(
                InfoBox { title: "hover", lines, selected: None, theme: &self.theme, scroll: 0 },
                rect,
            );
        } else if let Some(refs) = &self.lsp_refs {
            let lines: Vec<String> = refs
                .locations
                .iter()
                .map(|l| format!("{}:{}", display_path(&l.file), l.range.anchor))
                .collect();
            // Keep the selected row in view for a long list.
            let inner_h = popup_rect_for(area, &lines, 80, "references").height.saturating_sub(2) as usize;
            let scroll = refs.selected.saturating_sub(inner_h.saturating_sub(1));
            let rect = popup_rect_for(area, &lines, 80, "references");
            frame.render_widget(
                InfoBox { title: "references", lines: &lines, selected: Some(refs.selected), theme: &self.theme, scroll },
                rect,
            );
        } else if let Some(r) = &self.lsp_rename {
            let lines = vec![r.input.clone()];
            let rect = centered_rect(area, 40, 3);
            frame.render_widget(
                InfoBox { title: "rename to (Enter to apply, Esc to cancel)", lines: &lines, selected: None, theme: &self.theme, scroll: 0 },
                rect,
            );
        }
    }

    /// Draws the which-key popup when the editor has a key prefix pending.
    ///
    /// Passive: reads [`EditorHost::which_key`] and paints; it never consumes a
    /// key. See [`crate::ui::whichkey`].
    fn render_which_key(&self, frame: &mut Frame, windows_area: Rect) {
        if self.focus() == Focus::Overlay || self.hop.is_some() || self.host.command_line().is_some() {
            return;
        }
        let rows = self.host.which_key();
        if rows.is_empty() {
            return;
        }
        let rect = crate::ui::whichkey::popup_rect(windows_area, rows.len());
        frame.render_widget(crate::ui::whichkey::WhichKey { rows: &rows, theme: &self.theme }, rect);
    }

    fn render_overlay(&mut self, frame: &mut Frame, rect: Rect) {
        let focused = self.focus() == Focus::Overlay;
        // `Theme` and `IconSet` are `Copy`, so taking them out first lets the
        // overlay borrow `self.overlay` mutably (it needs to recompute its scroll
        // against the height it is finally being drawn at) without fighting the
        // borrow checker over disjoint fields of `self`.
        let (theme, icons) = (self.theme, self.icons);
        if let Some(overlay) = self.overlay.as_mut()
            && let Some(position) = overlay.render(frame, rect, &theme, icons, focused)
        {
            frame.set_cursor_position(position);
        }
    }

    fn render_windows(&mut self, frame: &mut Frame, area: Rect) {
        // Remembered so between-frame commands (spatial `<C-w>l`, hop) can be
        // resolved against real geometry (see `last_windows_area`'s docs).
        self.last_windows_area = area;

        let active_id = self.windows.active_id();
        let line_numbers =
            LineNumberMode::from_options(self.options.number, self.options.relativenumber);
        let layout = self.windows.layout(area);
        // A `Copy` snapshot of each window's saved state, so the loop can read
        // any window's buffer/cursor/scroll without re-borrowing the tree while
        // it also needs a mutable borrow for the active window's scroll.
        let windows: Vec<crate::ui::window::Window> =
            self.windows.windows().into_iter().copied().collect();

        // Tell the editor the active window's text height, so `<C-d>`/`<C-f>`
        // scroll by the right fraction of *this* window, not the whole screen.
        if let Some((_, rect)) = layout.iter().find(|(id, _)| *id == active_id) {
            self.host.set_viewport_height(rect.height as usize);
        }

        // The active cursor's screen cell, captured so the completion popup (a
        // later render pass) can anchor at it. Reset each frame.
        let mut cursor_screen: Option<(u16, u16)> = None;

        for (id, rect) in layout {
            let is_active = id == active_id;
            let Some(win) = windows.iter().find(|w| w.id == id).copied() else { continue };
            let buffer_id = win.buffer;
            // Each window shows ITS buffer with ITS cursor — the fix for
            // `kopitiam-cj0.10.3`. The active window's cursor is the editor's
            // live one; an inactive split's is whatever it last left behind.
            let cursor = if is_active { self.host.cursor() } else { win.cursor };

            let buffer_lines = self.host.buffer_by_id(buffer_id).map(BufferView::line_count).unwrap_or(1);
            let gutter_w = crate::ui::gutter::gutter_width(buffer_lines, line_numbers);
            let text_height = rect.height as usize;
            let text_width = rect.width.saturating_sub(gutter_w) as usize;

            // Only the active window's scroll follows the live cursor; an
            // inactive split keeps whatever scroll it last had (matching vim:
            // switching away from a window doesn't move its view).
            let scroll = if is_active {
                // Read what the scroll math needs from the buffer, as owned
                // values, so the immutable buffer borrow ends before the
                // mutable `active_mut()` write below.
                let (line, line_count) = {
                    let Some(buf) = self.host.buffer_by_id(buffer_id) else { continue };
                    (buf.line(cursor.line).unwrap_or_default(), buf.line_count())
                };
                let top = scrolling::vertical_scroll(
                    cursor.line,
                    line_count,
                    text_height,
                    self.options.scrolloff,
                    win.scroll.top,
                );
                let display_col = crate::ui::textarea::display_col_of_grapheme(
                    &line,
                    cursor.col,
                    self.options.tabstop,
                );
                let left = scrolling::horizontal_scroll(display_col, text_width, win.scroll.left);
                let s = Scroll { top, left };
                self.windows.active_mut().scroll = s;
                s
            } else {
                win.scroll
            };

            let Some(buffer) = self.host.buffer_by_id(buffer_id) else { continue };
            // Syntax highlighting is derived from the buffer's file extension,
            // gated on `vim.opt.syntax` (the maintainer's config leaves it on).
            // An unrecognised extension yields `None`, i.e. plain text — the
            // renderer never guesses a grammar. See `kopitiam-syntax`.
            let language = self
                .options
                .syntax
                .then(|| {
                    buffer
                        .path()
                        .and_then(|p| p.extension())
                        .and_then(|e| e.to_str())
                        .and_then(kopitiam_syntax::Language::from_extension)
                })
                .flatten();
            let text_area = TextArea {
                buffer,
                cursor,
                mode: if is_active { self.host.mode() } else { crate::core::Mode::Normal },
                scroll,
                line_numbers,
                colorcolumn: self.options.colorcolumn,
                tabstop: self.options.tabstop,
                theme: &self.theme,
                // Only the focused window has a live selection.
                selection: is_active
                    .then(|| {
                        self.host.selection().map(|(start, end)| Selection {
                            start,
                            end,
                            mode: self.host.mode(),
                        })
                    })
                    .flatten(),
                language,
            };

            // The terminal cursor belongs to whatever has focus. With the file
            // tree focused it sits on the selected row (see `render_overlay`),
            // while a `:` command is being typed it belongs on the command
            // line, and during a hop it stays hidden while the labels show.
            if is_active
                && self.focus() == Focus::Buffer
                && self.hop.is_none()
                && self.lsp_rename.is_none()
                && self.lsp_refs.is_none()
                && self.host.command_line().is_none()
                && let Some((x, y)) = text_area.cursor_screen_position(rect)
            {
                frame.set_cursor_position((x, y));
                cursor_screen = Some((x, y));
            }

            frame.render_widget(text_area, rect);

            // Diagnostics (gutter signs + underlines) paint over the text and
            // syntax pass on the active window. Under the hop labels.
            if is_active {
                self.paint_diagnostics(frame, rect, gutter_w, scroll, buffer_id);
            }

            // Hop labels overlay the active window's word-starts, painted last
            // so they sit on top of the text.
            if is_active && self.hop.is_some() {
                self.paint_hop_labels(frame, rect, gutter_w, scroll, buffer_id);
            }
        }

        self.last_cursor_screen = cursor_screen;
    }

    /// Paints the active hop's labels onto the buffer's word-starts, at the
    /// exact screen cells those `(line, column)` positions occupy — the reason
    /// hop is drawn here and not through the geometry-free `Overlay` seam (see
    /// [`crate::ui::hop`]).
    fn paint_hop_labels(&self, frame: &mut Frame, rect: Rect, gutter_w: u16, scroll: Scroll, buffer_id: BufferId) {
        let Some(hop) = &self.hop else { return };
        let text_x = rect.x + gutter_w;
        let text_width = rect.width.saturating_sub(gutter_w) as usize;
        let style = Style::default()
            .fg(self.theme.red_bright)
            .bg(self.theme.bg)
            .add_modifier(Modifier::BOLD);
        let buf = frame.buffer_mut();
        for hint in hop.visible() {
            let pos = hint.position;
            let Some(row) = pos.line.checked_sub(scroll.top) else { continue };
            if row >= rect.height as usize {
                continue;
            }
            let line = self.host.buffer_by_id(buffer_id).and_then(|b| b.line(pos.line)).unwrap_or_default();
            let display_col = crate::ui::textarea::display_col_of_grapheme(&line, pos.col, self.options.tabstop);
            let Some(col) = display_col.checked_sub(scroll.left) else { continue };
            if col + hint.label.chars().count() > text_width {
                continue;
            }
            let y = rect.y + row as u16;
            let x = text_x + col as u16;
            buf.set_stringn(x, y, &hint.label, hint.label.len(), style);
        }
    }

    fn render_statusline(&self, frame: &mut Frame, area: Rect) {
        let buffer = self.host.buffer();
        let data = StatuslineData {
            mode: self.host.mode(),
            file_name: display_file_name(buffer.path()),
            modified: buffer.is_modified(),
            filetype: filetype_from_path(buffer.path()),
            // Populated by the plugin layer's git integration once it
            // exists; see `ui/statusline.rs`'s module docs.
            git_branch: None,
            cursor: self.host.cursor(),
            line_count: buffer.line_count(),
        };
        let statusline = Statusline { data: &data, theme: &self.theme, glyphs: self.glyphs() };
        frame.render_widget(statusline, area);
    }

    /// The bottom row's state for this frame, **derived** from the editor rather
    /// than mirrored into a field the app has to remember to update.
    ///
    /// While the editor is in `Mode::Command`, `host.command_line()` is the one
    /// and only source of what the user has typed, and it is asked afresh every
    /// frame. That is what makes `:Neotree` appear character by character; the
    /// previous design had a private copy that nothing wrote to, and so showed a
    /// bare `:` forever.
    ///
    /// The cursor is pinned to the end of the typed text: the editor's
    /// `handle_command_key` only ever appends or backspaces (there is no
    /// left/right movement within the command line yet), so "the end" is not an
    /// approximation — it is where the editor's insertion point genuinely is. If
    /// the editor grows command-line motions, this is the line that has to learn
    /// about them, and `CmdlineState::cursor` is already grapheme-indexed and
    /// ready for it.
    fn cmdline_state(&self) -> CmdlineState {
        match self.host.command_line() {
            // The prompt kind (`:` vs `/` vs `?`) now comes from the editor via
            // `command_prompt()`: search landed, so this is the line that grew,
            // exactly as the previous comment here predicted it would.
            Some(input) => CmdlineState {
                kind: self.host.command_prompt(),
                cursor: input.graphemes(true).count(),
                input: input.to_string(),
                message: StatusMessage::None,
            },
            None => CmdlineState {
                kind: PromptKind::None,
                input: String::new(),
                cursor: 0,
                message: self.message.clone(),
            },
        }
    }

    fn render_cmdline(&self, frame: &mut Frame, area: Rect) {
        let state = self.cmdline_state();
        let cmdline = Cmdline { state: &state, theme: &self.theme };
        if state.kind != PromptKind::None
            && let Some(x) = cmdline.cursor_column(area)
        {
            frame.set_cursor_position((x, area.y));
        }
        frame.render_widget(cmdline, area);
    }
}

/// The name shown in the statusline's file segment: the file's base name,
/// or vim's `[No Name]` for a buffer with no backing file.
fn display_file_name(path: Option<&Path>) -> String {
    path.and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| "[No Name]".to_string())
}

/// The gruvbox colour a diagnostic severity paints in — for its gutter sign and
/// its underline. Matches editor convention: errors red, warnings yellow,
/// info/hints in the cooler hues.
fn severity_color(severity: crate::lsp::Severity, theme: &Theme) -> ratatui::style::Color {
    use crate::lsp::Severity;
    match severity {
        Severity::Error => theme.red_bright,
        Severity::Warning => theme.yellow_bright,
        Severity::Information => theme.blue_bright,
        Severity::Hint => theme.aqua_bright,
    }
}

/// Paint/precedence rank for a severity — higher wins a shared cell. Errors
/// outrank warnings outrank info outrank hints.
fn severity_priority(severity: crate::lsp::Severity) -> u8 {
    use crate::lsp::Severity;
    match severity {
        Severity::Hint => 0,
        Severity::Information => 1,
        Severity::Warning => 2,
        Severity::Error => 3,
    }
}

/// The one-character gutter sign for a diagnostic severity.
fn severity_sign(severity: crate::lsp::Severity) -> &'static str {
    use crate::lsp::Severity;
    match severity {
        Severity::Error => "E",
        Severity::Warning => "W",
        Severity::Information => "I",
        Severity::Hint => "H",
    }
}

/// The language-server filetype for `path`, or `None` when kvim has no server
/// registered for it.
///
/// `Cargo.toml` maps to `"rust"` on purpose: rust-analyzer serves `Cargo.toml`
/// (completion, diagnostics) as well as `.rs`, and `taplo` — the dedicated TOML
/// server — is not always installed. This is the "taplo, or rust-analyzer if
/// taplo absent" routing the finisher brief calls for, resolved statically to
/// rust-analyzer since taplo is not in the registry.
fn lsp_filetype(path: &Path) -> Option<String> {
    if path.file_name().and_then(|n| n.to_str()) == Some("Cargo.toml") {
        return Some("rust".to_string());
    }
    let ext = path.extension().and_then(|e| e.to_str())?;
    match ext {
        "rs" => Some("rust"),
        "lua" => Some("lua"),
        "tex" | "sty" | "cls" => Some("tex"),
        _ => None,
    }
    .map(str::to_string)
}

/// The identifier grapheme-substring of `line` surrounding grapheme column
/// `col`, used to seed the rename prompt. Word characters are alphanumerics and
/// `_`; anything else bounds the word.
fn word_under_cursor(line: &str, col: usize) -> String {
    let graphemes: Vec<&str> = line.graphemes(true).collect();
    let is_word = |g: &str| g.chars().all(|c| c.is_alphanumeric() || c == '_') && !g.is_empty();
    if graphemes.is_empty() {
        return String::new();
    }
    let clamped = col.min(graphemes.len().saturating_sub(1));
    if !is_word(graphemes[clamped]) {
        return String::new();
    }
    let mut start = clamped;
    while start > 0 && is_word(graphemes[start - 1]) {
        start -= 1;
    }
    let mut end = clamped;
    while end + 1 < graphemes.len() && is_word(graphemes[end + 1]) {
        end += 1;
    }
    graphemes[start..=end].concat()
}

/// The identifier prefix immediately before grapheme column `col` on `line`:
/// the trailing run of word graphemes (alphanumerics and `_`), and the column
/// it starts at. This is the token the completion menu filters on and, on
/// accept, overwrites. An empty run (the char before the cursor is not a word
/// char) yields `(col, "")`, which auto-closes the menu.
fn identifier_prefix(line: &str, col: usize) -> (usize, String) {
    let graphemes: Vec<&str> = line.graphemes(true).collect();
    let is_word = |g: &str| g.chars().all(|c| c.is_alphanumeric() || c == '_') && !g.is_empty();
    let end = col.min(graphemes.len());
    let mut start = end;
    while start > 0 && is_word(graphemes[start - 1]) {
        start -= 1;
    }
    (start, graphemes[start..end].concat())
}

/// Converts a semantic-layer [`CompletionItem`](kopitiam_semantic::CompletionItem)
/// (an LSP result) into the headless engine's [`CItem`], tagging it
/// [`CompletionSource::Lsp`], carrying its kind for the menu badge, and routing
/// a snippet item (`insertTextFormat == 2`) through the snippet path by keeping
/// its raw grammar in [`CItem::snippet`].
fn convert_completion_item(item: kopitiam_semantic::CompletionItem) -> CItem {
    let insert = item.insert_text.clone().unwrap_or_else(|| item.label.clone());
    let mut converted = CItem::new(item.label, CompletionSource::Lsp);
    converted.insert_text = insert.clone();
    converted.detail = item.detail;
    converted.kind = item.kind;
    if item.is_snippet {
        // A snippet item's `insert_text` is snippet grammar (`greet($0)`), to be
        // expanded on accept — never inserted verbatim.
        converted.snippet = Some(insert);
    }
    converted
}

/// A short display form for a location's path: the file name plus its parent
/// directory, so a references list reads `src/lib.rs` rather than an absolute
/// path that overflows the popup.
fn display_path(path: &Path) -> String {
    let file = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
    match path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()) {
        Some(parent) => format!("{parent}/{file}"),
        None => file.to_string(),
    }
}

/// The bottom-anchored rectangle a text popup of `lines` occupies: wide enough
/// for the longest line (capped at `max_width`), tall enough for every line
/// (capped to the area), and centred. `title` widens the minimum so the title
/// is never clipped.
fn popup_rect_for(area: Rect, lines: &[String], max_width: u16, title: &str) -> Rect {
    let longest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let width = (longest.max(title.len()) as u16 + 4).min(max_width);
    let height = lines.len() as u16 + 2;
    centered_rect(area, width, height)
}

/// A minimal, dependency-free filetype guess from a file extension, for the
/// statusline's filetype segment. This is deliberately not the real
/// filetype-detection engine (that belongs with the LSP/syntax layer, which
/// needs to look at shebangs and content, not just extensions) — it exists
/// only so the statusline shows *something* recognisable today.
fn filetype_from_path(path: Option<&Path>) -> String {
    let Some(ext) = path.and_then(|p| p.extension()).and_then(|e| e.to_str()) else {
        return String::new();
    };
    match ext {
        "rs" => "rust",
        "lua" => "lua",
        "tex" => "tex",
        "md" => "markdown",
        "toml" => "toml",
        "json" => "json",
        other => return other.to_string(),
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Mode, Position};
    use crate::ui::event::{Key, KeyPress, Modifiers};
    use crate::ui::test_support::{FakeBuffer, FakeHost};
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
    use ratatui::backend::TestBackend;

    fn app_with(lines: Vec<&str>) -> App<FakeHost> {
        let buffer = FakeBuffer::new(lines.into_iter().map(str::to_string).collect());
        let host = FakeHost::new(buffer);
        // The ASCII tier throughout: `needs_font()` is false, so this is also the
        // old `glyphs: false`, and every rendered icon is assertable on any
        // terminal.
        App::new(host, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ')
    }

    fn key_event(c: char) -> Event {
        Event::Key(KeyEvent {
            code: KeyCode::Char(c),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    /// A fixture directory, and an app whose file tree roots at it.
    fn app_with_tree() -> (tempfile::TempDir, App<FakeHost>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.path().join("README.md"), "# hi\nsecond line\n").unwrap();

        let mut app = app_with(vec!["a", "b", "c"]);
        app.tree_root = dir.path().to_path_buf();
        (dir, app)
    }

    /// Presses `<leader>e` the way the real editor reports it: the key reaches
    /// the host, whose keymap engine resolves it to an [`Action`].
    fn press_leader_e(app: &mut App<FakeHost>) -> LoopAction {
        app.host.answer_next_with(HostResponse::Action(Action::FileTreeToggle));
        app.handle_event(key_event('e'))
    }

    /// The painted screen, one string per row.
    fn screen(app: &mut App<FakeHost>, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buf.cell((x, y)).unwrap().symbol().to_string())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn a_diagnostic_paints_a_gutter_sign_and_underlines_its_range() {
        use crate::core::Range;
        use crate::lsp::{Diagnostic, Severity};
        use ratatui::style::Modifier;

        let buffer = FakeBuffer::new(vec!["let x = 1;".to_string()]).with_path("/tmp/x.rs");
        let host = FakeHost::new(buffer);
        let mut app = App::new(host, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ');
        let theme = app.theme;
        // A diagnostic over the identifier `x` (grapheme cols 4..=4).
        app.diagnostics.insert(
            PathBuf::from("/tmp/x.rs"),
            vec![Diagnostic {
                range: Range::new(Position::new(0, 4), Position::new(0, 5)),
                severity: Severity::Error,
                message: "cannot find value `x`".to_string(),
                source: Some("rustc".to_string()),
            }],
        );

        let mut terminal = Terminal::new(TestBackend::new(20, 3)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();

        // Gutter width for a 1-line buffer is 2 ("1" + pad); the error sign
        // overwrites the gutter's first column in red.
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "E", "gutter should show the error sign");
        assert_eq!(buf.cell((0, 0)).unwrap().style().fg, Some(theme.red_bright));
        // Gutter width has a 3-digit floor + 1 pad = 4, so `x` (display column
        // 4) sits at screen column 4+4 = 8, underlined in the error colour with
        // the character itself intact.
        let xc = buf.cell((8, 0)).unwrap();
        assert_eq!(xc.symbol(), "x");
        assert!(xc.style().add_modifier.contains(Modifier::UNDERLINED), "the flagged range must be underlined");
        assert_eq!(xc.style().underline_color, Some(theme.red_bright));
    }

    // ---- insert-mode completion menu (cj0.17) -------------------------

    /// A `.rs` buffer already in Insert mode with the cursor at `cursor` — the
    /// state the completion menu is driven from. The `.rs` path makes
    /// [`lsp_filetype`] report `rust`, so the built-in snippet source is active
    /// (no live server is spawned in a unit test — `server_available` is false —
    /// so the LSP source is simply empty, exactly as it degrades in the field).
    fn insert_app(lines: Vec<&str>, cursor: Position) -> App<FakeHost> {
        let buffer = FakeBuffer::new(lines.into_iter().map(str::to_string).collect()).with_path("/tmp/x.rs");
        let mut host = FakeHost::new(buffer);
        host.mode = Mode::Insert;
        host.cursor = cursor;
        App::new(host, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ')
    }

    fn ctrl(c: char) -> KeyPress {
        KeyPress::new(Key::Char(c), Modifiers { ctrl: true, alt: false, shift: false })
    }

    #[test]
    fn completion_menu_paints_buffer_word_candidates() {
        // Typing `va` on line 1; `value`/`valiant` are words already in the
        // buffer and must appear as candidates — asserted on the painted cells.
        let mut app = insert_app(vec!["value valiant", "va"], Position::new(1, 2));
        assert!(app.refresh_completion(false), "typing an identifier opens the menu");
        let joined = screen(&mut app, 60, 12).join("\n");
        assert!(joined.contains("value"), "buffer word `value` must be painted:\n{joined}");
        assert!(joined.contains("valiant"), "buffer word `valiant` must be painted:\n{joined}");
    }

    #[test]
    fn completion_menu_surfaces_builtin_snippets_with_a_snippet_badge() {
        // Prefix `f` on a rust buffer surfaces the built-in `fn` snippet, badged
        // `snp` so the source reads at a glance.
        let mut app = insert_app(vec!["", "f"], Position::new(1, 1));
        assert!(app.refresh_completion(false));
        let joined = screen(&mut app, 60, 14).join("\n");
        assert!(joined.contains("fn"), "the `fn` snippet must be painted:\n{joined}");
        assert!(joined.contains("snp"), "a snippet-source badge `snp` must be painted:\n{joined}");
    }

    #[test]
    fn accepting_a_candidate_replaces_the_typed_prefix() {
        let mut app = insert_app(vec!["value", "val"], Position::new(1, 3));
        assert!(app.refresh_completion(false));
        let expected = app.completion.as_ref().unwrap().items[0].insert_text.clone();
        app.accept_completion();
        assert_eq!(
            app.host.buffer.line(1).unwrap(),
            expected,
            "accepting must overwrite the typed prefix `val` with the item's insert text"
        );
        assert!(app.completion.is_none(), "accepting closes the menu");
    }

    #[test]
    fn accepting_an_lsp_snippet_expands_rather_than_inserting_dollar_zero() {
        // The (d) proof at integration level: an LSP snippet item (its body is
        // grammar) must be run through the expander, never typed verbatim — so
        // `greet($0)` becomes `greet(...)`, and the literal `$0` never lands in
        // the buffer. (Until the real `kopitiam-snippet` engine lands, its stub
        // strips the tabstop to `greet()`; the assertion holds either way.)
        let mut app = insert_app(vec!["gr"], Position::new(0, 2));
        let item = CItem::new("greet", CompletionSource::Lsp).with_snippet("greet($0)");
        app.completion = Some(CompletionMenu {
            items: vec![item],
            selected: 0,
            scroll: 0,
            anchor: Position::new(0, 0),
            explicit: true,
        });
        app.accept_completion();
        let line = app.host.buffer.line(0).unwrap();
        assert!(line.contains("greet("), "the snippet must expand into the buffer: {line:?}");
        assert!(!line.contains("$0"), "the literal `$0` must never be inserted: {line:?}");
    }

    #[test]
    fn ctrl_n_advances_the_selection() {
        let mut app = insert_app(vec!["alpha album", "al"], Position::new(1, 2));
        assert!(app.refresh_completion(false));
        assert!(app.completion.as_ref().unwrap().items.len() >= 2, "need two candidates to move between");
        let before = app.completion.as_ref().unwrap().selected;
        app.completion_intercept(ctrl('n'));
        let after = app.completion.as_ref().unwrap().selected;
        assert_ne!(before, after, "<C-n> must move the highlighted candidate");
    }

    #[test]
    fn typing_auto_opens_and_ctrl_e_dismisses_the_menu() {
        // Drives the real event path (not the internal refresh helper): a typed
        // identifier char opens the menu; `<C-e>` cancels it, staying in insert
        // mode with nothing accepted.
        let mut app = insert_app(vec!["value", ""], Position::new(1, 0));
        let typed = Event::Key(KeyEvent {
            code: KeyCode::Char('v'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });
        assert_eq!(app.handle_event(typed), LoopAction::Redraw);
        assert!(app.completion.is_some(), "typing an identifier char auto-opens the menu");
        assert_eq!(app.host.buffer.line(1).unwrap(), "v", "and the character is inserted");

        let cancel = Event::Key(KeyEvent {
            code: KeyCode::Char('e'),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });
        app.handle_event(cancel);
        assert!(app.completion.is_none(), "<C-e> dismisses the menu without inserting anything");
        assert_eq!(app.host.mode(), Mode::Insert, "and stays in insert mode");
    }

    #[test]
    fn leaving_insert_mode_closes_the_menu() {
        let mut app = insert_app(vec!["value", "v"], Position::new(1, 1));
        assert!(app.refresh_completion(false));
        assert!(app.completion.is_some());
        // <Esc> leaves insert mode; the menu must go with it.
        let esc = Event::Key(KeyEvent {
            code: KeyCode::Esc,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });
        app.handle_event(esc);
        assert!(app.completion.is_none(), "leaving insert mode closes the completion menu");
    }

    #[test]
    fn quit_key_stops_the_loop() {
        let mut app = app_with(vec!["a", "b"]);
        assert_eq!(app.handle_event(key_event('q')), LoopAction::Quit);
    }

    #[test]
    fn movement_key_redraws_and_updates_the_active_windows_cursor() {
        let mut app = app_with(vec!["a", "b", "c"]);
        assert_eq!(app.handle_event(key_event('j')), LoopAction::Redraw);
        assert_eq!(app.windows.active().cursor, Position::new(1, 0));
    }

    #[test]
    fn unmapped_key_does_not_request_a_redraw() {
        let mut app = app_with(vec!["a"]);
        // 'z' has no binding in FakeHost's script.
        assert_eq!(app.handle_event(key_event('z')), LoopAction::Continue);
    }

    #[test]
    fn resize_event_requests_a_redraw() {
        let mut app = app_with(vec!["a"]);
        assert_eq!(app.handle_event(Event::Resize(80, 24)), LoopAction::Redraw);
    }

    #[test]
    fn render_does_not_panic_on_a_full_frame() {
        let mut app = app_with(vec!["fn main() {}", "let x = 1;"]);
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        // Statusline row is the second-to-last row; it should mention the
        // mode label somewhere.
        let statusline_y = 22;
        let row: String =
            (0..80).map(|x| buffer.cell((x, statusline_y)).unwrap().symbol().to_string()).collect();
        assert!(row.contains("NORMAL"));
    }

    /// The regression test for **"typing `:Neotree` showed nothing"**.
    ///
    /// Note what it asserts: not that some state field holds `"Neotree"`, but
    /// that the characters are *on the screen*. The 305 tests that existed when
    /// this bug shipped all did the former, and the bug made every one of them
    /// pass. The command line is a view of the editor's state, so the only test
    /// that can catch it going missing is one that reads the painted cells.
    #[test]
    fn typing_an_ex_command_echoes_it_on_the_command_line() {
        let mut app = app_with(vec!["hello"]);
        app.host.mode = Mode::Command;
        app.host.command_line = Some("Neotree".to_string());

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        let cmdline_y = 23;
        let row: String =
            (0..80).map(|x| buffer.cell((x, cmdline_y)).unwrap().symbol().to_string()).collect();

        assert!(row.starts_with(":Neotree"), "the command line painted {row:?}");
        // And the cursor is at the end of what was typed, where the next
        // character will land: ':' + 7 graphemes = column 8.
        assert_eq!(terminal.get_cursor_position().unwrap(), (8, 23).into());
    }

    #[test]
    fn the_command_line_grows_character_by_character_as_it_is_typed() {
        let mut app = app_with(vec!["hello"]);
        app.host.mode = Mode::Command;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        for (typed, expected) in [("", ":"), ("w", ":w"), ("wq", ":wq")] {
            app.host.command_line = Some(typed.to_string());
            terminal.draw(|frame| app.render(frame)).unwrap();
            let buffer = terminal.backend().buffer();
            let row: String =
                (0..80).map(|x| buffer.cell((x, 23)).unwrap().symbol().to_string()).collect();
            assert!(row.starts_with(expected), "after typing {typed:?} the row was {row:?}");
        }
    }

    #[test]
    fn leaving_command_mode_puts_the_cursor_back_in_the_buffer() {
        let mut app = app_with(vec!["hello"]);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        app.host.mode = Mode::Command;
        app.host.command_line = Some("wq".to_string());
        terminal.draw(|frame| app.render(frame)).unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap().y, 23, "cursor belongs on the command line");

        app.host.mode = Mode::Normal;
        app.host.command_line = None;
        terminal.draw(|frame| app.render(frame)).unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap().y, 0, "and back in the text afterwards");
    }

    #[test]
    fn filetype_is_derived_from_the_file_extension() {
        assert_eq!(filetype_from_path(Some(Path::new("src/main.rs"))), "rust");
        assert_eq!(filetype_from_path(Some(Path::new("Cargo.toml"))), "toml");
        assert_eq!(filetype_from_path(None), "");
    }

    #[test]
    fn no_name_shown_for_a_buffer_with_no_backing_file() {
        assert_eq!(display_file_name(None), "[No Name]");
        assert_eq!(display_file_name(Some(Path::new("/tmp/x/main.rs"))), "main.rs");
    }

    #[test]
    fn own_key_press_is_ignored_by_the_release_filter_end_to_end() {
        let mut app = app_with(vec!["a"]);
        let ev = Event::Key(KeyEvent {
            code: KeyCode::Char('j'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        });
        assert_eq!(app.handle_event(ev), LoopAction::Continue);
        assert_eq!(app.windows.active().cursor, Position::ORIGIN);
    }

    #[test]
    fn handle_key_records_the_translated_keypress_on_the_host() {
        let mut app = app_with(vec!["a", "b"]);
        app.handle_event(key_event('j'));
        assert_eq!(app.host.received, vec![KeyPress::plain(Key::Char('j'))]);
        assert_eq!(app.host.mode(), Mode::Normal);
    }

    // ------------------------------------------------------------------
    // The file tree sidebar
    // ------------------------------------------------------------------

    #[test]
    fn leader_e_opens_the_sidebar_focused_and_it_paints() {
        let (dir, mut app) = app_with_tree();
        assert_eq!(press_leader_e(&mut app), LoopAction::Redraw);
        assert!(matches!(app.overlay, Some(Overlay::FileTree(_))));
        assert_eq!(app.focus(), Focus::Overlay);

        let rows = screen(&mut app, 80, 12);
        let root = dir.path().file_name().unwrap().to_string_lossy().to_string();
        assert!(rows[0].starts_with(&format!("[-] {root}")), "row 0 = {:?}", rows[0]);
        assert!(rows[1].starts_with("  [+] src"), "row 1 = {:?}", rows[1]);
        assert!(rows[2].starts_with("  [md] README.md"), "row 2 = {:?}", rows[2]);
    }

    #[test]
    fn the_sidebar_reserves_columns_and_the_buffer_keeps_the_rest() {
        let (_dir, mut app) = app_with_tree();
        // Before: the buffer's gutter/text starts at column 0.
        let before = screen(&mut app, 80, 12);
        assert!(before[0].starts_with("  1 a") || before[0].contains('a'), "{:?}", before[0]);

        press_leader_e(&mut app);
        let after = screen(&mut app, 80, 12);
        // The buffer text has moved right by the sidebar's width; the sidebar's
        // divider sits at its right edge.
        assert_eq!(after[0].chars().nth(29), Some('|'), "row 0 = {:?}", after[0]);
        assert!(after[0][30..].contains('a'), "the buffer should still render, at {:?}", &after[0][30..]);
    }

    #[test]
    fn with_the_tree_focused_j_moves_the_tree_and_not_the_text_cursor() {
        let (_dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        let keys_before = app.host.received.len();

        assert_eq!(app.handle_event(key_event('j')), LoopAction::Redraw);

        // The half that is easy to get right: the tree moved.
        let rows = screen(&mut app, 80, 12);
        assert!(rows[1].starts_with("  [+] src"));
        // And the half that breaks silently: the editor never saw the key, so
        // neither the text cursor nor the active window's cursor moved.
        assert_eq!(app.host.received.len(), keys_before, "the editor must not see keys aimed at the tree");
        assert_eq!(app.host.cursor(), Position::ORIGIN);
        assert_eq!(app.windows.active().cursor, Position::ORIGIN);
    }

    #[test]
    fn with_the_tree_open_but_the_buffer_focused_j_moves_the_text_cursor_again() {
        let (dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        // Open a file: focus returns to the buffer, and the tree stays visible.
        app.handle_event(key_event('j')); // -> src
        app.handle_event(key_event('j')); // -> README.md
        app.handle_event(key_event('o'));
        assert_eq!(app.focus(), Focus::Buffer);
        assert!(app.overlay.is_some(), "neo-tree stays open after opening a file");
        assert_eq!(app.host.opened, vec![dir.path().join("README.md")]);

        // The tree is now inert: `j` is the editor's again.
        app.handle_event(key_event('j'));
        assert_eq!(app.host.cursor().line, 1);
        assert_eq!(app.windows.active().cursor.line, 1);
    }

    #[test]
    fn o_on_a_directory_expands_it_and_leaves_focus_in_the_tree() {
        let (_dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        app.handle_event(key_event('j')); // -> src
        app.handle_event(key_event('o'));

        assert_eq!(app.focus(), Focus::Overlay, "expanding a folder must not steal focus");
        assert!(app.host.opened.is_empty(), "a directory is not a file to open");
        let rows = screen(&mut app, 80, 12);
        assert!(rows[1].starts_with("  [-] src"), "{:?}", rows[1]);
        assert!(rows[2].starts_with("    [rs] main.rs"), "{:?}", rows[2]);
    }

    #[test]
    fn leader_e_toggles_open_then_closed_then_open_restoring_focus_and_layout() {
        let (_dir, mut app) = app_with_tree();
        let bare = screen(&mut app, 80, 12);

        // Open: the buffer has focus, so the *editor's* keymap engine resolves
        // `<leader>e` and hands back the action.
        press_leader_e(&mut app);
        assert_eq!(app.focus(), Focus::Overlay);

        // Close: the tree has focus, so the editor never sees these keys and the
        // *panel* resolves the same sequence. Two paths, one keystroke — which is
        // exactly why the panel has to know the leader.
        app.handle_event(key_event(' '));
        app.handle_event(key_event('e'));
        assert!(app.overlay.is_none());
        assert_eq!(app.focus(), Focus::Buffer, "closing must give focus back to the buffer");
        assert_eq!(screen(&mut app, 80, 12), bare, "closing must restore the previous layout exactly");

        // And open again.
        press_leader_e(&mut app);
        assert!(matches!(app.overlay, Some(Overlay::FileTree(_))));
        assert_eq!(app.focus(), Focus::Overlay);
    }

    #[test]
    fn escape_and_q_close_the_tree_from_inside_it() {
        for key in ['q', '\u{1b}'] {
            let (_dir, mut app) = app_with_tree();
            press_leader_e(&mut app);
            let ev = if key == 'q' {
                key_event('q')
            } else {
                Event::Key(KeyEvent {
                    code: KeyCode::Esc,
                    modifiers: KeyModifiers::NONE,
                    kind: KeyEventKind::Press,
                    state: KeyEventState::NONE,
                })
            };
            assert_eq!(app.handle_event(ev), LoopAction::Redraw);
            assert!(app.overlay.is_none(), "{key:?} should have closed the tree");
            assert_eq!(app.focus(), Focus::Buffer);
        }
    }

    #[test]
    fn leader_e_from_inside_the_tree_closes_it_and_the_editor_never_sees_the_keys() {
        let (_dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        let keys_before = app.host.received.len();

        app.handle_event(key_event(' '));
        assert_eq!(app.handle_event(key_event('e')), LoopAction::Redraw);
        assert!(app.overlay.is_none());
        assert_eq!(app.focus(), Focus::Buffer);
        assert_eq!(app.host.received.len(), keys_before);
    }

    #[test]
    fn i_and_s_open_the_file_in_a_split() {
        for (key, expected_windows) in [('i', 2), ('s', 2)] {
            let (dir, mut app) = app_with_tree();
            press_leader_e(&mut app);
            app.handle_event(key_event('j'));
            app.handle_event(key_event('j')); // -> README.md
            app.handle_event(key_event(key));

            assert_eq!(app.windows.windows().len(), expected_windows, "{key} should have split");
            assert_eq!(app.host.opened, vec![dir.path().join("README.md")]);
            assert_eq!(app.focus(), Focus::Buffer);
        }
    }

    #[test]
    fn t_opens_the_file_and_says_plainly_that_tab_pages_do_not_exist() {
        let (dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        app.handle_event(key_event('j'));
        app.handle_event(key_event('j')); // -> README.md
        app.handle_event(key_event('t'));

        assert_eq!(app.host.opened, vec![dir.path().join("README.md")]);
        assert_eq!(app.windows.windows().len(), 1, "no tab pages, so no new window either");
        match &app.message {
            StatusMessage::Info(m) => assert!(m.contains("tab pages"), "{m}"),
            other => panic!("expected an honest note, got {other:?}"),
        }
    }

    #[test]
    fn opening_a_file_that_cannot_be_read_reports_the_error_and_keeps_the_tree() {
        let (dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        app.handle_event(key_event('j'));
        app.handle_event(key_event('j')); // -> README.md
        std::fs::remove_file(dir.path().join("README.md")).unwrap();

        assert_eq!(app.handle_event(key_event('o')), LoopAction::Redraw);
        assert!(matches!(app.message, StatusMessage::Error(_)));
        assert!(app.overlay.is_some(), "a failed open must not close the tree");
    }

    #[test]
    fn the_tree_survives_a_terminal_resize() {
        let (_dir, mut app) = app_with_tree();
        press_leader_e(&mut app);
        assert_eq!(app.handle_event(Event::Resize(30, 8)), LoopAction::Redraw);

        // Squeezed to 30 columns, the sidebar clamps to two fifths (12) and the
        // buffer keeps the rest — no panic, and the tree is still legible.
        let rows = screen(&mut app, 30, 8);
        assert!(rows[0].starts_with("[-] "), "{:?}", rows[0]);
        assert_eq!(rows[0].chars().nth(11), Some('|'), "divider should be at the clamped edge");

        // And back out again.
        let rows = screen(&mut app, 200, 40);
        assert!(rows[0].starts_with("[-] "), "{:?}", rows[0]);
        assert_eq!(rows[0].chars().nth(29), Some('|'));
    }

    #[test]
    fn the_terminal_cursor_follows_focus_between_the_tree_and_the_buffer() {
        let (_dir, mut app) = app_with_tree();
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();

        // Buffer focused: the cursor is in the text, past the gutter.
        terminal.draw(|frame| app.render(frame)).unwrap();
        let in_buffer = terminal.get_cursor_position().unwrap();
        assert!(in_buffer.x >= 2, "cursor should be in the text, got {in_buffer:?}");

        // Tree focused: the cursor moves onto its selected row, at column 0 — the
        // sidebar's own left edge, not the buffer's.
        press_leader_e(&mut app);
        terminal.draw(|frame| app.render(frame)).unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap(), (0, 0).into());

        app.handle_event(key_event('j')); // move the tree cursor down one row
        terminal.draw(|frame| app.render(frame)).unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap(), (0, 1).into());
    }

    #[test]
    fn an_action_with_no_ui_yet_still_says_so_rather_than_being_swallowed() {
        // `HopWords` is now wired (see the hop tests below), so this uses a
        // still-unwired action to keep pinning the honest-message path.
        let mut app = app_with(vec!["a"]);
        app.host.answer_next_with(HostResponse::Action(Action::HarpoonMenu));
        assert_eq!(app.handle_event(key_event('x')), LoopAction::Redraw);
        match &app.message {
            StatusMessage::Info(m) => assert!(m.contains("not wired into the UI yet"), "{m}"),
            other => panic!("expected the honest message, got {other:?}"),
        }
        assert!(app.overlay.is_none());
    }

    // ------------------------------------------------------------------
    // Window management, driven through the REAL editor.
    //
    // These use `App<editor::Editor>` rather than the `FakeHost`, because the
    // whole point is the seam between the window tree and the editor's
    // per-buffer state — a fake with one buffer cannot reproduce the
    // `kopitiam-cj0.10.3` "both panes show the same text" bug, and asserting on
    // the PAINTED CELLS (not editor state) is the only assertion that catches
    // it. See the crate report.
    // ------------------------------------------------------------------

    use crate::editor::Editor;

    fn enter_event() -> Event {
        Event::Key(KeyEvent {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    fn ctrl_event(c: char) -> Event {
        Event::Key(KeyEvent {
            code: KeyCode::Char(c),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    /// Feed a run of plain characters.
    fn feed_str(app: &mut App<Editor>, s: &str) {
        for c in s.chars() {
            app.handle_event(key_event(c));
        }
    }

    /// The painted screen of a real-editor app, one string per row.
    fn real_screen(app: &mut App<Editor>, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| (0..width).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect::<String>())
            .collect()
    }

    /// An app over a real editor with `file_a` already open, plus a second file
    /// on disk (`file_b`) ready for `:vs` / `:e` to open. Line numbers are off
    /// so assertions read the raw text without a gutter offset.
    fn real_app_two_files() -> (tempfile::TempDir, App<Editor>, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, "AAAAAAAA\nAAAAAAAA\n").unwrap();
        std::fs::write(&b, "BBBBBBBB\nBBBBBBBB\n").unwrap();

        let mut editor = Editor::new();
        editor.open(&a).unwrap();
        let options = Options { number: false, relativenumber: false, ..Default::default() };
        let app = App::new(editor, options, Theme::gruvbox_dark(), IconSet::Ascii, ' ');
        (dir, app, b)
    }

    #[test]
    fn vsplit_opening_a_second_file_shows_two_different_buffers_side_by_side() {
        // THE regression test for kopitiam-cj0.10.3: two panes, two buffers,
        // asserted on the painted cells.
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());

        let rows = real_screen(&mut app, 40, 6);
        // Left pane is the new (active) window showing b.txt; right pane still
        // shows a.txt. The divider sits at the halfway column (~20).
        assert!(rows[0].starts_with("BBBBBBBB"), "left pane should show b.txt, got {:?}", rows[0]);
        assert!(rows[0][20..].contains("AAAAAAAA"), "right pane should still show a.txt, got {:?}", &rows[0][20..]);
        assert_eq!(app.windows.window_count(), 2);
    }

    #[test]
    fn ctrl_w_l_moves_focus_to_the_right_pane_and_typing_goes_there() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        // Focus starts in the LEFT pane (the new split, b.txt). The right pane
        // holds a.txt (buffer 1). Move focus right.
        let left_active = app.windows.active_id();
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('l'));
        assert_ne!(app.windows.active_id(), left_active, "focus should have moved to the other window");
        // The editor now edits a.txt: its buffer is the right pane's.
        assert_eq!(app.host.buffer().text(), "AAAAAAAA\nAAAAAAAA\n");

        // Typing `x` deletes a char in the RIGHT pane (a.txt), not the left.
        app.handle_event(key_event('x'));
        let rows = real_screen(&mut app, 40, 6);
        assert!(rows[0].starts_with("BBBBBBBB"), "left pane untouched, got {:?}", rows[0]);
        assert!(rows[0][20..].contains("AAAAAAA") && !rows[0][20..].contains("AAAAAAAA"),
            "right pane should have lost one A, got {:?}", &rows[0][20..]);
    }

    #[test]
    fn each_window_keeps_its_own_cursor_across_focus_switches() {
        let (_dir, mut app, _b) = real_app_two_files();
        // Split the same buffer so both windows show a.txt; move the cursor in
        // window A, switch to B, move there, switch back — A's cursor is where
        // it was left.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('s')); // horizontal split
        // Active is the new (top) window. Move its cursor down a line.
        app.handle_event(key_event('j'));
        let top_cursor = app.host.cursor();
        assert_eq!(top_cursor, Position::new(1, 0));

        // Switch to the other window, which is still at the origin.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('w'));
        assert_eq!(app.host.cursor(), Position::ORIGIN, "the other window kept its own cursor");

        // Back to the first: its cursor is preserved.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('w'));
        assert_eq!(app.host.cursor(), Position::new(1, 0), "the first window's cursor survived the round trip");
    }

    #[test]
    fn quitting_one_split_keeps_the_editor_open_but_the_last_quit_exits() {
        let (_dir, mut app, _b) = real_app_two_files();
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('s'));
        assert_eq!(app.windows.window_count(), 2);

        // `:q` closes the active split, not the editor.
        feed_str(&mut app, ":q");
        assert_eq!(app.handle_event(enter_event()), LoopAction::Redraw);
        assert_eq!(app.windows.window_count(), 1);

        // `:q` on the last window quits.
        feed_str(&mut app, ":q");
        assert_eq!(app.handle_event(enter_event()), LoopAction::Quit);
    }

    // ------------------------------------------------------------------
    // Hop (`f`), driven through the real editor + config keymap.
    // ------------------------------------------------------------------

    fn real_app_one_file(content: &str) -> (tempfile::TempDir, App<Editor>) {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        std::fs::write(&a, content).unwrap();
        let mut editor = Editor::new();
        editor.open(&a).unwrap();
        let options = Options { number: false, relativenumber: false, ..Default::default() };
        // Leader is space, matching the default config; `f` is the hop keymap.
        let app = App::new(editor, options, Theme::gruvbox_dark(), IconSet::Ascii, ' ');
        (dir, app)
    }

    #[test]
    fn pressing_f_paints_hop_labels_on_the_word_starts() {
        let (_dir, mut app) = real_app_one_file("alpha bravo charlie\n");
        app.handle_event(key_event('f'));
        assert!(app.hop.is_some(), "f should start a hop");

        let rows = real_screen(&mut app, 40, 3);
        // The first word-start gets the first label ('a'), painted over the
        // first cell of "alpha".
        assert_eq!(rows[0].chars().next(), Some('a'), "first label paints on the first word-start: {:?}", rows[0]);
        // A later word-start carries a different label than the underlying text.
        // "bravo" begins at column 6; its label is the 2nd alphabet letter 's'.
        assert_eq!(rows[0].chars().nth(6), Some('s'), "second word-start labelled, got {:?}", rows[0]);
    }

    #[test]
    fn typing_a_hop_label_jumps_the_cursor_to_that_word() {
        let (_dir, mut app) = real_app_one_file("alpha bravo charlie\n");
        app.handle_event(key_event('f'));
        // Labels: alpha='a', bravo='s', charlie='d' (home-row alphabet order).
        app.handle_event(key_event('s'));
        assert!(app.hop.is_none(), "a unique label ends the hop");
        assert_eq!(app.host.cursor(), Position::new(0, 6), "jumped to 'bravo'");
    }

    #[test]
    fn a_two_char_hop_label_narrows_then_jumps() {
        // Many word-starts force two-character labels. Twelve words: the label
        // alphabet's first letters are single, later ones become prefixes.
        let words = (0..30).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let (_dir, mut app) = real_app_one_file(&format!("{words}\n"));
        app.handle_event(key_event('f'));
        let hop = app.hop.as_ref().unwrap();
        let before = hop.visible().len();
        // Find a label that is two characters long and type its first char.
        let two: String = hop.visible().iter().find(|h| h.label.chars().count() == 2).unwrap().label.clone();
        let first = two.chars().next().unwrap();
        app.handle_event(key_event(first));
        assert!(app.hop.is_some(), "a 2-char label's first key narrows, not jumps");
        let after = app.hop.as_ref().unwrap().visible().len();
        assert!(after < before, "the candidate set should have shrunk ({before} -> {after})");
        // Typing the second char jumps.
        app.handle_event(key_event(two.chars().nth(1).unwrap()));
        assert!(app.hop.is_none(), "the full label jumps and ends the hop");
    }

    #[test]
    fn escape_during_a_hop_cancels_without_moving_the_cursor() {
        let (_dir, mut app) = real_app_one_file("alpha bravo charlie\n");
        let before = app.host.cursor();
        app.handle_event(key_event('f'));
        app.handle_event(Event::Key(KeyEvent {
            code: KeyCode::Esc,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }));
        assert!(app.hop.is_none(), "Esc ends the hop");
        assert_eq!(app.host.cursor(), before, "and the cursor did not move");
        // The labels are gone from the screen.
        let rows = real_screen(&mut app, 40, 3);
        assert!(rows[0].starts_with("alpha"), "the buffer text is back with no labels: {:?}", rows[0]);
    }

    #[test]
    fn operator_composed_f_still_finds_a_char_and_does_not_trigger_hop() {
        // The subtle case: idle `f` is hop, but `d f x` must stay find-char.
        let (_dir, mut app) = real_app_one_file("foo(bar)baz\n");
        app.handle_event(key_event('d'));
        app.handle_event(key_event('f'));
        // Still no hop — `f` was consumed by the pending `d` operator.
        assert!(app.hop.is_none(), "operator-composed f must not start a hop");
        app.handle_event(key_event('('));
        assert_eq!(app.host.buffer().text(), "bar)baz\n", "df( deleted through the paren");
    }
}
