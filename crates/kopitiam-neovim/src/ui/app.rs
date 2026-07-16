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
use std::time::{Duration, Instant};

use crossterm::event::{self, Event};
use unicode_segmentation::UnicodeSegmentation;
use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction as LayoutDirection, Layout, Rect},
    Frame, Terminal,
};

use crate::config::{Action, Options};
use crate::core::{BufferId, Direction, Mode, Position, Range, ViewportScroll, WindowCommand};
use crate::editor::ex::QuickfixCommand;
use crate::editor::quickfix::{ListKind, NavError, QuickfixEntry, QuickfixList};
use crate::icons::IconSet;
use crate::plugins::git::GitStatus;
use crate::plugins::grep;
use crate::plugins::harpoon::Harpoon;
use crate::plugins::picker::walk_files;
use crate::lsp::completion::{self, CompletionItem as CItem, CompletionSource};
use crate::lsp::{Location as LspLocation, LspClient};
use crate::ui::completion_menu::{anchored_rect, menu_rect, Anchor, CompletionMenu as CompletionMenuWidget};
use crate::ui::lsp_ui::{centered_rect, InfoBox};
use crate::ui::snippet::SnippetSession;
use crate::ui::cmdline::{Cmdline, CmdlineState, PromptKind, StatusMessage, Wildmenu};
use crate::ui::event::{map_crossterm_key, BufferView, EditorHost, HostResponse, Key, KeyPress};
use crate::ui::filetree::FileTreePanel;
use crate::ui::gutter::LineNumberMode;
use crate::ui::harpoon::HarpoonMenuPanel;
use crate::ui::hop::{HopFeed, HopState};
use crate::ui::overlay::{Focus, OpenTarget, Overlay, OverlayOutcome};
use crate::ui::picker::{PickAction, PickRow, PickerPanel};
use crate::ui::scrolling;
use crate::ui::statusline::{Statusline, StatuslineData};
use crate::ui::textarea::{Scroll, Selection, TextArea};
use crate::ui::theme::Theme;
use crate::ui::window::{Separator, SplitKind, WindowTree};

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
    /// A numeric prefix typed *between* `<C-w>` and its command letter — vim's
    /// `[count]` for the resize/exchange family (`<C-w>10>` widens by ten
    /// steps, `<C-w>2x` swaps with window two). Digits accumulate here while
    /// [`Self::awaiting_window_key`] stays armed; the command consumes and
    /// clears it. `None` means "no count given", which resize reads as 1 and
    /// exchange reads as "the next window", two genuinely different defaults.
    pending_window_count: Option<u32>,
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
    /// The global quickfix list, populated by `:grep`/`:vimgrep` and walked by
    /// `:cnext`/`:cprev`/`:cc`. Shares its "current entry" cursor with the
    /// quickfix window, so a `j`/`k` in the window and a `:cnext` move the same
    /// thing. See [`crate::editor::quickfix`].
    quickfix: QuickfixList,
    /// The location list — the window-local twin of the quickfix list, driven by
    /// the `:l`-prefixed commands (`:lgrep`, `:lopen`, `:lnext`, …).
    ///
    /// SCOPE: kvim keeps a *single* location list on the app for now, not one
    /// per window as vim does. For the common one-or-two-window session this is
    /// indistinguishable; the per-window twin is filed as a follow-up bead. See
    /// this session's report.
    location: QuickfixList,
    /// Which list's window is currently open at the bottom of the screen, if
    /// any. Only one is shown at a time (`:copen` and `:lopen` share the bottom
    /// strip); opening one closes the other. `None` means no list window is up.
    qf_window: Option<ListKind>,
    /// Whether the open list window currently has the keyboard: `j`/`k` move the
    /// selected entry, `<CR>` jumps to it, `q`/`<Esc>` closes. Set by `:copen`
    /// (vim drops you into the quickfix window) and cleared when a jump moves
    /// focus back to the buffer or the window closes. Kept separate from
    /// [`Self::qf_window`] because the window can be *visible* while the buffer
    /// has focus (after a `<CR>` jump), the same visible-but-inert state the file
    /// tree's [`Focus`] models.
    qf_focused: bool,
    /// The open insert-mode completion popup, if any. Driven by
    /// [`App::refresh_completion`] as the user types, navigated with
    /// `<C-n>`/`<C-p>`, accepted with `<CR>`/`<Tab>` — see
    /// [`App::completion_intercept`].
    completion: Option<CompletionMenu>,
    /// Set once the user press `<C-x>` in insert mode: kvim is now in vim's
    /// "CTRL-X mode", waiting for the sub-key that pick a native completion
    /// source (`<C-f>` filename, `<C-l>` whole line, `<C-o>` omni,
    /// `<C-n>`/`<C-p>` this-buffer keyword). Cleared the moment that sub-key
    /// arrive — a recognised one open the matching submenu, an unrecognised one
    /// just cancel CTRL-X mode and take its own normal meaning. See
    /// [`App::completion_intercept`].
    ctrl_x_pending: bool,
    /// The active snippet expansion being navigated with `<Tab>`/`<S-Tab>`, if
    /// any. Set when a snippet completion is accepted; cleared on `<Tab>` past
    /// the final tabstop or when insert mode ends. See [`crate::ui::snippet`].
    snippet: Option<SnippetSession>,
    /// The active buffer cursor's screen cell on the last frame, so the
    /// completion popup can be anchored at the cursor (the popup is a render
    /// pass with no live `Frame` geometry of its own — the same between-frame
    /// trick as [`Self::last_windows_area`]).
    last_cursor_screen: Option<(u16, u16)>,
    /// Whether kvim is running inside a tmux session, detected once at startup
    /// from `$TMUX` (tmux sets it for every process in a pane). Governs the
    /// edge hand-off in [`Self::tmux_select_pane`]: a `<C-h/j/k/l>` move that
    /// runs off kvim's own layout crosses into the neighbouring tmux pane only
    /// when this holds. Detected once, not re-read per keystroke, for the same
    /// reason [`crate::icons`] is: the environment does not change under a
    /// running process, and a syscall per focus key is waste.
    in_tmux: bool,
    /// A pending, unanswered offer to fix the user's tmux `is_vim` config, if
    /// kvim noticed the vim-tmux-navigator problem at startup (see
    /// [`crate::tmux`]). While `Some`, it is a modal consent popup: it owns the
    /// keyboard before anything else, paints over the buffer, and is dismissed
    /// only by `y` (apply the fix) or `n`/`Esc`/`q` (decline, and remember not
    /// to nag). Nothing is written to the user's dotfile until they press `y` —
    /// that is the whole safety point of routing this through a popup rather
    /// than editing on sight.
    tmux_prompt: Option<crate::tmux::TmuxOffer>,
    /// Test-only record of the `tmux select-pane` hand-offs that
    /// [`Self::tmux_select_pane`] *would* have spawned. A tmux hand-off is an
    /// external side effect with no painted cells to assert on, so under
    /// `cfg(test)` the direction is recorded here instead of shelling out — the
    /// one seam that lets a unit test prove the edge-of-layout code path fires
    /// when `$TMUX` is set. The real binary always spawns tmux.
    #[cfg(test)]
    tmux_calls: Vec<Direction>,
    /// The live Lua runtime, when the user's `init.lua` bound one or more
    /// keymaps to a Lua *function* (`vim.keymap.set("n", "x", function() ...
    /// end)`). Those closures live here — [`Action::LuaKeymap`] carries only an
    /// index into the runtime's callback registry, because a Lua closure is not
    /// serialisable and cannot ride inside [`Config`]. `None` when there is no
    /// Lua config or it bound no function keymaps. Set by
    /// [`crate::ui::run`] after the config has executed; a unit test that builds
    /// an `App` directly leaves it `None`.
    lua: Option<crate::luaconfig::LuaRuntime>,
    /// This project's harpoon marks (`<leader>b` marks, `<leader><Esc>` menu,
    /// `<leader>q` find). Session-scoped for now — [`Harpoon::empty`] reads and
    /// writes no store file, so marks live only for the editor's lifetime. On-disk
    /// per-project persistence (the engine already supports it via
    /// [`Harpoon::load`]/[`Harpoon::save`]) is a deliberate follow-up. See
    /// [`crate::plugins::harpoon`].
    harpoon: Harpoon,
    /// The cached git branch/dirty state shown in the statusline (the
    /// vim-fugitive/airline slice — see [`crate::plugins::git`]). `None` when
    /// the active file is not inside a git repository, so the statusline simply
    /// omits the segment. **Never recomputed per frame:** the branch read is
    /// cheap, but the dirty check walks the worktree, so this is refreshed on
    /// the event loop's idle tick (throttled by [`Self::GIT_STATUS_TTL`]) and
    /// immediately when the active buffer's directory changes — never inside
    /// [`Self::render_statusline`], which only reads it. See
    /// [`Self::refresh_git_status`].
    git_status: Option<GitStatus>,
    /// The directory [`Self::git_status`] was last computed for. A change here
    /// (switching to a buffer in a different repository) forces an immediate
    /// recompute, bypassing the TTL throttle.
    git_status_dir: Option<PathBuf>,
    /// When [`Self::git_status`] was last recomputed, so an idle tick can skip
    /// re-walking the worktree until [`Self::GIT_STATUS_TTL`] has elapsed.
    git_status_checked: Instant,
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

/// The most files `\ff` will walk before stopping. Ten thousand is far past a
/// screenful yet cheap to score per keystroke — see [`walk_files`] for why a
/// bounded, responsive list beats a complete, frozen one on a monorepo.
const FILE_PICKER_CAP: usize = 10_000;

/// How many content rows the cursor-anchored hover box will paint before it stop
/// growing — a long rust-analyzer doc string kena truncate instead of eating up the
/// whole window. After this, [`anchored_rect`] still cap it again to whatever room
/// got on the chosen side of the cursor.
const MAX_HOVER_ROWS: usize = 16;

/// How wide the hover box can go, in columns, before the lines kena clip. This stop
/// a long type signature from stretching the popup across the whole terminal.
const MAX_HOVER_WIDTH: u16 = 72;

/// Which native completion source is feeding an open menu — vim's insert-mode
/// completion "modes". It tells [`App::refresh_completion`] where to re-gather
/// candidates from as the user keeps typing, so a `<C-x><C-f>` filename menu
/// stays a filename menu instead of quietly turning back into the default
/// identifier menu on the next keystroke.
///
/// The distinction matter because kvim reuse the one completion popup for every
/// source: without remembering *which* source opened it, the as-you-type
/// refresh would always fall back to the default identifier/path logic, and the
/// `<C-x>` submodes would only survive exactly one keystroke.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionKind {
    /// The default `<C-Space>` / as-you-type menu: LSP + snippets + buffer words
    /// + path, chosen automatically from the cursor context.
    Auto,
    /// Vim keyword completion (`<C-n>`/`<C-p>`, and `<C-x><C-n>`/`<C-x><C-p>`):
    /// identifier-like words already in the buffers. `this_buffer_only` is the
    /// difference between the two — plain `<C-n>` scan the current *and* other
    /// window buffers (vim's default `complete` sources), while `<C-x><C-n>`
    /// scan only the current one.
    Keyword { this_buffer_only: bool },
    /// Filename completion (`<C-x><C-f>`): filesystem entries under the path
    /// fragment before the cursor, relative to the buffer's own directory.
    File,
    /// Whole-line completion (`<C-x><C-l>`): buffer lines that begin with the
    /// text already typed on the current line.
    Line,
    /// Omni completion (`<C-x><C-o>`): routed to the language server, i.e. the
    /// same `textDocument/completion` the default menu folds in, but on its own
    /// so LSP is the *only* source.
    Omni,
}

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
    /// Whether the menu was opened explicitly with `<C-Space>` (or a native
    /// `<C-n>`/`<C-x>...` trigger). An explicit menu survives the prefix
    /// emptying (so `<C-Space>` on nothing lists everything); an auto-triggered
    /// one closes when there is no longer a prefix to filter.
    explicit: bool,
    /// Which native source is feeding this menu, so the as-you-type refresh
    /// re-gathers from the right place. See [`CompletionKind`].
    kind: CompletionKind,
}

impl<H: EditorHost> App<H> {
    /// How long [`event::poll`] blocks before giving the loop a chance to
    /// notice `should_quit` and exit even with no terminal event pending.
    /// Not a redraw interval — see the module docs.
    const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(250);

    /// How stale the cached git status ([`Self::git_status`]) may get before an
    /// idle tick recomputes it. Reading `.git/HEAD` for the branch is cheap,
    /// but the dirty check walks the whole worktree, so this bounds that walk to
    /// at most once per interval on a large repository rather than running it on
    /// every 250 ms idle tick. A buffer switch to a different directory bypasses
    /// this and recomputes immediately, so switching repos never shows a stale
    /// branch; only the dirty flag lags, by at most this long, after an
    /// external change (a `:w`, a `git checkout` in another terminal).
    const GIT_STATUS_TTL: Duration = Duration::from_millis(1000);

    pub fn new(host: H, options: Options, theme: Theme, icons: IconSet, leader: char) -> Self {
        // Seed the sole window from the editor's *current* buffer, not a
        // hard-coded `BufferId(0)`: `run()` opens the files before building the
        // `App`, so the active buffer is already whatever was opened last, and
        // a window pointing at buffer 0 would render the empty scratch buffer
        // instead (or, once `render_windows` respects `window.buffer`, nothing
        // useful).
        let windows = WindowTree::single(host.active_buffer_id());
        // Harpoon marks are scoped to the working directory, exactly as the file
        // tree roots there — the same `cwd`, resolved once.
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
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
            tree_root: cwd.clone(),
            overlay: None,
            focus: Focus::Buffer,
            hop: None,
            awaiting_window_key: false,
            pending_window_count: None,
            last_windows_area: Rect::default(),
            lsp: LspClient::new(),
            lsp_hover: None,
            lsp_refs: None,
            lsp_rename: None,
            diagnostics: std::collections::HashMap::new(),
            lsp_opened: std::collections::HashSet::new(),
            lsp_no_server: std::collections::HashSet::new(),
            pending_bracket: None,
            quickfix: QuickfixList::default(),
            location: QuickfixList::default(),
            qf_window: None,
            qf_focused: false,
            completion: None,
            ctrl_x_pending: false,
            snippet: None,
            last_cursor_screen: None,
            in_tmux: std::env::var_os("TMUX").is_some(),
            // Detected once, by `run()`, right after construction — not here,
            // so building an `App` in a unit test never reads the environment
            // or the filesystem. See `App::apply_startup_advice`.
            tmux_prompt: None,
            #[cfg(test)]
            tmux_calls: Vec::new(),
            lua: None,
            harpoon: Harpoon::empty(&cwd),
            // Computed lazily — not here, so building an `App` in a unit test
            // never touches the filesystem (same rule as `tmux_prompt`). The
            // first refresh happens at the top of `run()`; `git_status_dir`
            // being `None` also makes the first idle-tick refresh recompute
            // regardless of the throttle.
            git_status: None,
            git_status_dir: None,
            git_status_checked: Instant::now(),
        }
    }

    /// Hands the App the live Lua runtime, so a keymap whose right-hand side was
    /// a Lua *function* can be fired when its key is pressed. Called by
    /// [`crate::ui::run`] once the config has executed. See the [`Self::lua`]
    /// field and [`Action::LuaKeymap`].
    pub fn set_lua_runtime(&mut self, runtime: crate::luaconfig::LuaRuntime) {
        self.lua = Some(runtime);
    }

    /// Shows a one-line informational message on the statusline at startup —
    /// used by [`crate::ui::run`] to surface a summary of what a Lua config
    /// asked for and did not fully get. Coexists with the tmux consent popup,
    /// which lives in a separate field.
    pub fn set_startup_message(&mut self, message: String) {
        self.message = StatusMessage::Info(message);
    }

    /// Shuts down every running language server. Called once the event loop
    /// exits, so kvim does not leave orphaned `rust-analyzer`/`texlab`
    /// processes behind.
    pub fn shutdown_lsp(&mut self) {
        self.lsp.shutdown_all();
    }

    /// Acts on the multiplexer advice [`crate::tmux::startup_advice`] computed
    /// at startup.
    ///
    /// Kept separate from [`Self::new`] on purpose: the advice is derived from
    /// the environment and the real filesystem, which a unit test constructing
    /// an `App` must not trigger. `run()` computes it once and hands it here; a
    /// test can pass a fabricated [`crate::tmux::StartupAdvice`] to exercise the
    /// popup with no tmux and no dotfile in sight.
    ///
    /// * A [`crate::tmux::StartupAdvice::Note`] (screen / zellij) becomes a
    ///   one-line status message — non-modal, gone on the first keypress.
    /// * A [`crate::tmux::StartupAdvice::OfferFix`] arms the consent popup.
    pub fn apply_startup_advice(&mut self, advice: crate::tmux::StartupAdvice) {
        use crate::tmux::StartupAdvice;
        match advice {
            StartupAdvice::Nothing => {}
            StartupAdvice::Note(note) => self.message = StatusMessage::Info(note),
            StartupAdvice::OfferFix(offer) => self.tmux_prompt = Some(*offer),
        }
    }

    /// Handles a key while the tmux consent popup is up. This popup is modal: it
    /// owns the keyboard until answered, so keys never reach the editor.
    ///
    /// * `y`/`Y` → apply the fix (backing up the conf first), then show the one
    ///   follow-up the user must run themselves — kvim deliberately does **not**
    ///   run `tmux source-file` for them, exactly as `--install-font` leaves the
    ///   font-cache reload to the user.
    /// * `n`/`N`/`Esc`/`q` → decline, and remember it so kvim dun ask again.
    /// * anything else → ignored; the popup stay up until it get a clear answer.
    fn handle_tmux_prompt_key(&mut self, kp: KeyPress) -> LoopAction {
        match kp.key {
            Key::Char('y') | Key::Char('Y') => {
                let offer = self.tmux_prompt.take().expect("prompt is open");
                match offer.apply() {
                    Ok(backup) => {
                        let where_to = offer.path.display();
                        let backup_note = match backup {
                            Some(bak) => format!(" (backup at {})", bak.display()),
                            None => String::new(),
                        };
                        self.message = StatusMessage::Info(format!(
                            "kvim fixed {where_to}{backup_note}. Now you run this yourself to load it: \
                             tmux source-file {where_to}  (or just restart tmux)."
                        ));
                    }
                    Err(e) => {
                        self.message = StatusMessage::Error(format!(
                            "Alamak, kvim cannot write {}: {e}. Nothing changed.",
                            offer.path.display()
                        ));
                    }
                }
                LoopAction::Redraw
            }
            Key::Char('n') | Key::Char('N') | Key::Char('q') | Key::Escape => {
                self.tmux_prompt = None;
                crate::tmux::remember_decline();
                self.message = StatusMessage::Info(
                    "Ok, kvim leave your tmux.conf alone. (Delete kvim's marker file if you change \
                     your mind — see `:help tmux`.)"
                        .to_string(),
                );
                LoopAction::Redraw
            }
            // Not a yes/no: keep the modal up rather than let a stray key slip
            // through to the editor behind it.
            _ => LoopAction::Continue,
        }
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
        // Read the git branch once before the first paint so the statusline
        // shows it from frame one, rather than only after the first idle tick.
        self.refresh_git_status(true);
        terminal.draw(|frame| self.render(frame))?;
        loop {
            if !event::poll(Self::EVENT_POLL_INTERVAL)? {
                // Idle tick: diagnostics are pushed by the server asynchronously,
                // so poll for fresh ones here (never on a fixed redraw clock —
                // see the module docs) and repaint only if they changed. The git
                // status is refreshed on the same tick (throttled, and forced
                // when the active buffer's directory changed) — the one place
                // the worktree is re-read, never per frame. Both are called
                // unconditionally so neither short-circuits the other's refresh.
                let diagnostics_changed = self.refresh_diagnostics();
                let git_changed = self.refresh_git_status(false);
                if diagnostics_changed || git_changed {
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
                // The tmux consent popup is the outermost modal: while it is up,
                // it owns every key (it is asking a yes/no question about editing
                // the user's dotfile), so it is checked before focus, window
                // commands, and everything else.
                if self.tmux_prompt.is_some() {
                    return self.handle_tmux_prompt_key(kp);
                }
                // A `<C-w>` window command owns the *next* key no matter what
                // currently has focus — that is how `<C-w>l` moves focus out of
                // the file tree and `<C-w>h` moves it back in. Checked before the
                // focus branch so the pending command is not swallowed by the
                // tree's own key handling.
                if self.awaiting_window_key {
                    return self.handle_window_key(kp);
                }
                // The focus branch, and the reason `j` in the file tree does not
                // also move the text cursor: when an overlay has focus the editor
                // is not handed the key at all.
                match self.focus() {
                    Focus::Overlay => {
                        // Window navigation crosses the tree/editor boundary even
                        // while the tree has focus: `<C-w>` begins a window
                        // command, and bare `<C-h/j/k/l>` move focus — `<C-l>`
                        // (right) returns to the editor, the others hand off to a
                        // tmux pane at the tree's outer edge. The tree's own
                        // bare `h`/`j`/`k`/`l` (no Ctrl) still reach it untouched.
                        if kp.mods.ctrl && kp.key == Key::Char('w') {
                            self.awaiting_window_key = true;
                            return LoopAction::Continue;
                        }
                        if let Some(dir) = ctrl_hjkl_direction(kp) {
                            return self.move_focus(dir, true);
                        }
                        self.handle_overlay_key(kp)
                    }
                    Focus::Buffer => {
                        // The bottom quickfix/location window, when focused, owns
                        // the keyboard like an overlay: `j`/`k`/`<CR>`/`q` move,
                        // jump and close the list. Suspended while a `:` prompt is
                        // open, so a command typed *from* the list window (the `:`
                        // handler in `handle_quickfix_key` opens it) reaches the
                        // editor rather than looping back here.
                        if self.qf_focused && self.host.command_line().is_none() {
                            return self.handle_quickfix_key(kp);
                        }
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
                        // `]d`/`[d` diagnostic navigation: `]`/`[` in Normal mode
                        // arms the motion, `d` completes it. `d` is the one key
                        // the UI claims for itself here (diagnostics are a UI
                        // concern the editor knows nothing about); every other
                        // second key belongs to the editor's own bracket-motion
                        // grammar (`]}`, `[[`, `]m`, `` ]` ``, ...), so we replay
                        // the swallowed bracket back into the editor first, then
                        // feed the current key. Before the editor grew that
                        // grammar the bracket was just dropped here — that is no
                        // longer correct (kopitiam-cj0.35).
                        if let Some(bracket) = self.pending_bracket.take() {
                            if kp.key == Key::Char('d') {
                                return self.jump_diagnostic(bracket == ']');
                            }
                            let bracket_kp = KeyPress::plain(Key::Char(bracket));
                            if self.handle_host_key(bracket_kp) == LoopAction::Quit {
                                return LoopAction::Quit;
                            }
                            return self.host_key_then_refresh_completion(kp);
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
                        // Bare `<C-h/j/k/l>` move window focus (vim-tmux-navigator
                        // style), handing off to an adjacent tmux pane at the edge.
                        // Only in Normal mode: in Insert mode `<C-h>` is backspace
                        // and `<C-w>` is delete-word-back, both the editor's, so
                        // this must never shadow them.
                        if self.host.mode() == Mode::Normal
                            && let Some(dir) = ctrl_hjkl_direction(kp)
                        {
                            return self.move_focus(dir, true);
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
            HostResponse::Quickfix(cmd) => self.handle_quickfix(cmd),
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
            Action::FindFiles => self.open_file_picker(),
            Action::FindBuffers => self.open_buffer_picker(),
            Action::FindHelp => self.open_help_picker(),
            Action::HarpoonAdd => self.harpoon_mark(),
            Action::HarpoonMenu => self.harpoon_menu(),
            Action::HarpoonFind => self.harpoon_find(),
            Action::HopWords => self.start_hop(),
            Action::LspDefinition => self.lsp_definition(),
            Action::LspReferences => self.lsp_references(),
            Action::LspRename => self.lsp_start_rename(),
            Action::LspHover => self.lsp_hover(),
            Action::LuaKeymap(id) => self.fire_lua_keymap(id),
            other => self.info(format!("{other:?} is not wired into the UI yet")),
        }
    }

    /// Fires the Lua closure a `vim.keymap.set(mode, lhs, function() ... end)`
    /// bound to this key. A config bug in the closure surfaces on the statusline
    /// rather than crashing the editor — a keymap must never be able to take kvim
    /// down. Any `vim.notify` the closure raised is drained onto the statusline
    /// too, so a config that reports through a keymap is heard.
    fn fire_lua_keymap(&mut self, id: usize) -> LoopAction {
        let Some(runtime) = self.lua.as_mut() else {
            return self.info("this key is bound to a Lua function, but no Lua runtime is loaded".to_string());
        };
        let before = runtime.notifications().len();
        let result = runtime.fire_keymap(id);
        let fresh: Vec<String> = runtime.notifications().into_iter().skip(before).collect();
        match result {
            Ok(()) => {
                if let Some(msg) = fresh.into_iter().next_back() {
                    self.info(msg)
                } else {
                    LoopAction::Redraw
                }
            }
            Err(e) => self.error(format!("keymap error: {e}")),
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
            Err(e) if e.is_not_ready() => self.info(format!("{ft} LSP is still starting — try again in a moment")),
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
            Err(e) if e.is_not_ready() => self.info(format!("{ft} LSP is still starting — try again in a moment")),
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
            Err(e) if e.is_not_ready() => self.info(format!("{ft} LSP is still starting — try again in a moment")),
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
            Err(e) if e.is_not_ready() => self.info(format!("{} LSP is still starting — try again in a moment", r.filetype)),
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
    // Quickfix & location lists (`:grep`, `:copen`, `:cnext`, `:cdo`, …).
    //
    // The editor parses these and hands them here through
    // `HostResponse::Quickfix`; this layer owns the search root, the two lists,
    // the bottom list-window, and the jumps. The list *model* and its navigation
    // grammar live in `crate::editor::quickfix`; the search *engine* in
    // `crate::plugins::grep`. This is the wiring between them.
    // ---------------------------------------------------------------

    /// The most matches a single `:grep` will collect before it stops and reports
    /// the list as truncated. A broad pattern over a big tree must never build an
    /// unbounded list and lock the editor up; a thousand hits is already far more
    /// than anyone pages through by hand.
    const QUICKFIX_MATCH_CAP: usize = 1000;

    /// The list a command targets, mutably. The one place the `Quickfix`/`Location`
    /// choice turns into a concrete field, so nothing else branches on it.
    fn list_mut(&mut self, kind: ListKind) -> &mut QuickfixList {
        match kind {
            ListKind::Quickfix => &mut self.quickfix,
            ListKind::Location => &mut self.location,
        }
    }

    /// The list a command targets, immutably.
    fn list(&self, kind: ListKind) -> &QuickfixList {
        match kind {
            ListKind::Quickfix => &self.quickfix,
            ListKind::Location => &self.location,
        }
    }

    /// Performs one parsed quickfix / location-list command.
    fn handle_quickfix(&mut self, cmd: QuickfixCommand) -> LoopAction {
        match cmd {
            QuickfixCommand::Grep { kind, pattern, globs } => self.quickfix_grep(kind, &pattern, &globs),
            QuickfixCommand::Open(kind) => {
                // `:copen`/`:lopen` open the bottom window and drop focus into it,
                // the way vim leaves you in the quickfix window.
                self.qf_window = Some(kind);
                self.qf_focused = !self.list(kind).is_empty();
                LoopAction::Redraw
            }
            QuickfixCommand::Close(kind) => {
                if self.qf_window == Some(kind) {
                    self.qf_window = None;
                    self.qf_focused = false;
                }
                LoopAction::Redraw
            }
            QuickfixCommand::Window(kind) => {
                // `:cwindow`/`:lwindow`: open iff the list has entries, else close.
                if self.list(kind).is_empty() {
                    if self.qf_window == Some(kind) {
                        self.qf_window = None;
                        self.qf_focused = false;
                    }
                } else {
                    self.qf_window = Some(kind);
                    self.qf_focused = true;
                }
                LoopAction::Redraw
            }
            QuickfixCommand::Next(kind) => self.quickfix_nav(kind, QfNav::Next),
            QuickfixCommand::Prev(kind) => self.quickfix_nav(kind, QfNav::Prev),
            QuickfixCommand::First(kind) => self.quickfix_nav(kind, QfNav::First),
            QuickfixCommand::Last(kind) => self.quickfix_nav(kind, QfNav::Last),
            QuickfixCommand::Nth { kind, nr } => self.quickfix_nav(kind, QfNav::Nth(nr)),
            QuickfixCommand::Do { kind, cmd } => self.quickfix_do(kind, &cmd),
        }
    }

    /// Runs `:grep`/`:vimgrep` (or an `l`-twin): searches [`Self::tree_root`] with
    /// the pure-Rust engine, replaces the list, and — as vim does — jumps to the
    /// first match. An invalid regex or an empty result is reported rather than
    /// leaving a stale list up.
    fn quickfix_grep(&mut self, kind: ListKind, pattern: &str, globs: &[String]) -> LoopAction {
        let re = match regex::Regex::new(pattern) {
            Ok(re) => re,
            Err(e) => return self.error(format!("invalid pattern /{pattern}/: {e}")),
        };
        let root = self.tree_root.clone();
        let outcome = grep::grep(&root, &re, globs, Self::QUICKFIX_MATCH_CAP);
        let entries: Vec<QuickfixEntry> = outcome
            .matches
            .into_iter()
            .map(|m| QuickfixEntry { path: m.path, line: m.line, col: m.col, text: m.text })
            .collect();
        let n = entries.len();
        let first = entries.first().map(|e| (e.path.clone(), e.line, e.col));
        self.list_mut(kind).set(entries);

        if n == 0 {
            return self.info(format!("no match: {pattern}"));
        }
        // Vim jumps to the first match on `:grep`/`:vimgrep`.
        if let Some((path, line, col)) = first {
            self.jump_qf(&path, line, col);
        }
        let where_to = kind.label();
        if outcome.truncated {
            self.info(format!("{n} match(es) in the {where_to} list (truncated at {})", Self::QUICKFIX_MATCH_CAP))
        } else {
            self.info(format!("{n} match(es) in the {where_to} list"))
        }
    }

    /// Runs a `:cnext`/`:cprev`/`:cfirst`/`:clast`/`:cc` navigation, jumping to
    /// the landed entry or reporting the vim-style error at the ends.
    fn quickfix_nav(&mut self, kind: ListKind, nav: QfNav) -> LoopAction {
        // Compute the move against the list, then drop the borrow before jumping
        // (the jump needs `&mut self`), carrying only the owned target out.
        let landed: Result<(PathBuf, usize, usize), NavError> = {
            let list = self.list_mut(kind);
            let r = match nav {
                QfNav::Next => list.advance(),
                QfNav::Prev => list.retreat(),
                QfNav::First => list.first(),
                QfNav::Last => list.last(),
                QfNav::Nth(nr) => list.goto(nr),
            };
            r.map(|e| (e.path.clone(), e.line, e.col))
        };
        match landed {
            Ok((path, line, col)) => self.jump_qf(&path, line, col),
            Err(e) => self.info(nav_error_message(kind, e)),
        }
    }

    /// `:cdo {cmd}`/`:ldo {cmd}`: run `cmd` on each entry's buffer.
    ///
    /// SCOPE (stated plainly): kvim runs a *single* ex command per entry and then
    /// auto-saves that buffer — the implied `| update` of the usual
    /// `:cdo s/old/new/ | update`. It does **not** parse the `|`-chained form; a
    /// bare `:cdo s/old/new/` is the supported shape, and the save is automatic.
    /// The command runs with the cursor parked at column 0 of the entry's line,
    /// so a bare `:s` acts on that line. Multi-entry edits in one file work
    /// because each entry saves before the next reopens the file. The `|`-chain
    /// and richer `:cdo` semantics are a filed follow-up bead.
    fn quickfix_do(&mut self, kind: ListKind, cmd: &str) -> LoopAction {
        let targets: Vec<(PathBuf, usize)> =
            self.list(kind).entries().iter().map(|e| (e.path.clone(), e.line)).collect();
        if targets.is_empty() {
            return self.info(format!("no {} entries", kind.label()));
        }
        let mut ran = 0usize;
        let mut errors = 0usize;
        for (path, line) in &targets {
            if self.host.open(path).is_err() {
                errors += 1;
                continue;
            }
            self.host.move_cursor(Position::new(line.saturating_sub(1), 0));
            match self.host.run_ex(cmd) {
                Ok(()) => {
                    if self.host.save().is_ok() {
                        ran += 1;
                    } else {
                        errors += 1;
                    }
                }
                Err(_) => errors += 1,
            }
        }
        self.sync_active_window();
        if errors == 0 {
            self.info(format!("{}do: ran `{cmd}` on {ran} entr(ies)", kind.label().chars().next().unwrap_or('c')))
        } else {
            self.info(format!("{}do: ran on {ran}, {errors} error(s)", kind.label().chars().next().unwrap_or('c')))
        }
    }

    /// Jumps to a quickfix entry: converts its 1-based `line`/`col` to a 0-based
    /// [`Position`] and opens the file there. A jump moves focus back to the
    /// buffer (the list window, if open, stays visible but inert — matching vim,
    /// where `<CR>` in the quickfix window leaves the window open behind you).
    fn jump_qf(&mut self, path: &Path, line: usize, col: usize) -> LoopAction {
        let pos = Position::new(line.saturating_sub(1), col.saturating_sub(1));
        self.qf_focused = false;
        self.jump_to_location(path, pos)
    }

    /// A key while the bottom list-window has focus: `j`/`k` (and arrows) move the
    /// selected entry, `<CR>` jumps to it, `G` goes to the last, `q`/`<Esc>` close
    /// the window, and `:` opens the command line (so `:cnext` etc. still work
    /// from here). Every other key is inert while the window is focused.
    fn handle_quickfix_key(&mut self, kp: KeyPress) -> LoopAction {
        let Some(kind) = self.qf_window else {
            // Belt-and-braces: no window means nothing to focus.
            self.qf_focused = false;
            return LoopAction::Continue;
        };
        match kp.key {
            Key::Escape | Key::Char('q') => {
                self.qf_window = None;
                self.qf_focused = false;
                LoopAction::Redraw
            }
            Key::Char('j') | Key::Down => {
                let next = self.list(kind).current_index() + 1;
                self.list_mut(kind).select(next);
                LoopAction::Redraw
            }
            Key::Char('k') | Key::Up => {
                let prev = self.list(kind).current_index().saturating_sub(1);
                self.list_mut(kind).select(prev);
                LoopAction::Redraw
            }
            Key::Char('G') => {
                let last = self.list(kind).len().saturating_sub(1);
                self.list_mut(kind).select(last);
                LoopAction::Redraw
            }
            Key::Enter => {
                let target = self.list(kind).current().map(|e| (e.path.clone(), e.line, e.col));
                match target {
                    Some((path, line, col)) => self.jump_qf(&path, line, col),
                    None => LoopAction::Continue,
                }
            }
            // Let `:` open the command line so quickfix ex commands still work
            // from the list window. The command-line guard in `handle_event` then
            // routes the typed keys to the editor, not back here.
            Key::Char(':') => self.host_key_then_refresh_completion(kp),
            _ => LoopAction::Continue,
        }
    }

    /// Draws the bottom list-window (`:copen`/`:lopen`) into `area`, with the
    /// current entry highlighted. Each row is vim's quickfix format,
    /// `path|lnum col N| text`.
    fn render_quickfix_window(&self, frame: &mut Frame, area: Rect, kind: ListKind) {
        let list = self.list(kind);
        let lines = quickfix_lines(list);
        let current = list.current_index();
        let inner_h = area.height.saturating_sub(2) as usize;
        // Keep the current row visible in a long list.
        let scroll = current.saturating_sub(inner_h.saturating_sub(1));
        let title = if list.is_empty() {
            format!("{} list (empty)", kind.label())
        } else {
            format!("{} list — {} of {}", kind.label(), current + 1, list.len())
        };
        frame.render_widget(
            InfoBox {
                title: &title,
                lines: &lines,
                selected: if list.is_empty() { None } else { Some(current) },
                theme: &self.theme,
                scroll,
            },
            area,
        );
    }

    // ---------------------------------------------------------------
    // Insert-mode completion: the `blink.cmp` replacement *plus* vim's native
    // insert-mode completion.
    //
    // The headless engine (`lsp::completion`) already merges and ranks the four
    // sources; this layer decides *when* to (re)query, owns the popup state, and
    // turns an accepted item into a buffer edit (a plain insert, or a snippet
    // expansion driven by `kopitiam-snippet`). Frontend keys follow the
    // maintainer's `blink.cmp`/`LuaSnip` *and* vim exactly:
    //
    //   * `<C-Space>`               — the IDE-style all-sources menu.
    //   * `<C-n>` / `<C-p>`         — vim keyword completion from the current +
    //                                 other buffers (opens the menu when none is
    //                                 up, else move to the next / previous match).
    //   * `<C-x>` then …            — vim's CTRL-X submodes:
    //       `<C-x><C-n>`/`<C-x><C-p>`  keyword, *this* buffer only.
    //       `<C-x><C-f>`               filename.
    //       `<C-x><C-l>`               whole line.
    //       `<C-x><C-o>`               omni (the language server).
    //   * inside a cycle: `<C-n>`/`<C-p>` (and Down/Up) move, `<C-y>` / `<CR>` /
    //     `<Tab>` accept, `<C-e>` cancel (revert to the typed text), `<Tab>` /
    //     `<S-Tab>` drive snippet tabstops while a snippet is active.
    //
    // kvim reuse the *one* popup for every source (see `CompletionKind`); a
    // native trigger differs only in which source seed the menu, never in the
    // menu itself — this is input wiring, not a second engine.
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

        // CTRL-X mode: the previous key was `<C-x>`, so *this* key picks the
        // native source. Consume it and clear the pending flag either way. An
        // unrecognised sub-key falls out of this block (CTRL-X cancelled) and
        // then takes its own normal path — including a second `<C-x>`, which
        // re-arms below and so keeps CTRL-X mode alive (`<C-x><C-x>…`).
        if self.ctrl_x_pending {
            self.ctrl_x_pending = false;
            if kp.mods.ctrl {
                match kp.key {
                    Key::Char('f') => return Some(self.start_file_completion()),
                    Key::Char('l') => return Some(self.start_line_completion()),
                    Key::Char('o') => return Some(self.start_omni_completion()),
                    // `<C-x><C-n>`/`<C-x><C-p>`: keyword, this buffer only. If a
                    // menu is already up, just move within it (vim keeps cycling).
                    Key::Char('n') if self.completion.is_some() => return Some(self.menu_move(1)),
                    Key::Char('p') if self.completion.is_some() => return Some(self.menu_move(-1)),
                    Key::Char('n') => return Some(self.start_keyword_completion(true, true)),
                    Key::Char('p') => return Some(self.start_keyword_completion(true, false)),
                    _ => {}
                }
            }
        }

        // `<C-x>`: enter CTRL-X mode (the next key picks the source). Only in
        // insert mode — `<C-x>` has no completion meaning in normal mode.
        if insert && kp.mods.ctrl && kp.key == Key::Char('x') {
            self.ctrl_x_pending = true;
            return Some(LoopAction::Continue);
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

        // `<C-n>`/`<C-p>` with no menu open: vim keyword completion from the
        // current + other buffers. `<C-n>` seeds selecting the first match,
        // `<C-p>` the last.
        if insert
            && self.completion.is_none()
            && kp.mods.ctrl
            && matches!(kp.key, Key::Char('n') | Key::Char('p'))
        {
            let forward = kp.key == Key::Char('n');
            return Some(self.start_keyword_completion(false, forward));
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
            // `<C-e>`: cancel, staying in insert mode with the *typed* text
            // intact. kvim never inserts a preview as you cycle (the buffer is
            // only touched on accept), so simply dropping the menu is already
            // vim's "revert to what you typed".
            Key::Char('e') if kp.mods.ctrl => {
                self.completion = None;
                Some(LoopAction::Redraw)
            }
            // `<C-y>` yanks (accepts) the selected match, same as `<CR>`/`<Tab>`.
            Key::Char('y') if kp.mods.ctrl => Some(self.accept_completion()),
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
            self.ctrl_x_pending = false;
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

        // A native completion (a `<C-n>` keyword cycle, a `<C-x>` submode) keeps
        // re-gathering from *its own* source as the user types, rather than
        // falling back to the default identifier/path logic below — that is what
        // stops `<C-x><C-f>` reverting to an identifier menu on the next key.
        if let Some(kind) = self.completion.as_ref().map(|m| m.kind) {
            match kind {
                CompletionKind::Keyword { this_buffer_only } => return self.reseed_keyword(this_buffer_only),
                CompletionKind::File => return self.reseed_file(),
                CompletionKind::Line => return self.reseed_line(),
                CompletionKind::Omni => return self.reseed_omni(),
                CompletionKind::Auto => {}
            }
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
        self.set_completion_kind(items, anchor, explicit, CompletionKind::Auto)
    }

    /// Installs a candidate list, tagging the menu with the native source that
    /// produced it so [`Self::refresh_completion`] re-gathers from the same
    /// place on the next keystroke. See [`CompletionKind`]. As with
    /// [`Self::set_completion`], preserves the selection index when it still
    /// fits, and clears the popup on an empty list.
    fn set_completion_kind(
        &mut self,
        items: Vec<CItem>,
        anchor: Position,
        explicit: bool,
        kind: CompletionKind,
    ) -> bool {
        if items.is_empty() {
            return self.close_completion_if_open();
        }
        let selected = self
            .completion
            .as_ref()
            .map(|m| m.selected.min(items.len() - 1))
            .unwrap_or(0);
        let scroll = selected.saturating_sub(MAX_COMPLETION_ROWS - 1);
        self.completion = Some(CompletionMenu { items, selected, scroll, anchor, explicit, kind });
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

    // ---- vim native insert-mode completion sources --------------------
    //
    // Each `start_*` open a submenu on a key chord and translate a "nothing
    // found" into a status message (vim beeps "Pattern not found"; kvim say so).
    // Each `reseed_*` rebuild that same submenu against the cursor's current
    // state — called on the initial trigger *and* on every later keystroke while
    // the submenu stay up, so the source that opened the menu is the source that
    // keeps feeding it (see `CompletionKind` and `refresh_completion`).

    /// Opens vim keyword completion (`<C-n>`/`<C-p>`, or `<C-x><C-n>`/
    /// `<C-x><C-p>`). `this_buffer_only` scan only the current buffer (the
    /// `<C-x>` variants); otherwise the current *and* other window buffers are
    /// scanned, matching vim's default `complete` sources. `forward` (`<C-n>`)
    /// seed the selection on the first match; `<C-p>` on the last.
    fn start_keyword_completion(&mut self, this_buffer_only: bool, forward: bool) -> LoopAction {
        if !self.reseed_keyword(this_buffer_only) {
            return self.info("kopi cannot find any matching keyword leh".to_string());
        }
        if !forward {
            // `<C-p>`: land on the last match, like vim's search-upwards.
            if let Some(menu) = self.completion.as_mut() {
                menu.selected = menu.items.len() - 1;
                let visible = menu.items.len().min(MAX_COMPLETION_ROWS);
                menu.scroll = menu.selected + 1 - visible;
            }
        }
        LoopAction::Redraw
    }

    /// Rebuilds the keyword menu from the buffer words matching the identifier
    /// prefix before the cursor. The exact word already typed is dropped —
    /// completing a word to itself help nobody, and vim leaves it out too.
    fn reseed_keyword(&mut self, this_buffer_only: bool) -> bool {
        let cursor = self.host.cursor();
        let line = self.host.buffer().line(cursor.line).unwrap_or_default();
        let (anchor_col, prefix) = identifier_prefix(&line, cursor.col);
        let words = self.keyword_items(this_buffer_only);
        let mut ranked = completion::merge_and_rank(&prefix, vec![], vec![], words, vec![]);
        if !prefix.is_empty() {
            ranked.retain(|i| i.label != prefix);
        }
        let anchor = Position::new(cursor.line, anchor_col);
        self.set_completion_kind(ranked, anchor, true, CompletionKind::Keyword { this_buffer_only })
    }

    /// The buffer-word candidates for keyword completion: always the active
    /// buffer, plus — unless `this_buffer_only` — every other window's buffer,
    /// deduplicated by id. This is kvim's stand-in for vim's `.`+`w` `complete`
    /// sources (current buffer + buffers in other windows); [`buffer_words`]
    /// dedupes the words, so overlapping buffers cost nothing.
    ///
    /// [`buffer_words`]: crate::lsp::completion::buffer_words
    fn keyword_items(&self, this_buffer_only: bool) -> Vec<CItem> {
        let mut ids: Vec<BufferId> = vec![self.host.active_buffer_id()];
        if !this_buffer_only {
            for w in self.windows.windows() {
                if !ids.contains(&w.buffer) {
                    ids.push(w.buffer);
                }
            }
        }
        let mut lines: Vec<String> = Vec::new();
        for id in ids {
            if let Some(buf) = self.host.buffer_by_id(id) {
                for i in 0..buf.line_count() {
                    lines.push(buf.line(i).unwrap_or_default());
                }
            }
        }
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        completion::buffer_words(&refs)
    }

    /// Opens filename completion (`<C-x><C-f>`).
    fn start_file_completion(&mut self) -> LoopAction {
        if !self.reseed_file() {
            return self.info("kopi cannot find any matching file leh".to_string());
        }
        LoopAction::Redraw
    }

    /// Rebuilds the filename menu from the filesystem entries under the path
    /// fragment before the cursor, relative to the buffer's own directory.
    /// Unlike the auto path context, a bare filename (no `/`) still completes —
    /// that is the whole point of `<C-x><C-f>` versus typing a `/`.
    fn reseed_file(&mut self) -> bool {
        let cursor = self.host.cursor();
        let line = self.host.buffer().line(cursor.line).unwrap_or_default();
        let (start, token) = path_fragment(&line, cursor.col);
        let base = self.completion_base_dir();
        let items = completion::path_candidates(&token, &base);
        let fname = token.rsplit('/').next().unwrap_or("").to_string();
        let ranked = completion::merge_and_rank(&fname, vec![], vec![], vec![], items);
        self.set_completion_kind(ranked, Position::new(cursor.line, start), true, CompletionKind::File)
    }

    /// The directory filename completion resolves against: the active buffer's
    /// parent directory, falling back to the tree root (kvim's working
    /// directory) for an unnamed buffer *or* a buffer opened by a bare relative
    /// name. The relative-name case matter: `Path::new("test.txt").parent()` is
    /// `Some("")`, an empty path that [`read_dir`](std::fs::read_dir) cannot
    /// open — so an empty parent has to be treated as "no parent" and routed to
    /// the tree root, else `<C-x><C-f>` in a file opened as `kvim test.txt`
    /// would silently find nothing.
    fn completion_base_dir(&self) -> PathBuf {
        self.host
            .buffer()
            .path()
            .and_then(|p| p.parent())
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.tree_root.clone())
    }

    /// Opens whole-line completion (`<C-x><C-l>`).
    fn start_line_completion(&mut self) -> LoopAction {
        if !self.reseed_line() {
            return self.info("kopi cannot find any matching line leh".to_string());
        }
        LoopAction::Redraw
    }

    /// Rebuilds the whole-line menu: distinct buffer lines whose non-blank
    /// content begins with the text already typed on the current line (leading
    /// whitespace ignored when matching, as vim does), the current line itself
    /// excluded. Accepting replaces everything typed on the line so far with the
    /// matched line — so the anchor is column zero.
    fn reseed_line(&mut self) -> bool {
        let cursor = self.host.cursor();
        let line = self.host.buffer().line(cursor.line).unwrap_or_default();
        let graphemes: Vec<&str> = line.graphemes(true).collect();
        let end = cursor.col.min(graphemes.len());
        let typed: String = graphemes[..end].concat();
        let needle = typed.trim_start();

        let mut ids: Vec<BufferId> = vec![self.host.active_buffer_id()];
        for w in self.windows.windows() {
            if !ids.contains(&w.buffer) {
                ids.push(w.buffer);
            }
        }
        let mut seen = std::collections::HashSet::new();
        let mut items = Vec::new();
        for id in ids {
            let Some(buf) = self.host.buffer_by_id(id) else { continue };
            for i in 0..buf.line_count() {
                let l = buf.line(i).unwrap_or_default();
                let trimmed = l.trim_start();
                if trimmed.is_empty() || !trimmed.starts_with(needle) || l == typed {
                    continue;
                }
                if seen.insert(l.clone()) {
                    let mut item = CItem::new(l.trim_end().to_string(), CompletionSource::Buffer);
                    item.insert_text = l.clone();
                    items.push(item);
                }
            }
        }
        self.set_completion_kind(items, Position::new(cursor.line, 0), true, CompletionKind::Line)
    }

    /// Opens omni completion (`<C-x><C-o>`) — routed to the language server.
    fn start_omni_completion(&mut self) -> LoopAction {
        if !self.reseed_omni() {
            return self.info("kopi got no omni (LSP) completion for you now".to_string());
        }
        LoopAction::Redraw
    }

    /// Rebuilds the omni menu from `textDocument/completion` alone — the same
    /// LSP source the default menu folds in, but here it is the *only* source,
    /// which is what `<C-x><C-o>` means. Empty (so the menu closes) when no
    /// server is running for the buffer, exactly like the default LSP source.
    fn reseed_omni(&mut self) -> bool {
        let cursor = self.host.cursor();
        let line = self.host.buffer().line(cursor.line).unwrap_or_default();
        let (anchor_col, prefix) = identifier_prefix(&line, cursor.col);
        let lsp_items = self.lsp_completion_items();
        let ranked = completion::merge_and_rank(&prefix, lsp_items, vec![], vec![], vec![]);
        self.set_completion_kind(ranked, Position::new(cursor.line, anchor_col), true, CompletionKind::Omni)
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
    ///
    /// The `/` requirement is what keeps the *auto* (as-you-type) menu from
    /// treating every bare word as a filename — you opt in by typing a slash.
    /// Explicit `<C-x><C-f>` completion drops that requirement (see
    /// [`Self::reseed_file`]); both share the [`path_fragment`] scanner so their
    /// idea of "where the path token starts" can never drift apart.
    fn path_context(&self, line: &str, col: usize) -> Option<(usize, String, Vec<CItem>)> {
        let (start, token) = path_fragment(line, col);
        if !token.contains('/') {
            return None;
        }
        let base = self.completion_base_dir();
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
            match self.lsp.did_open(&ft, &file, &text) {
                // Announced successfully: the server is up. Record it so we
                // never re-announce (or re-spawn) on later idle ticks.
                Ok(()) => {
                    self.lsp_opened.insert(file.clone());
                }
                // Still starting: `did_open` fast-returned `NotReady` because
                // the background connect is in flight. Do *not* mark the file
                // opened — the next idle tick retries, and the UI stayed
                // responsive because this call never blocked. This is the whole
                // point of the async client (bead kopitiam-cj0.27 / AID-0028).
                Err(e) if e.is_not_ready() => return false,
                // Terminal failure (the server could not be spawned or exited
                // during the handshake). Retrying every tick would hammer a
                // dead server, so remember this file as server-less and stop.
                Err(_) => {
                    self.lsp_no_server.insert(file.clone());
                    return false;
                }
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
    /// To get back into a visible-but-unfocused tree, use `<C-h>` (or `<C-w>h`)
    /// from the leftmost editor window — [`Self::move_focus`] lands focus on the
    /// tree — exactly as a Neovim + neo-tree user would. `<leader>e` twice still
    /// works too (close then reopen), but is no longer the only way in.
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

    /// `\ff`. Opens the file picker: a gitignore-aware walk of the tree root,
    /// fuzzy-filtered as you type, `<CR>` to open. See [`crate::ui::picker`].
    ///
    /// The walk runs once, up front, and is capped (see [`walk_files`]) so a
    /// huge tree cannot stall the open. Rows carry the *absolute* path to open
    /// but display the *relative* one — you fuzzy-match `src/pick`, not
    /// `/home/you/proj/src/pick`.
    fn open_file_picker(&mut self) -> LoopAction {
        let root = self.tree_root.clone();
        let rows: Vec<PickRow> = walk_files(&root, FILE_PICKER_CAP)
            .into_iter()
            .map(|item| {
                let label = item.relative_path.to_string_lossy().into_owned();
                PickRow::new(label, PickAction::OpenFile(root.join(&item.relative_path)))
            })
            .collect();
        self.open_picker(PickerPanel::new("Find Files", rows))
    }

    /// `\fb`. Opens the buffer picker: every open buffer as `id name [+]`,
    /// fuzzy-filtered, `<CR>` to switch to it.
    fn open_buffer_picker(&mut self) -> LoopAction {
        let rows: Vec<PickRow> = self
            .host
            .buffers()
            .into_iter()
            .map(|b| {
                let name = if b.name.is_empty() { "[No Name]".to_string() } else { b.name };
                let modified = if b.modified { " [+]" } else { "" };
                PickRow::new(format!("{}  {name}{modified}", b.id.0), PickAction::SwitchBuffer(b.id))
            })
            .collect();
        self.open_picker(PickerPanel::new("Find Buffers", rows))
    }

    /// `\fh`. Opens the help-tag picker: every `:help` topic (from
    /// [`crate::editor::help::TOPICS`]), fuzzy-filtered, `<CR>` runs
    /// `:help <topic>` and lands on that section.
    fn open_help_picker(&mut self) -> LoopAction {
        let rows: Vec<PickRow> = crate::editor::help::TOPICS
            .iter()
            .map(|topic| {
                // Show the tag and its heading so the list reads like telescope's
                // help_tags; match against both so either the id or a word from
                // the title finds it.
                let label = format!("{}  —  {}", topic.id, topic.title);
                PickRow::new(label, PickAction::OpenHelp(topic.id.to_string()))
            })
            .collect();
        self.open_picker(PickerPanel::new("Find Help", rows))
    }

    /// `<leader>b`. Marks the current file at the cursor — harpoon's
    /// `mark.add_file`. A scratch buffer with no path cannot be marked (there is
    /// nothing to jump back *to*), so that is reported honestly rather than
    /// marking a phantom. Re-marking an already-marked file is a no-op, matching
    /// upstream. See [`crate::plugins::harpoon`].
    fn harpoon_mark(&mut self) -> LoopAction {
        let Some(path) = self.host.buffer().path().map(Path::to_path_buf) else {
            return self.info("Harpoon: buffer got no file to mark lah".to_string());
        };
        let cursor = self.host.cursor();
        let display = path.display().to_string();
        if self.harpoon.add_file(path, cursor.line, cursor.col) {
            self.info(format!("Harpoon: marked \"{display}\" liao — now got {}", self.harpoon.len()))
        } else {
            self.info(format!("Harpoon: \"{display}\" already marked liao"))
        }
    }

    /// `<leader><Esc>`. Toggles the harpoon quick menu — a floating, editable
    /// list of this project's marks. Pressing it again while the menu is open
    /// closes it, the way `toggle_quick_menu` does upstream. See
    /// [`crate::ui::harpoon`].
    fn harpoon_menu(&mut self) -> LoopAction {
        if matches!(self.overlay, Some(Overlay::HarpoonMenu(_))) {
            self.close_overlay();
            return LoopAction::Redraw;
        }
        let panel = HarpoonMenuPanel::new(self.harpoon.marks().to_vec());
        self.overlay = Some(Overlay::HarpoonMenu(panel));
        self.focus = Focus::Overlay;
        LoopAction::Redraw
    }

    /// `<leader>q`. Opens a fuzzy picker scoped to the harpoon marks — the same
    /// [`PickerPanel`] the `\ff`/`\fb`/`\fh` pickers use, as a fourth source
    /// (see [`crate::ui::picker`]). `<CR>` opens the chosen mark's file.
    ///
    /// Unlike the quick menu, the picker lands at the top of the file rather
    /// than the saved cursor: it reuses [`PickAction::OpenFile`] (the frozen
    /// picker contract), and jump-to-saved-cursor is the quick menu's job. The
    /// row still *shows* the saved line so you can tell two marks in the same
    /// file apart.
    fn harpoon_find(&mut self) -> LoopAction {
        if self.harpoon.is_empty() {
            return self.info("Harpoon: no marks yet lah — go mark one with <leader>b first".to_string());
        }
        let rows: Vec<PickRow> = self
            .harpoon
            .marks()
            .iter()
            .map(|mark| {
                let label = format!("{}:{}", mark.path.display(), mark.line + 1);
                PickRow::new(label, PickAction::OpenFile(mark.path.clone()))
            })
            .collect();
        self.open_picker(PickerPanel::new("Harpoon Marks", rows))
    }

    /// Shows `panel` as the active overlay and drops focus into it. Replaces any
    /// overlay already open (opening a picker over the file tree is fine — the
    /// picker takes focus, and closing it returns you underneath), matching how
    /// telescope floats over whatever you were doing.
    fn open_picker(&mut self, panel: PickerPanel) -> LoopAction {
        self.overlay = Some(Overlay::Picker(panel));
        self.focus = Focus::Overlay;
        LoopAction::Redraw
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
            // The picker family: do the thing, then close (telescope disappears
            // on select — unlike the file tree, which stays open).
            OverlayOutcome::PickPath(path) => {
                self.close_overlay();
                self.open_path(&path, OpenTarget::Current)
            }
            OverlayOutcome::PickBuffer(id) => {
                self.close_overlay();
                self.switch_to_buffer(id)
            }
            OverlayOutcome::PickHelp(topic) => {
                self.close_overlay();
                self.open_help_topic(&topic)
            }
            // The harpoon quick menu confirmed a mark: open it *at its saved
            // cursor* and close the menu (like the pick family, harpoon's menu
            // disappears the moment you jump).
            OverlayOutcome::OpenAt { path, cursor } => {
                self.close_overlay();
                self.open_at(&path, cursor)
            }
            // The harpoon menu deleted a line: drop that mark from the canonical
            // list. The menu stays open — its own snapshot already removed the
            // same index (see [`crate::ui::harpoon`]), so the two stay in step.
            OverlayOutcome::HarpoonRemove(index) => {
                self.harpoon.remove(index);
                LoopAction::Redraw
            }
        }
    }

    /// Opens `path` in the current window and restores `cursor` — the harpoon
    /// quick menu's jump. Distinct from [`Self::open_path`] in exactly one way:
    /// after the open (which lands the editor at the origin), it drives the
    /// cursor back to where the mark was set, which is the whole reason a
    /// harpoon mark stores a position.
    fn open_at(&mut self, path: &Path, cursor: Position) -> LoopAction {
        if let Err(e) = self.host.open(path) {
            return self.error(e);
        }
        self.host.move_cursor(cursor);
        self.focus = Focus::Buffer;
        self.sync_active_window();
        self.windows.active_mut().scroll = Scroll::default();
        self.message = StatusMessage::Info(format!("\"{}\"", path.display()));
        LoopAction::Redraw
    }

    /// Switches the active window to buffer `id` (the `\fb` accept). Goes through
    /// the editor seam so the buffer's own saved cursor is restored, then syncs
    /// the window tree so the active window records the new buffer.
    fn switch_to_buffer(&mut self, id: BufferId) -> LoopAction {
        self.host.focus_buffer(id);
        self.focus = Focus::Buffer;
        self.sync_active_window();
        // A different buffer means a different cursor, so the previous buffer's
        // scroll offset must not survive; the next render re-derives it.
        self.windows.active_mut().scroll = Scroll::default();
        LoopAction::Redraw
    }

    /// Runs `:help <topic>` (the `\fh` accept) through the editor's ex layer,
    /// which opens the help buffer and jumps to the section. Reports any error
    /// on the command line rather than swallowing it.
    fn open_help_topic(&mut self, topic: &str) -> LoopAction {
        if let Err(e) = self.host.run_ex(&format!("help {topic}")) {
            return self.error(e);
        }
        self.focus = Focus::Buffer;
        self.sync_active_window();
        self.windows.active_mut().scroll = Scroll::default();
        LoopAction::Redraw
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
    ///
    /// A digit here is not a command but a `[count]` prefix for the resize and
    /// exchange keys (`<C-w>10>`, `<C-w>2x`): it accumulates into
    /// [`Self::pending_window_count`] and re-arms the window state for the next
    /// key. A leading `0` is not a count (vim reserves bare `0` for start-of-
    /// line), and there is no `<C-w>0` command, so it simply falls through and
    /// is dropped.
    fn handle_window_key(&mut self, kp: KeyPress) -> LoopAction {
        if let Key::Char(c @ '0'..='9') = kp.key
            && !(c == '0' && self.pending_window_count.is_none())
        {
            let acc = self.pending_window_count.unwrap_or(0);
            self.pending_window_count =
                Some(acc.saturating_mul(10).saturating_add((c as u8 - b'0') as u32));
            self.awaiting_window_key = true; // stay armed for the next digit or the command
            return LoopAction::Continue;
        }
        self.awaiting_window_key = false;
        // Consume the accumulated `[count]`. `count` is the repeat/height for
        // the resize keys (default 1); `count_opt` preserves the "no count at
        // all" case that `<C-w>x` needs to tell "next window" from "window 1".
        let count_opt = self.pending_window_count.take().map(|n| n as usize);
        let count = count_opt.unwrap_or(1).max(1);
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
            Key::Char('+') => self.resize_window(false, true, count),
            Key::Char('-') => self.resize_window(false, false, count),
            Key::Char('>') => self.resize_window(true, true, count),
            Key::Char('<') => self.resize_window(true, false, count),
            Key::Char('_') => self.maximize_window(false),
            Key::Char('|') => self.maximize_window(true),
            Key::Char('x') => self.exchange_window(count_opt),
            Key::Char('r') => self.rotate_windows(true),
            Key::Char('R') => self.rotate_windows(false),
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

    /// `<C-w>h/j/k/l`: focus the spatially adjacent window. Pure kvim — no tmux
    /// edge hand-off (that belongs to the bare `<C-h/j/k/l>` bindings), matching
    /// the split the maintainer expects between the two families.
    fn focus_dir(&mut self, dir: Direction) -> LoopAction {
        self.move_focus(dir, false)
    }

    /// Moves window focus in `dir`, the shared engine behind both `<C-w>h/j/k/l`
    /// (`tmux_ok == false`) and the bare `<C-h/j/k/l>` bindings (`tmux_ok ==
    /// true`).
    ///
    /// It resolves three cases in order:
    ///
    /// 1. **From the file tree** (focus is on the sidebar overlay): a rightward
    ///    move returns to the editor; any other direction is an outer edge, so
    ///    it hands off to tmux when permitted, else does nothing. The tree is not
    ///    a [`WindowTree`] leaf — see [`crate::ui::overlay`] and AID-0018/0020 —
    ///    so it cannot participate in `WindowTree::focus_direction` and is
    ///    special-cased here instead.
    /// 2. **Into the file tree**: a leftward move from the leftmost editor window
    ///    (no window lies further left) focuses the open tree, the mirror of
    ///    case 1's `<C-l>`.
    /// 3. **Between editor windows**: delegated to
    ///    [`WindowTree::focus_direction`]; a move that runs off the edge of the
    ///    layout hands off to tmux when permitted, exactly as
    ///    vim-tmux-navigator does.
    fn move_focus(&mut self, dir: Direction, tmux_ok: bool) -> LoopAction {
        // Case 1: focus currently on the file tree overlay.
        if self.focus() == Focus::Overlay {
            if dir == Direction::Right {
                self.focus = Focus::Buffer;
                return LoopAction::Redraw;
            }
            if tmux_ok {
                self.tmux_select_pane(dir);
            }
            return LoopAction::Continue;
        }

        self.sync_active_window();
        let area = self.last_windows_area;
        if self.windows.focus_direction(area, dir).is_some() {
            self.load_active_window();
            return LoopAction::Redraw;
        }

        // Off the edge of the editor's own layout.
        // Case 2: leftward into an open file tree.
        if dir == Direction::Left && matches!(self.overlay, Some(Overlay::FileTree(_))) {
            self.focus = Focus::Overlay;
            return LoopAction::Redraw;
        }
        // Case 3: otherwise hand off to the adjacent tmux pane when inside tmux.
        if tmux_ok {
            self.tmux_select_pane(dir);
        }
        LoopAction::Continue
    }

    /// Hands focus to the adjacent tmux pane — kvim's half of the
    /// vim-tmux-navigator edge contract (christoomey/vim-tmux-navigator, MIT;
    /// studied for behaviour only, no code copied). Runs `tmux select-pane
    /// -L/-D/-U/-R` so that a `<C-h/j/k/l>` move which ran off the edge of
    /// kvim's own layout crosses seamlessly into the neighbouring tmux pane.
    ///
    /// A no-op when kvim is not inside tmux ([`Self::in_tmux`] is false), so an
    /// edge move is simply a dead end, as in plain vim. Best-effort otherwise: a
    /// failure to spawn tmux is swallowed, since there is nothing useful to
    /// report from a focus key.
    ///
    /// Under `cfg(test)` the direction is recorded in [`Self::tmux_calls`]
    /// instead of spawning a process — see that field.
    fn tmux_select_pane(&mut self, dir: Direction) {
        if !self.in_tmux {
            return;
        }
        #[cfg(test)]
        {
            self.tmux_calls.push(dir);
        }
        #[cfg(not(test))]
        {
            let _ = std::process::Command::new("tmux")
                .arg("select-pane")
                .arg(tmux_pane_flag(dir))
                .status();
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

    /// `<C-w>+`/`-`/`<`/`>`: grow or shrink the active pane by `count` steps.
    /// Pure layout — no cursor sync needed, so it does not touch the editor.
    fn resize_window(&mut self, vertical: bool, grow: bool, count: usize) -> LoopAction {
        self.windows.resize_active(vertical, grow, count);
        LoopAction::Redraw
    }

    /// `<C-w>_` (maximise height) / `<C-w>|` (maximise width).
    fn maximize_window(&mut self, vertical: bool) -> LoopAction {
        self.windows.maximize_active(vertical);
        LoopAction::Redraw
    }

    /// `<C-w>x`: exchange the active pane with another (`None` = the next one).
    /// The active window's live cursor is flushed into the tree first and the
    /// swapped-in one loaded after, so the editor edits the pane that followed
    /// the swap — see [`WindowTree::exchange`].
    fn exchange_window(&mut self, count: Option<usize>) -> LoopAction {
        self.sync_active_window();
        self.windows.exchange(count);
        self.load_active_window();
        LoopAction::Redraw
    }

    /// `<C-w>r` (`forward == true`) / `<C-w>R` (`forward == false`): rotate the
    /// panes' contents. Same cursor sync/load dance as [`Self::exchange_window`].
    fn rotate_windows(&mut self, forward: bool) -> LoopAction {
        self.sync_active_window();
        self.windows.rotate(forward);
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
            // `:bd`/`:bw`: the editor already deleted the buffer and switched to
            // `replacement`. Repoint every window that was showing the deleted
            // buffer at the survivor (a split could have been showing it too),
            // then sync the active window's cursor/scroll to the editor, which
            // landed on the alternate buffer's saved position.
            WindowCommand::BufferDeleted { deleted, replacement } => {
                self.windows.remap_buffer(deleted, replacement);
                self.sync_active_window();
                self.windows.active_mut().scroll = Scroll::default();
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

        // Reserve a bottom strip for the quickfix/location window, when open, so
        // the editor windows lay out in what remains and the list never paints
        // over live text — the same carve-then-shrink the left sidebar does.
        let (windows_area, quickfix_area) = match self.qf_window {
            Some(kind) => {
                let h = quickfix_window_height(self.list(kind).len(), windows_area.height);
                if h == 0 {
                    (windows_area, None)
                } else {
                    let windows = Rect { height: windows_area.height - h, ..windows_area };
                    let qf = Rect { y: windows_area.y + windows_area.height - h, height: h, ..windows_area };
                    (windows, Some((qf, kind)))
                }
            }
            None => (windows_area, None),
        };

        self.render_windows(frame, windows_area);
        self.render_statusline(frame, statusline_area);
        self.render_cmdline(frame, cmdline_area);
        // The `<Tab>` completion wildmenu, when open, sits in the status-line
        // row (just above the command line) exactly as vim's does, painted over
        // the statusline for as long as the cycle lasts.
        self.render_wildmenu(frame, statusline_area);
        if let Some(rect) = overlay_area {
            self.render_overlay(frame, rect);
            // The `WinSeparator` between the sidebar and the editor lives in the
            // column the sidebar split reserved (see `OverlayPlacement::split`):
            // the gap between the sidebar's right edge and the windows' left. If
            // the terminal was too narrow to spare it, there is no gap and this
            // paints nothing.
            let sidebar_right = rect.x + rect.width;
            if windows_area.x > sidebar_right {
                let border = Separator {
                    rect: Rect { x: sidebar_right, y: rect.y, width: 1, height: rect.height },
                    kind: SplitKind::Vertical,
                };
                self.paint_separator(frame, border);
            }
        }
        // which-key sits on top of everything else: it is a heads-up display
        // that appears the moment a multi-key prefix is buffered. Suppressed
        // while an overlay or hop owns the keyboard (their own keys are not the
        // editor's keymaps) or a `:` prompt is open.
        // The quickfix/location window sits in the strip carved for it above,
        // painted after the editor windows so its border is clean.
        if let Some((rect, kind)) = quickfix_area {
            self.render_quickfix_window(frame, rect, kind);
        }
        self.render_which_key(frame, windows_area);
        self.render_lsp_popups(frame, windows_area);
        // The completion popup sits on top of everything, anchored at the cursor
        // captured during `render_windows`.
        self.render_completion_menu(frame, windows_area);
        // The tmux consent popup is outermost of all: it is a modal question
        // that must sit above whatever the editor happens to be showing.
        self.render_tmux_prompt(frame, windows_area);
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
            let rect = hover_rect(area, lines, self.last_cursor_screen);
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

    /// Draws the tmux consent popup, when one is armed. A passive render pass —
    /// the yes/no keystrokes are handled in [`Self::handle_tmux_prompt_key`].
    fn render_tmux_prompt(&self, frame: &mut Frame, area: Rect) {
        let Some(offer) = &self.tmux_prompt else { return };
        let lines = tmux_prompt_lines(offer);
        let title = "kvim + tmux — fix your <C-h/j/k/l>?";
        let rect = popup_rect_for(area, &lines, 84, title);
        frame.render_widget(
            InfoBox { title, lines: &lines, selected: None, theme: &self.theme, scroll: 0 },
            rect,
        );
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

        // The search-match highlight (hlsearch/incsearch) pattern, compiled once
        // per frame and shared by every window — the pattern is global search
        // state, not per-window, and vim's `'hlsearch'` light matches in every
        // window that show the text. `None` when got nothing to highlight (no
        // active search, or `:noh` already dismiss it). See
        // [`crate::editor::Editor::search_highlight`].
        let search_re = self.host.search_highlight();

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
                search: search_re.as_ref(),
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

        // Divider lines between panes (Neovim's `WinSeparator`), painted after
        // the buffers since the layout already reserved their cells — they never
        // overpaint text. See [`WindowTree::separators`].
        for sep in self.windows.separators(area) {
            self.paint_separator(frame, sep);
        }

        self.last_cursor_screen = cursor_screen;
    }

    /// Paints one pane divider with the gruvbox `WinSeparator` styling: box-
    /// drawing glyphs (`│` for a side-by-side split, `─` for a stacked one) in
    /// the palette's dim divider tone (`bg3`) over the editor background.
    fn paint_separator(&self, frame: &mut Frame, sep: Separator) {
        let glyph = match sep.kind {
            SplitKind::Vertical => '│',
            SplitKind::Horizontal => '─',
        };
        let style = Style::default().fg(self.theme.bg3).bg(self.theme.bg);
        let r = sep.rect;
        let buf = frame.buffer_mut();
        // `cell_mut` returns `None` off-buffer, so no separate bounds check is
        // needed for a rect that runs to the frame's edge.
        for y in r.y..r.y.saturating_add(r.height) {
            for x in r.x..r.x.saturating_add(r.width) {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(glyph);
                    cell.set_style(style);
                }
            }
        }
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

    /// Recomputes the cached git status ([`Self::git_status`]) for the active
    /// buffer's repository, returning whether the displayed value changed (so
    /// the caller knows whether a repaint is warranted).
    ///
    /// This is the only place the worktree is re-read. It is called from the
    /// event loop's idle tick, never per frame; [`Self::render_statusline`]
    /// only *reads* the cache. With `force` false it skips the work entirely
    /// while the cache is fresh (younger than [`Self::GIT_STATUS_TTL`]) and the
    /// target directory is unchanged — so an idle repo costs one small
    /// `.git/HEAD` read plus a bounded worktree walk at most once per TTL. A
    /// change of target directory (switching to a buffer in another repo)
    /// always recomputes immediately, regardless of the throttle, so the branch
    /// on screen is never for the wrong repository.
    fn refresh_git_status(&mut self, force: bool) -> bool {
        let dir = self.git_status_dir_target();
        let dir_changed = self.git_status_dir.as_deref() != Some(dir.as_path());
        if !force && !dir_changed && self.git_status_checked.elapsed() < Self::GIT_STATUS_TTL {
            return false;
        }
        let new_status = crate::plugins::git::status(&dir);
        self.git_status_checked = Instant::now();
        self.git_status_dir = Some(dir);
        if new_status != self.git_status {
            self.git_status = new_status;
            true
        } else {
            false
        }
    }

    /// The directory the git status is resolved from: the active buffer's
    /// containing directory when it has a real path on disk, else the editor's
    /// launch directory ([`Self::tree_root`]). A scratch buffer (`[No Name]`)
    /// has no path, so it inherits the repository kvim was started in — the
    /// same one airline would show. A bare relative filename has an empty
    /// parent, which would resolve `.git` against an ambiguous root, so it
    /// falls back the same way.
    fn git_status_dir_target(&self) -> PathBuf {
        if let Some(parent) = self.host.buffer().path().and_then(Path::parent)
            && !parent.as_os_str().is_empty()
        {
            return parent.to_path_buf();
        }
        self.tree_root.clone()
    }

    /// Formats the cached [`GitStatus`] into the statusline segment text, the
    /// airline way: a branch glyph (Nerd Font `` U+E0A0) and the branch name,
    /// plus a trailing `*` when the worktree is dirty. When no Nerd Font is
    /// present the glyph would render as a tofu box, so a plain `git:` prefix is
    /// used instead — the same graceful-degradation rule the Powerline
    /// separators follow (see [`Statusline::glyphs`]).
    fn git_branch_segment(&self) -> Option<String> {
        let status = self.git_status.as_ref()?;
        let prefix = if self.glyphs() { "\u{e0a0} " } else { "git:" };
        let dirty = if status.dirty { "*" } else { "" };
        Some(format!("{prefix}{}{dirty}", status.branch))
    }

    fn render_statusline(&self, frame: &mut Frame, area: Rect) {
        let buffer = self.host.buffer();
        // Subtle hint while the async LSP client is connecting on its background
        // thread (bead kopitiam-cj0.27): shown only for a buffer whose language
        // server is still `Connecting`, and gone the moment it is ready.
        let lsp_status = buffer
            .path()
            .and_then(lsp_filetype)
            .filter(|ft| self.lsp.is_starting(ft))
            .map(|_| "LSP: starting…".to_string());
        let data = StatuslineData {
            mode: self.host.mode(),
            file_name: display_file_name(buffer.path()),
            modified: buffer.is_modified(),
            filetype: filetype_from_path(buffer.path()),
            // The vim-fugitive/airline branch slice, from the cache refreshed on
            // the idle tick — this only reads it, never re-walks the repo. See
            // `Self::refresh_git_status` and `Self::git_branch_segment`.
            git_branch: self.git_branch_segment(),
            lsp_status,
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
            Some(input) => {
                // The cursor now comes from the editor too: the command line is
                // a real line editor, so the caret can sit anywhere, not just at
                // the end. `command_cursor()` defaults to end-of-text for a host
                // that only appends, so the fallback is still correct.
                let cursor = self.host.command_cursor().unwrap_or_else(|| input.graphemes(true).count());
                let (completions, completion_selected) = match self.host.command_completions() {
                    Some((items, sel)) => (items, sel),
                    None => (Vec::new(), 0),
                };
                CmdlineState {
                    kind: self.host.command_prompt(),
                    cursor,
                    input: input.to_string(),
                    message: StatusMessage::None,
                    completions,
                    completion_selected,
                }
            }
            None => CmdlineState {
                kind: PromptKind::None,
                input: String::new(),
                cursor: 0,
                message: self.message.clone(),
                completions: Vec::new(),
                completion_selected: 0,
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

    /// Paints the `<Tab>` completion wildmenu over `area` (the status-line row)
    /// while a completion cycle is open, and nothing otherwise.
    fn render_wildmenu(&self, frame: &mut Frame, area: Rect) {
        let state = self.cmdline_state();
        if state.completions.is_empty() {
            return;
        }
        let menu = Wildmenu { items: &state.completions, selected: state.completion_selected, theme: &self.theme };
        frame.render_widget(menu, area);
    }
}

/// Maps a Ctrl-modified `h`/`j`/`k`/`l` keypress to the window-focus direction
/// it means, or `None` for anything else.
///
/// Kept separate from `<C-w>h/j/k/l` so the *bare* Ctrl bindings (the
/// vim-tmux-navigator ones) share one definition between the two focus branches
/// in [`App::handle_event`] — the buffer-focused case and the tree-focused case
/// must agree on which key is which direction, and one function is how they do.
fn ctrl_hjkl_direction(kp: KeyPress) -> Option<Direction> {
    if !kp.mods.ctrl {
        return None;
    }
    match kp.key {
        Key::Char('h') => Some(Direction::Left),
        Key::Char('j') => Some(Direction::Down),
        Key::Char('k') => Some(Direction::Up),
        Key::Char('l') => Some(Direction::Right),
        _ => None,
    }
}

/// The `tmux select-pane` flag for a directional focus move — `-L`/`-D`/`-U`/`-R`
/// for left/down/up/right, tmux's own compass letters. Pure, so the mapping is
/// unit-testable without spawning tmux (see [`App::tmux_select_pane`], which is
/// the only caller in a non-test build).
#[cfg_attr(test, allow(dead_code))]
fn tmux_pane_flag(dir: Direction) -> &'static str {
    match dir {
        Direction::Left => "-L",
        Direction::Down => "-D",
        Direction::Up => "-U",
        Direction::Right => "-R",
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

/// The trailing run of path-like graphemes ending at grapheme column `col` on
/// `line`, and the column it starts at. "Path-like" is the word graphemes plus
/// the handful a path fragment carries: `/`, `.`, `-`, `~`. An empty run yields
/// `(col, "")`.
///
/// Shared by the auto path context ([`App::path_context`], which additionally
/// require the token contain a `/` before it treats it as a path) and by
/// `<C-x><C-f>` filename completion ([`App::reseed_file`], which complete even a
/// bare filename). Keeping the scanner in one place means the two never disagree
/// on where a path token begins.
fn path_fragment(line: &str, col: usize) -> (usize, String) {
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
/// The tallest the bottom list-window grows, in rows (content + border). Vim's
/// quickfix window defaults to 10 lines; this caps near there and shrinks to fit
/// a short list or a short screen.
const QUICKFIX_WINDOW_MAX_ROWS: u16 = 12;

/// The height (rows) the bottom list-window takes: enough for its entries plus a
/// border, capped at [`QUICKFIX_WINDOW_MAX_ROWS`], at half the available height,
/// and always leaving at least one row for the editor above it. Returns `0` when
/// there is no room for both a `>= 3`-row bordered box and a `>= 1`-row editor (a
/// screen under four rows tall), so the caller skips drawing it.
fn quickfix_window_height(entries: usize, available: u16) -> u16 {
    if available < 4 {
        return 0;
    }
    let want = (entries as u16).saturating_add(2).clamp(3, QUICKFIX_WINDOW_MAX_ROWS);
    // Prefer half the screen, never below a 3-row box, never eating the last
    // editor row.
    want.min(available / 2).max(3).min(available - 1)
}

/// Which navigation a `:c*`/`:l*` step is — the internal shape
/// [`App::quickfix_nav`] dispatches on, so the five commands share one method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QfNav {
    Next,
    Prev,
    First,
    Last,
    Nth(Option<usize>),
}

/// The vim-style message for a failed quickfix navigation. The error numbers
/// match vim's so a user who knows `E553` recognises it.
fn nav_error_message(kind: ListKind, e: NavError) -> String {
    match e {
        NavError::Empty => format!("E42: no {} entries", kind.label()),
        NavError::AtEnd | NavError::AtStart => "E553: no more items".to_string(),
        NavError::OutOfRange => "E541: entry number out of range".to_string(),
    }
}

/// Formats a quickfix/location list's entries for the list window, one row each
/// in vim's `path|lnum col N| text` shape. The line text is trimmed of leading
/// whitespace so the columns line up regardless of source indentation.
fn quickfix_lines(list: &QuickfixList) -> Vec<String> {
    list.entries()
        .iter()
        .map(|e| format!("{}|{} col {}| {}", display_path(&e.path), e.line, e.col, e.text.trim_start()))
        .collect()
}

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
/// Builds the Singlish body of the tmux consent popup from an offer.
///
/// Kept a free function (not a method) so the exact wording, and the promise it
/// makes about what will change, can be asserted in a painted-cell test without
/// standing up a whole `App`. The popup always shows the user three things
/// before asking: *what* is broken, the *exact file* kvim will touch, and the
/// *exact line(s)* it will write — never a vague "let me fix it".
fn tmux_prompt_lines(offer: &crate::tmux::TmuxOffer) -> Vec<String> {
    use crate::tmux::FixKind;

    let mut lines = vec![
        "Eh, you running kvim inside tmux.".to_string(),
        "Your <C-h/j/k/l> pane-switch confirm cannot work: tmux's is_vim check".to_string(),
        "dunno \"kvim\", so it eats the keys before kvim can even see them.".to_string(),
        String::new(),
    ];

    match offer.edit.kind {
        FixKind::ExtendRegex => {
            lines.push(format!("Let kvim fix this line in {}?", offer.path.display()));
            lines.push(String::new());
            if let Some(old) = &offer.edit.old_line {
                lines.push("  now:  ".to_string());
                lines.push(format!("    {old}"));
            }
            lines.push("  after:".to_string());
            for l in &offer.edit.new_lines {
                lines.push(format!("    {l}"));
            }
        }
        FixKind::AppendBlock => {
            lines.push(format!("Let kvim add this block to {}?", offer.path.display()));
            lines.push(String::new());
            for l in &offer.edit.new_lines {
                lines.push(format!("    {l}"));
            }
        }
        FixKind::CreateFile => {
            lines.push(format!("You got no tmux.conf. Let kvim create {} with:", offer.path.display()));
            lines.push(String::new());
            for l in &offer.edit.new_lines {
                lines.push(format!("    {l}"));
            }
        }
    }

    lines.push(String::new());
    if offer.existed {
        lines.push("kvim back up your conf first (.kvim-bak). Nothing else kena touch.".to_string());
    } else {
        lines.push("Nothing existing kena touch — this is a brand-new file.".to_string());
    }
    lines.push("[y] yes, fix lah      [n] no, leave it alone".to_string());
    lines
}

fn popup_rect_for(area: Rect, lines: &[String], max_width: u16, title: &str) -> Rect {
    let longest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let width = (longest.max(title.len()) as u16 + 4).min(max_width);
    let height = lines.len() as u16 + 2;
    centered_rect(area, width, height)
}

/// The rectangle for the LSP hover box, anchored **at the cursor** same like how
/// Neovim's `vim.lsp.buf.hover` place it: size follow the hover content (widest line
/// capped at [`MAX_HOVER_WIDTH`], up to [`MAX_HOVER_ROWS`] rows only), and drop it
/// just *above* the cursor line, flip *below* only when the cursor too near the top
/// edge. All the flip-and-clamp go through the shared [`anchored_rect`] the
/// completion menu also use, so hover and completion behave the same at the screen
/// edges, no need maintain two copies.
///
/// `cursor` is the last painted cursor cell ([`App::last_cursor_screen`]). If got
/// nothing there — means no buffer had focus this frame, so no proper place to anchor
/// — then the box just fall back to the old centred placement, better than anyhow
/// guess a corner.
fn hover_rect(area: Rect, lines: &[String], cursor: Option<(u16, u16)>) -> Rect {
    let Some(cursor) = cursor else {
        return popup_rect_for(area, lines, 60, "hover");
    };
    let longest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    // +4: the two border columns plus a leading/trailing pad; the title ("hover")
    // sets a floor so it is never clipped by a shorter body.
    let width = (longest.max("hover".len()) as u16 + 4).min(MAX_HOVER_WIDTH);
    let rows = lines.len().min(MAX_HOVER_ROWS);
    anchored_rect(area, cursor, rows, width, Anchor::Above)
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

    /// A temp directory carrying a hand-written `.git/HEAD` on `branch`, and an
    /// app rooted there. The app's buffer is a scratch buffer (no path), so
    /// [`App::git_status_dir_target`] falls back to `tree_root` — the fixture
    /// repo. Pure fixture: no `git` binary is invoked (see [`crate::plugins::git`]).
    fn app_in_fake_repo(branch: &str) -> (tempfile::TempDir, App<FakeHost>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), format!("ref: refs/heads/{branch}\n")).unwrap();
        let mut app = app_with(vec!["fn main() {}"]);
        app.tree_root = dir.path().to_path_buf();
        (dir, app)
    }

    /// The wired hook: after a refresh, the branch read from `.git/HEAD` shows
    /// up as a painted statusline segment. ASCII icon tier (`app_with` uses
    /// [`IconSet::Ascii`]) means the plain `git:` prefix, not the Nerd Font
    /// glyph, so the exact text is assertable on any terminal.
    #[test]
    fn statusline_shows_the_git_branch_segment() {
        let (_dir, mut app) = app_in_fake_repo("main");
        app.refresh_git_status(true);
        let rows = screen(&mut app, 80, 6);
        assert!(
            rows.iter().any(|r| r.contains("git:main")),
            "expected a git:main segment; screen was:\n{}",
            rows.join("\n"),
        );
    }

    /// Outside any repository the segment is `None` and nothing git-related is
    /// painted — the statusline just omits it, exactly as for the LSP hint.
    #[test]
    fn git_branch_segment_is_none_outside_a_repo() {
        let dir = tempfile::tempdir().unwrap(); // no `.git` inside
        let mut app = app_with(vec!["x"]);
        app.tree_root = dir.path().to_path_buf();
        app.refresh_git_status(true);
        assert!(app.git_branch_segment().is_none());
        let rows = screen(&mut app, 80, 6);
        assert!(rows.iter().all(|r| !r.contains("git:")), "no git segment expected:\n{}", rows.join("\n"));
    }

    /// A dirty worktree gets a trailing `*` (airline's dirty marker).
    #[test]
    fn git_branch_segment_shows_a_dirty_marker() {
        let mut app = app_with(vec!["x"]); // ASCII tier -> `git:` prefix
        app.git_status = Some(GitStatus { branch: "main".into(), detached: false, dirty: true });
        assert_eq!(app.git_branch_segment().as_deref(), Some("git:main*"));
        app.git_status = Some(GitStatus { branch: "main".into(), detached: false, dirty: false });
        assert_eq!(app.git_branch_segment().as_deref(), Some("git:main"));
    }

    /// A second refresh within the TTL, with the directory unchanged, does no
    /// work and reports "unchanged" — this is what keeps the worktree walk off
    /// the per-frame path. A forced refresh always recomputes.
    #[test]
    fn refresh_git_status_is_throttled_within_the_ttl() {
        let (_dir, mut app) = app_in_fake_repo("main");
        assert!(app.refresh_git_status(true), "first read establishes the branch");
        // Immediately after, the cache is fresh and the dir is unchanged.
        assert!(!app.refresh_git_status(false), "a throttled refresh does no work");
    }

    /// A fixture project tree for the quickfix tests: three files under `src/`,
    /// each with `TODO` on a known line, and an app rooted at it.
    fn app_with_grep_tree() -> (tempfile::TempDir, App<FakeHost>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "// TODO one\nfn a() {}\n").unwrap();
        std::fs::write(dir.path().join("src/b.rs"), "fn b() {}\n// TODO two\n").unwrap();
        std::fs::write(dir.path().join("src/c.rs"), "// TODO three\n").unwrap();
        let mut app = app_with(vec!["x"]);
        app.tree_root = dir.path().to_path_buf();
        (dir, app)
    }

    #[test]
    fn grep_populates_the_quickfix_list_and_jumps_to_the_first_match() {
        let (_dir, mut app) = app_with_grep_tree();
        app.handle_quickfix(QuickfixCommand::Grep {
            kind: ListKind::Quickfix,
            pattern: "TODO".to_string(),
            globs: vec![],
        });
        // Three TODO lines across the three files, sorted → a.rs, b.rs, c.rs.
        assert_eq!(app.quickfix.len(), 3);
        assert_eq!(app.quickfix.current_index(), 0);
        // `:grep` jumps to the first match, so the host opened src/a.rs.
        assert!(app.host.opened.last().unwrap().ends_with("src/a.rs"), "opened: {:?}", app.host.opened);
    }

    #[test]
    fn cnext_and_cprev_move_the_current_entry_and_jump() {
        let (_dir, mut app) = app_with_grep_tree();
        app.handle_quickfix(QuickfixCommand::Grep { kind: ListKind::Quickfix, pattern: "TODO".into(), globs: vec![] });
        // :cnext advances current 0 → 1 and opens the second file (b.rs).
        app.handle_quickfix(QuickfixCommand::Next(ListKind::Quickfix));
        assert_eq!(app.quickfix.current_index(), 1);
        assert!(app.host.opened.last().unwrap().ends_with("src/b.rs"));
        // :cprev steps back to a.rs.
        app.handle_quickfix(QuickfixCommand::Prev(ListKind::Quickfix));
        assert_eq!(app.quickfix.current_index(), 0);
        assert!(app.host.opened.last().unwrap().ends_with("src/a.rs"));
        // :cprev at the first entry errors and does not move (vim E553).
        app.handle_quickfix(QuickfixCommand::Prev(ListKind::Quickfix));
        assert_eq!(app.quickfix.current_index(), 0);
    }

    #[test]
    fn cc_jumps_to_an_explicit_entry() {
        let (_dir, mut app) = app_with_grep_tree();
        app.handle_quickfix(QuickfixCommand::Grep { kind: ListKind::Quickfix, pattern: "TODO".into(), globs: vec![] });
        // :cc 3 goes to the third entry (c.rs) — 1-based.
        app.handle_quickfix(QuickfixCommand::Nth { kind: ListKind::Quickfix, nr: Some(3) });
        assert_eq!(app.quickfix.current_index(), 2);
        assert!(app.host.opened.last().unwrap().ends_with("src/c.rs"));
    }

    #[test]
    fn copen_renders_the_quickfix_window_with_file_line_and_text() {
        let (_dir, mut app) = app_with_grep_tree();
        app.handle_quickfix(QuickfixCommand::Grep { kind: ListKind::Quickfix, pattern: "TODO".into(), globs: vec![] });
        // :copen shows the bottom list-window.
        app.handle_quickfix(QuickfixCommand::Open(ListKind::Quickfix));
        assert_eq!(app.qf_window, Some(ListKind::Quickfix));
        let text = screen(&mut app, 80, 24).join("\n");
        // Every entry renders in vim's `path|lnum col N| text` shape.
        assert!(text.contains("a.rs|1 col 4| // TODO one"), "quickfix window text:\n{text}");
        assert!(text.contains("b.rs|2 col 4| // TODO two"), "quickfix window text:\n{text}");
        assert!(text.contains("c.rs|1 col 4| // TODO three"), "quickfix window text:\n{text}");
        // The title names the list and the current position.
        assert!(text.contains("quickfix list"), "quickfix window text:\n{text}");
    }

    #[test]
    fn a_grep_that_finds_nothing_reports_it_and_leaves_an_empty_list() {
        let (_dir, mut app) = app_with_grep_tree();
        app.handle_quickfix(QuickfixCommand::Grep {
            kind: ListKind::Quickfix,
            pattern: "no-such-token-anywhere".into(),
            globs: vec![],
        });
        assert!(app.quickfix.is_empty());
    }

    #[test]
    fn the_quickfix_window_focus_routes_navigation_keys() {
        let (_dir, mut app) = app_with_grep_tree();
        app.handle_quickfix(QuickfixCommand::Grep { kind: ListKind::Quickfix, pattern: "TODO".into(), globs: vec![] });
        app.handle_quickfix(QuickfixCommand::Open(ListKind::Quickfix));
        assert!(app.qf_focused, ":copen drops focus into the list window");
        // `j` in the focused window moves the selected entry (0 → 1).
        app.handle_event(key_event('j'));
        assert_eq!(app.quickfix.current_index(), 1);
        // `<CR>` jumps to the selected entry and returns focus to the buffer.
        let enter = Event::Key(KeyEvent {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });
        app.handle_event(enter);
        assert!(!app.qf_focused, "a <CR> jump returns focus to the buffer");
        assert!(app.host.opened.last().unwrap().ends_with("src/b.rs"));
        // The window stays open behind the jump, matching vim.
        assert_eq!(app.qf_window, Some(ListKind::Quickfix));
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
            kind: CompletionKind::Auto,
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

    // ---- vim native insert-mode completion (cj0.37) -------------------

    #[test]
    fn plain_ctrl_n_opens_keyword_completion_from_buffer_words() {
        // No popup up yet: `<C-n>` must open vim keyword completion seeded from
        // the buffer's own words (current + other buffers), not do nothing.
        let mut app = insert_app(vec!["value valiant", "va"], Position::new(1, 2));
        app.completion_intercept(ctrl('n'));
        let menu = app.completion.as_ref().expect("<C-n> opens the keyword menu");
        assert_eq!(menu.kind, CompletionKind::Keyword { this_buffer_only: false });
        let labels: Vec<&str> = menu.items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"value"), "buffer word `value` must be offered: {labels:?}");
        assert!(labels.contains(&"valiant"), "buffer word `valiant` must be offered: {labels:?}");
        assert!(!labels.contains(&"va"), "the exact typed word `va` must not complete to itself: {labels:?}");
    }

    #[test]
    fn plain_ctrl_n_menu_paints_the_buffer_word() {
        // Painted-cell proof: the keyword menu really shows on screen.
        let mut app = insert_app(vec!["value valiant", "va"], Position::new(1, 2));
        app.completion_intercept(ctrl('n'));
        let joined = screen(&mut app, 60, 12).join("\n");
        assert!(joined.contains("value"), "the keyword candidate must be painted:\n{joined}");
    }

    #[test]
    fn ctrl_p_seeds_the_last_match() {
        // `<C-p>` (search upwards) lands on the last match, not the first.
        let mut app = insert_app(vec!["value valiant", "va"], Position::new(1, 2));
        app.completion_intercept(ctrl('p'));
        let menu = app.completion.as_ref().expect("<C-p> opens the keyword menu");
        assert_eq!(menu.selected, menu.items.len() - 1, "<C-p> starts on the last match");
    }

    #[test]
    fn ctrl_x_ctrl_n_scans_this_buffer_only() {
        let mut app = insert_app(vec!["value valiant", "va"], Position::new(1, 2));
        assert_eq!(app.completion_intercept(ctrl('x')), Some(LoopAction::Continue));
        assert!(app.ctrl_x_pending, "<C-x> arms CTRL-X mode");
        app.completion_intercept(ctrl('n'));
        assert!(!app.ctrl_x_pending, "the sub-key clears CTRL-X mode");
        let menu = app.completion.as_ref().expect("<C-x><C-n> opens keyword completion");
        assert_eq!(menu.kind, CompletionKind::Keyword { this_buffer_only: true });
    }

    #[test]
    fn ctrl_x_ctrl_f_completes_filenames() {
        // The submode state machine: `<C-x>` then `<C-f>` routes to the filename
        // source, which lists entries under the buffer's own directory.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "").unwrap();
        let file = dir.path().join("main.rs");
        let buffer = FakeBuffer::new(vec!["READ".to_string()]).with_path(&file);
        let mut host = FakeHost::new(buffer);
        host.mode = Mode::Insert;
        host.cursor = Position::new(0, 4);
        let mut app = App::new(host, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ');

        app.completion_intercept(ctrl('x'));
        app.completion_intercept(ctrl('f'));
        let menu = app.completion.as_ref().expect("<C-x><C-f> opens the filename menu");
        assert_eq!(menu.kind, CompletionKind::File);
        let labels: Vec<&str> = menu.items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"README.md"), "the matching filename must be offered: {labels:?}");
        assert!(!labels.contains(&"lib.rs"), "`lib.rs` does not start with the typed `READ`: {labels:?}");
    }

    #[test]
    fn ctrl_x_ctrl_f_resolves_a_relative_buffer_name_against_the_tree_root() {
        // A buffer opened as a bare `test.txt` has an empty parent dir; filename
        // completion must fall back to the working directory (tree root) instead
        // of trying to read the empty path and finding nothing.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "").unwrap();
        let buffer = FakeBuffer::new(vec!["READ".to_string()]).with_path("test.txt");
        let mut host = FakeHost::new(buffer);
        host.mode = Mode::Insert;
        host.cursor = Position::new(0, 4);
        let mut app = App::new(host, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ');
        app.tree_root = dir.path().to_path_buf();

        app.completion_intercept(ctrl('x'));
        app.completion_intercept(ctrl('f'));
        let menu = app.completion.as_ref().expect("<C-x><C-f> opens the filename menu");
        assert!(
            menu.items.iter().any(|i| i.label == "README.md"),
            "filename completion must resolve against the tree root for a relative buffer name"
        );
    }

    #[test]
    fn ctrl_x_ctrl_l_completes_a_whole_line_and_accepts_it() {
        let mut app = insert_app(vec!["let answer = 42;", "let ans"], Position::new(1, 7));
        app.completion_intercept(ctrl('x'));
        app.completion_intercept(ctrl('l'));
        let menu = app.completion.as_ref().expect("<C-x><C-l> opens the line menu");
        assert_eq!(menu.kind, CompletionKind::Line);
        assert!(
            menu.items.iter().any(|i| i.insert_text == "let answer = 42;"),
            "the whole matching line must be offered"
        );
        app.completion_intercept(ctrl('y'));
        assert_eq!(
            app.host.buffer.line(1).unwrap(),
            "let answer = 42;",
            "accepting a line completion replaces what was typed on the line"
        );
    }

    #[test]
    fn ctrl_y_accepts_the_selected_keyword() {
        let mut app = insert_app(vec!["value", "val"], Position::new(1, 3));
        app.completion_intercept(ctrl('n'));
        let expected = {
            let menu = app.completion.as_ref().unwrap();
            menu.items[menu.selected].insert_text.clone()
        };
        app.completion_intercept(ctrl('y'));
        assert!(app.completion.is_none(), "<C-y> accepts and closes the menu");
        assert_eq!(app.host.buffer.line(1).unwrap(), expected, "<C-y> inserts the selected match");
    }

    #[test]
    fn ctrl_e_reverts_native_completion_to_the_typed_text() {
        let mut app = insert_app(vec!["value", "val"], Position::new(1, 3));
        app.completion_intercept(ctrl('n'));
        assert!(app.completion.is_some());
        app.completion_intercept(ctrl('e'));
        assert!(app.completion.is_none(), "<C-e> cancels the menu");
        assert_eq!(app.host.buffer.line(1).unwrap(), "val", "<C-e> leaves the typed text untouched");
        assert_eq!(app.host.mode(), Mode::Insert, "and stays in insert mode");
    }

    #[test]
    fn ctrl_x_ctrl_o_routes_to_the_language_server_and_reports_when_idle() {
        // No server is running in a unit test, so omni completion has nothing to
        // offer: no menu, and kvim says so rather than beeping silently.
        let mut app = insert_app(vec!["gr"], Position::new(0, 2));
        app.completion_intercept(ctrl('x'));
        app.completion_intercept(ctrl('o'));
        assert!(app.completion.is_none(), "no running server -> no omni menu");
        assert!(
            matches!(app.message, StatusMessage::Info(_)),
            "kvim reports there is nothing to omni-complete"
        );
    }

    #[test]
    fn keeps_a_native_menu_on_its_own_source_as_you_type() {
        // Regression guard for the whole design: after `<C-x><C-l>` a keystroke
        // must re-seed from the *line* source, not silently revert to the
        // identifier menu (which is what a naive refresh would do).
        let mut app = insert_app(vec!["let answer = 42;", "let an"], Position::new(1, 6));
        app.completion_intercept(ctrl('x'));
        app.completion_intercept(ctrl('l'));
        assert_eq!(app.completion.as_ref().unwrap().kind, CompletionKind::Line);
        // Type the next char through the real event path; the menu must stay a
        // line menu.
        let typed = Event::Key(KeyEvent {
            code: KeyCode::Char('s'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });
        app.handle_event(typed);
        assert_eq!(
            app.completion.as_ref().expect("the line menu survives typing").kind,
            CompletionKind::Line,
            "typing must re-seed from the line source, not fall back to the identifier menu"
        );
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
    fn the_command_line_caret_follows_a_mid_line_cursor() {
        // The line editor can put the caret anywhere now, not just at the end.
        // Paint `:hello` with the caret two graphemes in, and assert the screen
        // caret lands on the right column: `:` at col 0, `he` at cols 1-2, caret
        // after two graphemes -> col 3.
        let mut app = app_with(vec!["x"]);
        app.host.mode = Mode::Command;
        app.host.command_line = Some("hello".to_string());
        app.host.command_cursor = Some(2);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let pos = terminal.get_cursor_position().unwrap();
        assert_eq!((pos.x, pos.y), (3, 23), "caret sits after ':he'");
        // And the text itself is on the row.
        let buffer = terminal.backend().buffer();
        let row: String = (0..6).map(|x| buffer.cell((x, 23)).unwrap().symbol().to_string()).collect();
        assert_eq!(row, ":hello");
    }

    #[test]
    fn the_tab_completion_wildmenu_paints_above_the_command_line() {
        // A completion cycle open on `:w` offers a few candidates; the wildmenu
        // strip paints them in the status-line row (just above the `:` line),
        // with the selected one highlighted the way vim's WildMenu is.
        let mut app = app_with(vec!["x"]);
        let theme = app.theme;
        app.host.mode = Mode::Command;
        app.host.command_line = Some("wa".to_string());
        app.host.command_completions = Some((vec!["w".to_string(), "wa".to_string(), "write".to_string()], 1));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        // The wildmenu sits on row 22 (statusline row); the cmdline is row 23.
        let menu_row: String = (0..12).map(|x| buffer.cell((x, 22)).unwrap().symbol().to_string()).collect();
        assert!(menu_row.starts_with("w wa write"), "wildmenu row was {menu_row:?}");
        // The selected candidate "wa" (cols 2-3) is highlighted.
        let selected_cell = buffer.cell((2, 22)).unwrap();
        assert_eq!(selected_cell.style().bg, Some(theme.yellow_bright), "selected candidate highlighted");
        // The unselected "w" (col 0) is not.
        assert_ne!(buffer.cell((0, 22)).unwrap().style().bg, Some(theme.yellow_bright));
    }

    /// The painted-cell proof for `:help`: a real [`crate::editor::Editor`]
    /// runs `:help`, and we assert the Singlish manual — with an exact key name
    /// — actually reaches the screen, not just some state field. Same lesson as
    /// the `:Neotree` regression above: only reading painted cells catches text
    /// that never made it out of the editor.
    #[test]
    fn help_command_paints_the_singlish_manual_on_screen() {
        let mut editor = crate::editor::Editor::new();
        editor.execute_ex("help").unwrap();
        let mut app = App::new(editor, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ');

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        // Flatten every text row into one big string and look for our markers.
        let mut screen = String::new();
        for y in 0..24 {
            for x in 0..80 {
                screen.push_str(buffer.cell((x, y)).unwrap().symbol());
            }
            screen.push('\n');
        }
        assert!(screen.contains("kvim :help"), "the manual title must be painted; screen was:\n{screen}");
        assert!(screen.contains("<leader>"), "a real key name must survive to the screen");
    }

    /// `:help lsp` must scroll/position so the LSP section is actually visible
    /// on screen, not just move an off-screen cursor.
    #[test]
    fn help_topic_paints_that_section_on_screen() {
        let mut editor = crate::editor::Editor::new();
        editor.execute_ex("help lsp").unwrap();
        let mut app = App::new(editor, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ');
        // Give the editor a viewport height so scrolling can centre the target.
        app.host.set_viewport_height(22);

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut screen = String::new();
        for y in 0..24 {
            for x in 0..80 {
                screen.push_str(buffer.cell((x, y)).unwrap().symbol());
            }
            screen.push('\n');
        }
        assert!(screen.contains("LSP"), "the LSP section must be on screen; screen was:\n{screen}");
        assert!(screen.contains("<leader>gd"), "the go-to-definition binding must be painted");
    }

    /// Regression for kopitiam-cj0.35: the app-level `]`/`[` interception
    /// (which arms `]d`/`[d` diagnostic navigation) must **replay** the
    /// bracket into the editor when the second key is not `d`, so the
    /// editor's own bracket-motion grammar (`]}`, `[[`, `]m`, ...) still
    /// runs. Before the fix the `]` was silently dropped and `]}` degraded to
    /// a bare `}` (paragraph-forward), landing miles from the target.
    #[test]
    fn app_replays_a_dropped_bracket_into_the_editor_motion_grammar() {
        let mut editor = crate::editor::Editor::new();
        editor.buffer_mut().apply(crate::core::Edit::insert(crate::core::Position::ORIGIN, "{\n  body\n}\ntail".to_string())).unwrap();
        editor.move_cursor(crate::core::Position::new(1, 0)); // on "  body"
        let mut app = App::new(editor, Options::default(), Theme::gruvbox_dark(), IconSet::Ascii, ' ');
        // `]}` must land the cursor on the unmatched close brace at line 2.
        app.handle_event(key_event(']'));
        app.handle_event(key_event('}'));
        assert_eq!(app.host.cursor(), crate::core::Position::new(2, 0), "the bracket motion must reach the unmatched brace, not fall through as a bare paragraph motion");
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
        // The harpoon actions are now wired (see the harpoon tests below), so
        // this uses `EasyAlign` — still unwired — to keep pinning the
        // honest-message path.
        let mut app = app_with(vec!["a"]);
        app.host.answer_next_with(HostResponse::Action(Action::EasyAlign));
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
        // shows a.txt. The reserved divider column is 19, so the right pane
        // begins at char column 20 (a byte slice would land inside the
        // multibyte `│` glyph — index by chars).
        assert!(rows[0].starts_with("BBBBBBBB"), "left pane should show b.txt, got {:?}", rows[0]);
        let right: String = rows[0].chars().skip(20).collect();
        assert!(right.contains("AAAAAAAA"), "right pane should still show a.txt, got {right:?}");
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
        let right: String = rows[0].chars().skip(20).collect();
        assert!(right.contains("AAAAAAA") && !right.contains("AAAAAAAA"),
            "right pane should have lost one A, got {right:?}");
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

    /// The char column of the vertical divider `│` in a painted row, or `None`
    /// if there is no divider on that row.
    fn divider_col(row: &str) -> Option<usize> {
        row.chars().position(|c| c == '│')
    }

    #[test]
    fn ctrl_w_gt_widens_the_active_pane_and_ctrl_w_eq_re_equalises() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        // Force a paint so `last_windows_area` is populated, then read where the
        // divider sits with the panes evenly split.
        let even = divider_col(&real_screen(&mut app, 40, 6)[0]).expect("a vertical split paints a divider");

        // `<C-w>>` three times grows the active (left) pane, so the divider
        // marches right.
        for _ in 0..3 {
            app.handle_event(ctrl_event('w'));
            app.handle_event(key_event('>'));
        }
        let widened = divider_col(&real_screen(&mut app, 40, 6)[0]).expect("divider still painted");
        assert!(widened > even, "the split boundary should have moved right: {even} -> {widened}");

        // `<C-w>=` puts it back to the even split.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('='));
        let back = divider_col(&real_screen(&mut app, 40, 6)[0]).expect("divider still painted");
        assert_eq!(back, even, "equalise should restore the even divider column");
    }

    #[test]
    fn ctrl_w_count_gt_widens_more_than_a_single_press() {
        // `<C-w>3>` in one go should move the divider further than `<C-w>>`
        // once — the digit is consumed as a count, not typed into the buffer.
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        let even = divider_col(&real_screen(&mut app, 40, 6)[0]).unwrap();

        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('1'));
        app.handle_event(key_event('>'));
        let one = divider_col(&real_screen(&mut app, 40, 6)[0]).unwrap();

        // Reset, then apply a count of three at once.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('='));
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('3'));
        app.handle_event(key_event('>'));
        let three = divider_col(&real_screen(&mut app, 40, 6)[0]).unwrap();

        assert!(one > even, "a single step still widens");
        assert!(three > one, "count 3 must widen further than count 1: {one} vs {three}");
        // The digit must not have leaked into the buffer.
        assert_eq!(app.host.buffer().text(), "BBBBBBBB\nBBBBBBBB\n", "the count digit is not text");
    }

    #[test]
    fn ctrl_w_x_exchanges_the_two_panes_buffers() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        // Before: left pane shows b.txt, right pane shows a.txt.
        let before = real_screen(&mut app, 40, 6);
        assert!(before[0].starts_with("BBBBBBBB"), "left starts as b.txt, got {:?}", before[0]);

        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('x'));

        let after = real_screen(&mut app, 40, 6);
        assert!(after[0].starts_with("AAAAAAAA"), "left pane now shows a.txt, got {:?}", after[0]);
        let right: String = after[0].chars().skip(20).collect();
        assert!(right.contains("BBBBBBBB"), "right pane now shows b.txt, got {right:?}");
    }

    #[test]
    fn ctrl_w_r_rotates_which_buffer_sits_where() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        // Two panes: left b.txt, right a.txt. A rotate swaps their contents.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('r'));
        let rows = real_screen(&mut app, 40, 6);
        assert!(rows[0].starts_with("AAAAAAAA"), "rotate moved a.txt to the left, got {:?}", rows[0]);
        let right: String = rows[0].chars().skip(20).collect();
        assert!(right.contains("BBBBBBBB"), "rotate moved b.txt to the right, got {right:?}");
    }

    #[test]
    fn ctrl_w_pipe_maximises_the_active_pane_width() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        let even = divider_col(&real_screen(&mut app, 40, 6)[0]).unwrap();

        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('|'));
        let maxed = divider_col(&real_screen(&mut app, 40, 6)[0]).expect("divider stays: sibling keeps a sliver");
        assert!(maxed > even, "maximise should push the divider well right of even: {even} -> {maxed}");
    }

    // ------------------------------------------------------------------
    // Bare <C-h/j/k/l> window navigation (vim-tmux-navigator style) and the
    // tmux edge hand-off. See `App::move_focus`.
    // ------------------------------------------------------------------

    #[test]
    fn bare_ctrl_h_and_ctrl_l_move_focus_between_vertical_splits_and_typing_follows() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        // Focus starts in the LEFT pane (the new split, b.txt); a.txt is right.
        let left = app.windows.active_id();

        // Bare <C-l> (no <C-w> prefix) moves focus right, onto a.txt.
        app.handle_event(ctrl_event('l'));
        assert_ne!(app.windows.active_id(), left, "<C-l> should move focus to the right pane");
        assert_eq!(app.host.buffer().text(), "AAAAAAAA\nAAAAAAAA\n");

        // Typing `x` edits the RIGHT pane, proving keys go where focus went —
        // asserted on the painted cells.
        app.handle_event(key_event('x'));
        let rows = real_screen(&mut app, 40, 6);
        assert!(rows[0].starts_with("BBBBBBBB"), "left pane untouched, got {:?}", rows[0]);
        let right: String = rows[0].chars().skip(20).collect();
        assert!(right.contains("AAAAAAA") && !right.contains("AAAAAAAA"),
            "right pane lost one A, got {right:?}");

        // Bare <C-h> moves focus back to the left pane (b.txt).
        app.handle_event(ctrl_event('h'));
        assert_eq!(app.windows.active_id(), left, "<C-h> should move focus back to the left pane");
        assert_eq!(app.host.buffer().text(), "BBBBBBBB\nBBBBBBBB\n");
    }

    #[test]
    fn bare_ctrl_j_and_ctrl_k_move_focus_between_horizontal_splits() {
        let (_dir, mut app, _b) = real_app_two_files();
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('s')); // horizontal split; active is the TOP window
        app.handle_event(key_event('j')); // move the top window's text cursor down
        assert_eq!(app.host.cursor(), Position::new(1, 0));

        // <C-j> focuses the window below (still at the origin).
        app.handle_event(ctrl_event('j'));
        assert_eq!(app.host.cursor(), Position::ORIGIN, "the bottom window kept its own cursor");

        // <C-k> focuses the top window again, whose cursor survived.
        app.handle_event(ctrl_event('k'));
        assert_eq!(app.host.cursor(), Position::new(1, 0), "the top window's cursor survived the round trip");
    }

    #[test]
    fn a_vertical_split_paints_a_visible_divider_between_the_panes() {
        let (_dir, mut app, b) = real_app_two_files();
        feed_str(&mut app, &format!(":vs {}", b.display()));
        app.handle_event(enter_event());
        let rows = real_screen(&mut app, 40, 6);
        // The WinSeparator column carries the box-drawing glyph on every text
        // row. For a 40-wide 50/50 vsplit the reserved divider is column 19
        // (left pane 19 cols, divider, right pane 20 cols).
        for (y, row) in rows.iter().take(4).enumerate() {
            assert_eq!(row.chars().nth(19), Some('│'), "row {y} should paint a divider at col 19: {row:?}");
        }
    }

    #[test]
    fn a_horizontal_split_paints_a_visible_divider_row_between_the_panes() {
        let (_dir, mut app, _b) = real_app_two_files();
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('s')); // horizontal split
        let rows = real_screen(&mut app, 20, 9);
        // The windows area is rows 0..7 (statusline row 7, cmdline row 8). A
        // 50/50 stacked split of a 7-row area reserves one row as the divider,
        // filled with the horizontal box-drawing glyph across its width.
        let divider = rows.iter().take(7).find(|r| r.chars().all(|c| c == '─'));
        assert!(divider.is_some(), "a full-width `─` divider row must be painted between stacked panes:\n{rows:#?}");
    }

    #[test]
    fn ctrl_l_at_the_right_edge_hands_off_to_tmux_when_inside_tmux() {
        // A single window: <C-l> runs off the right edge with nowhere to go in
        // kvim, so it hands focus to the tmux pane on the right.
        let (_dir, mut app) = real_app_one_file("hello\n");
        app.in_tmux = true;
        // Establish window geometry the way a real frame would.
        real_screen(&mut app, 40, 6);

        app.handle_event(ctrl_event('l'));
        assert_eq!(app.tmux_calls, vec![Direction::Right], "edge <C-l> issues `tmux select-pane -R`");

        // And <C-h>/<C-j>/<C-k> map to their tmux compass directions.
        app.handle_event(ctrl_event('h'));
        app.handle_event(ctrl_event('j'));
        app.handle_event(ctrl_event('k'));
        assert_eq!(
            app.tmux_calls,
            vec![Direction::Right, Direction::Left, Direction::Down, Direction::Up],
        );
    }

    #[test]
    fn an_edge_move_is_a_no_op_outside_tmux() {
        let (_dir, mut app) = real_app_one_file("hello\n");
        app.in_tmux = false;
        real_screen(&mut app, 40, 6);
        assert_eq!(app.handle_event(ctrl_event('l')), LoopAction::Continue);
        assert!(app.tmux_calls.is_empty(), "no tmux hand-off when not inside tmux");
    }

    #[test]
    fn tmux_pane_flag_maps_directions_to_tmux_compass_letters() {
        assert_eq!(tmux_pane_flag(Direction::Left), "-L");
        assert_eq!(tmux_pane_flag(Direction::Down), "-D");
        assert_eq!(tmux_pane_flag(Direction::Up), "-U");
        assert_eq!(tmux_pane_flag(Direction::Right), "-R");
    }

    // ------------------------------------------------------------------
    // File tree as a focus target: <C-h>/<C-l> (and the <C-w> forms) cross the
    // tree/editor boundary. See `App::move_focus` and `ui::overlay`.
    // ------------------------------------------------------------------

    #[test]
    fn ctrl_h_focuses_the_tree_and_ctrl_l_returns_to_the_editor() {
        let (_dir, mut app) = app_with_tree();
        // Give the app real geometry, then open the tree (which takes focus).
        screen(&mut app, 80, 12);
        press_leader_e(&mut app);
        assert_eq!(app.focus(), Focus::Overlay, "opening the tree focuses it");

        // <C-l> from the tree returns focus to the editor.
        app.handle_event(ctrl_event('l'));
        assert_eq!(app.focus(), Focus::Buffer, "<C-l> from the tree returns to the editor");

        // <C-h> from the leftmost editor window focuses the tree again.
        app.handle_event(ctrl_event('h'));
        assert_eq!(app.focus(), Focus::Overlay, "<C-h> from the leftmost window focuses the tree");

        // The <C-w> forms do the same: <C-w>l back to the editor.
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('l'));
        assert_eq!(app.focus(), Focus::Buffer, "<C-w>l from the tree returns to the editor");
        app.handle_event(ctrl_event('w'));
        app.handle_event(key_event('h'));
        assert_eq!(app.focus(), Focus::Overlay, "<C-w>h focuses the tree");
    }

    #[test]
    fn the_tree_editor_boundary_paints_a_divider_column() {
        let (_dir, mut app) = app_with_tree();
        screen(&mut app, 80, 12);
        press_leader_e(&mut app);
        let rows = screen(&mut app, 80, 12);
        // The sidebar is `FileTreePanel::WIDTH` wide; the reserved separator
        // column sits immediately to its right, carrying the vertical glyph.
        let border_x = crate::ui::filetree::FileTreePanel::WIDTH as usize;
        assert_eq!(rows[0].chars().nth(border_x), Some('│'),
            "a `│` should separate the tree from the editor at col {border_x}: {:?}", rows[0]);
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

    // --- LSP hover popup: cursor-anchored placement (kopitiam-cj0.29) ---
    //
    // These assert on the *painted border cells* of the hover box relative to a
    // known cursor cell, not on a fixed screen corner — the whole point of the
    // change. The first three drive the production `hover_rect` geometry directly;
    // the last renders the whole editor and proves `render_lsp_popups` really does
    // read `last_cursor_screen` (so the box tracks the live cursor, not the centre).

    /// Renders the hover box for `lines` anchored at the screen cell `cursor` into a
    /// fresh `w`×`h` backend, returning the computed rect and the painted buffer so a
    /// test can assert on the actual border glyphs.
    fn render_hover(
        lines: &[String],
        cursor: (u16, u16),
        w: u16,
        h: u16,
    ) -> (Rect, ratatui::buffer::Buffer) {
        let theme = Theme::gruvbox_dark();
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        let captured = std::cell::Cell::new(Rect::default());
        terminal
            .draw(|frame| {
                let rect = hover_rect(frame.area(), lines, Some(cursor));
                captured.set(rect);
                frame.render_widget(
                    InfoBox { title: "hover", lines, selected: None, theme: &theme, scroll: 0 },
                    rect,
                );
            })
            .unwrap();
        (captured.get(), terminal.backend().buffer().clone())
    }

    #[test]
    fn hover_box_sits_just_above_and_column_aligned_with_the_cursor() {
        let lines = vec!["fn greet() -> &str".to_string()];
        let cursor = (12, 10);
        let (rect, buf) = render_hover(&lines, cursor, 80, 24);
        // Above the cursor: the box's bottom border row is the row just above it.
        assert_eq!(
            rect.y + rect.height,
            cursor.1,
            "hover bottom border must be the row directly above the cursor: {rect:?}"
        );
        // Column-aligned: the box starts at the cursor column.
        assert_eq!(rect.x, cursor.0, "hover must start at the cursor column: {rect:?}");
        // And the border is actually PAINTED there — the bottom-left corner glyph
        // sits at the cursor column, one row above the cursor.
        let corner = buf.cell((rect.x, rect.y + rect.height - 1)).unwrap();
        assert_eq!(
            corner.symbol(),
            "└",
            "the box's bottom-left corner must be painted just above the cursor, got {:?}",
            corner.symbol()
        );
        // The cursor column, one row up, must carry a bottom-border cell — proof the
        // box is adjacent to the cursor, not floating in a corner.
        let above_cursor = buf.cell((cursor.0, cursor.1 - 1)).unwrap();
        assert!(
            matches!(above_cursor.symbol(), "─" | "└" | "┘"),
            "a bottom-border cell must sit directly above the cursor column, got {:?}",
            above_cursor.symbol()
        );
    }

    #[test]
    fn hover_box_flips_below_when_the_cursor_is_near_the_top_edge() {
        // Eight content lines but the cursor on the second row: no room above, so
        // the Above-preferring hover box must flip to below the cursor.
        let lines: Vec<String> = (0..8).map(|i| format!("line {i}")).collect();
        let cursor = (5, 1);
        let (rect, buf) = render_hover(&lines, cursor, 80, 24);
        assert!(
            rect.y > cursor.1,
            "near the top edge the box must flip to below the cursor: {rect:?}"
        );
        assert_eq!(rect.y, cursor.1 + 1, "flipped box starts on the line just below the cursor");
        // Painted proof: a top-border cell sits directly below the cursor column.
        let below_cursor = buf.cell((cursor.0, cursor.1 + 1)).unwrap();
        assert!(
            matches!(below_cursor.symbol(), "─" | "┌" | "┐"),
            "a top-border cell must sit directly below the cursor column, got {:?}",
            below_cursor.symbol()
        );
    }

    #[test]
    fn hover_box_clamps_to_the_right_edge() {
        let lines =
            vec!["a very long hover line that would happily overflow the terminal".to_string()];
        // Cursor hard against the right edge of an 80-column screen.
        let cursor = (78, 10);
        let (rect, _buf) = render_hover(&lines, cursor, 80, 24);
        assert!(
            rect.x + rect.width <= 80,
            "the hover box must not run off the right edge: {rect:?}"
        );
    }

    #[test]
    fn full_render_anchors_hover_next_to_the_live_cursor_not_the_centre() {
        // A tall-enough buffer so the cursor line has room above it and the window
        // does not scroll; the cursor sits well left of screen centre so a still-
        // centred (old) box would visibly fail the adjacency assertions below.
        let lines: Vec<&str> = (0..15).map(|_| "let something_here = compute_value();").collect();
        let mut app = app_with(lines);
        app.host.cursor = Position::new(8, 3);
        app.lsp_hover = Some(vec!["pub fn compute_value() -> i64".to_string(), "the docs".to_string()]);

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();

        let (cx, cy) = app.last_cursor_screen.expect("the buffer cursor must be captured this frame");
        // Find the hover box by its top-left corner glyph. `┌` is painted nowhere
        // else on a plain-text buffer, so this locates the box unambiguously.
        let corner = (0..24)
            .flat_map(|y| (0..80).map(move |x| (x, y)))
            .find(|&(x, y)| buf.cell((x, y)).unwrap().symbol() == "┌");
        let (bx, by) = corner.expect("the hover box's top-left corner must be painted");

        // Column-aligned to the cursor (there is room, no right clamp needed here).
        assert_eq!(bx, cx, "the hover box must start at the cursor column, not the screen centre");
        // Vertically adjacent above the cursor: the box's bottom border is at cy-1.
        // Its bottom row is the last painted border row of this box in column bx.
        let bottom = (by..24)
            .take_while(|&y| {
                let s = buf.cell((bx, y)).unwrap().symbol();
                s == "┌" || s == "│" || s == "└"
            })
            .last()
            .unwrap();
        assert_eq!(
            bottom + 1,
            cy,
            "the hover box's bottom border must be the row directly above the live cursor (cx={cx}, cy={cy}), box at ({bx},{by})..={bottom}"
        );
    }

    // ------------------------------------------------------------------
    // tmux consent popup (kopitiam-cj0.31): the offer paints, `y` applies
    // the fix and backs the conf up, `n` leaves everything untouched.
    // ------------------------------------------------------------------

    const CONF_MISSING_KVIM: &str = "is_vim=\"ps | grep -iqE '(view|n?vim?x?|fzf)'\"\n";

    /// An app with a tmux consent popup armed against a real temp conf that has
    /// an `is_vim` regex missing `kvim`. Returns the app and the conf path so a
    /// test can inspect the file after answering.
    fn app_with_tmux_prompt() -> (tempfile::TempDir, App<Editor>, PathBuf) {
        let (dir, mut app) = real_app_one_file("hello\n");
        let conf = dir.path().join("tmux.conf");
        std::fs::write(&conf, CONF_MISSING_KVIM).unwrap();
        let edit = crate::tmux::compute_fix(Some(CONF_MISSING_KVIM)).unwrap();
        app.tmux_prompt = Some(crate::tmux::TmuxOffer { path: conf.clone(), existed: true, edit });
        (dir, app, conf)
    }

    #[test]
    fn tmux_consent_popup_paints_the_question_and_the_exact_change() {
        let (_dir, mut app, _conf) = app_with_tmux_prompt();
        let rows = real_screen(&mut app, 90, 24);
        let screen = rows.join("\n");
        assert!(screen.contains("inside tmux"), "the popup explains the tmux problem: {screen}");
        // The exact fixed regex line is shown before any edit happens.
        assert!(screen.contains("kvim|"), "the popup shows the exact change: {screen}");
        assert!(screen.contains("[y]") && screen.contains("[n]"), "the popup offers a yes/no: {screen}");
    }

    #[test]
    fn pressing_n_leaves_the_conf_untouched_and_dismisses_the_popup() {
        let (_dir, mut app, conf) = app_with_tmux_prompt();
        let action = app.handle_event(key_event('n'));
        assert_eq!(action, LoopAction::Redraw);
        assert!(app.tmux_prompt.is_none(), "n dismisses the popup");
        // The conf on disk is byte-for-byte what it was: no edit, no backup.
        assert_eq!(std::fs::read_to_string(&conf).unwrap(), CONF_MISSING_KVIM);
        let bak = conf.with_file_name("tmux.conf.kvim-bak");
        assert!(!bak.exists(), "declining must not write a backup either");
    }

    #[test]
    fn pressing_y_applies_the_fix_and_writes_a_backup() {
        let (_dir, mut app, conf) = app_with_tmux_prompt();
        let action = app.handle_event(key_event('y'));
        assert_eq!(action, LoopAction::Redraw);
        assert!(app.tmux_prompt.is_none(), "y dismisses the popup");
        // The conf now recognises kvim.
        let written = std::fs::read_to_string(&conf).unwrap();
        assert!(written.contains("kvim|"), "the fix was applied: {written}");
        // A backup of the original was made.
        let bak = conf.with_file_name("tmux.conf.kvim-bak");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), CONF_MISSING_KVIM, "backup holds the original");
    }

    #[test]
    fn an_unrelated_key_keeps_the_modal_up() {
        let (_dir, mut app, conf) = app_with_tmux_prompt();
        // A key that is neither yes nor no must not fall through to the editor,
        // and must not answer the question.
        let action = app.handle_event(key_event('j'));
        assert_eq!(action, LoopAction::Continue);
        assert!(app.tmux_prompt.is_some(), "the popup stays until a clear yes/no");
        assert_eq!(std::fs::read_to_string(&conf).unwrap(), CONF_MISSING_KVIM, "nothing edited");
    }

    #[test]
    fn a_screen_note_becomes_a_non_modal_status_message() {
        let (_dir, mut app) = real_app_one_file("hi\n");
        app.apply_startup_advice(crate::tmux::StartupAdvice::Note("you inside screen".to_string()));
        assert!(app.tmux_prompt.is_none(), "a note arms no modal popup");
        assert_eq!(app.message, StatusMessage::Info("you inside screen".to_string()));
    }

    // ------------------------------------------------------------------
    // The fuzzy pickers (\ff / \fb / \fh) — cj0.10 "wire the pickers".
    // ------------------------------------------------------------------

    /// `\ff`: the file picker paints its box + candidates, typing narrows the
    /// list to the matching file, and `<CR>` opens it. Driven through
    /// `FakeHost` so the open is asserted on its `opened` vec.
    #[test]
    fn find_files_picker_filters_then_opens_on_enter() {
        let (_dir, mut app) = app_with_tree(); // src/main.rs + README.md, rooted at a temp dir
        app.host.answer_next_with(HostResponse::Action(Action::FindFiles));
        app.handle_event(key_event('x'));
        assert!(matches!(app.overlay, Some(Overlay::Picker(_))), "\\ff opens a picker overlay");

        // The float paints its title and both candidates while unfiltered.
        let text = screen(&mut app, 60, 16).join("\n");
        assert!(text.contains("Find Files"), "picker box painted:\n{text}");
        assert!(text.contains("main.rs"), "candidate listed:\n{text}");
        assert!(text.contains("README.md"), "candidate listed:\n{text}");

        // Typing "main" narrows to src/main.rs and drops README.md.
        for c in "main".chars() {
            app.handle_event(key_event(c));
        }
        let text = screen(&mut app, 60, 16).join("\n");
        assert!(text.contains("main.rs"), "the matching candidate stays:\n{text}");
        assert!(!text.contains("README"), "the non-matching candidate is filtered out:\n{text}");

        // <CR> opens the selected file and closes the picker.
        app.handle_event(enter_event());
        assert!(app.overlay.is_none(), "the picker closes on select");
        assert!(
            app.host.opened.last().is_some_and(|p| p.ends_with("src/main.rs")),
            "the file was opened: {:?}",
            app.host.opened
        );
    }

    /// `\ff` with a nonsense query paints no candidates, and `<CR>` on an empty
    /// result opens nothing (telescope just closes).
    #[test]
    fn find_files_picker_with_no_match_opens_nothing() {
        let (_dir, mut app) = app_with_tree();
        app.host.answer_next_with(HostResponse::Action(Action::FindFiles));
        app.handle_event(key_event('x'));
        for c in "zzqqxx".chars() {
            app.handle_event(key_event(c));
        }
        app.handle_event(enter_event());
        assert!(app.overlay.is_none(), "an empty-result <CR> still closes the picker");
        assert!(app.host.opened.is_empty(), "nothing was opened: {:?}", app.host.opened);
    }

    /// `\fb`: the buffer picker lists the open buffers and `<CR>` switches to the
    /// selected one — asserted through `FakeHost::switched_to`.
    #[test]
    fn find_buffers_picker_switches_on_enter() {
        let mut app = app_with(vec!["x"]);
        app.host.buffer_entries = vec![
            crate::ui::event::BufferEntry { id: BufferId(1), name: "src/main.rs".into(), modified: false },
            crate::ui::event::BufferEntry { id: BufferId(2), name: "src/lib.rs".into(), modified: true },
        ];
        app.host.answer_next_with(HostResponse::Action(Action::FindBuffers));
        app.handle_event(key_event('x'));
        assert!(matches!(app.overlay, Some(Overlay::Picker(_))), "\\fb opens a picker overlay");

        let text = screen(&mut app, 60, 12).join("\n");
        assert!(text.contains("Find Buffers"), "picker box painted:\n{text}");
        assert!(text.contains("main.rs"), "a buffer is listed:\n{text}");
        assert!(text.contains("lib.rs [+]"), "the modified buffer shows its flag:\n{text}");

        // Filter to lib and switch to it (buffer id 2).
        for c in "lib".chars() {
            app.handle_event(key_event(c));
        }
        app.handle_event(enter_event());
        assert!(app.overlay.is_none(), "the picker closes on select");
        assert_eq!(app.host.switched_to, Some(BufferId(2)), "the editor was asked to switch to buffer 2");
    }

    /// `\fh`: the help picker lists `:help` topics and `<CR>` opens that section.
    /// Driven through the REAL editor, so the help buffer really opens and its
    /// manual is asserted on the painted cells.
    #[test]
    fn find_help_picker_opens_the_chosen_section() {
        let (_dir, mut app) = real_app_one_file("hi\n");
        feed_str(&mut app, "\\fh");
        assert!(matches!(app.overlay, Some(Overlay::Picker(_))), "\\fh opens a picker overlay");

        let text = real_screen(&mut app, 80, 20).join("\n");
        assert!(text.contains("Find Help"), "the help picker box painted:\n{text}");

        // Filter to the LSP topic and open it.
        for c in "lsp".chars() {
            app.handle_event(key_event(c));
        }
        app.handle_event(enter_event());
        assert!(app.overlay.is_none(), "the picker closes on select");

        // The help manual is now the active buffer, sitting on the LSP section.
        let text = real_screen(&mut app, 80, 24).join("\n");
        assert!(
            text.contains("go-to-definition") || text.contains("language servers"),
            "the LSP help section is on screen:\n{text}"
        );
    }

    // ------------------------------------------------------------------
    // Harpoon (<leader>b mark / <leader><Esc> menu / <leader>q find) — cj0.10.7.
    // ------------------------------------------------------------------

    /// Fires an editor [`Action`] the way the keymap engine would — through the
    /// host, as a [`HostResponse::Action`]. Only valid while the *buffer* has
    /// focus (an open float owns the keyboard and the host never sees the key).
    fn fire(app: &mut App<FakeHost>, action: Action) {
        app.host.answer_next_with(HostResponse::Action(action));
        app.handle_event(key_event('x'));
    }

    /// Points the fake host at `path` with the cursor at `(line, col)` — the
    /// state `<leader>b` reads when it marks. No file I/O: marking never touches
    /// disk, only the picker/jump paths do.
    fn at_file(app: &mut App<FakeHost>, path: &Path, line: usize, col: usize) {
        app.host.buffer = FakeBuffer::new(vec![String::from("x")]).with_path(path);
        app.host.cursor = Position::new(line, col);
    }

    #[test]
    fn marking_two_files_lists_both_in_the_quick_menu() {
        let mut app = app_with(vec!["x"]);
        at_file(&mut app, Path::new("a.rs"), 0, 0);
        fire(&mut app, Action::HarpoonAdd);
        at_file(&mut app, Path::new("bob.rs"), 4, 2);
        fire(&mut app, Action::HarpoonAdd);

        fire(&mut app, Action::HarpoonMenu);
        assert!(matches!(app.overlay, Some(Overlay::HarpoonMenu(_))), "<leader><Esc> opens the menu");

        let text = screen(&mut app, 60, 14).join("\n");
        assert!(text.contains("Harpoon (2)"), "the menu is titled with the count:\n{text}");
        assert!(text.contains("a.rs"), "the first mark is listed:\n{text}");
        // The second mark shows its file and its 1-based saved line.
        assert!(text.contains("bob.rs:5"), "the second mark and its saved line show:\n{text}");
    }

    #[test]
    fn marking_an_already_marked_file_does_not_duplicate_it() {
        let mut app = app_with(vec!["x"]);
        at_file(&mut app, Path::new("a.rs"), 1, 1);
        fire(&mut app, Action::HarpoonAdd);
        // Same path, different cursor — still a no-op (dedup is by path).
        at_file(&mut app, Path::new("a.rs"), 9, 9);
        fire(&mut app, Action::HarpoonAdd);
        assert!(
            matches!(&app.message, StatusMessage::Info(m) if m.contains("already marked")),
            "a dup is reported: {:?}",
            app.message
        );

        fire(&mut app, Action::HarpoonMenu);
        let text = screen(&mut app, 50, 10).join("\n");
        assert!(text.contains("Harpoon (1)"), "the list did not grow:\n{text}");
    }

    #[test]
    fn selecting_a_mark_in_the_menu_jumps_to_it_at_its_saved_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.rs");
        let b = dir.path().join("b.rs");
        std::fs::write(&a, "a0\na1\na2\n").unwrap();
        std::fs::write(&b, "b0\nb1\nb2\n").unwrap();

        let mut app = app_with(vec!["x"]);
        at_file(&mut app, &a, 3, 1);
        fire(&mut app, Action::HarpoonAdd);
        at_file(&mut app, &b, 2, 4);
        fire(&mut app, Action::HarpoonAdd);

        fire(&mut app, Action::HarpoonMenu);
        // Jump straight to slot 2 (b.rs) by its number.
        app.handle_event(key_event('2'));

        assert!(app.overlay.is_none(), "the menu closes on jump");
        assert_eq!(app.focus(), Focus::Buffer);
        assert_eq!(app.host.opened.last(), Some(&b), "the second mark's file opened: {:?}", app.host.opened);
        assert_eq!(app.host.cursor, Position::new(2, 4), "and the cursor landed on the saved position");
    }

    #[test]
    fn deleting_a_line_in_the_menu_removes_that_mark() {
        let mut app = app_with(vec!["x"]);
        at_file(&mut app, Path::new("a.rs"), 0, 0);
        fire(&mut app, Action::HarpoonAdd);
        at_file(&mut app, Path::new("b.rs"), 0, 0);
        fire(&mut app, Action::HarpoonAdd);

        fire(&mut app, Action::HarpoonMenu);
        // `d` deletes the selected (first) line.
        app.handle_event(key_event('d'));
        assert!(matches!(app.overlay, Some(Overlay::HarpoonMenu(_))), "the menu stays open after a delete");

        let text = screen(&mut app, 50, 10).join("\n");
        assert!(text.contains("Harpoon (1)"), "one mark left:\n{text}");
        assert!(!text.contains("a.rs"), "the deleted mark is gone:\n{text}");
        assert!(text.contains("b.rs"), "the surviving mark remains:\n{text}");
    }

    #[test]
    fn harpoon_find_picker_filters_the_marks() {
        let mut app = app_with(vec!["x"]);
        at_file(&mut app, Path::new("alpha.rs"), 0, 0);
        fire(&mut app, Action::HarpoonAdd);
        at_file(&mut app, Path::new("beta.rs"), 0, 0);
        fire(&mut app, Action::HarpoonAdd);

        fire(&mut app, Action::HarpoonFind);
        assert!(matches!(app.overlay, Some(Overlay::Picker(_))), "<leader>q opens a picker over the marks");
        if let Some(Overlay::Picker(p)) = &app.overlay {
            assert_eq!(p.match_count(), 2, "both marks show while unfiltered");
        }

        for c in "alpha".chars() {
            app.handle_event(key_event(c));
        }
        match &app.overlay {
            Some(Overlay::Picker(p)) => assert_eq!(p.match_count(), 1, "the query narrowed the marks"),
            other => panic!("the picker vanished: {}", other.is_some()),
        }
    }

    #[test]
    fn marking_a_bufferless_scratch_is_reported_not_faked() {
        let mut app = app_with(vec!["x"]); // a scratch buffer with no path
        fire(&mut app, Action::HarpoonAdd);
        assert!(
            matches!(&app.message, StatusMessage::Info(m) if m.contains("no file")),
            "a pathless buffer cannot be marked, and says so: {:?}",
            app.message
        );
        // And nothing landed in the list.
        fire(&mut app, Action::HarpoonMenu);
        let text = screen(&mut app, 50, 8).join("\n");
        assert!(text.contains("no marks"), "the menu is empty:\n{text}");
    }

    #[test]
    fn the_find_picker_with_no_marks_says_so_and_opens_no_overlay() {
        let mut app = app_with(vec!["x"]);
        fire(&mut app, Action::HarpoonFind);
        assert!(app.overlay.is_none(), "nothing to pick, so no picker opens");
        assert!(
            matches!(&app.message, StatusMessage::Info(m) if m.contains("no marks")),
            "{:?}",
            app.message
        );
    }
}
