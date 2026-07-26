//! `kopitiam tui` — a full-screen, kopitiam-themed terminal app that reaches
//! every function the CLI exposes.
//!
//! # What this is
//!
//! The chat surface that used to *be* `kopitiam tui` is now one view among many
//! (see [`chat`]). This module is the **router**: it owns the terminal, the
//! main event loop, and a small view/state machine ([`View`] + [`App`]) that
//! moves between a home menu and the feature views. Per `CLAUDE.md` ("clients
//! own no business logic"), no view invents anything — each calls straight into
//! the engine crates or the existing CLI command functions.
//!
//! # Structure
//!
//! * [`theme`] — the shared palette, banner, and footer helper.
//! * [`logic`] — pure, UI-free, unit-tested logic (PDF discovery, fuzzy
//!   ranking, output-path derivation).
//! * [`home`], [`convert`], [`explorer`], [`chat`], [`commands`] — the views,
//!   each a self-contained state struct with `on_key` → [`Transition`] and
//!   `render`.
//!
//! # How keys flow
//!
//! The loop reads one crossterm key event and calls [`App::handle_key`], which
//! dispatches to the active view's `on_key`. Each `on_key` returns a
//! [`Transition`] describing what the router should do (stay, go home, quit,
//! open another view). Structuring it this way keeps key handling a pure
//! function of `(App, KeyEvent)`, so the router's transitions are unit-tested
//! by feeding synthetic events (see this module's tests) without a TTY.
//!
//! # Android / Termux
//!
//! Everything is crossterm-based (via `ratatui::crossterm`), with no
//! platform-specific syscalls of our own, so it runs unchanged on the workspace's
//! Android (kmux fork) target. The two slow, output-streaming commands (Scan,
//! Plan) and the kvim editor hand the terminal over via
//! [`App::run_suspended`] — a plain leave/re-enter of the alternate screen,
//! which is exactly the portable crossterm handoff Termux supports.

mod chat;
mod commands;
mod convert;
mod explorer;
mod home;
mod logic;
mod theme;
/// `pub(crate)` so the standalone `kopitiam view` command ([`crate::view`]) can
/// reuse the exact same [`viewer::ViewDoc`] rendering/navigation path the TUI
/// drives — one shared viewer, no forked logic.
pub(crate) mod viewer;

use std::io::{self, Stdout, Write};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    crossterm::{
        event::{self, Event, KeyEvent, KeyEventKind},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    layout::{Constraint, Direction, Layout},
    widgets::Paragraph,
};

use crate::adapter::{SelectedAdapter, select_adapter};

use chat::ChatView;
use commands::{CommandPane, PlanPrompt};
use convert::{ConvertFolderState, ConvertPdfState};
use explorer::ExplorerState;
use home::HomeState;
use viewer::ViewerState;

/// Options for `kopitiam tui`. Mirrors [`crate::ai::ChatConfig`] so the chat
/// view seeds the model identically to `kopitiam ai chat`.
#[derive(Args, Debug)]
pub struct TuiArgs {
    /// System prompt seeding the AI Chat view. A gentle default is used when
    /// omitted.
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system: String,

    /// Cap on tokens generated per reply. Left to the adapter default when
    /// omitted.
    #[arg(long)]
    max_tokens: Option<u32>,
}

/// The persona seeded as the chat's [`kopitiam_ai::Role::System`] message.
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are KOPITIAM's local assistant. Answer concisely and helpfully.";

/// Which view a menu entry opens. `Copy` so the home menu can hand one back by
/// value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Route {
    ConvertPdf,
    ConvertFolder,
    Explorer,
    Viewer,
    Chat,
    Editor,
    Scan,
    Status,
    Plan,
    Models,
}

/// What a view's `on_key` asks the router to do. Views never mutate the router
/// directly; they return one of these and [`App::apply`] carries it out.
pub enum Transition {
    /// Stay in the current view.
    Stay,
    /// Return to the home menu.
    Home,
    /// Quit the app.
    Quit,
    /// Open the view for `Route`.
    Open(Route),
    /// Open the Convert PDF flow pre-selected on this file (from the explorer).
    OpenConvertFile(PathBuf),
    /// Open the PDF Viewer straight on this file (from the explorer).
    OpenViewFile(PathBuf),
    /// Run the plan workflow with this task (from the Plan prompt).
    RunPlan(String),
}

/// A blocking action that needs the terminal handed over: it suspends the TUI,
/// runs on the normal screen (or lets kvim own the screen), then restores.
enum Pending {
    Scan,
    Plan(String),
    Editor,
}

/// The active view and its owned state. Chat and the home menu live on [`App`]
/// itself so their state survives navigating away and back; the rest are
/// created fresh on entry.
enum View {
    Home,
    ConvertPdf(ConvertPdfState),
    ConvertFolder(ConvertFolderState),
    Explorer(ExplorerState),
    /// Boxed because [`ViewerState`] (which owns an optional open document with
    /// its rasteriser and graphics-protocol picker) is much larger than the
    /// other variants; boxing keeps `View` small.
    Viewer(Box<ViewerState>),
    Chat,
    Command(CommandPane),
    Plan(PlanPrompt),
}

/// The whole app: the current view, the persistent home + chat state, the
/// lazily-selected model adapter, and any pending blocking action.
struct App {
    view: View,
    home: HomeState,
    /// Built lazily on first entry to chat, then kept for the session.
    chat: Option<ChatView>,
    /// Selected lazily alongside `chat` (a model load can be slow, so it is
    /// deferred until the user actually opens chat).
    adapter: Option<SelectedAdapter>,
    system: String,
    max_tokens: Option<u32>,
    cwd: PathBuf,
    pending: Option<Pending>,
    should_quit: bool,
}

/// Entry point for `kopitiam tui`. Sets up the terminal (restoring it even on
/// panic), then runs the app loop.
pub fn run(args: TuiArgs) -> Result<()> {
    let mut terminal = setup_terminal().context("failed to initialise the terminal")?;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        previous_hook(info);
    }));

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut app = App::new(args.system, args.max_tokens, cwd);
    let result = app.main_loop(&mut terminal);

    restore_terminal().ok();
    let _ = std::panic::take_hook();
    result
}

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn setup_terminal() -> Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout)).map_err(Into::into)
}

/// Undo [`setup_terminal`]. Idempotent and best-effort so it is safe from the
/// panic hook and again on the normal path.
fn restore_terminal() -> Result<()> {
    let mut stdout = io::stdout();
    execute!(stdout, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

impl App {
    fn new(system: String, max_tokens: Option<u32>, cwd: PathBuf) -> Self {
        Self {
            view: View::Home,
            home: HomeState::new(),
            chat: None,
            adapter: None,
            system,
            max_tokens,
            cwd,
            pending: None,
            should_quit: false,
        }
    }

    /// Paint, drive any per-frame background work (chat streaming, batch
    /// conversion), handle at most one key, then run any pending blocking action.
    fn main_loop(&mut self, terminal: &mut Tui) -> Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| self.render(frame))?;

            // Per-frame background work for the active view.
            if matches!(self.view, View::Chat)
                && let Some(chat) = self.chat.as_mut()
            {
                chat.drain_stream();
            }
            if let View::ConvertFolder(folder) = &mut self.view
                && folder.needs_step()
            {
                folder.step();
            }
            // Run the viewer's (possibly slow) reflow render one frame after its
            // "rendering…" notice has painted, so the app never freezes on a
            // black screen.
            if let View::Viewer(viewer) = &mut self.view
                && viewer.needs_render()
            {
                viewer.render_now();
            }

            if event::poll(Duration::from_millis(30))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                self.handle_key(key);
            }

            if let Some(pending) = self.pending.take() {
                self.run_suspended(terminal, pending)?;
            }
        }
        Ok(())
    }

    /// Dispatch a key to the active view and apply the transition it returns.
    fn handle_key(&mut self, key: KeyEvent) {
        let transition = match &mut self.view {
            View::Home => self.home.on_key(key),
            View::ConvertPdf(state) => state.on_key(key),
            View::ConvertFolder(state) => state.on_key(key),
            View::Explorer(state) => state.on_key(key),
            View::Viewer(state) => state.on_key(key),
            View::Command(state) => state.on_key(key),
            View::Plan(state) => state.on_key(key),
            View::Chat => {
                // Chat needs the shared adapter; both are guaranteed present
                // because entering chat initialises them.
                match (self.chat.as_mut(), self.adapter.as_ref()) {
                    (Some(chat), Some(adapter)) => chat.on_key(key, adapter),
                    _ => Transition::Home,
                }
            }
        };
        self.apply(transition);
    }

    /// Carry out a [`Transition`].
    fn apply(&mut self, transition: Transition) {
        match transition {
            Transition::Stay => {}
            Transition::Home => self.view = View::Home,
            Transition::Quit => self.should_quit = true,
            Transition::Open(route) => self.open(route),
            Transition::OpenConvertFile(pdf) => {
                self.view = View::ConvertPdf(ConvertPdfState::with_file(pdf));
            }
            Transition::OpenViewFile(pdf) => {
                self.view = View::Viewer(Box::new(ViewerState::with_file(pdf)));
            }
            Transition::RunPlan(task) => {
                self.pending = Some(Pending::Plan(task));
                self.view = View::Home;
            }
        }
    }

    /// Enter the view (or arm the blocking action) for `route`.
    fn open(&mut self, route: Route) {
        self.view = match route {
            Route::ConvertPdf => View::ConvertPdf(ConvertPdfState::new(self.cwd.clone())),
            Route::ConvertFolder => View::ConvertFolder(ConvertFolderState::new(self.cwd.clone())),
            Route::Explorer => View::Explorer(ExplorerState::new(self.cwd.clone())),
            Route::Viewer => View::Viewer(Box::new(ViewerState::new(self.cwd.clone()))),
            Route::Chat => {
                self.ensure_chat();
                View::Chat
            }
            Route::Status => View::Command(CommandPane::status(&self.cwd)),
            Route::Models => View::Command(CommandPane::models()),
            Route::Plan => View::Plan(PlanPrompt::new()),
            // Scan and the editor run on the normal terminal; stay on Home and
            // let the loop pick up the pending action.
            Route::Scan => {
                self.pending = Some(Pending::Scan);
                View::Home
            }
            Route::Editor => {
                self.pending = Some(Pending::Editor);
                View::Home
            }
        };
    }

    /// Lazily select the adapter and build the chat view on first use.
    fn ensure_chat(&mut self) {
        if self.adapter.is_none() {
            self.adapter = Some(select_adapter());
        }
        if self.chat.is_none()
            && let Some(adapter) = self.adapter.as_ref()
        {
            self.chat = Some(ChatView::new(self.system.clone(), self.max_tokens, adapter));
        }
    }

    /// Suspend the TUI, run a blocking action on the normal terminal (or hand
    /// the screen to kvim), then restore and return to Home with a short notice.
    ///
    /// This is the Android/Termux-safe handoff: a plain leave/re-enter of the
    /// alternate screen, no fd redirection or platform syscalls. Scan and Plan
    /// stream their own progress to the normal screen (so a multi-minute cargo
    /// build is visible, not a frozen TUI); kvim owns the screen entirely while
    /// it runs.
    fn run_suspended(&mut self, terminal: &mut Tui, pending: Pending) -> Result<()> {
        restore_terminal().ok();
        execute!(io::stdout(), ratatui::crossterm::cursor::Show).ok();

        let notice = match pending {
            Pending::Editor => match run_editor() {
                Ok(()) => "kvim closed — back at the kopitiam.".to_string(),
                Err(err) => format!("kvim could not start: {err}"),
            },
            Pending::Scan => {
                println!("Running `kopitiam scan` on {} ...\n", self.cwd.display());
                let notice = match crate::scan::run(scan_args(&self.cwd)) {
                    Ok(()) => "scan finished.".to_string(),
                    Err(err) => format!("scan failed: {err}"),
                };
                pause_for_enter();
                notice
            }
            Pending::Plan(task) => {
                println!("Running `kopitiam plan` on {} ...\n", self.cwd.display());
                let notice = match crate::plan::run(plan_args(&self.cwd, task)) {
                    Ok(()) => "plan finished.".to_string(),
                    Err(err) => format!("plan failed: {err}"),
                };
                pause_for_enter();
                notice
            }
        };

        // Reclaim the terminal for the TUI and force a full repaint.
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        terminal.clear()?;

        self.home.set_notice(notice);
        self.view = View::Home;
        Ok(())
    }

    fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(area);
        let body = chunks[0];
        let footer = chunks[1];

        // Footer hints, computed before the mutable render borrow.
        let hints: Vec<(&str, &str)> = match &self.view {
            View::Home => self.home.footer_hints(),
            View::ConvertPdf(state) => state.footer_hints(),
            View::ConvertFolder(state) => state.footer_hints(),
            View::Explorer(state) => state.footer_hints(),
            View::Viewer(state) => state.footer_hints(),
            View::Command(state) => state.footer_hints(),
            View::Plan(state) => state.footer_hints(),
            View::Chat => self.chat.as_ref().map(ChatView::footer_hints).unwrap_or_default(),
        };

        match &mut self.view {
            View::Home => self.home.render(frame, body),
            View::ConvertPdf(state) => state.render(frame, body),
            View::ConvertFolder(state) => state.render(frame, body),
            View::Explorer(state) => state.render(frame, body),
            View::Viewer(state) => state.render(frame, body),
            View::Command(state) => state.render(frame, body),
            View::Plan(state) => state.render(frame, body),
            View::Chat => {
                if let Some(chat) = self.chat.as_mut() {
                    chat.render(frame, body);
                }
            }
        }

        frame.render_widget(Paragraph::new(theme::help_line(&hints)), footer);
    }
}

/// Build [`crate::scan::ScanArgs`] rooted at `cwd` with the default (fast)
/// providers — no rust-analyzer wait, not verbose.
fn scan_args(cwd: &std::path::Path) -> crate::scan::ScanArgs {
    crate::scan::ScanArgs { root: cwd.to_path_buf(), with_rust_analyzer: false, verbose: false }
}

/// Build [`crate::plan::PlanArgs`] rooted at `cwd` for `task`.
fn plan_args(cwd: &std::path::Path, task: String) -> crate::plan::PlanArgs {
    crate::plan::PlanArgs { root: cwd.to_path_buf(), task }
}

/// Launch the kvim editor **in-process** to completion.
///
/// kvim's `kopitiam_neovim::ui::run` owns its own terminal guard (it enters and
/// leaves the alternate screen and raw mode itself), so the caller must already
/// have suspended the TUI's terminal before calling this. On return, the caller
/// re-enters the TUI's own alternate screen. Chosen over spawning the `kvim`
/// binary as a subprocess because the crate exposes a clean callable entry, so
/// an in-process call is simpler, needs no PATH lookup, and cannot fail to find
/// a binary. Any error (config load / editor run) is returned rather than
/// panicking, so an editor problem never crashes the TUI.
fn run_editor() -> Result<()> {
    let config = kopitiam_neovim::Config::load()?;
    kopitiam_neovim::ui::run(config, &[])
}

/// Wait for the user to press Enter before restoring the TUI, so command output
/// on the normal screen can be read.
fn pause_for_enter() {
    print!("\n[kopitiam] Press Enter to return to the TUI...");
    let _ = io::stdout().flush();
    let mut buf = String::new();
    let _ = io::stdin().read_line(&mut buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};

    fn app() -> App {
        App::new(DEFAULT_SYSTEM_PROMPT.to_string(), None, PathBuf::from("."))
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn starts_on_home() {
        let app = app();
        assert!(matches!(app.view, View::Home));
    }

    #[test]
    fn enter_on_home_opens_convert_pdf() {
        let mut app = app();
        app.handle_key(key(KeyCode::Enter));
        assert!(matches!(app.view, View::ConvertPdf(_)));
    }

    #[test]
    fn esc_from_a_subview_returns_home() {
        let mut app = app();
        app.handle_key(key(KeyCode::Enter)); // -> ConvertPdf (Pick stage)
        assert!(matches!(app.view, View::ConvertPdf(_)));
        app.handle_key(key(KeyCode::Esc)); // Pick stage Esc -> Home
        assert!(matches!(app.view, View::Home));
    }

    #[test]
    fn j_then_enter_opens_convert_folder() {
        let mut app = app();
        app.handle_key(key(KeyCode::Char('j'))); // select "Convert Folder"
        app.handle_key(key(KeyCode::Enter));
        assert!(matches!(app.view, View::ConvertFolder(_)));
    }

    #[test]
    fn selecting_editor_arms_pending_and_stays_home() {
        let mut app = app();
        // Editor is index 5: Convert PDF, Convert Folder, File Explorer,
        // View PDF, AI Chat, kvim (Editor).
        for _ in 0..5 {
            app.handle_key(key(KeyCode::Char('j')));
        }
        app.handle_key(key(KeyCode::Enter));
        assert!(matches!(app.pending, Some(Pending::Editor)));
        // Control stays at Home; the editor runs via the pending action and
        // returns here on exit.
        assert!(matches!(app.view, View::Home));
    }

    #[test]
    fn plan_prompt_arms_pending_plan_on_enter() {
        let mut app = app();
        // Plan is index 8: ...View PDF(3), Chat(4), Editor(5), Scan(6),
        // Status(7), Plan(8).
        for _ in 0..8 {
            app.handle_key(key(KeyCode::Char('j')));
        }
        app.handle_key(key(KeyCode::Enter)); // -> Plan prompt
        assert!(matches!(app.view, View::Plan(_)));
        app.handle_key(key(KeyCode::Char('x'))); // type a non-empty task
        app.handle_key(key(KeyCode::Enter));
        assert!(matches!(app.pending, Some(Pending::Plan(_))));
        assert!(matches!(app.view, View::Home));
    }

    #[test]
    fn opening_a_pdf_from_the_explorer_enters_convert_prefilled() {
        let mut app = app();
        app.apply(Transition::OpenConvertFile(PathBuf::from("/x/y/paper.pdf")));
        assert!(matches!(app.view, View::ConvertPdf(_)));
    }

    #[test]
    fn home_to_viewer_and_back_home() {
        let mut app = app();
        // "View PDF" is index 3: Convert PDF, Convert Folder, File Explorer,
        // View PDF.
        for _ in 0..3 {
            app.handle_key(key(KeyCode::Char('j')));
        }
        app.handle_key(key(KeyCode::Enter));
        assert!(matches!(app.view, View::Viewer(_)));
        // Esc on the viewer's picker returns to the home menu.
        app.handle_key(key(KeyCode::Esc));
        assert!(matches!(app.view, View::Home));
    }

    #[test]
    fn opening_a_pdf_from_the_explorer_enters_the_viewer() {
        let mut app = app();
        app.apply(Transition::OpenViewFile(PathBuf::from("/x/y/paper.pdf")));
        assert!(matches!(app.view, View::Viewer(_)));
    }

    #[test]
    fn ctrl_c_quits_from_home() {
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }
}
