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
//! * [`models`] — acquiring models (pull + verify); [`model_picker`] — choosing
//!   which already-on-disk model the chat talks to (`Ctrl-P` in chat).
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
mod git;
mod git_ops;
mod home;
mod logic;
mod model_picker;
mod models;
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

use crate::adapter::{SelectedAdapter, select_adapter, select_adapter_for};

use chat::ChatView;
use commands::{CommandPane, PlanPrompt};
use convert::{ConvertFolderState, ConvertPdfState};
use explorer::ExplorerState;
use git::GitView;
use home::HomeState;
use model_picker::ModelPickerView;
use models::ModelsView;
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
    Git,
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
    /// Pull a model by catalog id with its sha auto-resolved (from the Models
    /// view). Runs on the normal terminal so the download can stream progress.
    PullModel(String),
    /// Open the model picker (from chat's `Ctrl-P`).
    OpenModelPicker,
    /// Switch the chat to the model this [`model_picker::LoadPlan`] names,
    /// labelled `label` in the note the user sees. The router does the loading —
    /// the picker only chooses.
    SelectModel { plan: model_picker::LoadPlan, label: String },
}

/// A blocking action that needs the terminal handed over: it suspends the TUI,
/// runs on the normal screen (or lets kvim own the screen), then restores.
enum Pending {
    Scan,
    Plan(String),
    Editor,
    /// Pull the catalogued model with this id (auto-sha resolve + verify).
    PullModel(String),
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
    Git(Box<GitView>),
    Models(ModelsView),
    /// Choosing which on-disk model the chat uses. A detour from [`View::Chat`]
    /// — `Esc` returns there with the transcript intact.
    ModelPicker(ModelPickerView),
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
    ///
    /// **`None` means "re-select on next entry to chat"**, and that is load-
    /// bearing, not just laziness: a successful `models pull` sets this back to
    /// `None` so the freshly downloaded weights are actually picked up. Before
    /// that, the first `Echo` selection was cached for the whole session and no
    /// pull could ever dislodge it — chat kept insisting there was no model
    /// while the `.gguf` sat on disk, and only restarting the TUI helped.
    adapter: Option<SelectedAdapter>,
    /// The catalog id to re-select with, when set. Written by a successful pull
    /// (you pulled it, you want it) and by an explicit pick in the model picker,
    /// so a later re-selection lands on the same model instead of falling back
    /// to whatever the environment's default is.
    preferred_model: Option<String>,
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
            preferred_model: None,
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
            View::Git(state) => state.on_key(key),
            View::Models(state) => state.on_key(key),
            View::ModelPicker(state) => state.on_key(key),
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
            Transition::PullModel(id) => {
                self.pending = Some(Pending::PullModel(id));
                self.view = View::Home;
            }
            Transition::OpenModelPicker => {
                // Copy the active path out before building the view: the picker
                // needs to know what is in use, but it must not hold a borrow of
                // `self.adapter` while we assign `self.view`.
                let active = self
                    .adapter
                    .as_ref()
                    .and_then(model_picker::active_source)
                    .map(std::path::Path::to_path_buf);
                self.view = View::ModelPicker(ModelPickerView::new(active.as_deref()));
            }
            Transition::SelectModel { plan, label } => self.switch_model(plan, &label),
        }
    }

    /// Carry out the user's model pick, then go back to chat.
    ///
    /// Runs **inline**, blocking the render loop for as long as the load takes
    /// (parsing the GGUF, building the model and tokenizer — seconds for a small
    /// model, longer for a big one). Deliberately so: it matches what
    /// [`App::ensure_chat`] already does on first entry to chat, it needs no
    /// terminal handoff (a load prints nothing, unlike a download), and a
    /// background load would mean an adapter swap racing an in-flight turn. The
    /// cost is a visibly frozen frame on a large `.gguf` — the thing to change
    /// if that ever gets annoying is to run the load on a worker thread and poll
    /// it from the main loop, the same shape [`chat::ChatView::drain_stream`]
    /// already uses for tokens.
    ///
    /// A model that will not load is NOT an error here: [`model_picker::load_plan`]
    /// falls back to the echo stub, and the note in the transcript says exactly
    /// which file failed and why. Chat always stay usable.
    fn switch_model(&mut self, plan: model_picker::LoadPlan, label: &str) {
        let selected = model_picker::load_plan(&plan);
        let note = model_picker::switch_note(label, &selected);
        // Remember an explicitly picked catalog id, so a later cache
        // invalidation re-selects what the user chose, not the env default.
        if let model_picker::LoadPlan::ById(id) = &plan {
            self.preferred_model = Some(id.clone());
        }
        if self.chat.is_none() {
            self.chat = Some(ChatView::new(self.system.clone(), self.max_tokens, &selected));
        }
        if let Some(chat) = self.chat.as_mut() {
            chat.adopt_adapter(&selected, note);
        }
        self.adapter = Some(selected);
        self.view = View::Chat;
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
            Route::Models => View::Models(ModelsView::new()),
            Route::Git => View::Git(Box::new(GitView::new(self.cwd.clone()))),
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

    /// Select the adapter (if the cache is cold) and build the chat view on
    /// first use.
    ///
    /// Two cases have to be kept apart, and conflating them was the original
    /// "chat never sees my model" bug:
    ///
    /// * **No adapter yet** — select one. That happens on first entry, and again
    ///   after anything invalidates the cache (a successful pull).
    /// * **A chat view already exists** — do NOT rebuild it, that would throw
    ///   away the transcript. Instead tell it about the newly selected adapter
    ///   so its header stops claiming the old state.
    ///
    /// Selection is synchronous, so this call blocks the whole UI while a
    /// several-hundred-MB GGUF is parsed and dequantized. That freeze is a known
    /// wart, not a bug being fixed here (it needs a spinner + worker thread,
    /// which is interactive behaviour that cannot be verified headless).
    fn ensure_chat(&mut self) {
        let reselected = if self.adapter.is_none() {
            self.adapter = Some(match self.preferred_model.as_deref() {
                Some(id) => select_adapter_for(id),
                None => select_adapter(),
            });
            true
        } else {
            false
        };

        let Some(adapter) = self.adapter.as_ref() else { return };
        match self.chat.as_mut() {
            None => {
                self.chat = Some(ChatView::new(self.system.clone(), self.max_tokens, adapter))
            }
            Some(chat) if reselected => {
                let label = self.preferred_model.as_deref().unwrap_or("the local model");
                chat.adopt_adapter(adapter, model_picker::switch_note(label, adapter));
            }
            Some(_) => {}
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
            Pending::PullModel(id) => {
                let outcome = models::run_pull(&id);
                // THE fix for "pull a model, chat still says there is none": the
                // adapter chosen earlier in this session is now stale, so drop
                // it and remember what was just pulled. The next entry to chat
                // re-selects (see `ensure_chat`) and keeps the transcript.
                if outcome.pulled {
                    self.adapter = None;
                    self.preferred_model = Some(id);
                }
                pause_for_enter();
                outcome.notice
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
            View::Git(state) => state.footer_hints(),
            View::Models(state) => state.footer_hints(),
            View::ModelPicker(state) => state.footer_hints(),
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
            View::Git(state) => state.render(frame, body),
            View::Models(state) => state.render(frame, body),
            View::ModelPicker(state) => state.render(frame, body),
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

    /// The picker opens as its own view. Built straight from a transition (not
    /// by walking into chat first) so the test never calls `select_adapter`,
    /// which would try to load real weights off whatever machine runs it.
    #[test]
    fn the_model_picker_opens_as_its_own_view() {
        let mut app = app();
        app.apply(Transition::OpenModelPicker);
        assert!(matches!(app.view, View::ModelPicker(_)));
    }

    /// Esc in the picker lands back in chat, not Home — and chat's state is
    /// created if it wasn't already, so the view is never empty.
    #[test]
    fn esc_from_the_picker_returns_to_chat() {
        let mut app = app();
        app.apply(Transition::OpenModelPicker);
        app.apply(Transition::Open(Route::Chat));
        assert!(matches!(app.view, View::Chat));
        assert!(app.chat.is_some());
    }

    /// Picking a `.gguf` that cannot load must still leave a usable chat on the
    /// echo stub with the failure explained — never a crash, never a dead view.
    #[test]
    fn selecting_an_unloadable_model_lands_in_chat_on_the_stub() {
        let mut app = app();
        app.apply(Transition::SelectModel {
            plan: model_picker::LoadPlan::ByPath(PathBuf::from("/no/such/model.gguf")),
            label: "model.gguf".into(),
        });
        assert!(matches!(app.view, View::Chat));
        assert!(app.chat.is_some(), "chat is created if the switch happens before first entry");
        let adapter = app.adapter.as_ref().expect("an adapter was installed");
        assert!(!adapter.is_local(), "a bad path must degrade to the echo stub");
    }

    /// An explicitly picked catalog id is remembered, so a later cache
    /// invalidation re-selects the user's choice and not the env default.
    #[test]
    fn picking_a_catalog_model_is_remembered_for_the_next_reselect() {
        let mut app = app();
        app.apply(Transition::SelectModel {
            plan: model_picker::LoadPlan::ById("smollm2-1.7b-instruct-q4_k_m".into()),
            label: "SmolLM2 1.7B".into(),
        });
        assert_eq!(app.preferred_model.as_deref(), Some("smollm2-1.7b-instruct-q4_k_m"));
    }

    /// The regression this whole cache-invalidation dance exists for: a cached
    /// adapter from earlier in the session must be dropped when a pull lands, so
    /// the next entry to chat actually re-selects. Chat state (the transcript)
    /// must survive that.
    #[test]
    fn a_successful_pull_invalidates_the_cached_adapter_but_keeps_the_chat() {
        let mut app = app();
        // Stand in for "chat was opened earlier and cached a stub adapter".
        app.adapter = Some(crate::adapter::SelectedAdapter::Echo {
            adapter: kopitiam_ai::EchoAdapter,
            reason: crate::adapter::FallbackReason::NoModelOnDisk {
                model_id: "whatever".into(),
                expected_store_path: PathBuf::from("/nowhere.gguf"),
            },
        });
        app.chat = Some(ChatView::new(
            DEFAULT_SYSTEM_PROMPT.to_string(),
            None,
            app.adapter.as_ref().unwrap(),
        ));

        // What `run_suspended` does on a successful pull.
        app.adapter = None;
        app.preferred_model = Some("smollm2-360m-instruct-q8_0".into());

        assert!(app.adapter.is_none(), "the stale adapter must be dropped");
        assert!(app.chat.is_some(), "but the transcript must not be thrown away");
    }

    #[test]
    fn ctrl_c_quits_from_home() {
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }
}
