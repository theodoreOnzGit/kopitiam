//! The **Git** view: a lazygit-style, keyboard-driven, multi-panel git panel
//! (#25) backed entirely by pure-Rust `gix` ([`super::git_ops`]) — LOCAL ops
//! only, no `git` binary ever shelled out to.
//!
//! # Panels & keys (clean-room from `vendor/lazygit`'s model)
//!
//! Three stacked left panels — **Status** (files), **Branches**, **Log**
//! (commits) — plus a right-hand **Diff/preview** pane that reflects the focused
//! panel's selection. `Tab`/`BackTab` cycle focus; `j`/`k`/arrows move; the diff
//! pane scrolls with `PgUp`/`PgDn`.
//!
//! * **Status**: `space` stages an unstaged file / unstages an already-staged
//!   one; `d` discards a file's unstaged changes (behind a `y/n` confirm);
//!   `c` opens the commit-message input.
//! * **Branches**: `n` (or `b`) creates a branch at HEAD. Checkout is **gated**
//!   (needs the disabled `worktree-mutation` feature) — pressing `Enter` says so.
//! * **Log**: the diff pane lists the selected commit's changed files.
//! * **Push/pull are gated** (`p`/`P` explain why: gix 0.86 has no high-level
//!   push, kopitiam #28).
//!
//! # Testability
//!
//! All key handling is a pure `(GitView, KeyEvent) -> Transition` function over
//! in-memory state, and the panel/selection/mode transitions are unit-tested
//! with seeded data — no repo, no TTY. The gix mutations themselves are exercised
//! by round-trip tests against a throwaway `gix`-init'd tempdir in
//! [`super::git_ops`]'s siblings; the interactive flows still need a human smoke
//! test (see the crate's TUI notes).

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use std::path::PathBuf;

use super::git_ops::{BranchInfo, CommitInfo, DiffLine, DiffLineKind, GitRepo, StatusEntry};
use super::theme::{CHILLI, DIM, GOLD, PALM, STEAM, TAN, USER};
use super::Transition;

/// Which of the three left panels currently has focus.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Panel {
    Status,
    Branches,
    Log,
}

impl Panel {
    fn next(self) -> Self {
        match self {
            Panel::Status => Panel::Branches,
            Panel::Branches => Panel::Log,
            Panel::Log => Panel::Status,
        }
    }
    fn prev(self) -> Self {
        match self {
            Panel::Status => Panel::Log,
            Panel::Branches => Panel::Status,
            Panel::Log => Panel::Branches,
        }
    }
}

/// A modal overlay the view can be in. `Normal` is the panel-navigation state.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Mode {
    Normal,
    /// Confirm discarding the given path's unstaged changes.
    ConfirmDiscard(String),
    /// Collecting a commit message.
    CommitInput,
    /// Collecting a new branch name.
    BranchInput,
}

/// The Git view state.
pub struct GitView {
    cwd: PathBuf,
    repo: Option<GitRepo>,
    /// Set when the CWD is not a git repo (or the repo failed to open).
    error: Option<String>,
    head_branch: Option<String>,

    focus: Panel,
    status: Vec<StatusEntry>,
    branches: Vec<BranchInfo>,
    log: Vec<CommitInfo>,
    sel_status: usize,
    sel_branch: usize,
    sel_log: usize,

    /// The right-pane preview lines for the current focus/selection.
    preview: Vec<DiffLine>,
    preview_scroll: u16,

    mode: Mode,
    input: String,
    notice: Option<String>,
}

/// How many commits to walk for the log panel.
const LOG_LIMIT: usize = 200;

impl GitView {
    /// Open the view on `cwd`'s repository, loading the initial panels. A missing
    /// repo is not an error to the router — it renders a friendly notice.
    pub fn new(cwd: PathBuf) -> Self {
        let mut view = Self {
            cwd,
            repo: None,
            error: None,
            head_branch: None,
            focus: Panel::Status,
            status: Vec::new(),
            branches: Vec::new(),
            log: Vec::new(),
            sel_status: 0,
            sel_branch: 0,
            sel_log: 0,
            preview: Vec::new(),
            preview_scroll: 0,
            mode: Mode::Normal,
            input: String::new(),
            notice: None,
        };
        match GitRepo::discover(&view.cwd) {
            Ok(repo) => {
                view.repo = Some(repo);
                view.reload();
            }
            Err(err) => view.error = Some(err.to_string()),
        }
        view
    }

    /// Re-read status, branches, and log from the repo and clamp selections.
    fn reload(&mut self) {
        let Some(repo) = self.repo.as_ref() else { return };
        self.head_branch = repo.head_branch();
        self.status = repo.status().unwrap_or_default();
        self.branches = repo.branches().unwrap_or_default();
        self.log = repo.log(LOG_LIMIT).unwrap_or_default();
        self.sel_status = clamp(self.sel_status, self.status.len());
        self.sel_branch = clamp(self.sel_branch, self.branches.len());
        self.sel_log = clamp(self.sel_log, self.log.len());
        self.refresh_preview();
    }

    /// Rebuild the right-pane preview for the focused panel's current selection.
    fn refresh_preview(&mut self) {
        self.preview_scroll = 0;
        let Some(repo) = self.repo.as_ref() else {
            self.preview = Vec::new();
            return;
        };
        self.preview = match self.focus {
            Panel::Status => match self.status.get(self.sel_status) {
                Some(entry) => repo.file_diff(entry).unwrap_or_else(|e| {
                    vec![DiffLine { kind: DiffLineKind::Meta, text: format!("(diff error: {e})") }]
                }),
                None => Vec::new(),
            },
            Panel::Log => match self.log.get(self.sel_log) {
                Some(commit) => commit_preview(repo, commit),
                None => Vec::new(),
            },
            Panel::Branches => Vec::new(),
        };
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        match &self.mode {
            Mode::ConfirmDiscard(_) => vec![("y", "discard"), ("n/Esc", "cancel")],
            Mode::CommitInput => vec![("type", "message"), ("Enter", "commit"), ("Esc", "cancel")],
            Mode::BranchInput => vec![("type", "name"), ("Enter", "create"), ("Esc", "cancel")],
            Mode::Normal => vec![
                ("Tab", "panel"),
                ("j/k", "move"),
                ("space", "stage"),
                ("c", "commit"),
                ("d", "discard"),
                ("n", "branch"),
                ("r", "refresh"),
                ("q", "home"),
            ],
        }
    }

    /// Handle one key. Pure over `(self, key)` → [`Transition`].
    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && key.code == KeyCode::Char('c') {
            return Transition::Quit;
        }
        match self.mode.clone() {
            Mode::Normal => self.on_key_normal(key),
            Mode::ConfirmDiscard(path) => self.on_key_confirm(key, &path),
            Mode::CommitInput => self.on_key_commit_input(key),
            Mode::BranchInput => self.on_key_branch_input(key),
        }
    }

    fn on_key_normal(&mut self, key: KeyEvent) -> Transition {
        // When the CWD is not a git repo, only navigation out is meaningful.
        // (A `None` repo without an error is the test seam: navigation/mode
        // transitions run, while the mutating branches self-guard on the repo.)
        if self.error.is_some() {
            return match key.code {
                KeyCode::Esc | KeyCode::Char('q') => Transition::Home,
                _ => Transition::Stay,
            };
        }
        self.notice = None;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return Transition::Home,
            KeyCode::Tab => {
                self.focus = self.focus.next();
                self.refresh_preview();
            }
            KeyCode::BackTab => {
                self.focus = self.focus.prev();
                self.refresh_preview();
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::PageDown => self.preview_scroll = self.preview_scroll.saturating_add(10),
            KeyCode::PageUp => self.preview_scroll = self.preview_scroll.saturating_sub(10),
            KeyCode::Char(' ') => self.toggle_stage(),
            KeyCode::Char('c') => {
                self.input.clear();
                self.mode = Mode::CommitInput;
            }
            KeyCode::Char('n') | KeyCode::Char('b') => {
                self.input.clear();
                self.mode = Mode::BranchInput;
            }
            KeyCode::Char('d') => self.begin_discard(),
            KeyCode::Char('r') => self.reload(),
            KeyCode::Char('p') | KeyCode::Char('P') => {
                self.notice =
                    Some("push/pull gated: gix 0.86 has no high-level push (pending #28)".into());
            }
            KeyCode::Enter if self.focus == Panel::Branches => {
                self.notice = Some(
                    "checkout gated: needs gix worktree-mutation (create branches with n)".into(),
                );
            }
            _ => {}
        }
        Transition::Stay
    }

    fn move_selection(&mut self, delta: isize) {
        match self.focus {
            Panel::Status => self.sel_status = step(self.sel_status, self.status.len(), delta),
            Panel::Branches => self.sel_branch = step(self.sel_branch, self.branches.len(), delta),
            Panel::Log => self.sel_log = step(self.sel_log, self.log.len(), delta),
        }
        self.refresh_preview();
    }

    /// `space` on the status panel: stage an unstaged file, else unstage a staged
    /// one.
    fn toggle_stage(&mut self) {
        if self.focus != Panel::Status {
            return;
        }
        let Some(entry) = self.status.get(self.sel_status).cloned() else { return };
        let Some(repo) = self.repo.as_ref() else { return };
        let result = if entry.unstaged.is_some() {
            repo.stage(&entry.path)
        } else {
            repo.unstage(&entry.path)
        };
        match result {
            Ok(()) => self.reload(),
            Err(e) => self.notice = Some(format!("stage failed: {e}")),
        }
    }

    fn begin_discard(&mut self) {
        if self.focus != Panel::Status {
            return;
        }
        if let Some(entry) = self.status.get(self.sel_status) {
            if entry.unstaged.is_some() {
                self.mode = Mode::ConfirmDiscard(entry.path.clone());
            } else {
                self.notice = Some("nothing unstaged to discard on this file".into());
            }
        }
    }

    fn on_key_confirm(&mut self, key: KeyEvent, path: &str) -> Transition {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                if let Some(repo) = self.repo.as_ref() {
                    match repo.discard(path) {
                        Ok(()) => {
                            self.notice = Some(format!("discarded unstaged changes in {path}"));
                            self.reload();
                        }
                        Err(e) => self.notice = Some(format!("discard failed: {e}")),
                    }
                }
                self.mode = Mode::Normal;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => self.mode = Mode::Normal,
            _ => {}
        }
        Transition::Stay
    }

    fn on_key_commit_input(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => self.mode = Mode::Normal,
            KeyCode::Enter => {
                let msg = self.input.trim().to_string();
                if msg.is_empty() {
                    self.notice = Some("commit message is empty".into());
                    return Transition::Stay;
                }
                if let Some(repo) = self.repo.as_ref() {
                    match repo.commit(&msg) {
                        Ok(short) => self.notice = Some(format!("committed {short}")),
                        Err(e) => self.notice = Some(format!("commit failed: {e}")),
                    }
                }
                self.input.clear();
                self.mode = Mode::Normal;
                self.reload();
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) => self.input.push(c),
            _ => {}
        }
        Transition::Stay
    }

    fn on_key_branch_input(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => self.mode = Mode::Normal,
            KeyCode::Enter => {
                let name = self.input.trim().to_string();
                if name.is_empty() {
                    self.notice = Some("branch name is empty".into());
                    return Transition::Stay;
                }
                if let Some(repo) = self.repo.as_ref() {
                    match repo.create_branch(&name) {
                        Ok(()) => self.notice = Some(format!("created branch {name}")),
                        Err(e) => self.notice = Some(format!("branch failed: {e}")),
                    }
                }
                self.input.clear();
                self.mode = Mode::Normal;
                self.reload();
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) => self.input.push(c),
            _ => {}
        }
        Transition::Stay
    }

    // ---- rendering -------------------------------------------------------

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if let Some(err) = &self.error {
            self.render_no_repo(frame, area, err);
            return;
        }

        // Reserve a bottom row for input/confirm/notice when needed.
        let bottom = if self.mode != Mode::Normal || self.notice.is_some() { 3 } else { 0 };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(bottom)])
            .split(area);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[0]);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(45),
                Constraint::Percentage(25),
                Constraint::Percentage(30),
            ])
            .split(cols[0]);

        self.render_status_panel(frame, left[0]);
        self.render_branches_panel(frame, left[1]);
        self.render_log_panel(frame, left[2]);
        self.render_preview(frame, cols[1]);

        if bottom > 0 {
            self.render_bottom(frame, rows[1]);
        }
    }

    fn render_no_repo(&self, frame: &mut Frame, area: Rect, err: &str) {
        let block = panel_block("git", false).border_style(Style::default().fg(CHILLI));
        let text = Text::from(vec![
            Line::from(Span::styled("Not a git repository.", Style::default().fg(CHILLI))),
            Line::raw(""),
            Line::from(Span::styled(
                self.cwd.display().to_string(),
                Style::default().fg(DIM),
            )),
            Line::from(Span::styled(err.to_string(), Style::default().fg(DIM))),
            Line::raw(""),
            Line::from(Span::styled("Press q / Esc to return to the menu.", Style::default().fg(STEAM))),
        ]);
        frame.render_widget(Paragraph::new(text).block(block).wrap(Wrap { trim: false }), area);
    }

    fn render_status_panel(&self, frame: &mut Frame, area: Rect) {
        let focused = self.focus == Panel::Status;
        let items: Vec<ListItem> = self
            .status
            .iter()
            .map(|e| {
                let (x, y) = e.xy();
                ListItem::new(Line::from(vec![
                    Span::styled(x.to_string(), Style::default().fg(PALM).add_modifier(Modifier::BOLD)),
                    Span::styled(y.to_string(), Style::default().fg(CHILLI).add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(e.path.clone(), Style::default().fg(TAN)),
                ]))
            })
            .collect();
        let title = if self.status.is_empty() {
            " status · clean (untracked not shown) ".to_string()
        } else {
            format!(" status · {} changed ", self.status.len())
        };
        self.render_list(frame, area, &title, items, self.sel_status, focused);
    }

    fn render_branches_panel(&self, frame: &mut Frame, area: Rect) {
        let focused = self.focus == Panel::Branches;
        let items: Vec<ListItem> = self
            .branches
            .iter()
            .map(|b| {
                let marker = if b.is_head { "* " } else { "  " };
                ListItem::new(Line::from(vec![
                    Span::styled(marker, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)),
                    Span::styled(
                        b.name.clone(),
                        Style::default().fg(if b.is_head { GOLD } else { TAN }),
                    ),
                    Span::styled(format!("  {}", b.short_id), Style::default().fg(DIM)),
                ]))
            })
            .collect();
        self.render_list(frame, area, " branches ", items, self.sel_branch, focused);
    }

    fn render_log_panel(&self, frame: &mut Frame, area: Rect) {
        let focused = self.focus == Panel::Log;
        let items: Vec<ListItem> = self
            .log
            .iter()
            .map(|c| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{} ", c.short_id), Style::default().fg(GOLD)),
                    Span::styled(format!("{} ", c.date), Style::default().fg(DIM)),
                    Span::styled(c.summary.clone(), Style::default().fg(TAN)),
                ]))
            })
            .collect();
        self.render_list(frame, area, " commits ", items, self.sel_log, focused);
    }

    fn render_list(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        items: Vec<ListItem>,
        selected: usize,
        focused: bool,
    ) {
        let block = panel_block(title, focused);
        let list = List::new(items).block(block).highlight_symbol("▍ ").highlight_style(
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
        let mut state = ListState::default();
        if focused {
            state.select(Some(selected));
        }
        frame.render_stateful_widget(list, area, &mut state);
    }

    fn render_preview(&mut self, frame: &mut Frame, area: Rect) {
        let title = match self.focus {
            Panel::Status => " diff ",
            Panel::Log => " commit files ",
            Panel::Branches => " branches ",
        };
        let block = panel_block(title, false);
        let inner_h = block.inner(area).height.max(1);

        let lines: Vec<Line> = if matches!(self.focus, Panel::Branches) {
            branch_preview(&self.head_branch)
        } else {
            self.preview.iter().map(diff_line_to_line).collect()
        };
        let max_scroll = (lines.len() as u16).saturating_sub(inner_h);
        self.preview_scroll = self.preview_scroll.min(max_scroll);
        frame.render_widget(
            Paragraph::new(Text::from(lines)).scroll((self.preview_scroll, 0)).block(block),
            area,
        );
    }

    fn render_bottom(&self, frame: &mut Frame, area: Rect) {
        let (title, body, color) = match &self.mode {
            Mode::CommitInput => (
                " commit message ",
                Line::from(vec![
                    Span::raw(self.input.clone()),
                    Span::styled("▌", Style::default().fg(GOLD)),
                ]),
                GOLD,
            ),
            Mode::BranchInput => (
                " new branch name ",
                Line::from(vec![
                    Span::raw(self.input.clone()),
                    Span::styled("▌", Style::default().fg(GOLD)),
                ]),
                GOLD,
            ),
            Mode::ConfirmDiscard(path) => (
                " confirm discard ",
                Line::from(Span::styled(
                    format!("Discard unstaged changes in {path}? (y/n)"),
                    Style::default().fg(CHILLI),
                )),
                CHILLI,
            ),
            Mode::Normal => (
                " note ",
                Line::from(Span::styled(
                    self.notice.clone().unwrap_or_default(),
                    Style::default().fg(STEAM).add_modifier(Modifier::ITALIC),
                )),
                DIM,
            ),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(color))
            .title(Span::styled(title.to_string(), Style::default().fg(color).add_modifier(Modifier::BOLD)));
        frame.render_widget(Paragraph::new(body).block(block), area);
    }
}

/// Clamp `sel` to `[0, len)`, or 0 when empty.
fn clamp(sel: usize, len: usize) -> usize {
    if len == 0 { 0 } else { sel.min(len - 1) }
}

/// Move a selection by `delta` within `[0, len)`, saturating at the ends.
fn step(sel: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let max = len - 1;
    let next = sel as isize + delta;
    next.clamp(0, max as isize) as usize
}

/// Build the right-pane preview for a selected commit: a header plus its changed
/// files (structural add/modify/delete).
fn commit_preview(repo: &GitRepo, commit: &CommitInfo) -> Vec<DiffLine> {
    let mut lines = vec![
        DiffLine { kind: DiffLineKind::Meta, text: format!("commit {}", commit.short_id) },
        DiffLine { kind: DiffLineKind::Meta, text: format!("author {}", commit.author) },
        DiffLine { kind: DiffLineKind::Meta, text: format!("date   {}", commit.date) },
        DiffLine { kind: DiffLineKind::Meta, text: format!("    {}", commit.summary) },
        DiffLine { kind: DiffLineKind::Meta, text: String::new() },
    ];
    match repo.commit_changed(&commit.full_id) {
        Ok(files) if files.is_empty() => {
            lines.push(DiffLine { kind: DiffLineKind::Meta, text: "(no file changes)".into() });
        }
        Ok(files) => {
            for f in files {
                let (x, _) = f.xy();
                let kind = match f.staged {
                    Some(super::git_ops::ChangeKind::Deleted) => DiffLineKind::Del,
                    Some(super::git_ops::ChangeKind::Added) => DiffLineKind::Add,
                    _ => DiffLineKind::Context,
                };
                lines.push(DiffLine { kind, text: format!("{x}  {}", f.path) });
            }
        }
        Err(e) => lines.push(DiffLine { kind: DiffLineKind::Meta, text: format!("(error: {e})") }),
    }
    lines
}

fn branch_preview(head: &Option<String>) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            format!("HEAD → {}", head.clone().unwrap_or_else(|| "(detached)".into())),
            Style::default().fg(GOLD),
        )),
        Line::raw(""),
        Line::from(Span::styled("n / b   create a branch at HEAD", Style::default().fg(TAN))),
        Line::from(Span::styled(
            "Enter   checkout — GATED (needs gix worktree-mutation)",
            Style::default().fg(DIM),
        )),
        Line::from(Span::styled(
            "p / P   push / pull — GATED (gix 0.86, pending #28)",
            Style::default().fg(DIM),
        )),
    ]
}

fn diff_line_to_line(dl: &DiffLine) -> Line<'static> {
    let (prefix, color) = match dl.kind {
        DiffLineKind::Add => ("+", PALM),
        DiffLineKind::Del => ("-", CHILLI),
        DiffLineKind::Context => (" ", DIM),
        DiffLineKind::Meta => ("", USER),
    };
    Line::from(Span::styled(format!("{prefix}{}", dl.text), Style::default().fg(color)))
}

fn panel_block(title: &str, focused: bool) -> Block<'static> {
    let color = if focused { GOLD } else { DIM };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(color))
        .title(
            Span::styled(
                format!(" {} ", title.trim()),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
        )
        .title_alignment(Alignment::Left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::git_ops::ChangeKind;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    /// A view with seeded panels but NO real repo, so we test pure UI-state
    /// transitions (focus/selection/mode) without touching git.
    fn seeded_view() -> GitView {
        GitView {
            cwd: PathBuf::from("."),
            repo: None,
            error: None,
            head_branch: Some("main".into()),
            focus: Panel::Status,
            status: vec![
                StatusEntry { path: "a.rs".into(), staged: None, unstaged: Some(ChangeKind::Modified) },
                StatusEntry { path: "b.rs".into(), staged: Some(ChangeKind::Added), unstaged: None },
            ],
            branches: vec![
                BranchInfo { name: "main".into(), is_head: true, short_id: "abc1234".into() },
                BranchInfo { name: "dev".into(), is_head: false, short_id: "def5678".into() },
            ],
            log: vec![CommitInfo {
                short_id: "abc1234".into(),
                full_id: "abc1234000000000000000000000000000000000".into(),
                summary: "init".into(),
                author: "t".into(),
                date: "2026-01-01".into(),
            }],
            sel_status: 0,
            sel_branch: 0,
            sel_log: 0,
            preview: Vec::new(),
            preview_scroll: 0,
            mode: Mode::Normal,
            input: String::new(),
            notice: None,
        }
    }

    #[test]
    fn panel_focus_cycles_forward_and_back() {
        assert_eq!(Panel::Status.next(), Panel::Branches);
        assert_eq!(Panel::Branches.next(), Panel::Log);
        assert_eq!(Panel::Log.next(), Panel::Status);
        assert_eq!(Panel::Status.prev(), Panel::Log);
    }

    #[test]
    fn tab_moves_focus_between_panels() {
        let mut v = seeded_view();
        assert_eq!(v.focus, Panel::Status);
        v.on_key(key(KeyCode::Tab));
        assert_eq!(v.focus, Panel::Branches);
        v.on_key(key(KeyCode::BackTab));
        assert_eq!(v.focus, Panel::Status);
    }

    #[test]
    fn jk_moves_selection_and_saturates() {
        let mut v = seeded_view();
        assert_eq!(v.sel_status, 0);
        v.on_key(key(KeyCode::Char('j')));
        assert_eq!(v.sel_status, 1);
        v.on_key(key(KeyCode::Char('j'))); // saturate at last (len 2)
        assert_eq!(v.sel_status, 1);
        v.on_key(key(KeyCode::Char('k')));
        assert_eq!(v.sel_status, 0);
        v.on_key(key(KeyCode::Char('k'))); // saturate at first
        assert_eq!(v.sel_status, 0);
    }

    #[test]
    fn c_opens_commit_input_and_esc_cancels() {
        let mut v = seeded_view();
        v.on_key(key(KeyCode::Char('c')));
        assert_eq!(v.mode, Mode::CommitInput);
        v.on_key(key(KeyCode::Char('x')));
        assert_eq!(v.input, "x");
        v.on_key(key(KeyCode::Esc));
        assert_eq!(v.mode, Mode::Normal);
    }

    #[test]
    fn n_opens_branch_input() {
        let mut v = seeded_view();
        v.on_key(key(KeyCode::Char('n')));
        assert_eq!(v.mode, Mode::BranchInput);
    }

    #[test]
    fn d_on_unstaged_file_opens_confirm() {
        let mut v = seeded_view(); // sel 0 = a.rs, unstaged
        v.on_key(key(KeyCode::Char('d')));
        assert_eq!(v.mode, Mode::ConfirmDiscard("a.rs".into()));
        v.on_key(key(KeyCode::Char('n')));
        assert_eq!(v.mode, Mode::Normal);
    }

    #[test]
    fn d_on_staged_only_file_sets_notice_not_confirm() {
        let mut v = seeded_view();
        v.sel_status = 1; // b.rs, staged only
        v.on_key(key(KeyCode::Char('d')));
        assert_eq!(v.mode, Mode::Normal);
        assert!(v.notice.is_some());
    }

    #[test]
    fn push_key_is_gated_with_a_notice() {
        let mut v = seeded_view();
        v.on_key(key(KeyCode::Char('p')));
        assert!(v.notice.as_deref().unwrap().contains("gated"));
    }

    #[test]
    fn checkout_on_branches_is_gated() {
        let mut v = seeded_view();
        v.on_key(key(KeyCode::Tab)); // focus Branches
        v.on_key(key(KeyCode::Enter));
        assert!(v.notice.as_deref().unwrap().contains("checkout gated"));
    }

    #[test]
    fn q_returns_home() {
        let mut v = seeded_view();
        assert!(matches!(v.on_key(key(KeyCode::Char('q'))), Transition::Home));
    }

    #[test]
    fn ctrl_c_quits() {
        let mut v = seeded_view();
        let ev = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(v.on_key(ev), Transition::Quit));
    }

    #[test]
    fn empty_commit_message_is_rejected() {
        let mut v = seeded_view();
        v.on_key(key(KeyCode::Char('c')));
        v.on_key(key(KeyCode::Enter)); // empty
        assert_eq!(v.mode, Mode::CommitInput);
        assert!(v.notice.is_some());
    }
}
