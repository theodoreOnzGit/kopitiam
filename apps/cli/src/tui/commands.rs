//! Thin surfaces onto the existing CLI command functions.
//!
//! Two shapes live here:
//!
//! * [`CommandPane`] — a scrollable, read-only text pane used by the **Status**
//!   view. The command is fast and pure, so its output is built into a `String`
//!   (via the `report` helper the command module exposes) and shown in-app. (The
//!   Models view is now interactive — see [`super::models`].)
//! * [`PlanPrompt`] — a one-line task input for the **Plan** view. Plan (and
//!   Scan) shell out to `cargo`/`rust-analyzer`/`rustdoc`, which are slow and
//!   stream their own progress, so rather than freeze the TUI those two run via
//!   the parent's suspend→run→restore path (see `super::run_suspended`) on the
//!   normal terminal; this prompt only collects the task string first.

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
};

use std::path::Path;

use super::theme::{CHILLI, DIM, GOLD, STEAM, TAN};
use super::Transition;

/// A scrollable, read-only pane of pre-computed text.
pub struct CommandPane {
    title: String,
    lines: Vec<String>,
    scroll: u16,
    is_error: bool,
}

impl CommandPane {
    /// Build the **Status** pane by invoking [`crate::status::report`].
    pub fn status(root: &Path) -> Self {
        match crate::status::report(root) {
            Ok(text) => Self::ok("kopitiam · status", text),
            Err(err) => Self::error("kopitiam · status", err.to_string()),
        }
    }

    fn ok(title: &str, text: String) -> Self {
        Self { title: title.to_string(), lines: split_lines(&text), scroll: 0, is_error: false }
    }

    fn error(title: &str, text: String) -> Self {
        let mut lines = vec!["This command reported an error:".to_string(), String::new()];
        lines.extend(split_lines(&text));
        Self { title: title.to_string(), lines, scroll: 0, is_error: true }
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![("↑/↓ j/k", "scroll"), ("Esc", "home")]
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Transition::Quit,
            KeyCode::Esc => Transition::Home,
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll = self.scroll.saturating_sub(1);
                Transition::Stay
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll = self.scroll.saturating_add(1);
                Transition::Stay
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(10);
                Transition::Stay
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
                Transition::Stay
            }
            _ => Transition::Stay,
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let border = if self.is_error { CHILLI } else { DIM };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));
        let inner_h = block.inner(area).height.max(1);
        let max_scroll = (self.lines.len() as u16).saturating_sub(inner_h);
        self.scroll = self.scroll.min(max_scroll);

        let lines: Vec<Line> = self
            .lines
            .iter()
            .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(TAN))))
            .collect();
        frame.render_widget(
            Paragraph::new(Text::from(lines)).scroll((self.scroll, 0)).block(block),
            area,
        );
    }
}

fn split_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        vec!["(no output)".to_string()]
    } else {
        text.lines().map(str::to_string).collect()
    }
}

/// The **Plan** view's task input: collect a one-line task, then hand it to the
/// parent to run via suspend→run→restore.
pub struct PlanPrompt {
    task: String,
}

impl PlanPrompt {
    pub fn new() -> Self {
        Self { task: String::new() }
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![("type", "task"), ("Enter", "run plan"), ("Esc", "home")]
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Transition::Quit,
            KeyCode::Esc => Transition::Home,
            KeyCode::Enter => {
                let task = self.task.trim();
                if task.is_empty() {
                    Transition::Stay
                } else {
                    Transition::RunPlan(task.to_string())
                }
            }
            KeyCode::Backspace => {
                self.task.pop();
                Transition::Stay
            }
            KeyCode::Char(c) if !ctrl => {
                self.task.push(c);
                Transition::Stay
            }
            _ => Transition::Stay,
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        frame.render_widget(
            Paragraph::new(
                "Runs the plan workflow over the current directory. It invokes cargo/rustdoc \
                 (and a model if one is installed), so the TUI hands the terminal over while it \
                 runs and returns here when done.",
            )
            .style(Style::default().fg(STEAM))
            .wrap(Wrap { trim: false }),
            chunks[0],
        );

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(GOLD))
            .title(Span::styled(" plan task ", Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(self.task.clone()),
                Span::styled("▌", Style::default().fg(GOLD)),
            ]))
            .block(block),
            chunks[1],
        );
    }
}
