//! The **Home** view: the app's front door — a menu that routes to every other
//! view. Arrow / `j` / `k` move the selection, `Enter` opens, `q` quits.

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph},
};

use super::theme::{BANNER, DIM, GOLD, STEAM, TAN};
use super::{Route, Transition};

/// One entry in the home menu: a label, a short blurb, and the route it opens.
struct MenuEntry {
    label: &'static str,
    blurb: &'static str,
    route: Option<Route>,
}

/// The static menu, in display order. `Quit` carries no route — selecting it
/// ends the app.
const MENU: &[MenuEntry] = &[
    MenuEntry { label: "Convert PDF", blurb: "fuzzy-find one PDF and convert it to Markdown", route: Some(Route::ConvertPdf) },
    MenuEntry { label: "Convert Folder", blurb: "recursively batch-convert a folder of PDFs", route: Some(Route::ConvertFolder) },
    MenuEntry { label: "File Explorer", blurb: "browse directories; open a PDF into the viewer", route: Some(Route::Explorer) },
    MenuEntry { label: "View PDF", blurb: "fuzzy-find a PDF and read it (reflow / image modes)", route: Some(Route::Viewer) },
    MenuEntry { label: "AI Chat", blurb: "chat with the local kopi-o model", route: Some(Route::Chat) },
    MenuEntry { label: "kvim (Editor)", blurb: "open the kvim modal editor; returns here on exit", route: Some(Route::Editor) },
    MenuEntry { label: "Scan", blurb: "scan a Rust project's real tooling", route: Some(Route::Scan) },
    MenuEntry { label: "Status", blurb: "show a project's persisted session memory", route: Some(Route::Status) },
    MenuEntry { label: "Plan", blurb: "run the plan workflow over a project", route: Some(Route::Plan) },
    MenuEntry { label: "Git", blurb: "lazygit-style local git panel (status/log/branches/commit)", route: Some(Route::Git) },
    MenuEntry { label: "Models", blurb: "browse the catalog; pull a model with its sha auto-resolved", route: Some(Route::Models) },
    MenuEntry { label: "Quit", blurb: "leave the kopitiam", route: None },
];

/// Home menu state: just the current selection index.
#[derive(Default)]
pub struct HomeState {
    selected: usize,
    /// A transient one-line notice (e.g. the result of returning from kvim),
    /// shown under the menu until the next navigation.
    notice: Option<String>,
}

impl HomeState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a transient notice shown beneath the menu.
    pub fn set_notice(&mut self, notice: impl Into<String>) {
        self.notice = Some(notice.into());
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![("↑/↓ j/k", "move"), ("Enter", "open"), ("q", "quit")]
    }

    fn move_up(&mut self) {
        self.selected = self.selected.checked_sub(1).unwrap_or(MENU.len() - 1);
    }

    fn move_down(&mut self) {
        self.selected = (self.selected + 1) % MENU.len();
    }

    /// Handle one key and return the router transition to apply.
    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Transition::Quit,
            KeyCode::Char('q') => Transition::Quit,
            KeyCode::Up | KeyCode::Char('k') => {
                self.notice = None;
                self.move_up();
                Transition::Stay
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.notice = None;
                self.move_down();
                Transition::Stay
            }
            KeyCode::Enter => {
                self.notice = None;
                match MENU[self.selected].route {
                    Some(route) => Transition::Open(route),
                    None => Transition::Quit,
                }
            }
            _ => Transition::Stay,
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        // Banner on top when there is comfortable room, then the menu.
        let show_banner = area.height >= 22 && area.width >= 60;
        let banner_h = if show_banner { BANNER.len() as u16 + 1 } else { 0 };
        let notice_h = if self.notice.is_some() { 1 } else { 0 };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(banner_h),
                Constraint::Min(3),
                Constraint::Length(notice_h),
            ])
            .split(area);

        if show_banner {
            let lines: Vec<Line> = BANNER
                .iter()
                .map(|(art, tone)| Line::from(Span::styled(*art, Style::default().fg(*tone))))
                .collect();
            frame.render_widget(
                Paragraph::new(Text::from(lines)).alignment(Alignment::Center),
                chunks[0],
            );
        }

        let items: Vec<ListItem> = MENU
            .iter()
            .map(|entry| {
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<16}", entry.label),
                        Style::default().fg(TAN).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(entry.blurb, Style::default().fg(DIM)),
                ]))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(
                " kopitiam · menu ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));

        let list = List::new(items).block(block).highlight_symbol("▍ ").highlight_style(
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
        let mut state = ListState::default();
        state.select(Some(self.selected));
        frame.render_stateful_widget(list, chunks[1], &mut state);

        if let Some(notice) = &self.notice {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    notice.clone(),
                    Style::default().fg(STEAM).add_modifier(Modifier::ITALIC),
                ))),
                chunks[2],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn j_and_k_wrap_around_the_menu() {
        let mut home = HomeState::new();
        assert_eq!(home.selected, 0);
        home.on_key(key(KeyCode::Char('k'))); // wraps to last
        assert_eq!(home.selected, MENU.len() - 1);
        home.on_key(key(KeyCode::Char('j'))); // wraps back to first
        assert_eq!(home.selected, 0);
    }

    #[test]
    fn enter_on_first_entry_opens_convert_pdf() {
        let mut home = HomeState::new();
        assert!(matches!(
            home.on_key(key(KeyCode::Enter)),
            Transition::Open(Route::ConvertPdf)
        ));
    }

    #[test]
    fn q_quits_from_home() {
        let mut home = HomeState::new();
        assert!(matches!(home.on_key(key(KeyCode::Char('q'))), Transition::Quit));
    }

    #[test]
    fn last_entry_quits() {
        let mut home = HomeState::new();
        for _ in 0..MENU.len() - 1 {
            home.on_key(key(KeyCode::Char('j')));
        }
        assert_eq!(MENU[home.selected].label, "Quit");
        assert!(matches!(home.on_key(key(KeyCode::Enter)), Transition::Quit));
    }
}
