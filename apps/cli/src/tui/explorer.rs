//! The **File Explorer** view: a small, robust directory browser. Enter a
//! directory to descend, `..` to go up, and pressing `Enter` on a `.pdf` jumps
//! straight into the Convert PDF flow pre-selected on that file.

use std::path::{Path, PathBuf};

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState},
};

use super::logic::is_pdf;
use super::theme::{DIM, GOLD, PALM, STEAM, TAN, USER};
use super::Transition;

/// One row in the listing.
struct Entry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    /// File size in bytes; `None` for directories and the `..` row.
    size: Option<u64>,
    /// The synthetic parent (`..`) row.
    is_parent: bool,
}

/// File Explorer state: the current directory and its (sorted) entries.
pub struct ExplorerState {
    cwd: PathBuf,
    entries: Vec<Entry>,
    selected: usize,
}

impl ExplorerState {
    /// Open the explorer rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        let mut state = Self { cwd, entries: Vec::new(), selected: 0 };
        state.reload();
        state
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![("↑/↓ j/k", "move"), ("Enter", "open"), ("Esc", "home")]
    }

    /// Re-read the current directory. Unreadable entries are skipped; a fully
    /// unreadable directory just yields the `..` row so the user can back out.
    fn reload(&mut self) {
        let mut dirs: Vec<Entry> = Vec::new();
        let mut files: Vec<Entry> = Vec::new();

        if let Ok(read) = std::fs::read_dir(&self.cwd) {
            for entry in read.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                let (is_dir, size) = match entry.metadata() {
                    Ok(meta) if meta.is_dir() => (true, None),
                    Ok(meta) => (false, Some(meta.len())),
                    Err(_) => (path.is_dir(), None),
                };
                let row = Entry { name, path, is_dir, size, is_parent: false };
                if is_dir { dirs.push(row) } else { files.push(row) }
            }
        }
        dirs.sort_by_key(|entry| entry.name.to_lowercase());
        files.sort_by_key(|entry| entry.name.to_lowercase());

        let mut entries = Vec::with_capacity(dirs.len() + files.len() + 1);
        if self.cwd.parent().is_some() {
            entries.push(Entry {
                name: "..".to_string(),
                path: self.cwd.parent().map(Path::to_path_buf).unwrap_or_else(|| self.cwd.clone()),
                is_dir: true,
                size: None,
                is_parent: true,
            });
        }
        entries.extend(dirs);
        entries.extend(files);

        self.entries = entries;
        self.selected = 0;
    }

    fn move_by(&mut self, delta: i32) {
        if self.entries.is_empty() {
            return;
        }
        let len = self.entries.len() as i32;
        self.selected = (self.selected as i32 + delta).rem_euclid(len) as usize;
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Transition::Quit,
            KeyCode::Esc => Transition::Home,
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_by(-1);
                Transition::Stay
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_by(1);
                Transition::Stay
            }
            KeyCode::Enter => self.activate(),
            _ => Transition::Stay,
        }
    }

    /// Act on the highlighted row: descend into a directory, ascend on `..`, or
    /// open a PDF into the converter.
    fn activate(&mut self) -> Transition {
        let Some(entry) = self.entries.get(self.selected) else {
            return Transition::Stay;
        };
        if entry.is_dir {
            self.cwd = entry.path.clone();
            self.reload();
            Transition::Stay
        } else if is_pdf(&entry.path) {
            Transition::OpenConvertFile(entry.path.clone())
        } else {
            Transition::Stay
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .entries
            .iter()
            .map(|entry| {
                let (icon, tone) = if entry.is_parent {
                    ("↩", STEAM)
                } else if entry.is_dir {
                    ("▸", USER)
                } else if is_pdf(&entry.path) {
                    ("◆", GOLD)
                } else {
                    ("·", TAN)
                };
                let size = entry
                    .size
                    .map(|s| format!("  {:>9}", human_bytes(s)))
                    .unwrap_or_default();
                let suffix = if entry.is_dir && !entry.is_parent { "/" } else { "" };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {icon} "), Style::default().fg(tone)),
                    Span::styled(format!("{}{suffix}", entry.name), Style::default().fg(tone)),
                    Span::styled(size, Style::default().fg(DIM)),
                ]))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(
                format!(" {} ", self.cwd.display()),
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));

        let list = List::new(items).block(block).highlight_symbol("▍").highlight_style(
            Style::default().fg(PALM).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
        let mut state = ListState::default();
        if !self.entries.is_empty() {
            state.select(Some(self.selected));
        }
        frame.render_stateful_widget(list, area, &mut state);
    }
}

/// Compact decimal byte formatting for the size column.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
