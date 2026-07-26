//! The **File Explorer** view: a small, robust directory browser. Enter a
//! directory to descend, `..` to go up, and pressing `Enter` on a `.pdf` opens
//! it straight in the PDF Viewer.

use std::path::{Path, PathBuf};

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph},
};
use walkdir::WalkDir;

use super::Transition;
use super::logic::{fuzzy_rank, is_pdf};
use super::theme::{DIM, GOLD, PALM, STEAM, TAN, USER};

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

/// File Explorer state: the current directory and its (sorted) entries, plus a
/// live fuzzy filter (optionally recursive across the subtree).
pub struct ExplorerState {
    cwd: PathBuf,
    /// The current directory's own entries (with the `..` row).
    entries: Vec<Entry>,
    /// The recursive listing of the whole subtree, built lazily when the user
    /// toggles recursive find. Its rows carry a path relative to `cwd`.
    find_entries: Vec<Entry>,
    /// Position within `ranked` of the highlighted row.
    selected: usize,
    /// Live fuzzy query.
    query: String,
    /// True while the query is being typed (keys feed the box, not navigation).
    filtering: bool,
    /// Whether the candidate pool is the recursive subtree (`find_entries`)
    /// rather than just the current directory (`entries`).
    recursive: bool,
    /// Indices into the active candidate list, ranked best-first for `query`.
    ranked: Vec<usize>,
}

impl ExplorerState {
    /// Open the explorer rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        let mut state = Self {
            cwd,
            entries: Vec::new(),
            find_entries: Vec::new(),
            selected: 0,
            query: String::new(),
            filtering: false,
            recursive: false,
            ranked: Vec::new(),
        };
        state.reload();
        state
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        if self.filtering {
            vec![
                ("type", "filter"),
                ("↑/↓", "move"),
                ("Tab", if self.recursive { "this dir" } else { "recurse" }),
                ("Enter", "open"),
                ("Esc", "clear"),
            ]
        } else {
            vec![
                ("↑/↓ j/k", "move"),
                ("/", "filter"),
                ("Enter", "open/view"),
                ("c", "convert pdf"),
                ("Esc", "home"),
            ]
        }
    }

    /// The active candidate list: the recursive subtree when recursive-find is on,
    /// else the current directory.
    fn active(&self) -> &[Entry] {
        if self.recursive { &self.find_entries } else { &self.entries }
    }

    /// The entry currently highlighted (resolved through `ranked`), if any.
    fn selected_entry(&self) -> Option<&Entry> {
        let idx = *self.ranked.get(self.selected)?;
        self.active().get(idx)
    }

    /// Recompute the ranked candidate list for the current query, clamping the
    /// selection. An empty query keeps the natural order (fuzzy_rank's identity
    /// case), so this drives both the unfiltered and filtered listings.
    fn refilter(&mut self) {
        let names: Vec<String> = self.active().iter().map(|e| e.name.clone()).collect();
        self.ranked = fuzzy_rank(&self.query, &names);
        if self.selected >= self.ranked.len() {
            self.selected = self.ranked.len().saturating_sub(1);
        }
    }

    /// Build the recursive subtree listing (files and directories under `cwd`),
    /// each row named by its path relative to `cwd`. Pure-Rust `walkdir`, so it
    /// stays Android/Termux-safe; unreadable entries are silently skipped.
    fn build_recursive(&mut self) {
        let mut rows: Vec<Entry> = WalkDir::new(&self.cwd)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path() != self.cwd)
            .map(|e| {
                let path = e.path().to_path_buf();
                let is_dir = e.file_type().is_dir();
                let size = if is_dir { None } else { e.metadata().ok().map(|m| m.len()) };
                let name = path.strip_prefix(&self.cwd).unwrap_or(&path).display().to_string();
                Entry { name, path, is_dir, size, is_parent: false }
            })
            .collect();
        rows.sort_by_key(|entry| entry.name.to_lowercase());
        self.find_entries = rows;
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
        // Descending resets any active filter and recursive mode.
        self.query.clear();
        self.filtering = false;
        self.recursive = false;
        self.refilter();
    }

    fn move_by(&mut self, delta: i32) {
        if self.ranked.is_empty() {
            return;
        }
        let len = self.ranked.len() as i32;
        self.selected = (self.selected as i32 + delta).rem_euclid(len) as usize;
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Char('c') && ctrl {
            return Transition::Quit;
        }
        if self.filtering {
            return self.on_key_filtering(key, ctrl);
        }
        match key.code {
            KeyCode::Esc => Transition::Home,
            KeyCode::Char('/') => {
                self.filtering = true;
                self.query.clear();
                self.refilter();
                Transition::Stay
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_by(-1);
                Transition::Stay
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_by(1);
                Transition::Stay
            }
            KeyCode::Enter => self.activate(),
            // `c` on a highlighted PDF sends it to the converter instead of the
            // viewer (the viewer is the default on `Enter`).
            KeyCode::Char('c') => match self.selected_entry() {
                Some(entry) if !entry.is_dir && is_pdf(&entry.path) => {
                    Transition::OpenConvertFile(entry.path.clone())
                }
                _ => Transition::Stay,
            },
            _ => Transition::Stay,
        }
    }

    /// Handle a key while the fuzzy filter box is active. Typing narrows the list
    /// live; arrows move; `Tab` toggles recursive-subtree find; `Enter` opens the
    /// highlighted match; `Esc` clears the filter and returns to browsing.
    fn on_key_filtering(&mut self, key: KeyEvent, ctrl: bool) -> Transition {
        match key.code {
            KeyCode::Esc => {
                self.filtering = false;
                self.query.clear();
                self.recursive = false;
                self.selected = 0;
                self.refilter();
            }
            KeyCode::Enter => return self.activate(),
            KeyCode::Tab => {
                self.recursive = !self.recursive;
                if self.recursive && self.find_entries.is_empty() {
                    self.build_recursive();
                }
                self.selected = 0;
                self.refilter();
            }
            KeyCode::Up => self.move_by(-1),
            KeyCode::Down => self.move_by(1),
            KeyCode::Char('p') if ctrl => self.move_by(-1),
            KeyCode::Char('n') if ctrl => self.move_by(1),
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
            }
            KeyCode::Char(c) if !ctrl => {
                self.query.push(c);
                self.refilter();
            }
            _ => {}
        }
        Transition::Stay
    }

    /// Act on the highlighted row: descend into a directory, ascend on `..`, or
    /// open a PDF into the viewer.
    fn activate(&mut self) -> Transition {
        let Some(entry) = self.selected_entry() else {
            return Transition::Stay;
        };
        if entry.is_dir {
            self.cwd = entry.path.clone();
            self.reload();
            Transition::Stay
        } else if is_pdf(&entry.path) {
            Transition::OpenViewFile(entry.path.clone())
        } else {
            Transition::Stay
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let show_filter = self.filtering || !self.query.is_empty();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(if show_filter { 1 } else { 0 })])
            .split(area);

        let active = self.active();
        let items: Vec<ListItem> = self
            .ranked
            .iter()
            .filter_map(|&idx| active.get(idx))
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

        let scope = if self.recursive { " (recursive)" } else { "" };
        let title = format!(" {}{scope}  ·  {} shown ", self.cwd.display(), self.ranked.len());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(title, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));

        let list = List::new(items).block(block).highlight_symbol("▍").highlight_style(
            Style::default().fg(PALM).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
        let mut state = ListState::default();
        if !self.ranked.is_empty() {
            state.select(Some(self.selected.min(self.ranked.len().saturating_sub(1))));
        }
        frame.render_stateful_widget(list, chunks[0], &mut state);

        if show_filter {
            let cursor = if self.filtering { "▌" } else { "" };
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(format!("/{}", self.query), Style::default().fg(GOLD)),
                    Span::styled(cursor.to_string(), Style::default().fg(GOLD)),
                ])),
                chunks[1],
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn typed(state: &mut ExplorerState, text: &str) {
        for c in text.chars() {
            state.on_key(key(KeyCode::Char(c)));
        }
    }

    /// The relative names of the rows currently shown (ranked over the active
    /// list).
    fn visible_names(state: &ExplorerState) -> Vec<String> {
        let active = state.active();
        state.ranked.iter().filter_map(|&i| active.get(i)).map(|e| e.name.clone()).collect()
    }

    fn fixture() -> (tempfile::TempDir, ExplorerState) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("deep")).unwrap();
        std::fs::write(root.join("annual_report.pdf"), b"x").unwrap();
        std::fs::write(root.join("budget.txt"), b"x").unwrap();
        std::fs::write(root.join("invoice.pdf"), b"x").unwrap();
        std::fs::write(root.join("deep/deep_report.pdf"), b"x").unwrap();
        let state = ExplorerState::new(root.to_path_buf());
        (dir, state)
    }

    #[test]
    fn slash_starts_fuzzy_filter_and_picks_expected_entries() {
        let (_dir, mut state) = fixture();
        // Without a filter every row (plus `..`) is visible.
        assert!(visible_names(&state).iter().any(|n| n == "annual_report.pdf"));

        state.on_key(key(KeyCode::Char('/')));
        assert!(state.filtering);
        typed(&mut state, "report");

        let shown = visible_names(&state);
        // The report file matches; unrelated names are dropped.
        assert!(shown.iter().any(|n| n == "annual_report.pdf"), "shown = {shown:?}");
        assert!(!shown.iter().any(|n| n == "budget.txt"), "shown = {shown:?}");
        assert!(!shown.iter().any(|n| n == "invoice.pdf"), "shown = {shown:?}");
    }

    #[test]
    fn non_matching_filter_drops_everything() {
        let (_dir, mut state) = fixture();
        state.on_key(key(KeyCode::Char('/')));
        typed(&mut state, "zzzzz");
        assert!(state.ranked.is_empty());
    }

    #[test]
    fn esc_clears_the_filter_and_restores_browsing() {
        let (_dir, mut state) = fixture();
        state.on_key(key(KeyCode::Char('/')));
        typed(&mut state, "report");
        state.on_key(key(KeyCode::Esc));
        assert!(!state.filtering);
        assert!(state.query.is_empty());
        assert!(!state.recursive);
        // Browsing is restored: the full listing is visible again.
        assert!(visible_names(&state).iter().any(|n| n == "invoice.pdf"));
    }

    #[test]
    fn tab_toggles_recursive_find_across_the_subtree() {
        let (_dir, mut state) = fixture();
        state.on_key(key(KeyCode::Char('/')));
        // The nested file is not in the current-directory listing...
        typed(&mut state, "deep_report");
        assert!(visible_names(&state).is_empty());
        // ...but recursive find reaches it.
        state.on_key(key(KeyCode::Tab));
        assert!(state.recursive);
        let shown = visible_names(&state);
        assert!(shown.iter().any(|n| n.ends_with("deep_report.pdf")), "shown = {shown:?}");
    }

    #[test]
    fn enter_on_a_filtered_pdf_opens_the_viewer() {
        let (_dir, mut state) = fixture();
        state.on_key(key(KeyCode::Char('/')));
        typed(&mut state, "annual");
        // The single match is highlighted; Enter opens it in the viewer.
        let transition = state.on_key(key(KeyCode::Enter));
        assert!(matches!(transition, Transition::OpenViewFile(_)));
    }
}
