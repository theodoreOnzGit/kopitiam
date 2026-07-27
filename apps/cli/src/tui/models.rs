//! The **Models** view: an interactive catalog browser that pulls a model with
//! its HuggingFace sha256 **auto-resolved** (#31), mirroring `kopitiam models
//! pull` exactly.
//!
//! # What it does
//!
//! Lists the built-in catalog ([`kopitiam_models::Catalog`]) with each entry's
//! family, license, total size, and whether it is present in the local store.
//! `Enter`/`p` pulls the selected model; `v` verifies a present model's
//! checksums; `r` refreshes presence.
//!
//! # Thin-client discipline
//!
//! No acquisition logic lives here. The pull is armed as a router
//! [`Transition::PullModel`] and carried out on the normal terminal via the
//! parent's suspend→run→restore path (a network download streams live progress,
//! so it must not freeze the TUI) — that runner calls
//! [`kopitiam_models::ensure_available_resolving`], the net-gated resolver that
//! fetches the real sha from the hub before verifying, so nothing here hardcodes
//! a checksum. Verify is fast and offline, so it runs inline.

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use std::io::Write as _;
use std::path::Path;

use kopitiam_models::{
    Architecture, Catalog, DEFAULT_MODEL_ID, Error as ModelsError, Fetcher, HttpFetcher, ModelSpec,
    ModelStore, ensure_available_resolving, hf_token_from_env,
};

use super::theme::{DIM, GOLD, PALM, STEAM, TAN};
use super::Transition;

/// A single catalog row as shown in the list.
#[derive(Clone, PartialEq, Eq, Debug)]
struct ModelRow {
    id: String,
    name: String,
    family: String,
    license: String,
    size: String,
    present: bool,
    is_default: bool,
}

/// Models view state.
pub struct ModelsView {
    rows: Vec<ModelRow>,
    selected: usize,
    notice: Option<String>,
}

impl ModelsView {
    /// Build the view from the built-in catalog and current store presence.
    pub fn new() -> Self {
        let (rows, notice) = load_rows();
        Self { rows, selected: 0, notice }
    }

    // A `Default` that mirrors `new()` so clippy's `new_without_default` is happy
    // even though `new()` does real work (reads the catalog + store presence).

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("↑/↓ j/k", "move"),
            ("Enter/p", "pull (auto-sha)"),
            ("v", "verify"),
            ("r", "refresh"),
            ("Esc", "home"),
        ]
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => Transition::Quit,
            KeyCode::Esc | KeyCode::Char('q') => Transition::Home,
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected = step(self.selected, self.rows.len(), 1);
                Transition::Stay
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = step(self.selected, self.rows.len(), -1);
                Transition::Stay
            }
            KeyCode::Enter | KeyCode::Char('p') => match self.rows.get(self.selected) {
                Some(row) => Transition::PullModel(row.id.clone()),
                None => Transition::Stay,
            },
            KeyCode::Char('v') => {
                self.verify_selected();
                Transition::Stay
            }
            KeyCode::Char('r') => {
                let (rows, notice) = load_rows();
                self.rows = rows;
                self.selected = clamp(self.selected, self.rows.len());
                self.notice = notice.or(Some("refreshed presence".into()));
                Transition::Stay
            }
            _ => Transition::Stay,
        }
    }

    /// Verify the selected model's checksums inline (offline, fast).
    fn verify_selected(&mut self) {
        let Some(row) = self.rows.get(self.selected) else { return };
        self.notice = Some(verify_notice(&row.id));
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let notice_h = if self.notice.is_some() { 3 } else { 0 };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(notice_h)])
            .split(area);

        // Header describing the auto-sha behaviour.
        frame.render_widget(
            Paragraph::new(
                "Pull resolves each artifact's HuggingFace sha256 from the hub automatically \
                 (net-gated), then downloads + verifies. The download streams progress on the \
                 normal terminal and returns here when done.",
            )
            .style(Style::default().fg(STEAM))
            .wrap(Wrap { trim: false }),
            rows[0],
        );

        let items: Vec<ListItem> = self
            .rows
            .iter()
            .map(|r| {
                let present = if r.present {
                    Span::styled("● present", Style::default().fg(PALM))
                } else {
                    Span::styled("○ absent ", Style::default().fg(DIM))
                };
                let mut spans = vec![
                    present,
                    Span::raw("  "),
                    Span::styled(format!("{:<28}", r.id), Style::default().fg(TAN).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("{:<8} ", r.family), Style::default().fg(DIM)),
                    Span::styled(format!("{:>10}  ", r.size), Style::default().fg(DIM)),
                    Span::styled(r.license.clone(), Style::default().fg(DIM)),
                ];
                if r.is_default {
                    spans.push(Span::styled("  (default)", Style::default().fg(GOLD)));
                }
                ListItem::new(Line::from(spans))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(
                " kopitiam · models ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));
        let list = List::new(items).block(block).highlight_symbol("▍ ").highlight_style(
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
        let mut state = ListState::default();
        state.select(Some(self.selected));
        frame.render_stateful_widget(list, rows[1], &mut state);

        if let Some(notice) = &self.notice {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM));
            frame.render_widget(
                Paragraph::new(Text::from(Line::from(Span::styled(
                    notice.clone(),
                    Style::default().fg(STEAM).add_modifier(Modifier::ITALIC),
                ))))
                .block(block)
                .wrap(Wrap { trim: false }),
                rows[2],
            );
        }
    }
}

impl Default for ModelsView {
    fn default() -> Self {
        Self::new()
    }
}

/// Pull a model by id on the normal terminal (called from the router's
/// suspend→run→restore path), with the sha AUTO-RESOLVED from the hub. Mirrors
/// `kopitiam models pull` exactly and returns a one-line notice for the home menu.
pub fn run_pull(id: &str) -> String {
    let Some(spec) = Catalog::find(id) else {
        return format!("unknown model id '{id}'");
    };
    let store = match ModelStore::with_default_root() {
        Ok(store) => store,
        Err(e) => return format!("model store unavailable: {e}"),
    };
    println!("Pulling {} ({}) with sha auto-resolved from HuggingFace...\n", spec.display_name, spec.id);

    // Same reasoning as the CLI's `models pull`: the only way to surface live
    // progress is to hand `ensure_available_resolving` a Fetcher that prints it.
    let fetcher = ProgressFetcher { inner: HttpFetcher };
    let token = hf_token_from_env();
    match ensure_available_resolving(&store, &spec, &fetcher, token.as_deref()) {
        Ok(acquired) => {
            println!("\nVerified. Local artifact path(s):");
            for path in &acquired.artifact_paths {
                println!("  {}", path.display());
            }
            format!("pulled {} — verified", spec.id)
        }
        Err(e) => format!("pull failed: {e}"),
    }
}

/// A [`Fetcher`] that prints one-line download progress before forwarding to the
/// real HTTP fetcher. Presentation only (mirrors the CLI's private wrapper; kept
/// local so the CLI command module is untouched).
struct ProgressFetcher<F> {
    inner: F,
}

impl<F: Fetcher> Fetcher for ProgressFetcher<F> {
    fn fetch(
        &self,
        url: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), ModelsError> {
        let mut printing = |downloaded: u64, total: Option<u64>| {
            match total {
                Some(total) => {
                    print!("\r  {} / {}   ", human_bytes(downloaded), human_bytes(total))
                }
                None => print!("\r  {}   ", human_bytes(downloaded)),
            }
            let _ = std::io::stdout().flush();
            progress(downloaded, total);
        };
        let result = self.inner.fetch(url, dest, &mut printing);
        println!();
        result
    }
}

/// Build the catalog rows and an optional store-error notice.
fn load_rows() -> (Vec<ModelRow>, Option<String>) {
    let catalog = Catalog::builtin();
    let store = ModelStore::with_default_root();
    let notice = store
        .as_ref()
        .err()
        .map(|e| format!("model store unavailable, presence unknown: {e}"));
    let rows = catalog
        .into_iter()
        .map(|spec| ModelRow {
            present: store.as_ref().map(|s| s.is_present(&spec)).unwrap_or(false),
            is_default: spec.id == DEFAULT_MODEL_ID,
            id: spec.id.clone(),
            name: spec.display_name.clone(),
            family: family_label(&spec.architecture),
            license: spec.license.clone(),
            size: human_bytes(total_size(&spec)),
        })
        .collect();
    (rows, notice)
}

/// Verify a model id's checksums and turn the result into a one-line notice.
/// Kept separate so the mapping is unit-tested without a store.
fn verify_notice(id: &str) -> String {
    let Some(spec) = Catalog::find(id) else {
        return format!("unknown model id '{id}'");
    };
    let store = match ModelStore::with_default_root() {
        Ok(store) => store,
        Err(e) => return format!("model store unavailable: {e}"),
    };
    match store.verify(&spec) {
        Ok(()) => format!("OK: {id} checksums match"),
        Err(ModelsError::ChecksumMismatch { artifact, .. }) => {
            format!("checksum mismatch in {artifact} — re-pull {id}")
        }
        Err(other) => format!("verify {id}: {other}"),
    }
}

fn total_size(spec: &ModelSpec) -> u64 {
    spec.artifacts.iter().map(|a| a.size_bytes).sum()
}

fn family_label(arch: &Architecture) -> String {
    match arch {
        Architecture::Qwen2 => "Qwen2".into(),
        Architecture::Llama => "Llama".into(),
        Architecture::Phi3 => "Phi3".into(),
        Architecture::Gemma => "Gemma".into(),
        Architecture::Other(name) => name.clone(),
    }
}

/// Decimal (SI) human byte formatting — presentation only, mirrors the CLI's.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn clamp(sel: usize, len: usize) -> usize {
    if len == 0 { 0 } else { sel.min(len - 1) }
}

fn step(sel: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let max = len - 1;
    (sel as isize + delta).clamp(0, max as isize) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_cover_the_catalog_and_flag_the_default() {
        let (rows, _) = load_rows();
        assert_eq!(rows.len(), Catalog::builtin().len());
        assert!(!rows.is_empty());
        let default = rows.iter().find(|r| r.is_default).expect("a default row");
        assert_eq!(default.id, DEFAULT_MODEL_ID);
        // Every row carries a non-empty size string and family label.
        assert!(rows.iter().all(|r| !r.size.is_empty() && !r.family.is_empty()));
    }

    #[test]
    fn selection_steps_and_saturates() {
        assert_eq!(step(0, 3, 1), 1);
        assert_eq!(step(2, 3, 1), 2); // saturate
        assert_eq!(step(0, 3, -1), 0); // saturate
        assert_eq!(step(0, 0, 1), 0); // empty
        assert_eq!(clamp(9, 3), 2);
        assert_eq!(clamp(9, 0), 0);
    }

    #[test]
    fn enter_arms_pull_for_selected_model() {
        let mut view = ModelsView::new();
        assert!(!view.rows.is_empty());
        let id = view.rows[0].id.clone();
        match view.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty())) {
            Transition::PullModel(got) => assert_eq!(got, id),
            _ => panic!("Enter should arm a PullModel transition"),
        }
    }

    #[test]
    fn verify_notice_handles_unknown_id() {
        assert!(verify_notice("no-such-model").contains("unknown model id"));
    }

    #[test]
    fn human_bytes_is_decimal() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1_500), "1.5 KB");
        assert_eq!(human_bytes(4_000_000_000), "4.0 GB");
    }
}
