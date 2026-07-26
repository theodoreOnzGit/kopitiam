//! The **Convert PDF** (single, fuzzy-found) and **Convert Folder** (recursive
//! batch) views, plus the thin pipeline wrapper both drive.
//!
//! The pipeline itself is owned by the engine crates, exactly as
//! [`crate::pdf2md`] does it on the command line: `kopitiam-pdf` extracts,
//! `kopitiam-document` reconstructs and validates, `kopitiam-markdown` renders.
//! [`convert_one`] is the one place the UI touches them, returning a small
//! plain-data [`ConvertOutcome`] (or an error string) so the views stay free of
//! engine types and the batch loop can render progress from it.

use std::path::{Path, PathBuf};

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use super::logic::{
    default_batch_output, derive_md_name, discover_dirs, discover_pdfs, fuzzy_rank,
    mirror_output_path,
};
use super::theme::{CHILLI, DIM, GOLD, PALM, STEAM, TAN, USER};
use super::Transition;

// ---- pipeline wrapper ---------------------------------------------------

/// The plain-data result of one successful PDF→Markdown conversion, surfaced to
/// the UI. Everything here is derived from the engine's `ConversionReport`, so
/// the views never depend on `kopitiam-document` types directly.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOutcome {
    /// Where the `.md` was written.
    pub output: PathBuf,
    /// Number of pages that made it into the document.
    pub pages: usize,
    /// The character-based recovery ratio (0.0–1.0); the engine's authoritative
    /// PASS/FAIL signal.
    pub recovery_ratio: f64,
    /// Whether the report passed its recovery threshold.
    pub passed: bool,
    /// How many pages the extractor skipped (unreadable fonts, etc.) but still
    /// succeeded around. Inferred from a gap between the highest page number
    /// seen and the number of pages recovered.
    pub skipped_pages: usize,
}

/// Run the full pdf2md pipeline for one file and write the Markdown.
///
/// Mirrors [`crate::pdf2md`] but returns structured data instead of printing,
/// and reports how many pages the extractor skipped so the UI can surface a
/// partial conversion honestly. Any failure comes back as a human-readable
/// `Err(String)` so the batch loop can record it and carry on.
///
/// # OCR fallback (`kopitiam_token_max.md` §2.2)
///
/// The automatic per-page OCR fallback (Task #10) is a pipeline-shape step, so it
/// is mirrored here from [`crate::pdf2md`]. The TUI has no flags, so it runs with
/// the defaults — `OcrMode::Auto` over the default language set — meaning a
/// scanned page gets recognized rather than converting to empty Markdown. It is a
/// no-op on born-digital pages (never loads a model, never rasterizes), so a
/// text PDF converts exactly as before.
pub fn convert_one(input: &Path, output: &Path) -> Result<ConvertOutcome, String> {
    let mut pages = match kopitiam_pdf::extract(input) {
        Ok(pages) => pages,
        Err(err @ kopitiam_pdf::ExtractError::UnsupportedFont(_)) => {
            return Err(format!(
                "could not extract text ({err}) — the PDF uses fonts with no unicode mapping"
            ));
        }
        Err(err) => return Err(format!("could not extract text: {err}")),
    };

    // The extractor skips a page (rather than aborting) when its fonts cannot be
    // decoded, so the number recovered can be fewer than the PDF's page count.
    // `Page::number` is the original 1-based page number, so the highest number
    // seen minus how many we actually have is how many were dropped.
    let highest = pages.iter().map(|p| p.number).max().unwrap_or(0);
    let skipped_pages = highest.saturating_sub(pages.len());

    // Automatic OCR fallback on the scanned (low-text) pages, with the TUI's
    // defaults. A missing model surfaces as a clear `Err(String)` (naming
    // `kopitiam models pull …`) the batch loop records like any other failure.
    let ocr_langs = crate::ocr_fallback::parse_langs(crate::ocr_fallback::DEFAULT_OCR_LANGS);
    crate::ocr_fallback::run_ocr_fallback(
        input,
        &mut pages,
        crate::ocr_fallback::OcrMode::Auto,
        &ocr_langs,
    )
    .map_err(|e| format!("OCR fallback failed: {e}"))?;

    let document = kopitiam_document::reconstruct(&pages);
    let markdown = kopitiam_markdown::render_document(&document);
    let report = kopitiam_document::validate(&pages, &document, &markdown);

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    std::fs::write(output, &markdown).map_err(|e| format!("could not write {}: {e}", output.display()))?;

    Ok(ConvertOutcome {
        output: output.to_path_buf(),
        pages: report.pages,
        recovery_ratio: report.recovery_ratio(),
        passed: report.passes(),
        skipped_pages,
    })
}

// ---- Convert PDF (single) ----------------------------------------------

#[derive(PartialEq)]
enum Focus {
    Query,
    Root,
}

enum Stage {
    /// The fuzzy finder (root + search inputs, candidate list).
    Pick,
    /// The output-name prompt.
    Name,
    /// The result pane.
    Result,
}

/// State for the single-file Convert PDF flow.
pub struct ConvertPdfState {
    root: PathBuf,
    root_input: String,
    /// Absolute paths of every discovered PDF.
    candidates: Vec<PathBuf>,
    /// Display strings (relative to `root`), parallel to `candidates`.
    display: Vec<String>,
    query: String,
    /// Indices into `candidates`, ranked best-first for the current query.
    ranked: Vec<usize>,
    /// Position within `ranked` of the highlighted row.
    selected: usize,
    focus: Focus,
    stage: Stage,
    chosen: Option<PathBuf>,
    name_input: String,
    result: Option<ResultPane>,
}

/// A rendered result pane: pre-built lines plus whether it represents success.
struct ResultPane {
    ok: bool,
    lines: Vec<String>,
}

impl ConvertPdfState {
    /// Open the finder rooted at `root`, discovering PDFs immediately.
    pub fn new(root: PathBuf) -> Self {
        let mut state = Self {
            root_input: root.display().to_string(),
            root,
            candidates: Vec::new(),
            display: Vec::new(),
            query: String::new(),
            ranked: Vec::new(),
            selected: 0,
            focus: Focus::Query,
            stage: Stage::Pick,
            chosen: None,
            name_input: String::new(),
            result: None,
        };
        state.rescan();
        state
    }

    /// Open the flow pre-selected on `pdf` (from the File Explorer): skip the
    /// finder and land straight on the output-name prompt.
    pub fn with_file(pdf: PathBuf) -> Self {
        let root = pdf.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let mut state = Self::new(root);
        state.name_input = derive_md_name(&pdf);
        state.chosen = Some(pdf);
        state.stage = Stage::Name;
        state
    }

    /// Re-walk `root` for PDFs and rebuild the display + ranking.
    fn rescan(&mut self) {
        self.candidates = discover_pdfs(&self.root);
        self.display = self
            .candidates
            .iter()
            .map(|p| {
                p.strip_prefix(&self.root)
                    .unwrap_or(p)
                    .display()
                    .to_string()
            })
            .collect();
        self.refilter();
    }

    /// Recompute the ranked list for the current query, clamping the selection.
    fn refilter(&mut self) {
        self.ranked = fuzzy_rank(&self.query, &self.display);
        if self.selected >= self.ranked.len() {
            self.selected = self.ranked.len().saturating_sub(1);
        }
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        match self.stage {
            Stage::Pick => vec![
                ("type", "filter"),
                ("↑/↓", "move"),
                ("Tab", "root/search"),
                ("Enter", "pick"),
                ("Esc", "home"),
            ],
            Stage::Name => vec![("type", "edit name"), ("Enter", "convert"), ("Esc", "back")],
            Stage::Result => vec![("Enter/Esc", "home")],
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Char('c') && ctrl {
            return Transition::Quit;
        }
        match self.stage {
            Stage::Pick => self.on_key_pick(key, ctrl),
            Stage::Name => self.on_key_name(key),
            Stage::Result => match key.code {
                KeyCode::Esc | KeyCode::Enter => Transition::Home,
                _ => Transition::Stay,
            },
        }
    }

    fn on_key_pick(&mut self, key: KeyEvent, ctrl: bool) -> Transition {
        match key.code {
            KeyCode::Esc => return Transition::Home,
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Query => Focus::Root,
                    Focus::Root => Focus::Query,
                };
            }
            KeyCode::Up => self.move_sel(-1),
            KeyCode::Down => self.move_sel(1),
            KeyCode::Char('p') if ctrl => self.move_sel(-1),
            KeyCode::Char('n') if ctrl => self.move_sel(1),
            KeyCode::Enter => match self.focus {
                Focus::Root => {
                    self.root = PathBuf::from(self.root_input.trim());
                    self.selected = 0;
                    self.rescan();
                    self.focus = Focus::Query;
                }
                Focus::Query => {
                    if let Some(&cand_idx) = self.ranked.get(self.selected) {
                        let pdf = self.candidates[cand_idx].clone();
                        self.name_input = derive_md_name(&pdf);
                        self.chosen = Some(pdf);
                        self.stage = Stage::Name;
                    }
                }
            },
            KeyCode::Backspace => {
                match self.focus {
                    Focus::Query => {
                        self.query.pop();
                        self.refilter();
                    }
                    Focus::Root => {
                        self.root_input.pop();
                    }
                }
            }
            KeyCode::Char(c) if !ctrl => match self.focus {
                Focus::Query => {
                    self.query.push(c);
                    self.refilter();
                }
                Focus::Root => self.root_input.push(c),
            },
            _ => {}
        }
        Transition::Stay
    }

    fn move_sel(&mut self, delta: i32) {
        if self.ranked.is_empty() {
            return;
        }
        let len = self.ranked.len() as i32;
        let next = (self.selected as i32 + delta).rem_euclid(len);
        self.selected = next as usize;
    }

    fn on_key_name(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => self.stage = Stage::Pick,
            KeyCode::Enter => self.run_conversion(),
            KeyCode::Backspace => {
                self.name_input.pop();
            }
            KeyCode::Char(c) => self.name_input.push(c),
            _ => {}
        }
        Transition::Stay
    }

    /// Run the pipeline for the chosen file and build the result pane.
    fn run_conversion(&mut self) {
        let Some(pdf) = self.chosen.clone() else {
            return;
        };
        let name = self.name_input.trim();
        let name = if name.is_empty() { derive_md_name(&pdf) } else { name.to_string() };
        // Write next to the source PDF (a bare name), or honour a path the user
        // typed (contains a separator / is absolute).
        let output = if Path::new(&name).parent().map(|p| !p.as_os_str().is_empty()).unwrap_or(false) {
            PathBuf::from(&name)
        } else {
            pdf.parent().map(|p| p.join(&name)).unwrap_or_else(|| PathBuf::from(&name))
        };

        self.result = Some(match convert_one(&pdf, &output) {
            Ok(outcome) => ResultPane { ok: true, lines: success_lines(&outcome) },
            Err(err) => ResultPane {
                ok: false,
                lines: vec![
                    format!("Failed to convert {}", pdf.display()),
                    String::new(),
                    err,
                ],
            },
        });
        self.stage = Stage::Result;
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        match self.stage {
            Stage::Pick => self.render_pick(frame, area),
            Stage::Name => self.render_name(frame, area),
            Stage::Result => self.render_result(frame, area),
        }
    }

    fn render_pick(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Length(3), Constraint::Min(3)])
            .split(area);

        frame.render_widget(input_box(" search root ", &self.root_input, self.focus == Focus::Root), chunks[0]);
        frame.render_widget(input_box(" fuzzy search ", &self.query, self.focus == Focus::Query), chunks[1]);

        let items: Vec<ListItem> = self
            .ranked
            .iter()
            .map(|&idx| ListItem::new(Line::from(Span::styled(self.display[idx].clone(), Style::default().fg(TAN)))))
            .collect();

        let title = format!(" {} PDF(s) · {} match ", self.candidates.len(), self.ranked.len());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(title, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));

        if self.ranked.is_empty() {
            let msg = if self.candidates.is_empty() {
                "No PDFs found under this root. Tab to the search-root box to change it."
            } else {
                "No matches for this query."
            };
            frame.render_widget(Paragraph::new(msg).style(Style::default().fg(STEAM)).block(block), chunks[2]);
        } else {
            let list = List::new(items).block(block).highlight_symbol("▍ ").highlight_style(
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
            );
            let mut state = ListState::default();
            state.select(Some(self.selected.min(self.ranked.len().saturating_sub(1))));
            frame.render_stateful_widget(list, chunks[2], &mut state);
        }
    }

    fn render_name(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        let chosen = self
            .chosen
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("Converting: ", Style::default().fg(STEAM)),
                Span::styled(chosen, Style::default().fg(USER)),
            ])),
            chunks[0],
        );
        frame.render_widget(input_box(" output .md name ", &self.name_input, true), chunks[1]);
    }

    fn render_result(&self, frame: &mut Frame, area: Rect) {
        let pane = self.result.as_ref();
        let ok = pane.map(|p| p.ok).unwrap_or(false);
        let (border, title) = if ok {
            (PALM, " converted ✓ ")
        } else {
            (CHILLI, " failed ✗ ")
        };
        let lines: Vec<Line> = pane
            .map(|p| {
                p.lines
                    .iter()
                    .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(TAN))))
                    .collect()
            })
            .unwrap_or_default();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border))
            .title(Span::styled(title, Style::default().fg(border).add_modifier(Modifier::BOLD)));
        frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }).block(block), area);
    }
}

/// Build the human-readable success lines for a completed conversion.
fn success_lines(outcome: &ConvertOutcome) -> Vec<String> {
    let mut lines = vec![
        format!("Wrote {}", outcome.output.display()),
        String::new(),
        format!("Pages recovered: {}", outcome.pages),
        format!(
            "Recovery ratio:  {:.1}%  ({})",
            outcome.recovery_ratio * 100.0,
            if outcome.passed { "PASS" } else { "FAIL — content may be missing" }
        ),
    ];
    if outcome.skipped_pages > 0 {
        lines.push(String::new());
        lines.push(format!(
            "Note: {} page(s) were skipped (fonts with no usable unicode mapping); \
             conversion continued with the rest.",
            outcome.skipped_pages
        ));
    }
    lines
}

// ---- Convert Folder (recursive batch) ----------------------------------

enum FolderStage {
    Input,
    Output,
    Running,
}

/// Per-file status inside a batch.
#[derive(Clone)]
enum BatchStatus {
    Pending,
    Done { ratio: f64, skipped: usize },
    Failed(String),
}

/// One file in the batch queue.
struct BatchItem {
    rel: String,
    input: PathBuf,
    output: PathBuf,
    status: BatchStatus,
}

/// State for the recursive Convert Folder flow.
pub struct ConvertFolderState {
    stage: FolderStage,
    // ---- input-folder fuzzy picker (FolderStage::Input) ----
    /// Where directory discovery is rooted.
    search_root: PathBuf,
    /// The editable search-root text (also the typed-path fallback).
    root_input: String,
    /// Every directory discovered under `search_root` (root included first).
    dir_candidates: Vec<PathBuf>,
    /// Display strings (relative to `search_root`), parallel to `dir_candidates`.
    dir_display: Vec<String>,
    query: String,
    /// Indices into `dir_candidates`, ranked best-first for the current query.
    ranked: Vec<usize>,
    /// Position within `ranked` of the highlighted row.
    selected: usize,
    focus: Focus,
    // ---- output + run ----
    output_text: String,
    input_root: PathBuf,
    output_root: PathBuf,
    items: Vec<BatchItem>,
    /// Index of the next item to convert.
    cursor: usize,
    scroll: u16,
}

impl ConvertFolderState {
    /// Open the batch flow with the fuzzy input-folder picker rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        let mut state = Self {
            stage: FolderStage::Input,
            root_input: cwd.display().to_string(),
            search_root: cwd.clone(),
            dir_candidates: Vec::new(),
            dir_display: Vec::new(),
            query: String::new(),
            ranked: Vec::new(),
            selected: 0,
            focus: Focus::Query,
            output_text: String::new(),
            // Prefilled to `cwd` so a direct `start()` (used in tests / headless
            // flows) has a sensible input root even without going through the
            // picker.
            input_root: cwd,
            output_root: PathBuf::new(),
            items: Vec::new(),
            cursor: 0,
            scroll: 0,
        };
        state.rescan_dirs();
        state
    }

    /// Re-walk `search_root` for directories and rebuild the display + ranking.
    fn rescan_dirs(&mut self) {
        self.dir_candidates = discover_dirs(&self.search_root);
        self.dir_display = self
            .dir_candidates
            .iter()
            .map(|p| {
                let rel = p.strip_prefix(&self.search_root).unwrap_or(p).display().to_string();
                // The root itself strips to an empty string; show it as ".".
                if rel.is_empty() { ".".to_string() } else { rel }
            })
            .collect();
        self.refilter_dirs();
    }

    /// Recompute the ranked directory list for the current query.
    fn refilter_dirs(&mut self) {
        self.ranked = fuzzy_rank(&self.query, &self.dir_display);
        if self.selected >= self.ranked.len() {
            self.selected = self.ranked.len().saturating_sub(1);
        }
    }

    fn move_sel(&mut self, delta: i32) {
        if self.ranked.is_empty() {
            return;
        }
        let len = self.ranked.len() as i32;
        self.selected = (self.selected as i32 + delta).rem_euclid(len) as usize;
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        match self.stage {
            FolderStage::Input => vec![
                ("type", "filter"),
                ("↑/↓", "move"),
                ("Tab", "root/search"),
                ("Enter", "pick folder"),
                ("Esc", "home"),
            ],
            FolderStage::Output => vec![("type", "output folder"), ("Enter", "start"), ("Esc", "back")],
            FolderStage::Running => {
                if self.is_done() {
                    vec![("↑/↓ j/k", "scroll"), ("Esc/Enter", "home")]
                } else {
                    vec![("↑/↓ j/k", "scroll"), ("Esc", "home (stop)")]
                }
            }
        }
    }

    /// True once every queued item has been processed.
    pub fn is_done(&self) -> bool {
        matches!(self.stage, FolderStage::Running) && self.cursor >= self.items.len()
    }

    /// True while there is still a pending item to convert.
    pub fn needs_step(&self) -> bool {
        matches!(self.stage, FolderStage::Running) && self.cursor < self.items.len()
    }

    /// Convert exactly one pending file and advance the cursor. Called by the
    /// parent loop once per frame so progress renders live without threads.
    /// A single failure is recorded and skipped — the batch never aborts.
    pub fn step(&mut self) {
        if self.cursor >= self.items.len() {
            return;
        }
        let idx = self.cursor;
        let (input, output) = {
            let item = &self.items[idx];
            (item.input.clone(), item.output.clone())
        };
        let status = match convert_one(&input, &output) {
            Ok(outcome) => BatchStatus::Done { ratio: outcome.recovery_ratio, skipped: outcome.skipped_pages },
            Err(err) => BatchStatus::Failed(err),
        };
        self.items[idx].status = status;
        self.cursor += 1;
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Char('c') && ctrl {
            return Transition::Quit;
        }
        match self.stage {
            FolderStage::Input => return self.on_key_input(key, ctrl),
            FolderStage::Output => match key.code {
                KeyCode::Esc => self.stage = FolderStage::Input,
                KeyCode::Enter => self.start(),
                KeyCode::Backspace => {
                    self.output_text.pop();
                }
                KeyCode::Char(c) => self.output_text.push(c),
                _ => {}
            },
            FolderStage::Running => match key.code {
                KeyCode::Esc | KeyCode::Enter => return Transition::Home,
                KeyCode::Up | KeyCode::Char('k') => self.scroll = self.scroll.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => self.scroll = self.scroll.saturating_add(1),
                _ => {}
            },
        }
        Transition::Stay
    }

    /// Handle a key on the fuzzy input-folder picker. Mirrors the single-PDF
    /// picker: type to fuzzy-filter, ↑/↓ (or Ctrl-p/n) to move, Tab to swap
    /// between the search-root box and the query, Enter to pick.
    fn on_key_input(&mut self, key: KeyEvent, ctrl: bool) -> Transition {
        match key.code {
            KeyCode::Esc => return Transition::Home,
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Query => Focus::Root,
                    Focus::Root => Focus::Query,
                };
            }
            KeyCode::Up => self.move_sel(-1),
            KeyCode::Down => self.move_sel(1),
            KeyCode::Char('p') if ctrl => self.move_sel(-1),
            KeyCode::Char('n') if ctrl => self.move_sel(1),
            KeyCode::Enter => match self.focus {
                Focus::Root => {
                    // Re-root discovery at the typed path (the typed-path fallback).
                    self.search_root = PathBuf::from(self.root_input.trim());
                    self.selected = 0;
                    self.rescan_dirs();
                    self.focus = Focus::Query;
                }
                Focus::Query => {
                    if let Some(&cand_idx) = self.ranked.get(self.selected) {
                        self.select_input_dir(self.dir_candidates[cand_idx].clone());
                    }
                }
            },
            KeyCode::Backspace => match self.focus {
                Focus::Query => {
                    self.query.pop();
                    self.refilter_dirs();
                }
                Focus::Root => {
                    self.root_input.pop();
                }
            },
            KeyCode::Char(c) if !ctrl => match self.focus {
                Focus::Query => {
                    self.query.push(c);
                    self.refilter_dirs();
                }
                Focus::Root => self.root_input.push(c),
            },
            _ => {}
        }
        Transition::Stay
    }

    /// Commit `dir` as the input root, prefill the default output, and advance to
    /// the output-folder stage.
    fn select_input_dir(&mut self, dir: PathBuf) {
        self.input_root = dir;
        self.output_text = default_batch_output(&self.input_root).display().to_string();
        self.stage = FolderStage::Output;
    }

    /// Discover the PDFs, build the mirrored output queue, and enter Running.
    fn start(&mut self) {
        self.output_root = PathBuf::from(self.output_text.trim());
        let pdfs = discover_pdfs(&self.input_root);
        self.items = pdfs
            .into_iter()
            .map(|pdf| {
                let output = mirror_output_path(&self.input_root, &self.output_root, &pdf);
                let rel = pdf
                    .strip_prefix(&self.input_root)
                    .unwrap_or(&pdf)
                    .display()
                    .to_string();
                BatchItem { rel, input: pdf, output, status: BatchStatus::Pending }
            })
            .collect();
        self.cursor = 0;
        self.stage = FolderStage::Running;
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        match self.stage {
            FolderStage::Input => self.render_input_picker(frame, area),
            FolderStage::Output => {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(2), Constraint::Length(3), Constraint::Min(0)])
                    .split(area);
                frame.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled("Input: ", Style::default().fg(STEAM)),
                        Span::styled(self.input_root.display().to_string(), Style::default().fg(USER)),
                    ])),
                    chunks[0],
                );
                frame.render_widget(input_box(" output folder ", &self.output_text, true), chunks[1]);
            }
            FolderStage::Running => self.render_running(frame, area),
        }
    }

    /// Render the fuzzy input-folder picker (the same shape as the single-PDF
    /// picker): a search-root box, a fuzzy-query box, and the ranked directory
    /// list.
    fn render_input_picker(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Length(3), Constraint::Min(3)])
            .split(area);

        frame.render_widget(input_box(" search root ", &self.root_input, self.focus == Focus::Root), chunks[0]);
        frame.render_widget(input_box(" fuzzy folder ", &self.query, self.focus == Focus::Query), chunks[1]);

        let items: Vec<ListItem> = self
            .ranked
            .iter()
            .map(|&idx| ListItem::new(Line::from(Span::styled(self.dir_display[idx].clone(), Style::default().fg(TAN)))))
            .collect();

        let title = format!(" {} folder(s) · {} match ", self.dir_candidates.len(), self.ranked.len());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(title, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));

        if self.ranked.is_empty() {
            let msg = if self.dir_candidates.is_empty() {
                "No folders found under this root. Tab to the search-root box to change it."
            } else {
                "No matches for this query."
            };
            frame.render_widget(Paragraph::new(msg).style(Style::default().fg(STEAM)).block(block), chunks[2]);
        } else {
            let list = List::new(items).block(block).highlight_symbol("▍ ").highlight_style(
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
            );
            let mut state = ListState::default();
            state.select(Some(self.selected.min(self.ranked.len().saturating_sub(1))));
            frame.render_stateful_widget(list, chunks[2], &mut state);
        }
    }

    fn render_running(&self, frame: &mut Frame, area: Rect) {
        let done = self.items.iter().filter(|i| matches!(i.status, BatchStatus::Done { .. })).count();
        let failed = self.items.iter().filter(|i| matches!(i.status, BatchStatus::Failed(_))).count();
        let total = self.items.len();

        let title = if self.is_done() {
            format!(" done · {done} converted, {failed} failed of {total} ")
        } else {
            format!(" converting {}/{total} · {done} ✓  {failed} ✗ ", (self.cursor + 1).min(total.max(1)))
        };

        let lines: Vec<Line> = self
            .items
            .iter()
            .map(|item| {
                let (mark, tone, detail) = match &item.status {
                    BatchStatus::Pending => ("·", DIM, String::new()),
                    BatchStatus::Done { ratio, skipped } => {
                        let extra = if *skipped > 0 { format!("  ({skipped} skipped)") } else { String::new() };
                        ("✓", PALM, format!("  {:.0}%{extra}", ratio * 100.0))
                    }
                    BatchStatus::Failed(reason) => ("✗", CHILLI, format!("  {reason}")),
                };
                Line::from(vec![
                    Span::styled(format!(" {mark} "), Style::default().fg(tone).add_modifier(Modifier::BOLD)),
                    Span::styled(item.rel.clone(), Style::default().fg(TAN)),
                    Span::styled(detail, Style::default().fg(tone)),
                ])
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(title, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));

        let body = if self.items.is_empty() {
            Paragraph::new("No PDFs found under the input folder.").style(Style::default().fg(STEAM)).block(block)
        } else {
            Paragraph::new(Text::from(lines)).scroll((self.scroll, 0)).block(block)
        };
        frame.render_widget(body, area);
    }
}

/// A single-line input box, its border brightened when focused.
fn input_box(title: &str, value: &str, focused: bool) -> Paragraph<'static> {
    let tone = if focused { GOLD } else { DIM };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(tone))
        .title(Span::styled(title.to_string(), Style::default().fg(tone).add_modifier(Modifier::BOLD)));
    let mut spans = vec![Span::raw(value.to_string())];
    if focused {
        spans.push(Span::styled("▌", Style::default().fg(GOLD)));
    }
    Paragraph::new(Line::from(spans)).block(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_file_lands_on_name_stage_prefilled() {
        let state = ConvertPdfState::with_file(PathBuf::from("/papers/deep/study.pdf"));
        assert!(matches!(state.stage, Stage::Name));
        assert_eq!(state.name_input, "study.md");
        assert_eq!(state.chosen.as_deref(), Some(Path::new("/papers/deep/study.pdf")));
    }

    #[test]
    fn success_lines_note_skipped_pages() {
        let outcome = ConvertOutcome {
            output: PathBuf::from("out.md"),
            pages: 9,
            recovery_ratio: 0.994,
            passed: true,
            skipped_pages: 2,
        };
        let lines = success_lines(&outcome);
        assert!(lines.iter().any(|l| l.contains("99.4%")));
        assert!(lines.iter().any(|l| l.contains("2 page(s) were skipped")));
    }

    #[test]
    fn success_lines_omit_skip_note_when_none_skipped() {
        let outcome = ConvertOutcome {
            output: PathBuf::from("out.md"),
            pages: 3,
            recovery_ratio: 1.0,
            passed: true,
            skipped_pages: 0,
        };
        let lines = success_lines(&outcome);
        assert!(!lines.iter().any(|l| l.contains("skipped")));
    }

    #[test]
    fn folder_input_picker_fuzzy_filters_directories() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("annual_reports/2024")).unwrap();
        std::fs::create_dir_all(root.join("invoices")).unwrap();

        let mut state = ConvertFolderState::new(root.to_path_buf());
        // The root plus three nested dirs are discovered.
        assert!(state.dir_candidates.len() >= 4, "candidates = {:?}", state.dir_display);

        // Fuzzy-filtering to "report" keeps the reports subtree and drops invoices.
        state.query = "report".to_string();
        state.refilter_dirs();
        assert!(!state.ranked.is_empty());
        let ranked_names: Vec<&String> = state.ranked.iter().map(|&i| &state.dir_display[i]).collect();
        // The reports subtree ranks; the unrelated "invoices" dir is dropped.
        assert!(ranked_names.iter().any(|n| n.contains("annual_reports")));
        assert!(!ranked_names.iter().any(|n| n.as_str() == "invoices"));

        // A non-matching query drops everything.
        state.query = "zzzzz".to_string();
        state.refilter_dirs();
        assert!(state.ranked.is_empty());
    }

    #[test]
    fn folder_input_picker_selects_dir_and_advances_to_output() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("papers")).unwrap();

        let mut state = ConvertFolderState::new(root.to_path_buf());
        state.select_input_dir(root.join("papers"));
        assert!(matches!(state.stage, FolderStage::Output));
        assert_eq!(state.input_root, root.join("papers"));
        // Default output is the `markdown` subdir of the chosen input.
        assert_eq!(state.output_text, root.join("papers").join("markdown").display().to_string());
    }

    #[test]
    fn batch_start_mirrors_paths_into_output_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("top.pdf"), b"x").unwrap();
        std::fs::write(root.join("a/b/deep.pdf"), b"x").unwrap();

        let mut state = ConvertFolderState::new(root.to_path_buf());
        state.output_text = root.join("out").display().to_string();
        state.start();

        assert_eq!(state.items.len(), 2);
        let deep = state.items.iter().find(|i| i.input.ends_with("deep.pdf")).unwrap();
        assert_eq!(deep.output, root.join("out/a/b/deep.md"));
        assert!(state.needs_step());
    }
}
