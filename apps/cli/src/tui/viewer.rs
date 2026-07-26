//! The **PDF Viewer** view: open a PDF and read it in the terminal, either as
//! reflowed Markdown (the default, always-available mode) or as rasterised page
//! images (plumbing now, real render later).
//!
//! # Two modes
//!
//! * **Reflow** (default) drives the *same* engine pipeline the CLI uses —
//!   [`kopitiam_pdf::extract`] → [`kopitiam_document::reconstruct`] →
//!   [`kopitiam_markdown::render_document`] — turning the PDF into a Markdown
//!   string that is shown in a scrollable, searchable pane. No rasteriser is
//!   involved, so this works on every terminal today, including Termux.
//! * **Image** displays a rasterised page through [`ratatui_image`], which
//!   auto-detects the terminal's graphics protocol (kitty / sixel / iTerm2) and
//!   **falls back to Unicode half-blocks** where none is available — the reason
//!   the viewer stays Android/Termux-safe. The pixels come from a
//!   [`PageRasterizer`]; the shipped [`MupdfRasterizer`] renders real pages
//!   through the ported MuPDF draw device (glyph outlines, vector, colour,
//!   images). A rasterise error (e.g. an unopenable PDF, or `Error::limit` on a
//!   pathological page) is caught and the pane shows a friendly notice instead
//!   of crashing.
//!
//! # Selecting a PDF
//!
//! Opened from the Home menu the view starts on a fuzzy finder (the same
//! `discover_pdfs` + `fuzzy_rank` picker pattern as `convert`); opened from the
//! File Explorer on a `.pdf` it lands straight on the document.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{DynamicImage, RgbaImage};
use kopitiam_pdf::mupdf::{PdfDocument, Pixmap, rasterize_page};
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use ratatui_image::{Image, Resize, picker::Picker};

use super::Transition;
use super::logic::{discover_pdfs, fuzzy_rank};
use super::theme::{CHILLI, DIM, GOLD, STEAM, TAN};

// ---- PageRasterizer seam ------------------------------------------------

/// The seam between the viewer's image mode and whatever turns a PDF page into
/// pixels. The UI depends only on this trait, so the real MuPDF rasteriser (a
/// future `kopitiam-pdf` wave) drops in by providing a new `impl` — no widget,
/// navigation, or state code has to change.
pub trait PageRasterizer {
    /// How many pages `pdf` has, so the viewer can bound page navigation.
    fn page_count(&self, pdf: &Path) -> Result<usize>;

    /// Rasterise one **0-based** `page` of `pdf` at `dpi` into an in-memory RGBA
    /// buffer, ready for [`ratatui_image`] to encode for the active terminal.
    fn rasterize(&self, pdf: &Path, page: usize, dpi: f32) -> Result<RgbaImage>;
}

/// A rasteriser whose every call returns a clear error. Image mode now defaults
/// to the real [`MupdfRasterizer`], so this is retained as the graceful
/// "rendering unavailable" fallback and as the erroring test double that
/// exercises the pane's friendly-notice path (it reports no page count and never
/// produces pixels). Constructed only from tests today, hence the `dead_code`
/// allowance on non-test builds.
#[cfg_attr(not(test), allow(dead_code))]
pub struct UnavailableRasterizer;

/// The one message both stub methods carry, so the notice and any log read the
/// same.
const RASTERIZER_UNAVAILABLE: &str =
    "the MuPDF page rasterizer is not wired up yet (coming in a future kopitiam-pdf wave)";

impl PageRasterizer for UnavailableRasterizer {
    fn page_count(&self, _pdf: &Path) -> Result<usize> {
        anyhow::bail!(RASTERIZER_UNAVAILABLE)
    }

    fn rasterize(&self, _pdf: &Path, _page: usize, _dpi: f32) -> Result<RgbaImage> {
        anyhow::bail!(RASTERIZER_UNAVAILABLE)
    }
}

/// The real page rasteriser: it opens the PDF once (cached), renders a page to a
/// [`Pixmap`] through the ported MuPDF draw device
/// ([`kopitiam_pdf::mupdf::rasterize_page`] — real glyph outlines, vector fills,
/// colour and images), and converts that to the [`RgbaImage`] `ratatui-image`
/// encodes for the terminal. This is the **single** rendering path both the
/// standalone `kopitiam view` command and the TUI's image mode drive.
#[derive(Default)]
pub struct MupdfRasterizer {
    /// The most-recently-opened document, keyed by its path, so paging and
    /// zooming do not reparse the whole PDF every frame. `RefCell` keeps the
    /// `&self`-only [`PageRasterizer`] seam while still caching.
    cached: RefCell<Option<(PathBuf, PdfDocument)>>,
}

impl MupdfRasterizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `f` against the opened document for `pdf`, (re)opening it from disk
    /// only on a cache miss. The closure runs while the cache is borrowed, so the
    /// parsed [`PdfDocument`] never has to be cloned.
    fn with_doc<T>(&self, pdf: &Path, f: impl FnOnce(&PdfDocument) -> Result<T>) -> Result<T> {
        {
            let mut slot = self.cached.borrow_mut();
            let hit = matches!(slot.as_ref(), Some((p, _)) if p == pdf);
            if !hit {
                let bytes = std::fs::read(pdf)
                    .with_context(|| format!("could not read {}", pdf.display()))?;
                let doc = PdfDocument::open(bytes)
                    .map_err(|e| anyhow::anyhow!("could not open {} as a PDF: {e}", pdf.display()))?;
                *slot = Some((pdf.to_path_buf(), doc));
            }
        }
        let slot = self.cached.borrow();
        let (_, doc) = slot.as_ref().expect("cache just populated above");
        f(doc)
    }
}

impl PageRasterizer for MupdfRasterizer {
    fn page_count(&self, pdf: &Path) -> Result<usize> {
        self.with_doc(pdf, |doc| Ok(doc.page_count()))
    }

    fn rasterize(&self, pdf: &Path, page: usize, dpi: f32) -> Result<RgbaImage> {
        // `rasterize_page` returns `Error::limit` for a pathological page (a huge
        // MediaBox and/or dpi); surface it as a normal `Err` so the viewer shows
        // a notice instead of the process aborting on a multi-GB allocation.
        self.with_doc(pdf, |doc| {
            let pixmap = rasterize_page(doc, page, dpi).map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(pixmap_to_rgba(&pixmap))
        })
    }
}

/// Convert a rasterised [`Pixmap`] into the [`RgbaImage`] `ratatui-image`
/// consumes. Handles the draw device's three component layouts — DeviceGray
/// (`n = 1`), DeviceRGB (`n = 3`, the default target) and RGBA (`n = 4`) — and
/// copies **row by row honouring `stride`**, so a pixmap whose `stride` exceeds
/// `w * n` (row padding) is laid out correctly rather than sheared. An undersized
/// `samples` buffer yields a blank image instead of an out-of-bounds panic.
fn pixmap_to_rgba(pm: &Pixmap) -> RgbaImage {
    let (w, h, n, stride) = (pm.w, pm.h, pm.n as usize, pm.stride);
    let mut img = RgbaImage::new(w, h);
    if n == 0 || pm.samples.len() < stride.saturating_mul(h as usize) {
        return img; // malformed: transparent-black rather than indexing OOB
    }
    for y in 0..h as usize {
        let row = y * stride;
        for x in 0..w as usize {
            let i = row + x * n;
            let rgba = match n {
                1 => {
                    let v = pm.samples[i];
                    image::Rgba([v, v, v, 255])
                }
                2 => {
                    let v = pm.samples[i];
                    image::Rgba([v, v, v, pm.samples[i + 1]])
                }
                3 => image::Rgba([pm.samples[i], pm.samples[i + 1], pm.samples[i + 2], 255]),
                _ => image::Rgba([
                    pm.samples[i],
                    pm.samples[i + 1],
                    pm.samples[i + 2],
                    pm.samples[i + 3],
                ]),
            };
            img.put_pixel(x as u32, y as u32, rgba);
        }
    }
    img
}

// ---- reflow pipeline (pure, UI-free) ------------------------------------

/// Run the extract → reconstruct → render pipeline for `pdf` and return the
/// Markdown, or a human-readable error string.
///
/// This is the *same* core the `convert` view and the CLI drive, minus writing a
/// file — the viewer only needs the string. Kept a free function (no `self`, no
/// ratatui) so it is unit-testable without a terminal.
pub fn reflow_to_markdown(pdf: &Path) -> Result<String, String> {
    let pages = match kopitiam_pdf::extract(pdf) {
        Ok(pages) => pages,
        Err(err @ kopitiam_pdf::ExtractError::UnsupportedFont(_)) => {
            return Err(format!(
                "could not extract text ({err}) — the PDF uses fonts with no unicode mapping"
            ));
        }
        Err(err) => return Err(format!("could not extract text: {err}")),
    };
    let document = kopitiam_document::reconstruct(&pages);
    Ok(kopitiam_markdown::render_document(&document))
}

// ---- viewer state -------------------------------------------------------

/// Which of the two panes the [`ViewDoc`] is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewerMode {
    /// Reflowed Markdown (default).
    Reflow,
    /// Rasterised page images.
    Image,
}

/// The reflow pane's content lifecycle. `Pending` renders a "rendering…" notice
/// for one frame; the router then calls [`ViewerState::render_now`], which runs
/// the (possibly slow) pipeline and lands on `Ready` or `Failed`. Splitting it
/// this way means the app paints the notice before blocking, rather than freezing
/// on a black screen.
enum Reflow {
    Pending,
    Ready(Vec<String>),
    Failed(String),
}

/// Live `/`-search over the reflow text.
#[derive(Default)]
struct Search {
    /// True while the query is being typed (keys feed the box, not the pane).
    typing: bool,
    query: String,
    /// Indices of content lines that contain the query (case-insensitive).
    matches: Vec<usize>,
    /// Position within `matches` of the currently-focused hit.
    current: usize,
}

const DPI_DEFAULT: f32 = 150.0;
const DPI_MIN: f32 = 50.0;
const DPI_MAX: f32 = 600.0;
const DPI_STEP: f32 = 25.0;

/// One rasterised page, cached so a real rasteriser is not re-run every frame —
/// only when the page or dpi changes.
struct Raster {
    page: usize,
    dpi_key: u32,
    image: RgbaImage,
}

/// The opened document: everything the actual viewer (post-pick) owns.
///
/// `pub(crate)` because the standalone `kopitiam view` command
/// ([`crate::view`]) drives one directly — the same navigation, zoom and
/// image-rendering path the TUI uses, so there is no forked viewer logic.
pub struct ViewDoc {
    pdf: PathBuf,
    mode: ViewerMode,
    // reflow pane
    content: Reflow,
    scroll: u16,
    /// Inner viewport height from the last render, for scroll clamping / `End`.
    viewport_h: u16,
    search: Search,
    // image pane
    rasterizer: Box<dyn PageRasterizer>,
    page: usize,
    /// Total pages if the rasteriser could report them; `None` (stub) means
    /// unknown, and navigation is only floored at 0.
    page_count: Option<usize>,
    dpi: f32,
    /// The last error the rasteriser returned, shown in the friendly notice.
    image_error: Option<String>,
    /// Cached rasterised page (see [`Raster`]).
    raster: Option<Raster>,
    /// `Some(buffer)` while a "go to page" number is being typed in image mode;
    /// digits feed the buffer, Enter commits, Esc cancels.
    goto: Option<String>,
    /// The graphics-protocol picker, built lazily the first time a real image is
    /// actually available (never with the stub), so terminal-querying I/O only
    /// happens when it can pay off.
    picker: Option<Picker>,
}

impl ViewDoc {
    /// Open `pdf` in the default (reflow) mode, backed by the given rasteriser
    /// for image mode. The reflow content starts `Pending` so the router renders
    /// a notice and then calls [`ViewerState::render_now`].
    fn new(pdf: PathBuf, rasterizer: Box<dyn PageRasterizer>) -> Self {
        // Ask the rasteriser for a page count up front; the stub errors, leaving
        // it unknown, which the UI handles.
        let page_count = rasterizer.page_count(&pdf).ok();
        Self {
            pdf,
            mode: ViewerMode::Reflow,
            content: Reflow::Pending,
            scroll: 0,
            viewport_h: 1,
            search: Search::default(),
            rasterizer,
            page: 0,
            page_count,
            dpi: DPI_DEFAULT,
            image_error: None,
            raster: None,
            goto: None,
            picker: None,
        }
    }

    /// Open `pdf` straight into **image** mode at an optional initial 1-based
    /// `page` and `dpi`, backed by `rasterizer`. This is the entry the standalone
    /// `kopitiam view` command uses; the reflow pane stays lazily `Pending` and is
    /// only ever rendered if the user toggles to it.
    pub(crate) fn open_image(
        pdf: PathBuf,
        rasterizer: Box<dyn PageRasterizer>,
        page: Option<usize>,
        dpi: Option<f32>,
    ) -> Self {
        let mut doc = Self::new(pdf, rasterizer);
        doc.mode = ViewerMode::Image;
        if let Some(dpi) = dpi {
            doc.dpi = dpi.clamp(DPI_MIN, DPI_MAX);
        }
        if let Some(page) = page {
            doc.goto_page_1based(page);
        }
        doc
    }

    /// The reflow pane is waiting for its first render.
    fn is_pending(&self) -> bool {
        matches!(self.content, Reflow::Pending)
    }

    /// True while a text sub-mode (search box or goto-page box) is capturing
    /// keys, so a host loop knows not to treat `q`/`Esc` as "quit".
    pub(crate) fn is_capturing_text(&self) -> bool {
        self.search.typing || self.goto.is_some()
    }

    /// The standalone viewer only needs to run the (slow) reflow pipeline if the
    /// user has actually switched to reflow mode and it has not rendered yet.
    pub(crate) fn needs_reflow_render(&self) -> bool {
        self.mode == ViewerMode::Reflow && self.is_pending()
    }

    /// Jump to a 1-based page number, clamped into range (floored at page 1, and
    /// capped at the last page when the count is known).
    fn goto_page_1based(&mut self, page_1based: usize) {
        let target = page_1based.saturating_sub(1);
        let clamped = match self.page_count {
            Some(count) if count > 0 => target.min(count - 1),
            _ => target,
        };
        if clamped != self.page {
            self.page = clamped;
            self.invalidate_raster();
        }
    }

    /// Run the reflow pipeline now (called by the router once, after the
    /// `Pending` notice has painted).
    pub(crate) fn render_now(&mut self) {
        self.content = match reflow_to_markdown(&self.pdf) {
            Ok(md) => Reflow::Ready(md.lines().map(str::to_string).collect()),
            Err(err) => Reflow::Failed(err),
        };
        self.recompute_matches();
    }

    /// The greatest scroll offset that still shows content, given the last
    /// viewport height.
    fn max_scroll(&self) -> u16 {
        match &self.content {
            Reflow::Ready(lines) => {
                (lines.len() as u16).saturating_sub(self.viewport_h.max(1))
            }
            _ => 0,
        }
    }

    /// Rebuild the set of matching line indices for the current query, keeping
    /// the focused-hit cursor in range.
    fn recompute_matches(&mut self) {
        let needle = self.search.query.to_lowercase();
        self.search.matches = if needle.is_empty() {
            Vec::new()
        } else if let Reflow::Ready(lines) = &self.content {
            lines
                .iter()
                .enumerate()
                .filter(|(_, l)| l.to_lowercase().contains(&needle))
                .map(|(i, _)| i)
                .collect()
        } else {
            Vec::new()
        };
        if self.search.current >= self.search.matches.len() {
            self.search.current = 0;
        }
    }

    /// Scroll so the currently-focused match sits at the top of the viewport.
    fn scroll_to_current_match(&mut self) {
        if let Some(&line) = self.search.matches.get(self.search.current) {
            self.scroll = (line as u16).min(self.max_scroll());
        }
    }

    /// Move the focused match by `delta` (wrapping) and scroll to it.
    fn step_match(&mut self, delta: i32) {
        if self.search.matches.is_empty() {
            return;
        }
        let len = self.search.matches.len() as i32;
        self.search.current = (self.search.current as i32 + delta).rem_euclid(len) as usize;
        self.scroll_to_current_match();
    }

    /// Advance to the next page (bounded by the page count when known).
    fn next_page(&mut self) {
        match self.page_count {
            Some(n) if self.page + 1 >= n => {}
            _ => {
                self.page += 1;
                self.invalidate_raster();
            }
        }
    }

    /// Go to the previous page (floored at the first).
    fn prev_page(&mut self) {
        if self.page > 0 {
            self.page -= 1;
            self.invalidate_raster();
        }
    }

    fn zoom_in(&mut self) {
        self.dpi = (self.dpi + DPI_STEP).min(DPI_MAX);
        self.invalidate_raster();
    }

    fn zoom_out(&mut self) {
        self.dpi = (self.dpi - DPI_STEP).max(DPI_MIN);
        self.invalidate_raster();
    }

    /// Drop any cached page pixels/protocol so the next image render re-fetches.
    fn invalidate_raster(&mut self) {
        self.raster = None;
    }

    pub(crate) fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        if self.goto.is_some() {
            return vec![("digits", "page"), ("Enter", "go"), ("Esc", "cancel")];
        }
        if self.search.typing {
            return vec![("type", "search"), ("Enter", "jump"), ("Esc", "cancel")];
        }
        match self.mode {
            ViewerMode::Reflow => vec![
                ("↑/↓ j/k", "scroll"),
                ("PgUp/PgDn", "page"),
                ("/", "search"),
                ("n/N", "match"),
                ("i", "image"),
                ("Esc", "home"),
            ],
            ViewerMode::Image => vec![
                ("←/→ n/p", "page"),
                ("g", "goto"),
                ("+/-", "zoom"),
                ("r", "reflow"),
                ("Esc", "home"),
            ],
        }
    }

    /// Handle one key while a document is open. Returns the router transition.
    pub(crate) fn on_key(&mut self, key: KeyEvent, ctrl: bool) -> Transition {
        // While typing a goto-page number (image mode), keys feed that box.
        if self.goto.is_some() {
            match key.code {
                KeyCode::Esc => self.goto = None,
                KeyCode::Enter => {
                    let buf = self.goto.take().unwrap_or_default();
                    if let Ok(n) = buf.trim().parse::<usize>()
                        && n >= 1
                    {
                        self.goto_page_1based(n);
                    }
                }
                KeyCode::Backspace => {
                    if let Some(buf) = self.goto.as_mut() {
                        buf.pop();
                    }
                }
                KeyCode::Char(c) if c.is_ascii_digit() => {
                    if let Some(buf) = self.goto.as_mut() {
                        buf.push(c);
                    }
                }
                _ => {}
            }
            return Transition::Stay;
        }

        // While typing a search query, keys feed the box.
        if self.search.typing {
            match key.code {
                KeyCode::Esc => {
                    self.search.typing = false;
                    self.search.query.clear();
                    self.recompute_matches();
                }
                KeyCode::Enter => {
                    self.search.typing = false;
                    self.search.current = 0;
                    self.scroll_to_current_match();
                }
                KeyCode::Backspace => {
                    self.search.query.pop();
                    self.recompute_matches();
                }
                KeyCode::Char(c) if !ctrl => {
                    self.search.query.push(c);
                    self.recompute_matches();
                }
                _ => {}
            }
            return Transition::Stay;
        }

        match key.code {
            KeyCode::Esc => return Transition::Home,
            KeyCode::Tab => self.toggle_mode(),
            _ => match self.mode {
                ViewerMode::Reflow => self.on_key_reflow(key),
                ViewerMode::Image => self.on_key_image(key),
            },
        }
        Transition::Stay
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            ViewerMode::Reflow => ViewerMode::Image,
            ViewerMode::Image => ViewerMode::Reflow,
        };
    }

    fn on_key_reflow(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('i') => self.mode = ViewerMode::Image,
            KeyCode::Char('/') => {
                self.search.typing = true;
                self.search.query.clear();
                self.recompute_matches();
            }
            KeyCode::Char('n') => self.step_match(1),
            KeyCode::Char('N') => self.step_match(-1),
            KeyCode::Up | KeyCode::Char('k') => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll = self.scroll.saturating_add(1).min(self.max_scroll());
            }
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(self.viewport_h.max(1)),
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(self.viewport_h.max(1)).min(self.max_scroll());
            }
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = self.max_scroll(),
            _ => {}
        }
    }

    fn on_key_image(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('r') => self.mode = ViewerMode::Reflow,
            KeyCode::Char('n') | KeyCode::Right | KeyCode::PageDown => self.next_page(),
            KeyCode::Char('p') | KeyCode::Left | KeyCode::PageUp => self.prev_page(),
            KeyCode::Char('+') | KeyCode::Char('=') => self.zoom_in(),
            KeyCode::Char('-') | KeyCode::Char('_') => self.zoom_out(),
            // Start typing a "go to page" number; digits/Enter handled in `on_key`.
            KeyCode::Char('g') => self.goto = Some(String::new()),
            _ => {}
        }
    }

    // ---- rendering ------------------------------------------------------

    pub(crate) fn render(&mut self, frame: &mut Frame, area: Rect) {
        match self.mode {
            ViewerMode::Reflow => self.render_reflow(frame, area),
            ViewerMode::Image => self.render_image(frame, area),
        }
    }

    fn render_reflow(&mut self, frame: &mut Frame, area: Rect) {
        // Reserve a row for the search box when searching or with an active query.
        let show_search = self.search.typing || !self.search.query.is_empty();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(if show_search { 1 } else { 0 })])
            .split(area);
        let pane = chunks[0];

        let name = self.pdf.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let title = format!(" {name} · reflow ");
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(title, Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));

        // Track the inner height so scroll clamps and `End`/`PgDn` stay correct.
        self.viewport_h = block.inner(pane).height.max(1);

        match &self.content {
            Reflow::Pending => {
                frame.render_widget(
                    Paragraph::new("Rendering… extracting text and reflowing to Markdown.")
                        .style(Style::default().fg(STEAM))
                        .block(block),
                    pane,
                );
            }
            Reflow::Failed(err) => {
                let lines = vec![
                    Line::from(Span::styled("Could not render this PDF as text:", Style::default().fg(CHILLI))),
                    Line::from(""),
                    Line::from(Span::styled(err.clone(), Style::default().fg(TAN))),
                    Line::from(""),
                    Line::from(Span::styled("Press `i` to try image mode, or Esc for home.", Style::default().fg(STEAM))),
                ];
                frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }).block(block), pane);
            }
            Reflow::Ready(lines) => {
                let clamped = self.scroll.min(self.max_scroll());
                let current_line = self.search.matches.get(self.search.current).copied();
                let rendered: Vec<Line> = lines
                    .iter()
                    .enumerate()
                    .map(|(i, l)| highlight_line(l, &self.search.query, Some(i) == current_line))
                    .collect();
                frame.render_widget(
                    Paragraph::new(Text::from(rendered)).scroll((clamped, 0)).block(block),
                    pane,
                );
            }
        }

        if show_search {
            let count = self.search.matches.len();
            let position = if count == 0 { 0 } else { self.search.current + 1 };
            let label = format!("/{}", self.search.query);
            let status = format!("  [{position}/{count}]");
            let cursor = if self.search.typing { "▌" } else { "" };
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(label, Style::default().fg(GOLD)),
                    Span::styled(cursor.to_string(), Style::default().fg(GOLD)),
                    Span::styled(status, Style::default().fg(DIM)),
                ])),
                chunks[1],
            );
        }
    }

    fn render_image(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(area);

        // Page / zoom indicator, plus the goto-page box when one is being typed.
        let total = self.page_count.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string());
        let mut spans = vec![
            Span::styled(" page ", Style::default().fg(DIM)),
            Span::styled(format!("{}/{total}", self.page + 1), Style::default().fg(GOLD).add_modifier(Modifier::BOLD)),
            Span::styled(format!("   ·   {:.0} dpi", self.dpi), Style::default().fg(STEAM)),
        ];
        if let Some(buf) = &self.goto {
            spans.push(Span::styled(format!("   ·   goto: {buf}"), Style::default().fg(GOLD)));
            spans.push(Span::styled("▌", Style::default().fg(GOLD)));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), chunks[0]);

        let pane = chunks[1];
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(" image ", Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));
        let inner = block.inner(pane);
        frame.render_widget(block, pane);

        // Fetch (and cache) the current page's pixels through the seam.
        self.ensure_raster();

        if let Some(raster) = &self.raster {
            // Real pixels: encode for the active terminal (kitty/sixel/iTerm2, or
            // half-blocks as the Termux-safe fallback) and paint the widget.
            let image = raster.image.clone();
            if self.picker.is_none() {
                self.picker = Some(acquire_picker());
            }
            if let Some(picker) = &self.picker {
                let dynamic = DynamicImage::ImageRgba8(image);
                match picker.new_protocol(dynamic, inner, Resize::Fit(None)) {
                    Ok(protocol) => frame.render_widget(Image::new(&protocol), inner),
                    Err(err) => frame.render_widget(notice_paragraph(&format!("could not encode page image: {err}")), inner),
                }
            }
        } else {
            // No pixels (the stub, or a genuine error): a friendly notice, never a
            // crash.
            let detail = self.image_error.clone().unwrap_or_else(|| RASTERIZER_UNAVAILABLE.to_string());
            let lines = vec![
                Line::from(Span::styled("Page image rendering is not available yet.", Style::default().fg(GOLD).add_modifier(Modifier::BOLD))),
                Line::from(""),
                Line::from(Span::styled(detail, Style::default().fg(TAN))),
                Line::from(""),
                Line::from(Span::styled("Press `r` for the reflow (Markdown) view, which works today.", Style::default().fg(STEAM))),
            ];
            frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }), inner);
        }
    }

    /// Ensure `self.raster` holds the pixels for the current page+dpi, calling
    /// the rasteriser only when the cache is stale. On error the cache stays
    /// empty and `image_error` records why.
    fn ensure_raster(&mut self) {
        let dpi_key = self.dpi.to_bits();
        if self.raster.as_ref().is_some_and(|r| r.page == self.page && r.dpi_key == dpi_key) {
            return;
        }
        match self.rasterizer.rasterize(&self.pdf, self.page, self.dpi) {
            Ok(image) => {
                self.image_error = None;
                self.raster = Some(Raster { page: self.page, dpi_key, image });
            }
            Err(err) => {
                self.image_error = Some(err.to_string());
                self.raster = None;
            }
        }
    }
}

/// Acquire a graphics-protocol picker, querying the terminal for its best
/// protocol and font size; on any failure fall back to a default font size,
/// which still yields a working **half-block** picker (the Termux-safe path).
fn acquire_picker() -> Picker {
    Picker::from_query_stdio().unwrap_or_else(|_| Picker::from_fontsize((10, 20)))
}

/// A single dim-toned notice paragraph.
fn notice_paragraph(text: &str) -> Paragraph<'static> {
    Paragraph::new(text.to_string()).style(Style::default().fg(STEAM)).wrap(Wrap { trim: false })
}

/// Build a styled [`Line`] for one content line, highlighting each occurrence of
/// `query` (case-insensitive). The line holding the focused match is tinted so
/// the current hit stands out from the rest.
fn highlight_line(text: &str, query: &str, is_current: bool) -> Line<'static> {
    let base = if is_current {
        Style::default().fg(TAN).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TAN)
    };
    if query.is_empty() {
        return Line::from(Span::styled(text.to_string(), base));
    }
    let hay = text.to_lowercase();
    let needle = query.to_lowercase();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = hay[at..].find(&needle) {
        let start = at + rel;
        if start > at {
            spans.push(Span::styled(text[at..start].to_string(), base));
        }
        let end = start + needle.len();
        spans.push(Span::styled(
            text[start..end].to_string(),
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD | Modifier::REVERSED),
        ));
        at = end;
    }
    if at < text.len() {
        spans.push(Span::styled(text[at..].to_string(), base));
    }
    Line::from(spans)
}

// ---- picker stage + the ViewerState the router owns ---------------------

#[derive(PartialEq)]
enum Focus {
    Query,
    Root,
}

enum Stage {
    /// The fuzzy finder for choosing a PDF.
    Pick,
    /// A document is open.
    View,
}

/// The Viewer view the router holds. It owns the PDF fuzzy-finder (the `Pick`
/// stage) and, once a file is chosen, the open [`ViewDoc`] (the `View` stage).
pub struct ViewerState {
    // picker (Pick stage) — mirrors the `convert` picker.
    root: PathBuf,
    root_input: String,
    candidates: Vec<PathBuf>,
    display: Vec<String>,
    query: String,
    ranked: Vec<usize>,
    selected: usize,
    focus: Focus,
    stage: Stage,
    doc: Option<ViewDoc>,
}

impl ViewerState {
    /// Open the viewer on the fuzzy finder, rooted at `root`.
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
            doc: None,
        };
        state.rescan();
        state
    }

    /// Open the viewer straight on `pdf` (from the File Explorer), skipping the
    /// finder.
    pub fn with_file(pdf: PathBuf) -> Self {
        let root = pdf.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let mut state = Self::new(root);
        state.open(pdf);
        state
    }

    /// Open `pdf` into the `View` stage, backed by the real MuPDF rasteriser —
    /// the same [`MupdfRasterizer`] the standalone `kopitiam view` command uses,
    /// so image mode shows real glyphs / vector / colour here too.
    fn open(&mut self, pdf: PathBuf) {
        self.doc = Some(ViewDoc::new(pdf, Box::new(MupdfRasterizer::new())));
        self.stage = Stage::View;
    }

    fn rescan(&mut self) {
        self.candidates = discover_pdfs(&self.root);
        self.display = self
            .candidates
            .iter()
            .map(|p| p.strip_prefix(&self.root).unwrap_or(p).display().to_string())
            .collect();
        self.refilter();
    }

    fn refilter(&mut self) {
        self.ranked = fuzzy_rank(&self.query, &self.display);
        if self.selected >= self.ranked.len() {
            self.selected = self.ranked.len().saturating_sub(1);
        }
    }

    /// True while the open document still needs its first reflow render.
    pub fn needs_render(&self) -> bool {
        self.doc.as_ref().is_some_and(ViewDoc::is_pending)
    }

    /// Run the pending reflow render (called by the router once per pending doc).
    pub fn render_now(&mut self) {
        if let Some(doc) = self.doc.as_mut() {
            doc.render_now();
        }
    }

    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        match self.stage {
            Stage::Pick => vec![
                ("type", "filter"),
                ("↑/↓", "move"),
                ("Tab", "root/search"),
                ("Enter", "view"),
                ("Esc", "home"),
            ],
            Stage::View => self.doc.as_ref().map(ViewDoc::footer_hints).unwrap_or_default(),
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Char('c') && ctrl {
            return Transition::Quit;
        }
        match self.stage {
            Stage::Pick => self.on_key_pick(key, ctrl),
            Stage::View => match self.doc.as_mut() {
                Some(doc) => doc.on_key(key, ctrl),
                None => Transition::Home,
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
                        self.open(pdf);
                    }
                }
            },
            KeyCode::Backspace => match self.focus {
                Focus::Query => {
                    self.query.pop();
                    self.refilter();
                }
                Focus::Root => {
                    self.root_input.pop();
                }
            },
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
        self.selected = (self.selected as i32 + delta).rem_euclid(len) as usize;
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        match self.stage {
            Stage::Pick => self.render_pick(frame, area),
            Stage::View => {
                if let Some(doc) = self.doc.as_mut() {
                    doc.render(frame, area);
                }
            }
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
}

/// A single-line input box, its border brightened when focused. (A local copy of
/// `convert`'s helper so the viewer stays self-contained.)
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
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    /// A test rasteriser: a known page count and a 1x1 image, so image-mode paths
    /// that need real pixels can be exercised without a PDF engine.
    struct FakeRasterizer {
        pages: usize,
    }
    impl PageRasterizer for FakeRasterizer {
        fn page_count(&self, _pdf: &Path) -> Result<usize> {
            Ok(self.pages)
        }
        fn rasterize(&self, _pdf: &Path, _page: usize, _dpi: f32) -> Result<RgbaImage> {
            Ok(RgbaImage::new(1, 1))
        }
    }

    fn doc_with(rasterizer: Box<dyn PageRasterizer>) -> ViewDoc {
        ViewDoc::new(PathBuf::from("/x/paper.pdf"), rasterizer)
    }

    fn ready_doc(lines: &[&str]) -> ViewDoc {
        let mut doc = doc_with(Box::new(UnavailableRasterizer));
        doc.content = Reflow::Ready(lines.iter().map(|s| s.to_string()).collect());
        doc.viewport_h = 3;
        doc
    }

    // ---- rasteriser seam ----

    #[test]
    fn stub_rasterizer_errors_on_both_methods() {
        let stub = UnavailableRasterizer;
        assert!(stub.page_count(Path::new("/x.pdf")).is_err());
        assert!(stub.rasterize(Path::new("/x.pdf"), 0, 150.0).is_err());
    }

    #[test]
    fn new_doc_defaults_to_reflow_and_pending() {
        let doc = doc_with(Box::new(UnavailableRasterizer));
        assert_eq!(doc.mode, ViewerMode::Reflow);
        assert!(doc.is_pending());
        // The stub cannot report a page count.
        assert_eq!(doc.page_count, None);
    }

    #[test]
    fn known_page_count_comes_from_the_rasterizer() {
        let doc = doc_with(Box::new(FakeRasterizer { pages: 7 }));
        assert_eq!(doc.page_count, Some(7));
    }

    // ---- reflow scroll ----

    #[test]
    fn reflow_scroll_down_and_up() {
        let mut doc = ready_doc(&["l0", "l1", "l2", "l3", "l4", "l5"]);
        // 6 lines, viewport 3 → max scroll 3.
        assert_eq!(doc.scroll, 0);
        doc.on_key(key(KeyCode::Char('j')), false);
        assert_eq!(doc.scroll, 1);
        doc.on_key(key(KeyCode::Down), false);
        assert_eq!(doc.scroll, 2);
        doc.on_key(key(KeyCode::Char('k')), false);
        assert_eq!(doc.scroll, 1);
    }

    #[test]
    fn reflow_scroll_clamps_and_end_home() {
        let mut doc = ready_doc(&["l0", "l1", "l2", "l3", "l4", "l5"]);
        doc.on_key(key(KeyCode::End), false);
        assert_eq!(doc.scroll, doc.max_scroll());
        assert_eq!(doc.scroll, 3);
        // Cannot scroll past the end.
        doc.on_key(key(KeyCode::Char('j')), false);
        assert_eq!(doc.scroll, 3);
        doc.on_key(key(KeyCode::Home), false);
        assert_eq!(doc.scroll, 0);
    }

    // ---- search ----

    #[test]
    fn search_flow_jumps_to_match_and_highlights() {
        let mut doc = ready_doc(&["alpha", "beta needle", "gamma", "delta needle"]);
        doc.viewport_h = 1;
        // `/` starts typing.
        doc.on_key(key(KeyCode::Char('/')), false);
        assert!(doc.search.typing);
        for c in "needle".chars() {
            doc.on_key(key(KeyCode::Char(c)), false);
        }
        // Two lines contain "needle".
        assert_eq!(doc.search.matches, vec![1, 3]);
        // Enter confirms and jumps to the first match (line 1).
        doc.on_key(key(KeyCode::Enter), false);
        assert!(!doc.search.typing);
        assert_eq!(doc.scroll, 1);
        // `n` advances to the next match (line 3).
        doc.on_key(key(KeyCode::Char('n')), false);
        assert_eq!(doc.search.current, 1);
        assert_eq!(doc.scroll, 3);
        // `N` wraps back to the first.
        doc.on_key(key(KeyCode::Char('N')), false);
        assert_eq!(doc.search.current, 0);
        assert_eq!(doc.scroll, 1);
    }

    #[test]
    fn search_esc_cancels_and_clears() {
        let mut doc = ready_doc(&["one needle", "two"]);
        doc.on_key(key(KeyCode::Char('/')), false);
        for c in "needle".chars() {
            doc.on_key(key(KeyCode::Char(c)), false);
        }
        assert!(!doc.search.matches.is_empty());
        doc.on_key(key(KeyCode::Esc), false);
        assert!(!doc.search.typing);
        assert!(doc.search.query.is_empty());
        assert!(doc.search.matches.is_empty());
    }

    #[test]
    fn highlight_line_splits_around_matches() {
        let line = highlight_line("a needle here", "needle", false);
        // "a " + "needle" + " here" → three spans.
        assert_eq!(line.spans.len(), 3);
        assert_eq!(line.spans[1].content, "needle");
    }

    // ---- mode toggle ----

    #[test]
    fn tab_and_keys_toggle_modes() {
        let mut doc = doc_with(Box::new(UnavailableRasterizer));
        assert_eq!(doc.mode, ViewerMode::Reflow);
        doc.on_key(key(KeyCode::Tab), false);
        assert_eq!(doc.mode, ViewerMode::Image);
        doc.on_key(key(KeyCode::Tab), false);
        assert_eq!(doc.mode, ViewerMode::Reflow);
        // `i` → image, `r` → reflow.
        doc.on_key(key(KeyCode::Char('i')), false);
        assert_eq!(doc.mode, ViewerMode::Image);
        doc.on_key(key(KeyCode::Char('r')), false);
        assert_eq!(doc.mode, ViewerMode::Reflow);
    }

    // ---- image-mode page navigation ----

    #[test]
    fn image_nav_updates_page_index() {
        let mut doc = doc_with(Box::new(FakeRasterizer { pages: 3 }));
        doc.on_key(key(KeyCode::Char('i')), false); // enter image mode
        assert_eq!(doc.page, 0);
        doc.on_key(key(KeyCode::Char('n')), false);
        assert_eq!(doc.page, 1);
        doc.on_key(key(KeyCode::Right), false);
        assert_eq!(doc.page, 2);
        // Bounded by the 3-page count: cannot go past the last page.
        doc.on_key(key(KeyCode::Char('n')), false);
        assert_eq!(doc.page, 2);
        doc.on_key(key(KeyCode::Char('p')), false);
        assert_eq!(doc.page, 1);
        doc.on_key(key(KeyCode::Left), false);
        assert_eq!(doc.page, 0);
        // Floored at the first page.
        doc.on_key(key(KeyCode::Char('p')), false);
        assert_eq!(doc.page, 0);
    }

    #[test]
    fn unknown_page_count_allows_forward_nav() {
        // The stub reports no page count, so nav is only floored at 0.
        let mut doc = doc_with(Box::new(UnavailableRasterizer));
        doc.on_key(key(KeyCode::Char('i')), false);
        doc.on_key(key(KeyCode::Char('n')), false);
        assert_eq!(doc.page, 1);
    }

    #[test]
    fn zoom_changes_dpi_within_bounds() {
        let mut doc = doc_with(Box::new(UnavailableRasterizer));
        doc.on_key(key(KeyCode::Char('i')), false);
        assert_eq!(doc.dpi, DPI_DEFAULT);
        doc.on_key(key(KeyCode::Char('+')), false);
        assert_eq!(doc.dpi, DPI_DEFAULT + DPI_STEP);
        doc.on_key(key(KeyCode::Char('-')), false);
        doc.on_key(key(KeyCode::Char('-')), false);
        assert_eq!(doc.dpi, DPI_DEFAULT - DPI_STEP);
    }

    // ---- picker stage ----

    #[test]
    fn with_file_opens_straight_into_view() {
        let state = ViewerState::with_file(PathBuf::from("/papers/study.pdf"));
        assert!(matches!(state.stage, Stage::View));
        assert!(state.doc.is_some());
        assert!(state.needs_render());
    }

    #[test]
    fn pick_esc_returns_home() {
        let mut state = ViewerState::new(PathBuf::from("."));
        assert!(matches!(state.on_key(key(KeyCode::Esc)), Transition::Home));
    }

    #[test]
    fn view_esc_returns_home() {
        let mut state = ViewerState::with_file(PathBuf::from("/papers/study.pdf"));
        assert!(matches!(state.on_key(key(KeyCode::Esc)), Transition::Home));
    }

    #[test]
    fn ctrl_c_quits_from_viewer() {
        let mut state = ViewerState::new(PathBuf::from("."));
        let ev = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(state.on_key(ev), Transition::Quit));
    }

    // ---- Pixmap -> RgbaImage conversion (the shared render seam's core) ----

    #[test]
    fn pixmap_to_rgba_maps_dimensions_and_rgb_pixels() {
        // 2x1 DeviceRGB, tightly packed (stride == w*n).
        let pm = Pixmap {
            x: 0,
            y: 0,
            w: 2,
            h: 1,
            n: 3,
            alpha: false,
            stride: 6,
            samples: vec![10, 20, 30, 40, 50, 60],
        };
        let img = pixmap_to_rgba(&pm);
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [40, 50, 60, 255]);
    }

    #[test]
    fn pixmap_to_rgba_honors_row_stride_padding() {
        // 2x2 DeviceRGB with 2 padding bytes per row (stride 8 != w*n 6). If the
        // conversion ignored stride, row 1 would read the padding and shear.
        let pm = Pixmap {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            n: 3,
            alpha: false,
            stride: 8,
            samples: vec![
                1, 2, 3, 4, 5, 6, 0, 0, // row 0 (last 2 bytes are padding)
                7, 8, 9, 10, 11, 12, 0, 0, // row 1
            ],
        };
        let img = pixmap_to_rgba(&pm);
        assert_eq!(img.dimensions(), (2, 2));
        assert_eq!(img.get_pixel(0, 0).0, [1, 2, 3, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [4, 5, 6, 255]);
        // Row 1 must start at byte offset `stride` (8), not `w*n` (6).
        assert_eq!(img.get_pixel(0, 1).0, [7, 8, 9, 255]);
        assert_eq!(img.get_pixel(1, 1).0, [10, 11, 12, 255]);
    }

    #[test]
    fn pixmap_to_rgba_expands_gray_and_survives_a_short_buffer() {
        // DeviceGray (n=1) -> replicated across RGB, opaque alpha.
        let gray = Pixmap { x: 0, y: 0, w: 2, h: 1, n: 1, alpha: false, stride: 2, samples: vec![77, 200] };
        let img = pixmap_to_rgba(&gray);
        assert_eq!(img.get_pixel(0, 0).0, [77, 77, 77, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [200, 200, 200, 255]);

        // A malformed pixmap (samples too short) yields a blank image, not a panic.
        let bad = Pixmap { x: 0, y: 0, w: 4, h: 4, n: 3, alpha: false, stride: 12, samples: vec![1, 2, 3] };
        let img = pixmap_to_rgba(&bad);
        assert_eq!(img.dimensions(), (4, 4));
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 0]);
    }

    // ---- goto-page navigation + image-first constructor ----

    #[test]
    fn goto_page_clamps_into_range() {
        let mut doc = doc_with(Box::new(FakeRasterizer { pages: 5 }));
        doc.on_key(key(KeyCode::Char('i')), false); // image mode
        // g, 3, Enter -> page index 2 (1-based 3).
        doc.on_key(key(KeyCode::Char('g')), false);
        assert!(doc.goto.is_some());
        doc.on_key(key(KeyCode::Char('3')), false);
        doc.on_key(key(KeyCode::Enter), false);
        assert_eq!(doc.page, 2);
        assert!(doc.goto.is_none());
        // An out-of-range page clamps to the last (index 4 of 5).
        doc.on_key(key(KeyCode::Char('g')), false);
        for c in "99".chars() {
            doc.on_key(key(KeyCode::Char(c)), false);
        }
        doc.on_key(key(KeyCode::Enter), false);
        assert_eq!(doc.page, 4);
        // Esc cancels goto and leaves the page unchanged.
        doc.on_key(key(KeyCode::Char('g')), false);
        doc.on_key(key(KeyCode::Char('1')), false);
        doc.on_key(key(KeyCode::Esc), false);
        assert!(doc.goto.is_none());
        assert_eq!(doc.page, 4);
    }

    #[test]
    fn open_image_starts_in_image_mode_and_clamps_page_and_dpi() {
        let doc = ViewDoc::open_image(
            PathBuf::from("/x/p.pdf"),
            Box::new(FakeRasterizer { pages: 10 }),
            Some(4),      // 1-based initial page -> index 3
            Some(9999.0), // absurd dpi -> clamped to DPI_MAX
        );
        assert_eq!(doc.mode, ViewerMode::Image);
        assert_eq!(doc.page, 3);
        assert_eq!(doc.dpi, DPI_MAX);
        assert!(!doc.is_capturing_text());
        // A page past the end clamps to the last.
        let doc = ViewDoc::open_image(
            PathBuf::from("/x/p.pdf"),
            Box::new(FakeRasterizer { pages: 10 }),
            Some(100),
            None,
        );
        assert_eq!(doc.page, 9);
    }

    #[test]
    fn mupdf_rasterizer_errors_cleanly_on_a_missing_file() {
        // The real rasteriser must return an Err (never panic) when the PDF is not
        // openable, so the viewer shows a notice instead of crashing.
        let r = MupdfRasterizer::new();
        assert!(r.page_count(Path::new("/no/such/file.pdf")).is_err());
        assert!(r.rasterize(Path::new("/no/such/file.pdf"), 0, 150.0).is_err());
    }
}
