//! `kpdf` -- a lightweight, native (egui/eframe) PDF viewer for kopitiam-pdf.
//!
//! A GUI sibling of `kopitiam view` (apps/cli/src/view.rs +
//! apps/cli/src/tui/viewer.rs): same rasterizer
//! ([`kopitiam_pdf::mupdf::rasterize_page`]), same page/zoom/goto/mode-toggle
//! model and keybindings -- reimplemented against egui instead of
//! ratatui-image, because a terminal graphics protocol isn't always available
//! (or wanted) when eyeballing a glyph-rendering fix on a real display.
//!
//! Lives as a dev-dependency-only *example*, not a `[[bin]]`: eframe/egui
//! never belong in kopitiam-pdf's own published dependency graph -- a
//! `cargo publish` consumer of the library never needs a GUI toolkit to
//! extract text or rasterize a page. This file only calls kopitiam_pdf's
//! public API, exactly as any other consumer of the crate would.
//!
//! ```text
//! cargo run --release -p kopitiam-pdf --example kpdf                      # native file picker
//! cargo run --release -p kopitiam-pdf --example kpdf -- path/to/file.pdf  # open directly
//! ```
//!
//! With no path argument, a native "Open" dialog ([`rfd`], pure Rust: the
//! default `xdg-portal` + `wayland` features on Linux, native Win32/Cocoa
//! pickers elsewhere -- no GTK or other C toolchain needed to build it) opens
//! before the viewer window does, so there is nothing to specify up front.
//! Cancelling it exits quietly (not an error).
//!
//! # Keybindings (mirrors `kopitiam view`)
//!
//! Image mode:
//! * `n` / `→` / `PageDown` -- next page; `p` / `←` / `PageUp` -- previous.
//! * `g`, then digits, then `Enter` -- go to a page (`Esc` cancels the entry
//!   without changing the page, same as the terminal viewer).
//! * `+` / `-` -- zoom in / out (render dpi).
//! * `r` / `Tab` -- switch to reflow (text) mode.
//!
//! Reflow mode:
//! * `j` / `↓` -- scroll down one line; `k` / `↑` -- scroll up one line.
//! * `PageDown` / `PageUp` -- scroll by a viewport.
//! * `i` / `Tab` -- switch to image mode.
//!
//! Global: `o` -- open a different PDF via the same native file picker
//! (replaces the current document in place; on cancel or a failed open, the
//! current document keeps showing and the status bar reports the error).
//! `q` / `Esc` quit (when not mid-`goto`, where `Esc` cancels the entry
//! instead, matching the terminal viewer's `is_capturing_text` guard);
//! `Ctrl+C` always quits.
//!
//! # Not carried over from `kopitiam view`
//!
//! Search (`/`, `n`/`N` match-stepping) is the one piece of the terminal
//! viewer's reflow mode this does not reimplement -- out of scope for a
//! "lightweight" viewer; scroll to read instead.

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;
use kopitiam_pdf::mupdf::{rasterize_page, PdfDocument, Pixmap};
use kopitiam_pdf::{extract_mupdf_from_bytes, Page as TextPage};

// Same render-dpi range/step as `kopitiam view`
// (apps/cli/src/tui/viewer.rs's DPI_DEFAULT/MIN/MAX/STEP), kept in parity on
// purpose -- there is no reason the two viewers should zoom differently.
const DPI_DEFAULT: f32 = 150.0;
const DPI_MIN: f32 = 50.0;
const DPI_MAX: f32 = 600.0;
const DPI_STEP: f32 = 25.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Image,
    Reflow,
}

struct KpdfApp {
    doc: PdfDocument,
    path: PathBuf,
    page_count: usize,
    /// 0-based, matching [`rasterize_page`]'s own indexing.
    page: usize,
    dpi: f32,
    mode: Mode,
    texture: Option<egui::TextureHandle>,
    /// The `(page, dpi)` the cached `texture` was rendered at, so a repaint
    /// with nothing changed doesn't re-rasterize every frame.
    texture_key: Option<(usize, u32)>,
    /// `Some(buf)` while a "go to page" number is being typed (image mode).
    goto: Option<String>,
    /// Lazily extracted once, on first entry to reflow mode -- extraction
    /// walks the whole document, so it is not worth doing eagerly for a
    /// session that might only ever use image mode.
    reflow_pages: Option<Result<Vec<TextPage>, String>>,
    reflow_scroll: f32,
    status: Option<String>,
}

impl KpdfApp {
    fn open(path: PathBuf) -> Result<KpdfApp, String> {
        let (doc, page_count) = load_doc(&path)?;
        Ok(KpdfApp {
            doc,
            path,
            page_count,
            page: 0,
            dpi: DPI_DEFAULT,
            mode: Mode::Image,
            texture: None,
            texture_key: None,
            goto: None,
            reflow_pages: None,
            reflow_scroll: 0.0,
            status: None,
        })
    }

    /// Load `path` into the *current* window in place -- used by the `o`
    /// keybinding. Keeps the user's current zoom/mode preference rather than
    /// resetting to [`DPI_DEFAULT`]/[`Mode::Image`]; everything document-
    /// specific (page position, cached texture, extracted reflow text) is
    /// reset. On failure the current document keeps showing; the error goes
    /// to the status bar instead of losing the open document over a bad pick.
    fn open_path(&mut self, path: PathBuf) {
        match load_doc(&path) {
            Ok((doc, page_count)) => {
                self.doc = doc;
                self.path = path;
                self.page_count = page_count;
                self.page = 0;
                self.texture = None;
                self.texture_key = None;
                self.goto = None;
                self.reflow_pages = None;
                self.reflow_scroll = 0.0;
                self.status = None;
            }
            Err(e) => self.status = Some(e),
        }
    }

    /// Prompt with the native file picker and, if a file was chosen, open it
    /// in place via [`KpdfApp::open_path`].
    fn open_via_dialog(&mut self) {
        if let Some(path) = pick_pdf() {
            self.open_path(path);
        }
    }

    fn next_page(&mut self) {
        if self.page + 1 < self.page_count {
            self.page += 1;
        }
    }

    fn prev_page(&mut self) {
        self.page = self.page.saturating_sub(1);
    }

    fn goto_page_1based(&mut self, page_1based: usize) {
        let clamped = page_1based.clamp(1, self.page_count);
        self.page = clamped - 1;
    }

    fn zoom_in(&mut self) {
        self.dpi = (self.dpi + DPI_STEP).min(DPI_MAX);
    }

    fn zoom_out(&mut self) {
        self.dpi = (self.dpi - DPI_STEP).max(DPI_MIN);
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::Reflow => Mode::Image,
            Mode::Image => Mode::Reflow,
        };
    }

    /// Rasterize the current page (if not already cached at this page/dpi)
    /// and upload it as a texture.
    fn ensure_texture(&mut self, ctx: &egui::Context) {
        let key = (self.page, self.dpi.to_bits());
        if self.texture_key == Some(key) && self.texture.is_some() {
            return;
        }
        match rasterize_page(&self.doc, self.page, self.dpi) {
            Ok(pix) => {
                let rgba = rgb_to_rgba(&pix);
                let image = egui::ColorImage::from_rgba_unmultiplied([pix.w as usize, pix.h as usize], &rgba);
                let handle = ctx.load_texture("kpdf-page", image, egui::TextureOptions::LINEAR);
                self.texture = Some(handle);
                self.texture_key = Some(key);
                self.status = None;
            }
            Err(e) => {
                self.texture = None;
                self.texture_key = None;
                self.status = Some(format!("rasterize page {}: {e}", self.page + 1));
            }
        }
    }

    /// Extract the whole document's text once, lazily, on first entry to
    /// reflow mode.
    fn ensure_reflow(&mut self) {
        if self.reflow_pages.is_some() {
            return;
        }
        self.reflow_pages = Some(match std::fs::read(&self.path) {
            Ok(bytes) => extract_mupdf_from_bytes(&bytes).map_err(|e| e.to_string()),
            Err(e) => Err(format!("{}: {e}", self.path.display())),
        });
    }

    fn handle_key(&mut self, ctx: &egui::Context) {
        let (events, ctrl_c) = ctx.input(|i| {
            let ctrl_c = i.modifiers.ctrl && i.key_pressed(egui::Key::C);
            (i.events.clone(), ctrl_c)
        });
        if ctrl_c {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        for event in &events {
            let egui::Event::Key { key, pressed: true, .. } = event else { continue };
            let key = *key;

            // While typing a goto-page number, keys feed that box -- same
            // `is_capturing_text` idea the terminal viewer uses to keep `q`
            // from quitting mid-entry.
            if self.goto.is_some() {
                match key {
                    egui::Key::Escape => self.goto = None,
                    egui::Key::Enter => {
                        let buf = self.goto.take().unwrap_or_default();
                        if let Ok(n) = buf.trim().parse::<usize>()
                            && n >= 1
                        {
                            self.goto_page_1based(n);
                        }
                    }
                    egui::Key::Backspace => {
                        if let Some(buf) = self.goto.as_mut() {
                            buf.pop();
                        }
                    }
                    _ => {
                        if let Some(d) = digit_char(key)
                            && let Some(buf) = self.goto.as_mut()
                        {
                            buf.push(d);
                        }
                    }
                }
                continue;
            }

            match key {
                egui::Key::Q | egui::Key::Escape => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    return;
                }
                egui::Key::Tab => self.toggle_mode(),
                egui::Key::O => self.open_via_dialog(),
                _ => match self.mode {
                    Mode::Image => self.handle_key_image(key),
                    Mode::Reflow => self.handle_key_reflow(key),
                },
            }
        }
    }

    fn handle_key_image(&mut self, key: egui::Key) {
        match key {
            egui::Key::R => self.mode = Mode::Reflow,
            egui::Key::N | egui::Key::ArrowRight | egui::Key::PageDown => self.next_page(),
            egui::Key::P | egui::Key::ArrowLeft | egui::Key::PageUp => self.prev_page(),
            egui::Key::Plus | egui::Key::Equals => self.zoom_in(),
            egui::Key::Minus => self.zoom_out(),
            egui::Key::G => self.goto = Some(String::new()),
            _ => {}
        }
    }

    fn handle_key_reflow(&mut self, key: egui::Key) {
        const LINE: f32 = 20.0;
        match key {
            egui::Key::I => self.mode = Mode::Image,
            egui::Key::J | egui::Key::ArrowDown => self.reflow_scroll += LINE,
            egui::Key::K | egui::Key::ArrowUp => self.reflow_scroll = (self.reflow_scroll - LINE).max(0.0),
            egui::Key::PageDown => self.reflow_scroll += LINE * 20.0,
            egui::Key::PageUp => self.reflow_scroll = (self.reflow_scroll - LINE * 20.0).max(0.0),
            _ => {}
        }
    }
}

/// Read and open `path` as a [`PdfDocument`], returning it with its page
/// count. Shared by [`KpdfApp::open`] (startup) and [`KpdfApp::open_path`]
/// (the `o` keybinding's in-place reload) so the two stay in sync.
fn load_doc(path: &std::path::Path) -> Result<(PdfDocument, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc = PdfDocument::open(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let page_count = doc.page_count();
    if page_count == 0 {
        return Err(format!("{}: no pages", path.display()));
    }
    Ok((doc, page_count))
}

/// The native "Open a PDF" file picker ([`rfd`] -- pure Rust to build: the
/// default `xdg-portal` + `wayland` features on Linux, native Win32/Cocoa
/// pickers on Windows/macOS). `None` if the user cancelled.
fn pick_pdf() -> Option<PathBuf> {
    rfd::FileDialog::new().add_filter("PDF", &["pdf"]).set_title("Open a PDF").pick_file()
}

fn digit_char(key: egui::Key) -> Option<char> {
    Some(match key {
        egui::Key::Num0 => '0',
        egui::Key::Num1 => '1',
        egui::Key::Num2 => '2',
        egui::Key::Num3 => '3',
        egui::Key::Num4 => '4',
        egui::Key::Num5 => '5',
        egui::Key::Num6 => '6',
        egui::Key::Num7 => '7',
        egui::Key::Num8 => '8',
        egui::Key::Num9 => '9',
        _ => return None,
    })
}

/// [`Pixmap`]'s DeviceRGB samples (`rasterize_page`'s output: `n = 3`,
/// `alpha = false`) to the RGBA8 bytes egui's `ColorImage` needs, appending an
/// opaque alpha byte per pixel. Also tolerates a grayscale or already-RGBA
/// pixmap, in case a caller ever hands this a different colourspace.
fn rgb_to_rgba(pix: &Pixmap) -> Vec<u8> {
    let n = pix.n as usize;
    let px_count = (pix.w as usize) * (pix.h as usize);
    let mut out = Vec::with_capacity(px_count * 4);
    for row in 0..pix.h as usize {
        let row_start = row * pix.stride;
        for col in 0..pix.w as usize {
            let i = row_start + col * n;
            let s = &pix.samples[i..i + n];
            match (n, pix.alpha) {
                (4, true) => out.extend_from_slice(s), // already RGBA
                (3, false) => {
                    out.extend_from_slice(s);
                    out.push(255);
                }
                (1, false) => out.extend_from_slice(&[s[0], s[0], s[0], 255]),
                _ => {
                    let r = s[0];
                    let g = s.get(1).copied().unwrap_or(r);
                    let b = s.get(2).copied().unwrap_or(r);
                    out.extend_from_slice(&[r, g, b, 255]);
                }
            }
        }
    }
    out
}

impl eframe::App for KpdfApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.handle_key(ui.ctx());

        egui::Panel::top("kpdf-status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(self.path.display().to_string());
                ui.separator();
                match self.mode {
                    Mode::Image => {
                        ui.label(format!("page {}/{}", self.page + 1, self.page_count));
                        ui.separator();
                        ui.label(format!("{:.0} dpi", self.dpi));
                        if let Some(buf) = &self.goto {
                            ui.separator();
                            ui.colored_label(egui::Color32::GOLD, format!("goto: {buf}"));
                        }
                    }
                    Mode::Reflow => {
                        ui.label("reflow");
                    }
                }
                if let Some(status) = &self.status {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_RED, status);
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| match self.mode {
            Mode::Image => {
                self.ensure_texture(ui.ctx());
                egui::ScrollArea::both().show(ui, |ui| {
                    if let Some(tex) = &self.texture {
                        ui.add(egui::Image::new(tex).shrink_to_fit());
                    }
                });
            }
            Mode::Reflow => {
                self.ensure_reflow();
                let scroll = self.reflow_scroll;
                egui::ScrollArea::vertical().vertical_scroll_offset(scroll).show(ui, |ui| {
                    match &self.reflow_pages {
                        Some(Ok(pages)) => {
                            if let Some(p) = pages.get(self.page) {
                                for span in &p.spans {
                                    ui.label(&span.text);
                                }
                            }
                        }
                        Some(Err(e)) => {
                            ui.colored_label(egui::Color32::LIGHT_RED, e.as_str());
                        }
                        None => {
                            ui.label("extracting...");
                        }
                    }
                });
            }
        });

        // Repaint continuously enough to notice key events promptly even
        // when nothing else is animating.
        ui.ctx().request_repaint_after(Duration::from_millis(50));
    }
}

fn main() -> eframe::Result {
    // No path argument: prompt with the native file picker instead of
    // requiring one up front. Cancelling it exits quietly, not an error.
    let path = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => match pick_pdf() {
            Some(p) => p,
            None => return Ok(()),
        },
    };

    let app = match KpdfApp::open(path) {
        Ok(app) => app,
        Err(e) => {
            eprintln!("kpdf: {e}");
            std::process::exit(1);
        }
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 1100.0]),
        ..Default::default()
    };
    eframe::run_native("kpdf", native_options, Box::new(|_cc| Ok(Box::new(app))))
}
