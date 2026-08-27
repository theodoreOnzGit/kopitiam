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
//! * `+` / `-` -- zoom in / out (render dpi). Zoom is a real on-screen
//!   concept, not just a sharpness knob: raising dpi both re-rasterizes at
//!   higher resolution ([`KpdfApp::ensure_texture`]) and grows the page's
//!   displayed size ([`page_display_size`]), so the page can exceed the
//!   window and is panned via a bidirectional `ScrollArea`. At
//!   [`DPI_DEFAULT`] the page is fitted to the window exactly as before.
//!   The same range/step is also reachable on-screen: the status bar
//!   carries `-` / percentage / `+` zoom buttons (the percentage button
//!   doubles as a reset to 100%), and each button disables itself at
//!   [`DPI_MIN`]/[`DPI_MAX`] instead of clicking uselessly. Zooming
//!   re-centres the scroll position on whatever was in the middle of the
//!   viewport ([`recentred_scroll_offset`]) rather than jumping to a random
//!   offset on the resized page. (No fit-to-width/fit-to-page control yet --
//!   that is a distinct "continuously re-fit dpi to the window" mode, a
//!   larger follow-up, not a small addition on top of this.)
//! * Ctrl+scroll-wheel over the page -- zoom in/out one step at a time, the
//!   gesture every other image/PDF viewer binds. Plain scrolling (no Ctrl)
//!   still scrolls/pans the page as usual.
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
use kopitiam_pdf::mupdf::{PdfDocument, Pixmap, rasterize_page};
use kopitiam_pdf::{Page as TextPage, extract_mupdf_from_bytes};

// Same render-dpi range/step as `kopitiam view`
// (apps/cli/src/tui/viewer.rs's DPI_DEFAULT/MIN/MAX/STEP), kept in parity on
// purpose -- there is no reason the two viewers should zoom differently.
const DPI_DEFAULT: f32 = 150.0;
const DPI_MIN: f32 = 50.0;
const DPI_MAX: f32 = 600.0;
const DPI_STEP: f32 = 25.0;

/// One [`DPI_STEP`] zoom notch's worth of accumulated Ctrl+scroll input, in
/// the same units as `egui::InputState::zoom_delta()` minus one (so `0.0`
/// means "no zoom this frame", positive means zooming in, negative zooming
/// out). Mouse wheels and trackpads report a Ctrl+scroll gesture as many
/// small per-frame nudges to `zoom_delta()` rather than one clean value, so
/// instead of moving `dpi` by a fraction of a step on every such frame --
/// which would re-rasterize the page on every one of those frames, since
/// `ensure_texture` is keyed on `(page, dpi)` -- the raw signal accumulates
/// in [`KpdfApp::scroll_zoom_accum`] and is only converted into a whole
/// [`DPI_STEP`] move once enough of it has piled up. See
/// `zoom_steps_from_zoom_delta` and its tests for the exact coalescing
/// behaviour. The threshold itself is a reasonable default, not a value
/// derived from measuring a real wheel -- there is no display available to
/// tune it against actual hardware in this environment, so a human dogfooder
/// eyeballing "does one flick of the wheel feel like the right number of
/// steps?" is exactly the kind of check this file's module docs flag as
/// unverifiable here.
const ZOOM_DELTA_PER_STEP: f32 = 0.05;

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
    /// Leftover Ctrl+scroll signal not yet large enough to make a whole
    /// [`DPI_STEP`] move -- see [`ZOOM_DELTA_PER_STEP`] and
    /// `zoom_steps_from_zoom_delta`. Persists across frames so a slow or
    /// interrupted scroll gesture still adds up correctly instead of being
    /// discarded on every repaint.
    scroll_zoom_accum: f32,
    /// dpi as of the last time the page's on-screen display size was laid
    /// out (see the `Mode::Image` arm of `eframe::App::ui`). Compared
    /// against the live `dpi` each frame to detect "a zoom just happened",
    /// which is the trigger for re-centring `page_scroll_offset` via
    /// [`recentred_scroll_offset`] instead of leaving the raw pixel offset
    /// pointing at a different part of the page after it resizes.
    prev_dpi: f32,
    /// The page image's on-screen size (in egui ui points) as of the last
    /// frame -- i.e. the last `(w, h)` passed to `fit_to_exact_size`. Needed
    /// alongside `page_scroll_offset`/`page_viewport_size` to compute the
    /// fraction of the page that was centred under the viewport before a
    /// zoom change, so that fraction can be re-centred after.
    page_content_size: egui::Vec2,
    /// The `ScrollArea`'s viewport size (visible area, excluding scrollbars)
    /// as of the last frame.
    page_viewport_size: egui::Vec2,
    /// The `ScrollArea`'s scroll offset as of the last frame (positive =
    /// scrolled down/right). Egui persists this internally too, but kpdf
    /// needs its own copy to compute a *new* offset when dpi changes -- see
    /// [`recentred_scroll_offset`].
    page_scroll_offset: egui::Vec2,
    /// Lazily extracted once, on first entry to reflow mode -- extraction
    /// walks the whole document, so it is not worth doing eagerly for a
    /// session that might only ever use image mode.
    reflow_pages: Option<Result<Vec<TextPage>, String>>,
    reflow_scroll: f32,
    status: Option<String>,
    /// How many drawable annotations the current page carries -- shown in the
    /// status bar so a human can tell "this page has none" from "this page has
    /// some and we failed to draw them". Recomputed whenever the page changes.
    annot_count: usize,
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
            scroll_zoom_accum: 0.0,
            prev_dpi: DPI_DEFAULT,
            page_content_size: egui::Vec2::ZERO,
            page_viewport_size: egui::Vec2::ZERO,
            page_scroll_offset: egui::Vec2::ZERO,
            reflow_pages: None,
            reflow_scroll: 0.0,
            status: None,
            annot_count: 0,
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
                // A new document has a different page size, so last frame's
                // layout numbers no longer describe anything meaningful --
                // reset rather than let a stale content/viewport size feed
                // into the next zoom's re-centring math.
                self.page_content_size = egui::Vec2::ZERO;
                self.page_viewport_size = egui::Vec2::ZERO;
                self.page_scroll_offset = egui::Vec2::ZERO;
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

    /// Reset zoom to [`DPI_DEFAULT`] -- the on-screen zoom readout doubles as
    /// this button, since clicking "100%" to get back to 100% is the
    /// discoverable affordance every other viewer offers.
    fn zoom_reset(&mut self) {
        self.dpi = DPI_DEFAULT;
    }

    /// Ctrl+scroll-wheel zoom over the page, the gesture users reach for
    /// reflexively because every other image/PDF viewer binds it. Image mode
    /// only, and only outside a `goto` entry (same guard the `+`/`-` keys
    /// get in [`KpdfApp::handle_key_image`] -- there's no reason a scroll
    /// gesture should behave differently from a keypress while a page number
    /// is being typed).
    ///
    /// Reads egui's own `zoom_delta()`, which is already exactly `1.0` (a
    /// no-op) whenever the zoom modifier (`Modifiers::COMMAND`, which is
    /// Ctrl on Linux/Windows and Cmd on macOS) is not held -- so plain
    /// scrolling of the page is completely unaffected, and egui itself
    /// already zeroes the ordinary scroll delta for that frame whenever the
    /// zoom modifier *is* held (see `egui::InputState::begin_pass`), so the
    /// page's `ScrollArea` never sees a competing scroll signal to fight
    /// over -- no manual event-consuming needed here.
    fn handle_scroll_zoom(&mut self, ctx: &egui::Context) {
        if self.mode != Mode::Image || self.goto.is_some() {
            return;
        }
        let zoom_delta = ctx.input(|i| i.zoom_delta());
        if zoom_delta == 1.0 {
            return;
        }
        match zoom_steps_from_zoom_delta(zoom_delta, &mut self.scroll_zoom_accum) {
            steps if steps > 0 => (0..steps).for_each(|_| self.zoom_in()),
            steps if steps < 0 => (0..-steps).for_each(|_| self.zoom_out()),
            _ => {}
        }
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
                let image = egui::ColorImage::from_rgba_unmultiplied(
                    [pix.w as usize, pix.h as usize],
                    &rgba,
                );
                let handle = ctx.load_texture("kpdf-page", image, egui::TextureOptions::LINEAR);
                self.texture = Some(handle);
                self.texture_key = Some(key);
                self.annot_count = drawable_annot_count(&self.doc, self.page);
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
            let egui::Event::Key {
                key, pressed: true, ..
            } = event
            else {
                continue;
            };
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
            egui::Key::K | egui::Key::ArrowUp => {
                self.reflow_scroll = (self.reflow_scroll - LINE).max(0.0)
            }
            egui::Key::PageDown => self.reflow_scroll += LINE * 20.0,
            egui::Key::PageUp => self.reflow_scroll = (self.reflow_scroll - LINE * 20.0).max(0.0),
            _ => {}
        }
    }
}

/// Count the annotations on `page_index` that a viewer is expected to draw.
///
/// Mirrors the skip rules the renderer itself applies (see
/// `kopitiam_pdf::mupdf::annot_run`): `/Popup` subtypes are never drawn inline,
/// and `/F` bit 2 (Hidden) or bit 6 (NoView) means "do not display" per PDF
/// 32000-1:2008 table 165. Anything else counts, whether or not it carries an
/// `/AP` -- an annotation stored as pure data still has an appearance
/// synthesised for it, so it is genuinely expected on screen.
fn drawable_annot_count(doc: &PdfDocument, page_index: usize) -> usize {
    let Ok(page) = doc.page(page_index) else {
        return 0;
    };
    let Ok(annots) = doc.resolve_get(page, "Annots") else {
        return 0;
    };
    (0..annots.array_len())
        .filter_map(|i| annots.array_get(i))
        .filter_map(|entry| doc.resolve(entry).ok())
        .filter(|annot| annot.is_dict())
        .filter(|annot| {
            doc.resolve_get(annot, "Subtype")
                .map(|st| st.to_name() != b"Popup")
                .unwrap_or(false)
        })
        .filter(|annot| {
            let flags = doc.resolve_get(annot, "F").map(|o| o.to_int()).unwrap_or(0);
            flags & 2 == 0 && flags & 32 == 0
        })
        .count()
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
    rfd::FileDialog::new()
        .add_filter("PDF", &["pdf"])
        .set_title("Open a PDF")
        .pick_file()
}

/// Render `dpi` as a percentage of [`DPI_DEFAULT`] -- "150%" is what a
/// reader acts on; the raw "108 dpi" the rasterizer actually uses is not.
/// Rounds to the nearest whole percent.
fn zoom_percent(dpi: f32) -> i32 {
    ((dpi / DPI_DEFAULT) * 100.0).round() as i32
}

/// Fold one frame's `zoom_delta` (egui's Ctrl+scroll / pinch-zoom signal,
/// where `1.0` means "no change this frame") into whole [`DPI_STEP`] zoom
/// steps, carrying any leftover in `accum` for the next frame -- see
/// [`ZOOM_DELTA_PER_STEP`] for why this accumulates rather than acting on
/// every frame's raw value directly. Positive steps mean zoom in (matching
/// [`KpdfApp::zoom_in`]'s direction), negative mean zoom out, `0` means "not
/// enough has accumulated yet".
///
/// Pure and independent of egui's `Context`/`InputState` and of `KpdfApp`,
/// so it is unit-tested directly without a display -- see the `tests`
/// module below.
fn zoom_steps_from_zoom_delta(zoom_delta: f32, accum: &mut f32) -> i32 {
    *accum += zoom_delta - 1.0;
    let steps = (*accum / ZOOM_DELTA_PER_STEP).trunc();
    *accum -= steps * ZOOM_DELTA_PER_STEP;
    steps as i32
}

/// Compute the on-screen size (in egui ui points) to draw the current page
/// texture at, given the dpi it was rasterized at.
///
/// # The bug this replaces
///
/// The obvious approach -- `egui::Image::new(tex).shrink_to_fit()` -- always
/// scales the texture to exactly fill the available panel, preserving
/// aspect ratio (`egui`'s own `ImageFit::Fraction` resolves to
/// `scale_to_fit(image_source_size, available_size, ..)`, i.e. `ratio =
/// min(available.x / tex.x, available.y / tex.y)`, computed straight from
/// the *current* texture's pixel size). Since [`KpdfApp::ensure_texture`]
/// re-rasterizes at a higher pixel density every time `dpi` rises, that
/// fit ratio shrinks by almost exactly the amount `dpi` grew by -- the two
/// cancel out, the page gets sharper but never bigger, and "zoom" is
/// silently a no-op on screen. This was true of the `+`/`-` keys from day
/// one; on-screen buttons only made it obvious.
///
/// # The fix
///
/// 1. Undo dpi's contribution to the texture's pixel size, recovering
///    `base` -- what the texture's pixel size would be at [`DPI_DEFAULT`].
///    This is dpi-invariant: it depends only on the page's own physical
///    dimensions, which don't change with dpi.
/// 2. Contain-fit `base` into `available` the same way `shrink_to_fit`
///    would (preserve aspect ratio, scale by
///    `min(available.w/base.w, available.h/base.h)`) -- this reproduces
///    today's on-screen size exactly when `dpi == DPI_DEFAULT`, per the
///    task's "at DPI_DEFAULT the page should look the way it does today"
///    requirement.
/// 3. Re-apply the zoom factor (`dpi / DPI_DEFAULT`) on top of that fitted
///    size, so raising dpi now visibly grows the displayed page instead of
///    cancelling out against step 2's shrink.
///
/// Returns `(0.0, 0.0)` for degenerate input (a non-positive texture
/// dimension or dpi) rather than dividing by zero / propagating a NaN into
/// egui's layout.
///
/// Pure arithmetic over plain `f32`s -- no egui types -- so it is
/// unit-tested directly; see the `tests` module below.
fn page_display_size(tex_w: f32, tex_h: f32, dpi: f32, available_w: f32, available_h: f32) -> (f32, f32) {
    if tex_w <= 0.0 || tex_h <= 0.0 || dpi <= 0.0 {
        return (0.0, 0.0);
    }
    let zoom = dpi / DPI_DEFAULT;
    let base_w = tex_w / zoom;
    let base_h = tex_h / zoom;
    let ratio = (available_w / base_w).min(available_h / base_h);
    let ratio = if ratio.is_finite() { ratio } else { 1.0 };
    (base_w * ratio * zoom, base_h * ratio * zoom)
}

/// After a zoom changes the page's displayed size, compute the scroll
/// offset (along one axis) that keeps whatever content point was centred in
/// the viewport *before* the resize still centred *after* it -- rather than
/// leaving the same raw pixel offset pointing at an arbitrary spot on the
/// now differently-sized page, which reads as the page randomly jumping.
///
/// Works per-axis (plain `f32`s, not `egui::Vec2`) so the same function
/// covers both width and height and is trivial to unit-test; the caller
/// (the `Mode::Image` arm of `eframe::App::ui`) applies it twice.
///
/// `content_size` and `viewport_size` describe the layout *before* the
/// resize (i.e. as of the last frame); `new_content_size` is the page's
/// freshly-computed [`page_display_size`] for this frame. Anchors to the
/// viewport centre rather than the cursor position -- simpler, and "nicer
/// but do not over-engineer it" was an explicit non-goal for the cursor
/// variant.
fn recentred_scroll_offset(offset: f32, content_size: f32, viewport_size: f32, new_content_size: f32) -> f32 {
    if content_size <= 0.0 {
        return 0.0;
    }
    // Fraction of the old content that was centred under the viewport
    // (0.0 = the content's top/left edge is centred, 1.0 = its
    // bottom/right edge is) -- clamped because a viewport bigger than the
    // content (nothing to scroll) would otherwise push this outside [0, 1].
    let center_fraction = ((offset + viewport_size / 2.0) / content_size).clamp(0.0, 1.0);
    (center_fraction * new_content_size - viewport_size / 2.0).max(0.0)
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
        self.handle_scroll_zoom(ui.ctx());

        egui::Panel::top("kpdf-status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(self.path.display().to_string());
                ui.separator();
                match self.mode {
                    Mode::Image => {
                        ui.label(format!("page {}/{}", self.page + 1, self.page_count));
                        ui.separator();

                        // On-screen zoom controls, wired to the exact same
                        // zoom_in/zoom_out/zoom_reset methods the `+`/`-`
                        // keys and Ctrl+scroll use -- one place owns the
                        // DPI_STEP clamping logic. Each arrow button disables
                        // itself at the DPI_MIN/DPI_MAX limit rather than
                        // sitting there clickable-but-useless.
                        if ui
                            .add_enabled(self.dpi > DPI_MIN, egui::Button::new("-"))
                            .on_hover_text("Zoom out (-)")
                            .clicked()
                        {
                            self.zoom_out();
                        }
                        if ui
                            .button(format!("{}%", zoom_percent(self.dpi)))
                            .on_hover_text("Reset zoom to 100%")
                            .clicked()
                        {
                            self.zoom_reset();
                        }
                        if ui
                            .add_enabled(self.dpi < DPI_MAX, egui::Button::new("+"))
                            .on_hover_text("Zoom in (+)")
                            .clicked()
                        {
                            self.zoom_in();
                        }
                        ui.label(format!("({:.0} dpi)", self.dpi));
                        ui.separator();
                        // Annotation count for the current page. kpdf renders
                        // annotations via `rasterize_page` like everything else,
                        // but a GUI cannot be verified headlessly -- so showing
                        // the count turns "are annotations working?" into
                        // something a human can check at a glance: if this says
                        // `3 annots` and the page looks bare, that is a bug
                        // worth reporting, and the number tells you it is a
                        // *rendering* bug rather than a parsing one.
                        let n = self.annot_count;
                        if n > 0 {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::LIGHT_BLUE,
                                format!("{n} annot{}", if n == 1 { "" } else { "s" }),
                            );
                        }
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
                // The viewport available to the page *before* the
                // ScrollArea claims it -- this, not the texture's own pixel
                // size, is what "fit to container at DPI_DEFAULT" is
                // relative to. See `page_display_size`'s docs for why using
                // the texture's current-dpi size directly (what
                // `Image::shrink_to_fit` used to do here) made zoom a no-op.
                let available = ui.available_size();
                if let Some(tex) = &self.texture {
                    let [tw, th] = tex.size();
                    let new_size = {
                        let (w, h) = page_display_size(tw as f32, th as f32, self.dpi, available.x, available.y);
                        egui::vec2(w, h)
                    };

                    // A zoom just happened (dpi differs from the value the
                    // last layout used) -- re-centre the scroll offset on
                    // whatever content point was in the middle of the
                    // viewport before, rather than let the same raw pixel
                    // offset now point at an arbitrary spot on the resized
                    // page (or leave it at the previous frame's clamp, which
                    // reads as "zoom did nothing" all over again).
                    let mut scroll_area = egui::ScrollArea::both();
                    if self.dpi != self.prev_dpi {
                        let recentred = egui::vec2(
                            recentred_scroll_offset(
                                self.page_scroll_offset.x,
                                self.page_content_size.x,
                                self.page_viewport_size.x,
                                new_size.x,
                            ),
                            recentred_scroll_offset(
                                self.page_scroll_offset.y,
                                self.page_content_size.y,
                                self.page_viewport_size.y,
                                new_size.y,
                            ),
                        );
                        scroll_area = scroll_area.scroll_offset(recentred);
                    }

                    let output = scroll_area.show(ui, |ui| {
                        ui.add(egui::Image::new(tex).fit_to_exact_size(new_size));
                    });

                    self.page_scroll_offset = output.state.offset;
                    self.page_content_size = new_size;
                    self.page_viewport_size = output.inner_rect.size();
                    self.prev_dpi = self.dpi;
                }
            }
            Mode::Reflow => {
                self.ensure_reflow();
                let scroll = self.reflow_scroll;
                egui::ScrollArea::vertical()
                    .vertical_scroll_offset(scroll)
                    .show(ui, |ui| match &self.reflow_pages {
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- zoom_percent -------------------------------------------------

    #[test]
    fn zoom_percent_at_default_is_100() {
        assert_eq!(zoom_percent(DPI_DEFAULT), 100);
    }

    #[test]
    fn zoom_percent_scales_with_dpi() {
        assert_eq!(zoom_percent(DPI_DEFAULT * 2.0), 200);
        assert_eq!(zoom_percent(DPI_MIN), 33); // 50/150 = 33.33...%
        assert_eq!(zoom_percent(75.0), 50); // 75/150
    }

    #[test]
    fn zoom_percent_rounds_to_nearest_whole_percent() {
        // 151/150 = 100.666...% -> rounds to 101, not truncates to 100.
        assert_eq!(zoom_percent(151.0), 101);
    }

    // -- zoom_steps_from_zoom_delta -----------------------------------

    #[test]
    fn zoom_delta_of_one_is_a_no_op() {
        let mut accum = 0.0;
        assert_eq!(zoom_steps_from_zoom_delta(1.0, &mut accum), 0);
        assert_eq!(accum, 0.0);
    }

    #[test]
    fn small_zoom_deltas_accumulate_before_stepping() {
        // Each nudge below is well under ZOOM_DELTA_PER_STEP (0.05) on its
        // own -- this is the "many small per-frame events" case the
        // accumulator exists for. No single call should fire a step until
        // enough of them have piled up. The final nudge is deliberately
        // larger than a clean boundary crossing (0.04 + 0.02 = 0.06, not
        // exactly 0.05) so the assertion doesn't hinge on f32 rounding
        // landing on the exact millimetre of the threshold.
        let mut accum = 0.0;
        for _ in 0..4 {
            assert_eq!(zoom_steps_from_zoom_delta(1.01, &mut accum), 0);
        }
        assert_eq!(zoom_steps_from_zoom_delta(1.02, &mut accum), 1);
    }

    #[test]
    fn large_zoom_delta_can_fire_multiple_steps_in_one_call() {
        // A single big jump (e.g. a fast trackpad pinch reported in one
        // frame) should still convert to the right whole number of steps,
        // not just clamp to one.
        let mut accum = 0.0;
        assert_eq!(zoom_steps_from_zoom_delta(1.23, &mut accum), 4); // 0.23 / 0.05 = 4.6 -> 4
        assert!(accum > 0.0 && accum < ZOOM_DELTA_PER_STEP);
    }

    #[test]
    fn zoom_out_direction_is_negative() {
        // Same reasoning as the zoom-in case above: the last nudge is
        // chosen to land clearly past the threshold (-0.06, not exactly
        // -0.05) so the test doesn't depend on f32 rounding at the boundary.
        let mut accum = 0.0;
        for _ in 0..4 {
            assert_eq!(zoom_steps_from_zoom_delta(0.99, &mut accum), 0);
        }
        assert_eq!(zoom_steps_from_zoom_delta(0.98, &mut accum), -1);
    }

    #[test]
    fn leftover_accum_is_never_dropped() {
        // Repeatedly feeding a delta that is *not* an exact multiple of the
        // step threshold must not lose the remainder over many calls --
        // total steps taken should track total input applied. 100 * 0.0011
        // = 0.11 units, deliberately not a clean multiple of
        // ZOOM_DELTA_PER_STEP (0.05) so the expected step count (2, with
        // 0.01 left over) isn't sitting exactly on an f32 rounding boundary
        // the way an exact multiple would be.
        let mut accum = 0.0;
        let mut total_steps = 0;
        for _ in 0..100 {
            total_steps += zoom_steps_from_zoom_delta(1.0011, &mut accum);
        }
        assert_eq!(total_steps, 2);
        assert!(accum > 0.0 && accum < ZOOM_DELTA_PER_STEP);
    }

    #[test]
    fn accum_never_grows_unbounded_once_a_step_is_taken() {
        let mut accum = 0.0;
        zoom_steps_from_zoom_delta(1.23, &mut accum);
        assert!(accum.abs() < ZOOM_DELTA_PER_STEP);
    }

    // -- page_display_size ---------------------------------------------

    #[test]
    fn at_default_dpi_matches_plain_contain_fit() {
        // zoom == 1.0, so this must reduce to an ordinary "contain fit"
        // (preserve aspect ratio, shrink to the tighter axis) exactly like
        // egui's own `shrink_to_fit` -- this is the "look the way it does
        // today" requirement at DPI_DEFAULT.
        //
        // Portrait texture (1000x2000), wide-ish viewport (500x1500):
        // width ratio 500/1000=0.5, height ratio 1500/2000=0.75 -> width
        // is the tighter axis, so the result is (500, 1000).
        let (w, h) = page_display_size(1000.0, 2000.0, DPI_DEFAULT, 500.0, 1500.0);
        assert!((w - 500.0).abs() < 1e-3);
        assert!((h - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn doubling_dpi_doubles_the_displayed_size() {
        // A page that exactly fills a 900x1200 window at DPI_DEFAULT (i.e.
        // its DPI_DEFAULT-rasterized texture is already 900x1200, a 1:1
        // contain fit). Doubling dpi doubles the texture's pixel size to
        // 1800x2400 (what ensure_texture would actually produce); the
        // displayed size must double too, to 1800x2400, overflowing the
        // still-900x1200 window -- that overflow is exactly what makes the
        // page pannable, and what was missing before this fix (previously
        // it would still have come out as 900x1200, unchanged).
        let (w, h) = page_display_size(1800.0, 2400.0, DPI_DEFAULT * 2.0, 900.0, 1200.0);
        assert!((w - 1800.0).abs() < 1e-2);
        assert!((h - 2400.0).abs() < 1e-2);
    }

    #[test]
    fn halving_dpi_halves_the_displayed_size() {
        // Same fixture as above, but zooming out: half the pixel density,
        // half the resolution -- the shown page should also shrink to half
        // of the fitted 900x1200 baseline, not stay put.
        let (w, h) = page_display_size(450.0, 600.0, DPI_DEFAULT / 2.0, 900.0, 1200.0);
        assert!((w - 450.0).abs() < 1e-2);
        assert!((h - 600.0).abs() < 1e-2);
    }

    #[test]
    fn zoom_preserves_aspect_ratio_when_container_constrains_the_other_axis() {
        // Landscape texture (2000x1000) into a viewport where height, not
        // width, is the tighter constraint (available 3000x500): contain
        // fit at DPI_DEFAULT gives ratio = 500/1000 = 0.5 -> (1000, 500).
        // At 2x zoom both dimensions must double to (2000, 1000) --
        // doubling the *texture* pixel size to 4000x2000 first, per
        // ensure_texture's real behaviour.
        let (w, h) = page_display_size(4000.0, 2000.0, DPI_DEFAULT * 2.0, 3000.0, 500.0);
        assert!((w - 2000.0).abs() < 1e-2);
        assert!((h - 1000.0).abs() < 1e-2);
    }

    #[test]
    fn degenerate_input_returns_zero_not_nan() {
        assert_eq!(page_display_size(0.0, 100.0, DPI_DEFAULT, 500.0, 500.0), (0.0, 0.0));
        assert_eq!(page_display_size(100.0, 0.0, DPI_DEFAULT, 500.0, 500.0), (0.0, 0.0));
        assert_eq!(page_display_size(100.0, 100.0, 0.0, 500.0, 500.0), (0.0, 0.0));
        assert_eq!(page_display_size(100.0, 100.0, -10.0, 500.0, 500.0), (0.0, 0.0));
    }

    #[test]
    fn stays_finite_across_the_whole_dpi_range() {
        for dpi in [DPI_MIN, DPI_DEFAULT, DPI_MAX] {
            let (w, h) = page_display_size(1000.0, 1400.0, dpi, 900.0, 1100.0);
            assert!(w.is_finite() && w > 0.0);
            assert!(h.is_finite() && h > 0.0);
        }
    }

    // -- recentred_scroll_offset -----------------------------------------

    #[test]
    fn recentre_tracks_the_same_relative_content_point_after_doubling() {
        // Viewport 500pt wide, old content 1000pt wide, scrolled to the
        // very start (offset 0) -- the viewport's centre sits at content
        // position 250, i.e. 25% into the content. After the content
        // doubles to 2000pt, that same 25%-in point is at absolute
        // position 500; centring it means an offset of 500 - 250 = 250.
        let new_offset = recentred_scroll_offset(0.0, 1000.0, 500.0, 2000.0);
        assert!((new_offset - 250.0).abs() < 1e-3);
    }

    #[test]
    fn recentre_is_a_no_op_when_content_size_is_unchanged() {
        let old_offset = 137.0;
        let new_offset = recentred_scroll_offset(old_offset, 1000.0, 500.0, 1000.0);
        assert!((new_offset - old_offset).abs() < 1e-3);
    }

    #[test]
    fn recentre_never_goes_negative() {
        // Viewport bigger than the content -- nothing to scroll -- must
        // clamp to a sane (non-negative) offset instead of going negative.
        let new_offset = recentred_scroll_offset(0.0, 100.0, 400.0, 100.0);
        assert!(new_offset >= 0.0);
    }

    #[test]
    fn recentre_handles_zero_content_size_without_panicking() {
        assert_eq!(recentred_scroll_offset(0.0, 0.0, 500.0, 800.0), 0.0);
    }
}
