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
//!
//! # Annotation tools and forms mode
//!
//! Image mode also carries a small toolbar of on-screen buttons (not
//! keyboard-only -- a maintainer dogfooding on a fresh install should never
//! need to read this doc comment to discover them):
//!
//! * **Open** -- the same native file picker `o` already opened, now also a
//!   button, so a mouse-only session can reach it.
//! * **Undo** / **Redo** -- step through [`EditHistory`], disabled when there
//!   is nothing to step to.
//! * **Save** -- write the current (possibly edited) bytes to a file chosen
//!   via a native save dialog. Never silently overwrites the original --
//!   see [`KpdfApp::save_via_dialog`].
//! * **Pen** -- drag on the page to draw; releasing commits the stroke as a
//!   real ink annotation via [`kopitiam_pdf::mupdf::annot_edit::add_ink_annot`].
//! * **Eraser** -- click or drag over an existing annotation to delete it,
//!   via [`kopitiam_pdf::mupdf::annot_edit::delete_annot`].
//! * **Forms** -- offered only when the open document actually has an
//!   `/AcroForm` ([`kopitiam_pdf::mupdf::form::has_acroform`]). While on,
//!   clicking a checkbox/radio widget toggles it and clicking a text field
//!   opens a small popup to type a new value, mirroring Okular's forms
//!   toggle.
//!
//! All of this is **additive**: page nav, `goto`, reflow mode, the zoom
//! controls and the annotation counter behave exactly as before when none of
//! the new tools are engaged (the default [`Tool::Pan`] + forms mode off).
//!
//! ## The coordinate trap
//!
//! egui reports pointer positions in **screen space** (ui points, origin
//! top-left, y down). The library's annotation/form APIs
//! ([`InkStroke`], [`AnnotRef::rect`](kopitiam_pdf::mupdf::annot_edit::AnnotRef::rect),
//! [`FormField::rect`](kopitiam_pdf::mupdf::form::FormField::rect)) all speak
//! **default user space** (PDF points, origin bottom-left, y **up**). Getting
//! the y-flip or the dpi/zoom unscaling wrong here puts ink in the wrong
//! place, or mirrored -- and it would look plausible enough to ship without a
//! careful eye. [`screen_to_page`]/[`page_to_screen`]/[`page_size_pts`]
//! isolate exactly that seam as pure, unit-tested functions (see the `tests`
//! module) so the mapping can be checked without a display.
//!
//! ## What is genuinely unverified here
//!
//! There is no display in this environment. Compiling, clippy, and the pure
//! coordinate-math/state-machine/hit-testing unit tests are all that has
//! actually been checked. Whether the pen visually tracks the cursor, whether
//! button layout/spacing reads well, whether click-to-erase "feels" precise,
//! and whether the forms popup is positioned sensibly are all **unverified
//! and need a human at a real display** -- per this project's GUI-surfaces
//! rule, that dogfooding is not something an agent can fake.

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;
use kopitiam_pdf::mupdf::annot_edit::{AnnotRef, EditHistory, InkAnnotSpec, InkStroke};
use kopitiam_pdf::mupdf::form::{FieldKind, FormField};
use kopitiam_pdf::mupdf::{PdfDocument, Pixmap, Rect, rasterize_page};
use kopitiam_pdf::{Page as TextPage, extract_mupdf_from_bytes};

/// Ink color for a newly-drawn stroke (DeviceRGB, 0..=1) -- plain black, the
/// least surprising default for a "draw on the page" tool. No UI to pick a
/// colour yet; that is a follow-up, not scope creep for this pass.
const INK_COLOR: [f32; 3] = [0.0, 0.0, 0.0];
/// Stroke width in PDF points, written via the annotation's `/Border`.
const INK_WIDTH: f32 = 2.0;
/// Constant opacity (`/CA`); 1.0 means fully opaque and writes no `/CA` at
/// all (see [`InkAnnotSpec::opacity`](kopitiam_pdf::mupdf::annot_edit::InkAnnotSpec::opacity)).
const INK_OPACITY: f32 = 1.0;

/// Which page-editing tool on-drag/on-click input is currently routed to.
/// Mutually exclusive with forms mode -- see [`select_tool`] and
/// [`toggle_forms_mode`] for why, and for the pure transition logic that
/// enforces it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    /// Default: clicks/drags do nothing special (the page just scrolls).
    Pan,
    /// Drag to draw an ink stroke; released on drag-stop.
    Draw,
    /// Click or drag over an annotation to delete it.
    Erase,
}

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
    /// The active annotation tool (image mode only). See [`Tool`].
    tool: Tool,
    /// Whether form fields on the current page are interactive (Okular-style
    /// toggle). Only ever `true` when [`KpdfApp::has_acroform`] is also
    /// `true` -- see [`toggle_forms_mode`].
    forms_mode: bool,
    /// Whether the open document has an `/AcroForm` at all
    /// ([`kopitiam_pdf::mupdf::form::has_acroform`]), computed once per
    /// document load. Gates whether the Forms button is shown -- there is no
    /// point offering a forms toggle for a document with no fields.
    has_acroform: bool,
    /// Edit history over the document's raw bytes, backing Undo/Redo/Save.
    /// `None` until the first edit is made in this session -- a document that
    /// is only ever viewed, never annotated, never pays for one (see
    /// [`KpdfApp::ensure_edit_history`]).
    edit_history: Option<EditHistory>,
    /// The in-progress ink stroke while [`Tool::Draw`] is being dragged, in
    /// **default user space** (already converted via [`screen_to_page`] on
    /// every drag frame) -- committed as a real annotation on drag-release
    /// (see [`KpdfApp::handle_draw`]) and cleared either way.
    draw_stroke: Vec<(f32, f32)>,
    /// `Some((obj_num, buf))` while the small "edit this text field" popup is
    /// open: `obj_num` identifies which [`FormField`] (re-looked-up by
    /// [`kopitiam_pdf::mupdf::form::page_form_fields`] each frame, since a
    /// `FormField` itself is not [`Clone`]), `buf` is the text being typed.
    form_edit: Option<(i32, String)>,
}

impl KpdfApp {
    fn open(path: PathBuf) -> Result<KpdfApp, String> {
        let (doc, page_count) = load_doc(&path)?;
        // See the module docs' "Forms" bullet: computed once per document
        // load, not per frame -- it is a property of the document's
        // `/AcroForm`, which ink/eraser edits never touch.
        let has_acroform = kopitiam_pdf::mupdf::form::has_acroform(&doc);
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
            tool: Tool::Pan,
            forms_mode: false,
            has_acroform,
            edit_history: None,
            draw_stroke: Vec::new(),
            form_edit: None,
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
                self.has_acroform = kopitiam_pdf::mupdf::form::has_acroform(&doc);
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
                // A different document is a different byte history -- the
                // old one's undo/redo stack describes edits to a file that
                // is no longer open. Forms mode is reset to off rather than
                // carried over stale, since the new document might not even
                // have a forms button to have turned it back on with.
                self.edit_history = None;
                self.draw_stroke.clear();
                self.form_edit = None;
                self.forms_mode = false;
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

    /// Switch tools, via the pure [`select_tool`] transition.
    fn select_tool(&mut self, requested: Tool) {
        select_tool(&mut self.tool, &mut self.forms_mode, requested);
    }

    /// Flip forms mode, via the pure [`toggle_forms_mode`] transition.
    fn toggle_forms_mode(&mut self) {
        toggle_forms_mode(&mut self.tool, &mut self.forms_mode);
    }

    /// Get (creating on first use) the edit history backing Undo/Redo/Save.
    ///
    /// Starts from [`PdfDocument::raw_bytes`] -- the *current in-memory*
    /// document's bytes, not a fresh read of `self.path` off disk -- so it is
    /// correct even if the file on disk has changed since this session
    /// opened it.
    fn ensure_edit_history(&mut self) -> &mut EditHistory {
        if self.edit_history.is_none() {
            self.edit_history = Some(EditHistory::new(self.doc.raw_bytes().to_vec()));
        }
        self.edit_history.as_mut().expect("just inserted above")
    }

    /// Common tail of every annotation/form edit: on success, push the new
    /// bytes into [`EditHistory`], reopen the document from them, and
    /// invalidate everything derived from the old one (texture, annotation
    /// count). On failure, report it in the status bar rather than losing
    /// the edit silently -- the document keeps showing its pre-edit state.
    ///
    /// This is the *only* place [`PdfDocument::open`] is called after
    /// startup, which is what keeps "reopen after every edit" (per the
    /// library's edit model -- see the module docs) from being duplicated at
    /// every call site.
    fn apply_edit(&mut self, result: kopitiam_pdf::mupdf::Result<Vec<u8>>) {
        match result {
            Ok(bytes) => {
                self.ensure_edit_history().push(bytes);
                let current = self
                    .edit_history
                    .as_ref()
                    .expect("just ensured")
                    .current()
                    .to_vec();
                self.reload_from_bytes(current);
            }
            Err(e) => self.status = Some(e.to_string()),
        }
    }

    /// Reopen `self.doc` from `bytes` (a state pulled from [`EditHistory`],
    /// whether from a fresh edit, an undo, or a redo) and invalidate every
    /// cache keyed on the old document. Page count/position are left alone:
    /// every edit this app makes (ink add, annotation delete, form-field
    /// write) preserves the page count and page ordering.
    fn reload_from_bytes(&mut self, bytes: Vec<u8>) {
        match PdfDocument::open(bytes) {
            Ok(doc) => {
                self.doc = doc;
                self.texture = None;
                self.texture_key = None;
                self.annot_count = drawable_annot_count(&self.doc, self.page);
                self.status = None;
            }
            Err(e) => self.status = Some(format!("reload after edit: {e}")),
        }
    }

    /// Step back one edit. Disabled (see the Undo button) when there is
    /// nothing to step back to.
    fn undo(&mut self) {
        let Some(hist) = &mut self.edit_history else {
            return;
        };
        if let Some(bytes) = hist.undo() {
            let bytes = bytes.to_vec();
            self.reload_from_bytes(bytes);
        }
    }

    /// Step forward one edit. Disabled (see the Redo button) when there is
    /// nothing to step forward to.
    fn redo(&mut self) {
        let Some(hist) = &mut self.edit_history else {
            return;
        };
        if let Some(bytes) = hist.redo() {
            let bytes = bytes.to_vec();
            self.reload_from_bytes(bytes);
        }
    }

    /// Write the current (edited) bytes to a file the user picks via a
    /// native save dialog. Defaults the suggested filename to the currently
    /// open document's, but never writes anywhere without the user
    /// confirming a destination through the dialog -- cancelling leaves the
    /// original file on disk untouched, same as cancelling Open does.
    fn save_via_dialog(&mut self) {
        let Some(hist) = &self.edit_history else {
            return;
        };
        let default_name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "document.pdf".to_string());
        let Some(target) = rfd::FileDialog::new()
            .add_filter("PDF", &["pdf"])
            .set_file_name(&default_name)
            .set_title("Save PDF as")
            .save_file()
        else {
            return;
        };
        match std::fs::write(&target, hist.current()) {
            Ok(()) => self.status = Some(format!("saved {}", target.display())),
            Err(e) => self.status = Some(format!("save {}: {e}", target.display())),
        }
    }

    /// Forms-mode click handling: hit-test against the current page's form
    /// fields and either toggle a checkbox/radio in place or open the
    /// text-field editor popup. All PDF-structure knowledge (what `/AS`,
    /// `/V`, an on-state name are) stays in `kopitiam_pdf::mupdf::form` --
    /// this only converts a click to page coordinates and dispatches.
    fn handle_forms_click(&mut self, response: &egui::Response, layout: PageLayout) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let (px, py) = screen_to_page(pos.x, pos.y, layout);
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, self.page);
        let Some(idx) = hit_test_field(px, py, &fields) else {
            return;
        };
        let field = &fields[idx];
        if field.read_only {
            self.status = Some(format!("{}: read-only field", field.name));
            return;
        }
        match field.kind {
            FieldKind::Checkbox | FieldKind::Radio => {
                let result = kopitiam_pdf::mupdf::form::toggle_checkbox(&self.doc, field);
                self.apply_edit(result);
            }
            FieldKind::Text => {
                self.form_edit = Some((field.obj_num, field.value.clone()));
            }
            _ => self.status = Some(format!("{}: unsupported field kind for kpdf", field.name)),
        }
    }

    /// Pen-tool drag handling: accumulate the drag into [`KpdfApp::draw_stroke`]
    /// (already converted to page space so the stroke is correct regardless
    /// of the current zoom), paint a live preview so the tool has visible
    /// feedback while dragging, and commit it as a real ink annotation on
    /// release via [`kopitiam_pdf::mupdf::annot_edit::add_ink_annot`].
    fn handle_draw(&mut self, response: &egui::Response, layout: PageLayout, ui: &egui::Ui) {
        if response.dragged()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let point = screen_to_page(pos.x, pos.y, layout);
            self.draw_stroke.push(point);
        }

        if self.draw_stroke.len() >= 2 {
            let screen_pts: Vec<egui::Pos2> = self
                .draw_stroke
                .iter()
                .map(|&(x, y)| {
                    let (sx, sy) = page_to_screen(x, y, layout);
                    egui::pos2(sx, sy)
                })
                .collect();
            // Deliberately a different colour from committed ink (INK_COLOR,
            // black) so a preview-in-progress is never mistaken for an
            // already-saved stroke.
            ui.painter().line(
                screen_pts,
                egui::Stroke::new(INK_WIDTH, egui::Color32::from_rgb(220, 40, 40)),
            );
        }

        if response.drag_stopped() && !self.draw_stroke.is_empty() {
            let points = std::mem::take(&mut self.draw_stroke);
            let spec = InkAnnotSpec {
                page_index: self.page,
                strokes: vec![InkStroke { points }],
                color: INK_COLOR,
                width: INK_WIDTH,
                opacity: INK_OPACITY,
                author: None,
            };
            let result = kopitiam_pdf::mupdf::annot_edit::add_ink_annot(&self.doc, &spec);
            self.apply_edit(result);
        }
    }

    /// Eraser-tool click/drag handling: hit-test the current page's
    /// annotations and delete whichever one the pointer lands on, via
    /// [`kopitiam_pdf::mupdf::annot_edit::delete_annot`].
    fn handle_erase(&mut self, response: &egui::Response, layout: PageLayout) {
        if !(response.clicked() || response.dragged()) {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let (px, py) = screen_to_page(pos.x, pos.y, layout);
        let refs = kopitiam_pdf::mupdf::annot_edit::page_annot_refs(&self.doc, self.page);
        let Some(num) = hit_test_annot(px, py, &refs) else {
            return;
        };
        let result = kopitiam_pdf::mupdf::annot_edit::delete_annot(&self.doc, self.page, num);
        self.apply_edit(result);
    }

    /// Draw the small "edit this text field" popup while
    /// [`KpdfApp::form_edit`] is `Some`, and handle its Set/Cancel/Enter.
    ///
    /// Re-fetches the page's form fields every frame it's open (rather than
    /// caching a `FormField`, which is not `Clone`) and matches by
    /// `obj_num` -- cheap for a form-sized field list, and always
    /// consistent with whatever the document currently is.
    fn show_form_edit_popup(&mut self, ctx: &egui::Context) {
        let Some((obj_num, _)) = self.form_edit else {
            return;
        };

        let mut commit = false;
        let mut cancel = false;
        if let Some((_, buf)) = self.form_edit.as_mut() {
            egui::Window::new("Edit field")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    let resp = ui.text_edit_singleline(buf);
                    let enter_pressed =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    ui.horizontal(|ui| {
                        if ui.button("Set").clicked() || enter_pressed {
                            commit = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                });
        }

        if commit {
            if let Some((_, value)) = self.form_edit.take() {
                let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, self.page);
                if let Some(field) = fields.iter().find(|f| f.obj_num == obj_num) {
                    let result =
                        kopitiam_pdf::mupdf::form::set_field_value(&self.doc, field, &value);
                    self.apply_edit(result);
                } else {
                    self.status = Some("field no longer present on this page".to_string());
                }
            }
        } else if cancel {
            self.form_edit = None;
        }
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

            // While the form-field text editor popup is open, keys feed that
            // box (via egui's own `TextEdit` focus handling, not this manual
            // loop) -- same `is_capturing_text` idea as `goto` below. Only
            // `Escape` is special-cased here, to close the popup without
            // committing, and to keep `q`/`g`/nav keys from doing anything
            // else while it's up.
            if self.form_edit.is_some() {
                if key == egui::Key::Escape {
                    self.form_edit = None;
                }
                continue;
            }

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
fn page_display_size(
    tex_w: f32,
    tex_h: f32,
    dpi: f32,
    available_w: f32,
    available_h: f32,
) -> (f32, f32) {
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
fn recentred_scroll_offset(
    offset: f32,
    content_size: f32,
    viewport_size: f32,
    new_content_size: f32,
) -> f32 {
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

// -- The coordinate trap: screen space <-> default user space -------------
//
// See the module docs' "The coordinate trap" section for the framing. These
// three functions are the entire seam: everything else (drag handling,
// hit-testing, the ink preview) only ever calls through them, never touches
// screen or page coordinates directly.

/// The page's physical size in **PDF points** (default user space),
/// recovered from a rasterized texture's pixel size and the dpi it was
/// rasterized at: `points = pixels / dpi * 72` (72 points per inch is the
/// PDF spec's own unit definition, not a kpdf convention).
///
/// This is **dpi-invariant** by construction -- rasterizing the same page at
/// a different dpi changes `tex_w`/`tex_h` but this function's return value
/// stays the same, which is exactly what makes it safe to use as the fixed
/// reference frame in [`screen_to_page`]/[`page_to_screen`] no matter what
/// zoom level is currently on screen.
fn page_size_pts(tex_w: f32, tex_h: f32, dpi: f32) -> (f32, f32) {
    if dpi <= 0.0 {
        return (0.0, 0.0);
    }
    (tex_w / dpi * 72.0, tex_h / dpi * 72.0)
}

/// Everything [`screen_to_page`]/[`page_to_screen`] need about the current
/// frame's layout: where the page image is drawn on screen, and the page's
/// own physical size. Bundled into one small `Copy` struct purely to keep
/// those two functions' argument count under clippy's `too_many_arguments`
/// lint -- it carries no behaviour beyond being a parameter bag, and every
/// field is still named at each construction site so nothing is positionally
/// ambiguous.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PageLayout {
    /// Top-left of the on-screen image rect (screen space).
    image_x: f32,
    image_y: f32,
    /// Displayed size of the page image (screen space) -- already fitted and
    /// zoomed by [`page_display_size`], **not** the texture's raw pixel size.
    image_w: f32,
    image_h: f32,
    /// The page's physical size in PDF points -- from [`page_size_pts`].
    page_w_pts: f32,
    page_h_pts: f32,
}

/// Screen space (egui ui points, origin top-left, y **down** -- what a
/// pointer event reports) to default user space (PDF points, origin
/// bottom-left, y **up** -- what [`InkStroke`]/[`AnnotRef::rect`]/
/// [`FormField::rect`] all expect).
///
/// Pure (plain `f32`s and [`PageLayout`], no `egui`/`mupdf` types), so it is
/// unit-tested directly without a display -- see the `tests` module's
/// round-trip and y-flip cases, which is the one part of this file's new
/// work a gate can actually check.
fn screen_to_page(screen_x: f32, screen_y: f32, layout: PageLayout) -> (f32, f32) {
    if layout.image_w <= 0.0 || layout.image_h <= 0.0 {
        return (0.0, 0.0);
    }
    let fx = (screen_x - layout.image_x) / layout.image_w;
    let fy = (screen_y - layout.image_y) / layout.image_h;
    let page_x = fx * layout.page_w_pts;
    // The y-flip: screen y grows downward from the image's top; page y
    // grows upward from the page's bottom. fy=0 (image top) must map to
    // page_y = page_h_pts (page top); fy=1 (image bottom) must map to
    // page_y = 0 (page bottom).
    let page_y = (1.0 - fy) * layout.page_h_pts;
    (page_x, page_y)
}

/// The inverse of [`screen_to_page`] -- default user space back to screen
/// space, e.g. to paint the in-progress ink preview
/// ([`KpdfApp::handle_draw`]) or to position something over a `/Rect`. See
/// [`screen_to_page`]'s docs for the parameters and the y-flip this undoes.
fn page_to_screen(page_x: f32, page_y: f32, layout: PageLayout) -> (f32, f32) {
    if layout.page_w_pts <= 0.0 || layout.page_h_pts <= 0.0 {
        return (layout.image_x, layout.image_y);
    }
    let fx = page_x / layout.page_w_pts;
    let fy = 1.0 - page_y / layout.page_h_pts;
    (
        layout.image_x + fx * layout.image_w,
        layout.image_y + fy * layout.image_h,
    )
}

// -- Tool / forms-mode state machine ---------------------------------------

/// Switch to `requested` and turn forms mode off. Forms mode and a drawing
/// tool are deliberately mutually exclusive -- a stray ink stroke while
/// trying to fill in a text field, or a checkbox toggling under the eraser,
/// would both be surprising, and Okular's own forms toggle and its
/// annotation tools don't operate at the same time either.
///
/// Pure state transition over plain fields (no `KpdfApp` borrow), so it is
/// unit-tested directly; see [`KpdfApp::select_tool`] for the method that
/// calls it.
fn select_tool(tool: &mut Tool, forms_mode: &mut bool, requested: Tool) {
    *tool = requested;
    *forms_mode = false;
}

/// Flip forms mode, and if it is turning on, drop back to [`Tool::Pan`] --
/// same mutual-exclusion reasoning as [`select_tool`]. See
/// [`KpdfApp::toggle_forms_mode`] for the method that calls it.
fn toggle_forms_mode(tool: &mut Tool, forms_mode: &mut bool) {
    *forms_mode = !*forms_mode;
    if *forms_mode {
        *tool = Tool::Pan;
    }
}

// -- Hit-testing ------------------------------------------------------------

/// Whether `(x, y)` (default user space) falls within `r` -- an inclusive
/// bounds check, since a click exactly on an annotation's edge should still
/// hit it.
fn rect_contains(r: Rect, x: f32, y: f32) -> bool {
    x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1
}

/// The annotation under `(page_x, page_y)` (default user space), or `None`
/// if `refs` has nothing there -- what the eraser tool needs in order to
/// know which annotation object number to delete.
///
/// Iterates back-to-front (`.rev()`) so an annotation stacked visually on
/// top of another (later in the page's `/Annots` array, per PDF's paint
/// order) wins the hit, matching what is actually seen on screen.
///
/// Pure over [`AnnotRef`]'s public fields, with no document access -- so it
/// is unit-tested directly against hand-built `AnnotRef`s, without needing
/// [`kopitiam_pdf::mupdf::annot_edit::page_annot_refs`] (still `todo!()` as
/// of this writing).
fn hit_test_annot(page_x: f32, page_y: f32, refs: &[AnnotRef]) -> Option<i32> {
    refs.iter()
        .rev()
        .find(|a| rect_contains(a.rect, page_x, page_y))
        .map(|a| a.num)
}

/// Same idea as [`hit_test_annot`], for form-field widgets. Returns an
/// **index into `fields`** rather than an object number, because
/// [`kopitiam_pdf::mupdf::form::set_field_value`]/`toggle_checkbox` take a
/// whole `&FormField`, not a bare handle.
fn hit_test_field(page_x: f32, page_y: f32, fields: &[FormField]) -> Option<usize> {
    fields
        .iter()
        .enumerate()
        .rev()
        .find(|(_, f)| rect_contains(f.rect, page_x, page_y))
        .map(|(i, _)| i)
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
        self.show_form_edit_popup(ui.ctx());

        egui::Panel::top("kpdf-status").show(ui, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    // Toolbar: Open/Undo/Redo/Save, then the annotation
                    // tools, then Forms mode (only for a document that has
                    // one) -- see the module docs' "Annotation tools and
                    // forms mode" section for what each does.
                    if ui.button("Open").on_hover_text("Open a PDF (o)").clicked() {
                        self.open_via_dialog();
                    }
                    ui.separator();

                    let can_undo = self
                        .edit_history
                        .as_ref()
                        .is_some_and(EditHistory::can_undo);
                    let can_redo = self
                        .edit_history
                        .as_ref()
                        .is_some_and(EditHistory::can_redo);
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Undo"))
                        .clicked()
                    {
                        self.undo();
                    }
                    if ui
                        .add_enabled(can_redo, egui::Button::new("Redo"))
                        .clicked()
                    {
                        self.redo();
                    }
                    // Saving is meaningful exactly when there is an edit to
                    // save, i.e. exactly when Undo is available (see
                    // `EditHistory::can_undo`'s docs: the whole history is
                    // rooted at the document as first opened).
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Save"))
                        .on_hover_text("Save to a new file")
                        .clicked()
                    {
                        self.save_via_dialog();
                    }
                    ui.separator();

                    if ui
                        .selectable_label(self.tool == Tool::Draw, "Pen")
                        .on_hover_text("Draw ink annotations")
                        .clicked()
                    {
                        self.select_tool(Tool::Draw);
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Erase, "Eraser")
                        .on_hover_text("Delete annotations")
                        .clicked()
                    {
                        self.select_tool(Tool::Erase);
                    }
                    if self.tool != Tool::Pan
                        && ui
                            .button("Pan")
                            .on_hover_text("Back to plain scrolling")
                            .clicked()
                    {
                        self.select_tool(Tool::Pan);
                    }

                    // Only offered when the document actually has an
                    // /AcroForm -- see `has_acroform`'s doc comment.
                    if self.has_acroform {
                        ui.separator();
                        if ui
                            .selectable_label(self.forms_mode, "Forms")
                            .on_hover_text("Toggle interactive form fields")
                            .clicked()
                        {
                            self.toggle_forms_mode();
                        }
                    }
                });

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
                        let (w, h) = page_display_size(
                            tw as f32,
                            th as f32,
                            self.dpi,
                            available.x,
                            available.y,
                        );
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

                    // `Sense::click_and_drag()` -- the `Image` widget senses
                    // nothing by default (`Sense::hover()`); without this,
                    // none of the tool below (draw/erase/forms click) would
                    // ever see a `Response::clicked()`/`dragged()`.
                    let output = scroll_area.show(ui, |ui| {
                        ui.add(
                            egui::Image::new(tex)
                                .fit_to_exact_size(new_size)
                                .sense(egui::Sense::click_and_drag()),
                        )
                    });

                    self.page_scroll_offset = output.state.offset;
                    self.page_content_size = new_size;
                    self.page_viewport_size = output.inner_rect.size();
                    self.prev_dpi = self.dpi;

                    // Route the click/drag to whichever tool is active. All
                    // three share the same coordinate conversion
                    // (`screen_to_page`/`page_size_pts`) -- see the module
                    // docs' "coordinate trap" section -- and all business
                    // logic (what a hit means, how to edit the PDF) lives in
                    // `kopitiam_pdf::mupdf::{annot_edit,form}`, not here.
                    let image_response = output.inner;
                    let img_rect = image_response.rect;
                    let (page_w_pts, page_h_pts) = page_size_pts(tw as f32, th as f32, self.dpi);
                    let layout = PageLayout {
                        image_x: img_rect.min.x,
                        image_y: img_rect.min.y,
                        image_w: img_rect.width(),
                        image_h: img_rect.height(),
                        page_w_pts,
                        page_h_pts,
                    };
                    if self.forms_mode {
                        self.handle_forms_click(&image_response, layout);
                    } else {
                        match self.tool {
                            Tool::Pan => {}
                            Tool::Draw => self.handle_draw(&image_response, layout, ui),
                            Tool::Erase => self.handle_erase(&image_response, layout),
                        }
                    }
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
        assert_eq!(
            page_display_size(0.0, 100.0, DPI_DEFAULT, 500.0, 500.0),
            (0.0, 0.0)
        );
        assert_eq!(
            page_display_size(100.0, 0.0, DPI_DEFAULT, 500.0, 500.0),
            (0.0, 0.0)
        );
        assert_eq!(
            page_display_size(100.0, 100.0, 0.0, 500.0, 500.0),
            (0.0, 0.0)
        );
        assert_eq!(
            page_display_size(100.0, 100.0, -10.0, 500.0, 500.0),
            (0.0, 0.0)
        );
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

    // -- page_size_pts ---------------------------------------------------

    #[test]
    fn page_size_pts_at_72_dpi_is_pixels_unchanged() {
        // 72 points per inch is the PDF spec's own definition of a "point",
        // so rasterizing at exactly 72 dpi must make pixels and points
        // numerically identical.
        assert_eq!(page_size_pts(612.0, 792.0, 72.0), (612.0, 792.0));
    }

    #[test]
    fn page_size_pts_is_dpi_invariant() {
        // The same page rasterized at two different dpis must recover the
        // *same* physical size -- this is the property that makes it safe
        // to use as screen_to_page/page_to_screen's fixed reference frame
        // regardless of the current zoom level.
        let at_150 = page_size_pts(1275.0, 1650.0, 150.0);
        let at_300 = page_size_pts(2550.0, 3300.0, 300.0);
        assert!((at_150.0 - at_300.0).abs() < 1e-3);
        assert!((at_150.1 - at_300.1).abs() < 1e-3);
        assert!((at_150.0 - 612.0).abs() < 1e-3); // US Letter width in points
    }

    #[test]
    fn page_size_pts_degenerate_dpi_returns_zero() {
        assert_eq!(page_size_pts(100.0, 100.0, 0.0), (0.0, 0.0));
        assert_eq!(page_size_pts(100.0, 100.0, -1.0), (0.0, 0.0));
    }

    // -- screen_to_page / page_to_screen (the coordinate trap) -----------

    /// A representative layout: a 612x792pt (US Letter) page displayed in a
    /// 300x400 screen-point image rect, offset from the window's origin (so
    /// a bug that forgets to subtract `image_x`/`image_y` shows up).
    fn sample_layout() -> PageLayout {
        PageLayout {
            image_x: 20.0,
            image_y: 40.0,
            image_w: 300.0,
            image_h: 400.0,
            page_w_pts: 612.0,
            page_h_pts: 792.0,
        }
    }

    #[test]
    fn y_flip_top_of_image_is_top_of_page() {
        // The image's top-left corner (screen space) must map to the
        // page's top-left corner in default user space -- i.e. x=0, and
        // y = page_h_pts (page y grows *up* from the bottom, so the top
        // edge is at the maximum y, not 0). Getting this backwards is
        // exactly "the coordinate trap" the module docs call out.
        let layout = sample_layout();
        let (px, py) = screen_to_page(layout.image_x, layout.image_y, layout);
        assert!((px - 0.0).abs() < 1e-3);
        assert!((py - layout.page_h_pts).abs() < 1e-3);
    }

    #[test]
    fn y_flip_bottom_of_image_is_bottom_of_page() {
        let layout = sample_layout();
        let (px, py) = screen_to_page(
            layout.image_x + layout.image_w,
            layout.image_y + layout.image_h,
            layout,
        );
        assert!((px - layout.page_w_pts).abs() < 1e-3);
        assert!((py - 0.0).abs() < 1e-3);
    }

    #[test]
    fn screen_to_page_center_maps_to_page_center() {
        let layout = sample_layout();
        let (px, py) = screen_to_page(
            layout.image_x + layout.image_w / 2.0,
            layout.image_y + layout.image_h / 2.0,
            layout,
        );
        assert!((px - layout.page_w_pts / 2.0).abs() < 1e-2);
        assert!((py - layout.page_h_pts / 2.0).abs() < 1e-2);
    }

    #[test]
    fn round_trip_page_to_screen_to_page() {
        // page_to_screen(screen_to_page(p)) ~= p, for a handful of points
        // spread across the image, not just the center -- the explicit
        // round-trip property the task calls for.
        let layout = sample_layout();
        for (sx, sy) in [
            (layout.image_x, layout.image_y),
            (
                layout.image_x + layout.image_w,
                layout.image_y + layout.image_h,
            ),
            (layout.image_x + 10.0, layout.image_y + 380.0),
            (layout.image_x + 150.0, layout.image_y + 200.0),
            (layout.image_x + 299.0, layout.image_y + 1.0),
        ] {
            let (px, py) = screen_to_page(sx, sy, layout);
            let (sx2, sy2) = page_to_screen(px, py, layout);
            assert!(
                (sx2 - sx).abs() < 1e-2,
                "x round-trip: {sx} -> {px} -> {sx2}"
            );
            assert!(
                (sy2 - sy).abs() < 1e-2,
                "y round-trip: {sy} -> {py} -> {sy2}"
            );
        }
    }

    #[test]
    fn round_trip_screen_to_page_to_screen() {
        // The other direction: starting from a page-space point (as if
        // reading a `/Rect` back), converting to screen and back must
        // recover it, across points that are not simply the corners.
        let layout = sample_layout();
        for (px, py) in [
            (0.0, 0.0),
            (612.0, 792.0),
            (100.0, 700.0),
            (306.0, 396.0),
            (611.0, 1.0),
        ] {
            let (sx, sy) = page_to_screen(px, py, layout);
            let (px2, py2) = screen_to_page(sx, sy, layout);
            assert!(
                (px2 - px).abs() < 1e-2,
                "x round-trip: {px} -> {sx} -> {px2}"
            );
            assert!(
                (py2 - py).abs() < 1e-2,
                "y round-trip: {py} -> {sy} -> {py2}"
            );
        }
    }

    #[test]
    fn screen_to_page_degenerate_image_size_returns_zero_not_nan() {
        let mut layout = sample_layout();
        layout.image_w = 0.0;
        assert_eq!(screen_to_page(50.0, 50.0, layout), (0.0, 0.0));
        layout = sample_layout();
        layout.image_h = -5.0;
        assert_eq!(screen_to_page(50.0, 50.0, layout), (0.0, 0.0));
    }

    #[test]
    fn page_to_screen_degenerate_page_size_returns_image_origin_not_nan() {
        let mut layout = sample_layout();
        layout.page_w_pts = 0.0;
        let (sx, sy) = page_to_screen(10.0, 10.0, layout);
        assert!(sx.is_finite() && sy.is_finite());
        assert_eq!((sx, sy), (layout.image_x, layout.image_y));
    }

    // -- Tool / forms-mode state machine ----------------------------------

    #[test]
    fn select_tool_sets_the_requested_tool() {
        let mut tool = Tool::Pan;
        let mut forms_mode = false;
        select_tool(&mut tool, &mut forms_mode, Tool::Draw);
        assert_eq!(tool, Tool::Draw);
    }

    #[test]
    fn selecting_a_tool_exits_forms_mode() {
        // Mutual exclusion, direction one: picking Pen/Eraser while forms
        // mode is on must turn forms mode off, so a click can never be
        // simultaneously "draw ink" and "toggle a checkbox".
        let mut tool = Tool::Pan;
        let mut forms_mode = true;
        select_tool(&mut tool, &mut forms_mode, Tool::Erase);
        assert_eq!(tool, Tool::Erase);
        assert!(!forms_mode);
    }

    #[test]
    fn toggling_forms_mode_on_resets_tool_to_pan() {
        // Mutual exclusion, direction two: turning forms mode on while a
        // drawing tool is active must fall back to Pan, not leave Draw/Erase
        // armed alongside it.
        let mut tool = Tool::Draw;
        let mut forms_mode = false;
        toggle_forms_mode(&mut tool, &mut forms_mode);
        assert!(forms_mode);
        assert_eq!(tool, Tool::Pan);
    }

    #[test]
    fn toggling_forms_mode_off_leaves_tool_at_pan() {
        let mut tool = Tool::Pan;
        let mut forms_mode = true;
        toggle_forms_mode(&mut tool, &mut forms_mode);
        assert!(!forms_mode);
        assert_eq!(tool, Tool::Pan);
    }

    #[test]
    fn toggling_forms_mode_twice_is_idempotent_on_tool() {
        let mut tool = Tool::Pan;
        let mut forms_mode = false;
        toggle_forms_mode(&mut tool, &mut forms_mode);
        toggle_forms_mode(&mut tool, &mut forms_mode);
        assert!(!forms_mode);
        assert_eq!(tool, Tool::Pan);
    }

    // -- hit_test_annot / hit_test_field -----------------------------------
    //
    // Both operate purely on the public fields of `AnnotRef`/`FormField` --
    // no document, no call into the still-`todo!()` `page_annot_refs`/
    // `page_form_fields`. Hand-built fixtures only.

    fn annot(num: i32, rect: Rect) -> AnnotRef {
        AnnotRef {
            num,
            subtype: "Ink".to_string(),
            rect,
        }
    }

    #[test]
    fn hit_test_annot_finds_containing_rect() {
        let refs = vec![
            annot(1, Rect::new(0.0, 0.0, 100.0, 100.0)),
            annot(2, Rect::new(200.0, 200.0, 300.0, 300.0)),
        ];
        assert_eq!(hit_test_annot(50.0, 50.0, &refs), Some(1));
        assert_eq!(hit_test_annot(250.0, 250.0, &refs), Some(2));
    }

    #[test]
    fn hit_test_annot_misses_return_none() {
        let refs = vec![annot(1, Rect::new(0.0, 0.0, 100.0, 100.0))];
        assert_eq!(hit_test_annot(150.0, 150.0, &refs), None);
        assert_eq!(hit_test_annot(150.0, 150.0, &[]), None);
    }

    #[test]
    fn hit_test_annot_boundary_is_inclusive() {
        let refs = vec![annot(1, Rect::new(0.0, 0.0, 100.0, 100.0))];
        assert_eq!(hit_test_annot(0.0, 0.0, &refs), Some(1));
        assert_eq!(hit_test_annot(100.0, 100.0, &refs), Some(1));
    }

    #[test]
    fn hit_test_annot_prefers_the_topmost_overlapping_annotation() {
        // Later entries in `/Annots` paint on top -- a hit inside the
        // overlap must resolve to the *later* (visually topmost) one, not
        // the first match encountered in array order.
        let refs = vec![
            annot(1, Rect::new(0.0, 0.0, 100.0, 100.0)),
            annot(2, Rect::new(50.0, 50.0, 150.0, 150.0)),
        ];
        assert_eq!(hit_test_annot(75.0, 75.0, &refs), Some(2));
    }

    fn field(obj_num: i32, kind: FieldKind, rect: Rect) -> FormField {
        FormField {
            obj_num,
            page_index: 0,
            kind,
            name: format!("field{obj_num}"),
            value: String::new(),
            rect,
            read_only: false,
            on_state: None,
        }
    }

    #[test]
    fn hit_test_field_finds_containing_rect_and_returns_index() {
        let fields = vec![
            field(10, FieldKind::Checkbox, Rect::new(0.0, 0.0, 20.0, 20.0)),
            field(11, FieldKind::Text, Rect::new(0.0, 100.0, 200.0, 120.0)),
        ];
        assert_eq!(hit_test_field(10.0, 10.0, &fields), Some(0));
        assert_eq!(hit_test_field(100.0, 110.0, &fields), Some(1));
    }

    #[test]
    fn hit_test_field_miss_returns_none() {
        let fields = vec![field(
            10,
            FieldKind::Checkbox,
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )];
        assert_eq!(hit_test_field(500.0, 500.0, &fields), None);
    }
}
