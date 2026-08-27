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
//!   every fillable field on the page is painted with a translucent
//!   highlight ([`KpdfApp::paint_and_edit_form_fields`]) -- Okular-style, so
//!   the fields are visible before anything is clicked, not discovered by
//!   hunting. Read-only fields and combobox/listbox (unsupported by
//!   [`kopitiam_pdf::mupdf::form::set_field_value`] in this release) get a
//!   visually distinct, non-actionable tint rather than being invited to
//!   click ([`field_highlight_kind`]). Clicking a checkbox/radio toggles it
//!   in place; clicking a text field opens an `egui::TextEdit` **in place**,
//!   positioned over the field's own rect on the page, rather than a
//!   separate popup window -- so editing happens where the text actually
//!   goes, matching what the highlight promised. A multiline field
//!   ([`kopitiam_pdf::mupdf::form::FormField::multiline`]) gets
//!   `egui::TextEdit::multiline`; `Enter` commits either kind (single- or
//!   multiline alike, per the maintainer's explicit call -- "enter saves the
//!   thing"), and only `Shift+Enter` inserts a newline in a multiline field
//!   ([`should_commit_on_enter`]). `Esc` cancels without writing anything
//!   (see [`KpdfApp::handle_key`]'s `form_edit` guard).
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
//! whether the form-field highlights read clearly against the rendered page,
//! whether the in-place text editor sits exactly over the field it edits at
//! every zoom level, whether a tiny checkbox is comfortably clickable at the
//! chosen minimum hit area ([`MIN_HIT_AREA_SCREEN`]), and whether `Enter`
//! vs. `Shift+Enter` in a multiline field feels right are all **unverified
//! and need a human at a real display** -- per this project's GUI-surfaces
//! rule, that dogfooding is not something an agent can fake.

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;
// Reusable page-layout/zoom/hit-testing/forms-UI/tool-state building blocks,
// lifted out of this binary into the library proper
// (kopitiam_pdf::gui_frontend, gated on the crate's `kpdf` feature) so other
// egui-based KOPITIAM front ends can reuse them without re-implementing this
// file.
use kopitiam_pdf::gui_frontend::{
    DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, FieldHighlight, PageLayout, Tool,
    consume_commit_enter, drawable_annot_count, field_highlight_kind, field_rect_to_screen,
    highlight_colors, hit_test_annot, hit_test_field, hit_test_field_expanded, min_hit_rect,
    page_display_size, page_size_pts, page_to_screen, recentred_scroll_offset, rgb_to_rgba,
    screen_to_page, select_tool, toggle_forms_mode, zoom_percent, zoom_steps_from_zoom_delta,
};
use kopitiam_pdf::mupdf::annot_edit::{EditHistory, InkAnnotSpec, InkStroke};
use kopitiam_pdf::mupdf::form::FieldKind;
use kopitiam_pdf::mupdf::{PdfDocument, Rect, rasterize_page};
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

/// Minimum on-screen size (egui ui points, one per pixel at 100% OS scaling)
/// a form field's clickable area is widened to before hit-testing (and,
/// for consistency -- the highlighted area and the clickable area should
/// always agree -- before painting too). Without this, a field whose
/// `/Rect` renders as a hairline or a near-zero box at low zoom (not
/// unheard of; some form authors size the widget to a printed underline
/// rather than the real input box) would be a pixel hunt to click.
///
/// 16.0 is a judgment call, not a measured value -- there is no display in
/// this environment to test "does a 16px target feel comfortable to
/// click", so this is exactly the kind of choice a human dogfooder should
/// eyeball. See [`min_hit_rect`].
const MIN_HIT_AREA_SCREEN: f32 = 16.0;

// `Tool`, the DPI_*/ZOOM_DELTA_PER_STEP constants, and the zoom/layout/
// hit-testing/forms-UI functions that used to live here now live in
// `kopitiam_pdf::gui_frontend` (see the imports above) -- lifted out so other
// egui-based KOPITIAM front ends can reuse them.

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
    /// `Some((obj_num, buf))` while the in-place text-field editor (see
    /// [`KpdfApp::paint_and_edit_form_fields`]) is open over a field on the
    /// page: `obj_num` identifies which [`FormField`] (re-looked-up by
    /// obj_num each frame, from `form_fields_cache` for position/kind and
    /// from a fresh [`kopitiam_pdf::mupdf::form::page_form_fields`] call at
    /// commit time, since a `FormField` itself is not [`Clone`]), `buf` is
    /// the text being typed. Cleared on commit, on `Esc`
    /// ([`KpdfApp::handle_key`]), and whenever the active tool or forms mode
    /// changes ([`KpdfApp::select_tool`]/[`KpdfApp::toggle_forms_mode`]) --
    /// an in-progress, uncommitted edit should not survive leaving forms
    /// mode with no way left to interact with it.
    form_edit: Option<(i32, String)>,
    /// `true` for exactly the one frame after a click opens `form_edit` --
    /// tells [`KpdfApp::paint_and_edit_form_fields`] to call
    /// `Response::request_focus()` once, so typing can start immediately
    /// without an extra click into the box. Must never stay `true` across
    /// frames: re-requesting focus every frame would mean the box could
    /// never lose focus, and therefore never commit-on-focus-loss.
    form_edit_focus_pending: bool,
    /// Cache of the current page's form fields for painting the highlight
    /// overlay, keyed by page index -- see
    /// [`KpdfApp::refresh_form_fields_cache`]. `None` until forms mode first
    /// needs it. Cleared to `None` (see [`KpdfApp::open_path`] and
    /// [`KpdfApp::reload_from_bytes`]) whenever `self.doc` is replaced,
    /// since a new `PdfDocument` can carry different field values/rects even
    /// at the same page index -- keyed on page alone would silently show a
    /// stale highlight otherwise.
    form_fields_cache: Option<(usize, Vec<CachedField>)>,
}

/// A page's form fields, captured as plain owned data for
/// [`KpdfApp::form_fields_cache`]. Unlike [`FormField`] (borrows nothing
/// itself, but carries no [`Clone`] -- it is meant to be built and consumed
/// within a single call), this is cheap to hold across frames, which is the
/// whole point: [`kopitiam_pdf::mupdf::form::page_form_fields`] walks the
/// page's `/Annots` (resolving every widget, its `/Parent` chain, its
/// appearance stream for `on_state`, ...) on every call, and painting the
/// highlight overlay runs every frame forms mode is on -- roughly 20/sec,
/// since [`eframe::App::ui`] ends with a `request_repaint_after`. Without
/// this cache, that would re-parse the PDF continuously.
#[derive(Clone)]
struct CachedField {
    obj_num: i32,
    kind: FieldKind,
    rect: Rect,
    read_only: bool,
    /// [`FormField::multiline`] -- which `egui::TextEdit` constructor
    /// [`KpdfApp::paint_and_edit_form_fields`] must use if this field is the
    /// one being edited.
    multiline: bool,
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
            form_edit_focus_pending: false,
            form_fields_cache: None,
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
                self.form_edit_focus_pending = false;
                self.form_fields_cache = None;
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
    ///
    /// Also drops any in-progress form-field edit: leaving forms mode (which
    /// this always does -- see [`select_tool`]) means there is no longer any
    /// path for the in-place editor to be interacted with (it is only drawn
    /// while forms mode is on, see
    /// [`KpdfApp::paint_and_edit_form_fields`]), so an edit left open would
    /// be stuck showing forever with no way to commit or cancel it short of
    /// `Esc`.
    fn select_tool(&mut self, requested: Tool) {
        select_tool(&mut self.tool, &mut self.forms_mode, requested);
        self.form_edit = None;
        self.form_edit_focus_pending = false;
    }

    /// Flip forms mode, via the pure [`toggle_forms_mode`] transition. Also
    /// drops any in-progress form-field edit when forms mode turns off --
    /// same reasoning as [`KpdfApp::select_tool`].
    fn toggle_forms_mode(&mut self) {
        toggle_forms_mode(&mut self.tool, &mut self.forms_mode);
        self.form_edit = None;
        self.form_edit_focus_pending = false;
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
                // A fresh `PdfDocument` invalidates `form_fields_cache`
                // exactly like it invalidates `texture` above -- form field
                // values/rects belong to the document that was just
                // replaced, not necessarily the new one (this is also how a
                // just-committed edit's new value shows up in the highlight
                // overlay/editor rather than the pre-edit one).
                self.form_fields_cache = None;
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
    /// in-place text-field editor (see
    /// [`KpdfApp::paint_and_edit_form_fields`]). All PDF-structure knowledge
    /// (what `/AS`, `/V`, an on-state name are) stays in
    /// `kopitiam_pdf::mupdf::form` -- this only converts a click to page
    /// coordinates and dispatches.
    ///
    /// Fetches a fresh `page_form_fields` rather than reusing
    /// `form_fields_cache`: a click is rare (unlike the highlight overlay's
    /// once-a-frame repaint, which is exactly what the cache exists to
    /// avoid re-parsing for), so there is no performance reason to risk
    /// acting on a stale cache here.
    fn handle_forms_click(&mut self, response: &egui::Response, layout: PageLayout) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if self.form_edit.is_some() {
            // A click on the page that the in-place editor itself didn't
            // consume (it sits in front of the page image, so a click *on*
            // it never reaches here) is exactly the "clicked away" case --
            // commit whatever is currently typed rather than silently
            // discarding it, whether the click lands on a different field
            // or on empty page space. `commit_form_edit` is idempotent
            // (`Option::take`), so if the editor's own `lost_focus` check
            // already committed it this same frame, this is a harmless
            // no-op.
            self.commit_form_edit();
        }
        let (px, py) = screen_to_page(pos.x, pos.y, layout);
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, self.page);
        // Exact rect first; only fall back to the widened hit area
        // (`MIN_HIT_AREA_SCREEN`) if nothing was hit outright, so a click
        // genuinely inside one field's real box is never second-guessed by
        // another field's widened area.
        let idx = hit_test_field(px, py, &fields)
            .or_else(|| hit_test_field_expanded(px, py, &fields, layout, MIN_HIT_AREA_SCREEN));
        let Some(idx) = idx else {
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
                self.form_edit_focus_pending = true;
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

    /// Populate (or reuse) `form_fields_cache` for the current page. Mirrors
    /// [`KpdfApp::ensure_texture`]'s `(page, dpi)`-keyed caching for the
    /// rasterised texture: see [`CachedField`]'s docs for why this exists at
    /// all. Keyed on page index only (no "doc version" component the way
    /// `texture_key` carries `dpi`) because every call site that can change
    /// `self.doc` out from under this cache -- [`KpdfApp::open_path`],
    /// [`KpdfApp::reload_from_bytes`] -- already clears it to `None`
    /// unconditionally there; see those methods.
    fn refresh_form_fields_cache(&mut self) {
        if self
            .form_fields_cache
            .as_ref()
            .is_some_and(|(page, _)| *page == self.page)
        {
            return;
        }
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, self.page)
            .iter()
            .map(|f| CachedField {
                obj_num: f.obj_num,
                kind: f.kind,
                rect: f.rect,
                read_only: f.read_only,
                multiline: f.multiline,
            })
            .collect();
        self.form_fields_cache = Some((self.page, fields));
    }

    /// Commit the in-place text editor's current buffer (see
    /// [`KpdfApp::paint_and_edit_form_fields`]) via
    /// [`kopitiam_pdf::mupdf::form::set_field_value`], then follow the same
    /// push-to-history/reopen/invalidate tail every edit in this file goes
    /// through ([`KpdfApp::apply_edit`]).
    ///
    /// A no-op if there is nothing to commit -- `Option::take` makes this
    /// safe to call from more than one place in the same frame. Both
    /// [`KpdfApp::paint_and_edit_form_fields`]'s own `lost_focus` check and
    /// [`KpdfApp::handle_forms_click`]'s "clicked away" guard can reach this
    /// for the very same click; only the first one to run does anything.
    ///
    /// Re-fetches the page's form fields fresh (rather than reading
    /// `form_fields_cache`) because the actual write needs a real
    /// [`FormField`] borrow, which the cache -- deliberately plain, `Clone`
    /// data, see [`CachedField`] -- does not carry.
    fn commit_form_edit(&mut self) {
        let Some((obj_num, value)) = self.form_edit.take() else {
            return;
        };
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, self.page);
        if let Some(field) = fields.iter().find(|f| f.obj_num == obj_num) {
            let result = kopitiam_pdf::mupdf::form::set_field_value(&self.doc, field, &value);
            self.apply_edit(result);
        } else {
            self.status = Some("field no longer present on this page".to_string());
        }
    }

    /// Paint the Okular-style "fillable area" overlay for every field on the
    /// current page that [`field_highlight_kind`] says gets one, and -- if a
    /// text field is currently being edited ([`KpdfApp::form_edit`]) -- draw
    /// the in-place `egui::TextEdit` over that field's own rect instead of a
    /// highlight for it (the editor box stands in for its own highlight, so
    /// the two are never drawn on top of each other).
    ///
    /// Called every frame forms mode is on, in image mode, right after the
    /// page's `ScrollArea` has laid out for this frame -- every screen rect
    /// is recomputed from `layout` here, never cached (only the *field
    /// data* -- kind/rect/read-only/multiline -- is cached, in
    /// `form_fields_cache`), so panning or zooming moves the overlay and the
    /// editor with the page underneath them on the very next frame. `clip`
    /// bounds both the painted rectangles and the editor to the
    /// `ScrollArea`'s visible viewport, so a field scrolled out of view
    /// doesn't bleed over the toolbar above the page.
    fn paint_and_edit_form_fields(
        &mut self,
        ui: &mut egui::Ui,
        layout: PageLayout,
        clip: egui::Rect,
    ) {
        let editing_obj = self.form_edit.as_ref().map(|(n, _)| *n);

        if let Some((_, fields)) = &self.form_fields_cache {
            let painter = ui.painter_at(clip);
            for field in fields {
                if Some(field.obj_num) == editing_obj {
                    continue;
                }
                let style = field_highlight_kind(field.kind, field.read_only);
                if style == FieldHighlight::None {
                    continue;
                }
                let hit_rect = min_hit_rect(field.rect, layout, MIN_HIT_AREA_SCREEN);
                let (x0, y0, x1, y1) = field_rect_to_screen(hit_rect, layout);
                let screen_rect = egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));
                let (fill, stroke) = highlight_colors(style);
                painter.rect_filled(screen_rect, 2.0, fill);
                painter.rect_stroke(screen_rect, 2.0, stroke, egui::StrokeKind::Inside);
            }
        }

        let Some((obj_num, _)) = self.form_edit else {
            return;
        };
        // `Rect`/`bool` are both `Copy`, so this extracts exactly the two
        // fields needed and releases the `form_fields_cache` borrow
        // immediately -- no cloning of the whole cached list needed just to
        // read one entry's geometry.
        let field_meta = self.form_fields_cache.as_ref().and_then(|(_, fields)| {
            fields
                .iter()
                .find(|f| f.obj_num == obj_num)
                .map(|f| (f.rect, f.multiline))
        });
        let Some((field_rect, multiline)) = field_meta else {
            // The field the editor was open for isn't in this page's cache
            // anymore (the page changed under it, or the doc was reloaded
            // and the cache hasn't been refreshed with it yet) -- there is
            // nothing sane left to draw the box over.
            self.form_edit = None;
            self.form_edit_focus_pending = false;
            return;
        };
        let hit_rect = min_hit_rect(field_rect, layout, MIN_HIT_AREA_SCREEN);
        let (x0, y0, x1, y1) = field_rect_to_screen(hit_rect, layout);
        let screen_rect = egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));

        // Enter alone must commit rather than insert a newline in a
        // multiline field -- the maintainer's explicit call: "enter saves
        // the thing. Shift-Enter gets a new line". `TextEdit::multiline`'s
        // own default is "any Enter is a newline" with no notion of shift,
        // so the shiftless-Enter event has to be pulled out of the input
        // queue *before* the widget runs below, or it would both commit
        // (via this flag) and still land in `buf` as a newline this same
        // frame. An unconsumed `Shift+Enter` reaches the widget untouched
        // and inserts a newline via its ordinary default handling -- so
        // this only ever needs to intervene for the plain-Enter case.
        let enter_commit = multiline && consume_commit_enter(ui);

        let prev_clip = ui.clip_rect();
        ui.set_clip_rect(clip);
        let mut commit = false;
        if let Some((_, buf)) = self.form_edit.as_mut() {
            ui.painter()
                .rect_filled(screen_rect, 0.0, egui::Color32::WHITE);
            let resp = if multiline {
                ui.put(screen_rect, egui::TextEdit::multiline(buf))
            } else {
                ui.put(screen_rect, egui::TextEdit::singleline(buf))
            };
            if self.form_edit_focus_pending {
                resp.request_focus();
                self.form_edit_focus_pending = false;
            }
            // For a single-line field, `TextEdit::singleline`'s own default
            // handling already surrenders focus on Enter (so `lost_focus()`
            // covers "Enter commits" for free there); `enter_commit` is
            // always `false` in that branch since it's gated on
            // `multiline`. For a multiline field, `lost_focus()` remains a
            // safety net for "clicked/tabbed away without pressing Enter".
            if enter_commit || resp.lost_focus() {
                commit = true;
            }
        }
        ui.set_clip_rect(prev_clip);

        if commit {
            self.commit_form_edit();
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

impl eframe::App for KpdfApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.handle_key(ui.ctx());
        self.handle_scroll_zoom(ui.ctx());

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
                    //
                    // The layout is computed here, inside the closure, from
                    // the image response's own rect (rather than a second
                    // time afterwards from `output.inner`) so there is only
                    // ever one place per frame that turns "where the image
                    // landed" into a `PageLayout` -- both halves of the
                    // closure's return value come from the same computation.
                    let output = scroll_area.show(ui, |ui| {
                        let img_resp = ui.add(
                            egui::Image::new(tex)
                                .fit_to_exact_size(new_size)
                                .sense(egui::Sense::click_and_drag()),
                        );
                        let (page_w_pts, page_h_pts) =
                            page_size_pts(tw as f32, th as f32, self.dpi);
                        let layout = PageLayout {
                            image_x: img_resp.rect.min.x,
                            image_y: img_resp.rect.min.y,
                            image_w: img_resp.rect.width(),
                            image_h: img_resp.rect.height(),
                            page_w_pts,
                            page_h_pts,
                        };
                        (img_resp, layout)
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
                    let (image_response, layout) = output.inner;
                    if self.forms_mode {
                        // Highlight overlay + in-place editor first (so a
                        // click that opens/commits/cancels an edit this same
                        // frame still sees a freshly-painted, freshly-cached
                        // field list), then the click itself.
                        self.refresh_form_fields_cache();
                        self.paint_and_edit_form_fields(ui, layout, output.inner_rect);
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
