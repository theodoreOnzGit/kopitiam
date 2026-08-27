//! `kpdf` -- a lightweight, native (egui/eframe) PDF viewer for kopitiam-pdf.
//!
//! A GUI sibling of `kopitiam view` (apps/cli/src/view.rs +
//! apps/cli/src/tui/viewer.rs): same rasterizer
//! ([`kopitiam_pdf::mupdf::rasterize_page_with_fallback`]), a related
//! zoom/goto/mode-toggle model and keybindings -- reimplemented against egui
//! instead of ratatui-image, because a terminal graphics protocol isn't
//! always available (or wanted) when eyeballing a glyph-rendering fix on a
//! real display.
//!
//! ```text
//! cargo run --release -p kopitiam-pdf --bin kpdf                      # native file picker
//! cargo run --release -p kopitiam-pdf --bin kpdf -- path/to/file.pdf  # open directly
//! ```
//!
//! With no path argument, a native "Open" dialog ([`rfd`], pure Rust: the
//! default `xdg-portal` + `wayland` features on Linux, native Win32/Cocoa
//! pickers elsewhere -- no GTK or other C toolchain needed to build it) opens
//! before the viewer window does, so there is nothing to specify up front.
//! Cancelling it exits quietly (not an error).
//!
//! # Continuous scroll (gh-89)
//!
//! Image mode shows every page of the document stacked in one scrollable
//! column -- Okular/most-PDF-readers style -- rather than one page at a
//! time. The overall approach (learn each page's on-screen size from its
//! already-cheap thumbnail render rather than a full rasterization; only
//! full-render pages that actually intersect the viewport, with an
//! off-screen page getting a plain placeholder rectangle instead; track
//! "current page" by which page has the most vertical overlap with the
//! viewport, recomputed every frame) is **ported from `kovan`**
//! (`outram-park-backend/crates/kovan/src/digitiser/gui/desktop/
//! pdf_reader.rs`, `continuous_pages_ui`/`thumbnail_texture`/
//! `full_res_texture`), a sibling KOPITIAM-adjacent app by the same
//! maintainer that solved continuous PDF scrolling first. Unlike that
//! source, which made continuous mode **view-only** (an explicit, documented
//! scope cut -- "re-deriving which page, and where on it, a click/drag
//! landed across a whole scrolling document is materially more work than the
//! single-page coordinate math the tools already use"), this port *does*
//! resolve pen/eraser/forms clicks to a page while scrolling: kopitiam-pdf
//! already has [`kopitiam_pdf::gui_frontend::screen_to_page`] and its
//! hit-testing, so the extra step is a binary search over a page-offset
//! table ([`kopitiam_pdf::gui_frontend::screen_to_page_at`]) plus
//! delegation, not a re-derivation. See that function's module
//! (`gui_frontend::geometry`) for the coordinate model and its unit tests
//! (page boundaries, the inter-page gap, above the first page, below the
//! last).
//!
//! Deliberately **not** ported from `kovan`'s reader: its box-annotation/
//! digitiser/text-selection tooling (out of scope here -- kpdf's own tools
//! are ink/eraser/forms, not `kovan`'s Digitise-graph/Read-table workflow),
//! its hot-reload-on-mtime-change feature, and its 3-pane bibliography/
//! project-context panels.
//!
//! # Keybindings
//!
//! Image mode:
//! * `n` / `→` / `PageDown` -- scroll to the next page; `p` / `←` /
//!   `PageUp` -- scroll to the previous page (via `egui`'s own
//!   `scroll_to_rect`, so the jump animates rather than snapping instantly).
//! * `:` opens a small **command line** (`Esc` cancels without acting). It is
//!   bound to `:` rather than a bare `g` specifically so vim's own `gg`/`G`
//!   (below) have `g` free. Commands, parsed by
//!   [`kopitiam_pdf::gui_frontend::parse_command`]:
//!
//!   * `:N<Enter>` -- go to page `N`. Vim's own line-number-jump idiom, not an
//!     invention. `:0` is reported as an error rather than clamped: PDF pages
//!     are 1-based to a reader, so `:0` means the user expected 0-based
//!     indexing and should be told.
//!   * `:w<Enter>` -- write the file **in place**, identical to `Ctrl+S`.
//!
//!   Anything else is reported in the status bar (`not a command: :q`) rather
//!   than silently ignored -- a command line that swallows input leaves the
//!   user unable to tell "not a command" from "command failed".
//! * `h`/`j`/`k`/`l` -- scroll left/down/up/right by a small step
//!   ([`kopitiam_pdf::gui_frontend::VIM_STEP`]).
//! * `Ctrl+d` / `Ctrl+u` -- scroll down/up by half a viewport (vim's own
//!   `:help CTRL-D`/`:help CTRL-U` contract).
//! * `gg` -- scroll to the first page; `G` -- scroll to the last page. `gg`
//!   is a two-key sequence tracked by
//!   [`kopitiam_pdf::gui_frontend::GPending`] with a timeout
//!   ([`kopitiam_pdf::gui_frontend::G_PENDING_TIMEOUT`]), so a stray `g` does
//!   not silently arm "the next g jumps to page 1" indefinitely.
//! * `+` / `-` -- zoom in / out (render dpi); Ctrl+scroll-wheel over the
//!   page does the same, one step per gesture.
//! * `r` / `Tab` -- switch to reflow (text) mode.
//!
//! None of the above fire while a keystroke is **captured** by something
//! else -- typing a `:`-entry page number, or typing into an open in-place
//! form-field editor. Every key event is routed through one predicate,
//! [`kopitiam_pdf::gui_frontend::keys_captured`], before any vim-motion or
//! nav-key arm runs (see [`KpdfApp::handle_key`]) -- a `j` typed into a form
//! field must insert the letter `j`, not scroll the page.
//!
//! Reflow mode:
//! * `j` / `↓` -- scroll down one line; `k` / `↑` -- scroll up one line.
//! * `PageDown` / `PageUp` -- scroll by a viewport.
//! * `i` / `Tab` -- switch to image mode.
//!
//! Global: `o` -- open a different PDF via the same native file picker
//! (replaces the current document in place; on cancel or a failed open, the
//! current document keeps showing and the status bar reports the error).
//! `Ctrl+S` -- save the current (possibly edited) bytes **in place**, over
//! the open file -- see [`KpdfApp::save_in_place`] for why this is written
//! atomically (temp file + rename) rather than truncate-then-write. Distinct
//! from the toolbar **Save** button, which always asks for a new
//! destination via a save dialog and never touches the original file.
//! `q` / `Esc` quit (when not mid-`goto`, where `Esc` cancels the entry
//! instead); `Ctrl+C` always quits.
//!
//! # Not implemented this pass (deferred, not half-wired)
//!
//! * `f` -- vimium/hop.nvim-style field-label hopping in forms mode. Scope
//!   cut for this pass; forms mode is fully usable by mouse click today (see
//!   below), just not by a keyboard hop shortcut.
//! * `i` -- begin editing the field under the cursor/last hopped-to field in
//!   forms mode (depends on the `f` hop above).
//!
//! Neither is reachable from any menu/button either, per this crate's "no
//! feature half-wired" rule -- they are simply absent, not a dead button.
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
//!   see [`KpdfApp::save_via_dialog`]. Also on **`Ctrl+Shift+S`**, the
//!   conventional Save-As chord.
//!   **Save in place (`Ctrl+S`, or `:w`)** -- the distinct in-place save
//!   described above ([`KpdfApp::save_in_place`]), also reachable as a button.
//!
//!   The two chords are deliberately distinguished: `Ctrl+S` explicitly checks
//!   that Shift is *not* held, because without that guard the in-place save
//!   would also fire on the Save-As chord and overwrite the very file the user
//!   was trying to preserve.
//! * **Pen** -- drag on the page to draw; releasing commits the stroke as a
//!   real ink annotation via [`kopitiam_pdf::mupdf::annot_edit::add_ink_annot`].
//!   An ink stroke belongs to whichever page the drag *started* on -- if the
//!   drag strays over a neighbouring page's row in the continuous view, later
//!   points there are ignored rather than appended (a stroke cannot span two
//!   PDF pages), see [`KpdfApp::handle_draw`].
//! * **Eraser** -- click or drag over an existing annotation to delete it,
//!   via [`kopitiam_pdf::mupdf::annot_edit::delete_annot`], resolved against
//!   whichever page the click actually landed on.
//! * **Forms** -- offered only when the open document actually has an
//!   `/AcroForm` ([`kopitiam_pdf::mupdf::form::has_acroform`]). While on,
//!   every fillable field on every *visible* page is painted with a
//!   translucent highlight (Okular-style, so the fields are visible before
//!   anything is clicked). Read-only fields and combobox/listbox
//!   (unsupported by [`kopitiam_pdf::mupdf::form::set_field_value`] in this
//!   release) get a visually distinct, non-actionable tint. Clicking a
//!   checkbox/radio toggles it in place; clicking a text field opens an
//!   `egui::TextEdit` **in place**, over the field's own rect on whichever
//!   page it belongs to. `Enter` commits (both single- and multiline
//!   fields); only `Shift+Enter` inserts a newline in a multiline field.
//!   `Esc` cancels without writing anything.
//! * **Fallback** toggle -- switches
//!   [`kopitiam_pdf::mupdf::rasterize_page_with_fallback`]'s cross-engine
//!   fallback on/off (default on, today's behaviour). Off pins rendering to
//!   kopitiam's own engine, advance boxes and all -- a debugging/
//!   bug-reporting affordance (see that function's own docs). Part of the
//!   page-texture cache key (`(page, dpi, fallback)`), so toggling it always
//!   shows a freshly rendered image rather than a stale one from the other
//!   engine.
//!
//! # The left page sidebar
//!
//! A collapsible thumbnail strip (toolbar "Hide/Show pages" button, or the
//! sidebar simply absent when `page_count <= 1`) lists every page as a small
//! clickable thumbnail; clicking one scrolls the main view to that page.
//! Thumbnails are rasterized once, at a low, fixed dpi
//! ([`THUMBNAIL_DPI`] -- far below render dpi, "cheap enough to render every
//! page's, up front, without it costing anything a reader would notice") and
//! cached for the life of the open document -- see
//! [`KpdfApp::thumbnail_texture`].
//!
//! ## The coordinate trap
//!
//! egui reports pointer positions in **screen space** (ui points, origin
//! top-left, y down). The library's annotation/form APIs
//! ([`InkStroke`], [`AnnotRef::rect`](kopitiam_pdf::mupdf::annot_edit::AnnotRef::rect),
//! [`FormField::rect`](kopitiam_pdf::mupdf::form::FormField::rect)) all speak
//! **default user space** (PDF points, origin bottom-left, y **up**). In a
//! continuous, all-pages-in-one-column view there is a *third* space in
//! between: **content space**, where content y=0 is the top of the very
//! first page and increases monotonically down the whole document,
//! independent of the current scroll position (see
//! [`kopitiam_pdf::gui_frontend::ContinuousSlot`]'s docs). Every click/drag
//! this file handles converts screen -> content (subtract the content
//! column's on-screen origin for this frame) -> page space
//! ([`kopitiam_pdf::gui_frontend::screen_to_page_at`]) in that order, through
//! the one shared background `Response` built in the `Mode::Image` branch of
//! [`eframe::App::ui`] -- getting any of those seams wrong would put ink in
//! the wrong place, or on the wrong page, and it would look plausible enough
//! to ship without a careful eye.
//!
//! ## What is genuinely unverified here
//!
//! There is no display in this environment. Compiling, clippy, and the pure
//! coordinate-math/state-machine/hit-testing/LRU-eviction unit tests (see
//! `kopitiam_pdf::gui_frontend`) are all that has actually been checked.
//! Whether continuous scrolling reads smoothly, whether the sidebar
//! thumbnails render legibly, whether the vim motions feel right (`VIM_STEP`
//! and `G_PENDING_TIMEOUT` are unmeasured judgment calls, flagged as such at
//! their definitions), whether the pen/eraser/forms tools still feel
//! precise while scrolling, and whether Ctrl+S's status-bar confirmation is
//! noticeable are all **unverified and need a human at a real display** --
//! per this project's GUI-surfaces rule, that dogfooding is not something an
//! agent can fake.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;
// Reusable page-layout/zoom/hit-testing/forms-UI/tool-state/continuous-scroll/
// vim-key/LRU building blocks, lifted out of this binary into the library
// proper (kopitiam_pdf::gui_frontend, gated on the crate's `kpdf` feature) so
// other egui-based KOPITIAM front ends can reuse them without
// re-implementing this file.
use kopitiam_pdf::gui_frontend::{
    Command, ContinuousSlot, DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, FieldHighlight, GPending,
    Lru, PageLayout, PageSize, Tool, VIM_STEP, consume_commit_enter, continuous_slot_visible,
    current_page_in_view, drawable_annot_count, field_highlight_kind, field_rect_to_screen,
    g_pending_expired, half_viewport_step, highlight_colors, hit_test_annot, hit_test_field,
    hit_test_field_expanded, keys_captured, layout_continuous_pages, min_hit_rect, page_size_pts,
    page_to_screen, parse_command, rgb_to_rgba, screen_to_page_at, select_tool, toggle_forms_mode,
    zoom_percent, zoom_steps_from_zoom_delta,
};
use kopitiam_pdf::mupdf::annot_edit::{EditHistory, InkAnnotSpec, InkStroke};
use kopitiam_pdf::mupdf::form::FieldKind;
use kopitiam_pdf::mupdf::{PdfDocument, Rect, rasterize_page_with_fallback};
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

/// Low, fixed dpi every page is rasterized at once for two purposes: the
/// sidebar thumbnail strip, and -- scaled up by `dpi / THUMBNAIL_DPI` --
/// learning every page's on-screen size for the continuous layout without a
/// full-resolution render (see [`KpdfApp::page_size_for_layout`]). This is
/// the technique `kovan`'s continuous-scroll reader uses (see this file's
/// module docs); its own constant is 36.0, chosen for a slightly larger
/// thumbnail strip in that app's 3-pane layout. 24.0 here instead, at the
/// upper end of "something like 16-24" from this crate's own brief --
/// smaller than kovan's for a slightly cheaper up-front render, still large
/// enough that a scaled-up placeholder box's aspect ratio reads correctly.
/// Either value is a judgment call, not a measured one.
const THUMBNAIL_DPI: f32 = 24.0;

/// Vertical gap (screen points) between consecutive pages in the continuous
/// layout -- large enough that the boundary between two pages is visually
/// obvious (and unambiguous for [`screen_to_page_at`]'s "this point is in
/// neither page" case), small enough not to waste much scrolling room.
/// Matches `kovan`'s own `spacing` constant for the same layout.
const CONTINUOUS_GAP: f32 = 12.0;

/// How far beyond the visible viewport (screen points, both edges) a page is
/// still worth full-resolution rendering -- see
/// [`kopitiam_pdf::gui_frontend::continuous_slot_visible`]. Widening by
/// roughly "most of a screen" means a fast scroll finds the next page or two
/// already rasterized instead of flashing a bare placeholder first. Not a
/// measured value; a bigger number trades some extra up-front render cost
/// for smoother fast scrolling.
const VISIBLE_MARGIN: f32 = 600.0;

/// How many `(page, dpi, fallback)` full-resolution page textures stay
/// resident at once (see [`kopitiam_pdf::gui_frontend::Lru`]) -- the gh-88
/// performance concern this pass exists to close: rendering (and holding)
/// every page of a long document would make continuous scroll *worse* than
/// the single-page viewer it replaces, not better.
///
/// 24 is sized for "comfortably more than what's visible plus
/// [`VISIBLE_MARGIN`] lookahead in either direction at once" -- a typical
/// window shows one to a few pages at a time, `VISIBLE_MARGIN` adds roughly
/// one more screen's worth on each side, and 24 leaves generous headroom
/// above that without holding an unbounded number of full-resolution
/// textures (each several megabytes at a normal render dpi) for a document
/// with hundreds or thousands of pages. Not a measured value -- a human
/// dogfooding a very long document at a very high zoom is the one who can
/// say whether it needs raising.
const PAGE_TEXTURE_CACHE_CAPACITY: usize = 24;

// `Tool`, the DPI_*/ZOOM_DELTA_PER_STEP constants, and the zoom/layout/
// hit-testing/forms-UI/continuous-scroll/vim-key/LRU functions that used to
// live here (or were newly added for this pass) now live in
// `kopitiam_pdf::gui_frontend` (see the imports above) -- lifted out so other
// egui-based KOPITIAM front ends can reuse them.

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Image,
    Reflow,
}

/// A page's full-resolution texture is cached by `(page_index, dpi.to_bits(),
/// fallback_enabled)` -- `dpi` and the fallback toggle are both part of the
/// key so changing either always shows a freshly rendered image rather than
/// one from the previous setting. `f32` has no [`Eq`]/[`Hash`], hence
/// `to_bits()`.
type PageTextureKey = (usize, u32, bool);

/// Cache key for the continuous-layout slot list ([`KpdfApp::slots_cache`]):
/// rebuild only when dpi, the page count (a new document), how many
/// thumbnails have been rasterized so far (more thumbnails means more
/// accurate page sizes have become available), or the fallback toggle
/// actually change -- not on every frame.
type SlotsCacheKey = (u32, usize, usize, bool);

struct KpdfApp {
    doc: PdfDocument,
    path: PathBuf,
    page_count: usize,
    /// The current "most-visible" page, 0-based -- recomputed every frame
    /// from viewport overlap (see [`current_page_in_view`]), not assigned
    /// directly by nav keys/buttons. `next_page`/`prev_page`/
    /// `goto_page_1based`/the `gg`/`G` vim motions all act on this
    /// indirectly, via [`KpdfApp::scroll_to_page`] -- assigning `page`
    /// directly here would just be overwritten by the next frame's
    /// recomputation before it ever painted.
    page: usize,
    dpi: f32,
    mode: Mode,
    /// `Some(page)` for exactly the frame a nav action (next/prev/goto/`gg`/
    /// `G`/a sidebar click) asks the continuous view to scroll to that page;
    /// consumed (via `egui::Ui::scroll_to_rect`) and cleared by
    /// `Mode::Image`'s rendering this same frame.
    scroll_to_page: Option<usize>,
    /// A vim `h`/`j`/`k`/`l`/`Ctrl+d`/`Ctrl+u` scroll nudge queued by
    /// [`KpdfApp::handle_key_image`], applied via `egui::Ui::
    /// scroll_with_delta` inside the continuous view's scroll area and reset
    /// to zero every frame after being applied (or not, if nothing queued
    /// one).
    pending_scroll_delta: egui::Vec2,
    /// `Some(buf)` while a "go to page" number is being typed (image mode,
    /// triggered by `:`).
    cmdline: Option<String>,
    /// Leftover Ctrl+scroll signal not yet large enough to make a whole
    /// [`DPI_STEP`] move -- see `zoom.rs`'s `ZOOM_DELTA_PER_STEP` and
    /// `zoom_steps_from_zoom_delta`. Persists across frames so a slow or
    /// interrupted scroll gesture still adds up correctly instead of being
    /// discarded on every repaint.
    scroll_zoom_accum: f32,
    /// The continuous view's viewport height as of the last frame it was
    /// measured -- `Ctrl+d`/`Ctrl+u`'s half-viewport step
    /// ([`half_viewport_step`]) needs this, and it is only known once the
    /// scroll area has actually laid out. Defaults to a plausible value
    /// before the first frame renders.
    last_viewport_h: f32,
    /// Full-resolution page textures for the continuous view -- see
    /// [`PageTextureKey`]. Bounded by [`page_textures_lru`]
    /// ([`PAGE_TEXTURE_CACHE_CAPACITY`]); see [`KpdfApp::ensure_page_texture`].
    page_textures: HashMap<PageTextureKey, egui::TextureHandle>,
    page_textures_lru: Lru<PageTextureKey>,
    /// Low-resolution page thumbnails, rasterized once at [`THUMBNAIL_DPI`]
    /// and never evicted for as long as the document stays open -- a
    /// thumbnail is tiny, so holding every page's costs little (same
    /// precedent `kovan`'s own equivalent cache sets; see the module docs).
    /// Used both for the sidebar strip and for [`KpdfApp::page_size_for_layout`].
    thumbnails: HashMap<usize, egui::TextureHandle>,
    /// Cached continuous-layout slots -- see [`SlotsCacheKey`] and
    /// [`KpdfApp::continuous_slots`].
    slots_cache: Option<(SlotsCacheKey, Vec<ContinuousSlot>)>,
    /// Whether the left page-thumbnail sidebar is hidden. Named (and
    /// defaulted to `false`, "shown") the same way `kovan`'s equivalent field
    /// is -- a fresh multi-page PDF is expected to open with the sidebar
    /// visible.
    hide_thumbnails: bool,
    /// Whether [`rasterize_page_with_fallback`]'s cross-engine fallback is
    /// used for both thumbnails and full-resolution renders. Default on
    /// (today's behaviour) -- see the module docs' "Fallback toggle"
    /// bullet. Part of both [`PageTextureKey`] and [`SlotsCacheKey`], so
    /// toggling it never leaves a stale image on screen.
    fallback_enabled: bool,
    /// Lazily extracted once, on first entry to reflow mode -- extraction
    /// walks the whole document, so it is not worth doing eagerly for a
    /// session that might only ever use image mode.
    reflow_pages: Option<Result<Vec<TextPage>, String>>,
    reflow_scroll: f32,
    status: Option<String>,
    /// How many drawable annotations the current page carries -- shown in the
    /// status bar so a human can tell "this page has none" from "this page has
    /// some and we failed to draw them". Recomputed whenever the current page
    /// changes.
    annot_count: usize,
    /// The active annotation tool (image mode only). See [`Tool`].
    tool: Tool,
    /// Whether form fields are interactive (Okular-style toggle). Only ever
    /// `true` when [`KpdfApp::has_acroform`] is also `true` -- see
    /// [`toggle_forms_mode`].
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
    /// **default user space** on the page it belongs to (see
    /// [`KpdfApp::draw_page`]) -- committed as a real annotation on
    /// drag-release ([`KpdfApp::handle_draw`]) and cleared either way.
    draw_stroke: Vec<(f32, f32)>,
    /// Which page [`KpdfApp::draw_stroke`] belongs to -- set on the first
    /// point of a drag, `None` otherwise. An ink stroke cannot span two PDF
    /// pages, so once armed, later drag points that resolve to a *different*
    /// page (the pointer strayed over a neighbouring row in the continuous
    /// view) are ignored rather than appended.
    draw_page: Option<usize>,
    /// `Some((page, obj_num, buf))` while the in-place text-field editor
    /// (see [`KpdfApp::paint_and_edit_form_fields`]) is open over a field on
    /// `page`: `obj_num` identifies which [`FormField`] (re-looked-up by
    /// obj_num each frame from `form_fields_cache` for position/kind, and
    /// from a fresh [`kopitiam_pdf::mupdf::form::page_form_fields`] call at
    /// commit time), `buf` is the text being typed. Cleared on commit, on
    /// `Esc`, and whenever the active tool or forms mode changes.
    form_edit: Option<(usize, i32, String)>,
    /// `true` for exactly the one frame after a click opens `form_edit` --
    /// tells [`KpdfApp::paint_and_edit_form_fields`] to call
    /// `Response::request_focus()` once, so typing can start immediately
    /// without an extra click into the box. Must never stay `true` across
    /// frames: re-requesting focus every frame would mean the box could
    /// never lose focus, and therefore never commit-on-focus-loss.
    form_edit_focus_pending: bool,
    /// Cache of each *visible* page's form fields for painting the highlight
    /// overlay, keyed by page index -- see
    /// [`KpdfApp::refresh_form_fields_cache`]. Unlike `page_textures`, not
    /// LRU-bounded: this is small, parsed metadata (not pixel data), so
    /// unbounded growth for as long as a document with an `/AcroForm` stays
    /// open is cheap. Cleared entirely whenever `self.doc` is replaced (open/
    /// undo/redo/edit) since field values/rects belong to the document that
    /// was just replaced.
    form_fields_cache: HashMap<usize, Vec<CachedField>>,
    /// The `gg` two-key sequence's pending state -- see
    /// [`kopitiam_pdf::gui_frontend::keys`].
    g_pending: GPending,
    /// Wall-clock time `g_pending` was last armed, so a stale one times out
    /// ([`g_pending_expired`]) instead of lingering for an unrelated later
    /// `g`. `None` whenever `g_pending` is not armed.
    g_armed_at: Option<Instant>,
}

/// A page's form fields, captured as plain owned data for
/// [`KpdfApp::form_fields_cache`]. Unlike [`FormField`] (borrows nothing
/// itself, but carries no [`Clone`] -- it is meant to be built and consumed
/// within a single call), this is cheap to hold across frames, which is the
/// whole point: [`kopitiam_pdf::mupdf::form::page_form_fields`] walks a
/// page's `/Annots` (resolving every widget, its `/Parent` chain, its
/// appearance stream for `on_state`, ...) on every call, and painting the
/// highlight overlay runs every frame forms mode is on, for every visible
/// page. Without this cache, that would re-parse the PDF continuously.
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
            scroll_to_page: None,
            pending_scroll_delta: egui::Vec2::ZERO,
            cmdline: None,
            scroll_zoom_accum: 0.0,
            last_viewport_h: 800.0,
            page_textures: HashMap::new(),
            page_textures_lru: Lru::new(PAGE_TEXTURE_CACHE_CAPACITY),
            thumbnails: HashMap::new(),
            slots_cache: None,
            hide_thumbnails: false,
            fallback_enabled: true,
            reflow_pages: None,
            reflow_scroll: 0.0,
            status: None,
            annot_count: 0,
            tool: Tool::Pan,
            forms_mode: false,
            has_acroform,
            edit_history: None,
            draw_stroke: Vec::new(),
            draw_page: None,
            form_edit: None,
            form_edit_focus_pending: false,
            form_fields_cache: HashMap::new(),
            g_pending: GPending::new(),
            g_armed_at: None,
        })
    }

    /// Load `path` into the *current* window in place -- used by the `o`
    /// keybinding. Keeps the user's current zoom/mode/sidebar/fallback
    /// preference rather than resetting them; everything document-specific
    /// (page position, cached textures/thumbnails, extracted reflow text) is
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
                self.scroll_to_page = None;
                self.pending_scroll_delta = egui::Vec2::ZERO;
                self.cmdline = None;
                self.page_textures.clear();
                self.page_textures_lru = Lru::new(PAGE_TEXTURE_CACHE_CAPACITY);
                self.thumbnails.clear();
                self.slots_cache = None;
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
                self.draw_page = None;
                self.form_edit = None;
                self.form_edit_focus_pending = false;
                self.form_fields_cache.clear();
                self.forms_mode = false;
                self.g_pending.cancel();
                self.g_armed_at = None;
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

    /// Scroll to the next/previous page, relative to [`KpdfApp::page`] (the
    /// currently most-visible page, tracked every frame -- see that field's
    /// docs). Sets [`KpdfApp::scroll_to_page`] rather than assigning `page`
    /// directly, since `page` is only ever a *read* of where the view
    /// actually is.
    fn next_page(&mut self) {
        if self.page + 1 < self.page_count {
            self.scroll_to_page = Some(self.page + 1);
        }
    }

    fn prev_page(&mut self) {
        if self.page > 0 {
            self.scroll_to_page = Some(self.page - 1);
        }
    }

    fn goto_page_1based(&mut self, page_1based: usize) {
        let clamped = page_1based.clamp(1, self.page_count.max(1));
        self.scroll_to_page = Some(clamped - 1);
    }

    /// vim `gg` -- scroll to the first page.
    fn go_to_first_page(&mut self) {
        if self.page_count > 0 {
            self.scroll_to_page = Some(0);
        }
    }

    /// vim `G` -- scroll to the last page.
    fn go_to_last_page(&mut self) {
        if self.page_count > 0 {
            self.scroll_to_page = Some(self.page_count - 1);
        }
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
    /// get in [`KpdfApp::handle_key_image`]).
    ///
    /// Reads egui's own `zoom_delta()`, which is already exactly `1.0` (a
    /// no-op) whenever the zoom modifier (`Modifiers::COMMAND`, which is
    /// Ctrl on Linux/Windows and Cmd on macOS) is not held -- so plain
    /// scrolling of the page is completely unaffected.
    fn handle_scroll_zoom(&mut self, ctx: &egui::Context) {
        if self.mode != Mode::Image || self.cmdline.is_some() {
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

    /// Rasterize page `page` at [`THUMBNAIL_DPI`] and cache the texture,
    /// returning it. `None` only on a render failure -- the sidebar simply
    /// shows nothing for that slot, and [`KpdfApp::page_size_for_layout`]
    /// falls back to a plausible US-Letter-shaped size, rather than either
    /// erroring the whole panel.
    ///
    /// Ported from `kovan`'s `thumbnail_texture` (see the module docs):
    /// unbounded cache, cleared only when the document is replaced or edited
    /// (see [`KpdfApp::open_path`]/[`KpdfApp::reload_from_bytes`]) -- a
    /// thumbnail is tiny, so holding every page's for as long as the
    /// document stays open (in its current, edited state) costs little.
    fn thumbnail_texture(
        &mut self,
        ctx: &egui::Context,
        page: usize,
    ) -> Option<egui::TextureHandle> {
        if let Some(t) = self.thumbnails.get(&page) {
            return Some(t.clone());
        }
        let pix =
            rasterize_page_with_fallback(&self.doc, page, THUMBNAIL_DPI, self.fallback_enabled)
                .ok()?;
        let rgba = rgb_to_rgba(&pix);
        let image =
            egui::ColorImage::from_rgba_unmultiplied([pix.w as usize, pix.h as usize], &rgba);
        let tex = ctx.load_texture(
            format!("kpdf-thumb-{page}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnails.insert(page, tex.clone());
        Some(tex)
    }

    /// This page's on-screen size (and physical point size) for the
    /// continuous layout, derived from its cached thumbnail texture (see
    /// [`KpdfApp::thumbnail_texture`]) rather than a full-resolution render
    /// -- ported from `kovan`'s continuous-scroll sizing trick (see the
    /// module docs): this is what makes opening a long document cheap.
    ///
    /// The physical point size comes from [`page_size_pts`] applied to the
    /// thumbnail's own pixel size at [`THUMBNAIL_DPI`] -- that function is
    /// dpi-invariant by construction (see its own docs), so this recovers
    /// the *exact* page size in PDF points without any separate `/MediaBox`
    /// lookup.
    ///
    /// Falls back to a US-Letter-shaped size (at the current `dpi`, 612x792
    /// points) if the thumbnail itself failed to render, so the layout still
    /// gets a plausible slot instead of a zero-size one.
    fn page_size_for_layout(&mut self, ctx: &egui::Context, page: usize) -> PageSize {
        match self.thumbnail_texture(ctx, page) {
            Some(t) => {
                let [tw, th] = t.size();
                let (tw, th) = (tw as f32, th as f32);
                let scale = self.dpi / THUMBNAIL_DPI;
                let (page_w_pts, page_h_pts) = page_size_pts(tw, th, THUMBNAIL_DPI);
                PageSize {
                    display_w: tw * scale,
                    display_h: th * scale,
                    page_w_pts,
                    page_h_pts,
                }
            }
            None => {
                let scale = self.dpi / 72.0;
                PageSize {
                    display_w: 612.0 * scale,
                    display_h: 792.0 * scale,
                    page_w_pts: 612.0,
                    page_h_pts: 792.0,
                }
            }
        }
    }

    /// The continuous layout's per-page slots, rebuilt only when
    /// [`SlotsCacheKey`] actually changes (dpi, page count, how many
    /// thumbnails have been rasterized so far, or the fallback toggle) --
    /// not on every frame, since a long document's slot list is otherwise
    /// non-trivial `Vec` churn to rebuild 20 times a second for nothing.
    fn continuous_slots(&mut self, ctx: &egui::Context) -> Vec<ContinuousSlot> {
        let key: SlotsCacheKey = (
            self.dpi.to_bits(),
            self.page_count,
            self.thumbnails.len(),
            self.fallback_enabled,
        );
        if let Some((cached_key, slots)) = &self.slots_cache
            && *cached_key == key
        {
            return slots.clone();
        }
        let sizes: Vec<PageSize> = (0..self.page_count)
            .map(|p| self.page_size_for_layout(ctx, p))
            .collect();
        let slots = layout_continuous_pages(&sizes, CONTINUOUS_GAP);
        self.slots_cache = Some((key, slots.clone()));
        slots
    }

    /// Rasterize page `page` at the current `dpi` (with the current
    /// fallback setting) for the continuous view, and cache the texture --
    /// see [`PageTextureKey`]/[`PAGE_TEXTURE_CACHE_CAPACITY`]. Bounded,
    /// unlike [`KpdfApp::thumbnail_texture`]: a full-resolution page is
    /// several megabytes, so an unbounded cache over a long document would
    /// reproduce the exact gh-88 memory concern this pass exists to close.
    fn ensure_page_texture(
        &mut self,
        ctx: &egui::Context,
        page: usize,
    ) -> Option<egui::TextureHandle> {
        let key: PageTextureKey = (page, self.dpi.to_bits(), self.fallback_enabled);
        if let Some(evicted) = self.page_textures_lru.touch(key) {
            self.page_textures.remove(&evicted);
        }
        if let Some(t) = self.page_textures.get(&key) {
            return Some(t.clone());
        }
        match rasterize_page_with_fallback(&self.doc, page, self.dpi, self.fallback_enabled) {
            Ok(pix) => {
                let rgba = rgb_to_rgba(&pix);
                let image = egui::ColorImage::from_rgba_unmultiplied(
                    [pix.w as usize, pix.h as usize],
                    &rgba,
                );
                let tex = ctx.load_texture(
                    format!("kpdf-page-{page}"),
                    image,
                    egui::TextureOptions::LINEAR,
                );
                self.page_textures.insert(key, tex.clone());
                self.status = None;
                Some(tex)
            }
            Err(e) => {
                self.status = Some(format!("rasterize page {}: {e}", page + 1));
                None
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
    /// Also drops any in-progress form-field edit and ink stroke: leaving
    /// forms mode (which this always does -- see [`select_tool`]) means
    /// there is no longer any path for the in-place editor to be interacted
    /// with, and switching away from the pen tool mid-drag should not leave
    /// a half-drawn stroke waiting to be revived by switching back.
    fn select_tool(&mut self, requested: Tool) {
        select_tool(&mut self.tool, &mut self.forms_mode, requested);
        self.form_edit = None;
        self.form_edit_focus_pending = false;
        self.draw_stroke.clear();
        self.draw_page = None;
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
    /// invalidate everything derived from the old one (textures, thumbnails,
    /// form-field cache). On failure, report it in the status bar rather
    /// than losing the edit silently -- the document keeps showing its
    /// pre-edit state.
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
    /// cache keyed on the old document's *content* -- page count/position
    /// are left alone (every edit this app makes preserves them), but
    /// full-resolution textures, thumbnails, the continuous-layout slot
    /// cache, and the form-fields cache all potentially show stale pixels
    /// or values from before the edit, so all four are cleared
    /// unconditionally.
    ///
    /// Clearing `thumbnails` (rather than only the touched page's entry)
    /// means every page's thumbnail is re-rasterized after any single edit
    /// -- correct, but more work than strictly necessary for a large
    /// document; a per-page-touched invalidation would need `apply_edit`'s
    /// callers to report which page they touched, which none currently do.
    /// Flagged as a follow-up rather than engineered here.
    fn reload_from_bytes(&mut self, bytes: Vec<u8>) {
        match PdfDocument::open(bytes) {
            Ok(doc) => {
                self.doc = doc;
                self.page_textures.clear();
                self.page_textures_lru = Lru::new(PAGE_TEXTURE_CACHE_CAPACITY);
                self.thumbnails.clear();
                self.slots_cache = None;
                self.annot_count = drawable_annot_count(&self.doc, self.page);
                self.status = None;
                self.form_fields_cache.clear();
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
    ///
    /// Distinct from [`KpdfApp::save_in_place`] (`Ctrl+S`), which overwrites
    /// the currently open file directly.
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

    /// `Ctrl+S`, `:w`, or the toolbar's "Save in place" button: overwrite
    /// `self.path` directly with the current (possibly edited) bytes.
    ///
    /// A no-op when there is nothing to save -- mirrors
    /// [`EditHistory::can_undo`]'s own "has this document actually diverged
    /// from what was opened" contract, the same gate the Save-as button and
    /// dialog use, so this never overwrites the original file with
    /// byte-identical content for no reason.
    ///
    /// **Written atomically**: the new bytes go to a temp file in the *same
    /// directory* as `self.path` (guaranteeing the rename below is on one
    /// filesystem, which atomicity requires), then [`std::fs::rename`]
    /// swaps it into place. This never truncates-then-writes the real file
    /// directly -- a crash or power loss mid-write would otherwise leave a
    /// half-written, corrupted document where the user's annotated PDF used
    /// to be. `std::fs::rename` maps to `rename(2)` on POSIX and
    /// `MoveFileExW` (without `MOVEFILE_COPY_ALLOWED`) on Windows, both
    /// atomic within one filesystem.
    ///
    /// Any failure (writing the temp file, or the rename itself) is
    /// reported to the status bar, never panics; a failed rename also tries
    /// to clean up the now-orphaned temp file (best-effort -- that cleanup's
    /// own failure must not mask the original error).
    fn save_in_place(&mut self) {
        let Some(hist) = &self.edit_history else {
            return; // nothing has ever been edited this session.
        };
        if !hist.can_undo() {
            return; // no unsaved changes since the document was opened.
        }
        let bytes = hist.current().to_vec();
        let dir = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        let file_name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "document.pdf".to_string());
        let tmp_path = dir.join(format!(".{file_name}.kpdf-tmp"));
        if let Err(e) = std::fs::write(&tmp_path, &bytes) {
            self.status = Some(format!("save {}: {e}", self.path.display()));
            return;
        }
        match std::fs::rename(&tmp_path, &self.path) {
            Ok(()) => self.status = Some(format!("saved {}", self.path.display())),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                self.status = Some(format!("save {}: {e}", self.path.display()));
            }
        }
    }

    /// Forms-mode click handling: resolve the click to a page (via
    /// [`screen_to_page_at`]), hit-test against that page's form fields, and
    /// either toggle a checkbox/radio in place or open the in-place
    /// text-field editor. All PDF-structure knowledge (what `/AS`, `/V`, an
    /// on-state name are) stays in `kopitiam_pdf::mupdf::form` -- this only
    /// converts a click to page coordinates and dispatches.
    ///
    /// `origin` is the content column's on-screen top-left for this frame
    /// (see the module docs' "coordinate trap" section); `slots` is this
    /// frame's continuous layout.
    fn handle_forms_click(
        &mut self,
        response: &egui::Response,
        origin: egui::Pos2,
        slots: &[ContinuousSlot],
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if self.form_edit.is_some() {
            // A click that the in-place editor itself didn't consume (it
            // sits in front of the page image, so a click *on* it never
            // reaches here) is exactly the "clicked away" case -- commit
            // whatever is currently typed rather than silently discarding
            // it. `commit_form_edit` is idempotent (`Option::take`), so if
            // the editor's own `lost_focus` check already committed it this
            // same frame, this is a harmless no-op.
            self.commit_form_edit();
        }
        let content_x = pos.x - origin.x;
        let content_y = pos.y - origin.y;
        let Some((page, px, py)) = screen_to_page_at(content_x, content_y, slots) else {
            return;
        };
        let Some(slot) = slots.get(page) else {
            return;
        };
        let layout = PageLayout {
            image_x: origin.x,
            image_y: origin.y + slot.top,
            image_w: slot.width,
            image_h: slot.height,
            page_w_pts: slot.page_w_pts,
            page_h_pts: slot.page_h_pts,
        };
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, page);
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
                self.form_edit = Some((page, field.obj_num, field.value.clone()));
                self.form_edit_focus_pending = true;
            }
            _ => self.status = Some(format!("{}: unsupported field kind for kpdf", field.name)),
        }
    }

    /// Pen-tool preview only: if a stroke is in progress on `page`, paint its
    /// live preview line using `layout` (this page's on-screen placement
    /// this frame). Called once per **visible** page inside the continuous
    /// view's per-slot loop, where each page's own [`PageLayout`] is
    /// naturally at hand and the loop's own child `Ui`/painter (with the
    /// right clip rect) is what has to do the painting -- split out from
    /// [`KpdfApp::handle_draw_update`], which reacts to the pointer using
    /// the one shared background `Response` available only *after* that
    /// loop finishes, and so cannot also be the thing painting into it.
    fn handle_draw_preview_only(&self, ui: &egui::Ui, page: usize, layout: PageLayout) {
        if self.draw_page != Some(page) || self.draw_stroke.len() < 2 {
            return;
        }
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

    /// Pen-tool drag handling (the pointer-reacting half -- see
    /// [`KpdfApp::handle_draw_preview_only`] for the paired preview-paint
    /// half, run separately per visible page): resolve every drag point to a
    /// page (an ink stroke cannot span two PDF pages -- see
    /// [`KpdfApp::draw_page`]), accumulate it into [`KpdfApp::draw_stroke`]
    /// (already in that page's own default user space), and commit it as a
    /// real ink annotation on release via
    /// [`kopitiam_pdf::mupdf::annot_edit::add_ink_annot`].
    fn handle_draw_update(
        &mut self,
        response: &egui::Response,
        origin: egui::Pos2,
        slots: &[ContinuousSlot],
    ) {
        if response.dragged()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let content_x = pos.x - origin.x;
            let content_y = pos.y - origin.y;
            if let Some((page, px, py)) = screen_to_page_at(content_x, content_y, slots) {
                match self.draw_page {
                    None => {
                        self.draw_page = Some(page);
                        self.draw_stroke.push((px, py));
                    }
                    Some(p) if p == page => self.draw_stroke.push((px, py)),
                    // Strayed onto a different page's row -- this point
                    // does not belong to the in-progress stroke.
                    Some(_) => {}
                }
            }
        }

        if response.drag_stopped() {
            let page = self.draw_page.take();
            if !self.draw_stroke.is_empty() {
                if let Some(page) = page {
                    let points = std::mem::take(&mut self.draw_stroke);
                    let spec = InkAnnotSpec {
                        page_index: page,
                        strokes: vec![InkStroke { points }],
                        color: INK_COLOR,
                        width: INK_WIDTH,
                        opacity: INK_OPACITY,
                        author: None,
                    };
                    let result = kopitiam_pdf::mupdf::annot_edit::add_ink_annot(&self.doc, &spec);
                    self.apply_edit(result);
                } else {
                    self.draw_stroke.clear();
                }
            }
        }
    }

    /// Eraser-tool click/drag handling: resolve the pointer to a page (via
    /// [`screen_to_page_at`]), hit-test that page's annotations, and delete
    /// whichever one the pointer lands on, via
    /// [`kopitiam_pdf::mupdf::annot_edit::delete_annot`].
    fn handle_erase(
        &mut self,
        response: &egui::Response,
        origin: egui::Pos2,
        slots: &[ContinuousSlot],
    ) {
        if !(response.clicked() || response.dragged()) {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let content_x = pos.x - origin.x;
        let content_y = pos.y - origin.y;
        let Some((page, px, py)) = screen_to_page_at(content_x, content_y, slots) else {
            return;
        };
        let refs = kopitiam_pdf::mupdf::annot_edit::page_annot_refs(&self.doc, page);
        let Some(num) = hit_test_annot(px, py, &refs) else {
            return;
        };
        let result = kopitiam_pdf::mupdf::annot_edit::delete_annot(&self.doc, page, num);
        self.apply_edit(result);
    }

    /// Populate (or reuse) `form_fields_cache` for `page`. Mirrors
    /// [`KpdfApp::ensure_page_texture`]'s caching shape for the rasterised
    /// texture: see [`CachedField`]'s docs for why this exists at all.
    fn refresh_form_fields_cache(&mut self, page: usize) {
        if self.form_fields_cache.contains_key(&page) {
            return;
        }
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, page)
            .iter()
            .map(|f| CachedField {
                obj_num: f.obj_num,
                kind: f.kind,
                rect: f.rect,
                read_only: f.read_only,
                multiline: f.multiline,
            })
            .collect();
        self.form_fields_cache.insert(page, fields);
    }

    /// Commit the in-place text editor's current buffer via
    /// [`kopitiam_pdf::mupdf::form::set_field_value`], then follow the same
    /// push-to-history/reopen/invalidate tail every edit in this file goes
    /// through ([`KpdfApp::apply_edit`]).
    ///
    /// A no-op if there is nothing to commit -- `Option::take` makes this
    /// safe to call from more than one place in the same frame. Both
    /// [`KpdfApp::paint_and_edit_form_fields`]'s own `lost_focus` check and
    /// [`KpdfApp::handle_forms_click`]'s "clicked away" guard can reach this
    /// for the very same click; only the first one to run does anything.
    fn commit_form_edit(&mut self) {
        let Some((page, obj_num, value)) = self.form_edit.take() else {
            return;
        };
        let fields = kopitiam_pdf::mupdf::form::page_form_fields(&self.doc, page);
        if let Some(field) = fields.iter().find(|f| f.obj_num == obj_num) {
            let result = kopitiam_pdf::mupdf::form::set_field_value(&self.doc, field, &value);
            self.apply_edit(result);
        } else {
            self.status = Some("field no longer present on this page".to_string());
        }
    }

    /// Paint the Okular-style "fillable area" overlay for `page` (one call
    /// per *visible* page, per frame -- see `Mode::Image`'s rendering) for
    /// every field [`field_highlight_kind`] says gets one, and -- if a text
    /// field on `page` specifically is currently being edited
    /// ([`KpdfApp::form_edit`]) -- draw the in-place `egui::TextEdit` over
    /// that field's own rect instead of a highlight for it.
    ///
    /// `layout` is `page`'s on-screen [`PageLayout`] for this frame (built
    /// from its continuous-layout slot plus the content column's current
    /// screen-space origin), never cached -- panning or zooming moves the
    /// overlay and the editor with the page underneath them on the very next
    /// frame. `clip` bounds both the painted rectangles and the editor to
    /// the viewport, so a field scrolled out of view doesn't bleed over
    /// neighbouring pages or the toolbar.
    fn paint_and_edit_form_fields(
        &mut self,
        ui: &mut egui::Ui,
        page: usize,
        layout: PageLayout,
        clip: egui::Rect,
    ) {
        let editing_here = self
            .form_edit
            .as_ref()
            .filter(|(p, ..)| *p == page)
            .map(|(_, n, _)| *n);

        if let Some(fields) = self.form_fields_cache.get(&page) {
            let painter = ui.painter_at(clip);
            for field in fields {
                if Some(field.obj_num) == editing_here {
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

        let Some(obj_num) = editing_here else {
            return;
        };
        let field_meta = self.form_fields_cache.get(&page).and_then(|fields| {
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
        // multiline field -- "enter saves the thing. Shift-Enter gets a new
        // line". `TextEdit::multiline`'s own default is "any Enter is a
        // newline" with no notion of shift, so the shiftless-Enter event has
        // to be pulled out of the input queue *before* the widget runs
        // below.
        let enter_commit = multiline && consume_commit_enter(ui);

        let prev_clip = ui.clip_rect();
        ui.set_clip_rect(clip);
        let mut commit = false;
        if let Some((_, _, buf)) = self.form_edit.as_mut() {
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
        let (events, ctrl_c, ctrl_s, ctrl_shift_s) = ctx.input(|i| {
            let ctrl_c = i.modifiers.ctrl && i.key_pressed(egui::Key::C);
            // Ctrl+S and Ctrl+Shift+S are DIFFERENT commands, so Ctrl+S must
            // check that Shift is *not* held. Without that guard the in-place
            // save would also fire on the Save-As chord and silently overwrite
            // the original file -- exactly what Save As exists to avoid.
            let ctrl_s = i.modifiers.ctrl && !i.modifiers.shift && i.key_pressed(egui::Key::S);
            let ctrl_shift_s = i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::S);
            (i.events.clone(), ctrl_c, ctrl_s, ctrl_shift_s)
        });
        if ctrl_c {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if ctrl_s {
            self.save_in_place();
        }
        if ctrl_shift_s {
            self.save_via_dialog();
        }

        for event in &events {
            let egui::Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } = event
            else {
                continue;
            };
            let key = *key;
            let modifiers = *modifiers;

            // Every vim-motion/nav key below is only ever reached when
            // `keys_captured` says nothing is capturing text -- the *gate*
            // is this one predicate; each capture kind's own specific
            // handling (digit entry vs. just Escape-to-cancel) still
            // differs beneath it. See `keys_captured`'s docs: a `j` typed
            // into a form field must insert `j`, never scroll.
            if keys_captured(self.cmdline.is_some(), self.form_edit.is_some()) {
                if self.form_edit.is_some() {
                    if key == egui::Key::Escape {
                        self.form_edit = None;
                    }
                } else if self.cmdline.is_some() {
                    match key {
                        egui::Key::Escape => self.cmdline = None,
                        egui::Key::Enter => {
                            let buf = self.cmdline.take().unwrap_or_default();
                            match parse_command(&buf) {
                                Command::GotoPage(n) => self.goto_page_1based(n),
                                Command::Write => self.save_in_place(),
                                Command::Empty => {}
                                Command::Unknown(what) => {
                                    // Say so rather than doing nothing: a
                                    // command line that silently ignores input
                                    // leaves the user unable to tell "not a
                                    // command" from "command failed".
                                    self.status = Some(format!("not a command: :{what}"));
                                }
                            }
                        }
                        egui::Key::Backspace => {
                            if let Some(buf) = self.cmdline.as_mut() {
                                buf.pop();
                            }
                        }
                        _ => {
                            // Digits for `:N`, letters for `:w`. Anything else
                            // (arrows, function keys) is ignored rather than
                            // stuffed into the buffer.
                            if let Some(c) = cmdline_char(key)
                                && let Some(buf) = self.cmdline.as_mut()
                            {
                                buf.push(c);
                            }
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
                    Mode::Image => self.handle_key_image(key, modifiers),
                    Mode::Reflow => self.handle_key_reflow(key),
                },
            }
        }
    }

    fn handle_key_image(&mut self, key: egui::Key, modifiers: egui::Modifiers) {
        // A pending `g` (armed by a previous frame's lone `g`, waiting for a
        // second one to complete `gg`) that has aged past the timeout, or
        // that a key other than `G` just interrupted, is dropped -- see
        // `GPending`'s docs. Checked before the match below so a stale `g`
        // can never combine with an unrelated later `g`.
        if self.g_pending.is_armed() {
            let stale = self
                .g_armed_at
                .is_some_and(|t| g_pending_expired(t.elapsed()));
            if stale || key != egui::Key::G {
                self.g_pending.cancel();
                self.g_armed_at = None;
            }
        }

        match key {
            egui::Key::R => self.mode = Mode::Reflow,
            egui::Key::N | egui::Key::ArrowRight | egui::Key::PageDown => self.next_page(),
            egui::Key::P | egui::Key::ArrowLeft | egui::Key::PageUp => self.prev_page(),
            egui::Key::Plus | egui::Key::Equals => self.zoom_in(),
            egui::Key::Minus => self.zoom_out(),
            // `:N<Enter>` -- vim's own line-number-jump idiom, reused here
            // for "go to page N". Bound to `:` rather than a bare `g` so `g`
            // is free for vim's own `gg`/`G` below.
            egui::Key::Colon => self.cmdline = Some(String::new()),
            // vim `G` -- go to the last page. Shift is what makes this `G`
            // rather than lower-case `g` on a real keyboard; egui reports
            // the same `Key::G` either way; the modifier is how the two are
            // told apart here.
            egui::Key::G if modifiers.shift => {
                self.go_to_last_page();
                self.g_pending.cancel();
                self.g_armed_at = None;
            }
            // vim `gg` -- go to the first page, on the second `g` of the
            // sequence.
            egui::Key::G => {
                if self.g_pending.press_g() {
                    self.go_to_first_page();
                    self.g_armed_at = None;
                } else {
                    self.g_armed_at = Some(Instant::now());
                }
            }
            // vim h/j/k/l -- a small nudge, applied via
            // `egui::Ui::scroll_with_delta` inside the scroll area (see
            // `Mode::Image`'s rendering). Down/right are a *negative* delta
            // and up/left a *positive* one -- `egui::Ui::scroll_with_delta`'s
            // own convention (scrolling down moves the *content* up).
            egui::Key::H => self.pending_scroll_delta.x += VIM_STEP,
            egui::Key::L => self.pending_scroll_delta.x -= VIM_STEP,
            egui::Key::J => self.pending_scroll_delta.y -= VIM_STEP,
            egui::Key::K => self.pending_scroll_delta.y += VIM_STEP,
            // Ctrl+d / Ctrl+u -- half a viewport, vim's own contract.
            egui::Key::D if modifiers.ctrl => {
                self.pending_scroll_delta.y -= half_viewport_step(self.last_viewport_h);
            }
            egui::Key::U if modifiers.ctrl => {
                self.pending_scroll_delta.y += half_viewport_step(self.last_viewport_h);
            }
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

    /// The left page-thumbnail sidebar (toggled via the toolbar's
    /// "Hide/Show pages" button, [`KpdfApp::hide_thumbnails`]) -- an
    /// Okular-style page picker. Ported from `kovan`'s `thumbnail_strip`
    /// (see the module docs): `show_rows` so only the thumbnails actually
    /// scrolled into view are rasterized/uploaded, rather than every page in
    /// the document up front (the sidebar's own scroll position is
    /// independent of the main continuous view's).
    fn thumbnail_sidebar(&mut self, ui: &mut egui::Ui) {
        let page_count = self.page_count;
        let row_height = 96.0;
        let current = self.page;
        egui::ScrollArea::vertical()
            .id_salt("kpdf_thumb_sidebar")
            .show_rows(ui, row_height, page_count, |ui, range| {
                for page in range {
                    let selected = page == current;
                    let frame = egui::Frame::new().inner_margin(4.0).fill(if selected {
                        egui::Color32::from_rgb(60, 90, 140)
                    } else {
                        egui::Color32::TRANSPARENT
                    });
                    let resp = frame.show(ui, |ui| {
                        ui.set_height(row_height - 8.0);
                        ui.vertical_centered(|ui| {
                            if let Some(tex) = self.thumbnail_texture(ui.ctx(), page) {
                                let [tw, th] = tex.size();
                                let aspect = th as f32 / (tw as f32).max(1.0);
                                let w = 72.0_f32;
                                ui.add(
                                    egui::Image::new(&tex)
                                        .fit_to_exact_size(egui::vec2(w, w * aspect)),
                                );
                            }
                            ui.label(format!("{}", page + 1));
                        });
                    });
                    if ui
                        .interact(
                            resp.response.rect,
                            ui.id().with(("kpdf-thumb", page)),
                            egui::Sense::click(),
                        )
                        .clicked()
                    {
                        self.scroll_to_page = Some(page);
                    }
                }
            });
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

/// The character a key contributes to the `:` command line, or `None` for a
/// key that contributes nothing (arrows, function keys, modifiers).
///
/// Accepts digits for `:N` page jumps and lowercase letters for named commands
/// like `:w`. Letters are lowercased because `egui` reports the physical key,
/// not the shifted character -- and vim's commands are lowercase anyway, so
/// folding here means `:w` works whether or not Caps Lock is on, while
/// [`parse_command`] can still reject a genuinely different command.
fn cmdline_char(key: egui::Key) -> Option<char> {
    use egui::Key::*;
    Some(match key {
        Num0 => '0',
        Num1 => '1',
        Num2 => '2',
        Num3 => '3',
        Num4 => '4',
        Num5 => '5',
        Num6 => '6',
        Num7 => '7',
        Num8 => '8',
        Num9 => '9',
        A => 'a',
        B => 'b',
        C => 'c',
        D => 'd',
        E => 'e',
        F => 'f',
        G => 'g',
        H => 'h',
        I => 'i',
        J => 'j',
        K => 'k',
        L => 'l',
        M => 'm',
        N => 'n',
        O => 'o',
        P => 'p',
        Q => 'q',
        R => 'r',
        S => 's',
        T => 't',
        U => 'u',
        V => 'v',
        W => 'w',
        X => 'x',
        Y => 'y',
        Z => 'z',
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
                    // Toolbar: Open/Undo/Redo/Save/Save-in-place, then the
                    // annotation tools, then Forms mode (only for a document
                    // that has one), then the fallback toggle and the
                    // sidebar visibility toggle -- see the module docs'
                    // "Annotation tools and forms mode" / "left page
                    // sidebar" sections for what each does.
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
                        .on_hover_text("Save as... (Ctrl+Shift+S) -- writes a new file, never touches the original")
                        .clicked()
                    {
                        self.save_via_dialog();
                    }
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Save in place"))
                        .on_hover_text("Overwrite the open file (Ctrl+S)")
                        .clicked()
                    {
                        self.save_in_place();
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
                    ui.separator();
                    if ui
                        .selectable_label(self.fallback_enabled, "Fallback")
                        .on_hover_text(
                            "Cross-engine (hayro) graceful fallback for undecodable glyphs -- \
                             turn off to see kopitiam's own engine unconditionally",
                        )
                        .clicked()
                    {
                        self.fallback_enabled = !self.fallback_enabled;
                    }
                    if self.page_count > 1 {
                        ui.separator();
                        let label = if self.hide_thumbnails {
                            "Show pages"
                        } else {
                            "Hide pages"
                        };
                        if ui.button(label).clicked() {
                            self.hide_thumbnails = !self.hide_thumbnails;
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
                            // annotations via `rasterize_page_with_fallback` like
                            // everything else, but a GUI cannot be verified
                            // headlessly -- so showing the count turns "are
                            // annotations working?" into something a human can
                            // check at a glance.
                            let n = self.annot_count;
                            if n > 0 {
                                ui.separator();
                                ui.colored_label(
                                    egui::Color32::LIGHT_BLUE,
                                    format!("{n} annot{}", if n == 1 { "" } else { "s" }),
                                );
                            }
                            if let Some(buf) = &self.cmdline {
                                ui.separator();
                                ui.colored_label(egui::Color32::GOLD, format!(":{buf}"));
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

        if self.mode == Mode::Image && self.page_count > 1 && !self.hide_thumbnails {
            egui::Panel::left("kpdf_thumbnails")
                .resizable(true)
                .default_size(110.0)
                .show(ui, |ui| self.thumbnail_sidebar(ui));
        }

        egui::CentralPanel::default().show(ui, |ui| match self.mode {
            Mode::Image => {
                let slots = self.continuous_slots(ui.ctx());
                let total_h = slots.last().map(ContinuousSlot::bottom).unwrap_or(0.0);
                let total_w = slots.iter().fold(0.0f32, |m, s| m.max(s.width)).max(1.0);
                let pending_delta = std::mem::take(&mut self.pending_scroll_delta);
                let scroll_target = self.scroll_to_page.take();

                let output = egui::ScrollArea::both()
                    .id_salt("kpdf_continuous_scroll")
                    .show(ui, |ui| {
                        if pending_delta != egui::Vec2::ZERO {
                            ui.scroll_with_delta(pending_delta);
                        }

                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(total_w, total_h.max(1.0)),
                            egui::Sense::click_and_drag(),
                        );
                        let origin = rect.min;
                        let viewport = ui.clip_rect();
                        let viewport_top = viewport.min.y - origin.y;
                        let viewport_bottom = viewport.max.y - origin.y;

                        if let Some(target) = scroll_target
                            && let Some(slot) = slots.get(target)
                        {
                            let page_rect = egui::Rect::from_min_size(
                                egui::pos2(origin.x, origin.y + slot.top),
                                egui::vec2(slot.width, slot.height),
                            );
                            ui.scroll_to_rect(page_rect, Some(egui::Align::TOP));
                        }

                        let painter = ui.painter_at(rect);
                        for slot in &slots {
                            if !continuous_slot_visible(
                                slot,
                                viewport_top,
                                viewport_bottom,
                                VISIBLE_MARGIN,
                            ) {
                                continue;
                            }
                            let page_screen_rect = egui::Rect::from_min_size(
                                origin + egui::vec2(0.0, slot.top),
                                egui::vec2(slot.width, slot.height),
                            );
                            if let Some(tex) = self.ensure_page_texture(ui.ctx(), slot.page_index) {
                                painter.image(
                                    tex.id(),
                                    page_screen_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                            } else {
                                painter.rect_filled(
                                    page_screen_rect,
                                    0.0,
                                    egui::Color32::from_gray(235),
                                );
                            }

                            let layout = PageLayout {
                                image_x: page_screen_rect.min.x,
                                image_y: page_screen_rect.min.y,
                                image_w: slot.width,
                                image_h: slot.height,
                                page_w_pts: slot.page_w_pts,
                                page_h_pts: slot.page_h_pts,
                            };

                            if self.forms_mode {
                                self.refresh_form_fields_cache(slot.page_index);
                                self.paint_and_edit_form_fields(
                                    ui,
                                    slot.page_index,
                                    layout,
                                    viewport,
                                );
                            } else if self.tool == Tool::Draw {
                                self.handle_draw_preview_only(ui, slot.page_index, layout);
                            }
                        }

                        (response, origin, viewport_top, viewport_bottom)
                    });

                self.last_viewport_h = output.inner_rect.height();
                let (response, origin, viewport_top, viewport_bottom) = output.inner;

                if let Some(cur) = current_page_in_view(&slots, viewport_top, viewport_bottom) {
                    self.page = cur;
                }
                self.annot_count = drawable_annot_count(&self.doc, self.page);

                // Route the click/drag to whichever tool is active. All
                // three share the same coordinate conversion
                // (`screen_to_page_at`) -- see the module docs' "coordinate
                // trap" section -- and all business logic (what a hit
                // means, how to edit the PDF) lives in
                // `kopitiam_pdf::mupdf::{annot_edit,form}`, not here.
                if self.forms_mode {
                    self.handle_forms_click(&response, origin, &slots);
                } else {
                    match self.tool {
                        Tool::Pan => {}
                        Tool::Draw => {
                            // The preview line was already painted per-page
                            // above (`handle_draw_preview_only`); this call
                            // only updates `draw_stroke`/`draw_page` and
                            // commits on release.
                            self.handle_draw_update(&response, origin, &slots);
                        }
                        Tool::Erase => self.handle_erase(&response, origin, &slots),
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
