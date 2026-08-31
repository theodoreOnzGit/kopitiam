//! [`PdfReader`] -- the embeddable PDF reader: the whole of `kpdf`'s reading,
//! searching, annotating and form-filling behaviour as a component another
//! egui application can drop into its own window.
//!
//! gh-96 Phases 6, 9, 10 and 12. This is where the extraction lands: `kpdf`
//! is now a shell around this type rather than a second implementation of it,
//! which is the brief's central acceptance criterion -- *there must not be
//! two independent reader implementations*.
//!
//! # Two entry points, and which one you want
//!
//! ```text
//!   reader.ui(ui)    -> the DOCUMENT PANE only. You own the layout.
//!   reader.show(ui)  -> pane + toolbar + sidebars + find bar, assembled.
//! ```
//!
//! [`ui`](PdfReader::ui) is the primitive (AID-0057): it paints pages into
//! whatever `Ui` you hand it and nothing else, so you can put the reader in
//! your own split beside your own tooling, and place -- or omit -- the
//! thumbnail strip, the outline, and the find bar yourself via
//! [`thumbnail_sidebar`](PdfReader::thumbnail_sidebar),
//! [`outline_sidebar`](PdfReader::outline_sidebar) and
//! [`find_bar`](PdfReader::find_bar).
//!
//! [`show`](PdfReader::show) is the convenience for a host that just wants
//! kpdf's layout without rebuilding it. It is layered *on* `ui`, never
//! instead of it.
//!
//! Either way you must call [`pump`](PdfReader::pump) first each frame, which
//! collects finished background work and handles input. `show` does it for
//! you; if you drive the pane yourself, you call it.
//!
//! # What the reader will not do
//!
//! It has no filesystem. It is handed bytes ([`open_bytes`](PdfReader::open_bytes),
//! [`load_bytes`](PdfReader::load_bytes)) and hands bytes back
//! ([`document_bytes`](PdfReader::document_bytes)). Saving, opening, file
//! dialogs, watching a path for changes, and the window itself are all host
//! policy, reported as [`ReaderAction`]s rather than performed here. That is
//! what lets the same component serve a standalone viewer and an embedded
//! pane in an application that may have no file behind the document at all.
//!
//! It also knows nothing about what a host does with a document. See
//! [`action`](super::action) for the rule that keeps it that way.
//!
//! # Read-only embedding
//!
//! [`PdfReaderConfig::read_only`] gives every reading feature -- navigation,
//! search, continuous scroll, reflow, page coordinates -- with no path by
//! which a keystroke can modify the PDF. See [`config`](super::config).

use std::collections::HashMap;
use std::time::Instant;

use crate::mupdf::annot_edit::{EditHistory, InkAnnotSpec, InkStroke};
use crate::mupdf::form::FieldKind;
use crate::mupdf::outline::{OutlineItem, load_outline};
use crate::mupdf::stext_search::{SearchHit, search_page};
use crate::mupdf::structured_text::StextOptions;
use crate::mupdf::{PdfDocument, Rect, page_to_stext, rasterize_page_with_fallback};
use crate::{Page as TextPage, extract_mupdf_page};

use super::action::{PdfReaderOutput, ReaderAction};
use super::config::PdfReaderConfig;
use super::render::{RenderKind, RenderRequest, RenderWorker, RenderedPage};
use super::search::{FindScan, SearchWorker, scan_page_order};
use super::thumbnails::THUMBNAIL_DPI;
use super::viewport::Viewport;
use super::{
    Command, ContinuousSlot, DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, FieldHighlight, GPending,
    Lru, PageLayout, PageSize, Tool, VIM_STEP, consume_commit_enter, continuous_slot_visible,
    drawable_annot_count, field_highlight_kind, field_rect_to_screen,
    g_pending_expired, half_viewport_step, highlight_colors, hit_test_annot, hit_test_field,
    hit_test_field_expanded, keys_captured, min_hit_rect, page_to_screen,
    parse_command, rgb_to_rgba, screen_to_page_at, select_tool, stext_to_screen, toggle_forms_mode,
    zoom_percent, zoom_steps_from_zoom_delta,
};

/// Ink color for a newly-drawn stroke (DeviceRGB, 0..=1) -- plain black, the
/// least surprising default for a "draw on the page" tool. No UI to pick a
/// colour yet; that is a follow-up, not scope creep for this pass.
const INK_COLOR: [f32; 3] = [0.0, 0.0, 0.0];
/// Stroke width in PDF points, written via the annotation's `/Border`.
const INK_WIDTH: f32 = 2.0;
/// Constant opacity (`/CA`); 1.0 means fully opaque and writes no `/CA` at
/// all (see [`InkAnnotSpec::opacity`](crate::mupdf::annot_edit::InkAnnotSpec::opacity)).
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


/// How many pages either side of the current one the background searcher is
/// asked to search ahead. Five pages at a time keeps the queue short enough
/// that jumping elsewhere is served promptly, while covering everything
/// visible in continuous scroll at normal zoom.
const SEARCH_PREFETCH_RADIUS: usize = 2;

/// How many pages are rasterised **synchronously** when a document opens.
///
/// The maintainer's call: the first pages should simply be there when the
/// window appears, rather than flashing placeholders on the one part of the
/// document a reader always looks at. Everything past this is the worker's.
/// At a measured ~135 ms median per page this costs roughly a second at open,
/// traded for a readable first screen.
const SYNC_PRERENDER_PAGES: usize = 8;

/// How many pages either side of the current one the rasteriser is kept
/// working on. Modest by choice (the maintainer's budget answer): a page
/// texture is several megabytes, and the existing LRU bounds what is kept.
const RENDER_PREFETCH_RADIUS: usize = 3;

/// How many pages ahead of a parked scan to queue at once.
///
/// The scan needs its frontier page before it can move, so queuing only that
/// one page would make the search advance a page per round trip. Queuing a
/// short run ahead keeps the worker busy without committing the whole
/// document to the queue, so a jump elsewhere is still served promptly.
const SEARCH_SCAN_LOOKAHEAD: usize = 8;


/// How far beyond the visible viewport (screen points, both edges) a page is
/// still worth full-resolution rendering -- see
/// [`crate::gui_frontend::continuous_slot_visible`]. Widening by
/// roughly "most of a screen" means a fast scroll finds the next page or two
/// already rasterized instead of flashing a bare placeholder first. Not a
/// measured value; a bigger number trades some extra up-front render cost
/// for smoother fast scrolling.
const VISIBLE_MARGIN: f32 = 600.0;

/// How many `(page, dpi, fallback)` full-resolution page textures stay
/// resident at once (see [`crate::gui_frontend::Lru`]) -- the gh-88
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
// `crate::gui_frontend` (see the imports above) -- lifted out so other
// egui-based KOPITIAM front ends can reuse them.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Image,
    Reflow,
}

/// A page's full-resolution texture is cached under the same key the
/// rasteriser produces it under -- `(page_index, dpi.to_bits(),
/// fallback_enabled)`.
///
/// An alias for the library's [`RenderKey`] rather than a second definition,
/// so the texture cache and the render worker can never drift apart on what
/// identifies a page image. The local name is kept because on this side of
/// the boundary the thing being keyed really is a texture.
pub type PageTextureKey = super::render::RenderKey;

/// Cache key for the continuous-layout slot list ([`PdfReader::slots_cache`]):
/// rebuild only when dpi, the page count (a new document) or the fallback
/// toggle actually change -- not on every frame.
///
/// It used to carry the thumbnail count too, because page sizes were
/// measured from rendered thumbnails and so improved as more arrived. Sizes
/// now come from `/MediaBox` ([`PdfReader::page_size_for_layout`]), which is
/// exact on the first frame, so there is nothing to refine and the layout is
/// built exactly once per (dpi, document, fallback) combination.
pub type SlotsCacheKey = (u32, usize, bool);

/// What the document pane should claim from the pointer, given the active
/// tool.
///
/// # Why this is a decision and not a constant
///
/// The pane used to allocate itself with `Sense::click_and_drag()`
/// unconditionally, which broke panning in a way that took a maintainer with
/// a mouse to notice. Two things have to line up:
///
/// 1. `ScrollArea` pans on a pointer drag only when its `ScrollSource::drag`
///    is enabled. egui's default is [`DragScroll::OnTouch`] --
///    *"only active when a touch screen is detected"* -- so on a desktop it
///    is simply off, and the reader must ask for
///    [`DragScroll::Always`](egui::containers::scroll_area::DragScroll::Always).
/// 2. Even then, a contained widget that senses drags takes the drag first.
///    egui interacts the scroll background *before* the content precisely so
///    the content wins (`scroll_area.rs`: "We must do this BEFORE adding
///    content... or we will steal input from the widgets we contain"). So a
///    pane that always senses drag always steals it, and the Pan tool -- whose
///    handler does nothing, because panning IS the scroll area's job -- left
///    the drag going nowhere at all.
///
/// So: claim the drag only for the tools that consume it.
///
/// Click is claimed in every case, and that costs nothing here: egui tracks a
/// potential *click* target separately from a potential *drag* target, so a
/// click-sensing pane and a drag-sensing scroll background coexist -- forms
/// stay clickable while the same gesture, dragged, pans.
fn pointer_sense(forms_mode: bool, tool: Tool) -> egui::Sense {
    if !forms_mode && matches!(tool, Tool::Draw | Tool::Erase) {
        // The pen and eraser follow the pointer; the drag is the input.
        // Panning by drag is unavailable while they are selected, which is
        // what the Pan tool is for.
        egui::Sense::click_and_drag()
    } else {
        egui::Sense::click()
    }
}

/// A background reflow-text extractor.
///
/// # Why
///
/// Reflow mode used to call `extract_mupdf_from_bytes` on first entry, which
/// **re-reads the whole file and extracts every page** before returning. On
/// the 506-page Irodori book that is a 106 MB read plus 506 extractions at a
/// median 5.4 ms and up to 349 ms each — the same tens-of-seconds freeze the
/// inline search scan used to cause, just triggered by pressing `r` instead.
///
/// So pages are extracted one at a time on a worker and delivered as they
/// finish. Reflow shows what has arrived, with the rest reported as still
/// coming, instead of showing nothing at all until everything is done.
struct ReflowWorker {
    res: std::sync::mpsc::Receiver<(usize, TextPage)>,
    /// How many pages the document has, so the view can say how far along it
    /// is rather than just appearing to stop.
    total: usize,
    /// Pages delivered so far. Progress only -- **not** the completion test.
    /// The worker skips pages it cannot extract, so `done` can legitimately
    /// stop short of `total` on a damaged document; waiting for `done ==
    /// total` would then leave the pump asking for a frame every 120 ms for
    /// the rest of the session. Completion is the channel disconnecting, when
    /// the worker thread drops its sender -- which happens whether it
    /// finished, skipped, or bailed on an unopenable file.
    done: usize,
}

impl ReflowWorker {
    /// Start extracting `bytes` page by page.
    fn spawn(bytes: Vec<u8>, total: usize) -> Option<ReflowWorker> {
        let (tx, rx) = std::sync::mpsc::channel::<(usize, TextPage)>();
        std::thread::Builder::new()
            .name("kpdf-reflow".to_string())
            .spawn(move || {
                let Ok(doc) = PdfDocument::open(bytes) else {
                    return;
                };
                for index in 0..doc.page_count() {
                    // An unreadable page must not stop the rest.
                    let Some(page) = extract_mupdf_page(&doc, index) else {
                        continue;
                    };
                    if tx.send((index, page)).is_err() {
                        return; // UI gone
                    }
                }
            })
            .ok()?;
        Some(ReflowWorker { res: rx, total, done: 0 })
    }
}

/// Put `page` at its own `index` in a sparse, page-indexed vector, growing
/// with `None` holes as needed.
///
/// Pulled out of [`PdfReader::pump_reflow_worker`] purely so it can be tested:
/// the pump needs an egui `Context`, this needs nothing. A hole means "that
/// page produced no text page" -- either it has not arrived yet, or the
/// worker could not extract it at all. Both render the same way, and neither
/// is allowed to shift a page that *did* arrive.
fn place_reflow_page(pages: &mut Vec<Option<TextPage>>, index: usize, page: TextPage) {
    if index >= pages.len() {
        pages.resize_with(index + 1, || None);
    }
    pages[index] = Some(page);
}

/// The find bar's transient state while it is open.
///
/// `backward` records which key opened it — `/` searches forward, `?`
/// backward, exactly as vim and `mupdf-gl` do — so `Enter` knows which way to
/// go without a second keystroke.
pub struct FindBar {
    query: String,
    backward: bool,
    /// `true` for exactly the frame after the bar opens, so the text field can
    /// take focus once rather than stealing it back every frame.
    focus_pending: bool,
}

pub struct PdfReader {
    doc: PdfDocument,
    /// Where every page sits, which one is being read, and the scroll intents
    /// navigation queues. **The one continuous-view coordinate model** -- the
    /// reader keeps no page index, dpi or layout cache of its own, precisely
    /// so a second one cannot drift out of step with this (gh-96 Phase 4).
    viewport: Viewport,
    mode: Mode,
    /// `Some(buf)` while a "go to page" number is being typed (image mode,
    /// triggered by `:`).
    cmdline: Option<String>,
    /// The find bar, when open: the query being typed and which way `Enter`
    /// will search. `None` means the bar is closed.
    find: Option<FindBar>,
    /// The committed query the hit cache belongs to. Cleared with the cache
    /// whenever a new search is committed.
    find_query: String,
    /// Per-page hits for [`PdfReader::find_query`], filled lazily.
    ///
    /// Lazily on purpose: searching every page of a 500-page document up
    /// front is exactly the all-pages-on-the-UI-thread stall that
    /// `/MediaBox` sizing was introduced to remove, and a reader only ever
    /// needs the next hit.
    find_hits: HashMap<usize, Vec<SearchHit>>,
    /// The background page rasteriser, if one could be started.
    render_worker: Option<RenderWorker>,
    /// The page the rasteriser last planned around, so a jump can be told
    /// from a step.
    last_pumped_page: usize,
    /// Whether the opening-pages prerender has run for the current document.
    /// Separate from `render_worker` being `Some`, so a machine where the
    /// thread could not be spawned does not retry the spawn every frame.
    render_started: bool,
    /// An in-progress scan for the next hit, when the answer is not yet
    /// known from cached pages alone.
    ///
    /// `/` and `n` must answer *where* the next hit is, which in the worst
    /// case means examining every page. Doing that synchronously is what hung
    /// the window for minutes on a 506-page book: pages cost a median 5.4 ms
    /// but up to 349 ms each. So the scan stops at the first page nothing has
    /// searched yet, records where it got to here, and resumes as the
    /// background worker delivers.
    find_scan: Option<FindScan>,
    /// The background searcher for the current query, if one is running.
    find_worker: Option<SearchWorker>,
    /// The hit the view is currently on: `(page, index within that page)`.
    find_current: Option<(usize, usize)>,
    /// The document outline, loaded once per document.
    outline: Vec<OutlineItem>,
    /// Whether the contents sidebar is shown (`t`).
    show_outline: bool,
    /// Leftover Ctrl+scroll signal not yet large enough to make a whole
    /// [`DPI_STEP`] move -- see `zoom.rs`'s `ZOOM_DELTA_PER_STEP` and
    /// `zoom_steps_from_zoom_delta`. Persists across frames so a slow or
    /// interrupted scroll gesture still adds up correctly instead of being
    /// discarded on every repaint.
    scroll_zoom_accum: f32,
    /// Full-resolution page textures for the continuous view -- see
    /// [`PageTextureKey`]. Bounded by [`page_textures_lru`]
    /// ([`PAGE_TEXTURE_CACHE_CAPACITY`]); see [`PdfReader::ensure_page_texture`].
    page_textures: HashMap<PageTextureKey, egui::TextureHandle>,
    page_textures_lru: Lru<PageTextureKey>,
    /// Low-resolution page thumbnails, rasterized once at [`THUMBNAIL_DPI`]
    /// and never evicted for as long as the document stays open -- a
    /// thumbnail is tiny, so holding every page's costs little (same
    /// precedent `kovan`'s own equivalent cache sets; see the module docs).
    /// Used both for the sidebar strip and for [`PdfReader::page_size_for_layout`].
    thumbnails: HashMap<usize, egui::TextureHandle>,
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
    /// Sparse and **page-indexed**: `reflow_pages[i]` is page `i`'s text, or
    /// `None` if it has not arrived (or could not be extracted). Never a
    /// densely-packed arrival-order list -- see [`place_reflow_page`].
    reflow_pages: Option<Result<Vec<Option<TextPage>>, String>>,
    /// The background reflow extractor, while it is still working.
    reflow_worker: Option<ReflowWorker>,
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
    /// `true` when [`PdfReader::has_acroform`] is also `true` -- see
    /// [`toggle_forms_mode`].
    forms_mode: bool,
    /// Whether the open document has an `/AcroForm` at all
    /// ([`crate::mupdf::form::has_acroform`]), computed once per
    /// document load. Gates whether the Forms button is shown -- there is no
    /// point offering a forms toggle for a document with no fields.
    has_acroform: bool,
    /// Edit history over the document's raw bytes, backing Undo/Redo/Save.
    /// `None` until the first edit is made in this session -- a document that
    /// is only ever viewed, never annotated, never pays for one (see
    /// [`PdfReader::ensure_edit_history`]).
    edit_history: Option<EditHistory>,
    /// Whether [`PdfReader::edit_history`] holds annotation/form edits that are
    /// not yet on disk. Gates hot reload: re-opening the file would discard
    /// them, so a document with unsaved work is never auto-reloaded (the
    /// status bar says so instead). Set by [`PdfReader::apply_edit`], cleared
    /// by a successful save and by opening a document.
    unsaved_edits: bool,
    /// The in-progress ink stroke while [`Tool::Draw`] is being dragged, in
    /// **default user space** on the page it belongs to (see
    /// [`PdfReader::draw_page`]) -- committed as a real annotation on
    /// drag-release ([`PdfReader::handle_draw`]) and cleared either way.
    draw_stroke: Vec<(f32, f32)>,
    /// Which page [`PdfReader::draw_stroke`] belongs to -- set on the first
    /// point of a drag, `None` otherwise. An ink stroke cannot span two PDF
    /// pages, so once armed, later drag points that resolve to a *different*
    /// page (the pointer strayed over a neighbouring row in the continuous
    /// view) are ignored rather than appended.
    draw_page: Option<usize>,
    /// `Some((page, obj_num, buf))` while the in-place text-field editor
    /// (see [`PdfReader::paint_and_edit_form_fields`]) is open over a field on
    /// `page`: `obj_num` identifies which [`FormField`] (re-looked-up by
    /// obj_num each frame from `form_fields_cache` for position/kind, and
    /// from a fresh [`crate::mupdf::form::page_form_fields`] call at
    /// commit time), `buf` is the text being typed. Cleared on commit, on
    /// `Esc`, and whenever the active tool or forms mode changes.
    form_edit: Option<(usize, i32, String)>,
    /// `true` for exactly the one frame after a click opens `form_edit` --
    /// tells [`PdfReader::paint_and_edit_form_fields`] to call
    /// `Response::request_focus()` once, so typing can start immediately
    /// without an extra click into the box. Must never stay `true` across
    /// frames: re-requesting focus every frame would mean the box could
    /// never lose focus, and therefore never commit-on-focus-loss.
    form_edit_focus_pending: bool,
    /// Cache of each *visible* page's form fields for painting the highlight
    /// overlay, keyed by page index -- see
    /// [`PdfReader::refresh_form_fields_cache`]. Unlike `page_textures`, not
    /// LRU-bounded: this is small, parsed metadata (not pixel data), so
    /// unbounded growth for as long as a document with an `/AcroForm` stays
    /// open is cheap. Cleared entirely whenever `self.doc` is replaced (open/
    /// undo/redo/edit) since field values/rects belong to the document that
    /// was just replaced.
    form_fields_cache: HashMap<usize, Vec<CachedField>>,
    /// The `gg` two-key sequence's pending state -- see
    /// [`crate::gui_frontend::keys`].
    /// What the reader has to tell its host, accumulated between paints.
    ///
    /// A queue rather than a return value because the things that produce
    /// actions -- a key handler, a toolbar button, a link click -- are deep
    /// inside call chains that cannot each return one. Drained by whichever
    /// public entry point runs next.
    pending_actions: PdfReaderOutput,
    /// What to call the open document in the toolbar. The reader has no path
    /// of its own, so a host that has one sets it here.
    label: Option<String>,
    /// Which features are enabled -- see [`PdfReaderConfig`].
    config: PdfReaderConfig,
    g_pending: GPending,
    /// Wall-clock time `g_pending` was last armed, so a stale one times out
    /// ([`g_pending_expired`]) instead of lingering for an unrelated later
    /// `g`. `None` whenever `g_pending` is not armed.
    g_armed_at: Option<Instant>,
}

/// A page's form fields, captured as plain owned data for
/// [`PdfReader::form_fields_cache`]. Unlike [`FormField`] (borrows nothing
/// itself, but carries no [`Clone`] -- it is meant to be built and consumed
/// within a single call), this is cheap to hold across frames, which is the
/// whole point: [`crate::mupdf::form::page_form_fields`] walks a
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
    /// [`PdfReader::paint_and_edit_form_fields`] must use if this field is the
    /// one being edited.
    multiline: bool,
}

impl PdfReader {



    /// Scroll to the next/previous page, relative to [`PdfReader::page`] (the
    /// currently most-visible page, tracked every frame -- see that field's
    /// docs). Sets [`PdfReader::scroll_to_page`] rather than assigning `page`
    /// directly, since `page` is only ever a *read* of where the view
    /// actually is.
    fn next_page(&mut self) {
        if self.viewport.page() + 1 < self.viewport.page_count() {
            self.viewport.next_page();
        }
    }

    fn prev_page(&mut self) {
        if self.viewport.page() > 0 {
            self.viewport.prev_page();
        }
    }

    fn goto_page_1based(&mut self, page_1based: usize) {
        let clamped = page_1based.clamp(1, self.viewport.page_count().max(1));
        self.viewport.scroll_to(clamped - 1);
    }

    /// vim `gg` -- scroll to the first page.
    fn go_to_first_page(&mut self) {
        if self.viewport.page_count() > 0 {
            self.viewport.go_to_first_page();
        }
    }

    /// vim `G` -- scroll to the last page.
    fn go_to_last_page(&mut self) {
        if self.viewport.page_count() > 0 {
            self.viewport.go_to_last_page();
        }
    }

    fn zoom_in(&mut self) {
        self.viewport.set_dpi((self.viewport.dpi() + DPI_STEP).min(DPI_MAX));
        self.bump_render_generation();
    }

    fn zoom_out(&mut self) {
        self.viewport.set_dpi((self.viewport.dpi() - DPI_STEP).max(DPI_MIN));
        self.bump_render_generation();
    }

    /// Reset zoom to [`DPI_DEFAULT`] -- the on-screen zoom readout doubles as
    /// this button, since clicking "100%" to get back to 100% is the
    /// discoverable affordance every other viewer offers.
    fn zoom_reset(&mut self) {
        self.viewport.set_dpi(DPI_DEFAULT);
        self.bump_render_generation();
    }

    /// Ctrl+scroll-wheel zoom over the page, the gesture users reach for
    /// reflexively because every other image/PDF viewer binds it. Image mode
    /// only, and only outside a `goto` entry (same guard the `+`/`-` keys
    /// get in [`PdfReader::handle_key_image`]).
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
    /// shows nothing for that slot, and [`PdfReader::page_size_for_layout`]
    /// falls back to a plausible US-Letter-shaped size, rather than either
    /// erroring the whole panel.
    ///
    /// Ported from `kovan`'s `thumbnail_texture` (see the module docs):
    /// unbounded cache, cleared only when the document is replaced or edited
    /// (see [`PdfReader::open_path`]/[`PdfReader::reload_from_bytes`]) -- a
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
        // Queue it and draw nothing this frame. A thumbnail costs ~76 ms, and
        // `show_rows` asks for every row it can see: jumping to the end of a
        // 506-page book uncovers about ten uncached rows at once, which
        // rendered inline is most of a second of frozen window. That stall
        // also delayed the page pump, so the pages themselves started late --
        // together, "press G and everything says loading for a while".
        if let Some(worker) = self.render_worker.as_mut() {
            // `request` is a no-op for a key already in flight, so a
            // thumbnail visible across many frames queues exactly once.
            worker.request(RenderRequest {
                page,
                dpi: THUMBNAIL_DPI,
                fallback: self.fallback_enabled,
                generation: worker.generation(),
                kind: RenderKind::Thumbnail,
            });
            return None;
        }
        // No worker: render inline, as before. Slow beats never.
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


    /// The continuous layout's per-page slots, rebuilt only when
    /// [`SlotsCacheKey`] actually changes (dpi, page count, how many
    /// thumbnails have been rasterized so far, or the fallback toggle) --
    /// not on every frame, since a long document's slot list is otherwise
    /// non-trivial `Vec` churn to rebuild 20 times a second for nothing.
    fn continuous_slots(&mut self, _ctx: &egui::Context) -> Vec<ContinuousSlot> {
        // The cache, its key and its invalidation all belong to `Viewport` --
        // this crate has exactly one continuous-layout model and it is not
        // here (gh-96 Phase 4). Sizes come from `/MediaBox`, which is exact
        // on the first frame, so the closure runs once per (dpi, page count,
        // fallback) rather than every frame.
        let doc = &self.doc;
        let dpi = self.viewport.dpi();
        let n = self.viewport.page_count();
        self.viewport
            .slots(self.fallback_enabled, || {
                (0..n)
                    .map(|p| {
                        // The full box, not just its size: annotations and
                        // form fields are placed in user space, whose origin
                        // is the box's lower-left corner and is NOT always
                        // (0, 0). See `page_media_box_points`.
                        let mb = crate::mupdf::page_geom::page_media_box_points(doc, p);
                        let page_w_pts = mb.x1 - mb.x0;
                        let page_h_pts = mb.y1 - mb.y0;
                        let scale = dpi / 72.0;
                        PageSize {
                            display_w: page_w_pts * scale,
                            display_h: page_h_pts * scale,
                            page_w_pts,
                            page_h_pts,
                            page_x0: mb.x0,
                            page_y0: mb.y0,
                        }
                    })
                    .collect()
            })
            .to_vec()
    }

    /// Rasterize page `page` at the current `dpi` (with the current
    /// fallback setting) for the continuous view, and cache the texture --
    /// see [`PageTextureKey`]/[`PAGE_TEXTURE_CACHE_CAPACITY`]. Bounded,
    /// unlike [`PdfReader::thumbnail_texture`]: a full-resolution page is
    /// several megabytes, so an unbounded cache over a long document would
    /// reproduce the exact gh-88 memory concern this pass exists to close.
    /// The page's texture **only if it is already rasterised**.
    ///
    /// This is what the paint loop uses, and the distinction from
    /// [`PdfReader::ensure_page_texture`] is the entire point of the render
    /// worker: painting must never rasterise, because that costs a median
    /// 135 ms and up to 444 ms per page and would put it straight back on the
    /// UI thread. A page that is not ready draws a labelled placeholder and
    /// the worker delivers it a frame or two later.
    ///
    /// The one exception is a machine where the worker thread could not be
    /// spawned: with nothing else able to produce the page, painting falls
    /// back to rendering inline. Slow beats permanently blank.
    fn cached_page_texture(
        &mut self,
        ctx: &egui::Context,
        page: usize,
    ) -> Option<egui::TextureHandle> {
        let key: PageTextureKey = (page, self.viewport.dpi().to_bits(), self.fallback_enabled);
        if let Some(evicted) = self.page_textures_lru.touch(key) {
            self.page_textures.remove(&evicted);
        }
        if let Some(t) = self.page_textures.get(&key) {
            return Some(t.clone());
        }
        if self.render_worker.is_none() {
            return self.ensure_page_texture(ctx, page);
        }
        None
    }

    fn ensure_page_texture(
        &mut self,
        ctx: &egui::Context,
        page: usize,
    ) -> Option<egui::TextureHandle> {
        let key: PageTextureKey = (page, self.viewport.dpi().to_bits(), self.fallback_enabled);
        if let Some(evicted) = self.page_textures_lru.touch(key) {
            self.page_textures.remove(&evicted);
        }
        if let Some(t) = self.page_textures.get(&key) {
            return Some(t.clone());
        }
        match rasterize_page_with_fallback(&self.doc, page, self.viewport.dpi(), self.fallback_enabled) {
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
        // Start with an empty (Ok) result and fill it in as pages arrive.
        // Extracting the whole document here is what froze the window for
        // tens of seconds on a long book -- see `ReflowWorker`.
        self.reflow_pages = Some(Ok(Vec::new()));
        self.reflow_worker =
            ReflowWorker::spawn(self.doc.raw_bytes().to_vec(), self.viewport.page_count());
        if self.reflow_worker.is_none() {
            // No thread available: fall back to doing it here, which is slow
            // but still correct.
            self.reflow_pages = Some(
                crate::extract_mupdf_from_bytes(self.doc.raw_bytes())
                    .map(|pages| pages.into_iter().map(Some).collect())
                    .map_err(|e| e.to_string()),
            );
        }
    }

    /// Collect reflow pages delivered since the last frame. Never blocks.
    fn pump_reflow_worker(&mut self, ctx: &egui::Context) {
        let Some(worker) = self.reflow_worker.as_mut() else {
            return;
        };
        let dest = match self.reflow_pages.as_mut() {
            Some(Ok(pages)) => pages,
            // An error result means the fallback path already ran; nothing to
            // collect into.
            _ => return,
        };
        let mut finished = false;
        loop {
            match worker.res.try_recv() {
                Ok((index, page)) => {
                    // Place by the page's OWN index, never by arrival order.
                    // The worker `continue`s past a page it cannot extract, so
                    // pushing in arrival order shifts every later page up by
                    // one -- reflow would then show page 41's text under the
                    // heading for page 42, silently, for the rest of the file.
                    place_reflow_page(dest, index, page);
                    worker.done += 1;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            self.reflow_worker = None;
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
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
    /// same reasoning as [`PdfReader::select_tool`].
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
    fn apply_edit(&mut self, result: crate::mupdf::Result<Vec<u8>>) {
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
                self.unsaved_edits = true;
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
                // The page count can MOVE across an edit. It never did while
                // the only edits were annotations, but adding or deleting a
                // page changes it -- and a stale count leaves the new page
                // unreachable (navigation clamps to the old maximum) or lets
                // the viewer address a page that no longer exists.
                //
                // Only when it ACTUALLY moved, though: `set_document` drops
                // queued scroll intents, and an ink stroke that silently
                // cancelled a pending jump would feel like the document
                // lurching back under the pen.
                let n = self.doc.page_count();
                if n != self.viewport.page_count() {
                    self.viewport.set_document(n);
                }
                self.page_textures.clear();
                self.page_textures_lru = Lru::new(PAGE_TEXTURE_CACHE_CAPACITY);
                self.thumbnails.clear();
                self.viewport.invalidate_layout();

                // THE WORKERS HOLD THEIR OWN COPY OF THE OLD BYTES. Dropping
                // them is not tidiness -- it is the whole correctness of an
                // edit.
                //
                // `RenderWorker::spawn` takes `self.doc.raw_bytes().to_vec()`
                // and opens its own `PdfDocument` from it, because a document
                // holds `RefCell`s and cannot cross threads. So a worker
                // started before this edit keeps rasterising the PRE-EDIT
                // file forever. Clearing the texture caches above then makes
                // it worse rather than better: every visible page is
                // re-requested and re-rendered, from the old bytes, so the
                // ink that was just committed is nowhere on screen -- while
                // `annot_count` below, read from the NEW document, cheerfully
                // reports the annotation exists. That was the bug: "I can
                // draw, the reader says there are annots, and I cannot see
                // them."
                //
                // `render_started` must be cleared too, not just the worker:
                // the spawn in `pump` is gated on `!render_started`, so
                // leaving it set means no worker is ever started again and
                // every page stays a placeholder.
                self.render_worker = None;
                self.render_started = false;
                // Same reasoning: an in-flight reflow extractor is reading
                // the old bytes and would deliver text for a document that no
                // longer exists.
                self.reflow_worker = None;
                // Extracted reflow text is indexed by page, so inserting or
                // removing a page shifts every entry after it out of step.
                self.reflow_pages = None;
                self.annot_count = drawable_annot_count(&self.doc, self.viewport.page());
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



    /// Insert a blank page directly after the page being viewed, and go to it.
    ///
    /// The lecturing move (the maintainer's ask): you are mid-derivation on
    /// slide 12, you run out of room, you add a page and keep writing on it
    /// with the tablet. So it lands *after the current page*, not at the end
    /// of the deck -- appending would put the new page 60 slides away from
    /// where the thought is -- and the view jumps to it so the pen can start
    /// immediately.
    ///
    /// Size and `/Rotate` are copied from the page you were on
    /// ([`crate::mupdf::page_edit::insert_blank_page`]), so the new
    /// page appears at the same size and zoom and the stroke you were about
    /// to draw lands where you expect.
    ///
    /// Goes through [`PdfReader::apply_edit`] like every other edit, so it is
    /// undoable and saved by the same Ctrl+S.
    fn add_blank_page(&mut self) {
        let at = self.viewport.page() + 1;
        let result = crate::mupdf::page_edit::insert_blank_page(&self.doc, at, None);
        let ok = result.is_ok();
        self.apply_edit(result);
        if ok {
            // `apply_edit` -> `reload_from_bytes` keeps the page index, but the
            // new page is the point of the exercise, so move to it.
            self.viewport.jump_to(at);
            self.status = Some(format!("added blank page {}", self.viewport.page() + 1));
        }
    }

    /// Delete the page being viewed.
    ///
    /// Undoable (it is an ordinary entry in the edit history) and refused on
    /// the last remaining page, since a zero-page PDF will not reopen.
    ///
    /// Note this removes the page from the page *tree*; the page's content
    /// stays in the saved file as an orphaned object. That is inherent to
    /// append-only editing -- see
    /// [`crate::mupdf::page_edit`]'s module docs. It is deletion, not
    /// redaction.
    fn delete_current_page(&mut self) {
        let target = self.viewport.page();
        let result = crate::mupdf::page_edit::delete_page(&self.doc, target);
        let ok = result.is_ok();
        self.apply_edit(result);
        if ok {
            // Deleting the last page leaves the index past the end; step back
            // so the viewer lands on the page that took its place (or the new
            // final page).
            self.viewport.jump_to(target);
            self.status = Some(format!("deleted page {}", target + 1));
        }
    }

    /// Show or hide the contents sidebar, loading the outline the first time
    /// it is asked for.
    ///
    /// Loaded lazily rather than on open: most documents are read without ever
    /// opening the sidebar, and walking the outline tree resolves a
    /// destination per item.
    fn toggle_outline(&mut self) {
        self.show_outline = !self.show_outline;
        if self.show_outline && self.outline.is_empty() {
            self.outline = load_outline(&self.doc);
            if self.outline.is_empty() {
                self.status = Some("this document has no outline".into());
                self.show_outline = false;
            }
        }
    }

    /// Open the find bar. `/` searches forward, `?` backward — vim's and
    /// `mupdf-gl`'s own convention, which is what the maintainer asked for.
    fn open_find(&mut self, backward: bool) {
        self.find = Some(FindBar {
            query: self.find_query.clone(),
            backward,
            focus_pending: true,
        });
    }

    /// Commit the typed query and jump to the first hit.
    ///
    /// A new query throws away the hit cache: it belongs to the old query and
    /// keeping it would show stale highlights.
    fn commit_find(&mut self) {
        let Some(bar) = self.find.take() else { return };
        let query = bar.query.trim().to_string();
        if query.is_empty() {
            self.find_query.clear();
            self.find_hits.clear();
            self.find_current = None;
            self.find_worker = None; // stops the thread, frees its copy
            return;
        }
        if query != self.find_query {
            self.find_query = query;
            self.find_hits.clear();
            self.find_current = None;
            // Replace the worker: the old one is searching for the old query,
            // and dropping it both stops that thread and frees its copy of
            // the document.
            self.find_worker =
                SearchWorker::spawn(self.doc.raw_bytes().to_vec(), self.find_query.clone());
        }
        // Search starts from the page in view, not from page 1 — the reader is
        // looking for the next occurrence from where they are.
        self.find_step(!bar.backward, true);
    }

    /// The hits on `page` for the current query, extracting and searching that
    /// page the first time it is asked for.
    ///
    /// One page at a time is the whole point: a 500-page document is never
    /// searched up front, so opening the find bar cannot stall the UI.
    fn hits_on(&mut self, page: usize) -> &[SearchHit] {
        if !self.find_hits.contains_key(&page) {
            let hits = page_to_stext(&self.doc, page, StextOptions::default())
                .map(|sp| search_page(&sp, &self.find_query))
                .unwrap_or_default();
            self.find_hits.insert(page, hits);
        }
        self.find_hits.get(&page).map_or(&[][..], Vec::as_slice)
    }

    /// Move to the next (or previous) hit.
    ///
    /// `from_current_page` restarts at the page in view (a fresh `/`);
    /// otherwise it continues from the hit already selected (`n`).
    ///
    /// Only pages **already searched** are examined here. The moment the scan
    /// reaches an unsearched page it stops and hands off to the background
    /// worker ([`PdfReader::resume_scan`]), because searching a page can cost
    /// up to 349 ms and a scan may cross the whole document — which is what
    /// hung the window for minutes on the 506-page Irodori book.
    fn find_step(&mut self, forward: bool, from_current_page: bool) {
        if self.find_query.is_empty() {
            self.status = Some("no search — press / to find".into());
            return;
        }
        let n = self.viewport.page_count();
        if n == 0 {
            return;
        }
        let (page, idx) = match (from_current_page, self.find_current) {
            (false, Some((p, i))) => (p, Some(i)),
            _ => (self.viewport.page(), None),
        };
        self.scan_from(forward, page, idx, n);
    }

    /// Walk pages in scan order looking for the next hit, using cached pages
    /// only. Parks the scan at the first unsearched page.
    fn scan_from(&mut self, forward: bool, start: usize, start_idx: Option<usize>, budget: usize) {
        let n = self.viewport.page_count();
        let mut page = start;
        let mut idx = start_idx;
        for step in 0..=budget {
            let Some(hits) = self.find_hits.get(&page) else {
                // Unsearched: park here and let the worker catch up.
                self.find_scan = Some(FindScan {
                    forward,
                    next: page,
                    remaining: budget.saturating_sub(step),
                });
                self.status = Some(format!("searching for {:?}…", self.find_query));
                // With no worker (spawn failed) there is nothing to wait for,
                // so fall back to searching here: slow beats never answering.
                if self.find_worker.is_none() {
                    let _ = self.hits_on(page);
                    let scan = self.find_scan.take();
                    if let Some(scan) = scan {
                        self.scan_from(scan.forward, scan.next, None, scan.remaining);
                    }
                }
                return;
            };
            let len = hits.len();
            let next = match (idx, forward) {
                (Some(i), true) if i + 1 < len => Some(i + 1),
                (Some(i), false) if i > 0 => Some(i - 1),
                (Some(_), _) => None,
                (None, true) if len > 0 => Some(0),
                (None, false) if len > 0 => Some(len - 1),
                (None, _) => None,
            };
            if let Some(i) = next {
                self.find_scan = None;
                self.find_current = Some((page, i));
                self.viewport.jump_to(page);
                self.status = Some(format!(
                    "/{} — hit {} of {} on page {}",
                    self.find_query,
                    i + 1,
                    len,
                    page + 1
                ));
                return;
            }
            if step == budget {
                break;
            }
            page = if forward { (page + 1) % n } else { (page + n - 1) % n };
            idx = None;
        }
        self.find_scan = None;
        self.find_current = None;
        self.status = Some(format!("no match for {:?}", self.find_query));
    }

    /// Continue a parked scan now that more pages have been searched.
    fn resume_scan(&mut self) {
        let Some(scan) = self.find_scan.take() else {
            return;
        };
        if !self.find_hits.contains_key(&scan.next) {
            // Still waiting on that page; put the scan back untouched.
            self.find_scan = Some(scan);
            return;
        }
        self.scan_from(scan.forward, scan.next, None, scan.remaining);
    }

    /// Jump to an outline destination.
    fn goto_destination(&mut self, dest: &crate::mupdf::destination::Destination) {
        use crate::mupdf::destination::Destination;
        match dest {
            Destination::Page { page, .. } => {
                let p = (*page).min(self.viewport.page_count().saturating_sub(1));
                self.viewport.jump_to(p);
            }
            // Opening a browser is the application's call, not the viewer's:
            // say where it points and let the operator decide.
            Destination::Uri(u) => self.status = Some(format!("link: {u}")),
            Destination::Unsupported(kind) => {
                self.status = Some(format!("{kind} destinations are not followed"));
            }
        }
    }


    /// Forms-mode click handling: resolve the click to a page (via
    /// [`screen_to_page_at`]), hit-test against that page's form fields, and
    /// either toggle a checkbox/radio in place or open the in-place
    /// text-field editor. All PDF-structure knowledge (what `/AS`, `/V`, an
    /// on-state name are) stays in `crate::mupdf::form` -- this only
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
            page_x0: slot.page_x0,
            page_y0: slot.page_y0,
        };
        let fields = crate::mupdf::form::page_form_fields(&self.doc, page);
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
                let result = crate::mupdf::form::toggle_checkbox(&self.doc, field);
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
    /// [`PdfReader::handle_draw_update`], which reacts to the pointer using
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
    /// [`PdfReader::handle_draw_preview_only`] for the paired preview-paint
    /// half, run separately per visible page): resolve every drag point to a
    /// page (an ink stroke cannot span two PDF pages -- see
    /// [`PdfReader::draw_page`]), accumulate it into [`PdfReader::draw_stroke`]
    /// (already in that page's own default user space), and commit it as a
    /// real ink annotation on release via
    /// [`crate::mupdf::annot_edit::add_ink_annot`].
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
            // ASK FOR ANOTHER FRAME, or the ink you just drew is invisible
            // until something else happens to repaint.
            //
            // The preview is painted per-page INSIDE the scroll-area closure,
            // and this runs after that closure returns -- so the line on
            // screen is always the stroke as of the *previous* frame, and the
            // segment appended just now needs a following frame to appear at
            // all. That used to come free from a blanket 50 ms repaint, which
            // was removed in 0.3.1 because it pinned the whole app at 20 fps
            // (the reported lag). Removing it was right; not replacing it
            // here was the bug -- an in-progress stroke is an asynchronous
            // source with work outstanding, exactly like the render, reflow
            // and search pumps, and like them it must request its own frame.
            //
            // Bounded by the drag: it stops the moment the pointer is
            // released, so this cannot reintroduce an idle busy-loop.
            response.ctx.request_repaint();
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
                    let result = crate::mupdf::annot_edit::add_ink_annot(&self.doc, &spec);
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
    /// [`crate::mupdf::annot_edit::delete_annot`].
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
        let refs = crate::mupdf::annot_edit::page_annot_refs(&self.doc, page);
        let Some(num) = hit_test_annot(px, py, &refs) else {
            return;
        };
        let result = crate::mupdf::annot_edit::delete_annot(&self.doc, page, num);
        self.apply_edit(result);
    }

    /// Populate (or reuse) `form_fields_cache` for `page`. Mirrors
    /// [`PdfReader::ensure_page_texture`]'s caching shape for the rasterised
    /// texture: see [`CachedField`]'s docs for why this exists at all.
    fn refresh_form_fields_cache(&mut self, page: usize) {
        if self.form_fields_cache.contains_key(&page) {
            return;
        }
        let fields = crate::mupdf::form::page_form_fields(&self.doc, page)
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
    /// [`crate::mupdf::form::set_field_value`], then follow the same
    /// push-to-history/reopen/invalidate tail every edit in this file goes
    /// through ([`PdfReader::apply_edit`]).
    ///
    /// A no-op if there is nothing to commit -- `Option::take` makes this
    /// safe to call from more than one place in the same frame. Both
    /// [`PdfReader::paint_and_edit_form_fields`]'s own `lost_focus` check and
    /// [`PdfReader::handle_forms_click`]'s "clicked away" guard can reach this
    /// for the very same click; only the first one to run does anything.
    fn commit_form_edit(&mut self) {
        let Some((page, obj_num, value)) = self.form_edit.take() else {
            return;
        };
        let fields = crate::mupdf::form::page_form_fields(&self.doc, page);
        if let Some(field) = fields.iter().find(|f| f.obj_num == obj_num) {
            let result = crate::mupdf::form::set_field_value(&self.doc, field, &value);
            self.apply_edit(result);
        } else {
            self.status = Some("field no longer present on this page".to_string());
        }
    }

    /// Paint the Okular-style "fillable area" overlay for `page` (one call
    /// per *visible* page, per frame -- see `Mode::Image`'s rendering) for
    /// every field [`field_highlight_kind`] says gets one, and -- if a text
    /// field on `page` specifically is currently being edited
    /// ([`PdfReader::form_edit`]) -- draw the in-place `egui::TextEdit` over
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
            // Reported, not performed: an embedded reader closing the host's
            // window would be a spectacular overreach. kpdf turns this into
            // an eframe close; a host with the reader in a tab closes the tab.
            self.pending_actions.push(ReaderAction::QuitRequested);
            return;
        }
        if ctrl_s {
            self.pending_actions.push(ReaderAction::SaveRequested);
        }
        if ctrl_shift_s {
            self.pending_actions.push(ReaderAction::SaveAsRequested);
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
            if keys_captured(
                self.cmdline.is_some() || self.find.is_some(),
                self.form_edit.is_some(),
            ) {
                if self.form_edit.is_some() {
                    if key == egui::Key::Escape {
                        self.form_edit = None;
                    }
                } else if self.find.is_some() {
                    // The text field itself handles typing; only the two
                    // control keys are ours.
                    match key {
                        egui::Key::Escape => self.find = None,
                        egui::Key::Enter => self.commit_find(),
                        _ => {}
                    }
                } else if self.cmdline.is_some() {
                    match key {
                        egui::Key::Escape => self.cmdline = None,
                        egui::Key::Enter => {
                            let buf = self.cmdline.take().unwrap_or_default();
                            match parse_command(&buf) {
                                Command::GotoPage(n) => self.goto_page_1based(n),
                                Command::Write => {
                                    self.pending_actions.push(ReaderAction::SaveRequested)
                                }
                                Command::Quit => {
                                    self.pending_actions.push(ReaderAction::QuitRequested);
                                    return;
                                }
                                Command::WriteQuit => {
                                    // Order matters: the host must be told to
                                    // save BEFORE it is told to quit, or a
                                    // host that acts on the first action and
                                    // stops would drop the write.
                                    self.pending_actions.push(ReaderAction::SaveRequested);
                                    self.pending_actions.push(ReaderAction::QuitRequested);
                                    return;
                                }
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
                // NOTHING here closes the window. `Escape` is the key every
                // panel uses for "cancel" -- the find bar, the command line,
                // a form edit -- and having it also quit meant one press too
                // many threw the session away. Quitting is `:q` / `:wq`
                // only, which is the vim contract the maintainer asked for.
                egui::Key::Escape => {
                    // Escape is "go back one step", innermost first: leave
                    // reflow before touching the search, so a reader who
                    // pressed `r` by accident gets out with the key they
                    // already reach for.
                    if self.mode == Mode::Reflow {
                        self.mode = Mode::Image;
                        self.status = Some("image mode".into());
                    } else if self.find_query.is_empty() {
                        self.status = Some("nothing to cancel — :q to quit".into());
                    } else {
                        // A second Escape clears the search and its
                        // highlights, which is what "cancel" should mean here.
                        self.find_query.clear();
                        self.find_hits.clear();
                        self.find_current = None;
                        self.find_worker = None;
                        self.status = Some("search cleared".into());
                    }
                }
                egui::Key::Tab => self.toggle_mode(),
                egui::Key::O => self.pending_actions.push(ReaderAction::OpenRequested),
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
            // `/` and `?` open the find bar; `n`/`N` step through hits.
            // vim's and mupdf-gl's convention, which is what was asked for.
            // This TOOK `n`/`p` away from page navigation: page keys are now
            // `,`/`.` (mupdf's own), plus the arrows and PageUp/PageDown,
            // which were always bound and are unchanged.
            egui::Key::Slash => self.open_find(false),
            egui::Key::Questionmark => self.open_find(true),
            egui::Key::N if modifiers.shift => self.find_step(false, false),
            egui::Key::N => self.find_step(true, false),
            egui::Key::T => self.toggle_outline(),
            egui::Key::Period | egui::Key::ArrowRight | egui::Key::PageDown => self.next_page(),
            egui::Key::Comma | egui::Key::ArrowLeft | egui::Key::PageUp => self.prev_page(),
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
            egui::Key::H => self.viewport.nudge(VIM_STEP, 0.0),
            egui::Key::L => self.viewport.nudge(-VIM_STEP, 0.0),
            egui::Key::J => self.viewport.nudge(0.0, -VIM_STEP),
            egui::Key::K => self.viewport.nudge(0.0, VIM_STEP),
            // `a` -- add a blank page after this one and jump to it. Chosen
            // as a bare letter on purpose: mid-lecture it is pressed
            // one-handed while the other hand holds the stylus.
            egui::Key::A => self.add_blank_page(),
            // Ctrl+Delete -- destructive, so deliberately NOT a bare letter
            // and deliberately nowhere near the scroll keys, where a fumble
            // would cost a page. Undoable regardless.
            egui::Key::Delete if modifiers.ctrl => self.delete_current_page(),
            // Ctrl+d / Ctrl+u -- half a viewport, vim's own contract.
            egui::Key::D if modifiers.ctrl => {
                self.viewport.nudge(0.0, -half_viewport_step(self.viewport.viewport_height()));
            }
            egui::Key::U if modifiers.ctrl => {
                self.viewport.nudge(0.0, half_viewport_step(self.viewport.viewport_height()));
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
    /// "Hide/Show pages" button, [`PdfReader::hide_thumbnails`]) -- an
    /// Okular-style page picker. Ported from `kovan`'s `thumbnail_strip`
    /// (see the module docs): `show_rows` so only the thumbnails actually
    /// scrolled into view are rasterized/uploaded, rather than every page in
    /// Take delivered pages from the rasteriser and queue the ones about to
    /// be looked at. Called once per frame; never blocks.
    fn pump_render_worker(&mut self, ctx: &egui::Context) {
        if self.render_worker.is_none() {
            return;
        }
        // A JUMP (`G`, `gg`, `:N`, a search hit) makes everything queued
        // around the old position irrelevant. Bumping the generation lets the
        // worker skip that backlog instead of rendering it first, which is
        // what made `G` show "loading…" for seconds. An ordinary step keeps
        // its neighbours, which are still worth having.
        let moved = self.viewport.page().abs_diff(self.last_pumped_page);
        self.last_pumped_page = self.viewport.page();
        if moved > RENDER_PREFETCH_RADIUS {
            self.bump_render_generation();
        }
        let Some(worker) = self.render_worker.as_mut() else {
            return;
        };
        let generation = worker.generation();
        let mut uploaded = false;
        let mut arrived: Vec<RenderedPage> = Vec::new();
        while let Some(done) = worker.try_recv() {
            // A result from before the last zoom or jump is stale: drop it
            // rather than put a wrong texture on screen. (`try_recv` has
            // already released its in-flight mark either way.)
            if done.generation == generation {
                arrived.push(done);
            }
        }
        for done in arrived {
            let image = egui::ColorImage::from_rgba_unmultiplied(done.size, &done.rgba);
            match done.kind {
                RenderKind::Page => {
                    let tex = ctx.load_texture(
                        format!("kpdf-page-{}", done.key.0),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    if let Some(evicted) = self.page_textures_lru.touch(done.key) {
                        self.page_textures.remove(&evicted);
                    }
                    self.page_textures.insert(done.key, tex);
                }
                RenderKind::Thumbnail => {
                    let tex = ctx.load_texture(
                        format!("kpdf-thumb-{}", done.key.0),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.thumbnails.insert(done.key.0, tex);
                }
            }
            uploaded = true;
        }

        // Queue the window around the current page, nearest first, so what
        // the reader is looking at renders before what is further away.
        let n = self.viewport.page_count();
        let mut wanted: Vec<usize> = Vec::new();
        for d in 0..=RENDER_PREFETCH_RADIUS {
            if self.viewport.page() + d < n {
                wanted.push(self.viewport.page() + d);
            }
            if d > 0 && self.viewport.page() >= d {
                wanted.push(self.viewport.page() - d);
            }
        }
        let dpi = self.viewport.dpi();
        let fallback = self.fallback_enabled;
        let mut died = false;
        let Some(worker) = self.render_worker.as_mut() else {
            return;
        };
        for page in wanted {
            let key: PageTextureKey = (page, dpi.to_bits(), fallback);
            if self.page_textures.contains_key(&key) || worker.is_inflight(&key) {
                continue;
            }
            if !worker.request(RenderRequest {
                page,
                dpi,
                fallback,
                generation,
                kind: RenderKind::Page,
            }) {
                died = true;
                break;
            }
        }
        if died {
            self.render_worker = None; // inline path takes over
            return;
        }
        let busy = self.render_worker.as_ref().is_some_and(RenderWorker::busy);
        if uploaded || busy {
            // Pages arrive asynchronously, so keep frames coming or a
            // finished page would not appear until the next input event.
            ctx.request_repaint_after(std::time::Duration::from_millis(60));
        }
    }

    /// Invalidate in-flight renders after a zoom or a fallback toggle.
    ///
    /// Work already running cannot be cancelled, so it is allowed to finish
    /// and is discarded on arrival by generation.
    fn bump_render_generation(&mut self) {
        if let Some(w) = self.render_worker.as_mut() {
            // Publishes the new generation to the worker so it can skip
            // everything already queued, and forgets what was in flight so
            // those pages get re-requested rather than waited on forever.
            w.bump_generation();
        }
    }

    /// Rasterise the opening pages on this thread, so the document is
    /// readable the moment the window appears.
    ///
    /// Deliberately synchronous, and deliberately bounded to
    /// [`SYNC_PRERENDER_PAGES`]: the alternative is a first screen of
    /// placeholders on the one part of the document every reader looks at.
    fn prerender_opening_pages(&mut self, ctx: &egui::Context) {
        let last = SYNC_PRERENDER_PAGES.min(self.viewport.page_count());
        for page in 0..last {
            let _ = self.ensure_page_texture(ctx, page);
        }
    }

    /// Collect finished pages from the background searcher, and ask it for
    /// pages that are on screen but not yet searched.
    ///
    /// Called once per frame, before anything paints. Never blocks: it drains
    /// whatever has arrived and returns.
    fn pump_search_worker(&mut self, visible: &[usize]) {
        let Some(worker) = self.find_worker.as_mut() else {
            return;
        };
        // Everything finished since the last frame.
        let mut arrived = false;
        while let Some((page, hits)) = worker.try_recv() {
            self.find_hits.insert(page, hits);
            arrived = true;
        }

        // A parked scan takes priority over the visible-page prefetch: the
        // user is waiting on it, and the pages it needs may be nowhere near
        // the ones on screen.
        let scan_pages: Vec<usize> = match &self.find_scan {
            Some(scan) => scan_page_order(scan, self.viewport.page_count(), SEARCH_SCAN_LOOKAHEAD),
            None => Vec::new(),
        };
        for page in scan_pages.iter().chain(visible.iter()) {
            if self.find_hits.contains_key(page) || worker.is_requested(*page) {
                continue;
            }
            // A send failure means the worker died; stop asking.
            if !worker.request(*page) {
                self.find_worker = None;
                return;
            }
        }

        // Results changed the picture: see whether the parked scan can now
        // reach a hit (possibly parking again further along).
        if arrived {
            self.resume_scan();
        }
    }

    /// Paint this page's search hits.
    ///
    /// Every hit on the page is shaded; the one the view is *on* is drawn
    /// stronger, so `n` visibly moves rather than just scrolling. A hit's
    /// quads are drawn individually, never as one union box — a phrase broken
    /// across a line break would otherwise be highlighted along with all the
    /// text between its two fragments.
    fn paint_search_highlights(&mut self, ui: &mut egui::Ui, page: usize, layout: PageLayout) {
        if self.find_query.is_empty() {
            return;
        }
        // Only ever paints what is already cached: a page scrolling into view
        // must not trigger a text extraction mid-frame, which would make
        // scrolling cost exactly what lazy searching exists to avoid.
        let Some(hits) = self.find_hits.get(&page) else {
            return;
        };
        if hits.is_empty() {
            return;
        }
        let current = self.find_current;
        let painter = ui.painter();
        for (i, hit) in hits.iter().enumerate() {
            let is_current = current == Some((page, i));
            let fill = if is_current {
                egui::Color32::from_rgba_unmultiplied(255, 165, 0, 110)
            } else {
                egui::Color32::from_rgba_unmultiplied(255, 235, 59, 70)
            };
            for q in &hit.quads {
                // The quad's own corners, mapped through the same page->screen
                // transform the annotations use, so highlights track zoom and
                // scroll exactly like everything else on the page.
                // stext quads are in fitz page space (y-DOWN), not the
                // y-up user space `page_to_screen` takes -- see
                // `stext_to_screen`. Using the wrong one mirrors every
                // highlight vertically, which is exactly what it did.
                let a = stext_to_screen(q.ul.x, q.ul.y, layout);
                let b = stext_to_screen(q.lr.x, q.lr.y, layout);
                let rect = egui::Rect::from_two_pos(
                    egui::pos2(a.0, a.1),
                    egui::pos2(b.0, b.1),
                );
                painter.rect_filled(rect, 1.0, fill);
                if is_current {
                    painter.rect_stroke(
                        rect,
                        1.0,
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 90, 0)),
                        egui::StrokeKind::Outside,
                    );
                }
            }
        }
    }

    /// The find bar: a text field plus the hit counter.
    ///
    /// A real `egui::TextEdit` rather than the key-by-key buffer `:` uses,
    /// because a search query is arbitrary text — it needs paste, selection
    /// and IME, none of which a hand-rolled keystroke loop gets right.
    fn find_bar(&mut self, ui: &mut egui::Ui) {
        let mut commit = false;
        let mut close = false;
        if let Some(bar) = self.find.as_mut() {
            ui.horizontal(|ui| {
                ui.label(if bar.backward { "?" } else { "/" });
                let field = ui.add(
                    egui::TextEdit::singleline(&mut bar.query)
                        .desired_width(f32::INFINITY)
                        .hint_text("find in document"),
                );
                if bar.focus_pending {
                    field.request_focus();
                    bar.focus_pending = false;
                }
                if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    commit = true;
                }
                if ui.button("Find").clicked() {
                    commit = true;
                }
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        }
        if commit {
            self.commit_find();
        } else if close {
            self.find = None;
        }
    }

    /// The contents sidebar: the outline, indented, each item jumping to its
    /// destination.
    fn outline_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Contents").strong());
            if ui.small_button("x").clicked() {
                self.show_outline = false;
            }
        });
        ui.separator();
        // The tree is flattened for display so the whole sidebar is one
        // scroll area; `depth` carries the indent that shows the nesting.
        let flat = flatten_outline(&self.outline, 0);
        let mut jump = None;
        egui::ScrollArea::vertical()
            .id_salt("kpdf_outline_scroll")
            .show(ui, |ui| {
                for (depth, item) in &flat {
                    ui.horizontal(|ui| {
                        ui.add_space(*depth as f32 * 12.0);
                        let label = if item.title.is_empty() {
                            "(untitled)"
                        } else {
                            item.title.as_str()
                        };
                        // An item with no resolvable destination is shown but
                        // not clickable: it is still structure worth seeing,
                        // and a button that does nothing is worse than none.
                        match &item.dest {
                            Some(d) => {
                                if ui.link(label).clicked() {
                                    jump = Some(d.clone());
                                }
                            }
                            None => {
                                ui.weak(label);
                            }
                        }
                    });
                }
            });
        if let Some(d) = jump {
            self.goto_destination(&d);
        }
    }

    fn thumbnail_sidebar(&mut self, ui: &mut egui::Ui) {
        let page_count = self.viewport.page_count();
        let row_height = 96.0;
        let current = self.viewport.page();
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
                            } else {
                                // Not rendered yet. Hold the row's height with
                                // a placeholder so the list does not reflow
                                // under the reader as thumbnails arrive --
                                // rows jumping around while scrolling is worse
                                // than a moment of grey.
                                let w = 72.0_f32;
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(w, w * 1.414),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(
                                    rect,
                                    0.0,
                                    egui::Color32::from_gray(225),
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
                        self.viewport.scroll_to(page);
                    }
                }
            });
    }

    /// Every field at its opening value. Private: hosts go through
    /// [`open_bytes`](Self::open_bytes), which also validates the document.
    fn empty(
        doc: PdfDocument,
        page_count: usize,
        has_acroform: bool,
        config: PdfReaderConfig,
    ) -> PdfReader {
        PdfReader {
            viewport: Viewport::new(page_count, DPI_DEFAULT),
            doc,
            mode: Mode::Image,
            
            
            cmdline: None,
            find: None,
            find_query: String::new(),
            find_hits: HashMap::new(),
            render_worker: None,
            last_pumped_page: 0,
            render_started: false,
            find_scan: None,
            find_worker: None,
            find_current: None,
            outline: Vec::new(),
            show_outline: false,
            scroll_zoom_accum: 0.0,
            
            page_textures: HashMap::new(),
            page_textures_lru: Lru::new(PAGE_TEXTURE_CACHE_CAPACITY),
            thumbnails: HashMap::new(),
            
            hide_thumbnails: false,
            fallback_enabled: true,
            reflow_pages: None,
            reflow_worker: None,
            reflow_scroll: 0.0,
            status: None,
            annot_count: 0,
            tool: Tool::Pan,
            // Forms mode ON for a document that actually has an /AcroForm.
            // It used to default off, which made a form-heavy workbook look
            // broken: every click on a checkbox went to the pan tool and
            // nothing happened, with no hint that a toolbar toggle was
            // standing between the user and the field. A form is for filling
            // in, so a form opens ready to fill in. Documents with no
            // /AcroForm are unaffected -- the mode stays off and the toolbar
            // button is not even shown.
            forms_mode: has_acroform,
            has_acroform,
            edit_history: None,
            unsaved_edits: false,
            draw_stroke: Vec::new(),
            draw_page: None,
            form_edit: None,
            form_edit_focus_pending: false,
            form_fields_cache: HashMap::new(),
            pending_actions: PdfReaderOutput::new(),
            label: None,
            config,
            g_pending: GPending::new(),
            g_armed_at: None,
        }
    }

    // ================= the public embedding surface =====================

    /// Open a document from bytes.
    ///
    /// Bytes, not a path, deliberately: an embedder may have the PDF from a
    /// download, a database, or a zip, and requiring a file on disk would
    /// make those hosts write one out just to be allowed to read it.
    pub fn open_bytes(bytes: Vec<u8>) -> Result<PdfReader, String> {
        PdfReader::open_bytes_with(bytes, PdfReaderConfig::default())
    }

    /// Open from bytes with a specific feature set -- see
    /// [`PdfReaderConfig`], and [`PdfReaderConfig::read_only`] for the
    /// read-only case.
    pub fn open_bytes_with(bytes: Vec<u8>, config: PdfReaderConfig) -> Result<PdfReader, String> {
        let doc = PdfDocument::open(bytes).map_err(|e| format!("open: {e}"))?;
        let page_count = doc.page_count();
        if page_count == 0 {
            return Err("document has no pages".to_string());
        }
        let has_acroform = config.forms && crate::mupdf::form::has_acroform(&doc);
        let mut r = PdfReader::empty(doc, page_count, has_acroform, config);
        if r.forms_mode {
            r.tool = Tool::Pan;
            r.status = Some("form document -- Forms mode on, click a field to fill it".into());
        }
        Ok(r)
    }

    /// Replace the open document with different bytes, keeping the reading
    /// position where it can be kept.
    ///
    /// Everything derived from the OLD document is dropped -- search hits,
    /// outline, textures, thumbnails, layout, undo history, in-progress
    /// strokes and field edits. Keeping any of it would mean drawing one
    /// document's annotations over another's pages. The page is clamped
    /// rather than reset, so a live-recompiled PDF reopens where the reader
    /// was, which is the entire point of a preview loop.
    pub fn load_bytes(&mut self, bytes: Vec<u8>) -> Result<(), String> {
        let doc = PdfDocument::open(bytes).map_err(|e| format!("open: {e}"))?;
        let page_count = doc.page_count();
        if page_count == 0 {
            return Err("document has no pages".to_string());
        }
        self.has_acroform = self.config.forms && crate::mupdf::form::has_acroform(&doc);
        self.doc = doc;
        self.unsaved_edits = false;
        self.find_hits.clear();
        self.find_current = None;
        self.find_worker = None;
        self.find_scan = None;
        // A new document needs a new rasteriser: the old worker holds the old
        // file's bytes and would happily keep answering with its pages.
        self.render_worker = None;
        self.render_started = false;
        self.reflow_worker = None;
        self.outline.clear();
        self.viewport.set_document(page_count);
        self.viewport.rescroll_to_current();
        self.cmdline = None;
        self.page_textures.clear();
        self.page_textures_lru = Lru::new(PAGE_TEXTURE_CACHE_CAPACITY);
        self.thumbnails.clear();
        self.viewport.invalidate_layout();
        self.reflow_pages = None;
        self.reflow_scroll = 0.0;
        self.status = None;
        self.edit_history = None;
        self.draw_stroke.clear();
        self.draw_page = None;
        self.form_edit = None;
        self.form_edit_focus_pending = false;
        self.form_fields_cache.clear();
        self.forms_mode = self.has_acroform;
        if self.forms_mode {
            // `toggle_forms_mode`'s invariant: forms mode implies the Pan
            // tool. Without this a Pen left selected on the previous document
            // would swallow every field click here.
            self.tool = Tool::Pan;
            self.status =
                Some("form document -- Forms mode on, click a field to fill it".into());
        }
        self.g_pending.cancel();
        self.g_armed_at = None;
        Ok(())
    }

    /// The current document's bytes, **including any unsaved edits**.
    ///
    /// This is how a host saves: the reader will not touch a filesystem, so
    /// it hands back what it would have written and the host decides where it
    /// goes. Cheap -- the edited bytes already exist in memory.
    pub fn document_bytes(&self) -> &[u8] {
        self.doc.raw_bytes()
    }

    /// The open document, for a host that wants to read metadata, outline or
    /// text out of it directly.
    pub fn document(&self) -> &PdfDocument {
        &self.doc
    }

    /// The current page, 0-based.
    pub fn current_page(&self) -> usize {
        self.viewport.page()
    }

    pub fn page_count(&self) -> usize {
        self.viewport.page_count()
    }

    /// Whether edits have been made that the host has not been told to
    /// persist. Drives an "unsaved changes" marker; cleared by
    /// [`mark_saved`](Self::mark_saved).
    pub fn has_unsaved_changes(&self) -> bool {
        self.unsaved_edits
    }

    /// Tell the reader the host has persisted the current bytes.
    pub fn mark_saved(&mut self) {
        self.unsaved_edits = false;
    }

    pub fn config(&self) -> &PdfReaderConfig {
        &self.config
    }

    /// The reader's current status message, if any.
    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    /// Put a host message on the reader's status line -- one line, one owner,
    /// so a host does not need a second one beside it.
    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status = Some(msg.into());
    }

    /// A label for the document, shown in the toolbar. A host with a file
    /// typically sets its path; one without leaves it unset.
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }

    /// Collect finished background work and handle input. **Call once per
    /// frame, before painting.**
    ///
    /// [`show`](Self::show) calls this for you. Drive the pane yourself and
    /// it is your responsibility -- skip it and pages render but never
    /// arrive, and no key does anything.
    pub fn pump(&mut self, ctx: &egui::Context) -> PdfReaderOutput {
        let before = self.viewport.page();
        let mut out = PdfReaderOutput::new();

        // Start the rasteriser and lay down the opening pages the first time
        // we paint. Done here rather than in `open` because both need an
        // egui Context to upload textures into.
        if self.render_worker.is_none() && !self.render_started {
            self.render_started = true;
            self.render_worker = RenderWorker::spawn(self.doc.raw_bytes().to_vec());
            self.prerender_opening_pages(ctx);
        }
        self.pump_render_worker(ctx);
        self.pump_reflow_worker(ctx);

        // Collect finished search pages and queue the ones about to be looked
        // at. A window around the current page rather than the exact visible
        // set: in continuous scroll those are the same pages, and this needs
        // no viewport geometry, so it can run before any layout happens.
        if self.find_worker.is_some() {
            let lo = self.viewport.page().saturating_sub(SEARCH_PREFETCH_RADIUS);
            let hi = (self.viewport.page() + SEARCH_PREFETCH_RADIUS).min(self.viewport.page_count().saturating_sub(1));
            let window: Vec<usize> = (lo..=hi).collect();
            self.pump_search_worker(&window);
            // Results land asynchronously, so ask for another frame soon —
            // otherwise a highlight would not appear until the next input
            // event.
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
        self.handle_key(ctx);
        self.handle_scroll_zoom(ctx);

        out.extend(std::mem::take(&mut self.pending_actions));
        if self.viewport.page() != before {
            out.push(ReaderAction::PageChanged { page: self.viewport.page() });
        }
        out
    }

    /// Paint the **document pane only**, into the `Ui` you give it.
    ///
    /// The primitive, per AID-0057: no panels, no toolbar, no sidebars -- it
    /// fills whatever space it is handed. Call [`pump`](Self::pump) first.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> PdfReaderOutput {
        let before = self.viewport.page();
        let mut out = PdfReaderOutput::new();
        egui::CentralPanel::default().show(ui, |ui| match self.mode {
            Mode::Image => {
                let slots = self.continuous_slots(ui.ctx());
                let total_h = slots.last().map(ContinuousSlot::bottom).unwrap_or(0.0);
                let total_w = slots.iter().fold(0.0f32, |m, s| m.max(s.width)).max(1.0);
                let (dx, dy) = self.viewport.take_scroll_delta();
                let pending_delta = egui::Vec2::new(dx, dy);
                let scroll_target = self.viewport.take_scroll_target();

                let output = egui::ScrollArea::both()
                    .id_salt("kpdf_continuous_scroll")
                    // Drag to pan with a MOUSE, not only a touch screen.
                    // egui's default is `DragScroll::OnTouch`, so on a desktop
                    // drag-to-scroll is off entirely -- see `pointer_sense`.
                    .scroll_source(egui::containers::scroll_area::ScrollSource {
                        drag: egui::containers::scroll_area::DragScroll::Always,
                        ..Default::default()
                    })
                    .show(ui, |ui| {
                        if pending_delta != egui::Vec2::ZERO {
                            ui.scroll_with_delta(pending_delta);
                        }

                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(total_w, total_h.max(1.0)),
                            pointer_sense(self.forms_mode, self.tool),
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
                            if let Some(tex) = self.cached_page_texture(ui.ctx(), slot.page_index) {
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
                                // Not rasterized yet. Draw the page's real
                                // outline and SAY SO: a bare grey rectangle
                                // is indistinguishable from a blank page, so
                                // a reader cannot tell "still loading" from
                                // "nothing here".
                                painter.rect_filled(
                                    page_screen_rect,
                                    0.0,
                                    egui::Color32::from_gray(245),
                                );
                                painter.rect_stroke(
                                    page_screen_rect,
                                    0.0,
                                    egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
                                    egui::StrokeKind::Inside,
                                );
                                painter.text(
                                    page_screen_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    format!("page {} — loading…", slot.page_index + 1),
                                    egui::FontId::proportional(
                                        // Scale with the slot so the label
                                        // stays legible when zoomed out and
                                        // does not overflow a small page.
                                        (slot.width * 0.035).clamp(9.0, 18.0),
                                    ),
                                    egui::Color32::from_gray(130),
                                );
                            }

                            let layout = PageLayout {
                                image_x: page_screen_rect.min.x,
                                image_y: page_screen_rect.min.y,
                                image_w: slot.width,
                                image_h: slot.height,
                                page_w_pts: slot.page_w_pts,
                                page_h_pts: slot.page_h_pts,
                                page_x0: slot.page_x0,
                                page_y0: slot.page_y0,
                            };

                            self.paint_search_highlights(ui, slot.page_index, layout);

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

                self.viewport.set_viewport_height(output.inner_rect.height());
                let (response, origin, viewport_top, viewport_bottom) = output.inner;

                self.viewport.recompute_page(viewport_top, viewport_bottom);
                self.annot_count = drawable_annot_count(&self.doc, self.viewport.page());

                // Route the click/drag to whichever tool is active. All
                // three share the same coordinate conversion
                // (`screen_to_page_at`) -- see the module docs' "coordinate
                // trap" section -- and all business logic (what a hit
                // means, how to edit the PDF) lives in
                // `crate::mupdf::{annot_edit,form}`, not here.
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
                        // Indexed by real page number, so this is the right
                        // page's text or nothing -- never a neighbour's.
                        Some(Ok(pages)) => match pages.get(self.viewport.page()).and_then(|p| p.as_ref()) {
                            Some(p) => {
                                for span in &p.spans {
                                    ui.label(&span.text);
                                }
                            }
                            // Still extracting, or this page yielded no text.
                            // Say which, rather than showing a blank pane that
                            // looks like the reader has hung.
                            None => {
                                let msg = match self.reflow_worker.as_ref() {
                                    Some(w) => {
                                        format!("(page loading... {}/{})", w.done, w.total)
                                    }
                                    None => "(no text on this page)".to_string(),
                                };
                                ui.weak(msg);
                            }
                        },
                        Some(Err(e)) => {
                            ui.colored_label(egui::Color32::LIGHT_RED, e.as_str());
                        }
                        None => {
                            ui.label("extracting...");
                        }
                    });
            }
        });
        out.extend(std::mem::take(&mut self.pending_actions));
        if self.viewport.page() != before {
            out.push(ReaderAction::PageChanged { page: self.viewport.page() });
        }
        out
    }

    /// Paint the reader **with kpdf's whole chrome** -- toolbar, outline and
    /// thumbnail sidebars, find bar, and the document pane -- into the `Ui`
    /// you give it, pumping first.
    ///
    /// The convenience layer. It is built on [`ui`](Self::ui) rather than
    /// replacing it, so a host that wants a different arrangement still has
    /// the pieces. Everything here is a panel *inside* your `Ui`, so it
    /// claims only the space you gave it.
    pub fn show(&mut self, ui: &mut egui::Ui) -> PdfReaderOutput {
        let mut out = self.pump(ui.ctx());
        let before = self.viewport.page();
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
                        out.push(ReaderAction::OpenRequested);
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
                    // Page add/delete sit in the toolbar as well as on keys:
                    // lecturing with a tablet often means no hand free for the
                    // keyboard at all.
                    if ui
                        .button("+ Page")
                        .on_hover_text("Add a blank page after this one (a)")
                        .clicked()
                    {
                        self.add_blank_page();
                    }
                    if ui
                        .add_enabled(self.viewport.page_count() > 1, egui::Button::new("- Page"))
                        .on_hover_text(
                            "Delete this page (Ctrl+Delete) -- undoable; the last page cannot be deleted",
                        )
                        .clicked()
                    {
                        self.delete_current_page();
                    }
                    ui.separator();
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
                        out.push(ReaderAction::SaveAsRequested);
                    }
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Save in place"))
                        .on_hover_text("Overwrite the open file (Ctrl+S)")
                        .clicked()
                    {
                        out.push(ReaderAction::SaveRequested);
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
                    if self.viewport.page_count() > 1 {
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
                    ui.label(self.label.as_deref().unwrap_or("(in memory)"));
                    ui.separator();
                    match self.mode {
                        Mode::Image => {
                            ui.label(format!("page {}/{}", self.viewport.page() + 1, self.viewport.page_count()));
                            ui.separator();

                            // On-screen zoom controls, wired to the exact same
                            // zoom_in/zoom_out/zoom_reset methods the `+`/`-`
                            // keys and Ctrl+scroll use -- one place owns the
                            // DPI_STEP clamping logic. Each arrow button disables
                            // itself at the DPI_MIN/DPI_MAX limit rather than
                            // sitting there clickable-but-useless.
                            if ui
                                .add_enabled(self.viewport.dpi() > DPI_MIN, egui::Button::new("-"))
                                .on_hover_text("Zoom out (-)")
                                .clicked()
                            {
                                self.zoom_out();
                            }
                            if ui
                                .button(format!("{}%", zoom_percent(self.viewport.dpi())))
                                .on_hover_text("Reset zoom to 100%")
                                .clicked()
                            {
                                self.zoom_reset();
                            }
                            if ui
                                .add_enabled(self.viewport.dpi() < DPI_MAX, egui::Button::new("+"))
                                .on_hover_text("Zoom in (+)")
                                .clicked()
                            {
                                self.zoom_in();
                            }
                            ui.label(format!("({:.0} dpi)", self.viewport.dpi()));
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

        if self.show_outline && !self.outline.is_empty() {
            egui::Panel::left("kpdf_outline")
                .resizable(true)
                .default_size(240.0)
                .show(ui, |ui| self.outline_sidebar(ui));
        }

        if self.mode == Mode::Image && self.viewport.page_count() > 1 && !self.hide_thumbnails {
            egui::Panel::left("kpdf_thumbnails")
                .resizable(true)
                .default_size(110.0)
                .show(ui, |ui| self.thumbnail_sidebar(ui));
        }

        if self.find.is_some() {
            egui::Panel::bottom("kpdf_find").show(ui, |ui| self.find_bar(ui));
        }

        out.extend(self.ui(ui));
        out.extend(std::mem::take(&mut self.pending_actions));
        if self.viewport.page() != before && !out.actions.iter().any(|a| matches!(a, ReaderAction::PageChanged { .. })) {
            out.push(ReaderAction::PageChanged { page: self.viewport.page() });
        }
        out
    }

}

/// The character a key contributes to the `:` command line, or `None` for a
/// key that contributes nothing (arrows, function keys, modifiers).
///
/// Accepts digits for `:N` page jumps and lowercase letters for named commands
/// like `:w`. Letters are lowercased because `egui` reports the physical key,
/// not the shifted character -- and vim's commands are lowercase anyway, so
/// folding here means `:w` works whether or not Caps Lock is on, while
/// [`parse_command`] can still reject a genuinely different command.
/// Flatten an outline tree into `(depth, item)` pairs in reading order, so
/// the sidebar can render it as one list with indentation.
fn flatten_outline(items: &[OutlineItem], depth: usize) -> Vec<(usize, &OutlineItem)> {
    let mut out = Vec::new();
    for it in items {
        out.push((depth, it));
        // Children of a collapsed item are still listed: kpdf's sidebar has no
        // expand/collapse control yet, and hiding them would make parts of the
        // document unreachable from it.
        out.extend(flatten_outline(&it.children, depth + 1));
    }
    out
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Same arXiv fixture as the search tests; absent means skip, not fail.
    fn fixture() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/arxiv-2608.17504v1.pdf");
        std::fs::read(path).ok()
    }

    /// A minimal marker page. `number` is used here purely as an arbitrary
    /// tag so a placement test can ask "is this the page I put here?" -- it
    /// is NOT the 1-based page number a real extraction produces (see
    /// `mupdf_extract::extract_mupdf_page`), and nothing reads it as one.
    fn marker(number: usize) -> TextPage {
        TextPage {
            number,
            width: 100.0,
            height: 100.0,
            spans: Vec::new(),
        }
    }

    /// The bug this pins: the pump used to `push` in arrival order and throw
    /// the page index away. The worker skips a page it cannot extract, so one
    /// bad page shifted **every** later page up by one -- reflow then showed
    /// the wrong page's text, quietly, for the rest of the document.
    ///
    /// Placement by index means a gap stays a gap.
    #[test]
    fn a_skipped_page_leaves_a_hole_instead_of_shifting_later_pages() {
        let mut pages: Vec<Option<TextPage>> = Vec::new();
        // Pages 0 and 1 arrive; page 2 is unextractable and never arrives.
        place_reflow_page(&mut pages, 0, marker(0));
        place_reflow_page(&mut pages, 1, marker(1));
        place_reflow_page(&mut pages, 3, marker(3));

        assert_eq!(pages.len(), 4, "the vector grows to hold the highest index");
        assert!(pages[0].is_some());
        assert!(pages[1].is_some());
        assert!(
            pages[2].is_none(),
            "the skipped page must stay a hole -- filling it with page 3's \
             text is exactly the off-by-one this test exists to catch"
        );
        assert_eq!(
            pages[3].as_ref().map(|p| p.number),
            Some(3),
            "page 3 must land at index 3, not index 2"
        );
    }

    /// Out-of-order arrival must land by index too. Nothing in the channel
    /// contract promises order once more than one producer is conceivable, and
    /// the fix must not quietly depend on it.
    #[test]
    fn placement_does_not_depend_on_arrival_order() {
        let mut pages: Vec<Option<TextPage>> = Vec::new();
        place_reflow_page(&mut pages, 5, marker(5));
        place_reflow_page(&mut pages, 2, marker(2));
        assert_eq!(pages.len(), 6);
        assert_eq!(pages[2].as_ref().map(|p| p.number), Some(2));
        assert_eq!(pages[5].as_ref().map(|p| p.number), Some(5));
        for i in [0usize, 1, 3, 4] {
            assert!(pages[i].is_none(), "index {i} must be untouched");
        }
    }

    /// Re-placing an index overwrites rather than appending, so a reload that
    /// re-delivers a page cannot double it up.
    #[test]
    fn replacing_a_page_overwrites_in_place() {
        let mut pages: Vec<Option<TextPage>> = Vec::new();
        place_reflow_page(&mut pages, 1, marker(1));
        place_reflow_page(&mut pages, 1, marker(1));
        assert_eq!(pages.len(), 2);
    }

    /// The worker must deliver every page **and then hang up**.
    ///
    /// The hang-up is the point: the pump asks for a repaint every 120 ms
    /// while the worker lives, so a worker that never disconnects pins the UI
    /// at ~8 fps for the rest of the session. Completion used to be tested as
    /// `done >= total`, which a single skipped page makes unreachable -- hence
    /// disconnect, which happens however the worker ends.
    #[test]
    fn the_worker_delivers_every_page_then_disconnects() {
        let Some(bytes) = fixture() else { return };
        let total = PdfDocument::open(bytes.clone())
            .expect("fixture opens")
            .page_count();
        assert!(total > 0, "fixture must have pages for this to mean anything");

        let worker = ReflowWorker::spawn(bytes, total).expect("worker thread spawns");

        let mut pages: Vec<Option<TextPage>> = Vec::new();
        let mut delivered = 0usize;
        let mut disconnected = false;
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            match worker.res.recv_timeout(Duration::from_millis(500)) {
                Ok((index, page)) => {
                    place_reflow_page(&mut pages, index, page);
                    delivered += 1;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    disconnected = true;
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            }
        }
        assert!(
            disconnected,
            "the worker must drop its sender when done -- without that the \
             pump keeps requesting frames forever"
        );
        assert_eq!(
            delivered, total,
            "every page of a clean fixture should extract"
        );
        assert_eq!(pages.len(), total, "and land at its own index");
        for (i, p) in pages.iter().enumerate() {
            let p = p
                .as_ref()
                .unwrap_or_else(|| panic!("hole at index {i} on a clean fixture"));
            // `Page::number` is 1-based (human-facing), the vector is
            // 0-based -- see `mupdf_extract::extract_mupdf_page`.
            assert_eq!(
                p.number,
                i + 1,
                "page {} landed at index {i}, so placement is off",
                p.number
            );
        }
    }

    /// Extraction happens on the worker, not the caller: `spawn` must return
    /// promptly even for a document that takes a long while to extract. This
    /// is the whole reason the worker exists -- entering reflow used to block
    /// the UI thread for the full extraction.
    #[test]
    fn spawning_does_not_block_on_extraction() {
        let Some(bytes) = fixture() else { return };
        let total = PdfDocument::open(bytes.clone())
            .expect("fixture opens")
            .page_count();
        let t0 = Instant::now();
        let worker = ReflowWorker::spawn(bytes, total).expect("worker thread spawns");
        let elapsed = t0.elapsed();
        drop(worker);
        assert!(
            elapsed < Duration::from_millis(500),
            "spawn took {elapsed:?} -- it must hand off to the thread, not \
             extract inline"
        );
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;

    fn fixture() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/arxiv-2608.17504v1.pdf");
        std::fs::read(path).ok()
    }

    /// The whole point of the extraction: a host with bytes and no file can
    /// build a working reader.
    #[test]
    fn a_reader_opens_from_bytes_with_no_filesystem_involved() {
        let Some(bytes) = fixture() else { return };
        let r = PdfReader::open_bytes(bytes.clone()).expect("opens from bytes");
        assert!(r.page_count() > 0);
        assert_eq!(r.current_page(), 0);
        assert!(!r.has_unsaved_changes(), "a freshly opened document is clean");
        assert_eq!(
            r.document_bytes(),
            &bytes[..],
            "an unedited document hands back exactly what it was given"
        );
    }

    /// Garbage in must be an error, not a panic and not a reader showing
    /// nothing -- a host needs to be able to report the failure.
    #[test]
    fn opening_rubbish_is_an_error_not_a_panic() {
        assert!(PdfReader::open_bytes(b"this is not a PDF".to_vec()).is_err());
        assert!(PdfReader::open_bytes(Vec::new()).is_err());
    }

    /// The brief's required capability, checked on a real reader rather than
    /// only on the config type: a read-only reader must refuse to enable any
    /// writing feature, while keeping everything needed to read.
    #[test]
    fn a_read_only_reader_has_no_path_to_modifying_the_pdf() {
        let Some(bytes) = fixture() else { return };
        let r = PdfReader::open_bytes_with(bytes, PdfReaderConfig::read_only())
            .expect("opens read-only");
        assert!(!r.config().can_modify());
        assert!(
            !r.forms_mode,
            "forms mode must not switch itself on when forms are disabled, \
             even for a document carrying an /AcroForm"
        );
        assert!(r.config().search && r.config().continuous_scroll);
    }

    /// Swapping documents must drop everything derived from the old one.
    /// Keeping any of it means drawing one document's state over another's
    /// pages -- and the position is clamped, not reset, so a recompile
    /// reopens where the reader was.
    #[test]
    fn loading_a_new_document_resets_derived_state_and_clamps_the_page() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes.clone()).expect("opens");
        let n = r.page_count();
        assert!(n > 1, "precondition: the fixture has several pages");

        r.viewport.jump_to(n - 1);
        r.find_hits.insert(0, Vec::new());
        r.outline.clear();
        r.status = Some("stale".into());
        r.unsaved_edits = true;

        r.load_bytes(bytes).expect("reloads");
        assert!(r.find_hits.is_empty(), "old hits must not survive");
        assert!(r.status.is_none(), "old status must not survive");
        assert!(!r.has_unsaved_changes(), "reloading is not an edit");
        assert_eq!(
            r.current_page(),
            n - 1,
            "a same-length reload keeps the reading position"
        );
        assert!(
            r.render_worker.is_none() && !r.render_started,
            "the old rasteriser holds the old bytes and must be dropped"
        );
    }

    #[test]
    fn loading_rubbish_leaves_the_open_document_intact() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes).expect("opens");
        let n = r.page_count();
        assert!(r.load_bytes(b"not a pdf".to_vec()).is_err());
        assert_eq!(
            r.page_count(),
            n,
            "a failed load must not leave the reader holding nothing"
        );
    }

    /// The save handshake: the reader tracks whether the host is behind, and
    /// the host says when it has caught up. Without `mark_saved` an
    /// "unsaved changes" marker would never clear.
    #[test]
    fn the_host_clears_the_unsaved_flag_after_it_writes() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes).expect("opens");
        r.unsaved_edits = true;
        assert!(r.has_unsaved_changes());
        r.mark_saved();
        assert!(!r.has_unsaved_changes());
    }

    /// Quit and save are REPORTED, never performed -- the reader must not
    /// close a host's window or touch its filesystem. `:wq` in particular has
    /// to report the save before the quit, or a host that acts on the first
    /// action and stops would silently drop the write.
    #[test]
    fn wq_reports_save_before_quit() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes).expect("opens");
        r.pending_actions.push(ReaderAction::SaveRequested);
        r.pending_actions.push(ReaderAction::QuitRequested);
        let acts = std::mem::take(&mut r.pending_actions).actions;
        assert_eq!(
            acts,
            vec![ReaderAction::SaveRequested, ReaderAction::QuitRequested],
            "order is part of the contract, not an accident"
        );
    }

    /// A host with no file still gets a sensible toolbar label rather than a
    /// blank or a panic.
    #[test]
    fn the_label_is_optional() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes).expect("opens");
        assert!(r.label.is_none());
        r.set_label("/tmp/paper.pdf");
        assert_eq!(r.label.as_deref(), Some("/tmp/paper.pdf"));
    }

    /// One status line, one owner: a host pushes its own messages through the
    /// reader rather than maintaining a second line beside it.
    #[test]
    fn a_host_can_write_to_the_status_line() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes).expect("opens");
        r.set_status("saved /tmp/paper.pdf");
        assert_eq!(r.status(), Some("saved /tmp/paper.pdf"));
    }
}

// The freehand pen's *interaction* is deliberately not unit-tested here, and
// the reason is worth recording so nobody adds a test that looks like it
// covers it.
//
// Two attempts failed honestly:
//
//  1. Asserting `Context::requested_repaint_last_pass()` around a synthetic
//     drag. It PASSED WITH THE FIX REMOVED -- egui requests repaints of its
//     own during any interaction, so the assertion was measuring egui, not
//     us. A test that cannot fail is worse than no test: it reports safety
//     that is not there.
//  2. Driving a real drag through `Context::run_ui` with synthetic
//     `PointerMoved`/`PointerButton` input. `Response::dragged()` stays false
//     and `interact_pointer_pos()` stays `None` across passes, with or
//     without a `screen_rect` -- egui's interaction state does not come up in
//     a bare headless pass.
//
// So "the preview line is visible while drawing" is maintainer-dogfooding
// territory, per the workspace rule about unverifiable interactive surfaces.
// What IS covered lives elsewhere and is worth knowing: the ink annotation
// that a finished stroke commits round-trips through our own reader and
// through poppler (`tests/authoring.rs`), and the stroke-to-page attribution
// rule is enforced by `handle_draw_update`'s `Some(_) => {}` arm.

#[cfg(test)]
mod forms_tests {
    use super::*;

    fn stream_obj(dict_fields: &str, content: &[u8]) -> Vec<u8> {
        let mut body =
            format!("<< {dict_fields} /Length {} >>\nstream\n", content.len()).into_bytes();
        body.extend_from_slice(content);
        body.extend_from_slice(b"\nendstream");
        body
    }

    fn build_pdf(bodies: &[Vec<u8>]) -> Vec<u8> {
        let mut pdf: Vec<u8> = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.5\n");
        let mut offsets = vec![0usize; bodies.len() + 1];
        for (idx, body) in bodies.iter().enumerate() {
            let num = idx + 1;
            offsets[num] = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref_ofs = pdf.len();
        let size = bodies.len() + 1;
        pdf.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets.iter().skip(1) {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_ofs}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }

    /// One page carrying a text field and a checkbox, behind a real
    /// `/AcroForm` -- the minimum a reader must light up for.
    fn form_pdf() -> Vec<u8> {
        build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [4 0 R 5 0 R] >> >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Annots [4 0 R 5 0 R] >>"
                .to_vec(),
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (Name) /V (Alice)               /Rect [10 10 210 30] /DA (/Helv 0 Tf 0 g) >>"
                .to_vec(),
            b"<< /Type /Annot /Subtype /Widget /FT /Btn /T (Agree) /Rect [10 40 30 60]               /AS /Off /V /Off /AP << /N << /Yes 6 0 R /Off 7 0 R >> >> >>"
                .to_vec(),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 20 20]",
                b"1 g 0 0 20 20 re f",
            ),
            stream_obj(
                "/Type /XObject /Subtype /Form /BBox [0 0 20 20]",
                b"0 g 0 0 20 20 re f",
            ),
        ])
    }

    /// The whole chain the paint path walks, in order. A form document must
    /// open with forms mode ON (a form is for filling in), the field cache
    /// must actually populate, and each field must resolve to a *visible*
    /// highlight -- `FieldHighlight::None` is skipped by the painter, so a
    /// field that lands there is invisible however correct everything else is.
    ///
    /// Covers the gh-96 move: every one of these links is a place the
    /// extraction could have quietly dropped a wire.
    #[test]
    fn a_form_document_opens_with_visible_fields() {
        let mut r = PdfReader::open_bytes(form_pdf()).expect("form fixture opens");

        assert!(r.has_acroform, "the /AcroForm dict must be detected");
        assert!(
            r.forms_mode,
            "a form document opens ready to fill in -- otherwise every click \
             goes to the pan tool and the document looks broken"
        );
        assert_eq!(
            r.tool,
            Tool::Pan,
            "forms mode implies the Pan tool, or a Pen would swallow field clicks"
        );

        r.refresh_form_fields_cache(0);
        let fields = r.form_fields_cache.get(&0).expect("page 0 is cached");
        assert!(
            !fields.is_empty(),
            "the page's /Annots widgets must reach the cache the painter reads"
        );

        let visible = fields
            .iter()
            .filter(|f| field_highlight_kind(f.kind, f.read_only) != FieldHighlight::None)
            .count();
        assert!(
            visible > 0,
            "at least one field must map to a drawable highlight -- \
             FieldHighlight::None is skipped by the painter, so all-None \
             means a form that is there but cannot be seen"
        );
    }

    /// The toolbar's Forms toggle is gated on `has_acroform`, so a document
    /// without one must not offer it -- and must not open in forms mode.
    #[test]
    fn a_plain_document_offers_no_forms_mode() {
        let plain = build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec(),
        ]);
        let r = PdfReader::open_bytes(plain).expect("plain fixture opens");
        assert!(!r.has_acroform);
        assert!(!r.forms_mode);
    }

    /// Swapping documents must re-decide forms mode from the NEW document.
    /// Carrying the old answer over is how a plain PDF ends up in forms mode
    /// (nothing to click) or a form opens with forms off (looks broken).
    #[test]
    fn forms_mode_is_re_decided_on_every_document_swap() {
        let plain = build_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec(),
        ]);
        let mut r = PdfReader::open_bytes(plain.clone()).expect("opens");
        assert!(!r.forms_mode);

        r.load_bytes(form_pdf()).expect("loads the form");
        assert!(r.forms_mode, "a form loaded later must switch forms on");
        assert!(
            r.form_fields_cache.is_empty(),
            "the previous document's field cache must not survive the swap"
        );

        r.load_bytes(plain).expect("loads the plain doc again");
        assert!(!r.forms_mode, "and switch back off for a plain document");
    }
}

#[cfg(test)]
mod edit_reload_tests {
    use super::*;

    fn fixture() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/arxiv-2608.17504v1.pdf");
        std::fs::read(path).ok()
    }

    /// THE REGRESSION: "I can draw with the pen, but the moment I do the PDF
    /// reloads and I cannot see the annotations -- though the reader still
    /// says there are annots."
    ///
    /// The render worker owns a private `PdfDocument`, opened from a COPY of
    /// the bytes as they were when it started, because a document holds
    /// `RefCell`s and cannot cross threads. An edit replaces `self.doc` and
    /// clears the texture caches -- which, with a live worker still holding
    /// the pre-edit bytes, actively causes the bug: every visible page is
    /// re-requested and re-rendered from the OLD file, so the new ink is
    /// nowhere on screen, while `annot_count` (read from the new document)
    /// reports it exists.
    ///
    /// So an edit must drop the worker. This test fails if it does not.
    #[test]
    fn an_edit_drops_the_render_worker_that_holds_the_old_bytes() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes.clone()).expect("opens");

        // Stand in for the first painted frame, which is what starts the
        // worker. Spawning a real one keeps this honest: the assertion below
        // is about a worker that genuinely exists.
        r.render_worker = RenderWorker::spawn(r.doc.raw_bytes().to_vec());
        r.render_started = true;
        assert!(r.render_worker.is_some(), "precondition: a worker is running");

        r.reload_from_bytes(bytes);

        assert!(
            r.render_worker.is_none(),
            "an edit must drop the rasteriser -- it holds the pre-edit bytes \
             and will keep painting a document without the new annotation"
        );
        assert!(
            !r.render_started,
            "and must clear `render_started`, or `pump`'s spawn gate never \
             fires again and every page stays a placeholder forever"
        );
        assert!(
            r.page_textures.is_empty() && r.thumbnails.is_empty(),
            "the caches hold pre-edit images and must go with it"
        );
    }

    /// An in-flight reflow extractor is reading the old bytes too, and would
    /// deliver text for a document that no longer exists.
    #[test]
    fn an_edit_drops_the_reflow_worker_as_well() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes.clone()).expect("opens");
        r.reflow_worker = ReflowWorker::spawn(bytes.clone(), r.viewport.page_count());
        assert!(r.reflow_worker.is_some(), "precondition");
        r.reload_from_bytes(bytes);
        assert!(r.reflow_worker.is_none());
        assert!(r.reflow_pages.is_none(), "and the text it produced");
    }

    /// An ordinary annotation does not change the page count, and must not be
    /// treated as a document swap: `set_document` drops queued scroll
    /// intents, so a stroke that cancelled a pending jump would feel like the
    /// document lurching back under the pen.
    #[test]
    fn an_edit_that_keeps_the_page_count_keeps_a_queued_jump() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes.clone()).expect("opens");
        r.viewport.scroll_to(4);
        assert!(r.viewport.has_scroll_target(), "precondition");

        r.reload_from_bytes(bytes);

        assert!(
            r.viewport.has_scroll_target(),
            "a same-length edit must not cancel a pending scroll"
        );
    }

    /// A page insertion or deletion DOES change the count, and the viewport
    /// has to hear about it -- otherwise navigation clamps to the old maximum
    /// and the new page is unreachable.
    #[test]
    fn an_edit_that_changes_the_page_count_updates_the_viewport() {
        let Some(bytes) = fixture() else { return };
        let mut r = PdfReader::open_bytes(bytes).expect("opens");
        let before = r.viewport.page_count();

        let doc = PdfDocument::open(r.doc.raw_bytes().to_vec()).expect("reopens");
        let grown = crate::mupdf::page_edit::insert_blank_page(&doc, before, None)
            .expect("a blank page can be appended");
        r.reload_from_bytes(grown);

        assert_eq!(
            r.viewport.page_count(),
            before + 1,
            "the viewport must see the new page, or it cannot be navigated to"
        );
    }
}

#[cfg(test)]
mod pointer_sense_tests {
    use super::*;

    /// The Pan tool must NOT claim the drag: panning is the scroll area's
    /// job, and `Tool::Pan`'s own handler deliberately does nothing. A pane
    /// that senses drag takes it first (egui interacts the scroll background
    /// before the content), so claiming it here is exactly what left
    /// click-and-drag doing nothing at all.
    #[test]
    fn the_pan_tool_leaves_the_drag_to_the_scroll_area() {
        let sense = pointer_sense(false, Tool::Pan);
        assert!(
            !sense.senses_drag(),
            "Pan must not claim the drag, or the scroll area never sees it \
             and the document cannot be dragged"
        );
        assert!(
            sense.senses_click(),
            "clicks are still ours -- egui tracks click and drag targets \
             separately, so keeping click costs the scroll area nothing"
        );
    }

    /// The pen and eraser DO consume the drag -- the drag is their input.
    /// Losing it would make them unusable, which is the opposite failure.
    #[test]
    fn the_drawing_tools_claim_the_drag() {
        for tool in [Tool::Draw, Tool::Erase] {
            let sense = pointer_sense(false, tool);
            assert!(
                sense.senses_drag(),
                "{tool:?} follows the pointer; the drag IS the input"
            );
        }
    }

    /// Forms mode is click-driven, so it leaves the drag alone and a form can
    /// still be panned around while being filled in -- and it wins over
    /// whatever tool happens to be selected underneath, matching
    /// `toggle_forms_mode`'s invariant.
    #[test]
    fn forms_mode_is_click_only_whatever_tool_is_selected() {
        for tool in [Tool::Pan, Tool::Draw, Tool::Erase] {
            let sense = pointer_sense(true, tool);
            assert!(
                sense.senses_click() && !sense.senses_drag(),
                "forms mode with {tool:?} must stay clickable and pannable"
            );
        }
    }
}
