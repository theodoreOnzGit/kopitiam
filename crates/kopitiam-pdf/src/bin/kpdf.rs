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
//! # Forms open ready to fill in
//!
//! A document carrying an `/AcroForm` opens with **Forms mode already on**,
//! so clicking a checkbox ticks it straight away. This used to default off,
//! which made a form-heavy workbook look broken -- every click went to the
//! pan tool and nothing happened, with nothing on screen saying a toolbar
//! toggle stood in the way. Documents with no `/AcroForm` are unaffected.
//!
//! To annotate a form with the pen instead, turn Forms off in the toolbar.
//!
//! # Wacom / stylus users on Wayland: run under XWayland (bd-wdh)
//!
//! **If your tablet does nothing in kpdf, this is why.** On a Wayland
//! session the stylus never draws: `winit` does not implement
//! `zwp_tablet_v2`, so tablet events never reach `egui` at all. Nothing in
//! kpdf is at fault and there is nothing to configure -- the protocol
//! support is simply absent upstream.
//!
//! The workaround, confirmed by the maintainer on the hardware:
//!
//! ```text
//! WAYLAND_DISPLAY= ./target/release/kpdf lecture.pdf
//! ```
//!
//! Emptying `WAYLAND_DISPLAY` forces the app onto XWayland, where the tablet
//! arrives via XInput2 -- which `winit` *does* support -- and the stylus
//! works. As a bonus it is also markedly less laggy than the Wayland path on
//! the same build and document (a separate, unmeasured performance gap;
//! bd-75k).
//!
//! This matters most for exactly the workflow the page-editing keys below
//! exist to serve: lecturing with a tablet. Worth putting in a shell alias
//! before a lecture rather than discovering it in front of a room.
//!
//! # Rendering: one UI thread, one worker
//!
//! Rasterising a page costs a **median 135 ms and up to 444 ms** at 150 dpi.
//! Doing that while painting — which kpdf did until 0.3.0 — stalls the window
//! for that long every time a page scrolls into view.
//!
//! So painting **never rasterises**. The first
//! [`SYNC_PRERENDER_PAGES`] pages are rendered synchronously at open, so the
//! document is readable the moment the window appears (measured on the
//! 506-page Irodori book: 165 ms to parse, 635 ms for those pages, ~800 ms to
//! a usable window). Everything after that belongs to a single
//! [`RenderWorker`] thread, which delivers pages **one at a time as each
//! finishes** so they fill in progressively rather than arriving in a lump.
//!
//! A page that is not ready yet draws its real outline with a
//! `page N — loading…` label, so "still working" is never mistaken for a
//! blank page. Sidebar thumbnails go through the same worker: rendered
//! inline they cost ~76 ms each, and `show_rows` uncovers about ten at once
//! when you jump, which froze the window for most of a second *and* delayed
//! the page queue behind it.
//!
//! **Jumps discard queued work.** The request queue is FIFO, so after `G` it
//! is full of pages near where you used to be. Rendering those first, at
//! 135 ms each, is what made a jump sit on "loading…" for seconds. A jump
//! (a move further than the prefetch radius) bumps the generation, which the
//! worker reads through an atomic and uses to drop stale requests *before*
//! paying for them. An ordinary step does not bump, so its neighbours
//! survive.
//!
//! Zoom invalidates in-flight work by generation rather than trying to cancel
//! it: a render already running is allowed to finish and is discarded on
//! arrival, which cannot leave a wrong-resolution texture on screen.
//!
//! Two honest limits. The worker needs **its own copy of the file's bytes**,
//! because a `PdfDocument` holds `RefCell`s and is `Send` but not `Sync` —
//! real for a 100 MB book, and alongside the search worker that is a third
//! copy. And **there is no GPU rasterisation**: `kopitiam-pdf` draws on the
//! CPU, so the "GPU thread pool" of the wider design has nothing to attach to
//! until raster kernels exist.
//!
//! # Find and contents (`/` `?` `n` `N` `t`)
//!
//! MuPDF's and vim's own bindings, as asked for:
//!
//! * **`/`** opens the find bar searching forward, **`?`** backward.
//! * **`n`** goes to the next hit, **`N`** (Shift+n) the previous. Both wrap
//!   once around the document and then report no match, rather than looping.
//! * **`t`** toggles the contents sidebar (mupdf-gl's key), which lists the
//!   outline and jumps on click.
//!
//! **This took `n`/`p` away from page navigation**, which they were bound to
//! before. Page keys are now **`,`** and **`.`** (mupdf's own), and the arrows
//! and PageUp/PageDown are unchanged — those were always bound.
//!
//! Searching is **incremental**: only the pages actually needed are extracted
//! and searched, resuming from the page in view. Searching all 506 pages of a
//! long document up front is precisely the all-pages-on-the-UI-thread stall
//! that `/MediaBox` page sizing was introduced to remove, and a reader only
//! ever needs the next hit.
//!
//! Pages around the one in view are searched **on a background thread**, so
//! scrolling onto a page of hits highlights it without `n` having to reach it
//! first. That is not a nicety: extracting a page's text costs a median of
//! 5.4 ms on the arXiv fixture but a **p90 of 300 ms and a worst case of
//! 349 ms** (the plot-heavy pages), so doing it as pages scroll into view
//! would drop a third of a second at a time — and a per-frame time budget
//! could not help either, since that cost is paid inside one indivisible
//! extraction. See [`SearchWorker`].
//!
//! Searching never blocks the window, including the first `/`. A scan for the
//! next hit examines only pages already searched; the moment it reaches an
//! unsearched one it parks and resumes as the worker delivers. That matters
//! more than it sounds: searching the 506-page Irodori book for a word it
//! does not contain scans **every page** and took **16.3 seconds** of frozen
//! UI when the scan ran inline.
//!
//! # Quitting is `:q` / `:wq` only
//!
//! `Escape` does **not** close the window. It is the key every panel uses for
//! "cancel" — the find bar, the command line, a form edit — so having it also
//! quit meant one press too many threw the session away. At top level it now
//! clears the search and its highlights, or says there is nothing to cancel.
//! `:q` quits, `:wq` (or `:x`) saves in place and quits, and the forcing
//! spellings `:q!`/`:wq!` are accepted as the same thing since kpdf never
//! blocks a quit on unsaved edits.
//!
//! # Live lecturing: blank pages on demand
//!
//! `a` inserts a blank page **after the page you are on** and jumps to it, at
//! that page's size and rotation, so a derivation that overruns its slide
//! continues onto fresh paper without leaving the flow. `Ctrl+Delete` removes
//! the current page. Both are ordinary entries in the edit history, so Undo
//! and `Ctrl+S` treat them exactly like ink. The page-tree mechanics, and why
//! a deleted page's bytes remain in the saved file, are in
//! [`kopitiam_pdf::mupdf::page_edit`].
//!
//! # Hot reload
//!
//! The open file's mtime is polled (throttled, not filesystem-watched) and an
//! external change re-opens it, keeping your page and zoom -- so a TeX/Typst
//! recompile refreshes in place. A document with unsaved annotations is never
//! auto-reloaded; see [`KpdfApp::check_hot_reload`].
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
//! * **+ Page** / **- Page** -- add a blank page after the current one
//!   (**`a`**), or delete the current one (**`Ctrl+Delete`**). Both are in
//!   the toolbar as well as on keys because the use case they exist for --
//!   lecturing with a graphics tablet -- often leaves no hand free for the
//!   keyboard. See [`KpdfApp::add_blank_page`].
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
use std::time::Instant;

use eframe::egui;
// Reusable page-layout/zoom/hit-testing/forms-UI/tool-state/continuous-scroll/
// vim-key/LRU building blocks, lifted out of this binary into the library
// proper (kopitiam_pdf::gui_frontend, gated on the crate's `kpdf` feature) so
// other egui-based KOPITIAM front ends can reuse them without
// re-implementing this file.
use kopitiam_pdf::gui_frontend::{
    Command, ContinuousSlot, DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, FieldHighlight, FindScan,
    GPending,
    HotReload, Lru, PageLayout, PageSize, RELOAD_CHECK_INTERVAL, ReloadDecision, RenderKey,
    RenderKind, RenderRequest, RenderWorker, RenderedPage, SearchWorker, Tool, VIM_STEP,
    consume_commit_enter, continuous_slot_visible,
    current_page_in_view, drawable_annot_count, field_highlight_kind, field_rect_to_screen,
    g_pending_expired, half_viewport_step, highlight_colors, hit_test_annot, hit_test_field,
    hit_test_field_expanded, keys_captured, layout_continuous_pages, min_hit_rect, stext_to_screen,
    page_to_screen, parse_command, rgb_to_rgba, scan_page_order, screen_to_page_at, select_tool,
    toggle_forms_mode,
    zoom_percent, zoom_steps_from_zoom_delta,
};
use kopitiam_pdf::mupdf::annot_edit::{EditHistory, InkAnnotSpec, InkStroke};
use kopitiam_pdf::mupdf::form::FieldKind;
use kopitiam_pdf::mupdf::outline::{OutlineItem, load_outline};
use kopitiam_pdf::mupdf::stext_search::{SearchHit, search_page};
use kopitiam_pdf::mupdf::structured_text::StextOptions;
use kopitiam_pdf::mupdf::{PdfDocument, Rect, page_to_stext, rasterize_page_with_fallback};
use kopitiam_pdf::{Page as TextPage, extract_mupdf_page};

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

/// A page's full-resolution texture is cached under the same key the
/// rasteriser produces it under -- `(page_index, dpi.to_bits(),
/// fallback_enabled)`.
///
/// An alias for the library's [`RenderKey`] rather than a second definition,
/// so the texture cache and the render worker can never drift apart on what
/// identifies a page image. The local name is kept because on this side of
/// the boundary the thing being keyed really is a texture.
type PageTextureKey = RenderKey;

/// Cache key for the continuous-layout slot list ([`KpdfApp::slots_cache`]):
/// rebuild only when dpi, the page count (a new document) or the fallback
/// toggle actually change -- not on every frame.
///
/// It used to carry the thumbnail count too, because page sizes were
/// measured from rendered thumbnails and so improved as more arrived. Sizes
/// now come from `/MediaBox` ([`KpdfApp::page_size_for_layout`]), which is
/// exact on the first frame, so there is nothing to refine and the layout is
/// built exactly once per (dpi, document, fallback) combination.
type SlotsCacheKey = (u32, usize, bool);

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
/// Pulled out of [`KpdfApp::pump_reflow_worker`] purely so it can be tested:
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
struct FindBar {
    query: String,
    backward: bool,
    /// `true` for exactly the frame after the bar opens, so the text field can
    /// take focus once rather than stealing it back every frame.
    focus_pending: bool,
}

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
    /// The find bar, when open: the query being typed and which way `Enter`
    /// will search. `None` means the bar is closed.
    find: Option<FindBar>,
    /// The committed query the hit cache belongs to. Cleared with the cache
    /// whenever a new search is committed.
    find_query: String,
    /// Per-page hits for [`KpdfApp::find_query`], filled lazily.
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
    /// Reload the document when it changes on disk -- the live-preview loop
    /// for a TeX/Typst recompile. Ported from `kovan`'s reader; see
    /// [`kopitiam_pdf::gui_frontend::hot_reload`] for the mechanism and for
    /// why it polls rather than watches. Default **on**, matching kovan.
    hot_reload: HotReload,
    /// Whether [`KpdfApp::edit_history`] holds annotation/form edits that are
    /// not yet on disk. Gates hot reload: re-opening the file would discard
    /// them, so a document with unsaved work is never auto-reloaded (the
    /// status bar says so instead). Set by [`KpdfApp::apply_edit`], cleared
    /// by a successful save and by opening a document.
    unsaved_edits: bool,
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
        let mut hot_reload = HotReload::new(true);
        // Claim the file we just read, so the first poll does not see our own
        // open as an external change.
        hot_reload.mark_current(&path);
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
            last_viewport_h: 800.0,
            page_textures: HashMap::new(),
            page_textures_lru: Lru::new(PAGE_TEXTURE_CACHE_CAPACITY),
            thumbnails: HashMap::new(),
            slots_cache: None,
            hide_thumbnails: false,
            fallback_enabled: true,
            reflow_pages: None,
            reflow_worker: None,
            reflow_scroll: 0.0,
            status: has_acroform
                .then(|| "form document -- Forms mode on, click a field to fill it".to_string()),
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
            hot_reload,
            unsaved_edits: false,
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
                // A different document: claim its mtime and drop the old
                // document's unsaved-edit state (its history goes too, below).
                self.hot_reload.mark_current(&self.path);
                self.unsaved_edits = false;
                // Search hits and the outline belong to the OLD document.
                self.find_hits.clear();
                self.find_current = None;
                self.find_worker = None;
                // A new document needs a new rasteriser: the old worker holds
                // the old file.
                self.render_worker = None;
                self.render_started = false;
                self.reflow_worker = None;
                self.outline.clear();
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
                // is no longer open. Forms mode is not carried over stale
                // either; it is re-decided from the NEW document, since the
                // old one's answer says nothing about this one.
                self.edit_history = None;
                self.draw_stroke.clear();
                self.draw_page = None;
                self.form_edit = None;
                self.form_edit_focus_pending = false;
                self.form_fields_cache.clear();
                self.forms_mode = self.has_acroform;
                if self.forms_mode {
                    // `toggle_forms_mode`'s invariant: forms mode implies the
                    // Pan tool. Without this a Pen left selected on the
                    // previous document would swallow every field click here.
                    self.tool = Tool::Pan;
                    self.status =
                        Some("form document -- Forms mode on, click a field to fill it".into());
                }
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
        self.bump_render_generation();
    }

    fn zoom_out(&mut self) {
        self.dpi = (self.dpi - DPI_STEP).max(DPI_MIN);
        self.bump_render_generation();
    }

    /// Reset zoom to [`DPI_DEFAULT`] -- the on-screen zoom readout doubles as
    /// this button, since clicking "100%" to get back to 100% is the
    /// discoverable affordance every other viewer offers.
    fn zoom_reset(&mut self) {
        self.dpi = DPI_DEFAULT;
        self.bump_render_generation();
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

    /// This page's on-screen size (and physical point size) for the
    /// continuous layout, read from `/MediaBox` (with `/Rotate` applied) via
    /// [`kopitiam_pdf::mupdf::page_geom::page_size_points`].
    ///
    /// # Why not the thumbnail any more
    ///
    /// This was ported from `kovan`, which measures the page's *rendered
    /// thumbnail* -- cheap for a short document, and it needs no `/MediaBox`
    /// lookup. On a long one it is ruinous: [`KpdfApp::continuous_slots`]
    /// asks for every page's size on the first frame, so measuring by
    /// rendering meant rasterizing the whole document before anything could
    /// be laid out. The Irodori Japanese-course workbook (506 pages, 106 MB)
    /// parsed in 133 ms and then hung the window for **36.6 seconds** doing
    /// exactly that.
    ///
    /// The file states the size outright, so read it: microseconds for the
    /// whole document, and `/Rotate` is handled properly into the bargain
    /// (the thumbnail route got that for free by rendering; a naive
    /// `/MediaBox` read would not, which is why `page_size_points` applies
    /// it).
    ///
    /// Thumbnails are still rendered for the sidebar, but only for the rows
    /// actually on screen (`show_rows` in
    /// [`KpdfApp::thumbnail_sidebar`]), which is bounded no matter how long
    /// the document is.
    fn page_size_for_layout(&mut self, _ctx: &egui::Context, page: usize) -> PageSize {
        let (page_w_pts, page_h_pts) =
            kopitiam_pdf::mupdf::page_geom::page_size_points(&self.doc, page);
        let scale = self.dpi / 72.0;
        PageSize {
            display_w: page_w_pts * scale,
            display_h: page_h_pts * scale,
            page_w_pts,
            page_h_pts,
        }
    }

    /// The continuous layout's per-page slots, rebuilt only when
    /// [`SlotsCacheKey`] actually changes (dpi, page count, how many
    /// thumbnails have been rasterized so far, or the fallback toggle) --
    /// not on every frame, since a long document's slot list is otherwise
    /// non-trivial `Vec` churn to rebuild 20 times a second for nothing.
    fn continuous_slots(&mut self, ctx: &egui::Context) -> Vec<ContinuousSlot> {
        let key: SlotsCacheKey = (self.dpi.to_bits(), self.page_count, self.fallback_enabled);
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
    /// The page's texture **only if it is already rasterised**.
    ///
    /// This is what the paint loop uses, and the distinction from
    /// [`KpdfApp::ensure_page_texture`] is the entire point of the render
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
        let key: PageTextureKey = (page, self.dpi.to_bits(), self.fallback_enabled);
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
        // Start with an empty (Ok) result and fill it in as pages arrive.
        // Extracting the whole document here is what froze the window for
        // tens of seconds on a long book -- see `ReflowWorker`.
        self.reflow_pages = Some(Ok(Vec::new()));
        self.reflow_worker =
            ReflowWorker::spawn(self.doc.raw_bytes().to_vec(), self.page_count);
        if self.reflow_worker.is_none() {
            // No thread available: fall back to doing it here, which is slow
            // but still correct.
            self.reflow_pages = Some(
                kopitiam_pdf::extract_mupdf_from_bytes(self.doc.raw_bytes())
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
                self.page_count = self.doc.page_count();
                self.page = self.page.min(self.page_count.saturating_sub(1));
                self.page_textures.clear();
                self.page_textures_lru = Lru::new(PAGE_TEXTURE_CACHE_CAPACITY);
                self.thumbnails.clear();
                self.slots_cache = None;
                // Extracted reflow text is indexed by page, so inserting or
                // removing a page shifts every entry after it out of step.
                self.reflow_pages = None;
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
            Ok(()) => {
                // Save-As can legitimately target the document we already
                // have open. When it does, claim the mtime we just wrote and
                // count the edits as saved -- otherwise hot reload sees our
                // own write as external. Saving elsewhere leaves the open
                // document's unsaved state exactly as it was, which is
                // correct: those edits are still not in *this* file.
                if target == self.path {
                    self.hot_reload.mark_current(&self.path);
                    self.unsaved_edits = false;
                }
                self.status = Some(format!("saved {}", target.display()));
            }
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
            Ok(()) => {
                // Claim the mtime our own rename just set. Without this the
                // watcher reads this save as an external change and reloads
                // the document out from under the user on every Ctrl+S --
                // see hot_reload.rs's module docs.
                self.hot_reload.mark_current(&self.path);
                self.unsaved_edits = false;
                self.status = Some(format!("saved {}", self.path.display()));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                self.status = Some(format!("save {}: {e}", self.path.display()));
            }
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
    /// ([`kopitiam_pdf::mupdf::page_edit::insert_blank_page`]), so the new
    /// page appears at the same size and zoom and the stroke you were about
    /// to draw lands where you expect.
    ///
    /// Goes through [`KpdfApp::apply_edit`] like every other edit, so it is
    /// undoable and saved by the same Ctrl+S.
    fn add_blank_page(&mut self) {
        let at = self.page + 1;
        let result = kopitiam_pdf::mupdf::page_edit::insert_blank_page(&self.doc, at, None);
        let ok = result.is_ok();
        self.apply_edit(result);
        if ok {
            // `apply_edit` -> `reload_from_bytes` keeps the page index, but the
            // new page is the point of the exercise, so move to it.
            self.page = at.min(self.page_count.saturating_sub(1));
            self.scroll_to_page = Some(self.page);
            self.status = Some(format!("added blank page {}", self.page + 1));
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
    /// [`kopitiam_pdf::mupdf::page_edit`]'s module docs. It is deletion, not
    /// redaction.
    fn delete_current_page(&mut self) {
        let target = self.page;
        let result = kopitiam_pdf::mupdf::page_edit::delete_page(&self.doc, target);
        let ok = result.is_ok();
        self.apply_edit(result);
        if ok {
            // Deleting the last page leaves the index past the end; step back
            // so the viewer lands on the page that took its place (or the new
            // final page).
            self.page = target.min(self.page_count.saturating_sub(1));
            self.scroll_to_page = Some(self.page);
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
    /// worker ([`KpdfApp::resume_scan`]), because searching a page can cost
    /// up to 349 ms and a scan may cross the whole document — which is what
    /// hung the window for minutes on the 506-page Irodori book.
    fn find_step(&mut self, forward: bool, from_current_page: bool) {
        if self.find_query.is_empty() {
            self.status = Some("no search — press / to find".into());
            return;
        }
        let n = self.page_count;
        if n == 0 {
            return;
        }
        let (page, idx) = match (from_current_page, self.find_current) {
            (false, Some((p, i))) => (p, Some(i)),
            _ => (self.page, None),
        };
        self.scan_from(forward, page, idx, n);
    }

    /// Walk pages in scan order looking for the next hit, using cached pages
    /// only. Parks the scan at the first unsearched page.
    fn scan_from(&mut self, forward: bool, start: usize, start_idx: Option<usize>, budget: usize) {
        let n = self.page_count;
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
                self.page = page;
                self.scroll_to_page = Some(page);
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
    fn goto_destination(&mut self, dest: &kopitiam_pdf::mupdf::destination::Destination) {
        use kopitiam_pdf::mupdf::destination::Destination;
        match dest {
            Destination::Page { page, .. } => {
                let p = (*page).min(self.page_count.saturating_sub(1));
                self.page = p;
                self.scroll_to_page = Some(p);
            }
            // Opening a browser is the application's call, not the viewer's:
            // say where it points and let the operator decide.
            Destination::Uri(u) => self.status = Some(format!("link: {u}")),
            Destination::Unsupported(kind) => {
                self.status = Some(format!("{kind} destinations are not followed"));
            }
        }
    }

    /// Poll the open file and reload it if it changed on disk (hot reload).
    ///
    /// Ported from `kovan`'s `check_hot_reload`, with one addition kovan does
    /// not need: kpdf holds **unsaved annotation edits** in memory, and
    /// re-opening would discard them. So a document with unsaved work is
    /// never auto-reloaded -- the status bar reports the change and waits for
    /// the user to save (or discard by reopening with `o`). Silently throwing
    /// away someone's ink because LaTeX recompiled underneath is exactly the
    /// kind of data loss this reader must not do.
    ///
    /// The page position is preserved across the reload (clamped, in case the
    /// recompile changed the page count), which is the whole point of a
    /// live-preview loop -- kovan does the same.
    fn check_hot_reload(&mut self, ctx: &egui::Context) {
        if !self.hot_reload.is_enabled() {
            return;
        }
        // Keep waking up even when idle, or an unattended window would never
        // poll (egui only repaints on input otherwise).
        ctx.request_repaint_after(RELOAD_CHECK_INTERVAL);
        if self.hot_reload.poll(&self.path, std::time::Instant::now()) != ReloadDecision::Changed {
            return;
        }
        if self.unsaved_edits {
            self.status = Some(format!(
                "{} changed on disk -- not reloaded, you have unsaved edits (Ctrl+S to keep them)",
                self.path.display()
            ));
            // Claim it anyway: otherwise this repeats every 500ms and buries
            // every other status message the user might need to read.
            self.hot_reload.mark_current(&self.path);
            return;
        }
        let keep_page = self.page;
        let keep_dpi = self.dpi;
        let path = self.path.clone();
        self.open_path(path);
        self.dpi = keep_dpi;
        self.page = keep_page.min(self.page_count.saturating_sub(1));
        self.scroll_to_page = Some(self.page);
        self.status = Some(format!("{} changed on disk -- reloaded", self.path.display()));
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
                                Command::Write => self.save_in_place(),
                                Command::Quit => {
                                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                                    return;
                                }
                                Command::WriteQuit => {
                                    self.save_in_place();
                                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
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
            egui::Key::H => self.pending_scroll_delta.x += VIM_STEP,
            egui::Key::L => self.pending_scroll_delta.x -= VIM_STEP,
            egui::Key::J => self.pending_scroll_delta.y -= VIM_STEP,
            egui::Key::K => self.pending_scroll_delta.y += VIM_STEP,
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
        let moved = self.page.abs_diff(self.last_pumped_page);
        self.last_pumped_page = self.page;
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
        let n = self.page_count;
        let mut wanted: Vec<usize> = Vec::new();
        for d in 0..=RENDER_PREFETCH_RADIUS {
            if self.page + d < n {
                wanted.push(self.page + d);
            }
            if d > 0 && self.page >= d {
                wanted.push(self.page - d);
            }
        }
        let dpi = self.dpi;
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
        let last = SYNC_PRERENDER_PAGES.min(self.page_count);
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
            Some(scan) => scan_page_order(scan, self.page_count, SEARCH_SCAN_LOOKAHEAD),
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

impl eframe::App for KpdfApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Before anything paints: pick up an external recompile of the open
        // file (hot reload). Cheap -- throttled to one `stat` per
        // RELOAD_CHECK_INTERVAL inside `HotReload::poll`.
        self.check_hot_reload(ui.ctx());

        // Start the rasteriser and lay down the opening pages the first time
        // we paint. Done here rather than in `open` because both need an
        // egui Context to upload textures into.
        if self.render_worker.is_none() && !self.render_started {
            self.render_started = true;
            self.render_worker = RenderWorker::spawn(self.doc.raw_bytes().to_vec());
            self.prerender_opening_pages(ui.ctx());
        }
        self.pump_render_worker(ui.ctx());
        self.pump_reflow_worker(ui.ctx());

        // Collect finished search pages and queue the ones about to be looked
        // at. A window around the current page rather than the exact visible
        // set: in continuous scroll those are the same pages, and this needs
        // no viewport geometry, so it can run before any layout happens.
        if self.find_worker.is_some() {
            let lo = self.page.saturating_sub(SEARCH_PREFETCH_RADIUS);
            let hi = (self.page + SEARCH_PREFETCH_RADIUS).min(self.page_count.saturating_sub(1));
            let window: Vec<usize> = (lo..=hi).collect();
            self.pump_search_worker(&window);
            // Results land asynchronously, so ask for another frame soon —
            // otherwise a highlight would not appear until the next input
            // event.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(120));
        }
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
                        .add_enabled(self.page_count > 1, egui::Button::new("- Page"))
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

        if self.show_outline && !self.outline.is_empty() {
            egui::Panel::left("kpdf_outline")
                .resizable(true)
                .default_size(240.0)
                .show(ui, |ui| self.outline_sidebar(ui));
        }

        if self.mode == Mode::Image && self.page_count > 1 && !self.hide_thumbnails {
            egui::Panel::left("kpdf_thumbnails")
                .resizable(true)
                .default_size(110.0)
                .show(ui, |ui| self.thumbnail_sidebar(ui));
        }

        if self.find.is_some() {
            egui::Panel::bottom("kpdf_find").show(ui, |ui| self.find_bar(ui));
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
                        // Indexed by real page number, so this is the right
                        // page's text or nothing -- never a neighbour's.
                        Some(Ok(pages)) => match pages.get(self.page).and_then(|p| p.as_ref()) {
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

        // NO blanket repaint here, deliberately. This used to ask for a frame
        // every 50 ms unconditionally, "to notice key events promptly" -- but
        // egui already wakes on input, so all that achieved was pinning kpdf
        // at 20 fps forever, laying out the whole window twenty times a second
        // with an idle document on screen. That is the lag the maintainer
        // reported while in reflow mode; reflow only made it visible, it was
        // burning a core the entire time.
        //
        // Every genuinely asynchronous source asks for its own frame instead,
        // and only while it actually has work outstanding:
        //   * `check_hot_reload`      -> RELOAD_CHECK_INTERVAL (500 ms)
        //   * `pump_render_worker`    -> 60 ms, while pages are in flight
        //   * `pump_reflow_worker`    -> 120 ms, until the worker disconnects
        //   * the search pump         -> 120 ms, while a query is running
        // Idle, with hot reload on, that is 2 wakeups a second instead of 20.
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
mod reflow_worker_tests {
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
