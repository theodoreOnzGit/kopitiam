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

//! # This file is a SHELL, not a reader
//!
//! Since gh-96 Phase 12, every line of reading behaviour above lives in
//! [`kopitiam_pdf::gui_frontend::PdfReader`] -- the same component any other
//! egui application can embed. What is left here is standalone-application
//! policy, and only that:
//!
//! * a command-line path argument, and the native Open/Save-As dialogs;
//! * writing bytes to a file, atomically, via a temp file and a rename;
//! * watching that file for external recompiles (hot reload);
//! * the eframe window and process lifecycle.
//!
//! The reader never touches a filesystem and never closes a window. It
//! reports [`ReaderAction`]s -- `SaveRequested`, `OpenRequested`,
//! `QuitRequested` -- and this shell decides what they mean. That is exactly
//! the boundary that lets kovan embed the same reader without inheriting
//! kpdf's opinions about files.
//!
//! So this file is the worked example a downstream host can read: roughly a
//! hundred lines of glue between a `PdfReader` and an operating system.

use std::path::PathBuf;

use kopitiam_pdf::gui_frontend::{
    HotReload, PdfReader, ReaderAction, RELOAD_CHECK_INTERVAL, ReloadDecision,
};

/// The standalone application: a reader, the file it came from, and a watcher
/// on that file.
struct KpdfApp {
    /// Every bit of PDF-reading behaviour. Not a field kpdf reaches into --
    /// it drives it through the public API, the same one an embedder gets.
    reader: PdfReader,
    /// Where the open document came from. The reader has no idea; it holds
    /// bytes.
    path: PathBuf,
    /// Reloads the document when something else rewrites `path` -- a live
    /// TeX/Typst recompile. Host-side because it is a property of the *file*,
    /// and an embedded reader may have no file at all.
    hot_reload: HotReload,
}

impl KpdfApp {
    fn open(path: PathBuf) -> Result<KpdfApp, String> {
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut reader = PdfReader::open_bytes(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        reader.set_label(path.display().to_string());
        let mut hot_reload = HotReload::new(true);
        // Claim the file we just read, so the first poll does not see our own
        // open as an external change.
        hot_reload.mark_current(&path);
        Ok(KpdfApp {
            reader,
            path,
            hot_reload,
        })
    }

    /// Open a different document, in place.
    fn open_path(&mut self, path: PathBuf) {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                self.reader.set_status(format!("{}: {e}", path.display()));
                return;
            }
        };
        match self.reader.load_bytes(bytes) {
            Ok(()) => {
                self.path = path;
                self.reader.set_label(self.path.display().to_string());
                // A different file: claim its mtime so the watcher does not
                // read our own open as an external change.
                self.hot_reload.mark_current(&self.path);
            }
            Err(e) => self.reader.set_status(format!("{}: {e}", path.display())),
        }
    }

    fn open_via_dialog(&mut self) {
        if let Some(path) = pick_pdf() {
            self.open_path(path);
        }
    }

    /// Overwrite the open file with the reader's current bytes.
    ///
    /// Through a temp file and a rename, so an interrupted save cannot leave
    /// a half-written PDF where the original was -- the rename is atomic on
    /// every platform this ships to.
    fn save_in_place(&mut self) {
        if !self.reader.has_unsaved_changes() {
            return;
        }
        let bytes = self.reader.document_bytes().to_vec();
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
            self.reader
                .set_status(format!("save {}: {e}", self.path.display()));
            return;
        }
        match std::fs::rename(&tmp_path, &self.path) {
            Ok(()) => {
                // Claim the mtime our own rename just set. Without this the
                // watcher reads this save as an external change and reloads
                // the document out from under the user on every Ctrl+S --
                // see hot_reload.rs's module docs.
                self.hot_reload.mark_current(&self.path);
                self.reader.mark_saved();
                // Repeat the decryption warning at the moment it matters. The
                // file now on disk has no password protection, and a note seen
                // when the document was opened an hour ago is not consent.
                let note = if self.reader.was_decrypted() {
                    " (DECRYPTED -- no password protection)"
                } else {
                    ""
                };
                self.reader
                    .set_status(format!("saved {}{note}", self.path.display()));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                self.reader
                    .set_status(format!("save {}: {e}", self.path.display()));
            }
        }
    }

    /// Write the reader's bytes wherever a native dialog says.
    fn save_via_dialog(&mut self) {
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
        let bytes = self.reader.document_bytes().to_vec();
        match std::fs::write(&target, &bytes) {
            Ok(()) => {
                // Save-As can legitimately target the document we already
                // have open. When it does, claim the mtime we just wrote and
                // count the edits as saved -- otherwise hot reload sees our
                // own write as external. Saving elsewhere leaves the open
                // document's unsaved state exactly as it was, which is
                // correct: those edits are still not in *this* file.
                if target == self.path {
                    self.hot_reload.mark_current(&self.path);
                    self.reader.mark_saved();
                }
                let note = if self.reader.was_decrypted() {
                    " (DECRYPTED -- no password protection)"
                } else {
                    ""
                };
                self.reader
                    .set_status(format!("saved {}{note}", target.display()));
            }
            Err(e) => self
                .reader
                .set_status(format!("save {}: {e}", target.display())),
        }
    }

    /// Reload the document when something else rewrote it on disk.
    ///
    /// A recompile is only picked up when there is nothing to lose. With
    /// unsaved ink or field edits the reload is REFUSED and said so, leaving
    /// it to the user to save (or discard by reopening with `o`). Silently
    /// throwing away someone's annotations because LaTeX recompiled
    /// underneath is exactly the kind of data loss this reader must not do.
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
        if self.reader.has_unsaved_changes() {
            self.reader.set_status(format!(
                "{} changed on disk -- not reloaded, you have unsaved edits (Ctrl+S to keep them)",
                self.path.display()
            ));
            // Claim the new mtime anyway, or this fires again every 500 ms
            // and buries every other status message under the same warning.
            self.hot_reload.mark_current(&self.path);
            return;
        }
        let path = self.path.clone();
        self.open_path(path);
        self.reader
            .set_status(format!("reloaded {}", self.path.display()));
    }

    /// Act on what the reader reported. This is the whole of kpdf's policy.
    fn handle_reader_actions(&mut self, out: kopitiam_pdf::gui_frontend::PdfReaderOutput, ctx: &egui::Context) {
        for action in out.actions {
            match action {
                ReaderAction::SaveRequested => self.save_in_place(),
                ReaderAction::SaveAsRequested => self.save_via_dialog(),
                ReaderAction::OpenRequested => self.open_via_dialog(),
                // Standalone means one document in one window, so quitting
                // closes it. An embedder would close a tab, or ignore this.
                ReaderAction::QuitRequested => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close)
                }
                // Everything else is information kpdf has no use for: it has
                // no page-number widget of its own, no citation manager, and
                // nothing to do with a selected region. A host like kovan
                // would act on exactly these.
                _ => {}
            }
        }
    }
}

impl eframe::App for KpdfApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Before anything paints: pick up an external recompile of the open
        // file. Cheap -- throttled to one `stat` per RELOAD_CHECK_INTERVAL
        // inside `HotReload::poll`.
        self.check_hot_reload(ui.ctx());

        // The entire reader, chrome included. `show` pumps its workers and
        // handles input itself, so this really is all of it.
        let out = self.reader.show(ui);
        self.handle_reader_actions(out, ui.ctx());
    }
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
