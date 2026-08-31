//! Reusable egui-based PDF-viewer building blocks.
//!
//! These are the pieces of the `kpdf` viewer (`src/bin/kpdf.rs`) that are
//! genuinely reusable by *any* egui-based KOPITIAM front end (kvim, kovan, a
//! future TUI-adjacent GUI) rather than specific to that one binary's app
//! state: single-page layout math, the continuous (all-pages-in-one-column)
//! scroll layout, the zoom model, hit-testing, forms-mode UI helpers, the
//! annotation-tool state machine, vim-motion key handling, a generic bounded
//! LRU eviction policy, the `Pixmap` -> egui texture bridge, and -- as of
//! gh-96 -- the **background render and search workers**. `kpdf.rs` itself
//! consumes this module rather than defining any of it a second time.
//!
//! Left behind in `kpdf.rs`, deliberately not moved here: `KpdfApp` itself
//! (the concrete eframe app state), `main`, `pick_pdf`, `load_doc`, the
//! eframe window/event-loop wiring, and `digit_char` (a `goto`-page-entry
//! helper tied to that one keybinding, not a generally reusable primitive).
//!
//! # Embedding this in your own egui application
//!
//! Depend on the crate with **`egui` alone**, not the default features:
//!
//! ```toml
//! [dependencies]
//! kopitiam-pdf = { version = "0.3", default-features = false, features = ["egui"] }
//! ```
//!
//! `default` is `kpdf`, which is the *standalone viewer application* -- it
//! adds an `eframe` event loop and the `rfd` native file picker on top of
//! everything here. An embedding application already has its own window and
//! its own idea of how a file gets chosen, so compiling those is pure cost
//! (and `rfd` in particular drags in a desktop-portal stack a host may
//! deliberately not want). Verified: `cargo tree` for the `egui`-only build
//! contains neither crate.
//!
//! What that gets you today is the *engine* half -- you drive it yourself:
//!
//! ```no_run
//! # use kopitiam_pdf::gui_frontend::{RenderWorker, RenderRequest, RenderKind};
//! # fn demo(bytes: Vec<u8>) -> Option<()> {
//! let mut worker = RenderWorker::spawn(bytes)?;
//! worker.request(RenderRequest {
//!     page: 0,
//!     dpi: 150.0,
//!     fallback: true,
//!     generation: worker.generation(),
//!     kind: RenderKind::Page,
//! });
//! // ...later, once per frame, never blocking:
//! while let Some(done) = worker.try_recv() {
//!     if done.generation == worker.generation() {
//!         // upload `done.rgba` at `done.size` as a texture and paint it
//!     }
//! }
//! # Some(()) }
//! ```
//!
//! A single `PdfReader` type that assembles all of this for you is the goal
//! of gh-96's later phases; it does not exist yet. Until it does, the pieces
//! here are usable directly, and `kpdf.rs` is the worked example of how they
//! fit together.
//!
//! When that reader does arrive, its shape is already settled (AID-0057):
//! `reader.ui(ui)` will paint **the document pane only**, with thumbnails and
//! outline as separate methods you call into a `Ui` you own. An
//! `egui::Panel` cannot be created inside a `Ui`, so the host keeps layout;
//! the library keeps the engines.
//!
//! Named `gui_frontend` rather than `egui` on purpose (maintainer's call):
//! a module literally named `egui` would shadow the external `egui` crate
//! for every file under it, forcing `::egui::` everywhere just to reach the
//! real crate. `gui_frontend` sidesteps that entirely -- ordinary `egui::Foo`
//! paths work as expected throughout this module tree.
//!
//! # Gating
//!
//! This whole module is behind the crate's `egui` feature, declared
//! `#[cfg(feature = "egui")]` in `lib.rs`. `kpdf` implies `egui` (enforced by
//! a `compile_error!` in `lib.rs`, so the two can never drift), which is why
//! the binary sees all of this. A `--no-default-features` build never sees
//! this module and never pulls in egui at all -- the headless core stays
//! headless.

pub mod forms;
pub mod geometry;
pub mod hit_test;
pub mod hot_reload;
pub mod keys;
pub mod lru;
pub mod pixmap;
pub mod render;
pub mod search;
pub mod tools;
pub mod zoom;

pub use forms::{
    FieldHighlight, consume_commit_enter, field_highlight_kind, highlight_colors,
    should_commit_on_enter,
};
pub use geometry::{stext_to_screen, 
    ContinuousSlot, PageLayout, PageSize, continuous_slot_visible, current_page_in_view,
    field_rect_to_screen, layout_continuous_pages, min_hit_rect, page_display_size, page_size_pts,
    page_to_screen, recentred_scroll_offset, rect_contains, screen_to_page, screen_to_page_at,
};
pub use hit_test::{hit_test_annot, hit_test_field, hit_test_field_expanded};
pub use hot_reload::{HotReload, RELOAD_CHECK_INTERVAL, ReloadDecision, read_mtime};
pub use keys::{
    Command, G_PENDING_TIMEOUT, GPending, VIM_STEP, g_pending_expired, half_viewport_step,
    keys_captured, parse_command,
};
pub use lru::Lru;
pub use pixmap::{drawable_annot_count, rgb_to_rgba};
pub use render::{RenderKey, RenderKind, RenderRequest, RenderWorker, RenderedPage};
pub use search::{FindScan, SearchWorker, scan_page_order};
pub use tools::{Tool, select_tool, toggle_forms_mode};
pub use zoom::{
    DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, ZOOM_DELTA_PER_STEP, zoom_percent,
    zoom_steps_from_zoom_delta,
};
