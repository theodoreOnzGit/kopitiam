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
//! Then hand it bytes and paint it. [`PdfReader`] is the whole reader --
//! rendering, search, thumbnails, vim navigation, forms, annotations:
//!
//! ```no_run
//! # use kopitiam_pdf::gui_frontend::{PdfReader, PdfReaderConfig, ReaderAction};
//! # fn demo(bytes: Vec<u8>, ui: &mut egui::Ui) -> Result<(), String> {
//! // Read-only: every reading feature, no path that rewrites the PDF.
//! let mut reader = PdfReader::open_bytes_with(bytes, PdfReaderConfig::read_only())?;
//!
//! // ...then once per frame, inside whatever Ui you give it:
//! for action in reader.show(ui).actions {
//!     match action {
//!         ReaderAction::PageChanged { page } => { /* update your own chrome */ }
//!         ReaderAction::SaveRequested => { /* YOU decide what saving means */ }
//!         _ => {}
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! `crates/kopitiam-pdf/examples/embed_reader.rs` is a complete third-party
//! host doing exactly this, and it is compiled by CI precisely so this stays
//! true.
//!
//! ## Two entry points
//!
//! [`PdfReader::show`] above assembles kpdf's chrome for you. The primitive
//! underneath it is [`PdfReader::ui`], which paints **the document pane
//! only** into the `Ui` you hand it, leaving you to place -- or omit --
//! [`thumbnail_sidebar`](PdfReader::thumbnail_sidebar),
//! [`outline_sidebar`](PdfReader::outline_sidebar) and
//! [`find_bar`](PdfReader::find_bar) yourself. Reach for `ui` when the reader
//! shares a window with your own tooling and you want control of the layout;
//! `show` when you just want a viewer. See AID-0057 for why the pane is the
//! primitive and the chrome is the convenience, rather than the other way
//! round.
//!
//! ## Driving the engines directly
//!
//! Everything `PdfReader` is built from is public too --
//! [`RenderWorker`], [`SearchWorker`], [`Viewport`], [`Thumbnails`] -- for a
//! host that wants, say, background rasterisation without any of the reading
//! UI. `PdfReader` is the assembly, not a wall.
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

pub mod action;
pub mod config;
pub mod forms;
pub mod geometry;
pub mod hit_test;
pub mod hot_reload;
pub mod keys;
pub mod lru;
pub mod pixmap;
pub mod reader;
pub mod render;
pub mod search;
pub mod thumbnails;
pub mod tools;
pub mod viewport;
pub mod zoom;

pub use action::{PdfReaderOutput, ReaderAction};
pub use config::PdfReaderConfig;
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
pub use reader::PdfReader;
pub use thumbnails::{THUMBNAIL_DPI, Thumbnails};
pub use viewport::{CONTINUOUS_GAP, SlotsCacheKey, Viewport};
pub use tools::{Tool, select_tool, toggle_forms_mode};
pub use zoom::{
    DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, ZOOM_DELTA_PER_STEP, zoom_percent,
    zoom_steps_from_zoom_delta,
};
