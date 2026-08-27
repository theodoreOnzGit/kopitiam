//! Reusable egui-based PDF-viewer building blocks.
//!
//! These are the pieces of the `kpdf` viewer (`src/bin/kpdf.rs`) that are
//! genuinely reusable by *any* egui-based KOPITIAM front end (kvim, kovan, a
//! future TUI-adjacent GUI) rather than specific to that one binary's app
//! state: single-page layout math, the continuous (all-pages-in-one-column)
//! scroll layout, the zoom model, hit-testing, forms-mode UI helpers, the
//! annotation-tool state machine, vim-motion key handling, a generic bounded
//! LRU eviction policy, and the `Pixmap` -> egui texture bridge. `kpdf.rs`
//! itself now consumes this module rather than defining these a second
//! time -- see its imports.
//!
//! Left behind in `kpdf.rs`, deliberately not moved here: `KpdfApp` itself
//! (the concrete eframe app state), `main`, `pick_pdf`, `load_doc`, the
//! eframe window/event-loop wiring, and `digit_char` (a `goto`-page-entry
//! helper tied to that one keybinding, not a generally reusable primitive).
//!
//! Named `gui_frontend` rather than `egui` on purpose (maintainer's call):
//! a module literally named `egui` would shadow the external `egui` crate
//! for every file under it, forcing `::egui::` everywhere just to reach the
//! real crate. `gui_frontend` sidesteps that entirely -- ordinary `egui::Foo`
//! paths work as expected throughout this module tree.
//!
//! # Gating
//!
//! This whole module is behind the crate's `kpdf` feature (the same feature
//! that turns on the optional `eframe`/`egui`/`rfd` dependencies -- see
//! `Cargo.toml`'s `[features]` section) and is declared `#[cfg(feature =
//! "kpdf")]` in `lib.rs`. A `--no-default-features` build never sees this
//! module, and never pulls in egui.

pub mod forms;
pub mod geometry;
pub mod hit_test;
pub mod keys;
pub mod lru;
pub mod pixmap;
pub mod tools;
pub mod zoom;

pub use forms::{
    FieldHighlight, consume_commit_enter, field_highlight_kind, highlight_colors,
    should_commit_on_enter,
};
pub use geometry::{
    ContinuousSlot, PageLayout, PageSize, continuous_slot_visible, current_page_in_view,
    field_rect_to_screen, layout_continuous_pages, min_hit_rect, page_display_size, page_size_pts,
    page_to_screen, recentred_scroll_offset, rect_contains, screen_to_page, screen_to_page_at,
};
pub use hit_test::{hit_test_annot, hit_test_field, hit_test_field_expanded};
pub use keys::{
    Command, G_PENDING_TIMEOUT, GPending, VIM_STEP, g_pending_expired, half_viewport_step,
    keys_captured, parse_command,
};
pub use lru::Lru;
pub use pixmap::{drawable_annot_count, rgb_to_rgba};
pub use tools::{Tool, select_tool, toggle_forms_mode};
pub use zoom::{
    DPI_DEFAULT, DPI_MAX, DPI_MIN, DPI_STEP, ZOOM_DELTA_PER_STEP, zoom_percent,
    zoom_steps_from_zoom_delta,
};
