#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![forbid(unsafe_code)]

//! Pure-data RMUX pane rendering core.
//!
//! This crate owns no daemon, IPC, process, filesystem, network, Tokio, or
//! terminal-driver integration. It contains only captured pane snapshot data and
//! deterministic ratatui projection code that can compile for
//! `wasm32-unknown-unknown`.
//! ---
//!
//! **Part of `kopitiam-mux`, a fork of [rmux](https://github.com/helvesec/rmux).**
//!
//! This crate's code was written by **The RMUX Authors** and is reused directly
//! under its original **MIT OR Apache-2.0** license (see `LICENSE-MIT` and
//! `LICENSE-APACHE` in `crates/kopitiam-mux/`). It is distributed as part of
//! KOPITIAM under **AGPL-3.0-only**. See `crates/kopitiam-mux/NOTICE`.
//!
//! KOPITIAM's changes add Android/Termux support. `rmux_os::runtime_dir`
//! documents every Android decision in the fork; read it before changing a
//! `cfg` gate.

mod snapshot;
mod state;
mod theme;
mod widget;

pub use snapshot::{
    PaneAttributes, PaneCell, PaneColor, PaneCursor, PaneGlyph, PaneSnapshot,
    PaneSnapshotShapeError,
};
pub use state::PaneState;
pub use theme::{cell_style, color, glyph_symbol, modifier};
pub use widget::PaneWidget;
