#![deny(missing_docs)]

//! Small OS-boundary helpers for RMUX.
//!
//! This crate is intentionally narrow. Add modules only when a real migrated
//! call site consumes them in the same change.
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

pub mod daemon;
pub mod host;
pub mod identity;
pub mod memory;
pub mod process;
#[cfg(unix)]
pub mod runtime_dir;
pub mod signals;
pub mod terminal;
