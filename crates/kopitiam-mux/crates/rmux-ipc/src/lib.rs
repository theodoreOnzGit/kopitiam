#![deny(missing_docs)]

//! Local IPC boundary for RMUX.
//!
//! This crate owns endpoint naming and local transport handles. It deliberately
//! transports bytes only; the RMUX request/response protocol stays in
//! `rmux-proto`.
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

mod endpoint;
mod listener;
mod stream;
#[cfg(windows)]
mod windows_mutex;

pub use endpoint::{
    default_endpoint, endpoint_for_label, resolve_endpoint, resolve_tmux_compatible_endpoint,
    LocalEndpoint,
};
pub use listener::LocalListener;
pub use stream::{
    connect_blocking, is_peer_disconnect, wait_for_peer_close, BlockingLocalStream, LocalStream,
    PeerIdentity,
};
#[cfg(windows)]
pub use windows_mutex::{
    acquire_named_mutex, NamedMutexAcquire, NamedMutexError, NamedMutexGuard, MAX_NAMED_MUTEX_LEN,
};
