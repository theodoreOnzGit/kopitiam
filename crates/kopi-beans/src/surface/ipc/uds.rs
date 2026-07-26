//! Cross-platform Unix-domain-socket types.
//!
//! The beads daemon transport is an AF_UNIX socket addressed by a filesystem
//! path. On unix these types come straight from `std`. Windows 10 (1803+) also
//! supports AF_UNIX sockets with filesystem paths, but Rust's `std` does not
//! expose them, so on Windows we re-export the API-compatible types from the
//! `uds_windows` crate. Because both speak filesystem paths, none of the
//! socket-path resolution logic in this crate needs to change.
//!
//! TODO(windows): `uds_windows` covers connect/bind/accept/read/write and
//! timeouts, which is everything the client and daemon use. If a future code
//! path needs `SocketAncillary`/fd-passing (not used today), it must be gated.

#[cfg(unix)]
pub use std::os::unix::net::{UnixListener, UnixStream};

#[cfg(windows)]
pub use uds_windows::{UnixListener, UnixStream};
