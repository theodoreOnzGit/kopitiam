//! kmux: a terminal multiplexer for KOPITIAM.
//!
//! This is the single-crate collapse of the former multi-crate `rmux` fork
//! (upstream `rmux`, MIT OR Apache-2.0), relicensed AGPL-3.0-only as part of
//! KOPITIAM, with Android/Termux (`cfg(unix)`) and Windows support. The ten
//! former `rmux-*` sub-crates are now the intra-crate modules below
//! (`rmux_core::X` -> `crate::core::X`, etc.). See NOTICE for provenance and the
//! record of the collapse.
//!
//! The `kmux` and `kmux-daemon` binaries link this library and reach the folded
//! surface as `kmux::client::…`, `kmux::server::…`, and so on.

// The former rmux-web-crypto crate declared `extern crate alloc;` at its own
// crate root; folded into a module, its submodules (`use alloc::vec::Vec;`) need
// `alloc` in the extern prelude crate-wide, which only a crate-root declaration
// provides. Harmless when unused (the web_crypto module is the only consumer).
extern crate alloc;

// Former rmux-types.
pub mod types;
// Former rmux-os.
pub mod os;
// Former rmux-proto.
pub mod proto;
// Former rmux-core.
pub mod core;
// Former rmux-ipc.
pub mod ipc;
// Former rmux-pty.
pub mod pty;
// Former rmux-sdk.
pub mod sdk;
// Former rmux-client.
pub mod client;
// Former rmux-server.
pub mod server;
// Former rmux-web-crypto. Only compiled with the web-share E2EE surface, which
// is the only consumer (the server's web feature); mirrors the upstream
// `dep:rmux-web-crypto` being pulled in solely by `rmux-server/web`.
#[cfg(feature = "web")]
pub mod web_crypto;
