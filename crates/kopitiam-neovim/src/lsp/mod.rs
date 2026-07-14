//! The language-server layer: kvim's Mason replacement, LSP client, and the
//! `blink.cmp` replacement — the feature that makes kvim viable on Android.
//!
//! # The problem this module exists to solve
//!
//! The maintainer's Neovim installs `rust-analyzer`, `lua-language-server`,
//! and `texlab` through `mason.nvim`. Mason's install step for every one of
//! those three shells out to a language-specific package manager: `npm` for
//! some servers, `pip`/`go install` for others, sometimes a `cargo install`
//! that compiles from source. On a typical Android execution environment
//! none of those toolchains are present (and `cargo install`-from-source
//! specifically is a non-starter for `rust-analyzer` regardless of
//! toolchain availability — compiling it takes minutes and gigabytes of RAM
//! even on a desktop). That is the actual, specific reason the maintainer's
//! setup does not work on their phone; it has nothing to do with Lua. See
//! `docs/ai-decisions/AID-0003-kopitiam-neovim-architecture.md`.
//!
//! Fixing it means answering three questions without ever invoking a
//! package manager:
//!
//! 1. **Which server serves this filetype, and how do I get it?**
//!    [`registry`] — a static table (name, executable, filetypes, and a
//!    per-`(os, arch)` prebuilt-binary download URL) that stands in for
//!    Mason's `ensure_installed`.
//! 2. **Given a download URL, how does the binary end up runnable?**
//!    [`install`] — fetch (pure-Rust TLS only — see that module's doc
//!    comment for why this crate does not yet perform the fetch itself),
//!    verify, unpack, and `chmod +x`, following the same `$XDG_DATA_HOME` /
//!    `$HOME/.local/share` convention [`crate::config`] uses for
//!    `$XDG_CONFIG_HOME`.
//! 3. **Once a server is running, how does kvim actually talk to it?**
//!    [`client`] wraps the LSP transport `kopitiam-semantic` already has
//!    working (rather than writing a second one — see that module's doc
//!    comment for exactly what is and isn't reachable through its public
//!    API today), and [`position`] handles the one part of that
//!    conversation that silently corrupts data if done carelessly:
//!    translating between kvim's grapheme-indexed
//!    [`Position`](crate::core::Position) and whichever of LSP's three wire
//!    position encodings a given server negotiates.
//!
//! [`completion`] is the `blink.cmp` replacement: it merges candidates from
//! the LSP, the current buffer's words, and file paths into one ranked,
//! headless list for the UI layer to render.
//!
//! # Priority order ([`registry`]'s `Source`)
//!
//! 1. Already on `PATH` — use it. Downloading something the user already
//!    has is rude, and actively hostile on a metered connection.
//! 2. A direct download URL for a prebuilt **static** binary, resolved per
//!    `(os, arch)`.
//! 3. Honestly unavailable. **Never** a silent fallback to `npm`/`pip`/
//!    `go install` — that fallback is the bug this module exists to fix.

pub mod client;
pub mod completion;
pub mod install;
pub mod position;
pub mod registry;

pub use client::{Diagnostic, Location, LspClient, LspError, Severity};
pub use completion::{CompletionItem, CompletionSource};
pub use install::Plan as InstallPlan;
pub use position::{LspPosition, PositionEncoding};
pub use registry::{Arch, LanguageServer, Os, Source as RegistrySource, Target};
