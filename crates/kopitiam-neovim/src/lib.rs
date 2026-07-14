//! **kvim** — a Rust-native, Android-capable modal editor.
//!
//! # Not a KDE application
//!
//! Despite the leading `k`, **kvim is unrelated to KDE and is not part of the
//! KDE Plasma workspace.** It is not a Qt application, it does not depend on
//! KDE Frameworks, and it has nothing to do with the historical KDE Vim
//! front-end of a similar name. The `k` is for KOPITIAM, whose other binary is
//! `kopitiam` and whose terminal multiplexer is `kmux` — the convention is this
//! project's, and the collision with KDE's `k*` naming is coincidental.
//!
//! kvim is KOPITIAM's answer to "Neovim, but it works on my phone." It is a
//! from-scratch modal editor with the Vim keybinding grammar, an LSP client,
//! and a full plugin suite **built in as native Rust** rather than downloaded
//! as Lua at runtime.
//!
//! # Why this exists
//!
//! The maintainer's Neovim setup is 20 Lua plugins managed by `lazy.nvim`,
//! with language servers installed by Mason. On Android that breaks — not
//! because of Lua, but because Mason shells out to `npm`, `pip`, and
//! `go install` to fetch servers, and those toolchains aren't there. Fixing
//! that properly means owning the whole stack:
//!
//! * **No plugin manager.** A plugin manager exists to install third-party
//!   downloads. When the plugins are compiled in, there is nothing to install,
//!   so `lazy.nvim` simply has no job here.
//! * **No Mason.** Language servers are acquired as prebuilt static binaries
//!   over pure-Rust TLS, or vendored outright. See [`lsp`].
//! * **Batteries included.** The Nerd Font itself ships inside the binary,
//!   because an icon is a codepoint, and a codepoint without a font is a tofu
//!   box. See [`icons`].
//!
//! # Layout
//!
//! * [`core`] — the vocabulary every subsystem agrees on (positions, modes,
//!   edits). Pure types, no logic.
//! * [`text`] — the rope buffer, undo tree, and marks.
//! * [`editor`] — the modal state machine: modes, motions, operators,
//!   registers, macros, ex commands.
//! * [`ui`] — terminal rendering, windows, statusline, themes.
//! * [`plugins`] — native replacements for the Lua plugins (fuzzy finder,
//!   file tree, harpoon, hop, align).
//! * [`lsp`] — the language-server client and the Mason replacement.
//! * [`config`] — configuration, whose defaults *are* the maintainer's Neovim
//!   config, ported.
//! * [`icons`] — devicons, with a three-tier fallback and an embedded font.
//!
//! # Status
//!
//! Under active construction. See `docs/ai-decisions/AID-0003` for the
//! architecture and its open questions, and bead `kopitiam-cj0` for the phase
//! plan.

pub mod config;
pub mod core;
pub mod editor;
pub mod icons;
pub mod lsp;
pub mod plugins;
pub mod text;
pub mod ui;

pub use config::Config;
pub use core::{BufferId, Edit, Error, Granularity, Mode, Position, Range, Result, WindowId};
