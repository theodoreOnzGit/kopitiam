//! Native replacements for the maintainer's 20 Lua plugins.
//!
//! # Why these exist
//!
//! `kvim` replaces a Neovim setup built from `lazy.nvim` plugins (telescope,
//! neo-tree, harpoon, hop, vim-easy-align, vim-fugitive/airline) with plugin
//! *behaviour* compiled directly into the binary. There is nothing to
//! download, so there is nothing a plugin manager or Mason needs to do —
//! which is also what makes `cargo install kopitiam-neovim` work on Android,
//! where those tools shell out to toolchains (`npm`, `pip`, `go`) that are
//! not there.
//!
//! # The contract every module here follows
//!
//! Each module is a headless engine: it takes data in (lines, paths, a
//! buffer's cached content) and returns data out (matches, edits, tree rows,
//! hints). **None of them touch the terminal.** [`crate::ui`] is the only
//! place allowed to call `crossterm`/`ratatui`; that split is what makes
//! every module below unit-testable without a terminal, a TTY, or a running
//! editor — see each module's tests for exactly that.
//!
//! | Module | Replaces | Driven by [`crate::config::Action`] |
//! |---|---|---|
//! | [`picker`] | telescope.nvim + plenary | `FindFiles`, `FindBuffers`, `FindHelp` |
//! | [`filetree`] | neo-tree + nui + web-devicons | `FileTreeToggle` |
//! | [`harpoon`] | ThePrimeagen/harpoon (v1) | `HarpoonAdd`, `HarpoonMenu`, `HarpoonFind` |
//! | [`hop`] | hop.nvim | `HopWords` |
//! | [`align`] | vim-easy-align | `EasyAlign` |
//! | [`git`] | vim-fugitive/airline (statusline slice only) | (consumed by the statusline, not a keymap) |
//! | [`grep`] | vim's `:grep`/`grepprg` (pure-Rust, no shell-out) | (driven by the `:grep`/`:vimgrep` ex commands, not a keymap) |

pub mod align;
pub mod filetree;
pub mod git;
pub mod grep;
pub mod harpoon;
pub mod hop;
pub mod picker;
