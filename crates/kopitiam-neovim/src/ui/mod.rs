//! The terminal user interface: kvim's ratatui/crossterm front end.
//!
//! # What this module is, architecturally
//!
//! Per `CLAUDE.md`'s "never place business logic inside user interfaces",
//! this module contains **no editor semantics**. It does not know what `dw`
//! means, it does not decide where the cursor goes when you press `j`, and it
//! does not implement `:s/foo/bar/`. All of that lives in [`crate::editor`].
//! This module does exactly two things:
//!
//! 1. **Translates** terminal input (crossterm key/resize/paste events) into
//!    the small vocabulary the editor understands ([`event::KeyPress`]).
//! 2. **Renders** whatever state the editor reports (mode, cursor, buffer
//!    contents) into terminal cells (ratatui widgets).
//!
//! Everything in between — motions, operators, registers, undo, ex commands —
//! is out of scope here by design, not by omission.
//!
//! # The `editor` seam
//!
//! At the time this module was written, `crate::editor` (owned by a
//! concurrently-working agent) was still a placeholder: no `Editor`, `Key`,
//! or `EditorResponse` types existed yet to compile against. Rather than
//! block on that landing, this module defines its own narrow traits —
//! [`event::EditorHost`] and [`event::BufferView`] — that describe exactly
//! the surface the UI needs (mode, cursor, buffer text; feed a key, get a
//! response). [`event::KeyPress`] is this module's own stand-in for the
//! editor's forthcoming `Key` type.
//!
//! Every renderer, the scrolling math, and the event loop in [`app`] are
//! written against these local traits, never against `crate::editor`
//! directly. When the real `editor::Editor` / `editor::Key` /
//! `editor::EditorResponse` land, wiring them in is a matter of writing one
//! small adapter `impl EditorHost for editor::Editor` (and, if the shapes
//! differ, a `KeyPress::from(editor::Key)` conversion) — not reshaping the
//! renderer, the scrolling logic, or the tests, all of which are already
//! exercised against the trait using lightweight fakes.
//!
//! # Layout
//!
//! * [`terminal`] — raw mode / alternate screen lifecycle, and the panic
//!   hook that guarantees the terminal is restored even when kvim crashes.
//! * [`event`] — the crossterm→editor key mapping, and the `EditorHost` /
//!   `BufferView` seam traits described above.
//! * [`theme`] — colour palettes as data, starting with gruvbox dark.
//! * [`scrolling`] — pure functions for scrolloff-aware vertical scrolling
//!   and horizontal (no-wrap) scrolling.
//! * [`gutter`] — hybrid (`number` + `relativenumber`) line-number labels.
//! * [`textarea`] — the main buffer viewport: gutter, colorcolumn, tab
//!   expansion, unicode-width-correct rendering, and cursor shape/placement.
//! * [`statusline`] — the vim-airline-style powerline statusline.
//! * [`cmdline`] — the `:`/`/`/`?` command line and message area.
//! * [`window`] — the window/split tree (`:sp`, `:vs`).
//! * [`tab`] — the tab-page collection ([`tab::TabPages`]): an ordered set of
//!   [`window::WindowTree`]s with one active, the layer above `window`. A tab
//!   is a whole window layout, vim-style, not a browser buffer-tab.
//! * [`tabline`] — the top-row tabline widget that paints those tab pages.
//! * [`overlay`] — the focus/placement layer shared by every panel that takes
//!   over the keyboard: the file tree today, the fuzzy pickers, hop and the
//!   harpoon menu next. Its module docs carry the reasoning for why a sidebar
//!   is *not* a leaf in [`window`]'s tree.
//! * [`filetree`] — the file-tree sidebar (`<leader>e`), presenting
//!   [`crate::plugins::filetree`] with NERDTree's in-tree keymaps.
//! * [`harpoon`] — the harpoon quick menu (`<leader><Esc>`), an editable
//!   floating list over [`crate::plugins::harpoon`]'s marks.
//! * [`app`] — the event loop that ties the above together.
//! * [`bootstrap`] — [`run`], the entry point `main.rs` calls, plus the
//!   permanent `impl BufferView for text::Buffer` and the temporary
//!   `EditorHost` placeholder used until `crate::editor::Editor` exists —
//!   see that module's docs for why the two are not the same kind of
//!   "temporary".
//!
//! # Key sequences: whose job?
//!
//! The editor owns keymap resolution (`<leader>e`, `\ff`, `ga`) — it has the
//! compiled table and the buffering state machine, and it is the thing with
//! focus most of the time. But an overlay with focus is *not* feeding keys to
//! the editor, so the two multi-key sequences that must work from inside an
//! overlay (`<leader>e` to close the tree, and later `<C-x>`-style picker keys)
//! are resolved by the overlay itself. That is not a second keymap engine; it
//! is the same "translate terminal input into the vocabulary of whatever has
//! focus" job this module already exists to do.

pub mod app;
pub mod bootstrap;
pub mod cmdline;
pub mod completion_menu;
pub mod event;
pub mod filetree;
pub mod gutter;
pub mod harpoon;
pub mod highlight;
pub mod hop;
pub mod lsp_ui;
pub mod overlay;
pub mod picker;
pub mod scrolling;
pub mod snippet;
pub mod statusline;
pub mod tab;
pub mod tabline;
pub mod termgrid;
pub mod terminal;
pub mod textarea;
pub mod theme;
pub mod whichkey;
pub mod window;

#[cfg(test)]
pub mod test_support;

pub use app::App;
pub use bootstrap::run;
pub use event::{BufferView, EditorHost, HostResponse, KeyPress};
pub use filetree::FileTreePanel;
pub use overlay::{Focus, OpenTarget, Overlay, OverlayOutcome, OverlayPlacement};
pub use tab::TabPages;
pub use terminal::TerminalGuard;
pub use theme::Theme;
pub use window::{SplitKind, WindowTree};
