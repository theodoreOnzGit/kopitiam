//! The overlay layer: panels that take focus, sit over (or beside) the
//! window tree, and hand keys back to the app as *outcomes*.
//!
//! # Why the file tree is an overlay and not a leaf in [`WindowTree`]
//!
//! In Neovim, neo-tree really is a window — because Neovim's window model is
//! "a viewport onto a *buffer*", and a buffer with `buftype=nofile` can hold
//! anything, including a rendered directory listing. kvim's window model is
//! narrower on purpose: [`crate::ui::window::Window`] is `{ buffer: BufferId,
//! cursor: Position, scroll: Scroll }` — a viewport onto *text*, with a
//! grapheme-indexed cursor. The file tree has no `BufferId`, no text, and its
//! "cursor" is a row index into a flattened directory listing, not a
//! `Position`. Putting it in the tree would mean one of:
//!
//! * inventing a fake `BufferId` that points at no buffer — and then
//!   [`crate::ui::app::App::render_windows`], which renders `host.buffer()`
//!   into every leaf, would happily paint the current file's text into the
//!   sidebar; or
//! * widening `Window` into an enum (`Window::Text | Window::Tree`), which
//!   changes the meaning of `WindowTree` for every existing caller, including
//!   `active_mut().cursor = host.cursor()` in the event loop — which would
//!   start writing the *text* cursor into the tree window.
//!
//! Neither is worth it for a panel that never splits, never scrolls
//! horizontally, and never shows a buffer. So the sidebar instead **carves its
//! strip out of the frame before the window tree is laid out** (see
//! [`OverlayPlacement::split`]): `WindowTree` keeps its whole area, unchanged,
//! minus 30 columns.
//!
//! The consequences of that choice, stated plainly so nobody has to rediscover
//! them:
//!
//! * `:sp`/`:vs` keep working and are entirely unaware of the sidebar.
//! * `:q` closes a *buffer* window and can never close the tree. The tree is
//!   closed with `q`, `<Esc>`, or `<leader>e` — which is what neo-tree users
//!   press anyway.
//! * Focus still moves between the tree and the editor with `<C-h>`/`<C-l>`
//!   (and their `<C-w>h`/`<C-w>l` forms): because the tree is not a
//!   [`WindowTree`] leaf, [`crate::ui::app::App::move_focus`] special-cases it
//!   — a leftward move off the leftmost editor window focuses the tree, and a
//!   rightward move from the tree returns to the editor. The [`Focus`] flag
//!   below is what those moves flip; nothing about the tree being an overlay
//!   rather than a window is visible to the user pressing those keys.
//!
//! If kvim ever grows Neovim's "a buffer can be anything" model, this decision
//! is worth revisiting — but it should be revisited *because the buffer model
//! changed*, not because a sidebar felt like it ought to be a window.
//!
//! # Why this layer is generic over one overlay today
//!
//! `<leader>e` (the file tree) is the first of six actions that all need the
//! same three things: reserve or float a rectangle, take focus while open, and
//! translate their own keys into a small set of things the app can do. The
//! pickers (`\ff`, `\fb`, `\fh`), the harpoon menu (`<leader><Esc>`) and the
//! hop label overlay (`f`) are the same shape with a different body. So the
//! shape is named here — [`OverlayPlacement`], [`OverlayOutcome`], [`Focus`],
//! and the [`Overlay`] enum — and only the file tree fills it in today. Adding
//! the picker is a variant plus its match arms, not a rewrite of the app's
//! focus and layout handling.
//!
//! [`WindowTree`]: crate::ui::window::WindowTree

use std::path::PathBuf;

use ratatui::{layout::Rect, Frame};

use crate::core::{BufferId, Position};
use crate::icons::IconSet;
use crate::ui::event::KeyPress;
use crate::ui::filetree::FileTreePanel;
use crate::ui::harpoon::HarpoonMenuPanel;
use crate::ui::picker::PickerPanel;
use crate::ui::theme::Theme;

/// Where keystrokes go. Modelled explicitly, and *never* inferred from "is an
/// overlay open" — the tree can be visible while the buffer has focus (in which
/// case the tree is drawn, but inert), which is a state no single boolean can
/// express without lying about one of the two cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// Keys go to the editor: `j` moves the *text* cursor.
    Buffer,
    /// Keys go to the open overlay: `j` moves the *overlay's* cursor and the
    /// editor never sees the key at all.
    Overlay,
}

/// How an overlay claims screen space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPlacement {
    /// A full-height strip down the left edge, *reserving* its columns: the
    /// window tree is laid out in what remains. This is the file tree.
    LeftSidebar { width: u16 },
    /// A centred box floating **over** the window tree, reserving nothing.
    /// Nothing uses this yet; it is the shape the fuzzy pickers and the
    /// harpoon menu need (telescope does not resize your buffers), and it
    /// exists so that landing them is a variant and an arm, not a redesign of
    /// [`OverlayPlacement::split`].
    Float { width_pct: u16, height_pct: u16 },
}

impl OverlayPlacement {
    /// Divides `area` into `(overlay_rect, windows_rect)`.
    ///
    /// A sidebar shrinks the windows' rect; a float leaves it untouched and is
    /// simply painted afterwards. Both clamp so that the buffer always keeps at
    /// least one column — a sidebar that eats the entire terminal on a narrow
    /// (phone-sized) screen is not a sidebar, it is a bug, and this crate
    /// targets Android.
    pub fn split(self, area: Rect) -> (Rect, Rect) {
        match self {
            Self::LeftSidebar { width } => {
                let width = sidebar_width(width, area.width);
                let sidebar = Rect { width, ..area };
                // Reserve one column between the sidebar and the windows for the
                // tree/editor `WinSeparator`, so the border does not overpaint the
                // first column of buffer text. The window strip therefore starts
                // one column further right than the sidebar's own width. When the
                // terminal is too narrow to spare that column (a phone-sized
                // screen), the border is dropped rather than eating the buffer.
                let border = if area.width > width.saturating_add(1) { 1 } else { 0 };
                let windows = Rect {
                    x: area.x + width + border,
                    width: area.width.saturating_sub(width + border),
                    ..area
                };
                (sidebar, windows)
            }
            Self::Float { width_pct, height_pct } => {
                let w = (u32::from(area.width) * u32::from(width_pct.min(100)) / 100) as u16;
                let h = (u32::from(area.height) * u32::from(height_pct.min(100)) / 100) as u16;
                let float = Rect {
                    x: area.x + (area.width.saturating_sub(w)) / 2,
                    y: area.y + (area.height.saturating_sub(h)) / 2,
                    width: w,
                    height: h,
                };
                (float, area)
            }
        }
    }
}

/// Clamps a sidebar's requested width to something the terminal can actually
/// spare: never more than two fifths of the screen, and never so wide that the
/// buffer is left with no columns at all.
fn sidebar_width(requested: u16, total: u16) -> u16 {
    let two_fifths = (u32::from(total) * 2 / 5) as u16;
    requested.min(two_fifths).min(total.saturating_sub(1))
}

/// Which window an overlay wants a file opened in.
///
/// Mirrors NERDTree's `o` / `i` / `s` / `t`. `Tab` opens the file in a new tab
/// page now (see [`crate::ui::tab`]); the app routes it through
/// `App::open_tab`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenTarget {
    Current,
    HorizontalSplit,
    VerticalSplit,
    Tab,
}

/// What the app must do after the focused overlay has seen a key.
///
/// The overlay never touches the editor, the window tree, or the command line
/// itself — it says what it wants and the app does it. That is the same
/// separation `EditorResponse` draws between the editor and its caller, for the
/// same reason: an overlay that can reach into the editor is an overlay you
/// cannot unit-test without one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayOutcome {
    /// The key meant nothing here; nothing changed, so nothing needs redrawing.
    Ignored,
    /// Handled; the overlay's own state changed, so redraw.
    Consumed,
    /// Close the overlay and give focus back to the buffer.
    Close,
    /// Open `path` in the editor, then focus the buffer. The overlay stays open
    /// (neo-tree keeps the tree visible after you open a file from it).
    OpenPath { path: PathBuf, target: OpenTarget },
    /// A picker confirmed a file (`\ff`): open it in the current window and
    /// **close** the picker. Distinct from [`OverlayOutcome::OpenPath`], which
    /// keeps its overlay open — telescope disappears the moment you pick, so the
    /// pickers say "close" rather than reusing the tree's stay-open semantics.
    PickPath(PathBuf),
    /// A picker confirmed a buffer (`\fb`): switch to it and close the picker.
    PickBuffer(BufferId),
    /// A picker confirmed a help topic (`\fh`): run `:help <topic>` and close
    /// the picker.
    PickHelp(String),
    /// The harpoon quick menu (`<leader><Esc>`) confirmed a mark: open `path`
    /// **at `cursor`** and close the menu. Distinct from
    /// [`OverlayOutcome::PickPath`], which lands at the top of the file — a
    /// harpoon mark's whole value is jumping back to the *line* you were on, so
    /// it carries the saved position the app then restores.
    OpenAt { path: PathBuf, cursor: Position },
    /// The harpoon quick menu deleted a line: drop the mark at this index from
    /// the canonical list. The menu **stays open** (you keep editing the list),
    /// so unlike the pick family this does not close the overlay.
    HarpoonRemove(usize),
    /// Show an informational message on the command line.
    Message(String),
    /// Show an error on the command line.
    Error(String),
}

/// The open overlay, if any. Exactly one at a time: opening a picker over the
/// tree replaces it, which is what neo-tree + telescope do in practice (the
/// picker takes focus, and closing it returns you to where you were).
pub enum Overlay {
    FileTree(FileTreePanel),
    /// A fuzzy picker (`\ff`/`\fb`/`\fh`) — see [`crate::ui::picker`]. Floats
    /// over the window tree rather than reserving a strip, which is why
    /// [`OverlayPlacement::Float`] finally has a user.
    Picker(PickerPanel),
    /// The harpoon quick menu (`<leader><Esc>`) — a small floating, editable
    /// list of the project's marks. See [`crate::ui::harpoon`].
    HarpoonMenu(HarpoonMenuPanel),
}

impl Overlay {
    /// How this overlay wants to be placed within the windows' area.
    pub fn placement(&self) -> OverlayPlacement {
        match self {
            Self::FileTree(_) => OverlayPlacement::LeftSidebar { width: FileTreePanel::WIDTH },
            // A telescope-sized float: wide and tall enough to browse, but not
            // fullscreen — you can still see the buffer it sits over.
            Self::Picker(_) => OverlayPlacement::Float { width_pct: 80, height_pct: 80 },
            // The harpoon menu holds at most a handful of marks, so it floats
            // smaller than the pickers — enough to read the list, no more.
            Self::HarpoonMenu(_) => OverlayPlacement::Float { width_pct: 60, height_pct: 50 },
        }
    }

    /// Feeds one key to the overlay. Only ever called when [`Focus::Overlay`]
    /// holds, so the editor genuinely never sees these keys.
    pub fn handle_key(&mut self, key: KeyPress) -> OverlayOutcome {
        match self {
            Self::FileTree(panel) => panel.handle_key(key),
            Self::Picker(panel) => panel.handle_key(key),
            Self::HarpoonMenu(panel) => panel.handle_key(key),
        }
    }

    /// Draws the overlay into `rect`, returning where the terminal cursor
    /// should sit (`None` to leave it hidden). Takes `&mut self` because an
    /// overlay's scroll offset depends on the height it is being drawn at,
    /// which it does not learn until now.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        rect: Rect,
        theme: &Theme,
        icons: IconSet,
        focused: bool,
    ) -> Option<(u16, u16)> {
        match self {
            Self::FileTree(panel) => panel.render(frame, rect, theme, icons, focused),
            Self::Picker(panel) => panel.render(frame, rect, theme, icons, focused),
            Self::HarpoonMenu(panel) => panel.render(frame, rect, theme, icons, focused),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sidebar_reserves_its_columns_from_the_left_plus_a_border() {
        let area = Rect { x: 0, y: 0, width: 100, height: 24 };
        let (side, windows) = OverlayPlacement::LeftSidebar { width: 30 }.split(area);
        assert_eq!(side, Rect { x: 0, y: 0, width: 30, height: 24 });
        // One column between them is reserved for the tree/editor separator, so
        // the windows start at 31, not 30, and the three parts fill the area.
        assert_eq!(windows, Rect { x: 31, y: 0, width: 69, height: 24 });
        assert_eq!(side.width + 1 + windows.width, area.width, "sidebar + border + windows fill the area");
    }

    #[test]
    fn a_sidebar_never_takes_more_than_two_fifths_of_a_narrow_screen() {
        // A phone terminal: 40 columns. A 30-column sidebar would leave 10.
        let area = Rect { x: 0, y: 0, width: 40, height: 20 };
        let (side, windows) = OverlayPlacement::LeftSidebar { width: 30 }.split(area);
        assert_eq!(side.width, 16); // 40 * 2/5
        // 40 - 16 sidebar - 1 border = 23 for the windows.
        assert_eq!(windows.width, 23);
        assert_eq!(windows.x, 17);
    }

    #[test]
    fn a_sidebar_always_leaves_the_buffer_at_least_one_column() {
        for width in 0..=4u16 {
            let area = Rect { x: 0, y: 0, width, height: 5 };
            let (side, windows) = OverlayPlacement::LeftSidebar { width: 30 }.split(area);
            assert!(side.width < width.max(1), "sidebar ate the whole {width}-column screen");
            // The border column is dropped on a screen too narrow to spare it, so
            // sidebar + (0-or-1 border) + windows always still tile the area.
            let border = width - side.width - windows.width;
            assert!(border <= 1, "at most one column goes to the border");
            assert_eq!(side.width + border + windows.width, width);
        }
    }

    #[test]
    fn a_float_reserves_nothing_and_is_centred() {
        let area = Rect { x: 0, y: 0, width: 100, height: 40 };
        let (float, windows) = OverlayPlacement::Float { width_pct: 50, height_pct: 50 }.split(area);
        assert_eq!(windows, area, "a float must not resize the windows beneath it");
        assert_eq!(float, Rect { x: 25, y: 10, width: 50, height: 20 });
    }
}
