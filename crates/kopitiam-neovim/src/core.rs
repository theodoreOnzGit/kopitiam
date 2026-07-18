//! The vocabulary every kvim subsystem agrees on.
//!
//! This module is the editor's equivalent of `kopitiam-ontology`: pure types,
//! no logic, no I/O. The text engine, the modal state machine, the renderer,
//! the LSP client, and the built-in plugins all speak these types, so none of
//! them has to depend on another merely to name a cursor position.
//!
//! # Why positions are (line, grapheme), not byte offsets
//!
//! An editor that indexes by byte offset gets CJK and emoji wrong the moment a
//! user presses `l`. One that indexes by `char` (Unicode scalar) still gets
//! combining marks and ZWJ emoji sequences wrong — `👨‍👩‍👧` is one thing on
//! screen but seven `char`s. The unit a *user* moves by is the grapheme
//! cluster, so that is the unit the cursor is measured in. Byte offsets still
//! exist, but only at the rope boundary, where they are an implementation
//! detail — see [`Position`].

use std::fmt;
use std::path::PathBuf;

/// A cursor position: zero-based line, zero-based grapheme column.
///
/// Both fields are display-oriented, not storage-oriented. Converting to and
/// from a rope byte offset is the text engine's job, and deliberately not
/// expressible here — that keeps every other subsystem from accidentally
/// doing byte arithmetic on UTF-8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Position {
    /// Zero-based line index.
    pub line: usize,
    /// Zero-based **grapheme** column within the line.
    pub col: usize,
}

impl Position {
    pub const fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }

    /// The origin, i.e. the first grapheme of the first line.
    pub const ORIGIN: Self = Self { line: 0, col: 0 };
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 1-based when shown to a human, because that is what the ruler and
        // every `:1234` jump means to them.
        write!(f, "{}:{}", self.line + 1, self.col + 1)
    }
}

/// An inclusive-start, exclusive-end span of text.
///
/// `anchor` is where the selection began and `head` is where the cursor is;
/// `head` may be *before* `anchor` when the user selected backwards. Use
/// [`Range::normalized`] when you need (start, end) ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Range {
    pub anchor: Position,
    pub head: Position,
}

impl Range {
    pub const fn new(anchor: Position, head: Position) -> Self {
        Self { anchor, head }
    }

    /// A zero-width range at `pos` — the shape a plain cursor takes.
    pub const fn point(pos: Position) -> Self {
        Self { anchor: pos, head: pos }
    }

    /// `(start, end)` in document order, regardless of selection direction.
    pub fn normalized(self) -> (Position, Position) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }

    pub fn is_empty(self) -> bool {
        self.anchor == self.head
    }
}

/// Which editing mode the editor is in.
///
/// The three visual variants are separate modes rather than a flag on one
/// `Visual`, because operators genuinely behave differently in each: `d` in
/// visual-line deletes whole lines, in visual-block it deletes a rectangle.
/// Collapsing them forces every operator to re-derive which it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Mode {
    #[default]
    Normal,
    Insert,
    Visual,
    VisualLine,
    VisualBlock,
    Replace,
    /// The `:` command line (also used for `/` and `?` searches).
    Command,
    /// An operator has been given and kvim is waiting for the motion that
    /// tells it what to operate on — the `d` in `d2w`, before the `2w`.
    OperatorPending,
}

impl Mode {
    /// The text shown in the statusline, matching vim's `--INSERT--` style.
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "NORMAL",
            Self::Insert => "INSERT",
            Self::Visual => "VISUAL",
            Self::VisualLine => "V-LINE",
            Self::VisualBlock => "V-BLOCK",
            Self::Replace => "REPLACE",
            Self::Command => "COMMAND",
            Self::OperatorPending => "O-PENDING",
        }
    }

    pub fn is_visual(self) -> bool {
        matches!(self, Self::Visual | Self::VisualLine | Self::VisualBlock)
    }
}

/// Whether a motion or register operates on whole lines or on characters.
///
/// This is what makes `dd` paste back as a whole line while `dw` pastes
/// inline — the *register* remembers which it was. Losing this distinction is
/// the classic way a vim clone ends up pasting text in the wrong place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Granularity {
    #[default]
    Charwise,
    Linewise,
    Blockwise,
}

/// Errors an editor operation can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("position {pos} is outside the buffer ({lines} lines)")]
    PositionOutOfBounds { pos: Position, lines: usize },

    #[error("no buffer {0:?}")]
    NoSuchBuffer(BufferId),

    #[error("nothing to undo")]
    NothingToUndo,

    #[error("nothing to redo")]
    NothingToRedo,

    #[error("unknown command: {0:?}")]
    UnknownCommand(String),

    #[error("invalid pattern {pattern:?}: {reason}")]
    InvalidPattern { pattern: String, reason: String },

    #[error("buffer has unsaved changes (add ! to override)")]
    UnsavedChanges,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Identifies an open buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u32);

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies a window (a viewport onto a buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WindowId(pub u32);

/// A spatial direction, used by `<C-w>h/j/k/l` window navigation.
///
/// Lives in `core` (the shared vocabulary) rather than in `ui::window`
/// because the editor grammar and the UI's window tree both need to name a
/// direction, and neither should have to depend on the other to do so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

/// A request to reposition the *viewport* relative to the cursor, or vice
/// versa — the `zz`/`zt`/`zb` family and `<C-e>`/`<C-y>`.
///
/// # Why this is a request the editor emits rather than an edit it performs
///
/// The viewport (which line is at the top of a window, how tall the window
/// is) is a property of the *window*, and windows live in the UI layer
/// (`ui::window`), not in the editor — the headless editor has no window at
/// all. So `zz` cannot be a buffer edit the way `dw` is. Instead the editor
/// *recognises* the keystroke (keeping the vi grammar where it belongs, out
/// of the UI) and hands back this description of what the window should do;
/// the UI, which owns the scroll offset, carries it out. This is the same
/// split `EditorResponse::Write` draws for file I/O, for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportScroll {
    /// `zz`: put the cursor line in the vertical centre of the window.
    CenterCursor,
    /// `zt`: put the cursor line at the top of the window.
    CursorToTop,
    /// `zb`: put the cursor line at the bottom of the window.
    CursorToBottom,
    /// `<C-e>`: scroll the view down one line (text moves up); the cursor
    /// follows only if it would otherwise leave the viewport.
    LineDown,
    /// `<C-y>`: scroll the view up one line (text moves down).
    LineUp,
}

/// A window-management command the editor parsed (from an ex command like
/// `:sp`) but cannot itself carry out, because the window tree lives in the
/// UI. Handed back through `EditorResponse::Window` for the UI to perform —
/// see [`ViewportScroll`] for the same pattern and the reasoning behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowCommand {
    /// `:sp`/`:vs [file]` (`scratch == false`) and `:new`/`:vnew`
    /// (`scratch == true`, always with `file == None`): split the active
    /// window. A non-scratch split with no file duplicates the current
    /// buffer's view; with a file, the new window opens that file; a scratch
    /// split opens a fresh empty buffer.
    Split { vertical: bool, file: Option<PathBuf>, scratch: bool },
    /// `:only`: close every window except the active one.
    Only,
    /// `:close`: close the active window (a no-op message on the last one —
    /// vim refuses to `:close` the final window).
    Close,
    /// `:bd`/`:bw`: a buffer was just deleted from the editor, so any window
    /// still pointing at `deleted` must be repointed at `replacement` (the
    /// surviving buffer the editor switched to). The editor owns the buffer
    /// table but not the window tree, so — exactly like [`WindowCommand::Split`]
    /// — it hands the UI the fact and lets the UI update its own tree. Without
    /// this, a split showing the deleted buffer would be left dangling on a
    /// dead id and paint blank.
    BufferDeleted { deleted: BufferId, replacement: BufferId },
}

/// A tab-page command the editor parsed (from `:tabnew`, `gt`, ...) but cannot
/// itself carry out, because tab pages live in the UI (`ui::tab::TabPages`),
/// one layer *above* the window tree.
///
/// # Why this rides the same "editor recognises, UI performs" seam as [`WindowCommand`]
///
/// A tab page in vim/neovim is a whole window layout — an ordered set of window
/// trees with one active. kvim keeps that collection in the UI (the headless
/// [`crate::editor::Editor`] got no window, let alone a tab of windows). So same
/// like `:sp` cannot split from inside the editor, `:tabnew` cannot open a tab
/// from inside the editor: the editor *recognises* the command (keeping the ex/
/// normal-mode grammar where it belongs) and passes back this description for the
/// UI to act on. Same split [`WindowCommand`] and [`ViewportScroll`] draw, one
/// level up. See AID-0020 (windows) and AID-0048 (tabs) for the reasoning.
///
/// The one-editor-cursor invariant still holds hor: a tab switch is a *viewport
/// swap*, not a second cursor — the UI parks the outgoing tab's window tree and
/// swaps the incoming one live, then reloads the single editor cursor from the
/// newly-active window (same sync/load dance AID-0020 uses for `<C-w>` focus).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabCommand {
    /// `:tabnew`/`:tabedit [file]`: open a fresh tab right after the current
    /// one and switch to it. With a file, the new tab opens that file; with
    /// none, it opens a fresh empty (scratch) buffer — vim's `:tabnew` with no
    /// argument lands you on an empty `[No Name]`.
    New { file: Option<PathBuf> },
    /// `:tabclose`: close the current tab (a no-op message on the last one —
    /// vim won't let you close the final tab page).
    Close,
    /// `:tabonly`: close every tab except the current one.
    Only,
    /// `gt`/`:tabnext` (no count) and `:tabnext`/`:tabprevious` with a *relative*
    /// count: move `by` tabs, wrapping round the ends like vim. `forward` picks
    /// next (`gt`) vs previous (`gT`).
    Step { by: usize, forward: bool },
    /// `{count}gt` / `:tabnext {count}`: jump straight to the 1-based tab
    /// `index` (the user sees tabs numbered from one). Vim's absolute form —
    /// different from [`TabCommand::Step`], which is relative.
    Goto { index: usize },
    /// `:tabfirst`/`:tabrewind`: go to the first tab.
    First,
    /// `:tablast`: go to the last tab.
    Last,
    /// `:tabs`: list the open tabs (and the windows in each) on the message
    /// line, so you can see what sits where without squinting at the tabline.
    List,
}

/// A single edit to a buffer: replace the text in `range` with `text`.
///
/// Both insertion (empty `range`) and deletion (empty `text`) are expressed as
/// a replacement, so the undo tree, the LSP `didChange` notifier, and the
/// syntax highlighter all only ever have to understand one operation instead
/// of three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub range: Range,
    pub text: String,
}

impl Edit {
    pub fn insert(at: Position, text: impl Into<String>) -> Self {
        Self { range: Range::point(at), text: text.into() }
    }

    pub fn delete(range: Range) -> Self {
        Self { range, text: String::new() }
    }

    pub fn replace(range: Range, text: impl Into<String>) -> Self {
        Self { range, text: text.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_displays_one_based_for_humans() {
        assert_eq!(Position::new(0, 0).to_string(), "1:1");
        assert_eq!(Position::new(41, 7).to_string(), "42:8");
    }

    #[test]
    fn range_normalizes_a_backwards_selection() {
        let forward = Range::new(Position::new(1, 0), Position::new(3, 5));
        let backward = Range::new(Position::new(3, 5), Position::new(1, 0));
        assert_eq!(forward.normalized(), backward.normalized());
        assert_eq!(forward.normalized(), (Position::new(1, 0), Position::new(3, 5)));
    }

    #[test]
    fn a_point_range_is_empty() {
        assert!(Range::point(Position::new(2, 2)).is_empty());
        assert!(!Range::new(Position::new(2, 2), Position::new(2, 3)).is_empty());
    }

    #[test]
    fn visual_modes_report_themselves_as_visual() {
        assert!(Mode::Visual.is_visual());
        assert!(Mode::VisualLine.is_visual());
        assert!(Mode::VisualBlock.is_visual());
        assert!(!Mode::Normal.is_visual());
        assert!(!Mode::Insert.is_visual());
    }
}
