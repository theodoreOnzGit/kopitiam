//! The window tree: `:sp`/`:vs` splits, laid out as a binary tree of
//! [`Rect`]s.
//!
//! # Why a tree, and why now, when only one window exists
//!
//! vim's window layout is not a flat list — `:vs` inside an already-split
//! window nests, producing an actual tree (a horizontal split containing a
//! vertical split containing two windows, say). Modelling it as a `Vec` of
//! windows with hand-tracked rectangles works for the first split and then
//! actively fights every split after that. Getting the recursive structure
//! right *before* the multi-window UI is built means adding the second and
//! third split later touches nothing but [`WindowTree::split`] call sites,
//! not the layout algorithm.
//!
//! # `SplitKind` naming
//!
//! Named after the vim command it implements, not after the divider's own
//! orientation, because that's the pairing people actually hold in their
//! head (":split, so the windows stack"). `:split` draws a **horizontal**
//! divider line but stacks windows top-to-bottom; `:vsplit` draws a
//! **vertical** divider and places them side by side. See [`SplitKind`]'s
//! doc comment for the full mapping.
//!
//! # Splits reserve a divider cell
//!
//! Every split reserves exactly one cell — a column for `:vs`, a row for
//! `:sp` — between its two children for the visible [`Separator`] line
//! (Neovim's `WinSeparator`). [`WindowTree::layout`] therefore returns the
//! *reduced* window rectangles (the ones text is actually painted into), and
//! [`WindowTree::separators`] returns where the divider glyphs go. Reserving
//! the cell in the layout, rather than overpainting a border on top of the
//! first column of a pane, is what keeps a border from silently eating a
//! column of buffer text.

use ratatui::layout::Rect;

use crate::core::{BufferId, Direction, Position, WindowId};

/// Which way a split divides the screen, named after the vim command.
///
/// * `Horizontal` = `:split`/`:sp` — draws a horizontal divider line,
///   stacking windows **top and bottom**. Maps to
///   `ratatui::layout::Direction::Vertical` (children arranged vertically).
/// * `Vertical` = `:vsplit`/`:vs` — draws a vertical divider line, placing
///   windows **side by side**. Maps to `Direction::Horizontal`.
///
/// The vim-name-vs-divider-orientation mismatch is exactly the kind of thing
/// that silently swaps top/bottom for left/right if re-derived from scratch at
/// each call site, so the axis decision lives only in
/// [`WindowTree::layout_node`] and is never re-reasoned elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitKind {
    Horizontal,
    Vertical,
}

/// A divider line between two panes — the geometry Neovim paints as
/// `WinSeparator`.
///
/// `rect` is exactly one cell thick: a full-height, one-column strip for a
/// [`SplitKind::Vertical`] (side-by-side) split, or a full-width, one-row
/// strip for a [`SplitKind::Horizontal`] (stacked) one. `kind` tells the
/// renderer which glyph to draw (`│` vs `─`); the geometry alone cannot,
/// because a one-cell strip is ambiguous when it is also one cell long.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Separator {
    pub rect: Rect,
    pub kind: SplitKind,
}

/// A single window: a viewport onto one buffer, with its own cursor and
/// scroll position — vim windows are independent views, not independent
/// buffers, so two windows can (and often do) show the same buffer at
/// different scroll positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub id: WindowId,
    pub buffer: BufferId,
    pub cursor: Position,
    pub scroll: crate::ui::textarea::Scroll,
}

impl Window {
    fn new(id: WindowId, buffer: BufferId) -> Self {
        Self { id, buffer, cursor: Position::ORIGIN, scroll: crate::ui::textarea::Scroll::default() }
    }
}

/// A node in the window tree: either a leaf (one visible window) or a split
/// dividing an area between two child nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Leaf(Window),
    Split {
        kind: SplitKind,
        /// The first child's share of the space, 1-99. The second child
        /// gets the remainder. Kept as a percentage (not a `Constraint`
        /// directly) so it round-trips through equality/debug cleanly for
        /// tests; converted to ratatui `Constraint::Percentage` only at
        /// layout time.
        first_percent: u16,
        first: Box<Node>,
        second: Box<Node>,
    },
}

/// The full window tree for one editor "tab" (vim's terminology; kvim does
/// not yet have multiple tab pages, so there is exactly one tree today).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowTree {
    root: Node,
    active: WindowId,
    /// The previously-active window, for `<C-w>p`. `None` until focus has
    /// moved at least once.
    prev: Option<WindowId>,
    next_id: u32,
}

impl WindowTree {
    /// A tree with exactly one window, showing `buffer`.
    pub fn single(buffer: BufferId) -> Self {
        let id = WindowId(0);
        Self { root: Node::Leaf(Window::new(id, buffer)), active: id, prev: None, next_id: 1 }
    }

    /// The currently active window's id — the one that receives keys and
    /// whose statusline is drawn in its "active" colours.
    pub fn active_id(&self) -> WindowId {
        self.active
    }

    /// Read-only access to the active window.
    pub fn active(&self) -> &Window {
        self.find(self.active).expect("active window id always refers to a live window")
    }

    /// Mutable access to the active window — e.g. so the event loop can
    /// update its cursor/scroll after a keypress.
    pub fn active_mut(&mut self) -> &mut Window {
        let id = self.active;
        self.find_mut(id).expect("active window id always refers to a live window")
    }

    /// All windows in the tree, in left-to-right / top-to-bottom traversal
    /// order — the order `:sp` and `Ctrl-W w` cycle through in real vim.
    pub fn windows(&self) -> Vec<&Window> {
        let mut out = Vec::new();
        Self::collect(&self.root, &mut out);
        out
    }

    fn collect<'a>(node: &'a Node, out: &mut Vec<&'a Window>) {
        match node {
            Node::Leaf(w) => out.push(w),
            Node::Split { first, second, .. } => {
                Self::collect(first, out);
                Self::collect(second, out);
            }
        }
    }

    /// Repoints every window showing `old` at `new` — the window-tree half of
    /// a `:bd`/`:bw`. When the editor deletes a buffer it switches to a
    /// surviving one, but each window keeps its own copy of "which buffer am I
    /// showing"; any window still holding the deleted id would render blank
    /// (its `buffer_by_id` lookup now returns `None`). This walks the whole
    /// tree — not just the active window — because a split could have been
    /// showing the deleted buffer too. Windows on other buffers are untouched.
    pub fn remap_buffer(&mut self, old: BufferId, new: BufferId) {
        fn go(node: &mut Node, old: BufferId, new: BufferId) {
            match node {
                Node::Leaf(w) => {
                    if w.buffer == old {
                        w.buffer = new;
                    }
                }
                Node::Split { first, second, .. } => {
                    go(first, old, new);
                    go(second, old, new);
                }
            }
        }
        go(&mut self.root, old, new);
    }

    fn find(&self, id: WindowId) -> Option<&Window> {
        fn go(node: &Node, id: WindowId) -> Option<&Window> {
            match node {
                Node::Leaf(w) if w.id == id => Some(w),
                Node::Leaf(_) => None,
                Node::Split { first, second, .. } => go(first, id).or_else(|| go(second, id)),
            }
        }
        go(&self.root, id)
    }

    fn find_mut(&mut self, id: WindowId) -> Option<&mut Window> {
        fn go(node: &mut Node, id: WindowId) -> Option<&mut Window> {
            match node {
                Node::Leaf(w) if w.id == id => Some(w),
                Node::Leaf(_) => None,
                Node::Split { first, second, .. } => {
                    if let Some(w) = go(first, id) {
                        Some(w)
                    } else {
                        go(second, id)
                    }
                }
            }
        }
        go(&mut self.root, id)
    }

    /// Splits the active window (`:sp` / `:vs`), replacing it with a
    /// `Split` node whose two children both show the same buffer the split
    /// window was showing — matching vim, where a fresh split is a second
    /// view onto the *same* buffer, not an empty one.
    ///
    /// The new window becomes active (vim's default: the new split is where
    /// the cursor lands) and is the **first** child, so it appears above
    /// (`Horizontal`) or to the left (`Vertical`) of the original — again
    /// matching vim's default split placement (`nosplitbelow`/`nosplitright`
    /// semantics; a `splitbelow`/`splitright` toggle is a config concern for
    /// a later pass, not this tree's job).
    ///
    /// Returns the new window's id.
    pub fn split(&mut self, kind: SplitKind) -> WindowId {
        let new_id = WindowId(self.next_id);
        self.next_id += 1;

        let active_id = self.active;
        Self::split_node(&mut self.root, active_id, kind, new_id);
        self.prev = Some(active_id);
        self.active = new_id;
        new_id
    }

    fn split_node(node: &mut Node, target: WindowId, kind: SplitKind, new_id: WindowId) -> bool {
        match node {
            Node::Leaf(w) if w.id == target => {
                // Both children start as second views onto the same buffer at
                // the *same* cursor and scroll the split window had — matching
                // vim, where `:sp` gives you two views of the same place, not
                // one reset to the top. (The split window's live cursor lives
                // in the editor; `App` syncs it into this `Window` before
                // calling `split`, so `*w` here is current.)
                let src = *w;
                let new_window = Window { id: new_id, ..src };
                *node = Node::Split {
                    kind,
                    first_percent: 50,
                    first: Box::new(Node::Leaf(new_window)),
                    second: Box::new(Node::Leaf(src)),
                };
                true
            }
            Node::Leaf(_) => false,
            Node::Split { first, second, .. } => {
                Self::split_node(first, target, kind, new_id)
                    || Self::split_node(second, target, kind, new_id)
            }
        }
    }

    /// Closes the active window (`:q`/`:close` when more than one window is
    /// open). The sibling (and everything below it) takes over the closed
    /// window's share of the screen; the sibling's leftmost/topmost leaf
    /// becomes active, matching vim's focus-follows-close behaviour.
    ///
    /// Returns `false` without changing anything if this is the last
    /// window — closing the last window is `:q`'s "quit the editor" case,
    /// not a tree operation, so the caller (the event loop, informed by the
    /// editor's `EditorResponse`) decides what "no more windows" means.
    pub fn close_active(&mut self) -> bool {
        if matches!(self.root, Node::Leaf(_)) {
            return false; // Last window: nothing to collapse into.
        }
        let active_id = self.active;
        // `close_node` consumes its input by value, so the current root has
        // to be moved out of `self` first. The placeholder written back
        // momentarily is never observed: it's overwritten by the real
        // result on the very next line, before any other method can run.
        let placeholder = Node::Leaf(Window::new(WindowId(u32::MAX), BufferId(0)));
        let root = std::mem::replace(&mut self.root, placeholder);
        self.root = Self::close_node(root, active_id)
            .expect("root always contains the active window, so closing it always yields a replacement");
        // The closed window's id is gone, so a `prev` pointing at it would
        // dangle; simplest correct thing is to forget it.
        self.prev = None;
        self.active = Self::first_leaf_id(&self.root);
        true
    }

    /// Returns `None` when `node` itself was the leaf to remove (the caller
    /// replaces `node`'s slot with whichever sibling survives); otherwise
    /// returns `Some(node)` with the target removed from within it.
    ///
    /// Only ever called with a `target` that is actually present somewhere
    /// in `node` (checked via [`Self::contains`] before recursing into the
    /// branch that holds it), so "target not found in either branch" is not
    /// a case this needs to represent.
    fn close_node(node: Node, target: WindowId) -> Option<Node> {
        match node {
            Node::Leaf(w) if w.id == target => None,
            Node::Leaf(_) => Some(node),
            Node::Split { kind, first_percent, first, second } => {
                if Self::contains(&first, target) {
                    match Self::close_node(*first, target) {
                        None => Some(*second), // first collapsed entirely: second survives whole.
                        Some(new_first) => Some(Node::Split {
                            kind,
                            first_percent,
                            first: Box::new(new_first),
                            second,
                        }),
                    }
                } else {
                    debug_assert!(Self::contains(&second, target), "target must live in one of the two branches");
                    match Self::close_node(*second, target) {
                        None => Some(*first), // second collapsed entirely: first survives whole.
                        Some(new_second) => Some(Node::Split {
                            kind,
                            first_percent,
                            first,
                            second: Box::new(new_second),
                        }),
                    }
                }
            }
        }
    }

    fn contains(node: &Node, target: WindowId) -> bool {
        match node {
            Node::Leaf(w) => w.id == target,
            Node::Split { first, second, .. } => Self::contains(first, target) || Self::contains(second, target),
        }
    }

    fn first_leaf_id(node: &Node) -> WindowId {
        match node {
            Node::Leaf(w) => w.id,
            Node::Split { first, .. } => Self::first_leaf_id(first),
        }
    }

    /// Computes each window's on-screen [`Rect`] within `area`, in the same
    /// traversal order as [`WindowTree::windows`].
    ///
    /// The rectangles are the ones text is painted into: each split has
    /// already reserved one cell for its [`Separator`], so summing the window
    /// widths of a single vertical split gives `area.width - 1`, not
    /// `area.width`. Ask [`WindowTree::separators`] for the reserved cells.
    pub fn layout(&self, area: Rect) -> Vec<(WindowId, Rect)> {
        let (windows, _) = self.layout_inner(area);
        windows
    }

    /// The divider cells between panes, in the same recursive order the tree
    /// is walked — one per split. Empty for a single-window tree.
    pub fn separators(&self, area: Rect) -> Vec<Separator> {
        let (_, seps) = self.layout_inner(area);
        seps
    }

    fn layout_inner(&self, area: Rect) -> (Vec<(WindowId, Rect)>, Vec<Separator>) {
        let mut windows = Vec::new();
        let mut seps = Vec::new();
        Self::layout_node(&self.root, area, &mut windows, &mut seps);
        (windows, seps)
    }

    /// Lays a subtree out within `area`, reserving one cell per split for the
    /// divider so window rectangles and separator rectangles never overlap.
    ///
    /// The split axis is decided here and nowhere else: a `Vertical` split
    /// (`:vs`) divides `area` left/right with a one-column divider between; a
    /// `Horizontal` split (`:sp`) divides it top/bottom with a one-row divider.
    fn layout_node(
        node: &Node,
        area: Rect,
        windows: &mut Vec<(WindowId, Rect)>,
        seps: &mut Vec<Separator>,
    ) {
        match node {
            Node::Leaf(w) => windows.push((w.id, area)),
            Node::Split { kind, first_percent, first, second } => {
                let (first_area, sep, second_area) = split_area(*kind, area, *first_percent);
                seps.push(Separator { rect: sep, kind: *kind });
                Self::layout_node(first, first_area, windows, seps);
                Self::layout_node(second, second_area, windows, seps);
            }
        }
    }

    // ---------------------------------------------------------------
    // Window navigation and management (`<C-w>` commands)
    // ---------------------------------------------------------------

    /// How many windows are open. `<C-w>q`/`:q` uses it to decide between
    /// closing a split and quitting the editor.
    pub fn window_count(&self) -> usize {
        self.windows().len()
    }

    /// The active window (mutable) — used by `App` to write the editor's live
    /// cursor/buffer back into the tree before a focus change or split.
    pub fn set_active(&mut self, id: WindowId) {
        if self.find(id).is_some() && id != self.active {
            self.prev = Some(self.active);
            self.active = id;
        }
    }

    /// `<C-w>w` / `<C-w>W`: cycle focus to the next / previous window in
    /// traversal order, wrapping around.
    pub fn cycle(&mut self, forward: bool) {
        let ids: Vec<WindowId> = self.windows().iter().map(|w| w.id).collect();
        if ids.len() < 2 {
            return;
        }
        let Some(pos) = ids.iter().position(|&i| i == self.active) else { return };
        let next = if forward {
            (pos + 1) % ids.len()
        } else {
            (pos + ids.len() - 1) % ids.len()
        };
        self.set_active(ids[next]);
    }

    /// `<C-w>p`: focus the previously-active window, if any.
    pub fn focus_prev(&mut self) {
        if let Some(prev) = self.prev {
            self.set_active(prev);
        }
    }

    /// `<C-w>h/j/k/l`: focus the window spatially adjacent to the active one
    /// in `dir`, laid out within `area`. Returns the newly-focused id, or
    /// `None` if there is no window that way.
    ///
    /// # How "the window to the left" is decided
    ///
    /// Splits tile `area` exactly, so a neighbour in direction `dir` is any
    /// window that lies wholly on that side of the active window's edge. Among
    /// those, the nearest wins (the edge closest to the active window), and
    /// ties are broken by how well the candidate's *perpendicular* span
    /// overlaps the active window's centre — so `<C-w>j` from a tall left
    /// pane lands in the bottom pane it actually sits above, not some distant
    /// window that merely starts lower. A pure cycle would be simpler but
    /// wrong: it would send `<C-w>l` to whatever comes next in traversal
    /// order, which is frequently *not* the window on the right.
    pub fn focus_direction(&mut self, area: Rect, dir: Direction) -> Option<WindowId> {
        let id = self.window_in_direction(area, dir)?;
        self.set_active(id);
        Some(id)
    }

    fn window_in_direction(&self, area: Rect, dir: Direction) -> Option<WindowId> {
        let layout = self.layout(area);
        let active = layout.iter().find(|(id, _)| *id == self.active).map(|(_, r)| *r)?;
        let ac = (active.x + active.width / 2, active.y + active.height / 2);

        let mut best: Option<(WindowId, u16, u16)> = None; // (id, primary dist, perpendicular dist)
        for (id, r) in &layout {
            if *id == self.active {
                continue;
            }
            let (in_dir, primary) = match dir {
                Direction::Left => (r.x + r.width <= active.x, active.x.saturating_sub(r.x + r.width)),
                Direction::Right => (r.x >= active.x + active.width, r.x.saturating_sub(active.x + active.width)),
                Direction::Up => (r.y + r.height <= active.y, active.y.saturating_sub(r.y + r.height)),
                Direction::Down => (r.y >= active.y + active.height, r.y.saturating_sub(active.y + active.height)),
            };
            if !in_dir {
                continue;
            }
            let perp = match dir {
                Direction::Left | Direction::Right => (r.y + r.height / 2).abs_diff(ac.1),
                Direction::Up | Direction::Down => (r.x + r.width / 2).abs_diff(ac.0),
            };
            let better = match best {
                None => true,
                Some((_, bp, bperp)) => (primary, perp) < (bp, bperp),
            };
            if better {
                best = Some((*id, primary, perp));
            }
        }
        best.map(|(id, _, _)| id)
    }

    /// `<C-w>o` / `:only`: close every window but the active one.
    pub fn only(&mut self) {
        let active = *self.active();
        self.root = Node::Leaf(active);
        self.prev = None;
    }

    /// `<C-w>=`: reset every split to an even 50/50 division.
    pub fn equalize(&mut self) {
        Self::equalize_node(&mut self.root);
    }

    fn equalize_node(node: &mut Node) {
        if let Node::Split { first_percent, first, second, .. } = node {
            *first_percent = 50;
            Self::equalize_node(first);
            Self::equalize_node(second);
        }
    }

    /// `<C-w>+`/`-` (`vertical == false`, height) and `<C-w>>`/`<` (`vertical
    /// == true`, width): grow or shrink the active window by nudging the
    /// nearest enclosing split of the matching orientation.
    pub fn resize_active(&mut self, vertical: bool, grow: bool) {
        let kind = if vertical { SplitKind::Vertical } else { SplitKind::Horizontal };
        let active = self.active;
        Self::resize_node(&mut self.root, active, kind, grow);
    }

    /// Returns `true` when `target` was found in `node` but has **not** yet
    /// been resized — i.e. an enclosing split of the matching orientation is
    /// still being looked for further up. Returns `false` once handled (or if
    /// `target` is not in this subtree), so the walk adjusts the split nearest
    /// the active leaf and no other.
    fn resize_node(node: &mut Node, target: WindowId, kind: SplitKind, grow: bool) -> bool {
        let Node::Split { kind: k, first_percent, first, second } = node else {
            return matches!(node, Node::Leaf(w) if w.id == target);
        };
        let in_first = Self::contains(first, target);
        let child = if in_first { first.as_mut() } else { second.as_mut() };
        if !Self::resize_node(child, target, kind, grow) {
            return false; // not found below, or already handled deeper
        }
        if *k == kind {
            const STEP: i32 = 8;
            // Growing the active window enlarges whichever side holds it: the
            // first child's percentage rises when active is in `first`.
            let delta = if in_first == grow { STEP } else { -STEP };
            *first_percent = (*first_percent as i32 + delta).clamp(10, 90) as u16;
            return false; // handled here; don't let an outer split move too
        }
        true // found but this split is the wrong orientation; keep looking up
    }

    /// `<C-w>x`: exchange the active window's contents with the next window's,
    /// and follow the swap (the other window becomes active), matching vim.
    pub fn exchange(&mut self) {
        let ids: Vec<WindowId> = self.windows().iter().map(|w| w.id).collect();
        if ids.len() < 2 {
            return;
        }
        let Some(pos) = ids.iter().position(|&i| i == self.active) else { return };
        let other = ids[(pos + 1) % ids.len()];
        let a = *self.find(self.active).expect("active is live");
        let b = *self.find(other).expect("other is live");
        Self::copy_contents(self.find_mut(self.active).expect("active is live"), &b);
        Self::copy_contents(self.find_mut(other).expect("other is live"), &a);
        self.set_active(other);
    }

    /// `<C-w>r`: rotate every window's contents one place forward (the last
    /// window's contents wrap to the first), leaving the layout itself fixed.
    pub fn rotate(&mut self) {
        let ids: Vec<WindowId> = self.windows().iter().map(|w| w.id).collect();
        if ids.len() < 2 {
            return;
        }
        let contents: Vec<Window> = self.windows().iter().map(|&&w| w).collect();
        let n = contents.len();
        for (i, &id) in ids.iter().enumerate() {
            let src = contents[(i + n - 1) % n];
            Self::copy_contents(self.find_mut(id).expect("id from live traversal"), &src);
        }
    }

    /// Copies the buffer/cursor/scroll of `src` into `dst`, leaving `dst`'s id
    /// (its identity and layout slot) alone.
    fn copy_contents(dst: &mut Window, src: &Window) {
        dst.buffer = src.buffer;
        dst.cursor = src.cursor;
        dst.scroll = src.scroll;
    }
}

/// Divides `area` into `(first, separator, second)` for a split of the given
/// `kind`, reserving exactly one cell for the separator between the children.
///
/// `first_percent` is the first child's share of the space *left after the
/// divider is taken out*, so a 50/50 split of an 80-column area gives the
/// children 39 and 40 columns with the 40th column (index-wise, column 39) as
/// the divider — the pane widths need not be equal once an odd cell is spent
/// on the border, and that is fine.
fn split_area(kind: SplitKind, area: Rect, first_percent: u16) -> (Rect, Rect, Rect) {
    match kind {
        // `:vs` — side by side, a one-column divider between.
        SplitKind::Vertical => {
            let avail = area.width.saturating_sub(1);
            let fw = (avail as u32 * first_percent as u32 / 100) as u16;
            let sw = avail - fw;
            let first = Rect { x: area.x, y: area.y, width: fw, height: area.height };
            let sep = Rect { x: area.x + fw, y: area.y, width: 1, height: area.height };
            let second = Rect { x: area.x + fw + 1, y: area.y, width: sw, height: area.height };
            (first, sep, second)
        }
        // `:sp` — stacked, a one-row divider between.
        SplitKind::Horizontal => {
            let avail = area.height.saturating_sub(1);
            let fh = (avail as u32 * first_percent as u32 / 100) as u16;
            let sh = avail - fh;
            let first = Rect { x: area.x, y: area.y, width: area.width, height: fh };
            let sep = Rect { x: area.x, y: area.y + fh, width: area.width, height: 1 };
            let second = Rect { x: area.x, y: area.y + fh + 1, width: area.width, height: sh };
            (first, sep, second)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_tree_has_exactly_one_window() {
        let tree = WindowTree::single(BufferId(1));
        assert_eq!(tree.windows().len(), 1);
        assert_eq!(tree.active().buffer, BufferId(1));
    }

    #[test]
    fn split_produces_two_windows_on_the_same_buffer() {
        let mut tree = WindowTree::single(BufferId(7));
        let new_id = tree.split(SplitKind::Horizontal);
        assert_eq!(tree.windows().len(), 2);
        assert_eq!(tree.active_id(), new_id);
        for w in tree.windows() {
            assert_eq!(w.buffer, BufferId(7));
        }
    }

    #[test]
    fn nested_splits_produce_three_windows() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Horizontal);
        tree.split(SplitKind::Vertical);
        assert_eq!(tree.windows().len(), 3);
    }

    #[test]
    fn horizontal_split_stacks_windows_top_and_bottom() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Horizontal);
        let area = Rect { x: 0, y: 0, width: 80, height: 40 };
        let rects = tree.layout(area);
        assert_eq!(rects.len(), 2);
        // Stacked: same x/width, different y.
        assert_eq!(rects[0].1.x, rects[1].1.x);
        assert_eq!(rects[0].1.width, rects[1].1.width);
        assert_ne!(rects[0].1.y, rects[1].1.y);
    }

    #[test]
    fn vertical_split_places_windows_side_by_side() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Vertical);
        let area = Rect { x: 0, y: 0, width: 80, height: 40 };
        let rects = tree.layout(area);
        assert_eq!(rects.len(), 2);
        // Side by side: same y/height, different x.
        assert_eq!(rects[0].1.y, rects[1].1.y);
        assert_eq!(rects[0].1.height, rects[1].1.height);
        assert_ne!(rects[0].1.x, rects[1].1.x);
    }

    #[test]
    fn layout_and_separator_together_cover_the_full_area_for_a_vertical_split() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Vertical);
        let area = Rect { x: 0, y: 0, width: 80, height: 40 };
        let rects = tree.layout(area);
        let seps = tree.separators(area);
        // The panes plus the one reserved divider column tile the area exactly.
        let total_width: u16 = rects.iter().map(|(_, r)| r.width).sum();
        let sep_width: u16 = seps.iter().map(|s| s.rect.width).sum();
        assert_eq!(total_width + sep_width, area.width, "panes + divider must fill the area");
        assert_eq!(seps.len(), 1);
        assert_eq!(seps[0].kind, SplitKind::Vertical);
    }

    #[test]
    fn a_vertical_split_reserves_a_full_height_divider_column_between_the_panes() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Vertical);
        let area = Rect { x: 0, y: 0, width: 80, height: 40 };
        let rects = tree.layout(area);
        let sep = tree.separators(area)[0];
        // The divider sits immediately right of the left pane, is one column
        // wide, spans the full height, and the right pane starts just past it.
        assert_eq!(sep.rect.width, 1);
        assert_eq!(sep.rect.height, area.height);
        assert_eq!(sep.rect.x, rects[0].1.x + rects[0].1.width, "divider abuts the left pane");
        assert_eq!(rects[1].1.x, sep.rect.x + 1, "right pane starts just past the divider");
    }

    #[test]
    fn a_horizontal_split_reserves_a_full_width_divider_row_between_the_panes() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Horizontal);
        let area = Rect { x: 0, y: 0, width: 80, height: 40 };
        let rects = tree.layout(area);
        let sep = tree.separators(area)[0];
        assert_eq!(sep.rect.height, 1);
        assert_eq!(sep.rect.width, area.width);
        assert_eq!(sep.rect.y, rects[0].1.y + rects[0].1.height, "divider abuts the top pane");
        assert_eq!(rects[1].1.y, sep.rect.y + 1, "bottom pane starts just past the divider");
    }

    #[test]
    fn a_single_window_has_no_separators() {
        let tree = WindowTree::single(BufferId(1));
        let area = Rect { x: 0, y: 0, width: 80, height: 40 };
        assert!(tree.separators(area).is_empty());
        assert_eq!(tree.layout(area)[0].1, area, "the sole window keeps the whole area");
    }

    #[test]
    fn closing_the_only_window_reports_false() {
        let mut tree = WindowTree::single(BufferId(1));
        assert!(!tree.close_active());
        assert_eq!(tree.windows().len(), 1);
    }

    #[test]
    fn closing_a_split_window_returns_to_one_window() {
        let mut tree = WindowTree::single(BufferId(1));
        tree.split(SplitKind::Horizontal);
        assert_eq!(tree.windows().len(), 2);
        assert!(tree.close_active());
        assert_eq!(tree.windows().len(), 1);
    }
}
