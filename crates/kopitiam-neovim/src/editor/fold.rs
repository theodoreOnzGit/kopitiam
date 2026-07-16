//! Manual folds (`foldmethod=manual`): the state and view math behind kvim's
//! `z`-prefixed fold family (`zf`, `zo`, `zc`, `za`, `zR`, `zM`, `zd`, ...).
//!
//! # Two types, two jobs
//!
//! * [`FoldSet`] is the **authoring model**: the mutable set of folds a buffer
//!   has, each an inclusive `[start, end]` line range that is either open or
//!   closed. Every `z`-command that *changes* folds (`zf` create, `zo`/`zc`
//!   open/close, `zd` delete, `zR`/`zM` open/close-all, `zi` toggle
//!   `foldenable`) is a method on this type. It knows nothing about screens or
//!   scroll offsets.
//!
//! * [`FoldRows`] is the **view**: the flattened, non-overlapping set of
//!   *effectively-closed* line ranges — the "visible lines" abstraction the
//!   renderer and the cursor-motion logic both need. A closed fold collapses
//!   its `[start+1, end]` lines to nothing and shows only its `start` line as a
//!   single fold header row; a fold nested inside an already-closed fold does
//!   not matter to the view (its parent already hid it), so `FoldRows` keeps
//!   only the *outermost* closed ranges. [`FoldSet::collapsed`] produces it.
//!
//! Keeping these apart is what lets the renderer (which owns no `FoldSet` — it
//! is handed a cheap [`FoldRows`] snapshot each frame across the
//! [`crate::ui::event::EditorHost`] seam) and the editor's motion code (which
//! owns the real `FoldSet`) agree on exactly which buffer lines are visible,
//! without either reaching into the other.
//!
//! # Why fold state is buffer-scoped, not window-scoped
//!
//! In real vim a fold is a property of a *window*: the same buffer shown in two
//! splits can have different folds open in each. kvim deliberately scopes folds
//! to the **buffer** instead, because kvim has a single logical cursor and the
//! overwhelming-common case is one view per buffer; a window-local fold table
//! would multiply the state (and the edit-tracking problem below) for a
//! distinction almost no one uses. This is a conscious deviation from vim,
//! recorded in the fold ADR. If per-window folds are ever needed, this type is
//! the thing that moves from `HashMap<BufferId, _>` on the editor to
//! `HashMap<(WindowId, BufferId), _>`; nothing about the fold *math* changes.
//!
//! # A fold does not follow edits (yet)
//!
//! Folds hold absolute line numbers and are **not** shifted when text is
//! inserted or deleted above them. After a structural edit a fold can therefore
//! cover the wrong lines. This is a known limitation of the first manual-fold
//! cut (tracked as a follow-up bead); vim keeps folds pinned to lines through
//! every edit, which requires the same shift-on-edit machinery the mark table
//! already has. Until that lands, `zE` (eliminate all folds) is the escape
//! hatch.

/// One manual fold: an inclusive range of buffer lines that can be shown
/// (`closed == false`) or collapsed to a single header row (`closed == true`).
///
/// `level` is the nesting depth (0 = outermost), recomputed by
/// [`FoldSet`] whenever the set changes rather than trusted from callers —
/// see [`FoldSet::recompute_levels`]. It exists for a faithful default
/// `foldtext` (vim draws one `-` per level) and for the "open/close *one*
/// level" semantics of `zo`/`zc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    /// First line of the fold (0-based, inclusive). This is the row that stays
    /// visible as the fold header when the fold is closed.
    pub start: usize,
    /// Last line of the fold (0-based, inclusive).
    pub end: usize,
    /// Whether the fold is currently collapsed.
    pub closed: bool,
    /// Nesting depth, 0 for an outermost fold. Computed by the owning
    /// [`FoldSet`], not set by callers.
    pub level: usize,
}

impl Fold {
    /// Does this fold's range cover `line`?
    pub fn contains_line(&self, line: usize) -> bool {
        self.start <= line && line <= self.end
    }

    /// Does this fold strictly contain `other` (cover it and be larger)?
    fn contains_fold(&self, other: &Fold) -> bool {
        self.start <= other.start && other.end <= self.end && (self.start != other.start || self.end != other.end)
    }

    /// Number of buffer lines this fold spans.
    pub fn line_count(&self) -> usize {
        self.end - self.start + 1
    }
}

/// The set of manual folds on one buffer, plus the `foldenable` flag.
///
/// Folds nest but never partially overlap — the only ways to create one
/// (`zf{motion}`, visual `zf`, `:{range}fold`) all take a clean line range, and
/// two ranges are always either disjoint or one-contains-the-other. Every
/// query here relies on that invariant.
#[derive(Debug, Clone)]
pub struct FoldSet {
    folds: Vec<Fold>,
    /// `foldenable` (`zi` toggles, `zn` clears, `zN` sets). When `false` every
    /// fold renders open regardless of its `closed` flag, but the flags are
    /// preserved so `zN`/`zi` restores the previous collapse state exactly —
    /// this is why disabling is not the same as `zR` (open all), which
    /// *clears* the flags.
    enabled: bool,
}

impl Default for FoldSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FoldSet {
    /// An empty fold set with `foldenable` on (vim's default).
    pub fn new() -> Self {
        Self { folds: Vec::new(), enabled: true }
    }

    /// Whether `foldenable` is on. When off, [`FoldSet::collapsed`] is empty
    /// (nothing is visually collapsed) even though closed flags survive.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// `true` when there are no folds at all.
    pub fn is_empty(&self) -> bool {
        self.folds.is_empty()
    }

    /// All folds, in no guaranteed order. For tests and `zj`/`zk` navigation.
    pub fn folds(&self) -> &[Fold] {
        &self.folds
    }

    // ---- creation / deletion -------------------------------------------------

    /// Creates a closed fold over lines `[start, end]` (`zf`, `:{range}fold`).
    ///
    /// vim closes a freshly-created manual fold immediately, so this does too.
    /// A degenerate single-line range (`start == end`) is rejected — vim will
    /// not fold one line — and `start`/`end` are ordered defensively. Returns
    /// the created fold's `start` (where the cursor should land) or `None` if
    /// nothing was created.
    pub fn create(&mut self, start: usize, end: usize) -> Option<usize> {
        let (start, end) = if start <= end { (start, end) } else { (end, start) };
        if start == end {
            return None;
        }
        self.folds.push(Fold { start, end, closed: true, level: 0 });
        self.recompute_levels();
        Some(start)
    }

    /// Deletes the innermost fold containing `line` (`zd`). Returns `true` if a
    /// fold was removed.
    pub fn delete_at(&mut self, line: usize) -> bool {
        let Some(idx) = self.innermost_index_at(line) else { return false };
        self.folds.remove(idx);
        self.recompute_levels();
        true
    }

    /// Removes every fold in the buffer (`zE`).
    pub fn delete_all(&mut self) {
        self.folds.clear();
    }

    // ---- open / close --------------------------------------------------------

    /// `zo`: open one level under the cursor — the *outermost* closed fold
    /// containing `line`, i.e. the one actually hiding it. Returns `true` if a
    /// fold was opened.
    pub fn open_one(&mut self, line: usize) -> bool {
        // Outermost = smallest level. Among closed folds covering `line`, the
        // one with the least level is the visible header the user sees.
        let target = self
            .folds
            .iter()
            .enumerate()
            .filter(|(_, f)| f.closed && f.contains_line(line))
            .min_by_key(|(_, f)| f.level)
            .map(|(i, _)| i);
        if let Some(i) = target {
            self.folds[i].closed = false;
            true
        } else {
            false
        }
    }

    /// `zc`: close one level under the cursor — the *innermost* open fold
    /// containing `line`. Returns the closed fold's `start` (cursor lands
    /// there) or `None` if there was no open fold to close.
    pub fn close_one(&mut self, line: usize) -> Option<usize> {
        let target = self
            .folds
            .iter()
            .enumerate()
            .filter(|(_, f)| !f.closed && f.contains_line(line))
            .max_by_key(|(_, f)| f.level)
            .map(|(i, _)| i);
        let i = target?;
        self.folds[i].closed = true;
        Some(self.folds[i].start)
    }

    /// `za`: toggle. If `line` sits under any closed fold, open one level;
    /// otherwise close one level. Returns the cursor's new line if a fold
    /// closed (so the caller can move onto the header), else `None`.
    pub fn toggle_one(&mut self, line: usize) -> Option<usize> {
        if self.folds.iter().any(|f| f.closed && f.contains_line(line)) {
            self.open_one(line);
            None
        } else {
            self.close_one(line)
        }
    }

    /// `zv`: view cursor — open just the folds that *contain* `line`, so the
    /// cursor line becomes visible. Folds nested inside them (but not covering
    /// `line`) are left as they are — this is the narrow "reveal the cursor"
    /// operation, distinct from `zO`'s whole-subtree open.
    pub fn view_cursor(&mut self, line: usize) {
        for f in &mut self.folds {
            if f.contains_line(line) {
                f.closed = false;
            }
        }
    }

    /// `zO`: open the fold at the cursor *recursively* — the outermost fold
    /// containing `line` and every fold nested within it, whether or not those
    /// nested folds cover `line` themselves. This is why `zO` on the first line
    /// of an outer fold also opens inner folds further down.
    pub fn open_recursive(&mut self, line: usize) {
        let Some((s, e)) = self.outer_containing(line) else { return };
        for f in &mut self.folds {
            if s <= f.start && f.end <= e {
                f.closed = false;
            }
        }
    }

    /// `zC`: close the fold at the cursor recursively — the outermost fold
    /// containing `line` and every fold nested within it. Returns that
    /// outermost fold's `start` (where the cursor lands), or `None`.
    pub fn close_recursive(&mut self, line: usize) -> Option<usize> {
        let (s, e) = self.outer_containing(line)?;
        for f in &mut self.folds {
            if s <= f.start && f.end <= e {
                f.closed = true;
            }
        }
        Some(s)
    }

    /// `zA`: recursive toggle. Opens the cursor's fold subtree if any fold at
    /// `line` is closed, else closes it. Returns the new cursor line if it
    /// closed, else `None`.
    pub fn toggle_recursive(&mut self, line: usize) -> Option<usize> {
        if self.folds.iter().any(|f| f.closed && f.contains_line(line)) {
            self.open_recursive(line);
            None
        } else {
            self.close_recursive(line)
        }
    }

    /// The `(start, end)` of the outermost fold covering `line` — the root of
    /// the fold subtree the cursor sits in — or `None` if no fold covers it.
    fn outer_containing(&self, line: usize) -> Option<(usize, usize)> {
        self.folds
            .iter()
            .filter(|f| f.contains_line(line))
            .min_by_key(|f| f.level)
            .map(|f| (f.start, f.end))
    }

    /// `zR`: open all folds (clears every `closed` flag).
    pub fn open_all(&mut self) {
        for f in &mut self.folds {
            f.closed = false;
        }
    }

    /// `zM`: close all folds (sets every `closed` flag).
    pub fn close_all(&mut self) {
        for f in &mut self.folds {
            f.closed = true;
        }
    }

    // ---- foldenable ----------------------------------------------------------

    /// `zn`: disable folding (`foldenable` off) — folds keep their closed
    /// flags but nothing collapses visually.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// `zN`: re-enable folding, restoring whatever was closed before `zn`.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// `zi`: toggle `foldenable`. Returns the new state.
    pub fn toggle_enabled(&mut self) -> bool {
        self.enabled = !self.enabled;
        self.enabled
    }

    // ---- navigation ----------------------------------------------------------

    /// `zj`: the start line of the next fold that begins after `line`, or
    /// `None` if there is none below.
    pub fn next_fold_start(&self, line: usize) -> Option<usize> {
        self.folds.iter().map(|f| f.start).filter(|&s| s > line).min()
    }

    /// `zk`: the end line of the previous fold that ends before `line`, or
    /// `None` if there is none above.
    pub fn prev_fold_end(&self, line: usize) -> Option<usize> {
        self.folds.iter().map(|f| f.end).filter(|&e| e < line).max()
    }

    /// `[z`: the start of the innermost fold containing `line`, or `None`.
    pub fn current_fold_start(&self, line: usize) -> Option<usize> {
        self.innermost_index_at(line).map(|i| self.folds[i].start)
    }

    /// `]z`: the end of the innermost fold containing `line`, or `None`.
    pub fn current_fold_end(&self, line: usize) -> Option<usize> {
        self.innermost_index_at(line).map(|i| self.folds[i].end)
    }

    // ---- the view ------------------------------------------------------------

    /// Flattens this set into the non-overlapping [`FoldRows`] the renderer and
    /// motion code consume. Empty when `foldenable` is off.
    ///
    /// Only *outermost* closed folds survive: a closed fold nested inside
    /// another closed fold is already hidden by its parent, so it contributes
    /// nothing to which lines are visible. Duplicate identical ranges (a fold
    /// created twice over the same lines) collapse to one row.
    pub fn collapsed(&self) -> FoldRows {
        if !self.enabled {
            return FoldRows::none();
        }
        let mut ranges: Vec<(usize, usize)> = self
            .folds
            .iter()
            .filter(|f| f.closed)
            .filter(|f| {
                // Keep only folds not strictly inside another *closed* fold.
                !self.folds.iter().any(|o| o.closed && o.contains_fold(f))
            })
            .map(|f| (f.start, f.end))
            .collect();
        ranges.sort_unstable();
        ranges.dedup();
        FoldRows { ranges }
    }

    // ---- internals -----------------------------------------------------------

    /// Index of the innermost (deepest-level, smallest) fold covering `line`.
    fn innermost_index_at(&self, line: usize) -> Option<usize> {
        self.folds
            .iter()
            .enumerate()
            .filter(|(_, f)| f.contains_line(line))
            .max_by_key(|(_, f)| (f.level, std::cmp::Reverse(f.line_count())))
            .map(|(i, _)| i)
    }

    /// Recomputes every fold's `level` from containment. A fold's level is the
    /// number of other folds that strictly contain it. O(n²), but n (folds in
    /// one buffer) is tiny and this only runs on create/delete.
    fn recompute_levels(&mut self) {
        let snapshot = self.folds.clone();
        for f in &mut self.folds {
            f.level = snapshot.iter().filter(|o| o.contains_fold(f)).count();
        }
    }
}

/// The flattened, non-overlapping set of *effectively-closed* line ranges — the
/// "visible lines" view shared by the renderer and the editor's vertical-motion
/// logic. Produced by [`FoldSet::collapsed`].
///
/// Every method here treats a closed fold `[start, end]` as a single visual row
/// anchored at `start`: `start` is visible (it becomes the fold header),
/// `start+1..=end` are hidden. Ranges are non-overlapping and sorted by start.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoldRows {
    ranges: Vec<(usize, usize)>,
}

impl FoldRows {
    /// No folds — every buffer line is its own visible row.
    pub fn none() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Build directly from `(start, end)` inclusive ranges (used by the render
    /// seam, which receives them across [`crate::ui::event::EditorHost`]).
    /// Ranges are assumed already non-overlapping; they are sorted here.
    pub fn from_ranges(mut ranges: Vec<(usize, usize)>) -> Self {
        ranges.sort_unstable();
        Self { ranges }
    }

    /// The raw closed ranges, sorted by start.
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// `true` when there is nothing collapsed — the fast path the renderer and
    /// motion code take to skip all fold math entirely.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The closed fold `(start, end)` that begins exactly at `line`, if `line`
    /// is a fold header. This is what the renderer checks at each row to decide
    /// whether to draw a fold header (and skip to `end + 1`).
    pub fn fold_at(&self, line: usize) -> Option<(usize, usize)> {
        self.ranges.iter().copied().find(|&(s, _)| s == line)
    }

    /// Is `line` hidden inside a closed fold (i.e. not a header)?
    pub fn is_hidden(&self, line: usize) -> bool {
        self.ranges.iter().any(|&(s, e)| s < line && line <= e)
    }

    /// The visible row `line` belongs to: if `line` is hidden inside a closed
    /// fold, the fold's header line; otherwise `line` itself. This is where a
    /// cursor that lands inside a closed fold gets snapped to — vim never lets
    /// the cursor sit on a hidden line.
    pub fn header_of(&self, line: usize) -> usize {
        self.ranges.iter().find(|&&(s, e)| s < line && line <= e).map(|&(s, _)| s).unwrap_or(line)
    }

    /// The last buffer line of the visual row that `line` heads: the fold's
    /// `end` if a closed fold starts at `line`, else `line` itself.
    fn row_end(&self, line: usize) -> usize {
        self.fold_at(line).map(|(_, e)| e).unwrap_or(line)
    }

    /// The start line of the next visible row below `line`'s row, clamped so it
    /// never exceeds `last_line`. Drives `j`: from a fold header it lands past
    /// the whole fold, treating the closed fold as one line.
    pub fn next_visible(&self, line: usize, last_line: usize) -> usize {
        let candidate = self.row_end(line) + 1;
        if candidate > last_line {
            // Already on the last visible row; stay put (`j` at EOF is a no-op).
            self.header_of(line.min(last_line))
        } else {
            self.header_of(candidate)
        }
    }

    /// The start line of the previous visible row above `line`'s row, clamped
    /// at 0. Drives `k`.
    pub fn prev_visible(&self, line: usize) -> usize {
        let header = self.header_of(line);
        if header == 0 {
            0
        } else {
            self.header_of(header - 1)
        }
    }

    /// How many visible rows lie between `top` and `line` (exclusive of `top`,
    /// inclusive of `line`'s row) — i.e. the screen row `line` renders on when
    /// `top` is the first rendered row. Used to place the terminal cursor.
    ///
    /// `top` and `line` are snapped to their headers first, so passing a hidden
    /// line still yields the row of its fold header.
    pub fn rows_between(&self, top: usize, line: usize) -> usize {
        let top = self.header_of(top);
        let target = self.header_of(line);
        if target <= top {
            return 0;
        }
        let mut rows = 0usize;
        let mut cur = top;
        while cur < target {
            cur = self.row_end(cur) + 1;
            rows += 1;
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(folds: &[(usize, usize, bool)]) -> FoldSet {
        let mut s = FoldSet::new();
        for &(a, b, closed) in folds {
            s.create(a, b);
            if !closed {
                s.open_one(a);
            }
        }
        s
    }

    #[test]
    fn create_makes_a_closed_fold() {
        let mut s = FoldSet::new();
        assert_eq!(s.create(1, 4), Some(1));
        assert_eq!(s.folds().len(), 1);
        assert!(s.folds()[0].closed);
        assert_eq!(s.folds()[0].line_count(), 4);
    }

    #[test]
    fn single_line_range_is_rejected() {
        let mut s = FoldSet::new();
        assert_eq!(s.create(3, 3), None);
        assert!(s.is_empty());
    }

    #[test]
    fn collapsed_hides_interior_lines_only() {
        let s = set(&[(1, 4, true)]);
        let rows = s.collapsed();
        assert_eq!(rows.fold_at(1), Some((1, 4)));
        assert!(!rows.is_hidden(1)); // header visible
        assert!(rows.is_hidden(2));
        assert!(rows.is_hidden(4));
        assert!(!rows.is_hidden(5));
    }

    #[test]
    fn disabled_foldenable_collapses_nothing() {
        let mut s = set(&[(1, 4, true)]);
        s.disable();
        assert!(s.collapsed().is_empty());
        s.enable();
        assert_eq!(s.collapsed().fold_at(1), Some((1, 4)));
    }

    #[test]
    fn j_over_a_closed_fold_skips_it() {
        // Lines 0..=5, closed fold on 1..=4. From header row 1, `j` lands on 5.
        let rows = set(&[(1, 4, true)]).collapsed();
        assert_eq!(rows.next_visible(1, 5), 5);
        // From line 0, `j` lands on the fold header 1.
        assert_eq!(rows.next_visible(0, 5), 1);
        // `k` from line 5 lands back on the header 1.
        assert_eq!(rows.prev_visible(5), 1);
    }

    #[test]
    fn open_reveals_the_content() {
        let mut s = set(&[(1, 4, true)]);
        assert!(s.open_one(1));
        assert!(s.collapsed().is_empty());
        assert!(!s.folds()[0].closed);
    }

    #[test]
    fn nested_open_close_one_level() {
        // Outer 0..=9 closed, inner 2..=5 closed. Cursor on line 0.
        let mut s = FoldSet::new();
        s.create(0, 9);
        s.create(2, 5);
        // Both closed; outermost at line 0 is the 0..=9 fold.
        assert_eq!(s.collapsed().ranges(), &[(0, 9)]);
        // zo opens outer, revealing the inner (still closed) header at line 2.
        assert!(s.open_one(0));
        assert_eq!(s.collapsed().ranges(), &[(2, 5)]);
        // zo on line 2 opens the inner.
        assert!(s.open_one(2));
        assert!(s.collapsed().is_empty());
    }

    #[test]
    fn close_recursive_then_open_recursive() {
        let mut s = FoldSet::new();
        s.create(0, 9);
        s.create(2, 5);
        s.open_all();
        assert!(s.collapsed().is_empty());
        // zC on a line inside both closes the whole subtree.
        s.close_recursive(3);
        assert_eq!(s.collapsed().ranges(), &[(0, 9)]);
        // zO on line 0 opens the outer fold *and* the inner one nested in it,
        // even though line 0 is not itself inside the inner fold.
        s.open_recursive(0);
        assert!(s.collapsed().is_empty());
        // zv only reveals folds covering the cursor line: closing both then
        // zv on line 0 leaves the inner (which does not cover line 0) closed.
        s.close_recursive(3);
        s.view_cursor(0);
        assert_eq!(s.collapsed().ranges(), &[(2, 5)]);
    }

    #[test]
    fn levels_reflect_nesting() {
        let mut s = FoldSet::new();
        s.create(0, 9); // outer
        s.create(2, 5); // inner
        let outer = s.folds().iter().find(|f| f.start == 0).unwrap();
        let inner = s.folds().iter().find(|f| f.start == 2).unwrap();
        assert_eq!(outer.level, 0);
        assert_eq!(inner.level, 1);
    }

    #[test]
    fn delete_removes_innermost() {
        let mut s = FoldSet::new();
        s.create(0, 9);
        s.create(2, 5);
        assert!(s.delete_at(3)); // innermost at line 3 is 2..=5
        assert_eq!(s.folds().len(), 1);
        assert_eq!(s.folds()[0].start, 0);
        s.delete_all();
        assert!(s.is_empty());
    }

    #[test]
    fn navigation_between_folds() {
        let s = set(&[(1, 3, true), (6, 8, true)]);
        assert_eq!(s.next_fold_start(0), Some(1));
        assert_eq!(s.next_fold_start(1), Some(6));
        assert_eq!(s.next_fold_start(6), None);
        assert_eq!(s.prev_fold_end(9), Some(8));
        assert_eq!(s.prev_fold_end(6), Some(3));
        assert_eq!(s.prev_fold_end(1), None);
        assert_eq!(s.current_fold_start(2), Some(1));
        assert_eq!(s.current_fold_end(2), Some(3));
        assert_eq!(s.current_fold_start(5), None);
    }

    #[test]
    fn rows_between_counts_visible_rows() {
        // Fold 1..=4 closed. Visible rows: 0, [1..4], 5, 6.
        let rows = set(&[(1, 4, true)]).collapsed();
        assert_eq!(rows.rows_between(0, 0), 0);
        assert_eq!(rows.rows_between(0, 1), 1); // fold header is row 1
        assert_eq!(rows.rows_between(0, 3), 1); // hidden line 3 maps to header row
        assert_eq!(rows.rows_between(0, 5), 2);
        assert_eq!(rows.rows_between(0, 6), 3);
    }

    #[test]
    fn toggle_one_opens_or_closes() {
        let mut s = set(&[(1, 4, true)]);
        // Under a closed fold -> opens.
        assert_eq!(s.toggle_one(1), None);
        assert!(s.collapsed().is_empty());
        // Now open -> toggling closes and reports the header line.
        assert_eq!(s.toggle_one(2), Some(1));
        assert_eq!(s.collapsed().ranges(), &[(1, 4)]);
    }
}
