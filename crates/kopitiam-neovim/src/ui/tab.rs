//! Tab pages: vim/neovim `:tabnew`/`gt`/`gT`/`:tabclose` — a whole window
//! layout per tab, not a browser buffer-tab.
//!
//! # What a tab page actually is here
//!
//! In vim a *tab page* is not "one file in a strip of file-tabs". It is an
//! entire [`WindowTree`] — the full split layout you see when you `:sp`/`:vs`.
//! Press `gt` and the whole screen's worth of windows swaps out for another
//! whole screen's worth. So [`TabPages`] is a `Vec<WindowTree>` (in tabline
//! display order) plus which index is active — one layout per tab.
//!
//! # THE CENTRAL INVARIANT — viewport-swap discipline (extends AID-0020)
//!
//! kvim keeps exactly **one** editor cursor; a window, and now a tab, is only a
//! viewport onto it. AID-0020 fixed the same discipline one level down: a
//! window is a viewport, so `App` syncs the editor's live cursor into the
//! active [`Window`](crate::ui::window::Window) before switching focus, and
//! loads the target window's cursor back afterwards. Tab pages mirror that
//! exactly, one level up.
//!
//! The load-bearing rule, which you MUST hold in your head to touch this file:
//!
//! * `App` already owns the **currently-active tab's LIVE `WindowTree`** in its
//!   own field `App.windows`. That is the real tree — the one being mutated as
//!   the user splits, moves focus, types.
//! * [`TabPages`] stores a `WindowTree` slot for **every** tab, active one
//!   included. BUT: **the active tab's slot inside `TabPages` is a PARKED
//!   PLACEHOLDER. It is STALE the whole time that tab is active**, because the
//!   real live tree for the active tab is sitting in `App.windows`, NOT here.
//!   Reading `slots[active]` gives you yesterday's layout, not today's.
//! * Switching tabs is therefore an **O(1) [`std::mem::swap`], never a clone**:
//!   park the current live tree back into its slot, move `active`, then swap the
//!   target tab's parked tree out into `live`. No `WindowTree` is ever copied;
//!   the two live/parked halves just trade places.
//!
//! Because of that, every method that changes *which* tab is active takes
//! `live: &mut WindowTree` — and that `&mut` is `&mut App.windows`. On return,
//! `App.windows` (i.e. `live`) is guaranteed to hold the **newly-active** tab's
//! tree, and the tab you left is safely parked in its slot. Get this wrong and
//! you either lose a tab's layout (overwrite without parking) or double a tree
//! (clone instead of swap) — hence the swap-not-clone rule is not a perf nicety,
//! it is what keeps the one-cursor / one-live-tree model honest.
//!
//! # Reading a tab's tree for the tabline
//!
//! [`TabPages::tree_at`] hands back the parked slot. For the **active** tab that
//! slot is stale (see above), so the tabline builder must read `App.windows` for
//! the active tab and [`TabPages::tree_at`] only for the *other* tabs. Its
//! rustdoc says so again at the point of use — cannot repeat this enough lah.

use crate::ui::window::WindowTree;

/// kvim's collection of vim tab pages: one [`WindowTree`] layout per tab, in
/// tabline order, plus the active index.
///
/// # Invariants this type upholds
///
/// * `count() >= 1` always — you can never close the last tab (vim won't, and
///   neither will [`TabPages::close_active`] / [`TabPages::only`]).
/// * `active` is always a valid index into `slots` (`0..count()`).
/// * `slots[active]` is a **stale parked placeholder** while that tab is active;
///   the live tree lives in `App.windows`. See the module docs — this is the
///   whole game.
///
/// Cloning a [`TabPages`] deep-copies every parked layout; that is only wanted
/// for snapshots/tests, never on the hot tab-switch path (which swaps, see the
/// module docs).
#[derive(Debug, Clone)]
pub struct TabPages {
    /// One layout per tab, in tabline display order. `slots[active]` is the
    /// stale parked placeholder for the active tab (its live tree is in
    /// `App.windows`); every other slot is the real, current tree for that tab.
    slots: Vec<WindowTree>,
    /// Which tab is active, 0-based. Always in `0..slots.len()`.
    active: usize,
}

impl TabPages {
    /// Start with exactly one tab, whose layout is `first`. `active == 0`.
    ///
    /// Note the parking discipline (module docs): from this moment `slots[0]` is
    /// treated as the active tab's *parked* slot, so the caller (`App`) is the
    /// one holding the live tree in `App.windows`. Typically `App` keeps its own
    /// live copy and hands an equal `first` in here; they drift apart the instant
    /// the user edits, and that drift is *expected* — `slots[0]` going stale is
    /// exactly the invariant, not a bug.
    pub fn single(first: WindowTree) -> Self {
        Self { slots: vec![first], active: 0 }
    }

    /// How many tab pages there are. Always `>= 1` — an invariant this type
    /// upholds, so you never have to guard against an empty tabline.
    pub fn count(&self) -> usize {
        self.slots.len()
    }

    /// The active tab's 0-based index.
    pub fn active(&self) -> usize {
        self.active
    }

    /// `true` when there is exactly one tab. The caller uses this to guard
    /// `:tabclose` / `:tabonly` — vim refuses to close the final tab, so knowing
    /// you are on the last one lets the UI show the right "already only one tab"
    /// message instead of a silent no-op.
    pub fn is_last(&self) -> bool {
        self.slots.len() == 1
    }

    /// Read-only peek at tab `index`'s **parked** tree, for the tabline builder.
    /// `None` if `index` is out of range.
    ///
    /// WARNING — read this before you trust the result: for the **active** tab
    /// this returns the STALE parked slot, because the live active tree is in
    /// `App.windows`, not here (see module docs). So the tabline builder must
    /// read `App.windows` for the active tab, and this method only for the *other*
    /// tabs. Use it for the active tab and your tabline will show a stale layout
    /// — cursor/splits from before the user's last few edits. Don't say never
    /// warned you hor.
    pub fn tree_at(&self, index: usize) -> Option<&WindowTree> {
        self.slots.get(index)
    }

    /// `:tabnew` — open a fresh tab **right after** the active tab, make it
    /// active, and put its layout (`new_tree`) into `live`.
    ///
    /// The outgoing active tree currently sitting in `live` is parked back into
    /// its slot first (swap, not clone), so no layout is lost. On return `live`
    /// holds `new_tree` and `active` points at the new tab (old `active + 1`).
    /// New tab lands immediately after the active one even when the active tab
    /// is in the middle — same as vim, not appended to the end.
    pub fn open_after_active(&mut self, live: &mut WindowTree, new_tree: WindowTree) {
        // Park the outgoing active tree back into its slot: after this,
        // `slots[active]` is real again and `live` holds throwaway stale bytes
        // that we are about to overwrite via the swap below.
        std::mem::swap(&mut self.slots[self.active], live);

        // Insert the newcomer right after the (now-parked) active tab, and make
        // it the active one.
        let at = self.active + 1;
        self.slots.insert(at, new_tree);
        self.active = at;

        // Swap the newcomer out into `live`; `slots[at]` becomes the stale
        // parked placeholder for the now-active new tab. `live` now holds
        // `new_tree`, as promised.
        std::mem::swap(&mut self.slots[self.active], live);
    }

    /// Switch the active tab to `target` (0-based). Out-of-range clamps to the
    /// last tab. A no-op when `target` is already active (in that case `live` is
    /// left exactly as it was).
    ///
    /// Performs the park/swap so `live` ends up holding `target`'s tree.
    pub fn focus(&mut self, live: &mut WindowTree, target: usize) {
        // count() >= 1 always, so the subtraction can't underflow.
        let target = target.min(self.count() - 1);
        if target == self.active {
            return; // Already here — leave `live` untouched.
        }
        self.swap_to(live, target);
    }

    /// `:tabclose` — close the active tab.
    ///
    /// **Refuses on the last tab**: returns `false` and changes nothing, because
    /// vim will not close the final tab page (that is `:qa` territory, not this
    /// type's job). Otherwise it drops the closed tab, moves `active` to its
    /// **left neighbour** (or `0` if it was already the leftmost), swaps that
    /// tab's tree into `live`, and returns `true`.
    ///
    /// The closed tab's live tree (the one in `live`) is simply dropped — it is
    /// the tab being closed, so we do *not* park it. The stale placeholder slot
    /// for the active tab is what gets removed from `slots`.
    pub fn close_active(&mut self, live: &mut WindowTree) -> bool {
        if self.is_last() {
            return false; // vim never closes the final tab page.
        }

        // Drop the active tab's (stale) parked slot. The real live tree in
        // `live` belongs to this closing tab and will be overwritten by the
        // swap below — no need to keep it.
        let closed = self.active;
        self.slots.remove(closed);

        // Land on the left neighbour, or 0 if we were already leftmost. After
        // the removal the left neighbour keeps its index, so `closed - 1` still
        // points at it.
        self.active = closed.saturating_sub(1);

        // Pull the new active tab's tree into `live`. `slots[active]` becomes
        // the stale placeholder (it now holds the closed tab's bytes as inert
        // garbage, which is fine — the active slot is *allowed* to be stale).
        std::mem::swap(&mut self.slots[self.active], live);
        true
    }

    /// `:tabonly` — drop every tab except the active one; `active` becomes `0`.
    ///
    /// The active tab's live tree in `live` is **untouched** — only the *other*
    /// tabs' parked layouts are dropped. This does not swap (there is no tab to
    /// swap in), but keeps `live` in the signature so the whole tab-mutation API
    /// reads uniformly and stays future-proof.
    pub fn only(&mut self, live: &mut WindowTree) {
        // Keep just the active tab's slot (a stale placeholder, as always for
        // the active tab) and drop the rest. `remove` shifts, but we clear right
        // after, so the shift cost doesn't matter.
        let keep = self.slots.remove(self.active);
        self.slots.clear();
        self.slots.push(keep);
        self.active = 0;
        // `live` deliberately stays as-is: the active layout is the one thing
        // `:tabonly` preserves. Named `live` (not `_live`) to keep the public
        // signature uniform; this discards the borrow without a clippy grumble.
        let _ = live;
    }

    /// `gt` / `gT` / `:tabnext` / `:tabprevious` by a relative step — move `by`
    /// tabs, **wrapping** round the ends like vim does. `forward == true` is
    /// next (`gt`), `false` is previous (`gT`). `by` is clamped up to at least
    /// `1` (a `0` step is meaningless, treat it as one). Park/swap into `live`.
    pub fn step(&mut self, live: &mut WindowTree, by: usize, forward: bool) {
        let n = self.count(); // >= 1
        // Reduce the step modulo the tab count so a huge `by` still wraps
        // correctly and cheaply. `step == 0` (a `by` that is a whole multiple of
        // `n`) lands back on the same tab — the honest wrap result.
        let step = by.max(1) % n;
        let target = if forward {
            (self.active + step) % n
        } else {
            // `+ n` before subtracting keeps this non-negative; `step` is in
            // `0..n`, so `active + n - step` is always `>= 1`.
            (self.active + n - step) % n
        };
        if target != self.active {
            self.swap_to(live, target);
        }
    }

    /// `{count}gt` / `:tabnext {count}` — jump to the **1-based** tab `index`
    /// (vim numbers tabs from 1 for the user). Clamped to `[1, count]`; an
    /// `index` of `0` is treated as `1`. Park/swap into `live`.
    pub fn goto_1based(&mut self, live: &mut WindowTree, index: usize) {
        let n = self.count(); // >= 1
        // Clamp into [1, n], then convert to a 0-based slot index.
        let idx = index.max(1).min(n) - 1;
        if idx != self.active {
            self.swap_to(live, idx);
        }
    }

    // ---------------------------------------------------------------
    // Private: the one place the park/swap discipline is implemented.
    // ---------------------------------------------------------------

    /// Move the active tab to `new_active`, carrying the parked/live trees with
    /// it via two O(1) swaps and no clone.
    ///
    /// Step 1 parks the outgoing live tree back into its slot (so nothing is
    /// lost); step 2 pulls the incoming tab's parked tree out into `live`. If
    /// `new_active == self.active` the two swaps cancel and `live` is left
    /// unchanged — safe, though the public callers already guard that case so
    /// they don't pay for a pointless double-swap.
    fn swap_to(&mut self, live: &mut WindowTree, new_active: usize) {
        std::mem::swap(&mut self.slots[self.active], live);
        self.active = new_active;
        std::mem::swap(&mut self.slots[self.active], live);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BufferId;

    /// A distinct stand-in tree, tagged by buffer id so a test can assert
    /// *which* tree is currently live via `live.active().buffer`.
    fn tree(n: u32) -> WindowTree {
        WindowTree::single(BufferId(n))
    }

    /// Which buffer id the live tree is currently showing — our proxy for
    /// "which tab's layout is live right now".
    fn live_buf(live: &WindowTree) -> u32 {
        live.active().buffer.0
    }

    #[test]
    fn single_starts_with_one_active_tab() {
        let tabs = TabPages::single(tree(0));
        assert_eq!(tabs.count(), 1);
        assert_eq!(tabs.active(), 0);
        assert!(tabs.is_last());
    }

    #[test]
    fn open_after_active_inserts_after_and_makes_new_tab_live() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);

        tabs.open_after_active(&mut live, tree(1));

        assert_eq!(tabs.count(), 2);
        assert_eq!(tabs.active(), 1, "new tab is the one after the old active");
        assert_eq!(live_buf(&live), 1, "the new tree is now live");
    }

    #[test]
    fn open_after_active_inserts_immediately_after_a_middle_tab() {
        // Build three tabs: buffers 0, 1, 2, ending focused on the middle one.
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // tabs: [0,1], active 1
        tabs.open_after_active(&mut live, tree(2)); // tabs: [0,1,2], active 2
        tabs.focus(&mut live, 1); // active 1 (the middle tab, buffer 1)
        assert_eq!(live_buf(&live), 1);

        // Insert a fresh tab (buffer 9) after the middle tab.
        tabs.open_after_active(&mut live, tree(9));

        assert_eq!(tabs.count(), 4);
        assert_eq!(tabs.active(), 2, "new tab sits right after the middle tab, not at the end");
        assert_eq!(live_buf(&live), 9);
        // The old buffer-2 tab got pushed to index 3, not overwritten.
        assert_eq!(tabs.tree_at(3).unwrap().active().buffer, BufferId(2));
    }

    #[test]
    fn focus_swaps_the_target_tree_into_live() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // active 1, live=1

        tabs.focus(&mut live, 0);
        assert_eq!(tabs.active(), 0);
        assert_eq!(live_buf(&live), 0, "tab 0's tree is live again");
    }

    #[test]
    fn focus_clamps_out_of_range_to_the_last_tab() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // 2 tabs, active 1
        tabs.focus(&mut live, 0); // back to tab 0

        tabs.focus(&mut live, 99); // way out of range → clamps to last (index 1)
        assert_eq!(tabs.active(), 1);
        assert_eq!(live_buf(&live), 1);
    }

    #[test]
    fn focus_on_the_already_active_tab_is_a_no_op() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // active 1, live=1

        tabs.focus(&mut live, 1); // already here
        assert_eq!(tabs.active(), 1);
        assert_eq!(live_buf(&live), 1, "live left untouched on a no-op focus");
    }

    #[test]
    fn close_active_refuses_on_the_last_tab() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);

        let closed = tabs.close_active(&mut live);
        assert!(!closed, "vim never closes the final tab page");
        assert_eq!(tabs.count(), 1);
        assert_eq!(live_buf(&live), 0, "live untouched when the close is refused");
    }

    #[test]
    fn close_active_lands_on_the_left_neighbour() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // [0,1], active 1
        tabs.open_after_active(&mut live, tree(2)); // [0,1,2], active 2

        let closed = tabs.close_active(&mut live); // close tab 2
        assert!(closed);
        assert_eq!(tabs.count(), 2);
        assert_eq!(tabs.active(), 1, "focus moves to the left neighbour");
        assert_eq!(live_buf(&live), 1, "the left neighbour's tree is now live");
    }

    #[test]
    fn close_active_leftmost_lands_on_zero() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // [0,1], active 1
        tabs.focus(&mut live, 0); // active 0 (leftmost)

        let closed = tabs.close_active(&mut live); // close the leftmost tab
        assert!(closed);
        assert_eq!(tabs.count(), 1);
        assert_eq!(tabs.active(), 0);
        assert_eq!(live_buf(&live), 1, "the surviving tab (old buffer 1) is live");
    }

    #[test]
    fn only_keeps_just_the_active_tab_and_leaves_live_alone() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1));
        tabs.open_after_active(&mut live, tree(2)); // [0,1,2], active 2, live=2

        tabs.only(&mut live);
        assert_eq!(tabs.count(), 1);
        assert_eq!(tabs.active(), 0);
        assert_eq!(live_buf(&live), 2, "the active tab's live tree is untouched by :tabonly");
    }

    #[test]
    fn step_wraps_forward_and_backward() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1));
        tabs.open_after_active(&mut live, tree(2)); // [0,1,2], active 2

        // gt from the last tab wraps to the first.
        tabs.step(&mut live, 1, true);
        assert_eq!(tabs.active(), 0);
        assert_eq!(live_buf(&live), 0);

        // gT from the first tab wraps back to the last.
        tabs.step(&mut live, 1, false);
        assert_eq!(tabs.active(), 2);
        assert_eq!(live_buf(&live), 2);
    }

    #[test]
    fn step_by_more_than_one_moves_that_many_tabs() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1));
        tabs.open_after_active(&mut live, tree(2));
        tabs.open_after_active(&mut live, tree(3)); // [0,1,2,3], active 3
        tabs.focus(&mut live, 0); // active 0

        tabs.step(&mut live, 2, true); // forward two → tab 2
        assert_eq!(tabs.active(), 2);
        assert_eq!(live_buf(&live), 2);

        // A step of 0 is clamped up to 1.
        tabs.step(&mut live, 0, true); // forward one → tab 3
        assert_eq!(tabs.active(), 3);
    }

    #[test]
    fn goto_1based_is_one_based_and_clamps() {
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1));
        tabs.open_after_active(&mut live, tree(2)); // [0,1,2], active 2

        // 1-based: tab 1 is slot 0.
        tabs.goto_1based(&mut live, 1);
        assert_eq!(tabs.active(), 0);
        assert_eq!(live_buf(&live), 0);

        // index 0 treated as 1.
        tabs.goto_1based(&mut live, 0);
        assert_eq!(tabs.active(), 0);

        // Clamp above the count → last tab.
        tabs.goto_1based(&mut live, 99);
        assert_eq!(tabs.active(), 2);
        assert_eq!(live_buf(&live), 2);
    }

    #[test]
    fn a_tabs_tree_round_trips_through_an_away_and_back_cycle() {
        // The staleness contract: parking then unparking must not lose or
        // corrupt a tab's layout. Prove tab 1's tree survives a full
        // switch-away-and-back cycle intact.
        let mut tabs = TabPages::single(tree(0));
        let mut live = tree(0);
        tabs.open_after_active(&mut live, tree(1)); // active 1, live=buffer 1

        // Away to tab 0...
        tabs.focus(&mut live, 0);
        assert_eq!(live_buf(&live), 0);
        // ...and back to tab 1.
        tabs.focus(&mut live, 1);
        assert_eq!(live_buf(&live), 1, "tab 1's tree survived the away-and-back with no data loss");

        // Tab 0 is now parked; peeking its slot still shows its real tree.
        assert_eq!(tabs.tree_at(0).unwrap().active().buffer, BufferId(0));
    }
}
