# AID-0036: kvim manual folds are buffer-scoped, and reach the renderer through a new `EditorHost::collapsed_folds` seam

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.36` (manual folds portion of the z-prefix bead)
* **Date:** 2026-07-17
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Add manual folds (`foldmethod=manual`) to kvim, matching vim: `zf{motion}`/
> visual `zf`/`:{range}fold` to create; `zo`/`zc`/`za`/`zO`/`zC`/`zA`/`zv` to
> open/close; `zR`/`zM`/`zn`/`zN`/`zi` for all/enable; `zd`/`zE` to delete;
> `zj`/`zk`/`[z`/`]z` to move. A closed fold renders as ONE fold line instead of
> its content (vim default foldtext `+-- {N} lines: {first line}····`); the
> cursor cannot sit inside a closed fold, and vertical motion skips a closed
> fold as one line. **"Scope it to the BUFFER for kvim's single-cursor model
> (vim's is window-local; document the deviation)."**

Two judgment calls in here are genuinely the maintainer's: *where fold state
lives*, and *how it crosses the UI seam to reach the renderer*. The brief
directed the first (buffer-scoped) but the consequences of that — and the second
question entirely — warrant a record.

## Decision 1: fold state is buffer-scoped, stored on the `Editor`, not on the `Buffer` and not per-window

**What was decided.** A `HashMap<BufferId, fold::FoldSet>` on `Editor`. A
`FoldSet` is the authoring model (a `Vec<Fold { start, end, closed, level }>`
plus the `foldenable` flag); every fold-mutating `z` command is a method on it.

**Why buffer-scoped and not window-scoped (the brief's instruction, recorded
here with its consequence).** Real vim folds are window-local: the same buffer
in two splits can have different folds open in each. kvim has a single logical
cursor and the overwhelmingly common case is one view per buffer, so a
window-local table would multiply both the state and the edit-tracking problem
(Decision 3) for a distinction almost nobody uses. If per-window folds are ever
needed, the fold map moves from `HashMap<BufferId, _>` to
`HashMap<(WindowId, BufferId), _>` and nothing about the fold *math* changes —
the `FoldSet`/`FoldRows` split was built to make that a relocation, not a
rewrite.

**Why on the `Editor`, not on the `text::Buffer`.** `text::Buffer` is the pure
text engine (rope + undo + marks + line-endings). Folds are a *view/editor*
concept, not text. Marks live on the buffer, so there was a precedent for
putting them there — but marks are addressed by the same grapheme `Position` the
rope already speaks, whereas folds are a distinct model that the *renderer* also
needs. Keeping `Buffer` free of them keeps the frozen `text::Buffer` read API
(and its `BufferView` mirror) unchanged. The editor already owns the buffer
table, so the fold map sits naturally beside it.

## Decision 2: the renderer reaches folds through a new defaulted `EditorHost::collapsed_folds`, receiving an owned `Vec<(usize,usize)>`, not a borrowed `FoldSet`

**What was decided.** `EditorHost` gains one defaulted method,
`fn collapsed_folds(&self, id: BufferId) -> Vec<(usize, usize)>` (default: empty
— correct for the fake/placeholder hosts). The real `Editor` overrides it to
return the *outermost effectively-closed* ranges for a buffer. `App` wraps them
in a `fold::FoldRows` view per frame and hands that to `TextArea` as a new
`folds` field. Nothing below `App` touches the authoring `FoldSet`.

**Why an owned `Vec` of flattened ranges, not `&FoldSet` or the full fold list.**
The render seam (`EditorHost`/`BufferView`) is deliberately a *read* contract the
UI cannot use to reach into editor internals. Returning `&FoldSet` would export
the authoring type across the seam and tie its lifetime to the borrow; returning
the flattened non-overlapping closed ranges gives the renderer exactly the
"visible lines" information it needs and nothing more. Folds per buffer number in
the low tens at most, so the per-frame `Vec` allocation is free in practice. This
is the same split `EditorHost::buffer_by_id` already draws for buffer text: the
editor owns the model, the UI borrows a view.

**Why `FoldRows` is a distinct type from `FoldSet`.** `FoldSet` answers authoring
questions (which fold is innermost at this line, open one level, delete). The
renderer and the vertical-motion code both need a *different* question answered —
"treating each closed fold as one visual row, which buffer line is visible where"
— and both must answer it *identically* or the cursor and the text drift apart.
`FoldRows` is that single shared answer, produced by `FoldSet::collapsed()` for
the editor side and by `FoldRows::from_ranges()` for the render side from the
`collapsed_folds` seam. One implementation, two callers, no chance of divergence.

## Decision 3: a fold does NOT follow buffer edits in this first cut

**What was decided.** Folds hold absolute line numbers and are *not* shifted when
text is inserted or deleted above them. After a structural edit a fold can cover
the wrong lines. `zE` (eliminate all folds) is the escape hatch. This is filed as
a follow-up bead.

**Why deferred.** Pinning folds to lines through every edit needs the same
shift-on-edit machinery the mark table already has (`text::mark`), applied to
both endpoints of every fold. That is a self-contained, well-understood piece of
work, but it is orthogonal to getting the fold *model, grammar, rendering and
motion* right — which is the actual hard part and the deliverable. Shipping the
fold subsystem without edit-tracking is honest and useful (folds you create and
use without editing above them behave exactly like vim); shipping neither would
have been worse.

## Also deferred (filed as follow-up beads, not judgment calls)

* Auto-fold methods (`foldmethod=indent`/`syntax`/`expr`), the fold column
  gutter, and incremental `foldlevel`/`zm`/`zr` — explicitly out of scope per the
  brief.
* Vertical *scrolling* math (`ui::scrolling::vertical_scroll`) still counts
  buffer lines, not visible rows; `scroll.top` is snapped to a fold header before
  rendering so the viewport never starts mid-fold, but on a buffer scrolled a
  long way past large folds the scrolloff arithmetic is approximate. Correctness
  of the fold render/motion does not depend on it; the fix is to make
  `vertical_scroll` fold-aware, filed separately.

## What would make these decisions wrong

* **Decision 1 wrong if** per-window fold state turns out to be wanted sooner
  than expected (e.g. a diff view that folds unchanged hunks in one split only).
  The relocation is mechanical, but every `self.folds.entry(self.current)` call
  site would need the window id threaded in.
* **Decision 2 wrong if** the flattened-range snapshot proves too lossy — e.g. if
  a future foldtext wants the true nesting *level* (vim draws one dash per
  level), which `collapsed()` discards by keeping only outermost ranges. The
  current renderer hard-codes level 0 (`+--`), which matches the brief's exact
  spec but is not vim's per-level dash count. Fixing that means passing level
  through the seam.
* **Decision 3 wrong if** users routinely edit above existing folds and find the
  drift surprising rather than tolerable. The mitigation (edit-tracking) is
  already scoped; this only decides it is not *blocking*.
