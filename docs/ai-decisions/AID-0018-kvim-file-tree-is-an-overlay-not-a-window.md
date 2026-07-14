# AID-0018: kvim's file tree is an overlay, not a window in the `WindowTree` — and the UI seam grew four methods to stop being a lie

* **Status:** Pending review
* **Bead:** `kopitiam-a1e` (review); implements `kopitiam-cj0.10`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Wire the file tree sidebar into kvim's UI, so that `<leader>e` actually opens
> it. [...] **A sidebar is a layout question; decide deliberately whether the
> tree is a window in this tree or a separate overlay, and justify it in
> rustdoc.** Neovim's neo-tree is a real window; that has consequences for focus
> and `:q`.

Mid-task, the maintainer reported three bugs from real use, two of which turned
out to live in the same seam. Those took priority and are recorded here too,
because the *reason* they existed is an architectural fact worth not repeating.

## Decision 1: the file tree is an overlay that reserves columns, not a leaf in `WindowTree`

**What was decided.** `ui::overlay::Overlay` is a new layer that sits between the
frame and the window tree. A sidebar carves its 30 columns out of the frame
*before* `WindowTree::layout` runs; the window tree is laid out in what remains
and never learns the sidebar exists.

**Why.** Neovim's neo-tree genuinely *is* a window, and that is not an accident
of its implementation — it is possible because Neovim's window abstraction is "a
viewport onto a **buffer**", and a buffer with `buftype=nofile` can hold anything,
including a rendered directory listing.

kvim's window abstraction is narrower, on purpose. `ui::window::Window` is
`{ buffer: BufferId, cursor: Position, scroll: Scroll }` — a viewport onto
**text**, with a grapheme-indexed cursor. The file tree has no `BufferId`, no
text, and its "cursor" is a row index into a flattened directory listing. To make
it a window, one of these would have to be true:

1. **Invent a fake `BufferId`.** But `App::render_windows` renders
   `host.buffer()` into every leaf of the tree, so the sidebar would be painted
   with the current file's text.
2. **Widen `Window` into an enum** (`Text | Tree`). This changes the meaning of
   `WindowTree` for every existing caller — including
   `windows.active_mut().cursor = host.cursor()` in the event loop, which would
   start writing the *text* cursor into the tree window.

Neither is worth it for a panel that never splits, never scrolls horizontally,
and never shows a buffer.

**Consequences, stated plainly so nobody rediscovers them:**

* `:sp` / `:vs` keep working, entirely unaware of the sidebar.
* `:q` closes a *buffer* window and can never close the tree. The tree closes
  with `q`, `<Esc>` or `<leader>e` — which is what neo-tree users press anyway.
* There is no `<C-w>h` to move focus into a visible-but-unfocused tree, because
  **kvim has no window-motion keys at all** (`WindowTree` has no `focus_next`).
  Today the way back into the tree is `<leader>e` twice. Filed as
  `kopitiam-cj0.10.2`; it is a window-tree feature, not a sidebar one.

**What would make this wrong.** If kvim ever grows Neovim's "a buffer can be
anything" model — a `Buffer` that is not text, e.g. for a terminal buffer or a
quickfix list — then the cheap thing becomes the right thing and the tree should
become a real window. The trigger to revisit is *the buffer model changing*, not
the sidebar feeling like it ought to be a window.

**The alternative I did not take, and why it still tempts.** A `Window::Tree`
variant is about 40 lines and would have given `<C-w>h` and `:q` for free. I
passed because those 40 lines are in `src/ui/window.rs`'s *semantics*, not its
line count: every consumer of `WindowTree` would need to learn that a leaf might
not have a cursor. That is a tax paid by all future window code, forever, to save
one feature today.

## Decision 2: the overlay layer is generic, though only one overlay exists

`OverlayPlacement` has a `Float` variant that nothing uses. `OverlayOutcome` has
variants (`OpenPath { target: Tab }`) that kvim cannot fully honour. This is
deliberate and is the one place I accepted speculative generality, because the
brief was explicit that five more actions (`FindFiles`, `HopWords`,
`HarpoonMenu`, ...) must drop in "without a rewrite" — and the pickers are
*floats*, not sidebars. Telescope does not resize your buffers.

**What would make this wrong:** if the pickers land and turn out to want
something `OverlayPlacement` cannot express, the abstraction bought nothing and
should be collapsed rather than extended.

## Decision 3 (forced by a live bug): `EditorHost` gained `command_line()` and `selection()`, and `App` stopped keeping a second copy of the command line

The maintainer reported: **typing `:Neotree` echoed nothing**, and **visual mode
highlighted nothing**. Both had the same shape, and it is worth naming:

> The editor's state was right. The renderer was right. **Nothing joined them.**

`EditorHost` — the trait the UI renders against — exposed `handle_key`, `mode`,
`cursor`, `buffer`. There was no way to ask *"what has the user typed?"* or
*"what is selected?"*. The editor had both (`Editor::command_line`,
`Editor::visual_anchor`). The UI had a `CmdlineState` of its own that **nothing
ever wrote to**, and so drew a bare `:` forever.

The fix was not to feed the mirror. It was to **delete** it: the command line is
now derived from the host every frame, exactly as `StatuslineData` always has
been. A second copy of state that already exists elsewhere is a bug waiting for
somebody to forget to sync it, and it duly was.

**The part worth preserving as knowledge, because it will happen again:** 305
tests passed while both bugs were live. Every one of them asserted on editor
*state* or on widget *inputs*. **Not one asserted on a painted cell.** A renderer
can only be tested by reading what it painted; the tests added alongside this fix
render through ratatui's `TestBackend` and assert the literal string `:Neotree`
is on screen. That is the only assertion that could have caught it, and it is now
the house style for `ui/`.

This is also the archetypal **parallel-agent seam failure** (`CLAUDE.md`,
"Parallel agents: one directory, one owner"). Two agents built two correct halves
against a frozen interface that was missing a method, and the interface's very
frozenness is what stopped either of them from noticing. The lesson is not "freeze
fewer interfaces" — it is that a frozen interface needs an owner who checks it is
*sufficient*, not merely stable.

## Decision 4: the panel probes directory readability itself

`plugins::filetree::FileTree` reads directories through `ignore::Walk` and drops
failed entries (`.filter_map(|e| e.ok())`). An unreadable directory therefore
expands successfully, yields zero children, and is **indistinguishable from an
empty directory**. Drawing an empty folder over a permissions error is a quiet
lie.

The engine is finished and was explicitly not mine to change, so the *panel*
probes with one `read_dir` at expand time and draws an honest error row. This is
not tree logic — it changes nothing about what the tree contains — but the clean
fix belongs in the engine (`TreeRow.error: Option<String>`, or `read_children`
surfacing its `io::Error`). Filed as `kopitiam-cj0.10.1`.

**What would make this wrong:** if the engine grows the error field, this probe
becomes redundant and should be deleted rather than left as a second opinion.
