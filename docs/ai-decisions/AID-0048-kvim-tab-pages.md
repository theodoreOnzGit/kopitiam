# AID-0048: kvim tab pages — how tabs own window trees, and where the active tab lives

Status: Pending review
Date: 2026-07-18
Crate: `kopitiam-neovim`
Related: AID-0020 (kvim window management — per-window state, the sync/load focus
discipline, `<C-w>` ownership), AID-0018 (the file tree is an overlay, not a
window), bead `kopitiam-ygk`

## Context

`:tabnew` did nothing useful. kvim had **no** tab-page concept at all: the ex
registry had no `tab*` commands, the `t` key in the file tree carried an
`OpenTarget::Tab` that resolved to the honest note "kvim has no tab pages yet —
opened in the current window", and one lone unit test literally asserted "tab
pages do not exist". The window layer (`ui::window::WindowTree`) already carried
a doc comment calling itself "one editor 'tab'", so the shape was anticipated —
just never built.

Tab pages are a real architectural addition: `App` gains a collection *above*
`WindowTree`. That crosses the editor/UI line `CLAUDE.md` draws, and it forced
two design calls the maintainer would normally make. Hence this AID.

## Decision 1 — a tab page is a whole `WindowTree`; `App` owns an ordered collection of them, and the active tab's live tree stays in `App.windows`

A tab page in vim/neovim is **not** a browser buffer-tab — it is a whole window
layout. So a tab page *is* a `WindowTree`, and the tab collection
(`ui::tab::TabPages`) is an ordered `Vec<WindowTree>` plus an active index. This
sits entirely in the UI, same reasoning AID-0020 used to keep `WindowTree` in
`ui`: a tab's content is window geometry (ratatui `Rect`s), which must not leak
into the headless editor.

The load-bearing choice is **where the active tab's live tree lives**. Two
shapes were on the table:

* **(A) Move every tab into `TabPages`, make `App.windows` an accessor method.**
  Clean ownership, but it rewrites all ~50 `self.windows.…` / `app.windows` call
  sites (including tests) into `self.windows_mut().…`, touching a lot of
  battle-tested code for no behavioural gain.
* **(B, chosen) Keep `pub windows: WindowTree` as the *active tab's* live tree;
  park the other tabs in `TabPages`.** Switching tabs is an O(1)
  `std::mem::swap` of the live tree with a parked slot — no clone. Every existing
  window/`<C-w>`/split call site keeps working unchanged, because each only ever
  wanted the *active* tab's layout, which is exactly what `App.windows` still is.

(B) is also the honest extension of AID-0020's discipline: a window-focus change
there is "sync the editor cursor out, move `active`, load the new cursor in"; a
tab switch here is the same dance one level up — "sync out, swap the whole tree,
load in". **One editor cursor throughout. A tab is a viewport, not a second edit
context** — the same invariant AID-0020 defends for windows.

The cost of (B) is a documented invariant: `TabPages` stores a slot for *every*
tab, but the active tab's slot is a **stale parked placeholder** while it is
active (the live copy is in `App.windows`). All reads of the active tab go
through `App.windows`; `TabPages::tree_at` is only trustworthy for the *other*
tabs. This is spelled out in `TabPages`'s module docs and on `tree_at`, and the
park/swap is encapsulated in one private helper so no caller open-codes it.

## Decision 2 — tab commands ride the existing "editor recognises, UI performs" seam, as a new `TabCommand`

`:tabnew`/`gt`/… are parsed by the editor (keeping the ex + normal-mode grammar
where it belongs) and handed back as `EditorResponse::Tab(TabCommand)` →
`HostResponse::Tab` → `App::handle_tab_command`. This is byte-for-byte the same
path `:sp`/`<C-w>` uses via `WindowCommand` (AID-0020). A **new** enum
`core::TabCommand` was added rather than overloading `WindowCommand`, because a
tab is genuinely not a window — conflating them would muddy both types' docs and
the dispatch. `TabCommand` carries the relative/absolute distinction vim draws
(`gt` = relative next, `2gt` = absolute tab 2; `:tabnext` vs `:tabnext 3`).

`gt`/`gT` are recognised in the operator-pending grammar (`pending.rs`, gated on
`operator.is_none()` so `dgt` is not a tab switch), not intercepted in `App`
like `<C-w>` — because `g` is already an editor prefix (`gg`, `gf`, `ge`), and
intercepting it in the UI would shadow all of those.

## What was delivered

* Ex-commands: `:tabnew`/`:tabedit [file]`, `:tabclose`, `:tabonly`,
  `:tabnext`/`:tabn [count]`, `:tabprevious`/`:tabp [count]`,
  `:tabfirst`/`:tabrewind`, `:tablast`, `:tabs`.
* Normal-mode: `gt` (next), `gT` (prev), `{count}gt` (absolute jump), and
  `<C-w>T` (move the current window to a new tab).
* A tabline at the top per `'showtabline'` **default** (show when ≥2 tabs): each
  entry is ` {n} {name}{+} `, active tab highlighted (`bg2`) vs inactive
  (`bg1`/`gray`), reusing the statusline's theme-segment idiom.
* `<C-w>` window commands stay *within* the active tab's `WindowTree` (a split in
  one tab does not leak into another — covered by a test).

## Consequences

* `App.windows` is now "the active tab's tree", not "the tree". Its doc says so.
* Switching tabs is allocation-free (swap, not clone) and preserves each tab's
  full split layout + per-window scroll.
* The `'showtabline'` `0` (never) / `2` (always) overrides are **deferred** — the
  default (`1`) is implemented directly in `App::should_show_tabline`. Follow-up
  bead filed. Wiring the knob needs the `:set` → `App.options` propagation, which
  is a separate seam.

## What would make this wrong

* **If the stale-active-slot invariant is violated** — e.g. a future caller reads
  `TabPages::tree_at(active)` expecting live data — the tabline (or any tab
  reader) would show the active tab's *pre-switch* layout. Mitigation: the
  invariant is documented at the type, `tree_at`, and `App.tabs`, and
  `tabline_entries` routes the active tab through `App.windows` explicitly. If
  this footgun bites in practice, shape (A) (make `App.windows` an accessor over
  a single `TabPages`) is the reversal — more churn, but no stale slot.
* **If tabs ever need per-tab editor state the single cursor cannot express**
  (e.g. a genuinely independent second cursor, or tab-local options), the
  "one cursor, tab is a viewport" invariant breaks and the editor would need to
  learn about tabs — exactly what AID-0020 argues against for windows. No such
  need exists today.
* **If `WindowId`s must be globally unique across tabs** (they are per-tree today,
  each tab's tree starting at `WindowId(0)`), some future cross-tab window
  addressing would collide. Nothing addresses windows across tabs today.
