# AID-0020: kvim window management — where per-window state lives, who owns `<C-w>`, and why hop is not an overlay

Status: accepted
Date: 2026-07-14
Crate: `kopitiam-neovim`
Related: AID-0003 (kvim architecture, engine/UI separation), AID-0018 (the file tree is an overlay, not a window), bead `kopitiam-cj0.10` family

## Context

Splits (`:sp`/`:vs`) laid out geometrically but were unusable: every pane
rendered the *active* buffer (bug `kopitiam-cj0.10.3`), there was no way to move
focus between panes (`kopitiam-cj0.10.2`), and the `<C-w>` command family did
not exist. Making splits genuinely work forced three architecture decisions,
each of which sits on the editor/UI line `CLAUDE.md` draws ("never place
business logic inside user interfaces").

## Decision 1 — per-window buffer+cursor state lives in the UI's `WindowTree`; the editor keeps a single cursor

`ui::window::Window` already carried `{ buffer: BufferId, cursor: Position,
scroll: Scroll }`. That is where per-window view state stays. The `editor::Editor`
continues to own exactly **one** cursor and **one** active buffer — it edits the
window that currently has focus, and nothing more.

Focus changes are a two-step handoff performed by `App`:

1. `sync_active_window()` writes the editor's live cursor + active buffer id
   back into the outgoing window's `Window` struct;
2. the tree's `active` moves; then `load_active_window()` calls
   `Editor::set_active(buffer, cursor)` to point the single-cursor editor at the
   newly-focused window's saved state.

**Why not give the editor N cursors?** Because a window is a *viewport*, not an
edit context — the same reasoning AID-0018 used to keep the file tree out of the
tree. Threading window identity through every motion, operator, and text object
(each of which reads/writes `self.cursor`) to support splits would contaminate
the entire modal engine with a concept it should never need. The engine stays
window-unaware and headlessly testable; the UI does the bookkeeping. The one new
seam this requires — `EditorHost::buffer_by_id` — is exactly what lets a split
render a *different* buffer than the active one, which is the whole of the
`kopitiam-cj0.10.3` fix.

**Why not move `WindowTree` into the editor?** Its layout is expressed in
ratatui `Rect`s and spatial `<C-w>l` needs pixel geometry; dragging that below
the UI line would pull ratatui into the headless engine. The *tree of splits*
is arguably editor state, but here it is inseparable from geometry, so it stays
in `ui`.

## Decision 2 — `<C-w>` window commands are handled in `App`, not the editor

`<C-w>` in Normal mode is intercepted by `App::handle_event` before the key
reaches the editor: `App` sets `awaiting_window_key` and routes the next key to
`handle_window_key`. Window management (focus, split, close, resize, exchange,
rotate) is the UI's domain — it owns the `WindowTree` — so the command
dispatch lives there.

The subtlety: `<C-w>` in **Insert** mode is the editor's (delete-word-back), so
the interception is gated on `host.mode() == Normal`. Splitting and quitting
that *originate* as ex commands (`:sp`, `:q`) still flow through the editor
(which parses them) and come back out as `EditorResponse::Window(..)` /
`HostResponse::QuitWindow` for `App` to carry out — the same "editor recognises,
UI performs" split `EditorResponse::Write` uses for file I/O. `:q` closes the
active window and only quits the process on the last one, which is why the editor
can no longer decide "quit" by itself.

## Decision 3 — hop (`f`) is a dedicated `App` state, not an `Overlay` variant

The file tree, pickers, and harpoon menu are `Overlay`s: each claims a rectangle
and renders itself into it, geometry-free. Hop is the opposite shape — it paints
labels **onto the buffer's word-starts**, at the exact screen cells those
`(line, column)` positions occupy, which needs the active window's rect, scroll
offset, and buffer text. None of that fits `Overlay::render(frame, rect, ...)`.

So hop reuses the overlay layer's *focus discipline* (while a hop is live, keys
go to it and the editor never sees them) without being an `Overlay`: it is a
small `App` field (`hop: Option<HopState>`) drawn inline in `render_windows`
after the active window's text. This is the same judgement AID-0018 records for
the file tree — share the model that fits, don't contort the code to force a
shared type.

The `f` overload is preserved: idle `f` in Normal mode is hop (the maintainer's
remap → `Action::HopWords`), but `df<x>` stays find-char, because the editor
only emits `Action::HopWords` when the operator-pending state is idle (existing
test `operator_composed_find_still_works_despite_the_f_keymap_shadow`, plus new
`operator_composed_f_still_finds_a_char_and_does_not_trigger_hop`).

## Consequences

- Each window renders its own buffer with its own cursor; inactive splits keep
  their scroll (`kopitiam-cj0.10.3` fixed).
- Spatial `<C-w>h/j/k/l` uses the laid-out rects; `w`/`W`/`p` cycle;
  `s`/`v`/`n`/`c`/`q`/`o`/`=`/`+`/`-`/`<`/`>`/`x`/`r` all work
  (`kopitiam-cj0.10.2` fixed).
- Deferred honestly: `<C-w>H/J/K/L` (move-to-edge, restructures the tree —
  `kopitiam-cj0.10.5`), `<C-w>T` / tab pages (`kopitiam-cj0.10.6`), and a real
  `:term` terminal emulator (`kopitiam-cj0.10.4`, which opens an honest
  placeholder buffer today).
- Verified through the real binary over a PTY: `:vs` + `<C-w>l` paints two
  different buffers with focus in the right pane, and `f` overlays jump labels
  on the focused pane's word-starts.
