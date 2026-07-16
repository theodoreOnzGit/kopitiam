# AID-0035: kvim's fuzzy pickers are one overlay panel over three sources, not three picker types

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.10` (pickers portion); harpoon/git split out to `kopitiam-cj0.10.7`/`.8`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Implement the picker OVERLAY — a telescope-style modal (prompt line + scrolling
> fuzzy-filtered list) for `\ff` (find files), `\fb` (find buffers), `\fh` (find
> help). Keep one shared picker component parameterised by its candidate source +
> on-accept action, so all three share the code (like telescope's
> picker/finder/sorter split).

The scoring engine (`plugins::picker::Picker<T>`, `nucleo`-backed, generic over a
`Searchable` item) already existed and was tested. What was missing was the *UI*:
the floating box, the query line-editor, the focus routing, and the wiring from
the three `Action`s into it.

## Decision: one `PickerPanel` over `Picker<PickRow>`, where a `PickRow` carries its own on-accept `PickAction`

**What was decided.** A single non-generic `ui::picker::PickerPanel` is the face
of all three pickers. It owns one `plugins::picker::Picker<PickRow>`, where a
`PickRow` is `{ label: String, action: PickAction }` and `PickAction` is a
three-variant enum (`OpenFile(PathBuf)` / `SwitchBuffer(BufferId)` /
`OpenHelp(String)`). The panel neither knows nor branches on *which* picker it
is — it reads the selected row's `action` on `<CR>` and turns it into the
matching `OverlayOutcome`. A "source" is therefore just "a `Vec<PickRow>`", built
by `App::open_file_picker` / `open_buffer_picker` / `open_help_picker`.

This mirrors the engine's own rationale (see `plugins::picker`'s module docs):
telescope is one widget with pluggable finders/sorters, and writing
`FilePickerPanel` / `BufferPickerPanel` / `HelpPickerPanel` would triplicate the
prompt-and-list rendering, the query line-editing, and the key routing — then
quadruplicate it for the next source (LSP symbols, git status).

**Why a `PickAction` enum rather than a boxed closure per row.** telescope stores
a Lua callback on each entry. kvim's UI layer is not allowed to reach into the
editor (that is the whole point of the `EditorHost` seam and `OverlayOutcome` —
an overlay that can touch the editor cannot be unit-tested without one). A closure
capturing `&mut editor` would break that; a closed enum of *intents* keeps the
panel pure and lets the `App` perform the effect. Three variants is the whole
current vocabulary, and adding a fourth source is one variant plus one arm in
`handle_overlay_key` — the same shape the file tree's `OpenTarget` already uses.

**Why the panel owns the query string, not the engine.** `Picker::set_query`
takes a finished `&str` and re-scores; it does not line-edit. Appending a char,
chopping one with `<BS>` — that is UI state, so `PickerPanel::query` holds it and
hands the result to the engine. Same division `cmdline` (collects the text) and
`ex` (parses it) already draw.

**Why new `OverlayOutcome` variants (`PickPath`/`PickBuffer`/`PickHelp`) instead
of reusing `OpenPath`.** The file tree's `OpenPath` deliberately keeps its
overlay *open* (neo-tree stays visible so `i`/`s` can open a second file).
telescope does the opposite: it disappears the instant you pick. Reusing
`OpenPath` would have forced a "close-after" flag onto an outcome whose documented
contract is "stay open". Three explicit pick-and-close outcomes say what they mean.

**Why `\fb`/`\fh` needed two new `EditorHost` methods.** The seam had no way to
enumerate open buffers or switch to one by id preserving its saved cursor
(`set_active` requires the caller to supply a cursor the UI does not have). Added
`buffers() -> Vec<BufferEntry>` and `focus_buffer(id)`, both defaulted so fakes
and the placeholder need not implement them. `\fh` needed nothing new — it runs
`:help <topic>` through the existing `run_ex` seam, so the jump table stays the
editor's `help::TOPICS` single source of truth.

**Why the file walk is capped (`FILE_PICKER_CAP = 10_000`).** `walk_files` gained
a `cap` parameter and `.take(cap)`. A monorepo can hold hundreds of thousands of
tracked files; materialising and re-scoring all of them on every keystroke would
stall the editor on open, and nobody scrolls past the first screen of a fuzzy
list — you type to narrow it. `.gitignore` already keeps `target/` out, so the
cap only bites on genuinely enormous trees, where a bounded responsive list beats
a complete frozen one. `require_git(false)` was added at the same time so the
ignore rules apply whether or not `git init` has run, matching the file tree and
`:grep`.

## Alternatives considered

1. **Three concrete panel types.** Rejected: triplicated rendering/editing/routing
   for zero behavioural difference, and the maintainer's brief explicitly asked
   for the shared component.
2. **A trait-object source (`Box<dyn PickerSource>`) with a `on_accept(&mut App)`
   method.** More "OO", but it either re-introduces the UI-touches-editor coupling
   the seam exists to prevent, or it just wraps the same `PickAction` enum in a
   vtable for no gain. The enum is simpler and testable.
3. **Making `Picker<T>` itself the overlay (generic `Overlay<T>`).** Rejected:
   `Overlay` is a single non-generic enum owned by `App`; making it generic over
   the picker's item type would infect every overlay (the file tree, future
   harpoon menu) with a type parameter they do not want. The erasure happens at
   the `PickRow` boundary instead — exactly where telescope erases it.
4. **Reusing `OpenPath` with a close flag.** Rejected as above: it lies about the
   file tree's documented stay-open contract.

## What would make this wrong

* **If a picker ever needs to stay open after accept** (a multi-select, or a
  "preview then keep browsing" mode). Then the pick-and-close outcomes are too
  coarse and would need a target/stay flag — but that is a real feature decision,
  not a reason to have merged the outcomes speculatively now.
* **If per-row behaviour stops being expressible as data** — e.g. a source whose
  accept genuinely needs to run arbitrary editor logic that cannot be reduced to
  an intent. Then the enum has to grow a general escape hatch, and the
  UI-stays-pure invariant would need re-examining. No current or near-term source
  (LSP symbols, git files, recent files, commands) needs this — they all reduce to
  "open a location" or "run an ex command".
* **If the 10k cap starts hiding files a user expects** (a legitimately huge repo
  browsed without a query). The honest fix is a background/streaming walk via
  `nucleo::Nucleo` (the async engine the sync engine's docs already point at), not
  a bigger constant. Filed as a follow-up if it ever bites.
* **If the char-per-column match highlight matters for CJK/wide filenames.** The
  highlight is cosmetic and the row is clipped either way, so it was left
  simple; a grapheme-width pass is the fix if a real filename ever misaligns.

## Scope left open (not this decision)

`cj0.10` also covers harpoon (`<leader>b`/`<leader><Esc>`/`<leader>q`) and the
statusline git branch. Those are **out of the pickers' scope** and were filed as
`kopitiam-cj0.10.7` (harpoon) and `kopitiam-cj0.10.8` (git statusline). LSP
`gd`/`gr`/`rn` remain pending per the `cj0.10` description. `cj0.10` stays
IN_PROGRESS.
