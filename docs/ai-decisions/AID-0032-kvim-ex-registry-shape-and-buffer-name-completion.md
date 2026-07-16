# AID-0032: kvim's ex-command registry keeps argument parsing in `ex::parse`, and `:b` learns to take a buffer name

* **Bead:** `kopitiam-cj0.13` (command-line history + editing + completion). Builds on AID-0019's confirmed recommendation (reviewed under the now-closed `kopitiam-cj0.24`).
* **Status:** Pending review.
* **Author:** AI (autonomous session).

## Context

AID-0019 (confirmed) recommended kvim grow a *typed ex-command registry* —
name, aliases, help, argument completer — and said "`ex::parse`/dispatch is
rebuilt on top of" it. That registry is the enabling substructure for
`kopitiam-cj0.13`'s `:`-completion. Implementing cj0.13 forced two concrete
shape decisions that the AID left open, and one small feature extension that
the completion work would otherwise be useless without.

## Decision 1 — the registry owns the *vocabulary*, `ex::parse` keeps the *argument grammar*

AID-0019's wording ("rebuilt on top of") admits two readings:

* **(a) Full rebuild.** Move all parsing into the registry: every `CommandSpec`
  carries a parser callback, and `ex::parse` becomes a thin driver.
* **(b) Vocabulary-only.** The registry owns the name/alias list and each
  command's argument *kind* (for completion); `ex::parse` still owns the
  per-command argument *grammar*.

**Chosen: (b).** `command::lookup(name)` returns a `CommandId`; `ex::parse`
matches on that `CommandId` instead of on a raw name string, then does exactly
the argument parsing it did before (ranges, `:s/pat/rep/g` delimiters,
`:set key=val`, `:b{n}`). The registry additionally tags each command with an
`ArgKind` (`None`/`File`/`Buffer`) that drives `<Tab>` completion.

**Why.** The argument grammars are genuinely per-command and irregular — a
substitute delimiter parser has nothing in common with a `:set` splitter. A
callback-per-command registry would not *share* any of that logic; it would
just relocate it into the table and add a layer of boxed indirection. The win
AID-0019 actually wants — a single enumerable source of command *names* so
completion/palette/help stop being impossible — is fully delivered by (b): the
alias list now lives only in `command::COMMANDS`, and `ex::parse` no longer
carries a duplicate copy. Behaviour is byte-for-byte identical (the existing
`ex` parse tests pass unchanged).

**Alternatives considered.**
* Full rebuild (a) — rejected as more code and more indirection for no
  additional capability today. It becomes attractive only if user-defined
  commands (`:command`) land, where a per-command handler *is* the natural
  shape; at that point (b) extends into (a) without rework, because the
  `CommandId` seam is already there.
* Leave the `match` alone, add a parallel name list only for completion —
  rejected: two lists that must be kept in sync is exactly the duplication
  AID-0019 set out to remove, and the failure mode (a new command dispatches
  but doesn't complete, or vice-versa) is silent.

## Decision 2 — `:b` accepts a buffer *name*, not only a number

The brief asks for buffer-name completion on `:b`/`:buffer`. But `:b` as it
stood parsed its argument with `arg.parse::<usize>()` — a completed *name*
(`:b alpha.txt`) would have parsed to `Unknown` and errored on Enter. Offering
a completion that cannot run is worse than offering none.

**Chosen:** `:b {arg}` now resolves a non-numeric `arg` to a buffer by name —
exact basename match first, then first path containing the substring — via a
new `ExCommand::GotoBufferName`. Numeric `:b{n}` is unchanged. This is vim's
actual behaviour (`:b` takes a unique-substring name as well as an index), so
it is a correctness gain, not scope creep.

## Decision 3 — wildmenu, and session-scoped history

Two smaller judgments, recorded for completeness:

* **`<Tab>` cycles, with a wildmenu strip.** kvim is neovim-modeled and neovim
  defaults `wildmenu` on, so `<Tab>` completes the first match and a horizontal
  candidate strip paints in the status-line row (vim's WildMenu position),
  selected item highlighted; repeated `<Tab>` cycles, `<S-Tab>` reverses. The
  cmdline text itself always shows the current candidate, so the feature is
  usable even where the strip is clipped.
* **History is session-scoped.** vim persists `:`/`/` history across sessions
  via `viminfo`/`shada`. kvim does not yet, and this commit does not add it —
  filed as a follow-up bead. The `History` type is deliberately serialisation-
  ready (a plain `Vec<String>`) so persistence is additive.

## What would make this wrong

* **Decision 1** is wrong if user-defined commands (`:command`) or a plugin
  command surface land *soon* and want a handler-per-command table — then the
  full rebuild (a) was the cheaper end state and (b) is an intermediate stop.
  Reversible: the `CommandId` match is the exact seam a handler table would
  replace, command by command.
* **Decision 2** is wrong if kvim deliberately wants `:b` to stay numeric-only
  (e.g. to reserve name-arg semantics for a future fuzzy buffer picker). Then
  drop `GotoBufferName` and make `:b`-completion offer numbers instead. Low
  cost either way.
* **Decision 3 (history persistence absent)** is wrong only if cross-session
  recall is considered table-stakes for the first cut — in which case the
  follow-up bead should be pulled forward, not deferred. Nothing here blocks
  it.
* All three reverse together if Phase 4 replaces kvim's command layer with the
  Neovim `vim.*` / Lua plumbing (AID-0003 puts Phase 4 after LSP, so not
  expected) — the target would then be Neovim's command model, not this
  registry.
