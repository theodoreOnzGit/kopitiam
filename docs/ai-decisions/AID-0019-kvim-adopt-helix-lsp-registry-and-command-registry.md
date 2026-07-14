# AID-0019: kvim should adopt two Helix infrastructure patterns — a workspace-keyed LSP registry, and a typed ex-command registry

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.24` (review), with implementation split across `kopitiam-cj0.12` (LSP sync) and `kopitiam-cj0.13` (command registry)
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

This AID records a *direction*, not a merged change: no kvim `src/` code was
written here (two other agents own that tree). It exists because studying
Helix surfaced two concrete architectural choices in kvim that are the
maintainer's to confirm, and because the beads that implement them
(`kopitiam-cj0.12`, `kopitiam-cj0.13`) will otherwise encode these decisions
silently.

## Context

Helix (`crates/kopitiam-ai/vendor/helix`, MPL-2.0) was studied clean-room as a
maturity reference — *what* a mature modal editor has and *how* it wires the
infrastructure — explicitly filtered through "kvim is vim-modeled, so Helix's
selection-first keymap is not the model to copy." See
`docs/kvim-maturity-reference.md` for the full feature map. Two of Helix's
*infrastructure* patterns (which are independent of the editing model) are
worth adopting as kvim's target architecture.

## Decision 1 — Key LSP sessions by (server, workspace root), not by filetype

**Observation.** `kvim`'s `lsp/client.rs` today holds
`sessions: HashMap<String, RustAnalyzerSession>` keyed by **filetype**. That
means at most one server per language, globally, for the whole editor process.

**Why that is a latent bug, not just a simplification.** The moment a user
opens two projects (two workspace roots) in one kvim session — a monorepo
subcrate and a sibling, or `:e ../other-project/src/lib.rs` — a filetype-keyed
map hands both buffers the *same* `rust-analyzer`, rooted at whichever project
happened to start first. Cross-project go-to-definition and diagnostics then
resolve against the wrong workspace. It works perfectly in the common
single-project case and fails invisibly the first time there are two.

**What Helix does (described, not copied).** Helix's language-server registry
is keyed by the pair *(language-server id, workspace root)*, lazily spawning a
new client the first time a file under a new root of that language is opened,
and reusing an existing client for further files under the same root. Multiple
servers per buffer (e.g. a language server plus a linter) and multiple roots
per session both fall out of that keying naturally.

**Recommendation.** kvim's session map key becomes `(server_id, root)`; server
lookup for a buffer resolves the buffer's workspace root first (walk up for the
project marker — `Cargo.toml`, `.git`, …) and then finds-or-spawns the session
for that `(server, root)`. This is folded into `kopitiam-cj0.12` (the document-
sync lifecycle bead), because lazy spawn and per-document sync are the same
piece of work.

## Decision 2 — Introduce a typed ex-command registry

**Observation.** `kvim`'s `editor/ex.rs` parses the `:` line with a
hand-written `match` on command names. There is no first-class list of "the
commands that exist," so nothing can enumerate them — which is why there is no
`:`-completion, no command palette, and no per-command help.

**What Helix does (described).** Helix keeps a table of typable commands, each
carrying its name, aliases, documentation, and a completion callback. That one
table powers three surfaces at once: parsing/dispatch, the completion menu when
you type in the prompt, and the command palette (its `Space-?`) rendered
through the fuzzy picker.

**Recommendation.** kvim grows an ex-command registry (name, aliases, help
text, argument completer, handler) that `ex::parse`/dispatch is rebuilt on top
of. This is the enabling substructure for `kopitiam-cj0.13` (command-line
history + completion) and a future vim-style command palette. kvim stays
vim-modeled — the *commands* and *keys* are vim's — but the registry structure
is the reusable idea borrowed from Helix.

## Alternatives considered

* **LSP: leave sessions keyed by filetype.** Simpler, and correct for a
  single-project session — which may be the only case the maintainer cares
  about on a phone. Rejected as the *target* because the failure mode is silent
  and the fix is cheap while the surrounding sync code is being written anyway.
  Fully reversible: it is one map key.
* **Commands: keep the hand-written `match`.** Fine while the command set is
  small. Rejected because three separately-requested features
  (`:`-completion, command palette, per-command help) all reduce to "enumerate
  the commands," and building the registry once is less code than bolting
  enumeration onto a `match` three times.
* **Copy Helix's structures directly.** Rejected on licensing (MPL-2.0) and on
  fit — Helix's command set is selection-model; only the *shape* (a registry)
  transfers.

## What would make this wrong

* **Decision 1** is wrong if kvim is only ever meant to hold one workspace at a
  time (e.g. the Android use case is strictly one project), in which case
  filetype keying is not merely adequate but simpler, and the `(server, root)`
  key is over-engineering. If so, keep the filetype key and drop the root
  resolution from `kopitiam-cj0.12`.
* **Decision 2** is wrong if the ex-command set is deliberately frozen at
  today's dozen commands — then a registry is ceremony. But
  `kopitiam-cj0.19` already proposes roughly a dozen more, so the set is
  growing, not frozen.
* Both are wrong if kvim's LSP and command layers are about to be replaced by
  the Lua `vim.*` surface (Phase 4) driving Neovim-compatible plumbing instead
  — in which case the target architecture is Neovim's, not Helix's. AID-0003
  puts Phase 4 after the LSP phase, so this is not expected, but it is the
  scenario that would reverse both decisions at once.
