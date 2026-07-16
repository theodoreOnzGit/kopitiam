# AI Decisions

Decisions that an AI collaborator made **on the maintainer's behalf**, while
working autonomously, where the maintainer would normally have been asked.

This directory is not a substitute for Architecture Decision Records. The two
serve different purposes:

* **ADRs** record what the project decided and why, regardless of who decided
  it. They are permanent architectural history.
* **AI Decisions (AIDs)** record that a *judgment call was made without the
  maintainer present*, so the maintainer can review, confirm, or reverse it
  when they return.

Every AID is also filed as a `bd` issue labelled `ai-decision`, so
`bd list --label=ai-decision` surfaces the full review queue.

## Lifecycle

1. AI hits a decision that is genuinely the maintainer's to make.
2. AI makes its best judgment, executes it, and writes an AID here explaining
   the reasoning, the alternatives, and what would have to be true for the
   decision to be wrong.
3. AI files a `bd` issue pointing at the AID.
4. Maintainer reviews. If they agree, the bead is closed and the AID is marked
   **Confirmed**. If they disagree, the AID is marked **Reversed**, a new AID
   records the replacement decision, and the code is corrected.

An AID is never deleted, even when reversed — a reversed decision is still
project history worth keeping.

## Status values

| Status | Meaning |
| --- | --- |
| **Pending review** | Executed by the AI, not yet seen by the maintainer. |
| **Confirmed** | Maintainer reviewed and agreed. |
| **Reversed** | Maintainer disagreed; superseded by a later AID. |
| **Superseded** | Overtaken by a later decision, not because it was wrong. |

## Index

| ID | Decision | Status | Bead |
| --- | --- | --- | --- |
| [AID-0001](AID-0001-runtime-crate-naming-and-layout.md) | Kopitiam Runtime crate naming and workspace layout | Pending review | `kopitiam-082.2` |
| [AID-0002](AID-0002-scope-of-finish-all-of-kopitiam.md) | What "finish all of kopitiam" was taken to mean | Pending review | — |
| [AID-0003](AID-0003-kopitiam-neovim-architecture.md) | kopitiam-neovim: Lua compatibility, Android, and what `kvim` is | Lua VM **confirmed**; rest pending | `kopitiam-cj0` |
| [AID-0004](AID-0004-devicons-and-font-shipping.md) | devicons on Android — ship the font, not the glyph | Pending review | `kopitiam-cj0.8` |
| [AID-0005](AID-0005-android-lsp-acquisition.md) | How kvim gets a language server on Android | Partly confirmed | `kopitiam-cj0.9` |
| [AID-0006](AID-0006-kopitiam-mux-fork-strategy.md) | How to fork rmux | **Confirmed: full fork** | `kopitiam-2yg` |
| [AID-0007](AID-0007-lua-coroutines-force-a-bytecode-vm.md) | `kopitiam-lua` is a bytecode VM, not the specified tree-walker — coroutines leave no other option | Pending review | `kopitiam-0pz` |
| [AID-0008](AID-0008-visual-basic-support.md) | Visual Basic: which dialects, and a native parser instead of an LSP | Pending review | `kopitiam-7ef` |
| [AID-0009](AID-0009-syntax-highlighting.md) | Syntax highlighting: no tree-sitter (grammar ecosystem isn't pure Rust), hand-written pure-Rust lexers instead | Pending review | `kopitiam-2qi` |
| [AID-0010](AID-0010-what-hdb-survey-means.md) | What "HDB survey" was taken to mean (redirected mid-task to the resale market) | Pending review | `kopitiam-z8f` |
| [AID-0011](AID-0011-cpf-what-to-populate-and-what-to-refuse.md) | CPF: what to populate, what to refuse, and how to date it (no date/decimal deps; post-55 allocation deliberately absent) | Pending review | `kopitiam-b1n` |
| [AID-0012](AID-0012-hdb-policy-populates-gaps-not-guesses.md) | HDB policy: an empty table beats a plausible number (what was populated, and what was refused) | Pending review | `kopitiam-6eo` |
| [AID-0013](AID-0013-web-search-engines-and-tls.md) | `kopitiam-web`: which search engines, and the fact that "rustls" is not pure Rust | Pending review | `kopitiam-b4u` |
| [AID-0014](AID-0014-legal-and-insurance-are-one-engine.md) | `kopitiam-legal` and `kopitiam-insurance` are one engine, and the seam is temporal (an endorsement IS an amendment) | Pending review | `kopitiam-3zj` |
| [AID-0015](AID-0015-insurance-document-engine-seam.md) | kopitiam-insurance builds on kopitiam-document per-page, and reports its shortfalls rather than forking | Pending review | `kopitiam-b1i` |
| [AID-0016](AID-0016-health-builds-on-insurance.md) | `kopitiam-health` builds **on** `kopitiam-insurance`, not beside it (stubs deleted mid-task when the insurance crate landed) | Pending review | `kopitiam-bfq` |
| [AID-0017](AID-0017-plot-vector-extraction.md) | `kopitiam-plot` reads PDF vector paths itself (not raster, not `pdf-extract`'s callbacks) — digitisation is geometry, not image processing | Pending review | `kopitiam-szg` |
| [AID-0018](AID-0018-bibliography-names-provenance-and-the-network-seam.md) | `kopitiam-bibliography`: an unsplittable name is split but the split is **never emitted**; no `kopitiam-web` dependency; a fifth provenance model that should be hoisted into the ontology | Pending review | `kopitiam-bjo` |
| [AID-0018](AID-0018-kvim-file-tree-is-an-overlay-not-a-window.md) | kvim's file tree is an overlay, not a `WindowTree` window — and the `EditorHost` seam was missing the two methods that made `:` commands and visual mode invisible | Pending review | `kopitiam-a1e` |
| [AID-0019](AID-0019-kvim-adopt-helix-lsp-registry-and-command-registry.md) | kvim should adopt two Helix infrastructure patterns: a workspace-keyed (not filetype-keyed) LSP session registry, and a typed ex-command registry powering completion/palette/help | Pending review | `kopitiam-cj0.24` |
| [AID-0020](AID-0020-kvim-window-focus-and-per-window-state.md) | kvim keeps one editor cursor; per-window view state (buffer, cursor, scroll) lives in the UI's `WindowTree`, and focus changes hand off via `sync_active_window`/`load_active_window` — a window is a viewport, not an edit context | Pending review | `kopitiam-cj0.10.2` |
| [AID-0021](AID-0021-semantic-lsp-request-layer-boundaries.md) | kopitiam-semantic's LSP request layer: pure `lsp_types` shape-parsers (Location/LocationLink, CompletionItem/CompletionList, Hover unions), `char`-offset conversion seams, and diagnostics as a notification-fed store rather than a request | Pending review | `kopitiam-yxj` |
| [AID-0022](AID-0022-lsp-server-readiness-detection.md) | LSP `wait_for_indexing`: rust-analyzer's indexing token is `rustAnalyzer/cachePriming` (not "index"), so connect blocked the full 180s on every start; now rust-analyzer waits for the precise cachePriming end and other servers use an idle heuristic — 180s → ~3s | Pending review | `kopitiam-uab` |
| [AID-0023](AID-0023-kvim-lsp-attaches-on-open.md) | kvim attaches the language server when a served file is opened (not on the first gd/hover), so diagnostics appear on their own; accepts a one-time synchronous connect stall per server, with an async LSP client as the filed follow-up | Pending review | `kopitiam-cj0.26` |
| [AID-0024](AID-0024-snippets-are-a-crate-completion-sources.md) | The snippet engine is its own crate (`kopitiam-snippet`, clean-room LSP snippet grammar, not a LuaSnip fork); the completion menu aggregates LSP + buffer + snippet sources through the existing `merge_and_rank` | Pending review | `kopitiam-cj0.28` |
| [AID-0025](AID-0025-real-paper-fixture-keeps-its-real-citations.md) | Originally kept a real-paper bibliography regression fixture; **reversed** on maintainer instruction (full corpus scrub) — the fixture was removed and its citations neutralized so no personal work product remains in the tree | Reversed | — |
| [AID-0026](AID-0026-completion-snippetsupport-and-tabstop-scope.md) | The LSP-snippet path needed a second semantic file (`snippetSupport: true` in `initialize`, else `insertTextFormat` is never `2` and the new flag is dead — safe, kvim is the only consumer that inserts `insert_text`); and the first completion-menu cut ships tabstop navigation but defers live placeholder-select / mirror editing until the stub engine lands | Pending review | `kopitiam-cj0.17` |
| [AID-0027](AID-0027-window-ux-batch-does-not-close-cj0-10-1.md) | The kvim window-UX batch closes `cj0.30`/`q8v` but deliberately leaves `cj0.10.1` open — that bead is a filetree *engine* bug (unreadable-vs-empty directory), not the file-tree-focus feature the task's one-line "closes cj0.10.1" implied | Pending review | `kopitiam-64c` |
| [AID-0028](AID-0028-async-lsp-session-actor.md) | The async LSP session (`AsyncRustAnalyzerSession`) is a single-owner background-thread **actor** running boxed-closure jobs, not an `Arc<Mutex<session>>` (a lock would reintroduce UI blocking) nor a typed request enum (needless duplication); pre-ready requests **reject** with `NotReady(state)` rather than queue, since the polling caller retries for free. Scaffold only — kvim wiring and workspace-keyed spawn dedup remain | Pending review | `kopitiam-cj0.27` |
| [AID-0029](AID-0029-adapter-selection-gate-is-load-not-checksum.md) | The CLI's local-model selection gate is "`LocalAdapter::load` succeeds", not "`ModelStore::verify` passes" (placeholder catalog checksums make verify useless today) and it never autofetches in a workflow command; a present, loadable `.gguf` gets real inference, everything else degrades to Echo with a note | Pending review | `kopitiam-oii` |
| [AID-0030](AID-0030-inclusive-motion-eol-newline.md) | Building `C`/`D`/`Y` surfaced a pre-existing bug: inclusive motions (`$`/`e`/`f`/`t`) at end-of-line swallowed the trailing newline, merging lines. Fixed at the source in `operator::charwise_range` (one place all inclusive motions funnel through), so `d$`/`de`/`y$` are corrected too; cross-line `%` is unaffected | Pending review | `kopitiam-cj0.41` |
| [AID-0031](AID-0031-kvim-tmux-autoconfig.md) | kvim's tmux `is_vim` auto-fix: writes the christoomey-**canonical** regex (matches `nvim` too) with `kvim` slotted in, not the README's abbreviated `k?vim?x?` (README block reconciled to match); a decline is remembered with a marker file in kvim's *own* dir, never the user's conf; `compute_fix` is pure and only `apply` (post-consent, backup-first) ever writes | Pending review | `kopitiam-cj0.31` |
| [AID-0032](AID-0032-kvim-ex-registry-shape-and-buffer-name-completion.md) | Building cj0.13's command-line completion: the ex-command **registry owns the vocabulary** (names/aliases + `ArgKind`) while `ex::parse` keeps each command's argument grammar — dispatch on a `CommandId` rather than a full parse rebuild; `:b` gains a **buffer-name** argument (new `GotoBufferName`, vim-correct) so buffer-name completion runs; wildmenu + session-scoped history (cross-session filed as follow-up) | Pending review | `kopitiam-6mx` |
