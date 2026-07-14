# AID-0022: LSP server-readiness detection — how `wait_for_indexing` decides a server is ready to answer

* **Status:** Pending review
* **Bead:** `kopitiam-uab`
* **Date:** 2026-07-15
* **Decided by:** AI (Claude), maintainer absent
* **Crate:** `kopitiam-semantic` (`src/lsp_client.rs`)

## Context

Wiring kvim's `<leader>gd`/`gr`/`rn`/`K` into live language servers surfaced a
latent defect in `kopitiam-semantic`'s connect path: **every** `RustAnalyzerSession::connect`
blocked for ~180 seconds before the first request could go out, making
go-to-definition feel like a hang.

The cause: `wait_for_indexing` waited for a `$/progress` "end" whose *token*
contained the substring `"index"`. Captured live, rust-analyzer's indexing
work-done token is actually `rustAnalyzer/cachePriming` (its human *title* is
`"Indexing"`, but the token is not). The old predicate therefore never matched,
so `wait_for_indexing` fell through to its full `index_timeout` (180 s) on every
start-up — for rust-analyzer *and* for `lua-language-server`/`texlab`, which
never emit that token at all.

## Decision

Readiness is now determined by the server kind, detected from the program name:

* **rust-analyzer** waits for the **precise** end of the substantive indexing
  pass: a `rustAnalyzer/cachePriming` (or a progress whose title is `"Indexing"`)
  `end`, but only after that pass has emitted at least one `report`. The
  report-gating matters because rust-analyzer emits a *trivial* empty
  cachePriming `begin`/`end` cycle as well as the real one; without gating,
  the trivial cycle's `end` would declare readiness before the index exists and
  leave `workspace/symbol` empty. No idle shortcut is used for rust-analyzer:
  on a large workspace it can sit silent for >10 s loading `cargo metadata`
  before its first progress, so any idle window short enough to feel responsive
  would fire during that silence.

* **Every other server** (lua-language-server, texlab, …) is treated as ready
  once it goes **quiet** for a short grace period (3 s) after `initialized`,
  since it emits no rust-analyzer indexing token to key off.

Result: connect drops from 180 s to ~3 s on a small crate, while
`workspace/symbol` on the full 43-crate workspace still waits for the real index
(measured 27 s) and returns complete results.

## Alternatives considered

* **Fix only the token substring** (match `"cachepriming"`/`"index"`/title
  `"Indexing"`) and keep waiting solely for that end. Rejected as insufficient
  on its own: non-rust-analyzer servers still never emit it, so they would keep
  blocking the full timeout. The token fix is *part* of this decision, not all
  of it.
* **A uniform idle heuristic for all servers.** Simplest, and it made
  definition fast — but it raced `workspace/symbol` on the big workspace,
  because rust-analyzer's silent `cargo metadata` startup gap is longer than any
  responsive idle window. Rejected: it traded a correct-but-slow connect for a
  fast-but-sometimes-empty index.
* **A per-server capability/behaviour table** (each server declares how it
  signals readiness). Cleaner long-term, and the likely eventual shape once
  there are many servers, but more machinery than the two cases in front of us
  today. Deferred; noted in the bead.

## What would make this wrong

* If a future rust-analyzer renames or restructures the cachePriming progress
  (or stops emitting `report`s on a trivial crate), rust-analyzer would fall
  through to the full `index_timeout` again. The predicate accepts several
  spellings (`cachepriming`, `index`, title `indexing`) to soften this, but a
  wholesale change upstream would need a new capture.
* Sniffing the server kind from the **program name** (`contains("rust-analyzer")`)
  is a heuristic: a rust-analyzer installed under a different basename would be
  treated as a generic idle-detected server (slower/less precise, but still
  correct). A capability-based detection would be more robust.
* The 3 s idle grace assumes a non-rust-analyzer server that has gone quiet is
  genuinely ready. A server that pauses >3 s mid-initialisation without any
  progress notification would be queried early; its own request timeouts cover
  correctness, but the first answer could be slow or empty until it catches up.
