# AID-0023: kvim attaches the language server when a served file is opened, not when the user first asks for gd/hover

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.26` (follow-up: `kopitiam-cj0.27`, async LSP client)
* **Date:** 2026-07-15
* **Decided by:** AI (Claude), maintainer absent
* **Crate:** `kopitiam-neovim`

## Context

The finisher agent wired kvim's LSP client as **lazily spawned on the first
request**: the `(server, root)` session for a language is created the first time
the user issues a gd / hover / references / rename for a file of that language
(`LspClient::session`, reached through those request methods).

Diagnostics (`cj0.16`) were then polled on the event loop's idle tick by
`App::refresh_diagnostics`, which began with:

```rust
if !self.lsp.is_running(&ft) { return false; }
```

The coordinator's independent verification found the consequence: **a file you
only open and read never shows diagnostics.** rust-analyzer connects fine (gd and
hover were confirmed working on a live server), but because nothing spawns the
server on open, `is_running` is false forever, the guard returns early, the
buffer is never `didOpen`'d, and no `publishDiagnostics` is ever requested or
received. Diagnostics only appeared *after* the user happened to issue some other
LSP action that spawned the server. Reproduced on a real `E0308` crate: no gutter
sign, no virtual text, `]d` a no-op — until a hover was issued, after which the
`E`/`■ mismatched types` rendering appeared correctly.

This is not how a modal editor's LSP is expected to behave. In Neovim the client
attaches on `BufReadPost`/`FileType`, and diagnostics for the file you are
looking at appear without you asking.

## Decision

**Attach the server when a served file is first shown, from the same idle tick
that polls diagnostics.** `refresh_diagnostics` no longer gates on `is_running`;
instead, the first time it sees a file it has not opened, it calls `did_open`
(which lazily spawns the server via `session()`), then polls. Three guards keep
the idle tick cheap and honest:

* `lsp_opened` — announce a present server exactly once, never re-spawn per tick.
* `server_available(ft)` — if no server is registered for the language, or its
  binary is not on `PATH`, degrade silently (no spawn attempt).
* `lsp_no_server` — remember that verdict per file, so an unserved buffer (plain
  text, or a language whose server is not installed) does not rescan `PATH` on
  every idle tick.

Verified on the real binary through a pyte PTY: opening a file with an `E0308`
and issuing **no** keys shows `E` + `■ mismatched types` within ~5 s (rust-
analyzer connect + flycheck), where before it showed nothing indefinitely.

## Alternatives considered

1. **Leave it lazy-on-request (status quo).** Rejected: it makes diagnostics —
   the most passive, always-on LSP feature — silently absent, which reads as
   "diagnostics are broken," not "diagnostics are lazy."
2. **Spawn every server at startup.** Rejected: wasteful, and wrong for a
   multi-language session — you would spawn rust-analyzer for a repo you only
   opened a `.lua` file in.
3. **Attach asynchronously (background thread) on open.** The *right* long-term
   answer, but out of scope here: kvim's LSP client is synchronous today, so the
   connect (and rust-analyzer's `cachePriming` wait, ~3 s even after AID-0022's
   fix) runs on the UI thread. See "What would make this wrong."

## What would make this wrong

* **The synchronous connect stall.** Because the client is synchronous, the
  first served file of a language per workspace freezes the UI for the connect
  duration (~3 s for rust-analyzer after AID-0022; a few hundred ms for
  lua/texlab). Subsequent files of the same language reuse the running
  `(server, root)` session and open instantly, so the cost is one stall per
  server per workspace — judged acceptable. If that stall proves annoying in
  practice, the fix is not to revert this decision but to make the LSP client
  asynchronous (a filed follow-up), after which attach-on-open becomes free.
* If `server_available` were expensive or wrong (e.g. it missed a server
  installed in a non-`PATH` data dir), the `lsp_no_server` cache would strand a
  buffer with no diagnostics for the session. It currently mirrors exactly what
  the spawn path resolves, so the two cannot disagree.
