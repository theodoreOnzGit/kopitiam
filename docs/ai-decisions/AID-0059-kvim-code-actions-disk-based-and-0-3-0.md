# AID-0059 — kvim code actions: disk-based, lazily resolved, and released as 0.3.0

**Status:** Pending review
**Date:** 2026-10-06 (SGT)
**Issue:** gh-117 / `bd-0tr`

## Context

The maintainer ask for LSP **code actions** in `kopitiam-neovim`'s `LspClient`
(for kovan's code-review fixes, outram-park-backend #740): `textDocument/codeAction`
with diagnostics in context, `codeAction/resolve` where the server needs it, the
resulting `WorkspaceEdit` applied through rename's path, `workspace/executeCommand`
for command-only actions, and a hardcoded `<leader>ca`. Preferences stay compiled
in as data; no Lua.

`kopitiam-semantic` already had a first cut (`RustAnalyzerSession::code_actions`,
`apply_code_action`), but it sent a point range with **empty** context
diagnostics, never resolved, and its client never advertised the `codeAction`
capability at all. Three judgment calls came out of finishing it, made while the
maintainer was not in the loop.

## Decision 1 — the protocol work lives in `kopitiam-semantic`, not kvim

Range requests, context diagnostics (pushed diagnostics on the requested lines,
sent back raw), `triggerKind: Invoked`, the `resolveSupport: ["edit"]` capability,
`codeAction/resolve`, the disabled-action refusal, edit-then-command order and the
"only execute commands the server advertised" guard all went into
`kopitiam-semantic` (`src/code_action.rs`, `session.rs`, `lsp_client.rs`). kvim's
`LspClient` only converts graphemes to chars and forwards. The rules follow the
LSP 3.17 spec and, where it leaves choices open, Neovim 0.12's
`runtime/lua/vim/lsp/buf.lua` (`M.code_action`, `on_code_action_results`),
studied clean-room.

**Consequence:** `kopitiam-semantic` must be released too (0.2.6, additive), and
the `kopitiam code-actions` CLI changes behaviour: rust-analyzer now sends its
assists lazily (only `data`), and `apply_code_action` resolves them on demand —
callers do not change.

## Decision 2 — code actions refuse on a modified buffer

The semantic layer re-opens the file **from disk** before every request (as it
does for rename), and kvim applies edits by writing to disk and reloading the
buffer (rename's path, reused as asked). Offering actions over unsaved text would
either place the edit against stale offsets or reload over the unsaved text. So
`<leader>ca` says "save first" on a modified buffer.

## Decision 3 — `kopitiam-neovim` goes to 0.3.0, not 0.2.6

`config::Action` is a public, exhaustive enum; adding `LspCodeAction` is a
breaking change under Cargo's SemVer reference ("adding a variant to an enum
without `#[non_exhaustive]` is major"), and for a `0.y` crate the minor is the
major. kovan pins `kopitiam-neovim = "0.2.5"`, so it will not pick 0.3.0 up
silently — it has to opt in, which it must anyway to call the new API.
`kopitiam-semantic` only gains items and a derive (`FileEdit: Debug + Clone +
PartialEq + Eq`), so it is 0.2.6. Every other kvim-tree crate holds at 0.2.5
(no commits since its publish), per CLAUDE.md's hold rule; `publish-kvim.sh` was
fixed to resolve each crate's own version via `cargo pkgid`, because its single
`WORKSPACE_VERSION` would have checked `kopitiam-neovim@0.2.5`, found it live,
and skipped the release.

## Alternatives considered

* **Buffer-based code actions** — send the live buffer via `did_change` and
  apply edits to the in-memory buffer. Better UX, but the semantic layer's
  open-from-disk preamble and its disk-reading edit computer would both need a
  buffer-text path; rename has the same limitation, so it is one follow-up for
  both, not a code-action special case.
* **0.2.6 for kvim** — matches the loose habit of earlier 0.2.x releases that
  also added `Action` variants, and kovan would pick it up via caret. Rejected:
  SemVer says breaking, and an exhaustive `match` on `Action` downstream would
  stop compiling on a patch-level update.
* **Mark `Action` `#[non_exhaustive]` now** — would make future variants
  additive, but is itself a breaking change; worth doing in this same 0.3.0 if
  the maintainer agrees (not done here: it is a wider API decision).

## What would make this wrong

* If kvim's consumers (kovan) **want** a caret update to flow without a manifest
  change, 0.2.6 would have been the pragmatic choice — at the cost of SemVer.
* If the maintainer prefers code actions on unsaved buffers to "just work" (auto
  `:w` first, Neovim-style in-memory edits), the refusal is the wrong default.
* If some server needs `context.diagnostics` matched by **column** (not just
  line) to offer its quick-fix, line-overlap filtering under-sends; rust-analyzer
  computes its own fixes and passes the live tests either way.
