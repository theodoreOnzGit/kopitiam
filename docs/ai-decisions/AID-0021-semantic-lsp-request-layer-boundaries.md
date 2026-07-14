# AID-0021: kopitiam-semantic's LSP request layer — where shapes are handled, where encodings are converted, and how pushed diagnostics are modelled

* **Status:** Pending review
* **Bead:** `kopitiam-yxj`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Scaffold the LSP request layer in `kopitiam-semantic` so that go-to-definition,
> find-references, hover, completion, and diagnostics work — not just rename and
> code-actions. `crates/kopitiam-semantic/**` only; do not touch kvim (two agents
> are editing it) or the four `providers/*` adapters.

kvim's `src/lsp/client.rs` documents, in a table and in the `Unsupported` error
strings of its stubbed `definition`/`references`/`completion`/`note_buffer_changed`
methods, the exact upstream signatures it is waiting for. That table was the spec.

None of the choices below are the maintainer's to make in a strong sense, but
each is a boundary decision that will be hard to move later, so they are recorded.

## Decision 1 — Shape-handling lives in a pure `lsp_types` module, not in the transport

The single hardest part of an LSP client is that one request has several legal
response shapes and a client that handles only one *silently* returns nothing for
the others: `definition` may be a lone `Location`, a `Location[]`, or a
`LocationLink[]`; `completion` a bare `CompletionItem[]` or a `CompletionList`;
`hover` contents a `MarkupContent`, a `MarkedString` (string *or* `{language,value}`),
or an array of those.

I put every parser as a free function over `&serde_json::Value` in a new
`lsp_types.rs`, separate from the `lsp_client.rs` transport. This makes the
shape-handling directly unit-testable with hand-written JSON and a synthetic
line-text source — no spawned server — which is exactly where these bugs hide.
28 unit tests cover the shape and encoding matrix; that is the highest-value part
of the whole change.

For `LocationLink` I resolve to `targetSelectionRange` (the identifier span) with
fallback to `targetRange` (the whole symbol). That is the distinction that makes
go-to-definition land on the *name* rather than the opening brace. The client now
advertises `definition.linkSupport: true` in `initialize` **on purpose**: it
invites the harder response shape rather than hoping rust-analyzer never picks it —
the same "don't be lucky, be correct" stance `crate::position` takes on encodings.

## Decision 2 — Encoding conversion sits at two seams, and positions are `char` offsets end to end

The crate's public contract is Unicode-scalar-value (`char`) offsets everywhere
(see `crate::position`). I honoured that for the new results too:

* The **query** position (`char` → wire) is converted at the `RustAnalyzerSession`
  boundary, which has the queried line's text — identical to how `rename` already
  works.
* The **result** positions (wire → `char`) are converted inside the parsers, which
  take a `line_text(uri, line)` callback because a `definition` result can point
  into a *different* file than the one queried. `LspClient` supplies that callback
  backed by a small disk-reading line cache; the tests supply an in-memory one.

The task text described the low-level `LspClient` methods as "returning `Vec<Location>`
with char-offset ranges" **and** described the conversion as happening "at the
`RustAnalyzerSession` boundary". Those are only superficially in tension: the
*result* conversion is done by the parsers (called from `LspClient`, so its methods
do return char-offset types), while the *query* conversion is done by the session.
No caller ever re-parses JSON or re-does an encoding conversion, which was the
stated goal.

## Decision 3 — Completion returns typed `CompletionItem`, not the `Vec<Value>` kvim's comment guessed

kvim's stub speculated `completion(...) -> Result<Vec<Value>>`. I returned typed
`CompletionItem`s (label, kind as a 25-variant `CompletionItemKind` enum, detail,
documentation, insert_text) instead — a caller should never re-parse JSON, which is
this layer's whole reason to exist. Completion `textEdit` *ranges* are deliberately
**not** exposed: they would need per-item encoding conversion for a replace-range
feature no current caller uses. A caller that needs them should add a field, not
re-parse the raw value.

## Decision 4 — Diagnostics are a *store fed by notifications*, not a request

Every other feature is request/response. Diagnostics are not: the server *pushes*
`textDocument/publishDiagnostics` notifications unprompted, so there is nothing to
"return". I added a `DiagnosticsStore` keyed by URI that `handle_incoming` populates
as notifications stream past, plus a non-blocking `pump_notifications()` that drains
the channel so a caller can poll for fresh diagnostics without issuing a request
(`diagnostics_for` pumps first). The **replace-not-append** semantics are
load-bearing and tested: `publishDiagnostics` always carries the complete current
set, and appending would double every diagnostic on each keystroke the server
re-analyses; an empty publish means "now clean" and clears the file.

## Decision 5 — A pre-existing `dead_code` warning on `LspClient::document_symbols` was silenced, not fixed

`cargo clippy --all-targets` flagged `LspClient::document_symbols` as never used —
verified pre-existing (present at HEAD before this change) because the C++/C#
adapters each roll their own per-server client rather than using the shared
`LspClient`. To meet the "clippy warning-free" gate without deleting deliberately-
public future-adapter API or inventing a fake caller, I added `#[allow(dead_code)]`
with a comment pointing at beads `kopitiam-gjg`/`kopitiam-mfo`. This is the one edit
in the change to code I did not otherwise need to touch.

## What is deliberately NOT done here

The **kvim-side wiring** (editor actions → these `RustAnalyzerSession` methods →
drawing hover/definition/reference results) is a later pass, once the two agents
currently editing `crates/kopitiam-neovim/**` free up `src/`. The upstream now
matches kvim's documented spec table exactly, so wiring each method is a small edit,
not a redesign.

## What would make this wrong

* If a server sends completion `textEdit` ranges that callers genuinely need to
  respect (e.g. replacing a partially-typed prefix), Decision 3's omission would
  surface as slightly wrong insertions; the fix is additive (a range field).
* If a future caller needs diagnostics to update *without* polling, Decision 4's
  poll model would need a callback/subscription instead of a store; the store stays
  correct, it just would not be sufficient on its own.
* `RustAnalyzerSession::definition`/`hover`/`completion` open the document from
  **disk** each call (like `rename`), so they see the last-saved file, not an
  unsaved buffer. `did_open`/`did_change` are exposed for the live-buffer case, but
  a caller that forgets to use them and queries a dirty buffer gets stale answers.
  This mirrors the existing `rename` limitation rather than introducing a new one.
