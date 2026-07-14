# AID-0026: advertising `snippetSupport` was required beyond the one stated semantic file, and live mirror/select is scoped out of the first completion-menu cut

* **Status:** Pending review
* **Beads:** `kopitiam-cj0.17` (completion menu), and a filed follow-up for live snippet editing
* **Date:** 2026-07-15
* **Decided by:** AI (Claude), maintainer absent
* **Crates:** `kopitiam-neovim`, `kopitiam-semantic`

## Context

The `cj0.17` brief scoped the completion menu's only out-of-crate change to a
single file: add an `insertTextFormat`/`is_snippet` field to
`kopitiam-semantic/src/lsp_types.rs` and parse it, so an LSP snippet completion
(`insertTextFormat == 2`) is expanded rather than inserted literally. Two
things surfaced while building and PTY-proving it that were genuinely the
maintainer's call.

## Decision 1 — flip `snippetSupport` to `true` (a *second* semantic file)

The parse-side field alone is **necessary but not sufficient**. rust-analyzer
only *sends* `insertTextFormat: 2` (a function as `greet($0)`, a macro as
`println!($0)`) when the client advertises
`completionItem.snippetSupport: true` in `initialize`. kvim's
`kopitiam-semantic` client advertised `snippetSupport: false`
(`lsp_client.rs:358`), so the server stripped every snippet to plain text, the
wire never carried `insertTextFormat: 2`, and the new `is_snippet` flag was
**dead** — deliverable (d) ("an LSP snippet completion expands rather than
inserting `$0` literally") was impossible to satisfy. The PTY test confirmed
this directly: with `false`, accepting `println!` inserted `println!` (no
parens); with `true`, it inserts `println!()` (the `$0` expanded away).

So I flipped it to `true` in `crates/kopitiam-semantic/src/lsp_client.rs` —
**a second semantic file beyond the one the brief named**. I judged this in
scope because the brief's *intent* (prove the LSP-snippet path end-to-end) is
unreachable without it, and the premise that "one field in `lsp_types.rs` is the
whole semantic change" was simply incomplete about how rust-analyzer gates
snippets. This is the "challenge the premise, build what the maintainer actually
wants" rule (per AID-0003/0004).

**Blast radius checked, not assumed.** Advertising `snippetSupport: true` means
the server now returns snippet-syntax `insertText` for *every* callable
completion workspace-wide. Any consumer that inserts a completion's
`insert_text` **verbatim** would now type `$0`/`${1:…}` into its output. I
grepped every consumer of `kopitiam-semantic`'s `completion()` /
`CompletionItem` across `crates/` and `apps/`: the **only** one is kvim, via
`kvim/src/lsp/client.rs` → `ui/app.rs`, which routes `is_snippet` items through
the `kopitiam-snippet` expander (`App::expand_snippet`). No CLI or other app
inserts `insert_text` literally. So the opt-in is safe today.

### What would make Decision 1 wrong

* If a future `kopitiam-semantic` consumer inserts a completion's `insert_text`
  verbatim (a `kopitiam complete` CLI, say), it would emit raw snippet grammar.
  The fix then is per-consumer expansion, not reverting the capability — but it
  is a real obligation this decision creates, and worth the maintainer knowing.
* If the maintainer wants `kopitiam-semantic`'s completion to stay
  "plain text only" as a matter of API contract, the right shape is a
  per-session toggle (snippetSupport advertised only for the kvim client), not a
  global `false`. I judged a global `true` simpler and harmless given the single
  consumer, but a toggle is the cleaner long-term boundary.

## Decision 2 — first cut ships tabstop *navigation*, not live mirror/select editing

The brief asks, for snippets, that after expansion the cursor sits on tabstop 1
(selecting its placeholder), `<Tab>`/`<S-Tab>` move between stops, and mirrored
ranges update together as the user types. I implemented and unit-tested:

* char-offset → grapheme-`Position` mapping of an `Expansion` onto the buffer
  (multi-line, astral-placeholder correct — `ui/snippet.rs`);
* a `SnippetSession` that places every tabstop (mirrors included), reports the
  cursor target and the placeholder range per stop, and navigates forward/back,
  ending past `$0`;
* `<Tab>`/`<S-Tab>` wiring that jumps the cursor between stops.

I **deferred** two interactive behaviours to a filed follow-up: (a) *selecting*
a placeholder so the next keystroke replaces it (needs a select-mode the editor
does not yet have — leaving the cursor at the placeholder start is the honest
degraded behaviour), and (b) *live* propagation of an edit at one stop into its
mirror ranges on every keystroke.

Rationale: the `kopitiam-snippet` engine is still the scaffold **stub** (its
`expand` returns an *empty* tabstop list) until `cj0.28` lands, so a session
does not even start yet in the field, and live mirror/select editing cannot be
end-to-end tested now. Shipping untested live-edit code against a not-yet-real
engine risks subtly-wrong behaviour the coordinator would then have to debug.
The mapping and navigation *are* verified now (pure unit tests over hand-built
`Expansion`s, whose fields are public), so the correct-when-the-engine-lands
core is proven; the interactive editing is the part that genuinely needs the
real engine to verify, so it waits for it.

### What would make Decision 2 wrong

* If the maintainer considers placeholder-select and live mirroring the *whole
  point* of snippets (fair — it is LuaSnip's signature feel), then "navigation
  only" is a thin first cut. The follow-up bead exists precisely so this is a
  scheduled gap, not a silent one; the coordinator re-verifies snippet editing
  after both `cj0.17` and `cj0.28` land.
* If a select-mode is cheap to add on the editor side, placeholder-select could
  ship sooner than assumed — I sized it as "needs new modal state" without
  building it, which may be an overestimate.
