# AID-0024: the snippet engine is its own crate; the completion menu aggregates LSP + buffer + snippet sources

* **Status:** Pending review
* **Beads:** `kopitiam-cj0.28` (snippet engine), `kopitiam-cj0.17` (completion menu)
* **Date:** 2026-07-15
* **Decided by:** AI (Claude), maintainer absent
* **Crates:** `kopitiam-snippet` (new), `kopitiam-neovim`

## Context

The maintainer asked to complete kvim's completion menu (`cj0.17`) "based on
LSP, text buffer and snippets" — the three sources their Neovim `blink.cmp`
setup aggregates (LSP, buffer words, and `LuaSnip`). Two source of the three
already exist headless in `kvim/src/lsp/completion.rs`: `buffer_words`,
`path_candidates`, and a `merge_and_rank` that fuzzy-filters with `nucleo` and
prefers LSP items on a label collision. LSP items are fetched but not yet fed
into the menu, there is no popup UI, and snippet expansion is explicitly punted
by that module ("Snippet placeholder expansion is a UI/editor concern").

Two decisions had to be made to finish it.

## Decision 1 — snippets are a separate crate, not a kvim module

`kopitiam-snippet` is a new crate owning the LSP snippet grammar: parse a body
(`$1`, `${1:ph}`, `${1|a,b|}`, `$VAR`, escapes, mirrors), expand to literal text,
and report tabstops as char ranges. It is UI-free; the editor consumes an
`Expansion { text, tabstops }` and drives cursor placement / `<Tab>` navigation /
mirrored edits itself.

Rationale:
* It matches KOPITIAM's stated architecture — "everything is a reusable engine;
  applications are clients." A snippet parser is a self-contained engine, useful
  beyond kvim (a future scaffolding or document tool could reuse it).
* The grammar is a *published spec*, so it is a clean-room target with a large,
  UI-free test surface — ideal to build and verify in isolation, in parallel
  with the menu UI, without either agent touching the other's directory (the
  one-owner rule).

Alternative considered: a `snippet` module inside `kopitiam-neovim`. Rejected —
it would bury a reusable engine inside an application crate and force it to be
tested through the editor, and it would make the menu-UI agent and the snippet
agent share `kvim/src`, which the one-directory-one-owner rule forbids.

## Decision 2 — the menu aggregates three sources through the existing engine

The completion menu keeps `merge_and_rank` as the single aggregation point and
adds snippets as a fourth input alongside `lsp_items` / `buffer_items` /
`path_items`. Snippets reach the menu two ways, both handled by the kvim agent
consuming `kopitiam-snippet`:
1. **A snippet source**: built-in snippets surfaced as menu candidates
   (source-tagged), so `fn`, `impl`, … appear while typing.
2. **LSP snippet items**: an LSP `CompletionItem` with `insertTextFormat == 2`
   (Snippet) is expanded on accept rather than inserted literally. This needs a
   one-field extension to `kopitiam-semantic`'s `CompletionItem` to carry the
   `insertTextFormat` flag (the raw JSON already has it; it is just not
   surfaced) — a small change the kvim agent makes, since no other agent touches
   `kopitiam-semantic` now.

Priority ordering (LSP > snippet > buffer > path) is a `CompletionSource`
extension; the exact rank of the snippet source relative to buffer words is the
kvim agent's call, tuned to feel like `blink.cmp`.

## What would make this wrong

* If snippets turn out to want deep editor integration that a UI-free
  `Expansion` cannot express (e.g. dynamic/scripted snippets like `LuaSnip`'s
  function nodes), the crate boundary would leak. The LSP grammar is *static*
  (text + tabstops + variables), so a static `Expansion` covers it; scripted
  snippets are explicitly out of scope and would be a different feature, not a
  reason to fold the crate back into kvim.
* If `insertTextFormat` were not actually present on rust-analyzer's items, the
  LSP-snippet path would be dead. It is a standard LSP field rust-analyzer sets
  on function/method completions; the kvim agent must confirm it round-trips
  (its PTY test expands a real rust-analyzer snippet completion).
* Char-offset tabstops must line up with kvim's grapheme positions. The rest of
  the LSP layer already works in char offsets at its boundary, so this matches;
  a snippet containing astral characters in a *placeholder* is the case to test.
