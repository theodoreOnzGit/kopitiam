# AID-0050: kvim completion goes **smart-case** (via nucleo's `Atom`), not force-lowercase

Status: Pending review
Date: 2026-07-25
Crate: `kopitiam-neovim` (`src/lsp/completion.rs`)
Related: AID-0024 (completion source priority: LSP > Snippet > Buffer > Path),
AID-0003 (what `kvim` is), beads `bd-8sh` (the crash), `bd-5js` (the 0.1.8
publish that ships this)

## Context

kvim was crashing outright — not a wrong ranking, a **panic** — the moment the
maintainer typed a capitalised word in a Markdown file:

```text
nucleo-matcher-0.3.1/src/fuzzy_optimal.rs:37: should have been caught by prefilter
```

Sound like nucleo's bug, but it's ours. `Matcher::fuzzy_match` documents that the
**caller** must hand over a needle already case-folded and unicode-normalised.
`merge_and_rank` was passing the raw typed prefix while `Config::DEFAULT` has
`ignore_case = true`, so nucleo's two halves disagreed:

* `prefilter_ascii` folds case **one way only** — needle `'t'` looks for `'t'` or
  `'T'`, but needle `'T'` looks for `'T'` alone;
* the matrix setup right after it lowercases the **haystack** and compares the
  needle **verbatim**.

Prefilter say "can match", matrix say "cannot", both sides ASCII, so nucleo
asserts its own broken invariant. It needs a *gap* to fire — when the prefilter
window comes back exactly `needle.len()` wide, `fuzzy_match` short-circuits to
`calculate_score` and never touches the matrix. That's why it looked random:
`"Th"` against `"Theorem"` is fine, `"Th"` against `"Through"` (the second `h` in
`-gh`) panics.

Markdown is where it bites because there's no LSP for it in kvim, so `<C-n>`
buffer-word completion **is** the whole completion story there, and prose is full
of Capitalised words. Rust identifiers are mostly lowercase, so this code path
looked healthy for months.

Fixing the crash is not the maintainer's call — that's just a bug. But **the fix
had to pick a case-matching policy**, and that is user-visible editor behaviour
the maintainer would normally choose. Hence this AID.

## Decision

Score through **`nucleo::pattern::Atom`** with `CaseMatching::Smart` +
`Normalization::Smart`, instead of calling `Matcher::fuzzy_match` raw.

Consequence, and the actual decision: **completion is now smart-case.** All-lower
prefix matches case-insensitively (`th` finds `The` and `this`); any uppercase
char in the prefix makes the whole match case-sensitive (`Th` finds `Through`,
not `this`).

`Atom::new` normalises the needle and `Atom::score` sets `ignore_case`/`normalize`
on the matcher to agree with the needle it actually holds — so prefilter and
matrix **cannot** disagree again. It leaves `prefer_prefix` alone, so AID-0024's
autocompletion tuning survives untouched.

## Alternatives considered

1. **`prefix.to_lowercase()` and keep always-case-insensitive.** Smallest diff,
   fixes the panic, preserves old behaviour exactly. Rejected for two reasons:
   it silently re-implements a normalisation nucleo already does properly
   (`to_lowercase` is not unicode case *folding* — ẞ, İ, Turkish dotless ı all
   disagree), and it leaves the same trap armed for the next caller who reaches
   for `Matcher` directly. Going through `Atom` removes the whole class.
2. **`CaseMatching::Ignore`.** Behaviour-preserving *and* correct. Rejected
   because kvim's picker (`plugins/picker.rs`) already uses `CaseMatching::Smart`,
   so Ignore would leave the editor with two different case policies in two
   places the user experiences as "fuzzy matching". One editor, one rule.
3. **`CaseMatching::Respect`.** Too strict for autocomplete — you'd have to
   match the capitalisation of a word you have not finished typing yet.
4. **Pin/patch nucleo, or report upstream and wait.** Not our bug to fix: nucleo
   documents the precondition we broke, and the assert is upstream **correctly**
   catching a caller error. Worth an upstream note that the assert message
   misleads (it reads as an internal invariant, not "you passed a bad needle"),
   but not a blocker.

## What would make this wrong

* **If the maintainer wants completion to stay case-blind.** Smart-case means
  typing `Th` will no longer offer `this`. That's the standard vim/telescope/
  blink expectation and matches kvim's own picker, but it *is* a behaviour change
  and it's the maintainer's editor. Reversal is one argument:
  `CaseMatching::Smart` → `CaseMatching::Ignore` in `merge_and_rank`, plus the
  `merge_and_rank_is_smart_case` test.
* **If smart-case turns out wrong for prose specifically.** Markdown is exactly
  where you type capitals most, and it's also where the candidate pool is buffer
  words rather than symbols. If it feels obstructive when writing English, the
  right answer is probably per-filetype case policy, not going back to blind.
* **If the extra `Atom` allocation ever shows up.** It builds one normalised
  `Utf32String` per `merge_and_rank` call (hoisted out of the item loop, so it's
  once per keystroke, not once per candidate). If completion ever gets called per
  candidate, revisit.
* **If a future nucleo release changes `Atom::score`'s config-mutation
  behaviour.** It currently overwrites `ignore_case`/`normalize` on the passed
  matcher and leaves everything else — `prefer_prefix` depends on that staying
  true. A nucleo upgrade should re-check the
  `merge_and_rank_scores_a_prefix_match_above_a_looser_fuzzy_match` test.

## Knowledge preserved

The full mechanism — one-way prefilter folding, the exact-window short-circuit
that makes it intermittent, and why Markdown surfaced it — is written into the
rustdoc on `merge_and_rank` itself, and the regression test says out loud why
`"Through"` is load-bearing and must not be swapped for a shorter word. Anybody
who reaches for `nucleo::Matcher` directly in this repo again should read it
first: **use the `pattern` API, nucleo's own docs say so, and now we know what it
costs when you don't.**
