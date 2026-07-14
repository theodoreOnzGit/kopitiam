# AID-0015 — `kopitiam-insurance` calls the Document Engine one page at a time, and reports its shortfalls rather than forking it

**Status:** Pending review
**Date:** 2026-07-14
**Review bead:** `kopitiam-b1i`
**Work bead:** `kopitiam-hvi.1` (kopitiam-insurance scaffold)
**Crate:** `crates/kopitiam-insurance`
**Upstream bugs found:** `kopitiam-1gb`, `kopitiam-mg3` (both in `kopitiam-document`)

## Context

`kopitiam-insurance` turns insurance documents into provenance-carrying
knowledge. Its governing constraint is that an insurance policy is a **legal
contract**: the crate extracts and locates what a document says, verbatim, with
a citation, and never interprets, advises, or adjudicates. Provenance is
therefore not optional metadata — it is the product, and it must include the
**page**, because a citation a reader cannot turn to in the paper document is
not a citation.

The brief was explicit: build on `kopitiam-document`, do **not** write a second
PDF or table parser, and if the Document Engine falls short for insurance
documents, **report exactly how** rather than forking.

Doing that surfaced a structural mismatch, and three secondary judgment calls.
This AID records all four.

---

## Decision 1 — reconstruct one page at a time (the substantive one)

`kopitiam_document::reconstruct(&[Page]) -> Document` returns a flat
`Vec<Block>` — headings, paragraphs, tables — with **no page attribution on any
block**. `Document::metadata.source_pages` is a count, not a mapping.

For a scientific paper that is right: the reader wants the prose, not the
pagination. For a legal contract it is fatal — it makes the mandatory page
component of `Provenance` unobtainable.

**Decision: call `reconstruct` on a one-page slice, per page**, so every block's
page is knowable by construction, then do the cross-page joining in
`kopitiam-insurance` at the level that matters for a contract — the **clause**.

| Alternative | Why rejected |
|---|---|
| Re-derive page attribution from `kopitiam_pdf::TextSpan` geometry inside `kopitiam-insurance` | This is a second reconstruction pipeline. Explicitly forbidden, and rightly. |
| Add page attribution to `kopitiam_document::Block` now | The correct long-term fix, but `kopitiam-document` is owned by another agent in this session and `Block` is a public type with existing consumers. Not mine to change mid-flight. **This is the fix I am asking for.** |
| Accept a `Document` as `kopitiam-insurance`'s input and cite without a page | Refused. It would make un-sourced extraction *representable*, which is the one thing this crate is built to prevent. |

The trade is genuinely favourable here, not merely tolerable. `reconstruct`'s own
cross-page merge joins two paragraphs when the first does not end a sentence and
the second does not begin one — a good heuristic for prose. A policy clause
routinely **does** end a sentence at a page break and **does** continue with a
fresh capitalised sentence within the same clause, so that heuristic would split
it. Clause numbering says exactly where a clause ends: at the next clause number.
Joining by clause number is both more correct for this document type *and*
page-attributed.

### The cost, stated plainly

Per-page reconstruction means `estimate_body_font_size` is computed **per page**
rather than per document. On a short page (a title page, a one-paragraph
endorsement) the estimate is unstable — and on a tie it is *nondeterministic*,
because `kopitiam-document` breaks the tie via `HashMap` iteration order
(`kopitiam-mg3`). That bug exists whole-document too; per-page calling makes it
easier to hit. If `Block` gains page attribution, this crate reverts to a single
whole-document `reconstruct` call and the cost disappears.

---

## Decision 2 — report the Document Engine's bugs; do not work around them

Two real defects were found and reproduced, and **neither is worked around inside
`kopitiam-insurance`**:

* **`kopitiam-1gb`** — a page whose lines are *all* table rows is misread as a
  two-column text layout and the table is **destroyed**: labels and values land
  in unrelated paragraphs and the pairing is unrecoverable. `split_columns`
  concludes "two columns" whenever spans exist on both sides of the midpoint and
  under 15% of lines straddle the gutter — and a table row's cells *never*
  straddle the gutter, because that is what makes them cells. A policy
  **Schedule** page is exactly a `label | value` table, and a benefit-table
  continuation page is nothing but table rows. These are the pages carrying the
  sums insured, limits, excess and premium.
* **`kopitiam-mg3`** — `reconstruct` is nondeterministic on font-size ties.

Working around either would mean re-implementing table detection or column
splitting here, i.e. forking the parser under a different name. The bugs are
filed with reproductions and suggested fixes; the schedule test fixture carries
a comment saying why it has a prose header line and what would happen without
one.

**What would make this wrong:** if `kopitiam-document` will not be fixed, this
crate is knowingly broken on schedule-only pages, which are common. Then the
right answer is to move table reconstruction *into* `kopitiam-document`'s
pre-column-split phase — still not to fork it.

---

## Decision 3 — money is exact integer cents; percentages are exact basis points; **`$` is not a currency**

No floating point appears anywhere in an extracted value. `0.1 + 0.2 != 0.3` in
binary floating point, and a co-insurance figure a cent away from the contract is
not a rounding artefact in this domain — it is a wrong statement about a legal
obligation. `Money` is `i64` cents; `Percentage` is `i64` basis points; a
percentage needing finer precision than 0.01% is **refused**
(`ScheduleValue::Unparseable`) rather than rounded. `Money::to_f64_lossy` exists
for statistics, and its name says what it costs.

Relatedly, and more consequentially: a bare `$` becomes `Currency::Ambiguous`,
**not** SGD and not USD. At least a dozen countries print their currency as `$`.
`Currency::iso()` returns `None` for it, so a consumer cannot obtain a currency
the document did not supply. Likewise `Nil` is **not** normalised to zero —
whether a nil *excess* and a nil *benefit* mean the same thing is a domain
judgment belonging to `kopitiam-health` / `kopitiam-finance`, not here.

**What would make this wrong:** if consumers find `Currency::Ambiguous` so
tedious that they all write `.unwrap_or(SGD)`, the type has failed at its job and
the guess should be made once, explicitly, with a documented default — not
smuggled into every call site. Worth watching when `kopitiam-health` lands.

---

## Decision 4 — `anyhow` and `tempfile` removed from this crate's manifest

The scaffolded `Cargo.toml` carried `anyhow` and a `tempfile` dev-dependency.
Both were removed.

`anyhow` puts an opaque error type in a library's public API. A caller of an
insurance engine must be able to *match* on what went wrong — a PDF that will not
open is a different problem from a citation that cannot be traced back to its
clause — so `kopitiam_insurance::Error` is a `thiserror` enum. `anyhow` belongs
in the binaries that consume this. `tempfile` is unused: the tests build
synthetic policies as in-memory `kopitiam_pdf::Page` values, because there is no
PDF *writer* in the workspace and adding one to produce a file that would
immediately be parsed back into exactly those spans would be a new dependency for
no test coverage.

Neither removal touches `Cargo.lock`: both crates remain in the tree via other
members. `kopitiam-health`'s manifest was not touched by me (it is another
agent's) — and did not need to be: that agent found this crate mid-session, added
`kopitiam-insurance.workspace = true` themselves, and rebuilt on it (AID-0008,
`kopitiam-bfq`). **No `[workspace.dependencies]` change is required** —
`kopitiam-insurance` was already declared there.

**What would make this wrong:** if the maintainer wants every domain crate's
manifest to look identical for consistency, this is churn. It is a two-line
revert.

---

## What no part of this crate does, and never will

There is no `is_covered()`. Coverage is not a boolean; it is a legal conclusion
about a contract, an event, and the rules of construction a court would apply,
and none of those are in a PDF. Every "I could not determine this" is surfaced as
an `Anomaly` carrying the verbatim text, because a clean-looking wrong answer
about somebody's insurance is worse than an honest "here is the clause, read it".

No real insurer's policy wording is reproduced, quoted, adapted or paraphrased
anywhere in this crate, including its tests. Every clause, definition, exclusion
and schedule figure in the test suite is synthetic. A plausible-looking fake
exclusion clause is not a harmless fixture — it is a statement about somebody's
insurance that no insurer ever made.
