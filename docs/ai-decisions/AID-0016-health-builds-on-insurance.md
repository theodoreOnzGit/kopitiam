# AID-0016: `kopitiam-health` builds on `kopitiam-insurance` rather than beside it

**Status:** Pending review
**Date:** 2026-07-14
**Bead:** `kopitiam-bfq`

## Context

`kopitiam-health` and `kopitiam-insurance` were scaffolded **concurrently, by two
different agents**, with no shared API contract frozen up front. The brief for the
health crate said, correctly, that health insurance is a *specialisation* of
insurance-document extraction and not a parallel universe — and that the generic
document machinery (clause extraction, definitions, exclusions, schedules,
provenance) belongs in `kopitiam-insurance`, with `kopitiam-health` owning only the
health domain.

When health-crate work began, `crates/kopitiam-insurance/src/lib.rs` was a
one-line stub. So the health crate initially grew its own `provenance`, `units` and
`ingest` modules, each carrying a rustdoc header saying *"this is generic and belongs
in kopitiam-insurance; lift it down."*

Midway through, `kopitiam-insurance` landed — ~6,000 lines, compiling, with
`Provenance`, `ExtractedTerm<T>`, `Clause`, `ClauseId`, `Definitions`/`Resolution`,
`Money`/`Currency`/`Percentage`, `ingest_pages`/`ingest_pdf` and `to_graph`. That is
precisely the layer the health crate had stubbed.

## Decision

**Delete the stubs and rebuild `kopitiam-health` on `kopitiam-insurance`.**

Concretely:

* `kopitiam-health/src/provenance.rs` — **deleted.** Uses
  `kopitiam_insurance::{Provenance, DocumentId, PageNumber, SectionPath, SourceText,
  ClauseId, ExtractedTerm}`.
* `kopitiam-health/src/ingest.rs` — **deleted.** Uses
  `kopitiam_insurance::{ingest_pages, ingest_pdf}` -> `PolicyDocument`, whose clause
  segmentation, definition extraction and classification the health crate now
  consumes rather than reimplements.
* `kopitiam-health/src/units.rs` — **deleted.** Uses `kopitiam_insurance::{Money,
  Currency, MonetaryAmount, Percentage}`.
* `PolicyTerm` is now a thin wrapper over `ExtractedTerm<TermValue>` plus a health
  `Scope`.
* `PolicyLayer` **retains the `PolicyDocument`**, so definitions resolve through
  `kopitiam-insurance`'s `Resolution` — which reports a policy that defines the same
  word twice, inconsistently, a case the health crate's own lookup could not express.
* `facts_for_policy` **calls** `kopitiam_insurance::to_graph` for the document and
  clause layer and adds health facts on top, rather than emitting a second, parallel
  document graph.
* `Cargo.toml` gains `kopitiam-insurance.workspace = true`. **No workspace change was
  needed** — `kopitiam-insurance` was already declared in `[workspace.dependencies]`.

The one thing deliberately *not* adopted: `kopitiam-health/src/money.rs`, a new
module adding **currency-checked arithmetic** (`Amount`) over
`kopitiam-insurance`'s money types. See "What we kept" below.

## Alternatives considered

1. **Ship the health crate with its own parallel provenance/ingest layer and let the
   maintainer reconcile.** Rejected. It was the fallback the brief authorised *only if
   the insurance crate had not landed*. It had. Handing back a knowing duplicate of a
   crate sitting in the same workspace, for a human to merge by hand, is work created
   rather than work done — and the two provenance models would have diverged further
   with every commit.

2. **Adopt only the types, not the pipeline** (keep a health-specific PDF/clause
   reader). Rejected: it is the specific thing the brief forbade, and
   `kopitiam-insurance`'s segmentation is better than the stub's — it handles clause
   numbering from both headings and paragraphs, tracks page-attributed lines so a
   clause spanning a page break cites the *right* page, and `Clause::cite` verifies a
   quotation against the clause it claims to come from.

3. **Wait for the insurance crate to stabilise.** Rejected: it compiles and its
   public API is coherent. Waiting would have meant shipping the duplicate anyway.

## What we kept, and why

**`kopitiam-health/src/money.rs`** is new, and is arguably generic. It exists because
`kopitiam-insurance` models money exactly right for *extraction* and therefore cannot
do *arithmetic*:

* `MonetaryAmount` pairs an amount with the currency **as printed**, which may be
  `Currency::Ambiguous` (a bare `$`) or `Currency::Unstated`. That is correct and it
  is why the type has no `add`.
* But a cost-share calculator must add, subtract and cap. So `money::Amount` is a
  second, narrower type: an amount whose currency the document *actually stated*. The
  conversion `Amount::try_from_extracted` is the **choke point** where an ambiguous
  currency becomes a refusal instead of an assumption.

This is a health crate today only because no other domain crate has needed
arithmetic yet. **If a second one does, `money.rs` should move down into
`kopitiam-insurance`.**

## Gaps this leaves for the maintainer to reconcile

Things `kopitiam-health` needs that `kopitiam-insurance` does not yet provide, and
which belong down there rather than up here:

1. **Benefit-table scope recovery.** `kopitiam-insurance` models the table
   (`BenefitTable` gives cells with provenance) but nothing maps a cell's *row and
   column headers* onto the scope of the figure in it. Wordings state deductibles as a
   grid of (ward class x age band); until that mapping exists, table-stated figures
   come out unscoped or not at all, and `compute_cost_share` refuses rather than
   mis-scoping them. The refusal is correct behaviour, but the recall loss is real.
2. **Prose amount scanning.** `parse_value` parses a whole schedule *cell*; nothing
   scans a *sentence* for the amounts and percentages embedded in it.
   `kopitiam-health::extract`'s `scan_money` / `scan_percent` / `scan_duration` do
   that and are not health-specific.
3. **Many-clauses-to-one-term extraction.** A rider whose cover is stated across two
   clauses cannot currently be assembled into one term with *both* clauses attached as
   provenance. `ExtractedTerm<T>` holds exactly one `Provenance`. This is generic (a
   motor policy's excess waiver has the same shape).
4. **`ClauseRole` over-application.** The generic classifier assigns roles by section,
   so a clause sitting under a "Definitions" heading inherits `ClauseRole::Definition`
   whether or not it defines anything. `kopitiam-health` therefore does **not** key off
   `clause.role()` — an earlier version did, and it silently dropped the
   integration-mode clause, the single most load-bearing term in the crate. Worth
   fixing in `kopitiam-insurance`, or documenting as advisory-only.

## What would make this decision wrong

* **If `kopitiam-insurance`'s API is still in flux and churns.** The health crate is
  now coupled to `Provenance`, `ExtractedTerm`, `Clause`, `PolicyDocument`,
  `Resolution`, `Money`, `Currency`, `MonetaryAmount` and `Percentage`. Every rename
  down there breaks it. Mitigation: the surface used is small and central, and the two
  crates already agree on the important things (no floats, `Currency::Ambiguous`
  rather than a guessed default, `Resolution::Conflicting` rather than an arbitrary
  pick) — that convergence is evidence the coupling is to the right abstractions.
* **If the maintainer intended `kopitiam-health` to be usable standalone**, without
  pulling in the generic engine. Nothing in the brief suggests that, and the brief in
  fact says the opposite, but it is the assumption that would overturn this.
* **If `Provenance`'s mandatory `PageNumber` proves wrong** for wordings that are not
  paginated (a Markdown or HTML wording scraped from a website). The health crate's
  deleted stub had an `Anchor` enum (`Page(..)` | `Bytes { start, end }`) precisely to
  avoid inventing "page 1" for an unpaginated source. `kopitiam-insurance` requires a
  page. That is fine while every input is a PDF, and it will be a real problem the
  first time one is not — at which point `Anchor` is the shape to reach for, and this
  paragraph is where to find it.
