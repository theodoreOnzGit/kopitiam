# AID-0014: `kopitiam-legal` and `kopitiam-insurance` are one engine, and the seam is temporal

**Status:** Pending review
**Bead:** `kopitiam-3zj`
**Date:** 2026-07-14
**Author:** AI (Opus 4.8), working autonomously on `crates/kopitiam-legal`

---

## Context

I was asked to scaffold `crates/kopitiam-legal` and, explicitly, to give a verdict
on a real architectural question:

> Is `kopitiam-legal` the *general* engine that `kopitiam-insurance` should be a
> specialisation of? Or are they genuinely distinct?

`kopitiam-insurance` was being built by a concurrent agent in the same session.
By the time I could read it, it had landed. This AID records what I found, what I
decided, and what would make the decision wrong.

## The decision

**They are one engine.** An insurance policy *is* a legal contract, and the two
crates independently converged on the same machinery — not loosely, but
line-for-line in design.

I did **not** unify them, because I own only `crates/kopitiam-legal` and
unification touches both. I built `kopitiam-legal` so that unification is
*possible later without a rewrite*, and I am recording the seam here so the
maintainer can reconcile them deliberately.

## The evidence: what the two crates converged on independently

Neither agent could see the other's code while designing. We produced:

| Concern | `kopitiam-legal` | `kopitiam-insurance` |
|---|---|---|
| Non-empty verbatim text | `VerbatimText` | `SourceText` |
| 1-based page | `PageNumber(NonZeroUsize)` | `PageNumber(NonZeroUsize)` |
| Document id | `DocumentId` (non-empty) | `DocumentId` (non-empty) |
| Provenance, private fields, one ctor | `Provenance` | `Provenance` |
| Closing the serde back door | `#[serde(try_from)]` shadow | `#[serde(try_from)]` |
| Defined-term override | `Definition` / `Dictionary` | `Definition` / `Definitions` |
| Term lookup outcome | `Resolution::{Defined, Conflicting, NotDefined, ...}` | `Resolution::{Defined, Conflicting, Undefined}` |
| Term occurrences in text | `TermOccurrence` | `TermOccurrence` |
| Cross-references | `CrossReference` / `ReferenceTarget` | `CrossReference` / `ReferenceKind` |
| "I refuse to guess" | `Anomaly` / `AnomalyKind` | `Anomaly` |
| Ontology emission | `to_graph`, `SOURCE` | `to_graph`, `SOURCE` |
| "No advice" disclaimer in crate docs | yes | yes |

This is not a family resemblance. Two independent agents, given the same problem
shape ("extract an operative instrument without misrepresenting it"), built the
same crate twice. That is the strongest possible signal that the abstraction is
real and that we are currently maintaining it twice.

## The seam: each crate has what the other is missing

The convergence is not total, and the *divergences* are the interesting part —
each crate solved a problem the other did not.

### `kopitiam-legal` has temporality; `kopitiam-insurance` does not

This is the important one. `kopitiam-insurance` models **override** but not
**time**:

* `Endorsement::effective_date()` returns `Option<&str>` — an *unparsed string*.
* `PolicyDocument::effective_clause(id)` takes **no as-at date**.

So a policy with two endorsements, one in 2022 and one in 2024, cannot answer
*"what did clause 4.2 say on 2023-06-01?"* — and that is precisely the question a
claim turns on, because a claim is adjudicated against the wording **in force on
the date of loss**. An insurance engine that cannot answer an as-at-date question
has the same defect as a statute engine that cannot: it will confidently return
the current wording to someone asking about a past event.

`kopitiam-legal` models this as its primary interface (`Validity`,
`ProvisionHistory::as_at`, `AsAtResult`'s four outcomes including *NotRecorded*,
and a `#[must_use] TemporalWarning` on the un-dated escape hatch). **An
endorsement is an amendment.** They are the same concept: an instrument that
supersedes a clause of a principal instrument with effect from a date.

### `kopitiam-insurance` has quote-verification; `kopitiam-legal` does not

`Clause::cite(fragment)` checks that the quoted fragment **actually occurs in the
clause it claims to come from**, returning `QuoteNotInClause` otherwise. That is a
genuinely better idea than anything in `kopitiam-legal`: it turns "verbatim" from
a promise into a checked invariant, and it makes a paraphrase or a fabricated
quotation *unrepresentable* rather than merely discouraged. `kopitiam-legal`
should adopt it.

It also has `ExtractedTerm<T>` — a provenance-carrying wrapper with a `map` that
lets a domain crate (`kopitiam-health`) refine a generic extraction into a domain
type *without the value ever escaping its citation*. That is exactly the
mechanism a layered architecture needs, and it is the second thing to lift.

## The recommended architecture

```
kopitiam-legal            <- THE engine for operative instruments.
                             Provenance, verbatim text, provision identity,
                             temporal validity + as-at queries, amendment,
                             definitions + scoped resolution, cross-references,
                             anomalies, ontology emission.

  kopitiam-insurance      <- domain layer: policy/schedule/endorsement/exclusion
                             vocabulary, benefit tables, currency. An Endorsement
                             IS an Amendment; a Clause IS a Provision.

    kopitiam-health       <- domain layer on insurance (already the plan there).

  kopitiam-finance, ...   <- other instrument domains, same base.
```

The base engine owns *everything that is true of any operative instrument*. Domain
crates own *vocabulary and structure peculiar to their document family* — and
those differences are real (a policy has a schedule and a premium; a statute has a
Part and a commencement notification), which is why the domain layers should
exist rather than being folded in.

**Concretely, `kopitiam-insurance` would keep:** `PolicyDocument`, `Endorsement`,
`Exclusion`, `ScheduleValue`, `Currency`, `ClauseRole`, `DocumentClass`.
**And would delete, in favour of `kopitiam-legal`:** `Provenance`, `SourceText`,
`PageNumber`, `DocumentId`, `Definition`, `Definitions`, `Resolution`,
`TermOccurrence`, `CrossReference`, `Anomaly` — about half its current surface.

## Alternatives considered

1. **Leave them separate (status quo).** Two provenance models, two definition
   resolvers, two anomaly taxonomies, maintained for a decade. They will drift.
   The drift will be *silent*, because both will keep passing their own tests. And
   the insurance one is already missing as-at dates — the exact failure this
   duplication invites. **Rejected.**

2. **Make insurance the base and legal the specialisation.** Backwards: a policy
   is a *kind of* contract, which is a *kind of* legal instrument. Insurance has
   no notion of a statute, a Part, an inserted section, or commencement. The
   general case cannot be a specialisation of the specific one. **Rejected.**

3. **Extract a third crate (`kopitiam-instrument`) that both depend on.**
   Defensible, and I would not fight it. It is architecturally cleanest and it
   avoids `kopitiam-insurance` depending on a crate called "legal", which is a
   slightly odd read. But it adds a crate to a workspace that already has 43, for
   a base that is already 90% of `kopitiam-legal`. My preference is to let
   `kopitiam-legal` *be* the base and rename later if the name grates.
   **Deferred to the maintainer** — this is a naming question, not a structural
   one, and the structure is the same either way.

4. **Unify them myself, now.** I own only `crates/kopitiam-legal`; the
   orchestrator was explicit that every other path belongs to a concurrently
   running agent. Editing `kopitiam-insurance` mid-flight would have collided with
   an agent actively writing it. **Correctly out of scope.**

## What would make this decision wrong

* **If insurance policies turn out not to need as-at-date queries in practice.**
  I do not believe this — a claim is assessed against the wording in force on the
  date of loss, which is an as-at query by definition — but if the real workflow is
  always "here is the current policy, tell me what it says today", then the
  temporal machinery is dead weight in insurance and the crates are less alike
  than I claim.

* **If the domain layers turn out to be thicker than the base.** My claim is that
  ~90% of `kopitiam-insurance`'s current provenance/definition/reference code is
  the base engine. If, once real policy wordings are ingested, insurance needs to
  *override* rather than *reuse* those types — e.g. because policy definitions
  have scoping rules genuinely unlike statutory ones — then a shared base becomes
  a straitjacket and two crates are right.

* **If the coupling cost exceeds the duplication cost.** A shared base means a
  change to `Provenance` ripples into insurance, health and finance at once. That
  is usually the *point*, but it is a real cost, and if the domains evolve at very
  different speeds the maintainer may reasonably prefer the duplication.

* **If `kopitiam-legal`'s Commonwealth-statute bias has leaked into the base.**
  I have tried to keep jurisdiction-specific parsing confined to `numbering.rs`,
  but if `Provision` or `ProvisionId` turn out to encode assumptions that a
  contract or a policy cannot satisfy, the base is not as general as I think.

---

## Secondary decisions recorded here

### A Part is context, not identity

`ProvisionId` is **section-rooted**: `s 12(3)(a)` — the Part is *not* in it.
Section numbers run uniquely across a whole Act, and every cross-reference says
"section 7", never "Part I, section 7". Baking the Part into the identity makes
every internal cross-reference dangle. The first version of this crate did exactly
that, and the test suite caught it. `Provision::part` carries the Part alongside,
and `DefinitionScope::Part` is checked against that context rather than by prefix.

### No `Date::today()`

`Date` cannot be constructed from the system clock. "Today" is the most dangerous
default in a legal research tool: it turns a reproducible answer into one that
changes underneath the reader. Callers who want the current date must obtain it
themselves and pass it in, which makes the non-determinism visible at the call
site. This also satisfies CLAUDE.md's determinism requirement without a
dependency on `chrono`/`time`.

### Amendment instructions are recorded, not applied

Real amending Acts are *edit scripts* ("delete 'may', substitute 'must'").
Applying them mechanically produces a plausible consolidated text that nobody
checked. `AmendmentOperation::TextualInstructionNotApplied` records the
instruction verbatim and declines. *"Here is the amendment; the consolidated text
is not derivable by this tool"* is a correct answer.

### Ratio and obiter are never auto-classified

`Holding` defaults to `Unmarked` and can only be set via `Holding::marked_by`,
which **requires the name of the human** who made the call. Classifying ratio vs
obiter is the central contested skill of common-law reasoning; a tool that
auto-labels paragraph [47] as "ratio" is doing law, badly, and someone will cite
it. The synthetic judgment fixture deliberately contains the word "obiter" in a
paragraph, and a test asserts it still comes back `Unmarked`.

### Nothing real was fabricated

No statute, regulation, contract, judgment or case name in this crate corresponds
to any real law of any real jurisdiction. All fixtures are transparently
synthetic (`SYNTHETIC Widget Licensing Act`, `[2099] SYNTH 1`). A plausible-looking
fake section of a *real* Act is genuinely dangerous — someone may rely on it — and
that risk is not worth a nicer-looking test suite. A sweep of 929 PDFs on this
machine found **no** genuine legal instrument, so none was ingested.

## Gap found in the Document Engine

`kopitiam_document::Block` **carries no page number** (`Paragraph { text: String }`;
`Metadata { source_pages: usize }` is only a total count). Page provenance is
mandatory in this crate, so `reconstruct()` cannot be used as-is.

Workaround: call `reconstruct()` **one page at a time**, so the page number is
known by construction. Cost: its cross-page `merge_page_breaks` pass cannot run,
and legal text splits across pages constantly.

**The fix is one field.** If `Block` carried its page, `kopitiam-legal` would
delete its per-page loop and get cross-page paragraph merging *and* page
provenance together. Every provenance-carrying consumer needs this — legal,
insurance, health, finance, literature all cite by page. Recommended as the
highest-value change the Document Engine could make. Filed for the maintainer to
route rather than forked around.
