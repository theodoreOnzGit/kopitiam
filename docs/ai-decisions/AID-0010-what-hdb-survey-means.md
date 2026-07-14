# AID-0010 — What "HDB survey" was taken to mean

**Status:** Pending review
**Date:** 2026-07-14
**Bead:** `kopitiam-gin` (HDB survey engine)
**Affects:** `crates/kopitiam-finance/src/hdb/survey/`

## The decision

The instruction was to scaffold "the HDB survey engine". The phrase is ambiguous
across three readings, and the code that results is almost entirely disjoint
depending on which is meant. This AID records which was chosen, why, and — since
the choice was **corrected mid-flight by the maintainer** — what the correction
teaches.

### The three readings

1. **Sample Household Survey** — HDB's periodic demographic study of resident
   households: household composition, commuting, dwelling satisfaction. A real
   HDB publication, and the most natural reading of the bare word "survey".
2. **Surveying the resale market** — the price data a buyer needs before
   purchasing a flat: transacted prices, the Resale Price Index, price
   distributions by town and flat type, lease decay, supply and waiting times.
3. **A building or land survey** — a surveyor's physical measurement of a block
   or parcel: settlement, spalling, cadastral boundaries.

### What was decided

**Reading 2 — surveying the resale market for a purchase decision.**

Reading 1 was implemented first, on the strength of the word "survey" alone. The
maintainer corrected it mid-task with four words: *"HDB is for buying house."*
The demographic model was discarded and the domain re-aimed at the buyer's
question before the module was finished.

Reading 3 was never plausible and remains rejected: a structural survey is not a
finance-domain concern and does not belong in `kopitiam-finance` at all.

## Alternatives considered

* **Build all three.** Rejected. They share no domain model beyond the trivial.
  A structural survey has no population, no sample size, and no stratification;
  a demographic survey has no price and no lease. Generalising across them would
  produce an abstraction that fits none of them — precisely the "unnecessary
  abstraction" CLAUDE.md forbids.
* **Build Reading 1 and let the maintainer redirect.** This is what was
  *started*, and the redirect duly came. The cost was low only because the
  correction arrived before the domain model was finished. Had it arrived a day
  later, the whole module would have been wasted.
* **Ask, and stall.** Rejected per Working Practices: make the best judgment,
  execute, and record it. But see "What would make this wrong" — the judgment
  here was in fact wrong, and the mechanism that caught it was the maintainer,
  not the reasoning.

## What survived the correction, and why that matters

The re-aim from demographics to the resale market changed the *domain* entirely —
`Stratum` went from (age band, household size, income decile) to (town, flat
type, storey range, lease band). But it changed **none** of the provenance
architecture:

* `Statistic` still cannot be constructed without its population, period,
  observation count and citation.
* Cross-stratum comparison is still a typed error.
* Series still refuse to join across a methodology change.
* Exact fixed-point arithmetic still replaces `f64` throughout.

This is worth noting as a design result in its own right: **the honesty
constraints were orthogonal to the domain.** They were not demographic-survey
machinery that happened to be reusable; they are what any statistical ingestion
engine needs, and they transferred to a different domain without modification.
That is evidence the constraints were modelled at the right level.

## What would make this wrong

* **If "survey" meant Reading 1 after all** — i.e. the maintainer's "buying
  house" remark was context for *why HDB matters to them*, not a redefinition of
  the task. Then the demographic model is still wanted, and this module answers a
  different question. Mitigation: the provenance core transfers; the strata and
  measures do not. Cost of reversal is moderate, not total.
* **If the buyer question needs affordability, not prices.** This module owns
  *market data* — what flats cost. It deliberately does **not** own eligibility,
  grants, or CPF withdrawal limits; those are rules, and they live in
  `src/cpf.rs` and `src/hdb/policy.rs`. If the maintainer expected this module to
  answer "can I afford this flat?", it does not, by design — it exposes the price
  and the remaining lease so that a layer *above* both can. If that seam is in the
  wrong place, this is the decision to revisit.
* **If small-sample thresholds should come from HDB, not from us.**
  `SMALL_SAMPLE_THRESHOLD = 20` and `LEASE_PROFILE_TOLERANCE_YEARS = 10` are
  **KOPITIAM heuristics, not HDB rules**. They are documented as such at the point
  of definition. HDB and SingStat do apply their own suppression and reliability
  thresholds; those were not available to verify offline, and inventing an
  authoritative-looking number would be exactly the failure this module exists to
  prevent. If a citable source establishes the real thresholds, replace the
  constants and cite them there.

## The lesson worth keeping

The first reading was defensible, well-argued, and wrong. What made it cheap to
fix was that the ambiguity was stated *loudly and up front* — in the crate
rustdoc and in the report — rather than being silently resolved and presented as
obvious. A maintainer can redirect a stated assumption in one sentence. They
cannot redirect an assumption they never saw.

**State the ambiguity. It is what makes being wrong survivable.**
