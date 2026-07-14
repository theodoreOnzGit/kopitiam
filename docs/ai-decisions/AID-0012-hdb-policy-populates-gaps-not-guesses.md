# AID-0012 — HDB policy: an empty table beats a plausible number

**Status:** Pending review
**Date:** 2026-07-14
**Bead:** `kopitiam-6eo` (kopitiam-finance: HDB policy engine)
**Crate:** `crates/kopitiam-finance/src/hdb/policy/`

## Context

`kopitiam-finance::hdb::policy` models HDB's rules — eligibility schemes, income
ceilings, grants, Minimum Occupation Period, Ethnic Integration Policy quotas —
as dated, cited policy tables rather than constants.

The brief was to scaffold the engine and *populate a small, honest, well-cited
slice*, with a standing instruction: **do not guess a figure you are unsure of.**

The engine was built with **no network access**. Every figure in `rules.rs` was
therefore transcribed from an AI agent's recollection of published HDB policy.
No figure was read off an HDB page by the process that wrote it.

That is the decision the maintainer would otherwise have made, and was not here
to make: *how much to populate, and what to do at the edge of confidence.*

## Decision

**Three things, executed.**

### 1. Confidence is encoded in the type system, not in a comment

`Provision<T>` has two variants: `InForce(Dated<T>)` and `NotModelled { effective,
reason, announced_in }`. A span of time where policy certainly applied and this
crate is not confident of the figures is a **first-class row in the table**. A
lookup landing there returns `TemporalError::NotModelled`, which becomes
`Eligibility::Indeterminate` — never a number, never a `false`.

The alternative (leave the row out) was rejected: an absent row is
indistinguishable from "no policy applies", and a caller enumerating the grants
would conclude a resale buyer receives only the EHG. An omission reads exactly
like a fact.

### 2. Every citation says nobody has checked it

`Verification::{Verified { retrieved }, Unverified { note }}`. **Every citation in
the crate is `Unverified`**, and a test asserts it. `HdbPolicy::unverified_provisions()`
returns all of them. When someone fetches the sources, that list shrinks; today
it is the whole crate, and saying so is the crate's first obligation.

### 3. What was populated, and what was refused

**Populated** (confident, with effective dates and citations):

| Rule | Figures |
|---|---|
| Income ceiling, new flat / resale-with-grant, family | $12,000 (from NDR 2015, 24 Aug 2015); $14,000 (from NDR 2019, 11 Sep 2019) |
| Income ceiling, new flat, single | $6,000; then $7,000, on the same two dates |
| Income ceiling, resale without a grant | `NoCeiling` — stated as a *rule*, not an absence |
| Minimum ages | 21 (Public, Fiancé/Fiancée, Non-Citizen Spouse); 35 (Single Singapore Citizen, Joint Singles) — anchored `in_force_at_least_from`, commencement not modelled |
| Minimum Occupation Period | 5 years (pre-framework flats, and Standard); 10 years (Plus, Prime, from the Oct 2024 exercise; PLH, from the Nov 2021 exercise) |
| Ethnic Integration Policy | Chinese 84%/87%, Malay 22%/25%, Indian-Other 12%/15% (neighbourhood/block); SPR quota 5%/8% — anchored at the 5 Mar 2010 revision |
| Enhanced CPF Housing Grant | The **2019** schedule only: $80,000 at ≤$1,500, tapering $5,000 per $500 band to $5,000 at $9,000; singles receive half |
| SPR household resale waiting period | 36 months (from 5 Jul 2013) |

**Refused, and left as declared `NotModelled` spans:**

* **The EHG from 20 August 2024.** Raised at NDR 2024. The revised amounts and
  taper were not confidently recalled. **Consequence: every present-day EHG query
  returns `Indeterminate`.** This is the single most visible cost of the decision,
  and it was taken deliberately — carrying the 2019 figures forward past the day
  they stopped being true would hand a household a number wrong by tens of
  thousands of dollars, with a citation attached to make it look reliable.
* **CPF Housing Grant (Family / Singles) amounts.** Revised in 2019 and again in
  2024; recalled only approximately.
* **Proximity Housing Grant amount.**
* **Resale levy amounts.**
* **The resale-with-grant income ceiling for a single applicant.** The generic
  singles ceiling is very likely the figure; "very likely" is not a standard worth
  meeting for a number a person will act on.

**Not represented at all** — enumerated in the public `rules::UNMODELLED` constant,
which is part of the API precisely so the crate can state its own blind spots:
whether a given block has EIP quota space (the limits are modelled; the live
composition is not, so *"can I buy this flat"* is **always** `Indeterminate`),
income assessment, whether an MOP has been served, flat-type restrictions per
scheme, Executive Condominiums, HDB loans / CPF withdrawal limits, and
prior-property and debarment rules.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Populate everything from recollection, flag the crate as unreliable in the README** | Nobody reads the README. They read the number. A crate-level disclaimer does not survive one `.value` access. |
| **Leave unmodelled spans out of the tables entirely** | An absent row reads as "no policy applies here". The gap must be *stated*, not merely *not asserted*. |
| **Carry the last known figure forward past its supersession** | The precise failure mode this crate exists to prevent. |
| **Model eligibility as `bool` and treat unknowns as `false`** | Tells a household "no" when the truthful answer is "we don't know which rule governs you". |
| **Wait for network access and populate properly** | Would have delivered no engine. The structure is the deliverable; the data is replaceable *because* the structure is right. |

## What would make this wrong

* **If the maintainer wanted a usable HDB calculator now**, this is the wrong
  trade: a present-day EHG query returns `Indeterminate`, which is useless to a
  household even though it is honest. The fix is not to change the design — it is
  to verify the sources and populate the tables, at which point the same design
  answers fully. But if the *intent* was a working calculator rather than a
  knowledge engine, that intent was not served this week.
* **If any populated figure is wrong**, the damage is worse than the gaps, because
  the populated ones look authoritative. The most likely candidates for error, in
  descending order of my own doubt: the $6,000 → $7,000 singles ceiling dates; the
  exact 2015 commencement (24 Aug 2015); the Indian/Other EIP limits (12%/15%) and
  whether they were revised on 5 Mar 2010 or earlier; the PLH and Standard/Plus/Prime
  exercise dates being modelled as calendar dates at all (HDB rules attach to
  *sales exercises*, and the first-of-month anchors are a proxy).
* **If `Verification::Unverified` is treated as a formality** and a downstream UI
  renders these figures without surfacing it, the design has failed at the only
  point that matters. `unverified_provisions()` exists so that cannot be done by
  accident; it can still be done on purpose.

## Follow-up

`kopitiam-6eo` tracks verification of every provision against HDB primary sources.
Until it is closed, no figure in this crate should be put in front of a person.

A second, structural follow-up: **`cpf/` and `hdb/policy/` independently grew the
same temporal model** (see the report on `kopitiam-6eo`). `Date`, `Dated<T>`,
`Citation` and integer-cent `Sgd` exist twice in one crate. They must be hoisted
into a shared `kopitiam-finance::temporal` / `::citation` / `::money`, and the
merge should keep HDB's `Provision::NotModelled` (CPF has no equivalent) and CPF's
`SourceKind` (HDB has no equivalent) — they are orthogonal and both are needed.
