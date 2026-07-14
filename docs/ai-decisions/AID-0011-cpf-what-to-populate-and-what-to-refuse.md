# AID-0011: CPF — what to populate, what to refuse, and how to date it

* **Status:** Pending review
* **Bead:** `kopitiam-b1n`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The brief

> Scaffold the CPF policy engine. Never hardcode a policy number as if it were a
> constant of nature. Provenance is mandatory. This crate does not give financial
> advice. Populate honestly and mark clearly what you did not fill in — a
> confidently wrong CPF rate is far worse than an absent one.

Four decisions here were genuinely the maintainer's to make. I made them,
executed them, and record them below with what would make each wrong.

---

## Decision 1: no date library, no decimal library — zero new dependencies

`crates/kopitiam-finance/Cargo.toml` is shared with two other concurrently-running
agents, and I was told not to edit it. That constraint turned out to be a *good*
one, because it forced a question I would otherwise have answered lazily.

**What I did:** hand-rolled `cpf::date::Date` (a proleptic-Gregorian civil date,
~80 lines) and `cpf::money::{Sgd, Rate, Unrounded}` (integer cents and integer
basis points). **No `chrono`, no `jiff`, no `rust_decimal`. No new Cargo.toml
line is needed.**

**Why, beyond the constraint:**

* CPF policy is expressed in **civil dates** — "with effect from 1 January 2025".
  There is no instant, no timezone, no clock. A `DateTime<Utc>` would imply a
  precision the domain does not have and would invite the bug where the same
  policy resolves differently depending on where the machine is.
* There is deliberately **no `Date::today()`**. A policy engine that can read the
  wall clock is a policy engine whose results are not reproducible. This falls
  straight out of CLAUDE.md's determinism requirement, and a date library would
  have handed me a `now()` I would then have had to resist.
* `rust_decimal` would be the wrong shape too. CPF money is *cents* (an integer)
  and CPF rates are *basis points* (an integer). Arbitrary-precision decimal is a
  more general tool than the domain needs, and generality here is surface area.

**What would make this wrong:** if KOPITIAM later needs real calendar arithmetic
(business-day offsets, durations, timezone-aware scheduling) across several
crates, a shared date type belongs in `kopitiam-core`, not re-invented per crate.
The swap is contained — `Date` is opaque and every construction goes through
`Date::new` — but I would rather the maintainer make that call than discover it.

---

## Decision 2: which CPF rules to populate — and, more importantly, which to refuse

This is the decision I most want reviewed, because **understating my confidence
was the instructed error, and I have tried to make that error deliberately.**

### Populated (transcribed from memory; **not** verified against source)

| Rule | Coverage | My confidence |
| --- | --- | --- |
| Contribution rates, Citizen/SPR-3rd-year+, wages ≥ $750/mo | 2024, 2025, 2026 revisions | **Totals: high. Employer/employee split within the 55–70 bands: good, not certain.** This is the figure most worth re-verifying first, and every citation says so in its `note`. |
| Allocation ratios OA/SA/MA | Ages **below 55 only** (4 bands), 2024→ | **High.** Each cross-checks exactly against `x/37` (23/37 = 0.6217, …) and sums to exactly 1.0000. Both asserted in tests. |
| Ordinary Wage ceiling / annual ceiling | 2023 (both halves), 2024, 2025, 2026 | **High.** $6,000 → $6,300 (1 Sep 2023) → $6,800 → $7,400 → $8,000; annual $102,000 throughout. |
| Retirement sums BRS/FRS/ERS | Cohorts turning 55 in 2023, 2024, 2025, 2026 | **High on BRS.** FRS = 2×BRS and ERS = 3×BRS (→2024) / 4×BRS (2025→) are recorded as `SourceKind::Derived` with the multiple stated. |
| Interest **floors** + extra-interest tiers | 2024→ | **High on the floors** (2.5% OA, 4% SA/MA/RA; +1% on first $60k with ≤$20k from OA; +1% more on first $30k at 55+). |

### Deliberately refused — these return a typed error, never a number

1. **Allocation ratios for members aged 55 and above.** *The largest gap.* The
   Special Account was **closed** for these members in January 2025 and their
   savings restructured into the Retirement and Ordinary Accounts. I do not
   confidently know the new ratios, nor the post-55 account structure itself.
   A 57-year-old asking this crate for a contribution gets
   `CpfError::NotPopulated` naming the Special Account closure as the reason.
2. **Graduated rates below $750/month total wages.** The employee's share is
   phased in by a formula depending on both wage *and* age band. Getting this
   wrong under- or over-deducts from **the lowest-paid members**, who can least
   afford it. Not attempted. (I am also not confident whether exactly $50 and
   exactly $500 fall in the lower or upper band — harmless today *precisely
   because* those bands answer nothing. The only sub-$750 boundary that changes
   an answer is $750 itself, which I am confident about and which is tested.)
3. **PR year-1 and year-2 rates.** Three rate tables each (graduated/graduated,
   full/graduated, full/full by joint election), each with its own allocation
   table. Not attempted.
4. **Declared (as opposed to floor) interest rates, and any interest
   computation.** Rates are declared quarterly against a pegged formula and can
   exceed the floor. A half-right interest engine is worse than none.
5. **Retirement sums for cohorts turning 55 from 2027 onward.** A schedule was
   announced. I did not transcribe it. A 2027-cohort query fails rather than
   extrapolating the 3.5%/year trend — and there is a test asserting exactly that
   refusal, because "the trend is obvious" is how a policy engine starts
   inventing policy.
6. **Housing withdrawal limits, CPF LIFE, Workfare, top-ups, Basic Healthcare
   Sum, self-employed, public-sector pensionable.** Out of scope.

**What would make this wrong:** if the maintainer's actual need is a *complete*
CPF calculator, then a crate that refuses to answer for anyone aged 55+ is not
fit for purpose and the gaps must be filled from primary sources. My reading of
the brief is the opposite — "a correct skeleton with 3 cited rules beats a
sprawling one with 300 uncited guesses" — so I optimised for refusing loudly. If
that reading is wrong, the fix is to populate, not to redesign: the shape already
has a slot for every one of these.

**The thing most likely to be actually wrong:** the employer/employee split
within the 2024/2025/2026 senior-worker bands (55–70). I believe the *totals*
(31%/32.5%/34% for 55–60; 22%/23.5%/25% for 60–65; 16.5% for 65–70; 12.5% above
70) more strongly than I believe the split. Verify those first.

---

## Decision 3: allocation rounding order — an assumption, labelled as one

CPF's allocation must be a **partition**: OA + SA + MA must equal the total
contribution exactly, or a member's money is lost or invented. That forces one
account to be the *residual*.

**What I did:** compute MediSave and Special from the ratios (rounded to the
nearest dollar) and give the **Ordinary Account the residual**. The invariant
`allocation.total() == contribution.total` is swept across 2,000 values in a test.

**What I am *not* claiming:** that this is CPF's own order. The residual
*structure* is forced; **which** account is the residual is an assumption I have
**not** verified against the CPF Board's worked examples. It can differ from CPF
by at most one dollar between two accounts of the same member, and never changes
the total.

This is recorded in the rustdoc at the point of use, and — deliberately — emitted
into the knowledge graph as a `GAP:` fact, so it is discoverable without reading
the source.

**What would make this wrong:** CPF documenting a different order. Cheap to fix;
cheap to verify against their public contribution calculator. Worth doing.

---

## Decision 4: two age-band APIs, because there are two questions

CPF does **not** change a member's rate on their birthday. It changes it on the
**first day of the month following** the birthday month. A member born 15 March
1970 is on the ≤55 rates for the whole of March 2025 and moves to the 55–60 rates
on 1 April.

That rule cannot be expressed as a function of an integer age, so I shipped both:

* `contribution_band_on(date_of_birth, month)` — **the correct path**, encoding
  the boundary as a *date* (`nth_birthday(...).start_of_next_month()`), which is
  what makes it impossible to get off by one. Tested on both sides of all four
  contribution thresholds and all six allocation thresholds.
* `ContributionAgeBand::for_age(age)` — the coarse path, which answers "what does
  the table say for a 57-year-old". Documented at length as **not** the right
  answer to "what do I deduct in March".

Shipping only the second would have been the classic bug. Shipping only the first
would have made the common table-inspection query awkward. Shipping both without
saying loudly which is which would have been worse than either.

**Open question I refused to resolve by guessing:** Singapore statute sometimes
applies the common-law rule that a person attains an age at the start of the day
*before* their birthday. If CPF applies it, a member born on the **1st** of a
month changes band a month earlier than this crate says. I implemented the
straightforward reading (anniversary = same day-of-month), documented the
ambiguity where the code lives, and emitted it as a `GAP:` fact. I do not know
the answer and did not pretend to.

---

## What I did not need to decide, but is worth the maintainer knowing

The temporal model is the deliverable, and it is the thing to scrutinise:

* `Dated<T> { value, effective: DateRange, source: Citation }` — the three are
  **inseparable by construction**. A lookup hands back all three, so a caller
  cannot obtain a number while discarding the evidence for it. That is the
  enforcement mechanism behind "the citation is part of the answer".
* `DateRange` is **half-open `[from, until)`**. Adjacent revisions meet at a
  boundary that belongs to exactly one of them, by construction — no
  `2024-12-31` to forget, no silent overlap.
* `PolicyTable::on(date)` is the **only** accessor. There is no `latest()`, and
  that omission is deliberate: "the latest entry in my table" and "the rule in
  force today" differ precisely when the table is stale, which is exactly when
  you most need to be told. A payroll run for March would have quietly picked up
  April's rates.
* `PolicyTable::validate()` rejects overlapping entries but **not gaps** — a gap
  is often the truth, and interpolating across one is the sin the crate exists to
  prevent. Gaps surface as `NoRuleInEffect` at query time, carrying a description
  of what the table *does* cover.
* Every gap is emitted into the knowledge graph as a `Fact` with
  `"populated": false` and a reason. A knowledge base that silently omits what it
  does not know cannot be distinguished from one that has nothing to say.
