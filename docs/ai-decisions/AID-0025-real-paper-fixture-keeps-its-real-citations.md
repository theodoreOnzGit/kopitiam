# AID-0025: The real-paper regression fixture keeps its real citations

> **Reversed 2026-07-15:** the maintainer chose a full corpus scrub;
> `real_paper.rs` was removed and this decision no longer stands. The body below
> has been neutralized of the specific paper/author identifiers it originally
> discussed (the maintainer's own work), per the scope refocus; the record of
> the decision and its reversal is preserved.

* **Status:** Reversed
* **Bead:** —
* **Date:** 2026-07-15
* **Decided by:** AI (Claude), maintainer absent

## The premise, and why it did not fully hold

The scope-refocus work (removing engineering-simulation framing from KOPITIAM's
docs and stated goals) included an instruction to swap the domain-specific
example citations in `crates/kopitiam-bibliography/tests/**` for neutral academic
examples, on the understanding that they were synthetic BibTeX parse fixtures.

That was true of exactly one of the two files:

* `tests/roundtrip.rs` contained a *synthetic, fabricated* string-macro fixture.
  It was swapped to a neutral academic example. No ground truth is disturbed; the
  test exercises the same string-macro round-trip.

* `tests/real_paper.rs` was **not** a synthetic fixture. It was a regression test
  pinned against a real, published paper (the maintainer's own genuine academic
  work), a PDF that lived on the maintainer's machine and was never redistributed.
  Every assertion — journal name, volume, issue, pages, the fabricated-author
  "Design" bug, digit-grouped article numbers, theses-printed-as-books — was
  checked *field by field against the printed page*, so its citation strings were
  the actual contents of that document, not invented example data.

## What was originally decided (now reversed)

`tests/real_paper.rs` was left unchanged, on the reasoning that neutralizing its
citation strings would replace ground-truth values with invented ones, making the
assertions assert falsehoods about a real document — which would break the
`#[ignore]`d tests the moment they were run against the real PDF, and quietly
corrupt a provenance record.

## Why it was reversed

The maintainer's intent was that **no** engineering-simulation / domain-specific
string — and, more importantly, none of their own academic work product — should
appear anywhere in the repository, including inside a private, `#[ignore]`d
fixture referencing their own paper, even at the cost of the test no longer
matching a real document. So the keep-decision was wrong for the maintainer's
actual goal. `real_paper.rs` was deleted outright (rather than re-pinned to a
different sample PDF), and the bibliography crate's example corpus was fully
replaced with neutral academic examples.

## Lesson

A "ground-truth regression test" whose ground truth is the maintainer's own
work product is a provenance liability as well as an asset. When separation of a
personal project from that work product is the goal, the test's value does not
outweigh keeping the work product out of the tree — delete or re-pin to a neutral
public document instead.
