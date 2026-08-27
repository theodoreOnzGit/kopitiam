# AID-0056 — The annotation port pins a *second* MuPDF commit (`5fe54ce`), not `19f1284`

**Status:** Pending review
**Date:** 2026-08-27
**Bead:** `bd-01x` (gh-79)

## Context

`kopitiam-pdf` rendered no annotations at all. Diagnosis (recorded in full on
`bd-01x` / gh-79): two independent gaps, neither a mistranslation.

1. Nothing on the render path ever reads `/Annots`. `run_page()` gathers only
   `/Contents`; upstream `fz_run_page` is `pdf_run_page_contents` **+**
   `pdf_run_page_annots`, and only the first half was ported.
2. The triggering file has **no `/AP` anywhere**. Okular writes `/Ink` annots as
   pure data (`/InkList`, `/C`, `/Border`) and expects the viewer to synthesise
   the appearance, as MuPDF and poppler both do.

The maintainer directed that these be **ported from MuPDF** rather than written
from the PDF spec, explicitly to save tokens and avoid debugging a from-scratch
implementation.

Every existing unit in `crates/kopitiam-pdf/src/mupdf/` is pinned at MuPDF
commit **`19f1284`** (AID-0051, `docs/port-ledger.md`, 81 units). That vendored
tree was not present on this machine, so a fresh sparse+shallow clone was taken
— and a shallow clone of `master` yields **HEAD**, which is now `5fe54ce`.

## Decision

**Pin the new annotation units at `5fe54ce` and record the two-commit split
openly, rather than back-dating the vendor tree to `19f1284`.**

Concretely:

* `crates/kopitiam-pdf/vendor/mupdf` is a sparse (`source/pdf`, `include/mupdf`)
  shallow clone at `5fe54ce`, gitignored, and already covered by the crate's
  `exclude = ["vendor/"]` so it never enters the published package.
* The new units — `annot_appearance.rs` (from `pdf-appearance.c`,
  `pdf-annot.c`) and `annot_run.rs` (from `pdf-run.c`, `pdf-annot.c`) — carry
  point-of-use provenance headers naming `5fe54ce` explicitly.
* `docs/ACKNOWLEDGEMENTS.md` and `docs/port-ledger.md` record that the port now
  spans **two** pinned commits, and which units belong to which.

## Alternatives considered

* **Deepen the clone and check out `19f1284` for consistency.** Rejected on
  cost, and the cost was the maintainer's stated priority: two agents were
  already mid-port reading `5fe54ce` when the mismatch surfaced, so switching
  would have invalidated in-flight work and re-spent the tokens the "port,
  don't write from scratch" instruction was meant to save. A single pin is
  tidier, but tidiness is not worth re-doing the work.
* **Silently cite `19f1284` on the new units.** Rejected outright. It is
  cheap, and it is a lie in the provenance record — the exact failure mode
  `CLAUDE.md`'s "a number with no source is a bug that has not fired yet" and
  the Provenance Standards exist to prevent. A future maintainer diffing our
  `annot_run.rs` against `19f1284`'s `pdf-run.c` would find drift with no
  explanation.
* **Write both halves from PDF 32000-1 instead of porting.** Rejected: the
  maintainer explicitly chose porting over from-spec work in this session
  ("I'd rather a port than writing from scratch and debugging, for token
  saving"). Note this *reverses the default* set by AID-0055 for `hayro-font`,
  consistent with the CLAUDE.md amendment that now prefers reuse over
  reinvention.

## What would make this wrong

* **If `pdf-appearance.c` / `pdf-run.c` changed materially between `19f1284`
  and `5fe54ce`** in ways that interact with the already-ported units (shared
  helpers, changed `fz_matrix` conventions, changed `pdf_annot_transform`
  semantics), then two pins in one crate is a real hazard, not just untidiness.
  This was **not** verified — the diff was not read. That is the single largest
  open risk in this decision, and the cheapest thing to check if the maintainer
  wants certainty.
* **If the two-commit split spreads.** One documented exception is
  maintainable; a crate where every unit floats to whatever HEAD happened to be
  on the day it was written is not. If a third pin ever appears, this decision
  should be revisited and the whole tree normalised to one commit.
* **If `vendor/` ever stops being excluded from the package**, an AGPL-3.0 C
  tree would ship inside a published crate. `exclude = ["vendor/"]` in
  `crates/kopitiam-pdf/Cargo.toml` and the repo-wide `.gitignore` `vendor/` rule
  are both load-bearing here.

## Licence note (unchanged, restated because it is load-bearing)

MuPDF is AGPL-3.0 (Artifex). A translation is a derivative work, so these new
units are AGPL-3.0-only exactly as the rest of `src/mupdf/` already is. This
adds no new constraint — `CLAUDE.md` already records that KOPITIAM cannot be
relicensed off AGPL while the MuPDF port is in the tree — but it does add more
ported surface area to that same commitment.
