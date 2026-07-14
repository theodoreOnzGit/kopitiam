# AID-0002: What "finish all of kopitiam" was taken to mean

> **Amendment (2026-07-15):** project scope was later refocused away from
> engineering-simulation domains; see the scope-refocus commit / current
> CLAUDE.md. The engineering-simulation Long-Term Goals referenced below
> (mesh generation, FEM/FVM DSLs, OpenFOAM/MOOSE/OpenMC, scientific
> visualization) are no longer in scope. The body of this AID is left as the
> historical record.

* **Status:** Pending review
* **Bead:** —
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The ambiguity

The maintainer asked to "continue and finish all of kopitiam." Taken
literally this is not achievable in one session, and CLAUDE.md says so
itself: KOPITIAM is framed as a project "actively developed over the next
decade," whose Long-Term Goals list includes OCR, plot digitization,
mesh generation, scientific visualization, FEM/FVM DSLs, MOOSE/OpenMC
integration, and a Neovim-compatible editor. Twenty-one crates in this
workspace are still `cargo new` stubs.

Rather than stop and ask — the maintainer explicitly said to use best
judgment and keep going — I scoped it.

## What was decided

"Finish all of kopitiam" was executed as:

1. **Drive every open bead to done.** That is the concrete, maintainer-authored
   definition of outstanding work that actually exists in this repo, as
   opposed to the aspirational Long-Term Goals list.
2. **Stand up the Kopitiam Runtime through Phase 1** — the newest and largest
   piece of scope the maintainer handed over this session, and the one thing
   blocking KOPITIAM's Offline First promise (today `kopitiam-ai` has only an
   `EchoAdapter`; without local inference, "running out of AI tokens" *does*
   stop work).
3. **Do not** speculatively fill in the other stub crates
   (`kopitiam-mesh`, `kopitiam-ocr`, `kopitiam-plot`, `kopitiam-solver`, ...).
   Each is a real subsystem deserving its own architectural discussion, and
   CLAUDE.md's standing instruction is to *stop and discuss the design first*
   when a request significantly affects architecture. Guessing at twenty
   subsystem designs unattended is exactly what that rule forbids.

## What "done" therefore excludes

The following remain genuinely unfinished, and I did not attempt them:

* Every Long-Term Goal not already tracked by a bead (OCR, equation
  extraction, plot digitization, BibTeX/Typst/LaTeX, mesh generation,
  visualization, Neovim editing, OpenFOAM/MOOSE/OpenMC integration).
* Kopitiam Runtime Phases 2 and 3 (scheduler, KV cache, quantization,
  operator fusion, SIMD, benchmarks) — filed as beads, not built.
* The twenty-one stub crates listed above.

## What would make this wrong

If the maintainer meant "finish" more narrowly — e.g. only the open beads,
without taking on the runtime — then Phase 1 runtime work was premature and
should have waited for a design conversation. I judged the opposite because
the maintainer had *just* handed over a detailed runtime architecture and
asked for it to be scaffolded into the plan, which reads as intent to build
it, not merely to file it.

## Standing constraints observed

* **No `crates.io` publishing.** The maintainer stated this explicitly. Five
  KOPITIAM crates are already live on crates.io from an earlier session;
  `scripts/publish.sh` was not run and no new crate was published. GitHub
  pushes only.
