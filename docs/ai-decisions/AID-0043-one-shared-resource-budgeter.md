# AID-0043: one shared resource budgeter gates both rust-analyzer and gguf loading — and it reuses the sysinfo probe already shipped, not a second /proc reader

* **Status:** Pending review
* **Bead:** `kopitiam-6a5` (review) · build task `kopitiam-ckv.2` (budgeter) · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §9. Includes a **premise-correction** the AI is recording under the challenge-the-premise rule — see the sysinfo reconciliation below.
* **Crate:** `kopitiam-workflow` (or a tiny shared crate both it and the gguf guard depend on) — the resource/capability module

## The brief

AID-0042 decided degradation must be *preemptive*: estimate cost before launching.
This AID is the mechanism. The single decision that stops the tablet dying is a
**cheap** probe — it cannot scan/index the whole project, because that is exactly
the cost we are trying to avoid. The decision is **relative**: device capacity ×
project weight.

## Decision: one budgeter, two clients, a cheap relative probe, biased conservative

**What was decided. ONE budgeter, two clients.** The preemptive guard needed for
the rust-analyzer OOM and the one needed for the oversized-gguf `SIGABRT` are the
*same machinery*. Build one:

```
resource::will_fit(cost_estimate, avail·margin) -> Fits | Degrade(Reason) | Refuse(Reason)
   ├── client A: "should I run rust-analyzer?"   (cost = est_ra_ram)
   └── client B: "should I load this gguf?"      (cost = file_size × materialize_factor)
```

Same `Reason` enum (AID-0042), both preemptive, one code path.

**Input 1 — device capacity (cheap, pure Rust).**

* **Cores:** `std::thread::available_parallelism()`.
* **Free RAM:** the kernel's estimate of what is allocatable *now* — `MemAvailable`
  on Linux/Android, **not** `MemTotal` (free is what OOM cares about, not total).
* **Volatility:** free RAM moves when the user opens other apps. Probe the stable
  bits once per session, but **re-read available memory right before each heavy
  launch** — cheap, and it is the number that actually predicts the kill.

**Input 2 — project weight (cheap proxies, ranked by predictive value).**

1. **Resolved dependency-graph size (primary predictor).** `cargo metadata` is
   cheap (resolves `Cargo.lock`, cached) and gives the full dep closure.
   rust-analyzer's RAM is dominated by **indexing all dependencies**, not your own
   code — so *crate count in the resolved graph* predicts its footprint better than
   source size does.
2. **Stat-only source byte-size (secondary).** Walk `src/`, sum file **metadata**
   sizes — never open a file. O(files), no content reads.
3. ~~LOC / content scan~~ — too expensive, skip. `stat` byte-size is a good-enough
   proxy.

Both cheap, both **deterministic** given `Cargo.lock` + tree.

**The decision + the asymmetric-risk rule.**

```
est_ra_ram ≈ k1·dep_crates + k2·source_MB        // fitted constants, see calibration
budget     = MemAvailable · 0.6                   // headroom: OS + UI + OOM fires early
                     ↓
  est <  budget   → FULL      (rust-analyzer, full index)
  est ≈  budget   → PARTIAL   (workspace-only / no-deps index)
  est >  budget   → SKIP      → degraded provider
```

**Load-bearing rule: bias conservative.** The failure is asymmetric and
uncatchable — a false `SKIP` costs some IDE niceties; a false `FULL` costs a
**crashed tablet** (`SIGKILL`, no recovery). When marginal, **always degrade**. The
goal is not accuracy; it is never crossing the cliff.

**What `SKIP` actually runs.** `SKIP` is not "no semantics". It drops to
`kopitiam-syntax` (hand-written pure-Rust lexers, AID-0009) + `cargo metadata`
facts — symbols/structure at a fraction of the RAM. The AI is told `ProjectTooLarge`
and reports the reduced fidelity honestly (AID-0042).

**Calibration — and the probe learns.** `k1,k2` can't be derived from first
principles, so **measure them**: run rust-analyzer on a handful of known projects,
record **peak RSS**, fit. Then the KOPITIAM move: **after each real run, record the
*actual* peak RSS vs the estimate and refine the constants.** Every session improves
the next prediction — "every AI interaction leaves knowledge" applied to the probe
itself. Keep defaults conservative and thresholds configurable (the maintainer tunes
for their specific tablet). Where the calibration data lives so the loop can read it
across sessions is an open question (`kopitiam-ckv.10`, design §8.4).

## The sysinfo reconciliation (challenge-the-premise)

The design handoff (§6) said: read `/proc/meminfo` for `MemAvailable` by hand,
"no C `sysinfo` crate (keeps Pure Rust Core)." **That premise is wrong, and it is
recorded here rather than silently followed.**

* **`sysinfo` is pure Rust.** It reads `/proc` under the hood on Linux/Android; it
  needs **no C toolchain**, no CMake, no bindgen. It does not violate the Pure Rust
  Core promise — the promise is about the build/dependency chain, and `sysinfo`'s
  chain is pure Cargo. The "no C `sysinfo`" worry was a factual mistake about the
  crate.
* **The maintainer has already adopted it.** `sysinfo` shipped in kvim's
  resource-aware LSP guard (`crates/kopitiam-neovim/src/lsp/resource_guard.rs`,
  **AID-0037**). One dependency there already serves the pre-start gate, the LSP
  startup progress-bar ETA, and a future runtime cap-and-kill guard. Building a
  second, hand-rolled `/proc/meminfo` parser for this budgeter would be a
  *duplicate* reader of the same numbers — less robust (no Android backend handling,
  no cross-platform fallbacks) and inconsistent with a decision already made.

**Therefore: the shared budgeter reuses the sysinfo-based probe, it does not build
a second `/proc` reader.** And note the direction of generalisation:
**AID-0043's budgeter is the generalisation of AID-0037's already-shipped logic** —
AID-0037 is the rust-analyzer-only, kvim-only special case (crate-count +
`.rs`-bytes estimate, `MemAvailable × headroom × core_factor` gate, sysinfo probe).
AID-0043 lifts that same estimate-then-gate shape into `kopitiam-workflow` and adds
a second client (gguf loading) plus the self-improving calibration loop. We are not
inventing a probe; we are hoisting one that already works.

## Alternatives considered

1. **Two separate guards (one for RA, one for gguf).** Rejected — they are the same
   preemptive "will this launch fit the live budget?" question with a different cost
   term. One budgeter with two `cost_estimate` inputs is less code, one `Reason`
   enum, one place to get the conservative-bias rule right.
2. **Hand-rolled `/proc/meminfo` parser (the design's literal instruction).**
   Rejected per the reconciliation above: `sysinfo` is pure Rust, already adopted in
   AID-0037, and more robust across the Android/desktop split. A second reader is
   duplication built on a false premise.
3. **Branch on a device class / machine category.** Rejected — capacity is relative
   (device × project) and volatile (free RAM moves), so a static category mis-fits.
   The live `MemAvailable`-vs-estimate comparison is the honest version.
4. **Optimise the estimate for accuracy.** Explicitly rejected as the wrong
   objective. Accuracy that occasionally over-estimates capacity kills the tablet.
   The objective is *never crossing the cliff*, which means deliberately biasing the
   estimate pessimistic and eating the occasional needless `SKIP`.

## What would make this wrong

* **If the cheap proxies stop predicting the real footprint.** `k1·dep_crates` is a
  blunt average; RA's true RSS blows up with **proc-macro and build-script load**,
  which crate-count doesn't capture (same limitation flagged in AID-0037). A
  proc-macro-heavy graph can be under-estimated (gate passes, tablet still struggles).
  The self-improving calibration is meant to catch systematic bias, but if the bias
  is *input-dependent* (proc-macro-heavy vs leaf-crate-heavy) rather than a constant
  offset, fitting `k1,k2` won't fix it — the model needs a proc-macro term. The fix
  is a better cost model, not scrapping the budgeter.
* **If `PARTIAL` isn't a real rust-analyzer configuration.** The three-way
  FULL/PARTIAL/SKIP decision assumes "workspace-only / no-deps index" is a supported
  RA mode. If it isn't, `PARTIAL` collapses into either FULL or SKIP and the middle
  rung is fiction. This must be verified against RA's real config surface before the
  budgeter commits to it (`kopitiam-ckv.9`, design §8.3).
* **If the calibration data has nowhere durable to live.** The self-improving loop
  needs the (project → peak RSS) points to survive across sessions. If they don't
  persist (open question `kopitiam-ckv.10`), the probe never learns and stays a
  static guess — still safe if biased conservative, but it never gets *better*.
* **If `MemAvailable` is read too early.** It is a snapshot; re-reading it right
  before each heavy launch is load-bearing, not optional. Probe once and cache it and
  a device that came under memory pressure meanwhile gets a stale, optimistic budget.

## Relationships

* **AID-0042** — the *why* (preemptive, not reactive, because the failures are
  uncatchable). This AID is the *how*.
* **AID-0037** — the already-shipped kvim special case this generalises; also the
  source of the sysinfo adoption that corrects the design's `/proc` premise.
* **AID-0009** — `kopitiam-syntax`, the degraded floor `SKIP` drops to.
* **Open questions:** `kopitiam-ckv.9` (PARTIAL RA mode feasibility),
  `kopitiam-ckv.10` (calibration-data provenance) — both decide-before-building.
* **`kopitiam-8v7.4.1`** — client B (gguf load) of this same budgeter.
