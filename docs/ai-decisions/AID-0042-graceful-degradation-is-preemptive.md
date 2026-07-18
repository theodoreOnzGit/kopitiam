# AID-0042: graceful degradation must be PREEMPTIVE, not reactive — the tablet-killers cannot be caught

* **Status:** Pending review
* **Bead:** `kopitiam-fjf` (review) · build tasks `kopitiam-ckv.2` (budgeter) + `kopitiam-8v7.4.1` (oversized-gguf guard) · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §9. Endorsed direction, plus a review finding from the LocalAdapter oversized-gguf check folded in.
* **Crate:** `kopitiam-workflow` (resource budgeter) + `kopitiam-ai` (`LocalAdapter` gguf load)

## The brief

KOPITIAM runs on an Android tablet. The motivating pain is real: rust-analyzer
OOM-kills the tablet when the project is too big for the device. The design
question: how do you degrade *gracefully* when the failure mode is the OS shooting
your process?

## Decision: estimate the cost BEFORE launching the expensive thing, and refuse-or-degrade up front

**What was decided.** `Result` enums handle *ordinary* failures — the kind you can
`?` out of. But the tablet-killers **cannot be caught at all**:

* rust-analyzer eating all RAM → OS **OOM-kill** → `SIGKILL`. No `Result`, no catch,
  no unwind — the kernel already shot the process.
* an oversized gguf → allocator **`SIGABRT`** (this is the LocalAdapter review
  finding, see below).
* a file truncated while mmapped mid-read → **`SIGBUS`**.

You cannot `?` your way out of a process the kernel already killed. **Therefore
graceful degradation = estimate the cost BEFORE launching, and refuse-or-degrade
up front.**

```
cheap capability probe (cores, free RAM)   ── never runs the expensive thing
     +  cheap project-size estimate
     ↓
budget decision:  FULL │ PARTIAL │ SKIP → degraded provider
     ↓
Result enum reports the DECISION, not a caught crash
```

The enum never has to survive a crash, because the crash never happens. The
`Result` here carries a *decision we made on purpose*, not the wreckage of a
failure we tried and lost.

**The resource-aware result type.**

```rust
enum Fetched<T> {
    Ready(T),                       // full answer
    Partial(T, Reason),             // usable but degraded — say why
    Unavailable(Reason),            // couldn't, here's why
}
enum Reason { InsufficientCpu, MemoryBudgetExceeded, Timeout, ProjectTooLarge, NotApplicable }
```

**Resource state is itself a deterministic fact** handed to the model (consistent
with AID-0041 — the runtime tells the model the truth about its own compute). So
the local AI is *told* "you're in reduced mode: `ProjectTooLarge`" and reflects it
honestly to the user ("eh, project too big for full analysis on this device ah,
symbol lookups best-effort only"). Same `Indeterminate` honesty the finance/legal
crates use, applied to compute instead of data.

**The LocalAdapter connection (review finding).** The oversized-gguf `SIGABRT` is
*the same shape of problem* as the rust-analyzer OOM: an uncatchable, preemptive
failure that a `Result` cannot rescue. The LocalAdapter review surfaced that
loading a gguf too big for the device aborts the allocator before any error can be
returned. That is not a separate mechanism — it is client B of the *same* preemptive
budgeter (AID-0043). Filed as `kopitiam-8v7.4.1` against the model-acquisition epic
because it also came out of that review, but it must reuse the shared budgeter, not
grow its own guard.

## Alternatives considered

1. **Reactive degradation — try it, catch the failure, fall back.** The obvious
   engineer's instinct, and correct for *ordinary* errors. **Rejected for the
   tablet-killers** because there is nothing to catch: `SIGKILL`/`SIGABRT`/`SIGBUS`
   do not surface as a `Result`, they end the process. You cannot write a fallback
   arm for an error that never reaches your code. On a device with swap you might
   get away with reactive handling (the OOM killer is lazier); on Android there is
   no swap, so the first big spike is the fatal one.
2. **Branch on machine class ("if tablet, do less").** Rejected here and in the
   context design (§4) — classifying the machine is fragile and the estimate we
   need is *relative* (device capacity × project weight), not a device category.
   A powerful tablet and a weak laptop don't fit neat buckets. The budgeter compares
   an estimated cost to a live budget instead. See AID-0043.
3. **A watcher thread that kills rust-analyzer if RSS crosses a cap
   (measure-not-estimate).** Genuinely complementary and already filed for kvim as a
   runtime cap-and-kill guard (AID-0037's follow-up). But it **cannot prevent the
   first spike**, which on Android is the one that OOM-kills before a watcher reacts.
   The preemptive gate is the first line of defence; the runtime guard is the safety
   net. Not either/or.

## What would make this wrong

* **If the cost estimate is systematically wrong in the dangerous direction.** The
  whole scheme trades accuracy for a hard "never cross the cliff" guarantee. A false
  `SKIP` costs some IDE niceties; a false `FULL` costs a *crashed tablet*. The
  estimate must be biased conservative precisely because the failure is asymmetric
  and uncatchable (that bias is AID-0043's load-bearing rule). If the estimate ever
  runs optimistic — under-counts RAM, ignores proc-macro blow-up, trusts a stale
  `MemAvailable` — the preemptive guard passes a launch that then dies, and we are
  back to the uncatchable crash we built all this to avoid.
* **If a new uncatchable failure appears that the pre-launch probe doesn't model.**
  The probe only defends against costs it estimates (RA RAM, gguf size). A different
  preemptive killer — a huge mmap, an OOM from a provider we forgot to budget — walks
  straight past a guard that doesn't know to estimate it. Every expensive,
  potentially-uncatchable launch must be routed through the budgeter; anything that
  isn't is an unguarded cliff. Mid-reasoning tool-use fetches count too (§4): they
  must be budget-checked, not allowed to OOM in the middle of a model turn.

## Relationships

* **AID-0043** — the *how*: one shared budgeter, the project-size probe, the
  conservative-bias rule. This AID is the *why* (preemptive, not reactive); AID-0043
  is the mechanism.
* **AID-0037** — kvim already ships the pre-start rust-analyzer gate this generalises;
  the runtime cap-and-kill guard filed there is the complementary reactive net.
* **AID-0009** — `kopitiam-syntax` is what `SKIP` degrades *to* (see AID-0043).
* **`kopitiam-8v7.4.1`** — the oversized-gguf preemptive guard (client B).
