# AID-0046: portable capability-tiered model ladder — one binary, rungs auto-discovered by the resource probe; the 0.5B is a permanent fast-reflex tier; compose by cascade first, speculative decoding later

* **Status:** Pending review
* **Bead:** `kopitiam-nhu` (review) · build task `kopitiam-ckv.18` (model-tier budgeter client) · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §12 (design §11).
* **Crates:** `kopitiam-workflow` (tier selection, escalation), `kopitiam-ai` (adapters), `kopitiam-models` catalog (epic `kopitiam-8v7`), the AID-0043 shared budgeter

## The brief

The Termux TUI (AID-0044) runs on a phone with a 0.5B; a capable laptop has room for a 7B and
maybe cloud. We do **not** want two codebases, or `if android { ... } else { ... }` branching
on machine class — that's the same fragile machine-classification the context-assembly design
(journal 2026-07-18) already rejected. One binary should light up whatever rungs the hardware
can actually carry.

## Decision: one ladder, rungs auto-discovered by the probe; 0.5B permanent; cascade then speculative

**One ladder, same binary everywhere. The AID-0043 resource probe lights up the rungs that
fit this hardware — no per-machine code.**

```
                       Termux phone     Capable laptop
deterministic (always)      ✅               ✅
0.5B local  (always)        ✅               ✅   ← permanent, see below
7B+ local   (if it fits)     ✗               ✅   ← the probe decides
cloud       (if key+net)    ✅ (online)       ✅
```

### 1. The probe gains a THIRD client

AID-0043 built "one budgeter, two clients" (should I run rust-analyzer? should I load this
gguf?). This adds a **third**: *model-tier selection.*

```
resource::will_fit(cost, avail·margin) ->
   ├── run rust-analyzer?     (cost = est_ra_ram)          ← AID-0043 client A
   ├── load this gguf?        (cost = file_size × materialize_factor)  ← client B
   └── can I run the 7B too?  (cost = 7B materialized)      ← NEW, client C
```

Same machinery, same `Reason` enum, same conservative bias (a false "yes, load the 7B" on a
phone is a `SIGABRT`, uncatchable — AID-0042). The `kopitiam-models` catalog (epic
`kopitiam-8v7`) already holds multiple models (0.5B / 3B / 7B …); the probe picks which to
*load* on this hardware. **This is why the acquisition layer mattered** — it's the substrate
the tier selection stands on.

### 2. The 0.5B stays everywhere — a permanent fast-reflex tier, NOT a fallback

Even on a big machine the 0.5B earns its keep (and it's only ~400 MB, trivial beside a 7B):

* **Routing / classification** — constrained-decode (AID-0045), instant.
* **Structured tool + choice-card emission** — AID-0045 / AID-0044's half 2.
* **Drafting** — a cheap first pass to compress context for the 7B.

> **0.5B = reflexes; 7B = deliberation.** The 0.5B is not the thing you use when nothing
> better is available; it is the thing you *always* use for the fast, structured, high-
> frequency work, so the expensive tier only wakes for the hard cases.

### 3. How the tiers compose — cascade first, speculative decoding later

* **Default — cascade / escalation** (reuses the AID-0040 ladder logic verbatim): 0.5B
  answers → deterministic verify (compile / test / confidence) → pass? done (fast) → honest
  miss or fail? escalate to 7B → verify again → cloud if still short. The 0.5B clears the
  easy/frequent stuff; the 7B only wakes for the hard cases. This is the starting point
  because it's *the same escalation ladder we already committed to*, just with model rungs.
* **Advanced (phase-N) — speculative decoding:** the 0.5B *drafts* tokens fast, the 7B
  *verifies/corrects* them in a batch → 7B quality at faster-than-7B speed. A real, known
  technique but complex (the target model scores the draft), and it's a **performance** play,
  not a starting point. Sequenced deliberately *after* cascade works.

### 4. Determinism holds as capability scales

Tier selection is a pure function of hardware + catalog (the deterministic probe); escalation
triggers are deterministic (verification / honest-miss / human choice, per AID-0040/0044). A
given machine always picks the same tier and the same escalation path → **reproducible
per-machine.** (Not reproducible *across* machines with different hardware — but that's
honest: a phone genuinely can't do what a laptop does. The determinism promise is "same
inputs → same output *on a given device*", and that holds.)

## Alternatives considered

1. **Per-machine builds / an Android build vs a desktop build.** Rejected — two artifacts
   drift, and the whole point of "capacity = throughput, not a code path" (context-assembly
   journal) applies here too. One binary, the probe decides what runs.
2. **Branch on a device class (`if phone { 0.5B } else { 7B }`).** Rejected — machine
   classification is fragile (a beefy tablet vs a weak laptop don't bucket cleanly) and it's
   the exact anti-pattern AID-0043 and the context journal already ruled out. The live
   `will_fit(7B_cost, MemAvailable)` check is the honest version: it asks the real question
   (does *this* model fit *this* device *right now*) instead of guessing from a label.
3. **Drop the 0.5B once a 7B is present ("use the best model you have").** Rejected — that
   treats the 0.5B as a fallback, which it isn't. It's the reflex tier: routing, structured
   emission, and drafting are *better* done by the cheap-instant model even when a 7B is
   loaded, because they're high-frequency and don't need deliberation. Spending 7B latency on
   a tool-name classification is waste.
4. **Lead with speculative decoding.** Rejected as premature — it's the harder, riskier
   composition and a performance optimisation. Cascade reuses machinery we already have and
   ships value first; speculative decoding is a later phase on top.

## What would make this wrong

* **If the 0.5B's reflex work isn't actually good enough to keep even when a 7B is loaded.**
  The "permanent tier" claim assumes routing / structured-emission / drafting on a 0.5B are
  *reliable* (leaning on AID-0045 for structure). If in practice the 0.5B's routing decisions
  or drafts are bad enough that the 7B has to redo them, the 0.5B stops saving anything on a
  capable machine and becomes dead weight there (it'd still earn its place on the phone). The
  fix would be to demote it to phone-only, not to change the ladder shape.
* **If `will_fit` mis-sizes a 7B and greenlights a load that `SIGABRT`s.** Client C inherits
  AID-0043's whole risk: the cost estimate (`file_size × materialize_factor`) has to be
  conservative, because a false "yes" is an uncatchable crash. `materialize_factor` for a gguf
  (mmap vs full materialize, KV-cache growth with context length) is exactly the kind of
  constant that needs measuring and the self-improving calibration to refine.
* **If speculative decoding never pays off on the target hardware.** Draft-then-verify only
  wins when the draft acceptance rate is high and the batch-verify is cheaper than plain 7B
  decode. On a memory-starved device the 7B may not even be present, and where it is, the
  speedup is workload-dependent. It's explicitly a later-phase bet; if it doesn't pay, cascade
  still stands on its own and nothing upstream depends on speculative decoding existing.
* **If "reproducible per-machine" quietly gets read as "reproducible everywhere."** The
  determinism here is device-relative by design. Any test or claim that assumes two different
  machines produce identical model output is mis-reading this decision.

## Relationships

* **AID-0043** — the shared budgeter this adds a third client to; same probe, same `Reason`
  enum, same conservative-bias rule.
* **AID-0044** — the constrained-oracle TUI; the 0.5B fast-reflex tier is its default model,
  and escalation-to-7B/cloud is how its judgment/reasoning rung reaches for more when present.
* **AID-0045** — grammar-constrained decoding; the 0.5B's reflex roles (routing, structured
  emission) depend on it holding.
* **AID-0040** — the dispatch ladder whose escalation logic the cascade reuses verbatim, now
  with model rungs.
* **Epic `kopitiam-8v7`** — the model-acquisition/catalog layer that holds the 0.5B/3B/7B and
  is the substrate the probe selects from; this is the payoff that made that layer matter.
