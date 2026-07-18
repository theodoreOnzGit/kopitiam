# AID-0040: AI system dispatch is a hard-coded Rust try-then-fallback ladder, not a predictive intent classifier

* **Status:** Pending review
* **Bead:** `kopitiam-4bu` (review) · build task `kopitiam-ckv.1` · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §9. Endorsed direction, not an absent-maintainer call — but it still must be written down properly so the reasoning no longer lives in chat.
* **Crate:** `kopitiam-workflow` (system dispatcher)

## The brief

How does KOPITIAM decide, per task, whether to answer from existing knowledge,
from native Rust tooling, from an internet search, from the local LLM, or from
the cloud LLM? This is the **system dispatch** layer — level 2 of the three
dispatches in the design (level 1 is a model-internal MoE detail, out of scope;
level 3 is the determinism boundary, which is AID-0041). We are only deciding
*who runs*, not *what a running model is allowed to know*.

## Decision: hard-code a try-then-fallback ladder; providers self-report an honest miss; the miss is the routing signal

**What was decided.** The dispatcher is **hard-coded in Rust** and lives in
`kopitiam-workflow`. Deterministic, testable, offline, burns no tokens, and is
versioned + reviewable like any other code. Critically: **no AI decides whether
to use AI** — that circularity is exactly the "AI-dependent workflow" `CLAUDE.md`
tells us to avoid.

The shape is a ladder, tried top-down lidat:

```rust
fn dispatch(task: &Task) -> Answer {
    if let Some(a) = knowledge_graph.answer(task) { return a; }   // 1. existing knowledge
    if let Some(a) = native_rust.compute(task)     { return a; }   // 2. native Rust (LSP, cargo, parsers)
    if let Some(a) = internet.research(task)        { return a; }   // 3. internet search (late/optional)
    if let Some(a) = local_model.reason(task)      { return a; }   // 4. local LLM
    cloud_model.reason(task)                                        // 5. cloud LLM (final fallback)
}
```

The dispatcher is deliberately **dumb** — that is the point, not a shortcut. The
intelligence sits in each provider: **every rung self-reports its own coverage**,
returning `None` / `Indeterminate` when it cannot honestly answer, instead of
bluffing. Same discipline the finance/legal crates already use (an empty table
beats a plausible number). A *miss* is the routing signal; you never need a model
to pick the lane.

**The internet-search rung (added on maintainer instruction).** Internet search
sits in the ladder as a first-class knowledge provider, via `kopitiam-web` (the
low-level search adapters) and `kopitiam-internet-research` (the higher-level
research crate, the renamed working name). Its **position** follows Offline-First
strictly: it is a **late/optional** source — tried *after* local existing
knowledge and native Rust, and around/before the cloud LLM — **never the first
reach**. No network, or a search that comes back empty → it reports
`Indeterminate` and the ladder walks past it, same as any other rung. TLS for it
rides `ureq` + `rustls`/`ring` per AID-0013.

**Two guardrails that travel with the decision:**

1. Routing lives in `kopitiam-workflow`, **never** in the CLI/TUI/GUI. Clients
   stay thin; business logic never leaks into an interface.
2. **Persist every escalation.** A task that drops through to the LLM is a gap the
   deterministic layer couldn't cover. Log it. Recurring escalations tell you
   *exactly* which deterministic provider to build next — the ladder teaches you
   where to extend it. This is "every AI interaction leaves behind knowledge"
   applied to the router itself.

## Alternatives considered

1. **A predictive intent classifier** — hard-code (or train) a thing that *guesses*
   up front "is this task answerable deterministically?" and routes on the guess.
   **Rejected.** You lose to phrasing variety forever: natural-language tasks come
   in endless surface forms, and a classifier that predicts answerability will
   mis-route on wording it hasn't seen. The ladder sidesteps the whole problem —
   don't predict, *attempt*. Trying a deterministic rung and taking an honest miss
   is strictly more reliable than guessing whether it would have hit.
2. **Let an AI decide whether to use AI** — ask a model "should I handle this
   deterministically or reason about it?" **Rejected as circular.** You've now
   spent a model call (tokens, latency, a network reach) to decide whether to spend
   a model call, and the deciding model can itself be wrong. It also makes the
   router non-deterministic and untestable. The whole benefit of the ladder is that
   the routing decision costs nothing and is reproducible.
3. **Internet search as an early/primary source.** Rejected on Offline-First
   grounds: reaching the network before consulting local knowledge and native Rust
   would make routine work depend on connectivity and burn the "runs fully offline"
   guarantee. Search earns its place only after the local rungs miss.

## What would make this wrong

* **If providers cannot cheaply and honestly self-report a miss.** The whole design
  rests on each rung returning `None`/`Indeterminate` *cheaply* when it cannot
  answer, rather than either (a) doing expensive work only to discover it can't, or
  (b) bluffing a low-confidence answer that the ladder then trusts. If a
  deterministic provider's "can I answer this?" check is as expensive as actually
  answering, the try-then-fallback shape loses its cost advantage over prediction.
  And if a rung bluffs instead of missing, the ladder stops early on a wrong answer
  and never escalates — silent corruption. The honest-miss contract is load-bearing;
  the exact confidence threshold at which a rung answers vs passes down is itself an
  open question (`kopitiam-ckv.8`, design §8.2) and must be settled before the
  ladder is built.
* **If the escalation log is never mined.** The "log every fall-through" guardrail
  only pays off if someone (or the runtime itself) reads it and builds the missing
  deterministic provider. If it just piles up, the ladder quietly leans harder on
  the LLM over time and the deterministic-first promise erodes without anyone
  noticing.

## Relationships

* **AID-0041** — grounding (LLM proposes, Rust disposes) is the *other* layer; §2
  routing and §3 grounding do not conflict.
* **AID-0013** — `kopitiam-web` search engines + the "rustls is not pure Rust
  (it needs ring)" finding; the internet rung inherits that TLS posture.
* **Epic `kopitiam-ckv`**, build task `kopitiam-ckv.1` (dispatch ladder),
  `kopitiam-ckv.5` (internet-search provider).
