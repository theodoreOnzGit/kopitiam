# AID-0044: the Termux TUI + agent loop is Rust-driven with the 0.5B model as a constrained oracle — and the human is the judgment rung, not a model-driven agentic loop

* **Status:** Pending review
* **Bead:** `kopitiam-02p` (review) · build tasks `kopitiam-ckv.11` (keystone) + `kopitiam-ckv.13..20`, decide-gate `kopitiam-ckv.21` · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §12. Endorsed direction, not absent-maintainer judgment — but it still needs to land as a proper AID so the reasoning doesn't evaporate into chat.
* **Crates:** `apps/tui` (thin ratatui shell), `kopitiam-workflow` (the only layer that invokes a model), `kopitiam-tools` (new), `kopitiam-ai` (adapters), `kopitiam-runtime` (grammar-constrained decoding — see AID-0045)

## The brief

The goal is a chat TUI, Claude-Code-shape, to talk to the local model with deterministic
tool addons (recursive search, read, write, edit, run), the AID-0040 dispatch ladder
underneath, and optional cloud via API keys. The hard constraint that drives *everything*:
**it must run on Termux / `aarch64-linux-android`**. On a phone the realistic local model
is a **0.5B**. That one fact kills the obvious design.

## Decision: Rust drives, the 0.5B is a constrained oracle, the human is the judgment rung

**A 0.5B local model cannot drive a reliable model-driven agentic tool-loop like Claude
Code.** Structured multi-step function-calling needs a reliability small models simply don't
have — they fumble JSON, invent tool names, lose the thread across steps. So on Android you
do **not** make the loop model-driven. You make it **Rust-driven, with the model as a
constrained oracle at fixed points**, and you move the one thing Rust can't supply — judgment
— to the human who is sitting right there.

The intelligence stack becomes three rungs, none of which is "trust a small model to plan":

> **deterministic tooling (facts) + constrained 0.5B (mechanical work + framing choices) +
> the human (judgment).**

### Half 1 — Rust drives; the model fills slots (five techniques)

Determinism comes from keeping the model *out* of the control flow:

1. **Control flow = named Rust workflows.** `kopitiam-workflow`'s `plan / implement /
   translate / review / verify / document / resume` are deterministic multi-step state
   machines. The model fills reasoning slots; it *never* decides the control flow.
2. **Grammar-constrained decoding (the keystone, its own decision — AID-0045).** Mask
   disallowed tokens' logits to `-inf` before sampling, so the 0.5B *physically cannot*
   emit invalid JSON, a non-existent tool name, or a malformed path. Turns "fumbles
   structure sometimes" into "cannot produce anything but valid structure."
3. **Generation → selection.** Rust enumerates options deterministically (run the search,
   offer N candidate files); the model *picks* rather than *generates*. Ranking is far
   easier for a small model than open planning.
4. **Deterministic verification.** The compiler / tests / LSP diagnostics are the JUDGE of
   "did it work?", not the model. Loop = model proposes edit (constrained) → Rust applies
   (with approval) → Rust runs `cargo build`/tests → fail? feed the *real* error back,
   **bounded** retry (Rust counts) → pass? done.
5. **Rust-gated tool execution.** Every tool call the model emits is validated by Rust
   before running — schema, path-inside-workspace, AID-0043 budget check. A hallucinated or
   oversized call is rejected deterministically, never executed.

### Half 2 — the human is the judgment rung (the keystone the above can't fix)

The one thing constrained decoding + verification can't supply is the 0.5B's weak *judgment*
at genuine forks. **Don't fix it — move it to the human.** The model stops being the decider
and becomes the **option-framer** — exactly how this assistant uses `AskUserQuestion`. No big
model needed for the hard calls: the human is the high-quality reasoner, and on a personal
tool they are right there.

* **Choice-card primitive** (first-class TUI element, own build task): a framed question +
  2–4 options with tradeoffs + a **recommended default** + an "other / free-text" escape.
  Framing options is a narrow structured task the 0.5B can do *reliably* once AID-0045
  guarantees a valid, renderable card. Rust owns the card schema.
* **The human is consulted with `AskUserQuestion` discipline** — only at *genuine* forks,
  always with a recommended default, never spammed with trivia (ask only when the answer
  changes what happens next). This is also the concrete answer to the design's §8.2
  "honest-miss threshold": on a personal tool with the human present, a miss is often best
  resolved by *framing the fork and asking*, not by guessing or burning cloud tokens. (Where
  the line sits between "genuine fork" and "apply a conventional default silently" is the one
  sub-question left open — `kopitiam-ckv.21`, decide-before-building.)
* **Safety = the same primitive.** The write/exec approval gate *is* a human-powered decision
  — the choice card wearing a safety hat. Diff preview before apply (reuse `similar` + the
  LSP-rename preview pattern), and the AID-0043 preemptive budget guard on any tool op.
* **Determinism holds.** Given the same state + the same human choices, the run is
  reproducible — the trace includes the human's decisions as inputs. Replay with the same
  picks → same result.

Human decisions don't stay one-shot: each one is persisted and hardens into a deterministic
rule, so the tool asks *less* over time. That flywheel is its own decision — **AID-0047**.

## Alternatives considered

1. **A model-driven agentic loop (the Claude-Code shape, ported straight to the phone).**
   Rejected — a 0.5B is unreliable at multi-step function-calling. It would fumble JSON,
   hallucinate tool names, and lose the thread across steps; you'd spend all your effort
   babysitting a loop that a bigger model handles for free. On-device we don't *have* the
   bigger model, so we redesign the loop instead of pretending the small one can drive it.
2. **Require a big local model (7B+).** Rejected — it must run on a phone. A 7B won't fit the
   realistic Termux device (AID-0005: rust-analyzer basically won't run there either). Making
   the whole TUI contingent on hardware the target user doesn't have defeats the point. The
   7B is a *bonus rung when it fits* (AID-0046), never a requirement.
3. **No tool use at all — plain chat only.** Rejected as too weak. The value is a workbench
   that can search/read/edit/run. The insight is that you *can* have reliable tool use on a
   0.5B — just not via a model-driven loop. Constrained decoding + human judgment gets you
   there.

## What would make this wrong

* **If constrained decoding + human framing still can't make multi-step tool use reliable
  on a 0.5B.** The whole bet is that (grammar-constrained structure) + (Rust owns control
  flow) + (deterministic verification) + (human owns judgment) together lift a 0.5B to
  *reliable, if shallow-reasoning,* tool use. If in practice the 0.5B can't even frame a
  coherent choice card or pick sanely among enumerated options — if its failure is in the
  *content* of the slots, not just their *structure* — then no amount of masking saves it,
  and the human ends up doing the model's job too, which isn't a tool anymore. The mitigation
  is the model ladder (AID-0046): escalate the *framing* itself to a 7B / cloud when present.
  But on a bare phone with only the 0.5B, this is the load-bearing assumption.
* **If the human-in-the-loop cadence is wrong.** Too many choice cards and it's a nagging
  wizard nobody wants; too few and it silently guesses at forks that mattered. The
  `AskUserQuestion` discipline (ask only when the answer changes what happens next) plus the
  AID-0047 flywheel (ask less over time) are meant to keep it on the right side, but the
  genuine-fork threshold (`kopitiam-ckv.21`) has to be gotten right or the UX sinks.
* **If Termux can't stream tokens smoothly enough for the UI to feel alive.** Phone CPU does
  a few tokens/sec; if the async-actor streaming (AID-0028 template, `kopitiam-ckv.15`) can't
  keep the UI responsive, the "constrained oracle at fixed points" still *works* but *feels*
  dead, and the human abandons it. Streaming is non-negotiable, not polish.

## Relationships

* **AID-0045** — grammar-constrained decoding, the keystone build item that makes half 1 #2
  (and therefore the reliable choice card) possible.
* **AID-0046** — the portable model ladder: the 0.5B here is the permanent fast-reflex tier;
  a 7B / cloud rung, when present, is what half 2's escalation reaches for.
* **AID-0047** — human decisions persist as deterministic rules (the offline flywheel), so
  the judgment rung gets cheaper over time.
* **AID-0040** — the dispatch ladder this loop runs on; the human is added as the *judgment
  rung* on that ladder.
* **AID-0041** — LLM-proposes / Rust-executes: every tool call and fact-query here obeys it.
* **AID-0028** — the async session actor reused for streaming, so `complete()` never freezes
  the phone UI.
* **AID-0005** — why rust-analyzer (and a big local model) basically won't run on Termux, so
  the degraded `kopitiam-syntax` tier is the *default* semantic tier there, not the fallback.
* **Open question:** `kopitiam-ckv.21` (genuine-fork vs silent-default threshold), which
  refines the §8.2 honest-miss question this AID partially resolves.
