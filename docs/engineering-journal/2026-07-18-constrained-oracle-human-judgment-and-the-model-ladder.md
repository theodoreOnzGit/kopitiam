# 2026-07-18 — The constrained-oracle + human-judgment-rung pattern, and the capability-tiered model ladder

*Two more patterns worth preserving from the 2026-07-17/18 hybrid-AI design session
(epic `kopitiam-ckv`), this time from the §10–§12 expansion about the Termux TUI. The
load-bearing *decisions* live in AID-0044..0047; this entry keeps the two *engineering
patterns* — the ideas the next person shouldn't have to re-derive, separate from the
formal decision they became.*

Related: AID-0044 (Rust-driven constrained-oracle TUI, human is the judgment rung),
AID-0045 (grammar-constrained decoding), AID-0046 (portable model ladder), AID-0047
(decision-persistence flywheel), AID-0040 (dispatch ladder), AID-0041 (grounding),
AID-0028 (async session actor), AID-0005 (Android LSP acquisition). Sibling entry:
`2026-07-18-progressive-context-and-self-improving-probe.md`.

---

## 1. The constrained-oracle + human-judgment-rung pattern — how to get reliable tool use out of a 0.5B

### The pain that forced it

The Termux TUI has to run on a phone, and on a phone the realistic local model is a
**0.5B**. A 0.5B **cannot drive a reliable model-driven agentic tool-loop** like Claude
Code — structured multi-step function-calling needs a reliability small models just don't
have. They fumble JSON, invent tool names, lose the thread across steps. The naive answer
("use a bigger model") isn't available: it must run on the phone. So the loop has to be
redesigned around what the small model *can* do reliably.

### The move: don't make the model drive — make it a constrained oracle, and hand judgment to the human

The reframe that unlocks everything: **stop asking the 0.5B to be the agent.** Split the
intelligence into three rungs, each doing only what it's actually good at:

> **deterministic tooling (facts) + constrained 0.5B (mechanical work + framing choices) +
> the human (judgment).**

* **Rust drives the control flow**, not the model. The `plan/implement/translate/...`
  workflows are deterministic state machines; the model only *fills reasoning slots* inside
  them. It never decides what happens next.
* **The model is a constrained oracle at fixed points.** Where it *does* emit structure
  (tool calls, choice cards, paths), grammar-constrained decoding (pattern 2 below, AID-0045)
  makes it *physically unable* to emit anything malformed. "Fumbles structure sometimes"
  becomes "cannot produce anything but valid structure."
* **Generation → selection.** Wherever possible, Rust *enumerates* the options (run the
  search, offer N files) and the model *picks*. Ranking is far easier for a small model than
  open-ended planning.
* **Deterministic verification is the judge.** The compiler / tests / LSP diagnostics decide
  "did it work?", not the model. Propose (constrained) → apply (with approval) → build/test →
  fail? feed the *real* error back, bounded retry → pass? done.

### The keystone insight: the thing the model is *worst* at, you move to the human

Constrained decoding fixes *structure*. It does nothing for *judgment* — the 0.5B is still
weak at the hard "which of these is actually the right call" decisions. **The insight is:
don't try to fix that in the model. Move it to the human, who is right there.** The model
stops being the decider and becomes the **option-framer** — exactly the `AskUserQuestion`
shape this assistant uses. No big model needed for the hard calls; the human is the
high-quality reasoner on a personal tool.

Two things make this actually work rather than being a nag:

* **The `AskUserQuestion` discipline.** Consult the human *only at genuine forks*, always
  with a recommended default, never for trivia — ask only when the answer changes what
  happens next. This is also the concrete answer to the design's long-open "honest-miss
  threshold" (§8.2): on a personal tool with the human present, a miss is often best resolved
  by *framing the fork and asking*, not by guessing or burning cloud tokens.
* **Persist every decision → the flywheel (AID-0047).** Each human choice is recorded
  (decision + rationale) and hardens into a deterministic rule. Next time the same fork
  recurs, Rust applies the recorded decision *without asking*. The judgment rung gets
  *cheaper over time* — the tool asks less as decisions turn into dispatch. "Knowledge
  endures" at the decision level.

### Why this keeps determinism

Given the same state **and the same human choices**, the run is reproducible — the trace
records the human's decisions as inputs. Replay with the same picks → same result. The human
isn't a source of nondeterminism; they're a recorded input, like any other. That's what lets
a human-in-the-loop agent still satisfy the Semantic Runtime's determinism principle.

### The one honest risk to hold onto

The whole bet is that *structure (masked) + control flow (Rust) + verification
(deterministic) + judgment (human)* together lift a 0.5B to **reliable-if-shallow** tool use.
If the 0.5B's failure turns out to be in the *content* of the slots (frames an incoherent
card, picks nonsensically among enumerated options) rather than their *structure*, masking
won't save it and the human ends up doing the model's job too. The escape hatch is the model
ladder (pattern 3): escalate the *framing itself* to a 7B/cloud when one is present. On a
bare phone with only the 0.5B, this is *the* load-bearing assumption — worth watching first
when the TUI is exercised for real.

---

## 2. Grammar-constrained decoding — the cheapest keystone in the whole design

This is written up formally as AID-0045; the pattern worth keeping in the journal is *why
it's the highest-leverage single feature* and *where the bugs hide*.

### The mechanism, in one line

At each generation step, **mask the logits of every currently-illegal token to `-inf` before
sampling.** A masked logit softmaxes to zero, so the sampler cannot pick it, whatever the
temperature. Recompute the mask as the decode advances (JSON schema / grammar / allowed-tool
set says what's legal here).

### Two things that are easy to get wrong

* **Placement.** The mask goes at the **front** of the sampling pipeline, *before*
  temperature/top-k/top-p. Mask *after* temperature and you've let temperature reshape a
  distribution that still contains illegal tokens. Mask first → every downstream step only
  ever sees the legal sub-distribution. (This slots into the front of `sampling.rs` in
  `kopitiam-runtime`.)
* **Token-vs-grammar alignment.** Grammars reason over characters; the model samples
  *tokens*, and one BPE token can straddle a grammar boundary (`":"` + the start of a value
  in a single token). Get this wrong and you either over-mask (block legal continuations, the
  model gets stuck) or under-mask (let something illegal through). This is the fiddly part of
  *every* constrained-decoding implementation — test it against the *actual* tokenizer, never
  assume.

### What it buys and what it doesn't

It guarantees **form, not sense.** The model still picks *which* legal token, so it can still
choose a valid-but-wrong file or frame a well-formed-but-unwise card. That's fine and by
design: structure is solved deterministically here, judgment is solved by the human (pattern
1), reasoning by a bigger tier (pattern 3). The mask's whole job is to make the 0.5B's output
*always parseable and in-schema* so Rust downstream never defends against garbage. Do not
ever let "valid structure" get quietly re-read as "correct output" — that's the trap.

### Why it's the keystone

It's *cheap* (a vector op over logits the runtime already holds each step) and it's what
turns the 0.5B from "unreliable at structure" into "cannot emit bad structure." Every other
piece of the constrained-oracle design — reliable tool calls, the guaranteed-renderable
choice card — stands on this one mask. Highest leverage per line of code in the whole
hybrid-AI design.

---

## 3. The capability-tiered model ladder — one binary, the probe lights up the rungs

Formally AID-0046; the pattern worth preserving is the *shape* and the anti-pattern it dodges.

### The shape

**One ladder, same binary everywhere.** The AID-0043 resource probe — already gating
rust-analyzer and gguf loading — gains a **third client**: model-tier selection ("can I run
the 7B on this hardware too?"). Same `will_fit`, same `Reason` enum, same conservative bias
(a false "yes, load the 7B" on a phone is an uncatchable `SIGABRT`).

```
                       Termux phone     Capable laptop
deterministic (always)      ✅               ✅
0.5B local  (always)        ✅               ✅   ← permanent fast-reflex tier
7B+ local   (if it fits)     ✗               ✅   ← the probe decides
cloud       (if key+net)    ✅ (online)       ✅
```

### The anti-pattern it dodges — do NOT branch on machine class

This is the *same lesson as the context-assembly journal*: don't write
`if phone { 0.5B } else { 7B }`. Machine classification is fragile (a beefy tablet vs a weak
laptop don't bucket cleanly) and it forks the code into paths that drift. Instead ask the
**real** question with the live probe: does *this* model fit *this* device *right now*?
**Capacity is a runtime measurement, not a compile-time label.** One binary, no device
classifier to get wrong — the same principle as "capacity = throughput, not a code path"
from the sibling entry.

### The counter-intuitive bit: the 0.5B is *permanent*, not a fallback

Even on a big machine that can run a 7B, keep the 0.5B loaded (it's ~400 MB, trivial beside a
7B). It earns its place as the **fast-reflex tier**: routing/classification (constrained,
instant), structured tool + choice-card emission, and cheap drafting to compress context for
the 7B.

> **0.5B = reflexes; 7B = deliberation.**

Spending 7B latency on a tool-name classification is waste. The small model isn't "what you
use when nothing better is around" — it's what you *always* use for the fast, structured,
high-frequency work, so the expensive tier only wakes for the genuinely hard cases.

### Composition: cascade first, speculative decoding later

* **Cascade / escalation (start here).** Reuses the AID-0040 ladder *verbatim*, just with
  model rungs: 0.5B answers → deterministic verify → pass? done → honest-miss/fail? escalate
  to 7B → verify → cloud if still short. Starting point *because* it's machinery we already
  committed to.
* **Speculative decoding (phase-N).** 0.5B drafts tokens fast, 7B verifies/corrects in a
  batch → 7B quality faster than plain 7B. Real technique, complex (target model scores the
  draft), a **performance** play — explicitly sequenced *after* cascade works, not the
  starting point. If it never pays off on the target hardware, cascade stands alone and
  nothing upstream depends on it.

### Determinism is device-relative — and that's honest

A given machine always picks the same tier and escalation path → reproducible *per-machine*.
It is **not** reproducible across machines with different hardware — but that's honest: a
phone genuinely can't do what a laptop does. The determinism promise is "same inputs → same
output *on a given device*", and that holds. Don't let anyone re-read it as "identical output
on every device" and write a cross-machine test that can't pass.

---

*Both patterns share a spine with the earlier entries: keep the model out of the control
flow, make capacity a runtime measurement not a code branch, and let accumulated knowledge
(calibration constants there; human decisions here) make the next run cheaper than the last.*
