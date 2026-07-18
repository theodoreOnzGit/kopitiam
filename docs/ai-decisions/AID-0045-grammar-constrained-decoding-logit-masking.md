# AID-0045: grammar-constrained decoding in `kopitiam-runtime` — mask disallowed tokens' logits to `-inf` before sampling, so a small model *cannot* emit invalid structure

* **Status:** Pending review
* **Bead:** `kopitiam-190` (review) · build task `kopitiam-ckv.11` · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §12 (design §10.1 #2, called out there as *THE keystone build item*).
* **Crate:** `kopitiam-runtime` — slots into the front of `sampling.rs`, *before* temperature/top-k/top-p.

## The brief

AID-0044 decided the Termux TUI is Rust-driven with the 0.5B as a constrained oracle. That
whole design rests on one mechanism working: the 0.5B has to be able to emit **structurally
valid** output — JSON tool calls, an allowed tool name, a workspace-relative path, a
renderable choice card — *reliably*, not just often. A small model, left to sample freely,
fumbles structure: it closes a brace early, invents a tool name, dribbles a malformed path.
This AID is the mechanism that removes that failure mode entirely.

## Decision: constrain the *sampler*, not the *prompt* — logit masking at each step

**At every generation step, before sampling picks a token, mask the logits of every token
that the grammar says is not allowed right now to `-inf`.** A masked logit softmaxes to
probability zero, so the sampler *cannot* pick it — regardless of temperature, top-k, or how
confidently the model wanted it. A grammar / JSON schema / allowed-tool-name set defines what
is valid at each position; the mask is recomputed as the decode advances.

```
logits (vocab-sized) ──▶ [ GRAMMAR MASK: disallowed → -inf ]  ◀── AID-0045
                                   │
                                   ▼
                       temperature / top-k / top-p / sample
                                   │
                                   ▼
                            next token (guaranteed valid)
```

**Placement is load-bearing: the mask goes at the *front* of the sampling pipeline**, before
temperature and top-k. Masking *after* temperature scaling would let temperature reshape a
distribution that still contains illegal tokens; masking first means every downstream step
only ever sees the legal sub-distribution. The mask is the first transform, sampling is
everything after.

**Effect.** The 0.5B *physically cannot* produce anything but valid structure. This is not
"we prompt it nicely and validate the output and retry on failure" — there is no invalid
output to retry, because the illegal tokens were never selectable. It turns "fumbles
structure sometimes" into "cannot produce anything but valid structure." It is cheap — just a
vector mask over logits the runtime already has in hand each step — and it is **the** single
feature that makes the small model tool-capable. Without it, AID-0044's constrained-oracle
design doesn't stand up.

**What it does and does not guarantee.** It guarantees *form*, not *sense*. The model still
chooses *which* legal token — so it can still pick the wrong (but well-formed) file, or frame
a valid-but-unwise choice card. Structure is solved deterministically; judgment is not, which
is exactly why AID-0044 routes judgment to the human and AID-0044/0046 route hard reasoning
to a bigger tier. The mask's job is to make the 0.5B's output *always parseable and always
in-schema*, so Rust downstream (tool dispatch, card rendering, path validation) never has to
defend against garbage.

## Alternatives considered

1. **Prompt-and-validate-and-retry.** Ask the model for JSON in the prompt, parse the output,
   reject and re-ask on malformed structure. Rejected as the primary mechanism — on a 0.5B
   the malformed-rate is high enough that you'd loop constantly, burning the phone's scarce
   tokens/sec, and you can still get stuck (the model repeats the same mistake). Masking makes
   the bad output *unrepresentable* instead of *caught after the fact*. (Validation still runs
   as a belt-and-braces check, but it should never fire on structure.)
2. **Fine-tune / use a model trained for tool-calling.** Rejected as insufficient and out of
   scope on-device — even tool-tuned models still occasionally break structure, and we can't
   ship a fine-tune pipeline to a phone. Masking is a *guarantee* a fine-tune can only make
   *more likely*. They compose fine, but the guarantee has to come from the sampler.
3. **Post-hoc repair (parse loosely, fix the JSON).** Rejected — silently repairing a model's
   malformed output re-introduces exactly the "trust what came out of the model's mouth"
   failure AID-0041 forbids. A repaired tool call might not be the call the model meant. Far
   better to make the only expressible calls the valid ones.
4. **Mask after temperature/top-k instead of before.** Rejected — see placement above.
   Temperature must only ever reshape the already-legal sub-distribution.

## What would make this wrong

* **If the grammar/mask machinery is too slow per step to be usable on a phone.** The mask is
  cheap in principle (a vector op), but computing *which* tokens are legal at each step — for
  a real grammar with a stack, not just a flat allow-list — can be non-trivial. If per-step
  grammar advancement dominates the (already slow) 0.5B decode on Termux, the guarantee costs
  more than it's worth. Mitigation: keep the common cases (fixed JSON schema, closed
  tool-name set, path charset) as cheap precomputed masks; reserve full context-free grammars
  for where they're genuinely needed.
* **If tokenization fights the grammar.** Grammars reason over characters/strings; the model
  samples *tokens*, and a single token can straddle a grammar boundary (one BPE token = `":"`
  plus the start of a value). Getting the token-vs-grammar alignment wrong either over-masks
  (blocks legal continuations, model gets stuck) or under-masks (lets something illegal
  through). This is the fiddly part of any constrained-decoding implementation and the most
  likely place for a subtle bug. It must be tested against the actual tokenizer, not assumed.
* **If "valid structure" gets mistaken for "correct output."** The mask guarantees form only.
  If the design ever leans on it for *semantic* correctness — trusts that a well-formed tool
  call is therefore the *right* tool call — that's a misread of what this buys, and AID-0041's
  grounding + AID-0044's human judgment rung are what actually cover correctness.
* **If a task needs structure no practical grammar can express.** Some outputs (free-form
  prose reasoning) genuinely shouldn't be constrained; forcing a grammar there would lobotomise
  the model. The mask is for the *structured* slots (tool calls, cards, paths), applied
  per-slot, not globally over every generation.

## Relationships

* **AID-0044** — the constrained-oracle TUI design this keystone underwrites; specifically
  half-1 technique #2 and the guaranteed-renderable choice card of half 2.
* **AID-0041** — LLMs are grounded, never trusted. Masking is the mechanical expression of
  that at the token level: the model can't even *say* an out-of-schema tool call, let alone
  have Rust trust one.
* **AID-0046** — the model ladder; the 0.5B's role as the "structured tool + choice-card
  emission" reflex tier depends entirely on this mask holding.
* **AID-0047** — persisted decisions can themselves narrow the grammar (a hardened rule can
  fix a slot to a constant), making the mask even tighter over time.
