# AID-0047: human decisions are persisted as deterministic rules — the offline decision flywheel, so the tool asks less over time

* **Status:** Pending review
* **Bead:** `kopitiam-h0l` (review) · build task `kopitiam-ckv.19` (flywheel) · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §12 (design §10.2).
* **Crates:** `kopitiam-workflow` (the decision-elicit + rule-apply step), `kopitiam-index` (durable rule store), `kopitiam-tools` / `apps/tui` (the choice card that raises the fork)

## The brief

AID-0044 made the human the **judgment rung**: at a genuine fork the constrained 0.5B frames a
choice card, the human picks. That's correct but, left alone, it's *forgetful* — the tool
would re-ask the same fork every time it recurs, and a personal workbench that nags you with
the same question weekly is a bad tool. The design's §10.2 answer: **don't let the human's
decision evaporate — persist it, and apply it deterministically next time.**

## Decision: each human decision hardens into a deterministic rule; the ask-rung gets cheaper over time

**Every choice the human powers is recorded** — the decision *and* its rationale — as durable,
first-class knowledge, in the same spirit as an AID / a preference / a translation rule.
**Next time the same fork appears, the deterministic layer applies the recorded decision
without asking again.**

```
fork appears
   │
   ▼
is there a persisted rule matching this fork?  ── yes ──▶ apply deterministically, don't ask
   │ no
   ▼
0.5B frames a choice card (AID-0044/0045) ──▶ human picks ──▶ execute
   │
   ▼
persist { fork-signature → decision + rationale }  ── the flywheel turns
```

This is the **offline decision flywheel**: the tool asks less as decisions harden into rules.
The "ask the human" rung (AID-0044) gets *cheaper* over time — not because the model got
smarter, but because accumulated human judgment turns into deterministic dispatch. It's
"Knowledge endures" (Core Philosophy) applied at the **decision** level: the reasoning behind a
choice is captured once and reused forever, instead of being re-litigated each session or lost
to chat history.

**Why this is the honest place for the knowledge to live.** A persisted decision is
deterministic dispatch, not a model memory — it sits in `kopitiam-index` as data the runtime
owns, consistent with "the runtime owns understanding; models borrow it" (AID-0041). The model
never becomes the store of what the human decided; Rust does. When a rule fires, no model runs
at all — it's a pure lookup, offline, zero tokens. That's why it's a *flywheel* and not just a
cache: each turn permanently lowers the cost of the next.

**The rationale is not optional.** Record *why*, not just *what*. A bare "chose option B"
can't be reviewed, reversed, or generalised later; the rationale is what lets the maintainer
audit the accumulated rules and what lets a later fork recognise "this is the same *kind* of
decision." (Same reason an AID records reasoning, not just the verdict.)

**Reversibility.** A hardened rule must be inspectable and revocable — the maintainer has to be
able to list "what has the tool decided to stop asking me about" and undo any of it. A rule
that silently entrenches a wrong early choice is worse than asking again. Persisted ≠ permanent;
persisted = *default until the human changes it.*

## The open sub-question this doesn't close

*When* is a fork a "genuine fork" worth a choice card (and therefore worth persisting), versus
something the tool should just apply a conventional default for silently? That threshold — the
same `AskUserQuestion` discipline — is left as a decide-before-building question
(`kopitiam-ckv.21`, design §12's partial resolution of §8.2). This AID decides *what happens to
a decision once made*; ckv.21 decides *which forks rise to a decision at all*. They're
adjacent but separable, and getting ckv.21 wrong would either over-persist trivia or under-ask
on things that mattered.

## Alternatives considered

1. **Don't persist — re-ask every time.** Rejected — a personal tool that asks the same
   question every session is a nag, and it throws away the human judgment that's the most
   expensive input in the whole stack (AID-0044). The judgment rung has to get cheaper over
   time or it doesn't scale to daily use.
2. **Let the model "remember" past decisions (stuff them in context / a fine-tune).** Rejected
   — that makes the *model* the store of truth about what the human decided, which is exactly
   the trust-the-model failure AID-0041 forbids. A model recalling a past preference
   *probabilistically* can misremember it; a deterministic rule can't. Persisted decisions are
   Rust-owned data, applied by Rust, not model memory.
3. **Persist the *what* only, drop the rationale.** Rejected — an un-rationalised rule can't be
   audited or safely generalised, and the maintainer can't tell a good hardened default from a
   bad one when reviewing. Cheaper to store, far more expensive to trust.
4. **Make hardened rules permanent / irreversible.** Rejected — an early wrong choice would
   entrench silently and the tool would confidently stop asking about the exact thing it's
   getting wrong. Rules are defaults-until-changed, listable and revocable.

## What would make this wrong

* **If "the same fork" can't be identified reliably.** The flywheel needs a stable
  *fork-signature* to match a new situation against a persisted decision. If the signature is
  too coarse it fires a past decision on a genuinely different fork (silently doing the wrong
  thing without asking); too fine and it never matches, so nothing is ever reused and the
  flywheel doesn't turn. Defining that signature is the real engineering risk here, and it's
  entangled with ckv.21's "what counts as the same genuine fork" question.
* **If entrenchment beats the human.** If revocation is clumsy or the accumulated rules aren't
  easily inspectable, the tool drifts toward confidently applying stale decisions the human has
  since changed their mind about — worse than a forgetful tool. Listability + easy undo are
  load-bearing, not polish.
* **If the persistence store isn't actually durable/portable.** The rules must survive session
  loss and, ideally, move with the maintainer's project state (this is the same durability
  question as the calibration data in AID-0043 / `kopitiam-ckv.10` — both want a home in
  `kopitiam-index`). If they don't persist, there is no flywheel, just a per-session cache that
  resets each run.
* **If it's read as replacing judgment rather than caching it.** The flywheel reuses *past
  human judgment*; it doesn't manufacture new judgment. A genuinely novel fork must still reach
  the human. If the matching is ever tuned so aggressively that it answers novel forks from
  loosely-similar old rules, it's stopped being a flywheel and become a bad model.

## Relationships

* **AID-0044** — the human-judgment-rung / choice-card decision loop this makes *cheaper over
  time*; the flywheel is the answer to "won't asking the human get tedious?"
* **AID-0041** — persisted decisions are Rust-owned deterministic dispatch, not model memory;
  the runtime owns the knowledge, the model doesn't.
* **AID-0045** — a hardened rule can further narrow the decoding grammar (fix a slot to the
  decided constant), tightening structure as decisions accumulate.
* **AID-0043 / `kopitiam-ckv.10`** — shares the "where does durable learned state live"
  question (calibration data there, decision rules here) — likely both `kopitiam-index`.
* **Open question:** `kopitiam-ckv.21` — the genuine-fork-vs-silent-default threshold that
  decides which forks are even eligible to become persisted rules.
