# AID-0041: LLMs are grounded, never trusted — the LLM proposes a fact-query, Rust disposes

* **Status:** Pending review
* **Bead:** `kopitiam-7w0` (review) · build task `kopitiam-ckv.4` · epic `kopitiam-ckv`
* **Date:** 2026-07-18
* **Decided by:** Maintainer-present design session (2026-07-17 → 2026-07-18 SGT); written up by AI per the design handoff's §9. Endorsed direction, written down so the reasoning does not vanish into chat.
* **Crate:** `kopitiam-workflow` (context builder) ↔ `kopitiam-knowledge` / `kopitiam-semantic` / `kopitiam-search` (fact providers) ↔ `kopitiam-ai` (model adapters)

## The brief

AID-0040 decided *who runs* (the dispatch ladder). This AID is the other layer:
**what a running model is allowed to know.** These are different seams and they do
not conflict — routing picks the lane, grounding governs the facts a model in that
lane is fed. This is level 3, the determinism boundary, made concrete.

## Decision: a fact must always come out of a Rust-executed deterministic provider, never out of the model's mouth

**What was decided. The invariant is "LLM proposes, Rust disposes."**

A fact must **always** come from a Rust-executed deterministic provider. The model
may *request* a fact — "eh, give me the return type of `select_adapter`" — but
`kopitiam-semantic` answers it deterministically and Rust hands the model the real
value. That is how "the runtime owns understanding" stays true even when the model
is the one driving the reasoning.

```
LLM *asks*, Rust *executes* the query, Rust feeds the result back   → correct
LLM *produces* the fact and you trust it                            → hallucination with extra steps
```

The model does the reasoning (planning, explanation, translation, summarising).
The runtime does the remembering and the fact-lookup. The seam between them is a
single typed **fact-query interface** — one clean "ask the runtime" API the LLM
proposes against, *not* the model poking ten different crates directly. That
interface is the thing everything here and in the context-assembly design (§4)
hangs on, so it gets designed first: its shape is an open question (`kopitiam-ckv.7`,
design §8.1) that must be settled before the fact-query build (`kopitiam-ckv.4`)
starts.

**The cloud-queries-both nesting.** When both a cloud model and a local model are
in play:

```
   CLOUD LLM  ── reasons over ──┐
      │  can consult            ├─ DETERMINISTIC FACTS  ← shared ground truth
      ▼                         │  (kopitiam-knowledge / -semantic / -search)
   LOCAL LLM  ── grounded by ───┘
      │
   (cheap draft / summary / first pass to compress context for the cloud)
```

The cloud model **may** consult the local model — cheap drafting/summarising saves
cloud tokens — **but it must re-ground against the deterministic facts itself**,
never just trust the local model's output. This is "cloud queries *both*", not
"cloud queries local which queried deterministic". The deterministic layer is the
**shared anchor** that keeps both reasoners honest.

## Alternatives considered

1. **Trust the model's stated facts (the "just ask the LLM" default).** Simplest,
   and how most LLM tooling works. **Rejected** — it makes the model the canonical
   source of truth, which the Semantic Runtime principle forbids ("the runtime owns
   knowledge, models borrow it"). A model that states a return type is guessing from
   its weights; the runtime *knows* it from rust-analyzer. Trusting the guess is a
   hallucination you dressed up as an answer.
2. **Cloud trusts the local model's output directly (chain the two reasoners).**
   Tempting because it is one hop cheaper. **Rejected** — chain two probabilistic
   reasoners naively and errors *compound*: a local mistake becomes a cloud premise,
   and the cloud has no way to tell a good local draft from a confidently-wrong one.
   Re-grounding the cloud against the deterministic facts is the firebreak. The local
   model stays useful (it compresses context and drafts cheaply) without being
   *trusted*.
3. **Let the model call each provider crate directly (no single seam).** More
   flexible on paper. **Rejected** — it spreads the "what may a model ask" contract
   across ten crates, makes the trust boundary impossible to audit, and couples the
   model to internal crate layout. One typed fact-query API keeps the boundary in
   one place where it can be reviewed and tested.

## What would make this wrong

* **If the fact-query interface cannot cover what models actually need to ask.**
  This is the load-bearing risk. The invariant only holds if, whenever a model
  needs a fact, it can *express that need as a fact-query* the runtime can execute
  deterministically. If models routinely need things the interface can't phrase —
  a fact no deterministic provider computes, or a query shape the API doesn't
  support — then either the model is forced to produce the fact itself (invariant
  broken) or the work stalls. The interface's expressive coverage is exactly why
  its shape is an open question to settle first (`kopitiam-ckv.7`). If it turns out
  models need a genuinely open-ended query surface that can't be typed cleanly, the
  "one narrow typed seam" premise needs revisiting — but the fix is a richer
  fact-query language, not letting the model invent facts.
* **If "re-ground against deterministic facts" is skipped under token pressure.**
  The cloud-queries-both discipline costs an extra lookup. If someone later
  short-circuits it to save tokens ("the local draft is probably fine"), compounding
  errors come straight back. The re-grounding is the whole point of querying *both*.

## Relationships

* **AID-0040** — the dispatch ladder decides *who runs*; this decides *what they may
  know*. Complementary, not competing.
* **Context-assembly design (§4)** — pre-fetch stuffs the prompt with facts from
  this same interface; tool-use lets the model propose more fact-queries mid-reason.
  Both stay "LLM proposes / Rust executes."
* **Open question `kopitiam-ckv.7`** (fact-query interface shape) — decide before
  building `kopitiam-ckv.4`.
