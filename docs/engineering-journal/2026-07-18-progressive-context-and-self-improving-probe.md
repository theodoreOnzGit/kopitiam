# 2026-07-18 — Progressive (anytime) context assembly + the self-improving resource probe

*Two patterns worth preserving from the 2026-07-17/18 hybrid-AI design session
(epic `kopitiam-ckv`). The four load-bearing *decisions* live in AID-0040..0043;
this entry keeps the two *engineering patterns* that aren't decisions so much as
techniques the next person shouldn't have to rediscover.*

Related: AID-0040 (dispatch ladder), AID-0041 (grounding), AID-0042 (preemptive
degradation), AID-0043 (shared budgeter + probe), AID-0028 (async LSP session
actor), AID-0037 (kvim resource-aware LSP guard).

---

## 1. Progressive (anytime) context assembly — usable now, better over time

### The pain that forced it

rust-analyzer OOM-kills an Android tablet when the project is too big. So context
for the AI **cannot** be "gather everything, then start" — on a weak device that
gather either never finishes or kills the box. Context assembly has to be
**resource-adaptive**: give the model something to work with *immediately*, then
keep making it richer in the background.

### The shape

* **Small essential core, fetched synchronously** → the AI starts fast, not laggy.
  Whatever the model *must* have to begin reasoning, and no more.
* **The rest streams in the background**, priority-ordered → the context gets
  better over time.
* Usable straight away; refined continuously. This is an **anytime algorithm** in
  the classic sense: stop it at any moment and you have a valid, if less complete,
  answer.

### Refinement 1 — do NOT branch on machine class

The tempting version is `if fast_machine { fetch everything } else { fetch a bit }`.
**Don't.** Classifying the machine is fragile (what counts as "fast"? a powerful
tablet vs a weak laptop don't bucket cleanly) and it forks your code into two paths
that drift.

Instead, **always do the same thing**: a small guaranteed-fast sync core, then
stream the remainder in priority order. On a fast box the stream drains quickly; on
the tablet it lags but the core already unblocked the AI. The insight to hold onto:

> **Capacity = throughput, not a code path.**

A slow device isn't a different algorithm; it's the same algorithm getting through
less of the priority queue in the same wall-clock. One code path, no device
classifier to get wrong.

### Refinement 2 — context must be deterministic-given-budget

Background concurrency is fine, but it must not make context a function of race
timing. **Order the fetches by priority so a given budget always yields the same
prefix of facts.** Then "how far it got" is a function of the *budget*, not of which
thread happened to win.

> Requirement: **context = f(task, budget)**, never f(wall-clock).

If context depended on wall-clock, you lose reproducibility and testability — two
runs of the same task on the same budget could feed the model different facts, and
you could never write a stable test. This keeps the Semantic Runtime's determinism
principle intact even though the fetching is concurrent. Concurrency for speed,
priority-ordering for determinism — you get both.

### Tool-use steers the stream

When the LLM proposes a tool call (a fact-query, per AID-0041's LLM-proposes /
Rust-executes invariant), that request **jumps the priority queue** in the
background fetcher. Enrichment then steers toward what the model is actually
reasoning about, instead of blindly draining a static priority list. Still
LLM-proposes / Rust-executes: the tool request *re-ranks* the queue, Rust runs the
fetch deterministically and **budget-checks it** — a mid-reasoning fetch is subject
to the same preemptive guard (AID-0042) as the initial launch; it must not be
allowed to OOM in the middle of a model turn.

### Pre-fetch first, tool-use second

| | Pre-fetch (Context Builder) | Tool-use (LLM decides mid-reasoning) |
|---|---|---|
| Who queries | Rust gathers facts first, stuffs the prompt | LLM emits "I need X", Rust runs it, loops |
| Determinism | High — reproducible, testable | Lower — model drives the loop |
| Cost | One model call | Many round-trips |
| Best for | Known task shapes | Open-ended, multi-hop |

Lead with **pre-fetch** (it matches `CLAUDE.md`'s pipeline: *load state → collect
facts → build context → invoke model → validate → persist*). Add **bounded**
tool-use only where a task genuinely needs multi-hop discovery. Do not lead with the
agentic loop — it is less deterministic and costs many round-trips.

### Reuse the concurrency you already shipped

Don't invent a second concurrency model. **AID-0028's async LSP session actor**
(commit `2c818cf`) already owns a background worker, streams results, and never
blocks the foreground — that *is* the template for the Context Session. Make it a
sibling actor. `std::thread` + `mpsc`, single-owner, boxed-closure jobs — same
pattern, new client.

---

## 2. The self-improving resource probe — the probe learns its own constants

### The linchpin

The single decision that stops the tablet dying (AID-0042/0043) is a **cheap**
estimate of whether an expensive launch (rust-analyzer, or a gguf load) will fit the
device *before* you launch it. The estimate model is:

```
est_ra_ram ≈ k1·dep_crates + k2·source_MB
```

`dep_crates` from `cargo metadata` (the resolved dep-graph size is the *primary*
predictor — RA's RAM is dominated by indexing all deps, not your own code),
`source_MB` from a stat-only walk (metadata sizes, never open a file).

### The knowledge worth preserving: `k1,k2` can't be derived — measure them, then keep measuring

You cannot get `k1,k2` from first principles. rust-analyzer's real peak RSS depends
on proc-macro load, build scripts, how much of the graph is cached — none of which a
closed-form gives you. So:

1. **Measure to seed.** Run rust-analyzer on a handful of known projects, record
   **peak RSS**, fit `k1,k2`.
2. **The KOPITIAM move — the probe *learns*.** After each real run, record the
   *actual* peak RSS versus what the probe *estimated*, and refine the constants.
   Every session improves the next prediction. This is "every AI interaction leaves
   behind knowledge" (Core Philosophy) applied to the probe itself — the runtime
   accumulates calibration knowledge about *its own device* over time, instead of
   shipping a static guess that's wrong for this particular tablet forever.

### The two rules that keep the learning safe

* **Bias conservative — always.** The learning must never talk itself into
  optimism. The failure is asymmetric and uncatchable: a false `SKIP` costs some IDE
  niceties; a false `FULL` costs a crashed tablet (`SIGKILL`, no recovery). So when
  the estimate is marginal, **degrade**. The goal isn't accuracy; it's *never
  crossing the cliff*. A self-improving probe that improves toward the cliff is worse
  than a dumb pessimistic one.
* **Keep it device-specific and configurable.** The fitted constants are hard-won
  *and* device-specific — the maintainer's tablet is not a CI box. Defaults stay
  conservative, thresholds stay tunable, and the fitted constants + the method get
  recorded as engineering knowledge (this entry, and AID-0043) so nobody later
  mistakes a device-specific number for a universal one.

### The open loose end (flagged, not solved)

For the learning loop to work across sessions, the `(project → actual peak RSS)`
data points need somewhere durable to live (`kopitiam-index`?). That's an open
decide-before-building question — `kopitiam-ckv.10`. Until it's settled, the probe
can still gate safely on seeded constants; it just won't *get better*. Safe but
static beats accurate but optimistic, so this is a nice-to-have loop on top of a
guard that already works, not a blocker for the guard.

### Don't build a second probe — generalise the one that ships

kvim already ships this estimate-then-gate logic for rust-analyzer only
(AID-0037, `crates/kopitiam-neovim/src/lsp/resource_guard.rs`), using the pure-Rust
`sysinfo` crate for the device read. The hybrid-AI budgeter (AID-0043)
**generalises** that: same shape, lifted into `kopitiam-workflow`, plus a second
client (gguf loading) and the calibration loop. Reuse `sysinfo`; do **not** hand-roll
a second `/proc/meminfo` reader (the design draft's "no C sysinfo" worry was a
factual slip — `sysinfo` is pure Rust and already adopted; see AID-0043's premise
correction).
