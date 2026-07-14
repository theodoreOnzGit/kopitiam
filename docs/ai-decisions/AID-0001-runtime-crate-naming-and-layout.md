# AID-0001: Kopitiam Runtime crate naming and workspace layout

* **Status:** Pending review
* **Bead:** `kopitiam-082.2`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

## The decision the maintainer would have made

The Kopitiam Runtime scaffold the maintainer supplied lists 17 crates,
including `kopitiam-cli` and `kopitiam-tui`, all under `crates/`. That
conflicts with two things already true in this repository:

1. `apps/cli` already exists and already builds a binary named `kopitiam`.
   CLAUDE.md's Architecture section is explicit that "Applications are
   clients. The platform owns the functionality," and names a future
   `apps/tui` as a client too. A `crates/kopitiam-cli` would put a *second*
   CLI in the place the codebase reserves for engines.
2. `crates/kopitiam-core` exists but its `Cargo.toml` declares
   `name = "core"` — almost certainly a `cargo new` accident. It is an
   untouched stub, is not in `[workspace.dependencies]`, and nothing
   depends on it.

## What was decided

**1. No `kopitiam-cli` or `kopitiam-tui` crate.** The runtime is an engine;
it gets exposed through the existing `apps/cli` binary as subcommands, the
same way `kopitiam-semantic` and `kopitiam-workflow` already are. The
scaffold's CLI/TUI layer is satisfied by `apps/cli` (today) and `apps/tui`
(when CLAUDE.md's roadmap gets there). This preserves the
apps-are-clients/crates-are-engines split rather than introducing a
parallel one.

**2. `crates/kopitiam-core` is repurposed** as the runtime's shared
primitives crate (dtypes, shapes, errors, device description), and its
package name is corrected from `core` to `kopitiam-core`. Repurposing beats
adding an 18th crate: the directory already exists under the intended name,
and `name = "core"` is a latent bug worth fixing regardless.

**3. Runtime crates are added incrementally, not all 17 at once.** Phase 1
gets `kopitiam-core`, `kopitiam-tensor`, `kopitiam-loader`,
`kopitiam-tokenizer`, and `kopitiam-runtime`. The remaining ten
(`kopitiam-graph`, `kopitiam-memory`, `kopitiam-kernels`,
`kopitiam-scheduler`, `kopitiam-transformers`, `kopitiam-attention`,
`kopitiam-kv-cache`, `kopitiam-sampling`, `kopitiam-quantization`,
`kopitiam-models`) are created when the phase that needs them starts.

Reason: CLAUDE.md's Success Criteria explicitly says not to measure success
by generated files, and its Engineering Principles reject "unnecessary
abstraction." Seventeen empty crates would be seventeen `cargo new` stubs
of exactly the kind this session has been *deleting*. A crate should appear
when there is real code to put in it.

**4. `kopitiam-attention` and `kopitiam-kv-cache` are folded into
`kopitiam-transformers`** when Phase 2 arrives, unless they grow large
enough to stand alone. Attention and its KV cache are one algorithm; the
cache exists only to serve attention. Splitting them across crate
boundaries buys nothing and forces a public API where an internal module
would do. This one is the most reversible of the four — say the word and
they get split back out.

## Alternatives considered

* **Follow the scaffold literally (17 crates, `kopitiam-cli` under
  `crates/`).** Rejected: creates a second CLI in the engine directory,
  contradicting the architecture rule the rest of the repo follows, and
  front-loads 17 stub crates against CLAUDE.md's explicit guidance.
* **Rename the scaffold's CLI to `kopitiam-runtime-cli` and keep it under
  `crates/`.** Rejected as a weaker version of decision 1 — it removes the
  name collision but keeps a client in the engine directory. If the runtime
  later needs a dedicated debug binary (e.g. to dump a GGUF header without
  booting the whole Semantic Runtime), the right home is `apps/`, and it can
  be added then.
* **Leave `name = "core"` alone.** Rejected: it would collide confusingly
  with Rust's own `core` in any error message, and it is not referenced
  anywhere, so fixing it now costs nothing.

## What would make this wrong

* If the maintainer wants the Kopitiam Runtime to be **separately usable
  outside KOPITIAM** — published on its own, embedded by other projects,
  with its own CLI — then `kopitiam-cli`/`kopitiam-tui` as first-class
  crates make sense and decision 1 should be reversed. Nothing in the
  scaffold said this explicitly, but the phrase "Kopitiam owns inference"
  and the separate `runtime/` directory in the proposed layout hint at it.
  This is the single most likely thing I got wrong.
* If the maintainer specifically wants all 17 crates registered up front as
  a visible architectural commitment (a map of intent, not a build
  artifact), decision 3 should be reversed. That is a legitimate style; it
  just contradicts the guidance in CLAUDE.md as I read it.
