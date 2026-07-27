# KOPITIAM 0.2.3 — release notes

> **Status: not yet published.** These notes cover the work landed on `main`
> since 0.2.2; nothing here is tagged or on crates.io yet. Dates and the exact
> commit set are in `git log`; provenance for every ported subsystem and forked
> crate is in `docs/ACKNOWLEDGEMENTS.md`, `docs/PROVENANCE.md`, and
> `docs/port-ledger.md`.

**Versioning for this wave.** The workspace crates bump **0.2.2 → 0.2.3**.
Two fork-collapses ship as standalone crates.io packages on their own tracks:
**kopi-beans → 0.1.1** (it already shipped 0.1.0), and **kmux debuts at 0.1.0**.

This wave is smaller and more consolidating than 0.2.2: a new default model with
runtime checksum resolution, a pass at `pdf2md` structural quality, a batch of
dogfood-driven CLI fixes with the next token-max plan written down, and — the
largest single piece — two vendored multi-crate forks collapsed into single,
crates.io-publishable crates, one of them now fully pure-Rust.

## Models — SmolLM2 default and HF-hub checksum resolution

* **SmolLM2 is the default local model.** `kopitiam_models::DEFAULT_MODEL_ID` now
  points at **SmolLM2-360M-Instruct** (HuggingFaceTB, **Apache-2.0**) — ~369 MB
  Q8_0, small and fast enough for local-first / Termux use — replacing the earlier
  Qwen2.5 / Gemma default. The larger **SmolLM2-1.7B-Instruct** (Q4_K_M, also
  Apache-2.0) is offered alongside it. Both are LLaMA-shaped, so they file under
  the existing `Architecture::Llama` catalog path.
* **HF-hub SHA-256 auto-resolution.** `kopitiam-models` no longer carries
  hardcoded catalog checksums. It resolves the expected SHA-256 for a GGUF from
  the HuggingFace hub at fetch time and verifies the download against it, so
  adding or moving a model no longer means hand-transcribing a digest into the
  catalog. Weights are still **downloaded, never vendored or committed**.

## pdf2md quality — adaptive headings and list nesting

Structural fidelity improvements in the `pdf2md` reconstruction path, from the
marker clean-room study (no code copied; see `ACKNOWLEDGEMENTS.md`):

* **Adaptive heading levels.** Heading depth (`#`, `##`, `###`, …) is assigned
  from the document's own observed font-size tiers rather than a fixed cutoff, so
  a paper's section hierarchy survives into the Markdown instead of flattening to
  one level.
* **List nesting.** Ordered/unordered list items reconstruct with their nesting
  depth from indentation, so nested lists emit as nested Markdown rather than a
  flat run of bullets.

## Token-efficiency and dogfooding — CLI fixes and the next plan

A batch of fixes driven by agents actually driving the CLI on kopitiam's own
(large) tree, plus the written-down plan for the follow-up wave:

* **`tokens` path normalization.** Path separators are normalized (forward-slash
  output per the CLAUDE.md cross-platform rule), fixing the `tokens` command on
  Windows/mixed-separator inputs.
* **`refs` / `outline` syntactic fallback via `--no-lsp`.** The semantic
  navigation commands gained a rust-analyzer-free syntactic fallback: on a
  workspace where rust-analyzer will not index inside the timeout, `--no-lsp`
  returns an instant syntactic answer instead of stalling out the full RA wait.
* **Code-navigation skill recipes.** `scripts/gen-kopitiam-skill.sh` now teaches
  the `tokens → outline → refs → read` loop, so an agent reads summaries and
  slices instead of whole files.
* **Token-max follow-ups plan.** `docs/token-max-followups.md` records the next
  `apps/cli` wave (syntactic-first-by-default, a `slice` range-read verb, an
  `orient` composite, `tokens --tree`, symbol-only anchoring, and a non-Rust
  syntactic scanner). Proposal only — nothing in it is implemented yet.

## New standalone crates — two forks collapsed for crates.io

The two vendored multi-crate forks are now **single, self-contained,
publishable crates**. Each upstream workspace's sub-crates were folded into one
crate as intra-crate modules, so publishing to crates.io squats **none** of the
upstream sub-crate names (the new convention is recorded in **AID-0053**).

* **kopi-beans 0.1.1** — the `beads-rs` fork (the `bd`-style issue tracker
  KOPITIAM files its own work in), collapsed from the former ten-crate
  `kopitiam-bds` fork into one `kopi-beans` crate, binary renamed **`bd` → `bn`**.
  It is now **fully pure-Rust and C-free**, which is what makes it cleanly
  publishable: **`rusqlite` (bundled C SQLite) was dropped for a pure-Rust
  `MemoryWalIndex`**, and **`git2` / libgit2 was ported to `gitoxide` (`gix`)**.
  The tree now contains none of `git2` / `libgit2-sys` / `openssl-sys` /
  `libz-sys`, so `bn` cross-compiles to Termux/Android with no NDK.
  **One gap is gated, not hidden:** `gix` 0.86 exposes no high-level push, so
  `bn`'s push path is gated pending a `gix-protocol` / `gix-transport` send-pack
  shim; everything else (fetch, `file://` / `git://` / `ssh` round-trips) works.
* **kmux 0.1.0** — the `rmux` terminal-multiplexer fork (Android/Termux support
  is the reason for the fork), collapsed from the former twelve-crate
  `crates/kmux/crates/` layout into one `kmux` crate. The `-V` banner and binary
  identity were corrected from `rmux` to `kmux` for the crates.io debut.
  `ratatui-rmux` and `rmux-render-core`, which the `kmux` binary never linked,
  were dropped in the collapse.

## Provenance

Upstreams credited or refreshed this wave in `docs/ACKNOWLEDGEMENTS.md`:

* **SmolLM2** (HuggingFaceTB, Apache-2.0) as the default downloaded model, with
  its reference clone at `crates/kopitiam-models/vendor/smollm` — already added
  in the 0.2.2 acknowledgements pass and carried forward.
* **gitoxide (`gix`)** (MIT OR Apache-2.0) — the pure-Rust Git library that
  replaces libgit2/`git2` in kopi-beans — added under shipped dependencies.
* The **kopi-beans** and **kmux** fork entries refreshed to the single-crate
  reality: the collapse, the `bd → bn` rename, and kopi-beans being pure-Rust.
* **huggingface/transformers** (Apache-2.0, reference clone at
  `crates/kopitiam-runtime/vendor/transformers`) and **jesseduffield/lazygit**
  (MIT, reference clone at `apps/cli/vendor/lazygit`, the git-panel UI reference)
  added as clean-room study references — vendored, gitignored, never built or
  shipped.

One new decision record: **AID-0053** (collapse a vendored multi-crate fork into
one publishable crate to avoid squatting upstream sub-crate names on crates.io —
the convention both kopi-beans and kmux followed this wave).
