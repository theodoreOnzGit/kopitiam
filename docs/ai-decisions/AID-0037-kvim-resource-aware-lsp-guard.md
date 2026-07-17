# AID-0037: kvim's resource-aware LSP guard — a rough RAM+CPU+size estimate, not a real measurement

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.61` (guard) + `kopitiam-cj0.61.1` (startup progress bar)
* **Date:** 2026-07-18
* **Decided by:** AI (Claude), maintainer absent

## The brief

> kvim runs on an Android tablet. rust-analyzer is fine on small Rust projects,
> but on a BIG project relative to the device's RAM/CPU it gets so laggy it
> crashes the whole tablet (OOM). Before auto-starting the LSP, check if the
> project is too big for this device's RAM & CPU; if so, do NOT start
> rust-analyzer, show a message instead. Manual override so the user is never
> locked out. Fold CPU into the actual gate, not just the message. Use the
> `sysinfo` crate for the device probe. Make the constants configurable.

The mechanism (probe → estimate → gate → message, with `:LspStart` to force) was
directed. What is genuinely the maintainer's call, and is recorded here, is **the
estimate model and its constants** — because they are a guess dressed as a
number, and the whole feature's behaviour rides on them.

## Decision: estimate rust-analyzer's footprint from cheap proxies, gate on a fraction of available RAM scaled by core count

**What was decided.** Before the attach-on-open path (AID-0023) spawns
rust-analyzer, kvim runs a pure, no-I/O-heavy check:

```text
est_ra_mb   = base_mb + per_dep_mb * num_deps + src_factor * src_mb
core_factor = min(1.0, logical_cores / core_ref_count)
budget_mb   = avail_mb * headroom * core_factor
allow       = est_ra_mb <= budget_mb
```

with defaults `base_mb = 150`, `per_dep_mb = 4`, `src_factor = 0.5`,
`headroom = 0.5`, `core_ref_count = 8`. Inputs:

* **`avail_mb`** — `sysinfo`'s available memory (the kernel's estimate of what is
  allocatable without swapping — the honest budget on Android, where there is no
  swap to catch an overshoot). Not total RAM, not free RAM.
* **`num_deps`** — count of `[[package]]` entries in the workspace `Cargo.lock`.
  rust-analyzer analyses the whole dependency graph, so peak RSS scales mostly
  with crate count. Cheap: one file read, no `cargo metadata`.
* **`src_mb`** — total first-party `.rs` bytes under the workspace root, walked
  with the `ignore` crate (skips `target/` and gitignored paths).
* **`logical_cores`** — `std::thread::available_parallelism()`. CPU is a **real**
  input: rust-analyzer is CPU-heavy while indexing, so a few-core tablet janks on
  a mid-size project even when RAM would fit. The core factor quarters a 2-core
  tablet's budget while leaving an 8+-core machine unpenalised.

When gated, kvim never silently skips — it shows a one-line Singlish message with
the numbers and `:LspStart` to force. `:LspStart` clears the gate for the whole
workspace root for the session; `:LspInfo` prints the probe, estimate, and
decision so the user can retune. All five constants live in
`Config.lsp_guard` ([`LspGuardConfig`]).

**Why proxies and not a real measurement.** The honest way to know rust-analyzer's
footprint is to run it and watch its RSS — which is exactly the thing that OOM-kills
the tablet. The guard's whole job is to decide *before* spawning, so it must
estimate from things knowable without spawning. Crate count and source size are
the cheapest signals that correlate with RA memory, and `Cargo.lock` gives the
first for free.

**Why `sysinfo` and not `/proc/meminfo` by hand.** `sysinfo` is pure Rust,
Android-capable, and battle-tested (it is what `bottom`/`btm` is built on), so it
keeps the Pure Rust Core promise while being more robust than a hand-rolled
`/proc` parser. One dependency now serves three kvim features: this guard, the LSP
startup progress bar's ETA (live CPU usage), and a future runtime
memory-cap-and-kill guard (per-process RSS). Library only — kvim never shells out
to any monitoring binary, since none exists on the tablet.

**Why fail-open.** The guard exists to stop a *tablet* OOM, not to second-guess a
capable machine. Whenever it cannot get an honest reading — no `sysinfo` backend,
zero available memory, no `Cargo.lock` — it treats the estimate as "fits" and
starts the LSP as before. On a desktop with tens of GB free the budget dwarfs any
realistic estimate, so the gate never fires there. That is the intended
behaviour, not a loophole.

## Alternatives considered

1. **Run rust-analyzer, watch its RSS, kill it if it crosses a cap
   (measure-not-estimate).** The accurate approach, and genuinely complementary —
   but it cannot *prevent* the first spike, which on Android is the one that can
   OOM-kill the app before a watcher reacts. Filed as a separate follow-up bead
   (`kopitiam-cj0.62`): a runtime cap-and-kill guard belongs *alongside* this
   pre-start gate, not instead of it. This gate is the cheap first line of
   defence; the runtime guard is the safety net.
2. **Always ask the user (a prompt on every big project).** Rejected as the
   default: attach-on-open is meant to be invisible, and a prompt every time you
   open a file in a large repo is worse UX than a smart default with an override.
   `:LspStart` gives the user the same control without the nag.
3. **A pure size cap (e.g. "skip if > N crates").** Simpler, but wrong on both
   ends: it gates a big project on a big desktop (which is fine) and passes a
   mid-project on a tiny tablet (which is not). The device budget has to be in
   the comparison, which means a probe.
4. **CPU as message-only flavour (RAM the sole gate axis).** The maintainer
   explicitly corrected this: fold CPU into the gate. A 4-core tablet lags on
   indexing regardless of RAM headroom, so the core factor scaling the budget is
   load-bearing, not cosmetic.

## What would make this wrong

* **The per-dep memory model is a rough constant.** `4 MB/dep` is a blunt
  average. rust-analyzer's real RSS varies a lot with **proc-macro and
  build-script load** — a workspace heavy in proc-macros (serde-derive, async
  ecosystems) or with expensive build scripts can cost far more per dep than one
  of leaf data crates. The model has no term for that, so it can under-estimate a
  proc-macro-heavy graph (gate passes, tablet still struggles) or over-estimate a
  flat one (gate needlessly fires). If real-world use shows a systematic bias,
  the fix is retuning the constants (they are config) or adding a proc-macro term
  — not scrapping the approach.
* **The core factor models core *count*, not RA's actual CPU behaviour.** RA's
  indexing parallelism and CPU cost also depend on proc-macro/build-script work
  and on how much of the graph is cached, none of which `logical_cores` captures.
  `core_ref_count = 8` as "enough parallelism" is a guess; a device with many
  weak cores (some ARM big.LITTLE tablets) is treated as capable when it is not.
* **`Cargo.lock` package count is a proxy, not the analysed-crate set.** It
  includes workspace members and can include deps that RA barely touches; a
  workspace with an enormous lock but a small active graph is over-estimated.
* **`MemAvailable` is a snapshot.** It reflects the moment of the probe; a device
  that is fine at open can come under memory pressure later (other apps), which a
  pre-start gate cannot see — again the reason the runtime cap-and-kill guard is
  filed as complementary.

If the maintainer, on a real tablet with real projects, finds the gate fires when
it should not (or vice versa), the first move is the config constants; the second
is a better memory model (proc-macro term); the mechanism and fail-open contract
should survive either way.

[`LspGuardConfig`]: ../../crates/kopitiam-neovim/src/config.rs
