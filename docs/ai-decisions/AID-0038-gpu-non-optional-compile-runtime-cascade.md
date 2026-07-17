# AID-0038: kopitiam-gpu — wgpu is non-optional at compile, GPU absence is a runtime `Result` cascade to CPU

* **Status:** Pending review
* **Bead:** `kopitiam-6tp` (follow-ups: `kopitiam-zml` tensor integration, `kopitiam-112` aarch64-android build check)
* **Date:** 2026-07-18
* **Decided by:** AI (Claude), acting on the maintainer's brief (maintainer absent)

## The decision

New crate `kopitiam-gpu`: a shared, domain-agnostic **GPU parallel-compute
foundation** on `wgpu`. Two load-bearing rules shape it, and this AID records
them because they are the kind of call the maintainer would normally sign off:

1. **wgpu is compiled in unconditionally — no feature gate on the GPU.** `wgpu`
   is a plain `[dependencies]` entry (via `[workspace.dependencies]`), always
   built. There is no `#[cfg(feature = "gpu")]` anywhere. The GPU code path is
   *always* in the binary.

2. **A missing GPU at runtime is handled by a `Result` cascade, GPU -> CPU.**
   `GpuContext::new() -> Result<GpuContext, GpuUnavailable>` probes the adapter
   + device once, caches it, and returns `Err` (never panics) when there is no
   usable GPU. Each operation is a `ComputeOp` with **both** a GPU impl (wgpu +
   WGSL compute shader) and a pure-Rust CPU impl; the `Executor` runs GPU-first
   and, on any `Err` (or no GPU at all), cascades to the CPU impl. The CPU path
   is a real, correct implementation — the floor the cascade lands on — not a
   stub. This mirrors KOPITIAM's Offline-First pipeline (existing -> native ->
   local -> cloud), here as **GPU -> CPU**.

First cut ships one demonstrator op (`VectorAdd`, elementwise `f32`) on both
paths, wired through the cascade, with the public API on plain `&[f32]`/`Vec<f32>`
so the crate is standalone (no internal KOPITIAM deps) and publishable now.
Seeded at `version = "0.0.1"` behind the workspace version, same as
kopitiam-core/tensor/models.

## Why non-optional compile, and why it does NOT break the Pure Rust Core

`wgpu` **Cargo-builds** with its Rust backends (Vulkan/GL via `ash`/`glow`,
etc.) — there is no C/C++/CMake toolchain in the *build*. So "wgpu always
compiled in" costs us nothing against the Pure Rust Core promise (that promise is
about the *build*, not about what hardware or drivers exist at run time). What
wgpu needs is GPU **drivers at runtime**, and their absence is exactly what the
cascade absorbs. Compile-time and run-time concerns kept cleanly apart: the
build is always the same; only the run adapts.

## Headless + Android Termux

The compute path requests adapter and device with **no surface/window**
(`compatible_surface: None`) — pure compute + buffer readback. That is what lets
it run under Termux (no display context, no `Activity`, no `ANativeWindow`, no
JNI). The instance enables `VULKAN | GL`, the two backends that matter on
Android; on Adreno (Qualcomm) the intended path is the Mesa **Turnip/Freedreno**
Vulkan ICD, reached by requesting a Vulkan adapter through the system loader. If
no Vulkan adapter is reachable (no ICD installed, or a Mali device on immature
Panfrost), the cascade lands on CPU. `GpuContext::describe_backend()` (logged
once at init) surfaces the adapter name + backend + device_type + driver so the
maintainer can verify on-device whether the GPU actually engaged (e.g.
`Turnip Adreno [Vulkan] Gpu`) or it fell back (e.g. `llvmpipe [Gl] Cpu`, or the
`CPU fallback (no GPU adapter)` line).

## Alternatives considered

* **Feature-gate the GPU (`feature = "gpu"`).** Rejected: the maintainer wants
  the GPU path *always available*, not something a downstream build might switch
  off. A feature gate also fragments the build matrix (now the CPU-only config is
  a separate thing to test) for no gain — the runtime cascade already gives the
  "works without a GPU" behaviour a feature flag would be trying to provide.
* **GPU-required (no CPU fallback).** Rejected outright: KOPITIAM must run on
  GPU-less devices (headless CI, a GPU-less tablet, a Mali phone with no working
  Vulkan driver). A GPU-required crate would panic or refuse to run on exactly
  the devices the maintainer uses. The pure-Rust CPU floor is non-negotiable.
* **Owned inputs on the `ComputeOp` trait (no borrows).** Rejected: the API is
  meant to take plain `&[f32]`. Modelled the trait's input as a GAT
  (`type Input<'a>`) so borrows work without forcing `'static`.

## What would make this wrong

The runtime cascade covers a **hardware/driver absence**. It does **not** — and
cannot — cover a **build failure**. Non-optional compile means: if `wgpu` fails
to *build* on some target we care about, that target is broken and no runtime
cascade can save it (there is nothing to fall back to if the crate won't
compile). The real risk points, in order:

* **wgpu not building under Termux's `aarch64-linux-android` toolchain.** This is
  the sharp one. If `wgpu` (or a transitive dep) fails to compile for
  `aarch64-linux-android`, the maintainer's primary on-device target can't build
  KOPITIAM at all. If that happens, the honest fix is a **narrow, documented**
  exception (e.g. a target-gated stub that keeps the CPU path and drops wgpu for
  that one target) — which is a real feature-gate-shaped retreat and should be
  its own AID. Until we've actually built the workspace for
  `aarch64-linux-android` in CI, "wgpu always compiles there" is an assumption,
  not a verified fact.
* **The Mesa Turnip ICD not being installable / present** in the maintainer's
  Termux. That one is *fine* — it just means CPU fallback, which is the designed
  behaviour, not a bug.
* **`wgpu`'s API churn.** wgpu moves fast (this cut is pinned to `wgpu = "30"`,
  wgpu-types 30.0.0; the API differs materially from 24/25 — `Instance::new`
  takes the descriptor by value, `PollType::Wait` is a struct variant, several
  descriptors gained fields). A major bump will need code changes, not just a
  version bump. Pin deliberately; bump deliberately.

Also wrong if the maintainer wanted the GPU foundation to integrate with
`kopitiam-core`/`kopitiam-tensor` (Device/dtype) from the start rather than on
plain `&[f32]`. I read "standalone + trivially publishable now" as the priority
and filed the tensor integration as a follow-up bead; if the priority was
actually the tensor wiring, this first cut aimed at the wrong target.

## Standing constraints observed

* **No crates.io publishing by an agent.** A `--dry-run` of
  `scripts/publish-gpu-seed.sh` was run to prove the crate *packages*; the real
  publish is left for the maintainer to run deliberately. GitHub push only (and
  this work is committed in an isolated worktree, not pushed).
* **Seed version 0.0.1**, pinned in the crate's own Cargo.toml, behind the
  workspace version on purpose — the real release supersedes it later.
