# kopitiam-gpu

GPU parallel-compute foundation for KOPITIAM, built on [`wgpu`](https://docs.rs/wgpu).

It's **general, domain-agnostic parallel-compute infrastructure** — the same
foundation AI matmul, iterative root-finding kernels, and plot shaders all can
sit on. Not tie to any one field. First cut is compute pipelines only; render
pipelines come later.

## The two hard rules (this is the whole point)

1. **wgpu is non-optional at compile.** No feature gate on the GPU, ever. The
   wgpu path is always compiled in. wgpu itself Cargo-builds with pure-Rust
   backends (no C toolchain at build time) so this don't break KOPITIAM's Pure
   Rust Core promise — it only needs GPU *drivers* at **runtime**.

2. **No GPU at runtime? Cascade to CPU via `Result`.** Cannot assume every
   machine got a GPU — headless CI, GPU-less tablet, Termux with no Vulkan driver.
   So the flow is:

   ```text
       GPU (wgpu compute shader)   -- laggy-fast path, when GPU is there
           |  Err / no adapter
           v
       CPU (pure Rust)             -- steady path, always can one
   ```

   Same shape as KOPITIAM's Offline-First cascade (existing -> native -> local ->
   cloud), here as **GPU -> CPU**. `GpuContext::new()` probe the GPU once and
   return `Err` (never panic) if none; `Executor` cache that and run each op
   GPU-first, CPU-on-fail. The CPU twin is a real correct pure-Rust impl, not a
   stub, so the answer is guaranteed on every machine.

## Quick start

```rust
use kopitiam_gpu::{Executor, ops::{VectorAdd, VectorAddInput}};

let exec = Executor::new();                 // probe GPU once (never panics)
let a = [1.0f32, 2.0, 3.0];
let b = [10.0f32, 20.0, 30.0];
let sum = exec.run(&VectorAdd, &VectorAddInput { a: &a, b: &b });
assert_eq!(sum, vec![11.0, 22.0, 33.0]);    // correct whether GPU or CPU
```

## Headless & Android Termux

The compute path ask for its adapter + device with **no surface, no window** —
that surface-free design is exactly what let it run inside **Termux**, which got
no display context, no `Activity`, no `ANativeWindow`. We never touch JNI for
compute.

On Android the target GPU path is **Adreno + Turnip** specifically. Turnip is
Mesa's open Vulkan driver (Freedreno) for Qualcomm Adreno GPUs, and wgpu reach it
by requesting a **Vulkan** adapter through the system Vulkan loader (you don't
name Turnip directly — the loader expose it when the ICD is installed). If no
Vulkan adapter is there (no Turnip/ICD, or a Mali device with immature Panfrost)
→ the cascade drop to CPU. No crash lah.

### On-device test recipe (Termux, Adreno)

1. Install the Vulkan loader + the Mesa Turnip/Freedreno ICD:

   ```bash
   pkg install vulkan-loader mesa-vulkan-icd-freedreno
   # optional, to eyeball the driver yourself:
   pkg install vulkan-tools && vulkaninfo | head
   ```

2. Run the demonstrator (any binary/test that builds an `Executor` will log the
   backend once at init to **stderr**):

   ```bash
   cargo test --release -p kopitiam-gpu -- --nocapture
   ```

3. Read the backend report line. This is how you confirm on the tablet whether
   the GPU actually engage or it fell back to CPU:

   * GPU engaged (what you want on Adreno):

     ```text
     [kopitiam-gpu] GPU: Turnip Adreno (TM) 7xx [Vulkan] DiscreteGpu (driver: turnip Mesa ...)
     ```

     If you see `[Vulkan]` and a Turnip/Adreno name, the device GPU is doing the
     work.

   * Fell back to CPU (no ICD, or Mali without a working driver):

     ```text
     [kopitiam-gpu] CPU fallback (no GPU adapter)
     ```

The backend string comes from `GpuContext::describe_backend()` (and the raw form
from `adapter_info()`), so any app embedding this crate can surface the same
verification.

## Tests / CI

CI has **no GPU**, and that's fine — the suite is written so a missing GPU is a
normal passing outcome:

* `cascade_is_always_correct` — runs the cascade; on a no-GPU box it lands on CPU
  and asserts the answer is correct. This is THE guarantee.
* `cpu_fallback_is_correct` — forces `Executor::cpu_only()` and checks the CPU
  floor, even on a GPU machine.
* `gpu_matches_cpu_when_present` — asserts GPU == CPU **only if** a GPU is
  present; skips gracefully otherwise.

```bash
cargo test --release -p kopitiam-gpu
cargo clippy --release -p kopitiam-gpu --all-targets
```

## Scope of the first cut

* One demonstrator op: elementwise `VectorAdd`, on both GPU (WGSL) and CPU.
* Public API on plain Rust types (`&[f32]`, `Vec<f32>`) so the crate is
  standalone (no internal KOPITIAM deps) and publishable on its own.
* Integrating with `kopitiam-core` / `kopitiam-tensor` (`Device`, dtypes) is a
  filed follow-up, not part of this cut.

## Licence

AGPL-3.0-only, same as the rest of KOPITIAM. See the workspace root.
