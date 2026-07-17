//! # kopitiam-gpu — GPU parallel-compute foundation for KOPITIAM
//!
//! A shared, **domain-agnostic** parallel-compute layer built on
//! [`wgpu`](https://docs.rs/wgpu). It is general compute infrastructure — the
//! same foundation AI matmul, iterative root-finding kernels, and plot shaders
//! all sit on — not a tool for any one field. First cut = compute pipelines;
//! render pipelines come later.
//!
//! ## The two rules that shape this crate
//!
//! **1. wgpu is non-optional at compile.** There is no feature gate on the GPU.
//! The wgpu code path is *always* compiled into the binary. wgpu itself
//! Cargo-builds with pure-Rust backends (no C toolchain at build time), so this
//! does not break KOPITIAM's Pure Rust Core build promise — it only needs GPU
//! *drivers* at **runtime**, and their absence is handled below, not by a `cfg`.
//!
//! **2. A missing GPU is a runtime condition, handled by a `Result` cascade.**
//! Plenty of machines have no usable GPU: headless CI, a GPU-less tablet, an
//! Android Termux userland with no Vulkan driver installed. So:
//!
//! ```text
//!     GPU (wgpu compute shader)   -- fast path, when a GPU is present
//!         |  Err / no adapter
//!         v
//!     CPU (pure Rust)             -- correct path, always available
//! ```
//!
//! This mirrors KOPITIAM's Offline-First pipeline (existing -> native -> local
//! -> cloud), here as **GPU -> CPU**. [`GpuContext::new`] probes for a GPU once
//! and returns `Err` (never panics) if there is none; [`Executor`] caches that
//! probe and runs each [`ComputeOp`] GPU-first, CPU-on-failure. Because the CPU
//! twin is a real, correct pure-Rust implementation — not a stub — the answer is
//! guaranteed on every machine.
//!
//! ## Quick start
//!
//! ```
//! use kopitiam_gpu::{Executor, ops::VectorAdd, ops::VectorAddInput};
//!
//! // Builds fine and runs correctly whether or not this machine has a GPU.
//! let exec = Executor::new();
//! let a = [1.0f32, 2.0, 3.0];
//! let b = [10.0f32, 20.0, 30.0];
//! let sum = exec.run(&VectorAdd, &VectorAddInput { a: &a, b: &b });
//! assert_eq!(sum, vec![11.0, 22.0, 33.0]);
//! ```
//!
//! ## Headless & Termux
//!
//! The compute path requests its adapter and device with **no surface / no
//! window**, which is exactly what lets it run under Android Termux (no display
//! context). The instance enables the `Vulkan | GL` backends — the two that
//! matter on Android. Whether a GPU is actually reachable there depends on the
//! device's Vulkan driver (Adreno via Mesa Turnip works well; Mali via Panfrost
//! is immature and often falls back to CPU). See [`GpuContext::new`] for the
//! full driver rundown, and the crate README for the on-device Termux test
//! recipe. Use [`GpuContext::describe_backend`] (logged once at init) to verify
//! on the tablet whether the GPU engaged or the cascade fell back to CPU.
//!
//! ## Scope of the first cut
//!
//! The public API is deliberately on plain Rust types (`&[f32]`, `Vec<f32>`) so
//! the crate is standalone (no internal KOPITIAM deps) and publishable on its
//! own. Integrating with `kopitiam-core` / `kopitiam-tensor` (their `Device` and
//! dtype types) is a follow-up, not part of this cut.

mod context;
mod executor;
pub mod ops;

pub use context::{GpuContext, GpuUnavailable};
pub use executor::{ComputeOp, Executor, GpuOpError};
