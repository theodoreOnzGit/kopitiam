//! The cascade: try the GPU, fall back to the CPU, always return an answer.
//!
//! This mirrors KOPITIAM's Offline-First pipeline (existing knowledge -> native
//! Rust -> local AI -> cloud AI). Here the same shape applies to compute:
//!
//! ```text
//!     GPU (wgpu compute shader)   -- fast path, used when a GPU is present
//!         |  Err / no adapter
//!         v
//!     CPU (pure Rust)             -- correct path, always available
//! ```
//!
//! Two moving parts:
//!
//! * [`ComputeOp`] — an operation that knows how to do itself BOTH ways: a GPU
//!   implementation (fallible — the GPU can be busy, out of memory, or absent)
//!   and a CPU implementation (pure Rust, infallible — it is the floor the
//!   cascade lands on).
//! * [`Executor`] — holds the cached [`GpuContext`] (or `None` if this machine
//!   has no GPU) and runs the cascade: GPU first, CPU on any failure.
//!
//! Because the CPU path is a real, correct implementation and not a stub, the
//! guarantee is total: `Executor::run` returns the right answer on every
//! machine, GPU or not. On a no-GPU machine it simply never touches wgpu.

use crate::context::GpuContext;

/// An operation implemented on BOTH the GPU and the CPU.
///
/// Implementors provide the two paths; the [`Executor`] chooses between them.
/// Keep `Input`/`Output` on plain Rust types (`&[f32]`, `Vec<f32>`, ...) so an
/// op is usable without pulling in wgpu types at the call site.
///
/// Contract the two paths MUST honour: for the same input, `compute_gpu` (when
/// it succeeds) and `compute_cpu` must produce the **same result** — same length
/// and, for floats, bit-for-bit equal for exactly-representable arithmetic like
/// elementwise add. The test suite asserts exactly this whenever a GPU is
/// present. If a kernel is only approximately equal to its CPU twin (e.g. a
/// fast-math reduction), say so in that op's docs and loosen its test to a
/// tolerance — do not weaken this trait's default expectation silently.
pub trait ComputeOp {
    /// The operation's input, borrowed. A GAT (`Input<'a>`) so the input can
    /// hold borrows (`&[f32]`) without forcing them to be `'static` — the whole
    /// reason a plain associated type wouldn't do here. Both paths take it by
    /// shared reference, so the cascade can hand the SAME input to the GPU path
    /// and then, on failure, to the CPU path.
    type Input<'a>;
    /// The operation's output.
    type Output;

    /// Run on the GPU. Fallible: returns `Err` if the GPU path could not
    /// complete for ANY reason (shader/pipeline error, buffer map failure,
    /// device lost). A returned `Err` is the cascade's signal to fall back to
    /// CPU — it must never be a panic.
    fn compute_gpu(
        &self,
        ctx: &GpuContext,
        input: &Self::Input<'_>,
    ) -> Result<Self::Output, GpuOpError>;

    /// Run on the CPU in pure Rust. Infallible and always correct — this is the
    /// floor the cascade lands on, so it must be a real implementation.
    fn compute_cpu(&self, input: &Self::Input<'_>) -> Self::Output;
}

/// A GPU path failed to complete. The cascade treats ANY of these as "fall back
/// to CPU", so the variants exist for logging/diagnosis, not for the caller to
/// recover differently per case.
#[derive(Debug, thiserror::Error)]
pub enum GpuOpError {
    /// A wgpu-level failure (buffer mapping, device poll, validation). String
    /// because the underlying wgpu error types are not uniform or `'static`.
    #[error("GPU compute failed: {0}")]
    Backend(String),

    /// The op's own precondition was violated in a way the GPU path can't run
    /// (e.g. mismatched input lengths). Kept distinct so a genuine caller bug is
    /// not silently masked by the CPU fallback in logs.
    #[error("invalid input for GPU op: {0}")]
    InvalidInput(String),
}

/// Runs [`ComputeOp`]s through the GPU->CPU cascade.
///
/// Build ONE and reuse it: it probes the GPU exactly once at construction and
/// caches the handle. A program that makes an `Executor` per call pays the
/// adapter-enumeration cost every time and defeats the caching.
#[derive(Debug, Default)]
pub struct Executor {
    /// The cached GPU handle, or `None` if this machine has no usable GPU. Once
    /// `None`, the cascade skips straight to CPU for every op — no repeated
    /// probing, no repeated failures.
    ctx: Option<GpuContext>,
}

impl Executor {
    /// Build an executor, probing for a GPU once.
    ///
    /// Never fails: a missing GPU is not an error here, it just means every op
    /// will take the CPU path. Use [`Executor::has_gpu`] if you need to know
    /// which way the cascade will go.
    pub fn new() -> Self {
        // `.ok()` collapses "no GPU" into `None`; that is the whole graceful-
        // degradation move. We deliberately discard the GpuUnavailable reason
        // here — callers that care can call GpuContext::new() themselves.
        let ctx = GpuContext::new().ok();
        if ctx.is_none() {
            // The GPU-present counterpart is logged inside GpuContext::new. This
            // is the line the maintainer looks for on the tablet to confirm we
            // ran on CPU (no reachable Vulkan/GL adapter), not the GPU. stderr,
            // so it never mixes into a command's stdout.
            eprintln!("[kopitiam-gpu] CPU fallback (no GPU adapter)");
        }
        Self { ctx }
    }

    /// Build an executor that is FORCED onto the CPU path, ignoring any GPU.
    ///
    /// This is what makes the CPU fallback testable on a machine that *does*
    /// have a GPU: with no adapter to borrow, `run` can only cascade to CPU.
    pub fn cpu_only() -> Self {
        Self { ctx: None }
    }

    /// Does this executor have a GPU to use? `false` means every `run` lands on
    /// CPU. Handy for tests (skip the GPU==CPU assertion when there is no GPU)
    /// and for logging which path a workload will take.
    pub fn has_gpu(&self) -> bool {
        self.ctx.is_some()
    }

    /// The cached GPU context, if any. Lets a caller run the GPU path directly
    /// (e.g. a test asserting GPU==CPU) without going through the cascade.
    pub fn gpu_context(&self) -> Option<&GpuContext> {
        self.ctx.as_ref()
    }

    /// Run an op through the cascade: **GPU first, CPU on any failure.**
    ///
    /// This is the one method callers normally use. It always returns a result:
    ///
    /// * GPU present and its path succeeds -> the GPU result.
    /// * GPU present but its path returns `Err` -> the CPU result (logged as a
    ///   fall-back at `warn`... once tracing is wired; today the `Err` is simply
    ///   swallowed into the CPU path).
    /// * no GPU at all -> straight to the CPU result.
    ///
    /// The cascade itself is the small `?`/`unwrap_or_else` dance below: build a
    /// `Result` for the GPU attempt (short-circuiting to `Err` if there is no
    /// context), then `unwrap_or_else` onto the infallible CPU path.
    pub fn run<O>(&self, op: &O, input: &O::Input<'_>) -> O::Output
    where
        O: ComputeOp,
    {
        self.try_gpu(op, input)
            .unwrap_or_else(|_| op.compute_cpu(input))
    }

    /// The GPU leg of the cascade as a single `Result`.
    ///
    /// `ok_or(...)?` turns "no cached context" into the same `Err` channel as a
    /// failed kernel, so [`run`](Self::run)'s `unwrap_or_else` handles both the
    /// no-GPU and the GPU-failed cases with one branch.
    fn try_gpu<O>(&self, op: &O, input: &O::Input<'_>) -> Result<O::Output, GpuOpError>
    where
        O: ComputeOp,
    {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| GpuOpError::Backend("no GPU context on this machine".into()))?;
        op.compute_gpu(ctx, input)
    }
}
