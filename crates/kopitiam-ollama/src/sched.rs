//! # `sched` -- who get to sit down, who kena chase out
//!
//! **Upstream:** `server/sched.go` (and its `server/sched_test.go`), from ollama
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28), MIT, Copyright (c)
//! Ollama. This is a **port**, not inspiration -- where we and ollama disagree,
//! ollama wins, and every deliberate divergence say so at the point where it
//! diverge.
//!
//! ## What this module decide
//!
//! The scheduler is the bouncer of the kopitiam. It answer four questions, and
//! only these four:
//!
//! 1. **Which models are resident right now**, keyed by a stable model key.
//! 2. **When must one be chased out** to make room for the next request -- and
//!    *which* one ([`Scheduler::find_runner_to_unload`]).
//! 3. **How many requests may share one loaded model** (`num_parallel`), and how
//!    many models may be resident at once (`max_runners`).
//! 4. **Does the next model actually fit in the VRAM we got left**, and on which
//!    device(s) should it sit.
//!
//! Get (4) wrong one way, KOPITIAM leave VRAM idle and run slow. Get it wrong
//! the other way, the allocator OOM and the whole box die. So this is policy
//! that must be *faithful*, not merely reasonable.
//!
//! ## The seam: no subprocess, no GPU driver, no threads
//!
//! Upstream's scheduler talk to `llm.LlamaServer` (a spawned llama-server
//! subprocess) and `discover.GPUDevices` (live NVML / ROCm SMI queries), and it
//! run the whole thing as two goroutines trading on four channels.
//!
//! **None of that come in here.** `kopitiam-ollama` deliberately depend on
//! nothing else in KOPITIAM (see `docs/ai-decisions/AID-0055`), so pulling in
//! `kopitiam-runtime` or `kopitiam-gpu` -- or spawning a process -- would invert
//! that. Instead the two moving parts are **a trait and a plain input struct**:
//!
//! * [`ModelRunner`] -- what the scheduler need *from* a loaded runner: how much
//!   VRAM it took, per-device, whether it still answer a ping, and how to close
//!   it. Seven methods, no I/O in the signature.
//! * [`DeviceInfo`] / [`SystemInfo`] -- a **snapshot** of the hardware, handed in
//!   by the caller. The scheduler never queries anything.
//!
//! Same seam upstream's own `sched_test.go` use -- it swap `newServerFn` for a
//! `mockLlm` and `getGpuFn` for a fake device list. We just make that seam the
//! only interface instead of a test-only escape hatch.
//!
//! ## The concurrency decision, and why
//!
//! **There are no threads and no async runtime in this module. The policy is a
//! pure, synchronous state machine; the caller own all the threading.** That is
//! a deliberate divergence from upstream's goroutines-and-channels, and here is
//! the reasoning, because it is the one design call in this file a future
//! maintainer might want to reverse:
//!
//! * This crate has **no async runtime and no `tokio`**, and adding one is a big
//!   architectural decision that belong to the maintainer, not to a port.
//! * Upstream's four channels (`pendingReqCh`, `finishedReqCh`, `expiredCh`,
//!   `unloadedCh`) exist to *serialise* access to one map. A `&mut Scheduler`
//!   already serialise it, for free, at compile time.
//! * Several of upstream's guards exist **only** because a channel message can
//!   be stale by the time it is read -- the "expired event with positive ref
//!   count, retry in 10ms" dance, the "duplicate expired event, ignoring" check,
//!   and the pid-mismatch orphan branch. Here expiry is a **query over current
//!   state** ([`Scheduler::take_expired`]), so a stale event cannot exist to be
//!   guarded against. Fewer moving parts, identical behaviour.
//! * Testability: every decision below is reachable without a thread, a sleep,
//!   or a timeout. Upstream's own tests are full of `time.Sleep(20ms)` and flake
//!   accordingly. Ours are not.
//!
//! What the caller therefore own: a loop, whatever locking it want around
//! `&mut Scheduler`, the actual load (spawn / mmap / whatever), and a clock. The
//! module take `now: Instant` as a **parameter** everywhere time matter, so a
//! test can jump five minutes ahead without waiting five minutes.
//!
//! **What would make this design wrong:** if a caller ever need two loads to
//! proceed genuinely concurrently *and* both to mutate scheduler state
//! mid-flight. Upstream cannot do that either -- it hold `activeLoading` to
//! enforce one load at a time (`sched.go`: *"We can only load one model at a
//! time"*) -- so the constraint is upstream's, not ours. If that invariant ever
//! change upstream, this design need revisiting.
//!
//! ## The refcount invariant -- read this before touching anything
//!
//! [`RunnerRef::ref_count`] is **the number of in-flight requests holding this
//! runner**. The invariants, exactly:
//!
//! * A runner with `ref_count > 0` is **never** unloaded. Not on keep-alive
//!   expiry, not to make room, not on evict-all. It only get marked
//!   expire-immediately, and it go when the last request let go.
//! * Every [`Scheduler::use_loaded_runner`] (+1) must be matched by exactly one
//!   [`Scheduler::request_finished`] (-1). Miss the release and the model is
//!   pinned in VRAM forever; call it twice and you unload a runner somebody is
//!   still streaming from.
//! * `ref_count` only drop to 0 in `request_finished`, and that is the **only**
//!   place a keep-alive deadline get armed. A runner that never went idle has no
//!   deadline, on purpose.
//!
//! ## What "fits" mean, in bytes
//!
//! Two different thresholds, and mixing them up is a real bug:
//!
//! * **Fit check** ([`fits_with_headroom`]): `predicted <= available * 80/100`.
//!   The 20% is headroom for the allocator's own slack, from `sched.go`'s
//!   *"Use 80% of free memory as threshold to leave headroom"*.
//! * **Per-device overhead** ([`DeviceInfo::available_after_overhead`]):
//!   `free - gpu_overhead - minimum_memory`, floored at 0. This is VRAM the
//!   *driver* eat, not the model.
//!
//! `available` itself is not just "sum of free VRAM" either -- on an integrated
//! GPU the device's reported free memory is a stale baseline over the same
//! physical RAM the OS is using, so [`available_memory_for_load`] take the
//! smaller of the two. Get that backwards on a Mac and you promise 300 GB.
//!
//! ## What is deliberately NOT ported
//!
//! Spawning llama-server (`newServerFn`, `WaitUntilRunning`, `Pid`) -- no
//! processes here. The live GPU poll inside `waitForVRAMRecovery` -- we port the
//! *predicate* ([`vram_recovery_converged`]) and leave the polling to the
//! caller. MLX runner selection -- Apple-only and needs a subprocess. And all
//! the `slog` calls -- this crate log nothing.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use indexmap::IndexMap;

use crate::api::Capability;
use crate::envconfig::{Env, Expiry};
use crate::format::{GIBIBYTE, GIGABYTE, MEBIBYTE};
use crate::memory::{FlashAttentionType, Kv};
use crate::options::Options;

// ---------------------------------------------------------------------------
// Constants -- every one of these name where it come from
// ---------------------------------------------------------------------------

/// How many models we let sit on one GPU when the user never said.
///
/// **Upstream:** `sched.go` `defaultModelsPerGPU = 3`, with its comment: *"Model
/// will still need to fit in VRAM, but loading many small models on a large GPU
/// can cause stalling."* So this is a stall guard, not a memory guard -- the
/// memory guard is the fit check.
pub const DEFAULT_MODELS_PER_GPU: u64 = 3;

/// How long we wait for a loaded runner to answer a health ping before deciding
/// it is dead and must be reloaded.
///
/// **Upstream:** `sched.go` `needsReload`, `timeout := 10 * time.Second`.
pub const NEEDS_RELOAD_PING_TIMEOUT: Duration = Duration::from_secs(10);

/// Same ping, but while the runner is still doing its **initial** load.
///
/// **Upstream:** `sched.go` `needsReload`, `timeout = 2 * time.Minute` with the
/// comment *"Initial load can take a long time for big models on slow
/// systems..."*. Two minutes is not paranoia lah -- a 70B off a spinning disk
/// genuinely take that long, and reloading it because it was slow would be the
/// worst possible response.
pub const NEEDS_RELOAD_LOADING_PING_TIMEOUT: Duration = Duration::from_secs(120);

/// How long the caller should keep polling for VRAM to come back after a runner
/// exit, before giving up and proceeding anyway.
///
/// **Upstream:** `sched.go` `InitScheduler`, `waitForRecovery: 5 * time.Second`.
pub const VRAM_RECOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Poll interval while waiting for that recovery.
///
/// **Upstream:** `sched.go` `waitForVRAMRecovery`,
/// `time.NewTicker(250 * time.Millisecond)`, with the note that *"typical
/// convergence is 0.5-1.5s"*.
pub const VRAM_RECOVERY_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Fraction of a dead runner's VRAM that must come back before we call it
/// recovered.
///
/// **Upstream:** `sched.go` `waitForVRAMRecovery`, *"If we're within ~75% of the
/// estimated memory usage recovered, bail out"*. Not 100% because driver
/// free-memory reporting lag and never quite converge; waiting for the last 25%
/// just burn the whole 5s timeout every single time.
pub const VRAM_RECOVERY_FRACTION: f32 = 0.75;

/// The headroom percentage every fit decision use.
///
/// **Upstream:** `sched.go` `load`, *"Use 80% of free memory as threshold to
/// leave headroom"*, and the same `*80/100` in `bestSingleGPUFit`,
/// `generationBatchFits` and `disableMmapForHostPressure`.
pub const FIT_HEADROOM_PERCENT: u64 = 80;

/// Smallest context we will ever hand to a runner.
///
/// **Upstream:** `sched.go` `getRunner`, `if opts.NumCtx < 4 { opts.NumCtx = 4 }`.
pub const MIN_NUM_CTX: u32 = 4;

/// Floor on context for a vision model.
///
/// **Upstream:** `sched.go` `getRunner`, *"multimodal models require at least
/// 2048 context"*. One image expand into a lot of tokens -- a 4-token window
/// cannot even hold a single tile.
pub const VISION_MIN_NUM_CTX: u32 = 2048;

/// VRAM the Metal backend eat before any model bytes land.
///
/// **Upstream:** `ml/device.go` `DeviceInfo.MinimumMemory()`,
/// `512 * format.MebiByte`.
pub const MINIMUM_MEMORY_METAL: u64 = 512 * MEBIBYTE;

/// Same, for every other backend.
///
/// **Upstream:** `ml/device.go` `DeviceInfo.MinimumMemory()`,
/// `457 * format.MebiByte`. The oddly specific 457 is upstream's measured figure
/// for CUDA context structures; do not round it "tidy".
pub const MINIMUM_MEMORY_DEFAULT: u64 = 457 * MEBIBYTE;

/// Default prompt-eval batch size.
///
/// **Upstream:** `sched.go` `llamaServerGenerationBatchDefault = 512`.
pub const GENERATION_BATCH_DEFAULT: u32 = 512;

/// Batch used on a small CUDA card that has flash attention switched off.
///
/// **Upstream:** `sched.go` `llamaServerGenerationBatchConstrained = 256`.
pub const GENERATION_BATCH_CONSTRAINED: u32 = 256;

/// Batch for a medium context window.
///
/// **Upstream:** `sched.go` `llamaServerGenerationBatchMedium = 1024`.
pub const GENERATION_BATCH_MEDIUM: u32 = 1024;

/// Batch for a large context window.
///
/// **Upstream:** `sched.go` `llamaServerGenerationBatchLarge = 2048`.
pub const GENERATION_BATCH_LARGE: u32 = 2048;

/// Predicted VRAM must sit under this share of available before we promote to
/// the medium batch.
///
/// **Upstream:** `sched.go` `llamaServerGenerationBatchMediumHeadroomPercent = 75`.
pub const GENERATION_BATCH_MEDIUM_HEADROOM_PERCENT: u64 = 75;

/// Same, for the large batch.
///
/// **Upstream:** `sched.go` `llamaServerGenerationBatchLargeHeadroomPercent = 60`.
pub const GENERATION_BATCH_LARGE_HEADROOM_PERCENT: u64 = 60;

/// Floor on the host-memory headroom we insist on before mmap-ing another model.
///
/// **Upstream:** `sched.go` `mmapHostPressureHeadroom`, `8 * format.GigaByte`
/// -- **decimal** GB, not GiB. Upstream really do use `format.GigaByte` here
/// while using `format.GibiByte` for the batch surcharges two functions away, so
/// do not "harmonise" them.
pub const MMAP_HOST_PRESSURE_HEADROOM_MIN: u64 = 8 * GIGABYTE;

/// Architectures that are **not safe** with more than one parallel sequence.
///
/// **Upstream:** `sched.go` `load`, the `slices.Contains([]string{...})` guard,
/// citing <https://github.com/ollama/ollama/issues/4165>. These are models whose
/// runner state is not per-sequence (recurrent / hybrid / interleaved-vision
/// architectures), so two sequences sharing one runner corrupt each other's
/// state. It is a **correctness** list, not a performance one -- never trim it
/// to "go faster".
pub const UNSAFE_PARALLEL_ARCHITECTURES: &[&str] = &[
    "mllama",
    "qwen3vl",
    "qwen3vlmoe",
    "qwen35",
    "qwen35moe",
    "qwen3next",
    "lfm2",
    "lfm2moe",
    "nemotron_h",
    "nemotron_h_moe",
    "nemotron_h_omni",
];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Something went wrong scheduling a request.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SchedError {
    /// The pending queue is full.
    ///
    /// **Upstream:** `sched.go` `ErrMaxQueue`. The message is kept byte-for-byte,
    /// double space and all, because clients match on the string.
    #[error("server busy, please try again.  maximum pending requests exceeded")]
    MaxQueue,

    /// Nothing else is loaded and the model **still** does not fit. No eviction
    /// would help.
    ///
    /// **Upstream:** `sched.go` `load`, *"model is too large for system memory"*.
    #[error("model is too large for system memory")]
    TooLarge,

    /// The runner died or never came up.
    #[error("runner load failed: {0}")]
    LoadFailed(String),
}

/// The runner stopped answering.
///
/// **Upstream:** the non-nil return of `llm.LlamaServer.Ping(ctx)`.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("runner not responding: {0}")]
pub struct RunnerUnhealthy(pub String);

// ---------------------------------------------------------------------------
// Hardware-shaped inputs -- a SNAPSHOT, handed in, never queried
// ---------------------------------------------------------------------------

/// Which device, in which backend.
///
/// **Upstream:** `ml/device.go` `DeviceID`.
///
/// `id` is only unique **within one `library`** -- CUDA's device 0 and ROCm's
/// device 0 are different cards, so anything keyed on a device must key on the
/// pair. That is why this is a struct and not a bare string, and why
/// [`by_library`] group before it compare.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId {
    /// Post-filter device index or UUID, as the backend report it.
    pub id: String,
    /// Backend name: `CUDA`, `ROCm`, `Metal`, `Vulkan`, `CPU`.
    pub library: String,
}

impl DeviceId {
    /// Build one inline. Handy in tests and in caller glue.
    pub fn new(id: impl Into<String>, library: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            library: library.into(),
        }
    }
}

/// One accelerator, as the caller saw it at snapshot time.
///
/// **Upstream:** `ml/device.go` `DeviceInfo`, trimmed to the fields the
/// scheduler actually read. Driver versions, PCI IDs, compute capability and
/// library paths are all dropped -- they matter to device *discovery*, which is
/// not this crate's job.
///
/// **`free_memory` is a snapshot and it lie a bit.** CUDA's free-memory
/// reporting lag behind reality by up to a couple of seconds after a runner
/// exit, which is the entire reason [`vram_recovery_converged`] exist.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Backend + index.
    pub device_id: DeviceId,
    /// Backend's own label for the card.
    pub name: String,
    /// Longer user-facing identification.
    pub description: String,
    /// Unfiltered device ID, when a numeric `id` was filtered.
    pub filter_id: String,
    /// **True for an integrated GPU sharing system RAM.** Drives two separate
    /// decisions: integrated free-memory get clamped by system free memory
    /// ([`available_memory_for_gpu`]), and integrated devices skip the VRAM
    /// recovery wait entirely (there is no separate pool to recover).
    pub integrated: bool,
    /// Total memory usable for models.
    pub total_memory: u64,
    /// Currently free, per the backend.
    pub free_memory: u64,
}

impl DeviceInfo {
    /// Shorthand for `self.device_id.library`.
    pub fn library(&self) -> &str {
        &self.device_id.library
    }

    /// Shorthand for `self.device_id.id`.
    pub fn id(&self) -> &str {
        &self.device_id.id
    }

    /// VRAM the backend itself eat, before a single model byte land.
    ///
    /// **Upstream:** `ml/device.go` `DeviceInfo.MinimumMemory()`.
    pub fn minimum_memory(&self) -> u64 {
        if self.library() == "Metal" {
            MINIMUM_MEMORY_METAL
        } else {
            MINIMUM_MEMORY_DEFAULT
        }
    }

    /// What is genuinely available on this device once driver overhead and the
    /// user's configured `gpu_overhead` are set aside.
    ///
    /// **Upstream:** `sched.go` `load`'s per-GPU block:
    /// `available := gpu.FreeMemory - envconfig.GpuOverhead() - gpu.MinimumMemory()`,
    /// with `available = 0` when free sit below the two overheads. Saturating
    /// subtraction reproduce that clamp exactly and cannot underflow.
    pub fn available_after_overhead(&self, gpu_overhead: u64) -> u64 {
        self.free_memory
            .saturating_sub(gpu_overhead)
            .saturating_sub(self.minimum_memory())
    }
}

/// Host memory, as the caller saw it.
///
/// **Upstream:** `ml/device.go` `SystemInfo`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemInfo {
    /// Total system RAM.
    pub total_memory: u64,
    /// Currently free system RAM. **A live measurement**, unlike an integrated
    /// GPU's `free_memory`, which is precisely why the two get compared.
    pub free_memory: u64,
    /// Free swap. Logged upstream, used in no decision -- kept because a caller
    /// explaining "why did this fall back to CPU" want it.
    pub free_swap: u64,
}

// ---------------------------------------------------------------------------
// The runner seam
// ---------------------------------------------------------------------------

/// What the scheduler need from a loaded model. Nothing more.
///
/// **Upstream:** the subset of `llm.LlamaServer` that `server/sched.go` actually
/// touch. Upstream's is a subprocess behind an HTTP socket; here it is whatever
/// the caller want -- an in-process runtime, a subprocess, or a fake.
///
/// `Send + Sync` is required so a caller can move a [`Scheduler`] onto a worker
/// thread. That is the whole point of leaving threading to the caller: if this
/// trait were not thread-safe, "caller owns the threading" would be a lie.
///
/// Interior mutability is the implementor's problem, exactly like upstream --
/// [`ModelRunner::unload`] take `&self` because upstream's `Close()` is called
/// through a shared pointer while other goroutines still hold it.
pub trait ModelRunner: Send + Sync {
    /// Bytes this runner put on the GPU(s).
    ///
    /// **Upstream:** the second return of `llm.LlamaServer.MemorySize()`.
    fn vram_size(&self) -> u64;

    /// Bytes this runner took in total, GPU + host.
    ///
    /// **Upstream:** the first return of `llm.LlamaServer.MemorySize()`.
    fn total_size(&self) -> u64;

    /// Bytes this runner put on **one specific** device.
    ///
    /// **Upstream:** `llm.LlamaServer.VRAMByGPU(id)`. Must return 0 for a device
    /// this runner never touched -- [`Scheduler::update_free_space`] sum this
    /// across every runner and every device, so a wrong non-zero here shrink the
    /// free-memory estimate of a card that is actually empty, and the next model
    /// get needlessly evicted to make room on it.
    fn vram_by_gpu(&self, id: &DeviceId) -> u64;

    /// The context window the runner actually settled on, which may be **smaller
    /// than requested** (clamped to the model's trained maximum). `0` mean "not
    /// known yet".
    ///
    /// **Upstream:** `llm.LlamaServer.ContextLength()`.
    fn context_length(&self) -> u32;

    /// Is the runner still alive and answering, within `timeout`?
    ///
    /// **Upstream:** `llm.LlamaServer.Ping(ctx)` under a `context.WithTimeout`.
    /// An `Err` force a reload -- see [`Scheduler::needs_reload`].
    fn ping(&self, timeout: Duration) -> Result<(), RunnerUnhealthy>;

    /// Shut the runner down and release its memory.
    ///
    /// **Upstream:** `llm.LlamaServer.Close()`. **Must be idempotent.** The
    /// duplicate-unload races upstream guard against cannot happen here (see the
    /// module header), but a caller retrying after an error still deserve not to
    /// get a double free.
    fn unload(&self);

    /// OS process id, or a negative number when there is no process.
    ///
    /// **Upstream:** `llm.LlamaServer.Pid()`. Purely diagnostic in this port --
    /// upstream use it to spot orphaned runners after a channel race, and that
    /// race cannot occur here.
    fn pid(&self) -> i32 {
        -1
    }

    /// Has the runner already gone away by itself (crashed, killed)?
    ///
    /// **Upstream:** `llm.LlamaServer.HasExited()`.
    fn has_exited(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// The model-shaped input
// ---------------------------------------------------------------------------

/// Everything the scheduler need to know about a model, without opening a file.
///
/// **Upstream:** the fields of `server.Model` that `sched.go` read, plus the two
/// facts it read off the decoded GGUF (`f.KV().ContextLength()` and
/// `f.KV().BlockCount()`).
///
/// **The caller fill this in.** GGUF decoding belong to `kopitiam-loader`, and
/// this crate does not depend on it, so `train_context`, `block_count` and
/// `file_size` come in rather than being read. `0` mean *unknown* for all three,
/// which is exactly what upstream's `modelTrainContext(nil)` and
/// `modelFileSize()` (on a failed `os.Stat`) return -- so "unknown" behave
/// identically here and there.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRef {
    /// Path to the weights. Empty for a safetensors / image model with no single
    /// file.
    pub model_path: String,
    /// Manifest digest. Becomes the scheduler key when `model_path` is empty.
    pub digest: String,
    /// Fully-qualified model name.
    pub name: String,
    /// Short display name.
    pub short_name: String,
    /// LoRA adapters, in order. **Order is significant** -- a different order is
    /// a different model, and [`Scheduler::needs_reload`] compare the slices
    /// element-wise, matching upstream's `reflect.DeepEqual`.
    pub adapter_paths: Vec<String>,
    /// Multimodal projectors, in order. Same ordering rule.
    pub projector_paths: Vec<String>,
    /// Architecture family, e.g. `llama`, `qwen3`, `deepseek2`. Drives the
    /// parallel-safety check and the context-shift check.
    pub model_family: String,
    /// Every family this model belong to. A model can list `deepseek2` here
    /// without it being the primary family.
    pub model_families: Vec<String>,
    /// What this model can do. Only [`Capability::Completion`] and
    /// [`Capability::Vision`] matter to the scheduler.
    pub capabilities: Vec<Capability>,
    /// The context length the model was **trained** at, from GGUF
    /// `<arch>.context_length`. `0` = unknown, meaning "do not clamp".
    pub train_context: u32,
    /// Transformer block count, from GGUF `<arch>.block_count`. `0` = unknown.
    /// Used to tell a full offload from a partial one.
    pub block_count: u64,
    /// Size of the weights file in bytes, `0` if unknown. Feeds the host-pressure
    /// mmap heuristic.
    pub file_size: u64,
    /// True for an MLX (Apple) model. Such a runner is not a llama-server, so the
    /// option comparison in [`Scheduler::needs_reload`] is skipped for it,
    /// matching upstream's `!runner.model.IsMLX()`.
    pub is_mlx: bool,
}

impl ModelRef {
    /// Does this model have the named capability?
    pub fn has_capability(&self, c: Capability) -> bool {
        self.capabilities.contains(&c)
    }

    /// The map key this model is scheduled under.
    ///
    /// **Upstream:** `sched.go` `schedulerModelKey`.
    ///
    /// The fallback ladder matter: GGUF-backed models key on `model_path`, but a
    /// safetensors or image model has none, so it fall back to the manifest
    /// digest -- *"so distinct models don't collide"*, in upstream's words. Key
    /// two different models the same and the scheduler will cheerfully hand one
    /// request the other model's runner.
    pub fn scheduler_key(&self) -> String {
        if !self.model_path.is_empty() {
            return self.model_path.clone();
        }
        if !self.digest.is_empty() {
            return format!("digest:{}", self.digest);
        }
        if !self.name.is_empty() {
            return format!("name:{}", self.name);
        }
        if !self.short_name.is_empty() {
            return format!("short:{}", self.short_name);
        }
        String::new()
    }

    /// Can this architecture shift its context window when it fill up?
    ///
    /// **Upstream:** `sched.go` `supportsContextShift`. Everything can, except
    /// `deepseek2` -- its MLA attention cache cannot be rolled forward the way a
    /// plain KV cache can, so shifting it produce garbage. Upstream report `true`
    /// for a `nil` model; here the `None` case belong to the caller, so this is a
    /// plain method on a real model.
    pub fn supports_context_shift(&self) -> bool {
        self.model_family != "deepseek2" && !self.model_families.iter().any(|f| f == "deepseek2")
    }
}

/// Resolve the effective context-shift setting for a request.
///
/// **Upstream:** `sched.go` `resolveContextShift`. An explicit request-level
/// `shift` win outright; otherwise ask the architecture.
pub fn resolve_context_shift(shift: Option<bool>, model: &ModelRef) -> bool {
    shift.unwrap_or_else(|| model.supports_context_shift())
}

/// Clamp a requested context to what the model was actually trained on.
///
/// **Upstream:** `sched.go` `effectiveContext`. `train_ctx == 0` mean unknown, so
/// no clamp. Asking for more context than the model was trained at does not give
/// you more usable context -- it give you a bigger KV cache full of positions the
/// model never learned.
pub fn effective_context(num_ctx: u32, train_ctx: u32) -> u32 {
    if train_ctx > 0 && num_ctx > train_ctx {
        train_ctx
    } else {
        num_ctx
    }
}

/// Total tokens the runner must hold across **all** parallel sequences.
///
/// **Upstream:** `sched.go` `effectiveLlamaServerContext` --
/// `effectiveModelContext(numCtx, f) * max(numParallel, 1)`.
///
/// This is the number that drive VRAM prediction, and it is easy to get wrong:
/// `num_ctx` is **per sequence**, but the KV cache must hold `num_parallel` of
/// them at once. Forget the multiply and you under-predict by 4x on a default
/// setup, then OOM on the second concurrent request.
pub fn effective_llama_server_context(num_ctx: u32, train_ctx: u32, num_parallel: u32) -> u64 {
    effective_context(num_ctx, train_ctx) as u64 * num_parallel.max(1) as u64
}

/// Does `predicted` fit inside `available`, keeping the standard 20% headroom?
///
/// **Upstream:** the `predicted > available*80/100` test repeated in `load`,
/// `bestSingleGPUFit`, `generationBatchFits` and `disableMmapForHostPressure`.
/// Written once here so a future change cannot drift the four apart.
///
/// **What would make this wrong:** integer order of operations. `available *
/// 80 / 100` (multiply first) is what upstream do; `available / 100 * 80` would
/// round differently and disagree at the boundary. Keep the multiply first.
pub fn fits_with_headroom(predicted: u64, available: u64) -> bool {
    predicted <= available.saturating_mul(FIT_HEADROOM_PERCENT) / 100
}

// ---------------------------------------------------------------------------
// How much memory got -- the arithmetic every fit decision stand on
// ---------------------------------------------------------------------------

/// The three numbers `availableMemoryForLoad` return, given names.
///
/// **Upstream:** the `(available, gpuFree uint64, systemLimited bool)` tuple of
/// `sched.go` `availableMemoryForLoad`. A struct instead of a tuple because
/// `available` and `gpu_free` are both `u64` and swapping them silently is
/// exactly the class of bug this module cannot afford.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Availability {
    /// Bytes we are willing to plan against. **Use this one for fit decisions.**
    pub available: u64,
    /// Raw sum of what the devices reported free. Diagnostics only -- planning
    /// against this on an iGPU over-promise by the whole system RAM.
    pub gpu_free: u64,
    /// True when `available` got cut down by system free memory rather than by
    /// the devices. Tells a human why the number is smaller than the GPU claim.
    pub system_limited: bool,
}

/// How much we may plan to use across a set of devices.
///
/// **Upstream:** `sched.go` `availableMemoryForLoad`.
///
/// ## The iGPU trap, spelled out
///
/// On an integrated GPU the "device free memory" is a **static or slowly
/// refreshed baseline over the same physical RAM the OS is using**. Upstream's
/// own comment: *"updateFreeSpace has already subtracted known Ollama runner
/// allocations from that baseline. Current system free memory is a separate live
/// measurement that already includes those loaded runners, so use the smaller
/// value for shared-memory GPUs without discounting discrete VRAM."*
///
/// So: when system free is **lower** than the shared-GPU free, we return
/// `discrete_free + system_free` -- the discrete cards keep their own real VRAM,
/// and the shared portion get replaced by the honest live number. Any other
/// combination either double-count RAM (promise 300 GB on a 96 GB Mac) or throw
/// away real discrete VRAM on a mixed box.
pub fn available_memory_for_load(system_info: &SystemInfo, gpus: &[DeviceInfo]) -> Availability {
    let mut gpu_free: u64 = 0;
    let mut shared_gpu_free: u64 = 0;
    let mut discrete_gpu_free: u64 = 0;

    for gpu in gpus {
        gpu_free = gpu_free.saturating_add(gpu.free_memory);
        if gpu.integrated {
            shared_gpu_free = shared_gpu_free.saturating_add(gpu.free_memory);
        } else {
            discrete_gpu_free = discrete_gpu_free.saturating_add(gpu.free_memory);
        }
    }

    if system_info.free_memory > 0 && shared_gpu_free > 0 && system_info.free_memory < shared_gpu_free
    {
        return Availability {
            available: discrete_gpu_free.saturating_add(system_info.free_memory),
            gpu_free,
            system_limited: true,
        };
    }

    Availability {
        available: gpu_free,
        gpu_free,
        system_limited: false,
    }
}

/// What one device can honestly offer.
///
/// **Upstream:** `sched.go` `availableMemoryForGPU`. Same iGPU clamp as
/// [`available_memory_for_load`], for a single card.
pub fn available_memory_for_gpu(system_info: &SystemInfo, gpu: &DeviceInfo) -> u64 {
    if gpu.integrated && system_info.free_memory > 0 && system_info.free_memory < gpu.free_memory {
        return system_info.free_memory;
    }
    gpu.free_memory
}

/// Narrow the device list down to whatever `main_gpu` pinned, if anything.
///
/// **Upstream:** `sched.go` `gpusForPlacement`. An out-of-range `main_gpu` is
/// **ignored rather than rejected** -- upstream pass the value straight through
/// to llama-server and let it complain. We keep that: a bad index should not fail
/// a request the backend might still handle.
pub fn gpus_for_placement<'a>(gpus: &'a [DeviceInfo], opts: &Options) -> &'a [DeviceInfo] {
    if let Some(main) = opts.runner.main_gpu
        && main >= 0
        && (main as usize) < gpus.len()
    {
        let i = main as usize;
        return &gpus[i..i + 1];
    }
    gpus
}

/// [`available_memory_for_load`], but honouring an explicit `main_gpu`.
///
/// **Upstream:** `sched.go` `availableMemoryForPlacement`.
pub fn available_memory_for_placement(
    system_info: &SystemInfo,
    gpus: &[DeviceInfo],
    opts: &Options,
) -> Availability {
    let placement = gpus_for_placement(gpus, opts);
    if placement.len() == 1 && opts.runner.main_gpu.is_some() {
        let gpu_free = placement[0].free_memory;
        let available = available_memory_for_gpu(system_info, &placement[0]);
        return Availability {
            available,
            gpu_free,
            system_limited: available < gpu_free,
        };
    }
    available_memory_for_load(system_info, placement)
}

/// Group devices by backend, keeping first-seen order.
///
/// **Upstream:** `ml/device.go` `ByLibrary`.
///
/// **Order is load-bearing**, which is why this is not a `HashMap`: placement
/// pick the *first* group that ties on merit, so shuffling the groups shuffle
/// which card a model land on between runs. `IndexMap` preserve insertion order
/// the way Go's parallel `libs []string` slice does.
pub fn by_library(gpus: &[DeviceInfo]) -> Vec<Vec<DeviceInfo>> {
    let mut groups: IndexMap<String, Vec<DeviceInfo>> = IndexMap::new();
    for gpu in gpus {
        groups
            .entry(gpu.library().to_string())
            .or_default()
            .push(gpu.clone());
    }
    groups.into_values().collect()
}

/// Is there at least one discrete card in here?
///
/// **Upstream:** `sched.go` `hasDiscreteGPU`.
pub fn has_discrete_gpu(gpus: &[DeviceInfo]) -> bool {
    gpus.iter().any(|g| !g.integrated)
}

/// Are they **all** discrete? Empty list is `false`.
///
/// **Upstream:** `sched.go` `allDiscreteGPUs`. The empty-list-is-false part is
/// not incidental: an empty list mean CPU-only, and the mmap host-pressure
/// heuristic must not fire there.
pub fn all_discrete_gpus(gpus: &[DeviceInfo]) -> bool {
    !gpus.is_empty() && gpus.iter().all(|g| !g.integrated)
}

/// Any device on this backend? Case-insensitive, like upstream's
/// `strings.EqualFold`.
///
/// **Upstream:** `sched.go` `hasDeviceLibrary`.
pub fn has_device_library(gpus: &[DeviceInfo], library: &str) -> bool {
    gpus.iter().any(|g| g.library().eq_ignore_ascii_case(library))
}

/// Every device on this backend? Empty list is `false`.
///
/// **Upstream:** `sched.go` `allDevicesLibrary`.
pub fn all_devices_library(gpus: &[DeviceInfo], library: &str) -> bool {
    !gpus.is_empty()
        && gpus
            .iter()
            .all(|g| g.library().eq_ignore_ascii_case(library))
}

/// Is `candidate` a better single-GPU home than `current`?
///
/// **Upstream:** `sched.go` `betterPlacementGPU`. **Discrete beat integrated
/// outright, no matter how much more memory the integrated one claim** -- a
/// discrete card with 10 GB win over an iGPU claiming 32 GB, because the iGPU's
/// 32 GB is system RAM it is sharing with everything else and its bandwidth is a
/// fraction. Only when both are the same kind does free memory decide.
pub fn better_placement_gpu(
    candidate: &DeviceInfo,
    candidate_available: u64,
    current: &DeviceInfo,
    current_available: u64,
) -> bool {
    if candidate.integrated != current.integrated {
        return !candidate.integrated;
    }
    candidate_available > current_available
}

/// Same comparison, one backend group against another.
///
/// **Upstream:** `sched.go` `betterPlacementGroup`. A group containing any
/// discrete card beat an all-integrated group, then total available decide.
pub fn better_placement_group(
    candidate: &[DeviceInfo],
    candidate_available: u64,
    current: &[DeviceInfo],
    current_available: u64,
) -> bool {
    let candidate_discrete = has_discrete_gpu(candidate);
    let current_discrete = has_discrete_gpu(current);
    if candidate_discrete != current_discrete {
        return candidate_discrete;
    }
    candidate_available > current_available
}

/// Pick the backend group with the most room, discrete-first.
///
/// **Upstream:** `sched.go` `bestGPUGroupByAvailableMemory`. Used when nothing
/// fit on one card, so the model has to be split -- and a split may only span
/// devices of **one backend**, since you cannot shard a model across CUDA and
/// ROCm.
pub fn best_gpu_group_by_available_memory(
    system_info: &SystemInfo,
    groups: &[Vec<DeviceInfo>],
) -> Vec<DeviceInfo> {
    let mut best: Option<(&Vec<DeviceInfo>, u64)> = None;
    for group in groups {
        let available = available_memory_for_load(system_info, group).available;
        let better = match best {
            None => true,
            Some((cur, cur_avail)) => better_placement_group(group, available, cur, cur_avail),
        };
        if better {
            best = Some((group, available));
        }
    }
    best.map(|(g, _)| g.clone()).unwrap_or_default()
}

/// Find the single best card the model actually fit on.
///
/// **Upstream:** `sched.go` `bestSingleGPUFit`. Returns `None` when nothing fit
/// with headroom, which is the signal to fall back to a multi-GPU split.
///
/// Compacting onto one card when possible is not a micro-optimisation: a split
/// model pay inter-device transfer on every single token, so one 20 GB card beat
/// two 10 GB cards for a model that fit on either.
pub fn best_single_gpu_fit(
    system_info: &SystemInfo,
    groups: &[Vec<DeviceInfo>],
    predicted_vram: u64,
) -> Option<(DeviceInfo, u64)> {
    let mut best: Option<(DeviceInfo, u64)> = None;
    for group in groups {
        for candidate in group {
            let candidate_available = available_memory_for_gpu(system_info, candidate);
            if !fits_with_headroom(predicted_vram, candidate_available) {
                continue;
            }
            let better = match &best {
                None => true,
                Some((cur, cur_avail)) => {
                    better_placement_gpu(candidate, candidate_available, cur, *cur_avail)
                }
            };
            if better {
                best = Some((candidate.clone(), candidate_available));
            }
        }
    }
    best
}

/// Honour an explicit `main_gpu` index, searching every backend group for it.
///
/// **Upstream:** `sched.go` `bestExplicitMainGPU`.
///
/// The subtlety: `main_gpu` is an index **within a backend group**, not a global
/// index, because that is what llama-server mean by it. So `main_gpu = 1` with
/// CUDA:[0] and ROCm:[0,1] present resolve to ROCm's device 1 -- the CUDA group
/// is simply too short. Returns `None` if no group is long enough, and the caller
/// then fall back to a split.
pub fn best_explicit_main_gpu(
    system_info: &SystemInfo,
    groups: &[Vec<DeviceInfo>],
    main_gpu: i32,
) -> Option<(DeviceInfo, u64)> {
    if main_gpu < 0 {
        return None;
    }
    let idx = main_gpu as usize;
    let mut best: Option<(DeviceInfo, u64)> = None;
    for group in groups {
        let Some(candidate) = group.get(idx) else {
            continue;
        };
        let candidate_available = available_memory_for_gpu(system_info, candidate);
        let better = match &best {
            None => true,
            Some((cur, cur_avail)) => {
                better_placement_gpu(candidate, candidate_available, cur, *cur_avail)
            }
        };
        if better {
            best = Some((candidate.clone(), candidate_available));
        }
    }
    best
}

/// Which device(s) this model should load onto, and the options to launch with.
///
/// **Upstream:** `sched.go` `selectLlamaServerPlacement`.
///
/// The decision ladder, in order:
///
/// 1. **One device or CPU-only (`num_gpu == 0`)** -- nothing to choose, pass
///    through untouched (`main_gpu` stay exactly as the caller set it).
/// 2. **Explicit `main_gpu`** -- honour it if any backend group is long enough;
///    otherwise fall through to a split and let llama-server see the raw value.
/// 3. **`sched_spread` off and we got a prediction** -- compact onto the single
///    best-fitting card.
/// 4. **Otherwise** -- best backend group, split across it.
///
/// When a single device is chosen, `main_gpu` is rewritten to `0`, because the
/// returned device list has exactly one entry and llama-server index into *that*
/// list, not into the original one. Forget this rewrite and a two-GPU box send
/// the model to the wrong card.
///
/// `sched_spread` come in as a parameter rather than being read off the
/// environment here -- upstream call `envconfig.SchedSpread()` inline, but a pure
/// parameter keep this function testable without touching process env, which is
/// what upstream's own test has to fight with `t.Setenv`.
pub fn select_llama_server_placement(
    system_info: &SystemInfo,
    gpus: &[DeviceInfo],
    predicted_vram: u64,
    opts: &Options,
    sched_spread: bool,
) -> (Vec<DeviceInfo>, Options) {
    let mut launch_opts = opts.clone();
    if gpus.len() <= 1 || opts.runner.num_gpu == 0 {
        return (gpus.to_vec(), launch_opts);
    }

    let groups = by_library(gpus);
    if groups.is_empty() {
        return (gpus.to_vec(), launch_opts);
    }

    if let Some(main) = opts.runner.main_gpu {
        return match best_explicit_main_gpu(system_info, &groups, main) {
            Some((gpu, _available)) => {
                launch_opts.runner.main_gpu = Some(0);
                (vec![gpu], launch_opts)
            }
            None => (
                best_gpu_group_by_available_memory(system_info, &groups),
                launch_opts,
            ),
        };
    }

    if !sched_spread
        && predicted_vram > 0
        && let Some((gpu, _available)) = best_single_gpu_fit(system_info, &groups, predicted_vram)
    {
        launch_opts.runner.main_gpu = Some(0);
        return (vec![gpu], launch_opts);
    }

    (
        best_gpu_group_by_available_memory(system_info, &groups),
        launch_opts,
    )
}

/// Did the user ask for a partial GPU offload outright?
///
/// **Upstream:** `sched.go` `explicitPartialGPUOffload`. `num_gpu` count
/// *layers*, and a full offload need `block_count + 1` of them (every block plus
/// the output layer), so anything less is a deliberate partial offload. That
/// matter because a deliberate partial offload must **not** trigger the
/// pre-flight evict -- the user already said they are happy to spill to CPU.
pub fn explicit_partial_gpu_offload(opts: &Options, block_count: u64) -> bool {
    if opts.runner.num_gpu <= 0 || block_count == 0 {
        return false;
    }
    (opts.runner.num_gpu as u64) < block_count + 1
}

// ---------------------------------------------------------------------------
// Automatic generation batch -- how many tokens per prompt-eval pass
// ---------------------------------------------------------------------------

/// The batch a context window of this size *want*, before affordability.
///
/// **Upstream:** `sched.go` `generationBatchForContext`. Bigger context = more
/// prompt to chew = a bigger batch pay off. The two thresholds (4096, 32768) are
/// upstream's, empirical, not derived.
pub fn generation_batch_for_context(effective_ctx: u64) -> u32 {
    if effective_ctx > 32768 {
        GENERATION_BATCH_LARGE
    } else if effective_ctx > 4096 {
        GENERATION_BATCH_MEDIUM
    } else {
        GENERATION_BATCH_DEFAULT
    }
}

/// Step one rung down the batch ladder.
///
/// **Upstream:** `sched.go` `nextLowerGenerationBatch`. Note it never go below
/// 512 -- [`GENERATION_BATCH_CONSTRAINED`] (256) is reached only by the CUDA
/// no-flash-attention path, never by stepping down.
pub fn next_lower_generation_batch(batch: u32) -> u32 {
    if batch > GENERATION_BATCH_MEDIUM {
        GENERATION_BATCH_MEDIUM
    } else {
        GENERATION_BATCH_DEFAULT
    }
}

/// Extra transient VRAM a bigger batch cost, on top of weights and KV cache.
///
/// **Upstream:** `sched.go` `generationBatchSurcharge` -- `2 * format.GibiByte`
/// at 2048, `768 * format.MebiByte` at 1024, nothing below. **Gibi/Mebi here**,
/// unlike [`MMAP_HOST_PRESSURE_HEADROOM_MIN`] which is decimal. Upstream mix the
/// two units within this one file; both are reproduced faithfully.
pub fn generation_batch_surcharge(batch: u32) -> u64 {
    if batch >= GENERATION_BATCH_LARGE {
        2 * GIBIBYTE
    } else if batch >= GENERATION_BATCH_MEDIUM {
        768 * MEBIBYTE
    } else {
        0
    }
}

/// Same, but zero for a non-completion (embedding) model.
///
/// **Upstream:** `sched.go` `generationBatchSurchargeForCompletion`. An embedding
/// model do one forward pass and stop -- no generation loop, so no generation
/// batch to pay for.
pub fn generation_batch_surcharge_for_completion(completion: bool, batch: u32) -> u64 {
    if completion {
        generation_batch_surcharge(batch)
    } else {
        0
    }
}

/// Is predicted VRAM comfortably under the bar this batch tier demand?
///
/// **Upstream:** `sched.go` `generationBatchHasHeadroom`. The bigger the batch,
/// the more slack we insist on **before** committing: 60% for 2048, 75% for 1024,
/// no extra bar below that. This is deliberately stricter than the plain 80% fit
/// check, because a batch that turn out too big OOM *mid-generation*, long after
/// the load "succeeded".
pub fn generation_batch_has_headroom(batch: u32, predicted_vram: u64, available_memory: u64) -> bool {
    if batch >= GENERATION_BATCH_LARGE {
        predicted_vram
            <= available_memory.saturating_mul(GENERATION_BATCH_LARGE_HEADROOM_PERCENT) / 100
    } else if batch >= GENERATION_BATCH_MEDIUM {
        predicted_vram
            <= available_memory.saturating_mul(GENERATION_BATCH_MEDIUM_HEADROOM_PERCENT) / 100
    } else {
        true
    }
}

/// Can we afford this batch at all?
///
/// **Upstream:** `sched.go` `generationBatchFits`.
///
/// **Unknown means yes.** `predicted == 0` or `available == 0` mean we have no
/// measurement, and upstream choose to proceed rather than pessimise -- see the
/// "medium context uses 1024 with unknown memory" case in its own test. That is a
/// real behavioural choice, not an oversight: refusing to promote whenever the
/// numbers are missing would permanently pin every unmeasured setup at 512.
pub fn generation_batch_fits(batch: u32, predicted_vram: u64, available_memory: u64) -> bool {
    if predicted_vram == 0 || available_memory == 0 {
        return true;
    }
    let threshold = available_memory.saturating_mul(FIT_HEADROOM_PERCENT) / 100;
    if predicted_vram > threshold {
        return false;
    }
    if !generation_batch_has_headroom(batch, predicted_vram, available_memory) {
        return false;
    }
    generation_batch_surcharge(batch) <= threshold - predicted_vram
}

/// Is this a small CUDA card running without flash attention?
///
/// **Upstream:** `sched.go` `constrainedCUDAWithoutFlashAttention`. Without flash
/// attention the attention scratch buffer scale with `batch * context`, so an
/// 8 GiB card past 4096 context genuinely cannot afford the normal batch.
///
/// The memory pick is subtle and worth keeping: it use `free_memory`, but fall
/// back to `total_memory` when free is 0 **or when total is smaller than free**
/// -- a nonsense reading some drivers really do produce, and taking the smaller
/// of the two is the safe response.
pub fn constrained_cuda_without_flash_attention(effective_ctx: u64, gpus: &[DeviceInfo]) -> bool {
    if effective_ctx <= 4096 {
        return false;
    }
    gpus.iter().any(|gpu| {
        if gpu.library() != "CUDA" {
            return false;
        }
        let mut memory = gpu.free_memory;
        if memory == 0 || (gpu.total_memory > 0 && gpu.total_memory < memory) {
            memory = gpu.total_memory;
        }
        memory > 0 && memory <= 8 * GIBIBYTE
    })
}

/// Any CUDA device present?
///
/// **Upstream:** `sched.go` `hasCUDADevice`. Case-**sensitive** on purpose:
/// upstream compare `gpu.Library == "CUDA"` here while using `EqualFold`
/// elsewhere in the same file. Kept as-is; the oracle wins even when it is
/// inconsistent with itself.
pub fn has_cuda_device(gpus: &[DeviceInfo]) -> bool {
    gpus.iter().any(|g| g.library() == "CUDA")
}

/// Choose the prompt-eval batch size automatically.
///
/// **Upstream:** `sched.go` `automaticGenerationBatch`.
///
/// Two separate regimes, and the CUDA one short-circuit the whole ladder:
///
/// * **Flash attention disabled AND a CUDA card present** -- 256 on a constrained
///   card, 512 otherwise. No promotion at all, whatever the memory say.
/// * **Otherwise** -- start from what the context want, then step down until it
///   fit.
pub fn automatic_generation_batch(
    effective_ctx: u64,
    predicted_vram: u64,
    available_memory: u64,
    flash_attention: FlashAttentionType,
    gpus: &[DeviceInfo],
) -> u32 {
    if flash_attention == FlashAttentionType::Disabled && has_cuda_device(gpus) {
        if constrained_cuda_without_flash_attention(effective_ctx, gpus) {
            return GENERATION_BATCH_CONSTRAINED;
        }
        return GENERATION_BATCH_DEFAULT;
    }

    let mut batch = generation_batch_for_context(effective_ctx);
    while batch > GENERATION_BATCH_DEFAULT
        && !generation_batch_fits(batch, predicted_vram, available_memory)
    {
        batch = next_lower_generation_batch(batch);
    }
    batch
}

/// The next context size down the automatic ladder, for the OOM retry.
///
/// **Upstream:** `sched.go` `nextLowerAutoNumCtx`. `None` mean there is no
/// smaller rung -- 4096 is the floor, and below that a retry would not be worth
/// the reload. Only ever applied to a context KOPITIAM *chose*, never to one the
/// user asked for: see [`LlmRequest::num_ctx_auto`].
pub fn next_lower_auto_num_ctx(num_ctx: u32) -> Option<u32> {
    if num_ctx > 32768 {
        Some(32768)
    } else if num_ctx > 4096 {
        Some(4096)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// mmap heuristics
// ---------------------------------------------------------------------------

/// Why we turned mmap off by default, or `None` if we did not.
///
/// **Upstream:** `sched.go` `disableMmapDefaultReason`, which return a reason
/// string (`""` for "keep the default"). Modelled as an enum here so the reasons
/// cannot be typo'd, with [`MmapDisableReason::as_str`] giving upstream's exact
/// strings back for logging and for test comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapDisableReason {
    /// No GPU in play at all -- pure CPU inference.
    Cpu,
    /// Windows + CUDA. Upstream disable unconditionally here; the Windows page
    /// cache and CUDA pinned memory interact badly enough that mmap is a net
    /// loss.
    WindowsCuda,
    /// Metal with only part of the model on the GPU.
    MetalPartialOffload,
}

impl MmapDisableReason {
    /// Upstream's exact reason string.
    pub fn as_str(&self) -> &'static str {
        match self {
            MmapDisableReason::Cpu => "cpu",
            MmapDisableReason::WindowsCuda => "windows_cuda",
            MmapDisableReason::MetalPartialOffload => "metal_partial_offload",
        }
    }
}

/// Should mmap be off by default for this load?
///
/// **Upstream:** `sched.go` `disableMmapDefaultReason`. `goos` come in as a
/// parameter exactly like upstream's, so a Linux CI box can test the Windows and
/// Darwin branches -- do not "simplify" it to `cfg!(target_os)`.
///
/// **An explicit `use_mmap` always win**, set either way. This function only
/// pick a *default*.
///
/// The Metal partial-offload branch has two shapes, and both matter: an explicit
/// layer count below `block_count + 1`, or an automatic (`num_gpu < 0`) load
/// whose prediction exceed what is available. In both cases part of the model
/// live on the CPU, and mmap-ing a file the GPU must then copy out of is slower
/// than just reading it in.
pub fn disable_mmap_default_reason(
    goos: &str,
    opts: &Options,
    gpus: &[DeviceInfo],
    block_count: u64,
    predicted_vram: u64,
    available_vram: u64,
) -> Option<MmapDisableReason> {
    if opts.runner.use_mmap.is_some() {
        return None;
    }
    if opts.runner.num_gpu == 0 || gpus.is_empty() || all_devices_library(gpus, "cpu") {
        return Some(MmapDisableReason::Cpu);
    }
    if goos == "windows" && has_device_library(gpus, "cuda") {
        return Some(MmapDisableReason::WindowsCuda);
    }
    if has_device_library(gpus, "metal") {
        if opts.runner.num_gpu > 0
            && block_count > 0
            && (opts.runner.num_gpu as u64) < block_count + 1
        {
            return Some(MmapDisableReason::MetalPartialOffload);
        }
        if opts.runner.num_gpu < 0
            && predicted_vram > 0
            && available_vram > 0
            && predicted_vram > available_vram
        {
            return Some(MmapDisableReason::MetalPartialOffload);
        }
    }
    None
}

/// How much host RAM we insist on keeping free before mmap-ing another model.
///
/// **Upstream:** `sched.go` `mmapHostPressureHeadroom` -- `max(8 GB,
/// total/10)`. The 8 GB floor exist so a small box does not compute a
/// meaninglessly tiny headroom.
pub fn mmap_host_pressure_headroom(total_memory: u64) -> u64 {
    if total_memory == 0 {
        return MMAP_HOST_PRESSURE_HEADROOM_MIN;
    }
    MMAP_HOST_PRESSURE_HEADROOM_MIN.max(total_memory / 10)
}

/// Should mmap be dropped because the **host** is under memory pressure?
///
/// **Upstream:** `sched.go` `disableMmapForHostPressure`. Linux only -- upstream
/// restrict it explicitly, and the comment call it "the Linux pressure heuristic
/// restored", i.e. it was reverted once already. Do not generalise it to other
/// platforms without evidence.
///
/// Every one of these five bail-outs is load-bearing:
///
/// * explicit `use_mmap` -> the user decided, leave it;
/// * not Linux -> out of scope;
/// * unknown model size or unknown system free -> no basis to judge;
/// * any integrated GPU -> shared memory make the arithmetic meaningless;
/// * **VRAM already tight** -> upstream's own comment: *"If VRAM is already
///   tight, disabling mmap can make partial CPU offload worse by turning
///   file-backed mappings into anonymous memory."* Anonymous pages cannot be
///   evicted to the file they came from, so on a tight box turning mmap off make
///   things strictly worse. This is the counter-intuitive one -- do not remove it
///   thinking it is redundant with the check above.
#[allow(clippy::too_many_arguments)]
pub fn disable_mmap_for_host_pressure(
    goos: &str,
    opts: &Options,
    system_info: &SystemInfo,
    gpus: &[DeviceInfo],
    model_size: u64,
    loaded_mmap_size: u64,
    predicted_vram: u64,
    available_vram: u64,
) -> bool {
    if opts.runner.use_mmap.is_some()
        || goos != "linux"
        || model_size == 0
        || system_info.free_memory == 0
        || !all_discrete_gpus(gpus)
    {
        return false;
    }

    if predicted_vram == 0 || available_vram == 0 || !fits_with_headroom(predicted_vram, available_vram)
    {
        return false;
    }

    let pressure = model_size
        .saturating_add(loaded_mmap_size)
        .saturating_add(mmap_host_pressure_headroom(system_info.total_memory));
    system_info.free_memory < pressure
}

// ---------------------------------------------------------------------------
// VRAM prediction and recovery
// ---------------------------------------------------------------------------

/// Rough VRAM a llama-server load will take: weights on disk plus an f16 KV
/// cache.
///
/// **Upstream:** `llm/llama_server.go` `PredictServerVRAM`, translated verbatim:
///
/// ```text
/// kv_cache = 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (f16)
/// ```
///
/// `head_dim` is `embedding_length / head_count_max`, and `kv_heads` use
/// `head_count_kv_min` -- the **minimum** across layers, matching upstream. A
/// `kv_heads` of 0 is clamped to 1 (upstream does the same) so a model with
/// missing metadata predict something rather than zero.
///
/// `weights` is the **file size**, passed in rather than `os.Stat`-ed, since this
/// crate does no I/O. Pass 0 if unknown; upstream's `os.Stat` failure path
/// produce exactly that.
///
/// **This is deliberately crude, and upstream know it.** It ignores quantisation
/// of the cache, ignores the compute graph, and treats the whole file as
/// GPU-resident. For finer arithmetic -- per-layer KV, and the partial vs. full
/// offload graph scratch -- the crate ship [`crate::memory::graph_size`], which
/// is the ported `fs/ggml.GraphSize`. **Known gap in that one:** several
/// architectures (qwen3, olmo3, nemotron_h, ...) fall through upstream's match
/// and get `partial_offload = full_offload = 0`, silently. If a caller feed
/// `graph_size` output into placement for such a model, the graph scratch is
/// costed at zero and the load can OOM after the fit check said yes. The
/// scheduler itself does **not** depend on `graph_size` -- it use this function,
/// exactly as upstream's scheduler does -- so that gap does not degrade
/// scheduling today. It would the moment somebody switch the estimator over.
pub fn predict_server_vram(weights_file_size: u64, kv: &Kv, num_ctx: u64) -> u64 {
    let layers = kv.block_count();
    let mut kv_heads = kv.head_count_kv_min();
    if kv_heads == 0 {
        kv_heads = 1;
    }
    // `head_dim = 0` when head_count_max is 0 -- upstream leave `headDim` at its
    // Go zero value in that case, which make the whole KV term vanish and leave
    // the prediction as just the weights. Faithful, and the only safe answer
    // when the metadata cannot say how wide a head is.
    let head_dim = kv.embedding_length().checked_div(kv.head_count_max()).unwrap_or(0);

    let kv_cache = 2u64
        .saturating_mul(layers)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(num_ctx)
        .saturating_mul(2);

    weights_file_size.saturating_add(kv_cache)
}

/// Has enough VRAM come back after a runner exit to stop waiting?
///
/// **Upstream:** the convergence test inside `sched.go` `waitForVRAMRecovery`:
/// `float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75`.
///
/// Why this exists at all, and it is the whole hard-won bit: **driver free-memory
/// reporting lag seconds behind a process exit.** Load the next model
/// immediately and it read a stale "still full" number, then quietly loads with
/// far fewer GPU layers -- or drops to CPU entirely -- and nobody can see why it
/// got slow. So the caller poll every [`VRAM_RECOVERY_POLL_INTERVAL`] until this
/// return `true` or [`VRAM_RECOVERY_TIMEOUT`] elapse, then proceed regardless.
///
/// The subtraction is done saturating: `free_now` below `free_before` mean some
/// other process grabbed memory in the meantime, and upstream's unsigned
/// subtraction would wrap to a colossal number and declare victory. **That is a
/// real bug in upstream** (`freeMemoryNow-freeMemoryBefore` on `uint64`, guarded
/// only in the logging branch, not in the comparison). We diverge here on
/// purpose: saturating to 0 keep waiting, which is the safe reading.
pub fn vram_recovery_converged(free_before: u64, free_now: u64, runner_vram_size: u64) -> bool {
    let recovered = free_now.saturating_sub(free_before);
    recovered as f32 > runner_vram_size as f32 * VRAM_RECOVERY_FRACTION
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

/// One request waiting for, or holding, a runner.
///
/// **Upstream:** `sched.go` `LlmRequest`, minus the plumbing that has no meaning
/// here: `ctx` (cancellation is the caller's), `successCh` / `errCh` (the caller
/// get a return value instead), and `schedAttempts` is kept because it is real
/// state, not plumbing.
///
/// ## The three `*_auto` flags, and why they are not cosmetic
///
/// `num_ctx_auto`, `num_batch_auto` and `use_mmap_auto` record **who chose the
/// value** -- KOPITIAM or the user. That distinction drive two decisions that
/// would otherwise be wrong:
///
/// * [`Scheduler::needs_reload`] must **not** reload a runner just because our
///   own automatic context differ from our own automatic context computed a
///   moment later under different free memory. A value we picked is not a value
///   the user asked for, so it does not count as a mismatch.
/// * The OOM retry may only step the context **down** when we picked it. Halving
///   a context the user explicitly asked for would silently give them something
///   other than what they requested -- see
///   [`Scheduler::on_load_failed`].
#[derive(Debug, Clone, Default)]
pub struct LlmRequest {
    /// Which model.
    pub model: ModelRef,
    /// Requested options. Mutated by the scheduler where upstream mutate
    /// `req.opts` -- context clamping, automatic batch, automatic mmap.
    pub opts: Options,
    /// Per-request keep-alive override. `None` mean use
    /// [`SchedulerConfig::keep_alive`].
    pub session_duration: Option<Expiry>,
    /// How many times this request has been through the scheduling loop.
    ///
    /// **Upstream:** `LlmRequest.schedAttempts`.
    pub sched_attempts: u32,
    /// Set once an evict-all-and-retry has already been spent on this request.
    ///
    /// **Upstream:** `LlmRequest.oomRetryAttempted`, whose comment is the whole
    /// point: *"Prevents infinite retry on persistent load failures."* Without
    /// it, a model too big for the box evict every other model, fail, evict
    /// again, forever.
    pub oom_retry_attempted: bool,
    /// `num_ctx` came from KOPITIAM's VRAM-tier default, not from the user.
    pub num_ctx_auto: bool,
    /// `num_batch` came from our defaults, not from the user.
    pub num_batch_auto: bool,
    /// `use_mmap` was derived by the scheduler, not requested.
    pub use_mmap_auto: bool,
    /// Resolved context-shift setting for this load.
    pub context_shift: bool,
    /// The raw request-level shift override, before resolution. `None` mean "ask
    /// the architecture".
    pub shift: Option<bool>,
}

impl LlmRequest {
    /// A request for this model with default options. Everything else is
    /// [`Default`].
    pub fn new(model: ModelRef, opts: Options) -> Self {
        Self {
            model,
            opts,
            ..Default::default()
        }
    }

    /// The scheduler map key for this request's model.
    pub fn key(&self) -> String {
        self.model.scheduler_key()
    }
}

/// Apply the two floors every request get before it is scheduled.
///
/// **Upstream:** the top of `sched.go` `getRunner`.
///
/// * `num_ctx < 4` -> 4. A window under four tokens cannot hold a BOS plus
///   anything.
/// * a vision model -> at least 2048. One image expand into hundreds of tokens.
///
/// Call this **before** [`Scheduler::enqueue`] or
/// [`Scheduler::acquire_if_loaded`], because both compare options against a
/// loaded runner's, and comparing an unclamped request against a clamped runner
/// report a spurious reload.
pub fn clamp_request_options(model: &ModelRef, opts: &mut Options) {
    if opts.runner.num_ctx < MIN_NUM_CTX {
        opts.runner.num_ctx = MIN_NUM_CTX;
    }
    if model.has_capability(Capability::Vision) {
        opts.runner.num_ctx = opts.runner.num_ctx.max(VISION_MIN_NUM_CTX);
    }
}

// ---------------------------------------------------------------------------
// The loaded runner
// ---------------------------------------------------------------------------

/// A model that is resident right now, plus the bookkeeping that keep it there.
///
/// **Upstream:** `sched.go` `runnerRef`. The mutex is gone -- `&mut Scheduler`
/// is the lock (see the module header) -- and so is `expireTimer`, replaced by
/// the plain [`RunnerRef::expires_at`] deadline that [`Scheduler::take_expired`]
/// query.
pub struct RunnerRef {
    /// **In-flight requests holding this runner.** See the module header for the
    /// full invariant; the short version is that a runner with `ref_count > 0` is
    /// never unloaded, only marked to go when the last holder let go.
    pub ref_count: u32,
    /// The live runner.
    pub runner: Arc<dyn ModelRunner>,
    /// OS pid, or negative when there is no process.
    pub pid: i32,
    /// `true` only during the initial load, then `false` forever. Widens the
    /// health-ping timeout -- see [`NEEDS_RELOAD_LOADING_PING_TIMEOUT`].
    pub loading: bool,
    /// Devices this runner was placed on, recorded at load time.
    pub gpus: Vec<DeviceId>,
    /// All of `gpus` are discrete. Drives whether a VRAM recovery wait is even
    /// needed on unload.
    pub discrete_gpus: bool,
    /// GPU bytes, snapshotted at load.
    pub vram_size: u64,
    /// Total bytes (GPU + host), snapshotted at load.
    pub total_size: u64,
    /// How long this runner should linger idle before unloading.
    /// `Expiry::After(ZERO)` mean **go the moment you are idle**; that is how
    /// eviction is signalled.
    pub session_duration: Expiry,
    /// When the idle timer will fire. `None` mean not idle (or never armed).
    pub expires_at: Option<Instant>,
    /// The model, or `None` once unloaded.
    pub model: Option<ModelRef>,
    /// The options it was loaded with, or `None` once unloaded.
    pub options: Option<Options>,
    /// Convenience copy of `model.model_path`, kept after unload for sorting and
    /// logging.
    pub model_path: String,
    /// The scheduler map key.
    pub model_key: String,
    /// Parallel sequences this runner was configured for.
    pub num_parallel: u32,
    /// `num_ctx` was chosen by us.
    pub num_ctx_auto: bool,
    /// `num_batch` was chosen by us.
    pub num_batch_auto: bool,
    /// `use_mmap` was chosen by us.
    pub use_mmap_auto: bool,
    /// Context shift enabled for this runner.
    pub context_shift: bool,
    /// The model's trained context length, 0 if unknown.
    pub train_context: u32,
}

impl std::fmt::Debug for RunnerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Upstream: runnerRef.LogValue (sched.go). Same fields, same intent --
        // enough to identify the runner without dumping the whole model.
        f.debug_struct("RunnerRef")
            .field("model", &self.model_path)
            .field("key", &self.model_key)
            .field("ref_count", &self.ref_count)
            .field("size", &self.total_size)
            .field("vram", &self.vram_size)
            .field("parallel", &self.num_parallel)
            .field("pid", &self.pid)
            .field("gpus", &self.gpus)
            .finish()
    }
}

impl RunnerRef {
    /// How long to wait for this runner's health ping.
    ///
    /// **Upstream:** `sched.go` `needsReload`'s `timeout` local.
    pub fn ping_timeout(&self) -> Duration {
        if self.loading {
            NEEDS_RELOAD_LOADING_PING_TIMEOUT
        } else {
            NEEDS_RELOAD_PING_TIMEOUT
        }
    }

    /// Is anybody currently using this runner?
    pub fn is_idle(&self) -> bool {
        self.ref_count == 0
    }

    /// The sort key `ByDurationAndName` compare on.
    ///
    /// **Upstream:** `sched.go` `ByDurationAndName.Less`, which cast
    /// `sessionDuration` to `uint64` before comparing. That cast is not
    /// incidental: `KeepAlive()` return `math.MaxInt64` for "forever", so a
    /// keep-forever runner sort **last** and is evicted last. Reproduce the cast
    /// or a forever-runner become the first thing you throw away.
    fn duration_sort_key(&self) -> u64 {
        self.session_duration.as_nanos_i64() as u64
    }

    /// The name half of that sort key: model path, falling back to the map key.
    ///
    /// **Upstream:** the `n1 := a[i].modelPath; if n1 == "" { n1 = a[i].modelKey }`
    /// dance in `ByDurationAndName.Less`.
    fn name_sort_key(&self) -> &str {
        if self.model_path.is_empty() {
            &self.model_key
        } else {
            &self.model_path
        }
    }

    /// Does this runner still mmap its weights?
    ///
    /// **Upstream:** `sched.go` `runnerUsesMmap`. **Unset mean yes** -- mmap is
    /// the backend's default, so an absent override count as on.
    fn uses_mmap(&self) -> bool {
        match &self.options {
            None => true,
            Some(o) => o.runner.use_mmap.unwrap_or(true),
        }
    }

    /// Drop the runner and let go of everything it held.
    ///
    /// **Upstream:** `sched.go` `runnerRef.unload`. Clearing `model` and
    /// `options` is not tidiness -- upstream do it so a stale reference cannot
    /// keep a whole `Model` alive, and so `needs_reload` on a corpse return
    /// `true` (via the `options == None` check) rather than comparing rubbish.
    fn unload(&mut self) {
        self.expires_at = None;
        self.runner.unload();
        self.model = None;
        self.options = None;
        self.gpus.clear();
        self.context_shift = false;
    }
}

/// A point-in-time view of a loaded model, safe to hand out.
///
/// **Upstream:** `sched.go` `loadedModel` + `Scheduler.loadedModels()`, which
/// exists *"for status reporting without exposing the scheduler's internal
/// runner bookkeeping"*.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    /// The model.
    pub model: ModelRef,
    /// Total bytes, live from the runner where available.
    pub size: u64,
    /// GPU bytes, live from the runner where available.
    pub size_vram: u64,
    /// Context window the runner settled on.
    pub context_length: u32,
    /// When it will be unloaded if nothing touch it. `None` mean "forever"
    /// (keep-alive is [`Expiry::Never`]).
    ///
    /// **Upstream** estimate this from `sessionDuration` when `expiresAt` is
    /// still the zero value, because *"the scheduler waits to set expiresAt, so a
    /// model that is still loading may have the zero value"*. Same here.
    pub expires_at: Option<Instant>,
}

/// A runner that has just been removed from the scheduler and needs closing.
///
/// Returned by [`Scheduler::take_expired`]. The scheduler has **already**
/// dropped it from its map, so the caller now own the shutdown.
#[derive(Clone)]
pub struct Unloaded {
    /// The key it was under.
    pub key: String,
    /// The runner. [`ModelRunner::unload`] has already been called on it by the
    /// scheduler; this handle is here so the caller can wait on the process,
    /// log, or re-check `has_exited`.
    pub runner: Arc<dyn ModelRunner>,
    /// GPU bytes it was holding -- feed this to [`vram_recovery_converged`].
    pub vram_size: u64,
    /// **Should the caller wait for VRAM to come back before loading the next
    /// model?**
    ///
    /// **Upstream:** the early-out at the top of `waitForVRAMRecovery` -- *"CPU,
    /// Metal and iGPUs don't need checking, so no waiting required"*. `false`
    /// here mean go straight ahead; `true` mean poll
    /// [`vram_recovery_converged`] every [`VRAM_RECOVERY_POLL_INTERVAL`] until it
    /// pass or [`VRAM_RECOVERY_TIMEOUT`] elapse.
    pub needs_vram_recovery_wait: bool,
}

impl std::fmt::Debug for Unloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Unloaded")
            .field("key", &self.key)
            .field("vram_size", &self.vram_size)
            .field("needs_vram_recovery_wait", &self.needs_vram_recovery_wait)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The knobs the scheduler read once and then hold.
///
/// **Upstream:** the `envconfig.*()` calls scattered through `sched.go`. Read
/// once here rather than per-decision, so a scheduling run cannot see the
/// environment change under it halfway through -- upstream re-read
/// `SchedSpread()` on every placement, which is a latent inconsistency we
/// deliberately close.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Hard cap on resident models. **`0` mean auto** -- derive
    /// [`DEFAULT_MODELS_PER_GPU`] per GPU on first use.
    ///
    /// **Upstream:** `envconfig.MaxRunners()` / `OLLAMA_MAX_LOADED_MODELS`.
    pub max_runners: u64,
    /// Sequences one runner serve at once. Clamped up to 1 at use.
    ///
    /// **Upstream:** `envconfig.NumParallel()` / `OLLAMA_NUM_PARALLEL`.
    pub num_parallel: u32,
    /// Default idle lifetime for a loaded model.
    ///
    /// **Upstream:** `envconfig.KeepAlive()` / `OLLAMA_KEEP_ALIVE`, default 5
    /// minutes. `Expiry::After(ZERO)` mean unload the instant the request end;
    /// [`Expiry::Never`] mean keep forever.
    pub keep_alive: Expiry,
    /// How long a load may stall before the caller gives up.
    ///
    /// **Upstream:** `envconfig.LoadTimeout()`. The scheduler itself never block,
    /// so this is carried for the caller's benefit -- **note the zero-value trap**:
    /// for load timeout `0` mean *wait forever*, the opposite of keep-alive.
    pub load_timeout: Expiry,
    /// Spread a model across every GPU instead of compacting onto the best one.
    ///
    /// **Upstream:** `envconfig.SchedSpread()` / `OLLAMA_SCHED_SPREAD`.
    pub sched_spread: bool,
    /// Extra per-GPU bytes to hold back, on top of the backend's own minimum.
    ///
    /// **Upstream:** `envconfig.GpuOverhead()` / `OLLAMA_GPU_OVERHEAD`.
    pub gpu_overhead: u64,
    /// Depth of the pending queue before [`SchedError::MaxQueue`].
    ///
    /// **Upstream:** `envconfig.MaxQueue()`, which size all four channels in
    /// `InitScheduler`.
    pub max_queue: u64,
}

impl Default for SchedulerConfig {
    /// Upstream's defaults with no environment set: auto runners, auto
    /// parallelism, 5-minute keep-alive, 5-minute load timeout, no spread, no
    /// extra overhead, 512-deep queue (`envconfig.MaxQueue()`'s default).
    fn default() -> Self {
        Self {
            max_runners: 0,
            num_parallel: 0,
            keep_alive: Expiry::After(Duration::from_secs(5 * 60)),
            load_timeout: Expiry::After(Duration::from_secs(5 * 60)),
            sched_spread: false,
            gpu_overhead: 0,
            max_queue: 512,
        }
    }
}

impl SchedulerConfig {
    /// Read every knob off an [`Env`].
    pub fn from_env(env: &Env) -> Self {
        Self {
            max_runners: env.max_runners(),
            num_parallel: env.num_parallel().min(u32::MAX as u64) as u32,
            keep_alive: env.keep_alive(),
            load_timeout: env.load_timeout(),
            sched_spread: env.sched_spread(),
            gpu_overhead: env.gpu_overhead(),
            max_queue: env.max_queue(),
        }
    }
}

// ---------------------------------------------------------------------------
// The decisions the state machine emit
// ---------------------------------------------------------------------------

/// Everything the caller need in order to actually perform a load.
///
/// **Upstream:** the locals computed in the first half of `sched.go` `load`,
/// before `newServerFn` is called. Everything after that call is the caller's
/// business, because it involve spawning something.
#[derive(Debug, Clone)]
pub struct LoadPlan {
    /// Scheduler map key to register the result under.
    pub key: String,
    /// Sequences this runner must serve. Already clamped for embedding models
    /// and for [`UNSAFE_PARALLEL_ARCHITECTURES`].
    pub num_parallel: u32,
    /// Idle lifetime to give the resulting runner.
    pub session_duration: Expiry,
    /// The device(s) to load onto, after placement.
    pub gpus: Vec<DeviceInfo>,
    /// Options to launch with. **Not the same object as the request's** -- in
    /// particular `main_gpu` may have been rewritten to index into `gpus`.
    pub launch_opts: Options,
    /// Must the model land **entirely** on the GPU(s)?
    ///
    /// `false` only when nothing else is loaded -- upstream's *"No models loaded.
    /// Load the model but prefer the best fit"* -- in which case a partial CPU
    /// offload is acceptable because evicting would not help anyway.
    pub require_full: bool,
    /// Predicted bytes, including the generation-batch surcharge.
    pub predicted_vram: u64,
    /// Aggregate context across all parallel sequences, in tokens.
    pub effective_num_ctx: u64,
    /// This model does completion (as opposed to embedding only).
    pub completion: bool,
    /// Resolved context-shift setting.
    pub context_shift: bool,
    /// Why mmap was defaulted off, if it was.
    pub mmap_disabled_reason: Option<MmapDisableReason>,
}

/// What the scheduler want done next for a pending request.
///
/// **Upstream:** the branches of the inner `for` loop in `sched.go`
/// `processPending`. Turned inside out: instead of the loop blocking on channels,
/// it hand the decision back and the caller drive.
#[derive(Clone)]
pub enum PendingAction {
    /// A usable runner is already loaded. Its refcount has **already been
    /// incremented** -- the caller now owe exactly one
    /// [`Scheduler::request_finished`].
    ///
    /// **Upstream:** `pending.useLoadedRunner(runner, s.finishedReqCh); break`.
    UseLoaded {
        /// Map key of the runner.
        key: String,
        /// The runner to serve the request with.
        runner: Arc<dyn ModelRunner>,
    },
    /// Go load it, then report back with [`Scheduler::runner_loaded`] or
    /// [`Scheduler::on_load_failed`].
    ///
    /// **Upstream:** `s.loadFn(pending, systemInfo, gpus, requireFull)`.
    Load(Box<LoadPlan>),
    /// Chase this one out first, then ask again.
    ///
    /// The runner has already been marked expire-immediately. If it was idle it
    /// will come back from the very next [`Scheduler::take_expired`]; if not, it
    /// go when its last request finish.
    ///
    /// **Upstream:** the `runnerToExpire` block -- stop the timer, zero the
    /// session duration, push to `expiredCh` if idle, then wait on `unloadedCh`.
    Evict {
        /// Map key of the victim.
        key: String,
    },
    /// Evict **everything else** and retry the load once. The OOM recovery path.
    ///
    /// **Upstream:** `s.evictAllAndWait(ctx, pendingKey)`. All the victims have
    /// already been marked; the listed keys are for the caller's logging.
    EvictAll {
        /// Keys marked for eviction.
        keys: Vec<String>,
    },
    /// Nothing to evict and nothing to load -- the state changed underneath us.
    /// Call [`Scheduler::next_action`] again.
    ///
    /// **Upstream:** *"runner to expire was nil, retrying"* -- the loaded runners
    /// unloaded in parallel while we were doing load calculations.
    Retry,
}

impl std::fmt::Debug for PendingAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Arc<dyn ModelRunner>` cannot derive Debug, and requiring `Debug` on
        // the trait would push that burden onto every implementor for no gain.
        match self {
            PendingAction::UseLoaded { key, .. } => {
                f.debug_struct("UseLoaded").field("key", key).finish()
            }
            PendingAction::Load(plan) => f.debug_tuple("Load").field(plan).finish(),
            PendingAction::Evict { key } => f.debug_struct("Evict").field("key", key).finish(),
            PendingAction::EvictAll { keys } => {
                f.debug_struct("EvictAll").field("keys", keys).finish()
            }
            PendingAction::Retry => f.write_str("Retry"),
        }
    }
}

/// Why a load failed, as far as the scheduler care.
///
/// **Upstream:** the error classification inside `sched.go` `load` --
/// `errors.Is(err, llm.ErrLoadRequiredFull)`, `llm.IsOutOfMemory(err)`, and
/// everything else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadFailure {
    /// The runner reported it could not fit fully on the GPU(s).
    ///
    /// **Upstream:** `llm.ErrLoadRequiredFull`.
    RequiredFullNotMet,
    /// The runner died allocating memory.
    ///
    /// **Upstream:** `llm.IsOutOfMemory(err)`.
    OutOfMemory(String),
    /// Anything else -- bad file, unsupported format, crash on startup.
    Other(String),
}

/// What to do about a failed load.
///
/// **Upstream:** the return value and side effects of the error branch of
/// `sched.go` `load`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadFailureAction {
    /// Give up and return the error to the requester.
    Fail(SchedError),
    /// Something else is resident -- evict it and try once more. The request's
    /// `oom_retry_attempted` has been set, so this can happen at most once.
    EvictAndRetry,
    /// Step the **automatic** context down and try once more, without evicting.
    ///
    /// **Upstream:** `reduceAutoNumCtxForLoadOOM`. Only ever reached when
    /// [`LlmRequest::num_ctx_auto`] is set -- a user-chosen context is never
    /// quietly shrunk.
    ReduceContextAndRetry {
        /// The context we were trying.
        old_num_ctx: u32,
        /// The smaller context to try instead.
        new_num_ctx: u32,
    },
}

// ---------------------------------------------------------------------------
// The scheduler
// ---------------------------------------------------------------------------

/// Decides what is loaded, what get evicted, and how big each load may be.
///
/// **Upstream:** `sched.go` `Scheduler`. See the module header for the
/// concurrency decision, the refcount invariant, and what "fits" mean.
///
/// ## How a caller drive it
///
/// ```text
///   clamp_request_options(&model, &mut opts)
///   sched.enqueue(req)?                        // ErrMaxQueue if full
///   while let Some(req) = sched.next_pending() {
///       loop {
///           match sched.next_action(&mut req, &sys, &mut gpus, &predict, flash) {
///               UseLoaded { runner, .. } => { serve(runner); break }
///               Load(plan)               => { ...actually load...; sched.runner_loaded(..); break }
///               Evict { .. } | EvictAll { .. } | Retry => {
///                   for u in sched.take_expired(Instant::now()) { close(u) }
///                   continue
///               }
///           }
///       }
///   }
/// ```
///
/// Everything in that sketch is synchronous. Where the caller want concurrency,
/// it put the `Scheduler` behind its own lock -- the type never assume one.
pub struct Scheduler {
    config: SchedulerConfig,
    loaded: IndexMap<String, RunnerRef>,
    pending: VecDeque<LlmRequest>,
    /// The key we are mid-load on, if any.
    ///
    /// **Upstream:** `Scheduler.activeLoading`, whose comment is the constraint
    /// this whole design lean on: *"We can only load one model at a time but new
    /// requests to models that already loaded can happen in parallel."*
    active_loading: Option<String>,
    /// `max_runners` once auto-derivation has happened. Cached because upstream
    /// cache it in `processPending`'s loop variable for the process lifetime.
    max_runners_resolved: Option<u64>,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("loaded", &self.loaded.keys().collect::<Vec<_>>())
            .field("pending", &self.pending.len())
            .field("active_loading", &self.active_loading)
            .finish()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

impl Scheduler {
    /// A scheduler with the given configuration.
    ///
    /// **Upstream:** `sched.go` `InitScheduler`.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            loaded: IndexMap::new(),
            pending: VecDeque::new(),
            active_loading: None,
            max_runners_resolved: None,
        }
    }

    /// A scheduler configured from the environment.
    pub fn from_env(env: &Env) -> Self {
        Self::new(SchedulerConfig::from_env(env))
    }

    /// The configuration in force.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// How many models are resident.
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }

    /// Is this model resident?
    pub fn is_loaded(&self, key: &str) -> bool {
        self.loaded.contains_key(key)
    }

    /// Borrow a resident runner's bookkeeping.
    pub fn runner(&self, key: &str) -> Option<&RunnerRef> {
        self.loaded.get(key)
    }

    /// Keys of every resident model, in load order.
    pub fn loaded_keys(&self) -> Vec<String> {
        self.loaded.keys().cloned().collect()
    }

    /// Requests waiting.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Mark a registered runner as still doing its **initial** load, or as done.
    ///
    /// **Upstream:** `runnerRef.loading`, set `true` when the `runnerRef` is
    /// built and cleared once `WaitUntilRunning` return.
    ///
    /// A caller that only call [`Scheduler::runner_loaded`] once the runner is
    /// fully up never need this -- `loading` default to `false` and the ordinary
    /// [`NEEDS_RELOAD_PING_TIMEOUT`] applies. A caller that register earlier, so
    /// that concurrent requests can queue behind a big model coming up, should
    /// set it `true` and clear it when the runner answer. That widen the health
    /// ping to [`NEEDS_RELOAD_LOADING_PING_TIMEOUT`], which is the difference
    /// between "a 70B is slow off a spinning disk" and "reload it, it must be
    /// dead".
    pub fn set_loading(&mut self, key: &str, loading: bool) {
        if let Some(runner_ref) = self.loaded.get_mut(key) {
            runner_ref.loading = loading;
        }
    }

    // -- queue ------------------------------------------------------------

    /// Put a request on the pending queue.
    ///
    /// **Upstream:** the `select { case s.pendingReqCh <- req: default: req.errCh
    /// <- ErrMaxQueue }` in `getRunner`. A **non-blocking** offer: full queue is
    /// an immediate error, never a wait, because a client that queue behind 512
    /// others has already lost.
    pub fn enqueue(&mut self, req: LlmRequest) -> Result<(), SchedError> {
        if self.pending.len() as u64 >= self.config.max_queue {
            return Err(SchedError::MaxQueue);
        }
        self.pending.push_back(req);
        Ok(())
    }

    /// Take the next pending request, bumping its attempt count.
    ///
    /// **Upstream:** `case pending := <-s.pendingReqCh: pending.schedAttempts++`.
    pub fn next_pending(&mut self) -> Option<LlmRequest> {
        let mut req = self.pending.pop_front()?;
        req.sched_attempts += 1;
        Some(req)
    }

    // -- the fast path ----------------------------------------------------

    /// If this model is already loaded **and** usable as-is, take a reference to
    /// it right now without queueing.
    ///
    /// **Upstream:** the head of `sched.go` `getRunner` --
    /// `if runner != nil && !runner.needsReload(c, req) { req.useLoadedRunner(...) }`.
    ///
    /// On `Some`, the refcount has been incremented and the caller owe exactly
    /// one [`Scheduler::request_finished`]. On `None`, enqueue the request.
    pub fn acquire_if_loaded(&mut self, req: &LlmRequest) -> Option<Arc<dyn ModelRunner>> {
        let key = req.key();
        if !self.loaded.contains_key(&key) {
            return None;
        }
        if self.needs_reload(&key, req) {
            return None;
        }
        self.use_loaded_runner(&key, req)
    }

    /// Hand a loaded runner to a request: +1 refcount, cancel the idle deadline,
    /// adopt the request's keep-alive.
    ///
    /// **Upstream:** `sched.go` `LlmRequest.useLoadedRunner`.
    ///
    /// Three things happen together and all three matter. The refcount rise, so
    /// nothing can evict it. The expiry deadline is **cleared**, not merely
    /// pushed back -- a busy runner has no deadline at all, and one is only armed
    /// again when it next go idle. And the request's `session_duration`, if it
    /// gave one, **replace** the runner's, so a later request with a longer
    /// keep-alive extend the model's life.
    ///
    /// Returns `None` if the key is not loaded.
    pub fn use_loaded_runner(
        &mut self,
        key: &str,
        req: &LlmRequest,
    ) -> Option<Arc<dyn ModelRunner>> {
        let runner_ref = self.loaded.get_mut(key)?;
        runner_ref.ref_count += 1;
        runner_ref.expires_at = None;
        if let Some(d) = req.session_duration {
            runner_ref.session_duration = d;
        }
        Some(Arc::clone(&runner_ref.runner))
    }

    /// One in-flight request has finished with this runner: -1 refcount, and arm
    /// the idle deadline if that was the last one.
    ///
    /// **Upstream:** the `case finished := <-s.finishedReqCh:` arm of
    /// `processCompleted`.
    ///
    /// The three-way branch at zero is upstream's exactly:
    ///
    /// * `session_duration` is zero -> **expire immediately**; this is how an
    ///   evicted-but-busy runner finally go.
    /// * [`Expiry::Never`] -> no deadline is ever armed, so
    ///   [`Scheduler::take_expired`] will never pick it up.
    /// * otherwise -> deadline at `now + session_duration`.
    ///
    /// A refcount that would go below zero is **clamped, not wrapped**. Upstream
    /// use an unsigned `refCount--` which wrap to `2^64-1` on a double release
    /// and pin the model in VRAM forever; upstream's own test file carry a `TODO`
    /// about exactly that scenario. Clamping is a deliberate divergence and the
    /// safer failure: a double release lose a reference rather than leaking the
    /// whole model.
    pub fn request_finished(&mut self, key: &str, now: Instant) {
        let Some(runner_ref) = self.loaded.get_mut(key) else {
            return;
        };
        runner_ref.ref_count = runner_ref.ref_count.saturating_sub(1);
        if runner_ref.ref_count > 0 {
            return;
        }
        match runner_ref.session_duration {
            Expiry::After(d) if d.is_zero() => runner_ref.expires_at = Some(now),
            Expiry::After(d) => runner_ref.expires_at = Some(now + d),
            Expiry::Never => runner_ref.expires_at = None,
        }
    }

    // -- reload decision --------------------------------------------------

    /// Must the loaded runner under `key` be torn down and rebuilt for this
    /// request?
    ///
    /// **Upstream:** `sched.go` `runnerRef.needsReload`.
    ///
    /// Reload when **any** of these hold:
    ///
    /// 1. the runner has no options (it was already unloaded);
    /// 2. the context-shift setting differ;
    /// 3. the adapter paths differ (element-wise, order included);
    /// 4. the projector paths differ, same rule;
    /// 5. the **load-time** options differ -- delegated to
    ///    [`Options::needs_reload`], which compare exactly the `Runner` half,
    ///    because that is the half baked into the runner at load time;
    /// 6. the runner fail its health ping.
    ///
    /// ## The four adjustments made before comparing, and why each exist
    ///
    /// * `num_ctx` is clamped to the runner's `train_context` first -- a request
    ///   for 1M context against a 32k model is not a *different* configuration,
    ///   it resolve to the same 32k.
    /// * if **both** sides chose `num_ctx` automatically, the request's is
    ///   overwritten with the runner's. We must not reload a model because our
    ///   own automatic tier moved when free VRAM shifted.
    /// * same for `num_batch`.
    /// * if the runner's mmap setting was automatic and the request has no
    ///   opinion, adopt the runner's.
    /// * `num_gpu < 0` on the request mean "you decide", so **both** sides are
    ///   normalised to `-1`. Upstream's comment: *"Don't reload runner if
    ///   num_gpu=-1 was provided."* Skip this and every request against a runner
    ///   that resolved to, say, 33 layers would look like a mismatch.
    ///
    /// MLX runners skip the option comparison entirely (upstream's
    /// `!runner.model.IsMLX()`), because their options are not what configure
    /// them.
    pub fn needs_reload(&self, key: &str, req: &LlmRequest) -> bool {
        let Some(runner_ref) = self.loaded.get(key) else {
            // Not loaded at all -- there is nothing to reuse, so the caller must
            // load. Upstream never reach needsReload in this state; returning
            // `true` keep every caller on the safe side of the question.
            return true;
        };
        let (Some(runner_model), Some(runner_opts)) = (&runner_ref.model, &runner_ref.options)
        else {
            // Upstream: `if runner.Options == nil { return true }`.
            return true;
        };

        let mut opts_existing = runner_opts.clone();
        let mut opts_new = req.opts.clone();

        opts_new.runner.num_ctx = effective_context(opts_new.runner.num_ctx, runner_ref.train_context);
        if runner_ref.num_ctx_auto && req.num_ctx_auto {
            opts_new.runner.num_ctx = opts_existing.runner.num_ctx;
        }
        if runner_ref.num_batch_auto && req.num_batch_auto {
            opts_new.runner.num_batch = opts_existing.runner.num_batch;
        }
        if runner_ref.use_mmap_auto && opts_new.runner.use_mmap.is_none() {
            opts_new.runner.use_mmap = opts_existing.runner.use_mmap;
        }
        if opts_new.runner.num_gpu < 0 {
            opts_existing.runner.num_gpu = -1;
            opts_new.runner.num_gpu = -1;
        }

        let context_shift = if req.model.model_path.is_empty() {
            req.context_shift
        } else {
            resolve_context_shift(req.shift, &req.model)
        };
        if runner_ref.context_shift != context_shift {
            return true;
        }

        if runner_model.adapter_paths != req.model.adapter_paths
            || runner_model.projector_paths != req.model.projector_paths
        {
            return true;
        }

        if !runner_model.is_mlx && opts_existing.needs_reload(&opts_new) {
            return true;
        }

        runner_ref.runner.ping(runner_ref.ping_timeout()).is_err()
    }

    // -- eviction ---------------------------------------------------------

    /// Pick the runner to chase out to make room.
    ///
    /// **Upstream:** `sched.go` `findRunnerToUnload`.
    ///
    /// Sort by `(session_duration, name)` -- shortest keep-alive first, name as
    /// the tie-break so the choice is deterministic across runs. Then:
    ///
    /// 1. **the first idle runner wins**, wherever it sit in the order. Evicting
    ///    an idle model cost nobody anything;
    /// 2. if none are idle, take the head of the list -- the one with the
    ///    shortest keep-alive, i.e. the one that was going to go soonest anyway.
    ///    It is not evicted on the spot (its refcount is positive); it is marked,
    ///    and it go when its last request finish.
    ///
    /// `None` mean nothing is loaded. Upstream leave a `TODO` here about picking
    /// by size instead -- deliberately not implemented, since diverging from the
    /// oracle on a heuristic would make every future upstream diff harder to
    /// read for no measured gain.
    pub fn find_runner_to_unload(&self) -> Option<String> {
        if self.loaded.is_empty() {
            return None;
        }
        let mut order: Vec<&RunnerRef> = self.loaded.values().collect();
        order.sort_by(|a, b| {
            a.duration_sort_key()
                .cmp(&b.duration_sort_key())
                .then_with(|| a.name_sort_key().cmp(b.name_sort_key()))
                // Final tie-break on the map key. Upstream's sort is unstable and
                // leaves genuine ties in arbitrary order; ours is total, so the
                // same state always give the same victim. Deliberate divergence,
                // in the direction of determinism.
                .then_with(|| a.model_key.cmp(&b.model_key))
        });

        if let Some(idle) = order.iter().find(|r| r.is_idle()) {
            return Some(idle.model_key.clone());
        }
        order.first().map(|r| r.model_key.clone())
    }

    /// Mark a runner to go as soon as it is idle.
    ///
    /// **Upstream:** `sched.go` `expireRunner`, and the identical block inside
    /// `processPending`'s `runnerToExpire` handling.
    ///
    /// Zeroing `session_duration` is the mechanism, and it is worth understanding
    /// rather than memorising: an idle runner get `expires_at = now` and go on
    /// the next [`Scheduler::take_expired`]; a **busy** one keep serving, and
    /// when its last request finish, [`Scheduler::request_finished`] see the zero
    /// duration and expire it immediately. One flag covers both cases with no
    /// special-casing.
    pub fn expire_runner(&mut self, key: &str, now: Instant) {
        let Some(runner_ref) = self.loaded.get_mut(key) else {
            return;
        };
        runner_ref.session_duration = Expiry::After(Duration::ZERO);
        runner_ref.expires_at = if runner_ref.is_idle() { Some(now) } else { None };
    }

    /// Mark every resident model except `keep_key` to go.
    ///
    /// **Upstream:** `sched.go` `evictAllAndWait`, minus the waiting -- the caller
    /// drain [`Scheduler::take_expired`] instead of blocking on `unloadedCh`.
    /// Returns the keys marked, for logging.
    pub fn evict_all_except(&mut self, keep_key: &str, now: Instant) -> Vec<String> {
        let victims: Vec<String> = self
            .loaded
            .keys()
            .filter(|k| k.as_str() != keep_key)
            .cloned()
            .collect();
        for key in &victims {
            self.expire_runner(key, now);
        }
        victims
    }

    /// Mark **every** resident model to go, because the runtime hit OOM.
    ///
    /// **Upstream:** `sched.go` `expireRunnersForRuntimeOOM`, whose log line say
    /// it best: *"runtime OOM detected; expiring loaded models to clear memory
    /// before next request"*. Note this fire on an OOM during **generation**, not
    /// during load -- by then the fit prediction has already been proven wrong,
    /// so the only safe response is to clear the board.
    ///
    /// The caller is responsible for having decided the error really is an OOM;
    /// upstream gate this on `llm.IsOutOfMemory(err)`, and that classification
    /// belong to whoever own the runner.
    pub fn expire_all_for_runtime_oom(&mut self, now: Instant) -> Vec<String> {
        let keys = self.loaded_keys();
        for key in &keys {
            self.expire_runner(key, now);
        }
        keys
    }

    /// Remove and return every runner whose deadline has passed **and** which
    /// nobody is using.
    ///
    /// **Upstream:** the `case runner := <-s.expiredCh:` arm of
    /// `processCompleted`, collapsed into a query.
    ///
    /// Each returned runner has been dropped from the map and had
    /// [`ModelRunner::unload`] called on it. The caller should then honour
    /// [`Unloaded::needs_vram_recovery_wait`] before loading the next model.
    ///
    /// **A runner with `ref_count > 0` is never returned**, however overdue it
    /// is. That is the refcount invariant, and it is what upstream's *"expired
    /// event with positive ref count, retrying"* goroutine is emulating with a
    /// 10 ms sleep. Here it needs no emulation: the runner simply is not due yet,
    /// and it will be picked up by a later call once it goes idle.
    pub fn take_expired(&mut self, now: Instant) -> Vec<Unloaded> {
        let due: Vec<String> = self
            .loaded
            .iter()
            .filter(|(_, r)| r.is_idle() && r.expires_at.is_some_and(|at| at <= now))
            .map(|(k, _)| k.clone())
            .collect();

        let mut out = Vec::with_capacity(due.len());
        for key in due {
            let Some(mut runner_ref) = self.loaded.shift_remove(&key) else {
                continue;
            };
            let unloaded = Unloaded {
                key,
                runner: Arc::clone(&runner_ref.runner),
                vram_size: runner_ref.vram_size,
                needs_vram_recovery_wait: needs_vram_recovery_wait(&runner_ref),
            };
            runner_ref.unload();
            out.push(unloaded);
        }
        out
    }

    /// Shut everything down, including a load in flight.
    ///
    /// **Upstream:** `sched.go` `unloadAllRunners`. Used at process shutdown, so
    /// it does **not** respect refcounts -- the process is going away and a
    /// half-served request is going with it either way.
    pub fn unload_all(&mut self) {
        self.active_loading = None;
        for (_, runner_ref) in self.loaded.iter_mut() {
            runner_ref.unload();
        }
        self.loaded.clear();
    }
}

/// Does unloading this runner call for a VRAM recovery wait?
///
/// **Upstream:** the early-out at the top of `sched.go` `waitForVRAMRecovery`:
/// no devices, not all discrete, or a single Metal device -- all skip the wait.
///
/// The reasoning behind each skip: **CPU** has no separate pool. **iGPU** free
/// memory is system RAM, which the OS reclaims synchronously. **Metal** report
/// unified memory that likewise updates immediately. Only discrete VRAM has the
/// lagging free-memory counter that the wait exists for.
fn needs_vram_recovery_wait(runner_ref: &RunnerRef) -> bool {
    if runner_ref.gpus.is_empty() || !runner_ref.discrete_gpus {
        return false;
    }
    !(runner_ref.gpus.len() == 1 && runner_ref.gpus[0].library == "Metal")
}

/// This host's name in **Go's** vocabulary, for the `goos` parameters.
///
/// **Why this exists and is not just `std::env::consts::OS`:** Rust say
/// `"macos"`, Go say `"darwin"`. Every `goos` comparison in this module is
/// against upstream's Go strings, so passing Rust's name straight through would
/// silently disable the whole Metal mmap branch on a Mac -- a bug that shows up
/// as "why is my Mac slower than it should be", months later, with nothing in
/// the logs. Windows and Linux happen to agree; macOS is the one that bite.
pub fn host_goos() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    }
}

/// The hardware and platform facts a load decision need, gathered in one place.
///
/// Everything in here is **handed in by the caller** -- the scheduler never
/// query a driver, stat a file, or read the process environment during a
/// decision.
pub struct LoadEnv<'a> {
    /// Host memory snapshot.
    pub system_info: &'a SystemInfo,
    /// Devices available. Taken by `&mut` because
    /// [`Scheduler::update_free_space`] rewrite `free_memory` in place to account
    /// for models already resident, exactly as upstream do before every fit
    /// decision.
    pub gpus: &'a mut Vec<DeviceInfo>,
    /// Resolved flash-attention mode.
    ///
    /// **Upstream:** `llm.LlamaServerFlashAttention(gpus)`
    /// (`llm/llama_server.go:606`), which combine `OLLAMA_FLASH_ATTENTION` with
    /// per-device driver support. Driver-version support checks are device
    /// *discovery*, not scheduling, so the resolved answer come in rather than
    /// being computed here.
    pub flash_attention: FlashAttentionType,
    /// Platform name in **Go's** spelling -- see [`host_goos`].
    pub goos: &'a str,
    /// Predict VRAM bytes for a given aggregate context length in tokens.
    ///
    /// Typically `&|ctx| predict_server_vram(file_size, &kv, ctx)`. A closure
    /// rather than a value because the scheduler decide `num_parallel` itself,
    /// and the aggregate context -- hence the prediction -- depend on it.
    pub predict_vram: &'a dyn Fn(u64) -> u64,
}

impl Scheduler {
    /// How many sequences one runner may serve for this model.
    ///
    /// **Upstream:** the top of `sched.go` `load`.
    ///
    /// Three clamps, in order: at least 1; **exactly 1 for a non-completion
    /// (embedding) model**, since there is no generation loop to overlap; and
    /// exactly 1 for anything in [`UNSAFE_PARALLEL_ARCHITECTURES`], which is a
    /// correctness clamp, not a tuning one.
    pub fn resolve_num_parallel(&self, model: &ModelRef) -> u32 {
        let mut num_parallel = self.config.num_parallel.max(1);
        if !model.has_capability(Capability::Completion) {
            num_parallel = 1;
        }
        if UNSAFE_PARALLEL_ARCHITECTURES.contains(&model.model_family.as_str()) {
            num_parallel = 1;
        }
        num_parallel
    }

    /// The resident-model cap, deriving it from the GPU count on first use.
    ///
    /// **Upstream:** the `if maxRunners <= 0 { maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1)) }`
    /// block in `processPending`, which cache the result in a loop local for the
    /// life of the process. Cached here for the same reason: re-deriving it as
    /// GPUs come and go would change the cap under a running workload.
    ///
    /// `max(gpu_count, 1)` mean a CPU-only box still get a cap of 3 rather than 0
    /// (which would mean "unlimited" and let a box thrash itself to death).
    pub fn resolve_max_runners(&mut self, gpu_count: usize) -> u64 {
        if self.config.max_runners > 0 {
            return self.config.max_runners;
        }
        if let Some(cached) = self.max_runners_resolved {
            return cached;
        }
        let resolved = DEFAULT_MODELS_PER_GPU * (gpu_count.max(1) as u64);
        self.max_runners_resolved = Some(resolved);
        resolved
    }

    /// Subtract what the resident models are already holding from each device's
    /// reported free memory.
    ///
    /// **Upstream:** `sched.go` `updateFreeSpace`.
    ///
    /// **Why not just trust the driver:** upstream's own comment answers it --
    /// *"maybe we should just always trust our numbers, since cuda's free memory
    /// reporting is laggy and we might unload models we didn't actually need to.
    /// The risk is if some other GPU intensive app is loaded after we start our
    /// first runner, then we'll never account for that, so picking the smallest
    /// free value seems prudent."* So the rule is **take the pessimistic of the
    /// two**: our own prediction (`total - predicted`) only replace the driver's
    /// number when it is smaller.
    ///
    /// A prediction exceeding total VRAM shouldn't happen; when it does, free is
    /// set to 0 rather than underflowing.
    pub fn update_free_space(&self, all_gpus: &mut [DeviceInfo]) {
        if all_gpus.is_empty() || self.loaded.is_empty() {
            return;
        }

        let mut predicted: Vec<u64> = vec![0; all_gpus.len()];
        for runner_ref in self.loaded.values() {
            for (i, gpu) in all_gpus.iter().enumerate() {
                predicted[i] =
                    predicted[i].saturating_add(runner_ref.runner.vram_by_gpu(&gpu.device_id));
            }
        }

        for (i, gpu) in all_gpus.iter_mut().enumerate() {
            let p = predicted[i];
            if p > gpu.total_memory {
                gpu.free_memory = 0;
            } else if gpu.total_memory - p < gpu.free_memory {
                gpu.free_memory = gpu.total_memory - p;
            }
        }
    }

    /// Total bytes the resident mmap-ed models are mapping.
    ///
    /// **Upstream:** `sched.go` `loadedMmapModelSizeLocked`. Feeds
    /// [`disable_mmap_for_host_pressure`]: mmap-ed weights sit in page cache and
    /// count against host memory, so the next load must know how much is already
    /// spoken for.
    ///
    /// Falls back to the runner's `total_size` when the model's `file_size` is
    /// unknown, exactly as upstream fall back when `os.Stat` fail.
    pub fn loaded_mmap_model_size(&self) -> u64 {
        self.loaded
            .values()
            .filter(|r| r.uses_mmap())
            .map(|r| {
                let size = r.model.as_ref().map(|m| m.file_size).unwrap_or(0);
                if size > 0 { size } else { r.total_size }
            })
            .fold(0u64, |acc, n| acc.saturating_add(n))
    }

    /// Snapshot of what is loaded, for status reporting.
    ///
    /// **Upstream:** `sched.go` `Scheduler.loadedModels`.
    pub fn loaded_models(&self, now: Instant) -> Vec<LoadedModel> {
        self.loaded
            .values()
            .filter_map(|r| {
                let model = r.model.clone()?;
                let expires_at = r.expires_at.or(match r.session_duration {
                    // Upstream: "The scheduler waits to set expiresAt, so a model
                    // that is still loading may have the zero value. Estimate
                    // expiration from the session duration instead."
                    Expiry::After(d) => Some(now + d),
                    Expiry::Never => None,
                });
                Some(LoadedModel {
                    model,
                    size: r.runner.total_size().max(r.total_size),
                    size_vram: r.runner.vram_size().max(r.vram_size),
                    context_length: r.runner.context_length(),
                    expires_at,
                })
            })
            .collect()
    }

    /// Work out everything about a load, short of performing it.
    ///
    /// **Upstream:** the first half of `sched.go` `load`, up to the point where it
    /// call `newServerFn`. Order matters and is upstream's: parallelism, then
    /// aggregate context, then prediction, then placement, then automatic batch,
    /// then the batch surcharge, then mmap defaults.
    ///
    /// This **mutates `req`** where upstream mutate `req.opts` -- specifically
    /// `num_batch` (automatic batch) and `use_mmap` (+ `use_mmap_auto`). Those are
    /// decisions about this request that later stages, including
    /// [`Scheduler::needs_reload`], must be able to see.
    pub fn plan_load(&self, req: &mut LlmRequest, env: &LoadEnv<'_>, require_full: bool) -> LoadPlan {
        let num_parallel = self.resolve_num_parallel(&req.model);
        let completion = req.model.has_capability(Capability::Completion);
        let session_duration = req.session_duration.unwrap_or(self.config.keep_alive);

        let effective_num_ctx = effective_llama_server_context(
            req.opts.runner.num_ctx,
            req.model.train_context,
            num_parallel,
        );
        let predicted = (env.predict_vram)(effective_num_ctx);

        let (load_gpus, mut launch_opts) = select_llama_server_placement(
            env.system_info,
            env.gpus,
            predicted,
            &req.opts,
            self.config.sched_spread,
        );

        let availability =
            available_memory_for_placement(env.system_info, &load_gpus, &launch_opts);

        // Upstream: req.applyAutomaticGenerationBatch -- only for completion
        // models whose batch we chose ourselves.
        if completion && req.num_batch_auto {
            req.opts.runner.num_batch = automatic_generation_batch(
                effective_num_ctx,
                predicted,
                availability.available,
                env.flash_attention,
                &load_gpus,
            );
        }
        launch_opts.runner.num_batch = req.opts.runner.num_batch;

        let predicted_for_load = predicted.saturating_add(
            generation_batch_surcharge_for_completion(completion, launch_opts.runner.num_batch),
        );

        // Upstream: applyLlamaServerMmapDefaults. The default reason is checked
        // first; only if it says nothing does the host-pressure heuristic get a
        // say. Both write through to req.opts so needs_reload can see them.
        let mmap_reason = disable_mmap_default_reason(
            env.goos,
            &req.opts,
            &load_gpus,
            req.model.block_count,
            predicted,
            availability.available,
        );
        let mmap_reason = match mmap_reason {
            Some(reason) => {
                req.opts.runner.use_mmap = Some(false);
                req.use_mmap_auto = true;
                Some(reason)
            }
            None => {
                let placement = gpus_for_placement(&load_gpus, &launch_opts);
                if disable_mmap_for_host_pressure(
                    env.goos,
                    &req.opts,
                    env.system_info,
                    placement,
                    req.model.file_size,
                    self.loaded_mmap_model_size(),
                    predicted,
                    availability.available,
                ) {
                    req.opts.runner.use_mmap = Some(false);
                    req.use_mmap_auto = true;
                }
                None
            }
        };
        launch_opts.runner.use_mmap = req.opts.runner.use_mmap;

        let context_shift = if req.model.model_path.is_empty() {
            req.context_shift
        } else {
            resolve_context_shift(req.shift, &req.model)
        };
        req.context_shift = context_shift;

        LoadPlan {
            key: req.key(),
            num_parallel,
            session_duration,
            gpus: load_gpus,
            launch_opts,
            require_full,
            predicted_vram: predicted_for_load,
            effective_num_ctx,
            completion,
            context_shift,
            mmap_disabled_reason: mmap_reason,
        }
    }

    /// The pre-flight fit check: will this plan leave room for what is already
    /// resident?
    ///
    /// **Upstream:** the *"Pre-flight check: estimate whether the model fits in
    /// remaining memory"* block in `sched.go` `load`, whose own comment explain
    /// the necessity: *"llama-server auto-detects layers based on available VRAM,
    /// so if we predict it won't fit, evict before spawning."* Spawning first and
    /// discovering the squeeze afterwards mean a partially-offloaded model and a
    /// user wondering why it got slow.
    ///
    /// Returns `true` (fits) whenever the check does not apply at all: a plan
    /// that does not require full offload, an explicitly partial offload, nothing
    /// else resident, or no GPUs. Those are upstream's four guards, and each one
    /// is a case where evicting would help nobody.
    pub fn plan_fits(&self, req: &LlmRequest, plan: &LoadPlan, system_info: &SystemInfo) -> bool {
        if !plan.require_full
            || explicit_partial_gpu_offload(&plan.launch_opts, req.model.block_count)
            || self.loaded.is_empty()
            || plan.gpus.is_empty()
        {
            return true;
        }
        let availability =
            available_memory_for_placement(system_info, &plan.gpus, &plan.launch_opts);
        fits_with_headroom(plan.predicted_vram, availability.available)
    }

    /// **The scheduling decision.** What should happen next for this request?
    ///
    /// **Upstream:** one pass through the inner `for` loop of `sched.go`
    /// `processPending`. Upstream loop internally and block on channels; here each
    /// branch return, and the caller loop.
    ///
    /// The ladder, in upstream's order:
    ///
    /// 1. **Already loaded and usable** -> [`PendingAction::UseLoaded`], refcount
    ///    already taken.
    /// 2. **Already loaded but stale** -> [`PendingAction::Evict`] it, so the
    ///    reload can happen.
    /// 3. **At the runner cap** -> evict whatever
    ///    [`Scheduler::find_runner_to_unload`] pick.
    /// 4. **Nothing resident** -> [`PendingAction::Load`] with
    ///    `require_full = false`. Upstream: *"No models loaded. Load the model but
    ///    prefer the best fit."* A partial CPU offload is acceptable here because
    ///    there is nothing to evict that would improve matters.
    /// 5. **Something resident** -> plan with `require_full = true`; if it fits,
    ///    load; if not, either [`PendingAction::EvictAll`] (when this request has
    ///    already spent its OOM retry) or evict one.
    /// 6. **Nothing evictable** -> [`PendingAction::Retry`]; state moved under us.
    ///
    /// Free memory on `env.gpus` is refreshed in place before any fit decision, so
    /// the caller may hand in the raw driver numbers.
    pub fn next_action(&mut self, req: &mut LlmRequest, env: &mut LoadEnv<'_>) -> PendingAction {
        let key = req.key();

        // 1 & 2 -- something is already loaded under this key.
        if self.loaded.contains_key(&key) {
            if self.needs_reload(&key, req) {
                let now = Instant::now();
                self.expire_runner(&key, now);
                return PendingAction::Evict { key };
            }
            if let Some(runner) = self.use_loaded_runner(&key, req) {
                return PendingAction::UseLoaded { key, runner };
            }
            return PendingAction::Retry;
        }

        // 3 -- at the cap. Note upstream check the cap against the value cached
        // from a *previous* iteration; on the very first request it is still 0
        // (auto) so this branch is skipped and the cap get derived below.
        let loaded_count = self.loaded.len();
        if self.config.max_runners > 0 && loaded_count >= self.config.max_runners as usize {
            return self.evict_one_or_retry();
        }
        if let Some(cap) = self.max_runners_resolved
            && loaded_count >= cap as usize
        {
            return self.evict_one_or_retry();
        }

        // A CPU-only request must not consult the GPU list at all -- upstream
        // replace `gpus` with an empty slice when `NumGPU == 0`.
        if req.opts.runner.num_gpu == 0 {
            env.gpus.clear();
        }
        let _ = self.resolve_max_runners(env.gpus.len());
        self.update_free_space(env.gpus);

        // 4 -- nothing resident: best-effort load, partial offload allowed.
        if loaded_count == 0 {
            let plan = self.plan_load(req, env, false);
            return PendingAction::Load(Box::new(plan));
        }

        // 5 -- something resident: must fit fully alongside it.
        let plan = self.plan_load(req, env, true);
        if self.plan_fits(req, &plan, env.system_info) {
            return PendingAction::Load(Box::new(plan));
        }

        if req.oom_retry_attempted {
            let now = Instant::now();
            let keys = self.evict_all_except(&key, now);
            return PendingAction::EvictAll { keys };
        }

        self.evict_one_or_retry()
    }

    /// Mark one victim, or ask to be called again if there is nothing to mark.
    ///
    /// **Upstream:** the shared tail of `processPending` -- `runnerToExpire :=
    /// s.findRunnerToUnload()`, then *"runner to expire was nil, retrying"*.
    fn evict_one_or_retry(&mut self) -> PendingAction {
        match self.find_runner_to_unload() {
            Some(key) => {
                let now = Instant::now();
                self.expire_runner(&key, now);
                PendingAction::Evict { key }
            }
            None => PendingAction::Retry,
        }
    }

    /// Register a runner the caller has just brought up.
    ///
    /// **Upstream:** the tail of `sched.go` `load` -- building the `runnerRef`,
    /// the *"model was still loaded"* safeguard, and the post-`WaitUntilRunning`
    /// `refCount++`.
    ///
    /// The returned handle already has **refcount 1**: the request that triggered
    /// the load is holding it, so the caller owe exactly one
    /// [`Scheduler::request_finished`].
    ///
    /// `gpu_ids` is where the runner actually ended up, which may differ from
    /// `plan.gpus` -- the backend has the last word. It is intersected with
    /// `plan.gpus` to decide `discrete_gpus`.
    ///
    /// The runner's own `context_length` overwrite the request's `num_ctx` when
    /// both are meaningful, matching upstream: the runner may have clamped the
    /// window, and every later reload comparison must be against what actually
    /// happened, not what was asked for.
    pub fn runner_loaded(
        &mut self,
        req: &mut LlmRequest,
        plan: &LoadPlan,
        runner: Arc<dyn ModelRunner>,
        gpu_ids: Vec<DeviceId>,
    ) -> Arc<dyn ModelRunner> {
        // Upstream's `iGPUScan`: discrete if ANY placed device is discrete.
        // (The struct comment upstream says "all devices are discrete"; the code
        // says any. The code wins -- it is what actually runs.)
        let discrete_gpus = gpu_ids.iter().any(|id| {
            plan.gpus
                .iter()
                .any(|dev| &dev.device_id == id && !dev.integrated)
        });

        // The plan is the authority on context shift -- `plan_load` already
        // resolved it (and wrote it back onto the request). Reading it from the
        // plan rather than the request means a caller that hand-builds a plan
        // cannot accidentally register a runner whose recorded setting disagrees
        // with the one it was launched with -- which would make every subsequent
        // `needs_reload` say "reload", forever.
        req.context_shift = plan.context_shift;
        let effective_num_ctx = runner.context_length();
        if !req.model.model_path.is_empty() && effective_num_ctx > 0 {
            req.opts.runner.num_ctx = effective_num_ctx;
            req.context_shift = resolve_context_shift(req.shift, &req.model);
        }

        // Upstream: "Shouldn't happen, but safeguard against leaking a runner".
        if let Some(mut old) = self.loaded.shift_remove(&plan.key) {
            old.unload();
        }

        let runner_ref = RunnerRef {
            ref_count: 1,
            runner: Arc::clone(&runner),
            pid: runner.pid(),
            loading: false,
            gpus: gpu_ids,
            discrete_gpus,
            vram_size: runner.vram_size(),
            total_size: runner.total_size(),
            session_duration: plan.session_duration,
            expires_at: None,
            model: Some(req.model.clone()),
            options: Some(req.opts.clone()),
            model_path: req.model.model_path.clone(),
            model_key: plan.key.clone(),
            num_parallel: plan.num_parallel,
            num_ctx_auto: req.num_ctx_auto,
            num_batch_auto: req.num_batch_auto,
            use_mmap_auto: req.use_mmap_auto,
            context_shift: req.context_shift,
            train_context: req.model.train_context,
        };

        self.loaded.insert(plan.key.clone(), runner_ref);
        self.active_loading = None;
        runner
    }

    /// Decide what to do about a load that failed.
    ///
    /// **Upstream:** the error branch of `sched.go` `load`.
    ///
    /// The order of the three recovery attempts is upstream's and is not
    /// arbitrary:
    ///
    /// 1. **Shrink an automatic context** first. Cheapest recovery -- it costs
    ///    the user context they never asked for, and evicts nobody. Only ever
    ///    applies when [`LlmRequest::num_ctx_auto`] is set.
    /// 2. **Evict everything else** and retry, if anything else is resident.
    /// 3. **Give up.**
    ///
    /// Either recovery set [`LlmRequest::oom_retry_attempted`], so a request get
    /// **at most one** retry. Without that, a model too big for the box would
    /// evict the world, fail, and evict again, forever -- which is exactly what
    /// upstream's comment on that field warn about.
    ///
    /// A `RequiredFullNotMet` with nothing else loaded is terminal
    /// ([`SchedError::TooLarge`]): we already allowed partial offload and it
    /// still did not fit, so no eviction can help.
    pub fn on_load_failed(
        &mut self,
        req: &mut LlmRequest,
        failure: LoadFailure,
        require_full: bool,
    ) -> LoadFailureAction {
        self.active_loading = None;

        match failure {
            LoadFailure::RequiredFullNotMet => {
                if require_full {
                    LoadFailureAction::EvictAndRetry
                } else {
                    LoadFailureAction::Fail(SchedError::TooLarge)
                }
            }
            LoadFailure::OutOfMemory(msg) => {
                if !req.oom_retry_attempted {
                    if let Some(action) = self.reduce_auto_num_ctx_for_load_oom(req) {
                        return action;
                    }
                    if !self.loaded.is_empty() {
                        req.oom_retry_attempted = true;
                        return LoadFailureAction::EvictAndRetry;
                    }
                }
                LoadFailureAction::Fail(SchedError::LoadFailed(msg))
            }
            LoadFailure::Other(msg) => LoadFailureAction::Fail(SchedError::LoadFailed(msg)),
        }
    }

    /// Step an **automatic** context down one rung after an OOM.
    ///
    /// **Upstream:** `sched.go` `LlmRequest.reduceAutoNumCtxForLoadOOM`.
    ///
    /// Returns `None` -- meaning "no reduction available, try something else" --
    /// when the context was the user's choice, or when it is already at the
    /// bottom rung, or when the clamped context is not actually smaller than what
    /// we asked for (which would make the "retry" identical to the attempt that
    /// just failed, i.e. an infinite loop).
    fn reduce_auto_num_ctx_for_load_oom(
        &self,
        req: &mut LlmRequest,
    ) -> Option<LoadFailureAction> {
        if !req.num_ctx_auto {
            return None;
        }
        let old_num_ctx = req.opts.runner.num_ctx;
        let effective = effective_context(old_num_ctx, req.model.train_context);
        let new_num_ctx = next_lower_auto_num_ctx(effective)?;
        if new_num_ctx >= old_num_ctx {
            return None;
        }
        req.opts.runner.num_ctx = new_num_ctx;
        req.oom_retry_attempted = true;
        Some(LoadFailureAction::ReduceContextAndRetry {
            old_num_ctx,
            new_num_ctx,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests -- ported from server/sched_test.go
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// Stand-in for a live runner.
    ///
    /// **Upstream:** `server/sched_test.go` `mockLlm`. Same idea, same purpose:
    /// upstream's own scheduler tests never spawn a llama-server either, which is
    /// the evidence that [`ModelRunner`] is a seam upstream already relies on
    /// rather than one we invented for our own convenience.
    struct MockRunner {
        vram_size: u64,
        total_size: u64,
        vram_by_gpu: HashMap<DeviceId, u64>,
        context_length: u32,
        ping_ok: AtomicBool,
        unload_calls: AtomicU32,
    }

    impl MockRunner {
        fn new(vram_size: u64) -> Self {
            Self {
                vram_size,
                total_size: vram_size,
                vram_by_gpu: HashMap::new(),
                context_length: 0,
                ping_ok: AtomicBool::new(true),
                unload_calls: AtomicU32::new(0),
            }
        }

        fn with_vram_by_gpu(mut self, entries: &[(DeviceId, u64)]) -> Self {
            self.vram_by_gpu = entries.iter().cloned().collect();
            self
        }

        fn arc(self) -> Arc<dyn ModelRunner> {
            Arc::new(self)
        }
    }

    impl ModelRunner for MockRunner {
        fn vram_size(&self) -> u64 {
            self.vram_size
        }
        fn total_size(&self) -> u64 {
            self.total_size
        }
        fn vram_by_gpu(&self, id: &DeviceId) -> u64 {
            self.vram_by_gpu.get(id).copied().unwrap_or(0)
        }
        fn context_length(&self) -> u32 {
            self.context_length
        }
        fn ping(&self, _timeout: Duration) -> Result<(), RunnerUnhealthy> {
            if self.ping_ok.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err(RunnerUnhealthy("foo".into()))
            }
        }
        fn unload(&self) {
            self.unload_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn gpu(id: &str, library: &str, free: u64) -> DeviceInfo {
        DeviceInfo {
            device_id: DeviceId::new(id, library),
            free_memory: free,
            total_memory: free,
            ..Default::default()
        }
    }

    fn igpu(id: &str, library: &str, free: u64) -> DeviceInfo {
        DeviceInfo {
            integrated: true,
            ..gpu(id, library, free)
        }
    }

    fn completion_model(path: &str) -> ModelRef {
        ModelRef {
            model_path: path.into(),
            capabilities: vec![Capability::Completion],
            ..Default::default()
        }
    }

    /// Register a runner the way a caller would, through the real
    /// [`Scheduler::runner_loaded`] path, then optionally release the reference
    /// so the runner sit idle.
    fn load_model(
        sched: &mut Scheduler,
        model: ModelRef,
        runner: Arc<dyn ModelRunner>,
        keep_alive: Expiry,
        gpu_ids: Vec<DeviceId>,
        release: bool,
    ) -> String {
        let key = model.scheduler_key();
        // `plan_load` resolves context shift for a real load; the helper must do
        // the same or every later `needs_reload` sees a phantom mismatch.
        let context_shift = resolve_context_shift(None, &model);
        let mut req = LlmRequest::new(model, Options::default());
        req.session_duration = Some(keep_alive);
        req.context_shift = context_shift;
        let plan = LoadPlan {
            key: key.clone(),
            num_parallel: 1,
            session_duration: keep_alive,
            gpus: gpu_ids
                .iter()
                .map(|id| DeviceInfo {
                    device_id: id.clone(),
                    ..Default::default()
                })
                .collect(),
            launch_opts: Options::default(),
            require_full: false,
            predicted_vram: 0,
            effective_num_ctx: 0,
            completion: true,
            context_shift,
            mmap_disabled_reason: None,
        };
        sched.runner_loaded(&mut req, &plan, runner, gpu_ids);
        if release {
            sched.request_finished(&key, Instant::now());
        }
        key
    }

    // -- keys and context shift -------------------------------------------

    /// **Upstream:** `TestSchedGetRunnerUsesDigestKeyWhenModelPathEmpty` plus
    /// `schedulerModelKey`'s own fallback ladder.
    #[test]
    fn the_scheduler_key_falls_back_from_path_to_digest_to_name() {
        let mut m = ModelRef {
            model_path: "/fake/model".into(),
            digest: "sha256:abc".into(),
            name: "library/qwen3:0.6b".into(),
            short_name: "qwen3".into(),
            ..Default::default()
        };
        assert_eq!(m.scheduler_key(), "/fake/model");
        m.model_path.clear();
        assert_eq!(m.scheduler_key(), "digest:sha256:abc");
        m.digest.clear();
        assert_eq!(m.scheduler_key(), "name:library/qwen3:0.6b");
        m.name.clear();
        assert_eq!(m.scheduler_key(), "short:qwen3");
        m.short_name.clear();
        assert_eq!(m.scheduler_key(), "");
    }

    /// Two safetensors models with no path must not collide -- the whole reason
    /// the digest fallback exists.
    #[test]
    fn two_pathless_models_with_different_digests_get_different_keys() {
        let a = ModelRef {
            digest: "sha256:aaa".into(),
            ..Default::default()
        };
        let b = ModelRef {
            digest: "sha256:bbb".into(),
            ..Default::default()
        };
        assert_ne!(a.scheduler_key(), b.scheduler_key());
    }

    /// **Upstream:** `TestResolveContextShift`.
    #[test]
    fn resolve_context_shift_honours_an_explicit_override_over_the_architecture() {
        let deepseek = ModelRef {
            model_family: "deepseek2".into(),
            ..Default::default()
        };
        let llama = ModelRef {
            model_family: "llama".into(),
            ..Default::default()
        };

        // No override: architecture decides.
        assert!(!resolve_context_shift(None, &deepseek));
        assert!(resolve_context_shift(None, &llama));

        // Explicit override wins both ways, even against deepseek2.
        assert!(resolve_context_shift(Some(true), &deepseek));
        assert!(!resolve_context_shift(Some(false), &llama));
    }

    /// A model can carry `deepseek2` as a secondary family and must still be
    /// caught. Upstream check `ModelFamily` **and** `ModelFamilies`.
    #[test]
    fn a_secondary_deepseek2_family_still_disables_context_shift() {
        let m = ModelRef {
            model_family: "llama".into(),
            model_families: vec!["llama".into(), "deepseek2".into()],
            ..Default::default()
        };
        assert!(!m.supports_context_shift());
    }

    /// **Upstream:** `sched.go` `effectiveContext`.
    #[test]
    fn a_requested_context_is_clamped_to_what_the_model_was_trained_on() {
        assert_eq!(effective_context(262144, 32768), 32768);
        assert_eq!(effective_context(4096, 32768), 4096);
        // 0 train context mean unknown -- no clamp at all.
        assert_eq!(effective_context(262144, 0), 262144);
    }

    /// The multiply that is easy to forget: the KV cache must hold every parallel
    /// sequence at once.
    #[test]
    fn the_effective_context_multiplies_by_the_parallel_sequence_count() {
        assert_eq!(effective_llama_server_context(4096, 0, 4), 16384);
        // Clamp first, multiply after.
        assert_eq!(effective_llama_server_context(262144, 8192, 2), 16384);
        // A zero parallel count is treated as one, never as zero context.
        assert_eq!(effective_llama_server_context(4096, 0, 0), 4096);
    }

    /// **Upstream:** the top of `getRunner`.
    #[test]
    fn the_context_floor_is_four_tokens_and_2048_for_a_vision_model() {
        let plain = ModelRef::default();
        let mut opts = Options::default();
        opts.runner.num_ctx = 1;
        clamp_request_options(&plain, &mut opts);
        assert_eq!(opts.runner.num_ctx, MIN_NUM_CTX);

        let vision = ModelRef {
            capabilities: vec![Capability::Vision],
            ..Default::default()
        };
        let mut opts = Options::default();
        opts.runner.num_ctx = 512;
        clamp_request_options(&vision, &mut opts);
        assert_eq!(opts.runner.num_ctx, VISION_MIN_NUM_CTX);

        // Already bigger than the floor -- left alone.
        let mut opts = Options::default();
        opts.runner.num_ctx = 8192;
        clamp_request_options(&vision, &mut opts);
        assert_eq!(opts.runner.num_ctx, 8192);
    }

    // -- memory arithmetic -------------------------------------------------

    /// **Upstream:** `TestAvailableMemoryForLoadUsesWorstSharedMemoryMeasurement`,
    /// every case.
    #[test]
    fn available_memory_for_load_uses_the_worst_shared_memory_measurement() {
        struct Case {
            name: &'static str,
            system_free: u64,
            gpus: Vec<DeviceInfo>,
            want_available: u64,
            want_gpu_free: u64,
            want_system_limited: bool,
        }

        let cases = vec![
            Case {
                name: "integrated metal uses lower system free",
                system_free: 80 * GIGABYTE,
                gpus: vec![igpu("0", "Metal", 300 * GIGABYTE)],
                want_available: 80 * GIGABYTE,
                want_gpu_free: 300 * GIGABYTE,
                want_system_limited: true,
            },
            Case {
                name: "integrated gpu uses lower system free",
                system_free: 6 * GIGABYTE,
                gpus: vec![igpu("0", "Vulkan", 12 * GIGABYTE)],
                want_available: 6 * GIGABYTE,
                want_gpu_free: 12 * GIGABYTE,
                want_system_limited: true,
            },
            Case {
                name: "discrete metal ignores lower system free",
                system_free: 6 * GIGABYTE,
                gpus: vec![gpu("0", "Metal", 12 * GIGABYTE)],
                want_available: 12 * GIGABYTE,
                want_gpu_free: 12 * GIGABYTE,
                want_system_limited: false,
            },
            Case {
                name: "discrete gpu ignores lower system free",
                system_free: 6 * GIGABYTE,
                gpus: vec![gpu("0", "CUDA", 12 * GIGABYTE)],
                want_available: 12 * GIGABYTE,
                want_gpu_free: 12 * GIGABYTE,
                want_system_limited: false,
            },
            Case {
                name: "mixed gpus only clamp the integrated contribution",
                system_free: 6 * GIGABYTE,
                gpus: vec![
                    gpu("0", "CUDA", 12 * GIGABYTE),
                    igpu("1", "Vulkan", 10 * GIGABYTE),
                ],
                want_available: 18 * GIGABYTE,
                want_gpu_free: 22 * GIGABYTE,
                want_system_limited: true,
            },
            Case {
                name: "shared gpu keeps the lower adjusted gpu baseline",
                system_free: 20 * GIGABYTE,
                gpus: vec![igpu("0", "Metal", 12 * GIGABYTE)],
                want_available: 12 * GIGABYTE,
                want_gpu_free: 12 * GIGABYTE,
                want_system_limited: false,
            },
        ];

        for c in cases {
            let sys = SystemInfo {
                free_memory: c.system_free,
                ..Default::default()
            };
            let got = available_memory_for_load(&sys, &c.gpus);
            assert_eq!(got.available, c.want_available, "{}: available", c.name);
            assert_eq!(got.gpu_free, c.want_gpu_free, "{}: gpu_free", c.name);
            assert_eq!(
                got.system_limited, c.want_system_limited,
                "{}: system_limited",
                c.name
            );
        }
    }

    /// The 20% headroom, at the boundary. Multiply-before-divide is what makes
    /// this exact.
    #[test]
    fn the_fit_check_keeps_exactly_twenty_percent_headroom() {
        assert!(fits_with_headroom(80, 100));
        assert!(!fits_with_headroom(81, 100));
        // Zero available fits nothing except zero.
        assert!(fits_with_headroom(0, 0));
        assert!(!fits_with_headroom(1, 0));
    }

    /// **Upstream:** `ml/device.go` `MinimumMemory` plus the per-GPU `available`
    /// line in `load`.
    #[test]
    fn device_overhead_is_metal_512_mib_and_457_mib_everywhere_else() {
        assert_eq!(gpu("0", "Metal", 0).minimum_memory(), 512 * MEBIBYTE);
        assert_eq!(gpu("0", "CUDA", 0).minimum_memory(), 457 * MEBIBYTE);

        let card = gpu("0", "CUDA", 1000 * MEBIBYTE);
        assert_eq!(
            card.available_after_overhead(100 * MEBIBYTE),
            (1000 - 100 - 457) * MEBIBYTE
        );
        // Free below the overheads clamps to zero rather than underflowing.
        let tiny = gpu("0", "CUDA", 100 * MEBIBYTE);
        assert_eq!(tiny.available_after_overhead(0), 0);
    }

    // -- placement ---------------------------------------------------------

    /// **Upstream:** `TestSelectLlamaServerPlacement`, every case.
    #[test]
    fn select_llama_server_placement_compacts_or_splits_the_way_upstream_does() {
        let sys = SystemInfo {
            free_memory: 14 * GIGABYTE,
            ..Default::default()
        };

        struct Case {
            name: &'static str,
            gpus: Vec<DeviceInfo>,
            predicted: u64,
            opts: Options,
            sched_spread: bool,
            want_library: &'static str,
            want_main_gpu: Option<i32>,
            want_selected: usize,
            want_gpu_id: Option<&'static str>,
        }

        let with_main_gpu = |m: i32| {
            let mut o = Options::default();
            o.runner.main_gpu = Some(m);
            o.runner.num_gpu = -1;
            o
        };

        let cases = vec![
            Case {
                name: "compacts onto the largest same-backend GPU",
                gpus: vec![
                    gpu("0", "CUDA", 10 * GIGABYTE),
                    gpu("1", "CUDA", 20 * GIGABYTE),
                ],
                predicted: 8 * GIGABYTE,
                opts: Options::default(),
                sched_spread: false,
                want_library: "CUDA",
                want_main_gpu: Some(0),
                want_selected: 1,
                want_gpu_id: Some("1"),
            },
            Case {
                name: "an explicit main gpu selects the matching backend group",
                gpus: vec![
                    gpu("0", "CUDA", 10 * GIGABYTE),
                    gpu("0", "ROCm", 20 * GIGABYTE),
                    gpu("1", "ROCm", 24 * GIGABYTE),
                ],
                predicted: 8 * GIGABYTE,
                opts: with_main_gpu(1),
                sched_spread: false,
                want_library: "ROCm",
                want_main_gpu: Some(0),
                want_selected: 1,
                want_gpu_id: Some("1"),
            },
            Case {
                name: "an integrated GPU is capped by system free memory",
                gpus: vec![
                    igpu("0", "Metal", 32 * GIGABYTE),
                    gpu("1", "Metal", 16 * GIGABYTE),
                ],
                predicted: 12 * GIGABYTE,
                opts: Options::default(),
                sched_spread: false,
                want_library: "Metal",
                want_main_gpu: Some(0),
                want_selected: 1,
                want_gpu_id: Some("1"),
            },
            Case {
                name: "a discrete GPU beats an integrated one with more memory",
                gpus: vec![
                    igpu("0", "Vulkan", 32 * GIGABYTE),
                    gpu("1", "Vulkan", 10 * GIGABYTE),
                ],
                predicted: 8 * GIGABYTE,
                opts: Options::default(),
                sched_spread: false,
                want_library: "Vulkan",
                want_main_gpu: Some(0),
                want_selected: 1,
                want_gpu_id: Some("1"),
            },
            Case {
                name: "spread disables automatic compaction",
                gpus: vec![
                    gpu("0", "CUDA", 10 * GIGABYTE),
                    gpu("1", "CUDA", 20 * GIGABYTE),
                ],
                predicted: 8 * GIGABYTE,
                opts: Options::default(),
                sched_spread: true,
                want_library: "CUDA",
                want_main_gpu: None,
                want_selected: 2,
                want_gpu_id: None,
            },
            Case {
                name: "no single fit chooses the best backend group to split across",
                gpus: vec![
                    gpu("0", "CUDA", 10 * GIGABYTE),
                    gpu("1", "CUDA", 18 * GIGABYTE),
                    gpu("0", "ROCm", 12 * GIGABYTE),
                ],
                predicted: 30 * GIGABYTE,
                opts: Options::default(),
                sched_spread: false,
                want_library: "CUDA",
                want_main_gpu: None,
                want_selected: 2,
                want_gpu_id: None,
            },
        ];

        for c in cases {
            let (selected, launch_opts) = select_llama_server_placement(
                &sys,
                &c.gpus,
                c.predicted,
                &c.opts,
                c.sched_spread,
            );
            assert_eq!(selected.len(), c.want_selected, "{}: count", c.name);
            assert_eq!(selected[0].library(), c.want_library, "{}: library", c.name);
            if let Some(id) = c.want_gpu_id {
                assert_eq!(selected[0].id(), id, "{}: id", c.name);
            }
            assert_eq!(
                launch_opts.runner.main_gpu, c.want_main_gpu,
                "{}: main_gpu",
                c.name
            );
        }
    }

    /// One device (or CPU-only) short-circuits placement entirely -- and crucially
    /// leaves `main_gpu` untouched.
    #[test]
    fn placement_with_one_device_or_no_gpu_passes_straight_through() {
        let sys = SystemInfo::default();
        let one = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let (sel, opts) = select_llama_server_placement(
            &sys,
            &one,
            GIGABYTE,
            &Options::default(),
            false,
        );
        assert_eq!(sel.len(), 1);
        assert_eq!(opts.runner.main_gpu, None);

        let mut cpu_only = Options::default();
        cpu_only.runner.num_gpu = 0;
        let two = vec![
            gpu("0", "CUDA", 10 * GIGABYTE),
            gpu("1", "CUDA", 20 * GIGABYTE),
        ];
        let (sel, opts) =
            select_llama_server_placement(&sys, &two, GIGABYTE, &cpu_only, false);
        assert_eq!(sel.len(), 2, "num_gpu=0 must not reshuffle the device list");
        assert_eq!(opts.runner.main_gpu, None);
    }

    /// Backends are grouped in first-seen order, which is what makes placement
    /// reproducible run to run.
    #[test]
    fn by_library_groups_backends_in_first_seen_order() {
        let gpus = vec![
            gpu("0", "CUDA", 1),
            gpu("0", "ROCm", 1),
            gpu("1", "CUDA", 1),
        ];
        let groups = by_library(&gpus);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0][0].library(), "CUDA");
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1][0].library(), "ROCm");
    }

    /// **Upstream:** `explicitPartialGPUOffload`. A full offload need
    /// `block_count + 1` layers.
    #[test]
    fn an_explicit_layer_count_below_block_count_plus_one_is_a_partial_offload() {
        let mut opts = Options::default();

        opts.runner.num_gpu = 20;
        assert!(explicit_partial_gpu_offload(&opts, 20));
        opts.runner.num_gpu = 21;
        assert!(!explicit_partial_gpu_offload(&opts, 20));

        // -1 mean "you decide", which is not an explicit partial offload.
        opts.runner.num_gpu = -1;
        assert!(!explicit_partial_gpu_offload(&opts, 20));
        // Unknown block count cannot answer the question.
        opts.runner.num_gpu = 5;
        assert!(!explicit_partial_gpu_offload(&opts, 0));
    }

    // -- generation batch --------------------------------------------------

    /// **Upstream:** `TestAutomaticGenerationBatch`, every case.
    #[test]
    fn automatic_generation_batch_matches_upstreams_table() {
        struct Case {
            name: &'static str,
            effective_ctx: u64,
            predicted: u64,
            available: u64,
            flash: FlashAttentionType,
            gpus: Vec<DeviceInfo>,
            want: u32,
        }

        let cases = vec![
            Case {
                name: "small context keeps the default",
                effective_ctx: 4096,
                predicted: 0,
                available: 0,
                flash: FlashAttentionType::Auto,
                gpus: vec![],
                want: 512,
            },
            Case {
                name: "medium context uses 1024 with unknown memory",
                effective_ctx: 32768,
                predicted: 0,
                available: 0,
                flash: FlashAttentionType::Auto,
                gpus: vec![],
                want: 1024,
            },
            Case {
                name: "large context uses 2048 when there is headroom",
                effective_ctx: 131072,
                predicted: 8 * GIBIBYTE,
                available: 14 * GIBIBYTE,
                flash: FlashAttentionType::Auto,
                gpus: vec![],
                want: 2048,
            },
            Case {
                name: "large context steps down to 1024 without 2048 headroom",
                effective_ctx: 131072,
                predicted: 9 * GIBIBYTE,
                available: 14 * GIBIBYTE,
                flash: FlashAttentionType::Auto,
                gpus: vec![],
                want: 1024,
            },
            Case {
                name: "large context steps down to 1024 for headroom",
                effective_ctx: 131072,
                predicted: 8 * GIBIBYTE,
                available: 11 * GIBIBYTE,
                flash: FlashAttentionType::Auto,
                gpus: vec![],
                want: 1024,
            },
            Case {
                name: "medium context steps down to 512 for headroom",
                effective_ctx: 32768,
                predicted: 8500 * MEBIBYTE,
                available: 11 * GIBIBYTE,
                flash: FlashAttentionType::Auto,
                gpus: vec![],
                want: 512,
            },
            Case {
                name: "flash attention disabled suppresses promotion",
                effective_ctx: 131072,
                predicted: 8 * GIBIBYTE,
                available: 14 * GIBIBYTE,
                flash: FlashAttentionType::Disabled,
                gpus: vec![gpu("0", "CUDA", 14 * GIBIBYTE)],
                want: 512,
            },
            Case {
                name: "constrained CUDA without flash attention uses a smaller batch",
                effective_ctx: 131072,
                predicted: 3 * GIBIBYTE,
                available: 6 * GIBIBYTE,
                flash: FlashAttentionType::Disabled,
                gpus: vec![gpu("0", "CUDA", 6 * GIBIBYTE)],
                want: 256,
            },
        ];

        for c in cases {
            let got = automatic_generation_batch(
                c.effective_ctx,
                c.predicted,
                c.available,
                c.flash,
                &c.gpus,
            );
            assert_eq!(got, c.want, "{}", c.name);
        }
    }

    /// The surcharge ladder, and the fact that it is zero for an embedding model.
    #[test]
    fn the_generation_batch_surcharge_only_applies_to_completion_models() {
        assert_eq!(generation_batch_surcharge(2048), 2 * GIBIBYTE);
        assert_eq!(generation_batch_surcharge(1024), 768 * MEBIBYTE);
        assert_eq!(generation_batch_surcharge(512), 0);
        assert_eq!(generation_batch_surcharge_for_completion(false, 2048), 0);
        assert_eq!(
            generation_batch_surcharge_for_completion(true, 2048),
            2 * GIBIBYTE
        );
    }

    /// Stepping down never goes below 512 -- 256 is reachable only via the CUDA
    /// no-flash-attention path.
    #[test]
    fn the_batch_ladder_never_steps_below_the_default() {
        assert_eq!(next_lower_generation_batch(2048), 1024);
        assert_eq!(next_lower_generation_batch(1024), 512);
        assert_eq!(next_lower_generation_batch(512), 512);
    }

    /// **Upstream:** `nextLowerAutoNumCtx`. 4096 is the floor.
    #[test]
    fn the_automatic_context_ladder_bottoms_out_at_4096() {
        assert_eq!(next_lower_auto_num_ctx(262144), Some(32768));
        assert_eq!(next_lower_auto_num_ctx(32769), Some(32768));
        assert_eq!(next_lower_auto_num_ctx(32768), Some(4096));
        assert_eq!(next_lower_auto_num_ctx(4096), None);
    }

    /// A small CUDA card only counts as constrained past 4096 context, and the
    /// memory reading falls back to total when free is nonsense.
    #[test]
    fn a_constrained_cuda_card_is_eight_gibibytes_or_less_past_4096_context() {
        let small = vec![gpu("0", "CUDA", 8 * GIBIBYTE)];
        assert!(constrained_cuda_without_flash_attention(8192, &small));
        assert!(!constrained_cuda_without_flash_attention(4096, &small));

        let big = vec![gpu("0", "CUDA", 24 * GIBIBYTE)];
        assert!(!constrained_cuda_without_flash_attention(8192, &big));

        // free = 0 -> fall back to total.
        let mut no_free = gpu("0", "CUDA", 0);
        no_free.total_memory = 6 * GIBIBYTE;
        assert!(constrained_cuda_without_flash_attention(8192, &[no_free]));

        // Non-CUDA cards never count.
        let rocm = vec![gpu("0", "ROCm", 6 * GIBIBYTE)];
        assert!(!constrained_cuda_without_flash_attention(8192, &rocm));
    }

    // -- mmap --------------------------------------------------------------

    /// **Upstream:** `TestDisableMmapDefaultReason`, every case.
    #[test]
    fn disable_mmap_default_reason_matches_upstreams_table() {
        let explicit_on = {
            let mut o = Options::default();
            o.runner.num_gpu = -1;
            o.runner.use_mmap = Some(true);
            o
        };
        let auto = |num_gpu: i32| {
            let mut o = Options::default();
            o.runner.num_gpu = num_gpu;
            o
        };

        struct Case {
            name: &'static str,
            goos: &'static str,
            opts: Options,
            gpus: Vec<DeviceInfo>,
            block_count: u64,
            predicted: u64,
            available: u64,
            want: Option<MmapDisableReason>,
        }

        let cases = vec![
            Case {
                name: "an explicit use_mmap=true wins",
                goos: "windows",
                opts: explicit_on,
                gpus: vec![gpu("0", "CUDA", 0)],
                block_count: 0,
                predicted: 0,
                available: 0,
                want: None,
            },
            Case {
                name: "a cpu-only request disables mmap",
                goos: "linux",
                opts: auto(0),
                gpus: vec![gpu("0", "CUDA", 0)],
                block_count: 0,
                predicted: 0,
                available: 0,
                want: Some(MmapDisableReason::Cpu),
            },
            Case {
                name: "no GPU devices disables mmap",
                goos: "linux",
                opts: auto(-1),
                gpus: vec![],
                block_count: 0,
                predicted: 0,
                available: 0,
                want: Some(MmapDisableReason::Cpu),
            },
            Case {
                name: "windows cuda disables mmap",
                goos: "windows",
                opts: auto(-1),
                gpus: vec![gpu("0", "CUDA", 0)],
                block_count: 0,
                predicted: 0,
                available: 0,
                want: Some(MmapDisableReason::WindowsCuda),
            },
            Case {
                name: "metal partial offload disables mmap",
                goos: "darwin",
                opts: auto(10),
                gpus: vec![gpu("0", "Metal", 0)],
                block_count: 20,
                predicted: 0,
                available: 0,
                want: Some(MmapDisableReason::MetalPartialOffload),
            },
            Case {
                name: "metal full offload keeps the default",
                goos: "darwin",
                opts: auto(21),
                gpus: vec![gpu("0", "Metal", 0)],
                block_count: 20,
                predicted: 0,
                available: 0,
                want: None,
            },
            Case {
                name: "metal auto partial offload disables mmap",
                goos: "darwin",
                opts: auto(-1),
                gpus: vec![gpu("0", "Metal", 0)],
                block_count: 0,
                predicted: 30 * GIGABYTE,
                available: 20 * GIGABYTE,
                want: Some(MmapDisableReason::MetalPartialOffload),
            },
            Case {
                name: "metal auto full offload keeps the default",
                goos: "darwin",
                opts: auto(-1),
                gpus: vec![gpu("0", "Metal", 0)],
                block_count: 0,
                predicted: 10 * GIGABYTE,
                available: 20 * GIGABYTE,
                want: None,
            },
            Case {
                name: "linux cuda keeps the default",
                goos: "linux",
                opts: auto(-1),
                gpus: vec![gpu("0", "CUDA", 0)],
                block_count: 0,
                predicted: 0,
                available: 0,
                want: None,
            },
        ];

        for c in cases {
            let got = disable_mmap_default_reason(
                c.goos,
                &c.opts,
                &c.gpus,
                c.block_count,
                c.predicted,
                c.available,
            );
            assert_eq!(got, c.want, "{}", c.name);
        }
    }

    /// **Upstream:** `TestDisableMmapForHostPressure`, every assertion.
    #[test]
    fn disable_mmap_for_host_pressure_is_linux_only_and_backs_off_when_vram_is_tight() {
        let gpus = vec![DeviceInfo {
            device_id: DeviceId::new("0", "CUDA"),
            total_memory: 100 * GIGABYTE,
            free_memory: 80 * GIGABYTE,
            ..Default::default()
        }];
        let sys = SystemInfo {
            total_memory: 100 * GIGABYTE,
            free_memory: 50 * GIGABYTE,
            free_swap: 0,
        };
        let auto = Options::default();

        assert!(
            disable_mmap_for_host_pressure(
                "linux",
                &auto,
                &sys,
                &gpus,
                20 * GIGABYTE,
                25 * GIGABYTE,
                30 * GIGABYTE,
                80 * GIGABYTE,
            ),
            "20 + 25 + 10 headroom > 50 free -- back off mmap"
        );

        let explicit = {
            let mut o = Options::default();
            o.runner.use_mmap = Some(true);
            o
        };
        assert!(
            !disable_mmap_for_host_pressure(
                "linux",
                &explicit,
                &sys,
                &gpus,
                20 * GIGABYTE,
                25 * GIGABYTE,
                30 * GIGABYTE,
                80 * GIGABYTE,
            ),
            "an explicit use_mmap=true should win"
        );

        assert!(
            !disable_mmap_for_host_pressure(
                "darwin",
                &auto,
                &sys,
                &gpus,
                20 * GIGABYTE,
                25 * GIGABYTE,
                30 * GIGABYTE,
                80 * GIGABYTE,
            ),
            "only the Linux pressure heuristic is restored"
        );

        let mut igpus = gpus.clone();
        igpus[0].integrated = true;
        assert!(
            !disable_mmap_for_host_pressure(
                "linux",
                &auto,
                &sys,
                &igpus,
                20 * GIGABYTE,
                25 * GIGABYTE,
                30 * GIGABYTE,
                80 * GIGABYTE,
            ),
            "shared-memory GPU loads keep the normal mmap path"
        );

        assert!(
            !disable_mmap_for_host_pressure(
                "linux",
                &auto,
                &sys,
                &gpus,
                20 * GIGABYTE,
                25 * GIGABYTE,
                70 * GIGABYTE,
                80 * GIGABYTE,
            ),
            "when VRAM is tight, dropping mmap could make partial CPU offload worse"
        );
    }

    /// **Upstream:** `mmapHostPressureHeadroom` -- `max(8 GB, total/10)`.
    #[test]
    fn the_mmap_host_pressure_headroom_is_eight_gigabytes_or_a_tenth_of_ram() {
        assert_eq!(mmap_host_pressure_headroom(0), 8 * GIGABYTE);
        assert_eq!(mmap_host_pressure_headroom(16 * GIGABYTE), 8 * GIGABYTE);
        assert_eq!(
            mmap_host_pressure_headroom(200 * GIGABYTE),
            20 * GIGABYTE
        );
    }

    /// Rust says "macos", Go says "darwin". Getting this wrong silently disables
    /// the whole Metal mmap branch on a Mac.
    #[test]
    fn host_goos_reports_darwin_not_macos() {
        assert_ne!(host_goos(), "macos");
        assert!(matches!(
            host_goos(),
            "darwin" | "linux" | "windows" | "android" | "freebsd" | "ios"
        ));
    }

    // -- VRAM prediction ---------------------------------------------------

    /// **Upstream:** `llm/llama_server.go` `PredictServerVRAM`, arithmetic checked
    /// term by term.
    #[test]
    fn predict_server_vram_is_the_weights_plus_an_f16_kv_cache() {
        let mut kv = Kv::new();
        kv.insert("general.architecture", "llama")
            .insert("llama.block_count", 32u32)
            .insert("llama.embedding_length", 4096u32)
            .insert("llama.attention.head_count", 32u32)
            .insert("llama.attention.head_count_kv", 8u32);

        // head_dim = 4096 / 32 = 128
        // kv_cache = 2 * 32 layers * 8 kv_heads * 128 * 4096 ctx * 2 bytes
        let expected_cache = 2u64 * 32 * 8 * 128 * 4096 * 2;
        assert_eq!(
            predict_server_vram(GIGABYTE, &kv, 4096),
            GIGABYTE + expected_cache
        );

        // Doubling the aggregate context doubles only the cache term.
        assert_eq!(
            predict_server_vram(GIGABYTE, &kv, 8192),
            GIGABYTE + expected_cache * 2
        );
    }

    /// Missing head metadata must not blow up -- upstream leave `headDim` at Go's
    /// zero value and the KV term simply vanishes.
    #[test]
    fn predict_server_vram_falls_back_to_weights_only_when_head_metadata_is_missing() {
        let mut kv = Kv::new();
        kv.insert("general.architecture", "llama")
            .insert("llama.block_count", 32u32);
        assert_eq!(predict_server_vram(GIGABYTE, &kv, 4096), GIGABYTE);
    }

    // -- VRAM recovery -----------------------------------------------------

    /// **Upstream:** the convergence test in `waitForVRAMRecovery`.
    #[test]
    fn vram_recovery_converges_once_three_quarters_has_come_back() {
        // 8 GiB runner: 6 GiB back is exactly 75%, which is NOT yet ">".
        assert!(!vram_recovery_converged(0, 6 * GIBIBYTE, 8 * GIBIBYTE));
        assert!(vram_recovery_converged(0, 7 * GIBIBYTE, 8 * GIBIBYTE));
    }

    /// The divergence from upstream: free memory going **down** must keep us
    /// waiting, not wrap around into a false convergence.
    #[test]
    fn vram_recovery_does_not_declare_victory_when_free_memory_went_down() {
        assert!(!vram_recovery_converged(
            10 * GIBIBYTE,
            2 * GIBIBYTE,
            8 * GIBIBYTE
        ));
    }

    // -- scheduler lifecycle ----------------------------------------------

    /// **Upstream:** `TestSchedInit`.
    #[test]
    fn a_new_scheduler_starts_with_nothing_loaded_and_nothing_queued() {
        let sched = Scheduler::default();
        assert_eq!(sched.loaded_count(), 0);
        assert_eq!(sched.pending_len(), 0);
        assert!(sched.find_runner_to_unload().is_none());
    }

    /// **Upstream:** `TestSchedUseLoadedRunner` -- refcount to 1, and the
    /// request's keep-alive replaces the runner's.
    #[test]
    fn using_a_loaded_runner_bumps_the_refcount_and_adopts_the_requests_keep_alive() {
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(10).arc(),
            Expiry::After(Duration::from_secs(1)),
            vec![],
            true,
        );
        assert_eq!(sched.runner(&key).unwrap().ref_count, 0);

        let mut req = LlmRequest::new(completion_model("/fake/model"), Options::default());
        req.session_duration = Some(Expiry::After(Duration::from_secs(2)));
        assert!(sched.use_loaded_runner(&key, &req).is_some());

        let r = sched.runner(&key).unwrap();
        assert_eq!(r.ref_count, 1);
        assert_eq!(r.session_duration, Expiry::After(Duration::from_secs(2)));
        assert!(
            r.expires_at.is_none(),
            "a busy runner must have no deadline at all, not merely a later one"
        );
    }

    /// **Upstream:** `TestSchedRequestsSameModelSameRequest` -- a second request
    /// for the same model, unchanged, reuses the runner without queueing.
    #[test]
    fn a_second_request_for_an_unchanged_loaded_model_reuses_the_runner() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(10).arc(),
            Expiry::After(Duration::from_secs(5)),
            vec![],
            true,
        );

        let req = LlmRequest::new(completion_model("/fake/model"), Options::default());
        assert!(sched.acquire_if_loaded(&req).is_some());
        assert_eq!(sched.runner("/fake/model").unwrap().ref_count, 1);
        assert_eq!(sched.loaded_count(), 1, "no second copy got loaded");
    }

    /// The refcount invariant, both halves: it goes back to zero and arms the
    /// deadline, and a double release clamps rather than wrapping.
    #[test]
    fn finishing_the_last_request_arms_the_keep_alive_deadline() {
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(10).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            false,
        );
        assert_eq!(sched.runner(&key).unwrap().ref_count, 1);

        let t0 = Instant::now();
        sched.request_finished(&key, t0);
        let r = sched.runner(&key).unwrap();
        assert_eq!(r.ref_count, 0);
        assert_eq!(r.expires_at, Some(t0 + Duration::from_secs(300)));

        // A stray extra release must not wrap the count to u32::MAX and pin the
        // model in VRAM forever -- our deliberate divergence from upstream's
        // unsigned `refCount--`.
        sched.request_finished(&key, t0);
        assert_eq!(sched.runner(&key).unwrap().ref_count, 0);
    }

    /// A keep-alive of "forever" must never arm a deadline at all.
    #[test]
    fn a_keep_forever_runner_never_gets_an_expiry_deadline() {
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(10).arc(),
            Expiry::Never,
            vec![],
            true,
        );
        assert!(sched.runner(&key).unwrap().expires_at.is_none());
        assert!(
            sched
                .take_expired(Instant::now() + Duration::from_secs(86_400))
                .is_empty()
        );
    }

    /// A zero keep-alive means "go the moment nobody is holding you".
    #[test]
    fn a_zero_keep_alive_runner_expires_the_moment_it_goes_idle() {
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(10).arc(),
            Expiry::After(Duration::ZERO),
            vec![],
            false,
        );
        let t0 = Instant::now();
        assert!(sched.take_expired(t0).is_empty(), "still held, must not go");

        sched.request_finished(&key, t0);
        let gone = sched.take_expired(t0);
        assert_eq!(gone.len(), 1);
        assert_eq!(gone[0].key, key);
        assert_eq!(sched.loaded_count(), 0);
    }

    /// **The refcount invariant, stated as a test.** However overdue a runner is,
    /// it is never taken while somebody is using it.
    #[test]
    fn a_busy_runner_is_never_taken_by_take_expired_however_overdue_it_is() {
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(10).arc(),
            Expiry::After(Duration::from_millis(1)),
            vec![],
            false,
        );
        let t0 = Instant::now();
        sched.expire_runner(&key, t0);
        assert!(
            sched
                .take_expired(t0 + Duration::from_secs(3600))
                .is_empty(),
            "ref_count is 1 -- an hour overdue changes nothing"
        );

        sched.request_finished(&key, t0);
        assert_eq!(sched.take_expired(t0).len(), 1);
    }

    /// **Upstream:** `TestSchedExpireRunner` -- marking a runner expires it and
    /// it leaves the map.
    #[test]
    fn expiring_an_idle_runner_unloads_it_and_drops_it_from_the_map() {
        let mut sched = Scheduler::default();
        let runner = Arc::new(MockRunner::new(10));
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            runner.clone() as Arc<dyn ModelRunner>,
            Expiry::After(Duration::from_secs(120)),
            vec![],
            true,
        );
        assert_eq!(sched.loaded_count(), 1);

        let t0 = Instant::now();
        sched.expire_runner(&key, t0);
        let gone = sched.take_expired(t0);
        assert_eq!(gone.len(), 1);
        assert_eq!(sched.loaded_count(), 0);
        assert_eq!(
            runner.unload_calls.load(Ordering::SeqCst),
            1,
            "the scheduler closes the runner exactly once"
        );
    }

    /// **Upstream:** `TestSchedFindRunnerToUnload` -- the idle one wins even
    /// though it sorts second; when nothing is idle, the shortest keep-alive wins.
    #[test]
    fn find_runner_to_unload_prefers_an_idle_runner_then_the_shortest_keep_alive() {
        let mut sched = Scheduler::default();
        // r1: shorter keep-alive, but busy.
        let busy = load_model(
            &mut sched,
            completion_model("/a"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_nanos(1)),
            vec![],
            false,
        );
        // r2: longer keep-alive, idle.
        let idle = load_model(
            &mut sched,
            completion_model("/b"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_nanos(2)),
            vec![],
            true,
        );

        assert_eq!(sched.find_runner_to_unload().as_deref(), Some(idle.as_str()));

        // Now nothing is idle: fall back to the head of the sort -- the shortest
        // keep-alive, i.e. the one that was going soonest anyway.
        let mut req = LlmRequest::new(completion_model("/b"), Options::default());
        req.session_duration = Some(Expiry::After(Duration::from_nanos(2)));
        sched.use_loaded_runner(&idle, &req);
        assert_eq!(sched.find_runner_to_unload().as_deref(), Some(busy.as_str()));
    }

    /// The `uint64` cast upstream make on `sessionDuration`: a keep-forever
    /// runner must sort **last**, so it is the last thing thrown away.
    #[test]
    fn a_keep_forever_runner_is_the_last_candidate_for_eviction() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/forever"),
            MockRunner::new(1).arc(),
            Expiry::Never,
            vec![],
            false,
        );
        load_model(
            &mut sched,
            completion_model("/brief"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(30)),
            vec![],
            false,
        );
        // Neither is idle, so the sort order alone decides.
        assert_eq!(sched.find_runner_to_unload().as_deref(), Some("/brief"));
    }

    /// **Upstream:** `TestSchedUpdateFreeSpace`.
    #[test]
    fn update_free_space_subtracts_every_loaded_runners_per_device_vram() {
        let d1 = DeviceId::new("1", "");
        let d2 = DeviceId::new("2", "");

        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/a"),
            MockRunner::new(0)
                .with_vram_by_gpu(&[(d1.clone(), 50), (d2.clone(), 50)])
                .arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![d1.clone(), d2.clone()],
            true,
        );
        load_model(
            &mut sched,
            completion_model("/b"),
            MockRunner::new(0)
                .with_vram_by_gpu(&[(d1.clone(), 125), (d2.clone(), 75)])
                .arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![d1.clone(), d2.clone()],
            true,
        );

        let mut gpus = vec![
            DeviceInfo {
                device_id: d1,
                total_memory: 1000,
                free_memory: 900,
                ..Default::default()
            },
            DeviceInfo {
                device_id: d2,
                total_memory: 2000,
                free_memory: 1900,
                ..Default::default()
            },
        ];
        sched.update_free_space(&mut gpus);
        assert_eq!(gpus[0].free_memory, 1000 - 50 - 125);
        assert_eq!(gpus[1].free_memory, 2000 - 50 - 75);
    }

    /// **Upstream:** `TestSchedulerTracksMultipleLoadedRunners`.
    #[test]
    fn the_scheduler_tracks_multiple_loaded_runners_on_one_device() {
        let metal = DeviceId::new("", "Metal");
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/fake/first/model"),
            MockRunner::new(8 * GIGABYTE)
                .with_vram_by_gpu(&[(metal.clone(), 8 * GIGABYTE)])
                .arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![metal.clone()],
            true,
        );
        load_model(
            &mut sched,
            completion_model("/fake/second/model"),
            MockRunner::new(4 * GIGABYTE)
                .with_vram_by_gpu(&[(metal.clone(), 4 * GIGABYTE)])
                .arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![metal.clone()],
            true,
        );
        assert_eq!(sched.loaded_count(), 2);

        let mut gpus = vec![DeviceInfo {
            device_id: metal,
            total_memory: 24 * GIGABYTE,
            free_memory: 24 * GIGABYTE,
            ..Default::default()
        }];
        sched.update_free_space(&mut gpus);
        assert_eq!(gpus[0].free_memory, 12 * GIGABYTE);
    }

    /// **Upstream:** `TestSchedNeedsReload`, step for step.
    #[test]
    fn needs_reload_walks_adapters_projectors_options_ping_and_context_shift() {
        let loaded_model = ModelRef {
            name: "test".into(),
            adapter_paths: vec!["adapter1".into()],
            projector_paths: vec!["projector1".into()],
            capabilities: vec![Capability::Completion],
            ..Default::default()
        };
        let runner = Arc::new(MockRunner::new(0));
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            loaded_model.clone(),
            runner.clone() as Arc<dyn ModelRunner>,
            Expiry::After(Duration::from_secs(60)),
            vec![],
            true,
        );

        let mut req = LlmRequest::new(
            ModelRef {
                adapter_paths: vec!["adapter2".into()],
                projector_paths: vec!["projector2".into()],
                ..loaded_model.clone()
            },
            Options::default(),
        );
        // The runner was loaded with context shift resolved from its (non-deepseek)
        // architecture, i.e. on. Start the request matching, so that the final
        // step below isolates the context-shift check.
        req.context_shift = true;

        assert!(sched.needs_reload(&key, &req), "adapters differ");

        req.model.adapter_paths = loaded_model.adapter_paths.clone();
        assert!(sched.needs_reload(&key, &req), "projectors differ");

        req.model.projector_paths = loaded_model.projector_paths.clone();
        req.opts.runner.num_batch = 1234;
        assert!(sched.needs_reload(&key, &req), "load-time options differ");

        req.opts.runner.num_batch = Options::default().runner.num_batch;
        runner.ping_ok.store(false, Ordering::SeqCst);
        assert!(sched.needs_reload(&key, &req), "the runner stopped answering");

        runner.ping_ok.store(true, Ordering::SeqCst);
        assert!(!sched.needs_reload(&key, &req), "everything matches now");

        req.opts.runner.num_gpu = 99;
        assert!(sched.needs_reload(&key, &req), "an explicit layer count differs");

        req.opts.runner.num_gpu = -1;
        assert!(
            !sched.needs_reload(&key, &req),
            "num_gpu=-1 means 'you decide', so it never counts as a mismatch"
        );

        req.context_shift = false;
        assert!(sched.needs_reload(&key, &req), "the context-shift setting differs");
    }

    /// **Upstream:** `TestSchedNeedsReloadIgnoresAutomaticNumCtxClamp` and its two
    /// siblings for num_batch and use_mmap.
    ///
    /// The point: a value **we** chose is not a value the user asked for, so our
    /// own automatic tier moving must never force a reload.
    #[test]
    fn needs_reload_ignores_values_that_both_sides_chose_automatically() {
        let model = ModelRef {
            name: "test".into(),
            capabilities: vec![Capability::Completion],
            ..Default::default()
        };
        let mut sched = Scheduler::default();

        let mut load_req = LlmRequest::new(model.clone(), Options::default());
        load_req.num_ctx_auto = true;
        load_req.num_batch_auto = true;
        load_req.use_mmap_auto = true;
        load_req.opts.runner.num_ctx = 32768;
        load_req.opts.runner.num_batch = 1024;
        load_req.opts.runner.use_mmap = Some(false);

        let plan = LoadPlan {
            key: model.scheduler_key(),
            num_parallel: 1,
            session_duration: Expiry::After(Duration::from_secs(60)),
            gpus: vec![],
            launch_opts: load_req.opts.clone(),
            require_full: false,
            predicted_vram: 0,
            effective_num_ctx: 32768,
            completion: true,
            context_shift: false,
            mmap_disabled_reason: None,
        };
        let key = plan.key.clone();
        sched.runner_loaded(
            &mut load_req,
            &plan,
            MockRunner::new(0).arc(),
            vec![],
        );
        sched.request_finished(&key, Instant::now());

        // A later automatic pass landed on different numbers. Same request, same
        // model -- must NOT reload.
        let mut req = LlmRequest::new(model.clone(), Options::default());
        req.num_ctx_auto = true;
        req.num_batch_auto = true;
        req.opts.runner.num_ctx = 4096;
        req.opts.runner.num_batch = 512;
        req.opts.runner.use_mmap = None;
        assert!(!sched.needs_reload(&key, &req));

        // But an EXPLICIT user choice is a real mismatch and must reload.
        let mut explicit = req.clone();
        explicit.num_ctx_auto = false;
        explicit.opts.runner.num_ctx = 4096;
        assert!(sched.needs_reload(&key, &explicit));
    }

    /// A runner still coming up gets the two-minute ping window, not ten seconds.
    #[test]
    fn a_runner_still_loading_gets_the_longer_health_ping_window() {
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/fake/model"),
            MockRunner::new(0).arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![],
            true,
        );
        assert_eq!(
            sched.runner(&key).unwrap().ping_timeout(),
            NEEDS_RELOAD_PING_TIMEOUT
        );
        sched.set_loading(&key, true);
        assert_eq!(
            sched.runner(&key).unwrap().ping_timeout(),
            NEEDS_RELOAD_LOADING_PING_TIMEOUT
        );
    }

    /// **Upstream:** `ErrMaxQueue`. A full queue is an immediate error, never a
    /// wait.
    #[test]
    fn the_pending_queue_rejects_once_it_is_full() {
        let mut sched = Scheduler::new(SchedulerConfig {
            max_queue: 2,
            ..Default::default()
        });
        let req = || LlmRequest::new(completion_model("/fake/model"), Options::default());
        assert!(sched.enqueue(req()).is_ok());
        assert!(sched.enqueue(req()).is_ok());
        assert_eq!(sched.enqueue(req()), Err(SchedError::MaxQueue));

        // Taking one off makes room again, and bumps the attempt count.
        let taken = sched.next_pending().expect("a request was queued");
        assert_eq!(taken.sched_attempts, 1);
        assert!(sched.enqueue(req()).is_ok());
    }

    /// **Upstream:** `TestSchedUnloadAllRunners`. Shutdown ignores refcounts --
    /// the process is going away regardless.
    #[test]
    fn unload_all_clears_every_runner_even_the_busy_ones() {
        let mut sched = Scheduler::default();
        let busy = Arc::new(MockRunner::new(1));
        load_model(
            &mut sched,
            completion_model("/a"),
            busy.clone() as Arc<dyn ModelRunner>,
            Expiry::After(Duration::from_secs(60)),
            vec![],
            false,
        );
        load_model(
            &mut sched,
            completion_model("/b"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![],
            true,
        );
        assert_eq!(sched.loaded_count(), 2);

        sched.unload_all();
        assert_eq!(sched.loaded_count(), 0);
        assert_eq!(busy.unload_calls.load(Ordering::SeqCst), 1);
    }

    /// **Upstream:** `TestSchedRuntimeOOMExpiresLoadedRunners`. An OOM during
    /// generation means the fit prediction was already proven wrong, so clear the
    /// board.
    #[test]
    fn a_runtime_oom_expires_every_loaded_runner() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/a"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );
        load_model(
            &mut sched,
            completion_model("/b"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let t0 = Instant::now();
        let marked = sched.expire_all_for_runtime_oom(t0);
        assert_eq!(marked.len(), 2);
        assert_eq!(sched.take_expired(t0).len(), 2);
        assert_eq!(sched.loaded_count(), 0);
    }

    /// **Upstream:** `evictAllAndWait` -- everything except the model being loaded.
    #[test]
    fn evict_all_except_spares_the_model_being_loaded() {
        let mut sched = Scheduler::default();
        for path in ["/a", "/b", "/keep"] {
            load_model(
                &mut sched,
                completion_model(path),
                MockRunner::new(1).arc(),
                Expiry::After(Duration::from_secs(300)),
                vec![],
                true,
            );
        }
        let t0 = Instant::now();
        let victims = sched.evict_all_except("/keep", t0);
        assert_eq!(victims.len(), 2);
        assert!(!victims.contains(&"/keep".to_string()));
        sched.take_expired(t0);
        assert_eq!(sched.loaded_keys(), vec!["/keep".to_string()]);
    }

    // -- num_parallel ------------------------------------------------------

    /// **Upstream:** the three clamps at the top of `load`.
    #[test]
    fn embedding_models_and_unsafe_architectures_always_get_one_sequence() {
        let sched = Scheduler::new(SchedulerConfig {
            num_parallel: 4,
            ..Default::default()
        });

        let completion = completion_model("/m");
        assert_eq!(sched.resolve_num_parallel(&completion), 4);

        let embedding = ModelRef {
            capabilities: vec![Capability::Embedding],
            ..completion.clone()
        };
        assert_eq!(
            sched.resolve_num_parallel(&embedding),
            1,
            "no completion capability -> no generation loop to overlap"
        );

        for arch in UNSAFE_PARALLEL_ARCHITECTURES {
            let unsafe_model = ModelRef {
                model_family: (*arch).to_string(),
                ..completion.clone()
            };
            assert_eq!(
                sched.resolve_num_parallel(&unsafe_model),
                1,
                "{arch} is not parallel-safe"
            );
        }

        // A zero setting still means one, never zero.
        let auto = Scheduler::default();
        assert_eq!(auto.resolve_num_parallel(&completion), 1);
    }

    /// **Upstream:** the `maxRunners <= 0` auto-derivation in `processPending`.
    #[test]
    fn the_runner_cap_defaults_to_three_per_gpu_and_is_cached() {
        let mut sched = Scheduler::default();
        assert_eq!(sched.resolve_max_runners(2), 6);
        // Cached: a later GPU count must not move the cap under a live workload.
        assert_eq!(sched.resolve_max_runners(8), 6);

        // CPU-only still gets a cap of 3, not 0 (which would mean unlimited).
        let mut cpu = Scheduler::default();
        assert_eq!(cpu.resolve_max_runners(0), 3);

        // An explicit setting always wins.
        let mut explicit = Scheduler::new(SchedulerConfig {
            max_runners: 1,
            ..Default::default()
        });
        assert_eq!(explicit.resolve_max_runners(8), 1);
    }

    // -- the decision ladder ----------------------------------------------

    fn env<'a>(
        sys: &'a SystemInfo,
        gpus: &'a mut Vec<DeviceInfo>,
        predict: &'a dyn Fn(u64) -> u64,
    ) -> LoadEnv<'a> {
        LoadEnv {
            system_info: sys,
            gpus,
            flash_attention: FlashAttentionType::Auto,
            goos: "linux",
            predict_vram: predict,
        }
    }

    /// **Upstream:** *"No models loaded. Load the model but prefer the best fit."*
    /// -- the first load may spill to CPU, because evicting would help nobody.
    #[test]
    fn the_first_load_is_allowed_to_offload_only_partially() {
        let mut sched = Scheduler::default();
        let sys = SystemInfo {
            free_memory: 32 * GIGABYTE,
            total_memory: 32 * GIGABYTE,
            free_swap: 0,
        };
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 100 * GIGABYTE;
        let mut req = LlmRequest::new(completion_model("/first"), Options::default());

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Load(plan) => assert!(
                !plan.require_full,
                "nothing else resident, so a partial offload is acceptable"
            ),
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// **Upstream:** `TestSchedLlamaServerFitsAlongside` -- a second model that
    /// fits alongside is simply loaded.
    #[test]
    fn a_second_model_that_fits_alongside_is_loaded_without_evicting() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(GIGABYTE).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| GIGABYTE;
        let mut req = LlmRequest::new(completion_model("/newcomer"), Options::default());

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Load(plan) => assert!(
                plan.require_full,
                "something is resident, so the newcomer must fit fully"
            ),
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// **Upstream:** `TestSchedLlamaServerEvictsWhenVRAMInsufficient`.
    #[test]
    fn a_second_model_that_does_not_fit_evicts_the_resident_one_first() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(GIGABYTE).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 100 * GIGABYTE;
        let mut req = LlmRequest::new(completion_model("/newcomer"), Options::default());

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Evict { key } => assert_eq!(key, "/resident"),
            other => panic!("expected Evict, got {other:?}"),
        }
        // Marked, idle, therefore collectable straight away.
        assert_eq!(sched.take_expired(Instant::now()).len(), 1);
    }

    /// **Upstream:** `TestSchedLlamaServerExplicitPartialNumGPUSkipsFullFitEviction`
    /// -- the user already said they are happy to spill to CPU.
    #[test]
    fn an_explicit_partial_offload_skips_the_full_fit_eviction() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(GIGABYTE).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 100 * GIGABYTE;

        let model = ModelRef {
            block_count: 40,
            ..completion_model("/newcomer")
        };
        let mut opts = Options::default();
        opts.runner.num_gpu = 10; // 10 < 40 + 1 -- a deliberate partial offload.
        let mut req = LlmRequest::new(model, opts);

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Load(plan) => assert!(plan.require_full),
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// **Upstream:** the OOM retry path -- once a request has spent its retry, a
    /// second squeeze evicts everything rather than picking one victim.
    #[test]
    fn a_request_that_already_spent_its_oom_retry_evicts_everything() {
        let mut sched = Scheduler::default();
        for path in ["/a", "/b"] {
            load_model(
                &mut sched,
                completion_model(path),
                MockRunner::new(GIGABYTE).arc(),
                Expiry::After(Duration::from_secs(300)),
                vec![],
                true,
            );
        }

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 100 * GIGABYTE;
        let mut req = LlmRequest::new(completion_model("/newcomer"), Options::default());
        req.oom_retry_attempted = true;

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::EvictAll { keys } => assert_eq!(keys.len(), 2),
            other => panic!("expected EvictAll, got {other:?}"),
        }
    }

    /// A request for an already-loaded, still-valid model short-circuits the whole
    /// ladder and takes a reference.
    #[test]
    fn next_action_hands_back_an_already_loaded_runner_immediately() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/m"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 1;
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::UseLoaded { key, .. } => assert_eq!(key, "/m"),
            other => panic!("expected UseLoaded, got {other:?}"),
        }
        assert_eq!(sched.runner("/m").unwrap().ref_count, 1);
    }

    /// A stale runner is evicted so the reload can happen -- not handed out.
    #[test]
    fn next_action_evicts_a_loaded_runner_that_needs_reloading() {
        let mut sched = Scheduler::default();
        let runner = Arc::new(MockRunner::new(1));
        load_model(
            &mut sched,
            completion_model("/m"),
            runner.clone() as Arc<dyn ModelRunner>,
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );
        runner.ping_ok.store(false, Ordering::SeqCst);

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 1;
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Evict { key } => assert_eq!(key, "/m"),
            other => panic!("expected Evict, got {other:?}"),
        }
    }

    /// At the runner cap, a newcomer must wait for a victim even when VRAM is
    /// plentiful. The cap is a stall guard, not a memory guard.
    #[test]
    fn at_the_runner_cap_a_newcomer_evicts_even_with_vram_to_spare() {
        let mut sched = Scheduler::new(SchedulerConfig {
            max_runners: 1,
            ..Default::default()
        });
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 1000 * GIGABYTE)];
        let predict = |_ctx: u64| 1;
        let mut req = LlmRequest::new(completion_model("/newcomer"), Options::default());

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Evict { key } => assert_eq!(key, "/resident"),
            other => panic!("expected Evict, got {other:?}"),
        }
    }

    /// A CPU-only request must not consult the GPU list at all.
    #[test]
    fn a_cpu_only_request_clears_the_gpu_list_before_planning() {
        let mut sched = Scheduler::default();
        let sys = SystemInfo::default();
        let mut gpus = vec![gpu("0", "CUDA", 10 * GIGABYTE)];
        let predict = |_ctx: u64| 1;

        let mut opts = Options::default();
        opts.runner.num_gpu = 0;
        let mut req = LlmRequest::new(completion_model("/m"), opts);

        match sched.next_action(&mut req, &mut env(&sys, &mut gpus, &predict)) {
            PendingAction::Load(plan) => assert!(plan.gpus.is_empty()),
            other => panic!("expected Load, got {other:?}"),
        }
        assert!(gpus.is_empty());
    }

    // -- load failure recovery --------------------------------------------

    /// **Upstream:** `TestSchedLoadOOMReducesAutomaticContextBeforeRetry`.
    #[test]
    fn an_oom_steps_an_automatic_context_down_before_evicting_anything() {
        let mut sched = Scheduler::default();
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());
        req.num_ctx_auto = true;
        req.opts.runner.num_ctx = 262144;

        let action =
            sched.on_load_failed(&mut req, LoadFailure::OutOfMemory("oom".into()), true);
        assert_eq!(
            action,
            LoadFailureAction::ReduceContextAndRetry {
                old_num_ctx: 262144,
                new_num_ctx: 32768,
            }
        );
        assert_eq!(req.opts.runner.num_ctx, 32768);
        assert!(req.oom_retry_attempted, "at most one retry per request");
    }

    /// **Upstream:** `TestSchedLoadOOMKeepsExplicitContextBeforeRetry` -- a
    /// user-chosen context is never quietly shrunk.
    #[test]
    fn an_oom_never_shrinks_a_context_the_user_asked_for() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );

        let mut req = LlmRequest::new(completion_model("/m"), Options::default());
        req.num_ctx_auto = false;
        req.opts.runner.num_ctx = 262144;

        let action =
            sched.on_load_failed(&mut req, LoadFailure::OutOfMemory("oom".into()), true);
        assert_eq!(action, LoadFailureAction::EvictAndRetry);
        assert_eq!(req.opts.runner.num_ctx, 262144, "untouched");
        assert!(req.oom_retry_attempted);
    }

    /// **Upstream:** `TestSchedLoadCrashNoOtherModelsFailsFast`. Nothing to evict,
    /// nothing to shrink -- give up rather than spin.
    #[test]
    fn an_oom_with_nothing_else_resident_and_an_explicit_context_fails_fast() {
        let mut sched = Scheduler::default();
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());
        req.opts.runner.num_ctx = 4096;

        let action =
            sched.on_load_failed(&mut req, LoadFailure::OutOfMemory("boom".into()), true);
        assert_eq!(
            action,
            LoadFailureAction::Fail(SchedError::LoadFailed("boom".into()))
        );
    }

    /// **Upstream:** `TestSchedLoadCrashTriggersEvictAllAndRetry` -- retry once,
    /// then fail fast on the second crash.
    #[test]
    fn an_oom_retry_is_spent_only_once_then_the_load_fails_fast() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());

        assert_eq!(
            sched.on_load_failed(&mut req, LoadFailure::OutOfMemory("oom".into()), true),
            LoadFailureAction::EvictAndRetry
        );
        assert_eq!(
            sched.on_load_failed(&mut req, LoadFailure::OutOfMemory("oom".into()), true),
            LoadFailureAction::Fail(SchedError::LoadFailed("oom".into())),
            "the second crash must not evict the world all over again"
        );
    }

    /// **Upstream:** `TestSchedLoadNonOOMWithOtherModelsFailsFast`. A corrupt file
    /// is not fixed by freeing memory.
    #[test]
    fn a_non_oom_failure_fails_fast_even_with_other_models_resident() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/resident"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            true,
        );
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());

        assert_eq!(
            sched.on_load_failed(&mut req, LoadFailure::Other("bad gguf".into()), true),
            LoadFailureAction::Fail(SchedError::LoadFailed("bad gguf".into()))
        );
        assert!(!req.oom_retry_attempted, "no retry was spent");
    }

    /// **Upstream:** *"No other models loaded, yet we still don't fit, so report an
    /// error."* No eviction can help.
    #[test]
    fn a_full_offload_failure_on_a_best_effort_load_is_terminal() {
        let mut sched = Scheduler::default();
        let mut req = LlmRequest::new(completion_model("/m"), Options::default());
        assert_eq!(
            sched.on_load_failed(&mut req, LoadFailure::RequiredFullNotMet, false),
            LoadFailureAction::Fail(SchedError::TooLarge)
        );
        assert_eq!(
            sched.on_load_failed(&mut req, LoadFailure::RequiredFullNotMet, true),
            LoadFailureAction::EvictAndRetry
        );
    }

    // -- registration ------------------------------------------------------

    /// The runner's own context length overwrites the request's, because later
    /// reload comparisons must be against what actually happened.
    #[test]
    fn a_loaded_runner_reports_back_the_context_it_actually_settled_on() {
        let mut sched = Scheduler::default();
        let mut runner = MockRunner::new(1);
        runner.context_length = 8192;

        let model = completion_model("/m");
        let mut req = LlmRequest::new(model.clone(), Options::default());
        req.opts.runner.num_ctx = 262144;
        let plan = LoadPlan {
            key: model.scheduler_key(),
            num_parallel: 1,
            session_duration: Expiry::After(Duration::from_secs(60)),
            gpus: vec![],
            launch_opts: Options::default(),
            require_full: false,
            predicted_vram: 0,
            effective_num_ctx: 262144,
            completion: true,
            context_shift: false,
            mmap_disabled_reason: None,
        };
        sched.runner_loaded(&mut req, &plan, runner.arc(), vec![]);

        assert_eq!(req.opts.runner.num_ctx, 8192);
        assert_eq!(
            sched.runner("/m").unwrap().ref_count,
            1,
            "the requesting caller is holding it"
        );
    }

    /// **Upstream:** `iGPUScan` -- discrete if ANY placed device is discrete, and
    /// only then is a VRAM recovery wait worth doing.
    #[test]
    fn only_a_discrete_non_metal_placement_needs_a_vram_recovery_wait() {
        let cuda = DeviceId::new("0", "CUDA");
        let metal = DeviceId::new("0", "Metal");

        // Discrete CUDA -> wait.
        let mut sched = Scheduler::default();
        let key = load_model(
            &mut sched,
            completion_model("/cuda"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::ZERO),
            vec![cuda.clone()],
            true,
        );
        let gone = sched.take_expired(Instant::now());
        assert_eq!(gone.len(), 1);
        assert_eq!(gone[0].key, key);
        assert!(gone[0].needs_vram_recovery_wait);

        // A single Metal device -> unified memory, no wait.
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/metal"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::ZERO),
            vec![metal],
            true,
        );
        let gone = sched.take_expired(Instant::now());
        assert!(!gone[0].needs_vram_recovery_wait);

        // No devices at all -> CPU, no wait.
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/cpu"),
            MockRunner::new(1).arc(),
            Expiry::After(Duration::ZERO),
            vec![],
            true,
        );
        let gone = sched.take_expired(Instant::now());
        assert!(!gone[0].needs_vram_recovery_wait);
    }

    /// **Upstream:** `loadedMmapModelSizeLocked` -- unset mmap counts as on, and
    /// an unknown file size falls back to the runner's total.
    #[test]
    fn the_loaded_mmap_total_counts_unset_as_mmapped_and_falls_back_to_total_size() {
        let mut sched = Scheduler::default();
        let with_size = ModelRef {
            file_size: 7 * GIGABYTE,
            ..completion_model("/sized")
        };
        load_model(
            &mut sched,
            with_size,
            MockRunner::new(1).arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![],
            true,
        );
        // No file size -> fall back to the runner's reported total.
        load_model(
            &mut sched,
            completion_model("/unsized"),
            MockRunner::new(3 * GIGABYTE).arc(),
            Expiry::After(Duration::from_secs(60)),
            vec![],
            true,
        );

        assert_eq!(sched.loaded_mmap_model_size(), 10 * GIGABYTE);
    }

    /// **Upstream:** `Scheduler.loadedModels` -- a snapshot with an estimated
    /// expiry for a runner whose deadline is not armed yet.
    #[test]
    fn loaded_models_reports_an_estimated_expiry_before_the_deadline_is_armed() {
        let mut sched = Scheduler::default();
        load_model(
            &mut sched,
            completion_model("/m"),
            MockRunner::new(4 * GIGABYTE).arc(),
            Expiry::After(Duration::from_secs(300)),
            vec![],
            false, // still held, so expires_at is None
        );

        let t0 = Instant::now();
        let models = sched.loaded_models(t0);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].size_vram, 4 * GIGABYTE);
        assert_eq!(models[0].expires_at, Some(t0 + Duration::from_secs(300)));
    }
}
