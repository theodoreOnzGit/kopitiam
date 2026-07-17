//! The GPU handle, and the one place we probe for a GPU.
//!
//! `GpuContext` owns the live `wgpu::Device` + `wgpu::Queue` for the machine we
//! are actually running on. Getting one is fallible on purpose: plenty of real
//! machines KOPITIAM runs on have no usable GPU — headless CI, a GPU-less
//! tablet, a server with no drivers installed, an Android Termux userland with
//! no Vulkan ICD installed. On all of those, [`GpuContext::new`] returns
//! `Err(GpuUnavailable)` and **never panics**. The whole point of this crate's
//! cascade is that "no GPU" is an ordinary runtime answer, not a crash.
//!
//! ## Headless by design
//!
//! We request an adapter and device with **no surface / no window**
//! (`compatible_surface: None`) — this is compute only: pipelines + buffer
//! readback, nothing drawn to a screen. That surface-free design is exactly what
//! lets the crate run under **Android Termux**, which has no display context, no
//! `Activity`, and no `ANativeWindow`. We never touch JNI or a window handle for
//! compute. (Render pipelines / windowing are deliberately out of scope for the
//! first cut.)
//!
//! Probe ONCE, cache the handle. Enumerating adapters and opening a device is
//! not free (millisecond-ish, and it spins up driver state), so a long-lived
//! program should build one `GpuContext` (or one [`crate::Executor`], which
//! holds one) and reuse it, not rebuild it per operation.

use thiserror::Error;

/// Why we could not get a GPU on this machine.
///
/// This is a plain "no usable GPU" signal, not a rich diagnostic — callers that
/// see it should stop trying the GPU and take the CPU path (that is exactly what
/// [`crate::Executor`] does). The two ways it happens map to the two async wgpu
/// steps that can fail with no hardware behind them.
#[derive(Debug, Error)]
pub enum GpuUnavailable {
    /// wgpu found no adapter matching our request — i.e. no GPU (or no driver
    /// wgpu can talk to) is present at all. On Termux this is the common
    /// "no Vulkan ICD installed / immature Mali driver" case. `String` because
    /// wgpu's own `RequestAdapterError` is not `'static`-friendly to store; we
    /// keep its message for the log, not for matching on.
    #[error("no GPU adapter available: {0}")]
    NoAdapter(String),

    /// An adapter existed but opening a logical device on it failed (driver in
    /// a bad state, limits we asked for unsupported, and so on).
    #[error("GPU adapter found but opening a device failed: {0}")]
    NoDevice(String),
}

/// A live GPU handle: one logical device, its command queue, and a snapshot of
/// which adapter we ended up on (so the maintainer can VERIFY, on-device,
/// whether the GPU actually engaged or we fell back to CPU).
///
/// Cheap to pass around by reference (`&GpuContext`); do not rebuild it per op.
/// Held for the lifetime of an [`crate::Executor`].
#[derive(Debug)]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Captured at construction from `adapter.get_info()`. Kept so
    /// [`describe_backend`](Self::describe_backend) works long after the adapter
    /// handle itself is gone.
    info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Probe for a GPU and open a device on it, or return why we can't.
    ///
    /// Blocking: wgpu's `request_adapter` / `request_device` are async, and this
    /// crate's API is synchronous, so we drive them to completion with
    /// `pollster::block_on`. Call this once and cache the result.
    ///
    /// We ask for [`wgpu::Limits::downlevel_defaults`] rather than the desktop
    /// defaults so the device also opens on mobile/GL-class adapters — this
    /// crate is meant to run on GPU-less *and* modest-GPU devices, and the
    /// demonstrator kernels stay well inside downlevel limits.
    ///
    /// ## The Termux GPU reality (hard-won, read before touching this)
    ///
    /// Whether a GPU adapter is even *reachable* from the Termux userland is
    /// **device + driver dependent**, not a wgpu limitation. Android GPU access
    /// from Termux goes through Vulkan (preferred) or GLES, which is why the
    /// instance below enables exactly `VULKAN | GL`:
    ///
    /// * **Adreno (Qualcomm / Snapdragon):** works via Termux's Mesa
    ///   **Turnip/Freedreno** Vulkan ICD — install `mesa-vulkan-icd-freedreno`
    ///   plus the Vulkan loader (`vulkan-loader`). This is the best-supported
    ///   on-device path and what the README's test recipe targets.
    /// * **Mali (ARM):** the open **Panfrost/PanVK** stack is immature; it may
    ///   present *no usable Vulkan adapter*, in which case we get `NoAdapter`
    ///   and the cascade lands on CPU. That is expected, not a bug.
    /// * **No ICD installed:** no adapter is found at all → `NoAdapter` → CPU.
    ///
    /// In every one of those "no adapter / no driver" cases this returns `Err`
    /// and the [`crate::Executor`] cascade falls to the pure-Rust CPU path — no
    /// crash, correct answer, just slower. Nothing here requires a working GPU.
    pub fn new() -> Result<Self, GpuUnavailable> {
        // Vulkan + GL only: those are the two backends that matter on Android/
        // Termux (Vulkan via the Mesa Turnip ICD, GLES as the fallback), and
        // they cover desktop Linux/Windows too. We do NOT ask for Metal/DX12
        // here — not because they'd hurt, but because VULKAN|GL is the honest
        // portable-compute set for our actual target devices; widen to
        // `Backends::all()` later if a Metal/DX12 host ever needs it.
        // InstanceDescriptor has no `Default` (its `display` field holds a boxed
        // trait object), so every field is spelled out. The sub-descriptors do
        // default, and `display: None` = headless, which is what we want.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        // No surface: headless compute, so `compatible_surface: None`. This is
        // the line that keeps us runnable under Termux (no display context).
        // `force_fallback_adapter: false` = take a real GPU if there is one.
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                // Don't bucket-round the adapter's reported limits; take them
                // as-is (we only need downlevel defaults anyway).
                apply_limit_buckets: false,
            },
        ))
        .map_err(|e| GpuUnavailable::NoAdapter(e.to_string()))?;

        // Snapshot the adapter identity now (name + backend + driver) so we can
        // report it. `get_info` borrows the adapter; `request_device` below also
        // only borrows it, so both are fine before the adapter drops.
        let info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kopitiam-gpu device"),
                required_features: wgpu::Features::empty(),
                // downlevel_defaults so mobile/GL adapters qualify too.
                required_limits: wgpu::Limits::downlevel_defaults(),
                // No experimental wgpu features requested.
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                // No API-trace capture (a debugging feature); off in production.
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|e| GpuUnavailable::NoDevice(e.to_string()))?;

        let ctx = Self {
            device,
            queue,
            info,
        };

        // Log once, at init, so the maintainer can SEE on the tablet which GPU
        // (and backend) engaged. Goes to stderr so it never pollutes a command's
        // stdout data. The CPU-fallback counterpart is logged by Executor::new.
        eprintln!("[kopitiam-gpu] {}", ctx.describe_backend());

        Ok(ctx)
    }

    /// The logical device (buffer/pipeline/shader creation goes through this).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The command queue (submit encoded command buffers here).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The selected adapter's identity: name, backend, device type, driver.
    ///
    /// This is the raw structured form; [`describe_backend`](Self::describe_backend)
    /// formats it into one human line. Use this to verify on-device that the GPU
    /// engaged and via which backend (Vulkan vs GL).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// A one-line human description of what we're running on, e.g.
    /// `GPU: Turnip Adreno (TM) 730 [Vulkan] (driver: Mesa 24.x, Turnip)`.
    ///
    /// This is the string logged at init and the one the README's Termux recipe
    /// tells the maintainer to look for — if they see a GPU/Vulkan line here, the
    /// device GPU engaged; if they see the `Executor`'s CPU-fallback line
    /// instead, it didn't.
    pub fn describe_backend(&self) -> String {
        let i = &self.info;
        // `Backend`/`DeviceType` are Debug, not guaranteed Display across wgpu
        // versions, so format via Debug for stability. driver/driver_info can be
        // empty on some backends; show them only when present.
        let driver = if i.driver.is_empty() && i.driver_info.is_empty() {
            String::new()
        } else {
            format!(" (driver: {} {})", i.driver, i.driver_info)
        };
        format!(
            "GPU: {} [{:?}] {:?}{}",
            i.name, i.backend, i.device_type, driver
        )
    }
}
