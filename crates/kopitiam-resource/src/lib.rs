//! `kopitiam-resource` — KOPITIAM's **preemptive** resource budgeter.
//!
//! # Why this crate exists (the Android-critical bit)
//!
//! On a small Android tablet, three of KOPITIAM's heavy jobs can kill the whole
//! process, and you **cannot** `?`-your-way out of any of them, because the
//! kernel already shot you dead before your code runs again:
//!
//! - **rust-analyzer** indexing a project too big for the RAM → the OS
//!   low-memory killer sends `SIGKILL`. No `Result`, no `catch`, no unwind.
//! - **an oversized gguf** model file → the allocator gives up with `SIGABRT`.
//! - **an mmapped file truncated mid-read** → `SIGBUS`.
//!
//! Ordinary `Result` enums only ever see *ordinary* failures. These three are
//! not ordinary — by the time you would inspect an error, the process is gone.
//!
//! So the only real defence is **preemptive**: estimate the cost BEFORE you
//! launch the heavy thing, and refuse-or-degrade up front. The crash never
//! happens, so nothing has to survive it. That is the whole design:
//!
//! ```text
//! cheap device probe (free RAM, cores)  ── never runs the expensive thing
//!      +  cheap project-weight estimate (stat-only, no file reads)
//!      ↓
//!  budget decision:  Fits │ Degrade(Reason) │ Refuse(Reason)
//!      ↓
//!  the Fetched enum reports the DECISION, not a caught crash
//! ```
//!
//! # One budgeter, two clients
//!
//! kvim already ships exactly this reasoning for rust-analyzer only
//! (`kopitiam-neovim`'s `lsp/resource_guard.rs`, ratified as **AID-0037**).
//! This crate **generalises** that one guard into a single shared budgeter with
//! two clients, so we do not grow a second copy of the arithmetic for the gguf
//! loader:
//!
//! - **Client A — rust-analyzer:** cost = [`clients::est_ra_ram`] (`base +
//!   per_dep·crates + src_factor·src_mb`). See [`clients::should_run_rust_analyzer`].
//! - **Client B — gguf load:** cost = [`clients::est_gguf_ram`] (`file_size ×
//!   materialize_factor`). See [`clients::should_load_gguf`].
//!
//! Both go through the same [`budget::will_fit`], the same [`Reason`] enum, both
//! preemptive.
//!
//! # The load-bearing rule: when marginal, DEGRADE
//!
//! The failure is **asymmetric and uncatchable**. A false `Refuse` (SKIP) costs
//! you some IDE niceties — annoying, recoverable. A false `Fits` (FULL) costs
//! you a **crashed tablet** — SIGKILL, no recovery, work lost. So the budgeter
//! is deliberately biased: near the budget, it never says `Fits`, it says
//! `Degrade`. The goal is not accuracy. The goal is **never crossing the
//! cliff**. This bias lives in [`budget::BudgetPolicy`] and is the single most
//! important thing in the crate — see its docs.
//!
//! # Layout
//!
//! | Module | What it owns |
//! |---|---|
//! | [`fetched`] | [`Fetched<T>`] + [`Reason`] — the resource-aware result type. |
//! | [`budget`] | [`will_fit`](budget::will_fit), [`Verdict`](budget::Verdict), [`BudgetPolicy`](budget::BudgetPolicy), the budget arithmetic + the conservative bias. |
//! | [`probe`] | [`DeviceProbe`](probe::DeviceProbe) trait (injectable), [`Capacity`](probe::Capacity) snapshot, [`SysinfoProbe`](probe::SysinfoProbe), [`FixedProbe`](probe::FixedProbe) for tests. |
//! | [`project`] | [`ProjectWeight`](project::ProjectWeight) + the cheap, stat-only estimators. |
//! | [`clients`] | Client A / Client B cost helpers + the two `should_*` convenience gates. |
//! | [`calibration`] | The self-improving seam: record actual-peak-RSS vs estimate (storage impl is a follow-up). |
//!
//! # Provenance
//!
//! Generalised from kvim's `lsp/resource_guard.rs` (AID-0037). The design it
//! implements is `temp_ai_design.md` §5 (preemptive, not reactive), §6 (the
//! project-size / capability probe + conservative-bias rule), §7 (crate
//! mapping). The fitted constants (`k1`, `k2`, `materialize_factor`) are
//! **hard-won and device-specific** — see [`clients::RaCoeffs`] and
//! [`clients::GgufCoeffs`] and the calibration note in [`calibration`].

#![forbid(unsafe_code)]

pub mod budget;
pub mod calibration;
pub mod clients;
pub mod fetched;
pub mod probe;
pub mod project;

// A flat re-export of the everyday names, so a caller writes
// `kopitiam_resource::{Fetched, Reason, Verdict, will_fit}` without hunting
// through modules. The modules stay the source of truth for the docs.
pub use budget::{budget_mb, core_factor, will_fit, BudgetPolicy, Verdict};
pub use calibration::{CalibrationSample, CalibrationSink, Client, NullSink};
pub use fetched::{Fetched, Reason};
pub use probe::{Capacity, DeviceProbe, FixedProbe, SysinfoProbe};
pub use project::{estimate_project_weight, ProjectWeight};
