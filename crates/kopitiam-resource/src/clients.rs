//! The two clients of the one budgeter — `temp_ai_design.md` §6's economy:
//! "ONE budgeter, two clients". Same [`will_fit`](crate::budget::will_fit), same
//! [`Reason`] enum, both preemptive.
//!
//! ```text
//! will_fit(cost, avail·headroom·core_factor) -> Fits | Degrade | Refuse
//!    ├── client A: "should I run rust-analyzer?"  cost = est_ra_ram
//!    └── client B: "should I load this gguf?"     cost = file_size · materialize_factor
//! ```
//!
//! # On the constants
//!
//! [`RaCoeffs`] and [`GgufCoeffs`] carry **fitted, hard-won, device-specific**
//! numbers. They cannot be derived from first principles — you *measure* them
//! (run the real thing, record peak RSS, fit). The defaults here are the
//! conservative ones ratified for kvim (AID-0037) plus a deliberately generous
//! gguf materialisation factor. The whole point of [`crate::calibration`] is to
//! let the probe *learn* better numbers from real runs over time. Keep the
//! defaults conservative and let calibration refine them — do not hand-tune them
//! optimistic.

use crate::budget::{budget_mb, will_fit, Verdict, DEFAULT_CORE_REF, DEFAULT_HEADROOM};
use crate::fetched::Reason;
use crate::probe::Capacity;
use crate::project::ProjectWeight;

/// Fitted coefficients for **client A** — estimating rust-analyzer's peak RSS.
///
/// The model is `est_mb = base_mb + per_dep_mb·dep_crates + src_factor·src_mb`.
/// RA's memory scales mostly with the number of crates it indexes (the whole dep
/// graph), which is why `per_dep_mb` is the dominant term and `src_factor` is
/// secondary.
///
/// Defaults match kvim's shipped guard (AID-0037), measured against real Rust
/// projects on the maintainer's device: `base 150 MB`, `4 MB/dep`, `0.5 MB per
/// MB of source`. **Device-specific** — a different tablet will fit different
/// numbers. Tune via [`crate::calibration`], keep them conservative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaCoeffs {
    /// Base rust-analyzer overhead in MB, independent of project size — the
    /// server process, empty VFS, its own machinery.
    pub base_mb: f64,
    /// Estimated MB of RA RSS per dependency crate (the dominant term).
    pub per_dep_mb: f64,
    /// MB of RA RSS per MB of first-party `.rs` source (secondary term).
    pub src_factor: f64,
}

impl Default for RaCoeffs {
    fn default() -> Self {
        Self { base_mb: 150.0, per_dep_mb: 4.0, src_factor: 0.5 }
    }
}

/// **Client A cost function:** estimate rust-analyzer's peak RSS in **MB** from a
/// project's [`ProjectWeight`], using [`RaCoeffs`].
///
/// `est_mb = base_mb + per_dep_mb·dep_crates + src_factor·src_mb`. Pure
/// arithmetic; the input weight is the cheap stat-only estimate from
/// [`crate::project`].
pub fn est_ra_ram(weight: ProjectWeight, k: RaCoeffs) -> f64 {
    k.base_mb + k.per_dep_mb * weight.dep_crates as f64 + k.src_factor * weight.src_mb()
}

/// Fitted coefficients for **client B** — estimating the resident cost of loading
/// a gguf model file.
///
/// The model is `est_mb = file_mb · materialize_factor`. The factor is `> 1.0`
/// because loading a model is not free of the file size: even mmapped, touched
/// pages become resident, and dequantising / KV-cache / scratch buffers add on
/// top. A generous default keeps us on the safe side of the **uncatchable**
/// `SIGABRT` an oversized allocation triggers (the LocalAdapter review finding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GgufCoeffs {
    /// Resident MB per MB of gguf file on disk. `> 1.0`. Default `1.2` — a
    /// deliberately conservative 20% over the raw file size to cover resident
    /// pages plus runtime scratch. Measure and refine per device/model via
    /// [`crate::calibration`].
    pub materialize_factor: f64,
}

impl Default for GgufCoeffs {
    fn default() -> Self {
        Self { materialize_factor: 1.2 }
    }
}

/// **Client B cost function:** estimate the resident cost in **MB** of loading a
/// gguf file of `file_bytes`, using [`GgufCoeffs`].
///
/// `est_mb = (file_bytes / 1024·1024) · materialize_factor`. Takes raw bytes (a
/// `std::fs::metadata(path).len()` — stat only, never open the file) and converts
/// to base-2 MB internally.
pub fn est_gguf_ram(file_bytes: u64, k: GgufCoeffs) -> f64 {
    let file_mb = file_bytes as f64 / (1024.0 * 1024.0);
    file_mb * k.materialize_factor
}

/// Everything the budget config needs beyond the per-client coefficients: the
/// headroom fraction and the core reference. Defaults are the conservative
/// [`DEFAULT_HEADROOM`] / [`DEFAULT_CORE_REF`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetInputs {
    /// Fraction of *available* RAM the budget may occupy. See [`DEFAULT_HEADROOM`].
    pub headroom: f64,
    /// "Plenty of cores" reference for the CPU factor. See [`DEFAULT_CORE_REF`].
    pub core_ref: f64,
}

impl Default for BudgetInputs {
    fn default() -> Self {
        Self { headroom: DEFAULT_HEADROOM, core_ref: DEFAULT_CORE_REF }
    }
}

/// **Client A gate:** "should I run rust-analyzer?" Combines the cheap probe +
/// weight into a [`Verdict`], attributing [`Reason::ProjectTooLarge`] on a
/// non-fit (the project-flavoured memory reason the model relays to the user).
///
/// A bogus zero-core reading is rejected up front as fail-open
/// ([`Verdict::Degrade`] / [`Reason::NotApplicable`]) rather than trusted — a
/// 0-core probe is nonsense, and dividing the budget by it would be worse than
/// standing down. Otherwise the budget is `avail · headroom · core_factor` and
/// the estimate is [`est_ra_ram`].
pub fn should_run_rust_analyzer(
    cap: Capacity,
    weight: ProjectWeight,
    coeffs: RaCoeffs,
    inputs: BudgetInputs,
) -> Verdict {
    if cap.logical_cores == 0 {
        return Verdict::Degrade(Reason::NotApplicable);
    }
    let cost = est_ra_ram(weight, coeffs);
    let budget = budget_mb(cap, inputs.headroom, inputs.core_ref);
    will_fit(cost, budget, Reason::ProjectTooLarge)
}

/// **Client B gate:** "should I load this gguf?" Combines the probe + the file
/// size into a [`Verdict`], attributing [`Reason::MemoryBudgetExceeded`] on a
/// non-fit (the raw-bytes memory reason).
///
/// Same machinery as client A — that reuse is the design's whole "one budgeter"
/// economy. Same fail-open on a bogus zero-core reading.
pub fn should_load_gguf(
    cap: Capacity,
    file_bytes: u64,
    coeffs: GgufCoeffs,
    inputs: BudgetInputs,
) -> Verdict {
    if cap.logical_cores == 0 {
        return Verdict::Degrade(Reason::NotApplicable);
    }
    let cost = est_gguf_ram(file_bytes, coeffs);
    let budget = budget_mb(cap, inputs.headroom, inputs.core_ref);
    will_fit(cost, budget, Reason::MemoryBudgetExceeded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big_device(avail_mb: u64) -> Capacity {
        // 8 cores so the core factor is 1.0 and RAM is the only axis.
        Capacity { avail_mb, total_mb: avail_mb * 2, logical_cores: 8, cpu_usage: 10.0 }
    }

    #[test]
    fn est_ra_ram_follows_the_fitted_model() {
        // 100 deps, no source -> 150 + 4*100 + 0 = 550.
        let w = ProjectWeight { dep_crates: 100, src_bytes: 0 };
        assert_eq!(est_ra_ram(w, RaCoeffs::default()), 550.0);
    }

    #[test]
    fn est_gguf_ram_applies_the_materialize_factor() {
        // 1000 MB file * 1.2 = 1200 MB resident estimate.
        let bytes = 1000 * 1024 * 1024;
        assert_eq!(est_gguf_ram(bytes, GgufCoeffs::default()), 1200.0);
    }

    #[test]
    fn small_project_on_a_big_device_fits() {
        let w = ProjectWeight { dep_crates: 30, src_bytes: 1024 * 1024 };
        let v = should_run_rust_analyzer(
            big_device(16_000),
            w,
            RaCoeffs::default(),
            BudgetInputs::default(),
        );
        assert_eq!(v, Verdict::Fits, "small project on a big device must run full");
    }

    #[test]
    fn heavy_project_on_a_weak_tablet_refuses() {
        // 340 deps -> est ~ 150 + 1360 + 0.5*20 = ~1520 MB.
        // 3000 MB free but 2 cores quarters it: 3000*0.6*0.25 = 450 MB. Way over
        // the +15% band -> Refuse (SKIP), with the project reason.
        let w = ProjectWeight { dep_crates: 340, src_bytes: 20 * 1024 * 1024 };
        let tablet = Capacity { avail_mb: 3000, total_mb: 4000, logical_cores: 2, cpu_usage: 40.0 };
        let v = should_run_rust_analyzer(tablet, w, RaCoeffs::default(), BudgetInputs::default());
        assert_eq!(v, Verdict::Refuse(Reason::ProjectTooLarge));
    }

    #[test]
    fn cpu_factor_flips_a_decision_ram_alone_would_pass() {
        // 100 deps -> est 550 MB. 2000 MB free.
        // 8 cores: budget 2000*0.6*1.0 = 1200 -> Fits.
        // 2 cores: budget 2000*0.6*0.25 = 300 -> 550 > 345 upper -> Refuse.
        let w = ProjectWeight { dep_crates: 100, src_bytes: 0 };
        let eight = Capacity { avail_mb: 2000, total_mb: 4000, logical_cores: 8, cpu_usage: 0.0 };
        let two = Capacity { avail_mb: 2000, total_mb: 4000, logical_cores: 2, cpu_usage: 0.0 };
        assert_eq!(
            should_run_rust_analyzer(eight, w, RaCoeffs::default(), BudgetInputs::default()),
            Verdict::Fits
        );
        assert_eq!(
            should_run_rust_analyzer(two, w, RaCoeffs::default(), BudgetInputs::default()),
            Verdict::Refuse(Reason::ProjectTooLarge),
            "same RAM, same project: weak CPU alone must gate"
        );
    }

    #[test]
    fn gguf_that_fits_and_one_that_does_not() {
        // 1 GB file -> 1200 MB resident estimate.
        // Big device 8000 MB free, 8 cores -> budget 4800 -> Fits.
        let one_gb = 1000 * 1024 * 1024;
        assert_eq!(
            should_load_gguf(big_device(8000), one_gb, GgufCoeffs::default(), BudgetInputs::default()),
            Verdict::Fits
        );
        // Same file on a 1500 MB-free tablet -> budget 900 -> 1200 > 1035 -> Refuse.
        let tablet = big_device(1500);
        assert_eq!(
            should_load_gguf(tablet, one_gb, GgufCoeffs::default(), BudgetInputs::default()),
            Verdict::Refuse(Reason::MemoryBudgetExceeded)
        );
    }

    #[test]
    fn zero_core_reading_stands_down_not_trusted() {
        let w = ProjectWeight { dep_crates: 100, src_bytes: 0 };
        let bogus = Capacity { avail_mb: 1000, total_mb: 2000, logical_cores: 0, cpu_usage: 0.0 };
        assert_eq!(
            should_run_rust_analyzer(bogus, w, RaCoeffs::default(), BudgetInputs::default()),
            Verdict::Degrade(Reason::NotApplicable)
        );
        assert_eq!(
            should_load_gguf(bogus, 999, GgufCoeffs::default(), BudgetInputs::default()),
            Verdict::Degrade(Reason::NotApplicable)
        );
    }
}
