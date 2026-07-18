//! The budgeter proper: [`will_fit`], [`Verdict`], [`BudgetPolicy`], and the
//! budget arithmetic ([`budget_mb`], [`core_factor`]).
//!
//! Everything here is **pure arithmetic over `f64` MB** — no I/O, no globals, no
//! probe. That separation is the whole point: [`will_fit`] can be hammered by
//! unit tests against synthetic numbers, and the *decision* is provably a
//! function of its inputs, not of wall-clock or machine state (the
//! determinism-given-budget requirement, `temp_ai_design.md` §4/§6).
//!
//! # Units contract (read this once, then it's obvious)
//!
//! **Every quantity in this module is in MEGABYTES (`MB`, base-2: 1 MB = 1024·1024
//! bytes) as an `f64`.** `cost_estimate_mb`, `budget_mb`, `avail_mb` — all MB.
//! The one exception is [`core_factor`], which is dimensionless. If you have
//! bytes, divide by `1024·1024` before you come in here.

use crate::fetched::Reason;
use crate::probe::Capacity;

/// The three-way budget verdict — the same FULL / PARTIAL / SKIP shape as
/// [`crate::Fetched`], but computed *before* anything launches.
///
/// - [`Verdict::Fits`] — FULL. Comfortably under budget; run the full path.
/// - [`Verdict::Degrade`] — PARTIAL. Near the budget; run a reduced-footprint
///   path and tell the user why (the carried [`Reason`]).
/// - [`Verdict::Refuse`] — SKIP. Over budget; do **not** launch the heavy path,
///   drop to the degraded provider entirely.
///
/// The mapping to [`crate::Fetched`] is [`Verdict::into_fetched`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Run the full thing.
    Fits,
    /// Run a reduced path; here's why it was cut.
    Degrade(Reason),
    /// Don't launch the heavy path at all; here's why.
    Refuse(Reason),
}

impl Verdict {
    /// `true` only for [`Verdict::Fits`].
    pub fn is_fits(&self) -> bool {
        matches!(self, Verdict::Fits)
    }

    /// The carried [`Reason`], if the verdict is not `Fits`.
    pub fn reason(&self) -> Option<Reason> {
        match self {
            Verdict::Fits => None,
            Verdict::Degrade(r) | Verdict::Refuse(r) => Some(*r),
        }
    }

    /// Turn a verdict into a [`crate::Fetched`], given the two payloads you have
    /// ready: the `full` result to use when it [`Fits`](Verdict::Fits), and the
    /// `degraded` result to use when it [`Degrade`](Verdict::Degrade)s. On
    /// [`Refuse`](Verdict::Refuse) there is no usable payload, so you get
    /// [`Fetched::Unavailable`](crate::Fetched::Unavailable).
    ///
    /// This is the bridge from "should I?" (a `Verdict`) to "here's what you
    /// got" (a `Fetched`). The caller supplies the payloads because only the
    /// caller can actually produce them — the budgeter never runs the work.
    pub fn into_fetched<T>(self, full: T, degraded: T) -> crate::Fetched<T> {
        match self {
            Verdict::Fits => crate::Fetched::Ready(full),
            Verdict::Degrade(r) => crate::Fetched::Partial(degraded, r),
            Verdict::Refuse(r) => crate::Fetched::Unavailable(r),
        }
    }
}

/// Default fraction of the budget the [marginal band](BudgetPolicy::marginal_band)
/// reaches on each side of the budget. `0.15` = "within 15% of the budget counts
/// as marginal → degrade, never full". Conservative; tune per device.
pub const DEFAULT_MARGINAL_BAND: f64 = 0.15;

/// How the budgeter draws the two boundaries between FULL / PARTIAL / SKIP.
///
/// # The load-bearing decision (do not remove the band)
///
/// `temp_ai_design.md` §6's asymmetric-risk rule: **when marginal, DEGRADE.** A
/// false `Refuse` costs IDE niceties; a false `Fits` crashes the tablet with an
/// **uncatchable** SIGKILL. So the "≈ budget" zone must resolve to
/// [`Verdict::Degrade`], never [`Verdict::Fits`]. That zone is a band of width
/// `marginal_band · budget` sitting on **both** sides of the budget line:
///
/// ```text
///   cost ≤ budget·(1 − band)        → Fits      (comfortably under → FULL)
///   budget·(1−band) < cost ≤ budget·(1+band)
///                                   → Degrade   (≈ budget → PARTIAL)
///   cost > budget·(1 + band)        → Refuse    (clearly over → SKIP)
/// ```
///
/// The lower edge (`1 − band`) is what makes it conservative: a cost that is
/// *just under* budget is still marginal, so it degrades instead of chancing a
/// full launch. The upper edge (`1 + band`) lets a cost that is *just over*
/// budget still attempt the reduced path (which uses less RAM and usually fits)
/// rather than skipping straight to syntax-only.
///
/// Set `marginal_band = 0.0` to get a hard cliff exactly at the budget (`Fits`
/// below, `Refuse` above, no `Degrade` — not recommended on a swapless device).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetPolicy {
    /// Half-width of the marginal band, as a fraction of the budget. Must be in
    /// `0.0..=1.0`; values outside are clamped. Default [`DEFAULT_MARGINAL_BAND`].
    pub marginal_band: f64,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self { marginal_band: DEFAULT_MARGINAL_BAND }
    }
}

impl BudgetPolicy {
    /// The core decision: does a `cost_estimate_mb` fit inside a `budget_mb`, and
    /// if not, degrade or refuse? See the [type docs](BudgetPolicy) for the exact
    /// boundaries.
    ///
    /// `reason` is the domain reason to attribute when it does **not** cleanly
    /// fit — so client A (rust-analyzer) passes [`Reason::ProjectTooLarge`] and
    /// client B (gguf) passes [`Reason::MemoryBudgetExceeded`], while the
    /// arithmetic stays identical. This is the one place the generic budgeter
    /// borrows the caller's vocabulary.
    ///
    /// # Fail-open on a bogus budget
    ///
    /// A non-finite or non-positive `budget_mb` means "no honest reading of the
    /// device" (e.g. the probe reported zero free RAM). We **fail open** —
    /// [`Verdict::Degrade`] with [`Reason::NotApplicable`] rather than a spurious
    /// `Refuse` — because the budgeter exists to stop a *tablet* OOM, not to
    /// block a machine it simply could not measure. (A caller that wants a true
    /// "unguarded, run full" on a missing probe should check the probe *before*
    /// calling and skip the gate; see [`crate::clients`].) A non-finite `cost`
    /// is treated as "enormous" → `Refuse`, since we could not trust it to be
    /// small.
    pub fn will_fit(&self, cost_estimate_mb: f64, budget_mb: f64, reason: Reason) -> Verdict {
        if !budget_mb.is_finite() || budget_mb <= 0.0 {
            // Can't measure the device honestly -> stand down conservatively but
            // do not hard-refuse a machine we never actually read.
            return Verdict::Degrade(Reason::NotApplicable);
        }
        if !cost_estimate_mb.is_finite() {
            // A NaN/inf cost is untrustworthy; treat as "too big to risk".
            return Verdict::Refuse(reason);
        }
        let band = self.marginal_band.clamp(0.0, 1.0);
        let lower = budget_mb * (1.0 - band);
        let upper = budget_mb * (1.0 + band);

        if cost_estimate_mb <= lower {
            Verdict::Fits
        } else if cost_estimate_mb <= upper {
            Verdict::Degrade(reason)
        } else {
            Verdict::Refuse(reason)
        }
    }
}

/// [`BudgetPolicy::will_fit`] with the default (conservative) policy. The
/// free-function form the design writes as
/// `will_fit(cost_estimate, budget) -> Fits | Degrade | Refuse`.
///
/// `reason` is the domain reason to report on a non-fit (see
/// [`BudgetPolicy::will_fit`]).
pub fn will_fit(cost_estimate_mb: f64, budget_mb: f64, reason: Reason) -> Verdict {
    BudgetPolicy::default().will_fit(cost_estimate_mb, budget_mb, reason)
}

/// Default fraction of *available* RAM the budget may occupy — the headroom
/// factor. `0.6` leaves 40% of free RAM for the editor, the OS, and the OOM
/// killer's own trigger margin.
///
/// Note kvim's shipped LSP guard (AID-0037) uses `0.5`; this crate's default is
/// slightly looser at `0.6` per `temp_ai_design.md` §6. Both are conservative
/// and both are configurable — on a swapless device, headroom is safety, not
/// politeness, so tune it *down* if you see kills, never blindly up.
pub const DEFAULT_HEADROOM: f64 = 0.6;

/// Default core-count reference: treat "this many logical cores or more" as
/// "plenty of indexing parallelism", so the [`core_factor`] saturates at `1.0`.
/// `8.0` matches AID-0037.
pub const DEFAULT_CORE_REF: f64 = 8.0;

/// The dimensionless CPU scaling applied to the budget: `min(1.0, cores /
/// core_ref)`.
///
/// CPU is a real input to the gate, not just message flavour — rust-analyzer is
/// CPU-heavy *while indexing*, so a 2-core tablet janks on a mid-size project
/// even when the RAM alone would have fit. With `core_ref = 8`, an 8+-core
/// machine keeps the full budget (factor `1.0`) while a 2-core tablet's budget is
/// **quartered** (factor `0.25`), holding the heavy path off far sooner there.
///
/// Saturates at `1.0`: a 32-core box does **not** get an inflated budget — extra
/// cores never buy you extra RAM, so the factor is capped and only ever *reduces*
/// the budget on a weak CPU.
///
/// A non-positive `cores` or `core_ref` is treated as "cannot scale" → returns
/// `1.0` (leave the RAM budget untouched rather than divide by nonsense). A
/// caller that wants to *reject* a bogus zero-core reading should do so before
/// building the budget; see [`crate::clients`].
pub fn core_factor(logical_cores: usize, core_ref: f64) -> f64 {
    if core_ref <= 0.0 || logical_cores == 0 {
        return 1.0;
    }
    (logical_cores as f64 / core_ref).min(1.0)
}

/// Build the RAM budget in MB from a device [`Capacity`] snapshot:
/// `avail_mb · headroom · core_factor(cores, core_ref)`.
///
/// This is the number [`will_fit`] compares a cost estimate against. It is
/// deliberately built from **available** RAM (what the OOM killer actually cares
/// about), never total RAM — see [`Capacity::avail_mb`]. Re-read the capacity
/// right before each heavy launch (free RAM is volatile; a user opening another
/// app changes it), then rebuild the budget from the fresh snapshot.
pub fn budget_mb(cap: Capacity, headroom: f64, core_ref: f64) -> f64 {
    cap.avail_mb as f64 * headroom * core_factor(cap.logical_cores, core_ref)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comfortably_under_fits() {
        // cost 100, budget 1000, band 0.15 -> lower edge 850. 100 <= 850 -> Fits.
        assert_eq!(
            will_fit(100.0, 1000.0, Reason::ProjectTooLarge),
            Verdict::Fits
        );
    }

    #[test]
    fn marginal_below_budget_degrades_not_fits() {
        // THE conservative-bias flip: 900 is UNDER the 1000 budget, but inside
        // the marginal band (>850), so it must DEGRADE, never Fits. A false Fits
        // here is the uncatchable-crash case.
        assert_eq!(
            will_fit(900.0, 1000.0, Reason::ProjectTooLarge),
            Verdict::Degrade(Reason::ProjectTooLarge)
        );
    }

    #[test]
    fn marginal_above_budget_degrades_not_refuses() {
        // 1050 is OVER budget but within the +15% band (<=1150), so the reduced
        // path is still worth attempting -> Degrade, not Refuse.
        assert_eq!(
            will_fit(1050.0, 1000.0, Reason::MemoryBudgetExceeded),
            Verdict::Degrade(Reason::MemoryBudgetExceeded)
        );
    }

    #[test]
    fn clearly_over_refuses() {
        // 1200 > 1150 upper edge -> Refuse (SKIP to degraded provider).
        assert_eq!(
            will_fit(1200.0, 1000.0, Reason::ProjectTooLarge),
            Verdict::Refuse(Reason::ProjectTooLarge)
        );
    }

    #[test]
    fn the_carried_reason_is_the_callers() {
        // will_fit stays generic arithmetic; client A vs client B choose the word.
        assert_eq!(
            will_fit(2000.0, 100.0, Reason::ProjectTooLarge).reason(),
            Some(Reason::ProjectTooLarge)
        );
        assert_eq!(
            will_fit(2000.0, 100.0, Reason::MemoryBudgetExceeded).reason(),
            Some(Reason::MemoryBudgetExceeded)
        );
    }

    #[test]
    fn zero_band_is_a_hard_cliff_at_budget() {
        let strict = BudgetPolicy { marginal_band: 0.0 };
        assert_eq!(strict.will_fit(1000.0, 1000.0, Reason::ProjectTooLarge), Verdict::Fits);
        assert_eq!(
            strict.will_fit(1000.1, 1000.0, Reason::ProjectTooLarge),
            Verdict::Refuse(Reason::ProjectTooLarge)
        );
    }

    #[test]
    fn bogus_budget_fails_open_to_degrade_not_refuse() {
        // Zero / negative / NaN budget = "couldn't measure the device". Must not
        // hard-refuse a machine we never read.
        for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let v = will_fit(500.0, bad, Reason::ProjectTooLarge);
            assert_eq!(v, Verdict::Degrade(Reason::NotApplicable), "budget {bad}");
        }
    }

    #[test]
    fn non_finite_cost_refuses() {
        assert_eq!(
            will_fit(f64::NAN, 1000.0, Reason::MemoryBudgetExceeded),
            Verdict::Refuse(Reason::MemoryBudgetExceeded)
        );
    }

    #[test]
    fn core_factor_penalises_weak_cpu_and_saturates() {
        assert_eq!(core_factor(2, 8.0), 0.25, "2-core tablet: budget quartered");
        assert_eq!(core_factor(8, 8.0), 1.0, "8 cores: full budget");
        assert_eq!(core_factor(32, 8.0), 1.0, "32 cores never inflate the budget");
        assert_eq!(core_factor(0, 8.0), 1.0, "bogus 0 cores -> don't scale, leave budget");
    }

    #[test]
    fn budget_mb_uses_available_ram_headroom_and_cores() {
        // 3000 MB free, 0.6 headroom, 2 cores of 8 -> factor 0.25.
        // budget = 3000 * 0.6 * 0.25 = 450.
        let cap = Capacity { avail_mb: 3000, total_mb: 4000, logical_cores: 2, cpu_usage: 0.0 };
        assert_eq!(budget_mb(cap, DEFAULT_HEADROOM, DEFAULT_CORE_REF), 450.0);
    }

    #[test]
    fn into_fetched_bridges_verdict_to_result() {
        assert_eq!(Verdict::Fits.into_fetched("full", "deg"), crate::Fetched::Ready("full"));
        assert_eq!(
            Verdict::Degrade(Reason::ProjectTooLarge).into_fetched("full", "deg"),
            crate::Fetched::Partial("deg", Reason::ProjectTooLarge)
        );
        assert_eq!(
            Verdict::Refuse(Reason::ProjectTooLarge).into_fetched("full", "deg"),
            crate::Fetched::Unavailable(Reason::ProjectTooLarge)
        );
    }
}
