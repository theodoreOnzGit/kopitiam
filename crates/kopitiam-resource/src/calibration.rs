//! The self-improving seam — `temp_ai_design.md` §6's "make the probe learn".
//!
//! The fitted constants ([`crate::clients::RaCoeffs`], [`crate::clients::GgufCoeffs`])
//! cannot be derived from first principles; you *measure* them. The KOPITIAM move
//! is that the probe keeps measuring: after each **real** launch, record the
//! *actual* peak RSS against what we *estimated*, and over time refine the
//! constants so every session predicts the next one better. That is "every AI
//! interaction leaves behind permanent knowledge" applied to the budgeter itself.
//!
//! # What is (and is NOT) in this crate
//!
//! This module is **only the seam** — the [`CalibrationSample`] shape and the
//! [`CalibrationSink`] trait that records one. **Storage is a deliberate
//! follow-up** and does not live here: the natural home is `kopitiam-index` (the
//! redb-backed persistent project state), so the samples survive across sessions
//! for the fitting loop to read. See open question 4 in `temp_ai_design.md` §8
//! ("Calibration data provenance"). A follow-up bead tracks building that store
//! and the fitting step; until then, [`NullSink`] is the no-op default so nothing
//! downstream has to care that the store is not built yet.
//!
//! The **fitting** (samples → refined coefficients) is likewise out of scope
//! here. This crate produces the estimate and accepts the ground-truth
//! observation; turning a pile of observations back into better `k1,k2,factor` is
//! the follow-up's job.

use crate::fetched::Reason;

/// Which client an observation belongs to — so a store keeps rust-analyzer and
/// gguf samples apart when fitting their (separate) coefficient sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Client {
    /// Client A: a rust-analyzer launch (coefficients in [`crate::clients::RaCoeffs`]).
    RustAnalyzer,
    /// Client B: a gguf model load (coefficients in [`crate::clients::GgufCoeffs`]).
    Gguf,
}

/// One observed data point: what we predicted vs what actually happened on a real
/// run. This is the ground truth the fitting loop refines constants against.
///
/// Record it **after** the launch you gated has run far enough to have a real
/// peak — the whole value is `actual_peak_rss_mb` being a measured number, not a
/// guess. All memory fields are **MB** (base-2), matching the rest of the crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationSample {
    /// Which client this observation calibrates.
    pub client: Client,
    /// What we estimated the peak RSS would be, in MB (from `est_ra_ram` /
    /// `est_gguf_ram`). Keeping the estimate alongside the actual is what lets the
    /// fitter see the error, not just the truth.
    pub estimated_mb: f64,
    /// The **actual** peak resident set size the launch reached, in MB — measured
    /// (e.g. via `sysinfo`'s per-process RSS), not inferred.
    pub actual_peak_rss_mb: f64,
    /// For a rust-analyzer sample: the dep-crate count that fed the estimate, so
    /// the fitter can re-attribute error across the `per_dep_mb` term. `None` for
    /// a gguf sample.
    pub dep_crates: Option<usize>,
    /// For a rust-analyzer sample: the source MB that fed the estimate (the
    /// `src_factor` term). `None` for a gguf sample.
    pub src_mb: Option<f64>,
    /// For a gguf sample: the model file size in MB that fed the estimate (the
    /// `materialize_factor` term). `None` for a rust-analyzer sample.
    pub file_mb: Option<f64>,
    /// The verdict-driving [`Reason`] context, if the launch was degraded/refused
    /// — useful for the fitter to know whether a sample came from a full run or a
    /// reduced one. `None` when the full path ran.
    pub reason: Option<Reason>,
}

impl CalibrationSample {
    /// Signed prediction error in MB: `actual − estimated`. Positive means we
    /// **under**-estimated (the dangerous direction — we thought it would fit when
    /// it wanted more), which is exactly what the conservative bias and the
    /// fitting loop most need to catch.
    pub fn error_mb(&self) -> f64 {
        self.actual_peak_rss_mb - self.estimated_mb
    }

    /// `true` if we under-estimated the peak — i.e. reality wanted more RAM than
    /// we predicted. On a swapless device this is the direction that crashes, so
    /// a store should weight these samples heavily when refitting.
    pub fn was_underestimate(&self) -> bool {
        self.error_mb() > 0.0
    }
}

/// The seam a persistent store implements to bank a [`CalibrationSample`].
///
/// Kept trivially small on purpose: the budgeter here only needs to *hand off* an
/// observation; where it goes and how it is later fitted is the follow-up's
/// concern (likely `kopitiam-index`). Downstream wires a real sink; tests and
/// callers-that-don't-care use [`NullSink`].
pub trait CalibrationSink {
    /// Bank one observation. Implementations must not panic on a bad sample —
    /// calibration is best-effort telemetry, never on the critical path of a
    /// launch. A store that cannot write should swallow the error (and log it),
    /// not bubble it up into the gate.
    fn record(&self, sample: CalibrationSample);
}

/// The no-op sink: drops every sample. The default until the real store lands, so
/// nothing downstream has to special-case "calibration not built yet".
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSink;

impl CalibrationSink for NullSink {
    fn record(&self, _sample: CalibrationSample) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_and_underestimate_flag() {
        let under = CalibrationSample {
            client: Client::RustAnalyzer,
            estimated_mb: 500.0,
            actual_peak_rss_mb: 700.0,
            dep_crates: Some(100),
            src_mb: Some(0.0),
            file_mb: None,
            reason: None,
        };
        assert_eq!(under.error_mb(), 200.0);
        assert!(under.was_underestimate(), "700 actual vs 500 est is an under-estimate");

        let over = CalibrationSample { actual_peak_rss_mb: 400.0, ..under };
        assert_eq!(over.error_mb(), -100.0);
        assert!(!over.was_underestimate());
    }

    #[test]
    fn null_sink_swallows_without_panicking() {
        let sink = NullSink;
        sink.record(CalibrationSample {
            client: Client::Gguf,
            estimated_mb: 1200.0,
            actual_peak_rss_mb: 1300.0,
            dep_crates: None,
            src_mb: None,
            file_mb: Some(1000.0),
            reason: Some(Reason::MemoryBudgetExceeded),
        });
    }
}
