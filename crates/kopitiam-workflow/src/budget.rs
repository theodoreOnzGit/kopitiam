//! Resource budget seam for context assembly (temp_ai_design.md §4, §5, §6).
//!
//! # Why a LOCAL trait, not a dependency
//!
//! The real budgeter — the preemptive `will_fit(cost, MemAvailable·margin)`
//! guard that stops rust-analyzer OOM-killing a tablet (§5/§6) — is being built
//! concurrently as `kopitiam-resource`. To keep `kopitiam-workflow` compiling
//! **independently** of that in-flight crate, we define a small local seam here
//! and code the context builder against it. Wiring this to `kopitiam-resource`
//! (so `kopitiam_resource::will_fit` *implements* [`ResourceBudget`]) is a
//! **follow-up bead**, not part of this scaffold.
//!
//! # The one property the budget must give the context builder
//!
//! §4 Refinement 2: context must be **deterministic-given-budget**:
//!
//! > **context = f(task, budget)** — never f(wall-clock).
//!
//! So a budget must be a *pure, stable* description of "how much may I pull in",
//! not something that changes as the clock ticks or threads race. Given the
//! same task and the same budget, the builder must always produce the same
//! prefix of facts. The budget below is therefore a plain value, deliberately
//! **not** "read the current free RAM right now" (that reading belongs in the
//! real `kopitiam-resource` guard, taken *once* to pick a budget, then held
//! fixed for the assembly so determinism holds).

/// How much context assembly is allowed to pull in.
///
/// The scaffold expresses budget as a **fact allowance**: the maximum number of
/// facts the builder may include. That is a stand-in — the real
/// `kopitiam-resource` budget is a RAM/token cost estimate — but it is enough to
/// make the builder's `context = f(task, budget)` determinism concrete and
/// testable now.
///
/// Contract for implementors: [`Self::fact_allowance`] must be **stable** for
/// the lifetime of one assembly. Returning a value that drifts (e.g. derived
/// from `Instant::now`) breaks the determinism §4 requires and is a bug.
pub trait ResourceBudget {
    /// Maximum number of facts context assembly may include under this budget.
    ///
    /// Must not change between calls within a single assembly — see the trait
    /// docs. `usize::MAX` means "no limit" (a fast box with headroom to spare).
    fn fact_allowance(&self) -> usize;
    // TODO(follow-up: wire kopitiam-resource): the real trait surface will be
    // richer — a cost-estimate check returning Fits | Degrade(Reason) |
    // Refuse(Reason) (§6), and a per-fetch preemptive `will_fit` guard so a
    // mid-reasoning tool-use fetch cannot OOM either (§4 "tool-use steers the
    // stream"). Kept to a single method here so the seam stays tiny and the
    // real budgeter can subsume it without churn.
}

/// A fixed fact-count budget — the simplest [`ResourceBudget`].
///
/// Trivially stable (it is just a number), so it satisfies the determinism
/// contract by construction. Handy for tests and for callers that already know
/// their cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactBudget(pub usize);

impl ResourceBudget for FactBudget {
    fn fact_allowance(&self) -> usize {
        self.0
    }
}

impl FactBudget {
    /// An unbounded budget: pull in everything the task's priority order
    /// offers. Models a machine with ample headroom — on the tablet the real
    /// `kopitiam-resource` guard would hand back a small [`FactBudget`] instead.
    pub const UNBOUNDED: FactBudget = FactBudget(usize::MAX);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_budget_reports_its_allowance_and_is_stable_across_calls() {
        let b = FactBudget(3);
        assert_eq!(b.fact_allowance(), 3);
        assert_eq!(b.fact_allowance(), 3, "budget must not drift between reads");
    }

    #[test]
    fn unbounded_is_effectively_no_limit() {
        assert_eq!(FactBudget::UNBOUNDED.fact_allowance(), usize::MAX);
    }
}
