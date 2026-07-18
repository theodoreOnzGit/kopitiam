//! The resource-aware result type — [`Fetched<T>`] and its [`Reason`].
//!
//! This is `temp_ai_design.md` §5's core idea in code: a fetch/launch does not
//! only succeed-or-fail, it can come back **usable-but-reduced**. And crucially,
//! the resource state itself is a **deterministic fact** we hand to the model —
//! so the local AI is *told* "you're in reduced mode: [`Reason::ProjectTooLarge`]"
//! and can report that honestly to the user ("eh, project too big for full
//! analysis on this device, symbol lookups best-effort only ah") instead of
//! silently pretending everything is fine.

/// What a resource-gated fetch or launch gave you back.
///
/// The three arms map one-to-one onto the budget decision FULL / PARTIAL / SKIP
/// (see [`crate::budget::Verdict`]):
///
/// - [`Fetched::Ready`] — FULL. The whole thing ran, here is the answer.
/// - [`Fetched::Partial`] — PARTIAL. A *reduced* provider ran (e.g. workspace-only
///   index, or `kopitiam-syntax` + `cargo metadata` instead of full
///   rust-analyzer). You still get a usable `T`, plus the [`Reason`] it was
///   degraded — pass that reason on to the user/model, don't swallow it.
/// - [`Fetched::Unavailable`] — could not run even the reduced path; here is
///   *why*, so the caller can say so honestly.
///
/// `T` is whatever the fetch produces — a context bundle, a symbol set, a loaded
/// model handle. The enum is generic on purpose: the budgeter does not care what
/// it is gating, only that the caller can carry a payload alongside the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fetched<T> {
    /// Full answer — nothing was cut.
    Ready(T),
    /// Usable but degraded — the payload is real, but a reduced-footprint path
    /// produced it. The [`Reason`] says why; surface it, do not hide it.
    Partial(T, Reason),
    /// Nothing usable was produced; the [`Reason`] says why the budgeter held
    /// off (or the reduced path also could not run).
    Unavailable(Reason),
}

impl<T> Fetched<T> {
    /// `true` only for [`Fetched::Ready`] — i.e. "you got the full thing".
    /// [`Fetched::Partial`] is deliberately **not** ready: it is usable, but the
    /// caller must still know it is reduced, so this stays `false` for it.
    pub fn is_ready(&self) -> bool {
        matches!(self, Fetched::Ready(_))
    }

    /// `true` if there is *any* usable payload — `Ready` or `Partial`. Use this
    /// when you just want to know "did I get a `T` at all", regardless of
    /// fidelity.
    pub fn has_payload(&self) -> bool {
        matches!(self, Fetched::Ready(_) | Fetched::Partial(_, _))
    }

    /// The degradation/unavailability [`Reason`], if any. `Ready` has none.
    pub fn reason(&self) -> Option<Reason> {
        match self {
            Fetched::Ready(_) => None,
            Fetched::Partial(_, r) | Fetched::Unavailable(r) => Some(*r),
        }
    }

    /// Borrow the payload if there is one (`Ready` or `Partial`).
    pub fn payload(&self) -> Option<&T> {
        match self {
            Fetched::Ready(t) | Fetched::Partial(t, _) => Some(t),
            Fetched::Unavailable(_) => None,
        }
    }
}

/// Why a fetch was degraded or refused — a small, closed vocabulary so both the
/// UI and the local model can pattern-match on it.
///
/// These are the exact five arms `temp_ai_design.md` §5 fixes. `Copy` because it
/// is a plain tag and gets handed around (into a [`Fetched`], to the model as a
/// fact, into a log line) constantly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    /// Not enough CPU to run the full path in reasonable time — e.g. a few-core
    /// tablet where rust-analyzer's indexing would jank the editor even if RAM
    /// alone would have fit. CPU is a real input to the gate, not just flavour.
    InsufficientCpu,
    /// The estimated memory cost breaches the RAM budget (`avail · headroom ·
    /// core_factor`). The generic memory reason — used by the gguf client, and
    /// anywhere the driver is raw bytes rather than "project too big".
    MemoryBudgetExceeded,
    /// The full path was abandoned because it took too long against a deadline.
    /// (The budgeter itself is instantaneous; this is for the streaming/anytime
    /// context assembly in §4 that sits on top of it.)
    Timeout,
    /// The *project* is too heavy for this device — the rust-analyzer-flavoured
    /// memory reason. Distinct from [`Reason::MemoryBudgetExceeded`] only so the
    /// message to the user can be specific ("project too big" vs "model file too
    /// big"); both mean "would have OOM'd".
    ProjectTooLarge,
    /// This budgeter does not apply to the situation — e.g. no `Cargo.lock` to
    /// size a project, or a probe that could not read the device. Signals
    /// **fail-open**: the caller should carry on as if unguarded (a capable
    /// desktop must never be blocked by an absent reading), not treat it as a
    /// refusal.
    NotApplicable,
}

impl Reason {
    /// A short Singlish sentence for this reason, ready to echo to the user or
    /// hand to the model as a fact. Kept blunt and specific — the point is the
    /// person (or the model relaying to them) understands *why* it went reduced.
    pub fn blurb(self) -> &'static str {
        match self {
            Reason::InsufficientCpu => {
                "device got too few cores to index the whole thing without janking lah"
            }
            Reason::MemoryBudgetExceeded => {
                "not enough free RAM for this one — would kena OOM-kill if we push"
            }
            Reason::Timeout => "took too long against the deadline, so we stopped short",
            Reason::ProjectTooLarge => {
                "project too big for full analysis on this device — best-effort only ah"
            }
            Reason::NotApplicable => {
                "cannot get an honest reading here, so budgeter stand down (fail-open)"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_is_ready_partial_is_not() {
        let r: Fetched<u32> = Fetched::Ready(1);
        let p: Fetched<u32> = Fetched::Partial(1, Reason::ProjectTooLarge);
        assert!(r.is_ready());
        assert!(!p.is_ready(), "Partial is usable but must NOT report as ready");
        assert!(r.has_payload());
        assert!(p.has_payload(), "Partial still carries a usable payload");
    }

    #[test]
    fn unavailable_has_no_payload_but_has_reason() {
        let u: Fetched<u32> = Fetched::Unavailable(Reason::MemoryBudgetExceeded);
        assert!(!u.has_payload());
        assert_eq!(u.payload(), None);
        assert_eq!(u.reason(), Some(Reason::MemoryBudgetExceeded));
    }

    #[test]
    fn ready_carries_no_reason() {
        let r: Fetched<&str> = Fetched::Ready("full");
        assert_eq!(r.reason(), None);
        assert_eq!(r.payload(), Some(&"full"));
    }

    #[test]
    fn every_reason_has_a_nonempty_blurb() {
        for r in [
            Reason::InsufficientCpu,
            Reason::MemoryBudgetExceeded,
            Reason::Timeout,
            Reason::ProjectTooLarge,
            Reason::NotApplicable,
        ] {
            assert!(!r.blurb().is_empty(), "{r:?} must have a blurb");
        }
    }
}
