//! The **approval seam** — stage 4 of the gate contract, and the concrete home of
//! the §10.2 "human is the judgment rung" idea.
//!
//! A read-only tool needs no permission. A **write** or **exec** tool is
//! consequential, so before it runs, the executor asks an [`ApprovalGate`] the
//! caller supplied. In the real TUI this gate renders a **choice card** (framed
//! question + a diff preview for edits + a recommended default) and waits for the
//! human — "safety = the same primitive" as every other §10.2 decision, just
//! wearing a safety hat. In tests, [`AutoApprove`] / [`AutoDeny`] stand in for the
//! human so the gating can be exercised without a UI.
//!
//! The crate deliberately does **not** decide *how* approval is obtained — that is
//! the client's job (TUI prompt, config policy, "always allow this dir", the
//! persist-decisions flywheel). This crate only guarantees that write/exec **do
//! not run** without an `Approve` coming back through this seam.

use std::path::PathBuf;

use crate::executor::ToolKind;

/// What the executor tells an [`ApprovalGate`] about the op awaiting a decision.
///
/// Enough for a human (or a policy) to make the call without re-deriving anything:
/// which tool, whether it writes or execs, exactly which paths it would touch, and
/// a one-line human-readable summary of the effect (e.g. "write 812 bytes to
/// `src/lib.rs`", "run `cargo test` in `.`").
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// The tool's stable name (`"write"`, `"edit"`, `"run"`).
    pub tool: &'static str,
    /// Write or Exec (read-only ops never reach the approval seam).
    pub kind: ToolKind,
    /// Every workspace-confined path the op would touch. Already gated inside
    /// root by the time approval is asked — so the human is deciding *whether*,
    /// not *whether it is safe to leave the sandbox* (that was already refused).
    pub paths: Vec<PathBuf>,
    /// A short human-readable description of the effect, for the choice card.
    pub summary: String,
}

/// The human's (or policy's) answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Go ahead — the executor proceeds to stage 5 (execute).
    Approve,
    /// Hold off — the executor returns [`crate::ToolError::ApprovalDenied`] and
    /// nothing runs.
    Deny,
}

impl ApprovalDecision {
    /// `true` only for [`ApprovalDecision::Approve`].
    pub fn is_approved(self) -> bool {
        matches!(self, ApprovalDecision::Approve)
    }
}

/// The seam the caller implements to say yes/no to a write/exec op.
///
/// One method, deliberately. Whatever the caller wants — prompt the human, apply a
/// remembered "always allow" rule, auto-deny in a headless run — it all reduces to
/// "given this [`ApprovalRequest`], approve or deny". The executor never inspects
/// *how* the decision was reached.
pub trait ApprovalGate {
    /// Decide whether the op may run. Called **only** for write/exec ops, **after**
    /// schema + path + budget have already passed — so a `Deny` here is a pure
    /// judgment call, not a safety backstop (safety was enforced by the earlier
    /// stages).
    fn decide(&self, request: &ApprovalRequest) -> ApprovalDecision;
}

/// Test/headless stub: **approve everything**. Use it to exercise the *execute*
/// path of write/exec tools in tests. **Never** wire this into a real interactive
/// session — it defeats the whole §10.2 human-in-the-loop point.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoApprove;

impl ApprovalGate for AutoApprove {
    fn decide(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Approve
    }
}

/// Test/safe-default stub: **deny everything**. This is the correct default for
/// any context that has no way to ask a human — better a denied write than an
/// unattended one. Use it to prove the gate actually blocks write/exec.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoDeny;

impl ApprovalGate for AutoDeny {
    fn decide(&self, _request: &ApprovalRequest) -> ApprovalDecision {
        ApprovalDecision::Deny
    }
}
