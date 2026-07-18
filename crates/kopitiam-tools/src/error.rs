//! [`ToolError`] — the one typed way a tool call can be turned down or fail.
//!
//! Every rejection in the gate contract (schema / path / budget / approval) and
//! every honest execution failure lands here as a typed value. **Never a panic,
//! never a silent execution.** The whole point of the crate is that a bad tool
//! call is a `Result::Err`, not a crashed process or a rogue side effect.

use crate::executor::ToolKind;
use kopitiam_resource::Reason;
use thiserror::Error;

/// Why a tool call was rejected (before running) or failed (while running).
///
/// The first four variants are the **gate rejections** — a call that hits any of
/// them is *never* executed (see the crate-level gate contract). The rest are
/// honest failures from a call that *passed* the gate and then hit reality (file
/// not there, not valid UTF-8, edit target missing, ...).
#[derive(Debug, Error)]
pub enum ToolError {
    /// **Stage 1 (schema).** The model's JSON did not deserialise into the tool's
    /// typed request — missing field, wrong type, malformed shape. Carries the
    /// `serde` message so the loop can feed the real reason back to the model.
    /// Nothing ran.
    #[error("tool call JSON tak match the schema: {0}")]
    Schema(String),

    /// **Stage 2 (path).** A path the call wanted to touch escapes the workspace
    /// sandbox — a `..`-climb above root, an absolute path outside root, or a
    /// symlink that leaves root. Nothing ran. The string says which path and why.
    #[error("path escape the workspace sandbox liao: {0}")]
    PathEscape(String),

    /// **Stage 3 (budget).** The §6 preemptive budgeter
    /// [`Refuse`](kopitiam_resource::Verdict::Refuse)d the op because its estimated
    /// cost would breach the RAM budget on this device (would kena OOM-kill).
    /// Nothing ran. Carries the budgeter's [`Reason`].
    #[error("budget say cannot run this op — would kena OOM-kill ({0:?})")]
    BudgetRefused(Reason),

    /// **Stage 4 (approval).** A [`ToolKind::Write`] or [`ToolKind::Exec`] op was
    /// put to the human (or the stub gate) and got denied. Nothing ran.
    #[error("human approval kena deny for a {0:?} op")]
    ApprovalDenied(ToolKind),

    /// **Stage 5 (execute).** The call passed every gate and then hit an I/O
    /// problem — file missing, permission, is-a-directory, non-UTF-8, etc. The
    /// side effect may be partial only for genuinely partial ops; single-file
    /// writes are atomic-ish (`std::fs::write`).
    #[error("io problem while running the tool: {0}")]
    Io(String),

    /// **Stage 5 (execute), edit-specific.** The `find` text was not present in
    /// the target file, so there was nothing to replace. The file is left
    /// untouched.
    #[error("edit target text cannot find in the file, so nothing changed")]
    EditTargetNotFound,

    /// **Stage 5 (execute), edit-specific.** The `find` text appears more than
    /// once and `all` was not set, so a single-replace would be ambiguous. The
    /// file is left untouched — narrow the `find`, or pass `all: true`.
    #[error("edit target text appear {0} times — not unique, so tak dare replace")]
    EditTargetNotUnique(usize),

    /// **Stage 5 (execute), generic refusal.** The call was well-formed and gated,
    /// but the tool itself declined — e.g. a bad regex in a search, a missing
    /// program for run. Carries a human-readable why.
    #[error("tool cannot run this one: {0}")]
    Refused(String),
}

impl ToolError {
    /// Convenience: wrap an [`std::io::Error`] tagged with the path it was about,
    /// so the message names the file instead of a bare "No such file".
    pub(crate) fn io(context: &str, err: std::io::Error) -> Self {
        ToolError::Io(format!("{context}: {err}"))
    }
}
