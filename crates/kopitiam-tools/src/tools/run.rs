//! [`RunTool`] — spawn a process with its working directory confined to the
//! workspace. **Scaffolded: the gate + approval are real; the spawn happens only
//! after an `Approve`.**
//!
//! This is the sharpest tool — a [`ToolKind::Exec`] op — so it carries the
//! heaviest gate. The `cwd` is confined inside the workspace root (a `..`-escape
//! `cwd` is rejected like any other path), and the process runs **only** after the
//! [`ApprovalGate`] approves. On Termux the filesystem is already sandboxed to the
//! app's own home, giving a second containment layer under the approval (§10.4).
//!
//! Note: this crate does **not** try to sandbox *what* the approved program does
//! once it runs — that is the human's call at the approval card (and the OS
//! sandbox on Termux). The crate's job is: no exec without a confined cwd and an
//! explicit `Approve`.
//!
//! [`ApprovalGate`]: crate::ApprovalGate

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::ToolError;
use crate::executor::{Tool, ToolKind, ToolRequest};
use crate::gate;

/// A run request the model emits.
#[derive(Debug, Clone, Deserialize)]
pub struct RunReq {
    /// The program to run (looked up on `PATH`, or an absolute program path).
    pub program: String,
    /// Arguments, in order.
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory, relative to the workspace root. `None` = the root itself.
    #[serde(default)]
    pub cwd: Option<String>,
}

impl RunReq {
    /// The cwd this run would use, relative to root (`"."` if unset).
    fn cwd_subdir(&self) -> &str {
        self.cwd.as_deref().unwrap_or(".")
    }

    /// A shell-ish rendering of the command, for the approval card.
    fn command_line(&self) -> String {
        if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        }
    }
}

/// The run result — the finished process's captured output.
#[derive(Debug, Clone, Serialize)]
pub struct RunResp {
    /// Exit code, or `None` if the process was killed by a signal.
    pub exit_code: Option<i32>,
    /// Captured stdout (UTF-8 lossy).
    pub stdout: String,
    /// Captured stderr (UTF-8 lossy).
    pub stderr: String,
}

impl ToolRequest for RunReq {
    const TOOL_NAME: &'static str = "run";
    const KIND: ToolKind = ToolKind::Exec;

    fn touched_paths(&self) -> Vec<PathBuf> {
        // The gated path is the cwd — the process runs *in* the workspace. We
        // cannot gate what an arbitrary program touches once running (that is the
        // approval + OS-sandbox's job), but the cwd must at least be inside root.
        vec![PathBuf::from(self.cwd_subdir())]
    }

    fn cost_estimate_mb(&self, _root: &Path) -> f64 {
        // A process's RAM is unknowable up front. We do not try to estimate it
        // here; the real containment for exec is the approval seam + the OS
        // sandbox, not the byte-budget. Report a nominal small cost so the budget
        // gate never *falsely* refuses a legitimate approved command — the human,
        // not the byte-budgeter, is the guard for exec (§10.2).
        0.0
    }

    fn approval_summary(&self) -> String {
        format!("run `{}` in `{}`", self.command_line(), self.cwd_subdir())
    }
}

/// The process-run tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunTool;

impl Tool for RunTool {
    type Req = RunReq;
    type Resp = RunResp;

    fn run(&self, request: RunReq, root: &Path) -> Result<RunResp, ToolError> {
        let cwd = gate::confine(root, Path::new(request.cwd_subdir()))?;
        let output = Command::new(&request.program)
            .args(&request.args)
            .current_dir(&cwd)
            .output()
            .map_err(|e| ToolError::io(&format!("spawning `{}`", request.program), e))?;
        Ok(RunResp {
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_summary_renders_the_command() {
        let req = RunReq {
            program: "cargo".into(),
            args: vec!["test".into(), "--release".into()],
            cwd: None,
        };
        assert_eq!(req.approval_summary(), "run `cargo test --release` in `.`");
    }

    #[test]
    fn a_cwd_escape_is_rejected_before_any_spawn() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let err = RunTool
            .run(
                RunReq { program: "true".into(), args: vec![], cwd: Some("../..".into()) },
                &root,
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)), "got {err:?}");
    }

    // NOTE: we deliberately do NOT spawn a real process in the unit tests. The
    // exec path is exercised end-to-end through the executor's approval gate in
    // the crate integration tests (with AutoDeny proving it is blocked, and a
    // trivial approved command proving the spawn works only after approval).
}
