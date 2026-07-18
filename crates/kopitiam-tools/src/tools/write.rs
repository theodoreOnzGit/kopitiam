//! [`WriteTool`] — create or overwrite one workspace file. **Scaffolded: the gate
//! + approval are real; the write happens only after an `Approve`.**
//!
//! This is a [`ToolKind::Write`] tool, so a call reaches [`Tool::run`] only after
//! the executor's schema + path + budget gates pass **and** the [`ApprovalGate`]
//! returns `Approve` (the §10.2 human-in-the-loop for consequential edits, with a
//! diff preview in the real TUI). The write itself is a plain, real
//! `std::fs::write` — "scaffolded" refers to the thin ergonomics (no atomic-rename
//! dance, no backup), not to a fake side effect.
//!
//! [`ApprovalGate`]: crate::ApprovalGate

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ToolError;
use crate::executor::{Tool, ToolKind, ToolRequest};
use crate::gate::{self, bytes_to_mb};

/// A write request the model emits.
#[derive(Debug, Clone, Deserialize)]
pub struct WriteReq {
    /// Path to write, relative to the workspace root. Parent directories inside
    /// root are created if missing.
    pub path: String,
    /// The full new contents of the file (overwrites any existing content).
    pub contents: String,
}

/// The write result.
#[derive(Debug, Clone, Serialize)]
pub struct WriteResp {
    /// Path written, relative to root.
    pub path: String,
    /// Bytes written.
    pub bytes_written: usize,
}

impl ToolRequest for WriteReq {
    const TOOL_NAME: &'static str = "write";
    const KIND: ToolKind = ToolKind::Write;

    fn touched_paths(&self) -> Vec<PathBuf> {
        vec![PathBuf::from(&self.path)]
    }

    fn cost_estimate_mb(&self, _root: &Path) -> f64 {
        // The write's cost is its own payload — no walk needed.
        bytes_to_mb(self.contents.len() as u64)
    }

    fn approval_summary(&self) -> String {
        format!("write {} bytes to `{}`", self.contents.len(), self.path)
    }
}

/// The file-write tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct WriteTool;

impl Tool for WriteTool {
    type Req = WriteReq;
    type Resp = WriteResp;

    fn run(&self, request: WriteReq, root: &Path) -> Result<WriteResp, ToolError> {
        let path = gate::confine(root, Path::new(&request.path))?;
        // Create parent dirs — the confine step already proved the parent is
        // inside root, so mkdir-p here cannot escape the sandbox.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ToolError::io(&format!("creating dirs for `{}`", request.path), e))?;
        }
        std::fs::write(&path, request.contents.as_bytes())
            .map_err(|e| ToolError::io(&format!("writing `{}`", request.path), e))?;
        Ok(WriteResp { path: request.path, bytes_written: request.contents.len() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_file_and_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let resp = WriteTool
            .run(
                WriteReq { path: "a/b/c.txt".into(), contents: "steady lah\n".into() },
                &root,
            )
            .unwrap();
        assert_eq!(resp.bytes_written, "steady lah\n".len());
        assert_eq!(std::fs::read_to_string(root.join("a/b/c.txt")).unwrap(), "steady lah\n");
    }

    #[test]
    fn approval_summary_names_the_effect() {
        let req = WriteReq { path: "x.txt".into(), contents: "hi".into() };
        assert_eq!(req.approval_summary(), "write 2 bytes to `x.txt`");
    }
}
