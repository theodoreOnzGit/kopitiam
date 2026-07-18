//! [`EditTool`] — replace text in one workspace file. **Scaffolded: the gate +
//! approval are real; the edit happens only after an `Approve`.**
//!
//! A [`ToolKind::Write`] tool. Semantics mirror this assistant's own edit tool:
//! an exact-string `find` must match, and (unless `all` is set) it must match
//! **exactly once**, so a single-replace can never silently hit the wrong spot.
//! In the real TUI the approval card shows a `similar`-powered diff preview before
//! the human approves; that preview is the UI's job (`apps/tui`), not this crate's
//! — here the safety is the gate + approval + the uniqueness check.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ToolError;
use crate::executor::{Tool, ToolKind, ToolRequest};
use crate::gate::{self, bytes_to_mb};

/// An edit request the model emits.
#[derive(Debug, Clone, Deserialize)]
pub struct EditReq {
    /// Path to edit, relative to the workspace root.
    pub path: String,
    /// The exact text to find.
    pub find: String,
    /// The text to replace it with.
    pub replace: String,
    /// Replace **every** occurrence. Default `false` = require exactly one match
    /// (else [`ToolError::EditTargetNotUnique`]).
    #[serde(default)]
    pub all: bool,
}

/// The edit result.
#[derive(Debug, Clone, Serialize)]
pub struct EditResp {
    /// Path edited, relative to root.
    pub path: String,
    /// How many occurrences were replaced.
    pub replacements: usize,
}

impl ToolRequest for EditReq {
    const TOOL_NAME: &'static str = "edit";
    const KIND: ToolKind = ToolKind::Write;

    fn touched_paths(&self) -> Vec<PathBuf> {
        vec![PathBuf::from(&self.path)]
    }

    fn cost_estimate_mb(&self, root: &Path) -> f64 {
        // Editing reads the whole file into a String, so the file's size is the
        // memory cost proxy.
        match gate::confine(root, Path::new(&self.path)) {
            Ok(p) => bytes_to_mb(std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)),
            Err(_) => 0.0,
        }
    }

    fn approval_summary(&self) -> String {
        let scope = if self.all { "all occurrences" } else { "1 occurrence" };
        format!("replace {scope} in `{}`", self.path)
    }
}

/// The find/replace edit tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct EditTool;

impl Tool for EditTool {
    type Req = EditReq;
    type Resp = EditResp;

    fn run(&self, request: EditReq, root: &Path) -> Result<EditResp, ToolError> {
        let path = gate::confine(root, Path::new(&request.path))?;
        let original = std::fs::read_to_string(&path)
            .map_err(|e| ToolError::io(&format!("reading `{}` to edit", request.path), e))?;

        let count = original.matches(request.find.as_str()).count();
        if count == 0 {
            return Err(ToolError::EditTargetNotFound);
        }
        if !request.all && count > 1 {
            // Ambiguous single-replace -> refuse, leave the file untouched.
            return Err(ToolError::EditTargetNotUnique(count));
        }

        let (edited, replacements) = if request.all {
            (original.replace(request.find.as_str(), &request.replace), count)
        } else {
            (original.replacen(request.find.as_str(), &request.replace, 1), 1)
        };

        std::fs::write(&path, edited.as_bytes())
            .map_err(|e| ToolError::io(&format!("writing edited `{}`", request.path), e))?;
        Ok(EditResp { path: request.path, replacements })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("f.rs"), contents).unwrap();
        (dir, root)
    }

    #[test]
    fn single_unique_replace_works() {
        let (_d, root) = file("let x = 1;\n");
        let resp = EditTool
            .run(
                EditReq { path: "f.rs".into(), find: "1".into(), replace: "2".into(), all: false },
                &root,
            )
            .unwrap();
        assert_eq!(resp.replacements, 1);
        assert_eq!(std::fs::read_to_string(root.join("f.rs")).unwrap(), "let x = 2;\n");
    }

    #[test]
    fn ambiguous_single_replace_is_refused_and_file_untouched() {
        let (_d, root) = file("a a a\n");
        let err = EditTool
            .run(
                EditReq { path: "f.rs".into(), find: "a".into(), replace: "b".into(), all: false },
                &root,
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::EditTargetNotUnique(3)), "got {err:?}");
        assert_eq!(std::fs::read_to_string(root.join("f.rs")).unwrap(), "a a a\n", "left untouched");
    }

    #[test]
    fn all_replaces_every_occurrence() {
        let (_d, root) = file("a a a\n");
        let resp = EditTool
            .run(
                EditReq { path: "f.rs".into(), find: "a".into(), replace: "b".into(), all: true },
                &root,
            )
            .unwrap();
        assert_eq!(resp.replacements, 3);
        assert_eq!(std::fs::read_to_string(root.join("f.rs")).unwrap(), "b b b\n");
    }

    #[test]
    fn missing_target_is_typed_error() {
        let (_d, root) = file("hello\n");
        let err = EditTool
            .run(
                EditReq { path: "f.rs".into(), find: "zzz".into(), replace: "b".into(), all: false },
                &root,
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::EditTargetNotFound), "got {err:?}");
    }
}
