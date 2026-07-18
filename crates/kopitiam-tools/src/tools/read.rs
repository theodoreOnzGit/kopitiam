//! [`ReadTool`] — read one workspace file's text. **Real, working.**
//!
//! Read-only, so no approval. Budget-gated by the file's own size (a giant file
//! read into a `String` is itself a memory cost). Non-UTF-8 files are rejected
//! with a typed error rather than returned as garbage — this is a *code/text*
//! read tool, not a binary dumper.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ToolError;
use crate::executor::{Tool, ToolKind, ToolRequest};
use crate::gate::{self, bytes_to_mb};

/// A read request the model emits.
#[derive(Debug, Clone, Deserialize)]
pub struct ReadReq {
    /// Path to the file, relative to the workspace root.
    pub path: String,
    /// Cap the returned text at this many bytes (truncating on a char boundary).
    /// `None` = whole file.
    #[serde(default)]
    pub max_bytes: Option<usize>,
}

/// The read result.
#[derive(Debug, Clone, Serialize)]
pub struct ReadResp {
    /// Path read, relative to root (echoed back for the model's bookkeeping).
    pub path: String,
    /// The file's text (possibly truncated — see [`ReadResp::truncated`]).
    pub contents: String,
    /// `true` if [`ReadReq::max_bytes`] cut the content short.
    pub truncated: bool,
}

impl ToolRequest for ReadReq {
    const TOOL_NAME: &'static str = "read";
    const KIND: ToolKind = ToolKind::ReadOnly;

    fn touched_paths(&self) -> Vec<PathBuf> {
        vec![PathBuf::from(&self.path)]
    }

    fn cost_estimate_mb(&self, root: &Path) -> f64 {
        // The file's on-disk size, stat-only, is a fair proxy for the RAM the read
        // will want. Confinement failure -> 0.0 (the path gate rejects anyway).
        match gate::confine(root, Path::new(&self.path)) {
            Ok(p) => bytes_to_mb(std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)),
            Err(_) => 0.0,
        }
    }
}

/// The file-read tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadTool;

impl Tool for ReadTool {
    type Req = ReadReq;
    type Resp = ReadResp;

    fn run(&self, request: ReadReq, root: &Path) -> Result<ReadResp, ToolError> {
        let path = gate::confine(root, Path::new(&request.path))?;
        if path.is_dir() {
            return Err(ToolError::Refused(format!(
                "`{}` is a directory, not a file — use search to list it",
                request.path
            )));
        }
        let bytes = std::fs::read(&path).map_err(|e| ToolError::io(&format!("reading `{}`", request.path), e))?;

        // Truncate on a char boundary if asked, BEFORE the UTF-8 check, so a cap
        // never splits a multi-byte char.
        let (slice, truncated) = match request.max_bytes {
            Some(max) if bytes.len() > max => {
                let mut cut = max;
                // Back off to the last char boundary at or below `max`.
                while cut > 0 && (bytes[cut] & 0b1100_0000) == 0b1000_0000 {
                    cut -= 1;
                }
                (&bytes[..cut], true)
            }
            _ => (&bytes[..], false),
        };

        let contents = std::str::from_utf8(slice)
            .map_err(|_| {
                ToolError::Refused(format!("`{}` is not valid UTF-8 text", request.path))
            })?
            .to_string();

        Ok(ReadResp { path: request.path, contents, truncated })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn reads_a_file_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::write(root.join("note.txt"), "makan already?\n").unwrap();
        let resp = ReadTool
            .run(ReadReq { path: "note.txt".into(), max_bytes: None }, &root)
            .unwrap();
        assert_eq!(resp.contents, "makan already?\n");
        assert!(!resp.truncated);
    }

    #[test]
    fn max_bytes_truncates_on_char_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        // "sé" is s (1 byte) + é (2 bytes). Cap at 2 must back off to just "s".
        fs::write(root.join("u.txt"), "sé").unwrap();
        let resp = ReadTool
            .run(ReadReq { path: "u.txt".into(), max_bytes: Some(2) }, &root)
            .unwrap();
        assert_eq!(resp.contents, "s");
        assert!(resp.truncated);
    }

    #[test]
    fn reading_a_directory_is_a_typed_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        let err = ReadTool.run(ReadReq { path: "sub".into(), max_bytes: None }, &root).unwrap_err();
        assert!(matches!(err, ToolError::Refused(_)), "got {err:?}");
    }

    #[test]
    fn reading_a_missing_file_is_a_typed_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let err = ReadTool.run(ReadReq { path: "ghost.txt".into(), max_bytes: None }, &root).unwrap_err();
        assert!(matches!(err, ToolError::Io(_)), "got {err:?}");
    }
}
