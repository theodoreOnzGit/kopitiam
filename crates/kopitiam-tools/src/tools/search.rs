//! [`SearchTool`] — recursive text search across the workspace, honouring
//! `.gitignore` and skipping `target/` (via the `ignore` crate). **Real, working.**
//!
//! This is a read-only tool, so it never touches the approval seam — but it *is*
//! budget-gated, because "search the whole tree" is exactly the op §10.2 calls out
//! as needing a preemptive budget check on a tablet. Its cost estimate is a cheap
//! stat-only byte count of the subtree it would scan
//! ([`crate::gate::stat_only_bytes`]); on a tiny device a huge tree is
//! [`Refuse`](kopitiam_resource::Verdict::Refuse)d before the walk ever starts.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ToolError;
use crate::executor::{Tool, ToolKind, ToolRequest};
use crate::gate::{self, bytes_to_mb};

/// Default cap on returned hits, so a broad query cannot flood the context window.
const DEFAULT_MAX_RESULTS: usize = 200;

/// A search request the model emits.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchReq {
    /// The needle. A plain substring by default; a regex if [`SearchReq::regex`]
    /// is set.
    pub query: String,
    /// Subdirectory (relative to the workspace root) to search under. `None` /
    /// omitted = the whole workspace root.
    #[serde(default)]
    pub path: Option<String>,
    /// Treat [`SearchReq::query`] as a regular expression (`regex` crate syntax)
    /// instead of a literal substring.
    #[serde(default)]
    pub regex: bool,
    /// Stop after this many hits. `None` = [`DEFAULT_MAX_RESULTS`].
    #[serde(default)]
    pub max_results: Option<usize>,
}

impl SearchReq {
    /// The subtree this search would walk, relative to root (`"."` if unset).
    fn search_subdir(&self) -> &str {
        self.path.as_deref().unwrap_or(".")
    }
}

/// One matching line.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SearchHit {
    /// Path of the file, **relative to the workspace root** (so the model never
    /// sees absolute host paths).
    pub path: String,
    /// 1-based line number of the match.
    pub line_no: usize,
    /// The full matching line, trimmed of its trailing newline.
    pub line: String,
}

/// The search result.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResp {
    /// The matches, in walk order, capped at `max_results`.
    pub hits: Vec<SearchHit>,
    /// `true` if the cap was hit and there may be more matches not returned.
    pub truncated: bool,
}

impl ToolRequest for SearchReq {
    const TOOL_NAME: &'static str = "search";
    const KIND: ToolKind = ToolKind::ReadOnly;

    fn touched_paths(&self) -> Vec<PathBuf> {
        // The only path it touches is the subtree root; the confinement of that
        // root is enough, because the `ignore` walk stays inside it.
        vec![PathBuf::from(self.search_subdir())]
    }

    fn cost_estimate_mb(&self, root: &Path) -> f64 {
        // Cheap stat-only proxy: how many bytes would the walk have to chew? If
        // the subdir does not confine, fall back to 0.0 — the path gate will
        // reject the call anyway, so the cost here is moot.
        match gate::confine(root, Path::new(self.search_subdir())) {
            Ok(dir) => bytes_to_mb(gate::stat_only_bytes(&dir)),
            Err(_) => 0.0,
        }
    }
}

/// The recursive-search tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchTool;

impl Tool for SearchTool {
    type Req = SearchReq;
    type Resp = SearchResp;

    fn run(&self, request: SearchReq, root: &Path) -> Result<SearchResp, ToolError> {
        let dir = gate::confine(root, Path::new(request.search_subdir()))?;
        let cap = request.max_results.unwrap_or(DEFAULT_MAX_RESULTS);

        // Build the matcher once. A bad regex is a caller error, not a panic.
        let matcher = Matcher::build(&request.query, request.regex)?;

        let mut hits = Vec::new();
        let mut truncated = false;

        // `require_git(false)` so `.gitignore` is honoured even outside a git repo
        // — matches the cost-estimate walk in `gate::stat_only_bytes`.
        'walk: for entry in ignore::WalkBuilder::new(&dir)
            .hidden(false)
            .require_git(false)
            .build()
            .flatten()
        {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            // Read the file. Budget already cleared this walk, so opening is safe.
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(_) => continue, // unreadable file -> skip, don't abort the walk
            };
            // Skip binary / non-UTF-8 files rather than emit garbage lines.
            let text = match std::str::from_utf8(&bytes) {
                Ok(t) => t,
                Err(_) => continue,
            };
            // Relative path for the report — never leak the absolute host path.
            let rel = path.strip_prefix(root).unwrap_or(path);
            for (idx, line) in text.lines().enumerate() {
                if matcher.is_match(line) {
                    hits.push(SearchHit {
                        path: rel.to_string_lossy().into_owned(),
                        line_no: idx + 1,
                        line: line.to_string(),
                    });
                    if hits.len() >= cap {
                        truncated = true;
                        break 'walk;
                    }
                }
            }
        }

        Ok(SearchResp { hits, truncated })
    }
}

/// Either a compiled regex or a literal substring — the two search modes, behind
/// one `is_match`.
enum Matcher {
    Literal(String),
    Regex(regex::Regex),
}

impl Matcher {
    fn build(query: &str, is_regex: bool) -> Result<Self, ToolError> {
        if is_regex {
            regex::Regex::new(query)
                .map(Matcher::Regex)
                .map_err(|e| ToolError::Refused(format!("bad regex `{query}`: {e}")))
        } else {
            Ok(Matcher::Literal(query.to_string()))
        }
    }

    fn is_match(&self, line: &str) -> bool {
        match self {
            Matcher::Literal(needle) => line.contains(needle.as_str()),
            Matcher::Regex(re) => re.is_match(line),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "fn chope() {}\n// TODO tidy\n").unwrap();
        fs::write(dir.path().join("src/main.rs"), "fn main() { chope(); }\n").unwrap();
        // A gitignored file that must NOT be searched.
        fs::write(dir.path().join(".gitignore"), "ignored.rs\n").unwrap();
        fs::write(dir.path().join("ignored.rs"), "fn chope() {}\n").unwrap();
        dir
    }

    #[test]
    fn substring_search_finds_matches_and_respects_gitignore() {
        let dir = tree();
        let root = dir.path().canonicalize().unwrap();
        let resp = SearchTool
            .run(
                SearchReq { query: "chope".into(), path: None, regex: false, max_results: None },
                &root,
            )
            .unwrap();
        // src/lib.rs and src/main.rs match; ignored.rs is gitignored -> excluded.
        assert_eq!(resp.hits.len(), 2, "hits: {:?}", resp.hits);
        assert!(resp.hits.iter().all(|h| !h.path.contains("ignored")));
        assert!(!resp.truncated);
    }

    #[test]
    fn regex_search_works() {
        let dir = tree();
        let root = dir.path().canonicalize().unwrap();
        let resp = SearchTool
            .run(
                SearchReq {
                    query: r"fn \w+\(\)".into(),
                    path: None,
                    regex: true,
                    max_results: None,
                },
                &root,
            )
            .unwrap();
        assert!(resp.hits.iter().any(|h| h.line.contains("fn chope()")));
    }

    #[test]
    fn max_results_truncates() {
        let dir = tree();
        let root = dir.path().canonicalize().unwrap();
        let resp = SearchTool
            .run(
                SearchReq { query: "chope".into(), path: None, regex: false, max_results: Some(1) },
                &root,
            )
            .unwrap();
        assert_eq!(resp.hits.len(), 1);
        assert!(resp.truncated);
    }

    #[test]
    fn bad_regex_is_a_typed_refusal_not_a_panic() {
        let dir = tree();
        let root = dir.path().canonicalize().unwrap();
        let err = SearchTool
            .run(
                SearchReq { query: "(".into(), path: None, regex: true, max_results: None },
                &root,
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::Refused(_)), "got {err:?}");
    }
}
