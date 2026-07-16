//! Project-wide content search — the pure-Rust engine behind `:grep` /
//! `:vimgrep`.
//!
//! # Why this is here and not a shell-out to `rg`/`grep`
//!
//! Vim's `:grep` shells out to whatever `grepprg` points at (usually `grep` or
//! `ripgrep`). kvim cannot: the whole reason this crate exists is to run on
//! Android, where there is no `grep`, no `rg`, and no guarantee a `PATH` binary
//! is anything at all. So the search is done *in-process* with two crates this
//! workspace already carries — [`ignore`] for the `.gitignore`-aware directory
//! walk (the same crate the file tree and file picker use) and [`regex`] for the
//! matching. No external process, no `grepprg`, nothing to install: the Pure
//! Rust Core promise, applied to search. See `CLAUDE.md`.
//!
//! # What it does and does not do (scope, stated plainly)
//!
//! * **One entry per matching line.** Like `grep -n`, a line that matches is
//!   reported once, at the column of its *first* match — not once per match on
//!   the line. (Vim's `:vimgrep` without the `g` flag does the same; with `g` it
//!   lists every match. kvim does not implement the `g` flag yet.)
//! * **`.gitignore` is honoured** by default, plus `.ignore` and git's global
//!   excludes — whatever [`ignore::WalkBuilder`] respects out of the box. This is
//!   what keeps `target/` and `node_modules/` out of your results.
//! * **Non-UTF-8 files are skipped**, not lossily searched: a binary that
//!   happens to contain the pattern's bytes is noise, and the honest answer for a
//!   grep is "text files only".
//! * **The result is capped** at a caller-supplied limit so a broad pattern over
//!   a huge tree cannot lock the editor up building a million-entry list; when
//!   the cap is hit the walk stops and [`GrepOutcome::truncated`] is set so the
//!   UI can say so.
//!
//! The engine is headless — it takes a root, a compiled pattern and a cap, and
//! returns data. It never touches a terminal, which is what lets the tests below
//! drive it over a real temp-dir tree with no editor in sight.

use std::path::{Path, PathBuf};

use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use regex::Regex;

/// One matching line found by [`grep`]: enough to become a quickfix entry and to
/// jump to. `line` and `col` are **1-based**, the way vim's quickfix list (and
/// every `file:line:col` convention) counts, so they can be shown to the user
/// as-is and only need converting to 0-based [`crate::core::Position`] at the
/// moment of the jump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepMatch {
    /// The file the match was found in, as walked from the search root.
    pub path: PathBuf,
    /// 1-based line number of the match.
    pub line: usize,
    /// 1-based column (in `char`s) of the first match on the line.
    pub col: usize,
    /// The whole matching line, trailing newline stripped, for display in the
    /// quickfix window.
    pub text: String,
}

/// The result of a [`grep`] run: the matches, and whether the cap cut the walk
/// short.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepOutcome {
    pub matches: Vec<GrepMatch>,
    /// `true` when the search stopped because it hit the cap — there may be more
    /// matches on disk than are in `matches`. The UI reports this rather than
    /// pretending the list is complete.
    pub truncated: bool,
}

/// Walks `root` (honouring `.gitignore`) and collects every line matching
/// `pattern`, up to `cap` matches.
///
/// `globs`, when non-empty, restrict the walk to paths matching them
/// (gitignore-style, e.g. `*.rs`, `src/**`) — this is the `[globs]` argument of
/// `:grep pattern globs`. An unparseable glob is skipped rather than failing the
/// whole search; a search that quietly finds nothing is a worse failure mode
/// than one that ignores a typo'd filter.
///
/// The walk is single-threaded on purpose: it runs synchronously on the UI
/// thread (like kvim's LSP calls), match order must be deterministic for the
/// quickfix list to be stable, and `ignore`'s parallel walker would give neither
/// for no real gain on the tree sizes an editor greps.
pub fn grep(root: &Path, pattern: &Regex, globs: &[String], cap: usize) -> GrepOutcome {
    let mut builder = WalkBuilder::new(root);
    // Honour `.gitignore` even when the project is not (yet) a git repository.
    // `ignore` defaults to applying `.gitignore` only inside a git worktree; an
    // editor's user expects `target/`/`node_modules/` skipped whether or not
    // they have run `git init`, so this opts into the ripgrep-style "respect
    // .gitignore regardless" behaviour.
    builder.require_git(false);
    // Restrict to the requested globs, if any. `OverrideBuilder`'s globs are a
    // whitelist: once any is added, a path matching none of them is skipped,
    // which is exactly "only search these globs".
    if !globs.is_empty() {
        let mut ob = OverrideBuilder::new(root);
        for g in globs {
            // A bad glob is dropped, not fatal — see the doc comment.
            let _ = ob.add(g);
        }
        if let Ok(over) = ob.build() {
            builder.overrides(over);
        }
    }

    let mut matches = Vec::new();
    let mut truncated = false;

    'walk: for dent in builder.build() {
        let Ok(dent) = dent else { continue };
        // Directories, symlinks-to-dirs, and anything without a plain-file type
        // carry no lines to search.
        if !dent.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = dent.path();
        // Non-UTF-8 (binary) files are skipped, not searched lossily.
        let Ok(content) = std::fs::read_to_string(path) else { continue };

        for (idx, line_text) in content.lines().enumerate() {
            let Some(m) = pattern.find(line_text) else { continue };
            // 1-based line, and 1-based *char* column (not byte offset): the
            // column is for humans and for a grapheme-indexed cursor, neither of
            // which counts bytes.
            let col = line_text[..m.start()].chars().count() + 1;
            matches.push(GrepMatch {
                path: path.to_path_buf(),
                line: idx + 1,
                col,
                text: line_text.to_string(),
            });
            if matches.len() >= cap {
                truncated = true;
                break 'walk;
            }
        }
    }

    // The `ignore` walk order is filesystem-dependent; sort into a stable
    // (path, line, col) order so the quickfix list is deterministic — the same
    // search gives the same list every time, which is what makes `:cc 3` mean a
    // fixed entry and what lets the tests assert on order.
    matches.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)).then(a.col.cmp(&b.col)));

    GrepOutcome { matches, truncated }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Builds a small project tree in a fresh temp dir:
    /// ```text
    ///   src/main.rs   -> two TODO lines
    ///   src/lib.rs    -> one TODO line
    ///   target/junk.rs-> a TODO line that .gitignore should hide
    ///   .gitignore    -> "target/"
    /// ```
    fn sample_tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir(root.join("src")).unwrap();
        fs::create_dir(root.join("target")).unwrap();
        fs::write(root.join(".gitignore"), "target/\n").unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {\n    // TODO: wire it up\n    let x = 1; // TODO again\n}\n").unwrap();
        fs::write(root.join("src/lib.rs"), "// TODO: document this\npub fn f() {}\n").unwrap();
        fs::write(root.join("target/junk.rs"), "// TODO: this is generated, ignore me\n").unwrap();
        dir
    }

    #[test]
    fn finds_matches_and_respects_gitignore() {
        let dir = sample_tree();
        let re = Regex::new("TODO").unwrap();
        let out = grep(dir.path(), &re, &[], 1000);
        // Three TODO lines across src/, and NONE from the git-ignored target/.
        assert_eq!(out.matches.len(), 3, "got: {:?}", out.matches);
        assert!(!out.truncated);
        assert!(out.matches.iter().all(|m| !m.path.to_string_lossy().contains("target")), "target/ must be ignored");
        // Every match carries the whole line and a sane 1-based line/col.
        let libs: Vec<_> = out.matches.iter().filter(|m| m.path.ends_with("lib.rs")).collect();
        assert_eq!(libs.len(), 1);
        assert_eq!(libs[0].line, 1);
        assert_eq!(libs[0].col, 4); // "// TODO" — T is the 4th char (1-based).
        assert!(libs[0].text.contains("document this"));
    }

    #[test]
    fn one_entry_per_matching_line_not_per_match() {
        let dir = sample_tree();
        let re = Regex::new("TODO").unwrap();
        let out = grep(dir.path(), &re, &[], 1000);
        // main.rs has two lines each mentioning TODO; the second line says "TODO"
        // once — so main.rs contributes exactly two entries, one per line.
        let mains: Vec<_> = out.matches.iter().filter(|m| m.path.ends_with("main.rs")).collect();
        assert_eq!(mains.len(), 2);
    }

    #[test]
    fn globs_restrict_the_walk() {
        let dir = sample_tree();
        let re = Regex::new("TODO").unwrap();
        // Only *.rs under lib — restrict to lib.rs by name.
        let out = grep(dir.path(), &re, &["**/lib.rs".to_string()], 1000);
        assert_eq!(out.matches.len(), 1);
        assert!(out.matches[0].path.ends_with("lib.rs"));
    }

    #[test]
    fn the_cap_truncates_and_flags_it() {
        let dir = sample_tree();
        let re = Regex::new("TODO").unwrap();
        let out = grep(dir.path(), &re, &[], 2);
        assert_eq!(out.matches.len(), 2);
        assert!(out.truncated, "hitting the cap must set truncated");
    }

    #[test]
    fn no_matches_is_an_empty_untruncated_result() {
        let dir = sample_tree();
        let re = Regex::new("definitely-not-in-the-tree").unwrap();
        let out = grep(dir.path(), &re, &[], 1000);
        assert!(out.matches.is_empty());
        assert!(!out.truncated);
    }
}
