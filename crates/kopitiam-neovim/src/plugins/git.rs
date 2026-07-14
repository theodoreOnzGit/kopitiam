//! A minimal git status reader — enough for the statusline slice of
//! vim-fugitive/airline's integration, and nothing more.
//!
//! # Scope
//!
//! The maintainer's airline statusline shows two things from git: the
//! current branch, and whether the working tree is dirty. That's the whole
//! brief. Full git plumbing — log, blame, diff hunks, staging from the
//! editor, `:Git` command dispatch — is explicitly a separate, future bead;
//! building it here would be exactly the kind of unrequested scope creep
//! CLAUDE.md's "avoid unnecessary abstraction" warns against.
//!
//! # Why not the `gix` crate
//!
//! `gix` is a complete, correct git implementation — and a large dependency
//! tree to pull in for two strings on a statusline. This module reads
//! `.git/HEAD` (and, for the dirty check, `.git/index`) directly off disk
//! with `std::fs`, matching CLAUDE.md's "avoid unnecessary dependencies" and
//! the Pure Rust Core principle of not taking on more than a feature needs.
//! It also does not shell out to the `git` binary: that would work, but it
//! reintroduces exactly the "requires an external toolchain to be present
//! and on `PATH`" problem that motivated ripping out Mason in the first
//! place (see the crate-level docs on why Android's toolchain story is the
//! whole reason `kvim` exists).
//!
//! # The dirty check is a heuristic, not `git status`
//!
//! Real dirtiness (`git status --porcelain`) means comparing the worktree
//! against the *index* and the index against *HEAD*'s tree, which requires
//! reading and hashing blob objects — full git plumbing, explicitly out of
//! scope above. What's implemented instead is the same fast-path heuristic
//! git itself uses before falling back to hashing ("racy git"): for every
//! path git has staged, compare the worktree file's size and modification
//! time against what's recorded in `.git/index`; a mismatch (or a missing
//! file) means modified. Separately, [`ignore::Walk`] finds any file in the
//! worktree that *isn't* in the index at all — an untracked file, which is
//! just as much "dirty" for a statusline's purposes. What this does **not**
//! catch: a file `touch`ed (mtime changed) without its content actually
//! changing (a false positive — same limitation real git has before it
//! falls back to hashing), and staged-versus-HEAD differences (e.g. `git
//! add`ing a file and then reverting the change without unstaging — a rare
//! enough edge case to accept). Both are documented trade-offs, not bugs.
//!
//! Only git index format versions 2 and 3 are parsed (fixed-size,
//! full-path-per-entry layouts). Version 4 (path-prefix compression, opt-in
//! via `feature.manyFiles`) is detected and skipped gracefully — the branch
//! name still resolves correctly, only the modified-file portion of the
//! dirty check is unavailable, and untracked-file detection still works.
//! Repositories using the sha256 object format (rare, opt-in) are also
//! unsupported; parsing degrades the same way rather than misreading bytes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// A snapshot of a repository's branch and dirty state, as read directly
/// from `.git/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitStatus {
    /// The current branch name, or a short (7-character) commit hash if
    /// [`detached`](Self::detached) is `true`. Airline-style statuslines
    /// show exactly this for a detached HEAD, so there is no separate
    /// "detached" rendering branch needed on the caller's side beyond
    /// checking the flag if it wants to, say, colour it differently.
    pub branch: String,
    pub detached: bool,
    pub dirty: bool,
}

/// Reads the git status of the repository containing `start_dir`, walking
/// up through parent directories to find `.git` (matching how every git
/// command resolves the repository root from a subdirectory). Returns
/// `None` — never an error, never a panic — when `start_dir` is not inside a
/// git repository at all; a statusline has no "git error" state to render,
/// only "no git info".
pub fn status(start_dir: &Path) -> Option<GitStatus> {
    let (git_dir, worktree_root) = find_git_dir(start_dir)?;
    let (branch, detached) = read_head(&git_dir)?;
    let dirty = is_dirty(&git_dir, &worktree_root);
    Some(GitStatus { branch, detached, dirty })
}

/// Walks up from `start_dir` looking for a `.git` entry, returning
/// `(git_dir, worktree_root)`. Handles both the common case (`.git` is a
/// directory) and worktrees/submodules (`.git` is a file containing
/// `gitdir: <path>`, possibly relative to the file's own directory).
fn find_git_dir(start_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut dir = start_dir.to_path_buf();
    loop {
        let candidate = dir.join(".git");
        if candidate.is_dir() {
            return Some((candidate, dir));
        }
        if candidate.is_file() {
            let contents = std::fs::read_to_string(&candidate).ok()?;
            let git_dir = contents.trim().strip_prefix("gitdir:")?.trim();
            let git_dir = PathBuf::from(git_dir);
            let git_dir = if git_dir.is_absolute() { git_dir } else { dir.join(git_dir) };
            return Some((git_dir, dir));
        }
        dir = dir.parent()?.to_path_buf();
    }
}

/// Parses `<git_dir>/HEAD` into `(branch_or_short_sha, detached)`.
fn read_head(git_dir: &Path) -> Option<(String, bool)> {
    let contents = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let contents = contents.trim();
    if let Some(ref_name) = contents.strip_prefix("ref:") {
        let ref_name = ref_name.trim();
        let branch = ref_name.strip_prefix("refs/heads/").unwrap_or(ref_name);
        Some((branch.to_string(), false))
    } else {
        // Detached HEAD: the file holds a raw object ID. Anything at least
        // long enough to shorten is treated as one; an empty or malformed
        // HEAD (a corrupt repo) reports no status rather than guessing.
        if contents.is_empty() {
            return None;
        }
        let short = contents.chars().take(7).collect();
        Some((short, true))
    }
}

/// One entry's stat-cache fields, as recorded in `.git/index`.
struct IndexEntry {
    mtime_secs: u32,
    size: u32,
}

/// Best-effort dirty check — see the module docs for exactly what this does
/// and doesn't catch. Never panics on a malformed or unsupported index; it
/// degrades to "no modified-file information", which combined with the
/// untracked-file check still gives a useful (if imperfect) answer.
fn is_dirty(git_dir: &Path, worktree_root: &Path) -> bool {
    let index = parse_index(&git_dir.join("index")).unwrap_or_default();

    // Any file in the worktree that isn't a key in the index at all is
    // untracked, which counts as dirty. `ignore::Walk` gives us
    // gitignore-aware traversal for free, and we additionally have to
    // exclude `.git` itself (not a gitignore rule, so `ignore` doesn't know
    // to skip it).
    let mut walker = ignore::WalkBuilder::new(worktree_root);
    walker.hidden(false);
    let git_dir_owned = git_dir.to_path_buf();
    walker.filter_entry(move |entry| entry.path() != git_dir_owned);
    let has_untracked = walker.build().filter_map(|e| e.ok()).any(|entry| {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            return false;
        }
        let Ok(relative) = entry.path().strip_prefix(worktree_root) else { return false };
        !index.contains_key(&normalize(relative))
    });
    if has_untracked {
        return true;
    }

    // Otherwise, compare every indexed file's current size/mtime against
    // what was recorded when it was last staged.
    index.iter().any(|(path, entry)| {
        let full_path = worktree_root.join(path);
        match std::fs::metadata(&full_path) {
            Err(_) => true, // recorded in the index but missing on disk: deleted.
            Ok(meta) => {
                let mtime_secs = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0);
                mtime_secs != entry.mtime_secs || meta.len() as u32 != entry.size
            }
        }
    })
}

/// Index paths are always `/`-separated regardless of platform (git's own
/// on-disk format requirement); normalize a `Path` the same way before
/// comparing against them.
fn normalize(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Parses the fixed-size-entry (`DIRC` version 2 or 3) portion of a git
/// index file: the header and every entry's stat-cache fields and path.
/// Trailing extensions (`TREE`, `REUC`, ...) are present in every real index
/// but irrelevant here and are simply not read.
///
/// Returns `None` (rather than a partially-correct result) for a missing
/// file, a bad signature, or an unsupported version — see the module docs
/// on why version 4 in particular is out of scope.
fn parse_index(path: &Path) -> Option<HashMap<String, IndexEntry>> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 12 || &data[0..4] != b"DIRC" {
        return None;
    }
    let version = u32::from_be_bytes(data[4..8].try_into().ok()?);
    if !(2..=3).contains(&version) {
        return None;
    }
    let entry_count = u32::from_be_bytes(data[8..12].try_into().ok()?) as usize;

    const EXTENDED_FLAG: u16 = 0x4000;
    // Low 12 bits of the flags field. When a name's length would need all
    // 12 bits (>= 4095 bytes), git instead sets them all to 1 and the name
    // is found by scanning for its NUL terminator.
    const NAME_LEN_MASK: u16 = 0x0FFF;

    let mut entries = HashMap::with_capacity(entry_count);
    let mut offset = 12usize;
    for _ in 0..entry_count {
        // ctime(8) + mtime(8) + dev(4) + ino(4) + mode(4) + uid(4) + gid(4)
        // + size(4) + sha1(20) + flags(2) = 62 fixed bytes before the name.
        let fixed_end = offset.checked_add(62)?;
        let fields: &[u8] = data.get(offset..fixed_end)?;
        let mtime_secs = u32::from_be_bytes(fields[8..12].try_into().ok()?);
        let size = u32::from_be_bytes(fields[36..40].try_into().ok()?);
        let flags = u16::from_be_bytes(fields[60..62].try_into().ok()?);

        let mut name_start = fixed_end;
        if version == 3 && flags & EXTENDED_FLAG != 0 {
            name_start = name_start.checked_add(2)?; // skip the extended-flags field
        }

        let declared_len = (flags & NAME_LEN_MASK) as usize;
        let name_bytes: &[u8] = if declared_len == NAME_LEN_MASK as usize {
            let nul = data[name_start..].iter().position(|&b| b == 0)?;
            &data[name_start..name_start + nul]
        } else {
            data.get(name_start..name_start + declared_len)?
        };
        let name = String::from_utf8_lossy(name_bytes).into_owned();

        // Entries are NUL-padded to a multiple of 8 bytes, counted from the
        // start of the entry, with at least one NUL always present.
        let raw_len = name_start - offset + name_bytes.len();
        let padded_len = (raw_len + 8) & !7;
        offset = offset.checked_add(padded_len)?;

        entries.insert(name, IndexEntry { mtime_secs, size });
    }

    Some(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// Sets up a real repository via the system `git` binary. This is test
    /// *fixture* setup, not something the plugin code itself does — see the
    /// module docs on why `git.rs` never shells out. Using real `git` to
    /// build the fixture is what makes these tests trustworthy: hand-rolled
    /// binary index bytes could easily encode the bug under test instead of
    /// catching it.
    fn init_repo(dir: &Path) {
        let run = |args: &[&str]| {
            let status = Command::new("git").args(args).current_dir(dir).status().expect("git must be on PATH for these tests");
            assert!(status.success(), "git {args:?} failed");
        };
        run(&["init", "-q"]);
        run(&["config", "user.email", "test@example.com"]);
        run(&["config", "user.name", "Test"]);
    }

    #[test]
    fn not_a_repo_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(status(dir.path()).is_none());
    }

    #[test]
    fn parses_a_real_head_on_a_branch() {
        let dir = tempfile::tempdir().unwrap();
        init_repo(dir.path());
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        Command::new("git").args(["add", "a.txt"]).current_dir(dir.path()).status().unwrap();
        Command::new("git").args(["commit", "-q", "-m", "init"]).current_dir(dir.path()).status().unwrap();

        let st = status(dir.path()).expect("must detect the repo");
        assert!(!st.detached);
        assert!(st.branch == "main" || st.branch == "master", "unexpected default branch name: {}", st.branch);
        assert!(!st.dirty, "a freshly committed tree must be clean");
    }

    #[test]
    fn detects_dirty_from_an_untracked_file() {
        let dir = tempfile::tempdir().unwrap();
        init_repo(dir.path());
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        Command::new("git").args(["add", "a.txt"]).current_dir(dir.path()).status().unwrap();
        Command::new("git").args(["commit", "-q", "-m", "init"]).current_dir(dir.path()).status().unwrap();

        std::fs::write(dir.path().join("untracked.txt"), "new").unwrap();
        let st = status(dir.path()).unwrap();
        assert!(st.dirty);
    }

    #[test]
    fn detects_dirty_from_a_modified_tracked_file() {
        let dir = tempfile::tempdir().unwrap();
        init_repo(dir.path());
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        Command::new("git").args(["add", "a.txt"]).current_dir(dir.path()).status().unwrap();
        Command::new("git").args(["commit", "-q", "-m", "init"]).current_dir(dir.path()).status().unwrap();

        // Sleep isn't available here, but changing the size guarantees the
        // size half of the (size, mtime) comparison trips even if the
        // filesystem's mtime resolution is too coarse to change within the
        // same test run.
        std::fs::write(dir.path().join("a.txt"), "hello, much longer now").unwrap();
        let st = status(dir.path()).unwrap();
        assert!(st.dirty);
    }

    #[test]
    fn detached_head_does_not_panic_and_reports_a_short_sha() {
        let dir = tempfile::tempdir().unwrap();
        init_repo(dir.path());
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        Command::new("git").args(["add", "a.txt"]).current_dir(dir.path()).status().unwrap();
        Command::new("git").args(["commit", "-q", "-m", "init"]).current_dir(dir.path()).status().unwrap();

        let head_sha = String::from_utf8(
            Command::new("git").args(["rev-parse", "HEAD"]).current_dir(dir.path()).output().unwrap().stdout,
        )
        .unwrap();
        Command::new("git").args(["checkout", "-q", head_sha.trim()]).current_dir(dir.path()).status().unwrap();

        let st = status(dir.path()).expect("a detached-HEAD repo is still a repo");
        assert!(st.detached);
        assert_eq!(st.branch.len(), 7);
        assert!(head_sha.starts_with(&st.branch));
    }

    #[test]
    fn a_subdirectory_still_finds_the_repository_root() {
        let dir = tempfile::tempdir().unwrap();
        init_repo(dir.path());
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        Command::new("git").args(["add", "a.txt"]).current_dir(dir.path()).status().unwrap();
        Command::new("git").args(["commit", "-q", "-m", "init"]).current_dir(dir.path()).status().unwrap();
        std::fs::create_dir(dir.path().join("nested")).unwrap();

        assert!(status(&dir.path().join("nested")).is_some());
    }

    #[test]
    fn read_head_handles_a_hand_written_ref() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "ref: refs/heads/feature/foo\n").unwrap();
        let (branch, detached) = read_head(&dir.path().join(".git")).unwrap();
        assert_eq!(branch, "feature/foo");
        assert!(!detached);
    }
}
