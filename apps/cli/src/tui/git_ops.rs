//! The pure-Rust `gix` layer behind the TUI's lazygit-style git panel
//! ([`super::git`]).
//!
//! # Why this module exists (thin-client discipline)
//!
//! The view ([`super::git::GitView`]) owns only selection/focus/rendering state;
//! every real git fact or mutation goes through [`GitRepo`] here, which talks to
//! `gitoxide` (`gix`) and hands back plain, owned data structs ([`StatusEntry`],
//! [`CommitInfo`], [`BranchInfo`], [`DiffLine`]). Splitting it this way keeps the
//! view's logic a pure function of those structs — unit-tested with seeded data,
//! no repo, no TTY — while the gix calls live behind one seam.
//!
//! # What the constrained `gix` feature set forces us to build by hand
//!
//! The workspace pins `gix` C-free with only `revision` (which pulls in
//! `index`), `sha1`, and the blocking network client. Crucially **`status`,
//! `blob-diff`, `dirwalk`, and `worktree-mutation` are OFF**. So:
//!
//! * There is no `repo.status()` — [`GitRepo::status`] computes it itself by
//!   comparing the HEAD tree, the index, and the working tree (tracked files
//!   only; untracked discovery needs `dirwalk`/gitignore, so it is omitted and
//!   the panel says so).
//! * There is no `gix` blob differ — [`line_diff`] is a small pure-Rust LCS line
//!   diff.
//! * There is no worktree checkout — branch **checkout is gated** (see
//!   [`GitRepo::create_branch`]'s note); we only create/read refs and move HEAD's
//!   own branch on commit.
//! * Push/pull are gated regardless (gix 0.86 has no high-level push, upstream
//!   #306 / kopitiam #28). Nothing ever shells out to the `git` binary.
//!
//! Staging, discard, and commit *are* implemented, because the enabled `index`
//! feature makes index reads/writes and object writes first-class.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use gix::bstr::ByteSlice;

/// How a path changed, in the git sense (the single letter git's `status`
/// prints).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChangeKind {
    Added,
    Modified,
    Deleted,
}

impl ChangeKind {
    /// The status letter git uses (`A`/`M`/`D`).
    pub fn letter(self) -> char {
        match self {
            ChangeKind::Added => 'A',
            ChangeKind::Modified => 'M',
            ChangeKind::Deleted => 'D',
        }
    }
}

/// One row of the status panel: a tracked path with its staged change (index vs
/// HEAD) and/or its unstaged change (working tree vs index). At least one of the
/// two is `Some` for a row to exist.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StatusEntry {
    pub path: String,
    pub staged: Option<ChangeKind>,
    pub unstaged: Option<ChangeKind>,
}

impl StatusEntry {
    /// The two-column `XY` code git shows: staged column then unstaged column,
    /// a space where there is no change in that column.
    pub fn xy(&self) -> (char, char) {
        (
            self.staged.map(ChangeKind::letter).unwrap_or(' '),
            self.unstaged.map(ChangeKind::letter).unwrap_or(' '),
        )
    }
}

/// One row of the log panel.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CommitInfo {
    pub short_id: String,
    /// Full hex object id, so the view can ask for this commit's changed files.
    pub full_id: String,
    pub summary: String,
    pub author: String,
    pub date: String,
}

/// One row of the branches panel.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BranchInfo {
    pub name: String,
    pub is_head: bool,
    pub short_id: String,
}

/// A single line of a rendered diff.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DiffLineKind {
    Context,
    Add,
    Del,
    /// A non-diff informational line (e.g. "binary file", "file added").
    Meta,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub text: String,
}

impl DiffLine {
    fn new(kind: DiffLineKind, text: impl Into<String>) -> Self {
        Self { kind, text: text.into() }
    }
}

/// Above this many lines on either side we skip the O(n·m) LCS diff and just say
/// so, to keep the TUI responsive on huge files.
const DIFF_LINE_CAP: usize = 4000;

// ---------------------------------------------------------------------------
// Pure helpers (no gix, no fs) — the unit-tested core.
// ---------------------------------------------------------------------------

/// Merge the staged (index-vs-HEAD) and unstaged (worktree-vs-index) change maps
/// into one sorted list of [`StatusEntry`] rows, one per path.
pub fn merge_status(
    staged: BTreeMap<String, ChangeKind>,
    unstaged: BTreeMap<String, ChangeKind>,
) -> Vec<StatusEntry> {
    let mut paths: Vec<String> = staged.keys().chain(unstaged.keys()).cloned().collect();
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .map(|path| StatusEntry {
            staged: staged.get(&path).copied(),
            unstaged: unstaged.get(&path).copied(),
            path,
        })
        .collect()
}

/// A minimal LCS-based line diff. Deterministic and dependency-free (the `gix`
/// blob differ is behind the disabled `blob-diff` feature).
pub fn line_diff(old: &str, new: &str) -> Vec<DiffLine> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();

    if a.len() > DIFF_LINE_CAP || b.len() > DIFF_LINE_CAP {
        return vec![DiffLine::new(
            DiffLineKind::Meta,
            format!("(file too large to diff: {} vs {} lines)", a.len(), b.len()),
        )];
    }
    if a == b {
        return vec![DiffLine::new(DiffLineKind::Meta, "(no textual changes)")];
    }

    let (n, m) = (a.len(), b.len());
    // dp[i][j] = length of the LCS of a[i..] and b[j..].
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if a[i] == b[j] {
            out.push(DiffLine::new(DiffLineKind::Context, a[i]));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            out.push(DiffLine::new(DiffLineKind::Del, a[i]));
            i += 1;
        } else {
            out.push(DiffLine::new(DiffLineKind::Add, b[j]));
            j += 1;
        }
    }
    while i < n {
        out.push(DiffLine::new(DiffLineKind::Del, a[i]));
        i += 1;
    }
    while j < m {
        out.push(DiffLine::new(DiffLineKind::Add, b[j]));
        j += 1;
    }
    out
}

/// Format a Unix timestamp (seconds since epoch, UTC) as `YYYY-MM-DD`, without a
/// date crate. Uses Howard Hinnant's civil-from-days algorithm.
pub fn fmt_unix_date(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

// ---------------------------------------------------------------------------
// The gix-backed repository handle.
// ---------------------------------------------------------------------------

/// A handle to the CWD's git repository, exposing exactly the LOCAL operations
/// the panel needs. Opened once; each read re-reads the index/refs so a refresh
/// reflects on-disk reality.
pub struct GitRepo {
    repo: gix::Repository,
}

impl GitRepo {
    /// Discover the repository containing `dir` (walking upward, like git). Errors
    /// cleanly when `dir` is not inside a repo so the view can show a friendly
    /// "not a git repo" message.
    pub fn discover(dir: &Path) -> Result<Self> {
        let repo = gix::discover(dir).context("not a git repository (or any parent)")?;
        Ok(Self { repo })
    }

    /// The short name of the branch HEAD points at (e.g. `main`), or `None` when
    /// detached.
    pub fn head_branch(&self) -> Option<String> {
        let name = self.repo.head_name().ok()??;
        Some(short_ref(name.as_bstr().to_str_lossy().as_ref()))
    }

    // ---- status ----------------------------------------------------------

    /// Compute the working status by hand: index-vs-HEAD gives staged changes,
    /// worktree-vs-index gives unstaged changes (tracked files only). Untracked
    /// files are not listed (needs the disabled `dirwalk`/gitignore support).
    pub fn status(&self) -> Result<Vec<StatusEntry>> {
        let head = self.head_tree_map().unwrap_or_default();
        let index = self.index_map()?;

        let mut staged: BTreeMap<String, ChangeKind> = BTreeMap::new();
        for (path, oid) in &index {
            match head.get(path) {
                None => {
                    staged.insert(path.clone(), ChangeKind::Added);
                }
                Some(head_oid) if head_oid != oid => {
                    staged.insert(path.clone(), ChangeKind::Modified);
                }
                Some(_) => {}
            }
        }
        for path in head.keys() {
            if !index.contains_key(path) {
                staged.insert(path.clone(), ChangeKind::Deleted);
            }
        }

        let mut unstaged: BTreeMap<String, ChangeKind> = BTreeMap::new();
        if let Some(workdir) = self.repo.workdir() {
            // Iterate the index ENTRIES (not the path->oid map) so each carries its
            // recorded stat cache. Like `git status`, we trust that cache: when the
            // working file's size + mtime still match the index, the content is
            // assumed unchanged and is NEVER read or hashed. Only a stat mismatch
            // falls back to reading + hashing (and still confirms via the oid, so a
            // merely `touch`ed file is not falsely reported Modified). Without this,
            // opening the panel on a large repo read + SHA-1'd every tracked file
            // on the render thread -- the startup lag.
            let idx = self.repo.open_index()?;
            for entry in idx.entries() {
                if entry.stage() != gix::index::entry::Stage::Unconflicted {
                    continue;
                }
                let path = entry.path(&idx).to_str_lossy().into_owned();
                let full = workdir.join(&path);
                let meta = match std::fs::metadata(&full) {
                    Err(_) => {
                        unstaged.insert(path, ChangeKind::Deleted);
                        continue;
                    }
                    Ok(meta) => meta,
                };
                if stat_unchanged(&meta, &entry.stat) {
                    continue; // fast path: no read, no hash
                }
                match std::fs::read(&full) {
                    Err(_) => {
                        unstaged.insert(path, ChangeKind::Deleted);
                    }
                    Ok(bytes) => {
                        if blob_oid(&bytes)? != entry.id {
                            unstaged.insert(path, ChangeKind::Modified);
                        }
                    }
                }
            }
        }

        Ok(merge_status(staged, unstaged))
    }

    // ---- log -------------------------------------------------------------

    /// The most recent `limit` commits reachable from HEAD, newest first.
    pub fn log(&self, limit: usize) -> Result<Vec<CommitInfo>> {
        let Ok(head) = self.repo.head_id() else {
            return Ok(Vec::new()); // unborn branch: no commits yet.
        };
        let mut out = Vec::new();
        for info in self.repo.rev_walk(Some(head.detach())).all()? {
            if out.len() >= limit {
                break;
            }
            let info = info?;
            let commit = self.repo.find_commit(info.id)?;
            let author = commit.author().ok();
            let name = author.as_ref().map(|a| a.name.to_str_lossy().into_owned()).unwrap_or_default();
            let secs = commit.time().map(|t| t.seconds).unwrap_or(0);
            let summary = commit
                .message()
                .ok()
                .map(|m| m.summary().to_str_lossy().into_owned())
                .unwrap_or_default();
            let full = info.id.to_hex().to_string();
            out.push(CommitInfo {
                short_id: short_hex(&full),
                full_id: full,
                summary,
                author: name,
                date: fmt_unix_date(secs),
            });
        }
        Ok(out)
    }

    // ---- branches --------------------------------------------------------

    /// All local branches (`refs/heads/*`), the current one flagged.
    pub fn branches(&self) -> Result<Vec<BranchInfo>> {
        let head = self.head_branch();
        let platform = self.repo.references()?;
        let mut out = Vec::new();
        for reference in platform.prefixed("refs/heads/")? {
            let mut reference = reference.map_err(|e| anyhow::anyhow!("{e}"))?;
            let full = reference.name().as_bstr().to_str_lossy().into_owned();
            let name = short_ref(&full);
            let short_id = reference
                .peel_to_id()
                .ok()
                .map(|id| short_hex(&id.to_hex().to_string()))
                .unwrap_or_default();
            let is_head = head.as_deref() == Some(name.as_str());
            out.push(BranchInfo { name, is_head, short_id });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// The files a commit changed relative to its first parent, as staged-style
    /// rows (root commits diff against the empty tree). This is the structural
    /// (added/modified/deleted) view; per-line commit diffs need the disabled
    /// `blob-diff` feature.
    pub fn commit_changed(&self, full_id_hex: &str) -> Result<Vec<StatusEntry>> {
        let oid = gix::ObjectId::from_hex(full_id_hex.as_bytes()).context("bad commit id")?;
        let commit = self.repo.find_commit(oid)?;
        let new = self.tree_map(commit.tree_id()?.detach()).unwrap_or_default();
        let old = commit
            .parent_ids()
            .next()
            .and_then(|p| self.repo.find_commit(p.detach()).ok())
            .and_then(|p| p.tree_id().ok())
            .and_then(|t| self.tree_map(t.detach()))
            .unwrap_or_default();

        let mut changes: BTreeMap<String, ChangeKind> = BTreeMap::new();
        for (path, oid) in &new {
            match old.get(path) {
                None => {
                    changes.insert(path.clone(), ChangeKind::Added);
                }
                Some(o) if o != oid => {
                    changes.insert(path.clone(), ChangeKind::Modified);
                }
                Some(_) => {}
            }
        }
        for path in old.keys() {
            if !new.contains_key(path) {
                changes.insert(path.clone(), ChangeKind::Deleted);
            }
        }
        Ok(merge_status(changes, BTreeMap::new()))
    }

    // ---- diff ------------------------------------------------------------

    /// Render a diff for a status row: its unstaged change (index → worktree) if
    /// any, else its staged change (HEAD → index).
    pub fn file_diff(&self, entry: &StatusEntry) -> Result<Vec<DiffLine>> {
        let (old, new, label) = if entry.unstaged.is_some() {
            (self.index_blob(&entry.path)?, self.worktree_blob(&entry.path), "unstaged")
        } else {
            (self.head_blob(&entry.path)?, self.index_blob(&entry.path)?, "staged")
        };
        Ok(diff_or_meta(old, new, label))
    }

    // ---- mutations -------------------------------------------------------

    /// Stage a path: add its current worktree content to the index (or stage its
    /// deletion when the file is gone). Writes the blob object then rewrites the
    /// index — no worktree mutation involved.
    pub fn stage(&self, path: &str) -> Result<()> {
        let workdir = self.repo.workdir().context("cannot stage in a bare repo")?;
        let mut index = self.open_or_empty_index();
        let path_bytes = path.as_bytes();

        match std::fs::read(workdir.join(path)) {
            Ok(bytes) => {
                let oid = self.repo.write_blob(&bytes)?.detach();
                let stage = gix::index::entry::Stage::Unconflicted;
                if let Some(existing) =
                    index.entry_mut_by_path_and_stage(path_bytes.as_bstr(), stage)
                {
                    existing.id = oid;
                    existing.mode = gix::index::entry::Mode::FILE;
                    existing.stat = gix::index::entry::Stat::default();
                } else {
                    index.dangerously_push_entry(
                        gix::index::entry::Stat::default(),
                        oid,
                        gix::index::entry::Flags::empty(),
                        gix::index::entry::Mode::FILE,
                        path_bytes.as_bstr(),
                    );
                    index.sort_entries();
                }
            }
            Err(_) => {
                // File removed on disk → stage the deletion.
                index.remove_entries(|_, p, _| p == path_bytes.as_bstr());
            }
        }
        self.write_index(&mut index)
    }

    /// Unstage a path: reset its index entry to the HEAD version (or drop it from
    /// the index when it is not in HEAD, i.e. an added file).
    pub fn unstage(&self, path: &str) -> Result<()> {
        let mut index = self.open_or_empty_index();
        let path_bytes = path.as_bytes();
        let head = self.head_tree_map().unwrap_or_default();

        match head.get(path) {
            Some(oid) => {
                let stage = gix::index::entry::Stage::Unconflicted;
                if let Some(existing) =
                    index.entry_mut_by_path_and_stage(path_bytes.as_bstr(), stage)
                {
                    existing.id = *oid;
                    existing.mode = gix::index::entry::Mode::FILE;
                    existing.stat = gix::index::entry::Stat::default();
                } else {
                    index.dangerously_push_entry(
                        gix::index::entry::Stat::default(),
                        *oid,
                        gix::index::entry::Flags::empty(),
                        gix::index::entry::Mode::FILE,
                        path_bytes.as_bstr(),
                    );
                    index.sort_entries();
                }
            }
            None => {
                index.remove_entries(|_, p, _| p == path_bytes.as_bstr());
            }
        }
        self.write_index(&mut index)
    }

    /// Discard a path's *unstaged* changes by rewriting the working-tree file
    /// from its staged (index) content. Only touches that one file via `std::fs`
    /// (no `gix` worktree-mutation feature needed). Destructive — the view guards
    /// it behind a confirm.
    pub fn discard(&self, path: &str) -> Result<()> {
        let workdir = self.repo.workdir().context("cannot discard in a bare repo")?;
        let content = self.index_blob(path)?;
        match content {
            Some(bytes) => {
                let full = workdir.join(path);
                if let Some(parent) = full.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::write(&full, bytes)
                    .with_context(|| format!("rewriting {path} from the index"))?;
                Ok(())
            }
            None => bail!("'{path}' is not tracked in the index; nothing to discard"),
        }
    }

    /// Create a new branch at the current HEAD commit. Does NOT check it out —
    /// worktree checkout needs the disabled `worktree-mutation` feature, so
    /// switching branches is gated (see the module docs).
    pub fn create_branch(&self, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            bail!("branch name is empty");
        }
        let head = self.repo.head_id().context("no commit to branch from yet")?.detach();
        let full = format!("refs/heads/{name}");
        if self.repo.find_reference(full.as_str()).is_ok() {
            bail!("branch '{name}' already exists");
        }
        self.repo
            .reference(full.as_str(), head, gix::refs::transaction::PreviousValue::MustNotExist, format!("branch: created {name}"))
            .with_context(|| format!("creating branch '{name}'"))?;
        Ok(())
    }

    /// Commit the current index as a new commit on HEAD's branch, and move that
    /// branch to it. Author/committer come from git config, falling back to a
    /// local identity. Returns the new commit's short id.
    pub fn commit(&self, message: &str) -> Result<String> {
        let message = message.trim();
        if message.is_empty() {
            bail!("empty commit message");
        }
        let tree_id = self.write_index_tree()?;

        let parents: Vec<gix::ObjectId> = self.repo.head_id().ok().map(|id| id.detach()).into_iter().collect();

        let sig = self.signature();
        let mut author_buf = gix::date::parse::TimeBuf::default();
        let mut committer_buf = gix::date::parse::TimeBuf::default();
        let author_ref = sig.to_ref(&mut author_buf);
        let committer_ref = sig.to_ref(&mut committer_buf);

        let commit = self
            .repo
            .new_commit_as(committer_ref, author_ref, message, tree_id, parents)?;
        let commit_id = commit.id().detach();

        // Move the branch HEAD points at (creating it for an unborn branch).
        if let Ok(Some(name)) = self.repo.head_name() {
            self.repo.reference(
                name.as_bstr().to_str_lossy().as_ref(),
                commit_id,
                gix::refs::transaction::PreviousValue::Any,
                message.to_owned(),
            )?;
        }
        Ok(short_hex(&commit_id.to_hex().to_string()))
    }

    // ---- internals -------------------------------------------------------

    /// The committer/author identity from git config, or a local default so a
    /// commit never fails purely for want of a configured name.
    fn signature(&self) -> gix::actor::Signature {
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let time = gix::date::Time::new(seconds, 0);
        if let Some(Ok(committer)) = self.repo.committer() {
            gix::actor::Signature {
                name: committer.name.to_owned(),
                email: committer.email.to_owned(),
                time,
            }
        } else {
            gix::actor::Signature {
                name: "kopitiam".into(),
                email: "kopitiam@localhost".into(),
                time,
            }
        }
    }

    /// Build a tree object from the current index and write it to the odb,
    /// returning its id. Hand-rolled because the index→tree helper lives behind
    /// the disabled `status`/`dirwalk` machinery.
    fn write_index_tree(&self) -> Result<gix::ObjectId> {
        let index = self.open_or_empty_index();
        let mut entries: Vec<(String, gix::ObjectId)> = Vec::new();
        for entry in index.entries() {
            if entry.stage() != gix::index::entry::Stage::Unconflicted {
                continue;
            }
            let path = entry.path(&index).to_str_lossy().into_owned();
            entries.push((path, entry.id));
        }
        self.write_subtree(&entries, "")
    }

    /// Recursively write the subtree rooted at `prefix` (a `""`-or-`"dir/"`
    /// path prefix) from the flat `(path, blob_oid)` list, returning the subtree
    /// id. Entries are grouped by their next path component.
    fn write_subtree(&self, entries: &[(String, gix::ObjectId)], prefix: &str) -> Result<gix::ObjectId> {
        use gix::objs::tree::{Entry, EntryKind};

        // Group children: direct blobs here, plus subdir → its entries.
        let mut blobs: Vec<Entry> = Vec::new();
        let mut subdirs: BTreeMap<String, Vec<(String, gix::ObjectId)>> = BTreeMap::new();

        for (path, oid) in entries {
            let rest = &path[prefix.len()..];
            match rest.split_once('/') {
                None => blobs.push(Entry {
                    mode: EntryKind::Blob.into(),
                    filename: rest.into(),
                    oid: *oid,
                }),
                Some((dir, _)) => {
                    subdirs.entry(dir.to_owned()).or_default().push((path.clone(), *oid));
                }
            }
        }

        let mut tree_entries = blobs;
        for (dir, child_entries) in subdirs {
            let child_prefix = format!("{prefix}{dir}/");
            let subtree_id = self.write_subtree(&child_entries, &child_prefix)?;
            tree_entries.push(Entry {
                mode: EntryKind::Tree.into(),
                filename: dir.into(),
                oid: subtree_id,
            });
        }
        tree_entries.sort();
        let tree = gix::objs::Tree { entries: tree_entries };
        Ok(self.repo.write_object(&tree)?.detach())
    }

    /// Open the index for mutation, synthesising an empty one when the repo has
    /// no index file yet (a brand-new repo), so the first stage can create it.
    fn open_or_empty_index(&self) -> gix::index::File {
        self.repo.open_index().unwrap_or_else(|_| {
            gix::index::File::from_state(
                gix::index::State::new(self.repo.object_hash()),
                self.repo.index_path(),
            )
        })
    }

    /// Rewrite the on-disk index. Drops the cached tree extension first (its
    /// paths may now be stale) per gix's own guidance.
    fn write_index(&self, index: &mut gix::index::File) -> Result<()> {
        index.remove_tree();
        index.write(gix::index::write::Options::default())?;
        Ok(())
    }

    /// Map of `path -> blob oid` for every blob in the HEAD tree. `None` when the
    /// branch is unborn (no HEAD commit yet).
    fn head_tree_map(&self) -> Option<BTreeMap<String, gix::ObjectId>> {
        let tree_id = self.repo.head_tree_id().ok()?.detach();
        self.tree_map(tree_id)
    }

    /// Map of `path -> blob oid` for every blob in an arbitrary tree.
    fn tree_map(&self, tree_id: gix::ObjectId) -> Option<BTreeMap<String, gix::ObjectId>> {
        let tree = self.repo.find_tree(tree_id).ok()?;
        let mut map = BTreeMap::new();
        self.collect_tree(&tree, "", &mut map).ok()?;
        Some(map)
    }

    fn collect_tree(
        &self,
        tree: &gix::Tree<'_>,
        prefix: &str,
        out: &mut BTreeMap<String, gix::ObjectId>,
    ) -> Result<()> {
        let decoded = tree.decode()?;
        for entry in decoded.entries.iter() {
            let name = entry.filename.to_str_lossy();
            if entry.mode.is_tree() {
                let subtree = self.repo.find_tree(entry.oid.to_owned())?;
                let child_prefix = format!("{prefix}{name}/");
                self.collect_tree(&subtree, &child_prefix, out)?;
            } else {
                out.insert(format!("{prefix}{name}"), entry.oid.to_owned());
            }
        }
        Ok(())
    }

    /// `path -> blob oid` for every stage-0 index entry.
    fn index_map(&self) -> Result<BTreeMap<String, gix::ObjectId>> {
        let index = self.repo.open_index()?;
        let mut map = BTreeMap::new();
        for entry in index.entries() {
            if entry.stage() != gix::index::entry::Stage::Unconflicted {
                continue;
            }
            map.insert(entry.path(&index).to_str_lossy().into_owned(), entry.id);
        }
        Ok(map)
    }

    /// Bytes of the HEAD version of `path`, if present.
    fn head_blob(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let map = self.head_tree_map().unwrap_or_default();
        self.blob_bytes(map.get(path).copied())
    }

    /// Bytes of the index (staged) version of `path`, if present.
    fn index_blob(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let map = self.index_map()?;
        self.blob_bytes(map.get(path).copied())
    }

    /// Bytes of the working-tree version of `path`, if the file exists.
    fn worktree_blob(&self, path: &str) -> Option<Vec<u8>> {
        let workdir = self.repo.workdir()?;
        std::fs::read(workdir.join(path)).ok()
    }

    fn blob_bytes(&self, oid: Option<gix::ObjectId>) -> Result<Option<Vec<u8>>> {
        match oid {
            Some(oid) => Ok(Some(self.repo.find_object(oid)?.data.clone())),
            None => Ok(None),
        }
    }
}

/// Compute the git blob object id of `bytes` WITHOUT writing it to the odb
/// (`gix_object::compute_hash`), so status checks never pollute the object store.
fn blob_oid(bytes: &[u8]) -> Result<gix::ObjectId> {
    gix::objs::compute_hash(gix::hash::Kind::Sha1, gix::objs::Kind::Blob, bytes)
        .context("hashing worktree file")
}

/// True when a working file's size + mtime still match the index entry's recorded
/// stat, so its content is (racily) assumed unchanged and need not be read or
/// hashed — the same shortcut `git status` takes via the index stat cache. A
/// mismatch is not conclusive (the caller confirms by hashing); a match is
/// treated as clean, which is what makes opening the panel cheap on a large repo.
fn stat_unchanged(meta: &std::fs::Metadata, stat: &gix::index::entry::Stat) -> bool {
    if meta.len() != u64::from(stat.size) {
        return false;
    }
    let Ok(mtime) = meta.modified() else { return false };
    let Ok(dur) = mtime.duration_since(std::time::UNIX_EPOCH) else { return false };
    if dur.as_secs() as u32 != stat.mtime.secs {
        return false;
    }
    // Sub-second precision only when the index actually recorded nanoseconds
    // (git built without USE_NSEC stores 0 there); don't let a 0 index nsec vs a
    // nonzero filesystem nsec spuriously force a re-hash.
    stat.mtime.nsecs == 0 || dur.subsec_nanos() == stat.mtime.nsecs
}

/// Turn an optional old/new blob pair into diff lines, annotating pure
/// additions/deletions and binary content.
fn diff_or_meta(old: Option<Vec<u8>>, new: Option<Vec<u8>>, label: &str) -> Vec<DiffLine> {
    match (old, new) {
        (None, None) => vec![DiffLine::new(DiffLineKind::Meta, format!("(no {label} content)"))],
        (None, Some(new)) => match String::from_utf8(new) {
            Ok(text) => {
                let mut lines = vec![DiffLine::new(DiffLineKind::Meta, "(new file)")];
                lines.extend(text.lines().map(|l| DiffLine::new(DiffLineKind::Add, l)));
                lines
            }
            Err(_) => vec![DiffLine::new(DiffLineKind::Meta, "(binary file added)")],
        },
        (Some(old), None) => match String::from_utf8(old) {
            Ok(text) => {
                let mut lines = vec![DiffLine::new(DiffLineKind::Meta, "(deleted file)")];
                lines.extend(text.lines().map(|l| DiffLine::new(DiffLineKind::Del, l)));
                lines
            }
            Err(_) => vec![DiffLine::new(DiffLineKind::Meta, "(binary file deleted)")],
        },
        (Some(old), Some(new)) => match (String::from_utf8(old), String::from_utf8(new)) {
            (Ok(old), Ok(new)) => line_diff(&old, &new),
            _ => vec![DiffLine::new(DiffLineKind::Meta, "(binary file changed)")],
        },
    }
}

/// `refs/heads/foo` → `foo`; anything else is returned as-is.
fn short_ref(full: &str) -> String {
    full.strip_prefix("refs/heads/").unwrap_or(full).to_string()
}

/// First 7 chars of a hex object id.
fn short_hex(hex: &str) -> String {
    hex.chars().take(7).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(diff: &[DiffLine]) -> Vec<(DiffLineKind, &str)> {
        diff.iter().map(|l| (l.kind, l.text.as_str())).collect()
    }

    #[test]
    fn merge_status_unions_and_sorts_paths() {
        let mut staged = BTreeMap::new();
        staged.insert("b.rs".to_string(), ChangeKind::Added);
        staged.insert("a.rs".to_string(), ChangeKind::Modified);
        let mut unstaged = BTreeMap::new();
        unstaged.insert("a.rs".to_string(), ChangeKind::Modified);
        unstaged.insert("c.rs".to_string(), ChangeKind::Deleted);

        let rows = merge_status(staged, unstaged);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].path, "a.rs");
        assert_eq!(rows[0].staged, Some(ChangeKind::Modified));
        assert_eq!(rows[0].unstaged, Some(ChangeKind::Modified));
        assert_eq!(rows[1].path, "b.rs");
        assert_eq!(rows[1].staged, Some(ChangeKind::Added));
        assert_eq!(rows[1].unstaged, None);
        assert_eq!(rows[2].path, "c.rs");
        assert_eq!(rows[2].unstaged, Some(ChangeKind::Deleted));
    }

    #[test]
    fn xy_codes_match_git_columns() {
        let e = StatusEntry {
            path: "x".into(),
            staged: Some(ChangeKind::Added),
            unstaged: Some(ChangeKind::Modified),
        };
        assert_eq!(e.xy(), ('A', 'M'));
        let e = StatusEntry { path: "x".into(), staged: None, unstaged: Some(ChangeKind::Deleted) };
        assert_eq!(e.xy(), (' ', 'D'));
        let e = StatusEntry { path: "x".into(), staged: Some(ChangeKind::Modified), unstaged: None };
        assert_eq!(e.xy(), ('M', ' '));
    }

    #[test]
    fn line_diff_reports_add_del_context() {
        let diff = line_diff("a\nb\nc\n", "a\nB\nc\n");
        assert_eq!(
            kinds(&diff),
            vec![
                (DiffLineKind::Context, "a"),
                (DiffLineKind::Del, "b"),
                (DiffLineKind::Add, "B"),
                (DiffLineKind::Context, "c"),
            ]
        );
    }

    #[test]
    fn line_diff_pure_insertions_and_deletions() {
        let ins = line_diff("a\nc\n", "a\nb\nc\n");
        assert_eq!(
            kinds(&ins),
            vec![
                (DiffLineKind::Context, "a"),
                (DiffLineKind::Add, "b"),
                (DiffLineKind::Context, "c"),
            ]
        );
        let del = line_diff("a\nb\nc\n", "a\nc\n");
        assert_eq!(
            kinds(&del),
            vec![
                (DiffLineKind::Context, "a"),
                (DiffLineKind::Del, "b"),
                (DiffLineKind::Context, "c"),
            ]
        );
    }

    #[test]
    fn line_diff_identical_is_meta() {
        let diff = line_diff("a\nb\n", "a\nb\n");
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0].kind, DiffLineKind::Meta);
    }

    #[test]
    fn diff_or_meta_flags_new_and_deleted_and_binary() {
        let new = diff_or_meta(None, Some(b"x\ny\n".to_vec()), "staged");
        assert_eq!(new[0].kind, DiffLineKind::Meta);
        assert!(new.iter().any(|l| l.kind == DiffLineKind::Add && l.text == "x"));

        let del = diff_or_meta(Some(b"x\n".to_vec()), None, "staged");
        assert_eq!(del[0].kind, DiffLineKind::Meta);
        assert!(del.iter().any(|l| l.kind == DiffLineKind::Del && l.text == "x"));

        let bin = diff_or_meta(None, Some(vec![0xff, 0xfe, 0x00]), "staged");
        assert_eq!(bin.len(), 1);
        assert_eq!(bin[0].kind, DiffLineKind::Meta);
    }

    #[test]
    fn fmt_unix_date_matches_known_epochs() {
        assert_eq!(fmt_unix_date(0), "1970-01-01");
        assert_eq!(fmt_unix_date(1_700_000_000), "2023-11-14");
    }

    #[test]
    fn short_ref_and_short_hex() {
        assert_eq!(short_ref("refs/heads/main"), "main");
        assert_eq!(short_ref("HEAD"), "HEAD");
        assert_eq!(short_hex("0123456789abcdef"), "0123456");
    }

    // ---- gix round-trip tests against a throwaway repo -------------------
    //
    // These exercise the real mutation plumbing (stage / unstage / commit /
    // create_branch) against a `gix`-init'd tempdir — no network, no TTY. They
    // are the automated counterpart to the human smoke test of the interactive
    // panel, verifying the fiddly index-write and tree-build paths are correct.

    fn init_repo(dir: &std::path::Path) -> GitRepo {
        gix::init(dir).expect("gix init");
        GitRepo::discover(dir).expect("discover")
    }

    #[test]
    fn stage_then_commit_shows_up_in_head_tree_and_log() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("a.txt"), "one\n").unwrap();
        let g = init_repo(root);

        // Untracked files are not listed, but stage() works on any path.
        g.stage("a.txt").unwrap();
        let status = g.status().unwrap();
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].path, "a.txt");
        assert_eq!(status[0].staged, Some(ChangeKind::Added));
        assert_eq!(status[0].unstaged, None);

        let short = g.commit("first commit").unwrap();
        assert!(!short.is_empty());

        // Reopen and confirm the commit and its file are real.
        let g2 = GitRepo::discover(root).unwrap();
        let log = g2.log(10).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].summary, "first commit");
        assert!(g2.status().unwrap().is_empty(), "clean after commit");

        let changed = g2.commit_changed(&log[0].full_id).unwrap();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].path, "a.txt");
        assert_eq!(changed[0].staged, Some(ChangeKind::Added));
    }

    #[test]
    fn modify_stage_and_unstage_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("f.txt"), "a\nb\n").unwrap();
        let g = init_repo(root);
        g.stage("f.txt").unwrap();
        g.commit("base").unwrap();

        // Modify the file → shows as unstaged Modified.
        std::fs::write(root.join("f.txt"), "a\nB\nc\n").unwrap();
        let status = g.status().unwrap();
        assert_eq!(status[0].unstaged, Some(ChangeKind::Modified));
        assert_eq!(status[0].staged, None);

        // The diff of the unstaged change reflects the edit.
        let diff = g.file_diff(&status[0]).unwrap();
        assert!(diff.iter().any(|l| l.kind == DiffLineKind::Add && l.text == "B"));

        // Stage it → Modified moves to the staged column.
        g.stage("f.txt").unwrap();
        let status = g.status().unwrap();
        assert_eq!(status[0].staged, Some(ChangeKind::Modified));
        assert_eq!(status[0].unstaged, None);

        // Unstage it → back to unstaged Modified.
        g.unstage("f.txt").unwrap();
        let status = g.status().unwrap();
        assert_eq!(status[0].staged, None);
        assert_eq!(status[0].unstaged, Some(ChangeKind::Modified));
    }

    #[test]
    fn discard_restores_worktree_from_index() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("d.txt"), "keep\n").unwrap();
        let g = init_repo(root);
        g.stage("d.txt").unwrap();
        g.commit("c").unwrap();

        std::fs::write(root.join("d.txt"), "trash\n").unwrap();
        g.discard("d.txt").unwrap();
        assert_eq!(std::fs::read_to_string(root.join("d.txt")).unwrap(), "keep\n");
        assert!(g.status().unwrap().is_empty());
    }

    #[test]
    fn create_branch_appears_in_branch_list() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("x"), "x").unwrap();
        let g = init_repo(root);
        g.stage("x").unwrap();
        g.commit("c").unwrap();

        g.create_branch("feature").unwrap();
        let names: Vec<String> = g.branches().unwrap().into_iter().map(|b| b.name).collect();
        assert!(names.iter().any(|n| n == "feature"));
        // Creating the same branch twice is an error, not a silent overwrite.
        assert!(g.create_branch("feature").is_err());
    }

    #[test]
    fn subdirectory_paths_commit_into_nested_trees() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src/inner")).unwrap();
        std::fs::write(root.join("src/inner/deep.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("top.rs"), "//\n").unwrap();
        let g = init_repo(root);
        g.stage("src/inner/deep.rs").unwrap();
        g.stage("top.rs").unwrap();
        let log_id = {
            g.commit("nested").unwrap();
            GitRepo::discover(root).unwrap().log(1).unwrap()[0].full_id.clone()
        };
        let g2 = GitRepo::discover(root).unwrap();
        let changed: Vec<String> =
            g2.commit_changed(&log_id).unwrap().into_iter().map(|e| e.path).collect();
        assert!(changed.iter().any(|p| p == "src/inner/deep.rs"));
        assert!(changed.iter().any(|p| p == "top.rs"));
    }
}
