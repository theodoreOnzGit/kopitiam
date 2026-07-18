//! Project weight — the cheap proxy for how much RAM rust-analyzer will want.
//!
//! `temp_ai_design.md` §6: rust-analyzer's peak RSS is dominated by **indexing
//! the whole dependency graph**, not by your own source. So the crate count in
//! the resolved graph predicts its footprint better than source size does. Both
//! signals here are cheap and, crucially, **deterministic given `Cargo.lock` +
//! the source tree** — so the resulting decision is a function of project state,
//! not of timing.
//!
//! # The cheapness contract (this is the whole point)
//!
//! The probe must never do the expensive thing it is trying to avoid. So:
//!
//! - **Dependency count** defaults to a **one-file line-scan of `Cargo.lock`**
//!   ([`count_deps_via_lock`]) — no subprocess, no TOML parse. A more accurate
//!   `cargo metadata` counter ([`count_deps_via_metadata`]) exists but is
//!   **opt-in**, because it shells out to `cargo`, which on a cold Android tablet
//!   is *not* cheap (the very thing we are guarding against). Accuracy is not the
//!   point; not crossing the cliff is.
//! - **Source size** is a **stat-only** walk ([`ProjectWeight::src_mb`] via
//!   [`estimate_project_weight`]): sum file *metadata* lengths, **never open a
//!   file**. `O(files)`, no content reads.

use std::path::{Path, PathBuf};

/// A cheap estimate of a workspace's weight — the input that drives
/// rust-analyzer's memory footprint. Built by [`estimate_project_weight`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectWeight {
    /// Resolved dependency-graph crate count — the **primary** predictor. By
    /// default this is the number of `[[package]]` entries in the workspace
    /// `Cargo.lock` (includes the workspace's own members; it is a proxy,
    /// exactness is not the point). `0` when no `Cargo.lock` was found.
    pub dep_crates: usize,
    /// Total bytes of first-party `.rs` source under the workspace root, from a
    /// stat-only walk (the `ignore` crate, so `.gitignore` and `target/` are
    /// skipped). The **secondary** term. Bytes, not MB — call
    /// [`ProjectWeight::src_mb`] for the MB the cost model wants.
    pub src_bytes: u64,
}

impl ProjectWeight {
    /// Workspace `.rs` source in **MB** (base-2), as the cost model's
    /// `src_factor` term wants.
    pub fn src_mb(&self) -> f64 {
        self.src_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Walks up from `start` to the nearest ancestor holding a `Cargo.lock`, and
/// returns `(lock_path, workspace_root)`. `None` if none is found up to the
/// filesystem root.
///
/// We key on `Cargo.lock` specifically (not `Cargo.toml`), because it lives at
/// the **workspace** root and lists the whole resolved graph — a nested
/// package's `Cargo.toml` would undercount the real dependency footprint RA
/// analyses. The directory holding it is the right place to both count deps and
/// walk `.rs` source.
fn find_cargo_lock(start: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut dir: &Path = start;
    loop {
        let candidate = dir.join("Cargo.lock");
        if candidate.is_file() {
            return Some((candidate, dir.to_path_buf()));
        }
        dir = dir.parent()?;
    }
}

/// Counts `[[package]]` entries in a `Cargo.lock`'s text — the resolved crate
/// count, our cheap proxy for how much of the dependency graph rust-analyzer will
/// index (and so how much RAM it will want).
///
/// A deliberately dumb line scan, **not** a TOML parse: `Cargo.lock` is a
/// generated file whose `[[package]]` array headers each sit alone on a line, so
/// counting those exact lines is correct in practice and far cheaper than pulling
/// in a TOML parser for one number. A line that merely *mentions* the token
/// (indented, or with trailing text) does not count — only a bare `[[package]]`.
pub fn count_lock_packages(lock_text: &str) -> usize {
    lock_text.lines().filter(|line| line.trim() == "[[package]]").count()
}

/// Cheap default dep count: read the workspace `Cargo.lock` above `dir` and count
/// its `[[package]]` entries. `None` if there is no `Cargo.lock` (not a cargo
/// workspace we can size this way → the caller fails open). One file read, no
/// subprocess.
pub fn count_deps_via_lock(dir: &Path) -> Option<usize> {
    let (lock_path, _root) = find_cargo_lock(dir)?;
    std::fs::read_to_string(&lock_path).ok().map(|t| count_lock_packages(&t))
}

/// Accurate but **not cheap** dep count: run `cargo metadata` and count the
/// resolved packages. Opt-in only — it shells out to `cargo`, which on a cold
/// tablet can take seconds and defeats the whole "cheap probe" purpose. Use it
/// only where you already have `cargo` warm and want the exact resolved closure
/// rather than the `Cargo.lock` line-count proxy.
///
/// `manifest_dir` is any directory inside the workspace; `cargo metadata` finds
/// the workspace root itself. `None` on any failure (no manifest, cargo missing,
/// parse error) → the caller falls back to the cheap path or fails open.
pub fn count_deps_via_metadata(manifest_dir: &Path) -> Option<usize> {
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.current_dir(manifest_dir);
    let meta = cmd.exec().ok()?;
    Some(meta.packages.len())
}

/// Estimate the weight of the workspace enclosing `dir`, cheaply. `None` if there
/// is no `Cargo.lock` above it (not a cargo workspace we can size → the caller
/// fails open with [`crate::Reason::NotApplicable`]).
///
/// Two numbers, both cheap and both deterministic given the tree:
///
/// 1. **Dep crates** — the `Cargo.lock` `[[package]]` count (the cheap default;
///    see [`count_deps_via_lock`]). No `cargo` subprocess.
/// 2. **Source bytes** — a **stat-only** walk of `.rs` under the workspace root
///    with `ignore` (respects `.gitignore`, skips `target/`, so a built tree does
///    not inflate the figure). We sum file *metadata* lengths and **never open a
///    file** — that no-open discipline is the cheapness guarantee.
pub fn estimate_project_weight(dir: &Path) -> Option<ProjectWeight> {
    let (lock_path, root) = find_cargo_lock(dir)?;

    let dep_crates = std::fs::read_to_string(&lock_path)
        .map(|t| count_lock_packages(&t))
        .unwrap_or(0);

    // Sum `.rs` bytes under the workspace root. `ignore` skips `target/` and
    // gitignored paths, so this is first-party source, not build artefacts. We
    // read only `metadata().len()` — never the file contents.
    let mut src_bytes: u64 = 0;
    for entry in ignore::WalkBuilder::new(&root).hidden(false).build().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs") {
            // metadata() can fail on a race; treat that file as 0 bytes rather
            // than aborting the whole walk.
            let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
            src_bytes = src_bytes.saturating_add(len);
        }
    }

    Some(ProjectWeight { dep_crates, src_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_package_count_from_a_cargo_lock() {
        let lock = "\
# This file is automatically @generated by Cargo.\nversion = 4\n\n\
[[package]]\nname = \"a\"\nversion = \"0.1.0\"\n\n\
[[package]]\nname = \"b\"\nversion = \"0.2.0\"\n\n\
[[package]]\nname = \"c\"\nversion = \"0.3.0\"\n";
        assert_eq!(count_lock_packages(lock), 3);
        // A line that merely mentions the token must not count.
        assert_eq!(count_lock_packages("dependencies = [[package]] inline\n"), 0);
        assert_eq!(count_lock_packages(""), 0);
    }

    #[test]
    fn src_mb_converts_bytes_base_two() {
        let w = ProjectWeight { dep_crates: 0, src_bytes: 2 * 1024 * 1024 };
        assert_eq!(w.src_mb(), 2.0);
    }

    #[test]
    fn estimate_reads_a_real_workspace_stat_only() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.lock"),
            "[[package]]\nname=\"a\"\n\n[[package]]\nname=\"b\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "fn a() {}\n").unwrap();
        let w = estimate_project_weight(dir.path()).expect("has a Cargo.lock");
        assert_eq!(w.dep_crates, 2);
        assert!(w.src_bytes >= 10, "counted the lib.rs metadata bytes");
    }

    #[test]
    fn estimate_is_none_without_a_lock() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("loose.rs"), "fn x() {}\n").unwrap();
        assert!(
            estimate_project_weight(dir.path()).is_none(),
            "no Cargo.lock above -> None -> fail open"
        );
    }

    #[test]
    fn count_deps_via_lock_walks_up_to_the_workspace_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.lock"), "[[package]]\nname=\"a\"\n").unwrap();
        let nested = dir.path().join("crates/inner/src");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(count_deps_via_lock(&nested), Some(1), "found the lock up the tree");
    }
}
