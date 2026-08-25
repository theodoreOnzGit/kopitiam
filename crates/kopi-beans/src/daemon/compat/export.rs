//! JSONL export for Go compatibility.
//!
//! Writes the canonical `issues.jsonl` into the daemon's data directory, then
//! mirrors it into each known clone's `.beads/` directory as a **real file**.
//! It used to symlink instead; [`ensure_clone_exports`] explains at length why
//! that had to stop (gh-68).

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::go_schema::{GoIssue, is_bead_blocked};
use crate::daemon::core::{BeadProjection, state::CanonicalState};

/// Context for Go-compatible exports.
///
/// Manages the export directory structure:
/// ```text
/// $XDG_DATA_HOME/beads-rs/exports/{remote_hash}/issues.jsonl
/// ```
#[derive(Clone, Debug)]
pub struct ExportContext {
    /// Base directory for all exports (e.g., ~/.local/share/beads-rs/exports)
    exports_dir: PathBuf,
}

impl ExportContext {
    /// Create a new export context.
    ///
    /// Uses `$XDG_DATA_HOME/beads-rs/exports` or `~/.local/share/beads-rs/exports`.
    pub fn new() -> io::Result<Self> {
        let data_dir = crate::daemon::paths::data_dir().join("exports");
        fs::create_dir_all(&data_dir)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&data_dir, fs::Permissions::from_mode(0o700));
        }

        Ok(Self {
            exports_dir: data_dir,
        })
    }

    /// Create an export context with a custom base directory (for testing).
    pub fn with_dir(exports_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&exports_dir)?;
        Ok(Self { exports_dir })
    }

    /// Get the export path for a given remote URL.
    pub fn export_path(&self, remote_url: &str) -> PathBuf {
        let hash = hash_remote(remote_url);
        self.exports_dir.join(&hash).join("issues.jsonl")
    }

    /// Ensure the directory for a remote exists.
    fn ensure_remote_dir(&self, remote_url: &str) -> io::Result<PathBuf> {
        let hash = hash_remote(remote_url);
        let dir = self.exports_dir.join(&hash);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}

impl Default for ExportContext {
    fn default() -> Self {
        match Self::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                let fallback = std::env::temp_dir().join("beads-rs").join("exports");
                if let Err(e) = fs::create_dir_all(&fallback) {
                    tracing::warn!(
                        "failed to create export context (primary: {}, fallback: {}): {}",
                        err,
                        fallback.display(),
                        e
                    );
                } else {
                    tracing::warn!(
                        "failed to create export context ({}); using fallback {}",
                        err,
                        fallback.display()
                    );
                }
                Self {
                    exports_dir: fallback,
                }
            }
        }
    }
}

/// Export the state to a JSONL file.
///
/// Returns the path to the exported file.
pub fn export_jsonl(
    state: &CanonicalState,
    ctx: &ExportContext,
    remote_url: &str,
) -> io::Result<PathBuf> {
    let dir = ctx.ensure_remote_dir(remote_url)?;
    let final_path = dir.join("issues.jsonl");
    let temp_path = dir.join(".issues.jsonl.tmp");

    // Collect and sort issues by ID for stable output
    let mut issues: Vec<GoIssue> = Vec::new();

    // Export live beads only - tombstones stay in CRDT layer (tombstones.jsonl)
    // but aren't exported to Go-compat format since:
    // 1. beads-go doesn't export them (deleted = removed from export)
    // 2. beads_viewer (bv) doesn't recognize status="tombstone"
    for (id, _) in state.iter_live() {
        let Some(view) = state.bead_view(id) else {
            continue;
        };
        let deps: Vec<_> = state.deps_from(id);
        let is_blocked = is_bead_blocked(id, state);
        let dep_stamp = state.dep_store().stamp();
        let projection = BeadProjection::from_view(&view);
        issues.push(GoIssue::from_projection(
            &projection,
            &deps,
            is_blocked,
            dep_stamp,
        ));
    }

    // Sort by ID for stable diffs
    issues.sort_by(|a, b| a.id.cmp(&b.id));

    // Write atomically: temp file -> fsync -> rename
    {
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        for issue in &issues {
            serde_json::to_writer(&mut writer, issue)?;
            writeln!(writer)?;
        }

        writer.flush()?;

        // fsync for durability
        let file = writer.into_inner()?;
        file.sync_all()?;
    }

    // Atomic rename
    fs::rename(&temp_path, &final_path)?;

    tracing::debug!("Exported {} issues to {:?}", issues.len(), final_path);

    Ok(final_path)
}

/// Mirror the canonical export into each known clone's `.beads/issues.jsonl`,
/// as a real file.
///
/// The write is atomic (staged alongside the destination, then renamed), and a
/// clone whose copy is already byte-identical is left completely untouched --
/// no rewrite, no mtime bump, nothing for `git status` to notice.
///
/// # Why this copies and does not symlink (gh-68 -- don't "optimise" it back)
///
/// This function used to point `.beads/issues.jsonl` at
/// `$XDG_DATA_HOME/beads-rs/exports/{remote_hash}/issues.jsonl` and, when a
/// real file was already sitting there, rename it aside to `issues.jsonl.bak`
/// first. That is wrong for the way beads is actually used, on three counts:
///
/// * **`.beads/issues.jsonl` is a tracked file in the consuming repository.**
///   Replacing it with a symlink makes every clone commit a link to an
///   absolute path that exists on exactly one machine, under one user's home
///   directory. Nobody else's checkout can resolve it -- and git stores the
///   link, not the content, so the export stops being exported at all.
/// * **It happened silently, on any write command.** A `bn update` that
///   touched export state was enough. The repo went from "real file" to
///   "symlink + stray `.bak`" with no output, so a session would commit the
///   corruption without ever being told. It bit this repository twice
///   (`f7e89cc`, then `ac32a5d`), and the daemon does it asynchronously, so
///   restoring the file by hand loses the race unless the daemon is stopped.
/// * **A symlink is a footgun for every later writer.** Anything that opens
///   the path for writing writes *through* the link into the data directory,
///   so a well-meaning repair can quietly corrupt the canonical export too.
///
/// Copying costs one write of a file that is measured in hundreds of KB, only
/// when the content actually changed. That is nothing next to silently
/// breaking the export for every other clone. Note the non-Unix branch was
/// always a plain copy for this same reason (symlinks need privileges on
/// Windows); this just makes every platform agree.
pub fn ensure_clone_exports(export_path: &Path, known_paths: &HashSet<PathBuf>) -> io::Result<()> {
    if known_paths.is_empty() {
        return Ok(());
    }

    // Read once: every clone gets the same bytes, and comparing against what is
    // already on disk needs them anyway.
    let payload = fs::read(export_path)?;

    for clone_path in known_paths {
        if !clone_path.exists() {
            tracing::debug!("Skipping export mirror for non-existent clone: {:?}", clone_path);
            continue;
        }

        let beads_dir = clone_path.join(".beads");
        let dest = beads_dir.join("issues.jsonl");

        if !beads_dir.exists()
            && let Err(e) = fs::create_dir_all(&beads_dir)
        {
            tracing::warn!("Failed to create .beads dir at {:?}: {}", beads_dir, e);
            continue;
        }

        if dest.is_symlink() {
            // Left over from the old behaviour (or an older kopi-beans still
            // running elsewhere). It must go before we write: renaming onto a
            // symlink replaces the link, but any *other* writer that opened the
            // path first would be writing through it into the data directory.
            if let Err(e) = fs::remove_file(&dest) {
                tracing::warn!("Failed to remove legacy symlink {:?}: {}", dest, e);
                continue;
            }
            tracing::info!("Replaced legacy symlink {:?} with a real file", dest);
        } else if fs::read(&dest).is_ok_and(|current| current == payload) {
            continue; // already current -- leave mtime and git status alone
        }

        // Stage beside the destination so the rename stays within one
        // filesystem and is therefore atomic: a concurrent reader sees either
        // the whole old export or the whole new one, never a half-written file.
        let staged = beads_dir.join(".issues.jsonl.tmp");
        if let Err(e) = fs::write(&staged, &payload) {
            tracing::warn!("Failed to stage export at {:?}: {}", staged, e);
            let _ = fs::remove_file(&staged);
            continue;
        }
        if let Err(e) = fs::rename(&staged, &dest) {
            tracing::warn!("Failed to install export at {:?}: {}", dest, e);
            let _ = fs::remove_file(&staged);
            continue;
        }

        // Clean up after the old behaviour, but only when it is provably safe:
        // a `.bak` holding exactly the bytes we just wrote carries nothing the
        // repo does not now have. One that differs is precisely the case where
        // somebody may still need it, so that one stays.
        let backup = beads_dir.join("issues.jsonl.bak");
        if fs::read(&backup).is_ok_and(|b| b == payload)
            && let Err(e) = fs::remove_file(&backup)
        {
            tracing::debug!("Failed to remove redundant {:?}: {}", backup, e);
        }
    }

    Ok(())
}

/// Former name of [`ensure_clone_exports`], from back when this really did
/// create symlinks. Kept so the rename is not a breaking change for anything
/// outside this crate; it just forwards.
#[deprecated(
    since = "0.1.8",
    note = "renamed to `ensure_clone_exports`: the clone copy is a real file now, not a symlink (gh-68)"
)]
pub fn ensure_symlinks(export_path: &Path, known_paths: &HashSet<PathBuf>) -> io::Result<()> {
    ensure_clone_exports(export_path, known_paths)
}

/// Hash a remote URL to a short, filesystem-safe identifier.
fn hash_remote(remote_url: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(remote_url.as_bytes());
    let result = hasher.finalize();
    // Use first 16 bytes (32 hex chars) for reasonable uniqueness
    hex::encode(&result[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_hash_remote_deterministic() {
        let url = "git@github.com:user/repo.git";
        let h1 = hash_remote(url);
        let h2 = hash_remote(url);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32); // 16 bytes = 32 hex chars
    }

    #[test]
    fn test_hash_remote_different_urls() {
        let h1 = hash_remote("git@github.com:user/repo1.git");
        let h2 = hash_remote("git@github.com:user/repo2.git");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_export_context_path() {
        let ctx = ExportContext {
            exports_dir: PathBuf::from("/tmp/test-exports"),
        };
        let path = ctx.export_path("git@github.com:user/repo.git");
        assert!(path.to_string_lossy().contains("issues.jsonl"));
        assert!(path.to_string_lossy().contains("/tmp/test-exports/"));
    }

    #[test]
    fn test_export_empty_state() {
        let tmp = TempDir::new().unwrap();
        let ctx = ExportContext::with_dir(tmp.path().join("exports")).unwrap();
        let state = CanonicalState::new();

        let path = export_jsonl(&state, &ctx, "git@example.com:test.git").unwrap();
        assert!(path.exists());

        // Empty state should produce empty file
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.is_empty());
    }

    #[test]
    fn test_export_with_beads() {
        use crate::daemon::core::IssueStatus;
        use crate::daemon::core::bead::{Bead, BeadCore, BeadFields};
        use crate::daemon::core::composite::Claim;
        use crate::daemon::core::crdt::Lww;
        use crate::daemon::core::domain::{BeadType, Priority};
        use crate::daemon::core::identity::{ActorId, BeadId};
        use crate::daemon::core::time::{Stamp, WriteStamp};

        let tmp = TempDir::new().unwrap();
        let ctx = ExportContext::with_dir(tmp.path().join("exports")).unwrap();

        let mut state = CanonicalState::new();
        let stamp = Stamp::new(
            WriteStamp::new(1700000000000, 0),
            ActorId::new("test@host").unwrap(),
        );

        let core = BeadCore::new(BeadId::parse("bd-abc").unwrap(), stamp.clone(), None);
        let fields = BeadFields {
            title: Lww::new("Test Issue".to_string(), stamp.clone()),
            description: Lww::new("A description".to_string(), stamp.clone()),
            design: Lww::new(None, stamp.clone()),
            acceptance_criteria: Lww::new(None, stamp.clone()),
            priority: Lww::new(Priority::HIGH, stamp.clone()),
            bead_type: Lww::new(BeadType::Bug, stamp.clone()),
            external_ref: Lww::new(None, stamp.clone()),
            source_repo: Lww::new(None, stamp.clone()),
            estimated_minutes: Lww::new(None, stamp.clone()),
            status: Lww::new(IssueStatus::Todo, stamp.clone()),
            closed_on_branch: Lww::new(None, stamp.clone()),
            claim: Lww::new(Claim::Unclaimed, stamp.clone()),
        };
        state.insert(Bead::new(core, fields)).unwrap();

        let path = export_jsonl(&state, &ctx, "git@example.com:test.git").unwrap();
        let content = fs::read_to_string(&path).unwrap();

        assert!(content.contains("bd-abc"));
        assert!(content.contains("Test Issue"));
        assert!(content.contains("\"status\":\"open\""));
        assert!(content.contains("\"issue_type\":\"bug\""));
        assert!(content.contains("\"priority\":1"));
    }

    #[test]
    fn test_tombstones_not_exported() {
        // Tombstones stay in CRDT layer (tombstones.jsonl) but are NOT exported
        // to Go-compat format - matches beads-go behavior where deleted = gone
        use crate::daemon::core::identity::{ActorId, BeadId};
        use crate::daemon::core::time::{Stamp, WriteStamp};
        use crate::daemon::core::tombstone::Tombstone;

        let tmp = TempDir::new().unwrap();
        let ctx = ExportContext::with_dir(tmp.path().join("exports")).unwrap();

        let mut state = CanonicalState::new();
        let stamp = Stamp::new(
            WriteStamp::new(1700000000000, 0),
            ActorId::new("test@host").unwrap(),
        );

        let tombstone = Tombstone::new(
            BeadId::parse("bd-xyz").unwrap(),
            stamp,
            Some("deleted by user".to_string()),
        );
        state.delete(tombstone);

        let path = export_jsonl(&state, &ctx, "git@example.com:test.git").unwrap();
        let content = fs::read_to_string(&path).unwrap();

        // Tombstones should NOT appear in export
        assert!(content.is_empty(), "tombstones should not be exported");
        assert!(!content.contains("bd-xyz"));
        assert!(!content.contains("tombstone"));
    }

    #[test]
    fn writes_a_real_file_when_none_exists() {
        let tmp = TempDir::new().unwrap();
        let clone_path = tmp.path().join("clone1");
        fs::create_dir_all(&clone_path).unwrap();

        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{\"canonical\": true}").unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(clone_path.clone());

        ensure_clone_exports(&export_file, &known_paths).unwrap();

        let dest = clone_path.join(".beads").join("issues.jsonl");
        assert!(!dest.is_symlink(), "gh-68: the clone copy must never be a symlink");
        assert!(dest.is_file());
        assert_eq!(fs::read_to_string(&dest).unwrap(), "{\"canonical\": true}");
        // Nothing left staged behind.
        assert!(!clone_path.join(".beads").join(".issues.jsonl.tmp").exists());
    }

    #[test]
    fn identical_copy_is_left_completely_untouched() {
        let tmp = TempDir::new().unwrap();
        let clone_path = tmp.path().join("clone1");
        let beads_dir = clone_path.join(".beads");
        fs::create_dir_all(&beads_dir).unwrap();

        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{\"canonical\": true}").unwrap();

        let dest = beads_dir.join("issues.jsonl");
        fs::write(&dest, "{\"canonical\": true}").unwrap();
        let before = fs::metadata(&dest).unwrap().modified().unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(clone_path.clone());

        ensure_clone_exports(&export_file, &known_paths).unwrap();

        // Same bytes in, no rewrite: the mtime must not move, or every export
        // would dirty `git status` in the consuming repo for no reason.
        assert_eq!(fs::metadata(&dest).unwrap().modified().unwrap(), before);
        assert_eq!(fs::read_to_string(&dest).unwrap(), "{\"canonical\": true}");
    }

    #[test]
    fn stale_content_is_refreshed_in_place_without_a_backup() {
        let tmp = TempDir::new().unwrap();
        let clone_path = tmp.path().join("clone1");
        let beads_dir = clone_path.join(".beads");
        fs::create_dir_all(&beads_dir).unwrap();

        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{\"canonical\": true}").unwrap();

        let dest = beads_dir.join("issues.jsonl");
        fs::write(&dest, "{\"stale\": true}").unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(clone_path.clone());

        ensure_clone_exports(&export_file, &known_paths).unwrap();

        assert!(dest.is_file());
        assert!(!dest.is_symlink());
        assert_eq!(fs::read_to_string(&dest).unwrap(), "{\"canonical\": true}");
        // The old behaviour renamed the tracked file aside; nothing should now.
        assert!(
            !beads_dir.join("issues.jsonl.bak").exists(),
            "gh-68: refreshing the export must not leave a stray .bak"
        );
    }

    #[cfg(unix)]
    #[test]
    fn legacy_symlink_is_replaced_by_a_real_file() {
        let tmp = TempDir::new().unwrap();
        let clone_path = tmp.path().join("clone1");
        let beads_dir = clone_path.join(".beads");
        fs::create_dir_all(&beads_dir).unwrap();

        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{\"canonical\": true}").unwrap();

        // Exactly what a pre-fix kopi-beans (or upstream beads-rs) leaves here.
        let dest = beads_dir.join("issues.jsonl");
        std::os::unix::fs::symlink(&export_file, &dest).unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(clone_path.clone());

        ensure_clone_exports(&export_file, &known_paths).unwrap();

        assert!(!dest.is_symlink(), "gh-68: the legacy symlink must be replaced");
        assert!(dest.is_file());
        assert_eq!(fs::read_to_string(&dest).unwrap(), "{\"canonical\": true}");
        // The canonical export is untouched by the swap.
        assert_eq!(fs::read_to_string(&export_file).unwrap(), "{\"canonical\": true}");
    }

    #[test]
    fn redundant_backup_is_cleaned_up_but_a_differing_one_is_kept() {
        let tmp = TempDir::new().unwrap();
        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{\"canonical\": true}").unwrap();

        // Clone A: .bak holds exactly what we are about to write -> redundant.
        let clone_a = tmp.path().join("cloneA");
        let beads_a = clone_a.join(".beads");
        fs::create_dir_all(&beads_a).unwrap();
        fs::write(beads_a.join("issues.jsonl.bak"), "{\"canonical\": true}").unwrap();

        // Clone B: .bak holds something else -> the one case somebody may need.
        let clone_b = tmp.path().join("cloneB");
        let beads_b = clone_b.join(".beads");
        fs::create_dir_all(&beads_b).unwrap();
        fs::write(beads_b.join("issues.jsonl.bak"), "{\"irreplaceable\": true}").unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(clone_a.clone());
        known_paths.insert(clone_b.clone());

        ensure_clone_exports(&export_file, &known_paths).unwrap();

        assert!(!beads_a.join("issues.jsonl.bak").exists());
        assert_eq!(
            fs::read_to_string(beads_b.join("issues.jsonl.bak")).unwrap(),
            "{\"irreplaceable\": true}",
            "a .bak that differs from the export must never be deleted"
        );
    }

    #[test]
    fn test_nonexistent_clone_path_skipped() {
        let tmp = TempDir::new().unwrap();
        let nonexistent_path = tmp.path().join("does_not_exist");

        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{}").unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(nonexistent_path.clone());

        // Should succeed without error
        ensure_clone_exports(&export_file, &known_paths).unwrap();

        // Nothing should be created for a clone that isn't there
        assert!(
            !nonexistent_path
                .join(".beads")
                .join("issues.jsonl")
                .exists()
        );
    }

    #[test]
    fn test_multiple_clones_same_canonical() {
        let tmp = TempDir::new().unwrap();
        let clone1 = tmp.path().join("clone1");
        let clone2 = tmp.path().join("clone2");
        let clone3 = tmp.path().join("clone3");

        fs::create_dir_all(&clone1).unwrap();
        fs::create_dir_all(&clone2).unwrap();
        fs::create_dir_all(&clone3).unwrap();

        let export_file = tmp.path().join("canonical.jsonl");
        fs::write(&export_file, "{\"shared\": true}").unwrap();

        let mut known_paths = HashSet::new();
        known_paths.insert(clone1.clone());
        known_paths.insert(clone2.clone());
        known_paths.insert(clone3.clone());

        ensure_clone_exports(&export_file, &known_paths).unwrap();

        // Every clone gets its own real copy of the same content -- each one
        // has to survive being committed and cloned somewhere else.
        for clone in [&clone1, &clone2, &clone3] {
            let dest = clone.join(".beads").join("issues.jsonl");
            assert!(dest.is_file(), "Expected a real file at {:?}", dest);
            assert!(!dest.is_symlink(), "gh-68: {:?} must not be a symlink", dest);
            assert_eq!(fs::read_to_string(&dest).unwrap(), "{\"shared\": true}");
        }
    }

    #[test]
    fn test_export_sorted_by_id() {
        use crate::daemon::core::IssueStatus;
        use crate::daemon::core::bead::{Bead, BeadCore, BeadFields};
        use crate::daemon::core::composite::Claim;
        use crate::daemon::core::crdt::Lww;
        use crate::daemon::core::domain::{BeadType, Priority};
        use crate::daemon::core::identity::{ActorId, BeadId};
        use crate::daemon::core::time::{Stamp, WriteStamp};

        let tmp = TempDir::new().unwrap();
        let ctx = ExportContext::with_dir(tmp.path().join("exports")).unwrap();

        let mut state = CanonicalState::new();
        let stamp = Stamp::new(
            WriteStamp::new(1700000000000, 0),
            ActorId::new("test@host").unwrap(),
        );

        // Insert beads in non-sorted order
        for id in ["bd-zzz", "bd-aaa", "bd-mmm"] {
            let core = BeadCore::new(BeadId::parse(id).unwrap(), stamp.clone(), None);
            let fields = BeadFields {
                title: Lww::new(format!("Issue {}", id), stamp.clone()),
                description: Lww::new(String::new(), stamp.clone()),
                design: Lww::new(None, stamp.clone()),
                acceptance_criteria: Lww::new(None, stamp.clone()),
                priority: Lww::new(Priority::MEDIUM, stamp.clone()),
                bead_type: Lww::new(BeadType::Task, stamp.clone()),
                external_ref: Lww::new(None, stamp.clone()),
                source_repo: Lww::new(None, stamp.clone()),
                estimated_minutes: Lww::new(None, stamp.clone()),
                status: Lww::new(IssueStatus::Todo, stamp.clone()),
                closed_on_branch: Lww::new(None, stamp.clone()),
                claim: Lww::new(Claim::Unclaimed, stamp.clone()),
            };
            state.insert(Bead::new(core, fields)).unwrap();
        }

        let path = export_jsonl(&state, &ctx, "git@example.com:test.git").unwrap();
        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Should be sorted: aaa, mmm, zzz
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("bd-aaa"));
        assert!(lines[1].contains("bd-mmm"));
        assert!(lines[2].contains("bd-zzz"));
    }
}
