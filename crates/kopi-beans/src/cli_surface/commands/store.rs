use std::io::Write;

use clap::{Args, Subcommand};

use crate::api::{
    AdminFsckOutput, AdminStoreLockInfoOutput, AdminStoreUnlockOutput, FsckStatus,
    StoreLockMetaOutput,
};
use crate::core::StoreId;

use super::CommandError;
use crate::cli_surface::backend::{CliHostBackend, StoreFsckRequest, StoreUnlockRequest};
use crate::cli_surface::render::print_json;

#[derive(Subcommand, Debug)]
pub enum StoreCmd {
    /// Unlock a store lock file.
    Unlock(StoreUnlockArgs),
    /// Offline WAL verification and repair.
    Fsck(StoreFsckArgs),
    /// Migrate `refs/heads/beads/store` from an older on-disk format to the current one.
    Migrate(StoreMigrateArgs),
}

#[derive(Args, Debug)]
pub struct StoreMigrateArgs {
    /// Show what the migration would do and write nothing. Do this first.
    #[arg(long = "dry-run")]
    pub dry_run: bool,

    /// Migrate the local ref only; do not push. Leaves the shared remote store
    /// (and every other machine reading it) exactly as it is.
    #[arg(long = "no-push")]
    pub no_push: bool,

    /// Proceed even when local and remote store refs have diverged with no
    /// common ancestor. Refused by default, because migrating across a
    /// divergence can drop beads that exist on only one side.
    #[arg(long)]
    pub force: bool,

    /// Retries when a concurrent push beats us to the remote ref.
    #[arg(long = "max-retries", value_name = "N", default_value_t = 3)]
    pub max_retries: usize,
}

#[derive(Args, Debug)]
pub struct StoreUnlockArgs {
    /// Store ID to unlock.
    #[arg(long = "store-id", alias = "id", value_name = "STORE_ID")]
    pub store_id: String,

    /// Force unlock even if a daemon appears to be running.
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct StoreFsckArgs {
    /// Store ID to check.
    #[arg(long = "store-id", alias = "id", value_name = "STORE_ID")]
    pub store_id: String,

    /// Apply safe repairs (prefix salvage truncation, no-prefix quarantine, wal.sqlite rebuild).
    #[arg(long)]
    pub repair: bool,
}

pub fn handle<B>(
    json: bool,
    cmd: StoreCmd,
    repo: Option<&std::path::Path>,
    backend: &B,
) -> std::result::Result<(), B::Error>
where
    B: CliHostBackend,
    B::Error: From<CommandError>,
{
    match cmd {
        StoreCmd::Unlock(args) => handle_unlock(json, args, backend),
        StoreCmd::Fsck(args) => handle_fsck(json, args, backend),
        StoreCmd::Migrate(args) => handle_migrate::<B>(json, args, repo),
    }
}

/// Runs `bn store migrate`.
///
/// # Why this one does NOT go through the daemon backend
///
/// Every other `store` subcommand asks the daemon to act on a loaded store.
/// This one cannot, and the reason is not convenience: **the store failing to
/// load is the exact condition a migration exists to fix.** A store written in
/// an older format makes every command — `list`, `ready`, `status`, even
/// `admin doctor` — fail with `unsupported meta format_version`, so routing the
/// repair through the thing that is broken would make it unreachable precisely
/// when it is needed. `migrate_store_ref_to_v2` operates directly on the git
/// refs for the same reason, so this handler just opens the repository and
/// calls it.
///
/// # Why this was missing
///
/// The migration itself has been here, fully implemented and tested, the whole
/// time — it simply had no CLI surface, so the only supported cure for an old
/// store was unreachable from the command line. This wires it up; it does not
/// change what the migration does.
fn handle_migrate<B>(
    json: bool,
    args: StoreMigrateArgs,
    repo: Option<&std::path::Path>,
) -> std::result::Result<(), B::Error>
where
    B: CliHostBackend,
    B::Error: From<CommandError>,
{
    use crate::git::gix_compat::Repository;

    let start = repo.unwrap_or_else(|| std::path::Path::new("."));
    let repository = Repository::discover(start)
        .map_err(|err| B::Error::from(CommandError::from(crate::git::SyncError::from(err))))?;
    let repo_path = repository.workdir().unwrap_or(start).to_path_buf();

    let outcome = crate::git::migrate_store_ref_to_v2(
        &repository,
        &repo_path,
        args.dry_run,
        args.force,
        args.no_push,
        args.max_retries,
    )
    .map_err(|err| B::Error::from(CommandError::from(err)))?;

    if json {
        return print_json(&render_store_migrate_json(&outcome, args.dry_run))
            .map_err(|err| B::Error::from(CommandError::from(err)));
    }
    write_stdout(&render_store_migrate(&outcome, args.dry_run))
        .map_err(|err| B::Error::from(CommandError::from(err)))?;
    Ok(())
}

fn render_store_migrate_json(
    outcome: &crate::git::MigrateStoreToV2Outcome,
    dry_run: bool,
) -> serde_json::Value {
    serde_json::json!({
        "dry_run": dry_run,
        "from_effective_version": outcome.from_effective_version,
        "to_version": crate::core::StoreMetaVersions::STORE_FORMAT_VERSION,
        "deps_format_before": format!("{:?}", outcome.deps_format_before),
        "converted_deps": outcome.converted_deps,
        "added_notes_file": outcome.added_notes_file,
        "wrote_checksums": outcome.wrote_checksums,
        "commit": outcome.commit_oid.map(|oid| oid.to_string()),
        "push": match outcome.push {
            crate::git::sync::MigratePushDisposition::Pushed => "pushed",
            crate::git::sync::MigratePushDisposition::SkippedNoPush => "skipped_no_push",
            crate::git::sync::MigratePushDisposition::SkippedNoRemote => "skipped_no_remote",
        },
        "warnings": outcome.warnings,
    })
}

pub fn render_store_migrate(
    outcome: &crate::git::MigrateStoreToV2Outcome,
    dry_run: bool,
) -> String {
    let mut out = String::new();
    out.push_str(if dry_run {
        "Store migrate (DRY RUN — nothing was written):\n"
    } else {
        "Store migrate:\n"
    });
    out.push_str(&format!(
        "  format_version: {} -> {}\n",
        outcome.from_effective_version,
        crate::core::StoreMetaVersions::STORE_FORMAT_VERSION
    ));
    out.push_str(&format!(
        "  deps_format_before: {:?}\n",
        outcome.deps_format_before
    ));
    out.push_str(&format!("  converted_deps: {}\n", outcome.converted_deps));
    out.push_str(&format!(
        "  added_notes_file: {}\n",
        outcome.added_notes_file
    ));
    out.push_str(&format!("  wrote_checksums: {}\n", outcome.wrote_checksums));
    match outcome.commit_oid {
        Some(oid) => out.push_str(&format!("  commit: {oid}\n")),
        None => out.push_str("  commit: none (nothing to write)\n"),
    }
    // On a dry run the disposition is always `SkippedNoPush`, because nothing
    // was written to push. Saying "SKIPPED (--no-push)" there would credit a
    // flag the caller may never have passed, and imply a choice was honoured
    // when in fact nothing happened at all.
    out.push_str(&format!(
        "  push: {}\n",
        match (dry_run, outcome.push) {
            (true, _) => "nothing pushed (dry run wrote nothing)",
            (false, crate::git::sync::MigratePushDisposition::Pushed) => "pushed to remote",
            (false, crate::git::sync::MigratePushDisposition::SkippedNoPush) =>
                "SKIPPED (--no-push) — remote and other machines untouched",
            (false, crate::git::sync::MigratePushDisposition::SkippedNoRemote) =>
                "skipped (no remote configured)",
        }
    ));
    if !outcome.warnings.is_empty() {
        out.push_str("  warnings:\n");
        for warning in &outcome.warnings {
            out.push_str(&format!("    - {warning}\n"));
        }
    }
    out
}

fn handle_fsck<B>(json: bool, args: StoreFsckArgs, backend: &B) -> std::result::Result<(), B::Error>
where
    B: CliHostBackend,
    B::Error: From<CommandError>,
{
    let store_id = StoreId::parse_str(&args.store_id)
        .map_err(|err| B::Error::from(CommandError::from(err)))?;
    let output = backend.run_store_fsck(StoreFsckRequest {
        store_id,
        repair: args.repair,
    })?;

    if json {
        return print_json(&output).map_err(|err| B::Error::from(CommandError::from(err)));
    }

    write_stdout(&render_admin_fsck(&output))
        .map_err(|err| B::Error::from(CommandError::from(err)))?;
    Ok(())
}

pub fn render_admin_fsck(output: &AdminFsckOutput) -> String {
    let mut out = String::new();
    out.push_str("Store fsck:\n");
    out.push_str(&format!("  store_id: {}\n", output.store_id));
    out.push_str(&format!("  checked_at_ms: {}\n", output.checked_at_ms));
    out.push_str(&format!(
        "  stats: namespaces={} segments={} records={}\n",
        output.stats.namespaces, output.stats.segments, output.stats.records
    ));
    out.push_str(&format!(
        "  summary: risk={} safe_to_accept_writes={} safe_to_prune_wal={} safe_to_rebuild_index={}\n",
        output.summary.risk,
        output.summary.safe_to_accept_writes,
        output.summary.safe_to_prune_wal,
        output.summary.safe_to_rebuild_index
    ));

    if !output.repairs.is_empty() {
        out.push_str("  repairs:\n");
        for repair in &output.repairs {
            match repair {
                crate::api::FsckRepair::PrefixSalvageTruncate {
                    segment_path,
                    truncate_to_offset,
                    discarded_suffix_bytes,
                    cause,
                } => out.push_str(&format!(
                    "    - prefix_salvage_truncate: segment={} truncate_to_offset={truncate_to_offset} discarded_suffix_bytes={discarded_suffix_bytes} cause={cause}\n",
                    segment_path.display()
                )),
                crate::api::FsckRepair::QuarantineNoValidPrefix {
                    original_segment_path,
                    quarantined_path,
                    cause,
                } => out.push_str(&format!(
                    "    - quarantine_no_valid_prefix: original={} quarantined={} cause={cause}\n",
                    original_segment_path.display(),
                    quarantined_path.display()
                )),
                crate::api::FsckRepair::RebuildIndex { index_path } => out.push_str(&format!(
                    "    - rebuild_index: index={}\n",
                    index_path.display()
                )),
            }
        }
    }

    out.push_str("  checks:\n");
    for check in &output.checks {
        out.push_str(&format!(
            "    - {}: {} (severity={}, issues={})\n",
            check.id,
            check.status,
            check.severity,
            check.evidence.len()
        ));
        if check.status != FsckStatus::Pass {
            for evidence in &check.evidence {
                let path = evidence
                    .path
                    .as_ref()
                    .map(|p| format!(" path={}", p.display()))
                    .unwrap_or_default();
                let namespace = evidence
                    .namespace
                    .as_ref()
                    .map(|ns| format!(" namespace={}", ns.as_str()))
                    .unwrap_or_default();
                let origin = evidence
                    .origin
                    .as_ref()
                    .map(|id| format!(" origin={id}"))
                    .unwrap_or_default();
                let seq = evidence
                    .seq
                    .map(|seq| format!(" seq={seq}"))
                    .unwrap_or_default();
                let offset = evidence
                    .offset
                    .map(|offset| format!(" offset={offset}"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "      * {}: {}{path}{namespace}{origin}{seq}{offset}\n",
                    evidence.code, evidence.message
                ));
            }
            if !check.suggested_actions.is_empty() {
                out.push_str("      actions:\n");
                for action in &check.suggested_actions {
                    out.push_str(&format!("        - {action}\n"));
                }
            }
        }
    }
    out
}

fn handle_unlock<B>(
    json: bool,
    args: StoreUnlockArgs,
    backend: &B,
) -> std::result::Result<(), B::Error>
where
    B: CliHostBackend,
    B::Error: From<CommandError>,
{
    let store_id = StoreId::parse_str(&args.store_id)
        .map_err(|err| B::Error::from(CommandError::from(err)))?;
    let output = backend.run_store_unlock(StoreUnlockRequest {
        store_id,
        force: args.force,
    })?;

    if json {
        return print_json(&output).map_err(|err| B::Error::from(CommandError::from(err)));
    }
    write_stdout(&render_admin_store_unlock(&output))
        .map_err(|err| B::Error::from(CommandError::from(err)))?;
    Ok(())
}

pub fn render_admin_store_unlock(output: &AdminStoreUnlockOutput) -> String {
    let mut out = String::new();
    out.push_str("Store lock:\n");
    out.push_str(&format!("  store_id: {}\n", output.store_id));
    out.push_str(&format!("  lock_path: {}\n", output.lock_path.display()));
    render_lock_meta_output(&mut out, output.meta.as_ref());
    if let Some(daemon_pid) = output.daemon_pid {
        out.push_str(&format!("  daemon_pid: {daemon_pid}\n"));
    }
    out.push_str(&format!("  action: {}\n", output.action));
    out
}

pub fn render_admin_store_lock_info(output: &AdminStoreLockInfoOutput) -> String {
    let mut out = String::new();
    out.push_str("Store lock:\n");
    out.push_str(&format!("  store_id: {}\n", output.store_id));
    out.push_str(&format!("  lock_path: {}\n", output.lock_path.display()));
    render_lock_meta_output(&mut out, output.meta.as_ref());
    out
}

fn render_lock_meta_output(out: &mut String, meta: Option<&StoreLockMetaOutput>) {
    if let Some(meta) = meta {
        out.push_str(&format!("  replica_id: {}\n", meta.replica_id));
        out.push_str(&format!("  pid: {}\n", meta.pid));
        out.push_str(&format!("  started_at_ms: {}\n", meta.started_at_ms));
        out.push_str(&format!("  daemon_version: {}\n", meta.daemon_version));
        if let Some(heartbeat) = meta.last_heartbeat_ms {
            out.push_str(&format!("  last_heartbeat_ms: {heartbeat}\n"));
        } else {
            out.push_str("  last_heartbeat_ms: none\n");
        }
    } else {
        out.push_str("  meta: none\n");
    }
}

fn write_stdout(text: &str) -> crate::cli_surface::Result<()> {
    let mut stdout = std::io::stdout().lock();
    if let Err(err) = writeln!(stdout, "{text}")
        && err.kind() != std::io::ErrorKind::BrokenPipe
    {
        return Err(crate::surface::ipc::IpcError::Transport { source: err });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{
        FsckCheck, FsckCheckId, FsckEvidence, FsckEvidenceCode, FsckRepair, FsckRisk, FsckSeverity,
        FsckStats, FsckSummary,
    };
    use crate::core::{NamespaceId, ReplicaId};
    use std::path::PathBuf;
    use uuid::Uuid;

    fn sample_fsck_output() -> AdminFsckOutput {
        let store_id = StoreId::new(Uuid::from_bytes([9u8; 16]));
        let replica_id = ReplicaId::new(Uuid::from_bytes([7u8; 16]));
        AdminFsckOutput {
            store_id,
            checked_at_ms: 12345,
            stats: FsckStats {
                namespaces: 1,
                segments: 2,
                records: 3,
            },
            summary: FsckSummary {
                risk: FsckRisk::High,
                safe_to_accept_writes: false,
                safe_to_prune_wal: false,
                safe_to_rebuild_index: true,
            },
            repairs: vec![
                FsckRepair::PrefixSalvageTruncate {
                    segment_path: PathBuf::from("/tmp/wal/segment-1"),
                    truncate_to_offset: 128,
                    discarded_suffix_bytes: 24,
                    cause: FsckEvidenceCode::FrameCrcMismatch,
                },
                FsckRepair::QuarantineNoValidPrefix {
                    original_segment_path: PathBuf::from("/tmp/wal/segment-2"),
                    quarantined_path: PathBuf::from("/tmp/wal/quarantine/segment-2"),
                    cause: FsckEvidenceCode::SegmentHeaderInvalid,
                },
                FsckRepair::RebuildIndex {
                    index_path: PathBuf::from("/tmp/index/wal.sqlite"),
                },
            ],
            checks: vec![
                FsckCheck {
                    id: FsckCheckId::SegmentFrames,
                    status: FsckStatus::Fail,
                    severity: FsckSeverity::High,
                    evidence: vec![FsckEvidence {
                        code: FsckEvidenceCode::FrameCrcMismatch,
                        message: "crc mismatch".to_string(),
                        path: Some(PathBuf::from("/tmp/wal/segment-1")),
                        namespace: Some(NamespaceId::core()),
                        origin: Some(replica_id),
                        seq: Some(5),
                        offset: Some(128),
                    }],
                    suggested_actions: vec![
                        "run `bd store fsck --repair` to truncate tail corruption".to_string(),
                    ],
                },
                FsckCheck {
                    id: FsckCheckId::IndexOffsets,
                    status: FsckStatus::Pass,
                    severity: FsckSeverity::Low,
                    evidence: Vec::new(),
                    suggested_actions: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn render_fsck_human_golden() {
        let output = render_admin_fsck(&sample_fsck_output());
        let expected = concat!(
            "Store fsck:\n",
            "  store_id: 09090909-0909-0909-0909-090909090909\n",
            "  checked_at_ms: 12345\n",
            "  stats: namespaces=1 segments=2 records=3\n",
            "  summary: risk=high safe_to_accept_writes=false safe_to_prune_wal=false safe_to_rebuild_index=true\n",
            "  repairs:\n",
            "    - prefix_salvage_truncate: segment=/tmp/wal/segment-1 truncate_to_offset=128 discarded_suffix_bytes=24 cause=frame_crc_mismatch\n",
            "    - quarantine_no_valid_prefix: original=/tmp/wal/segment-2 quarantined=/tmp/wal/quarantine/segment-2 cause=segment_header_invalid\n",
            "    - rebuild_index: index=/tmp/index/wal.sqlite\n",
            "  checks:\n",
            "    - segment_frames: fail (severity=high, issues=1)\n",
            "      * frame_crc_mismatch: crc mismatch path=/tmp/wal/segment-1 namespace=core origin=07070707-0707-0707-0707-070707070707 seq=5 offset=128\n",
            "      actions:\n",
            "        - run `bd store fsck --repair` to truncate tail corruption\n",
            "    - index_offsets: pass (severity=low, issues=0)\n",
        );
        assert_eq!(output, expected);
    }
}
