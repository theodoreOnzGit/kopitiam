//! Background Go-compat export worker.

use std::collections::HashSet;
use std::path::PathBuf;
use std::thread;

use crossbeam::channel::{Receiver, Sender};

use crate::daemon::compat::{ExportContext, ensure_clone_exports, export_jsonl};
use crate::daemon::core::CanonicalState;
use crate::daemon::remote::RemoteUrl;

pub(crate) struct ExportJob {
    pub(crate) remote: RemoteUrl,
    pub(crate) core_state: CanonicalState,
    pub(crate) known_paths: HashSet<PathBuf>,
}

enum ExportCommand {
    Job(Box<ExportJob>),
    Shutdown,
}

pub(crate) struct ExportWorkerHandle {
    tx: Sender<ExportCommand>,
}

impl ExportWorkerHandle {
    pub(crate) fn start(ctx: ExportContext) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        thread::Builder::new()
            .name("beads-export".to_string())
            .spawn(move || run_export_loop(ctx, rx))
            .expect("spawn export worker");
        Self { tx }
    }

    pub(crate) fn enqueue(&self, job: ExportJob) -> Result<(), ()> {
        self.tx
            .send(ExportCommand::Job(Box::new(job)))
            .map_err(|_| ())
    }

    pub(crate) fn shutdown(&self) {
        let _ = self.tx.send(ExportCommand::Shutdown);
    }
}

fn run_export_loop(ctx: ExportContext, rx: Receiver<ExportCommand>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            ExportCommand::Job(job) => {
                let job = *job;
                let export_path = match export_jsonl(&job.core_state, &ctx, job.remote.as_str()) {
                    Ok(path) => path,
                    Err(err) => {
                        tracing::warn!(
                            remote = %job.remote,
                            error = %err,
                            "Go-compat export failed"
                        );
                        continue;
                    }
                };

                if !job.known_paths.is_empty()
                    && let Err(err) = ensure_clone_exports(&export_path, &job.known_paths)
                {
                    tracing::warn!(
                        remote = %job.remote,
                        error = %err,
                        "Go-compat clone export mirror failed"
                    );
                }
            }
            ExportCommand::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn build_state() -> CanonicalState {
        use crate::daemon::core::IssueStatus;
        use crate::daemon::core::bead::{Bead, BeadCore, BeadFields};
        use crate::daemon::core::composite::Claim;
        use crate::daemon::core::crdt::Lww;
        use crate::daemon::core::domain::{BeadType, Priority};
        use crate::daemon::core::identity::{ActorId, BeadId};
        use crate::daemon::core::time::{Stamp, WriteStamp};

        let mut state = CanonicalState::new();
        let stamp = Stamp::new(
            WriteStamp::new(1_700_000_000_000, 0),
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
        state
    }

    #[test]
    fn export_worker_writes_jsonl_and_clone_copy() {
        let temp = TempDir::new().unwrap();
        let exports_dir = temp.path().join("exports");
        let ctx = ExportContext::with_dir(exports_dir).unwrap();
        let worker = ExportWorkerHandle::start(ctx.clone());

        let clone_path = temp.path().join("clone");
        fs::create_dir_all(&clone_path).unwrap();
        let mut known_paths = HashSet::new();
        known_paths.insert(clone_path.clone());

        let remote = RemoteUrl::new("git@example.com:test.git");
        worker
            .enqueue(ExportJob {
                remote: remote.clone(),
                core_state: build_state(),
                known_paths,
            })
            .unwrap();

        let export_path = ctx.export_path(remote.as_str());
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if export_path.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(export_path.exists());
        let contents = fs::read_to_string(&export_path).unwrap();
        assert!(contents.contains("bd-abc"));

        let clone_copy = clone_path.join(".beads").join("issues.jsonl");
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if clone_copy.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(clone_copy.exists());
        // gh-68: the clone's copy is a real file holding the same bytes, never
        // a symlink into the daemon's data directory -- it gets committed.
        assert!(!clone_copy.is_symlink());
        assert!(clone_copy.is_file());
        assert_eq!(
            fs::read_to_string(&clone_copy).unwrap(),
            fs::read_to_string(&export_path).unwrap()
        );

        worker.shutdown();
    }
}
