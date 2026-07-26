//! Persistent *conclusions*: durable facts an agent derived once, so the
//! next session does not pay to re-derive them (`kopitiam_token_max.md`
//! §258, "pay once for understanding"; Task II-5).
//!
//! A [`Conclusion`] is a short human/agent-written fact ("this crate does
//! X", "this invariant holds", "this test is flaky") tagged with the
//! content hashes of every file it was derived from. That provenance is the
//! whole point: a conclusion is trustworthy only as long as the sources it
//! rests on are unchanged.
//!
//! ## Staleness is content, never time (§0.4)
//!
//! We deliberately do **not** expire conclusions on a clock. A stale
//! conclusion an agent *trusts* is worse than no conclusion at all, and a
//! fresh one is still fresh a year later if its sources never moved.
//! [`ConclusionLog::stale_conclusions`] recomputes the current hash of each
//! source and reports a conclusion as stale iff any source's content drifted
//! (or the source is gone). The `recorded_at` timestamp is metadata for
//! humans only — it is never consulted when deciding staleness.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kopitiam_index::Store;
use serde::{Deserialize, Serialize};

/// redb key namespace for the conclusion log. Distinct from
/// `workspace/project_state`, so this is an *additive* key — no migration of
/// existing state is required (§10.1).
const CONCLUSIONS_KEY: &str = "workspace/conclusions";

/// The content hash of one source file a [`Conclusion`] was derived from,
/// captured at the moment the conclusion was recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceHash {
    /// Path to the source file, interpreted relative to the project root
    /// (the same `root` passed to [`ConclusionLog::load`]). Absolute paths
    /// are honored as-is.
    pub path: PathBuf,

    /// Content hash of the file's bytes at derivation time. See
    /// [`content_hash`] for the algorithm.
    pub hash: String,
}

/// A durable fact an agent derived, with the provenance needed to know when
/// it can no longer be trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conclusion {
    /// The conclusion itself, in prose ("kopitiam-index has no C deps").
    pub text: String,

    /// Every source file this conclusion was derived from, each pinned to
    /// the content hash it had at derivation time. Staleness is decided
    /// purely by comparing these against current file hashes.
    pub derived_from: Vec<SourceHash>,

    /// Unix timestamp (seconds) when the conclusion was recorded. Metadata
    /// only — **never** used to decide staleness (§0.4).
    pub recorded_at: Option<u64>,
}

/// How a single source drifted away from what a [`Conclusion`] pinned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceDrift {
    /// The file still exists but its content hash no longer matches.
    Changed {
        path: PathBuf,
        recorded_hash: String,
        current_hash: String,
    },
    /// The file the conclusion was derived from is gone (or unreadable).
    Missing { path: PathBuf },
}

impl SourceDrift {
    /// The source path this drift concerns, for display.
    pub fn path(&self) -> &Path {
        match self {
            SourceDrift::Changed { path, .. } => path,
            SourceDrift::Missing { path } => path,
        }
    }
}

/// A conclusion that can no longer be trusted, paired with exactly which of
/// its sources drifted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleConclusion {
    /// The stale conclusion.
    pub conclusion: Conclusion,
    /// The subset of `conclusion.derived_from` that drifted (non-empty).
    pub drifted: Vec<SourceDrift>,
}

/// The project's log of recorded [`Conclusion`]s, backed by the `.kopitiam`
/// redb store. Holds the project `root` so it can recompute current source
/// hashes when asked for staleness.
#[derive(Debug, Clone)]
pub struct ConclusionLog {
    root: PathBuf,
    conclusions: Vec<Conclusion>,
}

impl ConclusionLog {
    /// Loads the conclusion log for `root`'s `.kopitiam` directory, or an
    /// empty log if none has been recorded yet.
    pub fn load(root: &Path) -> Result<Self> {
        let store = Store::open(root)?;
        let conclusions = store.get_json(CONCLUSIONS_KEY)?.unwrap_or_default();
        Ok(Self {
            root: root.to_path_buf(),
            conclusions,
        })
    }

    /// All recorded conclusions, oldest first, regardless of staleness.
    pub fn conclusions(&self) -> &[Conclusion] {
        &self.conclusions
    }

    /// Number of currently *live* (non-stale) conclusions — the count a
    /// caller can trust without re-deriving.
    pub fn live_count(&self) -> usize {
        self.conclusions.len() - self.stale_conclusions().len()
    }

    /// Records a new conclusion derived from `sources`, hashing each source
    /// file's current content, appending it to the log, and persisting the
    /// whole log to the `.kopitiam` store.
    ///
    /// This is the **write path** the intended `remember`/`conclude`
    /// subcommand will call (see the crate docs / Task II-5 report). Source
    /// paths are read relative to the log's `root`.
    pub fn record(
        &mut self,
        text: impl Into<String>,
        sources: &[impl AsRef<Path>],
    ) -> Result<&Conclusion> {
        let mut derived_from = Vec::with_capacity(sources.len());
        for source in sources {
            let rel = source.as_ref();
            let abs = self.root.join(rel);
            let bytes = std::fs::read(&abs)
                .with_context(|| format!("hashing source {}", abs.display()))?;
            derived_from.push(SourceHash {
                path: rel.to_path_buf(),
                hash: content_hash(&bytes),
            });
        }

        let conclusion = Conclusion {
            text: text.into(),
            derived_from,
            recorded_at: now_unix_seconds(),
        };
        self.conclusions.push(conclusion);
        self.save()?;
        Ok(self.conclusions.last().expect("just pushed"))
    }

    /// Recomputes the current content hash of every source and returns the
    /// conclusions that can no longer be trusted, each with the exact
    /// sources that drifted. A conclusion with no `derived_from` sources can
    /// never go stale (there is nothing to invalidate it).
    pub fn stale_conclusions(&self) -> Vec<StaleConclusion> {
        let mut stale = Vec::new();
        for conclusion in &self.conclusions {
            let mut drifted = Vec::new();
            for source in &conclusion.derived_from {
                let abs = self.root.join(&source.path);
                match std::fs::read(&abs) {
                    Ok(bytes) => {
                        let current = content_hash(&bytes);
                        if current != source.hash {
                            drifted.push(SourceDrift::Changed {
                                path: source.path.clone(),
                                recorded_hash: source.hash.clone(),
                                current_hash: current,
                            });
                        }
                    }
                    Err(_) => drifted.push(SourceDrift::Missing {
                        path: source.path.clone(),
                    }),
                }
            }
            if !drifted.is_empty() {
                stale.push(StaleConclusion {
                    conclusion: conclusion.clone(),
                    drifted,
                });
            }
        }
        stale
    }

    /// Persists the current in-memory log to the `.kopitiam` store.
    fn save(&self) -> Result<()> {
        let store = Store::open(&self.root)?;
        store.put_json(CONCLUSIONS_KEY, &self.conclusions)
    }
}

/// Deterministic, dependency-free content hash of `bytes` (64-bit FNV-1a,
/// lower-case hex).
///
/// Chosen over `std::hash::DefaultHasher` on purpose: `DefaultHasher`'s
/// SipHash output is explicitly *not* stable across Rust releases, and these
/// hashes are *persisted* and compared across sessions (potentially built
/// with different toolchains). FNV-1a is fixed by its constants, so the same
/// bytes always hash the same everywhere. We stay dependency-free (no `sha2`
/// / `blake3` added — reusing them would force a `Cargo.lock` change that is
/// out of scope for this task). Collision resistance is not required: this
/// only needs to detect that a file *changed*, and if two revisions ever
/// collided the failure mode is a missed staleness signal, no worse than the
/// no-memory baseline.
pub fn content_hash(bytes: &[u8]) -> String {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

fn now_unix_seconds() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn content_hash_is_deterministic_and_sensitive() {
        assert_eq!(content_hash(b"kopitiam"), content_hash(b"kopitiam"));
        assert_ne!(content_hash(b"kopitiam"), content_hash(b"kopitiaN"));
        // Known-answer: pins the algorithm so a refactor can't silently
        // change persisted hashes.
        assert_eq!(content_hash(b""), format!("{:016x}", 0xcbf2_9ce4_8422_2325u64));
    }

    #[test]
    fn fresh_conclusion_is_not_stale() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("source.rs"), b"fn a() {}").unwrap();

        let mut log = ConclusionLog::load(root).unwrap();
        log.record("source.rs defines a()", &["source.rs"]).unwrap();

        assert!(log.stale_conclusions().is_empty());
        assert_eq!(log.live_count(), 1);
    }

    #[test]
    fn conclusion_goes_stale_when_source_content_changes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("source.rs"), b"fn a() {}").unwrap();

        let mut log = ConclusionLog::load(root).unwrap();
        log.record("source.rs defines a()", &["source.rs"]).unwrap();
        assert!(log.stale_conclusions().is_empty());

        // Mutate the file's *content* (not time): the pinned hash no longer
        // matches, so the conclusion must be reported stale.
        fs::write(root.join("source.rs"), b"fn a() {}\nfn b() {}").unwrap();

        let stale = log.stale_conclusions();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].conclusion.text, "source.rs defines a()");
        assert_eq!(stale[0].drifted.len(), 1);
        match &stale[0].drifted[0] {
            SourceDrift::Changed { path, .. } => assert_eq!(path, Path::new("source.rs")),
            other => panic!("expected Changed, got {other:?}"),
        }
        assert_eq!(log.live_count(), 0);
    }

    #[test]
    fn conclusion_goes_stale_when_source_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("gone.rs"), b"data").unwrap();

        let mut log = ConclusionLog::load(root).unwrap();
        log.record("derived from gone.rs", &["gone.rs"]).unwrap();

        fs::remove_file(root.join("gone.rs")).unwrap();

        let stale = log.stale_conclusions();
        assert_eq!(stale.len(), 1);
        assert!(matches!(stale[0].drifted[0], SourceDrift::Missing { .. }));
    }

    #[test]
    fn round_trips_through_the_redb_store() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.rs"), b"one").unwrap();
        fs::write(root.join("b.rs"), b"two").unwrap();

        {
            let mut log = ConclusionLog::load(root).unwrap();
            log.record("first", &["a.rs"]).unwrap();
            log.record("second, two sources", &["a.rs", "b.rs"]).unwrap();
        }

        // A brand-new process (fresh load) sees the persisted conclusions.
        let reloaded = ConclusionLog::load(root).unwrap();
        assert_eq!(reloaded.conclusions().len(), 2);
        assert_eq!(reloaded.conclusions()[0].text, "first");
        assert_eq!(reloaded.conclusions()[1].derived_from.len(), 2);
        assert!(reloaded.stale_conclusions().is_empty());
        assert_eq!(reloaded.live_count(), 2);
    }

    #[test]
    fn does_not_disturb_the_existing_project_state_key() {
        use crate::ProjectState;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("s.rs"), b"x").unwrap();

        let mut state = ProjectState::load(root).unwrap();
        state.set_current_task("do the thing");
        state.save(root).unwrap();

        let mut log = ConclusionLog::load(root).unwrap();
        log.record("a fact", &["s.rs"]).unwrap();

        // Conclusions live under a separate key; ProjectState is untouched.
        let reloaded_state = ProjectState::load(root).unwrap();
        assert_eq!(reloaded_state.current_task.as_deref(), Some("do the thing"));
        assert_eq!(ConclusionLog::load(root).unwrap().conclusions().len(), 1);
    }
}
