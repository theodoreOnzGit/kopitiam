//! Translation memory: a persistent cache of translations keyed by segment
//! **content hash**, so a revised document only pays to re-translate the
//! segments whose text actually changed (`kopitiam_token_max.md` §13, Task
//! III-4).
//!
//! ## The saving
//!
//! Each translation unit is identified by the FNV-1a hash of its normalized
//! text (a [`SegmentId`]; produced by `kopitiam_document::segments`). Because
//! the id is the content, an unchanged segment keeps its id across revisions
//! and [`TranslationMemory::plan`] classifies it as a **hit** — its
//! translation is reused for free. Only a segment whose text changed gets a
//! new id, misses the cache, and must be re-translated. On an iteratively
//! revised document this is the largest single saving: `reuse_fraction ×
//! per-segment cost` tokens never spent.
//!
//! ## Invalidation is content, never time (§0.4)
//!
//! There is no clock here. A cached translation is valid exactly as long as
//! its segment's id — its content hash — is still asked for. Change the text
//! and the id changes, which *is* the invalidation; leave it and the cache
//! stays fresh indefinitely. `recorded_at` is metadata for humans only, never
//! consulted when deciding a hit.
//!
//! ## Decoupled from the document model (no new dependency)
//!
//! This crate does not depend on `kopitiam-document`. [`plan`] is generic over
//! the [`TmSegment`] trait, which a caller implements over
//! `kopitiam_document::Segment` at the wiring site. The only contract between
//! the two crates is the hash string, which both compute with the identical
//! FNV-1a constants — so no cross-crate dependency edge, and no `Cargo.lock`
//! churn.
//!
//! [`plan`]: TranslationMemory::plan

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use kopitiam_index::Store;
use serde::{Deserialize, Serialize};

/// redb key namespace for the translation memory. Additive and disjoint from
/// `workspace/project_state` (Task I) and `workspace/conclusions` (Task II-5),
/// so this needs no store change and no migration (§10.1).
const TRANSLATION_MEMORY_KEY: &str = "workspace/translation_memory";

/// The stable identifier of a translation segment: the FNV-1a content hash of
/// its normalized text, as lower-case hex.
///
/// This mirrors `kopitiam_document::SegmentId`; the two crates agree on the
/// hex string (the same FNV-1a constants) but stay decoupled, so this crate
/// carries its own thin newtype rather than depending on the document model.
/// A caller bridges the two with [`SegmentId::new`]`(doc_id.as_str())`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SegmentId(String);

impl SegmentId {
    /// Wraps a hash string (typically `kopitiam_document::SegmentId::as_str`).
    pub fn new(id: impl Into<String>) -> Self {
        SegmentId(id.into())
    }

    /// The id as its hex string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for SegmentId {
    fn from(value: String) -> Self {
        SegmentId(value)
    }
}

impl From<&str> for SegmentId {
    fn from(value: &str) -> Self {
        SegmentId(value.to_string())
    }
}

/// A translation recorded once and reusable until its segment's content
/// changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedTranslation {
    /// The source text that was translated. Kept for provenance / auditing —
    /// the id already identifies the content.
    pub source_text: String,
    /// The translation to reuse.
    pub translation: String,
    /// Unix timestamp (seconds) when recorded. Metadata only — **never** used
    /// to decide validity (§0.4).
    pub recorded_at: Option<u64>,
}

/// What the translation memory needs from one document segment to plan a pass:
/// its stable content id and its source text.
///
/// Implemented by the caller over `kopitiam_document::Segment` (see the module
/// docs), which is how this crate consumes segments without depending on the
/// document model.
pub trait TmSegment {
    /// The segment's stable content id (its FNV-1a hash).
    fn tm_id(&self) -> SegmentId;
    /// The segment's translatable source text (only needed for misses).
    fn source_text(&self) -> &str;
}

/// A segment classified as a cache **hit**: its translation is reused at zero
/// re-translation cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TmHit {
    /// Position of this segment in the planned sequence.
    pub index: usize,
    /// The segment's content id (already in the cache).
    pub id: SegmentId,
    /// The cached translation to reuse.
    pub translation: String,
}

/// A segment classified as a cache **miss**: new or changed content that must
/// be translated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TmMiss {
    /// Position of this segment in the planned sequence.
    pub index: usize,
    /// The segment's content id (not yet in the cache).
    pub id: SegmentId,
    /// The source text to translate.
    pub source_text: String,
}

/// The result of planning a translation pass against the memory: which
/// segments are reused and which must be translated, plus the fraction reused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TmPlan {
    /// Segments already in the cache — reused for free.
    pub hits: Vec<TmHit>,
    /// Segments that must be translated (and then [`record`]ed).
    ///
    /// [`record`]: TranslationMemory::record
    pub misses: Vec<TmMiss>,
    /// `hits / total` in `[0.0, 1.0]` (0.0 for an empty document). Multiply by
    /// the per-segment translation cost to get the tokens this revision saves.
    pub reuse_fraction: f64,
}

/// A project's translation memory, backed by the `.kopitiam` redb store.
#[derive(Debug, Clone)]
pub struct TranslationMemory {
    root: PathBuf,
    /// Keyed by the id's hex string. A `BTreeMap` keeps a stable, sorted
    /// on-disk order so the serialized value is deterministic.
    entries: BTreeMap<String, CachedTranslation>,
}

impl TranslationMemory {
    /// Loads the translation memory for `root`'s `.kopitiam` directory, or an
    /// empty memory if none has been recorded yet.
    pub fn load(root: &Path) -> Result<Self> {
        let store = Store::open(root)?;
        let entries = store.get_json(TRANSLATION_MEMORY_KEY)?.unwrap_or_default();
        Ok(Self {
            root: root.to_path_buf(),
            entries,
        })
    }

    /// Returns the cached translation for `id`, if one was recorded.
    pub fn lookup(&self, id: &SegmentId) -> Option<CachedTranslation> {
        self.entries.get(id.as_str()).cloned()
    }

    /// Records the `translation` of `source_text` under `id`, overwriting any
    /// previous entry, and persists the memory. This is the write path a
    /// translation pass calls for each miss it resolves.
    pub fn record(
        &mut self,
        id: SegmentId,
        source_text: impl Into<String>,
        translation: impl Into<String>,
    ) -> Result<()> {
        self.entries.insert(
            id.0,
            CachedTranslation {
                source_text: source_text.into(),
                translation: translation.into(),
                recorded_at: now_unix_seconds(),
            },
        );
        self.save()
    }

    /// Number of cached translations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the memory holds no translations yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Classifies each of `segments` as a cache hit (reuse, zero cost) or miss
    /// (must translate), in order, and reports the reuse fraction.
    ///
    /// This is where the saving is realised: on a revised document only the
    /// segments whose text — and therefore id — changed miss the cache. Every
    /// unchanged segment is a hit and costs nothing to re-translate.
    pub fn plan<S: TmSegment>(&self, segments: &[S]) -> TmPlan {
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            let id = segment.tm_id();
            match self.entries.get(id.as_str()) {
                Some(cached) => hits.push(TmHit {
                    index,
                    id,
                    translation: cached.translation.clone(),
                }),
                None => misses.push(TmMiss {
                    index,
                    id,
                    source_text: segment.source_text().to_string(),
                }),
            }
        }
        let reuse_fraction = if segments.is_empty() {
            0.0
        } else {
            hits.len() as f64 / segments.len() as f64
        };
        TmPlan {
            hits,
            misses,
            reuse_fraction,
        }
    }

    /// Persists the current in-memory map to the `.kopitiam` store.
    fn save(&self) -> Result<()> {
        let store = Store::open(&self.root)?;
        store.put_json(TRANSLATION_MEMORY_KEY, &self.entries)
    }
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
    use crate::content_hash;

    /// A minimal stand-in for `kopitiam_document::Segment`, so these tests
    /// exercise `plan` without a dependency on the document crate. The id is
    /// derived from the text exactly as a real segment's is — the same FNV-1a
    /// the document crate uses — so a changed text yields a changed id, which
    /// is the whole revision story.
    struct TestSegment {
        text: String,
    }

    impl TestSegment {
        fn new(text: &str) -> Self {
            TestSegment {
                text: text.to_string(),
            }
        }
    }

    impl TmSegment for TestSegment {
        fn tm_id(&self) -> SegmentId {
            SegmentId::new(content_hash(self.text.as_bytes()))
        }
        fn source_text(&self) -> &str {
            &self.text
        }
    }

    fn segs(texts: &[&str]) -> Vec<TestSegment> {
        texts.iter().map(|t| TestSegment::new(t)).collect()
    }

    #[test]
    fn record_and_lookup_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let id = SegmentId::new(content_hash(b"digital twin"));
        {
            let mut tm = TranslationMemory::load(root).unwrap();
            assert!(tm.lookup(&id).is_none());
            tm.record(id.clone(), "digital twin", "数字孪生").unwrap();
        }

        // A fresh load (new process) sees the persisted translation.
        let tm = TranslationMemory::load(root).unwrap();
        let cached = tm.lookup(&id).unwrap();
        assert_eq!(cached.source_text, "digital twin");
        assert_eq!(cached.translation, "数字孪生");
        assert_eq!(tm.len(), 1);
    }

    #[test]
    fn a_one_block_revision_re_translates_only_the_changed_segment() {
        // The headline test: record every segment of a document, then revise
        // exactly one and prove the plan reuses all the others.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let original = segs(&[
            "Digital Twins",
            "A digital twin mirrors a physical asset.",
            "sensor feeds and a model",
            "We measured throughput.",
        ]);

        // Translate + record every segment of the first version.
        {
            let mut tm = TranslationMemory::load(root).unwrap();
            for seg in &original {
                tm.record(seg.tm_id(), &seg.text, format!("translated: {}", seg.text))
                    .unwrap();
            }
        }

        // Revise exactly one block's text; every other segment is byte-identical
        // and so keeps its id.
        let revised = segs(&[
            "Digital Twins",
            "A digital twin mirrors a physical asset in real time.", // changed
            "sensor feeds and a model",
            "We measured throughput.",
        ]);

        let tm = TranslationMemory::load(root).unwrap();
        let plan = tm.plan(&revised);

        // Three unchanged segments are hits (zero re-translation cost); only
        // the edited one is a miss.
        assert_eq!(plan.hits.len(), 3);
        assert_eq!(plan.misses.len(), 1);
        assert_eq!(plan.misses[0].index, 1);
        assert_eq!(
            plan.misses[0].source_text,
            "A digital twin mirrors a physical asset in real time."
        );
        // reuse_fraction = (n - 1) / n.
        assert!((plan.reuse_fraction - 3.0 / 4.0).abs() < 1e-9);

        // The hits carry the reused translations — no model call needed.
        assert_eq!(plan.hits[0].translation, "translated: Digital Twins");
        assert!(plan.hits.iter().all(|h| h.index != 1));
    }

    #[test]
    fn every_segment_misses_against_an_empty_memory() {
        let dir = tempfile::tempdir().unwrap();
        let tm = TranslationMemory::load(dir.path()).unwrap();
        let plan = tm.plan(&segs(&["a", "b", "c"]));
        assert_eq!(plan.hits.len(), 0);
        assert_eq!(plan.misses.len(), 3);
        assert_eq!(plan.reuse_fraction, 0.0);
    }

    #[test]
    fn identical_text_reuses_across_positions() {
        // Two segments with identical text share an id, so recording one
        // satisfies the other — the content hash is the whole key.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let repeated = "see Figure 1";

        {
            let mut tm = TranslationMemory::load(root).unwrap();
            tm.record(
                SegmentId::new(content_hash(repeated.as_bytes())),
                repeated,
                "见图 1",
            )
            .unwrap();
        }

        let tm = TranslationMemory::load(root).unwrap();
        let plan = tm.plan(&segs(&[repeated, "other", repeated]));
        assert_eq!(plan.hits.len(), 2);
        assert_eq!(plan.misses.len(), 1);
        assert_eq!(plan.misses[0].source_text, "other");
    }

    #[test]
    fn empty_document_has_zero_reuse_fraction() {
        let dir = tempfile::tempdir().unwrap();
        let tm = TranslationMemory::load(dir.path()).unwrap();
        let plan = tm.plan::<TestSegment>(&[]);
        assert_eq!(plan.reuse_fraction, 0.0);
        assert!(plan.hits.is_empty() && plan.misses.is_empty());
    }

    #[test]
    fn coexists_with_conclusions_and_project_state_keys() {
        use crate::{ConclusionLog, ProjectState};
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("s.rs"), b"x").unwrap();

        // Write all three consumers into the same store.
        let mut state = ProjectState::load(root).unwrap();
        state.set_current_task("translate the paper");
        state.save(root).unwrap();

        let mut log = ConclusionLog::load(root).unwrap();
        log.record("a fact", &["s.rs"]).unwrap();

        let id = SegmentId::new(content_hash(b"hello"));
        {
            let mut tm = TranslationMemory::load(root).unwrap();
            tm.record(id.clone(), "hello", "你好").unwrap();
        }

        // Each key round-trips independently; the others are untouched.
        assert_eq!(
            ProjectState::load(root).unwrap().current_task.as_deref(),
            Some("translate the paper")
        );
        assert_eq!(ConclusionLog::load(root).unwrap().conclusions().len(), 1);
        assert_eq!(
            TranslationMemory::load(root).unwrap().lookup(&id).unwrap().translation,
            "你好"
        );
    }
}
