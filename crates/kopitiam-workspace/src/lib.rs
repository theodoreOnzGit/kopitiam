//! Project State: short-lived-per-session working memory, persisted
//! through `kopitiam-index`'s `.kopitiam` directory.
//!
//! Distinct from `kopitiam-knowledge`'s semantic graph (facts about the
//! project) and from `bd` (long-lived task tracking) — this crate only
//! remembers what a session was recently focused on, so the next session
//! (or a different interface) can resume.

mod conclusion;
mod digest;
mod state;
mod translation_memory;

pub use conclusion::{
    Conclusion, ConclusionLog, SourceDrift, SourceHash, StaleConclusion, content_hash,
};
pub use digest::{
    ArchitectureDigest, CrateDigest, build_digest, run_cargo_metadata, source_hash,
};
pub use state::{ProjectState, WORKING_SET_CAPACITY};
pub use translation_memory::{
    CachedTranslation, SegmentId, TmHit, TmMiss, TmPlan, TmSegment, TranslationMemory,
};
