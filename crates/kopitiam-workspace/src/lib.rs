//! Project State: short-lived-per-session working memory, persisted
//! through `kopitiam-index`'s `.kopitiam` directory.
//!
//! Distinct from `kopitiam-knowledge`'s semantic graph (facts about the
//! project) and from `bd` (long-lived task tracking) — this crate only
//! remembers what a session was recently focused on, so the next session
//! (or a different interface) can resume.

mod state;

pub use state::{ProjectState, WORKING_SET_CAPACITY};
