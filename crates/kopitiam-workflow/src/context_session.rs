//! The Context Session — a progressive/anytime context-assembly actor
//! (temp_ai_design.md §4), built as a **sibling of AID-0028**.
//!
//! # Reused pattern, not a new one (AID-0028)
//!
//! `docs/ai-decisions/AID-0028` shipped the async LSP session as a
//! **single-owner background actor**: `spawn_async` returns immediately with a
//! handle in `Connecting`, a dedicated background thread owns the work outright,
//! flips a shared `AtomicU8` to `Ready`/`Failed`, and the foreground never
//! blocks. §4 says in as many words: *that* is the template for the Context
//! Session — "make it a sibling actor; do not invent a second concurrency
//! model." So this reuses AID-0028's shape: `std::thread` + a shared atomic
//! state, **no** tokio/async-std (same Pure-Core reason AID-0028 rejected an
//! async runtime).
//!
//! # One deliberate divergence from AID-0028
//!
//! AID-0028 decided pre-ready requests **reject** (`NotReady`), because a
//! partial LSP connection cannot answer correctly. The Context Session does the
//! **opposite on purpose**: a pre-ready [`ContextSession::snapshot`] returns the
//! *partial* context accumulated so far. That is the "anytime" contract of §4 —
//! "usable immediately; refined continuously." A partial *context* is valid
//! (fewer facts, still the top-priority ones); a partial LSP *answer* is not.
//! Same actor skeleton, opposite pre-ready policy, and the reason is the
//! difference in what "partial" means for each.
//!
//! # STATUS: scaffold — the actor *shape*, not full streaming
//!
//! This fills the buffer from a pre-computed priority order in one background
//! pass and marks `Ready`. It does **not** yet implement: the mpsc **job
//! channel** (AID-0028's `Box<dyn FnOnce + Send>` jobs) that would let a
//! mid-reasoning tool-use request **jump the priority queue** (§4 "tool-use
//! steers the stream"), nor the live re-ranking that goes with it. Those are
//! the finished-feature work; the atomic-state + owned-worker + non-blocking
//! snapshot boundary here is exactly where they slot in.
//!
//! # Determinism is preserved (§4 Refinement 2)
//!
//! The worker consumes a **pre-ordered** `Vec<Entity>` (the
//! `ContextBuilder::prioritized_candidates` order) and only ever pushes a
//! prefix of it. So even though a background thread does the pushing, the final
//! `Ready` snapshot under a given budget is identical every run — the concurrency
//! is throughput, never a source of nondeterminism. `context = f(task, budget)`
//! survives being assembled off-thread.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use kopitiam_ontology::Entity;

use crate::budget::ResourceBudget;

/// Where a [`ContextSession`] is in its life, mirroring AID-0028's
/// `Connecting`/`Ready`/`Failed` state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// The worker is still filling the buffer. `snapshot` returns a partial,
    /// still-usable context (the anytime contract).
    Building,
    /// The worker has streamed the full budgeted prefix. `snapshot` is complete.
    Ready,
    /// The worker panicked/aborted before finishing. `snapshot` returns
    /// whatever it managed before failing.
    Failed,
}

impl SessionState {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => SessionState::Building,
            1 => SessionState::Ready,
            _ => SessionState::Failed,
        }
    }
    fn as_u8(self) -> u8 {
        match self {
            SessionState::Building => 0,
            SessionState::Ready => 1,
            SessionState::Failed => 2,
        }
    }
}

/// A single-owner background context-assembly actor (AID-0028 sibling).
///
/// `spawn` returns immediately; a background thread owns the assembly and
/// streams the budgeted prefix into a shared buffer. The foreground calls
/// [`Self::snapshot`] whenever it likes — pre-ready it gets a partial context,
/// post-ready the full budgeted one — and **never blocks** on the worker.
pub struct ContextSession {
    state: Arc<AtomicU8>,
    /// Progressively filled, in priority order. Behind a `Mutex` so the
    /// foreground can snapshot mid-stream without racing the worker.
    buffer: Arc<Mutex<Vec<Entity>>>,
    worker: Option<JoinHandle<()>>,
}

impl ContextSession {
    /// Spawn a session that streams the first `budget.fact_allowance()` of
    /// `ordered` (a priority-ordered candidate list, e.g. from
    /// `ContextBuilder::prioritized_candidates`) into the buffer on a
    /// background thread. Returns straight away in [`SessionState::Building`].
    ///
    /// `ordered` is moved into the worker so the foreground shares nothing
    /// mutable with it except the buffer and the state atomic — the
    /// single-owner discipline AID-0028 established.
    pub fn spawn(ordered: Vec<Entity>, budget: &dyn ResourceBudget) -> Self {
        let allowance = budget.fact_allowance();
        let state = Arc::new(AtomicU8::new(SessionState::Building.as_u8()));
        let buffer = Arc::new(Mutex::new(Vec::new()));

        let worker_state = Arc::clone(&state);
        let worker_buffer = Arc::clone(&buffer);
        let worker = std::thread::spawn(move || {
            // Stream the budgeted prefix one fact at a time, so a foreground
            // snapshot mid-pass sees a growing prefix (the "anytime" refinement
            // continuously). Only ever the prefix — never past the budget.
            for entity in ordered.into_iter().take(allowance) {
                worker_buffer.lock().expect("context buffer mutex poisoned").push(entity);
            }
            worker_state.store(SessionState::Ready.as_u8(), Ordering::Release);
        });

        Self { state, buffer, worker: Some(worker) }
    }

    /// Current state. Cheap, non-blocking — poll it from a UI idle tick the way
    /// AID-0028's caller polls `is_ready`.
    pub fn state(&self) -> SessionState {
        SessionState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// True once the full budgeted prefix has streamed in.
    pub fn is_ready(&self) -> bool {
        self.state() == SessionState::Ready
    }

    /// A copy of the facts accumulated **so far** — a valid partial context
    /// pre-ready, the complete budgeted one post-ready. Never blocks on the
    /// worker (only briefly on the buffer mutex).
    pub fn snapshot(&self) -> Vec<Entity> {
        self.buffer.lock().expect("context buffer mutex poisoned").clone()
    }

    /// Test/convenience helper: block until the worker has finished, then
    /// return the final snapshot. Real callers poll [`Self::is_ready`] from an
    /// idle tick instead of blocking — this exists so a synchronous caller (and
    /// the tests) can get the settled result deterministically.
    pub fn join(mut self) -> Vec<Entity> {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        self.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{FactBudget, ResourceBudget};
    use kopitiam_ontology::EntityKind;

    fn facts(n: usize) -> Vec<Entity> {
        (0..n).map(|i| Entity::new(EntityKind::Fact, format!("fact-{i}"), "test")).collect()
    }

    #[test]
    fn streams_the_budgeted_prefix_and_reaches_ready() {
        let session = ContextSession::spawn(facts(10), &FactBudget(4));
        let settled = session.join();
        assert_eq!(settled.len(), 4, "only the budgeted prefix is streamed");
        assert_eq!(settled[0].name, "fact-0", "priority order is preserved off-thread");
    }

    #[test]
    fn ready_snapshot_is_deterministic_given_budget() {
        // Same ordered input + same budget -> identical settled snapshot every
        // run, even though a background thread assembled it. Concurrency is
        // throughput, not nondeterminism (§4 Refinement 2). The input is built
        // ONCE and cloned into each run — otherwise `Entity::new`'s fresh UUIDs
        // would make the *input* differ, which is not what we're testing.
        let input = facts(8);
        let run = || ContextSession::spawn(input.clone(), &FactBudget(5)).join();
        let a = run();
        let b = run();
        let names: Vec<&str> = a.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["fact-0", "fact-1", "fact-2", "fact-3", "fact-4"]);
        assert_eq!(a, b, "budget fixes the prefix; the clock does not");
    }

    #[test]
    fn a_tighter_budget_settles_to_a_prefix_of_a_looser_one() {
        let input = facts(8);
        let small = ContextSession::spawn(input.clone(), &FactBudget(2)).join();
        let large = ContextSession::spawn(input.clone(), &FactBudget(6)).join();
        assert_eq!(small, large[..2], "tighter budget is a strict prefix, via the actor too");
    }

    #[test]
    fn unbounded_budget_streams_everything() {
        let settled = ContextSession::spawn(facts(3), &FactBudget::UNBOUNDED).join();
        assert_eq!(settled.len(), 3);
        assert_eq!(FactBudget::UNBOUNDED.fact_allowance(), usize::MAX);
    }
}
