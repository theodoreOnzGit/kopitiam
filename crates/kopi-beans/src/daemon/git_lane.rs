//! Git checkpoint lane state for a single store.
//!
//! Provides:
//! - `GitLaneState` - in-memory state for legacy git sync/checkpoint lane

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use crate::daemon::core::{BeadSlug, WriteStamp};
use crate::daemon::env_flags::env_flag_truthy;

const DEFAULT_BACKOFF_BASE_MS: u64 = 500;
const TEST_FAST_BACKOFF_BASE_MS: u64 = 50;

fn backoff_base_ms() -> u64 {
    if env_flag_truthy("BD_TEST_FAST") {
        TEST_FAST_BACKOFF_BASE_MS
    } else {
        DEFAULT_BACKOFF_BASE_MS
    }
}

#[derive(Clone, Debug)]
pub struct FetchErrorRecord {
    pub message: String,
    pub wall_ms: u64,
}

#[derive(Clone, Debug)]
pub struct DivergenceRecord {
    pub local_oid: String,
    pub remote_oid: String,
    pub wall_ms: u64,
}

#[derive(Clone, Debug)]
pub struct ForcePushRecord {
    pub previous_remote_oid: String,
    pub remote_oid: String,
    pub wall_ms: u64,
}

#[derive(Clone, Debug)]
pub struct ClockSkewRecord {
    pub delta_ms: i64,
    pub wall_ms: u64,
}

/// What the last failed sync attempt actually said.
///
/// Kept around so a parked `SyncWait` waiter can be woken and TOLD why, instead
/// of sitting there forever waiting for a clean state that a failing remote is
/// never going to produce (kopitiam#25). Without this the only trace of a
/// failure was a `tracing::error!` line inside the daemon log — invisible to the
/// person staring at a hung `bn sync`.
#[derive(Clone, Debug)]
pub struct SyncFailureRecord {
    /// Rendered `SyncError` from the git lane. Keep the whole thing — the
    /// interesting bit (auth, non-fast-forward, DNS) is usually deep in there.
    pub message: String,
    /// Value of [`GitLaneState::consecutive_failures`] AFTER this failure was
    /// counted, i.e. always >= 1.
    pub consecutive_failures: u32,
    /// Wall clock (ms) when the failure got recorded.
    pub wall_ms: u64,
}

/// In-memory state for the legacy git checkpoint lane.
pub struct GitLaneState {
    /// Root slug for bead IDs (from meta.json).
    /// When set, new bead IDs will use this slug (e.g., "myproject-xxx").
    pub root_slug: Option<BeadSlug>,

    /// All known clone paths for this remote.
    pub known_paths: HashSet<PathBuf>,

    /// Whether the state has uncommitted changes.
    pub dirty: bool,

    /// Time of last mutation (for debounce scheduling).
    pub last_mutation: Option<Instant>,

    /// Whether a sync is currently in progress.
    pub sync_in_progress: bool,

    /// Whether a background refresh is currently in progress.
    pub refresh_in_progress: bool,

    /// Time of last successful sync.
    pub last_sync: Option<Instant>,

    /// Wall clock time (ms) of last successful sync - for IPC responses.
    pub last_sync_wall_ms: Option<u64>,

    /// Time of last successful refresh-from-remote.
    pub last_refresh: Option<Instant>,

    /// Whether this lane has been initialized from a successful git load.
    pub loaded_from_git: bool,

    /// Number of consecutive sync failures (for exponential backoff).
    ///
    /// Resets to 0 on a successful sync, so it is NOT safe to use for "did a
    /// new failure happen since I parked?" — use [`Self::sync_failures_total`]
    /// for that.
    pub consecutive_failures: u32,

    /// Monotonic count of every sync failure this lane has ever seen.
    ///
    /// Never reset, not even by a success. That is the whole point: a
    /// `SyncWait` waiter snapshots this number when it parks and gets woken the
    /// moment the lane's number goes past it. If we reused
    /// [`Self::consecutive_failures`] the waiter can get stranded — sync
    /// succeeds (counter → 0) but stays dirty because a mutation landed
    /// mid-sync, then the retry fails (counter → 1), and 1 is not greater than
    /// the 3 the waiter snapshotted. Monotonic counter, no such hole.
    pub sync_failures_total: u64,

    /// Last sync failure, if any. Cleared on a successful sync.
    pub last_sync_error: Option<SyncFailureRecord>,

    /// Last observed write stamp (from state or meta).
    pub last_seen_stamp: Option<WriteStamp>,

    /// Last fetch error (best-effort).
    pub last_fetch_error: Option<FetchErrorRecord>,

    /// Last divergence detected between local and remote.
    pub last_divergence: Option<DivergenceRecord>,

    /// Last detected remote force-push.
    pub last_force_push: Option<ForcePushRecord>,

    /// Last detected clock skew.
    pub last_clock_skew: Option<ClockSkewRecord>,
}

impl GitLaneState {
    /// Create a new GitLaneState.
    pub fn new() -> Self {
        GitLaneState {
            root_slug: None,
            known_paths: HashSet::new(),
            dirty: false,
            last_mutation: None,
            sync_in_progress: false,
            refresh_in_progress: false,
            last_sync: None,
            last_sync_wall_ms: None,
            last_refresh: None,
            loaded_from_git: false,
            consecutive_failures: 0,
            sync_failures_total: 0,
            last_sync_error: None,
            last_seen_stamp: None,
            last_fetch_error: None,
            last_divergence: None,
            last_force_push: None,
            last_clock_skew: None,
        }
    }

    /// Create a new GitLaneState with root slug and initial clone path.
    ///
    /// This only seeds path metadata; callers must explicitly call
    /// `mark_loaded_from_git()` after a successful git load/refresh.
    pub fn with_path(root_slug: Option<BeadSlug>, path: PathBuf) -> Self {
        let mut s = Self::new();
        s.root_slug = root_slug;
        s.known_paths.insert(path);
        s
    }

    /// Register another clone path for this remote.
    pub fn register_path(&mut self, path: PathBuf) {
        self.known_paths.insert(path);
    }

    /// Mark this lane as successfully initialized from git state.
    pub fn mark_loaded_from_git(&mut self) {
        self.loaded_from_git = true;
        self.last_refresh = Some(Instant::now());
    }

    /// Whether this lane has completed at least one successful git load.
    pub fn is_loaded_from_git(&self) -> bool {
        self.loaded_from_git
    }

    /// Pick any existing clone path to run git ops against.
    pub fn any_valid_path(&self) -> Option<&PathBuf> {
        self.known_paths.iter().find(|p| p.exists())
    }

    /// Mark the state as dirty (has uncommitted changes).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.last_mutation = Some(Instant::now());
    }

    /// Check if enough time has passed since last mutation for sync.
    pub fn ready_for_sync(&self, debounce_ms: u64) -> bool {
        if !self.dirty {
            return false;
        }
        if self.sync_in_progress {
            return false;
        }

        match self.last_mutation {
            Some(last) => last.elapsed().as_millis() >= debounce_ms as u128,
            None => true,
        }
    }

    /// Mark sync as started.
    pub fn start_sync(&mut self) {
        self.sync_in_progress = true;
        // Clear dirty to track whether mutations happen *during* this sync.
        // If a mutation occurs while syncing, mark_dirty() will set dirty=true again
        // and we will schedule a follow-up sync after completion.
        self.dirty = false;
    }

    /// Mark sync as completed successfully.
    pub fn complete_sync(&mut self, wall_ms: u64) {
        self.sync_in_progress = false;
        let now = Instant::now();
        self.last_sync = Some(now);
        self.last_sync_wall_ms = Some(wall_ms);
        self.last_refresh = Some(now);
        self.consecutive_failures = 0;
        self.last_sync_error = None;

        // Only clear dirty if no mutations happened during sync
        // (The daemon will re-mark dirty if needed)
    }

    /// Mark sync as failed, and remember WHY.
    ///
    /// `message` is the rendered `SyncError`; `wall_ms` is the daemon clock
    /// reading at the time of failure.
    ///
    /// # The trap this used to be (kopitiam#25)
    /// `dirty = true` here is deliberate — it is what makes the scheduler retry
    /// after [`Self::backoff_ms`]. But `flush_sync_waiters` releases a parked
    /// `bn sync` waiter only when `!dirty && !sync_in_progress`, so setting
    /// dirty on failure made that release condition permanently false. Retry
    /// fires, fails, sets dirty again, backoff caps at ~32s and goes on
    /// forever: the waiter is parked for good and `bn sync` hangs with zero
    /// output. Keeping `dirty` is still correct — the fix is that the waiter
    /// now watches [`Self::sync_failures_total`] as well, so a failure wakes it
    /// with a verdict instead of stranding it. Anyone who removes
    /// `sync_failures_total` must replace it with some other liveness signal,
    /// or the deadlock comes straight back.
    pub fn fail_sync(&mut self, message: String, wall_ms: u64) {
        self.sync_in_progress = false;
        self.consecutive_failures += 1;
        self.sync_failures_total = self.sync_failures_total.saturating_add(1);
        self.last_sync_error = Some(SyncFailureRecord {
            message,
            consecutive_failures: self.consecutive_failures,
            wall_ms,
        });
        // Keep dirty=true so we retry the sync after backoff.
        self.dirty = true;
    }

    /// Calculate backoff delay for retries (exponential).
    pub fn backoff_ms(&self) -> u64 {
        let base = backoff_base_ms();
        let max_exponent = 6; // Max ~32 seconds
        let exponent = self.consecutive_failures.min(max_exponent);
        base * 2u64.pow(exponent)
    }
}

impl Default for GitLaneState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_not_dirty() {
        let state = GitLaneState::new();
        assert!(!state.dirty);
        assert!(!state.sync_in_progress);
        assert!(!state.refresh_in_progress);
        assert!(!state.loaded_from_git);
        assert_eq!(state.consecutive_failures, 0);
    }

    #[test]
    fn with_path_starts_unloaded_until_git_load_succeeds() {
        let path = PathBuf::from("/tmp/repo");
        let state = GitLaneState::with_path(None, path.clone());
        assert!(state.known_paths.contains(&path));
        assert!(!state.is_loaded_from_git());
        assert!(state.last_refresh.is_none());
    }

    #[test]
    fn mark_dirty() {
        let mut state = GitLaneState::new();
        state.mark_dirty();
        assert!(state.dirty);
        assert!(state.last_mutation.is_some());
    }

    /// A failure must leave the lane dirty (so it retries) AND leave a trail a
    /// parked waiter can act on. kopitiam#25: without the trail, "dirty so we
    /// retry" is indistinguishable from "hang forever".
    #[test]
    fn fail_sync_records_why_and_keeps_the_lane_dirty() {
        let mut state = GitLaneState::new();
        state.mark_dirty();
        state.start_sync();
        assert!(!state.dirty, "start_sync clears dirty to track mid-sync writes");

        state.fail_sync("could not read Username".to_string(), 42);

        assert!(!state.sync_in_progress);
        assert!(state.dirty, "must stay dirty so the scheduler retries");
        assert_eq!(state.consecutive_failures, 1);
        assert_eq!(state.sync_failures_total, 1);
        let record = state.last_sync_error.as_ref().expect("failure recorded");
        assert_eq!(record.message, "could not read Username");
        assert_eq!(record.consecutive_failures, 1);
        assert_eq!(record.wall_ms, 42);
    }

    /// `sync_failures_total` must be monotonic across a success, or a waiter
    /// parked at total=N gets stranded by the very next failure.
    ///
    /// The stranding sequence, if this ever reverts to `consecutive_failures`:
    /// fail x3 (counter 3) -> waiter parks snapshotting 3 -> sync succeeds but
    /// a mutation landed mid-sync so the lane stays dirty (counter resets to 0)
    /// -> retry fails (counter 1). 1 is not > 3, waiter never wakes.
    #[test]
    fn sync_failures_total_never_goes_backwards() {
        let mut state = GitLaneState::new();
        state.fail_sync("one".to_string(), 1);
        state.fail_sync("two".to_string(), 2);
        state.fail_sync("three".to_string(), 3);
        assert_eq!(state.consecutive_failures, 3);
        assert_eq!(state.sync_failures_total, 3);

        state.complete_sync(4);
        assert_eq!(state.consecutive_failures, 0, "success resets the backoff counter");
        assert_eq!(
            state.sync_failures_total, 3,
            "but the monotonic counter must NOT reset"
        );
        assert!(state.last_sync_error.is_none(), "success clears the error");

        state.fail_sync("four".to_string(), 5);
        assert_eq!(state.consecutive_failures, 1);
        assert_eq!(
            state.sync_failures_total, 4,
            "a waiter parked at 3 must be able to see this failure"
        );
    }

    #[test]
    fn backoff_exponential() {
        let mut state = GitLaneState::new();
        let base = backoff_base_ms();
        let max_backoff = base * 2u64.pow(6);
        assert_eq!(state.backoff_ms(), base);

        state.consecutive_failures = 1;
        assert_eq!(state.backoff_ms(), base * 2);

        state.consecutive_failures = 2;
        assert_eq!(state.backoff_ms(), base * 4);

        state.consecutive_failures = 6;
        assert_eq!(state.backoff_ms(), max_backoff);

        // Should cap at 6
        state.consecutive_failures = 10;
        assert_eq!(state.backoff_ms(), max_backoff);
    }
}
