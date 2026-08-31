//! Test-only hooks for integration crash tests.

use crate::daemon::runtime::wal::WalIndexError;

#[cfg(any(feature = "slow-tests", feature = "test-harness"))]
pub(crate) fn maybe_pause(stage: &str) {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    let Ok(target) = std::env::var("BD_TEST_WAL_HANG_STAGE") else {
        return;
    };
    if target != stage {
        return;
    }

    let Ok(dir) = std::env::var("BD_TEST_WAL_HANG_DIR") else {
        return;
    };

    let marker = PathBuf::from(dir).join(format!("beads-wal-hang-{stage}"));
    let _ = fs::write(marker, b"");

    let timeout_ms = std::env::var("BD_TEST_WAL_HANG_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30_000);
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(not(any(feature = "slow-tests", feature = "test-harness")))]
pub(crate) fn maybe_pause(_stage: &str) {}

#[cfg(any(test, feature = "test-harness"))]
#[derive(Clone, Debug)]
struct AtomicCommitFailpoint {
    stage: String,
}

/// Which atomic-commit stage (if any) should blow up on **this thread**.
///
/// Thread-local on purpose, and it must stay that way — same pattern as
/// [`crate::daemon::paths`]'s `TEST_DATA_DIR_OVERRIDE`, for the same reason.
/// This used to be one process-wide `static OnceLock<Mutex<Option<_>>>`, i.e.
/// ONE slot shared by every test thread. That slot carried a `ThreadId` owner
/// so a failpoint could never fire on the wrong thread — which sounds safe, but
/// it guards the wrong hazard. The bug was that arming never STACKED:
///
/// 1. thread A arms stage `X` — slot is `Some(X, A)`
/// 2. thread B arms stage `Y` — slot is now `Some(Y, B)`, A's entry stashed in
///    B's guard as `previous`
/// 3. thread A reaches its injection point and asks for `X` — the slot says
///    `Y`, no match, so **A's failure never fires**
/// 4. A's "and then it rolled back" assertion sees nothing rolled back, fails
/// 5. B drops and politely restores `Some(X, A)` — which is exactly why this
///    showed up as an intermittent flake (2 tests, ~3 runs in 8) instead of a
///    constant failure, and why it went away under `--test-threads=1`
///
/// Per-thread state means one test cannot disarm another's failpoint, so the
/// `owner: ThreadId` field is gone — there is no longer any way to observe a
/// sibling's arming, let alone fire on it. See gh-101.
#[cfg(any(test, feature = "test-harness"))]
thread_local! {
    static ATOMIC_COMMIT_FAILPOINT: std::cell::RefCell<Option<AtomicCommitFailpoint>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(any(test, feature = "test-harness"))]
pub(crate) fn maybe_fail_atomic_commit(stage: &str) -> Result<(), WalIndexError> {
    let armed = ATOMIC_COMMIT_FAILPOINT
        .with(|slot| slot.borrow().as_ref().is_some_and(|fp| fp.stage == stage));
    if armed {
        return Err(WalIndexError::Sql {
            message: format!("injected atomic commit failure at {stage}"),
        });
    }
    Ok(())
}

#[cfg(not(any(test, feature = "test-harness")))]
pub(crate) fn maybe_fail_atomic_commit(_stage: &str) -> Result<(), WalIndexError> {
    Ok(())
}

/// Disarms on drop, restoring whatever this thread had armed before.
///
/// Nesting on one thread still works (the previous entry comes back), which is
/// the only nesting that can happen now that the slot is thread-local.
#[cfg(test)]
pub(crate) struct AtomicCommitFailpointGuard {
    previous: Option<AtomicCommitFailpoint>,
}

#[cfg(test)]
impl Drop for AtomicCommitFailpointGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ATOMIC_COMMIT_FAILPOINT.with(|slot| *slot.borrow_mut() = previous);
    }
}

#[cfg(test)]
pub(crate) fn set_atomic_commit_fail_stage_for_tests(stage: &str) -> AtomicCommitFailpointGuard {
    let previous = ATOMIC_COMMIT_FAILPOINT.with(|slot| {
        slot.borrow_mut().replace(AtomicCommitFailpoint {
            stage: stage.to_string(),
        })
    });
    AtomicCommitFailpointGuard { previous }
}
