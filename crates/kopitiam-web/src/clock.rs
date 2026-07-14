use std::sync::Mutex;

use chrono::{DateTime, Duration, Utc};

/// The source of "now" used to stamp a retrieval timestamp.
///
/// # Why this is a trait
///
/// Provenance is only as trustworthy as its timestamps, so the timestamp logic
/// has to be *tested*, and you cannot test time-dependent behaviour against the
/// real wall clock. Specifically, the single most important invariant in this
/// crate — **a cached result is returned with its original `retrieved_at`, not
/// re-stamped as fresh** — is only observable if a test can make "now" advance
/// on demand. [`SteppingClock`] exists precisely to catch that bug.
///
/// This is the same reason `kopitiam-ai` has an `EchoAdapter`: the deterministic
/// stub is not a toy, it is what makes the real thing verifiable.
///
/// All timestamps are UTC. Provenance recorded in local time would mean
/// different things to different readers of the same knowledge graph, which
/// defeats the purpose of recording it.
pub trait Clock: Send + Sync {
    /// The current instant, in UTC.
    fn now(&self) -> DateTime<Utc>;
}

/// The real wall clock. The only [`Clock`] that should appear in production.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A [`Clock`] frozen at one instant.
///
/// For tests that need a known timestamp but do not care about the passage of
/// time.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock(DateTime<Utc>);

impl FixedClock {
    /// Freezes time at `at`.
    pub fn new(at: DateTime<Utc>) -> Self {
        Self(at)
    }

    /// Freezes time at the given Unix timestamp (seconds since the epoch).
    ///
    /// # Panics
    ///
    /// If `secs` is not a representable timestamp. Test-only convenience; the
    /// production path never calls it.
    pub fn from_unix(secs: i64) -> Self {
        Self(DateTime::from_timestamp(secs, 0).expect("representable timestamp"))
    }
}

impl Clock for FixedClock {
    fn now(&self) -> DateTime<Utc> {
        self.0
    }
}

/// A [`Clock`] that advances by a fixed step on every read.
///
/// This is the instrument that makes the cache's central promise falsifiable.
/// Wire one into the *inner* provider, and every fresh search it performs
/// carries a visibly different timestamp; a cache hit that came back with a
/// *newer* timestamp than the one recorded would then be caught immediately.
/// Re-stamping a cached result as "retrieved now" is a lie about when the web
/// said what it said, and it is exactly the bug a real clock would hide.
#[derive(Debug)]
pub struct SteppingClock {
    state: Mutex<DateTime<Utc>>,
    step: Duration,
}

impl SteppingClock {
    /// Starts at `start` and advances by `step_secs` seconds per call to
    /// [`Clock::now`]. The first call returns `start` itself.
    pub fn new(start: DateTime<Utc>, step_secs: i64) -> Self {
        Self {
            state: Mutex::new(start),
            step: Duration::seconds(step_secs),
        }
    }

    /// Starts at the given Unix timestamp, advancing one hour per read.
    ///
    /// An hour is deliberately coarse: a difference of an hour is unmissable in
    /// a failing assertion, where a difference of a millisecond might be
    /// mistaken for noise.
    ///
    /// # Panics
    ///
    /// If `secs` is not a representable timestamp. Test-only convenience.
    pub fn hourly_from_unix(secs: i64) -> Self {
        Self::new(
            DateTime::from_timestamp(secs, 0).expect("representable timestamp"),
            3600,
        )
    }
}

impl Clock for SteppingClock {
    fn now(&self) -> DateTime<Utc> {
        let mut state = self.state.lock().expect("stepping clock mutex poisoned");
        let now = *state;
        *state += self.step;
        now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixed_clock_never_moves() {
        let clock = FixedClock::from_unix(1_700_000_000);
        assert_eq!(clock.now(), clock.now());
        assert_eq!(clock.now().timestamp(), 1_700_000_000);
    }

    #[test]
    fn a_stepping_clock_advances_on_every_read() {
        let clock = SteppingClock::hourly_from_unix(1_700_000_000);
        assert_eq!(clock.now().timestamp(), 1_700_000_000);
        assert_eq!(clock.now().timestamp(), 1_700_003_600);
        assert_eq!(clock.now().timestamp(), 1_700_007_200);
    }

    #[test]
    fn the_system_clock_is_in_utc_and_moves_forward() {
        let before = SystemClock.now();
        let after = SystemClock.now();
        assert!(after >= before);
    }
}
