use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::clock::Clock;

/// Enforces a minimum interval between requests to one provider.
///
/// # Why bother, when the server enforces its own limits anyway
///
/// Because the server enforces them by *refusing us*, and a refusal is a wasted
/// query on a metered plan, a failed workflow, and — on some plans — a strike
/// against the account. Brave's free tier, for instance, allows one query per
/// second; a loop over twenty literature references will trip it instantly and
/// get twenty errors instead of twenty answers.
///
/// Being a well-behaved client is also simply the right thing to do when the
/// service you are calling is somebody's self-hosted SearXNG instance rather
/// than a hyperscaler.
///
/// # Deliberately not a token bucket
///
/// A token bucket would allow bursts, which is exactly what these APIs punish.
/// A flat minimum interval is the honest model of "N queries per second" and it
/// is trivial to reason about, which matters more here than throughput: nothing
/// in KOPITIAM should ever be issuing enough web searches for the difference to
/// show.
///
/// # Testability
///
/// [`wait_needed`](Self::wait_needed) is pure — it takes "now" as an argument
/// and returns how long to sleep — so the policy can be tested against a
/// [`FixedClock`](crate::FixedClock) without any test ever sleeping. Only
/// [`acquire`](Self::acquire) blocks, and only the live HTTP adapters call it.
#[derive(Debug)]
pub struct RateLimiter {
    min_interval: Duration,
    last_request: Mutex<Option<DateTime<Utc>>>,
}

impl RateLimiter {
    /// At most `n` requests per second.
    ///
    /// # Panics
    ///
    /// If `n` is zero — a rate limiter that permits nothing is a mistake, not a
    /// configuration, and failing loudly at construction beats hanging forever
    /// at the first request.
    pub fn per_second(n: u32) -> Self {
        assert!(n > 0, "a rate limit of zero requests per second would never proceed");
        Self::with_min_interval(Duration::from_secs_f64(1.0 / f64::from(n)))
    }

    /// At least `min_interval` between consecutive requests.
    pub fn with_min_interval(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_request: Mutex::new(None),
        }
    }

    /// How long a request issued at `now` must wait. Zero if it may go at once.
    ///
    /// Pure: does not sleep, does not record. See
    /// [`acquire`](Self::acquire).
    pub fn wait_needed(&self, now: DateTime<Utc>) -> Duration {
        let last = *self.last_request.lock().expect("rate limiter mutex poisoned");
        let Some(last) = last else {
            return Duration::ZERO;
        };

        let elapsed = now.signed_duration_since(last);
        // A negative elapsed time means the clock went backwards (NTP step,
        // suspend/resume). Treat it as "no time has passed" and wait the full
        // interval: pausing too long is a nuisance, hammering a provider because
        // the clock jumped is a ban.
        match elapsed.to_std() {
            Ok(elapsed) => self.min_interval.saturating_sub(elapsed),
            Err(_) => self.min_interval,
        }
    }

    /// Records that a request was issued at `now`.
    pub fn record(&self, now: DateTime<Utc>) {
        *self.last_request.lock().expect("rate limiter mutex poisoned") = Some(now);
    }

    /// Blocks until a request may be issued, then records it.
    ///
    /// The one blocking call in this crate. It sleeps rather than returning an
    /// error because the caller has already decided to search; making them
    /// implement their own back-off loop would just get it wrong somewhere else.
    pub fn acquire(&self, clock: &dyn Clock) {
        let wait = self.wait_needed(clock.now());
        if !wait.is_zero() {
            std::thread::sleep(wait);
        }
        self.record(clock.now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FixedClock;

    #[test]
    fn the_first_request_never_waits() {
        let limiter = RateLimiter::per_second(1);
        assert_eq!(
            limiter.wait_needed(FixedClock::from_unix(1_700_000_000).now()),
            Duration::ZERO
        );
    }

    #[test]
    fn a_request_too_soon_after_the_last_one_must_wait_the_remainder() {
        let limiter = RateLimiter::per_second(1);
        limiter.record(FixedClock::from_unix(1_700_000_000).now());

        // 400ms later: 600ms still owed.
        let now = DateTime::from_timestamp_millis(1_700_000_000_400).unwrap();
        assert_eq!(limiter.wait_needed(now), Duration::from_millis(600));
    }

    #[test]
    fn a_request_after_the_interval_has_elapsed_goes_straight_through() {
        let limiter = RateLimiter::per_second(1);
        limiter.record(FixedClock::from_unix(1_700_000_000).now());

        let now = FixedClock::from_unix(1_700_000_002).now();
        assert_eq!(limiter.wait_needed(now), Duration::ZERO);
    }

    #[test]
    fn a_clock_that_jumps_backwards_makes_us_more_cautious_not_less() {
        let limiter = RateLimiter::per_second(1);
        limiter.record(FixedClock::from_unix(1_700_000_010).now());

        // NTP steps the clock back ten seconds. We must not conclude that we are
        // free to fire immediately.
        let now = FixedClock::from_unix(1_700_000_000).now();
        assert_eq!(limiter.wait_needed(now), Duration::from_secs(1));
    }

    #[test]
    #[should_panic(expected = "would never proceed")]
    fn a_rate_limit_of_zero_is_rejected_at_construction() {
        RateLimiter::per_second(0);
    }
}
