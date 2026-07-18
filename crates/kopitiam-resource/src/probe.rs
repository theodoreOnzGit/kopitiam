//! The device capability probe — [`Capacity`] (the numbers), [`DeviceProbe`]
//! (the injectable source of them), and two implementations: [`SysinfoProbe`]
//! (the real one) and [`FixedProbe`] (synthetic, for tests and downstream tests).
//!
//! # Why a trait, not just a function
//!
//! `temp_ai_design.md` §6 asks for an *injectable* probe so the budget decision
//! can be unit-tested against synthetic devices without touching the real
//! machine. So the probe is a trait; [`SysinfoProbe`] is the production impl,
//! [`FixedProbe`] hands back canned numbers. Everything downstream reasons over a
//! plain [`Capacity`] struct, so the arithmetic never depends on *how* the
//! numbers were obtained.
//!
//! # Why `sysinfo`, and why the design's "no C sysinfo" line is stale
//!
//! `temp_ai_design.md` §6 says to hand-parse `/proc/meminfo` and avoid "the C
//! `sysinfo` crate" to keep the Pure Rust Core. **That premise is wrong and the
//! maintainer has already overruled it.** The `sysinfo` *crate* is **pure Rust**
//! (it reads `/proc` under the hood on Linux/Android — no C, no build step), it
//! is Android-capable, it is the crate `bottom`/`btm` is built on, and kvim
//! already ships it as its adopted device probe (AID-0037). So we use it, one
//! dependency, and do **not** hand-roll `/proc` parsing. The design note is a
//! stale, incorrect premise; this comment is the correction so nobody re-derives
//! the mistake.

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// A snapshot of the device's resources, in the units the budgeter reasons in.
///
/// A plain `Copy` struct of numbers on purpose: the budget arithmetic
/// ([`crate::budget`]) takes one of these and nothing else, so it can be tested
/// against synthetic devices. Whoever built it — real probe or test fixture — is
/// irrelevant past this point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capacity {
    /// Kernel's estimate of RAM allocatable **without swapping**, in **MB**
    /// (base-2). This is the honest budget the OOM killer actually watches — NOT
    /// total RAM, NOT nominal free RAM. On Android (swapless) this is what you
    /// have before the low-memory killer wakes up, so it is the number that
    /// predicts the kill. **Re-read this right before each heavy launch** — free
    /// RAM is volatile.
    pub avail_mb: u64,
    /// Total physical RAM in **MB**. Message context only; the budget uses
    /// [`Capacity::avail_mb`].
    pub total_mb: u64,
    /// Logical CPU count (hardware threads) — what bounds rust-analyzer's
    /// indexing parallelism, and the axis [`crate::budget::core_factor`] scales
    /// on.
    pub logical_cores: usize,
    /// Current system-wide CPU usage `0.0..=100.0`, best-effort. Message flavour
    /// only; may read `0.0` on a cold single-sample probe (see [`SysinfoProbe`]),
    /// which is fine because the gate's CPU axis is the exact *core count*, not
    /// this.
    pub cpu_usage: f32,
}

/// The injectable source of a [`Capacity`] reading.
///
/// Production code holds a `&dyn DeviceProbe` (or a generic `P: DeviceProbe`) and
/// calls [`snapshot`](DeviceProbe::snapshot) right before each heavy launch;
/// tests hand in a [`FixedProbe`]. `None` means "no honest reading available on
/// this platform" — the caller treats that as **fail-open** (carry on unguarded;
/// a machine we cannot measure must not be blocked).
pub trait DeviceProbe {
    /// Take a fresh reading of the device *now*. Cheap enough to call before
    /// every launch — that repeated freshness is the whole point (§6 volatility).
    /// `None` = could not read honestly → fail open.
    fn snapshot(&self) -> Option<Capacity>;
}

/// The real probe: reads the current device via the `sysinfo` crate.
///
/// Stateless and zero-sized — hold one and call [`snapshot`](DeviceProbe::snapshot)
/// as often as you like; each call takes a fresh reading. (We deliberately do
/// **not** cache a `System` inside: available RAM is the volatile number we most
/// want fresh, and a fresh minimal `System` is cheap.)
#[derive(Debug, Clone, Copy, Default)]
pub struct SysinfoProbe;

impl DeviceProbe for SysinfoProbe {
    /// Reads available + total RAM, the logical core count, and current CPU
    /// usage. Returns `None` when available memory reads as `0`, which is how a
    /// platform with no usable `sysinfo` memory backend shows up — better to fail
    /// open than to gate on a bogus zero-budget reading.
    ///
    /// # A note on CPU usage (preserved format knowledge)
    ///
    /// `sysinfo` needs **two** samples spaced by `MINIMUM_CPU_UPDATE_INTERVAL` to
    /// report a meaningful global CPU percentage. This probe takes a single cheap
    /// sample (no sleep — we may be on a UI thread), so [`Capacity::cpu_usage`]
    /// can come back `0.0` on the first call. That is acceptable: CPU usage is
    /// only message flavour here; the gate's CPU axis is the *core count*, which
    /// is exact. Also note: `sysinfo` reports memory in **bytes** (it moved off
    /// kB several versions ago), so we divide by `1024·1024` for MB.
    fn snapshot(&self) -> Option<Capacity> {
        let sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_memory(MemoryRefreshKind::nothing().with_ram())
                .with_cpu(CpuRefreshKind::nothing().with_cpu_usage()),
        );

        let avail_mb = sys.available_memory() / (1024 * 1024);
        let total_mb = sys.total_memory() / (1024 * 1024);
        if avail_mb == 0 {
            // No usable memory backend on this platform -> fail open.
            return None;
        }

        // `available_parallelism` is pure std and is exactly "how many threads
        // can run at once" — what bounds RA's indexing. Fall back to sysinfo's
        // CPU list, then to 1.
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or_else(|_| sys.cpus().len().max(1));

        Some(Capacity {
            avail_mb,
            total_mb,
            logical_cores,
            cpu_usage: sys.global_cpu_usage(),
        })
    }
}

/// A probe that always returns the same canned [`Capacity`] — for unit tests
/// (here and downstream) that want to pin a synthetic device without touching
/// `sysinfo`. `FixedProbe(None)` models "no honest reading" (the fail-open path).
///
/// Public on purpose: crates that build on the budgeter should be able to inject
/// a synthetic device into their own tests the same way this crate does.
#[derive(Debug, Clone, Copy)]
pub struct FixedProbe(pub Option<Capacity>);

impl FixedProbe {
    /// A `FixedProbe` that reports the given capacity.
    pub fn some(cap: Capacity) -> Self {
        Self(Some(cap))
    }

    /// A `FixedProbe` that reports nothing — models the fail-open path.
    pub fn none() -> Self {
        Self(None)
    }
}

impl DeviceProbe for FixedProbe {
    fn snapshot(&self) -> Option<Capacity> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_probe_hands_back_its_capacity() {
        let cap = Capacity { avail_mb: 2000, total_mb: 4000, logical_cores: 4, cpu_usage: 5.0 };
        assert_eq!(FixedProbe::some(cap).snapshot(), Some(cap));
        assert_eq!(FixedProbe::none().snapshot(), None);
    }

    #[test]
    fn sysinfo_probe_is_sane_or_none() {
        // Host-dependent: can't assert exact numbers. If a reading comes back it
        // must be internally sane; on a backend-less CI box it may be None, which
        // is the fail-open path -- also fine.
        if let Some(c) = SysinfoProbe.snapshot() {
            assert!(c.avail_mb > 0, "a Some reading must have non-zero available RAM");
            assert!(c.logical_cores >= 1);
            assert!(c.total_mb >= c.avail_mb, "total must be >= available");
        }
    }
}
