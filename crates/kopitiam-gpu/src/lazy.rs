//! Probing for a GPU **off** the calling thread.
//!
//! [`Executor::new`](crate::Executor::new) is blocking by design: it calls
//! `pollster::block_on` around wgpu's `request_adapter` and `request_device`.
//! That is the right shape for a CLI, which has nothing better to do while it
//! waits.
//!
//! It is the wrong shape for a GUI. Adapter enumeration means loading and
//! initialising a graphics driver — tens of milliseconds on a warm desktop,
//! and far worse on a cold Vulkan stack or a Termux userland that has to
//! discover it has no usable ICD before it can say so. Doing that on the UI
//! thread means the window does not appear until it finishes, and a machine
//! with *no* GPU pays the longest wait of all: it must exhaust every backend
//! to conclude nothing is there.
//!
//! [`LazyExecutor`] moves that probe onto a background thread and hands the
//! caller a handle that is immediately usable:
//!
//! ```no_run
//! # use kopitiam_gpu::LazyExecutor;
//! let gpu = LazyExecutor::spawn();           // returns at once
//! // ... open the window, draw the first frame ...
//! if let Some(exec) = gpu.try_get() {        // still non-blocking
//!     // the probe finished; use it
//!     let _ = exec.has_gpu();
//! }
//! ```
//!
//! # Why `try_get` rather than only a blocking `get`
//!
//! A frame loop must never block. `try_get` returns `None` while the probe is
//! still running, which lets a UI run its CPU path for the first few frames
//! and pick the GPU up when it arrives, rather than stalling to find out. A
//! blocking [`LazyExecutor::get`] exists too, for the batch caller that
//! genuinely cannot proceed without an answer.
//!
//! # What this is actually worth, measured honestly
//!
//! On a machine with **no** GPU driver at all (a headless container, CI), the
//! blocking probe costs under a millisecond: wgpu finds no Vulkan ICD to load
//! and gives up immediately, so there is nothing to hide. Measured here:
//! `Executor::new()` 800 µs, `LazyExecutor::spawn()` 90 µs.
//!
//! The cost this exists to move off the calling thread is the one paid by a
//! machine that *does* have drivers — loading the ICD, enumerating physical
//! devices, creating a logical device and queue. That is the common case for
//! the maintainer's own hardware and the one worth measuring on it; it has
//! **not** been measured here, because this container cannot produce it.
//!
//! So the claim is deliberately narrow: this makes startup independent of
//! whatever the probe costs, rather than making the probe faster. If it turns
//! out to cost little on real hardware too, nothing is lost but a thread.
//!
//! # This changes *when*, never *whether*
//!
//! The GPU→CPU cascade is untouched: a machine with no adapter still gets a
//! CPU-only [`Executor`], just without having made the caller wait for that
//! conclusion. Nothing here can turn a working GPU into a missing one.

use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use crate::Executor;

/// A GPU probe running on a background thread.
///
/// Cheap to construct and safe to keep for the life of the program. Build one
/// and share it: like [`Executor`], probing more than once wastes the very
/// cost this exists to hide.
pub struct LazyExecutor {
    /// Filled exactly once by the background thread.
    slot: Arc<OnceLock<Executor>>,
    /// The probe thread, taken by whoever blocks on it first. `None` once
    /// joined, or when the probe finished before anyone asked.
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl LazyExecutor {
    /// Start probing for a GPU on a background thread. Returns immediately.
    ///
    /// If the thread cannot be spawned — a hard resource limit, which is rare
    /// but not impossible — the probe runs inline instead, so the handle is
    /// still valid and still yields an [`Executor`]. Failing to spawn must not
    /// mean failing to compute.
    pub fn spawn() -> LazyExecutor {
        let slot: Arc<OnceLock<Executor>> = Arc::new(OnceLock::new());
        let worker = Arc::clone(&slot);
        let handle = std::thread::Builder::new()
            .name("kopitiam-gpu-probe".to_string())
            .spawn(move || {
                // `set` can only fail if something else already filled the
                // slot, which nothing does — this is the sole writer.
                let _ = worker.set(Executor::new());
            })
            .ok();
        if handle.is_none() {
            let _ = slot.set(Executor::new());
        }
        LazyExecutor {
            slot,
            handle: Mutex::new(handle),
        }
    }

    /// A [`LazyExecutor`] that is already resolved to a CPU-only executor.
    ///
    /// No thread, no probe. For tests and for a caller that has decided
    /// against the GPU — the counterpart to [`Executor::cpu_only`].
    pub fn cpu_only() -> LazyExecutor {
        let slot = Arc::new(OnceLock::new());
        let _ = slot.set(Executor::cpu_only());
        LazyExecutor {
            slot,
            handle: Mutex::new(None),
        }
    }

    /// Has the probe finished?
    pub fn is_ready(&self) -> bool {
        self.slot.get().is_some()
    }

    /// The executor if the probe has finished, `None` if it is still running.
    ///
    /// **Never blocks** — this is the one to call from a frame loop.
    pub fn try_get(&self) -> Option<&Executor> {
        self.slot.get()
    }

    /// The executor, waiting for the probe if necessary.
    ///
    /// Blocks only on the first call, and only if the probe has not already
    /// finished. Do not call this from a UI thread — that reintroduces exactly
    /// the stall this type exists to remove; use [`try_get`](Self::try_get).
    pub fn get(&self) -> &Executor {
        if let Some(exec) = self.slot.get() {
            return exec;
        }
        // Join the probe thread. Whoever gets the handle first does the join;
        // any other waiter falls through to the park loop below, which is a
        // rare path (two threads blocking on a probe that runs once) and not
        // worth a condvar.
        if let Ok(mut guard) = self.handle.lock()
            && let Some(h) = guard.take()
        {
            let _ = h.join();
        }
        loop {
            if let Some(exec) = self.slot.get() {
                return exec;
            }
            std::thread::yield_now();
        }
    }
}

impl Default for LazyExecutor {
    fn default() -> LazyExecutor {
        LazyExecutor::spawn()
    }
}

impl std::fmt::Debug for LazyExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyExecutor")
            .field("ready", &self.is_ready())
            .field("has_gpu", &self.slot.get().map(Executor::has_gpu))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// The whole point: constructing must not pay the probe's cost.
    ///
    /// The bound is deliberately loose — this asserts "did not block on a
    /// driver", not a precise timing — because a CI box's scheduler is not
    /// something a test should be sensitive to.
    #[test]
    fn spawn_returns_without_waiting_for_the_probe() {
        let t = Instant::now();
        let lazy = LazyExecutor::spawn();
        let elapsed = t.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "spawn blocked for {elapsed:?}; it must hand back a handle at once"
        );
        // And it still resolves to something usable.
        let _ = lazy.get().has_gpu();
    }

    #[test]
    fn get_always_resolves() {
        let lazy = LazyExecutor::spawn();
        let exec = lazy.get();
        // Whether there is a GPU depends on the machine; that there IS an
        // executor does not.
        let _ = exec.has_gpu();
        assert!(lazy.is_ready(), "after get, the probe has finished");
    }

    #[test]
    fn get_is_idempotent_and_returns_the_same_executor() {
        let lazy = LazyExecutor::spawn();
        let a = lazy.get() as *const Executor;
        let b = lazy.get() as *const Executor;
        assert_eq!(a, b, "the executor is built once and shared");
    }

    #[test]
    fn try_get_never_blocks_and_agrees_with_get_once_ready() {
        let lazy = LazyExecutor::spawn();
        // Whatever try_get says now, it must not have waited.
        let t = Instant::now();
        let early = lazy.try_get().is_some();
        assert!(t.elapsed().as_millis() < 100, "try_get must not block");
        let _ = early;
        // Once resolved, the two agree.
        let via_get = lazy.get().has_gpu();
        assert_eq!(lazy.try_get().map(Executor::has_gpu), Some(via_get));
    }

    #[test]
    fn cpu_only_is_ready_immediately_and_has_no_gpu() {
        let lazy = LazyExecutor::cpu_only();
        assert!(lazy.is_ready());
        assert!(!lazy.get().has_gpu());
    }

    /// Several threads may wait on one probe without deadlocking — the path
    /// where one joins the handle and the others fall through to the loop.
    #[test]
    fn concurrent_waiters_all_resolve() {
        let lazy = Arc::new(LazyExecutor::spawn());
        let mut hs = Vec::new();
        for _ in 0..4 {
            let l = Arc::clone(&lazy);
            hs.push(std::thread::spawn(move || l.get().has_gpu()));
        }
        let results: Vec<bool> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.len(), 4);
        assert!(
            results.iter().all(|r| *r == results[0]),
            "every waiter must see the same executor"
        );
    }
}
