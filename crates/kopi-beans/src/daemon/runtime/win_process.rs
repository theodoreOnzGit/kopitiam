//! Windows process-liveness helper for the daemon.
//!
//! `beads-daemon` is `#![forbid(unsafe_code)]`, so it cannot call the Win32
//! process API directly. The actual `OpenProcess`/`GetExitCodeProcess` logic
//! lives in `beads-surface` (which permits `unsafe`); this is a thin,
//! safe delegating wrapper. Compiled only on Windows.

/// Three-state liveness, mirroring the unix `kill(pid, 0)` outcomes:
/// `Some(true)` alive, `Some(false)` no such process, `None` unknown.
pub(crate) fn pid_liveness(pid: u32) -> Option<bool> {
    crate::surface::ipc::proc_util::pid_liveness(pid)
}
