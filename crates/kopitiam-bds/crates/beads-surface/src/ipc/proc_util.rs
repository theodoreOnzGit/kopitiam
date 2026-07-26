//! Windows process helpers.
//!
//! Unix uses `kill(pid, 0)` for liveness and `SIGTERM`/`SIGKILL` for shutdown.
//! Windows has neither, so we go through the Win32 process API. This module is
//! compiled only on Windows.

use windows_sys::Win32::Foundation::{CloseHandle, ERROR_INVALID_PARAMETER, GetLastError};
use windows_sys::Win32::System::Threading::{
    GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE,
    TerminateProcess,
};

/// `STATUS_PENDING` — a still-running process reports this as its exit code.
const STILL_ACTIVE: u32 = 259;

/// Three-state liveness, mirroring the unix `kill(pid, 0)` outcomes:
/// `Some(true)` alive (incl. access-denied, like `EPERM`), `Some(false)` no such
/// process (like `ESRCH`), `None` unknown/unexpected error.
///
/// This is the shared primitive used by both this crate's IPC client and the
/// `beads-daemon` store/lock code (which `#![forbid(unsafe_code)]` and so cannot
/// call the Win32 API directly — it delegates here).
pub fn pid_liveness(pid: u32) -> Option<bool> {
    use windows_sys::Win32::Foundation::ERROR_ACCESS_DENIED;
    // SAFETY: OpenProcess with a plain pid; a null return signals failure and is
    // checked before use. On success the handle is always closed.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return match GetLastError() {
                ERROR_INVALID_PARAMETER => Some(false),
                ERROR_ACCESS_DENIED => Some(true),
                _ => None,
            };
        }
        let mut code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        if ok == 0 {
            return None;
        }
        Some(code == STILL_ACTIVE)
    }
}

/// Best-effort liveness check: is a process with this pid currently running?
/// Access-restricted and unknown states are treated as alive, matching the unix
/// `kill(pid, 0)` behaviour used by the IPC client.
pub fn pid_alive(pid: u32) -> bool {
    pid_liveness(pid) != Some(false)
}

/// Forcibly terminate a process by pid. Best-effort; a missing process is Ok.
pub fn terminate(pid: u32) -> std::io::Result<()> {
    // SAFETY: OpenProcess/TerminateProcess/CloseHandle called with a checked
    // handle; failures are surfaced as io::Error.
    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if handle.is_null() {
            if GetLastError() == ERROR_INVALID_PARAMETER {
                return Ok(()); // already gone
            }
            return Err(std::io::Error::last_os_error());
        }
        let ok = TerminateProcess(handle, 1);
        CloseHandle(handle);
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}
