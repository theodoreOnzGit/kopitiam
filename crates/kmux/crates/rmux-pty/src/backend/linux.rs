use std::os::fd::{BorrowedFd, OwnedFd, RawFd};

use std::io;
use std::mem::MaybeUninit;

use rustix::fs::{fcntl_getfl, fcntl_setfl, open, Mode, OFlags};
use rustix::process::{
    getpid, ioctl_tiocsctty, kill_process as rustix_kill_process, kill_process_group, setsid,
};
#[cfg(target_os = "linux")]
use rustix::pty::ioctl_tiocgptpeer;
use rustix::pty::{grantpt, openpt, ptsname, unlockpt, OpenptFlags};
use rustix::termios::{tcgetwinsize, tcsetpgrp, tcsetwinsize};

use super::unix_io;
use crate::{size, ProcessId, Result, Signal, TerminalGeometry, TerminalSize};

pub(crate) fn open_pty_pair() -> Result<(OwnedFd, OwnedFd)> {
    let master = openpt(OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC)?;
    grantpt(&master)?;
    unlockpt(&master)?;

    let slave = open_slave(&master)?;

    Ok((master, slave))
}

/// Opens the PTY slave for an already-unlocked master.
///
/// Prefers `TIOCGPTPEER` (Linux 4.13+), which hands back the peer descriptor
/// directly from the master. That is not just faster — it is race-free, where
/// resolving `ptsname()` to a path and then `open()`ing that path is a
/// name-based lookup that could in principle resolve to something else.
///
/// # Android
///
/// **`rustix` gates `ioctl_tiocgptpeer` on `target_os = "linux"` and provides
/// no Android binding**, even though Android's kernel has supported the ioctl
/// since 4.13 like any other Linux. So on Android we go straight to the
/// name-based path.
///
/// This is not a degradation in practice. `openpt`/`grantpt`/`unlockpt`/
/// `ptsname` all work under Bionic, Termux mounts a normal `devpts` at
/// `/dev/pts`, and the name-based route is exactly how Termux's own terminal
/// emulator acquires its PTY. The TOCTOU concern is also weaker here than it
/// looks: `devpts` entries are created by the kernel for the master we are
/// holding open, not by other users in a shared directory.
///
/// If `rustix` ever grows an Android binding for this ioctl, delete the cfg and
/// let both platforms take the fast path.
#[cfg(target_os = "linux")]
fn open_slave(master: &OwnedFd) -> Result<OwnedFd> {
    match ioctl_tiocgptpeer(
        master,
        OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC,
    ) {
        Ok(slave) => Ok(slave),
        Err(peer_error) => open_slave_by_name(master).map_err(|_| peer_error.into()),
    }
}

/// Android's `open_slave`. See the `target_os = "linux"` variant above for why
/// the `TIOCGPTPEER` fast path is unavailable here.
#[cfg(target_os = "android")]
fn open_slave(master: &OwnedFd) -> Result<OwnedFd> {
    open_slave_by_name(master)
}

fn open_slave_by_name(master: &OwnedFd) -> Result<OwnedFd> {
    let slave_name = ptsname(master, Vec::new())?;
    Ok(open(
        slave_name.as_c_str(),
        OFlags::RDWR | OFlags::NOCTTY | OFlags::CLOEXEC,
        Mode::empty(),
    )?)
}

pub(crate) fn query_size(fd: BorrowedFd<'_>) -> Result<TerminalSize> {
    Ok(size::from_winsize(tcgetwinsize(fd)?))
}

pub(crate) fn apply_size(fd: BorrowedFd<'_>, size: TerminalSize) -> Result<()> {
    tcsetwinsize(fd, size::into_winsize(size))?;
    Ok(())
}

pub(crate) fn apply_geometry(fd: BorrowedFd<'_>, geometry: TerminalGeometry) -> Result<()> {
    tcsetwinsize(fd, size::into_winsize_geometry(geometry))?;
    Ok(())
}

pub(crate) fn setup_child_controlling_terminal(raw_master_fd: RawFd) -> std::io::Result<()> {
    // SAFETY: This closes only the child process' inherited copy of the PTY
    // master fd. The parent still owns its separate descriptor.
    unsafe { rustix::io::close(raw_master_fd) };

    setsid().map_err(std::io::Error::from)?;

    // SAFETY: `stdin` has already been wired to the PTY slave by `Command`, so
    // fd 0 is a valid borrowed descriptor for the rest of the pre-exec setup.
    let slave_stdin = unsafe { BorrowedFd::borrow_raw(0) };
    ioctl_tiocsctty(slave_stdin).map_err(std::io::Error::from)?;
    tcsetpgrp(slave_stdin, getpid()).map_err(std::io::Error::from)?;

    Ok(())
}

pub(crate) fn kill_foreground_process_group(pid: ProcessId, signal: Signal) -> Result<()> {
    kill_process_group(pid.as_rustix_pid()?, signal.as_rustix_signal())?;
    Ok(())
}

pub(crate) fn kill_process(pid: ProcessId, signal: Signal) -> Result<()> {
    rustix_kill_process(pid.as_rustix_pid()?, signal.as_rustix_signal())?;
    Ok(())
}

pub(crate) fn stopped_signal(pid: ProcessId) -> Result<Option<i32>> {
    let mut info = MaybeUninit::<libc::siginfo_t>::zeroed();
    // SAFETY: `info` points to writable storage for one siginfo_t. WNOWAIT
    // observes the stopped status without consuming the child's eventual exit
    // status, which remains owned by `std::process::Child`.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            pid.as_u32() as libc::id_t,
            info.as_mut_ptr(),
            libc::WSTOPPED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if result == -1 {
        let errno = last_errno();
        if errno == rustix::io::Errno::CHILD {
            return Ok(None);
        }
        return Err(errno.into());
    }

    // SAFETY: `info` was zero-initialized before the call and `waitid`
    // returned success, so reading the initialized siginfo_t is valid.
    let info = unsafe { info.assume_init() };
    // SAFETY: `waitid` with WSTOPPED populates `si_pid` when a stopped child
    // is available. A zero pid means WNOHANG had no event.
    let si_pid = unsafe { info.si_pid() };
    if si_pid == 0 {
        Ok(None)
    } else {
        // SAFETY: a non-zero `si_pid` means this siginfo_t carries a real
        // child status for the requested child.
        Ok(Some(unsafe { info.si_status() }))
    }
}

pub(crate) fn read(fd: BorrowedFd<'_>, buffer: &mut [u8]) -> io::Result<usize> {
    unix_io::read(fd, buffer)
}

pub(crate) fn write_all(fd: BorrowedFd<'_>, buffer: &[u8]) -> io::Result<()> {
    unix_io::write_all(fd, buffer)
}

pub(crate) fn try_write_immediate(fd: BorrowedFd<'_>, buffer: &[u8]) -> io::Result<usize> {
    unix_io::try_write_immediate(fd, buffer)
}

pub(crate) fn set_nonblocking(fd: BorrowedFd<'_>) -> io::Result<()> {
    let flags = fcntl_getfl(fd).map_err(io::Error::other)?;
    fcntl_setfl(fd, flags | OFlags::NONBLOCK).map_err(io::Error::other)?;
    Ok(())
}

fn last_errno() -> rustix::io::Errno {
    let raw = std::io::Error::last_os_error()
        .raw_os_error()
        .unwrap_or(libc::EIO);
    rustix::io::Errno::from_raw_os_error(raw)
}
