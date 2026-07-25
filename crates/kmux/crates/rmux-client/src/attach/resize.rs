use std::os::fd::AsFd;

use rmux_proto::{TerminalGeometry, TerminalPixels, TerminalSize};
use rustix::termios::tcgetwinsize;

// SIGWINCH watching is the one place in this fork where Android does NOT take
// the Linux path, and the reason is a toolchain limitation rather than a kernel
// one.
//
// `resize/linux.rs` waits for SIGWINCH with `rustix::runtime::kernel_sigwait`.
// `rustix::runtime` exists only on rustix's `linux_raw` backend, and rustix
// enables `linux_raw` only for `target_os = "linux"` — on Android it falls back
// to its `libc` backend, where the `runtime` module is simply absent. So the
// Linux backend cannot compile for Android no matter how the cfg is written.
//
// Rather than hand-roll `sigwait`/`pthread_kill` against Bionic, Android reuses
// the `signal-hook` backend that upstream already wrote and maintains for
// macOS. `signal-hook` supports Android, is safe Rust over `sigaction` plus a
// self-pipe, and is already a `cfg(unix)` dependency of this crate — so this
// costs no new unsafe code and no new dependency. The file was renamed from
// `macos.rs` to `signal_hook.rs` to say what it actually is.
#[cfg(target_os = "linux")]
#[path = "resize/linux.rs"]
mod platform;
#[cfg(any(target_os = "macos", target_os = "android"))]
#[path = "resize/signal_hook.rs"]
mod platform;

pub(super) use platform::{ResizeWatcher, SignalMaskGuard};

use super::Result;

#[cfg(test)]
pub(super) fn terminal_size_from_fd<Fd>(fd: &Fd) -> Result<Option<TerminalSize>>
where
    Fd: AsFd,
{
    Ok(terminal_geometry_from_fd(fd)?.map(|geometry| geometry.size))
}

pub(super) fn terminal_geometry_from_fd<Fd>(fd: &Fd) -> Result<Option<TerminalGeometry>>
where
    Fd: AsFd,
{
    let winsize = tcgetwinsize(fd)?;
    let size = TerminalSize {
        cols: winsize.ws_col,
        rows: winsize.ws_row,
    };
    if size.cols == 0 || size.rows == 0 {
        return Ok(None);
    }
    let pixels = (winsize.ws_xpixel > 0 && winsize.ws_ypixel > 0)
        .then(|| TerminalPixels::new(winsize.ws_xpixel, winsize.ws_ypixel));
    Ok(Some(TerminalGeometry { size, pixels }))
}
