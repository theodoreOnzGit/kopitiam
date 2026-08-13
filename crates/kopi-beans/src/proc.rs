//! Spawn short-lived helper processes without a console window flashing up.
//!
//! # The bug this exists for (kopitiam#29 / `bd-u7j`)
//!
//! On Windows the maintainer got a black console window popping up on **every
//! single daemon sync** — super annoying lah. The chain that causes it:
//!
//! 1. The daemon is spawned with `DETACHED_PROCESS | CREATE_NO_WINDOW` (see
//!    `surface::ipc::spawn_sanitizer::spawn_daemon_process_windows`). That is
//!    correct for a long-lived background process, but it means the daemon
//!    process **owns no console** at all.
//! 2. Since 0.1.3 (kopitiam#19) network push shells out to the system `git`
//!    binary — [`crate::git::gix_compat`]'s `subprocess_push`.
//! 3. On Windows, when a process that has **no console** starts a
//!    console-subsystem child, the OS allocates the child a **brand new console
//!    window**. `git.exe` is console-subsystem, so up pops a window.
//! 4. The daemon debounces then auto-syncs, so it's one popup per sync. Jialat.
//!
//! # What to pass, and what NOT to pass
//!
//! Every caller here captures stdout/stderr with `.output()`, so nobody wants a
//! console at all — plain **`CREATE_NO_WINDOW`** is exactly right. It tells
//! `CreateProcess` "run console-subsystem, but don't give it a window", and the
//! pipes we set up still work normally.
//!
//! **Do not OR in `DETACHED_PROCESS` here**, even though the daemon spawn does.
//! Those two flags answer different questions:
//!
//! | Flag | Means | Right for |
//! |---|---|---|
//! | `CREATE_NO_WINDOW` | child runs console-subsystem, no window drawn | short-lived child whose output we capture |
//! | `DETACHED_PROCESS` | child gets no console *and* is cut loose from ours | long-lived background process that outlives us |
//!
//! Detaching a short-lived child we're about to `.wait()` on and read pipes
//! from buys nothing and muddies the lifetime story. Also note the two flags
//! are documented by Microsoft as mutually exclusive *in general* — the daemon
//! path passes both deliberately via its hand-rolled `CreateProcessW`, but
//! there is no reason to copy that here.
//!
//! # Provenance of the flag value
//!
//! `CREATE_NO_WINDOW` is `0x0800_0000`. We do **not** hardcode it: it comes
//! from `windows_sys::Win32::System::Threading::CREATE_NO_WINDOW`, which is
//! already a `[target.'cfg(windows)'.dependencies]` of this crate (feature
//! `Win32_System_Threading`), same place the daemon spawn gets it. One source
//! of truth, no magic number floating around.
//!
//! # Unix
//!
//! No-op — there is no such thing as a console window to suppress, and the
//! child's stdio is whatever we configured. The function still exists on Unix
//! so call sites stay `cfg`-free.

#![forbid(unsafe_code)]

use std::process::Command;

/// Tell `cmd` "eh, don't pop up any window ah" before it gets spawned.
///
/// Windows: sets the `CREATE_NO_WINDOW` creation flag. Unix: does nothing at
/// all — `cmd` comes back byte-for-byte the same command you passed in.
///
/// Returns the same `&mut Command` so it chains:
///
/// ```ignore
/// let mut cmd = Command::new("git");
/// let output = dun_popup(&mut cmd).arg("push").output()?;
/// ```
///
/// # When this is the WRONG helper
///
/// Only use it for a **short-lived child whose stdout/stderr we capture**. If
/// you ever need a child that outlives this process, that's the daemon case —
/// go look at `surface::ipc::spawn_sanitizer`, which needs
/// `DETACHED_PROCESS` too and hand-rolls `CreateProcessW`. And if a child is
/// ever *meant* to talk to the user's terminal interactively (a pager, an
/// editor, a credential prompt), do **not** call this — you'd be hiding the
/// very window the user must type into, and it'd hang forever with no way to
/// see why.
///
/// See the module docs for why `CREATE_NO_WINDOW` and not
/// `DETACHED_PROCESS | CREATE_NO_WINDOW`.
pub(crate) fn dun_popup(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unix arm must be a true no-op: same program, same args, nothing added.
    ///
    /// `Command`'s `Debug` is the only window std gives us into what it will
    /// spawn, so we compare that. Good enough — if the helper ever grows a
    /// stray `.arg()` or `.env()` on Unix, this catches it.
    #[cfg(unix)]
    #[test]
    fn unix_arm_dun_touch_the_command() {
        let mut cmd = Command::new("git");
        cmd.arg("push").arg("origin").env("GIT_TERMINAL_PROMPT", "0");
        let before = format!("{cmd:?}");

        let after = format!("{:?}", dun_popup(&mut cmd));

        assert_eq!(before, after, "dun_popup must be a no-op on unix");
    }

    /// Pin the constant we claim in the docs. If `windows-sys` ever renames or
    /// moves it, this fails loudly instead of us silently shipping some other
    /// flag.
    ///
    /// Value source: Microsoft `processthreadsapi.h` process creation flags,
    /// re-exported as `windows_sys::Win32::System::Threading::CREATE_NO_WINDOW`.
    #[cfg(windows)]
    #[test]
    fn create_no_window_is_the_flag_we_think_it_is() {
        use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

        assert_eq!(CREATE_NO_WINDOW, 0x0800_0000);
    }

    /// The flag must not break output capture — that's the whole reason we use
    /// `CREATE_NO_WINDOW` alone instead of dragging `DETACHED_PROCESS` along.
    ///
    /// Note honestly what this does and does not prove: std exposes **no
    /// getter** for creation flags, so no test can read the bit back off the
    /// `Command`. What we *can* verify is the property callers actually depend
    /// on — a windowless child still pipes its stdout back to us.
    #[cfg(windows)]
    #[test]
    fn windowless_child_still_gives_us_its_stdout() {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", "echo", "kopitiam"]);

        let output = dun_popup(&mut cmd).output().expect("cmd.exe should run");

        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "kopitiam");
    }
}
