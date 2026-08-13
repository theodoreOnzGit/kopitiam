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

use std::io;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

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

/// How a [`output_before_deadline`] child ended.
#[derive(Debug)]
pub(crate) enum DeadlineOutcome {
    /// Child exited on its own. Same `Output` you'd get from `Command::output`.
    Finished(Output),
    /// Deadline hit first — we killed it and reaped it. Whatever it had already
    /// written is in here, because the reader threads were running the whole
    /// time.
    TimedOut {
        waited: Duration,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
}

/// Poll floor. Start tight so a fast child (the normal case — `git push` of a
/// few beads) returns near-instantly instead of eating a fixed sleep.
const POLL_MIN: Duration = Duration::from_millis(1);
/// Poll ceiling. Past this the child is clearly not a quick one, and checking
/// more often than 20x/sec buys nothing but wakeups.
const POLL_MAX: Duration = Duration::from_millis(50);

/// Run `cmd` to completion, but kill it if it outlives `budget`.
///
/// # Why this exists rather than `Command::output()`
/// `std::process::Command` has **no wait-with-timeout**. `output()` blocks
/// until the child exits, full stop. For `git push` over a network that is a
/// hang with no upper bound: a TCP connect that never answers, a TLS handshake
/// that stalls, an ssh waiting on something. `GIT_TERMINAL_PROMPT=0` rules out
/// a credential *prompt*, but nothing else. That unbounded block was one of the
/// two ways `bn sync` hung forever (kopitiam#25).
///
/// # The pipe-buffer trap this deliberately avoids
/// The naive version — spawn, then poll `try_wait()` while the child's
/// stdout/stderr sit unread — deadlocks in a way that looks exactly like the
/// bug you were fixing. A pipe holds ~64 KiB; once the child fills it, the
/// child blocks on `write()` forever, so it never exits, so `try_wait()` never
/// returns `Some`, so we spin until the deadline and kill a child that was
/// perfectly healthy. So: one reader thread per pipe, draining to EOF, from the
/// moment we spawn. Killing the child closes the pipes, which is what lets
/// those threads finish.
///
/// # Windows
/// Goes through [`dun_popup`] itself rather than trusting the caller to have
/// done it — forgetting it brings back the console-window popup of
/// kopitiam#29, and there is no reason any caller of this helper would want a
/// window.
///
/// stdin is `/dev/null`: a child that decides to read stdin must fail fast, not
/// block until we shoot it.
pub(crate) fn output_before_deadline(
    cmd: &mut Command,
    budget: Duration,
) -> io::Result<DeadlineOutcome> {
    dun_popup(cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let started_at = Instant::now();
    let mut child = cmd.spawn()?;

    // Drain both pipes from the off. See the trap in the doc comment above.
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let stdout_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = std::io::Read::read_to_end(pipe, &mut buf);
        }
        buf
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = std::io::Read::read_to_end(pipe, &mut buf);
        }
        buf
    });

    let mut backoff = POLL_MIN;
    let finished = loop {
        match child.try_wait()? {
            Some(status) => break Some(status),
            None => {
                if started_at.elapsed() >= budget {
                    // Kill, then reap. `wait()` is what actually releases the
                    // zombie, and it returns promptly once the kill lands.
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(POLL_MAX);
            }
        }
    };

    // Safe to join now either way: the child is gone, so both pipes are at EOF.
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();

    match finished {
        Some(status) => Ok(DeadlineOutcome::Finished(Output {
            status,
            stdout,
            stderr,
        })),
        None => Ok(DeadlineOutcome::TimedOut {
            waited: started_at.elapsed(),
            stdout,
            stderr,
        }),
    }
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

    /// A child that finishes well inside its budget must come back normally,
    /// with its output intact and no wall-clock penalty from the poll loop.
    #[cfg(unix)]
    #[test]
    fn fast_child_finishes_and_keeps_its_output() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo kopitiam; echo oops >&2; exit 3"]);

        let started = Instant::now();
        let outcome = output_before_deadline(&mut cmd, Duration::from_secs(30)).expect("spawn");

        let DeadlineOutcome::Finished(output) = outcome else {
            panic!("a child that exits immediately must not be reported as timed out");
        };
        assert_eq!(output.status.code(), Some(3));
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "kopitiam");
        assert_eq!(String::from_utf8_lossy(&output.stderr).trim(), "oops");
        // The poll backoff must not add real latency to the common case.
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "fast child took {:?}, poll loop is sleeping too long",
            started.elapsed()
        );
    }

    /// The actual watchdog: a child that would hang forever gets killed at the
    /// deadline, and we get told so instead of blocking. This is the unit-level
    /// stand-in for "git push stalls on a TCP connect" — no network needed.
    #[cfg(unix)]
    #[test]
    fn stalled_child_gets_killed_at_the_deadline() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 60"]);

        let started = Instant::now();
        let outcome =
            output_before_deadline(&mut cmd, Duration::from_millis(200)).expect("spawn");

        let DeadlineOutcome::TimedOut { waited, .. } = outcome else {
            panic!("a `sleep 60` under a 200ms budget must time out");
        };
        assert!(waited >= Duration::from_millis(200), "waited {waited:?}");
        // Generous ceiling: CI boxes are slow. The point is it returned at all,
        // in roughly the budget, rather than in 60s.
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "watchdog did not return promptly: {:?}",
            started.elapsed()
        );
    }

    /// Regression guard for the pipe-buffer deadlock described in
    /// [`output_before_deadline`]'s docs.
    ///
    /// 256 KiB is comfortably past the ~64 KiB a pipe will hold, so a version
    /// that polls `try_wait()` without draining the pipes would wedge here: the
    /// child blocks writing, never exits, and we'd kill a healthy process at
    /// the deadline. Passing proves the reader threads are doing their job.
    #[cfg(unix)]
    #[test]
    fn child_output_bigger_than_the_pipe_buffer_dun_deadlock() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "dd if=/dev/zero bs=1024 count=256 2>/dev/null"]);

        let outcome = output_before_deadline(&mut cmd, Duration::from_secs(30)).expect("spawn");

        let DeadlineOutcome::Finished(output) = outcome else {
            panic!("child was killed at the deadline - the pipes are not being drained");
        };
        assert!(output.status.success());
        assert_eq!(output.stdout.len(), 256 * 1024);
    }

    /// Windows arm of the watchdog. `ping -n` is the portable "sleep" on
    /// stock Windows — no `timeout.exe` dependency, and it needs no network
    /// because it pings loopback.
    #[cfg(windows)]
    #[test]
    fn stalled_child_gets_killed_at_the_deadline() {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", "ping -n 60 127.0.0.1 >NUL"]);

        let outcome =
            output_before_deadline(&mut cmd, Duration::from_millis(500)).expect("spawn");

        assert!(
            matches!(outcome, DeadlineOutcome::TimedOut { .. }),
            "a 60s ping under a 500ms budget must time out"
        );
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
