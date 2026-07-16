//! Shell integration: running an external command and feeding text through it.
//!
//! # Why a trait, not a bare `std::process::Command` at the call site
//!
//! The four vim shell surfaces — `:!{cmd}` (run and show), `:r !{cmd}` (read
//! command output into the buffer), `:{range}!{cmd}` (filter a range through a
//! command) and the normal-mode `!{motion}` filter operator — all reduce to
//! one primitive: *give a command some standard input, get its standard output,
//! standard error and exit status back*. [`CommandRunner`] is that primitive.
//!
//! Putting it behind a trait is the same move [`super::clipboard`] makes for the
//! system clipboard: the editor core stays testable without a shell. Unit tests
//! inject a scripted runner ([`FnRunner`] under `#[cfg(test)]`) so `:%!sort`
//! reordering a buffer can be asserted *deterministically*, with no dependency
//! on which tools happen to be installed on the test machine. Production wires
//! in [`ShellRunner`], which really does spawn `sh -c` (or `cmd /C` on Windows).
//!
//! # Working directory
//!
//! [`ShellRunner`] does **not** set a working directory, so a spawned command
//! inherits the editor process's current working directory — the directory kvim
//! was launched from, which for this project is the workspace root. This matches
//! vim's default, where `:!` runs relative to the editor's `:pwd`, *not* the
//! directory of the file in the current buffer. Documented here because it is a
//! deliberate choice a reader would otherwise have to reverse-engineer.

use std::io;
use std::process::{Command, Stdio};

/// The result of running one external command: its captured output streams and
/// the exit status. Bytes are decoded lossily as UTF-8 — a scientific-computing
/// workbench overwhelmingly filters text (sort, column, jq, fmt), and a stray
/// non-UTF-8 byte should degrade to `U+FFFD` rather than fail the whole filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// The command's standard output.
    pub stdout: String,
    /// The command's standard error.
    pub stderr: String,
    /// The process exit code, or `None` when the process was terminated by a
    /// signal (Unix) and therefore has no ordinary exit code. `Some(0)` is
    /// success; any other `Some(n)` is the command's own failure code.
    pub code: Option<i32>,
}

impl CommandOutput {
    /// `true` when the command exited successfully (`code == Some(0)`).
    ///
    /// A signal-terminated command (`code == None`) is treated as *not*
    /// successful, so a killed filter never silently replaces buffer text with
    /// whatever partial output it managed to emit.
    pub fn is_success(&self) -> bool {
        self.code == Some(0)
    }

    /// A short human-readable description of a non-success exit, for the
    /// statusline. Prefers the first non-empty line of stderr (that is where a
    /// tool explains itself), falling back to the numeric code.
    pub fn failure_message(&self) -> String {
        let first_err = self.stderr.lines().find(|l| !l.trim().is_empty());
        match (first_err, self.code) {
            (Some(msg), _) => msg.trim().to_string(),
            (None, Some(code)) => format!("shell command returned {code}"),
            (None, None) => "shell command was killed".to_string(),
        }
    }
}

/// Runs an external command with a given standard input. The one seam every
/// shell surface in the editor goes through — see the module docs.
pub trait CommandRunner {
    /// Runs `cmd` (a whole shell command line, as typed after `:!`) with
    /// `stdin` piped to its standard input, and collects the result.
    ///
    /// Returns `Err` only when the command could not be *started* at all (no
    /// shell, spawn refused); a command that runs and then fails is a
    /// successful call returning a [`CommandOutput`] with a non-zero
    /// [`CommandOutput::code`]. Keeping "could not start" (`Err`) distinct from
    /// "ran and failed" (`Ok` + non-zero code) is what lets the filter path
    /// refuse to touch the buffer on a spawn failure while still reporting a
    /// tool's own non-zero exit as a message.
    fn run(&self, cmd: &str, stdin: &str) -> io::Result<CommandOutput>;
}

/// The production [`CommandRunner`]: spawns the command through the platform
/// shell (`sh -c` on Unix, `cmd /C` on Windows) so that pipes, redirection,
/// globbing and `$VAR` expansion all work exactly as a user expects from vim's
/// `:!`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShellRunner;

impl ShellRunner {
    pub fn new() -> Self {
        Self
    }
}

impl CommandRunner for ShellRunner {
    fn run(&self, cmd: &str, stdin: &str) -> io::Result<CommandOutput> {
        let mut command = shell_command(cmd);
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = command.spawn()?;

        // Write stdin on a separate thread. Writing all of stdin up front and
        // only then reading stdout would deadlock the moment the command's
        // output fills the OS pipe buffer before it has finished reading its
        // input (the classic `sort` of a large buffer hangs both ways). The
        // writer thread lets `wait_with_output` drain stdout/stderr
        // concurrently; dropping the stdin handle at the end of the thread
        // closes the pipe, signalling EOF so the command can finish.
        let writer = child.stdin.take().map(|mut sink| {
            let data = stdin.to_owned();
            std::thread::spawn(move || {
                use io::Write;
                let _ = sink.write_all(data.as_bytes());
                // `sink` drops here -> stdin pipe closes -> child sees EOF.
            })
        });

        let output = child.wait_with_output()?;
        if let Some(writer) = writer {
            // A command that never reads its stdin (e.g. `:!ls`) leaves the
            // writer blocked on a full pipe until the child exits and the pipe
            // is torn down; joining after `wait_with_output` is therefore
            // always safe and never hangs.
            let _ = writer.join();
        }

        Ok(CommandOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            code: output.status.code(),
        })
    }
}

/// Builds the platform shell invocation for `cmd`. Split out so the Unix and
/// Windows spellings live in exactly one place. Unix is the priority target;
/// the Windows arm keeps the feature usable there without a second code path in
/// every caller.
fn shell_command(cmd: &str) -> Command {
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    }
}

/// Removes exactly one trailing line terminator (`\n`, or `\r\n`) from `s`.
///
/// Command output is line-oriented and conventionally ends with a single
/// trailing newline (`echo hi` emits `"hi\n"`). When that output is spliced
/// into the buffer — replacing a filtered range, or read in below a line — the
/// buffer's own line structure already supplies the separating newlines, so the
/// command's trailing one must be dropped or every filter would grow a blank
/// line. Only *one* terminator is stripped: output that deliberately ends in a
/// blank line (`"a\n\n"`) keeps that blank line.
pub fn strip_one_trailing_newline(s: &str) -> &str {
    if let Some(rest) = s.strip_suffix('\n') {
        rest.strip_suffix('\r').unwrap_or(rest)
    } else {
        s
    }
}

/// A [`CommandRunner`] backed by a closure, for hermetic editor tests. Lets a
/// test script the exact output of a "command" without spawning anything, so
/// `:%!sort` can be proven to reorder a buffer deterministically regardless of
/// the host's installed tools.
#[cfg(test)]
pub(crate) struct FnRunner<F>(pub F);

#[cfg(test)]
impl<F> CommandRunner for FnRunner<F>
where
    F: Fn(&str, &str) -> io::Result<CommandOutput>,
{
    fn run(&self, cmd: &str, stdin: &str) -> io::Result<CommandOutput> {
        (self.0)(cmd, stdin)
    }
}

#[cfg(test)]
impl CommandOutput {
    /// Test convenience: a successful command emitting `stdout` and nothing on
    /// stderr.
    pub(crate) fn ok(stdout: impl Into<String>) -> Self {
        CommandOutput { stdout: stdout.into(), stderr: String::new(), code: Some(0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_runner_filters_stdin_through_a_real_command() {
        // Uses `tr`, which is available on the Unix test box; this is the one
        // place a real spawn is exercised at the unit level. The hermetic
        // editor-level tests use `FnRunner` instead.
        let out = ShellRunner.run("tr a-z A-Z", "kopitiam\n").unwrap();
        assert!(out.is_success());
        assert_eq!(out.stdout, "KOPITIAM\n");
    }

    #[test]
    fn shell_runner_reports_a_nonzero_exit_without_erroring() {
        let out = ShellRunner.run("exit 3", "").unwrap();
        assert_eq!(out.code, Some(3));
        assert!(!out.is_success());
    }

    #[test]
    fn strip_one_trailing_newline_removes_exactly_one() {
        assert_eq!(strip_one_trailing_newline("hi\n"), "hi");
        assert_eq!(strip_one_trailing_newline("hi\r\n"), "hi");
        assert_eq!(strip_one_trailing_newline("a\n\n"), "a\n");
        assert_eq!(strip_one_trailing_newline("no-newline"), "no-newline");
        assert_eq!(strip_one_trailing_newline(""), "");
    }
}
