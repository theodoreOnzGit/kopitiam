//! `bn daemon` — lifecycle control for the background daemon.
//!
//! The parent `daemon` command stays hidden, because `run` is an internal
//! plumbing detail nobody should be typing. `stop` and `restart` are a
//! different story: they are the supported cure when the daemon goes bad, so
//! their help text is written for a human in a hurry, not for a maintainer
//! reading the source.
//!
//! No stopping logic lives here. Every bit of it sits in
//! [`crate::surface::ipc::client`] — this module only decides how to say what
//! happened, in prose and in JSON.

use std::io::Write;

use clap::Subcommand;

use crate::api::DaemonInfo;
use crate::cli_surface::render::print_json;
use crate::surface::ipc::IpcError;
use crate::surface::ipc::client::{DaemonRestartOutcome, DaemonStopOutcome};

#[derive(Subcommand, Debug)]
pub enum DaemonCmd {
    /// Run the daemon in the foreground (internal).
    Run,

    /// Stop the running beads daemon.
    ///
    /// Safe to run when no daemon is running: it reports that and exits 0, so
    /// scripts and hooks can call it blind. Any command you run afterwards
    /// starts a fresh daemon automatically.
    Stop,

    /// Stop the running beads daemon and start a fresh one.
    ///
    /// Waits until the new daemon is actually serving before returning, and
    /// reports the old and new process ids so you can see it really bounced.
    /// If no daemon was running, this just starts one.
    Restart,
}

/// Runs `bn daemon stop`.
///
/// Idempotent on purpose — "already stopped" is success, not failure. See
/// [`DaemonStopOutcome`] for why that matters.
pub fn handle_stop<E>(json: bool) -> std::result::Result<(), E>
where
    E: From<IpcError>,
{
    let socket = crate::surface::ipc::client::socket_path();
    let outcome = crate::surface::ipc::client::stop_daemon_at(&socket)?;

    if json {
        print_json(&render_daemon_stop_json(&outcome, &socket))?;
        return Ok(());
    }
    write_stdout(&render_daemon_stop(&outcome))?;
    Ok(())
}

/// Runs `bn daemon restart`.
///
/// Returns only once the new daemon answers — the wait lives in
/// `ensure_daemon_at`, so by the time this prints, the next `bn` command will
/// find a daemon ready.
pub fn handle_restart<E>(json: bool) -> std::result::Result<(), E>
where
    E: From<IpcError>,
{
    let socket = crate::surface::ipc::client::socket_path();
    let outcome = crate::surface::ipc::client::restart_daemon_at(&socket)?;

    if json {
        print_json(&render_daemon_restart_json(&outcome, &socket))?;
        return Ok(());
    }
    write_stdout(&render_daemon_restart(&outcome))?;
    Ok(())
}

pub fn render_daemon_info(info: &DaemonInfo) -> String {
    format!(
        "daemon {} (protocol {}, pid {})",
        info.version, info.protocol_version, info.pid
    )
}

/// Human line for `bn daemon stop`.
///
/// The three outcomes read differently on purpose. "no daemon running" and
/// "cleared a corpse" are both exit 0, but they are not the same fact, and
/// somebody debugging a wedged tracker deserves to know which one they got.
pub fn render_daemon_stop(outcome: &DaemonStopOutcome) -> String {
    match outcome {
        DaemonStopOutcome::Stopped { pid } => format!("daemon stopped (pid {pid})"),
        DaemonStopOutcome::StaleCleaned { pid: Some(pid) } => {
            format!("no daemon running (cleared stale files left behind by pid {pid})")
        }
        DaemonStopOutcome::StaleCleaned { pid: None } => {
            "no daemon running (cleared a stale socket)".to_string()
        }
        DaemonStopOutcome::NotRunning => "no daemon running".to_string(),
    }
}

/// Human line for `bn daemon restart`.
///
/// Always names both pids when there were two, because "it restarted" with no
/// evidence is exactly the claim a user cannot check.
pub fn render_daemon_restart(outcome: &DaemonRestartOutcome) -> String {
    let started = &outcome.started;
    match outcome.stopped {
        DaemonStopOutcome::Stopped { pid: old } => format!(
            "daemon restarted (old pid {old} -> new pid {}, version {}, protocol {})",
            started.pid, started.version, started.protocol_version
        ),
        DaemonStopOutcome::StaleCleaned { pid: old } => {
            let cleared = match old {
                Some(pid) => format!(" (cleared stale files left behind by pid {pid})"),
                None => " (cleared a stale socket)".to_string(),
            };
            format!(
                "daemon started (pid {}, version {}, protocol {}); none was running before{cleared}",
                started.pid, started.version, started.protocol_version
            )
        }
        DaemonStopOutcome::NotRunning => format!(
            "daemon started (pid {}, version {}, protocol {}); none was running before",
            started.pid, started.version, started.protocol_version
        ),
    }
}

/// `--json` for `bn daemon stop`.
///
/// `action` is the field a script should branch on; `pid` is `null` whenever
/// there was no pid to name. `socket` is in there so a caller juggling several
/// runtime dirs can tell which daemon this was about.
fn render_daemon_stop_json(
    outcome: &DaemonStopOutcome,
    socket: &std::path::Path,
) -> serde_json::Value {
    serde_json::json!({
        "action": stop_action(outcome),
        "pid": outcome.pid(),
        "socket": socket_display(socket),
    })
}

/// `--json` for `bn daemon restart`.
///
/// `old_pid` / `new_pid` are lifted to the top level even though they are
/// derivable from the nested objects: proving the bounce is the main thing a
/// caller wants, and it should not need a two-level lookup to do it.
/// `old_pid` is `null` when nothing was running.
fn render_daemon_restart_json(
    outcome: &DaemonRestartOutcome,
    socket: &std::path::Path,
) -> serde_json::Value {
    serde_json::json!({
        "action": if outcome.stopped.stopped_a_live_daemon() { "restarted" } else { "started" },
        "socket": socket_display(socket),
        "old_pid": outcome.old_pid(),
        "new_pid": outcome.new_pid(),
        "stopped": {
            "action": stop_action(&outcome.stopped),
            "pid": outcome.stopped.pid(),
        },
        "started": {
            "version": outcome.started.version,
            "protocol_version": outcome.started.protocol_version,
            "pid": outcome.started.pid,
        },
    })
}

fn stop_action(outcome: &DaemonStopOutcome) -> &'static str {
    match outcome {
        DaemonStopOutcome::Stopped { .. } => "stopped",
        DaemonStopOutcome::StaleCleaned { .. } => "stale_cleaned",
        DaemonStopOutcome::NotRunning => "not_running",
    }
}

/// Socket paths in JSON always go out with forward slashes.
///
/// These tools run on Windows and on Termux/Linux, and a consumer parsing the
/// output should not have to cope with two separators depending on who ran the
/// command. Cheap to normalise here, painful to un-mix downstream.
fn socket_display(socket: &std::path::Path) -> String {
    socket.display().to_string().replace('\\', "/")
}

fn write_stdout(text: &str) -> crate::cli_surface::Result<()> {
    let mut stdout = std::io::stdout().lock();
    if let Err(err) = writeln!(stdout, "{text}")
        && err.kind() != std::io::ErrorKind::BrokenPipe
    {
        return Err(crate::surface::ipc::IpcError::Transport { source: err });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn sample_started() -> DaemonInfo {
        DaemonInfo {
            version: "0.1.4".to_string(),
            protocol_version: 3,
            pid: 9999,
            started_at_ms: None,
        }
    }

    #[test]
    fn render_daemon_info_matches_human_format() {
        let info = DaemonInfo {
            version: "0.1.26".to_string(),
            protocol_version: 3,
            pid: 4242,
            started_at_ms: None,
        };
        assert_eq!(
            render_daemon_info(&info),
            "daemon 0.1.26 (protocol 3, pid 4242)"
        );
    }

    #[test]
    fn render_stop_names_the_pid_it_killed() {
        assert_eq!(
            render_daemon_stop(&DaemonStopOutcome::Stopped { pid: 4242 }),
            "daemon stopped (pid 4242)"
        );
    }

    #[test]
    fn render_stop_not_running_reads_as_success() {
        let text = render_daemon_stop(&DaemonStopOutcome::NotRunning);
        assert_eq!(text, "no daemon running");
        // Must not read like a failure — this path exits 0.
        assert!(
            !text.contains("error"),
            "stop-when-idle must not sound like a failure: {text}"
        );
    }

    #[test]
    fn render_stop_distinguishes_stale_cleanup() {
        assert_eq!(
            render_daemon_stop(&DaemonStopOutcome::StaleCleaned { pid: Some(7) }),
            "no daemon running (cleared stale files left behind by pid 7)"
        );
        assert_eq!(
            render_daemon_stop(&DaemonStopOutcome::StaleCleaned { pid: None }),
            "no daemon running (cleared a stale socket)"
        );
    }

    #[test]
    fn render_restart_shows_both_pids() {
        let outcome = DaemonRestartOutcome {
            stopped: DaemonStopOutcome::Stopped { pid: 4242 },
            started: sample_started(),
        };
        assert_eq!(
            render_daemon_restart(&outcome),
            "daemon restarted (old pid 4242 -> new pid 9999, version 0.1.4, protocol 3)"
        );
    }

    #[test]
    fn render_restart_from_nothing_says_so() {
        let outcome = DaemonRestartOutcome {
            stopped: DaemonStopOutcome::NotRunning,
            started: sample_started(),
        };
        assert_eq!(
            render_daemon_restart(&outcome),
            "daemon started (pid 9999, version 0.1.4, protocol 3); none was running before"
        );
    }

    #[test]
    fn stop_json_shape_is_stable() {
        let value = render_daemon_stop_json(
            &DaemonStopOutcome::Stopped { pid: 4242 },
            Path::new("/run/user/1000/beads/daemon.sock"),
        );
        assert_eq!(
            value,
            serde_json::json!({
                "action": "stopped",
                "pid": 4242,
                "socket": "/run/user/1000/beads/daemon.sock",
            })
        );
    }

    #[test]
    fn stop_json_uses_null_pid_when_there_was_none() {
        let value = render_daemon_stop_json(
            &DaemonStopOutcome::NotRunning,
            Path::new("/run/user/1000/beads/daemon.sock"),
        );
        assert_eq!(value["action"], "not_running");
        assert!(
            value["pid"].is_null(),
            "pid must be null, got {}",
            value["pid"]
        );
    }

    #[test]
    fn stop_json_marks_stale_cleanup_separately() {
        let value = render_daemon_stop_json(
            &DaemonStopOutcome::StaleCleaned { pid: Some(7) },
            Path::new("/run/user/1000/beads/daemon.sock"),
        );
        assert_eq!(value["action"], "stale_cleaned");
        assert_eq!(value["pid"], 7);
    }

    #[test]
    fn restart_json_shape_is_stable() {
        let outcome = DaemonRestartOutcome {
            stopped: DaemonStopOutcome::Stopped { pid: 4242 },
            started: sample_started(),
        };
        let value =
            render_daemon_restart_json(&outcome, Path::new("/run/user/1000/beads/daemon.sock"));
        assert_eq!(
            value,
            serde_json::json!({
                "action": "restarted",
                "socket": "/run/user/1000/beads/daemon.sock",
                "old_pid": 4242,
                "new_pid": 9999,
                "stopped": { "action": "stopped", "pid": 4242 },
                "started": { "version": "0.1.4", "protocol_version": 3, "pid": 9999 },
            })
        );
    }

    #[test]
    fn restart_json_from_nothing_has_null_old_pid_and_started_action() {
        let outcome = DaemonRestartOutcome {
            stopped: DaemonStopOutcome::NotRunning,
            started: sample_started(),
        };
        let value =
            render_daemon_restart_json(&outcome, Path::new("/run/user/1000/beads/daemon.sock"));
        assert_eq!(value["action"], "started");
        assert!(value["old_pid"].is_null());
        assert_eq!(value["new_pid"], 9999);
    }

    #[test]
    fn json_socket_path_is_forward_slashed() {
        let value = render_daemon_stop_json(
            &DaemonStopOutcome::NotRunning,
            Path::new(r"C:\Users\bob\AppData\Local\Temp\beads\daemon.sock"),
        );
        assert_eq!(
            value["socket"],
            "C:/Users/bob/AppData/Local/Temp/beads/daemon.sock"
        );
    }
}
