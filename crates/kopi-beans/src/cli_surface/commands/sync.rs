//! `bn sync` - block until the debounced git sync has flushed.
//!
//! # The hang this module exists to never repeat (kopitiam#25)
//! `bn sync` used to send `SyncWait` and block on the socket with no deadline
//! at all, while the daemon parked the request with no deadline either and had
//! no way to notice the sync was failing. A store whose push kept getting
//! rejected produced `timeout 90 bn sync` -> `Terminated`, zero output. Every
//! other command on the same store answered instantly.
//!
//! So this command now:
//! * sends an explicit deadline to the daemon (`--timeout`, default below);
//! * puts a looser watchdog on the socket read too, so even a wedged daemon
//!   cannot hold the terminal;
//! * on failure or timeout, PRINTS the lane state (dirty, sync in progress,
//!   consecutive failures, last git error, remote) and exits non-zero.
//!
//! Timing out is a report, not a cancellation: the daemon keeps retrying the
//! sync on backoff after we stop waiting.

use std::time::Duration;

use clap::Args;

use crate::core::CliErrorCode;
use crate::core::error::details::{SyncWaitDetails, SyncWaitOutcome};
use crate::surface::ipc::{Request, SyncWaitPayload};

use super::{CommandResult, print_ok};
use crate::cli_surface::runtime::{CliRuntimeCtx, RuntimeError, send_with_read_timeout};

/// Default seconds `bn sync` waits before it stops waiting and reports.
///
/// 60 s, matching the daemon's own `DEFAULT_SYNC_WAIT_TIMEOUT_MS` — see
/// `daemon::runtime::server::dispatch` for the full reasoning behind the
/// number. Sending it explicitly (rather than leaving it to the daemon)
/// means the printed "waited Ns" always matches what the user asked for,
/// including against an older daemon.
const DEFAULT_SYNC_TIMEOUT_SECS: u64 = 60;

/// Slack we allow the daemon on top of our own deadline before the CLI gives up
/// on the socket itself.
///
/// The daemon is supposed to answer at its deadline; this second watchdog only
/// fires if the state loop is wedged and never answers at all. 10 s is plenty
/// of room for a busy housekeeping pass and short enough that a wedged daemon
/// still lets go of the terminal.
const SOCKET_GRACE_SECS: u64 = 10;

#[derive(Args, Debug, Clone, Default)]
pub struct SyncArgs {
    /// Seconds to wait for the sync to flush (0 = report current state and
    /// return immediately). Timing out does not cancel the sync.
    #[arg(long, value_name = "SECS", default_value_t = DEFAULT_SYNC_TIMEOUT_SECS)]
    pub timeout: u64,
}

pub fn handle(ctx: &CliRuntimeCtx, args: SyncArgs) -> CommandResult<()> {
    let timeout_ms = args.timeout.saturating_mul(1_000);
    let req = Request::SyncWait {
        ctx: ctx.repo_ctx(),
        payload: SyncWaitPayload {
            timeout_ms: Some(timeout_ms),
        },
    };
    let socket_timeout = Duration::from_secs(args.timeout.saturating_add(SOCKET_GRACE_SECS));

    match send_with_read_timeout(&req, socket_timeout) {
        Ok(ok) => print_ok(&ok, ctx.json),
        Err(err) => {
            report_failure(&err, ctx.json);
            Err(err.into())
        }
    }
}

/// Print what the daemon told us about the lane, so the user has something to
/// act on. Never silent — a bare exit code was the worst part of kopitiam#25.
fn report_failure(err: &RuntimeError, json: bool) {
    let RuntimeError::Daemon(daemon) = err else {
        // Transport-level failure (incl. the socket watchdog firing). The
        // error's own Display already says what happened; the daemon told us
        // nothing about the lane, so there is no state to print.
        return;
    };
    let payload = daemon.payload();
    if payload.code != CliErrorCode::SyncFailed.into() {
        return;
    }
    let Ok(Some(details)) = payload.details_as::<SyncWaitDetails>() else {
        return;
    };

    if json {
        // Human block would corrupt `--json` output; the details ride along in
        // the error payload the caller already gets.
        return;
    }

    match details.outcome {
        SyncWaitOutcome::Failed => eprintln!("sync failed after {}ms", details.waited_ms),
        // `--timeout 0` is a deliberate poll, not a missed deadline. Saying
        // "did not finish in time after 0ms" would read like a bug report.
        SyncWaitOutcome::Timeout if details.waited_ms == 0 => {
            eprintln!("sync has not finished yet")
        }
        SyncWaitOutcome::Timeout => {
            eprintln!("sync did not finish in time after {}ms", details.waited_ms)
        }
    }
    eprintln!("  remote:               {}", details.remote);
    eprintln!("  dirty:                {}", details.dirty);
    eprintln!("  sync in progress:     {}", details.sync_in_progress);
    eprintln!("  consecutive failures: {}", details.consecutive_failures);
    match &details.last_error {
        Some(message) => eprintln!("  last error:           {message}"),
        None => eprintln!("  last error:           (none recorded)"),
    }
    if details.sync_in_progress {
        eprintln!("The daemon is still syncing in the background; run `bn sync` again to recheck.");
    } else if details.dirty {
        eprintln!(
            "Local changes are not on the remote yet. The daemon retries on backoff; fix the \
             error above or run `bn sync` again."
        );
    }
}
