use std::time::{Duration, Instant};

use crossbeam::channel::Sender;

use super::super::QueryResult;
use super::super::core::{Daemon, HandleOutcome};
use super::super::git_worker::GitOp;
use super::super::ipc::{AdminOp, Request, Response, ResponseExt, ResponsePayload};
use super::super::ops::OpError;
use super::super::subscription::prepare_subscription;
use super::ServerReply;
use super::spans::record_ipc_request_metric;
use super::waiters::{
    CheckpointWaiter, DurabilityWaiter, RequestOutcome, RequestWaiters, SyncWaiter,
    checkpoint_wait_ready,
};

/// Deadline the daemon applies when a `SyncWait` client didn't name one.
///
/// 60 s. Reasoning, since there is no upstream to copy this from:
/// * The common failures now answer FAST, not at the deadline — a rejected
///   push or bad credentials wakes the waiter the moment the attempt fails,
///   typically a second or two. So this number only governs the "genuinely
///   slow" case, not the "broken" case.
/// * It sits above the git-side stall abort we configure on the push
///   (`http.lowSpeedTime = 30`, see `gix_compat::subprocess_push`), so a
///   stalled HTTP transfer turns into a real *failure* with a real git message
///   inside this window, rather than an uninformative timeout.
/// * It sits below the push watchdog (120 s, same module), which only covers
///   ssh / pre-transfer hangs that git's low-speed check cannot see. In that
///   case the client times out at 60 s and honestly reports
///   `sync_in_progress: true` — accurate, and better than blocking.
/// * Timing out costs the user nothing: the daemon does NOT abandon the sync,
///   it keeps retrying on backoff. The client just stops blocking.
///
/// What would make this wrong: a store whose legitimate first push takes over a
/// minute (huge history over a thin link). That user sees a timeout report
/// while the push is still healthily running, and should pass `--timeout`.
const DEFAULT_SYNC_WAIT_TIMEOUT_MS: u64 = 60_000;

pub(super) fn process_request_message(
    daemon: &mut Daemon,
    request: Request,
    respond: Sender<ServerReply>,
    git_tx: &Sender<GitOp>,
    waiters: &mut RequestWaiters<'_>,
    request_type: &'static str,
    request_started_at: Instant,
) -> RequestOutcome {
    // Sync barrier: wait until repo is clean, the sync fails, or the deadline
    // runs out — whichever lands first. Parking with no deadline and no failure
    // path is what made `bn sync` hang forever (kopitiam#25).
    if let Request::SyncWait { ctx, payload } = request {
        match daemon.ensure_loaded_and_maybe_start_sync(&ctx.path, git_tx) {
            Ok((loaded, _started)) => {
                let repo_state = loaded.lane();
                let clean = !repo_state.dirty && !repo_state.sync_in_progress;
                let failures_at_park = repo_state.sync_failures_total;

                if clean {
                    let _ = respond.send(ServerReply::Response(Response::ok(
                        ResponsePayload::synced(),
                    )));
                    record_ipc_request_metric(request_type, request_started_at, "ok");
                } else {
                    let started_at = Instant::now();
                    let timeout = Duration::from_millis(
                        payload.timeout_ms.unwrap_or(DEFAULT_SYNC_WAIT_TIMEOUT_MS),
                    );
                    // `checked_add` guards a caller who passes a silly-large
                    // timeout: saturating to `started_at` means "expire on the
                    // next housekeeping pass", never a panic.
                    let deadline = started_at.checked_add(timeout).unwrap_or(started_at);
                    waiters
                        .sync_waiters
                        .entry(loaded.remote().clone())
                        .or_default()
                        .push(SyncWaiter {
                            respond,
                            started_at,
                            deadline,
                            failures_at_park,
                        });
                    record_ipc_request_metric(request_type, request_started_at, "wait");
                }
            }
            Err(e) => {
                let _ = respond.send(ServerReply::Response(Response::err_from(e)));
                record_ipc_request_metric(request_type, request_started_at, "err");
            }
        }
        return RequestOutcome::Continue;
    }

    if let Request::Admin(AdminOp::CheckpointWait { ctx, payload }) = request {
        let proof = match daemon.ensure_repo_loaded_strict(&ctx.path, git_tx) {
            Ok(proof) => proof,
            Err(err) => {
                let _ = respond.send(ServerReply::Response(Response::err_from(err)));
                record_ipc_request_metric(request_type, request_started_at, "err");
                return RequestOutcome::Continue;
            }
        };
        let namespace = match proof.normalize_namespace(payload.namespace) {
            Ok(namespace) => namespace,
            Err(err) => {
                let _ = respond.send(ServerReply::Response(Response::err_from(err)));
                record_ipc_request_metric(request_type, request_started_at, "err");
                return RequestOutcome::Continue;
            }
        };
        let store_id = proof.store_id();
        drop(proof);
        let min_checkpoint_wall_ms = daemon.clock().wall_ms();
        let groups = daemon.force_checkpoint_for_namespace(store_id, &namespace);
        if groups.is_empty() {
            let _ = respond.send(ServerReply::Response(Response::err_from(
                OpError::InvalidRequest {
                    field: Some("checkpoint".into()),
                    reason: format!("no checkpoint groups scheduled for namespace {namespace}",),
                },
            )));
            record_ipc_request_metric(request_type, request_started_at, "err");
            return RequestOutcome::Continue;
        }

        match checkpoint_wait_ready(
            daemon,
            store_id,
            &namespace,
            min_checkpoint_wall_ms,
            &groups,
        ) {
            Ok(Some(output)) => {
                let _ = respond.send(ServerReply::Response(Response::ok(ResponsePayload::query(
                    QueryResult::AdminCheckpoint(output),
                ))));
                record_ipc_request_metric(request_type, request_started_at, "ok");
            }
            Ok(None) => {
                waiters.checkpoint_waiters.push(CheckpointWaiter {
                    respond,
                    store_id,
                    namespace,
                    min_checkpoint_wall_ms,
                    groups,
                });
                record_ipc_request_metric(request_type, request_started_at, "wait");
            }
            Err(err) => {
                let _ = respond.send(ServerReply::Response(Response::err_from(err)));
                record_ipc_request_metric(request_type, request_started_at, "err");
            }
        }

        return RequestOutcome::Continue;
    }

    if let Request::Subscribe { ctx, .. } = request {
        match prepare_subscription(daemon, &ctx.repo.path, ctx.read, git_tx) {
            Ok(reply) => {
                let _ = respond.send(ServerReply::Subscribe(reply));
                record_ipc_request_metric(request_type, request_started_at, "ok");
            }
            Err(err) => {
                let _ = respond.send(ServerReply::Response(Response::err_from(*err)));
                record_ipc_request_metric(request_type, request_started_at, "err");
            }
        }
        return RequestOutcome::Continue;
    }

    let is_shutdown = matches!(request, Request::Shutdown);

    let outcome = daemon.handle_request(request, git_tx);
    match outcome {
        HandleOutcome::Response(response) => {
            let metric_outcome = if matches!(response, Response::Err { .. }) {
                "err"
            } else {
                "ok"
            };
            let _ = respond.send(ServerReply::Response(response));
            record_ipc_request_metric(request_type, request_started_at, metric_outcome);
        }
        HandleOutcome::DurabilityWait(wait) => {
            let started_at = Instant::now();
            let deadline = started_at
                .checked_add(wait.wait_timeout)
                .unwrap_or(started_at);
            let span = tracing::Span::current();
            waiters.durability_waiters.push(DurabilityWaiter {
                respond,
                wait,
                span,
                started_at,
                deadline,
            });
            record_ipc_request_metric(request_type, request_started_at, "wait");
        }
    }

    if is_shutdown {
        RequestOutcome::Shutdown
    } else {
        RequestOutcome::Continue
    }
}
