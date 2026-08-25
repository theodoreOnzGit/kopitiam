use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crossbeam::channel::Sender;
use tracing::Span;

use super::super::QueryResult;
use super::super::core::{Daemon, ReadGateStatus, ReadScope};
use super::super::durability_coordinator::{DurabilityCoordinator, ReplicatedPoll};
use super::super::executor::DurabilityWait;
use super::super::git_worker::GitOp;
use super::super::ipc::{Request, Response, ResponseExt, ResponsePayload};
use super::super::ops::OpError;
use super::ServerReply;
use super::dispatch::process_request_message;
use crate::daemon::api::{AdminCheckpointGroup, AdminCheckpointOutput};
use crate::daemon::core::error::details as error_details;
use crate::daemon::core::error::details::SyncWaitOutcome;
use crate::daemon::core::{DurabilityClass, NamespaceId, StoreId};
use crate::daemon::remote::RemoteUrl;
use crate::daemon::runtime::metrics;

pub(super) struct ReadGateWaiter {
    pub(super) request: Request,
    pub(super) respond: Sender<ServerReply>,
    pub(super) repo: PathBuf,
    pub(super) read: ReadScope,
    pub(super) span: Span,
    pub(super) started_at: Instant,
    pub(super) deadline: Instant,
}

pub(super) struct DurabilityWaiter {
    pub(super) respond: Sender<ServerReply>,
    pub(super) wait: DurabilityWait,
    pub(super) span: Span,
    pub(super) started_at: Instant,
    pub(super) deadline: Instant,
}

/// One parked `bn sync` (`Request::SyncWait`) client.
///
/// Used to be a bare `Sender<ServerReply>` with no deadline and no way of
/// hearing about a failure — see [`flush_sync_waiters`] for the full story of
/// kopitiam#25.
pub(super) struct SyncWaiter {
    pub(super) respond: Sender<ServerReply>,
    pub(super) started_at: Instant,
    /// Hard stop. Once we are past this we answer with the lane state, we do
    /// NOT keep waiting. Note this does not cancel the sync — the daemon keeps
    /// retrying in the background; we just stop holding the client hostage.
    pub(super) deadline: Instant,
    /// `GitLaneState::sync_failures_total` read at the moment we parked.
    ///
    /// The lane's counter going ABOVE this is what "a sync failed while you
    /// were waiting" means. Snapshot-and-compare rather than a flag, because
    /// several clients can be parked on the same remote at different times and
    /// each only cares about failures after its own park.
    pub(super) failures_at_park: u64,
}

pub(super) type SyncWaiters = HashMap<RemoteUrl, Vec<SyncWaiter>>;

pub(super) struct CheckpointWaiter {
    pub(super) respond: Sender<ServerReply>,
    pub(super) store_id: StoreId,
    pub(super) namespace: NamespaceId,
    pub(super) min_checkpoint_wall_ms: u64,
    pub(super) groups: Vec<String>,
}

pub(super) enum RequestOutcome {
    Continue,
    Shutdown,
}

pub(super) struct RequestWaiters<'a> {
    pub(super) sync_waiters: &'a mut SyncWaiters,
    pub(super) checkpoint_waiters: &'a mut Vec<CheckpointWaiter>,
    pub(super) durability_waiters: &'a mut Vec<DurabilityWaiter>,
}

pub(super) fn flush_read_gate_waiters(
    daemon: &mut Daemon,
    waiters: &mut Vec<ReadGateWaiter>,
    git_tx: &Sender<GitOp>,
    sync_waiters: &mut SyncWaiters,
    checkpoint_waiters: &mut Vec<CheckpointWaiter>,
    durability_waiters: &mut Vec<DurabilityWaiter>,
) {
    if waiters.is_empty() {
        return;
    }

    let now = Instant::now();
    let mut remaining = Vec::new();
    for waiter in waiters.drain(..) {
        let request_type = waiter.request.info().op;
        let span = waiter.span.clone();
        let _guard = span.enter();
        let loaded = match daemon.ensure_repo_fresh(&waiter.repo, git_tx) {
            Ok(loaded) => loaded,
            Err(err) => {
                metrics::ipc_read_gate_wait_completed(
                    request_type,
                    "err",
                    now.duration_since(waiter.started_at),
                );
                let _ = waiter
                    .respond
                    .send(ServerReply::Response(Response::err_from(err)));
                continue;
            }
        };

        let status = loaded.read_gate_status(&waiter.read);
        drop(loaded);
        match status {
            Ok(ReadGateStatus::Satisfied) => {
                let request_started_at = waiter.started_at;
                metrics::ipc_read_gate_wait_completed(
                    request_type,
                    "satisfied",
                    now.duration_since(waiter.started_at),
                );
                let mut request_waiters = RequestWaiters {
                    sync_waiters,
                    checkpoint_waiters,
                    durability_waiters,
                };
                let _ = process_request_message(
                    daemon,
                    waiter.request,
                    waiter.respond,
                    git_tx,
                    &mut request_waiters,
                    request_type,
                    request_started_at,
                );
            }
            Ok(ReadGateStatus::Unsatisfied {
                required,
                current_applied,
            }) => {
                if now >= waiter.deadline {
                    let waited_ms = (now.duration_since(waiter.started_at).as_millis())
                        .min(u64::MAX as u128) as u64;
                    metrics::ipc_read_gate_wait_completed(
                        request_type,
                        "timeout",
                        now.duration_since(waiter.started_at),
                    );
                    let err = OpError::RequireMinSeenTimeout {
                        waited_ms,
                        required: Box::new(required),
                        current_applied: Box::new(current_applied),
                    };
                    let _ = waiter
                        .respond
                        .send(ServerReply::Response(Response::err_from(err)));
                    continue;
                }
                remaining.push(waiter);
            }
            Err(err) => {
                metrics::ipc_read_gate_wait_completed(
                    request_type,
                    "err",
                    now.duration_since(waiter.started_at),
                );
                let _ = waiter
                    .respond
                    .send(ServerReply::Response(Response::err_from(err)));
            }
        }
    }

    *waiters = remaining;
}

pub(super) fn flush_durability_waiters(waiters: &mut Vec<DurabilityWaiter>) {
    if waiters.is_empty() {
        return;
    }

    let now = Instant::now();
    let mut remaining = Vec::new();
    for waiter in waiters.drain(..) {
        let span = waiter.span.clone();
        let _guard = span.enter();
        let crate::daemon::runtime::durability_coordinator::DurabilityRequestClaim::Replicated(claim) =
            waiter.wait.claim.clone()
        else {
            let _ = waiter
                .respond
                .send(ServerReply::Response(Response::ok(ResponsePayload::Op(
                    waiter.wait.response,
                ))));
            continue;
        };
        let requested = DurabilityClass::ReplicatedFsync { k: claim.k };

        match waiter.wait.coordinator.poll_claim(
            &waiter.wait.namespace,
            waiter.wait.origin,
            waiter.wait.seq,
            &claim,
        ) {
            Ok(ReplicatedPoll::Satisfied { acked_by }) => {
                let mut response = waiter.wait.response;
                response.receipt = DurabilityCoordinator::achieved_receipt(
                    response.receipt,
                    requested,
                    claim.k,
                    acked_by,
                );
                let _ =
                    waiter
                        .respond
                        .send(ServerReply::Response(Response::ok(ResponsePayload::Op(
                            response,
                        ))));
            }
            Ok(ReplicatedPoll::Pending { acked_by, eligible }) => {
                if now >= waiter.deadline {
                    let waited_ms = (now.duration_since(waiter.started_at).as_millis())
                        .min(u64::MAX as u128) as u64;
                    let pending = DurabilityCoordinator::pending_replica_ids(&eligible, &acked_by);
                    let pending_receipt = DurabilityCoordinator::pending_receipt(
                        waiter.wait.response.receipt,
                        requested,
                        acked_by,
                    );
                    let err = OpError::DurabilityTimeout {
                        requested,
                        waited_ms,
                        pending_replica_ids: Some(pending),
                        receipt: Box::new(pending_receipt),
                    };
                    let _ = waiter
                        .respond
                        .send(ServerReply::Response(Response::err_from(err)));
                    continue;
                }
                remaining.push(waiter);
            }
            Err(err) => {
                let _ = waiter
                    .respond
                    .send(ServerReply::Response(Response::err_from(err)));
            }
        }
    }

    *waiters = remaining;
}

pub(super) fn checkpoint_wait_ready(
    daemon: &Daemon,
    store_id: StoreId,
    namespace: &NamespaceId,
    min_checkpoint_wall_ms: u64,
    groups: &[String],
) -> Result<Option<AdminCheckpointOutput>, OpError> {
    let snapshots = daemon.checkpoint_group_snapshots(store_id);
    let mut matched = Vec::new();
    for snapshot in snapshots {
        if groups.iter().any(|group| group == &snapshot.group) {
            matched.push(snapshot);
        }
    }

    if matched.len() != groups.len() {
        return Err(OpError::Internal("checkpoint group missing from scheduler"));
    }

    let ready = matched.iter().all(|snapshot| {
        !snapshot.dirty
            && !snapshot.in_flight
            && snapshot
                .last_checkpoint_wall_ms
                .is_some_and(|wall_ms| wall_ms >= min_checkpoint_wall_ms)
    });

    if !ready {
        return Ok(None);
    }

    let checkpoint_groups = matched
        .into_iter()
        .map(|snapshot| AdminCheckpointGroup {
            group: snapshot.group,
            namespaces: snapshot.namespaces,
            git_ref: snapshot.git_ref,
            dirty: snapshot.dirty,
            in_flight: snapshot.in_flight,
            last_checkpoint_wall_ms: snapshot.last_checkpoint_wall_ms,
        })
        .collect();

    Ok(Some(AdminCheckpointOutput {
        namespace: namespace.clone(),
        checkpoint_groups,
    }))
}

pub(super) fn flush_checkpoint_waiters(daemon: &Daemon, waiters: &mut Vec<CheckpointWaiter>) {
    if waiters.is_empty() {
        return;
    }

    let mut remaining = Vec::new();
    for waiter in waiters.drain(..) {
        match checkpoint_wait_ready(
            daemon,
            waiter.store_id,
            &waiter.namespace,
            waiter.min_checkpoint_wall_ms,
            &waiter.groups,
        ) {
            Ok(Some(output)) => {
                let _ = waiter.respond.send(ServerReply::Response(Response::ok(
                    ResponsePayload::query(QueryResult::AdminCheckpoint(output)),
                )));
            }
            Ok(None) => remaining.push(waiter),
            Err(err) => {
                let _ = waiter
                    .respond
                    .send(ServerReply::Response(Response::err_from(err)));
            }
        }
    }

    *waiters = remaining;
}

/// Answer every parked `bn sync` that now has an answer.
///
/// Three ways a waiter gets its answer, checked in this order:
///
/// 1. **Clean** (`!dirty && !sync_in_progress`) - the sync landed, reply `ok`.
/// 2. **Failed** - the lane's `sync_failures_total` has moved past what the
///    waiter snapshotted, i.e. a sync attempt blew up while this client was
///    parked. Reply with the error and the whole lane state.
/// 3. **Deadline** - clock ran out, reply with the lane state as it stands.
///
/// # Why a single failure wakes the waiter instead of riding out the retry
/// (kopitiam#25 — this is the decision the fix turns on.)
///
/// The lane retries a failed sync after [`GitLaneState::backoff_ms`], forever,
/// with the backoff capping at ~32s. So "keep waiting, the retry might work" is
/// a bet with no time limit on it: an auth failure or a rejected
/// non-fast-forward push fails identically on every retry, and the client waits
/// until the heat death of the universe. That is precisely the reported bug.
///
/// Waking on the first failure is the right trade because:
/// * the information is actionable NOW — the user can read the git error and go
///   fix their credentials / rebase, rather than staring at a blank terminal;
/// * nothing is lost — the daemon does NOT stop retrying just because we
///   answered. `dirty` stays set, backoff stays scheduled. The reply says
///   "still failing", not "gave up";
/// * a transient failure is cheap to re-check — the error is marked retryable
///   and `bn sync` is one command away.
///
/// What would make this wrong: if failures were overwhelmingly transient AND a
/// retry were near-instant, waking on the first one would turn a self-healing
/// blip into a scary error. Neither holds here — backoff starts at 500ms and
/// doubles, and the common real failures (auth, rejected ref, no network) are
/// all sticky.
///
/// [`GitLaneState::backoff_ms`]: crate::daemon::git_lane::GitLaneState::backoff_ms
pub(super) fn flush_sync_waiters(daemon: &Daemon, waiters: &mut SyncWaiters) {
    if waiters.is_empty() {
        return;
    }

    let now = Instant::now();
    waiters.retain(|remote, parked| {
        // No lane for this remote means nothing is pending against it, so treat
        // it as clean — same call the old code made.
        let Some(lane) = daemon.git_lane_state_by_url(remote) else {
            for waiter in parked.drain(..) {
                let _ = waiter.respond.send(ServerReply::Response(Response::ok(
                    ResponsePayload::synced(),
                )));
            }
            return false;
        };

        let clean = !lane.dirty && !lane.sync_in_progress;
        let mut remaining = Vec::with_capacity(parked.len());
        for waiter in parked.drain(..) {
            if clean {
                let _ = waiter.respond.send(ServerReply::Response(Response::ok(
                    ResponsePayload::synced(),
                )));
                continue;
            }

            let failed_since_park = lane.sync_failures_total > waiter.failures_at_park;
            let expired = now >= waiter.deadline;
            if !failed_since_park && !expired {
                remaining.push(waiter);
                continue;
            }

            let waited_ms = now.duration_since(waiter.started_at).as_millis().min(u64::MAX as u128)
                as u64;
            let outcome = if failed_since_park {
                SyncWaitOutcome::Failed
            } else {
                SyncWaitOutcome::Timeout
            };
            let last_error = lane
                .last_sync_error
                .as_ref()
                .map(|record| record.message.clone());
            let reason = match (outcome, last_error.as_deref()) {
                (SyncWaitOutcome::Failed, Some(message)) => message.to_string(),
                (SyncWaitOutcome::Failed, None) => "sync failed (no error recorded)".to_string(),
                (SyncWaitOutcome::Timeout, _) => {
                    "timed out waiting for the sync to finish".to_string()
                }
            };
            let details = error_details::SyncWaitDetails {
                outcome,
                remote: remote.as_str().to_string(),
                waited_ms,
                dirty: lane.dirty,
                sync_in_progress: lane.sync_in_progress,
                consecutive_failures: lane.consecutive_failures,
                last_error,
            };
            let err = OpError::SyncWaitFailed {
                remote: remote.as_str().to_string(),
                reason,
                waited_ms,
                details: Box::new(details),
            };
            let _ = waiter
                .respond
                .send(ServerReply::Response(Response::err_from(err)));
        }

        *parked = remaining;
        !parked.is_empty()
    });
}

/// Earliest deadline across all parked sync waiters.
///
/// The state loop needs this so `crossbeam::select!` wakes up to expire a
/// waiter. Without it the loop can block on `never()` — nothing scheduled,
/// nothing to receive — and the deadline never gets checked.
pub(super) fn next_sync_waiter_deadline(waiters: &SyncWaiters) -> Option<Instant> {
    waiters
        .values()
        .flatten()
        .map(|waiter| waiter.deadline)
        .min()
}
