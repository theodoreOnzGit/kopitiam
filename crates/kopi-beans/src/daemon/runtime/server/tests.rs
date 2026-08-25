use super::ServerReply;
use super::socket::stream_event_response;
use super::spans::{ReadConsistencyTag, read_consistency_tag, request_span};
use super::waiters::{
    DurabilityWaiter, ReadGateWaiter, SyncWaiter, SyncWaiters, flush_durability_waiters,
    flush_read_gate_waiters, flush_sync_waiters, next_sync_waiter_deadline,
};
use bytes::Bytes;
use crossbeam::channel::{Receiver, Sender};
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use uuid::Uuid;

use crate::daemon::broadcast::BroadcastEvent;
use crate::daemon::remote::RemoteUrl;
use crate::daemon_core::repl::proto::WatermarkState;

use crate::daemon::core::error::details as error_details;
use crate::daemon::core::replica_roster::ReplicaEntry;
use crate::daemon::core::{
    ActorId, Applied, BeadId, BeadType, ClientRequestId, DurabilityClass, DurabilityReceipt,
    Durable, EventBytes, EventId, HeadStatus, Limits, NamespaceId, NamespacePolicy, Opaque,
    Priority, ProtocolErrorCode, ReplicaDurabilityRole, ReplicaId, ReplicaRoster, Seq0, Seq1,
    Sha256, StoreEpoch, StoreId, StoreIdentity, TxnId, Watermark, Watermarks,
};
use crate::daemon::runtime::core::{Daemon, insert_store_for_tests};
use crate::daemon::runtime::durability_coordinator::{
    DurabilityCoordinator, DurabilityRequestClaim, ReplicatedDurabilityClaim,
};
use crate::daemon::runtime::executor::DurabilityWait;
use crate::daemon::runtime::git_worker::GitOp;
use crate::daemon::runtime::ipc::{
    MutationMeta, OpResponse, ReadConsistency, Request, Response, ResponsePayload,
};
use crate::daemon::runtime::repl::PeerAckTable;
use crate::surface::ops::OpResult;

struct TestEnv {
    _temp: TempDir,
    _override: crate::daemon::paths::DataDirOverride,
    repo_path: PathBuf,
    git_tx: Sender<GitOp>,
    _git_rx: Receiver<GitOp>,
    daemon: Daemon,
}

impl TestEnv {
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        let data_dir = temp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let override_guard = crate::daemon::paths::override_data_dir_for_tests(Some(data_dir.clone()));
        let actor = ActorId::new("test@host".to_string()).unwrap();
        let mut daemon = Daemon::new(actor);
        let repo_path = temp.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();
        let store_id = StoreId::new(Uuid::from_bytes([1u8; 16]));
        let remote = RemoteUrl::new("example.com/test/repo");
        insert_store_for_tests(&mut daemon, store_id, remote, &repo_path).unwrap();
        let (git_tx, git_rx) = crossbeam::channel::unbounded();

        Self {
            _temp: temp,
            _override: override_guard,
            repo_path,
            git_tx,
            _git_rx: git_rx,
            daemon,
        }
    }
}

fn watermark(seq: u64) -> Watermark<Applied> {
    let head = if seq == 0 {
        HeadStatus::Genesis
    } else {
        HeadStatus::Known([seq as u8; 32])
    };
    Watermark::new(Seq0::new(seq), head).expect("watermark")
}

#[test]
fn request_context_extracts_create_fields() {
    let repo = PathBuf::from("/tmp/repo");
    let namespace = NamespaceId::core();
    let actor = ActorId::new("actor@example.com").unwrap();
    let client_request_id = ClientRequestId::new(Uuid::from_bytes([7u8; 16]));
    let meta = MutationMeta {
        namespace: Some(namespace.clone()),
        client_request_id: Some(client_request_id),
        actor_id: Some(actor.clone()),
        durability: None,
    };
    let request = Request::Create {
        ctx: crate::daemon::runtime::ipc::MutationCtx::new(repo.clone(), meta),
        payload: crate::daemon::runtime::ipc::CreatePayload {
            id: None,
            parent: None,
            title: "title".to_string(),
            bead_type: BeadType::Task,
            priority: Priority::MEDIUM,
            description: None,
            design: None,
            acceptance_criteria: None,
            assignee: None,
            external_ref: None,
            estimated_minutes: None,
            labels: Vec::new(),
            dependencies: Vec::new(),
        },
    };

    let info = request.info();
    assert_eq!(info.op, "create");
    assert_eq!(info.repo, Some(repo.as_path()));
    assert_eq!(info.namespace, Some(&namespace));
    assert_eq!(info.actor_id, Some(&actor));
    assert_eq!(info.client_request_id, Some(&client_request_id));
    assert!(info.read.is_none());
}

#[test]
fn request_context_extracts_show_fields() {
    let repo = PathBuf::from("/tmp/repo");
    let namespace = NamespaceId::core();
    let read = ReadConsistency {
        namespace: Some(namespace.clone()),
        require_min_seen: None,
        wait_timeout_ms: None,
    };
    let request = Request::Show {
        ctx: crate::daemon::runtime::ipc::ReadCtx::new(repo.clone(), read),
        payload: crate::daemon::runtime::ipc::IdPayload {
            id: BeadId::parse("bd-123").expect("bead id"),
        },
    };

    let info = request.info();
    assert_eq!(info.op, "show");
    assert_eq!(info.repo, Some(repo.as_path()));
    assert_eq!(info.namespace, Some(&namespace));
    assert!(info.read.is_some());
    assert_eq!(
        info.read.map(read_consistency_tag),
        Some(ReadConsistencyTag::Default)
    );
}

#[test]
fn request_span_includes_schema_fields() {
    use crate::daemon::telemetry::schema;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use tracing::Subscriber;
    use tracing::field::{Field, Visit};
    use tracing_subscriber::Registry;
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    #[derive(Default)]
    struct FieldVisitor {
        fields: BTreeMap<String, String>,
    }

    impl FieldVisitor {
        fn record(&mut self, field: &Field, value: String) {
            self.fields.insert(field.name().to_string(), value);
        }
    }

    impl Visit for FieldVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.record(field, format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.record(field, value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.record(field, value.to_string());
        }
    }

    #[derive(Default)]
    struct SpanFields {
        fields: BTreeMap<String, String>,
    }

    struct CaptureLayer {
        spans: Arc<Mutex<Vec<BTreeMap<String, String>>>>,
    }

    impl CaptureLayer {
        fn new(spans: Arc<Mutex<Vec<BTreeMap<String, String>>>>) -> Self {
            Self { spans }
        }
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::Id,
            ctx: Context<'_, S>,
        ) {
            let mut visitor = FieldVisitor::default();
            attrs.record(&mut visitor);
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(SpanFields {
                    fields: visitor.fields,
                });
            }
        }

        fn on_record(
            &self,
            id: &tracing::Id,
            values: &tracing::span::Record<'_>,
            ctx: Context<'_, S>,
        ) {
            if let Some(span) = ctx.span(id) {
                let mut visitor = FieldVisitor::default();
                values.record(&mut visitor);
                let mut extensions = span.extensions_mut();
                if extensions.get_mut::<SpanFields>().is_none() {
                    extensions.insert(SpanFields::default());
                }
                let fields = extensions.get_mut::<SpanFields>().expect("span fields");
                fields.fields.extend(visitor.fields);
            }
        }

        fn on_close(&self, id: tracing::Id, ctx: Context<'_, S>) {
            let Some(span) = ctx.span(&id) else {
                return;
            };
            if span.metadata().name() != "ipc_request" {
                return;
            }
            let fields = span
                .extensions()
                .get::<SpanFields>()
                .map(|fields| fields.fields.clone())
                .unwrap_or_default();
            self.spans.lock().expect("span capture").push(fields);
        }
    }

    let spans = Arc::new(Mutex::new(Vec::new()));
    let layer = CaptureLayer::new(spans.clone());
    let subscriber = Registry::default().with(layer);

    tracing::dispatcher::with_default(&tracing::Dispatch::new(subscriber), || {
        let repo = PathBuf::from("/tmp/repo");
        let meta = MutationMeta {
            namespace: Some(NamespaceId::core()),
            client_request_id: Some(ClientRequestId::new(Uuid::from_bytes([7u8; 16]))),
            actor_id: Some(ActorId::new("actor@example.com").unwrap()),
            durability: None,
        };
        let request = Request::Create {
            ctx: crate::daemon::runtime::ipc::MutationCtx::new(repo.clone(), meta),
            payload: crate::daemon::runtime::ipc::CreatePayload {
                id: None,
                parent: None,
                title: "title".to_string(),
                bead_type: BeadType::Task,
                priority: Priority::MEDIUM,
                description: None,
                design: None,
                acceptance_criteria: None,
                assignee: None,
                external_ref: None,
                estimated_minutes: None,
                labels: Vec::new(),
                dependencies: Vec::new(),
            },
        };
        let info = request.info();
        let span = request_span(&info);
        let _guard = span.enter();
    });

    let captured = spans.lock().expect("span capture");
    let fields = captured.last().cloned().unwrap_or_default();
    for key in [
        schema::REQUEST_TYPE,
        schema::REPO,
        schema::NAMESPACE,
        schema::ACTOR_ID,
        schema::CLIENT_REQUEST_ID,
    ] {
        assert!(
            fields.contains_key(key),
            "ipc_request span missing {key}: {fields:?}"
        );
    }
}

#[test]
fn stream_event_decode_failure_is_wal_corrupt() {
    let namespace = NamespaceId::core();
    let origin = ReplicaId::new(Uuid::from_bytes([7u8; 16]));
    let event_id = EventId::new(origin, namespace.clone(), Seq1::from_u64(1).unwrap());
    let bytes = EventBytes::<Opaque>::new(Bytes::from(vec![0x01]));
    let event = BroadcastEvent::new(event_id, Sha256([0u8; 32]), None, bytes);

    let response = stream_event_response(event, &Limits::default());
    let Response::Err { err } = response else {
        panic!("expected corruption error");
    };
    assert_eq!(err.code, ProtocolErrorCode::WalCorrupt.into());
    let details = err
        .details_as::<error_details::WalCorruptDetails>()
        .unwrap()
        .expect("details");
    assert_eq!(details.namespace, namespace);
    assert!(!details.reason.is_empty());
}

#[test]
fn read_gate_waiter_releases_on_apply() {
    let mut env = TestEnv::new();
    let loaded = env
        .daemon
        .ensure_repo_fresh(&env.repo_path, &env.git_tx)
        .unwrap();
    let origin = loaded.runtime().meta.replica_id;
    let namespace = NamespaceId::core();

    let mut required = Watermarks::<Applied>::new();
    let required_wm = watermark(1);
    required
        .observe_at_least(&namespace, &origin, required_wm.seq(), required_wm.head())
        .unwrap();

    let read = ReadConsistency {
        namespace: Some(namespace.clone()),
        require_min_seen: Some(required),
        wait_timeout_ms: Some(200),
    };
    let request = Request::Status {
        ctx: crate::daemon::runtime::ipc::ReadCtx::new(env.repo_path.clone(), read.clone()),
        payload: crate::daemon::runtime::ipc::EmptyPayload {},
    };
    let normalized = loaded.read_scope(read).unwrap();
    drop(loaded);
    let (respond_tx, respond_rx) = crossbeam::channel::bounded(1);
    let started_at = Instant::now();
    let deadline = started_at + Duration::from_millis(normalized.wait_timeout_ms());
    let waiter = ReadGateWaiter {
        request,
        respond: respond_tx,
        repo: env.repo_path.clone(),
        read: normalized,
        span: tracing::Span::none(),
        started_at,
        deadline,
    };

    let mut waiters = vec![waiter];
    let mut sync_waiters = HashMap::new();
    let mut checkpoint_waiters = Vec::new();
    let mut durability_waiters = Vec::new();
    flush_read_gate_waiters(
        &mut env.daemon,
        &mut waiters,
        &env.git_tx,
        &mut sync_waiters,
        &mut checkpoint_waiters,
        &mut durability_waiters,
    );
    assert_eq!(waiters.len(), 1);
    assert!(respond_rx.try_recv().is_err());

    let applied_wm = watermark(1);
    let mut loaded = env
        .daemon
        .ensure_repo_fresh(&env.repo_path, &env.git_tx)
        .unwrap();
    loaded
        .runtime_mut()
        .watermarks_applied
        .observe_at_least(&namespace, &origin, applied_wm.seq(), applied_wm.head())
        .unwrap();
    drop(loaded);

    flush_read_gate_waiters(
        &mut env.daemon,
        &mut waiters,
        &env.git_tx,
        &mut sync_waiters,
        &mut checkpoint_waiters,
        &mut durability_waiters,
    );
    assert!(waiters.is_empty());

    let reply = respond_rx.recv().unwrap();
    let ServerReply::Response(response) = reply else {
        panic!("expected response");
    };
    assert!(matches!(response, Response::Ok { .. }));
}

#[test]
fn read_gate_waiter_times_out() {
    let mut env = TestEnv::new();
    let loaded = env
        .daemon
        .ensure_repo_fresh(&env.repo_path, &env.git_tx)
        .unwrap();
    let origin = loaded.runtime().meta.replica_id;
    let namespace = NamespaceId::core();

    let mut required = Watermarks::<Applied>::new();
    let required_wm = watermark(1);
    required
        .observe_at_least(&namespace, &origin, required_wm.seq(), required_wm.head())
        .unwrap();

    let read = ReadConsistency {
        namespace: Some(namespace.clone()),
        require_min_seen: Some(required),
        wait_timeout_ms: Some(10),
    };
    let request = Request::Status {
        ctx: crate::daemon::runtime::ipc::ReadCtx::new(env.repo_path.clone(), read.clone()),
        payload: crate::daemon::runtime::ipc::EmptyPayload {},
    };
    let normalized = loaded.read_scope(read).unwrap();
    drop(loaded);
    let (respond_tx, respond_rx) = crossbeam::channel::bounded(1);
    let started_at = Instant::now() - Duration::from_millis(20);
    let deadline = started_at;
    let waiter = ReadGateWaiter {
        request,
        respond: respond_tx,
        repo: env.repo_path.clone(),
        read: normalized,
        span: tracing::Span::none(),
        started_at,
        deadline,
    };

    let mut waiters = vec![waiter];
    let mut sync_waiters = HashMap::new();
    let mut checkpoint_waiters = Vec::new();
    let mut durability_waiters = Vec::new();
    flush_read_gate_waiters(
        &mut env.daemon,
        &mut waiters,
        &env.git_tx,
        &mut sync_waiters,
        &mut checkpoint_waiters,
        &mut durability_waiters,
    );
    assert!(waiters.is_empty());

    let reply = respond_rx.recv().unwrap();
    let ServerReply::Response(response) = reply else {
        panic!("expected response");
    };
    let Response::Err { err } = response else {
        panic!("expected timeout error");
    };
    assert_eq!(err.code, ProtocolErrorCode::RequireMinSeenTimeout.into());
}

fn replica(seed: u128) -> ReplicaId {
    ReplicaId::new(Uuid::from_u128(seed))
}

fn roster(entries: Vec<ReplicaEntry>) -> ReplicaRoster {
    ReplicaRoster { replicas: entries }
}

#[test]
fn durability_waiter_releases_on_quorum() {
    let namespace = NamespaceId::core();
    let local = replica(1);
    let peer_a = replica(2);
    let peer_b = replica(3);
    let roster = roster(vec![
        ReplicaEntry {
            replica_id: local,
            name: "local".to_string(),
            role: ReplicaDurabilityRole::anchor(true),
            allowed_namespaces: None,
            expire_after_ms: None,
        },
        ReplicaEntry {
            replica_id: peer_a,
            name: "peer-a".to_string(),
            role: ReplicaDurabilityRole::peer(true),
            allowed_namespaces: None,
            expire_after_ms: None,
        },
        ReplicaEntry {
            replica_id: peer_b,
            name: "peer-b".to_string(),
            role: ReplicaDurabilityRole::peer(true),
            allowed_namespaces: None,
            expire_after_ms: None,
        },
    ]);

    let mut policies = BTreeMap::new();
    policies.insert(namespace.clone(), NamespacePolicy::core_default());

    let peer_acks = Arc::new(Mutex::new(PeerAckTable::new()));
    let coordinator =
        DurabilityCoordinator::new(local, policies, Some(roster), Arc::clone(&peer_acks));

    let mut durable: WatermarkState<Durable> = BTreeMap::new();
    durable.entry(namespace.clone()).or_default().insert(
        local,
        Watermark::new(Seq0::new(2), HeadStatus::Known([2u8; 32])).unwrap(),
    );
    peer_acks
        .lock()
        .unwrap()
        .update_peer(peer_a, &durable, None, 10)
        .unwrap();
    peer_acks
        .lock()
        .unwrap()
        .update_peer(peer_b, &durable, None, 12)
        .unwrap();

    let store = StoreIdentity::new(StoreId::new(Uuid::from_u128(10)), StoreEpoch::ZERO);
    let receipt = DurabilityReceipt::local_fsync_defaults(
        store,
        TxnId::new(Uuid::from_u128(11)),
        Vec::new(),
        123,
    );
    let bead_id = BeadId::parse("bd-abc").unwrap();
    let response = OpResponse::new(OpResult::Updated { id: bead_id }, receipt);
    let wait = DurabilityWait {
        coordinator,
        namespace: namespace.clone(),
        origin: local,
        seq: Seq1::from_u64(2).unwrap(),
        claim: DurabilityRequestClaim::Replicated(ReplicatedDurabilityClaim {
            k: NonZeroU32::new(2).unwrap(),
            eligible: [peer_a, peer_b].into_iter().collect(),
        }),
        wait_timeout: Duration::from_millis(50),
        response,
    };

    let (respond_tx, respond_rx) = crossbeam::channel::bounded(1);
    let started_at = Instant::now();
    let deadline = started_at + Duration::from_millis(50);
    let mut waiters = vec![DurabilityWaiter {
        respond: respond_tx,
        wait,
        span: tracing::Span::none(),
        started_at,
        deadline,
    }];

    flush_durability_waiters(&mut waiters);
    assert!(waiters.is_empty());

    let reply = respond_rx.recv().unwrap();
    let ServerReply::Response(response) = reply else {
        panic!("expected response");
    };
    let Response::Ok { ok } = response else {
        panic!("expected ok response");
    };
    let ResponsePayload::Op(op) = ok else {
        panic!("expected op response");
    };

    assert!(op.receipt.outcome().is_achieved());
    assert_eq!(
        op.receipt.outcome().requested(),
        DurabilityClass::ReplicatedFsync {
            k: NonZeroU32::new(2).unwrap()
        }
    );
    assert_eq!(
        op.receipt.outcome().achieved(),
        Some(DurabilityClass::ReplicatedFsync {
            k: NonZeroU32::new(2).unwrap()
        })
    );

    let proof = op
        .receipt
        .durability_proof()
        .replicated
        .as_ref()
        .expect("replicated proof");
    assert_eq!(proof.k.get(), 2);
    assert_eq!(proof.acked_by.len(), 2);
    assert!(proof.acked_by.contains(&peer_a));
    assert!(proof.acked_by.contains(&peer_b));
}

#[test]
fn durability_waiter_times_out() {
    let namespace = NamespaceId::core();
    let local = replica(5);
    let peer = replica(6);
    let roster = roster(vec![
        ReplicaEntry {
            replica_id: local,
            name: "local".to_string(),
            role: ReplicaDurabilityRole::anchor(true),
            allowed_namespaces: None,
            expire_after_ms: None,
        },
        ReplicaEntry {
            replica_id: peer,
            name: "peer".to_string(),
            role: ReplicaDurabilityRole::peer(true),
            allowed_namespaces: None,
            expire_after_ms: None,
        },
    ]);

    let mut policies = BTreeMap::new();
    policies.insert(namespace.clone(), NamespacePolicy::core_default());

    let peer_acks = Arc::new(Mutex::new(PeerAckTable::new()));
    let coordinator =
        DurabilityCoordinator::new(local, policies, Some(roster), Arc::clone(&peer_acks));

    let store = StoreIdentity::new(StoreId::new(Uuid::from_u128(12)), StoreEpoch::ZERO);
    let receipt = DurabilityReceipt::local_fsync_defaults(
        store,
        TxnId::new(Uuid::from_u128(13)),
        Vec::new(),
        123,
    );
    let bead_id = BeadId::parse("bd-def").unwrap();
    let response = OpResponse::new(OpResult::Updated { id: bead_id }, receipt);
    let wait = DurabilityWait {
        coordinator,
        namespace: namespace.clone(),
        origin: local,
        seq: Seq1::from_u64(1).unwrap(),
        claim: DurabilityRequestClaim::Replicated(ReplicatedDurabilityClaim {
            k: NonZeroU32::new(1).unwrap(),
            eligible: [peer].into_iter().collect(),
        }),
        wait_timeout: Duration::from_millis(10),
        response,
    };

    let (respond_tx, respond_rx) = crossbeam::channel::bounded(1);
    let started_at = Instant::now() - Duration::from_millis(20);
    let deadline = started_at;
    let mut waiters = vec![DurabilityWaiter {
        respond: respond_tx,
        wait,
        span: tracing::Span::none(),
        started_at,
        deadline,
    }];

    flush_durability_waiters(&mut waiters);
    assert!(waiters.is_empty());

    let reply = respond_rx.recv().unwrap();
    let ServerReply::Response(response) = reply else {
        panic!("expected response");
    };
    let Response::Err { err } = response else {
        panic!("expected error");
    };
    assert_eq!(err.code, ProtocolErrorCode::DurabilityTimeout.into());
    let receipt = err
        .receipt_as::<DurabilityReceipt>()
        .unwrap()
        .expect("receipt");
    assert!(receipt.outcome().is_pending());
}

// =============================================================================
// SyncWait waiters - kopitiam#25
// =============================================================================

/// The remote `TestEnv` registers its one store against.
fn test_remote() -> RemoteUrl {
    RemoteUrl::new("example.com/test/repo")
}

/// Put the lane into "a sync is running right now" — dirty work was picked up,
/// `start_sync()` cleared `dirty` and set `sync_in_progress`.
fn begin_sync(env: &mut TestEnv) {
    let mut loaded = env
        .daemon
        .ensure_repo_fresh(&env.repo_path, &env.git_tx)
        .unwrap();
    let lane = loaded.lane_mut();
    lane.mark_dirty();
    lane.start_sync();
}

fn park_sync_waiter(
    env: &TestEnv,
    deadline: Instant,
) -> (SyncWaiters, Receiver<ServerReply>) {
    let (respond_tx, respond_rx) = crossbeam::channel::bounded(1);
    let failures_at_park = env
        .daemon
        .git_lane_state_by_url(&test_remote())
        .expect("lane")
        .sync_failures_total;
    let mut waiters: SyncWaiters = HashMap::new();
    waiters.entry(test_remote()).or_default().push(SyncWaiter {
        respond: respond_tx,
        started_at: Instant::now(),
        deadline,
        failures_at_park,
    });
    (waiters, respond_rx)
}

fn expect_sync_wait_error(reply: ServerReply) -> error_details::SyncWaitDetails {
    let ServerReply::Response(response) = reply else {
        panic!("expected a response");
    };
    let Response::Err { err } = response else {
        panic!("expected an error response, got {response:?}");
    };
    assert_eq!(err.code, crate::daemon::core::CliErrorCode::SyncFailed.into());
    err.details_as::<error_details::SyncWaitDetails>()
        .expect("details decode")
        .expect("sync wait details present")
}

/// **The regression test for kopitiam#25.**
///
/// A failing sync sets `dirty = true` so the scheduler retries it. That made
/// the old release condition (`!dirty && !sync_in_progress`) permanently false,
/// so the parked `bn sync` client was never answered — for any number of
/// failures, forever. Here the sync fails once and the waiter must be woken
/// with a verdict.
#[test]
fn failing_sync_wakes_a_parked_sync_waiter() {
    let mut env = TestEnv::new();
    begin_sync(&mut env);

    // Deadline far away, so anything that happens here is the FAILURE path and
    // not the timeout path quietly covering for it.
    let (mut waiters, respond_rx) = park_sync_waiter(&env, Instant::now() + Duration::from_secs(3600));

    // Sync still running: the waiter must stay put. Waking here would be a
    // premature answer, not a fix.
    flush_sync_waiters(&env.daemon, &mut waiters);
    assert_eq!(waiters.values().flatten().count(), 1);
    assert!(respond_rx.try_recv().is_err());

    // The sync blows up. This is the exact state that used to strand the
    // waiter: sync_in_progress back to false, dirty back to true.
    {
        let mut loaded = env
            .daemon
            .ensure_repo_fresh(&env.repo_path, &env.git_tx)
            .unwrap();
        loaded
            .lane_mut()
            .fail_sync("remote rejected refs/dolt/data".to_string(), 1234);
    }
    let lane_is_deadlocky = {
        let lane = env.daemon.git_lane_state_by_url(&test_remote()).unwrap();
        lane.dirty && !lane.sync_in_progress
    };
    assert!(
        lane_is_deadlocky,
        "precondition: a failed sync must leave dirty=true, which is what the old \
         release condition could never satisfy"
    );

    flush_sync_waiters(&env.daemon, &mut waiters);
    assert!(
        waiters.values().flatten().count() == 0,
        "a failed sync must release the waiter, not strand it"
    );

    let details = expect_sync_wait_error(respond_rx.recv().unwrap());
    assert_eq!(details.outcome, error_details::SyncWaitOutcome::Failed);
    assert_eq!(details.remote, "example.com/test/repo");
    assert!(details.dirty, "report must admit the store is still dirty");
    assert!(!details.sync_in_progress);
    assert_eq!(details.consecutive_failures, 1);
    assert_eq!(
        details.last_error.as_deref(),
        Some("remote rejected refs/dolt/data"),
        "the git error is the whole point - without it the user learns nothing"
    );
}

/// A waiter that parked BEFORE an earlier failure must not be woken by that old
/// failure — only by one that happens after it parked.
#[test]
fn sync_waiter_ignores_failures_that_predate_it() {
    let mut env = TestEnv::new();
    begin_sync(&mut env);
    {
        let mut loaded = env
            .daemon
            .ensure_repo_fresh(&env.repo_path, &env.git_tx)
            .unwrap();
        loaded.lane_mut().fail_sync("old news".to_string(), 1);
        // Retry picked up, running again.
        loaded.lane_mut().start_sync();
    }

    let (mut waiters, respond_rx) =
        park_sync_waiter(&env, Instant::now() + Duration::from_secs(3600));
    flush_sync_waiters(&env.daemon, &mut waiters);

    assert_eq!(waiters.values().flatten().count(), 1);
    assert!(
        respond_rx.try_recv().is_err(),
        "a stale failure must not answer a waiter parked after it"
    );
}

/// Deadline path: nothing fails, nothing succeeds, the clock just runs out. The
/// waiter must still be answered, and the answer must carry the live lane state
/// so the user can see the sync is genuinely still running.
#[test]
fn sync_waiter_times_out_and_reports_the_lane_state() {
    let mut env = TestEnv::new();
    begin_sync(&mut env);

    // Deadline already in the past - same trick the read-gate timeout test uses.
    let (mut waiters, respond_rx) = park_sync_waiter(&env, Instant::now() - Duration::from_millis(1));

    flush_sync_waiters(&env.daemon, &mut waiters);
    assert_eq!(waiters.values().flatten().count(), 0);

    let details = expect_sync_wait_error(respond_rx.recv().unwrap());
    assert_eq!(details.outcome, error_details::SyncWaitOutcome::Timeout);
    assert!(
        details.sync_in_progress,
        "timing out must not pretend the daemon stopped working"
    );
    assert_eq!(details.consecutive_failures, 0);
    assert_eq!(details.last_error, None);
}

/// The happy path must be untouched: lane goes clean, waiter gets `ok`.
#[test]
fn clean_lane_still_releases_sync_waiters_with_ok() {
    let mut env = TestEnv::new();
    begin_sync(&mut env);
    let (mut waiters, respond_rx) =
        park_sync_waiter(&env, Instant::now() + Duration::from_secs(3600));

    {
        let mut loaded = env
            .daemon
            .ensure_repo_fresh(&env.repo_path, &env.git_tx)
            .unwrap();
        loaded.lane_mut().complete_sync(99);
    }

    flush_sync_waiters(&env.daemon, &mut waiters);
    assert_eq!(waiters.values().flatten().count(), 0);

    let ServerReply::Response(response) = respond_rx.recv().unwrap() else {
        panic!("expected a response");
    };
    assert!(matches!(response, Response::Ok { .. }));
}

/// The state loop selects on the earliest deadline; if sync waiters were left
/// out of that calculation the loop could park on `never()` and no one would
/// ever expire them.
#[test]
fn next_sync_waiter_deadline_is_the_earliest_one() {
    let env = TestEnv::new();
    let now = Instant::now();
    let (respond_a, _rx_a) = crossbeam::channel::bounded(1);
    let (respond_b, _rx_b) = crossbeam::channel::bounded(1);
    let soon = now + Duration::from_secs(5);
    let later = now + Duration::from_secs(500);

    let mut waiters: SyncWaiters = HashMap::new();
    let list = waiters.entry(test_remote()).or_default();
    list.push(SyncWaiter {
        respond: respond_a,
        started_at: now,
        deadline: later,
        failures_at_park: 0,
    });
    list.push(SyncWaiter {
        respond: respond_b,
        started_at: now,
        deadline: soon,
        failures_at_park: 0,
    });

    assert_eq!(next_sync_waiter_deadline(&waiters), Some(soon));
    assert_eq!(next_sync_waiter_deadline(&HashMap::new()), None);
    drop(env);
}
