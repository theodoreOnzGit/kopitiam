# AID-0028: the asynchronous LSP session is a single-owner actor, and pre-ready requests reject rather than queue

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.27`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent
* **Crate:** `kopitiam-semantic` (`src/async_session.rs`, `src/session.rs`)

## Context

`docs/ai-decisions/AID-0023` established that kvim attaches a language server
the moment a served file is shown, and flagged the consequence in "What would
make this wrong": kvim's LSP client is synchronous, so the connect — and
rust-analyzer's `cachePriming` wait (~3 s even after AID-0022, tens of seconds
on a large workspace) — runs on the editor's UI thread. Opening a Rust file
therefore *hangs*. A live observation also showed a single kvim spawning three
rust-analyzer processes at once, hinting the workspace-keyed dedup is leaking.

This is the scaffold pass for the filed follow-up (`cj0.27`): build, in
`kopitiam-semantic`, a non-blocking connect + readiness model, without breaking
the existing synchronous callers (`apps/cli`'s `rename`/`code-actions`, the
`RustAnalyzerSession` test suite).

Two design questions were genuinely the maintainer's to settle and they were
absent, so both are recorded here.

## Decision

### 1. The async session is a single-owner actor, not a shared lock

`AsyncRustAnalyzerSession::spawn_async` returns immediately with a handle in
state `Connecting`. A dedicated **background thread owns a `RustAnalyzerSession`
outright**, runs the blocking connect there, and flips a shared `AtomicU8`
state to `Ready` (or `Failed`, with a recorded reason). Callers never touch the
session; they send it *jobs* — `Box<dyn FnOnce(&mut RustAnalyzerSession) + Send>`
closures — over an `mpsc` channel and receive results over a per-request reply
channel. The worker runs jobs one at a time and, on an idle tick, pumps pushed
diagnostics into the store so they surface without a caller request.

This keeps the entire existing synchronous request implementation intact and
unduplicated: the worker just runs those `&mut self` methods on the caller's
behalf. The synchronous API is untouched and still used by the CLI; the async
path is purely additive.

### 2. Requests issued before ready **reject**, they do not queue

A request on a non-`Ready` session returns `RequestError::NotReady(state)`
immediately (carrying `Connecting` or `Failed`), rather than being buffered and
replayed when the server becomes ready. The polling caller (kvim's idle tick)
treats `Connecting` as "try again next tick" and `Failed` as terminal.

## Alternatives considered

* **`Arc<Mutex<RustAnalyzerSession>>` shared between UI and a connect thread.**
  Simpler to write, but it reintroduces blocking through the back door: the
  session is a `&mut self` state machine, so a slow request (a 60 s
  `workspace/symbol` on a large repo) holds the lock and stalls an unrelated
  caller — and a UI thread that grabs the lock during the connect blocks on
  exactly the thing we set out to avoid. Rejected. The actor makes "one
  request in flight at a time" a structural fact, not a locking discipline.
* **A typed request/reply enum** (`Command::Definition { .. , reply }`, one
  variant per method). More discoverable in a debugger, but it duplicates every
  method signature into an enum and a matching dispatch arm, and every new
  request means editing three places. The boxed-closure `Job` reuses the
  session's own methods verbatim. Rejected as needless ceremony for a scaffold.
* **Queue pre-ready requests and replay them on ready.** Tempting for "issue
  gd immediately on open," but a UI already polls an idle tick, so retry is
  free there; and a queue raises questions this scaffold should not answer
  (bound? coalesce a stale hover behind three newer ones? drop on `Failed`?).
  Rejected in favour of an explicit `NotReady` the caller retries. Revisit if a
  non-polling caller ever needs fire-and-forget.
* **Pull in an async runtime (tokio/async-std).** Violates the Pure Rust Core
  "no mandatory heavy runtime" posture and is unnecessary: `std::thread` +
  `std::sync::mpsc` express a single-server actor cleanly. Rejected.

## What would make this wrong

* **If kvim needs concurrent in-flight requests to one server** (e.g. a
  background `workspace/symbol` while gd stays responsive), the one-at-a-time
  worker serialises them and a slow request delays a fast one. The underlying
  LSP client is single-consumer, so true concurrency needs a request-id
  multiplexer in `LspClient` first — a larger change than this scaffold. The
  actor boundary is the right place to add it later; nothing here forecloses it.
* **If `NotReady` retries feel laggy** because the caller only polls every few
  hundred ms, the fix is a readiness *callback/notification* (worker signals the
  UI to wake) rather than reverting to queuing.
* **Drop-while-connecting semantics.** Dropping the handle is non-blocking: it
  closes the channel and detaches the worker, which tears the half-built
  connection down on its own thread once the in-flight connect resolves (up to
  the index timeout). If a caller needs a *synchronous* "stop now and reclaim
  the process" it does not exist yet; the child is still killed via the
  session's `Drop`, just not synchronously with the handle's drop.

## Out of scope (remaining for the finished fix)

* **kvim wiring.** Re-point `kopitiam-neovim`'s synchronous attach path at this
  handle: `spawn_async` on open, poll `is_ready`/`state` on the idle tick, treat
  `RequestError::NotReady` as retry. Owned by a later pass in a crate this
  scaffold does not touch.
* **Workspace-keyed dedup (the three-rust-analyzers observation).** The
  `(server, root)` registry that should collapse duplicate spawns into one live
  session lives on the kvim side; this handle is per-call. The dedup fix belongs
  with the wiring pass.
