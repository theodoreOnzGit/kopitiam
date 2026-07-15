//! A non-blocking, asynchronous front end over [`RustAnalyzerSession`].
//!
//! # Why this exists
//!
//! [`RustAnalyzerSession::connect`] is *synchronous*: it spawns the language
//! server and then blocks the calling thread through the whole
//! `initialize` → `initialized` → `cachePriming` handshake before it returns a
//! usable session. Even after `docs/ai-decisions/AID-0022` cut that from ~180 s
//! to ~3 s on a small crate, ~3 s is still ~3 s of a frozen UI thread — and on
//! a large multi-crate workspace the wait is tens of seconds. `kopitiam-neovim`
//! (kvim) attaches the server the instant a served file is shown
//! (AID-0023), so that stall lands squarely on the editor's UI thread and
//! opening a Rust file *hangs*.
//!
//! This module breaks that coupling. [`AsyncRustAnalyzerSession::spawn_async`]
//! returns **immediately** with a handle whose state is [`LspState::Connecting`];
//! the actual connect (and rust-analyzer's cache-priming wait) runs on a
//! dedicated background thread. The caller polls [`Self::is_ready`] /
//! [`Self::state`] and keeps working; requests issued before the server is
//! ready return [`RequestError::NotReady`] rather than blocking, and once the
//! state flips to [`LspState::Ready`] they behave exactly like the synchronous
//! session.
//!
//! # Design: a single-owner actor, not a shared lock
//!
//! `RustAnalyzerSession` (and the [`crate::lsp_client::LspClient`] beneath it)
//! is a `&mut self` state machine — every request writes to the server's stdin
//! and consumes from the one reader-thread channel, so at most one request may
//! be in flight at a time. Rather than wrap it in a `Mutex` (which would let a
//! slow request stall an unrelated caller and reintroduce blocking through the
//! back door), the session is **owned outright by one background worker
//! thread**. Callers never touch it directly: they send it *jobs* — boxed
//! closures — over a channel, and receive results back over a per-request reply
//! channel. This keeps the entire existing synchronous request implementation
//! intact and unduplicated; the worker simply runs those methods on the caller's
//! behalf, one at a time, in the order they arrive.
//!
//! Shared state between the handle and the worker is deliberately tiny: an
//! [`AtomicU8`] carrying the [`LspState`] discriminant (so [`Self::is_ready`] is
//! a lock-free load on the hot polling path) and a `Mutex<Option<String>>`
//! holding a failure reason, read only when something has already gone wrong.
//!
//! Diagnostics are *pushed* by the server as unsolicited notifications, so the
//! worker pumps them into the session's store on an idle tick
//! ([`RustAnalyzerSession::pump`]) even when no caller request is outstanding —
//! this is what lets "open a file and just read it" surface diagnostics without
//! the caller asking (see `docs/ai-decisions/AID-0023`).
//!
//! # What remains (this is a scaffold)
//!
//! * **kvim wiring.** `kopitiam-neovim`'s synchronous `LspClient::session`
//!   attach path must be re-pointed at this async handle: attach-on-open should
//!   call `spawn_async`, the idle tick should poll `is_ready`/`state`, and
//!   requests should treat [`RequestError::NotReady`] as "try again next tick".
//!   That is a separate pass in a crate this scaffold does not own.
//! * **Workspace-keyed dedup.** A single kvim was observed spawning *three*
//!   rust-analyzer processes for one workspace. The `(server, root)` registry
//!   that should collapse those into one live session lives on the kvim side;
//!   this handle is per-call, so the dedup fix belongs with the wiring pass.

use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;

use crate::edit::FileEdit;
use crate::lsp_types::{CompletionItem, Diagnostic, Hover, Location};
use crate::session::{CodeAction, RustAnalyzerSession};

/// The default upper bound on the background connect, matching
/// [`RustAnalyzerSession::connect`]'s synchronous behaviour. Reaching it flips
/// the state to [`LspState::Failed`]; it is a ceiling, not a delay — a fast
/// connect resolves the moment indexing reports done.
pub const DEFAULT_INDEX_TIMEOUT: Duration = Duration::from_secs(180);

/// How long the worker waits for a job before pumping pushed notifications
/// (diagnostics) into the store. Short enough that diagnostics feel live,
/// long enough not to spin.
const PUMP_INTERVAL: Duration = Duration::from_millis(200);

// The `LspState` discriminant as stored in the shared `AtomicU8`. Kept as bare
// constants (rather than `LspState as u8`) so the mapping is explicit and the
// unknown-value fallback in `LspState::from_u8` is deliberate.
const STATE_CONNECTING: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_FAILED: u8 = 2;

/// The lifecycle state of an [`AsyncRustAnalyzerSession`], as observed by the
/// caller. Monotonic in practice: a session goes `Connecting → Ready` on a
/// successful connect, or `Connecting → Failed` if the server cannot be spawned
/// or does not become ready within the index timeout. It never leaves `Ready`
/// or `Failed` afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspState {
    /// The background thread is still spawning the server and running the
    /// `initialize`/cache-priming handshake. Requests return
    /// [`RequestError::NotReady`].
    Connecting,
    /// The handshake completed and indexing reported done; requests now behave
    /// like the synchronous [`RustAnalyzerSession`].
    Ready,
    /// The server could not be spawned, exited during the handshake, or did not
    /// become ready before the index timeout. [`AsyncRustAnalyzerSession::error`]
    /// carries the reason.
    Failed,
}

impl LspState {
    fn from_u8(value: u8) -> Self {
        match value {
            STATE_READY => LspState::Ready,
            STATE_FAILED => LspState::Failed,
            // Any unrecognised value is treated as still-connecting: the state
            // starts at `Connecting` and only advances, so "not yet a value I
            // know" can only mean "not yet advanced".
            _ => LspState::Connecting,
        }
    }
}

/// Why a request against an [`AsyncRustAnalyzerSession`] did not produce a
/// result. Distinguishes the three cases a caller must handle differently: the
/// server is not ready yet (retry later), the worker is gone (give up), and the
/// request itself failed on a ready server (a genuine error to surface).
#[derive(Debug)]
pub enum RequestError {
    /// The session was not [`LspState::Ready`] when the request was issued —
    /// it carries the state actually observed ([`LspState::Connecting`] while
    /// the connect is in flight, or [`LspState::Failed`] if it never
    /// succeeded). A caller polling on an idle tick should treat `Connecting`
    /// as "try again shortly" and `Failed` as terminal.
    NotReady(LspState),
    /// The background worker thread has stopped (the session was dropped, or
    /// the connect failed after this request was already queued). No further
    /// requests will ever succeed.
    Disconnected,
    /// The request reached a ready server and the underlying
    /// [`RustAnalyzerSession`] call returned an error (a malformed position, a
    /// server-side failure, and so on).
    Failed(anyhow::Error),
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::NotReady(state) => {
                write!(f, "language server is not ready (state: {state:?})")
            }
            RequestError::Disconnected => write!(f, "language server worker has stopped"),
            RequestError::Failed(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RequestError::Failed(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// A unit of work handed to the worker thread: a closure that runs against the
/// owned session and reports its own result over a captured reply channel.
/// Aliased so the request plumbing does not trip clippy's `type_complexity`.
type Job = Box<dyn FnOnce(&mut RustAnalyzerSession) + Send>;

/// A handle to a rust-analyzer session that connects on a background thread, so
/// the caller never blocks on the connect.
///
/// Construct one with [`Self::spawn_async`] (or a variant), poll
/// [`Self::is_ready`] / [`Self::state`], and issue the same requests as
/// [`RustAnalyzerSession`] once ready. See the [module docs](self) for the
/// design and for what remains to wire it into kvim.
///
/// Dropping the handle is **non-blocking**: it closes the job channel and
/// detaches the worker. The worker notices the closed channel, shuts the server
/// down cleanly, and its owned session's `Drop` kills the child process. A
/// handle dropped *while still connecting* returns from `drop` immediately; the
/// worker tears the half-built connection down on its own thread once the
/// in-flight connect resolves.
pub struct AsyncRustAnalyzerSession {
    /// The current [`LspState`] discriminant. Read lock-free by [`Self::state`];
    /// written only by the worker thread.
    state: Arc<AtomicU8>,
    /// The failure reason, populated by the worker when the connect fails.
    error: Arc<Mutex<Option<String>>>,
    /// The channel onto which caller requests are pushed as [`Job`]s. Dropping
    /// it (when the handle drops) is what signals the worker to stop.
    jobs: Sender<Job>,
    /// The worker thread. Held only to keep ownership tidy; deliberately never
    /// `join`ed, so dropping the handle cannot block on a slow connect.
    _worker: JoinHandle<()>,
}

impl AsyncRustAnalyzerSession {
    /// Starts connecting to `rust-analyzer` for the project at `root` on a
    /// background thread and returns immediately. The returned handle begins in
    /// [`LspState::Connecting`].
    pub fn spawn_async(root: &Path) -> Self {
        Self::spawn_async_with_args("rust-analyzer", &[], root, DEFAULT_INDEX_TIMEOUT)
    }

    /// Like [`Self::spawn_async`], but uses `binary` as the server executable
    /// (still with no arguments and the default index timeout).
    pub fn spawn_async_with_binary(binary: &str, root: &Path) -> Self {
        Self::spawn_async_with_args(binary, &[], root, DEFAULT_INDEX_TIMEOUT)
    }

    /// The general constructor: connect to `binary` (passing `args` on its
    /// command line, for servers that need flags to speak LSP on stdio) for
    /// `root`, bounding the background connect by `index_timeout`. Returns
    /// immediately; the connect runs on a background thread.
    pub fn spawn_async_with_args(binary: &str, args: &[&str], root: &Path, index_timeout: Duration) -> Self {
        let state = Arc::new(AtomicU8::new(STATE_CONNECTING));
        let error = Arc::new(Mutex::new(None));
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();

        // Own everything the worker needs so it outlives this call: the caller's
        // `&str`/`&Path` borrows do not cross the thread boundary.
        let binary = binary.to_string();
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let root = root.to_path_buf();
        let worker_state = Arc::clone(&state);
        let worker_error = Arc::clone(&error);

        let worker = thread::spawn(move || {
            let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
            let mut session = match RustAnalyzerSession::connect_with(&binary, &arg_refs, &root, index_timeout) {
                Ok(session) => {
                    worker_state.store(STATE_READY, Ordering::SeqCst);
                    session
                }
                Err(err) => {
                    // `{:#}` renders the full anyhow context chain (e.g.
                    // "failed to spawn `rust-analyzer`: No such file …").
                    *worker_error.lock().expect("error mutex poisoned") = Some(format!("{err:#}"));
                    worker_state.store(STATE_FAILED, Ordering::SeqCst);
                    return;
                }
            };

            // Serve jobs one at a time; on an idle tick, pump pushed
            // diagnostics into the store so they surface without a caller
            // request. A closed channel (handle dropped) ends the loop.
            loop {
                match jobs_rx.recv_timeout(PUMP_INTERVAL) {
                    Ok(job) => job(&mut session),
                    Err(RecvTimeoutError::Timeout) => {
                        let _ = session.pump();
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }

            // Best-effort clean shutdown; the child is killed regardless when
            // `session` drops here (see `LspClient`'s `Drop`).
            let _ = session.shutdown();
        });

        Self { state, error, jobs: jobs_tx, _worker: worker }
    }

    /// The current lifecycle [`LspState`]. A lock-free atomic load, cheap enough
    /// to call on every UI idle tick.
    pub fn state(&self) -> LspState {
        LspState::from_u8(self.state.load(Ordering::SeqCst))
    }

    /// True once the server has connected and finished indexing, i.e. requests
    /// will be served rather than rejected with [`RequestError::NotReady`].
    pub fn is_ready(&self) -> bool {
        self.state() == LspState::Ready
    }

    /// The failure reason if the state is [`LspState::Failed`], else `None`.
    pub fn error(&self) -> Option<String> {
        self.error.lock().expect("error mutex poisoned").clone()
    }

    /// Runs `f` against the owned session on the worker thread and returns its
    /// result, or a [`RequestError`] if the session is not ready / the worker is
    /// gone. This does not block for the *connect* — it fast-returns
    /// [`RequestError::NotReady`] unless the state is already
    /// [`LspState::Ready`] — but it does block for the request's own round trip
    /// once ready, exactly as the synchronous API does.
    fn run<T, F>(&self, f: F) -> Result<T, RequestError>
    where
        T: Send + 'static,
        F: FnOnce(&mut RustAnalyzerSession) -> Result<T> + Send + 'static,
    {
        let state = self.state();
        if state != LspState::Ready {
            return Err(RequestError::NotReady(state));
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        let job: Job = Box::new(move |session| {
            // If the caller has already hung up (`reply_rx` dropped), the send
            // fails harmlessly and the result is discarded.
            let _ = reply_tx.send(f(session));
        });
        self.jobs.send(job).map_err(|_| RequestError::Disconnected)?;
        match reply_rx.recv() {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(err)) => Err(RequestError::Failed(err)),
            Err(_) => Err(RequestError::Disconnected),
        }
    }

    /// Async counterpart of [`RustAnalyzerSession::definition`].
    pub fn definition(&self, file: &Path, line: u32, character: u32) -> Result<Vec<Location>, RequestError> {
        let file = file.to_path_buf();
        self.run(move |session| session.definition(&file, line, character))
    }

    /// Async counterpart of [`RustAnalyzerSession::references`].
    pub fn references(&self, file: &Path, line: u32, character: u32, include_declaration: bool) -> Result<Vec<Location>, RequestError> {
        let file = file.to_path_buf();
        self.run(move |session| session.references(&file, line, character, include_declaration))
    }

    /// Async counterpart of [`RustAnalyzerSession::hover`].
    pub fn hover(&self, file: &Path, line: u32, character: u32) -> Result<Option<Hover>, RequestError> {
        let file = file.to_path_buf();
        self.run(move |session| session.hover(&file, line, character))
    }

    /// Async counterpart of [`RustAnalyzerSession::completion`].
    pub fn completion(&self, file: &Path, line: u32, character: u32) -> Result<Vec<CompletionItem>, RequestError> {
        let file = file.to_path_buf();
        self.run(move |session| session.completion(&file, line, character))
    }

    /// Async counterpart of [`RustAnalyzerSession::rename`].
    pub fn rename(&self, file: &Path, line: u32, character: u32, new_name: &str) -> Result<Vec<FileEdit>, RequestError> {
        let file = file.to_path_buf();
        let new_name = new_name.to_string();
        self.run(move |session| session.rename(&file, line, character, &new_name))
    }

    /// Async counterpart of [`RustAnalyzerSession::code_actions`]. The returned
    /// [`CodeAction`]s are owned and `Send`, so a caller may hold one and later
    /// hand it back to [`Self::apply_code_action`].
    pub fn code_actions(&self, file: &Path, line: u32, character: u32) -> Result<Vec<CodeAction>, RequestError> {
        let file = file.to_path_buf();
        self.run(move |session| session.code_actions(&file, line, character))
    }

    /// Async counterpart of [`RustAnalyzerSession::apply_code_action`]. Takes the
    /// action by value because it must be moved onto the worker thread.
    pub fn apply_code_action(&self, action: CodeAction) -> Result<Vec<FileEdit>, RequestError> {
        self.run(move |session| session.apply_code_action(&action))
    }

    /// Async counterpart of [`RustAnalyzerSession::did_open`].
    pub fn did_open(&self, file: &Path, text: &str) -> Result<(), RequestError> {
        let file = file.to_path_buf();
        let text = text.to_string();
        self.run(move |session| session.did_open(&file, &text))
    }

    /// Async counterpart of [`RustAnalyzerSession::did_change`].
    pub fn did_change(&self, file: &Path, text: &str) -> Result<(), RequestError> {
        let file = file.to_path_buf();
        let text = text.to_string();
        self.run(move |session| session.did_change(&file, &text))
    }

    /// Async counterpart of [`RustAnalyzerSession::diagnostics`]. Note that the
    /// worker also pumps diagnostics on its idle tick, so the store stays fresh
    /// even between calls to this.
    pub fn diagnostics(&self, file: &Path) -> Result<Vec<Diagnostic>, RequestError> {
        let file = file.to_path_buf();
        self.run(move |session| session.diagnostics(&file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The discriminant mapping is total and its unknown-value fallback is
    /// `Connecting` — a pure check of the state machine's decode with no
    /// process involved.
    #[test]
    fn state_decodes_from_u8_with_connecting_fallback() {
        assert_eq!(LspState::from_u8(STATE_CONNECTING), LspState::Connecting);
        assert_eq!(LspState::from_u8(STATE_READY), LspState::Ready);
        assert_eq!(LspState::from_u8(STATE_FAILED), LspState::Failed);
        assert_eq!(LspState::from_u8(200), LspState::Connecting);
    }

    /// Spawning a non-existent binary must resolve to [`LspState::Failed`] with
    /// a recorded reason — and requests against a failed session must return
    /// promptly rather than block. Uses no real server.
    #[test]
    fn missing_binary_transitions_to_failed() {
        let dir = tempfile::tempdir().unwrap();
        let session = AsyncRustAnalyzerSession::spawn_async_with_args(
            "kopitiam-no-such-language-server-xyz",
            &[],
            dir.path(),
            Duration::from_secs(5),
        );

        // The spawn failure is near-instant; give the worker a brief window to
        // record it rather than asserting on a hard sleep.
        let mut waited = Duration::ZERO;
        while session.state() == LspState::Connecting && waited < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(20));
            waited += Duration::from_millis(20);
        }

        assert_eq!(session.state(), LspState::Failed, "a missing binary must fail the connect");
        assert!(session.error().is_some(), "a failed connect must record why");

        // A request on a failed session returns NotReady(Failed), not a hang.
        let err = session
            .definition(&dir.path().join("does_not_matter.rs"), 0, 0)
            .expect_err("a request on a failed session must error, not block");
        assert!(matches!(err, RequestError::NotReady(LspState::Failed)), "got {err:?}");
    }

    /// A request issued while the session is still connecting must return
    /// [`RequestError::NotReady`] immediately, and dropping a still-connecting
    /// handle must not block. Uses `sleep` as a stand-in server: it spawns
    /// cleanly but never speaks LSP, so the session stays `Connecting` for the
    /// window this test needs.
    #[test]
    fn requests_before_ready_return_not_ready_without_blocking() {
        if !crate::lsp_client::binary_available("sleep") {
            eprintln!("`sleep` not available; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let session = AsyncRustAnalyzerSession::spawn_async_with_args(
            "sleep",
            &["5"],
            dir.path(),
            Duration::from_secs(30),
        );

        // The atomic starts at Connecting and only advances after connect
        // returns, which cannot have happened yet.
        assert_eq!(session.state(), LspState::Connecting);

        let err = session
            .hover(&dir.path().join("x.rs"), 0, 0)
            .expect_err("a request before ready must error, not block");
        assert!(matches!(err, RequestError::NotReady(LspState::Connecting)), "got {err:?}");

        // Dropping mid-connect must return immediately (non-blocking teardown).
        let start = std::time::Instant::now();
        drop(session);
        assert!(start.elapsed() < Duration::from_secs(1), "drop blocked on the connect");
    }

    /// End-to-end proof against a live `rust-analyzer` that `spawn_async`
    /// returns *before* indexing finishes and the handle later flips to
    /// [`LspState::Ready`], after which a definition request resolves — the
    /// whole point of the async path.
    ///
    /// `#[ignore]`d: spawns a real server and waits for it to index. Run with:
    ///
    /// ```text
    /// cargo test --release -p kopitiam-semantic -- --ignored live_spawn_async
    /// ```
    #[test]
    #[ignore = "spawns a real rust-analyzer and waits for indexing; run with `-- --ignored`"]
    fn live_spawn_async_returns_before_ready_then_serves_definition() {
        if !crate::lsp_client::binary_available("rust-analyzer") {
            eprintln!("rust-analyzer not on PATH; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"sem_async_test\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let lib = dir.path().join("src/lib.rs");
        let source = "pub fn greet() -> &'static str {\n    \"hi\"\n}\n\npub fn caller() -> &'static str {\n    greet()\n}\n";
        std::fs::write(&lib, source).unwrap();

        // spawn_async must return effectively instantly — long before the
        // server has spawned, let alone finished indexing.
        let start = std::time::Instant::now();
        let session = AsyncRustAnalyzerSession::spawn_async(dir.path());
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "spawn_async blocked for {:?}; it must return before the connect",
            start.elapsed()
        );

        // Poll to readiness (or failure) rather than blocking on the connect.
        let mut waited = Duration::ZERO;
        while !session.is_ready() && session.state() != LspState::Failed && waited < Duration::from_secs(180) {
            thread::sleep(Duration::from_millis(100));
            waited += Duration::from_millis(100);
        }
        assert_eq!(session.state(), LspState::Ready, "never became ready; error: {:?}", session.error());

        // Once ready, requests behave like the synchronous session: the call to
        // `greet` on line 5 (col 4) resolves to its declaration on line 0.
        let locations = session.definition(&lib, 5, 4).expect("definition after ready");
        assert!(!locations.is_empty(), "the call to `greet` must resolve to a definition");
        assert_eq!(locations[0].range.start.line, 0);
    }
}
