//! kvim's live LSP client: a registry of running language servers, keyed by
//! **(server executable, workspace root)**, wrapping
//! [`kopitiam_semantic::RustAnalyzerSession`] — the transport KOPITIAM already
//! drives for `kopitiam rename` / `kopitiam code-actions`.
//!
//! # Why keyed by `(server, root)`, not by filetype
//!
//! An earlier scaffold kept `sessions: HashMap<String, _>` keyed by *filetype*,
//! i.e. at most one server per language for the whole process. That fails
//! silently the moment two projects are open at once (`:e ../other/src/lib.rs`):
//! both buffers get the *same* rust-analyzer, rooted at whichever project
//! started first, and cross-project go-to-definition resolves against the wrong
//! workspace. Keying by `(server, root)` — lazily spawning a client the first
//! time a file under a new root of a language is touched, and reusing it for
//! further files under that root — makes multiple projects and multiple servers
//! per session fall out naturally. This is the Helix infrastructure pattern
//! recorded in `docs/ai-decisions/AID-0019` (studied clean-room; no code
//! copied). See bead `kopitiam-cj0.24`/`kopitiam-cj0.12`.
//!
//! # Positions: graphemes here, chars on the wire
//!
//! kvim's [`Position`] is grapheme-indexed (see [`crate::core`]); the semantic
//! crate's public API is `char`-offset (Unicode scalar / LSP `utf-32`). This
//! module converts at exactly two seams — the query position grapheme→char on
//! the way out, and every result position char→grapheme on the way back, using
//! [`super::position`] with the target line's own text (a definition result can
//! point into a *different* file than the one queried). No caller ever sees a
//! wire encoding.
//!
//! # What `AsyncRustAnalyzerSession` gives us, despite the name
//!
//! `spawn_async_with_binary` spawns *any* executable and speaks generic LSP to
//! it, so this same session type drives `lua-language-server`, `texlab`, and
//! rust-analyzer (for both `.rs` files and `Cargo.toml`) alike — nothing about
//! it is rust-analyzer-specific once connected.
//!
//! # Non-blocking by construction: the UI thread never waits on a connect
//!
//! kvim attaches a language server the instant a served file is shown
//! (`docs/ai-decisions/AID-0023`), and that attach lands on the editor's **UI
//! thread**. The synchronous [`kopitiam_semantic::RustAnalyzerSession::connect`]
//! blocks that thread through rust-analyzer's whole `initialize` +
//! `cachePriming` handshake — ~3 s on a small crate, tens of seconds on a large
//! workspace — so opening a Rust file *froze* the editor (bead
//! `kopitiam-cj0.27`, AID-0028's "what remains").
//!
//! This client is built on [`AsyncRustAnalyzerSession`] instead: [`Self::session`]
//! calls [`AsyncRustAnalyzerSession::spawn_async_with_binary`], which returns
//! **immediately** with a handle in state [`LspState::Connecting`] while a
//! background thread runs the blocking connect. Every request method here is
//! therefore non-blocking on the connect: while the server is still coming up
//! they surface [`LspError::NotReady`] (which the UI shows as a brief "still
//! starting" note and retries on its idle tick), and once the handle flips to
//! [`LspState::Ready`] they behave exactly like the old synchronous calls. The
//! handle is inserted into the registry **at spawn time, not at ready time** —
//! that is what makes the `(server, root)` dedup hold under the async model: a
//! second file opened under the same root on the next idle tick finds the
//! still-`Connecting` handle already in the map and reuses it, rather than
//! spawning a second rust-analyzer (the "one kvim, three rust-analyzers"
//! observation in AID-0028).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use kopitiam_semantic::edit::FileEdit;
use kopitiam_semantic::{self as semantic, AsyncRustAnalyzerSession, LspState, ProgressSnapshot, RequestError};

use crate::core::{Position, Range};

use super::position::{self, PositionEncoding};
use super::registry::{self, LanguageServer};

/// The semantic crate's public positions are `char` offsets — LSP `utf-32` — so
/// that is the encoding this module converts kvim's grapheme columns to and
/// from. See the module doc comment.
const WIRE: PositionEncoding = PositionEncoding::Utf32;

/// Everything that can go wrong asking for an LSP operation.
#[derive(Debug, thiserror::Error)]
pub enum LspError {
    #[error("no language server is registered for filetype {0:?}")]
    NoServerForFiletype(String),
    /// The server for this buffer's language has been spawned but has not yet
    /// finished connecting/indexing. **Not an error to surface loudly** — the
    /// UI shows a brief "still starting" note and retries on its next idle
    /// tick. Distinguished from [`Self::Failed`] (terminal) by
    /// [`Self::is_not_ready`] so a caller can pick the right message without
    /// matching the whole enum.
    #[error("the {filetype} language server is still starting")]
    NotReady { filetype: String },
    /// The server could not be spawned, or exited during the handshake. Carries
    /// the reason the background connect recorded. Terminal for this
    /// `(server, root)`: retrying will not help until the buffer is reopened.
    #[error("the {filetype} language server failed to start: {reason}")]
    Failed { filetype: String, reason: String },
    #[error("failed to start `{executable}`: {source}")]
    Spawn {
        executable: String,
        #[source]
        source: anyhow::Error,
    },
    #[error(transparent)]
    Semantic(#[from] anyhow::Error),
}

impl LspError {
    /// Whether this is the transient [`Self::NotReady`] "server still starting"
    /// case, which a caller should treat as "try again shortly" rather than as
    /// a failure to report. Lets the UI branch on one predicate instead of
    /// re-matching the enum at every call site.
    pub fn is_not_ready(&self) -> bool {
        matches!(self, LspError::NotReady { .. })
    }
}

/// Maps a [`RequestError`] from the async session into this module's
/// [`LspError`], reading the recorded failure reason off `session` when the
/// connect has failed. Kept as a free function (rather than a `From`) because
/// it needs both the `filetype` label and the session handle in hand.
fn map_request_error(filetype: &str, session: &AsyncRustAnalyzerSession, err: RequestError) -> LspError {
    match err {
        // Still connecting: transient, the caller retries.
        RequestError::NotReady(LspState::Connecting) => LspError::NotReady { filetype: filetype.to_string() },
        // The connect failed outright; surface the recorded reason.
        RequestError::NotReady(LspState::Failed) => LspError::Failed {
            filetype: filetype.to_string(),
            reason: session.error().unwrap_or_else(|| "unknown reason".to_string()),
        },
        // `NotReady(Ready)` cannot occur (a ready session serves the request),
        // but the match must be total; treat it as transient.
        RequestError::NotReady(LspState::Ready) => LspError::NotReady { filetype: filetype.to_string() },
        RequestError::Disconnected => LspError::Failed {
            filetype: filetype.to_string(),
            reason: "the language server worker thread stopped".to_string(),
        },
        // A genuine request error against a ready server.
        RequestError::Failed(e) => LspError::Semantic(e),
    }
}

/// LSP's `DiagnosticSeverity` (1–4), spelled out so a non-exhaustive `match`
/// at a call site is a compile error, not a silently-wrong default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Information,
    Hint,
}

impl From<semantic::Severity> for Severity {
    fn from(s: semantic::Severity) -> Self {
        match s {
            semantic::Severity::Error => Self::Error,
            semantic::Severity::Warning => Self::Warning,
            semantic::Severity::Information => Self::Information,
            semantic::Severity::Hint => Self::Hint,
        }
    }
}

/// A location in a file: LSP's `Location`, with the URI resolved to a real
/// path and the range converted to grapheme-indexed [`Position`]s — everything
/// downstream speaks graphemes, never wire units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub file: PathBuf,
    pub range: Range,
}

/// One `textDocument/publishDiagnostics` entry, position-converted to
/// graphemes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: Severity,
    pub message: String,
    /// The producing tool, e.g. `"rustc"` or `"clippy"`.
    pub source: Option<String>,
}

/// The live language servers, one per `(server executable, workspace root)`.
///
/// See the module doc comment for the keying rationale. Constructed empty; a
/// server is spawned lazily (and non-blockingly) on the first request for a
/// file of its language under a not-yet-seen root.
#[derive(Default)]
pub struct LspClient {
    sessions: HashMap<(String, PathBuf), AsyncRustAnalyzerSession>,
}

impl LspClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a server for `filetype` (rooted anywhere) is spawned **and
    /// finished connecting** — i.e. it will actually serve a request rather
    /// than reject it with [`LspError::NotReady`].
    ///
    /// This is deliberately stricter than "a handle exists": completion is
    /// refreshed on every keystroke and gates on this, so a session that is
    /// still `Connecting` must read as *not* running, or every keystroke in a
    /// just-opened buffer would issue a request that only bounces back
    /// `NotReady`.
    pub fn is_running(&self, filetype: &str) -> bool {
        registry::for_filetype(filetype)
            .map(|s| {
                self.sessions
                    .iter()
                    .any(|((exe, _), session)| *exe == s.executable && session.is_ready())
            })
            .unwrap_or(false)
    }

    /// Whether a server for `filetype` (rooted anywhere) exists at all,
    /// regardless of readiness — including one still `Connecting`. Lets the UI
    /// show a "LSP: starting…" hint while the background connect is in flight.
    pub fn is_starting(&self, filetype: &str) -> bool {
        registry::for_filetype(filetype)
            .map(|s| {
                self.sessions
                    .iter()
                    .any(|((exe, _), session)| *exe == s.executable && session.state() == LspState::Connecting)
            })
            .unwrap_or(false)
    }

    /// The start-up progress of the server for `filetype` rooted at `file`'s
    /// workspace, or `None` when there is no progress to show (no session, not
    /// started, already ready, or the server sent no `$/progress`). Feeds kvim's
    /// LSP startup progress bar. See [`ProgressSnapshot`].
    pub fn progress(&self, filetype: &str, file: &Path) -> Option<ProgressSnapshot> {
        let server = registry::for_filetype(filetype)?;
        let root = workspace_root(file);
        let key = (server.executable.to_string(), root);
        self.sessions.get(&key).and_then(AsyncRustAnalyzerSession::progress)
    }

    /// Whether the registry knows a server for `filetype` *and* its executable
    /// is resolvable on `PATH`/data dir — i.e. an LSP request has any chance of
    /// working. Lets the UI degrade honestly (a statusline note) rather than
    /// spawning and failing.
    pub fn server_available(filetype: &str) -> bool {
        registry::for_filetype(filetype)
            .is_some_and(|s| registry::which(s.executable).is_some())
    }

    /// Jump-to-definition: the definition site(s) of the symbol at `pos` in
    /// `file`. `line_text` is the current text of `pos.line` (for the
    /// grapheme→char query conversion).
    pub fn definition(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Vec<Location>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        let locs = session
            .definition(file, line, character)
            .map_err(|e| map_request_error(filetype, session, e))?;
        Ok(locs.into_iter().map(convert_location).collect())
    }

    /// Find-references: every reference to the symbol at `pos` in `file`,
    /// including its declaration.
    pub fn references(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Vec<Location>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        let locs = session
            .references(file, line, character, true)
            .map_err(|e| map_request_error(filetype, session, e))?;
        Ok(locs.into_iter().map(convert_location).collect())
    }

    /// Hover text (type/signature/docs) for the symbol at `pos`, normalised to
    /// a single string, or `None` if the server has nothing to show.
    pub fn hover(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Option<String>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        Ok(session
            .hover(file, line, character)
            .map_err(|e| map_request_error(filetype, session, e))?
            .map(|h| h.contents))
    }

    /// Completion candidates at `pos` (typed, already JSON-parsed by the
    /// semantic layer).
    pub fn completion(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Vec<semantic::CompletionItem>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        session
            .completion(file, line, character)
            .map_err(|e| map_request_error(filetype, session, e))
    }

    /// Computes (does not write) the edits that would rename the symbol at
    /// `pos` to `new_name`. The caller applies the returned [`FileEdit`]s.
    pub fn rename(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str, new_name: &str) -> Result<Vec<FileEdit>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        session
            .rename(file, line, character, new_name)
            .map_err(|e| map_request_error(filetype, session, e))
    }

    /// Announces `file`'s current buffer `text` to its server as an open
    /// document — the prerequisite for the server to publish diagnostics about
    /// an unsaved buffer. Idempotent from the server's point of view; kvim
    /// calls it when a buffer of a served language is first shown.
    pub fn did_open(&mut self, filetype: &str, file: &Path, text: &str) -> Result<(), LspError> {
        let session = self.session(filetype, file)?;
        session
            .did_open(file, text)
            .map_err(|e| map_request_error(filetype, session, e))
    }

    /// Pushes the full new `text` of an already-open `file` (full-document
    /// sync). Incremental sync is a noted follow-up; a full resync is correct,
    /// just larger on the wire.
    pub fn did_change(&mut self, filetype: &str, file: &Path, text: &str) -> Result<(), LspError> {
        let session = self.session(filetype, file)?;
        session
            .did_change(file, text)
            .map_err(|e| map_request_error(filetype, session, e))
    }

    /// The diagnostics the server has most recently published for `file`,
    /// position-converted to graphemes. Diagnostics are *pushed*, so this
    /// reflects whatever has arrived so far (rust-analyzer publishes
    /// asynchronously after analysis — poll after [`Self::did_open`]).
    pub fn diagnostics(&mut self, filetype: &str, file: &Path) -> Result<Vec<Diagnostic>, LspError> {
        let session = self.session(filetype, file)?;
        let diags = session
            .diagnostics(file)
            .map_err(|e| map_request_error(filetype, session, e))?;
        let mut line_cache = LineCache::default();
        Ok(diags
            .into_iter()
            .map(|d| Diagnostic {
                range: convert_range(&mut line_cache, file, d.range),
                severity: d.severity.into(),
                message: d.message,
                source: d.source,
            })
            .collect())
    }

    /// Shuts down every running server, best-effort. Called on quit.
    ///
    /// Dropping an [`AsyncRustAnalyzerSession`] is itself non-blocking — it
    /// closes the job channel and detaches the worker, whose owned session's
    /// `Drop` kills the child process — so clearing the map is all that is
    /// needed, and it will not stall quit even on a session still connecting.
    pub fn shutdown_all(&mut self) {
        self.sessions.clear();
    }

    /// Finds or lazily spawns the server for `filetype` rooted at `file`'s
    /// workspace root, returning a **shared** handle to it.
    ///
    /// # This is the non-blocking, dedup-preserving heart of the client
    ///
    /// The spawn is [`AsyncRustAnalyzerSession::spawn_async_with_binary`], which
    /// returns immediately in state [`LspState::Connecting`] — the connect and
    /// rust-analyzer's `cachePriming` wait run on a background thread, never on
    /// the caller's (UI) thread. The handle is inserted into the map at this
    /// moment, *before* it is ready, so the `(server, root)` key dedups
    /// correctly under the async model: a request for a second file under the
    /// same root — arriving on the very next idle tick, while the first connect
    /// is still in flight — finds the existing `Connecting` handle and reuses
    /// it, instead of spawning a second rust-analyzer for the same workspace.
    ///
    /// Returns `&`, not `&mut`: the async handle's request methods take `&self`
    /// (they hand work to the worker thread over a channel), so no unique borrow
    /// is needed to issue a request.
    fn session(&mut self, filetype: &str, file: &Path) -> Result<&AsyncRustAnalyzerSession, LspError> {
        let server: &LanguageServer =
            registry::for_filetype(filetype).ok_or_else(|| LspError::NoServerForFiletype(filetype.to_string()))?;
        let root = workspace_root(file);
        let key = (server.executable.to_string(), root.clone());
        if !self.sessions.contains_key(&key) {
            let session = AsyncRustAnalyzerSession::spawn_async_with_binary(server.executable, &root);
            self.sessions.insert(key.clone(), session);
        }
        Ok(self.sessions.get(&key).expect("just inserted"))
    }
}

/// Converts a grapheme cursor position to the `(line, char-offset)` pair the
/// semantic API expects, given the queried line's text.
fn query_position(pos: Position, line_text: &str) -> (u32, u32) {
    let character = position::grapheme_col_to_unit(line_text, pos.col, WIRE);
    (pos.line as u32, character)
}

/// Converts a semantic [`Location`](semantic::Location) (char-offset range in a
/// possibly-different file) to a kvim [`Location`] (grapheme range), reading the
/// target file's lines to do the char→grapheme conversion correctly.
fn convert_location(loc: semantic::Location) -> Location {
    let mut cache = LineCache::default();
    Location { range: convert_range(&mut cache, &loc.path, loc.range), file: loc.path }
}

/// Converts a semantic char-offset [`Range`](semantic::Range) in `path` to a
/// grapheme-indexed [`Range`], using `cache` to avoid re-reading the file for
/// the two endpoints.
fn convert_range(cache: &mut LineCache, path: &Path, range: semantic::Range) -> Range {
    Range::new(convert_position(cache, path, range.start), convert_position(cache, path, range.end))
}

fn convert_position(cache: &mut LineCache, path: &Path, pos: semantic::Position) -> Position {
    let line_text = cache.line(path, pos.line as usize);
    let col = position::unit_to_grapheme_col(&line_text, pos.character, WIRE);
    Position::new(pos.line as usize, col)
}

/// A tiny per-conversion cache of a file's lines, so converting a multi-endpoint
/// range (or several diagnostics in one file) reads the file at most once.
#[derive(Default)]
struct LineCache {
    file: Option<(PathBuf, Vec<String>)>,
}

impl LineCache {
    fn line(&mut self, path: &Path, line: usize) -> String {
        if self.file.as_ref().map(|(p, _)| p.as_path()) != Some(path) {
            let lines = std::fs::read_to_string(path)
                .map(|s| s.lines().map(str::to_string).collect())
                .unwrap_or_default();
            self.file = Some((path.to_path_buf(), lines));
        }
        self.file
            .as_ref()
            .and_then(|(_, lines)| lines.get(line))
            .cloned()
            .unwrap_or_default()
    }
}

/// The workspace root for `file`: the nearest ancestor directory containing a
/// `Cargo.toml` or a `.git`, else the file's own directory. This is what a
/// server is rooted at, and half of a session's registry key.
fn workspace_root(file: &Path) -> PathBuf {
    let canonical = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
    let start = canonical.parent().unwrap_or(Path::new("."));
    let mut dir = start;
    loop {
        if dir.join("Cargo.toml").exists() || dir.join(".git").exists() {
            return dir.to_path_buf();
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }
    start.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_root_finds_the_enclosing_cargo_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let file = dir.path().join("src/lib.rs");
        std::fs::write(&file, "fn x() {}\n").unwrap();
        assert_eq!(workspace_root(&file), std::fs::canonicalize(dir.path()).unwrap());
    }

    #[test]
    fn workspace_root_falls_back_to_the_files_own_directory() {
        // A lone file with no Cargo.toml/.git above it roots at its parent —
        // exactly what a single .tex/.lua file wants.
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("papers");
        std::fs::create_dir(&sub).unwrap();
        let file = sub.join("thesis.tex");
        std::fs::write(&file, "\\documentclass{article}\n").unwrap();
        // No .git anywhere in the tempdir chain, so it falls back to `papers/`.
        let root = workspace_root(&file);
        assert_eq!(root, std::fs::canonicalize(&sub).unwrap());
    }

    #[test]
    fn line_cache_reads_a_file_once_and_serves_multiple_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.rs");
        std::fs::write(&file, "line zero\nline one\n線 two\n").unwrap();
        let mut cache = LineCache::default();
        assert_eq!(cache.line(&file, 0), "line zero");
        assert_eq!(cache.line(&file, 1), "line one");
        assert_eq!(cache.line(&file, 2), "線 two");
        assert_eq!(cache.line(&file, 99), "", "out of range is empty, never a panic");
    }

    #[test]
    fn starting_an_unregistered_filetype_is_a_clear_error() {
        let mut client = LspClient::new();
        let err = client
            .definition("cobol", Path::new("/tmp/x.cbl"), Position::ORIGIN, "")
            .unwrap_err();
        assert!(matches!(err, LspError::NoServerForFiletype(ft) if ft == "cobol"));
    }

    /// End-to-end proof that the generalized client resolves a real
    /// definition through a live `rust-analyzer`, keyed by `(server, root)`.
    ///
    /// `#[ignore]`d: spawns a real server and waits for indexing. Run with
    /// `cargo test --release -p kopitiam-neovim -- --ignored`.
    #[test]
    #[ignore = "spawns a real rust-analyzer and waits for indexing; run with `-- --ignored`"]
    fn live_rust_analyzer_resolves_a_definition_through_the_generalized_client() {
        if registry::which("rust-analyzer").is_none() {
            eprintln!("rust-analyzer not on PATH; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname=\"kvim_lsp_test\"\nversion=\"0.1.0\"\nedition=\"2021\"\n").unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let lib = dir.path().join("src/lib.rs");
        let source = "pub fn greet() -> &'static str {\n    \"hi\"\n}\n\npub fn caller() -> &'static str {\n    greet()\n}\n";
        std::fs::write(&lib, source).unwrap();

        let mut client = LspClient::new();
        // The call site `greet()` on line 5, grapheme column 4.
        let line5 = source.lines().nth(5).unwrap();

        // The client is now non-blocking: the first request *spawns* the
        // background connect and returns `NotReady` while it is in flight, so we
        // poll (as kvim's idle tick does) until the server is ready rather than
        // expecting the first call to block through indexing.
        let start = std::time::Instant::now();
        let locs = loop {
            match client.definition("rust", &lib, Position::new(5, 4), line5) {
                Ok(locs) => break locs,
                Err(e) if e.is_not_ready() => {
                    assert!(
                        start.elapsed() < std::time::Duration::from_secs(180),
                        "server never became ready within 180s"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => panic!("definition failed: {e}"),
            }
        };
        assert!(!locs.is_empty(), "the call to greet must resolve to its declaration");
        assert_eq!(locs[0].range.anchor.line, 0, "greet is declared on line 0");
        assert_eq!(locs[0].range.anchor.col, 7, "the identifier starts after `pub fn ` (7 graphemes)");

        client.shutdown_all();
    }
}
