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
//! # What `RustAnalyzerSession` gives us, despite the name
//!
//! `connect_with_binary` spawns *any* executable and speaks generic LSP to it,
//! so this same session type drives `lua-language-server`, `texlab`, and
//! rust-analyzer (for both `.rs` files and `Cargo.toml`) alike — nothing about
//! it is rust-analyzer-specific once connected.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use kopitiam_semantic::RustAnalyzerSession;
use kopitiam_semantic::edit::FileEdit;
use kopitiam_semantic::{self as semantic};

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
    #[error("failed to start `{executable}`: {source}")]
    Spawn {
        executable: String,
        #[source]
        source: anyhow::Error,
    },
    #[error(transparent)]
    Semantic(#[from] anyhow::Error),
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
/// server is spawned lazily on the first request for a file of its language
/// under a not-yet-seen root.
#[derive(Default)]
pub struct LspClient {
    sessions: HashMap<(String, PathBuf), RustAnalyzerSession>,
}

impl LspClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a server is already running for `filetype` rooted anywhere.
    pub fn is_running(&self, filetype: &str) -> bool {
        registry::for_filetype(filetype)
            .map(|s| self.sessions.keys().any(|(exe, _)| *exe == s.executable))
            .unwrap_or(false)
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
        let locs = session.definition(file, line, character)?;
        Ok(locs.into_iter().map(convert_location).collect())
    }

    /// Find-references: every reference to the symbol at `pos` in `file`,
    /// including its declaration.
    pub fn references(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Vec<Location>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        let locs = session.references(file, line, character, true)?;
        Ok(locs.into_iter().map(convert_location).collect())
    }

    /// Hover text (type/signature/docs) for the symbol at `pos`, normalised to
    /// a single string, or `None` if the server has nothing to show.
    pub fn hover(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Option<String>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        Ok(session.hover(file, line, character)?.map(|h| h.contents))
    }

    /// Completion candidates at `pos` (typed, already JSON-parsed by the
    /// semantic layer).
    pub fn completion(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str) -> Result<Vec<semantic::CompletionItem>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        Ok(session.completion(file, line, character)?)
    }

    /// Computes (does not write) the edits that would rename the symbol at
    /// `pos` to `new_name`. The caller applies the returned [`FileEdit`]s.
    pub fn rename(&mut self, filetype: &str, file: &Path, pos: Position, line_text: &str, new_name: &str) -> Result<Vec<FileEdit>, LspError> {
        let (line, character) = query_position(pos, line_text);
        let session = self.session(filetype, file)?;
        Ok(session.rename(file, line, character, new_name)?)
    }

    /// Announces `file`'s current buffer `text` to its server as an open
    /// document — the prerequisite for the server to publish diagnostics about
    /// an unsaved buffer. Idempotent from the server's point of view; kvim
    /// calls it when a buffer of a served language is first shown.
    pub fn did_open(&mut self, filetype: &str, file: &Path, text: &str) -> Result<(), LspError> {
        let session = self.session(filetype, file)?;
        session.did_open(file, text)?;
        Ok(())
    }

    /// Pushes the full new `text` of an already-open `file` (full-document
    /// sync). Incremental sync is a noted follow-up; a full resync is correct,
    /// just larger on the wire.
    pub fn did_change(&mut self, filetype: &str, file: &Path, text: &str) -> Result<(), LspError> {
        let session = self.session(filetype, file)?;
        session.did_change(file, text)?;
        Ok(())
    }

    /// The diagnostics the server has most recently published for `file`,
    /// position-converted to graphemes. Diagnostics are *pushed*, so this
    /// reflects whatever has arrived so far (rust-analyzer publishes
    /// asynchronously after analysis — poll after [`Self::did_open`]).
    pub fn diagnostics(&mut self, filetype: &str, file: &Path) -> Result<Vec<Diagnostic>, LspError> {
        let session = self.session(filetype, file)?;
        let diags = session.diagnostics(file)?;
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
    pub fn shutdown_all(&mut self) {
        for (_, session) in self.sessions.drain() {
            let _ = session.shutdown();
        }
    }

    /// Finds or lazily spawns the server for `filetype` rooted at `file`'s
    /// workspace root. Blocks on first spawn until the server finishes its
    /// initial indexing pass (rust-analyzer: seconds to minutes on a large
    /// workspace).
    fn session(&mut self, filetype: &str, file: &Path) -> Result<&mut RustAnalyzerSession, LspError> {
        let server: &LanguageServer =
            registry::for_filetype(filetype).ok_or_else(|| LspError::NoServerForFiletype(filetype.to_string()))?;
        let root = workspace_root(file);
        let key = (server.executable.to_string(), root.clone());
        if !self.sessions.contains_key(&key) {
            let session = RustAnalyzerSession::connect_with_binary(server.executable, &root)
                .map_err(|source| LspError::Spawn { executable: server.executable.to_string(), source })?;
            self.sessions.insert(key.clone(), session);
        }
        Ok(self.sessions.get_mut(&key).expect("just inserted"))
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
        let locs = client.definition("rust", &lib, Position::new(5, 4), line5).expect("definition should resolve");
        assert!(!locs.is_empty(), "the call to greet must resolve to its declaration");
        assert_eq!(locs[0].range.anchor.line, 0, "greet is declared on line 0");
        assert_eq!(locs[0].range.anchor.col, 7, "the identifier starts after `pub fn ` (7 graphemes)");

        client.shutdown_all();
    }
}
