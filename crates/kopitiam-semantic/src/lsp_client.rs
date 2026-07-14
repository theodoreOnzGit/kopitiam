//! A minimal, synchronous JSON-RPC client for the Language Server Protocol.
//!
//! This intentionally does not depend on `lsp-types`: it only needs to drive
//! `initialize` / `initialized` / `workspace/symbol` / `shutdown` / `exit`,
//! and reading the handful of fields it needs straight out of `serde_json`
//! keeps the dependency footprint small and avoids coupling to a specific
//! LSP crate's protocol-version assumptions.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::lsp_types::{self, CompletionItem, DiagnosticsStore, Hover, Location};
use crate::position::PositionEncoding;

/// Reads a `file://` URI's contents once and hands out individual lines, so a
/// single response that references the same file many times (e.g. a
/// `references` result with dozens of hits in one file) reads it from disk
/// only once. A URI that isn't a readable `file://` path yields `None` for
/// every line rather than erroring — [`crate::lsp_types`]'s parsers treat a
/// missing line as empty text, which degrades a position to column 0 instead
/// of dropping the whole result.
#[derive(Default)]
struct LineCache {
    files: HashMap<String, Option<Vec<String>>>,
}

impl LineCache {
    fn line(&mut self, uri: &str, line: u32) -> Option<String> {
        let entry = self.files.entry(uri.to_string()).or_insert_with(|| read_file_lines(uri));
        entry.as_ref().and_then(|lines| lines.get(line as usize).cloned())
    }
}

/// Reads `uri`'s file and splits it into lines (on `\n`, matching
/// [`crate::edit`]'s convention). `None` if the URI is not a `file://` path or
/// the file cannot be read.
fn read_file_lines(uri: &str) -> Option<Vec<String>> {
    let path = url::Url::parse(uri).ok()?.to_file_path().ok()?;
    let text = std::fs::read_to_string(path).ok()?;
    Some(text.split('\n').map(str::to_string).collect())
}

/// Returns true if `program` can be spawned at all (i.e. exists on `PATH`
/// and runs). Used by providers to degrade gracefully when their tool is
/// not installed, instead of failing the whole collection run.
pub fn binary_available(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn read_one_message(stdout: &mut BufReader<ChildStdout>) -> Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = stdout.read_line(&mut line)?;
        if n == 0 {
            return Ok(None); // EOF: server exited
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse()?);
        }
    }
    let len = content_length.context("LSP message missing Content-Length header")?;
    let mut buf = vec![0u8; len];
    stdout.read_exact(&mut buf)?;
    Ok(Some(serde_json::from_slice(&buf)?))
}

/// The `(token, kind, title)` of a `$/progress` notification, or `None` if
/// `msg` is not one.
fn progress_parts(msg: &Value) -> Option<(String, &str, String)> {
    if msg.get("method").and_then(Value::as_str) != Some("$/progress") {
        return None;
    }
    let token = msg.pointer("/params/token").and_then(Value::as_str).unwrap_or_default().to_ascii_lowercase();
    let kind = msg.pointer("/params/value/kind").and_then(Value::as_str)?;
    let title = msg.pointer("/params/value/title").and_then(Value::as_str).unwrap_or_default().to_ascii_lowercase();
    Some((token, kind, title))
}

/// True if `msg` is a `$/progress` for rust-analyzer's indexing pass.
///
/// The indexing work-done token is `rustAnalyzer/cachePriming` with the human
/// title `"Indexing"` — **not** a token literally containing `"index"** (an
/// earlier version checked the token for `"index"` and so never matched,
/// blocking the whole connect timeout on every start-up; the real token was
/// captured from a live server). Either identifier is accepted so a future
/// rename of one does not silently re-break detection.
fn is_indexing_progress(msg: &Value) -> bool {
    progress_parts(msg)
        .map(|(token, _, title)| token.contains("cachepriming") || token.contains("index") || title == "indexing")
        .unwrap_or(false)
}

/// True if `msg` is the *end* of the indexing pass.
fn is_indexing_progress_end(msg: &Value) -> bool {
    is_indexing_progress(msg) && progress_parts(msg).map(|(_, kind, _)| kind == "end").unwrap_or(false)
}

/// True if `msg` is indexing *doing work* — specifically a `report`. Used to
/// distinguish rust-analyzer's real cache-priming pass (which streams
/// `report`s as it indexes each crate) from the trivial empty `begin`/`end`
/// cachePriming cycle it also emits with no work in between, so readiness is
/// only declared after indexing has actually made progress. (A `begin` is
/// deliberately **not** counted: the trivial cycle has one, and counting it
/// would let that cycle's immediate `end` declare readiness before the real
/// index exists — which left `workspace/symbol` racing an unbuilt index.)
fn is_indexing_progress_work(msg: &Value) -> bool {
    is_indexing_progress(msg) && progress_parts(msg).map(|(_, kind, _)| kind == "report").unwrap_or(false)
}

pub struct LspClient {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    next_id: i64,
    /// The wire unit the server said `Position.character` is measured in,
    /// negotiated during `initialize` via the `general.positionEncodings`
    /// client capability and the server's `capabilities.positionEncoding`
    /// response (see [`crate::position`] for the full LSP 3.17 rundown).
    /// [`Self::rename`] and [`Self::code_actions`] take `character` values
    /// already expressed in this unit — converting from this crate's public
    /// `char`-offset contract is [`crate::session::RustAnalyzerSession`]'s
    /// job, via [`Self::position_encoding`], since that is the layer that
    /// has the line text needed to do the conversion.
    position_encoding: PositionEncoding,
    /// The most recent `textDocument/publishDiagnostics` set per document
    /// URI. Unlike every other feature here, diagnostics are *pushed* by the
    /// server as unsolicited notifications, so they are captured into this
    /// store as they stream past rather than returned from a request — see
    /// [`DiagnosticsStore`] for the full rationale and the replace-not-append
    /// semantics.
    diagnostics: DiagnosticsStore,
    /// Monotonic per-URI document version for `textDocument/didChange`. LSP
    /// requires each change to a document to carry a strictly increasing
    /// version; `didOpen` is version 1 (see [`Self::did_open_as`]), so
    /// changes start at 2.
    doc_versions: HashMap<String, i64>,
}

impl LspClient {
    /// Spawns `program`, performs the `initialize` / `initialized`
    /// handshake with `root` as the workspace root, and waits (up to
    /// `index_timeout`) for rust-analyzer to report indexing as complete
    /// before returning — otherwise a `workspace/symbol` query issued
    /// immediately after start-up would race the background index and
    /// come back empty.
    pub fn spawn(program: &str, root: &Path, index_timeout: Duration) -> Result<Self> {
        Self::spawn_with_args(program, &[], root, index_timeout)
    }

    /// Like [`Self::spawn`], but passes `args` to the server on its command
    /// line.
    ///
    /// # Why this exists
    ///
    /// rust-analyzer speaks LSP on stdio when invoked bare, so the original
    /// [`Self::spawn`] passed no arguments at all. **Every other language
    /// server needs arguments to speak LSP**, and without this method not one
    /// of them can even be launched:
    ///
    /// | Server | Invocation |
    /// |---|---|
    /// | rust-analyzer | *(no arguments)* |
    /// | pyright | `pyright-langserver --stdio` |
    /// | OmniSharp | `OmniSharp -lsp` |
    /// | Roslyn LSP | `Microsoft.CodeAnalysis.LanguageServer --stdio` |
    /// | clangd | *(no arguments, but takes `--compile-commands-dir`)* |
    ///
    /// This was found the hard way: four language adapters (Python, C#, C++,
    /// Visual Basic) were written concurrently against this client, and **all
    /// of them independently discovered they could not spawn their server.**
    /// When four separate implementers hit the same wall, the wall is the bug.
    pub fn spawn_with_args(program: &str, args: &[&str], root: &Path, index_timeout: Duration) -> Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn `{program}`"))?;
        let stdin = child.stdin.take().context("child has no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().context("child has no stdout")?);

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(Some(msg)) = read_one_message(&mut stdout) {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });

        let mut client = Self {
            child,
            stdin,
            rx,
            next_id: 1,
            position_encoding: PositionEncoding::Utf16,
            diagnostics: DiagnosticsStore::default(),
            doc_versions: HashMap::new(),
        };
        client.initialize(root)?;
        client.wait_for_indexing(index_timeout, program);
        Ok(client)
    }

    fn write_message(&mut self, value: &Value) -> Result<()> {
        let body = serde_json::to_vec(value)?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len())?;
        self.stdin.write_all(&body)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn recv(&self, timeout: Duration) -> Result<Value> {
        self.rx
            .recv_timeout(timeout)
            .context("timed out waiting for a message from the language server")
    }

    /// Sends a request and blocks until the matching response arrives.
    ///
    /// Any other message received while waiting is handled by
    /// [`Self::handle_incoming`]: notifications are discarded, and
    /// server-initiated requests (e.g. `workspace/applyEdit`, sent for
    /// command-based code actions) are answered inline so the server is
    /// never left hanging on us.
    fn request<P: Serialize, R: DeserializeOwned>(&mut self, method: &str, params: P, timeout: Duration) -> Result<R> {
        let id = self.next_id;
        self.next_id += 1;
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let msg = self.recv(remaining)?;
            if msg.get("id").and_then(Value::as_i64) == Some(id) && msg.get("method").is_none() {
                if let Some(error) = msg.get("error") {
                    bail!("{method} request failed: {error}");
                }
                let result = msg.get("result").cloned().unwrap_or(Value::Null);
                return Ok(serde_json::from_value(result)?);
            }
            self.handle_incoming(&msg)?;
        }
    }

    /// Handles a message that was not the response we were waiting for:
    /// `textDocument/publishDiagnostics` notifications are captured into the
    /// diagnostics store, any other notification (no `id`) is ignored, and a
    /// server-initiated request (has both `id` and `method`) is answered so
    /// the server's own in-flight operation can complete.
    fn handle_incoming(&mut self, msg: &Value) -> Result<()> {
        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            return Ok(()); // a response to some other/stale request id; drop it
        };
        // Diagnostics are pushed, not requested: capture the latest set for
        // the URI regardless of whether we happen to be inside a request loop
        // right now (see `DiagnosticsStore`). This is a notification, so there
        // is nothing to reply to.
        if method == "textDocument/publishDiagnostics" {
            self.diagnostics.ingest_notification(msg);
            return Ok(());
        }
        let Some(id) = msg.get("id").cloned() else {
            return Ok(()); // a plain notification; nothing to reply to
        };

        let result = match method {
            "workspace/applyEdit" => {
                // The positions inside this WorkspaceEdit are in whatever
                // encoding we negotiated during `initialize` — same as any
                // other WorkspaceEdit the server sends us.
                let edit = msg.pointer("/params/edit").cloned().unwrap_or(Value::Null);
                match crate::edit::apply_workspace_edit(&edit, self.position_encoding) {
                    Ok(_files) => json!({ "applied": true }),
                    Err(err) => json!({ "applied": false, "failureReason": err.to_string() }),
                }
            }
            // Capability registration and progress-token creation: we don't
            // track dynamic registrations, so acknowledging with `null` is
            // both correct per spec and sufficient for our purposes.
            "client/registerCapability" | "client/unregisterCapability" | "window/workDoneProgress/create" => {
                Value::Null
            }
            _ => Value::Null,
        };

        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }))
    }

    fn notify<P: Serialize>(&mut self, method: &str, params: P) -> Result<()> {
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
    }

    fn initialize(&mut self, root: &Path) -> Result<()> {
        let root_uri = format!("file://{}", root.display());
        let result: Value = self.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": root_uri,
                "capabilities": {
                    "workspace": {
                        "symbol": { "dynamicRegistration": false },
                        "applyEdit": true,
                        "workspaceEdit": { "documentChanges": true },
                        "executeCommand": { "dynamicRegistration": false },
                    },
                    "window": { "workDoneProgress": true },
                    // Advertise the read-only document features this client
                    // now drives, so a server actually answers them. Two of
                    // these are load-bearing:
                    //   * `definition.linkSupport: true` tells the server it
                    //     MAY reply with the `LocationLink[]` shape — which we
                    //     handle in `crate::lsp_types::parse_locations`. We opt
                    //     in on purpose: it exercises the harder response shape
                    //     rather than hoping the server never picks it.
                    //   * `publishDiagnostics` advertises that we accept pushed
                    //     diagnostics (they are captured in `handle_incoming`).
                    "textDocument": {
                        "synchronization": { "dynamicRegistration": false, "didSave": false },
                        "definition": { "dynamicRegistration": false, "linkSupport": true },
                        "references": { "dynamicRegistration": false },
                        "hover": { "dynamicRegistration": false, "contentFormat": ["markdown", "plaintext"] },
                        // `snippetSupport: true` is load-bearing, not cosmetic:
                        // rust-analyzer only marks a completion `insertTextFormat: 2`
                        // (Snippet) — e.g. a function as `greet($0)`, a macro as
                        // `println!($0)` — when the client advertises it here.
                        // Advertise `false` and the server strips every snippet
                        // to plain text, so `insertTextFormat` is never `2` and
                        // the `CompletionItem::is_snippet` flag is dead. A client
                        // that opts in MUST expand snippet `insertText` rather
                        // than inserting it verbatim (else `$0` lands in the
                        // buffer); kvim's completion layer does exactly that (see
                        // `docs/ai-decisions/AID-0026`). No other consumer of this
                        // crate inserts a completion's `insert_text` literally, so
                        // opting in is safe workspace-wide.
                        "completion": {
                            "dynamicRegistration": false,
                            "completionItem": { "snippetSupport": true, "documentationFormat": ["markdown", "plaintext"] },
                        },
                        "publishDiagnostics": { "relatedInformation": false },
                    },
                    // Advertise support for all three LSP 3.17 position
                    // encodings. Per spec, `"utf-8"` means byte offsets and
                    // `"utf-32"` means Unicode scalar value (char) offsets —
                    // NOT the other way around (see `crate::position`'s
                    // module docs for the full explanation and the bug this
                    // fixes). Whichever one the server picks back is
                    // honoured via `PositionEncoding`, converting to/from
                    // this crate's public `char`-offset contract at the
                    // `RustAnalyzerSession` layer, so this client is correct
                    // regardless of the server's choice — not just lucky
                    // that rust-analyzer happens to pick `"utf-16"` today.
                    "general": { "positionEncodings": ["utf-8", "utf-16", "utf-32"] },
                },
            }),
            Duration::from_secs(30),
        )?;
        self.position_encoding = PositionEncoding::from_capability(
            result.pointer("/capabilities/positionEncoding").and_then(Value::as_str),
        );
        self.notify("initialized", json!({}))
    }

    /// Waits until the server looks ready, or `timeout` elapses — best-effort,
    /// never a hard failure (callers proceed regardless; requests carry their
    /// own timeouts).
    ///
    /// Readiness is signalled two ways, whichever comes first:
    ///
    /// 1. **The rust-analyzer indexing-end token** ([`is_indexing_progress_end`]).
    ///    rust-analyzer streams `$/progress` `report`s while indexing and an
    ///    `end` when done; catching that `end` returns the instant it is ready.
    /// 2. **The server going quiet** — no message for `IDLE_GRACE`. A server
    ///    that has stopped emitting progress (indexing finished; or a server
    ///    like `lua-language-server`/`texlab` that never emits a rust-analyzer
    ///    indexing token at all; or a tiny crate rust-analyzer indexes so fast
    ///    the token races us) is, for our purposes, ready.
    ///
    /// # Why the quiet heuristic matters
    ///
    /// The previous version waited *only* for signal (1), passing the whole
    /// `timeout` to each `recv`. Any server that does not emit that exact
    /// rust-analyzer token — every non-rust server, and rust-analyzer itself on
    /// a trivial workspace — therefore blocked the **entire** timeout (180 s in
    /// practice) before the first request could go out, which made
    /// go-to-definition feel like a hang. Because a busy server keeps the loop
    /// alive with its own progress `report`s, the short idle window only fires
    /// once the server is genuinely quiet, so this speeds up the common case
    /// without returning mid-index on a large project.
    fn wait_for_indexing(&mut self, timeout: Duration, program: &str) {
        /// How long a **non-rust-analyzer** server may stay silent before we
        /// treat it as ready. Servers like `lua-language-server` and `texlab`
        /// do not emit rust-analyzer's indexing token, so "gone quiet after
        /// `initialized`" is the readiness signal we have for them.
        const IDLE_GRACE: Duration = Duration::from_secs(3);

        // rust-analyzer needs a *precise* readiness signal, not idle: on a
        // large multi-crate workspace it can sit silent for well over ten
        // seconds at start-up (loading `cargo metadata`) before its first
        // progress message, so any idle window short enough to feel responsive
        // would fire during that silence and declare it ready before it has
        // indexed anything (which left `workspace/symbol` empty). So for
        // rust-analyzer we wait — up to the full `timeout` — for the end of the
        // substantive `cachePriming` pass instead. Every other server uses the
        // idle heuristic above.
        let is_rust_analyzer = program.contains("rust-analyzer");
        let deadline = Instant::now() + timeout;
        // rust-analyzer emits a trivial empty cachePriming cycle *and* the real
        // one; only the real pass carries `report`s. Wait for the end of a pass
        // that actually did work, so the full index exists before we return.
        let mut indexing_did_work = false;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return;
            }
            let wait = if is_rust_analyzer { remaining } else { remaining.min(IDLE_GRACE) };
            match self.recv(wait) {
                Ok(msg) => {
                    if is_indexing_progress_work(&msg) {
                        indexing_did_work = true;
                    }
                    // Ready the instant the substantive indexing pass ends.
                    if indexing_did_work && is_indexing_progress_end(&msg) {
                        return;
                    }
                    continue;
                }
                // For a non-rust-analyzer server: quiet for `IDLE_GRACE` => ready.
                // For rust-analyzer: `wait` was the full remaining time, so this
                // is only reached at the overall deadline — proceed anyway (the
                // first request carries its own timeout).
                Err(_) => return,
            }
        }
    }

    /// Runs a `workspace/symbol` request and returns the raw
    /// `SymbolInformation[]` (or `WorkspaceSymbol[]`) JSON entries.
    pub fn workspace_symbols(&mut self, query: &str) -> Result<Vec<Value>> {
        let result: Value = self.request("workspace/symbol", json!({ "query": query }), Duration::from_secs(60))?;
        match result {
            Value::Array(items) => Ok(items),
            Value::Null => Ok(Vec::new()),
            other => bail!("unexpected workspace/symbol response shape: {other}"),
        }
    }

    /// Informs the server that `path`'s contents are `text`, as of version 1.
    ///
    /// LSP-correct clients open a document before requesting edits on it.
    /// rust-analyzer's workspace-wide VFS makes this less load-bearing than
    /// on some servers (it already scanned every file at start-up), but it
    /// costs nothing and keeps this client honest about protocol semantics.
    pub fn did_open(&mut self, uri: &str, text: &str) -> Result<()> {
        self.did_open_as(uri, "rust", text)
    }

    /// Like [`Self::did_open`], but declares the document's `languageId`.
    ///
    /// The original `did_open` hardcoded `"rust"`, which was harmless while
    /// rust-analyzer was the only server but is actively wrong for any other:
    /// a server that receives `languageId: "rust"` for a `.py` file will
    /// either ignore the document or misparse it. LSP's `languageId` values
    /// are the standard identifiers — `python`, `csharp`, `cpp`, `c`,
    /// `vb`, `lua`, ...
    pub fn did_open_as(&mut self, uri: &str, language_id: &str, text: &str) -> Result<()> {
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                },
            }),
        )
    }

    /// Runs a `textDocument/documentSymbol` request for one open document and
    /// returns the raw JSON array the server replied with.
    ///
    /// # Why this matters more than `workspace/symbol`
    ///
    /// [`Self::workspace_symbols`] returns the **flat** `SymbolInformation`
    /// shape, which throws away nesting: a method inside a class inside a
    /// namespace comes back as three unrelated symbols. `documentSymbol`
    /// returns the **hierarchical** `DocumentSymbol` shape, which preserves
    /// containment — and containment is exactly what the knowledge graph needs
    /// in order to say "this method belongs to that class."
    ///
    /// Servers may legally reply with *either* shape (the flat one is the
    /// legacy fallback), so callers must handle both. This method deliberately
    /// returns raw `Value`s rather than a typed tree, because each language
    /// adapter already knows how to interpret its own server's variant, and a
    /// lossy common type here would throw away the very nesting it exists to
    /// preserve.
    ///
    /// The document must have been opened first with [`Self::did_open_as`].
    // Public API for the language adapters, but no in-crate caller uses the
    // shared `LspClient` for it yet — the C++/C# adapters currently roll their
    // own per-server clients (see beads `kopitiam-gjg`/`kopitiam-mfo`). Kept
    // rather than deleted so that the moment an adapter switches to the shared
    // client the method is already here and correct; `#[allow(dead_code)]`
    // pending that first caller.
    #[allow(dead_code)]
    pub fn document_symbols(&mut self, uri: &str) -> Result<Vec<Value>> {
        let result: Value = self.request(
            "textDocument/documentSymbol",
            json!({ "textDocument": { "uri": uri } }),
            Duration::from_secs(60),
        )?;
        match result {
            Value::Array(symbols) => Ok(symbols),
            // A server with nothing to say replies `null`, which is not an
            // error — an empty file genuinely has no symbols.
            Value::Null => Ok(Vec::new()),
            other => bail!("unexpected textDocument/documentSymbol response shape: {other}"),
        }
    }

    /// The wire unit the server negotiated for `Position.character` during
    /// `initialize` — see the [`Self::position_encoding`] field docs.
    /// [`crate::session::RustAnalyzerSession`] uses this to convert its
    /// public `char`-offset positions to and from wire units.
    pub(crate) fn position_encoding(&self) -> PositionEncoding {
        self.position_encoding
    }

    /// Requests a rename of the symbol at `(line, character)` in `uri` to
    /// `new_name`, returning the raw `WorkspaceEdit` JSON. `character` must
    /// already be expressed in this client's negotiated
    /// [`PositionEncoding`] (see [`Self::position_encoding`]) — callers
    /// outside this module want [`crate::session::RustAnalyzerSession::rename`]
    /// instead, which accepts a `char` offset and does that conversion.
    /// Does not touch disk — pair with [`crate::edit::apply_workspace_edit`]
    /// or [`crate::edit::compute_workspace_edit`] to actually apply it.
    pub fn rename(&mut self, uri: &str, line: u32, character: u32, new_name: &str) -> Result<Value> {
        self.request(
            "textDocument/rename",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
                "newName": new_name,
            }),
            Duration::from_secs(60),
        )
    }

    /// Requests the code actions available for the range
    /// `(start_line, start_character)..(end_line, end_character)` in `uri`.
    /// The `*_character` arguments must already be expressed in this
    /// client's negotiated [`PositionEncoding`] (see
    /// [`Self::position_encoding`]) — see [`Self::rename`]'s docs for why.
    /// Returns the raw `(Command | CodeAction)[]` JSON; each entry either
    /// carries an `edit` directly or a `command` to run via
    /// [`Self::execute_command`].
    pub fn code_actions(
        &mut self,
        uri: &str,
        start_line: u32,
        start_character: u32,
        end_line: u32,
        end_character: u32,
    ) -> Result<Vec<Value>> {
        let result: Value = self.request(
            "textDocument/codeAction",
            json!({
                "textDocument": { "uri": uri },
                "range": {
                    "start": { "line": start_line, "character": start_character },
                    "end": { "line": end_line, "character": end_character },
                },
                "context": { "diagnostics": [] },
            }),
            Duration::from_secs(60),
        )?;
        match result {
            Value::Array(items) => Ok(items),
            Value::Null => Ok(Vec::new()),
            other => bail!("unexpected textDocument/codeAction response shape: {other}"),
        }
    }

    /// Executes a `Command` returned by [`Self::code_actions`] (used for
    /// code actions that compute their edit lazily). The resulting edit, if
    /// any, arrives as a server-initiated `workspace/applyEdit` request,
    /// which [`Self::handle_incoming`] answers by applying it directly —
    /// this method's return value is whatever `workspace/executeCommand`
    /// itself responds with, which is often just `null`.
    pub fn execute_command(&mut self, command: &str, arguments: Value) -> Result<Value> {
        self.request(
            "workspace/executeCommand",
            json!({ "command": command, "arguments": arguments }),
            Duration::from_secs(60),
        )
    }

    /// Notifies the server that `uri`'s in-memory contents are now `text`
    /// (full-document sync). Use this to make an *unsaved* buffer edit visible
    /// to the server before a definition/hover/completion/diagnostics query —
    /// without it, those requests see whatever [`Self::did_open_as`] last sent
    /// (typically the on-disk content).
    ///
    /// The document must have been opened first with [`Self::did_open_as`];
    /// this bumps the per-URI version [`Self::did_open_as`] started at 1.
    /// Full-document sync (`contentChanges: [{ text }]`) is deliberately
    /// chosen over incremental range edits: it is simpler, cannot desync, and
    /// rust-analyzer (and every other server this drives) accepts
    /// `TextDocumentSyncKind::Full`.
    pub fn did_change(&mut self, uri: &str, text: &str) -> Result<()> {
        let version = {
            let entry = self.doc_versions.entry(uri.to_string()).or_insert(1);
            *entry += 1;
            *entry
        };
        self.notify(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [ { "text": text } ],
            }),
        )
    }

    /// Requests the definition(s) of the symbol at `(line, character)` in
    /// `uri`. `character` must already be in the negotiated
    /// [`PositionEncoding`] (see [`Self::rename`]'s docs); the returned
    /// [`Location`]s carry `char`-offset ranges, converted here.
    ///
    /// Handles all three shapes a server may reply with — a single
    /// `Location`, a `Location[]`, or a `LocationLink[]` — via
    /// [`crate::lsp_types::parse_locations`]. The document should be open
    /// ([`Self::did_open_as`]) first.
    pub fn definition(&mut self, uri: &str, line: u32, character: u32) -> Result<Vec<Location>> {
        let result: Value = self.request(
            "textDocument/definition",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
            }),
            Duration::from_secs(60),
        )?;
        Ok(self.parse_locations(&result))
    }

    /// Requests all references to the symbol at `(line, character)` in `uri`,
    /// including its declaration when `include_declaration` is set. Same
    /// `character` and result conventions as [`Self::definition`] (the
    /// response is a `Location[]`, which the shared parser handles).
    pub fn references(&mut self, uri: &str, line: u32, character: u32, include_declaration: bool) -> Result<Vec<Location>> {
        let result: Value = self.request(
            "textDocument/references",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
                "context": { "includeDeclaration": include_declaration },
            }),
            Duration::from_secs(60),
        )?;
        Ok(self.parse_locations(&result))
    }

    /// Requests hover information for `(line, character)` in `uri`, normalising
    /// the several `Hover.contents` shapes to one string (see
    /// [`crate::lsp_types::parse_hover`]). `None` means the server had nothing
    /// to show. Same `character` convention as [`Self::definition`].
    pub fn hover(&mut self, uri: &str, line: u32, character: u32) -> Result<Option<Hover>> {
        let result: Value = self.request(
            "textDocument/hover",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
            }),
            Duration::from_secs(60),
        )?;
        let encoding = self.position_encoding;
        let mut cache = LineCache::default();
        Ok(lsp_types::parse_hover(&result, uri, encoding, |u, l| cache.line(u, l)))
    }

    /// Requests completion candidates at `(line, character)` in `uri`,
    /// handling both a bare `CompletionItem[]` and a `CompletionList { items }`
    /// (see [`crate::lsp_types::parse_completion`]). Same `character`
    /// convention as [`Self::definition`].
    pub fn completion(&mut self, uri: &str, line: u32, character: u32) -> Result<Vec<CompletionItem>> {
        let result: Value = self.request(
            "textDocument/completion",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
            }),
            Duration::from_secs(60),
        )?;
        Ok(lsp_types::parse_completion(&result))
    }

    /// Drains any messages the reader thread has queued without blocking,
    /// processing each through [`Self::handle_incoming`].
    ///
    /// This is how pushed [`textDocument/publishDiagnostics`] notifications get
    /// captured *outside* of a request: nothing consumes the channel except a
    /// live `request()` loop and this method, so a caller polling for fresh
    /// diagnostics must call it (or issue some request) to let queued
    /// notifications land in the store. [`Self::diagnostics_for`] does this for
    /// you.
    pub fn pump_notifications(&mut self) -> Result<()> {
        while let Ok(msg) = self.rx.try_recv() {
            self.handle_incoming(&msg)?;
        }
        Ok(())
    }

    /// Returns the latest diagnostics captured for `uri`, converted to
    /// `char`-offset ranges. Pumps pending notifications first so a caller
    /// sees everything the server has pushed up to now.
    ///
    /// This reads from a *store*, not from a request/response: see
    /// [`DiagnosticsStore`] for why diagnostics work fundamentally differently
    /// from every other method here.
    pub fn diagnostics_for(&mut self, uri: &str) -> Result<Vec<crate::lsp_types::Diagnostic>> {
        self.pump_notifications()?;
        let encoding = self.position_encoding;
        // Clone the small raw set out so the parser can borrow a fresh line
        // cache without also holding an immutable borrow of `self`.
        let raw = self.diagnostics.raw_for(uri).to_vec();
        let mut cache = LineCache::default();
        Ok(raw
            .iter()
            .filter_map(|d| lsp_types::parse_diagnostic(d, uri, encoding, |u, l| cache.line(u, l)))
            .collect())
    }

    /// Shared tail of [`Self::definition`] and [`Self::references`]: convert a
    /// definition/reference JSON response to `char`-offset [`Location`]s,
    /// resolving each target file's line text from disk on demand.
    fn parse_locations(&self, result: &Value) -> Vec<Location> {
        let encoding = self.position_encoding;
        let mut cache = LineCache::default();
        lsp_types::parse_locations(result, encoding, |u, l| cache.line(u, l))
    }

    pub fn shutdown(&mut self) -> Result<()> {
        let _ack: Value = self.request("shutdown", Value::Null, Duration::from_secs(10))?;
        self.notify("exit", Value::Null)
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
