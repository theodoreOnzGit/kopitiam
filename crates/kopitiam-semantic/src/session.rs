//! A higher-level, file-path-based API over [`LspClient`] for the
//! operations that mutate a project: rename and code actions.
//!
//! [`RustAnalyzerProvider`](crate::RustAnalyzerProvider) only ever *reads*
//! facts. [`RustAnalyzerSession`] is the write-capable counterpart used by
//! `kopitiam`'s `rename` and `code-actions` subcommands: it speaks in
//! [`Path`]s and plain strings instead of raw JSON-RPC, and always returns
//! computed [`FileEdit`]s rather than writing them — callers decide when
//! (and whether) to call [`apply_workspace_edit`] or
//! [`RustAnalyzerSession::apply`].

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::edit::{self, FileEdit};
use crate::lsp_client::LspClient;
use crate::lsp_types::{CompletionItem, Diagnostic, Hover, Location};
use crate::position;

/// One entry from a `textDocument/codeAction` response: either a `Command`
/// or a `CodeAction` per the LSP spec, kept as its raw JSON so
/// [`RustAnalyzerSession::apply_code_action`] can dispatch on its shape.
pub struct CodeAction {
    pub title: String,
    raw: Value,
}

/// A live rust-analyzer process, ready to answer rename and code-action
/// requests for the project rooted at the path given to [`Self::connect`].
pub struct RustAnalyzerSession {
    client: LspClient,
}

impl RustAnalyzerSession {
    /// Spawns `rust-analyzer` and waits for it to index `root`, exactly
    /// like [`crate::RustAnalyzerProvider`]. This can take anywhere from a
    /// couple of seconds to a couple of minutes on a large workspace.
    pub fn connect(root: &Path) -> Result<Self> {
        Self::connect_with_binary("rust-analyzer", root)
    }

    pub fn connect_with_binary(binary: &str, root: &Path) -> Result<Self> {
        Self::connect_with(binary, &[], root, Duration::from_secs(180))
    }

    /// The general connect used by both the convenience constructors above and
    /// the asynchronous [`crate::AsyncRustAnalyzerSession`]: spawns `binary`
    /// with `args`, performs the handshake against `root`, and blocks up to
    /// `index_timeout` for indexing (see [`LspClient::spawn_with_args`] and the
    /// readiness discussion in `docs/ai-decisions/AID-0022`).
    ///
    /// `args` exists because non-rust-analyzer servers need command-line flags
    /// to speak LSP on stdio (pyright's `--stdio`, OmniSharp's `-lsp`, …); an
    /// explicit `index_timeout` exists because the async front end and its
    /// tests want to bound the connect, rather than inheriting the 180 s default
    /// baked into [`Self::connect`].
    pub fn connect_with(binary: &str, args: &[&str], root: &Path, index_timeout: Duration) -> Result<Self> {
        Self::connect_with_observed(binary, args, root, index_timeout, |_| {})
    }

    /// Like [`Self::connect_with`], but forwards every start-up `$/progress`
    /// ([`ProgressUpdate`](crate::ProgressUpdate)) to `observer` while the
    /// server connects and indexes. Used by the asynchronous
    /// [`crate::AsyncRustAnalyzerSession`] to drive kvim's startup progress bar;
    /// the synchronous constructors pass a no-op. See
    /// [`LspClient::spawn_with_args_observed`].
    pub fn connect_with_observed(
        binary: &str,
        args: &[&str],
        root: &Path,
        index_timeout: Duration,
        observer: impl FnMut(crate::lsp_client::ProgressUpdate),
    ) -> Result<Self> {
        let client = LspClient::spawn_with_args_observed(binary, args, root, index_timeout, observer)?;
        Ok(Self { client })
    }

    /// Drains any server notifications the reader thread has queued — chiefly
    /// pushed `textDocument/publishDiagnostics` — into the client's diagnostics
    /// store, without issuing a request.
    ///
    /// A synchronous caller never needs this because [`Self::diagnostics`]
    /// pumps first, but the asynchronous [`crate::AsyncRustAnalyzerSession`]
    /// worker calls it on its idle tick so diagnostics keep flowing into the
    /// store even while the caller issues no requests — which is exactly the
    /// "a file you only open and read still shows diagnostics" behaviour
    /// `docs/ai-decisions/AID-0023` is about.
    pub fn pump(&mut self) -> Result<()> {
        self.client.pump_notifications()
    }

    /// Computes (but does not write) the edit that would rename the symbol
    /// at `file:line:character` to `new_name`. `line` and `character` are
    /// both 0-indexed; `character` is a Unicode scalar value (`char`)
    /// offset — plain `chars()` indexing, counting characters rather than
    /// bytes or UTF-16 code units. This method is responsible for
    /// converting that `char` offset to whatever wire encoding the server
    /// actually negotiated (see [`crate::position`]) before it goes out,
    /// and converting the response's positions back to `char` offsets
    /// before computing the edit — callers never need to think about the
    /// wire encoding at all.
    pub fn rename(&mut self, file: &Path, line: u32, character: u32, new_name: &str) -> Result<Vec<FileEdit>> {
        let (uri, text) = self.open(file)?;
        let encoding = self.client.position_encoding();
        let line_text = text.lines().nth(line as usize).unwrap_or("");
        let wire_character = position::char_col_to_unit(line_text, character, encoding);
        let raw_edit = self.client.rename(&uri, line, wire_character, new_name)?;
        edit::compute_workspace_edit(&raw_edit, encoding)
    }

    /// Lists the code actions available at `file:line:character`, in the
    /// same `char`-offset units as [`Self::rename`] — see that method's
    /// docs for what that means and why.
    pub fn code_actions(&mut self, file: &Path, line: u32, character: u32) -> Result<Vec<CodeAction>> {
        let (uri, text) = self.open(file)?;
        let encoding = self.client.position_encoding();
        let line_text = text.lines().nth(line as usize).unwrap_or("");
        let wire_character = position::char_col_to_unit(line_text, character, encoding);
        let raw_actions = self.client.code_actions(&uri, line, wire_character, line, wire_character)?;
        Ok(raw_actions
            .into_iter()
            .map(|raw| CodeAction {
                title: raw.get("title").and_then(Value::as_str).unwrap_or("(untitled)").to_string(),
                raw,
            })
            .collect())
    }

    /// Applies a [`CodeAction`] returned by [`Self::code_actions`].
    ///
    /// A `CodeAction` (or bare `Command`) carries its change one of two
    /// ways: an `edit` field with a `WorkspaceEdit` already computed, or a
    /// `command` to run via `workspace/executeCommand` — in which case the
    /// server computes the edit lazily and pushes it back to us as a
    /// `workspace/applyEdit` request, which [`LspClient`] answers (and thus
    /// writes to disk) as part of running the command. In that second case
    /// this method returns an empty list: there is nothing left to preview,
    /// the write already happened.
    pub fn apply_code_action(&mut self, action: &CodeAction) -> Result<Vec<FileEdit>> {
        if let Some(edit_value) = action.raw.get("edit") {
            return edit::compute_workspace_edit(edit_value, self.client.position_encoding());
        }
        if let Some(command_value) = action.raw.get("command") {
            let (command, arguments) = match command_value {
                Value::String(id) => (id.clone(), Value::Null),
                Value::Object(_) => (
                    command_value.get("command").and_then(Value::as_str).unwrap_or_default().to_string(),
                    command_value.get("arguments").cloned().unwrap_or(Value::Null),
                ),
                other => bail!("unexpected `command` shape in code action: {other}"),
            };
            self.client.execute_command(&command, arguments)?;
            return Ok(Vec::new());
        }
        bail!("code action `{}` has neither `edit` nor `command`", action.title)
    }

    /// Resolves the definition(s) of the symbol at `file:line:character`, in
    /// the same `char`-offset units as [`Self::rename`]. Opens the document
    /// from disk first (the LSP-correct precondition), converts the query
    /// `character` to the negotiated wire encoding, and returns
    /// [`Location`]s whose ranges are already back in `char` offsets — the
    /// caller works entirely in paths and `char` columns, never URIs or wire
    /// encodings.
    ///
    /// This is the method `kopitiam-neovim`'s `lsp/client.rs` documents as the
    /// upstream it is waiting for (`textDocument/definition`). It handles the
    /// `Location | Location[] | LocationLink[]` response union internally.
    pub fn definition(&mut self, file: &Path, line: u32, character: u32) -> Result<Vec<Location>> {
        let (uri, wire_character) = self.open_and_convert(file, line, character)?;
        self.client.definition(&uri, line, wire_character)
    }

    /// Finds all references to the symbol at `file:line:character`, including
    /// its declaration when `include_declaration` is set. Same unit and
    /// lifecycle conventions as [`Self::definition`]. Mirrors the signature
    /// `kopitiam-neovim`'s `lsp/client.rs` asks for.
    pub fn references(&mut self, file: &Path, line: u32, character: u32, include_declaration: bool) -> Result<Vec<Location>> {
        let (uri, wire_character) = self.open_and_convert(file, line, character)?;
        self.client.references(&uri, line, wire_character, include_declaration)
    }

    /// Requests hover text for `file:line:character` (type/signature/doc), or
    /// `None` if the server has nothing to show. Same conventions as
    /// [`Self::definition`]; the several LSP hover-content shapes are already
    /// normalised to one string.
    pub fn hover(&mut self, file: &Path, line: u32, character: u32) -> Result<Option<Hover>> {
        let (uri, wire_character) = self.open_and_convert(file, line, character)?;
        self.client.hover(&uri, line, wire_character)
    }

    /// Requests completion candidates at `file:line:character`. Same
    /// conventions as [`Self::definition`]; both the bare-array and
    /// `CompletionList` response shapes are handled.
    ///
    /// (`kopitiam-neovim`'s doc comment guessed this would return raw
    /// `Vec<Value>`; it returns typed [`CompletionItem`]s instead — a caller
    /// should never have to re-parse JSON, which is the whole point of this
    /// layer.)
    pub fn completion(&mut self, file: &Path, line: u32, character: u32) -> Result<Vec<CompletionItem>> {
        let (uri, wire_character) = self.open_and_convert(file, line, character)?;
        self.client.completion(&uri, line, wire_character)
    }

    /// Announces `file`'s current `text` to the server as an open document.
    ///
    /// [`Self::definition`], [`Self::hover`] and friends open the document
    /// from **disk** on every call, which is correct for a saved file. Use
    /// this (and [`Self::did_change`]) instead when you want the server to see
    /// an **unsaved buffer**: open it once with the buffer's live text, then
    /// push edits with [`Self::did_change`] before querying. This is also the
    /// prerequisite for meaningful [`Self::diagnostics`] on an unsaved buffer.
    pub fn did_open(&mut self, file: &Path, text: &str) -> Result<()> {
        let uri = path_to_uri(file)?;
        self.client.did_open(&uri, text)
    }

    /// Pushes the full new `text` of an already-open `file` to the server
    /// (full-document sync). Pair with [`Self::did_open`] to keep the server's
    /// view of an unsaved buffer in step with the editor.
    pub fn did_change(&mut self, file: &Path, text: &str) -> Result<()> {
        let uri = path_to_uri(file)?;
        self.client.did_change(&uri, text)
    }

    /// Returns the diagnostics the server has most recently published for
    /// `file`, with `char`-offset ranges.
    ///
    /// Diagnostics are **pushed** by the server, not requested (see
    /// `DiagnosticsStore` in `crate::lsp_types`), so this reflects whatever has
    /// arrived so far. rust-analyzer publishes them asynchronously after
    /// indexing/analysis, so a caller typically opens the file
    /// ([`Self::did_open`]) and then polls this until results appear rather
    /// than expecting them synchronously.
    pub fn diagnostics(&mut self, file: &Path) -> Result<Vec<Diagnostic>> {
        let uri = path_to_uri(file)?;
        self.client.diagnostics_for(&uri)
    }

    pub fn shutdown(mut self) -> Result<()> {
        self.client.shutdown()
    }

    /// Opens `file` from disk and converts a `char`-offset query `character`
    /// on `line` to the server's negotiated wire encoding — the common
    /// preamble to every read-only request. Returns the document URI and the
    /// wire-unit `character` to send.
    fn open_and_convert(&mut self, file: &Path, line: u32, character: u32) -> Result<(String, u32)> {
        let (uri, text) = self.open(file)?;
        let encoding = self.client.position_encoding();
        let line_text = text.lines().nth(line as usize).unwrap_or("");
        let wire_character = position::char_col_to_unit(line_text, character, encoding);
        Ok((uri, wire_character))
    }

    /// Resolves `file` to a `file://` URI and tells the server its current
    /// on-disk contents via `textDocument/didOpen`, so rename/code-action
    /// requests operate on a document the server considers open — the
    /// LSP-correct precondition for edit-producing requests, even though
    /// rust-analyzer's own workspace-wide VFS makes it work either way.
    ///
    /// Returns the file's text along with its URI: [`Self::rename`] and
    /// [`Self::code_actions`] need the requested line's own text to convert
    /// their `char`-offset `character` argument to the server's negotiated
    /// wire encoding (see [`crate::position`]), and re-reading the file a
    /// second time for that would risk seeing different content than what
    /// `didOpen` just told the server about.
    fn open(&mut self, file: &Path) -> Result<(String, String)> {
        let uri = path_to_uri(file)?;
        let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        self.client.did_open(&uri, &text)?;
        Ok((uri, text))
    }
}

fn path_to_uri(path: &Path) -> Result<String> {
    let absolute = path
        .canonicalize()
        .with_context(|| format!("resolving {}", path.display()))?;
    url::Url::from_file_path(&absolute)
        .map(|url| url.to_string())
        .map_err(|()| anyhow::anyhow!("could not build a file:// URI for {}", absolute.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end proof that `textDocument/definition` actually works through
    /// a live `rust-analyzer`: build a tiny two-function crate where `caller`
    /// calls `greet`, ask for the definition of the `greet` call site, and
    /// assert the answer points back at `greet`'s own declaration.
    ///
    /// `#[ignore]`d: this spawns a real server and waits for it to index,
    /// which takes ~minutes. It is never part of the normal suite. Run
    /// deliberately with:
    ///
    /// ```text
    /// cargo test --release -p kopitiam-semantic -- --ignored live_rust_analyzer_resolves_a_definition
    /// ```
    #[test]
    #[ignore = "spawns a real rust-analyzer and waits for indexing; run with `-- --ignored`"]
    fn live_rust_analyzer_resolves_a_definition() {
        if !crate::lsp_client::binary_available("rust-analyzer") {
            eprintln!("rust-analyzer not on PATH; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"sem_def_test\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let lib = dir.path().join("src/lib.rs");
        // Line 0:  pub fn greet() -> &'static str {   <- definition of `greet`
        // Line 5:      greet()                        <- call site we query
        let source = "pub fn greet() -> &'static str {\n    \"hi\"\n}\n\npub fn caller() -> &'static str {\n    greet()\n}\n";
        std::fs::write(&lib, source).unwrap();

        let mut session = RustAnalyzerSession::connect(dir.path()).expect("rust-analyzer should start and index");

        // "greet" on the call-site line 5 starts at char column 4 (after the
        // four-space indent).
        let locations = session.definition(&lib, 5, 4).expect("definition request should succeed");

        assert!(!locations.is_empty(), "the call to `greet` must resolve to at least one definition");
        let def = &locations[0];
        assert_eq!(def.path.canonicalize().unwrap(), lib.canonicalize().unwrap());
        assert_eq!(def.range.start.line, 0, "greet is declared on line 0");
        assert_eq!(def.range.start.character, 7, "the identifier `greet` starts after `pub fn ` (7 chars)");

        session.shutdown().ok();
    }
}
