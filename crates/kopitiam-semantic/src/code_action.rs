//! LSP code actions, the pure half: building the `textDocument/codeAction`
//! request, reading the response, and deciding what to *do* with the action
//! the user picked — resolve it, apply its edit, or run its command.
//!
//! Everything here is plain JSON in, plain JSON (or a decision) out, no I/O,
//! no server. The wire half — actually sending the request, `codeAction/resolve`,
//! `workspace/executeCommand` — is [`crate::lsp_client::LspClient`]'s job, and
//! the path/`char`-offset front door is
//! [`crate::RustAnalyzerSession::code_actions_in_range`] /
//! [`crate::RustAnalyzerSession::apply_code_action`]. Kept separate so the
//! request/response mapping can be unit-tested without spawning a server, ah.
//!
//! # Where the rules come from (so nobody invent them again)
//!
//! The flow follows the LSP 3.17 spec (`textDocument/codeAction`,
//! `codeAction/resolve`, `workspace/executeCommand`) and, for the choices the
//! spec leaves open, Neovim's own client — `runtime/lua/vim/lsp/buf.lua`,
//! `M.code_action` and `on_code_action_results` (Neovim 0.12, Apache-2.0;
//! studied, not copied — this is a clean-room Rust re-statement of the same
//! decisions):
//!
//! * **Context diagnostics** are the pushed diagnostics on the requested
//!   *lines* (Neovim: `vim.diagnostic.get(bufnr, { lnum = lnum })`). We filter
//!   by line overlap only, which is encoding-independent: comparing columns
//!   would need both sides in the same wire unit, and line numbers are the same
//!   in every encoding. A server is free to ignore the context (rust-analyzer
//!   computes its own fixes), but the spec says send it, so we send it.
//! * **`triggerKind: Invoked`** (1) — the user asked explicitly, so a server
//!   may also return *disabled* actions with a reason. Neovim sets the same.
//! * **Resolve** only when the server advertised
//!   `codeActionProvider.resolveProvider` and the action is not already
//!   complete (Neovim: `not (action.edit and action.command)`). A bare
//!   `Command` (title + string `command`) is never resolved — only
//!   `CodeAction` literals are resolvable.
//! * **Disabled** actions are refused with the server's own reason, never
//!   applied.
//! * **Edit first, then command** — the spec's order when a `CodeAction` has
//!   both.

use anyhow::{Result, bail};
use serde_json::{Value, json};

/// `CodeActionTriggerKind.Invoked` (LSP 3.17): the user asked for code actions
/// explicitly (a keymap, a menu), as opposed to `Automatic` (2, lightbulb-style
/// background polling). See the module docs for why it matters.
pub(crate) const TRIGGER_KIND_INVOKED: u8 = 1;

/// One entry from a `textDocument/codeAction` response: either a bare LSP
/// `Command` or a `CodeAction` literal, kept as its raw JSON so
/// [`crate::RustAnalyzerSession::apply_code_action`] can dispatch on its shape
/// (and so `codeAction/resolve` can send it back to the server byte-for-byte,
/// `data` field and all — the server needs that `data` to find the action
/// again).
///
/// `title` is public for listing; everything else is read through accessors so
/// the raw JSON stays the single source of truth.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeAction {
    pub title: String,
    raw: Value,
    /// Set when this value came back from `codeAction/resolve`, so it is never
    /// sent round again (an edit-only action would otherwise still read as
    /// "needs resolve" under Neovim's rule).
    resolved: bool,
}

/// What applying a [`CodeAction`] did, in full — the detailed result of
/// [`crate::RustAnalyzerSession::apply_code_action_detailed`].
#[derive(Debug, Default)]
pub struct AppliedCodeAction {
    /// The computed (NOT yet written) edits, one per touched file. Empty when
    /// the action was a command.
    pub edits: Vec<crate::edit::FileEdit>,
    /// A command the action carries *alongside* its edit, to be run — via
    /// [`crate::RustAnalyzerSession::execute_command`] — only after the caller
    /// has written `edits` (the spec's edit-then-command order). `None` for
    /// most actions.
    pub follow_up_command: Option<(String, Value)>,
    /// The command id that was executed, when the action was command-only. Its
    /// edits (if any) were already written to disk by the server's
    /// `workspace/applyEdit`, so the caller should reload from disk.
    pub ran_command: Option<String>,
}

impl CodeAction {
    /// Wraps one raw `(Command | CodeAction)` JSON entry. Public so a caller
    /// (or a test) holding JSON from elsewhere can build one; the title falls
    /// back to `"(untitled)"` for a malformed entry rather than failing the
    /// whole list.
    pub fn from_raw(raw: Value) -> Self {
        let title = raw.get("title").and_then(Value::as_str).unwrap_or("(untitled)").to_string();
        Self { title, raw, resolved: false }
    }

    /// Wraps the completed action `codeAction/resolve` returned; such a value
    /// never needs resolving again.
    pub(crate) fn from_resolved(raw: Value) -> Self {
        Self { resolved: true, ..Self::from_raw(raw) }
    }

    /// The raw JSON as the server sent it (or as `codeAction/resolve` returned
    /// it).
    pub fn raw(&self) -> &Value {
        &self.raw
    }

    /// The action's `kind` (`"quickfix"`, `"refactor.rewrite"`,
    /// `"source.organizeImports"`, …), or `None` for a bare `Command` or an
    /// action the server left unkinded.
    pub fn kind(&self) -> Option<&str> {
        self.raw.get("kind").and_then(Value::as_str)
    }

    /// Whether the server marked this the preferred fix (`isPreferred`) — the
    /// one a "fix it" shortcut would pick without asking.
    pub fn is_preferred(&self) -> bool {
        self.raw.get("isPreferred").and_then(Value::as_bool).unwrap_or(false)
    }

    /// The server's reason this action cannot be applied right now, when it
    /// sent the action `disabled` (only ever on an `Invoked` request). A
    /// disabled action is listed so the user sees *why*, but never applied.
    pub fn disabled_reason(&self) -> Option<&str> {
        self.raw.pointer("/disabled/reason").and_then(Value::as_str)
    }

    /// Whether this is a bare LSP `Command` (`{title, command: "<id>",
    /// arguments?}`) rather than a `CodeAction` literal. Distinguished by
    /// `command` being a *string*: a `CodeAction`'s own `command` field is an
    /// object. A bare command has no edit to preview — running it is the only
    /// thing to do, and the server pushes any resulting edit back to us as
    /// `workspace/applyEdit`.
    pub fn is_bare_command(&self) -> bool {
        self.raw.get("command").is_some_and(Value::is_string)
    }

    /// Whether the action already carries a `WorkspaceEdit`.
    pub fn has_edit(&self) -> bool {
        self.raw.get("edit").is_some_and(|e| !e.is_null())
    }

    /// The `(command id, arguments)` this action runs, if any — for a bare
    /// `Command` the entry itself, for a `CodeAction` its nested `command`
    /// object. `arguments` is `Value::Null` when the server sent none.
    pub fn command(&self) -> Option<(String, Value)> {
        match self.raw.get("command")? {
            Value::String(id) => Some((id.clone(), self.raw.get("arguments").cloned().unwrap_or(Value::Null))),
            obj @ Value::Object(_) => {
                let id = obj.get("command").and_then(Value::as_str)?.to_string();
                Some((id, obj.get("arguments").cloned().unwrap_or(Value::Null)))
            }
            _ => None,
        }
    }

    /// Whether this action should go through `codeAction/resolve` before it is
    /// applied, given whether the server advertised `resolveProvider`.
    ///
    /// Neovim's rule, restated: resolve a `CodeAction` literal unless it
    /// already has *both* an edit and a command (nothing left to fill in). A
    /// bare `Command` is never resolvable. rust-analyzer, told the client
    /// supports lazy `edit` resolution, sends its assists with only `data`, so
    /// for it this is the common path, not the exception.
    pub fn needs_resolve(&self, server_resolves: bool) -> bool {
        server_resolves
            && !self.resolved
            && !self.is_bare_command()
            && !(self.has_edit() && self.command().is_some())
    }
}

/// What to do with a chosen [`CodeAction`], decided purely from its JSON —
/// the dispatch [`crate::RustAnalyzerSession::apply_code_action`] follows.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CodeActionPlan {
    /// The server said no; surface its reason.
    Refuse(String),
    /// Apply this `WorkspaceEdit`. A trailing command, if the action also has
    /// one, is the caller's to run *after* writing the edit (see
    /// [`CodeAction::command`]).
    Edit(Value),
    /// Run this command via `workspace/executeCommand`; any edit comes back as
    /// a server-initiated `workspace/applyEdit`.
    Command { command: String, arguments: Value },
    /// Neither an edit nor a command — even after resolve. Nothing to apply.
    Nothing,
}

/// Decides how to apply an (already resolved, where resolving applies) action.
pub(crate) fn plan(action: &CodeAction) -> CodeActionPlan {
    if let Some(reason) = action.disabled_reason() {
        return CodeActionPlan::Refuse(reason.to_string());
    }
    if action.has_edit() {
        return CodeActionPlan::Edit(action.raw["edit"].clone());
    }
    match action.command() {
        Some((command, arguments)) => CodeActionPlan::Command { command, arguments },
        None => CodeActionPlan::Nothing,
    }
}

/// The pushed diagnostics (raw LSP JSON, wire units) that touch lines
/// `start_line..=end_line` — the `context.diagnostics` for a code-action
/// request over that range. Line-overlap only; see the module docs for why
/// columns are deliberately not compared.
pub(crate) fn diagnostics_on_lines(raw: &[Value], start_line: u32, end_line: u32) -> Vec<Value> {
    let (lo, hi) = if start_line <= end_line { (start_line, end_line) } else { (end_line, start_line) };
    raw.iter()
        .filter(|d| {
            let line = |p: &str| d.pointer(p).and_then(Value::as_u64).map(|l| l as u32);
            match (line("/range/start/line"), line("/range/end/line")) {
                (Some(s), Some(e)) => s <= hi && e >= lo,
                _ => false, // a diagnostic with no usable range cannot be placed; skip it
            }
        })
        .cloned()
        .collect()
}

/// The `CodeActionParams` for `uri` over the wire-unit range
/// `(start_line, start_character)..(end_line, end_character)`, carrying
/// `diagnostics` (raw, as the server published them) and `triggerKind:
/// Invoked` in its context.
pub(crate) fn request_params(
    uri: &str,
    (start_line, start_character): (u32, u32),
    (end_line, end_character): (u32, u32),
    diagnostics: Vec<Value>,
) -> Value {
    json!({
        "textDocument": { "uri": uri },
        "range": {
            "start": { "line": start_line, "character": start_character },
            "end": { "line": end_line, "character": end_character },
        },
        "context": { "diagnostics": diagnostics, "triggerKind": TRIGGER_KIND_INVOKED },
    })
}

/// Reads a `textDocument/codeAction` response: `(Command | CodeAction)[] |
/// null`. `null` means "nothing here", not an error.
pub(crate) fn parse_response(result: Value) -> Result<Vec<Value>> {
    match result {
        Value::Array(items) => Ok(items),
        Value::Null => Ok(Vec::new()),
        other => bail!("unexpected textDocument/codeAction response shape: {other}"),
    }
}

/// The `textDocument.codeAction` client capability this client advertises.
///
/// * `codeActionLiteralSupport` — we understand `CodeAction` literals, not
///   only bare `Command`s. Without it a server may only send commands.
/// * `resolveSupport.properties: ["edit"]` — the server may leave `edit` out
///   and let us fetch it via `codeAction/resolve` for just the action the user
///   picks. rust-analyzer uses this to avoid computing every assist's edit up
///   front; [`CodeAction::needs_resolve`] is the other half of the contract.
/// * `dataSupport` — we round-trip the `data` field untouched (the resolve
///   request sends the raw action back), which resolve relies on.
/// * `disabledSupport` / `isPreferredSupport` — we show both
///   ([`CodeAction::disabled_reason`], [`CodeAction::is_preferred`]).
pub(crate) fn client_capability() -> Value {
    json!({
        "dynamicRegistration": false,
        "codeActionLiteralSupport": {
            "codeActionKind": {
                "valueSet": [
                    "", "quickfix", "refactor", "refactor.extract", "refactor.inline",
                    "refactor.rewrite", "source", "source.organizeImports",
                ],
            },
        },
        "isPreferredSupport": true,
        "disabledSupport": true,
        "dataSupport": true,
        "resolveSupport": { "properties": ["edit"] },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(start: u64, end: u64, msg: &str) -> Value {
        json!({
            "range": { "start": { "line": start, "character": 0 }, "end": { "line": end, "character": 3 } },
            "message": msg,
            "code": "unused_imports",
        })
    }

    #[test]
    fn request_params_carry_range_diagnostics_and_invoked_trigger() {
        let d = diag(0, 0, "unused import");
        let p = request_params("file:///x/src/lib.rs", (0, 4), (2, 1), vec![d.clone()]);
        assert_eq!(p["textDocument"]["uri"], "file:///x/src/lib.rs");
        assert_eq!(p["range"]["start"], json!({ "line": 0, "character": 4 }));
        assert_eq!(p["range"]["end"], json!({ "line": 2, "character": 1 }));
        assert_eq!(p["context"]["diagnostics"], json!([d]), "diagnostics go over raw, code and all");
        assert_eq!(p["context"]["triggerKind"], 1);
    }

    #[test]
    fn diagnostics_are_picked_by_line_overlap_only() {
        let raw = vec![diag(0, 0, "a"), diag(3, 5, "b"), diag(9, 9, "c"), json!({ "message": "no range" })];
        let msgs = |v: Vec<Value>| v.iter().map(|d| d["message"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!(msgs(diagnostics_on_lines(&raw, 0, 0)), ["a"]);
        // A cursor on line 4 sits inside the 3..=5 diagnostic.
        assert_eq!(msgs(diagnostics_on_lines(&raw, 4, 4)), ["b"]);
        // A selection spanning 0..=3 touches both a and b; reversed bounds also fine.
        assert_eq!(msgs(diagnostics_on_lines(&raw, 3, 0)), ["a", "b"]);
        assert!(diagnostics_on_lines(&raw, 6, 8).is_empty());
    }

    #[test]
    fn response_null_is_empty_and_junk_is_an_error() {
        assert!(parse_response(Value::Null).unwrap().is_empty());
        assert_eq!(parse_response(json!([{ "title": "x" }])).unwrap().len(), 1);
        assert!(parse_response(json!({ "title": "x" })).is_err());
    }

    #[test]
    fn a_bare_command_is_never_resolved_and_plans_to_execute() {
        let a = CodeAction::from_raw(json!({ "title": "Run", "command": "rust-analyzer.run", "arguments": [1] }));
        assert!(a.is_bare_command());
        assert!(!a.needs_resolve(true));
        assert_eq!(
            plan(&a),
            CodeActionPlan::Command { command: "rust-analyzer.run".into(), arguments: json!([1]) }
        );
    }

    #[test]
    fn a_data_only_literal_needs_resolve_only_if_the_server_resolves() {
        // rust-analyzer's lazy assist shape: kind + data, no edit yet.
        let a = CodeAction::from_raw(json!({
            "title": "Fill match arms", "kind": "quickfix", "data": { "id": ["add_missing_match_arms", "QuickFix", 0] }
        }));
        assert_eq!(a.kind(), Some("quickfix"));
        assert!(!a.is_bare_command());
        assert!(a.needs_resolve(true));
        assert!(!a.needs_resolve(false), "never resolve against a server that did not advertise it");
        assert_eq!(plan(&a), CodeActionPlan::Nothing, "unresolved, there is nothing to apply yet");
    }

    #[test]
    fn edit_wins_over_command_and_complete_actions_skip_resolve() {
        let edit = json!({ "changes": {} });
        let a = CodeAction::from_raw(json!({
            "title": "Both", "edit": edit.clone(),
            "command": { "title": "after", "command": "x.after", "arguments": ["z"] }
        }));
        assert!(!a.needs_resolve(true), "edit and command both present: nothing to resolve");
        assert_eq!(plan(&a), CodeActionPlan::Edit(edit));
        assert_eq!(a.command(), Some(("x.after".to_string(), json!(["z"]))), "trailing command stays visible");
        // Edit only, no command: still resolvable (Neovim resolves unless both are present).
        let b = CodeAction::from_raw(json!({ "title": "Edit only", "edit": { "changes": {} } }));
        assert!(b.needs_resolve(true));
        // ...but once it has come back from `codeAction/resolve`, never again.
        let r = CodeAction::from_resolved(b.raw().clone());
        assert!(!r.needs_resolve(true));
    }

    #[test]
    fn disabled_actions_are_refused_with_the_servers_reason() {
        let a = CodeAction::from_raw(json!({
            "title": "Extract", "edit": { "changes": {} }, "disabled": { "reason": "select an expression first" }
        }));
        assert_eq!(a.disabled_reason(), Some("select an expression first"));
        assert_eq!(plan(&a), CodeActionPlan::Refuse("select an expression first".into()));
    }

    #[test]
    fn preferred_flag_and_untitled_fallback() {
        let a = CodeAction::from_raw(json!({ "title": "Remove import", "isPreferred": true }));
        assert!(a.is_preferred());
        assert_eq!(CodeAction::from_raw(json!({})).title, "(untitled)");
    }

    #[test]
    fn capability_advertises_literals_and_lazy_edit_resolution() {
        let c = client_capability();
        assert_eq!(c["resolveSupport"]["properties"], json!(["edit"]));
        assert_eq!(c["dataSupport"], true);
        assert!(c.pointer("/codeActionLiteralSupport/codeActionKind/valueSet").is_some());
    }
}
