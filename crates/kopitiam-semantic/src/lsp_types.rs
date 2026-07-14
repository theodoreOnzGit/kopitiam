//! Typed, caller-facing results for the read-only LSP requests
//! ([`textDocument/definition`], [`references`], [`hover`], [`completion`],
//! [`publishDiagnostics`]), plus the *pure* JSON parsers that produce them.
//!
//! # Why this module is separate from [`crate::lsp_client`]
//!
//! The single hardest thing about an LSP client is not the transport — it is
//! that **the same request can come back in several different JSON shapes**,
//! and a client that only handles one shape silently returns nothing for the
//! others. `textDocument/definition` may reply with a lone `Location`, an
//! array of `Location`, *or* an array of `LocationLink` (a completely
//! different object, keyed `targetUri`/`targetSelectionRange` instead of
//! `uri`/`range`); `textDocument/completion` may reply with a bare
//! `CompletionItem[]` *or* a `CompletionList { items }` wrapper; `hover`
//! contents may be a `MarkupContent`, a `MarkedString` (itself either a plain
//! string or a `{ language, value }` object), or an array of `MarkedString`.
//! A server advertising `definition.linkSupport` — which this crate now does,
//! deliberately, in [`crate::lsp_client`]'s `initialize` — is *entitled* to
//! pick the `LocationLink` shape, so "rust-analyzer happens to send plain
//! `Location` today" is not a defence any more than it was for the position
//! encoding (see [`crate::position`]).
//!
//! Keeping the shape-handling here, as free functions over `&serde_json::Value`,
//! makes it directly unit-testable with hand-written JSON and a synthetic
//! line-text source — no live language server, no spawned process — which is
//! exactly where an LSP client's real bugs hide. [`crate::lsp_client`] then
//! calls these parsers, supplying a line-text closure backed by disk reads.
//!
//! # Positions are `char` offsets, converted here, once
//!
//! Every position these types expose is in this crate's public unit — a
//! Unicode scalar value (`char`) offset — never a wire encoding. The parsers
//! perform that conversion using [`crate::position::unit_to_char_col`],
//! exactly as [`crate::edit`] already does for `WorkspaceEdit`s, so a caller
//! never has to re-do an encoding conversion or even know one happened. The
//! conversion needs the *real text* of the target line (not arithmetic on two
//! integers — see [`crate::position`] for why), which is why every
//! position-bearing parser takes a `line_text` callback keyed by
//! `(uri, line)`: a `definition` result can point into a *different* file
//! than the one queried, so the line whose text we need is the target's, not
//! the request's.

use std::path::PathBuf;

use serde_json::Value;

use crate::position::{self, PositionEncoding};

/// A 0-based source position in this crate's public unit: `line` is a line
/// number and `character` is a Unicode scalar value (`char`) offset within
/// that line — plain `chars()` indexing, never bytes or UTF-16 code units
/// (see [`crate::position`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// A half-open `[start, end)` span of [`Position`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// A resolved `textDocument/definition` / `references` target: the file the
/// symbol lives in (already converted from its `file://` URI to a real
/// [`PathBuf`]) and the `char`-offset range within it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub path: PathBuf,
    pub range: Range,
}

/// A `textDocument/hover` result: the documentation/type text the server
/// produced, normalised to a single plain-or-markdown string regardless of
/// which of LSP's three `Hover.contents` shapes it arrived in, plus the
/// optional range the hover applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hover {
    pub contents: String,
    pub range: Option<Range>,
}

/// LSP's `CompletionItemKind` (1–25), spelled out as an enum so a completion
/// UI can pick an icon by matching rather than switching on a raw integer.
/// Unknown/future values map to [`Self::Text`], the spec's neutral default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionItemKind {
    Text,
    Method,
    Function,
    Constructor,
    Field,
    Variable,
    Class,
    Interface,
    Module,
    Property,
    Unit,
    Value,
    Enum,
    Keyword,
    Snippet,
    Color,
    File,
    Reference,
    Folder,
    EnumMember,
    Constant,
    Struct,
    Event,
    Operator,
    TypeParameter,
}

impl CompletionItemKind {
    fn from_lsp(n: i64) -> Self {
        match n {
            2 => Self::Method,
            3 => Self::Function,
            4 => Self::Constructor,
            5 => Self::Field,
            6 => Self::Variable,
            7 => Self::Class,
            8 => Self::Interface,
            9 => Self::Module,
            10 => Self::Property,
            11 => Self::Unit,
            12 => Self::Value,
            13 => Self::Enum,
            14 => Self::Keyword,
            15 => Self::Snippet,
            16 => Self::Color,
            17 => Self::File,
            18 => Self::Reference,
            19 => Self::Folder,
            20 => Self::EnumMember,
            21 => Self::Constant,
            22 => Self::Struct,
            23 => Self::Event,
            24 => Self::Operator,
            25 => Self::TypeParameter,
            _ => Self::Text, // 1 (Text) and anything a future spec adds
        }
    }
}

/// One `textDocument/completion` candidate, reduced to the fields a
/// completion UI actually consumes.
///
/// `insert_text` is what should land in the buffer if the item is accepted:
/// the server's explicit `insertText` when present, otherwise the `newText`
/// of its `textEdit`, otherwise `None` (the caller falls back to `label`).
/// Completion `textEdit` *ranges* are deliberately not exposed — they would
/// need per-item encoding conversion for a feature (replace-range-aware
/// insertion) no current caller uses; a caller that needs them should add a
/// range field here rather than re-parse the raw JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionItem {
    pub label: String,
    pub kind: Option<CompletionItemKind>,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub insert_text: Option<String>,
    /// Whether [`Self::insert_text`] is an **LSP snippet** (the item's
    /// `insertTextFormat` was `2` / `Snippet`) rather than plain text
    /// (`1` / `PlainText`, or absent — the spec's default).
    ///
    /// # Why the caller must not ignore this
    ///
    /// A snippet's `insertText` is written in the snippet *grammar*
    /// (`greet($0)`, `${1:name}`), not literal text. rust-analyzer sets it on
    /// function/method completions so accepting `greet` inserts `greet()` with
    /// the cursor between the parentheses. A client that inserts a snippet
    /// `insertText` verbatim types the literal characters `$0`/`${1:...}` into
    /// the buffer — a classic silent-wrong completion bug. This flag is what
    /// lets the editor route such an item through a snippet expander
    /// (`kopitiam-snippet`) instead of a plain insert. The completion `textEdit`
    /// path is a snippet too when `insertTextFormat == 2`, so this flag governs
    /// whichever of the two produced [`Self::insert_text`].
    pub is_snippet: bool,
}

/// LSP's `DiagnosticSeverity` (1–4), spelled out so a non-exhaustive match at
/// a call site is a compile error, not a silently-wrong default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Information,
    Hint,
}

impl Severity {
    fn from_lsp(n: i64) -> Self {
        match n {
            1 => Self::Error,
            2 => Self::Warning,
            3 => Self::Information,
            _ => Self::Hint, // 4 (Hint), or anything a future spec adds
        }
    }
}

/// One `textDocument/publishDiagnostics` entry, position-converted to
/// `char` offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: Severity,
    pub message: String,
    /// The producing tool, e.g. `"rustc"` or `"clippy"` for
    /// rust-analyzer-relayed diagnostics.
    pub source: Option<String>,
    /// The diagnostic code (`E0308`, a lint name, ...), normalised to a
    /// string whether the server sent it as a string or a number.
    pub code: Option<String>,
}

// ---------------------------------------------------------------------------
// Parsers. Pure over `&Value` + a `line_text` source; no I/O of their own.
// ---------------------------------------------------------------------------

/// Parses a `textDocument/definition` (or `references`) response, handling
/// **all three** legal shapes: a lone `Location` object, a `Location[]`, and
/// a `LocationLink[]`. `references` only ever uses the first two, but sharing
/// one parser means a server that (wrongly, but harmlessly) answers
/// `references` with links still works.
///
/// A malformed individual entry is dropped, not fatal: one bad element of an
/// array should not sink the whole result.
pub(crate) fn parse_locations(
    value: &Value,
    encoding: PositionEncoding,
    mut line_text: impl FnMut(&str, u32) -> Option<String>,
) -> Vec<Location> {
    match value {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| parse_one_location(item, encoding, &mut line_text))
            .collect(),
        // A single `Location`/`LocationLink` object (definition of a symbol
        // with exactly one target).
        Value::Object(_) => parse_one_location(value, encoding, &mut line_text).into_iter().collect(),
        // `null` — no definition/references found. Not an error.
        _ => Vec::new(),
    }
}

/// Parses either a `Location` (`{ uri, range }`) or a `LocationLink`
/// (`{ targetUri, targetSelectionRange | targetRange, ... }`).
///
/// For a `LocationLink` we prefer `targetSelectionRange` (the precise
/// identifier span the cursor should land on) and fall back to `targetRange`
/// (the whole symbol, e.g. an entire function) — this is the distinction that
/// makes go-to-definition land on the name rather than the opening brace.
fn parse_one_location(
    value: &Value,
    encoding: PositionEncoding,
    line_text: &mut impl FnMut(&str, u32) -> Option<String>,
) -> Option<Location> {
    if let Some(uri) = value.get("uri").and_then(Value::as_str) {
        // `Location`
        let range = parse_range(value.get("range")?, uri, encoding, line_text)?;
        return Some(Location { path: uri_to_path(uri)?, range });
    }
    if let Some(uri) = value.get("targetUri").and_then(Value::as_str) {
        // `LocationLink`
        let range_value = value.get("targetSelectionRange").or_else(|| value.get("targetRange"))?;
        let range = parse_range(range_value, uri, encoding, line_text)?;
        return Some(Location { path: uri_to_path(uri)?, range });
    }
    None
}

/// Parses a `textDocument/hover` response, normalising every legal
/// `Hover.contents` shape to one string:
///
/// * `MarkupContent` `{ kind, value }`  → its `value`;
/// * `MarkedString` as a plain string   → itself;
/// * `MarkedString` as `{ language, value }` → its `value`;
/// * an array of any of the above       → the parts joined by a blank line.
///
/// Returns `None` for a `null` response or one whose contents normalise to an
/// empty string (the server had nothing to say).
pub(crate) fn parse_hover(
    value: &Value,
    uri: &str,
    encoding: PositionEncoding,
    mut line_text: impl FnMut(&str, u32) -> Option<String>,
) -> Option<Hover> {
    let contents = normalise_markup(value.get("contents")?)?;
    if contents.is_empty() {
        return None;
    }
    let range = value.get("range").and_then(|r| parse_range(r, uri, encoding, &mut line_text));
    Some(Hover { contents, range })
}

/// Flattens LSP's `MarkupContent` / `MarkedString` / array-of-either union to
/// plain text. Shared by [`parse_hover`] and [`CompletionItem`] documentation.
fn normalise_markup(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        // Both `MarkupContent { kind, value }` and the object form of
        // `MarkedString { language, value }` carry the text under `value`.
        Value::Object(map) => map.get("value").and_then(Value::as_str).map(str::to_string),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().filter_map(normalise_markup).collect();
            Some(parts.join("\n\n"))
        }
        _ => None,
    }
}

/// Parses a `textDocument/completion` response, handling both a bare
/// `CompletionItem[]` and a `CompletionList { isIncomplete, items }` wrapper.
/// (`isIncomplete` is intentionally ignored: this crate does not re-trigger
/// incremental completion — a caller that wants to would drive it, not this
/// parser.)
pub(crate) fn parse_completion(value: &Value) -> Vec<CompletionItem> {
    let items = match value {
        Value::Array(items) => items.as_slice(),
        Value::Object(map) => map.get("items").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]),
        _ => &[],
    };
    items.iter().filter_map(parse_completion_item).collect()
}

fn parse_completion_item(value: &Value) -> Option<CompletionItem> {
    let label = value.get("label").and_then(Value::as_str)?.to_string();
    let kind = value.get("kind").and_then(Value::as_i64).map(CompletionItemKind::from_lsp);
    let detail = value.get("detail").and_then(Value::as_str).map(str::to_string);
    let documentation = value.get("documentation").and_then(normalise_markup).filter(|s| !s.is_empty());
    let insert_text = value
        .get("insertText")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/textEdit/newText").and_then(Value::as_str))
        .map(str::to_string);
    // `insertTextFormat`: 1 = PlainText, 2 = Snippet. Absent means PlainText
    // (the spec default), so anything that is not exactly `2` is plain.
    let is_snippet = value.get("insertTextFormat").and_then(Value::as_i64) == Some(2);
    Some(CompletionItem { label, kind, detail, documentation, insert_text, is_snippet })
}

/// Parses one `Diagnostic` object (an element of a `publishDiagnostics`
/// notification's `diagnostics` array). `uri` is the file the notification
/// was about — every diagnostic range is expressed against that file's lines.
pub(crate) fn parse_diagnostic(
    value: &Value,
    uri: &str,
    encoding: PositionEncoding,
    mut line_text: impl FnMut(&str, u32) -> Option<String>,
) -> Option<Diagnostic> {
    let range = parse_range(value.get("range")?, uri, encoding, &mut line_text)?;
    let severity = value.get("severity").and_then(Value::as_i64).map(Severity::from_lsp).unwrap_or(Severity::Error);
    let message = value.get("message").and_then(Value::as_str)?.to_string();
    let source = value.get("source").and_then(Value::as_str).map(str::to_string);
    let code = value.get("code").and_then(normalise_code);
    Some(Diagnostic { range, severity, message, source, code })
}

/// A diagnostic `code` is `integer | string` per the spec; normalise both to
/// a string so callers have one type to render.
fn normalise_code(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn parse_range(
    value: &Value,
    uri: &str,
    encoding: PositionEncoding,
    line_text: &mut impl FnMut(&str, u32) -> Option<String>,
) -> Option<Range> {
    let start = parse_position(value.get("start")?, uri, encoding, line_text)?;
    let end = parse_position(value.get("end")?, uri, encoding, line_text)?;
    Some(Range { start, end })
}

/// Converts one wire `{ line, character }` position to a `char`-offset
/// [`Position`], looking up the target line's own text (via `line_text`)
/// because the wire→`char` conversion depends on the bytes on that line, not
/// just the integer — see [`crate::position`].
fn parse_position(
    value: &Value,
    uri: &str,
    encoding: PositionEncoding,
    line_text: &mut impl FnMut(&str, u32) -> Option<String>,
) -> Option<Position> {
    let line = value.get("line")?.as_u64()? as u32;
    let wire_character = value.get("character")?.as_u64()? as u32;
    let text = line_text(uri, line).unwrap_or_default();
    let character = position::unit_to_char_col(&text, wire_character, encoding);
    Some(Position { line, character })
}

/// Resolves a `file://` URI to a real path. Returns `None` (dropping the
/// entry) rather than erroring for a non-`file` URI — some servers can return
/// `untitled:`/`jdt:` URIs for virtual documents this crate cannot open.
fn uri_to_path(uri: &str) -> Option<PathBuf> {
    url::Url::parse(uri).ok()?.to_file_path().ok()
}

/// The latest pushed diagnostics per document URI.
///
/// # Why this is not a request/response
///
/// Every other LSP feature here is a request: we send `textDocument/X` and
/// block for the matching response. Diagnostics do not work that way — the
/// server *pushes* `textDocument/publishDiagnostics` **notifications**
/// (no `id`, no reply) whenever its analysis of a file changes, which can be
/// long after any request we made and entirely unprompted. So there is
/// nothing to "return": the transport must instead *capture* these
/// notifications as they stream in and stash the most recent set per URI,
/// and a caller reads that store whenever it wants to paint squiggles.
///
/// The **replace, never append** semantics are load-bearing:
/// `publishDiagnostics` always carries the *complete* current set for a file,
/// so each notification supersedes the previous one for that URI. Appending
/// instead would double every diagnostic on each keystroke the server
/// re-analyses. An empty array is a real message — "this file is now clean" —
/// and must clear the file's prior diagnostics, which storing an empty vec
/// does (a query returns no entries).
#[derive(Debug, Default)]
pub(crate) struct DiagnosticsStore {
    /// Raw `Diagnostic` JSON objects, kept unparsed until queried: parsing a
    /// range to `char` offsets needs the file's line text, which is only
    /// worth reading when a caller actually asks for a given URI's
    /// diagnostics, not on every push.
    by_uri: std::collections::HashMap<String, Vec<Value>>,
}

impl DiagnosticsStore {
    /// Applies one `textDocument/publishDiagnostics` notification, replacing
    /// (not extending) the recorded set for its URI. A malformed notification
    /// missing `params.uri` is ignored.
    pub(crate) fn ingest_notification(&mut self, msg: &Value) {
        let Some(uri) = msg.pointer("/params/uri").and_then(Value::as_str) else {
            return;
        };
        let diagnostics = msg
            .pointer("/params/diagnostics")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        self.by_uri.insert(uri.to_string(), diagnostics);
    }

    /// The raw diagnostic objects currently recorded for `uri` (empty if the
    /// server never mentioned it, or last said it was clean).
    pub(crate) fn raw_for(&self, uri: &str) -> &[Value] {
        self.by_uri.get(uri).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A `line_text` source backed by an in-memory map of `uri -> the text of
    /// the referenced line`. Mirrors what [`crate::lsp_client`] does from
    /// disk, but lets the shape and encoding tests run with zero I/O. The
    /// provided string is returned for *any* line number of a matching URI —
    /// these tests only ever reference one line per file, and what matters for
    /// the wire→`char` conversion is that line's actual text, not its number.
    fn lines_from<'a>(entries: &'a [(&'a str, &'a str)]) -> impl FnMut(&str, u32) -> Option<String> + 'a {
        move |uri, _line| entries.iter().find(|(u, _)| *u == uri).map(|(_, text)| text.to_string())
    }

    // ---- definition/references shape handling --------------------------

    #[test]
    fn definition_parses_a_single_location_object() {
        let value = json!({
            "uri": "file:///tmp/lib.rs",
            "range": { "start": { "line": 3, "character": 4 }, "end": { "line": 3, "character": 9 } },
        });
        let locs = parse_locations(&value, PositionEncoding::Utf16, lines_from(&[("file:///tmp/lib.rs", "    greet()")]));
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].path, PathBuf::from("/tmp/lib.rs"));
        assert_eq!(locs[0].range.start, Position { line: 3, character: 4 });
        assert_eq!(locs[0].range.end, Position { line: 3, character: 9 });
    }

    #[test]
    fn definition_parses_a_location_array() {
        let value = json!([
            { "uri": "file:///tmp/a.rs", "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } } },
            { "uri": "file:///tmp/b.rs", "range": { "start": { "line": 1, "character": 2 }, "end": { "line": 1, "character": 3 } } },
        ]);
        let locs = parse_locations(&value, PositionEncoding::Utf16, lines_from(&[("file:///tmp/a.rs", "x"), ("file:///tmp/b.rs", "  y")]));
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0].path, PathBuf::from("/tmp/a.rs"));
        assert_eq!(locs[1].path, PathBuf::from("/tmp/b.rs"));
        assert_eq!(locs[1].range.start, Position { line: 1, character: 2 });
    }

    #[test]
    fn definition_parses_a_location_link_array_using_the_selection_range() {
        // The classic silent-failure trap: a server that answers with
        // `LocationLink[]` (keyed `targetUri`/`targetSelectionRange`) rather
        // than `Location[]`. A client expecting only `Location` gets nothing.
        let value = json!([
            {
                "originSelectionRange": { "start": { "line": 5, "character": 4 }, "end": { "line": 5, "character": 9 } },
                "targetUri": "file:///tmp/lib.rs",
                // Whole function body...
                "targetRange": { "start": { "line": 0, "character": 0 }, "end": { "line": 2, "character": 1 } },
                // ...but the cursor should land on the *name*.
                "targetSelectionRange": { "start": { "line": 0, "character": 7 }, "end": { "line": 0, "character": 12 } },
            }
        ]);
        let locs = parse_locations(&value, PositionEncoding::Utf16, lines_from(&[("file:///tmp/lib.rs", "pub fn greet() {}")]));
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].path, PathBuf::from("/tmp/lib.rs"));
        assert_eq!(
            locs[0].range.start,
            Position { line: 0, character: 7 },
            "a LocationLink must resolve to targetSelectionRange (the name), not targetRange (the whole item)"
        );
    }

    #[test]
    fn definition_link_falls_back_to_target_range_when_no_selection_range() {
        let value = json!([
            {
                "targetUri": "file:///tmp/lib.rs",
                "targetRange": { "start": { "line": 4, "character": 0 }, "end": { "line": 4, "character": 3 } },
            }
        ]);
        let locs = parse_locations(&value, PositionEncoding::Utf16, lines_from(&[("file:///tmp/lib.rs", "let x")]));
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].range.start, Position { line: 4, character: 0 });
    }

    #[test]
    fn definition_null_is_no_results_not_an_error() {
        let locs = parse_locations(&Value::Null, PositionEncoding::Utf16, lines_from(&[]));
        assert!(locs.is_empty());
    }

    #[test]
    fn definition_drops_a_non_file_uri_but_keeps_its_siblings() {
        let value = json!([
            { "uri": "jdt://contents/Foo.class", "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } } },
            { "uri": "file:///tmp/real.rs", "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } } },
        ]);
        let locs = parse_locations(&value, PositionEncoding::Utf16, lines_from(&[("file:///tmp/real.rs", "x")]));
        assert_eq!(locs.len(), 1, "a virtual (non-file) URI is dropped; the real one survives");
        assert_eq!(locs[0].path, PathBuf::from("/tmp/real.rs"));
    }

    // ---- position encoding, per parser ---------------------------------

    #[test]
    fn definition_converts_the_target_offset_from_each_wire_encoding() {
        // Target line: "日本語 = fn()" — the identifier 'f' of "fn" sits
        // after three CJK chars + " = " (char column 6). Feed the server's
        // wire offset for column 6 under each encoding; every one must come
        // back as char column 6.
        let line = "日本語 = fn";
        let uri = "file:///tmp/cjk.rs";
        for encoding in [PositionEncoding::Utf8, PositionEncoding::Utf16, PositionEncoding::Utf32] {
            let wire = position::char_col_to_unit(line, 6, encoding);
            let value = json!({
                "uri": uri,
                "range": { "start": { "line": 0, "character": wire }, "end": { "line": 0, "character": wire } },
            });
            let locs = parse_locations(&value, encoding, lines_from(&[(uri, line)]));
            assert_eq!(
                locs[0].range.start,
                Position { line: 0, character: 6 },
                "encoding {encoding:?}: wire offset {wire} on a CJK line must resolve to char column 6"
            );
        }
    }

    #[test]
    fn definition_converts_an_offset_after_an_astral_emoji() {
        // "🚀x": the 'x' is char column 1, but 3 UTF-16 units / 5 UTF-8 bytes
        // in (the rocket is a surrogate pair / 4 bytes).
        let line = "🚀x";
        let uri = "file:///tmp/emoji.rs";
        for encoding in [PositionEncoding::Utf8, PositionEncoding::Utf16, PositionEncoding::Utf32] {
            let wire = position::char_col_to_unit(line, 1, encoding);
            let value = json!({
                "uri": uri,
                "range": { "start": { "line": 0, "character": wire }, "end": { "line": 0, "character": wire } },
            });
            let locs = parse_locations(&value, encoding, lines_from(&[(uri, line)]));
            assert_eq!(locs[0].range.start.character, 1, "encoding {encoding:?}: char after 🚀 is column 1");
        }
    }

    // ---- hover shape handling ------------------------------------------

    #[test]
    fn hover_reads_markup_content() {
        let value = json!({ "contents": { "kind": "markdown", "value": "`fn greet()`" } });
        let hover = parse_hover(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[])).unwrap();
        assert_eq!(hover.contents, "`fn greet()`");
        assert!(hover.range.is_none());
    }

    #[test]
    fn hover_reads_a_plain_marked_string() {
        let value = json!({ "contents": "just text" });
        let hover = parse_hover(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[])).unwrap();
        assert_eq!(hover.contents, "just text");
    }

    #[test]
    fn hover_reads_a_marked_string_object() {
        let value = json!({ "contents": { "language": "rust", "value": "fn greet()" } });
        let hover = parse_hover(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[])).unwrap();
        assert_eq!(hover.contents, "fn greet()");
    }

    #[test]
    fn hover_reads_an_array_of_marked_strings_joined_by_blank_lines() {
        let value = json!({
            "contents": [
                { "language": "rust", "value": "fn greet()" },
                "Greets the caller.",
            ]
        });
        let hover = parse_hover(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[])).unwrap();
        assert_eq!(hover.contents, "fn greet()\n\nGreets the caller.");
    }

    #[test]
    fn hover_with_a_range_converts_it() {
        let value = json!({
            "contents": { "kind": "plaintext", "value": "x" },
            "range": { "start": { "line": 0, "character": 3 }, "end": { "line": 0, "character": 6 } },
        });
        let hover = parse_hover(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[("file:///tmp/x.rs", "日本語foo")])).unwrap();
        let range = hover.range.unwrap();
        assert_eq!(range.start, Position { line: 0, character: 3 });
    }

    #[test]
    fn hover_null_and_empty_contents_are_none() {
        assert!(parse_hover(&Value::Null, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[])).is_none());
        let empty = json!({ "contents": "" });
        assert!(parse_hover(&empty, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[])).is_none());
    }

    // ---- completion shape handling -------------------------------------

    #[test]
    fn completion_parses_a_bare_item_array() {
        let value = json!([
            { "label": "greet", "kind": 3, "detail": "fn() -> &str" },
            { "label": "greeting", "kind": 6 },
        ]);
        let items = parse_completion(&value);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].label, "greet");
        assert_eq!(items[0].kind, Some(CompletionItemKind::Function));
        assert_eq!(items[0].detail.as_deref(), Some("fn() -> &str"));
        assert_eq!(items[1].kind, Some(CompletionItemKind::Variable));
    }

    #[test]
    fn completion_parses_a_completion_list_wrapper() {
        let value = json!({
            "isIncomplete": true,
            "items": [ { "label": "foo", "kind": 5 } ],
        });
        let items = parse_completion(&value);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "foo");
        assert_eq!(items[0].kind, Some(CompletionItemKind::Field));
    }

    #[test]
    fn completion_prefers_insert_text_then_text_edit_new_text() {
        let explicit = json!([{ "label": "x", "insertText": "x()" }]);
        assert_eq!(parse_completion(&explicit)[0].insert_text.as_deref(), Some("x()"));

        let via_edit = json!([{
            "label": "y",
            "textEdit": { "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } }, "newText": "y!()" },
        }]);
        assert_eq!(parse_completion(&via_edit)[0].insert_text.as_deref(), Some("y!()"));

        let neither = json!([{ "label": "z" }]);
        assert_eq!(parse_completion(&neither)[0].insert_text, None);
    }

    #[test]
    fn completion_reads_the_insert_text_format_snippet_flag() {
        // insertTextFormat == 2 (Snippet): rust-analyzer's function completion,
        // whose insertText is snippet syntax the editor must expand, not insert.
        let snippet = json!([{ "label": "greet", "insertText": "greet($0)", "insertTextFormat": 2 }]);
        let item = &parse_completion(&snippet)[0];
        assert!(item.is_snippet, "insertTextFormat 2 must set is_snippet");
        assert_eq!(item.insert_text.as_deref(), Some("greet($0)"));

        // insertTextFormat == 1 (PlainText) and absent both mean "not a snippet".
        let plain = json!([{ "label": "x", "insertText": "x", "insertTextFormat": 1 }]);
        assert!(!parse_completion(&plain)[0].is_snippet, "insertTextFormat 1 is plain text");
        let absent = json!([{ "label": "y", "insertText": "y" }]);
        assert!(!parse_completion(&absent)[0].is_snippet, "absent insertTextFormat defaults to plain text");
    }

    #[test]
    fn completion_normalises_markup_documentation() {
        let value = json!([{ "label": "d", "documentation": { "kind": "markdown", "value": "docs here" } }]);
        assert_eq!(parse_completion(&value)[0].documentation.as_deref(), Some("docs here"));
    }

    #[test]
    fn completion_null_is_empty() {
        assert!(parse_completion(&Value::Null).is_empty());
    }

    // ---- diagnostics parsing + store -----------------------------------

    #[test]
    fn diagnostic_reads_severity_message_source_and_code() {
        let value = json!({
            "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 3 } },
            "severity": 2,
            "message": "unused variable `foo`",
            "source": "rustc",
            "code": "unused_variables",
        });
        let diag = parse_diagnostic(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[("file:///tmp/x.rs", "foo")])).unwrap();
        assert_eq!(diag.severity, Severity::Warning);
        assert_eq!(diag.message, "unused variable `foo`");
        assert_eq!(diag.source.as_deref(), Some("rustc"));
        assert_eq!(diag.code.as_deref(), Some("unused_variables"));
    }

    #[test]
    fn diagnostic_numeric_code_is_stringified_and_missing_severity_defaults_to_error() {
        let value = json!({
            "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } },
            "message": "mismatched types",
            "code": 308,
        });
        let diag = parse_diagnostic(&value, "file:///tmp/x.rs", PositionEncoding::Utf16, lines_from(&[("file:///tmp/x.rs", "x")])).unwrap();
        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.code.as_deref(), Some("308"));
    }

    #[test]
    fn diagnostics_store_replaces_rather_than_appends() {
        let mut store = DiagnosticsStore::default();
        let uri = "file:///tmp/x.rs";
        let notif = |msgs: &[&str]| {
            json!({
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": uri,
                    "diagnostics": msgs.iter().map(|m| json!({
                        "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } },
                        "message": m,
                    })).collect::<Vec<_>>(),
                },
            })
        };

        store.ingest_notification(&notif(&["first"]));
        assert_eq!(store.raw_for(uri).len(), 1);

        // A second push for the same URI REPLACES — this is the semantics that,
        // done wrong, doubles diagnostics on every keystroke.
        store.ingest_notification(&notif(&["second", "third"]));
        assert_eq!(store.raw_for(uri).len(), 2, "publishDiagnostics replaces; it must not accumulate");

        // An empty push means "now clean".
        store.ingest_notification(&notif(&[]));
        assert!(store.raw_for(uri).is_empty(), "an empty publish clears the file's diagnostics");
    }

    #[test]
    fn diagnostics_store_ignores_a_notification_without_a_uri() {
        let mut store = DiagnosticsStore::default();
        store.ingest_notification(&json!({ "method": "textDocument/publishDiagnostics", "params": { "diagnostics": [] } }));
        assert!(store.raw_for("file:///tmp/x.rs").is_empty());
    }
}
