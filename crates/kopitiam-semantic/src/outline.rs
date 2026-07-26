//! Outline / skeleton mode — token-max Task II-2.
//!
//! [`outline`] turns a file into an *items-only* skeleton: one line per
//! declaration (module, struct, field, `impl` block, fn/method, const, type
//! alias, …) with the line it begins on and its signature/type — and **no
//! bodies**. An 811-line file whose outline is ~60 lines lets an agent orient
//! for roughly a tenth of the tokens a full read costs, then (with Task II-1)
//! read only the one function it actually needs.
//!
//! # Where the tree comes from
//!
//! [`RustAnalyzerSession::document_symbols`] runs a `textDocument/documentSymbol`
//! request, which returns rust-analyzer's **hierarchical** `DocumentSymbol`
//! tree — items and their containment, never the statements inside a body. So
//! "no bodies" is a property of the *source* here, not something this module
//! strips: the server simply never reports body contents as symbols.
//!
//! # Coordinates, not content (§0.1)
//!
//! Every item carries a 0-based `line`, matching the `file:line:char`, 0-based
//! convention the rest of this crate emits (see
//! [`RustAnalyzerSession::rename`]) — never a byte range or a copy of the
//! source. Line indices are encoding-independent, so unlike the
//! position-bearing results in [`crate::lsp_types`] the outline needs no
//! wire→`char` conversion at all.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::session::RustAnalyzerSession;

/// One item in a file's outline: a single declaration, positioned by the
/// 0-based `line` it begins on and nested under its container by `depth`
/// (a top-level item is depth 0; a method inside an `impl` is depth 1).
///
/// `kind` is a short, stable label derived from the LSP `SymbolKind`
/// (`"fn"`, `"struct"`, `"field"`, `"impl"`, `"const"`, …); `detail` is
/// rust-analyzer's own signature/type string for the item when it supplied one
/// (a fn's parameter list, a field's type), or `None`. Both are carried
/// verbatim — this type never invents text the server did not give it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutlineItem {
    /// 0-based line the declaration begins on.
    pub line: u32,
    /// Short kind label (`"fn"`, `"struct"`, `"field"`, `"impl"`, …).
    pub kind: String,
    /// The item's name (for an `impl` block, rust-analyzer's own
    /// `"impl Foo"` / `"impl Trait for Foo"` string).
    pub name: String,
    /// The server's signature/type detail, when present.
    pub detail: Option<String>,
    /// Nesting depth under containers (`impl`/`mod`); top level is 0.
    pub depth: usize,
}

/// A file's complete outline: its [`OutlineItem`]s in source order (sorted by
/// `line`). Serializes directly to the `--json` form the CLI emits (§0.2);
/// [`Self::to_skeleton`] renders the human, line-numbered skeleton.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Outline {
    pub items: Vec<OutlineItem>,
}

impl Outline {
    /// Renders the deterministic, line-numbered skeleton: one line per item,
    /// `LINE: <indent><kind> <name>[ <detail>]`, indented two spaces per depth
    /// level, ordered by (0-based) line. No bodies, no blank lines — just the
    /// map. When an item's `name` already begins with its `kind` (an `impl`
    /// block's name is `"impl Foo"`), the kind is not repeated.
    pub fn to_skeleton(&self) -> String {
        let mut out = String::new();
        for item in &self.items {
            let indent = "  ".repeat(item.depth);
            // Avoid "impl impl Foo": if the name already leads with the kind
            // word, the name alone is the head.
            let rest = item.name.strip_prefix(item.kind.as_str());
            let name_leads_with_kind = item.name == item.kind || matches!(rest, Some(r) if r.starts_with(' '));
            let head = if name_leads_with_kind {
                item.name.clone()
            } else {
                format!("{} {}", item.kind, item.name)
            };
            let detail = match &item.detail {
                Some(d) => format!(" {d}"),
                None => String::new(),
            };
            out.push_str(&format!("{}: {indent}{head}{detail}\n", item.line));
        }
        out
    }
}

/// Builds the [`Outline`] of `file` using `session`'s live rust-analyzer.
///
/// This requires a connected [`RustAnalyzerSession`] because `documentSymbol`
/// is a server request — but the token win is *structural*, not temporal: the
/// result replaces a multi-thousand-token file read with a ~60-line skeleton
/// regardless of wall-clock (§0.1). The tree is parsed by
/// [`parse_document_symbols`], which is the deterministically testable half.
pub fn outline(session: &mut RustAnalyzerSession, file: &Path) -> Result<Outline> {
    let symbols = session.document_symbols(file)?;
    Ok(Outline { items: parse_document_symbols(&symbols) })
}

/// Flattens an LSP `documentSymbol` response into ordered [`OutlineItem`]s.
///
/// Handles both reply shapes (see [`crate::lsp_client`]'s `document_symbols`):
/// the hierarchical `DocumentSymbol` tree (rust-analyzer's choice — nesting is
/// preserved as `depth`) and the legacy flat `SymbolInformation[]` (every item
/// at depth 0, since that shape discards containment). The result is
/// stable-sorted by line, so it reads top-to-bottom like the file itself.
pub(crate) fn parse_document_symbols(symbols: &[Value]) -> Vec<OutlineItem> {
    let mut items = Vec::new();
    for symbol in symbols {
        collect_symbol(symbol, 0, &mut items);
    }
    // Stable sort: a child's line always exceeds its parent's, so ordering by
    // line reproduces source order while keeping any same-line items in
    // traversal order.
    items.sort_by_key(|item| item.line);
    items
}

/// Appends `symbol` (and, for a hierarchical node, its descendants at
/// `depth + 1`) to `out` in preorder.
fn collect_symbol(symbol: &Value, depth: usize, out: &mut Vec<OutlineItem>) {
    let line = symbol_start_line(symbol);
    let name = symbol.get("name").and_then(Value::as_str).unwrap_or("<anonymous>").to_string();
    let kind = symbol_kind_label(symbol.get("kind").and_then(Value::as_i64).unwrap_or(0)).to_string();
    let detail = symbol
        .get("detail")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(str::to_string);
    out.push(OutlineItem { line, kind, name, detail, depth });

    // Hierarchical `DocumentSymbol` only; the flat shape has no `children`.
    if let Some(children) = symbol.get("children").and_then(Value::as_array) {
        for child in children {
            collect_symbol(child, depth + 1, out);
        }
    }
}

/// The 0-based line a symbol begins on: a hierarchical node's own `range`
/// (whole item) or `selectionRange` (its name span — same starting line), or a
/// flat `SymbolInformation`'s `location.range`. Defaults to 0 if absent.
fn symbol_start_line(symbol: &Value) -> u32 {
    symbol
        .get("range")
        .or_else(|| symbol.get("selectionRange"))
        .or_else(|| symbol.pointer("/location/range"))
        .and_then(|r| r.pointer("/start/line"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32
}

/// Maps an LSP `SymbolKind` integer to a short, stable label. rust-analyzer's
/// own mappings are noted where they diverge from the raw LSP name: it reports
/// a `trait` as `Interface`, an `impl` block as `Object`, and a type alias as
/// `TypeParameter`. Unknown/future kinds fall back to `"item"`.
fn symbol_kind_label(kind: i64) -> &'static str {
    match kind {
        2 => "mod",       // Module
        3 => "namespace", // Namespace
        5 => "class",     // Class
        6 => "fn",        // Method
        7 => "property",  // Property
        8 => "field",     // Field
        10 => "enum",     // Enum
        11 => "trait",    // Interface     (rust-analyzer: trait / trait alias)
        12 => "fn",       // Function      (rust-analyzer: fn / macro)
        13 => "let",      // Variable
        14 => "const",    // Constant      (rust-analyzer: const / static)
        19 => "impl",     // Object        (rust-analyzer: impl block)
        22 => "variant",  // EnumMember
        23 => "struct",   // Struct        (rust-analyzer: struct / union)
        26 => "type",     // TypeParameter (rust-analyzer: type alias / type param)
        _ => "item",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A hand-built hierarchical `DocumentSymbol` tree standing in for
    /// rust-analyzer's reply — deliberately out of line order (the `const` is
    /// declared first but listed last) to prove the sort. This is the whole
    /// point of the split: the formatter contract is proven here with zero I/O
    /// and no spawned server.
    fn doc_symbol_tree() -> Vec<Value> {
        vec![
            json!({
                "name": "Rebuilder",
                "kind": 23,
                "range": { "start": { "line": 3, "character": 0 }, "end": { "line": 6, "character": 1 } },
                "selectionRange": { "start": { "line": 3, "character": 11 }, "end": { "line": 3, "character": 20 } },
                "children": [
                    {
                        "name": "spans", "detail": "Vec<Span>", "kind": 8,
                        "range": { "start": { "line": 4, "character": 4 }, "end": { "line": 4, "character": 20 } },
                        "selectionRange": { "start": { "line": 4, "character": 4 }, "end": { "line": 4, "character": 9 } }
                    },
                    {
                        "name": "config", "detail": "Config", "kind": 8,
                        "range": { "start": { "line": 5, "character": 4 }, "end": { "line": 5, "character": 18 } },
                        "selectionRange": { "start": { "line": 5, "character": 4 }, "end": { "line": 5, "character": 10 } }
                    }
                ]
            }),
            json!({
                "name": "impl Rebuilder",
                "kind": 19,
                "range": { "start": { "line": 8, "character": 0 }, "end": { "line": 20, "character": 1 } },
                "selectionRange": { "start": { "line": 8, "character": 5 }, "end": { "line": 8, "character": 14 } },
                "children": [
                    {
                        "name": "new", "detail": "(config: Config) -> Self", "kind": 6,
                        "range": { "start": { "line": 9, "character": 4 }, "end": { "line": 11, "character": 5 } },
                        "selectionRange": { "start": { "line": 9, "character": 11 }, "end": { "line": 9, "character": 14 } }
                    },
                    {
                        "name": "rebuild", "detail": "(&self) -> Document", "kind": 6,
                        "range": { "start": { "line": 13, "character": 4 }, "end": { "line": 19, "character": 5 } },
                        "selectionRange": { "start": { "line": 13, "character": 11 }, "end": { "line": 13, "character": 18 } }
                    }
                ]
            }),
            json!({
                "name": "MAX_DEPTH", "detail": "usize", "kind": 14,
                "range": { "start": { "line": 1, "character": 0 }, "end": { "line": 1, "character": 30 } },
                "selectionRange": { "start": { "line": 1, "character": 6 }, "end": { "line": 1, "character": 15 } }
            }),
            json!({
                "name": "helper", "detail": "(x: u32) -> u32", "kind": 12,
                "range": { "start": { "line": 22, "character": 0 }, "end": { "line": 24, "character": 1 } },
                "selectionRange": { "start": { "line": 22, "character": 7 }, "end": { "line": 22, "character": 13 } }
            }),
        ]
    }

    #[test]
    fn parses_a_hierarchical_tree_into_ordered_depth_marked_items() {
        let items = parse_document_symbols(&doc_symbol_tree());
        let expected = vec![
            OutlineItem { line: 1, kind: "const".into(), name: "MAX_DEPTH".into(), detail: Some("usize".into()), depth: 0 },
            OutlineItem { line: 3, kind: "struct".into(), name: "Rebuilder".into(), detail: None, depth: 0 },
            OutlineItem { line: 4, kind: "field".into(), name: "spans".into(), detail: Some("Vec<Span>".into()), depth: 1 },
            OutlineItem { line: 5, kind: "field".into(), name: "config".into(), detail: Some("Config".into()), depth: 1 },
            OutlineItem { line: 8, kind: "impl".into(), name: "impl Rebuilder".into(), detail: None, depth: 0 },
            OutlineItem { line: 9, kind: "fn".into(), name: "new".into(), detail: Some("(config: Config) -> Self".into()), depth: 1 },
            OutlineItem { line: 13, kind: "fn".into(), name: "rebuild".into(), detail: Some("(&self) -> Document".into()), depth: 1 },
            OutlineItem { line: 22, kind: "fn".into(), name: "helper".into(), detail: Some("(x: u32) -> u32".into()), depth: 0 },
        ];
        assert_eq!(items, expected);
    }

    #[test]
    fn renders_a_line_numbered_skeleton_with_no_bodies() {
        let outline = Outline { items: parse_document_symbols(&doc_symbol_tree()) };
        // Items only, one per line, 0-based line numbers, nested by depth, the
        // `impl` kind not doubled, bodies entirely absent.
        let expected = "\
1: const MAX_DEPTH usize
3: struct Rebuilder
4:   field spans Vec<Span>
5:   field config Config
8: impl Rebuilder
9:   fn new (config: Config) -> Self
13:   fn rebuild (&self) -> Document
22: fn helper (x: u32) -> u32
";
        assert_eq!(outline.to_skeleton(), expected);
    }

    #[test]
    fn parses_the_flat_symbol_information_fallback_at_depth_zero() {
        // A server that only speaks the legacy `SymbolInformation[]` shape:
        // no `children`, positions under `location.range`. Nesting is lost
        // (everything is depth 0), but the items and lines survive.
        let flat = vec![
            json!({
                "name": "greet", "kind": 12,
                "location": { "uri": "file:///x.rs", "range": { "start": { "line": 5, "character": 0 }, "end": { "line": 7, "character": 1 } } }
            }),
            json!({
                "name": "Widget", "kind": 23, "containerName": "",
                "location": { "uri": "file:///x.rs", "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 3, "character": 1 } } }
            }),
        ];
        let items = parse_document_symbols(&flat);
        assert_eq!(items, vec![
            OutlineItem { line: 0, kind: "struct".into(), name: "Widget".into(), detail: None, depth: 0 },
            OutlineItem { line: 5, kind: "fn".into(), name: "greet".into(), detail: None, depth: 0 },
        ]);
    }

    #[test]
    fn empty_input_is_an_empty_outline() {
        assert_eq!(parse_document_symbols(&[]), Vec::<OutlineItem>::new());
        assert_eq!(Outline { items: Vec::new() }.to_skeleton(), "");
    }

    #[test]
    fn maps_lsp_symbol_kinds_to_rust_labels() {
        assert_eq!(symbol_kind_label(12), "fn"); // Function
        assert_eq!(symbol_kind_label(6), "fn"); // Method
        assert_eq!(symbol_kind_label(23), "struct");
        assert_eq!(symbol_kind_label(11), "trait"); // Interface
        assert_eq!(symbol_kind_label(19), "impl"); // Object
        assert_eq!(symbol_kind_label(14), "const"); // Constant
        assert_eq!(symbol_kind_label(26), "type"); // TypeParameter
        assert_eq!(symbol_kind_label(999), "item"); // unknown/future
    }

    /// End-to-end proof that `outline` produces a body-free skeleton through a
    /// live `rust-analyzer`. `#[ignore]`d — it spawns a real server and waits
    /// for indexing (~minutes), so it is never part of the offline suite. Run
    /// deliberately with:
    ///
    /// ```text
    /// cargo test --release -p kopitiam-semantic -- --ignored live_rust_analyzer_outlines_a_file
    /// ```
    #[test]
    #[ignore = "spawns a real rust-analyzer and waits for indexing; run with `-- --ignored`"]
    fn live_rust_analyzer_outlines_a_file() {
        if !crate::lsp_client::binary_available("rust-analyzer") {
            eprintln!("rust-analyzer not on PATH; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"sem_outline_test\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let lib = dir.path().join("src/lib.rs");
        let source = "pub const MAX: usize = 4;\n\npub struct Widget {\n    pub id: u32,\n}\n\nimpl Widget {\n    pub fn new(id: u32) -> Self {\n        Widget { id }\n    }\n}\n";
        std::fs::write(&lib, source).unwrap();

        let mut session = RustAnalyzerSession::connect(dir.path()).expect("rust-analyzer should start and index");
        let result = outline(&mut session, &lib).expect("documentSymbol should succeed");

        assert!(!result.items.is_empty(), "a non-empty file must produce outline items");
        assert!(result.items.iter().any(|i| i.name == "Widget" && i.kind == "struct"), "the struct must appear");
        assert!(result.items.iter().any(|i| i.name == "new" && i.kind == "fn"), "the method `new` must appear as a fn");
        // The whole point: the body statement must never reach the skeleton.
        assert!(!result.to_skeleton().contains("Widget { id }"), "outline must omit bodies");

        session.shutdown().ok();
    }
}
