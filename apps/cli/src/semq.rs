//! Semantic query subcommands — token-max Task II-1 (the highest-value card).
//!
//! These expose rust-analyzer's knowledge as **coordinate-returning** commands
//! so an agent gets ~40 tokens of `file:line:character` instead of grepping,
//! reading the hits with context, then opening files (`kopitiam_token_max.md`
//! §11 II-1, §0.1 "coordinates, not content"):
//!
//! * `refs   <symbol>` — every reference/call site, as coordinates, no context.
//! * `def    <symbol>` — where it is defined, plus its signature.
//! * `sig    <symbol>` — the signature alone.
//! * `callers <fn> [--depth N]` — who calls it (and who calls them, to depth N).
//! * `callees <fn> [--depth N]` — what it calls.
//! * `impls  <trait>` — the trait's `impl` sites.
//!
//! # How the primitives compose
//!
//! [`RustAnalyzerSession`] already exposes `references`, `definition`, `hover`
//! and `document_symbols` (all `char`-offset, 0-based — §10.1 finding 4). This
//! module builds the rest *on top of those* rather than adding new LSP methods:
//!
//! * `refs`/`def`/`sig` are thin wrappers over `references`/`document_symbols`/
//!   `hover`.
//! * `callers` = `references` to the function (its call sites), with each site's
//!   **enclosing function** resolved via `document_symbols` on the referencing
//!   file — recursed to `--depth` by then asking who calls *that* function.
//! * `callees` = the function's own body (its `document_symbols` range) scanned
//!   for call expressions, each resolved with `definition`.
//! * `impls` = `references` to the trait, filtered to the sites whose source
//!   line is an `impl` header.
//!
//! # Resolving a name to a position
//!
//! Every command names a *symbol*, but the session's queries take a
//! `file:line:character`. This module bridges that with `document_symbols`: the
//! required `--file` is scanned for a declaration whose name matches, and its
//! `selectionRange` (the identifier span) gives the precise position to query
//! from. (A workspace-wide name search would need `workspace/symbol` exposed on
//! the session; that lives in `kopitiam-semantic`, out of this wave's scope, so
//! the file is named explicitly — which also keeps the whole query inside a
//! *single* rust-analyzer session rather than paying the ~180 s index twice.)
//!
//! The pure pieces — name→position resolution, enclosing-symbol lookup,
//! signature extraction, call-site scanning, impl-header detection, and the
//! `--json` shaping — are all unit-tested here without a live server; the
//! orchestration that drives a real rust-analyzer is `#[ignore]`d.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::Args;
use kopitiam_semantic::{Location, RustAnalyzerSession};
use serde::Serialize;
use serde_json::{Value, json};

use crate::syntactic;

/// Shared arguments for the name-based queries (`refs`, `def`, `sig`, `impls`).
#[derive(Args, Debug)]
pub struct Locator {
    /// The symbol (or, for `impls`, trait) name to resolve.
    pub symbol: String,

    /// The file whose `documentSymbol` declares the symbol; its identifier
    /// position is resolved there and used as the query anchor.
    #[arg(long)]
    pub file: PathBuf,

    /// Directory containing the workspace `Cargo.toml`. Defaults to the current
    /// directory; passed to rust-analyzer as the root.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Emit JSON coordinates instead of the human `file:line:character` form.
    #[arg(long)]
    pub json: bool,

    /// Skip rust-analyzer and answer from a workspace-wide textual search
    /// (`refs` only) — grep-grade, not semantic, so each hit is labelled
    /// `(syntactic, not semantic — verify)`. The guaranteed path when
    /// rust-analyzer cannot index the workspace in a bounded time.
    #[arg(long = "no-lsp", visible_alias = "syntactic")]
    pub no_lsp: bool,

    /// Require rust-analyzer: on timeout/unavailability fail (non-zero exit)
    /// instead of falling back.
    #[arg(long, conflicts_with = "no_lsp")]
    pub lsp: bool,
}

/// Arguments for the call-hierarchy queries (`callers`, `callees`), which add a
/// recursion `--depth` to [`Locator`].
#[derive(Args, Debug)]
pub struct DepthArgs {
    #[command(flatten)]
    pub locator: Locator,

    /// How many hops of the call graph to follow (1 = direct callers/callees).
    #[arg(long, default_value_t = 1)]
    pub depth: usize,
}

// ---------------------------------------------------------------------------
// Public entry points (one per Command variant).
// ---------------------------------------------------------------------------

/// `refs <symbol>` — every reference site as coordinates.
///
/// Default behaviour asks rust-analyzer (semantic, cross-file references) and,
/// on timeout/unavailability, falls back automatically to a workspace-wide
/// textual search ([`syntactic::grep_symbol`]) whose hits are labelled
/// `(syntactic, not semantic — verify)`. `--no-lsp`/`--syntactic` forces the
/// textual search; `--lsp` forces rust-analyzer and makes a failure a hard,
/// non-zero exit (fix #1).
pub fn run_refs(args: Locator) -> Result<()> {
    if args.no_lsp {
        return refs_syntactic(&args);
    }
    // P1: on a large workspace, DEFAULT to the instant textual search rather
    // than waiting out a rust-analyzer index that will not finish (`--lsp` skips
    // this and forces rust-analyzer, hard-failing on timeout).
    if !args.lsp && syntactic::prefer_syntactic_default(&args.root) {
        eprintln!("large workspace: syntactic refs; --lsp for rust-analyzer");
        return refs_syntactic(&args);
    }
    syntactic::with_fallback(
        args.lsp,
        || refs_via_ra(&args),
        "rust-analyzer unavailable/timed out",
        || refs_syntactic(&args),
    )
}

/// The semantic `refs`: resolve the symbol in `--file`, ask rust-analyzer for
/// every reference, and print coordinates. Errors (connect/index timeout, or an
/// unresolved symbol) bubble up so the caller can fall back or exit non-zero.
fn refs_via_ra(args: &Locator) -> Result<()> {
    let mut session = connect(&args.root)?;
    let pos = resolve(&mut session, &args.file, &args.symbol)?;
    let locations = session.references(&args.file, pos.line, pos.character, false)?;
    let _ = session.shutdown();

    let coords = sorted_coords(&locations);
    if args.json {
        emit_json(&RefsJson { symbol: &args.symbol, references: &coords });
    } else if coords.is_empty() {
        println!("no references to `{}`", args.symbol);
    } else {
        for c in &coords {
            println!("{}", c.display());
        }
    }
    Ok(())
}

/// The syntactic `refs` fallback: a whole-word textual search for the symbol
/// across the workspace's `.rs` files. Hits are grep-grade — a mention in a
/// comment or an unrelated same-named symbol counts — so every line carries the
/// honesty label, and `--json` marks the whole result `"syntactic": true`.
fn refs_syntactic(args: &Locator) -> Result<()> {
    const LABEL: &str = "(syntactic, not semantic — verify)";
    let hits = syntactic::grep_symbol(&args.root, &args.symbol)?;

    if args.json {
        let references: Vec<Value> = hits
            .iter()
            .map(|h| json!({ "file": h.file, "line": h.line, "character": h.character }))
            .collect();
        emit_json(&json!({
            "symbol": args.symbol,
            "syntactic": true,
            "note": "textual whole-word matches — not rust-analyzer references; verify each",
            "references": references,
        }));
    } else if hits.is_empty() {
        println!("no textual matches for `{}` {LABEL}", args.symbol);
    } else {
        for h in &hits {
            println!("{}:{}:{}  {LABEL}", h.file, h.line, h.character);
        }
    }
    Ok(())
}

/// `def <symbol>` — definition location plus signature.
pub fn run_def(args: Locator) -> Result<()> {
    reject_no_lsp(&args, "def")?;
    let mut session = connect(&args.root)?;
    let pos = resolve(&mut session, &args.file, &args.symbol)?;
    // The declaration we matched in `--file` *is* the definition location; its
    // signature comes from hover at that identifier.
    let hover = session.hover(&args.file, pos.line, pos.character)?;
    let _ = session.shutdown();

    let signature = hover.map(|h| extract_signature(&h.contents));
    let coord = Coord::at(&args.file, pos.line, pos.character);
    if args.json {
        emit_json(&DefJson {
            symbol: &args.symbol,
            signature: signature.as_deref(),
            definition: &coord,
        });
    } else {
        if let Some(sig) = &signature {
            println!("{sig}");
        }
        println!("defined at {}", coord.display());
    }
    Ok(())
}

/// `sig <symbol>` — the signature alone.
pub fn run_sig(args: Locator) -> Result<()> {
    reject_no_lsp(&args, "sig")?;
    let mut session = connect(&args.root)?;
    let pos = resolve(&mut session, &args.file, &args.symbol)?;
    let hover = session.hover(&args.file, pos.line, pos.character)?;
    let _ = session.shutdown();

    let signature = hover.map(|h| extract_signature(&h.contents));
    if args.json {
        emit_json(&SigJson { symbol: &args.symbol, signature: signature.as_deref() });
    } else {
        match signature {
            Some(sig) => println!("{sig}"),
            None => println!("no signature for `{}`", args.symbol),
        }
    }
    Ok(())
}

/// `callers <fn> [--depth N]` — call sites, with each site's enclosing
/// function, recursed to `depth`.
pub fn run_callers(args: DepthArgs) -> Result<()> {
    let loc = args.locator;
    reject_no_lsp(&loc, "callers")?;
    let mut session = connect(&loc.root)?;
    let pos = resolve(&mut session, &loc.file, &loc.symbol)?;

    let edges = {
        let mut walk = CallerWalk { session: &mut session, visited: HashSet::new(), edges: Vec::new() };
        walk.walk(&loc.file, pos.line, pos.character, args.depth, 1)?;
        walk.edges
    };
    let _ = session.shutdown();

    if loc.json {
        emit_json(&CallersJson { function: &loc.symbol, depth: args.depth, callers: &edges });
    } else if edges.is_empty() {
        println!("no callers of `{}`", loc.symbol);
    } else {
        for e in &edges {
            let indent = "  ".repeat(e.hop.saturating_sub(1));
            match &e.in_function {
                Some(f) => println!("{indent}{} (in {f})", e.site.display()),
                None => println!("{indent}{}", e.site.display()),
            }
        }
    }
    Ok(())
}

/// `callees <fn> [--depth N]` — the functions this one calls.
pub fn run_callees(args: DepthArgs) -> Result<()> {
    let loc = args.locator;
    reject_no_lsp(&loc, "callees")?;
    let mut session = connect(&loc.root)?;
    let pos = resolve(&mut session, &loc.file, &loc.symbol)?;

    let mut edges = Vec::new();
    let mut visited = HashSet::new();
    collect_callees(&mut session, &loc.file, &pos, args.depth, 1, &mut visited, &mut edges)?;
    let _ = session.shutdown();

    if loc.json {
        emit_json(&CalleesJson { function: &loc.symbol, depth: args.depth, callees: &edges });
    } else if edges.is_empty() {
        println!("no resolved callees of `{}`", loc.symbol);
    } else {
        for e in &edges {
            let indent = "  ".repeat(e.hop.saturating_sub(1));
            println!("{indent}{} {}", e.name, e.site.display());
        }
    }
    Ok(())
}

/// `impls <trait>` — the trait's `impl` sites (references whose source line is
/// an `impl` header).
pub fn run_impls(args: Locator) -> Result<()> {
    reject_no_lsp(&args, "impls")?;
    let mut session = connect(&args.root)?;
    let pos = resolve(&mut session, &args.file, &args.symbol)?;
    let references = session.references(&args.file, pos.line, pos.character, false)?;

    // Keep only reference sites whose own source line begins an `impl` block.
    let mut impls: Vec<ImplSite> = Vec::new();
    for loc in &references {
        if let Some(header) = impl_header_at(&loc.path, loc.range.start.line) {
            impls.push(ImplSite { site: Coord::from(loc), header });
        }
    }
    let _ = session.shutdown();
    impls.sort_by(|a, b| (&a.site.file, a.site.line).cmp(&(&b.site.file, b.site.line)));

    if args.json {
        emit_json(&ImplsJson { trait_name: &args.symbol, impls: &impls });
    } else if impls.is_empty() {
        println!("no impls of `{}` found from `{}`", args.symbol, args.file.display());
    } else {
        for i in &impls {
            println!("{}: {}", i.site.display(), i.header);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Live orchestration (the parts that need a running rust-analyzer).
// ---------------------------------------------------------------------------

/// Spawns rust-analyzer for `root`, announcing progress on stderr so `--json`
/// stdout stays clean. Delegates to [`syntactic::connect_ra`], which honours the
/// `KOPITIAM_RA_TIMEOUT_SECS` index-timeout override and emits a "still
/// indexing…" note instead of a silent hang (fix #3).
fn connect(root: &Path) -> Result<RustAnalyzerSession> {
    syntactic::connect_ra(root)
}

/// The name-based queries other than `refs` have no textual fallback (a
/// coordinate/def/caller answer needs real resolution). If `--no-lsp` is asked
/// of one of them, fail clearly rather than silently ignoring the flag.
fn reject_no_lsp(args: &Locator, command: &str) -> Result<()> {
    if args.no_lsp {
        // A UsageError, not a `bail!`: this is the caller naming a flag that does
        // not apply, so it deserves one clean line and exit 2 — never the stack
        // trace a plain anyhow error prints under RUST_BACKTRACE.
        return Err(crate::usage(format!(
            "`{command}` has no syntactic fallback — only `refs` and `outline` do. \
             Re-run without --no-lsp/--syntactic (rust-analyzer is required here)."
        )));
    }
    Ok(())
}

/// Resolves `name` to its identifier position in `file` via `document_symbols`.
fn resolve(session: &mut RustAnalyzerSession, file: &Path, name: &str) -> Result<SymPos> {
    let symbols = session.document_symbols(file)?;
    match find_symbol(&symbols, name) {
        Some(pos) => Ok(pos),
        None => bail!(
            "no symbol named `{name}` found in {} (its documentSymbol tree lists no matching declaration)",
            file.display()
        ),
    }
}

/// Depth-first collection of callers, holding the session plus the accumulating
/// `visited`/`edges` state so the recursion stays a small method rather than an
/// eight-argument free function.
struct CallerWalk<'s> {
    session: &'s mut RustAnalyzerSession,
    visited: HashSet<(PathBuf, u32)>,
    edges: Vec<CallerEdge>,
}

impl CallerWalk<'_> {
    /// The references to the symbol at `file:line:character` are its call sites;
    /// each site's enclosing function is resolved and, when `depth` allows,
    /// recursed into.
    fn walk(&mut self, file: &Path, line: u32, character: u32, depth: usize, hop: usize) -> Result<()> {
        if depth == 0 {
            return Ok(());
        }
        let references = self.session.references(file, line, character, false)?;
        for loc in references {
            let key = (loc.path.clone(), loc.range.start.line);
            if !self.visited.insert(key) {
                continue;
            }
            let syms = self.session.document_symbols(&loc.path)?;
            let enclosing = enclosing_symbol(&syms, loc.range.start.line);
            self.edges.push(CallerEdge {
                site: Coord::from(&loc),
                in_function: enclosing.as_ref().map(|s| s.name.clone()),
                hop,
            });
            if depth > 1
                && let Some(encl) = enclosing
            {
                self.walk(&loc.path, encl.line, encl.character, depth - 1, hop + 1)?;
            }
        }
        Ok(())
    }
}

/// Depth-first collection of callees: scan the function's body (its
/// `document_symbols` range) for call expressions and resolve each with
/// `definition`.
fn collect_callees(
    session: &mut RustAnalyzerSession,
    file: &Path,
    func: &SymPos,
    depth: usize,
    hop: usize,
    visited: &mut HashSet<(PathBuf, u32)>,
    out: &mut Vec<CalleeEdge>,
) -> Result<()> {
    if depth == 0 {
        return Ok(());
    }
    let text = std::fs::read_to_string(file)?;
    let candidates = call_candidates_in_range(&text, func.range_start_line, func.range_end_line);
    for cand in candidates {
        let defs = session.definition(file, cand.line, cand.character)?;
        for def in defs {
            let key = (def.path.clone(), def.range.start.line);
            if !visited.insert(key) {
                continue;
            }
            out.push(CalleeEdge { name: cand.name.clone(), site: Coord::from(&def), hop });
            if depth > 1 {
                // Recurse into the callee's own file/body. Resolve its symbols
                // and find the one whose selection matches the definition site.
                if let Ok(syms) = session.document_symbols(&def.path)
                    && let Some(callee) = symbol_at(&syms, def.range.start.line)
                {
                    collect_callees(session, &def.path, &callee, depth - 1, hop + 1, visited, out)?;
                }
            }
        }
    }
    Ok(())
}

/// Reads `file` and returns the `impl`-header text at `line` (0-based), if that
/// line begins an `impl` block. Used to filter trait references down to impl
/// sites.
fn impl_header_at(file: &Path, line: u32) -> Option<String> {
    let text = std::fs::read_to_string(file).ok()?;
    let raw = text.lines().nth(line as usize)?;
    line_is_impl_header(raw).then(|| raw.trim().to_string())
}

// ---------------------------------------------------------------------------
// Pure, unit-tested helpers.
// ---------------------------------------------------------------------------

/// A symbol resolved from a `documentSymbol` tree: the identifier position to
/// query from (its `selectionRange` start) plus the whole-item line range (its
/// `range`) for body scanning.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SymPos {
    line: u32,
    character: u32,
    name: String,
    kind: String,
    range_start_line: u32,
    range_end_line: u32,
}

/// A call-site candidate found by scanning source: the identifier being called
/// and its 0-based `file` position.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CallCandidate {
    name: String,
    line: u32,
    character: u32,
}

/// Finds the first symbol (preorder) whose name matches `name` — either exactly
/// or, for a qualified `Type::method`, by the final segment — and returns its
/// position. Recurses into `children` (the hierarchical `DocumentSymbol` shape).
fn find_symbol(symbols: &[Value], name: &str) -> Option<SymPos> {
    let target = name.rsplit("::").next().unwrap_or(name);
    for symbol in symbols {
        let this = symbol.get("name").and_then(Value::as_str).unwrap_or_default();
        if this == name || this == target {
            return Some(sym_pos(symbol));
        }
        if let Some(children) = symbol.get("children").and_then(Value::as_array)
            && let Some(found) = find_symbol(children, name)
        {
            return Some(found);
        }
    }
    None
}

/// Finds the innermost symbol whose whole-item `range` contains `line` (0-based)
/// — the declaration a reference on that line sits inside, i.e. the *calling*
/// function. "Innermost" = the smallest containing range, so a method wins over
/// its `impl` block.
fn enclosing_symbol(symbols: &[Value], line: u32) -> Option<SymPos> {
    let mut best: Option<SymPos> = None;
    walk_containing(symbols, line, &mut best);
    best
}

/// Like [`enclosing_symbol`] but keyed on a symbol whose `selectionRange`
/// begins exactly at `line` — used to re-find a callee's declaration from the
/// `definition` site so recursion can descend into its body.
fn symbol_at(symbols: &[Value], sel_start_line: u32) -> Option<SymPos> {
    for symbol in symbols {
        let pos = sym_pos(symbol);
        if pos.line == sel_start_line {
            return Some(pos);
        }
        if let Some(children) = symbol.get("children").and_then(Value::as_array)
            && let Some(found) = symbol_at(children, sel_start_line)
        {
            return Some(found);
        }
    }
    None
}

fn walk_containing(symbols: &[Value], line: u32, best: &mut Option<SymPos>) {
    for symbol in symbols {
        let pos = sym_pos(symbol);
        if pos.range_start_line <= line && line <= pos.range_end_line {
            let span = pos.range_end_line - pos.range_start_line;
            let better = match best {
                Some(b) => span <= (b.range_end_line - b.range_start_line),
                None => true,
            };
            if better {
                *best = Some(pos.clone());
            }
            if let Some(children) = symbol.get("children").and_then(Value::as_array) {
                walk_containing(children, line, best);
            }
        }
    }
}

/// Builds a [`SymPos`] from one `DocumentSymbol` JSON object: `selectionRange`
/// gives the identifier position, `range` the whole-item span.
fn sym_pos(symbol: &Value) -> SymPos {
    let sel = symbol
        .pointer("/selectionRange/start")
        .or_else(|| symbol.pointer("/range/start"))
        .or_else(|| symbol.pointer("/location/range/start"));
    let line = sel.and_then(|s| s.get("line")).and_then(Value::as_u64).unwrap_or(0) as u32;
    let character = sel.and_then(|s| s.get("character")).and_then(Value::as_u64).unwrap_or(0) as u32;
    let range = symbol.pointer("/range").or_else(|| symbol.pointer("/location/range"));
    let range_start_line = range
        .and_then(|r| r.pointer("/start/line"))
        .and_then(Value::as_u64)
        .unwrap_or(line as u64) as u32;
    let range_end_line = range
        .and_then(|r| r.pointer("/end/line"))
        .and_then(Value::as_u64)
        .unwrap_or(line as u64) as u32;
    SymPos {
        line,
        character,
        name: symbol.get("name").and_then(Value::as_str).unwrap_or("<anonymous>").to_string(),
        kind: kind_label(symbol.get("kind").and_then(Value::as_i64).unwrap_or(0)).to_string(),
        range_start_line,
        range_end_line,
    }
}

/// Maps an LSP `SymbolKind` integer to a short label (mirrors
/// `kopitiam_semantic::outline`'s private mapping; kept local since that map is
/// not exported).
fn kind_label(kind: i64) -> &'static str {
    match kind {
        2 => "mod",
        5 => "class",
        6 => "fn",
        8 => "field",
        10 => "enum",
        11 => "trait",
        12 => "fn",
        14 => "const",
        19 => "impl",
        22 => "variant",
        23 => "struct",
        26 => "type",
        _ => "item",
    }
}

/// Extracts a one-line signature from a rust-analyzer hover string. Hover comes
/// back as markdown: fenced ```rust blocks (a module-path line, then the
/// signature), a `---` rule, then docs. This returns the first fenced line that
/// looks like a declaration (rather than a bare module path), falling back to
/// the first non-empty, non-fence content line.
fn extract_signature(hover: &str) -> String {
    let mut first_content: Option<&str> = None;
    for raw in hover.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("```") || line == "---" {
            continue;
        }
        if first_content.is_none() {
            first_content = Some(line);
        }
        if looks_like_signature(line) {
            return line.to_string();
        }
    }
    first_content.unwrap_or("").to_string()
}

/// True for a hover line that reads as a declaration rather than a bare module
/// path (`crate::module`). A signature contains a declaration keyword or a
/// field's `name: Type`; a module path is a single `::`-joined token with no
/// spaces.
fn looks_like_signature(line: &str) -> bool {
    const KEYWORDS: [&str; 9] =
        ["fn ", "struct ", "enum ", "trait ", "const ", "static ", "type ", "impl ", "macro "];
    if KEYWORDS.iter().any(|k| line.contains(k)) {
        return true;
    }
    // A field-like `name: Type` (has a space somewhere, i.e. not a bare path).
    line.contains(": ") && line.contains(char::is_whitespace)
}

/// Detects a source line that opens an `impl` block — used to filter trait
/// references down to impl sites. Matches a leading `impl` word (after
/// indentation), including generic forms like `impl<T> Trait for Foo`.
fn line_is_impl_header(line: &str) -> bool {
    let t = line.trim_start();
    t == "impl" || t.starts_with("impl ") || t.starts_with("impl<")
}

/// Scans lines `[start_line, end_line]` (0-based, inclusive) of `text` for call
/// expressions — an identifier immediately followed (past whitespace) by `(` —
/// and returns each as a [`CallCandidate`] at its 0-based file position, deduped
/// by name (first occurrence wins). Rust keywords that take a paren (`if`,
/// `while`, `match`, …) are excluded so only genuine call targets remain.
fn call_candidates_in_range(text: &str, start_line: u32, end_line: u32) -> Vec<CallCandidate> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let ln = i as u32;
        if ln < start_line || ln > end_line {
            continue;
        }
        for cand in call_candidates_on_line(line, ln) {
            if seen.insert(cand.name.clone()) {
                out.push(cand);
            }
        }
    }
    out
}

/// Finds every `ident(`-style call on a single line, returning each with its
/// 0-based char column.
fn call_candidates_on_line(line: &str, line_no: u32) -> Vec<CallCandidate> {
    const NOT_CALLS: [&str; 12] = [
        "if", "while", "for", "match", "return", "fn", "let", "in", "as", "where", "move", "dyn",
    ];
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        // Start of an identifier?
        let is_ident_start = chars[i] == '_' || chars[i].is_alphabetic();
        let prev_is_ident = i > 0 && (chars[i - 1] == '_' || chars[i - 1].is_alphanumeric());
        if is_ident_start && !prev_is_ident {
            let start = i;
            while i < chars.len() && (chars[i] == '_' || chars[i].is_alphanumeric()) {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();
            // Skip whitespace, then require an opening paren for a call.
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '(' && !NOT_CALLS.contains(&name.as_str()) {
                out.push(CallCandidate { name, line: line_no, character: start as u32 });
            }
        } else {
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Output shaping.
// ---------------------------------------------------------------------------

/// A 0-based coordinate (`file:line:character`), the unit every semantic query
/// emits (§10.1 finding 4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Coord {
    file: String,
    line: u32,
    character: u32,
}

impl Coord {
    fn at(file: &Path, line: u32, character: u32) -> Self {
        Coord { file: file.display().to_string(), line, character }
    }
    fn from(loc: &Location) -> Self {
        Coord {
            file: loc.path.display().to_string(),
            line: loc.range.start.line,
            character: loc.range.start.character,
        }
    }
    fn display(&self) -> String {
        format!("{}:{}:{}", self.file, self.line, self.character)
    }
}

/// One caller edge: a call site and the function it sits inside.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CallerEdge {
    site: Coord,
    #[serde(skip_serializing_if = "Option::is_none")]
    in_function: Option<String>,
    hop: usize,
}

/// One callee edge: a called function's name and where it is defined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CalleeEdge {
    name: String,
    site: Coord,
    hop: usize,
}

/// One impl site: its coordinate and the `impl` header line's text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ImplSite {
    site: Coord,
    header: String,
}

#[derive(Serialize)]
struct RefsJson<'a> {
    symbol: &'a str,
    references: &'a [Coord],
}
#[derive(Serialize)]
struct DefJson<'a> {
    symbol: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<&'a str>,
    definition: &'a Coord,
}
#[derive(Serialize)]
struct SigJson<'a> {
    symbol: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<&'a str>,
}
#[derive(Serialize)]
struct CallersJson<'a> {
    function: &'a str,
    depth: usize,
    callers: &'a [CallerEdge],
}
#[derive(Serialize)]
struct CalleesJson<'a> {
    function: &'a str,
    depth: usize,
    callees: &'a [CalleeEdge],
}
#[derive(Serialize)]
struct ImplsJson<'a> {
    #[serde(rename = "trait")]
    trait_name: &'a str,
    impls: &'a [ImplSite],
}

/// Converts `Location`s to sorted, deduped [`Coord`]s (by file, then line, then
/// column) — the deterministic order `refs`/`impls` print.
fn sorted_coords(locations: &[Location]) -> Vec<Coord> {
    let mut coords: Vec<Coord> = locations.iter().map(Coord::from).collect();
    coords.sort_by(|a, b| (&a.file, a.line, a.character).cmp(&(&b.file, b.line, b.character)));
    coords.dedup();
    coords
}

fn emit_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialize JSON: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    #[derive(Parser)]
    struct LocatorCli {
        #[command(flatten)]
        args: Locator,
    }
    #[derive(Parser)]
    struct DepthCli {
        #[command(flatten)]
        args: DepthArgs,
    }

    // ---- arg parsing --------------------------------------------------------

    #[test]
    fn locator_parses_symbol_file_and_flags() {
        let cli = LocatorCli::try_parse_from(["t", "try_table", "--file", "src/x.rs", "--json"]).unwrap();
        assert_eq!(cli.args.symbol, "try_table");
        assert_eq!(cli.args.file, PathBuf::from("src/x.rs"));
        assert!(cli.args.json);
        assert_eq!(cli.args.root, PathBuf::from("."));
    }

    #[test]
    fn locator_requires_a_file() {
        assert!(LocatorCli::try_parse_from(["t", "sym"]).is_err());
    }

    #[test]
    fn locator_parses_lsp_toggles_and_rejects_conflicts() {
        let cli = LocatorCli::try_parse_from(["t", "s", "--file", "a.rs", "--no-lsp"]).unwrap();
        assert!(cli.args.no_lsp && !cli.args.lsp);
        // `--syntactic` is an alias for `--no-lsp`.
        let cli = LocatorCli::try_parse_from(["t", "s", "--file", "a.rs", "--syntactic"]).unwrap();
        assert!(cli.args.no_lsp);
        // The two are mutually exclusive.
        assert!(LocatorCli::try_parse_from(["t", "s", "--file", "a.rs", "--lsp", "--no-lsp"]).is_err());
    }

    #[test]
    fn reject_no_lsp_blocks_the_fallback_free_commands() {
        let with = LocatorCli::try_parse_from(["t", "s", "--file", "a.rs", "--no-lsp"]).unwrap().args;
        assert!(reject_no_lsp(&with, "def").is_err(), "def has no syntactic fallback");
        let without = LocatorCli::try_parse_from(["t", "s", "--file", "a.rs"]).unwrap().args;
        assert!(reject_no_lsp(&without, "def").is_ok());
    }

    #[test]
    fn reject_no_lsp_is_a_usage_error_not_a_crash_report() {
        // Regression: this used to be a plain `bail!`, so mistyping a flag printed
        // a dozen frames of stack trace under RUST_BACKTRACE. For a tool whose job
        // is spending an agent's context carefully, that buried a one-line answer
        // under noise. Typing it as `UsageError` is what lets `main` keep it to one
        // line and exit 2 instead of 1.
        let with = LocatorCli::try_parse_from(["t", "s", "--file", "a.rs", "--no-lsp"]).unwrap().args;
        let err = reject_no_lsp(&with, "def").unwrap_err();
        assert!(
            err.downcast_ref::<crate::UsageError>().is_some(),
            "must stay a UsageError, or the backtrace dump comes back"
        );
    }

    #[test]
    fn depth_defaults_to_one_and_parses() {
        let cli = DepthCli::try_parse_from(["t", "f", "--file", "a.rs"]).unwrap();
        assert_eq!(cli.args.depth, 1);
        let cli = DepthCli::try_parse_from(["t", "f", "--file", "a.rs", "--depth", "3"]).unwrap();
        assert_eq!(cli.args.depth, 3);
    }

    // ---- name -> position resolution ---------------------------------------

    fn tree() -> Vec<Value> {
        vec![
            json!({
                "name": "Rebuilder", "kind": 23,
                "range": { "start": { "line": 3, "character": 0 }, "end": { "line": 20, "character": 1 } },
                "selectionRange": { "start": { "line": 3, "character": 11 }, "end": { "line": 3, "character": 20 } },
                "children": [
                    {
                        "name": "rebuild", "kind": 6,
                        "range": { "start": { "line": 8, "character": 4 }, "end": { "line": 15, "character": 5 } },
                        "selectionRange": { "start": { "line": 8, "character": 11 }, "end": { "line": 8, "character": 18 } }
                    }
                ]
            }),
            json!({
                "name": "helper", "kind": 12,
                "range": { "start": { "line": 22, "character": 0 }, "end": { "line": 24, "character": 1 } },
                "selectionRange": { "start": { "line": 22, "character": 7 }, "end": { "line": 22, "character": 13 } }
            }),
        ]
    }

    #[test]
    fn find_symbol_returns_the_selection_range_position() {
        let helper = find_symbol(&tree(), "helper").unwrap();
        assert_eq!((helper.line, helper.character), (22, 7));
        assert_eq!(helper.kind, "fn");
    }

    #[test]
    fn find_symbol_recurses_into_children_and_matches_qualified_names() {
        // Nested method, found by bare name...
        let m = find_symbol(&tree(), "rebuild").unwrap();
        assert_eq!((m.line, m.character), (8, 11));
        // ...and by a `Type::method` qualified form (final segment matches).
        let q = find_symbol(&tree(), "Rebuilder::rebuild").unwrap();
        assert_eq!((q.line, q.character), (8, 11));
    }

    #[test]
    fn find_symbol_missing_is_none() {
        assert!(find_symbol(&tree(), "nonexistent").is_none());
    }

    // ---- enclosing symbol (callers) ----------------------------------------

    #[test]
    fn enclosing_symbol_picks_the_innermost_container() {
        // Line 10 is inside `rebuild` (8..15), itself inside `Rebuilder` (3..20).
        // The method is the tighter span, so it wins.
        let encl = enclosing_symbol(&tree(), 10).unwrap();
        assert_eq!(encl.name, "rebuild");
        // Line 4 is inside Rebuilder but before the method: the struct wins.
        let encl = enclosing_symbol(&tree(), 4).unwrap();
        assert_eq!(encl.name, "Rebuilder");
        // A line outside every range has no enclosing symbol.
        assert!(enclosing_symbol(&tree(), 100).is_none());
    }

    #[test]
    fn symbol_at_refinds_a_declaration_by_selection_line() {
        let s = symbol_at(&tree(), 8).unwrap();
        assert_eq!(s.name, "rebuild");
    }

    // ---- signature extraction ----------------------------------------------

    #[test]
    fn extract_signature_skips_the_module_path_line() {
        let hover = "```rust\nkopitiam_semantic::session\n```\n\n```rust\npub fn references(&mut self, file: &Path) -> Result<Vec<Location>>\n```\n\n---\n\nFinds all references.";
        let sig = extract_signature(hover);
        assert_eq!(sig, "pub fn references(&mut self, file: &Path) -> Result<Vec<Location>>");
    }

    #[test]
    fn extract_signature_reads_a_field_type() {
        let hover = "```rust\nspans: Vec<Span>\n```";
        assert_eq!(extract_signature(hover), "spans: Vec<Span>");
    }

    #[test]
    fn extract_signature_empty_hover_is_empty() {
        assert_eq!(extract_signature("```rust\n```"), "");
    }

    // ---- impl-header detection ---------------------------------------------

    #[test]
    fn line_is_impl_header_matches_impl_openers() {
        assert!(line_is_impl_header("impl Widget {"));
        assert!(line_is_impl_header("    impl Display for Widget {"));
        assert!(line_is_impl_header("impl<T> Iterator for Wrap<T> {"));
        assert!(!line_is_impl_header("    let impl_note = 1;"));
        assert!(!line_is_impl_header("// impl details"));
        assert!(!line_is_impl_header("fn f() {}"));
    }

    // ---- call-candidate scanning (callees) ---------------------------------

    #[test]
    fn call_candidates_finds_calls_and_skips_keywords() {
        // A tiny body across lines 2..=4 (0-based). `if (` and `while(` are
        // keywords, not calls; `helper(` and `compute (` are.
        let text = "fn outer() {\n    if (cond) {}\n    helper(x);\n    let y = compute (z);\n}\n";
        let cands = call_candidates_in_range(text, 1, 4);
        let names: Vec<&str> = cands.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"helper"), "helper( is a call");
        assert!(names.contains(&"compute"), "compute ( (with space) is a call");
        assert!(!names.contains(&"if"), "if( is a keyword, not a call");
        // Positions are 0-based file coordinates.
        let helper = cands.iter().find(|c| c.name == "helper").unwrap();
        assert_eq!((helper.line, helper.character), (2, 4));
    }

    #[test]
    fn call_candidates_dedupe_by_name() {
        let text = "fn f() {\n    g(); g(); g();\n}\n";
        let cands = call_candidates_in_range(text, 0, 2);
        assert_eq!(cands.iter().filter(|c| c.name == "g").count(), 1);
    }

    #[test]
    fn call_candidates_respects_the_range() {
        // The call on line 5 is outside the 1..=3 window and must be ignored.
        let text = "a\nb\nc\nd\ne\noutside();\n";
        let cands = call_candidates_in_range(text, 1, 3);
        assert!(cands.is_empty());
    }

    // ---- output shaping -----------------------------------------------------

    fn loc(path: &str, line: u32, ch: u32) -> Location {
        use kopitiam_semantic::{Position, Range};
        Location {
            path: PathBuf::from(path),
            range: Range {
                start: Position { line, character: ch },
                end: Position { line, character: ch + 1 },
            },
        }
    }

    #[test]
    fn sorted_coords_sorts_and_dedupes() {
        let locs = vec![loc("b.rs", 2, 0), loc("a.rs", 5, 1), loc("a.rs", 5, 1), loc("a.rs", 1, 0)];
        let coords = sorted_coords(&locs);
        assert_eq!(coords.len(), 3, "the duplicate a.rs:5:1 collapses");
        assert_eq!(coords[0].display(), "a.rs:1:0");
        assert_eq!(coords[1].display(), "a.rs:5:1");
        assert_eq!(coords[2].display(), "b.rs:2:0");
    }

    #[test]
    fn refs_json_shape() {
        let coords = sorted_coords(&[loc("a.rs", 1, 2)]);
        let out = RefsJson { symbol: "sym", references: &coords };
        let json: Value = serde_json::from_str(&serde_json::to_string(&out).unwrap()).unwrap();
        assert_eq!(json["symbol"], "sym");
        assert_eq!(json["references"][0]["file"], "a.rs");
        assert_eq!(json["references"][0]["line"], 1);
        assert_eq!(json["references"][0]["character"], 2);
    }

    #[test]
    fn def_json_carries_signature_and_location() {
        let coord = Coord::at(Path::new("a.rs"), 3, 4);
        let out = DefJson { symbol: "s", signature: Some("fn s()"), definition: &coord };
        let json: Value = serde_json::from_str(&serde_json::to_string(&out).unwrap()).unwrap();
        assert_eq!(json["signature"], "fn s()");
        assert_eq!(json["definition"]["line"], 3);
    }

    #[test]
    fn impls_json_renames_trait_field() {
        let impls = vec![ImplSite { site: Coord::at(Path::new("a.rs"), 1, 0), header: "impl T for A {".into() }];
        let out = ImplsJson { trait_name: "T", impls: &impls };
        let json: Value = serde_json::from_str(&serde_json::to_string(&out).unwrap()).unwrap();
        assert_eq!(json["trait"], "T");
        assert_eq!(json["impls"][0]["header"], "impl T for A {");
    }
}
