//! Dependency-free **syntactic** fallbacks for the rust-analyzer-backed
//! navigation commands (`outline`, `refs`) — token-max Task II-1 / II-2 on a
//! repo too large for rust-analyzer to index in a bounded time.
//!
//! # Why this exists (dogfood finding)
//!
//! `outline` and `refs` are the marquee code-navigation commands, but they need
//! a live rust-analyzer, and on a workspace the size of kopitiam's own repo
//! rust-analyzer can fail to finish indexing inside the connect timeout ("timed
//! out waiting for a message from the language server"). CLAUDE.md's dogfood
//! rule requires these commands to work *on kopitiam itself*, so when the
//! semantic engine is unavailable the commands fall back to a **deterministic,
//! pure-Rust, no-server** approximation instead of hard-failing:
//!
//! * [`outline_file`] — a single-file item scan (top-level + `impl`/`trait`/`mod`
//!   scoped `fn`/`struct`/`enum`/`trait`/`impl`/`const`/`type`), signatures only,
//!   **no bodies**. This ports the brace/paren-aware signature-extraction idea
//!   from `scripts/skeleton-gen.sh` (its `scan_c` state machine) to Rust,
//!   retargeted at Rust syntax: comments and string/char literals are stripped
//!   so their braces never miscount, then a brace/paren-depth state machine
//!   walks the remaining structural stream.
//! * [`grep_symbol`] — a ripgrep-grade, word-boundary textual search for a
//!   symbol across the workspace's `.rs` files, emitted as `file:line:character`
//!   hits the caller must treat as approximate (they are labelled
//!   `(syntactic, not semantic — verify)` at the call site).
//!
//! Both are honest heuristics, not a parser: `refs`' textual hits include
//! matches in comments/strings, and `outline` does not resolve macros or
//! cross-file items. That is the deal — grep-grade coverage with zero indexing
//! latency, clearly labelled so a caller never mistakes it for the semantic
//! answer.
//!
//! [`with_fallback`] and [`connect_ra`] are the shared plumbing: `connect_ra`
//! spawns rust-analyzer with a **configurable** index timeout
//! (`KOPITIAM_RA_TIMEOUT_SECS`) and a "still indexing…" progress note, and
//! `with_fallback` runs the semantic path first, then either propagates the
//! error (when the caller forced `--lsp`) or runs the syntactic fallback with a
//! stderr note.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use kopitiam_semantic::{Outline, OutlineItem, ProgressKind, RustAnalyzerSession};

/// The default rust-analyzer index timeout (seconds) for the CLI's navigation
/// commands, overridable with `KOPITIAM_RA_TIMEOUT_SECS`. Matches the historic
/// [`RustAnalyzerSession::connect`] default; the env override lets a user extend
/// it on a very large workspace (or shorten it to reach the syntactic fallback
/// sooner).
const DEFAULT_RA_TIMEOUT_SECS: u64 = 180;

/// Reads `KOPITIAM_RA_TIMEOUT_SECS` (a positive integer number of seconds),
/// falling back to [`DEFAULT_RA_TIMEOUT_SECS`] when it is unset or unparsable.
fn ra_timeout() -> Duration {
    let secs = std::env::var("KOPITIAM_RA_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_RA_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Spawns rust-analyzer for `root` with a configurable index timeout, announcing
/// progress on **stderr** so a `--json` stdout stays clean.
///
/// Unlike the bare [`RustAnalyzerSession::connect`], this reads the timeout from
/// `KOPITIAM_RA_TIMEOUT_SECS` and, rather than sitting silent until the deadline,
/// prints a one-shot "still indexing…" note the first time rust-analyzer reports
/// indexing progress — so a large-workspace connect looks like work in progress,
/// not a hang (fix #3).
pub fn connect_ra(root: &Path) -> Result<RustAnalyzerSession> {
    let root = std::fs::canonicalize(root)
        .with_context(|| format!("resolving workspace root {}", root.display()))?;
    let timeout = ra_timeout();
    eprintln!(
        "Starting rust-analyzer and waiting for it to index {} (timeout {}s; \
         set KOPITIAM_RA_TIMEOUT_SECS to change, or pass --no-lsp for the syntactic fallback)...",
        root.display(),
        timeout.as_secs(),
    );
    let mut announced = false;
    RustAnalyzerSession::connect_with_observed("rust-analyzer", &[], &root, timeout, |update| {
        // The first real indexing `report` tells the user the server is chewing
        // through a big workspace rather than hung. Only once, to keep stderr calm.
        if !announced && matches!(update.kind, ProgressKind::Report) {
            eprintln!("rust-analyzer still indexing… (large workspace; this can take a while)");
            announced = true;
        }
    })
}

/// Runs `primary` (the rust-analyzer path); on error, either propagate it
/// (when `force_primary`, i.e. the caller passed `--lsp`) so the command exits
/// non-zero, or run `fallback` (the syntactic path) after a stderr note.
///
/// This is the single decision point behind fix #1 (never exit 0 on a failed
/// query) and fix #2 (auto-fallback): a forced-`--lsp` failure is a hard error,
/// while the default behaviour degrades to the syntactic approximation.
pub fn with_fallback<T>(
    force_primary: bool,
    primary: impl FnOnce() -> Result<T>,
    note: &str,
    fallback: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match primary() {
        Ok(value) => Ok(value),
        Err(err) if force_primary => {
            Err(err.context(format!("{note} (and --lsp forbids the syntactic fallback)")))
        }
        Err(err) => {
            eprintln!("{note} ({err:#}); using syntactic fallback");
            fallback()
        }
    }
}

// ---------------------------------------------------------------------------
// Single-file syntactic outline.
// ---------------------------------------------------------------------------

/// Builds a body-free [`Outline`] of `file` by a lightweight Rust item scan,
/// with no rust-analyzer. The result plugs into the same [`Outline::to_skeleton`]
/// / `--json` rendering the semantic path uses.
pub fn outline_file(file: &Path) -> Result<Outline> {
    let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    Ok(Outline { items: scan_items(&text) })
}

/// One item the scanner recognised, before it becomes an [`OutlineItem`].
struct Scanned {
    kind: &'static str,
    name: String,
    detail: Option<String>,
    /// Whether opening this item's `{` begins a *container* (`impl`/`trait`/`mod`)
    /// whose members are themselves items, versus a *body* (fn/struct/…) whose
    /// interior is skipped.
    container: bool,
}

/// Scans `text` for top-level and container-scoped item declarations, in source
/// order, with no bodies. See the module docs for the approach.
fn scan_items(text: &str) -> Vec<OutlineItem> {
    let code = strip_to_code(text);
    let mut items: Vec<OutlineItem> = Vec::new();
    // One entry per open `{`: true = container (scan its members), false = body.
    let mut stack: Vec<bool> = Vec::new();
    let mut unit = String::new();
    let mut unit_line: u32 = 0;
    let mut started = false;
    // Parenthesis/bracket depth, so a `;` or `{` inside `(...)`/`[...]`
    // (e.g. the `;` in `[u8; 4]`, or the parens of an attribute) is not
    // mistaken for a structural terminator.
    let mut pdepth: u32 = 0;

    let begin = |started: &mut bool, unit_line: &mut u32, line: u32| {
        if !*started {
            *started = true;
            *unit_line = line;
        }
    };

    for (ch, line) in code {
        match ch {
            '(' | '[' => {
                pdepth += 1;
                begin(&mut started, &mut unit_line, line);
                unit.push(ch);
            }
            ')' | ']' => {
                pdepth = pdepth.saturating_sub(1);
                begin(&mut started, &mut unit_line, line);
                unit.push(ch);
            }
            '{' if pdepth == 0 => {
                let eligible = stack.iter().all(|&c| c);
                let head = unit.trim();
                let mut pushed = false;
                if eligible && !head.is_empty()
                    && let Some(item) = classify(head, '{')
                {
                    items.push(OutlineItem {
                        line: unit_line,
                        kind: item.kind.to_string(),
                        name: item.name,
                        detail: item.detail,
                        depth: stack.len(),
                    });
                    stack.push(item.container);
                    pushed = true;
                }
                if !pushed {
                    stack.push(false);
                }
                unit.clear();
                started = false;
            }
            '}' if pdepth == 0 => {
                stack.pop();
                unit.clear();
                started = false;
            }
            ';' if pdepth == 0 => {
                let eligible = stack.iter().all(|&c| c);
                let head = unit.trim();
                if eligible && !head.is_empty()
                    && let Some(item) = classify(head, ';')
                {
                    items.push(OutlineItem {
                        line: unit_line,
                        kind: item.kind.to_string(),
                        name: item.name,
                        detail: item.detail,
                        depth: stack.len(),
                    });
                }
                unit.clear();
                started = false;
            }
            _ => {
                if !started {
                    if ch.is_whitespace() {
                        continue;
                    }
                    started = true;
                    unit_line = line;
                }
                unit.push(ch);
            }
        }
    }

    items.sort_by_key(|item| item.line);
    items
}

/// Classifies the accumulated text preceding a `{` or `;` terminator into an
/// item, or `None` if it is not a declaration this outline reports (a macro
/// invocation, a `use`/`mod;` decl, a statement, …).
fn classify(head_raw: &str, terminator: char) -> Option<Scanned> {
    let squeezed = squeeze(head_raw);
    let no_attrs = strip_leading_attrs(&squeezed);
    let mut h = strip_vis(no_attrs).trim();
    if h.is_empty() {
        return None;
    }

    // Peel leading modifier keywords that can precede an item keyword.
    loop {
        let (word, rest) = first_word(h);
        match word {
            "unsafe" | "async" | "default" | "auto" | "extern" => {
                h = rest.trim_start();
            }
            "const" => {
                // `const fn` is a function; a bare `const` is a const item.
                let (next, _) = first_word(rest.trim_start());
                if next == "fn" {
                    h = rest.trim_start();
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    let (kw, rem) = first_word(h);
    let rem = rem.trim();
    match kw {
        "fn" => {
            let (name, rest) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned {
                kind: "fn",
                name,
                detail: fn_detail(rest),
                container: false,
            })
        }
        "struct" | "union" => {
            let (name, _) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned { kind: "struct", name, detail: None, container: false })
        }
        "enum" => {
            let (name, _) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned { kind: "enum", name, detail: None, container: false })
        }
        "trait" => {
            let (name, _) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned { kind: "trait", name, detail: None, container: true })
        }
        "mod" => {
            // Only `mod name { … }` is an item here; a `mod name;` declaration is
            // an out-of-file module, which this single-file outline cannot show.
            if terminator != '{' {
                return None;
            }
            let (name, _) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned { kind: "mod", name, detail: None, container: true })
        }
        "impl" => {
            // rust-analyzer names an impl block by its whole header; mirror that so
            // `Outline::to_skeleton`'s "don't double the kind" rule kicks in.
            let name = if rem.starts_with('<') {
                format!("impl{rem}")
            } else {
                format!("impl {rem}")
            };
            Some(Scanned { kind: "impl", name: name.trim_end().to_string(), detail: None, container: true })
        }
        "const" | "static" => {
            let (name, rest) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned { kind: "const", name, detail: type_after_colon(rest), container: false })
        }
        "type" => {
            let (name, _) = split_ident(rem);
            (!name.is_empty()).then_some(Scanned { kind: "type", name, detail: None, container: false })
        }
        _ => None,
    }
}

/// The `(params) -> Ret` slice of a fn's post-name text (everything from the
/// first `(`), or `None` if there is no parameter list.
fn fn_detail(rest: &str) -> Option<String> {
    rest.find('(').map(|p| rest[p..].trim().to_string())
}

/// The type of a `const`/`static`, i.e. the text between `:` and `=` (or end),
/// trimmed. `None` if the declaration has no `: Type`.
fn type_after_colon(rest: &str) -> Option<String> {
    let after = rest.trim_start().strip_prefix(':')?;
    let ty = after.split('=').next().unwrap_or(after).trim();
    (!ty.is_empty()).then(|| ty.to_string())
}

/// The leading identifier of `s` (`[A-Za-z0-9_]+`) and the remainder after it.
fn split_ident(s: &str) -> (String, &str) {
    let s = s.trim_start();
    let end = s.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(s.len());
    (s[..end].to_string(), &s[end..])
}

/// The leading identifier-ish word of `s` and the remainder, as borrowed slices
/// (used to peek keywords without allocating).
fn first_word(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    let end = s.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(s.len());
    (&s[..end], &s[end..])
}

/// Collapses all runs of whitespace to single spaces and trims the ends.
fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strips leading `#[…]` / `#![…]` attribute groups (already whitespace-squeezed)
/// from the front of an item head.
fn strip_leading_attrs(s: &str) -> &str {
    let mut s = s.trim_start();
    loop {
        let t = s.trim_start();
        let after_hash = match t.strip_prefix('#') {
            Some(r) => r.strip_prefix('!').unwrap_or(r),
            None => break,
        };
        let Some(inner) = after_hash.trim_start().strip_prefix('[') else { break };
        // Find the matching `]` (attributes can nest brackets).
        let mut depth = 1usize;
        let mut cut = None;
        for (k, ch) in inner.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        cut = Some(k + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        match cut {
            Some(idx) => s = &inner[idx..],
            None => break,
        }
    }
    s.trim_start()
}

/// Strips a leading visibility modifier (`pub`, `pub(crate)`, `pub(in …)`, …)
/// from a whitespace-squeezed item head, respecting the word boundary so a name
/// like `pubkey` is not mangled.
fn strip_vis(s: &str) -> &str {
    let s = s.trim_start();
    if s == "pub" {
        return "";
    }
    if let Some(rest) = s.strip_prefix("pub(") {
        // Skip the restriction group up to its `)`.
        if let Some(close) = rest.find(')') {
            return rest[close + 1..].trim_start();
        }
    }
    if let Some(rest) = s.strip_prefix("pub ") {
        return rest.trim_start();
    }
    s
}

// ---------------------------------------------------------------------------
// Comment / string / char stripping (so their braces never miscount).
// ---------------------------------------------------------------------------

/// Rewrites `text` into a stream of `(structural_char, 0-based-line)`: line and
/// block comments and the *contents* of string/char literals become spaces, so
/// only real structural punctuation and identifiers remain. Line numbers stay
/// accurate (newlines inside comments/strings still advance the counter).
///
/// This is the Rust analogue of `skeleton-gen.sh`'s per-line comment/preprocessor
/// stripping pass that runs before its brace scanner.
fn strip_to_code(text: &str) -> Vec<(char, u32)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<(char, u32)> = Vec::with_capacity(n);
    let mut line: u32 = 0;
    let mut i = 0;

    while i < n {
        let c = chars[i];

        // Line comment: skip to end of line (the newline is handled next loop).
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            i += 2;
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // Block comment (Rust block comments nest).
        if c == '/' && i + 1 < n && chars[i + 1] == '*' {
            i += 2;
            let mut depth = 1usize;
            while i < n && depth > 0 {
                if chars[i] == '\n' {
                    line += 1;
                    i += 1;
                } else if chars[i] == '/' && i + 1 < n && chars[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && i + 1 < n && chars[i + 1] == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        // Raw string: r"…", r#"…"#, br"…", … — no escapes, so scan to the
        // closing quote followed by the matching number of `#`.
        if (c == 'r' || c == 'b') && is_raw_string_start(&chars, i) {
            i = skip_raw_string(&chars, i, &mut line);
            continue;
        }
        // Normal string literal.
        if c == '"' {
            i += 1;
            while i < n {
                match chars[i] {
                    '\\' => {
                        if i + 1 < n && chars[i + 1] == '\n' {
                            line += 1;
                        }
                        i += 2;
                    }
                    '\n' => {
                        line += 1;
                        i += 1;
                    }
                    '"' => {
                        i += 1;
                        break;
                    }
                    _ => i += 1,
                }
            }
            continue;
        }
        // Char literal vs lifetime.
        if c == '\'' {
            if is_char_literal(&chars, i) {
                i = skip_char_literal(&chars, i);
                continue;
            }
            // A lifetime (`'a`, `'static`, `'_`) — harmless, emit and move on.
            out.push((c, line));
            i += 1;
            continue;
        }
        if c == '\n' {
            // Emit a separator so adjacent tokens on different lines don't merge,
            // then advance the line counter.
            out.push((' ', line));
            line += 1;
            i += 1;
            continue;
        }
        out.push((c, line));
        i += 1;
    }

    out
}

/// True if a raw-string literal (`r"`, `r#"`, `br#"`, …) begins at `i`.
fn is_raw_string_start(chars: &[char], i: usize) -> bool {
    let n = chars.len();
    let mut j = i;
    if j < n && chars[j] == 'b' {
        j += 1;
    }
    if j < n && chars[j] == 'r' {
        j += 1;
    } else {
        return false;
    }
    while j < n && chars[j] == '#' {
        j += 1;
    }
    j < n && chars[j] == '"'
}

/// Advances past a raw-string literal starting at `i`, counting any interior
/// newlines into `line`. Returns the index just past the closing delimiter.
fn skip_raw_string(chars: &[char], i: usize, line: &mut u32) -> usize {
    let n = chars.len();
    let mut j = i;
    if chars[j] == 'b' {
        j += 1;
    }
    j += 1; // the `r`
    let mut hashes = 0usize;
    while j < n && chars[j] == '#' {
        hashes += 1;
        j += 1;
    }
    j += 1; // the opening `"`
    while j < n {
        if chars[j] == '\n' {
            *line += 1;
            j += 1;
        } else if chars[j] == '"' {
            let mut k = j + 1;
            let mut cnt = 0usize;
            while k < n && cnt < hashes && chars[k] == '#' {
                k += 1;
                cnt += 1;
            }
            if cnt == hashes {
                return k;
            }
            j += 1;
        } else {
            j += 1;
        }
    }
    j
}

/// Distinguishes a char literal (`'a'`, `'\n'`, `'{'`) from a lifetime (`'a`)
/// at position `i` (where `chars[i] == '\''`).
fn is_char_literal(chars: &[char], i: usize) -> bool {
    let n = chars.len();
    if i + 1 >= n {
        return false;
    }
    // An escape is always a char literal.
    if chars[i + 1] == '\\' {
        return true;
    }
    // `'x'` — a single char then a closing quote.
    i + 2 < n && chars[i + 2] == '\''
}

/// Advances past a char literal starting at `i`, handling escapes including
/// `'\u{…}'`. Returns the index just past the closing quote.
fn skip_char_literal(chars: &[char], i: usize) -> usize {
    let n = chars.len();
    let mut j = i + 1;
    if j < n && chars[j] == '\\' {
        j += 1; // the backslash
        if j + 1 < n && chars[j] == 'u' && chars[j + 1] == '{' {
            j += 2;
            while j < n && chars[j] != '}' {
                j += 1;
            }
            if j < n {
                j += 1; // the `}`
            }
        } else if j < n {
            j += 1; // a simple escape char
        }
    } else if j < n {
        j += 1; // the single char
    }
    if j < n && chars[j] == '\'' {
        j += 1;
    }
    j
}

// ---------------------------------------------------------------------------
// Workspace-wide textual symbol search (refs fallback).
// ---------------------------------------------------------------------------

/// One textual match of a symbol: a forward-slash `file` path plus a 0-based
/// `line`/`character`, mirroring the coordinate shape the semantic `refs` emits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextHit {
    pub file: String,
    pub line: u32,
    pub character: u32,
}

/// Searches every `.rs` file under `root` for whole-word occurrences of
/// `symbol`, returning sorted, deduped [`TextHit`]s. This is the `refs`
/// fallback: grep-grade, not semantic — a hit inside a comment or string counts,
/// and unrelated same-named symbols are not disambiguated.
///
/// `vendor/`, `target/`, and dot-directories are skipped, matching the blast
/// radius the deterministic `refactor` command already uses.
pub fn grep_symbol(root: &Path, symbol: &str) -> Result<Vec<TextHit>> {
    let mut hits: Vec<TextHit> = Vec::new();
    let walker = walkdir::WalkDir::new(root).sort_by_file_name().into_iter().filter_entry(|e| {
        // Never prune the search root itself (its name may legitimately start
        // with '.', e.g. a tempdir); only prune noise *sub*directories.
        if e.depth() == 0 {
            return true;
        }
        let name = e.file_name().to_string_lossy();
        !(e.file_type().is_dir() && (name == "target" || name == "vendor" || name.starts_with('.')))
    });
    for entry in walker.filter_map(std::result::Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else { continue };
        let file = to_slash(path);
        for (li, raw) in text.lines().enumerate() {
            for col in word_match_columns(raw, symbol) {
                hits.push(TextHit { file: file.clone(), line: li as u32, character: col });
            }
        }
    }
    hits.sort_by(|a, b| (&a.file, a.line, a.character).cmp(&(&b.file, b.line, b.character)));
    hits.dedup();
    Ok(hits)
}

/// The 0-based character columns at which `word` occurs in `line` as a whole
/// word — i.e. not flanked by identifier characters (`[A-Za-z0-9_]`). Empty
/// `word` never matches.
fn word_match_columns(line: &str, word: &str) -> Vec<u32> {
    let chars: Vec<char> = line.chars().collect();
    let target: Vec<char> = word.chars().collect();
    let (n, m) = (chars.len(), target.len());
    let mut out = Vec::new();
    if m == 0 || m > n {
        return out;
    }
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut i = 0;
    while i + m <= n {
        if chars[i..i + m] == target[..] {
            let before_ok = i == 0 || !is_ident(chars[i - 1]);
            let after_ok = i + m == n || !is_ident(chars[i + m]);
            if before_ok && after_ok {
                out.push(i as u32);
                i += m;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Normalises a path to forward slashes for stable, cross-platform output
/// (CLAUDE.md "Cross-platform paths"; same rule as `tokens::to_slash`).
fn to_slash(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- with_fallback: the exit-code decision -----------------------------

    #[test]
    fn with_fallback_forced_lsp_propagates_the_error() {
        // `--lsp` forces the semantic path: a failure must NOT be masked by the
        // fallback, so the command exits non-zero (fix #1).
        let out: Result<i32> = with_fallback(
            true,
            || anyhow::bail!("rust-analyzer timed out"),
            "rust-analyzer unavailable",
            || Ok(99),
        );
        assert!(out.is_err(), "forced --lsp must propagate the RA failure");
    }

    #[test]
    fn with_fallback_default_runs_the_fallback() {
        let out: Result<i32> =
            with_fallback(false, || anyhow::bail!("boom"), "note", || Ok(7));
        assert_eq!(out.unwrap(), 7, "default behaviour falls back on RA failure");
    }

    #[test]
    fn with_fallback_prefers_the_primary_on_success() {
        let out: Result<i32> = with_fallback(false, || Ok(3), "note", || Ok(7));
        assert_eq!(out.unwrap(), 3, "a working RA path is used as-is");
    }

    // ---- outline scan ------------------------------------------------------

    const FIXTURE: &str = r#"//! A module doc comment { with braces } that must be ignored.
use std::collections::HashMap;

/// Doc for MAX.
pub const MAX: usize = 4;

#[derive(Clone, Debug)]
pub struct Widget {
    pub id: u32,
    name: String,
}

pub enum Shape {
    Circle,
    Square,
}

pub trait Draw {
    fn draw(&self) -> String;
}

impl Draw for Widget {
    fn draw(&self) -> String {
        // a body brace } and a string "}" must not confuse the scanner
        let s = "{ not an item }";
        format!("{}", s)
    }
}

impl Widget {
    pub fn new(id: u32) -> Self {
        fn helper_hidden() {}
        Widget { id, name: String::new() }
    }
}

pub type Ids = [u32; 4];

pub fn free_fn(a: i32, b: i32) -> i32 {
    a + b
}
"#;

    fn find<'a>(items: &'a [OutlineItem], name: &str) -> Option<&'a OutlineItem> {
        items.iter().find(|i| i.name == name)
    }

    #[test]
    fn outline_scan_finds_items_with_no_bodies() {
        let items = scan_items(FIXTURE);

        // The const, struct, enum, trait, both impls, type alias, and free fn.
        assert_eq!(find(&items, "MAX").map(|i| i.kind.as_str()), Some("const"));
        assert_eq!(find(&items, "Widget").map(|i| i.kind.as_str()), Some("struct"));
        assert_eq!(find(&items, "Shape").map(|i| i.kind.as_str()), Some("enum"));
        assert_eq!(find(&items, "Draw").map(|i| i.kind.as_str()), Some("trait"));
        assert_eq!(find(&items, "Ids").map(|i| i.kind.as_str()), Some("type"));

        // A free function keeps its signature detail, no body.
        let free = find(&items, "free_fn").expect("free_fn present");
        assert_eq!(free.kind, "fn");
        assert_eq!(free.detail.as_deref(), Some("(a: i32, b: i32) -> i32"));
        assert_eq!(free.depth, 0);

        // Methods live at depth 1 under their impl/trait.
        let new = find(&items, "new").expect("new present");
        assert_eq!(new.kind, "fn");
        assert_eq!(new.depth, 1);

        // Two impl blocks, both named `impl …` at depth 0.
        let impls: Vec<&OutlineItem> = items.iter().filter(|i| i.kind == "impl").collect();
        assert_eq!(impls.len(), 2);
        assert!(impls.iter().all(|i| i.depth == 0));
        assert!(impls.iter().any(|i| i.name == "impl Draw for Widget"));
        assert!(impls.iter().any(|i| i.name == "impl Widget"));
    }

    #[test]
    fn outline_scan_omits_bodies_and_body_locals() {
        let items = scan_items(FIXTURE);
        // A nested fn inside a body must not surface as an item.
        assert!(find(&items, "helper_hidden").is_none(), "body-local items are not outlined");
        // No struct fields (they live inside the struct body).
        assert!(find(&items, "id").is_none());
        assert!(find(&items, "name").is_none());
        // The rendered skeleton never contains body statements or string braces.
        let skeleton = Outline { items }.to_skeleton();
        assert!(!skeleton.contains("format!"), "no body text in the skeleton");
        assert!(!skeleton.contains("not an item"), "string contents never leak");
    }

    #[test]
    fn outline_scan_is_line_ordered_and_zero_based() {
        let items = scan_items(FIXTURE);
        // Lines are 0-based and strictly increasing in emission order.
        let lines: Vec<u32> = items.iter().map(|i| i.line).collect();
        assert!(lines.windows(2).all(|w| w[0] <= w[1]), "items are sorted by line");
        // `pub const MAX` is on line index 4 (0-based) in FIXTURE.
        assert_eq!(find(&items, "MAX").unwrap().line, 4);
    }

    // ---- textual (refs) search --------------------------------------------

    #[test]
    fn word_match_columns_is_whole_word_only() {
        // Matches `id` as a word but not inside `ids` / `void` / `id_x`.
        assert_eq!(word_match_columns("let id = ids + void_id + id;", "id"), vec![4, 25]);
        // No match when it only appears as a substring.
        assert!(word_match_columns("consider considering", "consider").len() == 1);
        // Empty needle never matches.
        assert!(word_match_columns("anything", "").is_empty());
    }

    #[test]
    fn grep_symbol_finds_word_hits_across_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "fn to_slash() {}\nlet x = to_slash();\n").unwrap();
        std::fs::write(dir.path().join("src/b.rs"), "// to_slash mentioned\nto_slashy();\n").unwrap();
        // A skipped directory must not contribute hits.
        std::fs::create_dir(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/gen.rs"), "to_slash();\n").unwrap();

        let hits = grep_symbol(dir.path(), "to_slash").unwrap();
        // a.rs: def line 0, call line 1. b.rs: the comment mention (grep-grade).
        // `to_slashy` is NOT a whole-word match. `target/` is pruned.
        assert_eq!(hits.len(), 3, "two in a.rs, one comment hit in b.rs, target pruned");
        assert!(hits.iter().all(|h| !h.file.contains("target")));
        assert!(hits.iter().any(|h| h.file.ends_with("a.rs") && h.line == 0));
        assert!(hits.iter().any(|h| h.file.ends_with("a.rs") && h.line == 1));
        assert!(hits.iter().any(|h| h.file.ends_with("b.rs") && h.line == 0));
    }
}
