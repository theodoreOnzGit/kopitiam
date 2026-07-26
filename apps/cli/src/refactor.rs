//! The `refactor` subcommand group: deterministic, mechanical code
//! transformations — token-max Task II-8 (`kopitiam_token_max.md` §11).
//!
//! `rename` (`crate::rename`) already proves the shape a deterministic
//! refactor takes here: compute a pure `Vec<FileEdit>`, print a unified diff
//! by default, and only touch disk on `--apply`. This module reuses that
//! machinery verbatim — [`kopitiam_semantic::edit::FileEdit`],
//! [`kopitiam_semantic::edit::diff`], and
//! [`kopitiam_semantic::edit::write_file_edits`] — so a refactor here inherits
//! the exact same safe-by-default ergonomics without a rust-analyzer session.
//!
//! The first (and, so far, only) refactor is `refactor add-derive <Derive>
//! --filter <pattern>`: find every `struct`/`enum`/`union` definition whose
//! name matches `--filter` across a scanned directory and add `<Derive>` to it
//! — extending an existing single-line `#[derive(...)]`, or inserting a fresh
//! one when none is present. This replaces "the LLM opens N files and hand-edits
//! each derive list" (§0.3) with one verifiable command over a preview diff.
//!
//! # Why this is safe to do without a full parser
//!
//! The transform is a careful *line scan*, not a semantic rewrite, so it is
//! deliberately conservative — **precision over recall**. It only recognises a
//! type definition at the logical start of a line (optional visibility, then
//! `struct`/`enum`/`union`, then the name), and it *skips* any type whose
//! attribute block carries a `derive(` in a form it cannot fully account for
//! (a `cfg_attr(..., derive(..))`, say). Better to leave a type untouched than
//! to corrupt it or introduce a conflicting duplicate `impl`.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use kopitiam_semantic::edit::{self, FileEdit};

/// Options for `kopitiam refactor`.
#[derive(Args, Debug)]
pub struct RefactorArgs {
    #[command(subcommand)]
    action: RefactorAction,
}

/// The available deterministic refactors. New mechanical transforms land here
/// as sibling subcommands, each following the same preview/`--apply` gate.
#[derive(Subcommand, Debug)]
enum RefactorAction {
    /// Add a derive to every matching type definition across a directory,
    /// previewing the change as a diff unless `--apply` is given.
    AddDerive(AddDeriveArgs),
}

/// Options for `kopitiam refactor add-derive`.
#[derive(Args, Debug)]
pub struct AddDeriveArgs {
    /// The derive to add, e.g. `Clone`, `PartialEq`, or a path like
    /// `serde::Serialize`.
    pub derive: String,

    /// Restrict to type definitions whose name matches this pattern. A pattern
    /// containing `*` or `?` is treated as a glob anchored to the whole name
    /// (`Config*` matches `ConfigV2` but not `MyConfig`); otherwise it is a
    /// case-sensitive substring match (`Config` matches both). Required, so a
    /// bare run can never rewrite every type in the tree.
    #[arg(long)]
    pub filter: String,

    /// Directory (or a single `.rs` file) to scan. Defaults to the current
    /// directory. This bounds the blast radius: nothing outside it is read or
    /// written, and `vendor/`, `target/`, and dot-directories are always
    /// skipped.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Write the computed changes to disk. Without this flag, `add-derive`
    /// only prints a preview diff and leaves every file untouched.
    #[arg(long)]
    pub apply: bool,

    /// Emit the planned edits as JSON (each file with the type names and
    /// 1-based definition line numbers it would touch) instead of a diff. A
    /// listing only — it never writes, regardless of `--apply`.
    #[arg(long)]
    pub json: bool,
}

/// Runs `kopitiam refactor ...`.
pub fn run(args: RefactorArgs) -> Result<()> {
    match args.action {
        RefactorAction::AddDerive(a) => run_add_derive(a),
    }
}

/// What `add-derive` did (or would do) to one type definition.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    /// An existing single-line `#[derive(...)]` was extended in place.
    Extended,
    /// A fresh `#[derive(...)]` line was inserted above the definition.
    Added,
}

impl Action {
    fn as_str(&self) -> &'static str {
        match self {
            Action::Extended => "extended",
            Action::Added => "added",
        }
    }
}

/// One type definition the refactor touched.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TypeEdit {
    /// The type's name (`struct`/`enum`/`union` identifier).
    type_name: String,
    /// 1-based line of the definition in the *original* file.
    line: usize,
    action: Action,
}

/// The result of planning one file: its rewritten content plus a record of
/// every type touched. An empty `edits` means the file is unchanged (and
/// `updated` equals the input).
#[derive(Debug, Clone, PartialEq, Eq)]
struct FilePlan {
    updated: String,
    edits: Vec<TypeEdit>,
}

/// How `--filter` interprets its pattern.
enum Filter {
    /// No `*`/`?`: a case-sensitive substring test on the type name.
    Substring(String),
    /// Contains `*`/`?`: a glob anchored to the whole type name.
    Glob(String),
}

impl Filter {
    fn parse(pattern: &str) -> Self {
        if pattern.contains('*') || pattern.contains('?') {
            Filter::Glob(pattern.to_string())
        } else {
            Filter::Substring(pattern.to_string())
        }
    }

    fn matches(&self, name: &str) -> bool {
        match self {
            Filter::Substring(s) => name.contains(s.as_str()),
            Filter::Glob(g) => glob_match(g, name),
        }
    }
}

/// A classic `*`/`?` glob matcher, anchored to the whole `text`. `*` matches
/// any (possibly empty) run of characters; `?` matches exactly one. No
/// character classes — a type name never needs them. Iterative with
/// backtracking so it stays linear-ish and allocation-free.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    // Backtrack anchors: where to resume the pattern (`star`) and text
    // (`star_t`) after the most recent `*` if the current attempt fails.
    let (mut star, mut star_t): (Option<usize>, usize) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            star_t = ti;
            pi += 1;
        } else if let Some(s) = star {
            // Mismatch: let the last `*` swallow one more character.
            pi = s + 1;
            star_t += 1;
            ti = star_t;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// The pure heart of `add-derive`: given a file's `content`, the `derive` to
/// add, and the `filter`, compute the rewritten content and the list of type
/// definitions touched — without reading or writing anything.
///
/// This is deterministic (a single top-to-bottom scan, edits applied
/// bottom-to-top so line indices stay valid) and idempotent: a type that
/// already derives `derive` is left exactly as-is, so running twice makes no
/// second change. Line endings and indentation are preserved; only the derive
/// list of a matched type is altered.
fn plan_file(content: &str, derive: &str, filter: &Filter) -> FilePlan {
    // Peel off a leading UTF-8 BOM (which is not whitespace, so it would
    // otherwise mask a definition on the first line) and re-attach it verbatim
    // at the end, so the BOM is preserved exactly at the front and never moved.
    let (bom, body) = match content.strip_prefix('\u{FEFF}') {
        Some(rest) => ("\u{FEFF}", rest),
        None => ("", content),
    };

    // Segments each keep their own trailing newline (`\n`/`\r\n`), so simply
    // concatenating them reproduces the original bytes exactly.
    let mut lines: Vec<String> = body.split_inclusive('\n').map(str::to_string).collect();

    // Pass 1 (top-to-bottom): decide what to do at each matching definition,
    // recording the ORIGINAL line index so edits are reported in file order.
    let mut planned: Vec<(usize, PlannedOp, TypeEdit)> = Vec::new();
    for (idx, seg) in lines.iter().enumerate() {
        let Some(name) = type_def_name(strip_newline(seg).0) else {
            continue;
        };
        if !filter.matches(name) {
            continue;
        }
        let block = attribute_block(&lines, idx);
        // `None` means: already derives it, or an ambiguous derive form -> skip.
        if let Some(op) = derive_plan(&lines, &block, derive) {
            let action = match &op {
                PlannedOp::Extend { .. } => Action::Extended,
                PlannedOp::Insert => Action::Added,
            };
            planned.push((
                idx,
                op,
                TypeEdit { type_name: name.to_string(), line: idx + 1, action },
            ));
        }
    }

    // Pass 2 (bottom-to-top): apply, so an insert above a lower definition
    // never shifts the index of a higher one still to be edited.
    let mut edits: Vec<TypeEdit> = Vec::with_capacity(planned.len());
    for (idx, op, edit) in planned.iter().rev() {
        match op {
            PlannedOp::Extend { line: r, idents } => {
                let (_, nl) = strip_newline(&lines[*r]);
                let indent = leading_ws(strip_newline(&lines[*r]).0);
                let mut all = idents.clone();
                all.push(derive.to_string());
                lines[*r] = format!("{indent}#[derive({})]{nl}", all.join(", "));
            }
            PlannedOp::Insert => {
                let (def_text, def_nl) = strip_newline(&lines[*idx]);
                let indent = leading_ws(def_text);
                // Insert on its own line above the definition. If the
                // definition had no trailing newline (EOF), give the new line a
                // `\n` so the two do not run together.
                let nl = if def_nl.is_empty() { "\n" } else { def_nl };
                lines.insert(*idx, format!("{indent}#[derive({derive})]{nl}"));
            }
        }
        edits.push(edit.clone());
    }
    edits.reverse(); // restore file order

    FilePlan { updated: format!("{bom}{}", lines.concat()), edits }
}

/// What to do at one matched definition, resolved during pass 1.
enum PlannedOp {
    /// Extend the single-line `#[derive(...)]` at this original line index,
    /// whose current idents are `idents`.
    Extend { line: usize, idents: Vec<String> },
    /// Insert a fresh `#[derive(...)]` above the definition.
    Insert,
}

/// Decides how to add `derive` given the definition's `block` of attribute
/// line indices. Returns `None` to skip: either the type already derives it
/// (idempotent no-op), or the block contains a `derive(` in a form this scan
/// cannot safely rewrite (precision over recall).
fn derive_plan(lines: &[String], block: &[AttrLine], derive: &str) -> Option<PlannedOp> {
    let mut derive_lines: Vec<(usize, Vec<String>)> = Vec::new();
    let mut single_line_derives = 0usize;

    for attr in block {
        let joined: String = attr.indices.iter().map(|i| strip_newline(&lines[*i]).0).collect();
        if !joined.contains("derive(") {
            continue;
        }
        // A derive somewhere in this attribute. Only a clean `#[derive( ... )]`
        // (its whole text is exactly one derive attribute) is safe to reason
        // about; anything else (a `cfg_attr(..., derive(..))`, trailing code)
        // is ambiguous, so bail on the whole type via `?`.
        let idents = parse_clean_derive(&joined)?;
        if attr.indices.len() == 1 {
            single_line_derives += 1;
        }
        derive_lines.push((attr.indices[0], idents));
    }

    // Already present anywhere -> nothing to do (idempotent).
    let already = derive_lines.iter().any(|(_, ids)| ids.iter().any(|id| id == derive));
    if already {
        return None;
    }

    // Exactly one derive, and it is a single line -> extend it in place (this
    // is the common `#[derive(Debug)]` -> `#[derive(Debug, Clone)]` case).
    if derive_lines.len() == 1 && single_line_derives == 1 {
        let (line, idents) = derive_lines.into_iter().next().unwrap();
        return Some(PlannedOp::Extend { line, idents });
    }

    // No derive, or a shape we will not mangle (multi-line, or several derive
    // attributes): add a fresh, separate `#[derive(..)]` line. Valid Rust — a
    // type may carry multiple derive attributes — and idempotent on re-run.
    Some(PlannedOp::Insert)
}

/// One attribute in a definition's preceding attribute block, as the set of
/// line indices it spans (one for a single-line attribute, several for a
/// multi-line one).
struct AttrLine {
    indices: Vec<usize>,
}

/// Collects the contiguous attribute block immediately above the definition at
/// `def_idx`, walking upward. Blank lines and `//`/`///`/`//!` comments between
/// attributes (and between the block and the item) are allowed and skipped, as
/// Rust permits. A multi-line attribute (`#[foo(\n ... \n)]`) is bracket-matched
/// so it is captured as a single unit — critical for spotting a multi-line
/// `#[derive(\n ... \n)]` that would otherwise be missed and duplicated. The
/// walk stops at the first line that is neither attribute, comment, nor blank
/// (i.e. real code — the previous item).
///
/// Returned in top-to-bottom order.
fn attribute_block(lines: &[String], def_idx: usize) -> Vec<AttrLine> {
    let mut attrs: Vec<AttrLine> = Vec::new();
    let mut i = def_idx;
    while i > 0 {
        i -= 1;
        let trimmed = strip_newline(&lines[i]).0.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue; // blank line or comment: allowed between attrs and item
        }
        if trimmed.ends_with(']') {
            // End of an attribute (single- or multi-line). Walk up until the
            // `[`/`]` bracket balance closes on a line that opens with `#[`.
            let mut span = vec![i];
            let mut balance = bracket_balance(trimmed);
            while balance > 0 && i > 0 {
                i -= 1;
                let t = strip_newline(&lines[i]).0;
                balance += bracket_balance(t.trim());
                span.push(i);
            }
            let opens = strip_newline(&lines[span[span.len() - 1]]).0.trim_start();
            if balance == 0 && (opens.starts_with("#[") || opens.starts_with("#![")) {
                span.reverse();
                attrs.push(AttrLine { indices: span });
                continue;
            }
            // Unbalanced or not actually an attribute: treat as code, stop.
            break;
        }
        break; // real code (the previous item): the block ends here
    }
    attrs.reverse(); // top-to-bottom
    attrs
}

/// Net `]`-minus-`[` count on a line, for matching a multi-line attribute from
/// its closing line upward (a positive result means unmatched closes remain).
fn bracket_balance(s: &str) -> i32 {
    let mut bal = 0i32;
    for c in s.chars() {
        match c {
            ']' => bal += 1,
            '[' => bal -= 1,
            _ => {}
        }
    }
    bal
}

/// Parses a string that should be exactly one clean derive attribute —
/// `#[derive( A, B, c::D )]` (single- or multi-line, whitespace-normalised) —
/// into its list of derive idents. Returns `None` if the text is not exactly a
/// derive attribute (leading/trailing junk, unbalanced parens), so the caller
/// can conservatively skip an ambiguous type.
fn parse_clean_derive(joined: &str) -> Option<Vec<String>> {
    let s = joined.trim();
    let inner = s.strip_prefix("#[derive(")?.strip_suffix(")]")?;
    // No nested parens are expected inside a derive list; if present, the shape
    // is unusual enough to skip.
    if inner.contains('(') || inner.contains(')') {
        return None;
    }
    let idents: Vec<String> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    Some(idents)
}

/// If `line` is a type definition at its logical start (optional `pub`/
/// `pub(..)` visibility, then `struct`/`enum`/`union`, then a name), returns the
/// type name. Anything else — a use, an impl, a `struct` buried mid-line or in a
/// string, an inline-attribute-then-keyword line — returns `None`.
fn type_def_name(line: &str) -> Option<&str> {
    let rest = line.trim_start();
    // Strip an optional visibility modifier: `pub`, or `pub(...)`.
    let rest = match rest.strip_prefix("pub") {
        Some(after) => {
            if let Some(paren) = after.strip_prefix('(') {
                // `pub(crate)`, `pub(in path)`, ...: skip to the matching `)`.
                let close = paren.find(')')?;
                paren[close + 1..].trim_start()
            } else if after.starts_with(char::is_whitespace) {
                after.trim_start()
            } else {
                // `pub` was a prefix of a longer identifier (e.g. `publish`):
                // not a visibility modifier, so keep the original line.
                rest
            }
        }
        None => rest,
    };

    let after_kw = ["struct", "enum", "union"].iter().find_map(|kw| {
        let r = rest.strip_prefix(kw)?;
        // The keyword must be followed by whitespace, then the name.
        r.starts_with(char::is_whitespace).then(|| r.trim_start())
    })?;

    let name_len = after_kw
        .char_indices()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(after_kw.len());
    let name = &after_kw[..name_len];
    // A name must be non-empty and start with an identifier character (not a
    // digit) — guards against `struct 123` style noise.
    let first = name.chars().next()?;
    if name.is_empty() || first.is_ascii_digit() {
        return None;
    }
    Some(name)
}

/// Splits a `split_inclusive('\n')` segment into its text and its trailing
/// newline (`"\r\n"`, `"\n"`, or `""` at EOF).
fn strip_newline(seg: &str) -> (&str, &str) {
    if let Some(s) = seg.strip_suffix("\r\n") {
        (s, "\r\n")
    } else if let Some(s) = seg.strip_suffix('\n') {
        (s, "\n")
    } else {
        (seg, "")
    }
}

/// The leading whitespace (indentation) of a line.
fn leading_ws(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Runs `refactor add-derive`: scan, plan, then preview / JSON-list / apply.
fn run_add_derive(args: AddDeriveArgs) -> Result<()> {
    let derive = args.derive.trim();
    if derive.is_empty() {
        bail!("the derive to add must not be empty");
    }
    let filter = Filter::parse(&args.filter);

    let files = collect_rust_files(&args.root)?;
    let mut file_edits: Vec<FileEdit> = Vec::new();
    let mut plans: Vec<(PathBuf, Vec<TypeEdit>)> = Vec::new();
    for path in &files {
        let original = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue, // unreadable / non-UTF-8: skip, never guess
        };
        let plan = plan_file(&original, derive, &filter);
        if plan.edits.is_empty() {
            continue;
        }
        plans.push((path.clone(), plan.edits.clone()));
        file_edits.push(FileEdit { path: path.clone(), original, updated: plan.updated });
    }

    if args.json {
        print!("{}", plans_to_json(derive, &plans)?);
        return Ok(());
    }

    let type_count: usize = plans.iter().map(|(_, e)| e.len()).sum();
    if file_edits.is_empty() {
        println!(
            "No struct/enum/union definition matching --filter {:?} needs `{derive}` added (nothing to do).",
            args.filter
        );
        return Ok(());
    }

    if args.apply {
        edit::write_file_edits(&file_edits)?;
        println!(
            "Added `{derive}` to {type_count} type(s) across {} file(s):",
            file_edits.len()
        );
        for (path, edits) in &plans {
            for e in edits {
                println!("  {}:{} {} ({})", path.display(), e.line, e.type_name, e.action.as_str());
            }
        }
    } else {
        print!("{}", edit::diff(&file_edits));
        println!(
            "(preview only: `{derive}` would be added to {type_count} type(s) in {} file(s); \
             re-run with --apply to write these changes)",
            file_edits.len()
        );
        // The token-max win (§0.3): this is one command in place of the LLM
        // opening each of these files and hand-editing every derive list.
        println!(
            "This replaces opening and editing {} file(s) by hand with one verifiable command.",
            file_edits.len()
        );
    }

    Ok(())
}

/// Serialises the planned edits as JSON: an array of files, each with the type
/// names and 1-based definition lines it would touch.
fn plans_to_json(derive: &str, plans: &[(PathBuf, Vec<TypeEdit>)]) -> Result<String> {
    use serde_json::json;
    let files: Vec<_> = plans
        .iter()
        .map(|(path, edits)| {
            json!({
                "path": path.to_string_lossy(),
                "edits": edits
                    .iter()
                    .map(|e| json!({
                        "type": e.type_name,
                        "line": e.line,
                        "action": e.action.as_str(),
                    }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let doc = json!({ "derive": derive, "files": files });
    Ok(format!("{}\n", serde_json::to_string_pretty(&doc)?))
}

/// Collects the `.rs` files to scan under `root`, in a stable sorted order.
/// `root` may be a single `.rs` file (scanned alone) or a directory (walked
/// recursively). `vendor/`, `target/`, and dot-directories (`.git`,
/// `.kopitiam`, ...) are always skipped, and files carrying an `@generated`
/// marker near their top are left alone.
fn collect_rust_files(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        if root.extension().and_then(|e| e.to_str()) == Some("rs") && !is_generated(root) {
            return Ok(vec![root.to_path_buf()]);
        }
        return Ok(Vec::new());
    }

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_skipped_dir(e))
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if entry.file_type().is_file()
            && path.extension().and_then(|e| e.to_str()) == Some("rs")
            && !is_generated(path)
        {
            files.push(path.to_path_buf());
        }
    }
    // walkdir's order is not guaranteed stable across platforms; sort so the
    // planned edits (and any diff) are deterministic.
    files.sort();
    Ok(files)
}

/// Whether a directory entry is one the scan must never descend into:
/// `vendor/`, `target/`, or any dot-directory.
fn is_skipped_dir(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    // The root itself (depth 0) is never skipped, even if it is a dot-dir the
    // user pointed at deliberately.
    if entry.depth() == 0 {
        return false;
    }
    match entry.file_name().to_str() {
        Some(name) => name == "vendor" || name == "target" || name.starts_with('.'),
        None => false,
    }
}

/// Whether a `.rs` file looks machine-generated (a `@generated` marker in its
/// first few lines, the conventional signal), so the refactor leaves it alone.
fn is_generated(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    content.lines().take(5).any(|l| l.contains("@generated"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn plan(content: &str, derive: &str, pattern: &str) -> FilePlan {
        plan_file(content, derive, &Filter::parse(pattern))
    }

    // ---- the pure edit-computation fn (asserted exactly) --------------------

    #[test]
    fn a_struct_with_no_derives_gets_a_fresh_derive() {
        let src = "pub struct Config {\n    a: u32,\n}\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, "#[derive(Clone)]\npub struct Config {\n    a: u32,\n}\n");
        assert_eq!(
            out.edits,
            vec![TypeEdit { type_name: "Config".into(), line: 1, action: Action::Added }]
        );
    }

    #[test]
    fn an_existing_single_line_derive_is_extended_in_place() {
        let src = "#[derive(Debug)]\nstruct Config;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, "#[derive(Debug, Clone)]\nstruct Config;\n");
        assert_eq!(out.edits[0].action, Action::Extended);
    }

    #[test]
    fn a_type_that_already_derives_it_is_unchanged() {
        // Idempotence: the derive is already present, so nothing moves.
        let src = "#[derive(Debug, Clone)]\nstruct Config;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, src);
        assert!(out.edits.is_empty());
    }

    #[test]
    fn running_twice_makes_no_second_change() {
        let src = "pub struct Config {\n    a: u32,\n}\n";
        let once = plan(src, "Clone", "Config");
        let twice = plan(&once.updated, "Clone", "Config");
        assert_eq!(twice.updated, once.updated);
        assert!(twice.edits.is_empty());
    }

    #[test]
    fn a_non_matching_type_is_untouched() {
        let src = "struct Config;\nstruct Other;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, "#[derive(Clone)]\nstruct Config;\nstruct Other;\n");
        assert_eq!(out.edits.len(), 1);
        assert_eq!(out.edits[0].type_name, "Config");
    }

    // ---- scan precision -----------------------------------------------------

    #[test]
    fn enums_and_unions_and_visibility_are_recognised() {
        assert_eq!(type_def_name("enum E { A }"), Some("E"));
        assert_eq!(type_def_name("pub(crate) struct S<T>(T);"), Some("S"));
        assert_eq!(type_def_name("    pub union U { a: u8 }"), Some("U"));
        assert_eq!(type_def_name("pub struct Tuple(u32, u32);"), Some("Tuple"));
    }

    #[test]
    fn non_definitions_are_ignored() {
        // A `use`, an impl, a mid-line `struct` (e.g. in a string), an
        // inline-attribute line, and a `pub`-prefixed identifier are all skipped.
        assert_eq!(type_def_name("use foo::struct_like;"), None);
        assert_eq!(type_def_name("impl Config {"), None);
        assert_eq!(type_def_name("let s = \"struct Fake;\";"), None);
        assert_eq!(type_def_name("#[derive(Clone)] struct Inline;"), None);
        assert_eq!(type_def_name("pubfn not_vis() {}"), None);
    }

    #[test]
    fn a_cfg_attr_derive_is_skipped_conservatively() {
        // The scan cannot safely reason about a conditional derive, so it must
        // leave the whole type alone rather than risk a duplicate impl.
        let src = "#[cfg_attr(feature = \"x\", derive(Clone))]\nstruct Config;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, src);
        assert!(out.edits.is_empty());
    }

    #[test]
    fn a_multiline_derive_is_not_duplicated() {
        // A multi-line derive that already contains the target is detected via
        // bracket-matching, so no conflicting second derive is added.
        let src = "#[derive(\n    Debug,\n    Clone,\n)]\nstruct Config;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, src);
        assert!(out.edits.is_empty());
    }

    #[test]
    fn a_multiline_derive_missing_the_target_gets_a_fresh_line() {
        // Present target absent from the multi-line derive: add a separate,
        // valid `#[derive(..)]` rather than mangling the multi-line block.
        let src = "#[derive(\n    Debug,\n)]\nstruct Config;\n";
        let out = plan(src, "Clone", "Config");
        // The fresh derive lands directly above the definition line (below the
        // untouched multi-line derive) — valid Rust, and the multi-line block is
        // left byte-for-byte intact.
        assert_eq!(
            out.updated,
            "#[derive(\n    Debug,\n)]\n#[derive(Clone)]\nstruct Config;\n"
        );
        assert_eq!(out.edits[0].action, Action::Added);
    }

    #[test]
    fn indentation_and_crlf_line_endings_are_preserved() {
        let src = "mod m {\r\n    struct Inner;\r\n}\r\n";
        let out = plan(src, "Copy", "Inner");
        assert_eq!(out.updated, "mod m {\r\n    #[derive(Copy)]\r\n    struct Inner;\r\n}\r\n");
    }

    #[test]
    fn a_derive_survives_intervening_doc_comments() {
        // A doc comment between the derive and the struct must not hide the
        // existing derive (idempotence through comments).
        let src = "#[derive(Clone)]\n/// docs\nstruct Config;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, src);
        assert!(out.edits.is_empty());
    }

    #[test]
    fn a_leading_bom_is_preserved_and_does_not_mask_a_first_line_definition() {
        // A UTF-8 BOM is not whitespace, so a naive scan would miss a struct on
        // line 1; the BOM must survive at the front, unmoved.
        let src = "\u{FEFF}struct Config;\n";
        let out = plan(src, "Clone", "Config");
        assert_eq!(out.updated, "\u{FEFF}#[derive(Clone)]\nstruct Config;\n");
        assert_eq!(out.edits.len(), 1);
    }

    #[test]
    fn glob_and_substring_filters_select_the_right_types() {
        let src = "struct ConfigV2;\nstruct MyConfig;\nstruct Other;\n";
        // Substring: both Config-bearing names.
        let sub = plan(src, "Clone", "Config");
        assert_eq!(sub.edits.len(), 2);
        // Glob anchored to the whole name: only the one starting with Config.
        let glob = plan(src, "Clone", "Config*");
        assert_eq!(glob.edits.len(), 1);
        assert_eq!(glob.edits[0].type_name, "ConfigV2");
    }

    #[test]
    fn glob_matcher_basics() {
        assert!(glob_match("Config*", "ConfigV2"));
        assert!(!glob_match("Config*", "MyConfig"));
        assert!(glob_match("*Config", "MyConfig"));
        assert!(glob_match("Config?", "ConfigX"));
        assert!(!glob_match("Config?", "ConfigXY"));
        assert!(glob_match("*", "Anything"));
    }

    #[test]
    fn multiple_matches_in_one_file_are_all_planned_in_order() {
        let src = "struct A;\nstruct B;\n#[derive(Debug)]\nstruct C;\n";
        let out = plan(src, "Clone", "*"); // glob-all
        assert_eq!(out.edits.len(), 3);
        // Lines are the definitions' ORIGINAL 1-based line numbers: A=1, B=2,
        // and C=4 (its `#[derive(Debug)]` occupies line 3).
        assert_eq!(out.edits.iter().map(|e| e.line).collect::<Vec<_>>(), vec![1, 2, 4]);
        assert_eq!(
            out.updated,
            "#[derive(Clone)]\nstruct A;\n#[derive(Clone)]\nstruct B;\n#[derive(Debug, Clone)]\nstruct C;\n"
        );
    }

    // ---- disk behaviour: --apply writes, preview does not -------------------

    #[test]
    fn apply_writes_the_expected_bytes_and_preview_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("types.rs");
        let src = "struct Config;\n";
        std::fs::write(&path, src).unwrap();

        // Preview (no --apply): compute the edit but do NOT write it.
        let plan = plan_file(src, "Clone", &Filter::parse("Config"));
        let file_edits = vec![FileEdit {
            path: path.clone(),
            original: src.to_string(),
            updated: plan.updated.clone(),
        }];
        let rendered = edit::diff(&file_edits);
        assert!(rendered.contains("+#[derive(Clone)]"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), src, "preview must not touch the file");

        // Apply: the bytes on disk become exactly the planned content.
        edit::write_file_edits(&file_edits).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "#[derive(Clone)]\nstruct Config;\n"
        );
    }

    #[test]
    fn collect_rust_files_skips_vendor_target_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("b.rs"), "struct B;\n").unwrap();
        std::fs::write(root.join("a.rs"), "struct A;\n").unwrap();
        std::fs::create_dir_all(root.join("vendor")).unwrap();
        std::fs::write(root.join("vendor/v.rs"), "struct V;\n").unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("target/t.rs"), "struct T;\n").unwrap();

        let files = collect_rust_files(root).unwrap();
        let names: Vec<_> = files.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["a.rs", "b.rs"], "vendor/target skipped, sorted");
    }

    // ---- clap wiring --------------------------------------------------------

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: RefactorAction,
    }

    #[test]
    fn parses_add_derive_flags() {
        let cli =
            TestCli::try_parse_from(["r", "add-derive", "Clone", "--filter", "Config*", "--apply"])
                .unwrap();
        let RefactorAction::AddDerive(a) = cli.command;
        assert_eq!(a.derive, "Clone");
        assert_eq!(a.filter, "Config*");
        assert!(a.apply);
        assert!(!a.json);
    }

    #[test]
    fn filter_is_required() {
        // No --filter: a bare run must fail rather than rewrite the whole tree.
        assert!(TestCli::try_parse_from(["r", "add-derive", "Clone"]).is_err());
    }
}
