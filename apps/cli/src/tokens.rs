//! The `tokens` subcommand: token accounting — token-max Task II-7.
//!
//! `tokens <path>...` estimates how many BPE tokens a file (or every file under
//! a directory) would cost an LLM to read, so an agent can choose *read vs.
//! outline* informed rather than blind (`kopitiam_token_max.md` §0.7, §11
//! II-7). Takes **many paths at once** — sizing up three places shouldn't cost
//! three invocations — and counts any file named more than once only once, so
//! the grand total stay honest. It is the second measurement axis Part I asked for (bytes/lines vs.
//! tokens, §268): a wide table is cheap in bytes but expensive in tokens; a CJK
//! document is the reverse.
//!
//! The estimate comes straight from [`kopitiam_tokenizer::estimate_tokens`] — a
//! dependency-free, deterministic per-script heuristic (no bundled vocab, no
//! model needed). This command is a thin shell over it: walk paths, sum, format.
//! Everything numeric lives in the tokenizer crate; this file only does I/O and
//! `--json` shaping.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use kopitiam_tokenizer::{estimate_tokens, estimate_tokens_by_line};
use serde::Serialize;

/// Options for `kopitiam tokens`.
#[derive(Args, Debug)]
pub struct TokensArgs {
    /// One or more files / directories to estimate. Can pass many in one go —
    /// `tokens src/a.rs crates/b/src` — so you sizing up a few places at once
    /// don't need one call each. A directory is walked recursively and every
    /// readable UTF-8 file is summed; unreadable or non-UTF-8 files (binaries)
    /// are skipped, not counted.
    ///
    /// Overlapping paths are safe: the same file named twice (directly, or once
    /// directly and once via a parent directory) is counted **once**, so the
    /// grand total never double-counts. Files come out in the order you named
    /// their path, and sorted within each directory, so output stay
    /// deterministic.
    #[arg(required = true, num_args = 1..)]
    pub paths: Vec<PathBuf>,

    /// Emit machine-readable JSON: a per-file breakdown — each with its total
    /// and a per-line token count (`estimate_tokens_by_line`) — plus the grand
    /// total, instead of the human summary. (§0.2: a caller gates on the number
    /// without parsing prose.)
    #[arg(long)]
    pub json: bool,

    /// Also print the per-line breakdown in the human output (it is always in
    /// `--json`). Off by default so a single-file estimate stays one line.
    #[arg(long)]
    pub by_line: bool,

    /// Roll the estimate up **per directory** (each subtree's summed tokens and
    /// file count, tree-indented, heaviest subtree first) instead of the flat
    /// per-file list — so a token-heavy subtree is located in one call rather
    /// than several probe runs (follow-up P4). Composes with `--json`.
    #[arg(long)]
    pub tree: bool,

    /// In `--tree` mode, cap the printed directory depth. Aggregation still
    /// includes deeper files; they are just not broken out as their own nodes.
    #[arg(long, value_name = "N")]
    pub depth: Option<usize>,
}

/// One line's estimated token cost, mirroring
/// [`kopitiam_tokenizer::LineEstimate`] as a serializable record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct LineTokens {
    line: usize,
    tokens: usize,
}

/// The token estimate for a single file: its path (as given, for stable output),
/// its total estimated tokens, its line count, and — when requested — the
/// per-line breakdown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct FileTokens {
    path: String,
    tokens: usize,
    lines: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    by_line: Option<Vec<LineTokens>>,
}

/// The whole `tokens` result: every file estimated plus the grand total.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TokensReport {
    total_tokens: usize,
    files: Vec<FileTokens>,
}

/// Runs `kopitiam tokens`: collect the target files, estimate each, report.
pub fn run(args: TokensArgs) -> Result<()> {
    let contents = collect_all(&args.paths)?;
    // The per-line breakdown is meaningless in the per-directory rollup, so skip
    // computing it there.
    let report = build_report(contents, !args.tree && (args.json || args.by_line));

    if args.tree {
        let tree = build_dir_tree(&report.files);
        if args.json {
            let view = tree_view(&tree, args.depth);
            println!("{}", serde_json::to_string_pretty(&view)?);
        } else {
            print_tree(&tree, args.depth);
        }
    } else if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report, args.by_line);
    }
    Ok(())
}

/// Expands one user-given path into the concrete files it names: just itself if
/// it a file, or every file underneath if it a directory (walked recursively,
/// sorted so the output stay deterministic run to run).
///
/// The `bool` carries *how* the file got here, and that changes the error rule
/// downstream: `true` means the user named this file themselves, `false` means a
/// directory sweep picked it up. An explicitly named file that cannot be read is
/// a hard error (the user asked for it, so staying quiet would hide a typo);
/// one merely swept up is skipped, because a directory full of `.png` shouldn't
/// abort the scan.
fn expand(path: &Path) -> Vec<(PathBuf, bool)> {
    if path.is_dir() {
        let mut paths: Vec<PathBuf> = walkdir::WalkDir::new(path)
            .sort_by_file_name()
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(walkdir::DirEntry::into_path)
            .collect();
        paths.sort();
        paths.into_iter().map(|p| (p, false)).collect()
    } else {
        vec![(path.to_path_buf(), true)]
    }
}

/// Gathers `(display_path, content)` across every path the user named, reading
/// each distinct file exactly once.
///
/// Dedup matters for the total to mean anything: `tokens src src/lib.rs` names
/// `lib.rs` twice — once directly, once through its parent — and without dedup
/// the grand total would quietly count it twice and overstate the read cost,
/// which is the one number this command exists to get right.
///
/// The key is the **canonicalised** path, so two different spellings of one file
/// (`./a.rs` vs `a.rs`, or a symlink and its target) collapse together.
/// Canonicalising can fail — broken symlink, no permission — and then we fall
/// back to the path as given. That fallback direction is deliberate: worst case
/// we treat one file as two and overcount slightly, never the reverse where a
/// file the user explicitly asked about silently vanishes from the report.
fn collect_all(paths: &[PathBuf]) -> Result<Vec<(String, String)>> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for input in paths {
        for (p, explicit) in expand(input) {
            let key = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
            if !seen.insert(key) {
                continue;
            }
            match std::fs::read_to_string(&p) {
                Ok(text) => out.push((to_slash(&p), text)),
                Err(e) => {
                    if explicit {
                        return Err(e).with_context(|| format!("reading {}", p.display()));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Normalise a path to forward-slash form for stable output on BOTH Windows
/// (`\`) and Termux / Linux (`/`). `walkdir` joins the user-given base (often
/// already `/`) with the OS separator, so on Windows the raw display leaks
/// mixed `a/b\c`; forcing `/` keeps output copy-paste-, `grep`-, and
/// cross-platform-consistent. See CLAUDE.md "Cross-platform paths".
fn to_slash(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

/// Builds the [`TokensReport`] from already-read `(path, content)` pairs. This
/// is the pure, I/O-free core (deterministic given its inputs) so it is unit
/// tested without touching the filesystem. `include_by_line` attaches the
/// per-line breakdown to each file.
fn build_report(contents: Vec<(String, String)>, include_by_line: bool) -> TokensReport {
    let mut files = Vec::with_capacity(contents.len());
    let mut total = 0usize;
    for (path, text) in contents {
        let tokens = estimate_tokens(&text);
        total += tokens;
        let by_line = include_by_line.then(|| {
            estimate_tokens_by_line(&text)
                .into_iter()
                .map(|l| LineTokens { line: l.line, tokens: l.tokens })
                .collect()
        });
        // `str::lines` does not count a trailing newline as an extra line, which
        // is the intuitive "how many lines of content" a reader expects.
        let lines = text.lines().count();
        files.push(FileTokens { path, tokens, lines, by_line });
    }
    TokensReport { total_tokens: total, files }
}

/// Prints the human summary: one line per file, then a total. With `by_line`,
/// each file's per-line costs are listed underneath it.
fn print_human(report: &TokensReport, by_line: bool) {
    for file in &report.files {
        println!("{}: {} tokens ({} lines)", file.path, file.tokens, file.lines);
        if by_line
            && let Some(lines) = &file.by_line
        {
            for l in lines {
                println!("  {:>6}: {} tokens", l.line, l.tokens);
            }
        }
    }
    if report.files.len() != 1 {
        println!(
            "Total: {} tokens across {} files",
            report.total_tokens,
            report.files.len()
        );
    }
}

// ---------------------------------------------------------------------------
// `--tree`: per-directory token rollup (P4).
// ---------------------------------------------------------------------------

/// A directory node in the rollup tree: the token total and file count of the
/// whole subtree beneath it, plus its child directories. Files are not kept as
/// nodes — only their tokens/counts, aggregated into every ancestor — since the
/// point of `--tree` is to surface heavy *subtrees*, not re-list files.
#[derive(Default, Debug)]
struct DirTree {
    tokens: usize,
    files: usize,
    children: BTreeMap<String, DirTree>,
}

impl DirTree {
    /// Adds a file worth `tokens` at `components` (the path split on `/`,
    /// including the trailing file-name component) to this subtree, accumulating
    /// its cost into this node and every directory ancestor it passes through.
    fn insert(&mut self, components: &[&str], tokens: usize) {
        self.tokens += tokens;
        self.files += 1;
        if let Some((first, rest)) = components.split_first() {
            // A non-empty `rest` means `first` is a directory (the file name is
            // the final component); a file-name component creates no child node.
            if !rest.is_empty() {
                self.children.entry((*first).to_string()).or_default().insert(rest, tokens);
            }
        }
    }
}

/// Builds the [`DirTree`] from already-estimated files. The `path`s are the
/// forward-slash display paths [`build_report`] produced, so splitting on `/` is
/// platform-independent. Pure and deterministic — unit-tested without I/O.
fn build_dir_tree(files: &[FileTokens]) -> DirTree {
    let mut root = DirTree::default();
    for f in files {
        let components: Vec<&str> = f.path.split('/').filter(|s| !s.is_empty()).collect();
        if components.is_empty() {
            continue;
        }
        root.insert(&components, f.tokens);
    }
    root
}

/// Collapses a linear chain of single-child directories that hold no files of
/// their own (e.g. `apps` → `cli` → `src`) into one `apps/cli/src` label, so the
/// rollup does not waste indentation on directories that add no branching.
fn collapse(name: String, node: &DirTree) -> (String, &DirTree) {
    let mut name = name;
    let mut node = node;
    while node.children.len() == 1 {
        let (child_name, child) = node.children.iter().next().expect("len == 1");
        // Only collapse when every file lives deeper (no file sits directly in
        // `node`), i.e. the child accounts for all of the node's files.
        if child.files == node.files {
            name = format!("{name}/{child_name}");
            node = child;
        } else {
            break;
        }
    }
    (name, node)
}

/// A directory node's `--tree --json` form: the collapsed name plus its subtree
/// totals and (unless depth-capped) its children, heaviest first.
#[derive(Debug, Serialize)]
struct TreeNode {
    name: String,
    tokens: usize,
    files: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<TreeNode>,
}

/// Child directories of `node`, sorted by tokens descending (name ascending on a
/// tie), each collapsed, recursed until `max_depth` (None = unlimited).
fn children_view(node: &DirTree, depth: usize, max_depth: Option<usize>) -> Vec<TreeNode> {
    if max_depth.is_some_and(|md| depth >= md) {
        return Vec::new();
    }
    sorted_children(node)
        .into_iter()
        .map(|(name, child)| {
            let (name, node) = collapse(name, child);
            TreeNode {
                tokens: node.tokens,
                files: node.files,
                children: children_view(node, depth + 1, max_depth),
                name,
            }
        })
        .collect()
}

/// The whole tree as a serializable node rooted at `"."` (the scanned path),
/// carrying the grand totals with the per-directory breakdown beneath it.
fn tree_view(root: &DirTree, max_depth: Option<usize>) -> TreeNode {
    TreeNode {
        name: ".".to_string(),
        tokens: root.tokens,
        files: root.files,
        children: children_view(root, 0, max_depth),
    }
}

/// `node`'s children as `(name, node)` pairs sorted heaviest-first, breaking
/// ties by name so output is deterministic.
fn sorted_children(node: &DirTree) -> Vec<(String, &DirTree)> {
    let mut kids: Vec<(String, &DirTree)> =
        node.children.iter().map(|(n, c)| (n.clone(), c)).collect();
    kids.sort_by(|a, b| b.1.tokens.cmp(&a.1.tokens).then_with(|| a.0.cmp(&b.0)));
    kids
}

/// Prints the human rollup: each directory indented by depth as
/// `path/  <tokens> tokens (<files> files)`, heaviest subtree first, then a
/// grand-total line.
fn print_tree(root: &DirTree, max_depth: Option<usize>) {
    print_children(root, 0, max_depth);
    println!("Total: {} tokens across {} files", root.tokens, root.files);
}

fn print_children(node: &DirTree, depth: usize, max_depth: Option<usize>) {
    if max_depth.is_some_and(|md| depth >= md) {
        return;
    }
    for (name, child) in sorted_children(node) {
        let (name, node) = collapse(name, child);
        let indent = "  ".repeat(depth);
        println!("{indent}{name}/  {} tokens ({} files)", node.tokens, node.files);
        print_children(node, depth + 1, max_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: TokensArgs,
    }

    #[test]
    fn parses_a_bare_path() {
        let cli = TestCli::try_parse_from(["t", "src/lib.rs"]).unwrap();
        assert_eq!(cli.args.paths, vec![PathBuf::from("src/lib.rs")]);
        assert!(!cli.args.json);
        assert!(!cli.args.by_line);
    }

    #[test]
    fn parses_several_paths_in_one_go() {
        // The whole point of multi-path: one call instead of one per target.
        let cli = TestCli::try_parse_from(["t", "src/lib.rs", "crates/x/src", "README.md"]).unwrap();
        assert_eq!(
            cli.args.paths,
            vec![
                PathBuf::from("src/lib.rs"),
                PathBuf::from("crates/x/src"),
                PathBuf::from("README.md"),
            ]
        );
    }

    #[test]
    fn at_least_one_path_is_required() {
        // Bare `kopitiam tokens` must fail loudly, not silently estimate nothing
        // and print a confident "Total: 0 tokens".
        assert!(TestCli::try_parse_from(["t"]).is_err());
        assert!(TestCli::try_parse_from(["t", "--json"]).is_err());
    }

    #[test]
    fn parses_json_and_by_line_flags() {
        let cli = TestCli::try_parse_from(["t", "src", "--json", "--by-line"]).unwrap();
        assert!(cli.args.json);
        assert!(cli.args.by_line);
    }

    #[test]
    fn parses_tree_and_depth_flags() {
        let cli = TestCli::try_parse_from(["t", "src", "--tree", "--depth", "2"]).unwrap();
        assert!(cli.args.tree);
        assert_eq!(cli.args.depth, Some(2));
        // Tree is off and depth unset by default.
        let cli = TestCli::try_parse_from(["t", "src"]).unwrap();
        assert!(!cli.args.tree);
        assert!(cli.args.depth.is_none());
    }

    // ---- P4: --tree per-directory rollup ------------------------------------

    fn ft(path: &str, tokens: usize) -> FileTokens {
        FileTokens { path: path.to_string(), tokens, lines: 1, by_line: None }
    }

    #[test]
    fn build_dir_tree_rolls_up_tokens_and_file_counts() {
        let files = vec![ft("a/b/x.rs", 10), ft("a/b/y.rs", 20), ft("a/c/z.rs", 5)];
        let tree = build_dir_tree(&files);

        // The root aggregates every file.
        assert_eq!(tree.tokens, 35);
        assert_eq!(tree.files, 3);
        // `a` holds all three; `a/b` holds two summing to 30; `a/c` holds one.
        let a = &tree.children["a"];
        assert_eq!((a.tokens, a.files), (35, 3));
        assert_eq!((a.children["b"].tokens, a.children["b"].files), (30, 2));
        assert_eq!((a.children["c"].tokens, a.children["c"].files), (5, 1));
    }

    #[test]
    fn collapse_folds_a_single_child_chain() {
        // apps → cli → src (no files directly in apps or cli) collapses to one
        // label; the branching stops at `src`, which has two subdirectories.
        let files = vec![ft("apps/cli/src/a.rs", 3), ft("apps/cli/src/sub/b.rs", 7)];
        let tree = build_dir_tree(&files);
        let (name, node) = collapse("apps".to_string(), &tree.children["apps"]);
        assert_eq!(name, "apps/cli/src");
        assert_eq!(node.tokens, 10);
        // `src` has a file of its own (a.rs) plus a `sub/` dir, so it did not
        // collapse into `sub`.
        assert!(node.children.contains_key("sub"));
    }

    #[test]
    fn tree_view_sorts_children_heaviest_first() {
        // `heavy` outweighs `light`, so it sorts first regardless of name order.
        let files = vec![ft("light/a.rs", 1), ft("heavy/b.rs", 100)];
        let tree = build_dir_tree(&files);
        let view = tree_view(&tree, None);
        assert_eq!(view.name, ".");
        assert_eq!(view.tokens, 101);
        assert_eq!(view.children[0].name, "heavy");
        assert_eq!(view.children[1].name, "light");
    }

    #[test]
    fn tree_view_depth_cap_hides_deeper_nodes_but_keeps_totals() {
        let files = vec![ft("a/b/c/x.rs", 4)];
        let tree = build_dir_tree(&files);
        // depth 1: the collapsed `a/b/c` chain is one top node with no children
        // broken out, but its aggregate total is intact.
        let view = tree_view(&tree, Some(1));
        assert_eq!(view.children.len(), 1);
        assert_eq!(view.children[0].tokens, 4);
        assert!(view.children[0].children.is_empty());
    }

    #[test]
    fn build_report_sums_files_and_carries_totals() {
        // Two small files; the total is the sum of the per-file estimates.
        let files = vec![
            ("a.rs".to_string(), "fn main() {}\n".to_string()),
            ("b.rs".to_string(), "let x = 1;\n".to_string()),
        ];
        let report = build_report(files, false);
        assert_eq!(report.files.len(), 2);
        let a = estimate_tokens("fn main() {}\n");
        let b = estimate_tokens("let x = 1;\n");
        assert_eq!(report.files[0].tokens, a);
        assert_eq!(report.files[1].tokens, b);
        assert_eq!(report.total_tokens, a + b);
        // No per-line breakdown requested.
        assert!(report.files[0].by_line.is_none());
        // Line counts follow `str::lines`.
        assert_eq!(report.files[0].lines, 1);
    }

    #[test]
    fn build_report_attaches_per_line_when_requested() {
        let files = vec![("x.rs".to_string(), "alpha\nbeta gamma\n".to_string())];
        let report = build_report(files, true);
        let by_line = report.files[0].by_line.as_ref().expect("by_line present");
        // "a\nb\n" splits to three line entries via `estimate_tokens_by_line`.
        assert_eq!(by_line.len(), 3);
        assert_eq!(by_line[0].line, 1);
        assert_eq!(by_line[2].tokens, 0, "trailing empty line costs nothing");
    }

    #[test]
    fn empty_input_is_zero_tokens() {
        let report = build_report(vec![], false);
        assert_eq!(report.total_tokens, 0);
        assert!(report.files.is_empty());
    }

    #[test]
    fn json_shape_exposes_total_and_per_file() {
        let report = build_report(vec![("f.rs".to_string(), "hello world\n".to_string())], true);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
        assert!(json["total_tokens"].as_u64().is_some());
        let files = json["files"].as_array().unwrap();
        assert_eq!(files[0]["path"], "f.rs");
        assert!(files[0]["tokens"].as_u64().is_some());
        assert!(files[0]["by_line"].as_array().is_some());
    }

    #[test]
    fn overlapping_paths_count_each_file_once() {
        // `tokens <dir> <dir>/a.rs` names a.rs twice — once directly, once via
        // its parent. Without dedup the grand total silently doubles it, which
        // would break the one number this command exists to report.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn b() {}\n").unwrap();

        let both = collect_all(&[dir.path().to_path_buf(), dir.path().join("a.rs")]).unwrap();
        assert_eq!(both.len(), 2, "a.rs must not appear twice");

        // And the total matches the directory sweep on its own.
        let sweep = collect_all(&[dir.path().to_path_buf()]).unwrap();
        assert_eq!(
            build_report(both, false).total_tokens,
            build_report(sweep, false).total_tokens
        );
    }

    #[test]
    fn separate_paths_are_all_collected_in_argument_order() {
        // Two unrelated dirs: every file shows up, and the caller's ordering is
        // preserved so the output is stable and diffable.
        let one = tempfile::tempdir().unwrap();
        let two = tempfile::tempdir().unwrap();
        std::fs::write(one.path().join("first.rs"), "fn first() {}\n").unwrap();
        std::fs::write(two.path().join("second.rs"), "fn second() {}\n").unwrap();

        let got = collect_all(&[one.path().to_path_buf(), two.path().to_path_buf()]).unwrap();
        assert_eq!(got.len(), 2);
        assert!(got[0].0.ends_with("first.rs"), "argument order preserved");
        assert!(got[1].0.ends_with("second.rs"));
    }

    #[test]
    fn an_explicitly_named_missing_file_is_a_hard_error() {
        // The user named it, so a typo must surface — not be quietly skipped and
        // folded into a confident-looking total.
        let dir = tempfile::tempdir().unwrap();
        let err = collect_all(&[dir.path().join("nope.rs")]).unwrap_err();
        assert!(err.to_string().contains("nope.rs"), "error names the file");
    }

    #[test]
    fn to_slash_normalises_backslashes_cross_platform() {
        // On Windows the OS separator is `\`; forcing `/` gives identical,
        // grep-friendly output on Windows and Termux/Linux alike.
        assert_eq!(to_slash(Path::new("a\\b\\c")), "a/b/c");
        assert_eq!(to_slash(Path::new("a/b/c")), "a/b/c");
        assert_eq!(to_slash(Path::new("crates/x/src\\mod.rs")), "crates/x/src/mod.rs");
    }
}
