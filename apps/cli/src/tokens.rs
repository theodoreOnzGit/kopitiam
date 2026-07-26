//! The `tokens` subcommand: token accounting — token-max Task II-7.
//!
//! `tokens <path>` estimates how many BPE tokens a file (or every file under a
//! directory) would cost an LLM to read, so an agent can choose *read vs.
//! outline* informed rather than blind (`kopitiam_token_max.md` §0.7, §11
//! II-7). It is the second measurement axis Part I asked for (bytes/lines vs.
//! tokens, §268): a wide table is cheap in bytes but expensive in tokens; a CJK
//! document is the reverse.
//!
//! The estimate comes straight from [`kopitiam_tokenizer::estimate_tokens`] — a
//! dependency-free, deterministic per-script heuristic (no bundled vocab, no
//! model needed). This command is a thin shell over it: walk paths, sum, format.
//! Everything numeric lives in the tokenizer crate; this file only does I/O and
//! `--json` shaping.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use kopitiam_tokenizer::{estimate_tokens, estimate_tokens_by_line};
use serde::Serialize;

/// Options for `kopitiam tokens`.
#[derive(Args, Debug)]
pub struct TokensArgs {
    /// File or directory to estimate. A directory is walked recursively and
    /// every readable UTF-8 file is summed; unreadable or non-UTF-8 files
    /// (binaries) are skipped, not counted.
    pub path: PathBuf,

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
    let contents = collect_files(&args.path)?;
    let report = build_report(contents, args.json || args.by_line);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report, args.by_line);
    }
    Ok(())
}

/// Gathers `(display_path, content)` for every UTF-8 file at `path`: just the
/// one file if `path` is a file, or every readable UTF-8 file under it if it is
/// a directory (walked recursively, sorted by path for deterministic output).
/// Non-UTF-8 / unreadable files under a directory are silently skipped; a
/// single explicit file that cannot be read is a hard error (the user named it).
fn collect_files(path: &Path) -> Result<Vec<(String, String)>> {
    if path.is_dir() {
        let mut paths: Vec<PathBuf> = walkdir::WalkDir::new(path)
            .sort_by_file_name()
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(walkdir::DirEntry::into_path)
            .collect();
        paths.sort();
        let mut out = Vec::new();
        for p in paths {
            // A directory sweep skips binaries and unreadable files rather than
            // aborting the whole scan on the first one.
            if let Ok(text) = std::fs::read_to_string(&p) {
                out.push((to_slash(&p), text));
            }
        }
        Ok(out)
    } else {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        Ok(vec![(to_slash(path), text)])
    }
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
        assert_eq!(cli.args.path, PathBuf::from("src/lib.rs"));
        assert!(!cli.args.json);
        assert!(!cli.args.by_line);
    }

    #[test]
    fn parses_json_and_by_line_flags() {
        let cli = TestCli::try_parse_from(["t", "src", "--json", "--by-line"]).unwrap();
        assert!(cli.args.json);
        assert!(cli.args.by_line);
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
    fn to_slash_normalises_backslashes_cross_platform() {
        // On Windows the OS separator is `\`; forcing `/` gives identical,
        // grep-friendly output on Windows and Termux/Linux alike.
        assert_eq!(to_slash(Path::new("a\\b\\c")), "a/b/c");
        assert_eq!(to_slash(Path::new("a/b/c")), "a/b/c");
        assert_eq!(to_slash(Path::new("crates/x/src\\mod.rs")), "crates/x/src/mod.rs");
    }
}
