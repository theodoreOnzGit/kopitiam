//! Compact diagnostics — token-max Task II-4.
//!
//! `cargo check` / `cargo test` output is enormously verbose — backtraces,
//! repeated "for more information" notes, full type dumps — and agents run it
//! constantly. This module adds two subcommands that squeeze that firehose into
//! the few lines an agent actually needs (`kopitiam_token_max.md` §11 II-4):
//!
//! * `check [--compact] [--json]` — runs `cargo check --message-format=json`,
//!   collects the compiler diagnostics, and (with `--compact`/`--json`) emits
//!   **one line per distinct diagnostic, deduplicated, sorted by file**. The
//!   dedup is the whole point: one bad type routinely produces the *same*
//!   diagnostic across the lib, its tests, and its bins, and the raw stream
//!   repeats it once per target — collapsing those turns dozens of lines into
//!   the handful of real problems.
//! * `test [--compact] [--json]` — runs `cargo test`, and (compact) reports each
//!   failure as `name — assertion @ file:line`, not the full captured stdout.
//!
//! Without `--compact`/`--json`, each command streams cargo's normal output
//! unchanged, so nothing is lost when the full detail is wanted.
//!
//! # What is testable here
//!
//! The value lives in two *pure* functions — [`compactify`] (dedup + sort of
//! parsed diagnostics) and [`parse_test_failures`] (libtest output → failures)
//! — which are unit-tested against hand-built cargo JSON and sample test
//! output. Actually spawning cargo is a thin wrapper the tests do not exercise.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;
use serde_json::Value;

/// Options for `kopitiam check`.
#[derive(Args, Debug)]
pub struct CheckArgs {
    /// Directory to run `cargo check` in. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Restrict to one package (`cargo check -p <PACKAGE>`).
    #[arg(short, long)]
    pub package: Option<String>,

    /// Collapse the diagnostics to one deduplicated line per distinct problem,
    /// sorted by file. Without this (and without `--json`) the raw cargo output
    /// streams through unchanged.
    #[arg(long)]
    pub compact: bool,

    /// Emit the deduplicated diagnostics as JSON (implies the compact analysis).
    #[arg(long)]
    pub json: bool,
}

/// Options for `kopitiam test`.
#[derive(Args, Debug)]
pub struct TestArgs {
    /// Directory to run `cargo test` in. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Restrict to one package (`cargo test -p <PACKAGE>`).
    #[arg(short, long)]
    pub package: Option<String>,

    /// Report each failure as one line (`name — assertion @ file:line`) instead
    /// of the full captured output. Without this (and without `--json`) the raw
    /// cargo output streams through unchanged.
    #[arg(long)]
    pub compact: bool,

    /// Emit the failures as JSON (implies the compact analysis).
    #[arg(long)]
    pub json: bool,
}

// ---------------------------------------------------------------------------
// `check`
// ---------------------------------------------------------------------------

/// Runs `kopitiam check`.
pub fn run_check(args: CheckArgs) -> Result<()> {
    // Passthrough mode: no analysis requested, so just let cargo write to the
    // inherited stdout/stderr and return.
    if !args.compact && !args.json {
        let status = base_command("check", &args.root, args.package.as_deref())
            .status()
            .context("running `cargo check`")?;
        std::process::exit(status.code().unwrap_or(1));
    }

    let mut cmd = base_command("check", &args.root, args.package.as_deref());
    // `--all-targets` checks the lib, tests, bins, benches, and examples — the
    // realistic "check everything" pass. It is also exactly where cargo emits
    // the *same* source diagnostic once per target (a lib error shows up again
    // when the test target recompiles the same file), which is the duplication
    // `compactify` collapses.
    cmd.arg("--all-targets").arg("--message-format=json");
    let output = cmd.output().context("running `cargo check --message-format=json`")?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    let raws = parse_cargo_diagnostics(&stdout);
    let report = compactify(raws);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_compact_diagnostics(&report);
    }
    Ok(())
}

/// A single compiler diagnostic reduced to what a fix needs: severity, code,
/// message, and its primary location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Diag {
    level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    column: Option<u32>,
}

/// The compacted result: how many raw diagnostics collapsed to how many
/// distinct ones, and the distinct list (sorted by file/line).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CompactReport {
    raw_count: usize,
    distinct_count: usize,
    diagnostics: Vec<Diag>,
}

/// Parses `cargo --message-format=json` stdout into [`Diag`]s. Each line is a
/// JSON object; the ones with `reason == "compiler-message"` carry a `message`
/// object we reduce. Summary notes with no span (`aborting due to …`, `For more
/// information …`) are dropped — they are noise, not problems to fix.
fn parse_cargo_diagnostics(stdout: &str) -> Vec<Diag> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(Value::as_str) != Some("compiler-message") {
            continue;
        }
        if let Some(diag) = parse_one_message(value.get("message").unwrap_or(&Value::Null)) {
            out.push(diag);
        }
    }
    out
}

/// Reduces one rustc `message` object to a [`Diag`], or `None` if it is a
/// content-free summary note.
fn parse_one_message(message: &Value) -> Option<Diag> {
    let level = message.get("level").and_then(Value::as_str)?.to_string();
    // Only genuine problems: rustc reports summaries and cross-references as
    // `failure-note`/`note` with no useful span.
    if level != "error" && level != "warning" {
        return None;
    }
    let text = message.get("message").and_then(Value::as_str)?.to_string();
    if text.starts_with("aborting due to") || text.starts_with("For more information") {
        return None;
    }
    let code = message
        .pointer("/code/code")
        .and_then(Value::as_str)
        .map(str::to_string);

    // The primary span (`is_primary`), else the first span, gives the location.
    let spans = message.get("spans").and_then(Value::as_array);
    let primary = spans.and_then(|s| {
        s.iter()
            .find(|sp| sp.get("is_primary").and_then(Value::as_bool) == Some(true))
            .or_else(|| s.first())
    });
    let file = primary
        .and_then(|sp| sp.get("file_name"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let line = primary.and_then(|sp| sp.get("line_start")).and_then(Value::as_u64).map(|n| n as u32);
    let column = primary.and_then(|sp| sp.get("column_start")).and_then(Value::as_u64).map(|n| n as u32);

    Some(Diag { level, code, message: text, file, line, column })
}

/// Deduplicates and sorts parsed diagnostics — the core token win. Two
/// diagnostics are "the same" when their `(level, code, message, file, line,
/// column)` all match, which collapses the identical diagnostic cargo emits
/// once per compiled target (lib + tests + bins) into a single entry. The
/// survivors are sorted by file, then line, then column.
fn compactify(mut raws: Vec<Diag>) -> CompactReport {
    let raw_count = raws.len();
    raws.sort_by(|a, b| {
        (&a.file, a.line, a.column, &a.level, &a.code, &a.message)
            .cmp(&(&b.file, b.line, b.column, &b.level, &b.code, &b.message))
    });
    raws.dedup();
    CompactReport { raw_count, distinct_count: raws.len(), diagnostics: raws }
}

/// Prints the compact human view: one line per distinct diagnostic, then the
/// collapse ratio.
fn print_compact_diagnostics(report: &CompactReport) {
    for d in &report.diagnostics {
        let loc = match (&d.file, d.line, d.column) {
            (Some(f), Some(l), Some(c)) => format!("{f}:{l}:{c}"),
            (Some(f), Some(l), None) => format!("{f}:{l}"),
            (Some(f), _, _) => f.clone(),
            _ => "<no location>".to_string(),
        };
        let code = d.code.as_deref().map(|c| format!("[{c}] ")).unwrap_or_default();
        println!("{loc}: {}: {code}{}", d.level, d.message);
    }
    println!(
        "{} distinct diagnostic(s) from {} raw",
        report.distinct_count, report.raw_count
    );
}

// ---------------------------------------------------------------------------
// `test`
// ---------------------------------------------------------------------------

/// Runs `kopitiam test`.
pub fn run_test(args: TestArgs) -> Result<()> {
    if !args.compact && !args.json {
        let status = base_command("test", &args.root, args.package.as_deref())
            .status()
            .context("running `cargo test`")?;
        std::process::exit(status.code().unwrap_or(1));
    }

    let mut cmd = base_command("test", &args.root, args.package.as_deref());
    // Keep going after the first failing test binary so every failure is seen.
    cmd.arg("--no-fail-fast");
    let output = cmd.output().context("running `cargo test`")?;
    // libtest writes results to stdout; capture both streams to be safe.
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    let failures = parse_test_failures(&combined);
    let report = TestReport { failure_count: failures.len(), failures };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report.failures.is_empty() {
        println!("no test failures");
    } else {
        for f in &report.failures {
            let loc = f.location.as_deref().unwrap_or("<unknown location>");
            let assertion = f.assertion.as_deref().unwrap_or("(no message captured)");
            println!("{} — {assertion} @ {loc}", f.name);
        }
        println!("{} test failure(s)", report.failure_count);
    }
    Ok(())
}

/// One failed test, reduced to its name, the panic/assertion message, and the
/// `file:line:col` it fired at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TestFailure {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    assertion: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TestReport {
    failure_count: usize,
    failures: Vec<TestFailure>,
}

/// Parses libtest's human output into [`TestFailure`]s. It reads the per-failure
/// detail blocks that follow a `---- <name> stdout ----` header, pulling the
/// `panicked at <file>:<line>:<col>:` location and the message line after it.
/// (This is the stable-toolchain path; JSON test output is nightly-only.)
fn parse_test_failures(output: &str) -> Vec<TestFailure> {
    let lines: Vec<&str> = output.lines().collect();
    let mut failures = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim_end();
        // A detail block header: "---- <name> stdout ----".
        if let Some(name) = line
            .strip_prefix("---- ")
            .and_then(|rest| rest.strip_suffix(" stdout ----"))
        {
            let (location, assertion) = scan_detail_block(&lines, i + 1);
            failures.push(TestFailure { name: name.trim().to_string(), location, assertion });
        }
        i += 1;
    }
    failures
}

/// Scans the lines of one failure detail block (until the next `----` header or
/// the `failures:` summary) for a `panicked at` location and its message.
fn scan_detail_block(lines: &[&str], start: usize) -> (Option<String>, Option<String>) {
    let mut location = None;
    let mut assertion = None;
    let mut j = start;
    while j < lines.len() {
        let l = lines[j].trim();
        if l.starts_with("---- ") || l == "failures:" {
            break;
        }
        if let Some((loc, inline_msg)) = parse_panic_line(l) {
            location = Some(loc);
            // Modern rustc puts the message on the *next* line; older rustc
            // inlines it as `panicked at 'msg', file:line`.
            assertion = inline_msg.or_else(|| {
                lines
                    .get(j + 1)
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            });
            break;
        }
        j += 1;
    }
    (location, assertion)
}

/// Parses a `thread '…' panicked at …` line into `(location, inline_message?)`,
/// handling both the modern form (`panicked at src/x.rs:1:2:`, message on the
/// following line) and the legacy form (`panicked at 'msg', src/x.rs:1:2`).
fn parse_panic_line(line: &str) -> Option<(String, Option<String>)> {
    let after = line.split("panicked at ").nth(1)?;
    // Legacy: starts with a quoted message, then ", <location>".
    if let Some(rest) = after.strip_prefix('\'')
        && let Some((msg, loc)) = rest.rsplit_once("', ")
    {
        return Some((loc.trim().to_string(), Some(msg.to_string())));
    }
    // Modern: "<location>:" with the message on the next line.
    let loc = after.trim().trim_end_matches(':').to_string();
    if loc.is_empty() {
        return None;
    }
    Some((loc, None))
}

// ---------------------------------------------------------------------------
// shared
// ---------------------------------------------------------------------------

/// Builds `cargo <subcommand>` in `root`, optionally scoped to one package.
fn base_command(subcommand: &str, root: &std::path::Path, package: Option<&str>) -> Command {
    let mut cmd = Command::new("cargo");
    cmd.arg(subcommand).current_dir(root);
    if let Some(pkg) = package {
        cmd.arg("-p").arg(pkg);
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    #[derive(Parser)]
    struct CheckCli {
        #[command(flatten)]
        args: CheckArgs,
    }
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: TestArgs,
    }

    // ---- arg parsing --------------------------------------------------------

    #[test]
    fn check_args_parse() {
        let cli = CheckCli::try_parse_from(["t", "--compact", "-p", "kopitiam"]).unwrap();
        assert!(cli.args.compact);
        assert_eq!(cli.args.package.as_deref(), Some("kopitiam"));
        assert_eq!(cli.args.root, PathBuf::from("."));
    }

    #[test]
    fn test_args_parse_json() {
        let cli = TestCli::try_parse_from(["t", "--json", "--root", "/w"]).unwrap();
        assert!(cli.args.json);
        assert_eq!(cli.args.root, PathBuf::from("/w"));
    }

    // ---- cargo JSON parsing + dedup ----------------------------------------

    fn compiler_message(level: &str, code: &str, msg: &str, file: &str, line: u64) -> String {
        json!({
            "reason": "compiler-message",
            "message": {
                "level": level,
                "message": msg,
                "code": { "code": code },
                "spans": [ { "is_primary": true, "file_name": file, "line_start": line, "column_start": 5 } ],
            }
        })
        .to_string()
    }

    #[test]
    fn parse_cargo_diagnostics_reads_level_code_and_primary_span() {
        let stdout = compiler_message("error", "E0308", "mismatched types", "src/a.rs", 12);
        let diags = parse_cargo_diagnostics(&stdout);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, "error");
        assert_eq!(diags[0].code.as_deref(), Some("E0308"));
        assert_eq!(diags[0].message, "mismatched types");
        assert_eq!(diags[0].file.as_deref(), Some("src/a.rs"));
        assert_eq!(diags[0].line, Some(12));
        assert_eq!(diags[0].column, Some(5));
    }

    #[test]
    fn parse_cargo_diagnostics_drops_summary_notes_and_non_messages() {
        let mut stdout = String::new();
        // A build-script / artifact line (not a compiler-message) is ignored.
        stdout.push_str(&json!({ "reason": "compiler-artifact" }).to_string());
        stdout.push('\n');
        // A summary "aborting due to" note is dropped.
        stdout.push_str(
            &json!({ "reason": "compiler-message", "message": { "level": "error", "message": "aborting due to 3 previous errors", "spans": [] } }).to_string(),
        );
        stdout.push('\n');
        // A `failure-note` level is dropped.
        stdout.push_str(
            &json!({ "reason": "compiler-message", "message": { "level": "failure-note", "message": "For more information about this error, try `rustc --explain E0308`.", "spans": [] } }).to_string(),
        );
        stdout.push('\n');
        assert!(parse_cargo_diagnostics(&stdout).is_empty());
    }

    #[test]
    fn compactify_collapses_the_same_diagnostic_repeated_across_targets() {
        // The identical E0277 emitted for the lib, its test target, and a bin —
        // the classic "one bad type -> N copies" the dedup exists to defeat.
        let one = compiler_message("error", "E0277", "the trait bound is not satisfied", "src/lib.rs", 40);
        let stdout = format!("{one}\n{one}\n{one}");
        let raws = parse_cargo_diagnostics(&stdout);
        assert_eq!(raws.len(), 3, "three raw copies");
        let report = compactify(raws);
        assert_eq!(report.raw_count, 3);
        assert_eq!(report.distinct_count, 1, "collapsed to a single distinct fix");
        assert_eq!(report.diagnostics[0].code.as_deref(), Some("E0277"));
    }

    #[test]
    fn compactify_sorts_distinct_by_file_then_line() {
        let stdout = format!(
            "{}\n{}\n{}",
            compiler_message("warning", "unused", "unused import", "src/z.rs", 3),
            compiler_message("error", "E0308", "mismatched types", "src/a.rs", 50),
            compiler_message("error", "E0308", "mismatched types", "src/a.rs", 9),
        );
        let report = compactify(parse_cargo_diagnostics(&stdout));
        assert_eq!(report.distinct_count, 3);
        assert_eq!(report.diagnostics[0].file.as_deref(), Some("src/a.rs"));
        assert_eq!(report.diagnostics[0].line, Some(9), "a.rs:9 sorts before a.rs:50");
        assert_eq!(report.diagnostics[2].file.as_deref(), Some("src/z.rs"));
    }

    #[test]
    fn compact_report_json_shape() {
        let report = compactify(parse_cargo_diagnostics(&compiler_message("error", "E1", "boom", "a.rs", 1)));
        let json: Value = serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
        assert_eq!(json["raw_count"], 1);
        assert_eq!(json["distinct_count"], 1);
        assert_eq!(json["diagnostics"][0]["message"], "boom");
    }

    // ---- libtest failure parsing -------------------------------------------

    #[test]
    fn parse_test_failures_modern_panic_format() {
        // Rust >=1.72 puts the location on the `panicked at` line and the
        // assertion message on the following line.
        let output = "\
running 2 tests
test tests::ok ... ok
test tests::bad ... FAILED

failures:

---- tests::bad stdout ----
thread 'tests::bad' panicked at src/lib.rs:42:9:
assertion `left == right` failed
  left: 1
 right: 2

failures:
    tests::bad

test result: FAILED. 1 passed; 1 failed; 0 ignored;
";
        let failures = parse_test_failures(output);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].name, "tests::bad");
        assert_eq!(failures[0].location.as_deref(), Some("src/lib.rs:42:9"));
        assert_eq!(failures[0].assertion.as_deref(), Some("assertion `left == right` failed"));
    }

    #[test]
    fn parse_test_failures_legacy_panic_format() {
        // Older rustc inlines the message: panicked at 'msg', file:line:col.
        let output = "\
---- tests::old stdout ----
thread 'main' panicked at 'assertion failed: x > 0', src/old.rs:7:5

failures:
    tests::old
";
        let failures = parse_test_failures(output);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].name, "tests::old");
        assert_eq!(failures[0].location.as_deref(), Some("src/old.rs:7:5"));
        assert_eq!(failures[0].assertion.as_deref(), Some("assertion failed: x > 0"));
    }

    #[test]
    fn parse_test_failures_none_when_all_pass() {
        let output = "running 3 tests\ntest a ... ok\ntest b ... ok\ntest result: ok. 3 passed; 0 failed;\n";
        assert!(parse_test_failures(output).is_empty());
    }

    #[test]
    fn parse_panic_line_both_forms() {
        let (loc, msg) = parse_panic_line("thread 't' panicked at src/x.rs:1:2:").unwrap();
        assert_eq!(loc, "src/x.rs:1:2");
        assert_eq!(msg, None);

        let (loc, msg) = parse_panic_line("thread 'main' panicked at 'boom', src/y.rs:3:4").unwrap();
        assert_eq!(loc, "src/y.rs:3:4");
        assert_eq!(msg.as_deref(), Some("boom"));

        assert!(parse_panic_line("just a normal line").is_none());
    }
}
