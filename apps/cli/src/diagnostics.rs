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
//! # Profile selection — why `--release` is not optional sugar
//!
//! Both subcommands take `--release` and `--profile <NAME>`, and both feed the
//! **same** flag into the passthrough path *and* the `--compact`/`--json`
//! analysis path. This is a correctness matter, not convenience lah: cargo's
//! default is the `dev` profile, and dev vs release genuinely disagree about
//! what compiles and what passes —
//!
//! * `debug-assertions` / `overflow-checks` are **on** in `dev` and **off** in
//!   `release`, so an arithmetic overflow that panics a dev test just wraps
//!   quietly under release, and any `debug_assert!` simply never fires;
//! * optimisation reorders float work and changes timing, so precision- and
//!   timing-sensitive tests can pass one way and fail the other;
//! * `cfg(debug_assertions)` code paths compile only in `dev`, so a release
//!   build can fail to compile on code a dev check called clean.
//!
//! So asking a release-profile question and getting a dev-profile answer is a
//! **false all-clear**, not a rounding error. This workspace's own build rule is
//! release-only (see `CLAUDE.md`), which is exactly the case that made the gap
//! embarrassing — reported as GitHub issue #1 / bead `bd-7ab`.
//!
//! `--release` and `--profile` are mutually exclusive at the clap layer
//! ([`ProfileArg`] explains why we reject it ourselves instead of letting cargo
//! do it).
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

    /// Check with the `release` profile (`cargo check --release`) instead of
    /// cargo's default `dev`.
    ///
    /// Not the same question as a dev check hor: under `release`,
    /// `debug-assertions`/`overflow-checks` are off and `cfg(debug_assertions)`
    /// code disappears, so a release check can fail on code that a dev check
    /// waved through (and the other way round). In a release-only workspace —
    /// like this one, per `CLAUDE.md` — this is the flag you want, always.
    ///
    /// Conflicts with `--profile`: cargo itself refuses both together, so we
    /// reject it here with a clearer message.
    #[arg(long, conflicts_with = "profile")]
    pub release: bool,

    /// Check with a custom cargo profile (`cargo check --profile <NAME>`), e.g.
    /// a `[profile.bench]` or a project-defined `[profile.ci]`.
    ///
    /// `--profile release` is legal cargo and does the same thing as
    /// `--release`; use whichever reads better. `--profile dev` spells out the
    /// default explicitly. Conflicts with `--release`.
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// Everything after `--` goes straight to cargo, untouched.
    ///
    /// This is the escape hatch for whatever we have not wrapped: target
    /// selection, features, cross-compilation. The motivating case from the bug
    /// report is a lib-only check quietly skipping the examples and tests that
    /// must still compile for an Android target —
    /// `kopitiam check --release -- --all-targets --target aarch64-linux-android`.
    ///
    /// These land LAST in the argv, after our own flags; `append_passthrough`
    /// explains why that order is load-bearing.
    #[arg(last = true, value_name = "CARGO_ARGS")]
    pub cargo_args: Vec<String>,
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

    /// Run the suite with the `release` profile (`cargo test --release`)
    /// instead of cargo's default `dev`.
    ///
    /// A dev-green suite is **not** evidence that the release suite is green.
    /// Overflow checks and `debug_assert!` are compiled out under `release`, and
    /// optimisation shifts float results and timing, so the two profiles can
    /// honestly disagree about pass/fail. If your project builds release-only
    /// (this one does), a dev-only `kopitiam test` is a false all-clear.
    ///
    /// Conflicts with `--profile`.
    #[arg(long, conflicts_with = "profile")]
    pub release: bool,

    /// Run the suite with a custom cargo profile (`cargo test --profile
    /// <NAME>`), e.g. a `[profile.ci]` that keeps overflow checks on top of
    /// optimisation. Conflicts with `--release`.
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// Everything after `--` goes straight to cargo, untouched.
    ///
    /// Careful hor — these are CARGO's args, not libtest's. To reach the test
    /// harness you need a second `--`, exactly like bare cargo:
    /// `kopitiam test --release -- --all-features` hands `--all-features` to
    /// cargo, while `kopitiam test --release -- -- --nocapture` hands
    /// `--nocapture` to libtest.
    ///
    /// They land LAST in the argv so a user-supplied `--` cannot swallow our own
    /// flags; `append_passthrough` explains why.
    #[arg(last = true, value_name = "CARGO_ARGS")]
    pub cargo_args: Vec<String>,
}

// ---------------------------------------------------------------------------
// `check`
// ---------------------------------------------------------------------------

/// Runs `kopitiam check`.
pub fn run_check(args: CheckArgs) -> Result<()> {
    // Passthrough mode: no analysis requested, so just let cargo write to the
    // inherited stdout/stderr and return.
    if !args.compact && !args.json {
        let status = check_command(&args, Mode::Passthrough)
            .status()
            .context("running `cargo check`")?;
        std::process::exit(status.code().unwrap_or(1));
    }

    let output = check_command(&args, Mode::Analysis)
        .output()
        .context("running `cargo check --message-format=json`")?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    let raws = parse_cargo_diagnostics(&stdout);
    let report = compactify(raws);
    bail_if_cargo_broke(
        "cargo check",
        output.status.success(),
        output.status.code(),
        &output.stderr,
        report.distinct_count,
    )?;

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
        let status = test_command(&args, Mode::Passthrough)
            .status()
            .context("running `cargo test`")?;
        std::process::exit(status.code().unwrap_or(1));
    }

    let output = test_command(&args, Mode::Analysis)
        .output()
        .context("running `cargo test`")?;
    // libtest writes results to stdout; capture both streams to be safe.
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    let failures = parse_test_failures(&combined);
    let report = TestReport { failure_count: failures.len(), failures };
    bail_if_cargo_broke(
        "cargo test",
        output.status.success(),
        output.status.code(),
        &output.stderr,
        report.failure_count,
    )?;

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

/// Which cargo profile a run asks for.
///
/// Small enum instead of passing `(bool, Option<&str>)` around, so the three
/// states are exhaustive and the "both at once" state is simply not
/// representable. Cargo *does* reject `--release --profile foo` on its own, but
/// its message is about conflicting profile *sources* and lands after a process
/// spawn; clap's `conflicts_with` says it up front, in our own voice, before we
/// launch anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileArg<'a> {
    /// No flag at all — cargo's default `dev` profile. Debug assertions and
    /// overflow checks on, no optimisation.
    Dev,
    /// `--release`.
    Release,
    /// `--profile <NAME>` for a custom profile.
    Named(&'a str),
}

impl<'a> ProfileArg<'a> {
    /// Folds the two clap fields into one choice. `release` and `profile` are
    /// `conflicts_with` each other at the clap layer, so both-set never reaches
    /// here; if it somehow did (a caller building the struct by hand), the
    /// explicit `--profile` wins, matching "the more specific flag wins".
    fn resolve(release: bool, profile: Option<&'a str>) -> Self {
        match (release, profile) {
            (_, Some(name)) => ProfileArg::Named(name),
            (true, None) => ProfileArg::Release,
            (false, None) => ProfileArg::Dev,
        }
    }

    /// Appends the flag (if any) to a cargo invocation.
    fn apply(self, cmd: &mut Command) {
        match self {
            ProfileArg::Dev => {}
            ProfileArg::Release => {
                cmd.arg("--release");
            }
            ProfileArg::Named(name) => {
                cmd.arg("--profile").arg(name);
            }
        }
    }
}

impl CheckArgs {
    /// The profile this `check` run asks cargo for.
    fn profile_arg(&self) -> ProfileArg<'_> {
        ProfileArg::resolve(self.release, self.profile.as_deref())
    }
}

impl TestArgs {
    /// The profile this `test` run asks cargo for.
    fn profile_arg(&self) -> ProfileArg<'_> {
        ProfileArg::resolve(self.release, self.profile.as_deref())
    }
}

/// Builds `cargo <subcommand>` in `root`, optionally scoped to one package, on
/// the requested profile.
///
/// Every path — passthrough *and* the `--compact`/`--json` analysis — goes
/// through here, which is the point: a profile flag that only reached one of the
/// two output modes would be worse than no flag at all, because the answer would
/// silently depend on how you asked for it.
fn base_command(
    subcommand: &str,
    root: &std::path::Path,
    package: Option<&str>,
    profile: ProfileArg<'_>,
) -> Command {
    let mut cmd = Command::new("cargo");
    cmd.arg(subcommand).current_dir(root);
    if let Some(pkg) = package {
        cmd.arg("-p").arg(pkg);
    }
    profile.apply(&mut cmd);
    cmd
}

/// Appends the user's trailing `-- …` args to a finished cargo invocation.
///
/// **Must be the very last thing appended, no exception.** A user is allowed to
/// pass a bare `--` inside these args (that is how `kopitiam test -- --
/// --nocapture` reaches libtest), and everything after that `--` belongs to the
/// test harness, not cargo. Append our own flags afterwards and they would land
/// on the wrong side of that separator — `cargo test -- --nocapture
/// --no-fail-fast` hands `--no-fail-fast` to libtest, which does not know it.
/// Last means last.
fn append_passthrough(cmd: &mut Command, extra: &[String]) {
    cmd.args(extra);
}

/// How many lines of cargo's stderr we quote when cargo itself dies. Enough to
/// carry `error: profile `nosuch` is not defined` plus its context, short enough
/// that we are still a token-saving tool.
const CARGO_ERROR_TAIL_LINES: usize = 12;

/// Refuses to print a green-looking summary when cargo itself fell over.
///
/// The analysis paths capture cargo's stdout and reduce it, which means a cargo
/// that never got as far as compiling — profile name typo, unknown package, no
/// `Cargo.toml` under `--root`, a `-- <args>` cargo rejects — produces an empty
/// capture. Reduce an empty capture and you get `0 distinct diagnostic(s)` or
/// `no test failures`, exit 0: a **false all-clear**, the exact failure mode
/// this whole issue is about, just arriving by a different door. `--profile`
/// makes it a one-typo mistake, so it gets shut here.
///
/// The condition is deliberately narrow — cargo exited non-zero **and** we have
/// nothing at all to show. A build that fails with real diagnostics, or a suite
/// with real failures, both exit non-zero too and are reported as before; only
/// the "non-zero and silent" case bails, because that can only mean the tool
/// never ran the thing you asked about. `reported` is how many findings we are
/// about to print.
///
/// Takes the exit status in pieces rather than a `&std::process::Output`
/// because `ExitStatus` has no portable constructor — the unit tests would have
/// to reach for `std::os::unix::process::ExitStatusExt`, and this workspace also
/// builds for Windows and Termux.
fn bail_if_cargo_broke(
    what: &str,
    success: bool,
    code: Option<i32>,
    stderr_bytes: &[u8],
    reported: usize,
) -> Result<()> {
    if success || reported > 0 {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(stderr_bytes);
    let mut tail: Vec<&str> = stderr
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(CARGO_ERROR_TAIL_LINES)
        .collect();
    tail.reverse();
    let detail = if tail.is_empty() {
        "(cargo printed nothing to stderr)".to_string()
    } else {
        tail.join("\n")
    };
    let code = code.map_or_else(|| "signal".to_string(), |c| c.to_string());
    anyhow::bail!("`{what}` failed (exit {code}) without producing anything to report:\n{detail}")
}

/// Which of the two output modes a subcommand is running in.
///
/// The two modes build *different* cargo invocations (the analysis one asks for
/// machine-readable output and, for `check`, all targets), and that difference
/// is precisely where a flag can go missing from one path only. Naming the mode
/// keeps both invocations in one function each, so "does `--release` reach this
/// path?" has exactly one answer per subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// No `--compact`/`--json`: cargo's own output streams straight through.
    Passthrough,
    /// `--compact` or `--json`: we capture and reduce cargo's output.
    Analysis,
}

/// Builds the exact `cargo check` invocation for a run — the single source of
/// truth for both modes, and the thing the argv tests assert on.
fn check_command(args: &CheckArgs, mode: Mode) -> Command {
    let mut cmd = base_command("check", &args.root, args.package.as_deref(), args.profile_arg());
    if mode == Mode::Analysis {
        // `--all-targets` checks the lib, tests, bins, benches, and examples —
        // the realistic "check everything" pass. It is also exactly where cargo
        // emits the *same* source diagnostic once per target (a lib error shows
        // up again when the test target recompiles the same file), which is the
        // duplication `compactify` collapses.
        cmd.arg("--all-targets").arg("--message-format=json");
    }
    append_passthrough(&mut cmd, &args.cargo_args);
    cmd
}

/// Builds the exact `cargo test` invocation for a run — same deal as
/// [`check_command`].
fn test_command(args: &TestArgs, mode: Mode) -> Command {
    let mut cmd = base_command("test", &args.root, args.package.as_deref(), args.profile_arg());
    if mode == Mode::Analysis {
        // Keep going after the first failing test binary so every failure is
        // seen, not just the first crate's worth.
        cmd.arg("--no-fail-fast");
    }
    append_passthrough(&mut cmd, &args.cargo_args);
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    // `Debug` so `unwrap_err()` can report the *parsed* value when a conflict we
    // expect to be rejected is quietly accepted instead.
    #[derive(Parser, Debug)]
    struct CheckCli {
        #[command(flatten)]
        args: CheckArgs,
    }
    #[derive(Parser, Debug)]
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

    // ---- profile selection: parsing ----------------------------------------

    #[test]
    fn check_args_parse_release() {
        let cli = CheckCli::try_parse_from(["t", "--release"]).unwrap();
        assert!(cli.args.release);
        assert_eq!(cli.args.profile, None);
        assert_eq!(cli.args.profile_arg(), ProfileArg::Release);
    }

    #[test]
    fn test_args_parse_release() {
        let cli = TestCli::try_parse_from(["t", "--release", "--compact"]).unwrap();
        assert!(cli.args.release);
        assert_eq!(cli.args.profile_arg(), ProfileArg::Release);
    }

    #[test]
    fn check_args_parse_named_profile() {
        let cli = CheckCli::try_parse_from(["t", "--profile", "ci"]).unwrap();
        assert!(!cli.args.release);
        assert_eq!(cli.args.profile.as_deref(), Some("ci"));
        assert_eq!(cli.args.profile_arg(), ProfileArg::Named("ci"));
    }

    #[test]
    fn test_args_parse_named_profile() {
        let cli = TestCli::try_parse_from(["t", "--profile", "bench"]).unwrap();
        assert_eq!(cli.args.profile_arg(), ProfileArg::Named("bench"));
    }

    #[test]
    fn no_profile_flag_means_dev() {
        let cli = CheckCli::try_parse_from(["t"]).unwrap();
        assert_eq!(cli.args.profile_arg(), ProfileArg::Dev);
        let cli = TestCli::try_parse_from(["t"]).unwrap();
        assert_eq!(cli.args.profile_arg(), ProfileArg::Dev);
    }

    #[test]
    fn release_and_profile_together_are_rejected() {
        // Cargo would reject this too, but only after we spawn it and only with
        // a message about profile sources. clap says it up front.
        let err = CheckCli::try_parse_from(["t", "--release", "--profile", "foo"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);

        let err = TestCli::try_parse_from(["t", "--release", "--profile", "foo"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn trailing_cargo_args_parse_after_double_dash() {
        let cli = CheckCli::try_parse_from(["t", "--release", "--", "--all-targets", "--target", "aarch64-linux-android"]).unwrap();
        assert!(cli.args.release);
        assert_eq!(cli.args.cargo_args, ["--all-targets", "--target", "aarch64-linux-android"]);

        // A second `--` survives verbatim, which is how libtest args get through
        // `kopitiam test -- -- --nocapture`.
        let cli = TestCli::try_parse_from(["t", "--", "--", "--nocapture"]).unwrap();
        assert_eq!(cli.args.cargo_args, ["--", "--nocapture"]);
    }

    // ---- argv construction --------------------------------------------------
    //
    // Spawning cargo is untested (see the module docs), so the argv is the part
    // that can silently break: a flag that parses fine but never reaches the
    // child process looks exactly like a working flag from the outside.

    /// The child process's args, minus the program name, as plain strings.
    fn argv(cmd: &Command) -> Vec<String> {
        cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    fn check_argv(args: &[&str], mode: Mode) -> Vec<String> {
        let cli = CheckCli::try_parse_from(std::iter::once("t").chain(args.iter().copied())).unwrap();
        argv(&check_command(&cli.args, mode))
    }

    fn test_argv(args: &[&str], mode: Mode) -> Vec<String> {
        let cli = TestCli::try_parse_from(std::iter::once("t").chain(args.iter().copied())).unwrap();
        argv(&test_command(&cli.args, mode))
    }

    #[test]
    fn check_argv_default_profile_passes_no_profile_flag() {
        let av = check_argv(&[], Mode::Passthrough);
        assert_eq!(av, ["check"]);
        assert!(!av.iter().any(|a| a == "--release" || a == "--profile"));
    }

    #[test]
    fn check_release_reaches_both_paths() {
        // The whole bug: a profile flag honoured in one output mode only.
        assert_eq!(check_argv(&["--release"], Mode::Passthrough), ["check", "--release"]);
        assert_eq!(
            check_argv(&["--release", "--compact"], Mode::Analysis),
            ["check", "--release", "--all-targets", "--message-format=json"]
        );
    }

    #[test]
    fn test_release_reaches_both_paths() {
        assert_eq!(test_argv(&["--release"], Mode::Passthrough), ["test", "--release"]);
        assert_eq!(
            test_argv(&["--release", "--compact"], Mode::Analysis),
            ["test", "--release", "--no-fail-fast"]
        );
    }

    #[test]
    fn named_profile_reaches_both_paths_as_two_argv_entries() {
        // `--profile` takes a value, so it must be two argv entries, not one
        // "--profile ci" string that cargo would read as an unknown flag.
        assert_eq!(
            check_argv(&["--profile", "ci"], Mode::Passthrough),
            ["check", "--profile", "ci"]
        );
        assert_eq!(
            check_argv(&["--profile", "ci", "--json"], Mode::Analysis),
            ["check", "--profile", "ci", "--all-targets", "--message-format=json"]
        );
        assert_eq!(
            test_argv(&["--profile", "ci", "--json"], Mode::Analysis),
            ["test", "--profile", "ci", "--no-fail-fast"]
        );
    }

    #[test]
    fn package_and_profile_both_reach_the_argv() {
        assert_eq!(
            check_argv(&["-p", "kopi-beans", "--release", "--compact"], Mode::Analysis),
            ["check", "-p", "kopi-beans", "--release", "--all-targets", "--message-format=json"]
        );
    }

    #[test]
    fn passthrough_args_land_last_in_both_paths() {
        // Last matters: our own flags must never end up after a user's `--`.
        assert_eq!(
            check_argv(&["--release", "--", "--all-targets", "--target", "aarch64-linux-android"], Mode::Passthrough),
            ["check", "--release", "--all-targets", "--target", "aarch64-linux-android"]
        );
        assert_eq!(
            check_argv(&["--compact", "--", "--features", "foo"], Mode::Analysis),
            ["check", "--all-targets", "--message-format=json", "--features", "foo"]
        );
        let av = test_argv(&["--compact", "--", "--", "--nocapture"], Mode::Analysis);
        assert_eq!(av, ["test", "--no-fail-fast", "--", "--nocapture"]);
        // `--no-fail-fast` stays on cargo's side of the user's separator.
        let sep = av.iter().position(|a| a == "--").unwrap();
        assert!(av[..sep].iter().any(|a| a == "--no-fail-fast"));
    }

    // ---- cargo blew up with nothing to report -------------------------------

    #[test]
    fn a_dead_cargo_with_no_findings_is_an_error_not_a_clean_bill() {
        // `--profile nosuch`: cargo never compiles anything, so the capture is
        // empty and the old code printed "0 distinct diagnostic(s)", exit 0.
        let err = bail_if_cargo_broke(
            "cargo check",
            false,
            Some(101),
            b"error: profile `nosuch` is not defined\n",
            0,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("exit 101"), "{msg}");
        assert!(msg.contains("profile `nosuch` is not defined"), "{msg}");
    }

    #[test]
    fn a_dead_cargo_that_did_report_findings_is_left_alone() {
        // A failed build with real diagnostics, or a suite with real failures:
        // non-zero exit is expected and the findings are the answer.
        assert!(bail_if_cargo_broke("cargo check", false, Some(101), b"error: aborting", 3).is_ok());
        assert!(bail_if_cargo_broke("cargo test", false, Some(101), b"", 1).is_ok());
    }

    #[test]
    fn a_happy_cargo_never_bails() {
        assert!(bail_if_cargo_broke("cargo check", true, Some(0), b"warning: whatever", 0).is_ok());
    }

    #[test]
    fn the_quoted_stderr_is_the_tail_and_stays_short() {
        let noisy: String = (0..100).map(|i| format!("line {i}\n")).collect();
        let err = bail_if_cargo_broke("cargo test", false, None, noisy.as_bytes(), 0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("exit signal"), "no exit code -> killed by a signal: {msg}");
        assert!(msg.contains("line 99"), "keeps the last line: {msg}");
        assert!(!msg.contains("line 87"), "drops everything older than the tail: {msg}");
        assert_eq!(
            msg.lines().count(),
            CARGO_ERROR_TAIL_LINES + 1,
            "the tail plus our one-line preamble, nothing more"
        );
    }

    #[test]
    fn root_becomes_the_child_working_directory() {
        let cli = CheckCli::try_parse_from(["t", "--root", "/w", "--release"]).unwrap();
        let cmd = check_command(&cli.args, Mode::Passthrough);
        assert_eq!(cmd.get_current_dir(), Some(std::path::Path::new("/w")));
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
