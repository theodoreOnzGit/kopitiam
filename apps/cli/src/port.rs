//! The `port` subcommand group: `port status` and `port skeleton` — token-max
//! Tasks III-1 and III-3 (`kopitiam_token_max.md` §12).
//!
//! These are **thin wrappers over committed deterministic scripts** (§0.3), not
//! reimplementations. The scripts are the source of truth:
//!
//! * `scripts/port-ledger.sh` (III-1) derives the machine-maintained port ledger
//!   — source file → target → status (`unported`/`in-progress`/`ported`/
//!   `deliberately-diverged`) — as a pure function of the tree's provenance
//!   headers, writing `docs/port-ledger.{json,md}`. `port status` surfaces its
//!   `--report`; `port status --json` reads `docs/port-ledger.json` directly.
//! * `scripts/skeleton-gen.sh <file>` (III-3) mechanically emits Rust signature
//!   stubs from a vendored source file so a porter writes bodies only.
//!   `port skeleton <file>` surfaces its stdout.
//!
//! The scripts are located relative to the repo root (walked up from `--root`),
//! and the command **degrades gracefully** when a script or the ledger is not
//! found — a clear, actionable message rather than a crash.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

/// Options for `kopitiam port`.
#[derive(Args, Debug)]
pub struct PortArgs {
    #[command(subcommand)]
    action: PortAction,
}

/// The two `port` actions.
#[derive(Subcommand, Debug)]
enum PortAction {
    /// Show the port ledger (`scripts/port-ledger.sh --report`), or the raw
    /// `docs/port-ledger.json` with `--json`.
    Status(StatusArgs),
    /// Generate Rust signature stubs for a vendored source file
    /// (`scripts/skeleton-gen.sh <file>`).
    Skeleton(SkeletonArgs),
}

/// Options for `kopitiam port status`.
#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Where to start looking for the kopitiam repo root (the directory holding
    /// `scripts/port-ledger.sh`). Defaults to the current directory; the root is
    /// found by walking up from here.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Print `docs/port-ledger.json` directly (the machine view), instead of
    /// shelling the script's human `--report`. No subprocess is spawned.
    #[arg(long)]
    pub json: bool,
}

/// Options for `kopitiam port skeleton`.
#[derive(Args, Debug)]
pub struct SkeletonArgs {
    /// The vendored source file to generate signature stubs for.
    pub file: PathBuf,

    /// Where to start looking for the repo root (holding
    /// `scripts/skeleton-gen.sh`). Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Pass `--report` to the script: print its found/skipped breakdown instead
    /// of the stubs.
    #[arg(long)]
    pub report: bool,
}

/// Runs `kopitiam port ...`.
pub fn run(args: PortArgs) -> Result<()> {
    match args.action {
        PortAction::Status(a) => run_status(a),
        PortAction::Skeleton(a) => run_skeleton(a),
    }
}

/// `port status`: the ledger report, or the raw JSON view.
fn run_status(args: StatusArgs) -> Result<()> {
    if args.json {
        let root = find_repo_root(&args.root);
        let json_path = match &root {
            Some(root) => root.join("docs/port-ledger.json"),
            None => args.root.join("docs/port-ledger.json"),
        };
        print!("{}", read_ledger_json(&json_path)?);
        return Ok(());
    }

    let script = locate_script(&args.root, "port-ledger.sh")?;
    let root = script.parent().and_then(Path::parent).unwrap_or(Path::new("."));
    let output = run_script(&script, &["--report"], root)?;
    print!("{output}");
    Ok(())
}

/// `port skeleton <file>`: the generated Rust stubs (or the `--report` view).
fn run_skeleton(args: SkeletonArgs) -> Result<()> {
    let script = locate_script(&args.root, "skeleton-gen.sh")?;
    let root = script.parent().and_then(Path::parent).unwrap_or(Path::new("."));

    // Pass the file as an absolute path when we can resolve it, so the script's
    // cwd (the repo root) does not change which file it reads. Rendered
    // bash-friendly (see `to_bash_path`).
    let file = std::fs::canonicalize(&args.file).unwrap_or_else(|_| args.file.clone());
    let file = to_bash_path(&file);

    let script_args: Vec<&str> = if args.report {
        vec!["--report", &file]
    } else {
        vec![&file]
    };
    let output = run_script(&script, &script_args, root)?;
    print!("{output}");
    Ok(())
}

/// Reads and validates `docs/port-ledger.json`, returning its content verbatim.
/// Factored out so the JSON view is testable without a repo checkout: it parses
/// the file to confirm it is JSON, then returns the exact bytes (structure, not
/// re-serialised prose).
fn read_ledger_json(path: &Path) -> Result<String> {
    let content = std::fs::read_to_string(path).with_context(|| {
        format!(
            "reading {} — has `scripts/port-ledger.sh` been run to generate it?",
            path.display()
        )
    })?;
    serde_json::from_str::<serde_json::Value>(&content)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    Ok(content)
}

/// Finds the kopitiam repo root by walking up from `start` until a directory
/// containing `scripts/port-ledger.sh` is found. `None` if none is found up to
/// the filesystem root.
fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let start = std::fs::canonicalize(start).ok()?;
    let mut current: Option<&Path> = Some(start.as_path());
    while let Some(dir) = current {
        if dir.join("scripts/port-ledger.sh").is_file() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

/// Resolves `scripts/<name>` from the repo root, degrading gracefully with an
/// actionable message when the repo root (and therefore the script) cannot be
/// located.
fn locate_script(start: &Path, name: &str) -> Result<PathBuf> {
    let root = find_repo_root(start).ok_or_else(|| {
        anyhow::anyhow!(
            "could not find the kopitiam repo (no `scripts/port-ledger.sh` above {}). \
             Run this from inside the kopitiam checkout, or pass --root <repo>.",
            start.display()
        )
    })?;
    let script = root.join("scripts").join(name);
    if !script.is_file() {
        bail!(
            "found the repo root at {} but {} is missing",
            root.display(),
            script.display()
        );
    }
    Ok(script)
}

/// Renders a path as a string a `bash` (e.g. Git Bash on Windows) argument
/// accepts: the Windows verbatim `\\?\` / `\\?\UNC\` prefix that
/// [`std::fs::canonicalize`] emits is stripped, and backslashes (which bash
/// would treat as escapes) become forward slashes. A plain POSIX path is
/// returned unchanged.
fn to_bash_path(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let stripped = raw
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .or_else(|| raw.strip_prefix(r"\\?\").map(str::to_string))
        .unwrap_or_else(|| raw.into_owned());
    stripped.replace('\\', "/")
}

/// Shells a `bash` script and returns its stdout, degrading with a clear message
/// when `bash` is unavailable or the script exits non-zero.
fn run_script(script: &Path, args: &[&str], cwd: &Path) -> Result<String> {
    let output = Command::new("bash")
        .arg(to_bash_path(script))
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| {
            format!(
                "running {} (is `bash` available on PATH?)",
                script.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "{} exited with {}: {}",
            script.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: PortAction,
    }

    #[test]
    fn parses_status_and_skeleton() {
        let status = TestCli::try_parse_from(["p", "status", "--json"]).unwrap();
        assert!(matches!(status.command, PortAction::Status(a) if a.json));

        let skeleton =
            TestCli::try_parse_from(["p", "skeleton", "src/foo.c", "--report"]).unwrap();
        assert!(matches!(skeleton.command, PortAction::Skeleton(a) if a.report && a.file == *Path::new("src/foo.c")));
    }

    #[test]
    fn find_repo_root_walks_up_to_the_scripts_dir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("scripts")).unwrap();
        std::fs::write(root.join("scripts/port-ledger.sh"), b"#!/bin/sh\n").unwrap();
        let nested = root.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();

        let found = find_repo_root(&nested).unwrap();
        // Canonicalize both sides (tempdirs go through symlinks on macOS).
        assert_eq!(found, std::fs::canonicalize(root).unwrap());
    }

    #[test]
    fn find_repo_root_none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_repo_root(dir.path()).is_none());
    }

    #[test]
    fn read_ledger_json_returns_content_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("port-ledger.json");
        let body = "{\n  \"schema\": \"kopitiam-port-ledger/v1\",\n  \"entries\": []\n}";
        std::fs::write(&path, body).unwrap();
        assert_eq!(read_ledger_json(&path).unwrap(), body);
    }

    #[test]
    fn read_ledger_json_rejects_non_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("port-ledger.json");
        std::fs::write(&path, b"not json at all").unwrap();
        assert!(read_ledger_json(&path).is_err());
    }

    #[test]
    fn to_bash_path_strips_verbatim_prefix_and_normalises_slashes() {
        // A Windows verbatim path becomes a bash-friendly forward-slashed path.
        assert_eq!(
            to_bash_path(Path::new(r"\\?\C:\a\b\script.sh")),
            "C:/a/b/script.sh"
        );
        // A UNC verbatim path keeps its network `\\host` shape (as `//host`).
        assert_eq!(
            to_bash_path(Path::new(r"\\?\UNC\host\share\x.sh")),
            "//host/share/x.sh"
        );
        // A plain POSIX path is unchanged.
        assert_eq!(to_bash_path(Path::new("/usr/bin/x.sh")), "/usr/bin/x.sh");
    }

    #[test]
    fn locate_script_degrades_when_repo_is_absent() {
        let dir = tempfile::tempdir().unwrap();
        let err = locate_script(dir.path(), "port-ledger.sh").unwrap_err();
        assert!(err.to_string().contains("could not find the kopitiam repo"));
    }
}
