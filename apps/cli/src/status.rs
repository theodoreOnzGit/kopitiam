//! The `status` subcommand: read back a project's session memory.
//!
//! This is the read side of the persistence [`crate::scan`] writes to. It
//! demonstrates the point of `.kopitiam`/`kopitiam-index`/`kopitiam-workspace`
//! existing at all: a *new* process, with no chat history and no in-memory
//! state, can still answer "what was this project last doing?" by reading
//! `.kopitiam/state.redb` instead of asking a model to guess.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;
use kopitiam_workspace::{ConclusionLog, ProjectState, SourceDrift};

/// Options for `kopitiam status`.
#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Directory containing the project's `.kopitiam` state directory.
    /// Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Instead of the session summary, list conclusions whose source files
    /// have drifted (content hash mismatch) and can no longer be trusted.
    #[arg(long)]
    pub stale: bool,
}

/// Runs `kopitiam status`: load and print the persisted [`ProjectState`],
/// or, with `--stale`, the stale-conclusion report.
pub fn run(args: StatusArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;
    if args.stale {
        print!("{}", stale_report(&root)?);
    } else {
        print!("{}", report(&root)?);
    }
    Ok(())
}

/// Build the same textual status report `run` prints, as a `String`.
///
/// Factored out so the TUI's Status view can render this into a scrollable
/// pane without capturing stdout (which would require platform-specific fd
/// tricks that break Android/Termux). `run` prints exactly what this returns.
pub fn report(root: &Path) -> Result<String> {
    let root = std::fs::canonicalize(root)?;
    let state = ProjectState::load(&root)?;
    let mut out = String::new();

    match &state.current_task {
        Some(task) => writeln!(out, "Current task: {task}")?,
        None => writeln!(out, "Current task: (none recorded)")?,
    }

    if state.working_set.is_empty() {
        writeln!(out, "Working set: (empty)")?;
    } else {
        writeln!(out, "Working set (most recent last):")?;
        for entry in &state.working_set {
            writeln!(out, "  {entry}")?;
        }
    }

    if let Some(updated_at) = state.updated_at {
        writeln!(out, "Last updated: {updated_at} (unix seconds)")?;
    }

    // A live conclusion is a fact a prior session paid to derive and that is
    // still valid (its sources have not drifted), so the current session can
    // trust it without re-deriving. See `kopitiam status --stale` for the
    // ones that have gone stale.
    let log = ConclusionLog::load(&root)?;
    let total = log.conclusions().len();
    let stale = log.stale_conclusions().len();
    let live = total - stale;
    if total == 0 {
        writeln!(out, "Conclusions: (none recorded)")?;
    } else if stale == 0 {
        writeln!(out, "Conclusions: {live} live")?;
    } else {
        writeln!(
            out,
            "Conclusions: {live} live, {stale} stale (run `status --stale`)"
        )?;
    }

    Ok(out)
}

/// Build the `--stale` report: every conclusion whose sources have drifted,
/// naming which source drifted and how. This is the read side of Task II-5's
/// "invalidate on content hash, never time" rule — a caller uses it to tell
/// fresh conclusions (safe to reuse) from stale ones (must be re-derived).
pub fn stale_report(root: &Path) -> Result<String> {
    let root = std::fs::canonicalize(root)?;
    let log = ConclusionLog::load(&root)?;
    let stale = log.stale_conclusions();
    let mut out = String::new();

    if stale.is_empty() {
        writeln!(
            out,
            "Stale conclusions: (none) — all {} recorded conclusion(s) are live.",
            log.conclusions().len()
        )?;
        return Ok(out);
    }

    writeln!(out, "Stale conclusions ({}):", stale.len())?;
    for entry in &stale {
        writeln!(out, "  - {}", entry.conclusion.text)?;
        for drift in &entry.drifted {
            match drift {
                SourceDrift::Changed { path, .. } => {
                    writeln!(out, "      changed: {}", path.display())?;
                }
                SourceDrift::Missing { path } => {
                    writeln!(out, "      missing: {}", path.display())?;
                }
            }
        }
    }

    Ok(out)
}
