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
use kopitiam_workspace::ProjectState;

/// Options for `kopitiam status`.
#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Directory containing the project's `.kopitiam` state directory.
    /// Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,
}

/// Runs `kopitiam status`: load and print the persisted [`ProjectState`].
pub fn run(args: StatusArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;
    print!("{}", report(&root)?);
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

    Ok(out)
}
