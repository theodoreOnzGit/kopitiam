//! The `status` subcommand: read back a project's session memory.
//!
//! This is the read side of the persistence [`crate::scan`] writes to. It
//! demonstrates the point of `.kopitiam`/`kopitiam-index`/`kopitiam-workspace`
//! existing at all: a *new* process, with no chat history and no in-memory
//! state, can still answer "what was this project last doing?" by reading
//! `.kopitiam/state.redb` instead of asking a model to guess.

use std::path::PathBuf;

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
    let state = ProjectState::load(&root)?;

    match &state.current_task {
        Some(task) => println!("Current task: {task}"),
        None => println!("Current task: (none recorded)"),
    }

    if state.working_set.is_empty() {
        println!("Working set: (empty)");
    } else {
        println!("Working set (most recent last):");
        for entry in &state.working_set {
            println!("  {entry}");
        }
    }

    if let Some(updated_at) = state.updated_at {
        println!("Last updated: {updated_at} (unix seconds)");
    }

    Ok(())
}
