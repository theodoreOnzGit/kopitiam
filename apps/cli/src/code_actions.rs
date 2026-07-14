//! The `code-actions` subcommand: list and apply rust-analyzer's quick
//! fixes and refactorings at a given position.
//!
//! Two steps, matching how an editor presents this: run without `--apply`
//! to see a numbered list of available actions (`Add missing impl`,
//! `Extract into function`, ...), then re-run with `--apply <INDEX>` to
//! execute one. Unlike [`crate::rename`], applying a code action writes
//! immediately rather than printing a diff first — picking a specific,
//! named action from the listing is already the deliberate step; there is
//! no second confirmation.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use kopitiam_semantic::{RustAnalyzerSession, edit};

/// Options for `kopitiam code-actions`.
#[derive(Args, Debug)]
pub struct CodeActionsArgs {
    /// The Rust source file to query for code actions.
    pub file: PathBuf,

    /// 0-indexed line to query.
    #[arg(long)]
    pub line: u32,

    /// 0-indexed character offset to query, in Unicode scalar values.
    #[arg(long)]
    pub character: u32,

    /// Directory containing the workspace `Cargo.toml` that `file` belongs
    /// to. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Apply the action at this index from the listing (0-based). Without
    /// this flag, the command only lists what is available.
    #[arg(long)]
    pub apply: Option<usize>,
}

/// Runs `kopitiam code-actions`: list available actions, or apply one.
pub fn run(args: CodeActionsArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;
    println!("Starting rust-analyzer and waiting for it to index {}...", root.display());
    let mut session = RustAnalyzerSession::connect(&root)?;

    let actions = session.code_actions(&args.file, args.line, args.character)?;

    let Some(index) = args.apply else {
        if actions.is_empty() {
            println!("No code actions available at {}:{}:{}.", args.file.display(), args.line, args.character);
        } else {
            println!("Available code actions:");
            for (i, action) in actions.iter().enumerate() {
                println!("  [{i}] {}", action.title);
            }
            println!();
            println!("Re-run with --apply <INDEX> to apply one.");
        }
        let _ = session.shutdown();
        return Ok(());
    };

    let action = actions
        .get(index)
        .with_context(|| format!("no code action at index {index} (there are {})", actions.len()))?;
    println!("Applying: {}", action.title);
    let file_edits = session.apply_code_action(action)?;
    let _ = session.shutdown();

    if file_edits.is_empty() {
        println!("Done (the server applied this action's edit directly).");
    } else {
        edit::write_file_edits(&file_edits)?;
        println!("Applied to {} file(s):", file_edits.len());
        for file_edit in &file_edits {
            println!("  {}", file_edit.path.display());
        }
    }

    Ok(())
}
