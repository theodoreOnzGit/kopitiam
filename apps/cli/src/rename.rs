//! The `rename` subcommand: rust-analyzer-backed symbol rename.
//!
//! This is the first *write-capable* Semantic Runtime command — earlier
//! commands ([`crate::scan`]) only ever read facts. Renaming is safe by
//! default: without `--apply`, this prints a unified diff of what would
//! change and touches nothing on disk. Pass `--apply` once you have looked
//! at the diff and are happy with it.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use kopitiam_semantic::{RustAnalyzerSession, edit};

/// Options for `kopitiam rename`.
#[derive(Args, Debug)]
pub struct RenameArgs {
    /// The Rust source file containing the symbol to rename.
    pub file: PathBuf,

    /// 0-indexed line of the symbol's identifier.
    #[arg(long)]
    pub line: u32,

    /// 0-indexed character offset of the symbol's identifier, in Unicode
    /// scalar values (i.e. plain `chars()` indexing — count characters,
    /// not bytes or UTF-16 code units).
    #[arg(long)]
    pub character: u32,

    /// The new name for the symbol.
    #[arg(long)]
    pub new_name: String,

    /// Directory containing the workspace `Cargo.toml` that `file` belongs
    /// to. Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Write the computed changes to disk. Without this flag, `rename`
    /// only prints a preview diff and leaves every file untouched.
    #[arg(long)]
    pub apply: bool,
}

/// Runs `kopitiam rename`: spawn rust-analyzer, ask it to rename the symbol
/// at the given position, and either print a diff or write the result.
pub fn run(args: RenameArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;
    println!(
        "Starting rust-analyzer and waiting for it to index {}...",
        root.display()
    );
    let mut session = RustAnalyzerSession::connect(&root)?;

    let file_edits = session.rename(&args.file, args.line, args.character, &args.new_name)?;
    let _ = session.shutdown();

    if file_edits.is_empty() {
        println!("rust-analyzer returned no changes for this rename.");
        return Ok(());
    }

    if args.apply {
        edit::write_file_edits(&file_edits)?;
        println!("Applied rename to {} file(s):", file_edits.len());
        for file_edit in &file_edits {
            println!("  {}", file_edit.path.display());
        }
    } else {
        println!("{}", edit::diff(&file_edits));
        println!("(preview only; re-run with --apply to write these changes)");
    }

    Ok(())
}
