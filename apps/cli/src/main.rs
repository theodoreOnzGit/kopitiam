//! `kopitiam`: the KOPITIAM command-line interface.
//!
//! This binary is the thin human-facing shell over KOPITIAM's engines — see
//! `CLAUDE.md`'s Architecture section ("Applications are clients. The
//! platform owns the functionality.") and its "Dogfood the Semantic Runtime
//! CLI" rule. Concretely that means:
//!
//! * every subcommand here should be a short call into a `kopitiam-*` crate,
//!   never a place where new business logic is invented;
//! * as the Semantic Runtime crates (`kopitiam-knowledge`, `kopitiam-index`,
//!   `kopitiam-search`, `kopitiam-workspace`, `kopitiam-workflow`, ...) grow
//!   capable of more, this file grows more subcommands to expose them;
//! * this binary is meant to become the actual tool used to keep developing
//!   KOPITIAM (`resume`, `plan`, `architecture`, `translation-status`, ...),
//!   not a demo that gets left behind once the underlying crates work.
//!
//! Subcommands so far: [`Command::Pdf2md`] turns a PDF into semantic
//! Markdown via the Document Engine. [`Command::Scan`] is the first,
//! read-only slice of the Semantic Runtime (see [`scan`]).
//! [`Command::Rename`] and [`Command::CodeActions`] are the first
//! write-capable slice, driving a live rust-analyzer over LSP to rename
//! symbols and apply refactorings (see [`rename`] and [`code_actions`]).

mod code_actions;
mod plan;
mod rename;
mod scan;
mod status;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

/// Top-level CLI parser. `clap` derives the actual argument parsing from
/// this struct and the [`Command`] enum below; this file only wires parsed
/// arguments to the function that does the real work.
#[derive(Parser)]
#[command(name = "kopitiam", about = "KOPITIAM command-line interface")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Every subcommand `kopitiam` currently supports.
///
/// Add new engines here as they mature — each variant should map to exactly
/// one call into a `kopitiam-*` library crate, with this file handling only
/// argument parsing, I/O, and user-facing output.
#[derive(Subcommand)]
enum Command {
    /// Convert a PDF into semantic Markdown.
    ///
    /// Runs the full Document Engine pipeline: `kopitiam-pdf` extracts text
    /// per page, `kopitiam-document` reconstructs paragraph/heading/table
    /// structure across page breaks and columns, and `kopitiam-markdown`
    /// renders the result. A validation report comparing extracted vs.
    /// rendered word counts is printed alongside the output, as a cheap
    /// sanity check that the reconstruction did not silently drop content.
    Pdf2md {
        /// Input PDF file.
        input: PathBuf,
        /// Output Markdown file. Defaults to the input path with a .md extension.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Scan a Rust project's real tooling and report what the Semantic
    /// Runtime learned about it.
    ///
    /// This is the first Semantic Runtime command: it runs the
    /// `kopitiam-semantic` knowledge providers (cargo metadata always,
    /// rust-analyzer optionally, rustdoc JSON when a nightly toolchain is
    /// available) against a project, merges everything they report into a
    /// `kopitiam-knowledge` graph, and prints a summary. See
    /// `apps/cli/src/scan.rs` for the full explanation of why this command
    /// exists and where it is headed.
    Scan(scan::ScanArgs),

    /// Rename a symbol using a live rust-analyzer, previewing the change
    /// as a diff unless `--apply` is given.
    ///
    /// See `apps/cli/src/rename.rs` for the full explanation, including why
    /// this is safe-by-default.
    Rename(rename::RenameArgs),

    /// List or apply rust-analyzer's code actions (quick fixes and
    /// refactorings) at a file position.
    ///
    /// See `apps/cli/src/code_actions.rs` for the full explanation.
    CodeActions(code_actions::CodeActionsArgs),

    /// Print this project's persisted session memory (`.kopitiam/state.redb`).
    ///
    /// See `apps/cli/src/status.rs`: this is the read side of the state
    /// `scan` writes, proving persistence survives across process restarts.
    Status(status::StatusArgs),

    /// Run the `plan` workflow: build context from a live scan plus
    /// session memory, and invoke a model adapter (currently
    /// `kopitiam_ai::EchoAdapter` — no production adapter is wired in
    /// yet).
    ///
    /// See `apps/cli/src/plan.rs`: the first `kopitiam-workflow` command,
    /// proving the full `load state -> collect facts -> build context ->
    /// invoke model -> validate -> persist` pipeline end to end.
    Plan(plan::PlanArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Pdf2md { input, output } => pdf2md(&input, output)?,
        Command::Scan(args) => scan::run(args)?,
        Command::Rename(args) => rename::run(args)?,
        Command::CodeActions(args) => code_actions::run(args)?,
        Command::Status(args) => status::run(args)?,
        Command::Plan(args) => plan::run(args)?,
    }
    Ok(())
}

/// Implements [`Command::Pdf2md`]: PDF in, semantic Markdown out, plus a
/// validation report printed to stdout.
fn pdf2md(input: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let pages = kopitiam_pdf::extract(input)?;
    let document = kopitiam_document::reconstruct(&pages);
    let markdown = kopitiam_markdown::render_document(&document);
    let report = kopitiam_document::validate(&pages, &document, &markdown);

    let output = output.unwrap_or_else(|| input.with_extension("md"));
    std::fs::write(&output, &markdown)?;

    println!("Wrote {}", output.display());
    println!();
    println!("{report}");

    Ok(())
}
