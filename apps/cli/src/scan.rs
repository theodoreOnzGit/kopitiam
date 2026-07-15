//! The `scan` subcommand: the first real slice of the Semantic Runtime
//! wired into the CLI.
//!
//! `scan` is deliberately simple: it runs the Rust [`KnowledgeProvider`]s
//! from `kopitiam-semantic` against a project root, merges everything they
//! report into a single [`SemanticGraph`] from `kopitiam-knowledge`, and
//! prints a summary. No model is invoked anywhere in this path — this
//! command exists purely to prove, end to end, that the runtime can turn
//! real tool output (`cargo metadata`, a live `rust-analyzer` process,
//! `rustdoc` JSON) into `kopitiam-ontology` facts and answer questions about
//! them. It also records that a scan happened in `kopitiam-workspace`'s
//! project state (`.kopitiam/state.redb`), so `kopitiam status` can report
//! it later — the first end-to-end use of session-memory persistence.
//!
//! As more of the Semantic Runtime lands (`kopitiam-search`,
//! `kopitiam-workflow`), this command — and the subcommands that follow it
//! (`resume`, `plan`, `architecture`, ...) — is meant to become KOPITIAM's
//! actual day-to-day interface, per the "Dogfood the Semantic Runtime CLI"
//! rule in `CLAUDE.md`. Prefer extending this file over reaching for
//! ad-hoc scripts.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;
use kopitiam_knowledge::SemanticGraph;
use kopitiam_ontology::EntityKind;
use kopitiam_semantic::{
    CargoMetadataProvider, KnowledgeProvider, RustAnalyzerProvider, RustdocProvider,
};
use kopitiam_workspace::ProjectState;

/// Options for `kopitiam scan`.
///
/// `scan` walks a project's real tooling (never source text directly) and
/// reports what it learned. Each flag below turns one provider on or off so
/// that a user can trade completeness for speed.
#[derive(Args, Debug)]
pub struct ScanArgs {
    /// Directory containing the workspace `Cargo.toml` to scan.
    ///
    /// Defaults to the current directory. This is passed straight through
    /// to `cargo metadata`, the `rust-analyzer` process, and `cargo rustdoc`,
    /// so it must be (or be inside) a real Cargo workspace.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Also query a live `rust-analyzer` process over LSP for symbols.
    ///
    /// This is off by default because it has to wait for rust-analyzer to
    /// finish indexing the workspace, which can take anywhere from a few
    /// seconds to a few minutes depending on workspace size. Turn it on when
    /// you specifically want symbol-level facts (function/struct/trait
    /// names and locations), not just the package-level facts `cargo
    /// metadata` already gives you for free.
    #[arg(long)]
    pub with_rust_analyzer: bool,

    /// Print every collected entity and relationship, not just the counts.
    #[arg(long)]
    pub verbose: bool,
}

/// Runs `kopitiam scan`: collect facts, merge them into a graph, report.
pub fn run(args: ScanArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;
    let mut graph = SemanticGraph::new();

    run_provider(&mut graph, &root, &CargoMetadataProvider::new())?;
    run_provider(&mut graph, &root, &RustdocProvider::new())?;
    if args.with_rust_analyzer {
        run_provider(&mut graph, &root, &RustAnalyzerProvider::new())?;
    }

    print_summary(&graph, args.verbose);

    let mut state = ProjectState::load(&root)?;
    state.touch("scan");
    state.save(&root)?;

    Ok(())
}

/// Runs one provider and merges its output into `graph`, printing which
/// provider ran and how many facts it contributed. Kept as a free function
/// (rather than a method) so adding a fourth provider later is a one-line
/// change in [`run`].
fn run_provider(
    graph: &mut SemanticGraph,
    root: &Path,
    provider: &dyn KnowledgeProvider,
) -> Result<()> {
    let output = provider.collect(root)?;
    let entity_delta = output.entities.len();
    let relationship_delta = output.relationships.len();
    graph.extend(output.entities, output.relationships);
    println!(
        "  {:<16} +{} entities, +{} relationships (graph now has {} entities)",
        provider.name(),
        entity_delta,
        relationship_delta,
        graph.entity_count(),
    );
    Ok(())
}

/// Prints entity/relationship counts by kind, and optionally every fact.
fn print_summary(graph: &SemanticGraph, verbose: bool) {
    println!();
    println!(
        "Semantic graph: {} entities, {} relationships",
        graph.entity_count(),
        graph.relationship_count()
    );

    for kind in [
        EntityKind::Artifact,
        EntityKind::Symbol,
        EntityKind::Section,
        EntityKind::Fact,
        EntityKind::Summary,
        EntityKind::Decision,
        EntityKind::Task,
    ] {
        let count = graph.entities_of_kind(kind).count();
        if count > 0 {
            println!("  {kind:?}: {count}");
        }
    }

    if verbose {
        println!();
        println!("Entities:");
        for entity in graph.entities() {
            println!(
                "  [{:?}] {} (source: {})",
                entity.kind, entity.name, entity.source
            );
        }
    }
}
