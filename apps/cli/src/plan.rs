//! The `plan` subcommand: the first slice of the Workflow Engine wired
//! into the CLI.
//!
//! This runs `CLAUDE.md`'s full pipeline end to end — `load state ->
//! collect facts -> build context -> invoke model -> validate -> persist`
//! — via `kopitiam-workflow::run_workflow`, the same way every future
//! workflow command (`implement`, `translate`, `review`, ...) will.
//!
//! The model adapter is no longer hardcoded: [`crate::adapter::select_adapter`]
//! picks a real on-CPU [`kopitiam_ai::LocalAdapter`] when a `.gguf` is present
//! on disk, and only falls back to `kopitiam_ai::EchoAdapter` (the
//! deterministic stub that echoes the assembled context back) when no local
//! weights are around. Either way the plumbing being proven is the same —
//! context assembly from a live `kopitiam-knowledge` graph and
//! `kopitiam-workspace` session memory, request rendering, response
//! validation, persistence — and either way it runs with zero network access.
//! See `apps/cli/src/adapter.rs` for how the local-vs-echo choice is made.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use kopitiam_knowledge::SemanticGraph;
use kopitiam_semantic::{CargoMetadataProvider, KnowledgeProvider, RustdocProvider};
use kopitiam_workflow::{NamedWorkflow, WorkflowKind, run_workflow};
use kopitiam_workspace::ProjectState;

use crate::adapter::select_adapter;

/// Options for `kopitiam plan`.
#[derive(Args, Debug)]
pub struct PlanArgs {
    /// Directory containing the workspace `Cargo.toml` to plan against.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// What to plan. Recorded as the project's current task
    /// (`kopitiam-workspace`) before the workflow runs, so the resulting
    /// context reflects it.
    pub task: String,
}

/// Runs `kopitiam plan`: collect facts, record `task`, run the `plan`
/// workflow, print its response.
pub fn run(args: PlanArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;

    let mut graph = SemanticGraph::new();
    let cargo_output = CargoMetadataProvider::new().collect(&root)?;
    graph.extend(cargo_output.entities, cargo_output.relationships);
    let rustdoc_output = RustdocProvider::new().collect(&root)?;
    graph.extend(rustdoc_output.entities, rustdoc_output.relationships);

    let mut state = ProjectState::load(&root)?;
    state.set_current_task(&args.task);
    state.save(&root)?;

    // Pick the real local model if one is on disk, else the echo stub. The
    // note tells the user which rung of the Offline-First pipeline answered
    // and, when it's the stub, how to get a real model. Printed to stderr so
    // piping `kopitiam plan`'s stdout stays clean (the response is the
    // product; this is commentary).
    let selected = select_adapter();
    eprintln!("{}", selected.notice());
    eprintln!();

    let workflow = NamedWorkflow::new(WorkflowKind::Plan);
    let response = run_workflow(&root, &graph, &workflow, selected.adapter())?;

    if !selected.is_local() {
        println!("(this is kopitiam_ai::EchoAdapter echoing back the exact context that");
        println!("would have been sent to a real model — see the note above to get one)");
        println!();
    }
    println!("{}", response.content);

    Ok(())
}
