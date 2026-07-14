//! The `plan` subcommand: the first slice of the Workflow Engine wired
//! into the CLI.
//!
//! This runs `CLAUDE.md`'s full pipeline end to end — `load state ->
//! collect facts -> build context -> invoke model -> validate -> persist`
//! — via `kopitiam-workflow::run_workflow`, the same way every future
//! workflow command (`implement`, `translate`, `review`, ...) will. No
//! production model adapter exists yet, so this uses
//! `kopitiam_ai::EchoAdapter`: it proves the plumbing (context assembly
//! from a live `kopitiam-knowledge` graph and `kopitiam-workspace` session
//! memory, request rendering, response validation, persistence) works
//! without depending on network access or local model weights. Swapping in
//! a real adapter later is a one-line change here, not a redesign.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use kopitiam_ai::EchoAdapter;
use kopitiam_knowledge::SemanticGraph;
use kopitiam_semantic::{CargoMetadataProvider, KnowledgeProvider, RustdocProvider};
use kopitiam_workflow::{NamedWorkflow, WorkflowKind, run_workflow};
use kopitiam_workspace::ProjectState;

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

    let workflow = NamedWorkflow::new(WorkflowKind::Plan);
    let response = run_workflow(&root, &graph, &workflow, &EchoAdapter)?;

    println!("(no model adapter wired in yet — this is kopitiam_ai::EchoAdapter, echoing back");
    println!("the exact context that would have been sent to a real model)");
    println!();
    println!("{}", response.content);

    Ok(())
}
