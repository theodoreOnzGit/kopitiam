use std::path::Path;

use anyhow::{Result, bail};
use kopitiam_ai::{CompletionResponse, ModelAdapter};
use kopitiam_knowledge::SemanticGraph;
use kopitiam_workspace::ProjectState;

use crate::{ContextBuilder, Workflow};

/// Runs `CLAUDE.md`'s full pipeline for one [`Workflow`]: `load state ->
/// collect facts -> build context -> invoke model -> validate -> persist`.
///
/// `graph` is supplied by the caller (typically freshly rebuilt by
/// `kopitiam-semantic` providers, e.g. via `kopitiam scan`) rather than
/// loaded here, per the runtime's "reproducible, not synchronized" index
/// rule — this function only owns the *session-memory* half of "load
/// state" (`kopitiam_workspace::ProjectState`, which genuinely persists).
///
/// "Validate" is intentionally minimal in this scaffold: it only rejects
/// an empty response, so a broken adapter cannot silently mark a workflow
/// complete. Workflow-specific validation (e.g. `verify` checking a
/// translated unit's tests still pass) belongs in that workflow's own
/// [`Workflow::run`] override, not here.
pub fn run_workflow(
    root: &Path,
    graph: &SemanticGraph,
    workflow: &dyn Workflow,
    adapter: &dyn ModelAdapter,
) -> Result<CompletionResponse> {
    let state = ProjectState::load(root)?;
    let context = ContextBuilder::new(graph, state).build();

    let response = workflow.run(&context, adapter)?;
    if response.content.trim().is_empty() {
        bail!("{} workflow produced an empty response from adapter {:?}", workflow.kind().name(), adapter.name());
    }

    let mut state = ProjectState::load(root)?;
    state.touch(workflow.kind().name());
    state.save(root)?;

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_ai::EchoAdapter;
    use kopitiam_ontology::{Entity, EntityKind};

    use crate::{NamedWorkflow, WorkflowKind};

    #[test]
    fn runs_the_pipeline_and_persists_the_workflow_name_into_the_working_set() {
        let dir = tempfile::tempdir().unwrap();

        let mut state = ProjectState::load(dir.path()).unwrap();
        state.set_current_task("document the pipeline");
        state.save(dir.path()).unwrap();

        let mut graph = SemanticGraph::new();
        graph.insert_entity(Entity::new(EntityKind::Decision, "use redb", "test"));

        let workflow = NamedWorkflow::new(WorkflowKind::Document);
        let response = run_workflow(dir.path(), &graph, &workflow, &EchoAdapter).unwrap();
        assert!(response.content.contains("document the pipeline"));

        let reloaded = ProjectState::load(dir.path()).unwrap();
        assert!(reloaded.working_set.contains(&"document".to_string()));
        assert_eq!(reloaded.current_task.as_deref(), Some("document the pipeline"));
    }

    #[test]
    fn rejects_an_empty_response_without_persisting() {
        struct EmptyAdapter;
        impl ModelAdapter for EmptyAdapter {
            fn name(&self) -> &str {
                "empty"
            }
            fn complete(&self, _request: &kopitiam_ai::CompletionRequest) -> Result<CompletionResponse> {
                Ok(CompletionResponse { content: String::new(), model: "empty".to_string() })
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let graph = SemanticGraph::new();
        let workflow = NamedWorkflow::new(WorkflowKind::Plan);

        assert!(run_workflow(dir.path(), &graph, &workflow, &EmptyAdapter).is_err());

        let reloaded = ProjectState::load(dir.path()).unwrap();
        assert!(reloaded.working_set.is_empty());
    }
}
