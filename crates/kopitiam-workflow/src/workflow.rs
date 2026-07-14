use anyhow::Result;
use kopitiam_ai::{CompletionRequest, CompletionResponse, Message, ModelAdapter};

use crate::Context;

/// The named workflows from `CLAUDE.md`'s Architecture table
/// (`kopitiam-workflow` "Defines the `plan`, `implement`, `translate`,
/// `review`, `summarize`, `verify`, `document`, `resume` workflows").
///
/// Each variant maps to one [`Self::system_preamble`] describing the
/// model's job for that workflow; the [`Context`] built by
/// [`crate::ContextBuilder`] supplies everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkflowKind {
    /// Propose an implementation plan for the current task.
    Plan,
    /// Carry out an already-agreed plan.
    Implement,
    /// Translate a legacy source unit into idiomatic Rust (see
    /// `kopitiam-translation`).
    Translate,
    /// Review a change for correctness, maintainability, and fit with the
    /// existing architecture.
    Review,
    /// Condense the current context into a shorter summary, e.g. for a
    /// [`kopitiam_ontology::EntityKind::Summary`] entity.
    Summarize,
    /// Check a translated or implemented unit against its original
    /// behavior/assumptions.
    Verify,
    /// Produce or update documentation for the current task.
    Document,
    /// Reconstruct what a new session should know to continue where a
    /// previous one left off, from persisted state alone.
    Resume,
}

impl WorkflowKind {
    /// Stable identifier, also used as the `kopitiam_workspace::ProjectState`
    /// working-set entry recorded by [`crate::run_workflow`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Implement => "implement",
            Self::Translate => "translate",
            Self::Review => "review",
            Self::Summarize => "summarize",
            Self::Verify => "verify",
            Self::Document => "document",
            Self::Resume => "resume",
        }
    }

    /// The system-level instruction seeding a model invocation for this
    /// workflow, prepended to the rendered [`Context`].
    pub fn system_preamble(self) -> &'static str {
        match self {
            Self::Plan => {
                "You are KOPITIAM's planning workflow. Propose an implementation plan for \
                 the current task, grounded only in the facts and decisions provided below."
            }
            Self::Implement => {
                "You are KOPITIAM's implementation workflow. Carry out the current task, \
                 following the facts, decisions, and Rust Guidelines already recorded for \
                 this project."
            }
            Self::Translate => {
                "You are KOPITIAM's translation workflow. Translate the legacy source unit \
                 named in the current task into idiomatic Rust, preserving scientific intent \
                 rather than syntax, per the Translation Philosophy."
            }
            Self::Review => {
                "You are KOPITIAM's review workflow. Review the current task's change for \
                 correctness, maintainability, and architectural fit, as a ten-year maintainer \
                 would."
            }
            Self::Summarize => {
                "You are KOPITIAM's summarization workflow. Condense the facts and decisions \
                 below into a short summary suitable for a Summary entity in the knowledge \
                 graph."
            }
            Self::Verify => {
                "You are KOPITIAM's verification workflow. Check whether the current task's \
                 output preserves the original behavior/assumptions recorded in the facts \
                 below."
            }
            Self::Document => {
                "You are KOPITIAM's documentation workflow. Produce or update documentation \
                 for the current task, consistent with the facts and decisions below."
            }
            Self::Resume => {
                "You are KOPITIAM's resume workflow. Given only the persisted state below \
                 (no chat history), reconstruct what the next session should know to continue \
                 this task."
            }
        }
    }
}

/// One step of `CLAUDE.md`'s pipeline (`... -> build context -> invoke
/// model -> validate -> ...`): turns a [`Context`] into a model request and
/// returns its response.
///
/// This trait is the only thing in `kopitiam-workflow` that touches
/// `kopitiam-ai` — see [`crate::run_workflow`] for the surrounding
/// load/build/validate/persist steps, which do not.
pub trait Workflow {
    fn kind(&self) -> WorkflowKind;

    /// Renders `context` behind [`WorkflowKind::system_preamble`] and
    /// sends it to `adapter`. Override only if a workflow needs more than
    /// one model turn or extra messages beyond the rendered context.
    fn run(&self, context: &Context, adapter: &dyn ModelAdapter) -> Result<CompletionResponse> {
        let request = CompletionRequest::new(render_messages(self.kind(), context));
        adapter.complete(&request)
    }
}

/// The default [`Workflow`] implementation for a [`WorkflowKind`], with no
/// behavior beyond [`Workflow::run`]'s default. Covers all eight named
/// workflows until any of them needs bespoke logic (e.g. `translate`
/// updating `kopitiam-translation`'s `TranslationState`), at which point
/// that one gets its own type implementing this trait directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamedWorkflow(WorkflowKind);

impl NamedWorkflow {
    pub fn new(kind: WorkflowKind) -> Self {
        Self(kind)
    }
}

impl Workflow for NamedWorkflow {
    fn kind(&self) -> WorkflowKind {
        self.0
    }
}

fn render_messages(kind: WorkflowKind, context: &Context) -> Vec<Message> {
    let mut messages = vec![Message::system(kind.system_preamble())];

    let mut user_prompt = String::new();
    if let Some(task) = &context.current_task {
        user_prompt.push_str("Current task: ");
        user_prompt.push_str(task);
        user_prompt.push('\n');
    }
    if !context.working_set.is_empty() {
        user_prompt.push_str("Working set: ");
        user_prompt.push_str(&context.working_set.join(", "));
        user_prompt.push('\n');
    }
    if !context.facts.is_empty() {
        user_prompt.push_str("Known facts:\n");
        for fact in &context.facts {
            user_prompt.push_str(&format!("- [{:?}] {} (source: {})\n", fact.kind, fact.name, fact.source));
        }
    }

    messages.push(Message::user(user_prompt));
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_ai::{EchoAdapter, Role};
    use kopitiam_ontology::{Entity, EntityKind};

    #[test]
    fn every_named_workflow_has_a_distinct_preamble() {
        let kinds = [
            WorkflowKind::Plan,
            WorkflowKind::Implement,
            WorkflowKind::Translate,
            WorkflowKind::Review,
            WorkflowKind::Summarize,
            WorkflowKind::Verify,
            WorkflowKind::Document,
            WorkflowKind::Resume,
        ];
        let preambles: Vec<&str> = kinds.iter().map(|k| k.system_preamble()).collect();
        for (i, a) in preambles.iter().enumerate() {
            for (j, b) in preambles.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "{:?} and {:?} share a preamble", kinds[i], kinds[j]);
                }
            }
        }
    }

    #[test]
    fn renders_context_into_a_system_and_user_message() {
        let context = Context {
            current_task: Some("scaffold kopitiam-workflow".to_string()),
            working_set: vec!["kopitiam-knowledge".to_string()],
            facts: vec![Entity::new(EntityKind::Decision, "use redb", "test")],
        };

        let messages = render_messages(WorkflowKind::Plan, &context);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert!(messages[1].content.contains("scaffold kopitiam-workflow"));
        assert!(messages[1].content.contains("kopitiam-knowledge"));
        assert!(messages[1].content.contains("use redb"));
    }

    #[test]
    fn runs_end_to_end_against_the_echo_adapter() {
        let context = Context {
            current_task: Some("say hi".to_string()),
            working_set: vec![],
            facts: vec![],
        };
        let response = NamedWorkflow::new(WorkflowKind::Plan).run(&context, &EchoAdapter).unwrap();
        assert!(response.content.contains("say hi"));
    }
}
