use kopitiam_knowledge::SemanticGraph;
use kopitiam_ontology::{Entity, EntityKind};
use kopitiam_workspace::ProjectState;

/// Default cap on [`Context::facts`], so a large project's semantic graph
/// cannot silently balloon a model request. See [`ContextBuilder::with_max_facts`]
/// to override it.
pub const DEFAULT_MAX_FACTS: usize = 50;

/// Everything a [`crate::Workflow`] hands to a model: no more, no less.
///
/// This is the concrete answer to the AI Philosophy rule in `CLAUDE.md`
/// ("Never ask an AI model to rediscover information already present
/// inside KOPITIAM.") — a workflow only ever sees what a [`ContextBuilder`]
/// assembled from the runtime's own state, never a raw repository scan.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Context {
    /// What the project's session memory says is being worked on right
    /// now (`kopitiam_workspace::ProjectState::current_task`).
    pub current_task: Option<String>,
    /// Artifacts/symbols/documents recently touched
    /// (`kopitiam_workspace::ProjectState::working_set`).
    pub working_set: Vec<String>,
    /// Entities pulled from the semantic graph: everything in
    /// `working_set` that resolves to a known entity, followed by
    /// standing [`EntityKind::Fact`] and [`EntityKind::Decision`] entities,
    /// capped at the builder's `max_facts`.
    pub facts: Vec<Entity>,
}

/// Assembles a [`Context`] from `kopitiam-knowledge` (the semantic graph)
/// and `kopitiam-workspace` (session memory) — nothing else.
///
/// Deliberately has no dependency on `kopitiam-ai`: building a `Context`
/// must never require knowing how, or whether, it will be sent to a model.
/// Rendering a `Context` into a model request lives in [`crate::workflow`],
/// the one place in this crate (and the one crate in the whole platform)
/// allowed to touch `kopitiam-ai`.
pub struct ContextBuilder<'a> {
    graph: &'a SemanticGraph,
    state: ProjectState,
    max_facts: usize,
}

impl<'a> ContextBuilder<'a> {
    /// Builds context from `graph` (the current semantic graph — rebuilt
    /// fresh from tooling per the runtime's "reproducible, not
    /// synchronized" index rule, so it is always passed in rather than
    /// loaded here) and `state` (persisted session memory).
    pub fn new(graph: &'a SemanticGraph, state: ProjectState) -> Self {
        Self { graph, state, max_facts: DEFAULT_MAX_FACTS }
    }

    /// Overrides [`DEFAULT_MAX_FACTS`].
    pub fn with_max_facts(mut self, max_facts: usize) -> Self {
        self.max_facts = max_facts;
        self
    }

    pub fn build(&self) -> Context {
        let mut facts: Vec<Entity> = Vec::new();

        for name in &self.state.working_set {
            for entity in self.graph.entities().filter(|e| &e.name == name) {
                if !facts.iter().any(|f| f.id == entity.id) {
                    facts.push(entity.clone());
                }
            }
        }

        for entity in self
            .graph
            .entities_of_kind(EntityKind::Decision)
            .chain(self.graph.entities_of_kind(EntityKind::Fact))
        {
            if facts.len() >= self.max_facts {
                break;
            }
            if !facts.iter().any(|f| f.id == entity.id) {
                facts.push(entity.clone());
            }
        }

        facts.truncate(self.max_facts);

        Context {
            current_task: self.state.current_task.clone(),
            working_set: self.state.working_set.clone(),
            facts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_ontology::Relationship;

    fn entity(kind: EntityKind, name: &str) -> Entity {
        Entity::new(kind, name, "test")
    }

    #[test]
    fn carries_current_task_and_working_set_through_untouched() {
        let graph = SemanticGraph::new();
        let mut state = ProjectState::default();
        state.set_current_task("scaffold kopitiam-workflow");
        state.touch("kopitiam-knowledge");

        let context = ContextBuilder::new(&graph, state).build();
        assert_eq!(context.current_task.as_deref(), Some("scaffold kopitiam-workflow"));
        assert_eq!(context.working_set, vec!["kopitiam-knowledge"]);
        assert!(context.facts.is_empty());
    }

    #[test]
    fn pulls_in_entities_named_in_the_working_set() {
        let mut graph = SemanticGraph::new();
        let symbol = entity(EntityKind::Symbol, "SemanticGraph");
        graph.insert_entity(symbol.clone());
        graph.insert_entity(entity(EntityKind::Symbol, "unrelated"));

        let mut state = ProjectState::default();
        state.touch("SemanticGraph");

        let context = ContextBuilder::new(&graph, state).build();
        assert_eq!(context.facts, vec![symbol]);
    }

    #[test]
    fn falls_back_to_standing_facts_and_decisions_when_room_remains() {
        let mut graph = SemanticGraph::new();
        let decision = entity(EntityKind::Decision, "use redb over sqlite");
        let fact = entity(EntityKind::Fact, "no tests cover parse_pdf");
        graph.insert_entity(decision.clone());
        graph.insert_entity(fact.clone());

        let context = ContextBuilder::new(&graph, ProjectState::default()).build();
        assert!(context.facts.contains(&decision));
        assert!(context.facts.contains(&fact));
    }

    #[test]
    fn never_duplicates_an_entity_named_in_the_working_set_and_also_a_fact() {
        let mut graph = SemanticGraph::new();
        let fact = entity(EntityKind::Fact, "shared");
        graph.insert_entity(fact.clone());

        let mut state = ProjectState::default();
        state.touch("shared");

        let context = ContextBuilder::new(&graph, state).build();
        assert_eq!(context.facts.iter().filter(|e| e.id == fact.id).count(), 1);
    }

    #[test]
    fn respects_max_facts() {
        let mut graph = SemanticGraph::new();
        for i in 0..10 {
            graph.insert_entity(entity(EntityKind::Fact, &format!("fact-{i}")));
        }

        let context = ContextBuilder::new(&graph, ProjectState::default()).with_max_facts(3).build();
        assert_eq!(context.facts.len(), 3);
    }

    #[test]
    fn ignores_relationships_and_only_reads_entities() {
        let mut graph = SemanticGraph::new();
        let a = entity(EntityKind::Artifact, "a");
        let b = entity(EntityKind::Artifact, "b");
        let (a_id, b_id) = (a.id, b.id);
        graph.insert_entity(a);
        graph.insert_entity(b);
        graph.insert_relationship(Relationship::new(a_id, b_id, kopitiam_ontology::RelationshipKind::DependsOn));

        let context = ContextBuilder::new(&graph, ProjectState::default()).build();
        assert!(context.facts.is_empty());
    }
}
