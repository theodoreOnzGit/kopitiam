use kopitiam_knowledge::SemanticGraph;
use kopitiam_ontology::{Entity, EntityKind};
use kopitiam_workspace::ProjectState;

use crate::budget::ResourceBudget;

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

    /// The full priority-ordered candidate list, **before** any cap.
    ///
    /// This is the single source of truth for fact ordering, and it is what
    /// makes context **deterministic-given-budget** (temp_ai_design.md §4
    /// Refinement 2): the order is a pure function of `(graph, state)` — no
    /// wall-clock, no thread race — so taking the first *N* always yields the
    /// same prefix. `context = f(task, budget)`, never `f(wall-clock)`.
    ///
    /// Priority order (highest first), matching the runtime's "what the model
    /// most needs to see":
    /// 1. entities the current working set names (what we are touching now),
    /// 2. standing [`EntityKind::Decision`] entities (the rules in force),
    /// 3. standing [`EntityKind::Fact`] entities (tool-derived observations).
    ///
    /// Deduplicated by [`Entity::id`], stable within each tier by the graph's
    /// own iteration order.
    fn prioritized_candidates(&self) -> Vec<Entity> {
        let mut facts: Vec<Entity> = Vec::new();
        let push_unique = |facts: &mut Vec<Entity>, entity: &Entity| {
            if !facts.iter().any(|f| f.id == entity.id) {
                facts.push(entity.clone());
            }
        };

        for name in &self.state.working_set {
            for entity in self.graph.entities().filter(|e| &e.name == name) {
                push_unique(&mut facts, entity);
            }
        }
        for entity in self
            .graph
            .entities_of_kind(EntityKind::Decision)
            .chain(self.graph.entities_of_kind(EntityKind::Fact))
        {
            push_unique(&mut facts, entity);
        }
        facts
    }

    /// Builds context capped at the builder's `max_facts` (the eager,
    /// synchronous path — the "small essential core, fetched synchronously"
    /// of §4).
    pub fn build(&self) -> Context {
        self.assemble(self.max_facts)
    }

    /// Builds context capped at whatever `budget` allows, but never more than
    /// the builder's own `max_facts`.
    ///
    /// This is the progressive/anytime seam of §4: the same priority order as
    /// [`Self::build`], but the *prefix length* is a function of the budget. A
    /// tight budget (a tablet, per the §6 probe) yields a short but honest
    /// prefix of exactly the highest-priority facts; a generous budget yields
    /// more of the same list. Because the order never depends on timing, a
    /// smaller budget's facts are always a strict prefix of a larger budget's —
    /// see the `smaller_budget_is_a_prefix_of_larger` test. That prefix
    /// property is the whole point: reproducible, testable context.
    pub fn build_within(&self, budget: &dyn ResourceBudget) -> Context {
        self.assemble(self.max_facts.min(budget.fact_allowance()))
    }

    /// Shared assembly: prioritized candidates, truncated to `cap`.
    fn assemble(&self, cap: usize) -> Context {
        let mut facts = self.prioritized_candidates();
        facts.truncate(cap);
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
    fn build_within_caps_at_the_budgets_fact_allowance() {
        use crate::budget::FactBudget;
        let mut graph = SemanticGraph::new();
        for i in 0..10 {
            graph.insert_entity(entity(EntityKind::Fact, &format!("fact-{i}")));
        }
        let builder = ContextBuilder::new(&graph, ProjectState::default());
        assert_eq!(builder.build_within(&FactBudget(2)).facts.len(), 2);
        assert_eq!(builder.build_within(&FactBudget(4)).facts.len(), 4);
    }

    #[test]
    fn context_is_deterministic_given_budget_a_smaller_budget_is_a_prefix_of_larger() {
        // The §4 Refinement-2 property: context = f(task, budget), never
        // f(wall-clock). A tighter budget must yield EXACTLY the highest-
        // priority prefix of a looser budget's facts — same order, fewer of
        // them — so "how far it got" depends only on the budget, not timing.
        use crate::budget::FactBudget;
        let mut graph = SemanticGraph::new();
        for i in 0..8 {
            graph.insert_entity(entity(EntityKind::Fact, &format!("fact-{i}")));
        }
        let builder = ContextBuilder::new(&graph, ProjectState::default()).with_max_facts(usize::MAX);

        let small = builder.build_within(&FactBudget(3)).facts;
        let large = builder.build_within(&FactBudget(6)).facts;
        assert_eq!(small.len(), 3);
        assert_eq!(large.len(), 6);
        assert_eq!(small, large[..3], "small budget must be a strict prefix of large");

        // And repeated assembly at the same budget is byte-for-byte identical.
        let again = builder.build_within(&FactBudget(3)).facts;
        assert_eq!(small, again, "same budget -> same prefix, every time");
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
