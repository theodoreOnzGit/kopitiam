//! Emission into the shared semantic graph.
//!
//! # Why a policy engine bothers with an ontology at all
//!
//! It would be easy to leave HDB policy as a private lookup table behind
//! [`HdbPolicy::assess`]. It would also throw away the point of KOPITIAM. The
//! Knowledge Engine's premise (CLAUDE.md, "Scientific Knowledge") is that a fact
//! extracted from a document belongs in **one** graph, alongside every other
//! fact the platform holds, so that it can be searched, cited, and related to
//! facts from elsewhere — not siloed inside the crate that happened to parse it.
//!
//! An income ceiling and a CPF contribution rate are the same *kind of thing*:
//! a dated, cited fact drawn from a published source. They should land in the
//! same graph, related to the same documents, reachable by the same query. So
//! this module turns the policy tables into [`Entity`] and [`Relationship`]
//! values from [`kopitiam_ontology`]:
//!
//! * one [`EntityKind::Section`] per source document, so a fact can be traced
//!   back to what it came from;
//! * one [`EntityKind::Fact`] per **provision** — including the deliberately
//!   unmodelled ones, because "we do not model the EHG after August 2024" is
//!   itself a fact worth holding, and a graph that silently omitted it would
//!   assert, by omission, that nothing applies there;
//! * a [`RelationshipKind::DocumentedIn`] edge from each fact to its source.
//!
//! The `source` field of every entity is [`SOURCE`], so a consumer can tell at a
//! glance which provider produced the fact — and, given that every citation here
//! is unverified, how much to trust it.

use std::collections::HashMap;
use std::fmt;

use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde::Serialize;
use serde_json::json;

use super::HdbPolicy;
use super::citation::Citation;
use super::temporal::{PolicyTable, Provision, Timeline};

/// The provider name stamped on every entity this module emits.
pub const SOURCE: &str = "kopitiam-finance/hdb-policy";

/// The knowledge extracted from the policy tables, ready for ingestion by
/// `kopitiam-knowledge`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyKnowledge {
    /// One `Section` per source document, one `Fact` per provision.
    pub entities: Vec<Entity>,
    /// `Fact -documented_in-> Section` for every fact.
    pub relationships: Vec<Relationship>,
}

impl PolicyKnowledge {
    /// The facts, i.e. the provisions.
    pub fn facts(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter().filter(|e| e.kind == EntityKind::Fact)
    }

    /// The source documents.
    pub fn sections(&self) -> impl Iterator<Item = &Entity> {
        self.entities
            .iter()
            .filter(|e| e.kind == EntityKind::Section)
    }
}

/// Accumulates entities and relationships, deduplicating source documents.
struct Emitter {
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    documents: HashMap<String, EntityId>,
}

impl Emitter {
    fn new() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
            documents: HashMap::new(),
        }
    }

    /// Returns the `Section` entity for a citation's document, creating it once.
    ///
    /// Documents are keyed by publisher, title and section: two provisions citing
    /// the same section of the same document should hang off one node, or the
    /// graph would show the same source many times over and a query for "what did
    /// this page tell us" would fragment.
    fn document(&mut self, citation: &Citation) -> EntityId {
        let key = format!(
            "{}|{}|{}",
            citation.document.publisher, citation.document.title, citation.section
        );
        if let Some(id) = self.documents.get(&key) {
            return *id;
        }

        let entity = Entity::new(
            EntityKind::Section,
            format!("{}: {}", citation.document.title, citation.section),
            SOURCE,
        )
        .with_metadata(json!({
            "publisher": citation.document.publisher,
            "title": citation.document.title,
            "section": citation.section,
            "published": citation.published.to_string(),
            "url": citation.url,
            "verification": citation.verification,
        }));

        let id = entity.id;
        self.entities.push(entity);
        self.documents.insert(key, id);
        id
    }

    /// Emits one `Fact` per provision, linked to the document it cites.
    fn provision<V: Serialize>(&mut self, table: &str, key: &str, provision: &Provision<V>) {
        let (name, metadata, citation) = match provision {
            Provision::InForce(dated) => (
                format!("{table} [{key}] {}", dated.effective),
                json!({
                    "table": table,
                    "key": key,
                    "modelled": true,
                    "value": dated.value,
                    "effective_from": dated.effective.from.to_string(),
                    "effective_until": dated.effective.until.map(|d| d.to_string()),
                    "citation": dated.citation,
                }),
                Some(&dated.citation),
            ),
            Provision::NotModelled {
                effective,
                reason,
                announced_in,
            } => (
                format!("{table} [{key}] {effective} — NOT MODELLED"),
                json!({
                    "table": table,
                    "key": key,
                    // The single most important flag in the emitted graph. A
                    // consumer that ignores it will treat a declared gap as a
                    // fact, which is the failure this crate is built around.
                    "modelled": false,
                    "reason": reason,
                    "effective_from": effective.from.to_string(),
                    "effective_until": effective.until.map(|d| d.to_string()),
                    "citation": announced_in,
                }),
                announced_in.as_ref(),
            ),
        };

        let fact = Entity::new(EntityKind::Fact, name, SOURCE).with_metadata(metadata);
        let fact_id = fact.id;
        self.entities.push(fact);

        if let Some(citation) = citation {
            let document_id = self.document(citation);
            self.relationships.push(Relationship::new(
                fact_id,
                document_id,
                RelationshipKind::DocumentedIn,
            ));
        }
    }

    fn timeline<V: Serialize>(&mut self, table: &str, key: &str, timeline: &Timeline<V>) {
        for provision in timeline.provisions() {
            self.provision(table, key, provision);
        }
    }

    fn table<K: Ord + fmt::Debug, V: Serialize>(&mut self, table: &PolicyTable<K, V>) {
        for (key, timeline) in table.timelines() {
            self.timeline(table.name(), &format!("{key:?}"), timeline);
        }
    }
}

impl HdbPolicy {
    /// Turns every provision in every table into ontology facts.
    ///
    /// Reproducible, not persisted: the graph is rebuilt from the tables on
    /// demand, exactly as the Semantic Runtime's "indexes are reproducible, not
    /// synchronized" principle requires. Only [`EntityId`]s differ between runs,
    /// because they are freshly generated; the content does not.
    pub fn knowledge(&self) -> PolicyKnowledge {
        let mut emitter = Emitter::new();

        emitter.table(&self.income_ceilings);
        emitter.table(&self.minimum_ages);
        emitter.table(&self.minimum_occupation_periods);
        emitter.table(&self.ethnic_quotas);
        emitter.table(&self.enhanced_housing_grant);
        emitter.table(&self.other_grants);
        emitter.table(&self.resale_levy);
        emitter.timeline(self.spr_quota.name(), "all", &self.spr_quota);
        emitter.timeline(
            self.spr_resale_waiting_period.name(),
            "all",
            &self.spr_resale_waiting_period,
        );

        PolicyKnowledge {
            entities: emitter.entities,
            relationships: emitter.relationships,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_becomes_facts_and_sections_in_the_shared_graph() {
        let knowledge = HdbPolicy::published().knowledge();

        let facts: Vec<_> = knowledge.facts().collect();
        let sections: Vec<_> = knowledge.sections().collect();

        assert!(!facts.is_empty(), "the tables must produce facts");
        assert!(!sections.is_empty(), "the facts must have sources");
        assert!(
            knowledge.entities.iter().all(|e| e.source == SOURCE),
            "every entity names the provider that produced it"
        );

        // Every fact that cites something is linked to its source document.
        for relationship in &knowledge.relationships {
            assert_eq!(relationship.kind, RelationshipKind::DocumentedIn);
        }
        assert_eq!(
            knowledge.relationships.len(),
            facts.len(),
            "every fact here carries a citation, so every fact has an edge to its source"
        );
    }

    #[test]
    fn unmodelled_provisions_are_emitted_as_facts_flagged_unmodelled() {
        // A graph that dropped these would assert, by omission, that nothing
        // applies in those spans.
        let knowledge = HdbPolicy::published().knowledge();

        let unmodelled: Vec<_> = knowledge
            .facts()
            .filter(|f| f.metadata["modelled"] == json!(false))
            .collect();

        assert!(
            !unmodelled.is_empty(),
            "the declared gaps must reach the knowledge graph"
        );
        assert!(
            unmodelled
                .iter()
                .any(|f| f.name.contains("Enhanced CPF Housing Grant")),
            "the post-August-2024 EHG gap is the most consequential one and must be visible"
        );
        for fact in unmodelled {
            assert!(
                fact.metadata["reason"].is_string(),
                "an unmodelled span must say why"
            );
        }
    }

    #[test]
    fn a_modelled_fact_carries_its_citation_in_its_metadata() {
        let knowledge = HdbPolicy::published().knowledge();
        let modelled: Vec<_> = knowledge
            .facts()
            .filter(|f| f.metadata["modelled"] == json!(true))
            .collect();

        assert!(!modelled.is_empty());
        for fact in modelled {
            assert!(
                fact.metadata["citation"]["url"].is_string(),
                "a modelled fact without a citation would be exactly the bug this crate exists \
                 to prevent: {}",
                fact.name
            );
            assert!(fact.metadata["effective_from"].is_string());
        }
    }
}
