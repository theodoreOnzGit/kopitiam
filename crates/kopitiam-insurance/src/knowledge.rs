//! Emitting insurance knowledge into KOPITIAM's shared semantic graph.
//!
//! An extracted policy is not just an answer to today's question. It is
//! engineering knowledge — the Core Philosophy's "knowledge endures" applied to
//! a legal document — and it belongs in [`kopitiam_ontology`]'s graph alongside
//! everything else the runtime knows, so a later question can be answered
//! without re-reading the PDF.
//!
//! # The mapping, and why
//!
//! | Insurance thing | Ontology | Why |
//! |---|---|---|
//! | The document | [`EntityKind::Artifact`] | A versioned unit with an identity. |
//! | A [`Clause`] | [`EntityKind::Section`] | Ontology's own definition: "a structural unit of a document". A clause is exactly that. |
//! | A [`Definition`], an [`Exclusion`], a [`ScheduleEntry`] | [`EntityKind::Fact`] | Ontology's own definition: "a deterministic, tool-derived observation". Each of these is derived, not inferred — no model was asked what the policy says. |
//! | An [`Anomaly`] | [`EntityKind::Fact`] | **Deliberate.** What we could not determine is knowledge too, and it must survive into the graph. A graph that records only the confident findings is a graph that lies by omission. |
//!
//! Relationships: a Fact is `LocatedIn` its Clause; a Clause is `LocatedIn` the
//! document Artifact; a cross-reference is a `Custom("cross_references")` edge
//! between clauses; and an endorsement's target clause is `ModifiedBy` the
//! endorsement's clause — `ModifiedBy` being exactly the edge the ontology
//! already has for "this thing was changed by that thing".
//!
//! # Provenance survives the crossing
//!
//! Every emitted entity's `metadata` carries the full citation: document, page,
//! section, clause and **the verbatim source text**. An entity in the graph that
//! asserted something about an insurance contract without carrying the words it
//! came from would be an un-sourced claim about a legal document sitting in a
//! permanent store — which is the harm this crate exists to prevent, merely
//! relocated. The provenance goes with it.
//!
//! [`Clause`]: crate::Clause
//! [`Definition`]: crate::Definition
//! [`Exclusion`]: crate::Exclusion
//! [`ScheduleEntry`]: crate::ScheduleEntry
//! [`Anomaly`]: crate::Anomaly

use std::collections::BTreeMap;

use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::json;

use crate::endorsement::EndorsementEffect;
use crate::policy::PolicyDocument;
use crate::provenance::Provenance;

/// The knowledge-provider name recorded on every entity this crate emits, so a
/// consumer of the graph can tell where a fact came from and how far to trust
/// it.
pub const SOURCE: &str = "kopitiam-insurance";

/// A policy document rendered as semantic-graph entities and relationships.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    /// The entities: the document, its clauses, and the facts derived from them.
    pub entities: Vec<Entity>,
    /// The edges between them.
    pub relationships: Vec<Relationship>,
}

/// Turns an ingested policy into ontology entities and relationships.
///
/// Emission is **deterministic in content**: run twice on the same document and
/// you get the same entities, in the same order, carrying the same metadata.
/// (Their [`EntityId`]s differ between runs, because `kopitiam-ontology` mints
/// random UUIDs — a runtime-wide property, not one this crate can fix from
/// here. See `kopitiam-hvi.1`.)
pub fn to_graph(policy: &PolicyDocument) -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::default();

    let document = Entity::new(EntityKind::Artifact, policy.id().as_str(), SOURCE).with_metadata(
        json!({
            "document_class": policy.classification().class(),
            "confidence": policy.classification().confidence(),
            "evidence": policy.classification().evidence(),
            "pages": policy.pages(),
        }),
    );
    let document_id = document.id;
    graph.entities.push(document);

    // --- Clauses become Sections, each located in the document.
    let mut clause_ids: BTreeMap<String, EntityId> = BTreeMap::new();
    for clause in policy.clauses() {
        let entity = Entity::new(
            EntityKind::Section,
            format!("clause {}", clause.id()),
            SOURCE,
        )
        .with_metadata(json!({
            "clause": clause.id().to_string(),
            "role": clause.role(),
            "heading": clause.heading(),
            "section": clause.path().headings(),
            "page": clause.page().get(),
            "verbatim": clause.text(),
        }));
        clause_ids.insert(clause.id().to_string(), entity.id);
        graph.relationships.push(Relationship::new(
            entity.id,
            document_id,
            RelationshipKind::LocatedIn,
        ));
        graph.entities.push(entity);
    }

    let locate = |graph: &mut KnowledgeGraph, entity: Entity, clause: &str| {
        if let Some(&section) = clause_ids.get(clause) {
            graph.relationships.push(Relationship::new(
                entity.id,
                section,
                RelationshipKind::LocatedIn,
            ));
        }
        graph.entities.push(entity);
    };

    // --- Definitions: the policy's private vocabulary. The most valuable
    // thing in the graph, because it is what everything else in the document
    // means.
    for definition in policy.definitions().iter() {
        let provenance = definition.provenance();
        let entity = Entity::new(
            EntityKind::Fact,
            format!("definition of {:?}", definition.term()),
            SOURCE,
        )
        .with_metadata(json!({
            "fact": "definition",
            "term": definition.term(),
            "meaning": definition.meaning(),
            "provenance": provenance_json(provenance),
        }));
        locate(&mut graph, entity, &provenance.clause().to_string());
    }

    // --- Exclusions.
    for exclusion in policy.exclusions() {
        let clause = exclusion.clause();
        let entity = Entity::new(
            EntityKind::Fact,
            format!("exclusion in clause {}", clause.id()),
            SOURCE,
        )
        .with_metadata(json!({
            "fact": "exclusion",
            "effect": exclusion.effect(),
            "provenance": provenance_json(&clause.provenance()),
        }));
        locate(&mut graph, entity, &clause.id().to_string());
    }

    // --- Schedule entries: the policy-specific numbers.
    for entry in policy.schedule().entries() {
        let provenance = entry.value().provenance();
        let entity = Entity::new(
            EntityKind::Fact,
            format!("schedule: {}", entry.label()),
            SOURCE,
        )
        .with_metadata(json!({
            "fact": "schedule_entry",
            "label": entry.label(),
            "value": entry.value().value(),
            "provenance": provenance_json(provenance),
        }));
        locate(&mut graph, entity, &provenance.clause().to_string());
    }

    // --- Endorsements. The base clause is `ModifiedBy` the endorsement's
    // clause, which is precisely what the ontology's edge means.
    for endorsement in policy.endorsements() {
        let clause = endorsement.clause();
        let entity = Entity::new(
            EntityKind::Fact,
            format!("endorsement {}", endorsement.id()),
            SOURCE,
        )
        .with_metadata(json!({
            "fact": "endorsement",
            "endorsement": endorsement.id().as_str(),
            "effective_date": endorsement.effective_date(),
            "effect": endorsement.effect(),
            "provenance": provenance_json(&clause.provenance()),
        }));
        let entity_id = entity.id;
        locate(&mut graph, entity, &clause.id().to_string());

        if let Some(target) = endorsement.effect().target()
            && !matches!(endorsement.effect(), EndorsementEffect::Adds { .. })
            && let Some(&base) = clause_ids.get(&target.to_string())
        {
            graph.relationships.push(Relationship::new(
                base,
                entity_id,
                RelationshipKind::ModifiedBy,
            ));
        }
    }

    // --- Cross-references: the policy's graph structure, made explicit.
    for clause in policy.clauses() {
        let Some(&from) = clause_ids.get(&clause.id().to_string()) else {
            continue;
        };
        for reference in clause.cross_references() {
            if let Some(&to) = clause_ids.get(&reference.target().to_string()) {
                graph.relationships.push(Relationship::new(
                    from,
                    to,
                    RelationshipKind::Custom("cross_references".to_string()),
                ));
            }
        }
    }

    // --- Anomalies. What we could not determine is knowledge too. A graph
    // that recorded only the confident findings would lie by omission.
    for anomaly in policy.anomalies() {
        let entity = Entity::new(EntityKind::Fact, anomaly.summary(), SOURCE).with_metadata(json!({
            "fact": "anomaly",
            "anomaly": anomaly,
            "verbatim": anomaly.verbatim(),
        }));
        graph.relationships.push(Relationship::new(
            entity.id,
            document_id,
            RelationshipKind::LocatedIn,
        ));
        graph.entities.push(entity);
    }

    graph
}

fn provenance_json(provenance: &Provenance) -> serde_json::Value {
    json!({
        "document": provenance.document().as_str(),
        "page": provenance.page().get(),
        "section": provenance.section().headings(),
        "clause": provenance.clause().to_string(),
        // The words themselves. Everything else is a pointer; this is the
        // thing pointed at, and it must survive into the graph.
        "verbatim": provenance.verbatim().as_str(),
    })
}
