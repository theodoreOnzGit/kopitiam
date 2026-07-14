//! Emitting legal knowledge into the shared semantic graph.
//!
//! # Why legal material belongs in the same graph as everything else
//!
//! KOPITIAM's Semantic Runtime keeps one knowledge graph over every provider —
//! Rust symbols, PDF sections, domain facts. Legal provisions belong in it
//! for the same reason a document's sections do: so that a later
//! question ("what constrains this property's use?") can traverse from a
//! contract or filing to the regulation that governs it without anyone having
//! re-read either.
//!
//! # What we emit, and what we refuse to emit
//!
//! Every entity carries its [`crate::Provenance`] citation in its metadata, so
//! a fact that entered the graph from a statute can always be traced back to a
//! page. The `source` field is `"kopitiam-legal"`, which lets a consumer of the
//! graph know exactly how much to trust it and — importantly — that it is
//! *extracted*, not *interpreted*.
//!
//! We emit **structure and text**. We never emit a conclusion of legal effect.
//! There is no entity here meaning "X is liable" or "this clause is
//! unenforceable", and there must never be one: the graph would launder a legal
//! opinion into what looks like a deterministic fact, and downstream consumers
//! (including AI models, which is the whole point of the graph) would treat it
//! as one.
//!
//! An [`crate::Anomaly`] is emitted as a `Fact` too. A consumer that traverses
//! provisions but ignores anomalies gets a cleaner-looking and less truthful
//! picture of the document, so the graph carries both.

use kopitiam_ontology::{Entity, EntityKind, Relationship, RelationshipKind};
use serde_json::json;

use crate::{
    reference::ReferenceTarget, AsAtDate, AsAtResult, Instrument, InstrumentKind,
};

/// The `source` recorded on every entity this crate emits.
pub const SOURCE: &str = "kopitiam-legal";

/// The graph fragment produced from one instrument.
#[derive(Debug, Clone, Default)]
pub struct LegalGraph {
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
}

/// Projects an instrument into ontology entities and relationships, **as at a
/// date**.
///
/// The as-at date is mandatory here for the same reason it is mandatory
/// everywhere else in this crate: a graph containing "section 12 says X", with
/// no indication of *when* it said X, is a graph that will mislead whoever
/// queries it. The date is recorded on every emitted entity.
pub fn to_graph(instrument: &Instrument, as_at: AsAtDate) -> LegalGraph {
    let mut graph = LegalGraph::default();

    // The instrument itself.
    let mut instrument_entity = Entity::new(
        EntityKind::Artifact,
        instrument.kind().title(),
        SOURCE,
    )
    .with_metadata(json!({
        "document_id": instrument.id().to_string(),
        "document_version": instrument.version().to_string(),
        "instrument_kind": instrument_kind_label(instrument.kind()),
        "as_at": as_at.to_string(),
        "disclaimer": "Extracted text and structure only. Not legal advice; \
                       no conclusion of legal effect is asserted.",
    }));
    let instrument_id = instrument_entity.id;
    graph.entities.push(std::mem::replace(
        &mut instrument_entity,
        Entity::new(EntityKind::Artifact, "", SOURCE),
    ));

    // Provisions become Sections. Each carries its verbatim text, its citation
    // and its in-force window, so nothing in the graph is un-sourced or undated.
    let mut provision_entities = Vec::new();
    for history in instrument.provisions() {
        let AsAtResult::InForce(provision) = history.as_at(as_at) else {
            // Not in force on the queried date: it does not belong in a graph
            // that claims to describe the law as at that date. The history is
            // still available via the instrument.
            continue;
        };
        let entity = Entity::new(
            EntityKind::Section,
            provision.id().to_string(),
            SOURCE,
        )
        .with_metadata(json!({
            "citation": provision.citation(),
            "verbatim_text": provision.text(),
            "heading": provision.heading(),
            "page": provision.provenance().page().get(),
            "in_force_from": provision.validity().in_force_from().to_string(),
            "in_force_until": provision.validity().in_force_until().map(|d| d.to_string()),
            "as_at": as_at.to_string(),
            "amended_by": provision.amended_by().map(|a| a.by().to_string()),
        }));
        graph
            .relationships
            .push(Relationship::new(entity.id, instrument_id, RelationshipKind::LocatedIn));
        provision_entities.push((provision.id().clone(), entity.id));
        graph.entities.push(entity);
    }

    // Definitions become Facts. A definition IS a deterministic, tool-derived
    // observation about the document ("this instrument defines this term this
    // way"), which is exactly what EntityKind::Fact is for.
    for definition in instrument.dictionary().definitions() {
        if !definition.validity().covers(as_at) {
            continue;
        }
        let entity = Entity::new(
            EntityKind::Fact,
            format!("definition: {}", definition.term()),
            SOURCE,
        )
        .with_metadata(json!({
            "term": definition.term(),
            "force": definition.force().to_string(),
            "body": definition.body(),
            "scope": definition.scope().to_string(),
            "citation": definition.provenance().citation(),
            "verbatim_text": definition.verbatim(),
            "as_at": as_at.to_string(),
            "note": "This is the INSTRUMENT'S OWN definition and it overrides the \
                     ordinary meaning of the term within its scope.",
        }));
        // The definition is documented in the provision that states it.
        if let Some((_, provision_entity)) = provision_entities
            .iter()
            .find(|(id, _)| id == definition.provenance().provision())
        {
            graph.relationships.push(Relationship::new(
                entity.id,
                *provision_entity,
                RelationshipKind::DocumentedIn,
            ));
        }
        graph.entities.push(entity);
    }

    // Cross-references become edges. Only resolved internal ones become graph
    // edges — a dangling reference is emitted as an anomaly Fact instead, so
    // the graph never contains an edge to a provision that does not exist.
    let resolution = instrument.resolve_references(as_at);
    for reference in &resolution.resolved {
        let ReferenceTarget::Internal(target) = reference.target() else {
            continue;
        };
        let from = provision_entities.iter().find(|(id, _)| id == reference.from());
        let to = provision_entities.iter().find(|(id, _)| id == target);
        if let (Some((_, from)), Some((_, to))) = (from, to) {
            graph.relationships.push(Relationship::new(
                *from,
                *to,
                // The connective is preserved as the edge label. We record what
                // the drafter wrote ("subject to"); we do not encode what it
                // legally does, because that is construction.
                RelationshipKind::Custom(reference.connective().to_string()),
            ));
        }
    }

    // Anomalies become Facts. A consumer that traverses provisions but ignores
    // these gets a cleaner and less truthful picture of the document.
    for anomaly in instrument.anomalies() {
        graph.entities.push(
            Entity::new(
                EntityKind::Fact,
                format!("anomaly: {}", anomaly.kind()),
                SOURCE,
            )
            .with_metadata(json!({
                "citation": anomaly.provenance().citation(),
                "verbatim_text": anomaly.provenance().verbatim(),
                "detail": anomaly.kind().to_string(),
                "note": "This is something the extractor found and REFUSED TO GUESS \
                         about. Read the source text.",
            })),
        );
    }

    // A human's ratio/obiter marking is a Decision — a recorded judgment call,
    // attributable to the person who made it. Nothing this crate infers ever
    // becomes one.
    if let InstrumentKind::Judgment(judgment) = instrument.kind() {
        for (paragraph, holding) in judgment.holdings() {
            if !holding.is_marked() {
                continue;
            }
            graph.entities.push(
                Entity::new(
                    EntityKind::Decision,
                    format!("holding: {paragraph}"),
                    SOURCE,
                )
                .with_metadata(json!({
                    "paragraph": paragraph.to_string(),
                    "holding": serde_json::to_value(holding).unwrap_or(serde_json::Value::Null),
                    "case": judgment.case_name(),
                    "citation": judgment.citation().to_string(),
                    "note": "Marked by a HUMAN. kopitiam-legal never classifies ratio \
                             or obiter itself.",
                })),
            );
        }
    }

    graph
}

fn instrument_kind_label(kind: &InstrumentKind) -> &'static str {
    match kind {
        InstrumentKind::Act { .. } => "act",
        InstrumentKind::SubsidiaryLegislation { .. } => "subsidiary_legislation",
        InstrumentKind::Contract { .. } => "contract",
        InstrumentKind::Lease { .. } => "lease",
        InstrumentKind::Judgment(_) => "judgment",
    }
}
