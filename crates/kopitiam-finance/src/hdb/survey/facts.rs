//! Emitting market figures into the shared knowledge graph.
//!
//! A [`Statistic`] is a deterministic, tool-derived observation extracted from a
//! document — which is precisely [`EntityKind::Fact`] in the Semantic Runtime's
//! Common Semantic Model. Emitting them means a resale price becomes queryable
//! alongside everything else KOPITIAM knows, and survives the loss of this
//! module's own in-memory store.
//!
//! # The provenance must survive the crossing
//!
//! [`Entity`] is a flat structure — an id, a kind, a name, a source, and a JSON
//! metadata blob. It has no field for a sample size or a citation. So the
//! discipline this module spent so much effort enforcing in the type system has to
//! be carried across the boundary *by hand*, into `metadata`, or it is lost the
//! moment a figure enters the graph.
//!
//! That is what [`to_fact`] does, and it is why the `name` of the emitted entity
//! is the **fully-qualified rendering** of the statistic — measure, value,
//! stratum, period, basis, citation — rather than just the measure name. A
//! consumer that reads only the entity name still cannot come away with a bare,
//! contextless number. There is nowhere in this pipeline where "$222,222" exists
//! on its own.

use kopitiam_ontology::{Entity, EntityKind, Relationship, RelationshipKind};
use serde_json::json;

use super::statistic::Statistic;

/// The knowledge-provider name recorded on every entity this module emits.
///
/// Lets a consumer of the graph judge how much to trust a fact by knowing which
/// adapter produced it (CLAUDE.md, Scientific Standards).
pub const FACT_SOURCE: &str = "kopitiam-finance/hdb-survey";

/// Turns a statistic into a graph [`Fact`](EntityKind::Fact), provenance intact.
///
/// The `metadata` carries every component the type system required, so a consumer
/// reading the graph can reconstruct exactly what was measured, over what, when,
/// on what evidence, and from which document.
pub fn to_fact(statistic: &Statistic) -> Entity {
    let reliability = statistic.reliability();

    let metadata = json!({
        "measure": {
            "name": statistic.measure().name(),
            "unit": statistic.measure().unit().to_string(),
            "kind": statistic.measure().kind().to_string(),
            // The footnote that redefines the measure. Dropping this would let two
            // incompatible figures look identical in the graph.
            "definition": statistic.measure().definition(),
        },
        "value": statistic.quantity().to_string(),
        "population": statistic.population().to_string(),
        "stratum": statistic.stratum().to_string(),
        "period": statistic.period().to_string(),
        "basis": statistic.basis().to_string(),
        "observations": statistic.basis().observations().map(|n| n.get()),
        "lease_profile": statistic.lease_profile().to_string(),
        "methodology": statistic.methodology().id(),
        "citation": {
            "publication": statistic.citation().publication(),
            "publisher": statistic.citation().publisher(),
            "locator": statistic.citation().locator().to_string(),
            "published": statistic.citation().published().to_string(),
            "retrieved_from": statistic.citation().source(),
        },
        "reliability": reliability.to_string(),
        // A machine-readable flag, so a consumer does not have to string-match the
        // human-readable reliability to know it must warn.
        "needs_warning": reliability.needs_warning(),
    });

    // The name is the *whole* statistic, not just the measure. See the module
    // docs: there must be nowhere in the pipeline where a bare number exists.
    Entity::new(EntityKind::Fact, statistic.to_string(), FACT_SOURCE).with_metadata(metadata)
}

/// Emits a statistic as a fact, together with the document section it came from
/// and the edge between them.
///
/// The [`Section`](EntityKind::Section) entity is the cited table. The
/// [`DocumentedIn`](RelationshipKind::DocumentedIn) edge is what makes "where did
/// this number come from?" answerable *by traversing the graph*, rather than by
/// trusting a string a caller remembered to copy along.
///
/// Returns `(entities, relationships)`. The first entity is always the fact.
pub fn to_graph(statistic: &Statistic) -> (Vec<Entity>, Vec<Relationship>) {
    let fact = to_fact(statistic);

    let citation = statistic.citation();
    let section = Entity::new(
        EntityKind::Section,
        format!("{}, {}", citation.publication(), citation.locator()),
        FACT_SOURCE,
    )
    .with_metadata(json!({
        "publication": citation.publication(),
        "publisher": citation.publisher(),
        "locator": citation.locator().to_string(),
        "published": citation.published().to_string(),
        "retrieved_from": citation.source(),
    }));

    let documented_in = Relationship::new(fact.id, section.id, RelationshipKind::DocumentedIn);

    (vec![fact, section], vec![documented_in])
}

/// Emits a batch of statistics, sharing one [`Section`](EntityKind::Section)
/// entity per distinct citation.
///
/// A table of forty figures should produce forty facts and *one* section, not
/// forty duplicate sections. Deduplicated on the rendered citation, which is
/// stable because [`super::Citation`] renders deterministically.
pub fn to_graph_batch(statistics: &[Statistic]) -> (Vec<Entity>, Vec<Relationship>) {
    let mut entities = Vec::new();
    let mut relationships = Vec::new();
    let mut sections: Vec<(String, kopitiam_ontology::EntityId)> = Vec::new();

    for statistic in statistics {
        let fact = to_fact(statistic);
        let fact_id = fact.id;
        entities.push(fact);

        let citation = statistic.citation();
        let key = citation.to_string();

        let section_id = match sections.iter().find(|(existing, _)| *existing == key) {
            Some((_, id)) => *id,
            None => {
                let section = Entity::new(
                    EntityKind::Section,
                    format!("{}, {}", citation.publication(), citation.locator()),
                    FACT_SOURCE,
                )
                .with_metadata(json!({
                    "publication": citation.publication(),
                    "publisher": citation.publisher(),
                    "locator": citation.locator().to_string(),
                    "published": citation.published().to_string(),
                }));
                let id = section.id;
                sections.push((key, id));
                entities.push(section);
                id
            }
        };

        relationships.push(Relationship::new(
            fact_id,
            section_id,
            RelationshipKind::DocumentedIn,
        ));
    }

    (entities, relationships)
}

/// Whether an emitted fact is one a consumer must warn about before showing to a
/// human.
///
/// Exposed so a graph consumer — a CLI, an affordability layer — can honour the
/// small-sample discipline without depending on this crate's types.
pub fn fact_needs_warning(entity: &Entity) -> bool {
    entity
        .metadata
        .get("needs_warning")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true) // Absent metadata means unknown, and unknown means warn.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::survey::citation::{Citation, Locator};
    use crate::hdb::survey::period::Period;
    use crate::hdb::survey::quantity::{Quantity, SgdAmount, Unit};
    use crate::hdb::survey::statistic::{
        Basis, LeaseProfile, Measure, Methodology, SampleCount,
    };
    use crate::hdb::survey::stratum::{Dimension, Population, Stratum};

    fn synthetic_citation() -> Citation {
        Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("SYNTHETIC-1".into()),
            Period::Year(2024),
        )
    }

    fn price(town: &str, dollars: i64, observations: u32) -> Statistic {
        Statistic::new(
            Measure::new("Median resale price", Unit::Sgd)
                .with_definition("Prices are before grants"),
            Quantity::Money(SgdAmount::from_dollars(dollars)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, town),
            Period::Year(2023),
            Basis::Census {
                observations: SampleCount::new(observations),
            },
            LeaseProfile::Unstated,
            Methodology::new("SYNTHETIC METHODOLOGY A"),
            synthetic_citation(),
        )
        .unwrap()
    }

    #[test]
    fn a_fact_carries_its_full_provenance_across_the_boundary() {
        let entity = to_fact(&price("TAMPINES", 111_111, 400));

        assert_eq!(entity.kind, EntityKind::Fact);
        assert_eq!(entity.source, FACT_SOURCE);

        // Every component the type system insisted on must survive into metadata,
        // or the discipline ends at the graph boundary.
        let metadata = &entity.metadata;
        assert_eq!(metadata["population"], "resale transactions");
        assert_eq!(metadata["period"], "2023");
        assert_eq!(metadata["observations"], 400);
        assert_eq!(metadata["citation"]["publication"], "SYNTHETIC FIXTURE — NOT HDB DATA");
        assert_eq!(metadata["citation"]["locator"], "Table SYNTHETIC-1");
        // Including the footnote that redefines the measure.
        assert_eq!(
            metadata["measure"]["definition"],
            "Prices are before grants"
        );
    }

    #[test]
    fn the_entity_name_is_never_a_bare_number() {
        // A consumer reading only the name still cannot come away with a
        // contextless price.
        let entity = to_fact(&price("TAMPINES", 111_111, 400));
        assert!(entity.name.contains("TAMPINES"));
        assert!(entity.name.contains("2023"));
        assert!(entity.name.contains("census"));
        assert!(entity.name.contains("SYNTHETIC FIXTURE"));
    }

    #[test]
    fn a_thin_fact_is_flagged_for_warning_in_the_graph() {
        let thin = to_fact(&price("QUIET TOWN", 111_111, 3));
        assert_eq!(thin.metadata["needs_warning"], true);
        assert!(fact_needs_warning(&thin));
        assert!(thin.metadata["reliability"]
            .as_str()
            .unwrap()
            .contains("LOW PRECISION"));

        let solid = to_fact(&price("BUSY TOWN", 222_222, 400));
        assert_eq!(solid.metadata["needs_warning"], false);
        assert!(!fact_needs_warning(&solid));
    }

    #[test]
    fn an_entity_with_no_reliability_metadata_is_assumed_to_need_a_warning() {
        // Fail safe: an unknown provenance is a reason to warn, not to trust.
        let bare = Entity::new(EntityKind::Fact, "something", "elsewhere");
        assert!(fact_needs_warning(&bare));
    }

    #[test]
    fn a_fact_is_linked_to_the_table_it_came_from() {
        let (entities, relationships) = to_graph(&price("TAMPINES", 111_111, 400));

        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].kind, EntityKind::Fact);
        assert_eq!(entities[1].kind, EntityKind::Section);

        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].kind, RelationshipKind::DocumentedIn);
        // "Where did this number come from?" is answerable by traversing the edge.
        assert_eq!(relationships[0].from, entities[0].id);
        assert_eq!(relationships[0].to, entities[1].id);
    }

    #[test]
    fn a_batch_shares_one_section_per_citation() {
        // Forty figures from one table must not spawn forty duplicate sections.
        let statistics = vec![
            price("TAMPINES", 111_111, 400),
            price("QUEENSTOWN", 222_222, 300),
            price("PUNGGOL", 333_333, 200),
        ];
        let (entities, relationships) = to_graph_batch(&statistics);

        let facts = entities
            .iter()
            .filter(|e| e.kind == EntityKind::Fact)
            .count();
        let sections = entities
            .iter()
            .filter(|e| e.kind == EntityKind::Section)
            .count();

        assert_eq!(facts, 3);
        assert_eq!(sections, 1, "one table, one section");
        assert_eq!(relationships.len(), 3, "every fact still cites the table");

        let section_id = entities
            .iter()
            .find(|e| e.kind == EntityKind::Section)
            .unwrap()
            .id;
        assert!(relationships.iter().all(|r| r.to == section_id));
    }
}
