//! Emitting the citation graph into KOPITIAM's shared semantic graph.
//!
//! # This is the point of the crate
//!
//! Everything else here — the parsers, the identifiers, the BibTeX emitter — is
//! machinery. *This* is the product. A bibliography is **the graph of what a
//! field knows and who established it**, and a platform whose thesis is
//! "knowledge endures" should hold that graph permanently rather than
//! re-deriving it from a PDF every time somebody asks.
//!
//! Read one paper, and the runtime knows twelve more works exist, who wrote
//! them, and which claims in the paper rest on which of them. Read a hundred
//! papers, and it knows the shape of the literature — which is a thing no single
//! paper contains.
//!
//! # The mapping, and why
//!
//! | Bibliography thing | Ontology | Why |
//! |---|---|---|
//! | The citing document | [`EntityKind::Artifact`] | Ontology's own words: "a buildable/versioned unit". A paper is one. |
//! | A cited [`Reference`] | [`EntityKind::Artifact`] | **A cited work is a work.** It is not a "section" of the citing paper, and it is not a mere fact *about* it — it is a thing in the world with its own identity, which is exactly what makes the citation graph a graph rather than a list. |
//! | A reference-list entry's position | [`EntityKind::Section`] | The line as printed, page and all. Distinct from the work it names: the *entry* is in this paper; the *work* is not. |
//! | An [`Anomaly`] | [`EntityKind::Fact`] | **Deliberate.** What we could not determine is knowledge too. A graph recording only the confident findings lies by omission. |
//!
//! ## The edge
//!
//! `paper --cites--> work`, as `RelationshipKind::Custom("cites")`.
//!
//! ### Why `Custom`, and why that is a problem worth naming
//!
//! [`RelationshipKind`] has no `Cites` variant, and `kopitiam-ontology` was not
//! this crate's to change.
//!
//! That crate's own rustdoc tells the story of what happens next, in the docs for
//! `RelationshipKind::Inherits`: four language adapters were written
//! concurrently, each reached for a *different* encoding of inheritance
//! (`Custom("inherits")`, `ImplementedBy`, nearly `DependsOn`), and the shared
//! vocabulary — whose entire purpose is that a C++ base class and a Python base
//! class become the same shape of fact — was quietly defeated. `Inherits` was
//! promoted to a first-class variant precisely so that could not happen again.
//!
//! **`Cites` is the same case.** It is a fundamental relation in the scientific
//! literature, it will be reached for by `kopitiam-literature`, by any future
//! citation-analysis tooling, and by anything that ingests a bibliography from a
//! second format — and if each of them invents its own spelling, the graph cannot
//! answer *"what cites this?"*, which is the single most valuable question a
//! citation graph exists to answer.
//!
//! So this crate uses `Custom("cites")` **under protest**, with a bead filed
//! recommending `RelationshipKind::Cites` be promoted, exactly as `Inherits` was.
//! Until then, [`CITES`] is the one place the string is written, so that when the
//! variant lands there is exactly one line to change.
//!
//! # Provenance survives the crossing
//!
//! Every emitted entity's `metadata` carries the full citation: document, page,
//! and **the verbatim source string**. An entity in a permanent store that
//! asserted a scientist cited a paper, without carrying the words it read that
//! from, would be an un-sourced claim about somebody's academic conduct. The
//! provenance goes with it.
//!
//! [`Reference`]: crate::Reference
//! [`Anomaly`]: crate::Anomaly
//! [`RelationshipKind`]: kopitiam_ontology::RelationshipKind

use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::json;

use crate::bibliography::Bibliography;
use crate::entry::ParsedReference;
use crate::provenance::Provenance;
use crate::reference::Reference;

/// The knowledge-provider name recorded on every entity this crate emits, so a
/// consumer of the graph can tell where a fact came from and how far to trust it.
pub const SOURCE: &str = "kopitiam-bibliography";

/// The edge label for "paper A cites work B".
///
/// A `const` because it should be a [`RelationshipKind`] variant and is not
/// (see the module docs). One place to change when it is promoted.
///
/// [`RelationshipKind`]: kopitiam_ontology::RelationshipKind
pub const CITES: &str = "cites";

/// The edge label for "this reference-list entry names that work".
///
/// Distinct from [`CITES`]: the *entry* is a printed line in this paper; the
/// *work* is a thing in the world. Conflating them would make it impossible to
/// ask "where in the paper is this work listed?" — which is the question a reader
/// checking a citation actually asks.
pub const LISTS: &str = "lists";

/// A bibliography rendered as semantic-graph entities and relationships.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    /// The entities: the citing document, the works it cites, the reference-list
    /// entries, and the anomalies.
    pub entities: Vec<Entity>,
    /// The edges between them.
    pub relationships: Vec<Relationship>,
}

/// Turns an extracted bibliography into ontology entities and relationships.
///
/// # Determinism
///
/// Emission is **deterministic in content**: run it twice on the same
/// bibliography and you get the same entities, in the same order, carrying the
/// same metadata. Nothing here iterates a `HashMap`.
///
/// (The [`EntityId`]s themselves differ between runs, because `kopitiam-ontology`
/// mints random UUIDs in `Entity::new`. That is a runtime-wide property this
/// crate cannot fix from here — `kopitiam-insurance`'s `to_graph` carries the
/// same caveat, and it is tracked against the ontology crate, not this one.)
pub fn to_graph(bibliography: &Bibliography) -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::default();

    // -- The citing document.
    let document = Entity::new(
        EntityKind::Artifact,
        bibliography.document().as_str(),
        SOURCE,
    )
    .with_metadata(json!({
        "role": "citing_document",
        "entries": bibliography.entries().len(),
        "references_parsed": bibliography.references().count(),
        "references_unparsed": bibliography.unparsed().count(),
        "references_partial": bibliography.partial().count(),
        "citations_found": bibliography.citations().len(),
        "citations_resolved": bibliography.resolve_citations().len(),
    }));
    let document_id = document.id;
    graph.entities.push(document);

    // -- The cited works, and the reference-list entries that name them.
    //
    // Two entities per reference, not one, and the distinction is real: the
    // ENTRY is a line printed on page 15 of this paper; the WORK is a paper by
    // somebody else that exists whether or not this document mentions it. Only
    // the second is a node other documents can also point at, which is what makes
    // the graph accumulate rather than merely grow.
    let mut work_ids: Vec<Option<EntityId>> = Vec::with_capacity(bibliography.entries().len());

    for (index, entry) in bibliography.entries().iter().enumerate() {
        let label = index + 1;

        let entry_entity = Entity::new(
            EntityKind::Section,
            format!("reference {label}"),
            SOURCE,
        )
        .with_metadata(json!({
            "role": "reference_list_entry",
            "label": label,
            "status": match entry {
                ParsedReference::Parsed(_) => "parsed",
                ParsedReference::Partial(_) => "partial",
                ParsedReference::Unparsed(_) => "unparsed",
            },
            "provenance": provenance_json(entry.provenance()),
        }));
        let entry_id = entry_entity.id;
        graph.relationships.push(Relationship::new(
            entry_id,
            document_id,
            RelationshipKind::LocatedIn,
        ));
        graph.entities.push(entry_entity);

        // A reference that did not parse names no work. We do NOT invent one --
        // an Artifact node for a paper we could not identify would be a
        // fabricated work in a permanent store. The entry itself is still in the
        // graph, with its verbatim text, so nothing is lost.
        let Some(reference) = entry.reference() else {
            work_ids.push(None);
            continue;
        };

        let work = Entity::new(
            EntityKind::Artifact,
            reference
                .title()
                .unwrap_or_else(|| reference.provenance().verbatim().as_str()),
            SOURCE,
        )
        .with_metadata(reference_json(reference));
        let work_id = work.id;

        // The entry LISTS the work; the document CITES it. Two different claims.
        graph
            .relationships
            .push(Relationship::new(entry_id, work_id, custom(LISTS)));
        graph
            .relationships
            .push(Relationship::new(document_id, work_id, custom(CITES)));

        graph.entities.push(work);
        work_ids.push(Some(work_id));
    }

    // -- What we could not work out. Knowledge too, and it must survive into the
    //    graph -- a graph that recorded only the confident findings would lie by
    //    omission.
    for anomaly in bibliography.anomalies() {
        let entity = Entity::new(EntityKind::Fact, anomaly.summary(), SOURCE).with_metadata(json!({
            "fact": "anomaly",
            "is_an_assumption": anomaly.is_an_assumption(),
            "anomaly": anomaly,
            "provenance": provenance_json(anomaly.provenance()),
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

fn custom(label: &str) -> RelationshipKind {
    RelationshipKind::Custom(label.to_string())
}

/// The full record of a cited work, including what we do **not** know about it.
fn reference_json(reference: &Reference) -> serde_json::Value {
    let ids = reference.identifiers();

    json!({
        "role": "cited_work",
        "kind": reference.kind(),
        "title": reference.title(),
        // Names as WRITTEN. The graph stores what the document printed, not our
        // split of it -- see crate::author.
        "authors": reference
            .authors()
            .authors()
            .iter()
            .map(|author| json!({
                "as_written": author.as_written(),
                // `None` where we do not trust our own split. A consumer of the
                // graph must be able to tell "her family name is Waals" from
                // "we assumed Western name order and could be wrong".
                "family": author.family(),
                "sort_key": author.sort_key(),
            }))
            .collect::<Vec<_>>(),
        "author_list_truncated": reference.authors().is_truncated(),
        "container": reference.container(),
        "publisher": reference.publisher(),
        "institution": reference.institution(),
        "year": reference.year().map(|y| y.get()),
        "volume": reference.volume(),
        "issue": reference.issue(),
        "pages": reference.pages().map(ToString::to_string),
        "doi": ids.doi.as_ref().map(|d| d.as_str()),
        "arxiv": ids.arxiv.as_ref().map(|a| a.as_str()),
        "isbn": ids.isbn.as_ref().map(|i| i.as_str()),
        "issn": ids.issn.as_ref().map(|i| i.as_str()),
        "url": ids.url.as_ref().map(|u| u.as_str()),
        // The single most useful field for anyone auditing this: is there an
        // identifier at all, or is this work only findable by a human reading
        // the title? See crate::resolve.
        "identified": ids.any(),
        "unparsed_remainder": reference.unparsed(),
        "provenance": provenance_json(reference.provenance()),
    })
}

fn provenance_json(provenance: &Provenance) -> serde_json::Value {
    json!({
        "document": provenance.document().as_str(),
        "page": provenance.locator().page().map(|p| p.get()),
        "line": provenance.locator().line().map(|l| l.get()),
        // The words themselves. Everything else is a pointer; this is the thing
        // pointed at, and it must survive into the graph -- an un-sourced claim
        // about who cited whom is a claim about somebody's academic conduct.
        "verbatim": provenance.verbatim().as_str(),
        "normalised": provenance.normalised().as_str(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::parse_printed_name_list;
    use crate::citation::{CitationRef, SourcedCitation};
    use crate::entry::RawEntry;
    use crate::provenance::DocumentId;
    use crate::reference::{EntryKind, Year};

    fn doc() -> DocumentId {
        DocumentId::new("aligned_corpus.pdf").unwrap()
    }

    fn bibliography() -> Bibliography {
        let entry = |authors: &str, title: &str, year: i32| {
            let provenance =
                Provenance::from_page(&doc(), 15, format!("{authors}, {title}, {year}.")).unwrap();
            ParsedReference::Parsed(
                Reference::builder(provenance)
                    .kind(EntryKind::Article)
                    .authors(parse_printed_name_list(authors))
                    .title(title)
                    .year(Year::new(year).unwrap())
                    .build(),
            )
        };

        let citation = {
            let provenance = Provenance::from_page(&doc(), 3, "[1]").unwrap();
            SourcedCitation::new(CitationRef::parse("[1]"), provenance)
        };

        Bibliography::new(
            doc(),
            vec![
                entry("M. R. Chen", "An open-source toolkit", 2024),
                entry("R. Okafor", "Experimental validation", 2015),
            ],
            vec![citation],
            Vec::new(),
        )
    }

    #[test]
    fn the_citing_document_and_every_cited_work_become_artifacts() {
        let graph = to_graph(&bibliography());

        let artifacts: Vec<&Entity> = graph
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Artifact)
            .collect();

        // The citing paper, plus the two works it cites. A cited work IS a work.
        assert_eq!(artifacts.len(), 3);
        assert!(artifacts.iter().any(|e| e.name == "aligned_corpus.pdf"));
        assert!(artifacts.iter().any(|e| e.name == "An open-source toolkit"));
        assert!(artifacts.iter().any(|e| e.name == "Experimental validation"));
    }

    #[test]
    fn paper_a_cites_paper_b_is_an_edge() {
        // The entire point of the crate, asserted.
        let graph = to_graph(&bibliography());

        let cites: Vec<&Relationship> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Custom("cites".to_string()))
            .collect();

        assert_eq!(cites.len(), 2, "the paper cites two works");

        let document = graph
            .entities
            .iter()
            .find(|e| e.name == "aligned_corpus.pdf")
            .unwrap();
        assert!(
            cites.iter().all(|r| r.from == document.id),
            "the citing document is the source of every `cites` edge"
        );
    }

    #[test]
    fn the_reference_list_entry_and_the_work_it_names_are_different_things() {
        // The ENTRY is a line on page 15 of this paper. The WORK is a paper by
        // somebody else that exists whether or not this document mentions it.
        let graph = to_graph(&bibliography());

        let entries: Vec<&Entity> = graph
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Section)
            .collect();
        assert_eq!(entries.len(), 2);

        let lists: Vec<&Relationship> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Custom("lists".to_string()))
            .collect();
        assert_eq!(lists.len(), 2);
    }

    #[test]
    fn an_unparsed_reference_yields_no_fabricated_work() {
        // An Artifact node for a paper we could not identify would be a
        // fabricated work sitting in a permanent store.
        let provenance = Provenance::from_page(&doc(), 15, "qwertyuiop asdfghjkl").unwrap();
        let bibliography = Bibliography::new(
            doc(),
            vec![ParsedReference::Unparsed(RawEntry::new(provenance))],
            Vec::new(),
            Vec::new(),
        );
        let graph = to_graph(&bibliography);

        let artifacts = graph
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Artifact)
            .count();
        assert_eq!(artifacts, 1, "only the citing document; no work was invented");

        assert!(
            !graph
                .relationships
                .iter()
                .any(|r| r.kind == RelationshipKind::Custom("cites".to_string())),
            "and nothing is cited"
        );

        // ...but the entry itself, with its verbatim text, IS in the graph.
        let entry = graph
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Section)
            .unwrap();
        assert_eq!(entry.metadata["status"], "unparsed");
        assert_eq!(
            entry.metadata["provenance"]["verbatim"],
            "qwertyuiop asdfghjkl"
        );
    }

    #[test]
    fn every_cited_work_carries_the_words_it_was_read_from() {
        // An un-sourced claim about who cited whom is a claim about somebody's
        // academic conduct.
        let graph = to_graph(&bibliography());
        for entity in graph
            .entities
            .iter()
            .filter(|e| e.metadata["role"] == "cited_work")
        {
            let verbatim = entity.metadata["provenance"]["verbatim"]
                .as_str()
                .expect("every cited work must carry its verbatim source");
            assert!(!verbatim.is_empty());
            assert_eq!(entity.metadata["provenance"]["page"], 15);
        }
    }

    #[test]
    fn an_untrustworthy_family_name_is_null_in_the_graph_not_a_guess() {
        // A consumer of the graph must be able to tell "her family name is
        // Waals" from "we assumed Western name order and could be wrong".
        let provenance = Provenance::from_page(&doc(), 15, "Mao Zedong, On Practice, 1937.").unwrap();
        let bibliography = Bibliography::new(
            doc(),
            vec![ParsedReference::Parsed(
                Reference::builder(provenance)
                    .authors(parse_printed_name_list("Mao Zedong"))
                    .title("On Practice")
                    .year(Year::new(1937).unwrap())
                    .build(),
            )],
            Vec::new(),
            Vec::new(),
        );
        let graph = to_graph(&bibliography);

        let work = graph
            .entities
            .iter()
            .find(|e| e.metadata["role"] == "cited_work")
            .unwrap();
        let author = &work.metadata["authors"][0];

        assert_eq!(author["as_written"], "Mao Zedong");
        assert!(
            author["family"].is_null(),
            "we do not know his family name, and the graph must say so"
        );
    }

    #[test]
    fn anomalies_reach_the_graph_because_a_graph_of_only_confident_findings_lies() {
        let provenance = Provenance::from_page(&doc(), 15, "p. 111 144").unwrap();
        let bibliography = Bibliography::new(
            doc(),
            Vec::new(),
            Vec::new(),
            vec![crate::Anomaly::AssumedDigitGrouping {
                provenance,
                printed: "111 144".to_string(),
                read_as: "111144".to_string(),
            }],
        );
        let graph = to_graph(&bibliography);

        let fact = graph
            .entities
            .iter()
            .find(|e| e.kind == EntityKind::Fact)
            .expect("the anomaly must be in the graph");
        assert_eq!(fact.metadata["is_an_assumption"], true);
        assert!(fact.name.contains("ASSUMPTION"));
    }

    #[test]
    fn emission_is_deterministic_in_content() {
        let bibliography = bibliography();
        let first = to_graph(&bibliography);
        let second = to_graph(&bibliography);

        // Same entities, same order, same metadata. (The UUIDs differ -- that is
        // kopitiam-ontology's doing, not ours.)
        assert_eq!(first.entities.len(), second.entities.len());
        for (a, b) in first.entities.iter().zip(&second.entities) {
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.name, b.name);
            assert_eq!(a.metadata, b.metadata);
        }
    }
}
