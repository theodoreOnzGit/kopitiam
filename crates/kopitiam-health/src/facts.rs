//! Emitting policy knowledge into KOPITIAM's shared semantic graph.
//!
//! # Why a policy belongs in the knowledge graph at all
//!
//! CLAUDE.md's Core Philosophy: *knowledge endures, AI accelerates*. A policy
//! wording read once should not have to be read again — not by a human, and not
//! by a model. Once its clauses are extracted they become
//! [`kopitiam_ontology::Entity`] values like any other fact the runtime holds,
//! searchable and durable, and a later question ("what does my plan say about day
//! surgery?") is answered from the graph rather than by re-reading the PDF or, far
//! worse, by asking a model to remember.
//!
//! # What this emits, and what `kopitiam-insurance` already emitted
//!
//! `kopitiam-insurance` owns the *document* layer of the graph and emits it via
//! [`kopitiam_insurance::to_graph`]: the document artifact, its clauses, its
//! definitions, its exclusions, its anomalies. This module does **not** duplicate
//! any of that. It calls it, and then adds the layer above it — the **health
//! domain terms** this crate extracted:
//!
//! ```text
//!   Artifact  (the policy document)          <- kopitiam-insurance
//!      ^
//!      | located_in
//!   Section   (a clause, with its verbatim)  <- kopitiam-insurance
//!      ^
//!      | documented_in
//!   Fact      ("deductible = SGD 3,500 per policy year")   <- kopitiam-health
//! ```
//!
//! plus a `depends_on` edge from each layer of a [`PolicyStack`] to the one below
//! it, recording the stacking itself — because "this rider sits on that plan, which
//! integrates with that scheme" is knowledge a person needs and frequently does
//! not have.
//!
//! # Facts never travel without their clause
//!
//! Each health `Fact` carries the clause's verbatim text **and** its citation in
//! its own metadata, not only via the edge to its `Section`. That duplication is
//! deliberate: a consumer that finds the fact through a search index, with no idea
//! this crate exists and no intention of walking edges, must still be unable to
//! show a term without the words it came from.
//!
//! An unresolved clause is emitted **as a fact too**, flagged `ambiguous: true`
//! with its note. A graph that silently omitted everything we could not parse would
//! tell its readers that a policy is simpler than it is — which is the same lie as
//! a wrong number, told by omission.
//!
//! # Who asserted this? The document, or you?
//!
//! Not every fact about a policy comes *from* the policy. Which layer of the stack
//! a document is — universal basic scheme, integrated plan, or rider — is supplied
//! by the **caller** when they assemble a [`PolicyStack`]. The document does not say
//! it, and there is no clause to cite for it.
//!
//! That is a real distinction and it is recorded as one. Every fact this module
//! emits carries an `asserted_by` field:
//!
//! * `"document"` — read out of a clause. Carries `verbatim` and a `citation`.
//!   **Always.**
//! * `"caller"` — supplied by whoever assembled the stack. Carries **no** `verbatim`,
//!   because inventing one would be precisely the fabrication this crate exists to
//!   prevent.
//!
//! The temptation, when a test asks "does every fact carry its clause?", is to give
//! the caller-asserted fact a plausible-looking quotation so the invariant holds.
//! That is exactly backwards. The invariant is not "every fact has a string in the
//! verbatim field"; it is "**no fact claims the document said something it did
//! not**". A caller-asserted fact honours that by having no verbatim at all, and
//! saying why.

use serde_json::json;

use kopitiam_insurance::to_graph;
use kopitiam_ontology::{Entity, EntityKind, Relationship, RelationshipKind};

use crate::policy::{PolicyLayer, PolicyStack};
use crate::term::{PolicyTerm, TermValue};

/// A batch of entities and edges, ready for `kopitiam-knowledge` to ingest.
#[derive(Debug, Clone, Default)]
pub struct FactBatch {
    /// The nodes.
    pub entities: Vec<Entity>,
    /// The edges between them.
    pub relationships: Vec<Relationship>,
}

impl FactBatch {
    /// Merges another batch into this one.
    pub fn extend(&mut self, other: FactBatch) {
        self.entities.extend(other.entities);
        self.relationships.extend(other.relationships);
    }
}

/// The provider name recorded on every entity this crate emits, so a consumer can
/// tell where a fact came from — and calibrate its trust accordingly.
pub const SOURCE: &str = "kopitiam-health";

/// The disclaimer stamped onto every fact this crate emits.
///
/// It travels *in the metadata*, not in a README, because a fact can end up a long
/// way from the crate that made it — in a search index, in a model's context
/// window, in a report — and the one thing that must never be separated from it is
/// what it does and does not mean.
const DISCLAIMER: &str = "This records what a policy document states. It is not advice, and it \
                          is not a determination that any claim is payable.";

/// Emits one policy: the document and clause layer from `kopitiam-insurance`, plus
/// this crate's health terms on top.
pub fn facts_for_policy(layer: &PolicyLayer) -> FactBatch {
    // The document, its clauses, its definitions, its anomalies — all already
    // modelled by the generic engine. We do not re-derive them.
    let generic = to_graph(layer.document());
    let mut batch = FactBatch {
        entities: generic.entities,
        relationships: generic.relationships,
    };

    // The document artifact is the first entity `to_graph` emits; every health fact
    // hangs off it.
    let Some(document_id) = batch.entities.first().map(|e| e.id) else {
        return batch;
    };

    // Which layer of the stack this policy is. Note `asserted_by: "caller"` and the
    // deliberate absence of `verbatim` — see the module docs. The document does not
    // say which layer it is; the person who assembled the stack does.
    batch.entities.push(
        Entity::new(EntityKind::Fact, format!("layer: {}", layer.kind()), SOURCE).with_metadata(
            json!({
                "policy_id": layer.id().as_str(),
                "policy_name": layer.name(),
                "layer": layer.kind().to_string(),
                "asserted_by": "caller",
                "note": "Which layer of the stack this document is was supplied by the caller \
                         when the stack was assembled. The document does not state it, so there \
                         is no clause to cite and this fact carries no verbatim text.",
                "disclaimer": DISCLAIMER,
            }),
        ),
    );
    let layer_fact_id = batch.entities.last().expect("just pushed").id;
    batch.relationships.push(Relationship::new(
        layer_fact_id,
        document_id,
        RelationshipKind::DocumentedIn,
    ));

    for term in layer.terms() {
        let (clause, fact) = entities_for_term(layer, term);
        let clause_id = clause.id;
        let fact_id = fact.id;

        batch.relationships.push(Relationship::new(
            clause_id,
            document_id,
            RelationshipKind::LocatedIn,
        ));
        batch.relationships.push(Relationship::new(
            fact_id,
            clause_id,
            RelationshipKind::DocumentedIn,
        ));

        batch.entities.push(clause);
        batch.entities.push(fact);
    }

    batch
}

/// Emits every policy in a stack, plus the `depends_on` edges that record the
/// stacking itself.
///
/// The edges matter. Encoding "rider -> plan -> basic scheme" as graph structure
/// means a later query can recover the whole picture starting from the rider alone
/// — which is the thing a person holding a rider most often cannot do from memory.
pub fn facts_for_stack(stack: &PolicyStack) -> FactBatch {
    let mut batch = FactBatch::default();
    let mut document_ids = Vec::new();

    for layer in stack.layers() {
        let layer_batch = facts_for_policy(layer);
        if let Some(first) = layer_batch.entities.first() {
            document_ids.push(first.id);
        }
        batch.extend(layer_batch);
    }

    // Layers are stored bottom-up, so each depends on the one before it:
    // plan depends_on basic scheme; rider depends_on plan.
    for pair in document_ids.windows(2) {
        batch.relationships.push(Relationship::new(
            pair[1],
            pair[0],
            RelationshipKind::DependsOn,
        ));
    }

    batch
}

/// Builds the `Section` (the clause this term was read from) and the `Fact` (the
/// term itself).
fn entities_for_term(layer: &PolicyLayer, term: &PolicyTerm) -> (Entity, Entity) {
    let provenance = term.provenance();

    // The Section carries the clause verbatim. This is the node a human is shown
    // when they ask where a term came from.
    let clause = Entity::new(
        EntityKind::Section,
        format!("clause {}", provenance.clause()),
        SOURCE,
    )
    .with_metadata(json!({
        "document": provenance.document().as_str(),
        "page": provenance.page().get(),
        "section": provenance.section().headings(),
        "clause": provenance.clause().to_string(),
        "verbatim": provenance.verbatim().as_str(),
    }));

    let ambiguity = match term.value() {
        TermValue::Ambiguous(a) => Some(json!({
            "kind": a.kind.to_string(),
            "note": a.note,
        })),
        _ => None,
    };

    let fact = Entity::new(
        EntityKind::Fact,
        format!("{}: {}", term.kind(), term.value()),
        SOURCE,
    )
    .with_metadata(json!({
        "policy_id": layer.id().as_str(),
        "layer": layer.kind().to_string(),
        "term_kind": term.kind().to_string(),
        "scope": term.scope().to_string(),
        // This fact asserts something about *what the document says*, so it must
        // carry the document's words. See the module docs.
        "asserted_by": "document",
        // Duplicated onto the Fact deliberately — see the module docs. A search
        // index that returns facts without following edges must still be unable to
        // show a term without its clause.
        "verbatim": term.verbatim(),
        "citation": provenance.to_string(),
        "ambiguous": term.is_ambiguous(),
        "ambiguity": ambiguity,
        "disclaimer": DISCLAIMER,
    }));

    (clause, fact)
}
