//! Reading a policy: provenance, comparison, and emission into the knowledge graph.

mod common;

use kopitiam_health::{
    Comparability, LayerKind, TermKind, TermValue, compare, facts_for_policy, facts_for_stack,
};
use kopitiam_ontology::{EntityKind, RelationshipKind};

fn inclusive_plan() -> kopitiam_health::PolicyLayer {
    common::layer(
        "shield",
        "Teh Tarik Shield",
        LayerKind::IntegratedTopUp,
        common::SHIELD_INCLUSIVE,
    )
}

/// A rival plan whose deductible is the *same number* but whose "Hospitalisation"
/// covers a wider set of events.
fn rival_plan() -> kopitiam_health::PolicyLayer {
    common::layer(
        "rival",
        "Rival Shield",
        LayerKind::IntegratedTopUp,
        common::SHIELD_RIVAL_WIDER_DEFINITION,
    )
}

// ---------------------------------------------------------------------------
// Provenance.
// ---------------------------------------------------------------------------

/// **The invariant the whole crate rests on.**
///
/// Every term read out of a real document carries a document, a page, a section, a
/// clause number, and the clause's own words. There is no path through the type
/// system that produces one without them.
#[test]
fn every_extracted_term_carries_a_citation_and_the_clauses_own_words() {
    let plan = inclusive_plan();
    assert!(!plan.terms().is_empty());

    for term in plan.terms() {
        let p = term.provenance();

        assert_eq!(p.document().as_str(), "shield.pdf");
        assert!(p.page().get() >= 1, "there is no page zero");
        assert!(!p.section().is_empty(), "the clause's headings are recorded");
        assert!(
            !p.verbatim().as_str().trim().is_empty(),
            "a term with no source text could never be checked against the document"
        );

        // The citation renders into something a human can act on.
        let citation = p.to_string();
        assert!(citation.contains("shield.pdf"));
        assert!(citation.contains("clause"));
    }
}

/// The extracted value and the clause's words actually correspond — the deductible
/// term quotes the deductible clause, not some other clause that happened to be
/// nearby.
#[test]
fn a_terms_citation_quotes_the_clause_the_value_came_from() {
    let plan = inclusive_plan();

    let deductible = plan
        .terms_of_kind(TermKind::Deductible)
        .next()
        .expect("the wording states a deductible");

    assert_eq!(
        deductible.verbatim(),
        "3.1 The Deductible is S$3,500 for each policy year."
    );
    assert_eq!(deductible.provenance().clause().to_string(), "3.1");
    assert_eq!(deductible.provenance().page().get(), 3);

    let TermValue::Deductible(d) = deductible.value() else {
        panic!("expected a deductible");
    };
    assert_eq!(d.amount.amount().cents(), 350_000);
    assert_eq!(d.amount.currency().iso(), Some("SGD"));
}

/// A clause that merely *refers* to the deductible ("...above the Deductible") does
/// not state one, and must not be read as an unresolved deductible clause.
///
/// Getting this wrong made the calculator refuse to compute for a perfectly clear
/// policy — a failure in the safe direction, but a failure. A tool that cries wolf
/// on clean documents gets ignored on dirty ones.
#[test]
fn a_clause_that_only_mentions_the_deductible_does_not_state_one() {
    let plan = inclusive_plan();

    let deductible_terms: Vec<_> = plan.terms_of_kind(TermKind::Deductible).collect();
    assert_eq!(
        deductible_terms.len(),
        1,
        "only clause 3.1 states a deductible; clause 3.2 merely refers to it"
    );
    assert!(!deductible_terms[0].is_ambiguous());
}

/// The document's own definitions are recovered, and they are the policy's meaning,
/// not plain English.
#[test]
fn the_policys_own_definitions_are_recovered_from_the_document() {
    let plan = inclusive_plan();

    let hospitalisation = plan
        .definition("Hospitalisation")
        .expect("the wording defines it");
    assert!(hospitalisation.meaning().contains("at least one night"));

    // A word the wording does not define comes back as undefined, not as a guess.
    assert!(plan.definition("Ward Class").is_none());
}

// ---------------------------------------------------------------------------
// Comparison.
// ---------------------------------------------------------------------------

/// **The comparison trap.**
///
/// Both plans state a S$3,500 deductible on "Hospitalisation". One means an
/// overnight admission; the other includes day surgery. Those are not the same
/// deductible — they are two different deductibles on two different sets of events
/// that happen to share a number.
///
/// A tidy table showing them as equal would be a lie. So the comparison refuses to
/// call them comparable, while still showing both clauses.
#[test]
fn policies_that_define_a_term_differently_are_flagged_incomparable_not_silently_compared() {
    let mine = inclusive_plan();
    let rival = rival_plan();

    let comparison = compare(&[&mine, &rival], TermKind::Coverage);

    let Comparability::Incomparable { divergences } = &comparison.comparability else {
        panic!("two plans defining Hospitalisation differently must not be called comparable");
    };

    let hospitalisation = divergences
        .iter()
        .find(|d| d.word == "hospitalisation")
        .expect("the divergence must name the word");

    // Both definitions come back verbatim, so a human can look at them and decide.
    let described = hospitalisation.describe();
    assert!(described.contains("at least one night"));
    assert!(described.contains("day surgery"));

    // The verdict comes first, before anyone reads a table.
    let explained = comparison.explain();
    assert!(explained.starts_with("NOT COMPARABLE"));
    assert!(explained.contains("do not mean the same thing"));
}

/// Where the load-bearing definitions *do* agree, the comparison is allowed — and
/// still shows both clauses.
#[test]
fn policies_that_agree_on_the_definitions_a_term_rests_on_are_comparable() {
    let mine = inclusive_plan();
    let rival = rival_plan();

    // A deductible rests on "claimable amount" and "policy year", which both plans
    // word identically. It does *not* rest on "hospitalisation".
    let comparison = compare(&[&mine, &rival], TermKind::Deductible);
    assert!(
        comparison.comparability.is_comparable(),
        "these plans define 'claimable amount' and 'policy year' identically"
    );

    // Both plans' clauses are present, verbatim.
    assert_eq!(comparison.entries.len(), 2);
    for entry in &comparison.entries {
        let terms = entry.position.terms();
        assert_eq!(terms.len(), 1);
        assert!(terms[0].verbatim().contains("S$3,500"));
    }
}

/// A policy that does not state a term at all gets a named "not stated" position —
/// never a blank cell that a reader might take for "nil" or "excluded".
#[test]
fn a_policy_that_does_not_state_a_term_says_so_rather_than_leaving_a_blank() {
    let mine = inclusive_plan();
    let no_limit = common::layer(
        "no-limit",
        "Limitless Shield",
        LayerKind::IntegratedTopUp,
        common::SHIELD_NO_LIMIT,
    );

    let comparison = compare(&[&mine, &no_limit], TermKind::ClaimLimit);

    let missing = comparison
        .entries
        .iter()
        .find(|e| e.name == "Limitless Shield")
        .unwrap();
    assert_eq!(
        missing.position,
        kopitiam_health::PolicyPosition::NotStated,
        "an unstated term must be a named state, not an empty vec a renderer can ignore"
    );

    let explained = comparison.explain();
    assert!(explained.contains("NOT STATED"));
    assert!(
        explained.contains("not the same as 'nil'"),
        "the reader must be told what a missing figure does and does not mean"
    );
}

// ---------------------------------------------------------------------------
// Emission into the shared knowledge graph.
// ---------------------------------------------------------------------------

/// Policy knowledge enters the graph as ontology facts — and **no fact may claim
/// the document said something it did not**.
///
/// A fact asserted *by the document* carries the clause's own words, so that even a
/// consumer which found it through a search index and has never heard of this crate
/// cannot show it unsourced. A fact asserted *by the caller* (which layer of the
/// stack this document is — the document does not say) carries **no** verbatim at
/// all, and says so.
///
/// The tempting way to make an invariant like this hold is to give the
/// caller-asserted fact a plausible-looking quotation. That is exactly the
/// fabrication the crate exists to prevent, so the invariant is written to forbid
/// it rather than to invite it.
#[test]
fn policy_terms_enter_the_knowledge_graph_carrying_their_clauses() {
    let plan = inclusive_plan();
    let batch = facts_for_policy(&plan);

    let facts: Vec<_> = batch
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Fact && e.source == kopitiam_health::facts::SOURCE)
        .collect();
    assert!(!facts.is_empty());

    let mut from_document = 0;
    let mut from_caller = 0;

    for fact in &facts {
        let meta = &fact.metadata;
        let verbatim = meta.get("verbatim").and_then(|v| v.as_str());

        match meta.get("asserted_by").and_then(|v| v.as_str()) {
            Some("document") => {
                from_document += 1;
                assert!(
                    verbatim.is_some_and(|t| !t.trim().is_empty()),
                    "a fact asserting what the document says must carry the document's words"
                );
                assert!(
                    meta.get("citation").and_then(|v| v.as_str()).is_some(),
                    "...and where it says them"
                );
            }
            Some("caller") => {
                from_caller += 1;
                assert!(
                    verbatim.is_none(),
                    "a fact the caller asserted must NOT carry a quotation — there is no clause \
                     behind it, and inventing one is the fabrication this crate exists to prevent"
                );
            }
            other => panic!("every fact must say who asserted it, got {other:?}"),
        }

        assert!(
            meta.get("disclaimer")
                .and_then(|v| v.as_str())
                .is_some_and(|d| d.contains("not a determination that any claim is payable")),
            "the disclaimer must travel with the fact, not sit in a README"
        );
    }

    assert!(from_document > 0, "the wording's terms must reach the graph");
    assert_eq!(from_caller, 1, "exactly one caller-asserted fact: the layer");

    // The document/clause layer came from kopitiam-insurance; we did not re-derive it.
    assert!(
        batch
            .entities
            .iter()
            .any(|e| e.source == kopitiam_insurance::SOURCE),
        "the generic engine's document and clause entities must be present, not duplicated"
    );

    // Every health fact hangs off a clause.
    assert!(
        batch
            .relationships
            .iter()
            .any(|r| r.kind == RelationshipKind::DocumentedIn)
    );
    assert!(
        batch
            .entities
            .iter()
            .any(|e| e.kind == EntityKind::Section)
    );
}

/// The stacking itself is knowledge. A rider on a plan on a scheme becomes a chain
/// of `depends_on` edges, so a later query can recover the whole picture from the
/// rider alone — which is the thing a person holding a rider most often cannot do
/// from memory.
#[test]
fn the_stack_itself_is_recorded_as_graph_structure() {
    let stack = kopitiam_health::PolicyStack::new(vec![
        common::layer(
            "basic",
            "National Basic Health Scheme",
            LayerKind::UniversalBasic,
            common::BASIC_SCHEME,
        ),
        inclusive_plan(),
        common::layer("rider", "Kopi-O Rider", LayerKind::Rider, common::RIDER),
    ])
    .unwrap();

    let batch = facts_for_stack(&stack);

    let depends_on = batch
        .relationships
        .iter()
        .filter(|r| r.kind == RelationshipKind::DependsOn)
        .count();
    assert_eq!(
        depends_on, 2,
        "rider depends_on plan, plan depends_on basic scheme"
    );

    // Each layer's role is recorded as a fact of its own.
    let layer_facts = batch
        .entities
        .iter()
        .filter(|e| e.name.starts_with("layer: "))
        .count();
    assert_eq!(layer_facts, 3);
}

/// An unresolved clause is emitted **as a fact too**, flagged ambiguous. A graph
/// that silently dropped everything it could not parse would tell its readers a
/// policy is simpler than it is — the same lie as a wrong number, told by omission.
#[test]
fn unresolved_clauses_enter_the_graph_flagged_rather_than_being_dropped() {
    let vague = common::layer(
        "vague",
        "Vague Shield",
        LayerKind::IntegratedTopUp,
        common::SHIELD_UNDERSPECIFIED_DEDUCTIBLE,
    );

    let batch = facts_for_policy(&vague);
    let ambiguous: Vec<_> = batch
        .entities
        .iter()
        .filter(|e| {
            e.metadata
                .get("ambiguous")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
        })
        .collect();

    assert!(
        !ambiguous.is_empty(),
        "the unresolved deductible clause must appear in the graph, not vanish from it"
    );
    let fact = ambiguous[0];
    assert!(
        fact.metadata
            .get("verbatim")
            .and_then(|v| v.as_str())
            .is_some_and(|t| t.contains("The Deductible is S$3,500.")),
        "an ambiguous fact must still carry the clause a human needs to read"
    );
    assert!(fact.metadata.get("ambiguity").is_some());
}
