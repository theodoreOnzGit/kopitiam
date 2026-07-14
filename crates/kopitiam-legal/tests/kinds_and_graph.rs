//! Document kinds (contract, judgment) and the ontology projection.
//!
//! All documents here are **SYNTHETIC**. Nothing is real law, and no real case
//! was decided by the "Synthetic Court of Appeal".

use kopitiam_legal::{
    ingest, numbering::{parse_decimal_clause, parse_judgment_paragraph}, source, synthetic,
    to_graph, AsAtResult, Citation, Classification, Date, Holding, IngestRequest, Instrument,
    InstrumentKind, Judgment, NumberingScheme, Resolution, Treatment,
};
use kopitiam_ontology::{EntityKind, RelationshipKind};

// ---------------------------------------------------------------------------
// Contracts number differently from statutes, and must not be conflated.
// ---------------------------------------------------------------------------

fn agreement() -> Instrument {
    let lines =
        source::from_text_pages(&synthetic::services_agreement_pages()).expect("synthetic pages");
    ingest(IngestRequest {
        id: kopitiam_legal::DocumentId::new("SYNTHETIC Services Agreement").unwrap(),
        version: kopitiam_legal::DocumentVersion::AsAt(Date::new(2021, 1, 1).unwrap()),
        kind: InstrumentKind::Contract {
            title: "SYNTHETIC Services Agreement".into(),
            parties: vec!["Supplier".into(), "Customer".into()],
            effective_date: Some(Date::new(2021, 1, 1).unwrap()),
        },
        in_force_from: Date::new(2021, 1, 1).unwrap(),
        lines: &lines,
    })
    .expect("synthetic agreement ingests")
}

#[test]
fn a_contract_uses_decimal_clauses_not_statutory_subsections() {
    let agreement = agreement();
    assert_eq!(
        agreement.kind().numbering_scheme(),
        NumberingScheme::DecimalClause,
        "a contract must not be parsed with statutory numbering"
    );

    let cl_2_1 = parse_decimal_clause("2.1").unwrap();
    let AsAtResult::InForce(clause) = agreement
        .provision_as_at(&cl_2_1, synthetic::at(2022))
        .expect("cl 2.1 exists")
    else {
        panic!("cl 2.1 is in force in 2022");
    };
    assert_eq!(clause.id().to_string(), "cl 2.1");
    assert!(clause.text().contains("provide the Services"));
    assert_eq!(clause.provenance().page().get(), 1);
}

#[test]
fn a_contract_defines_its_own_terms_just_as_a_statute_does() {
    // "Services" and "Business Day" mean what THIS agreement says they mean.
    // An insurance policy is a contract; this is the same machinery.
    let agreement = agreement();
    let cl_2_2 = parse_decimal_clause("2.2").unwrap();

    let Resolution::Defined(services) =
        agreement.meaning_of("Services", &cl_2_2, synthetic::at(2022))
    else {
        panic!("the agreement defines 'Services'");
    };
    assert!(services.body().contains("widget maintenance"));

    let Resolution::Defined(business_day) =
        agreement.meaning_of("Business Day", &cl_2_2, synthetic::at(2022))
    else {
        panic!("the agreement defines 'Business Day'");
    };
    assert!(
        business_day.body().contains("Saturday"),
        "'5 Business Days' in cl 2.2 does not mean 5 days"
    );
}

// ---------------------------------------------------------------------------
// Judgments.
// ---------------------------------------------------------------------------

fn judgment_instrument() -> Instrument {
    let lines = source::from_text_pages(&synthetic::judgment_pages()).expect("synthetic pages");
    let judgment = Judgment::new(
        "SYNTHETIC Alpha v Beta",
        Citation::new("[2099] SYNTH 1").unwrap(),
        "Synthetic Court of Appeal",
        Date::new(2099, 3, 1).unwrap(),
    )
    .with_coram(vec!["Synthetic JA".into()]);

    ingest(IngestRequest {
        id: kopitiam_legal::DocumentId::new("SYNTHETIC Alpha v Beta [2099] SYNTH 1").unwrap(),
        version: kopitiam_legal::DocumentVersion::AsAt(Date::new(2099, 3, 1).unwrap()),
        kind: InstrumentKind::Judgment(judgment),
        in_force_from: Date::new(2099, 3, 1).unwrap(),
        lines: &lines,
    })
    .expect("synthetic judgment ingests")
}

#[test]
fn a_judgment_uses_bracketed_paragraph_numbering() {
    let j = judgment_instrument();
    assert_eq!(
        j.kind().numbering_scheme(),
        NumberingScheme::JudgmentParagraph
    );

    let para_3 = parse_judgment_paragraph("[3]").unwrap();
    let AsAtResult::InForce(p) = j
        .provision_as_at(&para_3, synthetic::at(2100))
        .expect("[3] exists")
    else {
        panic!("[3] is in force");
    };
    assert_eq!(p.id().to_string(), "[3]");
    assert!(p.text().contains("untenable"));
}

#[test]
fn ratio_and_obiter_are_never_auto_classified() {
    let j = judgment_instrument();
    let InstrumentKind::Judgment(judgment) = j.kind() else {
        panic!("it is a judgment");
    };

    // Paragraph [4] literally contains the word "obiter". A tool that
    // pattern-matched on that word would label it — and would be doing law,
    // badly. Every paragraph must come back Unmarked.
    for n in 1..=4 {
        let para = parse_judgment_paragraph(&format!("[{n}]")).unwrap();
        assert_eq!(
            judgment.holding(&para),
            Holding::Unmarked,
            "[{n}] must be Unmarked: classifying ratio vs obiter is legal judgment, \
             not extraction — even when the paragraph says 'obiter'"
        );
    }
}

#[test]
fn a_human_can_mark_a_holding_and_their_name_is_recorded() {
    let mut judgment = Judgment::new(
        "SYNTHETIC Alpha v Beta",
        Citation::new("[2099] SYNTH 1").unwrap(),
        "Synthetic Court of Appeal",
        Date::new(2099, 3, 1).unwrap(),
    );
    let para = parse_judgment_paragraph("[3]").unwrap();

    judgment.mark_holding(
        para.clone(),
        Holding::marked_by(
            Classification::Ratio,
            "J. Tan",
            Some("necessary to the disposal of the appeal".into()),
        ),
    );

    match judgment.holding(&para) {
        Holding::Ratio { marked_by, note } => {
            assert_eq!(
                marked_by, "J. Tan",
                "an unattributed legal judgment is exactly what this crate exists to prevent"
            );
            assert!(note.unwrap().contains("necessary"));
        }
        other => panic!("expected a human-marked ratio, got {other:?}"),
    }
}

#[test]
fn a_cited_authoritys_treatment_is_never_inferred() {
    let j = judgment_instrument();
    let InstrumentKind::Judgment(judgment) = j.kind() else {
        panic!("it is a judgment");
    };
    // Whether a case was followed, distinguished or doubted is not textually
    // inferable, so nothing here is ever set to anything but `Cited` unless a
    // human says otherwise.
    for authority in judgment.authorities() {
        assert_eq!(authority.treatment(), Treatment::Cited);
    }
}

// ---------------------------------------------------------------------------
// The ontology projection.
// ---------------------------------------------------------------------------

fn act() -> Instrument {
    let lines = source::from_text_pages(&synthetic::widget_act_pages()).expect("synthetic pages");
    ingest(IngestRequest {
        id: synthetic::act_id(),
        version: synthetic::act_version(),
        kind: synthetic::act_kind(),
        in_force_from: Date::new(2020, 1, 1).unwrap(),
        lines: &lines,
    })
    .expect("synthetic Act ingests")
}

#[test]
fn provisions_enter_the_shared_knowledge_graph_as_sourced_dated_sections() {
    let act = act();
    let graph = to_graph(&act, synthetic::at(2021));

    let sections: Vec<_> = graph
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Section)
        .collect();
    assert!(!sections.is_empty(), "provisions must reach the graph");

    // Every emitted section carries its citation, its verbatim text, its page
    // and its in-force window — so nothing in the shared graph is un-sourced or
    // undated.
    for entity in &sections {
        assert_eq!(entity.source, kopitiam_legal::ontology::SOURCE);
        let m = &entity.metadata;
        assert!(m["citation"].is_string(), "{} has no citation", entity.name);
        assert!(m["verbatim_text"].is_string(), "{} has no text", entity.name);
        assert!(m["page"].is_number(), "{} has no page", entity.name);
        assert!(
            m["in_force_from"].is_string(),
            "{} has no in-force date",
            entity.name
        );
        assert!(m["as_at"].is_string(), "{} has no as-at date", entity.name);
    }
}

#[test]
fn definitions_enter_the_graph_as_facts_and_say_they_override_ordinary_meaning() {
    let act = act();
    let graph = to_graph(&act, synthetic::at(2021));

    let definition = graph
        .entities
        .iter()
        .find(|e| e.kind == EntityKind::Fact && e.name.contains("dwelling-house"))
        .expect("the 'dwelling-house' definition must reach the graph");

    assert_eq!(definition.metadata["force"], "includes");
    assert!(
        definition.metadata["body"]
            .as_str()
            .unwrap()
            .contains("houseboat")
    );
    assert!(
        definition.metadata["note"]
            .as_str()
            .unwrap()
            .contains("overrides the"),
        "the graph must carry the warning that this displaces ordinary meaning"
    );
}

#[test]
fn resolved_cross_references_become_edges_and_dangling_ones_never_do() {
    let act = act();
    let graph = to_graph(&act, synthetic::at(2021));

    // "Subject to section 7" is a real edge, labelled with the drafter's own
    // connective.
    assert!(
        graph.relationships.iter().any(|r| matches!(
            &r.kind,
            RelationshipKind::Custom(label) if label == "subject to"
        )),
        "a resolved reference must become a labelled edge"
    );

    // The dangling reference to the non-existent s 99 must NOT become an edge —
    // the graph must never contain a link to a provision that does not exist.
    // It appears as an anomaly Fact instead.
    let anomaly = graph
        .entities
        .iter()
        .find(|e| e.kind == EntityKind::Fact && e.name.contains("anomaly") && e.name.contains("99"))
        .expect("the dangling reference must reach the graph as an anomaly");
    assert!(
        anomaly.metadata["note"]
            .as_str()
            .unwrap()
            .contains("REFUSED TO GUESS")
    );
}

#[test]
fn the_graph_records_that_this_is_extraction_not_advice() {
    let act = act();
    let graph = to_graph(&act, synthetic::at(2021));
    let instrument = graph
        .entities
        .iter()
        .find(|e| e.kind == EntityKind::Artifact)
        .expect("the instrument itself is an artifact");

    let disclaimer = instrument.metadata["disclaimer"].as_str().unwrap();
    assert!(
        disclaimer.contains("Not legal advice"),
        "a downstream consumer — including an AI model — must be able to see that \
         these facts are extracted, not interpreted"
    );
    assert_eq!(instrument.metadata["instrument_kind"], "act");
}

#[test]
fn the_graph_is_deterministic_across_runs() {
    // CLAUDE.md requires deterministic behaviour. Entity ids are fresh UUIDs
    // each run (that is the ontology's design), but the *content and order* of
    // what we emit must not vary — a report that comes out shuffled every run
    // is not reproducible.
    let act = act();
    let names = |g: &kopitiam_legal::LegalGraph| -> Vec<String> {
        g.entities.iter().map(|e| e.name.clone()).collect()
    };
    assert_eq!(
        names(&to_graph(&act, synthetic::at(2021))),
        names(&to_graph(&act, synthetic::at(2021)))
    );
}
