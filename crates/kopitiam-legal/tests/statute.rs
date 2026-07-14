//! End-to-end ingestion of a **SYNTHETIC** statute.
//!
//! Nothing here is real law. See `kopitiam_legal::synthetic`.

use kopitiam_legal::{
    ingest, numbering::parse_statutory, source, synthetic, AnomalyKind, Date,
    DefinitionForce, DefinitionScope, IngestRequest, Instrument, ReferenceConnective, Resolution,
    ReferenceTarget,
};

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

// ---------------------------------------------------------------------------
// Provenance: every extracted item can be traced back to a page.
// ---------------------------------------------------------------------------

#[test]
fn every_provision_carries_a_page_and_verbatim_text() {
    let act = act();
    let mut count = 0;
    for history in act.provisions() {
        for provision in history.versions() {
            assert!(
                provision.provenance().page().get() >= 1,
                "{} has no page",
                provision.id()
            );
            assert!(
                !provision.text().trim().is_empty(),
                "{} has no verbatim text",
                provision.id()
            );
            assert!(
                provision.citation().contains("SYNTHETIC Widget Licensing Act"),
                "{} does not cite its document",
                provision.id()
            );
            count += 1;
        }
    }
    assert!(count >= 8, "expected the synthetic Act to yield provisions, got {count}");
}

#[test]
fn page_provenance_survives_from_the_source_pages() {
    let act = act();
    // s 7 is on page 1 of the synthetic Act; s 12 is on page 2.
    let s7 = act.history(&parse_statutory("7").unwrap()).expect("s 7");
    assert_eq!(s7.latest_known().0.provenance().page().get(), 1);

    let s12_1 = act.history(&parse_statutory("12(1)").unwrap()).expect("s 12(1)");
    assert_eq!(
        s12_1.latest_known().0.provenance().page().get(),
        2,
        "a provision on page 2 must cite page 2, not page 1"
    );
}

// ---------------------------------------------------------------------------
// Nested numbering.
// ---------------------------------------------------------------------------

#[test]
fn parses_the_deeply_nested_provision() {
    let act = act();
    let id = parse_statutory("12(3)(b)(ii)").unwrap();
    let history = act
        .history(&id)
        .unwrap_or_else(|| panic!("s 12(3)(b)(ii) should exist; got {:?}", ids(&act)));
    let (provision, _) = history.latest_known();
    assert!(
        provision.text().contains("noise"),
        "s 12(3)(b)(ii) text was {:?}",
        provision.text()
    );
}

#[test]
fn an_inserted_section_sorts_between_its_neighbours() {
    let act = act();
    let order: Vec<String> = act.provisions().map(|h| h.id().to_string()).collect();

    let pos = |needle: &str| {
        order
            .iter()
            .position(|s| s.contains(needle))
            .unwrap_or_else(|| panic!("{needle} not found in {order:?}"))
    };
    assert!(
        pos("s 12(1)") < pos("s 12A(1)") && pos("s 12A(1)") < pos("s 14(1)"),
        "12 < 12A < 14 is the legal ordering; got {order:?}"
    );
}

// ---------------------------------------------------------------------------
// Definitions — the highest-value behaviour.
// ---------------------------------------------------------------------------

#[test]
fn the_documents_own_definition_overrides_ordinary_english() {
    let act = act();

    // In ordinary English a houseboat is not a dwelling-house. In THIS Act it
    // is, because s 2 says so — and a reader given s 7 without that definition
    // has been misled, even though s 7 was extracted perfectly.
    let s7 = parse_statutory("7").unwrap();
    let Resolution::Defined(definition) =
        act.meaning_of("dwelling-house", &s7, synthetic::at(2021))
    else {
        panic!(
            "the Act defines 'dwelling-house'; got {:?}",
            act.meaning_of("dwelling-house", &s7, synthetic::at(2021))
        );
    };

    assert!(
        definition.body().contains("houseboat"),
        "the resolved meaning must be the DOCUMENT'S, not the plain-English one; got {:?}",
        definition.body()
    );
    assert_eq!(
        definition.force(),
        DefinitionForce::Includes,
        "'includes' is extensive, not exhaustive; it must not be normalised to 'means'"
    );
    assert_eq!(definition.scope(), &DefinitionScope::Instrument);
    // And it is itself sourced.
    assert!(definition.provenance().citation().contains("p 1"));
}

#[test]
fn means_and_includes_are_not_conflated() {
    let act = act();
    let s7 = parse_statutory("7").unwrap();

    let Resolution::Defined(widget) = act.meaning_of("widget", &s7, synthetic::at(2021))
    else {
        panic!("the Act defines 'widget'");
    };
    assert_eq!(
        widget.force(),
        DefinitionForce::Means,
        "'widget' is defined with 'means' — exhaustive"
    );

    let Resolution::Defined(dh) =
        act.meaning_of("dwelling-house", &s7, synthetic::at(2021))
    else {
        panic!("the Act defines 'dwelling-house'");
    };
    assert_ne!(
        widget.force(),
        dh.force(),
        "'means' and 'includes' carry different legal consequences and must stay distinct"
    );
}

#[test]
fn a_narrower_section_scoped_definition_overrides_the_act_wide_one_inside_that_section() {
    let act = act();

    // s 14(1) says: In this section, "dwelling-house" means a houseboat only.
    // Inside s 14 that narrower definition governs. Everywhere else, the
    // Act-wide s 2 definition does.
    let inside_s14 = parse_statutory("14(2)").unwrap();
    let elsewhere = parse_statutory("7").unwrap();

    let Resolution::Defined(narrow) =
        act.meaning_of("dwelling-house", &inside_s14, synthetic::at(2021))
    else {
        panic!("s 14 defines 'dwelling-house' for itself");
    };
    let Resolution::Defined(wide) =
        act.meaning_of("dwelling-house", &elsewhere, synthetic::at(2021))
    else {
        panic!("s 2 defines 'dwelling-house' Act-wide");
    };

    assert_eq!(
        narrow.force(),
        DefinitionForce::Means,
        "the s 14 definition is exhaustive ('means a houseboat only')"
    );
    assert_eq!(wide.force(), DefinitionForce::Includes);
    assert_ne!(
        narrow.body(), wide.body(),
        "the narrower, section-scoped definition must win inside its own section"
    );
    assert!(matches!(narrow.scope(), DefinitionScope::Within(_)));
    assert_eq!(wide.scope(), &DefinitionScope::Instrument);
}

#[test]
fn an_undefined_term_is_reported_as_undefined_not_given_a_plain_english_gloss() {
    let act = act();
    let s7 = parse_statutory("7").unwrap();
    assert_eq!(
        act.meaning_of("reasonable", &s7, synthetic::at(2021)),
        Resolution::NotDefined,
        "this crate is not a dictionary and must not supply an ordinary meaning"
    );
}

#[test]
fn defined_terms_used_in_a_provision_are_surfaced_with_it() {
    let act = act();
    let s7 = parse_statutory("7").unwrap();
    let (provision, _) = act.history(&s7).expect("s 7").latest_known();

    let occurrences = act.defined_terms_in(provision, synthetic::at(2021));
    let terms: Vec<&str> = occurrences.iter().map(|o| o.term()).collect();
    assert!(
        terms.contains(&"dwelling-house") && terms.contains(&"widget"),
        "s 7 uses two defined terms and both must be surfaced; got {terms:?}"
    );
}

// ---------------------------------------------------------------------------
// Cross-references.
// ---------------------------------------------------------------------------

#[test]
fn cross_references_resolve_into_links() {
    let act = act();
    let resolution = act.resolve_references(synthetic::at(2021));

    // s 12(1): "Subject to section 7, ..."
    let subject_to = resolution
        .resolved
        .iter()
        .find(|r| r.connective() == ReferenceConnective::SubjectTo)
        .expect("s 12(1) is subject to s 7");
    assert_eq!(
        subject_to.target(),
        &ReferenceTarget::Internal(parse_statutory("7").unwrap())
    );
    assert!(subject_to.from().to_string().contains("s 12(1)"));
}

#[test]
fn a_relative_reference_binds_to_the_section_that_makes_it() {
    let act = act();
    let resolution = act.resolve_references(synthetic::at(2021));

    // s 12A(2): "...charged under subsection (1)..." means s 12A(1), NOT s 12(1).
    let from_12a2 = resolution
        .resolved
        .iter()
        .find(|r| r.from().to_string().contains("s 12A(2)"))
        .expect("s 12A(2) makes a relative reference");
    assert_eq!(
        from_12a2.target(),
        &ReferenceTarget::Internal(parse_statutory("12A(1)").unwrap()),
        "'subsection (1)' inside s 12A means s 12A(1), not s 12(1)"
    );
}

#[test]
fn a_dangling_reference_is_reported_and_never_dropped() {
    let act = act();
    let resolution = act.resolve_references(synthetic::at(2021));

    // s 14(2) is "subject to section 99". There is no section 99.
    let dangling = resolution
        .dangling
        .iter()
        .find(|r| r.raw().contains("99"))
        .expect("the reference to the non-existent s 99 must be REPORTED, not dropped");
    assert_eq!(
        dangling.target(),
        &ReferenceTarget::Internal(parse_statutory("99").unwrap())
    );

    // And it reaches the anomaly list, with its source text, so a caller who
    // only looks at anomalies still sees it.
    let anomaly = act
        .anomalies()
        .iter()
        .find(|a| matches!(a.kind(), AnomalyKind::DanglingCrossReference { raw, .. } if raw.contains("99")))
        .expect("a dangling reference must surface as an anomaly");
    assert!(
        anomaly.provenance().verbatim().contains("section 99"),
        "the anomaly must carry the original words so the reader can judge"
    );
}

// ---------------------------------------------------------------------------
// Ambiguity is surfaced, never guessed.
// ---------------------------------------------------------------------------

#[test]
fn an_unparseable_provision_label_survives_with_its_text() {
    // A provision whose bracketed label is garbage. It must NOT be dropped,
    // and it must NOT be coerced into a plausible-looking number.
    let pages = vec![(
        1,
        "1. A real section.\n\
         2.—(%%) This subsection has an unparseable number but real operative words.\n",
    )];
    let lines = source::from_text_pages(&pages).unwrap();
    let instrument = ingest(IngestRequest {
        id: synthetic::act_id(),
        version: synthetic::act_version(),
        kind: synthetic::act_kind(),
        in_force_from: Date::new(2020, 1, 1).unwrap(),
        lines: &lines,
    })
    .unwrap();

    let unparsed: Vec<_> = instrument
        .provisions()
        .filter(|h| h.id().has_unrecognized())
        .collect();
    assert_eq!(unparsed.len(), 1, "the bad label must produce a provision, not nothing");

    let (provision, _) = unparsed[0].latest_known();
    assert!(
        provision.text().contains("real operative words"),
        "the text must survive even when its label does not"
    );
    assert!(
        instrument
            .anomalies()
            .iter()
            .any(|a| matches!(a.kind(), AnomalyKind::UnparseableNumbering { .. })),
        "and the reader must be told the label could not be parsed"
    );
}

#[test]
fn conflicting_definitions_are_reported_rather_than_arbitrarily_chosen() {
    // The same term, defined twice, at the same scope, inconsistently. Which
    // governs is a question of construction — we refuse to pick.
    let pages = vec![(
        1,
        "2. In this Act, \"vehicle\" means a car; \"vehicle\" means a bicycle;\n",
    )];
    let lines = source::from_text_pages(&pages).unwrap();
    let instrument = ingest(IngestRequest {
        id: synthetic::act_id(),
        version: synthetic::act_version(),
        kind: synthetic::act_kind(),
        in_force_from: Date::new(2020, 1, 1).unwrap(),
        lines: &lines,
    })
    .unwrap();

    let s2 = parse_statutory("2").unwrap();
    match instrument.meaning_of("vehicle", &s2, synthetic::at(2021)) {
        Resolution::Conflicting(candidates) => {
            assert_eq!(candidates.len(), 2, "both competing definitions must come back");
            // Both are sourced, so a human can go and read them.
            for c in candidates {
                assert!(!c.provenance().citation().is_empty());
            }
        }
        other => panic!("expected Conflicting, got {other:?} — the crate must not choose"),
    }

    assert!(
        instrument
            .anomalies()
            .iter()
            .any(|a| matches!(a.kind(), AnomalyKind::ConflictingDefinition { .. })),
        "and it must be reported as an anomaly"
    );
}

#[test]
fn a_term_defined_only_in_another_section_does_not_leak_out_of_it() {
    // s 14 defines "dwelling-house" for itself. But suppose we ask about a
    // term that ONLY has a section-scoped definition, from outside that
    // section — the answer must be "defined, but not here", not a silent
    // application of a definition that has no force where you asked.
    let pages = vec![(1, "5.—(1) In this section, \"gadget\" means a small widget;\n6. A gadget may be sold.\n")];
    let lines = source::from_text_pages(&pages).unwrap();
    let instrument = ingest(IngestRequest {
        id: synthetic::act_id(),
        version: synthetic::act_version(),
        kind: synthetic::act_kind(),
        in_force_from: Date::new(2020, 1, 1).unwrap(),
        lines: &lines,
    })
    .unwrap();

    let s6 = parse_statutory("6").unwrap();
    match instrument.meaning_of("gadget", &s6, synthetic::at(2021)) {
        Resolution::DefinedButNotHere(out_of_scope) => {
            assert_eq!(out_of_scope.len(), 1);
            assert!(
                matches!(out_of_scope[0].scope(), DefinitionScope::Within(_)),
                "the definition exists but is scoped to s 5"
            );
        }
        other => panic!(
            "a section-scoped definition must not govern outside its section; got {other:?}"
        ),
    }
}

fn ids(act: &Instrument) -> Vec<String> {
    act.provisions().map(|h| h.id().to_string()).collect()
}
