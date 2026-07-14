//! End-to-end tests against a **synthetic** insurance policy.
//!
//! # Every word of the policy below is invented for this test
//!
//! No real insurer's policy wording is reproduced, quoted, adapted or
//! paraphrased anywhere in this file. The clauses, definitions, exclusions,
//! endorsement and schedule figures are written from scratch to exercise the
//! *structures* an insurance document uses — a definitions section, a bare
//! noun-phrase exclusion, a write-back, a cross-reference, a `label | value`
//! schedule table, a plan-by-plan benefit table, an endorsement that replaces a
//! clause.
//!
//! This matters. A plausible-looking fake exclusion clause is not a harmless
//! fixture: if it escaped into a report or a knowledge graph it would be a
//! statement about somebody's insurance that no insurer ever made. The wording
//! here is deliberately generic and obviously synthetic.
//!
//! # Why the pages are built as `TextSpan`s rather than as a PDF
//!
//! This crate writes no PDF parser; it consumes [`kopitiam_pdf::Page`]s. There
//! is no PDF *writer* in the workspace, so a real-PDF fixture would require a
//! new dependency to produce a file that would immediately be parsed back into
//! exactly these spans. Building the spans directly tests the same pipeline
//! (`kopitiam-document` reconstruction -> insurance structure) with one fewer
//! moving part and no new dependency.

use kopitiam_insurance::*;
use kopitiam_pdf::{Page, TextSpan};

// ---------------------------------------------------------------------------
// Synthetic document construction
// ---------------------------------------------------------------------------

const PAGE_WIDTH: f32 = 600.0;
const BODY_SIZE: f32 = 10.0;
const HEADING_SIZE: f32 = 14.0;
const TITLE_SIZE: f32 = 20.0;

/// A body-text line. Deliberately wide enough to cross the page midpoint, as
/// real prose does — see the note on `kopitiam-document`'s column detection in
/// [`schedule_page`].
fn prose(text: &str, y: f32, size: f32) -> TextSpan {
    TextSpan {
        text: text.to_string(),
        x: 50.0,
        y,
        width: (text.len() as f32 * 0.55 * size).max(320.0),
        height: size,
        font_size: size,
        ..TextSpan::default()
    }
}

/// A table cell at an explicit x position.
fn cell(text: &str, x: f32, y: f32) -> TextSpan {
    TextSpan {
        text: text.to_string(),
        x,
        y,
        width: text.len() as f32 * 5.0,
        height: BODY_SIZE,
        font_size: BODY_SIZE,
        ..TextSpan::default()
    }
}

/// Lays prose lines down a page, top to bottom.
fn prose_page(number: usize, lines: &[(&str, f32)]) -> Page {
    let mut spans = Vec::new();
    let mut y = 760.0;
    for (text, size) in lines {
        spans.push(prose(text, y, *size));
        y -= 22.0;
    }
    Page {
        number,
        width: PAGE_WIDTH,
        height: 800.0,
        spans,
    }
}

/// Page 1: title and the definitions section.
fn definitions_page() -> Page {
    prose_page(
        1,
        &[
            ("Kopitiam Synthetic Personal Accident Policy", TITLE_SIZE),
            ("Section 2 — Definitions", HEADING_SIZE),
            (
                "2.1 \"Accident\" means a sudden, violent, external and visible event occurring during the Period of Insurance.",
                BODY_SIZE,
            ),
            (
                "2.2 \"Hospital\" means an institution registered under the law of the place where it operates.",
                BODY_SIZE,
            ),
            (
                "2.3 \"Pre-existing Condition\" means any condition for which the Insured Person received treatment before the Policy began.",
                BODY_SIZE,
            ),
        ],
    )
}

/// Page 2: coverage, with a resolvable and a dangling cross-reference.
fn coverage_page() -> Page {
    prose_page(
        2,
        &[
            ("Section 3 — What Is Covered", HEADING_SIZE),
            (
                "3.1 We will pay the Benefit shown in the Schedule if the Insured Person suffers bodily injury caused by an Accident.",
                BODY_SIZE,
            ),
            ("Cover under this clause is subject to Clause 4.1.", BODY_SIZE),
            (
                "3.2 We will pay for treatment in a Hospital, subject to Clause 12.",
                BODY_SIZE,
            ),
        ],
    )
}

/// Page 3: the exclusions section — a bare noun phrase, a defined-term
/// exclusion, and a write-back.
fn exclusions_page() -> Page {
    prose_page(
        3,
        &[
            ("Section 4 — Exclusions", HEADING_SIZE),
            (
                "4.1 Any claim arising directly or indirectly from war or invasion.",
                BODY_SIZE,
            ),
            ("4.2 Any Pre-existing Condition of the Insured Person.", BODY_SIZE),
            (
                "4.3 This exclusion shall not apply to a Pre-existing Condition declared to us in writing before the Policy began.",
                BODY_SIZE,
            ),
            ("Section 5 — Conditions", HEADING_SIZE),
            (
                "5.1 The Insured Person must notify us of any Accident within 30 days.",
                BODY_SIZE,
            ),
        ],
    )
}

/// Page 4: the schedule.
///
/// The two prose lines at the top are not decoration. `kopitiam-document`'s
/// `split_columns` misreads a page whose lines are *all* table rows as a
/// two-column text layout and shreds the table (see `kopitiam-1gb`); it needs
/// straddling prose lines on the page to recognise it as single-column. Real
/// schedule pages carry exactly this kind of header block, which is why the
/// bug does not bite here — but a benefit-table *continuation* page would have
/// none, and it does. This crate does not work around it: fixing it belongs to
/// the Document Engine, not to a second table parser living here.
fn schedule_page() -> Page {
    let mut spans = vec![
        prose("Policy Schedule", 760.0, HEADING_SIZE),
        prose(
            "This Schedule forms part of the Policy and must be read together with the wording.",
            735.0,
            BODY_SIZE,
        ),
    ];

    let rows = [
        ("Item", "Details"),
        ("Sum Insured", "S$150,000"),
        ("Excess", "S$500"),
        ("Co-insurance", "10%"),
        ("Annual Premium", "S$420.00"),
        ("Deductible", "$1,500"),
        ("Sub-limit", "S$500 per claim, S$1,000 in the aggregate"),
        ("Optional Benefit", "Nil"),
    ];
    let mut y = 700.0;
    for (label, value) in rows {
        spans.push(cell(label, 50.0, y));
        spans.push(cell(value, 350.0, y));
        y -= 20.0;
    }

    Page {
        number: 4,
        width: PAGE_WIDTH,
        height: 800.0,
        spans,
    }
}

/// Page 5: a plan-by-plan benefit table.
fn benefit_page() -> Page {
    let mut spans = vec![
        prose("Summary of Benefits", 760.0, HEADING_SIZE),
        prose(
            "The table below compares the benefits payable under each plan offered.",
            735.0,
            BODY_SIZE,
        ),
    ];

    let rows = [
        ["Benefit", "Plan A", "Plan B"],
        ["Daily Hospital Cash", "S$100", "S$200"],
        ["Annual Limit", "S$50,000", "Unlimited"],
        ["Co-insurance", "10%", "5%"],
    ];
    let mut y = 700.0;
    for row in rows {
        spans.push(cell(row[0], 50.0, y));
        spans.push(cell(row[1], 250.0, y));
        spans.push(cell(row[2], 400.0, y));
        y -= 20.0;
    }

    Page {
        number: 5,
        width: PAGE_WIDTH,
        height: 800.0,
        spans,
    }
}

/// Page 6: an endorsement that replaces clause 4.2.
fn endorsement_page() -> Page {
    prose_page(
        6,
        &[
            ("Endorsement No. 1", TITLE_SIZE),
            (
                "It is hereby agreed that, with effect from 1 March 2026, Clause 4.2 is deleted and replaced with the following:",
                BODY_SIZE,
            ),
            (
                "Any Pre-existing Condition which was not declared to us in writing before the Policy began.",
                BODY_SIZE,
            ),
        ],
    )
}

/// The whole synthetic policy pack.
fn policy() -> PolicyDocument {
    let pages = [
        definitions_page(),
        coverage_page(),
        exclusions_page(),
        schedule_page(),
        benefit_page(),
        endorsement_page(),
    ];
    ingest_pages(
        DocumentId::new("synthetic-personal-accident-policy.pdf").unwrap(),
        &pages,
    )
    .expect("the synthetic policy ingests")
}

fn clause(policy: &PolicyDocument, id: &str) -> Clause {
    policy
        .base_clause(&ClauseId::printed(id).unwrap())
        .unwrap_or_else(|| panic!("clause {id} should have been extracted"))
        .clone()
}

// ---------------------------------------------------------------------------
// Provenance: an extracted term cannot exist without a location and the words
// ---------------------------------------------------------------------------

#[test]
fn every_clause_carries_document_page_section_id_and_verbatim_text() {
    let policy = policy();
    assert!(!policy.clauses().is_empty());

    for clause in policy.clauses() {
        let provenance = clause.provenance();
        // All five components of the citation, on every single clause, by
        // construction — there is no code path that produces a clause without
        // them, because `Clause::new` will not build one.
        assert_eq!(provenance.document().as_str(), policy.id().as_str());
        assert!(provenance.page().get() >= 1);
        assert_eq!(provenance.clause(), clause.id());
        assert!(!provenance.verbatim().as_str().trim().is_empty());
        // The words in the citation are the clause's own words.
        assert_eq!(provenance.verbatim().as_str(), clause.text());
    }
}

#[test]
fn a_citation_must_quote_the_document_and_cannot_be_a_paraphrase() {
    // The single most important negative test in the crate. If an extractor
    // could attach text of its own invention to a real clause, it could
    // produce authoritative-looking citations for things the policy does not
    // say — which is exactly the harm this engine exists to prevent.
    let policy = policy();
    let war = clause(&policy, "4.1");

    assert!(war.cite("war or invasion").is_ok());

    let err = war
        .cite("war is not covered under any circumstances")
        .expect_err("a paraphrase is not a quotation");
    assert!(matches!(err, ProvenanceError::QuoteNotInClause { .. }));

    // ... and therefore no `ExtractedTerm` can be built on it either.
    assert!(war.extract((), "war is not covered under any circumstances").is_err());
}

#[test]
fn provenance_survives_a_domain_crates_refinement() {
    // The seam `kopitiam-health` builds on: refine the value, keep the words.
    let policy = policy();
    let excess = policy.schedule().get("Excess").expect("Excess row");

    let refined = excess.value().clone().map(|value| match value {
        ScheduleValue::Money(money) => money.amount().cents(),
        _ => panic!("Excess should have typed as money"),
    });

    assert_eq!(*refined.value(), 50_000);
    assert_eq!(refined.provenance().page().get(), 4);
    assert!(refined.verbatim().contains("S$500"));
}

// ---------------------------------------------------------------------------
// Definitions — the highest-value test in the crate
// ---------------------------------------------------------------------------

#[test]
fn a_defined_term_resolves_to_the_policys_meaning_and_not_the_plain_english_one() {
    // THE test. The policy defines "Accident" narrowly. Clause 3.1 uses the
    // word. A reader (or an extractor) that takes it at face value concludes
    // that any accidental injury is covered — which is the opposite of what
    // this contract says, and is wrong *confidently*.
    let policy = policy();

    let Resolution::Defined(accident) = policy.meaning_of("Accident") else {
        panic!("the policy defines Accident; resolution must find it");
    };

    // The policy's meaning — narrow, and nothing like the plain English one.
    assert_eq!(
        accident.meaning(),
        "a sudden, violent, external and visible event occurring during the Period of Insurance."
    );
    // It is NOT the plain-English notion of an accident: the policy requires
    // the event to be violent, external and visible, and to occur inside the
    // Period of Insurance. All four qualifiers are load-bearing.
    for qualifier in ["violent", "external", "visible", "Period of Insurance"] {
        assert!(
            accident.meaning().contains(qualifier),
            "the policy's definition constrains {qualifier:?}; a plain-English reading does not"
        );
    }
    // And the definition is cited: page 1, clause 2.1.
    assert_eq!(accident.provenance().page().get(), 1);
    assert_eq!(accident.provenance().clause().to_string(), "2.1");

    // Now the mechanism: the word, where it is *used*, resolves to that meaning.
    let coverage = clause(&policy, "3.1");
    let occurrences = policy.defined_terms_in(&coverage);
    let accident_use = occurrences
        .iter()
        .find(|occurrence| occurrence.surface == "Accident")
        .expect("clause 3.1 uses the defined term Accident");

    let Resolution::Defined(resolved) = &accident_use.resolution else {
        panic!("the use of Accident in clause 3.1 must resolve to the policy's definition");
    };
    assert_eq!(resolved.meaning(), accident.meaning());
    assert!(accident_use.resolution.is_policy_defined());

    // The occurrence is located, so a reader can be shown exactly which word
    // in the clause is not being used in its ordinary sense.
    assert_eq!(&coverage.text()[accident_use.range.clone()], "Accident");
}

#[test]
fn an_undefined_term_is_reported_as_undefined_rather_than_guessed_at() {
    let policy = policy();
    assert_eq!(policy.meaning_of("Earthquake"), Resolution::Undefined);
    assert!(!policy.meaning_of("Earthquake").is_policy_defined());
}

#[test]
fn the_definitions_section_is_found_structurally() {
    let policy = policy();
    assert_eq!(policy.definitions().len(), 3);
    for term in ["Accident", "Hospital", "Pre-existing Condition"] {
        assert!(
            policy.meaning_of(term).is_policy_defined(),
            "{term} should be defined by the policy"
        );
    }
}

// ---------------------------------------------------------------------------
// Exclusions — an exclusion that reads as coverage is the worst possible bug
// ---------------------------------------------------------------------------

#[test]
fn exclusions_are_extracted_and_are_never_confused_with_coverage() {
    let policy = policy();

    let exclusions = policy.exclusions();
    let excluded: Vec<String> = exclusions
        .iter()
        .map(|exclusion| exclusion.clause().id().to_string())
        .collect();
    assert!(excluded.contains(&"4.1".to_string()), "got {excluded:?}");
    assert!(excluded.contains(&"4.2".to_string()), "got {excluded:?}");

    // Clause 4.1 is a bare noun phrase — "Any claim arising ... from war or
    // invasion." It contains no negation and nothing that reads as an
    // exclusion. It IS one, because of the section it is printed under. A
    // language-only classifier files this as coverage, and tells someone their
    // war loss is payable.
    let war = clause(&policy, "4.1");
    assert_eq!(war.role(), ClauseRole::Exclusion);
    assert!(war.text().contains("war or invasion"));

    // The two lists are disjoint. An exclusion appearing among the coverages
    // is the worst bug this crate could have.
    let covered: Vec<String> = policy
        .coverages()
        .iter()
        .map(|clause| clause.id().to_string())
        .collect();
    for id in &excluded {
        assert!(
            !covered.contains(id),
            "clause {id} is an exclusion and must NOT appear as coverage"
        );
    }
    assert!(covered.contains(&"3.1".to_string()), "got {covered:?}");
}

#[test]
fn a_write_back_inside_the_exclusions_section_is_not_read_as_an_exclusion() {
    // Clause 4.3 sits under `Exclusions` but *restores* cover. Read as an
    // exclusion it means the exact opposite of what it says.
    let policy = policy();
    let write_back = policy
        .exclusions()
        .into_iter()
        .find(|exclusion| exclusion.clause().id().to_string() == "4.3")
        .expect("clause 4.3 is in the exclusions section");

    assert_eq!(write_back.effect(), ExclusionEffect::WritesBack);
    assert!(write_back.text().contains("shall not apply"));

    // ... while its neighbours really do exclude.
    let war = policy
        .exclusions()
        .into_iter()
        .find(|exclusion| exclusion.clause().id().to_string() == "4.1")
        .unwrap();
    assert_eq!(war.effect(), ExclusionEffect::Excludes);
}

// ---------------------------------------------------------------------------
// Cross-references
// ---------------------------------------------------------------------------

#[test]
fn a_cross_reference_resolves_to_the_clause_it_names() {
    let policy = policy();
    let coverage = clause(&policy, "3.1");

    let references = policy.references_from(&coverage);
    let resolved = references
        .iter()
        .find(|reference| !reference.is_dangling())
        .expect("clause 3.1 refers to clause 4.1, which exists");

    let target = resolved.target().expect("resolved");
    assert_eq!(target.id().to_string(), "4.1");
    assert!(target.text().contains("war or invasion"));
}

#[test]
fn a_dangling_cross_reference_is_reported_and_not_ignored() {
    // Clause 3.2 defers to "Clause 12", which does not exist. A tool that
    // quietly dropped the reference would let a reader believe they had read
    // clause 3.2 to its conclusion — when in fact part of the contract is
    // missing.
    let policy = policy();
    let clause_3_2 = clause(&policy, "3.2");

    let dangling = policy
        .references_from(&clause_3_2)
        .into_iter()
        .find(|reference| reference.is_dangling())
        .expect("the reference to Clause 12 dangles");
    assert!(dangling.target().is_none());

    let reported = policy.anomalies().iter().any(|anomaly| {
        matches!(anomaly, Anomaly::DanglingCrossReference { target, .. }
            if target.to_string() == "12")
    });
    assert!(reported, "a dangling reference must be surfaced as an anomaly");
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

#[test]
fn schedule_rows_become_typed_values_with_units_and_provenance() {
    let policy = policy();
    let schedule = policy.schedule();

    let sum_insured = schedule.get("Sum Insured").expect("Sum Insured row");
    let money = sum_insured
        .value()
        .value()
        .as_money()
        .expect("a sum insured is money");
    assert_eq!(money.currency().iso(), Some("SGD"));
    assert_eq!(money.amount().cents(), 15_000_000);
    assert_eq!(money.amount().to_decimal_string(), "150000.00");

    // Provenance: the citation is the whole ROW, not the bare cell — "150,000"
    // on its own tells a reader nothing they can check.
    let provenance = sum_insured.value().provenance();
    assert_eq!(provenance.page().get(), 4);
    assert!(provenance.verbatim().as_str().contains("Sum Insured"));
    assert!(provenance.verbatim().as_str().contains("S$150,000"));

    let co_insurance = schedule.get("Co-insurance").expect("Co-insurance row");
    assert_eq!(
        co_insurance.value().value().as_percentage().unwrap().basis_points(),
        1000
    );

    // "Nil" is not zero, and this crate will not turn it into zero.
    let optional = schedule.get("Optional Benefit").expect("Optional Benefit row");
    assert!(matches!(optional.value().value(), ScheduleValue::Nil(_)));
    assert!(optional.value().value().as_money().is_none());
}

#[test]
fn an_ambiguous_currency_is_not_silently_resolved() {
    // The schedule prints the deductible as "$1,500". `$` is not a currency.
    let policy = policy();
    let deductible = policy.schedule().get("Deductible").expect("Deductible row");
    let money = deductible.value().value().as_money().unwrap();
    assert_eq!(money.currency(), &Currency::Ambiguous("$".to_string()));
    assert_eq!(money.currency().iso(), None);
    // The amount is still known exactly. Only the currency is not.
    assert_eq!(money.amount().cents(), 150_000);
}

#[test]
fn an_unparseable_schedule_value_is_surfaced_with_its_text_never_defaulted() {
    // The sub-limit is compound: "S$500 per claim, S$1,000 in the aggregate".
    // Typing it as S$500 would silently discard the aggregate limit while
    // looking completely convincing.
    let policy = policy();
    let sublimit = policy.schedule().get("Sub-limit").expect("Sub-limit row");
    assert!(sublimit.value().value().is_unparseable());

    let reported = policy.anomalies().iter().any(|anomaly| {
        matches!(anomaly, Anomaly::UnparseableScheduleValue { label, raw, .. }
            if label == "Sub-limit" && raw.contains("in the aggregate"))
    });
    assert!(reported, "an untypeable value must be surfaced, not dropped");

    // And the words are still there for a human to read.
    let anomaly = policy
        .anomalies()
        .iter()
        .find(|a| matches!(a, Anomaly::UnparseableScheduleValue { .. }))
        .unwrap();
    assert!(anomaly.verbatim().unwrap().contains("S$1,000"));
}

#[test]
fn a_plan_by_plan_benefit_table_keeps_each_plans_value_distinct() {
    // The shape `kopitiam-health` needs: a benefit is worth different amounts
    // under different plans, and flattening the table loses which is which.
    let policy = policy();
    let table = policy
        .benefit_tables()
        .first()
        .expect("the benefit page yields a benefit table");

    assert_eq!(table.plans(), ["Plan A", "Plan B"]);

    let plan_a = table
        .value_for("Daily Hospital Cash", "Plan A")
        .expect("Plan A daily hospital cash");
    let plan_b = table
        .value_for("Daily Hospital Cash", "Plan B")
        .expect("Plan B daily hospital cash");

    assert_eq!(plan_a.value().as_money().unwrap().amount().cents(), 10_000);
    assert_eq!(plan_b.value().as_money().unwrap().amount().cents(), 20_000);

    // Both carry the same row citation, which is what a reader needs to check.
    assert!(plan_a.verbatim().contains("Daily Hospital Cash"));
    assert_eq!(plan_a.provenance().page().get(), 5);

    // "Unlimited" is kept as printed rather than becoming a very large number.
    let annual_b = table.value_for("Annual Limit", "Plan B").unwrap();
    assert!(matches!(annual_b.value(), ScheduleValue::Unlimited(_)));
}

// ---------------------------------------------------------------------------
// Endorsements
// ---------------------------------------------------------------------------

#[test]
fn an_endorsement_overrides_the_base_clause_and_the_override_is_visible() {
    // A reader who misses the endorsement reads clause 4.2 as "Any
    // Pre-existing Condition" — excluding conditions they duly declared. The
    // endorsement narrows it to *undeclared* conditions. The difference is the
    // whole claim.
    let policy = policy();
    let target = ClauseId::printed("4.2").unwrap();

    // The base wording is still reachable — under a name that says what it is.
    let base = policy.base_clause(&target).expect("clause 4.2 in the wording");
    assert!(base.text().contains("Any Pre-existing Condition of the Insured Person."));

    // But the contract is the *effective* clause, and it cannot be read
    // without the endorsement being handed over.
    let effective = policy.effective_clause(&target);
    let EffectiveClause::Replaced { by, wording, base: superseded } = effective else {
        panic!("clause 4.2 was replaced by Endorsement No. 1; got {effective:?}");
    };

    assert!(effective.is_overridden());
    assert_eq!(by.id().as_str(), "Endorsement No. 1");
    assert_eq!(by.effective_date(), Some("1 March 2026"));
    assert!(wording.as_str().contains("was not declared to us in writing"));
    assert_eq!(superseded.id().to_string(), "4.2");

    // The reader gets the MODIFIED meaning from `wording()`, not the base one.
    let effective_text = policy.effective_clause(&target).wording().unwrap();
    assert!(effective_text.contains("not declared"));
    assert!(!effective_text.contains("Any Pre-existing Condition of the Insured Person."));

    // The endorsement itself is citable, verbatim, on its own page.
    assert_eq!(by.provenance().page().get(), 6);
}

#[test]
fn a_clause_no_endorsement_touches_reads_as_the_base_wording() {
    let policy = policy();
    let effective = policy.effective_clause(&ClauseId::printed("4.1").unwrap());
    assert!(matches!(effective, EffectiveClause::Base(_)));
    assert!(!effective.is_overridden());
    assert!(effective.endorsement().is_none());
    assert!(effective.wording().unwrap().contains("war or invasion"));
}

#[test]
fn a_clause_the_document_does_not_contain_is_not_invented() {
    let policy = policy();
    assert_eq!(
        policy.effective_clause(&ClauseId::printed("99.9").unwrap()),
        EffectiveClause::NotFound
    );
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[test]
fn a_policy_pack_is_classified_as_a_wording_with_stated_evidence() {
    let policy = policy();
    let classification = policy.classification();
    assert_eq!(classification.class(), DocumentClass::PolicyWording);
    assert!(
        !classification.evidence().is_empty(),
        "a classification with no evidence is an assertion, not a finding"
    );
}

#[test]
fn a_standalone_endorsement_is_not_mistaken_for_a_wording() {
    // The costliest misclassification: an endorsement filed as base wording has
    // its override absorbed into the contract's original terms and lost.
    let policy = ingest_pages(
        DocumentId::new("endorsement-1.pdf").unwrap(),
        &[endorsement_page()],
    )
    .unwrap();
    assert_eq!(
        policy.classification().class(),
        DocumentClass::Endorsement,
        "evidence: {:?}",
        policy.classification().evidence()
    );
}

// ---------------------------------------------------------------------------
// Ambiguity is surfaced, never swallowed
// ---------------------------------------------------------------------------

#[test]
fn nothing_is_silently_dropped_and_the_anomalies_carry_their_text() {
    let policy = policy();
    assert!(
        !policy.anomalies().is_empty(),
        "this document has a dangling reference and an untypeable value; \
         a report of no anomalies would be a lie"
    );

    for anomaly in policy.anomalies() {
        // Every anomaly says something a human can act on.
        assert!(!anomaly.summary().is_empty());
        // And every anomaly about a specific passage carries that passage.
        if let Some(verbatim) = anomaly.verbatim() {
            assert!(!verbatim.trim().is_empty());
        }
    }
}

#[test]
fn a_wording_with_no_definitions_section_is_loudly_reported() {
    // Without this, every defined term in the document would silently fall back
    // to plain English — which is how a policy gets read backwards.
    let policy = ingest_pages(
        DocumentId::new("no-definitions.pdf").unwrap(),
        &[exclusions_page(), coverage_page()],
    )
    .unwrap();

    assert!(
        policy
            .anomalies()
            .iter()
            .any(|anomaly| matches!(anomaly, Anomaly::NoDefinitionsSection { .. })),
        "a wording with no definitions section must be reported; got {:?}",
        policy.anomalies()
    );
}

#[test]
fn a_conflicting_definition_is_reported_rather_than_resolved() {
    // A second document (an endorsement pack) redefines "Accident" more widely.
    // Silently picking one meaning would decide a claim.
    //
    // The page carries several body-size lines deliberately. `kopitiam-document`
    // estimates body font size as the most common size on the page and breaks a
    // tie via `HashMap` iteration order, which Rust randomises per process — so
    // a page with one line at each size reconstructs differently between runs
    // (`kopitiam-mg3`). A realistic endorsement has body text; a three-line one
    // would be testing that bug rather than this behaviour.
    let redefinition = prose_page(
        1,
        &[
            ("Endorsement No. 2", TITLE_SIZE),
            ("Section 2 — Definitions", HEADING_SIZE),
            (
                "2.1 \"Accident\" means any sudden event, whether or not violent.",
                BODY_SIZE,
            ),
            (
                "This endorsement forms part of the Policy and amends the definition above.",
                BODY_SIZE,
            ),
            (
                "All other terms and conditions of the Policy remain unchanged.",
                BODY_SIZE,
            ),
        ],
    );

    let base = policy();
    let amended = ingest_pages(DocumentId::new("endorsement-2.pdf").unwrap(), &[redefinition])
        .unwrap();
    let combined = base.absorb(amended);

    match combined.meaning_of("Accident") {
        Resolution::Conflicting(all) => {
            assert_eq!(all.len(), 2);
            // Both citations survive, so the reader can see the contradiction.
            assert!(all.iter().any(|d| d.meaning().contains("violent, external")));
            assert!(all.iter().any(|d| d.meaning().contains("whether or not violent")));
        }
        other => panic!("expected Conflicting, got {other:?}"),
    }
    // It must NOT silently pick one.
    assert!(combined.meaning_of("Accident").definition().is_none());
    // But it is still policy-defined: plain English is not the answer.
    assert!(combined.meaning_of("Accident").is_policy_defined());

    assert!(
        combined
            .anomalies()
            .iter()
            .any(|anomaly| matches!(anomaly, Anomaly::ConflictingDefinition { term, .. }
                if term.eq_ignore_ascii_case("accident"))),
        "a conflicting definition must be surfaced"
    );
}

// ---------------------------------------------------------------------------
// The semantic graph
// ---------------------------------------------------------------------------

#[test]
fn the_policy_enters_the_knowledge_graph_with_its_provenance_intact() {
    use kopitiam_ontology::{EntityKind, RelationshipKind};

    let policy = policy();
    let graph = to_graph(&policy);

    assert!(
        graph
            .entities
            .iter()
            .any(|entity| entity.kind == EntityKind::Artifact)
    );
    assert!(
        graph
            .entities
            .iter()
            .filter(|entity| entity.kind == EntityKind::Section)
            .count()
            >= 8,
        "each clause becomes a Section"
    );

    // Every emitted Fact carries the verbatim words it was derived from. A fact
    // about a legal contract sitting in a permanent store without the text it
    // came from is an un-sourced claim.
    let facts: Vec<_> = graph
        .entities
        .iter()
        .filter(|entity| entity.kind == EntityKind::Fact)
        .collect();
    assert!(!facts.is_empty());
    for fact in &facts {
        assert_eq!(fact.source, kopitiam_insurance::SOURCE);
        let has_words = fact.metadata["provenance"]["verbatim"].is_string()
            || fact.metadata["verbatim"].is_string();
        assert!(has_words, "fact without its source text: {:?}", fact.name);
    }

    // The definitions made it in — the most valuable thing in the graph,
    // because it is what everything else in the document means.
    assert!(
        facts
            .iter()
            .any(|fact| fact.metadata["fact"] == "definition"
                && fact.metadata["term"] == "Accident")
    );

    // And the endorsement's override is an edge, not a footnote.
    assert!(
        graph
            .relationships
            .iter()
            .any(|relationship| relationship.kind == RelationshipKind::ModifiedBy),
        "the base clause must be recorded as ModifiedBy the endorsement"
    );
}

#[test]
fn extraction_is_deterministic() {
    // The Semantic Runtime's reproducibility principle: an index rebuilt from
    // the same source must be the same index.
    let first = policy();
    let second = policy();

    let ids = |policy: &PolicyDocument| -> Vec<String> {
        policy
            .clauses()
            .iter()
            .map(|clause| clause.id().to_string())
            .collect()
    };
    assert_eq!(ids(&first), ids(&second));

    let summaries = |policy: &PolicyDocument| -> Vec<String> {
        policy.anomalies().iter().map(Anomaly::summary).collect()
    };
    assert_eq!(summaries(&first), summaries(&second));

    let terms = |policy: &PolicyDocument| -> Vec<String> {
        policy
            .definitions()
            .iter()
            .map(|definition| definition.term().to_string())
            .collect()
    };
    assert_eq!(terms(&first), terms(&second));
}

