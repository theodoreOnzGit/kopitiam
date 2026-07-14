//! Amendment and as-at-date querying — the thing naive legal systems get
//! wrong.
//!
//! All documents here are **SYNTHETIC**. Nothing is real law.

use kopitiam_legal::{
    numbering::parse_statutory, Amendment, AmendmentOperation, AsAtDate, AsAtResult, Date,
    DocumentId, DocumentVersion, PageNumber, Provenance, Provision, ProvisionHistory, Validity,
    VerbatimText,
};

fn d(y: i32, m: u8, day: u8) -> Date {
    Date::new(y, m, day).unwrap()
}

fn at(y: i32) -> AsAtDate {
    AsAtDate::new(d(y, 6, 1))
}

/// Builds a version of s 12(1) with a given text and in-force window.
fn version(text: &str, validity: Validity) -> Provision {
    Provision::new(
        Provenance::new(
            DocumentId::new("SYNTHETIC Widget Licensing Act").unwrap(),
            DocumentVersion::Edition("SYNTHETIC 2024 Revised Edition".into()),
            parse_statutory("12(1)").unwrap(),
            PageNumber::new(2).unwrap(),
            VerbatimText::new(text).unwrap(),
        ),
        validity,
    )
}

const ORIGINAL: &str = "No person shall operate a widget without a licence.";
const AMENDED: &str = "No person shall operate a widget without a licence issued by the Authority.";

/// s 12(1) as originally enacted in 2020, substituted in 2022 by a SYNTHETIC
/// amending Act.
fn amended_history() -> ProvisionHistory {
    let mut history = ProvisionHistory::new(version(ORIGINAL, Validity::from(d(2020, 1, 1))));

    let amendment = Amendment::new(
        DocumentId::new("SYNTHETIC Widget Licensing (Amendment) Act").unwrap(),
        d(2022, 1, 1),
        AmendmentOperation::Substituted,
        Provenance::new(
            DocumentId::new("SYNTHETIC Widget Licensing (Amendment) Act").unwrap(),
            DocumentVersion::AsAt(d(2022, 1, 1)),
            parse_statutory("3").unwrap(),
            PageNumber::new(1).unwrap(),
            VerbatimText::new(
                "Section 12(1) of the principal Act is repealed and the following substituted.",
            )
            .unwrap(),
        ),
    );

    history
        .supersede(
            version(AMENDED, Validity::from(d(2022, 1, 1))).with_amendment(amendment),
        )
        .expect("the 2022 version supersedes the 2020 one");
    history
}

// ---------------------------------------------------------------------------
// The central test: the same provision, different answers on different dates.
// ---------------------------------------------------------------------------

#[test]
fn a_provision_as_at_2021_differs_from_as_at_2024() {
    let history = amended_history();

    let AsAtResult::InForce(before) = history.as_at(at(2021)) else {
        panic!("s 12(1) was in force in 2021");
    };
    assert_eq!(
        before.text(),
        ORIGINAL,
        "as at 2021 the ORIGINAL text was the law — the amendment had not commenced"
    );

    let AsAtResult::InForce(after) = history.as_at(at(2024)) else {
        panic!("s 12(1) was in force in 2024");
    };
    assert_eq!(
        after.text(),
        AMENDED,
        "as at 2024 the AMENDED text is the law"
    );

    assert_ne!(
        before.text(),
        after.text(),
        "this is the whole point: the answer depends on the date"
    );
}

#[test]
fn the_windows_tile_without_overlapping() {
    let history = amended_history();

    // Superseding the original must have CLOSED its window at the amendment's
    // commencement — otherwise both versions would claim to be in force in
    // 2023, and the provision would have said two things on one day.
    let original = &history.versions()[0];
    assert_eq!(
        original.validity().in_force_until(),
        Some(d(2022, 1, 1)),
        "the original's window must close when the amendment commences"
    );

    assert!(
        history.overlapping_versions().is_empty(),
        "a provision cannot have said two different things on the same day"
    );

    // The boundary is half-open: on the commencement day itself, the NEW text
    // is the law.
    let AsAtResult::InForce(on_commencement) =
        history.as_at(AsAtDate::new(d(2022, 1, 1)))
    else {
        panic!("in force on the commencement day");
    };
    assert_eq!(on_commencement.text(), AMENDED);

    let AsAtResult::InForce(day_before) = history.as_at(AsAtDate::new(d(2021, 12, 31))) else {
        panic!("in force the day before");
    };
    assert_eq!(day_before.text(), ORIGINAL);
}

#[test]
fn the_amendment_history_is_visible_not_just_the_current_text() {
    let history = amended_history();
    assert_eq!(history.versions().len(), 2, "both versions are retained");

    // The current version knows what changed it, and that amendment is itself
    // sourced — a reader can go and read the amending Act.
    let amendment = history.versions()[1]
        .amended_by()
        .expect("the 2022 version was made by an amending Act");
    assert_eq!(
        amendment.by().as_str(),
        "SYNTHETIC Widget Licensing (Amendment) Act"
    );
    assert_eq!(amendment.commencement(), d(2022, 1, 1));
    assert_eq!(amendment.operation(), &AmendmentOperation::Substituted);
    assert!(
        amendment.provenance().verbatim().contains("substituted"),
        "the amending words themselves must be quotable"
    );

    // The original has no amendment: it is the original.
    assert!(history.versions()[0].amended_by().is_none());
}

// ---------------------------------------------------------------------------
// The four honest answers.
// ---------------------------------------------------------------------------

#[test]
fn a_provision_not_yet_commenced_is_not_reported_as_in_force() {
    let history = amended_history();
    match history.as_at(at(2019)) {
        AsAtResult::NotYetInForce {
            earliest_commencement,
        } => assert_eq!(earliest_commencement, d(2020, 1, 1)),
        other => panic!("s 12(1) did not exist in 2019; got {other:?}"),
    }
}

#[test]
fn a_repealed_provision_reports_the_repeal_and_what_it_last_said() {
    // A provision in force 2020-2023, then repealed. Someone asking in 2024
    // about conduct in 2022 needs BOTH facts: that it is gone, and what it
    // said while it was law.
    let history = ProvisionHistory::new(version(
        ORIGINAL,
        Validity::between(d(2020, 1, 1), d(2023, 1, 1)).unwrap(),
    ));

    match history.as_at(at(2024)) {
        AsAtResult::Repealed {
            repealed_on,
            last_in_force_text,
        } => {
            assert_eq!(repealed_on, d(2023, 1, 1));
            assert_eq!(
                last_in_force_text.text(),
                ORIGINAL,
                "the text it last carried is usually exactly what the asker needs"
            );
        }
        other => panic!("expected Repealed, got {other:?}"),
    }

    // But in 2022 it was still the law.
    assert!(matches!(history.as_at(at(2022)), AsAtResult::InForce(_)));
}

#[test]
fn a_gap_in_the_record_is_reported_as_not_known_never_guessed() {
    // Our source records 2020-2021 and 2023-onwards, but nothing for 2022.
    // The honest answer for 2022 is "we do not know" — NOT "probably the
    // nearest version". A tool that cannot say "I don't know" says something
    // wrong instead.
    let mut history = ProvisionHistory::new(version(
        ORIGINAL,
        Validity::between(d(2020, 1, 1), d(2021, 1, 1)).unwrap(),
    ));
    history
        .supersede(version(AMENDED, Validity::from(d(2023, 1, 1))))
        .unwrap();

    match history.as_at(at(2022)) {
        AsAtResult::NotRecorded { recorded_windows } => {
            assert_eq!(recorded_windows.len(), 2, "the reader sees the shape of the gap");
        }
        other => panic!(
            "a hole in the record must be reported as NotRecorded, not filled in; got {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Querying without a date.
// ---------------------------------------------------------------------------

#[test]
fn querying_without_an_as_at_date_yields_a_loud_must_use_warning() {
    let history = amended_history();

    // There is NO `history.text()`. The only un-dated path hands back a
    // #[must_use] warning alongside the text, which a caller cannot silently
    // drop.
    let (provision, warning) = history.latest_known();
    assert_eq!(provision.text(), AMENDED);

    // The warning names the provision and the window of the text returned, and
    // says plainly that this may not be the law on the date the reader cares
    // about.
    assert_eq!(warning.provision(), &parse_statutory("12(1)").unwrap());
    assert_eq!(warning.validity().in_force_from(), d(2022, 1, 1));

    let message = warning.to_string();
    assert!(message.contains("WARNING"));
    assert!(
        message.contains("without an as-at date"),
        "the warning must say WHY it is a warning; got {message:?}"
    );
    assert!(
        message.contains("NOT necessarily the law"),
        "and it must say what the risk is; got {message:?}"
    );
}

// ---------------------------------------------------------------------------
// Amendments we refuse to apply.
// ---------------------------------------------------------------------------

#[test]
fn a_textual_amendment_instruction_is_recorded_but_not_applied() {
    // Real amending Acts are edit scripts: "delete 'may' and substitute
    // 'must'". Applying them mechanically is how you produce a confidently
    // wrong consolidated text. We record the instruction and decline.
    let instruction = "In section 12(1), delete the word \"may\" and substitute the word \"must\".";
    let amendment = Amendment::new(
        DocumentId::new("SYNTHETIC Amendment Act").unwrap(),
        d(2022, 1, 1),
        AmendmentOperation::TextualInstructionNotApplied {
            instruction: VerbatimText::new(instruction).unwrap(),
        },
        Provenance::new(
            DocumentId::new("SYNTHETIC Amendment Act").unwrap(),
            DocumentVersion::AsAt(d(2022, 1, 1)),
            parse_statutory("4").unwrap(),
            PageNumber::new(1).unwrap(),
            VerbatimText::new(instruction).unwrap(),
        ),
    );

    match amendment.operation() {
        AmendmentOperation::TextualInstructionNotApplied { instruction: kept } => {
            assert!(
                kept.as_str().contains("delete the word"),
                "the instruction is preserved verbatim for a human to apply"
            );
        }
        other => panic!("expected an unapplied textual instruction, got {other:?}"),
    }
    assert_eq!(
        amendment.operation().to_string(),
        "textual amendment (not applied)",
        "and it must SAY it was not applied"
    );
}

// ---------------------------------------------------------------------------
// Invariants.
// ---------------------------------------------------------------------------

#[test]
fn a_superseding_version_cannot_predate_the_one_it_supersedes() {
    let mut history = ProvisionHistory::new(version(ORIGINAL, Validity::from(d(2020, 1, 1))));
    let bad = version(AMENDED, Validity::from(d(2019, 1, 1)));
    assert!(
        history.supersede(bad).is_err(),
        "an amendment cannot commence before the provision it amends"
    );
}

#[test]
fn a_history_cannot_be_superseded_by_a_different_provision() {
    let mut history = ProvisionHistory::new(version(ORIGINAL, Validity::from(d(2020, 1, 1))));
    let other = Provision::new(
        Provenance::new(
            DocumentId::new("SYNTHETIC Widget Licensing Act").unwrap(),
            DocumentVersion::Edition("SYNTHETIC 2024 Revised Edition".into()),
            parse_statutory("13(1)").unwrap(), // <- a different provision
            PageNumber::new(2).unwrap(),
            VerbatimText::new("Something else entirely.").unwrap(),
        ),
        Validity::from(d(2022, 1, 1)),
    );
    assert!(history.supersede(other).is_err());
}
