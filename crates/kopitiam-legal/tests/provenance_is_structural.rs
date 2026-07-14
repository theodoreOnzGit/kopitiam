//! **The type-system claim, asserted.**
//!
//! This crate's central promise is that *an extracted item without provenance
//! is not representable*. This file pins that down from **outside** the crate
//! — which is the only place the claim actually matters, since inside the crate
//! private fields are visible.
//!
//! # What "enforced by the type system" means here, precisely
//!
//! For a `Provision` to exist, a `Provenance` must exist. For a `Provenance` to
//! exist, all five of `DocumentId`, `DocumentVersion`, `ProvisionId`,
//! `PageNumber` and `VerbatimText` must exist. Each of those rejects its
//! degenerate form. There is:
//!
//! * no `Provision::default()` / `Provenance::default()` — `Default` is not
//!   implemented, so `..Default::default()` cannot fill in a missing field;
//! * no public field on either — so no struct literal can be written outside
//!   the crate;
//! * no constructor that omits provenance or validity — `Provision::new` takes
//!   both, by value;
//! * no `From<String> for VerbatimText` — the only way in is the validating
//!   `VerbatimText::new`;
//! * no derived `Deserialize` that rebuilds the struct field-by-field — both
//!   types deserialize through a validating shadow, so a hand-edited JSON file
//!   cannot smuggle one in either.
//!
//! The compile-time half of that is asserted by *the absence of failures to
//! compile this file*; the runtime half is asserted below. The commented-out
//! lines are the important ones: each is a construction path that a careless
//! implementation would have left open, and each **does not compile**.

use kopitiam_legal::{
    numbering::parse_statutory, Date, DocumentId, DocumentVersion, LegalError, PageNumber,
    Provenance, Provision, Validity, VerbatimText,
};

fn valid_provenance() -> Provenance {
    Provenance::new(
        DocumentId::new("SYNTHETIC Widget Licensing Act").unwrap(),
        DocumentVersion::Edition("SYNTHETIC 2020 Revised Edition".into()),
        parse_statutory("12(3)").unwrap(),
        PageNumber::new(14).unwrap(),
        VerbatimText::new("A person must not operate a widget without a licence.").unwrap(),
    )
}

// ---------------------------------------------------------------------------
// Each provenance component rejects its degenerate form.
// ---------------------------------------------------------------------------

#[test]
fn verbatim_text_cannot_be_empty() {
    assert!(matches!(
        VerbatimText::new(""),
        Err(LegalError::MissingProvenance { .. })
    ));
    assert!(
        VerbatimText::new("  \t\n ").is_err(),
        "whitespace is not source text"
    );
}

#[test]
fn there_is_no_page_zero() {
    assert!(matches!(
        PageNumber::new(0),
        Err(LegalError::MissingProvenance { .. })
    ));
}

#[test]
fn a_document_must_be_identified() {
    assert!(DocumentId::new("").is_err());
    assert!(DocumentId::new("   ").is_err());
}

// ---------------------------------------------------------------------------
// The composite guarantee.
// ---------------------------------------------------------------------------

#[test]
fn a_provision_always_answers_where_it_came_from_and_when_it_was_law() {
    let p = Provision::new(valid_provenance(), Validity::from(Date::new(2020, 1, 1).unwrap()));

    // Which provision, what it said, where to check, and when it was the law.
    assert_eq!(p.id().to_string(), "s 12(3)");
    assert_eq!(
        p.text(),
        "A person must not operate a widget without a licence."
    );
    assert_eq!(p.provenance().page().get(), 14);
    assert_eq!(p.provenance().document().as_str(), "SYNTHETIC Widget Licensing Act");
    assert_eq!(p.validity().in_force_from(), Date::new(2020, 1, 1).unwrap());

    // And it renders as a citation, never as bare text.
    let rendered = p.to_string();
    assert!(rendered.contains("s 12(3)"));
    assert!(rendered.contains("p 14"));
    assert!(rendered.contains("in force from 2020-01-01"));
}

/// The constructions that a careless implementation would have allowed.
///
/// **Every commented-out line below is a compile error.** They are kept as
/// documentation of exactly which doors are shut, because a future contributor
/// deriving `Default` on `Provision` "for convenience" would silently reopen
/// all of them, and this is where they should find out why not.
#[test]
fn there_is_no_public_path_to_an_unsourced_or_undated_provision() {
    // No Default: cannot conjure one from nothing.
    //
    //     let p = Provision::default();
    //     ^ error[E0599]: no function or associated item named `default` found
    //
    // No public fields: cannot struct-literal one, nor omit a field.
    //
    //     let p = Provision { validity: Validity::from(date) };
    //     ^ error[E0451]: field `provenance` of struct `Provision` is private
    //
    // No constructor that omits provenance:
    //
    //     let p = Provision::new(Validity::from(date));
    //     ^ error[E0061]: this function takes 2 arguments but 1 was supplied
    //
    // No un-validated way to make source text:
    //
    //     let t: VerbatimText = String::new().into();
    //     ^ error[E0277]: the trait bound `VerbatimText: From<String>` is not satisfied
    //
    // No `..Default::default()` escape on Provenance either:
    //
    //     let pv = Provenance { page, ..Default::default() };
    //     ^ error[E0433] / E0451
    //
    // The one path that DOES exist requires everything:
    let _ = Provision::new(valid_provenance(), Validity::from(Date::new(2020, 1, 1).unwrap()));
}

// ---------------------------------------------------------------------------
// The deserialization back door — the one people forget.
// ---------------------------------------------------------------------------

#[test]
fn serde_cannot_smuggle_in_an_unsourced_provision() {
    // A derived `Deserialize` would rebuild the struct field-by-field and skip
    // every constructor above, so a hand-edited JSON file could inject a
    // provision with no source text. The validating shadow closes it.

    let no_provenance = r#"{"validity":{"in_force_from":"2020-01-01","in_force_until":null}}"#;
    assert!(
        serde_json::from_str::<Provision>(no_provenance).is_err(),
        "a provision with no provenance must not deserialize"
    );

    let empty_verbatim = r#"{
        "provenance": {
            "document": "SYNTHETIC Act",
            "version": {"edition": "2020 Rev Ed"},
            "provision": {"components": [{"section": {"number": 12, "suffix": null}}]},
            "page": 14,
            "verbatim": ""
        },
        "validity": {"in_force_from": "2020-01-01", "in_force_until": null}
    }"#;
    assert!(
        serde_json::from_str::<Provision>(empty_verbatim).is_err(),
        "empty verbatim text must be rejected on the deserialize path too"
    );

    let page_zero = empty_verbatim.replace(r#""verbatim": """#, r#""verbatim": "text""#)
        .replace(r#""page": 14"#, r#""page": 0"#);
    assert!(
        serde_json::from_str::<Provision>(&page_zero).is_err(),
        "page 0 must be rejected on the deserialize path too"
    );
}

#[test]
fn a_valid_provision_still_round_trips() {
    // The guard must not be so tight that legitimate data cannot survive a
    // save/load cycle — persistence is the whole point of the semantic runtime.
    let p = Provision::new(valid_provenance(), Validity::from(Date::new(2020, 1, 1).unwrap()));
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(serde_json::from_str::<Provision>(&json).unwrap(), p);
}

#[test]
fn an_invalid_in_force_window_is_rejected_at_construction_and_on_deserialize() {
    // A provision cannot cease to be in force before it commenced.
    assert!(
        Validity::between(Date::new(2024, 1, 1).unwrap(), Date::new(2020, 1, 1).unwrap()).is_err()
    );
    let inverted = r#"{"in_force_from":"2024-01-01","in_force_until":"2020-01-01"}"#;
    assert!(serde_json::from_str::<Validity>(inverted).is_err());
}
