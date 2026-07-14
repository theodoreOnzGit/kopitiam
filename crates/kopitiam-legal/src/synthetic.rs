//! **SYNTHETIC legal documents, invented wholesale for testing.**
//!
//! # Read this before using anything in this module
//!
//! Every document here is **fictional**. The "Widget Licensing Act 2020" does
//! not exist. The "Synthetic Court of Appeal" does not exist. No provision,
//! section number, definition, date or case name in this module corresponds to
//! any real law of any real jurisdiction, and none of it should be cited,
//! quoted, or relied on for any purpose whatsoever.
//!
//! This is a deliberate and load-bearing choice, not laziness. The obvious way
//! to test a legal extractor is to feed it a real statute. The problem is that
//! a *plausible-looking but subtly wrong* rendering of a real Act — a section
//! that does not exist, a definition off by one word, a repeal date a year out
//! — is genuinely dangerous, because someone may find it and rely on it. A
//! fabricated section of a real Act is worse than no section at all.
//!
//! So the fixtures are transparently fake. Every document title begins with
//! `SYNTHETIC`, the jurisdiction is invented, and the subject matter (widgets)
//! is chosen to be unmistakably not-law. If one of these strings ever escapes
//! into a report, it should be immediately obvious that it is not real.
//!
//! The *structures* they exercise, however, are entirely real: the
//! Singapore/Commonwealth `12.—(1)` opener, inserted sections (`12A`), quoted
//! definitions with `means`/`includes`, "subject to"/"notwithstanding"
//! cross-references, amendment and repeal. That is the point — real structure,
//! fake content.

use crate::{
    numbering::parse_statutory, AsAtDate, Date, DocumentId, DocumentVersion, InstrumentKind,
    PageNumber, Provenance, Provision, Validity, VerbatimText,
};

/// A synthetic provision, for unit tests that need a well-formed [`Provision`]
/// without the ceremony of a full ingest.
pub fn synthetic_provision(label: &str, text: &str, in_force_year: i32) -> Provision {
    let provenance = Provenance::new(
        DocumentId::new("SYNTHETIC Widget Licensing Act").expect("non-empty"),
        DocumentVersion::Edition("SYNTHETIC 2020 Revised Edition".into()),
        parse_statutory(label).expect("test label parses"),
        PageNumber::new(1).expect("page 1"),
        VerbatimText::new(text).expect("non-empty test text"),
    );
    Provision::new(
        provenance,
        Validity::from(Date::new(in_force_year, 1, 1).expect("valid date")),
    )
}

/// Shorthand for an as-at date on 1 June of a year.
pub fn at(year: i32) -> AsAtDate {
    AsAtDate::new(Date::new(year, 6, 1).expect("valid date"))
}

/// The document id used by the synthetic Act.
pub fn act_id() -> DocumentId {
    DocumentId::new("SYNTHETIC Widget Licensing Act").expect("non-empty")
}

/// The document version used by the synthetic Act.
pub fn act_version() -> DocumentVersion {
    DocumentVersion::Edition("SYNTHETIC 2020 Revised Edition".into())
}

/// The kind used by the synthetic Act.
pub fn act_kind() -> InstrumentKind {
    InstrumentKind::Act {
        short_title: "SYNTHETIC Widget Licensing Act".into(),
        act_number: Some("SYNTHETIC Act 1 of 2020".into()),
    }
}

/// **A SYNTHETIC statute.** Not real law. Invented for testing.
///
/// Exercises, deliberately:
///
/// * the Singapore `12.—(1)` section opener and bracketed sub-levels;
/// * an **inserted section** (`12A`), which must sort between 12 and 13;
/// * a **definition that overrides ordinary English** — a `dwelling-house`
///   here *includes a houseboat*, which it does not in ordinary usage. Any
///   extractor that hands a reader s 12 without this definition has misled
///   them;
/// * a **narrower, section-scoped definition** of the same term in s 14, which
///   must override the Act-wide one *inside s 14 only*;
/// * `subject to` and `notwithstanding` cross-references, one of which
///   (`section 99`) **dangles** and must be reported;
/// * a nested `12(3)(a)(ii)` provision, for numbering;
/// * an ambiguous, unparseable provision label, which must survive with its
///   text rather than being dropped or guessed.
///
/// Returned as `(page_number, page_text)` pairs so ingestion sees real page
/// provenance.
pub fn widget_act_pages() -> Vec<(usize, &'static str)> {
    vec![
        (
            1,
            r#"PART I
Interpretation
2.—(1) In this Act, unless the context otherwise requires —
"dwelling-house" includes a houseboat, a caravan and any structure occupied as a residence, whether or not affixed to land;
"widget" means a mechanically actuated device of a kind prescribed by the Minister;
"authorised officer" means a person appointed under section 4;
Appointment of authorised officers
4. The Minister may appoint any public officer to be an authorised officer for the purposes of this Act.
Restriction on operation
7. No person shall operate a widget in a dwelling-house except in accordance with section 12.
"#,
        ),
        (
            2,
            r#"PART II
Licensing of widgets
12.—(1) Subject to section 7, no person shall operate a widget without a licence granted under this Part.
(2) An application for a licence shall be made to an authorised officer in the prescribed form.
(3) The authorised officer may grant a licence if satisfied that —
(a) the applicant is a fit and proper person; and
(b) the widget complies with such requirements as may be prescribed, including —
(i) requirements as to safety; and
(ii) requirements as to noise emitted in a dwelling-house.
Offences
12A.—(1) Notwithstanding subsection (2), a person who operates a widget without a licence shall be guilty of an offence.
(2) It shall be a defence for a person charged under subsection (1) to prove that the widget was not operated in a dwelling-house.
Special provision for houseboats
14.—(1) In this section, "dwelling-house" means a houseboat only.
(2) A licence granted under section 12 in respect of a dwelling-house shall be subject to section 99.
"#,
        ),
    ]
}

/// **A SYNTHETIC contract.** Not a real agreement. Invented for testing.
///
/// Exercises decimal clause numbering (`1.2.3`), an instrument-wide definition
/// in a contract rather than a statute, and a cross-reference by clause.
pub fn services_agreement_pages() -> Vec<(usize, &'static str)> {
    vec![(
        1,
        r#"1. Interpretation
1.1 In this Agreement, "Services" means the widget maintenance services described in Schedule 1.
1.2 In this Agreement, "Business Day" means a day other than a Saturday, Sunday or public holiday.
2. Provision of Services
2.1 The Supplier shall provide the Services in accordance with clause 3.
2.2 The Supplier shall invoice the Customer within 5 Business Days.
3. Standard of Service
3.1 The Supplier shall perform the Services with reasonable skill and care.
"#,
    )]
}

/// **A SYNTHETIC judgment.** Not a real case. No real court decided this.
///
/// Exercises `[47]`-style paragraph numbering and a cited authority.
pub fn judgment_pages() -> Vec<(usize, &'static str)> {
    vec![(
        1,
        r#"[1] This is an appeal against the decision of the court below concerning the operation of a widget.
[2] The appellant contends that a houseboat is not a dwelling-house within the meaning of the SYNTHETIC Widget Licensing Act.
[3] In our judgment that contention is untenable, having regard to the definition in section 2.
[4] We would add, obiter, that the position might differ were the houseboat not occupied as a residence.
"#,
    )]
}
