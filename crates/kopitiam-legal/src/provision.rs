//! [`Provision`]: a located, verbatim, temporally-bounded unit of a legal
//! instrument.
//!
//! A provision is the atom of this crate. Everything else — definitions,
//! cross-references, amendments — hangs off one.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    Amendment, AsAtDate, Date, LegalError, Provenance, ProvisionId, Validity,
};

/// A single provision of a legal instrument, as it stood over one window of
/// time.
///
/// # Why every field here is mandatory
///
/// A `Provision` is only ever shown to a human who is trying to find out
/// what a document says. To be *usable* for that, and *safe* for that, it
/// must answer four questions at once:
///
/// | Question | Field |
/// |---|---|
/// | Which provision is this? | [`ProvisionId`] (in the provenance) |
/// | What did it literally say? | verbatim text (in the provenance) |
/// | Where do I go and check? | document, version, page (in the provenance) |
/// | *When* was this the law? | [`Validity`] |
///
/// Drop any one and the item becomes an unsourced assertion about the law.
/// So there is exactly one constructor, [`Provision::new`], it demands a
/// [`Provenance`] (which is itself unconstructable without document,
/// version, id, page and verbatim text) and a [`Validity`], and the fields
/// are private. There is no `Default`, no partial builder, and no
/// `Deserialize` that bypasses the constructor.
///
/// **There is deliberately no `fn text(&self) -> &str` on the crate's query
/// path.** You reach a provision's words through
/// [`crate::ProvisionHistory::as_at`], which forces you to name the date you
/// are asking about. `Provision::text` exists here only because once you
/// *hold* a specific version, you have already answered the temporal
/// question — the version you hold is the answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ProvisionRepr", into = "ProvisionRepr")]
pub struct Provision {
    provenance: Provenance,
    validity: Validity,
    /// The provision's own heading/marginal note, if it has one ("Duty to
    /// register"). Genuinely optional: most subsections have none.
    heading: Option<String>,
    /// The Part of the instrument this provision sits in, if any.
    ///
    /// # Why the Part is context and not identity
    ///
    /// It is tempting to make a provision's id `Part II > s 12 > (3)`. That is
    /// **wrong**, and getting it wrong breaks the crate:
    ///
    /// Section numbers run **uniquely across a whole Act**, not per-Part. `s 7`
    /// is `s 7` wherever it sits, and every citation, every cross-reference and
    /// every judgment that ever refers to it says "section 7" — never "Part I,
    /// section 7". So if the Part is baked into the identity, then "subject to
    /// section 7" resolves to `s 7`, the stored provision is `Part I, s 7`, the
    /// two do not match, and **every cross-reference in the Act dangles**. (That
    /// is not hypothetical — it is exactly what the first version of this crate
    /// did, and the test suite caught it.)
    ///
    /// So the Part rides alongside as context: it is displayed, it scopes "in
    /// this Part" definitions ([`crate::DefinitionScope::Part`]), and it is
    /// *not* part of [`ProvisionId`].
    part: Option<crate::Numeral>,
    /// The amendment that produced *this version* of the provision. `None`
    /// for the original enactment.
    amended_by: Option<Amendment>,
}

impl Provision {
    /// The only constructor. Provenance and validity are not optional and
    /// never will be.
    pub fn new(provenance: Provenance, validity: Validity) -> Self {
        Self {
            provenance,
            validity,
            heading: None,
            part: None,
            amended_by: None,
        }
    }

    /// Records the Part of the instrument this provision sits in. See the
    /// `part` field's docs for why this is context rather than identity.
    pub fn with_part(mut self, part: crate::Numeral) -> Self {
        self.part = Some(part);
        self
    }

    /// Attaches the provision's heading / marginal note.
    pub fn with_heading(mut self, heading: impl Into<String>) -> Self {
        self.heading = Some(heading.into());
        self
    }

    /// Records the amendment that produced this version.
    pub fn with_amendment(mut self, amendment: Amendment) -> Self {
        self.amended_by = Some(amendment);
        self
    }

    /// Narrows this version's in-force window (used when a later amendment
    /// supersedes it — see [`crate::ProvisionHistory::supersede`]).
    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = validity;
        self
    }

    pub fn id(&self) -> &ProvisionId {
        self.provenance.provision()
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    pub fn validity(&self) -> Validity {
        self.validity
    }

    pub fn heading(&self) -> Option<&str> {
        self.heading.as_deref()
    }

    /// The Part this provision sits in, if any. Context, not identity.
    pub fn part(&self) -> Option<crate::Numeral> {
        self.part
    }

    pub fn amended_by(&self) -> Option<&Amendment> {
        self.amended_by.as_ref()
    }

    /// The literal words of this version. Always the source text, never a
    /// paraphrase, summary, or interpretation.
    pub fn text(&self) -> &str {
        self.provenance.verbatim()
    }

    /// Whether this version was in force on the given date.
    pub fn in_force_at(&self, as_at: AsAtDate) -> bool {
        self.validity.covers(as_at)
    }

    /// A citation a human can follow to the original.
    pub fn citation(&self) -> String {
        self.provenance.citation()
    }

    pub(crate) fn close_validity_at(&mut self, until: Date) -> Result<(), LegalError> {
        self.validity.close_at(until)
    }
}

impl fmt::Display for Provision {
    /// Renders the provision the way it should always be presented: the
    /// citation, the temporal window, then the verbatim text. Never the text
    /// alone — text without its date and source is exactly the artefact this
    /// crate exists to stop producing.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.citation(), self.validity)?;
        if let Some(heading) = &self.heading {
            write!(f, "\n{heading}")?;
        }
        write!(f, "\n{}", self.text())
    }
}

/// Serde shadow for [`Provision`], closing the same deserialization back
/// door described in [`crate::provenance`]: a derived `Deserialize` would
/// rebuild the struct field-by-field and skip the constructor. Because
/// `provenance` and `validity` are non-`Option` here, serde itself rejects
/// their absence, and their own validating deserialize paths reject
/// malformed contents.
#[derive(Serialize, Deserialize)]
struct ProvisionRepr {
    provenance: Provenance,
    validity: Validity,
    #[serde(default)]
    heading: Option<String>,
    #[serde(default)]
    part: Option<crate::Numeral>,
    #[serde(default)]
    amended_by: Option<Amendment>,
}

impl TryFrom<ProvisionRepr> for Provision {
    type Error = LegalError;
    fn try_from(r: ProvisionRepr) -> Result<Self, Self::Error> {
        let mut provision = Provision::new(r.provenance, r.validity);
        provision.heading = r.heading;
        provision.part = r.part;
        provision.amended_by = r.amended_by;
        Ok(provision)
    }
}

impl From<Provision> for ProvisionRepr {
    fn from(p: Provision) -> Self {
        Self {
            provenance: p.provenance,
            validity: p.validity,
            heading: p.heading,
            part: p.part,
            amended_by: p.amended_by,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::synthetic_provision;
    use crate::{Date, DocumentId, DocumentVersion, PageNumber, VerbatimText};

    #[test]
    fn a_provision_carries_its_source_and_its_dates() {
        let p = synthetic_provision("12(3)", "A widget must be licensed.", 2020);
        assert_eq!(p.text(), "A widget must be licensed.");
        assert_eq!(p.id().to_string(), "s 12(3)");
        assert_eq!(p.provenance().page().get(), 1);
        assert!(p.in_force_at(crate::AsAtDate::new(Date::new(2021, 6, 1).unwrap())));
        assert!(!p.in_force_at(crate::AsAtDate::new(Date::new(2019, 6, 1).unwrap())));
    }

    #[test]
    fn display_never_shows_text_without_its_source_and_window() {
        let p = synthetic_provision("12(3)", "A widget must be licensed.", 2020);
        let rendered = p.to_string();
        assert!(rendered.contains("s 12(3)"), "citation");
        assert!(rendered.contains("p 1"), "page");
        assert!(rendered.contains("in force from 2020-01-01"), "window");
        assert!(rendered.contains("A widget must be licensed."), "verbatim");
    }

    /// The type-system claim, asserted. Building a `Provision` requires a
    /// `Provenance` and a `Validity`; a `Provenance` requires a document, a
    /// version, an id, a page and non-empty verbatim text. Each of those
    /// components rejects its empty/zero form, so there is no sequence of
    /// public calls that yields an un-sourced or undated provision.
    #[test]
    fn there_is_no_public_path_to_an_unsourced_provision() {
        // Every ingredient must be individually well-formed...
        assert!(VerbatimText::new("").is_err());
        assert!(PageNumber::new(0).is_err());
        assert!(DocumentId::new("").is_err());

        // ...and only then can a Provenance exist, and only then a Provision.
        let provenance = Provenance::new(
            DocumentId::new("SYNTHETIC Act").unwrap(),
            DocumentVersion::Edition("2020 Rev Ed".into()),
            crate::numbering::parse_statutory("1").unwrap(),
            PageNumber::new(1).unwrap(),
            VerbatimText::new("text").unwrap(),
        );
        let p = Provision::new(provenance, Validity::from(Date::new(2020, 1, 1).unwrap()));
        assert_eq!(p.text(), "text");

        // There is no `Provision::default()`, no `Provision { .. }` literal
        // from outside this module (fields are private), and no constructor
        // that omits provenance or validity. Those are compile-time facts;
        // `tests/type_safety.rs` pins them with `trybuild`-style prose and
        // the crate's public API surface.
    }

    #[test]
    fn round_trips_through_json_and_rejects_a_provision_with_no_source() {
        let p = synthetic_provision("12(3)", "A widget must be licensed.", 2020);
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Provision>(&json).unwrap(), p);

        // No provenance at all.
        assert!(
            serde_json::from_str::<Provision>(r#"{"validity":{"in_force_from":"2020-01-01","in_force_until":null}}"#)
                .is_err(),
            "a provision without provenance must not deserialize"
        );
        // No validity at all.
        let no_validity = serde_json::to_value(&p).unwrap();
        let mut obj = no_validity.as_object().unwrap().clone();
        obj.remove("validity");
        assert!(
            serde_json::from_value::<Provision>(serde_json::Value::Object(obj)).is_err(),
            "a provision without an in-force window must not deserialize"
        );
    }
}
