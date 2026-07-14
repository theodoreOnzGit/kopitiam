//! Provenance: where a piece of extracted text came from, and what it
//! literally said.
//!
//! # This is the most important module in the crate
//!
//! CLAUDE.md's Scientific Standards require that "scientific software should
//! always remain explainable". For legal material the bar is higher still,
//! because a legal document is an *operative instrument*: a statute, a
//! contract, a judgment actually does things to people. An extracted
//! assertion that cannot be traced back to a page and a verbatim sentence is
//! not merely unexplainable — it is a claim about the law with no evidence
//! behind it, and someone may act on it.
//!
//! So the rule in this crate is not "please attach provenance". It is:
//!
//! > **An extracted item without provenance must not be *representable*.**
//!
//! That is enforced three ways, and all three are necessary:
//!
//! 1. **Private fields.** [`Provenance`]'s fields are private, so no caller
//!    can struct-literal one into existence.
//! 2. **One fallible constructor.** [`Provenance::new`] is the only way in,
//!    it demands every component, and it *rejects* empty verbatim text. There
//!    is no `Default`, no builder with optional setters, and no
//!    `..Default::default()` escape hatch.
//! 3. **A validating deserialize path.** This is the one people forget.
//!    `#[derive(Deserialize)]` on a struct with private fields will happily
//!    reconstruct it field-by-field, *bypassing the constructor entirely* —
//!    so a hand-edited JSON file could inject a `Provenance` with empty
//!    verbatim text and defeat the whole design. We therefore deserialize
//!    through a shadow struct and re-run the constructor's validation via
//!    `#[serde(try_from = ...)]`. See [`ProvenanceRepr`].
//!
//! Everything downstream — [`crate::Provision`], [`crate::Definition`],
//! [`crate::CrossReference`], [`crate::Anomaly`] — holds a `Provenance` by
//! value, so the guarantee propagates: if it exists in this crate, you can
//! find out where it came from and read the original words.

use std::fmt;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::{Date, LegalError, ProvisionId};

/// Identifies the *document* an item was extracted from.
///
/// Deliberately an opaque non-empty string rather than a path: the same
/// instrument may be read from a PDF today and a database tomorrow, and the
/// identity of "the Companies Act" should not change when the file moves.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct DocumentId(String);

impl DocumentId {
    pub fn new(id: impl Into<String>) -> Result<Self, LegalError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(LegalError::MissingProvenance {
                what: "document id",
            });
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for DocumentId {
    type Error = LegalError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<DocumentId> for String {
    fn from(id: DocumentId) -> String {
        id.0
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// *Which version of the document* was read.
///
/// # This is not the same thing as when the provision was in force
///
/// The distinction is subtle, universally muddled, and matters. There are
/// two independent time axes:
///
/// * **`DocumentVersion` — the edition of the paper in front of you.** "The
///   2020 Revised Edition of the Companies Act", or "the contract as
///   endorsed on 3 March 2021". This is a property of the *source artefact*.
/// * **[`crate::Validity`] — when the provision itself had legal effect.**
///   This is a property of the *law*.
///
/// They come apart constantly. A 2020 Revised Edition is a *snapshot*: it
/// prints the text as at 2020 and generally does **not** tell you what s 12
/// said in 2018. So if you hold the 2020 edition and someone asks an as-at
/// 2018 question, the honest answer is often "this source cannot tell you" —
/// and the only way to *give* that honest answer is to have kept the two
/// axes apart. Collapsing them into one "date" field is precisely the bug
/// that makes a legal tool confidently wrong.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentVersion {
    /// A named printed/published edition, e.g. `"2020 Revised Edition"`.
    Edition(String),
    /// The document as published or current at a given date.
    AsAt(Date),
}

impl fmt::Display for DocumentVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Edition(name) => f.write_str(name),
            Self::AsAt(date) => write!(f, "as at {date}"),
        }
    }
}

/// A 1-based page number. `NonZeroUsize` because there is no page 0, and a
/// zero here would almost always mean "we didn't actually know".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PageNumber(NonZeroUsize);

impl PageNumber {
    pub fn new(page: usize) -> Result<Self, LegalError> {
        NonZeroUsize::new(page)
            .map(Self)
            .ok_or(LegalError::MissingProvenance { what: "page number" })
    }

    pub fn get(&self) -> usize {
        self.0.get()
    }
}

impl fmt::Display for PageNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p {}", self.0)
    }
}

/// The literal words of the source, unmodified.
///
/// Guaranteed non-empty. Whitespace inside is preserved as extracted: we do
/// not normalise, re-wrap, "clean up", or otherwise touch the drafter's
/// text, because the reader's whole reason for trusting this crate is that
/// what it shows them is what the document says.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct VerbatimText(String);

impl VerbatimText {
    pub fn new(text: impl Into<String>) -> Result<Self, LegalError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(LegalError::MissingProvenance {
                what: "verbatim source text",
            });
        }
        Ok(Self(text))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for VerbatimText {
    type Error = LegalError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<VerbatimText> for String {
    fn from(t: VerbatimText) -> String {
        t.0
    }
}

impl fmt::Display for VerbatimText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Everything needed to send a reader to the original words: which document,
/// which version of it, which provision, which page, and what it said.
///
/// Fields are private and there is exactly one constructor. See the module
/// docs for why that is the load-bearing design decision in this crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "ProvenanceRepr", into = "ProvenanceRepr")]
pub struct Provenance {
    document: DocumentId,
    version: DocumentVersion,
    provision: ProvisionId,
    page: PageNumber,
    verbatim: VerbatimText,
}

impl Provenance {
    /// The only way to construct a [`Provenance`]. Every component is
    /// required; none is defaultable.
    pub fn new(
        document: DocumentId,
        version: DocumentVersion,
        provision: ProvisionId,
        page: PageNumber,
        verbatim: VerbatimText,
    ) -> Self {
        Self {
            document,
            version,
            provision,
            page,
            verbatim,
        }
    }

    pub fn document(&self) -> &DocumentId {
        &self.document
    }

    pub fn version(&self) -> &DocumentVersion {
        &self.version
    }

    pub fn provision(&self) -> &ProvisionId {
        &self.provision
    }

    pub fn page(&self) -> PageNumber {
        self.page
    }

    /// The literal source words. This is what a reader should be shown.
    pub fn verbatim(&self) -> &str {
        self.verbatim.as_str()
    }

    /// A human-readable citation, e.g.
    /// `Companies Act (2020 Revised Edition), s 12(3)(a), p 14`.
    ///
    /// This is what any output of this crate should carry next to it. It is
    /// deliberately not a legal citation format for any particular court's
    /// style guide — it is a *pointer for a human to go and check*, which is
    /// the only thing this crate is entitled to produce.
    pub fn citation(&self) -> String {
        format!(
            "{} ({}), {}, {}",
            self.document, self.version, self.provision, self.page
        )
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.citation())
    }
}

/// Serde shadow for [`Provenance`].
///
/// Exists solely to close the deserialization back door: a derived
/// `Deserialize` on `Provenance` would rebuild it field-by-field and skip
/// [`Provenance::new`]'s guarantees. Routing through this struct means a
/// malformed JSON document is *rejected* rather than silently producing an
/// un-sourced item. The component newtypes ([`VerbatimText`], [`PageNumber`],
/// [`DocumentId`]) each re-validate on their own deserialize path too, so the
/// invariants hold even though this shadow is a plain struct.
#[derive(Serialize, Deserialize)]
struct ProvenanceRepr {
    document: DocumentId,
    version: DocumentVersion,
    provision: ProvisionId,
    page: PageNumber,
    verbatim: VerbatimText,
}

impl TryFrom<ProvenanceRepr> for Provenance {
    type Error = LegalError;

    fn try_from(repr: ProvenanceRepr) -> Result<Self, Self::Error> {
        // Re-run the constructor rather than assigning fields directly, so
        // that any future invariant added to `new` is automatically enforced
        // on the deserialize path as well.
        Ok(Provenance::new(
            repr.document,
            repr.version,
            repr.provision,
            repr.page,
            repr.verbatim,
        ))
    }
}

impl From<Provenance> for ProvenanceRepr {
    fn from(p: Provenance) -> Self {
        Self {
            document: p.document,
            version: p.version,
            provision: p.provision,
            page: p.page,
            verbatim: p.verbatim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbering::parse_statutory;

    fn provenance() -> Provenance {
        Provenance::new(
            DocumentId::new("SYNTHETIC Widgets Act").unwrap(),
            DocumentVersion::Edition("2020 Revised Edition".into()),
            parse_statutory("12(3)").unwrap(),
            PageNumber::new(14).unwrap(),
            VerbatimText::new("A person must not operate a widget without a licence.").unwrap(),
        )
    }

    #[test]
    fn empty_verbatim_text_is_rejected() {
        assert!(VerbatimText::new("").is_err());
        assert!(VerbatimText::new("   \n ").is_err(), "whitespace is not text");
        assert!(VerbatimText::new("x").is_ok());
    }

    #[test]
    fn there_is_no_page_zero() {
        assert!(PageNumber::new(0).is_err());
        assert_eq!(PageNumber::new(1).unwrap().get(), 1);
    }

    #[test]
    fn empty_document_id_is_rejected() {
        assert!(DocumentId::new("").is_err());
        assert!(DocumentId::new("  ").is_err());
    }

    #[test]
    fn citation_points_a_human_at_the_source() {
        let p = provenance();
        assert_eq!(
            p.citation(),
            "SYNTHETIC Widgets Act (2020 Revised Edition), s 12(3), p 14"
        );
        assert_eq!(
            p.verbatim(),
            "A person must not operate a widget without a licence."
        );
    }

    #[test]
    fn round_trips_through_json() {
        let p = provenance();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Provenance>(&json).unwrap(), p);
    }

    /// The deserialization back door: a derived `Deserialize` would rebuild
    /// `Provenance` field-by-field and let a hand-edited file smuggle in an
    /// item with no source text. The `try_from` shadow closes it.
    #[test]
    fn serde_cannot_smuggle_in_an_unsourced_provenance() {
        let empty_verbatim = r#"{
            "document": "SYNTHETIC Widgets Act",
            "version": {"edition": "2020 Revised Edition"},
            "provision": {"components": [{"section": {"number": 12, "suffix": null}}]},
            "page": 14,
            "verbatim": ""
        }"#;
        assert!(
            serde_json::from_str::<Provenance>(empty_verbatim).is_err(),
            "empty verbatim text must be rejected on the deserialize path too"
        );

        let page_zero = r#"{
            "document": "SYNTHETIC Widgets Act",
            "version": {"edition": "2020 Revised Edition"},
            "provision": {"components": [{"section": {"number": 12, "suffix": null}}]},
            "page": 0,
            "verbatim": "text"
        }"#;
        assert!(serde_json::from_str::<Provenance>(page_zero).is_err());

        let missing_page = r#"{
            "document": "SYNTHETIC Widgets Act",
            "version": {"edition": "2020 Revised Edition"},
            "provision": {"components": []},
            "verbatim": "text"
        }"#;
        assert!(
            serde_json::from_str::<Provenance>(missing_page).is_err(),
            "provenance components are mandatory, not defaultable"
        );
    }

    #[test]
    fn document_version_and_provision_validity_are_different_axes() {
        // The 2020 edition is a snapshot of the paper; it says nothing about
        // what the law was in 2018. Keeping these apart is what lets us
        // answer "this source cannot tell you" instead of guessing.
        let p = provenance();
        assert_eq!(
            p.version(),
            &DocumentVersion::Edition("2020 Revised Edition".into())
        );
    }
}
