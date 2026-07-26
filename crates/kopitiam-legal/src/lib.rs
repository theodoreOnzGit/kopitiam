//! # KOPITIAM legal engine
//!
//! Turns legal documents into structured, provenance-carrying knowledge.
//!
//! ---
//!
//! # ⚠ THIS CRATE DOES NOT GIVE LEGAL ADVICE
//!
//! **It does not interpret the law. It does not tell you what a document means
//! for you. It does not draw conclusions about legal effect, liability,
//! obligation, or entitlement, and it never will.**
//!
//! What it does is narrower and, done properly, far more useful:
//!
//! > It **locates and extracts what a document says**, verbatim, with a
//! > citation, so that a human can read it.
//!
//! The distinction is the entire crate. There is a world of difference between
//!
//! * *"Section 12(3) says this. Here are its exact words. It is on page 14 of
//!   the 2020 Revised Edition. It has been in force since 1 January 2021. The
//!   word 'dwelling-house' in it is defined by section 2 of the same Act, and
//!   that definition is not the ordinary English one — here it is. Go and
//!   read them."*
//!
//! and
//!
//! * *"You are liable."*
//!
//! The first is a research tool. The second is legal advice, it is outside this
//! crate's competence, and in many jurisdictions producing it is the
//! unauthorised practice of law. **If you find yourself extending this crate to
//! output a conclusion about legal effect, stop.** Output the provision and its
//! location, and let a qualified human judge.
//!
//! This is not defensive boilerplate. It is a design constraint that shows up
//! in the types: there is no `Conclusion`, no `Liability`, no `AppliesTo`, and
//! [`Holding`] — whether a paragraph of a judgment is binding ratio or
//! non-binding obiter — **cannot be set by this crate at all**, only by a named
//! human ([`crate::judgment`] explains why).
//!
//! ---
//!
//! # The four design commitments
//!
//! ## 1. Provenance is structural, not optional
//!
//! Every extracted item carries the document, its version, the provision id,
//! the **page**, and the **verbatim source text**. This is not a convention; it
//! is enforced by the type system. [`Provenance`] has private fields, one
//! fallible constructor, and a validating deserialize path, so **an un-sourced
//! extracted item is not representable**. See [`provenance`] — that module is
//! the single most important design decision in the crate.
//!
//! ## 2. The law is temporal, and that is modelled as the primary interface
//!
//! *"What does section 12 say?"* is **not a well-formed question.** Section 12
//! said one thing in 2018, was amended in 2021, and may since have been
//! repealed. The only well-formed question is *"what did section 12 say **as at**
//! date D?"*, so that is the only question this crate's API lets you ask:
//! [`Instrument::provision_as_at`] takes an [`AsAtDate`], and there is
//! deliberately no un-dated accessor that returns text. The single escape hatch,
//! [`ProvisionHistory::latest_known`], hands back a `#[must_use]`
//! [`TemporalWarning`] you cannot quietly ignore. See [`temporal`].
//!
//! ## 3. Definitions override ordinary meaning, and are resolved against
//!
//! When an instrument says `"dwelling-house" includes a houseboat`, then within
//! that instrument a houseboat *is* a dwelling-house, and a reader applying
//! their ordinary intuitions is wrong. Extracting s 12 verbatim is *correct and
//! still misleading* if the definition is not surfaced with it. So
//! [`Dictionary::resolve`] resolves a term against the document's own
//! definitions — scoped to where the word is used and to the date asked about,
//! because definitions are themselves scoped and amended. See [`definition`];
//! this is the highest-value thing in the crate.
//!
//! ## 4. Ambiguity is surfaced, never guessed away
//!
//! If a clause is ambiguous, contradictory, cross-references something that
//! does not exist, or cannot be parsed, it is reported **as such, with its
//! original text** — as an [`Anomaly`]. *"I could not determine this; here is
//! the provision; read it"* is a **correct answer**. Clean-looking wrongness is
//! not. Nothing is silently dropped and nothing is silently defaulted. See
//! [`anomaly`].
//!
//! ---
//!
//! # Jurisdiction
//!
//! The numbering and citation support targets **Singapore / Commonwealth**
//! statutory drafting (`Part II`, `s 12(3)(a)(ii)`, inserted sections `12A`),
//! which the UK, Singapore, Malaysia, Australia, NZ, HK and India largely
//! share. US conventions are **not** supported and will surface as unparsed
//! rather than being silently misread. See [`numbering`].
//!
//! # Example
//!
//! ```
//! use kopitiam_legal::{
//!     ingest, numbering::parse_statutory, source, synthetic,
//!     AsAtResult, Date, IngestRequest, Resolution,
//! };
//!
//! // A SYNTHETIC statute — invented for testing, not real law.
//! let lines = source::from_text_pages(&synthetic::widget_act_pages()).unwrap();
//! let act = ingest(IngestRequest {
//!     id: synthetic::act_id(),
//!     version: synthetic::act_version(),
//!     kind: synthetic::act_kind(),
//!     in_force_from: Date::new(2020, 1, 1).unwrap(),
//!     lines: &lines,
//! })
//! .unwrap();
//!
//! // You cannot ask "what does s 12(1) say?" — only what it said on a date.
//! let s12_1 = parse_statutory("12(1)").unwrap();
//! let AsAtResult::InForce(provision) = act.provision_as_at(&s12_1, synthetic::at(2021)).unwrap()
//! else {
//!     panic!("expected s 12(1) to be in force in 2021");
//! };
//!
//! // The text is verbatim, and it comes with a citation to a page.
//! assert!(provision.text().contains("without a licence"));
//! assert!(provision.citation().contains("p 2"));
//!
//! // "dwelling-house" in this Act does NOT mean what it means in English.
//! let s7 = parse_statutory("7").unwrap();
//! let Resolution::Defined(definition) = act.meaning_of("dwelling-house", &s7, synthetic::at(2021))
//! else {
//!     panic!("the Act defines this term");
//! };
//! assert!(definition.body().contains("houseboat"));
//!
//! // And everything the extractor found but refused to guess about.
//! for anomaly in act.anomalies() {
//!     println!("UNRESOLVED: {anomaly}");
//! }
//! ```
//!
//! # Licence and authorship
//!
//! AGPL-3.0-only, like all of KOPITIAM. This crate is original work; it forks
//! and adapts no upstream code. It builds on KOPITIAM's own [`kopitiam_pdf`]
//! and [`kopitiam_document`] engines rather than parsing PDFs itself.

pub mod anomaly;
pub mod date;
pub mod definition;
pub mod error;
pub mod ingest;
pub mod instrument;
pub mod judgment;
pub mod numbering;
pub mod ontology;
pub mod provenance;
pub mod provision;
pub mod reference;
pub mod source;
pub mod synthetic;
pub mod temporal;

pub use anomaly::{Anomaly, AnomalyKind};
pub use date::{AsAtDate, Date};
pub use definition::{
    Definition, DefinitionForce, DefinitionScope, Dictionary, Resolution, TermOccurrence,
};
pub use error::LegalError;
pub use ingest::{ingest, IngestRequest};
pub use instrument::{Instrument, InstrumentKind};
pub use judgment::{
    Citation, CitedAuthority, Classification, Holding, Judgment, Treatment,
};
pub use numbering::{
    NumberingScheme, Numeral, NumeralStyle, ProvisionComponent, ProvisionId, SectionNumber,
};
pub use ontology::{to_graph, LegalGraph};
pub use provenance::{DocumentId, DocumentVersion, PageNumber, Provenance, VerbatimText};
pub use provision::Provision;
pub use reference::{
    CrossReference, ReferenceConnective, ReferenceResolution, ReferenceTarget,
};
pub use source::{Emphasis, SourceLine};
pub use temporal::{
    Amendment, AmendmentOperation, AsAtResult, ProvisionHistory, TemporalWarning, Validity,
};
