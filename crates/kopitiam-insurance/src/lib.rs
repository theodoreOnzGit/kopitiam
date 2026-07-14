//! # KOPITIAM Insurance Engine
//!
//! Turns insurance documents — policy wordings, schedules, endorsements,
//! benefit tables — into **structured, provenance-carrying knowledge**.
//!
//! ---
//!
//! ## This crate does not interpret, advise, or adjudicate
//!
//! **Read this before using anything below.**
//!
//! An insurance policy is a **legal contract**. This crate **extracts and
//! locates** what a document says — verbatim, with a citation to the page and
//! clause it says it on. It does **not**:
//!
//! * decide whether a claim is payable,
//! * decide whether a person, event or loss falls inside or outside a clause,
//! * offer advice, a recommendation, or a comparison of policies,
//! * resolve an ambiguity in the wording,
//! * reconcile a contradiction in the document.
//!
//! There is no `is_covered()` in this API and there never will be. Coverage is
//! not a boolean; it is a legal conclusion about a contract, an event, and the
//! rules of construction a court would apply — and none of those are in a PDF.
//! What this crate offers instead is: *"Clause 4.2, on page 7, says this. Here
//! are the words. Here is what your policy's definitions section says the words
//! in it mean. Here is the endorsement that replaced it. Here is what I could
//! not work out. Go and read it."*
//!
//! Misrepresenting what a policy says is not a bug. It is a harm. Every design
//! decision in this crate follows from that.
//!
//! ---
//!
//! ## The four rules
//!
//! ### 1. Provenance is structural, not optional
//!
//! Every extracted item carries its **document, page, section, clause
//! identifier and the verbatim source text** — and it is the *type system*, not
//! a code review, that enforces it. [`ExtractedTerm<T>`] has no `Default`, no
//! `From<T>`, no public fields, and its only constructor demands a
//! [`Provenance`], which in turn demands all five components, of which
//! [`SourceText`] cannot be empty. An un-sourced extracted term does not
//! compile.
//!
//! Stronger still: [`Clause::cite`] checks the quotation **against the clause
//! it claims to come from**. A paraphrase, a summary, or an invented sentence
//! cannot be dressed up as the policy's own words.
//!
//! ### 2. Definitions are load-bearing
//!
//! Policies redefine ordinary words. A wording that pays on "Accident", and
//! defines *Accident* as *"a sudden, violent, external and visible event"*,
//! does not mean what an English speaker thinks it means — and a term extracted
//! without resolving it against the policy's own definitions section is
//! **misleading**. So the definitions section is modelled first-class, and
//! [`PolicyDocument::meaning_of`] / [`PolicyDocument::defined_terms_in`]
//! resolve against it. A term defined twice, inconsistently, comes back as
//! [`Resolution::Conflicting`] — never as an arbitrary pick.
//!
//! ### 3. Nothing ambiguous is silently normalised
//!
//! A clause that cannot be classified is [`ClauseRole::Unclassified`]. A
//! schedule value that cannot be typed is [`ScheduleValue::Unparseable`], with
//! the raw text. A bare `$` is [`Currency::Ambiguous`], not USD and not SGD.
//! `Nil` is not zero. Every such finding is also reported as an [`Anomaly`],
//! with the words it is about.
//!
//! *"I could not determine this — here is the clause, read it"* is a correct
//! and valuable answer. A clean-looking wrong answer is a dangerous one.
//!
//! ### 4. Endorsements override, visibly
//!
//! An endorsement rewrites the wording, and a reader who misses it gets the
//! wrong answer with total confidence. [`PolicyDocument::effective_clause`]
//! returns an [`EffectiveClause`] whose *variants are the override status*: you
//! cannot read the effective text of a replaced clause without being handed the
//! endorsement that replaced it.
//!
//! ---
//!
//! ## Architecture
//!
//! This crate is the **generic** insurance-document engine: motor, life,
//! travel, property, health. It owns clauses, definitions, exclusions,
//! schedules, endorsements, cross-references and provenance — the machinery of
//! *any* insurance document.
//!
//! Domain semantics live **above** it. `kopitiam-health` is a specialisation
//! that builds on the types here (see [`ExtractedTerm::map`], which lets a
//! domain crate refine a generic extraction into a domain type without the
//! value ever escaping its citation).
//!
//! Below it sits KOPITIAM's Document Engine: [`kopitiam_pdf`] recovers text
//! spans with geometry and font style, [`kopitiam_document`] reconstructs
//! headings, paragraphs, lists and tables. This crate writes **no PDF parser
//! and no table parser**; it adds insurance structure on top of theirs. And
//! [`kopitiam_ontology`] is what it emits into — see [`to_graph`] — so that a
//! policy read once enters the shared semantic graph and does not have to be
//! read again.
//!
//! ---
//!
//! ## Example
//!
//! ```no_run
//! use kopitiam_insurance::{EffectiveClause, Resolution, ingest_pdf};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let policy = ingest_pdf("policy.pdf")?;
//!
//! // What does *this policy* mean by "Accident"? (Not what English means.)
//! match policy.meaning_of("Accident") {
//!     Resolution::Defined(definition) => println!(
//!         "{} means: {} [{}]",
//!         definition.term(),
//!         definition.meaning(),
//!         definition.provenance(),
//!     ),
//!     Resolution::Conflicting(all) => {
//!         println!("the policy defines it {} times, inconsistently:", all.len());
//!         for definition in all {
//!             println!("  {} [{}]", definition.meaning(), definition.provenance());
//!         }
//!     }
//!     Resolution::Undefined => println!("not defined; plain meaning applies"),
//! }
//!
//! // What does the contract *now* say, after endorsements?
//! for clause in policy.clauses() {
//!     if let EffectiveClause::Replaced { by, wording, .. } =
//!         policy.effective_clause(clause.id())
//!     {
//!         println!("clause {} was replaced by {}: {wording}", clause.id(), by.id());
//!     }
//! }
//!
//! // And everything we could not work out.
//! for anomaly in policy.anomalies() {
//!     println!("UNRESOLVED: {}", anomaly.summary());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ---
//!
//! ## Provenance of this crate itself
//!
//! No insurer's policy wording is reproduced, quoted, or paraphrased anywhere
//! in this crate, including in its tests. Every clause, definition, exclusion
//! and schedule figure in the test suite is **synthetic — written for the
//! test**. A plausible-looking fake exclusion clause attributed to nobody is
//! still a fake exclusion clause, and in this domain that is actively
//! dangerous.

#![forbid(unsafe_code)]

mod anomaly;
mod classify;
mod clause;
mod crossref;
mod definition;
mod endorsement;
mod error;
mod exclusion;
mod ingest;
mod knowledge;
mod policy;
mod provenance;
mod schedule;

pub use anomaly::Anomaly;
pub use classify::{Classification, Confidence, DocumentClass};
pub use clause::{Clause, ClauseId, ClauseLine, ClauseRole};
pub use crossref::{CrossReference, ReferenceKind, ResolvedReference};
pub use definition::{Definition, Definitions, Resolution, TermOccurrence};
pub use endorsement::{EffectiveClause, Endorsement, EndorsementEffect, EndorsementId};
pub use error::Error;
pub use exclusion::{Exclusion, ExclusionEffect};
pub use ingest::{ingest_pages, ingest_pdf};
pub use knowledge::{KnowledgeGraph, SOURCE, to_graph};
pub use policy::PolicyDocument;
pub use provenance::{
    DocumentId, ExtractedTerm, PageNumber, Provenance, ProvenanceError, SectionPath, SourceText,
};
pub use schedule::{
    BenefitRow, BenefitTable, Currency, Money, MonetaryAmount, Percentage, Schedule, ScheduleEntry,
    ScheduleValue, parse_value,
};
