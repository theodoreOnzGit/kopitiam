//! # KOPITIAM Health Engine
//!
//! Models **what a health insurance policy document says**, with citations.
//!
//! ---
//!
//! ## This crate does not give insurance advice
//!
//! Read this before using anything below. It is not boilerplate; it is the design
//! constraint the whole crate is built around, and every API here was shaped by it.
//!
//! **This crate does not tell anyone what to buy. It does not decide whether a
//! claim is payable. It does not tell you whether you are covered.**
//!
//! What it does is narrower and, we think, more useful: it reads a policy document,
//! extracts the terms it can extract, and hands them back **welded to the clauses
//! they came from** — so a human can find the relevant wording, set it beside
//! another policy's wording, and make their own decision with the actual text in
//! front of them.
//!
//! The distinction is the entire point:
//!
//! | This crate says | This crate never says |
//! |---|---|
//! | "Clause 4.2, p.7, states: *'We will not pay for any Pre-existing Condition.'*" | "You are not covered." |
//! | "Per §2.1 (S$3,500 deductible) and §2.2 (10% co-insurance), the stated split of a S$10,000 claimable bill is S$4,150 / S$5,850." | "You will pay S$4,150." |
//! | "The wording states a 12-month waiting period (§6.1). Whether it applies to your claim is not something this crate determines." | "Your claim is within the waiting period." |
//!
//! Health insurance decides whether a person can afford to be ill. A misread
//! exclusion, a wrong pre-existing-condition clause, a mis-stated deductible — these
//! are the difference between a claim paid and a family ruined. A tool in this
//! domain that is *confidently wrong* is worse than no tool at all, because a
//! confident wrong answer displaces the reading of the document that would have
//! caught it.
//!
//! So: **when in doubt, this crate refuses.** Read [`cost_share::CostShareRefusal`]
//! — less an error type than a catalogue of the ways a document can fail to answer
//! a question, each returned with the clause attached. An honest *"I could not
//! determine this; here is the clause, read it yourself"* is a **correct** answer
//! here. A clean-looking wrong number is not.
//!
//! ---
//!
//! ## The three rules
//!
//! ### 1. Provenance is structural, not optional
//!
//! Every extracted term is a [`PolicyTerm`], and a `PolicyTerm` cannot be built
//! without an [`ExtractedTerm`](kopitiam_insurance::ExtractedTerm) — which cannot
//! be built without a [`Provenance`](kopitiam_insurance::Provenance): document,
//! page, section, clause, **and the clause's own words**. Not by convention: there
//! is no `Default`, no public fields, and no constructor that omits the citation.
//! An un-sourced term does not compile.
//!
//! Stronger still, terms are minted through
//! [`Clause::provenance`](kopitiam_insurance::Clause::provenance) /
//! [`Clause::cite`](kopitiam_insurance::Clause::cite), and `cite` **checks the
//! quotation against the clause it claims to come from**. A paraphrase cannot be
//! dressed up as the policy's own words.
//!
//! ### 2. Nothing is a `bool`, and nothing is a bare number
//!
//! A deductible, a co-insurance rate, an annual limit and a waiting period are four
//! different things and none of them is an `f64`. Money is integer cents
//! ([`money::Amount`]); a percentage is integer basis points. "Covered" is not a
//! `bool` — see [`CoverageStatement`], which can be `Silent` and can be
//! `StatedConditional`, because documents are. A twelve-month waiting period does
//! not convert to days ([`PolicyDuration`]), because a month is not a fixed number
//! of them.
//!
//! And a bare `$` is not SGD. It is `$`
//! ([`Currency::Ambiguous`](kopitiam_insurance::Currency::Ambiguous)), and
//! [`money::Amount`] will not compute with it.
//!
//! ### 3. No silent defaults, anywhere
//!
//! A missing deductible does not become zero. A missing claim limit does not become
//! infinity. A rider with no co-payment clause does not become a 0% co-payment. Each
//! of those defaults errs in the insured's favour and quietly *overstates* their
//! cover — the direction of error that gets someone into an operating theatre they
//! cannot pay for. Each is a refusal instead.
//!
//! ---
//!
//! ## What is modelled
//!
//! Singapore-flavoured, because that is the maintainer's context, but the core is
//! generic:
//!
//! * **The stack** — MediShield Life (universal basic), an Integrated Shield Plan on
//!   top of it, and a rider on top of that. See [`policy`], and above all
//!   [`IntegrationMode`], which models the single most misunderstood thing in
//!   Singapore health insurance and which this crate flatly **refuses to guess at**.
//! * **Cost sharing** — deductible first, then co-insurance on the remainder, then
//!   the insurer's limit. The order matters enormously; see [`cost_share`].
//! * **Ward class and panel status** — the same operation is covered very differently
//!   depending on where, and by whom, it is done. See [`domain`].
//! * **Comparison** — the same term across several policies, refusing to compare them
//!   when the policies do not *mean* the same thing by it. See [`compare`].
//! * **Emission into the shared knowledge graph** — see [`facts`].
//!
//! ---
//!
//! ## No real policy has been ingested
//!
//! Stated plainly, because leaving it ambiguous would be dangerous: **this crate
//! ships no real insurer's terms, and none of its tests use any.** Every figure in
//! the tests and the doc examples comes from a synthetic wording written for the
//! purpose (a fictional "Kopi Assurance" and a fictional national scheme).
//!
//! MediShield Life's real parameters are set by Singapore's Ministry of Health and
//! are revised from time to time. A figure hardcoded here would be a figure that
//! silently goes stale and then quietly lies. **If you want to know what your policy
//! says, ingest your policy.**
//!
//! ---
//!
//! ## Relationship to `kopitiam-insurance`
//!
//! Health insurance is a **specialisation** of insurance-document extraction, not a
//! parallel universe. This crate writes no PDF parser, no clause segmenter, no
//! definitions engine and no provenance model — all of that is
//! [`kopitiam_insurance`]'s, which in turn stands on `kopitiam-pdf` and
//! `kopitiam-document`. What lives here is only what a motor policy does not have:
//! wards, panels, the MediShield/IP/rider stack, and the deductible-then-co-insurance
//! arithmetic.
//!
//! The one exception is [`money`], which adds currency-checked *arithmetic* over
//! `kopitiam-insurance`'s money types. `kopitiam-insurance` is an extraction engine
//! and has had no reason to do arithmetic; if a second domain crate needs it, that
//! is the signal to move [`money`] down.

#![forbid(unsafe_code)]

pub mod compare;
pub mod cost_share;
pub mod domain;
pub mod extract;
pub mod facts;
pub mod money;
pub mod policy;
pub mod term;

pub use compare::{
    Comparability, Comparison, ComparisonEntry, DefinitionDivergence, DefinitionState,
    PolicyPosition, compare,
};
pub use cost_share::{
    Bill, BillError, BorneBy, Caveat, CostShareBreakdown, CostShareRefusal, CostShareStep,
    LayerOutcome, compute_cost_share,
};
pub use domain::{
    ClaimLimit, CoInsurance, Deductible, DeductibleBasis, PolicyDuration,
    PreExistingConditionTreatment, ProviderNetwork, Scope, TreatmentContext, WaitingPeriod,
    WardClass,
};
pub use extract::{ExtractionConfig, extract_from_clause, extract_terms};
pub use facts::{FactBatch, facts_for_policy, facts_for_stack};
pub use money::{Amount, MoneyError};
pub use policy::{LayerKind, PolicyId, PolicyLayer, PolicyStack, StackError};
pub use term::{
    Ambiguity, AmbiguityKind, CoPayment, CoPaymentBase, CoverageStatement, IntegrationMode,
    PolicyTerm, RiderCoverage, TermKind, TermValue,
};

// Re-exported so a consumer of `kopitiam-health` can read a citation without also
// having to depend on `kopitiam-insurance` directly. These are *their* types; they
// are surfaced here, not redefined.
pub use kopitiam_insurance::{
    Clause, ClauseId, Currency, DocumentId, ExtractedTerm, MonetaryAmount, Money, PageNumber,
    Percentage, PolicyDocument, Provenance, Resolution, SectionPath, SourceText,
};

/// Reads a policy PDF end to end: document -> clauses -> health terms.
///
/// The whole pipeline in one call. Ingestion (PDF text, layout, clause segmentation,
/// definitions, classification) is [`kopitiam_insurance::ingest_pdf`]'s work; this
/// adds only the health-domain extraction on top.
///
/// # Errors
///
/// [`kopitiam_insurance::Error`] if the PDF cannot be read, or if a citation could
/// not be built back to its own clause — which would mean the extractor produced
/// text the document does not contain, exactly the failure the provenance model
/// exists to catch.
pub fn read_policy_pdf(
    path: impl AsRef<std::path::Path>,
    config: &ExtractionConfig,
) -> Result<(PolicyDocument, Vec<PolicyTerm>), kopitiam_insurance::Error> {
    let document = kopitiam_insurance::ingest_pdf(path)?;
    let terms = extract_terms(document.clauses(), config);
    Ok((document, terms))
}

/// Reads already-extracted PDF pages end to end.
///
/// The entry point for callers that already have [`kopitiam_pdf::Page`]s — and the
/// one this crate's tests use, since a synthetic policy is built as pages rather
/// than as a PDF file on disk.
///
/// # Errors
///
/// As [`read_policy_pdf`].
pub fn read_policy_pages(
    id: DocumentId,
    pages: &[kopitiam_pdf::Page],
    config: &ExtractionConfig,
) -> Result<(PolicyDocument, Vec<PolicyTerm>), kopitiam_insurance::Error> {
    let document = kopitiam_insurance::ingest_pages(id, pages)?;
    let terms = extract_terms(document.clauses(), config);
    Ok((document, terms))
}
