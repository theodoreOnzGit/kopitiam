//! Policies, and how they stack.
//!
//! # Singapore's three layers, and why they are modelled as a stack
//!
//! A Singaporean's hospital cover is usually three things at once, and people
//! routinely do not know which of them is paying for what:
//!
//! 1. **MediShield Life** — the universal basic scheme. Everyone has it. It is a
//!    statutory scheme administered by the state, not a contract you shopped for,
//!    and its parameters are set by the Ministry of Health and revised from time
//!    to time. *This crate hardcodes none of them.* If you want to know
//!    MediShield Life's deductible you must ingest the current scheme document,
//!    exactly as for any private wording — a figure baked into source code is a
//!    figure that silently goes stale and then quietly lies.
//!
//! 2. **An Integrated Shield Plan** — a private plan sold by an insurer that
//!    *integrates* with MediShield Life to cover higher ward classes and private
//!    hospitals.
//!
//! 3. **A rider** — an optional add-on that absorbs the deductible and/or the
//!    co-insurance the Integrated Shield Plan itself leaves you paying.
//!
//! The word "integrated" in (2) is doing a great deal of quiet work, and it is
//! the crux of the whole model. See [`crate::IntegrationMode`], where the two
//! possible meanings — and this crate's refusal to guess between them — are set
//! out in full. Read it before trusting any number out of [`crate::cost_share`].
//!
//! The structural facts here (that MediShield Life exists, that IPs integrate
//! with it, that riders attach to IPs) are public, stable features of how the
//! scheme is organised. Every *number* — every deductible, rate and limit — comes
//! from an ingested document or does not exist.

use std::fmt;

use kopitiam_insurance::{Definition, DocumentId, PolicyDocument, Resolution};

use crate::domain::TreatmentContext;
use crate::term::{CoverageStatement, PolicyTerm, TermKind, TermValue};

/// Identifies a policy layer within a stack.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PolicyId(String);

/// A blank policy id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("policy id is empty")]
pub struct EmptyPolicyId;

impl PolicyId {
    /// Creates a policy id, rejecting blank input.
    pub fn new(id: impl Into<String>) -> Result<Self, EmptyPolicyId> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(EmptyPolicyId);
        }
        Ok(Self(id))
    }

    /// The id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which of the three layers a policy is.
///
/// The variant order is the stack order, and [`PolicyStack`] relies on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerKind {
    /// The universal basic scheme (MediShield Life).
    UniversalBasic,
    /// A private plan integrating with the basic scheme (an Integrated Shield
    /// Plan).
    IntegratedTopUp,
    /// An add-on absorbing the top-up plan's deductible and/or co-insurance.
    Rider,
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UniversalBasic => "universal basic scheme",
            Self::IntegratedTopUp => "integrated top-up plan",
            Self::Rider => "rider",
        })
    }
}

/// One policy: the document, and the terms read out of it.
///
/// The [`PolicyDocument`] is kept, not discarded after extraction. That is
/// deliberate: the document is the authority, the extracted terms are only a
/// reading of it, and a reading that has lost the thing it was a reading of
/// cannot be checked. Keeping it also means definitions resolve through
/// `kopitiam-insurance`'s [`Resolution`] — which, crucially, can report that a
/// policy defines the same word twice, inconsistently.
#[derive(Debug, Clone)]
pub struct PolicyLayer {
    id: PolicyId,
    name: String,
    kind: LayerKind,
    document: PolicyDocument,
    terms: Vec<PolicyTerm>,
}

impl PolicyLayer {
    /// Assembles a policy from its document and the terms extracted from it.
    pub fn new(
        id: PolicyId,
        name: impl Into<String>,
        kind: LayerKind,
        document: PolicyDocument,
        terms: Vec<PolicyTerm>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            document,
            terms,
        }
    }

    /// The policy's id.
    pub fn id(&self) -> &PolicyId {
        &self.id
    }

    /// The policy's name, as a human would say it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Which layer of the stack this is.
    pub fn kind(&self) -> LayerKind {
        self.kind
    }

    /// The source document — the authority the terms are only a reading of.
    pub fn document(&self) -> &PolicyDocument {
        &self.document
    }

    /// The document's identifier.
    pub fn document_id(&self) -> &DocumentId {
        self.document.id()
    }

    /// Every term extracted from the document, in extraction order.
    pub fn terms(&self) -> &[PolicyTerm] {
        &self.terms
    }

    /// Every term of a given kind, regardless of scope — including unresolved
    /// ones.
    pub fn terms_of_kind(&self, kind: TermKind) -> impl Iterator<Item = &PolicyTerm> {
        self.terms.iter().filter(move |t| t.kind() == kind)
    }

    /// What this policy means by a word.
    ///
    /// Delegates to `kopitiam-insurance`'s definitions engine, so the three
    /// honest answers come back distinguishable:
    ///
    /// * [`Resolution::Defined`] — the policy defines it, and its meaning here is
    ///   the policy's, overriding plain English.
    /// * [`Resolution::Conflicting`] — the policy defines it **twice,
    ///   inconsistently**. Not a bug in the reader; a defect in the document, and
    ///   one worth knowing about.
    /// * [`Resolution::Undefined`] — the policy does not define it, so plain
    ///   meaning applies.
    ///
    /// [`crate::compare`] consumes all three, and treats the second as a finding
    /// rather than a failure.
    pub fn meaning_of(&self, word: &str) -> Resolution<'_> {
        self.document.meaning_of(word)
    }

    /// The single definition of a word, when the policy gives exactly one.
    ///
    /// `None` for both "not defined" and "defined inconsistently" — deliberately.
    /// A caller that wants an answer out of a self-contradicting policy has to
    /// match on [`Self::meaning_of`] and look the contradiction in the eye.
    pub fn definition(&self, word: &str) -> Option<&Definition> {
        self.meaning_of(word).definition()
    }

    /// The terms of a kind that apply to a given treatment, most specific first.
    ///
    /// "Most specific first" means a clause stated for *(Private, non-panel)*
    /// outranks one stated for *(Private)*, which outranks an unqualified one.
    /// That is how a wording is actually read: the specific provision governs.
    ///
    /// Unresolved clauses ([`TermValue::Ambiguous`]) are **included**, and
    /// deliberately so — [`crate::cost_share`] must see them in order to refuse.
    /// Filtering them out here would turn "this clause is unclear" into "there is
    /// no such clause", which is the silent-default failure this crate exists to
    /// prevent.
    pub fn applicable_terms(&self, kind: TermKind, ctx: &TreatmentContext) -> Vec<&PolicyTerm> {
        let mut matches: Vec<&PolicyTerm> = self
            .terms
            .iter()
            .filter(|t| t.kind() == kind && t.scope().applies_to(ctx))
            .collect();
        matches.sort_by_key(|t| std::cmp::Reverse(t.scope().specificity()));
        matches
    }
}

/// A stack that could not be assembled as described.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StackError {
    /// No policies were supplied.
    #[error("a policy stack must contain at least one policy")]
    Empty,

    /// More than one universal basic scheme. Nobody has two.
    #[error("a stack may contain at most one universal basic scheme, found {0}")]
    MultipleUniversalBasic(usize),

    /// More than one integrated top-up plan.
    ///
    /// You cannot hold two Integrated Shield Plans at once. If two documents
    /// produced two, one of them has been misclassified — and proceeding would
    /// silently double someone's cover.
    #[error("a stack may contain at most one integrated top-up plan, found {0}")]
    MultipleIntegratedTopUp(usize),

    /// A rider with nothing to attach to.
    ///
    /// A rider does not pay hospital bills. It absorbs *another plan's*
    /// deductible and co-insurance. A rider on its own has no base plan whose
    /// cost-sharing it could absorb, so it can pay nothing at all — and a
    /// calculation run against it would look like it worked and return zero.
    #[error("a rider must attach to an integrated top-up plan; this stack has no top-up plan")]
    RiderWithoutTopUpPlan,
}

/// The layers of cover a person actually holds, ordered from the bottom up.
#[derive(Debug, Clone)]
pub struct PolicyStack {
    layers: Vec<PolicyLayer>,
}

impl PolicyStack {
    /// Assembles a stack, checking that the combination is one a person could
    /// really hold.
    ///
    /// Layers may be supplied in any order; they are sorted into stack order. The
    /// validation is not bureaucratic tidiness — each rejected combination is one
    /// that would otherwise produce a plausible, wrong number. See [`StackError`].
    pub fn new(mut layers: Vec<PolicyLayer>) -> Result<Self, StackError> {
        if layers.is_empty() {
            return Err(StackError::Empty);
        }

        let count = |k: LayerKind| layers.iter().filter(|l| l.kind() == k).count();

        let basics = count(LayerKind::UniversalBasic);
        if basics > 1 {
            return Err(StackError::MultipleUniversalBasic(basics));
        }

        let top_ups = count(LayerKind::IntegratedTopUp);
        if top_ups > 1 {
            return Err(StackError::MultipleIntegratedTopUp(top_ups));
        }

        if count(LayerKind::Rider) > 0 && top_ups == 0 {
            return Err(StackError::RiderWithoutTopUpPlan);
        }

        layers.sort_by_key(|l| l.kind());
        Ok(Self { layers })
    }

    /// Every layer, bottom-up.
    pub fn layers(&self) -> &[PolicyLayer] {
        &self.layers
    }

    /// The universal basic scheme, if the stack includes one.
    ///
    /// `None` is a meaningful state, not an oversight: a document set may contain
    /// only a private wording. What the calculation does about that depends on the
    /// plan's [`crate::IntegrationMode`], and it never assumes.
    pub fn universal_basic(&self) -> Option<&PolicyLayer> {
        self.layer_of(LayerKind::UniversalBasic)
    }

    /// The integrated top-up plan, if the stack includes one.
    pub fn integrated_top_up(&self) -> Option<&PolicyLayer> {
        self.layer_of(LayerKind::IntegratedTopUp)
    }

    /// The riders, in stack order.
    pub fn riders(&self) -> impl Iterator<Item = &PolicyLayer> {
        self.layers.iter().filter(|l| l.kind() == LayerKind::Rider)
    }

    fn layer_of(&self, kind: LayerKind) -> Option<&PolicyLayer> {
        self.layers.iter().find(|l| l.kind() == kind)
    }

    /// Every term in the stack that this crate deliberately does **not** evaluate,
    /// so that a caller can surface them to a human.
    ///
    /// # Why this exists
    ///
    /// Waiting periods, pre-existing-condition clauses and outright exclusions
    /// decide whether a claim is payable *at all* — and they turn on facts this
    /// crate does not have and must not pretend to (when symptoms began, what the
    /// insured declared, whether treatment was medically necessary). A cost-share
    /// figure computed while ignoring a pre-existing-condition exclusion is not
    /// merely incomplete; it is actively misleading, because it looks like an
    /// answer.
    ///
    /// So [`crate::cost_share`] calls this and attaches every such clause to its
    /// result as a caveat, verbatim. The number it gives you means *"if this claim
    /// is payable at all, here is the split"* — and this is the list of reasons it
    /// might not be.
    pub fn unevaluated_eligibility_terms(&self) -> Vec<&PolicyTerm> {
        self.layers
            .iter()
            .flat_map(PolicyLayer::terms)
            .filter(|t| {
                matches!(
                    t.kind(),
                    TermKind::WaitingPeriod | TermKind::PreExistingCondition
                ) || matches!(
                    t.value(),
                    TermValue::Coverage {
                        statement: CoverageStatement::StatedExcluded
                            | CoverageStatement::StatedConditional { .. },
                        ..
                    }
                )
            })
            .collect()
    }
}
