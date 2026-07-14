//! [`PolicyTerm`] — one thing a policy document says, and where it says it.
//!
//! # The central type, and the constraint that defines this crate
//!
//! A `PolicyTerm` has three parts and you cannot have two of them:
//!
//! 1. **What it says**, typed ([`TermValue`]).
//! 2. **When it applies** ([`crate::Scope`]).
//! 3. **Where it says it**, verbatim — an
//!    [`ExtractedTerm`](kopitiam_insurance::ExtractedTerm) from
//!    `kopitiam-insurance`, which is a value welded to a
//!    [`Provenance`](kopitiam_insurance::Provenance): document, page, section,
//!    clause, and the clause's own words.
//!
//! There is exactly one constructor and it takes an `ExtractedTerm` by value.
//! `ExtractedTerm` has no `Default`, no public fields and no constructor that
//! omits the citation; `Provenance` requires all five components; and
//! `SourceText` rejects blank text. So there is no `PolicyTerm` anywhere in a
//! running program that cannot be traced back to a quoted clause in a named
//! document. Not "should not be" — *cannot be*.
//!
//! Stronger still: terms are normally minted through
//! [`Clause::extract`](kopitiam_insurance::Clause::extract), which **checks the
//! quotation against the clause it claims to come from**. A paraphrase cannot be
//! dressed up as the policy's own words.
//!
//! That constraint is the design. Everything else in this crate is built on the
//! assumption that it holds.
//!
//! # Why "covered" is not a `bool`
//!
//! Because a document does not contain booleans. It contains sentences, and
//! sentences say things like "we will pay, provided the treatment is Medically
//! Necessary and takes place in a Restructured Hospital". Squeezing that into
//! `covered: true` throws away the two conditions that decide whether the money
//! actually arrives — and does so *invisibly*, leaving a caller holding a
//! confident `true` and no way to know what it cost them.
//!
//! [`CoverageStatement`] therefore models what the *document states*, including
//! the states a boolean cannot represent: silence, conditionality,
//! contradiction, and "we could not tell". Those are not failure modes to be
//! smoothed over. Three of them are the most common honest answers in this
//! domain.

use std::fmt;

use kopitiam_insurance::{ExtractedTerm, MonetaryAmount, Percentage, Provenance};

use crate::domain::{
    ClaimLimit, CoInsurance, Deductible, PreExistingConditionTreatment, Scope, WaitingPeriod,
    WardClass,
};

/// What a document states about whether it will pay for something.
///
/// Deliberately not a `bool`. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageStatement {
    /// The document states the benefit is payable — subject, always, to the
    /// plan's other terms (deductible, co-insurance, limits, exclusions
    /// elsewhere in the wording). This variant means *"this clause says yes"*,
    /// not *"you are covered"*.
    StatedPayable,

    /// The document states the benefit is not payable.
    StatedExcluded,

    /// The document states the benefit is payable **only if** conditions are
    /// met, and here they are in the document's own words.
    ///
    /// Whether the conditions are met in a particular case is a human's call.
    /// This crate carries them; it does not evaluate them.
    StatedConditional {
        /// The conditions, as the clause words them.
        conditions: Vec<String>,
    },

    /// The document does not address this benefit at all.
    ///
    /// **Silence is not exclusion, and it is not cover.** It is silence, and it
    /// usually means the answer lives in a clause we have not found, or in a
    /// different document entirely (a schedule, a benefit table, an
    /// endorsement). Reporting silence as `false` would be a fabrication in the
    /// direction that most often makes a person give up on a claim they were
    /// entitled to.
    Silent,
}

impl fmt::Display for CoverageStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StatedPayable => f.write_str("document states: payable (subject to other terms)"),
            Self::StatedExcluded => f.write_str("document states: excluded"),
            Self::StatedConditional { conditions } => {
                write!(
                    f,
                    "document states: payable only if — {}",
                    conditions.join("; ")
                )
            }
            Self::Silent => f.write_str("document is silent on this"),
        }
    }
}

/// A word the document defines for its own purposes.
///
/// # Why definitions are terms in their own right
///
/// "Hospitalisation" is not a universal concept. One wording means an overnight
/// stay; another includes day surgery; another requires admission on a doctor's
/// recommendation. Two policies can state the *same* deductible for
/// "hospitalisation" and mean materially different things by it.
///
/// So definitions are extracted, carried, and — crucially — checked before any
/// cross-policy comparison. See [`crate::compare`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    /// The word being defined, normalised to lowercase for lookup.
    pub term: String,
    /// What the document says it means, in the document's own words.
    pub meaning: String,
}

/// Why we could not turn a clause into a value — with the clause attached.
///
/// # An ambiguity is a result, not an error
///
/// This crate returns ambiguities as *data*, not as `Err(())`, because
/// "the wording is unclear here, read it yourself" is frequently the **correct
/// and most useful answer**. It is what a careful human adviser would say. The
/// failure mode we are guarding against is not "returned an ambiguity"; it is
/// "resolved an ambiguity by guessing and did not mention it".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ambiguity {
    /// What kind of term this was trying to be, so a consumer looking for a
    /// deductible finds the unresolved deductible clause rather than silently
    /// missing it.
    pub intended: TermKind,
    /// Why it could not be resolved.
    pub kind: AmbiguityKind,
    /// A short human-readable note. Never a substitute for the clause.
    pub note: String,
    /// **Every** clause involved, verbatim. Always non-empty — an ambiguity
    /// with nothing to read would be useless.
    pub sources: Vec<Provenance>,
}

/// The shapes an ambiguity comes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbiguityKind {
    /// Two or more clauses say things that cannot both be true.
    ///
    /// Real wordings do this — usually because a schedule and a body clause
    /// were revised out of step. It is the insured's most valuable finding and
    /// the last thing an extraction pipeline should paper over by picking one.
    Contradictory,

    /// The clause plainly concerns this term, but its text could not be turned
    /// into a typed value — an unparseable amount, a figure given as a
    /// cross-reference, a table we could not read.
    Unparseable,

    /// The clause states the term in language too vague to compute with: "a
    /// reasonable period", "such amount as we may determine".
    Underspecified,
}

impl fmt::Display for AmbiguityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Contradictory => "contradictory clauses",
            Self::Unparseable => "clause could not be parsed",
            Self::Underspecified => "clause is too vague to compute with",
        })
    }
}

/// How a rider says it deals with the base plan's cost-sharing.
///
/// A rider exists to absorb the deductible and/or the co-insurance that the
/// base plan leaves with the insured. Each is a [`CoverageStatement`] rather
/// than a `bool` for the reason given in the module docs: a rider that covers
/// the deductible "for treatment by a Panel Specialist" is not the same rider
/// as one that covers it unconditionally, and flattening both to `true` erases
/// the distinction that decides the bill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiderCoverage {
    /// What the rider says about the base plan's deductible.
    pub deductible: CoverageStatement,
    /// What the rider says about the base plan's co-insurance.
    pub co_insurance: CoverageStatement,
}

/// What the rider still makes the insured pay, even where it covers them.
///
/// Riders are commonly written with a residual co-payment so that the insured
/// keeps some exposure to the size of the bill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoPayment {
    /// The rate, as stated.
    pub rate: Percentage,
    /// The ceiling on the co-payment, if the document states one. `None` means
    /// **the document did not state a cap**, not that there is none.
    pub cap: Option<MonetaryAmount>,
    /// What the rate is applied *to*. See [`CoPaymentBase`].
    pub base: CoPaymentBase,
}

/// What a rider's co-payment rate is charged on.
///
/// # Why this is extracted and not assumed
///
/// 5% of the whole claimable bill and 5% of the amount the rider would
/// otherwise absorb are very different numbers on a large claim. Wordings
/// differ. Picking one and hardcoding it would bake a specific insurer's
/// mechanics into the engine and then apply them, silently, to every other
/// insurer's policy — producing a plausible figure that is simply wrong.
///
/// So the base is a term like any other: it comes from a clause or the
/// calculation refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoPaymentBase {
    /// The rate applies to the amount the rider would otherwise reimburse.
    AmountRiderWouldAbsorb,
    /// The rate applies to the whole claimable amount.
    ClaimableAmount,
}

impl fmt::Display for CoPaymentBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::AmountRiderWouldAbsorb => "the amount the rider would otherwise absorb",
            Self::ClaimableAmount => "the claimable amount",
        })
    }
}

/// How a private plan sits on top of the universal basic scheme.
///
/// # The single most misunderstood thing in Singapore health insurance
///
/// An Integrated Shield Plan is sold as "MediShield Life plus private cover",
/// and almost everyone — including people who own one — pictures that as two
/// insurers paying in sequence: MediShield Life knocks its share off the bill,
/// and the private plan's deductible then applies to whatever is left. Under
/// that mental model a large MediShield Life payout would shrink, or even wipe
/// out, the deductible you pay.
///
/// Whether that is what happens depends entirely on how the plan's own wording
/// is drafted, and the two possible drafts give **materially different
/// out-of-pocket figures for the same bill**. See the test
/// `integration_mode_changes_the_patient_bill` in `tests/stacking.rs`, which
/// runs the identical bill and the identical deductible through both and gets
/// different answers.
///
/// This crate therefore **refuses to choose**. The mode is a [`PolicyTerm`]
/// like any other: it is extracted from a clause, or
/// [`crate::cost_share::compute_cost_share`] declines to produce a number at
/// all. Getting this wrong is not a rounding error — it is the difference
/// between a patient budgeting for a few hundred dollars and budgeting for a
/// few thousand — and the one thing this crate must never do is guess it on a
/// user's behalf and sound sure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMode {
    /// The private plan's benefit is stated to be **inclusive of** the basic
    /// scheme's payout: the plan applies one deductible and one co-insurance to
    /// the whole claimable amount, and the insurer settles the basic scheme's
    /// share behind the scenes. The basic scheme does **not** reduce the bill
    /// before the deductible bites.
    InclusiveOfBasic,

    /// The private plan pays only what remains after the basic scheme has paid
    /// — a genuine second layer, each with its own deductible, co-insurance and
    /// limits.
    ExcessOfBasic,
}

impl fmt::Display for IntegrationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InclusiveOfBasic => {
                "private benefit is inclusive of the basic scheme's payout \
                 (one deductible on the whole claimable amount)"
            }
            Self::ExcessOfBasic => {
                "private benefit applies only to what the basic scheme leaves unpaid"
            }
        })
    }
}

/// The kind of a term, without its value.
///
/// Used to look a term up ("what does this policy say the deductible is?") and
/// to line the same term up across policies in [`crate::compare`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TermKind {
    /// A deductible.
    Deductible,
    /// A co-insurance rate.
    CoInsurance,
    /// A ceiling on what the insurer pays.
    ClaimLimit,
    /// A period before a benefit becomes available.
    WaitingPeriod,
    /// A statement about whether something is payable.
    Coverage,
    /// What the document says about pre-existing conditions.
    PreExistingCondition,
    /// The ward class the plan entitles the insured to.
    WardEntitlement,
    /// What a rider says it absorbs.
    RiderCoverage,
    /// A rider's residual co-payment.
    CoPayment,
    /// How a private plan stacks on the universal basic scheme.
    IntegrationMode,
    /// A word the document defines.
    Definition,
}

impl TermKind {
    /// The words whose *definition* this kind of term depends on.
    ///
    /// Comparing two policies' deductibles is meaningless if they define
    /// "claimable amount" differently, because the deductible is charged
    /// against that amount. [`crate::compare`] consults this list and refuses
    /// to compare across a definitional divergence.
    ///
    /// The lists are deliberately short and conservative. They name the
    /// definitions whose divergence would make the *numbers* incomparable, not
    /// every word a lawyer could quibble over.
    pub fn depends_on_definitions(self) -> &'static [&'static str] {
        match self {
            Self::Deductible | Self::CoInsurance => &["claimable amount", "policy year"],
            Self::ClaimLimit => &["claimable amount", "policy year"],
            Self::Coverage | Self::WaitingPeriod => &["hospitalisation"],
            Self::PreExistingCondition => &["pre-existing condition"],
            Self::CoPayment | Self::RiderCoverage => &["claimable amount"],
            Self::WardEntitlement | Self::IntegrationMode | Self::Definition => &[],
        }
    }
}

impl fmt::Display for TermKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Deductible => "deductible",
            Self::CoInsurance => "co-insurance",
            Self::ClaimLimit => "claim limit",
            Self::WaitingPeriod => "waiting period",
            Self::Coverage => "coverage",
            Self::PreExistingCondition => "pre-existing condition",
            Self::WardEntitlement => "ward entitlement",
            Self::RiderCoverage => "rider coverage",
            Self::CoPayment => "co-payment",
            Self::IntegrationMode => "integration mode",
            Self::Definition => "definition",
        })
    }
}

/// What a clause says, typed.
///
/// Note that [`TermValue::Ambiguous`] is a first-class value and not an error
/// return. A clause we could not resolve is still a *term the policy contains*,
/// and a consumer looking for the deductible must find it rather than conclude
/// there isn't one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermValue {
    /// A deductible, as stated.
    Deductible(Deductible),
    /// A co-insurance rate, as stated.
    CoInsurance(CoInsurance),
    /// A ceiling on what the insurer pays, as stated.
    ClaimLimit(ClaimLimit),
    /// A waiting period, as stated.
    WaitingPeriod(WaitingPeriod),
    /// What the document states about whether something is payable.
    Coverage {
        /// What the benefit is, in the document's own words.
        benefit: String,
        /// What the document says about it.
        statement: CoverageStatement,
    },
    /// What the document states about pre-existing conditions.
    PreExistingCondition(PreExistingConditionTreatment),
    /// The ward class the plan entitles the insured to.
    WardEntitlement(WardClass),
    /// What a rider says it absorbs.
    RiderCoverage(RiderCoverage),
    /// A rider's residual co-payment.
    CoPayment(CoPayment),
    /// How the plan stacks on the universal basic scheme.
    IntegrationMode(IntegrationMode),
    /// A word the document defines.
    Definition(Definition),
    /// A clause we could not resolve — with the clause attached.
    Ambiguous(Ambiguity),
}

impl TermValue {
    /// What kind of term this is (or, for an ambiguity, was trying to be).
    pub fn kind(&self) -> TermKind {
        match self {
            Self::Deductible(_) => TermKind::Deductible,
            Self::CoInsurance(_) => TermKind::CoInsurance,
            Self::ClaimLimit(_) => TermKind::ClaimLimit,
            Self::WaitingPeriod(_) => TermKind::WaitingPeriod,
            Self::Coverage { .. } => TermKind::Coverage,
            Self::PreExistingCondition(_) => TermKind::PreExistingCondition,
            Self::WardEntitlement(_) => TermKind::WardEntitlement,
            Self::RiderCoverage(_) => TermKind::RiderCoverage,
            Self::CoPayment(_) => TermKind::CoPayment,
            Self::IntegrationMode(_) => TermKind::IntegrationMode,
            Self::Definition(_) => TermKind::Definition,
            Self::Ambiguous(a) => a.intended,
        }
    }

    /// Whether this term is an unresolved clause.
    pub fn is_ambiguous(&self) -> bool {
        matches!(self, Self::Ambiguous(_))
    }
}

impl fmt::Display for TermValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deductible(d) => write!(f, "deductible {d}"),
            Self::CoInsurance(c) => write!(f, "co-insurance {c}"),
            Self::ClaimLimit(l) => write!(f, "limit {l}"),
            Self::WaitingPeriod(w) => write!(f, "{w}"),
            Self::Coverage { benefit, statement } => write!(f, "{benefit}: {statement}"),
            Self::PreExistingCondition(p) => write!(f, "pre-existing conditions: {p}"),
            Self::WardEntitlement(w) => write!(f, "ward entitlement: {w}"),
            Self::RiderCoverage(r) => write!(
                f,
                "rider — deductible: {}; co-insurance: {}",
                r.deductible, r.co_insurance
            ),
            Self::CoPayment(c) => {
                write!(f, "co-payment {}% of {}", c.rate.to_decimal_string(), c.base)?;
                match &c.cap {
                    Some(cap) => write!(f, ", capped at {}", cap.amount().to_decimal_string()),
                    None => Ok(()),
                }
            }
            Self::IntegrationMode(m) => write!(f, "{m}"),
            Self::Definition(d) => write!(f, "\"{}\" means: {}", d.term, d.meaning),
            Self::Ambiguous(a) => write!(f, "AMBIGUOUS {} ({}): {}", a.intended, a.kind, a.note),
        }
    }
}

/// One thing a policy document says, with the clause it says it in.
///
/// See the module docs. The invariant that matters: **there is no way to
/// construct one of these without a [`Provenance`].**
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyTerm {
    /// The value *and* its citation, inseparably. See the module docs.
    extracted: ExtractedTerm<TermValue>,
    scope: Scope,
}

impl PolicyTerm {
    /// Records what a clause says.
    ///
    /// The only constructor. It takes an
    /// [`ExtractedTerm`](kopitiam_insurance::ExtractedTerm) by value — a type
    /// that cannot exist without a [`Provenance`], which cannot exist without a
    /// non-empty quotation and a real location. So an un-sourced `PolicyTerm` is
    /// not something a caller can forget to supply. It is something the language
    /// will not let them express.
    pub fn new(extracted: ExtractedTerm<TermValue>, scope: Scope) -> Self {
        Self { extracted, scope }
    }

    /// What the clause says.
    pub fn value(&self) -> &TermValue {
        self.extracted.value()
    }

    /// What kind of term it is.
    pub fn kind(&self) -> TermKind {
        self.extracted.value().kind()
    }

    /// When it applies.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Where it says it — document, page, section, clause.
    pub fn provenance(&self) -> &Provenance {
        self.extracted.provenance()
    }

    /// The clause, verbatim. The answer to "where does it say that?".
    pub fn verbatim(&self) -> &str {
        self.extracted.verbatim()
    }

    /// The value and its citation together.
    pub fn extracted(&self) -> &ExtractedTerm<TermValue> {
        &self.extracted
    }

    /// Whether this is an unresolved clause.
    pub fn is_ambiguous(&self) -> bool {
        self.extracted.value().is_ambiguous()
    }
}

impl fmt::Display for PolicyTerm {
    /// Renders the term *and* its citation. There is deliberately no `Display`
    /// that renders the value alone: a bare value in a log or a report is
    /// precisely the artefact this crate refuses to produce.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] — {} — \"{}\"",
            self.value(),
            self.scope,
            self.provenance(),
            self.verbatim()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DeductibleBasis;
    use kopitiam_insurance::{
        ClauseId, Currency, DocumentId, Money, PageNumber, SectionPath, SourceText,
    };

    fn provenance(text: &str) -> Provenance {
        Provenance::new(
            DocumentId::new("synthetic-wording.pdf").unwrap(),
            PageNumber::new(3).unwrap(),
            SectionPath::new(["Part 2 — Cost sharing"]),
            ClauseId::printed("2.1").unwrap(),
            SourceText::new(text).unwrap(),
        )
    }

    fn term(value: TermValue, text: &str) -> PolicyTerm {
        PolicyTerm::new(ExtractedTerm::new(value, provenance(text)), Scope::any())
    }

    fn sgd(major: i64) -> MonetaryAmount {
        MonetaryAmount::new(Money::from_cents(major * 100), Currency::Iso("SGD".into()))
    }

    #[test]
    fn a_term_always_carries_its_clause() {
        let t = term(
            TermValue::Deductible(Deductible {
                amount: sgd(3_500),
                basis: DeductibleBasis::PerPolicyYear,
            }),
            "2.1 The Deductible is S$3,500 for each policy year.",
        );
        assert_eq!(
            t.verbatim(),
            "2.1 The Deductible is S$3,500 for each policy year."
        );
        // Display renders the citation alongside the value, always.
        let rendered = t.to_string();
        assert!(rendered.contains("clause 2.1"));
        assert!(rendered.contains("p.3"));
        assert!(rendered.contains("S$3,500"));
    }

    /// An ambiguity is a term, not a hole. A consumer hunting for the deductible
    /// must *find* the unresolved deductible clause, not conclude the policy has
    /// no deductible.
    #[test]
    fn an_ambiguity_is_findable_as_the_kind_it_was_trying_to_be() {
        let text = "2.1 The Deductible shall be such sum as we may determine.";
        let t = term(
            TermValue::Ambiguous(Ambiguity {
                intended: TermKind::Deductible,
                kind: AmbiguityKind::Underspecified,
                note: "amount stated as 'such sum as we may determine'".into(),
                sources: vec![provenance(text)],
            }),
            text,
        );
        assert_eq!(t.kind(), TermKind::Deductible);
        assert!(t.is_ambiguous());
        assert!(!t.value().to_string().is_empty());
    }

    /// Silence must remain distinguishable from exclusion. Collapsing them is how
    /// a person is talked out of a claim they were entitled to make.
    #[test]
    fn silence_is_not_exclusion() {
        assert_ne!(CoverageStatement::Silent, CoverageStatement::StatedExcluded);
        assert!(
            CoverageStatement::Silent
                .to_string()
                .contains("is silent on this")
        );
    }

    #[test]
    fn a_conditional_statement_keeps_its_conditions() {
        let s = CoverageStatement::StatedConditional {
            conditions: vec![
                "the treatment is Medically Necessary".into(),
                "the admission is to a Restructured Hospital".into(),
            ],
        };
        assert!(s.to_string().contains("Medically Necessary"));
        assert!(s.to_string().contains("Restructured Hospital"));
    }

    #[test]
    fn comparable_terms_declare_the_definitions_they_rest_on() {
        assert!(
            TermKind::Deductible
                .depends_on_definitions()
                .contains(&"claimable amount")
        );
        assert!(
            TermKind::Coverage
                .depends_on_definitions()
                .contains(&"hospitalisation")
        );
        assert!(TermKind::IntegrationMode.depends_on_definitions().is_empty());
    }

    /// The two integration modes must remain distinguishable and must both
    /// describe themselves, because the whole point is that a reader is made to
    /// notice which one their policy uses.
    #[test]
    fn the_two_integration_modes_describe_themselves_differently() {
        let inclusive = IntegrationMode::InclusiveOfBasic.to_string();
        let excess = IntegrationMode::ExcessOfBasic.to_string();
        assert_ne!(inclusive, excess);
        assert!(inclusive.contains("inclusive"));
        assert!(excess.contains("does not pay") || excess.contains("leaves unpaid"));
    }
}
