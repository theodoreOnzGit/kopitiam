//! What the *document says* the split of a bill is — or an honest refusal.
//!
//! # Read this before trusting a number that comes out of here
//!
//! [`compute_cost_share`] does not tell you what you will pay. It tells you what
//! the clauses it managed to extract, applied in the order those clauses set out,
//! come to — **assuming the claim is payable at all**, which is a question it
//! does not touch and cannot answer. Every figure it returns comes with the
//! clauses it was computed from, and with a list of the things it deliberately
//! did not evaluate. Read both.
//!
//! It refuses far more readily than it computes. That is the intended behaviour,
//! and the [`CostShareRefusal`] variants are worth reading as carefully as the
//! arithmetic: each is a case where returning a number would have meant inventing
//! a term the document does not contain.
//!
//! # The order of operations, and why it matters so much
//!
//! Deductible **first**, then co-insurance on **what is left**:
//!
//! ```text
//! claimable                                     10,000
//!   - deductible (borne by insured)            - 3,500
//!                                              -------
//!   remainder                                    6,500
//!   - co-insurance @10% (borne by insured)   -     650
//!                                              -------
//!   insurer pays (subject to claim limits)       5,850
//!   insured pays  3,500 + 650                =   4,150
//! ```
//!
//! Apply co-insurance to the *whole* bill first and the same policy appears to
//! leave the insured paying 4,500 and the insurer 5,500. Same clauses, same
//! numbers, S$350 of difference — and both orderings look entirely reasonable if
//! you have not thought about it. On a S$150,000 bill the same mistake is worth
//! thousands.
//!
//! # The boundary everybody gets wrong
//!
//! **A bill below the deductible is paid entirely by the insured.** The insurer
//! pays *nothing* — not "the bill minus a small share", not "most of it".
//! Nothing. This is the single most common misunderstanding of how a shield plan
//! works, and it is why people who *have* cover are blindsided by the bill for a
//! whole day-surgery episode. Tested explicitly in
//! `a_bill_below_the_deductible_is_paid_entirely_by_the_insured`.

use std::fmt;

use kopitiam_insurance::{MonetaryAmount, Provenance};

use crate::domain::{ClaimLimit, DeductibleBasis, TreatmentContext};
use crate::money::{Amount, MoneyError};
use crate::policy::{LayerKind, PolicyId, PolicyLayer, PolicyStack};
use crate::term::{
    Ambiguity, CoPaymentBase, CoverageStatement, IntegrationMode, PolicyTerm, TermKind, TermValue,
};

/// A hospital bill, split into what was charged and what the policy makes
/// claimable.
///
/// # Why `claimable` is supplied by the caller and not derived
///
/// Not every dollar of a hospital bill is claimable. Non-medical items, charges
/// above what the wording calls reasonable and customary, treatment excluded
/// elsewhere in the document — all of it is stripped out before any deductible
/// applies. Deriving `claimable` from `total` would mean adjudicating the whole
/// exclusions schedule against an itemised bill, which this crate does not do and
/// this scaffold does not pretend to.
///
/// So the caller supplies it, from the insurer's own settlement or from a human
/// reading the exclusions. The resulting breakdown always carries
/// [`Caveat::ClaimableAmountSuppliedByCaller`] to keep that visible: a figure
/// computed from a `claimable` that was really just a guess at `total` is a
/// figure with a large error nobody can see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bill {
    total: Amount,
    claimable: Amount,
}

/// A bill that does not describe a real bill.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BillError {
    /// The total and claimable amounts were in different currencies.
    #[error("bill total ({total}) and claimable amount ({claimable}) are in different currencies")]
    CurrencyMismatch {
        /// The total supplied.
        total: Amount,
        /// The claimable amount supplied.
        claimable: Amount,
    },
    /// One of the amounts was negative.
    #[error("a bill amount must not be negative")]
    Negative,
    /// More was claimable than was charged.
    #[error("claimable amount ({claimable}) exceeds the bill total ({total})")]
    ClaimableExceedsTotal {
        /// The claimable amount supplied.
        claimable: Amount,
        /// The total supplied.
        total: Amount,
    },
}

impl Bill {
    /// Records a bill.
    pub fn new(total: Amount, claimable: Amount) -> Result<Self, BillError> {
        if total.currency() != claimable.currency() {
            return Err(BillError::CurrencyMismatch { total, claimable });
        }
        if total.cents() < 0 || claimable.cents() < 0 {
            return Err(BillError::Negative);
        }
        if claimable.cents() > total.cents() {
            return Err(BillError::ClaimableExceedsTotal { claimable, total });
        }
        Ok(Self { total, claimable })
    }

    /// The amount charged.
    pub fn total(&self) -> &Amount {
        &self.total
    }

    /// The amount the policy makes claimable, per the caller.
    pub fn claimable(&self) -> &Amount {
        &self.claimable
    }
}

/// Something the crate deliberately did not do, attached to a result so that the
/// result cannot be read without it being seen.
///
/// A caveat is not a warning to be logged and forgotten. Several of these
/// (notably [`Caveat::EligibilityTermNotEvaluated`]) mean the number beside them
/// may not apply at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Caveat {
    /// The claimable amount came from the caller, not from this crate reading the
    /// exclusions schedule. Always present.
    ClaimableAmountSuppliedByCaller,

    /// A percentage did not divide evenly and had to be rounded. The wording
    /// stated no rounding rule; we rounded half away from zero. See
    /// [`crate::money::Amount::apply`].
    RoundingRuleNotStated {
        /// Which step rounded.
        step: String,
    },

    /// A per-policy-year, lifetime or otherwise cumulative ceiling was applied
    /// **as if this were the only claim in the period**.
    ///
    /// This crate has no claim history. If the insured has already claimed this
    /// year, the real remaining limit is lower than the one used here, and the
    /// insurer's share is correspondingly smaller than the figure shown.
    CumulativeLimitAppliedAsIfSingleClaim {
        /// Which policy's limit.
        policy: PolicyId,
        /// What was applied, in words.
        what: String,
        /// The clause it came from.
        clause: Provenance,
    },

    /// The plan's benefit is stated to be inclusive of the universal basic
    /// scheme's payout, so the insurer's figure below *contains* the basic
    /// scheme's share. This crate does not split the two apart — that would need
    /// the basic scheme's own limits applied to an itemised bill.
    BasicSchemeShareNotSeparated {
        /// The clause establishing the integration mode.
        clause: Provenance,
    },

    /// The stack contains no universal basic scheme document, so nothing here
    /// reflects what that scheme pays.
    NoUniversalBasicDocument,

    /// A term that decides whether the claim is payable **at all**, which this
    /// crate does not evaluate. Read it.
    ///
    /// Waiting periods, pre-existing-condition clauses, exclusions. The split
    /// shown is conditional on none of these biting.
    EligibilityTermNotEvaluated {
        /// The clause, verbatim, with its location.
        term: Box<PolicyTerm>,
    },

    /// An unresolvable clause of a kind that *was* used: a more specific clause
    /// governed, so this one did not decide the number — but it is unresolved, and
    /// if the specificity ranking is wrong, it should have.
    AmbiguousClauseNotUsed {
        /// The unresolved clause.
        ambiguity: Box<Ambiguity>,
    },

    /// A rider's co-payment, computed on the claimable amount, came out larger
    /// than the amount the rider was going to absorb. It was clamped, so the rider
    /// never leaves the insured worse off than having no rider at all. The wording
    /// did not state what happens here.
    CoPaymentClampedToAbsorbedAmount {
        /// Which rider.
        policy: PolicyId,
    },
}

impl fmt::Display for Caveat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClaimableAmountSuppliedByCaller => f.write_str(
                "The claimable amount was supplied by the caller, not derived from the document's \
                 exclusions. If it is wrong, every figure here is wrong.",
            ),
            Self::RoundingRuleNotStated { step } => write!(
                f,
                "'{step}' did not divide evenly. The wording states no rounding rule; rounded \
                 half away from zero (may differ from the insurer by one cent)."
            ),
            Self::CumulativeLimitAppliedAsIfSingleClaim {
                policy,
                what,
                clause,
            } => write!(
                f,
                "{policy}: {what} is cumulative, and was applied as if this were the only claim \
                 in the period. Prior claims would reduce it. Clause: {clause} — \"{}\"",
                clause.verbatim()
            ),
            Self::BasicSchemeShareNotSeparated { clause } => write!(
                f,
                "The insurer's share below includes the universal basic scheme's payout; this \
                 crate does not separate them. Clause: {clause} — \"{}\"",
                clause.verbatim()
            ),
            Self::NoUniversalBasicDocument => f.write_str(
                "No universal basic scheme document was supplied, so nothing here reflects what \
                 that scheme pays.",
            ),
            Self::EligibilityTermNotEvaluated { term } => write!(
                f,
                "NOT EVALUATED — this clause may mean the claim is not payable at all, and this \
                 crate does not decide that: {term}"
            ),
            Self::AmbiguousClauseNotUsed { ambiguity } => write!(
                f,
                "An unresolved {} clause was overridden by a more specific one and did not decide \
                 the figures ({}). Read it: {}",
                ambiguity.intended,
                ambiguity.kind,
                ambiguity
                    .sources
                    .iter()
                    .map(|p| format!("{p} — \"{}\"", p.verbatim()))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Self::CoPaymentClampedToAbsorbedAmount { policy } => write!(
                f,
                "{policy}: the co-payment exceeded what the rider was going to absorb and was \
                 clamped, so the rider does not leave the insured worse off. The wording does not \
                 say what happens here."
            ),
        }
    }
}

/// Why the crate declined to produce a number.
///
/// # A refusal is a correct answer
///
/// Every variant here is a case where the honest report is *"I could not
/// determine this — here is the clause, read it yourself"*. The alternative in
/// each case is a plausible number derived from a term the document does not
/// contain, and that is the failure this crate exists to make impossible.
///
/// **There is no silent default anywhere.** A missing deductible does not become
/// zero. A missing claim limit does not become infinity. A rider with no
/// co-payment clause does not become a 0% co-payment. Each of those defaults
/// would err in the insured's favour and quietly overstate their cover — the
/// direction of error that gets someone into an operating theatre they cannot
/// pay for.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CostShareRefusal {
    /// The document never stated a term the calculation needs.
    ///
    /// Not an internal error. Usually it means the wording puts the figure in a
    /// benefit schedule or a table this scaffold's extraction rules did not read.
    /// The right response is to go and read the schedule, not to pick a number.
    #[error(
        "{policy}: no {needed} was extracted for {context}. Not defaulting to a value — the \
         document must state it. (If the wording states it in a benefit table, this scaffold's \
         extraction did not read it.)"
    )]
    MissingTerm {
        /// The policy that lacks the term.
        policy: PolicyId,
        /// What was needed.
        needed: TermKind,
        /// The treatment it was needed for.
        context: String,
    },

    /// A clause governs, and we could not resolve it. Here it is.
    #[error("{policy}: the governing {} clause could not be resolved ({}). Read it: {}",
        .ambiguity.intended,
        .ambiguity.kind,
        .ambiguity.sources.iter()
            .map(|p| format!("{p} — \"{}\"", p.verbatim()))
            .collect::<Vec<_>>().join(" | "))]
    AmbiguousTerm {
        /// The policy.
        policy: PolicyId,
        /// The unresolved clause, with its text.
        ambiguity: Box<Ambiguity>,
    },

    /// Two equally specific clauses state different things, and nothing in the
    /// document says which governs. Picking one would be a coin toss dressed up as
    /// an answer.
    #[error("{policy}: {kind} is stated more than once, differently, at the same level of \
             specificity, and the document does not say which governs. Clauses: {}",
        .clauses.iter()
            .map(|p| format!("{p} — \"{}\"", p.verbatim()))
            .collect::<Vec<_>>().join(" | "))]
    ConflictingTerms {
        /// The policy.
        policy: PolicyId,
        /// The kind of term.
        kind: TermKind,
        /// Every conflicting clause, verbatim.
        clauses: Vec<Provenance>,
    },

    /// A statement this crate cannot evaluate: the clause makes the answer
    /// conditional on facts about the patient, or says nothing at all.
    #[error("{policy}: cannot determine {what} — {statement}. This is not a question this crate \
             answers. Clause: {clause} — \"{}\"", .clause.verbatim())]
    CannotEvaluateStatement {
        /// The policy.
        policy: PolicyId,
        /// What we were trying to determine.
        what: String,
        /// What the document actually said.
        statement: Box<CoverageStatement>,
        /// The clause.
        clause: Box<Provenance>,
    },

    /// The plan pays only what the basic scheme leaves unpaid, but no basic scheme
    /// document was supplied — so what it leaves unpaid is unknown.
    #[error(
        "the plan's benefit is stated to apply only to what the universal basic scheme leaves \
         unpaid, but no basic scheme document was supplied, so that residual is unknown. \
         Clause: {clause} — \"{}\"", .clause.verbatim()
    )]
    BasicSchemeRequiredButAbsent {
        /// The clause establishing the integration mode.
        clause: Box<Provenance>,
    },

    /// The governing limit needs information a bill does not carry (a per-day limit
    /// needs a length of stay).
    #[error("{policy}: the governing claim limit ({limit}) cannot be applied to a bill alone — it \
             needs information this crate was not given. Clause: {clause} — \"{}\"",
        .clause.verbatim())]
    LimitNotApplicable {
        /// The policy.
        policy: PolicyId,
        /// The limit that could not be applied.
        limit: Box<ClaimLimit>,
        /// The clause.
        clause: Box<Provenance>,
    },

    /// The document printed an amount whose currency it did not identify, or the
    /// arithmetic crossed currencies.
    #[error("{policy}: {error}")]
    Money {
        /// The policy whose figures could not be computed with.
        policy: PolicyId,
        /// What went wrong.
        error: MoneyError,
    },
}

/// One step of the calculation, with every clause it rested on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostShareStep {
    /// What this step did, in words.
    pub description: String,
    /// The amount going in.
    pub amount_before: Amount,
    /// The amount going out.
    pub amount_after: Amount,
    /// Who bore the difference.
    pub borne_by: BorneBy,
    /// **Every clause this step depended on.** Never empty.
    pub basis: Vec<Provenance>,
}

/// Who a slice of the bill fell to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorneBy {
    /// The patient.
    Insured,
    /// The insurer (or the basic scheme, where the two are not separated).
    Insurer,
    /// A rider.
    Rider,
}

impl fmt::Display for BorneBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Insured => "insured",
            Self::Insurer => "insurer",
            Self::Rider => "rider",
        })
    }
}

/// What one policy layer did to the amount presented to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOutcome {
    /// Which policy.
    pub policy: PolicyId,
    /// The policy's name.
    pub name: String,
    /// Which layer of the stack.
    pub kind: LayerKind,
    /// The amount presented to this layer.
    pub presented: Amount,
    /// The slice the insured bore as the deductible.
    pub deductible_borne: Amount,
    /// The slice the insured bore as co-insurance (or, for a rider, as the
    /// residual co-payment).
    pub co_insurance_borne: Amount,
    /// The slice above the policy's claim limit, which nobody in this layer pays.
    pub above_limit: Amount,
    /// What this layer's insurer pays.
    pub insurer_pays: Amount,
    /// What the insured is left with after this layer.
    pub insured_bears: Amount,
    /// The steps, with their clauses.
    pub steps: Vec<CostShareStep>,
}

/// The split, with every clause it was computed from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostShareBreakdown {
    /// The bill.
    pub bill: Bill,
    /// The treatment circumstances.
    pub context: TreatmentContext,
    /// What each layer did, bottom-up.
    pub layers: Vec<LayerOutcome>,
    /// What the insured is left paying, per the documents.
    pub insured_pays: Amount,
    /// What the insurers pay in total, per the documents.
    pub insurers_pay: Amount,
    /// Everything the crate did not do. **Read these.**
    pub caveats: Vec<Caveat>,
}

impl CostShareBreakdown {
    /// Every clause every figure above rested on, in order.
    ///
    /// The answer to "where does it say that?" for the *whole* computation. If
    /// this list is ever empty, something is very wrong.
    pub fn basis(&self) -> Vec<&Provenance> {
        self.layers
            .iter()
            .flat_map(|l| l.steps.iter())
            .flat_map(|s| s.basis.iter())
            .collect()
    }

    /// A full, human-readable account: the arithmetic, then every clause it came
    /// from, then every caveat.
    ///
    /// Deliberately verbose. A caller that wants only the number can take
    /// [`Self::insured_pays`] — but the point of this crate is that the number is
    /// not the deliverable. The reasoning is.
    pub fn explain(&self) -> String {
        let mut out = format!(
            "Bill {} (claimable {}), {}\n\nWHAT THE DOCUMENTS STATE:\n",
            self.bill.total(),
            self.bill.claimable(),
            self.context
        );

        for layer in &self.layers {
            out.push_str(&format!("\n  {} [{}]\n", layer.name, layer.kind));
            for step in &layer.steps {
                out.push_str(&format!(
                    "    {} : {} -> {} (borne by {})\n",
                    step.description, step.amount_before, step.amount_after, step.borne_by
                ));
                for p in &step.basis {
                    out.push_str(&format!("        per {p}\n            \"{}\"\n", p.verbatim()));
                }
            }
        }

        out.push_str(&format!(
            "\n  INSURED PAYS  {}\n  INSURERS PAY  {}\n\nWHAT THIS DOES NOT TELL YOU:\n",
            self.insured_pays, self.insurers_pay
        ));
        for caveat in &self.caveats {
            out.push_str(&format!("  - {caveat}\n"));
        }
        out.push_str(
            "\nThis is a reading of the documents. It is not advice, and it is not a \
             determination that any claim is payable.\n",
        );
        out
    }
}

/// Reads the documents and works out what they say the split is — or refuses.
///
/// See the module docs for the order of operations, and [`crate::IntegrationMode`]
/// for the stacking question this will refuse to guess at.
///
/// # What this is not
///
/// It is not a claims decision, it is not advice, and it does not know whether the
/// treatment is covered. It computes cost-sharing arithmetic *given that a claim
/// is payable*, from clauses it can quote, and it tells you every clause it could
/// not evaluate.
///
/// # Errors
///
/// See [`CostShareRefusal`]. Every variant is a case where a number would have had
/// to be invented.
pub fn compute_cost_share(
    stack: &PolicyStack,
    bill: &Bill,
    ctx: &TreatmentContext,
) -> Result<CostShareBreakdown, CostShareRefusal> {
    let mut caveats = vec![Caveat::ClaimableAmountSuppliedByCaller];

    // Eligibility clauses are surfaced, never evaluated — and surfaced first, so
    // that the habit is established before any arithmetic happens.
    for term in stack.unevaluated_eligibility_terms() {
        caveats.push(Caveat::EligibilityTermNotEvaluated {
            term: Box::new(term.clone()),
        });
    }

    let mut layers: Vec<LayerOutcome> = Vec::new();
    let basic = stack.universal_basic();
    let top_up = stack.integrated_top_up();

    // What the insuring layers leave the insured holding. Riders eat into this.
    let residual_from_plan: Option<LayerOutcome> = match (basic, top_up) {
        // Basic scheme only: no integration question arises.
        (Some(basic), None) => {
            let outcome = apply_layer(basic, bill.claimable().clone(), ctx, &mut caveats)?;
            layers.push(outcome.clone());
            Some(outcome)
        }

        // A private plan is present: how it stacks is the whole question.
        (basic, Some(plan)) => {
            let mode_term = resolve(plan, TermKind::IntegrationMode, ctx, &mut caveats)?;
            let TermValue::IntegrationMode(mode) = mode_term.value() else {
                unreachable!("resolve() honours the requested TermKind")
            };

            let presented = match (mode, basic) {
                (IntegrationMode::ExcessOfBasic, Some(basic)) => {
                    // A true second layer: the basic scheme runs first, and the
                    // private plan sees only what the basic scheme left the
                    // insured holding.
                    let outcome = apply_layer(basic, bill.claimable().clone(), ctx, &mut caveats)?;
                    let residual = outcome.insured_bears.clone();
                    layers.push(outcome);
                    residual
                }
                (IntegrationMode::ExcessOfBasic, None) => {
                    return Err(CostShareRefusal::BasicSchemeRequiredButAbsent {
                        clause: Box::new(mode_term.provenance().clone()),
                    });
                }
                (IntegrationMode::InclusiveOfBasic, basic) => {
                    // The plan's deductible bites on the *whole* claimable amount.
                    // The basic scheme's payout is a component of what the insurer
                    // pays, settled behind the scenes — it does **not** shrink the
                    // bill before the deductible. This is the misunderstanding.
                    caveats.push(Caveat::BasicSchemeShareNotSeparated {
                        clause: mode_term.provenance().clone(),
                    });
                    if basic.is_none() {
                        caveats.push(Caveat::NoUniversalBasicDocument);
                    }
                    bill.claimable().clone()
                }
            };

            let outcome = apply_layer(plan, presented, ctx, &mut caveats)?;
            layers.push(outcome.clone());
            Some(outcome)
        }

        // `PolicyStack::new` guarantees at least one layer and forbids a rider
        // without a top-up, so this is a stack of nothing that pays a hospital
        // bill.
        (None, None) => {
            caveats.push(Caveat::NoUniversalBasicDocument);
            None
        }
    };

    let zero = bill.claimable().zero_like();
    let mut insured_pays = residual_from_plan
        .as_ref()
        .map_or_else(|| zero.clone(), |o| o.insured_bears.clone());
    let mut rider_pays = zero.clone();

    // Riders absorb what the plan left the insured holding.
    if let Some(plan_outcome) = residual_from_plan.as_ref() {
        for rider in stack.riders() {
            let outcome = apply_rider(rider, plan_outcome, bill, ctx, &insured_pays, &mut caveats)?;
            rider_pays = money(rider, rider_pays.add(&outcome.insurer_pays))?;
            insured_pays = outcome.insured_bears.clone();
            layers.push(outcome);
        }
    }

    let mut insurers_pay = rider_pays;
    for layer in layers.iter().filter(|l| l.kind != LayerKind::Rider) {
        insurers_pay = insurers_pay
            .add(&layer.insurer_pays)
            .map_err(|error| CostShareRefusal::Money {
                policy: layer.policy.clone(),
                error,
            })?;
    }

    Ok(CostShareBreakdown {
        bill: bill.clone(),
        context: ctx.clone(),
        layers,
        insured_pays,
        insurers_pay,
        caveats,
    })
}

/// Runs one insuring layer's deductible / co-insurance / limit over an amount.
fn apply_layer(
    layer: &PolicyLayer,
    presented: Amount,
    ctx: &TreatmentContext,
    caveats: &mut Vec<Caveat>,
) -> Result<LayerOutcome, CostShareRefusal> {
    let deductible_term = resolve(layer, TermKind::Deductible, ctx, caveats)?;
    let co_insurance_term = resolve(layer, TermKind::CoInsurance, ctx, caveats)?;
    let limit_term = resolve(layer, TermKind::ClaimLimit, ctx, caveats)?;

    let TermValue::Deductible(deductible) = deductible_term.value() else {
        unreachable!("resolve() honours the requested TermKind")
    };
    let TermValue::CoInsurance(co_insurance) = co_insurance_term.value() else {
        unreachable!("resolve() honours the requested TermKind")
    };
    let TermValue::ClaimLimit(limit) = limit_term.value() else {
        unreachable!("resolve() honours the requested TermKind")
    };

    let mut steps = Vec::new();

    // --- Step 1: the deductible, off the top. --------------------------------
    //
    // The insured bears the whole of it, up to the size of the bill. A bill
    // smaller than the deductible is borne entirely by the insured, and nothing
    // reaches step 2 at all.
    let deductible_amount = computable(layer, &deductible.amount)?;
    let deductible_borne = money(layer, presented.min(&deductible_amount))?;
    let after_deductible = money(layer, presented.saturating_sub(&deductible_amount))?;

    steps.push(CostShareStep {
        description: format!("deductible ({deductible_amount} {})", deductible.basis),
        amount_before: presented.clone(),
        amount_after: after_deductible.clone(),
        borne_by: BorneBy::Insured,
        basis: vec![deductible_term.provenance().clone()],
    });

    if deductible.basis == DeductibleBasis::PerPolicyYear {
        caveats.push(Caveat::CumulativeLimitAppliedAsIfSingleClaim {
            policy: layer.id().clone(),
            what: format!("the deductible ({deductible_amount} per policy year)"),
            clause: deductible_term.provenance().clone(),
        });
    }

    // --- Step 2: co-insurance, on what is left. ------------------------------
    //
    // On the *remainder*, not on the original bill. See the module docs for the
    // size of the error the other way round.
    let (co_insurance_raw, rounded) = money(layer, after_deductible.apply(co_insurance.rate))?;
    if rounded {
        caveats.push(Caveat::RoundingRuleNotStated {
            step: format!(
                "{} co-insurance at {}%",
                layer.name(),
                co_insurance.rate.to_decimal_string()
            ),
        });
    }

    let co_insurance_borne = match &co_insurance.cap {
        Some(cap) => {
            let cap = computable(layer, cap)?;
            let capped = money(layer, co_insurance_raw.min(&cap))?;
            if capped != co_insurance_raw {
                caveats.push(Caveat::CumulativeLimitAppliedAsIfSingleClaim {
                    policy: layer.id().clone(),
                    what: format!("the cap on the insured's co-insurance ({cap})"),
                    clause: co_insurance_term.provenance().clone(),
                });
            }
            capped
        }
        None => co_insurance_raw,
    };

    let after_co_insurance = money(layer, after_deductible.saturating_sub(&co_insurance_borne))?;

    steps.push(CostShareStep {
        description: format!(
            "co-insurance ({}% of the amount above the deductible)",
            co_insurance.rate.to_decimal_string()
        ),
        amount_before: after_deductible,
        amount_after: after_co_insurance.clone(),
        borne_by: BorneBy::Insured,
        basis: vec![co_insurance_term.provenance().clone()],
    });

    // --- Step 3: the insurer's ceiling. --------------------------------------
    //
    // Whatever is above the limit falls back on the insured. It is not absorbed by
    // anyone: it is simply outside the cover.
    let insurer_gross = after_co_insurance;
    let insurer_pays = match limit {
        ClaimLimit::AsCharged => insurer_gross.clone(),
        ClaimLimit::PerDay(_) => {
            // A per-day ceiling needs a length of stay, which a bill does not
            // carry. Refuse rather than quietly treating it as a per-claim limit.
            return Err(CostShareRefusal::LimitNotApplicable {
                policy: layer.id().clone(),
                limit: Box::new(limit.clone()),
                clause: Box::new(limit_term.provenance().clone()),
            });
        }
        ClaimLimit::PerClaim(cap) | ClaimLimit::PerPolicyYear(cap) | ClaimLimit::Lifetime(cap) => {
            let cap = computable(layer, cap)?;
            money(layer, insurer_gross.min(&cap))?
        }
    };

    if matches!(
        limit,
        ClaimLimit::PerPolicyYear(_) | ClaimLimit::Lifetime(_)
    ) {
        caveats.push(Caveat::CumulativeLimitAppliedAsIfSingleClaim {
            policy: layer.id().clone(),
            what: format!("the claim limit ({limit})"),
            clause: limit_term.provenance().clone(),
        });
    }

    let above_limit = money(layer, insurer_gross.saturating_sub(&insurer_pays))?;

    steps.push(CostShareStep {
        description: format!("insurer's limit ({limit})"),
        amount_before: insurer_gross,
        amount_after: insurer_pays.clone(),
        borne_by: BorneBy::Insurer,
        basis: vec![limit_term.provenance().clone()],
    });

    let insured_bears = money(
        layer,
        money(layer, deductible_borne.add(&co_insurance_borne))?.add(&above_limit),
    )?;

    Ok(LayerOutcome {
        policy: layer.id().clone(),
        name: layer.name().to_string(),
        kind: layer.kind(),
        presented,
        deductible_borne,
        co_insurance_borne,
        above_limit,
        insurer_pays,
        insured_bears,
        steps,
    })
}

/// Runs a rider over what the plan left the insured holding.
///
/// # What a rider does and does not absorb
///
/// A rider reimburses the base plan's **deductible** and **co-insurance** — the
/// two slices the plan deliberately leaves with the insured to keep them exposed
/// to the size of the bill. It does **not** pay amounts above the plan's claim
/// limit: those are outside the plan's cover altogether, and there is nothing
/// there for a rider to reimburse. Treating a rider as covering them would
/// understate a catastrophic bill by exactly the amount that makes it
/// catastrophic.
fn apply_rider(
    rider: &PolicyLayer,
    plan: &LayerOutcome,
    bill: &Bill,
    ctx: &TreatmentContext,
    insured_currently_bears: &Amount,
    caveats: &mut Vec<Caveat>,
) -> Result<LayerOutcome, CostShareRefusal> {
    let coverage_term = resolve(rider, TermKind::RiderCoverage, ctx, caveats)?;
    let TermValue::RiderCoverage(coverage) = coverage_term.value() else {
        unreachable!("resolve() honours the requested TermKind")
    };

    // A rider that says nothing, or says "it depends", is a rider whose effect we
    // cannot compute. Refuse; do not assume it pays.
    let absorbs = |statement: &CoverageStatement,
                   what: &str,
                   amount: &Amount|
     -> Result<Amount, CostShareRefusal> {
        match statement {
            CoverageStatement::StatedPayable => Ok(amount.clone()),
            CoverageStatement::StatedExcluded => Ok(amount.zero_like()),
            other => Err(CostShareRefusal::CannotEvaluateStatement {
                policy: rider.id().clone(),
                what: what.to_string(),
                statement: Box::new(other.clone()),
                clause: Box::new(coverage_term.provenance().clone()),
            }),
        }
    };

    let absorbed_deductible = absorbs(
        &coverage.deductible,
        "whether the rider reimburses the plan's deductible",
        &plan.deductible_borne,
    )?;
    let absorbed_co_insurance = absorbs(
        &coverage.co_insurance,
        "whether the rider reimburses the plan's co-insurance",
        &plan.co_insurance_borne,
    )?;

    let would_absorb = money(rider, absorbed_deductible.add(&absorbed_co_insurance))?;

    let mut steps = vec![CostShareStep {
        description: format!(
            "rider absorbs the plan's deductible ({absorbed_deductible}) and co-insurance \
             ({absorbed_co_insurance})"
        ),
        amount_before: insured_currently_bears.clone(),
        amount_after: money(rider, insured_currently_bears.saturating_sub(&would_absorb))?,
        borne_by: BorneBy::Rider,
        basis: vec![coverage_term.provenance().clone()],
    }];

    // The residual co-payment. A rider with no co-payment clause is a refusal, not
    // a 0% co-payment — see the "no silent default" note on `CostShareRefusal`. A
    // rider that genuinely charges none must *say* it charges none.
    let co_payment_term = resolve(rider, TermKind::CoPayment, ctx, caveats)?;
    let TermValue::CoPayment(co_payment_terms) = co_payment_term.value() else {
        unreachable!("resolve() honours the requested TermKind")
    };

    let base_amount = match co_payment_terms.base {
        CoPaymentBase::AmountRiderWouldAbsorb => would_absorb.clone(),
        CoPaymentBase::ClaimableAmount => bill.claimable().clone(),
    };

    let (co_payment_raw, rounded) = money(rider, base_amount.apply(co_payment_terms.rate))?;
    if rounded {
        caveats.push(Caveat::RoundingRuleNotStated {
            step: format!(
                "{} co-payment at {}%",
                rider.name(),
                co_payment_terms.rate.to_decimal_string()
            ),
        });
    }

    let mut co_payment = match &co_payment_terms.cap {
        Some(cap) => {
            let cap = computable(rider, cap)?;
            let capped = money(rider, co_payment_raw.min(&cap))?;
            if capped != co_payment_raw {
                caveats.push(Caveat::CumulativeLimitAppliedAsIfSingleClaim {
                    policy: rider.id().clone(),
                    what: format!("the cap on the rider's co-payment ({cap})"),
                    clause: co_payment_term.provenance().clone(),
                });
            }
            capped
        }
        None => co_payment_raw,
    };

    // A rider must never leave the insured worse off than having no rider at all.
    if co_payment.cents() > would_absorb.cents() {
        co_payment = would_absorb.clone();
        caveats.push(Caveat::CoPaymentClampedToAbsorbedAmount {
            policy: rider.id().clone(),
        });
    }

    let rider_pays = money(rider, would_absorb.saturating_sub(&co_payment))?;
    let insured_bears = money(
        rider,
        money(rider, insured_currently_bears.saturating_sub(&would_absorb))?.add(&co_payment),
    )?;

    steps.push(CostShareStep {
        description: format!(
            "rider co-payment ({}% of {})",
            co_payment_terms.rate.to_decimal_string(),
            co_payment_terms.base
        ),
        amount_before: would_absorb,
        amount_after: rider_pays.clone(),
        borne_by: BorneBy::Insured,
        basis: vec![co_payment_term.provenance().clone()],
    });

    let zero = insured_currently_bears.zero_like();
    Ok(LayerOutcome {
        policy: rider.id().clone(),
        name: rider.name().to_string(),
        kind: rider.kind(),
        presented: insured_currently_bears.clone(),
        deductible_borne: zero.clone(),
        co_insurance_borne: co_payment,
        above_limit: zero,
        insurer_pays: rider_pays,
        insured_bears,
        steps,
    })
}

/// Finds the one clause of a kind that governs this treatment — or refuses.
///
/// The rules, in order:
///
/// 1. No applicable clause at all -> [`CostShareRefusal::MissingTerm`]. **Never a
///    default.**
/// 2. The most specific applicable clauses win: one written for
///    *(Private, non-panel)* beats one written for *(Private)*, which beats an
///    unqualified one. That is how a wording is read — the specific provision
///    governs.
/// 3. If a winning clause is unresolved -> [`CostShareRefusal::AmbiguousTerm`],
///    carrying its text.
/// 4. If two equally specific clauses say different things ->
///    [`CostShareRefusal::ConflictingTerms`], carrying both. The document does not
///    say which governs, so neither do we.
/// 5. Unresolved clauses that *lost* on specificity become a
///    [`Caveat::AmbiguousClauseNotUsed`]: they did not decide the number, but the
///    reader should know they are there.
fn resolve<'a>(
    layer: &'a PolicyLayer,
    kind: TermKind,
    ctx: &TreatmentContext,
    caveats: &mut Vec<Caveat>,
) -> Result<&'a PolicyTerm, CostShareRefusal> {
    let candidates = layer.applicable_terms(kind, ctx);

    let Some(first) = candidates.first() else {
        return Err(CostShareRefusal::MissingTerm {
            policy: layer.id().clone(),
            needed: kind,
            context: ctx.to_string(),
        });
    };

    let top = first.scope().specificity();
    let mut winners: Vec<&PolicyTerm> = Vec::new();
    for candidate in &candidates {
        if candidate.scope().specificity() == top {
            winners.push(candidate);
        } else if let TermValue::Ambiguous(a) = candidate.value() {
            caveats.push(Caveat::AmbiguousClauseNotUsed {
                ambiguity: Box::new(a.clone()),
            });
        }
    }

    for winner in &winners {
        if let TermValue::Ambiguous(a) = winner.value() {
            return Err(CostShareRefusal::AmbiguousTerm {
                policy: layer.id().clone(),
                ambiguity: Box::new(a.clone()),
            });
        }
    }

    if winners.len() > 1 && winners.iter().any(|t| t.value() != winners[0].value()) {
        return Err(CostShareRefusal::ConflictingTerms {
            policy: layer.id().clone(),
            kind,
            clauses: winners
                .iter()
                .map(|t| t.provenance().clone())
                .collect(),
        });
    }

    Ok(winners[0])
}

/// Accepts an amount for arithmetic — or refuses, because the document never said
/// what currency it was in.
///
/// This is the choke point between "what the document printed" and "what we are
/// willing to compute with". See [`crate::money`].
fn computable(layer: &PolicyLayer, amount: &MonetaryAmount) -> Result<Amount, CostShareRefusal> {
    Amount::try_from_extracted(amount).map_err(|error| CostShareRefusal::Money {
        policy: layer.id().clone(),
        error,
    })
}

/// Attributes a monetary error to the policy whose figures caused it.
fn money<T>(layer: &PolicyLayer, result: Result<T, MoneyError>) -> Result<T, CostShareRefusal> {
    result.map_err(|error| CostShareRefusal::Money {
        policy: layer.id().clone(),
        error,
    })
}
