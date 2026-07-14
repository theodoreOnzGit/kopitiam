//! Rules that turn a clause into a typed [`PolicyTerm`] — or into an honest
//! ambiguity.
//!
//! **This module is health-specific.** It is what knows that a deductible comes
//! before co-insurance and that a rider absorbs both. The document machinery it
//! stands on — PDF extraction, clause segmentation, definitions, provenance —
//! is `kopitiam-insurance`'s and `kopitiam-document`'s. This crate writes no PDF
//! parser and no clause segmenter.
//!
//! # These rules are keyword scanners, and they are meant to decline
//!
//! Every rule here is a keyword-and-number match. That is a crude technique,
//! chosen deliberately over anything cleverer, because the failure modes of a
//! crude technique are *legible*: a keyword rule either matches the clause in
//! front of it or it does not, and when it does not, the term is simply not
//! extracted and the calculator refuses. Nobody is misled.
//!
//! A cleverer extractor — an LLM, a statistical parser — fails differently. It
//! produces a plausible figure that is not in the document, with no signal that
//! anything went wrong, and that figure flows into a cost-share calculation that
//! a person reads and believes. In this domain the second failure mode is far
//! worse than the first, and it is worth giving up a great deal of recall to
//! avoid it.
//!
//! So the rules below are conservative to the point of being annoying:
//!
//! * A "deductible" clause with **two** amounts in it becomes an *ambiguity*, not
//!   a guess at which amount is the deductible.
//! * A deductible whose basis (per year? per claim?) is not stated becomes an
//!   ambiguity, not a default to the commoner one.
//! * A clause saying both "inclusive of" and "in excess of" the basic scheme
//!   becomes a *contradiction*, not a coin toss.
//! * A bare `$` becomes [`Currency::Ambiguous`], which the calculator refuses to
//!   compute with. `$` is not SGD; it is `$`.
//!
//! Each is a case where the clause is genuinely unclear, and the only safe output
//! is the clause itself with a note that a human must read it.
//!
//! # What these rules cannot do (and where the fix belongs)
//!
//! They cannot read a benefit table. Wordings routinely state deductibles and
//! limits as a grid of *(ward class x age band)*, and recovering the **scope** of
//! a figure from its row and column headers is real work this scaffold does not
//! attempt. `kopitiam-insurance` already models the table
//! ([`BenefitTable`](kopitiam_insurance::BenefitTable) gives cells with
//! provenance); what is missing is the mapping from a cell's headers to a
//! [`Scope`]. Until that exists, table-stated figures come out unscoped or not at
//! all, and the calculator refuses rather than mis-scoping them. That refusal is
//! correct behaviour, not a bug to be silenced by guessing at the column.

use kopitiam_insurance::{Clause, Currency, ExtractedTerm, MonetaryAmount, Money, Percentage};

use crate::domain::{
    ClaimLimit, CoInsurance, Deductible, DeductibleBasis, PolicyDuration,
    PreExistingConditionTreatment, ProviderNetwork, Scope, WaitingPeriod, WardClass,
};
use crate::policy::LayerKind;
use crate::term::{
    Ambiguity, AmbiguityKind, CoPayment, CoPaymentBase, CoverageStatement, IntegrationMode,
    PolicyTerm, RiderCoverage, TermKind, TermValue,
};

/// How to read a particular document.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// The currency the document states its amounts are in, if it states one.
    ///
    /// # This is an assertion, and it is the caller's to make
    ///
    /// Set it **only** when the document itself says so — typically in a clause
    /// like *"All amounts in this Policy are in Singapore dollars."* It is used
    /// for one thing: deciding what a bare `$` means.
    ///
    /// Leave it `None` if you are not sure. A bare `$` then comes out as
    /// [`Currency::Ambiguous`], and [`crate::cost_share`] **refuses to compute
    /// with it** rather than assuming. That refusal is the safe failure: in a
    /// wording that also mentions US dollars, `$` is a genuinely open question,
    /// and it is not this crate's to close.
    ///
    /// Amounts the document marks explicitly (`S$3,500`, `SGD 3,500`, `US$500`)
    /// ignore this field entirely — the document's own marker always wins.
    pub declared_currency: Option<String>,

    /// Which layer of the stack this document is.
    ///
    /// The rules genuinely differ: a rider's "deductible" clause is a statement
    /// about *the base plan's* deductible, not a deductible of its own, and
    /// reading it as the latter would give a rider a deductible it does not have.
    pub layer: LayerKind,
}

impl ExtractionConfig {
    /// Reads a document whose currency markers are explicit.
    pub fn new(layer: LayerKind) -> Self {
        Self {
            declared_currency: None,
            layer,
        }
    }

    /// Asserts that the document states its amounts are in a given currency.
    ///
    /// Only do this if it does. See [`Self::declared_currency`].
    pub fn with_declared_currency(mut self, code: impl Into<String>) -> Self {
        self.declared_currency = Some(code.into());
        self
    }
}

/// Reads every term the rules can find in a document's clauses.
///
/// Clauses no rule recognises produce nothing — which is correct: a policy
/// document is mostly prose about claims procedure, and inventing terms from it
/// would be worse than useless.
pub fn extract_terms(clauses: &[Clause], config: &ExtractionConfig) -> Vec<PolicyTerm> {
    clauses
        .iter()
        .flat_map(|clause| extract_from_clause(clause, config))
        .collect()
}

/// Reads one clause.
pub fn extract_from_clause(clause: &Clause, config: &ExtractionConfig) -> Vec<PolicyTerm> {
    let lower = clause.text().to_lowercase();
    let mut terms = Vec::new();

    // Definitions first. They are the foundation everything else rests on, and a
    // definitional clause ("'Deductible' means ...") must not *also* be read as a
    // clause that *states* a deductible.
    //
    // `kopitiam-insurance` has already extracted the definition itself into
    // `PolicyDocument::definitions()`, which is where [`crate::compare`] reads it
    // from; all we do here is decline to misread the clause a second time.
    //
    // The test is the word "means", not
    // [`ClauseRole::Definition`](kopitiam_insurance::ClauseRole::Definition).
    // Deliberately: the generic classifier assigns roles by section, so a clause
    // sitting under a "Definitions" heading inherits the role whether or not it
    // defines anything — and in practice it over-applies, marking neighbouring
    // clauses as definitions too. Keying off its verdict would silently drop
    // real terms. Keying off the document's own word for defining something does
    // not.
    if lower.contains(" means ") {
        return terms;
    }

    if let Some(term) = integration_mode_rule(clause, &lower) {
        terms.push(term);
    }

    match config.layer {
        LayerKind::Rider => {
            // A rider's clauses talk about the *base plan's* cost-sharing.
            if let Some(term) = rider_coverage_rule(clause, &lower) {
                terms.push(term);
            }
            if let Some(term) = co_payment_rule(clause, &lower, config) {
                terms.push(term);
            }
        }
        LayerKind::UniversalBasic | LayerKind::IntegratedTopUp => {
            if let Some(term) = deductible_rule(clause, &lower, config) {
                terms.push(term);
            }
            if let Some(term) = co_insurance_rule(clause, &lower, config) {
                terms.push(term);
            }
            if let Some(term) = claim_limit_rule(clause, &lower, config) {
                terms.push(term);
            }
            if let Some(term) = ward_entitlement_rule(clause, &lower) {
                terms.push(term);
            }
        }
    }

    if let Some(term) = waiting_period_rule(clause, &lower) {
        terms.push(term);
    }

    // Pre-existing conditions and general coverage statements both key off
    // exclusion language. If the pre-existing rule fired it has already said, more
    // precisely, everything the coverage rule would — so do not emit both and make
    // a human read the same clause twice.
    if let Some(term) = pre_existing_rule(clause, &lower) {
        terms.push(term);
    } else if let Some(term) = coverage_rule(clause, &lower) {
        terms.push(term);
    }

    terms
}

// ---------------------------------------------------------------------------
// Rules.
// ---------------------------------------------------------------------------

/// Phrases with which a wording *states* a deductible, as opposed to merely
/// mentioning one.
///
/// # Mentioning is not stating, and the difference is load-bearing
///
/// "A co-insurance of 10% applies to the amount **above the Deductible**" contains
/// the word "deductible" and states no deductible at all — it states a
/// *co-insurance*. An earlier version of this rule fired on the bare keyword,
/// found no amount in the clause, and dutifully reported an *ambiguous deductible*
/// — which made [`crate::cost_share`] refuse to compute for a perfectly clear
/// policy.
///
/// That failure was in the safe direction (a refusal, not a wrong number), and it
/// was still a bug: a tool that cries wolf on clean documents will be ignored on
/// dirty ones. So the rule now requires the clause to be *about* the deductible,
/// and a clause that only refers to one is left alone.
const STATES_DEDUCTIBLE: [&str; 5] = [
    "deductible is",
    "deductible of",
    "deductible shall be",
    "deductible payable is",
    "deductible:",
];

/// `4.1 The Deductible is S$3,500 for each policy year.`
fn deductible_rule(clause: &Clause, lower: &str, cfg: &ExtractionConfig) -> Option<PolicyTerm> {
    if !STATES_DEDUCTIBLE.iter().any(|p| lower.contains(p)) {
        return None;
    }

    let amounts = scan_money(clause.text(), cfg);

    let value = match amounts.as_slice() {
        [] => ambiguity(
            TermKind::Deductible,
            AmbiguityKind::Unparseable,
            "the clause concerns the deductible but states no amount this scanner could read \
             (it may be in a benefit table, or spelled out in words)",
            clause,
        ),
        [one] => match deductible_basis(lower) {
            Some(basis) => TermValue::Deductible(Deductible {
                amount: one.clone(),
                basis,
            }),
            // Per year or per claim is not a detail. It is the difference between
            // paying this once and paying it at every admission.
            None => ambiguity(
                TermKind::Deductible,
                AmbiguityKind::Underspecified,
                "the amount is stated but not what it is charged against (per policy year? per \
                 claim? per confinement?) — and that changes what the insured pays by a lot",
                clause,
            ),
        },
        many => ambiguity(
            TermKind::Deductible,
            AmbiguityKind::Unparseable,
            &format!(
                "the clause states {} amounts and this scanner cannot tell which is the \
                 deductible for which circumstance",
                many.len()
            ),
            clause,
        ),
    };

    Some(term(clause, value, scope_of(lower)))
}

fn deductible_basis(lower: &str) -> Option<DeductibleBasis> {
    if lower.contains("policy year") || lower.contains("each year") {
        Some(DeductibleBasis::PerPolicyYear)
    } else if lower.contains("per claim") || lower.contains("each claim") {
        Some(DeductibleBasis::PerClaim)
    } else if lower.contains("confinement") {
        Some(DeductibleBasis::PerConfinement)
    } else {
        None
    }
}

/// `4.2 A co-insurance of 10% applies to the amount above the Deductible.`
fn co_insurance_rule(clause: &Clause, lower: &str, cfg: &ExtractionConfig) -> Option<PolicyTerm> {
    if !(lower.contains("co-insurance")
        || lower.contains("coinsurance")
        || lower.contains("co insurance"))
    {
        return None;
    }

    let rates = scan_percent(clause.text());

    let value = match rates.as_slice() {
        [rate] => TermValue::CoInsurance(CoInsurance {
            rate: *rate,
            cap: capped_amount(clause, lower, cfg),
        }),
        [] => ambiguity(
            TermKind::CoInsurance,
            AmbiguityKind::Unparseable,
            "the clause concerns co-insurance but states no percentage this scanner could read",
            clause,
        ),
        many => ambiguity(
            TermKind::CoInsurance,
            AmbiguityKind::Unparseable,
            &format!(
                "the clause states {} percentages and this scanner cannot tell which is the \
                 co-insurance rate for which circumstance",
                many.len()
            ),
            clause,
        ),
    };

    Some(term(clause, value, scope_of(lower)))
}

/// `4.3 We will pay up to S$150,000 for each policy year.`
fn claim_limit_rule(clause: &Clause, lower: &str, cfg: &ExtractionConfig) -> Option<PolicyTerm> {
    // Do not let a deductible or co-insurance clause's amount be mistaken for a
    // limit: those clauses are about what the *insured* pays.
    if lower.contains("deductible") || lower.contains("insurance of") {
        return None;
    }

    let mentions_limit = lower.contains("limit")
        || lower.contains("we will pay up to")
        || lower.contains("maximum")
        || lower.contains("as charged");
    if !mentions_limit {
        return None;
    }

    let scope = scope_of(lower);

    // "As charged" is its own thing — see `ClaimLimit::AsCharged`. It is not a
    // very large number.
    if lower.contains("as charged") {
        return Some(term(
            clause,
            TermValue::ClaimLimit(ClaimLimit::AsCharged),
            scope,
        ));
    }

    let amounts = scan_money(clause.text(), cfg);
    let value = match amounts.as_slice() {
        [one] => match limit_basis(lower, one) {
            Some(limit) => TermValue::ClaimLimit(limit),
            None => ambiguity(
                TermKind::ClaimLimit,
                AmbiguityKind::Underspecified,
                "an amount is stated but not the period it caps (per claim? per policy year? \
                 lifetime? per day?)",
                clause,
            ),
        },
        [] => return None,
        many => ambiguity(
            TermKind::ClaimLimit,
            AmbiguityKind::Unparseable,
            &format!(
                "the clause states {} amounts and this scanner cannot tell which is the limit",
                many.len()
            ),
            clause,
        ),
    };

    Some(term(clause, value, scope))
}

fn limit_basis(lower: &str, amount: &MonetaryAmount) -> Option<ClaimLimit> {
    let amount = amount.clone();
    if lower.contains("lifetime") {
        Some(ClaimLimit::Lifetime(amount))
    } else if lower.contains("per day") || lower.contains("each day") || lower.contains("daily") {
        Some(ClaimLimit::PerDay(amount))
    } else if lower.contains("policy year") || lower.contains("each year") {
        Some(ClaimLimit::PerPolicyYear(amount))
    } else if lower.contains("per claim") || lower.contains("each claim") {
        Some(ClaimLimit::PerClaim(amount))
    } else {
        None
    }
}

/// `2.1 This Plan entitles the Insured to a Class A ward.`
fn ward_entitlement_rule(clause: &Clause, lower: &str) -> Option<PolicyTerm> {
    if !(lower.contains("entitle") || lower.contains("entitlement")) {
        return None;
    }
    let ward = ward_of(lower)?;
    Some(term(
        clause,
        TermValue::WardEntitlement(ward),
        Scope::any(),
    ))
}

/// `6.1 A waiting period of 12 months applies to ...`
fn waiting_period_rule(clause: &Clause, lower: &str) -> Option<PolicyTerm> {
    if !lower.contains("waiting period") {
        return None;
    }

    let value = match scan_duration(clause.text()) {
        Some(duration) => TermValue::WaitingPeriod(WaitingPeriod { duration }),
        None => ambiguity(
            TermKind::WaitingPeriod,
            AmbiguityKind::Unparseable,
            "the clause imposes a waiting period but states no duration this scanner could read",
            clause,
        ),
    };

    Some(term(clause, value, scope_of(lower)))
}

/// `4.5 We will not pay for any Pre-existing Condition.`
fn pre_existing_rule(clause: &Clause, lower: &str) -> Option<PolicyTerm> {
    if !(lower.contains("pre-existing") || lower.contains("preexisting")) {
        return None;
    }

    let excluded = lower.contains("will not pay")
        || lower.contains("not covered")
        || lower.contains("do not cover")
        || lower.contains("excluded")
        || lower.contains("exclude");

    let assessment = lower.contains("underwriting")
        || lower.contains("moratorium")
        || lower.contains("assessment")
        || lower.contains("declare")
        || lower.contains("declaration");

    let treatment = match (excluded, scan_duration(clause.text()), assessment) {
        // "excluded for the first 24 months" — an exclusion with an end date is a
        // very different thing from an exclusion without one.
        (true, Some(duration), _) => {
            PreExistingConditionTreatment::StatedExcludedForPeriod(duration)
        }
        (true, None, _) => PreExistingConditionTreatment::StatedExcluded,
        (false, _, true) => PreExistingConditionTreatment::StatedSubjectToAssessment {
            process: clause.text().to_string(),
        },
        (false, _, false) => PreExistingConditionTreatment::NotDetermined,
    };

    Some(term(
        clause,
        TermValue::PreExistingCondition(treatment),
        scope_of(lower),
    ))
}

/// A general statement of what is and is not paid for.
///
/// The `benefit` is taken to be the clause's innermost heading where there is one,
/// and the clause text otherwise. Recovering the actual benefit noun-phrase is
/// beyond a keyword scanner, and guessing at it would put a wrong label on a right
/// clause — which is worse than an unwieldy one.
fn coverage_rule(clause: &Clause, lower: &str) -> Option<PolicyTerm> {
    let excluding = lower.contains("will not pay")
        || lower.contains("do not cover")
        || lower.contains("is excluded")
        || lower.contains("are excluded");
    let paying = lower.contains("we will pay for") || lower.contains("we will cover");

    if !(excluding || paying) {
        return None;
    }

    // A clause that both promises and refuses to pay is not a clause we get to
    // resolve. It is a drafting error, and it is the reader's finding.
    if excluding && paying {
        return Some(term(
            clause,
            ambiguity(
                TermKind::Coverage,
                AmbiguityKind::Contradictory,
                "the clause states both that the benefit will be paid and that it will not",
                clause,
            ),
            scope_of(lower),
        ));
    }

    let statement = if excluding {
        CoverageStatement::StatedExcluded
    } else {
        conditional_or_payable(clause, lower)
    };

    let benefit = clause
        .path()
        .innermost()
        .map(str::to_string)
        .unwrap_or_else(|| clause.text().to_string());

    Some(term(
        clause,
        TermValue::Coverage { benefit, statement },
        scope_of(lower),
    ))
}

/// A promise to pay hedged with a condition is not a promise to pay.
fn conditional_or_payable(clause: &Clause, lower: &str) -> CoverageStatement {
    const MARKERS: [&str; 4] = ["provided that", "only if", "subject to", "so long as"];

    let conditions: Vec<String> = MARKERS
        .iter()
        .filter_map(|marker| {
            lower
                .find(marker)
                .map(|i| clause.text()[i..].trim().trim_end_matches('.').to_string())
        })
        .collect();

    if conditions.is_empty() {
        CoverageStatement::StatedPayable
    } else {
        CoverageStatement::StatedConditional { conditions }
    }
}

/// The clause that decides how the private plan stacks on the basic scheme.
///
/// See [`IntegrationMode`] — this is the term the whole cost-share calculation
/// turns on, and the one this crate most refuses to guess at.
fn integration_mode_rule(clause: &Clause, lower: &str) -> Option<PolicyTerm> {
    let mentions_basic = lower.contains("medishield")
        || lower.contains("basic scheme")
        || lower.contains("universal scheme")
        || lower.contains("the scheme");
    if !mentions_basic {
        return None;
    }

    let inclusive = lower.contains("inclusive of") || lower.contains("includes the benefit");
    let excess = lower.contains("in excess of")
        || lower.contains("not paid by")
        || lower.contains("after the scheme has paid");

    let value = match (inclusive, excess) {
        (true, true) => ambiguity(
            TermKind::IntegrationMode,
            AmbiguityKind::Contradictory,
            "the clause states both that the benefit is inclusive of the basic scheme's payout \
             and that it applies only to what the scheme does not pay — these give materially \
             different out-of-pocket figures and the clause cannot mean both",
            clause,
        ),
        (true, false) => TermValue::IntegrationMode(IntegrationMode::InclusiveOfBasic),
        (false, true) => TermValue::IntegrationMode(IntegrationMode::ExcessOfBasic),
        (false, false) => return None,
    };

    Some(term(clause, value, Scope::any()))
}

/// A rider clause: what it says it absorbs of the base plan's cost-sharing.
///
/// # Why this needs both halves in one clause
///
/// A [`RiderCoverage`] is a statement about the deductible **and** a statement
/// about the co-insurance, and this scaffold builds it from a single clause that
/// addresses both. A wording that splits them across two clauses produces an
/// ambiguity rather than a half-filled term.
///
/// That is a real limitation. The honest fix — aggregating statements from several
/// clauses into one term, with **all** the contributing clauses attached as
/// provenance — needs a many-clauses-to-one-term extraction model this scaffold
/// does not have. It is generic (a motor policy's excess waiver has the same
/// shape) and belongs in `kopitiam-insurance`.
fn rider_coverage_rule(clause: &Clause, lower: &str) -> Option<PolicyTerm> {
    let mentions_deductible = lower.contains("deductible");
    let mentions_co_insurance = lower.contains("co-insurance") || lower.contains("coinsurance");

    if !(mentions_deductible || mentions_co_insurance) {
        return None;
    }
    // A co-payment clause mentions co-insurance only in passing; do not read it as
    // a coverage statement.
    if lower.contains("co-payment") || lower.contains("copayment") {
        return None;
    }

    if !(mentions_deductible && mentions_co_insurance) {
        return Some(term(
            clause,
            ambiguity(
                TermKind::RiderCoverage,
                AmbiguityKind::Unparseable,
                "the clause addresses only one of the base plan's deductible and co-insurance, \
                 and this scaffold cannot assemble a rider's cover from clauses in isolation",
                clause,
            ),
            Scope::any(),
        ));
    }

    let statement = rider_statement(clause, lower);

    Some(term(
        clause,
        TermValue::RiderCoverage(RiderCoverage {
            deductible: statement.clone(),
            co_insurance: statement,
        }),
        scope_of(lower),
    ))
}

fn rider_statement(clause: &Clause, lower: &str) -> CoverageStatement {
    let refuses = lower.contains("will not")
        || lower.contains("does not cover")
        || lower.contains("do not reimburse");
    let pays = lower.contains("reimburse") || lower.contains("we will pay") || lower.contains("cover");

    match (pays, refuses) {
        (_, true) => CoverageStatement::StatedExcluded,
        (true, false) => conditional_or_payable(clause, lower),
        (false, false) => CoverageStatement::Silent,
    }
}

/// `2.2 A co-payment of 5% of the claimable amount applies, capped at S$3,000 each
/// policy year.`
fn co_payment_rule(clause: &Clause, lower: &str, cfg: &ExtractionConfig) -> Option<PolicyTerm> {
    if !(lower.contains("co-payment") || lower.contains("copayment") || lower.contains("co-pay")) {
        return None;
    }

    // A rider may genuinely charge none — but it has to *say* so. See the "no
    // silent default" discussion on `CostShareRefusal`: a missing co-payment
    // clause becoming 0% would understate what the patient pays, which is the
    // harmful direction.
    if lower.contains("no co-payment")
        || lower.contains("nil co-payment")
        || lower.contains("no copayment")
    {
        return Some(term(
            clause,
            TermValue::CoPayment(CoPayment {
                rate: Percentage::from_basis_points(0),
                cap: None,
                base: CoPaymentBase::AmountRiderWouldAbsorb,
            }),
            scope_of(lower),
        ));
    }

    let rates = scan_percent(clause.text());
    let value = match rates.as_slice() {
        [rate] => match co_payment_base(lower) {
            Some(base) => TermValue::CoPayment(CoPayment {
                rate: *rate,
                cap: capped_amount(clause, lower, cfg),
                base,
            }),
            // 5% of the whole bill and 5% of what the rider would have absorbed
            // are very different numbers. See `CoPaymentBase`.
            None => ambiguity(
                TermKind::CoPayment,
                AmbiguityKind::Underspecified,
                "the rate is stated but not what it is charged on (the claimable amount? the \
                 amount the rider would otherwise absorb?) — these differ by a lot on a big bill",
                clause,
            ),
        },
        _ => ambiguity(
            TermKind::CoPayment,
            AmbiguityKind::Unparseable,
            "the clause concerns a co-payment but this scanner could not read a single rate from it",
            clause,
        ),
    };

    Some(term(clause, value, scope_of(lower)))
}

fn co_payment_base(lower: &str) -> Option<CoPaymentBase> {
    if lower.contains("claimable amount") || lower.contains("of the bill") {
        Some(CoPaymentBase::ClaimableAmount)
    } else if lower.contains("would otherwise")
        || lower.contains("amount we would")
        || lower.contains("amount reimbursed")
        || lower.contains("amount payable under this rider")
    {
        Some(CoPaymentBase::AmountRiderWouldAbsorb)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// Mints a term with the clause as its citation.
///
/// [`Clause::provenance`] cites the whole clause, verbatim — which is the honest
/// citation for a rule that read the whole clause. (For a rule that keyed off one
/// sentence, [`Clause::cite`] would be better, and it checks the quotation against
/// the clause. These scanners look at the whole text, so the whole text is what
/// they cite.)
fn term(clause: &Clause, value: TermValue, scope: Scope) -> PolicyTerm {
    PolicyTerm::new(ExtractedTerm::new(value, clause.provenance()), scope)
}

/// Builds an ambiguity — always with the clause attached.
fn ambiguity(
    intended: TermKind,
    kind: AmbiguityKind,
    note: &str,
    clause: &Clause,
) -> TermValue {
    TermValue::Ambiguous(Ambiguity {
        intended,
        kind,
        note: note.to_string(),
        sources: vec![clause.provenance()],
    })
}

/// The cap a clause puts on an insured's share, if it states one.
///
/// `None` means **the clause states no cap**, not that there is none anywhere in
/// the document. The calculator treats an absent cap as "uncapped in this clause"
/// and says so, rather than hunting for one elsewhere and stitching two clauses
/// together on the reader's behalf.
fn capped_amount(clause: &Clause, lower: &str, cfg: &ExtractionConfig) -> Option<MonetaryAmount> {
    let signals_cap = lower.contains("capped")
        || lower.contains("subject to a maximum")
        || lower.contains("up to a maximum")
        || lower.contains("not exceed");
    if !signals_cap {
        return None;
    }

    match scan_money(clause.text(), cfg).as_slice() {
        [only] => Some(only.clone()),
        // Two amounts and one cap: we cannot tell which is which, so we report no
        // cap *in this clause* rather than picking. The clause is still attached to
        // the term, so a reader sees it.
        _ => None,
    }
}

/// Which ward class and panel status a clause is written for, if it says.
///
/// An unqualified clause gets [`Scope::any`]. Note the direction of error this
/// avoids: a *scoped* term wrongly read as unqualified would be applied to
/// treatments the document never wrote it for.
fn scope_of(lower: &str) -> Scope {
    Scope {
        ward: ward_of(lower),
        provider: provider_of(lower),
    }
}

fn ward_of(lower: &str) -> Option<WardClass> {
    // Order matters: "class b1" must be tested before "class b".
    if lower.contains("private hospital") || lower.contains("private ward") {
        Some(WardClass::Private)
    } else if lower.contains("day surgery") {
        Some(WardClass::DaySurgery)
    } else if lower.contains("class b1") {
        Some(WardClass::PublicB1)
    } else if lower.contains("class b2") {
        Some(WardClass::PublicB2)
    } else if lower.contains("class a") {
        Some(WardClass::PublicA)
    } else if lower.contains("class c") {
        Some(WardClass::PublicC)
    } else {
        None
    }
}

fn provider_of(lower: &str) -> Option<ProviderNetwork> {
    // "non-panel" contains "panel", so it must be tested first.
    if lower.contains("non-panel") || lower.contains("not on our panel") {
        Some(ProviderNetwork::NonPanel)
    } else if lower.contains("emergency") {
        Some(ProviderNetwork::Emergency)
    } else if lower.contains("panel") {
        Some(ProviderNetwork::Panel)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Scanners.
//
// Hand-written rather than regex-based: the crate takes no dependency it does not
// need, and the shapes below are simple enough that a scanner is clearer than a
// pattern — and, more to the point, easier to reason about when the question is
// "could this possibly have matched the wrong number?", which is the only question
// that matters here.
//
// These scan *prose*. `kopitiam-insurance` has `parse_value`, which parses a whole
// schedule cell; it does not scan a sentence for the amounts embedded in it. If a
// second domain crate needs that, these three functions are the thing to lift down.
// ---------------------------------------------------------------------------

/// Finds every monetary amount in a clause, with the currency **as printed**.
///
/// # Why an amount must carry a currency marker
///
/// Only text preceded by `$`, `S$`, `US$` or a three-letter ISO code is treated as
/// money. A bare `3,500` is not — because a wording is full of bare numbers (ward
/// classes, clause cross-references, days, ages), and a scanner that grabbed them
/// would produce a deductible of "12" from "12 months".
///
/// A bare `$` yields [`Currency::Ambiguous`], not a guess. See
/// [`ExtractionConfig::declared_currency`] for the one way a caller may resolve it,
/// and [`crate::money::Amount`] for what happens if they do not.
pub fn scan_money(text: &str, cfg: &ExtractionConfig) -> Vec<MonetaryAmount> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        // An ISO code followed by an amount: "SGD 150,000".
        if bytes[i].is_ascii_alphabetic()
            && !(i > 0 && bytes[i - 1].is_ascii_alphabetic())
            && i + 3 <= bytes.len()
            && bytes[i..i + 3].iter().all(u8::is_ascii_alphabetic)
            && !(i + 3 < bytes.len() && bytes[i + 3].is_ascii_alphabetic())
        {
            let code = text[i..i + 3].to_uppercase();
            if is_iso_code(&code)
                && let Some((cents, next)) = parse_amount_after(text, i + 3)
            {
                out.push(MonetaryAmount::new(
                    Money::from_cents(cents),
                    Currency::Iso(code),
                ));
                i = next;
                continue;
            }
        }

        if bytes[i] == b'$' {
            // Look back for a currency qualifier: "S$", "US$".
            let preceding = text[..i].to_uppercase();
            let currency = if preceding.ends_with("US") {
                Currency::Iso("USD".into())
            } else if preceding.ends_with('S') {
                Currency::Iso("SGD".into())
            } else {
                // A bare `$`. If the document declared its currency, honour that;
                // otherwise say plainly that we do not know.
                match &cfg.declared_currency {
                    Some(code) => Currency::Iso(code.to_uppercase()),
                    None => Currency::Ambiguous("$".into()),
                }
            };

            if let Some((cents, next)) = parse_amount_after(text, i + 1) {
                out.push(MonetaryAmount::new(Money::from_cents(cents), currency));
                i = next;
                continue;
            }
        }

        i += 1;
    }

    out
}

/// ISO 4217 codes an insurance wording in this region plausibly prints. Used only
/// to recognise a code that is *already written out*, never to guess one.
fn is_iso_code(code: &str) -> bool {
    const CODES: [&str; 8] = ["SGD", "USD", "MYR", "EUR", "GBP", "AUD", "HKD", "CNY"];
    CODES.contains(&code)
}

/// Parses `3,500` or `3,500.25` at or after `from`, skipping spaces. Returns the
/// amount in cents and the offset just past it.
fn parse_amount_after(text: &str, from: usize) -> Option<(i64, usize)> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }

    let mut major = String::new();
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b',') {
        if bytes[i].is_ascii_digit() {
            major.push(bytes[i] as char);
        }
        i += 1;
    }
    if major.is_empty() {
        return None;
    }

    let mut cents = major.parse::<i64>().ok()?.checked_mul(100)?;

    // Optional ".25" — but not a sentence-ending period, and not three decimals.
    if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
        let mut frac = String::new();
        let mut j = i + 1;
        while j < bytes.len() && bytes[j].is_ascii_digit() && frac.len() < 2 {
            frac.push(bytes[j] as char);
            j += 1;
        }
        // Three or more decimal digits is not an amount of money. Leave it unparsed
        // rather than silently truncating someone's figure.
        if !(j < bytes.len() && bytes[j].is_ascii_digit()) {
            while frac.len() < 2 {
                frac.push('0');
            }
            cents = cents.checked_add(frac.parse::<i64>().ok()?)?;
            i = j;
        }
    }

    Some((cents, i))
}

/// Finds every percentage in a clause: `10%`, `12.5%`.
///
/// A percentage above 100% or needing more than two decimal places is **not**
/// returned — it is not a co-insurance rate, and the caller ends up with an
/// ambiguity rather than an absurd figure.
pub fn scan_percent(text: &str) -> Vec<Percentage> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();

    for (i, b) in bytes.iter().enumerate() {
        if *b != b'%' {
            continue;
        }
        // Walk back over the number.
        let mut start = i;
        while start > 0
            && (bytes[start - 1].is_ascii_digit()
                || (bytes[start - 1] == b'.' && start >= 2 && bytes[start - 2].is_ascii_digit()))
        {
            start -= 1;
        }
        if start == i {
            continue;
        }

        let num = &text[start..i];
        let (whole, frac) = num.split_once('.').unwrap_or((num, ""));
        let Ok(whole) = whole.parse::<i64>() else {
            continue;
        };
        // Two decimal places of a percent is one basis point. More precision than
        // that does not appear in a wording, and rounding it would be inventing a
        // rate.
        let frac_bp: i64 = match frac.len() {
            0 => 0,
            1 => frac.parse::<i64>().map(|v| v * 10).unwrap_or(0),
            2 => frac.parse::<i64>().unwrap_or(0),
            _ => continue,
        };

        let bp = whole * 100 + frac_bp;
        if (0..=10_000).contains(&bp) {
            out.push(Percentage::from_basis_points(bp));
        }
    }

    out
}

/// Finds a duration: `12 months`, `30 days`, `2 years`.
///
/// Digits only. A wording that spells out "twelve months" is not matched, and the
/// caller gets an ambiguity rather than a wrong number. That is the right failure:
/// number words are a solved problem, but solving them here would mean this
/// scaffold quietly parsing more than it can be trusted to.
pub fn scan_duration(text: &str) -> Option<PolicyDuration> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    for pair in words.windows(2) {
        let Ok(n) = pair[0]
            .trim_matches(|c: char| !c.is_ascii_digit())
            .parse::<u32>()
        else {
            continue;
        };
        let unit = pair[1].trim_matches(|c: char| !c.is_alphanumeric());
        return Some(match unit {
            "day" | "days" => PolicyDuration::Days(n),
            "month" | "months" => PolicyDuration::Months(n),
            "year" | "years" => PolicyDuration::Years(n),
            _ => continue,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ExtractionConfig {
        ExtractionConfig::new(LayerKind::IntegratedTopUp)
    }

    #[test]
    fn money_needs_a_currency_marker() {
        assert!(
            scan_money("The Deductible is 3,500 for each policy year.", &cfg()).is_empty(),
            "a bare number must not be read as an amount"
        );
    }

    #[test]
    fn money_is_scanned_with_the_currency_the_document_printed() {
        let found = scan_money("The Deductible is S$3,500.00 each policy year.", &cfg());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].amount().cents(), 350_000);
        assert_eq!(found[0].currency().iso(), Some("SGD"));
    }

    /// The load-bearing honesty case. `$` is not SGD; it is `$`.
    #[test]
    fn a_bare_dollar_sign_stays_ambiguous_unless_the_document_declared_a_currency() {
        let found = scan_money("We will pay up to $150,000 each policy year.", &cfg());
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].currency(),
            &Currency::Ambiguous("$".into()),
            "a bare $ must not be assumed to be any particular currency"
        );

        // ...unless the caller asserts the document declared one.
        let declared = cfg().with_declared_currency("sgd");
        let found = scan_money("We will pay up to $150,000 each policy year.", &declared);
        assert_eq!(found[0].currency().iso(), Some("SGD"));
    }

    #[test]
    fn an_explicit_marker_beats_the_declared_currency() {
        let declared = cfg().with_declared_currency("SGD");
        let found = scan_money("A fee of US$500 applies.", &declared);
        assert_eq!(
            found[0].currency().iso(),
            Some("USD"),
            "the document's own marker always wins"
        );
    }

    #[test]
    fn a_clause_stating_two_amounts_yields_two_matches_not_a_guess() {
        let found = scan_money(
            "The Deductible is S$3,500 for Class B1 and S$5,000 for Class A.",
            &cfg(),
        );
        assert_eq!(
            found.len(),
            2,
            "the extractor must see both and refuse to pick"
        );
    }

    #[test]
    fn percentages_scan_exactly() {
        assert_eq!(
            scan_percent("a co-insurance of 10% applies"),
            vec![Percentage::from_basis_points(1_000)]
        );
        assert_eq!(
            scan_percent("12.5% of the balance"),
            vec![Percentage::from_basis_points(1_250)]
        );
    }

    #[test]
    fn an_impossible_percentage_is_not_returned() {
        assert!(
            scan_percent("a co-insurance of 150% applies").is_empty(),
            "150% is not a co-insurance rate; the caller must get an ambiguity"
        );
    }

    #[test]
    fn durations_keep_the_unit_the_document_wrote() {
        assert_eq!(
            scan_duration("a waiting period of 12 months applies"),
            Some(PolicyDuration::Months(12))
        );
        assert_eq!(
            scan_duration("within 30 days of admission"),
            Some(PolicyDuration::Days(30))
        );
        assert_eq!(
            scan_duration("a waiting period of twelve months applies"),
            None,
            "a spelled-out number must fail rather than be guessed"
        );
    }

    #[test]
    fn ward_and_panel_scopes_are_read_from_the_clause() {
        let scope = scope_of("for treatment in a private hospital by a non-panel specialist");
        assert_eq!(scope.ward, Some(WardClass::Private));
        assert_eq!(scope.provider, Some(ProviderNetwork::NonPanel));

        // "non-panel" contains "panel"; the negation must win.
        assert_eq!(
            provider_of("a non-panel provider"),
            Some(ProviderNetwork::NonPanel)
        );
    }
}
