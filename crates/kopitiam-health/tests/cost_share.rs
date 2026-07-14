//! The arithmetic, and the boundaries people get wrong.
//!
//! Every figure here is hand-computed in the test that uses it, and every fixture
//! is synthetic. See `tests/common/mod.rs`.

mod common;

use kopitiam_health::{
    Amount, Bill, BorneBy, Caveat, CostShareRefusal, LayerKind, PolicyStack, ProviderNetwork,
    TermKind, TreatmentContext, WardClass, compute_cost_share,
};

fn sgd(major: i64) -> Amount {
    Amount::major(major, "SGD").unwrap()
}

fn bill(total: i64, claimable: i64) -> Bill {
    Bill::new(sgd(total), sgd(claimable)).unwrap()
}

fn private_panel() -> TreatmentContext {
    TreatmentContext::new(WardClass::Private, ProviderNetwork::Panel)
}

fn basic() -> kopitiam_health::PolicyLayer {
    common::layer(
        "basic",
        "National Basic Health Scheme",
        LayerKind::UniversalBasic,
        common::BASIC_SCHEME,
    )
}

fn inclusive_plan() -> kopitiam_health::PolicyLayer {
    common::layer(
        "shield",
        "Teh Tarik Shield",
        LayerKind::IntegratedTopUp,
        common::SHIELD_INCLUSIVE,
    )
}

fn excess_plan() -> kopitiam_health::PolicyLayer {
    common::layer(
        "shield-excess",
        "Kopi Peng Shield",
        LayerKind::IntegratedTopUp,
        common::SHIELD_EXCESS,
    )
}

fn rider() -> kopitiam_health::PolicyLayer {
    common::layer("rider", "Kopi-O Rider", LayerKind::Rider, common::RIDER)
}

// ---------------------------------------------------------------------------
// The order of operations.
// ---------------------------------------------------------------------------

/// The central arithmetic, hand-computed.
///
/// Claimable S$10,000; deductible S$3,500; co-insurance 10%; limit S$150,000.
///
/// ```text
///   10,000 - 3,500 (deductible)          = 6,500
///    6,500 - 650   (10% co-insurance)    = 5,850  <- insurer
///   insured bears 3,500 + 650            = 4,150
/// ```
///
/// Apply the co-insurance to the whole S$10,000 *first* and you get S$4,500 /
/// S$5,500 instead — a S$350 error from an ordering that looks perfectly
/// reasonable if you have not thought about it.
#[test]
fn deductible_comes_off_first_then_co_insurance_on_the_remainder() {
    let stack = PolicyStack::new(vec![basic(), inclusive_plan()]).unwrap();
    let breakdown =
        compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap();

    assert_eq!(breakdown.insured_pays, sgd(4_150));
    assert_eq!(breakdown.insurers_pay, sgd(5_850));

    let plan = breakdown
        .layers
        .iter()
        .find(|l| l.kind == LayerKind::IntegratedTopUp)
        .unwrap();
    assert_eq!(plan.deductible_borne, sgd(3_500));
    assert_eq!(plan.co_insurance_borne, sgd(650));
    assert_eq!(plan.above_limit, sgd(0));

    // The wrong ordering would have produced these. Assert we did not.
    assert_ne!(breakdown.insured_pays, sgd(4_500));
    assert_ne!(breakdown.insurers_pay, sgd(5_500));
}

/// **The single most common misunderstanding of a shield plan.**
///
/// A S$800 bill against a S$3,500 deductible: the insurer pays *nothing*. Not "the
/// bill minus a small share". Nothing. This is why people who have cover are
/// blindsided by the bill for a whole day-surgery episode.
#[test]
fn a_bill_below_the_deductible_is_paid_entirely_by_the_insured() {
    let stack = PolicyStack::new(vec![basic(), inclusive_plan()]).unwrap();
    let breakdown = compute_cost_share(&stack, &bill(800, 800), &private_panel()).unwrap();

    assert_eq!(breakdown.insured_pays, sgd(800));
    assert_eq!(
        breakdown.insurers_pay,
        sgd(0),
        "below the deductible the insurer pays nothing at all"
    );

    // And no co-insurance step took anything: there was nothing above the
    // deductible to take it from.
    let plan = breakdown
        .layers
        .iter()
        .find(|l| l.kind == LayerKind::IntegratedTopUp)
        .unwrap();
    assert_eq!(plan.deductible_borne, sgd(800));
    assert_eq!(plan.co_insurance_borne, sgd(0));
}

// ---------------------------------------------------------------------------
// The stacking question.
// ---------------------------------------------------------------------------

/// **Why `IntegrationMode` exists.**
///
/// Two plans, word for word identical except for clause 2.1 — one says its benefit
/// is *inclusive of* the basic scheme's payout, the other that it applies only *in
/// excess of* it. Same bill, same deductible, same co-insurance rate, same limits.
///
/// * Inclusive: the plan's S$3,500 deductible bites on the whole S$10,000.
///   Insured bears 3,500 + 10%*6,500 = **S$4,150**.
/// * Excess: the basic scheme runs first (S$1,500 deductible, 10% co-insurance),
///   leaving the insured holding 1,500 + 10%*8,500 = S$2,350. The plan then sees
///   only that S$2,350, which is *below* its own S$3,500 deductible — so the plan
///   pays nothing and the insured bears **S$2,350**.
///
/// S$1,800 apart, from one clause. This is the thing this crate refuses to guess.
#[test]
fn the_integration_mode_changes_what_the_patient_pays() {
    let inclusive = PolicyStack::new(vec![basic(), inclusive_plan()]).unwrap();
    let excess = PolicyStack::new(vec![basic(), excess_plan()]).unwrap();

    let b = bill(12_000, 10_000);
    let inclusive = compute_cost_share(&inclusive, &b, &private_panel()).unwrap();
    let excess = compute_cost_share(&excess, &b, &private_panel()).unwrap();

    assert_eq!(inclusive.insured_pays, sgd(4_150));
    assert_eq!(excess.insured_pays, sgd(2_350));
    assert_ne!(
        inclusive.insured_pays, excess.insured_pays,
        "if these ever agree, the stacking model has stopped modelling anything"
    );

    // Under the inclusive reading, the basic scheme's share is folded into the
    // insurer's figure and this crate says so rather than pretending to split it.
    assert!(
        inclusive
            .caveats
            .iter()
            .any(|c| matches!(c, Caveat::BasicSchemeShareNotSeparated { .. }))
    );
}

/// A plan that pays only in excess of the basic scheme, with no basic scheme
/// document supplied, cannot be computed: what the scheme leaves unpaid is unknown.
/// Assuming it paid nothing would understate the deductible's bite.
#[test]
fn excess_of_basic_without_a_basic_scheme_document_refuses() {
    let stack = PolicyStack::new(vec![excess_plan()]).unwrap();
    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();

    assert!(matches!(
        err,
        CostShareRefusal::BasicSchemeRequiredButAbsent { .. }
    ));
    // The refusal quotes the clause that caused it.
    assert!(err.to_string().contains("in excess of"));
}

// ---------------------------------------------------------------------------
// Riders.
// ---------------------------------------------------------------------------

/// The rider absorbs the plan's deductible (S$3,500) and co-insurance (S$650) —
/// S$4,150 — then charges back a 5% co-payment on the claimable amount
/// (5% of S$10,000 = S$500, under its S$3,000 cap).
///
/// So the insured pays S$500, the rider pays S$3,650, and the plan pays S$5,850.
#[test]
fn a_rider_absorbs_the_plans_deductible_and_co_insurance_less_its_co_payment() {
    let with_rider = PolicyStack::new(vec![basic(), inclusive_plan(), rider()]).unwrap();
    let without = PolicyStack::new(vec![basic(), inclusive_plan()]).unwrap();

    let b = bill(12_000, 10_000);
    let with_rider = compute_cost_share(&with_rider, &b, &private_panel()).unwrap();
    let without = compute_cost_share(&without, &b, &private_panel()).unwrap();

    assert_eq!(without.insured_pays, sgd(4_150));
    assert_eq!(with_rider.insured_pays, sgd(500));
    assert_eq!(with_rider.insurers_pay, sgd(9_500));

    // Everything claimable is accounted for, to the cent.
    assert_eq!(
        with_rider
            .insured_pays
            .add(&with_rider.insurers_pay)
            .unwrap(),
        sgd(10_000)
    );

    let rider_layer = with_rider
        .layers
        .iter()
        .find(|l| l.kind == LayerKind::Rider)
        .unwrap();
    assert_eq!(rider_layer.insurer_pays, sgd(3_650));
    assert!(
        rider_layer
            .steps
            .iter()
            .any(|s| s.borne_by == BorneBy::Rider)
    );
}

/// A rider whose cover is conditional on a fact about the patient's treatment is a
/// rider this crate will not compute with. Whether the specialist was on the panel
/// is not a question a document reader answers.
#[test]
fn a_conditional_rider_is_refused_rather_than_assumed_to_pay() {
    let stack = PolicyStack::new(vec![
        basic(),
        inclusive_plan(),
        common::layer(
            "maybe",
            "Maybe Rider",
            LayerKind::Rider,
            common::RIDER_CONDITIONAL,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    assert!(matches!(
        err,
        CostShareRefusal::CannotEvaluateStatement { .. }
    ));
    // The condition itself is quoted back, not swallowed.
    assert!(err.to_string().contains("Panel Specialist"));
}

/// A rider cannot exist on its own: it has no base plan whose cost-sharing it could
/// absorb, so a calculation against it would look like it worked and return zero.
#[test]
fn a_rider_with_no_plan_beneath_it_is_rejected_at_assembly() {
    let err = PolicyStack::new(vec![basic(), rider()]).unwrap_err();
    assert_eq!(err, kopitiam_health::StackError::RiderWithoutTopUpPlan);
}

// ---------------------------------------------------------------------------
// Refusals: the crate declining rather than guessing.
// ---------------------------------------------------------------------------

/// **No silent default: a missing claim limit does not become "unlimited".**
///
/// Treating an unextracted ceiling as no ceiling would overstate the insurer's
/// share on exactly the large claims where the ceiling matters most.
#[test]
fn a_missing_claim_limit_is_refused_and_never_becomes_unlimited() {
    let stack = PolicyStack::new(vec![
        basic(),
        common::layer(
            "no-limit",
            "Limitless Shield",
            LayerKind::IntegratedTopUp,
            common::SHIELD_NO_LIMIT,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    match err {
        CostShareRefusal::MissingTerm { needed, .. } => {
            assert_eq!(needed, TermKind::ClaimLimit);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(
        err.to_string().contains("Not defaulting to a value"),
        "the refusal must say plainly that it is not defaulting"
    );
}

/// **No silent default: a rider with no co-payment clause does not get a 0%
/// co-payment.**
///
/// That default would understate what the patient pays — the harmful direction.
#[test]
fn a_rider_that_never_states_its_co_payment_is_refused_not_defaulted_to_zero() {
    let stack = PolicyStack::new(vec![
        basic(),
        inclusive_plan(),
        common::layer(
            "silent",
            "Silent Rider",
            LayerKind::Rider,
            common::RIDER_NO_COPAYMENT,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    match err {
        CostShareRefusal::MissingTerm { needed, .. } => assert_eq!(needed, TermKind::CoPayment),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// A deductible whose *basis* is not stated is unresolved, not assumed.
///
/// S$3,500 per policy year and S$3,500 per claim are wildly different for anyone
/// admitted more than once.
#[test]
fn a_deductible_with_no_stated_basis_is_surfaced_as_ambiguous_with_its_clause() {
    let stack = PolicyStack::new(vec![
        basic(),
        common::layer(
            "vague",
            "Vague Shield",
            LayerKind::IntegratedTopUp,
            common::SHIELD_UNDERSPECIFIED_DEDUCTIBLE,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    let CostShareRefusal::AmbiguousTerm { ambiguity, .. } = &err else {
        panic!("expected an ambiguity, got {err:?}");
    };

    assert_eq!(ambiguity.intended, TermKind::Deductible);
    assert!(!ambiguity.sources.is_empty(), "an ambiguity must carry its clause");

    // The clause itself comes back, verbatim, so a human can read it.
    let quoted = err.to_string();
    assert!(quoted.contains("The Deductible is S$3,500."));
    assert!(quoted.contains("clause 3.1"));
}

/// A wording that states the same deductible twice, differently, at the same level
/// of generality. Nothing in the document says which governs — so neither do we.
#[test]
fn two_conflicting_deductible_clauses_are_both_surfaced_not_silently_picked() {
    let stack = PolicyStack::new(vec![
        basic(),
        common::layer(
            "two-minds",
            "Two-Minds Shield",
            LayerKind::IntegratedTopUp,
            common::SHIELD_CONTRADICTORY_DEDUCTIBLE,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    let CostShareRefusal::ConflictingTerms { kind, clauses, .. } = &err else {
        panic!("expected conflicting terms, got {err:?}");
    };

    assert_eq!(*kind, TermKind::Deductible);
    assert_eq!(clauses.len(), 2);

    // Both figures are quoted. Neither was chosen.
    let quoted = err.to_string();
    assert!(quoted.contains("S$3,500"));
    assert!(quoted.contains("S$2,000"));
}

/// A clause that says the plan is both *inclusive of* and *in excess of* the basic
/// scheme cannot mean both. It is a drafting defect, and it is the reader's finding
/// — not ours to resolve by coin toss.
#[test]
fn a_self_contradictory_integration_clause_is_surfaced_as_contradictory() {
    let stack = PolicyStack::new(vec![
        basic(),
        common::layer(
            "schrodinger",
            "Schrodinger Shield",
            LayerKind::IntegratedTopUp,
            common::SHIELD_CONTRADICTORY_INTEGRATION,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    let CostShareRefusal::AmbiguousTerm { ambiguity, .. } = &err else {
        panic!("expected an ambiguity, got {err:?}");
    };

    assert_eq!(ambiguity.intended, TermKind::IntegrationMode);
    assert_eq!(ambiguity.kind, kopitiam_health::AmbiguityKind::Contradictory);
    assert!(
        err.to_string().contains("inclusive of"),
        "the contradictory clause must be quoted back verbatim"
    );
}

/// **A bare `$` is not SGD.**
///
/// The amount is extracted and citable — but the calculator will not compute with a
/// currency the document never named.
#[test]
fn amounts_in_an_unnamed_currency_are_refused_rather_than_assumed_to_be_local() {
    let stack = PolicyStack::new(vec![
        basic(),
        common::layer(
            "which-dollar",
            "Which Dollar Shield",
            LayerKind::IntegratedTopUp,
            common::SHIELD_BARE_DOLLAR,
        ),
    ])
    .unwrap();

    let err = compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap_err();
    assert!(
        matches!(err, CostShareRefusal::Money { .. }),
        "expected a currency refusal, got {err:?}"
    );
    assert!(err.to_string().contains("without identifying the currency"));
}

// ---------------------------------------------------------------------------
// What the answer carries with it.
// ---------------------------------------------------------------------------

/// Every figure comes back with the clauses it was computed from.
#[test]
fn every_computed_figure_carries_the_clauses_it_rested_on() {
    let stack = PolicyStack::new(vec![basic(), inclusive_plan(), rider()]).unwrap();
    let breakdown =
        compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap();

    // No step is ever unsourced.
    for layer in &breakdown.layers {
        for step in &layer.steps {
            assert!(
                !step.basis.is_empty(),
                "step '{}' has no clause behind it",
                step.description
            );
            for clause in &step.basis {
                assert!(!clause.verbatim().as_str().trim().is_empty());
            }
        }
    }

    // And the whole computation's basis quotes the actual wording.
    let basis: Vec<String> = breakdown
        .basis()
        .iter()
        .map(|p| p.verbatim().as_str().to_string())
        .collect();
    assert!(basis.iter().any(|t| t.contains("The Deductible is S$3,500")));
    assert!(basis.iter().any(|t| t.contains("co-insurance of 10%")));

    let explained = breakdown.explain();
    assert!(explained.contains("The Deductible is S$3,500"));
    assert!(explained.contains("clause 3.1"));
    assert!(
        explained.contains("not a determination that any claim is payable"),
        "the explanation must never be mistakable for a claims decision"
    );
}

/// The plan excludes pre-existing conditions and imposes a waiting period. This
/// crate does not evaluate either — and says so, loudly, attached to the number.
///
/// A cost-share figure computed while quietly ignoring a pre-existing-condition
/// exclusion is not merely incomplete. It is actively misleading, because it looks
/// like an answer.
#[test]
fn eligibility_clauses_are_surfaced_as_unevaluated_never_applied() {
    let stack = PolicyStack::new(vec![basic(), inclusive_plan()]).unwrap();
    let breakdown =
        compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap();

    let unevaluated: Vec<&kopitiam_health::PolicyTerm> = breakdown
        .caveats
        .iter()
        .filter_map(|c| match c {
            Caveat::EligibilityTermNotEvaluated { term } => Some(&**term),
            _ => None,
        })
        .collect();

    assert!(
        unevaluated
            .iter()
            .any(|t| t.kind() == TermKind::PreExistingCondition),
        "the pre-existing-condition exclusion must be surfaced"
    );
    assert!(
        unevaluated
            .iter()
            .any(|t| t.kind() == TermKind::WaitingPeriod),
        "the waiting period must be surfaced"
    );

    // Verbatim, with a citation — not a paraphrase.
    let pec = unevaluated
        .iter()
        .find(|t| t.kind() == TermKind::PreExistingCondition)
        .unwrap();
    assert_eq!(
        pec.verbatim(),
        "4.1 We will not pay for any Pre-existing Condition."
    );

    // But it did not change the arithmetic. The number is "if this claim is payable
    // at all, here is the split" — and the caveats are why it might not be.
    assert_eq!(breakdown.insured_pays, sgd(4_150));
}

/// A cumulative limit applied without any claim history must say so.
#[test]
fn cumulative_limits_admit_that_no_claim_history_was_considered() {
    let stack = PolicyStack::new(vec![basic(), inclusive_plan()]).unwrap();
    let breakdown =
        compute_cost_share(&stack, &bill(12_000, 10_000), &private_panel()).unwrap();

    assert!(
        breakdown
            .caveats
            .iter()
            .any(|c| matches!(c, Caveat::CumulativeLimitAppliedAsIfSingleClaim { .. })),
        "a per-policy-year figure applied with no claim history must admit it"
    );
    assert!(
        breakdown
            .caveats
            .iter()
            .any(|c| matches!(c, Caveat::ClaimableAmountSuppliedByCaller)),
        "the provenance of the claimable amount is always disclosed"
    );
}

/// A bill cannot claim more than it charged.
#[test]
fn a_bill_cannot_have_more_claimable_than_it_charged() {
    assert!(Bill::new(sgd(1_000), sgd(2_000)).is_err());
    assert!(Bill::new(sgd(1_000), Amount::major(500, "USD").unwrap()).is_err());
}
