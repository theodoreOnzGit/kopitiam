//! HDB policy as **dated, cited knowledge** — not as constants.
//!
//! # What this module is, and what it refuses to be
//!
//! The Housing & Development Board's rules decide whether a person can buy a
//! home. Around eighty per cent of Singaporeans live in an HDB flat, and the
//! rules that govern who may buy one — eligibility schemes, income ceilings,
//! grants, ethnic quotas, minimum occupation periods, the resale levy — are the
//! machinery by which that happens.
//!
//! Get one wrong and the software tells a real household they can buy a house
//! when they cannot, or cannot when they can. Everything below follows from
//! taking that seriously.
//!
//! **This module models what the published policy *says*, with citations. It
//! does not give housing advice, and it must not be made to pretend otherwise.**
//! "Here is the rule, and here is where it is written" is a tool. "Here is what
//! you should do" is a liability, and a false kindness besides — the rule this
//! crate quotes may be out of date, may not be the rule that governs the
//! household's particular case, and may be one of the many that this crate
//! openly does not model. Anyone acting on a figure here must check it against
//! HDB.
//!
//! # The three rules this module is built on
//!
//! ## 1. No policy number is ever a constant
//!
//! Income ceilings, grant amounts, MOP durations, quota percentages, *and even
//! the minimum ages* are all policy — all of them have changed, and all of them
//! will change again. Every figure therefore lives in a [`temporal::Timeline`]
//! or [`temporal::PolicyTable`], carries an
//! [`EffectiveRange`](temporal::EffectiveRange), and is looked up **by date**.
//! There is no "current income ceiling" in this API. There is only the ceiling
//! in force on the date you ask about.
//!
//! ## 2. Provenance is mandatory
//!
//! Every figure returned by this module arrives inside a [`temporal::Dated`],
//! which carries a non-optional [`Citation`]. The answer to "why?" is always
//! "§X of document Y, effective Z". It is never "because the code says so".
//!
//! ## 3. Not knowing is a first-class answer
//!
//! [`eligibility::Eligibility`] is not a `bool`. It is *eligible*,
//! *ineligible with reasons*, or **indeterminate because this crate does not
//! model that**. Likewise a policy table may hold an explicit
//! [`Provision::NotModelled`](temporal::Provision::NotModelled) span. Most
//! systems in this domain omit the third case, and omitting it is precisely how
//! they mislead: a household whose case turns on an unmodelled rule receives a
//! confident "no" instead of "we don't know".
//!
//! # The state of the data
//!
//! Every figure in [`rules`] was transcribed **offline, from recollection**.
//! Every [`Citation`] is therefore
//! [`Verification::Unverified`](citation::Verification::Unverified), and
//! [`HdbPolicy::unverified_provisions`] will confirm that all of them are. The
//! populated slice is small and deliberately conservative; where confidence ran
//! out, the tables say so rather than guessing. See [`rules::UNMODELLED`] for
//! the crate's own account of its blind spots, and read it before reading
//! anything else.
//!
//! # Example
//!
//! ```
//! use kopitiam_finance::hdb::policy::{
//!     HdbPolicy,
//!     domain::*,
//!     eligibility::{Eligibility, Query},
//!     quantity::{Age, MonthlyIncome, Sgd},
//!     temporal::Date,
//! };
//!
//! let policy = HdbPolicy::published();
//!
//! let household = Household {
//!     applicants: vec![
//!         Applicant {
//!             age: Age(30),
//!             residency: Residency::SingaporeCitizen,
//!             ethnicity: EthnicGroup::Chinese,
//!             subsidy_history: SubsidyHistory::FirstTimer,
//!             marital_status: MaritalStatus::Married,
//!         },
//!         Applicant {
//!             age: Age(29),
//!             residency: Residency::SingaporeCitizen,
//!             ethnicity: EthnicGroup::Chinese,
//!             subsidy_history: SubsidyHistory::FirstTimer,
//!             marital_status: MaritalStatus::Married,
//!         },
//!     ],
//!     nucleus: FamilyNucleus::SpousesOrParentsChildren,
//!     monthly_income: MonthlyIncome(Sgd::dollars(14_000)), // exactly at the ceiling
//! };
//!
//! let assessment = policy.assess(&Query {
//!     household,
//!     purchase: Purchase {
//!         mode: PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder),
//!         flat_type: FlatType::FourRoom,
//!         classification: Some(FlatClassification::Standard),
//!     },
//!     as_of: Date::new(2025, 1, 15).unwrap(),
//! });
//!
//! // "Not exceeding $14,000" admits exactly $14,000.
//! assert!(matches!(
//!     assessment.schemes[0].eligibility,
//!     Eligibility::Eligible { .. }
//! ));
//!
//! // And the figure that admitted them comes with the rule that says so.
//! let ceiling = assessment.income_ceiling.citation().unwrap();
//! assert!(ceiling.url.contains("hdb.gov.sg") || ceiling.url.contains("pmo.gov.sg"));
//! ```

pub mod citation;
pub mod domain;
pub mod eligibility;
pub mod ontology;
pub mod quantity;
pub mod rules;
pub mod temporal;

use citation::Citation;
use domain::{CeilingContext, EligibilityScheme, EthnicGroup, FlatType, Grant, HouseholdClass};
use quantity::{IncomeCeiling, MinimumAge, MinimumOccupationPeriod, Months, Sgd};
use rules::{EipLimits, GrantSchedule, MopKey};
use temporal::{PolicyTable, Provision, TableError, Timeline};

/// The HDB policy tables, and the queries that can be put to them.
///
/// Construct once with [`HdbPolicy::published`] and share it: the tables are
/// immutable, the lookups are pure, and nothing here touches a clock, a file, or
/// a network. A given `(household, purchase, date)` yields the same
/// [`Assessment`](eligibility::Assessment) on every run and every machine, which
/// is what "deterministic" means in CLAUDE.md's Engineering Principles and what
/// makes the tests below meaningful.
#[derive(Debug, Clone)]
pub struct HdbPolicy {
    income_ceilings: PolicyTable<(CeilingContext, HouseholdClass), IncomeCeiling>,
    minimum_ages: PolicyTable<EligibilityScheme, MinimumAge>,
    minimum_occupation_periods: PolicyTable<MopKey, MinimumOccupationPeriod>,
    ethnic_quotas: PolicyTable<EthnicGroup, EipLimits>,
    spr_quota: Timeline<EipLimits>,
    enhanced_housing_grant: PolicyTable<HouseholdClass, GrantSchedule>,
    other_grants: PolicyTable<(Grant, HouseholdClass), GrantSchedule>,
    resale_levy: PolicyTable<FlatType, Sgd>,
    spr_resale_waiting_period: Timeline<Months>,
    /// The citation for the prose eligibility conditions (citizenship, the shape
    /// of each scheme) that are not figures and so are not tabulated.
    eligibility_page: Citation,
}

impl HdbPolicy {
    /// The policy slice this crate holds.
    ///
    /// # Panics
    ///
    /// If the built-in tables are internally inconsistent — two provisions
    /// covering the same date, or a range that ends before it begins. That is a
    /// bug in [`rules`], not a runtime condition, and it is checked by
    /// [`HdbPolicy::try_published`] in the test suite. Failing loudly at
    /// construction is right: a contradictory policy table must never be
    /// *served*, because whichever of two overlapping rules it returned would be
    /// arbitrary.
    pub fn published() -> Self {
        Self::try_published().expect(
            "the built-in HDB policy tables are internally inconsistent (overlapping or inverted \
             effective ranges) — this is a bug in kopitiam-finance::hdb::policy::rules",
        )
    }

    /// [`HdbPolicy::published`], surfacing table-construction errors instead of
    /// panicking. Used by the tests that guard the tables' consistency.
    pub fn try_published() -> Result<Self, TableError> {
        Ok(Self {
            income_ceilings: rules::income_ceilings()?,
            minimum_ages: rules::minimum_ages()?,
            minimum_occupation_periods: rules::minimum_occupation_periods()?,
            ethnic_quotas: rules::ethnic_quotas()?,
            spr_quota: rules::spr_quota()?,
            enhanced_housing_grant: rules::enhanced_housing_grant()?,
            other_grants: rules::other_grants()?,
            resale_levy: rules::resale_levy()?,
            spr_resale_waiting_period: rules::spr_resale_waiting_period()?,
            eligibility_page: rules::eligibility_page(),
        })
    }

    /// The income-ceiling table, for callers who want the figures rather than an
    /// assessment.
    pub fn income_ceilings(&self) -> &PolicyTable<(CeilingContext, HouseholdClass), IncomeCeiling> {
        &self.income_ceilings
    }

    /// The minimum-age table.
    pub fn minimum_ages(&self) -> &PolicyTable<EligibilityScheme, MinimumAge> {
        &self.minimum_ages
    }

    /// The Minimum Occupation Period table.
    pub fn minimum_occupation_periods(&self) -> &PolicyTable<MopKey, MinimumOccupationPeriod> {
        &self.minimum_occupation_periods
    }

    /// The Ethnic Integration Policy limits.
    pub fn ethnic_quotas(&self) -> &PolicyTable<EthnicGroup, EipLimits> {
        &self.ethnic_quotas
    }

    /// The Singapore Permanent Resident quota.
    pub fn spr_quota(&self) -> &Timeline<EipLimits> {
        &self.spr_quota
    }

    /// The Enhanced CPF Housing Grant schedules.
    pub fn enhanced_housing_grant(&self) -> &PolicyTable<HouseholdClass, GrantSchedule> {
        &self.enhanced_housing_grant
    }

    /// The grants whose amounts are declared but not modelled.
    pub fn other_grants(&self) -> &PolicyTable<(Grant, HouseholdClass), GrantSchedule> {
        &self.other_grants
    }

    /// The resale levy (amounts not modelled).
    pub fn resale_levy(&self) -> &PolicyTable<FlatType, Sgd> {
        &self.resale_levy
    }

    /// Every citation in the tables that **nobody has checked against its
    /// source**.
    ///
    /// Today this returns *all* of them, because the tables were transcribed
    /// offline. That is not a defect of the API; it is the API doing its job.
    /// Any caller putting these figures in front of a person should be able to
    /// ask "has this been verified?" and get a true answer, and this is how.
    ///
    /// As sources are fetched and provisions are flipped to
    /// [`Verification::Verified`](citation::Verification::Verified), this list
    /// shrinks. When it is empty, the crate has earned a trust it does not
    /// currently deserve.
    pub fn unverified_provisions(&self) -> Vec<&Citation> {
        let mut out = Vec::new();
        collect_table(&self.income_ceilings, &mut out);
        collect_table(&self.minimum_ages, &mut out);
        collect_table(&self.minimum_occupation_periods, &mut out);
        collect_table(&self.ethnic_quotas, &mut out);
        collect_table(&self.enhanced_housing_grant, &mut out);
        collect_table(&self.other_grants, &mut out);
        collect_table(&self.resale_levy, &mut out);
        collect_timeline(&self.spr_quota, &mut out);
        collect_timeline(&self.spr_resale_waiting_period, &mut out);
        out.push(&self.eligibility_page);
        out.retain(|c| rules::is_unverified(c));
        out
    }
}

/// Gathers every citation in a timeline: the rules, and the announcements behind
/// the gaps.
fn collect_timeline<'a, V>(timeline: &'a Timeline<V>, out: &mut Vec<&'a Citation>) {
    for provision in timeline.provisions() {
        match provision {
            Provision::InForce(dated) => out.push(&dated.citation),
            Provision::NotModelled { announced_in, .. } => out.extend(announced_in.as_ref()),
        }
    }
}

fn collect_table<'a, K: Ord + std::fmt::Debug, V>(
    table: &'a PolicyTable<K, V>,
    out: &mut Vec<&'a Citation>,
) {
    for (_, timeline) in table.timelines() {
        collect_timeline(timeline, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use domain::*;
    use eligibility::{Eligibility, GrantOutcome, Query};
    use quantity::{Age, MonthlyIncome};
    use temporal::Date;

    fn on(year: i32, month: u32, day: u32) -> Date {
        Date::new(year, month, day).unwrap()
    }

    fn citizen(age: u32, marital_status: MaritalStatus) -> Applicant {
        Applicant {
            age: Age(age),
            residency: Residency::SingaporeCitizen,
            ethnicity: EthnicGroup::Chinese,
            subsidy_history: SubsidyHistory::FirstTimer,
            marital_status,
        }
    }

    fn couple(income_dollars: i64) -> Household {
        Household {
            applicants: vec![
                citizen(30, MaritalStatus::Married),
                citizen(29, MaritalStatus::Married),
            ],
            nucleus: FamilyNucleus::SpousesOrParentsChildren,
            monthly_income: MonthlyIncome(Sgd::dollars(income_dollars)),
        }
    }

    fn bto_4room() -> Purchase {
        Purchase {
            mode: PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder),
            flat_type: FlatType::FourRoom,
            classification: Some(FlatClassification::Standard),
        }
    }

    fn assess(household: Household, purchase: Purchase, as_of: Date) -> eligibility::Assessment {
        HdbPolicy::published().assess(&Query {
            household,
            purchase,
            as_of,
        })
    }

    #[test]
    fn the_built_in_tables_are_internally_consistent() {
        // Guards the `expect` in `published()`. If this fails, no overlapping or
        // inverted provision has been allowed to reach a caller.
        assert!(HdbPolicy::try_published().is_ok());
    }

    // -- Boundary conditions. The classic bug in this domain lives here. -----

    #[test]
    fn a_household_exactly_at_the_ceiling_is_eligible() {
        let a = assess(couple(14_000), bto_4room(), on(2025, 1, 15));
        assert!(
            matches!(a.schemes[0].eligibility, Eligibility::Eligible { .. }),
            "'not exceeding $14,000' admits a household earning exactly $14,000: {:?}",
            a.schemes[0].eligibility
        );
    }

    #[test]
    fn a_household_one_dollar_over_the_ceiling_is_ineligible_and_told_why() {
        let a = assess(couple(14_001), bto_4room(), on(2025, 1, 15));
        match &a.schemes[0].eligibility {
            Eligibility::Ineligible { reasons } => {
                assert_eq!(reasons.len(), 1);
                assert!(reasons[0].statement.contains("exceeds the ceiling"));
                // A refusal without a citation is an assertion of authority.
                assert!(!reasons[0].citation.url.is_empty());
                assert_eq!(reasons[0].citation.published, on(2019, 9, 11));
            }
            other => panic!("expected Ineligible with a reason, got {other:?}"),
        }
        assert!(!a.any_scheme_eligible());
    }

    #[test]
    fn one_cent_over_the_ceiling_is_over_the_ceiling() {
        let mut household = couple(0);
        household.monthly_income = MonthlyIncome(Sgd::cents(1_400_001));
        let a = assess(household, bto_4room(), on(2025, 1, 15));
        assert!(matches!(
            a.schemes[0].eligibility,
            Eligibility::Ineligible { .. }
        ));
    }

    #[test]
    fn the_ceiling_that_applies_is_the_one_in_force_on_the_queried_date() {
        // $13,000 was over the ceiling in 2018 and under it in 2025. Same
        // household, same flat, different answer -- because the rule changed,
        // and the crate looks the rule up rather than remembering one.
        let in_2018 = assess(couple(13_000), bto_4room(), on(2018, 6, 1));
        assert!(matches!(
            in_2018.schemes[0].eligibility,
            Eligibility::Ineligible { .. }
        ));

        let in_2025 = assess(couple(13_000), bto_4room(), on(2025, 6, 1));
        assert!(matches!(
            in_2025.schemes[0].eligibility,
            Eligibility::Eligible { .. }
        ));
    }

    #[test]
    fn a_single_applicant_at_exactly_35_is_eligible_and_at_34_is_not() {
        let single = |age: u32| Household {
            applicants: vec![citizen(age, MaritalStatus::Single)],
            nucleus: FamilyNucleus::SingleApplicant,
            monthly_income: MonthlyIncome(Sgd::dollars(5_000)),
        };
        let flat = Purchase {
            mode: PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder),
            flat_type: FlatType::TwoRoomFlexi,
            classification: Some(FlatClassification::Standard),
        };

        let at_35 = assess(single(35), flat, on(2025, 1, 15));
        assert!(
            matches!(at_35.schemes[0].eligibility, Eligibility::Eligible { .. }),
            "'at least 35' includes exactly 35: {:?}",
            at_35.schemes[0].eligibility
        );

        let at_34 = assess(single(34), flat, on(2025, 1, 15));
        match &at_34.schemes[0].eligibility {
            Eligibility::Ineligible { reasons } => {
                assert!(reasons[0].statement.contains("at least 35 years"));
                assert!(!reasons[0].citation.section.is_empty());
            }
            other => panic!("expected Ineligible at 34, got {other:?}"),
        }
    }

    // -- Indeterminate: the honest answer most systems omit. -----------------

    #[test]
    fn a_widowed_applicant_below_35_is_indeterminate_not_a_confident_no() {
        // A widowed applicant is subject to a lower minimum age, which this crate
        // does not hold. Answering "ineligible: you are under 35" would be a
        // cited, confident, and possibly WRONG refusal.
        let household = Household {
            applicants: vec![citizen(30, MaritalStatus::Widowed)],
            nucleus: FamilyNucleus::SingleApplicant,
            monthly_income: MonthlyIncome(Sgd::dollars(4_000)),
        };
        let a = assess(
            household,
            Purchase {
                mode: PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder),
                flat_type: FlatType::TwoRoomFlexi,
                classification: Some(FlatClassification::Standard),
            },
            on(2025, 1, 15),
        );

        match &a.schemes[0].eligibility {
            Eligibility::Indeterminate { unknowns } => {
                assert!(
                    unknowns.iter().any(|u| u.subject.contains("widowed")),
                    "the unknown must name what is missing: {unknowns:?}"
                );
            }
            other => panic!("expected Indeterminate for a widowed applicant, got {other:?}"),
        }
        assert!(!a.any_scheme_eligible());
    }

    #[test]
    fn a_scheme_the_crate_does_not_model_is_indeterminate_not_ineligible() {
        let household = Household {
            applicants: vec![
                citizen(25, MaritalStatus::Single),
                citizen(22, MaritalStatus::Single),
            ],
            nucleus: FamilyNucleus::OrphanedSiblings,
            monthly_income: MonthlyIncome(Sgd::dollars(6_000)),
        };
        let a = assess(household, bto_4room(), on(2025, 1, 15));

        match &a.schemes[0].eligibility {
            Eligibility::Indeterminate { unknowns } => {
                assert!(unknowns.iter().any(|u| u.subject.contains("Orphans")));
            }
            other => panic!("expected Indeterminate for the Orphans Scheme, got {other:?}"),
        }
    }

    #[test]
    fn a_definite_failure_beats_an_unknown_but_never_the_other_way_round() {
        // This household is over the ceiling (a modelled rule, definitively
        // failed) *and* buying under an unmodelled scheme. The failure decides.
        let household = Household {
            applicants: vec![
                citizen(25, MaritalStatus::Single),
                citizen(22, MaritalStatus::Single),
            ],
            nucleus: FamilyNucleus::OrphanedSiblings,
            monthly_income: MonthlyIncome(Sgd::dollars(20_000)),
        };
        let a = assess(household, bto_4room(), on(2025, 1, 15));
        assert!(
            matches!(a.schemes[0].eligibility, Eligibility::Ineligible { .. }),
            "an unmodelled rule can withhold a yes; it cannot overturn a no"
        );
    }

    #[test]
    fn a_date_before_the_earliest_provision_is_indeterminate_not_a_panic_and_not_a_guess() {
        let a = assess(couple(5_000), bto_4room(), on(2010, 1, 1));
        match &a.schemes[0].eligibility {
            Eligibility::Indeterminate { unknowns } => {
                assert!(
                    unknowns
                        .iter()
                        .any(|u| u.because.contains("earliest modelled provision")),
                    "the crate must say that it holds nothing that far back: {unknowns:?}"
                );
            }
            other => panic!("expected Indeterminate for a 2010 query, got {other:?}"),
        }
    }

    // -- Citizenship, the input that changes everything. ---------------------

    #[test]
    fn a_household_with_no_citizen_cannot_buy_a_new_flat_and_is_told_which_rule_says_so() {
        let household = Household {
            applicants: vec![
                Applicant {
                    age: Age(35),
                    residency: Residency::PermanentResident,
                    ethnicity: EthnicGroup::IndianOther,
                    subsidy_history: SubsidyHistory::FirstTimer,
                    marital_status: MaritalStatus::Married,
                },
                Applicant {
                    age: Age(33),
                    residency: Residency::PermanentResident,
                    ethnicity: EthnicGroup::IndianOther,
                    subsidy_history: SubsidyHistory::FirstTimer,
                    marital_status: MaritalStatus::Married,
                },
            ],
            nucleus: FamilyNucleus::SpousesOrParentsChildren,
            monthly_income: MonthlyIncome(Sgd::dollars(8_000)),
        };

        let new_flat = assess(household.clone(), bto_4room(), on(2025, 1, 15));
        match &new_flat.schemes[0].eligibility {
            Eligibility::Ineligible { reasons } => {
                assert!(reasons[0].statement.contains("Singapore Citizen"));
                assert!(!reasons[0].citation.url.is_empty());
            }
            other => {
                panic!("expected Ineligible for an all-PR household buying new, got {other:?}")
            }
        }

        // The same household buying a resale flat is NOT ineligible -- it is
        // indeterminate, because the three-year PR waiting period turns on a date
        // the crate is not given.
        let resale = assess(
            household,
            Purchase {
                mode: PurchaseMode::Resale {
                    grant: GrantIntent::NotApplying,
                },
                flat_type: FlatType::FourRoom,
                classification: None,
            },
            on(2025, 1, 15),
        );
        match &resale.schemes[0].eligibility {
            Eligibility::Indeterminate { unknowns } => {
                assert!(
                    unknowns
                        .iter()
                        .any(|u| u.subject.contains("Permanent Resident"))
                );
            }
            other => panic!("expected Indeterminate for an all-PR resale purchase, got {other:?}"),
        }
    }

    // -- Grants. -------------------------------------------------------------

    #[test]
    fn the_ehg_is_indeterminate_today_because_the_2024_figures_are_not_held() {
        // The most important test in the file. The 2019 schedule is NOT carried
        // forward past the day it was superseded, even though doing so would make
        // the crate look more capable.
        let a = assess(couple(3_000), bto_4room(), on(2025, 1, 15));
        let ehg = a
            .grants
            .iter()
            .find(|g| g.grant == Grant::EnhancedHousing)
            .expect("the EHG must be reported even when it cannot be quantified");

        match &ehg.outcome {
            GrantOutcome::Indeterminate(unknown) => {
                assert!(
                    unknown.because.contains("2024 National Day Rally"),
                    "the gap must name what changed: {unknown:?}"
                );
            }
            other => panic!("expected Indeterminate EHG on a 2025 date, got {other:?}"),
        }
    }

    #[test]
    fn the_2019_ehg_is_still_answerable_for_a_2019_application_and_carries_its_citation() {
        let a = assess(couple(3_000), bto_4room(), on(2020, 3, 1));
        let ehg = a
            .grants
            .iter()
            .find(|g| g.grant == Grant::EnhancedHousing)
            .unwrap();

        match &ehg.outcome {
            GrantOutcome::Indicative(dated) => {
                // $2,501-$3,000 band: $80,000 - 3 * $5,000.
                assert_eq!(dated.value.0, Sgd::dollars(65_000));
                assert!(!dated.citation.url.is_empty());
                assert!(dated.effective.contains(on(2020, 3, 1)));
            }
            other => panic!("expected an indicative 2019-schedule EHG, got {other:?}"),
        }
    }

    #[test]
    fn a_household_above_every_grant_band_is_told_the_grant_is_not_payable_with_a_citation() {
        // Distinct from "we don't know": this is a fact, and it is cited.
        let a = assess(couple(12_000), bto_4room(), on(2020, 3, 1));
        let ehg = a
            .grants
            .iter()
            .find(|g| g.grant == Grant::EnhancedHousing)
            .unwrap();

        match &ehg.outcome {
            GrantOutcome::NotPayable(reason) => {
                assert!(reason.statement.contains("above the highest band"));
                assert!(!reason.citation.url.is_empty());
            }
            other => panic!("expected NotPayable, got {other:?}"),
        }
    }

    #[test]
    fn the_grants_whose_amounts_are_not_modelled_are_reported_rather_than_omitted() {
        // An omitted grant reads exactly like a grant that does not apply.
        let a = assess(couple(5_000), bto_4room(), on(2025, 1, 15));
        let proximity = a
            .grants
            .iter()
            .find(|g| g.grant == Grant::Proximity)
            .expect("the Proximity Housing Grant must appear even though its amount is unknown");
        assert!(matches!(
            proximity.outcome,
            GrantOutcome::Indeterminate { .. }
        ));
    }

    // -- MOP, EIP, and the caveats. ------------------------------------------

    #[test]
    fn the_mop_follows_the_flat_classification_and_carries_its_citation() {
        let plus = Purchase {
            mode: PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder),
            flat_type: FlatType::FourRoom,
            classification: Some(FlatClassification::Plus),
        };
        let a = assess(couple(8_000), plus, on(2025, 1, 15));
        let mop = a
            .minimum_occupation_period
            .known()
            .expect("a Plus flat's MOP is modelled");
        assert_eq!(mop.value, MinimumOccupationPeriod(Months(120)));
        assert!(!mop.citation.url.is_empty());
    }

    #[test]
    fn a_purchase_with_no_classification_after_the_framework_began_yields_no_mop_and_says_why() {
        let unclassified = Purchase {
            mode: PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder),
            flat_type: FlatType::FourRoom,
            classification: None,
        };
        let a = assess(couple(8_000), unclassified, on(2025, 1, 15));
        assert!(
            a.minimum_occupation_period.known().is_none(),
            "Standard must not be assumed"
        );
        assert!(
            a.caveats
                .iter()
                .any(|c| c.subject.contains("flat classification"))
        );
    }

    #[test]
    fn the_eip_reports_its_limits_and_admits_it_cannot_answer_the_actual_question() {
        let a = assess(couple(8_000), bto_4room(), on(2025, 1, 15));
        let quota = &a.ethnic_quota;
        assert_eq!(quota.group, EthnicGroup::Chinese);

        let limits = quota.limits.known().expect("the EIP limits are modelled");
        assert!(!limits.citation.url.is_empty());

        assert!(
            quota.availability.because.contains("composition"),
            "knowing the limits is not knowing whether a block has space, and the crate must say so"
        );
    }

    // -- The properties the whole crate exists to guarantee. ------------------

    #[test]
    fn every_figure_in_an_assessment_carries_a_citation() {
        let a = assess(couple(8_000), bto_4room(), on(2025, 1, 15));

        assert!(a.income_ceiling.citation().is_some());
        assert!(a.minimum_occupation_period.citation().is_some());
        assert!(a.ethnic_quota.limits.citation().is_some());
        assert!(
            !a.citations().is_empty(),
            "an assessment with no citations has said nothing"
        );
        for citation in a.citations() {
            assert!(!citation.url.is_empty());
            assert!(!citation.section.is_empty());
        }
    }

    #[test]
    fn the_crate_admits_that_none_of_its_figures_have_been_verified() {
        // When someone fetches the sources and flips these to Verified, this test
        // should be tightened -- not deleted. Until then it is the truth.
        let policy = HdbPolicy::published();
        let unverified = policy.unverified_provisions();
        assert!(
            !unverified.is_empty(),
            "every citation in this crate was transcribed offline; claiming otherwise would be \
             the crate's first lie"
        );
        for citation in unverified {
            assert!(!citation.is_verified());
        }
    }

    #[test]
    fn an_assessment_always_carries_its_caveats() {
        let a = assess(couple(8_000), bto_4room(), on(2025, 1, 15));
        assert!(
            !a.caveats.is_empty(),
            "the things this crate does not check must travel with every answer it gives"
        );
        assert!(
            a.caveats
                .iter()
                .any(|c| c.subject.contains("grant eligibility"))
        );
    }
}
