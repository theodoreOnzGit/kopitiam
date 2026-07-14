//! The query API: *"for this member, this wage, this month — what does the
//! published policy say, and where does it say it?"*
//!
//! Note the shape of that question. It is not "what should this person do". It
//! is not even "what will CPF charge". It is **"what does the published policy
//! say"** — and the answer always arrives with its receipts attached.
//!
//! Every result type in this module carries a [`Citations`] block. That is not
//! metadata bolted on afterwards; it is a mandatory field, and the only way to
//! get a number out of this crate is to also be handed the source for it.

use std::collections::BTreeMap;

use crate::cpf::citation::Citation;
use crate::cpf::date::Date;
use crate::cpf::error::CpfError;
use crate::cpf::money::Sgd;
use crate::cpf::published;
use crate::cpf::rates::{
    Allocation, AllocationRatios, AllocationSchedule, ContributionRates, ContributionSchedule,
    ContributionSplit, InterestFloors, RetirementSums, WageCeilings,
};
use crate::cpf::structure::{
    Age, AllocationAgeBand, ContributionAgeBand, Residency, WageBand, allocation_band_on,
    contribution_band_on,
};
use crate::cpf::temporal::{Dated, PolicyTable};

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// How the caller identifies the member's age.
///
/// Two constructors, because there are genuinely two questions, and answering
/// one with the other is a bug (see [`ContributionAgeBand::for_age`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Member {
    /// **Preferred.** Age bands are resolved with CPF's month-after-the-birthday
    /// rule, which is the rule that actually governs payroll.
    BornOn(Date),

    /// Age bands are resolved from chronological age alone.
    ///
    /// Correct for the question *"what does the table say for a 57-year-old?"*.
    /// **Not** correct for *"what do I deduct from this employee in March?"* if
    /// they had a 55th, 60th, 65th or 70th birthday recently — for that, CPF's
    /// rule depends on the birthday *month*, which an age cannot express. Use
    /// [`Member::BornOn`].
    AgedExactly(Age),
}

/// A month's wages, split the way CPF splits them.
///
/// The distinction is not cosmetic: Ordinary Wages are capped **monthly** and
/// Additional Wages **annually**, against a ceiling that is a residual of the
/// first. See [`WageCeilings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonthlyWages {
    /// Wages due or granted wholly and exclusively for the month: salary.
    pub ordinary: Sgd,
    /// Everything else paid in the month: bonus, leave pay, commission.
    pub additional: Sgd,
}

impl MonthlyWages {
    pub const fn new(ordinary: Sgd, additional: Sgd) -> Self {
        Self {
            ordinary,
            additional,
        }
    }

    /// Ordinary-only wages, the common case.
    pub const fn salary(ordinary: Sgd) -> Self {
        Self {
            ordinary,
            additional: Sgd::ZERO,
        }
    }

    /// Total Wages — the figure the wage *band* is classified on.
    pub fn total(self) -> Sgd {
        self.ordinary + self.additional
    }
}

/// The calendar-year context an Additional Wage computation needs.
///
/// # Why this cannot be inferred from one month
///
/// The Additional Wage ceiling is **annual and a residual**:
/// `$102,000 − (Ordinary Wages subject to CPF for the year)`. Computing it for
/// March therefore requires knowing the *whole year's* Ordinary Wages — which in
/// March nobody does.
///
/// CPF's own answer is that employers compute the ceiling on the actual (or best
/// projected) Ordinary Wages for the year and reconcile at year end. KOPITIAM
/// does not hide that: the caller must supply the annual figure explicitly, and
/// is thereby made to confront the fact that a mid-year AW computation is
/// **provisional**. Anything that quietly produced a number from a single month's
/// data would be concealing a projection it had made on the caller's behalf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YearContext {
    /// Ordinary Wages **subject to CPF** — i.e. already capped at the monthly
    /// Ordinary Wage ceiling in force for each month — totalled across the whole
    /// calendar year. Actual, or the employer's best projection.
    ///
    /// **Not** the Ordinary Wages *paid*. Passing the paid figure for a
    /// high earner wrongly collapses their Additional Wage ceiling to zero. See
    /// [`WageCeilings`].
    ///
    /// Because the monthly ceiling has itself changed mid-year (September 2023),
    /// this must be accumulated month by month at each month's ceiling — it is not
    /// `12 x ceiling`.
    pub annual_ordinary_wages_subject_to_cpf: Sgd,

    /// Additional Wages already subjected to CPF earlier in the same calendar
    /// year, which have already consumed part of the annual ceiling.
    pub additional_wages_already_subject_to_cpf: Sgd,
}

impl YearContext {
    /// No Additional Wages anywhere in the year. Use with
    /// [`MonthlyWages::salary`] for the plain salaried case.
    pub const fn none() -> Self {
        Self {
            annual_ordinary_wages_subject_to_cpf: Sgd::ZERO,
            additional_wages_already_subject_to_cpf: Sgd::ZERO,
        }
    }

    /// The year's Ordinary Wages subject to CPF, with no Additional Wages yet
    /// paid this year.
    pub const fn with_annual_ordinary_wages(annual_ordinary_wages_subject_to_cpf: Sgd) -> Self {
        Self {
            annual_ordinary_wages_subject_to_cpf,
            additional_wages_already_subject_to_cpf: Sgd::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// The sources behind a [`ContributionBreakdown`]. **Not optional.**
///
/// One citation per policy table consulted. If a user asks "why 20%?", the
/// answer is [`Citations::contribution_rates`], verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct Citations {
    pub contribution_rates: Citation,
    pub allocation_ratios: Citation,
    pub wage_ceilings: Citation,
}

impl Citations {
    pub fn all(&self) -> [&Citation; 3] {
        [
            &self.contribution_rates,
            &self.allocation_ratios,
            &self.wage_ceilings,
        ]
    }
}

/// A complete, self-explaining answer: every intermediate figure, every band
/// that was selected, and every source.
///
/// The intermediate values are public and deliberately exhaustive. A breakdown
/// that reported only the final contribution would be un-auditable — and in a
/// domain where the answer depends on two ceilings, two different age-band
/// systems and a three-step statutory rounding rule, "un-auditable" means
/// "un-trustable".
#[derive(Debug, Clone, PartialEq)]
pub struct ContributionBreakdown {
    /// The month asked about.
    pub month: Date,
    pub residency: Residency,

    /// The band the contribution *rates* came from.
    pub contribution_band: ContributionAgeBand,
    /// The band the *allocation* ratios came from. Different band system — see
    /// [`crate::cpf::structure`].
    pub allocation_band: AllocationAgeBand,
    /// The Total Wages band. Only [`WageBand::AtOrAbove750`] is answerable.
    pub wage_band: WageBand,

    /// The ceilings in force for this month.
    pub ceilings: WageCeilings,
    /// Ordinary Wages after the monthly cap.
    pub ordinary_wages_subject_to_cpf: Sgd,
    /// The annual Additional Wage ceiling: `$102,000 − annual OW subject to CPF`.
    pub additional_wage_ceiling: Sgd,
    /// How much of that ceiling was still unused before this month's Additional
    /// Wages.
    pub additional_wage_headroom: Sgd,
    /// Additional Wages after the annual cap.
    pub additional_wages_subject_to_cpf: Sgd,
    /// The base the rates were applied to.
    pub total_wages_subject_to_cpf: Sgd,

    /// The published rates for the selected band.
    pub rates: ContributionRates,
    /// The contribution, split per CPF's statutory rounding rule.
    pub contribution: ContributionSplit,
    /// The published allocation ratios for the selected band.
    pub allocation_ratios: AllocationRatios,
    /// The contribution split across accounts.
    pub allocation: Allocation,

    /// Where every one of the above came from.
    pub citations: Citations,
}

// ---------------------------------------------------------------------------
// The policy
// ---------------------------------------------------------------------------

/// The assembled CPF policy tables, and the queries over them.
///
/// # Scope
///
/// This is **not** a complete model of CPF. It is a correctly-shaped one with a
/// small, honestly-labelled slice of policy loaded. See [`published`] for the
/// exhaustive list of what is populated and what is deliberately absent, and why.
///
/// # Not financial advice
///
/// See the crate-level documentation. This type answers questions about what a
/// published document says. It does not tell anyone what to do with their money.
#[derive(Debug, Clone)]
pub struct CpfPolicy {
    /// Keyed by residency. Statuses KOPITIAM has no cited data for map to an
    /// **empty table** — which is the honest representation of a gap, and which
    /// makes populating them later a data change rather than an API change.
    contribution: BTreeMap<Residency, PolicyTable<ContributionSchedule>>,
    allocation: BTreeMap<Residency, PolicyTable<AllocationSchedule>>,
    ceilings: PolicyTable<WageCeilings>,
    retirement_sums: PolicyTable<RetirementSums>,
    interest: PolicyTable<InterestFloors>,
}

impl CpfPolicy {
    /// The policy tables shipped with KOPITIAM. See [`published`] — read the
    /// confidence statement there before relying on any figure.
    pub fn published() -> Self {
        let mut contribution = BTreeMap::new();
        contribution.insert(
            Residency::CitizenOrPrFromThirdYear,
            published::contribution_rates_citizen_and_pr3plus(),
        );
        // Named, and empty. The gap is visible in the data rather than hidden in
        // a `match` arm.
        contribution.insert(
            Residency::PrFirstYear,
            PolicyTable::empty("contribution rates (SPR, 1st year)"),
        );
        contribution.insert(
            Residency::PrSecondYear,
            PolicyTable::empty("contribution rates (SPR, 2nd year)"),
        );

        let mut allocation = BTreeMap::new();
        allocation.insert(
            Residency::CitizenOrPrFromThirdYear,
            published::allocation_ratios_citizen_and_pr3plus(),
        );
        allocation.insert(
            Residency::PrFirstYear,
            PolicyTable::empty("allocation ratios (SPR, 1st year)"),
        );
        allocation.insert(
            Residency::PrSecondYear,
            PolicyTable::empty("allocation ratios (SPR, 2nd year)"),
        );

        Self {
            contribution,
            allocation,
            ceilings: published::wage_ceilings(),
            retirement_sums: published::retirement_sums(),
            interest: published::interest_floors(),
        }
    }

    /// Checks every table for internally overlapping effective periods, and every
    /// allocation band for summing to exactly 100%.
    ///
    /// Run over the built-in tables in the test suite. Exposed so that a caller
    /// who assembles their own [`CpfPolicy`] — from ingested documents, say — can
    /// hold it to the same standard.
    pub fn validate(&self) -> Result<(), CpfError> {
        for table in self.contribution.values() {
            table.validate()?;
        }
        for table in self.allocation.values() {
            table.validate()?;
            for entry in table.entries() {
                for (band, ratios) in entry.value.bands() {
                    ratios.validate(band)?;
                }
            }
        }
        self.ceilings.validate()?;
        self.retirement_sums.validate()?;
        self.interest.validate()?;
        Ok(())
    }

    // -- Individual table lookups ------------------------------------------

    /// The wage ceilings in force on `date`, with their citation.
    pub fn wage_ceilings_on(&self, date: Date) -> Result<&Dated<WageCeilings>, CpfError> {
        self.ceilings.on(date)
    }

    /// The interest **floors** in force on `date`. Not the declared rates — see
    /// [`InterestFloors`].
    pub fn interest_floors_on(&self, date: Date) -> Result<&Dated<InterestFloors>, CpfError> {
        self.interest.on(date)
    }

    /// The contribution rates for one age band, on one date, for one residency —
    /// with the citation.
    ///
    /// # Errors
    ///
    /// * [`CpfError::NotPopulated`] if the residency has no cited table, or the
    ///   revision does not carry that band.
    /// * [`CpfError::NoRuleInEffect`] if no revision covers `date`.
    pub fn contribution_rates_on(
        &self,
        date: Date,
        residency: Residency,
        band: ContributionAgeBand,
    ) -> Result<Dated<ContributionRates>, CpfError> {
        let entry = self.contribution_table(residency)?.on(date)?;
        let rates = entry.value.band(band)?;
        Ok(Dated::new(rates, entry.effective, entry.source.clone()))
    }

    /// The allocation ratios for one age band, on one date, for one residency —
    /// with the citation.
    ///
    /// # Errors
    ///
    /// [`CpfError::NotPopulated`] for every band at age 55 and above; KOPITIAM
    /// holds no cited post-55 allocation data. See [`published`].
    pub fn allocation_ratios_on(
        &self,
        date: Date,
        residency: Residency,
        band: AllocationAgeBand,
    ) -> Result<Dated<AllocationRatios>, CpfError> {
        let entry = self.allocation_table(residency)?.on(date)?;
        let ratios = entry.value.band(band)?;
        Ok(Dated::new(ratios, entry.effective, entry.source.clone()))
    }

    /// The retirement sums for the cohort a member born on `date_of_birth` belongs
    /// to — that is, the sums published for the year in which they **turn 55**.
    ///
    /// # This is the lookup people get wrong
    ///
    /// A member's retirement sum is fixed by their cohort and follows them for
    /// life. It is **not** the sum published in the year you happen to be asking.
    /// This method exists so that the right key — the 55th birthday — is computed
    /// for you rather than left as an opportunity.
    ///
    /// # Errors
    ///
    /// [`CpfError::NoRuleInEffect`] if KOPITIAM holds no sums for that cohort.
    /// Cohorts turning 55 from 2027 onward are absent, and the engine will *not*
    /// extrapolate the announced 3.5%/year trend to invent one.
    pub fn retirement_sums_for_cohort(
        &self,
        date_of_birth: Date,
    ) -> Result<&Dated<RetirementSums>, CpfError> {
        // The cohort key is the 55th birthday. Any date within the cohort year
        // selects the same entry, so the day-of-month is immaterial here — but
        // computing it from the date of birth, rather than accepting a bare year,
        // is what stops a caller from passing "this year" by mistake.
        let fifty_fifth = Date::new(
            date_of_birth.year() + 55,
            date_of_birth.month(),
            // 29 February -> 28 February in a non-leap year.
            if date_of_birth.month() == 2
                && date_of_birth.day() == 29
                && Date::new(date_of_birth.year() + 55, 2, 29).is_err()
            {
                28
            } else {
                date_of_birth.day()
            },
        )?;
        self.retirement_sums.on(fifty_fifth)
    }

    // -- The headline query -------------------------------------------------

    /// Given a member, a month's wages, and the year's context: what does the
    /// published policy say the contribution and allocation are — and on what
    /// authority?
    ///
    /// # The order of operations, which is the whole computation
    ///
    /// 1. Look up the **wage ceilings** in force for `month`.
    /// 2. Cap the **Ordinary Wages** at the monthly ceiling.
    /// 3. Compute the **Additional Wage ceiling** as the annual residual, subtract
    ///    the Additional Wages already charged this year, and cap this month's
    ///    Additional Wages at what is left.
    /// 4. Classify the **Total Wages band**. Anything below $750/month is not
    ///    answerable (see [`WageBand`]).
    /// 5. Select the **contribution age band** — by the month-after-the-birthday
    ///    rule, if a date of birth was given.
    /// 6. Apply the rates to the capped wages using CPF's **statutory rounding**
    ///    (total to the nearest dollar; employee's cents dropped; employer takes
    ///    the residual).
    /// 7. Select the **allocation age band** — a *different* band system — and
    ///    split the total across the accounts.
    ///
    /// Each step's citation is collected and returned.
    ///
    /// # Errors
    ///
    /// Every gap in KOPITIAM's data surfaces here as a typed error, never as a
    /// plausible number:
    ///
    /// * [`CpfError::NoRuleInEffect`] — no cited rule covers `month`.
    /// * [`CpfError::NotPopulated`] — the residency, wage band, or age band is
    ///   part of the model but has no cited data. The most common case by far is a
    ///   member **aged 55 or above**, whose allocation ratios KOPITIAM does not
    ///   hold.
    pub fn contribution(
        &self,
        residency: Residency,
        member: Member,
        wages: MonthlyWages,
        month: Date,
        year: YearContext,
    ) -> Result<ContributionBreakdown, CpfError> {
        // 1-3. Ceilings, and the OW/AW interaction.
        let ceilings_entry = self.ceilings.on(month)?;
        let ceilings = ceilings_entry.value;

        let ordinary_wages_subject_to_cpf = ceilings.ordinary_wages_subject_to_cpf(wages.ordinary);
        let additional_wage_ceiling =
            ceilings.additional_wage_ceiling(year.annual_ordinary_wages_subject_to_cpf);
        let additional_wage_headroom =
            (additional_wage_ceiling - year.additional_wages_already_subject_to_cpf)
                .clamp_non_negative();
        let additional_wages_subject_to_cpf = wages.additional.min(additional_wage_headroom);

        let total_wages_subject_to_cpf =
            ordinary_wages_subject_to_cpf + additional_wages_subject_to_cpf;

        // 4. The wage band is classified on Total Wages *as paid*, not on the
        //    capped figure: a member earning $10,000 is a full-rate member even
        //    though only $7,400 of it attracts CPF.
        let wage_band = WageBand::classify(wages.total());
        if !wage_band.is_populated() {
            return Err(CpfError::NotPopulated {
                dimension: format!("contribution rates for '{wage_band}'"),
                reason: "below $750/month the employee's share is phased in by a formula that \
                         depends on both the wage and the age band; KOPITIAM does not hold it, and \
                         guessing it would under- or over-deduct from the lowest-paid members",
            });
        }

        // 5. Bands. Two different band systems, resolved separately.
        let (contribution_band, allocation_band) = match member {
            Member::BornOn(dob) => (
                contribution_band_on(dob, month)?,
                allocation_band_on(dob, month)?,
            ),
            Member::AgedExactly(age) => (
                ContributionAgeBand::for_age(age),
                AllocationAgeBand::for_age(age),
            ),
        };

        // 6. Rates and the statutory split.
        let rates_entry = self.contribution_rates_on(month, residency, contribution_band)?;
        let contribution = rates_entry.value.split(total_wages_subject_to_cpf);

        // 7. Allocation. This is where a member aged 55+ is honestly turned away.
        let ratios_entry = self.allocation_ratios_on(month, residency, allocation_band)?;
        let allocation = ratios_entry.value.allocate(contribution.total);

        Ok(ContributionBreakdown {
            month,
            residency,
            contribution_band,
            allocation_band,
            wage_band,
            ceilings,
            ordinary_wages_subject_to_cpf,
            additional_wage_ceiling,
            additional_wage_headroom,
            additional_wages_subject_to_cpf,
            total_wages_subject_to_cpf,
            rates: rates_entry.value,
            contribution,
            allocation_ratios: ratios_entry.value,
            allocation,
            citations: Citations {
                contribution_rates: rates_entry.source,
                allocation_ratios: ratios_entry.source,
                wage_ceilings: ceilings_entry.source.clone(),
            },
        })
    }

    // -- Table accessors, for the ontology bridge and for callers who want to
    //    inspect coverage rather than query a point. -------------------------

    pub fn contribution_table(
        &self,
        residency: Residency,
    ) -> Result<&PolicyTable<ContributionSchedule>, CpfError> {
        let table = self
            .contribution
            .get(&residency)
            .expect("every Residency variant has a table, populated or empty");
        if table.is_empty() {
            return Err(CpfError::NotPopulated {
                dimension: format!("contribution rates for '{residency}'"),
                reason: "Permanent Residents in their 1st and 2nd year pay graduated rates, with \
                         three possible employer/employee combinations by joint election; KOPITIAM \
                         holds none of them and will not guess",
            });
        }
        Ok(table)
    }

    pub fn allocation_table(
        &self,
        residency: Residency,
    ) -> Result<&PolicyTable<AllocationSchedule>, CpfError> {
        let table = self
            .allocation
            .get(&residency)
            .expect("every Residency variant has a table, populated or empty");
        if table.is_empty() {
            return Err(CpfError::NotPopulated {
                dimension: format!("allocation ratios for '{residency}'"),
                reason: "Permanent Residents in their 1st and 2nd year have their own allocation \
                         tables; KOPITIAM holds none of them and will not guess",
            });
        }
        Ok(table)
    }

    pub fn wage_ceiling_table(&self) -> &PolicyTable<WageCeilings> {
        &self.ceilings
    }

    pub fn retirement_sum_table(&self) -> &PolicyTable<RetirementSums> {
        &self.retirement_sums
    }

    pub fn interest_table(&self) -> &PolicyTable<InterestFloors> {
        &self.interest
    }
}

impl Default for CpfPolicy {
    fn default() -> Self {
        Self::published()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpf::money::Rate;

    fn d(y: i32, m: u8, day: u8) -> Date {
        Date::new(y, m, day).unwrap()
    }

    const CITIZEN: Residency = Residency::CitizenOrPrFromThirdYear;

    // -- The built-in tables must be internally sound ------------------------

    /// Every shipped table: no overlapping effective periods, and every
    /// allocation band sums to exactly 100%. This is the guard that a
    /// transcription typo cannot get past.
    #[test]
    fn the_published_tables_validate() {
        CpfPolicy::published().validate().expect("built-in CPF tables must be internally consistent");
    }

    /// The sub-55 allocation ratios must each reproduce as `x/37` — the
    /// cross-check documented on `AllocationRatios`. A transcription error would
    /// almost certainly break this.
    #[test]
    fn every_published_allocation_ratio_sums_to_one() {
        let policy = CpfPolicy::published();
        let table = policy.allocation_table(CITIZEN).unwrap();
        let mut checked = 0;
        for entry in table.entries() {
            for (band, ratios) in entry.value.bands() {
                ratios.validate(band).unwrap();
                assert_eq!(
                    ratios.ordinary + ratios.special + ratios.medisave,
                    Rate::ONE
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 4, "exactly the four sub-55 bands are populated");
    }

    // -- Temporal correctness on real data -----------------------------------

    /// The headline temporal requirement, on the real wage-ceiling table: the
    /// rule effective in 2024 must not answer a 2025 question.
    #[test]
    fn the_2024_ceiling_is_not_returned_for_a_2025_query() {
        let policy = CpfPolicy::published();
        assert_eq!(
            policy.wage_ceilings_on(d(2024, 6, 1)).unwrap().value.ordinary_wage_monthly,
            Sgd::from_dollars(6_800)
        );
        assert_eq!(
            policy.wage_ceilings_on(d(2025, 6, 1)).unwrap().value.ordinary_wage_monthly,
            Sgd::from_dollars(7_400)
        );
        assert_eq!(
            policy.wage_ceilings_on(d(2026, 6, 1)).unwrap().value.ordinary_wage_monthly,
            Sgd::from_dollars(8_000)
        );
    }

    /// The mid-year change, which no `12 x ceiling` model can express.
    #[test]
    fn the_ordinary_wage_ceiling_changed_mid_2023() {
        let policy = CpfPolicy::published();
        assert_eq!(
            policy.wage_ceilings_on(d(2023, 8, 31)).unwrap().value.ordinary_wage_monthly,
            Sgd::from_dollars(6_000)
        );
        assert_eq!(
            policy.wage_ceilings_on(d(2023, 9, 1)).unwrap().value.ordinary_wage_monthly,
            Sgd::from_dollars(6_300)
        );
    }

    /// A date before any cited rule is an honest error, not a panic and not a
    /// fallback to the earliest entry.
    #[test]
    fn a_query_before_any_cited_rule_fails_honestly() {
        let policy = CpfPolicy::published();

        let err = policy.wage_ceilings_on(d(2015, 1, 1)).unwrap_err();
        assert!(matches!(err, CpfError::NoRuleInEffect { .. }));
        // The error must tell the user what IS covered.
        assert!(err.to_string().contains("2023-01-01"));

        let err = policy
            .contribution_rates_on(d(2020, 1, 1), CITIZEN, ContributionAgeBand::UpTo55)
            .unwrap_err();
        assert!(matches!(err, CpfError::NoRuleInEffect { .. }));
    }

    /// Senior rates step up every January. A 2025 rate must not be served for a
    /// 2026 payroll, nor a 2026 rate for 2025.
    #[test]
    fn senior_worker_rates_step_each_january() {
        let policy = CpfPolicy::published();
        let band = ContributionAgeBand::Above55To60;

        let r2024 = policy.contribution_rates_on(d(2024, 12, 31), CITIZEN, band).unwrap();
        let r2025 = policy.contribution_rates_on(d(2025, 1, 1), CITIZEN, band).unwrap();
        let r2026 = policy.contribution_rates_on(d(2026, 1, 1), CITIZEN, band).unwrap();

        assert_eq!(r2024.value.total(), Rate::from_percent_tenths(310)); // 31.0%
        assert_eq!(r2025.value.total(), Rate::from_percent_tenths(325)); // 32.5%
        assert_eq!(r2026.value.total(), Rate::from_percent_tenths(340)); // 34.0%

        // And the 55-and-below band did NOT move — which is exactly why "the CPF
        // rate" is not a thing.
        for date in [d(2024, 6, 1), d(2025, 6, 1), d(2026, 6, 1)] {
            let r = policy
                .contribution_rates_on(date, CITIZEN, ContributionAgeBand::UpTo55)
                .unwrap();
            assert_eq!(r.value.total(), Rate::from_percent_tenths(370));
        }
    }

    // -- Citations are part of the answer ------------------------------------

    /// Every figure that comes out of the headline query carries a source.
    #[test]
    fn every_returned_figure_carries_a_citation() {
        let policy = CpfPolicy::published();
        let breakdown = policy
            .contribution(
                CITIZEN,
                Member::BornOn(d(1990, 5, 20)),
                MonthlyWages::salary(Sgd::from_dollars(5_000)),
                d(2025, 3, 1),
                YearContext::none(),
            )
            .unwrap();

        for citation in breakdown.citations.all() {
            assert!(!citation.publisher.is_empty(), "a citation must name its publisher");
            assert!(!citation.document.is_empty(), "a citation must name its document");
            assert!(!citation.locator.is_empty(), "a citation must locate the claim within the document");
            assert!(citation.url.is_some(), "the CPF Board publishes online; record the URL");
        }

        // And it must be honest about how it got here.
        use crate::cpf::citation::SourceKind;
        assert_eq!(
            breakdown.citations.contribution_rates.source_kind,
            SourceKind::Transcribed,
            "shipped figures are transcribed, not extracted — the label must not drift"
        );

        // The rate itself, and where it says so.
        assert_eq!(breakdown.rates.employee, Rate::from_percent_tenths(200));
        assert!(
            breakdown.citations.contribution_rates.locator.contains("2025"),
            "the citation must point at the revision actually used"
        );
    }

    /// Individual table lookups also hand back the citation, inseparably.
    #[test]
    fn single_table_lookups_carry_citations_too() {
        let policy = CpfPolicy::published();

        let ceilings = policy.wage_ceilings_on(d(2023, 10, 1)).unwrap();
        assert!(ceilings.citation().note.as_deref().unwrap().contains("MID-YEAR"));

        let sums = policy.retirement_sums_for_cohort(d(1970, 3, 15)).unwrap();
        assert!(sums.citation().locator.contains("2025"));

        let interest = policy.interest_floors_on(d(2025, 1, 1)).unwrap();
        assert!(interest.citation().note.as_deref().unwrap().contains("FLOORS ONLY"));
    }

    // -- The headline computation --------------------------------------------

    /// A plain salaried member below the ceiling: 37% total, split 17/20,
    /// allocated by the sub-35 ratios.
    #[test]
    fn a_salaried_member_below_the_ceiling() {
        let policy = CpfPolicy::published();
        let b = policy
            .contribution(
                CITIZEN,
                Member::AgedExactly(Age::years(30)),
                MonthlyWages::salary(Sgd::from_dollars(5_000)),
                d(2025, 3, 1),
                YearContext::none(),
            )
            .unwrap();

        assert_eq!(b.ordinary_wages_subject_to_cpf, Sgd::from_dollars(5_000));
        assert_eq!(b.contribution_band, ContributionAgeBand::UpTo55);
        assert_eq!(b.allocation_band, AllocationAgeBand::UpTo35);

        // 37% of $5,000 = $1,850; employee 20% = $1,000; employer = residual $850.
        assert_eq!(b.contribution.total, Sgd::from_dollars(1_850));
        assert_eq!(b.contribution.employee, Sgd::from_dollars(1_000));
        assert_eq!(b.contribution.employer, Sgd::from_dollars(850));

        // Every cent lands somewhere.
        assert_eq!(b.allocation.total(), b.contribution.total);
    }

    /// The Ordinary Wage ceiling actually bites, and it bites differently in
    /// 2025 than in 2026.
    #[test]
    fn the_ordinary_wage_ceiling_caps_a_high_salary() {
        let policy = CpfPolicy::published();
        let query = |month| {
            policy
                .contribution(
                    CITIZEN,
                    Member::AgedExactly(Age::years(40)),
                    MonthlyWages::salary(Sgd::from_dollars(12_000)),
                    month,
                    YearContext::none(),
                )
                .unwrap()
        };

        let in_2025 = query(d(2025, 3, 1));
        assert_eq!(in_2025.ordinary_wages_subject_to_cpf, Sgd::from_dollars(7_400));
        // 37% of $7,400 = $2,738.
        assert_eq!(in_2025.contribution.total, Sgd::from_dollars(2_738));

        let in_2026 = query(d(2026, 3, 1));
        assert_eq!(in_2026.ordinary_wages_subject_to_cpf, Sgd::from_dollars(8_000));
        // 37% of $8,000 = $2,960.
        assert_eq!(in_2026.contribution.total, Sgd::from_dollars(2_960));
    }

    /// **The OW and AW ceilings interacting.** A high earner receives a bonus.
    ///
    /// 2025: OW ceiling $7,400/month, so a $12,000/month salary contributes
    /// `12 x $7,400 = $88,800` of Ordinary Wages subject to CPF. The Additional
    /// Wage ceiling is therefore `$102,000 − $88,800 = $13,200`. A $30,000 bonus
    /// is capped at $13,200.
    #[test]
    fn the_ordinary_and_additional_wage_ceilings_interact() {
        let policy = CpfPolicy::published();

        let annual_ow_subject = Sgd::from_dollars(7_400 * 12); // $88,800
        let b = policy
            .contribution(
                CITIZEN,
                Member::AgedExactly(Age::years(40)),
                MonthlyWages::new(Sgd::from_dollars(12_000), Sgd::from_dollars(30_000)),
                d(2025, 12, 1),
                YearContext::with_annual_ordinary_wages(annual_ow_subject),
            )
            .unwrap();

        assert_eq!(b.ordinary_wages_subject_to_cpf, Sgd::from_dollars(7_400));
        assert_eq!(b.additional_wage_ceiling, Sgd::from_dollars(13_200));
        assert_eq!(b.additional_wages_subject_to_cpf, Sgd::from_dollars(13_200));

        // Total base: the capped OW plus the capped AW.
        assert_eq!(b.total_wages_subject_to_cpf, Sgd::from_dollars(20_600));
        // 37% of $20,600 = $7,622.
        assert_eq!(b.contribution.total, Sgd::from_dollars(7_622));
        assert_eq!(b.allocation.total(), b.contribution.total);
    }

    /// A second bonus in the same year finds the Additional Wage ceiling already
    /// partly consumed.
    #[test]
    fn additional_wage_headroom_is_consumed_across_the_year() {
        let policy = CpfPolicy::published();
        let annual_ow_subject = Sgd::from_dollars(7_400 * 12); // AW ceiling = $13,200

        let b = policy
            .contribution(
                CITIZEN,
                Member::AgedExactly(Age::years(40)),
                MonthlyWages::new(Sgd::from_dollars(12_000), Sgd::from_dollars(10_000)),
                d(2025, 12, 1),
                YearContext {
                    annual_ordinary_wages_subject_to_cpf: annual_ow_subject,
                    additional_wages_already_subject_to_cpf: Sgd::from_dollars(10_000),
                },
            )
            .unwrap();

        assert_eq!(b.additional_wage_ceiling, Sgd::from_dollars(13_200));
        // Only $3,200 of headroom left, so the $10,000 bonus is capped there.
        assert_eq!(b.additional_wage_headroom, Sgd::from_dollars(3_200));
        assert_eq!(b.additional_wages_subject_to_cpf, Sgd::from_dollars(3_200));
        assert_eq!(b.total_wages_subject_to_cpf, Sgd::from_dollars(10_600));
    }

    /// Once the ceiling is exhausted, further Additional Wages attract nothing —
    /// and the headroom clamps at zero rather than going negative.
    #[test]
    fn an_exhausted_additional_wage_ceiling_yields_no_contribution_on_a_bonus() {
        let policy = CpfPolicy::published();
        let b = policy
            .contribution(
                CITIZEN,
                Member::AgedExactly(Age::years(40)),
                MonthlyWages::new(Sgd::from_dollars(12_000), Sgd::from_dollars(50_000)),
                d(2025, 12, 1),
                YearContext {
                    annual_ordinary_wages_subject_to_cpf: Sgd::from_dollars(88_800),
                    additional_wages_already_subject_to_cpf: Sgd::from_dollars(13_200),
                },
            )
            .unwrap();

        assert_eq!(b.additional_wage_headroom, Sgd::ZERO);
        assert!(!b.additional_wage_headroom.is_negative());
        assert_eq!(b.additional_wages_subject_to_cpf, Sgd::ZERO);
        // Only the capped Ordinary Wages remain.
        assert_eq!(b.total_wages_subject_to_cpf, Sgd::from_dollars(7_400));
    }

    // -- Age-band boundaries, through the full query --------------------------

    /// The month-after-birthday rule, exercised end to end: the same member, the
    /// same wage, one month apart, straddling their 55th birthday month — and the
    /// contribution changes.
    #[test]
    fn crossing_the_55_boundary_changes_the_contribution() {
        let policy = CpfPolicy::published();
        let dob = d(1970, 3, 15); // turns 55 on 15 March 2025
        let wage = MonthlyWages::salary(Sgd::from_dollars(5_000));

        // March 2025 — the birthday month. Still 37%, still below-55 allocation.
        let march = policy
            .contribution(CITIZEN, Member::BornOn(dob), wage, d(2025, 3, 1), YearContext::none())
            .unwrap();
        assert_eq!(march.contribution_band, ContributionAgeBand::UpTo55);
        assert_eq!(march.allocation_band, AllocationAgeBand::Above50To55);
        assert_eq!(march.rates.total(), Rate::from_percent_tenths(370));
        assert_eq!(march.contribution.total, Sgd::from_dollars(1_850));

        // April 2025 — the month after. The band moves, and KOPITIAM now has to
        // admit it does not hold post-55 allocation ratios.
        let april = policy.contribution(
            CITIZEN,
            Member::BornOn(dob),
            wage,
            d(2025, 4, 1),
            YearContext::none(),
        );
        let err = april.unwrap_err();
        assert!(
            matches!(err, CpfError::NotPopulated { .. }),
            "post-55 allocation is an honest gap, not a guess: {err}"
        );
        assert!(err.to_string().contains("Special Account"));

        // The *rates* for that band are populated, though — the gap is allocation
        // only, and the engine is precise about which.
        let rates = policy
            .contribution_rates_on(d(2025, 4, 1), CITIZEN, ContributionAgeBand::Above55To60)
            .unwrap();
        assert_eq!(rates.value.total(), Rate::from_percent_tenths(325));
    }

    /// The allocation band boundary at 35, both sides, through the full query.
    /// The contribution is identical; the *split* is not.
    #[test]
    fn crossing_the_35_allocation_boundary_changes_only_the_split() {
        let policy = CpfPolicy::published();
        let dob = d(1990, 6, 10); // turns 35 on 10 June 2025
        let wage = MonthlyWages::salary(Sgd::from_dollars(5_000));

        let june = policy
            .contribution(CITIZEN, Member::BornOn(dob), wage, d(2025, 6, 1), YearContext::none())
            .unwrap();
        let july = policy
            .contribution(CITIZEN, Member::BornOn(dob), wage, d(2025, 7, 1), YearContext::none())
            .unwrap();

        assert_eq!(june.allocation_band, AllocationAgeBand::UpTo35);
        assert_eq!(july.allocation_band, AllocationAgeBand::Above35To45);

        // Same contribution...
        assert_eq!(june.contribution.total, july.contribution.total);
        assert_eq!(june.contribution_band, july.contribution_band);
        // ...different destination.
        assert_ne!(june.allocation.ordinary, july.allocation.ordinary);
        assert!(july.allocation.ordinary < june.allocation.ordinary, "OA share falls with age");
        assert!(july.allocation.medisave > june.allocation.medisave, "MA share rises with age");

        // And still nothing is lost.
        assert_eq!(june.allocation.total(), june.contribution.total);
        assert_eq!(july.allocation.total(), july.contribution.total);
    }

    // -- Retirement sums: the cohort axis ------------------------------------

    #[test]
    fn retirement_sums_follow_the_cohort_not_the_query_date() {
        let policy = CpfPolicy::published();

        // Turns 55 in 2024.
        let cohort_2024 = policy.retirement_sums_for_cohort(d(1969, 8, 1)).unwrap();
        assert_eq!(cohort_2024.value.basic, Sgd::from_dollars(102_900));
        assert_eq!(cohort_2024.value.full, Sgd::from_dollars(205_800)); // 2 x BRS
        assert_eq!(cohort_2024.value.enhanced, Sgd::from_dollars(308_700)); // 3 x BRS

        // Turns 55 in 2025 — the year the Enhanced multiple went from 3x to 4x.
        let cohort_2025 = policy.retirement_sums_for_cohort(d(1970, 3, 15)).unwrap();
        assert_eq!(cohort_2025.value.basic, Sgd::from_dollars(106_500));
        assert_eq!(cohort_2025.value.full, Sgd::from_dollars(213_000)); // 2 x BRS
        assert_eq!(cohort_2025.value.enhanced, Sgd::from_dollars(426_000)); // 4 x BRS, not 3x

        // A hardcoded `3 * basic` would have been $106,500 short.
        assert_ne!(cohort_2025.value.enhanced, Sgd::from_dollars(106_500 * 3));
    }

    #[test]
    fn full_retirement_sum_is_always_exactly_twice_the_basic() {
        let policy = CpfPolicy::published();
        for entry in policy.retirement_sum_table().entries() {
            assert_eq!(
                entry.value.full,
                entry.value.basic + entry.value.basic,
                "FRS = 2 x BRS is definitional",
            );
        }
    }

    /// A cohort KOPITIAM does not hold: an honest failure, not an extrapolation
    /// of the announced 3.5%/year trend.
    #[test]
    fn a_future_cohort_is_not_extrapolated() {
        let policy = CpfPolicy::published();
        let err = policy.retirement_sums_for_cohort(d(1972, 1, 1)).unwrap_err(); // turns 55 in 2027
        assert!(matches!(err, CpfError::NoRuleInEffect { .. }));
        assert!(err.to_string().contains("will not extrapolate"));
    }

    // -- The gaps fail loudly -------------------------------------------------

    #[test]
    fn a_low_wage_member_is_turned_away_rather_than_guessed_at() {
        let policy = CpfPolicy::published();
        let err = policy
            .contribution(
                CITIZEN,
                Member::AgedExactly(Age::years(30)),
                MonthlyWages::salary(Sgd::from_dollars(600)),
                d(2025, 3, 1),
                YearContext::none(),
            )
            .unwrap_err();
        assert!(matches!(err, CpfError::NotPopulated { .. }));
        assert!(err.to_string().contains("phased in"));
    }

    #[test]
    fn a_new_permanent_resident_is_turned_away_rather_than_guessed_at() {
        let policy = CpfPolicy::published();
        for residency in [Residency::PrFirstYear, Residency::PrSecondYear] {
            let err = policy
                .contribution(
                    residency,
                    Member::AgedExactly(Age::years(30)),
                    MonthlyWages::salary(Sgd::from_dollars(5_000)),
                    d(2025, 3, 1),
                    YearContext::none(),
                )
                .unwrap_err();
            assert!(matches!(err, CpfError::NotPopulated { .. }), "{residency}");
            assert!(err.to_string().contains("graduated rates"));
        }
    }

    /// Exactly $750 is answerable; a cent below is not. The one sub-$750 boundary
    /// that changes an answer, and the one this crate is confident about.
    #[test]
    fn the_750_dollar_full_rate_threshold_is_exact() {
        let policy = CpfPolicy::published();
        let query = |dollars_cents| {
            policy.contribution(
                CITIZEN,
                Member::AgedExactly(Age::years(30)),
                MonthlyWages::salary(dollars_cents),
                d(2025, 3, 1),
                YearContext::none(),
            )
        };
        assert!(query(Sgd::from_dollars(750)).is_ok());
        assert!(query(Sgd::from_cents(74_999)).is_err());
    }
}
