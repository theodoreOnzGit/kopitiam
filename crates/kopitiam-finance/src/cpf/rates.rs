//! The policy *value* types: what a CPF table actually holds once you have
//! looked it up on a date.
//!
//! Each of these is a distinct type even where two of them are "a pile of
//! money" or "a pile of percentages", because in this domain they are not
//! interchangeable and the compiler is the cheapest place to find that out. A
//! [`RetirementSums`] cannot be passed where [`WageCeilings`] is expected; an
//! [`AllocationRatios`] cannot be passed where [`ContributionRates`] is
//! expected. Both mistakes are easy to make in a spreadsheet and impossible to
//! make here.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::cpf::error::CpfError;
use crate::cpf::money::{Rate, Sgd};
use crate::cpf::structure::{Account, AllocationAgeBand, ContributionAgeBand};

// ---------------------------------------------------------------------------
// Contribution rates
// ---------------------------------------------------------------------------

/// The employer's and employee's shares of the CPF contribution, as published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionRates {
    /// Paid by the employer, on top of the wage.
    pub employer: Rate,
    /// Deducted from the employee's wage.
    pub employee: Rate,
}

impl ContributionRates {
    pub const fn new(employer: Rate, employee: Rate) -> Self {
        Self { employer, employee }
    }

    /// The headline total contribution rate (e.g. 37% for a member aged 55 and
    /// below).
    pub fn total(self) -> Rate {
        self.employer + self.employee
    }

    /// Applies these rates to a wage, following **CPF's statutory rounding
    /// rule**.
    ///
    /// # The rule, and why it cannot be simplified
    ///
    /// 1. The **total** contribution is `wage x total_rate`, rounded to the
    ///    nearest dollar (50 cents rounds up).
    /// 2. The **employee's** share is `wage x employee_rate` with the **cents
    ///    dropped**.
    /// 3. The **employer's** share is the **residual**: total − employee.
    ///
    /// Step 3 is the one that catches people. The employer's share is *not*
    /// `wage x employer_rate` rounded — computing it that way is wrong by up to
    /// a dollar, in a way that depends on the cents of the wage and so slips
    /// through any test that uses round numbers. Two different rounding rules
    /// (nearest for the total, down for the employee) applied to two different
    /// products cannot be expected to reconcile, so CPF makes one of them a
    /// residual, and it is the employer's.
    ///
    /// The design of [`crate::cpf::money::Unrounded`] exists to make this rule
    /// expressible without any intermediate float ever coming into being.
    pub fn split(self, wages_subject_to_cpf: Sgd) -> ContributionSplit {
        let total = self.total().of(wages_subject_to_cpf).round_to_nearest_dollar();
        let employee = self.employee.of(wages_subject_to_cpf).drop_cents();
        ContributionSplit {
            total,
            employee,
            // Residual, by statute. Do not "simplify" this to
            // `self.employer.of(wages).round(...)`.
            employer: total - employee,
            wages_subject_to_cpf,
        }
    }
}

impl fmt::Display for ContributionRates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "employer {} + employee {} = {}",
            self.employer,
            self.employee,
            self.total()
        )
    }
}

/// A computed contribution, split per CPF's statutory rounding rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionSplit {
    /// The wages the rates were applied to — *after* ceilings. Carried along so
    /// a breakdown can show its own working.
    pub wages_subject_to_cpf: Sgd,
    /// Total contribution, rounded to the nearest dollar.
    pub total: Sgd,
    /// Employee's share, cents dropped.
    pub employee: Sgd,
    /// Employer's share, the residual.
    pub employer: Sgd,
}

impl ContributionSplit {
    /// Zero contribution on zero wages. Used where a ceiling has completely
    /// exhausted the wages subject to CPF.
    pub const ZERO: ContributionSplit = ContributionSplit {
        wages_subject_to_cpf: Sgd::ZERO,
        total: Sgd::ZERO,
        employee: Sgd::ZERO,
        employer: Sgd::ZERO,
    };
}

/// One published revision of the contribution-rate table: every age band, one
/// effective date, one citation.
///
/// Held whole rather than row-by-row because that is how CPF publishes it, and
/// because a table half from 2024 and half from 2025 should be unrepresentable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContributionSchedule {
    bands: BTreeMap<ContributionAgeBand, ContributionRates>,
}

impl ContributionSchedule {
    pub fn new(bands: BTreeMap<ContributionAgeBand, ContributionRates>) -> Self {
        Self { bands }
    }

    /// The rates for one age band.
    ///
    /// # Errors
    ///
    /// [`CpfError::NotPopulated`] if this revision does not carry that band —
    /// which would mean KOPITIAM has half a table, and should say so rather than
    /// substitute a neighbouring band's rates.
    pub fn band(&self, band: ContributionAgeBand) -> Result<ContributionRates, CpfError> {
        self.bands.get(&band).copied().ok_or(CpfError::NotPopulated {
            dimension: format!("contribution rates for age band '{band}'"),
            reason: "this published revision is not fully transcribed into KOPITIAM",
        })
    }

    pub fn bands(&self) -> impl Iterator<Item = (ContributionAgeBand, ContributionRates)> + '_ {
        self.bands.iter().map(|(k, v)| (*k, *v))
    }
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/// How a total contribution is divided between the Ordinary, Special and
/// MediSave accounts, expressed as ratios of the *total contribution* (not of
/// the wage).
///
/// # A cross-check worth writing down
///
/// CPF publishes these as four-decimal ratios (`0.6217`, `0.1621`, `0.2162`),
/// which look arbitrary. They are not. For a member aged 35 and below, the
/// published contribution is 23% of wages to OA, 6% to SA and 8% to MA, out of a
/// 37% total — and
///
/// ```text
/// 23 / 37 = 0.621621...  ->  0.6217   (OA)
///  6 / 37 = 0.162162...  ->  0.1621   (SA)
///  8 / 37 = 0.216216...  ->  0.2162   (MA)
/// ```
///
/// Every published ratio in the sub-55 table reproduces exactly this way. That
/// is a strong, cheap check on a transcription: if a ratio does not equal a
/// plausible `x/37`, it was typed in wrong. It also explains why the ratios must
/// be re-derived whenever the total rate changes — which is why the senior-worker
/// allocation tables move every January while the sub-55 ones do not.
///
/// Recorded here because working it out took real effort and it is exactly the
/// sort of knowledge that otherwise evaporates (CLAUDE.md: *"Preserve hard-won
/// format knowledge in the code"*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationRatios {
    pub ordinary: Rate,
    pub special: Rate,
    pub medisave: Rate,
}

impl AllocationRatios {
    pub const fn new(ordinary: Rate, special: Rate, medisave: Rate) -> Self {
        Self {
            ordinary,
            special,
            medisave,
        }
    }

    /// Checks that the three ratios partition the contribution exactly.
    ///
    /// They *are* a partition — every cent of the contribution lands in one of
    /// the three accounts. A table summing to 99.99% would quietly lose a
    /// member's money, so this is checked for every built-in band in the test
    /// suite rather than trusted.
    ///
    /// # Errors
    ///
    /// [`CpfError::AllocationDoesNotSumToOne`].
    pub fn validate(&self, band: AllocationAgeBand) -> Result<(), CpfError> {
        let sum = self.ordinary + self.special + self.medisave;
        if sum != Rate::ONE {
            return Err(CpfError::AllocationDoesNotSumToOne {
                band: band.label().to_string(),
                actual: sum,
            });
        }
        Ok(())
    }

    /// Splits a total contribution across the three accounts.
    ///
    /// # Rounding — a flagged assumption, not a cited rule
    ///
    /// The MediSave and Special amounts are computed from the ratios and rounded
    /// to the nearest dollar; the **Ordinary Account takes the residual**, so
    /// that `OA + SA + MA == total` holds exactly and no cent is lost.
    ///
    /// The *residual* structure is forced (a partition must sum to the whole).
    /// **Which** account is the residual, and the order in which the others are
    /// computed, is an assumption this crate makes and has **not** verified
    /// against the CPF Board's own worked examples. It can differ from CPF by at
    /// most one dollar between two accounts of the same member — it never changes
    /// the total. It is recorded honestly here rather than presented as cited.
    ///
    /// Verifying this against CPF's published contribution calculator is tracked
    /// work; see the crate-level "What is not modelled" list.
    pub fn allocate(&self, total_contribution: Sgd) -> Allocation {
        let medisave = self.medisave.of(total_contribution).round_to_nearest_dollar();
        let special = self.special.of(total_contribution).round_to_nearest_dollar();
        Allocation {
            ordinary: total_contribution - medisave - special,
            special,
            medisave,
        }
    }
}

impl fmt::Display for AllocationRatios {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OA {} / SA {} / MA {}",
            self.ordinary, self.special, self.medisave
        )
    }
}

/// A total contribution divided across accounts. Invariant, enforced by
/// construction and asserted in tests: `ordinary + special + medisave` equals
/// the contribution it was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    pub ordinary: Sgd,
    pub special: Sgd,
    pub medisave: Sgd,
}

impl Allocation {
    pub const ZERO: Allocation = Allocation {
        ordinary: Sgd::ZERO,
        special: Sgd::ZERO,
        medisave: Sgd::ZERO,
    };

    pub fn total(&self) -> Sgd {
        self.ordinary + self.special + self.medisave
    }

    /// The allocation as `(account, amount)` pairs, for display and for
    /// emitting into the knowledge graph.
    pub fn by_account(&self) -> [(Account, Sgd); 3] {
        [
            (Account::Ordinary, self.ordinary),
            (Account::Special, self.special),
            (Account::MediSave, self.medisave),
        ]
    }
}

/// One published revision of the allocation table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocationSchedule {
    bands: BTreeMap<AllocationAgeBand, AllocationRatios>,
}

impl AllocationSchedule {
    pub fn new(bands: BTreeMap<AllocationAgeBand, AllocationRatios>) -> Self {
        Self { bands }
    }

    /// The ratios for one age band.
    ///
    /// # Errors
    ///
    /// [`CpfError::NotPopulated`] if KOPITIAM does not hold cited ratios for
    /// that band. This is the expected outcome for every band at 55 and above —
    /// see [`crate::cpf::published`].
    pub fn band(&self, band: AllocationAgeBand) -> Result<AllocationRatios, CpfError> {
        self.bands.get(&band).copied().ok_or(CpfError::NotPopulated {
            dimension: format!("allocation ratios for age band '{band}'"),
            reason: "KOPITIAM holds cited allocation ratios only for members below age 55; \
                     the bands at 55 and above were restructured when the Special Account was \
                     closed for those members in January 2025, and the current ratios were not \
                     known with enough confidence to transcribe",
        })
    }

    pub fn bands(&self) -> impl Iterator<Item = (AllocationAgeBand, AllocationRatios)> + '_ {
        self.bands.iter().map(|(k, v)| (*k, *v))
    }
}

// ---------------------------------------------------------------------------
// Wage ceilings
// ---------------------------------------------------------------------------

/// The two wage ceilings, which **interact** — the whole reason they live in one
/// struct rather than two tables.
///
/// * **Ordinary Wages (OW)** are wages due for a month's work: salary. The
///   Ordinary Wage ceiling caps them *per month*.
/// * **Additional Wages (AW)** are everything else: bonuses, leave pay,
///   commissions. Their ceiling is *annual*, and it is a **residual**.
///
/// # The interaction, which is where the bugs live
///
/// ```text
/// AW ceiling for the year = annual_total_wage_ceiling
///                         − (Ordinary Wages SUBJECT TO CPF for the year)
/// ```
///
/// Two traps in that one line:
///
/// 1. It subtracts the Ordinary Wages **subject to CPF** — i.e. *already capped*
///    at the OW ceiling, month by month — not the wages actually paid. A member
///    earning $12,000/month has $144,000 of Ordinary Wages but, in 2025, only
///    `12 x $7,400 = $88,800` subject to CPF, leaving an AW ceiling of
///    `$102,000 − $88,800 = $13,200`, not zero. Subtracting the *paid* wages
///    gives a negative and silently denies them CPF on their entire bonus.
///
/// 2. Because the monthly OW ceiling itself changes over time — and has changed
///    **mid-year** (it went from $6,000 to $6,300 on 1 September 2023) — the
///    annual figure cannot be `12 x ceiling`. It must be accumulated month by
///    month against the ceiling in force for *each* month. That is why
///    [`Self::additional_wage_ceiling`] takes an accumulated total rather than a
///    monthly wage.
///
/// The residual can go negative for a high earner; it is clamped to zero,
/// because "no Additional Wages attract CPF" is not the same as "CPF owes you
/// money".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WageCeilings {
    /// Maximum Ordinary Wages per month that attract CPF.
    pub ordinary_wage_monthly: Sgd,
    /// Maximum total wages (Ordinary + Additional) per year that attract CPF.
    pub annual_total_wage: Sgd,
}

impl WageCeilings {
    pub const fn new(ordinary_wage_monthly: Sgd, annual_total_wage: Sgd) -> Self {
        Self {
            ordinary_wage_monthly,
            annual_total_wage,
        }
    }

    /// Ordinary Wages subject to CPF for one month: the wage, capped.
    pub fn ordinary_wages_subject_to_cpf(&self, ordinary_wages: Sgd) -> Sgd {
        ordinary_wages.min(self.ordinary_wage_monthly)
    }

    /// The Additional Wage ceiling for the year, given the Ordinary Wages
    /// **already subject to CPF** so far this year.
    ///
    /// Read the type docs above before calling this. The argument is not the
    /// wages paid; it is the wages *capped*, accumulated across the months of the
    /// year at whichever OW ceiling was in force for each.
    pub fn additional_wage_ceiling(&self, ordinary_wages_subject_to_cpf_ytd: Sgd) -> Sgd {
        (self.annual_total_wage - ordinary_wages_subject_to_cpf_ytd).clamp_non_negative()
    }
}

// ---------------------------------------------------------------------------
// Retirement sums
// ---------------------------------------------------------------------------

/// The three retirement sums for **one cohort**.
///
/// # These are indexed by cohort, not by "today"
///
/// A member's retirement sum is the one published for the year in which they
/// **turn 55**, and it stays with them for life. It is *not* the sum published
/// in the year you happen to be asking the question.
///
/// This is a different temporal axis from every other table in this crate, and
/// it is a real trap: looking up "the 2026 Full Retirement Sum" for a member who
/// turned 55 in 2024 gives a number $15,000 too high, and would tell them they
/// are short of a target they in fact already met.
///
/// KOPITIAM reuses [`crate::cpf::temporal::PolicyTable`] for these, keyed on the
/// date range of the **cohort year**, and the lookup takes the member's *55th
/// birthday* — see [`crate::cpf::query::CpfPolicy::retirement_sums_for_cohort`],
/// which computes it for you so the mistake is hard to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetirementSums {
    /// Basic Retirement Sum — the payout target for a member who owns a property
    /// they can live in for life and pledges it.
    pub basic: Sgd,
    /// Full Retirement Sum. Defined by policy as **exactly 2 x the Basic**.
    pub full: Sgd,
    /// Enhanced Retirement Sum — the maximum a member may top up to.
    ///
    /// Defined as a multiple of the Basic sum, and **that multiple changed**:
    /// 3 x through 2024, then 4 x from 2025 (announced at Budget 2024). A
    /// hardcoded `3 * basic` would have been quietly wrong from 1 January 2025
    /// by over a hundred thousand dollars.
    pub enhanced: Sgd,
}

impl RetirementSums {
    pub const fn new(basic: Sgd, full: Sgd, enhanced: Sgd) -> Self {
        Self {
            basic,
            full,
            enhanced,
        }
    }
}

// ---------------------------------------------------------------------------
// Interest
// ---------------------------------------------------------------------------

/// The **statutory floor** interest rates, and the extra-interest tiers.
///
/// # What this is, and emphatically what it is not
///
/// CPF's Ordinary Account rate is pegged to a formula based on local bank
/// interest rates, and the Special/MediSave/Retirement rate to the 10-year
/// Singapore Government Securities yield plus 1%. Both are subject to a
/// **legislated floor**, and both are *declared quarterly*.
///
/// KOPITIAM holds only the **floors**, because they are stable, legislated, and
/// the author is confident of them. The **declared** rate for any given quarter
/// is not modelled and is not guessed. A declared rate can exceed the floor.
///
/// So: this type tells you the *minimum* interest CPF will pay. It does not tell
/// you what CPF paid last quarter. Do not present it as if it did.
///
/// No interest *computation* is provided. Interest accrues monthly on the lowest
/// balance of the month, and the extra-interest tiers are applied across accounts
/// in a prescribed order — a computation this scaffold does not attempt, because
/// a half-right interest engine is worse than none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterestFloors {
    /// Floor rate on the Ordinary Account.
    pub ordinary: Rate,
    /// Floor rate on the Special, MediSave and Retirement Accounts.
    pub special_medisave_retirement: Rate,
    /// Extra interest on the first tier of combined balances, of which at most
    /// [`Self::extra_interest_ordinary_cap`] may come from the Ordinary Account.
    pub extra_interest_first_tier: Sgd,
    pub extra_interest_first_tier_rate: Rate,
    pub extra_interest_ordinary_cap: Sgd,
    /// A *further* extra interest on the first tier of combined balances, for
    /// members aged 55 and above.
    pub extra_interest_second_tier: Sgd,
    pub extra_interest_second_tier_rate: Rate,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- The statutory rounding rule ----------------------------------------

    /// A wage with cents, chosen so that the employer's share computed as a
    /// residual differs from the employer's share computed independently. This
    /// is the bug the residual rule exists to prevent, demonstrated.
    #[test]
    fn employer_share_is_the_residual_not_an_independent_rounding() {
        let rates = ContributionRates::new(
            Rate::from_percent_tenths(170), // 17%
            Rate::from_percent_tenths(200), // 20%
        );
        // $3,333.33 -> total 37% = $1,233.33321 -> nearest dollar = $1,233
        //           -> employee 20% = $666.666  -> drop cents     = $666
        //           -> employer     = residual                    = $567
        let wage = Sgd::parse("3333.33").unwrap();
        let split = rates.split(wage);

        assert_eq!(split.total, Sgd::from_dollars(1_233));
        assert_eq!(split.employee, Sgd::from_dollars(666));
        assert_eq!(split.employer, Sgd::from_dollars(567));

        // The residual is what makes the parts reconcile with the whole.
        assert_eq!(split.employer + split.employee, split.total);

        // Had we computed the employer share independently, we would have got
        // 17% of $3,333.33 = $566.66661 -> $567. Same here, but the invariant
        // below is what actually guarantees correctness, not luck.
    }

    /// The invariant that must never break, swept across a range of wages with
    /// awkward cents.
    #[test]
    fn employer_plus_employee_always_equals_total() {
        let rates = ContributionRates::new(
            Rate::from_percent_tenths(155), // 15.5%
            Rate::from_percent_tenths(170), // 17%
        );
        for cents in 75_000..75_200 {
            let split = rates.split(Sgd::from_cents(cents));
            assert_eq!(
                split.employer + split.employee,
                split.total,
                "reconciliation failed at {} cents",
                cents
            );
        }
    }

    #[test]
    fn total_rate_is_the_sum_of_the_shares() {
        let rates = ContributionRates::new(Rate::from_percent_tenths(170), Rate::from_percent_tenths(200));
        assert_eq!(rates.total(), Rate::from_percent_tenths(370));
        assert_eq!(rates.to_string(), "employer 17.00% + employee 20.00% = 37.00%");
    }

    // -- Allocation ---------------------------------------------------------

    /// The x/37 cross-check from the type docs, verified as arithmetic.
    #[test]
    fn sub_55_allocation_ratios_are_the_published_percentages_over_37() {
        // 23/37, 6/37, 8/37 rounded to 4dp.
        let ratios = AllocationRatios::new(
            Rate::from_basis_points(6217),
            Rate::from_basis_points(1621),
            Rate::from_basis_points(2162),
        );
        ratios.validate(AllocationAgeBand::UpTo35).unwrap();

        // 23 / 37 = 0.6216216...; to 4dp that is 0.6217 (CPF rounds up here so
        // the three ratios still sum to exactly 1).
        assert_eq!(23 * 10_000 / 37, 6216);
        assert_eq!(6 * 10_000 / 37, 1621);
        assert_eq!(8 * 10_000 / 37, 2162);
    }

    #[test]
    fn allocation_ratios_must_partition_the_contribution() {
        let bad = AllocationRatios::new(
            Rate::from_basis_points(6000),
            Rate::from_basis_points(1600),
            Rate::from_basis_points(2162),
        );
        let err = bad.validate(AllocationAgeBand::UpTo35).unwrap_err();
        assert!(matches!(err, CpfError::AllocationDoesNotSumToOne { .. }));
    }

    /// No cent may be lost or invented in allocation. Swept across a range of
    /// contributions, including ones where both roundings go the same way.
    #[test]
    fn allocation_conserves_every_cent() {
        let ratios = AllocationRatios::new(
            Rate::from_basis_points(6217),
            Rate::from_basis_points(1621),
            Rate::from_basis_points(2162),
        );
        for dollars in 0..2_000 {
            let total = Sgd::from_dollars(dollars);
            let allocated = ratios.allocate(total);
            assert_eq!(
                allocated.total(),
                total,
                "allocation of {total} lost or invented money"
            );
        }
    }

    // -- Wage ceiling interaction -------------------------------------------

    /// The trap: the Additional Wage ceiling is a residual against Ordinary
    /// Wages **already capped**, not against wages paid.
    #[test]
    fn additional_wage_ceiling_is_a_residual_against_capped_ordinary_wages() {
        // 2025 figures.
        let ceilings = WageCeilings::new(Sgd::from_dollars(7_400), Sgd::from_dollars(102_000));

        // A member paid $12,000/month. Ordinary Wages paid: $144,000/year.
        // Ordinary Wages *subject to CPF*: 12 x $7,400 = $88,800.
        let monthly_ow = Sgd::from_dollars(12_000);
        let capped_monthly = ceilings.ordinary_wages_subject_to_cpf(monthly_ow);
        assert_eq!(capped_monthly, Sgd::from_dollars(7_400));

        let ow_ytd: Sgd = std::iter::repeat_n(capped_monthly, 12).sum();
        assert_eq!(ow_ytd, Sgd::from_dollars(88_800));

        // The AW ceiling is therefore $13,200 — NOT zero, which is what you get
        // if you wrongly subtract the $144,000 actually paid.
        assert_eq!(
            ceilings.additional_wage_ceiling(ow_ytd),
            Sgd::from_dollars(13_200)
        );

        let wrong = ceilings.additional_wage_ceiling(Sgd::from_dollars(144_000));
        assert_eq!(wrong, Sgd::ZERO, "the wrong input gives the wrong answer — hence the docs");
    }

    /// A very high earner exhausts the annual ceiling entirely. The residual is
    /// clamped, never negative.
    #[test]
    fn a_high_earner_has_no_additional_wage_headroom() {
        let ceilings = WageCeilings::new(Sgd::from_dollars(7_400), Sgd::from_dollars(102_000));
        // 12 x $7,400 = $88,800 < $102,000, so even the highest earner retains
        // $13,200 of AW headroom under the 2025 ceilings. Contrive an exhausted
        // case to prove the clamp.
        let exhausted = ceilings.additional_wage_ceiling(Sgd::from_dollars(150_000));
        assert_eq!(exhausted, Sgd::ZERO);
        assert!(!exhausted.is_negative());
    }

    /// The OW ceiling changed **mid-year** in 2023 ($6,000 -> $6,300 on
    /// 1 September). The annual accumulation therefore cannot be `12 x ceiling`.
    #[test]
    fn a_mid_year_ceiling_change_makes_the_annual_total_non_uniform() {
        let before = WageCeilings::new(Sgd::from_dollars(6_000), Sgd::from_dollars(102_000));
        let after = WageCeilings::new(Sgd::from_dollars(6_300), Sgd::from_dollars(102_000));
        let high_wage = Sgd::from_dollars(20_000);

        // Jan-Aug at $6,000; Sep-Dec at $6,300.
        let jan_to_aug: Sgd = std::iter::repeat_n(before.ordinary_wages_subject_to_cpf(high_wage), 8).sum();
        let sep_to_dec: Sgd = std::iter::repeat_n(after.ordinary_wages_subject_to_cpf(high_wage), 4).sum();
        let ow_ytd = jan_to_aug + sep_to_dec;

        assert_eq!(ow_ytd, Sgd::from_dollars(48_000) + Sgd::from_dollars(25_200));
        assert_eq!(ow_ytd, Sgd::from_dollars(73_200));

        // Naively assuming a uniform ceiling gets a different — wrong — answer
        // either way you pick.
        assert_ne!(ow_ytd, Sgd::from_dollars(6_000 * 12));
        assert_ne!(ow_ytd, Sgd::from_dollars(6_300 * 12));

        assert_eq!(after.additional_wage_ceiling(ow_ytd), Sgd::from_dollars(28_800));
    }

    #[test]
    fn schedule_reports_an_unpopulated_band_honestly() {
        let schedule = ContributionSchedule::new(BTreeMap::new());
        let err = schedule.band(ContributionAgeBand::UpTo55).unwrap_err();
        assert!(matches!(err, CpfError::NotPopulated { .. }));

        let alloc = AllocationSchedule::new(BTreeMap::new());
        let err = alloc.band(AllocationAgeBand::Above55To60).unwrap_err();
        assert!(matches!(err, CpfError::NotPopulated { .. }));
        assert!(err.to_string().contains("Special Account"));
    }
}
