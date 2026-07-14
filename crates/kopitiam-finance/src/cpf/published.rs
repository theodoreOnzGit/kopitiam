//! The cited data. **Read the confidence statement below before trusting a
//! number out of this file.**
//!
//! # Provenance of everything here
//!
//! Every value in this module carries [`SourceKind::Transcribed`] (or
//! [`SourceKind::Derived`] where it follows by definition from a transcribed
//! one). That means:
//!
//! > These figures were written down from knowledge of the CPF Board's published
//! > tables. They were **not** machine-extracted from the source documents, and
//! > they have **not** been verified against the primary sources by this
//! > codebase. No network was available when they were transcribed.
//!
//! They are offered so the engine is useful today, and the label is on them so
//! that nobody can mistake them for something stronger. The endpoint is
//! [`SourceKind::ExtractedFromDocument`], via [`crate::cpf::document`]: feed the
//! CPF Board's own PDFs through KOPITIAM's Document Engine and let the
//! deterministic path replace the transcription. Until then, **verify before you
//! rely.**
//!
//! # What is populated
//!
//! | Table | Coverage |
//! |---|---|
//! | Contribution rates | 2024, 2025, 2026 revisions. Singapore Citizens and SPRs from the 3rd year of PR status, private sector, **total wages of $750/month and above** only. |
//! | Allocation ratios | Ages **below 55 only** (four bands), 2024 onward. |
//! | Wage ceilings | 2023 (both halves — the ceiling moved mid-year), 2024, 2025, 2026. |
//! | Retirement sums | Cohorts turning 55 in **2023, 2024, 2025, 2026**. |
//! | Interest | **Statutory floors only**, 2024 onward. Not the declared quarterly rates. |
//!
//! # What is deliberately NOT populated, and why
//!
//! Each of these returns a loud [`crate::cpf::error::CpfError::NotPopulated`] or
//! [`crate::cpf::error::CpfError::NoRuleInEffect`], never a plausible-looking
//! guess.
//!
//! * **Allocation ratios for members aged 55 and above.** The Special Account was
//!   *closed* for these members in January 2025 and their savings restructured
//!   into the Retirement and Ordinary Accounts. The current ratios, and the
//!   post-55 account structure itself, were not known with enough confidence to
//!   transcribe. This is the single largest gap.
//! * **Graduated rates for total wages below $750/month.** The employee's share
//!   is phased in below $750 by a formula that depends on both the wage and the
//!   age band. Encoding a wrong formula here would under- or over-deduct from the
//!   lowest-paid members. Not attempted.
//! * **Permanent Resident year-1 and year-2 rates.** Three distinct rate tables
//!   (graduated/graduated, full/graduated, full/full by joint election), each with
//!   its own allocation table. Not attempted.
//! * **Declared (as opposed to floor) interest rates**, and any interest
//!   *computation*. Interest accrues on the lowest monthly balance with extra
//!   interest applied across accounts in a prescribed order; a half-right interest
//!   engine is worse than none.
//! * **Retirement sums for cohorts turning 55 from 2027 onward.** A schedule has
//!   been announced; it is not transcribed here. A query for a 2027 cohort
//!   correctly fails rather than extrapolating the 3.5%/year trend — see the test
//!   for exactly that.
//! * **Public-sector pensionable employees, self-employed persons, and the
//!   Additional MediSave Contribution / MediSave contribution rates for the
//!   self-employed.** Entirely different schemes.
//! * **Housing withdrawal limits** (Valuation Limit, Withdrawal Limit), CPF LIFE,
//!   Workfare, top-up schemes, the Basic Healthcare Sum, and CPF transfers.
//!
//! # Confidence, stated plainly
//!
//! Highest confidence: the age-band structure, the wage ceilings, the sub-55
//! allocation ratios (which cross-check exactly against `x/37` — see
//! [`AllocationRatios`]), the 37% total for members 55 and below, the retirement
//! sum definitions (FRS = 2 x BRS; ERS = 3 x BRS through 2024, 4 x BRS from 2025).
//!
//! Lower confidence, and flagged in the citation `note` of each: the precise
//! employer/employee *split* within the senior-worker bands (55-70) for each of
//! 2024, 2025 and 2026. The totals are believed right; the split between employer
//! and employee within them is the part most worth re-verifying first.

use std::collections::BTreeMap;

use crate::cpf::citation::{Citation, SourceKind};
use crate::cpf::date::{Date, DateRange};
use crate::cpf::money::{Rate, Sgd};
use crate::cpf::rates::{
    AllocationRatios, AllocationSchedule, ContributionRates, ContributionSchedule, InterestFloors,
    RetirementSums, WageCeilings,
};
use crate::cpf::structure::{AllocationAgeBand, ContributionAgeBand};
use crate::cpf::temporal::{Dated, PolicyTable};

/// Convenience: a date that is known-good at authoring time.
fn d(year: i32, month: u8, day: u8) -> Date {
    Date::new(year, month, day).expect("published policy dates are valid by inspection")
}

fn range(from: (i32, u8, u8), until: (i32, u8, u8)) -> DateRange {
    DateRange::between(d(from.0, from.1, from.2), d(until.0, until.1, until.2))
        .expect("published policy ranges are non-empty by inspection")
}

fn open(from: (i32, u8, u8)) -> DateRange {
    DateRange::from(d(from.0, from.1, from.2))
}

const RATES_URL: &str = "https://www.cpf.gov.sg/employer/employer-obligations/how-much-cpf-contributions-to-pay";
const CEILING_URL: &str = "https://www.cpf.gov.sg/member/growing-your-savings/cpf-contributions/what-are-cpf-contribution-caps";
const RETIREMENT_URL: &str = "https://www.cpf.gov.sg/member/retirement-income/retirement-withdrawals/how-much-retirement-income-do-you-need";

/// The standard caveat carried by every transcribed contribution rate.
const SENIOR_SPLIT_CAVEAT: &str = "Transcribed from memory of the published table, not extracted \
     from the source document. The total contribution rate is held with high confidence; the \
     employer/employee split within the 55-70 bands is the figure most worth re-verifying against \
     the primary source first.";

fn rates_citation(effective_year: i32) -> Citation {
    Citation::transcribed_from_cpf_board(
        "CPF contribution rates (private sector employees, and public sector \
         non-pensionable employees)",
        format!(
            "Table of contribution rates effective 1 January {effective_year}; \
             Singapore Citizens and SPRs from the 3rd year of PR status; \
             total wages of $750/month and above"
        ),
    )
    .with_published(d(effective_year, 1, 1))
    .with_url(RATES_URL)
    .with_note(SENIOR_SPLIT_CAVEAT)
}

// ---------------------------------------------------------------------------
// Contribution rates
// ---------------------------------------------------------------------------

/// Builds one revision of the contribution schedule.
///
/// Rates are given as tenths of a percent — `170` is 17.0%, `155` is 15.5% —
/// which is exactly the precision CPF publishes at, so the literals below read
/// like the source table rather than like a units conversion.
fn schedule(rows: [(ContributionAgeBand, i32, i32); 5]) -> ContributionSchedule {
    let mut bands = BTreeMap::new();
    for (band, employer_tenths, employee_tenths) in rows {
        bands.insert(
            band,
            ContributionRates::new(
                Rate::from_percent_tenths(employer_tenths),
                Rate::from_percent_tenths(employee_tenths),
            ),
        );
    }
    ContributionSchedule::new(bands)
}

/// Contribution rates for Singapore Citizens and SPRs from the 3rd year of PR
/// status, private sector, total wages of $750/month and above.
///
/// Three revisions, one per January. The senior-worker bands (55-70) step up
/// every year under the phased increases recommended by the Tripartite Workgroup
/// on Older Workers; the 55-and-below band has sat at 37% throughout.
///
/// Note what changes and what does not: **only** the senior bands move. Modelling
/// this as "the CPF rate" — a single number — would be wrong for four of the five
/// bands, every January.
pub fn contribution_rates_citizen_and_pr3plus() -> PolicyTable<ContributionSchedule> {
    use ContributionAgeBand::*;

    PolicyTable::new(
        "contribution rates (citizen / SPR 3rd year+, wages >= $750/month)",
        vec![
            // (employer, employee) in tenths of a percent.
            Dated::new(
                schedule([
                    (UpTo55, 170, 200),      // 17.0 + 20.0 = 37.0%
                    (Above55To60, 150, 160), // 15.0 + 16.0 = 31.0%
                    (Above60To65, 115, 105), // 11.5 + 10.5 = 22.0%
                    (Above65To70, 90, 75),   //  9.0 +  7.5 = 16.5%
                    (Above70, 75, 50),       //  7.5 +  5.0 = 12.5%
                ]),
                range((2024, 1, 1), (2025, 1, 1)),
                rates_citation(2024),
            ),
            Dated::new(
                schedule([
                    (UpTo55, 170, 200),      // 17.0 + 20.0 = 37.0%
                    (Above55To60, 155, 170), // 15.5 + 17.0 = 32.5%
                    (Above60To65, 120, 115), // 12.0 + 11.5 = 23.5%
                    (Above65To70, 90, 75),   //  9.0 +  7.5 = 16.5%  (target reached in 2024)
                    (Above70, 75, 50),       //  7.5 +  5.0 = 12.5%  (unchanged since 2016)
                ]),
                range((2025, 1, 1), (2026, 1, 1)),
                rates_citation(2025),
            ),
            Dated::new(
                schedule([
                    (UpTo55, 170, 200),      // 17.0 + 20.0 = 37.0%
                    (Above55To60, 160, 180), // 16.0 + 18.0 = 34.0%
                    (Above60To65, 125, 125), // 12.5 + 12.5 = 25.0%
                    (Above65To70, 90, 75),   //  9.0 +  7.5 = 16.5%
                    (Above70, 75, 50),       //  7.5 +  5.0 = 12.5%
                ]),
                // Open-ended: in force until superseded. This is NOT a claim that
                // it holds forever — a query for 2028 will happily return this,
                // and that is a limitation of an open-ended range that the
                // curator must manage by closing it when the next revision lands.
                open((2026, 1, 1)),
                rates_citation(2026),
            ),
        ],
    )
}

// ---------------------------------------------------------------------------
// Allocation ratios
// ---------------------------------------------------------------------------

/// Allocation ratios for members **below age 55**, as ratios of the total
/// contribution.
///
/// The four bands below are the *complete* sub-55 table and each sums to exactly
/// 1.0000. The three bands at 55 and above are **absent**, not zeroed — see the
/// module docs. `AllocationSchedule::band` returns
/// [`crate::cpf::error::CpfError::NotPopulated`] for them, with the reason.
///
/// Each ratio is annotated with the percentage-of-wage it derives from, out of the
/// 37% total. That derivation (`23/37 = 0.6217`) is the transcription check
/// described on [`AllocationRatios`].
pub fn allocation_ratios_citizen_and_pr3plus() -> PolicyTable<AllocationSchedule> {
    use AllocationAgeBand::*;

    let mut bands = BTreeMap::new();
    //                                    OA      SA      MA      (of a 37% total)
    // 35 and below:        OA 23%, SA 6%,    MA 8%
    bands.insert(UpTo35, ratios(6217, 1621, 2162));
    // Above 35 to 45:      OA 21%, SA 7%,    MA 9%
    bands.insert(Above35To45, ratios(5677, 1891, 2432));
    // Above 45 to 50:      OA 19%, SA 8%,    MA 10%
    bands.insert(Above45To50, ratios(5136, 2162, 2702));
    // Above 50 to 55:      OA 15%, SA 11.5%, MA 10.5%
    bands.insert(Above50To55, ratios(4055, 3108, 2837));
    // Above 55 onwards:    DELIBERATELY ABSENT. See the module docs.

    PolicyTable::new(
        "allocation ratios (citizen / SPR 3rd year+, below age 55)",
        vec![Dated::new(
            AllocationSchedule::new(bands),
            open((2024, 1, 1)),
            Citation::transcribed_from_cpf_board(
                "CPF allocation rates (private sector employees, and public sector \
                 non-pensionable employees)",
                "Allocation table, bands '35 years and below' through 'Above 50 to 55 years'",
            )
            .with_url(RATES_URL)
            .with_note(
                "Ratios of the TOTAL contribution, not of wages. Each cross-checks exactly \
                 against the published percentage-of-wage over the 37% total rate \
                 (e.g. 23/37 = 0.6217), and each band sums to 1.0000 — both are asserted in \
                 the test suite. The effective range starts in 2024 because that is the \
                 window this crate is confident about; the sub-55 ratios are in fact \
                 long-standing, but KOPITIAM does not claim what it has not checked.",
            ),
        )],
    )
}

fn ratios(oa_bp: i32, sa_bp: i32, ma_bp: i32) -> AllocationRatios {
    AllocationRatios::new(
        Rate::from_basis_points(oa_bp),
        Rate::from_basis_points(sa_bp),
        Rate::from_basis_points(ma_bp),
    )
}

// ---------------------------------------------------------------------------
// Wage ceilings
// ---------------------------------------------------------------------------

/// The Ordinary Wage ceiling and the annual total wage ceiling.
///
/// # This table is the argument for the whole crate
///
/// Five revisions in four years, **one of them mid-year**: the Ordinary Wage
/// ceiling went from $6,000 to $6,300 on **1 September 2023**, then to $6,800,
/// $7,400 and $8,000 on successive 1 Januaries. A `const OW_CEILING` written in
/// August 2023 would have been wrong a month later and stayed wrong for three
/// years.
///
/// The annual total wage ceiling has sat at $102,000 throughout — which is
/// exactly why it must *also* be dated. A number that has not changed yet is not
/// a number that will not change, and there is no way to tell the two apart from
/// the value alone.
pub fn wage_ceilings() -> PolicyTable<WageCeilings> {
    const ANNUAL: Sgd = Sgd::from_dollars(102_000);

    let cite = |what: &str, note: &str| {
        Citation::transcribed_from_cpf_board("CPF contribution caps (wage ceilings)", what)
            .with_url(CEILING_URL)
            .with_note(note)
    };

    PolicyTable::new(
        "wage ceilings",
        vec![
            Dated::new(
                WageCeilings::new(Sgd::from_dollars(6_000), ANNUAL),
                range((2023, 1, 1), (2023, 9, 1)),
                cite(
                    "Ordinary Wage ceiling, 1 January 2023 to 31 August 2023",
                    "The $6,000 ceiling had stood for years before this; KOPITIAM claims only \
                     the window it is confident about.",
                ),
            ),
            Dated::new(
                WageCeilings::new(Sgd::from_dollars(6_300), ANNUAL),
                range((2023, 9, 1), (2024, 1, 1)),
                cite(
                    "Ordinary Wage ceiling, 1 September 2023 to 31 December 2023",
                    "A MID-YEAR change. Any annual computation across 2023 must accumulate the \
                     Ordinary Wages subject to CPF month by month; 12 x a single ceiling is \
                     wrong for 2023 whichever ceiling you pick.",
                ),
            ),
            Dated::new(
                WageCeilings::new(Sgd::from_dollars(6_800), ANNUAL),
                range((2024, 1, 1), (2025, 1, 1)),
                cite("Ordinary Wage ceiling, calendar year 2024", "Step 2 of 4 in the announced schedule."),
            ),
            Dated::new(
                WageCeilings::new(Sgd::from_dollars(7_400), ANNUAL),
                range((2025, 1, 1), (2026, 1, 1)),
                cite("Ordinary Wage ceiling, calendar year 2025", "Step 3 of 4 in the announced schedule."),
            ),
            Dated::new(
                WageCeilings::new(Sgd::from_dollars(8_000), ANNUAL),
                open((2026, 1, 1)),
                cite(
                    "Ordinary Wage ceiling, from 1 January 2026",
                    "Final step of the announced 4-step schedule. Open-ended because no \
                     successor has been announced — NOT because it is permanent.",
                ),
            ),
        ],
    )
}

// ---------------------------------------------------------------------------
// Retirement sums
// ---------------------------------------------------------------------------

/// Retirement sums, indexed by the calendar year in which the member **turns 55**.
///
/// # The date range here means something different
///
/// Everywhere else in this crate a [`DateRange`] means "the period during which
/// this rule was in force". Here it means "**the cohort**": the range of 55th
/// birthdays to which this set of sums applies, for life.
///
/// The machinery is the same; the key is not. Look these up with the member's
/// 55th birthday, never with today's date. `CpfPolicy::retirement_sums_for_cohort`
/// computes the birthday from a date of birth so the mistake is hard to make.
///
/// # The relationships are definitional, and one of them changed
///
/// * `FRS = 2 x BRS` — has held throughout.
/// * `ERS = 3 x BRS` through the 2024 cohort; **`ERS = 4 x BRS` from the 2025
///   cohort** (announced at Budget 2024). A hardcoded `3 * basic` would have been
///   wrong by $106,500 for a member turning 55 in 2025 — and would have told them
///   the ceiling on their voluntary top-up was a third lower than it is.
///
/// The `full` and `enhanced` figures are therefore recorded as
/// [`SourceKind::Derived`], with the multiple stated, so the relationship cannot
/// silently drift from the Basic sum it is defined against.
///
/// **Cohorts from 2027 onward are absent.** A schedule has been announced; it is
/// not transcribed. A query for the 2027 cohort fails loudly rather than
/// extrapolating the trend.
pub fn retirement_sums() -> PolicyTable<RetirementSums> {
    // (cohort year, BRS in whole dollars, ERS multiple of BRS)
    let cohorts = [
        (2023, 99_400, 3),
        (2024, 102_900, 3),
        (2025, 106_500, 4), // ERS multiple raised from 3x to 4x for this cohort.
        (2026, 110_200, 4),
        // 2027 onward: announced, not transcribed. Deliberately absent.
    ];

    let entries = cohorts
        .into_iter()
        .map(|(year, brs_dollars, ers_multiple)| {
            let basic = Sgd::from_dollars(brs_dollars);
            let full = Sgd::from_dollars(brs_dollars * 2);
            let enhanced = Sgd::from_dollars(brs_dollars * ers_multiple);

            let citation = Citation::transcribed_from_cpf_board(
                "CPF retirement sums",
                format!("Basic / Full / Enhanced Retirement Sum for members turning 55 in {year}"),
            )
            .with_url(RETIREMENT_URL)
            .with_note(format!(
                "Basic Retirement Sum transcribed (${brs_dollars} for the {year} cohort). \
                 Full Retirement Sum derived as 2 x Basic. Enhanced Retirement Sum derived as \
                 {ers_multiple} x Basic — the multiple was raised from 3x to 4x with effect from \
                 the 2025 cohort. Applies to the member for life; keyed on the 55th birthday, \
                 not on the date of the query."
            ));

            Dated::new(
                RetirementSums::new(basic, full, enhanced),
                range((year, 1, 1), (year + 1, 1, 1)),
                citation,
            )
        })
        .collect();

    PolicyTable::new("retirement sums (by cohort turning 55)", entries)
}

// ---------------------------------------------------------------------------
// Interest floors
// ---------------------------------------------------------------------------

/// The **statutory floor** interest rates and the extra-interest tiers.
///
/// Not the declared quarterly rates — see [`InterestFloors`], which explains at
/// length why that distinction matters and why KOPITIAM does not guess at the
/// declared rate.
pub fn interest_floors() -> PolicyTable<InterestFloors> {
    PolicyTable::new(
        "interest floors and extra-interest tiers",
        vec![Dated::new(
            InterestFloors {
                ordinary: Rate::from_percent_tenths(25), // 2.5%
                special_medisave_retirement: Rate::from_percent_tenths(40), // 4.0%
                extra_interest_first_tier: Sgd::from_dollars(60_000),
                extra_interest_first_tier_rate: Rate::from_percent_tenths(10), // +1%
                extra_interest_ordinary_cap: Sgd::from_dollars(20_000),
                extra_interest_second_tier: Sgd::from_dollars(30_000),
                extra_interest_second_tier_rate: Rate::from_percent_tenths(10), // +1% more, age 55+
            },
            open((2024, 1, 1)),
            Citation {
                publisher: "Central Provident Fund Board".to_string(),
                document: "CPF interest rates".to_string(),
                locator: "Statutory floor rates and extra interest".to_string(),
                published: None,
                url: Some("https://www.cpf.gov.sg/member/growing-your-savings/earning-higher-returns/earning-attractive-interest".to_string()),
                source_kind: SourceKind::Transcribed,
                note: Some(
                    "FLOORS ONLY. The Ordinary Account rate is pegged to a bank-rate formula and \
                     the Special/MediSave/Retirement rate to 10-year SGS + 1%, both subject to \
                     these floors and DECLARED QUARTERLY. The declared rate can exceed the floor \
                     and is not modelled — do not present these as the rate CPF actually paid. \
                     Extra interest: +1% on the first $60,000 of combined balances (of which at \
                     most $20,000 may come from the Ordinary Account), and a further +1% on the \
                     first $30,000 for members aged 55 and above. No interest COMPUTATION is \
                     provided. The effective range starts in 2024 because that is the window this \
                     crate is confident about, not because the floors began then."
                        .to_string(),
                ),
            },
        )],
    )
}
