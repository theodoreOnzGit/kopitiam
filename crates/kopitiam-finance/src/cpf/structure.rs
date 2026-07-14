//! The shape of the scheme: accounts, age bands, wage bands, residency.
//!
//! These are the *dimensions* CPF policy is indexed by. They are modelled as
//! enums rather than integers because every one of them is a closed set with
//! sharp, load-bearing boundaries, and because an exhaustive `match` is how you
//! find out at compile time that a new band has appeared.
//!
//! Note carefully that **the contribution age bands and the allocation age
//! bands are different**. Contributions step at 55/60/65/70; allocation steps at
//! 35/45/50/55/60/65. Reusing one band enum for both — the obvious
//! simplification — would silently give a 47-year-old the wrong Ordinary
//! Account split. They are two enums for that reason and must stay two enums.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::cpf::date::Date;
use crate::cpf::error::CpfError;
use crate::cpf::money::Sgd;

// ---------------------------------------------------------------------------
// Accounts
// ---------------------------------------------------------------------------

/// A CPF account.
///
/// Contributions are split across Ordinary, Special and MediSave while a member
/// is below 55. At 55 a Retirement Account is created, funded from the Special
/// and Ordinary Accounts up to the member's chosen retirement sum.
///
/// # The Special Account is not eternal
///
/// From January 2025 the Special Account was **closed for members aged 55 and
/// above** (announced at Budget 2024); their savings moved to the Retirement
/// Account up to the Full Retirement Sum, with the balance going to the
/// Ordinary Account. This is precisely the kind of change that breaks code
/// which treats the account structure as a constant of nature — the *set of
/// accounts a member has* is itself time- and age-dependent.
///
/// KOPITIAM does **not** currently model the post-55 account structure (see
/// [`crate::cpf::published`]); the enum names the accounts so the shape is
/// right, and queries for 55+ return [`CpfError::NotPopulated`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Account {
    /// Ordinary Account — housing, insurance, investment, education.
    Ordinary,
    /// Special Account — retirement, retirement-related investment.
    Special,
    /// MediSave Account — healthcare and approved medical insurance.
    MediSave,
    /// Retirement Account — created at 55 from Special and Ordinary savings.
    Retirement,
}

impl Account {
    /// The three accounts that receive contributions before age 55.
    pub const CONTRIBUTORY_BELOW_55: [Account; 3] =
        [Account::Ordinary, Account::Special, Account::MediSave];

    /// Conventional abbreviation, as used throughout CPF's own documents.
    pub fn abbreviation(self) -> &'static str {
        match self {
            Account::Ordinary => "OA",
            Account::Special => "SA",
            Account::MediSave => "MA",
            Account::Retirement => "RA",
        }
    }
}

impl fmt::Display for Account {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Account::Ordinary => "Ordinary Account",
            Account::Special => "Special Account",
            Account::MediSave => "MediSave Account",
            Account::Retirement => "Retirement Account",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// Age
// ---------------------------------------------------------------------------

/// A member's age in completed years.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Age(u32);

impl Age {
    pub const fn years(years: u32) -> Self {
        Self(years)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Completed years between `date_of_birth` and `on`.
    ///
    /// # Leap-day assumption
    ///
    /// A member born on 29 February is treated as having their anniversary on
    /// 28 February in non-leap years. This is the common convention, but it is
    /// an **assumption**, not a cited rule — see the open question in
    /// [`contribution_band_on`].
    pub fn attained(date_of_birth: Date, on: Date) -> Self {
        let mut years = on.year() - date_of_birth.year();
        let had_birthday = (on.month(), on.day()) >= (date_of_birth.month(), anniversary_day(date_of_birth, on.year()));
        if !had_birthday {
            years -= 1;
        }
        Self(years.max(0) as u32)
    }
}

impl fmt::Display for Age {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The day-of-month on which `date_of_birth`'s anniversary falls in `year`.
/// Only 29 February is interesting; it maps to 28 February in a non-leap year.
fn anniversary_day(date_of_birth: Date, year: i32) -> u8 {
    if date_of_birth.month() == 2 && date_of_birth.day() == 29 && Date::new(year, 2, 29).is_err() {
        28
    } else {
        date_of_birth.day()
    }
}

/// The date on which a member born on `date_of_birth` has their `n`th birthday.
///
/// Falls back to 28 February for a 29-February birth in a non-leap year.
fn nth_birthday(date_of_birth: Date, n: u32) -> Date {
    let year = date_of_birth.year() + n as i32;
    let day = anniversary_day(date_of_birth, year);
    Date::new(year, date_of_birth.month(), day)
        .expect("an anniversary of a valid date is a valid date")
}

// ---------------------------------------------------------------------------
// Contribution age bands
// ---------------------------------------------------------------------------

/// The age bands CPF **contribution rates** are indexed by.
///
/// Five bands, stepping at 55, 60, 65 and 70.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContributionAgeBand {
    /// "55 years and below".
    UpTo55,
    /// "Above 55 to 60 years".
    Above55To60,
    /// "Above 60 to 65 years".
    Above60To65,
    /// "Above 65 to 70 years".
    Above65To70,
    /// "Above 70 years".
    Above70,
}

impl ContributionAgeBand {
    pub const ALL: [ContributionAgeBand; 5] = [
        ContributionAgeBand::UpTo55,
        ContributionAgeBand::Above55To60,
        ContributionAgeBand::Above60To65,
        ContributionAgeBand::Above65To70,
        ContributionAgeBand::Above70,
    ];

    /// Selects a band by chronological age alone.
    ///
    /// # This is the coarse path, and it can be wrong for a real person
    ///
    /// CPF does **not** change a member's rate on their birthday. It changes it
    /// on the first day of the month *after* the birthday month (see
    /// [`contribution_band_on`]). This function cannot know that, because it is
    /// not given a date of birth — it maps `55 -> UpTo55`, which is what the
    /// published table header literally says, and which is the right answer to
    /// the question *"what does the table say for a 57-year-old?"*.
    ///
    /// It is **not** the right answer to *"what rate applies to this employee's
    /// March payroll?"*. For that you have a date of birth, and you must use
    /// [`contribution_band_on`].
    ///
    /// Both exist because both questions get asked, and conflating them is the
    /// bug.
    pub fn for_age(age: Age) -> Self {
        match age.get() {
            0..=55 => ContributionAgeBand::UpTo55,
            56..=60 => ContributionAgeBand::Above55To60,
            61..=65 => ContributionAgeBand::Above60To65,
            66..=70 => ContributionAgeBand::Above65To70,
            _ => ContributionAgeBand::Above70,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ContributionAgeBand::UpTo55 => "55 years and below",
            ContributionAgeBand::Above55To60 => "Above 55 to 60 years",
            ContributionAgeBand::Above60To65 => "Above 60 to 65 years",
            ContributionAgeBand::Above65To70 => "Above 65 to 70 years",
            ContributionAgeBand::Above70 => "Above 70 years",
        }
    }
}

impl fmt::Display for ContributionAgeBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The contribution age band applicable to the payroll month containing
/// `in_month`, for a member born on `date_of_birth`.
///
/// # The rule, and why it is its own function
///
/// > A member's CPF contribution rate changes from the **first day of the month
/// > following** the month in which they turn 55, 60, 65 or 70.
///
/// Worked example. A member born **15 March 1970** turns 55 on **15 March 2025**.
///
/// | Payroll month | Band |
/// |---|---|
/// | February 2025 | 55 years and below |
/// | **March 2025** (the birthday month) | **55 years and below** — the birthday does *not* move them |
/// | **April 2025** | **Above 55 to 60** |
///
/// Getting this wrong by one month over-deducts or under-deducts a full month's
/// contribution at a rate that differs by 4.5 percentage points. It is the
/// single most common bug in this domain, which is why it lives in a named
/// function with a table in its documentation rather than inline in a `match`.
///
/// # Open question (deliberately not guessed)
///
/// Singapore statute sometimes applies the common-law rule that a person attains
/// an age at the *start of the day before* the anniversary of their birth. If
/// CPF applies it, a member born on the **1st** of a month attains the age in
/// the *previous* month, and their band would change one month earlier than this
/// function says. KOPITIAM implements the straightforward reading (anniversary =
/// same day-of-month) and **does not pretend to know** which is operative. If
/// you are computing payroll for a member born on the 1st of a month in their
/// 55th/60th/65th/70th year, verify against the primary source.
///
/// # Errors
///
/// [`CpfError::InvalidDate`] if `in_month` is not a valid date.
pub fn contribution_band_on(date_of_birth: Date, in_month: Date) -> Result<ContributionAgeBand, CpfError> {
    // Normalise to the first of the payroll month: the rule operates on whole
    // months, so the day within the payroll month is irrelevant.
    let month_start = Date::first_of(in_month.year(), in_month.month())?;

    // The band changes on the first day of the month *after* the threshold
    // birthday. Encoding it as a boundary date, rather than as arithmetic on an
    // integer age, is what makes the rule impossible to get off by one.
    let switches_at = |n: u32| nth_birthday(date_of_birth, n).start_of_next_month();

    Ok(if month_start >= switches_at(70) {
        ContributionAgeBand::Above70
    } else if month_start >= switches_at(65) {
        ContributionAgeBand::Above65To70
    } else if month_start >= switches_at(60) {
        ContributionAgeBand::Above60To65
    } else if month_start >= switches_at(55) {
        ContributionAgeBand::Above55To60
    } else {
        ContributionAgeBand::UpTo55
    })
}

// ---------------------------------------------------------------------------
// Allocation age bands
// ---------------------------------------------------------------------------

/// The age bands CPF **allocation ratios** (the OA/SA/MA split) are indexed by.
///
/// **Seven bands, and they are not the contribution bands.** They step at 35,
/// 45, 50, 55, 60 and 65. A 47-year-old sits in a single contribution band
/// (`UpTo55`) but has a materially different Ordinary/Special split from a
/// 33-year-old. Any code that indexes allocation by [`ContributionAgeBand`] is
/// broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationAgeBand {
    /// "35 years and below".
    UpTo35,
    /// "Above 35 to 45 years".
    Above35To45,
    /// "Above 45 to 50 years".
    Above45To50,
    /// "Above 50 to 55 years".
    Above50To55,
    /// "Above 55 to 60 years".
    Above55To60,
    /// "Above 60 to 65 years".
    Above60To65,
    /// "Above 65 years".
    Above65,
}

impl AllocationAgeBand {
    pub const ALL: [AllocationAgeBand; 7] = [
        AllocationAgeBand::UpTo35,
        AllocationAgeBand::Above35To45,
        AllocationAgeBand::Above45To50,
        AllocationAgeBand::Above50To55,
        AllocationAgeBand::Above55To60,
        AllocationAgeBand::Above60To65,
        AllocationAgeBand::Above65,
    ];

    /// The bands for which KOPITIAM currently holds cited allocation ratios.
    /// Everything at 55 and above is unpopulated — see [`crate::cpf::published`].
    pub const POPULATED: [AllocationAgeBand; 4] = [
        AllocationAgeBand::UpTo35,
        AllocationAgeBand::Above35To45,
        AllocationAgeBand::Above45To50,
        AllocationAgeBand::Above50To55,
    ];

    /// Selects a band by chronological age alone. Carries the same caveat as
    /// [`ContributionAgeBand::for_age`] — prefer [`allocation_band_on`] when you
    /// have a date of birth.
    pub fn for_age(age: Age) -> Self {
        match age.get() {
            0..=35 => AllocationAgeBand::UpTo35,
            36..=45 => AllocationAgeBand::Above35To45,
            46..=50 => AllocationAgeBand::Above45To50,
            51..=55 => AllocationAgeBand::Above50To55,
            56..=60 => AllocationAgeBand::Above55To60,
            61..=65 => AllocationAgeBand::Above60To65,
            _ => AllocationAgeBand::Above65,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            AllocationAgeBand::UpTo35 => "35 years and below",
            AllocationAgeBand::Above35To45 => "Above 35 to 45 years",
            AllocationAgeBand::Above45To50 => "Above 45 to 50 years",
            AllocationAgeBand::Above50To55 => "Above 50 to 55 years",
            AllocationAgeBand::Above55To60 => "Above 55 to 60 years",
            AllocationAgeBand::Above60To65 => "Above 60 to 65 years",
            AllocationAgeBand::Above65 => "Above 65 years",
        }
    }
}

impl fmt::Display for AllocationAgeBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The allocation age band applicable to the payroll month containing
/// `in_month`. Applies the same month-after-the-birthday rule as
/// [`contribution_band_on`], at the allocation thresholds.
///
/// # Errors
///
/// [`CpfError::InvalidDate`] if `in_month` is not a valid date.
pub fn allocation_band_on(date_of_birth: Date, in_month: Date) -> Result<AllocationAgeBand, CpfError> {
    let month_start = Date::first_of(in_month.year(), in_month.month())?;
    let switches_at = |n: u32| nth_birthday(date_of_birth, n).start_of_next_month();

    Ok(if month_start >= switches_at(65) {
        AllocationAgeBand::Above65
    } else if month_start >= switches_at(60) {
        AllocationAgeBand::Above60To65
    } else if month_start >= switches_at(55) {
        AllocationAgeBand::Above55To60
    } else if month_start >= switches_at(50) {
        AllocationAgeBand::Above50To55
    } else if month_start >= switches_at(45) {
        AllocationAgeBand::Above45To50
    } else if month_start >= switches_at(35) {
        AllocationAgeBand::Above35To45
    } else {
        AllocationAgeBand::UpTo35
    })
}

// ---------------------------------------------------------------------------
// Wage bands
// ---------------------------------------------------------------------------

/// The **full** employee contribution rate applies only from $750 of total
/// monthly wages. Below that, contributions are phased in. This threshold is
/// cited and confident; see [`WageBand`].
pub const FULL_RATE_WAGE_THRESHOLD: Sgd = Sgd::from_dollars(750);

/// Total-wage bands that determine *how much* of the published rate applies.
///
/// CPF does not apply the headline rates to every wage. Low-wage employees have
/// employee contributions phased in, so that a raise from $499 to $501 does not
/// cost them money in take-home pay.
///
/// # Only [`WageBand::AtOrAbove750`] is populated in KOPITIAM
///
/// The phase-in formulas for the graduated bands are functions of the wage *and*
/// the age band, and change with each rate revision. The author of this crate is
/// **not confident** of their current form. Rather than encode a plausible-looking
/// formula and have it quietly under-deduct from the lowest-paid members — the
/// people who can least afford the error — the graduated bands return
/// [`CpfError::NotPopulated`].
///
/// The band boundaries below $750 (whether exactly $50 and exactly $500 fall in
/// the lower or upper band) are likewise **not** confidently known and should be
/// verified before the graduated bands are populated. That uncertainty is
/// harmless today precisely *because* those bands answer nothing: the only
/// boundary that currently changes an answer is $750, which is confident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WageBand {
    /// Total wages at or below $50/month. No contributions are payable.
    /// **Not populated** — see the type docs on the boundary uncertainty.
    UpTo50,
    /// Above $50 to $500/month. Employer contributes; the employee does not.
    /// **Not populated.**
    Above50To500,
    /// Above $500 and below $750/month. The employee's share is phased in.
    /// **Not populated.**
    Above500Below750,
    /// $750/month and above. Full published rates apply. **This is the band
    /// KOPITIAM can answer for.**
    AtOrAbove750,
}

impl WageBand {
    /// Classifies total monthly wages (Ordinary + Additional) into a band.
    pub fn classify(total_wages: Sgd) -> Self {
        if total_wages >= FULL_RATE_WAGE_THRESHOLD {
            WageBand::AtOrAbove750
        } else if total_wages > Sgd::from_dollars(500) {
            WageBand::Above500Below750
        } else if total_wages > Sgd::from_dollars(50) {
            WageBand::Above50To500
        } else {
            WageBand::UpTo50
        }
    }

    /// Whether KOPITIAM holds cited rules for this band.
    pub fn is_populated(self) -> bool {
        matches!(self, WageBand::AtOrAbove750)
    }

    pub fn label(self) -> &'static str {
        match self {
            WageBand::UpTo50 => "Total wages up to $50/month",
            WageBand::Above50To500 => "Total wages above $50 to $500/month",
            WageBand::Above500Below750 => "Total wages above $500 and below $750/month",
            WageBand::AtOrAbove750 => "Total wages of $750/month and above",
        }
    }
}

impl fmt::Display for WageBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Residency
// ---------------------------------------------------------------------------

/// A member's CPF residency status, which selects an entirely different rate
/// table.
///
/// # Only [`Residency::CitizenOrPrFromThirdYear`] is populated
///
/// Singapore Permanent Residents pay *graduated* rates in their first two years
/// of PR status, to ease the transition. Worse, employer and employee may
/// *jointly apply* to pay full rates early, and there is a third combination
/// (full employer / graduated employee). That is three distinct rate tables and
/// three distinct allocation tables per PR year, and the author is **not
/// confident** of their current values.
///
/// They are therefore named here — so the shape of the domain is honest and so a
/// future contributor knows exactly what is missing — and return
/// [`CpfError::NotPopulated`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    /// Singapore Citizen, or Permanent Resident from the third year of PR
    /// status onward. These share one rate table.
    CitizenOrPrFromThirdYear,
    /// Permanent Resident, first year of PR status. **Not populated.**
    PrFirstYear,
    /// Permanent Resident, second year of PR status. **Not populated.**
    PrSecondYear,
}

impl Residency {
    /// Whether KOPITIAM holds cited rules for this status.
    pub fn is_populated(self) -> bool {
        matches!(self, Residency::CitizenOrPrFromThirdYear)
    }

    pub fn label(self) -> &'static str {
        match self {
            Residency::CitizenOrPrFromThirdYear => {
                "Singapore Citizen, or SPR from the 3rd year of PR status"
            }
            Residency::PrFirstYear => "Singapore PR, 1st year of PR status",
            Residency::PrSecondYear => "Singapore PR, 2nd year of PR status",
        }
    }
}

impl fmt::Display for Residency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u8, day: u8) -> Date {
        Date::new(y, m, day).unwrap()
    }

    // -- The month-after-birthday rule: both sides of every threshold --------

    /// Born 15 March 1970. Turns 55 on 15 March 2025.
    ///
    /// The birthday month itself is still the lower band; the change lands on
    /// 1 April. This is the classic off-by-one, tested from both sides.
    #[test]
    fn contribution_band_changes_the_month_after_the_55th_birthday() {
        let dob = d(1970, 3, 15);

        // Well before.
        assert_eq!(contribution_band_on(dob, d(2025, 1, 31)).unwrap(), ContributionAgeBand::UpTo55);

        // The month *before* the birthday month.
        assert_eq!(contribution_band_on(dob, d(2025, 2, 28)).unwrap(), ContributionAgeBand::UpTo55);

        // The birthday month itself — before the birthday...
        assert_eq!(contribution_band_on(dob, d(2025, 3, 1)).unwrap(), ContributionAgeBand::UpTo55);
        // ...on the birthday...
        assert_eq!(contribution_band_on(dob, d(2025, 3, 15)).unwrap(), ContributionAgeBand::UpTo55);
        // ...and after it. The birthday does NOT move the band.
        assert_eq!(contribution_band_on(dob, d(2025, 3, 31)).unwrap(), ContributionAgeBand::UpTo55);

        // The first day of the next month. The band changes here, and only here.
        assert_eq!(contribution_band_on(dob, d(2025, 4, 1)).unwrap(), ContributionAgeBand::Above55To60);
        assert_eq!(contribution_band_on(dob, d(2025, 4, 30)).unwrap(), ContributionAgeBand::Above55To60);
    }

    /// A December birthday rolls the boundary into the next calendar year.
    #[test]
    fn a_december_birthday_switches_the_band_in_january() {
        let dob = d(1970, 12, 20);
        assert_eq!(contribution_band_on(dob, d(2025, 12, 31)).unwrap(), ContributionAgeBand::UpTo55);
        assert_eq!(contribution_band_on(dob, d(2026, 1, 1)).unwrap(), ContributionAgeBand::Above55To60);
    }

    /// Every contribution threshold, both sides.
    #[test]
    fn every_contribution_threshold_flips_on_the_first_of_the_following_month() {
        let dob = d(1960, 6, 10);
        let cases = [
            (55, ContributionAgeBand::UpTo55, ContributionAgeBand::Above55To60),
            (60, ContributionAgeBand::Above55To60, ContributionAgeBand::Above60To65),
            (65, ContributionAgeBand::Above60To65, ContributionAgeBand::Above65To70),
            (70, ContributionAgeBand::Above65To70, ContributionAgeBand::Above70),
        ];
        for (n, before, after) in cases {
            let birthday_year = 1960 + n;
            // Last day of the birthday month: still the old band.
            assert_eq!(
                contribution_band_on(dob, d(birthday_year, 6, 30)).unwrap(),
                before,
                "age {n}: last day of the birthday month must still be the lower band",
            );
            // First day of the next month: the new band.
            assert_eq!(
                contribution_band_on(dob, d(birthday_year, 7, 1)).unwrap(),
                after,
                "age {n}: first day of the following month must be the higher band",
            );
        }
    }

    /// Every allocation threshold, both sides. Different thresholds, same rule.
    #[test]
    fn every_allocation_threshold_flips_on_the_first_of_the_following_month() {
        let dob = d(1980, 9, 5);
        let cases = [
            (35, AllocationAgeBand::UpTo35, AllocationAgeBand::Above35To45),
            (45, AllocationAgeBand::Above35To45, AllocationAgeBand::Above45To50),
            (50, AllocationAgeBand::Above45To50, AllocationAgeBand::Above50To55),
            (55, AllocationAgeBand::Above50To55, AllocationAgeBand::Above55To60),
            (60, AllocationAgeBand::Above55To60, AllocationAgeBand::Above60To65),
            (65, AllocationAgeBand::Above60To65, AllocationAgeBand::Above65),
        ];
        for (n, before, after) in cases {
            let year = 1980 + n;
            assert_eq!(allocation_band_on(dob, d(year, 9, 30)).unwrap(), before, "age {n}, birthday month");
            assert_eq!(allocation_band_on(dob, d(year, 10, 1)).unwrap(), after, "age {n}, following month");
        }
    }

    /// The allocation bands are *not* the contribution bands. A 47-year-old is
    /// in one contribution band with a 33-year-old but a different allocation
    /// band — the exact confusion this test exists to prevent regressing.
    #[test]
    fn allocation_and_contribution_bands_are_genuinely_different() {
        let young = Age::years(33);
        let middle = Age::years(47);

        assert_eq!(ContributionAgeBand::for_age(young), ContributionAgeBand::for_age(middle));
        assert_ne!(AllocationAgeBand::for_age(young), AllocationAgeBand::for_age(middle));
    }

    /// A 29-February birth: the anniversary lands on 28 February in a non-leap
    /// year, so the band still changes on 1 March. Documented assumption, tested
    /// so it cannot drift silently.
    #[test]
    fn leap_day_birth_switches_on_the_first_of_march_in_a_non_leap_year() {
        let dob = d(1968, 2, 29);
        // 2023 is not a leap year; the 55th "birthday" is taken as 28 Feb 2023.
        assert_eq!(contribution_band_on(dob, d(2023, 2, 28)).unwrap(), ContributionAgeBand::UpTo55);
        assert_eq!(contribution_band_on(dob, d(2023, 3, 1)).unwrap(), ContributionAgeBand::Above55To60);
    }

    // -- Chronological-age band selection ------------------------------------

    #[test]
    fn for_age_matches_the_published_table_headers() {
        assert_eq!(ContributionAgeBand::for_age(Age::years(55)), ContributionAgeBand::UpTo55);
        assert_eq!(ContributionAgeBand::for_age(Age::years(56)), ContributionAgeBand::Above55To60);
        assert_eq!(ContributionAgeBand::for_age(Age::years(60)), ContributionAgeBand::Above55To60);
        assert_eq!(ContributionAgeBand::for_age(Age::years(61)), ContributionAgeBand::Above60To65);
        assert_eq!(ContributionAgeBand::for_age(Age::years(65)), ContributionAgeBand::Above60To65);
        assert_eq!(ContributionAgeBand::for_age(Age::years(66)), ContributionAgeBand::Above65To70);
        assert_eq!(ContributionAgeBand::for_age(Age::years(70)), ContributionAgeBand::Above65To70);
        assert_eq!(ContributionAgeBand::for_age(Age::years(71)), ContributionAgeBand::Above70);

        assert_eq!(AllocationAgeBand::for_age(Age::years(35)), AllocationAgeBand::UpTo35);
        assert_eq!(AllocationAgeBand::for_age(Age::years(36)), AllocationAgeBand::Above35To45);
        assert_eq!(AllocationAgeBand::for_age(Age::years(45)), AllocationAgeBand::Above35To45);
        assert_eq!(AllocationAgeBand::for_age(Age::years(46)), AllocationAgeBand::Above45To50);
        assert_eq!(AllocationAgeBand::for_age(Age::years(50)), AllocationAgeBand::Above45To50);
        assert_eq!(AllocationAgeBand::for_age(Age::years(51)), AllocationAgeBand::Above50To55);
        assert_eq!(AllocationAgeBand::for_age(Age::years(55)), AllocationAgeBand::Above50To55);
        assert_eq!(AllocationAgeBand::for_age(Age::years(56)), AllocationAgeBand::Above55To60);
        assert_eq!(AllocationAgeBand::for_age(Age::years(66)), AllocationAgeBand::Above65);
    }

    #[test]
    fn attained_age_counts_completed_years() {
        let dob = d(1970, 3, 15);
        assert_eq!(Age::attained(dob, d(2025, 3, 14)), Age::years(54));
        assert_eq!(Age::attained(dob, d(2025, 3, 15)), Age::years(55));
        assert_eq!(Age::attained(dob, d(2025, 3, 16)), Age::years(55));
        assert_eq!(Age::attained(dob, d(2026, 3, 14)), Age::years(55));
        assert_eq!(Age::attained(dob, d(2026, 3, 15)), Age::years(56));
    }

    // -- Wage bands ----------------------------------------------------------

    #[test]
    fn wage_band_boundaries() {
        assert_eq!(WageBand::classify(Sgd::from_dollars(0)), WageBand::UpTo50);
        assert_eq!(WageBand::classify(Sgd::from_dollars(50)), WageBand::UpTo50);
        assert_eq!(WageBand::classify(Sgd::from_cents(5_001)), WageBand::Above50To500);
        assert_eq!(WageBand::classify(Sgd::from_dollars(500)), WageBand::Above50To500);
        assert_eq!(WageBand::classify(Sgd::from_cents(50_001)), WageBand::Above500Below750);
        assert_eq!(WageBand::classify(Sgd::from_cents(74_999)), WageBand::Above500Below750);

        // The one boundary that changes an answer, and the one we are confident
        // about: exactly $750 gets the full published rates.
        assert_eq!(WageBand::classify(Sgd::from_dollars(750)), WageBand::AtOrAbove750);
        assert_eq!(WageBand::classify(Sgd::from_dollars(10_000)), WageBand::AtOrAbove750);
    }

    #[test]
    fn only_the_full_rate_band_and_citizen_status_are_populated() {
        assert!(WageBand::AtOrAbove750.is_populated());
        assert!(!WageBand::UpTo50.is_populated());
        assert!(!WageBand::Above50To500.is_populated());
        assert!(!WageBand::Above500Below750.is_populated());

        assert!(Residency::CitizenOrPrFromThirdYear.is_populated());
        assert!(!Residency::PrFirstYear.is_populated());
        assert!(!Residency::PrSecondYear.is_populated());
    }
}
