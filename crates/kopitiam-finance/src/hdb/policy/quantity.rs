//! The quantities HDB policy is written in — money, percentages, durations,
//! ages — each with its own type.
//!
//! # Why not `f64` (or `u32`) for all of it
//!
//! An income ceiling, a grant amount, an ethnic quota and a Minimum Occupation
//! Period are not four numbers. They are four *different kinds of thing*, and a
//! function that accepts any of them where it meant one of them is a function
//! that will eventually compare a household's income against a quota percentage
//! and be perfectly happy about it. CLAUDE.md asks for strong typing; this is
//! where the domain makes it pay.
//!
//! Money in particular is never `f64` here. `14_000.10_f64` is not
//! representable, grant tapers are computed by repeated subtraction, and an
//! income compared against a ceiling must be *exact* at the boundary — because
//! the boundary is where the domain's classic bug lives ("not exceeding
//! $14,000" includes $14,000). [`Sgd`] is therefore an integer count of cents,
//! and every arithmetic operation on it is checked.

use std::fmt;

use serde::{Deserialize, Serialize};

/// An amount of Singapore dollars, stored as an exact integer number of cents.
///
/// `i64` cents covers ±92 quadrillion dollars, which is ample for a policy about
/// flats, and leaves room for signed intermediate results (a taper subtracting
/// past zero must be *detectable*, not saturating).
///
/// There is deliberately **no** `From<f64>`. If a figure arrives as a float it
/// has already lost the property that makes it money.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Sgd(i64);

/// Money arithmetic that could not be carried out exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MoneyError {
    /// The result did not fit in the representation.
    #[error("SGD arithmetic overflowed")]
    Overflow,
}

impl Sgd {
    /// Zero dollars.
    pub const ZERO: Sgd = Sgd(0);

    /// A whole-dollar amount. Every figure HDB publishes for ceilings, grants
    /// and levies is a whole number of dollars, so this is the constructor the
    /// policy tables use.
    pub const fn dollars(dollars: i64) -> Self {
        Sgd(dollars * 100)
    }

    /// An exact amount of cents.
    pub const fn cents(cents: i64) -> Self {
        Sgd(cents)
    }

    /// The amount, in cents. The only lossless accessor, hence the only one.
    pub const fn as_cents(self) -> i64 {
        self.0
    }

    /// Checked addition.
    pub fn checked_add(self, other: Sgd) -> Result<Sgd, MoneyError> {
        self.0
            .checked_add(other.0)
            .map(Sgd)
            .ok_or(MoneyError::Overflow)
    }

    /// Checked subtraction. May go negative — that is a legitimate signal (a
    /// grant taper that has run past its floor), not something to clamp away
    /// silently.
    pub fn checked_sub(self, other: Sgd) -> Result<Sgd, MoneyError> {
        self.0
            .checked_sub(other.0)
            .map(Sgd)
            .ok_or(MoneyError::Overflow)
    }

    /// Whether the amount is negative.
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl fmt::Display for Sgd {
    /// `S$14,000.00` — grouped, two decimal places, no floating point anywhere
    /// in the formatting path.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let negative = self.0 < 0;
        let abs = self.0.unsigned_abs();
        let (dollars, cents) = (abs / 100, abs % 100);

        let digits = dollars.to_string();
        let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
        for (i, ch) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i) % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(ch);
        }

        write!(
            f,
            "{}S${}.{:02}",
            if negative { "-" } else { "" },
            grouped,
            cents
        )
    }
}

/// A monthly household income: the sum over the applicants, gross, before CPF.
///
/// A distinct type from [`IncomeCeiling`] on purpose. Comparing an income to a
/// ceiling should require a deliberate call ([`IncomeCeiling::admits`]), not a
/// stray `<=` that might as easily have been a `<`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MonthlyIncome(pub Sgd);

impl fmt::Display for MonthlyIncome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/month", self.0)
    }
}

/// The income limit that applies to a purchase.
///
/// Modelled as an enum rather than an `Option<Sgd>` because "there is no income
/// ceiling" is a **rule** — it is what HDB says about buying a resale flat
/// without a grant — and not an absence of one. `None` would mean "we don't
/// know", and the two must never be confused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncomeCeiling {
    /// No income ceiling applies to this purchase.
    NoCeiling,
    /// The household's average gross monthly income must not exceed this.
    ///
    /// **Inclusive.** HDB's wording is "not exceeding $14,000", so a household
    /// at exactly $14,000 is within the ceiling. The off-by-one here is the most
    /// consequential single character in the crate.
    NotExceeding(Sgd),
}

impl IncomeCeiling {
    /// Whether `income` is within this ceiling.
    ///
    /// At the boundary this returns `true`: "not exceeding $14,000" admits
    /// $14,000 and excludes $14,000.01.
    pub fn admits(self, income: MonthlyIncome) -> bool {
        match self {
            IncomeCeiling::NoCeiling => true,
            IncomeCeiling::NotExceeding(limit) => income.0 <= limit,
        }
    }
}

impl fmt::Display for IncomeCeiling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IncomeCeiling::NoCeiling => write!(f, "no income ceiling"),
            IncomeCeiling::NotExceeding(limit) => write!(f, "not exceeding {limit}/month"),
        }
    }
}

/// A housing grant amount.
///
/// Distinct from [`Sgd`] at the API boundary so that a grant cannot be silently
/// added to an income or compared to a ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GrantAmount(pub Sgd);

impl fmt::Display for GrantAmount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A percentage, stored in **basis points** (hundredths of a percent) so that
/// quota arithmetic stays exact.
///
/// HDB's Ethnic Integration Policy limits are whole percents today (84%, 22%,
/// 12%), but the type does not assume they will stay whole, and basis points
/// cost nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Percent {
    basis_points: u32,
}

impl Percent {
    /// A whole-percent figure, e.g. `Percent::whole(84)` for 84%.
    pub const fn whole(percent: u32) -> Self {
        Self {
            basis_points: percent * 100,
        }
    }

    /// The figure in basis points (84% == 8400 bp).
    pub const fn basis_points(self) -> u32 {
        self.basis_points
    }
}

impl fmt::Display for Percent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (whole, frac) = (self.basis_points / 100, self.basis_points % 100);
        if frac == 0 {
            write!(f, "{whole}%")
        } else {
            write!(f, "{whole}.{frac:02}%")
        }
    }
}

/// A duration in whole months — the unit HDB states occupation periods in.
///
/// A five-year MOP is 60 months, and it is *not* five `Date`s apart in any way
/// this crate needs to compute, so no calendar arithmetic is offered. Whether a
/// given household has *served* its MOP depends on the key-collection date and
/// on absences the crate does not model; see
/// [`rules::UNMODELLED`](super::rules::UNMODELLED).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Months(pub u32);

impl fmt::Display for Months {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_multiple_of(12) {
            let years = self.0 / 12;
            write!(
                f,
                "{} months ({} year{})",
                self.0,
                years,
                if years == 1 { "" } else { "s" }
            )
        } else {
            write!(f, "{} months", self.0)
        }
    }
}

/// The Minimum Occupation Period: how long a flat must be lived in before it may
/// be sold on the open market (or, for some flat classes, rented out whole).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MinimumOccupationPeriod(pub Months);

impl fmt::Display for MinimumOccupationPeriod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An age in completed years.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Age(pub u32);

impl fmt::Display for Age {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} years", self.0)
    }
}

/// The minimum age an eligibility scheme requires of its applicants.
///
/// **This is a policy number, and it therefore lives in a dated, cited table
/// like every other one** ([`rules::minimum_ages`](super::rules::minimum_ages)),
/// not in a `const`. The 35-year threshold of the Single Singapore Citizen
/// Scheme is exactly the kind of figure that looks eternal right up until it is
/// changed in a Rally speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MinimumAge(pub Age);

impl MinimumAge {
    /// Whether an applicant of `age` meets this threshold.
    ///
    /// Inclusive: "at least 21 years old" admits someone who is exactly 21.
    pub fn admits(self, age: Age) -> bool {
        age >= self.0
    }
}

impl fmt::Display for MinimumAge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at least {}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_is_exact_and_never_a_float() {
        assert_eq!(Sgd::dollars(14_000).as_cents(), 1_400_000);
        assert_eq!(Sgd::dollars(14_000).to_string(), "S$14,000.00");
        assert_eq!(Sgd::dollars(500).to_string(), "S$500.00");
        assert_eq!(Sgd::cents(1).to_string(), "S$0.01");
        assert_eq!(Sgd::dollars(1_234_567).to_string(), "S$1,234,567.00");
        assert_eq!(Sgd::dollars(-5).to_string(), "-S$5.00");

        // The sum that a binary float gets wrong. Ten lots of ten cents is
        // exactly one dollar, and here it is exactly one dollar.
        let mut total = Sgd::ZERO;
        for _ in 0..10 {
            total = total.checked_add(Sgd::cents(10)).unwrap();
        }
        assert_eq!(total, Sgd::dollars(1));
    }

    #[test]
    fn money_arithmetic_is_checked() {
        assert!(Sgd::cents(i64::MAX).checked_add(Sgd::cents(1)).is_err());
        assert!(
            Sgd::dollars(1)
                .checked_sub(Sgd::dollars(2))
                .unwrap()
                .is_negative()
        );
    }

    #[test]
    fn a_ceiling_admits_a_household_sitting_exactly_on_it() {
        let ceiling = IncomeCeiling::NotExceeding(Sgd::dollars(14_000));
        assert!(ceiling.admits(MonthlyIncome(Sgd::dollars(13_999))));
        assert!(
            ceiling.admits(MonthlyIncome(Sgd::dollars(14_000))),
            "'not exceeding $14,000' includes exactly $14,000"
        );
        assert!(
            !ceiling.admits(MonthlyIncome(Sgd::cents(1_400_001))),
            "one cent over is over"
        );
        assert!(!ceiling.admits(MonthlyIncome(Sgd::dollars(14_001))));
    }

    #[test]
    fn no_ceiling_admits_everyone() {
        assert!(IncomeCeiling::NoCeiling.admits(MonthlyIncome(Sgd::dollars(1_000_000))));
    }

    #[test]
    fn a_minimum_age_admits_someone_exactly_at_it() {
        let min = MinimumAge(Age(35));
        assert!(!min.admits(Age(34)));
        assert!(min.admits(Age(35)), "'at least 35' includes exactly 35");
        assert!(min.admits(Age(36)));
    }

    #[test]
    fn quantities_render_for_humans() {
        assert_eq!(Percent::whole(84).to_string(), "84%");
        assert_eq!(Percent::whole(84).basis_points(), 8400);
        assert_eq!(Months(60).to_string(), "60 months (5 years)");
        assert_eq!(Months(12).to_string(), "12 months (1 year)");
        assert_eq!(Months(7).to_string(), "7 months");
    }
}
