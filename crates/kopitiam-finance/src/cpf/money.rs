//! Exact money and rate arithmetic. **No floating point appears anywhere in
//! this module, or in any public API of the CPF engine.**
//!
//! # Why this matters more than usual
//!
//! CPF contributions are computed from a wage, a rate, and a *statutory
//! rounding rule*, and the result is a real deduction from a real payslip. A
//! binary `f64` cannot represent `0.1`, cannot represent a cent exactly, and
//! accumulates error under repeated addition. `wage * 0.20` in `f64` will,
//! for some wages, land one cent — and therefore, after rounding, one *dollar*
//! — away from what the CPF Board's own table says. That is not a rounding
//! nit; that is a wrong number on someone's payslip.
//!
//! So:
//!
//! * [`Sgd`] is an exact integer number of **cents**.
//! * [`Rate`] is an exact integer number of **basis points** (1 bp = 0.01%).
//!   Every CPF figure fits: contribution rates are published to one decimal
//!   place of a percent (`15.5%` = 1550 bp) and allocation ratios to four
//!   decimal places of a ratio (`0.6217` = 6217 bp).
//! * Multiplying the two produces an [`Unrounded`] value which **cannot be
//!   spent**. You must state a rounding rule to turn it into [`Sgd`].
//!
//! That last point is the whole design. See [`Unrounded`].

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Sub};

use serde::{Deserialize, Serialize};

use crate::cpf::error::CpfError;

/// Cents per dollar.
const CENTS_PER_DOLLAR: i64 = 100;

/// Basis points per unit (1.0 == 100% == 10 000 bp).
const BP_PER_UNIT: i128 = 10_000;

// ---------------------------------------------------------------------------
// Sgd
// ---------------------------------------------------------------------------

/// An exact amount of Singapore dollars, stored as a whole number of cents.
///
/// Negative amounts are representable (a `Sub` can legitimately go negative —
/// see the Additional Wage ceiling, which is a *residual* and can be exhausted)
/// but are meaningless as a contribution. Where a negative would be nonsense
/// the API says so.
///
/// `Ord` is exact and total, so `min`/`max` against a ceiling are trustworthy —
/// which is the single most common operation in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sgd {
    /// Serialised as a plain integer count of cents. Deliberately *not* a JSON
    /// float: a policy fact that round-trips through `1234.56` is a policy fact
    /// that can come back as `1234.5599999999999`.
    cents: i64,
}

impl Sgd {
    pub const ZERO: Sgd = Sgd { cents: 0 };

    /// From a whole number of dollars.
    pub const fn from_dollars(dollars: i64) -> Self {
        Self {
            cents: dollars * CENTS_PER_DOLLAR,
        }
    }

    /// From an exact number of cents.
    pub const fn from_cents(cents: i64) -> Self {
        Self { cents }
    }

    pub const fn cents(self) -> i64 {
        self.cents
    }

    /// Whole dollars, truncated toward zero. Use only for display; never as a
    /// rounding step (see [`Unrounded`]).
    pub const fn whole_dollars(self) -> i64 {
        self.cents / CENTS_PER_DOLLAR
    }

    pub const fn is_negative(self) -> bool {
        self.cents < 0
    }

    /// Clamps a negative amount to zero.
    ///
    /// Exists for exactly one reason: the Additional Wage ceiling is
    /// `annual ceiling − Ordinary Wages already subject to CPF`, and for a
    /// high earner that residual is negative. A negative ceiling means "no
    /// Additional Wages attract CPF", not "CPF owes you money".
    pub fn clamp_non_negative(self) -> Self {
        if self.cents < 0 { Self::ZERO } else { self }
    }

    /// Parses `"1234.56"`, `"1,234.56"`, `"$1,234.56"`, `"1234"`.
    ///
    /// Exact: the fractional part is parsed as an integer number of cents, so
    /// no value ever passes through a float. Rejects more than two decimal
    /// places rather than silently truncating — a wage of `100.005` is a
    /// data-quality problem the caller must resolve, not one this crate should
    /// paper over.
    pub fn parse(input: &str) -> Result<Self, CpfError> {
        let cleaned: String = input
            .trim()
            .trim_start_matches('$')
            .chars()
            .filter(|c| *c != ',' && !c.is_whitespace())
            .collect();

        let bad = || CpfError::InvalidMoney {
            input: input.to_string(),
        };

        let (negative, digits) = match cleaned.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, cleaned.as_str()),
        };

        let (whole_str, frac_str) = match digits.split_once('.') {
            Some((w, f)) => (w, f),
            None => (digits, ""),
        };

        if whole_str.is_empty() || !whole_str.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        if frac_str.len() > 2 || !frac_str.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }

        let whole: i64 = whole_str.parse().map_err(|_| bad())?;
        // "1.5" means 50 cents, not 5. Pad on the right.
        let frac: i64 = match frac_str.len() {
            0 => 0,
            1 => frac_str.parse::<i64>().map_err(|_| bad())? * 10,
            _ => frac_str.parse().map_err(|_| bad())?,
        };

        let cents = whole
            .checked_mul(CENTS_PER_DOLLAR)
            .and_then(|c| c.checked_add(frac))
            .ok_or_else(bad)?;

        Ok(Self {
            cents: if negative { -cents } else { cents },
        })
    }
}

impl Add for Sgd {
    type Output = Sgd;
    fn add(self, rhs: Sgd) -> Sgd {
        Sgd {
            cents: self.cents + rhs.cents,
        }
    }
}

impl Sub for Sgd {
    type Output = Sgd;
    fn sub(self, rhs: Sgd) -> Sgd {
        Sgd {
            cents: self.cents - rhs.cents,
        }
    }
}

impl Sum for Sgd {
    fn sum<I: Iterator<Item = Sgd>>(iter: I) -> Sgd {
        iter.fold(Sgd::ZERO, Add::add)
    }
}

impl fmt::Display for Sgd {
    /// `$1,234.56`. Always two decimal places — a CPF figure printed as
    /// `$1234.5` invites a misread.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.cents < 0 { "-" } else { "" };
        let abs = self.cents.unsigned_abs();
        let dollars = abs / CENTS_PER_DOLLAR as u64;
        let cents = abs % CENTS_PER_DOLLAR as u64;

        // Thousands separators, built right-to-left.
        let digits = dollars.to_string();
        let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
        for (i, c) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i).is_multiple_of(3) {
                grouped.push(',');
            }
            grouped.push(c);
        }

        write!(f, "{sign}${grouped}.{cents:02}")
    }
}

// ---------------------------------------------------------------------------
// Rate
// ---------------------------------------------------------------------------

/// An exact rate or ratio, stored in basis points (1 bp = 0.01% = 0.0001).
///
/// This is deliberately *not* the same type as [`Sgd`]. Multiplying two rates,
/// adding a rate to a wage ceiling, or passing a retirement sum where a
/// contribution rate belongs are all compile errors — which was the point.
///
/// A `Rate` carries no meaning beyond its magnitude. Whether it is an employer
/// contribution rate or an Ordinary Account allocation ratio is expressed by
/// the *struct that holds it* ([`super::rates::ContributionRates`],
/// [`super::rates::AllocationRatios`]), not by this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Rate {
    basis_points: i32,
}

impl Rate {
    pub const ZERO: Rate = Rate { basis_points: 0 };
    /// 100% — the sum an allocation table must hit exactly.
    pub const ONE: Rate = Rate {
        basis_points: 10_000,
    };

    /// From basis points. `1700` is 17%. `6217` is the ratio 0.6217.
    pub const fn from_basis_points(basis_points: i32) -> Self {
        Self { basis_points }
    }

    /// From tenths of a percent, which is the precision CPF publishes
    /// contribution rates at: `17%` is `from_percent_tenths(170)` and `15.5%`
    /// is `from_percent_tenths(155)`.
    ///
    /// Provided so the data tables in [`super::published`] read like the source
    /// document rather than like a units conversion.
    pub const fn from_percent_tenths(tenths: i32) -> Self {
        Self {
            basis_points: tenths * 10,
        }
    }

    pub const fn basis_points(self) -> i32 {
        self.basis_points
    }

    pub const fn is_zero(self) -> bool {
        self.basis_points == 0
    }

    /// Applies this rate to an amount, yielding an **unrounded** result.
    ///
    /// The result is exact and cannot be used as money until a rounding rule is
    /// chosen. See [`Unrounded`].
    pub fn of(self, amount: Sgd) -> Unrounded {
        // Units: cents × bp. Exact; i128 makes overflow unreachable for any
        // wage a human being is paid.
        Unrounded(amount.cents() as i128 * self.basis_points as i128)
    }
}

impl Add for Rate {
    type Output = Rate;
    fn add(self, rhs: Rate) -> Rate {
        Rate {
            basis_points: self.basis_points + rhs.basis_points,
        }
    }
}

impl Sum for Rate {
    fn sum<I: Iterator<Item = Rate>>(iter: I) -> Rate {
        iter.fold(Rate::ZERO, Add::add)
    }
}

impl fmt::Display for Rate {
    /// `17.00%`, `15.50%`, `62.17%`. Two decimal places of a percent is exactly
    /// the precision a basis point carries — no more, no less.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.basis_points < 0 { "-" } else { "" };
        let abs = self.basis_points.unsigned_abs();
        write!(f, "{sign}{}.{:02}%", abs / 100, abs % 100)
    }
}

// ---------------------------------------------------------------------------
// Unrounded
// ---------------------------------------------------------------------------

/// The exact, un-rounded product of a [`Sgd`] and a [`Rate`].
///
/// # Why this type exists
///
/// CPF's rounding rule is not "round the answer". It is, verbatim in substance:
///
/// 1. The **total** contribution is rounded to the **nearest dollar**; 50 cents
///    and above rounds **up**.
/// 2. The **employee's** share is obtained by **dropping the cents** (rounding
///    *down*).
/// 3. The **employer's** share is the **residual**: total − employee's share.
///
/// Read step 3 again. The employer's share is *not* `wage × employer_rate`
/// rounded. It is whatever is left. Compute it independently and you will be
/// off by up to a dollar, in a way that does not show up in casual testing
/// because it depends on the cents of the wage.
///
/// Two distinct rounding rules applied to the same product, plus a residual, is
/// exactly the kind of thing a `f64` pipeline gets wrong silently. So this type
/// makes the un-rounded product a first-class, unspendable value: you cannot
/// get [`Sgd`] out of it without naming which rule you are applying.
///
/// See [`super::rates::ContributionRates::split`] for the rule as code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Unrounded(
    /// Units: cents × basis points. Divide by 10 000 for cents, by 1 000 000
    /// for dollars.
    i128,
);

/// Sub-units per whole dollar: 100 cents × 10 000 bp.
const SUBUNITS_PER_DOLLAR: i128 = CENTS_PER_DOLLAR as i128 * BP_PER_UNIT;

impl Unrounded {
    /// CPF's rule for the **total** contribution: round to the nearest dollar,
    /// with 50 cents and above rounding up.
    pub fn round_to_nearest_dollar(self) -> Sgd {
        let half = SUBUNITS_PER_DOLLAR / 2;
        // Round half *away from zero* is round half *up* for the non-negative
        // amounts this domain deals in, and stays symmetric if a negative ever
        // reaches here.
        let dollars = if self.0 >= 0 {
            (self.0 + half) / SUBUNITS_PER_DOLLAR
        } else {
            (self.0 - half) / SUBUNITS_PER_DOLLAR
        };
        Sgd::from_dollars(dollars as i64)
    }

    /// CPF's rule for the **employee's** share: drop the cents.
    ///
    /// Note this is truncation toward zero, matching "cents are dropped".
    pub fn drop_cents(self) -> Sgd {
        Sgd::from_dollars((self.0 / SUBUNITS_PER_DOLLAR) as i64)
    }

    /// Round to the nearest cent, half away from zero.
    ///
    /// **Not a CPF rule.** Provided for intermediate quantities that are not
    /// contributions (e.g. displaying an interest accrual), and named so that
    /// nobody reaches for it thinking it is one.
    pub fn round_to_nearest_cent(self) -> Sgd {
        let half = BP_PER_UNIT / 2;
        let cents = if self.0 >= 0 {
            (self.0 + half) / BP_PER_UNIT
        } else {
            (self.0 - half) / BP_PER_UNIT
        };
        Sgd::from_cents(cents as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exactly_without_touching_a_float() {
        assert_eq!(Sgd::parse("1234.56").unwrap(), Sgd::from_cents(123_456));
        assert_eq!(Sgd::parse("$1,234.56").unwrap(), Sgd::from_cents(123_456));
        assert_eq!(Sgd::parse("1234").unwrap(), Sgd::from_cents(123_400));
        assert_eq!(Sgd::parse("0.1").unwrap(), Sgd::from_cents(10));
        assert_eq!(Sgd::parse("-5.05").unwrap(), Sgd::from_cents(-505));
        assert_eq!(Sgd::parse("  7400  ").unwrap(), Sgd::from_dollars(7400));

        assert!(Sgd::parse("1.234").is_err(), "3dp must not silently truncate");
        assert!(Sgd::parse("abc").is_err());
        assert!(Sgd::parse("").is_err());
        assert!(Sgd::parse(".5").is_err());
    }

    /// The headline reason this module exists.
    ///
    /// `0.1 + 0.2 != 0.3` in binary floating point. Ten cents plus twenty cents
    /// is thirty cents, exactly, here — and a hundred additions of a
    /// tenth-of-a-dollar is exactly ten dollars, not `9.999999999999998`.
    #[test]
    fn money_addition_is_exact_where_f64_is_not() {
        assert_ne!(0.1_f64 + 0.2_f64, 0.3_f64, "premise check: f64 is lossy");

        let a = Sgd::parse("0.10").unwrap();
        let b = Sgd::parse("0.20").unwrap();
        assert_eq!(a + b, Sgd::parse("0.30").unwrap());

        let hundred_tenths: Sgd = std::iter::repeat_n(Sgd::from_cents(10), 100).sum();
        assert_eq!(hundred_tenths, Sgd::from_dollars(10));
        assert_eq!(hundred_tenths.cents(), 1_000);
    }

    /// A wage that is lossy in binary: $1,234.56 at 20% is exactly $246.912.
    /// In `f64`, `1234.56 * 0.20` is `246.91200000000003`. Exactness matters
    /// here because the value sits near a rounding boundary in other cases.
    #[test]
    fn rate_application_is_exact() {
        let wage = Sgd::parse("1234.56").unwrap();
        let twenty_percent = Rate::from_percent_tenths(200);
        let product = twenty_percent.of(wage);

        // Exactly 24691.2 cents, held without loss.
        assert_eq!(product.round_to_nearest_cent(), Sgd::from_cents(24_691));
        assert_eq!(product.drop_cents(), Sgd::from_dollars(246));
        assert_eq!(product.round_to_nearest_dollar(), Sgd::from_dollars(247));
    }

    /// The exact half-dollar boundary: CPF rounds 50 cents **up**.
    #[test]
    fn rounds_exactly_half_a_dollar_up() {
        // $2.50 at 100% is $2.50 -> $3.
        let half = Rate::ONE.of(Sgd::from_cents(250));
        assert_eq!(half.round_to_nearest_dollar(), Sgd::from_dollars(3));
        assert_eq!(half.drop_cents(), Sgd::from_dollars(2));

        // One cent below the boundary rounds down.
        let just_under = Rate::ONE.of(Sgd::from_cents(249));
        assert_eq!(just_under.round_to_nearest_dollar(), Sgd::from_dollars(2));
    }

    #[test]
    fn rate_is_exact_and_displays_at_bp_precision() {
        assert_eq!(Rate::from_percent_tenths(155).basis_points(), 1550);
        assert_eq!(Rate::from_percent_tenths(155).to_string(), "15.50%");
        assert_eq!(Rate::from_basis_points(6217).to_string(), "62.17%");
        assert_eq!(Rate::ONE.to_string(), "100.00%");

        // Employer 17% + employee 20% == 37%, exactly.
        let total = Rate::from_percent_tenths(170) + Rate::from_percent_tenths(200);
        assert_eq!(total, Rate::from_percent_tenths(370));
    }

    #[test]
    fn money_displays_with_thousands_separators() {
        assert_eq!(Sgd::from_dollars(7_400).to_string(), "$7,400.00");
        assert_eq!(Sgd::from_cents(123_456).to_string(), "$1,234.56");
        assert_eq!(Sgd::from_dollars(426_000).to_string(), "$426,000.00");
        assert_eq!(Sgd::from_cents(5).to_string(), "$0.05");
        assert_eq!(Sgd::from_cents(-505).to_string(), "-$5.05");
    }

    #[test]
    fn negative_residual_clamps_to_zero() {
        let exhausted = Sgd::from_dollars(102_000) - Sgd::from_dollars(120_000);
        assert!(exhausted.is_negative());
        assert_eq!(exhausted.clamp_non_negative(), Sgd::ZERO);
    }
}
