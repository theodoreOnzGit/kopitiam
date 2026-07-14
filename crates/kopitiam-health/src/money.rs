//! Currency-checked arithmetic over `kopitiam-insurance`'s money types.
//!
//! # What this adds, and why it is not in `kopitiam-insurance`
//!
//! `kopitiam-insurance` models money exactly right for *extraction*:
//! [`Money`] is integer cents with no currency, and [`MonetaryAmount`] pairs it
//! with the currency **as the document printed it** — which may be
//! [`Currency::Ambiguous`] (a bare `$`) or [`Currency::Unstated`]. That
//! separation is deliberate and correct: an amount is known as soon as the
//! digits parse, but its currency very often is not, and fusing the two would
//! invite a "just default it to SGD" fix.
//!
//! But you cannot *compute* with a `MonetaryAmount`. There is no `add`, and
//! there should not be: what is `$3,500` plus `SGD 1,000`? The honest answer is
//! that it is not a number, and any `add` that returned one would be lying.
//!
//! So the cost-share calculator needs a second, narrower type: an amount whose
//! currency the document actually stated. [`Amount`] is that type, and the
//! conversion into it is the choke point where an unstated or ambiguous currency
//! becomes a **refusal** rather than a guess.
//!
//! That is the whole design:
//!
//! ```text
//!   extraction  ->  MonetaryAmount   (what the document printed, warts and all)
//!   arithmetic  ->  Amount           (currency stated, or we refuse to compute)
//! ```
//!
//! # This module is arguably GENERIC and could be lifted into `kopitiam-insurance`
//!
//! Nothing here is health-specific — a motor policy's excess arithmetic needs
//! exactly this. It lives here because `kopitiam-insurance` is deliberately an
//! extraction engine and has so far had no reason to do arithmetic. If a second
//! domain crate needs it, that is the signal to move it down.

use std::fmt;

use kopitiam_insurance::{Currency, MonetaryAmount, Money, Percentage};

/// Arithmetic that could not be done honestly.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MoneyError {
    /// The document printed an amount without saying what currency it was in
    /// (a bare `$`, or a bare number in a table).
    ///
    /// We refuse rather than assume. `$` is not SGD; it is `$`, and in a
    /// document that also mentions US dollars it is a genuinely open question
    /// that the reader — not this crate — has to close.
    #[error(
        "the document printed {printed} without identifying the currency, so this amount cannot \
         be computed with. Read the clause and establish the currency."
    )]
    CurrencyNotStated {
        /// The amount as printed, for quoting back at the reader.
        printed: String,
    },

    /// Two amounts in different currencies were combined.
    ///
    /// There is no exchange rate in this crate and there never will be: a
    /// policy's limits are stated in one currency, and converting them at a rate
    /// we picked would invent a term the document does not contain.
    #[error("currency mismatch: cannot combine {left} and {right} — this crate holds no exchange rates")]
    CurrencyMismatch {
        /// The left-hand currency.
        left: String,
        /// The right-hand currency.
        right: String,
    },

    /// The arithmetic overflowed `i64` cents. In practice this means a parse
    /// bug, not a real policy limit.
    #[error("monetary overflow")]
    Overflow,
}

/// An amount whose currency the document actually stated.
///
/// The only kind of money this crate is willing to do arithmetic on. Built from
/// a [`MonetaryAmount`] via [`Amount::try_from_extracted`], which is where an
/// ambiguous currency turns into an error instead of an assumption.
///
/// # Deliberately not `Ord`
///
/// Ordering two amounts is only meaningful when they share a currency, and a
/// derived `Ord` would happily tell you that `USD 1.00 < SGD 2.00` — a comparison
/// that is not false so much as meaningless. Worse, `Ord` brings a by-value
/// `min` that silently shadows [`Amount::min`], so a currency-checked comparison
/// would quietly become an unchecked one at the call site with no diagnostic at
/// all. (That is not hypothetical: it happened while this module was being
/// written, which is why the derive is gone and this paragraph is here.)
///
/// Use [`Amount::min`], which returns a `Result`, or compare [`Amount::cents`]
/// once you have established that the currencies match.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Amount {
    cents: Money,
    currency: String,
}

impl Amount {
    /// Accepts an extracted amount for arithmetic — if the document said what
    /// currency it was in.
    ///
    /// # Errors
    ///
    /// [`MoneyError::CurrencyNotStated`] when the document printed a bare `$`,
    /// or a bare number with no currency marker at all.
    pub fn try_from_extracted(amount: &MonetaryAmount) -> Result<Self, MoneyError> {
        match amount.currency() {
            Currency::Iso(code) => Ok(Self {
                cents: amount.amount(),
                currency: code.clone(),
            }),
            Currency::Ambiguous(symbol) => Err(MoneyError::CurrencyNotStated {
                printed: format!("{symbol}{}", amount.amount().to_decimal_string()),
            }),
            Currency::Unstated => Err(MoneyError::CurrencyNotStated {
                printed: amount.amount().to_decimal_string(),
            }),
        }
    }

    /// Builds an amount directly, for callers who know the currency (tests,
    /// and a caller who has read the document's "all amounts are in Singapore
    /// dollars" clause).
    pub fn new(cents: i64, currency: impl Into<String>) -> Self {
        Self {
            cents: Money::from_cents(cents),
            currency: currency.into(),
        }
    }

    /// A whole number of major units: `Amount::major(3_500, "SGD")`.
    pub fn major(units: i64, currency: impl Into<String>) -> Result<Self, MoneyError> {
        let cents = units.checked_mul(100).ok_or(MoneyError::Overflow)?;
        Ok(Self::new(cents, currency))
    }

    /// Zero, in this amount's currency.
    ///
    /// Currency-tagged even at zero: adding "0" of the wrong currency into a
    /// running total is exactly the slip this type exists to catch.
    pub fn zero_like(&self) -> Self {
        Self {
            cents: Money::from_cents(0),
            currency: self.currency.clone(),
        }
    }

    /// The amount in exact cents.
    pub fn cents(&self) -> i64 {
        self.cents.cents()
    }

    /// The currency's ISO code.
    pub fn currency(&self) -> &str {
        &self.currency
    }

    /// True when the amount is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.cents.cents() == 0
    }

    /// Adds two amounts of the same currency.
    pub fn add(&self, other: &Self) -> Result<Self, MoneyError> {
        self.check(other)?;
        self.with_cents(
            self.cents
                .cents()
                .checked_add(other.cents.cents())
                .ok_or(MoneyError::Overflow)?,
        )
    }

    /// Subtracts, flooring at zero.
    ///
    /// This is "the bill remaining after the deductible": a bill smaller than
    /// the deductible leaves nothing to co-insure, not a negative remainder that
    /// would later turn into a payment *to* the insurer.
    pub fn saturating_sub(&self, other: &Self) -> Result<Self, MoneyError> {
        self.check(other)?;
        self.with_cents((self.cents.cents() - other.cents.cents()).max(0))
    }

    /// The smaller of two amounts of the same currency.
    pub fn min(&self, other: &Self) -> Result<Self, MoneyError> {
        self.check(other)?;
        Ok(if self.cents <= other.cents {
            self.clone()
        } else {
            other.clone()
        })
    }

    /// Applies a percentage, reporting whether the result had to be rounded.
    ///
    /// # The rounding rule, and why it is a caveat rather than a decision
    ///
    /// 10% of S$6,543.21 is S$654.321 — not a representable amount of money.
    /// Somebody has to round, and **the wording almost never says who or how.**
    /// `kopitiam-insurance`'s [`Percentage::of`] rounds half away from zero,
    /// which is the schoolbook rule and the least surprising choice.
    ///
    /// But we do not pretend that choice came from the document. The `bool`
    /// returned is `true` whenever rounding actually changed the answer, and
    /// [`crate::cost_share`] turns it into a visible
    /// [`crate::cost_share::Caveat::RoundingRuleNotStated`]. A cent is not
    /// material to a patient; silently inventing a rule the document did not
    /// state, and not saying so, is the habit that makes a tool untrustworthy at
    /// the scale where it *is*.
    pub fn apply(&self, percentage: Percentage) -> Result<(Self, bool), MoneyError> {
        let share = percentage.of(self.cents).ok_or(MoneyError::Overflow)?;

        // `Percentage::of` has already rounded; recompute the exact product to
        // find out whether it had to.
        let exact = i128::from(self.cents.cents()) * i128::from(percentage.basis_points());
        let rounded = exact % 10_000 != 0;

        Ok((
            Self {
                cents: share,
                currency: self.currency.clone(),
            },
            rounded,
        ))
    }

    fn with_cents(&self, cents: i64) -> Result<Self, MoneyError> {
        Ok(Self {
            cents: Money::from_cents(cents),
            currency: self.currency.clone(),
        })
    }

    fn check(&self, other: &Self) -> Result<(), MoneyError> {
        if self.currency != other.currency {
            return Err(MoneyError::CurrencyMismatch {
                left: self.currency.clone(),
                right: other.currency.clone(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for Amount {
    /// `SGD 3,500.00` — with thousands separators, because these numbers are
    /// read by humans checking them against a page.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimal = self.cents.to_decimal_string();
        let (major, minor) = decimal.split_once('.').unwrap_or((decimal.as_str(), "00"));
        let negative = major.starts_with('-');
        let digits = major.trim_start_matches('-');

        let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
        for (i, ch) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i) % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(ch);
        }

        write!(
            f,
            "{} {}{}.{}",
            self.currency,
            if negative { "-" } else { "" },
            grouped,
            minor
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sgd(major: i64) -> Amount {
        Amount::major(major, "SGD").unwrap()
    }

    /// The choke point. A wording that prints a bare `$` gets its amount
    /// extracted and citable — but the calculator will not compute with it.
    #[test]
    fn a_bare_dollar_sign_is_refused_rather_than_assumed_to_be_sgd() {
        let printed = MonetaryAmount::new(
            Money::from_cents(350_000),
            Currency::Ambiguous("$".into()),
        );
        let err = Amount::try_from_extracted(&printed).unwrap_err();
        assert!(matches!(err, MoneyError::CurrencyNotStated { .. }));
    }

    #[test]
    fn an_amount_with_a_stated_currency_is_accepted() {
        let printed =
            MonetaryAmount::new(Money::from_cents(350_000), Currency::Iso("SGD".into()));
        let amount = Amount::try_from_extracted(&printed).unwrap();
        assert_eq!(amount, sgd(3_500));
    }

    #[test]
    fn cross_currency_arithmetic_is_refused_rather_than_converted() {
        let err = sgd(100).add(&Amount::major(100, "USD").unwrap()).unwrap_err();
        assert!(matches!(err, MoneyError::CurrencyMismatch { .. }));
    }

    #[test]
    fn saturating_sub_floors_at_zero() {
        // An S$800 bill against an S$3,500 deductible leaves nothing to
        // co-insure, not minus S$2,700.
        assert!(sgd(800).saturating_sub(&sgd(3_500)).unwrap().is_zero());
    }

    #[test]
    fn percentages_announce_when_they_round() {
        // 10% of S$6,500.00 is exactly S$650.00 — no rounding, no caveat.
        let (share, rounded) = sgd(6_500).apply(Percentage::from_basis_points(1_000)).unwrap();
        assert_eq!(share, sgd(650));
        assert!(!rounded);

        // 10% of S$6,543.21 is S$654.321 — not representable. Rounded, and said so.
        let (share, rounded) = Amount::new(654_321, "SGD")
            .apply(Percentage::from_basis_points(1_000))
            .unwrap();
        assert_eq!(share, Amount::new(65_432, "SGD"));
        assert!(rounded, "a rounded result must announce itself");
    }

    #[test]
    fn amounts_format_for_a_human_checking_them_against_a_page() {
        assert_eq!(sgd(3_500).to_string(), "SGD 3,500.00");
        assert_eq!(Amount::new(654_321, "SGD").to_string(), "SGD 6,543.21");
        assert_eq!(sgd(0).to_string(), "SGD 0.00");
    }
}
