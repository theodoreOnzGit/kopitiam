//! A minimal, dependency-free civil (calendar) date.
//!
//! # Why not `chrono` or `time`?
//!
//! Because we need almost nothing. Legal temporality is coarse: an Act
//! commences on a *date*, a clause is in force *from* a date, a judgment is
//! handed down *on* a date. There are no timezones, no clocks, no leap
//! seconds and no sub-day resolution anywhere in this crate's domain — a
//! provision does not come into force at 14:32 UTC, it comes into force on
//! 1 January 2020. Pulling in a general-purpose datetime crate to represent
//! a year/month/day triple would buy us a large API surface we would then
//! have to *forbid* people from using, because "now()" is exactly the kind
//! of ambient, non-deterministic input this crate must never touch.
//!
//! There is a second, sharper reason. CLAUDE.md requires deterministic
//! behaviour. A `Date` that cannot be constructed from the system clock is
//! a `Date` that cannot silently make an as-at query mean "today" — and
//! "today" is the single most dangerous default in a legal research tool,
//! because it turns a reproducible answer into one that changes underneath
//! the reader. So: no `Date::today()`. Callers who want the current date
//! must obtain it themselves and pass it in *explicitly*, which makes the
//! non-determinism visible at the call site where it belongs.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::LegalError;

/// A proleptic-Gregorian calendar date: year, month, day, no time, no zone.
///
/// Ordering is chronological (derived, because the field order
/// year/month/day makes the lexicographic derive coincide with the
/// chronological one — this is load-bearing, do not reorder the fields).
///
/// Construction is fallible and validating: there is no way to build a
/// 31 February.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}

impl Date {
    /// Builds a date, rejecting impossible ones (month 0 or >12, day 0, day
    /// past the end of the month, 29 February in a common year).
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, LegalError> {
        if !(1..=12).contains(&month) {
            return Err(LegalError::InvalidDate {
                detail: format!("month {month} is not in 1..=12"),
            });
        }
        let max = days_in_month(year, month);
        if day == 0 || day > max {
            return Err(LegalError::InvalidDate {
                detail: format!("day {day} is not in 1..={max} for {year}-{month:02}"),
            });
        }
        Ok(Self { year, month, day })
    }

    pub fn year(&self) -> i32 {
        self.year
    }

    pub fn month(&self) -> u8 {
        self.month
    }

    pub fn day(&self) -> u8 {
        self.day
    }
}

/// Days in a given month, honouring the Gregorian leap rule.
fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Gregorian leap rule: divisible by 4, except centuries, except those
/// divisible by 400.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

impl fmt::Display for Date {
    /// ISO 8601 extended calendar date, `YYYY-MM-DD`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl FromStr for Date {
    type Err = LegalError;

    /// Parses ISO 8601 `YYYY-MM-DD`. Deliberately the *only* accepted form:
    /// legal documents write dates a dozen ways ("1st January 2020",
    /// "01/02/2020" — which is ambiguous between two continents), and this
    /// type is the internal canonical representation, not a natural-language
    /// date parser. Recognising dates in document *prose* is a separate
    /// problem, and one where guessing wrong silently is exactly the failure
    /// mode this crate exists to avoid.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let invalid = || LegalError::InvalidDate {
            detail: format!("expected ISO 8601 YYYY-MM-DD, got {s:?}"),
        };
        let (year, rest) = s.split_once('-').ok_or_else(invalid)?;
        let (month, day) = rest.split_once('-').ok_or_else(invalid)?;
        if year.len() != 4 || month.len() != 2 || day.len() != 2 {
            return Err(invalid());
        }
        Date::new(
            year.parse().map_err(|_| invalid())?,
            month.parse().map_err(|_| invalid())?,
            day.parse().map_err(|_| invalid())?,
        )
    }
}

impl TryFrom<String> for Date {
    type Error = LegalError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<Date> for String {
    fn from(d: Date) -> String {
        d.to_string()
    }
}

/// The date a legal question is asked *about* — "what did section 12 say
/// **as at** 3 March 2021?".
///
/// This is a distinct newtype from [`Date`], and that is not ceremony. An
/// as-at date and a commencement date are both `Date`s structurally, but
/// they play opposite roles: one is a *query*, the other is a property of
/// the *instrument*. Passing one where the other is meant is a silent,
/// plausible-looking bug that produces a confidently wrong answer, which is
/// the precise category of failure this crate is built to prevent. The
/// newtype makes that swap a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AsAtDate(Date);

impl AsAtDate {
    pub fn new(date: Date) -> Self {
        Self(date)
    }

    pub fn date(&self) -> Date {
        self.0
    }
}

impl From<Date> for AsAtDate {
    fn from(date: Date) -> Self {
        Self(date)
    }
}

impl fmt::Display for AsAtDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_impossible_dates() {
        assert!(Date::new(2021, 2, 29).is_err(), "2021 is not a leap year");
        assert!(Date::new(2021, 13, 1).is_err());
        assert!(Date::new(2021, 0, 1).is_err());
        assert!(Date::new(2021, 4, 31).is_err(), "April has 30 days");
        assert!(Date::new(2021, 1, 0).is_err());
    }

    #[test]
    fn honours_the_gregorian_leap_rule() {
        assert!(Date::new(2020, 2, 29).is_ok(), "2020: divisible by 4");
        assert!(Date::new(2000, 2, 29).is_ok(), "2000: divisible by 400");
        assert!(Date::new(1900, 2, 29).is_err(), "1900: century, not /400");
    }

    #[test]
    fn orders_chronologically() {
        let a = Date::new(2020, 1, 1).unwrap();
        let b = Date::new(2020, 2, 1).unwrap();
        let c = Date::new(2021, 1, 1).unwrap();
        assert!(a < b && b < c);
    }

    #[test]
    fn round_trips_iso_8601() {
        let d: Date = "2020-02-29".parse().unwrap();
        assert_eq!(d.to_string(), "2020-02-29");
        assert_eq!((d.year(), d.month(), d.day()), (2020, 2, 29));
    }

    #[test]
    fn rejects_non_iso_forms_rather_than_guessing() {
        // 01/02/2020 is 1 February in London and 2 January in New York.
        // Guessing is how a legal tool produces a confidently wrong answer.
        assert!("01/02/2020".parse::<Date>().is_err());
        assert!("1st January 2020".parse::<Date>().is_err());
        assert!("2020-1-1".parse::<Date>().is_err(), "must be zero-padded");
    }

    #[test]
    fn round_trips_through_json() {
        let d = Date::new(2024, 7, 14).unwrap();
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, "\"2024-07-14\"");
        assert_eq!(serde_json::from_str::<Date>(&json).unwrap(), d);
    }

    #[test]
    fn serde_cannot_smuggle_in_an_invalid_date() {
        // The validating `TryFrom<String>` is on the deserialize path, so a
        // hand-edited JSON file cannot introduce a 31 February.
        assert!(serde_json::from_str::<Date>("\"2021-02-31\"").is_err());
    }
}
