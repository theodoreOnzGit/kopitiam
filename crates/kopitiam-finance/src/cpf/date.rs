//! Civil dates and half-open date ranges.
//!
//! # Why a hand-rolled date type
//!
//! CPF policy is expressed in *civil dates*: "with effect from 1 January 2025".
//! There is no instant, no timezone, no clock. A `DateTime<Utc>` would be a
//! lie — it would imply a precision the domain does not have, and it would
//! invite bugs where the same policy resolves differently depending on where
//! the machine is.
//!
//! We therefore model exactly what the domain has: a proleptic Gregorian
//! year/month/day, totally ordered, with no arithmetic beyond what policy
//! lookups need. This also keeps the Pure Rust Core promise with **zero new
//! dependencies** (CLAUDE.md: "Avoid unnecessary dependencies"). If KOPITIAM
//! later needs real calendar arithmetic across the workspace, swapping this
//! for `jiff`/`time` is a contained change: the type is opaque and every
//! construction goes through [`Date::new`].
//!
//! Determinism note: there is deliberately **no `Date::today()`**. A policy
//! engine that can read the wall clock is a policy engine whose results are
//! not reproducible. The caller always supplies the date they are asking
//! about — which is also the only honest framing, since "what is the CPF rate"
//! is not a well-formed question. "What was the CPF rate on 2025-03-01" is.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::cpf::error::CpfError;

/// A civil (calendar) date in the proleptic Gregorian calendar.
///
/// Ordering is chronological, which is what makes [`DateRange`] containment a
/// pair of comparisons. Construct via [`Date::new`], which rejects impossible
/// dates — an unvalidated `2025-02-30` in a policy table would silently skew
/// every lookup around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Date {
    // Field order is load-bearing: the derived `Ord` compares year, then
    // month, then day, which is exactly chronological order.
    year: i32,
    month: u8,
    day: u8,
}

impl Date {
    /// Constructs a date, rejecting impossible calendar values.
    ///
    /// # Errors
    ///
    /// Returns [`CpfError::InvalidDate`] if the month is not 1..=12 or the day
    /// is not valid for that month and year (leap years included).
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, CpfError> {
        if !(1..=12).contains(&month) {
            return Err(CpfError::InvalidDate {
                year,
                month,
                day,
                reason: "month must be in 1..=12",
            });
        }
        if day < 1 || day > days_in_month(year, month) {
            return Err(CpfError::InvalidDate {
                year,
                month,
                day,
                reason: "day is out of range for that month",
            });
        }
        Ok(Self { year, month, day })
    }

    /// The first day of the given month. The common case for policy effective
    /// dates, which almost always fall on the 1st.
    pub fn first_of(year: i32, month: u8) -> Result<Self, CpfError> {
        Self::new(year, month, 1)
    }

    pub fn year(self) -> i32 {
        self.year
    }

    pub fn month(self) -> u8 {
        self.month
    }

    pub fn day(self) -> u8 {
        self.day
    }

    /// The first day of the month *after* this date's month.
    ///
    /// This exists because of one specific CPF rule, and it is worth stating
    /// plainly: **a member's contribution rate does not change on their
    /// birthday.** It changes on the first day of the month *following* the
    /// month in which they had the birthday. See
    /// [`crate::cpf::structure::contribution_age`].
    pub fn start_of_next_month(self) -> Self {
        if self.month == 12 {
            Self {
                year: self.year + 1,
                month: 1,
                day: 1,
            }
        } else {
            Self {
                year: self.year,
                month: self.month + 1,
                day: 1,
            }
        }
    }
}

impl fmt::Display for Date {
    /// ISO 8601 (`2025-01-01`) — unambiguous, sorts lexicographically, and is
    /// what citations should carry.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// A half-open interval of dates: `[from, until)`.
///
/// # Why half-open
///
/// Because CPF policy revisions are *adjacent*, not overlapping. The Ordinary
/// Wage ceiling was $6,800 "from 1 Jan 2024" and $7,400 "from 1 Jan 2025".
/// With a half-open range, those two facts are `[2024-01-01, 2025-01-01)` and
/// `[2025-01-01, None)` — the boundary date belongs to exactly one of them, by
/// construction. With an inclusive `until`, every adjacency requires someone
/// to remember to write `2024-12-31`, and the day a revision lands on a
/// non-month-end (or someone writes `2025-01-01` in both) you get a silent
/// overlap and a lookup that returns whichever entry happened to be first in
/// the `Vec`.
///
/// [`crate::cpf::temporal::PolicyTable::validate`] enforces non-overlap, but
/// the representation is what makes overlap *unlikely* rather than merely
/// *detected*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DateRange {
    /// First date on which the rule has effect (inclusive).
    from: Date,
    /// First date on which the rule *no longer* has effect (exclusive).
    /// `None` means "still in force as of the last time this table was
    /// curated" — which is emphatically not the same as "in force forever".
    until: Option<Date>,
}

impl DateRange {
    /// A rule in force from `from` until superseded.
    pub fn from(from: Date) -> Self {
        Self { from, until: None }
    }

    /// A rule in force over `[from, until)`.
    ///
    /// # Errors
    ///
    /// Returns [`CpfError::EmptyDateRange`] if `until <= from`. An empty range
    /// is always a data-entry mistake, and a table containing one would have a
    /// silent hole in it.
    pub fn between(from: Date, until: Date) -> Result<Self, CpfError> {
        if until <= from {
            return Err(CpfError::EmptyDateRange { from, until });
        }
        Ok(Self {
            from,
            until: Some(until),
        })
    }

    pub fn start(self) -> Date {
        self.from
    }

    pub fn end(self) -> Option<Date> {
        self.until
    }

    /// Whether `date` falls within `[from, until)`.
    pub fn contains(self, date: Date) -> bool {
        date >= self.from && self.until.is_none_or(|until| date < until)
    }

    /// Whether two ranges share any date. Used by table validation; two
    /// entries in the same table that overlap mean the same query has two
    /// answers, which is a bug in the *data*, not the code.
    pub fn overlaps(self, other: Self) -> bool {
        let starts_before_other_ends = self.until.is_none_or(|u| u > other.from);
        let other_starts_before_self_ends = other.until.is_none_or(|u| u > self.from);
        starts_before_other_ends && other_starts_before_self_ends
    }
}

impl fmt::Display for DateRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.until {
            Some(until) => write!(f, "{} to {} (exclusive)", self.from, until),
            None => write!(f, "{} onwards", self.from),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u8, day: u8) -> Date {
        Date::new(y, m, day).expect("valid test date")
    }

    #[test]
    fn rejects_impossible_dates() {
        assert!(Date::new(2025, 2, 29).is_err(), "2025 is not a leap year");
        assert!(Date::new(2024, 2, 29).is_ok(), "2024 is a leap year");
        assert!(Date::new(2000, 2, 29).is_ok(), "2000 is a leap year (÷400)");
        assert!(Date::new(1900, 2, 29).is_err(), "1900 is not (÷100, not ÷400)");
        assert!(Date::new(2025, 13, 1).is_err());
        assert!(Date::new(2025, 0, 1).is_err());
        assert!(Date::new(2025, 4, 31).is_err());
        assert!(Date::new(2025, 1, 0).is_err());
    }

    #[test]
    fn orders_chronologically() {
        assert!(d(2024, 12, 31) < d(2025, 1, 1));
        assert!(d(2025, 1, 31) < d(2025, 2, 1));
        assert!(d(2025, 1, 1) < d(2025, 1, 2));
    }

    #[test]
    fn half_open_boundary_belongs_to_exactly_one_range() {
        let earlier = DateRange::between(d(2024, 1, 1), d(2025, 1, 1)).unwrap();
        let later = DateRange::from(d(2025, 1, 1));

        // The boundary date itself.
        assert!(!earlier.contains(d(2025, 1, 1)));
        assert!(later.contains(d(2025, 1, 1)));

        // One day either side.
        assert!(earlier.contains(d(2024, 12, 31)));
        assert!(!later.contains(d(2024, 12, 31)));

        assert!(!earlier.overlaps(later));
        assert!(!later.overlaps(earlier));
    }

    #[test]
    fn detects_overlap() {
        let a = DateRange::between(d(2024, 1, 1), d(2025, 6, 1)).unwrap();
        let b = DateRange::from(d(2025, 1, 1));
        assert!(a.overlaps(b));
        assert!(b.overlaps(a));

        // Two open-ended ranges always overlap.
        let c = DateRange::from(d(2030, 1, 1));
        assert!(b.overlaps(c));
    }

    #[test]
    fn rejects_empty_range() {
        assert!(DateRange::between(d(2025, 1, 1), d(2025, 1, 1)).is_err());
        assert!(DateRange::between(d(2025, 1, 2), d(2025, 1, 1)).is_err());
    }

    #[test]
    fn start_of_next_month_rolls_the_year() {
        assert_eq!(d(2025, 12, 31).start_of_next_month(), d(2026, 1, 1));
        assert_eq!(d(2025, 12, 1).start_of_next_month(), d(2026, 1, 1));
        assert_eq!(d(2025, 3, 15).start_of_next_month(), d(2025, 4, 1));
        assert_eq!(d(2024, 2, 29).start_of_next_month(), d(2024, 3, 1));
    }

    #[test]
    fn displays_iso() {
        assert_eq!(d(2025, 1, 1).to_string(), "2025-01-01");
        assert_eq!(d(2025, 12, 31).to_string(), "2025-12-31");
    }
}
