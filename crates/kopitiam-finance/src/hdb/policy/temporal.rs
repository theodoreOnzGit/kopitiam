//! The temporal core: dates, effective-date ranges, and policy tables that are
//! **queried by date**.
//!
//! # Why this module exists at all
//!
//! Almost every number HDB publishes has changed, and will change again. The
//! family income ceiling for a new flat was $10,000, then $12,000, then
//! $14,000. The Minimum Occupation Period was five years until the Prime
//! Location Public Housing model made it ten. Grant amounts are revised at
//! National Day Rallies and Budgets.
//!
//! A codebase that writes
//!
//! ```text
//! const INCOME_CEILING: u32 = 14_000;
//! ```
//!
//! has not modelled a policy. It has photographed one, and the photograph
//! silently rots: it will still answer questions about 2019 and about 2035 with
//! the same number, and it cannot say where the number came from. That is how a
//! system tells a real person they can buy a home when they cannot.
//!
//! So there is no "current" income ceiling in this crate. There is only *the
//! ceiling in force on date D*, and it arrives with the citation that proves
//! it. The temporal dimension is not a feature of these tables; it is their
//! primary key.
//!
//! # The three answers a lookup can give
//!
//! [`Timeline::on`] can return exactly three kinds of outcome, and the third is
//! the one that makes this crate honest:
//!
//! 1. **A rule was in force.** You get the value, its effective range, and its
//!    [`Citation`].
//! 2. **No rule is modelled for that date.** The date precedes the earliest
//!    provision we entered, or follows the last one. An error — never a silent
//!    fallback to the nearest neighbour, which would be a fabrication.
//! 3. **A rule exists, and we deliberately did not enter it.**
//!    [`Provision::NotModelled`] is a first-class span in the timeline. It says
//!    "policy applied here, we know it changed, we are not confident of the
//!    figures, and we refuse to guess." It is the data-structure form of "I
//!    don't know", and it is what propagates up into
//!    [`Eligibility::Indeterminate`](super::eligibility::Eligibility::Indeterminate).
//!
//! Systems that omit the third case do not thereby become more useful. They
//! become confidently wrong, which is worse.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::citation::Citation;

/// A civil (proleptic Gregorian) calendar date: the unit HDB policy changes on.
///
/// Deliberately hand-rolled rather than pulled from `chrono` or `time`:
///
/// * KOPITIAM's Pure Rust Core prefers no dependency to a small one, and this
///   is a genuinely small one — policy dates need ordering and validation, not
///   time zones, leap seconds, or clocks.
/// * It is **deterministic and clock-free by construction**. There is no
///   `Date::today()` here, and that is on purpose: a policy answer that depends
///   on the wall clock is not reproducible, and cannot be tested. The caller
///   supplies the date they are asking about. If they want "today", they must
///   say so explicitly, at the edge of the system.
///
/// # A caveat about what a date *means* to HDB
///
/// HDB rules frequently attach to a **sales exercise**, not to a calendar date:
/// "applies to flat applications from the October 2024 BTO exercise onwards".
/// A date is a faithful proxy for that only because exercises are ordered in
/// time. Where a provision's start is an exercise rather than a gazetted date,
/// the [`Citation`] says so. Do not read `from` as a gazette date unless the
/// citation calls it one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Date {
    // Field order matters: the derived `Ord` is lexicographic over the fields,
    // and year-then-month-then-day is exactly chronological order.
    year: i32,
    month: u8,
    day: u8,
}

/// Why a [`Date`] could not be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DateError {
    /// The month was outside 1..=12.
    #[error("month {0} is not in 1..=12")]
    Month(u32),
    /// The day was outside 1..=(length of that month, in that year).
    #[error("day {day} is not valid for month {month} of {year}")]
    Day {
        /// The rejected year.
        year: i32,
        /// The month whose length the day exceeded.
        month: u8,
        /// The rejected day.
        day: u32,
    },
}

impl Date {
    /// Builds a date, validating the month and the day (leap years included).
    ///
    /// Returns [`DateError`] rather than panicking: dates will eventually be
    /// parsed out of HDB documents by the Document Engine, and a malformed
    /// document is not a programmer error.
    pub fn new(year: i32, month: u32, day: u32) -> Result<Self, DateError> {
        if !(1..=12).contains(&month) {
            return Err(DateError::Month(month));
        }
        let month = month as u8;
        if day < 1 || day > u32::from(days_in_month(year, month)) {
            return Err(DateError::Day { year, month, day });
        }
        Ok(Self {
            year,
            month,
            day: day as u8,
        })
    }

    /// The year.
    pub fn year(self) -> i32 {
        self.year
    }

    /// The month, 1..=12.
    pub fn month(self) -> u32 {
        u32::from(self.month)
    }

    /// The day of the month, 1-based.
    pub fn day(self) -> u32 {
        u32::from(self.day)
    }
}

impl fmt::Display for Date {
    /// ISO-8601 (`2019-09-11`) — unambiguous, sortable, and the format HDB's
    /// own circulars are dated in when they are dated at all.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Length of a Gregorian month, honouring the leap-year rule.
fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => unreachable!("month validated by Date::new"),
    }
}

/// The full Gregorian rule, not the four-yearly approximation.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Convenience constructor for the hand-entered policy tables.
///
/// Panics on an invalid date. That is correct *here and only here*: the
/// arguments are literals written by a programmer, so an invalid one is a bug
/// in this crate, not bad input. Everything reaching this crate from outside
/// goes through [`Date::new`] and gets a [`Result`].
pub(super) fn date(year: i32, month: u32, day: u32) -> Date {
    Date::new(year, month, day).expect("policy table contains an invalid literal date")
}

/// The half-open span `[from, until)` over which a provision is in force.
///
/// **Half-open is not an arbitrary choice.** When HDB says a rule "applies to
/// applications from 11 September 2019", the old rule's last day is the 10th
/// and the new rule's first day is the 11th. A closed range would force every
/// table author to write "10 September" as an end date, and that off-by-one —
/// two adjacent provisions, one boundary, one wrong day — is *the* classic bug
/// in this domain. With `[from, until)` the successor's `from` is literally the
/// predecessor's `until`: the same date appears once, in both, and cannot
/// disagree with itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveRange {
    /// First date on which the provision applies (inclusive).
    pub from: Date,
    /// First date on which it no longer applies (exclusive). `None` means "in
    /// force with no announced end" — *not* "in force forever".
    pub until: Option<Date>,
}

impl EffectiveRange {
    /// A provision that began on `from` and has no announced end date.
    pub fn from(from: Date) -> Self {
        Self { from, until: None }
    }

    /// A provision in force over `[from, until)`.
    pub fn between(from: Date, until: Date) -> Self {
        Self {
            from,
            until: Some(until),
        }
    }

    /// A provision whose commencement we cannot cite, but which we can show was
    /// already in force by `anchor`.
    ///
    /// Several HDB rules (the minimum age of 21 under the Public Scheme, the 35
    /// of the Single Singapore Citizen Scheme) are long-standing, and their
    /// commencement dates are not something this crate can establish offline.
    /// The honest encoding is *not* to invent a start date, and *not* to leave
    /// the rule out: it is to start the provision at the earliest date we can
    /// actually cite the rule being in force, and let a query about any earlier
    /// date fail loudly with [`TemporalError::BeforeEarliestProvision`].
    ///
    /// Semantically identical to [`EffectiveRange::from`]; the distinct
    /// constructor exists so the *intent* survives in the source, which is the
    /// point of the whole crate.
    pub fn in_force_at_least_from(anchor: Date) -> Self {
        Self::from(anchor)
    }

    /// Whether `date` falls inside `[from, until)`.
    pub fn contains(&self, date: Date) -> bool {
        date >= self.from && self.until.is_none_or(|until| date < until)
    }
}

impl fmt::Display for EffectiveRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.until {
            Some(until) => write!(f, "{} to {} (exclusive)", self.from, until),
            None => write!(f, "{} onwards", self.from),
        }
    }
}

/// A value that is only meaningful together with *when it applied* and *where it
/// is written down*.
///
/// This is the crate's central type. Nothing that came out of a policy document
/// is ever handed to a caller bare: it arrives as a `Dated<T>`, so the answer to
/// "why?" is always "§X of document Y, effective Z" and never "because the code
/// says so". The [`Citation`] is part of the answer, not metadata attached to
/// it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dated<T> {
    /// The figure, rule, or schedule itself.
    pub value: T,
    /// When it applied.
    pub effective: EffectiveRange,
    /// Where it is written down.
    pub citation: Citation,
}

impl<T> Dated<T> {
    /// Pairs a value with its effective range and citation.
    pub fn new(value: T, effective: EffectiveRange, citation: Citation) -> Self {
        Self {
            value,
            effective,
            citation,
        }
    }
}

/// A span of a [`Timeline`]: either a rule we modelled, or an admission that we
/// did not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Provision<T> {
    /// A rule that was in force, with its value and citation.
    InForce(Dated<T>),
    /// A span in which policy certainly applied, and which this crate
    /// deliberately does not model.
    ///
    /// Reached when we know a rule changed (for instance, the Enhanced CPF
    /// Housing Grant was raised at the 2024 National Day Rally) but are not
    /// confident of the resulting figures. Entering a plausible-looking number
    /// here would be the single worst thing this crate could do, because it
    /// would be believed. A [`TemporalError::NotModelled`] is the correct
    /// output, and it carries `reason` so the caller learns *what* is missing
    /// rather than merely that something is.
    NotModelled {
        /// The span we are declining to model.
        effective: EffectiveRange,
        /// What applies here, and why we did not enter it.
        reason: String,
        /// Where the change was announced, when we know that much even though
        /// we do not know the figures.
        announced_in: Option<Citation>,
    },
}

impl<T> Provision<T> {
    /// The span this provision covers, whether modelled or not.
    pub fn effective(&self) -> EffectiveRange {
        match self {
            Provision::InForce(dated) => dated.effective,
            Provision::NotModelled { effective, .. } => *effective,
        }
    }
}

/// Why a policy lookup produced no figure.
///
/// Every variant is a refusal to make something up. None of them is recoverable
/// by retrying with a nearby date — that would be exactly the fabrication these
/// errors exist to prevent.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TemporalError {
    /// The date precedes every provision we entered.
    #[error(
        "{table}: no provision is modelled on {date}; the earliest modelled provision starts {earliest}"
    )]
    BeforeEarliestProvision {
        /// The table queried.
        table: String,
        /// The date asked about.
        date: Date,
        /// Start of the earliest provision we hold.
        earliest: Date,
    },
    /// The date follows the last provision we entered, and that provision had a
    /// stated end. (A rule with no announced end covers every later date, so
    /// this can only arise where we recorded an end and no successor.)
    #[error(
        "{table}: no provision is modelled on {date}; the latest modelled provision ended {latest_end}"
    )]
    AfterLatestProvision {
        /// The table queried.
        table: String,
        /// The date asked about.
        date: Date,
        /// End of the last provision we hold.
        latest_end: Date,
    },
    /// Policy applied on that date, and we deliberately did not model it.
    ///
    /// The honest answer, and the one that becomes
    /// [`Eligibility::Indeterminate`](super::eligibility::Eligibility::Indeterminate).
    #[error("{table}: policy on {date} is deliberately not modelled: {reason}")]
    NotModelled {
        /// The table queried.
        table: String,
        /// The date asked about.
        date: Date,
        /// What applies there, and why we declined to enter it.
        reason: String,
    },
    /// The table has no provision for that key at all (e.g. asking for the
    /// income ceiling of a scheme this crate does not model).
    #[error("{table}: no provisions are modelled for {key}")]
    UnknownKey {
        /// The table queried.
        table: String,
        /// The key, rendered for a human.
        key: String,
    },
    /// The table is empty. Only reachable for a table declared but never
    /// populated.
    #[error("{table}: the table is empty")]
    Empty {
        /// The table queried.
        table: String,
    },
}

/// Why a [`Timeline`] could not be constructed. These are programmer errors in
/// the policy tables, caught at construction rather than at query time, so a
/// contradictory table can never be *served*.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TableError {
    /// Two provisions claim the same date. Whichever we returned would be
    /// arbitrary, so we return neither and refuse to build the table.
    #[error("{table}: provisions overlap: {first} and {second} both cover part of the same span")]
    Overlap {
        /// The table being built.
        table: String,
        /// The earlier provision's range.
        first: EffectiveRange,
        /// The later provision's range.
        second: EffectiveRange,
    },
    /// A provision's `until` is not after its `from`.
    #[error("{table}: provision {range} ends before it begins")]
    InvertedRange {
        /// The table being built.
        table: String,
        /// The offending range.
        range: EffectiveRange,
    },
}

/// The history of one rule: a chronologically ordered, non-overlapping sequence
/// of [`Provision`]s.
///
/// Gaps between provisions are permitted and meaningful — they say "we model
/// nothing here", and a query lands on [`TemporalError::AfterLatestProvision`]
/// or [`TemporalError::BeforeEarliestProvision`]. Overlaps are *not* permitted,
/// because two rules in force on one date is not a policy, it is a bug.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline<T> {
    name: String,
    provisions: Vec<Provision<T>>,
}

impl<T> Timeline<T> {
    /// Builds a timeline, sorting provisions by start date and rejecting
    /// overlaps and inverted ranges.
    pub fn new(
        name: impl Into<String>,
        mut provisions: Vec<Provision<T>>,
    ) -> Result<Self, TableError> {
        let name = name.into();
        provisions.sort_by_key(|p| p.effective().from);

        for provision in &provisions {
            let range = provision.effective();
            if let Some(until) = range.until
                && until <= range.from
            {
                return Err(TableError::InvertedRange { table: name, range });
            }
        }

        for pair in provisions.windows(2) {
            let (first, second) = (pair[0].effective(), pair[1].effective());
            // Sorted by `from`, so the only possible overlap is the earlier
            // provision running past the later one's start. An open-ended
            // earlier provision overlaps anything that follows it.
            let overlaps = match first.until {
                Some(until) => until > second.from,
                None => true,
            };
            if overlaps {
                return Err(TableError::Overlap {
                    table: name,
                    first,
                    second,
                });
            }
        }

        Ok(Self { name, provisions })
    }

    /// The table's name, as it appears in errors and in emitted ontology facts.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every provision, modelled or not, in chronological order.
    pub fn provisions(&self) -> &[Provision<T>] {
        &self.provisions
    }

    /// **The** query: what rule was in force on `date`?
    ///
    /// There is no `latest()`, and there will not be one. "The current ceiling"
    /// is not a question this crate can answer, because the crate does not know
    /// what day it is and, more importantly, because the last row of a table is
    /// only "current" until the day it isn't.
    pub fn on(&self, date: Date) -> Result<&Dated<T>, TemporalError> {
        let Some(first) = self.provisions.first() else {
            return Err(TemporalError::Empty {
                table: self.name.clone(),
            });
        };

        if date < first.effective().from {
            return Err(TemporalError::BeforeEarliestProvision {
                table: self.name.clone(),
                date,
                earliest: first.effective().from,
            });
        }

        for provision in &self.provisions {
            if !provision.effective().contains(date) {
                continue;
            }
            return match provision {
                Provision::InForce(dated) => Ok(dated),
                Provision::NotModelled { reason, .. } => Err(TemporalError::NotModelled {
                    table: self.name.clone(),
                    date,
                    reason: reason.clone(),
                }),
            };
        }

        // Past every provision, or inside a gap between two of them. Both mean
        // the same thing to a caller: we hold nothing for that date.
        let latest_end = self
            .provisions
            .last()
            .and_then(|p| p.effective().until)
            .unwrap_or(first.effective().from);
        Err(TemporalError::AfterLatestProvision {
            table: self.name.clone(),
            date,
            latest_end,
        })
    }
}

/// A family of timelines, one per key — because most HDB rules are not a single
/// history but a history *per scheme*, *per flat type*, *per ethnic group*.
///
/// The income ceiling is not one number over time; it is one number per
/// (purchase mode, household class) over time. Flattening that into a single
/// timeline is how a system ends up quoting a family's ceiling to a single
/// applicant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyTable<K: Ord, V> {
    name: String,
    timelines: BTreeMap<K, Timeline<V>>,
}

impl<K: Ord + fmt::Debug, V> PolicyTable<K, V> {
    /// Builds a keyed table from `(key, provisions)` pairs.
    pub fn new(
        name: impl Into<String>,
        entries: impl IntoIterator<Item = (K, Vec<Provision<V>>)>,
    ) -> Result<Self, TableError> {
        let name = name.into();
        let mut timelines = BTreeMap::new();
        for (key, provisions) in entries {
            let timeline = Timeline::new(format!("{name} [{key:?}]"), provisions)?;
            timelines.insert(key, timeline);
        }
        Ok(Self { name, timelines })
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The rule in force for `key` on `date`.
    ///
    /// An unknown key is an error, never an empty answer: "we hold no ceiling
    /// for the Orphans Scheme" must not be indistinguishable from "the Orphans
    /// Scheme has no ceiling".
    pub fn on(&self, key: &K, date: Date) -> Result<&Dated<V>, TemporalError> {
        let timeline = self
            .timelines
            .get(key)
            .ok_or_else(|| TemporalError::UnknownKey {
                table: self.name.clone(),
                key: format!("{key:?}"),
            })?;
        timeline.on(date)
    }

    /// Every timeline in the table, for reporting and ontology emission.
    pub fn timelines(&self) -> impl Iterator<Item = (&K, &Timeline<V>)> {
        self.timelines.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::policy::citation::Citation;

    fn cite() -> Citation {
        Citation::hdb_infoweb("Test", "§1", date(2019, 9, 11), "https://example.invalid")
    }

    fn in_force(v: u32, from: Date, until: Option<Date>) -> Provision<u32> {
        Provision::InForce(Dated::new(v, EffectiveRange { from, until }, cite()))
    }

    #[test]
    fn rejects_invalid_dates_without_panicking() {
        assert!(Date::new(2024, 13, 1).is_err());
        assert!(Date::new(2024, 2, 30).is_err());
        // 2024 is a leap year; 2023 is not. The rule is not "divisible by 4".
        assert!(Date::new(2024, 2, 29).is_ok());
        assert!(Date::new(2023, 2, 29).is_err());
        assert!(
            Date::new(2100, 2, 29).is_err(),
            "1900/2100 are not leap years"
        );
        assert!(Date::new(2000, 2, 29).is_ok(), "2000 is a leap year");
    }

    #[test]
    fn dates_order_chronologically() {
        assert!(date(2019, 9, 10) < date(2019, 9, 11));
        assert!(date(2019, 12, 31) < date(2020, 1, 1));
        assert_eq!(date(2019, 9, 11).to_string(), "2019-09-11");
    }

    #[test]
    fn half_open_range_excludes_its_end() {
        let r = EffectiveRange::between(date(2015, 8, 24), date(2019, 9, 11));
        assert!(r.contains(date(2015, 8, 24)), "start is inclusive");
        assert!(r.contains(date(2019, 9, 10)), "day before end is inside");
        assert!(!r.contains(date(2019, 9, 11)), "end is exclusive");
    }

    #[test]
    fn adjacent_provisions_hand_over_on_the_boundary_day() {
        let t = Timeline::new(
            "ceiling",
            vec![
                in_force(12_000, date(2015, 8, 24), Some(date(2019, 9, 11))),
                in_force(14_000, date(2019, 9, 11), None),
            ],
        )
        .unwrap();

        assert_eq!(t.on(date(2019, 9, 10)).unwrap().value, 12_000);
        assert_eq!(t.on(date(2019, 9, 11)).unwrap().value, 14_000);
    }

    #[test]
    fn overlapping_provisions_are_rejected_at_construction() {
        let err = Timeline::new(
            "ceiling",
            vec![
                in_force(12_000, date(2015, 8, 24), Some(date(2019, 12, 31))),
                in_force(14_000, date(2019, 9, 11), None),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, TableError::Overlap { .. }));
    }

    #[test]
    fn an_open_ended_provision_cannot_be_followed_by_another() {
        let err = Timeline::new(
            "ceiling",
            vec![
                in_force(12_000, date(2015, 8, 24), None),
                in_force(14_000, date(2019, 9, 11), None),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, TableError::Overlap { .. }));
    }

    #[test]
    fn inverted_ranges_are_rejected_at_construction() {
        let err = Timeline::new(
            "ceiling",
            vec![in_force(1, date(2019, 9, 11), Some(date(2015, 8, 24)))],
        )
        .unwrap_err();
        assert!(matches!(err, TableError::InvertedRange { .. }));
    }

    #[test]
    fn a_date_before_every_provision_is_an_error_not_the_earliest_value() {
        let t = Timeline::new("ceiling", vec![in_force(14_000, date(2019, 9, 11), None)]).unwrap();
        let err = t.on(date(2000, 1, 1)).unwrap_err();
        assert!(
            matches!(err, TemporalError::BeforeEarliestProvision { earliest, .. } if earliest == date(2019, 9, 11))
        );
    }

    #[test]
    fn a_date_after_a_closed_provision_is_an_error_not_the_latest_value() {
        let t = Timeline::new(
            "ceiling",
            vec![in_force(12_000, date(2015, 8, 24), Some(date(2019, 9, 11)))],
        )
        .unwrap();
        let err = t.on(date(2025, 1, 1)).unwrap_err();
        assert!(matches!(err, TemporalError::AfterLatestProvision { .. }));
    }

    #[test]
    fn an_unmodelled_span_reports_why_rather_than_returning_a_number() {
        let t = Timeline::new(
            "grant",
            vec![
                in_force(80_000, date(2019, 9, 11), Some(date(2024, 8, 20))),
                Provision::NotModelled {
                    effective: EffectiveRange::from(date(2024, 8, 20)),
                    reason: "raised at NDR 2024; figures not entered".to_string(),
                    announced_in: None,
                },
            ],
        )
        .unwrap();

        assert_eq!(t.on(date(2024, 8, 19)).unwrap().value, 80_000);
        let err = t.on(date(2024, 8, 20)).unwrap_err();
        match err {
            TemporalError::NotModelled { reason, .. } => assert!(reason.contains("NDR 2024")),
            other => panic!("expected NotModelled, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_timeline_errors_rather_than_panicking() {
        let t: Timeline<u32> = Timeline::new("nothing", vec![]).unwrap();
        assert!(matches!(
            t.on(date(2025, 1, 1)).unwrap_err(),
            TemporalError::Empty { .. }
        ));
    }

    #[test]
    fn an_unknown_key_is_distinguishable_from_no_rule() {
        let table: PolicyTable<u8, u32> = PolicyTable::new(
            "ceilings",
            vec![(1u8, vec![in_force(14_000, date(2019, 9, 11), None)])],
        )
        .unwrap();

        assert_eq!(table.on(&1, date(2025, 1, 1)).unwrap().value, 14_000);
        assert!(matches!(
            table.on(&2, date(2025, 1, 1)).unwrap_err(),
            TemporalError::UnknownKey { .. }
        ));
    }
}
