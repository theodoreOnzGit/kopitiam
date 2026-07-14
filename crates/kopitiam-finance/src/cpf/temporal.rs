//! The temporal core: **there is no "current" CPF rate.**
//!
//! # The central modelling claim of this crate
//!
//! Every number CPF publishes is a function of time. The Ordinary Wage ceiling
//! has been $6,000, $6,300, $6,800, $7,400 and $8,000 within a three-year
//! window. Senior-worker contribution rates have stepped up every January since
//! 2022. The Basic Retirement Sum rises for each cohort. Even the *shape* of the
//! scheme changes: the Special Account ceased to exist for members aged 55 and
//! above in January 2025.
//!
//! A codebase that models this as `const OW_CEILING: f64` is not merely
//! imprecise — it is wrong every January, silently, and it has no way to answer
//! a question about the past at all. Payroll corrections, back-pay, and
//! disputes are *all* questions about the past.
//!
//! So the primitive here is not "a rate". It is **"a rate, over a period, from a
//! source"** — [`Dated<T>`] — and the only way to get one out of a
//! [`PolicyTable`] is to say *which date you are asking about*. There is no
//! `latest()`, and that omission is deliberate:
//!
//! * "The latest entry in my table" is not the same as "the rule in force
//!   today". They differ precisely when the table is stale — which is exactly
//!   when you most need to be told.
//! * A `latest()` would be called from a payroll run for March, quietly return
//!   the April rates, and nobody would notice until an audit.
//!
//! If you want today's rate, you pass today's date, and you get an honest
//! [`CpfError::NoRuleInEffect`] if the table has not been curated that far
//! forward.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::cpf::citation::Citation;
use crate::cpf::date::{Date, DateRange};
use crate::cpf::error::CpfError;

/// A value that held over a period, according to a source.
///
/// The three fields are inseparable by construction. You cannot build a `Dated`
/// without a citation, and lookups hand back the whole struct — so a caller
/// physically cannot obtain the number while discarding the evidence for it.
/// That is the enforcement mechanism behind "the citation is part of the
/// answer".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dated<T> {
    pub value: T,
    pub effective: DateRange,
    pub source: Citation,
}

impl<T> Dated<T> {
    pub fn new(value: T, effective: DateRange, source: Citation) -> Self {
        Self {
            value,
            effective,
            source,
        }
    }

    /// Borrows the value. Deliberately a method and not a `Deref`: reaching for
    /// the number should be a visible act.
    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn citation(&self) -> &Citation {
        &self.source
    }

    /// Applies a function to the value, carrying the effective period and
    /// citation through unchanged.
    ///
    /// This is how a derived figure keeps its provenance: projecting the
    /// Ordinary Wage ceiling out of a whole `WageCeilings` table does not
    /// launder away where the ceiling came from.
    pub fn map<U>(&self, f: impl FnOnce(&T) -> U) -> Dated<U> {
        Dated {
            value: f(&self.value),
            effective: self.effective,
            source: self.source.clone(),
        }
    }
}

/// A named, time-indexed set of policy revisions of the same kind.
///
/// Each entry is one *published revision* of the whole table — not one row of
/// it. That mirrors how CPF actually publishes: a single document titled
/// "CPF contribution rates from 1 January 2025" containing every age band. One
/// revision, one effective date, one citation. Splitting it per-row would
/// multiply the citations without adding any information, and would make it
/// possible to have half a table from 2024 and half from 2025.
///
/// Consequently `T` is usually itself a small table
/// (e.g. [`crate::cpf::rates::ContributionSchedule`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyTable<T> {
    name: &'static str,
    entries: Vec<Dated<T>>,
}

impl<T> PolicyTable<T> {
    /// An empty table.
    ///
    /// **An empty table is a legitimate, meaningful state**, not a placeholder
    /// to be filled with guesses. It means "KOPITIAM has no cited data for
    /// this". Every query against it fails loudly with
    /// [`CpfError::NoRuleInEffect`]. That is the correct behaviour, and it is
    /// strictly better than a table full of numbers nobody checked.
    pub fn empty(name: &'static str) -> Self {
        Self {
            name,
            entries: Vec::new(),
        }
    }

    pub fn new(name: &'static str, entries: Vec<Dated<T>>) -> Self {
        Self { name, entries }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[Dated<T>] {
        &self.entries
    }

    /// The rule in force on `date`, with its citation.
    ///
    /// # Errors
    ///
    /// [`CpfError::NoRuleInEffect`] if no entry covers `date`. It does **not**
    /// fall back to the nearest entry, the earliest entry, or the latest entry.
    /// A policy engine that extrapolates is a policy engine that invents policy.
    pub fn on(&self, date: Date) -> Result<&Dated<T>, CpfError> {
        self.entries
            .iter()
            .find(|entry| entry.effective.contains(date))
            .ok_or_else(|| CpfError::NoRuleInEffect {
                table: self.name,
                date,
                coverage: self.coverage(),
            })
    }

    /// Human-readable summary of what this table covers, used in error
    /// messages so a failed lookup immediately tells the user whether they hit
    /// a data gap or fat-fingered a year.
    pub fn coverage(&self) -> String {
        if self.entries.is_empty() {
            return "nothing — this table is deliberately empty (no cited data)".to_string();
        }
        self.entries
            .iter()
            .map(|e| e.effective.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Checks that no two entries claim the same date.
    ///
    /// Overlapping entries mean a query has two correct answers, and [`Self::on`]
    /// would return whichever happened to be first in the `Vec` — a bug that
    /// depends on data-entry order and would never be caught by an ordinary
    /// test. Every built-in table is run through this in the test suite.
    ///
    /// Note we do **not** check for *gaps*. A gap is often the truth: KOPITIAM
    /// may legitimately hold 2025 and 2026 rates but not 2024, and pretending
    /// otherwise by interpolating would be exactly the sin this crate exists to
    /// prevent. Gaps surface as [`CpfError::NoRuleInEffect`] at query time,
    /// which is where they belong.
    pub fn validate(&self) -> Result<(), CpfError> {
        for (i, a) in self.entries.iter().enumerate() {
            for b in &self.entries[i + 1..] {
                if a.effective.overlaps(b.effective) {
                    return Err(CpfError::OverlappingRules {
                        table: self.name,
                        first: a.effective.to_string(),
                        second: b.effective.to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

impl<T> fmt::Display for PolicyTable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.name, self.coverage())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpf::money::Sgd;

    fn d(y: i32, m: u8, day: u8) -> Date {
        Date::new(y, m, day).unwrap()
    }

    fn cite(what: &str) -> Citation {
        Citation::transcribed_from_cpf_board("test document", what)
    }

    /// Two adjacent revisions, as CPF actually publishes them.
    fn ceiling_table() -> PolicyTable<Sgd> {
        PolicyTable::new(
            "test ordinary wage ceiling",
            vec![
                Dated::new(
                    Sgd::from_dollars(6_800),
                    DateRange::between(d(2024, 1, 1), d(2025, 1, 1)).unwrap(),
                    cite("2024 row"),
                ),
                Dated::new(
                    Sgd::from_dollars(7_400),
                    DateRange::between(d(2025, 1, 1), d(2026, 1, 1)).unwrap(),
                    cite("2025 row"),
                ),
            ],
        )
    }

    /// The requirement, stated directly: a 2024 rule must not answer a 2025
    /// question.
    #[test]
    fn a_rule_effective_in_2024_is_not_returned_for_a_2025_query() {
        let table = ceiling_table();

        let in_2024 = table.on(d(2024, 6, 30)).unwrap();
        assert_eq!(in_2024.value, Sgd::from_dollars(6_800));

        let in_2025 = table.on(d(2025, 6, 30)).unwrap();
        assert_eq!(in_2025.value, Sgd::from_dollars(7_400));

        assert_ne!(in_2024.value, in_2025.value);
    }

    /// Adjacency: the boundary date resolves to exactly one revision, and the
    /// day before resolves to the other. This is the off-by-one that a
    /// closed-interval model gets wrong.
    #[test]
    fn adjacent_revisions_meet_cleanly_at_the_boundary() {
        let table = ceiling_table();
        assert_eq!(table.on(d(2024, 12, 31)).unwrap().value, Sgd::from_dollars(6_800));
        assert_eq!(table.on(d(2025, 1, 1)).unwrap().value, Sgd::from_dollars(7_400));
    }

    /// A query before any rule exists is an honest error, never a panic and
    /// never a silent fallback to the earliest entry.
    #[test]
    fn a_date_before_any_rule_is_an_honest_error() {
        let table = ceiling_table();
        let err = table.on(d(2019, 1, 1)).unwrap_err();
        match &err {
            CpfError::NoRuleInEffect { table: t, date, coverage } => {
                assert_eq!(*t, "test ordinary wage ceiling");
                assert_eq!(*date, d(2019, 1, 1));
                assert!(coverage.contains("2024-01-01"), "coverage must tell the user what IS known");
            }
            other => panic!("expected NoRuleInEffect, got {other:?}"),
        }
        // And it must not have quietly handed back the 2024 value.
        assert!(table.on(d(2019, 1, 1)).is_err());
    }

    /// A date *after* the last closed revision is equally an error. A table that
    /// stops at 2026 must not answer for 2030.
    #[test]
    fn a_date_after_the_last_closed_rule_is_an_honest_error() {
        let table = ceiling_table();
        assert!(table.on(d(2026, 1, 1)).is_err());
    }

    /// A gap in the middle is reported, not interpolated across.
    #[test]
    fn a_hole_between_revisions_is_an_honest_error() {
        let table = PolicyTable::new(
            "gappy",
            vec![
                Dated::new(
                    Sgd::from_dollars(1),
                    DateRange::between(d(2020, 1, 1), d(2021, 1, 1)).unwrap(),
                    cite("a"),
                ),
                Dated::new(
                    Sgd::from_dollars(3),
                    DateRange::from(d(2023, 1, 1)),
                    cite("c"),
                ),
            ],
        );
        assert!(table.validate().is_ok(), "a gap is not an overlap");
        assert!(table.on(d(2022, 6, 1)).is_err(), "must not interpolate across the hole");
    }

    #[test]
    fn validation_catches_overlapping_revisions() {
        let table = PolicyTable::new(
            "overlapping",
            vec![
                Dated::new(Sgd::from_dollars(1), DateRange::from(d(2024, 1, 1)), cite("a")),
                Dated::new(Sgd::from_dollars(2), DateRange::from(d(2025, 1, 1)), cite("b")),
            ],
        );
        let err = table.validate().unwrap_err();
        assert!(matches!(err, CpfError::OverlappingRules { .. }));
    }

    #[test]
    fn an_empty_table_answers_nothing_and_says_so() {
        let table: PolicyTable<Sgd> = PolicyTable::empty("unpopulated");
        assert!(table.is_empty());
        assert!(table.validate().is_ok());
        let err = table.on(d(2025, 1, 1)).unwrap_err();
        assert!(err.to_string().contains("deliberately empty"));
    }

    /// The citation travels with the value. A caller cannot get the number
    /// without also holding the evidence.
    #[test]
    fn every_lookup_carries_its_citation() {
        let table = ceiling_table();
        let hit = table.on(d(2025, 3, 1)).unwrap();
        assert_eq!(hit.citation().locator, "2025 row");
        assert_eq!(hit.citation().publisher, "Central Provident Fund Board");
    }

    /// `map` derives a new figure without laundering away its provenance.
    #[test]
    fn map_preserves_provenance() {
        let table = ceiling_table();
        let annualised = table.on(d(2025, 1, 1)).unwrap().map(|c| *c + *c);
        assert_eq!(annualised.value, Sgd::from_dollars(14_800));
        assert_eq!(annualised.source.locator, "2025 row");
        assert_eq!(annualised.effective.start(), d(2025, 1, 1));
    }
}
