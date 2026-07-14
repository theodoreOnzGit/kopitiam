//! Errors.
//!
//! # The most important variant is [`CpfError::NotPopulated`]
//!
//! This crate ships a *deliberately incomplete* slice of CPF policy. The
//! alternative — filling the gaps with plausible-looking numbers — is the
//! failure mode this whole crate is built to avoid. A CPF engine that
//! confidently returns a wrong contribution rate is worse than useless; it is
//! actively harmful, because the user has no way to tell.
//!
//! So every gap is a *typed, loud, documented* error carrying the reason it is
//! a gap. `NotPopulated` is not a bug. It is the crate working correctly.

use crate::cpf::date::Date;

/// Anything that can go wrong looking up or computing CPF policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CpfError {
    /// The date is not a real calendar date.
    #[error("invalid date {year:04}-{month:02}-{day:02}: {reason}")]
    InvalidDate {
        year: i32,
        month: u8,
        day: u8,
        reason: &'static str,
    },

    /// A date range that ends before (or when) it begins.
    #[error("empty date range: {from} to {until}")]
    EmptyDateRange { from: Date, until: Date },

    /// A monetary string that could not be parsed exactly.
    #[error("could not parse `{input}` as an exact SGD amount")]
    InvalidMoney { input: String },

    /// The query date falls before the earliest entry in a policy table — or
    /// in a hole between entries.
    ///
    /// This is an honest "I do not know", never a fallback to the nearest
    /// entry. Reaching for the nearest rate is how a 2019 wage ceiling ends up
    /// applied to a 2026 payslip.
    #[error(
        "no `{table}` entry covers {date}; this table is populated for {coverage} \
         — KOPITIAM will not extrapolate a policy value outside its cited effective period"
    )]
    NoRuleInEffect {
        /// Which policy table was queried.
        table: &'static str,
        /// The date that was asked about.
        date: Date,
        /// Human-readable description of what the table *does* cover, so the
        /// caller can see immediately whether this is a data gap or a typo.
        coverage: String,
    },

    /// Two entries in the same policy table claim the same date.
    ///
    /// Always a data-entry bug. Surfaced by
    /// [`crate::cpf::temporal::PolicyTable::validate`], which every built-in
    /// table is checked against in the test suite.
    #[error("`{table}` has overlapping entries: {first} overlaps {second}")]
    OverlappingRules {
        table: &'static str,
        first: String,
        second: String,
    },

    /// The requested policy dimension exists in the model but has not been
    /// populated with cited data.
    ///
    /// **Read the `reason`.** It says what is missing and why the author was
    /// not confident enough to fill it in. Treating this as "probably fine, use
    /// the nearest value" defeats the entire point of the crate.
    #[error("CPF policy for {dimension} is not populated in KOPITIAM: {reason}")]
    NotPopulated {
        /// The specific slice of policy that is missing, e.g.
        /// `"allocation ratios for members above age 55"`.
        dimension: String,
        /// Why it is missing. Always an honest statement of the author's
        /// uncertainty, never a shrug.
        reason: &'static str,
    },

    /// An allocation table whose Ordinary/Special/MediSave ratios do not sum to
    /// exactly 100%.
    ///
    /// Enforced because the ratios *are* a partition of the total contribution.
    /// A table that sums to 99.99% would quietly lose a member's money.
    #[error("allocation ratios for {band} sum to {actual}, not 100%")]
    AllocationDoesNotSumToOne {
        band: String,
        actual: crate::cpf::money::Rate,
    },
}
