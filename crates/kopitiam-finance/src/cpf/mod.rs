//! # CPF — Singapore's Central Provident Fund
//!
//! A model of what the CPF Board's **published policy says**, with a citation
//! attached to every figure.
//!
//! ---
//!
//! ## This is not financial advice, and it does not pretend to be
//!
//! This module answers exactly one kind of question:
//!
//! > *"What does the published policy say the contribution is, for a member of
//! > this age, on this wage, in this month — and where does it say it?"*
//!
//! It does **not** answer:
//!
//! > *"What should I do with my CPF?"*
//!
//! The distinction is not pedantry. CPF governs the retirement, healthcare and
//! housing savings of everyone who works in Singapore; a wrong contribution rate,
//! a wrong Ordinary/Special/MediSave split, a misread housing withdrawal limit or
//! a stale retirement sum has direct, material consequences for a real person's
//! home and old age. A tool that reports *what a rule says and where it is
//! written* is useful and auditable. A tool that tells someone what to do with
//! their money, on the strength of the same code, is a liability. This crate is
//! firmly the first, and every design decision below follows from that.
//!
//! **Verify against the CPF Board's own publications before acting on anything
//! here.** See [`published`] for exactly how strong (and how weak) the provenance
//! of the shipped figures currently is.
//!
//! ---
//!
//! ## The three rules this module is built around
//!
//! ### 1. There are no constants. Every number is dated.
//!
//! Contribution rates, allocation ratios, wage ceilings, retirement sums,
//! interest floors — **every single one changes**, typically annually, and they
//! vary by age band, wage level and residency status. The Ordinary Wage ceiling
//! took five different values between 2023 and 2026, one of the changes landing
//! *mid-year*. Senior-worker contribution rates have stepped up every January
//! since 2022. The Enhanced Retirement Sum was 3x the Basic sum, until it became
//! 4x. The Special Account stopped existing for members aged 55 and above.
//!
//! So there is no `const CPF_RATE`. There is no `latest()`. There is only
//! [`temporal::PolicyTable::on`] — *the rule in force on date D* — and an honest
//! error if no cited rule covers that date. See [`temporal`], which is the
//! architectural centre of this module.
//!
//! ### 2. Provenance is mandatory.
//!
//! Every value is a [`temporal::Dated<T>`]: a value, an effective period, and a
//! [`citation::Citation`] — a publisher, a document, a locator, and how the value
//! physically got into KOPITIAM. The three are inseparable by construction; a
//! lookup hands back all of them together, so a caller cannot obtain the number
//! while discarding the evidence.
//!
//! When a user asks *why*, the answer is "because §X of document Y, effective Z"
//! — never "because the code says so". This is CLAUDE.md's Scientific Standards
//! applied to a domain where explainability is not an academic virtue but the
//! difference between a tool someone can check and a black box they must trust.
//!
//! ### 3. What we do not know, we say we do not know.
//!
//! This is a **scaffold**: the shape is complete, the data is a small, honest,
//! well-cited slice. Every gap returns
//! [`error::CpfError::NotPopulated`] or [`error::CpfError::NoRuleInEffect`],
//! carrying the reason. Nothing extrapolates, nothing falls back to the nearest
//! entry, nothing guesses.
//!
//! A member aged 55 or above gets an error, not a number, because the post-55
//! allocation ratios changed when the Special Account closed and this crate does
//! not confidently know the new ones. **That error is the crate working
//! correctly.** A confidently wrong CPF rate is far worse than an absent one: the
//! absent one sends you to the source, and the wrong one does not.
//!
//! [`published`] carries the full inventory of what is populated and what is not,
//! and the gaps are emitted into the knowledge graph as facts in their own right
//! (see [`ontology`]) so a downstream consumer can discover the boundary of our
//! ignorance without reading this source.
//!
//! ---
//!
//! ## Type safety
//!
//! A wage ceiling and a retirement sum are not both `f64`, and neither is a
//! contribution rate:
//!
//! * [`money::Sgd`] — an exact integer number of cents. **No `f64` appears
//!   anywhere in this module's public API.** `wage * 0.20` in binary floating
//!   point lands, for some wages, a cent — and therefore, after CPF's statutory
//!   rounding, a *dollar* — away from the CPF Board's own table.
//! * [`money::Rate`] — an exact integer number of basis points. Every CPF figure
//!   fits: rates are published to a tenth of a percent, allocation ratios to four
//!   decimal places.
//! * [`money::Unrounded`] — the product of the two, which **cannot be spent**
//!   until you name a rounding rule. CPF's rule is three steps and one of them is
//!   a residual; this type makes it impossible to skip.
//! * [`rates::WageCeilings`], [`rates::RetirementSums`],
//!   [`rates::ContributionRates`], [`rates::AllocationRatios`] — distinct types,
//!   so passing one where another belongs is a compile error rather than a
//!   plausible-looking payslip.
//!
//! ---
//!
//! ## Where the bodies are buried
//!
//! Four things in this domain are counter-intuitive, cost real money when got
//! wrong, and are each documented at length where they are implemented:
//!
//! 1. **A member's rate does not change on their birthday.** It changes on the
//!    first day of the month *after* the birthday month. See
//!    [`structure::contribution_band_on`].
//! 2. **The contribution age bands and the allocation age bands are different.**
//!    Contributions step at 55/60/65/70; allocation at 35/45/50/55/60/65. See
//!    [`structure`].
//! 3. **The Additional Wage ceiling is a residual against Ordinary Wages already
//!    capped**, not against wages paid — and since the monthly cap has itself
//!    changed mid-year, it cannot be computed as `12 x ceiling`. See
//!    [`rates::WageCeilings`].
//! 4. **The employer's share is what is left over**, not an independent
//!    rounding. See [`rates::ContributionRates::split`].
//!
//! ---
//!
//! ## Getting started
//!
//! ```
//! use kopitiam_finance::cpf::{
//!     CpfPolicy, Date, Member, MonthlyWages, Residency, Sgd, YearContext,
//! };
//!
//! let policy = CpfPolicy::published();
//!
//! let breakdown = policy.contribution(
//!     Residency::CitizenOrPrFromThirdYear,
//!     Member::BornOn(Date::new(1990, 5, 20)?),
//!     MonthlyWages::salary(Sgd::from_dollars(5_000)),
//!     Date::new(2025, 3, 1)?,      // the rule in force in MARCH 2025, not "now"
//!     YearContext::none(),
//! )?;
//!
//! assert_eq!(breakdown.contribution.total, Sgd::from_dollars(1_850));
//! assert_eq!(breakdown.contribution.employee, Sgd::from_dollars(1_000));
//! assert_eq!(breakdown.contribution.employer, Sgd::from_dollars(850)); // the residual
//!
//! // And the answer knows why it is the answer.
//! for citation in breakdown.citations.all() {
//!     println!("{citation}");
//! }
//! # Ok::<(), kopitiam_finance::cpf::CpfError>(())
//! ```
//!
//! Ask about a member aged 55 or above and you get an honest refusal instead:
//!
//! ```
//! # use kopitiam_finance::cpf::*;
//! # let policy = CpfPolicy::published();
//! let answer = policy.contribution(
//!     Residency::CitizenOrPrFromThirdYear,
//!     Member::AgedExactly(Age::years(57)),
//!     MonthlyWages::salary(Sgd::from_dollars(5_000)),
//!     Date::new(2025, 3, 1).unwrap(),
//!     YearContext::none(),
//! );
//! assert!(matches!(answer, Err(CpfError::NotPopulated { .. })));
//! ```

pub mod citation;
pub mod date;
pub mod document;
pub mod error;
pub mod money;
pub mod ontology;
pub mod published;
pub mod query;
pub mod rates;
pub mod structure;
pub mod temporal;

pub use citation::{Citation, SourceKind};
pub use date::{Date, DateRange};
pub use error::CpfError;
pub use money::{Rate, Sgd, Unrounded};
pub use ontology::CpfKnowledge;
pub use query::{
    Citations, ContributionBreakdown, CpfPolicy, Member, MonthlyWages, YearContext,
};
pub use rates::{
    Allocation, AllocationRatios, AllocationSchedule, ContributionRates, ContributionSchedule,
    ContributionSplit, InterestFloors, RetirementSums, WageCeilings,
};
pub use structure::{
    Account, Age, AllocationAgeBand, ContributionAgeBand, Residency, WageBand,
    allocation_band_on, contribution_band_on,
};
pub use temporal::{Dated, PolicyTable};
