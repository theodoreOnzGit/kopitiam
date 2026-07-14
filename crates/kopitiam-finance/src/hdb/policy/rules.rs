//! The populated slice of HDB policy: the tables themselves.
//!
//! # Read this before trusting a number in this file
//!
//! Every figure below was **transcribed offline, from recollection, with no
//! network access**, by an AI agent building the engine that holds it. The
//! *structure* is the deliverable; the *content* is a small, deliberately
//! conservative starting slice.
//!
//! Consequently:
//!
//! * Every [`Citation`] here is [`Verification::Unverified`]. Not one figure has
//!   been read off an HDB page by the process that wrote it.
//!   [`HdbPolicy::unverified_provisions`](super::HdbPolicy::unverified_provisions)
//!   will tell you that all of them are.
//! * The URLs are the best-known **entry points** to the relevant HDB pages, not
//!   verified deep links. Treat them as "start here", not as proof.
//! * Where confidence ran out, the table says
//!   [`Provision::NotModelled`] and the query returns an error, which becomes an
//!   [`Indeterminate`](super::eligibility::Eligibility::Indeterminate) answer.
//!   **This happens a lot, and on purpose.** A grant amount left absent costs a
//!   caller an afternoon; a grant amount invented costs them a house.
//!
//! # What is populated
//!
//! | Table | Coverage |
//! |---|---|
//! | Income ceilings | New flats and resale-with-grant, family and (new-flat) single; the "no ceiling" rule for resale without a grant |
//! | Minimum ages | Public, Fiancé/Fiancée, Single Singapore Citizen, Joint Singles, Non-Citizen Spouse |
//! | Minimum Occupation Period | Unclassified (pre-2024) flats; Standard, Plus, Prime; Prime Location Public Housing |
//! | Ethnic Integration Policy | Block and neighbourhood limits per ethnic group, and the SPR quota |
//! | Enhanced CPF Housing Grant | The 2019 schedule only — **superseded from 20 Aug 2024 by figures this crate does not hold** |
//!
//! # What is declared but deliberately empty
//!
//! Present as [`Provision::NotModelled`] spans, so that asking yields "this
//! applies and we do not model it" rather than a misleading silence:
//! the CPF Housing Grant (Family and Singles), the Proximity Housing Grant, the
//! Enhanced CPF Housing Grant from 20 August 2024, the resale levy amounts, and
//! the resale-with-grant income ceiling for single applicants.
//!
//! Not represented at all, and listed in [`UNMODELLED`]: everything else.

use super::citation::{Citation, Verification};
use super::domain::{
    CeilingContext, EligibilityScheme, EthnicGroup, FlatClassification, FlatType, Grant,
    HouseholdClass,
};
use super::quantity::{
    Age, GrantAmount, IncomeCeiling, MinimumAge, MinimumOccupationPeriod, MonthlyIncome, Months,
    Percent, Sgd,
};
use super::temporal::{
    Date, Dated, EffectiveRange, PolicyTable, Provision, TableError, Timeline, date,
};

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Value types that only the tables use
// ---------------------------------------------------------------------------

/// The Ethnic Integration Policy limits applying to one group.
///
/// Two limits, because the EIP is enforced at two scales: an applicant may be
/// blocked because their *block* is at its limit even though the neighbourhood
/// is not, and vice versa.
///
/// **Knowing the limits is not knowing the answer.** Whether a *particular* flat
/// may be sold to a *particular* buyer depends on the current ethnic composition
/// of that block and neighbourhood, which HDB publishes per-block and this crate
/// does not hold. Any question of the form "can I buy this flat" is therefore
/// [`Indeterminate`](super::eligibility::Eligibility::Indeterminate), and saying
/// so is the only honest option available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EipLimits {
    /// Maximum proportion of the neighbourhood's flats held by the group.
    pub neighbourhood: Percent,
    /// Maximum proportion of the block's flats held by the group.
    pub block: Percent,
}

impl fmt::Display for EipLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} of a neighbourhood, {} of a block",
            self.neighbourhood, self.block
        )
    }
}

/// One row of an income-tapered grant table: "household income up to X ⇒ grant
/// of Y".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantBracket {
    /// The top of the income band, **inclusive** — HDB's tables read
    /// "$1,501 to $2,000", so a household at exactly $2,000 is in this band.
    pub income_upper_inclusive: Sgd,
    /// The grant for a household in this band.
    pub amount: GrantAmount,
}

/// An income-tapered grant schedule.
///
/// Modelled as explicit brackets rather than a formula. The 2019 Enhanced CPF
/// Housing Grant *does* happen to be a clean arithmetic taper, and it is
/// generated from that arithmetic below — but the published artefact is a table,
/// future revisions need not be linear (the 2024 revision, which this crate does
/// not hold, may well not be), and a formula that stops matching the table is
/// far harder to notice than a bracket that is simply wrong.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantSchedule {
    brackets: Vec<GrantBracket>,
}

impl GrantSchedule {
    /// Builds a schedule from brackets, sorted by income band.
    pub fn new(mut brackets: Vec<GrantBracket>) -> Self {
        brackets.sort_by_key(|b| b.income_upper_inclusive);
        Self { brackets }
    }

    /// The grant for a household at `income`.
    ///
    /// `None` means the income is above every band — i.e. **no grant**, which is
    /// a real answer, distinct from "we don't know" (that one is a
    /// [`TemporalError`](super::temporal::TemporalError) from the lookup, before
    /// this is ever called).
    pub fn amount_for(&self, income: MonthlyIncome) -> Option<GrantAmount> {
        self.brackets
            .iter()
            .find(|b| income.0 <= b.income_upper_inclusive)
            .map(|b| b.amount)
    }

    /// The brackets, in ascending income order.
    pub fn brackets(&self) -> &[GrantBracket] {
        &self.brackets
    }
}

/// The key of the Minimum Occupation Period table.
///
/// MOP attaches to the **flat**, via its classification, and is fixed at the
/// point of purchase — which is why the MOP table is queried with the *purchase*
/// date and not with "today".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MopKey {
    /// A flat sold under a classification (the Standard/Plus/Prime framework, or
    /// the earlier Prime Location Public Housing model).
    Classified(FlatClassification),
    /// A flat sold before any classification existed and outside the PLH model.
    Unclassified,
}

impl fmt::Display for MopKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MopKey::Classified(c) => write!(f, "{c}"),
            MopKey::Unclassified => f.write_str("unclassified (pre-framework) flat"),
        }
    }
}

/// An area of HDB policy this crate does not model, and what a caller should do
/// about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmodelledArea {
    /// The area.
    pub area: &'static str,
    /// Why it is absent, and what depends on it.
    pub note: &'static str,
}

/// Everything a caller might reasonably expect this crate to know, and it does
/// not.
///
/// This list is part of the API, not a TODO comment. A policy engine that cannot
/// enumerate its own blind spots is indistinguishable, from the outside, from
/// one that has none — and that is precisely the failure mode this crate is
/// built to avoid.
pub const UNMODELLED: &[UnmodelledArea] = &[
    UnmodelledArea {
        area: "Ethnic Integration Policy: whether a particular block has quota space",
        note: "The limits are modelled; the current ethnic composition of each block and \
               neighbourhood is not. Whether a specific flat may be sold to a specific buyer \
               is therefore always Indeterminate here. HDB publishes the live per-block \
               position; this crate does not ingest it.",
    },
    UnmodelledArea {
        area: "Enhanced CPF Housing Grant from 20 August 2024",
        note: "Raised at the 2024 National Day Rally. The revised amounts and taper are not \
               held here and were not guessed. Any EHG query on or after that date is \
               Indeterminate — which means, in practice, every present-day query.",
    },
    UnmodelledArea {
        area: "CPF Housing Grant (Family and Singles) and Proximity Housing Grant amounts",
        note: "The grants are declared so they cannot be silently omitted, but their amounts \
               are not modelled. Both were revised in the 2019 restructure and again in 2024.",
    },
    UnmodelledArea {
        area: "Resale levy amounts",
        note: "The levy is payable by a second-timer household buying a second subsidised \
               flat, and the amount depends on the first flat's type. The figures are not \
               entered.",
    },
    UnmodelledArea {
        area: "Income assessment",
        note: "How HDB computes 'average gross monthly household income' — the 12-month \
               averaging window, variable and self-employed income, whose income counts as \
               an essential occupier — is itself policy. This crate takes the income as an \
               input and does not derive it.",
    },
    UnmodelledArea {
        area: "Whether a Minimum Occupation Period has been served",
        note: "The MOP *duration* is modelled. Whether a given household has served it \
               depends on the key-collection date and on periods of non-occupation, neither \
               of which this crate holds.",
    },
    UnmodelledArea {
        area: "Flat-type restrictions per scheme",
        note: "Which flat types a scheme may buy (notably: single applicants and 2-room \
               Flexi, and the size limits that vary by location and classification) is not \
               modelled. Queries that turn on it are Indeterminate.",
    },
    UnmodelledArea {
        area: "Executive Condominiums",
        note: "ECs are sold by private developers under a separate income ceiling and a \
               different eligibility regime. Out of scope entirely — this crate will not \
               answer EC questions and does not pretend to.",
    },
    UnmodelledArea {
        area: "HDB housing loans, CPF withdrawal limits, and affordability",
        note: "The Loan-to-Value limit, the HDB Loan Eligibility letter, the CPF Valuation \
               Limit and Withdrawal Limit all gate what a household can actually buy. They \
               belong with the CPF engine (crates/kopitiam-finance/src/cpf) and are not \
               modelled here.",
    },
    UnmodelledArea {
        area: "Waiting periods, debarment, and prior-ownership rules",
        note: "Private-property ownership and disposal, the 30-month wait after disposing of \
               a private property, debarment after cancelling a flat application, and the \
               15-month wait for private-property owners buying a resale flat are not \
               modelled.",
    },
];

// ---------------------------------------------------------------------------
// Citations
// ---------------------------------------------------------------------------

/// URL of HDB's eligibility pages for new flats.
const URL_NEW_FLAT_ELIGIBILITY: &str = "https://www.hdb.gov.sg/residential/buying-a-flat/new";
/// URL of HDB's eligibility pages for resale flats.
const URL_RESALE_ELIGIBILITY: &str = "https://www.hdb.gov.sg/residential/buying-a-flat/resale";
/// URL of HDB's Minimum Occupation Period page.
const URL_MOP: &str = "https://www.hdb.gov.sg/residential/selling-a-flat/eligibility";
/// URL of HDB's Ethnic Integration Policy / SPR quota page.
const URL_EIP: &str =
    "https://www.hdb.gov.sg/residential/buying-a-flat/resale/ethnic-integration-policy";
/// URL of HDB's housing-grant pages.
const URL_GRANTS: &str = "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility";
/// URL of the Prime Minister's Office newsroom, where Rally speeches are published.
const URL_PMO: &str = "https://www.pmo.gov.sg/Newsroom";

// The dates below are policy dates, not arbitrary. They are named so that a
// reader can see at a glance which announcement a provision hangs off, and so
// that correcting one corrects every provision that cites it.

/// National Day Rally 2015: the family income ceiling for new flats was raised
/// to $12,000, applying to flat applications from this date.
fn ndr_2015() -> Date {
    date(2015, 8, 24)
}

/// National Day Rally 2019 and the accompanying HDB/MND announcement: income
/// ceilings raised, and the Enhanced CPF Housing Grant introduced, applying to
/// flat applications from this date.
fn ndr_2019() -> Date {
    date(2019, 9, 11)
}

/// The 5 March 2010 revision of the Ethnic Integration Policy limits, which also
/// introduced the separate Singapore Permanent Resident quota.
fn eip_revision_2010() -> Date {
    date(2010, 3, 5)
}

/// The 30 August 2010 property measures, which lengthened the Minimum Occupation
/// Period for non-subsidised resale flats to five years.
fn property_measures_2010() -> Date {
    date(2010, 8, 30)
}

/// The November 2021 Build-To-Order exercise, the first launched under the Prime
/// Location Public Housing model.
///
/// A *sales exercise*, not a gazette date — see [`Date`]'s caveat. The first of
/// the month is used as the exercise's effective anchor.
fn plh_first_exercise() -> Date {
    date(2021, 11, 1)
}

/// The October 2024 Build-To-Order exercise, the first launched under the
/// Standard/Plus/Prime framework announced at the 2023 National Day Rally.
///
/// Again an exercise, not a gazette date.
fn spp_first_exercise() -> Date {
    date(2024, 10, 1)
}

/// The 2024 National Day Rally, at which the Enhanced CPF Housing Grant was
/// raised. Applies to flat applications from this date.
///
/// This crate holds the *fact of* the change and not the figures, which is why
/// it appears only as the boundary of a [`Provision::NotModelled`] span.
fn ndr_2024_ehg() -> Date {
    date(2024, 8, 20)
}

/// A citation to an HDB InfoWEB eligibility or policy page, anchored at the
/// earliest date we are confident the rule was in force. See
/// [`Citation::hdb_infoweb`] for why undated web pages need an anchor.
fn infoweb(section: &str, anchor: Date, url: &str) -> Citation {
    Citation::hdb_infoweb(
        section,
        format!("in force at least from {anchor}"),
        anchor,
        url,
    )
}

/// A citation to a dated announcement (a Rally speech, a press release).
fn announced(publisher: &str, title: &str, section: &str, published: Date, url: &str) -> Citation {
    Citation::announcement(publisher, title, section, published, url)
}

// ---------------------------------------------------------------------------
// Income ceilings
// ---------------------------------------------------------------------------

/// Income ceilings, keyed by the purchase context and the household class.
///
/// # What is here
///
/// * **New flats, family**: $12,000 from the 2015 Rally, $14,000 from the 2019
///   Rally. The same two figures apply to a resale flat bought *with* a grant.
/// * **New flats, single**: $6,000, then $7,000, on the same two dates.
/// * **Resale without a grant**: [`IncomeCeiling::NoCeiling`] — a *rule*, stated
///   as one, not an absent entry.
///
/// # What is not
///
/// * **Resale with a grant, single applicant**: not modelled. The generic
///   singles ceiling is very likely the figure that applies, but "very likely"
///   is not a standard this crate is willing to meet for a number a person would
///   act on.
/// * Anything before the 2015 Rally. A query about 2014 gets
///   [`TemporalError::BeforeEarliestProvision`](super::temporal::TemporalError::BeforeEarliestProvision),
///   not the 2015 figure.
/// * Flat-type-specific variations. The ceiling table is not keyed by flat type,
///   and [`assess`](super::HdbPolicy::assess) raises an
///   [`Unknown`](super::eligibility::Unknown) where one is known to exist.
pub fn income_ceilings()
-> Result<PolicyTable<(CeilingContext, HouseholdClass), IncomeCeiling>, TableError> {
    let rally_2015 = |what: &str| {
        announced(
            "Prime Minister's Office",
            "National Day Rally 2015",
            &format!("{what} (applies to flat applications from 24 August 2015)"),
            ndr_2015(),
            URL_PMO,
        )
    };
    let rally_2019 = |what: &str| {
        announced(
            "Prime Minister's Office",
            "National Day Rally 2019",
            &format!("{what} (applies to flat applications from 11 September 2019)"),
            ndr_2019(),
            URL_PMO,
        )
    };

    let family_ceilings = |url: &'static str| {
        vec![
            Provision::InForce(Dated::new(
                IncomeCeiling::NotExceeding(Sgd::dollars(12_000)),
                EffectiveRange::between(ndr_2015(), ndr_2019()),
                rally_2015("family income ceiling for new flats raised to $12,000"),
            )),
            Provision::InForce(Dated::new(
                IncomeCeiling::NotExceeding(Sgd::dollars(14_000)),
                EffectiveRange::from(ndr_2019()),
                {
                    let mut c = rally_2019("family income ceiling raised to $14,000");
                    c.url = url.to_string();
                    c
                },
            )),
        ]
    };

    PolicyTable::new(
        "HDB income ceiling",
        vec![
            (
                (CeilingContext::NewFlat, HouseholdClass::Family),
                family_ceilings(URL_NEW_FLAT_ELIGIBILITY),
            ),
            (
                (CeilingContext::ResaleWithGrant, HouseholdClass::Family),
                family_ceilings(URL_RESALE_ELIGIBILITY),
            ),
            (
                (CeilingContext::NewFlat, HouseholdClass::Single),
                vec![
                    Provision::InForce(Dated::new(
                        IncomeCeiling::NotExceeding(Sgd::dollars(6_000)),
                        EffectiveRange::between(ndr_2015(), ndr_2019()),
                        rally_2015("singles income ceiling for new flats raised to $6,000"),
                    )),
                    Provision::InForce(Dated::new(
                        IncomeCeiling::NotExceeding(Sgd::dollars(7_000)),
                        EffectiveRange::from(ndr_2019()),
                        rally_2019("singles income ceiling raised to $7,000"),
                    )),
                ],
            ),
            (
                (CeilingContext::ResaleWithGrant, HouseholdClass::Single),
                vec![Provision::NotModelled {
                    effective: EffectiveRange::from(ndr_2015()),
                    reason: "the income ceiling for a single applicant buying a resale flat with \
                             a grant is not modelled: the generic singles ceiling is the likely \
                             figure, but grant-specific ceilings have differed and this was not \
                             confirmed offline."
                        .to_string(),
                    announced_in: Some(infoweb(
                        "Eligibility to buy a resale flat / grant eligibility",
                        ndr_2019(),
                        URL_GRANTS,
                    )),
                }],
            ),
            (
                (CeilingContext::ResaleWithoutGrant, HouseholdClass::Family),
                vec![no_resale_ceiling()],
            ),
            (
                (CeilingContext::ResaleWithoutGrant, HouseholdClass::Single),
                vec![no_resale_ceiling()],
            ),
        ],
    )
}

/// "No income ceiling applies to a resale purchase without a grant" — stated as
/// a provision, so that it can be cited like any other rule.
fn no_resale_ceiling() -> Provision<IncomeCeiling> {
    Provision::InForce(Dated::new(
        IncomeCeiling::NoCeiling,
        EffectiveRange::in_force_at_least_from(ndr_2015()),
        infoweb(
            "Eligibility to buy a resale flat: no income ceiling applies where no housing grant \
             is taken",
            ndr_2015(),
            URL_RESALE_ELIGIBILITY,
        ),
    ))
}

// ---------------------------------------------------------------------------
// Minimum ages
// ---------------------------------------------------------------------------

/// The minimum age each eligibility scheme requires of its applicants.
///
/// These look like constants of nature. They are not: 35 is a policy choice that
/// has been debated in Parliament repeatedly, and it is exactly the kind of
/// figure that moves in a Rally speech. It therefore lives in a dated table like
/// everything else.
///
/// Their commencement dates are *not* modelled — HDB's eligibility pages are
/// undated and this crate could not establish when the thresholds were set. Each
/// provision is anchored with
/// [`EffectiveRange::in_force_at_least_from`], so a query about 2005 fails
/// honestly rather than assuming the rule reached back that far.
///
/// The Single Singapore Citizen Scheme's *widowed or orphaned* applicants have a
/// lower threshold, which is deliberately absent: see
/// [`Unknown`](super::eligibility::Unknown) — the assessment raises it rather
/// than applying the 35 to someone it may not apply to.
pub fn minimum_ages() -> Result<PolicyTable<EligibilityScheme, MinimumAge>, TableError> {
    let rule = |scheme: EligibilityScheme, years: u32, url: &'static str| {
        (
            scheme,
            vec![Provision::InForce(Dated::new(
                MinimumAge(Age(years)),
                EffectiveRange::in_force_at_least_from(ndr_2019()),
                infoweb(
                    &format!(
                        "Eligibility to buy a flat: {scheme} — applicants must be at least {years}"
                    ),
                    ndr_2019(),
                    url,
                ),
            ))],
        )
    };

    PolicyTable::new(
        "HDB minimum applicant age",
        vec![
            rule(EligibilityScheme::Public, 21, URL_NEW_FLAT_ELIGIBILITY),
            rule(
                EligibilityScheme::FianceFiancee,
                21,
                URL_NEW_FLAT_ELIGIBILITY,
            ),
            rule(
                EligibilityScheme::SingleSingaporeCitizen,
                35,
                URL_NEW_FLAT_ELIGIBILITY,
            ),
            rule(
                EligibilityScheme::JointSingles,
                35,
                URL_NEW_FLAT_ELIGIBILITY,
            ),
            rule(
                EligibilityScheme::NonCitizenSpouse,
                21,
                URL_NEW_FLAT_ELIGIBILITY,
            ),
        ],
    )
}

// ---------------------------------------------------------------------------
// Minimum Occupation Period
// ---------------------------------------------------------------------------

/// The Minimum Occupation Period, keyed by the flat's classification.
///
/// # The shape of this table is the interesting part
///
/// MOP used to be a property of *how you bought* (five years for a BTO flat,
/// five for a subsidised resale flat, and — before the 30 August 2010 measures —
/// as little as one for some non-subsidised resale purchases financed by a bank
/// loan). Since the Standard/Plus/Prime framework it is a property of *the flat*:
/// a Plus flat carries ten years whether it was bought new or resale.
///
/// That reclassification is why [`MopKey::Unclassified`] ends where
/// [`MopKey::Classified`] begins, and why the table is queried with the
/// **purchase date**: the MOP is fixed at purchase and does not change under the
/// owner afterwards.
///
/// # What is not here
///
/// Anything before 30 August 2010 — the pre-2010 resale MOP varied by loan type
/// and this crate does not model it. And, emphatically, *whether a household has
/// served* its MOP: that needs the key-collection date and a record of
/// occupation, which the crate does not hold.
pub fn minimum_occupation_periods()
-> Result<PolicyTable<MopKey, MinimumOccupationPeriod>, TableError> {
    let five_years = MinimumOccupationPeriod(Months(60));
    let ten_years = MinimumOccupationPeriod(Months(120));

    PolicyTable::new(
        "HDB Minimum Occupation Period",
        vec![
            (
                MopKey::Unclassified,
                vec![Provision::InForce(Dated::new(
                    five_years,
                    EffectiveRange::between(property_measures_2010(), spp_first_exercise()),
                    infoweb(
                        "Minimum Occupation Period: five years for flats bought from HDB, and \
                         for resale flats bought on the open market from 30 August 2010",
                        property_measures_2010(),
                        URL_MOP,
                    ),
                ))],
            ),
            (
                MopKey::Classified(FlatClassification::Standard),
                vec![Provision::InForce(Dated::new(
                    five_years,
                    EffectiveRange::from(spp_first_exercise()),
                    announced(
                        "Prime Minister's Office",
                        "National Day Rally 2023",
                        "Standard/Plus/Prime framework: Standard flats keep the five-year \
                         Minimum Occupation Period (first applied at the October 2024 sales \
                         exercise)",
                        spp_first_exercise(),
                        URL_PMO,
                    ),
                ))],
            ),
            (
                MopKey::Classified(FlatClassification::Plus),
                vec![Provision::InForce(Dated::new(
                    ten_years,
                    EffectiveRange::from(spp_first_exercise()),
                    announced(
                        "Prime Minister's Office",
                        "National Day Rally 2023",
                        "Standard/Plus/Prime framework: Plus flats carry a ten-year Minimum \
                         Occupation Period (first applied at the October 2024 sales exercise)",
                        spp_first_exercise(),
                        URL_PMO,
                    ),
                ))],
            ),
            (
                MopKey::Classified(FlatClassification::Prime),
                vec![Provision::InForce(Dated::new(
                    ten_years,
                    EffectiveRange::from(spp_first_exercise()),
                    announced(
                        "Prime Minister's Office",
                        "National Day Rally 2023",
                        "Standard/Plus/Prime framework: Prime flats carry a ten-year Minimum \
                         Occupation Period (first applied at the October 2024 sales exercise)",
                        spp_first_exercise(),
                        URL_PMO,
                    ),
                ))],
            ),
            (
                MopKey::Classified(FlatClassification::Plh),
                vec![Provision::InForce(Dated::new(
                    ten_years,
                    EffectiveRange::from(plh_first_exercise()),
                    infoweb(
                        "Prime Location Public Housing model: ten-year Minimum Occupation \
                         Period, from the first PLH sales exercise (November 2021)",
                        plh_first_exercise(),
                        URL_MOP,
                    ),
                ))],
            ),
        ],
    )
}

// ---------------------------------------------------------------------------
// Ethnic Integration Policy
// ---------------------------------------------------------------------------

/// The Ethnic Integration Policy limits, per ethnic group.
///
/// The EIP has applied since 1 March 1989. The limits below are anchored at the
/// 5 March 2010 revision — the earliest date this crate can cite them at their
/// present values — so a query about 1995 fails honestly rather than projecting
/// today's figures backwards onto a policy that had different ones.
///
/// **These limits do not answer the question a buyer is actually asking.** See
/// [`EipLimits`].
pub fn ethnic_quotas() -> Result<PolicyTable<EthnicGroup, EipLimits>, TableError> {
    let limits = |group: EthnicGroup, neighbourhood: u32, block: u32| {
        (
            group,
            vec![Provision::InForce(Dated::new(
                EipLimits {
                    neighbourhood: Percent::whole(neighbourhood),
                    block: Percent::whole(block),
                },
                EffectiveRange::in_force_at_least_from(eip_revision_2010()),
                infoweb(
                    &format!(
                        "Ethnic Integration Policy: {group} limits — {neighbourhood}% of a \
                         neighbourhood, {block}% of a block (limits revised 5 March 2010)"
                    ),
                    eip_revision_2010(),
                    URL_EIP,
                ),
            ))],
        )
    };

    PolicyTable::new(
        "HDB Ethnic Integration Policy limits",
        vec![
            limits(EthnicGroup::Chinese, 84, 87),
            limits(EthnicGroup::Malay, 22, 25),
            limits(EthnicGroup::IndianOther, 12, 15),
        ],
    )
}

/// The separate Singapore Permanent Resident quota, introduced on 5 March 2010.
///
/// It applies to SPR households and excludes Malaysian SPRs. It is a *second*
/// gate: an SPR household must clear both this quota and the ethnic one.
pub fn spr_quota() -> Result<Timeline<EipLimits>, TableError> {
    Timeline::new(
        "HDB Singapore Permanent Resident quota",
        vec![Provision::InForce(Dated::new(
            EipLimits {
                neighbourhood: Percent::whole(5),
                block: Percent::whole(8),
            },
            EffectiveRange::from(eip_revision_2010()),
            infoweb(
                "SPR quota: 5% of a neighbourhood and 8% of a block, introduced 5 March 2010; \
                 Malaysian SPRs are excluded from the quota",
                eip_revision_2010(),
                URL_EIP,
            ),
        ))],
    )
}

// ---------------------------------------------------------------------------
// Grants
// ---------------------------------------------------------------------------

/// The Enhanced CPF Housing Grant schedules, per household class.
///
/// # The 2019 schedule
///
/// Introduced with the September 2019 measures: $80,000 for a family household
/// earning $1,500 a month or less, tapering by $5,000 for each $500 of income,
/// to $5,000 in the $8,501–$9,000 band. Above $9,000 there is no EHG. Singles
/// receive half of each figure.
///
/// The brackets are *generated* from that arithmetic rather than typed out
/// sixteen times — the taper is a documented property of the published table, and
/// a generated table cannot contain a transcription typo. [`ehg_family_matches_published_anchor_points`]
/// pins the three endpoints of the published table so that the generator cannot
/// drift away from it silently.
///
/// # From 20 August 2024: nothing
///
/// The EHG was raised at the 2024 National Day Rally. This crate does **not**
/// hold the revised figures, and did not guess them. Every EHG query on or after
/// 20 August 2024 — which is to say every present-day query — returns
/// [`Indeterminate`](super::eligibility::Eligibility::Indeterminate) naming this
/// gap.
///
/// That is a severe limitation, and stating it is the correct response to it.
/// The alternative — carrying the 2019 figures forward past the date they stopped
/// being true — would hand a household a number that is wrong by tens of
/// thousands of dollars, with a citation attached to make it look reliable.
pub fn enhanced_housing_grant() -> Result<PolicyTable<HouseholdClass, GrantSchedule>, TableError> {
    let superseded = |household: HouseholdClass| Provision::NotModelled {
        effective: EffectiveRange::from(ndr_2024_ehg()),
        reason: format!(
            "the Enhanced CPF Housing Grant for {household} households was raised at the 2024 \
             National Day Rally, applying to flat applications from 20 August 2024. The revised \
             amounts and taper are not held by this crate and were not guessed."
        ),
        announced_in: Some(announced(
            "Prime Minister's Office",
            "National Day Rally 2024",
            "Enhanced CPF Housing Grant raised (applies to flat applications from 20 August 2024)",
            ndr_2024_ehg(),
            URL_PMO,
        )),
    };

    let schedule_2019 = |household: HouseholdClass| {
        Provision::InForce(Dated::new(
            ehg_schedule_2019(household),
            EffectiveRange::between(ndr_2019(), ndr_2024_ehg()),
            announced(
                "Prime Minister's Office",
                "National Day Rally 2019",
                &format!(
                    "Enhanced CPF Housing Grant introduced for {household} households \
                     (applies to flat applications from 11 September 2019)"
                ),
                ndr_2019(),
                URL_GRANTS,
            ),
        ))
    };

    PolicyTable::new(
        "HDB Enhanced CPF Housing Grant",
        vec![
            (
                HouseholdClass::Family,
                vec![
                    schedule_2019(HouseholdClass::Family),
                    superseded(HouseholdClass::Family),
                ],
            ),
            (
                HouseholdClass::Single,
                vec![
                    schedule_2019(HouseholdClass::Single),
                    superseded(HouseholdClass::Single),
                ],
            ),
        ],
    )
}

/// Generates the 2019 EHG bracket table from its documented taper.
///
/// Family: $80,000 at $1,500 and below, falling $5,000 per $500 band, reaching
/// $5,000 in the $8,501–$9,000 band. Sixteen bands. Singles: half of each.
fn ehg_schedule_2019(household: HouseholdClass) -> GrantSchedule {
    const BANDS: i64 = 15; // bands above the first
    const BAND_WIDTH_DOLLARS: i64 = 500;
    const FIRST_BAND_TOP_DOLLARS: i64 = 1_500;
    const MAX_GRANT_DOLLARS: i64 = 80_000;
    const STEP_DOLLARS: i64 = 5_000;

    let brackets = (0..=BANDS)
        .map(|i| {
            let top = Sgd::dollars(FIRST_BAND_TOP_DOLLARS + i * BAND_WIDTH_DOLLARS);
            let family = MAX_GRANT_DOLLARS - i * STEP_DOLLARS;
            let amount = match household {
                HouseholdClass::Family => Sgd::dollars(family),
                // Singles receive half the family grant. Every family figure is
                // a whole multiple of $5,000, so the halving is exact in cents
                // and no rounding rule is needed — which is precisely why money
                // is an integer here.
                HouseholdClass::Single => Sgd::cents(Sgd::dollars(family).as_cents() / 2),
            };
            GrantBracket {
                income_upper_inclusive: top,
                amount: GrantAmount(amount),
            }
        })
        .collect();

    GrantSchedule::new(brackets)
}

/// The grants whose *existence* is modelled and whose *amounts* are not.
///
/// Declaring them matters. If the CPF Housing Grant were simply absent, a caller
/// enumerating this crate's grants would conclude a resale buyer gets only the
/// EHG — an omission that reads exactly like a fact. Present as
/// [`Provision::NotModelled`], they instead say: *this applies to you, and we
/// cannot tell you how much*.
pub fn other_grants() -> Result<PolicyTable<(Grant, HouseholdClass), GrantSchedule>, TableError> {
    let not_modelled = |grant: Grant, household: HouseholdClass, why: &str| {
        (
            (grant, household),
            vec![Provision::NotModelled {
                effective: EffectiveRange::from(ndr_2019()),
                reason: format!("{grant} ({household} households): {why}"),
                announced_in: Some(infoweb(
                    &format!("{grant} — amounts"),
                    ndr_2019(),
                    URL_GRANTS,
                )),
            }],
        )
    };

    PolicyTable::new(
        "HDB housing grants (amounts not modelled)",
        vec![
            not_modelled(
                Grant::Family,
                HouseholdClass::Family,
                "amounts vary by flat type and were revised in the 2019 restructure and again \
                 in 2024; not entered offline without the source table.",
            ),
            not_modelled(
                Grant::Singles,
                HouseholdClass::Single,
                "amounts vary by flat type and were revised in the 2019 restructure and again \
                 in 2024; not entered offline without the source table.",
            ),
            not_modelled(
                Grant::Proximity,
                HouseholdClass::Family,
                "amount depends on whether the household lives with or near the parents or \
                 child; not entered offline without the source table.",
            ),
            not_modelled(
                Grant::Proximity,
                HouseholdClass::Single,
                "amount depends on whether the household lives with or near the parents or \
                 child; not entered offline without the source table.",
            ),
        ],
    )
}

/// The resale levy, keyed by the type of the *first* subsidised flat.
///
/// Payable by a second-timer household taking a second housing subsidy. Every
/// amount is [`Provision::NotModelled`]: the levy is a fixed sum per first-flat
/// type, the sums were last set in 2006, and they were not entered from
/// recollection.
pub fn resale_levy() -> Result<PolicyTable<FlatType, Sgd>, TableError> {
    let levy_reform_2006 = date(2006, 3, 3);
    let not_modelled = |flat: FlatType| {
        (
            flat,
            vec![Provision::NotModelled {
                effective: EffectiveRange::from(levy_reform_2006),
                reason: format!(
                    "the resale levy for a household whose first subsidised flat was a {flat} is \
                     a fixed sum, payable on taking a second housing subsidy. The sums (set for \
                     flats sold on or after 3 March 2006) are not held by this crate."
                ),
                announced_in: Some(infoweb(
                    "Resale levy",
                    levy_reform_2006,
                    URL_RESALE_ELIGIBILITY,
                )),
            }],
        )
    };

    PolicyTable::new(
        "HDB resale levy",
        vec![
            not_modelled(FlatType::TwoRoomFlexi),
            not_modelled(FlatType::ThreeRoom),
            not_modelled(FlatType::FourRoom),
            not_modelled(FlatType::FiveRoom),
            not_modelled(FlatType::Executive),
            not_modelled(FlatType::ThreeGen),
        ],
    )
}

/// The waiting period an all-SPR household must serve before buying a resale
/// flat.
///
/// A household with no Singapore Citizen may not buy a new flat at all, and may
/// buy a resale flat only after each SPR has held that status for three years.
/// This is one of the clearest examples of residency being a first-class input:
/// change one applicant from PR to citizen and the rule vanishes.
pub fn spr_resale_waiting_period() -> Result<Timeline<Months>, TableError> {
    let introduced = date(2013, 7, 5);
    Timeline::new(
        "HDB SPR household resale waiting period",
        vec![Provision::InForce(Dated::new(
            Months(36),
            EffectiveRange::from(introduced),
            infoweb(
                "Eligibility to buy a resale flat: a household with no Singapore Citizen must \
                 wait three years from the grant of Permanent Resident status before buying a \
                 resale flat (introduced 5 July 2013)",
                introduced,
                URL_RESALE_ELIGIBILITY,
            ),
        ))],
    )
}

/// The citation for HDB's general eligibility conditions: the citizenship
/// requirement, and the shape each scheme requires of a household (how many
/// applicants, and of what status).
///
/// These conditions are prose on HDB's eligibility pages rather than a table of
/// figures, so they are cited rather than tabulated — but a refusal built on them
/// still arrives with its source attached, which is the point.
pub fn eligibility_page() -> Citation {
    infoweb(
        "Eligibility to buy a flat: citizenship, family nucleus, and the conditions of each \
         eligibility scheme",
        ndr_2019(),
        URL_NEW_FLAT_ELIGIBILITY,
    )
}

/// Whether a citation is one nobody has checked. Used by
/// [`HdbPolicy::unverified_provisions`](super::HdbPolicy::unverified_provisions).
pub(super) fn is_unverified(citation: &Citation) -> bool {
    matches!(citation.verification, Verification::Unverified { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::policy::temporal::TemporalError;

    #[test]
    fn every_table_builds_without_overlapping_provisions() {
        // If a table ever contains two rules covering one date, this fails at
        // construction rather than at some caller's query.
        assert!(income_ceilings().is_ok());
        assert!(minimum_ages().is_ok());
        assert!(minimum_occupation_periods().is_ok());
        assert!(ethnic_quotas().is_ok());
        assert!(spr_quota().is_ok());
        assert!(enhanced_housing_grant().is_ok());
        assert!(other_grants().is_ok());
        assert!(resale_levy().is_ok());
        assert!(spr_resale_waiting_period().is_ok());
    }

    #[test]
    fn the_family_new_flat_ceiling_changes_on_the_2019_boundary_day() {
        let t = income_ceilings().unwrap();
        let key = (CeilingContext::NewFlat, HouseholdClass::Family);

        let before = t.on(&key, date(2019, 9, 10)).unwrap();
        assert_eq!(
            before.value,
            IncomeCeiling::NotExceeding(Sgd::dollars(12_000))
        );

        let on_the_day = t.on(&key, date(2019, 9, 11)).unwrap();
        assert_eq!(
            on_the_day.value,
            IncomeCeiling::NotExceeding(Sgd::dollars(14_000))
        );
    }

    #[test]
    fn a_2023_query_and_a_2025_query_get_the_provision_in_force_then() {
        let t = income_ceilings().unwrap();
        let key = (CeilingContext::NewFlat, HouseholdClass::Family);

        // The ceiling did not change between 2023 and 2025, so both land on the
        // same provision -- but they land on it *because it covers those dates*,
        // not because it is the last row of the table.
        let in_2023 = t.on(&key, date(2023, 6, 1)).unwrap();
        let in_2025 = t.on(&key, date(2025, 6, 1)).unwrap();
        assert_eq!(in_2023.value, in_2025.value);
        assert!(in_2023.effective.contains(date(2025, 6, 1)));

        // And a 2016 query must NOT get the 2019 figure.
        let in_2016 = t.on(&key, date(2016, 1, 1)).unwrap();
        assert_eq!(
            in_2016.value,
            IncomeCeiling::NotExceeding(Sgd::dollars(12_000))
        );
        assert_ne!(in_2016.value, in_2025.value);
    }

    #[test]
    fn a_query_before_2015_is_refused_rather_than_answered_with_the_2015_figure() {
        let t = income_ceilings().unwrap();
        let err = t
            .on(
                &(CeilingContext::NewFlat, HouseholdClass::Family),
                date(2014, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, TemporalError::BeforeEarliestProvision { .. }));
    }

    #[test]
    fn a_resale_purchase_without_a_grant_has_no_ceiling_as_a_rule_not_as_an_absence() {
        let t = income_ceilings().unwrap();
        let found = t
            .on(
                &(CeilingContext::ResaleWithoutGrant, HouseholdClass::Family),
                date(2025, 1, 1),
            )
            .unwrap();
        assert_eq!(found.value, IncomeCeiling::NoCeiling);
        assert!(
            found.value.admits(MonthlyIncome(Sgd::dollars(50_000))),
            "no ceiling means no ceiling"
        );
    }

    #[test]
    fn the_singles_resale_grant_ceiling_is_declared_unmodelled_not_absent() {
        let t = income_ceilings().unwrap();
        let err = t
            .on(
                &(CeilingContext::ResaleWithGrant, HouseholdClass::Single),
                date(2025, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, TemporalError::NotModelled { .. }));
    }

    #[test]
    fn ehg_family_matches_published_anchor_points() {
        // Three points pin the published 2019 table. If the generator drifts,
        // one of these breaks.
        let s = ehg_schedule_2019(HouseholdClass::Family);
        assert_eq!(s.brackets().len(), 16);

        assert_eq!(
            s.amount_for(MonthlyIncome(Sgd::dollars(1_500))),
            Some(GrantAmount(Sgd::dollars(80_000))),
            "$1,500 and below: the maximum grant"
        );
        assert_eq!(
            s.amount_for(MonthlyIncome(Sgd::dollars(1_501))),
            Some(GrantAmount(Sgd::dollars(75_000))),
            "one dollar over the first band drops a whole step"
        );
        assert_eq!(
            s.amount_for(MonthlyIncome(Sgd::dollars(9_000))),
            Some(GrantAmount(Sgd::dollars(5_000))),
            "the last band tops out at $9,000"
        );
        assert_eq!(
            s.amount_for(MonthlyIncome(Sgd::dollars(9_001))),
            None,
            "above the last band there is no EHG -- a fact, not an unknown"
        );
    }

    #[test]
    fn ehg_singles_receive_exactly_half_with_no_rounding_slop() {
        let family = ehg_schedule_2019(HouseholdClass::Family);
        let single = ehg_schedule_2019(HouseholdClass::Single);

        for (f, s) in family.brackets().iter().zip(single.brackets()) {
            assert_eq!(f.income_upper_inclusive, s.income_upper_inclusive);
            assert_eq!(
                s.amount.0.as_cents() * 2,
                f.amount.0.as_cents(),
                "the singles grant is exactly half, in cents"
            );
        }
        assert_eq!(
            single.amount_for(MonthlyIncome(Sgd::dollars(1_500))),
            Some(GrantAmount(Sgd::dollars(40_000)))
        );
    }

    #[test]
    fn the_ehg_stops_being_answerable_the_day_it_was_raised() {
        let t = enhanced_housing_grant().unwrap();

        let last_day = t.on(&HouseholdClass::Family, date(2024, 8, 19)).unwrap();
        assert_eq!(
            last_day
                .value
                .amount_for(MonthlyIncome(Sgd::dollars(1_500))),
            Some(GrantAmount(Sgd::dollars(80_000)))
        );

        let err = t
            .on(&HouseholdClass::Family, date(2024, 8, 20))
            .unwrap_err();
        match err {
            TemporalError::NotModelled { reason, .. } => {
                assert!(reason.contains("2024 National Day Rally"))
            }
            other => panic!("expected NotModelled on the day the grant was raised, got {other:?}"),
        }
    }

    #[test]
    fn mop_is_five_years_for_standard_and_ten_for_plus_and_prime() {
        let t = minimum_occupation_periods().unwrap();
        let after = date(2025, 1, 1);

        assert_eq!(
            t.on(&MopKey::Classified(FlatClassification::Standard), after)
                .unwrap()
                .value,
            MinimumOccupationPeriod(Months(60))
        );
        assert_eq!(
            t.on(&MopKey::Classified(FlatClassification::Plus), after)
                .unwrap()
                .value,
            MinimumOccupationPeriod(Months(120))
        );
        assert_eq!(
            t.on(&MopKey::Classified(FlatClassification::Prime), after)
                .unwrap()
                .value,
            MinimumOccupationPeriod(Months(120))
        );
    }

    #[test]
    fn a_classification_cannot_be_asked_about_before_it_existed() {
        let t = minimum_occupation_periods().unwrap();
        // "Plus" did not exist in 2023. Answering "five years" or "ten years"
        // would both be fabrications.
        let err = t
            .on(
                &MopKey::Classified(FlatClassification::Plus),
                date(2023, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, TemporalError::BeforeEarliestProvision { .. }));

        // And an unclassified flat cannot be bought after the framework began.
        let err = t.on(&MopKey::Unclassified, date(2025, 1, 1)).unwrap_err();
        assert!(matches!(err, TemporalError::AfterLatestProvision { .. }));

        // A flat bought in 2015 was unclassified, and its MOP was five years.
        assert_eq!(
            t.on(&MopKey::Unclassified, date(2015, 1, 1)).unwrap().value,
            MinimumOccupationPeriod(Months(60))
        );
    }

    #[test]
    fn eip_limits_are_per_group_and_at_two_scales() {
        let t = ethnic_quotas().unwrap();
        let malay = t.on(&EthnicGroup::Malay, date(2025, 1, 1)).unwrap();
        assert_eq!(malay.value.neighbourhood, Percent::whole(22));
        assert_eq!(malay.value.block, Percent::whole(25));

        // Not projected backwards onto a policy that had different limits.
        assert!(t.on(&EthnicGroup::Malay, date(1995, 1, 1)).is_err());
    }

    #[test]
    fn every_populated_figure_carries_a_citation() {
        // The property this crate exists to guarantee. If a provision could be
        // built without a citation, the type would allow it -- it cannot.
        let t = income_ceilings().unwrap();
        let found = t
            .on(
                &(CeilingContext::NewFlat, HouseholdClass::Family),
                date(2025, 1, 1),
            )
            .unwrap();
        assert!(!found.citation.url.is_empty());
        assert!(!found.citation.section.is_empty());
        assert_eq!(found.citation.published, ndr_2019());
    }

    #[test]
    fn every_citation_in_this_crate_admits_it_is_unverified() {
        // Until someone fetches the sources, this must hold. When it stops
        // holding, that is progress -- and the test should be tightened, not
        // deleted.
        let t = income_ceilings().unwrap();
        for (_, timeline) in t.timelines() {
            for provision in timeline.provisions() {
                if let Provision::InForce(dated) = provision {
                    assert!(
                        is_unverified(&dated.citation),
                        "a citation claims to be verified, but nothing in this crate has been \
                         checked against a source: {}",
                        dated.citation
                    );
                }
            }
        }
    }

    #[test]
    fn the_crate_can_enumerate_its_own_blind_spots() {
        assert!(UNMODELLED.len() >= 10);
        assert!(
            UNMODELLED
                .iter()
                .any(|u| u.area.contains("Ethnic Integration Policy")),
            "the EIP block-composition gap is the one most likely to mislead, and must be listed"
        );
    }
}
