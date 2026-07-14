//! The query API: given a household, a purchase and a **date**, what do the
//! published rules say?
//!
//! # Eligibility is not a `bool`
//!
//! A boolean has two states, and this domain has three. A system that models it
//! with two must forge the third, and it will forge it as `false` — telling a
//! household "no" when the truthful answer was "the rule that decides your case
//! is one we do not model". That is not a rounding error. It is the difference
//! between a person applying for a flat and not applying for one.
//!
//! So [`Eligibility`] has three variants:
//!
//! * [`Eligible`](Eligibility::Eligible) — every rule this crate models is
//!   satisfied, and here are the citations for each of them.
//! * [`Ineligible`](Eligibility::Ineligible) — a modelled rule is definitively
//!   failed, and here is **which one, and where it is written**. Never a bare
//!   "no".
//! * [`Indeterminate`](Eligibility::Indeterminate) — the household's case turns
//!   on something this crate does not hold: a figure left unmodelled, a
//!   condition outside the model, a date before our earliest provision. Here is
//!   precisely what is missing.
//!
//! # How the three combine
//!
//! When a household definitively fails a modelled rule *and* some other rule is
//! unknown, the verdict is [`Ineligible`](Eligibility::Ineligible). A
//! definitively failed rule is decisive: an unmodelled rule can withhold a
//! *yes*, but it cannot overturn a *no*.
//!
//! The converse must be watched carefully, and is: a check is only allowed to
//! produce a [`Reason`] when the check is **fully modelled for that household**.
//! A widowed single applicant aged 30 does not get "ineligible: below 35",
//! because the 35 may not be their threshold — they get
//! [`Indeterminate`](Eligibility::Indeterminate) naming the gap. Emitting a
//! confident "no" from a rule that might not apply is the same failure as
//! emitting a confident number, wearing different clothes.
//!
//! # This is not advice
//!
//! [`Assessment`] reports what the published policy says, with citations. It
//! does not tell anyone what to do, it is not an application, it is not an
//! entitlement, and — given how much of HDB policy is deliberately unmodelled
//! here — it is not even a complete account of the rules. See the crate-level
//! documentation on [`policy`](super).

use serde::{Deserialize, Serialize};

use super::HdbPolicy;
use super::citation::Citation;
use super::domain::{
    EligibilityScheme, EthnicGroup, FamilyNucleus, Grant, Household, HouseholdClass, MaritalStatus,
    Purchase, PurchaseMode, Residency,
};
use super::quantity::{GrantAmount, IncomeCeiling, MinimumAge, MinimumOccupationPeriod, Months};
use super::rules::{EipLimits, MopKey};
use super::temporal::{Date, Dated, TemporalError};

/// A rule the household definitively fails, and where that rule is written.
///
/// A refusal without a citation is an assertion of authority. A refusal with one
/// is a claim that can be checked, and disputed, and corrected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reason {
    /// What the household fails, in plain words.
    pub statement: String,
    /// The rule it fails.
    pub citation: Citation,
}

/// Something this crate does not know, stated precisely enough to be acted on.
///
/// An `Unknown` is a *finding*, not an error. It is the crate's most valuable
/// output, because it is the part that a plausible-looking wrong answer would
/// have concealed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unknown {
    /// What could not be determined.
    pub subject: String,
    /// Why not.
    pub because: String,
    /// Where the answer would be found, when we know that much.
    pub see: Option<Citation>,
}

impl Unknown {
    /// An unknown arising from a policy table that holds nothing for that date.
    ///
    /// This is the bridge between the temporal layer's refusal to invent a
    /// figure and the eligibility layer's refusal to invent a verdict: a
    /// [`TemporalError`] *is* an [`Unknown`], and it propagates all the way to
    /// the caller with its explanation intact.
    pub fn from_lookup(subject: impl Into<String>, error: TemporalError) -> Self {
        Self {
            subject: subject.into(),
            because: error.to_string(),
            see: None,
        }
    }

    /// An unknown arising from a rule this crate has chosen not to model.
    pub fn unmodelled(
        subject: impl Into<String>,
        because: impl Into<String>,
        see: Option<Citation>,
    ) -> Self {
        Self {
            subject: subject.into(),
            because: because.into(),
            see,
        }
    }
}

/// A figure this crate either holds (with its date range and citation) or does
/// not (with an explanation).
///
/// Deliberately *not* `Option<T>`: an `Option` that is `None` says nothing about
/// why, and a caller who unwraps it into a default has silently invented policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Finding<T> {
    /// The rule in force on the queried date, with its citation.
    Known(Dated<T>),
    /// No figure, and precisely why not.
    Unknown(Unknown),
}

impl<T> Finding<T> {
    /// The figure, if this crate holds one.
    pub fn known(&self) -> Option<&Dated<T>> {
        match self {
            Finding::Known(dated) => Some(dated),
            Finding::Unknown(_) => None,
        }
    }

    /// The citation, if this crate holds a figure. Every known figure has one —
    /// the type makes it impossible not to.
    pub fn citation(&self) -> Option<&Citation> {
        self.known().map(|d| &d.citation)
    }

    /// Turns a table lookup into a `Finding`, preserving the reason for failure.
    fn from_lookup(subject: &str, result: Result<&Dated<T>, TemporalError>) -> Finding<T>
    where
        T: Clone,
    {
        match result {
            Ok(dated) => Finding::Known(dated.clone()),
            Err(e) => Finding::Unknown(Unknown::from_lookup(subject, e)),
        }
    }
}

/// The verdict under one eligibility scheme. See the module documentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Eligibility {
    /// Every modelled rule is satisfied. The citations are the rules that were
    /// checked — the receipt for the "yes".
    Eligible {
        /// The rules checked and passed.
        citations: Vec<Citation>,
    },
    /// A modelled rule is definitively failed.
    Ineligible {
        /// Which rules, and where they are written. Never empty.
        reasons: Vec<Reason>,
    },
    /// The case turns on something this crate does not model.
    Indeterminate {
        /// What is missing. Never empty.
        unknowns: Vec<Unknown>,
    },
}

/// The verdict under one scheme, and the scheme it is under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemeOutcome {
    /// The scheme assessed.
    pub scheme: EligibilityScheme,
    /// The verdict.
    pub eligibility: Eligibility,
}

/// What a grant assessment can say.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GrantOutcome {
    /// An **indicative** amount, from the schedule in force on the queried date.
    ///
    /// Indicative, not entitled: the amount follows from income alone, and grant
    /// eligibility additionally turns on first-timer status, prior subsidies,
    /// remaining lease, and conditions this crate does not model. Those appear
    /// in [`Assessment::caveats`].
    Indicative(Dated<GrantAmount>),
    /// The household's income is above every band of the schedule. **This is a
    /// fact, not an absence**: the grant is not payable, and here is the
    /// schedule that says so.
    NotPayable(Reason),
    /// The amount is not modelled on that date.
    Indeterminate(Unknown),
}

/// A grant, and what can be said about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrantFinding {
    /// The grant.
    pub grant: Grant,
    /// What the policy tables can say about it, on the queried date.
    pub outcome: GrantOutcome,
}

/// The Ethnic Integration Policy position: the limits, and the reason the limits
/// do not answer the buyer's actual question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EthnicQuotaFinding {
    /// The applicant ethnic group the limits were looked up for.
    pub group: EthnicGroup,
    /// The block and neighbourhood limits.
    pub limits: Finding<EipLimits>,
    /// The separate SPR quota, present only when a Permanent Resident is on the
    /// application.
    pub spr_quota: Option<Finding<EipLimits>>,
    /// Always populated: whether a *particular* block has quota space depends on
    /// its current composition, which this crate does not hold.
    pub availability: Unknown,
}

/// A question put to the policy tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// Who is buying.
    pub household: Household,
    /// What, and how.
    pub purchase: Purchase,
    /// **On what date.** Not "now": the crate has no clock, and an answer that
    /// silently depended on one would not be reproducible. For a purchase this
    /// is the application date, which is what HDB's rules attach to.
    pub as_of: Date,
}

/// What the published rules say about a [`Query`], with citations throughout.
///
/// Not advice. Not an application. Not an entitlement. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assessment {
    /// The date assessed. Every figure below is the one in force *then*.
    pub as_of: Date,
    /// The verdict under each scheme the household's nucleus makes available.
    pub schemes: Vec<SchemeOutcome>,
    /// The income ceiling for this purchase.
    pub income_ceiling: Finding<IncomeCeiling>,
    /// The grants, each with an indicative amount or an honest gap.
    pub grants: Vec<GrantFinding>,
    /// The Minimum Occupation Period attaching to this flat.
    pub minimum_occupation_period: Finding<MinimumOccupationPeriod>,
    /// The Ethnic Integration Policy position.
    pub ethnic_quota: EthnicQuotaFinding,
    /// Conditions and rules that bear on this household and that the crate does
    /// not check. Read these before reading anything else.
    pub caveats: Vec<Unknown>,
}

impl Assessment {
    /// Whether any scheme returned [`Eligibility::Eligible`].
    ///
    /// A convenience, and a slightly dangerous one — it collapses exactly the
    /// distinction this module exists to preserve. It is offered because callers
    /// legitimately need to filter, and it is named so that its return value
    /// cannot be mistaken for "you can buy a flat".
    pub fn any_scheme_eligible(&self) -> bool {
        self.schemes
            .iter()
            .any(|s| matches!(s.eligibility, Eligibility::Eligible { .. }))
    }

    /// Every citation supporting anything in this assessment.
    ///
    /// The audit trail. If this is empty, the assessment said nothing.
    pub fn citations(&self) -> Vec<&Citation> {
        let mut out: Vec<&Citation> = Vec::new();
        for scheme in &self.schemes {
            match &scheme.eligibility {
                Eligibility::Eligible { citations } => out.extend(citations),
                Eligibility::Ineligible { reasons } => {
                    out.extend(reasons.iter().map(|r| &r.citation))
                }
                Eligibility::Indeterminate { unknowns } => {
                    out.extend(unknowns.iter().filter_map(|u| u.see.as_ref()))
                }
            }
        }
        out.extend(self.income_ceiling.citation());
        out.extend(self.minimum_occupation_period.citation());
        out.extend(self.ethnic_quota.limits.citation());
        for grant in &self.grants {
            match &grant.outcome {
                GrantOutcome::Indicative(dated) => out.push(&dated.citation),
                GrantOutcome::NotPayable(reason) => out.push(&reason.citation),
                GrantOutcome::Indeterminate(unknown) => out.extend(unknown.see.as_ref()),
            }
        }
        out
    }
}

/// Accumulates the checks for one scheme, then resolves them into a verdict.
///
/// The resolution order encodes the rule stated in the module docs: a
/// definitively failed modelled rule beats an unmodelled one.
#[derive(Default)]
struct Checks {
    citations: Vec<Citation>,
    reasons: Vec<Reason>,
    unknowns: Vec<Unknown>,
}

impl Checks {
    /// A modelled rule the household satisfies.
    fn passed(&mut self, citation: Citation) {
        self.citations.push(citation);
    }

    /// A modelled rule the household definitively fails.
    fn failed(&mut self, statement: impl Into<String>, citation: Citation) {
        self.reasons.push(Reason {
            statement: statement.into(),
            citation,
        });
    }

    /// A rule that bears on this household and that we cannot evaluate.
    fn unknown(&mut self, unknown: Unknown) {
        self.unknowns.push(unknown);
    }

    fn resolve(self) -> Eligibility {
        if !self.reasons.is_empty() {
            Eligibility::Ineligible {
                reasons: self.reasons,
            }
        } else if !self.unknowns.is_empty() {
            Eligibility::Indeterminate {
                unknowns: self.unknowns,
            }
        } else {
            Eligibility::Eligible {
                citations: self.citations,
            }
        }
    }
}

impl HdbPolicy {
    /// Assesses a household against the rules in force on [`Query::as_of`].
    ///
    /// Every figure in the result carries its citation, and every gap in the
    /// result says what is missing. See [`Assessment`].
    pub fn assess(&self, query: &Query) -> Assessment {
        let household = &query.household;
        let purchase = query.purchase;
        let as_of = query.as_of;

        let ceiling = self.income_ceiling_finding(household, purchase, as_of);

        let schemes = self
            .candidate_schemes(household.nucleus)
            .into_iter()
            .map(|scheme| SchemeOutcome {
                scheme,
                eligibility: self.assess_scheme(scheme, household, purchase, as_of, &ceiling),
            })
            .collect();

        Assessment {
            as_of,
            schemes,
            income_ceiling: ceiling,
            grants: self.grant_findings(household, as_of),
            minimum_occupation_period: self.mop_finding(purchase, as_of),
            ethnic_quota: self.ethnic_quota_finding(household, as_of),
            caveats: self.caveats(household, purchase),
        }
    }

    /// The schemes a nucleus makes available. A nucleus this crate does not model
    /// yields no schemes, and the caller learns why from
    /// [`Assessment::caveats`] — not from an empty list, which says nothing.
    fn candidate_schemes(&self, nucleus: FamilyNucleus) -> Vec<EligibilityScheme> {
        match nucleus {
            FamilyNucleus::SpousesOrParentsChildren => vec![EligibilityScheme::Public],
            FamilyNucleus::Engaged => vec![EligibilityScheme::FianceFiancee],
            FamilyNucleus::SingleApplicant => vec![EligibilityScheme::SingleSingaporeCitizen],
            FamilyNucleus::JointSingles => vec![EligibilityScheme::JointSingles],
            FamilyNucleus::CitizenAndNonResidentSpouse => vec![EligibilityScheme::NonCitizenSpouse],
            FamilyNucleus::OrphanedSiblings => vec![EligibilityScheme::Orphans],
        }
    }

    fn income_ceiling_finding(
        &self,
        household: &Household,
        purchase: Purchase,
        as_of: Date,
    ) -> Finding<IncomeCeiling> {
        let key = (purchase.mode.ceiling_context(), household.class());
        Finding::from_lookup("income ceiling", self.income_ceilings.on(&key, as_of))
    }

    /// The per-scheme checks.
    fn assess_scheme(
        &self,
        scheme: EligibilityScheme,
        household: &Household,
        purchase: Purchase,
        as_of: Date,
        ceiling: &Finding<IncomeCeiling>,
    ) -> Eligibility {
        let mut checks = Checks::default();

        self.check_citizenship(scheme, household, purchase, as_of, &mut checks);
        self.check_age(scheme, household, as_of, &mut checks);
        self.check_income(household, ceiling, &mut checks);
        self.check_scheme_shape(scheme, household, &mut checks);

        checks.resolve()
    }

    /// Residency: the input that changes almost everything.
    fn check_citizenship(
        &self,
        scheme: EligibilityScheme,
        household: &Household,
        purchase: Purchase,
        as_of: Date,
        checks: &mut Checks,
    ) {
        let eligibility_page = self.eligibility_page.clone();

        if household.has_citizen() {
            checks.passed(eligibility_page);
        } else {
            match purchase.mode {
                PurchaseMode::NewFromHdb(_) => checks.failed(
                    "a flat bought from HDB requires at least one Singapore Citizen applicant; \
                     this household has none",
                    eligibility_page,
                ),
                PurchaseMode::Resale { .. } => {
                    // A household of Permanent Residents may buy a resale flat,
                    // but only after each has held PR status for three years --
                    // and this crate is not told when PR was granted. Answering
                    // "yes" would be a guess; answering "no" would be a
                    // different guess. The truthful answer is neither.
                    let see = self
                        .spr_resale_waiting_period
                        .on(as_of)
                        .ok()
                        .map(|d| d.citation.clone());
                    checks.unknown(Unknown::unmodelled(
                        "Permanent Resident household buying a resale flat",
                        "a household with no Singapore Citizen must have held Permanent Resident \
                         status for three years before buying a resale flat. This crate is not \
                         given the date PR was granted, so it cannot say whether the waiting \
                         period has been served.",
                        see,
                    ));
                }
            }
        }

        if scheme == EligibilityScheme::NonCitizenSpouse {
            checks.unknown(Unknown::unmodelled(
                "Non-Citizen Spouse Scheme: what may be bought",
                "the scheme restricts which flat types and purchase modes are open to a citizen \
                 with a non-resident spouse, and those restrictions are not modelled here.",
                None,
            ));
        }
    }

    /// Age thresholds, from the dated table — never from a constant.
    fn check_age(
        &self,
        scheme: EligibilityScheme,
        household: &Household,
        as_of: Date,
        checks: &mut Checks,
    ) {
        let Some(youngest) = household.youngest() else {
            checks.unknown(Unknown::unmodelled(
                "applicant age",
                "the household has no applicants",
                None,
            ));
            return;
        };

        // A widowed or orphaned single applicant is subject to a lower threshold
        // than 35, and this crate does not hold it. Applying the 35 to them
        // would produce a confident, cited, WRONG "no" -- the exact failure this
        // design exists to prevent.
        if scheme == EligibilityScheme::SingleSingaporeCitizen
            && household
                .applicants
                .iter()
                .any(|a| a.marital_status == MaritalStatus::Widowed)
        {
            checks.unknown(Unknown::unmodelled(
                "minimum age under the Single Singapore Citizen Scheme for a widowed applicant",
                "a widowed or orphaned applicant is subject to a lower minimum age than an \
                 unmarried one. That threshold is not modelled, so the 35-year threshold is NOT \
                 applied to this household.",
                None,
            ));
            return;
        }

        match self.minimum_ages.on(&scheme, as_of) {
            Ok(dated) => {
                let MinimumAge(minimum) = dated.value;
                if dated.value.admits(youngest) {
                    checks.passed(dated.citation.clone());
                } else {
                    checks.failed(
                        format!(
                            "the youngest applicant is {youngest}; the {scheme} requires every \
                             applicant to be at least {minimum}"
                        ),
                        dated.citation.clone(),
                    );
                }
            }
            Err(e) => checks.unknown(Unknown::from_lookup(
                format!("minimum age under the {scheme}"),
                e,
            )),
        }
    }

    /// Income against the ceiling in force. The boundary is inclusive; see
    /// [`IncomeCeiling::admits`].
    fn check_income(
        &self,
        household: &Household,
        ceiling: &Finding<IncomeCeiling>,
        checks: &mut Checks,
    ) {
        match ceiling {
            Finding::Known(dated) => {
                if dated.value.admits(household.monthly_income) {
                    checks.passed(dated.citation.clone());
                } else {
                    checks.failed(
                        format!(
                            "the household's income of {} exceeds the ceiling in force on this \
                             date ({})",
                            household.monthly_income, dated.value
                        ),
                        dated.citation.clone(),
                    );
                }
            }
            Finding::Unknown(unknown) => checks.unknown(unknown.clone()),
        }
    }

    /// The structural requirements of each scheme: how many applicants, of what
    /// residency.
    fn check_scheme_shape(
        &self,
        scheme: EligibilityScheme,
        household: &Household,
        checks: &mut Checks,
    ) {
        let eligibility_page = self.eligibility_page.clone();
        let count = household.applicants.len();

        match scheme {
            EligibilityScheme::Public => {
                if count < 2 {
                    checks.failed(
                        "the Public Scheme requires a family nucleus of at least two people",
                        eligibility_page,
                    );
                }
            }
            EligibilityScheme::FianceFiancee => {
                if count != 2 {
                    checks.failed(
                        "the Fiancé/Fiancée Scheme is for a couple: exactly two applicants",
                        eligibility_page,
                    );
                }
            }
            EligibilityScheme::JointSingles => {
                if !(2..=4).contains(&count) {
                    checks.failed(
                        format!(
                            "the Joint Singles Scheme is for two to four single applicants; this \
                             application has {count}"
                        ),
                        eligibility_page,
                    );
                } else if household
                    .applicants
                    .iter()
                    .any(|a| a.residency != Residency::SingaporeCitizen)
                {
                    checks.failed(
                        "every applicant under the Joint Singles Scheme must be a Singapore \
                         Citizen",
                        eligibility_page,
                    );
                }
            }
            EligibilityScheme::SingleSingaporeCitizen => {
                if count != 1 {
                    checks.failed(
                        "the Single Singapore Citizen Scheme is for one applicant",
                        eligibility_page,
                    );
                }
            }
            EligibilityScheme::Orphans
            | EligibilityScheme::Conversion
            | EligibilityScheme::NonCitizenFamily => {
                checks.unknown(Unknown::unmodelled(
                    format!("{scheme}"),
                    "this crate knows the scheme exists and does not model its conditions. It is \
                     listed rather than omitted, because an omitted scheme reads as a scheme \
                     that does not apply.",
                    None,
                ));
            }
            EligibilityScheme::NonCitizenSpouse => {
                if count != 2 {
                    checks.failed(
                        "the Non-Citizen Spouse Scheme is for a citizen and their spouse: \
                         exactly two applicants",
                        eligibility_page,
                    );
                }
            }
        }
    }

    /// Indicative grant amounts, and honest gaps where the amounts are not held.
    fn grant_findings(&self, household: &Household, as_of: Date) -> Vec<GrantFinding> {
        let class = household.class();
        let mut out = Vec::new();

        // The Enhanced CPF Housing Grant: an income-tapered schedule.
        let ehg = match self.enhanced_housing_grant.on(&class, as_of) {
            Ok(dated) => match dated.value.amount_for(household.monthly_income) {
                Some(amount) => GrantOutcome::Indicative(Dated::new(
                    amount,
                    dated.effective,
                    dated.citation.clone(),
                )),
                None => GrantOutcome::NotPayable(Reason {
                    statement: format!(
                        "the household's income of {} is above the highest band of the Enhanced \
                         CPF Housing Grant schedule in force on this date",
                        household.monthly_income
                    ),
                    citation: dated.citation.clone(),
                }),
            },
            Err(e) => GrantOutcome::Indeterminate(Unknown::from_lookup(
                "Enhanced CPF Housing Grant amount",
                e,
            )),
        };
        out.push(GrantFinding {
            grant: Grant::EnhancedHousing,
            outcome: ehg,
        });

        // The grants whose amounts this crate declares but does not hold. They
        // are reported precisely so that their absence cannot be read as "does
        // not apply".
        let others = match class {
            HouseholdClass::Family => [Grant::Family, Grant::Proximity],
            HouseholdClass::Single => [Grant::Singles, Grant::Proximity],
        };
        for grant in others {
            let outcome = match self.other_grants.on(&(grant, class), as_of) {
                // Unreachable while every entry is NotModelled, but the code must
                // not assume that stays true once someone populates the table.
                Ok(dated) => match dated.value.amount_for(household.monthly_income) {
                    Some(amount) => GrantOutcome::Indicative(Dated::new(
                        amount,
                        dated.effective,
                        dated.citation.clone(),
                    )),
                    None => GrantOutcome::NotPayable(Reason {
                        statement: format!(
                            "the household's income of {} is above the highest band of the {grant} \
                             schedule in force on this date",
                            household.monthly_income
                        ),
                        citation: dated.citation.clone(),
                    }),
                },
                Err(e) => {
                    GrantOutcome::Indeterminate(Unknown::from_lookup(format!("{grant} amount"), e))
                }
            };
            out.push(GrantFinding { grant, outcome });
        }

        out
    }

    /// The Minimum Occupation Period attaching to the flat being bought.
    ///
    /// Note what is *not* asked: whether the household has served it. That needs
    /// a key-collection date and an occupation record, and this crate holds
    /// neither.
    fn mop_finding(&self, purchase: Purchase, as_of: Date) -> Finding<MinimumOccupationPeriod> {
        let key = match purchase.classification {
            Some(classification) => MopKey::Classified(classification),
            // No guess at `Standard`. If the flat has a classification and the
            // caller did not supply it, the honest answer is that we do not know
            // the MOP -- and for a purchase after October 2024 the Unclassified
            // timeline has ended, so the lookup says exactly that.
            None => MopKey::Unclassified,
        };
        Finding::from_lookup(
            "Minimum Occupation Period",
            self.minimum_occupation_periods.on(&key, as_of),
        )
    }

    /// The EIP limits, plus the reason they do not settle the buyer's question.
    fn ethnic_quota_finding(&self, household: &Household, as_of: Date) -> EthnicQuotaFinding {
        // The quota is assessed against the applicants' ethnic group. Where a
        // household spans groups, HDB applies rules this crate does not model;
        // the first applicant's group is used and the gap is declared.
        let group = household
            .applicants
            .first()
            .map(|a| a.ethnicity)
            .unwrap_or(EthnicGroup::IndianOther);

        let limits = Finding::from_lookup(
            "Ethnic Integration Policy limits",
            self.ethnic_quotas.on(&group, as_of),
        );

        let spr_quota = household
            .applicants
            .iter()
            .any(|a| a.residency == Residency::PermanentResident)
            .then(|| {
                Finding::from_lookup(
                    "Singapore Permanent Resident quota",
                    self.spr_quota.on(as_of),
                )
            });

        let availability = Unknown::unmodelled(
            "whether a particular block or neighbourhood has quota space",
            "the Ethnic Integration Policy is applied against the current ethnic composition of \
             the block and the neighbourhood. This crate holds the limits, not the composition, \
             so it cannot say whether any specific flat may be sold to this household. HDB \
             publishes the live position per block.",
            limits.citation().cloned(),
        );

        EthnicQuotaFinding {
            group,
            limits,
            spr_quota,
            availability,
        }
    }

    /// Rules bearing on this household that the crate does not check at all.
    ///
    /// Read before anything else in the [`Assessment`].
    fn caveats(&self, household: &Household, purchase: Purchase) -> Vec<Unknown> {
        let mut caveats = vec![Unknown::unmodelled(
            "grant eligibility beyond income",
            "grant amounts reported here follow from income alone. Whether a grant is actually \
             payable also turns on first-timer status, prior housing subsidies, the flat's \
             remaining lease against the applicants' ages, and other conditions this crate does \
             not model.",
            None,
        )];

        if household.nucleus == FamilyNucleus::Engaged {
            caveats.push(Unknown::unmodelled(
                "the Fiancé/Fiancée Scheme's marriage condition",
                "the scheme requires the couple to marry within three months of collecting the \
                 keys. That is a future event; this crate cannot check it and does not treat it \
                 as satisfied.",
                None,
            ));
        }

        if !household.all_first_timers() {
            caveats.push(Unknown::unmodelled(
                "resale levy",
                "at least one applicant has had a housing subsidy before. A resale levy is \
                 payable on taking a second subsidy; its amount is not modelled here.",
                None,
            ));
        }

        if purchase.classification.is_none() {
            caveats.push(Unknown::unmodelled(
                "flat classification",
                "no classification was supplied. Since the October 2024 sales exercise, every \
                 flat is Standard, Plus or Prime, and the Minimum Occupation Period and resale \
                 conditions follow from that classification. It has NOT been assumed to be \
                 Standard.",
                None,
            ));
        }

        caveats.push(Unknown::unmodelled(
            "flat-type restrictions",
            format!(
                "which schemes may buy a {} is not modelled. In particular, single applicants \
                 buying new flats face flat-type restrictions this crate does not hold.",
                purchase.flat_type
            ),
            None,
        ));

        caveats
    }
}

impl HdbPolicy {
    /// The waiting period a household with no Singapore Citizen must serve
    /// before buying a resale flat, as in force on `as_of`.
    pub fn spr_resale_waiting_period(&self, as_of: Date) -> Finding<Months> {
        Finding::from_lookup(
            "SPR household resale waiting period",
            self.spr_resale_waiting_period.on(as_of),
        )
    }
}
