//! The HDB domain: who is buying, what they are buying, and how.
//!
//! These are the *keys* the policy tables in [`rules`](super::rules) are indexed
//! by. They carry no numbers — every figure lives in a dated, cited table — but
//! getting the vocabulary right matters just as much, because a policy engine
//! that cannot distinguish a Permanent Resident from a citizen, or a resale
//! purchase from a Build-To-Order one, will confidently answer the wrong
//! question.
//!
//! Enums throughout. A flat type is not a string ("4-room", "4 room", "Four
//! Room"), and residency is not a `bool`.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::quantity::{Age, MonthlyIncome};

/// Residency status — the single input that changes the most.
///
/// It decides whether a household may buy a new flat at all, which grants it can
/// receive, whether the Ethnic Integration Policy's separate SPR quota applies,
/// and (for a household with no citizen) whether a waiting period must elapse
/// first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Residency {
    /// Singapore Citizen.
    SingaporeCitizen,
    /// Singapore Permanent Resident.
    PermanentResident,
    /// Neither — a foreigner, including a non-resident spouse.
    NonResident,
}

impl fmt::Display for Residency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Residency::SingaporeCitizen => "Singapore Citizen",
            Residency::PermanentResident => "Singapore Permanent Resident",
            Residency::NonResident => "non-resident",
        };
        f.write_str(s)
    }
}

/// The ethnic classification used by the Ethnic Integration Policy.
///
/// This crate takes no view on the classification; it records the one HDB's
/// quota rules are written in, because a model that omitted it could not
/// represent the policy at all. The scheme is coarse by design in the source
/// policy, and "Indian/Other" is a single quota group there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EthnicGroup {
    /// Chinese.
    Chinese,
    /// Malay.
    Malay,
    /// Indian and Other — one quota group under the EIP.
    IndianOther,
}

impl fmt::Display for EthnicGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EthnicGroup::Chinese => "Chinese",
            EthnicGroup::Malay => "Malay",
            EthnicGroup::IndianOther => "Indian/Other",
        };
        f.write_str(s)
    }
}

/// Whether an applicant has already enjoyed a housing subsidy.
///
/// Drives grant eligibility and the resale levy. "First-timer" is a status HDB
/// assigns, not something derivable from the fields this crate holds, so it is
/// an **input**, not a computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SubsidyHistory {
    /// Has not previously received a housing subsidy from HDB.
    FirstTimer,
    /// Has previously bought a subsidised flat or taken a housing grant.
    SecondTimer,
}

/// Marital status, to the extent that HDB's schemes turn on it.
///
/// The Single Singapore Citizen Scheme's age threshold differs for a widowed or
/// orphaned applicant, so "single" is not one category. Where this crate does
/// not model the consequences of a status, it says
/// [`Indeterminate`](super::eligibility::Eligibility::Indeterminate) rather than
/// pretending the status does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MaritalStatus {
    /// Never married.
    Single,
    /// Married.
    Married,
    /// Engaged, intending to marry.
    Engaged,
    /// Divorced.
    Divorced,
    /// Widowed.
    Widowed,
}

/// One person on the application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Applicant {
    /// Completed years of age on the date the assessment is made *for*.
    pub age: Age,
    /// Citizenship or residency.
    pub residency: Residency,
    /// EIP classification.
    pub ethnicity: EthnicGroup,
    /// Whether they have had a housing subsidy before.
    pub subsidy_history: SubsidyHistory,
    /// Marital status.
    pub marital_status: MaritalStatus,
}

/// How the applicants are related to one another — which is what selects the
/// candidate eligibility schemes.
///
/// This is stated by the household rather than inferred: HDB's schemes turn on
/// the *family nucleus*, and a nucleus is a legal relationship, not something
/// recoverable from ages and residencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FamilyNucleus {
    /// A married couple, with or without children; or an applicant with
    /// parents/children. The Public Scheme's nucleus.
    SpousesOrParentsChildren,
    /// A couple intending to marry. The Fiancé/Fiancée Scheme's nucleus.
    Engaged,
    /// One applicant, unmarried, divorced, widowed or orphaned.
    SingleApplicant,
    /// Two to four unrelated single applicants buying together.
    JointSingles,
    /// A citizen and a non-resident spouse.
    CitizenAndNonResidentSpouse,
    /// Orphaned siblings.
    OrphanedSiblings,
}

/// The household making the application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Household {
    /// Everyone on the application. At least one.
    pub applicants: Vec<Applicant>,
    /// How they are related.
    pub nucleus: FamilyNucleus,
    /// Average gross monthly household income, summed over the applicants and
    /// the essential occupiers.
    ///
    /// Supplied, not computed: HDB's assessment of income (12-month averaging,
    /// treatment of variable and self-employed income, whose income counts) is
    /// itself policy this crate does not model. See
    /// [`rules::UNMODELLED`](super::rules::UNMODELLED).
    pub monthly_income: MonthlyIncome,
}

impl Household {
    /// Whether at least one applicant is a Singapore Citizen — the pivot of
    /// almost every new-flat rule.
    pub fn has_citizen(&self) -> bool {
        self.applicants
            .iter()
            .any(|a| a.residency == Residency::SingaporeCitizen)
    }

    /// The youngest applicant's age. Age thresholds in HDB schemes apply to
    /// *each* applicant, so the binding constraint is the youngest.
    pub fn youngest(&self) -> Option<Age> {
        self.applicants.iter().map(|a| a.age).min()
    }

    /// Whether every applicant is a first-timer.
    pub fn all_first_timers(&self) -> bool {
        self.applicants
            .iter()
            .all(|a| a.subsidy_history == SubsidyHistory::FirstTimer)
    }

    /// The household class the ceiling and grant tables are keyed by.
    pub fn class(&self) -> HouseholdClass {
        match self.nucleus {
            FamilyNucleus::SingleApplicant | FamilyNucleus::JointSingles => HouseholdClass::Single,
            _ => HouseholdClass::Family,
        }
    }
}

/// The coarse split that income ceilings and grant schedules are keyed by:
/// HDB publishes one set of figures for families and (broadly) half for singles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HouseholdClass {
    /// A family nucleus.
    Family,
    /// A single applicant, or joint singles.
    Single,
}

impl fmt::Display for HouseholdClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HouseholdClass::Family => f.write_str("family"),
            HouseholdClass::Single => f.write_str("single"),
        }
    }
}

/// The flat types HDB sells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FlatType {
    /// 2-room Flexi — the only new flat type most singles may buy.
    TwoRoomFlexi,
    /// 3-room.
    ThreeRoom,
    /// 4-room.
    FourRoom,
    /// 5-room.
    FiveRoom,
    /// Executive (including Executive Maisonette, no longer built).
    Executive,
    /// 3Gen — for multi-generation families.
    ThreeGen,
}

impl fmt::Display for FlatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FlatType::TwoRoomFlexi => "2-room Flexi",
            FlatType::ThreeRoom => "3-room",
            FlatType::FourRoom => "4-room",
            FlatType::FiveRoom => "5-room",
            FlatType::Executive => "Executive",
            FlatType::ThreeGen => "3Gen",
        };
        f.write_str(s)
    }
}

/// A new flat's classification, which since the October 2024 sales exercise is
/// what determines its Minimum Occupation Period and its resale restrictions.
///
/// The framework replaced the old mature/non-mature estate split. `Plh` is kept
/// as a distinct variant rather than folded into `Prime` because flats sold
/// under the 2021 Prime Location Public Housing model were bought under *that*
/// model's terms, and a flat does not retroactively change the rules it was sold
/// under. Collapsing the two would quietly rewrite history for their owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FlatClassification {
    /// Standard — the majority of flats; the baseline terms.
    Standard,
    /// Plus — better-located flats, with tighter resale conditions.
    Plus,
    /// Prime — the choicest locations, with the tightest conditions.
    Prime,
    /// Prime Location Public Housing: the 2021–2024 model that preceded the
    /// Standard/Plus/Prime framework.
    Plh,
}

impl fmt::Display for FlatClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FlatClassification::Standard => "Standard",
            FlatClassification::Plus => "Plus",
            FlatClassification::Prime => "Prime",
            FlatClassification::Plh => "Prime Location Public Housing",
        };
        f.write_str(s)
    }
}

/// Whether the buyer intends to take a housing grant.
///
/// An enum rather than a `bool` because it changes the *rule that applies*, not
/// merely a flag: a resale flat bought without a grant has no income ceiling at
/// all, while the same flat bought with one does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GrantIntent {
    /// Applying for one or more housing grants.
    Applying,
    /// Not applying for any grant.
    NotApplying,
}

/// The sales exercise a new flat is bought through.
///
/// These share eligibility rules almost entirely; they are distinguished because
/// they do not share *everything* (application frequency, flat availability,
/// and, historically, some grant conditions), and collapsing them would make
/// those differences unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NewFlatSale {
    /// Build-To-Order: the main launch exercises.
    BuildToOrder,
    /// Sale of Balance Flats.
    SaleOfBalanceFlats,
    /// Open Booking of unsold flats.
    OpenBooking,
}

/// How the flat is being bought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PurchaseMode {
    /// Direct from HDB.
    NewFromHdb(NewFlatSale),
    /// On the resale market, from an existing owner.
    Resale {
        /// Whether a grant is being applied for — which decides whether an
        /// income ceiling applies at all.
        grant: GrantIntent,
    },
}

impl PurchaseMode {
    /// The key under which income ceilings are tabulated.
    pub fn ceiling_context(self) -> CeilingContext {
        match self {
            PurchaseMode::NewFromHdb(_) => CeilingContext::NewFlat,
            PurchaseMode::Resale {
                grant: GrantIntent::Applying,
            } => CeilingContext::ResaleWithGrant,
            PurchaseMode::Resale {
                grant: GrantIntent::NotApplying,
            } => CeilingContext::ResaleWithoutGrant,
        }
    }
}

impl fmt::Display for PurchaseMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder) => f.write_str("BTO"),
            PurchaseMode::NewFromHdb(NewFlatSale::SaleOfBalanceFlats) => {
                f.write_str("Sale of Balance Flats")
            }
            PurchaseMode::NewFromHdb(NewFlatSale::OpenBooking) => f.write_str("Open Booking"),
            PurchaseMode::Resale {
                grant: GrantIntent::Applying,
            } => f.write_str("resale (with grant)"),
            PurchaseMode::Resale {
                grant: GrantIntent::NotApplying,
            } => f.write_str("resale (no grant)"),
        }
    }
}

/// The context an income ceiling is tabulated against.
///
/// Three contexts, because HDB really does apply three different rules: a
/// ceiling for new flats, a ceiling for a resale purchase that takes a grant,
/// and *no ceiling at all* for a resale purchase that does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CeilingContext {
    /// A flat bought new from HDB.
    NewFlat,
    /// A resale flat bought with a housing grant.
    ResaleWithGrant,
    /// A resale flat bought without any housing grant.
    ResaleWithoutGrant,
}

/// The eligibility scheme an application is made under.
///
/// HDB does not have one eligibility rule; it has a family of *schemes*, and an
/// applicant qualifies (or does not) under each independently. This is why
/// [`assess`](super::HdbPolicy::assess) returns a verdict *per scheme* rather
/// than one overall yes/no: "you are eligible under the Public Scheme but not
/// the Fiancé/Fiancée Scheme" is a meaningful, actionable answer that a single
/// boolean destroys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EligibilityScheme {
    /// Public Scheme — a family nucleus of spouses, parents or children.
    Public,
    /// Fiancé/Fiancée Scheme — a couple intending to marry.
    FianceFiancee,
    /// Single Singapore Citizen Scheme.
    SingleSingaporeCitizen,
    /// Joint Singles Scheme — two to four singles buying together.
    JointSingles,
    /// Non-Citizen Spouse Scheme.
    NonCitizenSpouse,
    /// Non-Citizen Family Scheme.
    NonCitizenFamily,
    /// Orphans Scheme.
    Orphans,
    /// Conversion Scheme.
    Conversion,
}

impl fmt::Display for EligibilityScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EligibilityScheme::Public => "Public Scheme",
            EligibilityScheme::FianceFiancee => "Fiancé/Fiancée Scheme",
            EligibilityScheme::SingleSingaporeCitizen => "Single Singapore Citizen Scheme",
            EligibilityScheme::JointSingles => "Joint Singles Scheme",
            EligibilityScheme::NonCitizenSpouse => "Non-Citizen Spouse Scheme",
            EligibilityScheme::NonCitizenFamily => "Non-Citizen Family Scheme",
            EligibilityScheme::Orphans => "Orphans Scheme",
            EligibilityScheme::Conversion => "Conversion Scheme",
        };
        f.write_str(s)
    }
}

/// The housing grants this crate knows exist.
///
/// Knowing a grant *exists* and knowing its *amount* are different things, and
/// the difference is the point: several of these are present here with their
/// amount timelines deliberately left as
/// [`Provision::NotModelled`](super::temporal::Provision::NotModelled), so that
/// asking for them yields "this grant applies and we do not model its amount"
/// rather than silence — which a caller would read as "no grant".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Grant {
    /// Enhanced CPF Housing Grant (EHG). Income-tapered.
    EnhancedHousing,
    /// CPF Housing Grant for resale flats (the "Family Grant").
    Family,
    /// The singles equivalent of the Family Grant.
    Singles,
    /// Proximity Housing Grant — for buying near or with parents/children.
    Proximity,
}

impl fmt::Display for Grant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Grant::EnhancedHousing => "Enhanced CPF Housing Grant",
            Grant::Family => "CPF Housing Grant (Family)",
            Grant::Singles => "CPF Housing Grant (Singles)",
            Grant::Proximity => "Proximity Housing Grant",
        };
        f.write_str(s)
    }
}

/// What is being bought, how, and under what classification: the other half of a
/// [`Query`](super::eligibility::Query), alongside the [`Household`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Purchase {
    /// Direct from HDB, or on the resale market.
    pub mode: PurchaseMode,
    /// Which flat type.
    pub flat_type: FlatType,
    /// The flat's classification, where it has one. `None` for a resale flat
    /// bought before the Standard/Plus/Prime framework, or where the buyer does
    /// not know it — which is itself a reason for an
    /// [`Indeterminate`](super::eligibility::Eligibility::Indeterminate)
    /// Minimum Occupation Period rather than a guess at `Standard`.
    pub classification: Option<FlatClassification>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::policy::quantity::Sgd;

    fn applicant(age: u32, residency: Residency) -> Applicant {
        Applicant {
            age: Age(age),
            residency,
            ethnicity: EthnicGroup::Chinese,
            subsidy_history: SubsidyHistory::FirstTimer,
            marital_status: MaritalStatus::Married,
        }
    }

    #[test]
    fn household_reports_its_binding_age_and_citizenship() {
        let h = Household {
            applicants: vec![
                applicant(30, Residency::SingaporeCitizen),
                applicant(26, Residency::PermanentResident),
            ],
            nucleus: FamilyNucleus::SpousesOrParentsChildren,
            monthly_income: MonthlyIncome(Sgd::dollars(8_000)),
        };
        assert!(h.has_citizen());
        assert_eq!(h.youngest(), Some(Age(26)), "the youngest applicant binds");
        assert_eq!(h.class(), HouseholdClass::Family);
    }

    #[test]
    fn a_household_with_no_citizen_says_so() {
        let h = Household {
            applicants: vec![applicant(40, Residency::PermanentResident)],
            nucleus: FamilyNucleus::SingleApplicant,
            monthly_income: MonthlyIncome(Sgd::dollars(5_000)),
        };
        assert!(!h.has_citizen());
        assert_eq!(h.class(), HouseholdClass::Single);
    }

    #[test]
    fn grant_intent_selects_the_ceiling_context_not_a_flag() {
        assert_eq!(
            PurchaseMode::Resale {
                grant: GrantIntent::NotApplying
            }
            .ceiling_context(),
            CeilingContext::ResaleWithoutGrant
        );
        assert_eq!(
            PurchaseMode::Resale {
                grant: GrantIntent::Applying
            }
            .ceiling_context(),
            CeilingContext::ResaleWithGrant
        );
        assert_eq!(
            PurchaseMode::NewFromHdb(NewFlatSale::BuildToOrder).ceiling_context(),
            CeilingContext::NewFlat
        );
    }
}
