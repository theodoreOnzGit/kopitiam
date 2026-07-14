//! The health-insurance vocabulary: wards, panels, deductibles, limits.
//!
//! **This module is health-specific.** It is the reason `kopitiam-health`
//! exists as a crate distinct from `kopitiam-insurance`: a motor policy has no
//! ward class, and a travel policy has no panel of specialists.
//!
//! Everything here models **what a document says**, in the document's own
//! terms. Not one figure is hardcoded. The types describe the *shape* of a
//! health policy; the *numbers* come from a clause, or they do not exist.

use std::fmt;

use serde::{Deserialize, Serialize};

use kopitiam_insurance::{Currency, MonetaryAmount, Percentage};

/// Renders an amount **as the document printed it**, currency and all.
///
/// A bare `$` stays a bare `$`. It is not silently rendered as `SGD`, because a
/// reader shown `SGD 3,500` for a clause that says `$3,500` has been told
/// something the document did not say — and in a wording that also mentions US
/// dollars, that is not a pedantic distinction.
fn show(amount: &MonetaryAmount) -> String {
    let figure = amount.amount().to_decimal_string();
    match amount.currency() {
        Currency::Iso(code) => format!("{code} {figure}"),
        Currency::Ambiguous(symbol) => format!("{symbol}{figure} (currency not stated)"),
        Currency::Unstated => format!("{figure} (currency not stated)"),
    }
}

/// A span of time as the policy states it.
///
/// # Why there is no `as_days()`
///
/// Because "a 12-month waiting period" is not 365 days, and it is not 360, and
/// which one it is depends on the policy's own definition of a month — which
/// most wordings do not give. Converting between these units would manufacture
/// a precision the document does not have, and a waiting period that is one day
/// out is the difference between a claim admitted and a claim denied.
///
/// So the units do not convert. If you need to compare a 12-month waiting period
/// with a 365-day one, that comparison is a human's to make, and this type makes
/// you notice that you are making it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDuration {
    /// A number of days, as stated.
    Days(u32),
    /// A number of months, as stated. Not convertible to days — see above.
    Months(u32),
    /// A number of years, as stated.
    Years(u32),
}

impl fmt::Display for PolicyDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (n, unit) = match *self {
            Self::Days(n) => (n, "day"),
            Self::Months(n) => (n, "month"),
            Self::Years(n) => (n, "year"),
        };
        write!(f, "{n} {unit}{}", if n == 1 { "" } else { "s" })
    }
}

/// The class of hospital accommodation a benefit is stated for.
///
/// # Why coverage depends on this so heavily
///
/// In Singapore, the same operation in the same hospital costs a very different
/// amount depending on the ward you are admitted to, because the public
/// (restructured) hospitals apply a means-tested government subsidy that varies
/// by ward class. Insurers price against that: a plan written for Class B1
/// wards states different deductibles, co-insurance and limits from one written
/// for private hospitals, and admitting yourself to a ward above your plan's
/// class is a well-known way to discover that your cover has quietly shrunk.
///
/// So a term is rarely a property of the policy alone. It is a property of
/// *(policy, ward class)*, which is why [`crate::Scope`] exists.
///
/// The class names here are the structural nomenclature of Singapore's public
/// hospital system, not any insurer's terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WardClass {
    /// Public (restructured) hospital, Class C — most heavily subsidised.
    PublicC,
    /// Public (restructured) hospital, Class B2.
    PublicB2,
    /// Public (restructured) hospital, Class B1.
    PublicB1,
    /// Public (restructured) hospital, Class A — unsubsidised.
    PublicA,
    /// Private hospital.
    Private,
    /// Day surgery / same-day discharge: no overnight ward at all. Many
    /// wordings state a separate deductible for this, precisely because it is
    /// not a ward class.
    DaySurgery,
    /// A class this crate does not have a variant for, kept as the document
    /// writes it rather than being forced into the nearest match. Forcing it
    /// would be a silent reclassification, and reclassifying someone's ward is
    /// how you accidentally change the deductible that applies to them.
    Other(String),
}

impl fmt::Display for WardClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PublicC => f.write_str("Public Class C"),
            Self::PublicB2 => f.write_str("Public Class B2"),
            Self::PublicB1 => f.write_str("Public Class B1"),
            Self::PublicA => f.write_str("Public Class A"),
            Self::Private => f.write_str("Private hospital"),
            Self::DaySurgery => f.write_str("Day surgery"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

/// Whether the treating doctor or hospital is on the insurer's panel.
///
/// Insurers negotiate fees with a *panel* of providers. Treatment outside the
/// panel is frequently subject to a higher co-insurance rate, a lower limit, or
/// both — and riders in particular tend to be far less generous off-panel. It
/// is one of the most consequential and least-read variables in a policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderNetwork {
    /// On the insurer's panel.
    Panel,
    /// Not on the insurer's panel.
    NonPanel,
    /// Emergency admission. Many wordings treat this as if it were on-panel
    /// regardless of the provider — but *only if the wording says so*, which is
    /// why this is a distinct variant and not an alias for `Panel`.
    Emergency,
}

impl fmt::Display for ProviderNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Panel => "panel",
            Self::NonPanel => "non-panel",
            Self::Emergency => "emergency",
        })
    }
}

/// The circumstances of a treatment, against which a policy's terms are
/// resolved.
///
/// This is deliberately thin. It carries what changes *which clause applies*,
/// and nothing that would tempt the crate into adjudicating a claim. There is,
/// for instance, no diagnosis field: this crate does not decide whether a
/// condition is pre-existing, and giving it a diagnosis would invite it to try.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreatmentContext {
    /// The ward the patient was (or would be) admitted to.
    pub ward: WardClass,
    /// Whether the provider is on the insurer's panel.
    pub provider: ProviderNetwork,
}

impl TreatmentContext {
    /// Describes a treatment.
    pub fn new(ward: WardClass, provider: ProviderNetwork) -> Self {
        Self { ward, provider }
    }
}

impl fmt::Display for TreatmentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.ward, self.provider)
    }
}

/// What a term applies to.
///
/// A wording states many deductibles, not one: a different figure for each ward
/// class, sometimes a different one on and off panel. `Scope` records which
/// combination a given extracted term was stated for.
///
/// `None` in a field means **unscoped** — the clause stated the term without
/// qualifying it, so it applies generally. It does *not* mean "we did not
/// look"; a rule that fails to determine the scope produces an ambiguity, not a
/// `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scope {
    /// The ward class this term was stated for, if the clause qualified it.
    pub ward: Option<WardClass>,
    /// The panel status this term was stated for, if the clause qualified it.
    pub provider: Option<ProviderNetwork>,
}

impl Scope {
    /// A term the document stated without qualification.
    pub fn any() -> Self {
        Self::default()
    }

    /// A term stated for one ward class.
    pub fn ward(ward: WardClass) -> Self {
        Self {
            ward: Some(ward),
            provider: None,
        }
    }

    /// A term stated for one ward class and one panel status.
    pub fn ward_and_provider(ward: WardClass, provider: ProviderNetwork) -> Self {
        Self {
            ward: Some(ward),
            provider: Some(provider),
        }
    }

    /// Whether this term is stated for the given treatment.
    ///
    /// An unscoped term matches everything; a scoped one matches only its own
    /// scope. Note there is no fuzzy matching and no "nearest ward class": if a
    /// wording states a deductible for Class B1 and the patient is in Class A,
    /// the B1 clause does **not** apply, and the caller ends up with a
    /// [`crate::cost_share::CostShareRefusal::MissingTerm`]. That refusal is
    /// correct. Quietly reusing the B1 figure for a Class A stay would produce
    /// a number that is wrong in the patient's favour, which is the most
    /// harmful direction to be wrong in.
    pub fn applies_to(&self, ctx: &TreatmentContext) -> bool {
        let ward_ok = self.ward.as_ref().is_none_or(|w| *w == ctx.ward);
        let provider_ok = self.provider.is_none_or(|p| p == ctx.provider);
        ward_ok && provider_ok
    }

    /// How specific this scope is: used to prefer a ward-and-panel-specific
    /// clause over a general one when both apply.
    pub(crate) fn specificity(&self) -> u8 {
        u8::from(self.ward.is_some()) + u8::from(self.provider.is_some())
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.ward, &self.provider) {
            (None, None) => f.write_str("all treatments"),
            (Some(w), None) => write!(f, "{w}"),
            (None, Some(p)) => write!(f, "{p}"),
            (Some(w), Some(p)) => write!(f, "{w} ({p})"),
        }
    }
}

/// What the deductible is charged against.
///
/// Materially different: a S$3,500 per-policy-year deductible is paid once a
/// year however many times you are admitted; the same figure per *claim* is
/// paid every admission. Conflating them can be an order-of-magnitude error for
/// a patient with a chronic condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeductibleBasis {
    /// Once per policy year, across all claims.
    PerPolicyYear,
    /// On every claim.
    PerClaim,
    /// On every hospital confinement.
    PerConfinement,
}

impl fmt::Display for DeductibleBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PerPolicyYear => "per policy year",
            Self::PerClaim => "per claim",
            Self::PerConfinement => "per confinement",
        })
    }
}

/// The first slice of the bill, which the insured bears before the insurer pays
/// anything at all.
///
/// The amount is a [`MonetaryAmount`] — the figure **as the document printed
/// it**, currency included, which may be [`kopitiam_insurance::Currency::Ambiguous`]
/// if the wording wrote a bare `$`. It is deliberately not a computable amount:
/// converting it into one is [`crate::money::Amount::try_from_extracted`]'s job,
/// and that is where an unstated currency becomes a refusal rather than a guess.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deductible {
    /// The amount, exactly as the document printed it.
    pub amount: MonetaryAmount,
    /// What it is charged against.
    pub basis: DeductibleBasis,
}

impl fmt::Display for Deductible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", show(&self.amount), self.basis)
    }
}

/// The share of the bill *above the deductible* that the insured continues to
/// bear.
///
/// A separate type from a bare [`Rate`] because co-insurance and a rider's
/// co-payment rate are both rates, and handing them to the wrong step of the
/// calculation is exactly the kind of mistake the type system should be made to
/// catch. The `cap` is the annual ceiling some wordings put on the insured's
/// co-insurance; `None` means **the document did not state one**, not "there
/// isn't one".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoInsurance {
    /// The rate, as stated.
    pub rate: Percentage,
    /// A cap on the insured's co-insurance, if the document states one.
    pub cap: Option<MonetaryAmount>,
}

impl fmt::Display for CoInsurance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}%", self.rate.to_decimal_string())?;
        match &self.cap {
            Some(cap) => write!(f, " (capped at {})", show(cap)),
            None => Ok(()),
        }
    }
}

/// A ceiling on what the insurer will pay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimLimit {
    /// A ceiling on a single claim.
    PerClaim(MonetaryAmount),
    /// A ceiling on everything claimed in one policy year.
    PerPolicyYear(MonetaryAmount),
    /// A ceiling on everything claimed over the life of the policy.
    Lifetime(MonetaryAmount),
    /// A ceiling per day of stay (typical for daily ward and ICU charges).
    PerDay(MonetaryAmount),
    /// "As charged": the wording states no monetary ceiling on this benefit.
    ///
    /// # This is not infinity
    ///
    /// "As charged" means the insurer pays the amount charged — but the amount
    /// charged is still filtered through the plan's *other* terms (what is a
    /// claimable expense, reasonable-and-customary limits, the ward class you
    /// were entitled to). Modelling it as `Money::MAX` would flatten all of
    /// that away and produce a computation which says the insurer pays
    /// everything. It does not. It is its own variant so that any code reading
    /// a limit has to think about what it means.
    AsCharged,
}

impl ClaimLimit {
    /// The monetary ceiling, if the limit states one. `None` for
    /// [`ClaimLimit::AsCharged`].
    pub fn amount(&self) -> Option<&MonetaryAmount> {
        match self {
            Self::PerClaim(m) | Self::PerPolicyYear(m) | Self::Lifetime(m) | Self::PerDay(m) => {
                Some(m)
            }
            Self::AsCharged => None,
        }
    }
}

impl fmt::Display for ClaimLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerClaim(m) => write!(f, "{} per claim", show(m)),
            Self::PerPolicyYear(m) => write!(f, "{} per policy year", show(m)),
            Self::Lifetime(m) => write!(f, "{} lifetime", show(m)),
            Self::PerDay(m) => write!(f, "{} per day", show(m)),
            Self::AsCharged => f.write_str("as charged (no monetary limit stated)"),
        }
    }
}

/// A period after inception during which a benefit is not yet available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaitingPeriod {
    /// How long, in the unit the document used.
    pub duration: PolicyDuration,
}

impl fmt::Display for WaitingPeriod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} waiting period", self.duration)
    }
}

/// How the document treats conditions the insured already had.
///
/// # Why this is not a `bool`, and why the crate will not decide it for you
///
/// "Is my condition pre-existing?" is the question that decides whether a large
/// claim is paid, and it is a *clinical and legal* question, not an arithmetic
/// one. It turns on what the insured knew, what a reasonable person would have
/// sought treatment for, when symptoms first appeared, and how this particular
/// wording defines the term — and wordings define it differently.
///
/// This crate models what the document *says about* pre-existing conditions. It
/// never says whether a given condition is one. If you want that answer, read
/// the clause — which is why every variant carries you back to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreExistingConditionTreatment {
    /// The document states pre-existing conditions are excluded outright.
    StatedExcluded,
    /// The document states they are excluded for a period, after which they may
    /// be covered.
    StatedExcludedForPeriod(PolicyDuration),
    /// The document states they may be covered subject to underwriting,
    /// declaration, moratorium, or some other process it describes.
    StatedSubjectToAssessment {
        /// The process the document names, in the document's own words.
        process: String,
    },
    /// The document mentions pre-existing conditions but this crate could not
    /// determine what it does about them. Read the clause.
    NotDetermined,
}

impl fmt::Display for PreExistingConditionTreatment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StatedExcluded => f.write_str("stated as excluded"),
            Self::StatedExcludedForPeriod(d) => write!(f, "stated as excluded for {d}"),
            Self::StatedSubjectToAssessment { process } => {
                write!(f, "stated as subject to assessment: {process}")
            }
            Self::NotDetermined => f.write_str("mentioned, but not determinable — read the clause"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_insurance::Money;

    fn sgd(major: i64) -> MonetaryAmount {
        MonetaryAmount::new(Money::from_cents(major * 100), Currency::Iso("SGD".into()))
    }

    #[test]
    fn an_unscoped_term_applies_everywhere() {
        let ctx = TreatmentContext::new(WardClass::Private, ProviderNetwork::NonPanel);
        assert!(Scope::any().applies_to(&ctx));
    }

    /// The safety-critical behaviour: a term stated for one ward does not leak
    /// into another. Being "helpfully" approximate here understates what the
    /// patient pays.
    #[test]
    fn a_ward_scoped_term_does_not_leak_into_another_ward() {
        let class_a = TreatmentContext::new(WardClass::PublicA, ProviderNetwork::Panel);
        assert!(!Scope::ward(WardClass::PublicB1).applies_to(&class_a));
        assert!(Scope::ward(WardClass::PublicA).applies_to(&class_a));
    }

    #[test]
    fn panel_status_narrows_a_term_further() {
        let on_panel = TreatmentContext::new(WardClass::Private, ProviderNetwork::Panel);
        let off_panel = TreatmentContext::new(WardClass::Private, ProviderNetwork::NonPanel);

        let panel_only = Scope::ward_and_provider(WardClass::Private, ProviderNetwork::Panel);
        assert!(panel_only.applies_to(&on_panel));
        assert!(!panel_only.applies_to(&off_panel));
    }

    #[test]
    fn emergency_is_not_silently_treated_as_panel() {
        let emergency = TreatmentContext::new(WardClass::Private, ProviderNetwork::Emergency);
        let panel_only = Scope::ward_and_provider(WardClass::Private, ProviderNetwork::Panel);
        assert!(
            !panel_only.applies_to(&emergency),
            "only the wording may say an emergency counts as on-panel"
        );
    }

    #[test]
    fn as_charged_is_not_an_infinite_amount() {
        assert_eq!(ClaimLimit::AsCharged.amount(), None);
        assert_eq!(
            ClaimLimit::PerClaim(sgd(150_000)).amount(),
            Some(&sgd(150_000))
        );
    }

    /// A figure the document printed with a bare `$` must never be rendered
    /// back to a reader as though the document had named a currency.
    #[test]
    fn an_ambiguous_currency_is_shown_as_ambiguous() {
        let printed = MonetaryAmount::new(
            Money::from_cents(350_000),
            Currency::Ambiguous("$".into()),
        );
        assert!(show(&printed).contains("currency not stated"));
    }

    #[test]
    fn more_specific_scopes_outrank_general_ones() {
        assert!(
            Scope::ward_and_provider(WardClass::Private, ProviderNetwork::Panel).specificity()
                > Scope::ward(WardClass::Private).specificity()
        );
        assert!(Scope::ward(WardClass::Private).specificity() > Scope::any().specificity());
    }
}
