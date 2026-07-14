//! The central type: a published figure that cannot exist without its provenance.
//!
//! Everything else in this module is in service of [`Statistic`] and of the two
//! questions it forces you to answer before you may quote a number:
//!
//! 1. **What is this a figure about?** — the measure, the population, the
//!    stratum, the period, the lease profile.
//! 2. **How much do we actually know?** — how many observations back it, and
//!    which document said so.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::citation::Citation;
use super::period::Period;
use super::quantity::{LeaseRemaining, Quantity, QuantityKind, SgdAmount, Unit};
use super::stratum::{Dimension, Population, Stratum};

/// How many observations back a statistic.
///
/// Distinct from [`super::UnitCount`], which is a *measured value*. This is
/// *metadata about evidence*: the `n` behind a median. Keeping them separate is
/// what stops "the median across 3 transactions" from turning into "3" being
/// reported as a price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SampleCount(u32);

impl SampleCount {
    pub const fn new(count: u32) -> Self {
        Self(count)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SampleCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n={}", self.0)
    }
}

/// Below this many backing observations, a figure is flagged
/// [`Reliability::LowPrecision`].
///
/// # This is a KOPITIAM heuristic, not an HDB rule
///
/// Statistical agencies do apply suppression and reliability thresholds to small
/// cells, but this constant is **not** a reproduction of HDB's or SingStat's
/// published threshold — that was not available to verify offline, and inventing
/// an authoritative-looking number is precisely what this module exists to
/// prevent. It is a deliberately conservative default.
///
/// If HDB's actual threshold is later established from a citable source, replace
/// this constant and cite it here.
pub const SMALL_SAMPLE_THRESHOLD: u32 = 20;

/// Two lease profiles more than this many years apart are treated as different
/// products, not comparable slices.
///
/// Also a KOPITIAM heuristic and not an HDB rule. A decade of lease decay is a
/// material difference in both price and CPF treatment; the exact figure is a
/// judgment call and is exposed here so it can be argued with.
const LEASE_PROFILE_TOLERANCE_YEARS: u32 = 10;

/// How the flats behind a figure sit on the lease-decay curve.
///
/// A required field on every [`Statistic`], because "the median 4-room in Town A
/// is $X" is not a meaningful sentence without it. A town of 1970s flats and a
/// town of 2015 flats produce wildly different medians for reasons that have
/// nothing to do with the town.
///
/// Forcing the caller to state this — even to state that it is *unknown* — is the
/// point. An `Unstated` profile is an honest gap that travels with the number and
/// surfaces as a [`Caveat`]; a *missing* profile would be a silent one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseProfile {
    /// The stratum already pins the lease band, so the band *is* the profile.
    Banded,
    /// The publication stated the median remaining lease of the underlying flats.
    /// The strongest case, and the only one that supports a checked cross-town
    /// comparison.
    Median(LeaseRemaining),
    /// The publication gave no lease information. Not an error — most price
    /// tables do not — but comparisons drawn across it carry
    /// [`Caveat::LeaseProfileUnstated`].
    Unstated,
    /// The figure is not about flats at all (a household count, a waiting time),
    /// so lease decay does not apply.
    NotApplicable,
}

impl fmt::Display for LeaseProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeaseProfile::Banded => f.write_str("banded by stratum"),
            LeaseProfile::Median(lease) => write!(f, "median lease {lease}"),
            LeaseProfile::Unstated => f.write_str("lease profile not stated"),
            LeaseProfile::NotApplicable => f.write_str("n/a"),
        }
    }
}

/// Whether a figure rests on a sample, a complete enumeration, or an unstated
/// basis.
///
/// # A census of three transactions is still a shaky guide
///
/// It is tempting to treat `Census` as beyond reproach — and in one sense it is:
/// the median of every transaction in a cell has no *sampling error*, because
/// nothing was sampled. But that is not the question a buyer is asking. They want
/// to know what a flat in that town *costs*, and a cell containing three sales is
/// three specific units, on three specific floors, in three specific conditions.
/// The median is exact and nearly uninformative.
///
/// So [`Reliability`] keys off the **observation count regardless of basis**. The
/// distinction the buyer needs is not "sample vs census" but "how much did we
/// actually see", and this module reports that.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    /// A survey sample; `respondents` is the achieved sample size.
    Sample { respondents: SampleCount },
    /// A complete enumeration — every transaction or flat in the cell.
    /// `observations` is how many that was.
    Census { observations: SampleCount },
    /// The publication reported a figure without stating what backs it.
    ///
    /// A *known unknown*. It is representable because real publications do this,
    /// and forbidding it would tempt a caller to invent a plausible `n` to get
    /// past the type — which is worse than admitting ignorance. It surfaces as
    /// [`Caveat::BasisUnstated`] and can never be mistaken for a well-evidenced
    /// figure.
    Unstated,
}

impl Basis {
    /// How many observations back the figure, if the publication said.
    pub fn observations(&self) -> Option<SampleCount> {
        match self {
            Basis::Sample { respondents } => Some(*respondents),
            Basis::Census { observations } => Some(*observations),
            Basis::Unstated => None,
        }
    }
}

impl fmt::Display for Basis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Basis::Sample { respondents } => write!(f, "sample, {respondents}"),
            Basis::Census { observations } => write!(f, "census, {observations}"),
            Basis::Unstated => f.write_str("basis not stated"),
        }
    }
}

/// How much confidence a figure can bear.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reliability {
    /// Enough observations to be worth quoting plainly.
    Adequate { observations: SampleCount },
    /// Too few observations to bear the weight of a purchase decision. See
    /// [`SMALL_SAMPLE_THRESHOLD`] and the note on [`Basis`].
    LowPrecision {
        observations: SampleCount,
        threshold: u32,
    },
    /// The publication never said how many observations back this figure, so its
    /// reliability is genuinely unknown — which is *not* the same as adequate.
    Unknown,
}

impl Reliability {
    /// Whether this figure should be presented with an explicit warning.
    pub fn needs_warning(&self) -> bool {
        !matches!(self, Reliability::Adequate { .. })
    }
}

impl fmt::Display for Reliability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reliability::Adequate { observations } => write!(f, "adequate ({observations})"),
            Reliability::LowPrecision {
                observations,
                threshold,
            } => write!(
                f,
                "LOW PRECISION: based on only {} observation(s), below the threshold of {threshold}",
                observations.get()
            ),
            Reliability::Unknown => {
                f.write_str("UNKNOWN: the publication did not state how many observations back this figure")
            }
        }
    }
}

/// The survey instrument or statistical release a figure came out of.
///
/// Identity matters more than content here: two figures share a methodology iff
/// they came out of the same release *edition*. HDB redefines things between
/// editions — what counts as a "household", how a town boundary is drawn, whether
/// a price includes grants — and a series that silently spans such a change is a
/// fabricated trend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Methodology {
    id: String,
    notes: Option<String>,
}

impl Methodology {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            notes: None,
        }
    }

    /// Records what is known about how the figures were produced.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn notes(&self) -> Option<&str> {
        self.notes.as_deref()
    }
}

impl fmt::Display for Methodology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

/// What was measured.
///
/// # Why the definition is part of the identity
///
/// A `Measure` compares equal to another only when its **definition** matches
/// too, not just its name. This is deliberate and it is load-bearing.
///
/// HDB tables carry footnotes that *redefine the column*: "Median resale price¹"
/// where "¹ Prices are before grants" means something materially different from
/// the same header where the footnote says "after grants". Two columns with
/// identical headers and different footnotes are **different measures**, and a
/// series that joins them is a fiction. Putting the definition inside the
/// identity means the [`super::Series`] break detector catches it for free.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Measure {
    name: String,
    kind: QuantityKind,
    unit: Unit,
    /// The footnote or prose that qualifies what this column actually counts.
    /// Part of the measure's identity — see the type-level docs.
    definition: Option<String>,
}

impl Measure {
    pub fn new(name: impl Into<String>, unit: Unit) -> Self {
        Self {
            name: name.into(),
            kind: unit.kind(),
            unit,
            definition: None,
        }
    }

    /// Attaches the footnote that qualifies this measure. Changes the measure's
    /// identity — that is the point.
    pub fn with_definition(mut self, definition: impl Into<String>) -> Self {
        self.definition = Some(definition.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> QuantityKind {
        self.kind
    }

    pub fn unit(&self) -> Unit {
        self.unit
    }

    pub fn definition(&self) -> Option<&str> {
        self.definition.as_deref()
    }

    /// Whether two measures share a name but disagree on what they mean — the
    /// signature of a redefinition between editions.
    pub fn redefines(&self, other: &Measure) -> bool {
        self.name == other.name && self.definition != other.definition
    }
}

impl fmt::Display for Measure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.unit)
    }
}

/// A single published figure, and everything needed to know what it means.
///
/// # You cannot build one without provenance
///
/// The fields are private and there is exactly one constructor,
/// [`Statistic::new`], which takes every mandatory component. There is no
/// `Default`, no partial builder, and no public field to leave unset. An
/// unprovenanced statistic is not *expressible*.
///
/// The `Deserialize` impl is not a back door: every field is required, so a JSON
/// document missing (say) the citation fails to deserialize rather than
/// materialising a statistic with a hole in it.
///
/// # What each component is for
///
/// | Component | The question it answers |
/// |---|---|
/// | [`Measure`] | What was measured — *including* the footnote that redefines it |
/// | [`Population`] | Over what frame — transactions? flats? households? |
/// | [`Stratum`] | Which slice — which town, type, storey, lease band |
/// | [`Period`] | When it was observed (**not** when it was published) |
/// | [`Basis`] | How many observations back it |
/// | [`LeaseProfile`] | Where on the lease-decay curve these flats sit |
/// | [`Methodology`] | Which release edition produced it |
/// | [`Citation`] | Which document said so, and where in it |
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Statistic {
    measure: Measure,
    quantity: Quantity,
    population: Population,
    stratum: Stratum,
    period: Period,
    basis: Basis,
    lease_profile: LeaseProfile,
    methodology: Methodology,
    citation: Citation,
}

/// A refusal to construct a statistic whose parts do not agree.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StatisticError {
    /// The value is not the kind of thing the measure says it is — e.g. a money
    /// value under a measure declared as a percentage. Almost always a
    /// mis-specified column, and wrong by orders of magnitude if let through.
    #[error(
        "measure `{measure}` is declared as {expected} but the value is {actual}; \
         this column is mis-specified"
    )]
    KindMismatch {
        measure: String,
        expected: QuantityKind,
        actual: QuantityKind,
    },
}

impl Statistic {
    /// Builds a statistic. Every argument is mandatory; that constraint *is* the
    /// design.
    ///
    /// Fails if the value's kind contradicts the measure's declared kind — the
    /// one internal consistency check that can be made without knowing the
    /// domain.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        measure: Measure,
        quantity: Quantity,
        population: Population,
        stratum: Stratum,
        period: Period,
        basis: Basis,
        lease_profile: LeaseProfile,
        methodology: Methodology,
        citation: Citation,
    ) -> Result<Self, StatisticError> {
        if measure.kind() != quantity.kind() {
            return Err(StatisticError::KindMismatch {
                measure: measure.name().to_string(),
                expected: measure.kind(),
                actual: quantity.kind(),
            });
        }
        Ok(Self {
            measure,
            quantity,
            population,
            stratum,
            period,
            basis,
            lease_profile,
            methodology,
            citation,
        })
    }

    pub fn measure(&self) -> &Measure {
        &self.measure
    }

    /// The value. For the CPF/policy seam, [`Quantity::as_money`] and
    /// [`Quantity::as_lease`] get you straight to the fields those layers need.
    pub fn quantity(&self) -> Quantity {
        self.quantity
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn stratum(&self) -> &Stratum {
        &self.stratum
    }

    /// When the figure was **observed**. Not when it was published — for that,
    /// see [`Citation::published`].
    pub fn period(&self) -> Period {
        self.period
    }

    pub fn basis(&self) -> &Basis {
        &self.basis
    }

    pub fn lease_profile(&self) -> &LeaseProfile {
        &self.lease_profile
    }

    pub fn methodology(&self) -> &Methodology {
        &self.methodology
    }

    /// The document this figure came from. Always present.
    pub fn citation(&self) -> &Citation {
        &self.citation
    }

    /// How much confidence this figure can bear, derived from its backing
    /// observation count.
    pub fn reliability(&self) -> Reliability {
        match self.basis.observations() {
            None => Reliability::Unknown,
            Some(observations) if observations.get() < SMALL_SAMPLE_THRESHOLD => {
                Reliability::LowPrecision {
                    observations,
                    threshold: SMALL_SAMPLE_THRESHOLD,
                }
            }
            Some(observations) => Reliability::Adequate { observations },
        }
    }

    /// Compares this figure against another, or explains why it cannot be done.
    ///
    /// The rules, in the order they are applied:
    ///
    /// 1. **Different measures** cannot be compared — including two measures that
    ///    share a name but carry different footnotes.
    /// 2. **Different populations** cannot be compared. Flats that *sold* are not
    ///    flats that *exist*.
    /// 3. **Different methodologies** cannot be compared. The definitions moved.
    /// 4. **More than one varying dimension** is a confounded comparison, and is
    ///    refused. See the module docs on [`super::stratum`].
    /// 5. **Divergent lease profiles**, where lease is not the thing being
    ///    varied, are refused: the flats are different products.
    /// 6. **Different index bases** are refused; the readings are on different
    ///    scales.
    ///
    /// Anything that survives all six is returned as a [`Comparison`] carrying
    /// whatever [`Caveat`]s still apply. A comparison with caveats is still a
    /// comparison — but the caveats must be shown to the user, not dropped.
    pub fn compare_with(&self, other: &Statistic) -> Result<Comparison, Incomparability> {
        if self.measure != other.measure {
            return Err(if self.measure.redefines(&other.measure) {
                Incomparability::MeasureRedefined {
                    measure: self.measure.name().to_string(),
                    left: self.measure.definition().map(str::to_string),
                    right: other.measure.definition().map(str::to_string),
                }
            } else {
                Incomparability::DifferentMeasure {
                    left: self.measure.to_string(),
                    right: other.measure.to_string(),
                }
            });
        }

        if self.population != other.population {
            return Err(Incomparability::DifferentPopulation {
                left: self.population.to_string(),
                right: other.population.to_string(),
            });
        }

        if self.methodology != other.methodology {
            return Err(Incomparability::MethodologyMismatch {
                left: self.methodology.id().to_string(),
                right: other.methodology.id().to_string(),
            });
        }

        let varying = self.stratum.varying_dimensions(&other.stratum);
        if varying.len() > 1 {
            return Err(Incomparability::ConfoundedStrata {
                left: self.stratum.to_string(),
                right: other.stratum.to_string(),
                varying,
            });
        }

        let varying_dimension = varying.first().cloned();

        // Index readings on different bases are different scales entirely.
        if let (Quantity::Index(left), Quantity::Index(right)) = (self.quantity, other.quantity)
            && left.base() != right.base()
        {
            return Err(Incomparability::DifferentIndexBase {
                left: left.base(),
                right: right.base(),
            });
        }

        let mut caveats = Vec::new();

        // Lease profile. If lease is the dimension being varied, divergence is
        // the *point* of the comparison ("what does 20 more years of lease cost
        // me?") and must not be treated as an error.
        let varying_lease = varying_dimension
            .as_ref()
            .is_some_and(|d| *d == Dimension::LeaseBand);
        if !varying_lease {
            match (&self.lease_profile, &other.lease_profile) {
                (LeaseProfile::Median(left), LeaseProfile::Median(right)) => {
                    let gap_years = left.whole_years().abs_diff(right.whole_years());
                    if gap_years > LEASE_PROFILE_TOLERANCE_YEARS {
                        return Err(Incomparability::LeaseProfileMismatch {
                            left: *left,
                            right: *right,
                            tolerance_years: LEASE_PROFILE_TOLERANCE_YEARS,
                        });
                    }
                }
                (LeaseProfile::Unstated, _) | (_, LeaseProfile::Unstated) => {
                    caveats.push(Caveat::LeaseProfileUnstated);
                }
                _ => {}
            }
        }

        if self.period != other.period {
            caveats.push(Caveat::DifferentPeriods {
                left: self.period,
                right: other.period,
            });
        }

        // A part-whole comparison — a town against the national figure — is
        // legitimate but not an independent contrast, because one contains the
        // other. Identical strata are excluded: a slice trivially contains itself.
        let nested = self.stratum.contains(&other.stratum) || other.stratum.contains(&self.stratum);
        if nested && self.stratum != other.stratum {
            caveats.push(Caveat::PartWhole);
        }

        for statistic in [self, other] {
            match statistic.reliability() {
                Reliability::LowPrecision { observations, .. } => {
                    caveats.push(Caveat::SmallSample {
                        stratum: statistic.stratum.to_string(),
                        observations,
                    });
                }
                Reliability::Unknown => caveats.push(Caveat::BasisUnstated {
                    stratum: statistic.stratum.to_string(),
                }),
                Reliability::Adequate { .. } => {}
            }
        }

        let difference = self.difference_from(other);

        Ok(Comparison {
            left: self.clone(),
            right: other.clone(),
            varying_dimension,
            difference,
            caveats,
        })
    }

    /// The difference between two values of the same kind, where subtraction is
    /// meaningful.
    ///
    /// Returns `None` for kinds where a difference is not a sensible object.
    /// Note that even where it *is* returned, the difference of two medians is
    /// the difference of the medians — it is emphatically **not** the median of
    /// the differences, and no amount of arithmetic here makes it so.
    fn difference_from(&self, other: &Statistic) -> Option<Quantity> {
        match (self.quantity, other.quantity) {
            (Quantity::Money(left), Quantity::Money(right)) => {
                Some(Quantity::Money(left.difference(right)))
            }
            _ => None,
        }
    }
}

impl fmt::Display for Statistic {
    /// Renders the figure *with* its caveats. There is deliberately no way to
    /// `Display` the bare number: if it is going in front of a human, the
    /// reliability goes with it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} = {} [{}, {}, {}, {}]",
            self.measure, self.quantity, self.stratum, self.period, self.basis, self.citation
        )?;
        if self.reliability().needs_warning() {
            write!(f, " !! {}", self.reliability())?;
        }
        Ok(())
    }
}

/// Something true about a comparison that the reader must be told, but which does
/// not invalidate it.
///
/// Caveats are not warnings to be swallowed. A comparison carrying
/// [`Caveat::SmallSample`] is a comparison between numbers one of which barely
/// exists, and any layer presenting it — an affordability calculator above this
/// module, say — must carry the caveat through to the person making the decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Caveat {
    /// One side rests on very few observations. See [`SMALL_SAMPLE_THRESHOLD`].
    SmallSample {
        stratum: String,
        observations: SampleCount,
    },
    /// One side never said what backs it.
    BasisUnstated { stratum: String },
    /// The two figures describe different stretches of time. The market moved in
    /// between, and some of the difference is that movement rather than the
    /// dimension under study.
    DifferentPeriods { left: Period, right: Period },
    /// One stratum contains the other — a town against the national figure. Not
    /// an independent contrast, because the town is *inside* the aggregate it is
    /// being measured against.
    PartWhole,
    /// Neither side stated its lease profile, so some of the difference may be
    /// lease decay rather than the dimension under study.
    LeaseProfileUnstated,
}

impl fmt::Display for Caveat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Caveat::SmallSample {
                stratum,
                observations,
            } => write!(
                f,
                "`{stratum}` rests on only {} observation(s) — too few to bear a purchase decision",
                observations.get()
            ),
            Caveat::BasisUnstated { stratum } => write!(
                f,
                "`{stratum}` does not state how many observations back it"
            ),
            Caveat::DifferentPeriods { left, right } => write!(
                f,
                "these figures cover different periods ({left} vs {right}); part of the \
                 difference is the market moving, not the dimension under study"
            ),
            Caveat::PartWhole => f.write_str(
                "one of these strata contains the other, so this is a part-whole comparison \
                 and not an independent contrast",
            ),
            Caveat::LeaseProfileUnstated => f.write_str(
                "lease profile is not stated, so some of this difference may be lease decay \
                 rather than the dimension under study",
            ),
        }
    }
}

/// Why two figures cannot honestly be compared.
///
/// This is the typed refusal that stands where a plausible-looking number would
/// otherwise be.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Incomparability {
    #[error("`{left}` and `{right}` measure different things")]
    DifferentMeasure { left: String, right: String },

    #[error(
        "`{measure}` was redefined between these figures (left: {left:?}, right: {right:?}); \
         the header is the same but the footnote is not, so these are different measures"
    )]
    MeasureRedefined {
        measure: String,
        left: Option<String>,
        right: Option<String>,
    },

    #[error(
        "these figures are drawn from different populations (`{left}` vs `{right}`); \
         flats that sold are not flats that exist"
    )]
    DifferentPopulation { left: String, right: String },

    #[error(
        "these figures come from different methodologies (`{left}` vs `{right}`); \
         the definitions moved between editions"
    )]
    MethodologyMismatch { left: String, right: String },

    #[error(
        "`{left}` and `{right}` differ on {} dimensions ({}); the difference between them \
         confounds all of these effects and cannot be attributed to any one of them",
        varying.len(),
        varying.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    )]
    ConfoundedStrata {
        left: String,
        right: String,
        varying: Vec<Dimension>,
    },

    #[error(
        "these flats sit {} years apart on the lease-decay curve ({left} vs {right}), \
         beyond the {tolerance_years}-year tolerance; they are different products, not \
         comparable slices",
        left.whole_years().abs_diff(right.whole_years())
    )]
    LeaseProfileMismatch {
        left: LeaseRemaining,
        right: LeaseRemaining,
        tolerance_years: u32,
    },

    #[error(
        "these index readings are on different bases ({left} = 100 vs {right} = 100); \
         they are different scales and their difference is not a price movement"
    )]
    DifferentIndexBase { left: Period, right: Period },
}

/// A comparison that survived every check — together with the caveats that still
/// apply to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    left: Statistic,
    right: Statistic,
    varying_dimension: Option<Dimension>,
    difference: Option<Quantity>,
    caveats: Vec<Caveat>,
}

impl Comparison {
    pub fn left(&self) -> &Statistic {
        &self.left
    }

    pub fn right(&self) -> &Statistic {
        &self.right
    }

    /// The single dimension under study, or `None` if the two strata are
    /// identical (a pure period-over-period comparison).
    pub fn varying_dimension(&self) -> Option<&Dimension> {
        self.varying_dimension.as_ref()
    }

    /// `left - right`, where that is a meaningful object.
    pub fn difference(&self) -> Option<Quantity> {
        self.difference
    }

    /// The difference as money, for the common case of comparing prices.
    pub fn price_difference(&self) -> Option<SgdAmount> {
        self.difference.and_then(Quantity::as_money)
    }

    /// Everything the reader must be told. **Do not drop these.**
    pub fn caveats(&self) -> &[Caveat] {
        &self.caveats
    }

    /// Whether this comparison can be quoted without qualification.
    pub fn is_clean(&self) -> bool {
        self.caveats.is_empty()
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.varying_dimension {
            Some(dimension) => write!(f, "comparison by {dimension}: ")?,
            None => write!(f, "comparison: ")?,
        }
        write!(f, "{} vs {}", self.left.quantity(), self.right.quantity())?;
        if let Some(difference) = self.difference {
            write!(f, " (difference {difference})")?;
        }
        for caveat in &self.caveats {
            write!(f, "\n  ! {caveat}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::survey::citation::Locator;
    use crate::hdb::survey::quantity::IndexPoint;
    use crate::hdb::survey::period::Quarter;

    /// A citation that announces, in its own title, that it is not real data.
    /// Every fixture in this module uses one. See the provenance note in the
    /// module docs of [`super::super`].
    fn synthetic_citation() -> Citation {
        Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("SYNTHETIC-1".into()),
            Period::Year(2024),
        )
    }

    fn methodology() -> Methodology {
        Methodology::new("SYNTHETIC METHODOLOGY A")
    }

    fn median_price() -> Measure {
        Measure::new("Median resale price", Unit::Sgd)
    }

    /// Deliberately implausible repdigit prices, so no fixture can be mistaken
    /// for a real Singapore statistic.
    fn price_statistic(
        town: &str,
        flat_type: &str,
        dollars: i64,
        observations: u32,
    ) -> Statistic {
        Statistic::new(
            median_price(),
            Quantity::Money(SgdAmount::from_dollars(dollars)),
            Population::ResaleTransactions,
            Stratum::all()
                .with(Dimension::Town, town)
                .with(Dimension::FlatType, flat_type),
            Period::Year(2024),
            Basis::Census {
                observations: SampleCount::new(observations),
            },
            LeaseProfile::Unstated,
            methodology(),
            synthetic_citation(),
        )
        .expect("fixture is internally consistent")
    }

    #[test]
    fn a_statistic_cannot_be_built_without_its_provenance() {
        // This test is a *compile-time* assertion dressed as a runtime one. The
        // only constructor is `Statistic::new`, and it takes all nine components.
        // There is no Default, no partial builder, and every field is private —
        // so the following line is the *only* way to obtain a Statistic, and it
        // cannot be written without a population, a period, a basis and a
        // citation. If someone adds a laxer constructor, this comment is the
        // thing they have to argue with.
        let statistic = price_statistic("TAMPINES", "4-ROOM", 111_111, 50);
        assert_eq!(statistic.citation().publication(), "SYNTHETIC FIXTURE — NOT HDB DATA");
        assert_eq!(statistic.period(), Period::Year(2024));
        assert_eq!(statistic.basis().observations().unwrap().get(), 50);
        assert_eq!(*statistic.population(), Population::ResaleTransactions);
    }

    #[test]
    fn a_value_that_contradicts_its_measure_is_refused() {
        // A percentage measure holding a money value: a mis-specified column,
        // wrong by orders of magnitude if let through.
        let err = Statistic::new(
            Measure::new("Share of transactions", Unit::Percent),
            Quantity::Money(SgdAmount::from_dollars(1111)),
            Population::ResaleTransactions,
            Stratum::all(),
            Period::Year(2024),
            Basis::Unstated,
            LeaseProfile::NotApplicable,
            methodology(),
            synthetic_citation(),
        )
        .unwrap_err();
        assert!(matches!(err, StatisticError::KindMismatch { .. }));
    }

    #[test]
    fn a_controlled_comparison_varying_one_dimension_succeeds() {
        let a = price_statistic("TAMPINES", "4-ROOM", 111_111, 50);
        let b = price_statistic("QUEENSTOWN", "4-ROOM", 222_222, 40);
        let comparison = a.compare_with(&b).unwrap();
        assert_eq!(comparison.varying_dimension(), Some(&Dimension::Town));
        assert_eq!(
            comparison.price_difference().unwrap(),
            SgdAmount::from_dollars(-111_111)
        );
    }

    #[test]
    fn confounded_strata_are_a_typed_error_not_a_number() {
        // THE central safety property. A 4-room in one town against a 5-room in
        // another: the difference is a town effect AND a flat-type effect, and
        // reporting a single number would invite exactly the wrong conclusion.
        let a = price_statistic("TAMPINES", "4-ROOM", 111_111, 50);
        let b = price_statistic("QUEENSTOWN", "5-ROOM", 222_222, 50);
        let err = a.compare_with(&b).unwrap_err();
        match err {
            Incomparability::ConfoundedStrata { varying, .. } => {
                assert_eq!(varying.len(), 2);
                assert!(varying.contains(&Dimension::Town));
                assert!(varying.contains(&Dimension::FlatType));
            }
            other => panic!("expected ConfoundedStrata, got {other:?}"),
        }
        // And critically: no number came back.
        assert!(a.compare_with(&b).is_err());
    }

    #[test]
    fn different_populations_cannot_be_compared() {
        let sold = price_statistic("TAMPINES", "4-ROOM", 111_111, 50);
        let stock = Statistic::new(
            median_price(),
            Quantity::Money(SgdAmount::from_dollars(222_222)),
            Population::FlatStock,
            Stratum::all()
                .with(Dimension::Town, "TAMPINES")
                .with(Dimension::FlatType, "4-ROOM"),
            Period::Year(2024),
            Basis::Census {
                observations: SampleCount::new(50),
            },
            LeaseProfile::Unstated,
            methodology(),
            synthetic_citation(),
        )
        .unwrap();
        assert!(matches!(
            sold.compare_with(&stock).unwrap_err(),
            Incomparability::DifferentPopulation { .. }
        ));
    }

    #[test]
    fn a_footnote_redefinition_makes_two_columns_different_measures() {
        // Same header, different footnote: "before grants" vs "after grants".
        // These are not the same measure and comparing them is meaningless.
        let before = Statistic::new(
            median_price().with_definition("Prices are before grants"),
            Quantity::Money(SgdAmount::from_dollars(111_111)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, "TAMPINES"),
            Period::Year(2024),
            Basis::Census {
                observations: SampleCount::new(50),
            },
            LeaseProfile::Unstated,
            methodology(),
            synthetic_citation(),
        )
        .unwrap();
        let after = Statistic::new(
            median_price().with_definition("Prices are after grants"),
            Quantity::Money(SgdAmount::from_dollars(222_222)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, "TAMPINES"),
            Period::Year(2024),
            Basis::Census {
                observations: SampleCount::new(50),
            },
            LeaseProfile::Unstated,
            methodology(),
            synthetic_citation(),
        )
        .unwrap();

        assert!(matches!(
            before.compare_with(&after).unwrap_err(),
            Incomparability::MeasureRedefined { .. }
        ));
    }

    #[test]
    fn a_tiny_sample_is_flagged_and_never_presented_with_false_confidence() {
        // Someone is about to make the largest purchase of their life. A median
        // of three transactions must SHOUT.
        let thin = price_statistic("QUIET TOWN", "5-ROOM", 111_111, 3);
        match thin.reliability() {
            Reliability::LowPrecision {
                observations,
                threshold,
            } => {
                assert_eq!(observations.get(), 3);
                assert_eq!(threshold, SMALL_SAMPLE_THRESHOLD);
            }
            other => panic!("3 transactions must be LowPrecision, got {other:?}"),
        }
        assert!(thin.reliability().needs_warning());
        // The warning is in the Display output — there is no way to render the
        // bare number to a human without it.
        assert!(thin.to_string().contains("LOW PRECISION"));

        // And a comparison drawn against it carries the caveat through.
        let solid = price_statistic("BUSY TOWN", "5-ROOM", 222_222, 500);
        let comparison = thin.compare_with(&solid).unwrap();
        assert!(!comparison.is_clean());
        assert!(comparison
            .caveats()
            .iter()
            .any(|c| matches!(c, Caveat::SmallSample { .. })));
    }

    #[test]
    fn an_unstated_basis_is_unknown_reliability_not_adequate() {
        let statistic = Statistic::new(
            median_price(),
            Quantity::Money(SgdAmount::from_dollars(111_111)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, "TAMPINES"),
            Period::Year(2024),
            Basis::Unstated,
            LeaseProfile::Unstated,
            methodology(),
            synthetic_citation(),
        )
        .unwrap();
        assert!(matches!(statistic.reliability(), Reliability::Unknown));
        assert!(statistic.reliability().needs_warning());
    }

    #[test]
    fn a_census_of_three_is_still_low_precision() {
        // The subtle one. A census has no sampling error — but a cell holding
        // three sales is three specific units, and its median tells a buyer
        // almost nothing about what a flat there costs. Reliability keys off the
        // observation count, NOT the basis.
        let census_of_three = price_statistic("QUIET TOWN", "EXECUTIVE", 111_111, 3);
        assert!(matches!(
            census_of_three.basis(),
            Basis::Census { .. }
        ));
        assert!(matches!(
            census_of_three.reliability(),
            Reliability::LowPrecision { .. }
        ));
    }

    #[test]
    fn divergent_lease_profiles_are_refused_when_lease_is_not_the_variable() {
        // A town of 1970s flats against a town of 2015 flats. The price gap is
        // mostly lease decay, and attributing it to the town would be wrong.
        let make = |town: &str, lease_years: u32| {
            Statistic::new(
                median_price(),
                Quantity::Money(SgdAmount::from_dollars(111_111)),
                Population::ResaleTransactions,
                Stratum::all().with(Dimension::Town, town),
                Period::Year(2024),
                Basis::Census {
                    observations: SampleCount::new(100),
                },
                LeaseProfile::Median(LeaseRemaining::from_years_months(lease_years, 0)),
                methodology(),
                synthetic_citation(),
            )
            .unwrap()
        };
        let old_estate = make("OLD TOWN", 55);
        let new_estate = make("NEW TOWN", 90);

        assert!(matches!(
            old_estate.compare_with(&new_estate).unwrap_err(),
            Incomparability::LeaseProfileMismatch { .. }
        ));

        // But two towns with similar lease profiles compare fine.
        let similar = make("OTHER TOWN", 58);
        assert!(old_estate.compare_with(&similar).is_ok());
    }

    #[test]
    fn varying_lease_deliberately_is_allowed_because_that_is_the_question() {
        // "What does twenty more years of lease cost me?" is a legitimate and
        // important question, and the lease-profile guard must not block it when
        // lease band is the dimension under study.
        let make = |band: &str, lease_years: u32, dollars: i64| {
            Statistic::new(
                median_price(),
                Quantity::Money(SgdAmount::from_dollars(dollars)),
                Population::ResaleTransactions,
                Stratum::all()
                    .with(Dimension::Town, "TAMPINES")
                    .with(Dimension::LeaseBand, band),
                Period::Year(2024),
                Basis::Census {
                    observations: SampleCount::new(100),
                },
                LeaseProfile::Median(LeaseRemaining::from_years_months(lease_years, 0)),
                methodology(),
                synthetic_citation(),
            )
            .unwrap()
        };
        let short = make("50-59 YEARS", 55, 111_111);
        let long = make("90-99 YEARS", 95, 222_222);
        let comparison = short.compare_with(&long).unwrap();
        assert_eq!(
            comparison.varying_dimension(),
            Some(&Dimension::LeaseBand)
        );
    }

    #[test]
    fn index_readings_on_different_bases_are_refused() {
        let make = |base_year: u16, points: i64| {
            let base = Period::Quarter {
                year: base_year,
                quarter: Quarter::Q1,
            };
            Statistic::new(
                Measure::new("Resale Price Index", Unit::IndexPoints),
                Quantity::Index(IndexPoint::from_thousandths(points, base)),
                Population::ResaleTransactions,
                Stratum::all(),
                Period::Year(2024),
                Basis::Unstated,
                LeaseProfile::NotApplicable,
                methodology(),
                synthetic_citation(),
            )
            .unwrap()
        };
        let old_base = make(2009, 100_000);
        let new_base = make(2014, 111_000);
        assert!(matches!(
            old_base.compare_with(&new_base).unwrap_err(),
            Incomparability::DifferentIndexBase { .. }
        ));
    }

    #[test]
    fn a_town_against_the_national_figure_is_flagged_part_whole() {
        let national = Statistic::new(
            median_price(),
            Quantity::Money(SgdAmount::from_dollars(111_111)),
            Population::ResaleTransactions,
            Stratum::all(),
            Period::Year(2024),
            Basis::Census {
                observations: SampleCount::new(9999),
            },
            LeaseProfile::NotApplicable,
            methodology(),
            synthetic_citation(),
        )
        .unwrap();
        let town = Statistic::new(
            median_price(),
            Quantity::Money(SgdAmount::from_dollars(222_222)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, "TAMPINES"),
            Period::Year(2024),
            Basis::Census {
                observations: SampleCount::new(300),
            },
            LeaseProfile::NotApplicable,
            methodology(),
            synthetic_citation(),
        )
        .unwrap();

        let comparison = national.compare_with(&town).unwrap();
        // Legitimate to ask, but Tampines is INSIDE the national aggregate, so
        // this is not an independent contrast and the user must be told.
        assert!(comparison.caveats().contains(&Caveat::PartWhole));
    }

    #[test]
    fn comparing_across_periods_warns_that_the_market_moved() {
        let make = |period: Period| {
            Statistic::new(
                median_price(),
                Quantity::Money(SgdAmount::from_dollars(111_111)),
                Population::ResaleTransactions,
                Stratum::all().with(Dimension::Town, "TAMPINES"),
                period,
                Basis::Census {
                    observations: SampleCount::new(100),
                },
                LeaseProfile::NotApplicable,
                methodology(),
                synthetic_citation(),
            )
            .unwrap()
        };
        let comparison = make(Period::Year(2023))
            .compare_with(&make(Period::Year(2024)))
            .unwrap();
        assert!(comparison
            .caveats()
            .iter()
            .any(|c| matches!(c, Caveat::DifferentPeriods { .. })));
    }
}
