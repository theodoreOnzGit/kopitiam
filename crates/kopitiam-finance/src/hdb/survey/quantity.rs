//! The values themselves — and why they are five different types.
//!
//! A resale price, a percentage, a Resale Price Index reading, a remaining lease
//! and a floor area are all "numbers", and modelling them all as `f64` would let
//! you add a lease to a price and get a plausible-looking answer. They are
//! modelled as distinct types precisely so that you cannot.
//!
//! Every one is a fixed-point integer. See [`super::fixed`] for why there is no
//! floating point anywhere in this module's value model.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::fixed::{FixedParseError, format_fixed, parse_fixed};
use super::period::Period;

/// Money, in **cents**.
///
/// Money is never floating point. `0.1 + 0.2 != 0.3` in binary floating point,
/// and a platform that claims provenance cannot have prices that drift depending
/// on how they were summed. Cents are exact, orderable, and hashable.
///
/// Singapore dollars. HDB publishes no other currency, so the currency is a
/// property of the type rather than a field — if that ever stops being true, this
/// is the type to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SgdAmount(i64);

impl SgdAmount {
    /// Money scale: two decimal places (cents).
    const SCALE: u32 = 2;

    /// An amount in whole dollars.
    pub const fn from_dollars(dollars: i64) -> Self {
        Self(dollars * 100)
    }

    /// An amount in cents.
    pub const fn from_cents(cents: i64) -> Self {
        Self(cents)
    }

    /// Parses a published money cell: `"540,000"`, `"$540,000"`, `"1,500.50"`.
    pub fn parse(input: &str) -> Result<Self, FixedParseError> {
        parse_fixed(input, Self::SCALE).map(Self)
    }

    /// The exact value in cents.
    pub const fn cents(self) -> i64 {
        self.0
    }

    /// The difference between two amounts.
    ///
    /// Note what this is *not*: subtracting two medians gives the difference of
    /// the medians, which is **not** the median of the differences. That
    /// distinction is why arithmetic on statistics lives behind
    /// [`super::Comparison`] rather than being handed out freely.
    pub fn difference(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl fmt::Display for SgdAmount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S${}", format_fixed(self.0, Self::SCALE))
    }
}

/// A percentage, in **hundredths of a percent** (basis points).
///
/// `87.3%` is stored as `8730`. Exact, so two published percentages that were
/// printed identically compare equal — which floating point cannot promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Percentage(i32);

impl Percentage {
    /// Percentage scale: two decimal places on the percent value.
    const SCALE: u32 = 2;

    /// A percentage from hundredths of a percent: `8730` is `87.30%`.
    pub const fn from_basis_points(points: i32) -> Self {
        Self(points)
    }

    /// Parses a published percentage cell: `"87.3"`, `"87.3%"`.
    ///
    /// Values outside `0..=100` are *permitted*: a year-on-year price change can
    /// legitimately exceed 100% or be negative. A percentage is not always a
    /// proportion, and clamping here would corrupt exactly the figures a buyer
    /// cares most about.
    pub fn parse(input: &str) -> Result<Self, FixedParseError> {
        let value = parse_fixed(input, Self::SCALE)?;
        i32::try_from(value)
            .map(Self)
            .map_err(|_| FixedParseError::OutOfRange {
                input: input.to_string(),
            })
    }

    /// The exact value in hundredths of a percent.
    pub const fn basis_points(self) -> i32 {
        self.0
    }
}

impl fmt::Display for Percentage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}%", format_fixed(i64::from(self.0), Self::SCALE))
    }
}

/// A Resale Price Index reading, in **thousandths of an index point**, carrying
/// its **base period**.
///
/// # Why the base period is inside the value
///
/// An index reading of `195.5` means nothing on its own. It means "95.5% above
/// the level of the base period" — and HDB has rebased the RPI more than once.
/// Two readings on different bases are simply different scales, and subtracting
/// one from the other produces a number that looks like a price movement and is
/// not one.
///
/// Carrying the base *inside the value* makes that mistake unrepresentable:
/// [`IndexPoint::change_from`] refuses to compute a change across a rebasing, and
/// [`super::Statistic::compare_with`] refuses the comparison outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexPoint {
    thousandths: i64,
    base: Period,
}

/// Refusal to compute an index change across a rebasing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "cannot compare index readings on different bases ({left} = 100 vs {right} = 100); \
     HDB has rebased this index, so the two readings are on different scales and their \
     difference is not a price movement"
)]
pub struct RebasedIndex {
    pub left: Period,
    pub right: Period,
}

impl IndexPoint {
    /// Index scale: three decimal places.
    const SCALE: u32 = 3;

    /// An index reading against a stated base period.
    pub const fn from_thousandths(thousandths: i64, base: Period) -> Self {
        Self { thousandths, base }
    }

    /// Parses a published index cell against a stated base period.
    ///
    /// The base is a required argument and not an `Option`: an index reading
    /// whose base you do not know is not usable, and defaulting it would invent
    /// the very fact that matters.
    pub fn parse(input: &str, base: Period) -> Result<Self, FixedParseError> {
        parse_fixed(input, Self::SCALE).map(|thousandths| Self { thousandths, base })
    }

    /// The exact reading in thousandths of an index point.
    pub const fn thousandths(self) -> i64 {
        self.thousandths
    }

    /// The period this index is based at (the period that equals 100).
    pub const fn base(self) -> Period {
        self.base
    }

    /// The percentage change from `earlier` to `self`, or a refusal if the two
    /// readings sit on different bases.
    pub fn change_from(self, earlier: Self) -> Result<Percentage, RebasedIndex> {
        if self.base != earlier.base {
            return Err(RebasedIndex {
                left: earlier.base,
                right: self.base,
            });
        }
        if earlier.thousandths == 0 {
            return Ok(Percentage::from_basis_points(0));
        }
        // (new - old) / old, expressed in hundredths of a percent. Integer
        // arithmetic throughout: 10_000 basis points to the unit.
        let delta = self.thousandths - earlier.thousandths;
        let basis_points = delta.saturating_mul(10_000) / earlier.thousandths;
        Ok(Percentage::from_basis_points(basis_points as i32))
    }
}

impl fmt::Display for IndexPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({} = 100)",
            format_fixed(self.thousandths, Self::SCALE),
            self.base
        )
    }
}

/// Remaining lease, in **whole months**.
///
/// # Why this is a first-class quantity and not a stratum label
///
/// Singapore HDB flats are 99-year leaseholds, and the remaining lease is not a
/// footnote on a flat — it is a principal determinant of both its price and what
/// the buyer may do with it. A lease that does not cover the youngest buyer to
/// age 95 restricts how much CPF may be used towards the purchase, which changes
/// the cash a buyer must find. That rule belongs to [`crate::cpf`], not here —
/// but the *input* to it is this value, so it is exposed as a typed quantity that
/// the CPF layer can read directly rather than something it must parse back out
/// of a label.
///
/// Two flats identical in town, type and floor area but forty years apart in
/// remaining lease are not the same product, and this module will not compare
/// them as though they were. See [`super::Incomparability::LeaseProfileMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LeaseRemaining(u32);

impl LeaseRemaining {
    /// A remaining lease in whole months.
    pub const fn from_months(months: u32) -> Self {
        Self(months)
    }

    /// A remaining lease in years and months, as HDB states it
    /// (`"64 years 03 months"`).
    pub const fn from_years_months(years: u32, months: u32) -> Self {
        Self(years * 12 + months)
    }

    /// The exact remaining lease in months.
    pub const fn months(self) -> u32 {
        self.0
    }

    /// Whole years remaining, truncated — the granularity HDB usually bands by.
    pub const fn whole_years(self) -> u32 {
        self.0 / 12
    }
}

impl fmt::Display for LeaseRemaining {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}y {}m", self.0 / 12, self.0 % 12)
    }
}

/// Floor area, in **tenths of a square metre**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FloorArea(u32);

impl FloorArea {
    const SCALE: u32 = 1;

    /// An area in whole square metres.
    pub const fn from_square_metres(sqm: u32) -> Self {
        Self(sqm * 10)
    }

    /// Parses a published area cell.
    pub fn parse(input: &str) -> Result<Self, FixedParseError> {
        let value = parse_fixed(input, Self::SCALE)?;
        u32::try_from(value)
            .map(Self)
            .map_err(|_| FixedParseError::OutOfRange {
                input: input.to_string(),
            })
    }

    /// The exact area in tenths of a square metre.
    pub const fn tenths_of_sqm(self) -> u32 {
        self.0
    }
}

impl fmt::Display for FloorArea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} sqm", format_fixed(i64::from(self.0), Self::SCALE))
    }
}

/// A count of things — transactions, flats, households.
///
/// Distinct from [`super::SampleCount`], which counts the observations *backing*
/// a statistic. `UnitCount` is a measured value in its own right ("2,222 flats
/// were launched"); `SampleCount` is metadata about how much evidence sits behind
/// some *other* number. Conflating them is how "the median was $X across 3
/// transactions" turns into "3" being reported as a price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct UnitCount(u64);

impl UnitCount {
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    /// Parses a published count cell.
    pub fn parse(input: &str) -> Result<Self, FixedParseError> {
        let value = parse_fixed(input, 0)?;
        u64::try_from(value)
            .map(Self)
            .map_err(|_| FixedParseError::OutOfRange {
                input: input.to_string(),
            })
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for UnitCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The unit a figure is expressed in, as declared in a table header.
///
/// Used to check a [`TableSpec`](super::TableSpec) against what the header
/// actually says. A header reading `($)` under a measure declared as a percentage
/// is a mis-specification, and ingestion refuses the table rather than emitting
/// numbers that are wrong by a factor of a hundred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// Singapore dollars.
    Sgd,
    /// Percent.
    Percent,
    /// Index points (dimensionless, relative to a base period).
    IndexPoints,
    /// Months (of remaining lease, or of waiting time).
    Months,
    /// Years.
    Years,
    /// Square metres.
    SquareMetres,
    /// A bare count of things.
    Count,
}

impl Unit {
    /// The unit markers that appear in real table headers, e.g. `"Median Price ($)"`.
    ///
    /// Deliberately conservative: an unrecognised marker yields `None` and the
    /// caller raises an issue, rather than this function guessing.
    pub fn parse_marker(marker: &str) -> Option<Self> {
        let normalised = marker.trim().trim_matches(['(', ')', '[', ']']).trim();
        match normalised.to_ascii_lowercase().as_str() {
            "$" | "s$" | "sgd" | "dollars" => Some(Unit::Sgd),
            "%" | "percent" | "pct" => Some(Unit::Percent),
            "index" | "index points" | "points" => Some(Unit::IndexPoints),
            "months" | "mth" | "mths" => Some(Unit::Months),
            "years" | "yrs" | "yr" => Some(Unit::Years),
            "sqm" | "sq m" | "m2" | "square metres" => Some(Unit::SquareMetres),
            "number" | "no." | "count" | "units" => Some(Unit::Count),
            _ => None,
        }
    }

    /// The kind of quantity this unit produces.
    pub fn kind(self) -> QuantityKind {
        match self {
            Unit::Sgd => QuantityKind::Money,
            Unit::Percent => QuantityKind::Percentage,
            Unit::IndexPoints => QuantityKind::Index,
            Unit::Months | Unit::Years => QuantityKind::Lease,
            Unit::SquareMetres => QuantityKind::Area,
            Unit::Count => QuantityKind::Count,
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Unit::Sgd => "$",
            Unit::Percent => "%",
            Unit::IndexPoints => "index points",
            Unit::Months => "months",
            Unit::Years => "years",
            Unit::SquareMetres => "sqm",
            Unit::Count => "count",
        };
        f.write_str(text)
    }
}

/// The discriminant of a [`Quantity`], used to declare in a
/// [`TableSpec`](super::TableSpec) what a column is expected to hold, before any
/// of it has been parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityKind {
    Money,
    Percentage,
    Index,
    Lease,
    Area,
    Count,
}

impl fmt::Display for QuantityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            QuantityKind::Money => "money",
            QuantityKind::Percentage => "percentage",
            QuantityKind::Index => "index",
            QuantityKind::Lease => "lease",
            QuantityKind::Area => "area",
            QuantityKind::Count => "count",
        };
        f.write_str(text)
    }
}

/// A published value, tagged with what kind of thing it is.
///
/// The sum type is the guardrail: there is no way to add a [`Quantity::Money`] to
/// a [`Quantity::Lease`] without matching on both and writing down what you think
/// that would mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantity {
    /// A price — a median, a mean, a floor or a ceiling of one.
    Money(SgdAmount),
    /// A proportion or a rate of change.
    Percentage(Percentage),
    /// A Resale Price Index reading, carrying its base period.
    Index(IndexPoint),
    /// A remaining lease.
    Lease(LeaseRemaining),
    /// A floor area.
    Area(FloorArea),
    /// A count of flats, transactions or households.
    Count(UnitCount),
}

impl Quantity {
    /// What kind of quantity this is.
    pub fn kind(self) -> QuantityKind {
        match self {
            Quantity::Money(_) => QuantityKind::Money,
            Quantity::Percentage(_) => QuantityKind::Percentage,
            Quantity::Index(_) => QuantityKind::Index,
            Quantity::Lease(_) => QuantityKind::Lease,
            Quantity::Area(_) => QuantityKind::Area,
            Quantity::Count(_) => QuantityKind::Count,
        }
    }

    /// Parses a cell into the declared kind of quantity.
    ///
    /// `index_base` is required when parsing an index, and ignored otherwise. It
    /// is threaded through rather than defaulted because an index without a base
    /// is not a value — see [`IndexPoint`].
    pub fn parse(
        input: &str,
        kind: QuantityKind,
        index_base: Option<Period>,
    ) -> Result<Self, FixedParseError> {
        match kind {
            QuantityKind::Money => SgdAmount::parse(input).map(Quantity::Money),
            QuantityKind::Percentage => Percentage::parse(input).map(Quantity::Percentage),
            QuantityKind::Index => {
                let base = index_base.ok_or_else(|| FixedParseError::NotANumber {
                    input: input.to_string(),
                    // An index with no declared base is a specification error, not
                    // a parse error, and the caller must be told which.
                    found: '?',
                })?;
                IndexPoint::parse(input, base).map(Quantity::Index)
            }
            QuantityKind::Lease => parse_fixed(input, 0).and_then(|months| {
                u32::try_from(months)
                    .map(|m| Quantity::Lease(LeaseRemaining::from_months(m)))
                    .map_err(|_| FixedParseError::OutOfRange {
                        input: input.to_string(),
                    })
            }),
            QuantityKind::Area => FloorArea::parse(input).map(Quantity::Area),
            QuantityKind::Count => UnitCount::parse(input).map(Quantity::Count),
        }
    }

    /// The price, if this is one. Convenience for the affordability seam — a CPF
    /// or policy layer wanting a price should not have to match six variants.
    pub fn as_money(self) -> Option<SgdAmount> {
        match self {
            Quantity::Money(amount) => Some(amount),
            _ => None,
        }
    }

    /// The remaining lease, if this is one. See the seam note in [`super`].
    pub fn as_lease(self) -> Option<LeaseRemaining> {
        match self {
            Quantity::Lease(lease) => Some(lease),
            _ => None,
        }
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Quantity::Money(v) => write!(f, "{v}"),
            Quantity::Percentage(v) => write!(f, "{v}"),
            Quantity::Index(v) => write!(f, "{v}"),
            Quantity::Lease(v) => write!(f, "{v}"),
            Quantity::Area(v) => write!(f, "{v}"),
            Quantity::Count(v) => write!(f, "{v}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::survey::period::Quarter;

    #[test]
    fn money_is_exact_to_the_cent() {
        let price = SgdAmount::parse("$540,000").unwrap();
        assert_eq!(price.cents(), 54_000_000);
        assert_eq!(price.to_string(), "S$540000.00");
    }

    #[test]
    fn an_index_carries_its_base_and_refuses_to_cross_a_rebasing() {
        let base_2009 = Period::Quarter {
            year: 2009,
            quarter: Quarter::Q1,
        };
        let base_2014 = Period::Quarter {
            year: 2014,
            quarter: Quarter::Q1,
        };

        let a = IndexPoint::parse("100.0", base_2009).unwrap();
        let b = IndexPoint::parse("110.0", base_2009).unwrap();
        // Same base: a 10% rise, computed exactly.
        assert_eq!(
            b.change_from(a).unwrap(),
            Percentage::from_basis_points(1000)
        );

        // Different base: this is the trap. `110` on a 2014 base and `100` on a
        // 2009 base are on different scales; their difference is not a price
        // movement, and must not be handed back as one.
        let rebased = IndexPoint::parse("110.0", base_2014).unwrap();
        let err = rebased.change_from(a).unwrap_err();
        assert_eq!(err.left, base_2009);
        assert_eq!(err.right, base_2014);
    }

    #[test]
    fn quantities_of_different_kinds_are_different_types() {
        let price = Quantity::parse("540000", QuantityKind::Money, None).unwrap();
        let lease = Quantity::parse("771", QuantityKind::Lease, None).unwrap();
        assert_eq!(price.kind(), QuantityKind::Money);
        assert_eq!(lease.kind(), QuantityKind::Lease);
        // The sum type is the guardrail: a price is not a lease, and asking for
        // one as the other yields None rather than a coerced number.
        assert!(price.as_lease().is_none());
        assert!(lease.as_money().is_none());
        assert_eq!(lease.as_lease().unwrap().whole_years(), 64);
    }

    #[test]
    fn parsing_an_index_without_a_base_fails_rather_than_defaulting() {
        assert!(Quantity::parse("195.5", QuantityKind::Index, None).is_err());
    }

    #[test]
    fn percentages_may_be_negative_because_prices_fall() {
        // Clamping to 0..=100 would corrupt a year-on-year decline, which is
        // exactly the figure a buyer most wants to be told honestly.
        let change = Percentage::parse("-3.5").unwrap();
        assert_eq!(change.basis_points(), -350);
        assert_eq!(change.to_string(), "-3.50%");
    }

    #[test]
    fn unit_markers_are_recognised_from_headers_or_refused() {
        assert_eq!(Unit::parse_marker("($)"), Some(Unit::Sgd));
        assert_eq!(Unit::parse_marker("(%)"), Some(Unit::Percent));
        assert_eq!(Unit::parse_marker("(sqm)"), Some(Unit::SquareMetres));
        // Not recognised -> None, so the caller raises an issue instead of this
        // function inventing a unit.
        assert_eq!(Unit::parse_marker("(furlongs)"), None);
    }

    #[test]
    fn lease_renders_the_way_hdb_states_it() {
        let lease = LeaseRemaining::from_years_months(64, 3);
        assert_eq!(lease.months(), 771);
        assert_eq!(lease.to_string(), "64y 3m");
    }
}
