//! The slices the market is published in — and which of them may be compared.
//!
//! HDB never publishes "the price of a flat". It publishes the price of a flat
//! *in a town*, *of a type*, *on a storey range*, *at a floor area*, *with a
//! remaining lease*. Those coordinates are a [`Stratum`], and they are the whole
//! reason a naive price comparison misleads.
//!
//! # The confounding rule
//!
//! > **Two strata may be compared only if they differ in at most one dimension.**
//!
//! This is the central safety property of the module. Compare a 4-room in
//! Tampines against a 5-room in Queenstown and the difference between them is
//! *both* a town effect and a flat-type effect, tangled together and impossible
//! to separate. A buyer reading "Queenstown is $180k more expensive" would be
//! drawing a conclusion the data does not support.
//!
//! Vary one dimension and you have a controlled comparison. Vary two and you have
//! a confounded one, which this module returns as a typed error rather than a
//! number. See [`super::Incomparability::ConfoundedStrata`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

/// The population frame a figure is drawn from.
///
/// This is the "what was measured over" that makes a number a fact rather than a
/// rumour, and it is *not* the same thing as a stratum. A stratum narrows a
/// population; the population says what kind of thing is being counted at all.
///
/// The distinction bites: the median price of **resale transactions** in a town
/// is a statistic about *flats that sold*. It is not a statistic about *flats
/// that exist* — the ones that sold are exactly the ones someone was willing to
/// sell, which is a different and self-selected set. Joining the two frames
/// produces nonsense, so they are different [`Population`]s and this module
/// refuses to compare across them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Population {
    /// Flats that changed hands on the resale market in the period. The frame
    /// behind every transacted-price figure.
    ResaleTransactions,
    /// The stock of HDB flats in existence, sold or not.
    FlatStock,
    /// Flats offered in a Build-To-Order or Sale of Balance exercise.
    FlatSupply,
    /// Households resident in HDB flats — the frame of the Sample Household
    /// Survey. Retained because HDB does publish it and a buyer may legitimately
    /// want it, but it is *not* the frame any price statistic is drawn from.
    ResidentHouseholds,
    /// A frame this module does not know about. Carrying it explicitly is better
    /// than forcing it into a wrong variant; comparisons against it are refused
    /// unless the names match exactly.
    Other(String),
}

impl fmt::Display for Population {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Population::ResaleTransactions => "resale transactions",
            Population::FlatStock => "HDB flat stock",
            Population::FlatSupply => "flat supply",
            Population::ResidentHouseholds => "resident HDB households",
            Population::Other(name) => name.as_str(),
        };
        f.write_str(text)
    }
}

/// An axis the market is sliced along.
///
/// Ordered (`Ord`) so that a [`Stratum`]'s dimensions iterate deterministically —
/// two strata built in different orders must render, hash and compare identically
/// (CLAUDE.md, deterministic behaviour).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dimension {
    /// The HDB town, e.g. `Tampines`, `Queenstown`.
    Town,
    /// The flat type, e.g. `3-Room`, `4-Room`, `Executive`.
    FlatType,
    /// The storey band, e.g. `07 TO 09`. Higher floors command a premium, so this
    /// is a genuine price dimension and not decoration.
    StoreyRange,
    /// A floor-area band, e.g. `90-99 sqm`.
    FloorAreaBand,
    /// A band of remaining lease, e.g. `60-69 years`. See the lease-profile note
    /// on [`super::LeaseRemaining`] — this is a first-class price driver.
    LeaseBand,
    /// The flat model, e.g. `Improved`, `Model A`, `Maisonette`.
    FlatModel,
    /// An axis this module does not know about, carried rather than discarded.
    Other(String),
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Dimension::Town => "town",
            Dimension::FlatType => "flat type",
            Dimension::StoreyRange => "storey range",
            Dimension::FloorAreaBand => "floor area",
            Dimension::LeaseBand => "remaining lease",
            Dimension::FlatModel => "flat model",
            Dimension::Other(name) => name.as_str(),
        };
        f.write_str(text)
    }
}

/// The value a stratum takes on one dimension, e.g. `"Tampines"`, `"4-Room"`.
///
/// Deliberately a string rather than an enum of towns. Towns get built, flat
/// types get introduced, and a closed enum would turn every HDB announcement into
/// a breaking change to this crate. The cost is that `"4-Room"` and `"4 ROOM"`
/// are different levels — so [`LevelValue::new`] normalises case and whitespace,
/// which is exactly the kind of thing that must be done once, here, rather than
/// at each call site.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LevelValue(String);

impl LevelValue {
    /// Normalises a level label so that the same slice, spelled differently in
    /// two publications, is recognised as the same slice.
    ///
    /// Collapses internal whitespace and upper-cases. `"4 room"`, `"4-ROOM"` and
    /// `"4  Room"` are *not* unified — the hyphen is meaningful punctuation this
    /// function will not second-guess — but `"4-Room"` and `"4-ROOM "` are.
    pub fn new(label: impl AsRef<str>) -> Self {
        let normalised = label
            .as_ref()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_uppercase();
        Self(normalised)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LevelValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A slice of the market: a set of (dimension, level) constraints.
///
/// The empty stratum is the whole population — "all resale transactions", with no
/// restriction. Adding a dimension narrows it.
///
/// Backed by a `BTreeMap` so iteration, `Display` and `Hash` are deterministic
/// regardless of insertion order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Stratum {
    levels: BTreeMap<Dimension, LevelValue>,
}

impl Stratum {
    /// The unrestricted stratum — the whole population.
    pub fn all() -> Self {
        Self::default()
    }

    /// Narrows this stratum along one dimension.
    pub fn with(mut self, dimension: Dimension, level: impl AsRef<str>) -> Self {
        self.levels.insert(dimension, LevelValue::new(level));
        self
    }

    /// The level this stratum takes on a dimension, if it constrains it at all.
    pub fn level(&self, dimension: &Dimension) -> Option<&LevelValue> {
        self.levels.get(dimension)
    }

    /// The dimensions this stratum constrains.
    pub fn dimensions(&self) -> impl Iterator<Item = &Dimension> {
        self.levels.keys()
    }

    /// Whether this stratum places no restriction at all.
    pub fn is_all(&self) -> bool {
        self.levels.is_empty()
    }

    /// The dimensions on which two strata take *different* levels — including
    /// dimensions one constrains and the other does not.
    ///
    /// This is the raw material of the confounding rule: a comparison is
    /// controlled when this returns exactly one dimension.
    pub fn varying_dimensions(&self, other: &Stratum) -> Vec<Dimension> {
        let keys: BTreeSet<&Dimension> = self.levels.keys().chain(other.levels.keys()).collect();
        keys.into_iter()
            .filter(|dimension| self.levels.get(*dimension) != other.levels.get(*dimension))
            .cloned()
            .collect()
    }

    /// Whether `self` wholly contains `other` — i.e. `self` is the broader slice.
    ///
    /// A stratum contains another when it places a *subset* of the constraints:
    /// `{}` (all flats) contains `{Town: Tampines}`, and `{Town: Tampines}`
    /// contains `{Town: Tampines, FlatType: 4-Room}`.
    ///
    /// This matters because comparing a town against the national figure is a
    /// **part-whole** comparison — Tampines is *inside* the national aggregate,
    /// so the two are not independent. That is a legitimate thing for a buyer to
    /// want ("is Tampines above the national median?"), so it is a
    /// [`super::Caveat`] rather than an error — but it must be *said*.
    pub fn contains(&self, other: &Stratum) -> bool {
        self.levels
            .iter()
            .all(|(dimension, level)| other.levels.get(dimension) == Some(level))
    }
}

impl fmt::Display for Stratum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.levels.is_empty() {
            return f.write_str("all");
        }
        let rendered: Vec<String> = self
            .levels
            .iter()
            .map(|(dimension, level)| format!("{dimension}={level}"))
            .collect();
        f.write_str(&rendered.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_labels_normalise_so_the_same_slice_matches_itself() {
        assert_eq!(LevelValue::new("4-Room"), LevelValue::new("4-ROOM "));
        assert_eq!(LevelValue::new("ang  mo kio"), LevelValue::new("Ang Mo Kio"));
    }

    #[test]
    fn strata_are_order_independent() {
        // Built in opposite orders, these must be the same slice of the market.
        let a = Stratum::all()
            .with(Dimension::Town, "Tampines")
            .with(Dimension::FlatType, "4-Room");
        let b = Stratum::all()
            .with(Dimension::FlatType, "4-Room")
            .with(Dimension::Town, "Tampines");
        assert_eq!(a, b);
        assert_eq!(a.to_string(), b.to_string());
    }

    #[test]
    fn varying_one_dimension_is_a_controlled_comparison() {
        let tampines = Stratum::all()
            .with(Dimension::Town, "Tampines")
            .with(Dimension::FlatType, "4-Room");
        let queenstown = Stratum::all()
            .with(Dimension::Town, "Queenstown")
            .with(Dimension::FlatType, "4-Room");
        assert_eq!(
            tampines.varying_dimensions(&queenstown),
            vec![Dimension::Town]
        );
    }

    #[test]
    fn varying_two_dimensions_is_confounded() {
        // The trap: a 4-room in Tampines against a 5-room in Queenstown. The
        // difference is a town effect AND a flat-type effect, and no amount of
        // arithmetic separates them.
        let tampines_4 = Stratum::all()
            .with(Dimension::Town, "Tampines")
            .with(Dimension::FlatType, "4-Room");
        let queenstown_5 = Stratum::all()
            .with(Dimension::Town, "Queenstown")
            .with(Dimension::FlatType, "5-Room");
        let varying = tampines_4.varying_dimensions(&queenstown_5);
        assert_eq!(varying.len(), 2);
        assert!(varying.contains(&Dimension::Town));
        assert!(varying.contains(&Dimension::FlatType));
    }

    #[test]
    fn an_unconstrained_dimension_still_counts_as_varying() {
        // "4-room flats in Tampines" vs "flats in Tampines" differ: the second
        // pools every flat type. Treating the missing constraint as "matches
        // anything" would silently compare a slice against the whole.
        let with_type = Stratum::all()
            .with(Dimension::Town, "Tampines")
            .with(Dimension::FlatType, "4-Room");
        let without_type = Stratum::all().with(Dimension::Town, "Tampines");
        assert_eq!(
            with_type.varying_dimensions(&without_type),
            vec![Dimension::FlatType]
        );
    }

    #[test]
    fn containment_detects_part_whole_relationships() {
        let all = Stratum::all();
        let tampines = Stratum::all().with(Dimension::Town, "Tampines");
        let tampines_4 = tampines.clone().with(Dimension::FlatType, "4-Room");

        assert!(all.contains(&tampines));
        assert!(tampines.contains(&tampines_4));
        assert!(!tampines_4.contains(&tampines));
        // Containment is not symmetric, and the direction is the whole point.
        assert!(!tampines.contains(&all));
    }
}
