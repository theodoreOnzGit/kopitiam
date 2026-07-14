//! Time series — and the joins this module refuses to make.
//!
//! A buyer watching the market wants a trend: "what has the median 4-room in this
//! town done over the last five years?" That is a legitimate and useful question,
//! and it is also where the most convincing fabrications come from.
//!
//! # The break rule
//!
//! > **A series may not span a methodology change, or a redefinition of its
//! > measure, unless the caller explicitly and in writing acknowledges it.**
//!
//! HDB revises things between editions. A town boundary moves. "Median resale
//! price" starts including grants, or stops. The Resale Price Index gets rebased.
//! Each of these produces two runs of numbers that *look* like one series and are
//! not — and plotting them together yields a trend line with a step in it that the
//! market never actually took.
//!
//! [`Series::new`] therefore **refuses** to build across a break. The caller who
//! genuinely needs the spliced series must pass an [`AcknowledgedBreak`] carrying
//! a written rationale, which is then preserved on the series and travels into
//! the knowledge graph. The point is not to make splicing impossible — sometimes
//! it is the right call — but to make it *impossible to do silently*.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::period::Period;
use super::statistic::{Caveat, Measure, Statistic};
use super::stratum::{Population, Stratum};

/// A discontinuity between two adjacent points in a would-be series.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MethodologyBreak {
    /// The two points came out of different release editions.
    Methodology {
        before: String,
        after: String,
        at: Period,
    },
    /// The measure kept its name but changed its footnote — the definition moved
    /// under the series. See [`Measure::redefines`].
    Definition {
        measure: String,
        before: Option<String>,
        after: Option<String>,
        at: Period,
    },
}

impl MethodologyBreak {
    /// The period at which the discontinuity occurs.
    pub fn at(&self) -> Period {
        match self {
            MethodologyBreak::Methodology { at, .. } => *at,
            MethodologyBreak::Definition { at, .. } => *at,
        }
    }
}

impl fmt::Display for MethodologyBreak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MethodologyBreak::Methodology { before, after, at } => write!(
                f,
                "at {at}, the methodology changed from `{before}` to `{after}`"
            ),
            MethodologyBreak::Definition {
                measure,
                before,
                after,
                at,
            } => write!(
                f,
                "at {at}, `{measure}` was redefined ({before:?} -> {after:?})"
            ),
        }
    }
}

/// A caller's written acknowledgement that they are splicing across a break, and
/// why.
///
/// The rationale is mandatory and cannot be empty. A boolean flag would let a
/// caller wave the check through with `true`; requiring prose forces them to
/// articulate the justification, and preserves it for whoever reads the series
/// later — which is the whole difference between a documented splice and a silent
/// one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcknowledgedBreak {
    at: Period,
    rationale: String,
}

impl AcknowledgedBreak {
    /// Acknowledges the break occurring at `at`, with a written reason.
    ///
    /// Rejects an empty or whitespace-only rationale: an acknowledgement that says
    /// nothing is not an acknowledgement.
    pub fn new(at: Period, rationale: impl Into<String>) -> Result<Self, SeriesError> {
        let rationale = rationale.into();
        if rationale.trim().is_empty() {
            return Err(SeriesError::EmptyRationale { at });
        }
        Ok(Self { at, rationale })
    }

    pub fn at(&self) -> Period {
        self.at
    }

    pub fn rationale(&self) -> &str {
        &self.rationale
    }
}

/// Why a series could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SeriesError {
    #[error("a series needs at least one point")]
    Empty,

    #[error(
        "the points in a series must all describe the same slice of the market; \
         found `{left}` and `{right}`"
    )]
    HeterogeneousStrata { left: String, right: String },

    #[error(
        "the points in a series must all be drawn from the same population; \
         found `{left}` and `{right}`"
    )]
    HeterogeneousPopulation { left: String, right: String },

    #[error(
        "the points in a series must all measure the same thing; \
         found `{left}` and `{right}`"
    )]
    HeterogeneousMeasure { left: String, right: String },

    #[error("two points in this series cover the same period ({period}); which is authoritative?")]
    DuplicatePeriod { period: Period },

    #[error(
        "this series spans {} unacknowledged break(s): {}. \
         Joining across a methodology change fabricates a trend the market never took. \
         If the splice is genuinely warranted, pass an AcknowledgedBreak saying why.",
        breaks.len(),
        breaks.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")
    )]
    UnacknowledgedBreak { breaks: Vec<MethodologyBreak> },

    #[error("an acknowledgement of the break at {at} must give a reason")]
    EmptyRationale { at: Period },
}

/// One measure, one stratum, one population, over time.
///
/// Points are held sorted by [`Period::start`]. The series knows about any breaks
/// it spans and the rationale for accepting them.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    measure: Measure,
    population: Population,
    stratum: Stratum,
    points: Vec<Statistic>,
    acknowledged: Vec<AcknowledgedBreak>,
}

impl Series {
    /// Builds a series, **refusing** any methodology or definition break.
    ///
    /// This is the constructor you should be reaching for. If it fails with
    /// [`SeriesError::UnacknowledgedBreak`], read the break before reaching for
    /// [`Series::new_acknowledging`] — the failure is usually telling you the two
    /// halves genuinely are not one series.
    pub fn new(points: Vec<Statistic>) -> Result<Self, SeriesError> {
        Self::build(points, Vec::new())
    }

    /// Builds a series across known breaks, each explicitly acknowledged with a
    /// written rationale.
    ///
    /// Any break *not* covered by an acknowledgement still fails the build — an
    /// acknowledgement is a statement about one specific discontinuity, not a
    /// blanket waiver.
    pub fn new_acknowledging(
        points: Vec<Statistic>,
        acknowledged: Vec<AcknowledgedBreak>,
    ) -> Result<Self, SeriesError> {
        Self::build(points, acknowledged)
    }

    fn build(
        mut points: Vec<Statistic>,
        acknowledged: Vec<AcknowledgedBreak>,
    ) -> Result<Self, SeriesError> {
        let first = points.first().ok_or(SeriesError::Empty)?;

        let measure = first.measure().clone();
        let population = first.population().clone();
        let stratum = first.stratum().clone();

        // A series must describe one thing. Mixed strata or populations are not a
        // trend, they are a pile of unrelated numbers.
        for point in &points {
            if point.stratum() != &stratum {
                return Err(SeriesError::HeterogeneousStrata {
                    left: stratum.to_string(),
                    right: point.stratum().to_string(),
                });
            }
            if point.population() != &population {
                return Err(SeriesError::HeterogeneousPopulation {
                    left: population.to_string(),
                    right: point.population().to_string(),
                });
            }
            // Names must match. Definitions may differ — that is a *break*, which
            // is detected below and is separately acknowledgeable. A wholly
            // different measure is not.
            if point.measure().name() != measure.name() {
                return Err(SeriesError::HeterogeneousMeasure {
                    left: measure.name().to_string(),
                    right: point.measure().name().to_string(),
                });
            }
        }

        // Sort explicitly by start month. `Period` is deliberately not `Ord`
        // (see its docs), so the ordering choice is made here, in the open.
        points.sort_by_key(|point| point.period().start());

        for window in points.windows(2) {
            if window[0].period() == window[1].period() {
                return Err(SeriesError::DuplicatePeriod {
                    period: window[0].period(),
                });
            }
        }

        let breaks = Self::detect_breaks(&points);
        let unacknowledged: Vec<MethodologyBreak> = breaks
            .into_iter()
            .filter(|brk| !acknowledged.iter().any(|ack| ack.at() == brk.at()))
            .collect();

        if !unacknowledged.is_empty() {
            return Err(SeriesError::UnacknowledgedBreak {
                breaks: unacknowledged,
            });
        }

        Ok(Self {
            measure,
            population,
            stratum,
            points,
            acknowledged,
        })
    }

    /// Finds every discontinuity between adjacent points.
    ///
    /// The break is reported at the period of the *later* point — the first
    /// observation produced under the new regime.
    fn detect_breaks(points: &[Statistic]) -> Vec<MethodologyBreak> {
        let mut breaks = Vec::new();
        for window in points.windows(2) {
            let (before, after) = (&window[0], &window[1]);

            if before.methodology() != after.methodology() {
                breaks.push(MethodologyBreak::Methodology {
                    before: before.methodology().id().to_string(),
                    after: after.methodology().id().to_string(),
                    at: after.period(),
                });
            }

            if after.measure().redefines(before.measure()) {
                breaks.push(MethodologyBreak::Definition {
                    measure: after.measure().name().to_string(),
                    before: before.measure().definition().map(str::to_string),
                    after: after.measure().definition().map(str::to_string),
                    at: after.period(),
                });
            }
        }
        breaks
    }

    pub fn measure(&self) -> &Measure {
        &self.measure
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn stratum(&self) -> &Stratum {
        &self.stratum
    }

    /// The points, in period order.
    pub fn points(&self) -> &[Statistic] {
        &self.points
    }

    /// The breaks this series was knowingly spliced across, and why. Empty for a
    /// clean series.
    ///
    /// **Show these to the user.** A spliced series is a claim with a footnote,
    /// and the footnote is load-bearing.
    pub fn acknowledged_breaks(&self) -> &[AcknowledgedBreak] {
        &self.acknowledged
    }

    /// Every caveat attaching to any point — chiefly small samples. A trend built
    /// from thin cells is a thin trend.
    pub fn caveats(&self) -> Vec<Caveat> {
        self.points
            .iter()
            .filter(|point| point.reliability().needs_warning())
            .map(|point| match point.basis().observations() {
                Some(observations) => Caveat::SmallSample {
                    stratum: format!("{} @ {}", point.stratum(), point.period()),
                    observations,
                },
                None => Caveat::BasisUnstated {
                    stratum: format!("{} @ {}", point.stratum(), point.period()),
                },
            })
            .collect()
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} for {} ({}), {} point(s)",
            self.measure,
            self.stratum,
            self.population,
            self.points.len()
        )?;
        for point in &self.points {
            writeln!(f, "  {} : {}", point.period(), point.quantity())?;
        }
        for ack in &self.acknowledged {
            writeln!(
                f,
                "  ! spliced across a break at {}: {}",
                ack.at(),
                ack.rationale()
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::survey::citation::{Citation, Locator};
    use crate::hdb::survey::quantity::{Quantity, SgdAmount, Unit};
    use crate::hdb::survey::statistic::{Basis, LeaseProfile, Methodology, SampleCount};
    use crate::hdb::survey::stratum::Dimension;

    fn synthetic_citation() -> Citation {
        Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("SYNTHETIC-1".into()),
            Period::Year(2024),
        )
    }

    fn point(year: u16, dollars: i64, methodology: &str, definition: Option<&str>) -> Statistic {
        let mut measure = Measure::new("Median resale price", Unit::Sgd);
        if let Some(definition) = definition {
            measure = measure.with_definition(definition);
        }
        Statistic::new(
            measure,
            Quantity::Money(SgdAmount::from_dollars(dollars)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, "TAMPINES"),
            Period::Year(year),
            Basis::Census {
                observations: SampleCount::new(100),
            },
            LeaseProfile::NotApplicable,
            Methodology::new(methodology),
            synthetic_citation(),
        )
        .unwrap()
    }

    #[test]
    fn a_clean_series_builds_and_sorts_by_period() {
        // Handed in deliberately out of order.
        let series = Series::new(vec![
            point(2024, 333_333, "M1", None),
            point(2022, 111_111, "M1", None),
            point(2023, 222_222, "M1", None),
        ])
        .unwrap();

        let periods: Vec<Period> = series.points().iter().map(Statistic::period).collect();
        assert_eq!(
            periods,
            vec![Period::Year(2022), Period::Year(2023), Period::Year(2024)]
        );
        assert!(series.acknowledged_breaks().is_empty());
    }

    #[test]
    fn a_methodology_change_refuses_the_join() {
        // THE test. Two runs of numbers that look like one series and are not.
        let err = Series::new(vec![
            point(2022, 111_111, "SHS-2018", None),
            point(2023, 222_222, "SHS-2023", None),
        ])
        .unwrap_err();

        match err {
            SeriesError::UnacknowledgedBreak { breaks } => {
                assert_eq!(breaks.len(), 1);
                assert_eq!(breaks[0].at(), Period::Year(2023));
                assert!(matches!(
                    breaks[0],
                    MethodologyBreak::Methodology { .. }
                ));
            }
            other => panic!("expected UnacknowledgedBreak, got {other:?}"),
        }
    }

    #[test]
    fn a_redefinition_also_breaks_the_series() {
        // Same methodology id, but the footnote under the column moved: prices
        // stopped being quoted before grants and started being quoted after.
        let err = Series::new(vec![
            point(2022, 111_111, "M1", Some("before grants")),
            point(2023, 222_222, "M1", Some("after grants")),
        ])
        .unwrap_err();

        match err {
            SeriesError::UnacknowledgedBreak { breaks } => {
                assert!(matches!(breaks[0], MethodologyBreak::Definition { .. }));
            }
            other => panic!("expected a definition break, got {other:?}"),
        }
    }

    #[test]
    fn an_acknowledged_break_permits_the_join_and_preserves_the_reason() {
        let ack = AcknowledgedBreak::new(
            Period::Year(2023),
            "Splicing deliberately: the 2023 revision changed only town boundaries, \
             which do not affect this town.",
        )
        .unwrap();

        let series = Series::new_acknowledging(
            vec![
                point(2022, 111_111, "SHS-2018", None),
                point(2023, 222_222, "SHS-2023", None),
            ],
            vec![ack],
        )
        .unwrap();

        assert_eq!(series.points().len(), 2);
        // The rationale survives into the series, so a reader downstream can see
        // that a judgment was made and what it was.
        assert_eq!(series.acknowledged_breaks().len(), 1);
        assert!(series.acknowledged_breaks()[0]
            .rationale()
            .contains("town boundaries"));
        assert!(series.to_string().contains("spliced across a break"));
    }

    #[test]
    fn an_acknowledgement_of_the_wrong_break_does_not_wave_through_another() {
        // An acknowledgement is about one specific discontinuity, not a blanket
        // waiver. Acknowledging a break at 2023 must not silently permit one at
        // 2024.
        let ack = AcknowledgedBreak::new(Period::Year(2023), "known and accepted").unwrap();
        let err = Series::new_acknowledging(
            vec![
                point(2022, 111_111, "M1", None),
                point(2023, 222_222, "M2", None),
                point(2024, 333_333, "M3", None),
            ],
            vec![ack],
        )
        .unwrap_err();

        match err {
            SeriesError::UnacknowledgedBreak { breaks } => {
                assert_eq!(breaks.len(), 1);
                assert_eq!(breaks[0].at(), Period::Year(2024));
            }
            other => panic!("expected the 2024 break to survive, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_rationale_is_not_an_acknowledgement() {
        assert!(matches!(
            AcknowledgedBreak::new(Period::Year(2023), "   "),
            Err(SeriesError::EmptyRationale { .. })
        ));
    }

    #[test]
    fn a_series_must_describe_one_slice_of_the_market() {
        let tampines = point(2022, 111_111, "M1", None);
        let elsewhere = Statistic::new(
            Measure::new("Median resale price", Unit::Sgd),
            Quantity::Money(SgdAmount::from_dollars(222_222)),
            Population::ResaleTransactions,
            Stratum::all().with(Dimension::Town, "QUEENSTOWN"),
            Period::Year(2023),
            Basis::Census {
                observations: SampleCount::new(100),
            },
            LeaseProfile::NotApplicable,
            Methodology::new("M1"),
            synthetic_citation(),
        )
        .unwrap();

        assert!(matches!(
            Series::new(vec![tampines, elsewhere]),
            Err(SeriesError::HeterogeneousStrata { .. })
        ));
    }

    #[test]
    fn two_points_for_the_same_period_are_ambiguous_and_refused() {
        assert!(matches!(
            Series::new(vec![
                point(2023, 111_111, "M1", None),
                point(2023, 222_222, "M1", None),
            ]),
            Err(SeriesError::DuplicatePeriod { .. })
        ));
    }

    #[test]
    fn an_empty_series_is_refused() {
        assert!(matches!(Series::new(vec![]), Err(SeriesError::Empty)));
    }
}
