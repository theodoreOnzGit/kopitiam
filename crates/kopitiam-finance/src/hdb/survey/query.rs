//! Asking the market a question — and being told honestly when it has no answer.
//!
//! # The refusal is the feature
//!
//! A buyer asks: "what did a 4-room in Punggol go for in 2021?" If no such figure
//! was published, there are two things this module could do:
//!
//! 1. Interpolate between 2020 and 2022 and hand back a number.
//! 2. Say that HDB did not publish that.
//!
//! Option 1 is what a plausible-looking system does, and it is indefensible. The
//! interpolated number would carry a citation — to documents that *do not contain
//! it*. It would carry a sample size — of transactions that were never counted. It
//! would look exactly like a fact, and it would be a fabrication.
//!
//! So [`SurveyStore::query`] returns [`QueryOutcome::NoData`], and the [`NoData`]
//! says what *is* available nearby, so the caller can ask a question that has an
//! answer. It never fills the gap itself.

use std::fmt;

use super::period::Period;
use super::statistic::Statistic;
use super::stratum::{Dimension, Population, Stratum};

/// A question about the market.
///
/// Every field is optional and acts as a filter; an empty query matches
/// everything. Filters are conjunctive.
#[derive(Debug, Clone, Default)]
pub struct Query {
    measure: Option<String>,
    population: Option<Population>,
    stratum: Option<Stratum>,
    period: Option<Period>,
    /// When set, a stratum matches if it is *at least as specific* as the query's
    /// — so asking for `{Town: Tampines}` also returns
    /// `{Town: Tampines, FlatType: 4-Room}`. Off by default, because a buyer
    /// asking about Tampines usually means the Tampines aggregate, not every
    /// sub-slice of it.
    include_narrower: bool,
}

impl Query {
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts to a measure by name. Note that two measures can share a name
    /// and differ by footnote — this matches both, and the results will carry the
    /// difference in [`Statistic::measure`].
    pub fn measure(mut self, name: impl Into<String>) -> Self {
        self.measure = Some(name.into());
        self
    }

    pub fn population(mut self, population: Population) -> Self {
        self.population = Some(population);
        self
    }

    pub fn stratum(mut self, stratum: Stratum) -> Self {
        self.stratum = Some(stratum);
        self
    }

    pub fn period(mut self, period: Period) -> Self {
        self.period = Some(period);
        self
    }

    /// Also return strata *narrower* than the one asked for. See the field docs.
    pub fn include_narrower(mut self, include: bool) -> Self {
        self.include_narrower = include;
        self
    }

    fn matches(&self, statistic: &Statistic) -> bool {
        if let Some(measure) = &self.measure
            && statistic.measure().name() != measure
        {
            return false;
        }
        if let Some(population) = &self.population
            && statistic.population() != population
        {
            return false;
        }
        if let Some(period) = self.period
            && statistic.period() != period
        {
            return false;
        }
        if let Some(stratum) = &self.stratum {
            let matched = if self.include_narrower {
                stratum.contains(statistic.stratum())
            } else {
                statistic.stratum() == stratum
            };
            if !matched {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(measure) = &self.measure {
            parts.push(format!("measure=`{measure}`"));
        }
        if let Some(population) = &self.population {
            parts.push(format!("population=`{population}`"));
        }
        if let Some(stratum) = &self.stratum {
            parts.push(format!("stratum=`{stratum}`"));
        }
        if let Some(period) = self.period {
            parts.push(format!("period=`{period}`"));
        }
        if parts.is_empty() {
            return f.write_str("(everything)");
        }
        f.write_str(&parts.join(", "))
    }
}

/// The honest answer when nothing matches.
///
/// Carries enough context for the caller to ask a *different*, answerable
/// question — without ever answering the unanswerable one.
#[derive(Debug, Clone, PartialEq)]
pub struct NoData {
    /// What was asked.
    pub query: String,
    /// Periods for which this measure and stratum *were* published. If the query
    /// asked for 2021 and this holds 2020 and 2022, the caller now knows the gap
    /// is real — and that KOPITIAM will not fill it for them.
    pub periods_available: Vec<Period>,
    /// Strata that exist for this measure, when the stratum was the thing that
    /// did not match. Truncated; this is a hint, not a dump.
    pub strata_available: Vec<String>,
    /// Why no number is being returned.
    pub explanation: &'static str,
}

impl NoData {
    const NO_INTERPOLATION: &'static str =
        "KOPITIAM reports what was published. It does not interpolate between periods, \
         extrapolate beyond them, or pool across strata. No figure matching this query was \
         published, so no figure is being returned. An interpolated number would carry a \
         citation to a document that does not contain it.";
}

impl fmt::Display for NoData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "No published figure matches {}.", self.query)?;
        writeln!(f, "{}", self.explanation)?;
        if !self.periods_available.is_empty() {
            let periods: Vec<String> = self
                .periods_available
                .iter()
                .map(ToString::to_string)
                .collect();
            writeln!(f, "Published periods for this slice: {}", periods.join(", "))?;
        }
        if !self.strata_available.is_empty() {
            writeln!(
                f,
                "Published strata for this measure: {}",
                self.strata_available.join("; ")
            )?;
        }
        Ok(())
    }
}

/// What a query returned.
///
/// An enum rather than an empty `Vec`, because "there is no such figure" is a
/// *result* that deserves an explanation, not an absence a caller can absent-
/// mindedly treat as zero rows and move on from.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOutcome {
    /// Figures matching the query, each with its citation, basis and reliability.
    Found(Vec<Statistic>),
    /// Nothing matched, and here is what does exist.
    NoData(NoData),
}

impl QueryOutcome {
    /// The matching figures, or an empty slice.
    pub fn statistics(&self) -> &[Statistic] {
        match self {
            QueryOutcome::Found(statistics) => statistics,
            QueryOutcome::NoData(_) => &[],
        }
    }

    /// The single matching figure, if the query identified exactly one.
    ///
    /// Returns `None` when several matched — an ambiguous answer is not an answer,
    /// and silently taking the first would be a coin flip presented as a fact.
    pub fn exactly_one(&self) -> Option<&Statistic> {
        match self {
            QueryOutcome::Found(statistics) if statistics.len() == 1 => statistics.first(),
            _ => None,
        }
    }

    /// Whether any returned figure needs a health warning shown alongside it.
    ///
    /// A caller rendering these to a human **must** check this. See
    /// [`Reliability`].
    pub fn needs_warning(&self) -> bool {
        self.statistics()
            .iter()
            .any(|statistic| statistic.reliability().needs_warning())
    }
}

/// The figures this module knows about.
///
/// Deliberately a plain `Vec` behind a query method rather than an indexed store.
/// The volumes involved — a few thousand published figures — do not justify an
/// index, and CLAUDE.md is explicit about avoiding premature optimization. If a
/// full resale-transaction dataset (hundreds of thousands of rows) is ever loaded
/// here, this is the type to revisit, and the query API above will not have to
/// change to do it.
#[derive(Debug, Clone, Default)]
pub struct SurveyStore {
    statistics: Vec<Statistic>,
}

impl SurveyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds figures — typically the `statistics` from an [`super::Ingested`].
    pub fn insert(&mut self, statistics: impl IntoIterator<Item = Statistic>) {
        self.statistics.extend(statistics);
    }

    /// Every figure held.
    pub fn statistics(&self) -> &[Statistic] {
        &self.statistics
    }

    pub fn len(&self) -> usize {
        self.statistics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.statistics.is_empty()
    }

    /// Answers a query, or explains why it cannot.
    pub fn query(&self, query: &Query) -> QueryOutcome {
        let matches: Vec<Statistic> = self
            .statistics
            .iter()
            .filter(|statistic| query.matches(statistic))
            .cloned()
            .collect();

        if !matches.is_empty() {
            return QueryOutcome::Found(matches);
        }

        // Nothing matched. Work out what *does* exist, so the caller can ask an
        // answerable question — without answering the unanswerable one.

        // Periods available for this measure and stratum, ignoring the period
        // filter. This is what tells a caller "2021 is genuinely a gap".
        let mut relaxed = query.clone();
        relaxed.period = None;
        let mut periods_available: Vec<Period> = self
            .statistics
            .iter()
            .filter(|statistic| relaxed.matches(statistic))
            .map(Statistic::period)
            .collect();
        periods_available.sort_by_key(|period| period.start());
        periods_available.dedup();

        // Strata available for this measure, ignoring the stratum and period.
        let mut by_measure = Query::new();
        by_measure.measure = query.measure.clone();
        by_measure.population = query.population.clone();
        let mut strata_available: Vec<String> = self
            .statistics
            .iter()
            .filter(|statistic| by_measure.matches(statistic))
            .map(|statistic| statistic.stratum().to_string())
            .collect();
        strata_available.sort();
        strata_available.dedup();
        strata_available.truncate(12);

        QueryOutcome::NoData(NoData {
            query: query.to_string(),
            periods_available,
            strata_available,
            explanation: NoData::NO_INTERPOLATION,
        })
    }

    /// The periods for which a given measure and stratum were published, in order.
    pub fn published_periods(&self, measure: &str, stratum: &Stratum) -> Vec<Period> {
        let mut periods: Vec<Period> = self
            .statistics
            .iter()
            .filter(|statistic| {
                statistic.measure().name() == measure && statistic.stratum() == stratum
            })
            .map(Statistic::period)
            .collect();
        periods.sort_by_key(|period| period.start());
        periods.dedup();
        periods
    }

    /// The levels published on a dimension — the towns, the flat types.
    ///
    /// What a UI needs to populate a dropdown *from the data*, rather than from a
    /// hardcoded list that drifts out of step with what was actually published.
    pub fn levels(&self, dimension: &Dimension) -> Vec<String> {
        let mut levels: Vec<String> = self
            .statistics
            .iter()
            .filter_map(|statistic| statistic.stratum().level(dimension))
            .map(|level| level.as_str().to_string())
            .collect();
        levels.sort();
        levels.dedup();
        levels
    }
}

impl FromIterator<Statistic> for SurveyStore {
    fn from_iter<T: IntoIterator<Item = Statistic>>(iter: T) -> Self {
        Self {
            statistics: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::survey::citation::{Citation, Locator};
    use crate::hdb::survey::quantity::{Quantity, SgdAmount, Unit};
    use crate::hdb::survey::statistic::{
        Basis, LeaseProfile, Measure, Methodology, Reliability, SampleCount,
    };

    fn synthetic_citation() -> Citation {
        Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("SYNTHETIC-1".into()),
            Period::Year(2024),
        )
    }

    fn price(town: &str, year: u16, dollars: i64, observations: u32) -> Statistic {
        Statistic::new(
            Measure::new("Median resale price", Unit::Sgd),
            Quantity::Money(SgdAmount::from_dollars(dollars)),
            Population::ResaleTransactions,
            Stratum::all()
                .with(Dimension::Town, town)
                .with(Dimension::FlatType, "4-Room"),
            Period::Year(year),
            Basis::Census {
                observations: SampleCount::new(observations),
            },
            LeaseProfile::NotApplicable,
            Methodology::new("SYNTHETIC METHODOLOGY A"),
            synthetic_citation(),
        )
        .unwrap()
    }

    fn store() -> SurveyStore {
        let mut store = SurveyStore::new();
        store.insert([
            price("TAMPINES", 2020, 111_111, 400),
            price("TAMPINES", 2022, 222_222, 450),
            price("PUNGGOL", 2022, 333_333, 5),
        ]);
        store
    }

    fn tampines() -> Stratum {
        Stratum::all()
            .with(Dimension::Town, "TAMPINES")
            .with(Dimension::FlatType, "4-Room")
    }

    #[test]
    fn a_matching_query_returns_the_figure_with_its_citation_and_basis() {
        let outcome = store().query(
            &Query::new()
                .measure("Median resale price")
                .stratum(tampines())
                .period(Period::Year(2022)),
        );
        let statistic = outcome.exactly_one().expect("exactly one 2022 figure");
        assert_eq!(
            statistic.quantity(),
            Quantity::Money(SgdAmount::from_dollars(222_222))
        );
        // The two things that make it a fact rather than a rumour came back with it.
        assert_eq!(
            statistic.citation().publication(),
            "SYNTHETIC FIXTURE — NOT HDB DATA"
        );
        assert_eq!(statistic.basis().observations().unwrap().get(), 450);
    }

    #[test]
    fn a_gap_in_the_data_is_an_honest_refusal_not_an_interpolation() {
        // THE test. 2020 and 2022 exist; 2021 does not. A system that returned
        // ~$166k here — the midpoint — would be fabricating a citation.
        let outcome = store().query(
            &Query::new()
                .measure("Median resale price")
                .stratum(tampines())
                .period(Period::Year(2021)),
        );

        let no_data = match outcome {
            QueryOutcome::NoData(no_data) => no_data,
            QueryOutcome::Found(found) => {
                panic!("2021 was never published; got {} figure(s)", found.len())
            }
        };

        // It tells the caller the gap is real, and what *does* exist.
        assert_eq!(
            no_data.periods_available,
            vec![Period::Year(2020), Period::Year(2022)]
        );
        assert!(no_data.explanation.contains("does not interpolate"));
        assert!(no_data.to_string().contains("2020"));
    }

    #[test]
    fn a_query_for_an_unpublished_town_says_which_towns_exist() {
        let outcome = store().query(
            &Query::new()
                .measure("Median resale price")
                .stratum(Stratum::all().with(Dimension::Town, "ATLANTIS")),
        );
        match outcome {
            QueryOutcome::NoData(no_data) => {
                assert!(!no_data.strata_available.is_empty());
                assert!(no_data
                    .strata_available
                    .iter()
                    .any(|s| s.contains("TAMPINES")));
            }
            QueryOutcome::Found(_) => panic!("ATLANTIS is not an HDB town"),
        }
    }

    #[test]
    fn an_ambiguous_query_does_not_silently_pick_one() {
        // Two figures match (2020 and 2022). Returning the first would be a coin
        // flip presented as a fact.
        let outcome = store().query(&Query::new().stratum(tampines()));
        assert_eq!(outcome.statistics().len(), 2);
        assert!(outcome.exactly_one().is_none());
    }

    #[test]
    fn a_thin_result_reports_that_it_needs_a_warning() {
        // Punggol 2022 rests on 5 transactions. A caller rendering this to a human
        // must be told to warn them.
        let outcome = store().query(
            &Query::new().stratum(
                Stratum::all()
                    .with(Dimension::Town, "PUNGGOL")
                    .with(Dimension::FlatType, "4-Room"),
            ),
        );
        assert!(outcome.needs_warning());
        assert!(matches!(
            outcome.statistics()[0].reliability(),
            Reliability::LowPrecision { .. }
        ));

        // Whereas Tampines does not.
        let solid = store().query(&Query::new().stratum(tampines()).period(Period::Year(2022)));
        assert!(!solid.needs_warning());
    }

    #[test]
    fn narrower_strata_are_excluded_unless_asked_for() {
        let mut store = SurveyStore::new();
        store.insert([price("TAMPINES", 2022, 111_111, 400)]);

        // Asking about Tampines *in general* (no flat type) must not silently
        // return the 4-room figure as though it were the town aggregate.
        let town_only = Stratum::all().with(Dimension::Town, "TAMPINES");
        assert!(matches!(
            store.query(&Query::new().stratum(town_only.clone())),
            QueryOutcome::NoData(_)
        ));

        // Unless the caller explicitly opts in.
        let outcome = store.query(&Query::new().stratum(town_only).include_narrower(true));
        assert_eq!(outcome.statistics().len(), 1);
    }

    #[test]
    fn levels_are_discovered_from_the_data_not_hardcoded() {
        let towns = store().levels(&Dimension::Town);
        assert_eq!(towns, vec!["PUNGGOL".to_string(), "TAMPINES".to_string()]);
    }

    #[test]
    fn published_periods_reports_the_real_gaps() {
        assert_eq!(
            store().published_periods("Median resale price", &tampines()),
            vec![Period::Year(2020), Period::Year(2022)]
        );
    }
}
