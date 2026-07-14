//! Turning a reconstructed PDF table into [`Statistic`]s — or refusing to.
//!
//! # What `kopitiam-document` actually hands us
//!
//! Exactly this, and nothing more:
//!
//! ```ignore
//! pub struct Table {
//!     pub headers: Vec<String>,
//!     pub rows: Vec<Vec<String>>,
//! }
//! ```
//!
//! A grid of strings. That is a perfectly reasonable thing for a document
//! reconstructor to produce, and it is *much* less than a statistical table
//! contains. Everything below is what has to be recovered on top of it, and where
//! the reconstruction cannot help:
//!
//! * **Units live in the header text**, as `"Median Price ($)"`. Recovered here by
//!   [`Unit::parse_marker`], and *checked* against what the caller declared. A
//!   header saying `(%)` under a measure declared as money is a mis-specification
//!   wrong by a factor of ten thousand, and it aborts the whole table.
//! * **Footnotes redefine columns**, and `Table` has nowhere to put them — the
//!   footnote text lands in a separate `Paragraph` block *after* the table, with
//!   nothing linking the two. The caller must therefore supply the footnote
//!   bodies in the [`TableSpec`]. If a marker appears in a header and no matching
//!   footnote was supplied, **the column is dropped**, because a footnote is
//!   exactly the thing that tells you the column means something other than what
//!   it says.
//! * **Merged and spanning cells do not survive.** `kopitiam-document` detects a
//!   table as a run of lines with the *same number* of x-aligned cells; a spanning
//!   header ("2023" over two sub-columns) breaks that test, so such a table either
//!   fails to be detected at all or arrives with the spanning row missing. There
//!   is no signal in `Table` to distinguish the two. This is a real gap and it is
//!   not papered over here.
//! * **Page breaks split a table in two.** The continuation gets its *first data
//!   row promoted to `headers`*. [`ingest_table`] detects the tell-tale — a header
//!   cell that parses as a number — and refuses the table rather than reading a
//!   price as a column name.
//!
//! # The governing principle
//!
//! This is **not a guesser**. It parses what can be parsed deterministically and
//! requires the caller to *declare* everything else. Where a cell cannot be fully
//! resolved, it emits an [`IngestIssue`] and **drops the cell**. It never emits a
//! number it is not sure of.
//!
//! A dropped cell is a visible hole. A guessed cell is an invisible lie.

use std::collections::BTreeMap;
use std::fmt;

use kopitiam_document::Table;

use super::citation::Citation;
use super::fixed::FixedParseError;
use super::period::Period;
use super::quantity::{Quantity, Unit};
use super::statistic::{Basis, LeaseProfile, Measure, Methodology, SampleCount, Statistic};
use super::stratum::{Dimension, Population, Stratum};

/// A footnote marker as it appears attached to a header or row label.
///
/// # Why plain trailing digits are not markers
///
/// The obvious implementation treats a trailing digit as a footnote reference —
/// and it is wrong. HDB column headers are *years*: `"2023"` would be read as the
/// text `"202"` with footnote `"3"`. Since a misread period silently reassigns a
/// number to the wrong slice of market history, only unambiguous markers are
/// recognised: superscript digits, `*`, `†`, `‡`, and parenthesised letters.
///
/// A publication using bare `1` as a footnote marker will therefore be *missed*
/// here rather than mangled. That is the right trade: the failure is a footnote we
/// did not attach, not a year we invented.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FootnoteMarker(String);

impl FootnoteMarker {
    pub fn new(marker: impl Into<String>) -> Self {
        Self(marker.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Splits a trailing footnote marker off a header or label.
    ///
    /// Returns the text with the marker removed, and the marker if one was found.
    fn split(text: &str) -> (String, Option<FootnoteMarker>) {
        const SUPERSCRIPTS: &[char] = &[
            '\u{00b9}', '\u{00b2}', '\u{00b3}', '\u{2074}', '\u{2075}', '\u{2076}', '\u{2077}',
            '\u{2078}', '\u{2079}', '\u{2070}',
        ];
        const SYMBOLS: &[char] = &['*', '\u{2020}', '\u{2021}', '\u{00a7}', '#'];

        let trimmed = text.trim();

        // Parenthesised letter: `Price (a)`.
        if let Some(open) = trimmed.rfind('(')
            && let Some(close) = trimmed[open + 1..].find(')')
        {
            let inner = &trimmed[open + 1..];
            let body = &inner[..close];
            let tail = inner[close + 1..].trim();
            if tail.is_empty()
                && body.len() == 1
                && body.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            {
                return (
                    trimmed[..open].trim().to_string(),
                    Some(FootnoteMarker::new(body)),
                );
            }
        }

        // Trailing superscripts or symbols, possibly several (`Price*†`).
        let marker: String = trimmed
            .chars()
            .rev()
            .take_while(|c| SUPERSCRIPTS.contains(c) || SYMBOLS.contains(c))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        if marker.is_empty() {
            return (trimmed.to_string(), None);
        }

        let text = trimmed[..trimmed.len() - marker.len()].trim().to_string();
        (text, Some(FootnoteMarker::new(marker)))
    }
}

impl fmt::Display for FootnoteMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the caller must declare about a table before it can be read.
///
/// A PDF table does not state, machine-readably, which measure it reports, over
/// what population, under which methodology, or from which document. Those facts
/// exist only in the surrounding prose and in the reader's head. Rather than
/// *infer* them — which would mean inventing provenance, the one thing this module
/// exists to prevent — the caller declares them here, once, and ingestion checks
/// everything it can against the table itself.
///
/// The shape assumed is the common one: **rows are strata, columns are periods.**
/// A table of median prices with towns down the side and years across the top.
#[derive(Debug, Clone)]
pub struct TableSpec {
    measure_name: String,
    unit: Unit,
    population: Population,
    methodology: Methodology,
    citation: Citation,
    row_dimension: Dimension,
    label_column: usize,
    observations_column: Option<usize>,
    base_stratum: Stratum,
    lease_profile: LeaseProfile,
    index_base: Option<Period>,
    footnotes: BTreeMap<FootnoteMarker, String>,
}

impl TableSpec {
    /// Declares a table whose rows are levels of `row_dimension` and whose
    /// remaining columns are headed by periods.
    pub fn new(
        measure_name: impl Into<String>,
        unit: Unit,
        population: Population,
        row_dimension: Dimension,
        methodology: Methodology,
        citation: Citation,
    ) -> Self {
        Self {
            measure_name: measure_name.into(),
            unit,
            population,
            methodology,
            citation,
            row_dimension,
            label_column: 0,
            observations_column: None,
            base_stratum: Stratum::all(),
            lease_profile: LeaseProfile::Unstated,
            index_base: None,
            footnotes: BTreeMap::new(),
        }
    }

    /// Which column holds the row label. Defaults to the first.
    pub fn label_column(mut self, column: usize) -> Self {
        self.label_column = column;
        self
    }

    /// A column holding the number of observations behind each row.
    ///
    /// Supply this whenever the publication offers it. Without it every statistic
    /// from the table is [`Basis::Unstated`], which is honest but weak — and for a
    /// price table it means no small-sample warning can ever fire, which is the
    /// warning that matters most.
    pub fn observations_column(mut self, column: usize) -> Self {
        self.observations_column = Some(column);
        self
    }

    /// Constraints that apply to every cell — e.g. the whole table is 4-room
    /// flats.
    pub fn base_stratum(mut self, stratum: Stratum) -> Self {
        self.base_stratum = stratum;
        self
    }

    /// Where the flats behind this table sit on the lease-decay curve.
    pub fn lease_profile(mut self, profile: LeaseProfile) -> Self {
        self.lease_profile = profile;
        self
    }

    /// The base period, required when the table holds index readings.
    pub fn index_base(mut self, base: Period) -> Self {
        self.index_base = Some(base);
        self
    }

    /// Supplies the body of a footnote whose marker appears in the table.
    ///
    /// Ingestion **drops** any column or row carrying a marker with no footnote
    /// supplied here. See the module docs.
    pub fn footnote(mut self, marker: impl Into<String>, body: impl Into<String>) -> Self {
        self.footnotes
            .insert(FootnoteMarker::new(marker), body.into());
        self
    }
}

/// Something ingestion could not do, reported rather than guessed around.
///
/// Every issue corresponds to data that was **dropped**. Read them.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IngestIssue {
    #[error(
        "column {column} is headed `{header}`, which is not a period; \
         the column was dropped rather than assigned to a guessed period"
    )]
    UnparseableColumnHeader { column: usize, header: String },

    #[error(
        "column {column} (`{header}`) carries footnote marker `{marker}` but no footnote \
         body was supplied for it; the column was DROPPED because a footnote is exactly \
         what tells you a column means something other than what it says"
    )]
    UnresolvedFootnote {
        column: usize,
        header: String,
        marker: FootnoteMarker,
    },

    #[error(
        "row {row} (`{label}`) carries footnote marker `{marker}` with no footnote body; \
         the row was dropped"
    )]
    UnresolvedRowFootnote {
        row: usize,
        label: String,
        marker: FootnoteMarker,
    },

    #[error(
        "column {column} is headed `{header}`, declaring unit `{found}`, but the spec \
         declares `{declared}`; the ENTIRE TABLE was rejected — a unit mismatch means the \
         column was misidentified, and the numbers would be wrong by orders of magnitude"
    )]
    UnitMismatch {
        column: usize,
        header: String,
        found: Unit,
        declared: Unit,
    },

    #[error(
        "the header row parses as data (`{header}` in column {column} is a number); this is \
         almost certainly the continuation of a table split across a page break, whose first \
         data row was promoted to the header. The ENTIRE TABLE was rejected rather than \
         reading a value as a column name"
    )]
    SuspectedContinuationTable { column: usize, header: String },

    #[error("row {row} has {found} cells but the header has {expected}; the row was dropped")]
    RaggedRow {
        row: usize,
        found: usize,
        expected: usize,
    },

    #[error("row {row}, column {column}: {source}; the cell was dropped")]
    UnparseableCell {
        row: usize,
        column: usize,
        #[source]
        source: FixedParseError,
    },

    #[error(
        "row {row}, column {column} is a suppression marker, not a value; the cell was \
         dropped and was NOT read as zero"
    )]
    SuppressedCell { row: usize, column: usize },

    #[error(
        "row {row}: the observation count in column {column} could not be read ({source}); \
         the row's statistics were emitted with an UNSTATED basis, so no small-sample \
         warning can fire for them"
    )]
    UnreadableObservationCount {
        row: usize,
        column: usize,
        #[source]
        source: FixedParseError,
    },

    #[error("the table has no rows")]
    EmptyTable,

    #[error(
        "the spec declares an index measure but no index base period; an index reading \
         without its base is not a value. The entire table was rejected"
    )]
    MissingIndexBase,
}

/// The outcome of reading a table: what was recovered, and what was not.
///
/// **`issues` is not a log to be ignored.** Every entry is data that was dropped.
/// A caller that reads `statistics` and discards `issues` has silently lost part
/// of the table, which is exactly the failure mode this type exists to prevent.
#[derive(Debug, Clone, Default)]
pub struct Ingested {
    pub statistics: Vec<Statistic>,
    pub issues: Vec<IngestIssue>,
}

impl Ingested {
    /// Whether the table was read completely, with nothing dropped.
    pub fn is_complete(&self) -> bool {
        self.issues.is_empty()
    }
}

/// The parts of a header cell: its text, its unit marker, and its footnote.
struct HeaderParts {
    text: String,
    unit: Option<Unit>,
    footnote: Option<FootnoteMarker>,
}

/// Pulls a header cell apart into text, unit and footnote marker.
///
/// `"2023 ($)¹"` becomes text `"2023"`, unit `Sgd`, footnote `"¹"`.
fn parse_header(raw: &str) -> HeaderParts {
    // The footnote marker sits outermost, after any unit parenthetical.
    let (without_footnote, footnote) = FootnoteMarker::split(raw);

    // A trailing parenthetical that is a *unit*, e.g. `Median Price ($)`.
    let mut text = without_footnote.clone();
    let mut unit = None;
    if let Some(open) = without_footnote.rfind('(')
        && without_footnote.trim_end().ends_with(')')
        && let Some(found) = Unit::parse_marker(&without_footnote[open..])
    {
        unit = Some(found);
        text = without_footnote[..open].trim().to_string();
    }

    HeaderParts {
        text: text.trim().to_string(),
        unit,
        footnote,
    }
}

/// Reads a reconstructed table into statistics, under a declared [`TableSpec`].
///
/// Never panics, never guesses, and never emits a number it could not fully
/// resolve. See the module docs for what it can and cannot recover.
pub fn ingest_table(table: &Table, spec: &TableSpec) -> Ingested {
    let mut out = Ingested::default();

    if table.rows.is_empty() {
        out.issues.push(IngestIssue::EmptyTable);
        return out;
    }

    if spec.unit == Unit::IndexPoints && spec.index_base.is_none() {
        out.issues.push(IngestIssue::MissingIndexBase);
        return out;
    }

    let column_count = table.headers.len();
    let kind = spec.unit.kind();

    // --- Whole-table aborts ------------------------------------------------
    //
    // Two conditions mean the table is not the table we think it is. Both reject
    // everything rather than emitting a subset that looks complete.

    for (column, header) in table.headers.iter().enumerate() {
        if column == spec.label_column || Some(column) == spec.observations_column {
            continue;
        }
        let parts = parse_header(header);

        // A header that is a bare number is the signature of a continuation
        // table: the page broke, and the first data row got promoted to headers.
        // Reading `"541,000"` as a column name would be absurd; reading it as a
        // period would be worse.
        //
        // Note this uses the module's own fixed-point parser rather than
        // `str::parse::<f64>()`. A published figure is written `"111,111"`, and
        // `f64` parsing rejects the thousands separator — so the naive check
        // silently fails to fire on exactly the format HDB actually prints.
        let looks_like_a_value = super::fixed::parse_fixed(&parts.text, 2).is_ok();
        if looks_like_a_value && Period::parse(&parts.text).is_err() {
            out.issues.push(IngestIssue::SuspectedContinuationTable {
                column,
                header: header.clone(),
            });
            return out;
        }

        // A declared unit that contradicts the header is a misidentified column.
        if let Some(found) = parts.unit
            && found != spec.unit
        {
            out.issues.push(IngestIssue::UnitMismatch {
                column,
                header: header.clone(),
                found,
                declared: spec.unit,
            });
            return out;
        }
    }

    // --- Resolve the value columns -----------------------------------------

    struct ValueColumn {
        index: usize,
        period: Period,
        measure: Measure,
    }

    let mut value_columns: Vec<ValueColumn> = Vec::new();

    for (column, header) in table.headers.iter().enumerate() {
        if column == spec.label_column || Some(column) == spec.observations_column {
            continue;
        }
        let parts = parse_header(header);

        // A footnote on a column header may redefine the measure. If we cannot
        // resolve it, we do not know what the column means, so we drop it.
        let definition = match &parts.footnote {
            Some(marker) => match spec.footnotes.get(marker) {
                Some(body) => Some(body.clone()),
                None => {
                    out.issues.push(IngestIssue::UnresolvedFootnote {
                        column,
                        header: header.clone(),
                        marker: marker.clone(),
                    });
                    continue;
                }
            },
            None => None,
        };

        let period = match Period::parse(&parts.text) {
            Ok(period) => period,
            Err(_) => {
                out.issues.push(IngestIssue::UnparseableColumnHeader {
                    column,
                    header: header.clone(),
                });
                continue;
            }
        };

        let mut measure = Measure::new(&spec.measure_name, spec.unit);
        if let Some(definition) = definition {
            // The footnote travels into the measure's identity, which is what
            // makes the series break-detector catch a redefinition later.
            measure = measure.with_definition(definition);
        }

        value_columns.push(ValueColumn {
            index: column,
            period,
            measure,
        });
    }

    // --- Read the rows -----------------------------------------------------

    for (row_index, row) in table.rows.iter().enumerate() {
        if row.len() != column_count {
            out.issues.push(IngestIssue::RaggedRow {
                row: row_index,
                found: row.len(),
                expected: column_count,
            });
            continue;
        }

        let raw_label = &row[spec.label_column];
        let (label, row_footnote) = FootnoteMarker::split(raw_label);

        // A footnote on the row label can redefine the stratum ("excludes
        // executive maisonettes"), so the same rule applies as for headers.
        if let Some(marker) = row_footnote
            && !spec.footnotes.contains_key(&marker)
        {
            out.issues.push(IngestIssue::UnresolvedRowFootnote {
                row: row_index,
                label: label.clone(),
                marker,
            });
            continue;
        }

        let stratum = spec
            .base_stratum
            .clone()
            .with(spec.row_dimension.clone(), &label);

        // The observation count backing this row, if the table offers one.
        let basis = match spec.observations_column {
            Some(column) => match SampleCount::parse_cell(&row[column]) {
                Ok(count) => Basis::Census {
                    observations: count,
                },
                Err(source) => {
                    out.issues.push(IngestIssue::UnreadableObservationCount {
                        row: row_index,
                        column,
                        source,
                    });
                    Basis::Unstated
                }
            },
            None => Basis::Unstated,
        };

        for value_column in &value_columns {
            let cell = &row[value_column.index];
            let quantity = match Quantity::parse(cell, kind, spec.index_base) {
                Ok(quantity) => quantity,
                Err(FixedParseError::Suppressed { .. }) => {
                    // A suppressed cell is the publication declining to tell us.
                    // It is emphatically not a zero.
                    out.issues.push(IngestIssue::SuppressedCell {
                        row: row_index,
                        column: value_column.index,
                    });
                    continue;
                }
                Err(source) => {
                    out.issues.push(IngestIssue::UnparseableCell {
                        row: row_index,
                        column: value_column.index,
                        source,
                    });
                    continue;
                }
            };

            match Statistic::new(
                value_column.measure.clone(),
                quantity,
                spec.population.clone(),
                stratum.clone(),
                value_column.period,
                basis.clone(),
                spec.lease_profile.clone(),
                spec.methodology.clone(),
                spec.citation.clone(),
            ) {
                Ok(statistic) => out.statistics.push(statistic),
                Err(_) => {
                    // Unreachable in practice: `kind` is derived from the same
                    // unit the measure was built from. Kept as a drop rather than
                    // an unwrap so a future refactor cannot turn it into a panic.
                    out.issues.push(IngestIssue::UnparseableCell {
                        row: row_index,
                        column: value_column.index,
                        source: FixedParseError::NotANumber {
                            input: cell.clone(),
                            found: '?',
                        },
                    });
                }
            }
        }
    }

    out
}

impl SampleCount {
    /// Reads an observation count from a table cell.
    fn parse_cell(cell: &str) -> Result<Self, FixedParseError> {
        let value = super::fixed::parse_fixed(cell, 0)?;
        u32::try_from(value)
            .map(SampleCount::new)
            .map_err(|_| FixedParseError::OutOfRange {
                input: cell.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdb::survey::citation::Locator;
    use crate::hdb::survey::quantity::SgdAmount;
    use crate::hdb::survey::statistic::Reliability;

    fn synthetic_citation() -> Citation {
        Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("SYNTHETIC-1".into()),
            Period::Year(2024),
        )
    }

    fn spec() -> TableSpec {
        TableSpec::new(
            "Median resale price",
            Unit::Sgd,
            Population::ResaleTransactions,
            Dimension::Town,
            Methodology::new("SYNTHETIC METHODOLOGY A"),
            synthetic_citation(),
        )
    }

    /// A synthetic table in the shape HDB publishes: towns down the side, years
    /// across the top, a unit marker in the headers. Values are deliberately
    /// implausible repdigits — this is NOT real HDB data.
    fn synthetic_price_table() -> Table {
        Table {
            headers: vec![
                "Town".into(),
                "2022 ($)".into(),
                "2023 ($)".into(),
            ],
            rows: vec![
                vec!["TAMPINES".into(), "111,111".into(), "222,222".into()],
                vec!["QUEENSTOWN".into(), "333,333".into(), "444,444".into()],
            ],
        }
    }

    #[test]
    fn a_well_formed_table_yields_statistics_with_periods_units_and_strata() {
        let ingested = ingest_table(&synthetic_price_table(), &spec());
        assert!(
            ingested.is_complete(),
            "unexpected issues: {:?}",
            ingested.issues
        );
        assert_eq!(ingested.statistics.len(), 4);

        let tampines_2023 = ingested
            .statistics
            .iter()
            .find(|s| {
                s.period() == Period::Year(2023)
                    && s.stratum().level(&Dimension::Town).unwrap().as_str() == "TAMPINES"
            })
            .expect("Tampines 2023 must have been read");

        // The unit in the header ($) was recognised, and the cell parsed exactly.
        assert_eq!(
            tampines_2023.quantity(),
            Quantity::Money(SgdAmount::from_dollars(222_222))
        );
        // The citation travelled all the way through.
        assert_eq!(
            tampines_2023.citation().publication(),
            "SYNTHETIC FIXTURE — NOT HDB DATA"
        );
    }

    #[test]
    fn a_footnote_in_the_header_is_not_silently_dropped() {
        // THE test the brief asks for. A header footnote redefines the column.
        let table = Table {
            headers: vec!["Town".into(), "2023 ($)\u{00b9}".into()],
            rows: vec![vec!["TAMPINES".into(), "111,111".into()]],
        };

        // Case 1: the footnote body is supplied. It must reach the measure's
        // definition, because that is what makes a later series join notice that
        // this column is not the same measure as an unfootnoted one.
        let resolved = ingest_table(
            &table,
            &spec().footnote("\u{00b9}", "Prices are before grants"),
        );
        assert!(resolved.is_complete(), "{:?}", resolved.issues);
        assert_eq!(
            resolved.statistics[0].measure().definition(),
            Some("Prices are before grants")
        );

        // Case 2: the footnote body is NOT supplied. The marker is visible in the
        // header, so we know the column means something other than what it says —
        // and we do not know what. The column is dropped, loudly.
        let unresolved = ingest_table(&table, &spec());
        assert!(unresolved.statistics.is_empty());
        assert!(matches!(
            unresolved.issues[0],
            IngestIssue::UnresolvedFootnote { .. }
        ));
    }

    #[test]
    fn a_unit_mismatch_rejects_the_whole_table() {
        // The header says percent; the spec says dollars. One of them is wrong,
        // and emitting numbers under either reading would be wrong by orders of
        // magnitude. Reject everything.
        let table = Table {
            headers: vec!["Town".into(), "2023 (%)".into()],
            rows: vec![vec!["TAMPINES".into(), "87.3".into()]],
        };
        let ingested = ingest_table(&table, &spec());
        assert!(ingested.statistics.is_empty());
        assert!(matches!(
            ingested.issues[0],
            IngestIssue::UnitMismatch {
                found: Unit::Percent,
                declared: Unit::Sgd,
                ..
            }
        ));
    }

    #[test]
    fn a_page_break_continuation_is_detected_and_refused() {
        // kopitiam-document splits a table across a page break into two Tables,
        // and the second one's first DATA row becomes its header. Reading
        // "111,111" as a column name — or worse, as a period — must not happen.
        let continuation = Table {
            headers: vec!["TAMPINES".into(), "111,111".into(), "222,222".into()],
            rows: vec![vec!["QUEENSTOWN".into(), "333,333".into(), "444,444".into()]],
        };
        let ingested = ingest_table(&continuation, &spec());
        assert!(ingested.statistics.is_empty());
        assert!(matches!(
            ingested.issues[0],
            IngestIssue::SuspectedContinuationTable { .. }
        ));
    }

    #[test]
    fn a_suppressed_cell_is_dropped_and_never_read_as_zero() {
        // A statistics agency printing `-` means "we are not telling you". If that
        // became $0, a buyer would see the cheapest town in Singapore.
        let table = Table {
            headers: vec!["Town".into(), "2023 ($)".into()],
            rows: vec![
                vec!["TAMPINES".into(), "111,111".into()],
                vec!["QUIET TOWN".into(), "-".into()],
            ],
        };
        let ingested = ingest_table(&table, &spec());
        assert_eq!(ingested.statistics.len(), 1);
        assert!(matches!(
            ingested.issues[0],
            IngestIssue::SuppressedCell { row: 1, .. }
        ));
        // Nothing in the output claims a price of zero.
        assert!(!ingested
            .statistics
            .iter()
            .any(|s| s.quantity() == Quantity::Money(SgdAmount::from_cents(0))));
    }

    #[test]
    fn an_observations_column_drives_the_small_sample_warning() {
        let table = Table {
            headers: vec![
                "Town".into(),
                "2023 ($)".into(),
                "No. of transactions".into(),
            ],
            rows: vec![
                vec!["BUSY TOWN".into(), "111,111".into(), "450".into()],
                vec!["QUIET TOWN".into(), "222,222".into(), "3".into()],
            ],
        };
        let ingested = ingest_table(&table, &spec().observations_column(2));
        assert!(ingested.is_complete(), "{:?}", ingested.issues);
        assert_eq!(ingested.statistics.len(), 2);

        let quiet = ingested
            .statistics
            .iter()
            .find(|s| s.stratum().level(&Dimension::Town).unwrap().as_str() == "QUIET TOWN")
            .unwrap();
        // Three transactions. Someone is about to buy a flat against this number.
        assert!(matches!(
            quiet.reliability(),
            Reliability::LowPrecision { .. }
        ));
        assert!(quiet.to_string().contains("LOW PRECISION"));

        let busy = ingested
            .statistics
            .iter()
            .find(|s| s.stratum().level(&Dimension::Town).unwrap().as_str() == "BUSY TOWN")
            .unwrap();
        assert!(matches!(busy.reliability(), Reliability::Adequate { .. }));
    }

    #[test]
    fn an_unparseable_column_header_drops_that_column_only() {
        let table = Table {
            headers: vec![
                "Town".into(),
                "2023 ($)".into(),
                "Change from last year".into(),
            ],
            rows: vec![vec!["TAMPINES".into(), "111,111".into(), "5.2".into()]],
        };
        let ingested = ingest_table(&table, &spec());
        // The 2023 column survives; the un-periodable one is dropped with an issue.
        assert_eq!(ingested.statistics.len(), 1);
        assert_eq!(ingested.statistics[0].period(), Period::Year(2023));
        assert!(matches!(
            ingested.issues[0],
            IngestIssue::UnparseableColumnHeader { column: 2, .. }
        ));
    }

    #[test]
    fn header_parsing_separates_period_unit_and_footnote() {
        let parts = parse_header("2023 ($)\u{00b9}");
        assert_eq!(parts.text, "2023");
        assert_eq!(parts.unit, Some(Unit::Sgd));
        assert_eq!(parts.footnote, Some(FootnoteMarker::new("\u{00b9}")));
    }

    #[test]
    fn a_year_is_not_mistaken_for_a_footnote_marker() {
        // The bug this guards against: reading "2023" as text "202" + footnote "3".
        let (text, marker) = FootnoteMarker::split("2023");
        assert_eq!(text, "2023");
        assert_eq!(marker, None);
        assert_eq!(parse_header("2023").text, "2023");
    }

    #[test]
    fn an_index_table_without_a_base_is_rejected_entirely() {
        let index_spec = TableSpec::new(
            "Resale Price Index",
            Unit::IndexPoints,
            Population::ResaleTransactions,
            Dimension::Town,
            Methodology::new("SYNTHETIC METHODOLOGY A"),
            synthetic_citation(),
        );
        let table = Table {
            headers: vec!["Town".into(), "2023".into()],
            rows: vec![vec!["TAMPINES".into(), "111.1".into()]],
        };
        let ingested = ingest_table(&table, &index_spec);
        assert!(ingested.statistics.is_empty());
        assert!(matches!(ingested.issues[0], IngestIssue::MissingIndexBase));
    }

    #[test]
    fn a_ragged_row_is_dropped_rather_than_misaligned() {
        let table = Table {
            headers: vec!["Town".into(), "2022 ($)".into(), "2023 ($)".into()],
            rows: vec![
                vec!["TAMPINES".into(), "111,111".into(), "222,222".into()],
                // A short row: if we zipped it against the headers, the 2022 value
                // would silently become the 2023 value.
                vec!["QUEENSTOWN".into(), "333,333".into()],
            ],
        };
        let ingested = ingest_table(&table, &spec());
        assert_eq!(ingested.statistics.len(), 2);
        assert!(matches!(
            ingested.issues[0],
            IngestIssue::RaggedRow {
                row: 1,
                found: 2,
                expected: 3
            }
        ));
    }

    #[test]
    fn a_base_stratum_applies_to_every_row() {
        // The whole table is 4-room flats; the rows vary only by town.
        let ingested = ingest_table(
            &synthetic_price_table(),
            &spec().base_stratum(Stratum::all().with(Dimension::FlatType, "4-Room")),
        );
        assert!(ingested.is_complete());
        for statistic in &ingested.statistics {
            assert_eq!(
                statistic
                    .stratum()
                    .level(&Dimension::FlatType)
                    .unwrap()
                    .as_str(),
                "4-ROOM"
            );
        }
    }
}
