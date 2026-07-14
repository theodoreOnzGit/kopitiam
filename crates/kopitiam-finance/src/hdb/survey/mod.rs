//! Surveying the HDB resale **market** — the price data behind a flat purchase.
//!
//! # What "HDB survey" means here, and what it was first taken to mean
//!
//! This history is preserved deliberately, because the correction is the kind of
//! thing that is expensive to rediscover. See
//! `docs/ai-decisions/AID-0010-what-hdb-survey-means.md`.
//!
//! **First reading (wrong):** "HDB survey" was initially read as HDB's *Sample
//! Household Survey* — the periodic demographic study of HDB residents
//! (household composition, commuting, dwelling satisfaction). That is a real HDB
//! publication and a defensible reading of the words.
//!
//! **Corrected by the maintainer:** *"HDB is for buying house."* The purpose is
//! to **survey the resale market in support of a purchase decision** — what
//! flats actually transact for, where, at what age of lease, and how that is
//! moving. Not academic demography. This module is built for the buyer's
//! question, and the demographic reading was discarded.
//!
//! **Still not this:** a *building* or *land* survey — a surveyor's physical
//! measurement of a block (settlement, spalling, cadastral boundaries). Nothing
//! here would serve that, and it would not belong in a finance crate.
//!
//! # The buyer's question
//!
//! > "What can I get for my money, in this town, at this flat type — and how
//! > confident should I be in that number?"
//!
//! The data that answers it:
//!
//! * **Resale transactions** — what flats *actually sold for*, by town, flat
//!   type, storey range, floor area, **remaining lease**, and month. This is the
//!   core dataset; everything else is derived from or contextualises it.
//! * **Resale Price Index (RPI)** — the market trend, quarter by quarter.
//! * **Price distributions** — the median and the spread within a town and flat
//!   type. A median alone hides the range a buyer will actually face.
//! * **Lease decay** — remaining lease is a first-class variable in Singapore,
//!   not a footnote. It moves price, and it constrains how much CPF may be used.
//! * **Supply** — BTO launches, flat supply, waiting times: what is actually
//!   available, not just what things cost.
//!
//! # Why this module is so unwilling to give you a number
//!
//! Someone is about to make the largest purchase of their life against these
//! figures. Statistics mislead more easily than prose: a policy clause quoted out
//! of context still *looks* like a quotation, but a number quoted out of context
//! looks like a fact.
//!
//! > **A statistic without its population, period, backing observation count and
//! > source citation is not a fact — it is a rumour.**
//!
//! "The median 4-room in Punggol is $X" is a meaningless sentence. Over what
//! period? At what remaining lease? Across how many transactions — thirty, or
//! three? This module makes those questions unavoidable by putting them in the
//! *type system*, not in a comment or a review checklist:
//!
//! * [`Statistic`] has private fields and exactly one constructor, taking every
//!   one of those components. No `Default`, no builder that can finish early, no
//!   public field to leave unset. An unprovenanced statistic is not
//!   *expressible* in these types.
//! * Values are not interchangeable. A [`SampleCount`], a [`Percentage`], an
//!   [`SgdAmount`], a [`LeaseRemaining`] and an [`IndexPoint`] are different
//!   types. You cannot add a transaction count to a price, and you cannot compare
//!   index points measured against different bases.
//! * Comparing strata that are not legitimately comparable — two towns that also
//!   differ in flat type, or two prices at very different lease ages — is a typed
//!   [`Incomparability`] error, not a silent number. This is the central risk of
//!   the whole module.
//! * A median backed by very few transactions is flagged as
//!   [`Reliability::LowPrecision`]; see [`SMALL_SAMPLE_THRESHOLD`]. It is never
//!   presented with the same confidence as a well-populated cell.
//! * Joining a time series across a methodology change is *refused* unless the
//!   caller supplies a written [`AcknowledgedBreak`].
//! * A query with no matching data returns [`NoData`] — with an explanation and
//!   the periods that *do* exist — never an interpolated guess.
//!
//! **A confidently wrong price is worse than no price.** Every trade-off here
//! resolves in favour of refusing to answer.
//!
//! # The seam with CPF and HDB policy
//!
//! This module owns **market data — what flats cost**. It does *not* own
//! eligibility, grants, CPF withdrawal limits, or affordability. Those are rules,
//! and they live in [`crate::cpf`] and [`super::policy`], written alongside this.
//!
//! The affordability question ("can *this buyer* afford *that flat*?") is a join
//! of the two, and belongs above both. What this module exposes for that join:
//!
//! * [`Statistic::quantity`] yielding [`Quantity::Money`] — the price, in exact
//!   cents, with its citation and observation count attached.
//! * [`Quantity::Lease`] / [`LeaseRemaining`] — remaining lease in exact months.
//!   This is the field CPF usage rules key off (a lease that does not cover the
//!   youngest buyer to age 95 restricts CPF use), so it is exposed as a first-
//!   class typed quantity rather than buried in a stratum label.
//! * [`Stratum`] — the (town, flat type, storey, area, lease band) coordinates a
//!   policy rule can match against.
//! * [`SurveyStore::query`] — the lookup a policy or affordability layer calls to
//!   get a price *with* its caveats, or an honest [`NoData`].
//!
//! Any layer joining these must carry the [`Reliability`] and [`Caveat`] values
//! through to the user. An affordability figure computed from a three-transaction
//! median is a three-transaction affordability figure, and must say so.
//!
//! # There is no `f64` in this module's value model
//!
//! Published statistics are *decimal* quantities: a median price is `$540,000`,
//! the Resale Price Index is `195.5`, a remaining lease is `64 years 3 months`.
//! These numbers were published exactly. Binary floating point would introduce
//! drift into values that have no error bar of their own, and would forfeit `Eq`,
//! `Hash` and `Ord` — which this module needs for deterministic behaviour
//! (CLAUDE.md, Engineering Principles).
//!
//! Every quantity is a fixed-point integer with a documented scale: money in
//! cents, percentages in hundredths of a percent, index points in thousandths,
//! lease in months. Parsing *rejects* input more precise than the scale rather
//! than silently rounding it away — see [`fixed`](self::fixed).
//!
//! # What this module does not do
//!
//! It reports **what was published**. It does not forecast, model, interpolate,
//! extrapolate, seasonally adjust, or reweight. It will not tell a buyer what a
//! flat will be worth in 2030, and it will not invent a median for a town-month
//! that HDB never published. If you ask for a number that does not exist, the
//! answer is "that was not published" — and that *is* the correct answer.
//!
//! # Provenance status: no real HDB data has been ingested
//!
//! No HDB publication or resale dataset was present on the machine this module
//! was written on, and the platform has no network access by design (CLAUDE.md,
//! Offline First). The machinery here is exercised entirely against **synthetic**
//! tables constructed in the tests, whose values are deliberately implausible
//! repdigits (`1111`, `2222`) and whose citations carry `SYNTHETIC FIXTURE` in
//! the publication title — so that no fixture can ever be mistaken for a real
//! Singapore price. Fabricated price data inside a system that looks
//! authoritative is the worst thing this module could possibly produce, and it
//! has been avoided deliberately rather than accidentally.
//!
//! Wiring in the real feeds (HDB's resale price statistics, and the resale
//! transaction dataset published on data.gov.sg) is the obvious next step, and is
//! what [`TableSpec`] and [`ingest_table`] exist to make safe.
//!
//! # Pipeline
//!
//! ```text
//!   HDB PDF / published table
//!     -> kopitiam_pdf::Page          (glyphs + geometry)
//!     -> kopitiam_document::Document (reconstructed Blocks, incl. Table)
//!     -> ingest::ingest_table        (+ a caller-supplied TableSpec)
//!     -> Vec<Statistic> + Vec<IngestIssue>
//!     -> SurveyStore (query)  /  facts::to_graph (ontology)
//! ```
//!
//! [`ingest`] is deliberately **not** a guesser. It parses what can be parsed
//! deterministically (numbers, units and periods in headers, footnote markers)
//! and requires the caller to declare, in a [`TableSpec`], what a PDF table does
//! not state machine-readably: which measure it reports, over which population,
//! under which methodology, and which document it came from. Where a cell cannot
//! be fully resolved it emits an [`IngestIssue`] and **drops the cell** rather
//! than emitting a plausible-looking wrong number.

mod citation;
mod facts;
mod fixed;
mod ingest;
mod period;
mod quantity;
mod query;
mod series;
mod statistic;
mod stratum;

pub use citation::{Citation, Locator};
pub use facts::{FACT_SOURCE, fact_needs_warning, to_fact, to_graph, to_graph_batch};
pub use fixed::FixedParseError;
pub use ingest::{FootnoteMarker, IngestIssue, Ingested, TableSpec, ingest_table};
pub use period::{Period, PeriodError, Quarter};
pub use quantity::{
    FloorArea, IndexPoint, LeaseRemaining, Percentage, Quantity, QuantityKind, RebasedIndex,
    SgdAmount, Unit, UnitCount,
};
pub use query::{NoData, Query, QueryOutcome, SurveyStore};
pub use series::{AcknowledgedBreak, MethodologyBreak, Series, SeriesError};
pub use statistic::{
    Basis, Caveat, Comparison, Incomparability, LeaseProfile, Measure, Methodology, Reliability,
    SMALL_SAMPLE_THRESHOLD, SampleCount, Statistic, StatisticError,
};
pub use stratum::{Dimension, LevelValue, Population, Stratum};
