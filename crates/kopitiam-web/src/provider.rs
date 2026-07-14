use std::collections::HashMap;
use std::sync::Arc;

use crate::clock::{Clock, SystemClock};
use crate::error::SearchError;
use crate::query::SearchQuery;
use crate::response::SearchResponse;

/// A pluggable connection to one search engine.
///
/// This is the entire surface the rest of KOPITIAM is allowed to see of the
/// web. It mirrors `kopitiam-ai`'s `ModelAdapter` deliberately, and for the
/// same reason: the platform must never be written against one vendor, and must
/// remain fully functional when that vendor is absent, unaffordable, or
/// unreachable.
///
/// # The contract
///
/// Implementations must obey three rules. They are not stylistic.
///
/// 1. **Never fabricate.** If you did not retrieve it, do not return it. A
///    provider that cannot search returns `Err`; it does not return an empty
///    [`SearchResponse`], and it certainly does not return a plausible-looking
///    one. A caller cannot tell a fabricated result from a real one, and
///    neither can the knowledge graph it ends up in.
/// 2. **Fail honestly and specifically.** Return the [`SearchError`] variant
///    that says what actually went wrong, so the caller can decide whether to
///    retry, to tell the human to set an API key, or to carry on without the
///    web. This is the Offline First ladder in action: running out of tokens,
///    quota or network must never stop productive scientific work.
/// 3. **Stamp provenance.** Build responses through
///    [`SearchResponse::builder`], which makes this automatic.
///
/// # Where this sits
///
/// Below every other source of knowledge KOPITIAM has. See the crate docs. A
/// provider that is *cheap and convenient to call* would be an
/// anti-feature — the friction is the design.
pub trait SearchProvider {
    /// Stable identifier for this provider (`"brave"`, `"searxng"`, `"null"`).
    ///
    /// It is recorded in every result's provenance and forms part of the cache
    /// key, so **changing it invalidates recorded sessions and rewrites
    /// history**. Treat it as a wire format.
    fn name(&self) -> &str;

    /// Performs the search.
    ///
    /// `Ok(response)` means an engine was asked and answered — even if the
    /// answer was "nothing". `Err` means it was not asked, or its answer could
    /// not be understood. Nothing about the world may be inferred from `Err`.
    fn search(&self, query: &SearchQuery) -> Result<SearchResponse, SearchError>;
}

/// The provider for a KOPITIAM that does not search the web — which is the
/// default, and should stay the default.
///
/// Every call returns [`SearchError::Disabled`].
///
/// # Why an error, and not an empty result set
///
/// This is the single most important design decision in the crate, so it is
/// worth being blunt about it.
///
/// The lazy implementation of "web search is off" is to return zero results.
/// It compiles, it never fails, and every caller handles it without a special
/// case. It is also a lie. A caller handed zero results concludes *there is
/// nothing about this on the web* — and then writes that conclusion into a
/// document, or hands it to a model as context, or records it in the knowledge
/// graph as a dated finding. The absence of a search has been silently
/// laundered into evidence of absence.
///
/// So `NullProvider` refuses. "I did not look" is a different sentence from
/// "there is nothing there", and this crate will not let the two be confused —
/// least of all by the code path that runs for almost every user.
///
/// The cost is that callers must handle an error they might have preferred to
/// ignore. That cost is the feature.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullProvider;

impl SearchProvider for NullProvider {
    fn name(&self) -> &str {
        "null"
    }

    fn search(&self, _query: &SearchQuery) -> Result<SearchResponse, SearchError> {
        Err(SearchError::Disabled {
            provider: self.name().to_string(),
        })
    }
}

/// A [`SearchProvider`] backed by canned results and no network whatsoever.
///
/// This is the counterpart of `kopitiam-ai`'s `EchoAdapter`, and it matters for
/// the same reason: it lets every workflow, CLI command and document pipeline
/// that *can* consume web results be exercised deterministically, offline, in
/// CI, on a plane, and long after whichever search vendor we picked has gone out
/// of business.
///
/// It is more important than any real adapter. A real adapter proves KOPITIAM
/// can talk to Brave. This one proves KOPITIAM does not *need* to.
///
/// # Behaviour on an unknown query
///
/// Returns an empty [`SearchResponse`] — "this engine searched and found
/// nothing" — rather than an error. That is the truthful description of a
/// fixture-backed engine whose fixtures do not cover the query, and it gives
/// downstream tests a way to exercise the found-nothing path, which is the path
/// most callers get wrong.
///
/// (Contrast [`NullProvider`], which errors: it did not search at all.)
///
/// # Timestamps
///
/// Stamped from a [`Clock`], which defaults to the real one. Tests that care
/// about provenance should install a [`FixedClock`](crate::FixedClock) or a
/// [`SteppingClock`](crate::SteppingClock) so the timestamps are as
/// reproducible as the results.
pub struct StaticProvider {
    name: String,
    clock: Arc<dyn Clock>,
    /// Keyed on query *text* only, not on the full [`SearchQuery`]. A fixture
    /// is a stand-in for "what the web says about this topic", and making a
    /// test author restate `max_results` to get a hit would be friction with no
    /// truth-value behind it.
    fixtures: HashMap<String, Vec<CannedResult>>,
}

/// One fixture row: exactly the three fields an engine gives us.
type CannedResult = (String, String, String);

impl StaticProvider {
    /// A provider named `name` with no fixtures — every query finds nothing.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            clock: Arc::new(SystemClock),
            fixtures: HashMap::new(),
        }
    }

    /// Uses `clock` to stamp retrieval timestamps.
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Registers what this provider will "find" for `query`, as
    /// `(title, url, snippet)` triples in rank order.
    pub fn with_results<I, T, U, S>(mut self, query: impl Into<String>, results: I) -> Self
    where
        I: IntoIterator<Item = (T, U, S)>,
        T: Into<String>,
        U: Into<String>,
        S: Into<String>,
    {
        self.fixtures.insert(
            query.into(),
            results
                .into_iter()
                .map(|(t, u, s)| (t.into(), u.into(), s.into()))
                .collect(),
        );
        self
    }
}

impl SearchProvider for StaticProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn search(&self, query: &SearchQuery) -> Result<SearchResponse, SearchError> {
        let mut builder = SearchResponse::builder(query, &self.name, self.clock.now());

        if let Some(canned) = self.fixtures.get(query.text()) {
            // Honour max_results, exactly as a real engine would: a fixture
            // that ignored it would let a test pass against behaviour no live
            // provider exhibits.
            for (title, url, snippet) in canned.iter().take(query.max_results()) {
                builder.push(title, url, snippet);
            }
        }

        Ok(builder.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FixedClock;

    #[test]
    fn the_null_provider_refuses_rather_than_reporting_emptiness() {
        let error = NullProvider
            .search(&SearchQuery::new("anything"))
            .expect_err("a disabled provider must not return a response");

        assert!(matches!(error, SearchError::Disabled { .. }));
        assert!(error.is_configuration());
        assert!(!error.is_transient(), "retrying a disabled provider is pointless");
    }

    #[test]
    fn a_disabled_search_and_an_empty_search_are_different_values() {
        // The whole point, asserted directly. These two calls model "we did not
        // look" and "we looked and found nothing", and they must not be
        // confusable by any caller.
        let did_not_look = NullProvider.search(&SearchQuery::new("q"));
        let looked_and_found_nothing = StaticProvider::new("static").search(&SearchQuery::new("q"));

        assert!(did_not_look.is_err());

        let response = looked_and_found_nothing.expect("the static provider did search");
        assert!(response.found_nothing());
        // ... and the second one is still evidence: it is dated and attributed.
        assert_eq!(response.engine(), "static");
        assert_eq!(response.query().text(), "q");
    }

    #[test]
    fn the_static_provider_returns_its_fixtures_deterministically() {
        let clock = Arc::new(FixedClock::from_unix(1_752_486_660));
        let provider = StaticProvider::new("static")
            .with_clock(clock)
            .with_results(
                "write-ahead logging",
                [
                    ("Write-ahead logging", "https://example.org/wal", "A durability technique."),
                    ("Redo logging", "https://example.org/redo", "A related variant."),
                ],
            );

        let first = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        let second = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();

        // Same results, same ranks, same provenance, same timestamps -- twice.
        assert_eq!(first, second);
        assert_eq!(first.results().len(), 2);
        assert_eq!(first.results()[0].rank(), 1);
        assert_eq!(first.results()[0].url(), "https://example.org/wal");
        assert_eq!(first.retrieved_at().timestamp(), 1_752_486_660);
    }

    #[test]
    fn the_static_provider_honours_max_results_like_a_real_engine_would() {
        let provider = StaticProvider::new("static").with_results(
            "q",
            [
                ("one", "https://example.org/1", "..."),
                ("two", "https://example.org/2", "..."),
                ("three", "https://example.org/3", "..."),
            ],
        );

        let response = provider
            .search(&SearchQuery::new("q").with_max_results(2))
            .unwrap();
        assert_eq!(response.results().len(), 2);
    }

    #[test]
    fn an_unknown_query_finds_nothing_but_still_searched() {
        let provider = StaticProvider::new("static").with_results("known", [("t", "u", "s")]);
        let response = provider.search(&SearchQuery::new("unknown")).unwrap();
        assert!(response.found_nothing());
    }

    #[test]
    fn static_results_carry_full_provenance() {
        let provider = StaticProvider::new("static")
            .with_clock(Arc::new(FixedClock::from_unix(1_752_486_660)))
            .with_results("q", [("t", "https://example.org/", "s")]);

        let response = provider.search(&SearchQuery::new("q")).unwrap();
        let provenance = response.results()[0].provenance();

        assert_eq!(provenance.engine(), "static");
        assert_eq!(provenance.query(), "q");
        assert_eq!(provenance.retrieved_at().timestamp(), 1_752_486_660);
        assert_eq!(provenance.retrieved_at_rfc3339(), "2025-07-14T09:51:00Z");
    }
}
