use serde_json::Value;

use crate::clock::{Clock, SystemClock};
use crate::error::SearchError;
use crate::http::client::{HttpClient, urlencode};
use crate::provider::SearchProvider;
use crate::query::SearchQuery;
use crate::response::SearchResponse;

/// Where to find the operator's SearXNG instance.
pub const SEARXNG_URL_ENV: &str = "KOPITIAM_SEARXNG_URL";

/// The engine name recorded in provenance and in every cache key.
///
/// A wire format: changing it orphans every recorded session.
const ENGINE: &str = "searxng";

/// Conservative default pacing. The instance is very likely the user's own
/// laptop or a small VPS, and it is aggregating upstream engines that will
/// throttle *it* if we hammer it.
const DEFAULT_REQUESTS_PER_SECOND: u32 = 1;

/// Search through a [SearXNG] instance — **the recommended provider**.
///
/// [SearXNG]: https://docs.searxng.org
///
/// # Why this one is first
///
/// SearXNG is a metasearch aggregator that you run. It queries the upstream
/// engines on your behalf and returns their results, and it is AGPL-3.0 — the
/// same licence as KOPITIAM.
///
/// For a platform whose founding principle is that no external service may ever
/// be *required*, an engine the user owns outright is not just an attractive
/// option, it is the only one that is philosophically consistent:
///
/// * **No API key.** Nothing to obtain, nothing to leak, nothing to bill.
/// * **No vendor.** No terms of service that can change under you, no account
///   that can be closed, no company that can be acquired and shut down. KOPITIAM
///   is meant to last a decade; most search API companies will not.
/// * **No cost**, so the Offline First ladder's bottom rung stops being a
///   financial cliff.
/// * **You can point it wherever you like** — including at a private,
///   institutional or air-gapped index.
///
/// The honest costs, because there always are some:
///
/// * **You must run it.** For a scientist who wants a search box and not a
///   sysadmin's afternoon, that is a real barrier — which is why
///   [`BraveProvider`](crate::BraveProvider) exists alongside it.
/// * **JSON must be enabled.** SearXNG serves HTML by default; the JSON format
///   has to be switched on in `settings.yml` (`search: formats: [html, json]`).
///   A stock instance answers a JSON request with **403**, and this adapter says
///   so in as many words rather than leaving you to guess.
/// * **Public instances are not a substitute.** Most disable JSON, rate-limit
///   aggressively, and block bots — and pointing KOPITIAM at a stranger's server
///   reintroduces exactly the third-party dependency we removed. Run your own.
/// * **It is a scraper underneath.** Upstream engines periodically break it. The
///   failure is visible and honest (an empty or erroring instance), which is the
///   most one can ask.
///
/// # Configuration
///
/// ```text
/// export KOPITIAM_SEARXNG_URL=http://localhost:8888
/// ```
///
/// # Example
///
/// ```no_run
/// use kopitiam_web::{SearchProvider, SearchQuery, SearxngProvider};
///
/// // Errors, actionably, if KOPITIAM_SEARXNG_URL is not set.
/// let provider = SearxngProvider::from_env()?;
/// let response = provider.search(&SearchQuery::new("sqlite write-ahead log"))?;
///
/// for result in response.results() {
///     println!("{}", result.citation());
/// }
/// # Ok::<(), kopitiam_web::SearchError>(())
/// ```
#[derive(Debug)]
pub struct SearxngProvider {
    base_url: String,
    client: HttpClient,
}

impl SearxngProvider {
    /// Reads the instance URL from `KOPITIAM_SEARXNG_URL`.
    ///
    /// A missing variable is [`SearchError::Misconfigured`] naming it — never a
    /// panic, and never an empty result set pretending the web is silent.
    pub fn from_env() -> Result<Self, SearchError> {
        let base_url = std::env::var(SEARXNG_URL_ENV)
            .ok()
            .filter(|url| !url.trim().is_empty())
            .ok_or_else(|| SearchError::Misconfigured {
                provider: ENGINE.to_string(),
                reason: format!(
                    "no SearXNG instance configured: set {SEARXNG_URL_ENV} to your instance \
                     (e.g. http://localhost:8888). No search was performed."
                ),
            })?;

        Self::new(base_url)
    }

    /// Uses the SearXNG instance at `base_url`.
    pub fn new(base_url: impl Into<String>) -> Result<Self, SearchError> {
        let base_url = base_url.into().trim_end_matches('/').to_string();

        // Catch the obvious mistake now, with a message, rather than as an
        // inscrutable transport error at the first search.
        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(SearchError::Misconfigured {
                provider: ENGINE.to_string(),
                reason: format!("{base_url:?} is not an http:// or https:// URL"),
            });
        }

        Ok(Self {
            base_url,
            client: HttpClient::new(ENGINE, DEFAULT_REQUESTS_PER_SECOND),
        })
    }
}

impl SearchProvider for SearxngProvider {
    fn name(&self) -> &str {
        ENGINE
    }

    fn search(&self, query: &SearchQuery) -> Result<SearchResponse, SearchError> {
        // SearXNG has no result-count parameter: it returns a page of results
        // and we take what we asked for. Trimming here (rather than pretending
        // the engine honoured max_results) keeps the response an honest record
        // of a real request.
        let mut url = format!(
            "{}/search?format=json&q={}",
            self.base_url,
            urlencode(query.text())
        );
        if let Some(language) = query.language() {
            url.push_str(&format!("&language={}", urlencode(language)));
        }

        let body = self.client.get_json(&url, &[("Accept", "application/json")])?;

        // The retrieval timestamp is taken *after* the request returns: it
        // records when we actually heard the answer, not when we started asking.
        let retrieved_at = SystemClock.now();
        parse(query, retrieved_at, &body)
    }
}

/// Reads a SearXNG JSON search response.
///
/// The shape (confirmed against the SearXNG documentation, retrieved
/// 2026-07-14) is:
///
/// ```json
/// {
///   "query": "...",
///   "number_of_results": 0,
///   "results": [
///     { "url": "...", "title": "...", "content": "...", "engine": "google", "score": 1.0 }
///   ],
///   "answers": [], "infoboxes": [], "suggestions": []
/// }
/// ```
///
/// Two deliberate choices about robustness:
///
/// * A missing `results` array is [`SearchError::MalformedResponse`], **not** an
///   empty result set. We have no idea what we received; concluding "the web
///   contains nothing" from an unparseable payload is the exact confusion this
///   crate exists to prevent. The most likely cause is an instance with JSON
///   disabled, which returns an HTML page.
/// * A result missing `url` or `title` is skipped rather than invented. `content`
///   may legitimately be absent — some upstream engines return no snippet — and
///   an empty snippet is honest where a fabricated one would not be.
fn parse(
    query: &SearchQuery,
    retrieved_at: chrono::DateTime<chrono::Utc>,
    body: &Value,
) -> Result<SearchResponse, SearchError> {
    let results = body
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| SearchError::MalformedResponse {
            provider: ENGINE.to_string(),
            reason: "no `results` array in the response. Does this instance have JSON enabled? \
                     SearXNG needs `search: formats: [html, json]` in settings.yml, and answers \
                     403 otherwise."
                .to_string(),
        })?;

    let mut builder = SearchResponse::builder(query, ENGINE, retrieved_at);
    for result in results.iter().take(query.max_results()) {
        let (Some(title), Some(url)) = (
            result.get("title").and_then(Value::as_str),
            result.get("url").and_then(Value::as_str),
        ) else {
            continue;
        };
        let snippet = result
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();

        builder.push(title, url, snippet);
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FixedClock;

    fn at() -> chrono::DateTime<chrono::Utc> {
        FixedClock::from_unix(1_752_486_660).now()
    }

    #[test]
    fn parses_a_real_searxng_payload() {
        let body: Value = serde_json::from_str(
            r#"{
                "query": "write-ahead logging",
                "number_of_results": 2,
                "results": [
                    {
                        "url": "https://en.wikipedia.org/wiki/Write-ahead_logging",
                        "title": "Write-ahead logging",
                        "content": "A technique providing atomicity and durability in databases.",
                        "engine": "wikipedia",
                        "score": 3.0
                    },
                    {
                        "url": "https://sqlite.org/wal.html",
                        "title": "SQLite WAL mode",
                        "content": "The write-ahead log journaling mode.",
                        "engine": "google",
                        "score": 1.5
                    }
                ]
            }"#,
        )
        .unwrap();

        let response = parse(&SearchQuery::new("write-ahead logging"), at(), &body).unwrap();

        assert_eq!(response.results().len(), 2);
        assert_eq!(response.results()[0].title(), "Write-ahead logging");
        assert_eq!(response.results()[0].rank(), 1);
        assert_eq!(response.results()[1].url(), "https://sqlite.org/wal.html");
        assert_eq!(response.engine(), "searxng");
        assert_eq!(response.retrieved_at(), at());

        // Provenance, on every one of them.
        for result in response.results() {
            assert_eq!(result.provenance().engine(), "searxng");
            assert_eq!(result.provenance().query(), "write-ahead logging");
            assert_eq!(result.provenance().retrieved_at(), at());
        }
    }

    #[test]
    fn an_empty_results_array_is_a_finding_not_an_error() {
        let body: Value = serde_json::from_str(r#"{"results": []}"#).unwrap();
        let response = parse(&SearchQuery::new("q"), at(), &body).unwrap();
        assert!(response.found_nothing());
    }

    #[test]
    fn a_payload_with_no_results_array_is_an_error_not_an_empty_finding() {
        // This is the case that matters. An instance with JSON disabled, a proxy
        // login page, a 200 with an HTML body -- none of them mean "the web
        // contains nothing about this", and none of them may be reported as if
        // they did.
        let body: Value = serde_json::from_str(r#"{"error": "format not allowed"}"#).unwrap();
        let error = parse(&SearchQuery::new("q"), at(), &body)
            .expect_err("an unreadable payload must not be reported as zero results");

        match error {
            SearchError::MalformedResponse { reason, .. } => {
                // ... and it tells the operator the most likely cause.
                assert!(reason.contains("settings.yml"), "{reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[test]
    fn a_result_missing_its_url_is_skipped_never_invented() {
        let body: Value = serde_json::from_str(
            r#"{"results": [
                {"title": "no url here", "content": "..."},
                {"url": "https://example.org/", "title": "fine", "content": "..."}
            ]}"#,
        )
        .unwrap();

        let response = parse(&SearchQuery::new("q"), at(), &body).unwrap();
        assert_eq!(response.results().len(), 1);
        assert_eq!(response.results()[0].url(), "https://example.org/");
    }

    #[test]
    fn a_missing_snippet_stays_empty_rather_than_being_filled_in() {
        let body: Value =
            serde_json::from_str(r#"{"results": [{"url": "https://example.org/", "title": "t"}]}"#)
                .unwrap();

        let response = parse(&SearchQuery::new("q"), at(), &body).unwrap();
        assert_eq!(response.results()[0].snippet(), "");
    }

    #[test]
    fn max_results_is_honoured_because_searxng_cannot_do_it_for_us() {
        let body: Value = serde_json::from_str(
            r#"{"results": [
                {"url": "https://example.org/1", "title": "1"},
                {"url": "https://example.org/2", "title": "2"},
                {"url": "https://example.org/3", "title": "3"}
            ]}"#,
        )
        .unwrap();

        let response = parse(&SearchQuery::new("q").with_max_results(2), at(), &body).unwrap();
        assert_eq!(response.results().len(), 2);
    }

    #[test]
    fn a_base_url_without_a_scheme_is_rejected_with_an_explanation() {
        let error = SearxngProvider::new("localhost:8888").expect_err("must reject a bare host");
        assert!(matches!(error, SearchError::Misconfigured { .. }));
    }

    #[test]
    fn a_trailing_slash_does_not_produce_a_double_slash() {
        let provider = SearxngProvider::new("http://localhost:8888/").unwrap();
        assert_eq!(provider.base_url, "http://localhost:8888");
    }

    /// Requires a running SearXNG instance with JSON enabled. Not part of the
    /// normal test run: the suite must pass with no network.
    #[test]
    #[ignore = "requires a live SearXNG instance in KOPITIAM_SEARXNG_URL"]
    fn live_searxng_search() {
        let provider = SearxngProvider::from_env().expect("KOPITIAM_SEARXNG_URL must be set");
        let response = provider
            .search(&SearchQuery::new("unicode normalization"))
            .expect("the instance must answer");

        assert!(!response.found_nothing(), "a live instance should find something");
        for result in response.results() {
            assert!(result.url().starts_with("http"));
            assert_eq!(result.provenance().engine(), "searxng");
        }
    }
}
