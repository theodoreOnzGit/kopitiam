use serde_json::Value;

use crate::clock::{Clock, SystemClock};
use crate::error::SearchError;
use crate::http::client::{ApiKey, HttpClient, urlencode};
use crate::provider::SearchProvider;
use crate::query::SearchQuery;
use crate::response::SearchResponse;

/// Where the Brave Search API key is read from. Never hardcoded, never logged,
/// never cached, never recorded in provenance.
pub const BRAVE_API_KEY_ENV: &str = "KOPITIAM_BRAVE_API_KEY";

/// The engine name recorded in provenance and in every cache key. A wire format.
const ENGINE: &str = "brave";

/// Brave's endpoint (documentation retrieved 2026-07-14).
const ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";

/// Brave's free tier permits one query per second. Pace to it by default: a 429
/// is a wasted query, and on some plans a strike.
const DEFAULT_REQUESTS_PER_SECOND: u32 = 1;

/// Brave caps `count` at 20 (documentation retrieved 2026-07-14). Asking for
/// more is an error rather than a silent truncation, so clamp.
const MAX_COUNT: usize = 20;

/// Search through the [Brave Search API].
///
/// [Brave Search API]: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
///
/// # Why Brave, of the vendors
///
/// Brave is here as the pragmatic option for a user who does not want to run
/// their own [`SearxngProvider`](crate::SearxngProvider) — and that is genuinely
/// most people. Of the commercial APIs it is the least bad fit for KOPITIAM:
///
/// * It runs an **independent crawl and index**. Serper and SerpAPI resell
///   Google; paying them buys a costlier copy of an answer, not a second
///   opinion. When KOPITIAM records "Brave said X on this date", that is a claim
///   about a distinct index, which is worth something in a provenance record.
/// * The API is a **plain, documented, stable HTTP endpoint** returning JSON,
///   not a scraping layer that breaks when someone changes a CSS class.
/// * There is a free tier, so it can be tried without a commitment.
///
/// The costs, stated plainly:
///
/// * **An API key**, an account, and a vendor relationship. Every one of those
///   can be revoked, repriced, or wound up, and KOPITIAM intends to outlive all
///   three.
/// * **Metered.** The free tier is limited; beyond it, searching costs money —
///   and a scientific workbench whose knowledge pipeline has a credit card in it
///   is exactly what the Offline First principle exists to avoid.
/// * **A smaller index than Google's.** For obscure scientific queries this
///   sometimes shows.
///
/// Hence the ordering in the crate docs: SearXNG first, Brave second, and
/// neither of them required.
///
/// # Configuration
///
/// ```text
/// export KOPITIAM_BRAVE_API_KEY=<your key>
/// ```
///
/// If it is not set, [`from_env`](Self::from_env) returns
/// [`SearchError::MissingApiKey`] naming the variable. It does not panic (a
/// missing config value is not a bug in the program) and it emphatically does
/// not return zero results, which would tell the caller that the web is silent
/// on a subject nobody ever asked about.
/// `Debug` is safe to derive here: [`ApiKey`] redacts its own value.
#[derive(Debug)]
pub struct BraveProvider {
    api_key: ApiKey,
    client: HttpClient,
}

impl BraveProvider {
    /// Reads the API key from `KOPITIAM_BRAVE_API_KEY`.
    pub fn from_env() -> Result<Self, SearchError> {
        Ok(Self {
            api_key: ApiKey::from_env(ENGINE, BRAVE_API_KEY_ENV)?,
            client: HttpClient::new(ENGINE, DEFAULT_REQUESTS_PER_SECOND),
        })
    }
}

impl SearchProvider for BraveProvider {
    fn name(&self) -> &str {
        ENGINE
    }

    fn search(&self, query: &SearchQuery) -> Result<SearchResponse, SearchError> {
        let count = query.max_results().clamp(1, MAX_COUNT);
        let mut url = format!(
            "{ENDPOINT}?q={}&count={count}",
            urlencode(query.text())
        );
        if let Some(language) = query.language() {
            url.push_str(&format!("&search_lang={}", urlencode(language)));
        }

        let body = self.client.get_json(
            &url,
            &[
                ("Accept", "application/json"),
                // The one place the key is ever exposed: handed straight to the
                // transport, never stored, never formatted into anything.
                ("X-Subscription-Token", self.api_key.expose()),
            ],
        )?;

        // Stamped after the answer arrives: this records when we heard it, not
        // when we asked.
        let retrieved_at = SystemClock.now();
        parse(query, retrieved_at, &body)
    }
}

/// Reads a Brave web-search response.
///
/// The shape (confirmed against Brave's documentation, retrieved 2026-07-14):
///
/// ```json
/// {
///   "query": { "original": "...", "more_results_available": true },
///   "web": {
///     "results": [
///       { "title": "...", "url": "https://...", "description": "..." }
///     ]
///   }
/// }
/// ```
///
/// Note `description`, where SearXNG says `content` — the two engines' dialects
/// differ, which is precisely why this crate has a common
/// [`SearchResult`](crate::SearchResult) rather than passing vendor JSON around.
///
/// A response with no `web.results` array is [`SearchError::MalformedResponse`],
/// **not** an empty result set. Brave omits `web` entirely for some queries, and
/// we genuinely cannot tell "the index has nothing" from "the schema changed" —
/// so we say we cannot tell, rather than guessing in the direction that happens
/// to be convenient. An explicit `"web": {"results": []}` *is* zero results, and
/// is reported as such.
fn parse(
    query: &SearchQuery,
    retrieved_at: chrono::DateTime<chrono::Utc>,
    body: &Value,
) -> Result<SearchResponse, SearchError> {
    let results = body
        .get("web")
        .and_then(|web| web.get("results"))
        .and_then(Value::as_array)
        .ok_or_else(|| SearchError::MalformedResponse {
            provider: ENGINE.to_string(),
            reason: "no `web.results` array in the response; the payload is not a web-search \
                     result we can read, and it must not be mistaken for an empty one"
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
            .get("description")
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
    fn parses_a_brave_payload() {
        let body: Value = serde_json::from_str(
            r#"{
                "query": {"original": "write-ahead logging"},
                "web": {
                    "results": [
                        {
                            "title": "Write-ahead logging - Wikipedia",
                            "url": "https://en.wikipedia.org/wiki/Write-ahead_logging",
                            "description": "A technique providing atomicity and durability in databases."
                        }
                    ]
                }
            }"#,
        )
        .unwrap();

        let response = parse(&SearchQuery::new("write-ahead logging"), at(), &body).unwrap();

        assert_eq!(response.results().len(), 1);
        assert_eq!(response.results()[0].title(), "Write-ahead logging - Wikipedia");
        assert_eq!(
            response.results()[0].snippet(),
            "A technique providing atomicity and durability in databases."
        );
        assert_eq!(response.results()[0].provenance().engine(), "brave");
        assert_eq!(response.results()[0].provenance().retrieved_at(), at());
    }

    #[test]
    fn an_explicit_empty_result_array_is_zero_results() {
        let body: Value = serde_json::from_str(r#"{"web": {"results": []}}"#).unwrap();
        let response = parse(&SearchQuery::new("q"), at(), &body).unwrap();
        assert!(response.found_nothing());
    }

    #[test]
    fn a_response_without_a_web_block_is_an_error_not_zero_results() {
        // Brave omits `web` for some queries, and a schema change would look the
        // same. We cannot distinguish "nothing indexed" from "we no longer
        // understand the answer" -- so we refuse to guess, rather than guessing
        // in the direction that silently produces "the web knows nothing".
        let body: Value = serde_json::from_str(r#"{"query": {"original": "q"}}"#).unwrap();
        let error = parse(&SearchQuery::new("q"), at(), &body)
            .expect_err("an unreadable payload must not become an empty finding");
        assert!(matches!(error, SearchError::MalformedResponse { .. }));
    }

    #[test]
    fn a_missing_key_is_reported_honestly_rather_than_as_no_results() {
        // Deliberately not using the real variable name, so this test cannot be
        // affected by a developer's shell.
        let error = ApiKey::from_env(ENGINE, "KOPITIAM_BRAVE_API_KEY_DEFINITELY_UNSET")
            .expect_err("an absent key must be an error");

        assert!(error.is_configuration());
        assert!(!error.is_transient(), "retrying will not conjure up a key");
        assert!(error.to_string().contains("No search was performed"));
    }

    #[test]
    fn debug_printing_the_whole_provider_does_not_leak_the_key() {
        // `BraveProvider` derives Debug, so it *can* be logged -- which is
        // exactly how API keys escape in practice. Assert that even the whole
        // provider, printed with {:?}, says nothing.
        let provider = BraveProvider {
            api_key: ApiKey::from_env(ENGINE, "PATH")
                .expect("PATH is set in every environment; any non-empty value will do"),
            client: HttpClient::new(ENGINE, DEFAULT_REQUESTS_PER_SECOND),
        };

        let debugged = format!("{provider:?}");
        let secret = std::env::var("PATH").unwrap();
        assert!(!debugged.contains(&secret), "the key must never be printable: {debugged}");
        assert!(debugged.contains("<redacted>"), "{debugged}");
    }

    #[test]
    fn the_api_key_environment_variable_is_the_documented_one() {
        // Guards against a typo silently making the documented variable useless.
        assert_eq!(BRAVE_API_KEY_ENV, "KOPITIAM_BRAVE_API_KEY");
    }

    /// Requires a real API key and a network. Never runs in the normal suite.
    #[test]
    #[ignore = "requires a live Brave API key in KOPITIAM_BRAVE_API_KEY"]
    fn live_brave_search() {
        let provider = BraveProvider::from_env().expect("KOPITIAM_BRAVE_API_KEY must be set");
        let response = provider
            .search(&SearchQuery::new("unicode normalization").with_max_results(3))
            .expect("Brave must answer");

        assert!(!response.found_nothing());
        assert!(response.results().len() <= 3);
        for result in response.results() {
            assert!(result.url().starts_with("http"));
            assert_eq!(result.provenance().engine(), "brave");
        }
    }
}
