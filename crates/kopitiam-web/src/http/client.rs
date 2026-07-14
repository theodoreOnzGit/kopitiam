use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::clock::{Clock, SystemClock};
use crate::error::SearchError;
use crate::rate_limit::RateLimiter;

/// How long we will wait for a whole search request before giving up.
///
/// Short on purpose. A search is the *last* thing KOPITIAM tries; if it is
/// slow, the right answer is to stop waiting and get on without it, not to
/// block a scientist's workflow behind a stalled TCP connection.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// The most of an error body we will keep for diagnosis.
///
/// Bounded because a provider that is failing may well be returning a megabyte
/// of HTML, and that megabyte should not end up in a log, an error chain, or a
/// knowledge graph.
const MAX_ERROR_BODY: usize = 512;

/// An API key, which knows not to say itself out loud.
///
/// The [`Debug`] implementation redacts the value, so a key cannot leak through
/// the route it usually leaks by: someone deriving `Debug` on a struct three
/// layers up and logging it, or an error being formatted with `{:?}` into a
/// crash report. There is no `Display`, and no accessor other than
/// [`expose`](ApiKey::expose), which is deliberately ugly to type and easy to
/// grep for.
///
/// The key is only ever read from the environment. It is never written to the
/// cache, never included in a [`SearchError`], never put in a provenance record
/// and never logged.
#[derive(Clone)]
pub(crate) struct ApiKey(String);

impl ApiKey {
    /// Reads `env_var`, or reports which variable the human has to set.
    ///
    /// An absent key is an honest, actionable [`SearchError::MissingApiKey`] —
    /// not a panic (this is a library; a missing config value is not a bug in
    /// the program) and, above all, not an empty result set, which would tell
    /// the caller that the web contains nothing on the subject.
    ///
    /// An empty or whitespace-only value is treated as absent, because
    /// `KOPITIAM_BRAVE_API_KEY=""` in a shell profile means the same thing to a
    /// human as not setting it, and a confusing 401 later is a worse experience
    /// than a clear message now.
    pub(crate) fn from_env(provider: &str, env_var: &'static str) -> Result<Self, SearchError> {
        match std::env::var(env_var) {
            Ok(value) if !value.trim().is_empty() => Ok(Self(value)),
            _ => Err(SearchError::MissingApiKey {
                provider: provider.to_string(),
                env_var,
            }),
        }
    }

    /// The key itself. Call this only where it is handed to the transport.
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// A rate-limited, rustls-backed HTTP client shared by the live adapters.
///
/// Synchronous, because [`SearchProvider`](crate::SearchProvider) is
/// synchronous, because `kopitiam-ai`'s `ModelAdapter` is synchronous, and
/// because KOPITIAM has no async runtime to justify: a search is one request,
/// made rarely, whose latency is dominated by a remote index.
pub(crate) struct HttpClient {
    agent: ureq::Agent,
    limiter: RateLimiter,
    clock: Arc<dyn Clock>,
    provider: String,
}

impl fmt::Debug for HttpClient {
    /// Hand-written rather than derived, because `Arc<dyn Clock>` is not `Debug`
    /// — and because a derived impl on a client that (in other providers) sits
    /// next to an [`ApiKey`] is precisely the accident that leaks keys into logs.
    /// Nothing here can print a secret, because nothing here prints a field.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpClient")
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl HttpClient {
    /// A client for `provider`, allowing at most `requests_per_second`.
    pub(crate) fn new(provider: impl Into<String>, requests_per_second: u32) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            // ureq turns 4xx/5xx into Err by default, which throws away the
            // response body -- and the body is where a provider explains *why*
            // it refused. We want to read it and map it onto a precise
            // SearchError, so we handle statuses ourselves.
            .http_status_as_error(false)
            .build();

        Self {
            agent: ureq::Agent::new_with_config(config),
            limiter: RateLimiter::per_second(requests_per_second),
            clock: Arc::new(SystemClock),
            provider: provider.into(),
        }
    }

    /// Issues a GET and parses the response as JSON.
    ///
    /// Blocks first for as long as the rate limiter requires. Every failure mode
    /// is mapped onto the [`SearchError`] variant that describes it, so the
    /// caller can tell a throttle from an outage from a bad key — and none of
    /// them are ever mapped onto "no results".
    pub(crate) fn get_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
    ) -> Result<Value, SearchError> {
        self.limiter.acquire(self.clock.as_ref());

        let mut request = self.agent.get(url);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }

        let mut response = request.call().map_err(|err| self.transport_error(err))?;
        let status = response.status().as_u16();

        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|err| SearchError::Network {
                provider: self.provider.clone(),
                reason: format!("reading the response body failed: {err}"),
            })?;

        if status == 429 {
            return Err(SearchError::RateLimited {
                provider: self.provider.clone(),
                // Retry-After is optional and providers differ on whether they
                // send it. We report what we were told and never invent a delay.
                retry_after_secs: response
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.trim().parse::<u64>().ok()),
            });
        }

        if !(200..300).contains(&status) {
            return Err(SearchError::Http {
                provider: self.provider.clone(),
                status,
                body: truncate(&body, MAX_ERROR_BODY),
            });
        }

        serde_json::from_str(&body).map_err(|err| SearchError::MalformedResponse {
            provider: self.provider.clone(),
            // Not the body: an error page can be enormous, and a search endpoint
            // that answers with HTML has told us everything we needed to know
            // already.
            reason: format!("the response is not JSON: {err}"),
        })
    }

    /// Maps a ureq transport failure onto the right [`SearchError`].
    fn transport_error(&self, err: ureq::Error) -> SearchError {
        let provider = self.provider.clone();
        match err {
            // Unreachable in practice (http_status_as_error is off), but the
            // enum is #[non_exhaustive] and we would rather not guess.
            ureq::Error::StatusCode(status) => SearchError::Http {
                provider,
                status,
                body: String::new(),
            },
            ureq::Error::BadUri(uri) => SearchError::Misconfigured {
                provider,
                reason: format!("not a usable URL: {uri}"),
            },
            // Everything else -- DNS, TLS, timeouts, resets, no route to host --
            // is one thing to the caller: the web was not reachable, so carry on
            // without it.
            other => SearchError::Network {
                provider,
                reason: other.to_string(),
            },
        }
    }
}

/// Percent-encodes a query-string parameter value.
///
/// Hand-rolled rather than pulling in a URL crate for one function: the rule is
/// RFC 3986's unreserved set kept as-is, every other byte escaped, and getting
/// it wrong is caught by the tests below.
///
/// Space becomes `%20`, not `+`. Both are legal in a query string, but `%20` is
/// unambiguous everywhere, including on servers that read `+` as a literal plus
/// — and a query silently mangled into a different query would poison the cache
/// key and the provenance record alike.
pub(crate) fn urlencode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

/// Cuts `text` to at most `limit` bytes, on a character boundary.
fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name no sane environment defines, so these tests cannot be perturbed by
    /// the machine they run on.
    const ABSENT: &str = "KOPITIAM_WEB_TEST_KEY_THAT_IS_NEVER_SET";

    #[test]
    fn a_missing_key_is_an_actionable_error_naming_the_variable() {
        let error = ApiKey::from_env("brave", ABSENT)
            .expect_err("a key that is not in the environment must not be conjured up");

        match error {
            SearchError::MissingApiKey { provider, env_var } => {
                assert_eq!(provider, "brave");
                assert_eq!(env_var, ABSENT);
            }
            other => panic!("expected MissingApiKey, got {other:?}"),
        }
    }

    #[test]
    fn a_key_never_prints_itself() {
        // The leak path this closes: someone derives Debug on a provider struct
        // and logs it.
        let key = ApiKey("super-secret-value".to_string());
        let debugged = format!("{key:?}");
        assert!(!debugged.contains("super-secret-value"), "{debugged}");
        assert_eq!(debugged, "ApiKey(<redacted>)");
    }

    #[test]
    fn truncation_never_splits_a_character() {
        // A provider failing with a multi-byte error page must not make us panic
        // on top of it.
        let text = "\u{1F600}".repeat(100); // 4 bytes each
        let cut = truncate(&text, 10);
        assert!(cut.ends_with("..."));
        assert_eq!(cut.chars().filter(|c| *c == '\u{1F600}').count(), 2);
    }

    #[test]
    fn short_text_is_left_alone() {
        assert_eq!(truncate("brief", 512), "brief");
    }

    #[test]
    fn query_encoding_escapes_everything_outside_the_unreserved_set() {
        assert_eq!(urlencode("unicode normalization"), "unicode%20normalization");
        assert_eq!(urlencode("read-only_cache.v2~"), "read-only_cache.v2~");
        // An unescaped `&` or `=` would smuggle extra parameters into the URL.
        assert_eq!(urlencode("a&b=c"), "a%26b%3Dc");
        // Non-ASCII is encoded per UTF-8 byte.
        assert_eq!(urlencode("é"), "%C3%A9");
    }
}
