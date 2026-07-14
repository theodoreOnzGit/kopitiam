use thiserror::Error;

/// Everything that can go wrong *instead of* searching.
///
/// # Why this is not `anyhow::Error`
///
/// `kopitiam-ai`'s [`ModelAdapter`] returns `anyhow::Result` because a caller
/// there only ever needs to know *that* the model failed, and then falls back
/// to the next rung of the Offline First ladder.
///
/// Web search is different: the *reason* it failed changes what the caller
/// should do about it, and there is no rung below the web to fall back to.
///
/// * A [`RateLimited`](SearchError::RateLimited) call should be retried later,
///   possibly after `retry_after`. The knowledge is out there; we asked too
///   fast.
/// * A [`MissingApiKey`](SearchError::MissingApiKey) is a *configuration*
///   problem the human must fix. Retrying will never help, and it is not a
///   fact about the world.
/// * A [`Disabled`](SearchError::Disabled) provider means the operator chose
///   not to search. Respect that; do not route around it.
/// * A [`CacheMiss`](SearchError::CacheMiss) in replay mode means the recorded
///   session did not cover this query — a *reproducibility* failure, and
///   silently going to the network to paper over it would destroy the very
///   determinism replay exists to provide.
///
/// So the variants are the API. Callers are expected to match on them; see
/// [`SearchError::is_transient`] and [`SearchError::is_configuration`] for the
/// two coarse questions most callers actually have.
///
/// # What this type is *never* allowed to become
///
/// An empty result list. Returning `Ok(SearchResponse { results: [] })` for
/// any of these conditions would tell the caller "the web contains nothing
/// about this", which is a different — and false — claim. See the crate docs.
///
/// [`ModelAdapter`]: https://docs.rs/kopitiam-ai
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SearchError {
    /// Web search is switched off. Returned by [`NullProvider`], which is the
    /// correct default for a KOPITIAM installation that has not opted in.
    ///
    /// This is emphatically not "no results": the operator declined to look.
    ///
    /// [`NullProvider`]: crate::NullProvider
    #[error(
        "web search is disabled (provider `{provider}`): no search was performed, \
         which is not the same as finding no results"
    )]
    Disabled {
        /// The provider that declined.
        provider: String,
    },

    /// A required API key was not present in the environment.
    ///
    /// The message names the exact environment variable, because an error a
    /// human cannot act on is barely better than a panic. The key's *value* is
    /// never included anywhere in this type — the `ApiKey` newtype in the
    /// `http` module redacts itself even under `{:?}`.
    #[error(
        "web search provider `{provider}` needs an API key: set the environment \
         variable {env_var}. No search was performed."
    )]
    MissingApiKey {
        /// The provider that needs the key.
        provider: String,
        /// The environment variable to set, e.g. `KOPITIAM_BRAVE_API_KEY`.
        env_var: &'static str,
    },

    /// The provider is configured wrongly — a malformed base URL, a nonsense
    /// result count, an instance that does not speak JSON.
    #[error("web search provider `{provider}` is misconfigured: {reason}")]
    Misconfigured {
        /// The provider concerned.
        provider: String,
        /// What is wrong, in terms the operator can act on.
        reason: String,
    },

    /// The provider refused us for asking too often.
    ///
    /// Callers should back off. A `retry_after` of `None` means the provider
    /// did not say how long to wait (many do not); it does *not* mean "retry
    /// immediately".
    #[error("web search provider `{provider}` rate-limited the request{}", .retry_after_secs.map(|s| format!(" (retry after {s}s)")).unwrap_or_default())]
    RateLimited {
        /// The provider that throttled us.
        provider: String,
        /// Seconds to wait, if the provider said (e.g. via `Retry-After`).
        retry_after_secs: Option<u64>,
    },

    /// The request never completed: no network, DNS failure, TLS failure,
    /// timeout, connection reset.
    ///
    /// This is the ordinary consequence of KOPITIAM running where it is
    /// designed to run — offline. It is an error, not a catastrophe, and the
    /// caller is expected to carry on without the web.
    #[error("web search provider `{provider}` could not be reached: {reason}")]
    Network {
        /// The provider we failed to reach.
        provider: String,
        /// The transport-level reason.
        reason: String,
    },

    /// The provider answered, but with an HTTP status we cannot interpret as a
    /// search (401, 403, 500, ...). 429 is [`RateLimited`] instead.
    ///
    /// [`RateLimited`]: SearchError::RateLimited
    #[error("web search provider `{provider}` returned HTTP {status}: {body}")]
    Http {
        /// The provider concerned.
        provider: String,
        /// The HTTP status code.
        status: u16,
        /// A truncated, secret-free excerpt of the response body, for
        /// diagnosis.
        body: String,
    },

    /// The provider answered with something that is not the search response we
    /// know how to read — a changed schema, an HTML error page, a truncated
    /// body.
    ///
    /// This is a hard error on purpose. The tempting alternative — skip the
    /// fields we cannot parse and return whatever survived — would fabricate a
    /// result set that no engine ever produced.
    #[error("web search provider `{provider}` returned a response we cannot read: {reason}")]
    MalformedResponse {
        /// The provider concerned.
        provider: String,
        /// What was wrong with the payload.
        reason: String,
    },

    /// [`CacheMode::Replay`] was requested and the query is not in the cache.
    ///
    /// Replay exists to make a run reproducible. A miss means the recording is
    /// incomplete; going to the network to fill the gap would silently turn a
    /// replay back into a live search, so we refuse and say so.
    ///
    /// [`CacheMode::Replay`]: crate::CacheMode::Replay
    #[error(
        "no cached result for query {query:?} on provider `{provider}`, and replay mode \
         forbids going to the network. The recorded session does not cover this query."
    )]
    CacheMiss {
        /// The provider the query would have gone to.
        provider: String,
        /// The query text that was not found in the cache.
        query: String,
    },

    /// The cache itself failed — the redb store could not be opened, a record
    /// could not be deserialized.
    #[error("web search cache failure: {0}")]
    Cache(#[source] anyhow::Error),
}

impl SearchError {
    /// Whether waiting and trying again could plausibly succeed.
    ///
    /// True for rate limits and network failures; false for everything a human
    /// has to fix (missing key, misconfiguration) and everything that is a
    /// deliberate choice (disabled, replay miss).
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            SearchError::RateLimited { .. } | SearchError::Network { .. } | SearchError::Http { .. }
        )
    }

    /// Whether a human must change the configuration before search can work.
    ///
    /// A caller that is deciding whether to *tell the user something* — rather
    /// than quietly carrying on — should ask this.
    pub fn is_configuration(&self) -> bool {
        matches!(
            self,
            SearchError::MissingApiKey { .. }
                | SearchError::Misconfigured { .. }
                | SearchError::Disabled { .. }
        )
    }

    /// The provider this error came from, where one is identifiable.
    pub fn provider(&self) -> Option<&str> {
        match self {
            SearchError::Disabled { provider }
            | SearchError::MissingApiKey { provider, .. }
            | SearchError::Misconfigured { provider, .. }
            | SearchError::RateLimited { provider, .. }
            | SearchError::Network { provider, .. }
            | SearchError::Http { provider, .. }
            | SearchError::MalformedResponse { provider, .. }
            | SearchError::CacheMiss { provider, .. } => Some(provider),
            SearchError::Cache(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_failures_so_a_caller_can_fall_back() {
        let throttled = SearchError::RateLimited {
            provider: "brave".to_string(),
            retry_after_secs: Some(30),
        };
        assert!(throttled.is_transient());
        assert!(!throttled.is_configuration());

        let no_key = SearchError::MissingApiKey {
            provider: "brave".to_string(),
            env_var: "KOPITIAM_BRAVE_API_KEY",
        };
        assert!(!no_key.is_transient());
        assert!(no_key.is_configuration());
    }

    #[test]
    fn a_missing_key_names_the_variable_the_human_must_set() {
        let err = SearchError::MissingApiKey {
            provider: "brave".to_string(),
            env_var: "KOPITIAM_BRAVE_API_KEY",
        };
        let message = err.to_string();
        assert!(message.contains("KOPITIAM_BRAVE_API_KEY"), "{message}");
        // And it must say that nothing was searched, so the message can never
        // be misread as "we searched and found nothing".
        assert!(message.contains("No search was performed"), "{message}");
    }

    #[test]
    fn a_disabled_provider_says_so_rather_than_implying_emptiness() {
        let err = SearchError::Disabled {
            provider: "null".to_string(),
        };
        assert!(err.to_string().contains("not the same as finding no results"));
        assert_eq!(err.provider(), Some("null"));
    }

    #[test]
    fn a_rate_limit_reports_the_retry_delay_when_the_provider_gave_one() {
        let with_delay = SearchError::RateLimited {
            provider: "brave".to_string(),
            retry_after_secs: Some(12),
        };
        assert!(with_delay.to_string().contains("retry after 12s"));

        // ... and does not invent one when it did not.
        let without = SearchError::RateLimited {
            provider: "brave".to_string(),
            retry_after_secs: None,
        };
        assert!(!without.to_string().contains("retry after"));
    }
}
