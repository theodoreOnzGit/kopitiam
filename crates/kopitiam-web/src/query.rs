use std::fmt;

use serde::{Deserialize, Serialize};

use crate::hash::ContentHash;

/// The default number of results to ask an engine for.
///
/// Ten because that is what a search page has meant since 1998, and because
/// every metered API bills per query rather than per result, so there is no
/// saving in asking for fewer.
const DEFAULT_MAX_RESULTS: usize = 10;

/// What to ask the web.
///
/// Deliberately small. Every field added here is a field that must be folded
/// into the [`CacheKey`] (or the cache will start returning answers to
/// questions nobody asked), must be mapped onto every provider's dialect, and
/// must be honestly reported when a provider cannot honour it. Search APIs
/// offer dozens of knobs — freshness windows, safe-search levels, geographic
/// bias, site restrictions. None of them are here until a KOPITIAM workflow
/// actually needs one.
///
/// The field order is load-bearing: [`CacheKey`] hashes the canonical JSON
/// encoding of this struct, and serde emits fields in declaration order.
/// Reordering them changes every cache key in existence, which invalidates
/// recorded sessions. Add new fields at the end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The query text, exactly as it will be sent to the engine.
    text: String,
    /// How many results to ask for.
    max_results: usize,
    /// An optional BCP-47-ish language hint (`"en"`, `"de"`), passed through to
    /// providers that accept one and ignored by those that do not.
    language: Option<String>,
}

impl SearchQuery {
    /// A query for `text`, asking for [`DEFAULT_MAX_RESULTS`] results.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            max_results: DEFAULT_MAX_RESULTS,
            language: None,
        }
    }

    /// Asks for at most `max` results.
    ///
    /// This is a request, not a guarantee: an engine may return fewer (it found
    /// fewer) or, in principle, cap it lower. What comes back is what came
    /// back — this crate never pads a short result set.
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Hints a preferred content language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// The query text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The requested result count.
    pub fn max_results(&self) -> usize {
        self.max_results
    }

    /// The language hint, if any.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

impl fmt::Display for SearchQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// The identity of one (engine, query, parameters) triple in the cache.
///
/// # Why the parameters are part of the key
///
/// A cache keyed on the query text alone would happily answer a request for 20
/// German results with 5 English ones it happened to have — and it would do so
/// while carrying a provenance record claiming the engine was asked the second
/// question. That is not a stale cache; that is a forged citation. Every input
/// that can change what an engine returns must therefore be inside the key.
///
/// The engine is in the key for the same reason. "Brave said X" and "my SearXNG
/// instance said X" are different claims about the world, and a knowledge graph
/// that conflates them cannot be audited.
///
/// The key is a stable string — `<engine>/<sha256 of the canonical query>` —
/// because it doubles as a redb key in [`StoreCache`], and a persistent store's
/// keys must survive process restarts, compiler upgrades and hash-seed changes.
/// (A `HashMap`-style hash would not: Rust deliberately randomizes `SipHash`
/// seeds per process.)
///
/// [`StoreCache`]: crate::StoreCache
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CacheKey(String);

impl CacheKey {
    /// Derives the key for `query` as it would be sent to `engine`.
    pub fn new(engine: &str, query: &SearchQuery) -> Self {
        // serde_json is a canonical encoding *for this struct*: no maps with
        // arbitrary iteration order, no floats, fields emitted in declaration
        // order. If SearchQuery ever grows a HashMap field, this stops being
        // true and the key must be built explicitly instead.
        let canonical = serde_json::to_string(query)
            .expect("SearchQuery is plain data and cannot fail to serialize");
        Self(format!("{engine}/{}", ContentHash::of(canonical).as_hex()))
    }

    /// The key as the string used to address the persistent store.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_ten_results_and_no_language() {
        let query = SearchQuery::new("rust async runtime");
        assert_eq!(query.text(), "rust async runtime");
        assert_eq!(query.max_results(), DEFAULT_MAX_RESULTS);
        assert_eq!(query.language(), None);
    }

    #[test]
    fn the_same_query_always_yields_the_same_key() {
        let a = CacheKey::new("brave", &SearchQuery::new("write-ahead log"));
        let b = CacheKey::new("brave", &SearchQuery::new("write-ahead log"));
        assert_eq!(a, b);
    }

    #[test]
    fn different_engines_never_share_a_key() {
        let query = SearchQuery::new("write-ahead log");
        assert_ne!(CacheKey::new("brave", &query), CacheKey::new("searxng", &query));
    }

    #[test]
    fn every_parameter_that_changes_the_answer_changes_the_key() {
        let base = SearchQuery::new("write-ahead log");
        let key = CacheKey::new("brave", &base);

        // A different result count is a different question.
        assert_ne!(key, CacheKey::new("brave", &base.clone().with_max_results(20)));
        // So is a different language.
        assert_ne!(key, CacheKey::new("brave", &base.clone().with_language("de")));
        // So is different text.
        assert_ne!(key, CacheKey::new("brave", &SearchQuery::new("write-ahead logs")));
    }

    #[test]
    fn the_key_is_a_stable_string_not_a_process_local_hash() {
        // Regression guard: if this key ever changes, every recorded session in
        // every user's .kopitiam directory silently becomes unreplayable. The
        // literal is therefore pinned on purpose, and a failure here means you
        // have changed the wire format of the cache.
        let key = CacheKey::new("searxng", &SearchQuery::new("unicode normalization"));
        assert_eq!(
            key.as_str(),
            "searxng/ce73b6eda4b162c379c9a33bc5ddb37c72fb6e3185bd148bb665e2bb4e7de5fe"
        );
    }
}
