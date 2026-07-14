use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use chrono::Duration;
use kopitiam_index::Store;

use crate::clock::{Clock, SystemClock};
use crate::error::SearchError;
use crate::provider::SearchProvider;
use crate::query::{CacheKey, SearchQuery};
use crate::response::SearchResponse;

/// Somewhere to record what the web said, so it can be said again.
///
/// A cache implementation must obey one rule above all others: **what goes in
/// comes out unchanged**. In particular it may not touch
/// [`SearchResponse::retrieved_at`]. See [`CachedProvider`] for why that is the
/// hill this crate dies on.
pub trait SearchCache {
    /// The recorded response for `key`, if one was ever recorded.
    fn get(&self, key: &CacheKey) -> Result<Option<SearchResponse>, SearchError>;

    /// Records `response` under `key`, replacing any previous recording.
    fn put(&self, key: &CacheKey, response: &SearchResponse) -> Result<(), SearchError>;
}

/// An in-process cache that forgets everything when the process exits.
///
/// Useful for tests and for a single CLI invocation that might ask the same
/// question twice. For a cache that survives — and therefore for anything that
/// deserves the name *provenance record* — use [`StoreCache`].
#[derive(Debug, Default)]
pub struct MemoryCache {
    entries: RwLock<HashMap<CacheKey, SearchResponse>>,
}

impl MemoryCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many searches are recorded.
    pub fn len(&self) -> usize {
        self.entries.read().expect("cache lock poisoned").len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SearchCache for MemoryCache {
    fn get(&self, key: &CacheKey) -> Result<Option<SearchResponse>, SearchError> {
        Ok(self
            .entries
            .read()
            .expect("cache lock poisoned")
            .get(key)
            .cloned())
    }

    fn put(&self, key: &CacheKey, response: &SearchResponse) -> Result<(), SearchError> {
        self.entries
            .write()
            .expect("cache lock poisoned")
            .insert(key.clone(), response.clone());
        Ok(())
    }
}

/// The key prefix under which searches are stored in the project's redb
/// database.
///
/// Namespaced because `.kopitiam/state.redb` is shared with the rest of the
/// Semantic Runtime (session memory, working set, graph snapshots) and a
/// collision would be silent.
const STORE_PREFIX: &str = "web/search/";

/// A cache persisted in the project's `.kopitiam` directory.
///
/// Backed by [`kopitiam_index::Store`] — redb, pure Rust, ACID, no C dependency
/// — rather than a bespoke file format. Persistence is a solved problem inside
/// KOPITIAM and this crate has no business re-solving it; a search cache that
/// invented its own on-disk format would be one more thing to version, corrupt,
/// and migrate.
///
/// # This is the provenance store
///
/// It is tempting to think of this as an optimization — fewer API calls, lower
/// bills. That is a side effect. Its actual job is to be the **permanent record
/// of what the web said and when**, which is the only thing that makes a web
/// citation checkable, a run reproducible, and a knowledge graph auditable
/// after the source page has been rewritten or deleted. Deleting the cache
/// destroys evidence, not just speed.
pub struct StoreCache {
    store: Store,
}

impl StoreCache {
    /// Opens (creating if needed) the cache in `root`'s `.kopitiam` directory.
    pub fn open(root: &Path) -> Result<Self, SearchError> {
        let store = Store::open(root).map_err(SearchError::Cache)?;
        Ok(Self { store })
    }

    fn store_key(key: &CacheKey) -> String {
        format!("{STORE_PREFIX}{key}")
    }
}

impl SearchCache for StoreCache {
    fn get(&self, key: &CacheKey) -> Result<Option<SearchResponse>, SearchError> {
        self.store
            .get_json(&Self::store_key(key))
            .map_err(SearchError::Cache)
    }

    fn put(&self, key: &CacheKey, response: &SearchResponse) -> Result<(), SearchError> {
        self.store
            .put_json(&Self::store_key(key), response)
            .map_err(SearchError::Cache)
    }
}

/// How a [`CachedProvider`] should treat the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheMode {
    /// Serve from the cache when possible; go to the network on a miss, and
    /// record what comes back. The normal mode.
    #[default]
    ReadThrough,

    /// Serve **only** from the cache. A miss is [`SearchError::CacheMiss`]; the
    /// network is never touched.
    ///
    /// This is what makes a run that used the web reproducible. Record a
    /// session once in [`ReadThrough`](CacheMode::ReadThrough); re-run it in
    /// `Replay` and get byte-identical results — the same pages, the same
    /// ranks, the same timestamps — on a machine with no network at all,
    /// months later, from a vendor that may since have shut down.
    ///
    /// Staleness is deliberately ignored in this mode: an expiry policy that
    /// silently reached for the network would turn a replay back into a live
    /// search, which is the one thing replay exists to prevent.
    Replay,

    /// Ignore any cached entry and always go to the network, recording the
    /// result.
    ///
    /// The refreshed response carries a *new* retrieval timestamp, because it
    /// is a new retrieval. The old record is overwritten; if you need both, you
    /// need a history, and this cache does not pretend to be one.
    Refresh,
}

/// Wraps any [`SearchProvider`] with recording and replay.
///
/// # The invariant this type exists to protect
///
/// **A cached result is returned with the timestamp of its original retrieval,
/// never re-stamped as fresh.**
///
/// The bug is easy to write and almost impossible to see. You fetch a cached
/// response; you are about to hand it to a caller; and somewhere — in a
/// convenience constructor, in a `Provenance::new(.., Utc::now(), ..)` that
/// looked harmless — the timestamp gets refreshed. Now the knowledge graph
/// asserts that the web said something *today* which it actually said in March,
/// and possibly no longer says at all. Every downstream citation inherits the
/// lie, and nothing about the data looks wrong.
///
/// So this type never constructs a [`Provenance`](crate::Provenance) and never
/// touches a timestamp. It moves recorded bytes, verbatim. The only thing it
/// changes on the way out is [`SearchResponse::from_cache`], which describes
/// *this delivery* rather than the response, and which is deliberately not
/// persisted.
///
/// [`SearchResponse::from_cache`]: crate::SearchResponse::from_cache
///
/// # Errors are not cached
///
/// A rate limit, a dropped connection or a missing API key is a fact about
/// *us*, at one moment, not a fact about the world. Recording one would poison
/// the cache with a failure that replays forever. Only successful searches —
/// including successful searches that found nothing — are recorded.
pub struct CachedProvider<P, C> {
    inner: P,
    cache: C,
    mode: CacheMode,
    clock: Arc<dyn Clock>,
    max_age: Option<Duration>,
}

impl<P: SearchProvider, C: SearchCache> CachedProvider<P, C> {
    /// Wraps `inner`, recording into `cache`, in [`CacheMode::ReadThrough`].
    pub fn new(inner: P, cache: C) -> Self {
        Self {
            inner,
            cache,
            mode: CacheMode::ReadThrough,
            clock: Arc::new(SystemClock),
            max_age: None,
        }
    }

    /// Sets the caching mode.
    pub fn with_mode(mut self, mode: CacheMode) -> Self {
        self.mode = mode;
        self
    }

    /// Treats a cached entry older than `max_age` as stale, re-querying it in
    /// [`CacheMode::ReadThrough`].
    ///
    /// There is no default, on purpose. Only the caller knows whether their
    /// evidence goes stale: the URL of a 1980 paper does not rot, while "the
    /// current stable release of Rust" rots in weeks. A crate-wide default TTL
    /// would be a guess imposed on every caller, and the failure mode of
    /// guessing too long is a silently outdated citation.
    ///
    /// Ignored in [`CacheMode::Replay`], where reproducibility outranks
    /// freshness.
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Uses `clock` to judge staleness. Only meaningful with
    /// [`with_max_age`](Self::with_max_age).
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// The provider underneath.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// The cache this provider records into.
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Whether a cached response has aged past `max_age`.
    fn is_stale(&self, response: &SearchResponse) -> bool {
        match self.max_age {
            // Note the direction: staleness is measured from the response's
            // *original* retrieval time. Anything else would make an entry
            // immortal simply by being read.
            Some(max_age) => self.clock.now() - response.retrieved_at() > max_age,
            None => false,
        }
    }
}

impl<P: SearchProvider, C: SearchCache> SearchProvider for CachedProvider<P, C> {
    fn name(&self) -> &str {
        // The cache is transparent: results must be attributed to the engine
        // that actually produced them, not to the wrapper. A provenance record
        // reading `engine: "cached"` would tell a future reader nothing about
        // who made the claim.
        self.inner.name()
    }

    fn search(&self, query: &SearchQuery) -> Result<SearchResponse, SearchError> {
        let key = CacheKey::new(self.inner.name(), query);

        if self.mode != CacheMode::Refresh
            && let Some(mut recorded) = self.cache.get(&key)?
        {
            let stale = self.mode != CacheMode::Replay && self.is_stale(&recorded);
            if !stale {
                // The one line that matters in this file: the recorded response
                // is handed back exactly as it was recorded. Its retrieved_at is
                // the moment the *engine* answered -- not now, not the moment the
                // cache was read. Only `from_cache` is set, and only because it
                // describes this delivery rather than the evidence.
                recorded.mark_from_cache();
                return Ok(recorded);
            }
        }

        if self.mode == CacheMode::Replay {
            return Err(SearchError::CacheMiss {
                provider: self.inner.name().to_string(),
                query: query.text().to_string(),
            });
        }

        // A failure here propagates untouched and unrecorded: the caller needs
        // to know it could not search, and the cache must not learn it.
        let fresh = self.inner.search(query)?;
        self.cache.put(&key, &fresh)?;
        Ok(fresh)
    }
}

/// Lets `&C` be used wherever a [`SearchCache`] is wanted, so several providers
/// (or a recorder and a replayer, as in the tests) can share one cache without
/// an `Arc`.
impl<C: SearchCache + ?Sized> SearchCache for &C {
    fn get(&self, key: &CacheKey) -> Result<Option<SearchResponse>, SearchError> {
        (**self).get(key)
    }

    fn put(&self, key: &CacheKey, response: &SearchResponse) -> Result<(), SearchError> {
        (**self).put(key, response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{FixedClock, SteppingClock};
    use crate::provider::{NullProvider, StaticProvider};

    /// A provider whose every search carries a *different* timestamp, so that a
    /// cache which re-stamped its hits could not possibly hide.
    fn stepping_provider() -> StaticProvider {
        StaticProvider::new("stepping")
            .with_clock(Arc::new(SteppingClock::hourly_from_unix(1_700_000_000)))
            .with_results(
                "write-ahead logging",
                [("Write-ahead logging", "https://example.org/wal", "A durability technique.")],
            )
    }

    #[test]
    fn a_cache_hit_returns_the_original_retrieval_time_not_a_fresh_one() {
        // THE test of this crate. The inner provider's clock advances an hour on
        // every call, so a re-stamped result would come back an hour newer.
        let provider = CachedProvider::new(stepping_provider(), MemoryCache::new());
        let query = SearchQuery::new("write-ahead logging");

        let first = provider.search(&query).unwrap();
        assert!(!first.from_cache());
        assert_eq!(first.retrieved_at().timestamp(), 1_700_000_000);

        let second = provider.search(&query).unwrap();
        assert!(second.from_cache(), "the second call must be served from the cache");

        // Not 1_700_003_600. The web said this at 1_700_000_000, and it will go
        // on having said it at 1_700_000_000 forever.
        assert_eq!(second.retrieved_at().timestamp(), 1_700_000_000);
        assert_eq!(
            second.results()[0].provenance().retrieved_at().timestamp(),
            1_700_000_000,
            "per-result provenance must not be re-stamped either",
        );
    }

    #[test]
    fn a_cache_hit_is_byte_identical_to_the_original_but_for_the_delivery_flag() {
        let provider = CachedProvider::new(stepping_provider(), MemoryCache::new());
        let query = SearchQuery::new("write-ahead logging");

        let mut first = provider.search(&query).unwrap();
        let second = provider.search(&query).unwrap();

        assert_eq!(first.results(), second.results());
        first.mark_from_cache();
        assert_eq!(first, second);
    }

    #[test]
    fn replay_mode_reproduces_a_recorded_session_without_a_network() {
        // Record once...
        let cache = MemoryCache::new();
        let recorded = {
            let recorder = CachedProvider::new(stepping_provider(), &cache);
            recorder.search(&SearchQuery::new("write-ahead logging")).unwrap()
        };

        // ... then replay through a provider with the same name but *no
        // fixtures at all*. If replay ever fell through to it, the query would
        // find nothing; the only way to get the recorded hit back is from the
        // recording itself.
        //
        // (The name must match, and that is not an accident of the test: it is
        // the cache key doing its job. A recording made by one engine is not an
        // answer from another.)
        let replayed = CachedProvider::new(StaticProvider::new("stepping"), &cache)
            .with_mode(CacheMode::Replay)
            .search(&SearchQuery::new("write-ahead logging"))
            .unwrap();

        assert!(replayed.from_cache());
        assert_eq!(replayed.results().len(), 1);
        assert_eq!(replayed.results(), recorded.results());
        assert_eq!(replayed.retrieved_at(), recorded.retrieved_at());
    }

    #[test]
    fn replay_mode_refuses_to_go_to_the_network_on_a_miss() {
        let cache = MemoryCache::new();
        let provider = CachedProvider::new(stepping_provider(), &cache).with_mode(CacheMode::Replay);

        let error = provider
            .search(&SearchQuery::new("a query nobody recorded"))
            .expect_err("replay must not silently fall through to the inner provider");

        match error {
            SearchError::CacheMiss { query, provider } => {
                assert_eq!(query, "a query nobody recorded");
                assert_eq!(provider, "stepping");
            }
            other => panic!("expected a CacheMiss, got {other:?}"),
        }

        // And nothing was recorded, because nothing was searched.
        assert!(cache.is_empty());
    }

    #[test]
    fn a_failed_search_is_not_cached() {
        // A rate limit or a dead network is a fact about us, not about the web.
        // Caching one would replay the failure forever.
        let cache = MemoryCache::new();
        let provider = CachedProvider::new(NullProvider, &cache);

        assert!(provider.search(&SearchQuery::new("q")).is_err());
        assert!(cache.is_empty(), "an error must never enter the cache");
    }

    #[test]
    fn an_empty_result_set_is_cached_because_it_is_a_real_finding() {
        let cache = MemoryCache::new();
        let provider = CachedProvider::new(
            StaticProvider::new("stepping")
                .with_clock(Arc::new(SteppingClock::hourly_from_unix(1_700_000_000))),
            &cache,
        );

        let first = provider.search(&SearchQuery::new("nothing to find")).unwrap();
        assert!(first.found_nothing());
        assert_eq!(cache.len(), 1, "\"we looked and found nothing\" is worth recording");

        // And it replays with its original date, like any other evidence.
        let second = provider.search(&SearchQuery::new("nothing to find")).unwrap();
        assert!(second.from_cache());
        assert!(second.found_nothing());
        assert_eq!(second.retrieved_at().timestamp(), 1_700_000_000);
    }

    #[test]
    fn the_cached_provider_reports_the_engines_name_not_its_own() {
        let provider = CachedProvider::new(stepping_provider(), MemoryCache::new());
        assert_eq!(provider.name(), "stepping");

        // ... and so the provenance credits the engine that actually answered.
        let response = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        assert_eq!(response.results()[0].provenance().engine(), "stepping");
    }

    #[test]
    fn a_stale_entry_is_requeried_and_the_new_answer_is_dated_now() {
        let cache = MemoryCache::new();
        let inner = stepping_provider(); // hands out 1_700_000_000, then +1h, ...

        // "now" for the staleness check, well past the recorded entry.
        let observer = Arc::new(FixedClock::from_unix(1_700_000_000 + 86_400));
        let provider = CachedProvider::new(inner, &cache)
            .with_clock(observer)
            .with_max_age(Duration::hours(1));

        let first = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        assert_eq!(first.retrieved_at().timestamp(), 1_700_000_000);

        // A day old with a one-hour max age: stale, so we ask again -- and the
        // fresh answer honestly carries a *new* timestamp, because it is a new
        // retrieval. (Refreshing is allowed to move a timestamp forward. What is
        // forbidden is moving one forward *without asking again*.)
        let second = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        assert!(!second.from_cache());
        assert_eq!(second.retrieved_at().timestamp(), 1_700_003_600);
    }

    #[test]
    fn a_fresh_entry_is_not_requeried() {
        let cache = MemoryCache::new();
        let provider = CachedProvider::new(stepping_provider(), &cache)
            .with_clock(Arc::new(FixedClock::from_unix(1_700_000_060))) // one minute later
            .with_max_age(Duration::hours(1));

        provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        let second = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        assert!(second.from_cache());
        assert_eq!(second.retrieved_at().timestamp(), 1_700_000_000);
    }

    #[test]
    fn replay_ignores_staleness_because_reproducibility_outranks_freshness() {
        let cache = MemoryCache::new();
        CachedProvider::new(stepping_provider(), &cache)
            .search(&SearchQuery::new("write-ahead logging"))
            .unwrap();

        // Ten years later, in replay mode, with a one-second max age.
        let replay = CachedProvider::new(StaticProvider::new("stepping"), &cache)
            .with_mode(CacheMode::Replay)
            .with_clock(Arc::new(FixedClock::from_unix(2_000_000_000)))
            .with_max_age(Duration::seconds(1));

        let response = replay.search(&SearchQuery::new("write-ahead logging")).unwrap();
        assert!(response.from_cache());
        assert_eq!(response.retrieved_at().timestamp(), 1_700_000_000);
    }

    #[test]
    fn refresh_mode_always_asks_again() {
        let cache = MemoryCache::new();
        let provider =
            CachedProvider::new(stepping_provider(), &cache).with_mode(CacheMode::Refresh);

        let first = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();
        let second = provider.search(&SearchQuery::new("write-ahead logging")).unwrap();

        assert!(!first.from_cache());
        assert!(!second.from_cache());
        assert_eq!(first.retrieved_at().timestamp(), 1_700_000_000);
        assert_eq!(second.retrieved_at().timestamp(), 1_700_003_600);
    }

    #[test]
    fn the_store_cache_survives_the_process_that_wrote_it() {
        let dir = tempfile::tempdir().unwrap();
        let query = SearchQuery::new("write-ahead logging");

        let recorded = {
            let cache = StoreCache::open(dir.path()).unwrap();
            CachedProvider::new(stepping_provider(), cache)
                .search(&query)
                .unwrap()
        };

        // A different process, a different day, no network: replay from disk.
        let cache = StoreCache::open(dir.path()).unwrap();
        let replayed = CachedProvider::new(StaticProvider::new("stepping"), cache)
            .with_mode(CacheMode::Replay)
            .search(&query)
            .unwrap();

        assert!(replayed.from_cache());
        assert_eq!(replayed.retrieved_at(), recorded.retrieved_at());
        assert_eq!(replayed.results(), recorded.results());
        assert_eq!(
            replayed.results()[0].provenance().content_hash(),
            recorded.results()[0].provenance().content_hash(),
        );
    }

    #[test]
    fn two_engines_never_read_each_others_recordings() {
        let cache = MemoryCache::new();
        let query = SearchQuery::new("write-ahead logging");

        CachedProvider::new(stepping_provider(), &cache)
            .search(&query)
            .unwrap();

        // Another engine, same query. "Brave said X" is not "SearXNG said X",
        // and a replay must not pretend otherwise.
        let other = CachedProvider::new(StaticProvider::new("other-engine"), &cache)
            .with_mode(CacheMode::Replay);
        assert!(matches!(
            other.search(&query),
            Err(SearchError::CacheMiss { .. })
        ));
    }
}
