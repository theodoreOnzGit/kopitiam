use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::hash::ContentHash;
use crate::query::SearchQuery;

/// Where a [`SearchResult`] came from, and when.
///
/// # Structurally impossible to omit
///
/// Every field is private and there is exactly one constructor, which demands
/// all of them. There is no `Default`, no builder that can skip a field, and no
/// `Option` anywhere. You cannot hold a `SearchResult` that does not know where
/// it came from, because you cannot build one — and serde will refuse to
/// deserialize a record whose provenance is missing rather than filling in a
/// plausible blank.
///
/// This is not defensive programming for its own sake. Web content is the least
/// trustworthy material that enters KOPITIAM's knowledge graph, and it is the
/// only material that can *change after we record it*. An unprovenanced web
/// snippet sitting in a knowledge graph is indistinguishable from a fact
/// derived deterministically by rust-analyzer, and that is precisely the
/// confusion the Scientific Standards section of CLAUDE.md exists to prevent.
///
/// # What the fields buy you
///
/// * `query` — what was actually asked. A snippet is an answer to a question;
///   detached from the question it is just a sentence.
/// * `engine` — who answered. "Brave said X" and "my SearXNG instance said X"
///   are different claims with different reliability.
/// * `retrieved_at` — **when**. The web changes underneath you. A source
///   without a retrieval date supports nothing: the page you are citing may
///   have said the opposite last week and may say the opposite next week.
/// * `content_hash` — *what exactly* was said, hashed. This is what makes the
///   citation checkable rather than merely asserted; see [`ContentHash`].
///
/// The URL is deliberately *not* duplicated here — it lives on the
/// [`SearchResult`] itself, since it is the thing being cited rather than
/// evidence about the citation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    query: String,
    engine: String,
    retrieved_at: DateTime<Utc>,
    content_hash: ContentHash,
}

impl Provenance {
    /// Records the provenance of one result.
    ///
    /// The only way to build one. All four facts are required.
    pub fn new(
        query: impl Into<String>,
        engine: impl Into<String>,
        retrieved_at: DateTime<Utc>,
        content_hash: ContentHash,
    ) -> Self {
        Self {
            query: query.into(),
            engine: engine.into(),
            retrieved_at,
            content_hash,
        }
    }

    /// The query text that produced this result.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// The engine that returned it (`"brave"`, `"searxng"`, `"static"`, ...).
    pub fn engine(&self) -> &str {
        &self.engine
    }

    /// When it was retrieved, in UTC.
    ///
    /// For a result served from the cache this is the timestamp of the
    /// *original* retrieval — never the moment the cache was read. See
    /// [`CachedProvider`](crate::CachedProvider).
    pub fn retrieved_at(&self) -> DateTime<Utc> {
        self.retrieved_at
    }

    /// A SHA-256 hash of the result exactly as the engine returned it.
    pub fn content_hash(&self) -> &ContentHash {
        &self.content_hash
    }

    /// `retrieved_at` as an RFC 3339 / ISO 8601 string, to the second.
    ///
    /// The form a human reads in a citation, and the form a document engine
    /// should embed.
    pub fn retrieved_at_rfc3339(&self) -> String {
        self.retrieved_at.to_rfc3339_opts(SecondsFormat::Secs, true)
    }
}

/// One hit: a page an engine believes is relevant, plus the evidence of how we
/// came to hear about it.
///
/// The fields are private and the only constructor is
/// [`SearchResponseBuilder::push`], which stamps [`Provenance`] itself from the
/// response's engine, query and retrieval time. A provider therefore *cannot*
/// produce a result whose provenance disagrees with the search that produced
/// it — the class of bug where an adapter is refactored, the loop that fills in
/// `engine` is dropped, and results start arriving unattributed simply cannot
/// be written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    title: String,
    url: String,
    snippet: String,
    rank: usize,
    provenance: Provenance,
}

impl SearchResult {
    /// The page title as the engine reported it.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The URL of the page.
    ///
    /// Note what this is *not*: it is not a promise that the page still exists,
    /// still says what the snippet says, or ever said it. It is the address at
    /// which, at [`Provenance::retrieved_at`], an engine claimed something was
    /// to be found.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The engine's excerpt of the page.
    ///
    /// This is the *engine's* summary, generated by the engine, and it is not
    /// the page. Do not treat it as the page's content; see
    /// [`SearchResponse`]'s note on fetching.
    pub fn snippet(&self) -> &str {
        &self.snippet
    }

    /// The engine's ranking, starting at 1.
    ///
    /// Preserved because rank is information — it is the engine's own statement
    /// of confidence — and because a reproduced search that returns the same
    /// pages in a different order has, in fact, changed.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Where this result came from, and when. Always present.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// A citation line fit to paste into a document, a commit message, or a
    /// paper.
    ///
    /// ```text
    /// Unicode equivalence. <https://en.wikipedia.org/wiki/Unicode_equivalence>
    ///   (searxng, query "unicode normalization", retrieved 2026-07-14T09:31:00Z,
    ///    sha256:1b2c...)
    /// ```
    ///
    /// The retrieval date is not decoration. A web citation without one is
    /// worthless, because the reader has no way to know which version of the
    /// page you read — and the hash lets them check whether they are now
    /// reading a different one.
    pub fn citation(&self) -> String {
        format!(
            "{}. <{}> ({}, query {:?}, retrieved {}, {})",
            self.title,
            self.url,
            self.provenance.engine(),
            self.provenance.query(),
            self.provenance.retrieved_at_rfc3339(),
            self.provenance.content_hash(),
        )
    }
}

/// The outcome of a search that **actually happened**.
///
/// This type is the crate's answer to the most dangerous bug in the domain:
/// confusing *"the engine found nothing"* with *"we never managed to ask"*. A
/// `SearchResponse` cannot be constructed except by a provider that performed a
/// search, and it carries the engine, the query and the retrieval time that
/// prove it. Therefore:
///
/// * `Ok(response)` with `response.results()` empty means **the engine searched
///   and there was nothing** — a real, dated, citable finding about the world,
///   worth caching and worth putting in the knowledge graph.
/// * `Err(`[`SearchError`]`)` means **no search occurred**. Nothing may be
///   concluded about the world from it.
///
/// A caller can safely act on the first and must never act on the second as
/// though it were the first.
///
/// [`SearchError`]: crate::SearchError
///
/// # This is not page content
///
/// The results carry the engine's *snippets*. Turning a URL into the readable
/// text of the page it points at is a separate and genuinely hard problem
/// (boilerplate removal, paywalls, JavaScript, PDFs — and `kopitiam-document`
/// already handles the last of those). This crate does not do it, and it does
/// not pretend to: a snippet is labelled a snippet. Nothing here will ever
/// invent page content it did not retrieve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResponse {
    query: SearchQuery,
    engine: String,
    retrieved_at: DateTime<Utc>,
    results: Vec<SearchResult>,
    /// Not serialized: whether *this particular delivery* came out of a cache.
    ///
    /// It is a property of how you got the response, not of the response, so it
    /// must not be persisted — a cached record that remembered it was "from
    /// cache" would be claiming something about a read that had not happened
    /// yet. `retrieved_at`, by contrast, *is* persisted and never changes.
    #[serde(skip)]
    from_cache: bool,
}

impl SearchResponse {
    /// Starts building the response to `query`, as answered by `engine` at
    /// `retrieved_at`.
    ///
    /// Providers call this and then [`SearchResponseBuilder::push`] once per
    /// hit; provenance is stamped for them.
    pub fn builder(
        query: &SearchQuery,
        engine: impl Into<String>,
        retrieved_at: DateTime<Utc>,
    ) -> SearchResponseBuilder {
        SearchResponseBuilder {
            response: SearchResponse {
                query: query.clone(),
                engine: engine.into(),
                retrieved_at,
                results: Vec::new(),
                from_cache: false,
            },
        }
    }

    /// The hits, in the engine's own ranking order.
    ///
    /// Empty means the engine looked and found nothing. It never means we
    /// failed to look — that is an `Err`.
    pub fn results(&self) -> &[SearchResult] {
        &self.results
    }

    /// Whether the engine found nothing.
    ///
    /// Named for what it means, rather than `is_empty()`, so that a reader of
    /// the calling code cannot mistake it for "we have no data".
    pub fn found_nothing(&self) -> bool {
        self.results.is_empty()
    }

    /// The query that was asked.
    pub fn query(&self) -> &SearchQuery {
        &self.query
    }

    /// The engine that answered.
    pub fn engine(&self) -> &str {
        &self.engine
    }

    /// When the engine answered — **not** when this response was handed to you.
    ///
    /// For a cache hit these differ, sometimes by months. That gap is the whole
    /// point: it tells the caller how stale the evidence is.
    pub fn retrieved_at(&self) -> DateTime<Utc> {
        self.retrieved_at
    }

    /// Whether this delivery was served from the cache rather than the network.
    ///
    /// Callers that must not act on stale evidence should check this together
    /// with [`retrieved_at`](Self::retrieved_at) — and note that "fresh" is a
    /// judgement only the caller can make. A ten-year-old paper's URL does not
    /// go stale; a version number does.
    pub fn from_cache(&self) -> bool {
        self.from_cache
    }

    /// Marks the response as cache-served. Crate-internal: only
    /// [`CachedProvider`](crate::CachedProvider) may say this.
    pub(crate) fn mark_from_cache(&mut self) {
        self.from_cache = true;
    }
}

/// Accumulates a [`SearchResponse`], stamping provenance on every result.
///
/// The builder holds the engine, the query and the retrieval timestamp *once*,
/// and derives each result's [`Provenance`] from them. That is what makes it
/// impossible for a provider to emit a result whose provenance contradicts the
/// search it came from.
#[derive(Debug)]
pub struct SearchResponseBuilder {
    response: SearchResponse,
}

impl SearchResponseBuilder {
    /// Appends one hit. Rank is assigned in insertion order, starting at 1.
    ///
    /// The content hash is computed here, over the result exactly as the engine
    /// gave it to us — before any normalization, trimming or rewriting we might
    /// be tempted to do later. Hashing a cleaned-up version would record a
    /// digest of something the engine never said.
    pub fn push(
        &mut self,
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
    ) -> &mut Self {
        let (title, url, snippet) = (title.into(), url.into(), snippet.into());
        let content_hash = ContentHash::of_result(&title, &url, &snippet);
        let rank = self.response.results.len() + 1;

        self.response.results.push(SearchResult {
            title,
            url,
            snippet,
            rank,
            provenance: Provenance::new(
                self.response.query.text(),
                &self.response.engine,
                self.response.retrieved_at,
                content_hash,
            ),
        });
        self
    }

    /// Finishes the response.
    ///
    /// A response with no results is perfectly valid and is *not* an error: it
    /// is the dated, attributable finding that the engine had nothing.
    pub fn finish(self) -> SearchResponse {
        self.response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{Clock, FixedClock};

    fn a_response() -> SearchResponse {
        let query = SearchQuery::new("unicode normalization");
        let at = FixedClock::from_unix(1_752_486_660).now(); // 2025-07-14T09:51:00Z
        let mut builder = SearchResponse::builder(&query, "searxng", at);
        builder
            .push(
                "Unicode equivalence",
                "https://en.wikipedia.org/wiki/Unicode_equivalence",
                "The property that some code point sequences represent the same character.",
            )
            .push(
                "Normalization forms in Unicode",
                "https://example.org/nfc",
                "A practical introduction with worked NFC and NFD examples.",
            );
        builder.finish()
    }

    #[test]
    fn every_result_carries_the_provenance_of_the_search_that_produced_it() {
        let response = a_response();
        for result in response.results() {
            let provenance = result.provenance();
            assert_eq!(provenance.query(), "unicode normalization");
            assert_eq!(provenance.engine(), "searxng");
            assert_eq!(provenance.retrieved_at(), response.retrieved_at());
        }
    }

    #[test]
    fn ranks_are_assigned_in_the_engines_order_starting_at_one() {
        let response = a_response();
        assert_eq!(response.results()[0].rank(), 1);
        assert_eq!(response.results()[1].rank(), 2);
    }

    #[test]
    fn the_content_hash_is_of_what_the_engine_actually_said() {
        let response = a_response();
        let first = &response.results()[0];
        assert_eq!(
            first.provenance().content_hash(),
            &ContentHash::of_result(first.title(), first.url(), first.snippet()),
        );
    }

    #[test]
    fn a_citation_names_the_engine_and_the_retrieval_date() {
        let citation = a_response().results()[0].citation();
        assert!(citation.contains("searxng"), "{citation}");
        assert!(citation.contains("2025-07-14T09:51:00Z"), "{citation}");
        assert!(citation.contains("sha256:"), "{citation}");
        assert!(citation.contains("https://en.wikipedia.org/wiki/Unicode_equivalence"));
    }

    #[test]
    fn an_empty_response_is_a_finding_not_a_failure() {
        let query = SearchQuery::new("a query with no hits at all");
        let at = FixedClock::from_unix(1_752_486_660).now();
        let response = SearchResponse::builder(&query, "searxng", at).finish();

        // It found nothing -- but it is still a dated, attributable statement
        // about the world, and it says so.
        assert!(response.found_nothing());
        assert_eq!(response.engine(), "searxng");
        assert_eq!(response.retrieved_at(), at);
        assert_eq!(response.query().text(), "a query with no hits at all");
    }

    #[test]
    fn a_result_cannot_be_deserialized_without_its_provenance() {
        // The compiler stops you constructing one; serde must stop you smuggling
        // one in through the cache, a config file, or a hand-edited JSON blob.
        let unprovenanced = r#"{
            "title": "Unicode equivalence",
            "url": "https://example.org/nfc",
            "snippet": "...",
            "rank": 1
        }"#;
        let error = serde_json::from_str::<SearchResult>(unprovenanced)
            .expect_err("a result without provenance must not deserialize");
        assert!(error.to_string().contains("provenance"), "{error}");
    }

    #[test]
    fn from_cache_is_never_persisted() {
        // It describes how you obtained the response, not what the response is.
        let mut response = a_response();
        response.mark_from_cache();
        assert!(response.from_cache());

        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains("from_cache"));

        let back: SearchResponse = serde_json::from_str(&json).unwrap();
        assert!(!back.from_cache(), "a stored response must not claim to be a cache hit");
        // ... but the retrieval timestamp, which *is* a property of the
        // response, survives untouched.
        assert_eq!(back.retrieved_at(), response.retrieved_at());
    }

    #[test]
    fn round_trips_through_json_with_provenance_intact() {
        let response = a_response();
        let json = serde_json::to_string(&response).unwrap();
        let back: SearchResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.results(), response.results());
    }
}
