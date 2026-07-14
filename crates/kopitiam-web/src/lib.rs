//! Web search for KOPITIAM — **the last resort, never the first**.
//!
//! # Read this before you use this crate
//!
//! CLAUDE.md's Offline First principle fixes the order in which KOPITIAM is
//! allowed to look for an answer:
//!
//! 1. Existing knowledge (the semantic graph, translation memory, notes)
//! 2. A native Rust implementation
//! 3. Local AI
//! 4. Cloud AI
//!
//! The web sits *below* all four. It is the most expensive, least
//! reproducible, least trustworthy source in the platform:
//!
//! * it **requires a network**, which KOPITIAM promises never to require;
//! * it **costs money** (most usable search APIs are metered);
//! * it is **non-deterministic** — the same query returns different results
//!   tomorrow, from an index nobody involved controls;
//! * its results are **unvetted** — an engine's job is to rank pages, not to
//!   establish that anything on them is true.
//!
//! KOPITIAM's whole thesis is that the runtime owns knowledge so models don't
//! have to rediscover it. Reaching for the web is an admission that the
//! runtime *didn't* know something. That is sometimes the right call — but it
//! should feel like a decision, not a reflex. The API here is shaped to make
//! that obvious rather than convenient.
//!
//! # The three things this crate exists to guarantee
//!
//! ### 1. Nothing in KOPITIAM may require the network
//!
//! [`SearchProvider`] is a trait, exactly as `kopitiam-ai`'s `ModelAdapter` is
//! a trait. The default build of this crate contains **no HTTP client at all**
//! (the live adapters sit behind the non-default `searxng` / `brave` cargo
//! features), and [`NullProvider`] gives every downstream caller a working,
//! honest implementation that needs nothing.
//!
//! ### 2. Determinism must be recoverable
//!
//! A live search is inherently non-deterministic, which sits badly with a
//! platform that demands deterministic behaviour. The resolution is the
//! **cache** ([`CachedProvider`]): every query and its results are recorded,
//! so a run can be replayed exactly ([`CacheMode::Replay`]) with no network.
//!
//! A cached result is not merely an optimization. It is a **provenance
//! record** — the evidence of what the web said, on the day you asked it.
//! That is the point; the speed is a side effect.
//!
//! ### 3. Provenance is mandatory
//!
//! Every [`SearchResult`] carries a [`Provenance`]: the query, the engine, the
//! **retrieval timestamp**, and a content hash of exactly what came back. It
//! is structurally impossible to construct a result without one.
//!
//! "This claim is supported by X" is meaningless unless you can say *when* X
//! said it. See [`SearchResult::citation`].
//!
//! # "No results" is not "could not search"
//!
//! This is the dangerous bug the crate is built to prevent, so it is worth
//! stating plainly. A caller handed an empty list will reasonably conclude
//! *nothing about this exists*. If that empty list was actually produced by a
//! missing API key, a rate limit, or an unplugged ethernet cable, the caller
//! has just been lied to — and, downstream, so has a model, a document, or a
//! scientist.
//!
//! The type system separates the two cases and never lets them meet:
//!
//! | Meaning | Value |
//! |---|---|
//! | The engine searched, and found nothing | `Ok(`[`SearchResponse`]`)` whose [`results`] is empty |
//! | We did not manage to search at all | `Err(`[`SearchError`]`)` |
//!
//! A [`SearchResponse`] cannot exist unless a search actually happened: it
//! carries the engine, the query and the retrieval time that prove it. So an
//! empty `SearchResponse` is a *positive finding* ("we looked, on this date,
//! with this engine, and there was nothing"), which is itself worth caching
//! and worth recording in the knowledge graph.
//!
//! [`results`]: SearchResponse::results
//!
//! # Layout
//!
//! ```text
//!   SearchProvider (trait)
//!     ├── NullProvider     — search is switched off; every call is an honest error
//!     ├── StaticProvider   — deterministic canned results, no network (for tests)
//!     ├── CachedProvider   — wraps any provider; records, replays, preserves timestamps
//!     ├── SearxngProvider  — live, feature = "searxng"  (self-hosted, no API key)
//!     └── BraveProvider    — live, feature = "brave"    (independent index, API key)
//! ```
//!
//! # A typical wiring
//!
//! ```
//! use kopitiam_web::{CachedProvider, MemoryCache, SearchProvider, SearchQuery, StaticProvider};
//!
//! // In production the inner provider would be a live one; offline it is a
//! // stub, and nothing further down the pipeline can tell the difference.
//! let inner = StaticProvider::new("offline").with_results(
//!     "unicode normalization",
//!     [(
//!         "Unicode equivalence",
//!         "https://en.wikipedia.org/wiki/Unicode_equivalence",
//!         "The property that some sequences of code points represent the same character...",
//!     )],
//! );
//! let provider = CachedProvider::new(inner, MemoryCache::new());
//!
//! let response = provider.search(&SearchQuery::new("unicode normalization"))?;
//! assert_eq!(response.results().len(), 1);
//!
//! // Every result knows where it came from and, crucially, *when*.
//! println!("{}", response.results()[0].citation());
//! # Ok::<(), kopitiam_web::SearchError>(())
//! ```

mod cache;
mod clock;
mod error;
mod hash;
mod ontology;
mod provider;
mod query;
mod rate_limit;
mod response;

#[cfg(feature = "http")]
mod http;

pub use cache::{CacheMode, CachedProvider, MemoryCache, SearchCache, StoreCache};
pub use clock::{Clock, FixedClock, SteppingClock, SystemClock};
pub use error::SearchError;
pub use hash::ContentHash;
pub use provider::{NullProvider, SearchProvider, StaticProvider};
pub use query::{CacheKey, SearchQuery};
pub use rate_limit::RateLimiter;
pub use response::{Provenance, SearchResponse, SearchResponseBuilder, SearchResult};

#[cfg(feature = "brave")]
pub use http::BraveProvider;
#[cfg(feature = "searxng")]
pub use http::SearxngProvider;
