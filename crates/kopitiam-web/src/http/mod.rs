//! The live search adapters, and the HTTP plumbing they share.
//!
//! Everything in here is behind the non-default `http` cargo feature (enabled
//! transitively by `searxng` and `brave`). The default build of `kopitiam-web`
//! does not compile a byte of it, has no TLS stack, and cannot open a socket.
//! That is the Pure Rust Core and the Offline First promise being kept, not
//! merely asserted.
//!
//! # Which engines, and why
//!
//! The choice of search API is not a technical detail; it is a statement about
//! who KOPITIAM depends on. The full reasoning, including the ones we rejected,
//! is in `docs/ai-decisions/AID-0013`. In brief:
//!
//! | Engine | Key? | Cost | Self-hostable | Verdict |
//! |---|---|---|---|---|
//! | **SearXNG** | no | free | **yes** | **Implemented. Recommended.** |
//! | **Brave** | yes | free tier, then metered | no | **Implemented.** Independent index. |
//! | Tavily | yes | metered credits | no | Rejected: sells LLM-rewritten answers, which destroys provenance. |
//! | Serper / SerpAPI | yes | metered | no | Rejected: Google scraping proxies. Grey-area ToS, pure rent. |
//! | DuckDuckGo | — | — | no | Rejected: has no web-search API. Its "API" returns instant answers only. |
//!
//! **SearXNG is the recommended provider**, and the ordering above is not
//! alphabetical. It is a metasearch aggregator you run yourself: no API key, no
//! account, no bill, no vendor, no terms of service that can change under you,
//! and — happily — AGPL-3.0, the same licence as KOPITIAM. For a project whose
//! founding principle is that no external service may be *required*, an engine
//! the user owns outright is not merely a nice option; it is the only one that
//! is philosophically consistent. Renting Google through a proxy is the exact
//! dependency we are trying not to have.
//!
//! Brave is implemented alongside it as the pragmatic answer for someone who
//! does not want to run infrastructure, and because it is one of the very few
//! search APIs backed by a genuinely independent crawl rather than a resold
//! Google index — which makes it a real second opinion rather than a costlier
//! copy of the first.
//!
//! # TLS
//!
//! `ureq` with rustls, never OpenSSL. See the comments in `Cargo.toml` for the
//! exact feature flags and the honest caveat about rustls' crypto provider.

mod client;

#[cfg(feature = "brave")]
mod brave;
#[cfg(feature = "searxng")]
mod searxng;

#[cfg(feature = "brave")]
pub use brave::BraveProvider;
#[cfg(feature = "searxng")]
pub use searxng::SearxngProvider;
