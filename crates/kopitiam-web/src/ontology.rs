//! Turning retrieved web pages into entities in the shared knowledge graph.
//!
//! Without this, a search result evaporates the moment the process exits, and
//! the next session asks the web the same question again — which is precisely
//! the pattern KOPITIAM exists to eliminate ("Never allow valuable reasoning to
//! disappear into chat history"). With it, an expensive, metered, non-repeatable
//! retrieval becomes a permanent, dated, checkable node that every later
//! workflow can consult *before* reaching for the network again.

use kopitiam_ontology::{Entity, EntityKind};
use serde_json::json;

use crate::response::{SearchResponse, SearchResult};

impl SearchResult {
    /// Records this result as an [`Entity`] in the semantic graph.
    ///
    /// # Why [`EntityKind::Artifact`] and emphatically not [`EntityKind::Fact`]
    ///
    /// `Fact` is defined in `kopitiam-ontology` as "a deterministic,
    /// tool-derived observation" — the sort of thing rust-analyzer, `cargo
    /// metadata` or clippy produces, where the same input always yields the same
    /// output and the output is *true by construction*.
    ///
    /// A web search result is none of those things. It is non-deterministic (a
    /// different index, a different day, a different answer), it is unvetted (an
    /// engine ranks pages; it does not check them), and it is a *claim someone
    /// else made*, not an observation we derived. Filing it as a `Fact` would
    /// let unverified web content sit in the graph wearing the same badge as a
    /// symbol table extracted from the AST — and a later consumer, quite
    /// reasonably, would trust the two equally.
    ///
    /// So a retrieved page is an `Artifact`: a real thing that exists at a URL,
    /// which we saw, on a date, and about which the engine asserted something.
    /// What it *says* remains a claim until a human or a workflow promotes it.
    /// That boundary is the difference between a knowledge graph and a rumour
    /// mill.
    ///
    /// # The `source` field
    ///
    /// `kopitiam-web:<engine>` — so a graph query can find, isolate, and if
    /// necessary distrust every node that came from the web, or from one
    /// specific engine. Provenance you cannot filter on is decoration.
    pub fn to_entity(&self) -> Entity {
        let provenance = self.provenance();

        Entity::new(
            EntityKind::Artifact,
            self.title(),
            format!("kopitiam-web:{}", provenance.engine()),
        )
        .with_metadata(json!({
            "url": self.url(),
            // Named "snippet" rather than "content" or "text" on purpose: it is
            // the *engine's* excerpt, not the page. Nothing downstream should be
            // able to mistake it for the document.
            "snippet": self.snippet(),
            "rank": self.rank(),
            "query": provenance.query(),
            "engine": provenance.engine(),
            "retrieved_at": provenance.retrieved_at_rfc3339(),
            "content_hash": provenance.content_hash().to_string(),
            "citation": self.citation(),
        }))
    }
}

impl SearchResponse {
    /// Records every result as an [`Entity`], in rank order.
    ///
    /// Note what this does *not* do: an empty response produces no entities. The
    /// finding "we searched and found nothing" is real and worth keeping, but it
    /// belongs in the cache (where it is dated and replayable), not as a node
    /// asserting the existence of a page that does not exist.
    pub fn to_entities(&self) -> Vec<Entity> {
        self.results().iter().map(SearchResult::to_entity).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::clock::FixedClock;
    use crate::provider::{SearchProvider, StaticProvider};
    use crate::query::SearchQuery;

    fn a_response() -> SearchResponse {
        StaticProvider::new("searxng")
            .with_clock(Arc::new(FixedClock::from_unix(1_752_486_660)))
            .with_results(
                "write-ahead logging",
                [(
                    "Write-ahead logging",
                    "https://example.org/wal",
                    "A durability technique.",
                )],
            )
            .search(&SearchQuery::new("write-ahead logging"))
            .unwrap()
    }

    #[test]
    fn a_web_result_enters_the_graph_as_an_artifact_never_as_a_fact() {
        let entity = a_response().to_entities().remove(0);

        assert_eq!(entity.kind, EntityKind::Artifact);
        assert_ne!(
            entity.kind,
            EntityKind::Fact,
            "unvetted web content must not wear the same badge as a tool-derived fact",
        );
        assert_eq!(entity.name, "Write-ahead logging");
    }

    #[test]
    fn the_source_names_the_engine_so_the_graph_can_distrust_it_selectively() {
        let entity = a_response().to_entities().remove(0);
        assert_eq!(entity.source, "kopitiam-web:searxng");
    }

    #[test]
    fn every_provenance_field_survives_the_crossing_into_the_graph() {
        let entity = a_response().to_entities().remove(0);
        let metadata = &entity.metadata;

        assert_eq!(metadata["url"], "https://example.org/wal");
        assert_eq!(metadata["query"], "write-ahead logging");
        assert_eq!(metadata["engine"], "searxng");
        assert_eq!(metadata["rank"], 1);
        assert_eq!(metadata["retrieved_at"], "2025-07-14T09:51:00Z");
        assert!(
            metadata["content_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        // The engine's excerpt is labelled an excerpt, not the page's content.
        assert!(metadata.get("snippet").is_some());
        assert!(metadata.get("content").is_none());
    }

    #[test]
    fn an_empty_response_asserts_the_existence_of_nothing() {
        let response = StaticProvider::new("searxng")
            .search(&SearchQuery::new("nothing"))
            .unwrap();
        assert!(response.to_entities().is_empty());
    }
}
