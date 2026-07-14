//! Full-text/symbol search for KOPITIAM's Semantic Runtime.
//!
//! Backed by [`tantivy`], a pure-Rust search engine library, so this crate
//! stays inside the Pure Rust Core rule the way `kopitiam-index` does with
//! `redb` in place of SQLite. It indexes [`kopitiam_ontology::Entity`]
//! records — the same Common Semantic Model every knowledge provider and
//! `kopitiam-knowledge` already speak — rather than raw file text, so a
//! search hit is always traceable back to a typed, provenance-carrying
//! entity.

mod index;

pub use index::{SearchHit, SearchIndex};
