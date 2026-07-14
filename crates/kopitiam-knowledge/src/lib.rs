//! The Knowledge Engine: owns the unified semantic graph.
//!
//! Consumes facts from any `kopitiam_ontology`-speaking provider
//! (`kopitiam-semantic`'s Rust providers, document providers, future
//! language adapters) and lets callers query and traverse them. Storage and
//! model invocation are deliberately out of scope here — see
//! `kopitiam-index` for persistence and `kopitiam-workflow` for orchestration.

mod graph;

pub use graph::SemanticGraph;
