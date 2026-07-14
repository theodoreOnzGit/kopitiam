use std::path::Path;

use kopitiam_ontology::{Entity, Relationship};

/// Facts produced by a single [`KnowledgeProvider`] run.
#[derive(Debug, Default, Clone)]
pub struct ProviderOutput {
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
}

impl ProviderOutput {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// A source of deterministic facts about a project.
///
/// Implementations shell out to or embed real tooling (cargo, rust-analyzer,
/// rustdoc, document parsers, ...). They never ask a model to infer anything
/// a tool can compute. A provider whose tool is unavailable in the current
/// environment (e.g. rustdoc JSON requires a nightly toolchain) must degrade
/// to [`ProviderOutput::empty`] rather than fail the whole collection run.
pub trait KnowledgeProvider {
    /// Stable identifier recorded as `Entity::source` / `Entity::metadata`
    /// provenance for every fact this provider emits.
    fn name(&self) -> &str;

    /// Collects facts about the project rooted at `root` (a directory
    /// containing a `Cargo.toml`, for Rust providers).
    fn collect(&self, root: &Path) -> anyhow::Result<ProviderOutput>;
}
