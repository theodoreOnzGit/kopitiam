use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::EntityId;

/// The kind of a node in the semantic graph.
///
/// This is the "Common Semantic Model" from the Semantic Runtime vision:
/// every knowledge provider (Rust, documents, future language adapters)
/// emits entities using these kinds, regardless of the source language
/// or document format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    /// A buildable/versioned unit: a crate, package, module, or file.
    Artifact,
    /// A named code element: function, struct, trait, impl, etc.
    Symbol,
    /// A structural unit of a document: a heading, equation, or table.
    Section,
    /// A deterministic, tool-derived observation (e.g. "function X has no tests").
    Fact,
    /// A model-generated condensation of one or more entities.
    Summary,
    /// A recorded architectural or engineering decision (e.g. an ADR).
    Decision,
    /// A unit of planned or tracked work.
    Task,
}

/// A node in the semantic graph.
///
/// `source` records which knowledge provider produced this entity (e.g.
/// `"cargo-metadata"`, `"rust-analyzer"`, `"kopitiam-pdf"`), preserving the
/// provenance required by the Scientific Standards in CLAUDE.md and letting
/// consumers judge how much to trust a fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    pub name: String,
    pub source: String,
    #[serde(default)]
    pub metadata: Value,
}

impl Entity {
    pub fn new(kind: EntityKind, name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id: EntityId::new(),
            kind,
            name: name.into(),
            source: source.into(),
            metadata: Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_entity_with_defaults() {
        let e = Entity::new(EntityKind::Artifact, "kopitiam-ontology", "cargo-metadata");
        assert_eq!(e.kind, EntityKind::Artifact);
        assert_eq!(e.name, "kopitiam-ontology");
        assert_eq!(e.metadata, Value::Null);
    }

    #[test]
    fn round_trips_through_json() {
        let e = Entity::new(EntityKind::Symbol, "parse_pdf", "rust-analyzer")
            .with_metadata(serde_json::json!({ "line": 42 }));
        let json = serde_json::to_string(&e).unwrap();
        let back: Entity = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }
}
