use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::Value;

use crate::lsp_client::{LspClient, binary_available};
use crate::provider::{KnowledgeProvider, ProviderOutput};

/// Facts derived from a real rust-analyzer instance via the Language Server
/// Protocol: one [`Entity`] per symbol reported by `workspace/symbol`,
/// `LocatedIn` relationships to the artifact (file) that defines it.
///
/// If the configured binary is not installed, `collect` degrades to
/// [`ProviderOutput::empty`] instead of failing — rust-analyzer is a
/// deterministic source of truth when present, not a hard requirement.
pub struct RustAnalyzerProvider {
    binary: String,
}

impl RustAnalyzerProvider {
    pub fn new() -> Self {
        Self {
            binary: "rust-analyzer".to_string(),
        }
    }

    pub fn with_binary(binary: impl Into<String>) -> Self {
        Self {
            binary: binary.into(),
        }
    }
}

impl Default for RustAnalyzerProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeProvider for RustAnalyzerProvider {
    fn name(&self) -> &str {
        "rust-analyzer"
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        if !binary_available(&self.binary) {
            tracing::warn!(
                binary = %self.binary,
                "rust-analyzer not found on PATH; skipping (no facts collected)"
            );
            return Ok(ProviderOutput::empty());
        }

        let mut client = LspClient::spawn(&self.binary, root, std::time::Duration::from_secs(180))?;
        let symbols = client.workspace_symbols("")?;
        let _ = client.shutdown();

        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let mut artifacts: HashMap<String, EntityId> = HashMap::new();

        for symbol in symbols {
            let Some(name) = symbol.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let uri = symbol
                .pointer("/location/uri")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let line = symbol.pointer("/location/range/start/line").and_then(Value::as_u64);
            let lsp_kind = symbol.get("kind").and_then(Value::as_u64);
            let container = symbol.get("containerName").and_then(Value::as_str);

            let artifact_id = *artifacts.entry(uri.clone()).or_insert_with(|| {
                let artifact = Entity::new(EntityKind::Artifact, uri.clone(), "rust-analyzer");
                let id = artifact.id;
                entities.push(artifact);
                id
            });

            let symbol_entity = Entity::new(EntityKind::Symbol, name, "rust-analyzer").with_metadata(
                serde_json::json!({
                    "uri": uri,
                    "line": line,
                    "lsp_kind": lsp_kind,
                    "container": container,
                }),
            );
            let symbol_id = symbol_entity.id;
            entities.push(symbol_entity);
            relationships.push(Relationship::new(symbol_id, artifact_id, RelationshipKind::LocatedIn));
        }

        Ok(ProviderOutput {
            entities,
            relationships,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_symbols_from_this_workspace_when_rust_analyzer_is_installed() {
        if !binary_available("rust-analyzer") {
            eprintln!("skipping: rust-analyzer not installed in this environment");
            return;
        }

        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir.parent().and_then(Path::parent).expect("workspace root");

        let output = RustAnalyzerProvider::new()
            .collect(workspace_root)
            .expect("rust-analyzer collection should succeed");

        assert!(
            output.entities.iter().any(|e| e.kind == EntityKind::Symbol),
            "expected at least one Symbol entity from rust-analyzer"
        );
        assert!(
            output.entities.iter().any(|e| e.kind == EntityKind::Artifact),
            "expected at least one Artifact (file) entity from rust-analyzer"
        );
    }

    #[test]
    fn degrades_gracefully_when_binary_missing() {
        let root = std::env::temp_dir();
        let output = RustAnalyzerProvider::with_binary("kopitiam-nonexistent-lsp-binary")
            .collect(&root)
            .expect("missing binary must not error");
        assert!(output.entities.is_empty());
        assert!(output.relationships.is_empty());
    }
}
