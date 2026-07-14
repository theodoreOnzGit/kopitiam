use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};

use crate::provider::{KnowledgeProvider, ProviderOutput};

/// Facts derived from `cargo metadata`: one [`Entity`] per workspace member
/// package, and `DependsOn` relationships between workspace members.
///
/// External (non-workspace) dependencies are intentionally not turned into
/// entities here — with hundreds of transitive crates that would swamp the
/// graph with facts nobody queries. The full dependency tree remains
/// available on demand via `cargo tree`; this provider only records the
/// internal dependency graph as a deterministic fact.
pub struct CargoMetadataProvider;

impl CargoMetadataProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CargoMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeProvider for CargoMetadataProvider {
    fn name(&self) -> &str {
        "cargo-metadata"
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(root.join("Cargo.toml"))
            .exec()
            .context("running `cargo metadata`")?;

        let workspace_members: std::collections::HashSet<_> =
            metadata.workspace_members.iter().cloned().collect();

        let mut entities = Vec::new();
        let mut ids: HashMap<cargo_metadata::PackageId, EntityId> = HashMap::new();

        for package in &metadata.packages {
            if !workspace_members.contains(&package.id) {
                continue;
            }
            let targets: Vec<_> = package
                .targets
                .iter()
                .map(|t| serde_json::json!({ "name": t.name, "kind": t.kind }))
                .collect();
            let entity = Entity::new(EntityKind::Artifact, package.name.to_string(), self.name())
                .with_metadata(serde_json::json!({
                    "version": package.version.to_string(),
                    "manifest_path": package.manifest_path.to_string(),
                    "targets": targets,
                }));
            ids.insert(package.id.clone(), entity.id);
            entities.push(entity);
        }

        let mut relationships = Vec::new();
        if let Some(resolve) = &metadata.resolve {
            for node in &resolve.nodes {
                let Some(&from) = ids.get(&node.id) else {
                    continue;
                };
                for dep in &node.dependencies {
                    if let Some(&to) = ids.get(dep) {
                        relationships.push(Relationship::new(from, to, RelationshipKind::DependsOn));
                    }
                }
            }
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
    fn collects_this_workspace() {
        // Walk up from CARGO_MANIFEST_DIR (crates/kopitiam-semantic) to the
        // workspace root so this test is independent of the crate's depth.
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");

        let output = CargoMetadataProvider::new()
            .collect(workspace_root)
            .expect("cargo metadata should succeed against this real workspace");

        assert!(
            output
                .entities
                .iter()
                .any(|e| e.name == "kopitiam-semantic"),
            "expected kopitiam-semantic to be reported as a workspace artifact"
        );
        assert!(
            output.entities.iter().any(|e| e.name == "kopitiam-ontology"),
            "expected kopitiam-ontology to be reported as a workspace artifact"
        );
        assert!(
            output.relationships.iter().any(|r| r.kind == RelationshipKind::DependsOn),
            "expected at least one internal DependsOn edge (kopitiam-semantic -> kopitiam-ontology)"
        );
    }
}
