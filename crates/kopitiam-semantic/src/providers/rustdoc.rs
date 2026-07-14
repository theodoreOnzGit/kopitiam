use std::collections::HashSet;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use kopitiam_ontology::{Entity, EntityKind, Relationship, RelationshipKind};
use rustdoc_types::{Crate, ItemEnum};

use crate::provider::{KnowledgeProvider, ProviderOutput};

/// Facts derived from real `rustdoc` JSON output (`cargo rustdoc -- -Z
/// unstable-options --output-format json`).
///
/// Rustdoc JSON output is nightly-only as of this writing (tracking issue:
/// rust-lang/rust#76578). This workspace's `rust-toolchain.toml` pins
/// `stable`, so on a typical KOPITIAM checkout this provider has nothing to
/// run against. It detects that up front and degrades to
/// [`ProviderOutput::empty`] rather than failing collection — consistent
/// with "cloud/optional tooling is an accelerator, not a requirement".
/// Install a nightly toolchain (`rustup toolchain install nightly`) to
/// exercise the real path.
pub struct RustdocProvider {
    toolchain: String,
}

impl RustdocProvider {
    pub fn new() -> Self {
        Self {
            toolchain: "nightly".to_string(),
        }
    }

    pub fn with_toolchain(toolchain: impl Into<String>) -> Self {
        Self {
            toolchain: toolchain.into(),
        }
    }
}

impl Default for RustdocProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Checks whether `toolchain` is already installed, without ever invoking
/// `cargo +toolchain` (or `rustup run`) to do so: some rustup
/// configurations auto-install a missing toolchain the moment it is
/// invoked, which would turn a mere availability check into an unrequested
/// multi-hundred-megabyte download. Querying `rustup toolchain list`
/// instead has no such side effect.
fn toolchain_available(toolchain: &str) -> bool {
    let Ok(output) = Command::new("rustup").args(["toolchain", "list"]).output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line.split_whitespace().next().is_some_and(|name| name.starts_with(toolchain)))
}

impl KnowledgeProvider for RustdocProvider {
    fn name(&self) -> &str {
        "rustdoc-json"
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        if !toolchain_available(&self.toolchain) {
            tracing::warn!(
                toolchain = %self.toolchain,
                "no `{}` toolchain available; rustdoc JSON output requires -Z unstable-options \
                 (nightly-only). Skipping (no facts collected).",
                self.toolchain
            );
            return Ok(ProviderOutput::empty());
        }

        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(root.join("Cargo.toml"))
            .exec()
            .context("running `cargo metadata`")?;
        let workspace_members: HashSet<_> = metadata.workspace_members.iter().cloned().collect();

        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        for package in &metadata.packages {
            if !workspace_members.contains(&package.id) {
                continue;
            }
            let Some(lib_target) = package
                .targets
                .iter()
                .find(|t| t.kind.iter().any(|k| matches!(k, cargo_metadata::TargetKind::Lib)))
            else {
                continue; // rustdoc JSON is only emitted for library targets
            };

            let status = Command::new("cargo")
                .arg(format!("+{}", self.toolchain))
                .args(["rustdoc", "-p", package.name.as_str(), "--lib"])
                .arg("--target-dir")
                .arg(metadata.target_directory.as_str())
                .arg("--")
                .args(["-Z", "unstable-options", "--output-format", "json"])
                .current_dir(root)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .with_context(|| format!("running `cargo +{} rustdoc` for {}", self.toolchain, package.name))?;

            if !status.success() {
                tracing::warn!(package = %package.name, "cargo rustdoc failed; skipping this package");
                continue;
            }

            let json_path = metadata
                .target_directory
                .join("doc")
                .join(format!("{}.json", lib_target.name.replace('-', "_")));
            let contents = match std::fs::read_to_string(&json_path) {
                Ok(contents) => contents,
                Err(err) => {
                    tracing::warn!(path = %json_path, %err, "rustdoc JSON output not found; skipping");
                    continue;
                }
            };
            let krate: Crate = match serde_json::from_str(&contents) {
                Ok(krate) => krate,
                Err(err) => {
                    tracing::warn!(
                        package = %package.name,
                        %err,
                        "rustdoc JSON did not match the `rustdoc-types` schema this crate was built \
                         against (installed nightly vs. crate version drift); skipping"
                    );
                    continue;
                }
            };

            let artifact = Entity::new(EntityKind::Artifact, package.name.to_string(), self.name());
            let artifact_id = artifact.id;
            entities.push(artifact);

            for item in krate.index.values() {
                let Some(name) = &item.name else { continue };
                let Some(item_kind) = item_kind_label(&item.inner) else {
                    continue;
                };
                let symbol = Entity::new(EntityKind::Symbol, name.clone(), self.name()).with_metadata(
                    serde_json::json!({
                        "item_kind": item_kind,
                        "docs": item.docs,
                        "visibility": format!("{:?}", item.visibility),
                    }),
                );
                relationships.push(Relationship::new(symbol.id, artifact_id, RelationshipKind::LocatedIn));
                entities.push(symbol);
            }
        }

        Ok(ProviderOutput {
            entities,
            relationships,
        })
    }
}

fn item_kind_label(inner: &ItemEnum) -> Option<&'static str> {
    Some(match inner {
        ItemEnum::Function(_) => "function",
        ItemEnum::Struct(_) => "struct",
        ItemEnum::Enum(_) => "enum",
        ItemEnum::Trait(_) => "trait",
        ItemEnum::Module(_) => "module",
        ItemEnum::TypeAlias(_) => "type_alias",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degrades_gracefully_without_a_nightly_toolchain() {
        let provider = RustdocProvider::with_toolchain("kopitiam-nonexistent-toolchain-xyz");
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir.parent().and_then(Path::parent).expect("workspace root");

        let output = provider
            .collect(workspace_root)
            .expect("a missing toolchain must not error");
        assert!(output.entities.is_empty());
        assert!(output.relationships.is_empty());
    }
}
