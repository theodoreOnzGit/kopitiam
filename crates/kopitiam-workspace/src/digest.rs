//! Cached *architecture digest*: a compact, persistent "what does each crate
//! do" summary — crate → responsibility → workspace-internal dependency edges
//! — built once, read cheaply every session, and regenerated only when a
//! manifest hash changes (`kopitiam_token_max.md` §11 Task II-3, §252).
//!
//! ## The saving
//!
//! `scan` already runs `cargo metadata` and learns the whole crate graph, but
//! then **throws it away** (`kopitiam_token_max.md` §275): nothing is cached,
//! so every session an agent re-derives "which crate does what and depends on
//! what" with a full exploration pass. This module serialises that answer once
//! into the existing `.kopitiam` redb store and hands it back for free on
//! subsequent sessions.
//!
//! ## Invalidation is content, never time (§0.4)
//!
//! There is no clock here. The digest pins the FNV-1a [`source_hash`] of the
//! concatenated workspace `Cargo.toml` manifests at build time. A session
//! decides staleness by re-hashing those manifests — cheap, no `cargo` run —
//! and comparing ([`ArchitectureDigest::is_stale`]). Match ⇒ the cache is
//! trusted and `cargo metadata` is never invoked; mismatch (a manifest's bytes
//! drifted) ⇒ regenerate. Reading manifests is orders of magnitude cheaper than
//! running `cargo metadata`, which is the whole point of hashing manifests
//! rather than the metadata JSON.
//!
//! ## Decoupled from the `cargo` invocation, for testability
//!
//! [`build_digest`] is a **pure** function of the metadata JSON (and the
//! manifest bytes it hashes) — no process spawn, no filesystem — so it is unit
//! tested against a hand-built fixture with no live `cargo` run. The live path,
//! [`run_cargo_metadata`], is the thin wrapper that shells `cargo metadata` and
//! feeds its stdout to `build_digest`; the one test that exercises it is
//! `#[ignore]`d so `cargo test` stays hermetic.
//!
//! ## Additive persistence, no migration (§10.1)
//!
//! The digest lives under a new redb key, `workspace/architecture_digest`,
//! disjoint from `workspace/project_state` (Task I), `workspace/conclusions`
//! (Task II-5) and `workspace/translation_memory` (Task III-4). The KV table
//! already takes arbitrary serde_json blobs, so this needs no schema change.
//!
//! ## Deferred (follow-ups, not built here)
//!
//! - The CLI surface (`scan --digest` / a `digest` view) is owned by
//!   `apps/cli` and is out of scope for this crate.
//! - The richer "key types per crate" field needs rust-analyzer / rustdoc data
//!   from `scan`; responsibility + dependency edges is the substantive cache
//!   this wave. `CrateDigest` can gain a `key_types` field additively later.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use kopitiam_index::Store;
use serde::{Deserialize, Serialize};

/// redb key namespace for the architecture digest. Additive and disjoint from
/// every other `workspace/*` key, so this needs no store change and no
/// migration (§10.1).
const ARCHITECTURE_DIGEST_KEY: &str = "workspace/architecture_digest";

/// One workspace crate's entry in the digest: what it is, and which other
/// workspace crates it depends on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrateDigest {
    /// The crate (package) name, e.g. `kopitiam-index`.
    pub name: String,

    /// A one-line "what does this crate do" summary.
    ///
    /// **Source:** the crate manifest's `package.description` field, as
    /// reported by `cargo metadata`. When a crate declares no `description`
    /// this is the empty string (a later wave can fall back to the crate's
    /// top-level doc comment once `scan`'s rustdoc data is wired in — see the
    /// module docs' deferred note).
    pub responsibility: String,

    /// The names of the **workspace-internal** crates this crate depends on,
    /// sorted and de-duplicated. External (registry / git) dependencies are
    /// intentionally excluded — with hundreds of transitive crates they would
    /// swamp the digest with edges nobody queries (mirroring the
    /// `CargoMetadataProvider` policy). Dev-dependencies are excluded so the
    /// digest reflects the shipping architecture, not the test rig.
    pub dependencies: Vec<String>,

    /// The crate's directory (the parent of its `Cargo.toml`), as reported by
    /// `cargo metadata`'s `manifest_path`.
    pub path: String,
}

/// The whole workspace's architecture, plus the content hash that decides when
/// it must be rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureDigest {
    /// One entry per workspace member crate, sorted by name for a
    /// deterministic, diff-friendly serialisation.
    pub crates: Vec<CrateDigest>,

    /// FNV-1a hash of the concatenated workspace `Cargo.toml` manifests at
    /// build time — the invalidation key. Compared against a freshly computed
    /// [`source_hash`] to decide staleness (§0.4); **never** a timestamp.
    pub source_hash: String,
}

impl ArchitectureDigest {
    /// Loads the cached digest for `root`'s `.kopitiam` directory, or `None` if
    /// none has been built yet.
    pub fn load_or_none(root: &Path) -> Result<Option<Self>> {
        let store = Store::open(root)?;
        store.get_json(ARCHITECTURE_DIGEST_KEY)
    }

    /// Persists this digest to `root`'s `.kopitiam` store under the
    /// [`ARCHITECTURE_DIGEST_KEY`] key, overwriting any previous digest.
    pub fn store(&self, root: &Path) -> Result<()> {
        let store = Store::open(root)?;
        store.put_json(ARCHITECTURE_DIGEST_KEY, self)
    }

    /// Whether this cached digest is stale for the current manifests: `true`
    /// iff the FNV-1a hash of `current_manifest_bytes` differs from the
    /// [`source_hash`](ArchitectureDigest::source_hash) pinned at build time.
    ///
    /// The caller assembles `current_manifest_bytes` by concatenating the
    /// workspace `Cargo.toml` manifests — a cheap read that avoids spawning
    /// `cargo metadata` just to check freshness. A stale digest should be
    /// regenerated via [`run_cargo_metadata`] + [`build_digest`]; a fresh one
    /// is trusted indefinitely (§0.4).
    pub fn is_stale(&self, current_manifest_bytes: &[u8]) -> bool {
        self.source_hash != source_hash(current_manifest_bytes)
    }
}

/// The FNV-1a content hash used as the digest's invalidation key.
///
/// Reuses the workspace's shared [`content_hash`](crate::content_hash) (the
/// same fixed-constant FNV-1a as Tasks II-5 / III-4), so persisted hashes are
/// stable across toolchains and sessions and no new dependency is pulled in.
pub fn source_hash(manifest_bytes: &[u8]) -> String {
    crate::content_hash(manifest_bytes)
}

// ---- The pure JSON → digest transform -------------------------------------

/// The slice of `cargo metadata --format-version 1` we deserialise. Only the
/// fields the digest needs are named; serde ignores the rest.
#[derive(Deserialize)]
struct RawMetadata {
    packages: Vec<RawPackage>,
    /// Package **ids** of the workspace's own members (not external crates).
    workspace_members: Vec<String>,
}

#[derive(Deserialize)]
struct RawPackage {
    name: String,
    id: String,
    /// `null` in the JSON when the manifest declares no description.
    #[serde(default)]
    description: Option<String>,
    manifest_path: String,
    #[serde(default)]
    dependencies: Vec<RawDependency>,
}

#[derive(Deserialize)]
struct RawDependency {
    name: String,
    /// `null` for a normal dependency, `"dev"` or `"build"` otherwise.
    #[serde(default)]
    kind: Option<String>,
}

/// Builds an [`ArchitectureDigest`] from `cargo metadata --format-version 1`
/// JSON and the workspace manifest bytes to hash.
///
/// This is a **pure** function — no process spawn, no filesystem — so it is
/// unit tested against a fixture with no live `cargo` run (§11 acceptance). The
/// live wrapper [`run_cargo_metadata`] supplies `metadata_json` at runtime.
///
/// - `metadata_json`: stdout of `cargo metadata --format-version 1`
///   (`--no-deps` is fine; the internal edges are read from each package's own
///   `dependencies` list, not from the resolved graph).
/// - `manifest_bytes`: the concatenated workspace `Cargo.toml` manifests,
///   hashed into [`ArchitectureDigest::source_hash`] as the invalidation key.
///
/// Only workspace members become [`CrateDigest`] entries; `dependencies` are
/// filtered to other workspace members (external and dev deps dropped). Output
/// is deterministic: crates sorted by name, each crate's deps sorted and
/// de-duplicated.
pub fn build_digest(metadata_json: &str, manifest_bytes: &[u8]) -> Result<ArchitectureDigest> {
    let raw: RawMetadata =
        serde_json::from_str(metadata_json).context("parsing `cargo metadata` JSON")?;

    // Workspace members are listed by package id; resolve those ids to the set
    // of member names so a dependency edge can be classified internal/external
    // by name.
    let member_ids: HashSet<&str> = raw.workspace_members.iter().map(String::as_str).collect();
    let member_names: HashSet<&str> = raw
        .packages
        .iter()
        .filter(|p| member_ids.contains(p.id.as_str()))
        .map(|p| p.name.as_str())
        .collect();

    let mut crates = Vec::new();
    for pkg in &raw.packages {
        if !member_ids.contains(pkg.id.as_str()) {
            continue; // external crate — not part of the workspace's architecture
        }

        let mut dependencies: Vec<String> = pkg
            .dependencies
            .iter()
            .filter(|d| d.kind.as_deref() != Some("dev")) // shipping deps only
            .filter(|d| member_names.contains(d.name.as_str())) // workspace-internal only
            .filter(|d| d.name != pkg.name) // never a self-edge
            .map(|d| d.name.clone())
            .collect();
        dependencies.sort();
        dependencies.dedup();

        let path = Path::new(&pkg.manifest_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| pkg.manifest_path.clone());

        crates.push(CrateDigest {
            name: pkg.name.clone(),
            responsibility: pkg.description.clone().unwrap_or_default(),
            dependencies,
            path,
        });
    }
    crates.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(ArchitectureDigest {
        crates,
        source_hash: source_hash(manifest_bytes),
    })
}

/// Runs `cargo metadata --format-version 1 --no-deps` in `root` and returns its
/// stdout for [`build_digest`].
///
/// This is the only part of the module that touches a live `cargo`, so it is
/// kept thin and out of the unit tests (the one test that calls it is
/// `#[ignore]`d). `--no-deps` keeps the output to the workspace's own packages;
/// internal dependency edges are still present on each package's `dependencies`
/// list, which is all `build_digest` reads.
pub fn run_cargo_metadata(root: &Path) -> Result<String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .context("spawning `cargo metadata`")?;

    if !output.status.success() {
        bail!(
            "`cargo metadata` failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    String::from_utf8(output.stdout).context("`cargo metadata` output was not UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built `cargo metadata --format-version 1` fixture: three
    /// workspace members (`alpha`, `beta`, `gamma`) plus one external crate
    /// (`serde`, *not* a workspace member). `alpha` normal-depends on `beta`
    /// and `serde` and dev-depends on `gamma`; `gamma` declares no description.
    /// This exercises member filtering, external-dep exclusion, dev-dep
    /// exclusion, and the empty-description fallback — with no live `cargo`.
    const FIXTURE: &str = r#"{
        "workspace_members": [
            "path+file:///w/alpha#0.1.0",
            "path+file:///w/beta#0.1.0",
            "path+file:///w/gamma#0.1.0"
        ],
        "packages": [
            {
                "name": "alpha",
                "id": "path+file:///w/alpha#0.1.0",
                "description": "Alpha drives the front end.",
                "manifest_path": "/w/alpha/Cargo.toml",
                "dependencies": [
                    { "name": "beta", "kind": null },
                    { "name": "serde", "kind": null },
                    { "name": "gamma", "kind": "dev" }
                ]
            },
            {
                "name": "beta",
                "id": "path+file:///w/beta#0.1.0",
                "description": "Beta stores things.",
                "manifest_path": "/w/beta/Cargo.toml",
                "dependencies": [
                    { "name": "serde", "kind": null }
                ]
            },
            {
                "name": "gamma",
                "id": "path+file:///w/gamma#0.1.0",
                "description": null,
                "manifest_path": "/w/gamma/Cargo.toml",
                "dependencies": []
            },
            {
                "name": "serde",
                "id": "registry+https://github.com/rust-lang/crates.io-index#serde@1.0.0",
                "description": "A serialization framework.",
                "manifest_path": "/reg/serde/Cargo.toml",
                "dependencies": []
            }
        ]
    }"#;

    const MANIFESTS: &[u8] = b"[workspace]\nmembers = [\"alpha\", \"beta\", \"gamma\"]\n";

    #[test]
    fn build_digest_yields_expected_crates_responsibilities_and_edges() {
        let digest = build_digest(FIXTURE, MANIFESTS).unwrap();

        // Only the three workspace members, sorted by name; `serde` is dropped.
        let names: Vec<&str> = digest.crates.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta", "gamma"]);

        let alpha = &digest.crates[0];
        assert_eq!(alpha.responsibility, "Alpha drives the front end.");
        // `serde` (external) and `gamma` (dev-dep) are excluded; only `beta`
        // remains as a workspace-internal shipping edge.
        assert_eq!(alpha.dependencies, ["beta"]);
        assert!(alpha.path.ends_with("alpha"), "path was {}", alpha.path);

        let beta = &digest.crates[1];
        assert_eq!(beta.responsibility, "Beta stores things.");
        assert!(beta.dependencies.is_empty()); // only external `serde`

        // Missing `description` falls back to the empty string.
        let gamma = &digest.crates[2];
        assert_eq!(gamma.responsibility, "");
        assert!(gamma.dependencies.is_empty());
    }

    #[test]
    fn build_digest_is_deterministic() {
        // Same inputs ⇒ byte-identical digest (stable ordering, stable hash).
        assert_eq!(
            build_digest(FIXTURE, MANIFESTS).unwrap(),
            build_digest(FIXTURE, MANIFESTS).unwrap()
        );
    }

    #[test]
    fn source_hash_tracks_manifest_bytes_only() {
        let base = build_digest(FIXTURE, MANIFESTS).unwrap();

        // Stable when the manifest bytes do not change.
        assert_eq!(base.source_hash, build_digest(FIXTURE, MANIFESTS).unwrap().source_hash);

        // Changes when the manifest bytes change (a crate added to the members).
        let changed =
            build_digest(FIXTURE, b"[workspace]\nmembers = [\"alpha\", \"beta\", \"gamma\", \"delta\"]\n")
                .unwrap();
        assert_ne!(base.source_hash, changed.source_hash);
    }

    #[test]
    fn is_stale_compares_content_not_time() {
        let digest = build_digest(FIXTURE, MANIFESTS).unwrap();

        // Identical manifest bytes ⇒ fresh, regardless of when it was built.
        assert!(!digest.is_stale(MANIFESTS));
        // Any drift in the manifest bytes ⇒ stale ⇒ regenerate.
        assert!(digest.is_stale(b"[workspace]\nmembers = [\"alpha\"]\n"));
    }

    #[test]
    fn round_trips_through_the_redb_store() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Nothing cached yet.
        assert!(ArchitectureDigest::load_or_none(root).unwrap().is_none());

        let built = build_digest(FIXTURE, MANIFESTS).unwrap();
        built.store(root).unwrap();

        // A fresh load (new process) sees the persisted digest verbatim.
        let loaded = ArchitectureDigest::load_or_none(root).unwrap().unwrap();
        assert_eq!(loaded, built);
        assert!(!loaded.is_stale(MANIFESTS));
    }

    #[test]
    fn coexists_with_conclusions_translation_memory_and_project_state() {
        use crate::{ConclusionLog, ProjectState, SegmentId, TranslationMemory, content_hash};
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("s.rs"), b"x").unwrap();

        // Write all four consumers into the same `.kopitiam` store.
        let mut state = ProjectState::load(root).unwrap();
        state.set_current_task("cache the architecture");
        state.save(root).unwrap();

        let mut log = ConclusionLog::load(root).unwrap();
        log.record("a fact", &["s.rs"]).unwrap();

        let seg = SegmentId::new(content_hash(b"hello"));
        {
            let mut tm = TranslationMemory::load(root).unwrap();
            tm.record(seg.clone(), "hello", "你好").unwrap();
        }

        build_digest(FIXTURE, MANIFESTS).unwrap().store(root).unwrap();

        // Every key round-trips independently; the digest disturbs none of the
        // pre-existing keys, and they disturb none of it.
        assert_eq!(
            ProjectState::load(root).unwrap().current_task.as_deref(),
            Some("cache the architecture")
        );
        assert_eq!(ConclusionLog::load(root).unwrap().conclusions().len(), 1);
        assert_eq!(
            TranslationMemory::load(root).unwrap().lookup(&seg).unwrap().translation,
            "你好"
        );
        let digest = ArchitectureDigest::load_or_none(root).unwrap().unwrap();
        assert_eq!(digest.crates.len(), 3);
    }

    /// Exercises the live `cargo metadata` path against this real workspace.
    /// `#[ignore]`d so the default `cargo test` stays hermetic (no process
    /// spawn); run explicitly with `cargo test -- --ignored`.
    #[test]
    #[ignore = "spawns a live `cargo metadata`; not hermetic"]
    fn run_cargo_metadata_builds_a_real_digest() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir.parent().and_then(Path::parent).expect("workspace root");

        let json = run_cargo_metadata(workspace_root).expect("cargo metadata should run");
        let manifests = std::fs::read(workspace_root.join("Cargo.toml")).unwrap();
        let digest = build_digest(&json, &manifests).unwrap();

        assert!(
            digest.crates.iter().any(|c| c.name == "kopitiam-workspace"),
            "expected kopitiam-workspace among the workspace crates"
        );
    }
}
