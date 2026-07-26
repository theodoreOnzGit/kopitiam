//! The `digest` subcommand: a cached architecture digest — token-max Task II-3.
//!
//! `scan` already runs `cargo metadata` and learns the whole crate graph, but
//! throws it away (`kopitiam_token_max.md` §275), so every session re-derives
//! "which crate does what and depends on what" with a full exploration pass.
//! This command serialises that answer once into the project's `.kopitiam` redb
//! store and hands it back for free on subsequent sessions.
//!
//! The real work is `kopitiam_workspace`'s digest engine:
//! [`run_cargo_metadata`] → [`build_digest`] (a pure JSON→digest transform),
//! persisted via [`ArchitectureDigest::store`] and validated against a
//! manifest content hash via [`ArchitectureDigest::is_stale`]. This file only
//! decides *when* to rebuild and formats the result.
//!
//! # Invalidation is content, never time (§0.4)
//!
//! The digest pins the FNV-1a hash of the concatenated workspace `Cargo.toml`
//! manifests. A run re-reads those manifests (cheap) and rebuilds only when the
//! hash drifted or `--refresh` is given; otherwise the cached digest is trusted
//! and `cargo metadata` is never spawned. Reading manifests is orders of
//! magnitude cheaper than a `cargo metadata` run, which is the whole point.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use kopitiam_workspace::{ArchitectureDigest, build_digest, run_cargo_metadata};

/// Options for `kopitiam digest`.
#[derive(Args, Debug)]
pub struct DigestArgs {
    /// The workspace root (holding the top `Cargo.toml` and `.kopitiam`).
    /// Defaults to the current directory.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Force a rebuild from `cargo metadata` even if the cached digest is still
    /// fresh for the current manifests.
    #[arg(long)]
    pub refresh: bool,

    /// Print the digest as JSON (crate → responsibility → deps + the source
    /// hash) instead of the human-readable listing (§0.2). Notices go to stderr.
    #[arg(long)]
    pub json: bool,
}

/// Runs `kopitiam digest`: load-or-rebuild, persist, print.
pub fn run(args: DigestArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)
        .with_context(|| format!("resolving --root {}", args.root.display()))?;
    let manifest_bytes = read_manifest_bytes(&root)?;

    let cached = ArchitectureDigest::load_or_none(&root)?;
    let (digest, regenerated) = match cached {
        Some(digest) if !args.refresh && !digest.is_stale(&manifest_bytes) => (digest, false),
        _ => {
            let metadata = run_cargo_metadata(&root)
                .context("running `cargo metadata` to (re)build the architecture digest")?;
            let digest = build_digest(&metadata, &manifest_bytes)?;
            digest.store(&root)?;
            (digest, true)
        }
    };

    if args.json {
        eprintln!("{}", source_notice(regenerated));
        println!("{}", serde_json::to_string_pretty(&digest)?);
    } else {
        print!("{}", render_human(&digest, regenerated));
    }
    Ok(())
}

/// A short note on whether the printed digest was rebuilt or served from cache.
fn source_notice(regenerated: bool) -> &'static str {
    if regenerated {
        "digest: rebuilt from `cargo metadata`"
    } else {
        "digest: served from cache (manifests unchanged)"
    }
}

/// The invalidation key material: the workspace's `Cargo.toml` manifests
/// concatenated in a stable order (root first, then every `crates/*` and
/// `apps/*` member manifest sorted by path). Reading these is cheap; the same
/// helper is used to build the digest's `source_hash` and to check staleness,
/// so the two never disagree.
fn read_manifest_bytes(root: &Path) -> Result<Vec<u8>> {
    let mut paths: Vec<PathBuf> = vec![root.join("Cargo.toml")];
    for group in ["crates", "apps"] {
        let base = root.join(group);
        if !base.is_dir() {
            continue;
        }
        let mut members: Vec<PathBuf> = std::fs::read_dir(&base)
            .with_context(|| format!("reading {}", base.display()))?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .map(|path| path.join("Cargo.toml"))
            .filter(|path| path.is_file())
            .collect();
        members.sort();
        paths.extend(members);
    }

    let mut bytes = Vec::new();
    for path in paths {
        if let Ok(content) = std::fs::read(&path) {
            bytes.extend_from_slice(&content);
        }
    }
    Ok(bytes)
}

/// Renders the human-readable digest: a header, then one stanza per crate —
/// name, responsibility, and workspace-internal dependency edges.
fn render_human(digest: &ArchitectureDigest, regenerated: bool) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Architecture digest — {} crates ({})",
        digest.crates.len(),
        if regenerated { "rebuilt" } else { "from cache" },
    );
    let _ = writeln!(out, "source_hash: {}", digest.source_hash);
    let _ = writeln!(out);
    for krate in &digest.crates {
        let _ = writeln!(out, "{}", krate.name);
        let responsibility = if krate.responsibility.is_empty() {
            "(no description)"
        } else {
            &krate.responsibility
        };
        let _ = writeln!(out, "  {responsibility}");
        if !krate.dependencies.is_empty() {
            let _ = writeln!(out, "  deps: {}", krate.dependencies.join(", "));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: DigestArgs,
    }

    /// A tiny `cargo metadata` fixture (two workspace members, one edge) so the
    /// human renderer is tested through the real `build_digest` with no live
    /// `cargo` run.
    const FIXTURE: &str = r#"{
        "workspace_members": [
            "path+file:///w/alpha#0.1.0",
            "path+file:///w/beta#0.1.0"
        ],
        "packages": [
            {
                "name": "alpha",
                "id": "path+file:///w/alpha#0.1.0",
                "description": "Alpha drives the front end.",
                "manifest_path": "/w/alpha/Cargo.toml",
                "dependencies": [ { "name": "beta", "kind": null } ]
            },
            {
                "name": "beta",
                "id": "path+file:///w/beta#0.1.0",
                "description": null,
                "manifest_path": "/w/beta/Cargo.toml",
                "dependencies": []
            }
        ]
    }"#;

    #[test]
    fn parses_flags() {
        let cli = TestCli::try_parse_from(["d", "--root", "/w", "--refresh", "--json"]).unwrap();
        assert_eq!(cli.args.root, PathBuf::from("/w"));
        assert!(cli.args.refresh);
        assert!(cli.args.json);
    }

    #[test]
    fn render_human_lists_crates_responsibilities_and_deps() {
        let digest = build_digest(FIXTURE, b"[workspace]\n").unwrap();
        let text = render_human(&digest, true);
        assert!(text.contains("Architecture digest — 2 crates (rebuilt)"));
        assert!(text.contains("alpha"));
        assert!(text.contains("Alpha drives the front end."));
        assert!(text.contains("deps: beta"));
        // A crate with no description falls back to the placeholder.
        assert!(text.contains("(no description)"));
    }

    #[test]
    fn manifest_bytes_are_deterministic_and_track_content() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("Cargo.toml"), b"[workspace]\n").unwrap();
        std::fs::create_dir_all(root.join("crates/a")).unwrap();
        std::fs::create_dir_all(root.join("crates/b")).unwrap();
        std::fs::write(root.join("crates/a/Cargo.toml"), b"name=a\n").unwrap();
        std::fs::write(root.join("crates/b/Cargo.toml"), b"name=b\n").unwrap();

        let first = read_manifest_bytes(root).unwrap();
        let second = read_manifest_bytes(root).unwrap();
        assert_eq!(first, second, "same tree must hash to the same bytes");

        // Changing a member manifest changes the bytes (so is_stale fires).
        std::fs::write(root.join("crates/b/Cargo.toml"), b"name=b2\n").unwrap();
        assert_ne!(first, read_manifest_bytes(root).unwrap());
    }

    /// Exercises the live `run_cargo_metadata` + persistence path against this
    /// real workspace. `#[ignore]`d so the default `cargo test` stays hermetic.
    #[test]
    #[ignore = "spawns a live `cargo metadata`; not hermetic"]
    fn live_digest_builds_and_caches() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = crate_dir.parent().and_then(Path::parent).expect("workspace root");
        let manifest_bytes = read_manifest_bytes(root).unwrap();
        let metadata = run_cargo_metadata(root).unwrap();
        let digest = build_digest(&metadata, &manifest_bytes).unwrap();
        assert!(digest.crates.iter().any(|c| c.name == "kopitiam"));
        assert!(!digest.is_stale(&manifest_bytes));
    }
}
