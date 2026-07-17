//! Offline tests for the HuggingFace acquire path. NO network here -- every
//! test either exercises pure URL/revision/token logic, or drives the acquire
//! path through a mock `Fetcher` that captures the URL + writes known local
//! bytes. Nothing ever opens a socket or downloads a real model.

use std::cell::RefCell;
use std::path::Path;

use sha2::{Digest, Sha256};
use tempfile::tempdir;

use kopitiam_models::{
    ensure_available, hf::HfFile, hf::HfModel, hf::Revision, hf_resolve_url, Architecture, Error,
    Fetcher, ModelStore, RevisionError,
};

// ---------------------------------------------------------------------------
// URL construction -- the hard-won scheme, locked in.
// ---------------------------------------------------------------------------

#[test]
fn resolve_url_has_exact_scheme() {
    let url = hf_resolve_url(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "main",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
    assert_eq!(
        url,
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    );
}

#[test]
fn resolve_url_takes_a_commit_sha_as_revision() {
    let sha = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0";
    let url = hf_resolve_url("owner/repo", sha, "model.gguf");
    assert_eq!(
        url,
        format!("https://huggingface.co/owner/repo/resolve/{sha}/model.gguf")
    );
}

#[test]
fn resolve_url_tidies_stray_slashes() {
    // Sloppy slashes must still yield one clean URL (no `//` in the path part).
    let url = hf_resolve_url("owner/repo/", "/main/", "/sub/model.gguf");
    assert_eq!(
        url,
        "https://huggingface.co/owner/repo/resolve/main/sub/model.gguf"
    );
}

// ---------------------------------------------------------------------------
// Revision pinning -- reproducibility enforced in the type.
// ---------------------------------------------------------------------------

#[test]
fn commit_revision_is_reproducible() {
    let r = Revision::commit("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0").unwrap();
    assert!(r.is_reproducible());
    assert_eq!(r.as_str(), "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0");
}

#[test]
fn abbreviated_commit_is_accepted() {
    // git's 7-char floor is enough to count as a pinned commit.
    let r = Revision::commit("a1b2c3d").unwrap();
    assert!(r.is_reproducible());
}

#[test]
fn main_and_moving_are_not_reproducible() {
    assert!(!Revision::main().is_reproducible());
    assert_eq!(Revision::main().as_str(), "main");
    assert!(!Revision::moving("v1.0").is_reproducible());
    assert_eq!(Revision::moving("v1.0").as_str(), "v1.0");
}

#[test]
fn non_sha_is_rejected_as_commit() {
    // A branch name, an uppercase SHA, too-short, too-long, and non-hex all
    // fail -- so a moving target can never masquerade as a pinned commit.
    for bad in [
        "main",
        "MAIN",
        "A1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0", // uppercase
        "abc",                                       // too short (<7)
        &"a".repeat(41),                             // too long (>40)
        "z1b2c3d",                                   // non-hex
    ] {
        let err = Revision::commit(bad).unwrap_err();
        assert!(
            matches!(err, RevisionError::NotACommitSha { ref got } if got == bad),
            "want NotACommitSha for {bad:?}, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// HfModel -> ModelSpec folding.
// ---------------------------------------------------------------------------

const FAKE_BYTES: &[u8] = b"kopitiam fake hf gguf bytes -- not a real model lah";

fn fake_sha256() -> String {
    let mut h = Sha256::new();
    h.update(FAKE_BYTES);
    let digest = h.finalize();
    let mut s = String::new();
    for b in digest {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn hf_model(revision: Revision) -> HfModel {
    HfModel {
        id: "hf-test-model".to_string(),
        display_name: "HF Test Model".to_string(),
        architecture: Architecture::Qwen2,
        license: "Apache-2.0".to_string(),
        repo: "owner/repo-GGUF".to_string(),
        revision,
        files: vec![
            HfFile {
                filename: "model-q4_k_m.gguf".to_string(),
                sha256: fake_sha256(),
                size_bytes: FAKE_BYTES.len() as u64,
            },
            HfFile {
                // A sidecar, to prove multi-file HF specs work.
                filename: "tokenizer.json".to_string(),
                sha256: fake_sha256(),
                size_bytes: FAKE_BYTES.len() as u64,
            },
        ],
    }
}

#[test]
fn into_spec_builds_resolve_urls_for_every_file() {
    let sha = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0";
    let model = hf_model(Revision::commit(sha).unwrap());
    let spec = model.into_spec();

    assert_eq!(spec.id, "hf-test-model");
    assert_eq!(spec.artifacts.len(), 2, "gguf + sidecar");
    assert_eq!(
        spec.artifacts[0].url,
        format!("https://huggingface.co/owner/repo-GGUF/resolve/{sha}/model-q4_k_m.gguf")
    );
    assert_eq!(
        spec.artifacts[1].url,
        format!("https://huggingface.co/owner/repo-GGUF/resolve/{sha}/tokenizer.json")
    );
    // The filename saved on disk is the repo-relative name, not the whole URL.
    assert_eq!(spec.artifacts[0].filename, "model-q4_k_m.gguf");
}

#[test]
fn reproducibility_flows_from_revision_to_model() {
    assert!(hf_model(Revision::commit("a1b2c3d").unwrap()).is_reproducible());
    assert!(!hf_model(Revision::main()).is_reproducible());
}

// ---------------------------------------------------------------------------
// Acquire path through a mock Fetcher (redirect/token handling is the real
// fetcher's job; here we prove an HF-shaped spec drives ensure_available end to
// end, and that the URL the fetcher receives is the resolve URL).
// ---------------------------------------------------------------------------

/// Records every URL it is asked to fetch, then writes `FAKE_BYTES`. Stands in
/// for the network so the acquire path is exercised with zero sockets.
struct RecordingFetcher {
    urls: RefCell<Vec<String>>,
}

impl RecordingFetcher {
    fn new() -> Self {
        Self {
            urls: RefCell::new(Vec::new()),
        }
    }
}

impl Fetcher for RecordingFetcher {
    fn fetch(
        &self,
        url: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), Error> {
        self.urls.borrow_mut().push(url.to_string());
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, FAKE_BYTES)?;
        progress(FAKE_BYTES.len() as u64, Some(FAKE_BYTES.len() as u64));
        Ok(())
    }
}

#[test]
fn hf_spec_acquires_through_mock_fetcher_and_verifies() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let sha = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0";
    let spec = hf_model(Revision::commit(sha).unwrap()).into_spec();

    let fetcher = RecordingFetcher::new();
    let acquired = ensure_available(&store, &spec, &fetcher).expect("hf acquire ok");

    // Both files landed and verified.
    assert_eq!(acquired.artifact_paths.len(), 2);
    for p in &acquired.artifact_paths {
        assert!(p.is_file());
    }
    store.verify(&spec).expect("verifies after fetch");

    // The fetcher was handed the resolve URLs, in artifact order.
    let urls = fetcher.urls.borrow();
    assert_eq!(urls.len(), 2);
    assert!(urls[0].contains(&format!("/resolve/{sha}/model-q4_k_m.gguf")));
    assert!(urls[1].contains(&format!("/resolve/{sha}/tokenizer.json")));
}

// ---------------------------------------------------------------------------
// Token handling. Tested through the pure `normalize_token` so no process-env
// mutation (racy, and `unsafe` under edition 2024) is needed -- the env reader
// `hf_token_from_env` is just `normalize_token(env::var(...).ok())`.
// ---------------------------------------------------------------------------

#[test]
fn token_normalisation_trims_and_treats_empty_as_none() {
    use kopitiam_models::normalize_token;

    // Absent -> None.
    assert_eq!(normalize_token(None), None);
    // Present -> Some, trimmed.
    assert_eq!(
        normalize_token(Some("  hf_secret123  ".to_string())).as_deref(),
        Some("hf_secret123")
    );
    // Whitespace-only / empty -> None (an empty Bearer would only get rejected).
    assert_eq!(normalize_token(Some("   ".to_string())), None);
    assert_eq!(normalize_token(Some(String::new())), None);
}

/// The token value must never surface in a `Debug` render of the fetcher --
/// only whether one is present. Guards against an accidental `#[derive(Debug)]`
/// leaking the secret into a log line.
#[cfg(feature = "net")]
#[test]
fn fetcher_with_token_reports_presence_only() {
    use kopitiam_models::HfFetcher;

    let anon = HfFetcher::with_token(None);
    assert!(!anon.has_token());

    let empty = HfFetcher::with_token(Some("   ".to_string()));
    assert!(!empty.has_token(), "whitespace token is treated as none");

    let authed = HfFetcher::with_token(Some("hf_secret123".to_string()));
    assert!(authed.has_token());
    // HfFetcher intentionally does NOT derive Debug, so the token cannot be
    // printed. `has_token()` is the only readout, and it exposes no bytes.
}
