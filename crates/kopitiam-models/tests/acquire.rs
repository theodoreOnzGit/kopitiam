//! Offline tests for the acquire path. NO network here at all -- every test
//! uses a `tempdir` store, a hand-written `Fetcher` that writes known local
//! bytes, and a `ModelSpec` whose sha256 we compute from those exact bytes with
//! `sha2`. That way the verification gate is tested against real hashes, but
//! nothing ever leaves the machine.

use std::cell::Cell;
use std::path::Path;

use sha2::{Digest, Sha256};
use tempfile::tempdir;

use kopitiam_models::{
    ensure_available, Architecture, Artifact, Catalog, Error, Fetcher, ModelSpec,
    ModelStore,
};

/// The bytes our fake "model" is made of. Small, known, and hashable.
const FAKE_BYTES: &[u8] = b"kopitiam fake gguf bytes -- not a real model lah";

/// sha256 of `FAKE_BYTES` as lowercase hex, computed the same way the crate
/// does, so `verify` / `ensure_available` will actually pass.
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

/// A spec with one artifact, whose sha256 matches whatever `sha256` you pass.
/// The URL is a dummy -- the fake fetcher ignores it and just writes bytes.
fn one_artifact_spec(sha256: String) -> ModelSpec {
    ModelSpec {
        id: "test-model".to_string(),
        display_name: "Test Model".to_string(),
        architecture: Architecture::Other("test".to_string()),
        license: "Apache-2.0".to_string(),
        artifacts: vec![Artifact {
            filename: "model.gguf".to_string(),
            url: "https://example.invalid/model.gguf".to_string(),
            sha256,
            size_bytes: FAKE_BYTES.len() as u64,
        }],
    }
}

/// A fetcher that writes `FAKE_BYTES` to `dest` and counts how many times it was
/// called. Used to PROVE the BYO short-circuit skips the fetcher.
struct CountingFetcher {
    calls: Cell<u32>,
}

impl CountingFetcher {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
        }
    }
}

impl Fetcher for CountingFetcher {
    fn fetch(
        &self,
        _url: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), Error> {
        self.calls.set(self.calls.get() + 1);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, FAKE_BYTES)?;
        progress(FAKE_BYTES.len() as u64, Some(FAKE_BYTES.len() as u64));
        Ok(())
    }
}

/// A fetcher that PANICS if called. If the BYO short-circuit is correct, this
/// one is never touched.
struct PanicFetcher;

impl Fetcher for PanicFetcher {
    fn fetch(
        &self,
        _url: &str,
        _dest: &Path,
        _progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), Error> {
        panic!("fetcher must NOT be called when the artifact is already present and valid");
    }
}

/// A fetcher that writes the WRONG bytes -- to drive a post-fetch mismatch.
struct WrongBytesFetcher;

impl Fetcher for WrongBytesFetcher {
    fn fetch(
        &self,
        _url: &str,
        dest: &Path,
        _progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), Error> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, b"these are not the promised bytes")?;
        Ok(())
    }
}

#[test]
fn missing_file_gets_fetched_then_verified() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let spec = one_artifact_spec(fake_sha256());
    let fetcher = CountingFetcher::new();

    assert!(!store.is_present(&spec), "should start absent");

    let acquired = ensure_available(&store, &spec, &fetcher).expect("acquire ok");

    assert_eq!(fetcher.calls.get(), 1, "fetcher called exactly once");
    assert_eq!(acquired.artifact_paths.len(), 1);
    assert!(acquired.artifact_paths[0].is_file(), "file now on disk");
    assert!(store.is_present(&spec));
    store.verify(&spec).expect("verifies after fetch");
}

#[test]
fn byo_short_circuit_does_not_call_fetcher() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let spec = one_artifact_spec(fake_sha256());

    // Drop the correct file in by hand (the BYO case).
    let path = store.artifact_path(&spec, &spec.artifacts[0]);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, FAKE_BYTES).unwrap();
    assert!(store.is_present(&spec));

    // PanicFetcher proves it is never called.
    let acquired =
        ensure_available(&store, &spec, &PanicFetcher).expect("BYO acquire ok");
    assert_eq!(acquired.artifact_paths[0], path);
    store.verify(&spec).expect("BYO file verifies");
}

#[test]
fn present_but_wrong_file_is_checksum_mismatch_without_refetch() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let spec = one_artifact_spec(fake_sha256());

    // A present file with the WRONG bytes.
    let path = store.artifact_path(&spec, &spec.artifacts[0]);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"corrupt contents").unwrap();

    // Even with a working fetcher available, a present-but-wrong file must NOT
    // be silently re-downloaded -- it is a hard ChecksumMismatch.
    let counting = CountingFetcher::new();
    let err = ensure_available(&store, &spec, &counting).unwrap_err();
    assert_eq!(counting.calls.get(), 0, "present file: no fetch attempted");
    match err {
        Error::ChecksumMismatch { artifact, .. } => assert_eq!(artifact, "model.gguf"),
        other => panic!("expected ChecksumMismatch, got {other:?}"),
    }
}

#[test]
fn fetched_wrong_bytes_is_checksum_mismatch() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let spec = one_artifact_spec(fake_sha256());

    let err = ensure_available(&store, &spec, &WrongBytesFetcher).unwrap_err();
    match err {
        Error::ChecksumMismatch {
            artifact,
            expected,
            actual,
        } => {
            assert_eq!(artifact, "model.gguf");
            assert_eq!(expected, fake_sha256());
            assert_ne!(actual, expected, "actual hash differs from expected");
        }
        other => panic!("expected ChecksumMismatch, got {other:?}"),
    }
}

#[test]
fn store_verify_catches_bad_hash() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    // Deliberately wrong expected hash (all f's).
    let spec = one_artifact_spec("f".repeat(64));

    let path = store.artifact_path(&spec, &spec.artifacts[0]);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, FAKE_BYTES).unwrap();

    assert!(store.is_present(&spec), "file exists");
    match store.verify(&spec).unwrap_err() {
        Error::ChecksumMismatch { actual, .. } => {
            assert_eq!(actual, fake_sha256(), "reports the real hash");
        }
        other => panic!("expected ChecksumMismatch, got {other:?}"),
    }
}

#[test]
fn is_present_false_when_missing() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let spec = one_artifact_spec(fake_sha256());
    assert!(!store.is_present(&spec));
}

#[test]
fn artifact_path_is_namespaced_by_id() {
    let dir = tempdir().unwrap();
    let store = ModelStore::with_root(dir.path());
    let spec = one_artifact_spec(fake_sha256());
    let path = store.artifact_path(&spec, &spec.artifacts[0]);
    assert!(path.ends_with("test-model/model.gguf"));
    assert!(path.starts_with(store.root()));
}

#[test]
fn builtin_catalog_has_at_least_two_families() {
    let specs = Catalog::builtin();
    assert!(specs.len() >= 2, "need >= 2 entries");

    let mut saw_qwen = false;
    let mut saw_llama = false;
    for s in &specs {
        match s.architecture {
            Architecture::Qwen2 => saw_qwen = true,
            Architecture::Llama => saw_llama = true,
            _ => {}
        }
    }
    assert!(saw_qwen, "catalog must ship a Qwen2 entry");
    assert!(saw_llama, "catalog must ship a Llama entry");
}

#[test]
fn builtin_checksums_are_placeholders() {
    // Documents the intentional placeholder state: every shipped checksum is
    // 64 zeros until a real pull records the true value.
    for s in Catalog::builtin() {
        for a in &s.artifacts {
            assert_eq!(a.sha256, "0".repeat(64), "{} still placeholder", a.filename);
        }
    }
}

#[test]
fn find_returns_known_and_none_for_unknown() {
    assert!(Catalog::find("qwen2.5-0.5b-instruct-q4_0").is_some());
    assert!(Catalog::find("no-such-model").is_none());
}
