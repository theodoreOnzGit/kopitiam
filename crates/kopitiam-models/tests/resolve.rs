//! Tests for HF-hub sha256 auto-resolution.
//!
//! Everything here except the one `#[ignore]`d live test is **fully offline**:
//! the JSON/pointer parsing and the pinned-vs-resolved cross-check are pure, so
//! they run against a captured fixture and hand-written strings with zero
//! sockets. The live test is opt-in and documents its own run command.

use kopitiam_models::{check_pinned, parse_hf_tree, parse_lfs_pointer, Error, HfTreeLookup};

/// A real, captured `GET /api/models/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/tree/main`
/// body. Contains a git-LFS GGUF (the model) plus two small non-LFS blobs
/// (`.gitattributes`, `README.md`), so both branches of `parse_hf_tree` are
/// exercised against genuine hub JSON.
const SMOLLM2_360M_TREE: &str = include_str!("fixtures/smollm2_360m_tree.json");

/// The pinned sha256 the catalog carries for SmolLM2-360M-Instruct (Q8_0). The
/// whole point of resolution is that this same value comes back from the hub.
const SMOLLM2_360M_SHA256: &str =
    "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201";
const SMOLLM2_360M_SIZE: u64 = 386_404_992;

// ---------------------------------------------------------------------------
// tree-API JSON parsing (deterministic, no network).
// ---------------------------------------------------------------------------

#[test]
fn parse_tree_extracts_lfs_oid_and_size() {
    let got = parse_hf_tree(SMOLLM2_360M_TREE, "smollm2-360m-instruct-q8_0.gguf")
        .expect("well-formed tree JSON");
    assert_eq!(
        got,
        HfTreeLookup::Lfs {
            sha256: SMOLLM2_360M_SHA256.to_string(),
            size_bytes: SMOLLM2_360M_SIZE,
        },
        "the LFS oid IS the sha256, and lfs.size is the byte count"
    );
}

#[test]
fn parse_tree_reports_non_lfs_file() {
    // README.md is a small plain blob (no `lfs` object) -- its git oid is a
    // SHA-1, not the sha256 we want, so resolution must fall back to hashing.
    let got = parse_hf_tree(SMOLLM2_360M_TREE, "README.md").expect("well-formed");
    assert_eq!(got, HfTreeLookup::NotLfs { size_bytes: 1847 });
}

#[test]
fn parse_tree_reports_not_found_for_absent_file() {
    let got = parse_hf_tree(SMOLLM2_360M_TREE, "does-not-exist.gguf").expect("well-formed");
    assert_eq!(got, HfTreeLookup::NotFound);
}

#[test]
fn parse_tree_rejects_non_array_body() {
    // A hub error is a JSON object, not the documented array -- must be an Err,
    // never silently "not found".
    let err = parse_hf_tree(r#"{"error":"Repository not found"}"#, "x.gguf").unwrap_err();
    assert!(matches!(err, Error::Http(_)), "got {err:?}");
}

// ---------------------------------------------------------------------------
// LFS pointer parsing (the /raw/ fallback), deterministic.
// ---------------------------------------------------------------------------

#[test]
fn parse_lfs_pointer_extracts_sha_and_size() {
    // Exactly the text `.../raw/main/<gguf>` returns for an LFS file.
    let pointer = "version https://git-lfs.github.com/spec/v1\n\
                   oid sha256:48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201\n\
                   size 386404992\n";
    let (sha, size) = parse_lfs_pointer(pointer).expect("valid pointer");
    assert_eq!(sha, SMOLLM2_360M_SHA256);
    assert_eq!(size, SMOLLM2_360M_SIZE);
}

#[test]
fn parse_lfs_pointer_rejects_non_pointer_text() {
    // Actual (non-LFS) file bytes must NOT be mistaken for a pointer.
    assert!(parse_lfs_pointer("{\n  \"hidden_size\": 960\n}\n").is_none());
    // Missing the size line -> not a complete pointer.
    assert!(parse_lfs_pointer("oid sha256:deadbeef").is_none());
}

// ---------------------------------------------------------------------------
// Pinned-vs-resolved cross-check (defense in depth), deterministic.
// ---------------------------------------------------------------------------

#[test]
fn pinned_sha_mismatch_is_a_hard_error() {
    let wrong_pin = "0".repeat(64);
    let err = check_pinned(
        "smollm2-360m-instruct-q8_0.gguf",
        SMOLLM2_360M_SHA256,
        Some(&wrong_pin),
    )
    .unwrap_err();
    match err {
        Error::ChecksumMismatch {
            artifact,
            expected,
            actual,
        } => {
            assert_eq!(artifact, "smollm2-360m-instruct-q8_0.gguf");
            assert_eq!(expected, wrong_pin, "the pin is reported as `expected`");
            assert_eq!(actual, SMOLLM2_360M_SHA256, "the hub value as `actual`");
        }
        other => panic!("expected ChecksumMismatch, got {other:?}"),
    }
}

#[test]
fn pinned_sha_match_and_absent_pin_are_ok() {
    // A matching pin passes.
    check_pinned("f.gguf", SMOLLM2_360M_SHA256, Some(SMOLLM2_360M_SHA256))
        .expect("matching pin is fine");
    // No pin (None) or an empty pin means "trust the resolved value".
    check_pinned("f.gguf", SMOLLM2_360M_SHA256, None).expect("absent pin is fine");
    check_pinned("f.gguf", SMOLLM2_360M_SHA256, Some("")).expect("empty pin is fine");
}

// ---------------------------------------------------------------------------
// The catalog wiring: the SmolLM2 entries carry an HF source now, so their
// checksum CAN be auto-resolved. Verified here without network by checking the
// declaration is well-formed (the live resolve is the ignored test below).
// ---------------------------------------------------------------------------

#[test]
fn smollm2_entries_declare_an_hf_source() {
    use kopitiam_models::Catalog;

    for id in ["smollm2-360m-instruct-q8_0", "smollm2-1.7b-instruct-q4_k_m"] {
        let spec = Catalog::find(id).unwrap_or_else(|| panic!("{id} in catalog"));
        let a = spec.artifacts.first().expect("one artifact");
        let src = a
            .hf
            .as_ref()
            .unwrap_or_else(|| panic!("{id} must carry an HfSource for auto-resolution"));
        assert!(src.repo.starts_with("HuggingFaceTB/"), "{id}: HF repo");
        assert_eq!(src.revision, "main");
        assert_eq!(src.file, a.filename, "{id}: source file == artifact filename");
        // The pinned sha is still present (defense-in-depth), not blanked.
        assert_eq!(a.sha256.len(), 64, "{id}: pin retained");
    }
}

// ---------------------------------------------------------------------------
// LIVE test -- hits huggingface.co. Ignored by default. Run it with:
//
//   cargo test -p kopitiam-models --features net -- --ignored resolve_real_smollm2_360m_sha
//
// It resolves the real SmolLM2-360M-Instruct (Q8_0) sha256 from the hub and
// asserts it matches the pinned catalog value -- proving the resolver against
// live infrastructure, and catching the day upstream re-quantises.
// ---------------------------------------------------------------------------

#[cfg(feature = "net")]
#[test]
#[ignore = "hits the network; run with --ignored"]
fn resolve_real_smollm2_360m_sha() {
    use kopitiam_models::resolve_hf_sha256;

    let resolved = resolve_hf_sha256(
        "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        "smollm2-360m-instruct-q8_0.gguf",
    )
    .expect("live resolve should succeed");

    assert_eq!(
        resolved.sha256, SMOLLM2_360M_SHA256,
        "resolved sha must match the pinned catalog value"
    );
    assert_eq!(resolved.size_bytes, SMOLLM2_360M_SIZE);
    assert!(resolved.url.ends_with("smollm2-360m-instruct-q8_0.gguf"));
}
