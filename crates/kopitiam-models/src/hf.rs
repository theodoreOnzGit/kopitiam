//! HuggingFace as a first-class acquire path.
//!
//! This module is the one place in the crate that knows the shape of
//! <https://huggingface.co>. It gives two things:
//!
//! * A **declaration type** ([`HfModel`] + [`HfFile`] + [`Revision`]) for
//!   spelling out an HF-hosted model as `repo` / `filename` / `revision` /
//!   `sha256`, which folds down into the plain [`crate::ModelSpec`] the rest of
//!   the crate already knows how to acquire and verify. No new acquire engine --
//!   it reuses [`crate::ensure_available`] and the sha256 gate wholesale.
//! * A **fetcher** ([`HfFetcher`], behind the `net` feature) that adds the one
//!   thing plain HTTP is missing for HF: an optional `Authorization: Bearer`
//!   header pulled from `HF_TOKEN`, for gated / private repos.
//!
//! ## The HF direct-download URL scheme (hard-won -- do NOT paraphrase away)
//!
//! HuggingFace serves a raw file from a repo at:
//!
//! ```text
//! https://huggingface.co/{repo}/resolve/{revision}/{filename}
//! ```
//!
//! Worked example, so nobody has to re-derive it:
//!
//! * `repo`     = `Qwen/Qwen2.5-0.5B-Instruct-GGUF`  (the `owner/name` slug)
//! * `revision` = `main`  (a branch/tag) **or** a 40-char commit SHA
//! * `filename` = `qwen2.5-0.5b-instruct-q4_k_m.gguf`
//!
//! giving
//! `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf`.
//!
//! **The `resolve` URL is a redirect, not the bytes.** HuggingFace answers it
//! with a `302` to a CDN (LFS/`cas`/CloudFront) presigned URL, and the actual
//! GGUF bytes come from *there*. So any fetcher used against these URLs **must
//! follow redirects** -- [`HfFetcher`] does, and so does the plain
//! [`crate::HttpFetcher`] (ureq follows by default). If one day a fetcher is
//! wired that does NOT follow redirects, HF downloads will come back as a tiny
//! 302 body, not a model -- and the sha256 gate will (correctly) reject it.
//!
//! ## Reproducibility: pin a commit, or accept that `main` moves
//!
//! `revision = "main"` **MOVES**. Upstream can push a new quant, re-convert the
//! GGUF, or retag, and the bytes behind the exact same URL change under you --
//! which then trips the sha256 gate with a "mismatch" that is really "upstream
//! changed". For a fetch that is reproducible and verifiable for keeps, pin a
//! **commit SHA** as the revision AND record the expected **sha256**. That pair
//! -- pinned revision + pinned hash -- is what makes an [`HfModel`] safe to bake
//! into the catalog.
//!
//! [`Revision`] carries this distinction in the *type*: [`Revision::commit`]
//! only accepts a hex commit SHA and reports [`Revision::is_reproducible`] as
//! `true`; [`Revision::moving`] (and the `main` default) report `false`. A
//! moving revision is allowed -- it is the convenient default for a first pull
//! -- but [`Revision::is_reproducible`] lets a catalog author (or a lint) refuse
//! to *ship* one.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::catalog::{Architecture, Artifact, HfSource, ModelSpec};
use crate::error::Error;

/// The environment variable [`HfFetcher`] reads an optional bearer token from.
///
/// Same name the official `huggingface_hub` client uses, so a machine already
/// logged in for Python picks up here for free.
pub const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// A pinned-or-moving pointer at a revision inside an HF repo.
///
/// This is the reproducibility guardrail, done in the type instead of in a
/// comment nobody reads. Two shapes:
///
/// * [`Revision::Commit`] -- a git commit SHA. The bytes at a commit never move,
///   so this is the **reproducible** one. Build it through [`Revision::commit`],
///   which validates the SHA shape.
/// * [`Revision::Moving`] -- a branch or tag (`main`, `v1.0`, ...). Convenient,
///   but the bytes behind it can change any time upstream pushes. **Not
///   reproducible.**
///
/// See the module docs for why this matters against the sha256 gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Revision {
    /// A pinned commit SHA (lowercase hex, 7..=40 chars). Reproducible.
    Commit(String),
    /// A moving ref -- a branch or tag name like `main`. NOT reproducible.
    Moving(String),
}

impl Revision {
    /// Pin to a commit SHA. This is the reproducible path.
    ///
    /// Accepts 7..=40 lowercase-hex chars -- a full 40-char git SHA, or a
    /// git-style abbreviation (7+ chars is git's own "unambiguous enough"
    /// floor). Anything else is rejected, because a "commit" that is really a
    /// branch name would silently give you a moving target dressed up as a
    /// pinned one -- exactly the confusion this type exists to stop.
    ///
    /// # Errors
    ///
    /// [`RevisionError::NotACommitSha`] if `sha` is not 7..=40 lowercase-hex
    /// chars. Uppercase is rejected on purpose: git prints lowercase, and the
    /// rest of KOPITIAM compares lowercase hex, so accepting uppercase here
    /// would only invite a mismatch later.
    pub fn commit(sha: impl Into<String>) -> Result<Self, RevisionError> {
        let sha = sha.into();
        if is_commit_sha(&sha) {
            Ok(Revision::Commit(sha))
        } else {
            Err(RevisionError::NotACommitSha { got: sha })
        }
    }

    /// Point at a moving ref (branch or tag). NOT reproducible -- the bytes can
    /// change under you. Use [`Revision::commit`] for anything you intend to
    /// pin and ship.
    pub fn moving(name: impl Into<String>) -> Self {
        Revision::Moving(name.into())
    }

    /// The `main` branch -- the usual HF default, and a *moving* target. Handy
    /// for a first exploratory pull; do NOT bake it into a shipped catalog entry
    /// (see [`Revision::is_reproducible`]).
    pub fn main() -> Self {
        Revision::Moving("main".to_string())
    }

    /// The revision string exactly as it goes into the `{revision}` slot of the
    /// resolve URL -- the SHA for a commit, the ref name for a moving one.
    pub fn as_str(&self) -> &str {
        match self {
            Revision::Commit(s) | Revision::Moving(s) => s,
        }
    }

    /// `true` only for a pinned commit. A `false` here means the bytes behind
    /// this revision can change upstream without warning -- a catalog author who
    /// cares about reproducibility should refuse to ship such an entry.
    pub fn is_reproducible(&self) -> bool {
        matches!(self, Revision::Commit(_))
    }
}

/// One file to pull out of an HF repo -- the GGUF itself, or a sidecar
/// (`config.json`, `tokenizer.json`, ...). All files of one [`HfModel`] share
/// the same `repo` and `revision`; only the per-file bits live here.
///
/// The [`HfFile::sha256`] is the same verification gate the whole crate runs on
/// -- lowercase-hex sha256 of the expected bytes, checked after download. Pin it
/// together with a commit [`Revision`] for a reproducible fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfFile {
    /// The path of the file *inside the repo*, which is also what it gets saved
    /// as on disk (e.g. `qwen2.5-0.5b-instruct-q4_k_m.gguf`, or a subfolder path
    /// like `tokenizer/tokenizer.json`).
    pub filename: String,
    /// Lowercase-hex sha256 of the expected bytes. The verification gate.
    pub sha256: String,
    /// Expected size in bytes -- for progress + a cheap sanity check. The sha256
    /// is the real guarantee.
    pub size_bytes: u64,
}

/// An HF-hosted model, spelled out the HuggingFace way (`repo` + `revision` +
/// files), ready to fold down into a plain [`ModelSpec`] with
/// [`HfModel::into_spec`].
///
/// This is the "clean way to declare an HF model" the catalog will use once the
/// maintainer names which models to ship. It deliberately does NOT invent a
/// second acquire path: [`HfModel::into_spec`] builds ordinary [`Artifact`]s
/// whose `url` is the resolve URL, and from there [`crate::ensure_available`]
/// does all the fetching + verifying exactly as before.
///
/// ```
/// use kopitiam_models::hf::{HfFile, HfModel, Revision};
/// use kopitiam_models::Architecture;
///
/// // Reproducible declaration: a pinned commit + a pinned sha256.
/// let model = HfModel {
///     id: "example-q4".to_string(),
///     display_name: "Example (Q4)".to_string(),
///     architecture: Architecture::Qwen2,
///     license: "Apache-2.0".to_string(),
///     repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string(),
///     revision: Revision::commit("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0").unwrap(),
///     files: vec![HfFile {
///         filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
///         // Real hash goes here once recorded from a first pull.
///         sha256: "0".repeat(64),
///         size_bytes: 400_000_000,
///     }],
/// };
/// assert!(model.is_reproducible());
/// let spec = model.into_spec();
/// assert!(spec.artifacts[0].url.contains("/resolve/"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    /// Stable catalog key, e.g. `"qwen2.5-0.5b-instruct-q4_k_m"`.
    pub id: String,
    /// Human-friendly name for showing the user.
    pub display_name: String,
    /// Which family this is -- drives the forward pass downstream.
    pub architecture: Architecture,
    /// SPDX-ish licence id of the *model weights*, separate from KOPITIAM's own
    /// AGPL-3.0-only code licence.
    pub license: String,
    /// The `owner/name` repo slug, e.g. `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.
    pub repo: String,
    /// Which revision to pull from. Pin a [`Revision::commit`] for a
    /// reproducible entry; [`Revision::main`] is allowed but moves.
    pub revision: Revision,
    /// Every file that must land + verify: the GGUF, plus any sidecars.
    pub files: Vec<HfFile>,
}

impl HfModel {
    /// `true` only when this declaration is fully reproducible: the revision is
    /// a pinned commit. A catalog lint can gate `into_spec` on this before
    /// baking an entry in.
    ///
    /// Note this speaks to the *revision*, not the hashes -- a placeholder
    /// sha256 is still caught later, at the download-time gate.
    pub fn is_reproducible(&self) -> bool {
        self.revision.is_reproducible()
    }

    /// Fold this HF declaration down into the plain [`ModelSpec`] the acquire
    /// path already understands. Each [`HfFile`] becomes an [`Artifact`] whose
    /// `url` is the resolve URL built from `repo` + `revision` + `filename`.
    ///
    /// From here nothing is HF-specific anymore: [`crate::ensure_available`]
    /// with [`HfFetcher`] (or any [`crate::Fetcher`]) fetches and verifies it
    /// like any other spec.
    pub fn into_spec(self) -> ModelSpec {
        let revision = self.revision.as_str();
        let artifacts = self
            .files
            .into_iter()
            .map(|f| Artifact {
                url: hf_resolve_url(&self.repo, revision, &f.filename),
                // Carry the HF origin onto the artifact, so the acquire path can
                // re-resolve this file's sha256 from the hub and cross-check the
                // pinned one (defense in depth). A file declared with an empty
                // `sha256` becomes a fully auto-resolved artifact.
                hf: Some(HfSource {
                    repo: self.repo.clone(),
                    revision: revision.to_string(),
                    file: f.filename.clone(),
                }),
                filename: f.filename,
                sha256: f.sha256,
                size_bytes: f.size_bytes,
            })
            .collect();
        ModelSpec {
            id: self.id,
            display_name: self.display_name,
            architecture: self.architecture,
            license: self.license,
            artifacts,
        }
    }
}

/// Build the HF direct-download (`resolve`) URL for one file.
///
/// `https://huggingface.co/{repo}/resolve/{revision}/{filename}` -- see the
/// module docs for the full story on this scheme, the 302-to-CDN redirect, and
/// why `revision` should be a pinned commit for reproducibility.
///
/// This does a light tidy so callers can be a bit sloppy with slashes: a leading
/// `/` on `filename` and stray slashes around `repo`/`revision` are trimmed, so
/// `repo = "a/b/"` + `filename = "/x.gguf"` still gives one clean URL. It does
/// NOT percent-encode -- HF repo slugs, refs and GGUF filenames are already
/// URL-safe in practice; if a truly exotic filename ever needs encoding, that is
/// a documented follow-up, not a silent surprise here.
pub fn hf_resolve_url(repo: &str, revision: &str, filename: &str) -> String {
    let repo = repo.trim_matches('/');
    let revision = revision.trim_matches('/');
    let filename = filename.trim_start_matches('/');
    format!("https://huggingface.co/{repo}/resolve/{revision}/{filename}")
}

/// Read the optional HF bearer token from the `HF_TOKEN` environment variable.
///
/// `None` means "fetch anonymously", which is correct and expected for public
/// models. A present-but-empty / whitespace-only value is treated as `None` too
/// -- an empty `Authorization: Bearer ` header would only make HF reject the
/// request. The actual trim/empty normalisation lives in [`normalize_token`],
/// which is the pure, testable half of this.
///
/// The token is a **secret**: it is never logged, never put in an error string,
/// and (via ureq's redirect policy in [`HfFetcher`]) never forwarded off
/// `huggingface.co` to the CDN. See [`HfFetcher`].
pub fn hf_token_from_env() -> Option<String> {
    normalize_token(std::env::var(HF_TOKEN_ENV).ok())
}

/// Normalise a raw token value: trim surrounding whitespace, and collapse an
/// empty / whitespace-only value to `None`.
///
/// Split out from [`hf_token_from_env`] so the rule (and it IS a rule -- an
/// empty `Bearer ` header just gets rejected by HF) can be unit-tested without
/// mutating the process environment. Both the env path and
/// [`HfFetcher::with_token`] funnel through here, so they cannot drift apart.
pub fn normalize_token(raw: Option<String>) -> Option<String> {
    raw.map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// Something wrong with a [`Revision`] value.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RevisionError {
    /// [`Revision::commit`] was handed something that is not a commit SHA
    /// (not 7..=40 lowercase-hex chars) -- most likely a branch/tag name, which
    /// belongs in [`Revision::moving`] instead.
    #[error(
        "`{got}` is not a commit SHA (want 7..=40 lowercase-hex chars) -- \
         use Revision::moving for a branch or tag name"
    )]
    NotACommitSha {
        /// The value that was passed in.
        got: String,
    },
}

/// Is `s` a plausible git commit SHA: 7..=40 chars, all lowercase hex?
///
/// 40 is a full SHA-1 object name; 7 is git's conventional shortest
/// "unambiguous" abbreviation. We do not accept SHA-256 object names (64 chars)
/// yet -- HF revisions are SHA-1 today; widen this the day that changes.
fn is_commit_sha(s: &str) -> bool {
    (7..=40).contains(&s.len())
        && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

// ===========================================================================
// SHA-256 auto-resolution from the HF hub.
//
// The whole point: a catalog author should be able to name a `{repo, file}`
// and let the machine fetch the checksum, exactly like `huggingface_hub` /
// `transformers` do -- no more pasting a 64-char hash by hand. Two hub facts
// this leans on (hard-won, do NOT paraphrase away):
//
//  * `GET https://huggingface.co/api/models/{repo}/tree/{rev}` returns a JSON
//    array of siblings. For a git-LFS file (every real GGUF), the sibling
//    carries `"lfs": { "oid": "<sha256>", "size": <bytes> }` -- and `lfs.oid`
//    IS the sha256 we verify against. No download needed.
//  * `GET https://huggingface.co/{repo}/raw/{rev}/{file}` returns, for an LFS
//    file, the LFS *pointer* text (`oid sha256:<hex>` + `size <n>`); for a
//    small NON-LFS file it returns the actual bytes. So the raw endpoint is
//    both the fallback-when-the-tree-API-is-unavailable AND the way to handle
//    a small non-LFS file (hash the bytes it returns).
//
// The JSON/pointer parsing is pure and net-free (so it unit-tests against a
// captured fixture with zero sockets); only the HTTP calls sit behind `net`.
// ===========================================================================

/// A sha256 resolved from the HF hub, ready to drop onto an [`Artifact`].
///
/// This is the return shape of [`resolve_hf_sha256`] -- the checksum, the size,
/// and the direct-download `resolve` URL for the same file, so a caller has
/// everything needed to build (or verify) a catalog entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfResolved {
    /// Lowercase-hex sha256 of the file's bytes -- the git-LFS `oid`, the LFS
    /// pointer's `oid sha256:`, or (for a small non-LFS file) the hash of the
    /// bytes themselves. This is the value the verification gate checks against.
    pub sha256: String,
    /// The file's size in bytes, as the hub reports it (or the byte count when
    /// resolved by hashing a non-LFS file).
    pub size_bytes: u64,
    /// The direct-download (`resolve`) URL for this exact file.
    pub url: String,
}

/// What the `tree` API told us about one file. Pure result of parsing the JSON,
/// so it can be produced and asserted on with no network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfTreeLookup {
    /// A git-LFS file: `sha256` is the `lfs.oid`, `size_bytes` the `lfs.size`.
    /// This is the fast path -- the checksum is known without any download.
    Lfs {
        /// Lowercase-hex sha256 (the LFS object id).
        sha256: String,
        /// Size in bytes.
        size_bytes: u64,
    },
    /// The file exists but is NOT git-LFS (a small plain blob). Its git `oid`
    /// is a SHA-1, not the sha256 we want, so the caller must fetch the raw
    /// bytes and hash them. `size_bytes` is carried through for progress.
    NotLfs {
        /// Size in bytes as reported by the tree API.
        size_bytes: u64,
    },
    /// No sibling in the repo matched the requested filename.
    NotFound,
}

/// The tree-API URL for a repo at a revision:
/// `https://huggingface.co/api/models/{repo}/tree/{revision}`.
///
/// Same slash-tidying as [`hf_resolve_url`] so callers can be sloppy.
pub fn hf_tree_url(repo: &str, revision: &str) -> String {
    let repo = repo.trim_matches('/');
    let revision = revision.trim_matches('/');
    format!("https://huggingface.co/api/models/{repo}/tree/{revision}")
}

/// The raw-file URL for one file:
/// `https://huggingface.co/{repo}/raw/{revision}/{file}`.
///
/// For a git-LFS file this returns the LFS *pointer* text; for a small non-LFS
/// file it returns the actual bytes. Contrast [`hf_resolve_url`], which 302s to
/// the CDN with the real bytes for any file.
pub fn hf_raw_url(repo: &str, revision: &str, file: &str) -> String {
    let repo = repo.trim_matches('/');
    let revision = revision.trim_matches('/');
    let file = file.trim_start_matches('/');
    format!("https://huggingface.co/{repo}/raw/{revision}/{file}")
}

/// Parse a `tree` API JSON body and report what it says about `file`.
///
/// Pure -- no network, no I/O. This is the deterministic core of resolution, so
/// it can be tested against a captured fixture. Looks for the array element
/// whose `path` equals `file`; if it carries an `lfs.oid` that is the sha256
/// ([`HfTreeLookup::Lfs`]); otherwise the file is a plain blob
/// ([`HfTreeLookup::NotLfs`]); no match is [`HfTreeLookup::NotFound`].
///
/// # Errors
///
/// [`Error::Http`] if the body is not the JSON array shape the tree API is
/// documented to return (malformed JSON, or a non-array top level -- e.g. an
/// error object). The variant is `Http` because a shape surprise here means the
/// hub answered with something other than the tree we asked for.
pub fn parse_hf_tree(json: &str, file: &str) -> Result<HfTreeLookup, Error> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| Error::Http(format!("HF tree API returned invalid JSON: {e}")))?;
    let entries = value
        .as_array()
        .ok_or_else(|| Error::Http("HF tree API JSON was not an array".to_string()))?;

    for entry in entries {
        if entry.get("path").and_then(|p| p.as_str()) != Some(file) {
            continue;
        }
        // Matched. Prefer the LFS oid (the sha256); fall back to "plain blob".
        if let Some(oid) = entry
            .get("lfs")
            .and_then(|lfs| lfs.get("oid"))
            .and_then(|o| o.as_str())
        {
            let size_bytes = entry
                .get("lfs")
                .and_then(|lfs| lfs.get("size"))
                .and_then(|s| s.as_u64())
                .or_else(|| entry.get("size").and_then(|s| s.as_u64()))
                .unwrap_or(0);
            return Ok(HfTreeLookup::Lfs {
                sha256: oid.to_ascii_lowercase(),
                size_bytes,
            });
        }
        let size_bytes = entry.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        return Ok(HfTreeLookup::NotLfs { size_bytes });
    }
    Ok(HfTreeLookup::NotFound)
}

/// Parse a git-LFS pointer file into `(sha256, size)`, or `None` if the text is
/// not an LFS pointer (e.g. it is the actual file bytes of a non-LFS file).
///
/// Pure -- no network. An LFS pointer looks like:
///
/// ```text
/// version https://git-lfs.github.com/spec/v1
/// oid sha256:48ab3034...0201
/// size 386404992
/// ```
///
/// Requires BOTH an `oid sha256:` line and a `size` line, so it does not
/// mistake arbitrary text for a pointer. The sha is lowercased for the
/// lowercase-hex comparison the rest of the crate does.
pub fn parse_lfs_pointer(text: &str) -> Option<(String, u64)> {
    let mut sha256 = None;
    let mut size = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("oid sha256:") {
            sha256 = Some(rest.trim().to_ascii_lowercase());
        } else if let Some(rest) = line.strip_prefix("size ") {
            size = rest.trim().parse::<u64>().ok();
        }
    }
    match (sha256, size) {
        (Some(s), Some(sz)) => Some((s, sz)),
        _ => None,
    }
}

/// Cross-check a resolved sha256 against an optional pinned one.
///
/// Pure -- no network, so the defense-in-depth rule is unit-testable without a
/// socket. When `pinned` is `Some` and non-empty and differs from `resolved`,
/// that is a hard [`Error::ChecksumMismatch`] (upstream re-quantised, or the
/// pin was wrong). An empty or absent pin means "trust the resolved value".
///
/// # Errors
///
/// [`Error::ChecksumMismatch`] on a pinned-vs-resolved disagreement.
pub fn check_pinned(file: &str, resolved: &str, pinned: Option<&str>) -> Result<(), Error> {
    match pinned {
        Some(p) if !p.is_empty() && p != resolved => Err(Error::ChecksumMismatch {
            artifact: file.to_string(),
            expected: p.to_string(),
            actual: resolved.to_string(),
        }),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// The live hub calls. Only compiled when the `net` feature is on.
// ---------------------------------------------------------------------------

/// Resolve the sha256 of one HF-hosted file from the hub, anonymously, at the
/// `main` revision. The convenience front door over [`resolve_hf_sha256_rev`].
///
/// Queries the `tree` API for the git-LFS `oid` (no download); falls back to the
/// `/raw/` LFS pointer, and for a small non-LFS file hashes the bytes it
/// returns. The result is cached in-process (see [`resolve_hf_sha256_rev`]).
///
/// # Errors
///
/// * [`Error::NotFound`] -- no such file in the repo.
/// * [`Error::Http`] -- network failure, or the hub answered with an
///   unexpected shape.
///
/// # Example run (documented; hits the network, so not a default test)
///
/// ```no_run
/// # #[cfg(feature = "net")]
/// # fn demo() -> Result<(), kopitiam_models::Error> {
/// let r = kopitiam_models::resolve_hf_sha256(
///     "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
///     "smollm2-360m-instruct-q8_0.gguf",
/// )?;
/// assert_eq!(r.sha256.len(), 64);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "net")]
pub fn resolve_hf_sha256(repo: &str, file: &str) -> Result<HfResolved, Error> {
    resolve_hf_sha256_rev(repo, file, "main", None)
}

/// Resolve the sha256 of one HF-hosted file at a specific `revision`, with an
/// optional bearer `token` for gated / private repos.
///
/// Strategy, in order:
///
/// 1. Consult the in-process cache (keyed by `repo@revision/file`); a hit
///    returns immediately.
/// 2. `GET .../api/models/{repo}/tree/{revision}` and look for the file's
///    git-LFS `oid` -- the fast path, no file download.
/// 3. If the tree says the file is present but NOT LFS, or the tree API itself
///    is unreachable, `GET .../{repo}/raw/{revision}/{file}`: parse it as an LFS
///    pointer, or (a non-LFS file) hash the returned bytes.
///
/// A successful resolve is memoised so a repeated lookup (e.g. `models list`
/// then `models pull`) does not re-hit the hub.
///
/// # Errors
///
/// * [`Error::NotFound`] -- the tree API had no sibling matching `file`.
/// * [`Error::Http`] -- both the tree API and the raw fallback failed, or the
///   hub returned an unexpected shape.
#[cfg(feature = "net")]
pub fn resolve_hf_sha256_rev(
    repo: &str,
    file: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<HfResolved, Error> {
    let key = format!("{repo}@{revision}/{file}");
    if let Some(hit) = resolve_cache_get(&key) {
        return Ok(hit);
    }

    let resolved = match try_resolve_via_tree(repo, file, revision, token) {
        Ok(HfTreeLookup::Lfs { sha256, size_bytes }) => HfResolved {
            sha256,
            size_bytes,
            url: hf_resolve_url(repo, revision, file),
        },
        // Present but not LFS -> hash its raw bytes.
        Ok(HfTreeLookup::NotLfs { .. }) => resolve_via_raw(repo, file, revision, token)?,
        Ok(HfTreeLookup::NotFound) => {
            return Err(Error::NotFound(format!(
                "file `{file}` not found in HF repo `{repo}` at revision `{revision}`"
            )));
        }
        // Tree endpoint unreachable -> last-resort raw fallback; if THAT also
        // fails, surface the original tree error (the more informative one).
        Err(tree_err) => resolve_via_raw(repo, file, revision, token).map_err(|_| tree_err)?,
    };

    resolve_cache_put(key, resolved.clone());
    Ok(resolved)
}

/// Resolve straight from an [`HfSource`] declaration, applying the
/// pinned-vs-resolved cross-check via [`check_pinned`].
///
/// `pinned` is the artifact's currently-recorded sha256 (pass `None` or `""`
/// when it is to be resolved fresh). On success the returned [`HfResolved`] is
/// guaranteed consistent with any non-empty pin.
///
/// # Errors
///
/// Everything [`resolve_hf_sha256_rev`] can return, plus
/// [`Error::ChecksumMismatch`] when a non-empty `pinned` disagrees with the hub.
#[cfg(feature = "net")]
pub fn resolve_hf_source(
    src: &HfSource,
    pinned: Option<&str>,
    token: Option<&str>,
) -> Result<HfResolved, Error> {
    let resolved = resolve_hf_sha256_rev(&src.repo, &src.file, &src.revision, token)?;
    check_pinned(&src.file, &resolved.sha256, pinned)?;
    Ok(resolved)
}

/// Return a copy of `spec` with every [`Artifact::hf`]-sourced checksum resolved
/// from the hub and cross-checked against its pin.
///
/// For each artifact carrying an [`HfSource`]: resolve its sha256 (and size, and
/// url if missing), verify it against the artifact's existing `sha256` when that
/// is a non-empty pin, and write the resolved value in. Artifacts with no
/// [`HfSource`] are passed through untouched. The result is a plain [`ModelSpec`]
/// with concrete checksums, ready for [`crate::ensure_available`].
///
/// This is the bridge that lets [`crate::ensure_available_resolving`] (and thus
/// the CLI's `models pull` / `list`) get auto-SHA for free.
///
/// # Errors
///
/// Everything [`resolve_hf_source`] can return, including
/// [`Error::ChecksumMismatch`] on a pin disagreement.
#[cfg(feature = "net")]
pub fn resolve_spec(spec: &ModelSpec, token: Option<&str>) -> Result<ModelSpec, Error> {
    let mut out = spec.clone();
    for a in &mut out.artifacts {
        let Some(src) = a.hf.clone() else { continue };
        let pinned = if a.sha256.is_empty() {
            None
        } else {
            Some(a.sha256.as_str())
        };
        let resolved = resolve_hf_source(&src, pinned, token)?;
        a.sha256 = resolved.sha256;
        if a.size_bytes == 0 {
            a.size_bytes = resolved.size_bytes;
        }
        if a.url.is_empty() {
            a.url = resolved.url;
        }
    }
    Ok(out)
}

/// Query the tree API and parse it. `Ok(..)` on a well-formed reply (including
/// [`HfTreeLookup::NotFound`]); `Err` only on transport / shape failure.
#[cfg(feature = "net")]
fn try_resolve_via_tree(
    repo: &str,
    file: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<HfTreeLookup, Error> {
    let url = hf_tree_url(repo, revision);
    let bytes = hf_get_bytes(&url, token)?;
    let json = String::from_utf8_lossy(&bytes);
    parse_hf_tree(&json, file)
}

/// Resolve via the `/raw/` endpoint: parse an LFS pointer, or hash the bytes of
/// a non-LFS file.
#[cfg(feature = "net")]
fn resolve_via_raw(
    repo: &str,
    file: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<HfResolved, Error> {
    let raw_url = hf_raw_url(repo, revision, file);
    let bytes = hf_get_bytes(&raw_url, token)?;
    let url = hf_resolve_url(repo, revision, file);

    // An LFS file's /raw/ body is the pointer text; a non-LFS file's is the
    // real bytes. Try the pointer first, else hash the bytes we got.
    if let Some((sha256, size_bytes)) = std::str::from_utf8(&bytes).ok().and_then(parse_lfs_pointer)
    {
        return Ok(HfResolved {
            sha256,
            size_bytes,
            url,
        });
    }
    Ok(HfResolved {
        sha256: sha256_hex(&bytes),
        size_bytes: bytes.len() as u64,
        url,
    })
}

/// One small GET, whole body into memory. Only used for tree JSON and LFS
/// pointers / small non-LFS files -- all small, so buffering is fine (the big
/// GGUF bytes go through the streaming [`crate::Fetcher`], never here).
///
/// Follows redirects with the same same-host bearer policy as [`HfFetcher`], so
/// a token never leaks to the CDN. Never interpolates the token into an error.
#[cfg(feature = "net")]
fn hf_get_bytes(url: &str, token: Option<&str>) -> Result<Vec<u8>, Error> {
    use std::io::Read;

    use ureq::config::RedirectAuthHeaders;

    let agent = ureq::Agent::config_builder()
        .redirect_auth_headers(RedirectAuthHeaders::SameHost)
        .build()
        .new_agent();

    let mut request = agent.get(url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }

    let response = request
        .call()
        .map_err(|e| Error::Http(format!("GET {url} failed: {e}")))?;

    let mut buf = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| Error::Http(format!("reading body of {url}: {e}")))?;
    Ok(buf)
}

/// sha256 of a byte slice as lowercase hex. Local to the resolver (the store's
/// streaming `sha256_file` is for on-disk artifacts); these buffers are small.
#[cfg(feature = "net")]
fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;

    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Process-wide memo of resolved shas, so `list` then `pull` (or repeated
/// pulls) don't re-hit the hub. Trivial and stdlib-only: a `Mutex<HashMap>`
/// behind a `OnceLock`. Keyed by `repo@revision/file`.
#[cfg(feature = "net")]
fn resolve_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, HfResolved>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, HfResolved>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[cfg(feature = "net")]
fn resolve_cache_get(key: &str) -> Option<HfResolved> {
    resolve_cache()
        .lock()
        .ok()
        .and_then(|m| m.get(key).cloned())
}

#[cfg(feature = "net")]
fn resolve_cache_put(key: String, value: HfResolved) {
    if let Ok(mut m) = resolve_cache().lock() {
        m.insert(key, value);
    }
}

// ---------------------------------------------------------------------------
// The HF fetcher. Only compiled when the `net` feature is on.
// ---------------------------------------------------------------------------

/// First-class HuggingFace fetcher: plain streaming HTTP **plus** an optional
/// `Authorization: Bearer` header from `HF_TOKEN`, for gated / private repos.
///
/// Only exists when the `net` feature is enabled (on by default). Switch the
/// feature off for a pure bring-your-own, offline build with no HTTP stack.
///
/// # What it adds over [`crate::HttpFetcher`]
///
/// Just the auth. For a public model the two behave the same, so you could use
/// either. The point of a dedicated type is the gated-repo case: this one reads
/// [`hf_token_from_env`] once at construction and sends `Bearer <token>` so
/// gated / private repos actually resolve.
///
/// # Redirects and where the token is (and is NOT) sent
///
/// The HF `resolve` URL 302-redirects to a CDN (see the module docs). This
/// fetcher follows redirects, and pins ureq's redirect-auth policy to
/// **same-host**: the `Authorization` header is replayed only if a redirect
/// stays on `huggingface.co`, and is **dropped** the moment it hops to the CDN.
/// That is exactly right -- HF checks the token on the first request and hands
/// back a *presigned* CDN URL that needs no auth of its own, so forwarding the
/// secret there would be a needless leak. The token therefore reaches
/// `huggingface.co` only, never the CDN.
///
/// The token is never logged and never placed in an [`crate::Error`] string.
///
/// # The ring/rustls caveat
///
/// Same as [`crate::HttpFetcher`]: "rustls" is a pure-Rust TLS *protocol* but
/// ureq's `rustls` feature uses the `ring` crypto provider (C + perlasm),
/// accepted on purpose and chope-d behind the off-by-default-able `net` feature.
/// See `docs/ai-decisions/AID-0013`.
#[cfg(feature = "net")]
pub struct HfFetcher {
    /// The bearer token, if `HF_TOKEN` was set. `None` == anonymous. Kept
    /// private so it cannot be read back out and accidentally logged.
    token: Option<String>,
}

#[cfg(feature = "net")]
impl HfFetcher {
    /// Make a fetcher, reading `HF_TOKEN` from the environment once, now. Cheap
    /// -- no connection is opened until a [`crate::Fetcher::fetch`] call runs.
    ///
    /// If `HF_TOKEN` is set it is used for every fetch; if not, fetches go out
    /// anonymously (fine for public models).
    pub fn new() -> Self {
        Self {
            token: hf_token_from_env(),
        }
    }

    /// Make a fetcher with an explicit token, bypassing the environment. Mainly
    /// for callers that get the token from somewhere other than `HF_TOKEN` (a
    /// keychain, a config file). `None` means anonymous.
    pub fn with_token(token: Option<String>) -> Self {
        Self {
            token: normalize_token(token),
        }
    }

    /// Does this fetcher carry a token (i.e. will it send `Authorization`)?
    /// Returns only the yes/no -- the token value itself never leaves the type.
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }
}

#[cfg(feature = "net")]
impl Default for HfFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "net")]
impl crate::Fetcher for HfFetcher {
    fn fetch(
        &self,
        url: &str,
        dest: &std::path::Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), crate::Error> {
        use std::io::{Read, Write};

        use ureq::config::RedirectAuthHeaders;

        // Make the parent directory first, so the write below has somewhere to
        // land.
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Pin the redirect-auth policy to same-host: the bearer token is
        // replayed only while a redirect stays on huggingface.co, and dropped
        // the instant it hops to the presigned CDN URL. Following redirects is
        // on (ureq default), which the resolve->CDN 302 needs.
        let agent = ureq::Agent::config_builder()
            .redirect_auth_headers(RedirectAuthHeaders::SameHost)
            .build()
            .new_agent();

        let mut request = agent.get(url);
        if let Some(token) = &self.token {
            // Bearer auth for gated / private repos. Never logged.
            request = request.header("Authorization", format!("Bearer {token}"));
        }

        // ureq 3: any transport / non-2xx status comes back as Err, flattened
        // into our own Error::Http so callers never depend on ureq's error type.
        // Deliberately NOT interpolating the token into this string.
        let response = request
            .call()
            .map_err(|e| crate::Error::Http(format!("GET {url} failed: {e}")))?;

        // Content-Length for progress `total`, if the server gave one. After a
        // redirect this is the CDN's length -- the real file size.
        let total: Option<u64> = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok());

        // Stream body -> file in chunks. Streamed, never buffered whole, because
        // a GGUF is big (multi-GB in the large-model case).
        let mut reader = response.into_body().into_reader();
        let mut file = std::fs::File::create(dest)?;
        let mut buf = [0u8; 64 * 1024];
        let mut downloaded: u64 = 0;

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| crate::Error::Http(format!("reading body of {url}: {e}")))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            progress(downloaded, total);
        }

        file.flush()?;
        Ok(())
    }
}
