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

use crate::catalog::{Architecture, Artifact, ModelSpec};

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
