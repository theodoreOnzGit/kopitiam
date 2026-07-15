//! The network seam, and the autofetch-first acquire path.
//!
//! Everything network in this crate goes through the [`Fetcher`] trait. The
//! reason is offline-testability: [`ensure_available`] takes a `&dyn Fetcher`,
//! so a test can pass a fake that writes known bytes from a local buffer and
//! never touches a socket. The one real implementation, [`HttpFetcher`], lives
//! behind the default-on `net` feature; switch that feature off and the crate
//! still builds and tests, just with BYO-only (bring-your-own) acquisition.
//!
//! This mirrors how the rest of KOPITIAM keep its I/O behind a seam:
//! `kopitiam-ai`'s `ModelAdapter` is the one boundary a model is called
//! through, and `kopitiam-loader` deliberately stops one step short of
//! depending on `kopitiam-tensor`. Same idea here -- the acquisition core
//! doesn't know or care *how* bytes arrive, only that they arrive and then
//! verify.

use std::path::{Path, PathBuf};

use crate::catalog::ModelSpec;
use crate::error::Error;
use crate::store::ModelStore;

/// The one network touch-point, behind a trait so the core stay offline.
///
/// Implement this to teach [`ensure_available`] a new way to get bytes onto
/// disk (HTTP, a local mirror, a corporate artifact cache, whatever). The core
/// only ever calls [`Fetcher::fetch`]; it never assume HTTP.
pub trait Fetcher {
    /// Download `url` into `dest`, creating any missing parent directories
    /// along the way.
    ///
    /// `progress(downloaded, total)` gets called every now and then as bytes
    /// come in: `downloaded` is the running byte count, and `total` is the
    /// full size when the source tell us (e.g. a `Content-Length`), or `None`
    /// when it doesn't. A no-op closure is perfectly fine if the caller don't
    /// care about progress.
    ///
    /// # Contract
    ///
    /// On `Ok(())`, the full bytes must be sitting at `dest`. This function does
    /// NOT verify the checksum -- that is [`ensure_available`]'s job, done after
    /// the fetch returns. On error, whether a partial file is left at `dest` is
    /// implementation-defined, so callers must not trust a `dest` after an
    /// `Err`.
    fn fetch(
        &self,
        url: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), Error>;
}

/// A resolved, on-disk, verified model -- ready to hand to `kopitiam-loader` /
/// `kopitiam-ai`.
///
/// The whole point of getting one of these back is the guarantee that comes
/// with it: every path in [`AcquiredModel::artifact_paths`] exists AND its bytes
/// hashed to the catalog's sha256. No unverified path ever ends up in here.
#[derive(Debug, Clone)]
pub struct AcquiredModel {
    /// The spec that was acquired.
    pub spec: ModelSpec,
    /// One path per artifact, in the SAME order as `spec.artifacts`. So
    /// `artifact_paths[i]` is the on-disk location of `spec.artifacts[i]` --
    /// the caller can zip the two lists together and trust the pairing.
    pub artifact_paths: Vec<PathBuf>,
}

/// Autofetch-first entry point: make sure every artifact of `spec` is present
/// AND verified, fetching only whatever is missing.
///
/// The flow, per artifact:
///
/// 1. **Already on disk?** Then verify it straight away. If it verifies, we are
///    done with this artifact and the fetcher is NEVER called for it. This is
///    the bring-your-own (BYO) short-circuit -- if you already dropped every
///    correct file into the store yourself, `ensure_available` does zero
///    network work, and would work fine even with a fetcher that panics.
///    If a present file does NOT verify, that is a corrupt / wrong file, and we
///    fail with [`Error::ChecksumMismatch`] -- we do NOT silently re-download
///    over a file the caller put there.
/// 2. **Missing?** Then call `fetcher` to pull it, then verify the freshly
///    landed bytes. A post-fetch mismatch is again [`Error::ChecksumMismatch`]
///    (the download gave us the wrong thing).
///
/// So the fetcher is only ever touched for genuinely-missing files, and the
/// returned [`AcquiredModel`] is verified end to end.
///
/// # A heads-up about the built-in catalog
///
/// Because [`crate::Catalog::builtin`]'s checksums are placeholders (64 zeros),
/// calling this on a shipped entry will fetch the real bytes and then fail with
/// [`Error::ChecksumMismatch`] -- the real hash cannot equal 64 zeros. That is
/// expected until the true sha256 is recorded. It works fully today with any
/// spec whose checksum is real (which is exactly what the tests do).
///
/// # Errors
///
/// * [`Error::ChecksumMismatch`] -- a present-but-wrong file, or a download that
///   produced the wrong bytes.
/// * [`Error::Http`] -- the fetcher failed to get the bytes.
/// * [`Error::Io`] -- a filesystem problem making directories or reading files.
pub fn ensure_available(
    store: &ModelStore,
    spec: &ModelSpec,
    fetcher: &dyn Fetcher,
) -> Result<AcquiredModel, Error> {
    let mut artifact_paths = Vec::with_capacity(spec.artifacts.len());

    for a in &spec.artifacts {
        let path = store.artifact_path(spec, a);

        if path.is_file() {
            // BYO short-circuit: it's already here, so just check it. Whether
            // it verifies or not, we do NOT call the fetcher -- present means
            // present. A bad present file is a hard error, not a re-download.
            let actual = crate::store::sha256_file(&path)?;
            if actual != a.sha256 {
                return Err(Error::ChecksumMismatch {
                    artifact: a.filename.clone(),
                    expected: a.sha256.clone(),
                    actual,
                });
            }
        } else {
            // Missing -> fetch it, then verify what landed.
            let mut noop = |_downloaded: u64, _total: Option<u64>| {};
            fetcher.fetch(&a.url, &path, &mut noop)?;

            let actual = crate::store::sha256_file(&path)?;
            if actual != a.sha256 {
                return Err(Error::ChecksumMismatch {
                    artifact: a.filename.clone(),
                    expected: a.sha256.clone(),
                    actual,
                });
            }
        }

        artifact_paths.push(path);
    }

    Ok(AcquiredModel {
        spec: spec.clone(),
        artifact_paths,
    })
}

// ---------------------------------------------------------------------------
// The one real Fetcher. Only compiled when the `net` feature is on.
// ---------------------------------------------------------------------------

/// Autofetch-first HTTP implementation, built on `ureq` + `rustls`.
///
/// Only exists when the `net` feature is enabled (it is on by default). Switch
/// the feature off for a pure BYO, offline build with no HTTP stack, no TLS, no
/// `ring` compiled at all.
///
/// # The ring/rustls caveat (must say, don't hide)
///
/// "rustls" does NOT mean "no C" ah. rustls is a pure-Rust TLS *protocol*
/// implementation, but ureq's `rustls` feature picks the `ring` provider (C +
/// perlasm) to do the actual crypto. `ring` is accepted on purpose -- it
/// cross-compiles clean to the targets KOPITIAM care about (including
/// Android/aarch64, where OpenSSL famously cannot), and it stay chope-d behind
/// this off-by-default-able feature, so the BYO-only build never compile even
/// one byte of it. Same tradeoff kopitiam-web already made; see
/// `docs/ai-decisions/AID-0013`.
#[cfg(feature = "net")]
pub struct HttpFetcher;

#[cfg(feature = "net")]
impl HttpFetcher {
    /// Make a new HTTP fetcher. Cheap -- no connection is opened until a
    /// [`Fetcher::fetch`] call actually runs.
    pub fn new() -> Self {
        HttpFetcher
    }
}

#[cfg(feature = "net")]
impl Default for HttpFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "net")]
impl Fetcher for HttpFetcher {
    fn fetch(
        &self,
        url: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), Error> {
        use std::io::{Read, Write};

        // Make the parent directory first (e.g. `<root>/<spec.id>/`), so the
        // write below has somewhere to land.
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // ureq 3: any transport / non-2xx status comes back as an Err, which we
        // flatten into our own Error::Http(String) so callers never have to
        // depend on ureq's error type.
        let response = ureq::get(url)
            .call()
            .map_err(|e| Error::Http(format!("GET {url} failed: {e}")))?;

        // Pull Content-Length out for progress `total`, if the server gave one.
        let total: Option<u64> = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok());

        // Stream body -> file in chunks, reporting progress as we go. Streamed,
        // not buffered whole, because a model file is big.
        let mut reader = response.into_body().into_reader();
        let mut file = std::fs::File::create(dest)?;
        let mut buf = [0u8; 64 * 1024];
        let mut downloaded: u64 = 0;

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| Error::Http(format!("reading body of {url}: {e}")))?;
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
