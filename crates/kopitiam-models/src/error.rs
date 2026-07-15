//! The crate's one and only typed error surface.
//!
//! On purpose `thiserror`-based, not `anyhow`: this is a public API boundary
//! leh, and this workspace's convention is the caller must be able to match on
//! concrete variants (e.g. "the download itself failed" is a different story
//! from "the bytes come already but the checksum wrong") instead of catching
//! one opaque `anyhow::Error`. `anyhow` can appear in the binaries sitting above
//! this crate; it never appears in this crate's public signatures.

use thiserror::Error;

/// Everything that acquisition can fail with.
///
/// More variants may come over time, but the four below are frozen by the
/// public-API contract -- confirm plus chop, won't rename.
#[derive(Debug, Error)]
pub enum Error {
    /// A filesystem operation kena problem -- making the cache directory,
    /// opening a model file to hash it, writing a downloaded artifact, all that.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The network layer failed: a non-success HTTP status, a transport error, a
    /// DNS failure, a TLS handshake that cannot go through. The string carry the
    /// underlying detail. Only ever come out of a [`crate::Fetcher`]
    /// implementation (e.g. [`crate::HttpFetcher`]); the offline core never
    /// throw this one.
    #[error("http error: {0}")]
    Http(String),

    /// An artifact's bytes on disk did not hash to the sha256 written down in the
    /// catalog. This is the verification gate hor: it fire whether the bytes come
    /// from a fresh download or was already sitting there (a corrupt or wrong
    /// bring-your-own file). A mismatch is always fatal -- acquisition confirm
    /// never hand back an unverified path.
    ///
    /// Take note: with the shipped catalog this is actually the *expected*
    /// outcome, because the catalog's checksums are all placeholder (see
    /// [`crate::Catalog::builtin`]).
    #[error(
        "checksum mismatch for artifact `{artifact}`: expected {expected}, got {actual}"
    )]
    ChecksumMismatch {
        /// The artifact's local filename.
        artifact: String,
        /// The lowercase-hex sha256 the catalog promise.
        expected: String,
        /// The lowercase-hex sha256 the on-disk bytes really give.
        actual: String,
    },

    /// Something you ask for cannot be found: an unknown catalog id, or a
    /// bring-your-own artifact that is missing from disk with no way to fetch it
    /// (e.g. the `net` feature is off and no `Fetcher` managed to produce the
    /// file).
    #[error("not found: {0}")]
    NotFound(String),
}
