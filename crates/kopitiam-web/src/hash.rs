use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A SHA-256 digest of retrieved content, recorded as part of a [`Provenance`].
///
/// # Why a hash belongs in provenance at all
///
/// The retrieval timestamp says *when* we looked. The hash says *what we saw*.
/// Together they make a claim falsifiable: given a KOPITIAM knowledge graph
/// asserting "on 2026-07-14, Brave's top hit for `write-ahead logging` said X",
/// a reader can re-run the query, hash the answer, and discover that the web
/// has since changed its mind. Without the hash, "the source said X" is
/// unverifiable folklore; the page is free to have been silently rewritten.
///
/// This matters more in science than almost anywhere else. A citation that
/// cannot be checked is not a citation.
///
/// # SHA-256, specifically
///
/// Not a fast non-cryptographic hash (FNV, xxhash, ahash): a provenance record
/// is an integrity claim, and an integrity claim built on a 64-bit hash that
/// anyone can collide on demand is theatre. SHA-256 from RustCrypto's `sha2`
/// is pure Rust with no C dependency, satisfying the Pure Rust Core rule.
///
/// [`Provenance`]: crate::Provenance
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContentHash(String);

impl ContentHash {
    /// Hashes `bytes` with SHA-256.
    pub fn of(bytes: impl AsRef<[u8]>) -> Self {
        let digest = Sha256::digest(bytes.as_ref());
        Self(format!("{digest:x}"))
    }

    /// Hashes the fields of one search result as the engine returned them.
    ///
    /// The separator is a NUL byte rather than, say, a newline, because NUL
    /// cannot occur in any of the three fields. That makes the encoding
    /// unambiguous: no combination of title, URL and snippet can be rearranged
    /// into a different triple with the same hash input. (Hashing
    /// `title + url + snippet` naively would let `("ab", "c")` and `("a",
    /// "bc")` collide, which would quietly weaken exactly the property this
    /// type exists to provide.)
    pub fn of_result(title: &str, url: &str, snippet: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(title.as_bytes());
        hasher.update([0u8]);
        hasher.update(url.as_bytes());
        hasher.update([0u8]);
        hasher.update(snippet.as_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }

    /// The digest as lowercase hex, without the `sha256:` prefix that
    /// [`Display`](fmt::Display) adds.
    pub fn as_hex(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContentHash {
    /// Renders as `sha256:<hex>`.
    ///
    /// The algorithm is written out rather than assumed, so that a hash copied
    /// into a document, a commit message or a paper stays self-describing if we
    /// ever add a second algorithm.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_known_sha256_of_the_empty_input() {
        // The canonical SHA-256 test vector. If this ever fails, the hash we
        // are recording in provenance is not the hash we are claiming to.
        assert_eq!(
            ContentHash::of(b"").as_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn matches_the_known_sha256_of_abc() {
        assert_eq!(
            ContentHash::of(b"abc").as_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(ContentHash::of("kopitiam"), ContentHash::of("kopitiam"));
        assert_ne!(ContentHash::of("kopitiam"), ContentHash::of("kopitiam "));
    }

    #[test]
    fn field_boundaries_cannot_be_shifted_without_changing_the_hash() {
        // The bug this guards: a naive concatenation would make these two
        // different results hash identically.
        let a = ContentHash::of_result("ab", "c", "d");
        let b = ContentHash::of_result("a", "bc", "d");
        assert_ne!(a, b);
    }

    #[test]
    fn displays_with_its_algorithm() {
        let hash = ContentHash::of(b"");
        assert!(hash.to_string().starts_with("sha256:"));
    }

    #[test]
    fn round_trips_through_json() {
        let hash = ContentHash::of(b"abc");
        let json = serde_json::to_string(&hash).unwrap();
        let back: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(hash, back);
    }
}
