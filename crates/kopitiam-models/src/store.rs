//! Where models sit on disk, and how we check they are the real thing.
//!
//! [`ModelStore`] is just a rooted directory plus the rules for laying files
//! out inside it and hashing them. No network here -- that is the
//! [`crate::Fetcher`]'s job. This half of the crate is fully offline and fully
//! testable with a `tempdir`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::catalog::{Artifact, ModelSpec};
use crate::error::Error;

/// The on-disk home for acquired models, and the thing that can tell you what
/// is present and whether it verifies.
///
/// Layout inside the root is flat-per-model: every artifact of a spec goes into
/// `<root>/<spec.id>/<artifact.filename>`. Namespacing by `spec.id` means two
/// models that happen to name a file the same (say both ship a `tokenizer.json`)
/// don't step on each other hor.
pub struct ModelStore {
    root: PathBuf,
}

impl ModelStore {
    /// Build a store rooted at KOPITIAM's default model cache, following the
    /// XDG Base Directory spec: `$XDG_CACHE_HOME/kopitiam/models`, and if
    /// `XDG_CACHE_HOME` is not set, fall back to `$HOME/.cache/kopitiam/models`.
    ///
    /// This does NOT create the directory yet -- it only resolves the path.
    /// The directory gets made lazily when something is first written into it
    /// (see [`crate::ensure_available`]). We hand-roll the XDG lookup with
    /// `std::env` on purpose, to avoid pulling in the `directories` crate for
    /// something this small.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] if neither `XDG_CACHE_HOME` nor `HOME` is set in the
    /// environment -- without at least one of them there is no sensible place to
    /// resolve to, and we would rather say so loudly than guess.
    pub fn with_default_root() -> Result<Self, Error> {
        let base = if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME")
            .filter(|v| !v.is_empty())
        {
            PathBuf::from(xdg)
        } else if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty())
        {
            PathBuf::from(home).join(".cache")
        } else {
            return Err(Error::NotFound(
                "cannot resolve default model cache: neither XDG_CACHE_HOME nor \
                 HOME is set -- pass an explicit root with ModelStore::with_root"
                    .to_string(),
            ));
        };
        Ok(Self {
            root: base.join("kopitiam").join("models"),
        })
    }

    /// Build a store rooted at an explicit path. This is the bring-your-own /
    /// tests entry point -- point it at a `tempdir`, or at wherever the user
    /// already keep their `.gguf` files.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory this store is anchored at.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where a given artifact of a given spec should live:
    /// `<root>/<spec.id>/<artifact.filename>`.
    ///
    /// Pure path arithmetic -- it does not check anything is actually there.
    pub fn artifact_path(&self, spec: &ModelSpec, a: &Artifact) -> PathBuf {
        self.root.join(&spec.id).join(&a.filename)
    }

    /// Is every artifact of `spec` sitting on disk?
    ///
    /// This only checks *existence* of each file hor, NOT that the bytes are
    /// correct. For "present AND correct" you want [`ModelStore::verify`]. A
    /// `true` here with a `false` verify is exactly the corrupt / wrong-file
    /// case.
    pub fn is_present(&self, spec: &ModelSpec) -> bool {
        spec.artifacts
            .iter()
            .all(|a| self.artifact_path(spec, a).is_file())
    }

    /// Hash every artifact of `spec` and check it against the catalog's sha256.
    ///
    /// This is the verification gate. It walks each artifact, streams the file
    /// through SHA-256, and compares lowercase-hex against
    /// [`Artifact::sha256`]. The first artifact that does not match stops the
    /// whole thing with [`Error::ChecksumMismatch`].
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] if an artifact file cannot be opened or read (this is
    ///   also what you get if the file is simply not there -- use
    ///   [`ModelStore::is_present`] first if you want to tell "missing" apart
    ///   from "unreadable").
    /// * [`Error::ChecksumMismatch`] if a file's bytes hash to something other
    ///   than the catalog's recorded value. Remember: with the shipped catalog
    ///   this is the *expected* result, because those checksums are placeholders
    ///   (see [`crate::Catalog::builtin`]).
    pub fn verify(&self, spec: &ModelSpec) -> Result<(), Error> {
        for a in &spec.artifacts {
            let path = self.artifact_path(spec, a);
            let actual = sha256_file(&path)?;
            if actual != a.sha256 {
                return Err(Error::ChecksumMismatch {
                    artifact: a.filename.clone(),
                    expected: a.sha256.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// Stream a file through SHA-256 and return the digest as lowercase hex.
///
/// Streamed in 64 KiB chunks, never slurped whole into memory -- a `.gguf` can
/// be many hundreds of MB, so reading it all into a `Vec<u8>` just to hash it
/// would be quite the waste sia.
pub(crate) fn sha256_file(path: &Path) -> Result<String, Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Bytes -> lowercase hex string. Small helper so we don't pull in a hex crate
/// just for this one spot.
fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // width-2, zero-padded hex per byte -- write! to a String cannot fail.
        let _ = write!(s, "{b:02x}");
    }
    s
}
