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
    /// Build a store rooted at KOPITIAM's default model cache.
    ///
    /// An explicit `$XDG_CACHE_HOME` override wins on every platform. Otherwise
    /// the base is the OS's conventional per-user cache directory:
    /// * **Unix / macOS / Termux**: `$HOME/.cache` (Termux sets `HOME`).
    /// * **Windows**: `%LOCALAPPDATA%` (Windows has no `HOME`/XDG; falling back
    ///   to `HOME` there is why `kopitiam models` failed with a NotFound about
    ///   XDG). `%USERPROFILE%\.cache` is a secondary fallback.
    ///
    /// The final path is `<base>/kopitiam/models`. This does NOT create the
    /// directory yet -- it only resolves the path; the directory is made lazily
    /// on first write (see [`crate::ensure_available`]). We hand-roll the lookup
    /// with `std::env` on purpose, to avoid pulling in the `directories` crate.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] only when none of the platform's location variables
    /// are set (e.g. on Windows neither `LOCALAPPDATA` nor `USERPROFILE`; on Unix
    /// no `HOME`) -- rather than guess, pass an explicit root with
    /// [`ModelStore::with_root`].
    pub fn with_default_root() -> Result<Self, Error> {
        let base = match std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
            Some(xdg) => PathBuf::from(xdg),
            None => Self::platform_cache_dir()?,
        };
        Ok(Self {
            root: base.join("kopitiam").join("models"),
        })
    }

    /// The OS cache directory on Windows: `%LOCALAPPDATA%`, else
    /// `%USERPROFILE%\.cache`.
    #[cfg(windows)]
    fn platform_cache_dir() -> Result<PathBuf, Error> {
        if let Some(local) = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()) {
            Ok(PathBuf::from(local))
        } else if let Some(profile) = std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()) {
            Ok(PathBuf::from(profile).join(".cache"))
        } else {
            Err(Error::NotFound(
                "cannot resolve default model cache: neither LOCALAPPDATA nor \
                 USERPROFILE is set -- pass an explicit root with \
                 ModelStore::with_root"
                    .to_string(),
            ))
        }
    }

    /// The OS cache directory on Unix / macOS / Termux: `$HOME/.cache`.
    #[cfg(not(windows))]
    fn platform_cache_dir() -> Result<PathBuf, Error> {
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            Ok(PathBuf::from(home).join(".cache"))
        } else {
            Err(Error::NotFound(
                "cannot resolve default model cache: HOME is not set -- pass an \
                 explicit root with ModelStore::with_root"
                    .to_string(),
            ))
        }
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
