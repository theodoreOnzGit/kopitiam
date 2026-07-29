//! # `manifest` -- the content-addressed model store, on disk
//!
//! **Upstream:** the whole `manifest/` package --
//! `crates/kopitiam-ai/vendor/ollama/manifest/manifest.go`, `layer.go`,
//! `paths.go` -- plus the store half of
//! `crates/kopitiam-ai/vendor/ollama/server/images.go` (`GetModel`'s blob
//! reads, `CopyModel`, `deleteUnusedLayers`, `PruneLayers`, `verifyBlob`,
//! `GetSHA256Digest`).
//!
//! ## What this thing actually is
//!
//! Two directories under one root (`~/.ollama/models` by default):
//!
//! ```text
//!   <root>/
//!     manifests/<host>/<namespace>/<model>/<tag>     <- small JSON file
//!     blobs/sha256-<64 hex>                          <- the actual bytes
//! ```
//!
//! A manifest is a **list of layers**, and every layer is nothing but a
//! `sha256:` digest, a media type and a size. The digest *is* the address --
//! the bytes live at `blobs/sha256-<hex>` and nowhere else. That indirection is
//! the whole point, so say it plainly:
//!
//! > **Two tags that share weights store those weights once.**
//!
//! `qwen3:0.6b` and `my-qwen:latest` built `FROM qwen3:0.6b` are two manifest
//! files of maybe 1 KB each, both pointing at the same 400 MB blob. Delete one
//! tag, the blob stays (still got referrer). Delete the second one, only then
//! the blob goes. [`Store::remove_unused_layers`] is where that rule lives, and
//! `two_tags_sharing_a_layer_store_the_bytes_once` in the tests is where it is
//! proven, not just claimed.
//!
//! ## Why `sha256-` on disk but `sha256:` in the JSON
//!
//! Not a typo, and **don't go "tidy" it**. `:` is an illegal character in a
//! Windows filename (NTFS reads it as an alternate-data-stream separator), so a
//! blob named `sha256:abcd...` simply cannot exist on Windows. Upstream handles
//! this in `BlobsPath` with a blunt `strings.ReplaceAll(digest, ":", "-")`; we
//! do the same in [`Digest::blob_filename`]. KOPITIAM runs on Windows *and* on
//! Termux/Android, so this one is load-bearing on one of our two main platforms
//! hor.
//!
//! Same reason every string this module hands back for display goes through
//! [`to_slash`]: a store path printed on Windows must read the same as on
//! Termux, else logs and beads from the two platforms cannot be diffed.
//!
//! ## Security: everything here takes untrusted input
//!
//! A model name and a layer digest both arrive **off the network** during a
//! pull. Both get used to build filesystem paths. So both are gated:
//!
//! * **Names** go through [`Name::filepath`], which returns `None` unless every
//!   part is well-formed -- no `/`, no `\`, no leading `.` -- so a name can never
//!   contain `..` and can never climb out of `manifests/`. We don't re-parse
//!   names here; `crate::name` already owns that grammar.
//! * **Digests** go through [`Digest::parse`], and [`Store::blob_path`] accepts
//!   *only* a parsed [`Digest`], never a `&str`. That is deliberate: the type
//!   system, not a code review, is what stops `../../../etc/passwd` from
//!   reaching `blobs/`.
//!
//! The traversal attempts are tested
//! (`a_name_that_tries_to_climb_out_of_the_store_is_refused`,
//! `a_digest_that_tries_to_climb_out_of_the_store_is_refused`). Whoever touches
//! path building here must keep those green.
//!
//! ## Hashing is injected, not owned
//!
//! This module never computes a SHA-256 itself -- it takes one, through the
//! [`Sha256Hasher`] trait. KOPITIAM hasn't decided this crate's hash dependency
//! yet, and hand-rolling a hash would be worse than useless. So the store does
//! the *store* part (where the bytes go, what the file is called, already there
//! or not) and the caller brings the arithmetic. Wire `sha2::Sha256` to
//! [`Sha256Hasher`] in about six lines the day the dep lands.
//!
//! ## What is deliberately NOT here
//!
//! The network. `server/download.go` and `server/upload.go` -- pull, push,
//! registry auth, chunked resume -- are a later stage. This module owns the
//! **on-disk store only**: read, write, enumerate, verify, delete, size. It does
//! no HTTP and needs no network to be fully tested, which is exactly the
//! property that makes it testable at all.
//!
//! Ported against ollama `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};

use crate::api::ConfigV2;
use crate::format::human_bytes;
use crate::name::Name;

// ---------------------------------------------------------------------------
// Constants. Every one of these names its upstream source -- a magic string in
// a store format is a compatibility contract, not a style choice.
// ---------------------------------------------------------------------------

/// The only schema version ollama writes. **Upstream:** `WriteManifest` in
/// `manifest/manifest.go` hardcodes `SchemaVersion: 2`.
pub const SCHEMA_VERSION: i32 = 2;

/// The manifest's own media type. **Upstream:** `WriteManifest`, and the
/// `Accept` header in `pullModelManifest` (`server/images.go`).
///
/// Yah it says *docker* -- ollama's store is an OCI/Docker registry v2 manifest
/// with ollama-specific layer media types inside. That compatibility is exactly
/// why the string must stay byte-for-byte.
pub const MEDIA_TYPE_MANIFEST: &str = "application/vnd.docker.distribution.manifest.v2+json";

/// Media type of the **config** layer -- the blob holding [`ConfigV2`].
/// **Upstream:** `server/create.go:1438`,
/// `manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")`.
pub const MEDIA_TYPE_CONFIG: &str = "application/vnd.docker.container.image.v1+json";

/// The weights themselves (GGUF / safetensors). **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_MODEL: &str = "application/vnd.ollama.image.model";
/// A speculative-decoding draft model. **Upstream:** `manifest.MediaTypeImageDraft`.
pub const MEDIA_TYPE_DRAFT: &str = "application/vnd.ollama.image.draft";
/// A single tensor, split out. **Upstream:** `manifest.MediaTypeImageTensor`.
pub const MEDIA_TYPE_TENSOR: &str = "application/vnd.ollama.image.tensor";
/// A LoRA / adapter. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_ADAPTER: &str = "application/vnd.ollama.image.adapter";
/// A vision projector (mmproj). **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_PROJECTOR: &str = "application/vnd.ollama.image.projector";
/// The Go prompt template. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_TEMPLATE: &str = "application/vnd.ollama.image.template";
/// Older spelling of the template layer, still honoured on read.
/// **Upstream:** `server/images.go` `GetModel` treats it identically to `.template`.
pub const MEDIA_TYPE_PROMPT: &str = "application/vnd.ollama.image.prompt";
/// The `SYSTEM` string. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_SYSTEM: &str = "application/vnd.ollama.image.system";
/// `PARAMETER` values, as a JSON object. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_PARAMS: &str = "application/vnd.ollama.image.params";
/// Baked-in few-shot `MESSAGE`s. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_MESSAGES: &str = "application/vnd.ollama.image.messages";
/// A `LICENSE` blob. Can have several. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_LICENSE: &str = "application/vnd.ollama.image.license";
/// A named side-car JSON blob, looked up by [`Layer::name`].
/// **Upstream:** `Manifest.ReadConfigJSON` in `manifest/manifest.go`.
pub const MEDIA_TYPE_JSON: &str = "application/vnd.ollama.image.json";
/// Deprecated since ollama 0.1.2 and ignored on read. Kept so a store written by
/// some ancient ollama still round-trips. **Upstream:** `server/images.go` `GetModel`.
pub const MEDIA_TYPE_EMBED: &str = "application/vnd.ollama.image.embed";

/// Subdirectory holding manifest JSON files. **Upstream:** `manifest.Path()`.
pub const MANIFESTS_DIR: &str = "manifests";
/// Subdirectory holding content-addressed blobs. **Upstream:** `manifest.BlobsPath()`.
pub const BLOBS_DIR: &str = "blobs";

/// A blob younger than this never gets pruned, even if nothing references it.
///
/// **Upstream:** `layerPruneGracePeriod = time.Hour` in `server/images.go:39`.
///
/// The reason is a race, not tidiness: a pull writes blobs *before* it writes
/// the manifest that references them, so for a moment a perfectly good blob got
/// zero referrers. Prune without a grace window and you delete a download still
/// in flight. One hour is upstream's number; don't shrink it unless you
/// understand that window.
pub const LAYER_PRUNE_GRACE_PERIOD: Duration = Duration::from_secs(60 * 60);

/// The digest algorithm prefix. Only sha256 exists in this store.
/// **Upstream:** the regex `^sha256[:-][0-9a-fA-F]{64}$` in `manifest/paths.go` `BlobsPath`.
const DIGEST_ALGO: &str = "sha256";
/// 32 bytes of sha256, hex-encoded. **Upstream:** same regex, the `{64}`.
const DIGEST_HEX_LEN: usize = 64;

/// Lowercase hex alphabet, for [`Digest::from_hash`]. Go gets this from `%x`.
const HEX: &[u8; 16] = b"0123456789abcdef";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong talking to the store.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// **Upstream:** `manifest.ErrInvalidDigestFormat`.
    #[error("invalid digest format: {0:?}")]
    InvalidDigest(String),

    /// **Upstream:** `errors.New("opening layer with empty digest")` /
    /// `"creating new layer from layer with empty digest"` in `manifest/layer.go`.
    ///
    /// A layer with no digest is not addressable -- no blob to open, because no
    /// address.
    #[error("layer has an empty digest")]
    EmptyDigest,

    /// The name is missing a part, or got a malformed one, so it has no store
    /// path. **Upstream:** `model.Unqualified(n)`.
    #[error("unqualified model name: {0}")]
    Unqualified(String),

    /// No manifest file at that name. **Upstream:** the `os.Open` `ErrNotExist`
    /// coming out of `ParseNamedManifest`.
    #[error("model not found: {0}")]
    NotFound(String),

    /// The blob on disk doesn't hash to the digest that names it -- truncated or
    /// corrupted download. **Upstream:** `errDigestMismatch` in `server/images.go`.
    #[error("digest mismatch, file must be downloaded again: want {want}, got {got}")]
    DigestMismatch { want: String, got: String },

    /// **Upstream:** `fmt.Errorf("config %q not found in manifest", configPath)`
    /// in `Manifest.ReadConfigJSON`.
    #[error("config {0:?} not found in manifest")]
    ConfigNotFound(String),

    /// A path under `manifests/` that is not four well-formed name parts.
    /// **Upstream:** `slog.Warn("bad manifest name", ...)` in `Manifests`, which
    /// only becomes an error when `continueOnError` is false.
    #[error("bad manifest path: {0}")]
    BadManifestPath(String),

    /// Any filesystem failure, with the operation attached -- a bare `io::Error`
    /// on its own never tells you *which* file died.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: io::Error,
    },

    /// Malformed JSON in a manifest or a JSON blob.
    #[error("{context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },
}

/// Shorthand so every `?` on a filesystem call can say what it was doing.
fn io_ctx(context: impl Into<String>) -> impl FnOnce(io::Error) -> ManifestError {
    move |source| ManifestError::Io {
        context: context.into(),
        source,
    }
}

/// Same idea for `serde_json`.
fn json_ctx(context: impl Into<String>) -> impl FnOnce(serde_json::Error) -> ManifestError {
    move |source| ManifestError::Json {
        context: context.into(),
        source,
    }
}

/// Result alias for this module.
pub type Result<T, E = ManifestError> = std::result::Result<T, E>;

// ---------------------------------------------------------------------------
// Hashing, injected
// ---------------------------------------------------------------------------

/// A SHA-256 implementation.
///
/// **The crate ships one** -- [`Sha256`], backed by RustCrypto's `sha2` -- so
/// almost every caller should just use that and ignore this trait. It stays a
/// trait for two reasons that are worth the indirection:
///
/// * **Tests can inject a fake.** The store's own tests drive a deliberately
///   labelled non-hash so they can pin the *plumbing* (right address, dedup,
///   mismatch detection, reset semantics) without depending on real digests.
/// * **The arithmetic is separable from the store.** A wrong byte here silently
///   corrupts every blob address in the store, so keeping the hash behind one
///   named seam means there is exactly one thing to audit.
///
/// **Contract, and it is not optional:**
///
/// * [`update`](Sha256Hasher::update) appends to the running message.
/// * [`finalize_and_reset`](Sha256Hasher::finalize_and_reset) returns the digest
///   of everything fed since the *last* reset, **and puts the hasher back to the
///   empty state**, so one hasher can be reused across many blobs. An
///   implementation that forgets to reset will silently chain blob N+1's bytes
///   onto blob N's -- every digest after the first would be wrong, and the store
///   would still happily write them. This is the one thing to get right.
///
pub trait Sha256Hasher {
    /// Feed more bytes into the running hash.
    fn update(&mut self, chunk: &[u8]);
    /// Finish the current message, return its 32 bytes, and reset back to empty.
    fn finalize_and_reset(&mut self) -> [u8; 32];
}

/// The real SHA-256, from RustCrypto's `sha2`. **This is the one you want.**
///
/// `sha2::Digest::finalize_reset` has exactly the reset semantics
/// [`Sha256Hasher`] demands, so the contract comes free rather than being
/// something this wrapper has to remember to honour.
///
/// ```
/// use kopitiam_ollama::manifest::{Sha256, Sha256Hasher};
///
/// let mut h = Sha256::new();
/// h.update(b"abc");
/// // The canonical SHA-256("abc") test vector, from FIPS 180-4 Appendix B.1.
/// assert_eq!(
///     hex(&h.finalize_and_reset()),
///     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
/// );
/// # fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }
/// ```
#[derive(Default, Clone)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the internal state -- it is neither useful nor stable.
        f.write_str("Sha256(..)")
    }
}

impl Sha256Hasher for Sha256 {
    fn update(&mut self, chunk: &[u8]) {
        sha2::Digest::update(&mut self.0, chunk);
    }

    fn finalize_and_reset(&mut self) -> [u8; 32] {
        sha2::Digest::finalize_reset(&mut self.0).into()
    }
}

/// Hash everything a reader gives, returning `(digest, bytes read)`.
///
/// **Upstream:** `GetSHA256Digest(r)` in `server/images.go`, which returns
/// `(fmt.Sprintf("sha256:%x", ...), n)`.
///
/// Streams in 64 KiB chunks on purpose -- a model blob is routinely 20 GB, so
/// anything that reads it whole into memory is not an option lah.
///
/// **Deliberate divergence:** upstream calls `log.Fatal` if the copy fails,
/// which kills the process. We return the `io::Error` -- a library has no
/// business deciding the process should die.
pub fn sha256_of_reader(
    mut r: impl Read,
    hasher: &mut dyn Sha256Hasher,
) -> io::Result<(Digest, u64)> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((Digest::from_hash(hasher.finalize_and_reset()), total))
}

/// Hash a slice already in memory. Convenience over [`sha256_of_reader`].
pub fn sha256_of_bytes(bytes: &[u8], hasher: &mut dyn Sha256Hasher) -> Digest {
    hasher.update(bytes);
    Digest::from_hash(hasher.finalize_and_reset())
}

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

/// A validated content address: `sha256:` followed by 64 lowercase hex digits.
///
/// **Upstream:** the regex `^sha256[:-][0-9a-fA-F]{64}$` guarding
/// `manifest.BlobsPath`, plus the minting side
/// `fmt.Sprintf("sha256:%x", sha256sum.Sum(nil))` in `manifest/layer.go`.
///
/// This is a **capability type**, not decoration. [`Store::blob_path`] takes a
/// `&Digest` and refuses a `&str`, so the only way to name a file under
/// `blobs/` is to have gone through [`Digest::parse`] first. A digest arriving
/// inside a manifest pulled off a registry is untrusted input; this is where it
/// stops being untrusted.
///
/// ```
/// # use kopitiam_ollama::manifest::Digest;
/// let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
/// let d = Digest::parse(&format!("sha256:{hex}")).unwrap();
/// assert_eq!(d.blob_filename(), format!("sha256-{hex}"));
/// // Traversal, wrong length, no algorithm -- all refused.
/// assert!(Digest::parse("sha256:../../../etc/passwd").is_err());
/// assert!(Digest::parse("sha256:abc").is_err());
/// assert!(Digest::parse("../blobs/x").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest {
    /// Canonical form, always `sha256:<64 lowercase hex>`.
    canonical: String,
}

impl Digest {
    /// Parse and validate. Accepts both separators (`sha256:` from a manifest,
    /// `sha256-` from a blob filename), because upstream's regex does -- and
    /// [`Store::prune_blobs`] depends on the dash form parsing.
    ///
    /// **Deliberate divergence from upstream:** ollama's regex allows
    /// `[0-9a-fA-F]`; we insist on **lowercase**. Reason -- on a case-sensitive
    /// filesystem (every Linux, so every Termux) `sha256-ABC…` and `sha256-abc…`
    /// are two different files, so accepting both spellings would let the same
    /// content land in the store twice and quietly break the one property this
    /// whole module exists for. Nothing upstream ever *writes* uppercase
    /// (`fmt.Sprintf("%x")` is lowercase by definition), so in practice this
    /// rejects only malformed input, never a real ollama store.
    pub fn parse(s: &str) -> Result<Self> {
        let bad = || ManifestError::InvalidDigest(s.to_string());
        let rest = s
            .strip_prefix(DIGEST_ALGO)
            .ok_or_else(bad)?
            .strip_prefix([':', '-'])
            .ok_or_else(bad)?;
        if rest.len() != DIGEST_HEX_LEN {
            return Err(bad());
        }
        if !rest
            .bytes()
            .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase() && c <= b'f')
        {
            return Err(bad());
        }
        Ok(Self {
            canonical: format!("{DIGEST_ALGO}:{rest}"),
        })
    }

    /// Mint a digest from 32 raw hash bytes.
    ///
    /// **Upstream:** `fmt.Sprintf("sha256:%x", sha256sum.Sum(nil))` in
    /// `manifest/layer.go` `NewLayer`. `%x` on a byte slice is lowercase hex,
    /// which is why lowercase is the canonical form here.
    pub fn from_hash(hash: [u8; 32]) -> Self {
        let mut canonical = String::with_capacity(DIGEST_ALGO.len() + 1 + DIGEST_HEX_LEN);
        canonical.push_str(DIGEST_ALGO);
        canonical.push(':');
        for b in hash {
            canonical.push(char::from(HEX[(b >> 4) as usize]));
            canonical.push(char::from(HEX[(b & 0x0f) as usize]));
        }
        Self { canonical }
    }

    /// The canonical `sha256:<hex>` spelling -- what goes **in the JSON**.
    pub fn as_str(&self) -> &str {
        &self.canonical
    }

    /// Just the 64 hex digits, no algorithm prefix.
    pub fn hex(&self) -> &str {
        &self.canonical[DIGEST_ALGO.len() + 1..]
    }

    /// The `sha256-<hex>` spelling -- what goes **on the filesystem**.
    ///
    /// **Upstream:** `strings.ReplaceAll(digest, ":", "-")` in
    /// `manifest/paths.go` `BlobsPath`.
    ///
    /// The dash is not cosmetic hor: `:` is illegal in a Windows filename (NTFS
    /// reads it as an alternate-data-stream separator), so a colon here means the
    /// store cannot exist on Windows at all. KOPITIAM ships on Windows. Keep the
    /// dash.
    pub fn blob_filename(&self) -> String {
        format!("{DIGEST_ALGO}-{}", self.hex())
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.canonical)
    }
}

impl std::str::FromStr for Digest {
    type Err = ManifestError;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// One entry in a manifest: a media type plus the address of some bytes.
///
/// **Upstream:** `type Layer struct` in `manifest/layer.go`. Field names and
/// `omitempty` flags match exactly -- a manifest we write must be readable by
/// ollama and vice versa, so the JSON shape is a contract, not a preference.
///
/// [`digest`](Layer::digest) is a plain `String`, **not** a [`Digest`], on
/// purpose: a manifest read off disk may carry an empty or malformed digest
/// (upstream explicitly branches on `layer.Digest == ""`), and silently dropping
/// that during deserialisation would hide corruption instead of reporting it.
/// Call [`Layer::checked_digest`] when you want the validated form.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    /// e.g. [`MEDIA_TYPE_MODEL`]. **Upstream:** `MediaType`.
    #[serde(rename = "mediaType", default)]
    pub media_type: String,

    /// `sha256:<hex>`, raw as it appeared in the JSON. **Upstream:** `Digest`.
    #[serde(default)]
    pub digest: String,

    /// Blob length in bytes. `i64` to match Go's `int64` exactly -- a negative
    /// size is nonsense, but a corrupt manifest can still contain one, and
    /// clamping it into a `u64` at parse time would turn a detectable corruption
    /// into a huge number instead.
    /// **Upstream:** `Size int64`.
    #[serde(default)]
    pub size: i64,

    /// The parent model this layer came from, for a `FROM`-derived model.
    /// **Upstream:** `From string \`json:"from,omitempty"\``.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub from: String,

    /// Tensor name, e.g. `text_encoder/model.embed_tokens.weight`; also the
    /// lookup key for [`Store::read_layer_json`].
    /// **Upstream:** `Name string \`json:"name,omitempty"\``.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,

    /// Human-facing progress line ("creating new layer sha256:…"). Never
    /// serialised. **Upstream:** `Status string \`json:"-"\``.
    #[serde(skip)]
    pub status: String,
}

impl Layer {
    /// Build a layer for bytes already known to be in the store.
    pub fn new(media_type: impl Into<String>, digest: &Digest, size: i64) -> Self {
        Self {
            media_type: media_type.into(),
            digest: digest.as_str().to_string(),
            size,
            ..Default::default()
        }
    }

    /// The validated digest, or why it isn't one.
    ///
    /// **Upstream:** the `if l.Digest == ""` guard at the top of `Layer.Open`
    /// and `NewLayerFromLayer`, followed by `BlobsPath`'s regex check.
    pub fn checked_digest(&self) -> Result<Digest> {
        if self.digest.is_empty() {
            return Err(ManifestError::EmptyDigest);
        }
        Digest::parse(&self.digest)
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// Go's `json.Marshal` writes a nil slice as `null`, not `[]`, so any manifest
/// ollama wrote with no layers reads back as `"layers": null`. Plain
/// `#[serde(default)]` does **not** cover that -- default only fires when the key
/// is *absent*, and an explicit `null` would blow up with "invalid type: null,
/// expected a sequence". This adaptor makes `null` mean "empty", which is what
/// Go means by it.
fn null_as_default<'de, D, T>(d: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

/// A model's manifest: schema header, one config layer, N content layers.
///
/// **Upstream:** `type Manifest struct` in `manifest/manifest.go`.
///
/// The four public fields are the JSON, in Go's declaration order (serde
/// serialises in declaration order too, so the byte layout matches). The private
/// fields mirror upstream's unexported `filepath` / `fi` / `digest`, and only get
/// populated when the manifest came off disk via [`Store::read_manifest`] -- one
/// you built yourself has them empty, same as Go.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Always [`SCHEMA_VERSION`] for anything we write.
    #[serde(rename = "schemaVersion", default)]
    pub schema_version: i32,

    /// Always [`MEDIA_TYPE_MANIFEST`] for anything we write.
    #[serde(rename = "mediaType", default)]
    pub media_type: String,

    /// Points at the [`ConfigV2`] blob. Read it with [`Store::read_config`].
    #[serde(default)]
    pub config: Layer,

    /// Weights, adapters, template, system prompt, licences -- everything else.
    #[serde(default, deserialize_with = "null_as_default")]
    pub layers: Vec<Layer>,

    #[serde(skip)]
    path: PathBuf,
    #[serde(skip)]
    file_size: u64,
    #[serde(skip)]
    modified: Option<SystemTime>,
    #[serde(skip)]
    digest: String,
}

impl Manifest {
    /// Total bytes of every layer **plus** the config layer.
    ///
    /// **Upstream:** `(*Manifest).Size()`, which sums `append(m.Layers, m.Config)`.
    ///
    /// Careful what this number means: it is the size of this model *as
    /// declared*, and it **double counts anything shared with another tag**. For
    /// "how much disk am I actually using", ask [`Store::disk_usage`]. The gap
    /// between the two is the dedup saving, and
    /// `two_tags_sharing_a_layer_store_the_bytes_once` asserts it.
    pub fn size(&self) -> i64 {
        self.layers_and_config().map(|l| l.size).sum()
    }

    /// [`Manifest::size`] rendered for humans, via `crate::format::human_bytes`.
    pub fn human_size(&self) -> String {
        human_bytes(self.size())
    }

    /// Every layer, config last -- upstream's `append(m.Layers, m.Config)` idiom,
    /// which shows up in `Size`, `RemoveLayers`, `Layer.Remove` and
    /// `deleteUnusedLayers`. Same order, no allocation.
    pub fn layers_and_config(&self) -> impl Iterator<Item = &Layer> {
        self.layers.iter().chain(std::iter::once(&self.config))
    }

    /// The sha256 of the manifest **file** (hex, no `sha256:` prefix), or `""`
    /// if it was never computed.
    ///
    /// **Upstream:** `(*Manifest).Digest()`, filled in by `ParseNamedManifest`.
    /// This is the value `GetModel` reports as a model's `Digest` and what
    /// `ollama list` shows -- it identifies the *manifest*, not any one layer.
    ///
    /// Only [`Store::read_manifest_with`] fills it; [`Store::read_manifest`]
    /// leaves it empty because it got no hasher. Check for empty, don't assume.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Where this manifest was read from. Empty for a manifest you built.
    /// **Upstream:** the unexported `filepath` field.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// [`Manifest::path`] with forward slashes, safe to print on any platform.
    pub fn path_slash(&self) -> String {
        to_slash(&self.path)
    }

    /// Size of the manifest **file** itself (not the model).
    /// **Upstream:** `(*Manifest).FileInfo().Size()`.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Manifest file mtime -- what `ollama list` prints as "modified".
    /// **Upstream:** `(*Manifest).FileInfo().ModTime()`.
    pub fn modified(&self) -> Option<SystemTime> {
        self.modified
    }

    /// First layer with this media type, if any.
    ///
    /// **Upstream:** the `switch layer.MediaType` in `server/images.go`
    /// `GetModel`, expressed as a lookup instead of a loop.
    pub fn layer(&self, media_type: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.media_type == media_type)
    }

    /// Every layer with this media type -- for the ones that legitimately repeat
    /// ([`MEDIA_TYPE_LICENSE`], [`MEDIA_TYPE_ADAPTER`], [`MEDIA_TYPE_TENSOR`]).
    pub fn layers_of<'a>(&'a self, media_type: &'a str) -> impl Iterator<Item = &'a Layer> {
        self.layers
            .iter()
            .filter(move |l| l.media_type == media_type)
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Render a path with forward slashes, whatever platform built it.
///
/// **KOPITIAM rule, not upstream:** CLAUDE.md requires CLI output to use `/` on
/// every platform, so a log line or a bead written on Windows diffs cleanly
/// against one written on Termux. Upstream just prints native separators.
///
/// For **display only**. Don't feed the result back into `fs::` on Windows and
/// assume it means the same thing as the original -- it usually does, but
/// "usually" is not a contract. Note also that on Unix a backslash is a legal
/// filename character; inside this store it cannot occur (name parts are
/// validated by `crate::name`, blob names are hex), so the substitution is safe
/// *here* and this is not a general-purpose path converter.
pub fn to_slash(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// Remove empty directories, depth first.
///
/// **Upstream:** `manifest.PruneDirectory` in `manifest/paths.go`.
///
/// Deleting `library/qwen3/latest` leaves `library/qwen3/` behind, and a store
/// full of empty skeletons makes [`Store::manifests`] walk rubbish forever. Note
/// upstream removes `path` itself too if it ends up empty -- including the
/// `manifests/` root. That is intended; the root gets recreated on next write.
///
/// Symlinked directories are left alone (upstream checks `ModeSymlink`), so a
/// store where somebody symlinked `blobs/` onto a bigger disk doesn't get its
/// target chewed.
pub fn prune_directory(path: &Path) -> Result<()> {
    let md = fs::symlink_metadata(path).map_err(io_ctx(format!("stat {}", to_slash(path))))?;

    if !md.is_dir() || md.file_type().is_symlink() {
        return Ok(());
    }

    for entry in fs::read_dir(path).map_err(io_ctx(format!("read dir {}", to_slash(path))))? {
        let entry = entry.map_err(io_ctx(format!("read dir {}", to_slash(path))))?;
        prune_directory(&entry.path())?;
    }

    // Re-read: children may have just been removed above.
    let still_has = fs::read_dir(path)
        .map_err(io_ctx(format!("read dir {}", to_slash(path))))?
        .next()
        .is_some();
    if still_has {
        return Ok(());
    }

    fs::remove_dir(path).map_err(io_ctx(format!("remove dir {}", to_slash(path))))
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// What happened when bytes were handed to the store.
///
/// **Upstream:** the `status` string in `manifest.NewLayer` -- `"using existing
/// layer"` vs `"creating new layer"`. Upstream builds a sentence; we return the
/// fact and let the caller phrase it, because a TUI and a log line want
/// different words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobOutcome {
    /// Bytes were written. **Upstream:** `"creating new layer"`.
    Created,
    /// The blob was already there, so nothing was written -- this is dedup
    /// actually happening. **Upstream:** `"using existing layer"`.
    Reused,
}

impl BlobOutcome {
    /// Upstream's exact wording, for progress output that must match ollama's.
    /// **Upstream:** `manifest/layer.go` `NewLayer`.
    pub fn status(&self) -> &'static str {
        match self {
            BlobOutcome::Created => "creating new layer",
            BlobOutcome::Reused => "using existing layer",
        }
    }
}

/// A model store rooted at one directory.
///
/// **Upstream:** the free functions in `manifest/paths.go` (`Path`,
/// `PathForName`, `BlobsPath`), which read the root from the
/// `envconfig.Models()` global.
///
/// **Deliberate divergence: the root is a field, not a global.** Upstream reads
/// `OLLAMA_MODELS` (or `~/.ollama/models`) from the process environment at every
/// call, which makes the whole package untestable without `t.Setenv` and makes
/// two stores in one process impossible. Passing the root in costs one argument
/// and buys: parallel tests with no shared global, and a KOPITIAM store that can
/// sit somewhere other than ollama's.
///
/// Resolving the *default* root -- `$OLLAMA_MODELS` else `~/.ollama/models`, per
/// `envconfig.Models()` in `envconfig/config.go:113` -- belongs to
/// `crate::envconfig`. Once that lands, callers do
/// `Store::new(envconfig::models())`. This module deliberately reads no
/// environment at all.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// A store rooted at `root` (the equivalent of `envconfig.Models()`).
    /// Creates nothing; call [`Store::ensure_layout`], or just write something.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `<root>/manifests`. **Upstream:** `manifest.Path()`, minus the mkdir.
    pub fn manifests_dir(&self) -> PathBuf {
        self.root.join(MANIFESTS_DIR)
    }

    /// `<root>/blobs`. **Upstream:** `manifest.BlobsPath("")`, minus the mkdir.
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join(BLOBS_DIR)
    }

    /// Create `manifests/` and `blobs/` if missing.
    ///
    /// **Upstream:** `Path()` and `BlobsPath("")` each `os.MkdirAll(..., 0o755)`
    /// as a side effect of being *asked for a path*. We split that out, because
    /// a function called `manifests_dir()` creating a directory is a nasty
    /// surprise. Every write path here calls this first, so the observable
    /// behaviour matches; only the read-only accessors are now honestly
    /// read-only.
    pub fn ensure_layout(&self) -> Result<()> {
        for d in [self.manifests_dir(), self.blobs_dir()] {
            fs::create_dir_all(&d).map_err(io_ctx(format!(
                "create {} -- ensure path elements are traversable",
                to_slash(&d)
            )))?;
        }
        Ok(())
    }

    /// Where a model's manifest file lives.
    ///
    /// **Upstream:** `manifest.PathForName(n)`, which returns `os.ErrNotExist`
    /// for an invalid name.
    ///
    /// **This is the traversal gate.** [`Name::filepath`] returns `None` unless
    /// all four parts are well-formed -- no `/`, no `\`, no leading `.` -- so the
    /// joined path provably stays under `manifests/`. Don't "helpfully" add a
    /// `&str` overload.
    pub fn manifest_path(&self, name: &Name) -> Result<PathBuf> {
        let rel = name
            .filepath()
            .ok_or_else(|| ManifestError::Unqualified(name.to_string()))?;
        Ok(self.manifests_dir().join(rel))
    }

    /// Where a blob lives: `<root>/blobs/sha256-<hex>`.
    ///
    /// **Upstream:** `manifest.BlobsPath(digest)`.
    ///
    /// Infallible on purpose: taking a [`Digest`] means validation already
    /// happened, so there is no error left to return. That's the whole reason
    /// [`Digest`] exists.
    pub fn blob_path(&self, digest: &Digest) -> PathBuf {
        self.blobs_dir().join(digest.blob_filename())
    }

    // -- reading -----------------------------------------------------------

    /// Read a manifest. Leaves [`Manifest::digest`] empty (no hasher given).
    ///
    /// **Upstream:** `manifest.ParseNamedManifest(n)`, minus the sha256.
    pub fn read_manifest(&self, name: &Name) -> Result<Manifest> {
        self.read_manifest_inner(name, None)
    }

    /// Read a manifest **and** compute its file digest.
    ///
    /// **Upstream:** `manifest.ParseNamedManifest(n)` in full.
    ///
    /// Divergence worth knowing: upstream hashes through
    /// `io.TeeReader(f, sha256sum)` wrapped in a `json.Decoder`, so it digests
    /// *whatever the decoder happened to buffer* -- which for any real manifest
    /// is the whole file, trailing newline included, but that is an
    /// implementation detail rather than a promise. We read the file and hash all
    /// of it, which gives the same answer for every manifest either program
    /// writes, and is at least defined for the ones it wouldn't.
    pub fn read_manifest_with(
        &self,
        name: &Name,
        hasher: &mut dyn Sha256Hasher,
    ) -> Result<Manifest> {
        self.read_manifest_inner(name, Some(hasher))
    }

    fn read_manifest_inner(
        &self,
        name: &Name,
        hasher: Option<&mut dyn Sha256Hasher>,
    ) -> Result<Manifest> {
        if !name.is_fully_qualified() {
            return Err(ManifestError::Unqualified(name.to_string()));
        }
        let p = self.manifest_path(name)?;

        let bytes = match fs::read(&p) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(ManifestError::NotFound(name.to_string()))
            }
            Err(e) => return Err(io_ctx(format!("read {}", to_slash(&p)))(e)),
        };
        let md = fs::metadata(&p).map_err(io_ctx(format!("stat {}", to_slash(&p))))?;

        let mut m: Manifest = serde_json::from_slice(&bytes)
            .map_err(json_ctx(format!("parse manifest {}", to_slash(&p))))?;

        m.path = p;
        m.file_size = md.len();
        m.modified = md.modified().ok();
        if let Some(h) = hasher {
            h.update(&bytes);
            // Hex only, no `sha256:` prefix -- upstream's
            // `hex.EncodeToString(sha256sum.Sum(nil))`.
            m.digest = Digest::from_hash(h.finalize_and_reset()).hex().to_string();
        }
        Ok(m)
    }

    /// Got a manifest for this name or not?
    pub fn exists(&self, name: &Name) -> bool {
        self.manifest_path(name)
            .map(|p| p.is_file())
            .unwrap_or(false)
    }

    /// Every model in the store, sorted by store path.
    ///
    /// **Upstream:** `manifest.Manifests(continueOnError)`.
    ///
    /// Walks exactly four levels -- `manifests/*/*/*/*` -- and takes only the
    /// **files** at depth four. A directory there (somebody's `tag/one/` mess) is
    /// skipped, matching upstream's `if !fi.IsDir()`. A leaf whose four parts
    /// don't form a valid [`Name`] (a `.hidden` file, say) is skipped too.
    ///
    /// `continue_on_error` decides what happens to a leaf that *looks* like a
    /// model but won't parse: `true` skips it (upstream logs a warning), `false`
    /// returns the error. Deleting a half-written model needs `true`, otherwise
    /// one corrupt manifest makes the whole store un-enumerable and un-fixable --
    /// that's why upstream's own delete path passes `true`.
    ///
    /// **Divergences, both deliberate:**
    /// * Returns a `Vec<(Name, Manifest)>` rather than a map. `crate::name::Name`
    ///   implements no `Hash` (another module's type, not ours to change), and a
    ///   sorted `Vec` is the better answer anyway -- enumeration order becomes
    ///   reproducible instead of hash-random, so `list` output is stable across
    ///   runs and across platforms.
    /// * No sha256 unless you use [`Store::manifests_with`], since we hold no
    ///   hasher.
    pub fn manifests(&self, continue_on_error: bool) -> Result<Vec<(Name, Manifest)>> {
        self.manifests_inner(continue_on_error, None)
    }

    /// [`Store::manifests`], but fills in each [`Manifest::digest`].
    /// **Upstream:** what `Manifests` does natively, since it calls
    /// `ParseNamedManifest`.
    pub fn manifests_with(
        &self,
        continue_on_error: bool,
        hasher: &mut dyn Sha256Hasher,
    ) -> Result<Vec<(Name, Manifest)>> {
        self.manifests_inner(continue_on_error, Some(hasher))
    }

    fn manifests_inner(
        &self,
        continue_on_error: bool,
        mut hasher: Option<&mut dyn Sha256Hasher>,
    ) -> Result<Vec<(Name, Manifest)>> {
        // Upstream's `Path()` mkdirs before globbing, so an empty store
        // enumerates as empty instead of erroring. Same here.
        self.ensure_layout()?;
        let root = self.manifests_dir();

        let mut out = Vec::new();
        // Four levels: host / namespace / model / tag. Sorted at every level so
        // the result is deterministic -- Go's filepath.Glob sorts too.
        for host in sorted_children(&root)? {
            for ns in sorted_children(&host)? {
                for model in sorted_children(&ns)? {
                    for tag in sorted_children(&model)? {
                        // Depth four must be a FILE. `manifests/h/n/m/t/one` is
                        // upstream's "subdir" test case, and is not a model.
                        if tag.is_dir() {
                            continue;
                        }
                        let rel = tag.strip_prefix(&root).unwrap_or(&tag);
                        let Some(name) = Name::parse_from_filepath(rel) else {
                            if continue_on_error {
                                continue;
                            }
                            return Err(ManifestError::BadManifestPath(to_slash(rel)));
                        };
                        let read = match hasher.as_deref_mut() {
                            Some(h) => self.read_manifest_with(&name, h),
                            None => self.read_manifest(&name),
                        };
                        match read {
                            Ok(m) => out.push((name, m)),
                            Err(e) if continue_on_error => {
                                let _ = e;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Read the config blob as a [`ConfigV2`].
    ///
    /// **Upstream:** the `if mf.Config.Digest != ""` block in `server/images.go`
    /// `GetModel`, which opens `BlobsPath(mf.Config.Digest)` and JSON-decodes it.
    ///
    /// A manifest with an empty config digest gets [`ConfigV2::default`], same as
    /// upstream (which just skips the block, leaving `m.Config` zero).
    pub fn read_config(&self, manifest: &Manifest) -> Result<ConfigV2> {
        if manifest.config.digest.is_empty() {
            return Ok(ConfigV2::default());
        }
        let d = manifest.config.checked_digest()?;
        let bytes = self.read_blob(&d)?;
        serde_json::from_slice(&bytes).map_err(json_ctx(format!("parse config blob {d}")))
    }

    /// Read a named JSON side-car layer.
    ///
    /// **Upstream:** `(*Manifest).ReadConfigJSON(configPath, v)` in
    /// `manifest/manifest.go` -- scans `m.Layers` for a layer whose media type is
    /// [`MEDIA_TYPE_JSON`] **and** whose [`Layer::name`] equals `config_path`,
    /// then unmarshals its blob.
    ///
    /// Note it searches `layers` only, never `config` -- upstream does the same,
    /// so a `config_path` naming the config layer will not be found.
    pub fn read_layer_json<T: DeserializeOwned>(
        &self,
        manifest: &Manifest,
        config_path: &str,
    ) -> Result<T> {
        let layer = manifest
            .layers
            .iter()
            .find(|l| l.media_type == MEDIA_TYPE_JSON && l.name == config_path)
            .ok_or_else(|| ManifestError::ConfigNotFound(config_path.to_string()))?;
        let d = layer.checked_digest()?;
        let bytes = self.read_blob(&d)?;
        serde_json::from_slice(&bytes).map_err(json_ctx(format!("parse json layer {config_path}")))
    }

    /// Open a blob for reading. **Upstream:** `(*Layer).Open()`.
    pub fn open_blob(&self, digest: &Digest) -> Result<File> {
        let p = self.blob_path(digest);
        File::open(&p).map_err(io_ctx(format!("open blob {}", to_slash(&p))))
    }

    /// Slurp a blob whole. Fine for configs, templates and params; **don't** use
    /// it on a [`MEDIA_TYPE_MODEL`] layer -- those run to tens of gigabytes, use
    /// [`Store::open_blob`] and stream.
    pub fn read_blob(&self, digest: &Digest) -> Result<Vec<u8>> {
        let p = self.blob_path(digest);
        fs::read(&p).map_err(io_ctx(format!("read blob {}", to_slash(&p))))
    }

    /// Blob on disk already or not? The dedup question, asked directly.
    pub fn has_blob(&self, digest: &Digest) -> bool {
        self.blob_path(digest).is_file()
    }

    /// Blob length in bytes. **Upstream:** the `os.Stat(blob)` in
    /// `NewLayerFromLayer`.
    pub fn blob_size(&self, digest: &Digest) -> Result<u64> {
        let p = self.blob_path(digest);
        Ok(fs::metadata(&p)
            .map_err(io_ctx(format!("stat blob {}", to_slash(&p))))?
            .len())
    }

    /// **Actual** bytes on disk under `blobs/` -- every blob counted once, no
    /// matter how many manifests point at it.
    ///
    /// KOPITIAM addition, no upstream equivalent. It exists so the dedup claim in
    /// this module's header is a measurement rather than a story: compare it
    /// against [`Store::declared_size`] and the difference is what
    /// content-addressing saved you.
    pub fn disk_usage(&self) -> Result<u64> {
        let dir = self.blobs_dir();
        if !dir.is_dir() {
            return Ok(0);
        }
        let mut total = 0u64;
        for entry in fs::read_dir(&dir).map_err(io_ctx(format!("read dir {}", to_slash(&dir))))? {
            let entry = entry.map_err(io_ctx(format!("read dir {}", to_slash(&dir))))?;
            let md = entry
                .metadata()
                .map_err(io_ctx(format!("stat {}", to_slash(&entry.path()))))?;
            if md.is_file() {
                total += md.len();
            }
        }
        Ok(total)
    }

    /// Sum of every manifest's declared size -- a shared layer counted **once per
    /// model**. Pair it with [`Store::disk_usage`] to show the saving.
    ///
    /// KOPITIAM addition. Corrupt manifests are skipped (`continue_on_error`),
    /// because a size report must never be the thing that fails.
    pub fn declared_size(&self) -> Result<i64> {
        Ok(self.manifests(true)?.iter().map(|(_, m)| m.size()).sum())
    }

    // -- writing -----------------------------------------------------------

    /// Write a manifest for `name`.
    ///
    /// **Upstream:** `manifest.WriteManifest(name, config, layers)`.
    ///
    /// Byte-format notes, and they matter because [`Manifest::digest`] is taken
    /// over these exact bytes:
    /// * **Compact JSON, then a single `\n`.** Go's `json.Encoder.Encode` always
    ///   appends that newline; drop it and every manifest digest changes.
    /// * Field order is Go's struct order -- `schemaVersion`, `mediaType`,
    ///   `config`, `layers` -- which serde gives us from declaration order.
    /// * Known cosmetic divergence: Go's encoder HTML-escapes `<`, `>` and `&` by
    ///   default, serde_json does not. No field in a real manifest (media types,
    ///   hex digests, model names, tensor names) can contain those, so the bytes
    ///   agree in practice -- but if a future field ever can, this is the line
    ///   that breaks digest parity.
    ///
    /// Returns the path written, so a caller can log it.
    pub fn write_manifest(
        &self,
        name: &Name,
        config: Layer,
        layers: Vec<Layer>,
    ) -> Result<PathBuf> {
        self.ensure_layout()?;
        let p = self.manifest_path(name)?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(io_ctx(format!("create {}", to_slash(parent))))?;
        }

        let m = Manifest {
            schema_version: SCHEMA_VERSION,
            media_type: MEDIA_TYPE_MANIFEST.to_string(),
            config,
            layers,
            ..Default::default()
        };

        let mut bytes =
            serde_json::to_vec(&m).map_err(json_ctx(format!("encode manifest {name}")))?;
        bytes.push(b'\n');
        fs::write(&p, &bytes).map_err(io_ctx(format!("write {}", to_slash(&p))))?;
        Ok(p)
    }

    /// Copy one model's manifest to another name. The blobs are **not** touched
    /// -- that's the whole trick, the copy is a few hundred bytes.
    ///
    /// **Upstream:** `server.CopyModel(src, dst)`.
    ///
    /// Both names must be fully qualified; copying a name onto itself is a no-op
    /// (upstream returns nil when the two filepaths match).
    pub fn copy_manifest(&self, src: &Name, dst: &Name) -> Result<()> {
        if !dst.is_fully_qualified() {
            return Err(ManifestError::Unqualified(dst.to_string()));
        }
        if !src.is_fully_qualified() {
            return Err(ManifestError::Unqualified(src.to_string()));
        }
        let srcpath = self.manifest_path(src)?;
        let dstpath = self.manifest_path(dst)?;
        if srcpath == dstpath {
            return Ok(());
        }
        if let Some(parent) = dstpath.parent() {
            fs::create_dir_all(parent).map_err(io_ctx(format!("create {}", to_slash(parent))))?;
        }
        match fs::copy(&srcpath, &dstpath) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Err(ManifestError::NotFound(src.to_string()))
            }
            Err(e) => Err(io_ctx(format!(
                "copy {} -> {}",
                to_slash(&srcpath),
                to_slash(&dstpath)
            ))(e)),
        }
    }

    /// Put bytes into the store at an address the caller already computed.
    ///
    /// **Upstream:** the second half of `manifest.NewLayer` -- stat the blob,
    /// rename the temp file into place only if it is missing.
    ///
    /// **The digest is trusted to describe `bytes`.** Get it from
    /// [`sha256_of_bytes`], or this quietly stores content at the wrong address
    /// and every later [`Store::verify_blob`] fails. If you got a reader rather
    /// than a slice, use [`Store::new_layer_from_reader`] -- it cannot make that
    /// mistake, because it hashes as it writes.
    ///
    /// Write is atomic-ish: temp file in the same directory, then rename. Same as
    /// upstream, and it is what stops a killed process from leaving a truncated
    /// blob sitting at a valid address.
    pub fn put_blob(&self, digest: &Digest, bytes: &[u8]) -> Result<BlobOutcome> {
        self.ensure_layout()?;
        let blob = self.blob_path(digest);
        if blob.exists() {
            self.touch_blob(digest)?;
            return Ok(BlobOutcome::Reused);
        }
        let temp = self.temp_blob_path();
        fs::write(&temp, bytes).map_err(io_ctx(format!("write {}", to_slash(&temp))))?;
        self.finish_blob(&temp, &blob)?;
        Ok(BlobOutcome::Created)
    }

    /// Stream a reader into the store, hashing as it goes, and hand back the
    /// [`Layer`] describing it.
    ///
    /// **Upstream:** `manifest.NewLayer(r, mediatype)` in `manifest/layer.go`,
    /// including its `io.Copy(io.MultiWriter(temp, sha256sum), r)` -- write and
    /// hash in one pass, because a 20 GB model must not be read twice.
    ///
    /// [`Layer::status`] gets upstream's exact sentence
    /// (`"creating new layer sha256:…"` / `"using existing layer sha256:…"`) so
    /// progress output can match ollama's.
    pub fn new_layer_from_reader(
        &self,
        mut r: impl Read,
        media_type: &str,
        hasher: &mut dyn Sha256Hasher,
    ) -> Result<Layer> {
        self.ensure_layout()?;
        let temp = self.temp_blob_path();
        let mut f = File::create(&temp).map_err(io_ctx(format!("create {}", to_slash(&temp))))?;

        let mut buf = vec![0u8; 64 * 1024];
        let mut size = 0i64;
        let copied = (|| -> io::Result<()> {
            loop {
                let n = r.read(&mut buf)?;
                if n == 0 {
                    return Ok(());
                }
                f.write_all(&buf[..n])?;
                hasher.update(&buf[..n]);
                size += n as i64;
            }
        })()
        .and_then(|()| f.sync_all());

        // Whatever happened, don't leave a stray temp file behind -- upstream
        // gets this from `defer os.Remove(temp.Name())`.
        if let Err(e) = copied {
            drop(f);
            let _ = fs::remove_file(&temp);
            // The hasher is mid-message; wind it back so the next caller isn't
            // handed our half-written bytes chained onto theirs.
            let _ = hasher.finalize_and_reset();
            return Err(io_ctx(format!("write {}", to_slash(&temp)))(e));
        }
        drop(f);

        let digest = Digest::from_hash(hasher.finalize_and_reset());
        let blob = self.blob_path(&digest);

        let outcome = if blob.exists() {
            // Already got it -- dedup. Bin the temp copy.
            let _ = fs::remove_file(&temp);
            self.touch_blob(&digest)?;
            BlobOutcome::Reused
        } else {
            self.finish_blob(&temp, &blob)?;
            BlobOutcome::Created
        };

        Ok(Layer {
            media_type: media_type.to_string(),
            digest: digest.as_str().to_string(),
            size,
            status: format!("{} {digest}", outcome.status()),
            ..Default::default()
        })
    }

    /// [`Store::new_layer_from_reader`] for bytes already in memory.
    pub fn new_layer(
        &self,
        bytes: &[u8],
        media_type: &str,
        hasher: &mut dyn Sha256Hasher,
    ) -> Result<Layer> {
        self.new_layer_from_reader(bytes, media_type, hasher)
    }

    /// Build a layer pointing at a blob that must **already** be in the store.
    ///
    /// **Upstream:** `manifest.NewLayerFromLayer(digest, mediatype, from)`. This
    /// is how `FROM another-model` reuses the parent's weights -- no bytes move,
    /// only a second manifest entry appears. Errors if the blob is missing,
    /// deliberately: a manifest referencing a blob that isn't there is a broken
    /// model, and the failure belongs here, not at load time.
    pub fn new_layer_from_layer(
        &self,
        digest: &Digest,
        media_type: &str,
        from: &str,
    ) -> Result<Layer> {
        let size = self.blob_size(digest)? as i64;
        self.touch_blob(digest)?;
        Ok(Layer {
            media_type: media_type.to_string(),
            digest: digest.as_str().to_string(),
            size,
            from: from.to_string(),
            status: format!("{} {digest}", BlobOutcome::Reused.status()),
            ..Default::default()
        })
    }

    /// Re-hash a blob and check it still matches its own address.
    ///
    /// **Upstream:** `server.verifyBlob(digest)`, which returns
    /// `errDigestMismatch` -- "file must be downloaded again".
    ///
    /// This is the one check that catches a truncated download, a half-written
    /// file from a killed process, or bit rot. Costs a full read of the blob, so
    /// it is a repair tool, not something to run on every load.
    pub fn verify_blob(&self, digest: &Digest, hasher: &mut dyn Sha256Hasher) -> Result<()> {
        let p = self.blob_path(digest);
        let f = File::open(&p).map_err(io_ctx(format!("open blob {}", to_slash(&p))))?;
        let (got, _) =
            sha256_of_reader(f, hasher).map_err(io_ctx(format!("hash blob {}", to_slash(&p))))?;
        if &got != digest {
            return Err(ManifestError::DigestMismatch {
                want: digest.to_string(),
                got: got.to_string(),
            });
        }
        Ok(())
    }

    /// Bump a blob's mtime to now, so [`Store::prune_blobs`] leaves it alone for
    /// another grace period.
    ///
    /// **Upstream:** `touchLayer` in `manifest/layer.go`
    /// (`os.Chtimes(path, now, now)`), called from both `NewLayer` and
    /// `NewLayerFromLayer`.
    ///
    /// A missing blob is not an error here -- upstream would fail, but there is
    /// nothing to protect from a pruner that already can't see it, and turning a
    /// bookkeeping nicety into a hard failure helps nobody.
    pub fn touch_blob(&self, digest: &Digest) -> Result<()> {
        let p = self.blob_path(digest);
        let f = match File::options().write(true).open(&p) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(io_ctx(format!("open blob {}", to_slash(&p)))(e)),
        };
        let now = SystemTime::now();
        f.set_times(fs::FileTimes::new().set_accessed(now).set_modified(now))
            .map_err(io_ctx(format!("touch blob {}", to_slash(&p))))
    }

    // -- deleting ----------------------------------------------------------

    /// Delete a model's manifest file, then tidy up the empty directories it
    /// leaves behind. **Blobs untouched** -- another tag may still want them.
    ///
    /// **Upstream:** `(*Manifest).Remove()`, which is `os.Remove(m.filepath)`
    /// followed by `PruneDirectory(manifests)`.
    ///
    /// Call [`Store::remove_unused_layers`] **before** this if you also want the
    /// bytes gone; ordering matters, since that function decides "unused" by
    /// looking at which manifests still exist.
    pub fn remove_manifest(&self, name: &Name) -> Result<()> {
        let p = self.manifest_path(name)?;
        match fs::remove_file(&p) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(ManifestError::NotFound(name.to_string()))
            }
            Err(e) => return Err(io_ctx(format!("remove {}", to_slash(&p)))(e)),
        }
        let manifests = self.manifests_dir();
        if manifests.exists() {
            prune_directory(&manifests)?;
        }
        Ok(())
    }

    /// Delete the blobs of `manifest` that **no other manifest** references.
    ///
    /// **Upstream:** `(*Manifest).RemoveLayers()` in `manifest/manifest.go`.
    ///
    /// This is the content-addressed store's whole deletion rule, in one place:
    /// build the set of digests every *other* manifest still uses, then remove
    /// only the ones outside it. Delete `qwen3:0.6b` while `my-qwen:latest` still
    /// points at the same weights and nothing goes -- delete the second one and
    /// the bytes finally go.
    ///
    /// Enumerates with `continue_on_error = true`, and upstream's comment says
    /// exactly why: *"Ignore corrupt manifests to avoid blocking deletion of
    /// layers that are freshly orphaned."* One unreadable manifest must not make
    /// the store impossible to clean.
    ///
    /// Returns the digests actually removed. A missing blob is not an error
    /// (upstream logs `"layer does not exist"` at debug and carries on).
    pub fn remove_unused_layers(&self, manifest: &Manifest) -> Result<Vec<Digest>> {
        let mut in_use: BTreeSet<String> = BTreeSet::new();
        for (_, other) in self.manifests(true)? {
            // Don't count the manifest being deleted as a referrer of itself.
            // Upstream gets away without this because `Remove()` deletes the
            // file first; we support calling this either way round, so compare
            // paths. An in-memory manifest has an empty path and can never match.
            if !manifest.path.as_os_str().is_empty() && other.path == manifest.path {
                continue;
            }
            for layer in other.layers_and_config() {
                if !layer.digest.is_empty() {
                    in_use.insert(layer.digest.clone());
                }
            }
        }

        let mut removed = Vec::new();
        for layer in manifest.layers_and_config() {
            if layer.digest.is_empty() || in_use.contains(&layer.digest) {
                continue;
            }
            let d = layer.checked_digest()?;
            let p = self.blob_path(&d);
            match fs::remove_file(&p) {
                Ok(()) => removed.push(d),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_ctx(format!("remove blob {}", to_slash(&p)))(e)),
            }
        }
        Ok(removed)
    }

    /// Delete one layer's blob, but only if nothing in the store references it.
    ///
    /// **Upstream:** `(*Layer).Remove()`. Returns `true` if the blob went,
    /// `false` if it was kept because something still points at it -- upstream
    /// returns a bare `nil` for both, which loses the one fact the caller wanted.
    pub fn remove_layer(&self, layer: &Layer) -> Result<bool> {
        if layer.digest.is_empty() {
            return Ok(false);
        }
        for (_, m) in self.manifests(true)? {
            if m.layers_and_config().any(|l| l.digest == layer.digest) {
                return Ok(false);
            }
        }
        let d = layer.checked_digest()?;
        let p = self.blob_path(&d);
        match fs::remove_file(&p) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io_ctx(format!("remove blob {}", to_slash(&p)))(e)),
        }
    }

    /// Sweep `blobs/` for anything unreferenced or malformed.
    ///
    /// **Upstream:** `server.PruneLayers()` plus `server.deleteUnusedLayers()` in
    /// `server/images.go`, fused -- upstream splits them only because the second
    /// is also called from the delete route.
    ///
    /// Three rules, in order:
    /// 1. A blob younger than `grace` is left alone, full stop. See
    ///    [`LAYER_PRUNE_GRACE_PERIOD`] -- a pull writes blobs before it writes
    ///    the manifest, so a brand-new blob legitimately has no referrer yet.
    /// 2. A filename that is not a valid `sha256-<64 hex>` is a partial download
    ///    and gets deleted outright (upstream: *"remove invalid blobs (e.g.
    ///    partial downloads)"*).
    /// 3. Everything else goes only if no manifest references it.
    ///
    /// Returns the blob filenames removed. A file whose mtime is somehow in the
    /// future is treated as brand new and kept -- clock skew must never be the
    /// reason your weights get deleted.
    pub fn prune_blobs(&self, grace: Duration) -> Result<Vec<String>> {
        self.ensure_layout()?;
        let dir = self.blobs_dir();
        let now = SystemTime::now();

        let mut removed = Vec::new();
        let mut candidates: Vec<Digest> = Vec::new();

        let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
            .map_err(io_ctx(format!("read dir {}", to_slash(&dir))))?
            .map(|e| e.map(|e| e.path()))
            .collect::<io::Result<_>>()
            .map_err(io_ctx(format!("read dir {}", to_slash(&dir))))?;
        entries.sort();

        for path in entries {
            let md = fs::metadata(&path).map_err(io_ctx(format!("stat {}", to_slash(&path))))?;
            if md.is_dir() {
                continue;
            }
            // Clock skew, or an mtime the platform won't give us => "too young".
            let age = md
                .modified()
                .ok()
                .and_then(|m| now.duration_since(m).ok())
                .unwrap_or(Duration::ZERO);
            if age < grace {
                continue;
            }

            let filename = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            match Digest::parse(&filename) {
                Ok(d) => candidates.push(d),
                Err(_) => {
                    fs::remove_file(&path)
                        .map_err(io_ctx(format!("remove {}", to_slash(&path))))?;
                    removed.push(filename);
                }
            }
        }

        // Now subtract everything any manifest still points at.
        let mut in_use: BTreeSet<String> = BTreeSet::new();
        for (_, m) in self.manifests(true)? {
            for layer in m.layers_and_config() {
                if !layer.digest.is_empty() {
                    in_use.insert(layer.digest.clone());
                }
            }
        }

        for d in candidates {
            if in_use.contains(d.as_str()) {
                continue;
            }
            let p = self.blob_path(&d);
            match fs::remove_file(&p) {
                Ok(()) => removed.push(d.blob_filename()),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_ctx(format!("remove blob {}", to_slash(&p)))(e)),
            }
        }
        Ok(removed)
    }

    // -- internals ---------------------------------------------------------

    /// A unique scratch filename inside `blobs/`.
    ///
    /// **Upstream:** `os.CreateTemp(blobs, "sha256-")`.
    ///
    /// Must live in the **same directory** as the final blob, so the rename that
    /// follows is same-filesystem and therefore atomic. Deliberately does *not*
    /// look like a valid digest, so [`Store::prune_blobs`] treats an abandoned
    /// one as a partial download and sweeps it -- same fate upstream's temp files
    /// get, for the same reason.
    ///
    /// pid + a process-wide counter, so two threads and two processes both stay
    /// out of each other's way without pulling in a random-number dependency.
    fn temp_blob_path(&self) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        self.blobs_dir()
            .join(format!("sha256-partial.{}.{n}", std::process::id()))
    }

    /// Move a finished temp file to its content address, and set the mode.
    ///
    /// **Upstream:** the `os.Rename` + `os.Chmod(blob, 0o644)` pair in `NewLayer`.
    ///
    /// The chmod is Unix-only. Windows has no mode bits to set, so we skip it --
    /// that's a platform fact, not a divergence in intent, and a Windows blob
    /// ends up with the directory's inherited ACL as usual.
    fn finish_blob(&self, temp: &Path, blob: &Path) -> Result<()> {
        fs::rename(temp, blob).map_err(io_ctx(format!(
            "rename {} -> {}",
            to_slash(temp),
            to_slash(blob)
        )))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(blob, fs::Permissions::from_mode(0o644))
                .map_err(io_ctx(format!("chmod {}", to_slash(blob))))?;
        }
        Ok(())
    }
}

/// Children of `dir`, sorted by path. A missing directory reads as empty --
/// Go's `filepath.Glob` also just returns nothing rather than erroring.
///
/// The sort is what makes [`Store::manifests`] reproducible; `read_dir` order is
/// whatever the filesystem feels like, and NTFS and ext4 don't agree.
fn sorted_children(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(io_ctx(format!("read dir {}", to_slash(dir))))?
        .map(|e| e.map(|e| e.path()))
        .collect::<io::Result<_>>()
        .map_err(io_ctx(format!("read dir {}", to_slash(dir))))?;
    out.sort();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Hex, for the known-answer vectors below.
    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// **The known-answer test.** Everything else in this module proves the
    /// plumbing with a deliberately fake hash; this one proves the *arithmetic*
    /// against the published SHA-256 vectors (FIPS 180-4, Appendix B).
    ///
    /// Without it, a store could be perfectly self-consistent and still put
    /// every blob at an address ollama would never look in -- and nothing would
    /// notice until a real pull silently re-downloaded everything.
    #[test]
    fn the_real_hasher_matches_the_published_sha256_vectors() {
        let mut h = Sha256::new();

        // FIPS 180-4 B.1: SHA-256("abc")
        h.update(b"abc");
        assert_eq!(
            hex(&h.finalize_and_reset()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        // FIPS 180-4 B.2: the 448-bit message.
        h.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&h.finalize_and_reset()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );

        // The empty message -- the case an off-by-one in padding always breaks.
        assert_eq!(
            hex(&h.finalize_and_reset()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// The reset contract, asserted directly. An implementation that forgets to
    /// reset chains blob N+1's bytes onto blob N's, every digest after the first
    /// is wrong, and the store still happily writes them.
    #[test]
    fn the_real_hasher_resets_so_one_instance_can_hash_many_blobs() {
        let mut shared = Sha256::new();
        shared.update(b"abc");
        let first = shared.finalize_and_reset();
        shared.update(b"abc");
        let second = shared.finalize_and_reset();
        assert_eq!(first, second, "a reused hasher must not chain messages");

        let mut fresh = Sha256::new();
        fresh.update(b"abc");
        assert_eq!(first, fresh.finalize_and_reset());
    }

    /// Streaming must equal one-shot, or a 20 GB blob (which is always streamed
    /// in chunks) would land at a different address than the same bytes hashed
    /// whole.
    #[test]
    fn chunked_updates_hash_the_same_as_one_shot() {
        let msg: Vec<u8> = (0u8..=255).cycle().take(200_000).collect();

        let mut one = Sha256::new();
        one.update(&msg);
        let whole = one.finalize_and_reset();

        let mut many = Sha256::new();
        for chunk in msg.chunks(7) {
            many.update(chunk);
        }
        assert_eq!(whole, many.finalize_and_reset());
    }

    /// End to end with the REAL hasher: a blob must land at the address its own
    /// content dictates, and that address must be the one ollama would compute.
    #[test]
    fn a_blob_stored_with_the_real_hasher_lands_at_its_content_address() {
        let tmp = TempDir::new().unwrap();
        let store = Store::new(tmp.path());
        let mut h = Sha256::new();

        let layer = store
            .new_layer(b"abc", "application/vnd.ollama.image.model", &mut h)
            .unwrap();

        assert_eq!(
            layer.digest,
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "the digest must match SHA-256(\"abc\") exactly"
        );

        // ...and the file really is at that address, under the dash spelling.
        let d = Digest::parse(&layer.digest).unwrap();
        assert!(store.has_blob(&d));
        assert_eq!(store.read_blob(&d).unwrap(), b"abc");
        assert!(
            store.blob_path(&d).to_string_lossy().contains("sha256-"),
            "colon is illegal in a Windows filename"
        );

        // And verification agrees.
        store.verify_blob(&d, &mut h).unwrap();
    }

    /// **NOT SHA-256. Not even close, and never let it near a real store.**
    ///
    /// This crate has no hash dependency (see [`Sha256Hasher`]), so the tests
    /// inject a stand-in that satisfies the *shape* of the contract -- 32 bytes
    /// out, deterministic, different input gives different output, resets on
    /// finalize -- which is exactly what the store logic depends on. What it does
    /// **not** give is collision resistance or agreement with ollama's digests.
    /// So it can prove "the plumbing routes bytes to the right address and dedups
    /// identical content"; it cannot prove "our digests match ollama's". That
    /// second one needs a real sha256 plus a known-answer vector, and is the one
    /// outstanding item on this module.
    #[derive(Default)]
    struct FauxSha256 {
        state: [u8; 32],
        n: u64,
    }

    impl Sha256Hasher for FauxSha256 {
        fn update(&mut self, chunk: &[u8]) {
            for &b in chunk {
                let i = (self.n % 32) as usize;
                self.state[i] = self.state[i]
                    .wrapping_mul(31)
                    .wrapping_add(b)
                    .wrapping_add((self.n & 0xff) as u8);
                self.n += 1;
            }
        }
        fn finalize_and_reset(&mut self) -> [u8; 32] {
            let mut out = self.state;
            // Fold the length in, so "abc" and a longer repeat of it can't land
            // on the same 32 bytes purely by wrapping.
            for (i, byte) in self.n.to_le_bytes().iter().enumerate() {
                out[i] ^= byte;
            }
            self.state = [0; 32];
            self.n = 0;
            out
        }
    }

    fn store() -> (TempDir, Store) {
        let d = TempDir::new().expect("tempdir");
        let s = Store::new(d.path());
        (d, s)
    }

    fn digest_of(bytes: &[u8]) -> Digest {
        sha256_of_bytes(bytes, &mut FauxSha256::default())
    }

    // -- Digest ------------------------------------------------------------

    #[test]
    fn a_digest_round_trips_between_its_json_form_and_its_filename_form() {
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let d = Digest::parse(&format!("sha256:{hex}")).expect("parses");
        assert_eq!(d.as_str(), format!("sha256:{hex}"));
        assert_eq!(d.hex(), hex);
        assert_eq!(d.blob_filename(), format!("sha256-{hex}"));
        // The dash form is what a blob filename looks like, and it must parse
        // back to the same digest -- prune_blobs depends on exactly this.
        assert_eq!(Digest::parse(&d.blob_filename()).expect("parses"), d);
    }

    #[test]
    fn a_blob_filename_never_contains_a_colon_because_windows_forbids_it() {
        let d = Digest::from_hash([0xab; 32]);
        assert!(
            !d.blob_filename().contains(':'),
            "a colon in a filename cannot exist on NTFS"
        );
        assert!(
            d.as_str().contains(':'),
            "but the JSON form keeps the colon"
        );
    }

    #[test]
    fn a_digest_that_tries_to_climb_out_of_the_store_is_refused() {
        for bad in [
            "sha256:../../../etc/passwd",
            "sha256-../../../etc/passwd",
            "../../blobs/sha256-00",
            "sha256:..",
            "sha256:/etc/passwd",
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8/5",
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8\\5",
        ] {
            assert!(
                Digest::parse(bad).is_err(),
                "{bad:?} must never reach the filesystem"
            );
        }
    }

    #[test]
    fn a_digest_must_be_exactly_sixty_four_lowercase_hex_digits() {
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert!(Digest::parse(&format!("sha256:{hex}")).is_ok());
        assert!(Digest::parse(&format!("sha256-{hex}")).is_ok());
        // Too short, too long, wrong alphabet, no algorithm, wrong algorithm.
        assert!(Digest::parse("sha256:abc").is_err());
        assert!(Digest::parse(&format!("sha256:{hex}0")).is_err());
        assert!(Digest::parse(&format!("sha256:{}g", &hex[..63])).is_err());
        assert!(Digest::parse(hex).is_err());
        assert!(Digest::parse(&format!("sha512:{hex}")).is_err());
        // Deliberate divergence from upstream's [0-9a-fA-F]: uppercase is out,
        // because on ext4 it would address a second copy of the same content.
        assert!(Digest::parse(&format!("sha256:{}", hex.to_uppercase())).is_err());
    }

    #[test]
    fn from_hash_produces_lowercase_hex_the_way_gos_percent_x_does() {
        let d = Digest::from_hash([0x0f; 32]);
        assert_eq!(d.hex(), "0f".repeat(32));
        assert_eq!(Digest::parse(d.as_str()).expect("round trip"), d);
    }

    // -- Manifest JSON -----------------------------------------------------

    #[test]
    fn a_manifest_serialises_to_exactly_the_bytes_go_writes() {
        let (_d, s) = store();
        let n = Name::parse("host/namespace/model:tag");
        let config = Layer::new(MEDIA_TYPE_CONFIG, &Digest::from_hash([0xaa; 32]), 50);
        let layer = Layer::new(MEDIA_TYPE_MODEL, &Digest::from_hash([0xbb; 32]), 999);
        let p = s.write_manifest(&n, config, vec![layer]).expect("write");

        let raw = fs::read_to_string(&p).expect("read back");
        assert!(
            raw.ends_with('\n'),
            "json.Encoder.Encode appends a newline; the manifest digest is taken over it"
        );
        assert_eq!(raw.matches('\n').count(), 1, "compact, not pretty-printed");
        // Field order is Go's struct order.
        let head = format!(
            r#"{{"schemaVersion":2,"mediaType":"{MEDIA_TYPE_MANIFEST}","config":{{"mediaType":"{MEDIA_TYPE_CONFIG}","digest":"sha256:{}","size":50}}"#,
            "aa".repeat(32)
        );
        assert!(raw.starts_with(&head), "got: {raw}");
    }

    #[test]
    fn a_manifest_with_a_go_nil_layer_slice_reads_back_as_empty() {
        // Go marshals a nil slice as `null`, which plain serde `default` will
        // not accept. Upstream's own test writes exactly this, via `Manifest{}`.
        let m: Manifest =
            serde_json::from_str(r#"{"schemaVersion":0,"mediaType":"","layers":null}"#)
                .expect("null layers must read as empty");
        assert!(m.layers.is_empty());
        assert_eq!(m.size(), 0);
    }

    #[test]
    fn size_counts_every_layer_plus_the_config() {
        let m = Manifest {
            config: Layer::new(MEDIA_TYPE_CONFIG, &Digest::from_hash([1; 32]), 50),
            layers: vec![
                Layer::new(MEDIA_TYPE_MODEL, &Digest::from_hash([2; 32]), 900),
                Layer::new(MEDIA_TYPE_SYSTEM, &Digest::from_hash([3; 32]), 50),
            ],
            ..Default::default()
        };
        assert_eq!(m.size(), 1000);
        assert_eq!(m.layers_and_config().count(), 3);
        assert_eq!(m.layer(MEDIA_TYPE_MODEL).expect("model layer").size, 900);
        assert_eq!(m.layers_of(MEDIA_TYPE_SYSTEM).count(), 1);
        assert!(!m.human_size().is_empty());
    }

    #[test]
    fn a_layer_with_no_digest_has_no_address() {
        let l = Layer::default();
        assert!(matches!(
            l.checked_digest(),
            Err(ManifestError::EmptyDigest)
        ));
    }

    // -- paths -------------------------------------------------------------

    #[test]
    fn a_manifest_path_is_host_namespace_model_tag_under_manifests() {
        let (_d, s) = store();
        let n = Name::parse("qwen3");
        let p = s.manifest_path(&n).expect("valid name");
        let rel = p.strip_prefix(s.root()).expect("under root");
        assert_eq!(
            to_slash(rel),
            "manifests/registry.ollama.ai/library/qwen3/latest"
        );
    }

    #[test]
    fn a_name_that_tries_to_climb_out_of_the_store_is_refused() {
        let (_d, s) = store();
        for bad in [
            "../../etc/passwd",
            "..",
            "host/../../model:tag",
            "host/namespace/model:..",
        ] {
            let n = Name::parse(bad);
            assert!(
                s.manifest_path(&n).is_err(),
                "{bad:?} must not resolve to a store path"
            );
        }
    }

    #[test]
    fn every_path_this_module_prints_uses_forward_slashes() {
        let (_d, s) = store();
        let n = Name::parse("qwen3");
        s.write_manifest(&n, Layer::default(), vec![])
            .expect("write");
        let m = s.read_manifest(&n).expect("read");
        assert!(
            !m.path_slash().contains('\\'),
            "Windows separators must not leak into output: {}",
            m.path_slash()
        );
        assert!(m
            .path_slash()
            .ends_with("registry.ollama.ai/library/qwen3/latest"));
    }

    // -- the content-addressed property ------------------------------------

    /// The claim this whole module exists to make, measured instead of asserted
    /// by hand-waving: two tags, same weights, one copy on disk.
    #[test]
    fn two_tags_sharing_a_layer_store_the_bytes_once() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let weights = vec![7u8; 4096];

        // First model writes the bytes.
        let l1 = s
            .new_layer(&weights, MEDIA_TYPE_MODEL, &mut h)
            .expect("first layer");
        assert!(l1.status.starts_with("creating new layer"));

        // Second model hands over identical bytes -- nothing new gets written.
        let l2 = s
            .new_layer(&weights, MEDIA_TYPE_MODEL, &mut h)
            .expect("second layer");
        assert!(l2.status.starts_with("using existing layer"));
        assert_eq!(l1.digest, l2.digest, "same content, same address");

        let a = Name::parse("qwen3:0.6b");
        let b = Name::parse("my-qwen:latest");
        s.write_manifest(&a, Layer::default(), vec![l1.clone()])
            .expect("write a");
        s.write_manifest(&b, Layer::default(), vec![l2])
            .expect("write b");

        // Exactly one blob file, for two models.
        let blobs: Vec<_> = fs::read_dir(s.blobs_dir())
            .expect("blobs")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert_eq!(blobs.len(), 1, "two tags must not store the weights twice");

        // And the numbers say the same thing: declared twice, on disk once.
        assert_eq!(s.declared_size().expect("declared"), 8192);
        assert_eq!(s.disk_usage().expect("disk"), 4096);
    }

    #[test]
    fn deleting_one_of_two_tags_keeps_the_shared_blob_and_deleting_both_frees_it() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let layer = s
            .new_layer(b"shared weights", MEDIA_TYPE_MODEL, &mut h)
            .expect("layer");
        let d = layer.checked_digest().expect("digest");

        let a = Name::parse("qwen3:0.6b");
        let b = Name::parse("my-qwen:latest");
        s.write_manifest(&a, Layer::default(), vec![layer.clone()])
            .expect("write a");
        s.write_manifest(&b, Layer::default(), vec![layer])
            .expect("write b");

        // Drop the first tag. The blob is still spoken for.
        let ma = s.read_manifest(&a).expect("read a");
        s.remove_manifest(&a).expect("remove a");
        let removed = s.remove_unused_layers(&ma).expect("prune a");
        assert!(removed.is_empty(), "still referenced by the other tag");
        assert!(s.has_blob(&d), "blob must survive");

        // Drop the second. Now nobody wants it.
        let mb = s.read_manifest(&b).expect("read b");
        s.remove_manifest(&b).expect("remove b");
        let removed = s.remove_unused_layers(&mb).expect("prune b");
        assert_eq!(removed, vec![d.clone()]);
        assert!(!s.has_blob(&d), "last referrer gone, so the bytes go");
    }

    #[test]
    fn copying_a_model_copies_the_manifest_and_not_the_weights() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let layer = s
            .new_layer(&vec![3u8; 10_000], MEDIA_TYPE_MODEL, &mut h)
            .expect("layer");
        let src = Name::parse("qwen3:0.6b");
        let dst = Name::parse("copy:latest");
        s.write_manifest(&src, Layer::default(), vec![layer])
            .expect("write");

        s.copy_manifest(&src, &dst).expect("copy");

        assert_eq!(s.disk_usage().expect("disk"), 10_000, "no second copy");
        assert_eq!(
            s.read_manifest(&src).expect("src").layers,
            s.read_manifest(&dst).expect("dst").layers
        );
        // Same name in, same name out: a no-op, not an error.
        s.copy_manifest(&src, &src).expect("self copy is a no-op");
    }

    #[test]
    fn removing_a_manifest_prunes_the_empty_directories_behind_it() {
        let (_d, s) = store();
        let n = Name::parse("host/namespace/model:tag");
        s.write_manifest(&n, Layer::default(), vec![])
            .expect("write");
        assert!(s.manifests_dir().join("host/namespace/model").is_dir());

        s.remove_manifest(&n).expect("remove");
        assert!(
            !s.manifests_dir().join("host").exists(),
            "empty skeleton directories must not be left behind"
        );
    }

    // -- enumeration (upstream's TestManifests table) -----------------------

    /// Write a zero-value manifest at a raw store path, bypassing name
    /// validation -- upstream's `createManifest` test helper, which is how the
    /// invalid cases get onto disk at all.
    fn create_manifest_at(s: &Store, rel: &str) {
        let p = s.manifests_dir().join(rel);
        fs::create_dir_all(p.parent().expect("has parent")).expect("mkdir");
        let mut bytes = serde_json::to_vec(&Manifest::default()).expect("encode");
        bytes.push(b'\n');
        fs::write(&p, &bytes).expect("write");
    }

    /// **Upstream:** `TestManifests` in `manifest/manifest_test.go`, whole table.
    #[test]
    fn manifests_enumerates_exactly_the_valid_four_part_leaves() {
        struct Case {
            name: &'static str,
            paths: &'static [&'static str],
            want_valid: usize,
        }
        let cases = [
            Case {
                name: "empty",
                paths: &[],
                want_valid: 0,
            },
            Case {
                name: "single",
                paths: &["host/namespace/model/tag"],
                want_valid: 1,
            },
            Case {
                name: "multiple",
                paths: &[
                    "registry.ollama.ai/library/llama3/latest",
                    "registry.ollama.ai/library/llama3/q4_0",
                    "registry.ollama.ai/library/llama3/q4_1",
                    "registry.ollama.ai/library/llama3/q8_0",
                    "registry.ollama.ai/library/llama3/q5_0",
                    "registry.ollama.ai/library/llama3/q5_1",
                    "registry.ollama.ai/library/llama3/q2_K",
                    "registry.ollama.ai/library/llama3/q3_K_S",
                    "registry.ollama.ai/library/llama3/q3_K_M",
                    "registry.ollama.ai/library/llama3/q3_K_L",
                    "registry.ollama.ai/library/llama3/q4_K_S",
                    "registry.ollama.ai/library/llama3/q4_K_M",
                    "registry.ollama.ai/library/llama3/q5_K_S",
                    "registry.ollama.ai/library/llama3/q5_K_M",
                    "registry.ollama.ai/library/llama3/q6_K",
                ],
                want_valid: 15,
            },
            // A dotfile is a legal filename but not a legal tag -- a leading `.`
            // is exactly what `is_valid_part` forbids, and that is what stops a
            // name from ever being `..`.
            Case {
                name: "hidden",
                paths: &["host/namespace/model/tag", "host/namespace/model/.hidden"],
                want_valid: 1,
            },
            // Depth five: the level-four entry is a DIRECTORY, so it is not a
            // model. Upstream's `if !fi.IsDir()`.
            Case {
                name: "subdir",
                paths: &[
                    "host/namespace/model/tag/one",
                    "host/namespace/model/tag/another/one",
                ],
                want_valid: 0,
            },
            Case {
                name: "upper tag",
                paths: &["host/namespace/model/TAG"],
                want_valid: 1,
            },
            Case {
                name: "upper model",
                paths: &["host/namespace/MODEL/tag"],
                want_valid: 1,
            },
            Case {
                name: "upper namespace",
                paths: &["host/NAMESPACE/model/tag"],
                want_valid: 1,
            },
            Case {
                name: "upper host",
                paths: &["HOST/namespace/model/tag"],
                want_valid: 1,
            },
        ];

        for case in cases {
            let (_d, s) = store();
            for p in case.paths {
                create_manifest_at(&s, p);
            }
            let got = s.manifests(true).expect("enumerate");
            assert_eq!(
                got.len(),
                case.want_valid,
                "case {:?}: got {:?}",
                case.name,
                got.iter().map(|(n, _)| n.to_string()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn manifests_comes_back_sorted_so_listings_are_reproducible() {
        let (_d, s) = store();
        for tag in ["zulu", "alpha", "mike"] {
            create_manifest_at(&s, &format!("host/namespace/model/{tag}"));
        }
        let tags: Vec<String> = s
            .manifests(true)
            .expect("enumerate")
            .iter()
            .map(|(n, _)| n.tag.clone())
            .collect();
        assert_eq!(tags, vec!["alpha", "mike", "zulu"]);
    }

    #[test]
    fn a_corrupt_manifest_is_skipped_when_asked_to_continue_and_reported_otherwise() {
        let (_d, s) = store();
        create_manifest_at(&s, "host/namespace/good/tag");
        let bad = s.manifests_dir().join("host/namespace/bad/tag");
        fs::create_dir_all(bad.parent().expect("parent")).expect("mkdir");
        fs::write(&bad, b"{not json at all").expect("write");

        assert_eq!(s.manifests(true).expect("lenient").len(), 1);
        assert!(
            s.manifests(false).is_err(),
            "strict mode must surface the corruption"
        );
    }

    // -- blobs -------------------------------------------------------------

    #[test]
    fn a_blob_lands_at_its_content_address_and_verifies_against_it() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let bytes = b"the quick brown fox";
        let layer = s.new_layer(bytes, MEDIA_TYPE_MODEL, &mut h).expect("layer");
        let d = layer.checked_digest().expect("digest");

        assert_eq!(layer.size, bytes.len() as i64);
        assert_eq!(
            s.blob_path(&d)
                .file_name()
                .expect("filename")
                .to_string_lossy(),
            d.blob_filename()
        );
        assert_eq!(s.read_blob(&d).expect("read"), bytes);
        s.verify_blob(&d, &mut h).expect("verifies");
    }

    #[test]
    fn a_truncated_blob_fails_verification_instead_of_loading_silently() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let layer = s
            .new_layer(b"complete contents", MEDIA_TYPE_MODEL, &mut h)
            .expect("layer");
        let d = layer.checked_digest().expect("digest");

        fs::write(s.blob_path(&d), b"trunc").expect("corrupt it");
        assert!(matches!(
            s.verify_blob(&d, &mut h),
            Err(ManifestError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn a_layer_built_from_an_existing_blob_moves_no_bytes() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let first = s
            .new_layer(&vec![9u8; 2048], MEDIA_TYPE_MODEL, &mut h)
            .expect("layer");
        let d = first.checked_digest().expect("digest");

        let derived = s
            .new_layer_from_layer(&d, MEDIA_TYPE_MODEL, "qwen3:0.6b")
            .expect("derived");
        assert_eq!(derived.size, 2048);
        assert_eq!(derived.from, "qwen3:0.6b");
        assert_eq!(derived.digest, first.digest);
        assert_eq!(s.disk_usage().expect("disk"), 2048);

        // A digest not in the store is an error, not a silent stub.
        let missing = Digest::from_hash([0xfe; 32]);
        assert!(s
            .new_layer_from_layer(&missing, MEDIA_TYPE_MODEL, "")
            .is_err());
    }

    #[test]
    fn put_blob_reports_whether_it_wrote_or_reused() {
        let (_d, s) = store();
        let bytes = b"content";
        let d = digest_of(bytes);
        assert_eq!(s.put_blob(&d, bytes).expect("put"), BlobOutcome::Created);
        assert_eq!(
            s.put_blob(&d, bytes).expect("put again"),
            BlobOutcome::Reused
        );
        assert_eq!(s.disk_usage().expect("disk"), bytes.len() as u64);
        assert_eq!(BlobOutcome::Created.status(), "creating new layer");
        assert_eq!(BlobOutcome::Reused.status(), "using existing layer");
    }

    #[test]
    fn a_failed_write_leaves_no_temp_file_lying_around() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();

        struct Exploding;
        impl Read for Exploding {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("disk on fire"))
            }
        }

        assert!(s
            .new_layer_from_reader(Exploding, MEDIA_TYPE_MODEL, &mut h)
            .is_err());
        let left: Vec<_> = fs::read_dir(s.blobs_dir())
            .expect("blobs")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert!(left.is_empty(), "temp file leaked: {left:?}");
    }

    // -- config ------------------------------------------------------------

    #[test]
    fn the_config_layer_round_trips_a_configv2() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let cfg = ConfigV2 {
            model_format: "gguf".into(),
            model_family: "qwen3".into(),
            model_type: "0.6B".into(),
            file_type: "Q4_K_M".into(),
            architecture: "amd64".into(),
            os: "linux".into(),
            ..Default::default()
        };
        let blob = serde_json::to_vec(&cfg).expect("encode config");
        let config_layer = s
            .new_layer(&blob, MEDIA_TYPE_CONFIG, &mut h)
            .expect("config layer");

        let n = Name::parse("qwen3:0.6b");
        s.write_manifest(&n, config_layer, vec![]).expect("write");
        let m = s.read_manifest(&n).expect("read");
        assert_eq!(s.read_config(&m).expect("config"), cfg);
    }

    #[test]
    fn a_manifest_with_no_config_digest_reads_back_a_default_config() {
        let (_d, s) = store();
        let n = Name::parse("qwen3:0.6b");
        s.write_manifest(&n, Layer::default(), vec![])
            .expect("write");
        let m = s.read_manifest(&n).expect("read");
        assert_eq!(s.read_config(&m).expect("config"), ConfigV2::default());
    }

    #[test]
    fn a_named_json_layer_is_found_by_its_name_and_missing_ones_say_so() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let mut layer = s
            .new_layer(br#"{"answer":42}"#, MEDIA_TYPE_JSON, &mut h)
            .expect("layer");
        layer.name = "text_encoder/config.json".into();

        let n = Name::parse("qwen3:0.6b");
        s.write_manifest(&n, Layer::default(), vec![layer])
            .expect("write");
        let m = s.read_manifest(&n).expect("read");

        let v: serde_json::Value = s
            .read_layer_json(&m, "text_encoder/config.json")
            .expect("found");
        assert_eq!(v["answer"], 42);
        assert!(matches!(
            s.read_layer_json::<serde_json::Value>(&m, "nope.json"),
            Err(ManifestError::ConfigNotFound(_))
        ));
    }

    // -- digest of the manifest file ---------------------------------------

    #[test]
    fn reading_with_a_hasher_fills_the_manifest_digest_and_reading_without_leaves_it_empty() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let n = Name::parse("qwen3:0.6b");
        s.write_manifest(&n, Layer::default(), vec![])
            .expect("write");

        assert_eq!(s.read_manifest(&n).expect("plain").digest(), "");
        let m = s.read_manifest_with(&n, &mut h).expect("hashed");
        assert_eq!(m.digest().len(), 64, "hex, no sha256: prefix");
        assert!(m
            .digest()
            .bytes()
            .all(|c| c.is_ascii_digit() || (c.is_ascii_lowercase() && c <= b'f')));
        assert_eq!(m.file_size(), fs::metadata(m.path()).expect("stat").len());
        assert!(m.modified().is_some());

        // The enumerating variant fills it too.
        let all = s.manifests_with(true, &mut h).expect("enumerate");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].1.digest(), m.digest());
    }

    #[test]
    fn a_missing_model_is_not_found_rather_than_a_raw_io_error() {
        let (_d, s) = store();
        assert!(matches!(
            s.read_manifest(&Name::parse("nope:latest")),
            Err(ManifestError::NotFound(_))
        ));
        assert!(!s.exists(&Name::parse("nope:latest")));
        assert!(matches!(
            s.remove_manifest(&Name::parse("nope:latest")),
            Err(ManifestError::NotFound(_))
        ));
    }

    // -- pruning -----------------------------------------------------------

    #[test]
    fn pruning_keeps_referenced_blobs_and_sweeps_orphans_and_partial_downloads() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();

        let kept = s
            .new_layer(b"referenced weights", MEDIA_TYPE_MODEL, &mut h)
            .expect("kept");
        let orphan = s
            .new_layer(b"nobody wants me", MEDIA_TYPE_MODEL, &mut h)
            .expect("orphan");
        let n = Name::parse("qwen3:0.6b");
        s.write_manifest(&n, Layer::default(), vec![kept.clone()])
            .expect("write");

        // A partial download: a name that is not a valid content address.
        let partial = s.blobs_dir().join("sha256-1234567890");
        fs::write(&partial, b"half a download").expect("write partial");

        // Zero grace, so everything is old enough to consider.
        let removed = s.prune_blobs(Duration::ZERO).expect("prune");
        assert!(
            s.has_blob(&kept.checked_digest().expect("d")),
            "referenced blob must survive"
        );
        assert!(
            !s.has_blob(&orphan.checked_digest().expect("d")),
            "orphan must go"
        );
        assert!(!partial.exists(), "partial download must go");
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn a_blob_inside_the_grace_period_survives_even_with_nobody_referencing_it() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let fresh = s
            .new_layer(b"just downloaded", MEDIA_TYPE_MODEL, &mut h)
            .expect("layer");
        let d = fresh.checked_digest().expect("digest");

        // This is the pull race upstream's grace period exists for: blobs get
        // written before the manifest that references them.
        let removed = s.prune_blobs(LAYER_PRUNE_GRACE_PERIOD).expect("prune");
        assert!(removed.is_empty());
        assert!(s.has_blob(&d));
    }

    #[test]
    fn remove_layer_refuses_while_anything_still_points_at_the_blob() {
        let (_d, s) = store();
        let mut h = FauxSha256::default();
        let layer = s
            .new_layer(b"weights", MEDIA_TYPE_MODEL, &mut h)
            .expect("layer");
        let n = Name::parse("qwen3:0.6b");
        s.write_manifest(&n, Layer::default(), vec![layer.clone()])
            .expect("write");

        assert!(!s.remove_layer(&layer).expect("still used"));
        s.remove_manifest(&n).expect("remove manifest");
        assert!(s.remove_layer(&layer).expect("now orphaned"));
        assert!(!s.has_blob(&layer.checked_digest().expect("d")));
        // An empty digest is nothing to remove, and must not error.
        assert!(!s.remove_layer(&Layer::default()).expect("empty digest"));
    }

    #[test]
    fn prune_directory_removes_empty_trees_but_leaves_occupied_ones_alone() {
        let (_d, s) = store();
        let root = s.manifests_dir();
        fs::create_dir_all(root.join("a/b/c")).expect("mkdir");
        fs::create_dir_all(root.join("d/e")).expect("mkdir");
        fs::write(root.join("d/e/keep"), b"x").expect("write");

        prune_directory(&root).expect("prune");
        assert!(!root.join("a").exists());
        assert!(root.join("d/e/keep").is_file());
    }
}
