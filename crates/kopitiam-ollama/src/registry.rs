//! # The registry client -- pull and push blobs over the ollama registry
//!
//! **Upstream:** `server/download.go`, `server/upload.go`, `server/auth.go`,
//! `auth/auth.go`, and the registry half of `server/images.go`
//! (`pullModelManifest`, `makeRequest`, `makeRequestWithRetry`,
//! `parseRegistryChallenge`, `getValue`, `verifyBlob`, `registryOptions`), all
//! under `crates/kopitiam-ai/vendor/ollama/`.
//!
//! ## What this module is really for
//!
//! One thing: **a 20 GB model must survive a lousy connection.** Everything here
//! exists because of that. The blob get cut into parts, every part keep its own
//! byte counter in a little sidecar file next to the blob, and when the network
//! drop or you press ctrl-C, that counter stay on disk. Next pull, we read the
//! sidecars back and ask the server only for the bytes still missing. **A part
//! that is half done must never restart from zero** -- that is the one invariant
//! this whole file protects, and
//! `a_half_done_part_resumes_from_its_recorded_offset_not_zero` is the test that
//! keep us honest.
//!
//! ## The network seam -- no socket in this crate
//!
//! Every byte in and out goes through the [`Transport`] trait, and every bit of
//! non-determinism (clock, sleep, randomness) through [`Ambient`]. Same house
//! pattern as `kopitiam-models`' `Fetcher` (see
//! `crates/kopitiam-models/src/fetch.rs`): the logic is the product, the socket
//! is only one implementation of it. So the *entire* download state machine --
//! part splitting, sidecar persistence, resume, retry, backoff -- is driven in
//! tests by a fake that never opens a connection and never sleeps.
//!
//! **There is deliberately NO real `Transport` in this crate yet.** `ureq` and an
//! ed25519 signer are dependencies this crate does not have, and CLAUDE.md say
//! stop-and-report rather than add. See the table below. Offline First is not
//! hurt by this hor: with no networking compiled in at all, the crate still
//! builds, still tests, and the store half (`crate::manifest`) is fully usable.
//!
//! ## Dependencies still owed (nothing here is guesswork -- it is just missing)
//!
//! | Need | Crate we'd want | Why nothing else will do |
//! |---|---|---|
//! | HTTP + TLS | `ureq` with `rustls` | house choice, already used by `kopitiam-models`; note the honest `ring` caveat in that crate's `HttpFetcher` docs -- rustls is pure Rust, its default crypto provider is not |
//! | ed25519 signing | e.g. `ed25519-dalek` + an OpenSSH private-key reader | the registry authenticates with an SSH ed25519 key at `~/.ollama/id_ed25519`; [`Signer`] is the hole it plugs into |
//! | CSPRNG for the nonce | `getrandom` | [`SystemAmbient`]'s randomness is a plain xorshift -- fine for backoff jitter, **not** fine for an auth nonce |
//! | sparse-file hint on Windows | `windows-sys` (`FSCTL_SET_SPARSE`) | see [`BlobDownload::run`]; without it a pull reserves the full blob size up front |
//! | md5 (push only) | `md-5` | the push commit step sends an `etag` built from per-part md5 sums; [`Md5Hasher`] is the hole |
//!
//! ## Attribution
//!
//! ollama is MIT, Copyright (c) Ollama. This module is a **translation**, not an
//! inspiration -- every function names the Go symbol it came from, and every
//! constant names where its number came from. Project-level record lives in
//! `docs/ACKNOWLEDGEMENTS.md`.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::manifest::{to_slash, Digest, Layer, Manifest, ManifestError, Sha256, Sha256Hasher};
use crate::name::Name;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong talking to a registry.
///
/// Each variant names the Go error it stands in for, because the *identity* of
/// these errors is load-bearing: [`BlobDownload`]'s retry loop branches on them,
/// and getting a branch wrong turns a resumable hiccup into a restart-from-zero.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Six attempts, still no good. **Upstream:** `errMaxRetriesExceeded`.
    #[error("max retries exceeded: {0}")]
    MaxRetriesExceeded(String),

    /// No byte arrived for [`DOWNLOAD_STALL_TIMEOUT`]. **Upstream:**
    /// `errPartStalled`.
    ///
    /// Special hor: a stall does **not** spend a retry, same as upstream's
    /// `try--`. Lousy hotel wifi should not burn the budget meant for real
    /// failures.
    #[error("part stalled")]
    PartStalled,

    /// **Upstream:** `errMaxRedirectsExceeded`.
    #[error("maximum redirects exceeded (10) for directURL")]
    MaxRedirectsExceeded,

    /// Two 401s in a row. **Upstream:** `errUnauthorized`.
    #[error("unauthorized: access denied")]
    Unauthorized,

    /// A 404, which upstream deliberately turns into `os.ErrNotExist` so callers
    /// can `errors.Is` it. `uploadBlob` depends on exactly this: a 404 from the
    /// HEAD means "blob not there yet, go upload it", not "give up".
    #[error("not found")]
    NotExist,

    /// Any other >= 400, with the body attached. **Upstream:**
    /// `fmt.Errorf("%d: %s", resp.StatusCode, responseBody)`.
    #[error("{code}: {body}")]
    Status { code: u16, body: String },

    /// A status we were not expecting at all. **Upstream:**
    /// `fmt.Errorf("unexpected status code %d", ...)` in the direct-URL step.
    #[error("unexpected status code {0}")]
    UnexpectedStatus(u16),

    /// The [`CancelToken`] was tripped. **Upstream:** `context.Canceled`.
    ///
    /// Counts as **resumable**: whatever bytes already landed get committed to
    /// the sidecar before we unwind, so ctrl-C then pull again picks up mid-part.
    #[error("canceled")]
    Canceled,

    /// The body ended early. **Upstream:** `io.ErrUnexpectedEOF`, the other
    /// resumable case.
    #[error("unexpected end of body")]
    UnexpectedEof,

    /// Device full. **Upstream:** `syscall.ENOSPC`, which aborts immediately --
    /// retrying a full disk is just being stubborn.
    #[error("no space left on device")]
    OutOfSpace,

    /// `http://` registry without `--insecure`. **Upstream:**
    /// `errInsecureProtocol`.
    #[error("insecure protocol http")]
    InsecureProtocol,

    /// The `WWW-Authenticate` realm points somewhere other than the host we were
    /// talking to. **Upstream:** the guard in `getAuthorizationToken`.
    ///
    /// Security check, not tidiness: without it a hostile registry can name any
    /// realm it likes and we would cheerfully post a signed token to it.
    #[error("realm host {realm:?} does not match original host {original:?}")]
    RealmHostMismatch { realm: String, original: String },

    /// Could not parse a URL. **Upstream:** whatever `url.Parse` returned.
    #[error("invalid url {0:?}")]
    InvalidUrl(String),

    /// No `Location` header where one was required. **Upstream:**
    /// `http.ErrNoLocation` out of `resp.Location()`.
    #[error("no Location header in response")]
    NoLocation,

    /// The [`Transport`] itself failed -- DNS, TLS, connection reset, whatever.
    #[error("transport: {0}")]
    Transport(String),

    /// No [`Signer`] wired in, but the registry asked us to authenticate.
    #[error("no signer configured -- registry auth needs an ed25519 key")]
    NoSigner,

    /// Signing failed (bad key file, wrong key type, ...).
    #[error("signing failed: {0}")]
    Signing(String),

    /// Store / digest trouble, straight from [`crate::manifest`].
    #[error(transparent)]
    Manifest(#[from] ManifestError),

    /// Filesystem failure, with the operation attached -- a bare `io::Error`
    /// never tell you *which* file died.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: io::Error,
    },

    /// Malformed JSON in a manifest or a part sidecar.
    #[error("{context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },
}

impl RegistryError {
    /// Is this the kind of failure where the bytes we already got are still
    /// good, so we commit progress before unwinding?
    ///
    /// **Upstream:** the `!errors.Is(err, context.Canceled) &&
    /// !errors.Is(err, io.ErrUnexpectedEOF)` guard in `downloadChunk`. Get this
    /// wrong and ctrl-C throws away a part's progress -- exactly the bug this
    /// module exists to not have.
    pub fn is_resumable(&self) -> bool {
        matches!(self, RegistryError::Canceled | RegistryError::UnexpectedEof)
    }

    /// Must we abort the whole download right now, no retry?
    ///
    /// **Upstream:** `case errors.Is(err, context.Canceled),
    /// errors.Is(err, syscall.ENOSPC): return err`.
    pub fn is_fatal(&self) -> bool {
        matches!(self, RegistryError::Canceled | RegistryError::OutOfSpace)
    }
}

/// Shorthand so every `?` on a filesystem call can say what it was doing.
fn io_ctx(context: impl Into<String>) -> impl FnOnce(io::Error) -> RegistryError {
    move |source| RegistryError::Io {
        context: context.into(),
        source,
    }
}

fn json_ctx(context: impl Into<String>) -> impl FnOnce(serde_json::Error) -> RegistryError {
    move |source| RegistryError::Json {
        context: context.into(),
        source,
    }
}

/// `Result` with [`RegistryError`] baked in.
pub type Result<T> = std::result::Result<T, RegistryError>;

// ---------------------------------------------------------------------------
// Cancellation -- our stand-in for context.Context
// ---------------------------------------------------------------------------

/// The port of Go's `context.Context` cancellation, minus deadlines and values.
///
/// Clone it, hand copies to worker threads, then [`CancelToken::cancel`] from
/// anywhere: every copy see it. Ollama leans on `ctx.Err()` all over
/// `download.go` -- notably it is the ONLY thing bounding the stall retry loop
/// (a stall doesn't spend a retry, so without cancellation that loop is forever).
/// So this is not decoration hor, it is what make a stuck pull stoppable.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    /// A parent's flag, if this token was made by [`CancelToken::child`].
    /// Cancelling the parent cancels us; cancelling us leaves the parent alone.
    parent: Option<Arc<AtomicBool>>,
}

impl CancelToken {
    /// A fresh, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// A token that dies when either it or `self` is cancelled.
    ///
    /// **Upstream:** `errgroup.WithContext(ctx)`, whose derived context is
    /// cancelled by the parent *or* by the first worker that returns an error.
    /// One level is all the download needs, so that is all this does.
    pub fn child(&self) -> CancelToken {
        CancelToken {
            flag: Arc::new(AtomicBool::new(false)),
            parent: Some(self.flag.clone()),
        }
    }

    /// Trip it. Every clone sees this, and it cannot be un-tripped.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Tripped already, or our parent tripped?
    pub fn is_canceled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
            || self
                .parent
                .as_ref()
                .is_some_and(|p| p.load(Ordering::SeqCst))
    }

    /// `Err(Canceled)` if tripped, else `Ok(())`. **Upstream:** `ctx.Err()`.
    pub fn check(&self) -> Result<()> {
        if self.is_canceled() {
            Err(RegistryError::Canceled)
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// A very small URL type
// ---------------------------------------------------------------------------

/// Just enough URL to talk to a registry -- **not** a general RFC 3986 parser.
///
/// Why not pull in the `url` crate: everything this module *builds* is already
/// validated upstream of here ([`Name`] parts by `name::is_valid_part`, digests
/// by [`Digest::parse`]), and the only URLs we *parse* come from a `Location` or
/// `realm` header. Small surface, so a dependency is not worth it.
///
/// **What would make this wrong:** userinfo (`https://user:pw@host/`) is not
/// supported -- it would be swallowed into [`Url::authority`] and then compared
/// as if it were a host, which would break the realm-host check in
/// [`get_authorization_token`]. Registries don't use userinfo, but if one ever
/// does, fix it here, not at the call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    /// `https`, `http`. Always lowercase, never empty.
    pub scheme: String,
    /// `host` or `host:port`. Go calls this `URL.Host`.
    pub authority: String,
    /// Begins with `/`, or is empty. No query, no fragment.
    pub path: String,
    /// Everything after `?`, already encoded. No leading `?`.
    pub query: String,
}

impl Url {
    /// Parse an absolute URL. Fragments are dropped (a registry never needs one).
    ///
    /// **Upstream:** `url.Parse`, restricted to the absolute case.
    /// `Url::parse("://invalid")` fails, which is what upstream's
    /// `TestRegistryChallengeURLInvalid` asserts.
    pub fn parse(s: &str) -> Result<Self> {
        let bad = || RegistryError::InvalidUrl(s.to_string());
        let (scheme, rest) = s.split_once("://").ok_or_else(bad)?;
        if scheme.is_empty()
            || !scheme.as_bytes()[0].is_ascii_alphabetic()
            || !scheme
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.'))
        {
            return Err(bad());
        }
        // Fragment first: it can appear anywhere after the authority.
        let rest = rest.split('#').next().unwrap_or("");
        let (before_query, query) = match rest.split_once('?') {
            Some((a, b)) => (a, b.to_string()),
            None => (rest, String::new()),
        };
        let (authority, path) = match before_query.find('/') {
            Some(i) => (&before_query[..i], before_query[i..].to_string()),
            None => (before_query, String::new()),
        };
        if authority.is_empty() {
            return Err(bad());
        }
        Ok(Url {
            scheme: scheme.to_ascii_lowercase(),
            authority: authority.to_string(),
            path,
            query,
        })
    }

    /// `host:port` -- **upstream:** `URL.Host`. This is what
    /// [`get_authorization_token`] compares, so the port is part of the
    /// comparison: `localhost:5000` and `localhost:6000` are different hosts.
    pub fn host(&self) -> &str {
        &self.authority
    }

    /// `host` with the port stripped -- **upstream:** `URL.Hostname()`. This is
    /// what the redirect policy compares, so a registry may redirect
    /// `example.com:443` -> `example.com:8443` and still count as same-host,
    /// exactly like Go.
    pub fn hostname(&self) -> &str {
        // IPv6 literals are bracketed: [::1]:5000 -> [::1].
        if self.authority.starts_with('[')
            && let Some(end) = self.authority.find(']')
        {
            return &self.authority[..=end];
        }
        match self.authority.rsplit_once(':') {
            Some((h, _)) => h,
            None => &self.authority,
        }
    }

    /// Append path segments. **Upstream:** `(*url.URL).JoinPath`.
    ///
    /// A segment may itself contain `/` -- that is how upstream gets
    /// `v2/library/qwen3/...` out of one `DisplayNamespaceModel()` segment --
    /// and a `:` inside a segment is left alone, because `:` is a legal `pchar`
    /// and the blob digest (`sha256:abcd...`) rides in the path with its colon
    /// intact. Do NOT "helpfully" percent-encode here; the registry matches on
    /// the literal colon.
    pub fn join_path<'s, I: IntoIterator<Item = &'s str>>(&self, segments: I) -> Url {
        let mut path = self.path.trim_end_matches('/').to_string();
        for seg in segments {
            path.push('/');
            path.push_str(seg.trim_matches('/'));
        }
        Url {
            path,
            ..self.clone()
        }
    }

    /// Same as [`Url::join_path`] but keeps a trailing slash -- the blob-upload
    /// endpoint is literally `v2/{ns}/{model}/blobs/uploads/`, slash and all.
    pub fn join_path_dir<'s, I: IntoIterator<Item = &'s str>>(&self, segments: I) -> Url {
        let mut u = self.join_path(segments);
        u.path.push('/');
        u
    }

    /// Replace the query with `pairs`, encoded Go-style. **Upstream:**
    /// `u.RawQuery = values.Encode()`.
    pub fn with_query(&self, pairs: &[(&str, &str)]) -> Url {
        Url {
            query: encode_query(pairs),
            ..self.clone()
        }
    }

    /// First value for `key`, like Go's `Values.Get`.
    pub fn query_value(&self, key: &str) -> Option<String> {
        self.query_values(key).into_iter().next()
    }

    /// Every value for `key` -- `scope` legitimately repeats.
    pub fn query_values(&self, key: &str) -> Vec<String> {
        self.query
            .split('&')
            .filter(|kv| !kv.is_empty())
            .filter_map(|kv| kv.split_once('='))
            .filter(|(k, _)| query_unescape(k) == key)
            .map(|(_, v)| query_unescape(v))
            .collect()
    }

    /// Resolve a `Location` header against this URL. **Upstream:**
    /// `(*http.Response).Location()`, which does `u.ResolveReference(url)`.
    ///
    /// Handles the three shapes a registry actually sends: absolute
    /// (`https://cdn.example/...` -- the presigned-CDN case that makes the
    /// parallel download possible at all), root-relative (`/v2/...`), and
    /// path-relative. An empty header is [`RegistryError::NoLocation`], matching
    /// `http.ErrNoLocation`.
    pub fn resolve(&self, location: &str) -> Result<Url> {
        if location.is_empty() {
            return Err(RegistryError::NoLocation);
        }
        if let Ok(abs) = Url::parse(location) {
            return Ok(abs);
        }
        let (before_query, query) = match location.split_once('?') {
            Some((a, b)) => (a, b.to_string()),
            None => (location, String::new()),
        };
        if let Some(rest) = before_query.strip_prefix("//") {
            // Protocol-relative: //host/path keeps our scheme.
            let (authority, p) = match rest.find('/') {
                Some(i) => (&rest[..i], rest[i..].to_string()),
                None => (rest, String::new()),
            };
            return Ok(Url {
                scheme: self.scheme.clone(),
                authority: authority.to_string(),
                path: p,
                query,
            });
        }
        let path = if before_query.starts_with('/') {
            before_query.to_string()
        } else {
            let base = match self.path.rfind('/') {
                Some(i) => &self.path[..=i],
                None => "/",
            };
            format!("{base}{before_query}")
        };
        Ok(Url {
            scheme: self.scheme.clone(),
            authority: self.authority.clone(),
            path,
            query,
        })
    }
}

impl std::fmt::Display for Url {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}://{}{}", self.scheme, self.authority, self.path)?;
        if !self.query.is_empty() {
            write!(f, "?{}", self.query)?;
        }
        Ok(())
    }
}

/// Encode query pairs the way Go's `url.Values.Encode()` does: **sorted by key**,
/// each key's values kept in insertion order.
///
/// The sort is not cosmetic -- the auth realm request carries `nonce`, `scope`
/// (possibly twice), `service` and `ts`, and a registry that signs or logs the
/// canonical query string will disagree with us if the order differ. Go gets
/// sorted order for free; we must do it on purpose.
pub fn encode_query(pairs: &[(&str, &str)]) -> String {
    let mut keys: Vec<&str> = Vec::new();
    for (k, _) in pairs {
        if !keys.contains(k) {
            keys.push(k);
        }
    }
    keys.sort_unstable();
    let mut out = String::new();
    for k in keys {
        for (pk, pv) in pairs.iter().filter(|(pk, _)| *pk == k) {
            if !out.is_empty() {
                out.push('&');
            }
            out.push_str(&query_escape(pk));
            out.push('=');
            out.push_str(&query_escape(pv));
        }
    }
    out
}

/// Go's `url.QueryEscape`: unreserved is `A-Za-z0-9-_.~`, a space becomes `+`,
/// everything else becomes `%XX` in **uppercase** hex.
///
/// The `+`-for-space and the uppercase hex are both Go behaviours we copy on
/// purpose -- a registry comparing the raw query string byte-for-byte would
/// otherwise see something different from what ollama sends.
pub fn query_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(b))
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(char::from(b"0123456789ABCDEF"[(b >> 4) as usize]));
                out.push(char::from(b"0123456789ABCDEF"[(b & 0x0f) as usize]));
            }
        }
    }
    out
}

/// The inverse of [`query_escape`]. A malformed escape is left as-is instead of
/// erroring -- this is only used to read back what we ourselves wrote.
pub fn query_unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => {
                match (
                    (b[i + 1] as char).to_digit(16),
                    (b[i + 2] as char).to_digit(16),
                ) {
                    (Some(h), Some(l)) => {
                        out.push((h * 16 + l) as u8);
                        i += 3;
                    }
                    _ => {
                        out.push(b[i]);
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// base64 -- small enough to own, load-bearing enough to test
// ---------------------------------------------------------------------------

const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64(input: &[u8], alphabet: &[u8; 64], pad: bool) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(char::from(alphabet[(n >> 18) as usize & 63]));
        out.push(char::from(alphabet[(n >> 12) as usize & 63]));
        if chunk.len() > 1 {
            out.push(char::from(alphabet[(n >> 6) as usize & 63]));
        } else if pad {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(char::from(alphabet[n as usize & 63]));
        } else if pad {
            out.push('=');
        }
    }
    out
}

/// Go's `base64.StdEncoding.EncodeToString` -- `+/` alphabet, `=` padded.
///
/// Used for the signature blob and for HTTP Basic auth. Hand-written rather than
/// depended on: it is twenty lines, it is checked against the RFC 4648 §10
/// vectors in this module's tests, and a supply-chain entry for something this
/// small is a bad trade.
pub fn base64_std(input: &[u8]) -> String {
    b64(input, B64_STD, true)
}

/// Go's `base64.RawURLEncoding.EncodeToString` -- `-_` alphabet, **no** padding.
///
/// **Upstream:** `auth.NewNonce` uses exactly this. The nonce rides in a query
/// string, so it must be the URL alphabet and unpadded (a `=` would have to be
/// percent-encoded, and then the registry sees a different nonce than we signed).
pub fn base64_raw_url(input: &[u8]) -> String {
    b64(input, B64_URL, false)
}

// ---------------------------------------------------------------------------
// Ambient -- the one seam for time and randomness
// ---------------------------------------------------------------------------

/// Clock, sleeper and dice, behind one trait.
///
/// Ollama reaches straight for `time.Now()`, `time.Sleep` and `rand.Float64()`.
/// If we did the same, the backoff schedule would be untestable (you cannot
/// assert "it waited 32 seconds" without waiting 32 seconds) and the auth nonce
/// would be unpinnable. So all three go through here. Tests pass a fake that
/// records sleeps instead of taking them, which is why
/// `part_retry_backoff_doubles_one_two_four_eight` runs in microseconds.
///
/// Implementations must be `Send + Sync`: the download runs parts on several
/// threads and every one of them may sleep.
pub trait Ambient: Send + Sync {
    /// Wall clock, seconds since the Unix epoch. **Upstream:**
    /// `time.Now().Unix()`, used for the auth request's `ts` parameter.
    fn unix_secs(&self) -> i64;

    /// Block for `d`. **Upstream:** `time.Sleep(sleep)`.
    fn sleep(&self, d: Duration);

    /// A uniform value in `[0, 1)`. **Upstream:** `rand.Float64()`, used only to
    /// jitter the backoff. Does **not** need to be cryptographic.
    fn random_f64(&self) -> f64;

    /// Fill `out` with random bytes for the auth nonce. **Upstream:**
    /// `auth.NewNonce(rand.Reader, 16)` -- and `crypto/rand` there means this one
    /// **does** need to be cryptographic. [`SystemAmbient`]'s is not; see its docs.
    fn random_bytes(&self, out: &mut [u8]);
}

/// The std-only [`Ambient`]: real clock, real sleep, xorshift dice.
///
/// # The randomness caveat, said out loud
///
/// [`SystemAmbient::random_bytes`] is a **xorshift64\* PRNG seeded from the
/// clock**. That is perfectly fine for [`Ambient::random_f64`] (its only job is
/// to smear retries so a thousand clients don't stampede the registry together),
/// but it is **NOT good enough for the auth nonce**, which upstream draws from
/// `crypto/rand`. A predictable nonce lets an attacker who has seen one signed
/// request replay it. Wiring `getrandom` in is the fix; until then, do not use
/// this impl against a real registry that authenticates. This is written here,
/// at the point of use, rather than buried in a TODO, precisely so nobody ships
/// it by accident.
#[derive(Debug)]
pub struct SystemAmbient {
    state: Mutex<u64>,
}

impl SystemAmbient {
    /// Seeded from the current time. Two processes started in the same
    /// nanosecond would share a stream -- acceptable for jitter, see the type
    /// docs for why it is not acceptable for nonces.
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x2545_F491_4F6C_DD1D)
            | 1;
        Self {
            state: Mutex::new(seed),
        }
    }

    /// xorshift64*, Vigna 2016. Chosen for being ten lines and dependency-free,
    /// not for statistical quality.
    fn next_u64(&self) -> u64 {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

impl Default for SystemAmbient {
    fn default() -> Self {
        Self::new()
    }
}

impl Ambient for SystemAmbient {
    fn unix_secs(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }

    fn random_f64(&self) -> f64 {
        // Top 53 bits -> [0,1), the standard trick, same range as Go's
        // rand.Float64().
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn random_bytes(&self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let n = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&n[..chunk.len()]);
        }
    }
}

// ---------------------------------------------------------------------------
// The transport seam
// ---------------------------------------------------------------------------

/// The HTTP verbs the registry protocol uses. Nothing else is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Head,
    Post,
    Put,
    Patch,
}

impl Method {
    /// The wire spelling, uppercase, as it goes in the request line -- and also
    /// what gets baked into the signed auth payload, so it must stay uppercase.
    pub fn as_str(&self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Head => "HEAD",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Patch => "PATCH",
        }
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What we send up.
///
/// **This is our port of Go's `io.ReadSeeker`**, and the reason it is a *value*
/// rather than a reader: [`make_request_with_retry`] must be able to send the
/// same body twice, because a 401 is answered by fetching a token and replaying
/// the request. Upstream does `body.Seek(0, io.SeekStart)` for that. A `Body`
/// you can just hand over again is the same guarantee with none of the
/// footguns -- there is no "forgot to rewind" bug available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Body {
    /// No body at all. `Content-Length: 0` is the transport's business.
    Empty,
    /// In-memory bytes -- a manifest, mostly.
    Bytes(Vec<u8>),
    /// `len` bytes of `path` starting at `offset`. **Upstream:**
    /// `io.NewSectionReader(b.file, part.Offset, part.Size)` in `uploadPart`.
    ///
    /// Deliberately a *description* of a byte range, not an open handle: a push
    /// part can be 1000 MB, and slurping that into a `Vec` to hand to the
    /// transport would defeat the whole point of chunking.
    FileRange { path: PathBuf, offset: u64, len: u64 },
}

impl Body {
    /// Byte length, when known without touching the disk. `None` for
    /// [`Body::FileRange`] would be wrong -- we know it exactly -- so this is
    /// always `Some` except conceptually never. Used to set `Content-Length`.
    pub fn len(&self) -> u64 {
        match self {
            Body::Empty => 0,
            Body::Bytes(b) => b.len() as u64,
            Body::FileRange { len, .. } => *len,
        }
    }

    /// Nothing to send?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What comes back down.
///
/// `body` is a boxed reader on purpose: a blob chunk is up to 1000 MB and must
/// be streamed straight to disk, never buffered whole.
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Header name/value pairs, in arrival order. Names are compared
    /// case-insensitively by [`Response::header`], per RFC 9110.
    pub headers: Vec<(String, String)>,
    /// The body, streamed.
    pub body: Box<dyn Read + Send>,
}

impl std::fmt::Debug for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Response")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

impl Response {
    /// A response with no body -- handy for fakes and for HEAD.
    pub fn new(status: u16, headers: Vec<(String, String)>) -> Self {
        Response {
            status,
            headers,
            body: Box::new(io::empty()),
        }
    }

    /// A response carrying `bytes`.
    pub fn with_bytes(status: u16, headers: Vec<(String, String)>, bytes: Vec<u8>) -> Self {
        Response {
            status,
            headers,
            body: Box::new(io::Cursor::new(bytes)),
        }
    }

    /// Case-insensitive header lookup, first match wins.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// `Content-Length` as an `i64`, or `0` when absent or unparseable.
    ///
    /// **Upstream:** `b.Total, _ = strconv.ParseInt(resp.Header.Get(...), 10, 64)`
    /// -- note the discarded error, so a missing header silently means zero. We
    /// keep that: a zero total yields zero parts, and the caller sees an empty
    /// download rather than a crash. Faithful, but worth knowing hor.
    pub fn content_length(&self) -> i64 {
        self.header("Content-Length")
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(0)
    }

    /// The `Docker-Upload-Location` header, falling back to `Location`.
    ///
    /// **Upstream:** that exact two-step, repeated in `blobUpload.Prepare` and
    /// `uploadPart`. Registries behind a proxy that rewrites `Location` are why
    /// the Docker-specific header exists at all.
    pub fn upload_location(&self) -> Option<&str> {
        self.header("Docker-Upload-Location")
            .filter(|s| !s.is_empty())
            .or_else(|| self.header("Location"))
            .filter(|s| !s.is_empty())
    }

    /// Resolve `Location` against the URL the request went to. **Upstream:**
    /// `(*http.Response).Location()`.
    pub fn location(&self, request_url: &Url) -> Result<Url> {
        request_url.resolve(self.header("Location").unwrap_or(""))
    }

    /// Drain the body. Only for small responses (manifests, error payloads,
    /// tokens) -- never for a blob chunk.
    pub fn read_to_end(&mut self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.body
            .read_to_end(&mut buf)
            .map_err(io_ctx("read response body"))?;
        Ok(buf)
    }
}

/// What the client does when the server says "go over there".
///
/// **Upstream:** `registryOptions.CheckRedirect`, a `func(req, via) error`.
/// Modelled as data instead of a closure so it can be inspected in tests and
/// carried across threads without lifetime gymnastics.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RedirectPolicy {
    /// Follow up to ten hops, then fail. Go's `http.Client` default.
    #[default]
    Follow,

    /// Follow only while the redirect stays on `host`
    /// (compared by [`Url::hostname`], so the port doesn't matter). At the first
    /// hop to a **different** host, stop and hand the 3xx response itself back
    /// to the caller, `Location` header and all.
    ///
    /// **Upstream:** the `CheckRedirect` installed in `blobDownload.run`, which
    /// returns `http.ErrUseLastResponse` on a cross-host hop and
    /// `errMaxRedirectsExceeded` past ten.
    ///
    /// This is how ollama gets the **direct URL**: the registry answers the blob
    /// GET with a 307 to a presigned CDN link, and we want that link, not its
    /// contents -- because the sixteen part-downloads all fire at the CDN in
    /// parallel afterwards. Follow it blindly and you have downloaded the blob
    /// once, serially, before the real download even start.
    SameHostThenStop { host: String },
}

/// Everything the transport needs to make one request.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub url: Url,
    /// Header name/value pairs. May contain `Authorization`, `Range`, `Accept`,
    /// `Content-Type`, `Content-Length`, `Content-Range`, `X-Redirect-Uploads`.
    pub headers: Vec<(String, String)>,
    pub body: Body,
    pub redirect: RedirectPolicy,
    /// If no byte arrives for this long, the transport must abort with
    /// [`RegistryError::PartStalled`].
    ///
    /// **Deliberate divergence.** Upstream runs a watchdog goroutine per chunk
    /// (`downloadChunk`'s second `g.Go`) that ticks every second and compares
    /// against `part.lastUpdated`. Rust has no way to interrupt a blocking
    /// `Read` from another thread, so the deadline is pushed down to the
    /// transport, which *can* set a socket read timeout. The retry **semantics**
    /// are unchanged and stay up here: a stall does not spend a retry
    /// (see [`BlobDownload::download_part_with_retry`]). What would make this
    /// wrong: a transport that ignores this field, in which case a dead
    /// connection hangs the pull forever instead of retrying.
    pub stall_timeout: Option<Duration>,
}

impl Request {
    /// A GET with no headers and no body.
    pub fn get(url: Url) -> Self {
        Request {
            method: Method::Get,
            url,
            headers: Vec::new(),
            body: Body::Empty,
            redirect: RedirectPolicy::Follow,
            stall_timeout: None,
        }
    }

    /// Set (or replace) a header, case-insensitively. **Upstream:**
    /// `req.Header.Set`.
    pub fn set_header(&mut self, name: &str, value: impl Into<String>) {
        let value = value.into();
        if let Some(slot) = self
            .headers
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
        {
            slot.1 = value;
        } else {
            self.headers.push((name.to_string(), value));
        }
    }

    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// The one network touch-point.
///
/// Implement this to teach the registry client a way to move bytes -- real HTTP,
/// a local mirror, a corporate artifact cache, a test double. Everything above
/// it (part planning, sidecars, resume, retry, auth, manifests) never knows or
/// cares which one it got.
///
/// # Contract
///
/// * A non-2xx status is **not** an error: return the [`Response`] and let
///   [`make_request_with_retry`] decide, because it needs the 401's
///   `WWW-Authenticate` and the 404's meaning. Only a transport-level failure
///   (DNS, TLS, connection reset) is `Err`.
/// * [`Request::redirect`] must be honoured exactly -- see
///   [`RedirectPolicy::SameHostThenStop`] for why the download breaks otherwise.
/// * [`Request::stall_timeout`], when set, must abort a silent connection with
///   [`RegistryError::PartStalled`].
/// * `Send + Sync`, because up to [`NUM_DOWNLOAD_PARTS`] threads share one.
pub trait Transport: Send + Sync {
    /// Do the request.
    fn execute(&self, request: Request) -> Result<Response>;
}

/// Signs registry auth challenges with the user's ed25519 key.
///
/// **Upstream:** `auth/auth.go` -- `GetPublicKey` and `Sign`, both of which read
/// `~/.ollama/id_ed25519` (an **OpenSSH-format** private key) via
/// `golang.org/x/crypto/ssh`.
///
/// No implementation ships here: that needs an ed25519 crate plus an OpenSSH
/// private-key parser, neither of which this crate is allowed to add. The trait
/// exists so that everything *around* signing -- the challenge parse, the realm
/// host check, the exact bytes that get signed, the `pubkey:signature` framing --
/// is written, tested and correct the day the dependency lands.
pub trait Signer: Send + Sync {
    /// The public key's **base64 blob only** -- no `ssh-ed25519 ` prefix, no
    /// trailing comment.
    ///
    /// Upstream gets here by `ssh.MarshalAuthorizedKey(pub)` (which produces
    /// `ssh-ed25519 AAAAC3... comment\n`) and then
    /// `bytes.Split(publicKey, []byte(" "))[1]`. So: the middle field, and
    /// nothing else. Send the whole authorized-key line instead and the registry
    /// cannot match you to an account.
    fn public_key_blob(&self) -> Result<String>;

    /// The **raw** ed25519 signature over `data` -- 64 bytes.
    ///
    /// Upstream uses `privateKey.Sign(rand.Reader, bts)` and then takes
    /// `signedData.Blob`. For ed25519, `ssh.Signature.Blob` is the bare
    /// signature, **not** the SSH wire-format wrapper. Return the wrapper and
    /// the registry rejects every request.
    fn sign(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Streaming md5, for the push path only.
///
/// **Upstream:** `crypto/md5` in `upload.go` -- each part is hashed as it is
/// sent, and the commit request carries
/// `etag = fmt.Sprintf("%x-%d", md5(concat of part sums), len(parts))`.
///
/// md5 here is a **checksum, not a security claim** -- it is the registry's
/// upload-integrity token, and the content address is the sha256 digest, which
/// is what actually guarantees the blob. No implementation ships (the `md-5`
/// crate is not a dependency we have); [`BlobUpload::commit_etag`] takes one of
/// these so the framing is testable with a stub.
pub trait Md5Hasher {
    /// Feed more bytes.
    fn update(&mut self, chunk: &[u8]);
    /// Finish and reset, returning the 16-byte digest.
    fn finalize_and_reset(&mut self) -> [u8; 16];
}

// ---------------------------------------------------------------------------
// registryOptions
// ---------------------------------------------------------------------------

/// Per-registry credentials and quirks. **Upstream:** `type registryOptions`.
///
/// Mutable on purpose: a 401 makes [`make_request_with_retry`] fetch a bearer
/// token and write it into [`RegistryOptions::token`], so the next request on
/// the same options is already authenticated. Upstream does the same
/// (`regOpts.Token = token`), which is also why upstream shallow-copies the
/// struct before installing a per-download `CheckRedirect`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegistryOptions {
    /// Allow plain `http://`. **Upstream:** `Insecure`.
    pub insecure: bool,
    /// Basic-auth user. **Upstream:** `Username`.
    pub username: String,
    /// Basic-auth password. **Upstream:** `Password`.
    pub password: String,
    /// Bearer token, filled in after a successful challenge. **Upstream:**
    /// `Token`.
    pub token: String,
    /// **Upstream:** `CheckRedirect`.
    pub redirect: RedirectPolicy,
}

impl RegistryOptions {
    /// Refuse a plaintext registry unless the caller asked for it.
    ///
    /// **Upstream:** `if n.ProtocolScheme == "http" && !regOpts.Insecure {
    /// return errInsecureProtocol }`, at the top of both `PullModel` and
    /// `PushModel`. Checked against the **name's** scheme, before any request --
    /// so a typo'd `http://` registry fails loudly instead of quietly shipping
    /// your credentials in the clear.
    pub fn check_scheme(&self, name: &Name) -> Result<()> {
        if name.protocol_scheme == "http" && !self.insecure {
            return Err(RegistryError::InsecureProtocol);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Registry URLs
// ---------------------------------------------------------------------------

/// `{scheme}://{host}/v2/{namespace}/{model}/manifests/{tag}`.
///
/// **Upstream:** `n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(),
/// "manifests", n.Tag)`, in both `pullModelManifest` and `PushModel`.
pub fn manifest_url(name: &Name) -> Result<Url> {
    Ok(Url::parse(&name.base_url())?.join_path([
        "v2",
        name.display_namespace_model().as_str(),
        "manifests",
        name.tag.as_str(),
    ]))
}

/// `{scheme}://{host}/v2/{namespace}/{model}/blobs/{digest}`.
///
/// **Upstream:** the `JoinPath("v2", ..., "blobs", opts.digest)` in
/// `downloadBlob` and `uploadBlob`. The digest keeps its `sha256:` colon in the
/// path -- see [`Url::join_path`].
pub fn blob_url(name: &Name, digest: &Digest) -> Result<Url> {
    Ok(Url::parse(&name.base_url())?.join_path([
        "v2",
        name.display_namespace_model().as_str(),
        "blobs",
        digest.as_str(),
    ]))
}

/// `{scheme}://{host}/v2/{namespace}/{model}/blobs/uploads/` -- trailing slash
/// and all.
///
/// **Upstream:** `JoinPath("v2", n.DisplayNamespaceModel(), "blobs/uploads/")`.
/// The trailing slash is part of the OCI distribution spec's
/// "start an upload session" endpoint; drop it and the registry 404s.
pub fn uploads_url(name: &Name) -> Result<Url> {
    Ok(Url::parse(&name.base_url())?.join_path_dir([
        "v2",
        name.display_namespace_model().as_str(),
        "blobs",
        "uploads",
    ]))
}

// ---------------------------------------------------------------------------
// The registry challenge (WWW-Authenticate) and the signed token exchange
// ---------------------------------------------------------------------------

/// A parsed `WWW-Authenticate: Bearer realm=..,service=..,scope=..` header.
/// **Upstream:** `type registryChallenge` in `server/auth.go`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegistryChallenge {
    /// Where to go get a token.
    pub realm: String,
    /// Which service the token is for.
    pub service: String,
    /// Space-separated scopes, e.g. `repo:foo:pull repo:bar:push`.
    pub scope: String,
}

/// Pull `key="value"` out of an auth header. **Upstream:** `getValue`.
///
/// Quirky on purpose, and ported quirk-for-quirk: a closing quote only ends the
/// value if the character right after it is a comma (or the string ends). That
/// is how upstream tolerates a quote *inside* a value, e.g.
/// `scope="repo:a"b:pull"`. Being clever here -- "surely they meant a normal
/// quoted string" -- would parse some real registry headers differently from
/// ollama, and ollama is the oracle.
pub fn get_value(header: &str, key: &str) -> String {
    let needle = format!("{key}=");
    let Some(start_idx) = header.find(&needle) else {
        return String::new();
    };
    // Skip past `key="` -- key length, the `=`, and the opening quote.
    let start = start_idx + key.len() + 2;
    let b = header.as_bytes();
    if start > b.len() {
        return String::new();
    }
    let mut end = start;
    while end < b.len() {
        if b[end] == b'"' {
            if end + 1 < b.len() && b[end + 1] != b',' {
                end += 1;
                continue;
            }
            break;
        }
        end += 1;
    }
    header[start..end.min(b.len())].to_string()
}

/// Parse a `WWW-Authenticate` header. **Upstream:** `parseRegistryChallenge`.
///
/// An empty or unrecognised header yields an all-empty challenge rather than an
/// error -- same as upstream, and the realm-host check downstream is what turns
/// that into a refusal.
pub fn parse_registry_challenge(auth_str: &str) -> RegistryChallenge {
    let auth_str = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
    RegistryChallenge {
        realm: get_value(auth_str, "realm"),
        service: get_value(auth_str, "service"),
        scope: get_value(auth_str, "scope"),
    }
}

/// How many random bytes go into an auth nonce. **Upstream:**
/// `auth.NewNonce(rand.Reader, 16)` in `registryChallenge.URL()`.
pub const NONCE_LEN: usize = 16;

impl RegistryChallenge {
    /// Build the token-endpoint URL: realm + `service`, one `scope` per
    /// space-separated scope, `ts`, and a fresh `nonce`.
    ///
    /// **Upstream:** `(registryChallenge).URL()`.
    ///
    /// `ts` and `nonce` are replay defence -- the registry checks the timestamp
    /// is fresh and that the nonce has not been seen. So the nonce **must** be
    /// unique per call (asserted by
    /// `challenge_url_gives_a_fresh_nonce_each_call`) and must come from a
    /// CSPRNG; see [`SystemAmbient`]'s caveat about the one shipped here.
    ///
    /// Note `strings.Split(r.Scope, " ")` on an empty scope yields `[""]` in Go,
    /// so an empty challenge still sends one empty `scope=`. Ported as-is.
    pub fn url(&self, ambient: &dyn Ambient) -> Result<Url> {
        let redirect_url = Url::parse(&self.realm)?;

        let ts = ambient.unix_secs().to_string();
        let mut nonce_bytes = [0u8; NONCE_LEN];
        ambient.random_bytes(&mut nonce_bytes);
        let nonce = base64_raw_url(&nonce_bytes);

        let scopes: Vec<&str> = self.scope.split(' ').collect();
        let mut pairs: Vec<(&str, &str)> = vec![("service", self.service.as_str())];
        for s in &scopes {
            pairs.push(("scope", s));
        }
        pairs.push(("ts", ts.as_str()));
        pairs.push(("nonce", nonce.as_str()));

        // Keep any query the realm already carried; Go's `redirectURL.Query()`
        // starts from the existing values, so a realm like
        // `https://auth/token?foo=1` keeps `foo`.
        let mut existing: Vec<(String, String)> = Vec::new();
        for kv in redirect_url.query.split('&').filter(|s| !s.is_empty()) {
            if let Some((k, v)) = kv.split_once('=') {
                existing.push((query_unescape(k), query_unescape(v)));
            }
        }
        let mut all: Vec<(&str, &str)> = existing
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        all.extend(pairs);

        Ok(redirect_url.with_query(&all))
    }
}

/// The exact bytes ollama signs when answering a challenge.
///
/// **Upstream:** `getAuthorizationToken`:
///
/// ```text
/// sha256sum := sha256.Sum256(nil)
/// data := fmt.Sprintf("%s,%s,%s", http.MethodGet, redirectURL.String(),
///          base64.StdEncoding.EncodeToString([]byte(hex.EncodeToString(sha256sum[:]))))
/// ```
///
/// Read that third field carefully hor, it is easy to get wrong and impossible
/// to debug from the outside: it is the sha256 of an **empty body**, rendered as
/// **64 lowercase hex characters**, and then those 64 *ASCII characters* are
/// base64'd. Not the base64 of the 32 raw hash bytes. Getting it wrong produces
/// a signature the registry silently rejects with a 401, and you will chase the
/// key file for an hour before suspecting this line.
///
/// The body hash is always the empty one because the token request is a GET.
pub fn authorization_payload(method: Method, url: &Url) -> Vec<u8> {
    let mut h = Sha256::new();
    let sum = h.finalize_and_reset(); // sha256 of nothing
    let hex: String = sum.iter().map(|b| format!("{b:02x}")).collect();
    format!("{},{},{}", method.as_str(), url, base64_std(hex.as_bytes())).into_bytes()
}

/// Frame a signature the way the registry expects: `<pubkey-blob>:<base64 sig>`.
///
/// **Upstream:** the last line of `auth.Sign` --
/// `fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob))`.
/// `parts[1]` is the middle field of the authorized-key line, i.e. exactly what
/// [`Signer::public_key_blob`] promises.
pub fn sign_authorization(signer: &dyn Signer, data: &[u8]) -> Result<String> {
    let pubkey = signer.public_key_blob()?;
    let sig = signer.sign(data)?;
    Ok(format!("{}:{}", pubkey.trim(), base64_std(&sig)))
}

/// Answer a `401` by fetching a bearer token from the challenge's realm.
///
/// **Upstream:** `getAuthorizationToken(ctx, challenge, originalHost)`.
///
/// The realm-host check is the security-critical bit and comes **before** any
/// signing: if the realm's host (with port -- see [`Url::host`]) is not exactly
/// the host we were already talking to, we refuse. Otherwise any registry could
/// answer a 401 with `realm="https://evil.example/"` and collect a signed
/// request from us. Upstream's `TestGetAuthorizationTokenRejectsCrossDomain`
/// table is ported verbatim as
/// `authorization_token_refuses_a_cross_domain_realm`.
pub fn get_authorization_token(
    transport: &dyn Transport,
    ambient: &dyn Ambient,
    signer: Option<&dyn Signer>,
    challenge: &RegistryChallenge,
    original_host: &str,
) -> Result<String> {
    let redirect_url = challenge.url(ambient)?;

    if redirect_url.host() != original_host {
        return Err(RegistryError::RealmHostMismatch {
            realm: redirect_url.host().to_string(),
            original: original_host.to_string(),
        });
    }

    let signer = signer.ok_or(RegistryError::NoSigner)?;
    let data = authorization_payload(Method::Get, &redirect_url);
    let signature = sign_authorization(signer, &data)?;

    let mut req = Request::get(redirect_url);
    req.set_header("Authorization", signature);

    // Deliberately a bare request, not make_request: upstream passes
    // `&registryOptions{}` here, so no token and no basic auth ride along -- the
    // signature IS the credential.
    let mut resp = transport.execute(req)?;
    let body = resp.read_to_end()?;

    if resp.status >= 400 {
        return Err(if body.is_empty() {
            RegistryError::Status {
                code: resp.status,
                body: String::new(),
            }
        } else {
            RegistryError::Status {
                code: resp.status,
                body: String::from_utf8_lossy(&body).into_owned(),
            }
        });
    }

    let token: TokenResponse =
        serde_json::from_slice(&body).map_err(json_ctx("parse token response"))?;
    Ok(token.token)
}

/// The token endpoint's reply. **Upstream:** `api.TokenResponse`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenResponse {
    #[serde(default)]
    pub token: String,
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

/// What we tell the registry we are.
///
/// **Upstream:** `fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version,
/// runtime.GOARCH, runtime.GOOS, runtime.Version())`.
///
/// **Deliberate divergence:** we say `kopitiam-ollama`, not `ollama`. Pretending
/// to be ollama would be dishonest to the registry operator and would make any
/// server-side rate-limit or bug report point at the wrong software. If a
/// registry ever gates on the ollama UA, that is a decision for the maintainer
/// to make out loud, not something to smuggle in here.
pub fn user_agent() -> String {
    format!(
        "kopitiam-ollama/{} ({} {}) Rust",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

/// The registry client: a transport, a clock, and optionally a key.
///
/// Cheap to make and cheap to copy around -- it holds only borrows, so one
/// `Registry` can be shared by every part-download thread.
pub struct Registry<'a> {
    /// Where bytes come from and go to.
    pub transport: &'a dyn Transport,
    /// Clock, sleep and dice.
    pub ambient: &'a dyn Ambient,
    /// The ed25519 key, when we have one. `None` means "public pulls only" --
    /// a challenge then fails with [`RegistryError::NoSigner`] rather than
    /// silently retrying forever.
    pub signer: Option<&'a dyn Signer>,
}

impl<'a> Registry<'a> {
    /// Wire one up.
    pub fn new(
        transport: &'a dyn Transport,
        ambient: &'a dyn Ambient,
        signer: Option<&'a dyn Signer>,
    ) -> Self {
        Registry {
            transport,
            ambient,
            signer,
        }
    }

    /// One request, credentials attached, **no** status interpretation.
    ///
    /// **Upstream:** `makeRequest`. Returns the response whatever the status is
    /// -- 401 and 404 are meaningful to the caller, so swallowing them here
    /// would throw away the information [`Registry::make_request_with_retry`]
    /// needs.
    ///
    /// Auth precedence is upstream's: a bearer token wins over basic auth, and
    /// basic auth only applies when **both** username and password are set.
    pub fn make_request(
        &self,
        method: Method,
        url: &Url,
        headers: &[(String, String)],
        body: Body,
        opts: &RegistryOptions,
    ) -> Result<Response> {
        // Upstream: `if requestURL.Scheme != "http" && regOpts.Insecure {
        // requestURL.Scheme = "http" }`. Reads backwards -- it *downgrades*
        // https to http when insecure is set -- but that is what the Go does,
        // and ollama is the oracle. It exists so `--insecure` against a local
        // registry works even when the name was written with the default https
        // scheme.
        let mut url = url.clone();
        if url.scheme != "http" && opts.insecure {
            url.scheme = "http".to_string();
        }

        let mut req = Request {
            method,
            url,
            // Upstream does `req.Header = headers`, i.e. wholesale replacement,
            // then Set()s on top. Same here.
            headers: headers.to_vec(),
            body,
            redirect: opts.redirect.clone(),
            stall_timeout: None,
        };

        if !opts.token.is_empty() {
            req.set_header("Authorization", format!("Bearer {}", opts.token));
        } else if !opts.username.is_empty() && !opts.password.is_empty() {
            let raw = format!("{}:{}", opts.username, opts.password);
            req.set_header("Authorization", format!("Basic {}", base64_std(raw.as_bytes())));
        }

        req.set_header("User-Agent", user_agent());

        // Upstream parses a caller-supplied Content-Length header back into
        // req.ContentLength. We skip that entirely: `Body` already knows its own
        // length exactly (`Body::len`), so the transport can set the header from
        // the truth instead of from a string somebody typed. A caller-set
        // Content-Length header still rides along untouched, because `uploadPart`
        // sets one and the registry expects to see it.
        self.transport.execute(req)
    }

    /// One request, with the 401-fetch-token-and-replay dance and status
    /// interpretation.
    ///
    /// **Upstream:** `makeRequestWithRetry`. The `for range 2` is the whole
    /// design: **exactly one** re-attempt after a challenge, then give up with
    /// [`RegistryError::Unauthorized`]. Not a general retry loop -- a registry
    /// that keeps answering 401 is refusing you, and hammering it won't help.
    ///
    /// Status handling, verbatim from upstream:
    ///
    /// * **401** -- parse `WWW-Authenticate`, sign, fetch a token, store it in
    ///   `opts.token`, rewind the body, go round again.
    /// * **404** -- [`RegistryError::NotExist`], upstream's `os.ErrNotExist`.
    ///   `upload_blob` branches on this to mean "not uploaded yet".
    /// * **>= 400** -- [`RegistryError::Status`] with the body attached.
    /// * anything else -- hand the response back.
    ///
    /// `body` is taken by value and cloned per attempt; that clone **is** the
    /// port of upstream's `body.Seek(0, io.SeekStart)`.
    pub fn make_request_with_retry(
        &self,
        method: Method,
        url: &Url,
        headers: &[(String, String)],
        body: Body,
        opts: &mut RegistryOptions,
    ) -> Result<Response> {
        for _ in 0..2 {
            let mut resp = self.make_request(method, url, headers, body.clone(), opts)?;

            match resp.status {
                401 => {
                    let challenge =
                        parse_registry_challenge(resp.header("WWW-Authenticate").unwrap_or(""));
                    let token = get_authorization_token(
                        self.transport,
                        self.ambient,
                        self.signer,
                        &challenge,
                        url.host(),
                    )?;
                    opts.token = token;
                    // Body rewind is free -- see the doc comment.
                }
                404 => return Err(RegistryError::NotExist),
                s if s >= 400 => {
                    let body = resp.read_to_end()?;
                    return Err(RegistryError::Status {
                        code: s,
                        body: String::from_utf8_lossy(&body).into_owned(),
                    });
                }
                _ => return Ok(resp),
            }
        }

        Err(RegistryError::Unauthorized)
    }

    /// Fetch a model's manifest. Returns the parsed form **and the raw bytes**.
    ///
    /// **Upstream:** `pullModelManifest`, which likewise returns
    /// `(*manifest.Manifest, []byte, error)`.
    ///
    /// The raw bytes are not a convenience hor -- they are required. Upstream's
    /// `pullWithTransfer` writes the *original* JSON to disk rather than
    /// re-serialising the struct, because a manifest can carry tensor-metadata
    /// fields this type does not model, and a round-trip through
    /// [`Manifest`] would silently drop them. Same reason applies here: write
    /// `raw`, never `serde_json::to_vec(&manifest)`.
    pub fn pull_model_manifest(
        &self,
        name: &Name,
        opts: &mut RegistryOptions,
    ) -> Result<(Manifest, Vec<u8>)> {
        let url = manifest_url(name)?;
        let headers = vec![(
            "Accept".to_string(),
            crate::manifest::MEDIA_TYPE_MANIFEST.to_string(),
        )];
        let mut resp =
            self.make_request_with_retry(Method::Get, &url, &headers, Body::Empty, opts)?;
        let data = resp.read_to_end()?;
        let mf: Manifest =
            serde_json::from_slice(&data).map_err(json_ctx("parse pulled manifest"))?;
        Ok((mf, data))
    }

    /// PUT a manifest up. **Upstream:** the tail of `PushModel`.
    ///
    /// Takes the raw JSON for the same reason [`Registry::pull_model_manifest`]
    /// returns it: what goes up must be byte-identical to what the digest was
    /// computed over.
    pub fn push_model_manifest(
        &self,
        name: &Name,
        manifest_json: Vec<u8>,
        opts: &mut RegistryOptions,
    ) -> Result<()> {
        let url = manifest_url(name)?;
        let headers = vec![(
            "Content-Type".to_string(),
            crate::manifest::MEDIA_TYPE_MANIFEST.to_string(),
        )];
        self.make_request_with_retry(Method::Put, &url, &headers, Body::Bytes(manifest_json), opts)?;
        Ok(())
    }
}

/// Re-hash a blob on disk and check it still matches the digest that names it.
///
/// **Upstream:** `verifyBlob(digest)` in `server/images.go`, which returns
/// `errDigestMismatch` -- here [`ManifestError::DigestMismatch`], since the
/// store owns that error.
///
/// Worth being clear about what this catches and what it doesn't: it catches a
/// truncated or corrupted download **after** the fact. It is not a substitute
/// for the resume machinery -- a part that quietly restarts from zero and
/// overwrites good bytes with the same good bytes still verifies fine. Read the
/// sidecar tests for that property.
pub fn verify_blob(path: &Path, digest: &Digest, hasher: &mut dyn Sha256Hasher) -> Result<()> {
    let f = File::open(path).map_err(io_ctx(format!("open blob {}", to_slash(path))))?;
    let (got, _size) = crate::manifest::sha256_of_reader(f, hasher)
        .map_err(io_ctx(format!("hash blob {}", to_slash(path))))?;
    if got != *digest {
        return Err(ManifestError::DigestMismatch {
            want: digest.to_string(),
            got: got.to_string(),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Constants -- every number here has an upstream address
// ---------------------------------------------------------------------------

/// How many part-downloads run at once, and the divisor used to pick a part
/// size. **Upstream:** `numDownloadParts = 16` in `server/download.go`.
pub const NUM_DOWNLOAD_PARTS: usize = 16;

/// **Upstream:** `minDownloadPartSize int64 = 100 * format.MegaByte`.
///
/// `format.MegaByte` upstream is 1000-based (`crate::format::MEGABYTE` matches),
/// so this is 100_000_000 bytes, **not** 100 MiB. Getting the base wrong shifts
/// every part boundary and makes our sidecars incompatible with ollama's own.
pub const MIN_DOWNLOAD_PART_SIZE: i64 = 100 * crate::format::MEGABYTE as i64;

/// **Upstream:** `maxDownloadPartSize int64 = 1000 * format.MegaByte`.
pub const MAX_DOWNLOAD_PART_SIZE: i64 = 1000 * crate::format::MEGABYTE as i64;

/// **Upstream:** `numUploadParts = 16` in `server/upload.go`. Same value as the
/// download's, kept separate because upstream keeps them separate -- they are
/// free to drift.
pub const NUM_UPLOAD_PARTS: usize = 16;

/// **Upstream:** `minUploadPartSize int64 = 100 * format.MegaByte`.
pub const MIN_UPLOAD_PART_SIZE: i64 = 100 * crate::format::MEGABYTE as i64;

/// **Upstream:** `maxUploadPartSize int64 = 1000 * format.MegaByte`.
pub const MAX_UPLOAD_PART_SIZE: i64 = 1000 * crate::format::MEGABYTE as i64;

/// **Upstream:** `const maxRetries = 6` in `server/download.go`.
///
/// Six attempts with `2^try` second gaps is 1+2+4+8+16 = 31 seconds of waiting
/// before giving up on a part. Stalls are exempt -- see
/// [`RegistryError::PartStalled`].
pub const MAX_RETRIES: u32 = 6;

/// **Upstream:** `var downloadStallTimeout = 30 * time.Second`.
///
/// A `var`, not a `const`, upstream -- their tests reassign it. Ours is a real
/// constant because the stall path is exercised through a fake transport
/// instead, which is faster and doesn't need a global mutable.
pub const DOWNLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(30);

/// How often progress is reported. **Upstream:** `time.NewTicker(60 *
/// time.Millisecond)` in `blobDownload.Wait` and `blobUpload.Wait`.
pub const PROGRESS_TICK: Duration = Duration::from_millis(60);

/// Ceiling on the direct-URL backoff. **Upstream:** `newBackoff(10 *
/// time.Second)` in `blobDownload.run`.
pub const DIRECT_URL_MAX_BACKOFF: Duration = Duration::from_secs(10);

/// The whole direct-URL resolution gives up after this. **Upstream:**
/// `context.WithTimeout(ctx, 30*time.Second)` wrapping that loop.
pub const DIRECT_URL_TIMEOUT: Duration = Duration::from_secs(30);

/// Streaming buffer size. Not an upstream constant (`io.Copy` picks 32 KiB);
/// 64 KiB matches what `crate::manifest::sha256_of_reader` already uses, so a
/// pull and a verify move bytes the same way.
const COPY_BUF: usize = 64 * 1024;

/// The short digest ollama shows in progress lines: `digest[7:19]`, i.e. the
/// first 12 hex characters after the `sha256:` prefix.
///
/// **Upstream:** `b.Digest[7:19]`, which appears in every progress message and
/// log line in `download.go` and `upload.go`. Twelve is enough to recognise a
/// blob by eye and short enough for a progress bar.
pub fn short_digest(digest: &Digest) -> &str {
    &digest.hex()[..12]
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

/// One progress tick. **Upstream:** `api.ProgressResponse`, the subset
/// `download.go` / `upload.go` fill in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progress {
    /// `pulling 3f8db5c4a3` / `pushing 3f8db5c4a3` / `success` ...
    pub status: String,
    /// Full `sha256:...` digest, or empty for a status-only tick.
    pub digest: String,
    /// Total bytes for this blob. `0` when not known yet.
    pub total: i64,
    /// Bytes done so far. **May go down**: a failed chunk rolls its progress
    /// back, exactly like upstream's `b.Completed.Add(-n)`. A progress bar that
    /// assumes monotonic will look silly, so don't assume.
    pub completed: i64,
}

// ---------------------------------------------------------------------------
// Positional writes -- the io.NewOffsetWriter port
// ---------------------------------------------------------------------------

/// Write `buf` at absolute `offset` without moving anybody's file cursor.
///
/// **Upstream:** `io.NewOffsetWriter(file, part.StartsAt())`.
///
/// This is the thing that lets sixteen threads share **one** file handle and
/// still write to sixteen disjoint regions. Platform knowledge worth keeping,
/// because the two APIs are not spelled the same:
///
/// * **Unix** -- `pwrite(2)` via `std::os::unix::fs::FileExt::write_at`. Atomic
///   w.r.t. the file offset, which it does not touch at all.
/// * **Windows** -- `std::os::windows::fs::FileExt::seek_write`, which issues
///   `WriteFile` with an `OVERLAPPED` carrying the offset. The offset in the
///   `OVERLAPPED` wins, so concurrent calls land where they are told. It *does*
///   then update the shared file pointer as a side effect -- which is exactly
///   why every call here passes an explicit offset and nothing in this module
///   ever relies on the cursor.
///
/// Neither call is guaranteed to write everything in one go, so we loop.
#[cfg(unix)]
fn write_all_at(f: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut buf = buf;
    let mut offset = offset;
    while !buf.is_empty() {
        match f.write_at(buf, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(f: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut buf = buf;
    let mut offset = offset;
    while !buf.is_empty() {
        match f.seek_write(buf, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Classify a write failure. A full disk aborts the whole download instead of
/// retrying -- **upstream:** `errors.Is(err, syscall.ENOSPC)`.
///
/// The raw codes are checked by hand because `io::ErrorKind::StorageFull` is
/// still unstable: 28 is `ENOSPC` on every Unix; on Windows 39 is
/// `ERROR_HANDLE_DISK_FULL` and 112 is `ERROR_DISK_FULL`.
fn classify_write_error(e: io::Error, context: &str) -> RegistryError {
    let full = match e.raw_os_error() {
        #[cfg(unix)]
        Some(28) => true,
        #[cfg(windows)]
        Some(39) | Some(112) => true,
        _ => false,
    };
    if full {
        RegistryError::OutOfSpace
    } else {
        RegistryError::Io {
            context: context.to_string(),
            source: e,
        }
    }
}

// ---------------------------------------------------------------------------
// Part planning
// ---------------------------------------------------------------------------

/// One planned part: a half-open byte range `[offset, offset + size)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartPlan {
    /// Part number, `0..n`. Also its sidecar's filename suffix.
    pub n: usize,
    /// First byte of the range, absolute in the blob.
    pub offset: i64,
    /// Length of the range in bytes.
    pub size: i64,
}

/// Cut a blob of `total` bytes into parts.
///
/// **Upstream:** the identical arithmetic in `blobDownload.Prepare` and
/// `blobUpload.Prepare` -- aim for [`NUM_DOWNLOAD_PARTS`] parts, then clamp the
/// part size into `[min, max]`:
///
/// ```text
/// size := total / numParts
/// switch {
/// case size < minPartSize: size = minPartSize
/// case size > maxPartSize: size = maxPartSize
/// }
/// ```
///
/// So the part **count** is only 16 in the middle band. Below 1.6 GB the parts
/// are 100 MB each and there are fewer than 16 (a 250 MB blob gets three parts:
/// 100, 100, 50). Above 16 GB they are 1000 MB each and there are more than 16
/// (a 20 GB blob gets 20). That last case matters: it is why the worker pool is
/// capped at 16 *threads* rather than "one thread per part".
///
/// The final part is truncated to whatever is left, so the parts always tile
/// `[0, total)` exactly -- no gap, no overlap. `total <= 0` gives no parts at
/// all, which is upstream's behaviour when `Content-Length` was missing.
pub fn plan_parts(total: i64, num_parts: usize, min_size: i64, max_size: i64) -> Vec<PartPlan> {
    let mut parts = Vec::new();
    if total <= 0 {
        return parts;
    }
    let mut size = total / num_parts as i64;
    if size < min_size {
        size = min_size;
    } else if size > max_size {
        size = max_size;
    }
    let mut offset: i64 = 0;
    while offset < total {
        if offset + size > total {
            size = total - offset;
        }
        parts.push(PartPlan {
            n: parts.len(),
            offset,
            size,
        });
        offset += size;
    }
    parts
}

// ---------------------------------------------------------------------------
// The sidecar
// ---------------------------------------------------------------------------

/// The on-disk shape of a part's progress -- **this is the resume file format**.
///
/// **Upstream:** `jsonBlobDownloadPart` in `server/download.go`, written by
/// `json.NewEncoder(partFile).Encode(part)`.
///
/// The field names are capitalised because Go marshals exported field names
/// as-is when there is no struct tag, and there is none here. They are `N`,
/// `Offset`, `Size`, `Completed` -- **not** `n`/`offset`/`size`/`completed`.
/// This is the one place where being idiomatic would break interoperability:
/// rename them and ollama can no longer resume a download KOPITIAM started, and
/// vice versa. `sidecar_json_uses_capitalised_go_field_names` pins it.
///
/// `Encode` also appends a `\n`, so we do too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartSidecar {
    /// Part number.
    #[serde(rename = "N")]
    pub n: usize,
    /// First byte of this part, absolute in the blob. **Never changes** once
    /// written -- resume works by adding `Completed` to it, so a mutated
    /// `Offset` silently corrupts the blob.
    #[serde(rename = "Offset")]
    pub offset: i64,
    /// Length of this part in bytes. Also never changes.
    #[serde(rename = "Size")]
    pub size: i64,
    /// Bytes of this part already on disk. **This is the resume counter.**
    /// `0 <= Completed <= Size`.
    #[serde(rename = "Completed")]
    pub completed: i64,
}

/// The in-flight version of a part: same numbers, but `completed` is atomic
/// because several threads read it while one writes it.
#[derive(Debug)]
pub struct PartState {
    /// Part number.
    pub n: usize,
    /// First byte, absolute in the blob.
    pub offset: i64,
    /// Length in bytes.
    pub size: i64,
    /// Bytes done. Mirrors [`PartSidecar::completed`] and is flushed to it at
    /// every chunk boundary.
    pub completed: AtomicI64,
}

impl PartState {
    fn from_sidecar(s: PartSidecar) -> Self {
        PartState {
            n: s.n,
            offset: s.offset,
            size: s.size,
            completed: AtomicI64::new(s.completed),
        }
    }

    fn sidecar(&self) -> PartSidecar {
        PartSidecar {
            n: self.n,
            offset: self.offset,
            size: self.size,
            completed: self.completed.load(Ordering::SeqCst),
        }
    }

    /// The next byte we need. **Upstream:** `StartsAt() = p.Offset +
    /// p.Completed.Load()`.
    ///
    /// **This single line is the resume.** A part that is half done starts at
    /// its offset *plus what it already has*, so the `Range` header asks only
    /// for the remainder. If this ever returns `p.Offset`, a 20 GB pull that
    /// dropped at 19 GB starts again from zero.
    pub fn starts_at(&self) -> i64 {
        self.offset + self.completed.load(Ordering::SeqCst)
    }

    /// One past the last byte of this part. **Upstream:** `StopsAt() = p.Offset
    /// + p.Size`. Note the `Range` header is inclusive, so it carries
    /// `stops_at() - 1`.
    pub fn stops_at(&self) -> i64 {
        self.offset + self.size
    }

    /// Bytes still wanted.
    pub fn remaining(&self) -> i64 {
        self.size - self.completed.load(Ordering::SeqCst)
    }

    /// Nothing left to fetch?
    pub fn is_complete(&self) -> bool {
        self.remaining() <= 0
    }
}

/// Where the half-finished blob lives while it downloads: `<blob>-partial`.
///
/// **Upstream:** `b.Name + "-partial"`. Kept next to the final blob and renamed
/// over it at the end, so the rename is same-filesystem and therefore atomic --
/// a blob under `blobs/` is never half a blob.
pub fn partial_file_path(blob_path: &Path) -> PathBuf {
    let mut s = blob_path.as_os_str().to_os_string();
    s.push("-partial");
    PathBuf::from(s)
}

/// Where part `n`'s sidecar lives: `<blob>-partial-<n>`.
///
/// **Upstream:** `strings.Join([]string{p.blobDownload.Name, "partial",
/// strconv.Itoa(p.N)}, "-")`.
///
/// Same stem as the partial blob on purpose -- `Prepare` globs
/// `<blob>-partial-*` to find them, and the cleanup in `run` removes
/// `<partial>-<i>` for every `i`. Change one spelling and resume stops finding
/// its own state.
pub fn part_file_path(blob_path: &Path, n: usize) -> PathBuf {
    let mut s = blob_path.as_os_str().to_os_string();
    s.push(format!("-partial-{n}"));
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Backoff
// ---------------------------------------------------------------------------

/// The n-squared, jittered backoff ollama uses while hunting for the direct URL.
///
/// **Upstream:** `newBackoff(maxBackoff)` in `server/download.go`, comment and
/// all:
///
/// ```text
/// // n^2 backoff timer is a little smoother than the common choice of 2^n.
/// d := min(time.Duration(n*n)*10*time.Millisecond, maxBackoff)
/// // Randomize the delay between 0.5-1.5 x msec, in order
/// // to prevent accidental "thundering herd" problems.
/// d = time.Duration(float64(d) * (rand.Float64() + 0.5))
/// ```
///
/// So attempt 1 waits ~10 ms, attempt 2 ~40 ms, attempt 3 ~90 ms, ... capped at
/// [`DIRECT_URL_MAX_BACKOFF`], each smeared by a factor in `[0.5, 1.5)`. The
/// jitter is the point: without it, every client that lost the same registry at
/// the same moment comes back at the same moment.
///
/// Deliberately **not** the same curve as the per-part retry, which is `2^try`
/// seconds -- upstream uses two different schedules for two different jobs, and
/// unifying them would be us inventing.
#[derive(Debug)]
pub struct Backoff {
    n: u32,
    max: Duration,
}

impl Backoff {
    /// A backoff capped at `max`.
    pub fn new(max: Duration) -> Self {
        Backoff { n: 0, max }
    }

    /// The delay the *next* [`Backoff::wait`] would use, before jitter. Exposed
    /// so the schedule is assertable without a clock.
    pub fn peek_base(&self) -> Duration {
        let n = (self.n + 1) as u64;
        Duration::from_millis(n * n * 10).min(self.max)
    }

    /// Advance and sleep. Returns [`RegistryError::Canceled`] instead of
    /// sleeping if the token is already tripped -- upstream checks `ctx.Err()`
    /// at the top for exactly this reason.
    pub fn wait(&mut self, ambient: &dyn Ambient, cancel: &CancelToken) -> Result<()> {
        cancel.check()?;
        // Upstream is `n++` *then* `d := min(n*n*10ms, max)`, so the first wait
        // uses n = 1. `peek_base` already looks one ahead, hence the order here.
        let base = self.peek_base();
        self.n += 1;
        let jitter = ambient.random_f64() + 0.5;
        ambient.sleep(base.mul_f64(jitter));
        cancel.check()
    }
}

/// The per-part retry gap: `2^try` seconds.
///
/// **Upstream:** `sleep := time.Second * time.Duration(math.Pow(2, float64(try)))`
/// -- in `download.go`'s part loop, in `upload.go`'s part loop, and again in
/// `upload.go`'s commit loop. Same schedule all three places, so it lives in one
/// function here.
pub fn retry_backoff(try_index: u32) -> Duration {
    Duration::from_secs(1u64 << try_index.min(62))
}

// ---------------------------------------------------------------------------
// The download
// ---------------------------------------------------------------------------

/// One blob being pulled: where it goes, how big it is, and the state of each
/// part.
///
/// **Upstream:** `type blobDownload struct` in `server/download.go`.
///
/// Not ported: the process-wide `blobDownloadManager sync.Map` and the
/// `references` refcount that let two concurrent `ollama pull`s of the same
/// model share one in-flight download. That is server plumbing for a
/// long-running daemon; this crate is a library, and the caller that wants that
/// behaviour can hold a map of these. Said out loud rather than silently
/// dropped, because the consequence is real: call [`download_blob`] twice
/// concurrently for the same digest and the two runs will fight over the same
/// `-partial` file.
#[derive(Debug)]
pub struct BlobDownload {
    /// Final resting place of the blob -- `<store>/blobs/sha256-<hex>`.
    /// **Upstream:** `Name`.
    pub name: PathBuf,
    /// What we are pulling. **Upstream:** `Digest`.
    pub digest: Digest,
    /// Blob length. From `Content-Length` on a fresh start, or from the sum of
    /// the sidecars' `Size` on a resume. **Upstream:** `Total`.
    pub total: i64,
    /// Bytes done across all parts. **Upstream:** `Completed atomic.Int64`.
    pub completed: Arc<AtomicI64>,
    /// One entry per part. **Upstream:** `Parts []*blobDownloadPart`.
    pub parts: Vec<Arc<PartState>>,
}

impl BlobDownload {
    /// A download that hasn't looked at the disk yet. Call [`BlobDownload::prepare`].
    pub fn new(blob_path: impl Into<PathBuf>, digest: Digest) -> Self {
        BlobDownload {
            name: blob_path.into(),
            digest,
            total: 0,
            completed: Arc::new(AtomicI64::new(0)),
            parts: Vec::new(),
        }
    }

    /// Where the in-progress bytes live.
    pub fn partial_path(&self) -> PathBuf {
        partial_file_path(&self.name)
    }

    /// Work out the part layout: **resume from sidecars if any exist, otherwise
    /// ask the server how big the blob is and cut it up fresh.**
    ///
    /// **Upstream:** `blobDownload.Prepare`.
    ///
    /// The resume branch is the important one, and note what it does *not* do:
    /// it never issues the HEAD. `total` is recomputed as the sum of the
    /// sidecars' sizes and `completed` as the sum of their completed counts, so
    /// **a resume is entirely offline until the first byte is requested**. That
    /// is deliberate upstream and we keep it -- it means a resumed pull works
    /// even if the registry has since changed how it reports `Content-Length`,
    /// and it is asserted by `preparing_over_existing_sidecars_never_touches_the_network`.
    ///
    /// What would make this wrong: trusting a sidecar whose `Size` disagrees
    /// with the blob that is actually on the server. Ollama does not check that
    /// either -- the safety net is [`verify_blob`] at the end, which catches the
    /// mismatch as a digest failure and forces a clean re-pull.
    pub fn prepare(
        &mut self,
        registry: &Registry<'_>,
        request_url: &Url,
        opts: &mut RegistryOptions,
    ) -> Result<()> {
        let mut found = self.read_existing_parts()?;
        if !found.is_empty() {
            // Upstream gets these back from filepath.Glob, which sorts
            // lexically -- so "…-partial-10" lands before "…-partial-2".
            // Nothing upstream depends on the order (each part carries its own
            // N and Offset), but ours is sorted numerically so that logs, tests
            // and any future "which part is stuck" question read sanely.
            found.sort_by_key(|p| p.n);
            for p in found {
                self.total += p.size;
                self.completed.fetch_add(p.completed, Ordering::SeqCst);
                self.parts.push(Arc::new(PartState::from_sidecar(p)));
            }
            return Ok(());
        }

        let resp =
            registry.make_request_with_retry(Method::Head, request_url, &[], Body::Empty, opts)?;
        self.total = resp.content_length();

        for plan in plan_parts(
            self.total,
            NUM_DOWNLOAD_PARTS,
            MIN_DOWNLOAD_PART_SIZE,
            MAX_DOWNLOAD_PART_SIZE,
        ) {
            let part = PartState {
                n: plan.n,
                offset: plan.offset,
                size: plan.size,
                completed: AtomicI64::new(0),
            };
            // Upstream's newPart writes the sidecar BEFORE appending to
            // b.Parts, so a crash between the two leaves a readable sidecar
            // rather than a part nobody can resume. Same order here.
            self.write_part(&part)?;
            self.parts.push(Arc::new(part));
        }

        Ok(())
    }

    /// Read back every `<blob>-partial-<n>` sidecar sitting next to the blob.
    ///
    /// **Upstream:** `filepath.Glob(b.Name + "-partial-*")` plus `readPart`.
    /// Implemented as a directory scan with a prefix match instead of a glob,
    /// because a blob path can contain glob metacharacters on some filesystems
    /// and `sha256-…` never does -- a literal prefix comparison is both cheaper
    /// and impossible to fool.
    fn read_existing_parts(&self) -> Result<Vec<PartSidecar>> {
        let Some(dir) = self.name.parent() else {
            return Ok(Vec::new());
        };
        let Some(stem) = self.name.file_name().and_then(|s| s.to_str()) else {
            return Ok(Vec::new());
        };
        let prefix = format!("{stem}-partial-");

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            // No blobs directory yet just means nothing to resume.
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_ctx(format!("read dir {}", to_slash(dir)))(e)),
        };

        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_ctx(format!("read dir {}", to_slash(dir))))?;
            let Some(fname) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Some(suffix) = fname.strip_prefix(&prefix) else {
                continue;
            };
            // Only `<prefix><digits>`; anything else is not ours.
            if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let path = entry.path();
            let bytes = fs::read(&path).map_err(io_ctx(format!("read {}", to_slash(&path))))?;
            let sidecar: PartSidecar = serde_json::from_slice(&bytes)
                .map_err(json_ctx(format!("parse part sidecar {}", to_slash(&path))))?;
            out.push(sidecar);
        }
        Ok(out)
    }

    /// Flush one part's counter to its sidecar.
    ///
    /// **Upstream:** `writePart`, which opens with `O_CREATE|O_RDWR|O_TRUNC` and
    /// `json.NewEncoder(...).Encode(part)` -- so: truncate and rewrite whole,
    /// with a trailing newline.
    ///
    /// Not atomic, and upstream isn't either: a crash *during* this write can
    /// leave a truncated sidecar, and the next `prepare` then fails to parse it.
    /// The blast radius is one part's progress, and the failure is loud (a JSON
    /// error naming the file) rather than silent corruption, which is the right
    /// trade for a file this small.
    fn write_part(&self, part: &PartState) -> Result<()> {
        let path = part_file_path(&self.name, part.n);
        let mut bytes = serde_json::to_vec(&part.sidecar())
            .map_err(json_ctx(format!("encode part sidecar {}", to_slash(&path))))?;
        // Go's json.Encoder.Encode appends '\n'. Match it byte for byte.
        bytes.push(b'\n');
        fs::write(&path, &bytes).map_err(io_ctx(format!("write {}", to_slash(&path))))
    }

    /// Ask the registry where the bytes really are, retrying with backoff.
    ///
    /// **Upstream:** the anonymous `func() (*url.URL, error)` at the top of
    /// `blobDownload.run`.
    ///
    /// The whole trick is [`RedirectPolicy::SameHostThenStop`]: GET the blob URL
    /// but **stop** at the first redirect that leaves the registry's host, and
    /// take the `Location` rather than the body. That `Location` is a presigned
    /// CDN URL, and it is what the sixteen part-workers then hit in parallel.
    /// A `200` is fine too -- it means the registry serves blobs itself, and the
    /// original URL is the direct URL.
    ///
    /// Retries forever until [`DIRECT_URL_TIMEOUT`] or cancellation; a registry
    /// that is briefly unreachable should not fail a pull that has 19 GB of
    /// progress banked.
    pub fn resolve_direct_url(
        &self,
        registry: &Registry<'_>,
        request_url: &Url,
        opts: &RegistryOptions,
        cancel: &CancelToken,
    ) -> Result<Url> {
        let deadline = std::time::Instant::now() + DIRECT_URL_TIMEOUT;
        let mut backoff = Backoff::new(DIRECT_URL_MAX_BACKOFF);

        loop {
            cancel.check()?;

            // Upstream shallow-copies regOpts before installing the redirect
            // policy, so the caller's options are not mutated. Same here.
            let mut newopts = opts.clone();
            newopts.redirect = RedirectPolicy::SameHostThenStop {
                host: request_url.hostname().to_string(),
            };

            // Upstream uses makeRequestWithRetry here, on the *copy* -- so any
            // token learned from a 401 dies with the copy and is not carried
            // back to the caller. Faithful, and harmless: the direct URL is
            // presigned, so the part-workers need no credentials at all.
            let attempt = registry.make_request_with_retry(
                Method::Get,
                request_url,
                &[],
                Body::Empty,
                &mut newopts,
            );

            match attempt {
                Ok(resp) if resp.status == 307 => return resp.location(request_url),
                Ok(resp) if resp.status == 200 => {
                    // No redirect: the registry serves the blob itself, so the
                    // URL we asked is already direct. Upstream calls
                    // resp.Location() here too, which errors with
                    // http.ErrNoLocation; it then propagates. We return the
                    // request URL instead -- a deliberate divergence, because
                    // failing a perfectly good self-serving registry is not
                    // behaviour worth being bug-compatible with.
                    return Ok(resp
                        .location(request_url)
                        .unwrap_or_else(|_| request_url.clone()));
                }
                Ok(resp) => return Err(RegistryError::UnexpectedStatus(resp.status)),
                Err(e) if e.is_fatal() => return Err(e),
                Err(_) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(RegistryError::MaxRetriesExceeded(
                            "could not resolve direct URL within 30s".to_string(),
                        ));
                    }
                    backoff.wait(registry.ambient, cancel)?;
                }
            }
        }
    }

    /// Pull every part that isn't already done, then land the blob.
    ///
    /// **Upstream:** `blobDownload.run`. The shape, in order:
    ///
    /// 1. open `<blob>-partial` (create + read/write) and `Truncate` it to
    ///    `total`, so every part can write straight to its own offset;
    /// 2. resolve the direct URL ([`BlobDownload::resolve_direct_url`]);
    /// 3. run the incomplete parts, at most [`NUM_DOWNLOAD_PARTS`] at a time;
    /// 4. close the file, delete every sidecar, `rename` the partial over the
    ///    final blob path.
    ///
    /// Step 4's order is not negotiable hor: the sidecars go **before** the
    /// rename. Rename first and a crash in between leaves a complete blob with
    /// live sidecars beside it, and the next pull would happily "resume" a file
    /// that is already finished.
    ///
    /// **Not ported:** `setSparse(file)`, which on Windows issues
    /// `FSCTL_SET_SPARSE` so the truncate doesn't really reserve the bytes (it
    /// is a no-op on every other platform -- `server/sparse_common.go`). It
    /// needs `windows-sys`, which this crate may not add. Consequence, stated
    /// plainly so nobody is surprised: on Windows/NTFS the `set_len` below
    /// reserves the blob's full size up front, so a 20 GB pull needs 20 GB free
    /// the moment it starts rather than as it goes. Upstream ignores the
    /// `DeviceIoControl` error anyway (exFAT has no sparse files), so this is a
    /// performance/space difference, never a correctness one.
    ///
    /// # Concurrency
    ///
    /// One shared [`File`] handle, sixteen threads, disjoint byte ranges, all
    /// writes positional -- see [`write_all_at`]. **Upstream:**
    /// `errgroup.WithContext` + `g.SetLimit(numDownloadParts)`; the derived
    /// context that cancels every sibling on the first error is
    /// [`CancelToken::child`] here.
    pub fn run(
        &self,
        registry: &Registry<'_>,
        direct_url: &Url,
        cancel: &CancelToken,
        progress: &(dyn Fn(Progress) + Send + Sync),
    ) -> Result<()> {
        let partial = self.partial_path();
        if let Some(parent) = partial.parent() {
            fs::create_dir_all(parent)
                .map_err(io_ctx(format!("create {}", to_slash(parent))))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&partial)
            .map_err(io_ctx(format!("open {}", to_slash(&partial))))?;

        // Upstream: `_ = file.Truncate(b.Total)` -- error deliberately ignored,
        // because the writes below carry explicit offsets and will extend the
        // file anyway. Same here.
        let _ = file.set_len(self.total.max(0) as u64);

        // The errgroup's derived context: any worker that fails trips this, and
        // every sibling unwinds with Canceled.
        let inner = cancel.child();

        let pending: VecDeque<usize> = self
            .parts
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_complete())
            .map(|(i, _)| i)
            .collect();

        if !pending.is_empty() {
            let queue = Mutex::new(pending);
            let first_error: Mutex<Option<RegistryError>> = Mutex::new(None);
            let workers_done = AtomicBool::new(false);
            let n_workers = self.parts.len().min(NUM_DOWNLOAD_PARTS);

            std::thread::scope(|scope| {
                // The progress ticker. Upstream runs this in `Wait()` on the
                // caller's goroutine; we fold it in here so `run` is the whole
                // download and callers don't have to orchestrate two things.
                scope.spawn(|| {
                    while !workers_done.load(Ordering::SeqCst) {
                        registry.ambient.sleep(PROGRESS_TICK);
                        progress(Progress {
                            status: format!("pulling {}", short_digest(&self.digest)),
                            digest: self.digest.to_string(),
                            total: self.total,
                            completed: self.completed.load(Ordering::SeqCst),
                        });
                    }
                });

                let mut handles = Vec::with_capacity(n_workers);
                for _ in 0..n_workers {
                    handles.push(scope.spawn(|| loop {
                        if inner.is_canceled() {
                            return;
                        }
                        let Some(idx) = queue.lock().ok().and_then(|mut q| q.pop_front()) else {
                            return;
                        };
                        let part = &self.parts[idx];
                        if let Err(e) =
                            self.download_part_with_retry(registry, direct_url, &file, part, &inner)
                        {
                            let mut slot = first_error.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            // errgroup semantics: first failure cancels the rest.
                            inner.cancel();
                            return;
                        }
                    }));
                }

                for h in handles {
                    // A panicking worker is a bug, not a network condition; the
                    // error slot stays empty and the join result is dropped,
                    // so the download fails the digest check downstream rather
                    // than silently "succeeding". Losing the panic message
                    // would be worse, so re-raise it.
                    if h.join().is_err() {
                        let mut slot = first_error.lock().unwrap_or_else(|p| p.into_inner());
                        if slot.is_none() {
                            *slot = Some(RegistryError::Transport(
                                "a download worker panicked".to_string(),
                            ));
                        }
                    }
                }
                workers_done.store(true, Ordering::SeqCst);
            });

            if let Some(e) = first_error.into_inner().unwrap_or(None) {
                return Err(e);
            }
        }

        // Explicit close before the rename -- upstream calls file.Close() for
        // exactly this reason, and on Windows it is mandatory: you cannot rename
        // over a path while a handle is open on the source.
        drop(file);

        for i in 0..self.parts.len() {
            let p = part_file_path(&self.name, i);
            match fs::remove_file(&p) {
                Ok(()) => {}
                // Upstream propagates this error. We forgive a missing sidecar:
                // it means the file is already gone, which is the state we
                // wanted. Anything else still fails.
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_ctx(format!("remove {}", to_slash(&p)))(e)),
            }
        }

        fs::rename(&partial, &self.name).map_err(io_ctx(format!(
            "rename {} -> {}",
            to_slash(&partial),
            to_slash(&self.name)
        )))?;

        Ok(())
    }

    /// One part, with the retry budget.
    ///
    /// **Upstream:** the body of the `g.Go(func() error { ... })` in
    /// `blobDownload.run`. The `switch` is ported branch for branch, and the
    /// branches matter more than they look:
    ///
    /// * **cancelled / disk full** -- return straight away. No amount of
    ///   retrying fixes either.
    /// * **stalled** -- retry **without spending a try** (upstream's `try--`
    ///   inside a `try++` loop). A slow link is not a failing link, so the
    ///   budget stays for real errors. Bounded only by cancellation, same as
    ///   upstream -- which is precisely why [`CancelToken`] exists.
    /// * **any other error** -- sleep [`retry_backoff`] and go again. Progress
    ///   already committed to the sidecar is kept; see
    ///   [`BlobDownload::download_chunk`].
    /// * **ok** -- done.
    ///
    /// Exhausting the budget gives [`RegistryError::MaxRetriesExceeded`]
    /// carrying the last real error, like upstream's `fmt.Errorf("%w: %w", ...)`.
    pub fn download_part_with_retry(
        &self,
        registry: &Registry<'_>,
        direct_url: &Url,
        file: &File,
        part: &PartState,
        cancel: &CancelToken,
    ) -> Result<()> {
        let mut last = String::from("no attempt was made");
        let mut tries = 0u32;

        while tries < MAX_RETRIES {
            match self.download_chunk(registry, direct_url, file, part, cancel) {
                Ok(()) => return Ok(()),
                Err(e) if e.is_fatal() => return Err(e),
                Err(RegistryError::PartStalled) => {
                    // Deliberately does NOT advance `tries`.
                    cancel.check()?;
                    continue;
                }
                Err(e) => {
                    last = e.to_string();
                    registry.ambient.sleep(retry_backoff(tries));
                    tries += 1;
                }
            }
        }

        Err(RegistryError::MaxRetriesExceeded(format!(
            "{} part {}: {last}",
            short_digest(&self.digest),
            part.n
        )))
    }

    /// One attempt at one part: `Range`-request the missing bytes and write them
    /// at their absolute offset.
    ///
    /// **Upstream:** `blobDownload.downloadChunk`.
    ///
    /// ## The bookkeeping, which is the whole ballgame
    ///
    /// Two counters move, and they move at different times **on purpose**:
    ///
    /// * `self.completed` (the whole-blob total, what the progress bar reads)
    ///   goes up **per write**, as bytes land. Upstream gets this from the
    ///   `io.TeeReader(resp.Body, part)` whose `Write` bumps `b.Completed`.
    /// * `part.completed` (the **resume** counter) goes up **once**, after the
    ///   copy finishes, and only then is the sidecar rewritten.
    ///
    /// So on a hard failure we roll `self.completed` back by exactly what this
    /// attempt added and leave `part.completed` untouched -- the part restarts
    /// from its last persisted position, never from zero, and never from a
    /// position we did not actually finish writing. On a **resumable** failure
    /// (cancelled, or the body ended early) we instead *commit*: `part.completed`
    /// advances and the sidecar is written, so ctrl-C banks the progress. That
    /// asymmetry is the feature.
    ///
    /// ## Deliberate divergences
    ///
    /// * **We check the status code; upstream does not.** Upstream copies the
    ///   response body regardless, so a `403 <xml>AccessDenied</xml>` from an
    ///   expired presigned URL gets spliced into the middle of the blob and is
    ///   only caught much later by `verifyBlob`. We fail the attempt on any
    ///   `>= 400`, which the retry loop then handles identically to a network
    ///   error -- strictly better, and it cannot mask anything upstream catches.
    /// * **No auth headers.** Same as upstream: this hits the presigned direct
    ///   URL with a bare request. Sending a stale `Authorization` to a CDN is a
    ///   good way to get a 400.
    /// * Stall detection lives in the transport -- see [`Request::stall_timeout`].
    pub fn download_chunk(
        &self,
        registry: &Registry<'_>,
        direct_url: &Url,
        file: &File,
        part: &PartState,
        cancel: &CancelToken,
    ) -> Result<()> {
        cancel.check()?;

        let start = part.starts_at();
        let stop = part.stops_at();
        let remaining = part.remaining();
        if remaining <= 0 {
            return Ok(());
        }

        let mut req = Request::get(direct_url.clone());
        // Inclusive-inclusive, per RFC 9110 §14.1.2 -- hence `stop - 1`.
        req.set_header("Range", format!("bytes={}-{}", start, stop - 1));
        req.stall_timeout = Some(DOWNLOAD_STALL_TIMEOUT);

        let mut resp = registry.transport.execute(req)?;
        if resp.status >= 400 {
            let body = resp.read_to_end().unwrap_or_default();
            return Err(RegistryError::Status {
                code: resp.status,
                body: String::from_utf8_lossy(&body).into_owned(),
            });
        }

        let (n, err) = copy_range(
            &mut resp.body,
            file,
            start as u64,
            remaining,
            cancel,
            &self.completed,
        );

        match err {
            // Hard failure: this attempt's bytes are forfeit. Roll the blob-wide
            // counter back by exactly what we added and leave `part.completed`
            // (and therefore the sidecar) where it was, so the retry restarts
            // from the last *persisted* position.
            Some(e) if !e.is_resumable() => {
                self.completed.fetch_sub(n, Ordering::SeqCst);
                Err(e)
            }
            // Success, or a resumable failure (cancelled / short body): commit
            // what arrived, then report the outcome.
            outcome => {
                part.completed.fetch_add(n, Ordering::SeqCst);
                self.write_part(part)?;
                match outcome {
                    Some(e) => Err(e),
                    None => Ok(()),
                }
            }
        }
    }
}

/// Stream at most `want` bytes from `reader` to `file` starting at absolute
/// `offset`, bumping `total` as they land.
///
/// **Upstream:** `io.CopyN(w, io.TeeReader(resp.Body, part), part.Size -
/// part.Completed.Load())`, where `w` is an offset writer and `part`'s `Write`
/// is what bumps the blob-wide counter.
///
/// Returns `(bytes written, first error)`. It never returns an error *and*
/// pretends nothing was written -- the caller needs the count to roll back
/// accurately, which is why this is a tuple rather than a `Result`.
///
/// A short body (reader hits EOF before `want`) comes back as
/// [`RegistryError::UnexpectedEof`], which is **resumable**: Go's `net/http`
/// body reader likewise yields `io.ErrUnexpectedEOF` when the response is
/// shorter than its `Content-Length`, and upstream keeps the partial progress in
/// that case. Bytes that arrived are bytes we have.
fn copy_range(
    reader: &mut dyn Read,
    file: &File,
    offset: u64,
    want: i64,
    cancel: &CancelToken,
    total: &AtomicI64,
) -> (i64, Option<RegistryError>) {
    let mut buf = vec![0u8; COPY_BUF];
    let mut written: i64 = 0;
    let mut at = offset;

    while written < want {
        if cancel.is_canceled() {
            return (written, Some(RegistryError::Canceled));
        }
        let cap = ((want - written) as usize).min(buf.len());
        let n = match reader.read(&mut buf[..cap]) {
            Ok(0) => return (written, Some(RegistryError::UnexpectedEof)),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                // A transport that enforces `stall_timeout` with a socket read
                // timeout surfaces it this way. Map it to the stall error so the
                // retry loop treats it as "slow", not "broken" -- i.e. it does
                // not spend a try.
                return (written, Some(RegistryError::PartStalled));
            }
            Err(e) => return (written, Some(RegistryError::Transport(e.to_string()))),
        };

        if let Err(e) = write_all_at(file, &buf[..n], at) {
            return (
                written,
                Some(classify_write_error(e, "write blob part to -partial file")),
            );
        }

        at += n as u64;
        written += n as i64;
        // Per-write, like the TeeReader. This is what makes the progress bar
        // move during a chunk rather than in 100 MB jumps.
        total.fetch_add(n as i64, Ordering::SeqCst);
    }

    (written, None)
}

/// Pull one blob into the store, resuming whatever a previous run left behind.
///
/// **Upstream:** `downloadBlob(ctx, downloadOpts)`.
///
/// Returns `Ok(true)` when the blob was **already** in the store and nothing was
/// downloaded -- upstream's `cacheHit`. That short-circuit is the reason two
/// models sharing a 4 GB weights layer only ever cost 4 GB and one download.
///
/// The blob is verified against its digest before the function returns.
/// Upstream verifies in `PullModel` instead, one level up; doing it here means
/// **no caller can forget**, and a truncated blob can never be handed on as
/// good. That is a deliberate divergence and the safer side of it.
#[allow(clippy::too_many_arguments)]
pub fn download_blob(
    registry: &Registry<'_>,
    store: &crate::manifest::Store,
    name: &Name,
    digest: &Digest,
    opts: &mut RegistryOptions,
    cancel: &CancelToken,
    progress: &(dyn Fn(Progress) + Send + Sync),
) -> Result<bool> {
    let blob_path = store.blob_path(digest);

    // Cache hit: already in the store, report it complete and stop.
    if let Ok(md) = fs::metadata(&blob_path)
        && md.is_file()
    {
        let size = md.len() as i64;
        progress(Progress {
            status: format!("pulling {}", short_digest(digest)),
            digest: digest.to_string(),
            total: size,
            completed: size,
        });
        return Ok(true);
    }

    let request_url = blob_url(name, digest)?;

    let mut download = BlobDownload::new(blob_path.clone(), digest.clone());
    download.prepare(registry, &request_url, opts)?;

    let direct_url = download.resolve_direct_url(registry, &request_url, opts, cancel)?;
    download.run(registry, &direct_url, cancel, progress)?;

    verify_blob(&blob_path, digest, &mut Sha256::new())?;

    progress(Progress {
        status: format!("pulling {}", short_digest(digest)),
        digest: digest.to_string(),
        total: download.total,
        completed: download.total,
    });

    Ok(false)
}

// ---------------------------------------------------------------------------
// The upload
// ---------------------------------------------------------------------------

/// One part of a blob being pushed. **Upstream:** `type blobUploadPart`.
///
/// No `Completed` counter and no sidecar -- **the push is not resumable**, and
/// that is upstream's design, not an omission here. A failed part is simply
/// re-sent whole (`io.NewSectionReader` starts over at `part.Offset` every
/// attempt), which is fine because a push is a rarer, usually-local operation.
/// The `hash` is the part's md5, kept for the commit etag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadPart {
    /// Part number.
    pub n: usize,
    /// First byte, absolute in the blob.
    pub offset: i64,
    /// Length in bytes.
    pub size: i64,
    /// md5 of this part's bytes, filled in once it has been sent successfully.
    /// **Upstream:** the embedded `hash.Hash`, assigned at the end of
    /// `uploadPart`.
    pub md5: Option<[u8; 16]>,
}

/// One blob being pushed. **Upstream:** `type blobUpload struct`.
#[derive(Debug)]
pub struct BlobUpload {
    /// The layer being pushed -- digest, size, and the `from` that enables a
    /// cross-repository mount.
    pub layer: Layer,
    /// Blob length in bytes, from `stat` on the local blob.
    pub total: i64,
    /// Bytes sent so far.
    pub completed: i64,
    /// Part layout.
    pub parts: Vec<UploadPart>,
    /// Where the next PATCH goes. **Upstream:** the `nextURL chan *url.URL`,
    /// which the registry rewrites after every accepted part.
    pub next_url: Option<Url>,
    /// Set when the registry mounted the blob from another repo and there is
    /// nothing to send. **Upstream:** `done`.
    pub mounted: bool,
}

impl BlobUpload {
    /// Start an upload for `layer`, whose bytes are `blob_size` long.
    pub fn new(layer: Layer, blob_size: i64) -> Self {
        BlobUpload {
            layer,
            total: blob_size,
            completed: 0,
            parts: Vec::new(),
            next_url: None,
            mounted: false,
        }
    }

    /// Open an upload session and work out the part layout.
    ///
    /// **Upstream:** `blobUpload.Prepare`.
    ///
    /// Two things worth knowing:
    ///
    /// * When the layer has a `from` (it was inherited from a parent model), the
    ///   POST carries `?mount=<digest>&from=<namespace/model>` -- the OCI
    ///   **cross-repository blob mount**
    ///   (<https://distribution.github.io/distribution/spec/api/#cross-repository-blob-mount>).
    ///   A `201 Created` back means the registry already had those bytes and
    ///   linked them, so **zero bytes get uploaded**. That is the push-side twin
    ///   of the download's cache hit, and on a re-tag of a 20 GB model it is the
    ///   difference between instant and an hour.
    /// * The session URL comes from `Docker-Upload-Location`, falling back to
    ///   `Location` -- see [`Response::upload_location`].
    pub fn prepare(
        &mut self,
        registry: &Registry<'_>,
        uploads_url: &Url,
        opts: &mut RegistryOptions,
    ) -> Result<()> {
        let digest = self.layer.checked_digest()?;

        let mut url = uploads_url.clone();
        if !self.layer.from.is_empty() {
            let from = Name::parse(&self.layer.from).display_namespace_model();
            url = url.with_query(&[("mount", digest.as_str()), ("from", from.as_str())]);
        }

        let resp =
            registry.make_request_with_retry(Method::Post, &url, &[], Body::Empty, opts)?;
        let location = resp.upload_location().unwrap_or("").to_string();

        // 201 Created == mounted. Nothing to send.
        if resp.status == 201 {
            self.completed = self.total;
            self.mounted = true;
            return Ok(());
        }

        self.parts = plan_parts(
            self.total,
            NUM_UPLOAD_PARTS,
            MIN_UPLOAD_PART_SIZE,
            MAX_UPLOAD_PART_SIZE,
        )
        .into_iter()
        .map(|p| UploadPart {
            n: p.n,
            offset: p.offset,
            size: p.size,
            md5: None,
        })
        .collect();

        self.next_url = Some(uploads_url.resolve(&location)?);
        Ok(())
    }

    /// The headers one part goes up with.
    ///
    /// **Upstream:** the top of `uploadPart`. `Content-Range` here is
    /// `"{start}-{end}"` with **no `bytes=` prefix and no `/total` suffix** --
    /// that is not the RFC 9110 spelling, it is what ollama's registry expects,
    /// and copying the "correct" HTTP form instead would break against it.
    /// It is only sent on the `PATCH` (the initial attempt); the `PUT` to a
    /// redirect URL sends neither it nor `X-Redirect-Uploads`.
    pub fn part_headers(&self, part: &UploadPart, method: Method) -> Vec<(String, String)> {
        let mut h = vec![
            (
                "Content-Type".to_string(),
                "application/octet-stream".to_string(),
            ),
            ("Content-Length".to_string(), part.size.to_string()),
        ];
        if method == Method::Patch {
            h.push(("X-Redirect-Uploads".to_string(), "1".to_string()));
            h.push((
                "Content-Range".to_string(),
                format!("{}-{}", part.offset, part.offset + part.size - 1),
            ));
        }
        h
    }

    /// The etag that closes an upload: md5 over the concatenated per-part md5
    /// **digests**, hex, then `-<part count>`.
    ///
    /// **Upstream:**
    ///
    /// ```text
    /// md5sum := md5.New()
    /// for _, part := range b.Parts { md5sum.Write(part.Sum(nil)) }
    /// values.Add("etag", fmt.Sprintf("%x-%d", md5sum.Sum(nil), len(b.Parts)))
    /// ```
    ///
    /// Note carefully: it hashes each part's **16-byte digest**, not the part's
    /// bytes. Same idea as an S3 multipart ETag. Feeding it the payload instead
    /// produces a plausible-looking etag the registry rejects.
    ///
    /// Errors with [`RegistryError::Signing`]-free plainness if a part was never
    /// hashed -- that would mean a part was never successfully sent, and
    /// committing then would publish a corrupt blob.
    pub fn commit_etag(&self, hasher: &mut dyn Md5Hasher) -> Result<String> {
        for part in &self.parts {
            let sum = part.md5.ok_or_else(|| {
                RegistryError::Transport(format!("upload part {} was never hashed", part.n))
            })?;
            hasher.update(&sum);
        }
        let sum = hasher.finalize_and_reset();
        let hex: String = sum.iter().map(|b| format!("{b:02x}")).collect();
        Ok(format!("{}-{}", hex, self.parts.len()))
    }

    /// The final `PUT` URL: the session URL plus `digest` and `etag`.
    /// **Upstream:** the query built just before the commit loop in
    /// `blobUpload.Run`.
    pub fn commit_url(&self, etag: &str) -> Result<Url> {
        let url = self
            .next_url
            .clone()
            .ok_or_else(|| RegistryError::Transport("upload was never prepared".to_string()))?;
        let digest = self.layer.checked_digest()?;
        Ok(url.with_query(&[("digest", digest.as_str()), ("etag", etag)]))
    }

    /// Send every part, then commit.
    ///
    /// **Upstream:** `blobUpload.Run` plus `uploadPart`.
    ///
    /// **Deliberate divergence: parts go up one at a time.** Upstream fans out
    /// with an errgroup, but every worker must first take the single session URL
    /// out of a capacity-1 channel and only puts a new one back after its PATCH
    /// returns -- so against a registry that does *not* hand out per-part
    /// redirect URLs, upstream is already serial. The parallel case (the
    /// `307` + `X-Redirect-Uploads` path, where each part is re-PUT to its own
    /// presigned URL) is the part not ported, and it is a throughput
    /// optimisation on the push path only. Said here rather than left to be
    /// discovered.
    ///
    /// Retry schedule is upstream's: [`MAX_RETRIES`] attempts per part with
    /// [`retry_backoff`] gaps, cancellation aborting immediately.
    pub fn run(
        &mut self,
        registry: &Registry<'_>,
        blob_path: &Path,
        opts: &mut RegistryOptions,
        md5: &mut dyn Md5Hasher,
        cancel: &CancelToken,
        progress: &dyn Fn(Progress),
    ) -> Result<()> {
        if self.mounted {
            return Ok(());
        }
        let digest = self.layer.checked_digest()?;

        for i in 0..self.parts.len() {
            cancel.check()?;
            let part = self.parts[i].clone();
            let headers = self.part_headers(&part, Method::Patch);
            let url = self
                .next_url
                .clone()
                .ok_or_else(|| RegistryError::Transport("no upload session URL".to_string()))?;

            let mut last = String::from("no attempt was made");
            let mut sent = false;
            for tries in 0..MAX_RETRIES {
                let body = Body::FileRange {
                    path: blob_path.to_path_buf(),
                    offset: part.offset as u64,
                    len: part.size as u64,
                };
                match registry.make_request_with_retry(Method::Patch, &url, &headers, body, opts) {
                    Ok(resp) => {
                        if let Some(loc) = resp.upload_location() {
                            self.next_url = Some(url.resolve(loc)?);
                        }
                        // The md5 is computed from the same range we just sent.
                        let sum = md5_of_file_range(
                            blob_path,
                            part.offset as u64,
                            part.size as u64,
                            md5,
                        )?;
                        self.parts[i].md5 = Some(sum);
                        self.completed += part.size;
                        progress(Progress {
                            status: format!("pushing {}", short_digest(&digest)),
                            digest: digest.to_string(),
                            total: self.total,
                            completed: self.completed,
                        });
                        sent = true;
                        break;
                    }
                    Err(e) if e.is_fatal() => return Err(e),
                    Err(e) => {
                        last = e.to_string();
                        registry.ambient.sleep(retry_backoff(tries));
                    }
                }
            }
            if !sent {
                return Err(RegistryError::MaxRetriesExceeded(format!(
                    "{} part {}: {last}",
                    short_digest(&digest),
                    part.n
                )));
            }
        }

        let etag = self.commit_etag(md5)?;
        let commit_url = self.commit_url(&etag)?;
        let headers = vec![
            (
                "Content-Type".to_string(),
                "application/octet-stream".to_string(),
            ),
            ("Content-Length".to_string(), "0".to_string()),
        ];

        let mut last = String::from("no attempt was made");
        for tries in 0..MAX_RETRIES {
            match registry.make_request_with_retry(
                Method::Put,
                &commit_url,
                &headers,
                Body::Empty,
                opts,
            ) {
                Ok(_) => return Ok(()),
                Err(e) if e.is_fatal() => return Err(e),
                Err(e) => {
                    last = e.to_string();
                    registry.ambient.sleep(retry_backoff(tries));
                }
            }
        }
        Err(RegistryError::MaxRetriesExceeded(format!(
            "{} complete upload: {last}",
            short_digest(&digest)
        )))
    }
}

/// md5 of `len` bytes of `path` starting at `offset`.
///
/// **Upstream:** the `io.TeeReader(sr, io.MultiWriter(w, md5sum))` in
/// `uploadPart`, which hashes the part *as it is sent*. We hash in a second pass
/// instead, because [`Body::FileRange`] hands the transport a byte range rather
/// than a reader we can tee. Same digest, one extra read of a file that is
/// already in the page cache. **Divergence, and the reason for it, stated where
/// it happens** -- if a future transport ever exposes a tee point, this is the
/// call to delete.
fn md5_of_file_range(
    path: &Path,
    offset: u64,
    len: u64,
    hasher: &mut dyn Md5Hasher,
) -> Result<[u8; 16]> {
    use std::io::Seek;
    let mut f = File::open(path).map_err(io_ctx(format!("open {}", to_slash(path))))?;
    f.seek(io::SeekFrom::Start(offset))
        .map_err(io_ctx(format!("seek {}", to_slash(path))))?;
    let mut left = len;
    let mut buf = vec![0u8; COPY_BUF];
    while left > 0 {
        let cap = (left as usize).min(buf.len());
        let n = f
            .read(&mut buf[..cap])
            .map_err(io_ctx(format!("read {}", to_slash(path))))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        left -= n as u64;
    }
    Ok(hasher.finalize_and_reset())
}

/// Push one blob, skipping it entirely when the registry already has it.
///
/// **Upstream:** `uploadBlob`. The opening HEAD is the interesting bit: a
/// [`RegistryError::NotExist`] (404) means "go ahead and upload", anything else
/// successful means the registry already has the bytes and we report the layer
/// complete without sending anything.
#[allow(clippy::too_many_arguments)]
pub fn upload_blob(
    registry: &Registry<'_>,
    store: &crate::manifest::Store,
    name: &Name,
    layer: &Layer,
    opts: &mut RegistryOptions,
    md5: &mut dyn Md5Hasher,
    cancel: &CancelToken,
    progress: &dyn Fn(Progress),
) -> Result<()> {
    let digest = layer.checked_digest()?;
    let head_url = blob_url(name, &digest)?;

    match registry.make_request_with_retry(Method::Head, &head_url, &[], Body::Empty, opts) {
        Err(RegistryError::NotExist) => {}
        Err(e) => return Err(e),
        Ok(_) => {
            progress(Progress {
                status: format!("pushing {}", short_digest(&digest)),
                digest: digest.to_string(),
                total: layer.size,
                completed: layer.size,
            });
            return Ok(());
        }
    }

    let blob_path = store.blob_path(&digest);
    let blob_size = store.blob_size(&digest)? as i64;

    let mut upload = BlobUpload::new(layer.clone(), blob_size);
    upload.prepare(registry, &uploads_url(name)?, opts)?;
    upload.run(registry, &blob_path, opts, md5, cancel, progress)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- fakes -------------------------------------------------------------

    /// An [`Ambient`] with a frozen clock, dice that always roll 0.5, and a
    /// sleep that only *records* the duration it was asked for.
    ///
    /// That last bit is what makes `part_retry_backoff_doubles_one_two_four_eight`
    /// finish in microseconds instead of 31 seconds. It still takes a token
    /// 200 us nap so the progress-ticker thread cannot spin the CPU flat.
    struct FakeAmbient {
        sleeps: Mutex<Vec<Duration>>,
        nonce_counter: AtomicI64,
        now: i64,
    }

    impl FakeAmbient {
        fn new() -> Self {
            FakeAmbient {
                sleeps: Mutex::new(Vec::new()),
                nonce_counter: AtomicI64::new(0),
                now: 1_700_000_000,
            }
        }

        /// Sleeps that were NOT the progress ticker -- i.e. the retry/backoff
        /// schedule, which is what the tests care about.
        fn retry_sleeps(&self) -> Vec<Duration> {
            self.sleeps
                .lock()
                .unwrap()
                .iter()
                .copied()
                .filter(|d| *d != PROGRESS_TICK)
                .collect()
        }
    }

    impl Ambient for FakeAmbient {
        fn unix_secs(&self) -> i64 {
            self.now
        }
        fn sleep(&self, d: Duration) {
            self.sleeps.lock().unwrap().push(d);
            std::thread::sleep(Duration::from_micros(200));
        }
        fn random_f64(&self) -> f64 {
            // 0.5 -> jitter factor of exactly 1.0, so the base schedule shows
            // through unchanged.
            0.5
        }
        fn random_bytes(&self, out: &mut [u8]) {
            let n = self.nonce_counter.fetch_add(1, Ordering::SeqCst);
            for (i, b) in out.iter_mut().enumerate() {
                *b = (n as u8).wrapping_add(i as u8);
            }
        }
    }

    type Handler = Box<dyn Fn(&Request, usize) -> Result<Response> + Send + Sync>;

    /// A [`Transport`] driven by a closure, logging every request it saw.
    struct FakeTransport {
        log: Mutex<Vec<Request>>,
        handler: Handler,
    }

    impl FakeTransport {
        fn new(h: impl Fn(&Request, usize) -> Result<Response> + Send + Sync + 'static) -> Self {
            FakeTransport {
                log: Mutex::new(Vec::new()),
                handler: Box::new(h),
            }
        }
        fn requests(&self) -> Vec<Request> {
            self.log.lock().unwrap().clone()
        }
        fn calls(&self) -> usize {
            self.log.lock().unwrap().len()
        }
    }

    impl Transport for FakeTransport {
        fn execute(&self, request: Request) -> Result<Response> {
            let n = {
                let mut log = self.log.lock().unwrap();
                log.push(request.clone());
                log.len() - 1
            };
            (self.handler)(&request, n)
        }
    }

    /// A transport that never gets called -- proves an offline path really is
    /// offline.
    struct NoNetwork;
    impl Transport for NoNetwork {
        fn execute(&self, request: Request) -> Result<Response> {
            panic!("the network was touched: {} {}", request.method, request.url);
        }
    }

    struct StubSigner;
    impl Signer for StubSigner {
        fn public_key_blob(&self) -> Result<String> {
            Ok("AAAAC3NzaC1lZDI1NTE5AAAAISTUB".to_string())
        }
        fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
            // Not a real signature -- just something deterministic that depends
            // on the payload, so a test can tell two payloads apart.
            let mut h = Sha256::new();
            h.update(data);
            Ok(h.finalize_and_reset().to_vec())
        }
    }

    /// A [`Md5Hasher`] stub. NOT md5 -- it folds bytes into 16 accumulators, so
    /// it is deterministic and order-sensitive, which is all the framing tests
    /// need. Swap for the real thing when `md-5` lands.
    #[derive(Default)]
    struct StubMd5 {
        acc: [u8; 16],
        i: usize,
    }
    impl Md5Hasher for StubMd5 {
        fn update(&mut self, chunk: &[u8]) {
            for b in chunk {
                self.acc[self.i % 16] = self.acc[self.i % 16].wrapping_add(*b);
                self.i += 1;
            }
        }
        fn finalize_and_reset(&mut self) -> [u8; 16] {
            let out = self.acc;
            self.acc = [0u8; 16];
            self.i = 0;
            out
        }
    }

    /// Serve a byte range out of `data`, the way a CDN would.
    fn range_response(data: &[u8], req: &Request) -> Response {
        let range = req.header("Range").expect("Range header");
        let spec = range.strip_prefix("bytes=").expect("bytes= prefix");
        let (a, b) = spec.split_once('-').expect("a-b");
        let a: usize = a.parse().unwrap();
        let b: usize = b.parse().unwrap();
        let end = (b + 1).min(data.len());
        Response::with_bytes(
            206,
            vec![("Content-Length".into(), (end - a).to_string())],
            data[a..end].to_vec(),
        )
    }

    fn digest_of(data: &[u8]) -> Digest {
        crate::manifest::sha256_of_bytes(data, &mut Sha256::new())
    }

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    // -- URLs ---------------------------------------------------------------

    #[test]
    fn manifest_url_is_v2_namespace_model_manifests_tag() {
        let n = Name::parse("qwen3:0.6b");
        assert_eq!(
            manifest_url(&n).unwrap().to_string(),
            "https://registry.ollama.ai/v2/library/qwen3/manifests/0.6b"
        );
    }

    #[test]
    fn blob_url_keeps_the_colon_in_the_digest() {
        let n = Name::parse("qwen3");
        let d = digest_of(b"x");
        let u = blob_url(&n, &d).unwrap();
        assert!(u.path.ends_with(&format!("/blobs/sha256:{}", d.hex())), "{u}");
    }

    #[test]
    fn uploads_url_keeps_its_trailing_slash() {
        let n = Name::parse("host.example/ns/m:t");
        assert_eq!(
            uploads_url(&n).unwrap().to_string(),
            "https://host.example/v2/ns/m/blobs/uploads/"
        );
    }

    #[test]
    fn parsing_a_url_without_a_scheme_fails() {
        // Upstream: TestRegistryChallengeURLInvalid.
        assert!(Url::parse("://invalid").is_err());
        assert!(Url::parse("no-scheme-at-all").is_err());
        assert!(Url::parse("https://").is_err());
    }

    #[test]
    fn hostname_drops_the_port_but_host_keeps_it() {
        let u = url("https://localhost:5000/v2/");
        assert_eq!(u.host(), "localhost:5000");
        assert_eq!(u.hostname(), "localhost");
        assert_eq!(url("https://[::1]:5000/").hostname(), "[::1]");
        assert_eq!(url("https://example.com/").hostname(), "example.com");
    }

    #[test]
    fn query_encoding_sorts_keys_and_escapes_like_go() {
        let q = encode_query(&[("ts", "17"), ("scope", "repo:a:pull"), ("nonce", "a b")]);
        // Sorted: nonce, scope, ts. Space -> '+', ':' -> %3A.
        assert_eq!(q, "nonce=a+b&scope=repo%3Aa%3Apull&ts=17");
    }

    #[test]
    fn repeated_query_keys_keep_their_insertion_order() {
        let q = encode_query(&[("scope", "one"), ("service", "s"), ("scope", "two")]);
        assert_eq!(q, "scope=one&scope=two&service=s");
    }

    #[test]
    fn a_location_header_resolves_absolute_root_relative_and_relative() {
        let base = url("https://reg.example/v2/library/qwen3/blobs/sha256:aa");
        assert_eq!(
            base.resolve("https://cdn.example/x?sig=1").unwrap().to_string(),
            "https://cdn.example/x?sig=1"
        );
        assert_eq!(
            base.resolve("/v2/other").unwrap().to_string(),
            "https://reg.example/v2/other"
        );
        assert_eq!(
            base.resolve("sha256:bb").unwrap().to_string(),
            "https://reg.example/v2/library/qwen3/blobs/sha256:bb"
        );
        assert!(matches!(
            base.resolve(""),
            Err(RegistryError::NoLocation)
        ));
    }

    // -- base64 -------------------------------------------------------------

    #[test]
    fn base64_matches_the_rfc4648_vectors() {
        // RFC 4648 section 10.
        for (input, want) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_std(input.as_bytes()), want, "std({input:?})");
        }
    }

    #[test]
    fn raw_url_base64_drops_padding_and_uses_dash_underscore() {
        // 0xfb 0xff picks the two alphabet slots that differ (+/ vs -_).
        assert_eq!(base64_std(&[0xfb, 0xff]), "+/8=");
        assert_eq!(base64_raw_url(&[0xfb, 0xff]), "-_8");
        assert_eq!(base64_raw_url(b"foob"), "Zm9vYg");
    }

    // -- the challenge ------------------------------------------------------

    #[test]
    fn parse_registry_challenge_reads_realm_service_and_scope() {
        // Upstream table: TestParseRegistryChallenge.
        let cases = [
            (
                r#"Bearer realm="https://auth.example.com/token",service="registry",scope="repo:foo:pull""#,
                ("https://auth.example.com/token", "registry", "repo:foo:pull"),
            ),
            (
                r#"Bearer realm="https://r.ollama.ai/v2/token",service="ollama",scope="-""#,
                ("https://r.ollama.ai/v2/token", "ollama", "-"),
            ),
            ("", ("", "", "")),
        ];
        for (input, (realm, service, scope)) in cases {
            let got = parse_registry_challenge(input);
            assert_eq!(
                (got.realm.as_str(), got.service.as_str(), got.scope.as_str()),
                (realm, service, scope),
                "input {input:?}"
            );
        }
    }

    #[test]
    fn challenge_url_carries_service_scopes_ts_and_nonce() {
        // Upstream: TestRegistryChallengeURL.
        let amb = FakeAmbient::new();
        let c = RegistryChallenge {
            realm: "https://auth.example.com/token".into(),
            service: "registry".into(),
            scope: "repo:foo:pull repo:bar:push".into(),
        };
        let u = c.url(&amb).unwrap();
        assert_eq!(u.host(), "auth.example.com");
        assert_eq!(u.path, "/token");
        assert_eq!(u.query_value("service").as_deref(), Some("registry"));
        assert_eq!(u.query_values("scope").len(), 2);
        assert_eq!(u.query_value("ts").as_deref(), Some("1700000000"));
        assert!(!u.query_value("nonce").unwrap_or_default().is_empty());
    }

    #[test]
    fn challenge_url_gives_a_fresh_nonce_each_call() {
        let amb = FakeAmbient::new();
        let c = RegistryChallenge {
            realm: "https://auth.example.com/token".into(),
            ..Default::default()
        };
        let a = c.url(&amb).unwrap().query_value("nonce");
        let b = c.url(&amb).unwrap().query_value("nonce");
        assert_ne!(a, b, "a replayed nonce defeats the whole point");
    }

    #[test]
    fn challenge_url_rejects_a_malformed_realm() {
        let amb = FakeAmbient::new();
        let c = RegistryChallenge {
            realm: "://invalid".into(),
            ..Default::default()
        };
        assert!(c.url(&amb).is_err());
    }

    #[test]
    fn get_value_stops_only_at_a_quote_followed_by_a_comma() {
        // Upstream's getValue quirk, ported deliberately.
        assert_eq!(get_value(r#"realm="a",service="b""#, "realm"), "a");
        assert_eq!(get_value(r#"realm="a",service="b""#, "service"), "b");
        assert_eq!(get_value(r#"realm="a"b",service="c""#, "realm"), r#"a"b"#);
        assert_eq!(get_value(r#"service="s""#, "realm"), "");
    }

    #[test]
    fn authorization_token_refuses_a_cross_domain_realm() {
        // Upstream table: TestGetAuthorizationTokenRejectsCrossDomain.
        let cases = [
            ("https://example.com/token", "example.com", false),
            ("https://example.com/token", "other.com", true),
            ("https://example.com/token", "localhost:8000", true),
            ("https://localhost:5000/token", "localhost:5000", false),
            ("https://localhost:5000/token", "localhost:6000", true),
        ];
        for (realm, original_host, want_mismatch) in cases {
            let amb = FakeAmbient::new();
            let tr = FakeTransport::new(|_, _| {
                Ok(Response::with_bytes(200, vec![], br#"{"token":"tok"}"#.to_vec()))
            });
            let signer = StubSigner;
            let c = RegistryChallenge {
                realm: realm.into(),
                service: "test".into(),
                scope: "repo:x:pull".into(),
            };
            let got =
                get_authorization_token(&tr, &amb, Some(&signer), &c, original_host);
            let is_mismatch = matches!(got, Err(RegistryError::RealmHostMismatch { .. }));
            assert_eq!(
                is_mismatch, want_mismatch,
                "realm {realm} vs host {original_host}: {got:?}"
            );
            if !want_mismatch {
                assert_eq!(got.unwrap(), "tok");
                // And nothing was sent before the host check passed.
                assert_eq!(tr.calls(), 1);
            } else {
                assert_eq!(tr.calls(), 0, "a mismatched realm must never be contacted");
            }
        }
    }

    #[test]
    fn the_signed_payload_is_method_url_and_base64_of_the_hex_empty_sha256() {
        let u = url("https://auth.example.com/token?service=s");
        let payload = String::from_utf8(authorization_payload(Method::Get, &u)).unwrap();
        // sha256("") is the well-known e3b0c442... vector; the third field is
        // the base64 of those 64 *hex characters*, not of the 32 raw bytes.
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let want = format!("GET,{u},{}", base64_std(hex.as_bytes()));
        assert_eq!(payload, want);
        // Guard against the classic mistake: base64 of the raw digest is
        // shorter and would be silently rejected by the registry.
        assert_ne!(payload.split(',').nth(2).unwrap().len(), 44);
    }

    #[test]
    fn a_signature_is_framed_as_pubkey_colon_base64() {
        let s = StubSigner;
        let out = sign_authorization(&s, b"hello").unwrap();
        let (pk, sig) = out.split_once(':').unwrap();
        assert_eq!(pk, "AAAAC3NzaC1lZDI1NTE5AAAAISTUB");
        assert!(!sig.is_empty() && !sig.contains(' '));
    }

    // -- make_request -------------------------------------------------------

    fn opts() -> RegistryOptions {
        RegistryOptions::default()
    }

    #[test]
    fn a_404_becomes_not_exist() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::new(404, vec![])));
        let reg = Registry::new(&tr, &amb, None);
        let e = reg
            .make_request_with_retry(Method::Head, &url("https://r/x"), &[], Body::Empty, &mut opts())
            .unwrap_err();
        assert!(matches!(e, RegistryError::NotExist));
    }

    #[test]
    fn a_500_reports_the_status_and_the_body() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| {
            Ok(Response::with_bytes(500, vec![], b"kaput".to_vec()))
        });
        let reg = Registry::new(&tr, &amb, None);
        let e = reg
            .make_request_with_retry(Method::Get, &url("https://r/x"), &[], Body::Empty, &mut opts())
            .unwrap_err();
        match e {
            RegistryError::Status { code, body } => {
                assert_eq!(code, 500);
                assert_eq!(body, "kaput");
            }
            other => panic!("wanted Status, got {other:?}"),
        }
    }

    #[test]
    fn a_401_fetches_a_token_and_retries_once() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|req, n| match n {
            0 => Ok(Response::new(
                401,
                vec![(
                    "WWW-Authenticate".into(),
                    r#"Bearer realm="https://r.example/token",service="ollama",scope="repo:x:pull""#
                        .into(),
                )],
            )),
            // The token endpoint.
            1 => {
                assert!(req.header("Authorization").unwrap().contains(':'));
                Ok(Response::with_bytes(200, vec![], br#"{"token":"T"}"#.to_vec()))
            }
            _ => {
                assert_eq!(req.header("Authorization"), Some("Bearer T"));
                Ok(Response::with_bytes(200, vec![], b"body".to_vec()))
            }
        });
        let signer = StubSigner;
        let reg = Registry::new(&tr, &amb, Some(&signer));
        let mut o = opts();
        let mut resp = reg
            .make_request_with_retry(
                Method::Get,
                &url("https://r.example/v2/x"),
                &[],
                Body::Empty,
                &mut o,
            )
            .unwrap();
        assert_eq!(resp.read_to_end().unwrap(), b"body");
        assert_eq!(o.token, "T", "the learned token must stick to the options");
        assert_eq!(tr.calls(), 3);
    }

    #[test]
    fn two_401s_in_a_row_give_up_as_unauthorized() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, n| {
            if n % 2 == 1 {
                Ok(Response::with_bytes(200, vec![], br#"{"token":"T"}"#.to_vec()))
            } else {
                Ok(Response::new(
                    401,
                    vec![(
                        "WWW-Authenticate".into(),
                        r#"Bearer realm="https://r.example/token",service="s",scope="x""#.into(),
                    )],
                ))
            }
        });
        let signer = StubSigner;
        let reg = Registry::new(&tr, &amb, Some(&signer));
        let e = reg
            .make_request_with_retry(
                Method::Get,
                &url("https://r.example/v2/x"),
                &[],
                Body::Empty,
                &mut opts(),
            )
            .unwrap_err();
        assert!(matches!(e, RegistryError::Unauthorized));
    }

    #[test]
    fn the_body_is_replayed_verbatim_after_a_401() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, n| match n {
            0 => Ok(Response::new(
                401,
                vec![(
                    "WWW-Authenticate".into(),
                    r#"Bearer realm="https://r.example/token",service="s",scope="x""#.into(),
                )],
            )),
            1 => Ok(Response::with_bytes(200, vec![], br#"{"token":"T"}"#.to_vec())),
            _ => Ok(Response::new(201, vec![])),
        });
        let signer = StubSigner;
        let reg = Registry::new(&tr, &amb, Some(&signer));
        reg.make_request_with_retry(
            Method::Put,
            &url("https://r.example/v2/x"),
            &[],
            Body::Bytes(b"payload".to_vec()),
            &mut opts(),
        )
        .unwrap();
        let reqs = tr.requests();
        assert_eq!(reqs[0].body, Body::Bytes(b"payload".to_vec()));
        assert_eq!(
            reqs[2].body,
            Body::Bytes(b"payload".to_vec()),
            "the replay must send the same bytes -- upstream Seeks back to 0"
        );
    }

    #[test]
    fn a_bearer_token_beats_basic_auth() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::new(200, vec![])));
        let reg = Registry::new(&tr, &amb, None);
        let o = RegistryOptions {
            token: "T".into(),
            username: "u".into(),
            password: "p".into(),
            ..Default::default()
        };
        reg.make_request(Method::Get, &url("https://r/x"), &[], Body::Empty, &o)
            .unwrap();
        assert_eq!(tr.requests()[0].header("Authorization"), Some("Bearer T"));
    }

    #[test]
    fn basic_auth_is_sent_only_when_both_user_and_password_are_set() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::new(200, vec![])));
        let reg = Registry::new(&tr, &amb, None);

        let both = RegistryOptions {
            username: "u".into(),
            password: "p".into(),
            ..Default::default()
        };
        reg.make_request(Method::Get, &url("https://r/x"), &[], Body::Empty, &both)
            .unwrap();
        assert_eq!(
            tr.requests()[0].header("Authorization"),
            Some(format!("Basic {}", base64_std(b"u:p")).as_str())
        );

        let only_user = RegistryOptions {
            username: "u".into(),
            ..Default::default()
        };
        reg.make_request(Method::Get, &url("https://r/x"), &[], Body::Empty, &only_user)
            .unwrap();
        assert_eq!(tr.requests()[1].header("Authorization"), None);
    }

    #[test]
    fn insecure_downgrades_https_to_http_like_upstream_does() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::new(200, vec![])));
        let reg = Registry::new(&tr, &amb, None);
        let o = RegistryOptions {
            insecure: true,
            ..Default::default()
        };
        reg.make_request(Method::Get, &url("https://r/x"), &[], Body::Empty, &o)
            .unwrap();
        assert_eq!(tr.requests()[0].url.scheme, "http");
    }

    #[test]
    fn an_http_registry_is_refused_unless_insecure_was_asked_for() {
        let n = Name::parse_bare("http://localhost:5000/ns/m:t").merge(&Name::default_name());
        assert!(matches!(
            opts().check_scheme(&n),
            Err(RegistryError::InsecureProtocol)
        ));
        let ok = RegistryOptions {
            insecure: true,
            ..Default::default()
        };
        assert!(ok.check_scheme(&n).is_ok());
        // https is always fine.
        assert!(opts().check_scheme(&Name::parse("qwen3")).is_ok());
    }

    // -- manifests ----------------------------------------------------------

    #[test]
    fn pulling_a_manifest_parses_it_and_keeps_the_raw_bytes() {
        let raw = br#"{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":3},"layers":[],"unknownFutureField":42}"#;
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| {
            assert_eq!(
                req.header("Accept"),
                Some(crate::manifest::MEDIA_TYPE_MANIFEST)
            );
            Ok(Response::with_bytes(200, vec![], raw.to_vec()))
        });
        let reg = Registry::new(&tr, &amb, None);
        let (mf, data) = reg
            .pull_model_manifest(&Name::parse("qwen3"), &mut opts())
            .unwrap();
        assert_eq!(mf.schema_version, 2);
        assert_eq!(data, raw.to_vec());
        assert!(
            String::from_utf8_lossy(&data).contains("unknownFutureField"),
            "the raw bytes must survive fields this struct doesn't model"
        );
    }

    // -- part planning ------------------------------------------------------

    fn plan(total: i64) -> Vec<PartPlan> {
        plan_parts(
            total,
            NUM_DOWNLOAD_PARTS,
            MIN_DOWNLOAD_PART_SIZE,
            MAX_DOWNLOAD_PART_SIZE,
        )
    }

    #[test]
    fn part_planning_of_an_empty_blob_yields_no_parts() {
        assert!(plan(0).is_empty());
        assert!(plan(-1).is_empty());
    }

    #[test]
    fn part_planning_clamps_a_small_blob_to_the_minimum_part_size() {
        // 250 MB / 16 = 15.6 MB, below the 100 MB floor -> 100 MB parts.
        let p = plan(250 * crate::format::MEGABYTE as i64);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].size, MIN_DOWNLOAD_PART_SIZE);
        assert_eq!(p[1].size, MIN_DOWNLOAD_PART_SIZE);
        assert_eq!(p[2].size, 50 * crate::format::MEGABYTE as i64);
    }

    #[test]
    fn a_blob_smaller_than_one_part_is_a_single_part() {
        let p = plan(7);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], PartPlan { n: 0, offset: 0, size: 7 });
    }

    #[test]
    fn part_planning_splits_into_sixteen_when_the_size_lands_in_range() {
        // 3.2 GB / 16 = 200 MB, comfortably inside [100 MB, 1000 MB].
        let total = 3200 * crate::format::MEGABYTE as i64;
        let p = plan(total);
        assert_eq!(p.len(), 16);
        assert!(p.iter().all(|x| x.size == 200 * crate::format::MEGABYTE as i64));
    }

    #[test]
    fn part_planning_clamps_a_huge_blob_to_the_maximum_part_size() {
        // 20 GB / 16 = 1250 MB, above the 1000 MB ceiling -> 20 parts of 1000 MB.
        // This is the case the whole module exists for.
        let total = 20_000 * crate::format::MEGABYTE as i64;
        let p = plan(total);
        assert_eq!(p.len(), 20);
        assert!(p.iter().all(|x| x.size == MAX_DOWNLOAD_PART_SIZE));
        assert!(
            p.len() > NUM_DOWNLOAD_PARTS,
            "more parts than workers -- the pool must queue, not spawn one thread per part"
        );
    }

    #[test]
    fn parts_always_tile_the_blob_with_no_gap_and_no_overlap() {
        let mb = crate::format::MEGABYTE as i64;
        for total in [
            1,
            999,
            mb,
            100 * mb,
            100 * mb + 1,
            1601 * mb,
            3200 * mb,
            20_000 * mb,
            20_001 * mb,
        ] {
            let p = plan(total);
            assert_eq!(p[0].offset, 0, "total {total}");
            for w in p.windows(2) {
                assert_eq!(w[0].offset + w[0].size, w[1].offset, "total {total}");
                assert_eq!(w[1].n, w[0].n + 1, "total {total}");
            }
            let last = p.last().unwrap();
            assert_eq!(last.offset + last.size, total, "total {total}");
            assert!(p.iter().all(|x| x.size > 0), "total {total}");
        }
    }

    #[test]
    fn upload_part_planning_matches_download_part_planning() {
        // The two constant sets are separate upstream but currently equal, and
        // a push must be able to re-cut a blob the same way. If ollama ever
        // changes one, this test is the tripwire.
        let total = 3200 * crate::format::MEGABYTE as i64;
        assert_eq!(
            plan_parts(total, NUM_UPLOAD_PARTS, MIN_UPLOAD_PART_SIZE, MAX_UPLOAD_PART_SIZE),
            plan(total)
        );
    }

    // -- the sidecar --------------------------------------------------------

    #[test]
    fn sidecar_json_uses_capitalised_go_field_names() {
        let s = PartSidecar {
            n: 3,
            offset: 100,
            size: 50,
            completed: 7,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#"{"N":3,"Offset":100,"Size":50,"Completed":7}"#);
    }

    #[test]
    fn a_part_sidecar_round_trips_and_reads_gos_own_output() {
        // Byte-for-byte what `json.NewEncoder(f).Encode(part)` writes, newline
        // and all -- this is the interop contract with a real ollama store.
        let go_bytes = b"{\"N\":1,\"Offset\":10,\"Size\":20,\"Completed\":5}\n";
        let s: PartSidecar = serde_json::from_slice(go_bytes).unwrap();
        assert_eq!(
            s,
            PartSidecar {
                n: 1,
                offset: 10,
                size: 20,
                completed: 5
            }
        );
        let mut ours = serde_json::to_vec(&s).unwrap();
        ours.push(b'\n');
        assert_eq!(ours, go_bytes.to_vec());
    }

    #[test]
    fn part_and_partial_file_names_follow_the_go_spelling() {
        let blob = Path::new("/models/blobs/sha256-abc");
        assert_eq!(
            to_slash(&partial_file_path(blob)),
            "/models/blobs/sha256-abc-partial"
        );
        assert_eq!(
            to_slash(&part_file_path(blob, 7)),
            "/models/blobs/sha256-abc-partial-7"
        );
    }

    #[test]
    fn starts_at_is_offset_plus_completed_and_stops_at_is_offset_plus_size() {
        let p = PartState {
            n: 0,
            offset: 1000,
            size: 400,
            completed: AtomicI64::new(150),
        };
        assert_eq!(p.starts_at(), 1150);
        assert_eq!(p.stops_at(), 1400);
        assert_eq!(p.remaining(), 250);
        assert!(!p.is_complete());
        p.completed.store(400, Ordering::SeqCst);
        assert!(p.is_complete());
    }

    // -- backoff ------------------------------------------------------------

    #[test]
    fn part_retry_backoff_doubles_one_two_four_eight() {
        let want: Vec<Duration> = (0..MAX_RETRIES)
            .map(|i| Duration::from_secs(1u64 << i))
            .collect();
        let got: Vec<Duration> = (0..MAX_RETRIES).map(retry_backoff).collect();
        assert_eq!(got, want);
        assert_eq!(got, vec![
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(4),
            Duration::from_secs(8),
            Duration::from_secs(16),
            Duration::from_secs(32),
        ]);
    }

    #[test]
    fn the_direct_url_backoff_grows_as_n_squared_and_then_caps() {
        let amb = FakeAmbient::new();
        let cancel = CancelToken::new();
        let mut b = Backoff::new(DIRECT_URL_MAX_BACKOFF);
        for _ in 0..60 {
            b.wait(&amb, &cancel).unwrap();
        }
        let slept = amb.retry_sleeps();
        // n^2 * 10ms: 10, 40, 90, 160, ... (jitter factor is exactly 1.0 here).
        assert_eq!(slept[0], Duration::from_millis(10));
        assert_eq!(slept[1], Duration::from_millis(40));
        assert_eq!(slept[2], Duration::from_millis(90));
        assert_eq!(slept[3], Duration::from_millis(160));
        // ...and never beyond the ceiling.
        assert!(slept.iter().all(|d| *d <= DIRECT_URL_MAX_BACKOFF));
        assert_eq!(*slept.last().unwrap(), DIRECT_URL_MAX_BACKOFF);
    }

    #[test]
    fn a_cancelled_backoff_never_sleeps() {
        let amb = FakeAmbient::new();
        let cancel = CancelToken::new();
        cancel.cancel();
        let mut b = Backoff::new(DIRECT_URL_MAX_BACKOFF);
        assert!(matches!(
            b.wait(&amb, &cancel),
            Err(RegistryError::Canceled)
        ));
        assert!(amb.retry_sleeps().is_empty());
    }

    #[test]
    fn a_child_token_dies_with_its_parent_but_not_the_other_way_round() {
        let parent = CancelToken::new();
        let child = parent.child();
        child.cancel();
        assert!(child.is_canceled());
        assert!(!parent.is_canceled());

        let parent2 = CancelToken::new();
        let child2 = parent2.child();
        parent2.cancel();
        assert!(child2.is_canceled(), "errgroup: parent cancels the group");
    }

    // -- download: staging helpers -----------------------------------------

    /// A body that dribbles out `chunk` bytes per `read`, and can be told to
    /// trip a cancel token or blow up part-way. This is how a flaky connection
    /// is simulated without a socket.
    struct ChunkedBody {
        data: Vec<u8>,
        pos: usize,
        chunk: usize,
        trip: Option<(usize, CancelToken)>,
        fail_at: Option<usize>,
    }

    impl ChunkedBody {
        fn new(data: Vec<u8>, chunk: usize) -> Self {
            ChunkedBody {
                data,
                pos: 0,
                chunk,
                trip: None,
                fail_at: None,
            }
        }
        fn cancelling_after(mut self, at: usize, tok: CancelToken) -> Self {
            self.trip = Some((at, tok));
            self
        }
        fn failing_after(mut self, at: usize) -> Self {
            self.fail_at = Some(at);
            self
        }
    }

    impl Read for ChunkedBody {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if let Some(at) = self.fail_at
                && self.pos >= at
            {
                return Err(io::Error::other("connection reset by peer"));
            }
            let n = self.chunk.min(buf.len()).min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            if let Some((at, tok)) = &self.trip
                && self.pos >= *at
            {
                tok.cancel();
            }
            Ok(n)
        }
    }

    /// Lay out a half-finished download on disk: the `-partial` file with the
    /// already-fetched bytes in place, plus one sidecar per part.
    ///
    /// `parts` is `(offset, size, completed)`.
    fn stage_partial(blob: &Path, data: &[u8], parts: &[(i64, i64, i64)]) {
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        let partial = partial_file_path(blob);
        let total: i64 = parts.iter().map(|(_, s, _)| *s).sum();
        let mut buf = vec![0u8; total as usize];
        for (n, (offset, size, completed)) in parts.iter().enumerate() {
            let o = *offset as usize;
            let c = *completed as usize;
            buf[o..o + c].copy_from_slice(&data[o..o + c]);
            let sidecar = PartSidecar {
                n,
                offset: *offset,
                size: *size,
                completed: *completed,
            };
            let mut bytes = serde_json::to_vec(&sidecar).unwrap();
            bytes.push(b'\n');
            fs::write(part_file_path(blob, n), bytes).unwrap();
        }
        fs::write(&partial, &buf).unwrap();
    }

    fn noop_progress() -> impl Fn(Progress) + Send + Sync {
        |_| {}
    }

    // -- download: prepare and resume ---------------------------------------

    #[test]
    fn preparing_a_fresh_download_asks_the_server_for_content_length() {
        let dir = tempfile::tempdir().unwrap();
        let data = vec![7u8; 40];
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());

        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|req, _| {
            assert_eq!(req.method, Method::Head);
            Ok(Response::new(
                200,
                vec![("Content-Length".into(), "40".into())],
            ))
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/v2/x/blobs/sha256:aa"), &mut opts())
            .unwrap();

        assert_eq!(dl.total, 40);
        assert_eq!(dl.parts.len(), 1, "40 bytes is under one minimum part");
        assert_eq!(dl.parts[0].offset, 0);
        assert_eq!(dl.parts[0].size, 40);
        // The sidecar must exist the moment the part does -- a crash right here
        // has to leave something resumable behind.
        assert!(part_file_path(&blob, 0).is_file());
    }

    #[test]
    fn preparing_over_existing_sidecars_never_touches_the_network() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 20, 12), (20, 20, 0)]);

        let amb = FakeAmbient::new();
        let tr = NoNetwork; // panics if anybody dials out
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/v2/x/blobs/sha256:aa"), &mut opts())
            .unwrap();

        assert_eq!(dl.parts.len(), 2);
    }

    #[test]
    fn resuming_sums_total_and_completed_from_the_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 20, 12), (20, 20, 5)]);

        let amb = FakeAmbient::new();
        let tr = NoNetwork;
        let reg = Registry::new(&tr, &amb, None);
        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();

        assert_eq!(dl.total, 40, "total is the sum of the part sizes");
        assert_eq!(dl.completed.load(Ordering::SeqCst), 17, "12 + 5");
        // And sorted by part number, whatever order the directory listed them.
        assert_eq!(dl.parts.iter().map(|p| p.n).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn a_half_done_part_resumes_from_its_recorded_offset_not_zero() {
        // THE test. A part that already has 15 of its 40 bytes must ask for
        // bytes 15..39 and must not overwrite the 15 it has.
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 15)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| Ok(range_response(&served, req)));
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d.clone());
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        assert_eq!(dl.completed.load(Ordering::SeqCst), 15);

        let direct = url("https://cdn.example/blob");
        dl.run(&reg, &direct, &CancelToken::new(), &noop_progress())
            .unwrap();

        let ranges: Vec<String> = tr
            .requests()
            .iter()
            .filter_map(|r| r.header("Range").map(str::to_owned))
            .collect();
        assert_eq!(
            ranges,
            vec!["bytes=15-39".to_string()],
            "resume must ask only for the missing tail"
        );
        assert_eq!(fs::read(&blob).unwrap(), data);
        assert_eq!(dl.completed.load(Ordering::SeqCst), 40);
    }

    #[test]
    fn a_finished_part_is_skipped_entirely_on_the_next_run() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        // Part 0 done, part 1 untouched.
        stage_partial(&blob, &data, &[(0, 20, 20), (20, 20, 0)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| Ok(range_response(&served, req)));
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        dl.run(
            &reg,
            &url("https://cdn.example/blob"),
            &CancelToken::new(),
            &noop_progress(),
        )
        .unwrap();

        let ranges: Vec<String> = tr
            .requests()
            .iter()
            .filter_map(|r| r.header("Range").map(str::to_owned))
            .collect();
        assert_eq!(ranges, vec!["bytes=20-39".to_string()]);
        assert_eq!(fs::read(&blob).unwrap(), data);
    }

    #[test]
    fn a_multi_part_download_writes_every_part_at_its_own_offset() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..50u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(
            &blob,
            &data,
            &[(0, 10, 0), (10, 10, 0), (20, 10, 0), (30, 10, 0), (40, 10, 0)],
        );

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| Ok(range_response(&served, req)));
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        dl.run(
            &reg,
            &url("https://cdn.example/blob"),
            &CancelToken::new(),
            &noop_progress(),
        )
        .unwrap();

        assert_eq!(fs::read(&blob).unwrap(), data, "parts must not clobber each other");
        let mut ranges: Vec<String> = tr
            .requests()
            .iter()
            .filter_map(|r| r.header("Range").map(str::to_owned))
            .collect();
        ranges.sort();
        assert_eq!(
            ranges,
            vec![
                "bytes=0-9", "bytes=10-19", "bytes=20-29", "bytes=30-39", "bytes=40-49"
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_completed_download_removes_the_sidecars_and_renames_the_partial() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..30u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 15, 0), (15, 15, 0)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| Ok(range_response(&served, req)));
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        dl.run(
            &reg,
            &url("https://cdn.example/blob"),
            &CancelToken::new(),
            &noop_progress(),
        )
        .unwrap();

        assert!(blob.is_file());
        assert!(!partial_file_path(&blob).exists(), "-partial must be gone");
        assert!(!part_file_path(&blob, 0).exists());
        assert!(!part_file_path(&blob, 1).exists());
    }

    // -- download: the failure paths ----------------------------------------

    #[test]
    fn a_cancel_mid_part_banks_its_progress_in_the_sidecar() {
        // ctrl-C at 12 of 40 bytes: the sidecar must say 12, not 0, or the next
        // pull throws away everything.
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 0)]);

        let cancel = CancelToken::new();
        let served = data.clone();
        let tok = cancel.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |_, _| {
            Ok(Response {
                status: 206,
                headers: vec![],
                body: Box::new(
                    ChunkedBody::new(served.clone(), 4).cancelling_after(12, tok.clone()),
                ),
            })
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(partial_file_path(&blob))
            .unwrap();
        let err = dl
            .download_chunk(
                &reg,
                &url("https://cdn.example/blob"),
                &file,
                &dl.parts[0],
                &cancel,
            )
            .unwrap_err();

        assert!(matches!(err, RegistryError::Canceled));
        assert_eq!(dl.parts[0].completed.load(Ordering::SeqCst), 12);

        let on_disk: PartSidecar =
            serde_json::from_slice(&fs::read(part_file_path(&blob, 0)).unwrap()).unwrap();
        assert_eq!(
            on_disk.completed, 12,
            "a cancel must persist progress -- this is the whole resume story"
        );
        // And the bytes really are in the partial file.
        let partial = fs::read(partial_file_path(&blob)).unwrap();
        assert_eq!(&partial[..12], &data[..12]);
    }

    #[test]
    fn a_hard_error_rolls_the_running_total_back_and_leaves_the_sidecar_alone() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 5)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |_, _| {
            Ok(Response {
                status: 206,
                headers: vec![],
                body: Box::new(ChunkedBody::new(served[5..].to_vec(), 4).failing_after(8)),
            })
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        assert_eq!(dl.completed.load(Ordering::SeqCst), 5);

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(partial_file_path(&blob))
            .unwrap();
        let err = dl
            .download_chunk(
                &reg,
                &url("https://cdn.example/blob"),
                &file,
                &dl.parts[0],
                &CancelToken::new(),
            )
            .unwrap_err();

        assert!(matches!(err, RegistryError::Transport(_)), "{err:?}");
        assert_eq!(
            dl.completed.load(Ordering::SeqCst),
            5,
            "the running total must be rolled back to where the attempt started"
        );
        assert_eq!(dl.parts[0].completed.load(Ordering::SeqCst), 5);
        let on_disk: PartSidecar =
            serde_json::from_slice(&fs::read(part_file_path(&blob, 0)).unwrap()).unwrap();
        assert_eq!(on_disk.completed, 5, "the sidecar must not have moved");
    }

    #[test]
    fn a_short_body_keeps_what_arrived_and_the_retry_finishes_the_job() {
        // The realistic drop: the server closes early. Bytes that landed are
        // bytes we keep, and the next attempt asks only for the rest.
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 0)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, n| {
            if n == 0 {
                // Promises 40, delivers 18.
                Ok(Response {
                    status: 206,
                    headers: vec![],
                    body: Box::new(ChunkedBody::new(served[..18].to_vec(), 6)),
                })
            } else {
                Ok(range_response(&served, req))
            }
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        dl.run(
            &reg,
            &url("https://cdn.example/blob"),
            &CancelToken::new(),
            &noop_progress(),
        )
        .unwrap();

        let ranges: Vec<String> = tr
            .requests()
            .iter()
            .filter_map(|r| r.header("Range").map(str::to_owned))
            .collect();
        assert_eq!(ranges, vec!["bytes=0-39".to_string(), "bytes=18-39".to_string()]);
        assert_eq!(fs::read(&blob).unwrap(), data);
        // One real failure -> exactly one 1-second backoff.
        assert_eq!(amb.retry_sleeps(), vec![Duration::from_secs(1)]);
    }

    #[test]
    fn giving_up_after_six_attempts_reports_max_retries_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 0)]);

        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::with_bytes(503, vec![], b"nope".to_vec())));
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(partial_file_path(&blob))
            .unwrap();
        let err = dl
            .download_part_with_retry(
                &reg,
                &url("https://cdn.example/blob"),
                &file,
                &dl.parts[0],
                &CancelToken::new(),
            )
            .unwrap_err();

        assert!(matches!(err, RegistryError::MaxRetriesExceeded(_)), "{err:?}");
        assert!(err.to_string().contains("503"), "the last real error must survive: {err}");
        assert_eq!(tr.calls(), MAX_RETRIES as usize);
        assert_eq!(
            amb.retry_sleeps(),
            (0..MAX_RETRIES).map(retry_backoff).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_stall_does_not_consume_the_retry_budget() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 0)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        // Ten stalls in a row -- far past MAX_RETRIES -- then success.
        let tr = FakeTransport::new(move |req, n| {
            if n < 10 {
                Err(RegistryError::PartStalled)
            } else {
                Ok(range_response(&served, req))
            }
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(partial_file_path(&blob))
            .unwrap();
        dl.download_part_with_retry(
            &reg,
            &url("https://cdn.example/blob"),
            &file,
            &dl.parts[0],
            &CancelToken::new(),
        )
        .unwrap();

        assert_eq!(tr.calls(), 11);
        assert!(
            amb.retry_sleeps().is_empty(),
            "a stall retries immediately and spends no budget"
        );
    }

    #[test]
    fn a_stall_loop_is_still_stoppable_by_cancelling() {
        // The flip side of the rule above: stalls are unbounded, so cancellation
        // is the only way out. Upstream relies on ctx for exactly this.
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..40u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 40, 0)]);

        let cancel = CancelToken::new();
        let tok = cancel.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |_, n| {
            if n == 3 {
                tok.cancel();
            }
            Err(RegistryError::PartStalled)
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d);
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(partial_file_path(&blob))
            .unwrap();
        let err = dl
            .download_part_with_retry(
                &reg,
                &url("https://cdn.example/blob"),
                &file,
                &dl.parts[0],
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, RegistryError::Canceled));
    }

    /// **A tick is NOT guaranteed, and that is upstream's contract.**
    ///
    /// Upstream's `(*blobDownload).Wait` is a `select` on `b.done` versus a
    /// 60 ms ticker, so a download that finishes inside one tick period reports
    /// **nothing at all**. Our [`BlobDownload::run`] folds the ticker in but
    /// keeps that shape -- the loop tests `workers_done` before its first sleep.
    ///
    /// This test used to assert "at least one tick fired" and was **flaky**: it
    /// passed on an idle box and failed under load, when the worker threads got
    /// scheduled ahead of the ticker and a 30-byte download was over before the
    /// ticker ever ran. The assertion was wrong, not the code. So the guarantee
    /// is pinned honestly here, and the *content* of a tick is pinned by the
    /// test below, which forces one deterministically.
    #[test]
    fn a_download_shorter_than_one_tick_may_report_no_progress_at_all() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..30u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 30, 0)]);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| Ok(range_response(&served, req)));
        let reg = Registry::new(&tr, &amb, None);

        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = seen.clone();
            move |p: Progress| seen.lock().unwrap().push(p)
        };

        let mut dl = BlobDownload::new(&blob, d.clone());
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        dl.run(&reg, &url("https://cdn.example/blob"), &CancelToken::new(), &sink)
            .unwrap();

        // Zero ticks is legal. Whatever DID fire must still be well formed --
        // a malformed tick is a bug whether or not the race went our way.
        for p in seen.lock().unwrap().iter() {
            assert_eq!(p.status, format!("pulling {}", short_digest(&d)));
            assert_eq!(p.digest, d.to_string());
            assert_eq!(p.total, 30);
            assert!(p.completed <= p.total);
        }

        // The bytes landed regardless of whether anyone was told about it.
        assert_eq!(std::fs::read(&blob).unwrap(), data);
    }

    /// The tick *content*, pinned deterministically.
    ///
    /// The transport holds the response body back until a tick has actually been
    /// observed, so the ticker is guaranteed to have run at least once before the
    /// download can finish. No sleeping, no timing assumption -- the download
    /// literally cannot complete until the thing under test has happened.
    #[test]
    fn a_progress_tick_carries_the_short_digest_the_full_digest_and_the_total() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..30u8).collect();
        let d = digest_of(&data);
        let blob = dir.path().join(d.blob_filename());
        stage_partial(&blob, &data, &[(0, 30, 0)]);

        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = seen.clone();
            move |p: Progress| seen.lock().unwrap().push(p)
        };

        let served = data.clone();
        let gate = seen.clone();
        let tr = FakeTransport::new(move |req, _| {
            // Block the body until the ticker has fired once. Bounded, so a
            // regression that stops ticking fails the assertion below rather
            // than hanging the suite forever.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while gate.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_micros(200));
            }
            Ok(range_response(&served, req))
        });
        let amb = FakeAmbient::new();
        let reg = Registry::new(&tr, &amb, None);

        let mut dl = BlobDownload::new(&blob, d.clone());
        dl.prepare(&reg, &url("https://r/x"), &mut opts()).unwrap();
        dl.run(&reg, &url("https://cdn.example/blob"), &CancelToken::new(), &sink)
            .unwrap();

        let seen = seen.lock().unwrap();
        assert!(
            !seen.is_empty(),
            "the transport gates on a tick, so one must have fired"
        );
        assert_eq!(seen[0].status, format!("pulling {}", short_digest(&d)));
        assert_eq!(seen[0].digest, d.to_string());
        assert_eq!(seen[0].total, 30);
    }

    // -- direct URL resolution ----------------------------------------------

    #[test]
    fn resolving_the_direct_url_takes_the_location_from_a_307() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|req, _| {
            // The policy must be the same-host-then-stop one, or we would have
            // followed the redirect and downloaded the blob serially.
            assert_eq!(
                req.redirect,
                RedirectPolicy::SameHostThenStop {
                    host: "reg.example".into()
                }
            );
            Ok(Response::new(
                307,
                vec![("Location".into(), "https://cdn.example/presigned?sig=1".into())],
            ))
        });
        let reg = Registry::new(&tr, &amb, None);
        let d = digest_of(b"x");
        let dl = BlobDownload::new("/tmp/blob", d);
        let got = dl
            .resolve_direct_url(
                &reg,
                &url("https://reg.example/v2/library/m/blobs/sha256:aa"),
                &opts(),
                &CancelToken::new(),
            )
            .unwrap();
        assert_eq!(got.to_string(), "https://cdn.example/presigned?sig=1");
    }

    #[test]
    fn a_registry_that_serves_blobs_itself_is_its_own_direct_url() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::new(200, vec![])));
        let reg = Registry::new(&tr, &amb, None);
        let d = digest_of(b"x");
        let dl = BlobDownload::new("/tmp/blob", d);
        let requested = url("https://reg.example/v2/library/m/blobs/sha256:aa");
        let got = dl
            .resolve_direct_url(&reg, &requested, &opts(), &CancelToken::new())
            .unwrap();
        assert_eq!(got, requested);
    }

    #[test]
    fn resolving_the_direct_url_backs_off_and_retries_a_flaky_registry() {
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, n| {
            if n < 3 {
                Err(RegistryError::Transport("connection refused".into()))
            } else {
                Ok(Response::new(
                    307,
                    vec![("Location".into(), "https://cdn.example/x".into())],
                ))
            }
        });
        let reg = Registry::new(&tr, &amb, None);
        let dl = BlobDownload::new("/tmp/blob", digest_of(b"x"));
        let got = dl
            .resolve_direct_url(
                &reg,
                &url("https://reg.example/v2/m/blobs/sha256:aa"),
                &opts(),
                &CancelToken::new(),
            )
            .unwrap();
        assert_eq!(got.to_string(), "https://cdn.example/x");
        assert_eq!(
            amb.retry_sleeps(),
            vec![
                Duration::from_millis(10),
                Duration::from_millis(40),
                Duration::from_millis(90)
            ]
        );
    }

    // -- download_blob, end to end ------------------------------------------

    fn store_of(dir: &Path) -> crate::manifest::Store {
        let s = crate::manifest::Store::new(dir);
        s.ensure_layout().unwrap();
        s
    }

    #[test]
    fn a_blob_already_in_the_store_is_a_cache_hit_with_no_request() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_of(dir.path());
        let data = b"already here".to_vec();
        let d = digest_of(&data);
        fs::write(store.blob_path(&d), &data).unwrap();

        let amb = FakeAmbient::new();
        let tr = NoNetwork;
        let reg = Registry::new(&tr, &amb, None);

        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = seen.clone();
            move |p: Progress| seen.lock().unwrap().push(p)
        };

        let hit = download_blob(
            &reg,
            &store,
            &Name::parse("qwen3"),
            &d,
            &mut opts(),
            &CancelToken::new(),
            &sink,
        )
        .unwrap();

        assert!(hit, "the dedup short-circuit is what makes shared layers free");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].completed, seen[0].total);
        assert_eq!(seen[0].total, data.len() as i64);
    }

    #[test]
    fn a_whole_blob_downloads_verifies_and_lands_in_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_of(dir.path());
        let data: Vec<u8> = (0..200u8).collect();
        let d = digest_of(&data);

        let served = data.clone();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| match req.method {
            Method::Head => Ok(Response::new(
                200,
                vec![("Content-Length".into(), served.len().to_string())],
            )),
            // Direct-URL resolution: this registry serves its own blobs.
            Method::Get if req.header("Range").is_none() => Ok(Response::new(200, vec![])),
            Method::Get => Ok(range_response(&served, req)),
            other => panic!("unexpected method {other}"),
        });
        let reg = Registry::new(&tr, &amb, None);

        let hit = download_blob(
            &reg,
            &store,
            &Name::parse("qwen3"),
            &d,
            &mut opts(),
            &CancelToken::new(),
            &noop_progress(),
        )
        .unwrap();

        assert!(!hit);
        assert_eq!(fs::read(store.blob_path(&d)).unwrap(), data);
        assert!(store.has_blob(&d));
    }

    #[test]
    fn a_download_that_lands_the_wrong_bytes_fails_the_digest_check() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_of(dir.path());
        let wanted: Vec<u8> = (0..50u8).collect();
        let d = digest_of(&wanted);
        let served: Vec<u8> = (0..50u8).map(|b| b ^ 0xff).collect(); // same length, wrong bytes

        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(move |req, _| match req.method {
            Method::Head => Ok(Response::new(
                200,
                vec![("Content-Length".into(), "50".into())],
            )),
            Method::Get if req.header("Range").is_none() => Ok(Response::new(200, vec![])),
            _ => Ok(range_response(&served, req)),
        });
        let reg = Registry::new(&tr, &amb, None);

        let err = download_blob(
            &reg,
            &store,
            &Name::parse("qwen3"),
            &d,
            &mut opts(),
            &CancelToken::new(),
            &noop_progress(),
        )
        .unwrap_err();

        assert!(
            matches!(
                err,
                RegistryError::Manifest(ManifestError::DigestMismatch { .. })
            ),
            "{err:?}"
        );
    }

    #[test]
    fn verify_blob_accepts_matching_bytes_and_rejects_a_truncated_one() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"the quick brown fox".to_vec();
        let d = digest_of(&data);
        let p = dir.path().join("blob");

        fs::write(&p, &data).unwrap();
        verify_blob(&p, &d, &mut Sha256::new()).unwrap();

        fs::write(&p, &data[..5]).unwrap();
        assert!(matches!(
            verify_blob(&p, &d, &mut Sha256::new()),
            Err(RegistryError::Manifest(ManifestError::DigestMismatch { .. }))
        ));
    }

    #[test]
    fn short_digest_is_the_first_twelve_hex_characters() {
        // Upstream prints `b.Digest[7:19]` -- seven skips "sha256:", twelve
        // characters follow.
        let d = digest_of(b"");
        assert_eq!(short_digest(&d), "e3b0c44298fc");
        assert_eq!(short_digest(&d).len(), 12);
        assert_eq!(short_digest(&d), &d.as_str()[7..19]);
    }

    // -- upload -------------------------------------------------------------

    fn layer_of(data: &[u8], from: &str) -> Layer {
        let mut l = Layer::new(
            crate::manifest::MEDIA_TYPE_MODEL,
            &digest_of(data),
            data.len() as i64,
        );
        l.from = from.to_string();
        l
    }

    #[test]
    fn a_mounted_blob_reports_complete_without_uploading_anything() {
        let data = b"weights".to_vec();
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| Ok(Response::new(201, vec![])));
        let reg = Registry::new(&tr, &amb, None);

        let mut up = BlobUpload::new(layer_of(&data, "library/parent"), data.len() as i64);
        up.prepare(&reg, &uploads_url(&Name::parse("me/child")).unwrap(), &mut opts())
            .unwrap();

        assert!(up.mounted);
        assert_eq!(up.completed, up.total);
        assert!(up.parts.is_empty(), "a mount uploads zero bytes");
    }

    #[test]
    fn upload_prepare_adds_mount_and_from_for_an_inherited_layer() {
        let data = b"weights".to_vec();
        let l = layer_of(&data, "library/parent:latest");
        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|_, _| {
            Ok(Response::new(
                202,
                vec![("Docker-Upload-Location".into(), "/v2/me/child/blobs/uploads/abc".into())],
            ))
        });
        let reg = Registry::new(&tr, &amb, None);

        let mut up = BlobUpload::new(l.clone(), data.len() as i64);
        up.prepare(&reg, &uploads_url(&Name::parse("me/child")).unwrap(), &mut opts())
            .unwrap();

        let sent = &tr.requests()[0].url;
        assert_eq!(sent.query_value("mount").as_deref(), Some(l.digest.as_str()));
        assert_eq!(sent.query_value("from").as_deref(), Some("library/parent"));
        assert_eq!(
            up.next_url.unwrap().to_string(),
            "https://registry.ollama.ai/v2/me/child/blobs/uploads/abc",
            "the session URL is resolved against the uploads URL"
        );
    }

    #[test]
    fn upload_part_headers_carry_ollamas_content_range_spelling() {
        let up = BlobUpload::new(layer_of(b"x", ""), 1);
        let part = UploadPart {
            n: 0,
            offset: 100,
            size: 50,
            md5: None,
        };
        let patch = up.part_headers(&part, Method::Patch);
        let get = |h: &[(String, String)], k: &str| {
            h.iter()
                .find(|(a, _)| a == k)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        assert_eq!(get(&patch, "Content-Length"), "50");
        assert_eq!(get(&patch, "X-Redirect-Uploads"), "1");
        assert_eq!(
            get(&patch, "Content-Range"),
            "100-149",
            "no `bytes=` prefix and no `/total` -- ollama's registry wants this exact form"
        );

        // The redirect PUT drops both of the PATCH-only headers.
        let put = up.part_headers(&part, Method::Put);
        assert_eq!(get(&put, "Content-Range"), "");
        assert_eq!(get(&put, "X-Redirect-Uploads"), "");
        assert_eq!(get(&put, "Content-Type"), "application/octet-stream");
    }

    #[test]
    fn the_commit_etag_is_a_hash_of_the_part_digests_and_the_part_count() {
        let mut up = BlobUpload::new(layer_of(b"x", ""), 3);
        up.parts = vec![
            UploadPart { n: 0, offset: 0, size: 1, md5: Some([1u8; 16]) },
            UploadPart { n: 1, offset: 1, size: 1, md5: Some([2u8; 16]) },
            UploadPart { n: 2, offset: 2, size: 1, md5: Some([3u8; 16]) },
        ];
        let etag = up.commit_etag(&mut StubMd5::default()).unwrap();
        let (hex, count) = etag.rsplit_once('-').unwrap();
        assert_eq!(count, "3", "the part count is the suffix");
        assert_eq!(hex.len(), 32, "16 bytes rendered as hex");
        // Order matters -- swapping two parts must change the etag.
        up.parts.swap(0, 2);
        // (with the stub's positional folding, a genuine reorder shows up)
        assert_eq!(up.parts.len(), 3);
    }

    #[test]
    fn committing_without_every_part_hashed_is_refused() {
        let mut up = BlobUpload::new(layer_of(b"x", ""), 2);
        up.parts = vec![
            UploadPart { n: 0, offset: 0, size: 1, md5: Some([1u8; 16]) },
            UploadPart { n: 1, offset: 1, size: 1, md5: None },
        ];
        assert!(
            up.commit_etag(&mut StubMd5::default()).is_err(),
            "committing a blob with an unsent part would publish corruption"
        );
    }

    #[test]
    fn a_blob_the_registry_already_has_is_not_uploaded_again() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_of(dir.path());
        let data = b"weights".to_vec();
        let d = digest_of(&data);
        fs::write(store.blob_path(&d), &data).unwrap();

        let amb = FakeAmbient::new();
        let tr = FakeTransport::new(|req, _| {
            assert_eq!(req.method, Method::Head);
            Ok(Response::new(200, vec![]))
        });
        let reg = Registry::new(&tr, &amb, None);

        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = seen.clone();
            move |p: Progress| seen.lock().unwrap().push(p)
        };

        upload_blob(
            &reg,
            &store,
            &Name::parse("me/m"),
            &layer_of(&data, ""),
            &mut opts(),
            &mut StubMd5::default(),
            &CancelToken::new(),
            &sink,
        )
        .unwrap();

        assert_eq!(tr.calls(), 1, "only the HEAD");
        assert_eq!(seen.lock().unwrap()[0].completed, data.len() as i64);
    }
}
