//! A git-2-shaped compatibility shim implemented over `gitoxide` (`gix`).
//!
//! This module exists so the kopi-beans git layer can drop its dependency on
//! `git-2` (which vendors libgit-2 + OpenSSL, a C toolchain) in favour of the
//! pure-Rust `gix`. The ~90 former `git-2::` call sites were mechanically
//! repointed at `crate::git::gix_compat::`, so the surface here deliberately
//! mirrors the small slice of the `git-2` API the crate actually used:
//! `Repository`, `Oid`, `Signature`/`Time`, `Reference`, `Tree`/`TreeEntry`/
//! `TreeBuilder`, `Commit`, `Object`/`Blob`, `Remote`, plus `Error`/`ErrorCode`
//! and the `ObjectType`/`TreeWalk*` enums.
//!
//! # What is faithful vs. gated
//! - Object/tree/commit/ref plumbing is a faithful translation.
//! - **Fetch** is implemented over `gix`'s blocking network client
//!   (`remote.connect(Fetch) -> prepare_fetch -> receive`).
//! - **Push** is implemented for **local** remotes (a filesystem path or a
//!   `file://` URL): [`Repository::push_refspecs`] copies the objects the target
//!   repo lacks into its object database and updates the destination refs with
//!   fast-forward + lock semantics, entirely in-process via gix (no `git`
//!   subprocess, no network protocol). **Network** transports
//!   (`http(s)`/`ssh`/`git`/scp-like) are delegated to the system `git`
//!   binary, because gix 0.86 exposes no high-level push and no
//!   `gix-protocol` send-pack helper (upstream #306), so a protocol push cannot
//!   be built in-process without a `[patch.crates-io]` (which would not survive
//!   `cargo publish`). Shelling out keeps the *build* pure-Rust — `git` is a
//!   runtime dependency of network push only, not a linked C toolchain — and
//!   inherits the user's existing credential configuration (helpers, netrc,
//!   SSH agent), which an in-process implementation would have to reproduce.
//!   [`ErrorCode::PushUnsupported`] is still returned when no usable `git` is
//!   found on `PATH`, so callers that gate on it keep working.
//! - Remote *configuration* (create/read `remote.<name>.url`) is managed by
//!   this shim directly against the on-disk git config, so a freshly opened
//!   handle observes remotes added through it without a reload.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gix::bstr::{BStr, ByteSlice};

// =============================================================================
// Push stall guards (kopitiam#25)
// =============================================================================
//
// # Provenance, honestly stated
//
// There is no upstream default to copy here, and that IS the finding: **git
// ships its own stall detection switched off.** In `http.c` git only calls
// `curl_easy_setopt(CURLOPT_LOW_SPEED_LIMIT/_TIME)` when
// `http.lowSpeedLimit`/`http.lowSpeedTime` are configured; on a stock install
// both are unset (`git config --get http.lowSpeedLimit` prints nothing), so
// curl's low-speed abort never arms and a transfer that goes quiet hangs until
// the TCP stack gives up — which can be many minutes, or never. git-config(1)
// documents the pair as opt-in for exactly this purpose: "if the HTTP transfer
// speed is less than `http.lowSpeedLimit` for longer than
// `http.lowSpeedTime` seconds, the transfer is aborted."
//
// So these three numbers are ours, not borrowed, and each is justified below
// along with what would make it wrong. They are passed per-invocation with
// `-c`, so the user's own config is never modified and an explicit user
// setting can still be layered on by editing this call site if it ever matters.

/// Bytes/sec below which an HTTP push counts as stalled.
///
/// 1000 B/s. Rationale: this must sit far below any link on which a push is
/// worth attempting (even a bad mobile connection sustains tens of KB/s) yet
/// far above zero, so that a transfer genuinely dribbling along is not killed.
/// A bead store push is kilobytes, so a healthy push never spends 30s under
/// 1 KB/s — it is finished long before.
///
/// What would make this wrong: pushing a genuinely huge pack over a link slower
/// than 1 KB/s sustained. That push would be aborted here. If that ever becomes
/// a real workload, this needs to be configurable rather than merely lowered.
const PUSH_LOW_SPEED_LIMIT_BYTES: u64 = 1000;

/// Seconds a push may sit under [`PUSH_LOW_SPEED_LIMIT_BYTES`] before git kills
/// it.
///
/// 30 s. Deliberately shorter than the daemon's 60 s `SyncWait` default
/// deadline, so a stalled HTTP push surfaces as a real *failure carrying git's
/// own message* well inside the window a waiting `bn sync` is prepared to sit
/// for — instead of an uninformative client-side timeout. Long enough to ride
/// out a slow server-side pack negotiation on a big repo.
const PUSH_LOW_SPEED_TIME_SECS: u64 = 30;

/// Hard ceiling on one `git push` invocation, enforced by killing the process.
///
/// 120 s. This is the backstop for everything
/// [`PUSH_LOW_SPEED_LIMIT_BYTES`]/[`PUSH_LOW_SPEED_TIME_SECS`] cannot see:
/// ssh transports (the `http.*` settings simply do not apply), DNS resolution,
/// TCP connect, and TLS handshake — all of which happen before curl has any
/// transfer rate to measure.
///
/// Sized at 4x the low-speed window so it never pre-empts the cleaner
/// mechanism: for an http remote git should always abort itself first and hand
/// us a real error message, and this watchdog should only ever fire on the
/// cases git cannot self-abort. Killing a push is the blunt option — we lose
/// git's diagnosis and learn only "it never answered" — so it is the last
/// resort, not the first.
///
/// What would make this wrong: a legitimate push that takes over two minutes
/// (very large initial history over a thin link). It would be killed mid-flight
/// and retried on backoff, never converging. The store would need this raised.
const PUSH_WATCHDOG: Duration = Duration::from_secs(120);

// =============================================================================
// Error + ErrorCode
// =============================================================================

/// git-2-shaped error codes. Only the variants the crate inspects are modelled;
/// everything else collapses to `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    NotFound,
    Locked,
    /// gix 0.86 has no high-level push; surfaced so callers can gate cleanly.
    PushUnsupported,
    Other,
}

/// git-2-shaped error carrying a code and message.
#[derive(Clone, Debug)]
pub struct Error {
    code: ErrorCode,
    message: String,
}

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }
    pub fn locked(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Locked, message)
    }
    pub fn other(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Other, message)
    }
    /// Mirror of `git-2::Error::from_str`.
    pub fn from_str(message: &str) -> Self {
        Self::new(ErrorCode::Other, message)
    }
    pub fn code(&self) -> ErrorCode {
        self.code
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// Classify a gix error whose `Display` we trust to mention "not found".
fn other<E: fmt::Display>(e: E) -> Error {
    Error::other(e.to_string())
}

/// Build a message including the full `source()` chain (for diagnostics).
fn chain_message<E: std::error::Error>(e: &E) -> String {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(" -> ");
        msg.push_str(&s.to_string());
        src = s.source();
    }
    msg
}

// =============================================================================
// Oid
// =============================================================================

/// A git object id (newtype over `gix::ObjectId`, which is `Copy`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Oid(gix::ObjectId);

impl Oid {
    /// The all-zero SHA-1 oid (git-2's `Oid::zero`).
    pub fn zero() -> Self {
        Oid(gix::ObjectId::null(gix::hash::Kind::Sha1))
    }
    pub fn is_zero(&self) -> bool {
        self.0.is_null()
    }
    /// Build an oid from raw bytes (20 for SHA-1).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        gix::ObjectId::try_from(bytes)
            .map(Oid)
            .map_err(|e| Error::other(format!("invalid oid bytes: {e}")))
    }
    /// Parse an oid from a hex string (git-2's `Oid::from_str`).
    pub fn from_str(s: &str) -> Result<Self, Error> {
        gix::ObjectId::from_hex(s.as_bytes())
            .map(Oid)
            .map_err(|e| Error::other(format!("invalid oid hex: {e}")))
    }
    fn inner(self) -> gix::ObjectId {
        self.0
    }
}

impl From<gix::ObjectId> for Oid {
    fn from(id: gix::ObjectId) -> Self {
        Oid(id)
    }
}
impl From<Oid> for gix::ObjectId {
    fn from(id: Oid) -> Self {
        id.0
    }
}

impl fmt::Display for Oid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
impl fmt::Debug for Oid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

// =============================================================================
// Time + Signature
// =============================================================================

/// git-2-shaped commit time (seconds since epoch + tz offset in seconds).
#[derive(Clone, Copy, Debug)]
pub struct Time {
    seconds: i64,
    offset: i32,
}

impl Time {
    pub fn new(seconds: i64, offset: i32) -> Self {
        Self { seconds, offset }
    }
    pub fn seconds(&self) -> i64 {
        self.seconds
    }
}

impl From<gix::date::Time> for Time {
    fn from(t: gix::date::Time) -> Self {
        Time {
            seconds: t.seconds,
            offset: t.offset,
        }
    }
}

/// git-2-shaped signature owning its fields.
#[derive(Clone, Debug)]
pub struct Signature {
    name: String,
    email: String,
    time: Time,
}

impl Signature {
    /// Signature stamped with the current wall-clock time.
    pub fn now(name: &str, email: &str) -> Result<Self, Error> {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Ok(Self {
            name: name.to_owned(),
            email: email.to_owned(),
            time: Time::new(seconds, 0),
        })
    }
    /// Signature at an explicit time.
    pub fn new(name: &str, email: &str, time: &Time) -> Result<Self, Error> {
        Ok(Self {
            name: name.to_owned(),
            email: email.to_owned(),
            time: *time,
        })
    }

    fn to_gix(&self) -> gix::actor::Signature {
        gix::actor::Signature {
            name: self.name.clone().into(),
            email: self.email.clone().into(),
            time: gix::date::Time::new(self.time.seconds, self.time.offset),
        }
    }
}

// =============================================================================
// ObjectType + TreeWalk enums
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectType {
    Blob,
    Tree,
    Commit,
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeWalkMode {
    PreOrder,
    PostOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeWalkResult {
    Ok,
    Skip,
    Abort,
}

// =============================================================================
// Blob + Object
// =============================================================================

pub struct Blob<'repo> {
    inner: gix::Blob<'repo>,
}

impl<'repo> Blob<'repo> {
    pub fn content(&self) -> &[u8] {
        &self.inner.data
    }
}

pub struct Object<'repo> {
    inner: gix::Object<'repo>,
}

impl<'repo> Object<'repo> {
    /// git-2's `Object::peel_to_blob`.
    pub fn peel_to_blob(self) -> Result<Blob<'repo>, Error> {
        self.inner
            .try_into_blob()
            .map(|inner| Blob { inner })
            .map_err(|e| Error::other(format!("object is not a blob: {e}")))
    }
}

// =============================================================================
// TreeEntry
// =============================================================================

/// git-2-shaped tree entry, owning its data to avoid borrow entanglement.
#[derive(Clone, Debug)]
pub struct TreeEntry {
    id: Oid,
    name: String,
    kind: ObjectType,
}

impl TreeEntry {
    pub fn id(&self) -> Oid {
        self.id
    }
    pub fn name(&self) -> Option<&str> {
        Some(&self.name)
    }
    pub fn kind(&self) -> Option<ObjectType> {
        Some(self.kind)
    }
}

fn entry_kind(mode: gix::objs::tree::EntryMode) -> ObjectType {
    use gix::objs::tree::EntryKind::*;
    match mode.kind() {
        Tree => ObjectType::Tree,
        Commit => ObjectType::Commit,
        _ => ObjectType::Blob,
    }
}

// =============================================================================
// Tree + TreeBuilder
// =============================================================================

pub struct Tree<'repo> {
    inner: gix::Tree<'repo>,
    repo: &'repo Repository,
}

impl<'repo> Tree<'repo> {
    pub fn id(&self) -> Oid {
        Oid(self.inner.id().detach())
    }

    /// git-2's `Tree::get_name`.
    pub fn get_name(&self, name: &str) -> Option<TreeEntry> {
        let entry = self.inner.find_entry(name)?;
        Some(TreeEntry {
            id: Oid(entry.oid().to_owned()),
            name: name.to_owned(),
            kind: entry_kind(entry.mode()),
        })
    }

    /// git-2's `Tree::get_path`.
    pub fn get_path(&self, path: &Path) -> Result<TreeEntry, Error> {
        let path_str = path.to_string_lossy();
        let components: Vec<&[u8]> = path_str
            .as_ref()
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.as_bytes())
            .collect();
        match self.inner.lookup_entry(components.iter().copied()) {
            Ok(Some(entry)) => Ok(TreeEntry {
                id: Oid(entry.oid().to_owned()),
                name: entry.filename().to_str_lossy().into_owned(),
                kind: entry_kind(entry.mode()),
            }),
            Ok(None) => Err(Error::not_found(format!(
                "path not found in tree: {}",
                path.display()
            ))),
            Err(e) => Err(other(e)),
        }
    }

    /// git-2's `Tree::walk` (only `PreOrder` is exercised).
    pub fn walk<C>(&self, _mode: TreeWalkMode, mut callback: C) -> Result<(), Error>
    where
        C: FnMut(&str, &TreeEntry) -> TreeWalkResult,
    {
        walk_tree(self.repo, &self.inner, "", &mut callback).map(|_| ())
    }
}

fn walk_tree<C>(
    repo: &Repository,
    tree: &gix::Tree<'_>,
    prefix: &str,
    callback: &mut C,
) -> Result<TreeWalkResult, Error>
where
    C: FnMut(&str, &TreeEntry) -> TreeWalkResult,
{
    let decoded = tree.decode().map_err(other)?;
    for entry_ref in decoded.entries.iter() {
        let name = entry_ref.filename.to_str_lossy().into_owned();
        let kind = entry_kind(entry_ref.mode);
        let entry = TreeEntry {
            id: Oid(entry_ref.oid.to_owned()),
            name: name.clone(),
            kind,
        };
        match callback(prefix, &entry) {
            TreeWalkResult::Abort => return Ok(TreeWalkResult::Abort),
            TreeWalkResult::Skip => continue,
            TreeWalkResult::Ok => {}
        }
        if kind == ObjectType::Tree {
            let subtree = repo
                .gix
                .find_tree(entry_ref.oid.to_owned())
                .map_err(other)?;
            let child_prefix = format!("{prefix}{name}/");
            if walk_tree(repo, &subtree, &child_prefix, callback)? == TreeWalkResult::Abort {
                return Ok(TreeWalkResult::Abort);
            }
        }
    }
    Ok(TreeWalkResult::Ok)
}

/// git-2's `TreeBuilder`. Collects `(name, oid, filemode)` then writes a sorted
/// `gix_object::Tree`.
pub struct TreeBuilder<'repo> {
    repo: &'repo Repository,
    entries: Vec<gix::objs::tree::Entry>,
}

impl<'repo> TreeBuilder<'repo> {
    /// git-2's `TreeBuilder::insert`. `filemode` is an octal git mode.
    pub fn insert(&mut self, name: &str, oid: Oid, filemode: i32) -> Result<(), Error> {
        let kind = match filemode {
            0o040000 => gix::objs::tree::EntryKind::Tree,
            0o100755 => gix::objs::tree::EntryKind::BlobExecutable,
            0o120000 => gix::objs::tree::EntryKind::Link,
            0o160000 => gix::objs::tree::EntryKind::Commit,
            _ => gix::objs::tree::EntryKind::Blob,
        };
        // Replace any existing entry with the same name (git-2 semantics).
        self.entries.retain(|e| e.filename != name.as_bytes());
        self.entries.push(gix::objs::tree::Entry {
            mode: kind.into(),
            filename: name.into(),
            oid: oid.inner(),
        });
        Ok(())
    }

    /// git-2's `TreeBuilder::write`.
    pub fn write(mut self) -> Result<Oid, Error> {
        self.entries.sort();
        let tree = gix::objs::Tree {
            entries: self.entries,
        };
        self.repo
            .gix
            .write_object(&tree)
            .map(|id| Oid(id.detach()))
            .map_err(other)
    }
}

// =============================================================================
// Commit
// =============================================================================

pub struct Commit<'repo> {
    inner: gix::Commit<'repo>,
    repo: &'repo Repository,
}

impl<'repo> Commit<'repo> {
    pub fn id(&self) -> Oid {
        Oid(self.inner.id().detach())
    }
    pub fn tree(&self) -> Result<Tree<'repo>, Error> {
        self.inner
            .tree()
            .map(|inner| Tree {
                inner,
                repo: self.repo,
            })
            .map_err(other)
    }
    pub fn time(&self) -> Time {
        self.inner
            .time()
            .map(Time::from)
            .unwrap_or(Time::new(0, 0))
    }
    pub fn parent_count(&self) -> usize {
        self.inner.parent_ids().count()
    }
    pub fn parent(&self, i: usize) -> Result<Commit<'repo>, Error> {
        let id = self
            .inner
            .parent_ids()
            .nth(i)
            .ok_or_else(|| Error::not_found(format!("parent {i} out of range")))?;
        self.repo.find_commit(Oid(id.detach()))
    }
    pub fn parent_id(&self, i: usize) -> Result<Oid, Error> {
        self.inner
            .parent_ids()
            .nth(i)
            .map(|id| Oid(id.detach()))
            .ok_or_else(|| Error::not_found(format!("parent {i} out of range")))
    }
}

// =============================================================================
// Reference
// =============================================================================

/// git-2-shaped reference. Owns its name + resolved target so it can outlive the
/// borrow used to read it, and drives mutations through `edit_reference`.
pub struct Reference<'repo> {
    repo: &'repo Repository,
    name: String,
    target: Option<Oid>,
}

impl<'repo> Reference<'repo> {
    pub fn name(&self) -> Option<&str> {
        Some(&self.name)
    }
    pub fn target(&self) -> Option<Oid> {
        self.target
    }
    /// git-2's `Reference::set_target` (force-updates the ref).
    pub fn set_target(&mut self, oid: Oid, message: &str) -> Result<(), Error> {
        self.repo.set_ref(&self.name, oid, true, message)?;
        self.target = Some(oid);
        Ok(())
    }
    /// git-2's `Reference::delete`.
    pub fn delete(&mut self) -> Result<(), Error> {
        self.repo.delete_ref(&self.name)
    }
    /// git-2's `Reference::peel_to_commit`.
    pub fn peel_to_commit(&mut self) -> Result<Commit<'repo>, Error> {
        let oid = self
            .target
            .ok_or_else(|| Error::not_found("reference has no direct target"))?;
        self.repo.find_commit(oid)
    }
}

/// git-2-shaped `References` iterator element used by `references_glob`.
#[derive(Clone, Debug)]
pub struct RefEntry {
    name: String,
    target: Option<Oid>,
}

impl RefEntry {
    pub fn name(&self) -> Option<&str> {
        Some(&self.name)
    }
    pub fn target(&self) -> Option<Oid> {
        self.target
    }
}

// =============================================================================
// Remote
// =============================================================================

/// git-2-shaped remote. Only the `url()` accessor is needed by callers; fetch and
/// push are driven through [`Repository`] methods.
pub struct Remote {
    url: Option<String>,
}

impl Remote {
    pub fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }
}

// =============================================================================
// Repository
// =============================================================================

pub struct Repository {
    gix: gix::Repository,
}

impl Repository {
    // ---- open / discover / init ------------------------------------------

    /// Lenient open options: tolerate invalid/unreadable global/system config
    /// values instead of hard-erroring, matching libgit-2's historical behaviour
    /// (gix defaults to strict config like `git` itself does).
    fn lenient_open_options() -> gix::open::Options {
        gix::open::Options::default().strict_config(false)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        gix::open_opts(path.as_ref(), Self::lenient_open_options())
            .map(|gix| Self { gix })
            .map_err(|e| Error::not_found(chain_message(&e)))
    }

    pub fn open_from_env() -> Result<Self, Error> {
        if let Some(dir) = std::env::var_os("GIT_DIR").filter(|d| !d.is_empty()) {
            gix::open_opts(PathBuf::from(dir), Self::lenient_open_options())
                .map(|gix| Self { gix })
                .map_err(|e| Error::not_found(e.to_string()))
        } else {
            Self::discover(".")
        }
    }

    pub fn discover(path: impl AsRef<Path>) -> Result<Self, Error> {
        gix::discover_opts(
            path.as_ref(),
            gix::discover::upwards::Options::default(),
            Self::lenient_open_options(),
        )
        .map(|gix| Self { gix })
        .map_err(|e| Error::not_found(chain_message(&e)))
    }

    pub fn init(path: impl AsRef<Path>) -> Result<Self, Error> {
        gix::init(path.as_ref())
            .map(|gix| Self { gix })
            .map_err(other)
    }

    pub fn init_bare(path: impl AsRef<Path>) -> Result<Self, Error> {
        gix::init_bare(path.as_ref())
            .map(|gix| Self { gix })
            .map_err(other)
    }

    // ---- location --------------------------------------------------------

    pub fn path(&self) -> &Path {
        self.gix.git_dir()
    }

    pub fn workdir(&self) -> Option<&Path> {
        self.gix.workdir()
    }

    /// git-2's `Repository::commondir`.
    pub fn commondir(&self) -> &Path {
        self.gix.common_dir()
    }

    // ---- object access ---------------------------------------------------

    pub fn blob(&self, data: &[u8]) -> Result<Oid, Error> {
        self.gix
            .write_blob(data)
            .map(|id| Oid(id.detach()))
            .map_err(other)
    }

    pub fn find_blob(&self, oid: Oid) -> Result<Blob<'_>, Error> {
        self.gix
            .find_blob(oid.inner())
            .map(|inner| Blob { inner })
            .map_err(other)
    }

    pub fn find_object(&self, oid: Oid, _kind: Option<ObjectType>) -> Result<Object<'_>, Error> {
        self.gix
            .find_object(oid.inner())
            .map(|inner| Object { inner })
            .map_err(other)
    }

    pub fn find_tree(&self, oid: Oid) -> Result<Tree<'_>, Error> {
        self.gix
            .find_tree(oid.inner())
            .map(|inner| Tree { inner, repo: self })
            .map_err(other)
    }

    pub fn find_commit(&self, oid: Oid) -> Result<Commit<'_>, Error> {
        self.gix
            .find_commit(oid.inner())
            .map(|inner| Commit { inner, repo: self })
            .map_err(other)
    }

    pub fn treebuilder(&self, _base: Option<&Tree<'_>>) -> Result<TreeBuilder<'_>, Error> {
        Ok(TreeBuilder {
            repo: self,
            entries: Vec::new(),
        })
    }

    /// git-2's `Repository::commit`. When `update_ref` is `Some`, the ref is
    /// force-pointed at the new commit; when `None`, only the object is written.
    pub fn commit(
        &self,
        update_ref: Option<&str>,
        author: &Signature,
        committer: &Signature,
        message: &str,
        tree: &Tree<'_>,
        parents: &[&Commit<'_>],
    ) -> Result<Oid, Error> {
        let author_sig = author.to_gix();
        let committer_sig = committer.to_gix();
        let mut author_buf = gix::date::parse::TimeBuf::default();
        let mut committer_buf = gix::date::parse::TimeBuf::default();
        let author_ref = author_sig.to_ref(&mut author_buf);
        let committer_ref = committer_sig.to_ref(&mut committer_buf);
        let tree_id = tree.inner.id().detach();
        let parent_ids: Vec<gix::ObjectId> =
            parents.iter().map(|c| c.inner.id().detach()).collect();

        let commit = self
            .gix
            .new_commit_as(committer_ref, author_ref, message, tree_id, parent_ids)
            .map_err(other)?;
        let commit_oid = Oid(commit.id().detach());

        if let Some(name) = update_ref {
            self.set_ref(name, commit_oid, true, message)?;
        }
        Ok(commit_oid)
    }

    // ---- references ------------------------------------------------------

    pub fn refname_to_id(&self, name: &str) -> Result<Oid, Error> {
        let mut reference = self.find_gix_reference(name)?;
        reference
            .peel_to_id_in_place()
            .map(|id| Oid(id.detach()))
            .map_err(other)
    }

    pub fn find_reference(&self, name: &str) -> Result<Reference<'_>, Error> {
        let mut reference = self.find_gix_reference(name)?;
        let target = reference.peel_to_id_in_place().ok().map(|id| Oid(id.detach()));
        Ok(Reference {
            repo: self,
            name: name.to_owned(),
            target,
        })
    }

    /// git-2's `Repository::reference` (create/overwrite a direct ref).
    pub fn reference(
        &self,
        name: &str,
        oid: Oid,
        force: bool,
        message: &str,
    ) -> Result<Reference<'_>, Error> {
        self.set_ref(name, oid, force, message)?;
        Ok(Reference {
            repo: self,
            name: name.to_owned(),
            target: Some(oid),
        })
    }

    /// git-2's `Repository::head`.
    pub fn head(&self) -> Result<Reference<'_>, Error> {
        let target = self.gix.head_id().ok().map(|id| Oid(id.detach()));
        Ok(Reference {
            repo: self,
            name: "HEAD".to_owned(),
            target,
        })
    }

    /// git-2's `Repository::references_glob`. Only `*` wildcards are honoured.
    pub fn references_glob(
        &self,
        glob: &str,
    ) -> Result<std::vec::IntoIter<Result<RefEntry, Error>>, Error> {
        let prefix: String = glob.split('*').next().unwrap_or("").to_owned();
        let platform = self.gix.references().map_err(other)?;
        let iter = if prefix.is_empty() {
            platform.all().map_err(other)?
        } else {
            platform.prefixed(prefix.as_str()).map_err(other)?
        };
        let mut out: Vec<Result<RefEntry, Error>> = Vec::new();
        for reference in iter {
            match reference {
                Ok(mut r) => {
                    let name = r.name().as_bstr().to_str_lossy().into_owned();
                    if !glob_matches(glob, &name) {
                        continue;
                    }
                    let target = r.peel_to_id_in_place().ok().map(|id| Oid(id.detach()));
                    out.push(Ok(RefEntry { name, target }));
                }
                Err(e) => out.push(Err(other(e))),
            }
        }
        Ok(out.into_iter())
    }

    // ---- remotes ---------------------------------------------------------

    /// git-2's `Repository::find_remote`. Reads `remote.<name>.url` fresh from the
    /// on-disk config so remotes added via [`Repository::remote`] are visible.
    pub fn find_remote(&self, name: &str) -> Result<Remote, Error> {
        match self.read_remote_url(name) {
            Some(url) => Ok(Remote { url: Some(url) }),
            None => Err(Error::not_found(format!("remote '{name}' not found"))),
        }
    }

    /// git-2's `Repository::remote` (create + persist a named remote).
    pub fn remote(&self, name: &str, url: &str) -> Result<Remote, Error> {
        self.write_remote_url(name, url)?;
        Ok(Remote {
            url: Some(url.to_owned()),
        })
    }

    /// git-2's `Repository::remote_anonymous`.
    pub fn remote_anonymous(&self, url: &str) -> Result<Remote, Error> {
        Ok(Remote {
            url: Some(url.to_owned()),
        })
    }

    /// Real fetch over gix's blocking network client.
    ///
    /// Fetches `refspec` from `url` into the local object db + tracking ref.
    pub fn fetch_refspec(&self, url: &str, refspec: &str) -> Result<(), Error> {
        // Remote-tracking refs are force-updated by convention (the configured
        // fetch refspec this shim writes is `+refs/heads/*:refs/remotes/<name>/*`).
        // gix honours the per-refspec force marker, so a refspec passed without a
        // leading `+` would refuse a non-fast-forward tracking-ref update and leave
        // the tracking ref stale. Normalise to a forced fetch to match git/libgit2.
        let forced_refspec = if refspec.starts_with('+') {
            refspec.to_owned()
        } else {
            format!("+{refspec}")
        };
        let remote = self
            .gix
            .remote_at(url)
            .map_err(other)?
            .with_refspecs(
                Some(BStr::new(forced_refspec.as_bytes())),
                gix::remote::Direction::Fetch,
            )
            .map_err(other)?;
        let connection = remote
            .connect(gix::remote::Direction::Fetch)
            .map_err(other)?;
        let prepare = connection
            .prepare_fetch(gix::progress::Discard, Default::default())
            .map_err(other)?;
        let interrupt = AtomicBool::new(false);
        prepare
            .receive(gix::progress::Discard, &interrupt)
            .map_err(other)?;
        Ok(())
    }

    /// Push `refspecs` to the remote at `url`.
    ///
    /// **Local** remotes (a filesystem path or a `file://` URL) are pushed in
    /// process: the objects the target repo lacks are copied into its object
    /// database and each destination ref is updated with fast-forward + lock
    /// semantics. No `git` subprocess is spawned and no network protocol runs.
    ///
    /// **Network** remotes (`http(s)`/`ssh`/`git`/scp-like `user@host:path`)
    /// are delegated to the system `git` binary — see [`Self::subprocess_push`].
    /// gix 0.86 has no high-level push or `gix-protocol` send-pack helper
    /// (upstream #306), so there is no in-process path for these.
    pub fn push_refspecs(&self, url: &str, refspecs: &[String]) -> Result<(), Error> {
        match local_repo_path(url) {
            Some(path) => self.local_push(&path, refspecs),
            None => self.subprocess_push(url, refspecs),
        }
    }

    /// Push `refspecs` to a network remote by invoking the system `git`.
    ///
    /// # Why a subprocess
    /// gix 0.86 implements no high-level push and no `gix-protocol` send-pack
    /// helper (upstream #306). Reintroducing `git-2`/libgit-2 would restore the
    /// C toolchain this shim exists to remove, and would break the Windows /
    /// Android-Termux builds that motivated the fork. Delegating only this one
    /// operation keeps the build pure-Rust and, just as importantly, inherits
    /// the user's credential setup (credential helpers, netrc, SSH agent)
    /// rather than reimplementing it.
    ///
    /// # Behaviour
    /// Refspecs are passed through verbatim; `git push` accepts the same
    /// `[+]<src>:<dst>` syntax [`parse_refspec`] handles, including the empty
    /// source (`:dst`) form meaning delete.
    ///
    /// `GIT_TERMINAL_PROMPT=0` is set deliberately. stdout/stderr are captured
    /// for error reporting, so an interactive credential prompt would be
    /// invisible and would hang the caller indefinitely — a particularly bad
    /// failure mode under the daemon. Failing fast with git's own
    /// "could not read Username" message is the better trade. Users with a
    /// configured credential helper, netrc, or SSH agent are unaffected.
    ///
    /// A missing `git` yields [`ErrorCode::PushUnsupported`], preserving the
    /// error code callers already gate on.
    ///
    /// # Windows: no popup window
    /// The child is spawned through [`crate::proc::output_before_deadline`],
    /// which applies [`crate::proc::dun_popup`] (`CREATE_NO_WINDOW`) itself.
    /// Without it the daemon — which owns no console — would make Windows
    /// allocate `git.exe` a fresh console window on every debounced auto-sync
    /// (kopitiam#29). Safe to suppress precisely because stdout/stderr are
    /// captured and `GIT_TERMINAL_PROMPT=0` rules out any interactive prompt
    /// that would need a visible window.
    ///
    /// # Two independent stall guards (kopitiam#25)
    /// A `git push` that stalls used to hang the daemon's sync lane forever,
    /// which in turn hung `bn sync` forever. Both belts are worn:
    ///
    /// 1. **git's own low-speed abort** — `http.lowSpeedLimit` /
    ///    `http.lowSpeedTime`, passed with `-c` so we never touch the user's
    ///    config. This is the *better* mechanism for the common case: it is
    ///    curl aborting a transfer it knows has gone quiet, so we get git's own
    ///    error message rather than a corpse and a guess.
    /// 2. **A process watchdog** — [`PUSH_WATCHDOG`]. Covers everything the
    ///    low-speed check cannot see: ssh transports (not http, so the
    ///    `http.*` settings do not apply), DNS, and any hang before a byte of
    ///    transfer starts.
    fn subprocess_push(&self, url: &str, refspecs: &[String]) -> Result<(), Error> {
        let mut cmd = std::process::Command::new("git");
        cmd.arg(format!("--git-dir={}", self.path().display()));
        if let Some(workdir) = self.workdir() {
            cmd.arg(format!("--work-tree={}", workdir.display()));
        }
        // `-c` must come before the subcommand. These only bind http(s)
        // transports; on ssh/file remotes git ignores them and the watchdog
        // below is the only guard.
        cmd.arg("-c")
            .arg(format!("http.lowSpeedLimit={PUSH_LOW_SPEED_LIMIT_BYTES}"))
            .arg("-c")
            .arg(format!("http.lowSpeedTime={PUSH_LOW_SPEED_TIME_SECS}"));
        cmd.arg("push").arg(url).args(refspecs);
        cmd.env("GIT_TERMINAL_PROMPT", "0");

        let outcome = crate::proc::output_before_deadline(&mut cmd, PUSH_WATCHDOG).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::new(
                    ErrorCode::PushUnsupported,
                    format!(
                        "push to non-local remote '{url}' requires the `git` binary, which was \
                         not found on PATH: gix 0.86 has no high-level push / send-pack \
                         (upstream #306), so only local (file://) push is in-process"
                    ),
                )
            } else {
                Error::new(
                    ErrorCode::Other,
                    format!("failed to run `git push` for remote '{url}': {e}"),
                )
            }
        })?;

        let output = match outcome {
            crate::proc::DeadlineOutcome::Finished(output) => output,
            crate::proc::DeadlineOutcome::TimedOut {
                waited,
                stdout,
                stderr,
            } => {
                // Whatever git managed to say before we shot it is the only
                // clue the user gets, so take it from either stream — progress
                // reporting lands on stderr, but a stalled `push` can leave its
                // last words on stdout.
                let stderr = String::from_utf8_lossy(&stderr);
                let tail = match stderr.trim() {
                    "" => String::from_utf8_lossy(&stdout).trim().to_string(),
                    text => text.to_string(),
                };
                let tail = if tail.is_empty() {
                    "(git printed nothing before it was killed)".to_string()
                } else {
                    tail
                };
                return Err(Error::new(
                    ErrorCode::Other,
                    format!(
                        "`git push` to '{url}' was killed after {}s with no result: {tail}",
                        waited.as_secs()
                    ),
                ));
            }
        };

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        let detail = if detail.is_empty() {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        } else {
            detail.to_string()
        };
        Err(Error::new(
            ErrorCode::Other,
            format!("`git push` to '{url}' failed ({}): {detail}", output.status),
        ))
    }

    /// Push every refspec to a local (on-disk) target repository.
    fn local_push(&self, remote_path: &Path, refspecs: &[String]) -> Result<(), Error> {
        for spec in refspecs {
            self.push_one_refspec(remote_path, spec)?;
        }
        Ok(())
    }

    /// Apply one `[+]<src>:<dst>` refspec against the on-disk target repo.
    fn push_one_refspec(&self, remote_path: &Path, spec: &str) -> Result<(), Error> {
        let (force, src, dst) = parse_refspec(spec)?;

        let remote = open_local_repo(remote_path)?;

        // Empty source ("`:dst`") means: delete `dst` on the remote.
        if src.is_empty() {
            return push_delete_ref(&remote, dst);
        }

        let new_oid = self.resolve_push_source(src)?;
        let old_oid = remote_ref_target(&remote, dst);

        // Copy every object reachable from `new_oid` that the remote lacks. Done
        // first so the remote can compute ancestry against its existing tip.
        copy_missing_objects(&self.gix, &remote, new_oid)?;

        // Re-open so the ancestry check + ref update observe the just-written
        // loose objects through a fresh odb view.
        let remote = open_local_repo(remote_path)?;

        if !force
            && let Some(old) = old_oid
            && old != new_oid
            && !is_fast_forward(&remote, old, new_oid)
        {
            return Err(Error::other(format!(
                "failed to push ref '{dst}': non-fast-forward update rejected (fetch first)"
            )));
        }

        write_remote_ref(&remote, dst, new_oid, old_oid)
    }

    /// Resolve a push source (a ref name, or a raw hex object id) to an [`Oid`]
    /// in *this* (local) repository.
    fn resolve_push_source(&self, src: &str) -> Result<Oid, Error> {
        match self.find_gix_reference(src) {
            Ok(mut r) => r.peel_to_id().map(|id| Oid(id.detach())).map_err(other),
            Err(_) => Oid::from_str(src),
        }
    }

    // ---- graph -----------------------------------------------------------

    pub fn merge_base(&self, one: Oid, two: Oid) -> Result<Oid, Error> {
        match self.gix.merge_base(one.inner(), two.inner()) {
            Ok(id) => Ok(Oid(id.detach())),
            Err(e) => {
                if is_merge_base_not_found(&e) {
                    Err(Error::not_found(e.to_string()))
                } else {
                    Err(other(e))
                }
            }
        }
    }

    /// git-2's `graph_descendant_of(commit, ancestor)` — is `ancestor` a strict
    /// ancestor of `commit`?
    pub fn graph_descendant_of(&self, commit: Oid, ancestor: Oid) -> Result<bool, Error> {
        if commit == ancestor {
            return Ok(false);
        }
        match self.gix.merge_base(commit.inner(), ancestor.inner()) {
            Ok(base) => Ok(Oid(base.detach()) == ancestor),
            Err(e) => {
                if is_merge_base_not_found(&e) {
                    Ok(false)
                } else {
                    Err(other(e))
                }
            }
        }
    }

    // ---- internals -------------------------------------------------------

    fn find_gix_reference(&self, name: &str) -> Result<gix::Reference<'_>, Error> {
        use gix::reference::find::existing::Error as FindErr;
        match self.gix.find_reference(name) {
            Ok(r) => Ok(r),
            Err(FindErr::NotFound { .. }) => {
                Err(Error::not_found(format!("reference '{name}' not found")))
            }
            Err(e) => Err(other(e)),
        }
    }

    fn set_ref(&self, name: &str, oid: Oid, force: bool, message: &str) -> Result<(), Error> {
        use gix::refs::transaction::PreviousValue;
        let constraint = if force {
            PreviousValue::Any
        } else {
            PreviousValue::MustNotExist
        };
        self.gix
            .reference(name, oid.inner(), constraint, message)
            .map(|_| ())
            .map_err(map_ref_edit_err)
    }

    fn delete_ref(&self, name: &str) -> Result<(), Error> {
        use gix::refs::transaction::{Change, RefEdit, RefLog};
        use gix::refs::{Target, transaction::PreviousValue};
        let full: gix::refs::FullName = name
            .try_into()
            .map_err(|e| Error::other(format!("invalid ref name '{name}': {e}")))?;
        let edit = RefEdit {
            change: Change::Delete {
                expected: PreviousValue::Any,
                log: RefLog::AndReference,
            },
            name: full,
            deref: false,
        };
        // Deletion target is ignored for Change::Delete.
        let _ = Target::Object(Oid::zero().inner());
        self.gix
            .edit_reference(edit)
            .map(|_| ())
            .map_err(map_ref_edit_err)
    }

    // Remote config on disk (git config INI). We manage it ourselves so a live
    // handle observes remotes we add without a reload.

    fn config_path(&self) -> PathBuf {
        self.gix.git_dir().join("config")
    }

    fn read_remote_url(&self, name: &str) -> Option<String> {
        let text = std::fs::read_to_string(self.config_path()).ok()?;
        let header = format!("[remote \"{name}\"]");
        let mut in_section = false;
        let mut found: Option<String> = None;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_section = trimmed == header;
                continue;
            }
            if in_section
                && let Some(rest) = trimmed.strip_prefix("url")
            {
                let rest = rest.trim_start();
                if let Some(value) = rest.strip_prefix('=') {
                    found = Some(value.trim().to_owned());
                }
            }
        }
        found
    }

    fn write_remote_url(&self, name: &str, url: &str) -> Result<(), Error> {
        use std::io::Write;
        let path = self.config_path();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::other(format!("open git config: {e}")))?;
        write!(
            file,
            "\n[remote \"{name}\"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/{name}/*\n"
        )
        .map_err(|e| Error::other(format!("write git config: {e}")))?;
        Ok(())
    }
}

// =============================================================================
// helpers
// =============================================================================

// ---- push helpers -----------------------------------------------------------

/// Decide whether `url` names a **local** (on-disk) repository, returning its
/// filesystem path if so. Network schemes (`http(s)`/`ssh`/`git`) and scp-like
/// `user@host:path` return `None`.
fn local_repo_path(url: &str) -> Option<PathBuf> {
    if let Some(rest) = url.strip_prefix("file://") {
        // `file://<authority>/<path>` or `file:///<path>` (empty authority).
        let path_part = match rest.find('/') {
            Some(idx) => &rest[idx..], // keep the leading '/'
            None => rest,
        };
        let trimmed = path_part.trim_start_matches('/');
        #[cfg(windows)]
        {
            // `file:///C:/repo` -> `C:/repo`; otherwise keep the rooted path.
            if trimmed.as_bytes().get(1) == Some(&b':') {
                return Some(PathBuf::from(trimmed));
            }
            return Some(PathBuf::from(path_part));
        }
        #[cfg(not(windows))]
        {
            return Some(PathBuf::from(format!("/{trimmed}")));
        }
    }
    // Any explicit `scheme://` is a network remote.
    if url.contains("://") {
        return None;
    }
    // A bare path that resolves to a directory is local; scp-like `user@host:path`
    // does not resolve to a directory and so is treated as a network remote.
    let path = Path::new(url);
    if path.is_dir() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

/// Open a local repository as a push target, tolerating lenient config.
fn open_local_repo(path: &Path) -> Result<gix::Repository, Error> {
    gix::open_opts(path, Repository::lenient_open_options()).map_err(|e| {
        Error::other(format!(
            "open push target '{}': {}",
            path.display(),
            chain_message(&e)
        ))
    })
}

/// Split `[+]<src>:<dst>` into `(force, src, dst)`. A colon-less spec pushes a
/// ref to the same name on the remote.
fn parse_refspec(spec: &str) -> Result<(bool, &str, &str), Error> {
    let (force, body) = match spec.strip_prefix('+') {
        Some(rest) => (true, rest),
        None => (false, spec),
    };
    let (src, dst) = match body.split_once(':') {
        Some((s, d)) => (s, d),
        None => (body, body),
    };
    if dst.is_empty() {
        return Err(Error::other(format!(
            "invalid refspec '{spec}': empty destination"
        )));
    }
    Ok((force, src, dst))
}

/// Current target of `name` on the remote, if the ref exists and resolves.
fn remote_ref_target(remote: &gix::Repository, name: &str) -> Option<Oid> {
    let mut reference = remote.find_reference(name).ok()?;
    reference.peel_to_id().ok().map(|id| Oid(id.detach()))
}

/// Copy every object reachable from `tip` that `remote` lacks from `local` into
/// `remote`'s object database. Relies on git connectivity: if the remote already
/// has an object it has that object's full closure, so traversal is pruned there.
fn copy_missing_objects(
    local: &gix::Repository,
    remote: &gix::Repository,
    tip: Oid,
) -> Result<(), Error> {
    use gix::objs::{CommitRefIter, Kind, TagRefIter, TreeRefIter, Write, tree::EntryKind};

    let hash = gix::hash::Kind::Sha1;
    let mut stack: Vec<gix::ObjectId> = vec![tip.inner()];
    let mut seen: std::collections::HashSet<gix::ObjectId> = std::collections::HashSet::new();

    while let Some(oid) = stack.pop() {
        if !seen.insert(oid) {
            continue;
        }
        if remote.has_object(oid) {
            continue;
        }
        let obj = local
            .find_object(oid)
            .map_err(|e| Error::other(format!("read object {oid} for push: {e}")))?;
        let kind = obj.kind;
        match kind {
            Kind::Commit => {
                let mut it = CommitRefIter::from_bytes(&obj.data, hash);
                if let Ok(tree) = it.tree_id() {
                    stack.push(tree);
                }
                for parent in CommitRefIter::from_bytes(&obj.data, hash).parent_ids() {
                    stack.push(parent);
                }
            }
            Kind::Tag => {
                if let Ok(target) = TagRefIter::from_bytes(&obj.data, hash).target_id() {
                    stack.push(target);
                }
            }
            Kind::Tree => {
                for entry in TreeRefIter::from_bytes(&obj.data, hash) {
                    let entry =
                        entry.map_err(|e| Error::other(format!("decode tree {oid}: {e}")))?;
                    // Skip gitlink (submodule) entries: the remote never has them.
                    if entry.mode.kind() == EntryKind::Commit {
                        continue;
                    }
                    stack.push(entry.oid.to_owned());
                }
            }
            Kind::Blob => {}
        }
        remote
            .objects
            .write_buf(kind, &obj.data)
            .map_err(|e| Error::other(format!("write object {oid} to push target: {e}")))?;
    }
    Ok(())
}

/// Is advancing `old` to `new` a fast-forward (i.e. is `old` an ancestor of
/// `new`)? Computed on the receiving repo, which — after [`copy_missing_objects`]
/// — has both objects and their history.
fn is_fast_forward(remote: &gix::Repository, old: Oid, new: Oid) -> bool {
    if old == new {
        return true;
    }
    match remote.merge_base(old.inner(), new.inner()) {
        Ok(base) => base.detach() == old.inner(),
        Err(_) => false,
    }
}

/// Update `name` on the remote to `new`, constraining on the previously observed
/// value so a concurrent move (or a held `.lock`) is surfaced.
fn write_remote_ref(
    remote: &gix::Repository,
    name: &str,
    new: Oid,
    old: Option<Oid>,
) -> Result<(), Error> {
    use gix::refs::Target;
    use gix::refs::transaction::PreviousValue;
    let constraint = match old {
        Some(old) => PreviousValue::MustExistAndMatch(Target::Object(old.inner())),
        None => PreviousValue::MustNotExist,
    };
    remote
        .reference(name, new.inner(), constraint, "push: update reference")
        .map(|_| ())
        .map_err(|e| map_push_ref_edit_err(name, e))
}

/// Delete `name` on the remote.
fn push_delete_ref(remote: &gix::Repository, name: &str) -> Result<(), Error> {
    use gix::refs::transaction::{Change, PreviousValue, RefEdit, RefLog};
    let full: gix::refs::FullName = name
        .try_into()
        .map_err(|e| Error::other(format!("invalid ref name '{name}': {e}")))?;
    let edit = RefEdit {
        change: Change::Delete {
            expected: PreviousValue::Any,
            log: RefLog::AndReference,
        },
        name: full,
        deref: false,
    };
    remote
        .edit_reference(edit)
        .map(|_| ())
        .map_err(|e| map_push_ref_edit_err(name, e))
}

/// Map a remote ref-edit failure into a git-2-shaped [`Error`]. Lock contention
/// and previous-value mismatches are normalised to messages the sync/publish
/// retry classifiers (`is_retryable_push_error_message`, `is_non_fast_forward`)
/// recognise, so a locked or moved remote ref is retried rather than fatal.
fn map_push_ref_edit_err(dst: &str, e: gix::reference::edit::Error) -> Error {
    let msg = chain_message(&e);
    let lower = msg.to_lowercase();
    if lower.contains("lock") {
        Error::locked(format!(
            "cannot lock ref '{dst}': failed to lock file: {msg}"
        ))
    } else if lower.contains("expected")
        || lower.contains("existing value")
        || lower.contains("should have content")
        || lower.contains("supposed to exist")
    {
        // Previous-value constraint failed: the remote ref moved under us.
        Error::other(format!(
            "failed to update ref '{dst}': non-fast-forward (fetch first): {msg}"
        ))
    } else {
        Error::other(format!("failed to update ref '{dst}': {msg}"))
    }
}

fn map_ref_edit_err(e: gix::reference::edit::Error) -> Error {
    let msg = e.to_string();
    let lower = msg.to_lowercase();
    if lower.contains("lock") {
        Error::locked(msg)
    } else {
        Error::other(msg)
    }
}

fn is_merge_base_not_found(e: &gix::repository::merge_base::Error) -> bool {
    matches!(e, gix::repository::merge_base::Error::NotFound { .. })
}

/// Minimal glob matcher supporting `*` (matches any run, including `/`).
fn glob_matches(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut pos = 0usize;
    // Anchor the leading literal.
    if let Some(first) = parts.first() {
        if !text[pos..].starts_with(first) {
            return false;
        }
        pos += first.len();
    }
    // Middle literals in order.
    for part in &parts[1..parts.len() - 1] {
        if part.is_empty() {
            continue;
        }
        match text[pos..].find(part) {
            Some(idx) => pos += idx + part.len(),
            None => return false,
        }
    }
    // Anchor the trailing literal.
    if let Some(last) = parts.last() {
        if last.is_empty() {
            return true;
        }
        text[pos..].ends_with(last) && text.len() - pos >= last.len()
    } else {
        true
    }
}

/// Message for the one remaining gated case: a *network* push attempted on a
/// host with no `git` binary on `PATH`. Local (`file://`) push runs in-process
/// via gix; network push delegates to `git`, because gix 0.86 has no
/// high-level push / `gix-protocol` send-pack (upstream #306).
pub const PUSH_UNSUPPORTED_MSG: &str =
    "push to non-local remote requires the `git` binary, which was not found on PATH: gix 0.86 has no high-level push / send-pack (upstream #306)";
