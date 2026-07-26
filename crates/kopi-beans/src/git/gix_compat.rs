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
//! - **Push** is GATED: gix 0.86 exposes no high-level push (upstream #306),
//!   so [`Repository::push_refspecs`] returns [`ErrorCode::PushUnsupported`].
//!   A send-pack shim over `gix-protocol`/`gix-transport` is a follow-up.
//! - Remote *configuration* (create/read `remote.<name>.url`) is managed by
//!   this shim directly against the on-disk git config, so a freshly opened
//!   handle observes remotes added through it without a reload.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use gix::bstr::{BStr, ByteSlice};

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
        let remote = self
            .gix
            .remote_at(url)
            .map_err(other)?
            .with_refspecs(Some(BStr::new(refspec.as_bytes())), gix::remote::Direction::Fetch)
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

    /// GATED push. gix 0.86 has no high-level push (upstream #306); a send-pack
    /// shim is a follow-up. Returns [`ErrorCode::PushUnsupported`].
    pub fn push_refspecs(&self, _url: &str, _refspecs: &[String]) -> Result<(), Error> {
        Err(Error::new(
            ErrorCode::PushUnsupported,
            "push over gix not yet implemented — gix 0.86 has no high-level push; send-pack shim pending",
        ))
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

/// Push-gated error surfaced to sync/publish callers as a convenience.
pub const PUSH_UNSUPPORTED_MSG: &str =
    "push over gix not yet implemented — gix 0.86 has no high-level push; send-pack shim pending";
