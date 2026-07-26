//! Windows stand-in for `nix::fcntl::Flock`.
//!
//! The store guards concurrent access with an exclusive advisory lock on the
//! lock file. On unix that is `flock(2)` via `nix::fcntl::Flock`. On Windows the
//! equivalent is `LockFileEx`, exposed portably by the `fs2` crate. This shim
//! reproduces the small slice of nix's `Flock` API the store uses:
//!
//!   * `Flock::lock(file, FlockArg::LockExclusive) -> Result<Flock<File>, (File, i32)>`
//!   * `Deref`/`DerefMut` to the underlying file (so `read`/`seek`/`metadata` work)
//!   * unlock-on-drop
//!
//! The lock is released when the `Flock` is dropped, matching nix's semantics.

use std::ops::{Deref, DerefMut};

use fs2::FileExt;

/// The subset of `nix::fcntl::FlockArg` the store uses. Blocking exclusive lock.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FlockArg {
    LockExclusive,
}

/// An exclusive advisory lock held over `inner`, released on drop.
pub(crate) struct Flock<T: FileExt> {
    inner: Option<T>,
}

impl<T: FileExt> Flock<T> {
    /// Take an exclusive (blocking) lock. On failure the file is returned
    /// alongside the raw OS error code, matching nix's `(file, errno)` shape so
    /// the shared call site (`errno as i32`) is identical across platforms.
    pub(crate) fn lock(inner: T, _arg: FlockArg) -> Result<Self, (T, i32)> {
        match FileExt::lock_exclusive(&inner) {
            Ok(()) => Ok(Self { inner: Some(inner) }),
            Err(err) => {
                let code = err.raw_os_error().unwrap_or(1);
                Err((inner, code))
            }
        }
    }
}

impl<T: FileExt> Deref for Flock<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.as_ref().expect("flock inner present until drop")
    }
}

impl<T: FileExt> DerefMut for Flock<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.inner.as_mut().expect("flock inner present until drop")
    }
}

impl<T: FileExt> Drop for Flock<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = FileExt::unlock(&inner);
        }
    }
}
