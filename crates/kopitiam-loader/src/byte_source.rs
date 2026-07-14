//! Backing storage for a loaded model's bytes.
//!
//! A model file is opened once, up front, and every tensor's raw bytes are
//! served as a slice into that single buffer. [`ByteSource`] is the enum
//! that decides *how* those bytes are held in memory.

use std::fs::File;
use std::path::Path;

use kopitiam_core::{Error, Result};

/// Either a memory-mapped file or a fully-read buffer.
///
/// # Why prefer `mmap`
///
/// A 7B-parameter model in `f16` is already ~14 GiB on disk. Reading that
/// into a `Vec<u8>` means the OS page cache holds one copy and the process
/// heap holds a second, doubling peak memory for no benefit — the loader
/// never needs the *whole* file resident at once, only whichever tensor a
/// caller is currently consuming. Memory-mapping lets the OS page tensor
/// data in on first touch and evict it under memory pressure, which a
/// `Vec<u8>` cannot do.
///
/// # Why not *only* `mmap`
///
/// `mmap`-ing a file is refused by some platforms/filesystems for empty
/// files, and [`memmap2::Mmap::map`] is `unsafe`: the OS gives no guarantee
/// the file won't be truncated or overwritten by another process while it
/// is mapped, which would turn a read into a segfault or torn data rather
/// than a clean I/O error. Both are edge cases worth tolerating rather than
/// treating as load failures, so [`ByteSource::open`] falls back to reading
/// the file fully into a `Vec<u8>` whenever `mmap` cannot be used. This
/// crate accepts that reduced (but well-understood and documented) safety
/// margin in exchange for not doubling memory on the common path; a future
/// caller that cannot accept it at all is free to force the `Owned` path by
/// constructing one directly once such a constructor is needed.
pub(crate) enum ByteSource {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl ByteSource {
    /// Opens `path`, preferring a memory map and falling back to a full
    /// read if mapping is not possible (e.g. a zero-length file, or a
    /// filesystem that does not support `mmap`).
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;

        // SAFETY: this crate does not itself mutate or truncate the file
        // after mapping it, and mapping a file that another process
        // concurrently truncates is a documented, accepted risk (see the
        // type-level doc comment above) rather than something this loader
        // can prevent from a single-process API.
        match unsafe { memmap2::Mmap::map(&file) } {
            Ok(mmap) => Ok(Self::Mmap(mmap)),
            Err(_) => {
                // Re-open rather than reuse `file`: on the fallback path we
                // want a plain sequential read from the start, and `file`'s
                // cursor position after a failed mmap attempt is not part
                // of any documented contract worth relying on.
                let bytes = std::fs::read(path)?;
                Ok(Self::Owned(bytes))
            }
        }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(mmap) => mmap,
            Self::Owned(bytes) => bytes,
        }
    }

    /// Returns the `[offset, offset + len)` byte range, bounds-checked
    /// against the actual buffer length.
    ///
    /// Every tensor byte range in a model file is an offset and length read
    /// from untrusted input. Centralizing the bounds check here means a
    /// hostile or truncated file can never turn into an out-of-bounds slice
    /// panic — it always becomes an [`Error::MalformedModel`].
    pub(crate) fn slice(&self, format: &'static str, offset: usize, len: usize) -> Result<&[u8]> {
        let bytes = self.as_slice();
        let end = offset.checked_add(len).ok_or_else(|| Error::MalformedModel {
            format,
            reason: format!("tensor byte range {offset}..+{len} overflows"),
        })?;
        bytes.get(offset..end).ok_or_else(|| Error::MalformedModel {
            format,
            reason: format!(
                "tensor byte range {offset}..{end} is out of bounds for a {}-byte file",
                bytes.len()
            ),
        })
    }
}
