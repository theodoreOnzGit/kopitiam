//! The local error type for `kopitiam-ocr`.
//!
//! This is **not** a line-for-line port of any single Tesseract file: Tesseract's
//! (de)serialisers and the `.traineddata` container reader signal failure by
//! returning `bool` (`false`) and, for hard invariants, by aborting via
//! `ASSERT_HOST` (see `src/ccutil/serialis.cpp` / `src/ccutil/tessdatamanager.cpp`
//! at Tesseract commit db0ec62, Apache-2.0, © Google Inc. and Hewlett-Packard
//! Ltd.). Rust has a better vocabulary for that, so — mirroring the discipline of
//! `kopitiam-pdf::mupdf::error` but kept entirely local to this crate — every
//! `return false` failure path on the read side becomes a typed [`Error`] carried
//! in a [`Result`]. Attribution for the translated Tesseract sources is recorded
//! per docs/ACKNOWLEDGEMENTS.md and at the point of use in each ported module.
//!
//! The taxonomy is deliberately small: the container read path has exactly three
//! ways to fail, and each Tesseract `false`/assert maps onto one of them.

use core::fmt;

// ---------------------------------------------------------------------------
// Error taxonomy
// ---------------------------------------------------------------------------

/// The category of a `kopitiam-ocr` failure.
///
/// Each variant corresponds to a distinct family of `return false` (or
/// `ASSERT_HOST`) sites in the Tesseract read path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// A read ran past the end of the buffer.
    ///
    /// Tesseract's `TFile::FRead` clamps a short read to the bytes remaining and
    /// returns fewer items than requested; its `DeSerialize` wrappers then compare
    /// `== count` and yield `false` (`serialis.cpp:239`, `serialis.h:89`). This is
    /// that mismatch reified.
    UnexpectedEof,
    /// The container structure is malformed.
    ///
    /// A component offset points outside the file, an entry has a negative size,
    /// or an entry's successor offset is out of range — every `return false`
    /// inside `TessdataManager::LoadMemBuffer`'s offset-table walk
    /// (`tessdatamanager.cpp:136`).
    Format,
    /// A hard bound was exceeded.
    ///
    /// The container declares more than `kMaxNumTessdataEntries` (1000) entries
    /// even after endianness correction (`tessdatamanager.cpp:128`), or a
    /// length-prefixed vector exceeds Tesseract's 50,000,000-element guard
    /// (`serialis.cpp:102`).
    Limit,
}

impl ErrorKind {
    /// A short, stable lowercase name for the category.
    pub const fn name(self) -> &'static str {
        match self {
            ErrorKind::UnexpectedEof => "unexpected-eof",
            ErrorKind::Format => "format",
            ErrorKind::Limit => "limit",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Error value
// ---------------------------------------------------------------------------

/// A `kopitiam-ocr` error: a [`ErrorKind`] category plus a human-readable message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    /// Construct an error of `kind` with an already-formatted `message`.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
        }
    }

    /// The error category.
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The formatted message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// A read ran past the end of the buffer ([`ErrorKind::UnexpectedEof`]).
    pub fn unexpected_eof(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::UnexpectedEof, message)
    }

    /// The container structure is malformed ([`ErrorKind::Format`]).
    pub fn format(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Format, message)
    }

    /// A hard bound was exceeded ([`ErrorKind::Limit`]).
    pub fn limit(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Limit, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} error: {}", self.kind.name(), self.message)
    }
}

impl std::error::Error for Error {}

/// The crate-wide fallible result. `Ok` is Tesseract's `true`; an [`Error`] `Err`
/// is one of its `return false` failure paths.
pub type Result<T> = core::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_set_the_right_kind() {
        assert_eq!(Error::unexpected_eof("x").kind(), ErrorKind::UnexpectedEof);
        assert_eq!(Error::format("x").kind(), ErrorKind::Format);
        assert_eq!(Error::limit("x").kind(), ErrorKind::Limit);
    }

    #[test]
    fn display_is_category_then_message() {
        assert_eq!(
            Error::format("offset 999 past end 245").to_string(),
            "format error: offset 999 past end 245"
        );
        assert_eq!(
            Error::unexpected_eof("wanted 4 bytes").to_string(),
            "unexpected-eof error: wanted 4 bytes"
        );
    }

    #[test]
    fn message_is_preserved() {
        let e = Error::limit(format!("{} > {}", 2000, 1000));
        assert_eq!(e.message(), "2000 > 1000");
    }

    #[test]
    fn usable_as_std_error_in_result() {
        fn may_fail(ok: bool) -> Result<u32> {
            if ok {
                Ok(7)
            } else {
                Err(Error::unexpected_eof("eof"))
            }
        }
        assert_eq!(may_fail(true).unwrap(), 7);
        let err = may_fail(false).unwrap_err();
        let dyn_err: &dyn std::error::Error = &err;
        assert_eq!(dyn_err.to_string(), "unexpected-eof error: eof");
        assert!(dyn_err.source().is_none());
    }
}
