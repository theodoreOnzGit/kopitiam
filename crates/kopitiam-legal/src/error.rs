//! Errors.
//!
//! Note what is *not* here: there is no "could not determine meaning" error,
//! because determining meaning is not this crate's job. Failures here are
//! all failures to *locate* or *structure* source material. When the source
//! itself is ambiguous or contradictory, that is not an error at all — it is
//! an [`crate::Anomaly`], reported alongside the material rather than
//! replacing it.

use thiserror::Error;

use crate::ProvisionId;

/// Everything that can go wrong while structuring a legal document.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum LegalError {
    /// A mandatory provenance component was absent or empty. See
    /// [`crate::provenance`] for why this is a hard error rather than a
    /// tolerated gap.
    #[error("missing mandatory provenance: {what} (an extracted item without provenance is an unsourced claim about the law, and is not representable in this crate)")]
    MissingProvenance { what: &'static str },

    #[error("invalid date: {detail}")]
    InvalidDate { detail: String },

    #[error("invalid in-force window: {detail}")]
    InvalidValidity { detail: String },

    /// A provision label could not be parsed into a hierarchy. The label is
    /// preserved so the caller can still show the reader what the document
    /// actually said.
    #[error("could not parse provision label {label:?} into a hierarchy")]
    UnparseableNumbering { label: String },

    /// A provision was requested that this instrument does not contain.
    #[error("instrument does not contain provision {id}")]
    NoSuchProvision { id: ProvisionId },

    /// The same provision id was defined twice in one instrument.
    #[error("duplicate provision {id} in one instrument")]
    DuplicateProvision { id: ProvisionId },

    #[error("ingest failed: {0}")]
    Ingest(String),

    #[error("PDF extraction failed: {0}")]
    Pdf(String),
}
