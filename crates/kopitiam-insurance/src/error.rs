//! Errors.
//!
//! Note what is *not* here: there is no error for "could not understand a
//! clause". That is not an error, it is a **result** — see [`crate::Anomaly`].
//! An insurance document that this crate only partly understands still yields a
//! [`crate::PolicyDocument`]; the parts it could not understand are reported,
//! with their text, rather than aborting the ingestion or being quietly
//! dropped.
//!
//! The errors that *are* here are the ones where continuing would mean lying:
//! a PDF that will not open, and a citation that cannot be traced back to the
//! document it claims to come from.

use crate::provenance::ProvenanceError;

/// Something went wrong that makes the result untrustworthy.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The PDF could not be read or parsed.
    #[error("could not read the PDF")]
    Pdf(#[from] kopitiam_pdf::ExtractError),

    /// A citation could not be traced back to its source.
    ///
    /// This should be impossible by construction — every quotation this crate
    /// mints is a substring of the clause it came from. If it happens, the
    /// extractor has produced text the document does not contain, which is the
    /// exact failure the provenance model exists to catch. It is an error, and
    /// it is loud.
    #[error("provenance could not be established")]
    Provenance(#[from] ProvenanceError),
}
