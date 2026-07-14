//! Errors.
//!
//! Note what is **not** here: there is no error for "could not parse a
//! reference". That is not an error, it is a **result** — see
//! [`crate::ParsedReference::Unparsed`] and [`crate::Anomaly`]. A document this
//! crate only partly understands still yields a [`crate::Bibliography`]; the
//! parts it could not understand are reported, with their text, rather than
//! aborting the extraction or being quietly dropped.
//!
//! The errors that *are* here are the ones where continuing would mean lying: a
//! PDF that will not open, a `.bib` file whose structure cannot be read without
//! inventing some, and a citation that cannot be traced back to a source.

use crate::bibtex::BibtexError;
use crate::provenance::ProvenanceError;

/// Something went wrong that makes the result untrustworthy.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The PDF could not be read or parsed.
    #[error("could not read the PDF")]
    Pdf(#[from] kopitiam_pdf::ExtractError),

    /// A `.bib` file could not be parsed without guessing at its structure.
    #[error("could not parse the BibTeX")]
    Bibtex(#[from] BibtexError),

    /// A citation could not be traced back to its source.
    ///
    /// Should be impossible by construction — every provenance this crate mints
    /// comes from a page and a non-empty string. If it happens, the extractor
    /// has produced a reference the document does not contain, which is the
    /// exact failure the provenance model exists to catch. It is an error, and
    /// it is loud.
    #[error("provenance could not be established")]
    Provenance(#[from] ProvenanceError),

    /// A file could not be read.
    #[error("could not read {path}")]
    Io {
        /// The path.
        path: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
}
