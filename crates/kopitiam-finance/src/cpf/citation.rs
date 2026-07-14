//! Provenance. A CPF figure without a citation is not a fact — it is a rumour.
//!
//! CLAUDE.md, Scientific Standards: *"preserve provenance ... scientific
//! software should always remain explainable."* In this domain "explainable"
//! has a precise operational meaning: when a user asks **"why is my employee
//! contribution 20%?"**, the answer must be
//!
//! > because the *CPF Contribution Rates* table published by the CPF Board,
//! > effective from 1 January 2025, row "55 years and below", column
//! > "Employee's share", says 20% — and here is the URL.
//!
//! and never
//!
//! > because that is what the code says.
//!
//! This is why [`Citation`] is a **non-optional** field of every dated policy
//! value ([`crate::cpf::temporal::Dated`]), and why lookups return the value
//! and its citation *together*, in one struct, rather than offering the
//! citation as some side-channel the caller may forget to consult.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::cpf::date::Date;

/// How a policy value physically got into KOPITIAM.
///
/// # Why this is not decoration
///
/// There is an enormous, and entirely invisible, difference between a number
/// that was machine-extracted from the CPF Board's own PDF and a number that
/// someone (or some model) typed in from memory. Both produce a `Citation`
/// that *looks* authoritative. Only one of them actually is.
///
/// Every value currently shipped in [`crate::cpf::published`] is
/// [`SourceKind::Transcribed`]. That label is the truth, and it must not be
/// quietly upgraded. It travels with the value into the knowledge graph, so a
/// downstream consumer can filter on it — and so nobody can pretend the
/// provenance is stronger than it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Transcribed by hand (or by a language model) from a published source,
    /// and **not yet verified against the primary document by the Document
    /// Engine**.
    ///
    /// Trust this exactly as much as you trust the transcriber. It is offered
    /// so the engine is useful today; it is not a substitute for ingestion.
    Transcribed,

    /// Extracted deterministically from an ingested document by
    /// `kopitiam-pdf` + `kopitiam-document`, with a locator pointing at the
    /// section it came from.
    ///
    /// This is the target state for every value in this crate. See
    /// [`crate::cpf::document`].
    ExtractedFromDocument,

    /// Derived by arithmetic from other cited values, with the derivation
    /// stated in the citation's `note`.
    ///
    /// Example: the Full Retirement Sum is defined by policy as exactly twice
    /// the Basic Retirement Sum. Recording that as *derived* rather than
    /// re-transcribing it means the relationship cannot drift, and the reader
    /// can see it is a definition rather than an independent observation.
    Derived,
}

impl fmt::Display for SourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SourceKind::Transcribed => "transcribed",
            SourceKind::ExtractedFromDocument => "extracted-from-document",
            SourceKind::Derived => "derived",
        };
        f.write_str(s)
    }
}

/// Where a policy value came from. Mandatory on every dated value.
///
/// Fields are public because a citation is inert data whose whole job is to be
/// read, serialised, and displayed. There is nothing to encapsulate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    /// The publishing body. Practically always `"Central Provident Fund Board"`
    /// or `"Ministry of Finance, Singapore"`, but recorded rather than assumed
    /// — CPF parameters are also set in Budget statements and in the CPF Act.
    pub publisher: String,

    /// The document, named as the publisher names it, e.g.
    /// `"CPF contribution rates (private sector and public sector non-pensionable employees)"`.
    pub document: String,

    /// Where *within* the document: a table name, a row, a section, a
    /// paragraph. This is what turns "the CPF website says so" into a claim
    /// somebody can actually go and check.
    pub locator: String,

    /// When the source document was published or last revised, where known.
    pub published: Option<Date>,

    /// A stable URL, where one exists.
    pub url: Option<String>,

    /// How this value got here. See [`SourceKind`].
    pub source_kind: SourceKind,

    /// Anything a careful reader needs in order to trust — or distrust — this
    /// value. Derivations, known ambiguities, and explicit statements of
    /// uncertainty belong here.
    pub note: Option<String>,
}

impl Citation {
    /// A citation to a CPF Board publication, transcribed rather than ingested.
    ///
    /// Named for what it is. If this function's name makes you uncomfortable
    /// about the strength of the provenance, it is doing its job.
    pub fn transcribed_from_cpf_board(document: impl Into<String>, locator: impl Into<String>) -> Self {
        Self {
            publisher: "Central Provident Fund Board".to_string(),
            document: document.into(),
            locator: locator.into(),
            published: None,
            url: None,
            source_kind: SourceKind::Transcribed,
            note: None,
        }
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    pub fn with_published(mut self, published: Date) -> Self {
        self.published = Some(published);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Marks this citation as a value derived by arithmetic from others, and
    /// records the derivation.
    pub fn derived(mut self, derivation: impl Into<String>) -> Self {
        self.source_kind = SourceKind::Derived;
        self.note = Some(derivation.into());
        self
    }

    /// A short human-readable rendering, suitable for a CLI footnote.
    pub fn short(&self) -> String {
        format!("{}, {} — {}", self.publisher, self.document, self.locator)
    }
}

impl fmt::Display for Citation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short())?;
        if let Some(published) = self.published {
            write!(f, " (published {published})")?;
        }
        write!(f, " [{}]", self.source_kind)?;
        if let Some(url) = &self.url {
            write!(f, " <{url}>")?;
        }
        if let Some(note) = &self.note {
            write!(f, " — {note}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribed_is_the_honest_default() {
        let c = Citation::transcribed_from_cpf_board("CPF contribution rates", "Table 1, row '55 and below'");
        assert_eq!(c.source_kind, SourceKind::Transcribed);
        assert!(c.to_string().contains("[transcribed]"));
        assert!(c.short().contains("Central Provident Fund Board"));
    }

    #[test]
    fn derived_records_its_derivation() {
        let c = Citation::transcribed_from_cpf_board("Retirement sums", "FRS")
            .derived("Full Retirement Sum is defined as 2 x the Basic Retirement Sum");
        assert_eq!(c.source_kind, SourceKind::Derived);
        assert!(c.note.as_deref().unwrap().contains("2 x"));
    }
}
