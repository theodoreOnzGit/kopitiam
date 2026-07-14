//! The bridge to KOPITIAM's Document Engine.
//!
//! CPF publishes its policy as PDFs and web pages. KOPITIAM already knows how to
//! turn a PDF into structured `Section`/`Fact` entities — `kopitiam-pdf` extracts
//! the page layout, `kopitiam-document` reconstructs a semantic `Document` from
//! it. **This module reuses that.** It does not contain a PDF parser, and it must
//! never grow one.
//!
//! # What this module does, and what it deliberately does not
//!
//! It **does**: take a CPF source document, run it through the Document Engine,
//! and lift its structure into the knowledge graph as `Section` entities — so
//! that a [`Citation`] can point at a section KOPITIAM has actually *seen*,
//! rather than at a string somebody typed.
//!
//! It **does not**: scrape contribution rates out of the PDF text. That is the
//! obvious next thing to build and it is deliberately not built here, because a
//! half-working table scraper applied to a rate table is a machine for producing
//! confident, wrong, *apparently-cited* numbers — which is strictly worse than
//! the honest transcription this crate currently ships. Extracting a rate from a
//! CPF table requires knowing which column is the employer's and which the
//! employee's, and getting that backwards is silent and catastrophic.
//!
//! The path from here is therefore: ingest the document (this module), locate the
//! rate tables within it, and only then — with the table's own headers as
//! evidence — promote a transcribed [`SourceKind::Transcribed`] value to
//! [`SourceKind::ExtractedFromDocument`] after checking it *agrees* with the
//! table. Verification first, extraction second. That is the correct order for a
//! domain where being wrong costs someone their house.

use std::path::Path;

use kopitiam_ontology::{Entity, EntityKind};
use serde_json::json;

use crate::cpf::citation::{Citation, SourceKind};

/// The provenance string stamped on every entity this crate emits, in the sense
/// of `kopitiam_ontology::Entity::source`.
pub const PROVIDER: &str = "kopitiam-finance/cpf";

/// A CPF source document, as understood by the Document Engine.
#[derive(Debug, Clone)]
pub struct SourceDocument {
    /// The document as the publisher names it. Becomes the `document` field of
    /// any [`Citation`] pointing into it.
    pub document: String,
    /// The publisher.
    pub publisher: String,
    /// A stable URL, if the document has one.
    pub url: Option<String>,
    /// The headings the Document Engine reconstructed, in document order. These
    /// are the citable locators.
    pub sections: Vec<Section>,
    /// How many pages the source had. Recorded because a reconstruction that
    /// found no sections in a fifty-page PDF is a reconstruction that failed, and
    /// the caller should be able to see that.
    pub source_pages: usize,
    /// How many tables the Document Engine found. CPF policy lives in tables;
    /// this is the number a future extractor would have to work with.
    pub tables: usize,
}

/// A heading in an ingested CPF document — a place a citation can point at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Heading depth, as reconstructed.
    pub level: usize,
    /// The heading text, which is what a [`Citation::locator`] would name.
    pub title: String,
}

/// Ingests a CPF policy PDF through KOPITIAM's Document Engine.
///
/// This is `kopitiam_pdf::extract` followed by `kopitiam_document::reconstruct` —
/// the *same* path every other document provider in KOPITIAM uses. CPF gets no
/// special parser, and should not have one.
///
/// # Errors
///
/// Propagates `kopitiam_pdf::ExtractError` if the file is not a readable PDF.
pub fn ingest_pdf(
    path: impl AsRef<Path>,
    document: impl Into<String>,
    publisher: impl Into<String>,
) -> Result<SourceDocument, kopitiam_pdf::ExtractError> {
    let pages = kopitiam_pdf::extract(path)?;
    Ok(from_pages(&pages, document, publisher))
}

/// As [`ingest_pdf`], from bytes already in memory.
///
/// # Errors
///
/// Propagates `kopitiam_pdf::ExtractError` if the bytes are not a readable PDF.
pub fn ingest_pdf_bytes(
    bytes: &[u8],
    document: impl Into<String>,
    publisher: impl Into<String>,
) -> Result<SourceDocument, kopitiam_pdf::ExtractError> {
    let pages = kopitiam_pdf::extract_from_bytes(bytes)?;
    Ok(from_pages(&pages, document, publisher))
}

fn from_pages(
    pages: &[kopitiam_pdf::Page],
    document: impl Into<String>,
    publisher: impl Into<String>,
) -> SourceDocument {
    let reconstructed = kopitiam_document::reconstruct(pages);

    let sections = reconstructed
        .blocks
        .iter()
        .filter_map(|block| match block {
            kopitiam_document::Block::Heading(h) => Some(Section {
                level: h.level,
                title: h.text.clone(),
            }),
            _ => None,
        })
        .collect();

    let tables = reconstructed
        .blocks
        .iter()
        .filter(|b| matches!(b, kopitiam_document::Block::Table(_)))
        .count();

    SourceDocument {
        document: document.into(),
        publisher: publisher.into(),
        url: None,
        sections,
        source_pages: reconstructed.metadata.source_pages,
        tables,
    }
}

impl SourceDocument {
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// A citation pointing at one of this document's sections.
    ///
    /// The [`SourceKind`] is [`SourceKind::ExtractedFromDocument`] — and that is
    /// *earned* here, because the section demonstrably exists in a document
    /// KOPITIAM has parsed. Note carefully what this asserts and what it does
    /// not: it asserts that **the section is real**. It says nothing about
    /// whether any particular *number* was read out of it correctly. Promoting a
    /// value's provenance to `ExtractedFromDocument` requires the second step
    /// described in the module docs, not merely this citation.
    pub fn cite(&self, section: &Section) -> Citation {
        Citation {
            publisher: self.publisher.clone(),
            document: self.document.clone(),
            locator: section.title.clone(),
            published: None,
            url: self.url.clone(),
            source_kind: SourceKind::ExtractedFromDocument,
            note: Some(format!(
                "Section located by KOPITIAM's Document Engine (kopitiam-pdf + \
                 kopitiam-document) at heading level {} of a {}-page source.",
                section.level, self.source_pages
            )),
        }
    }

    /// Lifts this document's structure into the knowledge graph as
    /// [`EntityKind::Section`] entities.
    ///
    /// This is the point of the whole module: CPF's source documents land in the
    /// *same* graph as everything else KOPITIAM knows, addressable by the same
    /// queries, so that a CPF rule and the paper it came from are one hop apart.
    pub fn entities(&self) -> Vec<Entity> {
        self.sections
            .iter()
            .map(|section| {
                Entity::new(EntityKind::Section, section.title.clone(), PROVIDER).with_metadata(json!({
                    "document": self.document,
                    "publisher": self.publisher,
                    "url": self.url,
                    "heading_level": section.level,
                    "source_pages": self.source_pages,
                    "tables_in_document": self.tables,
                    "domain": "cpf",
                }))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine reports honestly on a document it could not parse, rather than
    /// producing an empty `SourceDocument` that looks successfully ingested.
    #[test]
    fn a_non_pdf_is_rejected_rather_than_silently_yielding_no_sections() {
        let result = ingest_pdf_bytes(b"this is not a pdf", "CPF contribution rates", "CPF Board");
        assert!(result.is_err());
    }

    /// A document with no headings yields no sections — and, crucially, no
    /// citations. There is nothing to point at, so nothing pretends there is.
    #[test]
    fn a_document_without_headings_yields_no_citable_sections() {
        let doc = SourceDocument {
            document: "CPF contribution rates".to_string(),
            publisher: "Central Provident Fund Board".to_string(),
            url: None,
            sections: Vec::new(),
            source_pages: 3,
            tables: 0,
        };
        assert!(doc.entities().is_empty());
    }

    #[test]
    fn an_ingested_section_becomes_a_citable_ontology_entity() {
        let doc = SourceDocument {
            document: "CPF contribution rates".to_string(),
            publisher: "Central Provident Fund Board".to_string(),
            url: None,
            sections: vec![Section {
                level: 2,
                title: "Contribution rates from 1 January 2025".to_string(),
            }],
            source_pages: 4,
            tables: 3,
        }
        .with_url("https://www.cpf.gov.sg/");

        let citation = doc.cite(&doc.sections[0]);
        assert_eq!(citation.source_kind, SourceKind::ExtractedFromDocument);
        assert_eq!(citation.locator, "Contribution rates from 1 January 2025");
        assert_eq!(citation.url.as_deref(), Some("https://www.cpf.gov.sg/"));

        let entities = doc.entities();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].kind, EntityKind::Section);
        assert_eq!(entities[0].source, PROVIDER);
        assert_eq!(entities[0].metadata["heading_level"], 2);
        assert_eq!(entities[0].metadata["domain"], "cpf");
    }
}
