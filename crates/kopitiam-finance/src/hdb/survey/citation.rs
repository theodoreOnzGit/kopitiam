//! Where a number came from.
//!
//! Provenance is mandatory in KOPITIAM (CLAUDE.md, Scientific Standards). For a
//! resale price it is not academic bookkeeping: a buyer who is told "the median
//! is $X" must be able to get back to the exact table, in the exact
//! publication, that said so — and to see how old it is.
//!
//! # `Citation` here is not `kopitiam_document::Citation`
//!
//! The name collides, and the meanings are unrelated. `kopitiam_document`'s
//! `Citation` records *that a citation-looking string was seen inside a
//! paragraph* — it is a pointer into rendered text. This `Citation` is a
//! **provenance record**: the document, the table within it, and the date it was
//! published. Do not substitute one for the other.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::period::Period;

/// Points at the specific place inside a publication that a number was read
/// from.
///
/// Deliberately an enum rather than a free-text string: "Table 3.2" and "page
/// 47" are different claims about where to look, and a reader chasing a figure
/// needs to know which one they were given.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locator {
    /// A numbered table, e.g. `Table 3.2`. The strongest locator, and the one
    /// most HDB statistical releases support.
    Table(String),
    /// A named or numbered section.
    Section(String),
    /// A page number. Weaker than a table reference — pagination changes between
    /// editions — but better than nothing.
    Page(u32),
    /// A named dataset and, where it has one, a resource identifier — for
    /// machine-readable feeds such as the resale transaction dataset published
    /// on data.gov.sg, which have no tables or pages.
    Dataset { name: String, resource: Option<String> },
}

impl fmt::Display for Locator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Locator::Table(id) => write!(f, "Table {id}"),
            Locator::Section(id) => write!(f, "Section {id}"),
            Locator::Page(n) => write!(f, "p. {n}"),
            Locator::Dataset { name, resource } => match resource {
                Some(resource) => write!(f, "dataset `{name}` ({resource})"),
                None => write!(f, "dataset `{name}`"),
            },
        }
    }
}

/// The document a statistic was published in, and where inside it.
///
/// Every [`super::Statistic`] carries one. There is no way to build a statistic
/// without it, which is the point.
///
/// # Published date is not collection period
///
/// These are routinely conflated and the distinction matters to a buyer. A
/// report *published* in 2024 may report transactions *collected* through 2023.
/// [`Citation::published`] is when the document came out;
/// [`super::Statistic::period`] is when the world it describes was observed. A
/// figure can be freshly published and already stale.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Citation {
    publication: String,
    publisher: String,
    locator: Locator,
    published: Period,
    retrieved_from: Option<String>,
}

impl Citation {
    /// Records where a figure was published.
    ///
    /// `retrieved_from` is a URL or local path recording where the document was
    /// obtained, so a claim can be re-checked years later. It is *recorded*,
    /// never fetched — this module performs no network access (CLAUDE.md,
    /// Offline First).
    pub fn new(
        publication: impl Into<String>,
        publisher: impl Into<String>,
        locator: Locator,
        published: Period,
    ) -> Self {
        Self {
            publication: publication.into(),
            publisher: publisher.into(),
            locator,
            published,
            retrieved_from: None,
        }
    }

    /// Records where the document was obtained from, so the figure can be
    /// re-checked against its source.
    pub fn retrieved_from(mut self, source: impl Into<String>) -> Self {
        self.retrieved_from = Some(source.into());
        self
    }

    /// The publication title, e.g. `"HDB Resale Price Statistics"`.
    pub fn publication(&self) -> &str {
        &self.publication
    }

    /// The issuing body, e.g. `"Housing & Development Board, Singapore"`.
    pub fn publisher(&self) -> &str {
        &self.publisher
    }

    /// Where inside the publication the figure appears.
    pub fn locator(&self) -> &Locator {
        &self.locator
    }

    /// When the *document* was issued — not when the data was collected. See the
    /// type-level docs.
    pub fn published(&self) -> Period {
        self.published
    }

    /// Where the document was obtained, if recorded.
    pub fn source(&self) -> Option<&str> {
        self.retrieved_from.as_deref()
    }
}

impl fmt::Display for Citation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}, {} ({}), {}",
            self.publisher, self.publication, self.published, self.locator
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_traceable_reference() {
        let citation = Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("3.2".into()),
            Period::Year(2023),
        );
        let rendered = citation.to_string();
        assert!(rendered.contains("Table 3.2"));
        assert!(rendered.contains("2023"));
    }

    #[test]
    fn published_date_is_distinct_from_any_collection_period() {
        // A 2024 publication reporting 2023 data: the citation knows only about
        // the former. Conflating them is the bug this separation prevents.
        let citation = Citation::new(
            "SYNTHETIC FIXTURE — NOT HDB DATA",
            "KOPITIAM test suite",
            Locator::Table("1".into()),
            Period::Year(2024),
        );
        assert_eq!(citation.published(), Period::Year(2024));
    }
}
