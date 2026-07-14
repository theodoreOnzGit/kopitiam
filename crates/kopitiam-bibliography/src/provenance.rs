//! The provenance model: why an extracted reference can be trusted.
//!
//! A citation is a **claim about provenance**. "This result is due to Okafor
//! (2015)" is an assertion about who established what, and it is checked by
//! people who care — reviewers, examiners, the authors themselves. Attributing
//! a result to the wrong paper is not a rendering bug; it is an
//! academic-integrity problem, and no amount of "the parser was only 90%
//! confident" makes it acceptable.
//!
//! So provenance here is **structural**, not a field a careless caller might
//! forget. The types in this module are built so that a reference *without* a
//! source document, a locator, and the **verbatim source string**
//! cannot be constructed at all.
//!
//! # This is not a fourth provenance model
//!
//! It is deliberately the **same pattern** as [`kopitiam_insurance`]'s (private
//! fields, exactly one constructor, no `Default`, `#[serde(try_from = ...)]`
//! so deserialisation cannot smuggle in an un-sourced value). The *components*
//! differ because the domain differs: an insurance term is located by clause,
//! a bibliographic reference is not. A reference is located by **page** (when
//! it was read off a document) or by **line** (when it was read out of a `.bib`
//! file) — see [`Locator`].
//!
//! Long-term, this pattern belongs in `kopitiam-ontology` (which is exactly
//! where shared vocabulary lives) rather than being re-implemented in each
//! domain engine. That is recorded as a decision to be made by the maintainer
//! — see `docs/ai-decisions/AID-0018.md` — because `kopitiam-ontology` was not
//! this crate's to change.
//!
//! [`kopitiam_insurance`]: https://github.com/kopitiam-project/kopitiam
//!
//! # The four properties that make it a guarantee
//!
//! 1. [`SourceText`] wraps a `String` whose non-emptiness is validated in its
//!    only constructor, and it has no `Default`. There is no way to obtain an
//!    empty `SourceText`.
//! 2. [`Provenance`] has private fields and exactly one constructor, taking
//!    every component by value. No `Default`, no optional-field builder, no
//!    `..Default::default()` escape hatch.
//! 3. Every reference-bearing type in this crate ([`crate::Reference`],
//!    [`crate::RawEntry`], [`crate::SourcedCitation`]) holds a `Provenance` by
//!    value and exposes no constructor that omits it.
//! 4. Deserialisation is not a back door: [`SourceText`], [`PageNumber`],
//!    [`LineNumber`] and [`DocumentId`] all deserialise *through* their
//!    validating constructors, so a hand-written JSON blob with an empty
//!    `verbatim` is a deserialisation error rather than a silently-invalid
//!    value.
//!
//! # `verbatim` means verbatim
//!
//! [`Provenance::verbatim`] holds the source text **exactly as it was printed**,
//! including the line breaks a PDF's reference list wraps at. It is *not* the
//! de-hyphenated, line-joined string the parser actually works on — that is a
//! derived artefact, available separately as [`Provenance::normalised`], and
//! the transformation between them is documented in [`crate::normalise`].
//!
//! Keeping both is the point. A reader who wants to check the crate's work
//! needs the words as they appear on page 14, not the words after we tidied
//! them.

use std::fmt;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Something was wrong with a piece of claimed provenance.
///
/// Every variant is a case where a caller tried to build a provenance-carrying
/// value that would have been a lie. They are errors, never papered over with
/// a default.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProvenanceError {
    /// Verbatim source text was empty or entirely whitespace. There is no such
    /// thing as a citation to nothing.
    #[error("verbatim source text is empty: an extracted reference must quote its source")]
    EmptySourceText,

    /// A document identifier was empty.
    #[error("document identifier is empty")]
    EmptyDocumentId,

    /// PDF pages are numbered from 1. A zero page means the caller lost track
    /// of where the text came from.
    #[error("page number is zero: pages are numbered from 1")]
    ZeroPageNumber,

    /// File lines are numbered from 1, for the same reason.
    #[error("line number is zero: lines are numbered from 1")]
    ZeroLineNumber,
}

/// Verbatim text copied out of a source, guaranteed non-empty.
///
/// The wrapped `String` is private, there is no `Default`, and
/// `#[serde(try_from = "String")]` routes deserialisation through the same
/// check the constructor applies. The invariant therefore survives a round trip
/// through JSON.
///
/// This type strips leading and trailing whitespace and **does nothing else**.
/// It does not normalise, re-case, de-hyphenate, or collapse interior
/// newlines: those are the source's own words and its own line breaks, and
/// altering them is precisely the harm this crate exists to avoid.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SourceText(String);

impl SourceText {
    /// Wraps verbatim source text.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptySourceText`] if the text is empty or blank.
    pub fn new(text: impl Into<String>) -> Result<Self, ProvenanceError> {
        let text = text.into();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(ProvenanceError::EmptySourceText);
        }
        Ok(Self(trimmed.to_string()))
    }

    /// The verbatim text, line breaks and all.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for SourceText {
    type Error = ProvenanceError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::new(text)
    }
}

impl From<SourceText> for String {
    fn from(text: SourceText) -> Self {
        text.0
    }
}

impl fmt::Display for SourceText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A 1-based page number, matching [`kopitiam_pdf::Page::number`].
///
/// A `NonZeroUsize` because page 0 does not exist: a caller holding a 0 has
/// lost the location, and losing the location must not be expressible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct PageNumber(NonZeroUsize);

impl PageNumber {
    /// Builds a page number from a 1-based page index.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::ZeroPageNumber`] if `page` is 0.
    pub fn new(page: usize) -> Result<Self, ProvenanceError> {
        NonZeroUsize::new(page)
            .map(Self)
            .ok_or(ProvenanceError::ZeroPageNumber)
    }

    /// The 1-based page number.
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for PageNumber {
    type Error = ProvenanceError;

    fn try_from(page: usize) -> Result<Self, Self::Error> {
        Self::new(page)
    }
}

impl From<PageNumber> for usize {
    fn from(page: PageNumber) -> Self {
        page.get()
    }
}

impl fmt::Display for PageNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p.{}", self.0)
    }
}

/// A 1-based line number in a text file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct LineNumber(NonZeroUsize);

impl LineNumber {
    /// Builds a line number from a 1-based line index.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::ZeroLineNumber`] if `line` is 0.
    pub fn new(line: usize) -> Result<Self, ProvenanceError> {
        NonZeroUsize::new(line)
            .map(Self)
            .ok_or(ProvenanceError::ZeroLineNumber)
    }

    /// The 1-based line number.
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for LineNumber {
    type Error = ProvenanceError;

    fn try_from(line: usize) -> Result<Self, Self::Error> {
        Self::new(line)
    }
}

impl From<LineNumber> for usize {
    fn from(line: LineNumber) -> Self {
        line.get()
    }
}

impl fmt::Display for LineNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}", self.0)
    }
}

/// Identifies the source a reference was read out of.
///
/// A caller-supplied string (a file name, a path, a DOI of the *citing* paper)
/// rather than a generated id, for the same reason `kopitiam-insurance` made
/// that choice: a citation must mean something to a human holding the document,
/// and a UUID does not.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct DocumentId(String);

impl DocumentId {
    /// Names a source document.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptyDocumentId`] if the name is empty or blank.
    pub fn new(name: impl Into<String>) -> Result<Self, ProvenanceError> {
        let name = name.into();
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(ProvenanceError::EmptyDocumentId);
        }
        Ok(Self(trimmed.to_string()))
    }

    /// The document's name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for DocumentId {
    type Error = ProvenanceError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::new(name)
    }
}

impl From<DocumentId> for String {
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where inside a source a reference was found.
///
/// # Why an enum rather than an `Option<PageNumber>`
///
/// A bibliography arrives from two genuinely different kinds of source, and
/// flattening them into "page, possibly missing" would throw away the one thing
/// a reader needs. A reference read off page 14 of a PDF is checkable by
/// turning to page 14. A reference read out of line 87 of `refs.bib` is
/// checkable by opening that file at that line. Both are complete citations;
/// neither is a degraded version of the other.
///
/// The `Option<PageNumber>` encoding would have made every `.bib`-sourced
/// reference look like a PDF-sourced one whose page we *lost* — which is a lie
/// about the quality of our own knowledge, and exactly the kind of lie the rest
/// of this module exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locator {
    /// The 1-based page of a paginated document (a PDF).
    Page(PageNumber),
    /// The 1-based line of a text file (a `.bib`).
    Line(LineNumber),
}

impl Locator {
    /// The page, if this reference came from a paginated document.
    ///
    /// Returns `None` for a file-sourced reference — which is a *fact about the
    /// source*, not missing data. A `.bib` file has no pages.
    pub fn page(self) -> Option<PageNumber> {
        match self {
            Self::Page(page) => Some(page),
            Self::Line(_) => None,
        }
    }

    /// The line, if this reference came from a text file.
    pub fn line(self) -> Option<LineNumber> {
        match self {
            Self::Line(line) => Some(line),
            Self::Page(_) => None,
        }
    }
}

impl fmt::Display for Locator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Page(page) => fmt::Display::fmt(page, f),
            Self::Line(line) => fmt::Display::fmt(line, f),
        }
    }
}

/// A complete, checkable citation back to where a reference was read.
///
/// Three components, all mandatory, all supplied at construction: the
/// **document**, the **locator** (page or line), and — the one that makes the
/// other two verifiable — the **verbatim source text**.
///
/// # Two strings, and why both are kept
///
/// A reference list in a PDF is printed wrapped, hyphenated, and with URLs
/// broken across lines. To parse it, the text must be joined back up. That
/// joining is lossy in one direction (a hyphen removed at a line break cannot
/// be distinguished afterwards from one that was never there), so this type
/// keeps **both** strings:
///
/// * [`Provenance::verbatim`] — the source's own words, its own line breaks.
///   This is what a human checks against the page.
/// * [`Provenance::normalised`] — what the parser actually read. This is what
///   a human checks against *us* when they think we got it wrong.
///
/// For a reference that needed no joining (a single-line `.bib` value), the two
/// are identical, and no information is implied by that.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Provenance {
    document: DocumentId,
    locator: Locator,
    verbatim: SourceText,
    normalised: SourceText,
}

impl Provenance {
    /// Records where a reference was read.
    ///
    /// `verbatim` is the source's own text; `normalised` is the line-joined,
    /// de-hyphenated form the parser consumed. Pass the same value for both
    /// when no joining was needed.
    ///
    /// This is the only constructor. It has no optional arguments, no defaults,
    /// and no builder — by design.
    pub fn new(
        document: DocumentId,
        locator: Locator,
        verbatim: SourceText,
        normalised: SourceText,
    ) -> Self {
        Self {
            document,
            locator,
            verbatim,
            normalised,
        }
    }

    /// Convenience constructor for a reference read off a page of a PDF, whose
    /// normalised form is derived from the verbatim text by [`crate::normalise`].
    ///
    /// # Errors
    ///
    /// [`ProvenanceError`] if the document name is blank, the page is 0, or the
    /// text is empty.
    pub fn from_page(
        document: &DocumentId,
        page: usize,
        verbatim: impl Into<String>,
    ) -> Result<Self, ProvenanceError> {
        let verbatim = verbatim.into();
        let normalised = crate::normalise(&verbatim);
        Ok(Self::new(
            document.clone(),
            Locator::Page(PageNumber::new(page)?),
            SourceText::new(verbatim)?,
            SourceText::new(normalised)?,
        ))
    }

    /// Convenience constructor for a reference read out of a text file at a
    /// 1-based line.
    ///
    /// No normalisation is applied: a `.bib` value is already the author's
    /// intended string, and joining its lines is the BibTeX parser's job, not a
    /// typographic repair.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError`] if the document name is blank, the line is 0, or the
    /// text is empty.
    pub fn from_line(
        document: &DocumentId,
        line: usize,
        verbatim: impl Into<String>,
    ) -> Result<Self, ProvenanceError> {
        let verbatim = SourceText::new(verbatim)?;
        Ok(Self::new(
            document.clone(),
            Locator::Line(LineNumber::new(line)?),
            verbatim.clone(),
            verbatim,
        ))
    }

    /// The source document.
    pub fn document(&self) -> &DocumentId {
        &self.document
    }

    /// Where in it the reference was found.
    pub fn locator(&self) -> Locator {
        self.locator
    }

    /// **The source's own words**, with its own line breaks. Everything else in
    /// this struct is a pointer; this is the thing pointed at, and the thing a
    /// reader should be shown whenever a machine-extracted reference is put in
    /// front of them.
    pub fn verbatim(&self) -> &SourceText {
        &self.verbatim
    }

    /// The line-joined, de-hyphenated text the parser actually consumed.
    ///
    /// Differs from [`Self::verbatim`] only where [`crate::normalise`] had work
    /// to do. When a parse looks wrong, this is where to look first: the bug is
    /// as likely to be in the joining as in the parsing.
    pub fn normalised(&self) -> &SourceText {
        &self.normalised
    }
}

impl fmt::Display for Provenance {
    /// A human-readable citation, e.g. `paper.pdf, p.14`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.document, self.locator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_text_rejects_empty_and_whitespace() {
        assert_eq!(SourceText::new(""), Err(ProvenanceError::EmptySourceText));
        assert_eq!(
            SourceText::new("  \n\t "),
            Err(ProvenanceError::EmptySourceText)
        );
    }

    #[test]
    fn source_text_preserves_interior_line_breaks() {
        // The whole point of "verbatim": a wrapped reference list entry keeps
        // its wraps, so a reader can find it on the page as printed.
        let text = SourceText::new("  M. R. Chen, \u{201c}An open-source multilingual\ntext aligner.\u{201d}  ")
            .unwrap();
        assert!(text.as_str().contains('\n'));
        assert!(text.as_str().starts_with("M. R. Chen"));
    }

    #[test]
    fn page_number_rejects_zero() {
        assert_eq!(PageNumber::new(0), Err(ProvenanceError::ZeroPageNumber));
        assert_eq!(PageNumber::new(14).unwrap().get(), 14);
    }

    #[test]
    fn line_number_rejects_zero() {
        assert_eq!(LineNumber::new(0), Err(ProvenanceError::ZeroLineNumber));
    }

    #[test]
    fn document_id_rejects_empty() {
        assert_eq!(DocumentId::new("   "), Err(ProvenanceError::EmptyDocumentId));
    }

    #[test]
    fn deserialisation_cannot_smuggle_in_empty_verbatim_text() {
        // The invariant would be worthless if serde could rebuild a
        // SourceText("") straight from JSON, bypassing the constructor.
        let err = serde_json::from_str::<SourceText>(r#""""#).unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    #[test]
    fn deserialisation_cannot_smuggle_in_page_zero() {
        let err = serde_json::from_str::<PageNumber>("0").unwrap_err();
        assert!(err.to_string().contains("zero"), "got: {err}");
    }

    #[test]
    fn a_file_sourced_reference_has_no_page_and_says_so() {
        let doc = DocumentId::new("refs.bib").unwrap();
        let provenance = Provenance::from_line(&doc, 87, "@article{chen2024, ...}").unwrap();
        assert_eq!(provenance.locator().page(), None);
        assert_eq!(provenance.locator().line().unwrap().get(), 87);
        assert_eq!(provenance.to_string(), "refs.bib, line 87");
    }

    #[test]
    fn a_page_sourced_reference_keeps_both_the_printed_and_the_joined_text() {
        let doc = DocumentId::new("paper.pdf").unwrap();
        let provenance =
            Provenance::from_page(&doc, 14, "an open-source thermo-\nhydraulic solver").unwrap();

        // Verbatim: exactly as printed, hyphen and newline intact.
        assert_eq!(
            provenance.verbatim().as_str(),
            "an open-source thermo-\nhydraulic solver"
        );
        // Normalised: what the parser read.
        assert_eq!(
            provenance.normalised().as_str(),
            "an open-source thermohydraulic solver"
        );
        assert_eq!(provenance.to_string(), "paper.pdf, p.14");
    }

    #[test]
    fn provenance_round_trips_through_json_with_both_strings_intact() {
        let doc = DocumentId::new("paper.pdf").unwrap();
        let provenance = Provenance::from_page(&doc, 14, "thermo-\nhydraulic").unwrap();
        let json = serde_json::to_string(&provenance).unwrap();
        let back: Provenance = serde_json::from_str(&json).unwrap();
        assert_eq!(provenance, back);
    }
}
