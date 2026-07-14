//! The provenance model: the reason this crate can be trusted.
//!
//! An insurance policy is a legal contract. An extracted statement about what
//! a policy says is worthless — worse, dangerous — unless the reader can go
//! back to the page and read the words for themselves. So provenance here is
//! not a nice-to-have field that a careless caller might forget to fill in.
//! It is **structural**: the types in this module are built so that a term
//! *without* a document, a page, a section, a clause identifier and the
//! verbatim source text **cannot be constructed at all**.
//!
//! Concretely, the enforcement rests on four properties, and every one of them
//! is load-bearing:
//!
//! 1. [`SourceText`] wraps a `String` whose non-emptiness is validated in the
//!    only constructor, and it does **not** implement `Default`. There is no
//!    way to obtain an empty `SourceText`.
//! 2. [`Provenance`] has private fields and exactly one constructor, which
//!    takes all five provenance components by value. There is no `Default`,
//!    no builder with optional fields, and no `..Default::default()` escape.
//! 3. [`ExtractedTerm`] has private fields and its only constructor takes a
//!    `Provenance` by value. `ExtractedTerm<T>` is therefore uninhabitable
//!    without one, for every `T`. It deliberately does **not** implement
//!    `From<T>`, because that would be exactly the un-sourced construction
//!    path this module exists to forbid.
//! 4. Deserialization does not open a back door: [`SourceText`],
//!    [`PageNumber`] and [`DocumentId`] all deserialize *through* their
//!    validating constructors (`#[serde(try_from = ...)]`), so a hand-written
//!    JSON blob with an empty `verbatim` field is a deserialization error, not
//!    a silently-invalid value.
//!
//! There is one more property, and it is the one that turns "provenance"
//! from bookkeeping into a guarantee: [`crate::Clause::cite`] will only mint a
//! `Provenance` for a fragment of text that **actually occurs in the clause**.
//! You cannot attach a paraphrase, a summary, or an invented quotation to a
//! real clause and have it come out looking sourced. The quote is checked
//! against the source.

use std::fmt;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Something was wrong with a piece of claimed provenance.
///
/// Every variant here represents a case where a caller tried to build a
/// provenance-carrying value that would have been a lie. They are errors, not
/// warnings, and they are never papered over with a default.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProvenanceError {
    /// Verbatim source text was empty or entirely whitespace. There is no
    /// such thing as a citation to nothing.
    #[error("verbatim source text is empty: an extracted term must quote the document")]
    EmptySourceText,

    /// A document identifier was empty.
    #[error("document identifier is empty")]
    EmptyDocumentId,

    /// A clause identifier as printed in the document was empty.
    #[error("clause identifier is empty")]
    EmptyClauseId,

    /// PDF pages are numbered from 1. A zero page number means the caller
    /// lost track of where the text came from.
    #[error("page number is zero: pages are numbered from 1")]
    ZeroPageNumber,

    /// A caller tried to cite a fragment of text that does not occur in the
    /// clause it claims to come from.
    ///
    /// This is the check that makes "verbatim" mean verbatim. It fires on a
    /// paraphrase, a summary, a translation, or an outright fabrication — all
    /// of which are things an insurance-document tool must never present as
    /// the policy's own words.
    #[error(
        "quoted fragment does not occur in clause {clause}: {fragment:?} \
         (a citation must quote the document, not paraphrase it)"
    )]
    QuoteNotInClause {
        /// The clause the fragment claimed to come from.
        clause: String,
        /// The fragment that could not be found in it.
        fragment: String,
    },
}

/// Verbatim text copied out of a source document, guaranteed non-empty.
///
/// The wrapped `String` is private and there is no `Default`, so the only way
/// to get a `SourceText` is through [`SourceText::new`], which rejects empty
/// and whitespace-only input. `#[serde(try_from = "String")]` routes
/// deserialization through the same check, so the invariant survives a
/// round-trip through JSON.
///
/// This type does not normalise, trim-to-taste, sentence-case, or otherwise
/// tidy its contents beyond stripping leading/trailing whitespace: it is the
/// document's words, and altering them is precisely the harm this crate is
/// designed to avoid.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SourceText(String);

impl SourceText {
    /// Wraps verbatim document text.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptySourceText`] if the text is empty or contains
    /// only whitespace.
    pub fn new(text: impl Into<String>) -> Result<Self, ProvenanceError> {
        let text = text.into();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(ProvenanceError::EmptySourceText);
        }
        Ok(Self(trimmed.to_string()))
    }

    /// The verbatim text.
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
/// A `NonZeroUsize` rather than a `usize` because page 0 does not exist: if a
/// caller has a 0 in hand, they have lost the location, and losing the
/// location must not be expressible.
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

/// Identifies the source document a term was extracted from.
///
/// Deliberately a caller-supplied string (a file name, a path, a policy
/// number) rather than a generated id: a citation has to mean something to a
/// human holding the paper document, and a UUID does not.
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

/// Where a clause sits in the document's heading hierarchy, outermost first —
/// e.g. `["Part II — Benefits", "Section 4 — Exclusions"]`.
///
/// This is the "section" component of provenance, and it is not decoration.
/// In an insurance policy the enclosing section is frequently the *only* thing
/// that tells you what a clause does: "Any claim arising from war." is a bare
/// noun phrase that says nothing about coverage on its own — it is an
/// exclusion solely because it is printed under a heading that says
/// `Exclusions`. Losing the section path loses the meaning. See
/// [`crate::Exclusion`] for how this is used.
///
/// May legitimately be empty, for a clause that appears before any heading
/// (a preamble or a recital). Empty is a fact about the document, not missing
/// data, which is why this type — unlike the others in this module — does
/// permit an empty value.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SectionPath(Vec<String>);

impl SectionPath {
    /// Builds a section path from outermost heading to innermost.
    pub fn new(headings: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self(headings.into_iter().map(Into::into).collect())
    }

    /// The headings, outermost first.
    pub fn headings(&self) -> &[String] {
        &self.0
    }

    /// The innermost (most specific) heading, if any.
    pub fn innermost(&self) -> Option<&str> {
        self.0.last().map(String::as_str)
    }

    /// Whether any heading in the path satisfies `predicate`.
    ///
    /// Used by role classification, which must consider the *whole* path: a
    /// clause nested three levels below a `General Exclusions` heading is
    /// still an exclusion even though its own immediate heading may say
    /// nothing about exclusions.
    pub fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
        self.0.iter().any(|heading| predicate(heading))
    }

    /// Whether the path has no headings.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for SectionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.join(" > "))
    }
}

/// A complete, checkable citation into a source document.
///
/// Five components, all mandatory, all supplied at construction: the document,
/// the page, the section path, the clause identifier, and — the one that makes
/// the other four verifiable — **the verbatim source text**.
///
/// A `Provenance` is normally not built by hand. It is minted by
/// [`crate::Clause::provenance`] (citing the whole clause) or
/// [`crate::Clause::cite`] (citing a fragment of it, checked against the
/// clause's actual text). [`Provenance::new`] exists for callers assembling a
/// citation from another source, and is still total in its arguments: there is
/// no way to leave a component out.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Provenance {
    document: DocumentId,
    page: PageNumber,
    section: SectionPath,
    clause: crate::ClauseId,
    verbatim: SourceText,
}

impl Provenance {
    /// Records where a piece of text came from.
    ///
    /// All five components are required. This is the only constructor, and it
    /// has no optional arguments, no defaults, and no builder — by design.
    pub fn new(
        document: DocumentId,
        page: PageNumber,
        section: SectionPath,
        clause: crate::ClauseId,
        verbatim: SourceText,
    ) -> Self {
        Self {
            document,
            page,
            section,
            clause,
            verbatim,
        }
    }

    /// The source document.
    pub fn document(&self) -> &DocumentId {
        &self.document
    }

    /// The page the quoted text is printed on.
    pub fn page(&self) -> PageNumber {
        self.page
    }

    /// The heading hierarchy the quoted text sits under.
    pub fn section(&self) -> &SectionPath {
        &self.section
    }

    /// The clause the quoted text belongs to.
    pub fn clause(&self) -> &crate::ClauseId {
        &self.clause
    }

    /// **The document's own words.** Everything else in this struct is a
    /// pointer; this is the thing pointed at, and the thing a reader should
    /// be shown whenever a machine-extracted value is presented to them.
    pub fn verbatim(&self) -> &SourceText {
        &self.verbatim
    }
}

impl fmt::Display for Provenance {
    /// A human-readable citation, e.g.
    /// `MyPolicy.pdf, p.4, Section 4 — Exclusions, clause 4.2`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.document, self.page)?;
        if !self.section.is_empty() {
            write!(f, ", {}", self.section)?;
        }
        write!(f, ", clause {}", self.clause)
    }
}

/// A value extracted from an insurance document, inseparable from the citation
/// that justifies it.
///
/// This is the type [`crate::Clause`]-derived knowledge travels in, and the
/// type a domain crate (e.g. `kopitiam-health`) refines. The point of the
/// wrapper is not convenience — it is that **`ExtractedTerm<T>` cannot be
/// constructed without a [`Provenance`]**. There is no `Default`, no
/// `From<T>`, no public field, and no constructor that omits the citation. An
/// un-sourced extracted term is not a bug you have to remember not to write;
/// it is a program that does not compile.
///
/// # Refinement, and why `map` preserves provenance
///
/// `kopitiam-insurance` is generic across motor, life, travel, property and
/// health. It extracts, say, an `ExtractedTerm<ScheduleValue>` — a number with
/// a citation — and knows nothing about what the number *means*. A domain
/// crate then refines that into its own type (a health crate might turn it
/// into a deductible, a motor crate into an excess). [`ExtractedTerm::map`]
/// and [`ExtractedTerm::try_map`] exist so that refinement can happen **without
/// the value ever being separated from its citation**: the domain type comes
/// out the far end still carrying the page and the verbatim words that
/// justified it.
///
/// If a domain crate could unwrap the value, transform it, and re-wrap it, the
/// guarantee would be worth nothing — so [`ExtractedTerm::into_value`] does not
/// exist, and `map` is the sanctioned path.
///
/// ```
/// # use kopitiam_insurance::*;
/// # fn main() -> Result<(), ProvenanceError> {
/// let provenance = Provenance::new(
///     DocumentId::new("policy.pdf")?,
///     PageNumber::new(3)?,
///     SectionPath::new(["Schedule"]),
///     ClauseId::printed("2.1")?,
///     SourceText::new("Excess: S$500 each and every claim")?,
/// );
/// let excess = ExtractedTerm::new(500_i64, provenance);
///
/// // A domain crate refines the value; the citation comes along for free.
/// let refined = excess.map(|dollars| format!("{dollars} dollars"));
/// assert_eq!(refined.value(), "500 dollars");
/// assert_eq!(refined.provenance().page().get(), 3);
/// assert_eq!(
///     refined.verbatim(),
///     "Excess: S$500 each and every claim",
/// );
/// # Ok(())
/// # }
/// ```
///
/// An `ExtractedTerm` cannot be conjured out of a bare value — this does not
/// compile, and that is the whole point:
///
/// ```compile_fail
/// use kopitiam_insurance::ExtractedTerm;
/// let term: ExtractedTerm<i64> = ExtractedTerm { value: 500, source: unimplemented!() };
/// ```
///
/// ```compile_fail
/// use kopitiam_insurance::ExtractedTerm;
/// let term: ExtractedTerm<i64> = ExtractedTerm::default();
/// ```
///
/// Nor can the citation be an empty string, because [`SourceText`] has no
/// `Default` and its constructor rejects blank text:
///
/// ```compile_fail
/// use kopitiam_insurance::SourceText;
/// let text = SourceText(String::new());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExtractedTerm<T> {
    value: T,
    source: Provenance,
}

impl<T> ExtractedTerm<T> {
    /// Binds an extracted value to the citation that justifies it.
    pub fn new(value: T, source: Provenance) -> Self {
        Self { value, source }
    }

    /// The extracted value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// The citation: document, page, section, clause, verbatim text.
    pub fn provenance(&self) -> &Provenance {
        &self.source
    }

    /// Shorthand for the document's own words backing this term.
    pub fn verbatim(&self) -> &str {
        self.source.verbatim().as_str()
    }

    /// Refines the value while keeping the citation attached.
    ///
    /// This is how a domain crate turns a generic extraction into a domain
    /// type without the value ever escaping its provenance. See the type-level
    /// docs.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ExtractedTerm<U> {
        ExtractedTerm {
            value: f(self.value),
            source: self.source,
        }
    }

    /// Fallible refinement, for a domain crate that must reject values it
    /// cannot make sense of.
    ///
    /// Returning `Err` here is the honest outcome when a document says
    /// something the domain model has no representation for. The alternative —
    /// coercing it into the nearest representable value — is exactly the
    /// "clean-looking wrong answer" this crate refuses to produce.
    ///
    /// # Errors
    ///
    /// Whatever `f` returns.
    pub fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<ExtractedTerm<U>, E> {
        Ok(ExtractedTerm {
            value: f(self.value)?,
            source: self.source,
        })
    }

    /// Borrows the value and its citation together.
    ///
    /// Deliberately paired: there is no accessor that hands out an owned value
    /// with the citation left behind.
    pub fn parts(&self) -> (&T, &Provenance) {
        (&self.value, &self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> Provenance {
        Provenance::new(
            DocumentId::new("synthetic-policy.pdf").unwrap(),
            PageNumber::new(2).unwrap(),
            SectionPath::new(["Section 3 — What Is Covered"]),
            crate::ClauseId::printed("3.1").unwrap(),
            SourceText::new("We will pay the Benefit shown in the Schedule.").unwrap(),
        )
    }

    #[test]
    fn source_text_rejects_empty_and_whitespace() {
        assert_eq!(SourceText::new(""), Err(ProvenanceError::EmptySourceText));
        assert_eq!(
            SourceText::new("   \n\t "),
            Err(ProvenanceError::EmptySourceText)
        );
        assert_eq!(SourceText::new("  text  ").unwrap().as_str(), "text");
    }

    #[test]
    fn page_number_rejects_zero() {
        assert_eq!(PageNumber::new(0), Err(ProvenanceError::ZeroPageNumber));
        assert_eq!(PageNumber::new(1).unwrap().get(), 1);
    }

    #[test]
    fn document_id_rejects_empty() {
        assert_eq!(DocumentId::new("  "), Err(ProvenanceError::EmptyDocumentId));
    }

    #[test]
    fn deserialization_cannot_smuggle_in_empty_verbatim_text() {
        // The invariant would be worthless if `serde` could rebuild a
        // `SourceText("")` straight from JSON, bypassing the constructor.
        // `#[serde(try_from = "String")]` is what closes that door.
        let err = serde_json::from_str::<SourceText>(r#""""#).unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    #[test]
    fn deserialization_cannot_smuggle_in_page_zero() {
        let err = serde_json::from_str::<PageNumber>("0").unwrap_err();
        assert!(err.to_string().contains("zero"), "got: {err}");
    }

    #[test]
    fn extracted_term_round_trips_through_json_with_its_citation_intact() {
        let term = ExtractedTerm::new(42_i64, provenance());
        let json = serde_json::to_string(&term).unwrap();
        let back: ExtractedTerm<i64> = serde_json::from_str(&json).unwrap();
        assert_eq!(term, back);
        assert_eq!(back.provenance().page().get(), 2);
        assert!(back.verbatim().starts_with("We will pay"));
    }

    #[test]
    fn map_carries_the_citation_across_a_refinement() {
        let term = ExtractedTerm::new(42_i64, provenance());
        let refined = term.map(|n| n * 2);
        assert_eq!(*refined.value(), 84);
        assert_eq!(refined.provenance().page().get(), 2);
        assert_eq!(
            refined.verbatim(),
            "We will pay the Benefit shown in the Schedule."
        );
    }

    #[test]
    fn try_map_lets_a_domain_crate_refuse_a_value_it_cannot_model() {
        let term = ExtractedTerm::new("not a number".to_string(), provenance());
        let refined: Result<ExtractedTerm<i64>, _> = term.try_map(|s| s.parse::<i64>());
        assert!(refined.is_err());
    }

    #[test]
    fn provenance_displays_as_a_human_citation() {
        assert_eq!(
            provenance().to_string(),
            "synthetic-policy.pdf, p.2, Section 3 — What Is Covered, clause 3.1"
        );
    }
}
