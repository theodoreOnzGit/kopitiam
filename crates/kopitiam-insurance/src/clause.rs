//! [`Clause`]: a located, verbatim unit of an insurance document.
//!
//! Everything this crate extracts is extracted *from* a clause, and every
//! citation it mints points *at* a clause. So the `Clause` is where the
//! provenance guarantee is actually enforced (see [`Clause::cite`]), and it is
//! the type `kopitiam-health` and any other domain crate will spend most of
//! its time holding.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::crossref::{self, CrossReference};
use crate::provenance::{
    DocumentId, PageNumber, Provenance, ProvenanceError, SectionPath, SourceText,
};

/// How a clause is identified.
///
/// Most clauses in a policy wording carry a printed number — `4`, `4.2`,
/// `4.2.1`. Some do not: preambles, recitals, the sentence under a heading
/// that introduces the numbered list beneath it.
///
/// The tempting shortcut is to mint a sequential number for the unnumbered
/// ones. **This crate refuses to.** A synthetic "clause 12" is
/// indistinguishable, in a citation shown to a human, from a clause the
/// document actually numbered 12 — and pointing a reader at a clause number
/// that does not exist in their policy is precisely the kind of confident
/// nonsense this crate is built to avoid. Unnumbered clauses are therefore
/// labelled as such, positionally, and their [`fmt::Display`] output can never
/// be mistaken for a printed clause number.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseId {
    /// The identifier exactly as printed in the document, e.g. `"4.2.1"`.
    /// This is the only variant a cross-reference can ever resolve to, since
    /// a document can only cross-reference numbers it actually prints.
    Printed(String),

    /// The document did not number this clause. Located by page and by its
    /// ordinal position on that page, so it is still addressable — but
    /// visibly not a document clause number.
    Unnumbered {
        /// The page it appears on.
        page: PageNumber,
        /// Its 0-based position among the unnumbered clauses on that page.
        ordinal: usize,
    },
}

impl ClauseId {
    /// A clause identifier as printed in the document.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptyClauseId`] if the label is empty or blank.
    pub fn printed(label: impl Into<String>) -> Result<Self, ProvenanceError> {
        let label = label.into();
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return Err(ProvenanceError::EmptyClauseId);
        }
        Ok(Self::Printed(trimmed.to_string()))
    }

    /// The printed label, if the document printed one.
    pub fn printed_label(&self) -> Option<&str> {
        match self {
            Self::Printed(label) => Some(label),
            Self::Unnumbered { .. } => None,
        }
    }
}

impl fmt::Display for ClauseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Printed(label) => f.write_str(label),
            // Bracketed and spelled out, so that a citation rendered into a
            // report reads "clause [unnumbered, p.3 #1]" and no reader goes
            // looking for a clause number that isn't in their policy.
            Self::Unnumbered { page, ordinal } => {
                write!(f, "[unnumbered, {page} #{ordinal}]")
            }
        }
    }
}

/// What a clause *does* in the contract.
///
/// This crate does not interpret clauses, but it does have to route them —
/// an exclusion must not end up on the coverage list. The role is therefore a
/// structural classification (mostly: what heading is this printed under, and
/// how does the sentence open), never a judgment about what the clause means.
///
/// [`ClauseRole::Unclassified`] is a first-class, honest answer and is the
/// default. Guessing a role would be worse than admitting we do not know one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseRole {
    /// Defines a term. See [`crate::Definition`].
    Definition,
    /// States what the insurer will pay for (an insuring agreement, a benefit).
    Coverage,
    /// States what is *not* covered. See [`crate::Exclusion`].
    Exclusion,
    /// A condition, duty or warranty the insured must satisfy.
    Condition,
    /// A row or table of policy-specific numbers. See [`crate::Schedule`].
    Schedule,
    /// Text belonging to an endorsement. See [`crate::Endorsement`].
    Endorsement,
    /// We could not determine what this clause does. Not a failure to report —
    /// a fact to report.
    Unclassified,
}

/// One block of a clause's text, attributed to the page it is printed on.
///
/// A clause routinely runs across a page break. Storing the clause's text as
/// a flat `String` would force every citation into it to name a single page —
/// and roughly half of them would then name the wrong one. Keeping the
/// page-attributed blocks means [`Clause::cite`] can resolve a quotation to
/// the page it is *actually* printed on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClauseLine {
    page: PageNumber,
    text: SourceText,
}

impl ClauseLine {
    /// One page-attributed block of clause text.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptySourceText`] if the text is blank.
    pub fn new(page: PageNumber, text: impl Into<String>) -> Result<Self, ProvenanceError> {
        Ok(Self {
            page,
            text: SourceText::new(text)?,
        })
    }

    /// The page this block is printed on.
    pub fn page(&self) -> PageNumber {
        self.page
    }

    /// The verbatim text of this block.
    pub fn text(&self) -> &str {
        self.text.as_str()
    }
}

/// A located, verbatim unit of an insurance document.
///
/// Every field is private and set once, in [`Clause::new`], which requires the
/// document, the identifier, the section path and at least one non-empty line
/// of text. A `Clause` with no source text does not exist.
///
/// Cross-references are **derived** from the text in the constructor rather
/// than passed in, so [`Clause::cross_references`] can never disagree with
/// [`Clause::text`]. An insurance policy is a graph of cross-references
/// ("subject to Clause 7", "as defined in Section 2"), not a linear text, and
/// a clause read without its references is routinely read wrongly.
///
/// A clause cannot be assembled field by field — the fields are private and
/// there is no `Default`, so this does not compile:
///
/// ```compile_fail
/// use kopitiam_insurance::{Clause, ClauseId, ClauseRole, SectionPath};
/// let clause = Clause {
///     id: ClauseId::printed("4.2").unwrap(),
///     role: ClauseRole::Exclusion,
///     path: SectionPath::default(),
///     // ... and no text, and no page.
/// };
/// ```
///
/// Nor can a clause be built with no text: [`Clause::new`] requires at least
/// one non-empty [`ClauseLine`], and a `ClauseLine` requires a [`PageNumber`]
/// and non-blank text. There is no path to a clause that does not say where it
/// is or what it says.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    document: DocumentId,
    id: ClauseId,
    heading: Option<String>,
    path: SectionPath,
    role: ClauseRole,
    lines: Vec<ClauseLine>,
    /// Cached join of `lines`, so `text()` can hand out a `&str`.
    text: SourceText,
    cross_references: Vec<CrossReference>,
}

impl Clause {
    /// Builds a clause from its page-attributed lines.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptySourceText`] if `lines` is empty — a clause
    /// with no text is not a clause.
    pub fn new(
        document: DocumentId,
        id: ClauseId,
        heading: Option<String>,
        path: SectionPath,
        role: ClauseRole,
        lines: Vec<ClauseLine>,
    ) -> Result<Self, ProvenanceError> {
        let joined = lines
            .iter()
            .map(ClauseLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let text = SourceText::new(joined)?;
        let cross_references = crossref::scan(text.as_str());

        Ok(Self {
            document,
            id,
            heading,
            path,
            role,
            lines,
            text,
            cross_references,
        })
    }

    /// The source document.
    pub fn document(&self) -> &DocumentId {
        &self.document
    }

    /// The clause identifier.
    pub fn id(&self) -> &ClauseId {
        &self.id
    }

    /// The clause's own heading, if it had one (`"Exclusions"` for a clause
    /// printed as `4. Exclusions`).
    pub fn heading(&self) -> Option<&str> {
        self.heading.as_deref()
    }

    /// The heading hierarchy this clause sits under.
    pub fn path(&self) -> &SectionPath {
        &self.path
    }

    /// What this clause does in the contract.
    pub fn role(&self) -> ClauseRole {
        self.role
    }

    /// The clause's verbatim text, blocks joined by newlines.
    pub fn text(&self) -> &str {
        self.text.as_str()
    }

    /// The page-attributed blocks the text was reconstructed from.
    pub fn lines(&self) -> &[ClauseLine] {
        &self.lines
    }

    /// The first page the clause appears on.
    pub fn page(&self) -> PageNumber {
        self.lines
            .first()
            .expect("Clause::new rejects an empty line list")
            .page()
    }

    /// Every page the clause appears on, in order, without repeats. More than
    /// one when the clause runs across a page break.
    pub fn pages(&self) -> Vec<PageNumber> {
        let mut pages: Vec<PageNumber> = Vec::new();
        for line in &self.lines {
            if pages.last() != Some(&line.page()) {
                pages.push(line.page());
            }
        }
        pages
    }

    /// Cross-references this clause makes to other clauses, derived from its
    /// own text. Resolve them with [`crate::PolicyDocument::resolve_reference`].
    pub fn cross_references(&self) -> &[CrossReference] {
        &self.cross_references
    }

    /// A citation to this clause as a whole. The verbatim text is the clause's
    /// full text; the page is the first page it appears on.
    pub fn provenance(&self) -> Provenance {
        Provenance::new(
            self.document.clone(),
            self.page(),
            self.path.clone(),
            self.id.clone(),
            self.text.clone(),
        )
    }

    /// A citation to a **fragment** of this clause — and the check that makes
    /// the word "verbatim" mean something.
    ///
    /// `fragment` must actually occur in the clause's text. If it does not,
    /// this returns [`ProvenanceError::QuoteNotInClause`] rather than a
    /// citation. A paraphrase, a summary, a tidied-up restatement, or an
    /// invented quotation therefore cannot be dressed up as the policy's own
    /// words: the quote is checked against the source before a `Provenance` is
    /// issued.
    ///
    /// The page in the returned citation is the page the fragment is *printed
    /// on*, resolved through the clause's page-attributed lines — not merely
    /// the clause's first page. A clause spanning pages 7 and 8 yields a
    /// citation to page 8 for text that is on page 8.
    ///
    /// Matching is whitespace-insensitive (the fragment `"war or invasion"`
    /// matches source text that a line break split as `"war  or\ninvasion"`),
    /// because the whitespace is an artefact of PDF line-breaking, not of the
    /// contract. Nothing else about the text is normalised: casing,
    /// punctuation and wording must match exactly.
    ///
    /// # Errors
    ///
    /// - [`ProvenanceError::EmptySourceText`] if `fragment` is blank.
    /// - [`ProvenanceError::QuoteNotInClause`] if `fragment` does not occur in
    ///   this clause.
    pub fn cite(&self, fragment: &str) -> Result<Provenance, ProvenanceError> {
        let verbatim = SourceText::new(fragment)?;
        let needle = squash_whitespace(verbatim.as_str());

        // Prefer the specific page: find the line the fragment is printed on.
        let page = self
            .lines
            .iter()
            .find(|line| squash_whitespace(line.text()).contains(&needle))
            .map(ClauseLine::page)
            // A fragment straddling two blocks of the clause is still a real
            // quotation; fall back to the clause's first page rather than
            // rejecting a citation that is honest about its text.
            .or_else(|| {
                squash_whitespace(self.text.as_str())
                    .contains(&needle)
                    .then(|| self.page())
            })
            .ok_or_else(|| ProvenanceError::QuoteNotInClause {
                clause: self.id.to_string(),
                fragment: verbatim.as_str().to_string(),
            })?;

        Ok(Provenance::new(
            self.document.clone(),
            page,
            self.path.clone(),
            self.id.clone(),
            verbatim,
        ))
    }

    /// Extracts a value backed by a quotation from this clause.
    ///
    /// The ergonomic front door to the provenance model: pass the value and
    /// the words that justify it, and get back an [`crate::ExtractedTerm`] —
    /// or an error, if the words are not in the clause.
    ///
    /// # Errors
    ///
    /// As [`Clause::cite`].
    pub fn extract<T>(
        &self,
        value: T,
        fragment: &str,
    ) -> Result<crate::ExtractedTerm<T>, ProvenanceError> {
        Ok(crate::ExtractedTerm::new(value, self.cite(fragment)?))
    }
}

/// Collapses every run of whitespace to a single space, for whitespace-
/// insensitive quotation matching. See [`Clause::cite`] for why.
pub(crate) fn squash_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> DocumentId {
        DocumentId::new("synthetic-policy.pdf").unwrap()
    }

    fn page(n: usize) -> PageNumber {
        PageNumber::new(n).unwrap()
    }

    /// A clause split across a page break, as a real policy wording routinely
    /// is. All wording here is invented for the test; it is not any insurer's.
    fn page_spanning_clause() -> Clause {
        Clause::new(
            document(),
            ClauseId::printed("4.2").unwrap(),
            Some("War and Related Perils".to_string()),
            SectionPath::new(["Section 4 — Exclusions"]),
            ClauseRole::Exclusion,
            vec![
                ClauseLine::new(page(7), "We will not pay for any loss caused by").unwrap(),
                ClauseLine::new(page(8), "war, invasion or acts of foreign enemies.").unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn clause_cannot_be_built_without_text() {
        let err = Clause::new(
            document(),
            ClauseId::printed("1").unwrap(),
            None,
            SectionPath::default(),
            ClauseRole::Unclassified,
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(err, ProvenanceError::EmptySourceText);
    }

    #[test]
    fn unnumbered_clause_id_cannot_be_mistaken_for_a_printed_one() {
        let id = ClauseId::Unnumbered {
            page: page(3),
            ordinal: 1,
        };
        assert_eq!(id.to_string(), "[unnumbered, p.3 #1]");
        assert_eq!(id.printed_label(), None);
    }

    #[test]
    fn citing_a_fragment_resolves_to_the_page_it_is_printed_on() {
        let clause = page_spanning_clause();
        assert_eq!(clause.page().get(), 7);
        assert_eq!(
            clause.pages().iter().map(|p| p.get()).collect::<Vec<_>>(),
            vec![7, 8]
        );

        let on_page_7 = clause.cite("We will not pay").unwrap();
        assert_eq!(on_page_7.page().get(), 7);

        // The critical case: text on page 8 must not be cited to page 7 just
        // because the clause *starts* on page 7.
        let on_page_8 = clause.cite("war, invasion").unwrap();
        assert_eq!(on_page_8.page().get(), 8);
        assert_eq!(on_page_8.verbatim().as_str(), "war, invasion");
    }

    #[test]
    fn citing_across_a_page_break_falls_back_to_the_clauses_first_page() {
        let clause = page_spanning_clause();
        let straddling = clause.cite("caused by war, invasion").unwrap();
        assert_eq!(straddling.page().get(), 7);
    }

    #[test]
    fn a_paraphrase_cannot_be_dressed_up_as_a_quotation() {
        // This is the single most important negative test in the crate: an
        // extractor that could attach text of its own invention to a real
        // clause would be able to produce citations that look authoritative
        // and say something the policy does not.
        let clause = page_spanning_clause();
        let err = clause
            .cite("war is excluded from this policy")
            .expect_err("a paraphrase is not a quotation and must be rejected");
        assert!(matches!(err, ProvenanceError::QuoteNotInClause { .. }));

        // ... and an extracted term built on it therefore cannot exist either.
        assert!(clause.extract(true, "war is excluded").is_err());
    }

    #[test]
    fn quotation_matching_ignores_pdf_line_breaking_but_nothing_else() {
        let clause = page_spanning_clause();
        // Whitespace across the block join is an artefact of the PDF, not the
        // contract, so this matches.
        assert!(clause.cite("caused by\n  war,   invasion").is_ok());
        // Casing is *not* an artefact, so this does not.
        assert!(clause.cite("WAR, INVASION").is_err());
    }

    #[test]
    fn extract_binds_a_value_to_a_real_quotation() {
        let clause = page_spanning_clause();
        let term = clause.extract("war", "war, invasion").unwrap();
        assert_eq!(*term.value(), "war");
        assert_eq!(term.provenance().page().get(), 8);
        assert_eq!(term.provenance().clause().to_string(), "4.2");
    }
}
