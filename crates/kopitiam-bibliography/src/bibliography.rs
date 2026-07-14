//! [`Bibliography`] — everything a document's references amount to.
//!
//! A `Bibliography` is what one document *knows about the literature*: the works
//! it cites, where in itself it cited them, which citations resolved and which
//! did not, and — kept together with all of that rather than filed away
//! separately — every assumption the engine made getting there.
//!
//! # Why the anomalies live in the same struct
//!
//! It would have been tidier to return `(Bibliography, Vec<Anomaly>)` and let
//! callers ignore the second half. That is exactly why it is not done.
//!
//! A caller who has a `Bibliography` in hand has, by construction, also got the
//! list of things that might be wrong with it. `bibliography.entries()` and
//! `bibliography.anomalies()` are the same object, and there is no ergonomic
//! path that drops one and keeps the other. The friction is the feature —
//! `kopitiam-web`'s `NullProvider` makes the same argument for the same reason.

use serde::{Deserialize, Serialize};

use crate::anomaly::Anomaly;
use crate::citation::{CitationRef, SourcedCitation};
use crate::entry::ParsedReference;
use crate::provenance::DocumentId;
use crate::reference::Reference;

/// A citation resolved to the work it names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedCitation {
    /// The in-text citation, and the page it was printed on.
    pub citation: SourcedCitation,
    /// The index into [`Bibliography::entries`] of the work it names.
    pub entry: usize,
}

/// One document's references and citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bibliography {
    document: DocumentId,
    entries: Vec<ParsedReference>,
    citations: Vec<SourcedCitation>,
    anomalies: Vec<Anomaly>,
}

impl Bibliography {
    /// Assembles a bibliography.
    pub fn new(
        document: DocumentId,
        entries: Vec<ParsedReference>,
        citations: Vec<SourcedCitation>,
        anomalies: Vec<Anomaly>,
    ) -> Self {
        Self {
            document,
            entries,
            citations,
            anomalies,
        }
    }

    /// The document these references were read from.
    pub fn document(&self) -> &DocumentId {
        &self.document
    }

    /// Every reference-list entry, **including the ones that did not parse**.
    ///
    /// Deliberately `&[ParsedReference]` rather than `&[Reference]`: a caller
    /// asking for the bibliography gets the failures too, in the same list, in
    /// reference-list order. Filtering them out has to be a *decision*, made
    /// with [`Self::references`], and not something that happens by default
    /// because the API made it the path of least resistance.
    pub fn entries(&self) -> &[ParsedReference] {
        &self.entries
    }

    /// Only the entries that yielded a [`Reference`] — parsed or partial.
    pub fn references(&self) -> impl Iterator<Item = &Reference> {
        self.entries.iter().filter_map(ParsedReference::reference)
    }

    /// The entries that could not be parsed at all.
    pub fn unparsed(&self) -> impl Iterator<Item = &ParsedReference> {
        self.entries.iter().filter(|entry| !entry.is_parsed())
    }

    /// The entries that were only partially understood.
    pub fn partial(&self) -> impl Iterator<Item = &ParsedReference> {
        self.entries.iter().filter(|entry| entry.is_partial())
    }

    /// Every in-text citation found in the document body.
    pub fn citations(&self) -> &[SourcedCitation] {
        &self.citations
    }

    /// **Everything the engine could not work out, and every assumption it
    /// made.** Read this.
    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// The assumptions specifically — the anomalies that could have produced a
    /// *confident wrong answer* rather than an honest gap.
    ///
    /// This is the list to audit first. See [`Anomaly::is_an_assumption`].
    pub fn assumptions(&self) -> impl Iterator<Item = &Anomaly> {
        self.anomalies.iter().filter(|a| a.is_an_assumption())
    }

    /// Resolves each in-text citation to the reference-list entry it names.
    ///
    /// # Numeric citations
    ///
    /// `[1]` means "entry 1 of the reference list" — the **printed label**, not
    /// an array index. The entries are held in reference-list order, so label
    /// *n* is `entries[n - 1]`.
    ///
    /// A label with no matching entry (`[13]` in a twelve-entry list) resolves
    /// to **nothing**. It is not rounded, not clamped, and not dropped: it comes
    /// back in [`Self::unresolved_citations`], because an unresolved citation is
    /// almost always a finding about *our own extraction* — we missed a
    /// reference — and that must be shouted, not swallowed.
    ///
    /// # Author-year citations
    ///
    /// Matched on the first author's **family name as printed** plus the year.
    /// A match requires both. `(Smith, 2019)` will not resolve against a Smith
    /// who published in 2018, and it will not resolve against a *Smyth* who
    /// published in 2019 — this crate does not do fuzzy name matching, because
    /// "close enough" on a person's name is how a citation ends up pointing at
    /// the wrong researcher.
    pub fn resolve_citations(&self) -> Vec<ResolvedCitation> {
        let mut resolved = Vec::new();

        for citation in &self.citations {
            match citation.citation() {
                CitationRef::Numeric(labels) => {
                    for &label in labels {
                        let index = (label as usize).checked_sub(1);
                        if let Some(index) = index
                            && index < self.entries.len()
                        {
                            resolved.push(ResolvedCitation {
                                citation: citation.clone(),
                                entry: index,
                            });
                        }
                    }
                }
                CitationRef::AuthorYear {
                    authors,
                    year,
                    suffix,
                    ..
                } => {
                    // A disambiguating suffix (`2019a`) means the source itself
                    // knew there were two candidate works. We have no way to
                    // tell which is `a` and which is `b` -- the letter is
                    // assigned by the citation STYLE at typesetting time, from
                    // an ordering we cannot see. Guessing would be a coin toss
                    // between two real papers, so we decline.
                    if suffix.is_some() {
                        continue;
                    }
                    let Some(first) = authors.first() else {
                        continue;
                    };
                    for (index, entry) in self.entries.iter().enumerate() {
                        let Some(reference) = entry.reference() else {
                            continue;
                        };
                        let matches_year = reference.year().is_some_and(|y| y.get() == *year);
                        let matches_author = reference
                            .authors()
                            .first()
                            .is_some_and(|author| author_matches(author, first));
                        if matches_year && matches_author {
                            resolved.push(ResolvedCitation {
                                citation: citation.clone(),
                                entry: index,
                            });
                            break;
                        }
                    }
                }
                CitationRef::Unrecognised(_) => {}
            }
        }

        resolved
    }

    /// The citations that could **not** be matched to any entry.
    ///
    /// A non-empty result usually means the reference list was extracted
    /// incompletely. It is a report card on this crate, and it is meant to be
    /// read as one.
    pub fn unresolved_citations(&self) -> Vec<&SourcedCitation> {
        let resolved = self.resolve_citations();
        self.citations
            .iter()
            .filter(|citation| {
                !resolved
                    .iter()
                    .any(|r| r.citation.provenance() == citation.provenance()
                        && r.citation.citation() == citation.citation())
            })
            .collect()
    }

    /// How many distinct works this document cites at least once.
    pub fn cited_entry_count(&self) -> usize {
        let mut seen: Vec<usize> = self
            .resolve_citations()
            .into_iter()
            .map(|r| r.entry)
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }

    /// The entries that are in the reference list but never cited in the body.
    ///
    /// Useful, and slightly judgemental. An uncited reference is not
    /// necessarily an error — it may be cited in a table, a caption, or an
    /// appendix that was not extracted — so this reports rather than complains.
    pub fn uncited_entries(&self) -> Vec<usize> {
        let cited: Vec<usize> = self.resolve_citations().into_iter().map(|r| r.entry).collect();
        (0..self.entries.len())
            .filter(|index| !cited.contains(index))
            .collect()
    }
}

/// Whether a reference's first author matches the family name printed in an
/// author-year citation.
///
/// Exact, case-insensitive, on the **family name** — with a fallback to the
/// name as written for the authors whose split we do not trust (see
/// [`crate::NameConfidence`]). No fuzzy matching, no edit distance, no
/// "close enough": a citation that resolves to the wrong researcher because two
/// surnames were one letter apart is precisely the failure this crate exists to
/// prevent.
fn author_matches(author: &crate::Author, printed: &str) -> bool {
    if let Some(family) = author.family()
        && family.eq_ignore_ascii_case(printed)
    {
        return true;
    }
    // For an `Assumed` name we have no trustworthy family name, so compare
    // against the whole name as written -- which will match when the citation
    // prints the same string, and will honestly fail to match otherwise.
    author.as_written().eq_ignore_ascii_case(printed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::parse_printed_name_list;
    use crate::citation::CitationRef;
    use crate::entry::RawEntry;
    use crate::provenance::Provenance;
    use crate::reference::{Reference, Year};

    fn doc() -> DocumentId {
        DocumentId::new("paper.pdf").unwrap()
    }

    fn reference(authors: &str, year: i32) -> ParsedReference {
        let provenance = Provenance::from_page(&doc(), 15, format!("{authors}, {year}.")).unwrap();
        ParsedReference::Parsed(
            Reference::builder(provenance)
                .authors(parse_printed_name_list(authors))
                .year(Year::new(year).unwrap())
                .title("A Title")
                .build(),
        )
    }

    fn citation(text: &str, page: usize) -> SourcedCitation {
        let provenance = Provenance::from_page(&doc(), page, text).unwrap();
        SourcedCitation::new(CitationRef::parse(text), provenance)
    }

    #[test]
    fn a_numeric_citation_resolves_to_the_entry_with_that_printed_label() {
        // `[2]` means the SECOND entry, not `entries[2]`.
        let bibliography = Bibliography::new(
            doc(),
            vec![
                reference("M. R. Chen", 2024),
                reference("L. Vega", 2021),
                reference("R. Okafor", 2015),
            ],
            vec![citation("[2]", 3)],
            Vec::new(),
        );

        let resolved = bibliography.resolve_citations();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].entry, 1, "label 2 is entries[1]");
    }

    #[test]
    fn a_range_citation_resolves_to_every_work_it_names() {
        let bibliography = Bibliography::new(
            doc(),
            vec![
                reference("A. One", 2020),
                reference("B. Two", 2021),
                reference("C. Three", 2022),
            ],
            vec![citation("[1-3]", 3)],
            Vec::new(),
        );
        assert_eq!(bibliography.resolve_citations().len(), 3);
        assert_eq!(bibliography.cited_entry_count(), 3);
    }

    #[test]
    fn a_citation_beyond_the_reference_list_resolves_to_nothing_and_is_reported() {
        // `[13]` in a twelve-entry list. NOT rounded down to 12, NOT dropped.
        // This almost always means WE missed a reference, and it must be loud.
        let bibliography = Bibliography::new(
            doc(),
            vec![reference("A. One", 2020)],
            vec![citation("[13]", 3)],
            Vec::new(),
        );

        assert!(bibliography.resolve_citations().is_empty());
        assert_eq!(bibliography.unresolved_citations().len(), 1);
    }

    #[test]
    fn an_author_year_citation_needs_both_the_author_and_the_year() {
        let bibliography = Bibliography::new(
            doc(),
            vec![reference("J. Smith", 2019), reference("A. Jones", 2018)],
            vec![
                citation("(Smith, 2019)", 3),
                citation("(Smith, 2018)", 4), // right author, wrong year
                citation("(Smyth, 2019)", 5), // one letter out
            ],
            Vec::new(),
        );

        let resolved = bibliography.resolve_citations();
        assert_eq!(resolved.len(), 1, "only the exact match resolves");
        assert_eq!(resolved[0].entry, 0);

        // The near-misses are reported, not fuzzy-matched. A citation that
        // resolves to the wrong researcher because two surnames were one letter
        // apart is the exact failure this crate exists to prevent.
        assert_eq!(bibliography.unresolved_citations().len(), 2);
    }

    #[test]
    fn a_suffixed_author_year_citation_declines_to_guess_between_two_works() {
        // The style assigned `a`/`b` from an ordering we cannot see. Picking one
        // would be a coin toss between two real papers.
        let bibliography = Bibliography::new(
            doc(),
            vec![reference("J. Smith", 2019), reference("J. Smith", 2019)],
            vec![citation("(Smith, 2019a)", 3)],
            Vec::new(),
        );
        assert!(bibliography.resolve_citations().is_empty());
        assert_eq!(bibliography.unresolved_citations().len(), 1);
    }

    #[test]
    fn a_citation_never_resolves_to_an_unparsed_entry_by_author_year() {
        let provenance = Provenance::from_page(&doc(), 15, "garbage line").unwrap();
        let bibliography = Bibliography::new(
            doc(),
            vec![ParsedReference::Unparsed(RawEntry::new(provenance))],
            vec![citation("(Smith, 2019)", 3)],
            Vec::new(),
        );
        assert!(bibliography.resolve_citations().is_empty());
    }

    #[test]
    fn uncited_entries_are_reported_without_complaint() {
        let bibliography = Bibliography::new(
            doc(),
            vec![reference("A. One", 2020), reference("B. Two", 2021)],
            vec![citation("[1]", 3)],
            Vec::new(),
        );
        assert_eq!(bibliography.uncited_entries(), vec![1]);
    }

    #[test]
    fn the_failures_are_in_the_same_list_as_the_successes() {
        // A caller with a Bibliography in hand has, by construction, also got
        // the list of things that might be wrong with it.
        let provenance = Provenance::from_page(&doc(), 15, "garbage line").unwrap();
        let bibliography = Bibliography::new(
            doc(),
            vec![
                reference("A. One", 2020),
                ParsedReference::Unparsed(RawEntry::new(provenance)),
            ],
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(bibliography.entries().len(), 2, "both, in reference-list order");
        assert_eq!(bibliography.references().count(), 1);
        assert_eq!(bibliography.unparsed().count(), 1);
    }
}
