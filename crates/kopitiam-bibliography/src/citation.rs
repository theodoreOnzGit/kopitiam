//! In-text citations: `[1]`, `(Smith, 2019)`, `Smith et al. 2019`.
//!
//! # The gap this fills
//!
//! [`kopitiam_document`] already **detects** citations in paragraph text. Its
//! entire model of one is:
//!
//! ```ignore
//! pub struct Citation {
//!     pub text: String,
//! }
//! ```
//!
//! That is a faithful record that a citation was *seen*, and it deliberately
//! goes no further — the Document Engine's job is layout, not literature. But a
//! `String` cannot answer any of the questions a citation exists to answer:
//! *which* work is being cited, does it appear in the reference list, is it
//! cited anywhere else, and — the one that turns a document into knowledge —
//! **what does this paper cite?**
//!
//! This module turns that string into a [`CitationRef`], and
//! [`crate::Bibliography`] resolves it against the reference list. The result is
//! the edge `paper A cites paper B`, which is what enters the knowledge graph
//! (see [`crate::knowledge`]) and is the entire point of the exercise.
//!
//! # Never invent a target
//!
//! [`CitationRef::Unrecognised`] exists and is used. A citation-shaped string
//! that we cannot read is kept as a string. A numeric citation `[13]` in a
//! document whose reference list only has twelve entries resolves to
//! **nothing**, loudly ([`Anomaly::UnresolvedCitation`]) — it does not get
//! rounded down to `[12]`, and it does not get quietly dropped.
//!
//! An unresolved citation is usually a finding about *our own extraction* (we
//! missed a reference), which is exactly the sort of thing that must be shouted
//! rather than swallowed.
//!
//! [`Anomaly::UnresolvedCitation`]: crate::Anomaly::UnresolvedCitation

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::provenance::Provenance;
use crate::text::fold_typography;

/// `[1]`, `[1, 2]`, `[1-3]`, `[1,2,5-7]`.
static NUMERIC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\s*(\d+(?:\s*[-,]\s*\d+)*)\s*\]$").unwrap());

/// `(Smith, 2019)`, `(Smith and Jones, 2019)`, `(Smith et al., 2019a)`.
static AUTHOR_YEAR_PAREN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\(\s*(.+?)\s*,?\s*(\d{4})([a-z])?\s*\)$").unwrap()
});

/// `Smith et al. 2019`, `Smith and Jones (2019)` — the narrative form, where the
/// author is part of the sentence.
static AUTHOR_YEAR_NARRATIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(.+?)\s*\(?\s*(\d{4})([a-z])?\s*\)?$").unwrap()
});

/// A structured in-text citation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationRef {
    /// A numeric citation: `[1]`, `[1, 2]`, `[1-3]`.
    ///
    /// The `Vec` holds every label the citation names, **ranges expanded**, so
    /// `[1-3]` is `[1, 2, 3]`. That expansion is safe (a numeric range means
    /// exactly what it says) and it is what makes a citation graph possible: a
    /// paper citing `[1-3]` really does cite three works.
    Numeric(Vec<u32>),

    /// An author-year citation: `(Smith, 2019)`, `Smith et al. 2019`.
    AuthorYear {
        /// The author names as printed — **not** parsed into [`crate::Author`],
        /// because an in-text citation prints family names only and running the
        /// full name grammar over `Smith` would manufacture certainty about a
        /// person from a single word.
        authors: Vec<String>,
        /// The year.
        year: i32,
        /// Whether the citation said `et al.`
        et_al: bool,
        /// The disambiguating suffix, if the source printed one (`2019a`).
        ///
        /// **Load-bearing.** `(Smith 2019a)` and `(Smith 2019b)` are *different
        /// works*, and a resolver that ignored the letter would merge two papers
        /// into one — which is a misattribution, not a formatting slip.
        suffix: Option<char>,
    },

    /// A citation-shaped string that could not be read.
    ///
    /// Kept, not discarded. A citation we cannot parse is still evidence that
    /// the author cited *something* here, and that is worth knowing.
    Unrecognised(String),
}

impl CitationRef {
    /// Parses a raw citation string — typically one that
    /// [`kopitiam_document`] found in a paragraph.
    ///
    /// Never fails: an unreadable string becomes [`Self::Unrecognised`].
    ///
    /// ```
    /// use kopitiam_bibliography::citation::CitationRef;
    ///
    /// assert_eq!(CitationRef::parse("[1]"), CitationRef::Numeric(vec![1]));
    /// // A range means what it says, so expanding it is safe.
    /// assert_eq!(CitationRef::parse("[1-3]"), CitationRef::Numeric(vec![1, 2, 3]));
    /// assert_eq!(CitationRef::parse("[1, 4, 9]"), CitationRef::Numeric(vec![1, 4, 9]));
    ///
    /// // Garbage stays garbage. It does not become a citation of paper 1.
    /// assert!(matches!(
    ///     CitationRef::parse("(see the appendix)"),
    ///     CitationRef::Unrecognised(_),
    /// ));
    /// ```
    pub fn parse(raw: &str) -> Self {
        let text = fold_typography(raw.trim());

        if let Some(caps) = NUMERIC.captures(&text) {
            let mut labels = Vec::new();
            for part in caps[1].split(',') {
                let part = part.trim();
                match part.split_once('-') {
                    Some((start, end)) => {
                        // A range. Expand it -- `[1-3]` really does cite three
                        // works, and a citation graph that recorded one would be
                        // wrong about what the paper depends on.
                        let (Ok(start), Ok(end)) =
                            (start.trim().parse::<u32>(), end.trim().parse::<u32>())
                        else {
                            continue;
                        };
                        // A backwards or absurd range is a typo, not a citation
                        // of ten thousand papers. Refuse it rather than
                        // allocating for it.
                        if start > end || end.saturating_sub(start) > 512 {
                            continue;
                        }
                        labels.extend(start..=end);
                    }
                    None => {
                        if let Ok(label) = part.parse::<u32>() {
                            labels.push(label);
                        }
                    }
                }
            }
            if !labels.is_empty() {
                return Self::Numeric(labels);
            }
        }

        if let Some(caps) = AUTHOR_YEAR_PAREN
            .captures(&text)
            .or_else(|| AUTHOR_YEAR_NARRATIVE.captures(&text))
        {
            let raw_authors = &caps[1];
            let et_al = raw_authors.to_lowercase().contains("et al");
            let year: i32 = caps[2].parse().unwrap_or(0);
            let suffix = caps.get(3).and_then(|m| m.as_str().chars().next());

            let authors = split_citation_authors(raw_authors);
            if !authors.is_empty() && (1000..=2999).contains(&year) {
                return Self::AuthorYear {
                    authors,
                    year,
                    et_al,
                    suffix,
                };
            }
        }

        Self::Unrecognised(raw.trim().to_string())
    }

    /// The numeric labels this citation names, if it is numeric.
    pub fn labels(&self) -> &[u32] {
        match self {
            Self::Numeric(labels) => labels,
            _ => &[],
        }
    }

    /// Whether the citation could be read at all.
    pub fn is_recognised(&self) -> bool {
        !matches!(self, Self::Unrecognised(_))
    }
}

/// Splits the author part of an author-year citation on `and`, `&`, and commas.
///
/// Returns family names as printed.
///
/// # `et al.` is a marker, not a person
///
/// It is stripped **before** splitting, not filtered out afterwards, because
/// `"Cohen et al."` is a single chunk — there is no comma and no `and` to split
/// it on — so a post-hoc filter never sees the `et al.` in isolation and lets an
/// author named `"Cohen et al"` straight through. An author called *al.* is a bug
/// real bibliography software has actually shipped, and this is exactly how.
///
/// # What counts as a name
///
/// At least two alphabetic characters, beginning with an upper-case letter.
/// Without that guard, `"(2019)"` parses as a citation by an author called `"("`
/// — which is not a hypothetical, it is what the first draft of this function
/// did.
fn split_citation_authors(text: &str) -> Vec<String> {
    // Strip the truncation marker first. See the doc comment.
    let mut body = text.trim();
    for marker in ["et al.", "et al", "et. al.", "and others"] {
        if let Some(index) = body.to_lowercase().find(marker) {
            body = body[..index].trim_end_matches([',', ' ', '.']).trim();
            break;
        }
    }

    body.replace(" & ", " and ")
        .replace(',', " and ")
        .split(" and ")
        .map(|name| name.trim().trim_end_matches('.').trim())
        .filter(|name| is_a_plausible_family_name(name))
        .map(str::to_string)
        .collect()
}

/// Whether a token from an author-year citation could be somebody's family name.
///
/// Deliberately strict. A false negative costs an unresolved citation, which is
/// reported. A false positive costs a citation attributed to an author called
/// `"("`, which is not reported because nothing looks wrong with it.
fn is_a_plausible_family_name(name: &str) -> bool {
    let letters = name.chars().filter(|c| c.is_alphabetic()).count();
    letters >= 2
        && name
            .chars()
            .find(|c| c.is_alphabetic())
            .is_some_and(char::is_uppercase)
}

/// An in-text citation, with the page it was printed on.
///
/// The provenance is what makes a citation graph *checkable*: an edge saying
/// "this paper cites Okafor (2015)" is only worth anything if a human can be
/// told **where on which page** the claim was made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcedCitation {
    citation: CitationRef,
    provenance: Provenance,
}

impl SourcedCitation {
    /// Records a citation and where it was printed.
    pub fn new(citation: CitationRef, provenance: Provenance) -> Self {
        Self {
            citation,
            provenance,
        }
    }

    /// The parsed citation.
    pub fn citation(&self) -> &CitationRef {
        &self.citation
    }

    /// Where it was printed, and the words around it.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_numeric_style_ieee_papers_use() {
        // An IEEE-style paper cites [1], [4], [9], [10] and so on throughout.
        assert_eq!(CitationRef::parse("[1]"), CitationRef::Numeric(vec![1]));
        assert_eq!(CitationRef::parse("[12]"), CitationRef::Numeric(vec![12]));
        assert_eq!(
            CitationRef::parse("[1, 2]"),
            CitationRef::Numeric(vec![1, 2])
        );
        assert_eq!(
            CitationRef::parse("[1,4,9]"),
            CitationRef::Numeric(vec![1, 4, 9])
        );
    }

    #[test]
    fn a_numeric_range_is_expanded_because_it_really_does_cite_every_one() {
        assert_eq!(
            CitationRef::parse("[1-3]"),
            CitationRef::Numeric(vec![1, 2, 3])
        );
        assert_eq!(
            CitationRef::parse("[1, 3-5, 9]"),
            CitationRef::Numeric(vec![1, 3, 4, 5, 9])
        );
    }

    #[test]
    fn an_absurd_range_is_refused_rather_than_allocated_for() {
        // `[5-1]` is a typo, not a citation. `[1-99999]` is a typo, not a
        // citation of ninety-nine thousand papers.
        assert!(matches!(
            CitationRef::parse("[5-1]"),
            CitationRef::Unrecognised(_)
        ));
        assert!(matches!(
            CitationRef::parse("[1-99999]"),
            CitationRef::Unrecognised(_)
        ));
    }

    #[test]
    fn parses_parenthesised_author_year() {
        let citation = CitationRef::parse("(Smith, 2019)");
        assert_eq!(
            citation,
            CitationRef::AuthorYear {
                authors: vec!["Smith".to_string()],
                year: 2019,
                et_al: false,
                suffix: None,
            }
        );
    }

    #[test]
    fn parses_two_authors_and_et_al() {
        assert_eq!(
            CitationRef::parse("(Smith and Jones, 2019)"),
            CitationRef::AuthorYear {
                authors: vec!["Smith".to_string(), "Jones".to_string()],
                year: 2019,
                et_al: false,
                suffix: None,
            }
        );

        let citation = CitationRef::parse("(Cohen et al., 1991)");
        let CitationRef::AuthorYear { authors, et_al, .. } = &citation else {
            panic!("expected author-year, got {citation:?}");
        };
        assert!(et_al, "the truncation must be recorded");
        assert_eq!(authors, &["Cohen".to_string()]);
        assert!(
            !authors.iter().any(|a| a.contains("al")),
            "there must be no author called `al.`"
        );
    }

    #[test]
    fn parses_the_narrative_form() {
        // "as Vega et al. (2021) showed" -- the author is part of the sentence.
        let citation = CitationRef::parse("Vega et al. 2021");
        let CitationRef::AuthorYear { authors, year, et_al, .. } = &citation else {
            panic!("expected author-year, got {citation:?}");
        };
        assert_eq!(authors, &["Vega".to_string()]);
        assert_eq!(*year, 2021);
        assert!(et_al);
    }

    #[test]
    fn the_disambiguating_suffix_is_kept_because_2019a_and_2019b_are_different_papers() {
        // A resolver that ignored the letter would merge two works into one --
        // a misattribution, not a formatting slip.
        let a = CitationRef::parse("(Smith, 2019a)");
        let b = CitationRef::parse("(Smith, 2019b)");
        assert_ne!(a, b, "2019a and 2019b are different works");

        let CitationRef::AuthorYear { suffix, .. } = a else {
            panic!("expected author-year");
        };
        assert_eq!(suffix, Some('a'));
    }

    #[test]
    fn garbage_is_unrecognised_and_never_becomes_a_citation_of_paper_one() {
        for text in ["(see the appendix)", "[]", "[abc]", "", "(2019)", "()"] {
            assert!(
                !CitationRef::parse(text).is_recognised(),
                "{text:?} must not resolve to a citation"
            );
        }
    }

    #[test]
    fn a_year_out_of_range_is_not_an_author_year_citation() {
        // "(Figure 3, 12)" must not become a citation of a work from year 12.
        assert!(!CitationRef::parse("(Figure 3, 12)").is_recognised());
    }

    #[test]
    fn typographic_quotes_and_dashes_are_folded_first() {
        // A citation printed with an en-dash range: [1\u{2013}3].
        assert_eq!(
            CitationRef::parse("[1\u{2013}3]"),
            CitationRef::Numeric(vec![1, 2, 3])
        );
    }

    #[test]
    fn round_trips_through_json() {
        let citation = CitationRef::parse("[1, 3-5]");
        let json = serde_json::to_string(&citation).unwrap();
        assert_eq!(serde_json::from_str::<CitationRef>(&json).unwrap(), citation);
    }
}
