//! What the engine could not work out — and the assumptions it made anyway.
//!
//! # Why "what I don't know" is a first-class output
//!
//! A bibliography extractor that reports only its confident findings **lies by
//! omission**. The reader of a clean-looking twelve-entry bibliography has no
//! way to know that three of the entries had their tails silently dropped, that
//! one page number rested on an assumption about digit grouping, or that five
//! "books" are actually PhD theses. They will believe all twelve equally.
//!
//! So every assumption and every shortfall becomes an [`Anomaly`], and the
//! anomalies travel all the way through: into
//! [`Bibliography::anomalies`](crate::Bibliography::anomalies), into the
//! emitted `.bib` file's `note` fields, and into the knowledge graph as
//! `Fact` entities. There is nowhere in this crate where they get quietly
//! filtered out.
//!
//! This mirrors `kopitiam-insurance`'s `Anomaly` and `kopitiam-plot`'s
//! `warnings` deliberately: it is the house style, and it is the house style
//! because it is the only honest way to ship a heuristic extractor.
//!
//! # Every anomaly carries its provenance
//!
//! An anomaly that cannot be traced back to a page and a verbatim string is
//! just an anxiety. Each variant therefore carries a [`Provenance`], so a human
//! can be shown *"page 15 says this, and here is what I could not do with it"*.

use serde::{Deserialize, Serialize};

use crate::provenance::Provenance;
use crate::reference::EntryKind;

/// Something the engine could not determine, or determined only by assumption.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anomaly {
    /// A reference-list line yielded nothing at all. The raw text is kept.
    UnparseableEntry {
        /// The line, and where it was printed.
        provenance: Provenance,
    },

    /// A reference was recovered, but part of its line was not understood.
    PartialEntry {
        /// The line, and where it was printed.
        provenance: Provenance,
        /// The text that could not be accounted for.
        remainder: String,
    },

    /// The kind of work could not be determined, and one reading was chosen.
    ///
    /// The important instance: a **thesis printed exactly like a book**. The
    /// `biblatex` `ieee` style drops the "PhD thesis" designator, so
    /// *"Title. University of California, Berkeley, 2024."* is a dissertation
    /// and a monograph in the same words. Five of the twelve references in the
    /// maintainer's own paper are this case.
    AmbiguousEntryKind {
        /// The line, and where it was printed.
        provenance: Provenance,
        /// What was chosen (the reading the string literally supports).
        chosen: EntryKind,
        /// What it might equally be.
        alternative: EntryKind,
        /// Why the two cannot be told apart here.
        reason: String,
    },

    /// A number was printed with an internal space, read as a digit-group
    /// separator.
    ///
    /// `p. 111 144` is `biblatex` printing article number `111144`. Joining is
    /// almost certainly right — and *almost certainly* is exactly what has to be
    /// declared rather than assumed silently.
    AssumedDigitGrouping {
        /// The line, and where it was printed.
        provenance: Provenance,
        /// The number as printed: `111 144`.
        printed: String,
        /// How it was read: `111144`.
        read_as: String,
    },

    /// An identifier was printed and **failed validation** — a bad ISBN check
    /// digit, a malformed DOI.
    ///
    /// Reported rather than accepted, because accepting it would put a *wrong
    /// identifier* into a bibliography, where it resolves to somebody else's
    /// work. Almost always an OCR error or a typo, and reporting it is how it
    /// gets fixed.
    InvalidIdentifier {
        /// The line, and where it was printed.
        provenance: Provenance,
        /// What was wrong with it.
        reason: String,
    },

    /// A line break inside a word forced a hyphen decision that the text alone
    /// cannot settle. See [`crate::text`].
    AssumedHyphenation {
        /// The line, and where it was printed.
        provenance: Provenance,
        /// The reconstructed word, so a human can check it.
        reconstructed: String,
        /// Whether the hyphen was removed (`true`) or kept (`false`).
        hyphen_removed: bool,
    },

    /// An in-text citation could not be matched to any entry in the reference
    /// list.
    ///
    /// Usually means the reference list was extracted incompletely — which is a
    /// **finding about our own extraction**, and one worth surfacing loudly.
    UnresolvedCitation {
        /// The citation, and where it was printed.
        provenance: Provenance,
        /// The citation text (`[13]`).
        citation: String,
    },

    /// The reference section could not be found in the document at all.
    NoReferenceSection {
        /// The document.
        provenance: Provenance,
    },
}

impl Anomaly {
    /// Where the anomaly was found.
    pub fn provenance(&self) -> &Provenance {
        match self {
            Self::UnparseableEntry { provenance }
            | Self::PartialEntry { provenance, .. }
            | Self::AmbiguousEntryKind { provenance, .. }
            | Self::AssumedDigitGrouping { provenance, .. }
            | Self::InvalidIdentifier { provenance, .. }
            | Self::AssumedHyphenation { provenance, .. }
            | Self::UnresolvedCitation { provenance, .. }
            | Self::NoReferenceSection { provenance } => provenance,
        }
    }

    /// The source's own words that this anomaly is about.
    ///
    /// Every anomaly can show a human the text it is complaining about. An
    /// anomaly that could not would be an anxiety, not a finding.
    pub fn verbatim(&self) -> &str {
        self.provenance().verbatim().as_str()
    }

    /// Whether this anomaly means **we made an assumption that could be wrong**,
    /// as opposed to **we admitted we did not know**.
    ///
    /// The distinction matters enormously to a reader. "I could not parse line
    /// 7" is safe: nothing downstream will believe anything false. "I assumed
    /// `111 144` means `111144`" is *not* safe in the same way: a wrong
    /// assumption here produces a confident, plausible, incorrect page number.
    ///
    /// Callers auditing an extraction should look at these **first**.
    pub fn is_an_assumption(&self) -> bool {
        matches!(
            self,
            Self::AmbiguousEntryKind { .. }
                | Self::AssumedDigitGrouping { .. }
                | Self::AssumedHyphenation { .. }
        )
    }

    /// A one-line human-readable summary.
    pub fn summary(&self) -> String {
        match self {
            Self::UnparseableEntry { provenance } => {
                format!("could not parse a reference at {provenance}")
            }
            Self::PartialEntry { provenance, remainder } => {
                format!("partially parsed a reference at {provenance}; not understood: {remainder:?}")
            }
            Self::AmbiguousEntryKind {
                provenance,
                chosen,
                alternative,
                reason,
            } => format!(
                "entry kind at {provenance} is ambiguous: read as `{chosen}`, may be `{alternative}` ({reason})"
            ),
            Self::AssumedDigitGrouping {
                provenance,
                printed,
                read_as,
            } => format!(
                "ASSUMPTION at {provenance}: read the page number {printed:?} as {read_as:?} \
                 (an internal space taken to be a digit-group separator)"
            ),
            Self::InvalidIdentifier { provenance, reason } => {
                format!("invalid identifier at {provenance}: {reason}")
            }
            Self::AssumedHyphenation {
                provenance,
                reconstructed,
                hyphen_removed,
            } => format!(
                "ASSUMPTION at {provenance}: a line-break hyphen was {} giving {reconstructed:?}",
                if *hyphen_removed { "removed" } else { "kept" }
            ),
            Self::UnresolvedCitation {
                provenance,
                citation,
            } => format!(
                "the citation {citation} at {provenance} matches no entry in the reference list \
                 (the reference list may have been extracted incompletely)"
            ),
            Self::NoReferenceSection { provenance } => {
                format!("no reference section was found in {provenance}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DocumentId;

    fn provenance() -> Provenance {
        let doc = DocumentId::new("paper.pdf").unwrap();
        Provenance::from_page(&doc, 15, "p. 111 144").unwrap()
    }

    #[test]
    fn an_assumption_is_distinguished_from_an_admission_of_ignorance() {
        // The distinction that matters to a reader auditing an extraction.
        let assumption = Anomaly::AssumedDigitGrouping {
            provenance: provenance(),
            printed: "111 144".to_string(),
            read_as: "111144".to_string(),
        };
        assert!(
            assumption.is_an_assumption(),
            "a wrong assumption produces a confident, plausible, INCORRECT answer"
        );

        let admission = Anomaly::UnparseableEntry {
            provenance: provenance(),
        };
        assert!(
            !admission.is_an_assumption(),
            "an admission of ignorance cannot mislead anyone"
        );
    }

    #[test]
    fn every_anomaly_can_show_the_words_it_is_about() {
        let anomaly = Anomaly::UnparseableEntry {
            provenance: provenance(),
        };
        assert_eq!(anomaly.verbatim(), "p. 111 144");
        assert!(anomaly.summary().contains("paper.pdf, p.15"));
    }

    #[test]
    fn an_assumption_summary_shouts_about_itself() {
        let anomaly = Anomaly::AssumedDigitGrouping {
            provenance: provenance(),
            printed: "111 144".to_string(),
            read_as: "111144".to_string(),
        };
        assert!(anomaly.summary().contains("ASSUMPTION"));
        assert!(anomaly.summary().contains("111144"));
    }
}
