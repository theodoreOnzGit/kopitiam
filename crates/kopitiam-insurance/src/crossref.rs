//! Cross-references between clauses.
//!
//! An insurance policy is not a linear text. It is a **graph**: "subject to
//! Clause 7", "the Exclusions in Section 4 apply", "as defined in Clause 2.1".
//! A clause read on its own, without following its references, is routinely
//! read wrongly — a grant of cover that is silently gutted by the clause it
//! defers to is the classic example.
//!
//! So references are extracted as links, resolved against the document's
//! actual clauses, and — importantly — **a reference that resolves to nothing
//! is reported, not dropped**. A dangling cross-reference in a legal document
//! is a real defect (usually a renumbering that was never propagated), and it
//! means the clause cannot be fully read. Swallowing it would hide that.
//!
//! # What is detected
//!
//! A referencing keyword (`Clause`, `Section`, `Sub-clause`, `Paragraph`,
//! `Item`, `Part`, `Endorsement`, singular or plural, any casing) followed by
//! one or more dotted-decimal numbers joined by `,`, `and`, `or`, `to` or `-`.
//! `"Clauses 7 and 8"` yields two references; `"Clause 4.2.1"` yields one.
//!
//! # What is deliberately *not* detected
//!
//! Alphabetic and roman sub-clause labels (`Clause (a)`, `Part IV`). Detecting
//! them reliably needs the enclosing clause's own numbering scheme, which we
//! do not always recover — and a cross-reference resolved to the *wrong*
//! clause is worse than one honestly not resolved at all. See
//! `kopitiam-hvi.1` for the follow-up.

use serde::{Deserialize, Serialize};

use crate::clause::ClauseId;
use crate::provenance::SourceText;

/// The word the document used to make the reference. Kept because policies do
/// distinguish them (a "Section" is usually a group of "Clauses"), and a
/// consumer may want to weigh them differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    Clause,
    SubClause,
    Section,
    Paragraph,
    Item,
    Part,
    Endorsement,
}

/// Keywords that introduce a cross-reference, longest first so that
/// `"Sub-clause"` is matched before `"clause"` would be.
const KEYWORDS: &[(&str, ReferenceKind)] = &[
    ("sub-clauses", ReferenceKind::SubClause),
    ("sub-clause", ReferenceKind::SubClause),
    ("subclauses", ReferenceKind::SubClause),
    ("subclause", ReferenceKind::SubClause),
    ("endorsements", ReferenceKind::Endorsement),
    ("endorsement", ReferenceKind::Endorsement),
    ("paragraphs", ReferenceKind::Paragraph),
    ("paragraph", ReferenceKind::Paragraph),
    ("sections", ReferenceKind::Section),
    ("section", ReferenceKind::Section),
    ("clauses", ReferenceKind::Clause),
    ("clause", ReferenceKind::Clause),
    ("items", ReferenceKind::Item),
    ("item", ReferenceKind::Item),
    ("parts", ReferenceKind::Part),
    ("part", ReferenceKind::Part),
];

/// Words that continue a list of reference targets: `"Clauses 7, 8 and 9"`.
const LIST_JOINERS: &[&str] = &["and", "or", "to", "through", "&"];

/// A reference from one clause to another, as printed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossReference {
    kind: ReferenceKind,
    target: ClauseId,
    raw: SourceText,
}

impl CrossReference {
    /// The clause this reference points at.
    pub fn target(&self) -> &ClauseId {
        &self.target
    }

    /// What the document called it (`Clause`, `Section`, ...).
    pub fn kind(&self) -> ReferenceKind {
        self.kind
    }

    /// The reference exactly as printed, e.g. `"Clauses 7 and 8"`. This is a
    /// verbatim substring of the referring clause, so it can be cited with
    /// [`crate::Clause::cite`].
    pub fn raw(&self) -> &str {
        self.raw.as_str()
    }
}

/// A cross-reference, once looked up against the document's actual clauses.
///
/// The `Dangling` variant is the point of this enum. A reference to a clause
/// that does not exist is surfaced — the alternative, silently returning
/// nothing, would let a reader believe they had followed a clause to its
/// conclusion when in fact a piece of the contract is missing.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedReference<'a> {
    /// The reference points at a clause that exists.
    Resolved {
        /// The reference as printed.
        reference: &'a CrossReference,
        /// The clause it points at.
        target: &'a crate::Clause,
    },
    /// The reference points at a clause **that is not in this document**.
    /// Report this to the reader; do not pretend the clause was fully read.
    Dangling {
        /// The reference as printed.
        reference: &'a CrossReference,
    },
}

impl<'a> ResolvedReference<'a> {
    /// The clause referred to, if it exists.
    pub fn target(&self) -> Option<&'a crate::Clause> {
        match self {
            Self::Resolved { target, .. } => Some(target),
            Self::Dangling { .. } => None,
        }
    }

    /// Whether the reference points at nothing.
    pub fn is_dangling(&self) -> bool {
        matches!(self, Self::Dangling { .. })
    }
}

/// Finds every cross-reference in a clause's text.
///
/// Hand-rolled rather than regex-driven: the grammar is small, the scan is
/// linear, and it keeps `kopitiam-insurance`'s dependency list to the
/// engines it genuinely builds on.
pub(crate) fn scan(text: &str) -> Vec<CrossReference> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut references = Vec::new();
    let mut cursor = 0;

    while cursor < bytes.len() {
        let Some((keyword_len, kind)) = keyword_at(&lower, cursor) else {
            cursor += 1;
            continue;
        };

        let start = cursor;
        let mut scan_at = cursor + keyword_len;
        let mut targets: Vec<(String, usize)> = Vec::new();

        // Consume `<number> [(, | and | or | to) <number>]*` after the keyword.
        loop {
            let after_space = skip_spaces(bytes, scan_at);
            let Some((number, number_end)) = number_at(text, &lower, after_space) else {
                break;
            };
            targets.push((number, number_end));
            scan_at = number_end;

            let after_number = skip_spaces(bytes, scan_at);
            match joiner_at(&lower, after_number) {
                Some(joiner_len) => scan_at = after_number + joiner_len,
                None => break,
            }
        }

        if targets.is_empty() {
            // A bare "in this Section" — a word, not a reference.
            cursor += keyword_len;
            continue;
        }

        let end = targets
            .last()
            .map(|&(_, end)| end)
            .expect("targets is non-empty");
        // The raw span covers the whole reference, list and all, so that every
        // target minted from `"Clauses 7 and 8"` cites text that really does
        // occur in the clause.
        let raw = SourceText::new(&text[start..end])
            .expect("a matched reference span is never blank");

        for (number, _) in targets {
            let Ok(target) = ClauseId::printed(number) else {
                continue;
            };
            references.push(CrossReference {
                kind,
                target,
                raw: raw.clone(),
            });
        }

        cursor = end;
    }

    references
}

/// Matches a reference keyword at `at`, requiring word boundaries on both
/// sides so that `"declause"` and `"sectional"` are not keywords.
///
/// Compares **bytes**, not string slices. `&lower[at..end] == keyword` looks
/// equivalent and is not: `end` is `at + keyword.len()`, which routinely lands
/// in the middle of a multi-byte character (an em-dash in `"Section 2 —
/// Definitions"` is three bytes), and slicing a `&str` across a character
/// boundary panics. Byte comparison cannot match a keyword at a non-boundary
/// anyway — a UTF-8 continuation byte is never an ASCII letter — so this is
/// both correct and panic-free.
fn keyword_at(lower: &str, at: usize) -> Option<(usize, ReferenceKind)> {
    let bytes = lower.as_bytes();
    if at > 0 && is_word_byte(bytes[at - 1]) {
        return None;
    }
    KEYWORDS.iter().find_map(|&(keyword, kind)| {
        let end = at + keyword.len();
        if end <= bytes.len()
            && &bytes[at..end] == keyword.as_bytes()
            && (end == bytes.len() || !is_word_byte(bytes[end]))
        {
            Some((keyword.len(), kind))
        } else {
            None
        }
    })
}

/// Matches a dotted-decimal clause number at `at`, e.g. `7` or `4.2.1`. A
/// trailing `.` (sentence punctuation) is not part of the number.
fn number_at(text: &str, lower: &str, at: usize) -> Option<(String, usize)> {
    let bytes = lower.as_bytes();
    if at >= bytes.len() || !bytes[at].is_ascii_digit() {
        return None;
    }

    let mut end = at;
    while end < bytes.len() {
        let byte = bytes[end];
        // A digit, or an interior '.' with a digit after it ("4.2.1"). A
        // trailing '.' is sentence punctuation and stops the scan.
        let in_number = byte.is_ascii_digit()
            || (byte == b'.' && end + 1 < bytes.len() && bytes[end + 1].is_ascii_digit());
        if !in_number {
            break;
        }
        end += 1;
    }

    Some((text[at..end].to_string(), end))
}

/// Matches a list joiner (`,`, `and`, `or`, `to`, ...) at `at`, returning how
/// many bytes it spans *including* the separator punctuation.
fn joiner_at(lower: &str, at: usize) -> Option<usize> {
    let bytes = lower.as_bytes();
    if at >= bytes.len() {
        return None;
    }

    if bytes[at] == b',' {
        // "Clauses 7, 8 and 9" — a comma may itself be followed by a word
        // joiner, but the outer loop's `skip_spaces` + next-number attempt
        // handles "7, 8" directly, and the "and" case is picked up on the
        // following iteration.
        return Some(1);
    }

    // Byte comparison, for the same reason as `keyword_at`.
    LIST_JOINERS.iter().find_map(|joiner| {
        let end = at + joiner.len();
        if end <= bytes.len()
            && &bytes[at..end] == joiner.as_bytes()
            && (end == bytes.len() || !is_word_byte(bytes[end]))
        {
            Some(joiner.len())
        } else {
            None
        }
    })
}

fn skip_spaces(bytes: &[u8], mut at: usize) -> usize {
    while at < bytes.len() && bytes[at].is_ascii_whitespace() {
        at += 1;
    }
    at
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(text: &str) -> Vec<String> {
        scan(text)
            .iter()
            .map(|reference| reference.target().to_string())
            .collect()
    }

    #[test]
    fn finds_a_simple_reference() {
        let references = scan("Cover under this clause is subject to Clause 7.");
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].target().to_string(), "7");
        assert_eq!(references[0].kind(), ReferenceKind::Clause);
        assert_eq!(references[0].raw(), "Clause 7");
    }

    #[test]
    fn finds_a_dotted_reference_and_stops_at_sentence_punctuation() {
        let references = scan("as defined in Clause 4.2.1.");
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].target().to_string(), "4.2.1");
        assert_eq!(references[0].raw(), "Clause 4.2.1");
    }

    #[test]
    fn expands_a_list_of_targets() {
        assert_eq!(targets("subject to Clauses 7, 8 and 9"), ["7", "8", "9"]);
        assert_eq!(targets("see Sections 2 to 4"), ["2", "4"]);
    }

    #[test]
    fn a_list_reference_quotes_the_whole_printed_span() {
        // Every reference minted from "Clauses 7 and 8" must quote text that
        // really occurs in the clause, so that it can be cited.
        let references = scan("subject to Clauses 7 and 8 below");
        assert_eq!(references.len(), 2);
        for reference in &references {
            assert_eq!(reference.raw(), "Clauses 7 and 8");
        }
    }

    #[test]
    fn a_keyword_without_a_number_is_not_a_reference() {
        assert!(scan("Nothing in this Section shall apply.").is_empty());
        assert!(scan("The clause is void.").is_empty());
    }

    #[test]
    fn a_keyword_inside_a_longer_word_is_not_a_reference() {
        assert!(scan("The sectional door 4 is not a reference.").is_empty());
    }

    #[test]
    fn distinguishes_the_referencing_keyword() {
        assert_eq!(scan("see Sub-clause 3.1")[0].kind(), ReferenceKind::SubClause);
        assert_eq!(scan("see Endorsement 2")[0].kind(), ReferenceKind::Endorsement);
        assert_eq!(scan("see Item 5")[0].kind(), ReferenceKind::Item);
    }

    #[test]
    fn scanning_is_case_insensitive_but_preserves_the_printed_form() {
        let references = scan("SUBJECT TO CLAUSE 7");
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].raw(), "CLAUSE 7");
    }
}
