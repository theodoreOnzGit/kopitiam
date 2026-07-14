//! Anomalies — the things this crate could not determine, said out loud.
//!
//! Rule 3 of this engine: **never silently normalise away ambiguity or
//! failure.** If a clause is ambiguous, contradictory, or unparseable, it is
//! surfaced *as such*, with the original text attached.
//!
//! "I could not determine this — here is the clause, read it" is a correct and
//! valuable answer about a legal contract. A clean-looking wrong answer is a
//! dangerous one. Every variant here exists because there is a specific way an
//! insurance-document extractor can be confidently wrong, and reporting the
//! doubt is the only defence.
//!
//! Every anomaly carries a [`Provenance`] — so the report is never "something
//! went wrong somewhere", but "on page 7, clause 4.2 says this, and here is
//! what I could not work out about it".
//!
//! Retrieve them with [`crate::PolicyDocument::anomalies`]. A consumer that
//! presents extracted policy terms to a human **should show these too**. They
//! are not warnings to be filtered out of a log; they are part of the answer.

use serde::{Deserialize, Serialize};

use crate::clause::ClauseId;
use crate::endorsement::EndorsementId;
use crate::provenance::Provenance;

/// Something this crate could not determine, or something the document itself
/// gets wrong.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Anomaly {
    /// A clause refers to another clause **that is not in this document**.
    ///
    /// Usually a renumbering the drafter never propagated. It means the
    /// referring clause cannot be fully read — and a consumer that quietly
    /// dropped the reference would let a reader believe otherwise.
    DanglingCrossReference {
        /// The clause that makes the reference.
        from: ClauseId,
        /// The reference exactly as printed (`"subject to Clause 12"`).
        raw: String,
        /// The clause it points at, which does not exist here.
        target: ClauseId,
        /// Where the reference is printed.
        provenance: Provenance,
    },

    /// The policy defines the same term **twice, inconsistently**.
    ///
    /// There is no safe way to pick a winner: the definitions section is what
    /// gives the policy's words their meaning, and a contradiction in it
    /// propagates to every clause using the term.
    ConflictingDefinition {
        /// The term defined more than once.
        term: String,
        /// Every definition of it, with its citation.
        definitions: Vec<Provenance>,
    },

    /// A schedule value looked like a number and could not be typed. The raw
    /// text is preserved: **do not** substitute a zero.
    UnparseableScheduleValue {
        /// The row's label.
        label: String,
        /// The value exactly as printed.
        raw: String,
        /// Why it could not be typed.
        reason: String,
        /// Where it is printed.
        provenance: Provenance,
    },

    /// An endorsement modifies a clause **that is not in this document**.
    ///
    /// The most dangerous variant in this enum. It means either the base
    /// wording is missing from the pack we were given (so the reader is about
    /// to read a contract with a hole in it), or the endorsement names a
    /// clause the wording does not have. Either way, something is being read
    /// wrongly, and the reader must know.
    EndorsementTargetNotFound {
        /// The endorsement.
        endorsement: EndorsementId,
        /// The clause it claims to modify.
        target: ClauseId,
        /// Where the endorsement is printed.
        provenance: Provenance,
    },

    /// An endorsement is present but we could not determine what it changes.
    /// It is **not** ignored: the reader is told there is an endorsement in the
    /// pack they must read for themselves.
    UnspecifiedEndorsement {
        /// The endorsement.
        endorsement: EndorsementId,
        /// Where it is printed.
        provenance: Provenance,
    },

    /// A clause's **structure and its language disagree** — e.g. a clause
    /// printed under `Exclusions` whose sentence reads as a grant of cover.
    ///
    /// Structure wins for classification (see [`crate::Exclusion`]), but the
    /// disagreement is reported, because one of the two signals is wrong and
    /// we do not know which.
    ConflictingClauseSignals {
        /// The clause.
        clause: ClauseId,
        /// What the disagreement is.
        reason: String,
        /// Where the clause is printed, with its text.
        provenance: Provenance,
    },

    /// The document prints two clauses with the **same identifier**. A
    /// cross-reference to that number is therefore ambiguous.
    DuplicateClauseId {
        /// The repeated identifier.
        id: ClauseId,
        /// Every clause printed under it.
        occurrences: Vec<Provenance>,
    },

    /// A schedule labels two different rows identically.
    DuplicateScheduleLabel {
        /// The repeated label.
        label: String,
        /// Every row printed under it.
        occurrences: Vec<Provenance>,
    },

    /// A document classified as a policy wording has **no definitions
    /// section**.
    ///
    /// Every policy wording has one — it is where the contract's private
    /// vocabulary lives. Its absence almost always means our structural
    /// detection missed it, in which case **every defined term in the document
    /// will silently resolve to its plain English meaning**, which is precisely
    /// the failure the definitions machinery exists to prevent. Loud by design.
    NoDefinitionsSection {
        /// What we classified the document as.
        classified_as: String,
    },
}

impl Anomaly {
    /// A one-line, human-readable statement of the doubt. Always names the
    /// location, because the only useful response to most of these is "go and
    /// read the words".
    pub fn summary(&self) -> String {
        match self {
            Self::DanglingCrossReference {
                from, raw, target, provenance,
            } => format!(
                "clause {from} refers to {raw:?}, but clause {target} is not in this document \
                 ({provenance})"
            ),
            Self::ConflictingDefinition { term, definitions } => format!(
                "{term:?} is defined {} times, inconsistently ({})",
                definitions.len(),
                definitions
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            Self::UnparseableScheduleValue { label, raw, reason, provenance } => format!(
                "schedule value for {label:?} ({raw:?}) could not be typed: {reason} ({provenance})"
            ),
            Self::EndorsementTargetNotFound { endorsement, target, provenance } => format!(
                "{endorsement} modifies clause {target}, which is not in this document — \
                 the base wording may be missing from the pack ({provenance})"
            ),
            Self::UnspecifiedEndorsement { endorsement, provenance } => format!(
                "{endorsement} is present but its effect could not be determined; \
                 it must be read ({provenance})"
            ),
            Self::ConflictingClauseSignals { clause, reason, provenance } => {
                format!("clause {clause}: {reason} ({provenance})")
            }
            Self::DuplicateClauseId { id, occurrences } => format!(
                "clause {id} is printed {} times; references to it are ambiguous",
                occurrences.len()
            ),
            Self::DuplicateScheduleLabel { label, occurrences } => format!(
                "schedule label {label:?} appears {} times",
                occurrences.len()
            ),
            Self::NoDefinitionsSection { classified_as } => format!(
                "no definitions section found in a document classified as {classified_as}; \
                 every defined term will fall back to its plain English meaning, which is \
                 how a policy gets read backwards"
            ),
        }
    }

    /// The verbatim source text this anomaly is about, when it is about a
    /// specific passage. `None` for document-wide anomalies.
    pub fn verbatim(&self) -> Option<&str> {
        match self {
            Self::DanglingCrossReference { provenance, .. }
            | Self::UnparseableScheduleValue { provenance, .. }
            | Self::EndorsementTargetNotFound { provenance, .. }
            | Self::UnspecifiedEndorsement { provenance, .. }
            | Self::ConflictingClauseSignals { provenance, .. } => {
                Some(provenance.verbatim().as_str())
            }
            Self::ConflictingDefinition { definitions, .. } => {
                definitions.first().map(|p| p.verbatim().as_str())
            }
            Self::DuplicateClauseId { occurrences, .. }
            | Self::DuplicateScheduleLabel { occurrences, .. } => {
                occurrences.first().map(|p| p.verbatim().as_str())
            }
            Self::NoDefinitionsSection { .. } => None,
        }
    }
}
