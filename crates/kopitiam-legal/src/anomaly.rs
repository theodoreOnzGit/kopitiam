//! Anomalies: the things this crate found and **refused to guess about**.
//!
//! # "I could not determine this — here is the provision, read it" is a
//! # correct answer
//!
//! Legal documents are ambiguous, contradictory, and full of references to
//! things that are not where they say they are. That is not a defect in our
//! parser; it is a property of the material. Drafters make mistakes,
//! consolidations go stale, a contract's Schedule 3 is genuinely missing
//! from the PDF someone sent you.
//!
//! There are two possible responses to that, and only one of them is safe:
//!
//! * **Resolve it.** Pick the most likely reading, return a clean answer.
//!   The output looks authoritative. Sometimes it is wrong, and nothing in
//!   the output says which times.
//! * **Surface it.** Return the provision, its verbatim text, and a note
//!   saying precisely what could not be determined and why. The reader — who
//!   is the one qualified to judge — resolves it.
//!
//! This crate always does the second. An [`Anomaly`] is not an error and does
//! not abort ingestion: the material still comes through, fully sourced, with
//! the anomaly attached. Nothing is ever silently dropped and nothing is ever
//! silently defaulted.
//!
//! Every anomaly carries a [`Provenance`], so "there is a problem here" is
//! itself a citation the reader can follow.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Provenance, ProvisionId};

/// What kind of thing we found and would not guess about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyKind {
    /// A provision label we could not parse into a hierarchy. The provision
    /// is still emitted, with an [`crate::ProvisionComponent::Unrecognized`]
    /// component carrying the original label.
    UnparseableNumbering { label: String },

    /// A cross-reference pointing at a provision this instrument does not
    /// contain — "subject to section 7" where there is no section 7.
    ///
    /// **This is reported, never dropped.** A dangling reference is a real
    /// and important signal: it usually means the reader is holding an
    /// incomplete document, or that the reference is to *another*
    /// instrument, or that the drafter erred. All three matter, and all
    /// three are invisible if the parser quietly discards references it
    /// cannot resolve.
    DanglingCrossReference {
        raw: String,
        /// The target we parsed out of the text, if we got that far.
        target: Option<ProvisionId>,
    },

    /// Two different definitions of the same term at the same scope. Which
    /// one governs is a question of legal construction, not parsing, so we
    /// return both and decline to choose.
    ConflictingDefinition {
        term: String,
        /// Citations of the competing definitions.
        competing: Vec<String>,
    },

    /// The same provision id appeared twice in one instrument.
    DuplicateProvision { id: ProvisionId },

    /// Two versions of one provision claim to be in force on the same date.
    /// A provision cannot have said two different things on one day, so this
    /// is a contradiction in the source or in our ingestion of it.
    OverlappingValidity {
        id: ProvisionId,
        windows: Vec<String>,
    },

    /// An amending instrument gave a textual edit instruction ("delete 'may'
    /// and substitute 'must'") which we recorded but did **not** apply. See
    /// [`crate::AmendmentOperation::TextualInstructionNotApplied`].
    AmendmentNotApplied { instruction: String },

    /// Source text that reached ingestion but could not be attributed to any
    /// provision. Emitted so that "we lost some of the document" can never
    /// happen silently.
    UnattributedText,

    /// A catch-all for a source that is internally contradictory in a way we
    /// can describe but not classify.
    Ambiguous { detail: String },
}

impl fmt::Display for AnomalyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnparseableNumbering { label } => {
                write!(f, "could not parse the provision label {label:?}")
            }
            Self::DanglingCrossReference { raw, target } => match target {
                Some(t) => write!(
                    f,
                    "cross-reference {raw:?} points at {t}, which this instrument does not contain"
                ),
                None => write!(f, "could not resolve the cross-reference {raw:?}"),
            },
            Self::ConflictingDefinition { term, competing } => write!(
                f,
                "the term {term:?} is defined more than once at the same scope ({}); \
                 which governs is a question of construction, not parsing",
                competing.join("; ")
            ),
            Self::DuplicateProvision { id } => write!(f, "{id} appears more than once"),
            Self::OverlappingValidity { id, windows } => write!(
                f,
                "{id} has overlapping in-force windows ({}); it cannot have said two \
                 different things on the same day",
                windows.join("; ")
            ),
            Self::AmendmentNotApplied { instruction } => write!(
                f,
                "amendment instruction recorded but NOT applied: {instruction:?}; \
                 the consolidated text is not derivable by this tool"
            ),
            Self::UnattributedText => {
                f.write_str("source text could not be attributed to any provision")
            }
            Self::Ambiguous { detail } => write!(f, "ambiguous: {detail}"),
        }
    }
}

/// A finding, with a citation the reader can follow.
///
/// Carries a [`Provenance`] like every other extracted item: an anomaly is a
/// statement about a specific place in a specific document, and it is no more
/// entitled to be un-sourced than a provision is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anomaly {
    kind: AnomalyKind,
    provenance: Provenance,
}

impl Anomaly {
    pub fn new(kind: AnomalyKind, provenance: Provenance) -> Self {
        Self { kind, provenance }
    }

    pub fn kind(&self) -> &AnomalyKind {
        &self.kind
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

impl fmt::Display for Anomaly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n  at {}\n  source text: {:?}",
            self.kind,
            self.provenance.citation(),
            self.provenance.verbatim()
        )
    }
}
