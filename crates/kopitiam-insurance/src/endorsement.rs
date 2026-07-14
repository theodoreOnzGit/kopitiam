//! Endorsements — the modifications that **override** the base wording.
//!
//! An endorsement (a.k.a. a rider, a memorandum, an amendment) is a separate
//! sheet of paper that changes what the policy says. It is issued later, it is
//! often shorter than a page, and it silently rewrites the contract:
//!
//! > Endorsement No. 1. It is hereby agreed that Clause 4.2 is deleted and
//! > replaced with the following: ...
//!
//! **A reader who misses the endorsement gets the wrong answer**, with total
//! confidence, from a clause that is right there in front of them in the
//! wording. This is one of the most common and most consequential failure modes
//! in reading an insurance policy, and it is the one an automated extractor is
//! *most* likely to reproduce — because the base wording parses beautifully and
//! nothing about clause 4.2, read on its own, hints that it has been revoked.
//!
//! # How the override is made impossible to miss
//!
//! There is no API here that hands back "the wording of clause 4.2". There is
//! [`crate::PolicyDocument::effective_clause`], which returns an
//! [`EffectiveClause`] — an enum whose variants *are* the override status. A
//! caller cannot read the effective text of a replaced clause without pattern-
//! matching on [`EffectiveClause::Replaced`] and thereby seeing the
//! endorsement that replaced it. The override is structurally visible, not a
//! flag someone has to remember to check.
//!
//! The base wording is still reachable — via
//! [`crate::PolicyDocument::base_clause`], whose name says what it is. That
//! honesty is deliberate: a caller who wants the *superseded* text (to show a
//! diff, say) should have it; a caller who wants *the contract* should not
//! reach for it by accident.
//!
//! # What is deliberately not attempted
//!
//! Where an endorsement amends a clause in prose ("Clause 7 is amended by
//! adding the words 'and their spouse' after 'Insured Person'"), this crate
//! does **not** try to mechanically apply the edit and synthesise a new
//! clause. Producing a clause text that appears nowhere in either document,
//! and presenting it as what the contract now says, would be manufacturing
//! legal wording. Instead the result is
//! [`EndorsementEffect::AmendsUnspecified`] / [`EffectiveClause::Amended`]:
//! *this clause has been changed, here is the endorsement, read them together*.
//! An honest "read both" beats a synthesised sentence that no document
//! contains.

use serde::{Deserialize, Serialize};

use crate::clause::{Clause, ClauseId};
use crate::crossref;
use crate::provenance::{Provenance, ProvenanceError, SourceText};

/// Phrases by which an endorsement announces what it does to a base clause.
/// Ordered so that the more specific ("deleted and replaced") is tested before
/// the less specific ("deleted") — reading a replacement as a plain deletion
/// would remove cover the endorsement actually re-granted.
const REPLACE_PHRASES: &[&str] = &[
    "is deleted and replaced with the following",
    "is deleted and replaced by the following",
    "is deleted and replaced with",
    "is deleted and replaced by",
    "is replaced with the following",
    "is replaced by the following",
    "is replaced with",
    "is replaced by",
    "shall be deleted and replaced with",
    "shall be replaced with",
    "is hereby deleted and substituted with",
];

const DELETE_PHRASES: &[&str] = &[
    "is deleted in its entirety",
    "is hereby deleted",
    "is deleted",
    "shall be deleted",
    "is cancelled",
    "no longer applies",
    "shall not apply",
];

const AMEND_PHRASES: &[&str] = &[
    "is amended",
    "is hereby amended",
    "shall be amended",
    "is modified",
    "is extended",
    "is varied",
];

const ADD_PHRASES: &[&str] = &[
    "the following clause is added",
    "the following exclusion is added",
    "the following is added",
    "is added to",
    "the following clause is inserted",
];

/// Headings that mark a document (or a section of one) as an endorsement.
pub(crate) const ENDORSEMENT_HEADINGS: &[&str] = &[
    "endorsement",
    "endorsements",
    "memorandum",
    "amendment",
    "policy amendment",
    "rider",
];

/// Identifies an endorsement, as printed (`"Endorsement No. 1"`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EndorsementId(String);

impl EndorsementId {
    /// Names an endorsement.
    ///
    /// # Errors
    ///
    /// [`ProvenanceError::EmptyClauseId`] if the label is blank.
    pub fn new(label: impl Into<String>) -> Result<Self, ProvenanceError> {
        let label = label.into();
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return Err(ProvenanceError::EmptyClauseId);
        }
        Ok(Self(trimmed.to_string()))
    }

    /// The endorsement's label, as printed.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EndorsementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What an endorsement does to the base wording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndorsementEffect {
    /// The base clause's wording is **replaced**. Reading the base wording now
    /// gives the wrong answer.
    Replaces {
        /// The clause being replaced.
        target: ClauseId,
        /// The new wording, verbatim from the endorsement.
        wording: SourceText,
    },

    /// The base clause is **deleted**. There is no wording; the clause is gone.
    Deletes {
        /// The clause being deleted.
        target: ClauseId,
    },

    /// The base clause is changed, but in prose we will not mechanically
    /// apply. Base + endorsement must be **read together**. See the module
    /// docs for why no synthesised text is produced.
    AmendsUnspecified {
        /// The clause being amended.
        target: ClauseId,
    },

    /// A wholly new clause is introduced by the endorsement, with no base
    /// wording behind it.
    Adds {
        /// The new clause's identifier, if the endorsement printed one.
        target: Option<ClauseId>,
    },

    /// The endorsement's own text gave no recoverable mechanics — we could not
    /// tell what it changes. **Not** ignored: it is surfaced, so a reader knows
    /// there is an endorsement in the pack they must read for themselves.
    Unspecified,
}

impl EndorsementEffect {
    /// The base clause this endorsement touches, if it names one.
    pub fn target(&self) -> Option<&ClauseId> {
        match self {
            Self::Replaces { target, .. }
            | Self::Deletes { target }
            | Self::AmendsUnspecified { target } => Some(target),
            Self::Adds { target } => target.as_ref(),
            Self::Unspecified => None,
        }
    }
}

/// A modification to the base policy wording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endorsement {
    id: EndorsementId,
    /// The endorsement's effective date **exactly as printed**. Not parsed
    /// into a calendar type: date formats in insurance documents are wildly
    /// inconsistent (`01/02/2026` is two different days depending on which
    /// side of the Atlantic drafted it), and mis-parsing the day cover starts
    /// is not a bug worth risking for the convenience of a typed date.
    effective_date: Option<String>,
    clause: Clause,
    effect: EndorsementEffect,
}

impl Endorsement {
    pub(crate) fn new(
        id: EndorsementId,
        effective_date: Option<String>,
        clause: Clause,
        effect: EndorsementEffect,
    ) -> Self {
        Self {
            id,
            effective_date,
            clause,
            effect,
        }
    }

    /// The endorsement's identifier, as printed.
    pub fn id(&self) -> &EndorsementId {
        &self.id
    }

    /// Its effective date, exactly as printed. See [`Endorsement::effective_date`]'s
    /// field docs for why this is a string.
    pub fn effective_date(&self) -> Option<&str> {
        self.effective_date.as_deref()
    }

    /// The endorsement's own clause: its verbatim text, page and citation.
    pub fn clause(&self) -> &Clause {
        &self.clause
    }

    /// What it does to the base wording.
    pub fn effect(&self) -> &EndorsementEffect {
        &self.effect
    }

    /// Where the endorsement is printed.
    pub fn provenance(&self) -> Provenance {
        self.clause.provenance()
    }
}

/// The wording of a clause **as the contract now stands** — base wording plus
/// any endorsement that changed it.
///
/// The variants are the override status. There is no way to obtain the
/// effective text of a replaced clause without matching on
/// [`EffectiveClause::Replaced`] and being handed the [`Endorsement`] that
/// replaced it. That is the whole design: the override cannot be silent,
/// because the type will not let it be.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectiveClause<'a> {
    /// No endorsement touches this clause. The base wording is the contract.
    Base(&'a Clause),

    /// An endorsement **replaced** this clause. The base wording is superseded;
    /// reading it would give the wrong answer.
    Replaced {
        /// The superseded base wording (useful for a diff; not the contract).
        base: &'a Clause,
        /// The endorsement that replaced it.
        by: &'a Endorsement,
        /// The wording that now applies, verbatim from the endorsement.
        wording: &'a SourceText,
    },

    /// An endorsement **deleted** this clause. It has no wording at all — a
    /// caller that falls back to the base text here is reading a clause the
    /// contract no longer contains.
    Deleted {
        /// The deleted base wording.
        base: &'a Clause,
        /// The endorsement that deleted it.
        by: &'a Endorsement,
    },

    /// An endorsement **amended** this clause in prose we did not mechanically
    /// apply. Neither text alone is the contract: they must be read together.
    Amended {
        /// The base wording.
        base: &'a Clause,
        /// The endorsement that amends it.
        by: &'a Endorsement,
    },

    /// The clause exists only because an endorsement added it.
    Added {
        /// The endorsement that introduced it.
        by: &'a Endorsement,
        /// The added clause.
        clause: &'a Clause,
    },

    /// No such clause in this document — neither in the wording nor in any
    /// endorsement.
    NotFound,
}

impl<'a> EffectiveClause<'a> {
    /// Whether an endorsement has changed this clause. If this is `true`, the
    /// base wording is **not** the contract.
    pub fn is_overridden(&self) -> bool {
        matches!(
            self,
            Self::Replaced { .. } | Self::Deleted { .. } | Self::Amended { .. } | Self::Added { .. }
        )
    }

    /// The endorsement responsible for the override, if any.
    pub fn endorsement(&self) -> Option<&'a Endorsement> {
        match self {
            Self::Replaced { by, .. }
            | Self::Deleted { by, .. }
            | Self::Amended { by, .. }
            | Self::Added { by, .. } => Some(by),
            Self::Base(_) | Self::NotFound => None,
        }
    }

    /// The wording that **now applies**, if a single text can honestly be said
    /// to apply.
    ///
    /// `None` for:
    ///
    /// * [`EffectiveClause::Deleted`] — there is no wording; the clause is gone.
    ///   Falling back to the base text here would resurrect a deleted clause.
    /// * [`EffectiveClause::Amended`] — no single text is the contract; both
    ///   must be read. See the module docs.
    /// * [`EffectiveClause::NotFound`].
    ///
    /// A caller that wants a string come what may is being told, by this
    /// `None`, that there isn't one.
    pub fn wording(&self) -> Option<&'a str> {
        match self {
            Self::Base(clause) | Self::Added { clause, .. } => Some(clause.text()),
            Self::Replaced { wording, .. } => Some(wording.as_str()),
            Self::Deleted { .. } | Self::Amended { .. } | Self::NotFound => None,
        }
    }
}

/// Reads an endorsement's own text to work out what it does to the base
/// wording.
///
/// Deliberately conservative. Where the mechanics are not clear, the result is
/// [`EndorsementEffect::AmendsUnspecified`] or
/// [`EndorsementEffect::Unspecified`] — never a guess, because a guessed
/// override rewrites a contract.
pub(crate) fn parse_effect(text: &str) -> EndorsementEffect {
    let lower = text.to_ascii_lowercase();
    let target = first_reference(text);

    // Order matters: "deleted and replaced" must not be read as "deleted".
    if let Some(at) = find_any(&lower, REPLACE_PHRASES)
        && let Some(target) = target
    {
        let wording = text[at.end..]
            .trim_start()
            .trim_start_matches([':', '-', '\u{2013}', '\u{2014}'])
            .trim();
        if let Ok(wording) = SourceText::new(wording) {
            return EndorsementEffect::Replaces { target, wording };
        }
        // "deleted and replaced with the following:" with the replacement text
        // somewhere we did not recover. We know the clause changed but not to
        // what: say exactly that.
        return EndorsementEffect::AmendsUnspecified { target };
    }

    if find_any(&lower, DELETE_PHRASES).is_some()
        && let Some(target) = target
    {
        return EndorsementEffect::Deletes { target };
    }

    if find_any(&lower, ADD_PHRASES).is_some() {
        return EndorsementEffect::Adds { target };
    }

    if find_any(&lower, AMEND_PHRASES).is_some()
        && let Some(target) = target
    {
        return EndorsementEffect::AmendsUnspecified { target };
    }

    EndorsementEffect::Unspecified
}

/// The first clause the endorsement's text points at — the clause it is about.
fn first_reference(text: &str) -> Option<ClauseId> {
    crossref::scan(text)
        .first()
        .map(|reference| reference.target().clone())
}

struct Match {
    end: usize,
}

fn find_any(haystack: &str, needles: &[&str]) -> Option<Match> {
    needles.iter().find_map(|needle| {
        haystack.find(needle).map(|at| Match {
            end: at + needle.len(),
        })
    })
}

/// Whether a heading marks an endorsement.
pub(crate) fn is_endorsement_heading(heading: &str) -> bool {
    let lower = heading.to_lowercase();
    ENDORSEMENT_HEADINGS
        .iter()
        .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All endorsement wording here is invented for the test.
    #[test]
    fn reads_a_replacement_and_captures_the_new_wording() {
        let effect = parse_effect(
            "It is hereby agreed that Clause 4.2 is deleted and replaced with the following: \
             We will not pay for loss caused by war, save where the Insured is a passenger \
             on a commercial airline.",
        );
        let EndorsementEffect::Replaces { target, wording } = effect else {
            panic!("expected Replaces, got {effect:?}");
        };
        assert_eq!(target.to_string(), "4.2");
        assert!(wording.as_str().starts_with("We will not pay"));
        assert!(wording.as_str().contains("commercial airline"));
    }

    #[test]
    fn a_replacement_is_not_read_as_a_bare_deletion() {
        // "deleted and replaced" contains "deleted". Matching the shorter
        // phrase first would strip cover the endorsement actually re-granted.
        let effect = parse_effect("Clause 7 is deleted and replaced by: Cover is worldwide.");
        assert!(matches!(effect, EndorsementEffect::Replaces { .. }), "{effect:?}");
    }

    #[test]
    fn reads_a_deletion() {
        let effect = parse_effect("Clause 5.3 is hereby deleted in its entirety.");
        let EndorsementEffect::Deletes { target } = effect else {
            panic!("expected Deletes, got {effect:?}");
        };
        assert_eq!(target.to_string(), "5.3");
    }

    #[test]
    fn a_prose_amendment_is_not_mechanically_applied() {
        // Synthesising "the clause as amended" would be manufacturing legal
        // wording that appears in no document. Say "read both" instead.
        let effect = parse_effect(
            "Clause 7 is amended by adding the words 'and their Spouse' after 'Insured Person'.",
        );
        let EndorsementEffect::AmendsUnspecified { target } = effect else {
            panic!("expected AmendsUnspecified, got {effect:?}");
        };
        assert_eq!(target.to_string(), "7");
    }

    #[test]
    fn an_endorsement_with_no_recoverable_mechanics_is_still_surfaced() {
        assert_eq!(
            parse_effect("This endorsement forms part of the Policy."),
            EndorsementEffect::Unspecified
        );
    }

    #[test]
    fn effective_clause_wording_is_none_for_a_deleted_clause() {
        // A caller falling back to the base text of a deleted clause would be
        // reading a clause the contract no longer contains. `None` prevents it.
        use crate::provenance::{DocumentId, PageNumber, SectionPath};
        use crate::{ClauseLine, ClauseRole};

        let base = Clause::new(
            DocumentId::new("wording.pdf").unwrap(),
            ClauseId::printed("5.3").unwrap(),
            None,
            SectionPath::default(),
            ClauseRole::Exclusion,
            vec![ClauseLine::new(PageNumber::new(5).unwrap(), "Flood is excluded.").unwrap()],
        )
        .unwrap();

        let endorsement_clause = Clause::new(
            DocumentId::new("endorsement.pdf").unwrap(),
            ClauseId::printed("E1").unwrap(),
            None,
            SectionPath::new(["Endorsement No. 1"]),
            ClauseRole::Endorsement,
            vec![ClauseLine::new(PageNumber::new(1).unwrap(), "Clause 5.3 is deleted.").unwrap()],
        )
        .unwrap();

        let endorsement = Endorsement::new(
            EndorsementId::new("Endorsement No. 1").unwrap(),
            None,
            endorsement_clause,
            EndorsementEffect::Deletes {
                target: ClauseId::printed("5.3").unwrap(),
            },
        );

        let effective = EffectiveClause::Deleted {
            base: &base,
            by: &endorsement,
        };
        assert!(effective.is_overridden());
        assert_eq!(effective.wording(), None);
        assert_eq!(
            effective.endorsement().unwrap().id().as_str(),
            "Endorsement No. 1"
        );
    }
}
