//! [`PolicyDocument`]: an ingested insurance document, and the query surface a
//! domain crate builds on.
//!
//! This is the aggregate: the clauses, the definitions section, the schedule,
//! the endorsements, and — as a first-class part of the answer, not an
//! afterthought — everything that could not be determined
//! ([`PolicyDocument::anomalies`]).
//!
//! # It does not adjudicate
//!
//! There is no `is_covered()`, and there never will be. This type will tell you
//! *what the document says* and *where it says it*. Whether a particular claim
//! is payable is a legal question about a contract, and answering it requires
//! facts about an event, a policy in force, and rules of construction — none of
//! which are in a PDF. See the crate docs.

use crate::anomaly::Anomaly;
use crate::classify::Classification;
use crate::clause::{Clause, ClauseId, ClauseRole};
use crate::crossref::{CrossReference, ResolvedReference};
use crate::definition::{Definitions, Resolution, TermOccurrence};
use crate::endorsement::{EffectiveClause, Endorsement, EndorsementEffect};
use crate::exclusion::{self, Exclusion};
use crate::ingest;
use crate::provenance::DocumentId;
use crate::schedule::{BenefitTable, Schedule};

/// An ingested insurance document.
#[derive(Debug, Clone)]
pub struct PolicyDocument {
    id: DocumentId,
    classification: Classification,
    clauses: Vec<Clause>,
    definitions: Definitions,
    schedule: Schedule,
    benefit_tables: Vec<BenefitTable>,
    endorsements: Vec<Endorsement>,
    anomalies: Vec<Anomaly>,
    pages: usize,
}

/// Everything [`crate::ingest`] extracts, before the audit runs over it.
///
/// A parts struct rather than an eight-argument constructor: eight positional
/// arguments of which three are `Vec`s is a call site nobody can read, and
/// swapping two of them would compile.
pub(crate) struct Parts {
    pub id: DocumentId,
    pub classification: Classification,
    pub clauses: Vec<Clause>,
    pub definitions: Definitions,
    pub schedule: Schedule,
    pub benefit_tables: Vec<BenefitTable>,
    pub endorsements: Vec<Endorsement>,
    pub pages: usize,
}

impl PolicyDocument {
    /// Assembles a document and runs the audit that produces its anomalies.
    pub(crate) fn assemble(parts: Parts) -> Self {
        let mut policy = Self {
            id: parts.id,
            classification: parts.classification,
            clauses: parts.clauses,
            definitions: parts.definitions,
            schedule: parts.schedule,
            benefit_tables: parts.benefit_tables,
            endorsements: parts.endorsements,
            anomalies: Vec::new(),
            pages: parts.pages,
        };
        policy.anomalies = ingest::audit(&policy);
        policy
    }

    /// The source document.
    pub fn id(&self) -> &DocumentId {
        &self.id
    }

    /// What kind of document this is, and the evidence for it.
    pub fn classification(&self) -> &Classification {
        &self.classification
    }

    /// How many pages it has.
    pub fn pages(&self) -> usize {
        self.pages
    }

    /// Every clause, in document order.
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// The clause **as printed in the base wording**, ignoring any endorsement
    /// that changed it.
    ///
    /// The name is a warning. If an endorsement has replaced or deleted this
    /// clause, **this is not the contract** — see
    /// [`PolicyDocument::effective_clause`], which is what almost every caller
    /// actually wants. This accessor exists for the callers that genuinely
    /// need the superseded text (showing a diff, auditing an amendment), and
    /// it is named so that nobody reaches for it by accident.
    pub fn base_clause(&self, id: &ClauseId) -> Option<&Clause> {
        self.clauses.iter().find(|clause| clause.id() == id)
    }

    /// The clause **as the contract now stands** — base wording plus any
    /// endorsement that changed it.
    ///
    /// The return type is an [`EffectiveClause`], whose variants *are* the
    /// override status. A caller cannot obtain the effective wording of a
    /// replaced clause without being handed the [`Endorsement`] that replaced
    /// it, and cannot obtain any wording at all for a deleted one. The
    /// override is visible by construction; there is no path through this API
    /// on which it is silent.
    pub fn effective_clause(&self, id: &ClauseId) -> EffectiveClause<'_> {
        let base = self.base_clause(id);

        let modifier = self.endorsements.iter().find(|endorsement| {
            endorsement.effect().target() == Some(id)
                && !matches!(endorsement.effect(), EndorsementEffect::Adds { .. })
        });

        if let Some(endorsement) = modifier {
            let Some(base) = base else {
                // An endorsement modifying a clause we do not have. Reported as
                // `Anomaly::EndorsementTargetNotFound`; there is no wording to
                // give, and inventing one is not an option.
                return EffectiveClause::NotFound;
            };
            return match endorsement.effect() {
                EndorsementEffect::Replaces { wording, .. } => EffectiveClause::Replaced {
                    base,
                    by: endorsement,
                    wording,
                },
                EndorsementEffect::Deletes { .. } => EffectiveClause::Deleted {
                    base,
                    by: endorsement,
                },
                EndorsementEffect::AmendsUnspecified { .. } => EffectiveClause::Amended {
                    base,
                    by: endorsement,
                },
                EndorsementEffect::Adds { .. } | EndorsementEffect::Unspecified => {
                    EffectiveClause::Base(base)
                }
            };
        }

        // A clause that exists only because an endorsement added it.
        let added = self.endorsements.iter().find(|endorsement| {
            matches!(endorsement.effect(), EndorsementEffect::Adds { target: Some(target) } if target == id)
        });
        if let Some(endorsement) = added {
            return EffectiveClause::Added {
                by: endorsement,
                clause: endorsement.clause(),
            };
        }

        match base {
            Some(base) => EffectiveClause::Base(base),
            None => EffectiveClause::NotFound,
        }
    }

    /// The policy's definitions section.
    pub fn definitions(&self) -> &Definitions {
        &self.definitions
    }

    /// **What this policy says a word means** — which overrides what the word
    /// ordinarily means.
    ///
    /// The single most important query in this crate. See
    /// [`crate::definition`].
    pub fn meaning_of(&self, term: &str) -> Resolution<'_> {
        self.definitions.resolve(term)
    }

    /// Every defined term appearing in a clause, located and resolved to the
    /// policy's own meaning.
    ///
    /// A clause read without this is a clause read in plain English, and a
    /// policy's whole point is that its words do not mean what they ordinarily
    /// mean.
    pub fn defined_terms_in(&self, clause: &Clause) -> Vec<TermOccurrence<'_>> {
        self.definitions.occurrences_in(clause)
    }

    /// Everything the policy does **not** cover, including write-backs (which
    /// give cover back — see [`crate::ExclusionEffect`]).
    pub fn exclusions(&self) -> Vec<Exclusion<'_>> {
        self.clauses
            .iter()
            .filter(|clause| clause.role() == ClauseRole::Exclusion)
            .map(|clause| Exclusion::new(clause, exclusion::effect_of(clause.text())))
            .collect()
    }

    /// The clauses that grant cover.
    ///
    /// Guaranteed disjoint from [`PolicyDocument::exclusions`] — a clause has
    /// exactly one [`ClauseRole`]. An exclusion appearing on this list would be
    /// the worst bug this crate could have, and the type system rules it out.
    pub fn coverages(&self) -> Vec<&Clause> {
        self.clauses_with_role(ClauseRole::Coverage)
    }

    /// The clauses stating conditions, duties and warranties.
    pub fn conditions(&self) -> Vec<&Clause> {
        self.clauses_with_role(ClauseRole::Condition)
    }

    /// Clauses this crate could not classify. Worth showing a reader: they are
    /// part of the contract, and we are declining to say what they do.
    pub fn unclassified(&self) -> Vec<&Clause> {
        self.clauses_with_role(ClauseRole::Unclassified)
    }

    fn clauses_with_role(&self, role: ClauseRole) -> Vec<&Clause> {
        self.clauses
            .iter()
            .filter(|clause| clause.role() == role)
            .collect()
    }

    /// The policy-specific numbers.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Plan-by-plan benefit tables. See [`BenefitTable`].
    pub fn benefit_tables(&self) -> &[BenefitTable] {
        &self.benefit_tables
    }

    /// The endorsements that modify the base wording.
    pub fn endorsements(&self) -> &[Endorsement] {
        &self.endorsements
    }

    /// **Everything this crate could not determine**, with the text it could
    /// not determine it about.
    ///
    /// Not a log. Part of the answer. A consumer presenting extracted terms to
    /// a human should present these too — see [`crate::Anomaly`].
    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// Looks up the clause a cross-reference points at.
    ///
    /// A reference to a clause that is not in this document comes back as
    /// [`ResolvedReference::Dangling`], never as an empty `Option` — because
    /// "this clause refers to a clause I do not have" is something the reader
    /// needs to be told.
    pub fn resolve_reference<'a>(&'a self, reference: &'a CrossReference) -> ResolvedReference<'a> {
        match self.base_clause(reference.target()) {
            Some(target) => ResolvedReference::Resolved { reference, target },
            None => ResolvedReference::Dangling { reference },
        }
    }

    /// Every cross-reference a clause makes, resolved.
    pub fn references_from<'a>(&'a self, clause: &'a Clause) -> Vec<ResolvedReference<'a>> {
        clause
            .cross_references()
            .iter()
            .map(|reference| self.resolve_reference(reference))
            .collect()
    }

    /// Binds an endorsement document to this base wording, as the paper policy
    /// pack does.
    ///
    /// Insurance is routinely delivered as several PDFs: a wording, a schedule,
    /// and one endorsement per amendment. Read separately, the wording looks
    /// complete and is wrong. This joins them, and re-runs the audit — so an
    /// endorsement naming a clause the wording does not contain shows up as
    /// [`Anomaly::EndorsementTargetNotFound`], which is exactly the situation
    /// a reader must not be allowed to miss.
    ///
    /// The other document's clauses are appended too, so its endorsement text
    /// remains citable.
    #[must_use]
    pub fn absorb(mut self, other: PolicyDocument) -> Self {
        self.clauses.extend(other.clauses);
        self.endorsements.extend(other.endorsements);
        for entry in other.schedule.entries() {
            self.schedule.push(entry.clone());
        }
        self.benefit_tables.extend(other.benefit_tables);
        self.pages += other.pages;
        // Definitions are *not* merged: an endorsement that redefines a term
        // must show up as a conflict, and re-extracting over the combined
        // clause list is what surfaces it.
        self.definitions = Definitions::extract(&self.clauses).unwrap_or_default();
        self.anomalies = ingest::audit(&self);
        self
    }
}
