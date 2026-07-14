//! [`Instrument`]: a whole legal document, its provisions, its dictionary,
//! its reference graph, and everything about it we refused to guess.
//!
//! # Document kinds are not interchangeable
//!
//! A statute, a contract and a judgment are all "legal documents" the way a
//! poem and a spreadsheet are both "files". They differ in ways that a single
//! flat model would erase:
//!
//! | | Numbering | Temporal anchor | Who it binds |
//! |---|---|---|---|
//! | **Act** | `Part II, s 12(3)(a)` | commencement, amendment, repeal | everyone in the jurisdiction |
//! | **Subsidiary legislation** | `reg 4(2)` | commencement; *dies with its parent Act* | everyone, but only within the parent's power |
//! | **Contract / lease / deed** | `cl 1.2.3` | effective date; endorsements | the parties only |
//! | **Judgment** | `[47]` | date handed down; can be *overruled* | via precedent |
//!
//! The temporal model differs most sharply, and it is the reason
//! [`InstrumentKind`] exists rather than a `kind: String`. A statute is
//! *amended* (the text changes, and the old text was the law until it did). A
//! judgment is never amended — it is *overruled*, which does not change a word
//! of it but destroys its authority prospectively. Modelling both as "versions
//! of a document" would be wrong about both.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    reference::{extract_references, ReferenceResolution, ReferenceTarget},
    Anomaly, AnomalyKind, AsAtDate, AsAtResult, Date, Definition, DefinitionScope, Dictionary,
    DocumentId, DocumentVersion, Judgment, LegalError, NumberingScheme, Provision, ProvisionHistory,
    ProvisionId, Resolution,
};

/// What kind of legal instrument this is. See the module docs for why this is
/// an enum and not a label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentKind {
    /// An Act of a legislature.
    Act {
        short_title: String,
        /// e.g. `"Act 12 of 1994"`.
        act_number: Option<String>,
    },
    /// Regulations, rules, orders — legislation made *under* an Act, whose
    /// validity depends on the parent Act's continuing to confer the power.
    SubsidiaryLegislation {
        title: String,
        /// The Act under which this was made. Recorded because subsidiary
        /// legislation is void to the extent it exceeds its parent's power,
        /// and because it falls with the parent if the parent is repealed.
        made_under: String,
    },
    /// A contract between parties.
    Contract {
        title: String,
        parties: Vec<String>,
        effective_date: Option<Date>,
    },
    /// A lease or deed. Structurally a contract, but kept distinct because
    /// leases carry habendum/reddendum/covenant structure and a term of
    /// years that contracts generally do not.
    Lease {
        title: String,
        parties: Vec<String>,
        effective_date: Option<Date>,
        /// e.g. `"99 years from 1 January 2000"`, recorded verbatim rather
        /// than parsed into a duration — lease terms are written in prose and
        /// computing an expiry date from them is construction.
        term: Option<String>,
    },
    /// A judgment or other reasoned decision. See [`Judgment`].
    Judgment(Judgment),
}

impl InstrumentKind {
    /// The numbering convention this kind of document uses. Determined by the
    /// document kind rather than sniffed per-line, because sniffing is how you
    /// end up citing "clause 12(3)" of a judgment.
    pub fn numbering_scheme(&self) -> NumberingScheme {
        match self {
            Self::Act { .. } | Self::SubsidiaryLegislation { .. } => NumberingScheme::Statutory,
            Self::Contract { .. } | Self::Lease { .. } => NumberingScheme::DecimalClause,
            Self::Judgment(_) => NumberingScheme::JudgmentParagraph,
        }
    }

    pub fn title(&self) -> &str {
        match self {
            Self::Act { short_title, .. } => short_title,
            Self::SubsidiaryLegislation { title, .. }
            | Self::Contract { title, .. }
            | Self::Lease { title, .. } => title,
            Self::Judgment(j) => j.case_name(),
        }
    }
}

/// A whole legal document, structured.
///
/// # What this is not
///
/// It is not an opinion about what the document means, what it requires of
/// anyone, or what follows from it. It is a **map of what the document says
/// and where**: provisions with their verbatim text and page numbers, the
/// terms it defines for itself, the graph of its internal references, the
/// amendment history of each provision, and an explicit list of everything
/// that could not be determined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instrument {
    id: DocumentId,
    version: DocumentVersion,
    kind: InstrumentKind,
    /// Every provision, keyed by id, each with its full amendment history.
    /// A `BTreeMap` so iteration order is the legal ordering (`s 12 < s 12A
    /// < s 13`) and therefore deterministic — CLAUDE.md requires
    /// deterministic behaviour, and a `HashMap` here would make every report
    /// this crate produces come out in a different order each run.
    provisions: BTreeMap<ProvisionId, ProvisionHistory>,
    dictionary: Dictionary,
    /// Everything we found and refused to guess about.
    anomalies: Vec<Anomaly>,
}

impl Instrument {
    pub fn new(id: DocumentId, version: DocumentVersion, kind: InstrumentKind) -> Self {
        Self {
            id,
            version,
            kind,
            provisions: BTreeMap::new(),
            dictionary: Dictionary::default(),
            anomalies: Vec::new(),
        }
    }

    pub fn id(&self) -> &DocumentId {
        &self.id
    }

    pub fn version(&self) -> &DocumentVersion {
        &self.version
    }

    pub fn kind(&self) -> &InstrumentKind {
        &self.kind
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    /// Everything this crate found and would not resolve by guessing. A
    /// caller that ignores this list is throwing away the tool's honesty.
    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    pub fn provisions(&self) -> impl Iterator<Item = &ProvisionHistory> {
        self.provisions.values()
    }

    /// Adds a provision, or a superseding version of one already present.
    pub fn add_provision(&mut self, provision: Provision) -> Result<(), LegalError> {
        let id = provision.id().clone();
        match self.provisions.get_mut(&id) {
            Some(history) => history.supersede(provision),
            None => {
                self.provisions.insert(id, ProvisionHistory::new(provision));
                Ok(())
            }
        }
    }

    pub fn add_definition(&mut self, definition: Definition) {
        self.dictionary.insert(definition);
    }

    pub fn add_anomaly(&mut self, anomaly: Anomaly) {
        self.anomalies.push(anomaly);
    }

    /// Whether the instrument contains this provision (at any date).
    pub fn contains(&self, id: &ProvisionId) -> bool {
        self.provisions.contains_key(id)
    }

    /// The amendment history of one provision.
    pub fn history(&self, id: &ProvisionId) -> Option<&ProvisionHistory> {
        self.provisions.get(id)
    }

    /// **The primary query.** What did `id` say as at `as_at`?
    ///
    /// There is no un-dated variant. See [`crate::temporal`] for why.
    pub fn provision_as_at(
        &self,
        id: &ProvisionId,
        as_at: AsAtDate,
    ) -> Result<AsAtResult<'_>, LegalError> {
        self.provisions
            .get(id)
            .map(|history| history.as_at(as_at))
            .ok_or_else(|| LegalError::NoSuchProvision { id: id.clone() })
    }

    /// Every provision in force on `as_at`, in legal order.
    pub fn in_force_at(&self, as_at: AsAtDate) -> Vec<&Provision> {
        self.provisions
            .values()
            .filter_map(|history| match history.as_at(as_at) {
                AsAtResult::InForce(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    /// **Resolve a word against the instrument's own dictionary**, at the
    /// provision where it is used and as at a date.
    ///
    /// This is the ergonomic entry point for the highest-value operation in the
    /// crate. It exists on `Instrument` rather than only on [`Dictionary`]
    /// because the instrument is what knows which **Part** a provision sits in,
    /// and Part is needed to decide whether an "in this Part" definition
    /// governs — see [`crate::Provision::part`] for why Part is context rather
    /// than identity.
    ///
    /// Remember what a [`Resolution`] is and is not: it tells you *the
    /// instrument defines this word this way, here is where it says so*. It
    /// does not tell you what the provision means.
    pub fn meaning_of(
        &self,
        term: &str,
        used_in: &ProvisionId,
        as_at: AsAtDate,
    ) -> Resolution<'_> {
        self.dictionary.resolve(term, used_in, self.part_of(used_in), as_at)
    }

    /// Every defined term occurring in a provision's text, with the definition
    /// that governs it there and then. See [`Dictionary::terms_used_in`].
    pub fn defined_terms_in(
        &self,
        provision: &Provision,
        as_at: AsAtDate,
    ) -> Vec<crate::TermOccurrence<'_>> {
        self.dictionary.terms_used_in(
            provision.text(),
            provision.id(),
            provision.part(),
            as_at,
        )
    }

    /// The Part a provision sits in, if the instrument records one.
    fn part_of(&self, id: &ProvisionId) -> Option<crate::Numeral> {
        self.provisions
            .get(id)
            .and_then(|history| history.latest_known().0.part())
    }

    /// Resolves every cross-reference in the instrument against its own
    /// contents.
    ///
    /// Dangling references are **returned**, not dropped — see
    /// [`crate::reference`] for why a dangling reference is one of the most
    /// informative things this tool can report.
    pub fn resolve_references(&self, as_at: AsAtDate) -> ReferenceResolution {
        let mut resolution = ReferenceResolution::default();
        for history in self.provisions.values() {
            let AsAtResult::InForce(provision) = history.as_at(as_at) else {
                continue;
            };
            for reference in extract_references(provision) {
                match reference.target() {
                    ReferenceTarget::Internal(target) => {
                        // A reference resolves if the exact provision exists,
                        // OR if some provision the instrument holds sits
                        // inside the referenced unit ("subject to Part II"
                        // resolves if Part II has any provisions).
                        let exists = self.contains(target)
                            || self.provisions.keys().any(|id| target.contains(id));
                        if exists {
                            resolution.resolved.push(reference);
                        } else {
                            resolution.dangling.push(reference);
                        }
                    }
                    ReferenceTarget::External { .. } => resolution.external.push(reference),
                    ReferenceTarget::Unparsed(_) => resolution.unparsed.push(reference),
                }
            }
        }
        resolution
    }

    /// Runs every consistency check and records what it finds as anomalies.
    ///
    /// Called at the end of ingestion. Deliberately *additive*: it never
    /// removes or rewrites a provision, only annotates the instrument with
    /// what could not be determined.
    pub fn audit(&mut self, as_at: AsAtDate) {
        let mut found = Vec::new();

        // Dangling cross-references.
        for reference in self.resolve_references(as_at).dangling {
            let target = match reference.target() {
                ReferenceTarget::Internal(id) => Some(id.clone()),
                _ => None,
            };
            found.push(Anomaly::new(
                AnomalyKind::DanglingCrossReference {
                    raw: reference.raw().to_string(),
                    target,
                },
                reference.provenance().clone(),
            ));
        }

        // Overlapping in-force windows: a provision cannot have said two
        // different things on the same day.
        for history in self.provisions.values() {
            for (a, b) in history.overlapping_versions() {
                found.push(Anomaly::new(
                    AnomalyKind::OverlappingValidity {
                        id: history.id().clone(),
                        windows: vec![a.validity().to_string(), b.validity().to_string()],
                    },
                    a.provenance().clone(),
                ));
            }
        }

        // Unparseable numbering, surfaced from the ids themselves.
        for history in self.provisions.values() {
            if history.id().has_unrecognized() {
                let (provision, _warning) = history.latest_known();
                found.push(Anomaly::new(
                    AnomalyKind::UnparseableNumbering {
                        label: history.id().to_string(),
                    },
                    provision.provenance().clone(),
                ));
            }
        }

        // Conflicting definitions: same term, same scope, both in force.
        let mut by_key: BTreeMap<(String, String), Vec<&Definition>> = BTreeMap::new();
        for definition in self.dictionary.definitions() {
            if !definition.validity().covers(as_at) {
                continue;
            }
            let key = (
                definition.term().to_lowercase(),
                match definition.scope() {
                    DefinitionScope::Instrument => String::from("<instrument>"),
                    DefinitionScope::Part(n) => format!("Part {n}"),
                    DefinitionScope::Within(id) => id.to_string(),
                },
            );
            by_key.entry(key).or_default().push(definition);
        }
        for ((term, _scope), competing) in by_key {
            if competing.len() > 1 {
                found.push(Anomaly::new(
                    AnomalyKind::ConflictingDefinition {
                        term,
                        competing: competing
                            .iter()
                            .map(|d| d.provenance().citation())
                            .collect(),
                    },
                    competing[0].provenance().clone(),
                ));
            }
        }

        self.anomalies.extend(found);
    }
}
