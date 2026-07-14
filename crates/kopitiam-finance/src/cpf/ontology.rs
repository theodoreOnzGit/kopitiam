//! Emitting CPF policy into KOPITIAM's shared knowledge graph.
//!
//! # Why CPF policy is ontology facts and not a lookup table
//!
//! The alternative design — CPF as a self-contained calculator with its own
//! private data — would work, and would be a dead end. KOPITIAM's architecture is
//! that *everything the platform knows lands in one graph*: Rust symbols, PDF
//! sections, engineering decisions, and now policy rules, all as
//! `kopitiam_ontology::Entity`. That is what makes it possible to ask questions
//! that cross domains — "which sections of which documents does our CPF model
//! depend on, and have any of them changed?" — without every consumer growing a
//! bespoke CPF integration.
//!
//! So each dated, cited policy rule becomes an [`EntityKind::Fact`]; each source
//! document section becomes an [`EntityKind::Section`]; and each fact is joined to
//! its section by a [`RelationshipKind::DocumentedIn`] edge. The citation is not
//! flattened into prose — it goes into the metadata as structured JSON, so a
//! consumer can filter on `source_kind`, on effective date, or on publisher.
//!
//! # The gaps are emitted too
//!
//! [`CpfKnowledge`] also emits a `Fact` for every part of CPF that KOPITIAM
//! *deliberately does not model* (see [`crate::cpf::published`]), carrying
//! `"populated": false` and the reason.
//!
//! This is not padding. A knowledge graph that silently omits what it does not
//! know cannot be distinguished from one that has nothing to say — and a
//! downstream agent querying it would have no way to learn that post-55
//! allocation ratios are *missing* rather than *irrelevant*. Recording the shape
//! of our ignorance is the whole difference between an honest knowledge base and
//! a misleading one. It is also, concretely, the work queue.

use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::{Value, json};

use crate::cpf::citation::Citation;
use crate::cpf::document::PROVIDER;
use crate::cpf::query::CpfPolicy;
use crate::cpf::structure::Residency;
use crate::cpf::temporal::Dated;

/// Facts and relationships produced from a [`CpfPolicy`], ready to be ingested by
/// `kopitiam-knowledge`.
///
/// Mirrors the shape of `kopitiam_semantic::ProviderOutput` deliberately, but is
/// declared here rather than imported: `kopitiam-finance` has no business
/// depending on the Rust-toolchain provider crate just to name a pair of `Vec`s,
/// and the one-way dependency rule in CLAUDE.md is worth more than the four lines
/// this saves.
#[derive(Debug, Clone, Default)]
pub struct CpfKnowledge {
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
}

impl CpfKnowledge {
    /// Turns the whole of a [`CpfPolicy`] — everything it knows, and everything it
    /// knows it does not know — into graph entities.
    pub fn from_policy(policy: &CpfPolicy) -> Self {
        // Every distinct source document section gets exactly one Section entity,
        // and every fact derived from it is linked back to it. That deduplication
        // is the edge that makes "which document does this rule come from?" a
        // graph query rather than a string comparison.
        let mut builder = Builder::default();

        // -- Contribution rates -------------------------------------------
        if let Ok(table) = policy.contribution_table(Residency::CitizenOrPrFromThirdYear) {
            for entry in table.entries() {
                for (band, rates) in entry.value.bands() {
                    builder.emit(
                        format!("CPF contribution rate — {band} — {}", entry.effective),
                        entry,
                        json!({
                            "rule": "contribution_rate",
                            "residency": Residency::CitizenOrPrFromThirdYear.label(),
                            "wage_band": "Total wages of $750/month and above",
                            "age_band": band.label(),
                            "employer_rate": rates.employer.to_string(),
                            "employee_rate": rates.employee.to_string(),
                            "total_rate": rates.total().to_string(),
                            "employer_basis_points": rates.employer.basis_points(),
                            "employee_basis_points": rates.employee.basis_points(),
                        }),
                    );
                }
            }
        }

        // -- Allocation ratios ---------------------------------------------
        if let Ok(table) = policy.allocation_table(Residency::CitizenOrPrFromThirdYear) {
            for entry in table.entries() {
                for (band, ratios) in entry.value.bands() {
                    builder.emit(
                        format!("CPF allocation ratio — {band} — {}", entry.effective),
                        entry,
                        json!({
                            "rule": "allocation_ratio",
                            "age_band": band.label(),
                            "ordinary_account": ratios.ordinary.to_string(),
                            "special_account": ratios.special.to_string(),
                            "medisave_account": ratios.medisave.to_string(),
                            "note": "Ratios of the TOTAL contribution, not of wages. Sum to exactly 1.",
                        }),
                    );
                }
            }
        }

        // -- Wage ceilings ---------------------------------------------------
        for entry in policy.wage_ceiling_table().entries() {
            builder.emit(
                format!("CPF wage ceilings — {}", entry.effective),
                entry,
                json!({
                    "rule": "wage_ceiling",
                    "ordinary_wage_ceiling_monthly": entry.value.ordinary_wage_monthly.to_string(),
                    "annual_total_wage_ceiling": entry.value.annual_total_wage.to_string(),
                    "additional_wage_ceiling_formula":
                        "annual total wage ceiling MINUS ordinary wages SUBJECT TO CPF for the year",
                }),
            );
        }

        // -- Retirement sums --------------------------------------------------
        for entry in policy.retirement_sum_table().entries() {
            builder.emit(
                format!(
                    "CPF retirement sums — cohort turning 55 in {}",
                    entry.effective.start().year()
                ),
                entry,
                json!({
                    "rule": "retirement_sums",
                    "cohort_year": entry.effective.start().year(),
                    "keyed_on": "the member's 55th birthday, NOT the date of the query",
                    "basic_retirement_sum": entry.value.basic.to_string(),
                    "full_retirement_sum": entry.value.full.to_string(),
                    "enhanced_retirement_sum": entry.value.enhanced.to_string(),
                }),
            );
        }

        // -- Interest floors ---------------------------------------------------
        for entry in policy.interest_table().entries() {
            builder.emit(
                format!("CPF interest floors — {}", entry.effective),
                entry,
                json!({
                    "rule": "interest_floor",
                    "ordinary_account_floor": entry.value.ordinary.to_string(),
                    "special_medisave_retirement_floor":
                        entry.value.special_medisave_retirement.to_string(),
                    "extra_interest_first_tier": entry.value.extra_interest_first_tier.to_string(),
                    "extra_interest_ordinary_cap": entry.value.extra_interest_ordinary_cap.to_string(),
                    "extra_interest_second_tier": entry.value.extra_interest_second_tier.to_string(),
                    "caveat": "FLOOR rates only. The declared quarterly rate can exceed these and is not modelled.",
                }),
            );
        }

        let mut out = builder.finish();

        // -- What we know we do not know ---------------------------------------
        out.entities.extend(gaps());

        out
    }

    /// The facts only — convenience for a caller that does not want the document
    /// structure.
    pub fn facts(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter().filter(|e| e.kind == EntityKind::Fact)
    }
}

/// Accumulates entities while deduplicating the `Section` entities that facts
/// point at, so that every rule from one published revision converges on a single
/// document node.
#[derive(Default)]
struct Builder {
    out: CpfKnowledge,
    /// Section key -> the Section entity already emitted for it.
    sections: Vec<(String, EntityId)>,
}

impl Builder {
    /// Emits one policy `Fact`, plus (once) the `Section` it is documented in, and
    /// the `DocumentedIn` edge joining them.
    fn emit<T>(&mut self, name: String, entry: &Dated<T>, rule: Value) {
        let section = self.section_for(&entry.source);
        let fact = fact(name, entry, rule);
        self.out.relationships.push(Relationship::new(
            fact.id,
            section,
            RelationshipKind::DocumentedIn,
        ));
        self.out.entities.push(fact);
    }

    fn section_for(&mut self, citation: &Citation) -> EntityId {
        let key = format!("{} — {}", citation.document, citation.locator);
        if let Some((_, id)) = self.sections.iter().find(|(k, _)| *k == key) {
            return *id;
        }
        let entity = Entity::new(EntityKind::Section, key.clone(), PROVIDER).with_metadata(json!({
            "domain": "cpf",
            "publisher": citation.publisher,
            "document": citation.document,
            "locator": citation.locator,
            "url": citation.url,
            "source_kind": citation.source_kind.to_string(),
            "published": citation.published.map(|d| d.to_string()),
        }));
        let id = entity.id;
        self.out.entities.push(entity);
        self.sections.push((key, id));
        id
    }

    fn finish(self) -> CpfKnowledge {
        self.out
    }
}

/// Builds a `Fact` entity carrying its effective period and its full citation as
/// structured metadata.
///
/// The citation goes in as JSON rather than as a rendered string so that a
/// consumer can *filter* on it — "show me every CPF fact that is still merely
/// transcribed" is the query that turns this crate's honesty into a work queue.
fn fact<T>(name: String, entry: &Dated<T>, mut rule: Value) -> Entity {
    let citation = &entry.source;
    if let Some(obj) = rule.as_object_mut() {
        obj.insert("domain".into(), json!("cpf"));
        obj.insert("populated".into(), json!(true));
        obj.insert("effective_from".into(), json!(entry.effective.start().to_string()));
        obj.insert(
            "effective_until".into(),
            json!(entry.effective.end().map(|d| d.to_string())),
        );
        obj.insert(
            "citation".into(),
            json!({
                "publisher": citation.publisher,
                "document": citation.document,
                "locator": citation.locator,
                "url": citation.url,
                "published": citation.published.map(|d| d.to_string()),
                "source_kind": citation.source_kind.to_string(),
                "note": citation.note,
            }),
        );
    }
    Entity::new(EntityKind::Fact, name, PROVIDER).with_metadata(rule)
}

/// Every part of CPF that KOPITIAM deliberately does not model, as a `Fact`.
///
/// Keep this list in step with the "What is deliberately NOT populated" table in
/// [`crate::cpf::published`]. It is the honest inventory of the crate's ignorance,
/// and — usefully — a machine-readable backlog.
fn gaps() -> Vec<Entity> {
    const GAPS: &[(&str, &str)] = &[
        (
            "CPF allocation ratios for members aged 55 and above",
            "The Special Account was closed for members aged 55 and above in January 2025 and \
             their savings restructured into the Retirement and Ordinary Accounts. The current \
             ratios, and the post-55 account structure itself, are not known with enough \
             confidence to transcribe. This is the largest gap in KOPITIAM's CPF model.",
        ),
        (
            "CPF graduated contribution rates for total wages below $750/month",
            "Below $750/month the employee's share is phased in by a formula depending on both \
             the wage and the age band. Encoding a wrong formula would under- or over-deduct from \
             the lowest-paid members. Not attempted.",
        ),
        (
            "CPF contribution rates for Permanent Residents in their 1st and 2nd year",
            "Graduated rates, with three employer/employee combinations available by joint \
             election, each with its own allocation table. Not attempted.",
        ),
        (
            "CPF declared (as opposed to floor) interest rates, and interest computation",
            "Rates are declared quarterly against a pegged formula and can exceed the statutory \
             floor. Interest accrues on the lowest monthly balance with extra interest applied \
             across accounts in a prescribed order. Only the floors are modelled; no computation \
             is provided.",
        ),
        (
            "CPF retirement sums for cohorts turning 55 from 2027 onward",
            "A schedule has been announced but is not transcribed here. Queries for these cohorts \
             fail rather than extrapolating the 3.5%/year trend.",
        ),
        (
            "CPF housing withdrawal limits (Valuation Limit, Withdrawal Limit)",
            "Not modelled. Misreading a housing withdrawal limit can cost someone their home; this \
             crate does not guess at one.",
        ),
        (
            "CPF LIFE, Workfare, top-up schemes, Basic Healthcare Sum, self-employed contributions, \
             and public-sector pensionable employees",
            "Separate schemes with their own rules. Out of scope for this scaffold.",
        ),
        (
            "Allocation rounding order (which account takes the residual)",
            "KOPITIAM computes MediSave and Special from the ratios and gives the Ordinary Account \
             the residual, so that no cent is lost. The residual structure is forced; WHICH account \
             is the residual is an assumption not verified against the CPF Board's worked examples. \
             It can differ from CPF by at most one dollar between two accounts of the same member, \
             and never changes the total.",
        ),
        (
            "Whether a member born on the 1st of a month attains their age in the previous month",
            "Singapore statute sometimes applies the common-law rule that a person attains an age \
             at the start of the day before their birthday. If CPF applies it, a member born on the \
             1st changes contribution band one month earlier than KOPITIAM says. The straightforward \
             reading is implemented; the ambiguity is recorded rather than resolved by guessing.",
        ),
    ];

    GAPS.iter()
        .map(|(dimension, reason)| {
            Entity::new(EntityKind::Fact, format!("GAP: {dimension}"), PROVIDER).with_metadata(json!({
                "domain": "cpf",
                "rule": "not_modelled",
                "populated": false,
                "dimension": dimension,
                "reason": reason,
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_policy_becomes_graph_entities() {
        let knowledge = CpfKnowledge::from_policy(&CpfPolicy::published());

        let facts: Vec<_> = knowledge.facts().collect();
        let sections: Vec<_> = knowledge
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Section)
            .collect();

        assert!(!facts.is_empty());
        assert!(!sections.is_empty());
        assert!(!knowledge.relationships.is_empty());

        // 3 revisions x 5 contribution bands = 15
        // 1 revision  x 4 allocation bands   =  4
        // 5 wage-ceiling revisions           =  5
        // 4 retirement cohorts               =  4
        // 1 interest revision                =  1
        //                                    = 29 populated facts
        let populated = facts
            .iter()
            .filter(|e| e.metadata["populated"] == json!(true))
            .count();
        assert_eq!(populated, 29, "every cited rule must reach the graph");

        // Every populated fact is linked to the section it came from.
        assert_eq!(
            knowledge.relationships.len(),
            populated,
            "every policy fact must be DocumentedIn a source section"
        );
        assert!(
            knowledge
                .relationships
                .iter()
                .all(|r| r.kind == RelationshipKind::DocumentedIn)
        );
    }

    /// Every emitted fact carries its citation and its effective period as
    /// structured, filterable metadata — not as prose.
    #[test]
    fn every_emitted_fact_carries_a_structured_citation() {
        let knowledge = CpfKnowledge::from_policy(&CpfPolicy::published());

        for fact in knowledge.facts().filter(|e| e.metadata["populated"] == json!(true)) {
            assert_eq!(fact.source, PROVIDER);

            let citation = &fact.metadata["citation"];
            assert!(citation.is_object(), "{} has no citation", fact.name);
            assert_eq!(citation["publisher"], json!("Central Provident Fund Board"));
            assert!(citation["document"].as_str().is_some_and(|s| !s.is_empty()));
            assert!(citation["locator"].as_str().is_some_and(|s| !s.is_empty()));
            assert!(citation["url"].as_str().is_some(), "{} has no URL", fact.name);

            // The honesty label must survive the trip into the graph.
            let kind = citation["source_kind"].as_str().unwrap();
            assert!(
                kind == "transcribed" || kind == "derived",
                "shipped facts are transcribed or derived, not extracted: {} claims {kind}",
                fact.name
            );

            assert!(fact.metadata["effective_from"].as_str().is_some());
        }
    }

    /// The gaps reach the graph as first-class facts. A consumer can discover
    /// what KOPITIAM does not know without reading the source.
    #[test]
    fn the_gaps_are_emitted_as_facts_too() {
        let knowledge = CpfKnowledge::from_policy(&CpfPolicy::published());

        let gaps: Vec<_> = knowledge
            .facts()
            .filter(|e| e.metadata["populated"] == json!(false))
            .collect();

        assert_eq!(gaps.len(), 9, "the inventory of ignorance is complete");
        for gap in &gaps {
            assert!(gap.name.starts_with("GAP: "));
            assert!(
                gap.metadata["reason"].as_str().is_some_and(|r| r.len() > 40),
                "a gap must explain itself: {}",
                gap.name
            );
        }

        // The biggest one specifically.
        assert!(
            gaps.iter()
                .any(|g| g.name.contains("aged 55 and above")),
            "the post-55 allocation gap must be discoverable from the graph"
        );
    }

    /// Facts from the same published revision share one Section entity — the
    /// document structure is deduplicated, so "what does this rule come from"
    /// converges on a single node.
    #[test]
    fn facts_from_one_revision_share_one_section() {
        let knowledge = CpfKnowledge::from_policy(&CpfPolicy::published());
        let sections = knowledge
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Section)
            .count();

        // 3 contribution revisions + 1 allocation + 5 ceilings + 4 cohorts + 1 interest
        assert_eq!(sections, 14);
    }
}
