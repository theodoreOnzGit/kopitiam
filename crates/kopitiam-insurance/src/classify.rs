//! Document classification: what *kind* of insurance document is this?
//!
//! A policy wording, a schedule, an endorsement and a benefit summary are four
//! structurally different documents, and mistaking one for another is a common
//! and consequential failure:
//!
//! * A **schedule** read as a wording yields no clauses and no definitions —
//!   and a caller may conclude the policy has none.
//! * A **wording** read as a schedule yields no numbers — and a caller may
//!   conclude there is no cover limit.
//! * An **endorsement** read as a wording is the worst of the four: its
//!   clauses look like base wording, so its overriding text gets filed as the
//!   contract's original terms and the override is lost entirely.
//!
//! So classification is explicit, it reports the **evidence** that decided it,
//! and it has an honest [`DocumentClass::Unknown`]. Guessing is not on the
//! menu: a document we cannot classify is a document a human should look at.
//!
//! # Composite packs
//!
//! Real policy documents are frequently *packs*: wording + schedule +
//! endorsements bound into one PDF. [`classify`] reports the **dominant**
//! class and lists what else it saw in `evidence`, while
//! [`crate::ingest`] extracts clauses, schedule and endorsements from whatever
//! is actually there. Classification steers the reader; it does not gate the
//! extraction.

use serde::{Deserialize, Serialize};

use crate::clause::ClauseRole;
use crate::endorsement;

/// What kind of insurance document this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentClass {
    /// The contract's terms: definitions, insuring clauses, exclusions,
    /// conditions. The same for every policyholder.
    PolicyWording,

    /// The policy-specific numbers: sums insured, limits, excess, premium,
    /// period of insurance. Mostly tables.
    Schedule,

    /// A modification to the wording. Short, and it **overrides**.
    Endorsement,

    /// A marketing or comparison summary of benefits, usually a plan-by-plan
    /// table. Note that a benefit summary is **not the contract** — the
    /// wording is — and a consumer should say so.
    BenefitSummary,

    /// We could not tell. An honest answer, and a signal that a human should
    /// look at the document.
    Unknown,
}

impl std::fmt::Display for DocumentClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::PolicyWording => "a policy wording",
            Self::Schedule => "a schedule",
            Self::Endorsement => "an endorsement",
            Self::BenefitSummary => "a benefit summary",
            Self::Unknown => "an unclassified document",
        };
        f.write_str(name)
    }
}

/// How sure we are.
///
/// Two values, not a number. A percentage confidence would be a number we
/// could not justify, and this crate does not manufacture those. What a reader
/// can actually use is the **evidence**, which is why it is carried alongside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Several independent signals agree.
    Strong,
    /// One weak signal, or signals that partly disagree. Worth a human's eye.
    Weak,
}

/// A classification, and the reasons for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Classification {
    class: DocumentClass,
    confidence: Confidence,
    evidence: Vec<String>,
}

impl Classification {
    /// What kind of document this is.
    pub fn class(&self) -> DocumentClass {
        self.class
    }

    /// How sure we are.
    pub fn confidence(&self) -> Confidence {
        self.confidence
    }

    /// The signals that decided it, in plain English. This is the part a human
    /// can actually check, so it is not optional.
    pub fn evidence(&self) -> &[String] {
        &self.evidence
    }
}

/// Structural facts about a document, gathered during ingestion, that
/// classification reasons over.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Shape {
    pub headings: usize,
    pub numbered_clauses: usize,
    pub table_rows: usize,
    pub definition_clauses: usize,
    pub exclusion_clauses: usize,
    pub endorsement_clauses: usize,
    pub total_clauses: usize,
    pub pages: usize,
    /// Columns in the widest table seen. A benefit summary's table has one
    /// column per plan (so >= 3); a schedule's is `label | value` (so 2).
    pub widest_table: usize,
}

/// The most words a *section banner* has. Longer "headings" are clause bodies.
///
/// `kopitiam-document` classifies a numbered clause's opening line as a
/// heading, so the heading list handed to this function contains whole
/// sentences: `"3.1 We will pay the Benefit shown in the Schedule if the
/// Insured Person suffers bodily injury..."`. Matching document-type keywords
/// against those sentences is a trap — an ordinary *coverage* clause that
/// happens to mention "the Schedule" would make the whole document a schedule,
/// and the wording's definitions section would then never be looked for. Only
/// short headings are treated as banners naming the document's type.
const MAX_BANNER_WORDS: usize = 8;

/// Classifies a document from its title, its headings and its shape.
pub(crate) fn classify(title: Option<&str>, headings: &[String], shape: Shape) -> Classification {
    let mut evidence: Vec<String> = Vec::new();
    let title_lower = title.unwrap_or("").to_lowercase();
    let heading_text = headings
        .iter()
        .filter(|heading| heading.split_whitespace().count() <= MAX_BANNER_WORDS)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ; ")
        .to_lowercase();

    let says = |needle: &str| title_lower.contains(needle) || heading_text.contains(needle);

    // --- Endorsement. Checked first: it is the most costly to miss, because
    // an endorsement misread as a wording loses the override entirely.
    let endorsement_titled = endorsement::ENDORSEMENT_HEADINGS
        .iter()
        .any(|marker| title_lower.contains(marker));
    if endorsement_titled {
        evidence.push(format!("the title says {:?}", title.unwrap_or("")));
    }
    if shape.endorsement_clauses > 0 {
        evidence.push(format!(
            "{} clause(s) use endorsement language (\"is deleted and replaced\", ...)",
            shape.endorsement_clauses
        ));
    }
    if endorsement_titled && shape.endorsement_clauses > 0 {
        return Classification {
            class: DocumentClass::Endorsement,
            confidence: Confidence::Strong,
            evidence,
        };
    }
    if endorsement_titled || (shape.endorsement_clauses > 0 && shape.definition_clauses == 0) {
        return Classification {
            class: DocumentClass::Endorsement,
            confidence: Confidence::Weak,
            evidence,
        };
    }

    // --- Policy wording: a definitions section plus a numbered clause
    // hierarchy is the signature, and no other document type has both.
    if shape.definition_clauses > 0 {
        evidence.push(format!(
            "{} clause(s) sit under a definitions heading",
            shape.definition_clauses
        ));
    }
    if shape.exclusion_clauses > 0 {
        evidence.push(format!(
            "{} clause(s) sit under an exclusions heading",
            shape.exclusion_clauses
        ));
    }
    if shape.numbered_clauses > 0 {
        evidence.push(format!(
            "{} numbered clause(s) (e.g. \"4.2.1\")",
            shape.numbered_clauses
        ));
    }
    if shape.definition_clauses > 0 && shape.numbered_clauses >= 3 {
        return Classification {
            class: DocumentClass::PolicyWording,
            confidence: Confidence::Strong,
            evidence,
        };
    }

    // --- Benefit summary: a wide plan-by-plan table dominates the document.
    let table_dominated = shape.table_rows > 0 && shape.table_rows * 2 >= shape.total_clauses;
    if shape.widest_table >= 3 && table_dominated {
        evidence.push(format!(
            "a {}-column table dominates the document (one column per plan)",
            shape.widest_table
        ));
        if says("summary of benefits") || says("benefit summary") || says("benefits") {
            evidence.push("the title or headings say \"benefits\"".to_string());
            return Classification {
                class: DocumentClass::BenefitSummary,
                confidence: Confidence::Strong,
                evidence,
            };
        }
        return Classification {
            class: DocumentClass::BenefitSummary,
            confidence: Confidence::Weak,
            evidence,
        };
    }

    // --- Schedule: `label | value` tables, and few or no prose clauses.
    let schedule_titled =
        says("schedule") || says("certificate of insurance") || says("policy certificate");
    if schedule_titled {
        evidence.push("the title or a heading says \"schedule\" / \"certificate\"".to_string());
    }
    if table_dominated {
        evidence.push(format!(
            "{} table row(s) dominate {} clause(s) of prose",
            shape.table_rows, shape.total_clauses
        ));
    }
    if schedule_titled && table_dominated {
        return Classification {
            class: DocumentClass::Schedule,
            confidence: Confidence::Strong,
            evidence,
        };
    }
    if schedule_titled || (table_dominated && shape.definition_clauses == 0) {
        return Classification {
            class: DocumentClass::Schedule,
            confidence: Confidence::Weak,
            evidence,
        };
    }

    // --- A wording without a definitions section we could find. Weak, and the
    // missing definitions section is itself reported as an anomaly.
    if shape.numbered_clauses >= 3 || shape.exclusion_clauses > 0 {
        return Classification {
            class: DocumentClass::PolicyWording,
            confidence: Confidence::Weak,
            evidence,
        };
    }

    evidence.push("no structural signal identified this document".to_string());
    Classification {
        class: DocumentClass::Unknown,
        confidence: Confidence::Weak,
        evidence,
    }
}

/// Tallies a clause into the document's [`Shape`].
pub(crate) fn count_clause(shape: &mut Shape, role: ClauseRole, numbered: bool) {
    shape.total_clauses += 1;
    if numbered {
        shape.numbered_clauses += 1;
    }
    match role {
        ClauseRole::Definition => shape.definition_clauses += 1,
        ClauseRole::Exclusion => shape.exclusion_clauses += 1,
        ClauseRole::Endorsement => shape.endorsement_clauses += 1,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_definitions_section_plus_numbered_clauses_is_a_policy_wording() {
        let shape = Shape {
            definition_clauses: 6,
            numbered_clauses: 20,
            exclusion_clauses: 5,
            total_clauses: 30,
            ..Shape::default()
        };
        let result = classify(Some("Personal Accident Policy"), &[], shape);
        assert_eq!(result.class(), DocumentClass::PolicyWording);
        assert_eq!(result.confidence(), Confidence::Strong);
        assert!(!result.evidence().is_empty());
    }

    #[test]
    fn an_endorsement_is_not_mistaken_for_a_wording() {
        // The costliest misclassification: an endorsement filed as base wording
        // has its override silently absorbed into the contract's terms.
        let shape = Shape {
            endorsement_clauses: 1,
            numbered_clauses: 1,
            total_clauses: 2,
            ..Shape::default()
        };
        let result = classify(Some("Endorsement No. 1"), &[], shape);
        assert_eq!(result.class(), DocumentClass::Endorsement);
        assert_eq!(result.confidence(), Confidence::Strong);
    }

    #[test]
    fn a_label_value_table_document_is_a_schedule() {
        let shape = Shape {
            table_rows: 8,
            widest_table: 2,
            total_clauses: 9,
            ..Shape::default()
        };
        let result = classify(Some("Policy Schedule"), &[], shape);
        assert_eq!(result.class(), DocumentClass::Schedule);
        assert_eq!(result.confidence(), Confidence::Strong);
    }

    #[test]
    fn a_wide_plan_table_is_a_benefit_summary_not_a_schedule() {
        let shape = Shape {
            table_rows: 10,
            widest_table: 4,
            total_clauses: 11,
            ..Shape::default()
        };
        let result = classify(Some("Summary of Benefits"), &[], shape);
        assert_eq!(result.class(), DocumentClass::BenefitSummary);
    }

    #[test]
    fn an_unclassifiable_document_says_unknown_rather_than_guessing() {
        let result = classify(None, &[], Shape::default());
        assert_eq!(result.class(), DocumentClass::Unknown);
        assert!(!result.evidence().is_empty());
    }
}
