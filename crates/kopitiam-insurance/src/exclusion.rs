//! Exclusions — what the policy does **not** cover.
//!
//! An insurance policy is defined at least as much by its exclusions as by its
//! grant of cover. The insuring clause is a wide, generous sentence; the
//! exclusions are what it actually amounts to. Treating them as an
//! afterthought — a list to be shown if the user asks — is how a tool ends up
//! telling someone they are covered when they are not.
//!
//! # Why classification is structural, not linguistic
//!
//! The naive approach is to look for exclusionary language ("we will not pay",
//! "is excluded"). It fails on the most common exclusion of all, which reads:
//!
//! > 4.1  Any claim arising directly or indirectly from war or invasion.
//!
//! That is a bare noun phrase. It contains no verb, no negation, and nothing
//! whatsoever to say it is an exclusion. It is an exclusion **only because it
//! is printed under a heading that says `Exclusions`** — and a classifier that
//! reads sentences instead of structure will happily file it as coverage. An
//! exclusion presented as coverage is the worst bug this crate can have, so
//! **the enclosing section wins**: [`classify`] consults the whole
//! [`SectionPath`], and only falls back to sentence-level language when the
//! structure says nothing.
//!
//! # Write-backs
//!
//! An exclusion section is not uniformly exclusionary. Standard drafting puts
//! **write-backs** (also: carve-backs) inside it — sentences that *restore*
//! cover the exclusion has just taken away:
//!
//! > This exclusion shall not apply to a fire caused by an insured peril.
//!
//! Read as an exclusion, that sentence means the exact opposite of what it
//! says. [`ExclusionEffect::WritesBack`] exists so it cannot be.
//!
//! [`SectionPath`]: crate::SectionPath

use serde::{Deserialize, Serialize};

use crate::clause::{Clause, ClauseRole};
use crate::provenance::{Provenance, SectionPath};

/// Heading text that marks a section as exclusionary. Matched as a
/// case-insensitive substring against **every** heading in a clause's section
/// path, not just its immediate parent — a clause nested three levels below
/// `General Exclusions` is still an exclusion.
const EXCLUSION_HEADINGS: &[&str] = &[
    "exclusion",
    "exclusions",
    "what is not covered",
    "what we do not cover",
    "what we will not pay",
    "not covered",
    "limitations and exclusions",
];

/// Heading text that marks a section as granting cover.
const COVERAGE_HEADINGS: &[&str] = &[
    "what is covered",
    "what we cover",
    "what we will pay",
    "insuring agreement",
    "insuring clause",
    "scope of cover",
    "benefits",
    "benefit",
    "cover",
    "coverage",
];

/// Heading text that marks a section as conditions/duties.
const CONDITION_HEADINGS: &[&str] = &[
    "condition",
    "conditions",
    "general conditions",
    "duties of the insured",
    "your duties",
    "warranties",
    "claims procedure",
];

/// Sentence openings that state an exclusion outright.
const EXCLUSIONARY_PHRASES: &[&str] = &[
    "we will not pay",
    "we shall not pay",
    "the company will not pay",
    "the company shall not be liable",
    "this policy does not cover",
    "this policy excludes",
    "no benefit is payable",
    "no benefit shall be payable",
    "is not covered",
    "are not covered",
    "shall be excluded",
    "is excluded",
    "are excluded",
];

/// Sentence openings that state a grant of cover.
const COVERAGE_PHRASES: &[&str] = &[
    "we will pay",
    "we shall pay",
    "the company will pay",
    "the company shall pay",
    "this policy covers",
    "we will reimburse",
    "we will indemnify",
    "the company will indemnify",
    "cover is provided",
    "benefit is payable",
];

/// Phrases that **restore** cover inside an exclusion section — a write-back.
/// Read as exclusions, these clauses mean the opposite of what they say. See
/// the module docs.
const WRITE_BACK_PHRASES: &[&str] = &[
    "this exclusion shall not apply",
    "this exclusion does not apply",
    "these exclusions shall not apply",
    "these exclusions do not apply",
    "the above exclusion shall not apply",
    "the above exclusions shall not apply",
    "shall not apply to",
    "notwithstanding the above, we will pay",
];

/// What an exclusion-section clause actually *does*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionEffect {
    /// Takes cover away.
    Excludes,
    /// **Gives cover back.** A write-back sitting inside an exclusion section
    /// ("this exclusion shall not apply to..."). Reading it as an exclusion
    /// inverts its meaning.
    WritesBack,
}

/// A clause that removes (or, for a write-back, restores) cover.
///
/// Carries the clause it came from, so its verbatim text and citation travel
/// with it. This crate never decides whether an exclusion *bites* on a given
/// set of facts — that is adjudication, and it is not ours to do. It locates
/// the exclusion and shows the reader the words.
#[derive(Debug, Clone, PartialEq)]
pub struct Exclusion<'a> {
    clause: &'a Clause,
    effect: ExclusionEffect,
}

impl<'a> Exclusion<'a> {
    pub(crate) fn new(clause: &'a Clause, effect: ExclusionEffect) -> Self {
        Self { clause, effect }
    }

    /// The clause that states the exclusion.
    pub fn clause(&self) -> &'a Clause {
        self.clause
    }

    /// Whether this clause takes cover away or gives it back.
    pub fn effect(&self) -> ExclusionEffect {
        self.effect
    }

    /// The exclusion's verbatim wording.
    pub fn text(&self) -> &'a str {
        self.clause.text()
    }

    /// Where it is printed.
    pub fn provenance(&self) -> Provenance {
        self.clause.provenance()
    }
}

/// Decides what a clause does in the contract, from its structure first and
/// its language second.
///
/// The section path is consulted before the sentence, for the reason in the
/// module docs: an exclusion routinely does not *read* like one, and getting
/// this backwards is the most dangerous error this crate can make.
///
/// When neither structure nor language gives an answer, the result is
/// [`ClauseRole::Unclassified`] — an honest "we do not know", not a guess.
pub(crate) fn classify(path: &SectionPath, heading: Option<&str>, text: &str) -> ClauseRole {
    let lower = text.to_lowercase();
    let heading_lower = heading.unwrap_or("").to_lowercase();

    // Structure first. A clause printed under `Exclusions` is an exclusion,
    // whatever its sentence looks like.
    let in_exclusions = path.any(|h| contains_any(&h.to_lowercase(), EXCLUSION_HEADINGS))
        || contains_any(&heading_lower, EXCLUSION_HEADINGS);

    if in_exclusions {
        return ClauseRole::Exclusion;
    }

    if crate::definition::is_definitions_heading(&heading_lower)
        || path.any(crate::definition::is_definitions_heading)
    {
        return ClauseRole::Definition;
    }

    if path.any(|h| contains_any(&h.to_lowercase(), CONDITION_HEADINGS))
        || contains_any(&heading_lower, CONDITION_HEADINGS)
    {
        return ClauseRole::Condition;
    }

    if path.any(|h| contains_any(&h.to_lowercase(), COVERAGE_HEADINGS))
        || contains_any(&heading_lower, COVERAGE_HEADINGS)
    {
        return ClauseRole::Coverage;
    }

    // Structure said nothing. Only now look at the sentence — and check the
    // exclusionary reading first, because a false "covered" is worse than a
    // false "not covered".
    if contains_any(&lower, EXCLUSIONARY_PHRASES) {
        return ClauseRole::Exclusion;
    }
    if contains_any(&lower, COVERAGE_PHRASES) {
        return ClauseRole::Coverage;
    }

    ClauseRole::Unclassified
}

/// Whether an exclusion-section clause is a write-back (restores cover) rather
/// than an exclusion (removes it).
pub(crate) fn effect_of(text: &str) -> ExclusionEffect {
    let lower = text.to_lowercase();
    if contains_any(&lower, WRITE_BACK_PHRASES) {
        ExclusionEffect::WritesBack
    } else {
        ExclusionEffect::Excludes
    }
}

/// Whether a clause's structure and its language **disagree** — e.g. a clause
/// printed under `Exclusions` whose sentence reads as a grant of cover.
///
/// Structure wins for classification (see [`classify`]), but the disagreement
/// itself is worth surfacing: it usually means either our heading detection
/// went wrong, or the document really does say something confusing. Either way
/// the reader should go and look at the words. See [`crate::Anomaly`].
pub(crate) fn signals_conflict(role: ClauseRole, text: &str) -> bool {
    let lower = text.to_lowercase();
    match role {
        ClauseRole::Exclusion => {
            // A write-back is a *known* exception, not a conflict.
            !contains_any(&lower, WRITE_BACK_PHRASES) && contains_any(&lower, COVERAGE_PHRASES)
        }
        ClauseRole::Coverage => contains_any(&lower, EXCLUSIONARY_PHRASES),
        _ => false,
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All wording invented for the test; none of it is any insurer's.
    #[test]
    fn a_bare_noun_phrase_under_an_exclusions_heading_is_an_exclusion() {
        // The clause says nothing about coverage. Only the heading does. This
        // is the case a language-only classifier gets exactly backwards.
        let role = classify(
            &SectionPath::new(["Section 4 — General Exclusions"]),
            None,
            "Any claim arising directly or indirectly from war or invasion.",
        );
        assert_eq!(role, ClauseRole::Exclusion);
    }

    #[test]
    fn a_deeply_nested_exclusion_is_still_an_exclusion() {
        let role = classify(
            &SectionPath::new(["Part II — Benefits", "Section 4 — Exclusions", "4.2 War"]),
            Some("Nuclear risks"),
            "Ionising radiation from any nuclear fuel.",
        );
        assert_eq!(role, ClauseRole::Exclusion);
    }

    #[test]
    fn a_coverage_clause_is_not_mistaken_for_an_exclusion() {
        let role = classify(
            &SectionPath::new(["Section 3 — What Is Covered"]),
            None,
            "We will pay the daily hospital benefit shown in the Schedule.",
        );
        assert_eq!(role, ClauseRole::Coverage);
    }

    #[test]
    fn language_only_fires_when_structure_is_silent() {
        // No heading at all: fall back to the sentence.
        assert_eq!(
            classify(
                &SectionPath::default(),
                None,
                "The Company shall not be liable for any consequential loss.",
            ),
            ClauseRole::Exclusion
        );
        assert_eq!(
            classify(
                &SectionPath::default(),
                None,
                "We will indemnify the Insured against legal liability.",
            ),
            ClauseRole::Coverage
        );
    }

    #[test]
    fn an_unclassifiable_clause_says_so() {
        assert_eq!(
            classify(&SectionPath::default(), None, "This Policy is governed by law."),
            ClauseRole::Unclassified
        );
    }

    #[test]
    fn a_write_back_inside_an_exclusions_section_is_not_an_exclusion() {
        // Read as an exclusion, this sentence means the opposite of what it
        // says — it *gives cover back*.
        assert_eq!(
            effect_of("This exclusion shall not apply to a fire caused by lightning."),
            ExclusionEffect::WritesBack
        );
        assert_eq!(
            effect_of("Any claim arising from war or invasion."),
            ExclusionEffect::Excludes
        );
    }

    #[test]
    fn a_write_back_is_not_reported_as_a_signal_conflict() {
        assert!(!signals_conflict(
            ClauseRole::Exclusion,
            "This exclusion shall not apply and we will pay the claim.",
        ));
    }

    #[test]
    fn structure_and_language_disagreeing_is_surfaced() {
        assert!(signals_conflict(
            ClauseRole::Exclusion,
            "We will pay for damage caused by fire.",
        ));
        assert!(signals_conflict(
            ClauseRole::Coverage,
            "This policy does not cover wear and tear.",
        ));
    }
}
