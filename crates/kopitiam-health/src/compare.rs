//! Compare the same term across several policies — and refuse when the term
//! does not mean the same thing in each.
//!
//! # The comparison trap
//!
//! Policy comparison is the thing a human most wants from a tool like this, and
//! it is where a tool like this most easily does harm. Two wordings state a
//! deductible of S$3,500 for "hospitalisation". A table showing them side by
//! side, equal, is a lie if one document defines "hospitalisation" as an
//! overnight admission and the other includes day surgery — because then they
//! are two different deductibles, on two different sets of events, that happen
//! to share a number.
//!
//! The whole value of the comparison is destroyed by the one thing that makes
//! it look useful: the tidy table.
//!
//! So: before comparing any term, this module checks the definitions that term
//! rests on (see [`crate::TermKind::depends_on_definitions`]). If the policies
//! word a load-bearing definition differently, the comparison comes back
//! [`Comparability::Incomparable`] — **with both terms still attached**, and
//! both definitions, verbatim, so a human can look at them and decide. It does
//! not hide the terms; it declines to tell you they are equivalent.
//!
//! # We compare text, not meaning
//!
//! Divergence is detected by comparing the definitions' *text* (whitespace- and
//! case-normalised). Two definitions that differ only in phrasing will be
//! flagged even though they may mean the same thing.
//!
//! This is deliberate, and it is the direction to be wrong in. Deciding that
//! two differently-worded definitions of "pre-existing condition" are
//! *equivalent* is a legal judgment, and a wrong one costs a claimant their
//! claim. Deciding they are *different* costs a human thirty seconds of
//! reading. We spend the thirty seconds.

use std::collections::BTreeMap;

use kopitiam_insurance::{Provenance, Resolution};

use crate::policy::{PolicyId, PolicyLayer};
use crate::term::{PolicyTerm, TermKind};

/// What one policy says about the term being compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyPosition {
    /// The policy states the term. Possibly more than once — a wording states a
    /// different deductible per ward class, and all of them belong in the
    /// comparison.
    Stated(Vec<PolicyTerm>),

    /// The policy does not state the term anywhere this crate looked.
    ///
    /// **Not the same as stating zero, and not the same as excluding it.** It
    /// most often means the figure lives in a benefit schedule or table that
    /// this scaffold's extraction did not read. Rendering it as a blank cell in
    /// a table and letting the reader draw their own conclusion is exactly the
    /// harm this type exists to prevent — hence a named variant, which a
    /// renderer has to handle deliberately.
    NotStated,
}

impl PolicyPosition {
    /// The terms, if any were stated.
    pub fn terms(&self) -> &[PolicyTerm] {
        match self {
            Self::Stated(terms) => terms,
            Self::NotStated => &[],
        }
    }

    /// Whether any stated term is unresolved.
    pub fn has_ambiguity(&self) -> bool {
        self.terms().iter().any(PolicyTerm::is_ambiguous)
    }
}

/// One policy's entry in a comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonEntry {
    /// Which policy.
    pub policy: PolicyId,
    /// Its name.
    pub name: String,
    /// What it says.
    pub position: PolicyPosition,
}

/// What one policy says a word means.
///
/// Mirrors `kopitiam-insurance`'s [`Resolution`], flattened into owned citations
/// so a divergence can outlive the borrow of the documents it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefinitionState {
    /// The policy defines the word exactly once.
    Defined(Provenance),

    /// The policy defines the word **more than once, inconsistently**.
    ///
    /// A defect in the document, not in the reader. Worth surfacing loudly: a
    /// policy that cannot agree with itself about what "hospitalisation" means is
    /// a policy whose deductible clause has no settled subject.
    Conflicting(Vec<Provenance>),

    /// The policy does not define the word, so plain meaning applies there.
    Undefined,
}

impl DefinitionState {
    fn describe(&self) -> String {
        match self {
            Self::Defined(p) => format!("{p} — \"{}\"", p.verbatim()),
            Self::Conflicting(all) => format!(
                "DEFINES IT {} TIMES, INCONSISTENTLY: {}",
                all.len(),
                all.iter()
                    .map(|p| format!("{p} — \"{}\"", p.verbatim()))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Self::Undefined => "does not define this word at all (plain meaning applies)".into(),
        }
    }
}

/// A definition that the policies do not agree on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinitionDivergence {
    /// The word, e.g. `"hospitalisation"`.
    pub word: String,
    /// What each policy says it means, keyed by policy. A policy that does not
    /// define the word at all appears as [`DefinitionState::Undefined`] — which is
    /// its own kind of divergence, and arguably the worst one, because the reader
    /// of *that* policy has no idea a special meaning is in play elsewhere.
    pub definitions: BTreeMap<PolicyId, DefinitionState>,
}

impl DefinitionDivergence {
    /// A short account of the divergence, quoting each policy's own wording.
    pub fn describe(&self) -> String {
        let mut out = format!("the policies do not define \"{}\" the same way:\n", self.word);
        for (policy, state) in &self.definitions {
            out.push_str(&format!("  {policy}: {}\n", state.describe()));
        }
        out
    }
}

/// Whether the terms may honestly be set side by side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Comparability {
    /// Every definition the term rests on is worded identically across the
    /// policies (or the term rests on none).
    Comparable,

    /// At least one load-bearing definition diverges. The terms are still in
    /// the comparison — but they are **not** the same term, and a renderer must
    /// not present them as though they were.
    Incomparable {
        /// Every definition that diverged.
        divergences: Vec<DefinitionDivergence>,
    },
}

impl Comparability {
    /// Whether the terms may be presented as directly comparable.
    pub fn is_comparable(&self) -> bool {
        matches!(self, Self::Comparable)
    }
}

/// The same term, across several policies, with the source clauses attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    /// The term compared.
    pub kind: TermKind,
    /// One entry per policy, in the order supplied.
    pub entries: Vec<ComparisonEntry>,
    /// Whether the entries mean the same thing.
    pub comparability: Comparability,
}

impl Comparison {
    /// A full account: the verdict first, then every policy's clauses verbatim.
    ///
    /// The verdict comes **first** on purpose. A reader who stops after the
    /// table should already have been told not to trust the table.
    pub fn explain(&self) -> String {
        let mut out = String::new();

        match &self.comparability {
            Comparability::Comparable => {
                out.push_str(&format!(
                    "Comparing {}: the policies word the definitions this term rests on \
                     identically, so the clauses below are about the same thing.\n\n",
                    self.kind
                ));
            }
            Comparability::Incomparable { divergences } => {
                out.push_str(&format!(
                    "NOT COMPARABLE — the policies state {} but do not mean the same thing by \
                     it. Do not read the clauses below as alternatives to one another.\n\n",
                    self.kind
                ));
                for d in divergences {
                    out.push_str(&d.describe());
                    out.push('\n');
                }
            }
        }

        for entry in &self.entries {
            out.push_str(&format!("{} ({}):\n", entry.name, entry.policy));
            match &entry.position {
                PolicyPosition::NotStated => out.push_str(
                    "  NOT STATED — this crate found no such clause. That is not the same as \
                     'nil', and it is not the same as 'excluded'. It most likely means the \
                     figure is in a benefit schedule this scaffold did not read.\n",
                ),
                PolicyPosition::Stated(terms) => {
                    for term in terms {
                        out.push_str(&format!("  {term}\n"));
                    }
                }
            }
        }

        out
    }
}

/// Lines the same term up across policies, refusing to call it comparable when
/// the policies do not mean the same thing by it.
///
/// See the module docs. Note that this never *hides* a term: an incomparable
/// comparison still carries every clause. It withholds only the claim that they
/// are equivalent — which was never this crate's to make.
pub fn compare(policies: &[&PolicyLayer], kind: TermKind) -> Comparison {
    let entries = policies
        .iter()
        .map(|p| {
            let terms: Vec<PolicyTerm> = p.terms_of_kind(kind).cloned().collect();
            ComparisonEntry {
                policy: p.id().clone(),
                name: p.name().to_string(),
                position: if terms.is_empty() {
                    PolicyPosition::NotStated
                } else {
                    PolicyPosition::Stated(terms)
                },
            }
        })
        .collect();

    Comparison {
        kind,
        entries,
        comparability: assess_comparability(policies, kind),
    }
}

/// Checks every definition the term rests on for divergence across the policies.
///
/// Definitions are resolved through `kopitiam-insurance`'s [`Resolution`], so a
/// policy that defines the same word twice, inconsistently, is caught here too —
/// and is *always* a divergence, even against an identically-worded second
/// policy, because a self-contradicting definition has no single meaning to
/// compare against.
fn assess_comparability(policies: &[&PolicyLayer], kind: TermKind) -> Comparability {
    // A single policy is trivially self-consistent with itself, and a term that
    // rests on no definitions cannot diverge.
    if policies.len() < 2 {
        return Comparability::Comparable;
    }

    let mut divergences = Vec::new();

    for word in kind.depends_on_definitions() {
        let mut definitions: BTreeMap<PolicyId, DefinitionState> = BTreeMap::new();
        // The comparison key: the normalised meaning, or `None` for undefined.
        // A self-conflicting policy gets a key nothing can equal.
        let mut keys: Vec<Option<String>> = Vec::new();
        let mut any_conflict = false;

        for policy in policies {
            let (state, key) = match policy.meaning_of(word) {
                Resolution::Defined(d) => (
                    DefinitionState::Defined(d.provenance().clone()),
                    Some(normalise(d.meaning())),
                ),
                Resolution::Conflicting(all) => {
                    any_conflict = true;
                    (
                        DefinitionState::Conflicting(
                            all.iter().map(|d| d.provenance().clone()).collect(),
                        ),
                        None,
                    )
                }
                Resolution::Undefined => (DefinitionState::Undefined, None),
            };
            definitions.insert(policy.id().clone(), state);
            keys.push(key);
        }

        // A word that *no* policy defines is not a divergence. It is simply a word
        // none of them thought worth defining, and comparing on it is no more
        // dangerous than on any other ordinary English word.
        let all_undefined = definitions
            .values()
            .all(|s| matches!(s, DefinitionState::Undefined));
        if all_undefined {
            continue;
        }

        // Any difference at all — a different wording, or one policy defining it
        // and another not, or one policy contradicting itself — is a divergence.
        // See the module docs on why we err in this direction.
        let first = &keys[0];
        if any_conflict || keys.iter().any(|k| k != first) {
            divergences.push(DefinitionDivergence {
                word: (*word).to_string(),
                definitions,
            });
        }
    }

    if divergences.is_empty() {
        Comparability::Comparable
    } else {
        Comparability::Incomparable { divergences }
    }
}

/// Case-folds and collapses whitespace, so that a definition re-flowed
/// differently by the PDF extractor is not mistaken for a different definition.
///
/// This is the *only* normalisation applied. In particular no synonym matching,
/// no stemming, no stop-word removal: those would start deciding that two
/// wordings mean the same thing, which is precisely the judgment this module
/// refuses to make.
fn normalise(text: &str) -> String {
    text.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_survives_reflow_but_not_rewording() {
        assert_eq!(
            normalise("admission   to a\n  Hospital"),
            normalise("Admission to a hospital")
        );
        assert_ne!(
            normalise("admission to a hospital for at least one night"),
            normalise("admission to a hospital, including day surgery")
        );
    }
}
