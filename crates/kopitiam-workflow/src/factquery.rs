//! The fact-query seam — the determinism boundary (temp_ai_design.md §3), and
//! an **open design question** (§8.1) still being settled.
//!
//! # The invariant this seam exists to enforce — LLM proposes, Rust disposes
//!
//! > **A fact must always come out of a Rust-executed deterministic provider,
//! > never out of the model's mouth.** The model may *request* a fact — e.g.
//! > "give me the return type of `select_adapter`" — but a deterministic
//! > provider (`kopitiam-semantic`, `kopitiam-knowledge`, `kopitiam-search`)
//! > computes the answer and Rust hands the model the real value.
//!
//! - LLM *asks*, Rust *executes* the query, Rust feeds the result back → ✅
//! - LLM *produces* the fact and you trust it → ❌ hallucination with extra steps
//!
//! That is how "the runtime owns understanding; models borrow it" stays true
//! even while a model is driving the reasoning. This module is the **one typed
//! seam** the model proposes against — it does not get to poke ten different
//! crates directly. One clean "ask the runtime" API, Rust disposes behind it.
//!
//! # STATUS: scaffold only — the real vocabulary is undecided (bead kopitiam-6ud)
//!
//! temp_ai_design.md §8.1 names this "the seam everything in §3–§4 hangs on —
//! design it first", and it is **not yet decided**. The types below fix the
//! *shape* of the seam (a typed request enum, a disposed answer, a trait the
//! runtime implements) so the rest of `kopitiam-workflow` can compile and be
//! reasoned about — but the actual set of query variants is deliberately tiny
//! and marked `TODO(decide-bead)`. Do **not** grow this enum casually: every
//! variant is a commitment the whole runtime has to honour. Widen it only once
//! bead `kopitiam-6ud` settles the vocabulary.

use kopitiam_ontology::Entity;

/// A typed fact request the model proposes against the runtime.
///
/// This is the model's half of "LLM proposes, Rust disposes": the model emits
/// one of these; it never emits the *answer*. The runtime ([`FactOracle`])
/// disposes it into a [`FactAnswer`].
///
/// The variants here are **placeholders to fix the seam's shape**, not the
/// decided query language. Each real decision below is a `TODO(decide-bead)`
/// against bead `kopitiam-6ud`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive] // the real vocabulary is unresolved — callers must not assume this list is final
pub enum FactQuery {
    /// "What does the runtime know about the symbol named `name`?" — resolve a
    /// symbol to its [`Entity`].
    ///
    /// TODO(decide-bead kopitiam-6ud): is symbol identity a bare name, or a
    /// fully-qualified path + crate + disambiguator? A bare name is ambiguous
    /// across a workspace; the real key almost certainly is not a `String`.
    SymbolByName {
        /// The symbol's name as the model referred to it.
        name: String,
    },

    /// "What is the return type of the symbol named `name`?" — the worked
    /// example from §3 ("give me the return type of `select_adapter`").
    ///
    /// TODO(decide-bead kopitiam-6ud): the answer type. A `String` type-name is
    /// lossy; the real answer may need a structured type reference (generics,
    /// lifetimes, the defining crate). Left as an [`Entity`] for now so the
    /// seam does not over-commit.
    ReturnTypeOf {
        /// The symbol whose return type is wanted.
        name: String,
    },
    // TODO(decide-bead kopitiam-6ud): the full query vocabulary — callers,
    // implementors, references, doc lookups, section text, cross-language
    // facts, ... — is the real meat and is NOT decided here. Keep this list
    // minimal until it is.
}

/// The runtime's disposed answer to a [`FactQuery`].
///
/// Crucially there is a [`FactAnswer::Unknown`] arm: when the runtime has no
/// such fact, it says so **honestly** — the same `Indeterminate` discipline as
/// the dispatch ladder. It never fabricates an [`Entity`] to satisfy the model.
/// An honest `Unknown` handed back to the model is correct; a guessed fact is
/// the exact failure this whole seam exists to prevent.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FactAnswer {
    /// Rust disposed the query into a real, deterministically-derived entity.
    Resolved(Entity),
    /// The runtime genuinely has no such fact. Honest miss, never a guess.
    Unknown,
    // TODO(decide-bead kopitiam-6ud): richer answers — a set of entities, a
    // typed value, a "known but out of budget to compute" variant tying into
    // the resource budgeter (§5/§6). Undecided.
}

/// The runtime's fact-disposal interface — the "Rust disposes" half.
///
/// Whoever implements this (ultimately backed by `kopitiam-knowledge` /
/// `kopitiam-semantic` / `kopitiam-search`) is the **only** thing allowed to
/// turn a model's [`FactQuery`] into a [`FactAnswer`]. The model calls *through*
/// this trait; it never reaches the fact stores itself.
///
/// This scaffold defines the trait but ships no real backend — see
/// [`NullOracle`]. Wiring a real oracle over the knowledge crates is a
/// follow-up, gated on the vocabulary decision (bead `kopitiam-6ud`).
pub trait FactOracle {
    /// Dispose one query into an answer. Must return [`FactAnswer::Unknown`]
    /// rather than invent a fact it cannot deterministically derive.
    fn resolve(&self, query: &FactQuery) -> FactAnswer;
}

/// The scaffold oracle: knows nothing, so honestly answers
/// [`FactAnswer::Unknown`] to everything.
///
/// This is not a placeholder-that-lies — it is the *correct* behaviour of an
/// oracle with an empty backend: honest ignorance. It lets the seam compile,
/// be passed around, and be tested before any real fact store is wired in.
pub struct NullOracle;

impl FactOracle for NullOracle {
    fn resolve(&self, _query: &FactQuery) -> FactAnswer {
        FactAnswer::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_ontology::EntityKind;

    #[test]
    fn null_oracle_is_honestly_ignorant_never_bluffs() {
        let oracle = NullOracle;
        assert_eq!(oracle.resolve(&FactQuery::SymbolByName { name: "select_adapter".into() }), FactAnswer::Unknown);
        assert_eq!(oracle.resolve(&FactQuery::ReturnTypeOf { name: "select_adapter".into() }), FactAnswer::Unknown);
    }

    #[test]
    fn a_resolved_answer_carries_a_real_deterministic_entity() {
        // Demonstrates the "Rust disposes" side: a resolved answer is an
        // Entity carrying provenance (`source`), never a model-produced string.
        struct FixedOracle(Entity);
        impl FactOracle for FixedOracle {
            fn resolve(&self, query: &FactQuery) -> FactAnswer {
                match query {
                    FactQuery::SymbolByName { name } if name == &self.0.name => FactAnswer::Resolved(self.0.clone()),
                    _ => FactAnswer::Unknown,
                }
            }
        }

        let entity = Entity::new(EntityKind::Symbol, "select_adapter", "rust-analyzer");
        let oracle = FixedOracle(entity.clone());
        match oracle.resolve(&FactQuery::SymbolByName { name: "select_adapter".into() }) {
            FactAnswer::Resolved(e) => {
                assert_eq!(e, entity);
                assert_eq!(e.source, "rust-analyzer", "the fact carries deterministic provenance");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
        // An unknown symbol stays honestly Unknown.
        assert_eq!(oracle.resolve(&FactQuery::SymbolByName { name: "nope".into() }), FactAnswer::Unknown);
    }
}
