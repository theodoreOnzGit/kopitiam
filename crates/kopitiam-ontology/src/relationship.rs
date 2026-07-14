use serde::{Deserialize, Serialize};

use crate::{EntityId, RelationshipId};

/// The kind of edge between two entities in the semantic graph.
///
/// These are the example relationships named in the Semantic Runtime vision
/// (`Function -implemented_by-> Rust Module`, `Function -documented_in-> PDF
/// Section`, etc.), generalized across providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipKind {
    ImplementedBy,
    DocumentedIn,
    TestedBy,
    ModifiedBy,
    DependsOn,
    LocatedIn,

    /// `from` **inherits from** `to` — derived type points at its base type.
    ///
    /// # Why this exists as a first-class variant
    ///
    /// It was added after four language adapters (Python, C#, C++, Visual
    /// Basic) were written concurrently and **each reached for a different
    /// encoding of the same idea**: two settled on `Custom("inherits")`, one on
    /// `ImplementedBy`, and one nearly flattened it into [`Self::DependsOn`].
    ///
    /// That divergence is precisely the failure the shared vocabulary exists to
    /// prevent. The entire point of every adapter emitting `kopitiam-ontology`
    /// is that a C++ base class and a Python base class become *the same shape*
    /// of fact, so the knowledge graph and the Translation Platform can reason
    /// across languages without knowing which one a fact came from. Four
    /// adapters encoding inheritance four ways means the graph cannot answer
    /// "what derives from this?" — and inheritance is not incidental to
    /// translation, it is central to it (a large object-oriented C++ codebase
    /// is typically deeply hierarchical).
    ///
    /// Collapsing it into [`Self::DependsOn`] would also lose real information:
    /// "is-a" and "uses" are different relations, and a translation that cannot
    /// tell them apart cannot preserve the type hierarchy.
    ///
    /// # Direction
    ///
    /// **Derived → base.** `Relationship::new(derived, base, Inherits)` reads
    /// "derived inherits base". This is the opposite direction from
    /// [`Self::ImplementedBy`] (which reads interface → implementor), so the two
    /// are not interchangeable — do not reach for `ImplementedBy` to express a
    /// base class.
    Inherits,

    /// `from` **cites** `to` — a work pointing at a work it references.
    ///
    /// The citation graph is one of the most valuable structures KOPITIAM can
    /// hold: it is the record of what a field knows and who established it, and
    /// it is what makes "show me everything downstream of this result"
    /// answerable at all.
    ///
    /// # Why this is first-class rather than `Custom("cites")`
    ///
    /// Because [`Self::Inherits`]' own rustdoc — written after four language
    /// adapters each invented a different encoding for inheritance — argued
    /// exactly this, and then `kopitiam-bibliography` reached for
    /// `Custom("cites")` anyway. The author flagged it themselves, under
    /// protest, which is the system working: they could not edit this file, so
    /// they said so instead of quietly diverging.
    ///
    /// A `Custom` string is unenforceable. Nothing stops the next crate from
    /// emitting `Custom("citation")` or `Custom("references")`, at which point
    /// the graph can no longer answer the one question it exists to answer.
    /// **If a relation is real, name it.** `Custom` is for the genuinely
    /// bespoke, not for concepts we merely have not gotten round to.
    ///
    /// # Direction
    ///
    /// **Citing → cited.** `Relationship::new(citing, cited, Cites)` reads "this
    /// paper cites that one". A citation is a *claim about provenance*, and
    /// getting its direction backwards inverts credit — which is not a cosmetic
    /// error in an academic context.
    Cites,

    Custom(String),
}

/// A directed edge between two entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Relationship {
    pub id: RelationshipId,
    pub from: EntityId,
    pub to: EntityId,
    pub kind: RelationshipKind,
}

impl Relationship {
    pub fn new(from: EntityId, to: EntityId, kind: RelationshipKind) -> Self {
        Self {
            id: RelationshipId::new(),
            from,
            to,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_relationship() {
        let a = EntityId::new();
        let b = EntityId::new();
        let rel = Relationship::new(a, b, RelationshipKind::DependsOn);
        assert_eq!(rel.from, a);
        assert_eq!(rel.to, b);
        assert_eq!(rel.kind, RelationshipKind::DependsOn);
    }
}
