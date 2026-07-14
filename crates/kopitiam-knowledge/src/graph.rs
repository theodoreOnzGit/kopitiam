use std::collections::HashMap;

use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipId};
use serde::{Deserialize, Serialize};

/// The unified knowledge graph: every entity and relationship produced by
/// any [`kopitiam_ontology`]-speaking provider, merged into one structure.
///
/// This type is storage-agnostic on purpose. It holds no file handles and
/// makes no assumptions about where it is persisted — it derives
/// `Serialize`/`Deserialize` so `kopitiam-index` (or anything else) can snapshot
/// it without `kopitiam-knowledge` depending on a storage engine.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SemanticGraph {
    entities: HashMap<EntityId, Entity>,
    relationships: HashMap<RelationshipId, Relationship>,
    #[serde(default)]
    outgoing: HashMap<EntityId, Vec<RelationshipId>>,
    #[serde(default)]
    incoming: HashMap<EntityId, Vec<RelationshipId>>,
}

impl SemanticGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts one entity, keyed by its own [`EntityId`]. Re-inserting an
    /// id already present overwrites the previous entity.
    pub fn insert_entity(&mut self, entity: Entity) -> EntityId {
        let id = entity.id;
        self.entities.insert(id, entity);
        id
    }

    /// Inserts one relationship and indexes it for traversal in both
    /// directions.
    pub fn insert_relationship(&mut self, relationship: Relationship) -> RelationshipId {
        let id = relationship.id;
        self.outgoing.entry(relationship.from).or_default().push(id);
        self.incoming.entry(relationship.to).or_default().push(id);
        self.relationships.insert(id, relationship);
        id
    }

    /// Merges a batch of facts from a provider run in one call — the usual
    /// entry point after calling a `kopitiam_semantic::KnowledgeProvider`.
    pub fn extend(
        &mut self,
        entities: impl IntoIterator<Item = Entity>,
        relationships: impl IntoIterator<Item = Relationship>,
    ) {
        for entity in entities {
            self.insert_entity(entity);
        }
        for relationship in relationships {
            self.insert_relationship(relationship);
        }
    }

    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    pub fn relationship(&self, id: RelationshipId) -> Option<&Relationship> {
        self.relationships.get(&id)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values()
    }

    pub fn entities_of_kind(&self, kind: EntityKind) -> impl Iterator<Item = &Entity> {
        self.entities.values().filter(move |e| e.kind == kind)
    }

    /// Relationships whose `from` is `id`.
    pub fn relationships_from(&self, id: EntityId) -> impl Iterator<Item = &Relationship> {
        self.outgoing
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(move |rel_id| self.relationships.get(rel_id))
    }

    /// Relationships whose `to` is `id`.
    pub fn relationships_to(&self, id: EntityId) -> impl Iterator<Item = &Relationship> {
        self.incoming
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(move |rel_id| self.relationships.get(rel_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_ontology::RelationshipKind;

    #[test]
    fn inserts_and_queries_entities() {
        let mut graph = SemanticGraph::new();
        let a = Entity::new(EntityKind::Artifact, "kopitiam-knowledge", "test");
        let b = Entity::new(EntityKind::Symbol, "SemanticGraph", "test");
        let (a_id, b_id) = (a.id, b.id);
        graph.extend([a, b], []);

        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.entities_of_kind(EntityKind::Symbol).count(), 1);
        assert!(graph.entity(a_id).is_some());
        assert!(graph.entity(b_id).is_some());
    }

    #[test]
    fn traverses_relationships_in_both_directions() {
        let mut graph = SemanticGraph::new();
        let a = Entity::new(EntityKind::Symbol, "SemanticGraph", "test");
        let b = Entity::new(EntityKind::Artifact, "kopitiam-knowledge", "test");
        let (a_id, b_id) = (a.id, b.id);
        graph.insert_entity(a);
        graph.insert_entity(b);
        graph.insert_relationship(Relationship::new(a_id, b_id, RelationshipKind::LocatedIn));

        let out: Vec<_> = graph.relationships_from(a_id).collect();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, RelationshipKind::LocatedIn);

        let inn: Vec<_> = graph.relationships_to(b_id).collect();
        assert_eq!(inn.len(), 1);
        assert_eq!(inn[0].from, a_id);
    }

    #[test]
    fn round_trips_through_json() {
        let mut graph = SemanticGraph::new();
        let a = Entity::new(EntityKind::Artifact, "a", "test");
        let b = Entity::new(EntityKind::Symbol, "b", "test");
        let (a_id, b_id) = (a.id, b.id);
        graph.extend([a, b], [Relationship::new(a_id, b_id, RelationshipKind::DependsOn)]);

        let json = serde_json::to_string(&graph).unwrap();
        let back: SemanticGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entity_count(), 2);
        assert_eq!(back.relationship_count(), 1);
        assert_eq!(back.relationships_from(a_id).count(), 1);
    }
}
