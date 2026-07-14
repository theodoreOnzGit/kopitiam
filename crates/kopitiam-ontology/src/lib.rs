//! Shared vocabulary for KOPITIAM's Semantic Runtime.
//!
//! This crate is pure data: entity and relationship types that every
//! knowledge provider (Rust, documents, future language adapters) and the
//! knowledge graph itself agree on. It has no storage and no logic, so any
//! part of the runtime can depend on it without pulling in providers,
//! persistence, or search.

mod entity;
mod id;
mod relationship;

pub use entity::{Entity, EntityKind};
pub use id::{EntityId, RelationshipId};
pub use relationship::{Relationship, RelationshipKind};
