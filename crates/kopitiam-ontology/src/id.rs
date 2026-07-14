use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Identifies an [`crate::Entity`] within the semantic graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(Uuid);

/// Identifies a [`crate::Relationship`] within the semantic graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationshipId(Uuid);

macro_rules! id_impl {
    ($name:ident) => {
        impl $name {
            /// Generates a new, randomly assigned id.
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl From<Uuid> for $name {
            fn from(uuid: Uuid) -> Self {
                Self(uuid)
            }
        }

        impl From<$name> for Uuid {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

id_impl!(EntityId);
id_impl!(RelationshipId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_distinct_types() {
        let a = EntityId::new();
        let b = EntityId::new();
        assert_ne!(a, b);

        let r = RelationshipId::new();
        assert_ne!(a.to_string(), "");
        assert_ne!(r.to_string(), "");
    }
}
