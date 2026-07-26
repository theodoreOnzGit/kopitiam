//! Identity vocabulary surface used by `rmux-core` consumers.
//!
//! The canonical identity newtypes (`SessionName`, `SessionId`,
//! `WindowId`, `PaneId`) are defined exactly once in `rmux-proto`. This
//! module re-exports those types so `rmux-core` consumers can address
//! them through a single core-side surface regardless of which crate
//! originally introduced the value. Allocation, lookup, and resolution
//! remain in `rmux-::core::session`; nothing in this module mutates
//! identity state.

pub use crate::proto::{PaneId, SessionId, SessionName, WindowId};

#[cfg(test)]
mod tests {
    use super::{PaneId, SessionId, SessionName, WindowId};

    #[test]
    fn core_identity_re_export_matches_proto_definition() {
        let proto_pane: crate::proto::PaneId = PaneId::new(7);
        assert_eq!(proto_pane.as_u32(), 7);
        assert_eq!(proto_pane.to_string(), "%7");

        let proto_window: crate::proto::WindowId = WindowId::new(3);
        assert_eq!(proto_window.to_string(), "@3");

        let proto_session: crate::proto::SessionId = SessionId::new(2);
        assert_eq!(proto_session.to_string(), "$2");

        let name: crate::proto::SessionName = SessionName::new("alpha").expect("valid");
        assert_eq!(name.as_str(), "alpha");
    }

    #[test]
    fn core_identity_re_exports_match_pane_module_re_export() {
        assert_eq!(
            std::any::TypeId::of::<crate::core::PaneId>(),
            std::any::TypeId::of::<PaneId>(),
            "::core::PaneId from pane.rs and ::core::identity::PaneId must converge to one type",
        );
        assert_eq!(
            std::any::TypeId::of::<crate::core::WindowId>(),
            std::any::TypeId::of::<WindowId>(),
            "::core::WindowId and ::core::identity::WindowId must converge to one type",
        );
        assert_eq!(
            std::any::TypeId::of::<crate::core::SessionId>(),
            std::any::TypeId::of::<SessionId>(),
            "::core::SessionId and ::core::identity::SessionId must converge to one type",
        );
        assert_eq!(
            std::any::TypeId::of::<crate::proto::PaneId>(),
            std::any::TypeId::of::<crate::core::PaneId>(),
            "core re-exports must resolve to crate::proto::PaneId",
        );
        assert_eq!(
            std::any::TypeId::of::<crate::proto::WindowId>(),
            std::any::TypeId::of::<crate::core::WindowId>(),
            "core re-exports must resolve to crate::proto::WindowId",
        );
        assert_eq!(
            std::any::TypeId::of::<crate::proto::SessionId>(),
            std::any::TypeId::of::<crate::core::SessionId>(),
            "core re-exports must resolve to crate::proto::SessionId",
        );
        assert_eq!(
            std::any::TypeId::of::<crate::proto::SessionName>(),
            std::any::TypeId::of::<crate::core::SessionName>(),
            "core re-exports must resolve to crate::proto::SessionName",
        );
    }
}
