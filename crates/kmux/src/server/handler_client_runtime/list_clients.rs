use crate::os::identity::UserIdentity;

#[derive(Debug, Clone)]
pub(in crate::server::handler) struct ListClientSnapshot {
    pub(in crate::server::handler) name: String,
    pub(in crate::server::handler) pid: u32,
    pub(in crate::server::handler) tty: String,
    pub(in crate::server::handler) control: bool,
    pub(in crate::server::handler) session_name: Option<crate::proto::SessionName>,
    pub(in crate::server::handler) order: u64,
    pub(in crate::server::handler) width: u16,
    pub(in crate::server::handler) height: u16,
    pub(in crate::server::handler) termname: String,
    pub(in crate::server::handler) termtype: String,
    pub(in crate::server::handler) termfeatures: String,
    pub(in crate::server::handler) utf8: bool,
    pub(in crate::server::handler) key_table: Option<String>,
    pub(in crate::server::handler) uid: u32,
    pub(in crate::server::handler) user: UserIdentity,
    pub(in crate::server::handler) flags: String,
}

impl ListClientSnapshot {
    pub(in crate::server::handler) fn key_table_name(&self) -> &str {
        self.key_table.as_deref().unwrap_or("root")
    }

    pub(in crate::server::handler) fn prefix_value(&self) -> &'static str {
        if self.key_table.as_deref() == Some("prefix") {
            "1"
        } else {
            "0"
        }
    }
}
