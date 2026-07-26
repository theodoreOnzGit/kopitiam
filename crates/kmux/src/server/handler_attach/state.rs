use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::Arc;
use std::time::Instant;

use crate::core::KeyCode;
use crate::os::identity::UserIdentity;
use crate::proto::{PaneTarget, TerminalPixels, TerminalSize, WindowTarget};
use tokio::sync::mpsc;

use super::super::mode_tree_support::ModeTreeClientState;
use super::super::overlay_support::ClientOverlayState;
use super::super::prompt_support::ClientPromptState;
use crate::server::client_flags::ClientFlags;
use crate::server::handler_support::{ambiguous_attached_client, attached_client_required};
use crate::server::mouse::ClientMouseState;
use crate::server::outer_terminal::OuterTerminalContext;
use crate::server::pane_io::AttachControl;

#[derive(Debug, Default)]
pub(in crate::server::handler) struct ActiveAttachState {
    pub(in crate::server::handler) next_id: u64,
    pub(in crate::server::handler) next_size_sequence: u64,
    pub(in crate::server::handler) by_pid: HashMap<u32, ActiveAttach>,
}

#[derive(Debug)]
pub(in crate::server::handler) struct ActiveAttach {
    pub(in crate::server::handler) id: u64,
    pub(in crate::server::handler) session_name: crate::proto::SessionName,
    pub(in crate::server::handler) last_session: Option<crate::proto::SessionName>,
    pub(in crate::server::handler) flags: ClientFlags,
    pub(in crate::server::handler) pan_window: Option<u32>,
    pub(in crate::server::handler) pan_ox: u32,
    pub(in crate::server::handler) pan_oy: u32,
    pub(in crate::server::handler) control_tx: mpsc::UnboundedSender<AttachControl>,
    pub(in crate::server::handler) control_backlog: Arc<AtomicUsize>,
    pub(in crate::server::handler) render_stream: bool,
    pub(in crate::server::handler) render_refresh_pending: bool,
    pub(in crate::server::handler) uid: u32,
    pub(in crate::server::handler) user: UserIdentity,
    pub(in crate::server::handler) can_write: bool,
    pub(in crate::server::handler) suspended: bool,
    pub(in crate::server::handler) closing: Arc<AtomicBool>,
    pub(in crate::server::handler) terminal_context: OuterTerminalContext,
    pub(in crate::server::handler) client_size: TerminalSize,
    pub(in crate::server::handler) client_pixels: Option<TerminalPixels>,
    pub(in crate::server::handler) size_sequence: u64,
    pub(in crate::server::handler) persistent_overlay_epoch: Arc<AtomicU64>,
    pub(in crate::server::handler) render_generation: u64,
    pub(in crate::server::handler) overlay_generation: u64,
    pub(in crate::server::handler) overlay_state_id: u64,
    pub(in crate::server::handler) display_panes_state_id: u64,
    pub(in crate::server::handler) key_table_name: Option<String>,
    pub(in crate::server::handler) key_table_set_at: Option<Instant>,
    pub(in crate::server::handler) repeat_deadline: Option<Instant>,
    pub(in crate::server::handler) repeat_active: bool,
    pub(in crate::server::handler) last_key: Option<KeyCode>,
    pub(in crate::server::handler) mouse: ClientMouseState,
    pub(in crate::server::handler) prompt: Option<ClientPromptState>,
    pub(in crate::server::handler) mode_tree_state_id: u64,
    pub(in crate::server::handler) mode_tree: Option<ModeTreeClientState>,
    pub(in crate::server::handler) mode_tree_frame: Option<Vec<u8>>,
    pub(in crate::server::handler) overlay: Option<ClientOverlayState>,
    pub(in crate::server::handler) display_panes: Option<DisplayPanesClientState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::server::handler) struct DisplayPanesClientState {
    pub(in crate::server::handler) id: u64,
    pub(in crate::server::handler) window: WindowTarget,
    pub(in crate::server::handler) labels: Vec<DisplayPanesLabel>,
    pub(in crate::server::handler) input: String,
    pub(in crate::server::handler) template: Option<String>,
    pub(in crate::server::handler) clear_frame: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::server::handler) struct DisplayPanesLabel {
    pub(in crate::server::handler) label: String,
    pub(in crate::server::handler) target: PaneTarget,
    pub(in crate::server::handler) target_string: String,
}

#[derive(Debug)]
pub(crate) struct AttachRegistration {
    pub(crate) control_tx: mpsc::UnboundedSender<AttachControl>,
    pub(crate) control_backlog: Arc<AtomicUsize>,
    pub(crate) closing: Arc<AtomicBool>,
    pub(crate) persistent_overlay_epoch: Arc<AtomicU64>,
    pub(crate) terminal_context: OuterTerminalContext,
    pub(crate) flags: ClientFlags,
    pub(crate) render_stream: bool,
    pub(crate) uid: u32,
    pub(crate) user: UserIdentity,
    pub(crate) can_write: bool,
    pub(crate) client_size: Option<TerminalSize>,
}

impl ActiveAttachState {
    pub(in crate::server::handler) fn attached_count(
        &self,
        session_name: &crate::proto::SessionName,
    ) -> usize {
        self.by_pid
            .values()
            .filter(|active| &active.session_name == session_name && !active.suspended)
            .count()
    }

    pub(in crate::server::handler) fn rename_session(
        &mut self,
        session_name: &crate::proto::SessionName,
        new_name: &crate::proto::SessionName,
    ) {
        for active in self.by_pid.values_mut() {
            if &active.session_name == session_name {
                active.session_name = new_name.clone();
            }
            if active.last_session.as_ref() == Some(session_name) {
                active.last_session = Some(new_name.clone());
            }
        }
    }

    pub(in crate::server::handler) fn toggle_read_only(
        &mut self,
        attach_pid: u32,
    ) -> Result<ClientFlags, crate::proto::RmuxError> {
        let active = self.by_pid.get_mut(&attach_pid).ok_or_else(|| {
            crate::proto::RmuxError::Server("attached client disappeared".to_owned())
        })?;
        active.flags.toggle_read_only();
        Ok(active.flags)
    }

    pub(in crate::server::handler) fn last_session_for_client(
        &self,
        attach_pid: u32,
    ) -> Result<Option<crate::proto::SessionName>, crate::proto::RmuxError> {
        self.by_pid
            .get(&attach_pid)
            .map(|active| active.last_session.clone())
            .ok_or_else(|| crate::proto::RmuxError::Server("attached client disappeared".to_owned()))
    }

    pub(in crate::server::handler) fn attached_client_pids_for_session(
        &self,
        session_name: &crate::proto::SessionName,
        except_pid: Option<u32>,
    ) -> Vec<u32> {
        let mut pids = self
            .by_pid
            .iter()
            .filter_map(|(pid, active)| {
                (&active.session_name == session_name && except_pid != Some(*pid)).then_some(*pid)
            })
            .collect::<Vec<_>>();
        pids.sort_unstable();
        pids
    }

    pub(in crate::server::handler) fn attached_client_pids_except(&self, except_pid: u32) -> Vec<u32> {
        let mut pids = self
            .by_pid
            .keys()
            .copied()
            .filter(|pid| *pid != except_pid)
            .collect::<Vec<_>>();
        pids.sort_unstable();
        pids
    }

    pub(in crate::server::handler) fn session_for_attached_client(
        &self,
        requester_pid: u32,
        command_name: &str,
    ) -> Result<Option<crate::proto::SessionName>, crate::proto::RmuxError> {
        if self.by_pid.is_empty() {
            return Ok(None);
        }

        let attach_pid = self.resolve_attached_client_pid(requester_pid, command_name)?;
        Ok(self
            .by_pid
            .get(&attach_pid)
            .map(|active| active.session_name.clone()))
    }

    pub(in crate::server::handler) fn current_session_candidate(
        &self,
        requester_pid: u32,
    ) -> Option<crate::proto::SessionName> {
        if let Some(active) = self.by_pid.get(&requester_pid) {
            return Some(active.session_name.clone());
        }

        if self.by_pid.len() == 1 {
            return self
                .by_pid
                .values()
                .next()
                .map(|active| active.session_name.clone());
        }

        None
    }

    pub(in crate::server::handler) fn resolve_attached_client_pid(
        &self,
        requester_pid: u32,
        command_name: &str,
    ) -> Result<u32, crate::proto::RmuxError> {
        if self.by_pid.contains_key(&requester_pid) {
            return Ok(requester_pid);
        }

        match self.by_pid.len() {
            0 => Err(attached_client_required(command_name)),
            1 => Ok(*self
                .by_pid
                .keys()
                .next()
                .expect("single-entry attach map must have one key")),
            _ => Err(ambiguous_attached_client(command_name)),
        }
    }
}
