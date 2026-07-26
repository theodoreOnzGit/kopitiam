use super::{session_name, Session};
use crate::core::PaneGeometry;
use crate::proto::{RmuxError, RotateWindowDirection, TerminalSize};

#[path = "window_tests/navigation.rs"]
mod navigation;

#[path = "window_tests/move_swap_reindex.rs"]
mod move_swap_reindex;

#[path = "window_tests/rotate_respawn.rs"]
mod rotate_respawn;
