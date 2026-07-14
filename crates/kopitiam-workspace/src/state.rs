use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use kopitiam_index::Store;
use serde::{Deserialize, Serialize};

const STATE_KEY: &str = "workspace/project_state";

/// A project's session memory: what's being worked on right now, and what
/// was recently relevant, so a new session (or a different interface —
/// CLI, TUI, Android) can resume without re-deriving it.
///
/// This is deliberately small. It is not the semantic graph (that's
/// `kopitiam-knowledge`) and not a task tracker (KOPITIAM uses `bd` for
/// that) — it is just enough state that a `resume`-style command can say
/// "you were working on X, touching Y and Z" without asking a model to
/// guess from chat history that may no longer exist.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectState {
    /// A short human-written description of what's currently being worked
    /// on, set by whatever workflow is running (see `kopitiam-workflow`).
    pub current_task: Option<String>,

    /// Artifacts/symbols/documents touched recently, most recent last.
    /// Capped at [`WORKING_SET_CAPACITY`] entries so this stays a
    /// "working set", not an ever-growing log.
    pub working_set: Vec<String>,

    /// Unix timestamp (seconds) of the last change to this state.
    pub updated_at: Option<u64>,
}

/// Maximum number of entries kept in [`ProjectState::working_set`].
pub const WORKING_SET_CAPACITY: usize = 50;

impl ProjectState {
    /// Loads the project state for `root` from its `.kopitiam` directory,
    /// or returns a fresh, empty state if none has been saved yet.
    pub fn load(root: &Path) -> Result<Self> {
        let store = Store::open(root)?;
        Ok(store.get_json(STATE_KEY)?.unwrap_or_default())
    }

    /// Persists this state to `root`'s `.kopitiam` directory.
    pub fn save(&self, root: &Path) -> Result<()> {
        let store = Store::open(root)?;
        store.put_json(STATE_KEY, self)
    }

    /// Records `task` as the current focus and refreshes [`Self::updated_at`].
    pub fn set_current_task(&mut self, task: impl Into<String>) {
        self.current_task = Some(task.into());
        self.touch_timestamp();
    }

    /// Adds `entry` to the working set (moving it to the end if already
    /// present), evicting the oldest entry once
    /// [`WORKING_SET_CAPACITY`] is exceeded.
    pub fn touch(&mut self, entry: impl Into<String>) {
        let entry = entry.into();
        self.working_set.retain(|existing| existing != &entry);
        self.working_set.push(entry);
        while self.working_set.len() > WORKING_SET_CAPACITY {
            self.working_set.remove(0);
        }
        self.touch_timestamp();
    }

    fn touch_timestamp(&mut self) {
        self.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_a_fresh_default_state_when_nothing_was_saved() {
        let dir = tempfile::tempdir().unwrap();
        let state = ProjectState::load(dir.path()).unwrap();
        assert_eq!(state, ProjectState::default());
    }

    #[test]
    fn persists_across_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = ProjectState::load(dir.path()).unwrap();
        state.set_current_task("scaffold the Semantic Runtime");
        state.touch("kopitiam-workspace");
        state.touch("kopitiam-index");
        state.save(dir.path()).unwrap();

        let reloaded = ProjectState::load(dir.path()).unwrap();
        assert_eq!(reloaded.current_task.as_deref(), Some("scaffold the Semantic Runtime"));
        assert_eq!(reloaded.working_set, vec!["kopitiam-workspace", "kopitiam-index"]);
        assert!(reloaded.updated_at.is_some());
    }

    #[test]
    fn touching_an_existing_entry_moves_it_to_the_end_without_duplicating() {
        let mut state = ProjectState::default();
        state.touch("a");
        state.touch("b");
        state.touch("a");
        assert_eq!(state.working_set, vec!["b", "a"]);
    }

    #[test]
    fn working_set_is_capped() {
        let mut state = ProjectState::default();
        for i in 0..WORKING_SET_CAPACITY + 10 {
            state.touch(format!("entry-{i}"));
        }
        assert_eq!(state.working_set.len(), WORKING_SET_CAPACITY);
        assert_eq!(state.working_set.first().unwrap(), &format!("entry-{}", 10));
    }
}
