//! A short, ordered list of marked files — the native replacement for
//! ThePrimeagen/harpoon (v1 API: `mark.add_file`, `ui.toggle_quick_menu`,
//! `ui.nav_file`).
//!
//! # Why persistence is per-project
//!
//! Harpoon's entire value proposition is "four keystrokes back to the file I
//! was just in", and that only holds if the marks are still there after
//! restarting the editor. But marks are inherently project-scoped — the
//! four files you're bouncing between while working on `kopitiam-neovim`
//! have nothing to do with the four you'd mark in an unrelated project — so
//! a single global mark list would mean every project pollutes every other
//! project's harpoon menu. [`Harpoon::load`] therefore keys the persisted
//! store by the canonicalized working directory, matching upstream
//! harpoon's own per-project behaviour.
//!
//! # Why the store path isn't read from `config.rs`
//!
//! [`crate::config::Config::config_path`] resolves `~/.config/kvim/`
//! (`$XDG_CONFIG_HOME` falling back to `$HOME/.config`) and is being
//! actively developed alongside this module. [`harpoon_store_path`]
//! deliberately re-derives the same three-line convention locally rather
//! than calling into `config.rs`, so this module keeps compiling and
//! passing its own tests independent of what shape `config.rs` is in on any
//! given commit. Yes, this duplicates three lines; that's cheaper than a
//! cross-module dependency between two files owned by different concurrent
//! agents. If the two ever drift, that's a five-minute fix, not an
//! architecture problem.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// `~/.config/kvim/harpoon.json`, or `$XDG_CONFIG_HOME/kvim/harpoon.json`.
/// Mirrors [`crate::config::Config::config_path`]'s resolution order — see
/// the module docs for why it's a copy rather than a call.
pub fn harpoon_store_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("kvim").join("harpoon.json"))
}

/// The on-disk shape: every project's marks, keyed by that project's
/// canonicalized working directory. One file for the whole machine rather
/// than one file per project, so there's a single place to look (and a
/// single file to `.gitignore` from a dotfiles repo) — the per-project
/// scoping lives in the map's keys, not in the filesystem layout.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    projects: BTreeMap<String, Vec<PathBuf>>,
}

/// An ordered list of marked files for one project (one working directory).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Harpoon {
    project_key: String,
    marks: Vec<PathBuf>,
}

impl Harpoon {
    /// Loads the marks for the project rooted at `cwd` from
    /// [`harpoon_store_path`], or starts empty if there is no store file yet
    /// (a missing store is normal on first use, not an error — same
    /// reasoning as [`crate::config::Config::load`]).
    pub fn load(cwd: &Path) -> anyhow::Result<Self> {
        match harpoon_store_path() {
            Some(path) => Self::load_from(&path, cwd),
            // No resolvable config directory (e.g. neither $XDG_CONFIG_HOME
            // nor $HOME is set) — degrade to an in-memory, unpersisted
            // session rather than failing to open at all.
            None => Ok(Self::empty(cwd)),
        }
    }

    /// [`Harpoon::load`] against an explicit store file, for tests (and
    /// anything else that wants to override the default location).
    pub fn load_from(store_path: &Path, cwd: &Path) -> anyhow::Result<Self> {
        let project_key = project_key(cwd);
        if !store_path.exists() {
            return Ok(Self { project_key, marks: Vec::new() });
        }
        let text = std::fs::read_to_string(store_path)?;
        let store: Store = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("malformed harpoon store at {}: {e}", store_path.display()))?;
        let marks = store.projects.get(&project_key).cloned().unwrap_or_default();
        Ok(Self { project_key, marks })
    }

    fn empty(cwd: &Path) -> Self {
        Self { project_key: project_key(cwd), marks: Vec::new() }
    }

    /// Persists this project's marks to [`harpoon_store_path`], merging with
    /// (rather than clobbering) whatever other projects are already in the
    /// store — two projects share one file, so a save from one must not
    /// erase the other's marks.
    pub fn save(&self) -> anyhow::Result<()> {
        let Some(path) = harpoon_store_path() else {
            anyhow::bail!("no resolvable config directory ($XDG_CONFIG_HOME or $HOME); nothing to persist marks to");
        };
        self.save_to(&path)
    }

    /// [`Harpoon::save`] against an explicit store file, for tests.
    pub fn save_to(&self, store_path: &Path) -> anyhow::Result<()> {
        let mut store: Store = if store_path.exists() {
            let text = std::fs::read_to_string(store_path)?;
            serde_json::from_str(&text).unwrap_or_default()
        } else {
            Store::default()
        };
        store.projects.insert(self.project_key.clone(), self.marks.clone());
        if let Some(parent) = store_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(store_path, serde_json::to_string_pretty(&store)?)?;
        Ok(())
    }

    /// Marks `path` — the `<leader>b` keymap. A no-op (not a duplicate
    /// entry) if the file is already marked, matching upstream harpoon's
    /// `mark.add_file`. Returns whether a new mark was actually added.
    pub fn add_file(&mut self, path: PathBuf) -> bool {
        if self.marks.contains(&path) {
            return false;
        }
        self.marks.push(path);
        true
    }

    /// Removes the mark at `index`, if any.
    pub fn remove(&mut self, index: usize) -> Option<PathBuf> {
        (index < self.marks.len()).then(|| self.marks.remove(index))
    }

    /// Moves the mark at `from` to `to`, shifting the marks between them —
    /// how the quick-menu buffer's line reordering (editing the harpoon
    /// popup like a normal buffer and saving it) is implemented upstream.
    pub fn reorder(&mut self, from: usize, to: usize) {
        if from >= self.marks.len() || to >= self.marks.len() || from == to {
            return;
        }
        let item = self.marks.remove(from);
        self.marks.insert(to, item);
    }

    /// The file at 1-based slot `n` — `nav_file(1)` through `nav_file(9)` in
    /// upstream harpoon's default keymaps (`<leader>1`..`<leader>9`).
    pub fn nav_file(&self, n: usize) -> Option<&PathBuf> {
        n.checked_sub(1).and_then(|i| self.marks.get(i))
    }

    /// All marks, in order — what both the quick-menu (`<leader><Esc>`) and
    /// the `\|leader|q` picker display. Whether that display takes the
    /// shape of a popup buffer or a [`crate::plugins::picker::Picker`] is a
    /// UI decision; this module only owns the ordered data.
    pub fn marks(&self) -> &[PathBuf] {
        &self.marks
    }

    /// The project key marks are stored under (the canonicalized `cwd` this
    /// instance was loaded for).
    pub fn project_key(&self) -> &str {
        &self.project_key
    }
}

fn project_key(cwd: &Path) -> String {
    cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf()).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove_and_reorder() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());

        assert!(h.add_file(PathBuf::from("a.rs")));
        assert!(h.add_file(PathBuf::from("b.rs")));
        assert!(h.add_file(PathBuf::from("c.rs")));
        assert_eq!(h.marks(), &[PathBuf::from("a.rs"), PathBuf::from("b.rs"), PathBuf::from("c.rs")]);

        // Adding an already-marked file is a no-op, not a duplicate.
        assert!(!h.add_file(PathBuf::from("b.rs")));
        assert_eq!(h.marks().len(), 3);

        h.reorder(0, 2);
        assert_eq!(h.marks(), &[PathBuf::from("b.rs"), PathBuf::from("c.rs"), PathBuf::from("a.rs")]);

        assert_eq!(h.remove(1), Some(PathBuf::from("c.rs")));
        assert_eq!(h.marks(), &[PathBuf::from("b.rs"), PathBuf::from("a.rs")]);
    }

    #[test]
    fn nav_file_is_one_based() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());
        h.add_file(PathBuf::from("first.rs"));
        h.add_file(PathBuf::from("second.rs"));

        assert_eq!(h.nav_file(1), Some(&PathBuf::from("first.rs")));
        assert_eq!(h.nav_file(2), Some(&PathBuf::from("second.rs")));
        assert_eq!(h.nav_file(0), None, "harpoon slots are 1-based; 0 is not a valid slot");
        assert_eq!(h.nav_file(3), None);
    }

    #[test]
    fn persists_and_reloads_from_a_tempdir() {
        let store = tempfile::tempdir().unwrap();
        let store_path = store.path().join("harpoon.json");
        let project = tempfile::tempdir().unwrap();

        let mut h = Harpoon::load_from(&store_path, project.path()).unwrap();
        assert!(h.marks().is_empty(), "a fresh store has no marks yet");
        h.add_file(PathBuf::from("main.rs"));
        h.add_file(PathBuf::from("lib.rs"));
        h.save_to(&store_path).unwrap();

        let reloaded = Harpoon::load_from(&store_path, project.path()).unwrap();
        assert_eq!(reloaded.marks(), &[PathBuf::from("main.rs"), PathBuf::from("lib.rs")]);
    }

    #[test]
    fn two_projects_do_not_share_marks() {
        let store = tempfile::tempdir().unwrap();
        let store_path = store.path().join("harpoon.json");
        let project_a = tempfile::tempdir().unwrap();
        let project_b = tempfile::tempdir().unwrap();

        let mut a = Harpoon::load_from(&store_path, project_a.path()).unwrap();
        a.add_file(PathBuf::from("a_only.rs"));
        a.save_to(&store_path).unwrap();

        let mut b = Harpoon::load_from(&store_path, project_b.path()).unwrap();
        assert!(b.marks().is_empty(), "project B must not see project A's marks");
        b.add_file(PathBuf::from("b_only.rs"));
        b.save_to(&store_path).unwrap();

        // Saving B must not have clobbered A's entry in the shared store file.
        let a_again = Harpoon::load_from(&store_path, project_a.path()).unwrap();
        assert_eq!(a_again.marks(), &[PathBuf::from("a_only.rs")]);
    }

    #[test]
    fn removing_an_out_of_range_index_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());
        h.add_file(PathBuf::from("only.rs"));
        assert_eq!(h.remove(5), None);
        assert_eq!(h.marks().len(), 1);
    }
}
