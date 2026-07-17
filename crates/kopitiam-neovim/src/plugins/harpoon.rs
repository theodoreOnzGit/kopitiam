//! A short, ordered list of marked files — the native replacement for
//! ThePrimeagen/harpoon (v1 API: `mark.add_file`, `ui.toggle_quick_menu`,
//! `ui.nav_file`). Clean-room: the *behaviour* is read from harpoon.nvim
//! (MIT), the code here is original.
//!
//! # A mark is a file *and a place in it*
//!
//! Upstream harpoon stores a row/col with every mark, not just a path — the
//! whole point of "four keystrokes back to the file I was just in" is landing
//! on the *line* you were on, not the top of the file. So a [`Mark`] carries
//! its 0-based cursor ([`Mark::line`]/[`Mark::col`]), captured at the moment
//! `<leader>b` was pressed, and the quick menu jumps you back to exactly there.
//! Dedup is still by *path* (marking a file already marked is a no-op that keeps
//! the original cursor), matching upstream's `add_file`.
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
//! store by the **canonicalized working directory** (see [`project_key`]),
//! matching upstream harpoon's own per-project behaviour.
//!
//! # Exactly how a project is keyed — read this before you touch the format
//!
//! The project key is `cwd.canonicalize()` rendered with
//! [`std::path::Path::to_string_lossy`]; if canonicalize fail (the directory
//! got deleted under us, say), we fall back to the raw `cwd` path string. So
//! two kvim sessions started from the *same real directory* — even via
//! different symlinks — land on the same key and share marks, while any two
//! different project roots get independent lists and can never clobber each
//! other. This exact resolution is the on-disk contract: change it and every
//! previously-saved project's marks become unreachable (the new key won't
//! match the old one), so don't anyhow change, hor.
//!
//! # Where the store live, and why not `~/.config/kvim`
//!
//! All KOPITIAM apps share one per-user root, `~/.kopitiam/<crate>/`, resolved
//! by the `kopitiam-config` crate (honours `$KOPITIAM_HOME`, else falls back to
//! `$HOME`/`$USERPROFILE`; no XDG, because Android got no XDG convention — see
//! that crate's docs). kvim's harpoon store therefore live at
//! `~/.kopitiam/kopitiam-neovim/harpoon.json`, sit together with the rest of
//! kvim's per-user config, instead of inventing its own dir or hardcoding
//! `$HOME`. [`harpoon_store_path`] is the one place that path is spelled out.
//!
//! # One file, many projects, one schema version
//!
//! The whole machine's marks sit in one JSON file: a `version` number plus a
//! map from project key to that project's ordered marks (see [`Store`]). The
//! per-project scoping live in the map's *keys*, not in separate files, so
//! there's one place to look and one file to `.gitignore` from a dotfiles
//! repo. The `version` field ([`SCHEMA_VERSION`]) is there so a future format
//! change can *migrate* an old file rather than choke on it — today's reader
//! stay lenient (an absent or unknown version still loads, since the [`Mark`]
//! shape never change yet), and every save stamps the current version.
//!
//! # Corrupt or missing store never stops kvim opening
//!
//! First run got no file — that's normal, start empty. A file that got
//! mangled (half-written, hand-edited into invalid JSON) also starts that
//! project empty rather than refusing to open the editor; the next save
//! rewrites a clean file. Losing a scrambled mark list beats not being able
//! to open your editor, lah.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The on-disk schema version stamped into every [`Store`] we write.
///
/// Bump this **only** when the JSON shape change in a way an old reader cannot
/// understand, and add a migration keyed off the value read back — never
/// silently reinterpret an old file as a new one, that's how you corrupt
/// somebody's marks. `1` is the first versioned format: a `version` field plus
/// a `projects` map of `project key -> [ {path, line, col} ]`.
pub const SCHEMA_VERSION: u32 = 1;

/// The basename of the harpoon store inside kvim's per-user config dir.
const HARPOON_STORE_FILE: &str = "harpoon.json";

/// `~/.kopitiam/kopitiam-neovim/harpoon.json` — kvim's per-user harpoon store,
/// under the shared KOPITIAM per-user root (see the module docs).
///
/// Returns `None` when no home directory can be resolved (an Android host that
/// set neither `$KOPITIAM_HOME` nor `$HOME`/`$USERPROFILE`). That's not a
/// crash — callers just degrade to an in-memory, unpersisted session, same
/// posture `kopitiam-config` itself take.
pub fn harpoon_store_path() -> Option<PathBuf> {
    kopitiam_config::app_dir("kopitiam-neovim").map(|dir| dir.join(HARPOON_STORE_FILE))
}

/// One harpoon mark: a file and the cursor it was marked at.
///
/// The cursor is stored as a plain 0-based `line`/`col` pair rather than a
/// `crate::core::Position` so this engine's on-disk format stays self-contained
/// — the persisted JSON does not depend on how `core::Position` happens to
/// serialize, and the engine keeps compiling with only `std` + `serde`. The UI
/// layer converts to and from `Position` at the seam.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mark {
    /// The marked file.
    pub path: PathBuf,
    /// 0-based cursor line captured when the file was marked. `#[serde(default)]`
    /// so an older store written before cursors were tracked still loads (its
    /// marks simply land at the top of the file).
    #[serde(default)]
    pub line: usize,
    /// 0-based cursor column captured when the file was marked.
    #[serde(default)]
    pub col: usize,
}

impl Mark {
    /// A mark for `path` at cursor `(line, col)` (both 0-based).
    pub fn new(path: PathBuf, line: usize, col: usize) -> Self {
        Self { path, line, col }
    }
}

/// The on-disk shape: every project's marks, keyed by that project's
/// canonicalized working directory. One file for the whole machine rather
/// than one file per project, so there's a single place to look (and a
/// single file to `.gitignore` from a dotfiles repo) — the per-project
/// scoping lives in the map's keys, not in the filesystem layout.
#[derive(Debug, Serialize, Deserialize)]
struct Store {
    /// Schema version, so a future format change can migrate instead of
    /// corrupting (see [`SCHEMA_VERSION`]). `#[serde(default)]` so a file
    /// written by the pre-versioning code — which had no `version` field —
    /// still loads: it deserialises as `0` (legacy/unversioned), which the
    /// current reader accepts because the [`Mark`] shape never changed. Every
    /// save rewrites this to [`SCHEMA_VERSION`].
    #[serde(default)]
    version: u32,
    /// Each project's ordered marks, keyed by [`project_key`].
    #[serde(default)]
    projects: BTreeMap<String, Vec<Mark>>,
}

impl Default for Store {
    /// A brand-new store is stamped with the *current* schema version, not
    /// `0` — `0` is reserved for "read back from a legacy, unversioned file".
    fn default() -> Self {
        Self { version: SCHEMA_VERSION, projects: BTreeMap::new() }
    }
}

/// An ordered list of marked files for one project (one working directory).
///
/// # Session-scoped vs. persisted, decided at construction
///
/// A `Harpoon` remembers *where* it saves via [`Self::store_path`]:
///
/// * [`Harpoon::load`] / [`Harpoon::load_from`] set it, so [`Harpoon::save`]
///   writes to disk — this is the real editor launch.
/// * [`Harpoon::empty`] leaves it `None`, so [`Harpoon::save`] is a **silent
///   no-op** — this is the session-scoped list a unit-test `App` holds, and
///   it's what keeps tests from ever writing into the real
///   `~/.kopitiam/kopitiam-neovim/harpoon.json`.
///
/// That single flag is why callers (the app's `<leader>b` / delete handlers)
/// can call `save()` unconditionally after every mutation without caring
/// whether they're in a test or a real session — the `Harpoon` itself knows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Harpoon {
    project_key: String,
    marks: Vec<Mark>,
    /// Where [`Harpoon::save`] writes, or `None` for a session-scoped list
    /// that never touches disk. See the type-level docs.
    store_path: Option<PathBuf>,
}

impl Harpoon {
    /// Loads the marks for the project rooted at `cwd` from
    /// [`harpoon_store_path`], or starts empty if there is no store file yet
    /// (a missing store is normal on first use, not an error — same
    /// reasoning as [`crate::config::Config::load`]).
    pub fn load(cwd: &Path) -> anyhow::Result<Self> {
        match harpoon_store_path() {
            Some(path) => Self::load_from(&path, cwd),
            // No resolvable config directory (e.g. an Android host that set
            // neither $KOPITIAM_HOME nor $HOME/$USERPROFILE) — degrade to an
            // in-memory, unpersisted session rather than failing to open.
            None => Ok(Self::empty(cwd)),
        }
    }

    /// [`Harpoon::load`] against an explicit store file, for tests (and
    /// anything else that wants to override the default location). The
    /// returned `Harpoon` remembers `store_path`, so a later
    /// [`Harpoon::save`] writes straight back there.
    ///
    /// A **corrupt** store (invalid JSON) is treated the same as an absent
    /// one: this project starts empty rather than the whole editor refusing
    /// to open — see the module docs. The next save rewrites a clean file.
    pub fn load_from(store_path: &Path, cwd: &Path) -> anyhow::Result<Self> {
        let project_key = project_key(cwd);
        let store_path_buf = Some(store_path.to_path_buf());
        if !store_path.exists() {
            return Ok(Self { project_key, marks: Vec::new(), store_path: store_path_buf });
        }
        let text = std::fs::read_to_string(store_path)?;
        // Mangled JSON -> empty, not an error. `read_to_string` above can
        // still fail on a real IO error (permissions), which *does* propagate
        // — that's not "corrupt", it's the environment being broken, and the
        // caller (`Harpoon::load`) degrades that to a session anyway.
        let marks = match serde_json::from_str::<Store>(&text) {
            Ok(store) => store.projects.get(&project_key).cloned().unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        Ok(Self { project_key, marks, store_path: store_path_buf })
    }

    /// A fresh, empty, **session-scoped** mark list for the project rooted at
    /// `cwd` — no store file is read and, crucially, none is ever written:
    /// [`Harpoon::save`] on this instance is a no-op ([`Self::store_path`] is
    /// `None`). This is what a unit-test `App` holds, so tests never touch the
    /// real `~/.kopitiam` store. The real editor launch swaps this for a
    /// [`Harpoon::load`]ed list (see `crate::ui::bootstrap::run`).
    pub fn empty(cwd: &Path) -> Self {
        Self { project_key: project_key(cwd), marks: Vec::new(), store_path: None }
    }

    /// Persists this project's marks to wherever this `Harpoon` was loaded
    /// from ([`Self::store_path`]), merging with — rather than clobbering —
    /// whatever other projects are already in the store, since two projects
    /// share one file and a save from one must not erase the other's marks.
    ///
    /// A **session-scoped** `Harpoon` (built by [`Harpoon::empty`], so
    /// `store_path` is `None`) saves nothing and returns `Ok(())`. Callers can
    /// therefore fire `save()` after every mark add/remove without branching
    /// on whether persistence is even on — the no-op is the whole point.
    pub fn save(&self) -> anyhow::Result<()> {
        match &self.store_path {
            Some(path) => self.save_to(path),
            None => Ok(()),
        }
    }

    /// [`Harpoon::save`] against an explicit store file, for tests. Always
    /// writes, regardless of [`Self::store_path`].
    pub fn save_to(&self, store_path: &Path) -> anyhow::Result<()> {
        // Read-modify-write the shared file so other projects survive. A
        // corrupt existing file is discarded (unwrap_or_default) rather than
        // aborting the save — we'd sooner rewrite one clean file than wedge
        // saving forever on a file somebody hand-mangled.
        let mut store: Store = if store_path.exists() {
            let text = std::fs::read_to_string(store_path)?;
            serde_json::from_str(&text).unwrap_or_default()
        } else {
            Store::default()
        };
        store.version = SCHEMA_VERSION;
        store.projects.insert(self.project_key.clone(), self.marks.clone());
        if let Some(parent) = store_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(store_path, serde_json::to_string_pretty(&store)?)?;
        Ok(())
    }

    /// Marks `path` at cursor `(line, col)` — the `<leader>b` keymap. A no-op
    /// (not a duplicate entry, and the original cursor is kept) if the file is
    /// already marked, matching upstream harpoon's `mark.add_file`, which
    /// dedups by path. Returns whether a new mark was actually added.
    pub fn add_file(&mut self, path: PathBuf, line: usize, col: usize) -> bool {
        if self.marks.iter().any(|m| m.path == path) {
            return false;
        }
        self.marks.push(Mark::new(path, line, col));
        true
    }

    /// Removes the mark at `index`, if any — the quick menu's delete-a-line.
    pub fn remove(&mut self, index: usize) -> Option<Mark> {
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

    /// The mark at 1-based slot `n` — `nav_file(1)` through `nav_file(9)` in
    /// upstream harpoon's default keymaps, and what the quick menu's number
    /// keys select.
    pub fn nav_file(&self, n: usize) -> Option<&Mark> {
        n.checked_sub(1).and_then(|i| self.marks.get(i))
    }

    /// All marks, in order — what both the quick-menu (`<leader><Esc>`) and
    /// the `<leader>q` picker display. Whether that display takes the
    /// shape of a popup buffer or a [`crate::plugins::picker::Picker`] is a
    /// UI decision; this module only owns the ordered data.
    pub fn marks(&self) -> &[Mark] {
        &self.marks
    }

    /// How many marks are set. A cheap read for confirmation messages
    /// ("marked, 3 marks") without cloning the list.
    pub fn len(&self) -> usize {
        self.marks.len()
    }

    /// Whether there are no marks yet — the `<leader>q`/`<leader><Esc>` "got
    /// nothing to show" guard.
    pub fn is_empty(&self) -> bool {
        self.marks.is_empty()
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

    fn paths(h: &Harpoon) -> Vec<PathBuf> {
        h.marks().iter().map(|m| m.path.clone()).collect()
    }

    #[test]
    fn add_remove_and_reorder() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());

        assert!(h.add_file(PathBuf::from("a.rs"), 0, 0));
        assert!(h.add_file(PathBuf::from("b.rs"), 0, 0));
        assert!(h.add_file(PathBuf::from("c.rs"), 0, 0));
        assert_eq!(paths(&h), vec![PathBuf::from("a.rs"), PathBuf::from("b.rs"), PathBuf::from("c.rs")]);

        // Adding an already-marked file is a no-op, not a duplicate.
        assert!(!h.add_file(PathBuf::from("b.rs"), 9, 9));
        assert_eq!(h.marks().len(), 3);

        h.reorder(0, 2);
        assert_eq!(paths(&h), vec![PathBuf::from("b.rs"), PathBuf::from("c.rs"), PathBuf::from("a.rs")]);

        assert_eq!(h.remove(1), Some(Mark::new(PathBuf::from("c.rs"), 0, 0)));
        assert_eq!(paths(&h), vec![PathBuf::from("b.rs"), PathBuf::from("a.rs")]);
    }

    #[test]
    fn re_marking_keeps_the_original_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());
        assert!(h.add_file(PathBuf::from("a.rs"), 12, 3));
        // Re-marking the same path with a different cursor must not update it
        // (upstream `add_file` is a no-op on a duplicate).
        assert!(!h.add_file(PathBuf::from("a.rs"), 99, 99));
        assert_eq!(h.marks()[0], Mark::new(PathBuf::from("a.rs"), 12, 3));
    }

    #[test]
    fn nav_file_is_one_based() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());
        h.add_file(PathBuf::from("first.rs"), 4, 2);
        h.add_file(PathBuf::from("second.rs"), 0, 0);

        assert_eq!(h.nav_file(1), Some(&Mark::new(PathBuf::from("first.rs"), 4, 2)));
        assert_eq!(h.nav_file(2), Some(&Mark::new(PathBuf::from("second.rs"), 0, 0)));
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
        h.add_file(PathBuf::from("main.rs"), 7, 1);
        h.add_file(PathBuf::from("lib.rs"), 0, 0);
        h.save_to(&store_path).unwrap();

        let reloaded = Harpoon::load_from(&store_path, project.path()).unwrap();
        // The cursor survives the round-trip, not just the path.
        assert_eq!(
            reloaded.marks(),
            &[Mark::new(PathBuf::from("main.rs"), 7, 1), Mark::new(PathBuf::from("lib.rs"), 0, 0)]
        );
    }

    #[test]
    fn two_projects_do_not_share_marks() {
        let store = tempfile::tempdir().unwrap();
        let store_path = store.path().join("harpoon.json");
        let project_a = tempfile::tempdir().unwrap();
        let project_b = tempfile::tempdir().unwrap();

        let mut a = Harpoon::load_from(&store_path, project_a.path()).unwrap();
        a.add_file(PathBuf::from("a_only.rs"), 0, 0);
        a.save_to(&store_path).unwrap();

        let mut b = Harpoon::load_from(&store_path, project_b.path()).unwrap();
        assert!(b.marks().is_empty(), "project B must not see project A's marks");
        b.add_file(PathBuf::from("b_only.rs"), 0, 0);
        b.save_to(&store_path).unwrap();

        // Saving B must not have clobbered A's entry in the shared store file.
        let a_again = Harpoon::load_from(&store_path, project_a.path()).unwrap();
        assert_eq!(paths(&a_again), vec![PathBuf::from("a_only.rs")]);
    }

    #[test]
    fn removing_an_out_of_range_index_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(dir.path());
        h.add_file(PathBuf::from("only.rs"), 0, 0);
        assert_eq!(h.remove(5), None);
        assert_eq!(h.marks().len(), 1);
    }

    #[test]
    fn a_corrupt_store_starts_empty_and_never_panics() {
        let store = tempfile::tempdir().unwrap();
        let store_path = store.path().join("harpoon.json");
        let project = tempfile::tempdir().unwrap();
        // Somebody hand-mangled the file into invalid JSON.
        std::fs::write(&store_path, b"{ this is not json at all lah ]]").unwrap();

        // Loading must not panic and must not error — it degrades to empty.
        let h = Harpoon::load_from(&store_path, project.path()).unwrap();
        assert!(h.marks().is_empty(), "a corrupt store starts the project empty");

        // And the next save rewrites a clean, well-formed file that reloads.
        let mut h = h;
        h.add_file(PathBuf::from("fresh.rs"), 1, 2);
        h.save_to(&store_path).unwrap();
        let reloaded = Harpoon::load_from(&store_path, project.path()).unwrap();
        assert_eq!(reloaded.marks(), &[Mark::new(PathBuf::from("fresh.rs"), 1, 2)]);
    }

    #[test]
    fn a_saved_store_carries_the_schema_version() {
        let store = tempfile::tempdir().unwrap();
        let store_path = store.path().join("harpoon.json");
        let project = tempfile::tempdir().unwrap();

        let mut h = Harpoon::load_from(&store_path, project.path()).unwrap();
        h.add_file(PathBuf::from("a.rs"), 0, 0);
        h.save_to(&store_path).unwrap();

        // The version field must be present on disk (so a future format change
        // can migrate, not corrupt), and it must be the current one.
        let text = std::fs::read_to_string(&store_path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            value.get("version").and_then(serde_json::Value::as_u64),
            Some(SCHEMA_VERSION as u64),
            "the written store must stamp the current schema version:\n{text}"
        );
    }

    #[test]
    fn a_legacy_unversioned_store_still_loads() {
        // A file written by the pre-versioning code has no `version` field.
        // It must still deserialise (the Mark shape never changed).
        let store = tempfile::tempdir().unwrap();
        let store_path = store.path().join("harpoon.json");
        let project = tempfile::tempdir().unwrap();
        let key = project_key(project.path());
        let legacy = format!(
            r#"{{"projects":{{"{}":[{{"path":"old.rs","line":3,"col":1}}]}}}}"#,
            key.replace('\\', "\\\\")
        );
        std::fs::write(&store_path, legacy).unwrap();

        let h = Harpoon::load_from(&store_path, project.path()).unwrap();
        assert_eq!(h.marks(), &[Mark::new(PathBuf::from("old.rs"), 3, 1)]);
    }

    #[test]
    fn a_session_scoped_harpoon_persists_nothing() {
        // `empty()` gives a no-store_path list: save() is a silent no-op, so a
        // unit-test App holding one can never write into the real ~/.kopitiam.
        let project = tempfile::tempdir().unwrap();
        let mut h = Harpoon::empty(project.path());
        h.add_file(PathBuf::from("ghost.rs"), 0, 0);
        // Does not error, and writes no file anywhere.
        h.save().unwrap();
    }
}
