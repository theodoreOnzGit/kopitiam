//! A lazy directory-tree model — the native replacement for neo-tree (+ nui
//! for its popups, + web-devicons for its icons, which live in
//! [`crate::icons`] instead).
//!
//! # Why lazy
//!
//! neo-tree, like every practical file explorer, reads a directory's
//! children only when that directory is expanded. Reading the whole subtree
//! on open would mean `<leader>e` on a large monorepo (or a `node_modules`
//! someone forgot to `.gitignore`) hangs the editor before it draws a single
//! row. [`FileTree::new`] therefore reads exactly one level — the root's
//! immediate children — and every other directory's `children` stays `None`
//! (never touched the filesystem) until [`FileTree::expand`] or
//! [`FileTree::toggle`] is called on it. The laziness test in this module
//! asserts precisely that: a freshly-opened tree over `root/a/b/` has
//! `a`'s children still unread.
//!
//! # Why the whole tree isn't just `Vec<TreeRow>`
//!
//! A flat list can't answer "is this path currently expanded" or "what are
//! this directory's children if the user expands it right now" without
//! re-walking the disk. [`FileTree`] keeps the real (partial) tree
//! in-memory; [`FileTree::render`] is the one-way projection from that tree
//! to the flat, depth-annotated list the UI actually draws. Nothing in this
//! module knows how to draw a row — that is [`crate::ui`]'s job, per the
//! [`crate::plugins`] module contract.
//!
//! # NERDTree-style operations
//!
//! The maintainer's `plugins.lua` documents NERDTree-style in-tree keymaps
//! (`o`, `t`, `i`, `s`, `O`, `x`, `X`, `R`, `q`, `?`, `I`, `u`/`U`/`P`, `C`).
//! Of these, `q` (close window) and `?` (help) are pure UI chrome with
//! nothing for this engine to do; `t`/`i`/`s` (open in tab/hsplit/vsplit)
//! are the UI's job once it has a path — this module just needs to hand back
//! *which* path is under the cursor, which [`FileTree::render`]'s
//! [`TreeRow::path`] already does. Everything that actually changes tree
//! *state* is a method here: `o`/`x` → [`FileTree::toggle`], `O` →
//! [`FileTree::expand_all`], `X` → [`FileTree::collapse_all`], `R` →
//! [`FileTree::refresh`], `I` → [`FileTree::toggle_hidden`], `u`/`U`/`P` →
//! [`FileTree::navigate_up`], `C` → [`FileTree::set_root`].

use std::io;
use std::path::{Path, PathBuf};

/// One node in the tree. `children: None` means "not read from disk yet",
/// which is distinct from `children: Some(vec![])` ("read, and empty") —
/// collapsing that distinction would make it impossible to tell laziness
/// apart from an empty directory.
struct Node {
    path: PathBuf,
    name: String,
    is_dir: bool,
    expanded: bool,
    children: Option<Vec<Node>>,
}

impl Node {
    fn leaf(path: PathBuf, name: String, is_dir: bool) -> Self {
        Self { path, name, is_dir, expanded: false, children: None }
    }

    /// Reads this node's immediate children from disk via `ignore::Walk`
    /// (so `.gitignore` is honoured, matching what the file picker does —
    /// see the `ignore` dependency's justification in `Cargo.toml`). Always
    /// reads *every* entry, hidden files included; [`FileTree::render`]
    /// filters hidden entries at display time so toggling visibility never
    /// needs to touch the disk again.
    fn read_children(&self) -> io::Result<Vec<Node>> {
        if !self.is_dir {
            return Ok(Vec::new());
        }
        let mut entries: Vec<Node> = ignore::WalkBuilder::new(&self.path)
            .max_depth(Some(1))
            .hidden(false)
            .build()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != self.path) // depth 0 is the dir itself
            .map(|entry| {
                let is_dir = entry.file_type().is_some_and(|ft| ft.is_dir());
                let name = entry.file_name().to_string_lossy().into_owned();
                Node::leaf(entry.path().to_path_buf(), name, is_dir)
            })
            .collect();
        // Directories first, then alphabetical (case-insensitive) — the
        // conventional file-explorer ordering neo-tree also uses.
        entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
        Ok(entries)
    }

    fn find_mut(&mut self, target: &Path) -> Option<&mut Node> {
        if self.path == target {
            return Some(self);
        }
        self.children.as_mut()?.iter_mut().find_map(|child| child.find_mut(target))
    }

    fn set_expanded_recursive(&mut self, expanded: bool) {
        self.expanded = expanded;
        if let Some(children) = &mut self.children {
            for child in children {
                child.set_expanded_recursive(expanded);
            }
        }
    }
}

/// One row of the flattened, render-ready tree — what [`FileTree::render`]
/// hands to the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeRow {
    pub path: PathBuf,
    pub name: String,
    /// 0 for the root, 1 for its direct children, and so on.
    pub depth: usize,
    pub is_dir: bool,
    pub is_expanded: bool,
}

/// A lazily-populated directory tree rooted at some path.
pub struct FileTree {
    root: Node,
    show_hidden: bool,
}

impl FileTree {
    /// Opens a tree at `root`, eagerly reading only `root`'s own immediate
    /// children (one level) — see the module docs on why one level and not
    /// zero or all of them.
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| root.display().to_string());
        let mut root_node = Node { path: root, name, is_dir: true, expanded: true, children: None };
        root_node.children = Some(root_node.read_children()?);
        Ok(Self { root: root_node, show_hidden: false })
    }

    /// Re-roots the tree at `path` (the `C` keymap: "set root"), discarding
    /// everything below the old root and eagerly reading the new root's
    /// immediate children, same as [`FileTree::new`].
    pub fn set_root(&mut self, path: impl Into<PathBuf>) -> io::Result<()> {
        let show_hidden = self.show_hidden;
        *self = Self::new(path.into())?;
        // The hidden-file preference is a per-session UI setting, not a
        // property of any particular root, so re-rooting must not silently
        // reset it back to "hidden files off".
        self.show_hidden = show_hidden;
        Ok(())
    }

    /// The tree's current root path.
    pub fn root_path(&self) -> &Path {
        &self.root.path
    }

    /// Re-roots the tree at the current root's parent directory (`u`/`U`/`P`
    /// in NERDTree). A no-op — not an error, not a panic — when the current
    /// root has no parent (already at a filesystem root like `/`), since
    /// "go up" from the top has nowhere sensible to go.
    pub fn navigate_up(&mut self) -> io::Result<bool> {
        let Some(parent) = self.root.path.parent().map(Path::to_path_buf) else {
            return Ok(false);
        };
        self.set_root(parent)?;
        Ok(true)
    }

    /// Expands the directory at `path`, reading its children from disk if
    /// they haven't been read yet. A no-op if `path` isn't a known,
    /// currently-reachable directory node (e.g. an unexpanded ancestor) —
    /// the UI only ever calls this with a path it got from [`render`], so
    /// that path is always reachable in practice.
    ///
    /// [`render`]: FileTree::render
    pub fn expand(&mut self, path: &Path) -> io::Result<()> {
        let Some(node) = self.root.find_mut(path) else { return Ok(()) };
        if !node.is_dir {
            return Ok(());
        }
        if node.children.is_none() {
            node.children = Some(node.read_children()?);
        }
        node.expanded = true;
        Ok(())
    }

    /// Collapses the directory at `path`. Cached children are kept (not
    /// dropped), so re-expanding is instant and doesn't re-hit the disk —
    /// only [`FileTree::refresh`] forces a re-read.
    pub fn collapse(&mut self, path: &Path) {
        if let Some(node) = self.root.find_mut(path) {
            node.expanded = false;
        }
    }

    /// Toggles expand/collapse — the `o`/`x` keymaps.
    pub fn toggle(&mut self, path: &Path) -> io::Result<()> {
        let is_expanded = self.root.find_mut(path).map(|n| n.expanded).unwrap_or(false);
        if is_expanded {
            self.collapse(path);
            Ok(())
        } else {
            self.expand(path)
        }
    }

    /// Recursively expands `path` and every directory beneath it — the `O`
    /// keymap. Reads from disk as needed; on a very large subtree this is
    /// exactly as expensive as it sounds, which is why it's a distinct,
    /// explicitly user-requested operation rather than the default.
    pub fn expand_all(&mut self, path: &Path) -> io::Result<()> {
        self.expand(path)?;
        let Some(node) = self.root.find_mut(path) else { return Ok(()) };
        let child_paths: Vec<PathBuf> =
            node.children.iter().flatten().filter(|c| c.is_dir).map(|c| c.path.clone()).collect();
        for child in child_paths {
            self.expand_all(&child)?;
        }
        Ok(())
    }

    /// Collapses every directory back down, root included — the `X` keymap.
    /// Cached children are retained, matching [`FileTree::collapse`].
    pub fn collapse_all(&mut self) {
        self.root.set_expanded_recursive(false);
    }

    /// Re-reads `path`'s children from disk, discarding the cache — the `R`
    /// keymap, for picking up changes made outside the editor. Leaves the
    /// node's expanded state untouched.
    pub fn refresh(&mut self, path: &Path) -> io::Result<()> {
        let Some(node) = self.root.find_mut(path) else { return Ok(()) };
        if node.is_dir {
            node.children = Some(node.read_children()?);
        }
        Ok(())
    }

    /// Toggles whether dotfiles are shown — the `I` keymap. Never touches
    /// the disk: children are always read in full (see
    /// [`Node::read_children`]) and filtered only at render time.
    pub fn toggle_hidden(&mut self) {
        self.show_hidden = !self.show_hidden;
    }

    pub fn show_hidden(&self) -> bool {
        self.show_hidden
    }

    /// Flattens the currently-visible part of the tree (root, plus every
    /// expanded directory's children, recursively) into UI-ready rows.
    pub fn render(&self) -> Vec<TreeRow> {
        let mut rows = Vec::new();
        self.render_node(&self.root, 0, &mut rows);
        rows
    }

    fn render_node(&self, node: &Node, depth: usize, rows: &mut Vec<TreeRow>) {
        rows.push(TreeRow {
            path: node.path.clone(),
            name: node.name.clone(),
            depth,
            is_dir: node.is_dir,
            is_expanded: node.expanded,
        });
        if !node.expanded {
            return;
        }
        let Some(children) = &node.children else { return };
        for child in children {
            if !self.show_hidden && child.name.starts_with('.') {
                continue;
            }
            self.render_node(child, depth + 1, rows);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(dir: &Path) -> FileTree {
        FileTree::new(dir).unwrap()
    }

    #[test]
    fn root_children_are_read_eagerly_but_grandchildren_are_not() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::write(dir.path().join("a/b/leaf.txt"), "").unwrap();

        let t = tree(dir.path());
        let a = t.root.children.as_ref().unwrap().iter().find(|n| n.name == "a").unwrap();
        assert!(a.children.is_none(), "children of an unexpanded directory must not be read from disk");
    }

    #[test]
    fn expand_then_collapse_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        std::fs::create_dir(&a).unwrap();
        std::fs::write(a.join("leaf.txt"), "").unwrap();

        let mut t = tree(dir.path());
        assert!(!row_expanded(&t, &a));

        t.expand(&a).unwrap();
        assert!(row_expanded(&t, &a));
        assert!(t.render().iter().any(|r| r.name == "leaf.txt"), "expanding must reveal children in render()");

        t.collapse(&a);
        assert!(!row_expanded(&t, &a));
        assert!(!t.render().iter().any(|r| r.name == "leaf.txt"), "collapsing must hide children again");
    }

    #[test]
    fn toggle_flips_expand_state() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        std::fs::create_dir(&a).unwrap();

        let mut t = tree(dir.path());
        t.toggle(&a).unwrap();
        assert!(row_expanded(&t, &a));
        t.toggle(&a).unwrap();
        assert!(!row_expanded(&t, &a));
    }

    #[test]
    fn hidden_files_are_filtered_only_at_render_time() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "").unwrap();
        std::fs::write(dir.path().join(".hidden"), "").unwrap();

        let mut t = tree(dir.path());
        fn names(t: &FileTree) -> Vec<String> {
            t.render().iter().map(|r| r.name.clone()).collect()
        }
        assert!(names(&t).contains(&"visible.txt".to_string()));
        assert!(!names(&t).contains(&".hidden".to_string()), "hidden files must be hidden by default");

        t.toggle_hidden();
        assert!(names(&t).contains(&".hidden".to_string()), "toggling must reveal them without touching disk");
    }

    #[test]
    fn navigate_up_from_filesystem_root_is_a_no_op() {
        let mut t = tree(Path::new("/"));
        let before = t.root_path().to_path_buf();
        let changed = t.navigate_up().unwrap();
        assert!(!changed);
        assert_eq!(t.root_path(), before);
    }

    #[test]
    fn navigate_up_reroots_at_the_parent() {
        let dir = tempfile::tempdir().unwrap();
        let child = dir.path().join("a");
        std::fs::create_dir(&child).unwrap();

        let mut t = tree(&child);
        let changed = t.navigate_up().unwrap();
        assert!(changed);
        assert_eq!(t.root_path(), dir.path());
    }

    #[test]
    fn expand_all_reveals_nested_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::write(dir.path().join("a/b/leaf.txt"), "").unwrap();

        let mut t = tree(dir.path());
        t.expand_all(dir.path()).unwrap();
        assert!(t.render().iter().any(|r| r.name == "leaf.txt"));
    }

    #[test]
    fn collapse_all_hides_everything_but_keeps_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        std::fs::create_dir(&a).unwrap();
        std::fs::write(a.join("leaf.txt"), "").unwrap();

        let mut t = tree(dir.path());
        t.expand(&a).unwrap();
        t.collapse_all();
        assert_eq!(t.render().len(), 1, "only the root row should remain visible");

        // Re-expanding must not need to touch the disk again: delete the
        // directory out from under the tree and confirm the cached
        // "leaf.txt" entry still renders.
        std::fs::remove_dir_all(&a).unwrap();
        t.expand(dir.path()).unwrap();
        t.expand(&a).unwrap();
        assert!(t.render().iter().any(|r| r.name == "leaf.txt"));
    }

    #[test]
    fn refresh_re_reads_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = tree(dir.path());
        assert!(!t.render().iter().any(|r| r.name == "new.txt"));

        std::fs::write(dir.path().join("new.txt"), "").unwrap();
        t.refresh(dir.path()).unwrap();
        assert!(t.render().iter().any(|r| r.name == "new.txt"));
    }

    fn row_expanded(t: &FileTree, path: &Path) -> bool {
        t.render().iter().find(|r| r.path == path).map(|r| r.is_expanded).unwrap_or(false)
    }
}
