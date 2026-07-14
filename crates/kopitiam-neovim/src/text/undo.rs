//! A branching undo history.
//!
//! Vim's undo is a tree, not a stack: press `u`, then type something new,
//! and the branch you undid away from is not destroyed — `g-`/`g+` and
//! `:undolist` can still reach it, because vim keeps it as a sibling of the
//! new edit. Modeling this as a stack and bolting branches on later means
//! rewriting everything that ever called `undo`/`redo`; modeling it as a
//! tree from the start costs almost nothing and avoids that rewrite
//! entirely — see bead-worthy note in the module's own history if this ever
//! needs revisiting.
//!
//! # Shape
//!
//! Each [`Node`] holds the forward [`Edit`]s that transform its *parent's*
//! text into its own, and their precomputed inverses (transforming its own
//! text back into the parent's). [`UndoTree::undo`] hands back a node's
//! inverses and moves `current` to that node's parent; [`UndoTree::redo`]
//! hands back a child's forwards and moves `current` to that child. Editing
//! while `current` already has children — i.e. after an `undo` — appends a
//! **new** child rather than overwriting one: the old branch stays in
//! `Node::children`, merely no longer the one `last_child` points at. Only
//! "redo the most-recently-touched branch" (plain `u` / `<C-r>`) is exposed
//! here; sibling-aware `g-`/`g+` browsing is an editor-layer concern that
//! can walk this same tree later without any change to this module.
//!
//! # Grouping
//!
//! An insert-mode session is one undo step in vim, not one per keystroke.
//! [`UndoTree::begin_group`] / [`UndoTree::end_group`] buffer edits into a
//! pending, not-yet-committed node instead of committing one node per
//! [`UndoTree::record`] call; the whole group commits as a single node when
//! the group closes. Calls nest via a depth counter, so an editor-layer
//! helper that itself brackets a smaller helper in `begin_group`/`end_group`
//! doesn't have to know whether it's already inside a larger group.

use crate::core::Edit;

/// Identifies a node in the undo tree. Never exposed outside `text` —
/// callers only ever get positions and text back out of `undo`/`redo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct NodeId(usize);

#[derive(Debug)]
struct Node {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    /// Index into `children` that `redo` will follow. Updated both when a
    /// new child is committed and when `redo` traverses one, so redo always
    /// continues down whichever branch was most recently touched — matching
    /// plain vim `u`/`<C-r>`, which does not require re-selecting a branch
    /// after every undo.
    last_child: Option<usize>,
    /// Edits, in application order, that turn the parent's text into this
    /// node's text. A grouped insert session stores every keystroke here.
    forward: Vec<Edit>,
    /// Edits that undo `forward`, in the order they must be *replayed*
    /// (i.e. already reversed relative to `forward`): the last edit applied
    /// is the first one undone.
    inverse: Vec<Edit>,
}

/// A branching, groupable undo history. See the module docs for the shape
/// and the grouping contract.
#[derive(Debug)]
pub(crate) struct UndoTree {
    nodes: Vec<Node>,
    current: NodeId,
    group_depth: u32,
    pending_forward: Vec<Edit>,
    pending_inverse: Vec<Edit>,
}

impl UndoTree {
    /// A tree with just the root node — the buffer's pristine, never-edited
    /// state. `current` starts here, and there is nothing to undo to yet.
    pub(crate) fn new() -> Self {
        let root = Node { parent: None, children: Vec::new(), last_child: None, forward: Vec::new(), inverse: Vec::new() };
        Self {
            nodes: vec![root],
            current: NodeId(0),
            group_depth: 0,
            pending_forward: Vec::new(),
            pending_inverse: Vec::new(),
        }
    }

    /// The node `current` points at right now. Used by `Buffer` purely as
    /// an opaque "have we changed since save?" token — compared for
    /// equality, never dereferenced.
    pub(crate) fn current_id(&self) -> NodeId {
        self.current
    }

    /// Opens (or re-enters, if already open) an undo group: edits recorded
    /// via [`UndoTree::record`] until the matching [`UndoTree::end_group`]
    /// coalesce into a single undo step.
    pub(crate) fn begin_group(&mut self) {
        self.group_depth += 1;
    }

    /// Closes one level of undo group. Once the outermost `begin_group` is
    /// matched, the buffered edits (if any were recorded) commit as one
    /// node. Unbalanced calls (more `end_group` than `begin_group`) are a
    /// caller bug but are tolerated as a no-op rather than panicking, since
    /// a text engine primitive should not be able to crash the editor over
    /// a bookkeeping mismatch in the layer above it.
    pub(crate) fn end_group(&mut self) {
        if self.group_depth == 0 {
            return;
        }
        self.group_depth -= 1;
        if self.group_depth == 0 {
            self.commit_pending();
        }
    }

    /// Records one already-applied edit and its inverse. Commits
    /// immediately as a single-edit node unless a group is open, in which
    /// case it is buffered until [`UndoTree::end_group`] closes the group.
    pub(crate) fn record(&mut self, forward: Edit, inverse: Edit) {
        self.pending_forward.push(forward);
        self.pending_inverse.push(inverse);
        if self.group_depth == 0 {
            self.commit_pending();
        }
    }

    fn commit_pending(&mut self) {
        if self.pending_forward.is_empty() {
            return;
        }
        let forward = std::mem::take(&mut self.pending_forward);
        let mut inverse = std::mem::take(&mut self.pending_inverse);
        // `inverse` was accumulated in application order; undoing the group
        // must peel off the *last* edit first.
        inverse.reverse();

        let new_id = NodeId(self.nodes.len());
        self.nodes.push(Node { parent: Some(self.current), children: Vec::new(), last_child: None, forward, inverse });

        let parent = &mut self.nodes[self.current.0];
        parent.children.push(new_id);
        parent.last_child = Some(parent.children.len() - 1);
        self.current = new_id;
    }

    /// The edits to replay, in order, to undo the current node, plus the
    /// node `current` moves to afterward. `None` at the root — there is
    /// nothing before the buffer's pristine state.
    ///
    /// Flushes any open group first, so calling `undo` mid-group still
    /// undoes the group's edits as a single step rather than leaving them
    /// stranded as pending state.
    pub(crate) fn undo(&mut self) -> Option<(Vec<Edit>, NodeId)> {
        self.commit_pending();
        let node = &self.nodes[self.current.0];
        let parent = node.parent?;
        let edits = node.inverse.clone();
        self.current = parent;
        Some((edits, parent))
    }

    /// The edits to replay, in order, to redo into the most-recently
    /// touched child of the current node, plus that child's id. `None` if
    /// the current node has no children (nothing to redo).
    pub(crate) fn redo(&mut self) -> Option<(Vec<Edit>, NodeId)> {
        self.commit_pending();
        let node = &self.nodes[self.current.0];
        let child_idx = node.last_child?;
        let child_id = node.children[child_idx];
        let edits = self.nodes[child_id.0].forward.clone();
        self.current = child_id;
        Some((edits, child_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Position, Range};

    /// A placeholder edit; these tests only care about tree *shape*, not
    /// about real buffer content, so any distinct `Edit` value works as a
    /// stand-in and the label makes failures readable.
    fn edit(label: &str) -> Edit {
        Edit { range: Range::point(Position::ORIGIN), text: label.to_string() }
    }

    #[test]
    fn fresh_tree_has_nothing_to_undo_or_redo() {
        let mut tree = UndoTree::new();
        assert!(tree.undo().is_none());
        assert!(tree.redo().is_none());
    }

    #[test]
    fn record_without_a_group_commits_one_node_per_edit() {
        let mut tree = UndoTree::new();
        tree.record(edit("a"), edit("undo-a"));
        let root = tree.current_id();
        tree.record(edit("b"), edit("undo-b"));
        assert_ne!(tree.current_id(), root);

        let (edits, _) = tree.undo().unwrap();
        assert_eq!(edits, vec![edit("undo-b")], "second edit undoes alone, as its own step");
    }

    #[test]
    fn grouped_edits_undo_as_a_single_step_in_reverse_order() {
        let mut tree = UndoTree::new();
        tree.begin_group();
        for ch in ['1', '2', '3'] {
            tree.record(edit(&ch.to_string()), edit(&format!("undo-{ch}")));
        }
        tree.end_group();

        // Nothing should have committed until the group closed: exactly one
        // new node beyond the root.
        assert_eq!(tree.nodes.len(), 2);

        let (edits, parent) = tree.undo().unwrap();
        assert_eq!(edits, vec![edit("undo-3"), edit("undo-2"), edit("undo-1")]);
        assert_eq!(parent, NodeId(0));
        assert!(tree.undo().is_none(), "the whole group was one step; nothing left to undo");
    }

    #[test]
    fn nested_groups_collapse_into_one_step() {
        let mut tree = UndoTree::new();
        tree.begin_group();
        tree.begin_group();
        tree.record(edit("a"), edit("undo-a"));
        tree.end_group(); // inner close: depth 1, must NOT commit yet
        assert_eq!(tree.nodes.len(), 1, "inner end_group must not commit while the outer group is still open");
        tree.record(edit("b"), edit("undo-b"));
        tree.end_group(); // outer close: depth 0, commits both edits as one node
        assert_eq!(tree.nodes.len(), 2);
    }

    #[test]
    fn diverging_after_undo_keeps_the_old_branch_as_a_sibling_not_a_replacement() {
        let mut tree = UndoTree::new();
        tree.record(edit("a"), edit("undo-a")); // node 1, child of root
        tree.record(edit("b"), edit("undo-b")); // node 2, child of node 1
        assert_eq!(tree.nodes.len(), 3);

        let (_edits, parent) = tree.undo().unwrap(); // current: node 1
        assert_eq!(parent, NodeId(1));

        tree.record(edit("c"), edit("undo-c")); // node 3, a NEW sibling of node 2
        assert_eq!(tree.nodes.len(), 4, "diverging must create a new node, not reuse or drop the old one");

        let node1_children = &tree.nodes[1].children;
        assert_eq!(node1_children.len(), 2, "node 1 must have BOTH the b-branch and the c-branch as children");
        assert!(node1_children.contains(&NodeId(2)), "the old (b) branch must still exist");
        assert!(node1_children.contains(&NodeId(3)), "the new (c) branch must exist alongside it");

        // `record` above already moved `current` to the new node (3), so
        // walk back to node 1 first to exercise the interesting case: redo
        // *from the branch point* must follow the newest branch (c),
        // matching plain vim `u`/`<C-r>` semantics, without having deleted
        // the older (b) branch.
        let (_edits, back_at_branch_point) = tree.undo().unwrap();
        assert_eq!(back_at_branch_point, NodeId(1));

        let (edits, redone) = tree.redo().unwrap();
        assert_eq!(redone, NodeId(3));
        assert_eq!(edits, vec![edit("c")]);
    }
}
