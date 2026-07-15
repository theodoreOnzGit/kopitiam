# AID-0027: The window-UX batch does not close cj0.10.1

* **Status:** Pending review
* **Bead:** `kopitiam-cj0.30`, `kopitiam-q8v` (both closed by this batch); `kopitiam-cj0.10.1` (deliberately left open)
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The premise, and why it did not hold

The task that shipped the kvim window-UX batch (bare `<C-h/j/k/l>` window
navigation + tmux edge hand-off, the file tree as a focus target, visible split
borders, and the quit-all/write-all ex-commands) instructed me to close
`kopitiam-cj0.10.1` "if the tree focus works", framing that bead as the
focusable-file-explorer feature.

`cj0.10.1` is **not** that feature. It is a filetree *engine* bug:

> kvim: `plugins::filetree` cannot distinguish an unreadable directory from an
> empty one. `FileTree::read_children` walks with `ignore::Walk` and drops
> failed entries (`.filter_map(|e| e.ok())`). Expanding a directory with no read
> permission therefore SUCCEEDS, yields zero children, and is indistinguishable
> from an empty directory.

The feature bead for this batch is a different one, `kopitiam-q8v` ("kvim:
`<C-h/j/k/l>` window nav, tmux edge-handoff, file-tree focus target, visible
split borders"). Its own notes are explicit about the relationship: "Closes
cj0.10.1 **if filetree engine gap resolved**." This batch resolves no filetree
engine gap — it touches window focus, borders, navigation and ex-commands, none
of which go near `read_children`'s `filter_map(|e| e.ok())`.

## What was decided

* Closed `kopitiam-cj0.30` (the in-flight verify/finish/commit bead).
* Closed `kopitiam-q8v` (all three feature gaps delivered, tested, and
  PTY-verified on the real binary; tmux hand-off confirmed both by unit test and
  a real two-pane tmux session).
* **Left `kopitiam-cj0.10.1` open.** Closing it would have marked an untouched
  permission-vs-emptiness engine bug as fixed, hiding a real gap from the review
  queue. The "file-tree focus target" part of q8v is done, but that is not what
  cj0.10.1 tracks.

## Alternatives considered

1. **Close cj0.10.1 as instructed.** Rejected: it would falsely record the
   unreadable-vs-empty-directory bug as resolved. The UI already carries a
   workaround (a `std::fs::read_dir` probe at expand time drawing an honest error
   row — see `ui/filetree.rs` and AID-0018), but the *engine* still silently
   drops unreadable entries. Marking the engine bug closed because a UI patch
   exists is precisely the "UI-side patch over an engine gap" the bead calls out.
2. **Close cj0.10.1 and file a fresh follow-up for the engine gap.** Rejected as
   churn: the existing bead already describes the gap precisely; re-filing it
   under a new id loses history for no gain.

## What would make this wrong

* If the maintainer's real intent was that `cj0.10.1` *is* the focusable-explorer
  feature and the filetree-engine text on it is stale/mislabelled, then leaving
  it open is wrong and it should be closed (and its description corrected). I
  judged the bead's written description authoritative over the task's one-line
  framing, because the description is specific, technical, and matches live code.
* If a future change to `FileTree::read_children` makes the walk report
  unreadable directories distinctly from empty ones, cj0.10.1 closes then — on
  its own merits, not as a side effect of window UX.

## Lesson

A one-line "closes bead X" in a task is a claim to verify against bead X's actual
description, not an instruction to execute blind. When they disagree, the bead's
technical description wins, and the disagreement itself is worth recording — a
wrongly-closed bead removes a real defect from the review queue, which is more
expensive than an extra open bead.
