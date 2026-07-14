//! Mark fixup: how a mark's rope offset moves when an edit lands before,
//! after, or on top of it.
//!
//! A mark is stored internally by [`super::buffer::Buffer`] as a rope
//! **char** offset, not a [`crate::Position`] — an offset shifts with one
//! integer add/subtract on every edit, while shifting a `Position` would
//! mean re-deriving which line the edit crossed on every keystroke.
//! `Buffer` converts to and from `Position` only at its own `mark` /
//! `set_mark` API boundary; internally, marks and the fixup in this module
//! never see a `Position`.

use std::collections::HashMap;

/// Where a mark at char offset `mark` ends up once the range `[start, end)`
/// (a normalized edit range, so `start <= end`) is replaced with `new_len`
/// chars.
///
/// - A mark strictly before the edit does not move.
/// - A mark at or after the edit's end shifts by the edit's length delta
///   (`new_len - (end - start)`), so it keeps pointing at the same text.
/// - A mark strictly *inside* the deleted range points at text that no
///   longer exists. kvim clamps it to `start` — the position where that
///   text used to begin — rather than dropping the mark outright: a
///   clamped-but-present mark degrades gracefully (an LSP diagnostic anchor
///   that lands one character off is still useful; a diagnostic that
///   vanishes is not), and marks like `` ` `` and `'` are marks vim itself
///   never lets go fully invalid. Callers that surface marks to a user
///   (e.g. a diagnostics panel, `:marks`) should be aware a mark can end up
///   coincident with another after a large deletion.
pub(crate) fn shift(mark: usize, start: usize, end: usize, new_len: usize) -> usize {
    debug_assert!(start <= end, "edit range passed to shift() must be normalized");
    if mark <= start {
        mark
    } else if mark >= end {
        // Signed arithmetic sidesteps having to reason about whether
        // new_len < (end - start); the buffer is always small enough
        // relative to isize::MAX for this to be exact.
        let delta = new_len as i64 - (end - start) as i64;
        (mark as i64 + delta) as usize
    } else {
        start
    }
}

/// Applies [`shift`] to every mark in `marks` for one edit. Called from
/// [`super::buffer::Buffer::raw_apply`] after the rope itself has been
/// mutated (the fixup only needs the edit's coordinates, not the resulting
/// text).
pub(crate) fn shift_all(marks: &mut HashMap<char, usize>, start: usize, end: usize, new_len: usize) {
    for offset in marks.values_mut() {
        *offset = shift(*offset, start, end, new_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_before_the_edit_is_unaffected() {
        assert_eq!(shift(2, 5, 8, 1), 2);
    }

    #[test]
    fn mark_at_the_edit_start_is_unaffected() {
        // A mark exactly at the insertion/deletion point does not get
        // dragged along by text inserted after it.
        assert_eq!(shift(5, 5, 5, 3), 5);
    }

    #[test]
    fn mark_after_an_insertion_shifts_right() {
        // Insert 3 chars at offset 5; a mark at 10 must move to 13.
        assert_eq!(shift(10, 5, 5, 3), 13);
    }

    #[test]
    fn mark_after_a_deletion_shifts_left() {
        // Delete chars [5, 8); a mark at 10 must move to 7.
        assert_eq!(shift(10, 5, 8, 0), 7);
    }

    #[test]
    fn mark_exactly_at_deletion_end_shifts_with_the_boundary() {
        assert_eq!(shift(8, 5, 8, 0), 5);
    }

    #[test]
    fn mark_inside_a_deleted_range_clamps_to_the_start() {
        assert_eq!(shift(6, 5, 8, 0), 5);
    }

    #[test]
    fn mark_inside_a_replaced_range_clamps_to_the_start() {
        assert_eq!(shift(6, 5, 8, 20), 5);
    }
}
