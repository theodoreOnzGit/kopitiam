//! Pure viewport-scrolling math.
//!
//! Kept free of ratatui, crossterm, and the editor traits entirely — these
//! are plain `usize -> usize` functions — because scrolling arithmetic is
//! exactly the kind of code where off-by-ones hide at the edges of the file,
//! and the cheapest way to pin those down is a table of `(input, expected)`
//! unit tests with no terminal, no fake editor, and no rendering involved.

/// Computes the new top-of-viewport line so that `cursor_line` stays within
/// `scrolloff` lines of the viewport's top and bottom edge — vim's
/// `scrolloff` option — clamped so the viewport never scrolls past the start
/// or end of the buffer.
///
/// # Behaviour at the edges (where the bugs are)
///
/// * Near the **top** of the buffer, `scrolloff` is not honoured once doing
///   so would require scrolling above line 0 — vim does not pad the top of
///   the file with blank lines to satisfy scrolloff, it just stops at 0.
/// * Near the **bottom**, the same: once the last line is at the bottom of
///   the viewport, further downward cursor movement stops scrolling (the
///   cursor keeps moving within the viewport, up to the last line) rather
///   than padding past the end of the buffer.
/// * If the whole buffer fits within `viewport_height`, the top is always 0
///   — there is nothing to scroll.
/// * `scrolloff` is clamped to at most half the viewport height (rounded
///   down), matching vim: a `scrolloff` >= half the window would otherwise
///   make the cursor unable to reach some lines at all as you scroll.
pub fn vertical_scroll(
    cursor_line: usize,
    total_lines: usize,
    viewport_height: usize,
    scrolloff: usize,
    current_top: usize,
) -> usize {
    if viewport_height == 0 {
        return current_top;
    }
    if total_lines <= viewport_height {
        return 0;
    }

    // vim clamps scrolloff to floor((height - 1) / 2) so it can never pin
    // the cursor to a position it can't leave.
    let scrolloff = scrolloff.min(viewport_height.saturating_sub(1) / 2);

    let mut top = current_top;

    // Cursor must be at least `scrolloff` lines below the top edge...
    if cursor_line < top + scrolloff {
        top = cursor_line.saturating_sub(scrolloff);
    }
    // ...and at least `scrolloff` lines above the bottom edge.
    let bottom_bound = top + viewport_height;
    if cursor_line + scrolloff + 1 > bottom_bound {
        top = cursor_line + scrolloff + 1 - viewport_height;
    }

    // Never scroll past the last page.
    top.min(total_lines - viewport_height)
}

/// Computes the new left-of-viewport display column so `cursor_col`
/// (a **display** column, already accounting for tab expansion and
/// wide-character width — see [`crate::ui::textarea::expand_line`]) stays
/// visible. There is no horizontal analogue of scrolloff in vim; the cursor
/// is kept exactly at the edge, not padded, matching `wrap=false` behaviour.
pub fn horizontal_scroll(cursor_col: usize, viewport_width: usize, current_left: usize) -> usize {
    if viewport_width == 0 {
        return current_left;
    }
    if cursor_col < current_left {
        cursor_col
    } else if cursor_col >= current_left + viewport_width {
        cursor_col + 1 - viewport_width
    } else {
        current_left
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_scroll_needed_when_cursor_already_within_bounds() {
        // 100-line file, 20-row viewport, cursor comfortably in the middle
        // of a viewport that's already positioned there.
        assert_eq!(vertical_scroll(50, 100, 20, 5, 45), 45);
    }

    #[test]
    fn scrolls_down_to_keep_scrolloff_below_cursor() {
        // viewport_height=21 keeps scrolloff=5 unclamped (clamp threshold is
        // (21-1)/2=10), so this exercises the "normal" scrolloff request
        // rather than the clamped edge case covered separately below.
        // Cursor at line 20, top at 0: bottom-visible line is
        // top+height-1=20, cursor needs 5 lines of context below it, so top
        // must become 20+5+1-21 = 5.
        assert_eq!(vertical_scroll(20, 100, 21, 5, 0), 5);
    }

    #[test]
    fn small_viewport_clamps_scrolloff_to_half_height() {
        // viewport_height=10 clamps scrolloff to (10-1)/2=4, not the
        // requested 5. Cursor at line 10, top 0: top becomes
        // 10+4+1-10 = 5, not the 6 an unclamped scrolloff=5 would give.
        assert_eq!(vertical_scroll(10, 100, 10, 5, 0), 5);
    }

    #[test]
    fn scrolls_up_to_keep_scrolloff_above_cursor() {
        // top currently 20, cursor moves to line 22 (only 2 lines below top,
        // needs 5): top must become 22-5=17.
        assert_eq!(vertical_scroll(22, 100, 20, 5, 20), 17);
    }

    #[test]
    fn top_of_file_does_not_pad_scrolloff_with_blank_space() {
        // Cursor on line 2 (0-based) of a 10-line file that fits in a
        // 10-row... no: use a taller file than viewport so scrolling is
        // possible in principle, but the cursor is too close to line 0 to
        // honour scrolloff=5 without going negative.
        assert_eq!(vertical_scroll(2, 100, 10, 5, 0), 0);
    }

    #[test]
    fn bottom_of_file_clamps_instead_of_padding_past_the_end() {
        // 20-line file, viewport 10 rows: max top is 20-10=10. Cursor on the
        // very last line (19) with scrolloff 5 would ask for top=19+5+1-10=15,
        // but that must clamp to 10.
        assert_eq!(vertical_scroll(19, 20, 10, 5, 0), 10);
    }

    #[test]
    fn whole_buffer_fits_in_viewport_top_is_always_zero() {
        assert_eq!(vertical_scroll(3, 8, 20, 5, 7), 0);
    }

    #[test]
    fn scrolloff_larger_than_half_viewport_is_clamped() {
        // viewport_height=10, so scrolloff clamps to (10-1)/2 = 4, not 5.
        // Cursor at line 4, top 0: bottom_bound check: 4+4+1=9 <= 0+10, top
        // check: 4 < 0+4? no. So top stays 0.
        assert_eq!(vertical_scroll(4, 100, 10, 5, 0), 0);
    }

    #[test]
    fn horizontal_scroll_follows_cursor_past_the_right_edge() {
        assert_eq!(horizontal_scroll(100, 80, 0), 21);
    }

    #[test]
    fn horizontal_scroll_follows_cursor_past_the_left_edge() {
        assert_eq!(horizontal_scroll(5, 80, 50), 5);
    }

    #[test]
    fn horizontal_scroll_is_unchanged_when_cursor_already_visible() {
        assert_eq!(horizontal_scroll(30, 80, 10), 10);
    }
}
