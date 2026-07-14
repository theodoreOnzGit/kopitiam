//! Line-number gutter labels, including vim's "hybrid" mode.
//!
//! # What hybrid mode is
//!
//! With both `number` and `relativenumber` set (the maintainer's config has
//! both `true`), vim shows the cursor's line as its **absolute** number and
//! every other visible line as its **relative** distance from the cursor.
//! This is the single most-glanced-at piece of chrome in a modal editor —
//! `3j` only works if you can see there's a `3` to jump — so the two pure
//! functions here ([`gutter_width`] and [`line_number_label`]) are kept
//! trivial and unit-tested against an explicit table, rather than folded
//! into the wider text-area rendering code where an off-by-one would be
//! harder to spot.

/// Which of vim's `number`/`relativenumber` options are active — the two
/// options together determine gutter behaviour, so they travel as a pair
/// rather than as two separate booleans threaded through every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineNumberMode {
    pub number: bool,
    pub relativenumber: bool,
}

impl LineNumberMode {
    pub const fn from_options(number: bool, relativenumber: bool) -> Self {
        Self { number, relativenumber }
    }

    /// No gutter at all — `set nonumber norelativenumber`.
    pub const fn none() -> Self {
        Self { number: false, relativenumber: false }
    }
}

/// The gutter's column width, including one trailing padding column between
/// the numbers and the text so numbers never touch the buffer content.
///
/// Width is driven by `total_lines` (the widest number that can ever appear
/// is the last line's absolute number), with a floor of 3 digits — vim's own
/// minimum gutter width — so the gutter doesn't visibly resize as a file
/// grows past 9, 99, 999 lines while scrolling.
pub fn gutter_width(total_lines: usize, mode: LineNumberMode) -> u16 {
    if !mode.number && !mode.relativenumber {
        return 0;
    }
    let digits = total_lines.max(1).to_string().len();
    (digits.max(3) + 1) as u16
}

/// The label to draw for buffer line `line_idx` (0-based) given the cursor
/// is on `cursor_line` (0-based).
///
/// Returns `None` when neither option is set — callers should not draw a
/// gutter column at all in that case (see [`gutter_width`] returning 0).
/// The returned string is the bare number (e.g. `"5"` or `"42"`); the
/// caller is responsible for right-aligning it within [`gutter_width`]
/// columns, since alignment is a rendering concern, not a labelling one.
pub fn line_number_label(line_idx: usize, cursor_line: usize, mode: LineNumberMode) -> Option<String> {
    match (mode.number, mode.relativenumber) {
        (false, false) => None,
        // Hybrid: absolute on the cursor line, relative distance elsewhere.
        (true, true) => Some(if line_idx == cursor_line {
            (line_idx + 1).to_string()
        } else {
            line_idx.abs_diff(cursor_line).to_string()
        }),
        // Pure relative: the cursor's own line is conventionally `0`,
        // matching vim (not its absolute number — that's what `number`
        // being also-set is for).
        (false, true) => Some(line_idx.abs_diff(cursor_line).to_string()),
        // Pure absolute.
        (true, false) => Some((line_idx + 1).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HYBRID: LineNumberMode = LineNumberMode { number: true, relativenumber: true };

    #[test]
    fn hybrid_mode_shows_absolute_on_cursor_line_and_relative_elsewhere() {
        // 10-line file (indices 0..10), cursor on line index 4 (the 5th
        // line, i.e. vim's absolute line 5).
        let cursor_line = 4;
        let expected: [(usize, &str); 10] = [
            (0, "4"),
            (1, "3"),
            (2, "2"),
            (3, "1"),
            (4, "5"), // cursor line: absolute, 1-based.
            (5, "1"),
            (6, "2"),
            (7, "3"),
            (8, "4"),
            (9, "5"),
        ];
        for (line_idx, want) in expected {
            assert_eq!(
                line_number_label(line_idx, cursor_line, HYBRID).as_deref(),
                Some(want),
                "line_idx={line_idx}"
            );
        }
    }

    #[test]
    fn no_numbers_when_both_options_are_off() {
        assert_eq!(line_number_label(3, 3, LineNumberMode::none()), None);
        assert_eq!(gutter_width(1000, LineNumberMode::none()), 0);
    }

    #[test]
    fn pure_relative_shows_zero_on_cursor_line() {
        let mode = LineNumberMode { number: false, relativenumber: true };
        assert_eq!(line_number_label(7, 7, mode).as_deref(), Some("0"));
        assert_eq!(line_number_label(5, 7, mode).as_deref(), Some("2"));
    }

    #[test]
    fn pure_absolute_ignores_cursor_position() {
        let mode = LineNumberMode { number: true, relativenumber: false };
        assert_eq!(line_number_label(0, 99, mode).as_deref(), Some("1"));
        assert_eq!(line_number_label(41, 0, mode).as_deref(), Some("42"));
    }

    #[test]
    fn gutter_width_has_a_three_digit_floor_plus_padding() {
        assert_eq!(gutter_width(5, HYBRID), 4); // 1 digit -> floor 3 + 1 pad.
        assert_eq!(gutter_width(999, HYBRID), 4); // 3 digits + 1 pad.
        assert_eq!(gutter_width(1000, HYBRID), 5); // 4 digits + 1 pad.
    }
}
