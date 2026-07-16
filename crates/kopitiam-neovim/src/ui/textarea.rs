//! The main buffer viewport: line-number gutter, colorcolumn guide, tab
//! expansion, unicode-width-correct text, and cursor shape/placement.
//!
//! This is a ratatui [`Widget`] plus a handful of standalone functions
//! ([`expand_line`], [`display_width`], [`cursor_style_for_mode`]) that are
//! useful, and independently testable, without a `Buffer` or a terminal at
//! all.
//!
//! # `wrap = false`
//!
//! The maintainer's config disables soft-wrap. A long line is truncated at
//! the viewport's right edge, not wrapped to the next row, and the viewport
//! scrolls **horizontally** (see [`crate::ui::scrolling::horizontal_scroll`])
//! to follow the cursor past that edge — exactly vim's behaviour with
//! `nowrap`. Implementing wrap would mean a buffer line no longer maps to
//! exactly one screen row, which cascades into scrolling, cursor placement,
//! and window-height math; `wrap=true` is deliberately out of scope until a
//! concrete need for it shows up.

use std::collections::HashMap;

use kopitiam_syntax::{HighlightSpan, Highlighter, Language};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::core::{Mode, Position};
use crate::ui::event::BufferView;
use crate::ui::gutter::{self, LineNumberMode};
use crate::ui::highlight::line_display_colors;
use crate::ui::theme::Theme;

/// Expands every tab in `line` to spaces, stopping at the next multiple of
/// `tabstop` **display columns** (not character count) — matching
/// `tabstop=4` meaning "the next column divisible by 4", which is not the
/// same as "4 spaces" once a tab isn't the first character on the line.
///
/// Operates on grapheme clusters (not `char`s) so combining marks and
/// multi-codepoint emoji are moved as a unit rather than split across the
/// expansion.
pub fn expand_line(line: &str, tabstop: usize) -> String {
    let tabstop = tabstop.max(1);
    let mut out = String::with_capacity(line.len());
    let mut col = 0usize;
    for g in line.graphemes(true) {
        if g == "\t" {
            let next_stop = (col / tabstop + 1) * tabstop;
            let width = next_stop - col;
            out.extend(std::iter::repeat_n(' ', width));
            col = next_stop;
        } else {
            out.push_str(g);
            col += g.width();
        }
    }
    out
}

/// The rendered display width of `line` in terminal cells, after tab
/// expansion and honouring wide characters (CJK, most emoji) occupying two
/// cells. This is the function that makes CJK-containing lines line up
/// correctly — using `line.chars().count()` or `line.len()` here instead
/// would misalign every line after the first wide character.
pub fn display_width(line: &str, tabstop: usize) -> usize {
    expand_line(line, tabstop).width()
}

/// The display column (0-based, post tab-expansion) at which grapheme index
/// `grapheme_idx` of `line` begins. Used to place the cursor: the editor
/// tracks cursor position in graphemes ([`Position::col`]), but the
/// terminal only understands display columns.
///
/// `grapheme_idx` at or past the end of the line returns the line's total
/// display width, matching where a cursor sits on an empty line or one
/// grapheme past the last character (both valid cursor positions in insert
/// mode, at end-of-line).
pub fn display_col_of_grapheme(line: &str, grapheme_idx: usize, tabstop: usize) -> usize {
    let tabstop = tabstop.max(1);
    let mut col = 0usize;
    for (i, g) in line.graphemes(true).enumerate() {
        if i == grapheme_idx {
            return col;
        }
        col += if g == "\t" { (col / tabstop + 1) * tabstop - col } else { g.width() };
    }
    col
}

/// Slices an already tab-expanded string to the display-column range
/// `[skip_cols, skip_cols + max_cols)`, for horizontal (`nowrap`) scrolling.
///
/// A wide character straddling either boundary is dropped whole rather than
/// rendered as a corrupted half-glyph — the same trade-off terminals
/// themselves make when a wide character is clipped by a window edge. This
/// only affects the exact frame in which horizontal scroll interrupts a
/// wide character; the next scroll step (or the cursor moving off it)
/// resolves it.
fn slice_by_display_columns(expanded: &str, skip_cols: usize, max_cols: usize) -> String {
    let mut out = String::new();
    let mut col = 0usize;
    let mut taken = 0usize;
    for ch in expanded.chars() {
        let w = ch.width().unwrap_or(0);
        if col + w <= skip_cols {
            col += w;
            continue;
        }
        if col < skip_cols {
            // Straddles the left edge: drop it.
            col += w;
            continue;
        }
        if taken + w > max_cols {
            break;
        }
        out.push(ch);
        taken += w;
        col += w;
    }
    out
}

/// Which real terminal cursor shape corresponds to each editor [`Mode`],
/// matching vim's conventions: a block in normal/visual (the cursor
/// *replaces* a character when you act on it), a bar in insert (text is
/// inserted *between* characters), and an underline in replace (the
/// character under the cursor is about to be overwritten in place).
pub fn cursor_style_for_mode(mode: Mode) -> crossterm::cursor::SetCursorStyle {
    use crossterm::cursor::SetCursorStyle;
    match mode {
        Mode::Insert => SetCursorStyle::SteadyBar,
        Mode::Replace => SetCursorStyle::SteadyUnderScore,
        // Normal, Visual*, Command, and OperatorPending all read as "acting
        // on the character under the cursor", so they share the block.
        Mode::Normal
        | Mode::Visual
        | Mode::VisualLine
        | Mode::VisualBlock
        | Mode::Command
        | Mode::OperatorPending => SetCursorStyle::SteadyBlock,
    }
}

/// The scroll offset of a window's viewport, in display units: `top` is a
/// line index, `left` is a display column (post tab-expansion) — see
/// [`crate::ui::scrolling`] for how these are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Scroll {
    pub top: usize,
    pub left: usize,
}

/// A visual-mode selection, ready to be painted.
///
/// # Why the *renderer* expands the selection
///
/// The editor hands back only `(start, end)` — the normalised anchor/cursor
/// pair — plus the mode (see `Editor::selection`). It deliberately does not say
/// *which cells* are selected, because that is not one answer: the same pair
/// means three different shapes depending on the mode, and "which cells" is a
/// question about the screen, not about the document. So the expansion lives
/// here.
///
/// The three shapes, and the trap in the middle one:
///
/// * [`Mode::Visual`] — **charwise**: from `start` to `end`, partial on the
///   first and last lines, whole lines in between.
/// * [`Mode::VisualLine`] — **linewise**: whole lines, *columns ignored
///   entirely*. Highlighting `start.col..=end.col` here is the classic bug —
///   `V` on a line with the cursor at column 20 must select the whole line, not
///   the tail of it.
/// * [`Mode::VisualBlock`] — **blockwise**: a rectangle. The column range
///   `min(start.col, end.col)..=max(...)` on every line in the row range —
///   note the columns are *re-normalised*, because a block dragged up-and-left
///   has an `end` whose column is smaller than `start`'s even after the pair
///   has been put in document order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub start: Position,
    pub end: Position,
    pub mode: Mode,
}

impl Selection {
    /// The **inclusive** grapheme-column range selected on `line`, or `None` if
    /// this line is untouched by the selection.
    ///
    /// Grapheme columns, not display columns: the caller converts (via
    /// [`display_col_of_grapheme`]) because that conversion needs the line's
    /// text, which this type does not have and should not need.
    pub fn graphemes_on(&self, line: usize, line_len: usize) -> Option<(usize, usize)> {
        if line < self.start.line || line > self.end.line {
            return None;
        }
        let last_grapheme = line_len.saturating_sub(1);
        match self.mode {
            Mode::VisualLine => Some((0, last_grapheme)),
            Mode::VisualBlock => {
                let lo = self.start.col.min(self.end.col);
                let hi = self.start.col.max(self.end.col);
                Some((lo, hi))
            }
            // Charwise. Anything that is not one of the three visual modes
            // cannot produce a `Selection` in the first place (the editor returns
            // `None`), so treating the remainder as charwise is not a silent
            // fallback — it is the only reachable case.
            _ => {
                let first = if line == self.start.line { self.start.col } else { 0 };
                let last = if line == self.end.line { self.end.col } else { last_grapheme };
                Some((first, last))
            }
        }
    }
}

/// Everything [`TextArea`] needs to render one buffer viewport.
///
/// Borrowed rather than owned throughout: a `TextArea` is constructed fresh
/// every frame from state that lives elsewhere (the window tree, the
/// editor, the config), so it is a thin, `Copy`-cheap view, not a place
/// where state lives.
pub struct TextArea<'a, B: BufferView> {
    pub buffer: &'a B,
    pub cursor: Position,
    pub mode: Mode,
    pub scroll: Scroll,
    pub line_numbers: LineNumberMode,
    /// 1-based column to draw the `colorcolumn` guide at, or `None`.
    pub colorcolumn: Option<usize>,
    pub tabstop: usize,
    pub theme: &'a Theme,
    /// The visual selection to highlight, if any. `None` in normal/insert mode,
    /// and `None` for an inactive split (only the focused window has a live
    /// selection). See [`Selection`].
    pub selection: Option<Selection>,
    /// The language to syntax-highlight this buffer as, or `None` for no
    /// highlighting (an unrecognised filetype, or `vim.opt.syntax` off). When
    /// `Some`, the viewport is highlighted as a render pass **beneath** the
    /// selection (a colour change to the foreground, so a selected keyword stays
    /// both its syntax colour and visibly selected). See
    /// [`crate::ui::highlight`].
    pub language: Option<Language>,
    /// The compiled search pattern whose matches should be highlighted across
    /// this viewport (`'hlsearch'`/`'incsearch'`), or `None` for no search
    /// highlight. Borrowed: the regex kena compiled once per frame by the caller
    /// (see [`crate::ui::app::App::render_windows`]) and shared by every window,
    /// so it live one level up instead of being rebuilt here per line.
    ///
    /// Painted as a render pass **under** [`Self::selection`] (a selected match
    /// show the selection colour, not the search colour — the selection win the
    /// cell) but **over** the syntax pass, following vim's highlight precedence.
    /// See the `render` body.
    pub search: Option<&'a regex::Regex>,
}

impl<'a, B: BufferView> TextArea<'a, B> {
    /// The screen position the terminal cursor should be placed at, in
    /// coordinates relative to `area`'s origin (i.e. add `area.x`/`area.y`
    /// yourself, or pass `area` through and use the return value directly
    /// with `Frame::set_cursor_position`, which is what
    /// [`crate::ui::app`] does).
    ///
    /// Returns `None` if the cursor's line or column is not currently
    /// within the rendered viewport — which should not happen when `scroll`
    /// was computed by [`crate::ui::scrolling`] for this same cursor, but is
    /// checked explicitly rather than assumed, because a stale `scroll`
    /// (e.g. after a resize this frame hasn't accounted for yet) is exactly
    /// the kind of desync that produces a cursor rendered off-screen.
    pub fn cursor_screen_position(&self, area: Rect) -> Option<(u16, u16)> {
        let gutter_w = gutter::gutter_width(self.buffer.line_count(), self.line_numbers);
        let row = self.cursor.line.checked_sub(self.scroll.top)?;
        if row >= area.height as usize {
            return None;
        }
        let line = self.buffer.line(self.cursor.line)?;
        let display_col = display_col_of_grapheme(&line, self.cursor.col, self.tabstop);
        let col = display_col.checked_sub(self.scroll.left)?;
        let text_width = area.width.saturating_sub(gutter_w) as usize;
        if col >= text_width {
            return None;
        }
        Some((area.x + gutter_w + col as u16, area.y + row as u16))
    }

    /// The on-screen rectangle (one row tall) to paint as selected on `line_idx`,
    /// clipped to the viewport's horizontal scroll and width.
    ///
    /// Converts the selection's **grapheme** columns into **display** columns
    /// here, where the line's text is in hand: a tab is one grapheme and four
    /// cells, and a CJK character is one grapheme and two — a highlight computed
    /// in grapheme units would drift right across any line containing either.
    fn selection_rect(
        &self,
        selection: &Selection,
        line_idx: usize,
        line: &str,
        text_x: u16,
        y: u16,
        text_width: usize,
    ) -> Option<Rect> {
        let line_len = self.buffer.line_len(line_idx);
        let (first, last) = selection.graphemes_on(line_idx, line_len)?;

        let start_col = display_col_of_grapheme(line, first, self.tabstop);
        // `last` is inclusive, so the exclusive display end is where the *next*
        // grapheme begins. Past end-of-line that saturates at the line's width,
        // which is what makes a block selection stop at a short line's end
        // instead of painting empty cells.
        let end_col = display_col_of_grapheme(line, last + 1, self.tabstop);
        // An empty line, or a cursor sitting past the last character, still gets
        // one visible cell — otherwise selecting a blank line looks like nothing
        // happened.
        let end_col = end_col.max(start_col + 1);

        // Clip into the horizontally-scrolled viewport.
        let visible_start = start_col.max(self.scroll.left);
        let visible_end = end_col.min(self.scroll.left + text_width);
        if visible_end <= visible_start {
            return None;
        }

        Some(Rect {
            x: text_x + (visible_start - self.scroll.left) as u16,
            y,
            width: (visible_end - visible_start) as u16,
            height: 1,
        })
    }

    /// Highlight spans for every **visible** line, keyed by buffer line index.
    ///
    /// Returns `None` when [`Self::language`] is `None` (no highlighting). When
    /// `Some`, the highlighter is run from **line 0** down to the last visible
    /// line, because a multi-line construct (a block comment, a triple-quoted
    /// string) open above the viewport must colour the lines below it correctly
    /// — see [`kopitiam_syntax`]'s incrementality docs. Only spans for lines at
    /// or below `scroll.top` are kept; the lines above are scanned purely to
    /// carry the [`kopitiam_syntax::LineState`] into view.
    ///
    /// This is O(scroll.top + height) per frame. For a buffer scrolled a long
    /// way down that is more work than strictly necessary; caching the
    /// per-line entry state across frames (the crate is built for exactly that)
    /// is a worthwhile follow-up, tracked separately — correctness first.
    fn visible_highlights(&self, height: u16) -> Option<HashMap<usize, Vec<HighlightSpan>>> {
        let language = self.language?;
        let last_visible = (self.scroll.top + height as usize).min(self.buffer.line_count());
        let mut highlighter = Highlighter::new(language);
        let mut map = HashMap::new();
        for idx in 0..last_visible {
            let line = self.buffer.line(idx).unwrap_or_default();
            let spans = highlighter.highlight_line(&line);
            if idx >= self.scroll.top {
                map.insert(idx, spans);
            }
        }
        Some(map)
    }

    /// Paint every search match on this line with the theme's search colours,
    /// clipped to the horizontal scroll. Each match is one contiguous run of
    /// cells, so — not like the visual selection, which is one span per line — a
    /// line can carry a few.
    ///
    /// A search match set both foreground and background (vim's `Search` group
    /// do like that), which is why this recolour whole cells instead of only
    /// tinting the background the way the selection pass do: bright-yellow-on-
    /// cream cannot read, so the text flip to dark over the fill. The selection
    /// pass run *after* this one and only touch the background, so where a
    /// selection overlap a match the selection background win while the dark
    /// match foreground stay — can lah, and the selection is still clearly the
    /// boss highlight.
    fn paint_search(
        &self,
        re: &regex::Regex,
        line: &str,
        buf: &mut Buffer,
        text_x: u16,
        y: u16,
        text_width: usize,
    ) {
        let search_bg = self.theme.search_bg();
        let search_fg = self.theme.search_fg();
        for (first, last) in crate::editor::search::line_match_cols(line, re) {
            // `last` is an exclusive grapheme end; convert both edges to display
            // columns with the line in hand (a tab is one grapheme, several
            // cells), then clip into the scrolled viewport — same arithmetic
            // `selection_rect` use.
            let start_col = display_col_of_grapheme(line, first, self.tabstop);
            let end_col = display_col_of_grapheme(line, last, self.tabstop);
            let visible_start = start_col.max(self.scroll.left);
            let visible_end = end_col.min(self.scroll.left + text_width);
            if visible_end <= visible_start {
                continue;
            }
            for col in visible_start..visible_end {
                let x = text_x + (col - self.scroll.left) as u16;
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_bg(search_bg);
                    cell.set_fg(search_fg);
                }
            }
        }
    }
}

impl<'a, B: BufferView> Widget for TextArea<'a, B> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        let gutter_w = gutter::gutter_width(self.buffer.line_count(), self.line_numbers);
        let text_x = area.x + gutter_w;
        let text_width = area.width.saturating_sub(gutter_w) as usize;
        let base_style = Style::default().fg(self.theme.fg).bg(self.theme.bg);
        let gutter_style = Style::default().fg(self.theme.gray).bg(self.theme.bg);
        let cursor_line_gutter_style = Style::default().fg(self.theme.yellow_bright).bg(self.theme.bg);

        // Fill the whole area with the base style first so short lines and
        // beyond-EOF rows don't show through whatever was previously drawn
        // in this Rect on a prior frame.
        buf.set_style(area, base_style);

        // Syntax highlighting for the whole visible range, computed once with
        // the multi-line state carried down from the top of the buffer.
        let highlights = self.visible_highlights(area.height);

        for row in 0..area.height {
            let y = area.y + row;
            let line_idx = self.scroll.top + row as usize;

            // Gutter.
            if gutter_w > 0 {
                let label = if line_idx < self.buffer.line_count() {
                    gutter::line_number_label(line_idx, self.cursor.line, self.line_numbers)
                } else {
                    None
                };
                let style =
                    if line_idx == self.cursor.line { cursor_line_gutter_style } else { gutter_style };
                match label {
                    Some(label) => {
                        // Right-align within gutter_w - 1 columns, leaving
                        // the last column as a blank separator before text.
                        let field_width = (gutter_w - 1) as usize;
                        let padded = format!("{label:>field_width$} ");
                        buf.set_stringn(area.x, y, &padded, gutter_w as usize, style);
                    }
                    None => {
                        // Past EOF: vim draws a bare `~` in the gutter area.
                        buf.set_stringn(area.x, y, "~", gutter_w as usize, style);
                    }
                }
            }

            // Text.
            if line_idx < self.buffer.line_count()
                && let Some(line) = self.buffer.line(line_idx)
            {
                let expanded = expand_line(&line, self.tabstop);
                let visible = slice_by_display_columns(&expanded, self.scroll.left, text_width);
                buf.set_stringn(text_x, y, &visible, text_width, base_style);

                // Syntax highlighting: recolour each cell's *foreground* to its
                // token colour. A background pass (selection, colorcolumn) then
                // layers on top, so a highlighted token inside a selection keeps
                // both its colour and the selection tint.
                if let Some(spans) = highlights.as_ref().and_then(|h| h.get(&line_idx)) {
                    let colors = line_display_colors(&line, self.tabstop, spans, self.theme);
                    for i in 0..text_width {
                        let display_col = self.scroll.left + i;
                        if let Some(&colour) = colors.get(display_col)
                            && colour != self.theme.fg
                            && let Some(cell) = buf.cell_mut((text_x + i as u16, y))
                        {
                            cell.set_fg(colour);
                        }
                    }
                }

                // Search-match highlight (hlsearch/incsearch): painted over the
                // syntax pass but under the selection, so a match keep its own
                // colour everywhere except where a visual selection cover it —
                // over there the selection win (see `paint_search`).
                if let Some(re) = self.search {
                    self.paint_search(re, &line, buf, text_x, y, text_width);
                }

                // Selection highlight, painted over the text as a background
                // change so the characters underneath stay legible — a selection
                // that replaced the text's colours would fight syntax
                // highlighting the moment that lands.
                if let Some(selection) = self.selection
                    && let Some(rect) =
                        self.selection_rect(&selection, line_idx, &line, text_x, y, text_width)
                {
                    buf.set_style(rect, Style::default().bg(self.theme.selection_bg()));
                }
            }
        }

        // colorcolumn: a full-height vertical guide, drawn after the text
        // so it's visible as a background tint over (or past) it, matching
        // how terminal vim renders `colorcolumn`.
        if let Some(col_1based) = self.colorcolumn {
            let target_display_col = col_1based - 1; // 0-based
            if target_display_col >= self.scroll.left {
                let x_offset = target_display_col - self.scroll.left;
                if x_offset < text_width {
                    let x = text_x + x_offset as u16;
                    let guide_rect = Rect { x, y: area.y, width: 1, height: area.height };
                    buf.set_style(guide_rect, Style::default().bg(self.theme.bg1));
                }
            }
        }
    }
}

/// Public re-export of the colour type used above, so callers building a
/// `TextArea` don't need a separate `use ratatui::style::Color` just to
/// satisfy `Theme`'s field types.
pub type ThemeColor = Color;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::FakeBuffer;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn theme() -> Theme {
        Theme::gruvbox_dark()
    }

    /// Renders `lines` with `selection` active and returns, for each row, the
    /// set of display columns painted with the selection background.
    ///
    /// Asserting on the *painted background* is the whole point: "visual mode
    /// highlighted nothing" was a bug that every state-level assertion in the
    /// suite happily agreed was fine.
    fn selected_columns(lines: &[&str], selection: Selection, width: u16) -> Vec<Vec<u16>> {
        let fb = FakeBuffer::new(lines.iter().map(|l| l.to_string()).collect());
        let height = lines.len() as u16;
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: selection.end,
                    mode: selection.mode,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: Some(selection),
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| {
                (0..width)
                    .filter(|&x| buf.cell((x, y)).unwrap().style().bg == Some(th.selection_bg()))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn charwise_visual_highlights_from_the_anchor_to_the_cursor() {
        // `v` at (0,1), then `ll` -> (0,3): columns 1..=3 inclusive.
        let painted = selected_columns(
            &["abcdef"],
            Selection { start: Position::new(0, 1), end: Position::new(0, 3), mode: Mode::Visual },
            10,
        );
        assert_eq!(painted[0], vec![1, 2, 3]);
    }

    #[test]
    fn charwise_visual_across_lines_is_partial_at_each_end_and_whole_in_between() {
        let painted = selected_columns(
            &["abcd", "efgh", "ijkl"],
            Selection { start: Position::new(0, 2), end: Position::new(2, 1), mode: Mode::Visual },
            10,
        );
        assert_eq!(painted[0], vec![2, 3], "first line: from the anchor to its end");
        assert_eq!(painted[1], vec![0, 1, 2, 3], "middle line: all of it");
        assert_eq!(painted[2], vec![0, 1], "last line: up to the cursor");
    }

    /// The one people get wrong: `V` selects the **whole line**, whatever column
    /// the cursor happens to be sitting in. A renderer that highlights
    /// `start.col..=end.col` here passes every charwise test and is still wrong.
    #[test]
    fn linewise_visual_selects_the_whole_line_regardless_of_the_cursor_column() {
        let painted = selected_columns(
            &["abcdef"],
            // Cursor parked at column 4 — a charwise reading would light only
            // column 4, or 4..=4. Linewise must light the entire line.
            Selection {
                start: Position::new(0, 4),
                end: Position::new(0, 4),
                mode: Mode::VisualLine,
            },
            10,
        );
        assert_eq!(painted[0], vec![0, 1, 2, 3, 4, 5], "V must select the whole line");
    }

    #[test]
    fn linewise_visual_selects_every_line_in_range_whole() {
        let painted = selected_columns(
            &["ab", "cdef", "gh"],
            Selection {
                start: Position::new(0, 1),
                end: Position::new(2, 0),
                mode: Mode::VisualLine,
            },
            10,
        );
        assert_eq!(painted[0], vec![0, 1]);
        assert_eq!(painted[1], vec![0, 1, 2, 3]);
        assert_eq!(painted[2], vec![0, 1]);
    }

    #[test]
    fn blockwise_visual_selects_a_rectangle() {
        let painted = selected_columns(
            &["abcdef", "ghijkl", "mnopqr"],
            Selection {
                start: Position::new(0, 1),
                end: Position::new(2, 3),
                mode: Mode::VisualBlock,
            },
            10,
        );
        // The same column range on every row — a rectangle, not a flow.
        for row in &painted {
            assert_eq!(row, &vec![1, 2, 3]);
        }
    }

    #[test]
    fn a_block_selection_stops_at_the_end_of_a_short_line() {
        let painted = selected_columns(
            &["abcdef", "gh"],
            Selection {
                start: Position::new(0, 1),
                end: Position::new(1, 4),
                mode: Mode::VisualBlock,
            },
            10,
        );
        assert_eq!(painted[0], vec![1, 2, 3, 4]);
        // The short line has nothing at columns 2..4, so the block clips rather
        // than painting empty cells past its end.
        assert_eq!(painted[1], vec![1]);
    }

    #[test]
    fn selecting_an_empty_line_still_shows_one_highlighted_cell() {
        let painted = selected_columns(
            &["abc", "", "def"],
            Selection { start: Position::new(0, 1), end: Position::new(2, 1), mode: Mode::Visual },
            10,
        );
        assert_eq!(painted[1], vec![0], "an empty line in a selection must still be visible");
    }

    /// A selection is a *background* change: the characters underneath must still
    /// be drawn (and, once syntax highlighting lands, still be their own colour).
    #[test]
    fn the_selection_highlights_the_background_and_keeps_the_text() {
        let fb = FakeBuffer::new(vec!["abcdef".to_string()]);
        let mut terminal = Terminal::new(TestBackend::new(10, 1)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(0, 2),
                    mode: Mode::Visual,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: Some(Selection {
                        start: Position::new(0, 1),
                        end: Position::new(0, 2),
                        mode: Mode::Visual,
                    }),
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        assert_eq!(buf.cell((1, 0)).unwrap().symbol(), "b");
        assert_eq!(buf.cell((1, 0)).unwrap().style().bg, Some(th.selection_bg()));
        assert_eq!(buf.cell((1, 0)).unwrap().style().fg, Some(th.fg), "the text must stay legible");
        // Just outside the selection: ordinary background.
        assert_eq!(buf.cell((3, 0)).unwrap().style().bg, Some(th.bg));
    }

    /// Tabs are one grapheme and four cells; a highlight computed in grapheme
    /// units would land in the wrong place on any line containing one.
    #[test]
    fn the_selection_is_measured_in_display_columns_not_graphemes() {
        let painted = selected_columns(
            &["\tab"],
            // Select the two characters *after* the tab: graphemes 1..=2.
            Selection { start: Position::new(0, 1), end: Position::new(0, 2), mode: Mode::Visual },
            10,
        );
        // The tab occupies display columns 0..=3, so 'a' and 'b' are at 4 and 5.
        assert_eq!(painted[0], vec![4, 5]);
    }

    #[test]
    fn no_selection_paints_no_highlight() {
        let fb = FakeBuffer::new(vec!["abcdef".to_string()]);
        let mut terminal = Terminal::new(TestBackend::new(10, 1)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::ORIGIN,
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        for x in 0..10 {
            assert_eq!(buf.cell((x, 0)).unwrap().style().bg, Some(th.bg));
        }
    }

    /// Renders `lines` with `pattern` as the active search highlight and, for
    /// each row, returns the display columns painted in the search background.
    ///
    /// The search-highlight analogue of [`selected_columns`]: it asserts on the
    /// *painted cell*, because "the search jumped but highlighted nothing" is
    /// exactly the bug this feature fixes, and only a painted-cell assertion can
    /// see it.
    fn searched_columns(lines: &[&str], pattern: &str, width: u16) -> (Theme, Vec<Vec<u16>>) {
        let fb = FakeBuffer::new(lines.iter().map(|l| l.to_string()).collect());
        let height = lines.len() as u16;
        let re = crate::editor::search::build_regex(pattern, false, false).unwrap();
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::ORIGIN,
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: Some(&re),
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let cols = (0..height)
            .map(|y| {
                (0..width)
                    .filter(|&x| buf.cell((x, y)).unwrap().style().bg == Some(th.search_bg()))
                    .collect()
            })
            .collect();
        (th, cols)
    }

    /// hlsearch lights **every** occurrence on the visible lines, not just the
    /// one the cursor jumped to — the whole point of `'hlsearch'`.
    #[test]
    fn hlsearch_paints_every_match_on_the_visible_lines() {
        let (_th, painted) = searched_columns(&["foo bar foo", "baz foo qux"], "foo", 20);
        // Line 0: "foo" at 0..3 and 8..11.
        assert_eq!(painted[0], vec![0, 1, 2, 8, 9, 10]);
        // Line 1: "foo" at 4..7.
        assert_eq!(painted[1], vec![4, 5, 6]);
    }

    /// A search match sets both colours (vim's `Search` group): a bright fill
    /// with dark text on top, so the match stays readable.
    #[test]
    fn a_search_match_paints_dark_text_on_the_search_background() {
        let fb = FakeBuffer::new(vec!["a foo b".to_string()]);
        let re = crate::editor::search::build_regex("foo", false, false).unwrap();
        let mut terminal = Terminal::new(TestBackend::new(10, 1)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::ORIGIN,
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: Some(&re),
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        // "foo" begins at column 2; its text stays legible and the character is
        // still there.
        assert_eq!(buf.cell((2, 0)).unwrap().symbol(), "f");
        assert_eq!(buf.cell((2, 0)).unwrap().style().bg, Some(th.search_bg()));
        assert_eq!(buf.cell((2, 0)).unwrap().style().fg, Some(th.search_fg()));
        // The space just before it is untouched — ordinary background.
        assert_eq!(buf.cell((1, 0)).unwrap().style().bg, Some(th.bg));
    }

    /// Where a visual selection covers a search match, the **selection** wins the
    /// cell's background — search highlight sits under it (vim's precedence).
    #[test]
    fn the_selection_wins_over_the_search_highlight_where_they_overlap() {
        let fb = FakeBuffer::new(vec!["foofoo".to_string()]);
        let re = crate::editor::search::build_regex("foo", false, false).unwrap();
        let mut terminal = Terminal::new(TestBackend::new(10, 1)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(0, 2),
                    mode: Mode::Visual,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    // Select columns 0..=2 — over the first "foo" match.
                    selection: Some(Selection {
                        start: Position::new(0, 0),
                        end: Position::new(0, 2),
                        mode: Mode::Visual,
                    }),
                    language: None,
                    search: Some(&re),
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        // Columns 0..=2 are selected: selection background wins, even though a
        // search match also covers them.
        for x in 0..=2 {
            assert_eq!(
                buf.cell((x, 0)).unwrap().style().bg,
                Some(th.selection_bg()),
                "selection must win the cell at {x}"
            );
        }
        // Columns 3..=5 are the second match, unselected: search background.
        for x in 3..=5 {
            assert_eq!(buf.cell((x, 0)).unwrap().style().bg, Some(th.search_bg()));
        }
    }

    /// Search highlight is measured in display columns: a match after a tab lands
    /// past the tab's expansion, not at its grapheme index.
    #[test]
    fn the_search_highlight_is_measured_in_display_columns_not_graphemes() {
        // The tab occupies display columns 0..=3, so "foo" is at 4..=6.
        let (_th, painted) = searched_columns(&["\tfoo"], "foo", 12);
        assert_eq!(painted[0], vec![4, 5, 6]);
    }

    /// Syntax highlighting recolours the right cells: a Rust keyword's cells
    /// carry the theme's keyword colour, and the identifier after it keeps the
    /// default foreground. Asserts the PAINTED CELL, not any internal state.
    #[test]
    fn a_rust_keyword_is_painted_in_the_theme_keyword_colour_via_test_backend() {
        let fb = FakeBuffer::new(vec!["fn main() {}".to_string()]);
        let mut terminal = Terminal::new(TestBackend::new(20, 1)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::ORIGIN,
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: Some(Language::Rust),
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        // "fn" occupies columns 0..=1 and is a keyword (gruvbox red).
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "f");
        assert_eq!(buf.cell((0, 0)).unwrap().style().fg, Some(th.red_bright), "the keyword must be red");
        assert_eq!(buf.cell((1, 0)).unwrap().style().fg, Some(th.red_bright));
        // The space at column 2 is not part of any token: default fg.
        assert_eq!(buf.cell((2, 0)).unwrap().symbol(), " ");
        assert_eq!(buf.cell((2, 0)).unwrap().style().fg, Some(th.fg), "the gap keeps default fg");
        // "main" at column 3 is a function-call name, a distinct colour.
        assert_eq!(buf.cell((3, 0)).unwrap().symbol(), "m");
        assert_eq!(buf.cell((3, 0)).unwrap().style().fg, Some(th.aqua_bright), "the call name is a function colour");
    }

    /// A block comment left open on the line *above* the viewport must still
    /// colour the visible line as a comment — the multi-line-state carry the
    /// whole highlighter design exists for. The window is scrolled past the
    /// opening `/*`, so a naive "start fresh at the top visible line" renderer
    /// would paint this as plain code.
    #[test]
    fn a_block_comment_open_above_the_viewport_still_colours_the_visible_line() {
        let fb = FakeBuffer::new(vec![
            "/* opening".to_string(),
            "still inside".to_string(),
            "closes here */".to_string(),
        ]);
        let mut terminal = Terminal::new(TestBackend::new(20, 1)).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(1, 0),
                    mode: Mode::Normal,
                    // Scrolled so only line index 1 ("still inside") is visible.
                    scroll: Scroll { top: 1, left: 0 },
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: Some(Language::Rust),
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "s");
        assert_eq!(
            buf.cell((0, 0)).unwrap().style().fg,
            Some(th.gray),
            "a line inside a block comment opened above the viewport must be gray"
        );
    }

    #[test]
    fn tabs_expand_to_tabstop_columns() {
        assert_eq!(expand_line("\tx", 4), "    x");
        assert_eq!(expand_line("ab\tx", 4), "ab  x"); // tab from col 2 -> col 4.
        assert_eq!(display_width("\t", 4), 4);
    }

    #[test]
    fn cjk_and_emoji_report_double_width() {
        // Each CJK character is 2 cells; "中文" is 4 cells wide, not 2 chars.
        assert_eq!(display_width("中文", 4), 4);
        assert_eq!(display_width("a中b", 4), 4); // 1 + 2 + 1.
    }

    #[test]
    fn cursor_shapes_match_vim_conventions() {
        use crossterm::cursor::SetCursorStyle;
        assert_eq!(cursor_style_for_mode(Mode::Insert), SetCursorStyle::SteadyBar);
        assert_eq!(cursor_style_for_mode(Mode::Replace), SetCursorStyle::SteadyUnderScore);
        assert_eq!(cursor_style_for_mode(Mode::Normal), SetCursorStyle::SteadyBlock);
    }

    #[test]
    fn cursor_screen_position_accounts_for_gutter_and_scroll() {
        let lines: Vec<String> = (0..10).map(|i| format!("line {i}")).collect();
        let fb = FakeBuffer::new(lines);
        let ta = TextArea {
            buffer: &fb,
            cursor: Position::new(4, 2),
            mode: Mode::Normal,
            scroll: Scroll { top: 0, left: 0 },
            line_numbers: LineNumberMode::from_options(true, true),
            colorcolumn: None,
            tabstop: 4,
            theme: &theme(),
            selection: None,
            language: None,
            search: None,
        };
        let area = Rect { x: 0, y: 0, width: 40, height: 10 };
        // gutter_width(10 lines) = 3 digits floor + 1 pad = 4.
        let (x, y) = ta.cursor_screen_position(area).unwrap();
        assert_eq!(y, 4);
        assert_eq!(x, 4 + 2); // gutter width 4 + display col 2.
    }

    /// Renders via `TestBackend` (per the testing brief: "ratatui has a
    /// `TestBackend` — use it") and asserts the CJK line occupies the
    /// correct number of terminal cells rather than the correct char count.
    #[test]
    fn cjk_line_renders_at_correct_width_via_test_backend() {
        let fb = FakeBuffer::new(vec!["中文abc".to_string()]);
        let backend = TestBackend::new(20, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(0, 0),
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        // No gutter (line numbers off), so text starts at column 0: 中(2)
        // 文(2) a b c -> "abc" begins at display column 4.
        assert_eq!(buffer.cell((4, 0)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((5, 0)).unwrap().symbol(), "b");
        assert_eq!(buffer.cell((6, 0)).unwrap().symbol(), "c");
    }

    /// Same brief: confirms a tab occupies `tabstop` (4) columns end to end
    /// through the real widget render path, not just the pure function.
    #[test]
    fn tab_occupies_four_columns_via_test_backend() {
        let fb = FakeBuffer::new(vec!["\tx".to_string()]);
        let backend = TestBackend::new(20, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(0, 0),
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((0, 0)).unwrap().symbol(), " ");
        assert_eq!(buffer.cell((3, 0)).unwrap().symbol(), " ");
        assert_eq!(buffer.cell((4, 0)).unwrap().symbol(), "x");
    }

    /// Same brief: colorcolumn=75 appears at the right screen column.
    #[test]
    fn colorcolumn_guide_appears_at_configured_column_via_test_backend() {
        let fb = FakeBuffer::new(vec!["x".repeat(100)]);
        let backend = TestBackend::new(120, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(0, 0),
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::none(),
                    colorcolumn: Some(75),
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        // 1-based column 75 -> 0-based display column 74 (no gutter here).
        assert_eq!(buffer.cell((74, 0)).unwrap().style().bg, Some(th.bg1));
        assert_eq!(buffer.cell((73, 0)).unwrap().style().bg, Some(th.bg));
    }

    /// Renders a 10-line buffer with the cursor on line index 4 (vim's
    /// absolute line 5) through the full widget, and asserts the *rendered*
    /// gutter text shows hybrid numbers correctly — the same case
    /// `gutter::tests::hybrid_mode_shows_absolute_on_cursor_line_and_relative_elsewhere`
    /// covers at the pure-function level, exercised here end to end via
    /// `TestBackend` per the testing brief.
    #[test]
    fn hybrid_line_numbers_render_correctly_via_test_backend() {
        let lines: Vec<String> = (0..10).map(|i| format!("l{i}")).collect();
        let fb = FakeBuffer::new(lines);
        let backend = TestBackend::new(10, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let th = theme();
        terminal
            .draw(|frame| {
                let ta = TextArea {
                    buffer: &fb,
                    cursor: Position::new(4, 0),
                    mode: Mode::Normal,
                    scroll: Scroll::default(),
                    line_numbers: LineNumberMode::from_options(true, true),
                    colorcolumn: None,
                    tabstop: 4,
                    theme: &th,
                    selection: None,
                    language: None,
                    search: None,
                };
                frame.render_widget(ta, frame.area());
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        let row_text = |y: u16| -> String {
            (0..3).map(|x| buffer.cell((x, y)).unwrap().symbol().to_string()).collect()
        };
        // gutter_width(10) = 4 (3 digit floor + 1 pad); numbers right-align
        // in the first 3 columns.
        assert_eq!(row_text(0).trim(), "4"); // |cursor_line - 0| = 4
        assert_eq!(row_text(4).trim(), "5"); // cursor line: absolute 5.
        assert_eq!(row_text(9).trim(), "5"); // |9 - 4| = 5
    }
}
