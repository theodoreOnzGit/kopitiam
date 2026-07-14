//! Bridges [`kopitiam_syntax`]'s theme-agnostic [`HighlightKind`] onto a
//! concrete gruvbox colour, and turns a line's **byte-offset** highlight spans
//! into a per-**display-column** colour map the [`crate::ui::textarea`] can
//! paint cell by cell.
//!
//! # Why this lives in the UI, not in `kopitiam-syntax`
//!
//! `kopitiam-syntax` is deliberately theme-blind: it says "this text is a
//! keyword", never "this text is red" — so the same lexer output can drive any
//! theme (see that crate's [`HighlightKind`] docs). Choosing *which* gruvbox
//! hue a keyword takes is a rendering decision, and rendering decisions belong
//! here, next to the [`Theme`] they read from.
//!
//! # Byte offsets → display columns
//!
//! A [`HighlightSpan`]'s `start`/`end` are byte offsets into the raw line
//! (matching `&line[start..end]` slicing). The textarea, though, paints in
//! **display columns** after tab expansion and unicode-width — a tab is one
//! byte and four cells, a CJK char is three bytes and two cells. So a colour
//! computed in byte units would drift right across any line containing either.
//! [`line_display_colors`] does the conversion once, producing a `Vec<Color>`
//! indexed by display column, which the renderer then reads by column with no
//! further arithmetic.
//!
//! # Colour choices
//!
//! The mapping follows Pavel Pertsev's gruvbox conventions (keywords red,
//! types yellow, strings green, numbers/constants purple, comments gray,
//! operators orange) — see [`crate::ui::theme`] for the palette's attribution.
//! These are the same semantic assignments the upstream gruvbox Vim theme
//! makes to Vim's syntax groups; only the *values* are shared, no code.

use kopitiam_syntax::{HighlightKind, HighlightSpan};
use ratatui::style::Color;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::ui::textarea::display_width;
use crate::ui::theme::Theme;

/// The gruvbox foreground colour a given [`HighlightKind`] paints in.
///
/// [`HighlightKind::Punctuation`] maps to the theme's default `fg`: gruvbox
/// does not tint ordinary brackets/commas differently from surrounding text,
/// and leaving them at `fg` also means [`line_display_colors`] can treat "no
/// span" and "punctuation" identically (both stay `fg`), which is exactly what
/// the renderer wants — it only repaints cells whose colour actually differs
/// from the base.
pub fn kind_color(kind: HighlightKind, theme: &Theme) -> Color {
    match kind {
        HighlightKind::Keyword => theme.red_bright,
        HighlightKind::Type => theme.yellow_bright,
        HighlightKind::String => theme.green_bright,
        HighlightKind::Escape => theme.orange_bright,
        HighlightKind::Comment => theme.gray,
        HighlightKind::Number => theme.purple_bright,
        HighlightKind::Function => theme.aqua_bright,
        HighlightKind::Operator => theme.orange_bright,
        HighlightKind::Punctuation => theme.fg,
        HighlightKind::Macro => theme.aqua_bright,
        HighlightKind::Attribute => theme.aqua,
        HighlightKind::Heading => theme.yellow_bright,
        HighlightKind::Emphasis => theme.orange_bright,
        HighlightKind::Link => theme.blue_bright,
        HighlightKind::CodeBlock => theme.green_bright,
    }
}

/// Builds a per-display-column colour map for `line`, honouring tab expansion
/// and wide characters exactly as the textarea renders them.
///
/// The returned vector has one entry per display column of the tab-expanded
/// line; a column untouched by any span (and every column a tab expands into)
/// keeps the theme's default `fg`. The renderer indexes this by
/// `scroll.left + on_screen_column` and only repaints cells whose colour is
/// not the default, so the base text pass already drew.
pub fn line_display_colors(line: &str, tabstop: usize, spans: &[HighlightSpan], theme: &Theme) -> Vec<Color> {
    let tabstop = tabstop.max(1);
    let total = display_width(line, tabstop);
    let mut colors = vec![theme.fg; total];
    let mut col = 0usize;
    for (byte, g) in line.grapheme_indices(true) {
        let width = if g == "\t" { (col / tabstop + 1) * tabstop - col } else { g.width() };
        // Tabs are whitespace and get no colour; only real graphemes are tinted.
        if g != "\t"
            && let Some(kind) = span_at(spans, byte)
        {
            let colour = kind_color(kind, theme);
            let end = (col + width).min(total);
            for cell in colors[col..end].iter_mut() {
                *cell = colour;
            }
        }
        col += width;
    }
    colors
}

/// The [`HighlightKind`] of the span covering byte offset `byte`, if any.
///
/// `kopitiam-syntax` emits spans in ascending, non-overlapping order, so the
/// first span whose half-open `[start, end)` contains `byte` is the answer; a
/// linear scan is more than fast enough for one line's worth of spans.
fn span_at(spans: &[HighlightSpan], byte: usize) -> Option<HighlightKind> {
    spans.iter().find(|s| s.start <= byte && byte < s.end).map(|s| s.kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        Theme::gruvbox_dark()
    }

    #[test]
    fn keyword_span_colours_only_its_columns() {
        // "let x" — a keyword span over bytes 0..3 ("let").
        let spans = [HighlightSpan { start: 0, end: 3, kind: HighlightKind::Keyword }];
        let colors = line_display_colors("let x", 4, &spans, &theme());
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], theme().red_bright);
        assert_eq!(colors[2], theme().red_bright);
        assert_eq!(colors[3], theme().fg, "the space after the keyword is default fg");
        assert_eq!(colors[4], theme().fg, "the identifier is uncovered, so default fg");
    }

    #[test]
    fn a_tab_before_a_span_shifts_the_colour_to_the_right_display_columns() {
        // "\tfn" — the tab is display columns 0..=3, so "fn" (bytes 1..3) must
        // colour display columns 4 and 5, not 1 and 2.
        let spans = [HighlightSpan { start: 1, end: 3, kind: HighlightKind::Keyword }];
        let colors = line_display_colors("\tfn", 4, &spans, &theme());
        assert_eq!(colors.len(), 6); // 4 (tab) + 2 (fn)
        for c in &colors[0..4] {
            assert_eq!(*c, theme().fg, "the tab cells are never tinted");
        }
        assert_eq!(colors[4], theme().red_bright);
        assert_eq!(colors[5], theme().red_bright);
    }

    #[test]
    fn a_wide_char_before_a_span_shifts_by_two_columns_per_char() {
        // "中x" where 'x' (byte 3..4) is a keyword: the CJK char is display
        // columns 0..=1, so 'x' colours display column 2.
        let spans = [HighlightSpan { start: 3, end: 4, kind: HighlightKind::Keyword }];
        let colors = line_display_colors("中x", 4, &spans, &theme());
        assert_eq!(colors.len(), 3); // 2 (中) + 1 (x)
        assert_eq!(colors[0], theme().fg);
        assert_eq!(colors[1], theme().fg);
        assert_eq!(colors[2], theme().red_bright);
    }

    #[test]
    fn punctuation_maps_to_default_fg_so_it_is_not_repainted() {
        assert_eq!(kind_color(HighlightKind::Punctuation, &theme()), theme().fg);
    }
}
