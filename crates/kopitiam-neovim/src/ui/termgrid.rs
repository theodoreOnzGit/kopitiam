//! Painting a [`vt100::Screen`] into a kvim window — the *view* half of
//! `:term`, the [`crate::termemu`] session being the model half.
//!
//! [`crate::ui::app::App`] calls [`paint_terminal`] once per frame for each
//! window whose buffer is a terminal, inside the session's
//! [`crate::termemu::TermSession::with_screen`] closure (so the screen is read
//! under the parser lock and the borrow never escapes). It walks the grid cell
//! by cell, maps each `vt100` cell's glyph + colours + attributes onto a
//! ratatui cell, and returns where the terminal cursor sits so the caller can
//! place the real terminal cursor there.
//!
//! # Why a free function, not a ratatui `Widget`
//!
//! A `Widget` would have to *own* the `&vt100::Screen` for the length of
//! `render_widget`, but the screen only exists borrowed, inside the mutex-guard
//! closure of `with_screen`. A plain function that takes `&Screen` +
//! `&mut Buffer` sidesteps that lifetime knot: the App calls it from *inside*
//! the closure, where the borrow is valid, and paints straight into the frame's
//! cell buffer. Same reason kvim paints diagnostics with a helper rather than a
//! widget — the data being painted is not owned by the paint call.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use crate::ui::theme::Theme;

/// Paint `screen` into `area` of the ratatui cell buffer `buf`, styled against
/// `theme`. Returns the terminal cursor's screen position `(x, y)` if it should
/// be shown, or `None` when the program has hidden it (`vt100`'s `hide_cursor`)
/// or it falls outside `area`.
///
/// Clipped to the smaller of `area` and the grid: if the window is momentarily
/// bigger than the pty (a resize in flight) the extra cells stay the editor
/// background; if it is smaller, the overflow is simply not drawn. Neither
/// case reads out of bounds.
pub fn paint_terminal(
    screen: &vt100::Screen,
    buf: &mut Buffer,
    area: Rect,
    theme: &Theme,
) -> Option<(u16, u16)> {
    let (grid_rows, grid_cols) = screen.size();
    let rows = grid_rows.min(area.height);
    let cols = grid_cols.min(area.width);

    // Fill the whole area with the terminal's default background first, so a
    // pty smaller than the window does not leave the editor's own theme
    // showing through around the edges.
    buf.set_style(area, Style::default().fg(theme.fg).bg(theme.bg));

    for row in 0..rows {
        for col in 0..cols {
            let Some(cell) = screen.cell(row, col) else { continue };

            // A wide glyph's second column is a "continuation" cell with no
            // contents of its own — ratatui already reserved that column when we
            // set the wide glyph's symbol, so skip it rather than overwrite it
            // with a blank (which would chop the wide glyph in half).
            if cell.is_wide_continuation() {
                continue;
            }

            let x = area.x + col;
            let y = area.y + row;
            let Some(target) = buf.cell_mut((x, y)) else { continue };

            let mut style = style_for(cell, theme);
            if cell.inverse() {
                style = style.add_modifier(Modifier::REVERSED);
            }

            let contents = cell.contents();
            if contents.is_empty() {
                // A blank cell: keep the space but honour the cell's colours, so
                // a coloured-but-empty region (a selection bar, a status line)
                // paints its background.
                target.set_symbol(" ");
            } else {
                target.set_symbol(&contents);
            }
            target.set_style(style);
        }
    }

    if screen.hide_cursor() {
        return None;
    }
    let (cy, cx) = screen.cursor_position();
    if cy < rows && cx < cols {
        Some((area.x + cx, area.y + cy))
    } else {
        None
    }
}

/// Paint a `[Process exited N]` banner across the last row of `area` — the
/// signal that a `:term` child has exited and the buffer is now just its final
/// output, kept open for scrollback. Matches neovim's own end-of-job line.
///
/// Drawn *over* the last grid row on purpose: the shell's parting output (a
/// prompt, `logout`) already sits there and the banner replacing it is the
/// clearest "this is done" cue. Styled reversed so it reads as a status strip,
/// not as more shell output. A no-op if `area` has zero height. `code` is the
/// child's exit code as reported by the pty layer (see
/// [`crate::termemu::TermSession::exit_code`]).
pub fn paint_exit_banner(buf: &mut Buffer, area: Rect, code: u32, theme: &Theme) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let y = area.y + area.height - 1;
    let label = format!("[Process exited {code}]");
    let style = Style::default().fg(theme.bg).bg(theme.fg).add_modifier(Modifier::BOLD);
    // Repaint the whole banner row in the banner background first, then write
    // the label — so it reads as one solid strip rather than a few words
    // floating over the old shell output.
    for col in 0..area.width {
        if let Some(target) = buf.cell_mut((area.x + col, y)) {
            target.set_symbol(" ");
            target.set_style(style);
        }
    }
    for (i, ch) in label.chars().enumerate() {
        let x = area.x + i as u16;
        if x >= area.x + area.width {
            break;
        }
        if let Some(target) = buf.cell_mut((x, y)) {
            let mut cell_buf = [0u8; 4];
            target.set_symbol(ch.encode_utf8(&mut cell_buf));
            target.set_style(style);
        }
    }
}

/// Map one `vt100` cell's colours + bold/italic/underline onto a ratatui
/// [`Style`]. Inverse is applied by the caller (it is a modifier, not a
/// colour). See [`vt_color`] for the colour mapping.
fn style_for(cell: &vt100::Cell, theme: &Theme) -> Style {
    let mut style = Style::default()
        .fg(vt_color(cell.fgcolor(), theme.fg, theme))
        .bg(vt_color(cell.bgcolor(), theme.bg, theme));
    if cell.bold() {
        style = style.add_modifier(Modifier::BOLD);
    }
    if cell.italic() {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if cell.underline() {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

/// Turn a `vt100::Color` into a ratatui [`Color`].
///
/// * `Default` → `fallback` (the theme's fg for a foreground, bg for a
///   background) so the terminal's "default colour" blends with kvim's theme
///   rather than snapping to a hard black/white.
/// * `Idx(i)` → the ANSI 256-colour palette index, which the host terminal
///   renders with its own palette (so a user's terminal theme still applies).
/// * `Rgb(r, g, b)` → a true-colour cell, passed straight through.
fn vt_color(color: vt100::Color, fallback: Color, _theme: &Theme) -> Color {
    match color {
        vt100::Color::Default => fallback,
        vt100::Color::Idx(i) => Color::Indexed(i),
        vt100::Color::Rgb(r, g, b) => Color::Rgb(r, g, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::theme::Theme;

    /// Feed a parser some plain text and paint it; the glyphs must land in the
    /// ratatui buffer at the right cells.
    #[test]
    fn plain_text_paints_into_the_buffer() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        parser.process(b"hello");
        let theme = Theme::gruvbox_dark();
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);

        let _ = paint_terminal(parser.screen(), &mut buf, area, &theme);

        let painted: String = (0..5).map(|x| buf.cell((x, 0)).unwrap().symbol().to_string()).collect();
        assert_eq!(painted, "hello");
    }

    /// The cursor position comes back so the caller can place the real cursor.
    #[test]
    fn cursor_position_is_returned_offset_by_area() {
        let mut parser = vt100::Parser::new(24, 80, 0);
        parser.process(b"abc");
        let theme = Theme::gruvbox_dark();
        // Offset the area so we prove the (x, y) is area-relative, not grid 0,0.
        let area = Rect::new(3, 2, 40, 10);
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 24));

        let cursor = paint_terminal(parser.screen(), &mut buf, area, &theme);
        // After "abc" the cursor sits at grid col 3, row 0 → area (3+3, 2+0).
        assert_eq!(cursor, Some((6, 2)));
    }

    /// A grid bigger than the window must not read or write out of bounds.
    #[test]
    fn oversized_grid_is_clipped_to_area() {
        let mut parser = vt100::Parser::new(50, 200, 0);
        parser.process(b"x");
        let theme = Theme::gruvbox_dark();
        let area = Rect::new(0, 0, 10, 5);
        let mut buf = Buffer::empty(area);
        // Must not panic.
        let _ = paint_terminal(parser.screen(), &mut buf, area, &theme);
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "x");
    }

    #[test]
    fn exit_banner_paints_on_the_last_row() {
        let theme = Theme::gruvbox_dark();
        let area = Rect::new(0, 0, 40, 5);
        let mut buf = Buffer::empty(area);
        paint_exit_banner(&mut buf, area, 0, &theme);
        // Bottom row (y = 4) starts with the banner text.
        let row: String = (0..40).map(|x| buf.cell((x, 4)).unwrap().symbol().to_string()).collect();
        assert!(row.starts_with("[Process exited 0]"), "banner row was {row:?}");
        // A non-zero code is shown verbatim.
        let mut buf2 = Buffer::empty(area);
        paint_exit_banner(&mut buf2, area, 130, &theme);
        let row2: String = (0..40).map(|x| buf2.cell((x, 4)).unwrap().symbol().to_string()).collect();
        assert!(row2.starts_with("[Process exited 130]"), "banner row was {row2:?}");
    }

    #[test]
    fn exit_banner_on_zero_height_is_a_noop() {
        let theme = Theme::gruvbox_dark();
        // Zero-height area must not panic / read out of bounds.
        let mut buf = Buffer::empty(Rect::new(0, 0, 10, 1));
        paint_exit_banner(&mut buf, Rect::new(0, 0, 10, 0), 0, &theme);
    }

    #[test]
    fn default_colours_fall_back_to_theme() {
        let theme = Theme::gruvbox_dark();
        assert_eq!(vt_color(vt100::Color::Default, theme.fg, &theme), theme.fg);
        assert_eq!(vt_color(vt100::Color::Idx(4), theme.fg, &theme), Color::Indexed(4));
        assert_eq!(vt_color(vt100::Color::Rgb(1, 2, 3), theme.fg, &theme), Color::Rgb(1, 2, 3));
    }
}
