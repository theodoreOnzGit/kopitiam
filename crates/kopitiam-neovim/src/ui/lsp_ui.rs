//! Small floating panels for the interactive LSP features: the hover popup,
//! the references (location) list, and the rename input.
//!
//! These are the same shape as [`crate::ui::whichkey`] — a bordered gruvbox box
//! painted as a final render pass — but, unlike which-key, they take focus (a
//! reference list navigates with `j`/`k`, the rename box reads text). The focus
//! bookkeeping lives in [`crate::ui::app::App`], the same way hop's does; this
//! module owns only the drawing.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Block, Borders, Clear, Widget},
};

use crate::ui::theme::Theme;

/// A bordered floating box of text lines, optionally with one row highlighted
/// (the selected reference in a location list). Centred-ish: the caller passes
/// the rectangle.
pub struct InfoBox<'a> {
    pub title: &'a str,
    pub lines: &'a [String],
    /// The highlighted row (for a navigable list), or `None` for a plain popup.
    pub selected: Option<usize>,
    pub theme: &'a Theme,
    /// Scroll offset: the first `lines` index shown, so a long hover/list can be
    /// paged. `0` for the common short case.
    pub scroll: usize,
}

impl Widget for InfoBox<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 4 || area.height < 3 {
            return;
        }
        let bg = self.theme.bg1;
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.bg3).bg(bg))
            .title(self.title)
            .title_style(Style::default().fg(self.theme.yellow_bright).bg(bg));
        let inner = block.inner(area);
        // Wipe the cells under the box before we paint. `set_style` alone only
        // change the bg colour but keep whatever symbol already sitting there —
        // the buffer text underneath — so the text bleed through and the popup
        // look see-through. `Clear` reset every cell to blank first, then we lay
        // our gruvbox bg on top, so the box come out fully opaque.
        Clear.render(area, buf);
        buf.set_style(area, Style::default().bg(bg));
        block.render(area, buf);

        for row in 0..inner.height {
            let idx = self.scroll + row as usize;
            let Some(line) = self.lines.get(idx) else { break };
            let y = inner.y + row;
            let selected = self.selected == Some(idx);
            let style = if selected {
                Style::default().fg(self.theme.bg).bg(self.theme.yellow_bright).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(self.theme.fg).bg(bg)
            };
            // Paint the whole inner width so a selected row's highlight spans
            // the box, not just the text.
            if selected {
                let full = Rect { x: inner.x, y, width: inner.width, height: 1 };
                buf.set_style(full, style);
            }
            buf.set_stringn(inner.x, y, line, inner.width as usize, style);
        }
    }
}

/// Computes a centred rectangle `width` × `height` inside `area`, clamped to
/// fit. Used to place hover/reference popups over the buffer.
pub fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width.saturating_sub(2)).max(4);
    let h = height.min(area.height.saturating_sub(2)).max(3);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    #[test]
    fn info_box_paints_its_lines_and_highlights_the_selection() {
        let theme = Theme::gruvbox_dark();
        let lines = vec!["src/lib.rs:1:8".to_string(), "src/main.rs:4:5".to_string()];
        let mut terminal = Terminal::new(TestBackend::new(40, 6)).unwrap();
        terminal
            .draw(|frame| {
                let rect = centered_rect(frame.area(), 30, 4);
                frame.render_widget(
                    InfoBox { title: "references", lines: &lines, selected: Some(1), theme: &theme, scroll: 0 },
                    rect,
                );
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text: String = (0..6)
            .map(|y| (0..40).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("src/lib.rs:1:8"), "{text}");
        assert!(text.contains("src/main.rs:4:5"), "{text}");
        // The selected (second) row is painted on the yellow highlight bg.
        let has_selected_bg = (0..40)
            .flat_map(|x| (0..6).map(move |y| (x, y)))
            .any(|(x, y)| {
                let c = buf.cell((x, y)).unwrap();
                c.symbol() == "m" && c.style().bg == Some(theme.yellow_bright)
            });
        assert!(has_selected_bg, "the selected reference row should carry the highlight background");
    }

    /// The exact bug this widget's `Clear` fix is about (InfoBox is reused by
    /// hover, references, quickfix, rename and `:help`): paint the whole screen
    /// with `X` (stand-in for the buffer text under the box), drop the InfoBox on
    /// top, then assert not one `X` survive inside its rect and every cell there
    /// carry an opaque bg.
    #[test]
    fn info_box_is_opaque_no_buffer_text_bleeds_through() {
        let theme = Theme::gruvbox_dark();
        let lines = vec!["src/lib.rs:1:8".to_string()];
        let area = Rect { x: 0, y: 0, width: 40, height: 8 };
        let rect = centered_rect(area, 30, 4);
        let mut terminal = Terminal::new(TestBackend::new(area.width, area.height)).unwrap();
        terminal
            .draw(|frame| {
                let fill = "X".repeat(area.width as usize);
                for y in 0..area.height {
                    frame.buffer_mut().set_string(0, y, &fill, Style::default());
                }
                frame.render_widget(
                    InfoBox { title: "hover", lines: &lines, selected: None, theme: &theme, scroll: 0 },
                    rect,
                );
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        for y in rect.top()..rect.bottom() {
            for x in rect.left()..rect.right() {
                let c = buf.cell((x, y)).unwrap();
                assert_ne!(c.symbol(), "X", "buffer text bled through the info box at ({x},{y})");
                assert!(c.style().bg.is_some(), "cell ({x},{y}) inside the info box is not opaque");
            }
        }
    }
}
