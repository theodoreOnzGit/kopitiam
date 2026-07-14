//! The which-key popup: a floating panel that lists the continuations of a
//! pending key prefix, so pressing `<leader>` (Space) or `g` shows what each
//! next key does.
//!
//! # Passive, not a mode
//!
//! which-key never intercepts a key. The editor's keymap engine still resolves
//! the full sequence (`<leader>e` → toggle file tree); this panel only *renders*
//! the choices while a prefix is buffered ([`crate::editor::Editor::which_key`]).
//! That is exactly Neovim's which-key: a heads-up display over the editor, not a
//! layer in front of it. So [`crate::ui::app::App`] draws it as a final pass and
//! feeds keys to the editor unchanged — there is no focus handoff, unlike an
//! [`crate::ui::overlay::Overlay`].
//!
//! # Styling
//!
//! Docked to the bottom of the window area (Neovim which-key's default
//! position), a bordered box in gruvbox: the key label in bright yellow, a
//! leaf's description in the default foreground, and a `+group` prefix in aqua
//! so "leads to more keys" reads differently from "does a thing."

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Block, Borders, Widget},
};

use crate::ui::theme::Theme;

/// One row of the popup: the next key's label and where it leads. Mirrors
/// [`crate::editor::WhichKeyItem`], kept as a UI-local type so the widget layer
/// does not depend on the editor's enum (the bootstrap adapter maps between
/// them, the same pattern every other `EditorHost` method uses).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhichKeyRow {
    pub keys: String,
    pub desc: String,
    pub is_group: bool,
}

/// The height (including both border rows) a popup of `n` rows needs, capped so
/// it never eats more than it should of a `max_area_height`-tall window.
pub fn popup_height(rows: usize, max_area_height: u16) -> u16 {
    let wanted = rows as u16 + 2; // borders top + bottom
    wanted.min(max_area_height.max(3)).max(3)
}

/// Computes the bottom-docked rectangle the popup occupies inside `area`.
pub fn popup_rect(area: Rect, rows: usize) -> Rect {
    let height = popup_height(rows, area.height);
    Rect { x: area.x, y: area.y + area.height.saturating_sub(height), width: area.width, height }
}

/// The which-key popup widget.
pub struct WhichKey<'a> {
    pub rows: &'a [WhichKeyRow],
    pub theme: &'a Theme,
}

impl Widget for WhichKey<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 4 || area.height < 3 {
            return;
        }
        let border_style = Style::default().fg(self.theme.bg3).bg(self.theme.bg1);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title("which-key")
            .title_style(Style::default().fg(self.theme.yellow_bright).bg(self.theme.bg1));
        let inner = block.inner(area);
        // Fill the box background first so it reads as a panel over the text.
        buf.set_style(area, Style::default().bg(self.theme.bg1));
        block.render(area, buf);

        let key_style = Style::default().fg(self.theme.yellow_bright).bg(self.theme.bg1).add_modifier(Modifier::BOLD);
        let sep_style = Style::default().fg(self.theme.gray).bg(self.theme.bg1);
        for (i, row) in self.rows.iter().enumerate() {
            let y = inner.y + i as u16;
            if y >= inner.y + inner.height {
                break;
            }
            let mut x = inner.x + 1;
            // key label
            let label = &row.keys;
            buf.set_stringn(x, y, label, inner.width as usize, key_style);
            x += label.chars().count() as u16 + 1;
            // arrow separator
            let arrow = "→ ";
            buf.set_stringn(x, y, arrow, inner.width as usize, sep_style);
            x += arrow.chars().count() as u16;
            // description
            let desc_style = if row.is_group {
                Style::default().fg(self.theme.aqua_bright).bg(self.theme.bg1)
            } else {
                Style::default().fg(self.theme.fg).bg(self.theme.bg1)
            };
            let remaining = (inner.x + inner.width).saturating_sub(x) as usize;
            buf.set_stringn(x, y, &row.desc, remaining, desc_style);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    #[test]
    fn a_row_paints_its_key_and_description() {
        let rows = vec![
            WhichKeyRow { keys: "e".into(), desc: "Toggle file explorer".into(), is_group: false },
            WhichKeyRow { keys: "g".into(), desc: "+2 more".into(), is_group: true },
        ];
        let theme = Theme::gruvbox_dark();
        let mut terminal = Terminal::new(TestBackend::new(40, 5)).unwrap();
        terminal
            .draw(|frame| {
                let rect = popup_rect(frame.area(), rows.len());
                frame.render_widget(WhichKey { rows: &rows, theme: &theme }, rect);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text: Vec<String> = (0..5)
            .map(|y| (0..40).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect())
            .collect();
        let joined = text.join("\n");
        assert!(joined.contains("Toggle file explorer"), "screen was:\n{joined}");
        assert!(joined.contains("+2 more"), "screen was:\n{joined}");
        // The key label 'e' is painted in the keyword-yellow.
        let e_cell = (0..40)
            .flat_map(|x| (0..5).map(move |y| (x, y)))
            .find(|&(x, y)| buf.cell((x, y)).unwrap().symbol() == "e" && buf.cell((x, y)).unwrap().style().fg == Some(theme.yellow_bright));
        assert!(e_cell.is_some(), "the key label should be painted in bright yellow");
    }
}
