//! The harpoon quick menu (`<leader><Esc>`) — a small floating, editable list
//! of the project's marks.
//!
//! # What this is, and what it is not
//!
//! Upstream harpoon's quick menu is a real Neovim buffer: it opens in a float,
//! you edit it like any buffer (delete a line, reorder lines), and on `:w` the
//! new order is written back to the mark list. kvim's buffer model is narrower
//! (see [`crate::ui::overlay`] for why panels are not buffers), so this menu is
//! a *purpose-built list widget* that offers the same **behaviour** without
//! pretending to be a text buffer:
//!
//! * `j`/`k` (and the arrows) move the selection, wrapping;
//! * a digit `1`..`9` jumps straight to that slot, the way harpoon's numbered
//!   lines invite;
//! * `<CR>` jumps to the selected mark — landing at the *saved cursor*, which
//!   is the whole point of a harpoon mark (see [`crate::plugins::harpoon`]);
//! * `d` or `x` deletes the selected line, i.e. removes that mark;
//! * `q`/`<Esc>` closes.
//!
//! Reordering — harpoon's other in-buffer edit — is deliberately left to a
//! follow-up; delete is the edit the task asked for, and doing one thing that
//! works beats two that half-work.
//!
//! # Why the panel holds a snapshot, not the live [`Harpoon`]
//!
//! Like every overlay in kvim, this one never reaches into the app's state (see
//! [`OverlayOutcome`]'s docs). It is built from a *snapshot* of the marks and
//! reports what it wants done — [`OverlayOutcome::OpenAt`] to jump,
//! [`OverlayOutcome::HarpoonRemove`] to delete. A delete is applied to the
//! snapshot **and** reported, so the panel's own view and the canonical
//! [`Harpoon`] stay in lockstep (both start identical and apply the same index
//! removals) without the panel ever borrowing the app. The menu is modal —
//! nothing else mutates the marks while it is up — so the two cannot drift.
//!
//! [`Harpoon`]: crate::plugins::harpoon::Harpoon

use ratatui::buffer::Buffer as Surface;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, Borders, Widget};
use ratatui::Frame;

use crate::core::Position;
use crate::icons::IconSet;
use crate::plugins::harpoon::Mark;
use crate::ui::event::{Key, KeyPress};
use crate::ui::overlay::OverlayOutcome;
use crate::ui::theme::Theme;

/// The harpoon quick menu: a floating, numbered list of marks with a selection
/// you move, jump from, and delete from. Takes focus while open — the editor
/// never sees these keys.
pub struct HarpoonMenuPanel {
    /// The marks as they were when the menu opened, kept in lockstep with the
    /// canonical list by mirroring deletes (see the module docs).
    marks: Vec<Mark>,
    /// The highlighted row, always clamped into `0..marks.len()` (or `0` when
    /// empty).
    selected: usize,
    /// First visible row — resolved each render against the height the panel is
    /// finally drawn at, the same scroll-into-view trick the picker uses.
    scroll_top: usize,
}

impl HarpoonMenuPanel {
    /// Builds the menu over `marks`, selection on the first row.
    pub fn new(marks: Vec<Mark>) -> Self {
        Self { marks, selected: 0, scroll_top: 0 }
    }

    /// How many marks the menu is showing. Exposed for tests.
    pub fn len(&self) -> usize {
        self.marks.len()
    }

    /// Whether the menu is empty (every mark deleted, or none to begin with).
    pub fn is_empty(&self) -> bool {
        self.marks.is_empty()
    }

    /// Feeds one key while the menu has focus.
    pub fn handle_key(&mut self, key: KeyPress) -> OverlayOutcome {
        match key.key {
            Key::Escape => OverlayOutcome::Close,
            Key::Char('q') => OverlayOutcome::Close,

            Key::Char('j') | Key::Down => {
                self.select_next();
                OverlayOutcome::Consumed
            }
            Key::Char('k') | Key::Up => {
                self.select_prev();
                OverlayOutcome::Consumed
            }

            // A digit jumps straight to that 1-based slot, if it exists.
            Key::Char(c @ '1'..='9') => {
                let slot = c as usize - '0' as usize; // 1..=9
                match self.marks.get(slot - 1) {
                    Some(mark) => open_at(mark),
                    None => OverlayOutcome::Ignored,
                }
            }

            // Confirm the selected mark.
            Key::Enter => match self.marks.get(self.selected) {
                Some(mark) => open_at(mark),
                // Nothing to jump to (an emptied list) — behave like `<Esc>`.
                None => OverlayOutcome::Close,
            },

            // Delete the selected line = remove that mark.
            Key::Char('d') | Key::Char('x') => self.delete_selected(),

            _ => OverlayOutcome::Ignored,
        }
    }

    /// Moves the selection down one, wrapping at the bottom.
    fn select_next(&mut self) {
        if self.marks.is_empty() {
            return;
        }
        self.selected = (self.selected + 1) % self.marks.len();
    }

    /// Moves the selection up one, wrapping at the top.
    fn select_prev(&mut self) {
        if self.marks.is_empty() {
            return;
        }
        self.selected = (self.selected + self.marks.len() - 1) % self.marks.len();
    }

    /// Removes the selected mark from the snapshot and reports the same index so
    /// the app drops it from the canonical list too. The selection is clamped
    /// back onto a valid row (or `0` when the list empties). A no-op on an
    /// already-empty list.
    fn delete_selected(&mut self) -> OverlayOutcome {
        if self.selected >= self.marks.len() {
            return OverlayOutcome::Ignored;
        }
        let index = self.selected;
        self.marks.remove(index);
        if self.selected >= self.marks.len() {
            self.selected = self.marks.len().saturating_sub(1);
        }
        OverlayOutcome::HarpoonRemove(index)
    }

    /// Draws the menu into `rect`. Returns `None` — the menu paints its own
    /// selection highlight and has no separate text caret to place.
    pub fn render(
        &mut self,
        frame: &mut Frame,
        rect: Rect,
        theme: &Theme,
        _icons: IconSet,
        _focused: bool,
    ) -> Option<(u16, u16)> {
        if rect.width < 4 || rect.height < 3 {
            return None;
        }
        let list_h = rect.height.saturating_sub(2) as usize; // top+bottom border
        self.scroll_into_view(list_h);
        frame.render_widget(HarpoonMenuView { panel: self, theme }, rect);
        None
    }

    /// Keeps the selected row within the `height`-row window, scrolling the
    /// minimum needed — the same clamp the picker and file tree use.
    fn scroll_into_view(&mut self, height: usize) {
        if height == 0 {
            self.scroll_top = 0;
            return;
        }
        if self.selected < self.scroll_top {
            self.scroll_top = self.selected;
        } else if self.selected >= self.scroll_top + height {
            self.scroll_top = self.selected + 1 - height;
        }
        let max_top = self.marks.len().saturating_sub(height);
        self.scroll_top = self.scroll_top.min(max_top);
    }
}

/// Builds the "jump to this mark" outcome, translating the engine's stored
/// 0-based `line`/`col` into a [`Position`] for the editor to restore.
fn open_at(mark: &Mark) -> OverlayOutcome {
    OverlayOutcome::OpenAt { path: mark.path.clone(), cursor: Position::new(mark.line, mark.col) }
}

/// The menu as a ratatui widget, rebuilt each frame from the panel — the same
/// borrow-don't-own shape as [`crate::ui::picker`]'s view.
struct HarpoonMenuView<'a> {
    panel: &'a HarpoonMenuPanel,
    theme: &'a Theme,
}

impl Widget for HarpoonMenuView<'_> {
    fn render(self, area: Rect, buf: &mut Surface) {
        if area.width < 4 || area.height < 3 {
            return;
        }
        let bg = self.theme.bg1;
        let title = format!(" Harpoon ({}) ", self.panel.marks.len());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.bg3).bg(bg))
            .title(title)
            .title_style(Style::default().fg(self.theme.yellow_bright).bg(bg));
        let inner = block.inner(area);
        buf.set_style(area, Style::default().bg(bg));
        block.render(area, buf);
        if inner.width == 0 || inner.height == 0 {
            return;
        }

        // An empty menu says so rather than painting a blank box — the marks
        // are session-scoped, so "nothing here" is a normal, common state.
        if self.panel.marks.is_empty() {
            let style = Style::default().fg(self.theme.bg3).bg(bg).add_modifier(Modifier::ITALIC);
            buf.set_stringn(inner.x, inner.y, " no marks yet lah — press <leader>b to mark one", inner.width as usize, style);
            return;
        }

        let list_h = inner.height as usize;
        for row in 0..list_h {
            let idx = self.panel.scroll_top + row;
            let Some(mark) = self.panel.marks.get(idx) else { break };
            let y = inner.y + row as u16;
            let is_selected = idx == self.panel.selected;

            let style = if is_selected {
                Style::default().fg(self.theme.bg).bg(self.theme.yellow_bright).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(self.theme.fg).bg(bg)
            };
            if is_selected {
                let full = Rect { x: inner.x, y, width: inner.width, height: 1 };
                buf.set_style(full, style);
            }

            // "N  path:line" — the 1-based slot number harpoon labels its lines
            // with, then the file and the 1-based line the mark will land on.
            let label = format!(" {}  {}:{}", idx + 1, mark.path.display(), mark.line + 1);
            buf.set_stringn(inner.x, y, &label, inner.width as usize, style);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::event::{Key, Modifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::path::PathBuf;

    fn marks() -> Vec<Mark> {
        vec![
            Mark::new(PathBuf::from("/p/src/main.rs"), 10, 4),
            Mark::new(PathBuf::from("/p/src/lib.rs"), 0, 0),
            Mark::new(PathBuf::from("/p/README.md"), 3, 2),
        ]
    }

    fn press(c: char) -> KeyPress {
        KeyPress { key: Key::Char(c), mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    fn special(key: Key) -> KeyPress {
        KeyPress { key, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    #[test]
    fn enter_jumps_to_the_selected_mark_at_its_saved_cursor() {
        let mut panel = HarpoonMenuPanel::new(marks());
        panel.handle_key(press('j')); // -> src/lib.rs
        let out = panel.handle_key(special(Key::Enter));
        assert_eq!(out, OverlayOutcome::OpenAt { path: PathBuf::from("/p/src/lib.rs"), cursor: Position::new(0, 0) });
    }

    #[test]
    fn a_digit_jumps_straight_to_that_slot() {
        let mut panel = HarpoonMenuPanel::new(marks());
        let out = panel.handle_key(press('3'));
        assert_eq!(
            out,
            OverlayOutcome::OpenAt { path: PathBuf::from("/p/README.md"), cursor: Position::new(3, 2) },
            "slot 3 is the third mark, and it carries its own cursor"
        );
        // An out-of-range slot does nothing rather than jumping wrongly.
        assert_eq!(panel.handle_key(press('9')), OverlayOutcome::Ignored);
    }

    #[test]
    fn d_deletes_the_selected_mark_and_reports_its_index() {
        let mut panel = HarpoonMenuPanel::new(marks());
        panel.handle_key(press('j')); // select index 1 (lib.rs)
        let out = panel.handle_key(press('d'));
        assert_eq!(out, OverlayOutcome::HarpoonRemove(1));
        assert_eq!(panel.len(), 2, "the snapshot shrank in lockstep with the report");
    }

    #[test]
    fn deleting_the_last_row_pulls_the_selection_back_into_range() {
        let mut panel = HarpoonMenuPanel::new(marks());
        panel.handle_key(press('k')); // wrap up to the last row (index 2)
        assert_eq!(panel.handle_key(press('d')), OverlayOutcome::HarpoonRemove(2));
        // Selection must not dangle past the end; a following <CR> jumps to a
        // real (now-last) mark rather than closing.
        let out = panel.handle_key(special(Key::Enter));
        assert!(matches!(out, OverlayOutcome::OpenAt { .. }), "selection stayed valid, got {out:?}");
    }

    #[test]
    fn selection_wraps_both_ways() {
        let mut panel = HarpoonMenuPanel::new(marks());
        panel.handle_key(press('k')); // wrap from 0 up to last (2)
        assert_eq!(panel.selected, 2);
        panel.handle_key(press('j')); // wrap from last back to 0
        assert_eq!(panel.selected, 0);
    }

    #[test]
    fn escape_and_q_both_close() {
        let mut panel = HarpoonMenuPanel::new(marks());
        assert_eq!(panel.handle_key(special(Key::Escape)), OverlayOutcome::Close);
        assert_eq!(panel.handle_key(press('q')), OverlayOutcome::Close);
    }

    #[test]
    fn the_box_paints_a_numbered_list_of_marks() {
        let theme = Theme::gruvbox_dark();
        let mut panel = HarpoonMenuPanel::new(marks());
        let mut terminal = Terminal::new(TestBackend::new(50, 10)).unwrap();
        terminal
            .draw(|frame| {
                let area = frame.area();
                panel.render(frame, area, &theme, IconSet::Ascii, true);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text: String = (0..10)
            .map(|y| (0..50).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Harpoon (3)"), "the titled count must paint:\n{text}");
        assert!(text.contains("1  /p/src/main.rs:11"), "slot 1 with 1-based line must paint:\n{text}");
        assert!(text.contains("main.rs"), "a marked file must be listed:\n{text}");
    }

    #[test]
    fn an_empty_menu_says_so() {
        let theme = Theme::gruvbox_dark();
        let mut panel = HarpoonMenuPanel::new(Vec::new());
        assert!(panel.is_empty());
        let mut terminal = Terminal::new(TestBackend::new(50, 6)).unwrap();
        terminal
            .draw(|frame| {
                let area = frame.area();
                panel.render(frame, area, &theme, IconSet::Ascii, true);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text: String = (0..6)
            .map(|y| (0..50).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("no marks"), "an empty menu must explain itself:\n{text}");
    }
}
