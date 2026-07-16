//! The fuzzy-picker overlay — telescope's floating finder, kvim-style, lah.
//!
//! # One panel, three sources (the telescope split)
//!
//! telescope.nvim is one widget wearing many hats: `find_files`, `buffers` and
//! `help_tags` are the *same* prompt-plus-list, differing only in what fills the
//! list and what happens on `<CR>`. kvim keep exactly that shape. The matching
//! brain already live in [`crate::plugins::picker::Picker`] (a generic scorer over
//! anything [`Searchable`]); this module is only its *face* — the box you type
//! into, the scrolling candidate list, and the focus discipline that keeps your
//! keystrokes out of the buffer underneath while the picker is up.
//!
//! So the three pickers (`\ff`, `\fb`, `\fh`) are **not** three types. They are
//! one [`PickerPanel`] built over one `Picker<`[`PickRow`]`>`, where a row is
//! nothing but "the text to fuzzy-match" plus "what to do when chosen"
//! ([`PickAction`]). Add a fourth source later (LSP symbols, git files, whatever)
//! and it is a new way to build the rows — not a new widget, not new key
//! handling, not new rendering. That is the whole point of doing it this way
//! instead of writing `FilePickerPanel`/`BufferPickerPanel`/`HelpPickerPanel` and
//! copying the prompt-and-list code three times, then four.
//!
//! # Why the panel drives the query, not the engine
//!
//! [`crate::plugins::picker::Picker`] own the *scored, ranked* view of the query
//! but not the *editing* of the query text (it takes a finished `&str` and
//! re-scores). Line-editing — a char appended, a `<BS>` chopping the last one —
//! is a UI concern, so the panel keep its own [`PickerPanel::query`] string,
//! edit that, and hand the result to the engine. Same division the command line
//! draw between `cmdline` (the text editor) and `ex` (the parser): the thing that
//! *reasons* about the input never also *collects* it.

use ratatui::buffer::Buffer as Surface;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, Borders, Clear, Widget};
use ratatui::Frame;
use std::path::PathBuf;

use crate::core::BufferId;
use crate::icons::IconSet;
use crate::plugins::picker::{Picker, Searchable};
use crate::ui::event::{Key, KeyPress};
use crate::ui::overlay::OverlayOutcome;
use crate::ui::theme::Theme;

/// What confirming a row should make the editor do.
///
/// Carried on every [`PickRow`] so the panel itself never has to know *which*
/// picker it is — it just reads the selected row's action and turns it into the
/// matching [`OverlayOutcome`]. The three variants are the three pickers'
/// payloads, and nothing else in the panel branches on the picker kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickAction {
    /// `\ff`: open this (absolute) path in the current window.
    OpenFile(PathBuf),
    /// `\fb`: switch the active buffer to this id.
    SwitchBuffer(BufferId),
    /// `\fh`: run `:help <topic>` and land on that section.
    OpenHelp(String),
}

/// One candidate: the text the fuzzy matcher scores against, and what to do
/// when it is chosen. See the module docs — a "source" is just a `Vec<PickRow>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickRow {
    /// What is shown in the list *and* what `nucleo` matches the query against.
    /// For files this is the repo-relative path (what a human types), for
    /// buffers the `id + name + modified` line, for help the topic tag.
    pub label: String,
    /// What happens on `<CR>`.
    pub action: PickAction,
}

impl PickRow {
    pub fn new(label: impl Into<String>, action: PickAction) -> Self {
        Self { label: label.into(), action }
    }
}

impl Searchable for PickRow {
    fn search_text(&self) -> &str {
        &self.label
    }
}

/// A floating fuzzy picker: a prompt line you type into, and a scrolling,
/// fuzzy-filtered list of candidates beneath it. Takes focus while open (the
/// editor never sees these keys) and reports what the chosen row wants done via
/// an [`OverlayOutcome`], never touching the editor itself — the same
/// hands-off discipline the file tree keep.
pub struct PickerPanel {
    /// The box title, e.g. `"Find Files"`. Static per picker kind.
    title: &'static str,
    /// The scoring engine. Rebuilt-per-keystroke scoring lives here.
    picker: Picker<PickRow>,
    /// The prompt text as edited so far. The panel owns the *editing*; the
    /// engine owns the *scoring* (see module docs).
    query: String,
    /// First visible match index — recomputed each render against the height the
    /// panel is finally drawn at, so the selection is always on screen. Same
    /// scroll-into-view trick the file tree and reference list use.
    scroll_top: usize,
}

impl PickerPanel {
    /// Builds a picker titled `title` over `rows`, initially unfiltered (the
    /// whole list shown, selection on the first row) — telescope opens showing
    /// everything, not an empty list waiting for you to type.
    pub fn new(title: &'static str, rows: Vec<PickRow>) -> Self {
        Self { title, picker: Picker::new(rows), query: String::new(), scroll_top: 0 }
    }

    /// How many candidates currently match. Exposed for tests asserting that
    /// typing narrows the list.
    pub fn match_count(&self) -> usize {
        self.picker.matches().len()
    }

    /// Feeds one key while the picker has focus.
    ///
    /// The bindings mirror telescope's insert-mode defaults, which is what the
    /// maintainer's fingers expect:
    ///
    /// * a printable char (no ctrl/alt) extends the query and re-filters;
    /// * `<BS>` chops the last query char;
    /// * `<C-n>`/`<Down>` and `<C-p>`/`<Up>` move the selection (wrapping);
    /// * `<CR>` confirms the highlighted row;
    /// * `<Esc>` and `<C-c>` cancel.
    ///
    /// Everything else is [`OverlayOutcome::Ignored`] — a picker is a modal
    /// prompt, so an unbound key does nothing rather than leaking to the editor.
    pub fn handle_key(&mut self, key: KeyPress) -> OverlayOutcome {
        let ctrl = key.mods.ctrl;
        match key.key {
            // Cancel. `<C-c>` is telescope's "get me out"; plain `<Esc>` too.
            Key::Escape => OverlayOutcome::Close,
            Key::Char('c') if ctrl => OverlayOutcome::Close,

            // Selection movement. `<C-n>`/`<C-p>` are the telescope defaults;
            // the arrow keys do the same, for people who reach for them.
            Key::Char('n') if ctrl => {
                self.picker.select_next();
                OverlayOutcome::Consumed
            }
            Key::Char('p') if ctrl => {
                self.picker.select_prev();
                OverlayOutcome::Consumed
            }
            Key::Down => {
                self.picker.select_next();
                OverlayOutcome::Consumed
            }
            Key::Up => {
                self.picker.select_prev();
                OverlayOutcome::Consumed
            }

            // Confirm.
            Key::Enter => match self.picker.confirm() {
                Some(row) => match &row.action {
                    PickAction::OpenFile(path) => OverlayOutcome::PickPath(path.clone()),
                    PickAction::SwitchBuffer(id) => OverlayOutcome::PickBuffer(*id),
                    PickAction::OpenHelp(topic) => OverlayOutcome::PickHelp(topic.clone()),
                },
                // `<CR>` on an empty result list just closes, the way telescope
                // does — there is nothing to open, so pretend `<Esc>`.
                None => OverlayOutcome::Close,
            },

            // Query editing.
            Key::Backspace => {
                if self.query.pop().is_some() {
                    self.picker.set_query(&self.query);
                    OverlayOutcome::Consumed
                } else {
                    // Empty prompt + `<BS>` is a no-op in telescope, not a close.
                    OverlayOutcome::Ignored
                }
            }
            Key::Char(c) if !ctrl && !key.mods.alt => {
                self.query.push(c);
                self.picker.set_query(&self.query);
                OverlayOutcome::Consumed
            }

            _ => OverlayOutcome::Ignored,
        }
    }

    /// Draws the picker into `rect` and returns where the terminal cursor
    /// should sit (at the end of the prompt), so it blinks where you are typing.
    /// `&mut self` because the scroll offset is resolved against the height the
    /// panel is finally drawn at, exactly like the file tree.
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
        // The candidate list occupies the inner box minus the prompt row and its
        // separator row. Compute it now so scroll-into-view uses the true height.
        let inner_h = rect.height.saturating_sub(2) as usize; // top+bottom border
        let list_h = inner_h.saturating_sub(2); // prompt row + separator row
        self.scroll_into_view(list_h);

        // The prompt caret: one column past `> ` plus the query graphemes. `rect`
        // is the outer box, so +1 for the left border and +1 for the prompt row.
        let prompt_prefix = 2u16; // "> "
        let caret_col = rect.x + 1 + prompt_prefix + self.query.chars().count() as u16;
        let caret_col = caret_col.min(rect.x + rect.width.saturating_sub(2));
        let cursor = (rect.y + 1, caret_col);

        frame.render_widget(PickerView { panel: self, theme }, rect);
        Some((cursor.1, cursor.0))
    }

    /// Keeps the selected match within the `height`-row list window, scrolling
    /// the minimum needed — the same clamp the file tree and reference list use.
    fn scroll_into_view(&mut self, height: usize) {
        if height == 0 {
            self.scroll_top = 0;
            return;
        }
        let selected = self.picker.selected_index();
        if selected < self.scroll_top {
            self.scroll_top = selected;
        } else if selected >= self.scroll_top + height {
            self.scroll_top = selected + 1 - height;
        }
        // A shrunken match list (a narrower query) can leave scroll_top past the
        // end; pull it back so the last page is filled.
        let count = self.picker.matches().len();
        let max_top = count.saturating_sub(height);
        self.scroll_top = self.scroll_top.min(max_top);
    }
}

/// The picker as a ratatui widget, rebuilt each frame from the panel — same
/// borrow-don't-own shape as [`crate::ui::lsp_ui::InfoBox`] and the file tree
/// view.
struct PickerView<'a> {
    panel: &'a PickerPanel,
    theme: &'a Theme,
}

impl Widget for PickerView<'_> {
    fn render(self, area: Rect, buf: &mut Surface) {
        if area.width < 4 || area.height < 3 {
            return;
        }
        let bg = self.theme.bg1;
        let picker = &self.panel.picker;
        // Count in the title so you can see the list narrowing as you type — the
        // telescope results counter, and the cheapest possible proof the filter
        // is live.
        let title = format!(" {} ({}) ", self.panel.title, picker.matches().len());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.bg3).bg(bg))
            .title(title)
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
        if inner.width == 0 || inner.height == 0 {
            return;
        }

        // Row 0: the prompt.
        let prompt_style = Style::default().fg(self.theme.fg).bg(bg).add_modifier(Modifier::BOLD);
        let prompt = format!("> {}", self.panel.query);
        buf.set_stringn(inner.x, inner.y, &prompt, inner.width as usize, prompt_style);

        // Row 1: a thin separator, so the prompt reads as distinct from results.
        if inner.height >= 2 {
            let sep_style = Style::default().fg(self.theme.bg3).bg(bg);
            let sep: String = "─".repeat(inner.width as usize);
            buf.set_stringn(inner.x, inner.y + 1, &sep, inner.width as usize, sep_style);
        }

        // Rows 2..: the scrolling candidate list.
        let list_y = inner.y + 2;
        let list_h = inner.height.saturating_sub(2);
        let matches = picker.matches();
        let items = picker.items();
        let selected_idx = picker.selected_index();
        for row in 0..list_h {
            let match_idx = self.panel.scroll_top + row as usize;
            let Some(m) = matches.get(match_idx) else { break };
            let Some(item) = items.get(m.item_index) else { break };
            let y = list_y + row;
            let is_selected = match_idx == selected_idx;

            let base = if is_selected {
                Style::default().fg(self.theme.bg).bg(self.theme.yellow_bright).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(self.theme.fg).bg(bg)
            };
            if is_selected {
                // Paint the whole row so the highlight spans the box width.
                let full = Rect { x: inner.x, y, width: inner.width, height: 1 };
                buf.set_style(full, base);
            }

            // Paint the label char-by-char so fuzzy-matched characters can be
            // bolded/coloured, exactly as telescope underlines its hits. The
            // match indices are *char* offsets into the label (see
            // `PickerMatch::indices`).
            let matched_style = if is_selected {
                base
            } else {
                Style::default().fg(self.theme.yellow_bright).bg(bg).add_modifier(Modifier::BOLD)
            };
            // One char per column: fine for the ASCII-ish paths, buffer names
            // and help tags a picker lists (a CJK filename would under-count, but
            // the highlight is cosmetic and the row is clipped to the box either
            // way). `ci` is both the char index the match points at and the
            // column it paints in.
            let max_cols = inner.width as usize;
            for (ci, ch) in item.label.chars().take(max_cols).enumerate() {
                let style = if m.indices.contains(&(ci as u32)) { matched_style } else { base };
                let mut tmp = [0u8; 4];
                buf.set_stringn(inner.x + ci as u16, y, ch.encode_utf8(&mut tmp), 1, style);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::event::{Key, Modifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn rows() -> Vec<PickRow> {
        vec![
            PickRow::new("src/main.rs", PickAction::OpenFile(PathBuf::from("/p/src/main.rs"))),
            PickRow::new("src/lib.rs", PickAction::OpenFile(PathBuf::from("/p/src/lib.rs"))),
            PickRow::new("README.md", PickAction::OpenFile(PathBuf::from("/p/README.md"))),
        ]
    }

    fn press(c: char) -> KeyPress {
        KeyPress { key: Key::Char(c), mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    fn ctrl(c: char) -> KeyPress {
        KeyPress { key: Key::Char(c), mods: Modifiers { ctrl: true, alt: false, shift: false } }
    }

    fn special(key: Key) -> KeyPress {
        KeyPress { key, mods: Modifiers { ctrl: false, alt: false, shift: false } }
    }

    #[test]
    fn typing_narrows_the_match_list() {
        let mut panel = PickerPanel::new("Find Files", rows());
        assert_eq!(panel.match_count(), 3, "opens showing everything");
        panel.handle_key(press('l'));
        panel.handle_key(press('i'));
        panel.handle_key(press('b'));
        // Only src/lib.rs contains a fuzzy "lib".
        assert_eq!(panel.match_count(), 1);
    }

    #[test]
    fn backspace_widens_the_list_again() {
        let mut panel = PickerPanel::new("Find Files", rows());
        for c in "lib".chars() {
            panel.handle_key(press(c));
        }
        assert_eq!(panel.match_count(), 1);
        panel.handle_key(special(Key::Backspace));
        panel.handle_key(special(Key::Backspace));
        panel.handle_key(special(Key::Backspace));
        assert_eq!(panel.match_count(), 3, "an empty query matches everything again");
    }

    #[test]
    fn enter_confirms_the_selected_row_as_an_open() {
        let mut panel = PickerPanel::new("Find Files", rows());
        for c in "main".chars() {
            panel.handle_key(press(c));
        }
        let out = panel.handle_key(special(Key::Enter));
        assert_eq!(out, OverlayOutcome::PickPath(PathBuf::from("/p/src/main.rs")));
    }

    #[test]
    fn ctrl_n_and_ctrl_p_move_the_selection() {
        let mut panel = PickerPanel::new("Find Files", rows());
        // Empty query: order is the input order.
        let first = panel.picker.selected().unwrap().label.clone();
        panel.handle_key(ctrl('n'));
        let second = panel.picker.selected().unwrap().label.clone();
        assert_ne!(first, second, "<C-n> moves the selection");
        panel.handle_key(ctrl('p'));
        assert_eq!(panel.picker.selected().unwrap().label, first, "<C-p> moves it back");
    }

    #[test]
    fn escape_and_ctrl_c_both_close() {
        let mut panel = PickerPanel::new("Find Files", rows());
        assert_eq!(panel.handle_key(special(Key::Escape)), OverlayOutcome::Close);
        assert_eq!(panel.handle_key(ctrl('c')), OverlayOutcome::Close);
    }

    #[test]
    fn the_box_paints_the_prompt_and_the_candidates() {
        let theme = Theme::gruvbox_dark();
        let mut panel = PickerPanel::new("Find Files", rows());
        panel.handle_key(press('r'));
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        terminal
            .draw(|frame| {
                let area = frame.area();
                panel.render(frame, area, &theme, IconSet::Ascii, true);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text: String = (0..10)
            .map(|y| (0..40).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("> r"), "the prompt with the typed query must paint:\n{text}");
        // README.md is a strong "r" match and must be listed.
        assert!(text.contains("README.md"), "a matching candidate must paint:\n{text}");
    }

    /// The exact bug this widget's `Clear` fix is about: paint the whole screen
    /// with `X` (stand-in for the buffer text under the picker), drop the picker
    /// on top, then assert not one `X` survive inside its rect and every cell
    /// there carry an opaque bg.
    #[test]
    fn picker_is_opaque_no_buffer_text_bleeds_through() {
        let theme = Theme::gruvbox_dark();
        let mut panel = PickerPanel::new("Find Files", rows());
        let area = Rect { x: 0, y: 0, width: 40, height: 10 };
        let mut terminal = Terminal::new(TestBackend::new(area.width, area.height)).unwrap();
        terminal
            .draw(|frame| {
                let fill = "X".repeat(area.width as usize);
                for y in 0..area.height {
                    frame.buffer_mut().set_string(0, y, &fill, Style::default());
                }
                panel.render(frame, area, &theme, IconSet::Ascii, true);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                let c = buf.cell((x, y)).unwrap();
                assert_ne!(c.symbol(), "X", "buffer text bled through the picker at ({x},{y})");
                assert!(c.style().bg.is_some(), "cell ({x},{y}) inside the picker is not opaque");
            }
        }
    }
}
