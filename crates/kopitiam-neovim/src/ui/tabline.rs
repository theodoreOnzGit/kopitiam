//! The tabline: kvim's top-of-screen bar that show the open tab pages, one
//! per neovim tab (`:tabnew`, `gt`/`gT`), same idea as neovim's own
//! `tabline`.
//!
//! # What it draw
//!
//! One row, full width, sitting right at the top of the screen. Each tab page
//! get one entry, laid out left-to-right, in the shape:
//!
//! ```text
//!  {number} {name}{modifier}
//! ```
//!
//! - `number` is the 1-based tab number the user see (tab 1, tab 2, ...).
//! - `name` is the tab's active-window buffer display name — the tail /
//!   basename already worked out by the caller (`App`), or `"[No Name]"` for a
//!   scratch buffer. This module never touch buffers or windows itself; it
//!   just paint the [`TablineEntry`] it kena handed. That keep business logic
//!   out of the UI, same as `CLAUDE.md` say.
//! - `modifier` is `" +"` (leading space, then the plus) when the active
//!   buffer got unsaved changes, else nothing.
//!
//! There is one space of padding on both ends, so a clean tab read as
//! `" 1 main.rs "` and a dirty one as `" 2 notes.md + "`.
//!
//! # Active vs inactive — the one visual job that matter
//!
//! The whole point of a tabline is "which tab am I on now?", so the active tab
//! must jump out. This module paint:
//!
//! - **active** entry: bright foreground [`Theme::fg`] on a distinctly lighter
//!   background [`Theme::bg2`] — same "lift the active segment" trick the
//!   statusline use.
//! - **inactive** entries: dim foreground [`Theme::gray`] on the base tabline
//!   background [`Theme::bg1`].
//!
//! # Filling the whole row
//!
//! After the last entry, the rest of the row kena painted with the tabline
//! background [`Theme::bg1`] so every cell up to `area.width` got a colour — no
//! leftover see-through cells. This work exactly like the statusline: paint the
//! base background over the whole `area` first, then write the entry spans on
//! top with [`ratatui::buffer::Buffer::set_line`].
//!
//! # Overflow (too many / too wide tabs)
//!
//! If the entries don't all fit, we stop adding entries once the row is full,
//! and the last partial entry kena clipped at the edge by `set_line` (which is
//! unicode-width-correct and never write past the `Rect`). Widths are measured
//! with [`unicode_width::UnicodeWidthStr`] — the same crate/trait the rest of
//! the crate use for display width (see `textarea.rs`, `cmdline.rs`) — so a CJK
//! or emoji tab name count for its real terminal columns, not its byte or
//! `char` count. Never panic, never write outside the `area`.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Widget,
};
use unicode_width::UnicodeWidthStr;

use crate::ui::theme::Theme;

/// One tab page's data for the tabline. The caller (`App`) already resolve the
/// active window's buffer name + modified flag for each tab, so this struct is
/// pure display data — no window, no buffer, no logic hiding inside.
#[derive(Debug, Clone)]
pub struct TablineEntry {
    /// 1-based tab number shown to the user.
    pub number: usize,
    /// The tab's active-window buffer display name (tail/basename), or
    /// "[No Name]".
    pub name: String,
    /// Active buffer has unsaved changes → render a ` +`.
    pub modified: bool,
    /// This is the currently-active tab → highlight it.
    pub active: bool,
}

impl TablineEntry {
    /// The exact text one entry occupy, padding and modifier and all:
    /// `" {number} {name}{modifier} "`. Private on purpose — it is the one
    /// place the entry shape is decided, so the width measurement and the
    /// painting can never drift apart (measure and draw the *same* string).
    fn label(&self) -> String {
        let modifier = if self.modified { " +" } else { "" };
        format!(" {} {}{} ", self.number, self.name, modifier)
    }
}

/// The tabline widget: a full-width top row painting the tab pages.
///
/// Consumed by value on [`Widget::render`], same as the statusline. It borrow
/// its data ([`entries`](Self::entries)) and its colours
/// ([`theme`](Self::theme)) — it own nothing, so building one every frame is
/// cheap.
pub struct Tabline<'a> {
    pub entries: &'a [TablineEntry],
    pub theme: &'a Theme,
}

impl<'a> Tabline<'a> {
    /// The `(foreground, background)` pair for one entry. Active tab get the
    /// bright `fg` on the lighter `bg2` so it pop; inactive tabs get the dim
    /// `gray` on the base `bg1` so they recede.
    fn entry_style(&self, active: bool) -> (Color, Color) {
        if active {
            (self.theme.fg, self.theme.bg2)
        } else {
            (self.theme.gray, self.theme.bg1)
        }
    }
}

impl Widget for Tabline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Caller promise a height-1 Rect, but guard anyway — a zero-height (or
        // zero-width) area got nothing to draw and must not touch the buffer.
        if area.height == 0 || area.width == 0 {
            return;
        }

        // Step 1: paint the whole row with the tabline background first. This
        // is what fill every trailing cell after the last entry, so the row is
        // fully coloured to `area.width` — no leftover cells.
        buf.set_style(area, Style::default().bg(self.theme.bg1));

        // Step 2: build the entry spans left-to-right, stopping once the row is
        // full. We measure each entry's display width with UnicodeWidthStr so
        // the "does it still fit?" check count real terminal columns. The last
        // entry we push may overflow a bit; that's fine — `set_line` below clip
        // it at the edge, unicode-correct, without ever writing past `area`.
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut used: u16 = 0;
        for entry in self.entries {
            if used >= area.width {
                break;
            }
            let text = entry.label();
            let width = UnicodeWidthStr::width(text.as_str()) as u16;
            let (fg, bg) = self.entry_style(entry.active);
            spans.push(Span::styled(text, Style::default().fg(fg).bg(bg)));
            used = used.saturating_add(width);
        }

        // Step 3: write the spans over the painted background. `set_line` clip
        // to `area.width`, so a too-wide last entry kena trimmed and nothing
        // spill out of the Rect.
        let line = Line::from(spans);
        buf.set_line(area.x, area.y, &line, area.width);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        Theme::gruvbox_dark()
    }

    /// Collect the whole rendered row into a `String` so we can assert names
    /// and numbers show up, never mind their exact column.
    fn row_text(buf: &Buffer, width: u16) -> String {
        (0..width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().to_string())
            .collect()
    }

    fn render(entries: &[TablineEntry], width: u16) -> Buffer {
        let t = theme();
        let area = Rect::new(0, 0, width, 1);
        let mut buf = Buffer::empty(area);
        Tabline { entries, theme: &t }.render(area, &mut buf);
        buf
    }

    #[test]
    fn two_entries_render_numbers_names_and_highlight_the_active_one() {
        let t = theme();
        let entries = vec![
            TablineEntry { number: 1, name: "main.rs".into(), modified: false, active: true },
            TablineEntry { number: 2, name: "other.rs".into(), modified: false, active: false },
        ];
        let buf = render(&entries, 40);
        let text = row_text(&buf, 40);
        // Both tabs show their number and name.
        assert!(text.contains("1 main.rs"), "row was {text:?}");
        assert!(text.contains("2 other.rs"), "row was {text:?}");

        // Active entry (tab 1) start at x=0 as " 1 main.rs "; the '1' sit at
        // x=1. It must carry the active bg (bg2) and bright fg.
        let active_cell = buf.cell((1, 0)).unwrap();
        assert_eq!(active_cell.symbol(), "1");
        assert_eq!(active_cell.style().bg, Some(t.bg2), "active tab must use bg2");
        assert_eq!(active_cell.style().fg, Some(t.fg), "active tab must use bright fg");

        // Entry 1 " 1 main.rs " is 11 columns wide, so entry 2 " 2 other.rs "
        // begin at x=11 and its '2' sit at x=12. It is inactive → base bg1,
        // dim gray, definitely not the active bg.
        let inactive_cell = buf.cell((12, 0)).unwrap();
        assert_eq!(inactive_cell.symbol(), "2");
        assert_eq!(inactive_cell.style().bg, Some(t.bg1), "inactive tab must use bg1");
        assert_eq!(inactive_cell.style().fg, Some(t.gray), "inactive tab must use gray");
        assert_ne!(inactive_cell.style().bg, Some(t.bg2), "inactive tab must not steal the active bg");
    }

    #[test]
    fn a_modified_tab_shows_a_plus() {
        let entries = vec![TablineEntry {
            number: 1,
            name: "notes.md".into(),
            modified: true,
            active: true,
        }];
        let buf = render(&entries, 40);
        let text = row_text(&buf, 40);
        assert!(text.contains("notes.md +"), "modified marker missing, row was {text:?}");
    }

    #[test]
    fn the_row_is_fully_painted_to_the_full_width() {
        let t = theme();
        let entries = vec![TablineEntry {
            number: 1,
            name: "main.rs".into(),
            modified: false,
            active: true,
        }];
        // One short entry, wide row — the far-right cell is trailing filler and
        // must carry the tabline background bg1, not the terminal default.
        let buf = render(&entries, 40);
        let last = buf.cell((39, 0)).unwrap();
        assert_eq!(last.style().bg, Some(t.bg1), "trailing filler must be the tabline bg (bg1)");
    }

    #[test]
    fn overflow_into_a_tiny_area_does_not_panic_and_stays_in_bounds() {
        let entries = vec![
            TablineEntry { number: 1, name: "alpha.rs".into(), modified: false, active: true },
            TablineEntry { number: 2, name: "bravo.rs".into(), modified: true, active: false },
            TablineEntry { number: 3, name: "charlie.rs".into(), modified: false, active: false },
        ];
        // Width 6 cannot hold even one full entry. Must not panic, and the
        // buffer must stay exactly 6 wide (nothing written outside).
        let buf = render(&entries, 6);
        assert_eq!(buf.area().width, 6);
        // Every one of the 6 cells is reachable (i.e. we never wrote past the
        // edge, which would have panicked inside the buffer already).
        for x in 0..6 {
            assert!(buf.cell((x, 0)).is_some());
        }
    }

    #[test]
    fn no_name_buffer_renders_its_placeholder() {
        let entries = vec![TablineEntry {
            number: 1,
            name: "[No Name]".into(),
            modified: false,
            active: true,
        }];
        let buf = render(&entries, 40);
        let text = row_text(&buf, 40);
        assert!(text.contains("[No Name]"), "row was {text:?}");
    }

    #[test]
    fn zero_width_area_is_a_no_op_and_does_not_panic() {
        let t = theme();
        let entries = vec![TablineEntry {
            number: 1,
            name: "main.rs".into(),
            modified: false,
            active: true,
        }];
        let area = Rect::new(0, 0, 0, 1);
        let mut buf = Buffer::empty(area);
        Tabline { entries: &entries, theme: &t }.render(area, &mut buf);
        assert_eq!(buf.area().width, 0);
    }
}
