//! The insert-mode completion popup: a floating, cursor-anchored menu of
//! candidates (LSP + snippet + buffer word + path), styled like the
//! maintainer's `blink.cmp` — a kind/source badge, the label, and a detail
//! column, with the selected row highlighted.
//!
//! # A cursor-anchored popup, not a docked one
//!
//! Unlike [`crate::ui::whichkey`] (docked to the bottom) or the hover box
//! ([`crate::ui::lsp_ui`], centred), a completion menu belongs *at the word
//! being typed* — that is where the eye already is. So this module's rectangle
//! helper ([`menu_rect`]) takes the cursor's screen cell and drops the box just
//! below it, flipping above when there is no room underneath, exactly as an IDE
//! popup does. The panel is a passive render pass; focus and key handling live
//! in [`crate::ui::app::App`], the same division of labour every other kvim
//! popup uses.
//!
//! # Painting, and why the tests assert cells
//!
//! Every test here renders through `ratatui`'s `TestBackend` and asserts on the
//! **painted cells** — the literal glyphs and their fg/bg — never on a piece of
//! widget state. That is a hard house rule in this crate: real bugs (an
//! invisible `:` prompt, an unhighlighted visual selection) slipped a
//! 400-plus-test suite precisely because the tests checked state, not pixels.
//! A completion menu that computes the right rows but paints them off-screen,
//! or paints the wrong row as selected, is exactly that failure mode, so the
//! assertions are on what the user would actually see.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Block, Borders, Clear, Widget},
};

use kopitiam_semantic::CompletionItemKind;

use crate::lsp::completion::{CompletionItem, CompletionSource};
use crate::ui::theme::Theme;

/// A short, fixed-width badge for a completion item — the kind when the source
/// knows it (LSP items), else the source. Three characters keeps the badge
/// column aligned without a per-frame width computation.
pub fn badge(item: &CompletionItem) -> &'static str {
    if let Some(kind) = item.kind {
        return kind_badge(kind);
    }
    match item.source {
        CompletionSource::Lsp => "lsp",
        CompletionSource::Snippet => "snp",
        CompletionSource::Buffer => "buf",
        CompletionSource::Path => "pth",
    }
}

/// The badge text for an LSP `CompletionItemKind`. Abbreviated to three glyphs
/// so every row's label column starts at the same screen x.
fn kind_badge(kind: CompletionItemKind) -> &'static str {
    match kind {
        CompletionItemKind::Function => "fn ",
        CompletionItemKind::Method => "mth",
        CompletionItemKind::Constructor => "new",
        CompletionItemKind::Field => "fld",
        CompletionItemKind::Variable => "var",
        CompletionItemKind::Class | CompletionItemKind::Struct => "typ",
        CompletionItemKind::Interface => "ifc",
        CompletionItemKind::Module => "mod",
        CompletionItemKind::Property => "prp",
        CompletionItemKind::Enum => "enm",
        CompletionItemKind::EnumMember => "emb",
        CompletionItemKind::Keyword => "kw ",
        CompletionItemKind::Snippet => "snp",
        CompletionItemKind::Constant => "cst",
        CompletionItemKind::Value | CompletionItemKind::Unit => "val",
        CompletionItemKind::File => "fil",
        CompletionItemKind::Folder => "dir",
        CompletionItemKind::Reference => "ref",
        CompletionItemKind::TypeParameter => "tpm",
        CompletionItemKind::Operator => "op ",
        CompletionItemKind::Event => "evt",
        CompletionItemKind::Color | CompletionItemKind::Text => "txt",
    }
}

/// The badge colour, bucketed by kind/source so callables, types, and snippets
/// read differently at a glance — the blink.cmp habit of colour-coding the
/// kind icon.
fn badge_color(item: &CompletionItem, theme: &Theme) -> ratatui::style::Color {
    if let Some(kind) = item.kind {
        return match kind {
            CompletionItemKind::Function
            | CompletionItemKind::Method
            | CompletionItemKind::Constructor => theme.green_bright,
            CompletionItemKind::Class
            | CompletionItemKind::Struct
            | CompletionItemKind::Interface
            | CompletionItemKind::Enum
            | CompletionItemKind::TypeParameter => theme.yellow_bright,
            CompletionItemKind::Keyword | CompletionItemKind::Operator => theme.red_bright,
            CompletionItemKind::Snippet => theme.purple_bright,
            CompletionItemKind::Module | CompletionItemKind::File | CompletionItemKind::Folder => {
                theme.blue_bright
            }
            CompletionItemKind::Field
            | CompletionItemKind::Property
            | CompletionItemKind::Variable
            | CompletionItemKind::EnumMember => theme.aqua_bright,
            _ => theme.orange_bright,
        };
    }
    match item.source {
        CompletionSource::Snippet => theme.purple_bright,
        CompletionSource::Path => theme.blue_bright,
        _ => theme.gray,
    }
}

/// The completion menu widget: a bordered box of candidate rows.
pub struct CompletionMenu<'a> {
    pub items: &'a [CompletionItem],
    /// The highlighted (currently-selected) row, an index into `items`.
    pub selected: usize,
    /// The first `items` index shown, so a long list can scroll to keep the
    /// selection visible.
    pub scroll: usize,
    pub theme: &'a Theme,
}

impl CompletionMenu<'_> {
    /// The visible-content width the menu wants: the widest `badge + label +
    /// detail` row, plus separators and borders, capped at `max`.
    pub fn desired_width(items: &[CompletionItem], max: u16) -> u16 {
        let widest = items
            .iter()
            .map(|i| {
                // "xxx " badge + label + "  " + detail
                4 + i.label.chars().count()
                    + i.detail.as_ref().map(|d| d.chars().count() + 2).unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        // + 2 for the border columns, +1 leading pad.
        ((widest + 3) as u16).clamp(12, max)
    }
}

impl Widget for CompletionMenu<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 4 || area.height < 3 {
            return;
        }
        let bg = self.theme.bg1;
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.bg3).bg(bg))
            .title("completion")
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

        let detail_style = Style::default().fg(self.theme.gray).bg(bg);
        for row in 0..inner.height {
            let idx = self.scroll + row as usize;
            let Some(item) = self.items.get(idx) else { break };
            let y = inner.y + row;
            let selected = idx == self.selected;
            let row_bg = if selected { self.theme.bg3 } else { bg };
            // Paint the whole row background first so the selection band spans
            // the panel width, not just the text.
            let row_rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
            buf.set_style(row_rect, Style::default().bg(row_bg));

            let mut x = inner.x + 1;
            let width_end = inner.x + inner.width;

            // Badge (kind/source), colour-coded.
            let badge_text = badge(item);
            let badge_style = Style::default().fg(badge_color(item, self.theme)).bg(row_bg);
            buf.set_stringn(x, y, badge_text, (width_end - x) as usize, badge_style);
            x += badge_text.chars().count() as u16 + 1;

            // Label — the thing that gets inserted; brightest on the selected
            // row so the choice is unmistakable.
            let label_style = if selected {
                Style::default().fg(self.theme.fg).bg(row_bg).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(self.theme.fg).bg(row_bg)
            };
            if x < width_end {
                buf.set_stringn(x, y, &item.label, (width_end - x) as usize, label_style);
                x += item.label.chars().count() as u16;
            }

            // Detail (type signature / snippet description), de-emphasised and
            // right after the label with a two-space gap.
            if let Some(detail) = &item.detail {
                x += 2;
                if x < width_end {
                    let one_line = detail.lines().next().unwrap_or(detail);
                    buf.set_stringn(x, y, one_line, (width_end - x) as usize, detail_style);
                }
            }
        }
    }
}

/// Which side of the cursor line a cursor-anchored popup want to sit on, *before*
/// we flip at the edge. Two popup anchor at the cursor but lean opposite side one:
/// the completion menu drop **below** the word you typing (so it never block the
/// word), but the LSP hover box sit **above** the cursor line same like how
/// `vim.lsp.buf.hover` do. Same flip-and-clamp maths, just the starting side
/// different — so [`anchored_rect`] take this enum instead of keeping two almost-
/// same copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    /// Try the line below the cursor first; only flip above if cannot fit.
    Below,
    /// Try the lines above the cursor first; only flip below if cannot fit.
    Above,
}

/// The rectangle for a box of `rows` rows and `width` columns, anchored at the
/// cursor cell `(cx, cy)` inside `area`.
///
/// Up-down wise it prefer whichever side `anchor` say (below for the completion
/// menu, above for the hover box), and only flip to the other side when the
/// preferred side cannot fit the box *and* the other side got strictly more room —
/// standard IDE popup placement lah. Left-right wise the box start at the cursor
/// column, then shift left just enough to stay inside `area`. Height is `rows + 2`
/// (the borders), capped so it never overflow the space got on the chosen side.
///
/// This is the one and only anchoring maths in the whole crate: both the completion
/// menu ([`menu_rect`]) and the LSP hover popup go through here, so their edge
/// behaviour always the same, never drift apart.
pub fn anchored_rect(area: Rect, cursor: (u16, u16), rows: usize, width: u16, anchor: Anchor) -> Rect {
    let (cx, cy) = cursor;
    let width = width.min(area.width).max(4);

    // Space above / below the cursor line, in rows, inside `area`.
    let below = area.y + area.height - (cy + 1).min(area.y + area.height);
    let above = cy.saturating_sub(area.y);

    let wanted = rows as u16 + 2; // borders

    // Place the box below the cursor line, starting at `cy + 1`.
    let place_below = |room: u16| -> (u16, u16) { (cy + 1, wanted.min(room).max(3)) };
    // Place the box above the cursor line, ending at `cy` (exclusive).
    let place_above = |room: u16| -> (u16, u16) {
        let h = wanted.min(room).max(3);
        (cy.saturating_sub(h), h)
    };

    let (y, height) = match anchor {
        // Prefer below; flip above only if below cannot fit and above has more room.
        Anchor::Below => {
            if wanted <= below || below >= above {
                place_below(below)
            } else {
                place_above(above)
            }
        }
        // Prefer above; flip below only if above cannot fit and below has more room.
        Anchor::Above => {
            if wanted <= above || above >= below {
                place_above(above)
            } else {
                place_below(below)
            }
        }
    };

    // Clamp x so the whole box stays inside `area`.
    let max_x = area.x + area.width.saturating_sub(width);
    let x = cx.min(max_x).max(area.x);
    Rect { x, y, width, height }
}

/// The rectangle for a completion menu of `rows` rows and `width` columns, anchored
/// just below the cursor cell `(cx, cy)`. Just a thin wrapper over [`anchored_rect`]
/// with [`Anchor::Below`] — the placement a blink.cmp-style popup want: drop under
/// the word you typing, and flip above only when near the bottom edge.
pub fn menu_rect(area: Rect, cursor: (u16, u16), rows: usize, width: u16) -> Rect {
    anchored_rect(area, cursor, rows, width, Anchor::Below)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    fn lsp_item(label: &str, kind: CompletionItemKind, detail: &str) -> CompletionItem {
        CompletionItem::new(label, CompletionSource::Lsp).with_kind(kind).with_detail(detail)
    }

    fn render(items: &[CompletionItem], selected: usize) -> (String, Buffer) {
        let theme = Theme::gruvbox_dark();
        let mut terminal = Terminal::new(TestBackend::new(48, 8)).unwrap();
        terminal
            .draw(|frame| {
                let area = frame.area();
                let width = CompletionMenu::desired_width(items, area.width);
                let rect = menu_rect(area, (2, 1), items.len(), width);
                frame.render_widget(
                    CompletionMenu { items, selected, scroll: 0, theme: &theme },
                    rect,
                );
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let text: String = (0..8)
            .map(|y| (0..48).map(|x| buf.cell((x, y)).unwrap().symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        (text, buf)
    }

    #[test]
    fn menu_paints_label_badge_and_detail() {
        let items = vec![lsp_item("greet", CompletionItemKind::Function, "fn() -> &str")];
        let (text, _) = render(&items, 0);
        assert!(text.contains("greet"), "the label must be painted: {text}");
        assert!(text.contains("fn"), "the Function badge `fn` must be painted: {text}");
        assert!(text.contains("fn() -> &str"), "the detail column must be painted: {text}");
    }

    #[test]
    fn menu_highlights_the_selected_row_with_a_distinct_background() {
        let items = vec![
            CompletionItem::new("alpha", CompletionSource::Buffer),
            CompletionItem::new("beta", CompletionSource::Buffer),
        ];
        let theme = Theme::gruvbox_dark();
        let (_, buf) = render(&items, 1); // select "beta"
        // 'b' is unique to "beta"; it must sit on the selection background.
        let beta_selected = (0..48)
            .flat_map(|x| (0..8).map(move |y| (x, y)))
            .any(|(x, y)| {
                let c = buf.cell((x, y)).unwrap();
                c.symbol() == "b" && c.style().bg == Some(theme.bg3)
            });
        assert!(beta_selected, "the selected row 'beta' must be painted on the selection background");
        // 'l' is unique to "alpha" (beta has no 'l'); the unselected row must
        // keep the panel background, NOT the selection band.
        let alpha_selected = (0..48)
            .flat_map(|x| (0..8).map(move |y| (x, y)))
            .any(|(x, y)| {
                let c = buf.cell((x, y)).unwrap();
                c.symbol() == "l" && c.style().bg == Some(theme.bg3)
            });
        assert!(!alpha_selected, "the unselected row 'alpha' must keep the panel background");
    }

    #[test]
    fn badge_reads_the_kind_for_lsp_and_the_source_otherwise() {
        assert_eq!(badge(&lsp_item("x", CompletionItemKind::Method, "")), "mth");
        assert_eq!(badge(&CompletionItem::new("w", CompletionSource::Buffer)), "buf");
        assert_eq!(badge(&CompletionItem::new("p/", CompletionSource::Path)), "pth");
        let snip = CompletionItem::new("fn", CompletionSource::Snippet)
            .with_kind(CompletionItemKind::Snippet)
            .with_snippet("body");
        assert_eq!(badge(&snip), "snp");
    }

    #[test]
    fn menu_rect_drops_below_the_cursor_when_there_is_room() {
        let area = Rect { x: 0, y: 0, width: 80, height: 24 };
        let rect = menu_rect(area, (10, 5), 3, 30);
        assert_eq!(rect.y, 6, "the box should start on the line below the cursor row (5)");
        assert_eq!(rect.x, 10, "and at the cursor column");
        assert_eq!(rect.height, 5, "3 rows + 2 borders");
    }

    #[test]
    fn menu_rect_flips_above_the_cursor_near_the_bottom_edge() {
        let area = Rect { x: 0, y: 0, width: 80, height: 24 };
        // Cursor on the last usable row: no room below, so flip up.
        let rect = menu_rect(area, (10, 23), 5, 30);
        assert!(rect.y + rect.height <= 23, "the box must sit entirely above the cursor row: {rect:?}");
    }

    #[test]
    fn menu_rect_shifts_left_to_stay_inside_the_area() {
        let area = Rect { x: 0, y: 0, width: 40, height: 24 };
        // Cursor near the right edge with a wide box: it must not overflow.
        let rect = menu_rect(area, (38, 2), 2, 30);
        assert!(rect.x + rect.width <= area.width, "the box must not overflow the right edge: {rect:?}");
    }

    #[test]
    fn anchored_above_sits_above_the_cursor_and_col_aligns() {
        let area = Rect { x: 0, y: 0, width: 80, height: 24 };
        // Plenty of room above row 10: the box (hover-style) must sit above it.
        let rect = anchored_rect(area, (12, 10), 3, 30, Anchor::Above);
        assert_eq!(rect.height, 5, "3 rows + 2 borders");
        assert_eq!(rect.y + rect.height, 10, "the box's bottom border must touch the cursor row (10)");
        assert_eq!(rect.x, 12, "and start at the cursor column");
    }

    #[test]
    fn anchored_above_flips_below_near_the_top_edge() {
        let area = Rect { x: 0, y: 0, width: 80, height: 24 };
        // Cursor on row 1: not enough room above, so an Above popup flips below.
        let rect = anchored_rect(area, (5, 1), 6, 30, Anchor::Above);
        assert!(rect.y > 1, "the box must flip to below the cursor row: {rect:?}");
        assert_eq!(rect.y, 2, "starting on the line just below the cursor");
    }

    #[test]
    fn anchored_above_clamps_the_right_edge() {
        let area = Rect { x: 0, y: 0, width: 40, height: 24 };
        let rect = anchored_rect(area, (38, 10), 2, 30, Anchor::Above);
        assert!(rect.x + rect.width <= area.width, "the box must not overflow the right edge: {rect:?}");
    }

    /// The exact bug this widget's `Clear` fix is about: a popup that don't wipe
    /// its cells let the buffer text behind bleed through and look transparent.
    /// Paint the whole screen with `X` (stand-in for the buffer text under the
    /// menu), drop the menu on top, then assert not one `X` survive inside the
    /// menu box and every cell there carry an opaque bg.
    #[test]
    fn menu_is_opaque_no_buffer_text_bleeds_through() {
        let items = vec![lsp_item("greet", CompletionItemKind::Function, "fn() -> &str")];
        let theme = Theme::gruvbox_dark();
        let area = Rect { x: 0, y: 0, width: 48, height: 8 };
        let width = CompletionMenu::desired_width(&items, area.width);
        let rect = menu_rect(area, (2, 1), items.len(), width);
        let mut terminal = Terminal::new(TestBackend::new(area.width, area.height)).unwrap();
        terminal
            .draw(|frame| {
                let fill = "X".repeat(area.width as usize);
                for y in 0..area.height {
                    frame.buffer_mut().set_string(0, y, &fill, Style::default());
                }
                frame.render_widget(
                    CompletionMenu { items: &items, selected: 0, scroll: 0, theme: &theme },
                    rect,
                );
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        for y in rect.top()..rect.bottom() {
            for x in rect.left()..rect.right() {
                let c = buf.cell((x, y)).unwrap();
                assert_ne!(c.symbol(), "X", "buffer text bled through the menu at ({x},{y})");
                assert!(c.style().bg.is_some(), "cell ({x},{y}) inside the menu is not opaque");
            }
        }
    }
}
