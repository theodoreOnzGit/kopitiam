//! The statusline: a from-scratch replacement for `vim-airline`'s
//! powerline-style, colour-segmented status bar.
//!
//! # Segments
//!
//! Left-aligned: **mode** (coloured per [`Mode`], using [`Mode::label`]),
//! **git branch** (a hook only — see below), **file name** with a `[+]`
//! modified marker. Right-aligned: **filetype**, **line:col**, and
//! **percentage through file**.
//!
//! # Nerd Font fallback
//!
//! Airline's look depends on the powerline arrow glyphs (`` U+E0B0,
//! `` U+E0B2), which only render as arrows in a Nerd Font — in any other
//! font they're tofu boxes. Whether a Nerd Font is available is a font-
//! detection concern owned by another agent (`crate::icons`), so this
//! module never decides it: [`Statusline::glyphs`] is a plain `bool`
//! parameter the caller supplies, defaulting to plain ASCII separators when
//! `false` so the statusline degrades gracefully rather than showing boxes.
//!
//! # Git branch is a hook, not an implementation
//!
//! `git_branch: Option<String>` on [`StatuslineData`] is populated by
//! whoever owns git integration (the plugins agent, per `crate::plugins`) —
//! this module just renders `Some(branch)` as a segment and omits the
//! segment entirely for `None`. Determining the branch (running `git`,
//! reading `.git/HEAD`, or a pure-Rust git library) is explicitly not this
//! module's job; rendering UI must never reach for git itself, per
//! `CLAUDE.md`'s "never place business logic inside user interfaces".

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Widget,
};

use crate::core::{Mode, Position};
use crate::ui::theme::Theme;

/// Powerline arrow glyphs (Nerd Font / any font with Unicode Private Use
/// Area / Powerline symbols support).
pub const GLYPH_SEP_RIGHT: char = '\u{e0b0}';
pub const GLYPH_SEP_LEFT: char = '\u{e0b2}';

/// Plain-ASCII fallback separators, used when [`Statusline::glyphs`] is
/// `false`.
pub const PLAIN_SEP_RIGHT: char = '>';
pub const PLAIN_SEP_LEFT: char = '<';

/// Everything the statusline needs to know, gathered by the caller from the
/// editor state, the buffer, and (for `git_branch`) the plugin layer.
#[derive(Debug, Clone)]
pub struct StatuslineData {
    pub mode: Mode,
    /// Display name, e.g. `"main.rs"` or `"[No Name]"` for a scratch buffer.
    pub file_name: String,
    pub modified: bool,
    /// e.g. `"rust"`, or empty for an unrecognised/plain-text file.
    pub filetype: String,
    /// `None` when git integration hasn't populated this yet, or the file
    /// isn't in a git repository — the segment is simply omitted.
    pub git_branch: Option<String>,
    /// A short language-server status hint, e.g. `"LSP: starting…"` while the
    /// server for this buffer is connecting on its background thread. `None`
    /// once the server is ready (or when the buffer has no server), so the
    /// segment is shown only while it carries information — the async LSP
    /// client's non-blocking connect (bead `kopitiam-cj0.27`) is otherwise
    /// invisible, and this is the subtle cue that it is warming up.
    pub lsp_status: Option<String>,
    pub cursor: Position,
    pub line_count: usize,
}

/// The percentage-through-file segment, matching vim's ruler convention of
/// showing `Top`/`Bot`/`All` at the file's edges instead of `0%`/`100%`
/// (which reads as "there's more above/below" when there isn't).
pub fn percent_through_file(cursor_line: usize, line_count: usize) -> String {
    if line_count <= 1 {
        return "All".to_string();
    }
    if cursor_line == 0 {
        return "Top".to_string();
    }
    if cursor_line + 1 >= line_count {
        return "Bot".to_string();
    }
    let pct = (cursor_line * 100) / (line_count - 1);
    format!("{pct}%")
}

/// The mode segment's background colour — the single most important visual
/// cue in the statusline, since it's the fastest way to notice "wait, am I
/// in insert mode?" without reading text.
pub fn mode_color(mode: Mode, theme: &Theme) -> Color {
    match mode {
        Mode::Normal => theme.blue_bright,
        Mode::Insert => theme.green_bright,
        Mode::Visual | Mode::VisualLine | Mode::VisualBlock => theme.purple_bright,
        Mode::Replace => theme.red_bright,
        Mode::Command => theme.yellow_bright,
        Mode::OperatorPending => theme.orange_bright,
    }
}

/// One coloured statusline segment: text on a background colour, rendered
/// with dark (`theme.bg`) foreground so it reads clearly against the
/// bright, mode-derived segment colours airline is known for.
struct Segment {
    text: String,
    bg: Color,
}

/// The statusline widget.
pub struct Statusline<'a> {
    pub data: &'a StatuslineData,
    pub theme: &'a Theme,
    /// Whether the active font supports Nerd Font / Powerline glyphs. Owned
    /// and supplied by the icon/font-detection layer — see module docs.
    pub glyphs: bool,
}

impl<'a> Statusline<'a> {
    fn left_segments(&self) -> Vec<Segment> {
        let mut segments = vec![Segment {
            text: format!(" {} ", self.data.mode.label()),
            bg: mode_color(self.data.mode, self.theme),
        }];
        if let Some(branch) = &self.data.git_branch {
            segments.push(Segment { text: format!(" {branch} "), bg: self.theme.aqua });
        }
        let modified_marker = if self.data.modified { " [+]" } else { "" };
        segments.push(Segment {
            text: format!(" {}{} ", self.data.file_name, modified_marker),
            bg: self.theme.bg2,
        });
        segments
    }

    fn right_segments(&self) -> Vec<Segment> {
        let mut segments = Vec::new();
        // The LSP hint sits at the far left of the right group (outermost, so it
        // reads first) and only while the server is starting.
        if let Some(status) = &self.data.lsp_status {
            segments.push(Segment { text: format!(" {status} "), bg: self.theme.bg2 });
        }
        if !self.data.filetype.is_empty() {
            segments.push(Segment { text: format!(" {} ", self.data.filetype), bg: self.theme.bg2 });
        }
        segments.push(Segment {
            text: format!(" {} ", self.data.cursor),
            bg: self.theme.bg3,
        });
        segments.push(Segment {
            text: format!(" {} ", percent_through_file(self.data.cursor.line, self.data.line_count)),
            bg: mode_color(self.data.mode, self.theme),
        });
        segments
    }

    fn sep_right(&self) -> char {
        if self.glyphs { GLYPH_SEP_RIGHT } else { PLAIN_SEP_RIGHT }
    }

    fn sep_left(&self) -> char {
        if self.glyphs { GLYPH_SEP_LEFT } else { PLAIN_SEP_LEFT }
    }

    /// Builds the full statusline as a single [`Line`] of styled spans, on
    /// the base statusline background (`theme.bg1`) that fills any space
    /// between the left and right segment groups.
    ///
    /// Exposed independent of [`Widget::render`] so the exact text (and its
    /// segment structure) is assertable in tests without needing a
    /// `Buffer`/`Rect` at all.
    pub fn build_line(&self) -> Line<'static> {
        let base_bg = self.theme.bg1;
        let mut spans: Vec<Span<'static>> = Vec::new();

        let left = self.left_segments();
        for (i, seg) in left.iter().enumerate() {
            spans.push(Span::styled(
                seg.text.clone(),
                Style::default().fg(self.theme.bg).bg(seg.bg),
            ));
            let next_bg = left.get(i + 1).map(|s| s.bg).unwrap_or(base_bg);
            spans.push(Span::styled(
                self.sep_right().to_string(),
                Style::default().fg(seg.bg).bg(next_bg),
            ));
        }

        // The right group is built right-to-left so each separator's
        // colours (own bg -> previous segment's bg, reading outward) come
        // out correct without a second reversal pass.
        let right = self.right_segments();
        let mut right_spans: Vec<Span<'static>> = Vec::new();
        for (i, seg) in right.iter().enumerate() {
            let prev_bg = if i == 0 { base_bg } else { right[i - 1].bg };
            right_spans.push(Span::styled(
                self.sep_left().to_string(),
                Style::default().fg(seg.bg).bg(prev_bg),
            ));
            right_spans.push(Span::styled(
                seg.text.clone(),
                Style::default().fg(self.theme.bg).bg(seg.bg),
            ));
        }

        spans.extend(right_spans);
        Line::from(spans).style(Style::default().bg(base_bg))
    }
}

impl<'a> Widget for Statusline<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 {
            return;
        }
        buf.set_style(area, Style::default().bg(self.theme.bg1));
        let line = self.build_line();
        // A single-row Paragraph would also work, but Buffer::set_line
        // avoids pulling in the `widgets::Paragraph` type for one line.
        buf.set_line(area.x, area.y, &line, area.width);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        Theme::gruvbox_dark()
    }

    fn sample_data() -> StatuslineData {
        StatuslineData {
            mode: Mode::Normal,
            file_name: "main.rs".to_string(),
            modified: true,
            filetype: "rust".to_string(),
            git_branch: Some("main".to_string()),
            lsp_status: None,
            cursor: Position::new(9, 3),
            line_count: 42,
        }
    }

    #[test]
    fn percent_through_file_reports_top_bot_all() {
        assert_eq!(percent_through_file(0, 100), "Top");
        assert_eq!(percent_through_file(99, 100), "Bot");
        assert_eq!(percent_through_file(0, 1), "All");
        assert_eq!(percent_through_file(0, 0), "All");
    }

    #[test]
    fn percent_through_file_computes_a_percentage_in_the_middle() {
        // line_count=101 -> denominator 100; cursor_line=50 -> 50%.
        assert_eq!(percent_through_file(50, 101), "50%");
    }

    #[test]
    fn mode_color_differs_between_normal_and_insert() {
        let t = theme();
        assert_ne!(mode_color(Mode::Normal, &t), mode_color(Mode::Insert, &t));
    }

    #[test]
    fn glyph_separators_are_used_when_enabled() {
        let data = sample_data();
        let t = theme();
        let sl = Statusline { data: &data, theme: &t, glyphs: true };
        let line = sl.build_line();
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains(GLYPH_SEP_RIGHT));
        assert!(text.contains(GLYPH_SEP_LEFT));
    }

    #[test]
    fn plain_separators_are_used_when_glyphs_disabled() {
        let data = sample_data();
        let t = theme();
        let sl = Statusline { data: &data, theme: &t, glyphs: false };
        let line = sl.build_line();
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(!text.contains(GLYPH_SEP_RIGHT));
        assert!(!text.contains(GLYPH_SEP_LEFT));
        assert!(text.contains(PLAIN_SEP_RIGHT));
    }

    #[test]
    fn mode_label_and_filename_appear_in_the_rendered_line() {
        let data = sample_data();
        let t = theme();
        let sl = Statusline { data: &data, theme: &t, glyphs: false };
        let line = sl.build_line();
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("NORMAL"));
        assert!(text.contains("main.rs"));
        assert!(text.contains("[+]"));
        assert!(text.contains("rust"));
        assert!(text.contains("main")); // git branch
    }

    #[test]
    fn git_branch_segment_is_omitted_when_none() {
        let mut data = sample_data();
        data.git_branch = None;
        let t = theme();
        let sl = Statusline { data: &data, theme: &t, glyphs: false };
        let line = sl.build_line();
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(!text.contains(" main "));
    }
}
