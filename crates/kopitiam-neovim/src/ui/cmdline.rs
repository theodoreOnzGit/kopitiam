//! The command line / message area at the bottom of the screen: `:` ex
//! commands, `/` and `?` searches, and status messages/errors.
//!
//! # Why this state lives in `ui/`, not `editor/`
//!
//! The *text* typed into `:`/`/`/`?` is business logic (parsing an ex
//! command, compiling a search pattern) and belongs entirely to
//! `crate::editor` — this module never interprets what's typed. What
//! belongs here is purely presentational: which prefix character to show,
//! where the cursor sits within the typed text, and how to colour an error
//! versus an informational message. [`CmdlineState`] is a rendering-only
//! mirror of "the editor is currently in `Mode::Command`", not a second
//! source of truth for command-line contents — see [`crate::ui::app`] for
//! how it's kept in sync with whatever the editor reports.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    widgets::Widget,
};

use crate::ui::theme::Theme;

/// Which of the three command-line prompts is active, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptKind {
    /// Not currently prompting for input; the message area (if any) is
    /// shown instead.
    #[default]
    None,
    /// `:` — ex command.
    Command,
    /// `/` — forward search.
    SearchForward,
    /// `?` — backward search.
    SearchBackward,
}

impl PromptKind {
    /// The literal prefix character shown before the typed text, matching
    /// vim exactly (`:`, `/`, `?`).
    pub fn prefix(self) -> Option<char> {
        match self {
            PromptKind::None => None,
            PromptKind::Command => Some(':'),
            PromptKind::SearchForward => Some('/'),
            PromptKind::SearchBackward => Some('?'),
        }
    }
}

/// A message shown in the command-line area when no prompt is active —
/// `:w` succeeding ("written"), an unknown-command error, and so on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum StatusMessage {
    #[default]
    None,
    Info(String),
    Error(String),
}

/// The full state of the bottom line, for one frame.
///
/// A prompt and a status message are mutually exclusive in vim (typing `:`
/// replaces whatever message was showing), which this type expresses
/// structurally: [`Cmdline::render`] always prefers `prompt` over `message`
/// when `prompt.kind` is not [`PromptKind::None`].
#[derive(Debug, Clone, Default)]
pub struct CmdlineState {
    pub kind: PromptKind,
    /// The text typed so far, not including the prefix character.
    pub input: String,
    /// Cursor position within `input`, in **grapheme** units — matching
    /// every other cursor position in kvim (see `crate::core::Position`'s
    /// docs on why graphemes, not bytes or chars).
    pub cursor: usize,
    pub message: StatusMessage,
    /// The `<Tab>` completion candidates being cycled and the selected index,
    /// for the wildmenu strip. Empty when nothing is being completed.
    pub completions: Vec<String>,
    pub completion_selected: usize,
}

/// The command-line/message-area widget: a single terminal row.
pub struct Cmdline<'a> {
    pub state: &'a CmdlineState,
    pub theme: &'a Theme,
}

impl<'a> Cmdline<'a> {
    /// The exact text this row shows, e.g. `:wq` or `Written.` — split out
    /// from [`Widget::render`] so it's assertable without a `Buffer`.
    pub fn text(&self) -> String {
        if let Some(prefix) = self.state.kind.prefix() {
            format!("{prefix}{}", self.state.input)
        } else {
            match &self.state.message {
                StatusMessage::None => String::new(),
                StatusMessage::Info(s) | StatusMessage::Error(s) => s.clone(),
            }
        }
    }

    /// The style this row should render in — errors get the theme's bright
    /// red, everything else (prompts, info messages, empty) the plain
    /// foreground, matching vim's `ErrorMsg` highlight convention of using
    /// colour only for the case that actually needs the user's attention.
    fn style(&self) -> Style {
        let fg = match (&self.state.kind, &self.state.message) {
            (PromptKind::None, StatusMessage::Error(_)) => self.theme.red_bright,
            _ => self.theme.fg,
        };
        Style::default().fg(fg).bg(self.theme.bg)
    }

    /// The screen column the cursor should be placed at while a prompt is
    /// active (`None` when there is no prompt — a message area doesn't
    /// have an editable cursor).
    ///
    /// Uses the grapheme-count of `input` up to `cursor`, via
    /// `unicode_segmentation`, for the same reason
    /// [`crate::ui::textarea::display_col_of_grapheme`] does: a byte or
    /// `char` offset would misplace the cursor the moment the typed text
    /// contains a multi-byte or multi-codepoint grapheme.
    pub fn cursor_column(&self, area: Rect) -> Option<u16> {
        use unicode_segmentation::UnicodeSegmentation;
        use unicode_width::UnicodeWidthStr;

        // Prompt prefixes are always ASCII (`:`, `/`, `?`), so they always
        // occupy exactly 1 display column.
        self.state.kind.prefix()?;
        let graphemes: Vec<&str> = self.state.input.graphemes(true).collect();
        let up_to = graphemes.iter().take(self.state.cursor).copied().collect::<String>();
        let col = 1 + up_to.width(); // +1 for the prefix character's own column.
        Some(area.x + col as u16)
    }
}

impl<'a> Widget for Cmdline<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 {
            return;
        }
        let style = self.style();
        buf.set_style(area, Style::default().bg(self.theme.bg));
        let text = self.text();
        buf.set_stringn(area.x, area.y, &text, area.width as usize, style);
    }
}

/// The `<Tab>` completion strip, vim's "wildmenu": a single horizontal row of
/// candidates with the selected one highlighted, drawn just above the command
/// line while a completion cycle is open. Modelled on vim, where the wildmenu
/// occupies the status-line row and disappears the moment the cycle ends.
pub struct Wildmenu<'a> {
    /// The candidate strings, in cycle order.
    pub items: &'a [String],
    /// Which candidate is currently selected (the one the command line shows).
    pub selected: usize,
    pub theme: &'a Theme,
}

impl<'a> Wildmenu<'a> {
    /// The rendered row as `(text, selected_byte_range)` — split out from
    /// [`Widget::render`] so a test can assert the exact string and which slice
    /// is highlighted without a `Buffer`.
    pub fn line(&self) -> String {
        self.items.join(" ")
    }
}

impl<'a> Widget for Wildmenu<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || self.items.is_empty() {
            return;
        }
        buf.set_style(area, Style::default().bg(self.theme.bg));
        // Paint each candidate, space-separated, highlighting the selected one
        // with reversed colours the way vim's `WildMenu` highlight does.
        let mut col = area.x;
        let right = area.x + area.width;
        for (i, item) in self.items.iter().enumerate() {
            if col >= right {
                break;
            }
            let style = if i == self.selected {
                Style::default().fg(self.theme.bg).bg(self.theme.yellow_bright)
            } else {
                Style::default().fg(self.theme.fg).bg(self.theme.bg)
            };
            let remaining = (right - col) as usize;
            let (end_col, _) = buf.set_stringn(col, area.y, item, remaining, style);
            col = end_col;
            if col < right {
                let (end_col, _) = buf.set_stringn(col, area.y, " ", (right - col) as usize, Style::default().bg(self.theme.bg));
                col = end_col;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        Theme::gruvbox_dark()
    }

    #[test]
    fn command_prompt_shows_colon_prefix() {
        let state = CmdlineState { kind: PromptKind::Command, input: "wq".to_string(), cursor: 2, ..Default::default() };
        let t = theme();
        let cl = Cmdline { state: &state, theme: &t };
        assert_eq!(cl.text(), ":wq");
    }

    #[test]
    fn search_prompts_show_slash_or_question_mark() {
        let t = theme();
        let fwd = CmdlineState { kind: PromptKind::SearchForward, input: "foo".to_string(), ..Default::default() };
        assert_eq!((Cmdline { state: &fwd, theme: &t }).text(), "/foo");
        let bwd = CmdlineState { kind: PromptKind::SearchBackward, input: "foo".to_string(), ..Default::default() };
        assert_eq!((Cmdline { state: &bwd, theme: &t }).text(), "?foo");
    }

    #[test]
    fn message_shown_when_no_prompt_is_active() {
        let state = CmdlineState { message: StatusMessage::Info("3 lines written".to_string()), ..Default::default() };
        let t = theme();
        let cl = Cmdline { state: &state, theme: &t };
        assert_eq!(cl.text(), "3 lines written");
    }

    #[test]
    fn error_messages_use_the_error_colour() {
        let state = CmdlineState { message: StatusMessage::Error("E492: not an editor command".to_string()), ..Default::default() };
        let t = theme();
        let cl = Cmdline { state: &state, theme: &t };
        assert_eq!(cl.style().fg, Some(t.red_bright));
    }

    #[test]
    fn info_messages_do_not_use_the_error_colour() {
        let state = CmdlineState { message: StatusMessage::Info("written".to_string()), ..Default::default() };
        let t = theme();
        let cl = Cmdline { state: &state, theme: &t };
        assert_ne!(cl.style().fg, Some(t.red_bright));
    }

    #[test]
    fn cursor_column_accounts_for_the_prefix_and_grapheme_position() {
        let state = CmdlineState { kind: PromptKind::Command, input: "wq".to_string(), cursor: 1, ..Default::default() };
        let t = theme();
        let cl = Cmdline { state: &state, theme: &t };
        let area = Rect { x: 0, y: 0, width: 40, height: 1 };
        // ':' at col 0, 'w' at col 1, cursor after 1 grapheme -> col 2.
        assert_eq!(cl.cursor_column(area), Some(2));
    }

    #[test]
    fn no_cursor_column_when_no_prompt_is_active() {
        let state = CmdlineState::default();
        let t = theme();
        let cl = Cmdline { state: &state, theme: &t };
        let area = Rect { x: 0, y: 0, width: 40, height: 1 };
        assert_eq!(cl.cursor_column(area), None);
    }
}
