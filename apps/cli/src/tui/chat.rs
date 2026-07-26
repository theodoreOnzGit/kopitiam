//! The **AI Chat** view: the original `kopitiam tui` chat surface, unchanged in
//! behaviour, repackaged as one view inside the full app.
//!
//! This is the exact same streamed-chat path `kopitiam ai chat` drives: a
//! [`SelectedAdapter`] (a real on-CPU [`kopitiam_ai::LocalAdapter`] when a
//! `.gguf` is on disk, otherwise the deterministic [`kopitiam_ai::EchoAdapter`])
//! held by the parent [`super::App`], with a running [`Message`] history that
//! each turn drains the adapter's token stream into. The UI invents nothing —
//! it owns key handling, layout and paint, and calls straight into
//! `kopitiam-ai`.
//!
//! The adapter lives one level up (in [`super::App`]) so it is created lazily on
//! first entry to chat and shared for the session; [`ChatView`] borrows it for
//! [`ChatView::submit`]. Streaming stays a poll of the adapter's
//! [`std::sync::mpsc::Receiver`], so the render loop never blocks on the model.

use std::sync::mpsc::{Receiver, TryRecvError};

use kopitiam_ai::{CompletionRequest, Message, Role, StreamChunk};
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Paragraph},
};

use crate::adapter::SelectedAdapter;

use super::Transition;
use super::theme::{DIM, GOLD, STEAM, TAN, USER};

/// The chat view's state: the model history, the composing input line, the live
/// reply being streamed, and scroll bookkeeping.
pub struct ChatView {
    /// A short, pre-rendered status line describing which rung answered.
    status: String,
    is_local: bool,
    max_tokens: Option<u32>,

    /// The running conversation sent to the model. Seeded with the system
    /// persona; the [`Role::System`] entry is never displayed.
    history: Vec<Message>,
    /// What the user is currently typing.
    input: String,
    /// The reply streaming in right now, if a turn is in flight.
    live_reply: Option<String>,
    /// The adapter's token channel for the in-flight turn.
    stream: Option<Receiver<StreamChunk>>,

    /// Rows scrolled up from the bottom of the transcript. 0 = pinned to the
    /// latest.
    scroll: u16,
    stick_to_bottom: bool,
}

impl ChatView {
    /// Build a chat view seeded with `system` as the persona, reading the
    /// which-rung status straight off `selected`.
    pub fn new(system: String, max_tokens: Option<u32>, selected: &SelectedAdapter) -> Self {
        Self {
            status: short_status(selected),
            is_local: selected.is_local(),
            max_tokens,
            history: vec![Message::system(system)],
            input: String::new(),
            live_reply: None,
            stream: None,
            scroll: 0,
            stick_to_bottom: true,
        }
    }

    /// True while a reply is streaming — input is locked to one turn at a time.
    pub fn is_streaming(&self) -> bool {
        self.stream.is_some()
    }

    /// The footer hint pairs for the chat view.
    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("Enter", "send"),
            ("PgUp/PgDn", "scroll"),
            ("Esc", "home"),
            ("Ctrl-C", "quit"),
        ]
    }

    /// Handle one key. Returns the router [`Transition`] the parent should
    /// apply: `Esc` goes home, `Ctrl-C` quits, everything else stays.
    pub fn on_key(&mut self, key: KeyEvent, selected: &SelectedAdapter) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => return Transition::Quit,
            KeyCode::Esc => return Transition::Home,
            KeyCode::Enter => self.submit(selected),
            KeyCode::Backspace if !self.is_streaming() => {
                self.input.pop();
            }
            KeyCode::Char(c) if !ctrl && !self.is_streaming() => {
                self.input.push(c);
            }
            KeyCode::PageUp => self.scroll_up(5),
            KeyCode::PageDown => self.scroll_down(5),
            KeyCode::Up => self.scroll_up(1),
            KeyCode::Down => self.scroll_down(1),
            _ => {}
        }
        Transition::Stay
    }

    fn scroll_up(&mut self, n: u16) {
        self.scroll = self.scroll.saturating_add(n);
        self.stick_to_bottom = false;
    }

    fn scroll_down(&mut self, n: u16) {
        self.scroll = self.scroll.saturating_sub(n);
        if self.scroll == 0 {
            self.stick_to_bottom = true;
        }
    }

    /// Send the composed line to the model: append it to history, open the
    /// adapter's stream, and switch into streaming mode. No-op on a blank line
    /// or while a reply is already in flight.
    fn submit(&mut self, selected: &SelectedAdapter) {
        if self.is_streaming() {
            return;
        }
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.input.clear();
        self.history.push(Message::user(text));

        let mut request = CompletionRequest::new(self.history.clone());
        if let Some(max) = self.max_tokens {
            request = request.with_max_tokens(max);
        }

        self.stream = Some(selected.adapter().stream(&request));
        self.live_reply = Some(String::new());
        self.pin_to_bottom();
    }

    /// Pull every token available this frame without blocking. On `Done` (or a
    /// disconnected channel) the reply is finalised into history. Called each
    /// frame by the parent loop while the chat view is active.
    pub fn drain_stream(&mut self) {
        let Some(rx) = self.stream.take() else {
            return;
        };
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(StreamChunk::Token(token)) => {
                    if let Some(reply) = self.live_reply.as_mut() {
                        reply.push_str(&token);
                    }
                    self.pin_to_bottom();
                }
                Ok(StreamChunk::Done) => {
                    finished = true;
                    break;
                }
                Ok(StreamChunk::Error(message)) => {
                    if let Some(reply) = self.live_reply.as_mut() {
                        reply.push_str(&format!("\n[stream error: {message}]"));
                    }
                    finished = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }

        if finished {
            let reply = self.live_reply.take().unwrap_or_default();
            self.history.push(Message {
                role: Role::Assistant,
                content: reply,
            });
        } else {
            self.stream = Some(rx);
        }
    }

    fn pin_to_bottom(&mut self) {
        if self.stick_to_bottom {
            self.scroll = 0;
        }
    }

    /// Paint the chat view into `area`: a transcript pane above a one-line
    /// input box. The parent draws the global footer.
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(area);
        self.render_transcript(frame, chunks[0]);
        self.render_input(frame, chunks[1]);
    }

    fn render_transcript(&mut self, frame: &mut Frame, area: Rect) {
        let dot = if self.is_local { "●" } else { "○" };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(
                format!(" kopitiam · chat  {dot} {} ", self.status),
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let width = inner.width.max(1) as usize;
        let lines = self.transcript_lines(width);
        let total = lines.len() as u16;
        let viewport = inner.height.max(1);

        let max_scroll = total.saturating_sub(viewport);
        let from_bottom = self.scroll.min(max_scroll);
        self.scroll = from_bottom;
        let offset = max_scroll.saturating_sub(from_bottom);

        let paragraph = Paragraph::new(Text::from(lines)).scroll((offset, 0));
        frame.render_widget(paragraph, inner);
    }

    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let (label, tone) = if self.is_streaming() {
            (" kopi-o brewing… ", STEAM)
        } else {
            (" you ", USER)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(tone))
            .title(Span::styled(
                label,
                Style::default().fg(tone).add_modifier(Modifier::BOLD),
            ));

        let text = if self.is_streaming() {
            Line::from(Span::styled(
                "waiting for the reply to finish…",
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            ))
        } else {
            Line::from(vec![
                Span::styled("› ", Style::default().fg(GOLD)),
                Span::raw(self.input.clone()),
                Span::styled("▌", Style::default().fg(GOLD)),
            ])
        };

        frame.render_widget(Paragraph::new(text).block(block), area);
    }

    /// Build the full transcript as exact, pre-wrapped styled lines.
    fn transcript_lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut out: Vec<Line<'static>> = Vec::new();

        out.push(Line::from(Span::styled(
            "kopi-o> Welcome to the kopitiam lah! Ask me anything — I run on your machine.",
            Style::default().fg(TAN).add_modifier(Modifier::ITALIC),
        )));
        out.push(Line::default());

        for message in &self.history {
            match message.role {
                Role::System => continue,
                Role::User => push_turn(&mut out, "you", USER, &message.content, width),
                Role::Assistant => push_turn(&mut out, "kopi-o", GOLD, &message.content, width),
            }
        }

        if let Some(reply) = &self.live_reply {
            let mut shown = reply.clone();
            shown.push('▌');
            push_turn(&mut out, "kopi-o", GOLD, &shown, width);
        }

        out
    }
}

/// Append one speaker's turn: a bold header line, then the message body wrapped
/// to `width` and tinted for the speaker.
fn push_turn(out: &mut Vec<Line<'static>>, who: &str, tone: Color, body: &str, width: usize) {
    out.push(Line::from(Span::styled(
        format!("▍ {who}"),
        Style::default().fg(tone).add_modifier(Modifier::BOLD),
    )));
    let body_tone = if tone == USER { USER } else { TAN };
    for wrapped in wrap(body, width) {
        out.push(Line::from(Span::styled(wrapped, Style::default().fg(body_tone))));
    }
    out.push(Line::default());
}

/// Word-wrap `text` to `width` columns, honouring existing newlines and
/// hard-splitting any single word longer than the width. Counts by `char`, not
/// byte, so multi-byte content wraps at sensible boundaries.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    for raw in text.split('\n') {
        if raw.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut current_len = 0usize;
        for word in raw.split(' ') {
            let word_len = word.chars().count();
            if current_len == 0 {
                push_word(&mut lines, &mut current, &mut current_len, word, word_len, width);
            } else if current_len + 1 + word_len <= width {
                current.push(' ');
                current.push_str(word);
                current_len += 1 + word_len;
            } else {
                lines.push(std::mem::take(&mut current));
                current_len = 0;
                push_word(&mut lines, &mut current, &mut current_len, word, word_len, width);
            }
        }
        lines.push(current);
    }
    lines
}

/// Place `word` at the start of `current`, hard-splitting it across lines if it
/// alone exceeds `width`.
fn push_word(
    lines: &mut Vec<String>,
    current: &mut String,
    current_len: &mut usize,
    word: &str,
    word_len: usize,
    width: usize,
) {
    if word_len <= width {
        *current = word.to_string();
        *current_len = word_len;
        return;
    }
    let mut chars = word.chars().peekable();
    let mut chunk = String::new();
    let mut chunk_len = 0;
    while let Some(c) = chars.next() {
        chunk.push(c);
        chunk_len += 1;
        if chunk_len == width && chars.peek().is_some() {
            lines.push(std::mem::take(&mut chunk));
            chunk_len = 0;
        }
    }
    *current = chunk;
    *current_len = chunk_len;
}

/// A compact one-line status for the header, derived from the full
/// [`SelectedAdapter::notice`].
fn short_status(selected: &SelectedAdapter) -> String {
    if selected.is_local() {
        selected
            .notice()
            .lines()
            .next()
            .unwrap_or("local model on CPU")
            .to_string()
    } else {
        "echo stub — no .gguf yet (run `kopitiam models pull`)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_on_word_boundaries_within_width() {
        let lines = wrap("the quick brown fox", 9);
        assert!(lines.iter().all(|l| l.chars().count() <= 9), "{lines:?}");
        assert_eq!(lines.join(" "), "the quick brown fox");
    }

    #[test]
    fn wrap_hard_splits_a_word_longer_than_width() {
        let lines = wrap("supercalifragilistic", 5);
        assert!(lines.iter().all(|l| l.chars().count() <= 5), "{lines:?}");
        assert_eq!(lines.concat(), "supercalifragilistic");
    }

    #[test]
    fn wrap_preserves_blank_lines_from_newlines() {
        let lines = wrap("a\n\nb", 10);
        assert_eq!(lines, vec!["a".to_string(), String::new(), "b".to_string()]);
    }

    #[test]
    fn wrap_zero_width_is_a_noop_not_a_panic() {
        assert_eq!(wrap("anything", 0), vec!["anything".to_string()]);
    }
}
