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
use super::model_picker::short_status;
use super::theme::{CHILLI, DIM, GOLD, STEAM, TAN, USER};

/// The chat view's state: the model history, the composing input line, the live
/// reply being streamed, and scroll bookkeeping.
pub struct ChatView {
    /// A short, pre-rendered status line describing which rung answered.
    status: String,
    is_local: bool,
    max_tokens: Option<u32>,

    /// The last model-switch result, shown at the tail of the transcript. Kept
    /// as a field (not pushed into `history`) on purpose: it is a note about the
    /// session, not a turn, so it must never be sent to the model as context.
    switch_note: Option<String>,

    /// The adapter's **full** multi-line [`SelectedAdapter::notice`], shown at
    /// the top of the transcript whenever the stub is answering.
    ///
    /// The header only has room for one squeezed line, and that is where the
    /// old bug lived: a `.gguf` that was found but rejected by the loader looked
    /// exactly like having no model at all, with the loader's actual error
    /// dropped at the UI boundary. Keeping the whole notice here means the
    /// reason (the failing path, the parser's complaint, the three ways to fix
    /// it) is on screen where the user already is.
    stub_notice: Option<String>,

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
            switch_note: None,
            stub_notice: stub_notice(selected),
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

    /// Take on a model the user just picked in [`super::model_picker`], keeping
    /// the whole transcript.
    ///
    /// Only the *presentation* of which model is answering changes here — the
    /// adapter itself lives on [`super::App`], and the next [`ChatView::submit`]
    /// borrows whatever is there by then. Two consequences worth being exact
    /// about: (1) a reply already streaming finishes on the OLD model, because
    /// its worker thread holds its own `Arc`s to those weights; (2) the history
    /// carries over unchanged, so the new model sees the previous turns as
    /// context — switching does not start a fresh conversation.
    pub fn adopt_adapter(&mut self, selected: &SelectedAdapter, note: String) {
        self.status = short_status(selected);
        self.is_local = selected.is_local();
        self.stub_notice = stub_notice(selected);
        self.switch_note = Some(note);
        // Scroll back down so the note is actually seen, not left off-screen.
        self.stick_to_bottom = true;
        self.scroll = 0;
    }

    /// The footer hint pairs for the chat view.
    pub fn footer_hints(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("Enter", "send"),
            ("Ctrl-P", "pick model"),
            ("PgUp/PgDn", "scroll"),
            ("Esc", "home"),
            ("Ctrl-C", "quit"),
        ]
    }

    /// Handle one key. Returns the router [`Transition`] the parent should
    /// apply: `Esc` goes home, `Ctrl-C` quits, `Ctrl-P` opens the model picker,
    /// everything else stays.
    ///
    /// `Ctrl-P` (and not a bare `m`) because every unmodified printable char is
    /// typing — a plain letter must go into the message, never into a command.
    pub fn on_key(&mut self, key: KeyEvent, selected: &SelectedAdapter) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => return Transition::Quit,
            KeyCode::Char('p') if ctrl => return Transition::OpenModelPicker,
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
        let mut ended_without_done = false;
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
                // The sender is gone and no `Done`/`Error` ever arrived — the
                // generation thread died or was dropped mid-flight. Without a
                // marker this finalises as a blank assistant turn, which reads
                // as "the model replied with nothing" instead of "the model
                // stopped". Say which one it was.
                Err(TryRecvError::Disconnected) => {
                    finished = true;
                    ended_without_done = true;
                    break;
                }
            }
        }

        if finished {
            let mut reply = self.live_reply.take().unwrap_or_default();
            if ended_without_done {
                reply.push_str("\n[stream ended without a Done — the generation thread stopped]");
            }
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

        // When the stub is answering, say why right here — not just in the
        // squeezed header line. See `stub_notice`.
        if let Some(notice) = &self.stub_notice {
            for wrapped in wrap(notice, width) {
                out.push(Line::from(Span::styled(wrapped, Style::default().fg(CHILLI))));
            }
            out.push(Line::default());
        }

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

        // The model-switch note sits at the tail, after the last turn, so it
        // reads as "from here on, different model" — which is exactly true.
        if let Some(note) = &self.switch_note {
            for wrapped in wrap(&format!("— {note}"), width) {
                out.push(Line::from(Span::styled(
                    wrapped,
                    Style::default().fg(STEAM).add_modifier(Modifier::ITALIC),
                )));
            }
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

/// The full stub explanation to show in the transcript, or `None` when a real
/// model is answering and there is nothing to explain.
///
/// [`SelectedAdapter::notice`] already words both stub cases properly (which
/// file failed and why, or the three routes to getting weights) — the job here
/// is only to stop throwing it away, which is exactly what the UI used to do.
fn stub_notice(selected: &SelectedAdapter) -> Option<String> {
    (!selected.is_local()).then(|| selected.notice())
}

// The header's one-line status now comes from [`super::model_picker::short_status`]
// — it lives next to the picker because it has to tell "no model on disk" apart
// from "model found but won't load", which is the distinction the picker exists
// to resolve, and it points at the `Ctrl-P` this view now offers.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{FallbackReason, SelectedAdapter};
    use kopitiam_ai::EchoAdapter;

    /// A stub-backed adapter — enough to build a [`ChatView`] in a test without
    /// any weights on disk.
    fn echo() -> SelectedAdapter {
        SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::NoModelOnDisk {
                model_id: "test-model".into(),
                expected_store_path: std::path::PathBuf::from("/nowhere/x.gguf"),
            },
        }
    }

    /// A stub that got a file and could not load it — the case the UI used to
    /// render identically to "no model at all".
    fn load_failed() -> SelectedAdapter {
        SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::LoadFailed {
                source: std::path::PathBuf::from("/cache/broken.gguf"),
                error: "bad GGUF magic".into(),
            },
        }
    }

    /// Flatten rendered lines back to plain text, so a test can assert on what
    /// the user actually sees without reaching into ratatui's span structure.
    fn rendered(view: &ChatView, width: usize) -> String {
        view.transcript_lines(width)
            .iter()
            .map(|line| {
                line.spans.iter().map(|s| s.content.as_ref()).collect::<Vec<_>>().join("")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

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

    /// `Ctrl-P` must reach the router as a picker request, not get typed into
    /// the message. This is the entry point of the whole model-selection flow.
    #[test]
    fn ctrl_p_asks_the_router_for_the_model_picker() {
        let selected = echo();
        let mut view = ChatView::new("sys".into(), None, &selected);

        let transition =
            view.on_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL), &selected);

        assert!(matches!(transition, Transition::OpenModelPicker));
        assert!(view.input.is_empty(), "Ctrl-P must not land in the input line");
    }

    /// A plain `p` is still just typing — the modifier is what makes it a
    /// command, and getting this backwards would make the chat unusable.
    #[test]
    fn a_bare_p_is_still_typed_into_the_message() {
        let selected = echo();
        let mut view = ChatView::new("sys".into(), None, &selected);

        view.on_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::empty()), &selected);

        assert_eq!(view.input, "p");
    }

    /// Adopting a new model keeps the transcript and only re-labels who is
    /// answering — switching model must never wipe the conversation.
    #[test]
    fn adopting_a_model_keeps_the_history_and_updates_the_status() {
        let selected = echo();
        let mut view = ChatView::new("sys".into(), None, &selected);
        view.history.push(Message::user("earlier turn"));
        let before = view.history.len();

        view.adopt_adapter(&selected, "switched to whatever".into());

        assert_eq!(view.history.len(), before, "the transcript survives a switch");
        assert_eq!(view.switch_note.as_deref(), Some("switched to whatever"));
        assert!(!view.is_local);
        // The note is shown to the user but never sent to the model.
        assert!(!view.history.iter().any(|m| m.content.contains("switched to")));
        assert!(rendered(&view, 80).contains("switched to whatever"));
    }

    /// The loader's real complaint must reach the screen. Before this, every
    /// stub case rendered "no .gguf yet" and the error was dropped at the UI
    /// boundary — a broken model was indistinguishable from a missing one.
    #[test]
    fn a_failed_load_shows_the_loaders_own_error_in_the_transcript() {
        let broken = load_failed();
        let view = ChatView::new("sys".into(), None, &broken);

        let text = rendered(&view, 100);
        assert!(text.contains("broken.gguf"), "which file: {text}");
        assert!(text.contains("bad GGUF magic"), "and the loader's reason: {text}");

        // ...and a working model explains nothing, because there is nothing to
        // explain — the stub notice only appears for the stub.
        assert!(stub_notice(&broken).is_some());
        let text = rendered(&ChatView::new("sys".into(), None, &echo()), 100);
        assert!(text.contains("kopitiam models pull"), "the no-model routes: {text}");
    }

    /// A generation thread that dies without sending `Done` must not finalise as
    /// a blank assistant turn — that looks like the model answered with nothing.
    #[test]
    fn a_stream_that_dies_without_done_is_marked_not_left_blank() {
        let selected = echo();
        let mut view = ChatView::new("sys".into(), None, &selected);

        let (tx, rx) = std::sync::mpsc::channel::<StreamChunk>();
        tx.send(StreamChunk::Token("half a rep".into())).unwrap();
        drop(tx); // the worker died mid-flight
        view.stream = Some(rx);
        view.live_reply = Some(String::new());

        view.drain_stream();

        let last = view.history.last().expect("the reply was finalised");
        assert_eq!(last.role, Role::Assistant);
        assert!(last.content.contains("half a rep"), "keep what did arrive: {last:?}");
        assert!(last.content.contains("stream ended without a Done"), "{last:?}");
        assert!(!view.is_streaming(), "the turn is over");
    }

    /// A clean `Done` must NOT get the marker — it only means an abnormal end.
    #[test]
    fn a_clean_stream_is_finalised_without_the_marker() {
        let selected = echo();
        let mut view = ChatView::new("sys".into(), None, &selected);

        let (tx, rx) = std::sync::mpsc::channel::<StreamChunk>();
        tx.send(StreamChunk::Token("all done".into())).unwrap();
        tx.send(StreamChunk::Done).unwrap();
        view.stream = Some(rx);
        view.live_reply = Some(String::new());

        view.drain_stream();

        let last = view.history.last().unwrap();
        assert_eq!(last.content, "all done");
    }
}
