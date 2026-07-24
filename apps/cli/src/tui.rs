//! `kopitiam tui` — a full-screen, kopitiam-themed chat TUI over the local
//! model.
//!
//! # What this is (and is not)
//!
//! This is the ratatui front-end onto the exact same streamed-chat path that
//! `kopitiam ai chat` (see [`crate::ai`]) drives from a plain terminal: pick an
//! adapter with [`crate::adapter::select_adapter`] (a real on-CPU
//! [`kopitiam_ai::LocalAdapter`] when a `.gguf` is on disk, otherwise the
//! deterministic [`kopitiam_ai::EchoAdapter`]), then hold a running
//! [`kopitiam_ai::Message`] history and, per turn, drain the adapter's token
//! stream into the transcript.
//!
//! Per `CLAUDE.md` ("clients own no business logic"), the UI invents nothing:
//! it owns key handling, layout and paint, and calls straight into
//! `kopitiam-ai`. Any grounding/routing (knowledge → native → local → cloud)
//! belongs one layer down in `kopitiam-workflow`, exactly as noted for
//! `ai chat`; wiring the TUI to that ladder is a deliberate follow-up, not this
//! slice.
//!
//! # Why polling, not a hand-rolled worker thread
//!
//! [`kopitiam_ai::ModelAdapter::stream`] already returns a
//! [`std::sync::mpsc::Receiver<StreamChunk>`] — the adapter spawns its own
//! actor thread and streams chunks back. So the render loop simply
//! [`Receiver::try_recv`]s each frame: tokens land in the live reply as they
//! arrive and the UI never blocks waiting on the model. When the channel
//! yields [`StreamChunk::Done`] (or disconnects), the turn is finalised into
//! history and the input unlocks.

use std::io::{self, Stdout};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use kopitiam_ai::{CompletionRequest, Message, Role, StreamChunk};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        execute,
        terminal::{
            EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
        },
    },
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Paragraph},
};

use crate::adapter::{SelectedAdapter, select_adapter};

/// Options for `kopitiam tui`. Mirrors [`crate::ai::ChatConfig`] so the two
/// front-ends seed the model identically.
#[derive(Args, Debug)]
pub struct TuiArgs {
    /// System prompt seeding the conversation. A gentle default is used when
    /// omitted.
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system: String,

    /// Cap on tokens generated per reply. Left to the adapter default when
    /// omitted.
    #[arg(long)]
    max_tokens: Option<u32>,
}

/// The persona seeded as the [`Role::System`] message. Kept short so it does
/// not crowd a small model's context; Singlish to match the CLI's voice.
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are KOPITIAM's local assistant. Answer concisely and helpfully.";

// ---- Coffeeshop palette -------------------------------------------------
// Warm kopitiam tones. Truecolor terminals render the Rgb values; 256/16-colour
// terminals approximate them. Nothing here depends on exact hues.
const GOLD: Color = Color::Rgb(224, 170, 90); // title, kopi-o
const TAN: Color = Color::Rgb(232, 205, 160); // assistant text
const PALM: Color = Color::Rgb(104, 168, 88); // palm fronds
const STEAM: Color = Color::Rgb(180, 180, 170); // steam / subtle
const USER: Color = Color::Rgb(120, 200, 220); // user text
const DIM: Color = Color::DarkGray;

/// Entry point for `kopitiam tui`. Selects the adapter (before touching the
/// terminal, so a slow model load or a fallback note happens on the normal
/// screen), then runs the UI with the terminal guaranteed to be restored even
/// on panic.
pub fn run(args: TuiArgs) -> Result<()> {
    let selected = select_adapter();

    let mut terminal = setup_terminal().context("failed to initialise the terminal")?;
    // Restore the terminal on panic too, so a crash never leaves the user in a
    // wrecked raw-mode alt-screen.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        previous_hook(info);
    }));

    let mut app = App::new(selected, &args);
    let result = app.run(&mut terminal);

    restore_terminal().ok();
    let _ = std::panic::take_hook();
    result
}

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn setup_terminal() -> Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    Ok(terminal)
}

/// Undo [`setup_terminal`]. Written to be idempotent and best-effort so it is
/// safe to call from the panic hook and again on the normal path.
fn restore_terminal() -> Result<()> {
    let mut stdout = io::stdout();
    execute!(stdout, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

/// The whole TUI state: the model history, the composing input line, the live
/// reply being streamed, and view/scroll bookkeeping.
struct App {
    /// The adapter chosen for the whole session (owns the model, if any).
    selected: SelectedAdapter,
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
    /// latest, which is where we snap back on every new token.
    scroll: u16,
    /// When true (the default), new content keeps the view pinned to the
    /// bottom. PageUp releases it; scrolling back to the bottom re-arms it.
    stick_to_bottom: bool,
    should_quit: bool,
}

impl App {
    fn new(selected: SelectedAdapter, args: &TuiArgs) -> Self {
        let status = short_status(&selected);
        let is_local = selected.is_local();
        Self {
            selected,
            status,
            is_local,
            max_tokens: args.max_tokens,
            history: vec![Message::system(args.system.clone())],
            input: String::new(),
            live_reply: None,
            stream: None,
            scroll: 0,
            stick_to_bottom: true,
            should_quit: false,
        }
    }

    /// The render + input loop. Each iteration: paint, drain any streamed
    /// tokens, then handle at most one input event (with a short poll timeout
    /// so streaming stays smooth even while the user is idle).
    fn run(&mut self, terminal: &mut Tui) -> Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| self.render(frame))?;
            self.drain_stream();

            if event::poll(Duration::from_millis(30))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                self.on_key(key);
            }
        }
        Ok(())
    }

    /// True while a reply is streaming — input is locked to one turn at a time.
    fn is_streaming(&self) -> bool {
        self.stream.is_some()
    }

    // ---- input ----------------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => self.should_quit = true,
            KeyCode::Esc => self.should_quit = true,
            KeyCode::Enter => self.submit(),
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
    fn submit(&mut self) {
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

        // `stream` returns a Receiver backed by the adapter's own actor thread;
        // we just poll it. The immutable borrow of `self.selected` ends as the
        // owned Receiver is returned.
        self.stream = Some(self.selected.adapter().stream(&request));
        self.live_reply = Some(String::new());
        self.pin_to_bottom();
    }

    // ---- streaming ------------------------------------------------------

    /// Pull every token available this frame without blocking. On `Done` (or a
    /// disconnected channel) the reply is finalised into history.
    fn drain_stream(&mut self) {
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
            // Record the assistant turn (even if empty) so the alternation
            // stays honest and the next message carries context.
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

    // ---- rendering ------------------------------------------------------

    fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // A banner only when there's comfortable room; otherwise give the
        // whole screen to the conversation.
        let show_banner = area.height >= 24 && area.width >= 54;
        let banner_h = if show_banner { BANNER.len() as u16 + 2 } else { 0 };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(banner_h),
                Constraint::Min(3),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(area);

        if show_banner {
            self.render_banner(frame, chunks[0]);
        }
        self.render_transcript(frame, chunks[1]);
        self.render_input(frame, chunks[2]);
        self.render_footer(frame, chunks[3]);
    }

    fn render_banner(&self, frame: &mut Frame, area: Rect) {
        let lines: Vec<Line> = BANNER
            .iter()
            .map(|(art, tone)| Line::from(Span::styled(*art, Style::default().fg(*tone))))
            .collect();
        let banner = Paragraph::new(Text::from(lines))
            .alignment(Alignment::Center)
            .block(Block::default());
        frame.render_widget(banner, area);
    }

    fn render_transcript(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM))
            .title(Span::styled(
                " kopitiam · chat ",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Wrap to the inner width so our own line count is exact, which makes
        // scroll maths (and bottom-pinning) precise regardless of wrapping.
        let width = inner.width.max(1) as usize;
        let lines = self.transcript_lines(width);
        let total = lines.len() as u16;
        let viewport = inner.height.max(1);

        // Clamp scroll to content, translating "rows from bottom" into a
        // top-anchored Paragraph offset.
        let max_scroll = total.saturating_sub(viewport);
        let from_bottom = self.scroll.min(max_scroll);
        self.scroll = from_bottom;
        let offset = max_scroll.saturating_sub(from_bottom);

        let paragraph = Paragraph::new(Text::from(lines)).scroll((offset, 0));
        frame.render_widget(paragraph, inner);
    }

    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let (label, label_tone) = if self.is_streaming() {
            (" kopi-o brewing… ", STEAM)
        } else {
            (" you ", USER)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(if self.is_streaming() { STEAM } else { USER }))
            .title(Span::styled(
                label,
                Style::default().fg(label_tone).add_modifier(Modifier::BOLD),
            ));

        let text = if self.is_streaming() {
            Line::from(Span::styled(
                "waiting for the reply to finish…",
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            ))
        } else {
            // A block cursor at the end of the composed line.
            Line::from(vec![
                Span::styled("› ", Style::default().fg(GOLD)),
                Span::raw(self.input.clone()),
                Span::styled("▌", Style::default().fg(GOLD)),
            ])
        };

        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let dot = if self.is_local { "●" } else { "○" };
        let status_tone = if self.is_local { PALM } else { STEAM };
        let footer = Line::from(vec![
            Span::styled(format!(" {dot} "), Style::default().fg(status_tone)),
            Span::styled(self.status.clone(), Style::default().fg(status_tone)),
            Span::styled("   │   ", Style::default().fg(DIM)),
            Span::styled(
                "Enter",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" send   ", Style::default().fg(DIM)),
            Span::styled(
                "PgUp/PgDn",
                Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" scroll   ", Style::default().fg(DIM)),
            Span::styled("Esc", Style::default().fg(GOLD).add_modifier(Modifier::BOLD)),
            Span::styled(" quit", Style::default().fg(DIM)),
        ]);
        frame.render_widget(Paragraph::new(footer), area);
    }

    /// Build the full transcript as exact, pre-wrapped styled lines: a welcome,
    /// then each user/assistant turn, then the live reply with a cursor.
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
            shown.push('▌'); // live cursor at the growing edge
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
        out.push(Line::from(Span::styled(
            wrapped,
            Style::default().fg(body_tone),
        )));
    }
    out.push(Line::default());
}

/// Word-wrap `text` to `width` columns, honouring existing newlines and
/// hard-splitting any single word longer than the width. Counts by `char`, not
/// byte, so multi-byte content wraps at sensible boundaries.
fn wrap(text: &str, width: usize) -> Vec<String> {
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

/// A compact one-line status for the footer, derived from the full
/// [`SelectedAdapter::notice`] (which is a multi-line paragraph unsuitable for a
/// single row).
fn short_status(selected: &SelectedAdapter) -> String {
    if selected.is_local() {
        // The notice's first line is "Using local model on CPU: <path>".
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

/// The kopitiam banner: two palm trees flanking a steaming cup of kopi-o, each
/// row tinted by its second field. Kept ASCII-only so it renders on any font.
#[rustfmt::skip]
const BANNER: &[(&str, Color)] = &[
    (r"   \\ | //          K O P I T I A M   ·   A I          \\ | //   ", GOLD),
    (r"  \ \|/ /        ~ your local kopi-o companion ~        \ \|/ /  ", STEAM),
    (r"   \\|//               (   (   )   )               \\|//   ",       PALM),
    (r"    \|/                 )   )   (                    \|/    ",      PALM),
    (r"     |                .-=============-.                |     ",     PALM),
    (r"     |                |               |               |     ",      GOLD),
    (r"    /|\               |    k o p i    |]             /|\    ",      GOLD),
    (r"   //|\\              |      -o-      |             //|\\   ",      GOLD),
    (r"  ///|\\\              \\_____________/             ///|\\\  ",     PALM),
];

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
