//! The `ai` subcommand group — the maintainer's **testable AI interface**.
//!
//! Right now it has one action, `kopitiam ai chat`: an interactive,
//! **streamed** chat against the local model. This is phase 1 of
//! `temp_ai_design.md` §10.6 ("Chat TUI over `LocalAdapter`, streaming, no
//! tools — proves the async-actor loop"). The full ratatui `apps/tui` is a
//! later phase; this CLI command is the thing you can *run and talk to*
//! today, so the streaming path is exercised end to end with a real (or
//! echo-stubbed) model.
//!
//! # How it stays runnable everywhere
//!
//! Adapter choice is [`crate::adapter::select_adapter`]'s job, the same
//! decision every workflow command uses: a real on-CPU
//! [`kopitiam_ai::LocalAdapter`] when a `.gguf` is on disk, otherwise
//! [`kopitiam_ai::EchoAdapter`], the deterministic stub. So `kopitiam ai
//! chat` **always runs** — with no weights it echoes your line back, streamed
//! word by word, which still proves the streaming loop works offline.
//!
//! # Architectural note — phase 1 talks to the adapter directly
//!
//! This loop calls [`kopitiam_ai::ModelAdapter::stream`] straight, with no
//! knowledge assembly and no tools. The architectural target is to route chat
//! through `kopitiam-workflow`'s dispatch ladder (existing knowledge → native
//! Rust → local model → cloud) and its context builder, so the model is
//! *grounded* rather than answering cold — see `temp_ai_design.md` §2–§4 and
//! §10.3. That is a deliberate **follow-up bead**, not this phase: phase 1's
//! job is only to prove the streamed async-actor chat loop. Keeping the call
//! direct here also keeps the CLI honestly thin (`CLAUDE.md`: clients own no
//! business logic) — the routing belongs one layer down, not inlined here.
//!
//! # Why the loop is factored over `Read` + `Write`
//!
//! [`chat_loop`] takes an input [`BufRead`] and an output [`Write`] rather
//! than reaching for `stdin`/`stdout` itself. That's what makes it testable
//! **headlessly**: a test feeds a scripted script of lines through a byte
//! slice and captures the streamed output in a `Vec<u8>`, with no terminal
//! and no real model (the [`kopitiam_ai::EchoAdapter`] gives deterministic
//! output). The real command ([`run`]) just passes `stdin().lock()` and
//! `stdout().lock()`.

use std::io::{BufRead, Write};

use anyhow::Result;
use clap::{Args, Subcommand};
use kopitiam_ai::{CompletionRequest, Message, ModelAdapter, Role, StreamChunk};

use crate::adapter::select_adapter;

/// Options for `kopitiam ai`.
#[derive(Args, Debug)]
pub struct AiArgs {
    #[command(subcommand)]
    command: AiCommand,
}

/// The actions under `kopitiam ai`.
#[derive(Subcommand, Debug)]
enum AiCommand {
    /// Chat with the local model, streamed token by token.
    ///
    /// Type a line, press enter, watch the reply stream in. `/quit` (or
    /// `/exit`, or an EOF / Ctrl-D) ends the session. With no local `.gguf`
    /// on disk this echoes your line back via the deterministic stub, so it
    /// runs even with no weights and no network.
    Chat(ChatArgs),
}

/// Options for `kopitiam ai chat`.
#[derive(Args, Debug)]
pub struct ChatArgs {
    /// System prompt to seed the conversation with. A gentle default is used
    /// when omitted.
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system: String,

    /// Cap on tokens generated per reply. Left to the model/adapter default
    /// when omitted.
    #[arg(long)]
    max_tokens: Option<u32>,
}

/// The default persona seeded as the [`Role::System`] message. Singlish to
/// match the CLI's voice; kept short so it doesn't crowd a small model's
/// context.
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are KOPITIAM's local assistant. Answer concisely and helpfully.";

/// Entry point for `kopitiam ai <command>`. Wires real stdin/stdout into the
/// factored [`chat_loop`] and prints which adapter answered.
pub fn run(args: AiArgs) -> Result<()> {
    match args.command {
        AiCommand::Chat(chat_args) => run_chat(chat_args),
    }
}

/// Implements `kopitiam ai chat`: pick the adapter, announce it, then run the
/// interactive loop over the real terminal.
fn run_chat(args: ChatArgs) -> Result<()> {
    let selected = select_adapter();
    // The adapter note goes to stderr so it's visible but never mixed into a
    // piped transcript on stdout — the same convention `kopitiam plan` uses.
    eprintln!("{}", selected.notice());
    eprintln!("Chat is streamed token-by-token. Type your message and press enter.");
    eprintln!("End with /quit, /exit, or Ctrl-D.");
    eprintln!();

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let config = ChatConfig { system: args.system, max_tokens: args.max_tokens };
    chat_loop(selected.adapter(), config, stdin.lock(), stdout.lock())
}

/// The knobs [`chat_loop`] runs with, separated from clap's [`ChatArgs`] so
/// the loop can be driven directly from a test without constructing CLI args.
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// The system prompt seeding the conversation.
    pub system: String,
    /// Optional per-reply token cap.
    pub max_tokens: Option<u32>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self { system: DEFAULT_SYSTEM_PROMPT.to_string(), max_tokens: None }
    }
}

/// One line the user typed, classified. Splitting the read+classify step out
/// of [`chat_loop`] keeps the loop's control flow legible and lets the
/// quit-word set be tested on its own.
enum Line {
    /// A real message to send to the model.
    Message(String),
    /// A blank line — ignore, re-prompt.
    Blank,
    /// The user asked to quit (`/quit`, `/exit`, `:q`).
    Quit,
    /// End of input (EOF / Ctrl-D) — also ends the session.
    Eof,
}

/// Classifies one raw input line into a [`Line`]. Quit words are matched on
/// the trimmed line so trailing newlines/spaces don't hide them.
fn classify(raw: &str) -> Line {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Line::Blank;
    }
    match trimmed {
        "/quit" | "/exit" | ":q" => Line::Quit,
        _ => Line::Message(trimmed.to_string()),
    }
}

/// The interactive chat loop, factored over an input [`BufRead`] and an output
/// [`Write`] so it runs identically against a real terminal and against a
/// scripted test buffer.
///
/// Each turn: prompt, read a line, and — unless it's blank or a quit word —
/// build a [`CompletionRequest`] carrying the full conversation so far, then
/// **stream** the reply, writing each [`StreamChunk::Token`] to `output` the
/// moment it arrives (flushing so a real terminal shows it live). The
/// assistant's reply is accumulated and appended to the history, so the model
/// sees prior turns on the next message.
///
/// A [`StreamChunk::Error`] mid-reply is printed inline and the turn ends;
/// the session continues (one bad generation shouldn't kill the chat). EOF or
/// a quit word ends the loop cleanly.
///
/// Returns `Err` only on a genuine I/O failure writing to `output` or reading
/// from `input` — never for anything the model or adapter does (adapter
/// trouble surfaces as a `StreamChunk::Error`, which is data, not an error).
pub fn chat_loop<R: BufRead, W: Write>(
    adapter: &dyn ModelAdapter,
    config: ChatConfig,
    mut input: R,
    mut output: W,
) -> Result<()> {
    // The running conversation. Seeded with the system persona; each user
    // line and each completed reply is appended, so the model sees context.
    let mut history: Vec<Message> = vec![Message::system(config.system.clone())];

    loop {
        write!(output, "\nyou> ")?;
        output.flush()?;

        let mut raw = String::new();
        let read = input.read_line(&mut raw)?;
        let line = if read == 0 { Line::Eof } else { classify(&raw) };

        match line {
            Line::Eof | Line::Quit => break,
            Line::Blank => continue,
            Line::Message(text) => {
                history.push(Message::user(text));

                let mut request = CompletionRequest::new(history.clone());
                if let Some(max) = config.max_tokens {
                    request = request.with_max_tokens(max);
                }

                write!(output, "kopi> ")?;
                output.flush()?;

                // Drain the stream, rendering each token live and building up
                // the full reply for the history.
                let mut reply = String::new();
                for chunk in adapter.stream(&request) {
                    match chunk {
                        StreamChunk::Token(token) => {
                            reply.push_str(&token);
                            write!(output, "{token}")?;
                            output.flush()?;
                        }
                        StreamChunk::Done => {
                            writeln!(output)?;
                            break;
                        }
                        StreamChunk::Error(message) => {
                            writeln!(output, "\n[stream error: {message}]")?;
                            break;
                        }
                    }
                }

                // Record the assistant turn so the next message has context.
                // An empty reply (e.g. the model emitted EOS immediately) is
                // still recorded as a Role::Assistant turn to keep the
                // alternation honest.
                history.push(Message { role: Role::Assistant, content: reply });
            }
        }
    }

    writeln!(output, "\nbye lah!")?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kopitiam_ai::EchoAdapter;

    /// Drives [`chat_loop`] headlessly against the deterministic
    /// [`EchoAdapter`]: two scripted lines then `/quit`. Asserts the echoed
    /// text streams back for both lines, **in order**, and the loop exits
    /// cleanly on the quit word. This is the maintainer's "make sure I have a
    /// testable interface" made provable — the whole streamed chat path,
    /// exercised with no terminal and no model weights.
    #[test]
    fn headless_chat_streams_echoed_replies_in_order_then_quits() {
        let script = "hello world\nsecond message here\n/quit\n";
        let mut output: Vec<u8> = Vec::new();

        chat_loop(&EchoAdapter, ChatConfig::default(), script.as_bytes(), &mut output).unwrap();

        let transcript = String::from_utf8(output).expect("output is UTF-8");

        // Both user lines came back echoed (streamed), in the order sent.
        let first = transcript.find("hello world").expect("first reply echoed");
        let second = transcript.find("second message here").expect("second reply echoed");
        assert!(first < second, "replies must appear in the order the lines were sent");

        // Each reply was rendered on its own `kopi>` line — two turns, two
        // prompts (the loop prints one per non-blank, non-quit line).
        assert_eq!(transcript.matches("kopi> ").count(), 2);

        // The loop terminated on /quit, printing the sign-off.
        assert!(transcript.contains("bye lah!"), "loop must exit cleanly on /quit");
    }

    /// EOF (empty read, i.e. Ctrl-D) ends the loop just like a quit word —
    /// no panic, clean sign-off — even mid-session with no quit word typed.
    #[test]
    fn eof_ends_the_loop_cleanly() {
        let script = "just one line\n"; // no /quit; stream then EOF
        let mut output: Vec<u8> = Vec::new();

        chat_loop(&EchoAdapter, ChatConfig::default(), script.as_bytes(), &mut output).unwrap();

        let transcript = String::from_utf8(output).unwrap();
        assert!(transcript.contains("just one line"), "the one line should have been echoed");
        assert!(transcript.contains("bye lah!"), "EOF must end the loop cleanly");
    }

    /// Blank lines are skipped, not sent: a blank between two messages
    /// produces exactly two `kopi>` replies, not three.
    #[test]
    fn blank_lines_are_ignored() {
        let script = "one\n\n   \ntwo\n/quit\n";
        let mut output: Vec<u8> = Vec::new();

        chat_loop(&EchoAdapter, ChatConfig::default(), script.as_bytes(), &mut output).unwrap();

        let transcript = String::from_utf8(output).unwrap();
        assert_eq!(transcript.matches("kopi> ").count(), 2, "blank lines must not trigger a reply");
    }

    #[test]
    fn classify_recognises_every_quit_word() {
        assert!(matches!(classify("/quit"), Line::Quit));
        assert!(matches!(classify("/exit\n"), Line::Quit));
        assert!(matches!(classify("  :q  "), Line::Quit));
        assert!(matches!(classify(""), Line::Blank));
        assert!(matches!(classify("   \n"), Line::Blank));
        assert!(matches!(classify("hello"), Line::Message(_)));
    }
}
