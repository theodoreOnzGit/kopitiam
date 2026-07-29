//! Qwen3.5 (a.k.a. `ornith`) response parser.
//!
//! **Upstream:** `model/parsers/qwen35.go` (ollama, MIT).
//!
//! ## Division of labour
//!
//! This parser owns **thinking only**. Everything after `</think>` -- including
//! the XML-ish `<tool_call>` bodies -- is handed straight to a
//! [`Qwen3CoderParser`], because qwen3.5 emits qwen3-coder's tool-call format.
//! Upstream embeds the coder parser by value for exactly this reason; we hold
//! one too.
//!
//! ## The 9b quirk that earned its own branch
//!
//! `qwen3.5:9b` sometimes **forgets `</think>`** and jumps straight into
//! `<tool_call>`. Upstream's fix is blunt and effective: when a `<tool_call>`
//! turns up while still in thinking mode, rewrite the buffer to
//! `thinking + "</think>" + "<tool_call>" + rest` and re-run the machine, which
//! then closes thinking through the normal path. Kept verbatim -- it looks like a
//! hack because it is one, and it is upstream's.

use crate::api::{Message, ThinkValue, Tool};

use super::qwen3coder::{Qwen3CoderParser, TOOL_CLOSE_TAG, TOOL_OPEN_TAG};
use super::{Parsed, Parser, ParserError, emit_unambiguous, overlap, split_at_tag};

/// **Upstream:** the `qwen35*Tag` consts.
const THINKING_OPEN_TAG: &str = "<think>";
const THINKING_CLOSE_TAG: &str = "</think>";

/// **Upstream:** `qwen35ParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    CollectingThinking,
    ThinkingDoneEatingWhitespace,
    CollectingContent,
}

/// **Upstream:** the `qwen35Event` sum type. No raw-tool-call variant here --
/// tool calls are the coder parser's problem.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Content(String),
    Thinking(String),
}

/// **Upstream:** `Qwen35Parser`.
#[derive(Debug, Default)]
pub struct Qwen35Parser {
    tool_parser: Qwen3CoderParser,
    state: State,
    buffer: String,
    /// One-shot latch: strip at most one redundant leading `<think>`, same reason
    /// as [`super::Qwen3Parser`] -- the prompt already opened thinking, but some
    /// checkpoints open it again.
    allow_leading_think_open_tag: bool,
}

impl Qwen35Parser {
    /// **Upstream:** `eatLeadingWhitespaceAndTransitionTo`.
    fn eat_leading_whitespace_and_transition_to(&mut self, next: State) -> (Vec<Event>, bool) {
        let trimmed = self.buffer.trim_start().to_string();
        self.buffer.clear();
        if trimmed.is_empty() {
            return (Vec::new(), false);
        }
        self.state = next;
        self.buffer.push_str(&trimmed);
        (Vec::new(), true)
    }

    /// **Upstream:** `maybeConsumeLeadingThinkOpenTag`.
    ///
    /// Returns `(handled, continue_now)`. `handled = true` means the caller must
    /// stop and honour `continue_now` -- either loop again (the tag was consumed
    /// and there is more behind it) or wait for the next chunk (the tag is still
    /// arriving).
    fn maybe_consume_leading_think_open_tag(&mut self, acc: &str) -> (bool, bool) {
        if !self.allow_leading_think_open_tag {
            return (false, false);
        }

        let trimmed = acc.trim_start().to_string();
        if let Some(after) = trimmed.strip_prefix(THINKING_OPEN_TAG) {
            let after = after.trim_start().to_string();
            self.buffer.clear();
            self.buffer.push_str(&after);
            if after.is_empty() {
                // Latch stays armed -- nothing followed the tag yet.
                return (true, false);
            }
            self.allow_leading_think_open_tag = false;
            return (true, true);
        }

        if THINKING_OPEN_TAG.starts_with(&trimmed) {
            // Could still grow into `<think>`; hold everything.
            return (true, false);
        }

        self.allow_leading_think_open_tag = false;
        (false, false)
    }

    /// **Upstream:** `(*Qwen35Parser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        let mut events = Vec::new();

        match self.state {
            State::CollectingThinking => {
                let acc = self.buffer.clone();

                let (handled, continue_now) = self.maybe_consume_leading_think_open_tag(&acc);
                if handled {
                    return (events, continue_now);
                }

                if acc.contains(THINKING_CLOSE_TAG) {
                    let (thinking, remaining) =
                        split_at_tag(&mut self.buffer, THINKING_CLOSE_TAG, true);
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    self.state = if remaining.is_empty() {
                        State::ThinkingDoneEatingWhitespace
                    } else {
                        State::CollectingContent
                    };
                    return (events, true);
                }

                let overlap_len =
                    overlap(&acc, THINKING_CLOSE_TAG).max(overlap(&acc, TOOL_OPEN_TAG));
                if overlap_len > 0 {
                    let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                    if !unambiguous.is_empty() {
                        events.push(Event::Thinking(unambiguous));
                    }
                    return (events, false);
                }

                if acc.contains(TOOL_OPEN_TAG) {
                    // The 9b quirk: `<tool_call>` with no `</think>` in front of
                    // it. Splice the missing close tag in and re-run -- the normal
                    // path then closes thinking correctly.
                    let (thinking, tooling) = split_at_tag(&mut self.buffer, TOOL_OPEN_TAG, true);
                    self.buffer.clear();
                    self.buffer.push_str(&thinking);
                    self.buffer.push_str(THINKING_CLOSE_TAG);
                    self.buffer.push_str(TOOL_OPEN_TAG);
                    self.buffer.push_str(&tooling);
                    return (events, true);
                }

                let unambiguous = emit_unambiguous(&mut self.buffer, 0);
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::ThinkingDoneEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingContent)
            }

            State::CollectingContent => {
                // Nothing clever here: hand the whole buffer to the coder parser,
                // which does its own ambiguity buffering for the tool tags.
                if self.buffer.is_empty() {
                    return (events, false);
                }
                let content = std::mem::take(&mut self.buffer);
                events.push(Event::Content(content));
                (events, false)
            }
        }
    }

    /// **Upstream:** `parseEvents`.
    fn parse_events(&mut self) -> Vec<Event> {
        let mut all = Vec::new();
        let mut keep_looping = true;
        while keep_looping {
            let (events, again) = self.eat();
            keep_looping = again;
            all.extend(events);
        }
        all
    }
}

impl Parser for Qwen35Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.buffer.clear();
        self.tool_parser = Qwen3CoderParser::default();
        self.tool_parser.init(tools.clone(), None, None);

        // `None` defaults to ON for qwen3.5 -- the opposite of plain `qwen3`.
        // That is upstream's choice per family, not a global default.
        let thinking_enabled = match think {
            Some(t) => t.enabled(),
            None => true,
        };

        // An assistant prefill means the caller already put words in the model's
        // mouth, so thinking is over before the stream even starts.
        let assistant_prefill = last_message
            .is_some_and(|m| m.role == "assistant" && !m.content.is_empty());

        if thinking_enabled && !assistant_prefill {
            self.state = State::CollectingThinking;
            self.allow_leading_think_open_tag = true;
        } else {
            self.state = State::CollectingContent;
            self.allow_leading_think_open_tag = false;
        }

        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let events = self.parse_events();

        let mut out = Parsed::default();
        for event in events {
            match event {
                Event::Content(c) => {
                    let (content, calls) = self.tool_parser.add_content(&c)?;
                    out.content.push_str(&content);
                    out.calls.extend(calls);
                }
                Event::Thinking(t) => out.thinking.push_str(&t),
            }
        }
        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![
            THINKING_OPEN_TAG,
            THINKING_CLOSE_TAG,
            TOOL_OPEN_TAG,
            TOOL_CLOSE_TAG,
        ]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ThinkValue;
    use serde_json::json;

    fn thinking() -> Qwen35Parser {
        let mut p = Qwen35Parser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        p
    }

    /// Upstream `TestQwen35ParserRespectsNoThink`.
    #[test]
    fn with_thinking_off_everything_is_content() {
        let mut p = Qwen35Parser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(false)));
        let got = p.add("Hello! How can I help you today?", true).expect("add");
        assert_eq!(got.content, "Hello! How can I help you today?");
        assert!(got.thinking.is_empty());
        assert!(got.calls.is_empty());
    }

    #[test]
    fn an_unspecified_think_value_defaults_to_thinking_on_for_this_family() {
        let mut p = Qwen35Parser::default();
        p.init(Vec::new(), None, None);
        let got = p.add("reasoning</think>answer", true).expect("add");
        assert_eq!(got.thinking, "reasoning");
        assert_eq!(got.content, "answer");
    }

    /// An assistant prefill means the caller already started the reply, so there
    /// is no thinking block left to open.
    #[test]
    fn an_assistant_prefill_starts_the_stream_in_content_mode() {
        let mut p = Qwen35Parser::default();
        let last = Message::new("assistant", "Sure, here goes:");
        p.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = p.add("plain text", true).expect("add");
        assert_eq!(got.content, "plain text");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn thinking_then_content_splits_at_the_close_tag() {
        let mut p = thinking();
        let got = p.add("hmm let me see</think>Here you go.", true).expect("add");
        assert_eq!(got.thinking, "hmm let me see");
        assert_eq!(got.content, "Here you go.");
    }

    #[test]
    fn a_redundant_leading_think_tag_is_stripped_exactly_once() {
        let mut p = thinking();
        let got = p
            .add("<think>\nhmm let me see</think>Here you go.", true)
            .expect("add");
        assert_eq!(got.thinking, "hmm let me see");
        assert_eq!(got.content, "Here you go.");
    }

    /// The 9b quirk: no `</think>` before `<tool_call>`. Upstream splices the
    /// missing tag in rather than letting the tool call leak into thinking.
    #[test]
    fn a_missing_thinking_close_tag_before_a_tool_call_is_repaired() {
        let mut p = thinking();
        let got = p
            .add(
                "I should check the weather<tool_call><function=get_weather><parameter=city>SG</parameter></function></tool_call>",
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "I should check the weather");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[0].function.arguments.get("city"), Some(&json!("SG")));
    }

    /// One byte at a time must give the same answer as one big chunk -- the whole
    /// point of the buffering rules.
    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = "reasoning</think>Hello <tool_call><function=f><parameter=a>1</parameter></function></tool_call>bye";

        let mut whole = thinking();
        let want = whole.add(input, true).expect("add");

        let mut streamed = thinking();
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let piece = &input[i..i + ch.len_utf8()];
            let part = streamed
                .add(piece, i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
            got.calls.extend(part.calls);
        }

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.calls.len(), want.calls.len());
    }

    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        let mut p = thinking();
        let a = p.add("thought</thi", false).expect("add");
        assert_eq!(a.thinking, "thought");
        assert!(a.content.is_empty());
        let b = p.add("nk>visible", true).expect("add");
        assert!(b.thinking.is_empty());
        assert_eq!(b.content, "visible");
    }
}
