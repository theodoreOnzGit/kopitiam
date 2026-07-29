//! Qwen3-VL response parser (instruct and thinking variants).
//!
//! **Upstream:** `model/parsers/qwen3vl.go` (ollama, MIT).
//!
//! ## How it differs from [`super::Qwen3Parser`]
//!
//! Same tags, different tool-call payload and slightly different whitespace
//! handling:
//!
//! * the tool-call body is **plain JSON shaped like a `ToolCallFunction`**
//!   (`{"name": ..., "arguments": {...}}`), decoded straight into the api type
//!   -- no schema coercion, and, unlike qwen3, **no empty-name check**;
//! * there is no `<think>` *opening* tag in the state machine at all. The prompt
//!   opened thinking; output starts inside it. A stray `<think>` in the output
//!   would be treated as thinking text, not stripped;
//! * after a tool call closes, the parser eats leading whitespace before calling
//!   anything content -- upstream's `ToolCallDoneEatingWhitespace` state.
//!
//! ## A faithful wart
//!
//! Upstream's `Init` resets `tools` and `callIndex` but **not** the buffer. So a
//! parser reused across turns without a fresh struct carries leftover bytes into
//! the next generation. Ported as-is, because behaviour differences from the
//! oracle are how a port rots. Practically it does not bite: callers build a
//! parser per generation via `ParserForName`.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallFunction};

use super::qwen3coder::{TOOL_CLOSE_TAG, TOOL_OPEN_TAG};
use super::{Parsed, Parser, ParserError, chop, emit_unambiguous, overlap, split_at_tag};

/// **Upstream:** `thinkingCloseTag` in `qwen3vl.go`.
const THINKING_CLOSE_TAG: &str = "</think>";

/// **Upstream:** the `qwenParserState` consts declared at the top of `qwen3vl.go`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    CollectingThinkingContent,
    CollectingContent,
    CollectingToolContent,
    ThinkingDoneEatingWhitespace,
    ToolCallDoneEatingWhitespace,
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Content(String),
    Thinking(String),
    RawToolCall(String),
}

/// **Upstream:** `Qwen3VLParser`.
#[derive(Debug, Default)]
pub struct Qwen3VlParser {
    state: State,
    buffer: String,
    tools: Vec<Tool>,
    call_index: usize,
    has_thinking_support: bool,
}

impl Qwen3VlParser {
    /// `"qwen3-vl-instruct"` is `new(false)`; `"qwen3-vl-thinking"` is `new(true)`.
    pub fn new(has_thinking_support: bool) -> Self {
        Self {
            has_thinking_support,
            ..Default::default()
        }
    }

    /// **Upstream:** `setInitialState`.
    ///
    /// Note the prefill test is subtly different from qwen3.5's: the role must be
    /// `assistant` **and** the content non-empty. A prefill with empty content is
    /// not a prefill -- it is just an empty assistant turn, and thinking still
    /// opens.
    fn set_initial_state(&mut self, last_message: Option<&Message>) {
        if !self.has_thinking_support {
            self.state = State::CollectingContent;
            return;
        }
        let prefill = last_message.is_some_and(|m| m.role == "assistant");
        if prefill && last_message.is_some_and(|m| !m.content.is_empty()) {
            self.state = State::CollectingContent;
            return;
        }
        self.state = State::CollectingThinkingContent;
    }

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

    /// **Upstream:** `(*Qwen3VLParser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        let mut events = Vec::new();

        match self.state {
            State::CollectingContent => {
                if self.buffer.contains(TOOL_OPEN_TAG) {
                    // `trim_after = false`: the tool body keeps its leading
                    // whitespace, unlike qwen3's handling.
                    let (before, _) = split_at_tag(&mut self.buffer, TOOL_OPEN_TAG, false);
                    if !before.is_empty() {
                        events.push(Event::Content(before));
                    }
                    self.state = State::CollectingToolContent;
                    return (events, true);
                }
                let overlap_len = overlap(&self.buffer, TOOL_OPEN_TAG);
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Content(unambiguous));
                }
                (events, false)
            }

            State::CollectingToolContent => {
                let Some(idx) = self.buffer.find(TOOL_CLOSE_TAG) else {
                    return (events, false);
                };
                let (before, rest) = chop(&self.buffer, idx);
                let raw = before.to_string();
                let after = rest[TOOL_CLOSE_TAG.len()..].to_string();
                events.push(Event::RawToolCall(raw));
                self.buffer = after;
                self.state = State::ToolCallDoneEatingWhitespace;
                (events, true)
            }

            State::CollectingThinkingContent => {
                let acc = self.buffer.clone();
                let thinking_close_idx = acc.find(THINKING_CLOSE_TAG);
                let tool_open_idx = acc.find(TOOL_OPEN_TAG);

                // A tool call before `</think>` ends thinking.
                let tool_first = match (tool_open_idx, thinking_close_idx) {
                    (Some(t), Some(c)) => t < c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if tool_first {
                    let (before, _) = split_at_tag(&mut self.buffer, TOOL_OPEN_TAG, false);
                    if !before.is_empty() {
                        events.push(Event::Thinking(before));
                    }
                    self.state = State::CollectingToolContent;
                    return (events, true);
                }

                if thinking_close_idx.is_some() {
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
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::ThinkingDoneEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingContent)
            }

            State::ToolCallDoneEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingContent)
            }
        }
    }

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

impl Parser for Qwen3VlParser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        // Faithful wart: upstream does NOT clear the buffer here. See module docs.
        self.tools = tools.clone();
        self.call_index = 0;
        self.set_initial_state(last_message);
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let events = self.parse_events();

        let mut out = Parsed::default();
        for event in events {
            match event {
                Event::RawToolCall(raw) => out.calls.push(parse_json_tool_call(&raw)?),
                Event::Thinking(t) => out.thinking.push_str(&t),
                // TODO(upstream drifkin): interleaved content events in one turn
                // are naively concatenated, because the API cannot represent a
                // model emitting several messages per turn.
                Event::Content(c) => out.content.push_str(&c),
            }
        }

        // Indices are stamped after the whole batch, not as each call is parsed.
        // Same outcome as qwen3's stamp-as-you-go, kept in upstream's shape.
        for call in &mut out.calls {
            call.function.index = self.call_index;
            self.call_index += 1;
        }

        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![THINKING_CLOSE_TAG, TOOL_OPEN_TAG, TOOL_CLOSE_TAG]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        self.has_thinking_support
    }
}

/// **Upstream:** `parseJSONToolCall`. Decodes straight into a
/// [`ToolCallFunction`], so the wire keys are `name` / `arguments` / `index`.
///
/// Deliberately **no** empty-name check, unlike qwen3's `parseQwen3ToolCall`.
/// Upstream is inconsistent here; we follow it rather than tidy it, because a
/// caller that special-cases an empty name would then behave differently against
/// the same model output.
fn parse_json_tool_call(raw: &str) -> Result<ToolCall, ParserError> {
    let function: ToolCallFunction = serde_json::from_str(raw)?;
    Ok(ToolCall {
        id: String::new(),
        function,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn thinking() -> Qwen3VlParser {
        let mut p = Qwen3VlParser::new(true);
        p.init(Vec::new(), None, None);
        p
    }

    fn instruct() -> Qwen3VlParser {
        let mut p = Qwen3VlParser::new(false);
        p.init(Vec::new(), None, None);
        p
    }

    #[test]
    fn the_instruct_variant_starts_in_content_mode_and_reports_no_thinking() {
        let mut p = instruct();
        assert!(!p.has_thinking_support());
        let got = p.add("straight answer", true).expect("add");
        assert_eq!(got.content, "straight answer");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn the_thinking_variant_starts_inside_the_thinking_block() {
        let mut p = thinking();
        assert!(p.has_thinking_support());
        let got = p.add("weighing it up</think>the answer", true).expect("add");
        assert_eq!(got.thinking, "weighing it up");
        assert_eq!(got.content, "the answer");
    }

    /// A non-empty assistant prefill means thinking already closed.
    #[test]
    fn a_non_empty_assistant_prefill_starts_in_content_mode() {
        let mut p = Qwen3VlParser::new(true);
        let last = Message::new("assistant", "Well,");
        p.init(Vec::new(), Some(&last), None);
        let got = p.add("carrying on", true).expect("add");
        assert_eq!(got.content, "carrying on");
        assert!(got.thinking.is_empty());
    }

    /// ...but an EMPTY assistant turn is not a prefill, so thinking still opens.
    /// This is the subtle half of upstream's `setInitialState`.
    #[test]
    fn an_empty_assistant_turn_is_not_a_prefill() {
        let mut p = Qwen3VlParser::new(true);
        let last = Message::new("assistant", "");
        p.init(Vec::new(), Some(&last), None);
        let got = p.add("still reasoning</think>done", true).expect("add");
        assert_eq!(got.thinking, "still reasoning");
        assert_eq!(got.content, "done");
    }

    #[test]
    fn a_json_tool_call_is_decoded_and_indexed() {
        let mut p = instruct();
        let got = p
            .add(
                r#"<tool_call>{"name":"get_weather","arguments":{"city":"Singapore"}}</tool_call>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(
            got.calls[0].function.arguments.get("city"),
            Some(&json!("Singapore"))
        );
    }

    #[test]
    fn a_tool_call_before_the_thinking_close_tag_ends_thinking() {
        let mut p = thinking();
        let got = p
            .add(
                r#"let me look<tool_call>{"name":"f","arguments":{}}</tool_call>"#,
                true,
            )
            .expect("add");
        assert_eq!(got.thinking, "let me look");
        assert_eq!(got.calls.len(), 1);
    }

    #[test]
    fn whitespace_after_a_tool_call_is_eaten_before_content_resumes() {
        let mut p = instruct();
        let got = p
            .add(
                "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\n\n  after",
                true,
            )
            .expect("add");
        assert_eq!(got.content, "after");
    }

    #[test]
    fn parallel_calls_are_indexed_in_order_across_chunks() {
        let mut p = instruct();
        let mut all = Vec::new();
        all.extend(
            p.add(r#"<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","#, false)
                .expect("add")
                .calls,
        );
        all.extend(
            p.add(r#""arguments":{}}</tool_call>"#, true)
                .expect("add")
                .calls,
        );
        let got: Vec<_> = all
            .iter()
            .map(|c| (c.function.name.as_str(), c.function.index))
            .collect();
        assert_eq!(got, [("a", 0), ("b", 1)]);
    }

    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = r#"thinking bits</think>hello <tool_call>{"name":"f","arguments":{"x":1}}</tool_call>bye"#;

        let mut whole = thinking();
        let want = whole.add(input, true).expect("add");

        let mut streamed = thinking();
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = streamed
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
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
    fn a_tool_call_body_that_is_not_json_is_an_error() {
        let mut p = instruct();
        assert!(p.add("<tool_call>nope</tool_call>", true).is_err());
    }
}
