//! Qwen3 (and Qwen3-thinking) response parser.
//!
//! **Upstream:** `model/parsers/qwen3.go` (ollama, MIT).
//!
//! ## The one thing that makes Qwen3 different
//!
//! Qwen3's **prompt already ends with `<think>`** when thinking is on. So the
//! model's output starts *inside* the thinking block with no opening tag to find
//! -- which is why [`Qwen3Parser::init`] starts the machine in
//! [`State::CollectingThinking`] rather than hunting for `<think>`.
//!
//! But some checkpoints emit the opening `<think>` **anyway**, redundantly. That
//! is what `maybe_thinking_open_at_bol` is for: strip **exactly one** leading
//! `<think>` if it shows up, then never look again. Getting this wrong either
//! leaks a literal `<think>` into the thinking text, or (worse) eats a `<think>`
//! that the model meant as content.
//!
//! ## Tool calls can start before `</think>` closes
//!
//! If `<tool_call>` appears **before** `</think>` in the buffer, upstream treats
//! that as the end of thinking and jumps straight into tool-call mode. Models do
//! this. Do not "fix" it.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, emit_unambiguous, overlap, split_at_tag};

/// **Upstream:** the four `qwen3*Tag` consts in `qwen3.go`. These are the exact
/// literals Qwen3 was fine-tuned to emit -- not a convention we chose, and not
/// interchangeable with the `<|tool_call>` spelling other families use.
const THINKING_OPEN_TAG: &str = "<think>";
const THINKING_CLOSE_TAG: &str = "</think>";
const TOOL_OPEN_TAG: &str = "<tool_call>";
const TOOL_CLOSE_TAG: &str = "</tool_call>";

/// **Upstream:** `qwen3ParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    /// Only reachable if somebody constructs the parser without `init`. Kept
    /// because upstream keeps it: it hunts for a leading `<think>` and falls back
    /// to content mode if the first non-whitespace thing is not one.
    #[default]
    LookingForThinkingOpen,
    /// Just consumed `<think>` and the buffer went empty -- swallow whatever
    /// whitespace arrives next before calling it thinking.
    ThinkingStartedEatingWhitespace,
    CollectingThinking,
    ThinkingDoneEatingWhitespace,
    CollectingContent,
    ToolStartedEatingWhitespace,
    CollectingToolContent,
}

/// **Upstream:** the `qwen3Event` sum type. Go needs an interface + four empty
/// marker methods for this; Rust just needs an enum.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Content(String),
    Thinking(String),
    RawToolCall(String),
}

/// **Upstream:** `Qwen3Parser`.
#[derive(Debug, Default)]
pub struct Qwen3Parser {
    state: State,
    buffer: String,
    tools: Vec<Tool>,
    call_index: usize,
    has_thinking_support: bool,
    default_thinking: bool,
    /// One-shot latch: a redundant leading `<think>` may still be stripped.
    maybe_thinking_open_at_bol: bool,
}

impl Qwen3Parser {
    /// `"qwen3"` is `new(false, false)`; `"qwen3-thinking"` is `new(true, true)`.
    /// **Upstream:** the two `ParserForName` arms that differ only in these flags.
    pub fn new(has_thinking_support: bool, default_thinking: bool) -> Self {
        Self {
            has_thinking_support,
            default_thinking,
            ..Default::default()
        }
    }

    /// **Upstream:** `eatLeadingWhitespaceAndTransitionTo`.
    ///
    /// Note it returns `(no events, false)` when the buffer is *all* whitespace:
    /// the whitespace is dropped and we stay put, waiting for real content. Only
    /// once something non-whitespace turns up do we move on -- and the buffer is
    /// rewritten to the trimmed remainder, so the leading whitespace never
    /// reaches the output.
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

    fn split_at_tag(&mut self, tag: &str, trim_after: bool) -> (String, String) {
        split_at_tag(&mut self.buffer, tag, trim_after)
    }

    /// Run [`Self::eat`] until it says "nothing more to do right now".
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

    /// One step of the state machine.
    ///
    /// **Upstream:** `(*Qwen3Parser).eat`. Returns the events that became
    /// unambiguous, plus whether a state transition happened and so another step
    /// might produce more. Upstream `panic("unreachable")`s on an unknown state;
    /// we cannot have an unknown state, because [`State`] is an enum.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        let mut events = Vec::new();

        match self.state {
            State::LookingForThinkingOpen => {
                let trimmed = self.buffer.trim_start().to_string();
                if let Some(after) = trimmed.strip_prefix(THINKING_OPEN_TAG) {
                    let after = after.trim_start().to_string();
                    self.buffer.clear();
                    self.buffer.push_str(&after);
                    self.state = if after.is_empty() {
                        State::ThinkingStartedEatingWhitespace
                    } else {
                        State::CollectingThinking
                    };
                    return (events, true);
                }
                // Still could grow INTO `<think>` -- e.g. the buffer is `"<thi"`.
                // Hold everything and wait for more.
                if THINKING_OPEN_TAG.starts_with(&trimmed) {
                    return (events, false);
                }
                self.state = State::CollectingContent;
                (events, true)
            }

            State::ThinkingStartedEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingThinking)
            }

            State::CollectingThinking => {
                let acc = self.buffer.clone();

                // Strip at most one redundant leading `<think>`. See the module
                // docs: the prompt already opened thinking, but some checkpoints
                // open it a second time.
                if self.maybe_thinking_open_at_bol {
                    let trimmed = acc.trim_start().to_string();
                    if let Some(after) = trimmed.strip_prefix(THINKING_OPEN_TAG) {
                        let after = after.trim_start().to_string();
                        self.buffer.clear();
                        self.buffer.push_str(&after);
                        if after.is_empty() {
                            // Nothing after the tag yet -- keep the latch armed so
                            // the next chunk still gets checked.
                            return (events, false);
                        }
                        self.maybe_thinking_open_at_bol = false;
                        return (events, true);
                    }
                    if THINKING_OPEN_TAG.starts_with(&trimmed) {
                        // Might still become `<think>`. Buffer, don't emit.
                        return (events, false);
                    }
                    self.maybe_thinking_open_at_bol = false;
                }

                let thinking_close_idx = acc.find(THINKING_CLOSE_TAG);
                let tool_open_idx = acc.find(TOOL_OPEN_TAG);

                // A tool call that starts before `</think>` ends thinking. Models
                // really do this -- upstream's comment, and it is load-bearing.
                let tool_first = match (tool_open_idx, thinking_close_idx) {
                    (Some(t), Some(c)) => t < c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if tool_first {
                    let (before, after) = self.split_at_tag(TOOL_OPEN_TAG, true);
                    if !before.is_empty() {
                        events.push(Event::Thinking(before));
                    }
                    self.state = if after.is_empty() {
                        State::ToolStartedEatingWhitespace
                    } else {
                        State::CollectingToolContent
                    };
                    return (events, true);
                }

                if thinking_close_idx.is_some() {
                    let (thinking, remaining) = self.split_at_tag(THINKING_CLOSE_TAG, true);
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

                // Neither tag is complete. Whatever tail could still grow into
                // `</think>` OR `<tool_call>` stays buffered -- the longer of the
                // two overlaps wins, because the shorter one would emit bytes the
                // longer one still needs.
                let overlap_len = overlap(&acc, THINKING_CLOSE_TAG).max(overlap(&acc, TOOL_OPEN_TAG));
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::ThinkingDoneEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingContent)
            }

            State::CollectingContent => {
                let acc = self.buffer.clone();
                if acc.contains(TOOL_OPEN_TAG) {
                    let (before, after) = self.split_at_tag(TOOL_OPEN_TAG, true);
                    if !before.is_empty() {
                        events.push(Event::Content(before));
                    }
                    self.state = if after.is_empty() {
                        State::ToolStartedEatingWhitespace
                    } else {
                        State::CollectingToolContent
                    };
                    return (events, true);
                }
                let overlap_len = overlap(&acc, TOOL_OPEN_TAG);
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Content(unambiguous));
                }
                (events, false)
            }

            State::ToolStartedEatingWhitespace => {
                self.eat_leading_whitespace_and_transition_to(State::CollectingToolContent)
            }

            State::CollectingToolContent => {
                // No overlap check here on purpose. We never stream a half-parsed
                // tool call back to anyone, so there is nothing to be eager about
                // -- just wait for the whole closing tag.
                if self.buffer.contains(TOOL_CLOSE_TAG) {
                    let (tool_content, _) = self.split_at_tag(TOOL_CLOSE_TAG, true);
                    events.push(Event::RawToolCall(tool_content));
                    self.state = State::CollectingContent;
                    return (events, true);
                }
                (events, false)
            }
        }
    }
}

impl Parser for Qwen3Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tools = tools.clone();
        self.buffer.clear();
        self.call_index = 0;

        // `None` means "the caller never said" -- which is NOT the same as
        // `Some(false)`. Upstream falls back to the family default only in the
        // `None` case.
        let thinking_enabled = match think {
            Some(t) => t.enabled(),
            None => self.default_thinking,
        };

        if self.has_thinking_support && thinking_enabled {
            self.state = State::CollectingThinking;
            self.maybe_thinking_open_at_bol = true;
        } else {
            self.state = State::CollectingContent;
            self.maybe_thinking_open_at_bol = false;
        }
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let events = self.parse_events();

        let mut out = Parsed::default();
        for event in events {
            match event {
                Event::RawToolCall(raw) => {
                    let mut call = parse_qwen3_tool_call(&raw)?;
                    call.function.index = self.call_index;
                    self.call_index += 1;
                    out.calls.push(call);
                }
                Event::Thinking(t) => out.thinking.push_str(&t),
                Event::Content(c) => out.content.push_str(&c),
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
        self.has_thinking_support
    }
}

/// Parse one `<tool_call>...</tool_call>` body.
///
/// **Upstream:** `parseQwen3ToolCall`. Qwen3 puts plain JSON in there
/// (`{"name": ..., "arguments": {...}}`), so unlike qwen3-coder there is **no**
/// schema-driven type coercion -- upstream's own `_ = tools` says as much. The
/// argument order the model chose survives, because [`ToolCallArguments`] is
/// insertion-ordered.
fn parse_qwen3_tool_call(raw: &str) -> Result<ToolCall, ParserError> {
    #[derive(serde::Deserialize)]
    struct Raw {
        #[serde(default)]
        name: String,
        #[serde(default)]
        arguments: ToolCallArguments,
    }

    let parsed: Raw = serde_json::from_str(raw)?;
    if parsed.name.is_empty() {
        return Err(ParserError::EmptyFunctionName);
    }

    Ok(ToolCall {
        id: String::new(),
        function: ToolCallFunction {
            index: 0,
            name: parsed.name,
            arguments: parsed.arguments,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ThinkValue;
    use serde_json::json;

    fn thinking_parser() -> Qwen3Parser {
        let mut p = Qwen3Parser::new(true, true);
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        p
    }

    fn instruct_parser() -> Qwen3Parser {
        let mut p = Qwen3Parser::new(false, false);
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(false)));
        p
    }

    /// Upstream `TestQwen3ParserThinkingEnabled`.
    #[test]
    fn thinking_runs_until_the_close_tag_then_the_rest_is_content() {
        let mut p = thinking_parser();
        let got = p.add("Let me think...</think>Answer.", true).expect("add");
        assert_eq!(got.thinking, "Let me think...");
        assert_eq!(got.content, "Answer.");
        assert!(got.calls.is_empty());
    }

    /// Upstream `TestQwen3ParserThinkingEnabledWithExplicitOpeningTag` -- the
    /// redundant `<think>` some checkpoints emit even though the prompt already
    /// opened thinking.
    #[test]
    fn a_redundant_leading_think_tag_is_stripped_exactly_once() {
        let mut p = thinking_parser();
        let got = p
            .add("<think>\nLet me think...</think>Answer.", true)
            .expect("add");
        assert_eq!(got.thinking, "Let me think...");
        assert_eq!(got.content, "Answer.");
        assert!(got.calls.is_empty());
    }

    /// Upstream `TestQwen3ParserThinkingEnabledWithSplitOpeningTag`.
    #[test]
    fn an_opening_tag_split_across_chunks_emits_nothing_until_it_completes() {
        let mut p = thinking_parser();
        let first = p.add("<thi", false).expect("add");
        assert!(first.is_empty(), "a half tag must never be emitted: {first:?}");

        let got = p
            .add("nk>Let me think...</think>Answer.", true)
            .expect("add");
        assert_eq!(got.thinking, "Let me think...");
        assert_eq!(got.content, "Answer.");
    }

    /// Upstream `TestQwen3ParserThinkingDisabled`.
    #[test]
    fn with_thinking_off_everything_is_content() {
        let mut p = instruct_parser();
        let got = p.add("Direct answer", true).expect("add");
        assert_eq!(got.content, "Direct answer");
        assert!(got.thinking.is_empty());
    }

    /// Upstream `TestQwen3ParserNilThinkDefaultsToContentForInstructParser` --
    /// `None` falls back to the family default, which for plain `qwen3` is off.
    #[test]
    fn an_unspecified_think_value_falls_back_to_the_family_default() {
        let mut p = Qwen3Parser::new(false, false);
        p.init(Vec::new(), None, None);
        let got = p.add("Direct answer", true).expect("add");
        assert_eq!(got.content, "Direct answer");
        assert!(got.thinking.is_empty());
    }

    /// Upstream `TestQwen3ParserToolCall`.
    #[test]
    fn a_tool_call_becomes_a_structured_call_and_no_content() {
        let mut p = instruct_parser();
        let got = p
            .add(
                r#"<tool_call>{"name":"get_weather","arguments":{"location":"San Francisco","unit":"celsius"}}</tool_call>"#,
                true,
            )
            .expect("add");
        assert!(got.content.is_empty());
        assert!(got.thinking.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(
            got.calls[0].function.arguments.get("location"),
            Some(&json!("San Francisco"))
        );
        assert_eq!(
            got.calls[0].function.arguments.get("unit"),
            Some(&json!("celsius"))
        );
    }

    /// Upstream `TestQwen3ParserThinkingWithToolCallBeforeThinkingClose` -- a
    /// `<tool_call>` before `</think>` ends thinking. Models do this.
    #[test]
    fn a_tool_call_before_the_thinking_close_tag_ends_thinking() {
        let mut p = thinking_parser();
        let got = p
            .add(
                r#"Let me think<tool_call>{"name":"get_weather","arguments":{"location":"San Francisco","unit":"celsius"}}</tool_call>"#,
                true,
            )
            .expect("add");
        assert!(got.content.is_empty());
        assert_eq!(got.thinking, "Let me think");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
    }

    /// Upstream `TestQwen3ParserThinkingWithSplitToolOpenTag`.
    #[test]
    fn a_tool_open_tag_split_across_chunks_holds_back_only_the_partial_tag() {
        let mut p = thinking_parser();
        let first = p.add("Let me think<tool_ca", false).expect("add");
        assert_eq!(first.thinking, "Let me think");
        assert!(first.content.is_empty());
        assert!(first.calls.is_empty());

        let got = p
            .add(r#"ll>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>"#, true)
            .expect("add");
        assert!(got.content.is_empty());
        assert!(
            got.thinking.is_empty(),
            "the held-back partial tag must not resurface as thinking"
        );
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
    }

    /// Upstream `TestQwen3ParserToolCallIndexing` -- the index is what a streamed
    /// delta uses to say which of several parallel calls it belongs to, so an
    /// off-by-one here silently merges two different tool calls.
    #[test]
    fn parallel_tool_calls_are_indexed_in_the_order_the_model_emitted_them() {
        let mut p = instruct_parser();
        let input = concat!(
            r#"<tool_call>{"name":"first","arguments":{"a":"1"}}</tool_call>"#,
            "\n",
            r#"<tool_call>{"name":"second","arguments":{"b":"2"}}</tool_call>"#,
            "\n",
            r#"<tool_call>{"name":"third","arguments":{"c":"3"}}</tool_call>"#,
        );
        let got = p.add(input, true).expect("add");
        let names: Vec<_> = got.calls.iter().map(|c| c.function.name.as_str()).collect();
        assert_eq!(names, ["first", "second", "third"]);
        for (i, c) in got.calls.iter().enumerate() {
            assert_eq!(c.function.index, i);
        }
    }

    /// Upstream `TestQwen3ParserToolCallIndexingStreaming` -- indices keep
    /// counting across chunk boundaries, including one that splits a call body.
    #[test]
    fn tool_call_indices_keep_counting_across_chunks() {
        let mut p = instruct_parser();
        let mut all = Vec::new();
        all.extend(
            p.add(
                r#"<tool_call>{"name":"first","arguments":{"a":"1"}}</tool_call><tool_call>{"name":"second","arguments":{"b":"2"}"#,
                false,
            )
            .expect("add")
            .calls,
        );
        all.extend(
            p.add(
                r#"}</tool_call><tool_call>{"name":"third","arguments":{"c":"3"}}</tool_call>"#,
                true,
            )
            .expect("add")
            .calls,
        );
        let got: Vec<_> = all
            .iter()
            .map(|c| (c.function.name.as_str(), c.function.index))
            .collect();
        assert_eq!(got, [("first", 0), ("second", 1), ("third", 2)]);
    }

    /// Upstream `TestQwen3ParserToolCallIndexResetOnInit`.
    #[test]
    fn init_resets_the_tool_call_index_for_the_next_turn() {
        let mut p = instruct_parser();
        p.add(r#"<tool_call>{"name":"first","arguments":{"a":"1"}}</tool_call>"#, true)
            .expect("add");
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(false)));
        let got = p
            .add(r#"<tool_call>{"name":"second","arguments":{"b":"2"}}</tool_call>"#, true)
            .expect("add");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.index, 0);
    }

    /// The strongest streaming test there is: one byte at a time. Nothing may be
    /// mis-split, and the totals must match the single-shot answer exactly.
    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = r#"reasoning here</think>Hello there<tool_call>{"name":"f","arguments":{"x":"1"}}</tool_call>bye"#;

        let mut whole = thinking_parser();
        let want = whole.add(input, true).expect("add");

        let mut streamed = thinking_parser();
        let mut got = Parsed::default();
        let bytes = input.len();
        for (i, ch) in input.char_indices() {
            let piece = &input[i..i + ch.len_utf8()];
            let done = i + ch.len_utf8() == bytes;
            let part = streamed.add(piece, done).expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
            got.calls.extend(part.calls);
        }

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.calls.len(), want.calls.len());
        assert_eq!(got.calls[0].function.name, "f");
    }

    /// A malformed tool-call body is a hard error, not a silent drop -- the
    /// caller has to know the model asked for something it could not honour.
    #[test]
    fn a_tool_call_that_is_not_json_is_an_error() {
        let mut p = instruct_parser();
        assert!(p.add("<tool_call>not json at all</tool_call>", true).is_err());
    }

    #[test]
    fn a_tool_call_with_no_name_is_an_error() {
        let mut p = instruct_parser();
        assert!(matches!(
            p.add(r#"<tool_call>{"arguments":{}}</tool_call>"#, true),
            Err(ParserError::EmptyFunctionName)
        ));
    }

    /// Trailing whitespace is held back, because it might turn out to be the
    /// padding in front of a tag -- and once emitted you cannot take it back.
    #[test]
    fn trailing_whitespace_is_withheld_until_something_follows_it() {
        let mut p = instruct_parser();
        assert_eq!(p.add("hello ", false).expect("add").content, "hello");
        assert_eq!(p.add("there", false).expect("add").content, " there");
    }
}
