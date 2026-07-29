//! DeepSeek-V3 response parser.
//!
//! **Upstream:** `model/parsers/deepseek3.go` (ollama, MIT).
//!
//! ## The tags are not ASCII, and that is not a typo
//!
//! DeepSeek frames everything with **fullwidth** and **block-element** code
//! points: `U+FF5C` FULLWIDTH VERTICAL LINE for the bars, and `U+2581` LOWER ONE
//! EIGHTH BLOCK for the separators inside a word. So the begin tag is
//! `<`, `U+FF5C`, `tool`, `U+2581`, `calls`, `U+2581`, `begin`, `U+FF5C`, `>` --
//! **not** `<|tool_calls_begin|>`. Retyping one of these by hand with an ASCII
//! pipe or underscore silently kills every tool call for the family, and nothing
//! will look wrong in the diff. There is a test below that pins the code points.
//!
//! ## Four states, two of which are about tool *output*
//!
//! Besides thinking and content, DeepSeek has a nested block for tool calls
//! (`calls-begin` ... one or more `call-begin name SEP {json} call-end` ...
//! `calls-end`) **and** a block for tool *output* being echoed back, which is
//! unwrapped into plain content.
//!
//! ## Known upstream weakness, ported faithfully
//!
//! In content mode this parser does **no** ambiguity buffering: it flushes the
//! whole buffer every call. So if a `calls-begin` tag is split across two chunks,
//! the first half leaks into content and the tool call is missed. Upstream's own
//! streaming tests only ever split *inside* `</think>` (which is handled) or on
//! whole-tag boundaries, so the gap is real but unexercised. We keep the
//! behaviour rather than silently improve on the oracle -- but the gap is written
//! down here so whoever hits it knows it is upstream's, not ours, and can fix
//! both.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, chop, emit_unambiguous, overlap};

/// **Upstream:** the `deepseek*Tag` consts in `deepseek3.go`. See the module docs
/// -- the bars are `U+FF5C`, the word separators are `U+2581`.
const THINKING_CLOSE_TAG: &str = "</think>";
const TOOL_CALLS_BEGIN_TAG: &str = "<｜tool▁calls▁begin｜>";
const TOOL_CALLS_END_TAG: &str = "<｜tool▁calls▁end｜>";
const TOOL_CALL_BEGIN_TAG: &str = "<｜tool▁call▁begin｜>";
const TOOL_CALL_END_TAG: &str = "<｜tool▁call▁end｜>";
const TOOL_SEP_TAG: &str = "<｜tool▁sep｜>";
const TOOL_OUTPUT_BEGIN_TAG: &str = "<｜tool▁output▁begin｜>";
const TOOL_OUTPUT_END_TAG: &str = "<｜tool▁output▁end｜>";

/// **Upstream:** `DeepSeek3ParserState`.
///
/// The shared `Collecting` prefix is upstream's (`DeepSeekCollectingThinking`,
/// ...). Kept so a reader can grep the Go and land on the same variant; clippy's
/// tidiness suggestion loses that, which is a worse trade for a port.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    CollectingThinking,
    CollectingContent,
    CollectingToolCalls,
    CollectingToolOutput,
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Thinking(String),
    Content(String),
    ToolCall(Box<ToolCall>),
}

/// **Upstream:** `DeepSeek3Parser`.
#[derive(Debug, Default)]
pub struct DeepSeek3Parser {
    state: State,
    buffer: String,
    call_index: usize,
    has_thinking_support: bool,
}

impl DeepSeek3Parser {
    /// `ParserForName("deepseek3")` is `new(true)`.
    pub fn new(has_thinking_support: bool) -> Self {
        Self {
            has_thinking_support,
            ..Default::default()
        }
    }

    /// **Upstream:** `setInitialState`.
    ///
    /// Note the `None` case: unlike qwen3.5, DeepSeek treats "the caller never
    /// said" as **off**, because the test is
    /// `thinkValue != nil && thinkValue.Bool()`. Model capability AND request
    /// preference must both say yes.
    fn set_initial_state(&mut self, last_message: Option<&Message>, think: Option<&ThinkValue>) {
        let thinking_enabled = self.has_thinking_support && think.is_some_and(|t| t.enabled());
        if !thinking_enabled {
            self.state = State::CollectingContent;
            return;
        }
        let prefill = last_message.is_some_and(|m| m.role == "assistant");
        if prefill && last_message.is_some_and(|m| !m.content.is_empty()) {
            self.state = State::CollectingContent;
            return;
        }
        self.state = State::CollectingThinking;
    }

    /// **Upstream:** `(*DeepSeek3Parser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        let mut events = Vec::new();
        if self.buffer.is_empty() {
            return (events, false);
        }
        let buf = self.buffer.clone();

        match self.state {
            State::CollectingThinking => {
                if let Some(idx) = buf.find(THINKING_CLOSE_TAG) {
                    let (thinking, rest) = chop(&buf, idx);
                    let thinking = thinking.trim_end().to_string();
                    let remaining = rest[THINKING_CLOSE_TAG.len()..].trim_start().to_string();
                    self.buffer = remaining;
                    self.state = State::CollectingContent;
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    return (events, true);
                }
                // Only `</think>` is guarded here -- see the module note about the
                // gap in content mode.
                let overlap_len = overlap(&buf, THINKING_CLOSE_TAG);
                let unambiguous = emit_unambiguous(&mut self.buffer, overlap_len);
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::CollectingContent => {
                if let Some(idx) = buf.find(TOOL_CALLS_BEGIN_TAG) {
                    let (before, rest) = chop(&buf, idx);
                    let content_before = before.trim_end().to_string();
                    self.buffer = rest[TOOL_CALLS_BEGIN_TAG.len()..].to_string();
                    self.state = State::CollectingToolCalls;
                    if !content_before.is_empty() {
                        events.push(Event::Content(content_before));
                    }
                    return (events, true);
                }
                if let Some(idx) = buf.find(TOOL_OUTPUT_BEGIN_TAG) {
                    // Whitespace is NOT trimmed here, unlike the calls case --
                    // tool output is spliced back into the middle of a sentence,
                    // so the space before it is real content.
                    let (before, rest) = chop(&buf, idx);
                    let content_before = before.to_string();
                    self.buffer = rest[TOOL_OUTPUT_BEGIN_TAG.len()..].to_string();
                    self.state = State::CollectingToolOutput;
                    if !content_before.is_empty() {
                        events.push(Event::Content(content_before));
                    }
                    return (events, true);
                }
                // Flush everything. No withholding -- see the module note.
                self.buffer.clear();
                if !buf.is_empty() {
                    events.push(Event::Content(buf));
                }
                (events, false)
            }

            State::CollectingToolCalls => {
                if let Some(idx) = buf.find(TOOL_CALL_BEGIN_TAG) {
                    let start = idx + TOOL_CALL_BEGIN_TAG.len();
                    if let Some(end_rel) = buf[start..].find(TOOL_CALL_END_TAG) {
                        let body = &buf[start..start + end_rel];
                        // A body that will not parse is NOT an error -- upstream
                        // logs a warning and falls through to look for the
                        // calls-end tag, so one malformed call cannot wedge the
                        // whole stream.
                        if let Ok(call) = parse_tool_call_content(body) {
                            let after = start + end_rel + TOOL_CALL_END_TAG.len();
                            self.buffer = buf[after..].trim_start().to_string();
                            events.push(Event::ToolCall(Box::new(call)));
                            return (events, true);
                        }
                    }
                }

                if let Some(idx) = buf.find(TOOL_CALLS_END_TAG) {
                    self.buffer = buf[idx + TOOL_CALLS_END_TAG.len()..].trim_start().to_string();
                    self.state = State::CollectingContent;
                    return (events, true);
                }

                (events, false)
            }

            State::CollectingToolOutput => {
                let Some(idx) = buf.find(TOOL_OUTPUT_END_TAG) else {
                    return (events, false);
                };
                let (output, rest) = chop(&buf, idx);
                let output = output.to_string();
                // Again no trimming -- the space after the closing tag belongs to
                // the sentence the output was spliced into.
                self.buffer = rest[TOOL_OUTPUT_END_TAG.len()..].to_string();
                self.state = State::CollectingContent;
                if !output.is_empty() {
                    events.push(Event::Content(output));
                }
                (events, true)
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

impl Parser for DeepSeek3Parser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        // Faithful wart: upstream does not clear the buffer here either.
        self.call_index = 0;
        self.set_initial_state(last_message, think);
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);
        let events = self.parse_events();

        let mut out = Parsed::default();
        for event in events {
            match event {
                Event::ToolCall(call) => out.calls.push(*call),
                Event::Thinking(t) => out.thinking.push_str(&t),
                Event::Content(c) => out.content.push_str(&c),
            }
        }

        for call in &mut out.calls {
            call.function.index = self.call_index;
            self.call_index += 1;
        }

        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![
            THINKING_CLOSE_TAG,
            TOOL_CALLS_BEGIN_TAG,
            TOOL_CALLS_END_TAG,
            TOOL_CALL_BEGIN_TAG,
            TOOL_CALL_END_TAG,
            TOOL_SEP_TAG,
            TOOL_OUTPUT_BEGIN_TAG,
            TOOL_OUTPUT_END_TAG,
        ]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        self.has_thinking_support
    }
}

/// Parse `tool_name<SEP>{args}`. **Upstream:** `parseToolCallContent`.
///
/// Both halves are `TrimSpace`d before use, so the newlines DeepSeek puts around
/// the separator do not end up in the tool name or break the JSON parse.
fn parse_tool_call_content(content: &str) -> Result<ToolCall, ParserError> {
    let Some(idx) = content.find(TOOL_SEP_TAG) else {
        return Err(ParserError::MalformedToolCall("invalid format".into()));
    };
    let name = content[..idx].trim().to_string();
    let args_json = content[idx + TOOL_SEP_TAG.len()..].trim();
    let arguments: ToolCallArguments = serde_json::from_str(args_json)?;

    Ok(ToolCall {
        id: String::new(),
        function: ToolCallFunction {
            index: 0,
            name,
            arguments,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parser(thinking: bool) -> DeepSeek3Parser {
        let mut p = DeepSeek3Parser::new(thinking);
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(thinking)));
        p
    }

    /// Pins the exact code points. If somebody retypes a tag with an ASCII pipe
    /// or underscore, every tool call for this family dies silently -- this test
    /// is the only thing that would notice.
    #[test]
    fn the_deepseek_tags_use_fullwidth_bars_and_block_separators() {
        assert_eq!(
            TOOL_CALLS_BEGIN_TAG.chars().collect::<Vec<_>>(),
            vec![
                '<', '\u{FF5C}', 't', 'o', 'o', 'l', '\u{2581}', 'c', 'a', 'l', 'l', 's',
                '\u{2581}', 'b', 'e', 'g', 'i', 'n', '\u{FF5C}', '>'
            ]
        );
        assert!(TOOL_SEP_TAG.contains('\u{FF5C}'));
        assert!(TOOL_SEP_TAG.contains('\u{2581}'));
        // ...and none of them is the ASCII spelling.
        assert!(!TOOL_CALLS_BEGIN_TAG.contains('|'));
        assert!(!TOOL_CALLS_BEGIN_TAG.contains('_'));
    }

    /// Upstream `TestDeepSeekParser_Streaming`, "streaming_simple_content".
    #[test]
    fn plain_content_streams_straight_through() {
        let mut p = parser(false);
        let mut got = String::new();
        for c in ["Hello, ", "how are ", "you?"] {
            got.push_str(&p.add(c, false).expect("add").content);
        }
        assert_eq!(got, "Hello, how are you?");
    }

    /// Upstream "streaming_thinking".
    #[test]
    fn thinking_ends_at_the_close_tag_and_the_rest_is_content() {
        let mut p = parser(true);
        let (mut content, mut thinking) = (String::new(), String::new());
        for c in [
            "I need to ",
            "think about this",
            "...</think>",
            "The answer is 42.",
        ] {
            let r = p.add(c, false).expect("add");
            content.push_str(&r.content);
            thinking.push_str(&r.thinking);
        }
        assert_eq!(thinking, "I need to think about this...");
        assert_eq!(content, "The answer is 42.");
    }

    /// Upstream "streaming_thinking_with_partial_tag" and
    /// "streaming_thinking_with_split_end_tag" -- `</think>` broken in two places.
    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        for chunks in [
            ["Thinking about this", "...</", "think>", "Done thinking."],
            ["Thinking content", "</th", "ink>", "Regular content"],
        ] {
            let mut p = parser(true);
            let (mut content, mut thinking) = (String::new(), String::new());
            for c in chunks {
                let r = p.add(c, false).expect("add");
                content.push_str(&r.content);
                thinking.push_str(&r.thinking);
            }
            assert!(!thinking.contains("</"), "a half tag leaked: {thinking:?}");
            assert!(!content.contains("think>"), "a half tag leaked: {content:?}");
        }
    }

    /// Upstream "streaming_tool_call".
    #[test]
    fn a_tool_call_arriving_in_pieces_is_assembled_and_parsed() {
        let mut p = parser(false);
        let mut content = String::new();
        let mut calls = Vec::new();
        for c in [
            "I'll check weather.",
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>get_weather",
            "<｜tool▁sep｜>{\"location\":\"Paris\"}",
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ] {
            let r = p.add(c, false).expect("add");
            content.push_str(&r.content);
            calls.extend(r.calls);
        }
        assert_eq!(content, "I'll check weather.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments.get("location"),
            Some(&json!("Paris"))
        );
    }

    /// Upstream "streaming_tool_call_with_split_json" -- the JSON body itself is
    /// split, which is fine because nothing is parsed until `call-end` arrives.
    #[test]
    fn a_tool_call_whose_json_body_is_split_still_parses() {
        let mut p = parser(false);
        let mut calls = Vec::new();
        for c in [
            "Processing.",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>calc<｜tool▁sep｜>{\"x\":",
            "42,\"y\":",
            "24}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ] {
            calls.extend(p.add(c, false).expect("add").calls);
        }
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].function.arguments.get("x"), Some(&json!(42)));
        assert_eq!(calls[0].function.arguments.get("y"), Some(&json!(24)));
    }

    /// Upstream "streaming_tool_output" -- tool output is unwrapped into content,
    /// and the whitespace either side of it is deliberately preserved.
    #[test]
    fn tool_output_is_unwrapped_into_content_with_its_spacing_intact() {
        let mut p = parser(false);
        let mut content = String::new();
        for c in [
            "Weather info: ",
            "<｜tool▁output▁begin｜>",
            "25\u{00b0}C, Sunny",
            "<｜tool▁output▁end｜>",
            " Enjoy!",
        ] {
            content.push_str(&p.add(c, false).expect("add").content);
        }
        assert_eq!(content, "Weather info: 25\u{00b0}C, Sunny Enjoy!");
    }

    /// Upstream "streaming_multiple_tool_outputs".
    #[test]
    fn several_tool_outputs_in_one_reply_are_all_unwrapped() {
        let mut p = parser(false);
        let mut content = String::new();
        for c in [
            "Results: ",
            "<｜tool▁output▁begin｜>",
            "Paris: 22\u{00b0}C",
            "<｜tool▁output▁end｜>",
            " and ",
            "<｜tool▁output▁begin｜>",
            "London: 18\u{00b0}C",
            "<｜tool▁output▁end｜>",
        ] {
            content.push_str(&p.add(c, false).expect("add").content);
        }
        assert_eq!(content, "Results: Paris: 22\u{00b0}C and London: 18\u{00b0}C");
    }

    /// Upstream "streaming_with_split_tags". Note the DOUBLE space in the
    /// expected content: the trailing space of "Content before " is emitted
    /// verbatim (content mode does no withholding), and " after" brings its own.
    #[test]
    fn content_around_a_tool_call_block_keeps_its_own_spacing() {
        let mut p = parser(false);
        let mut content = String::new();
        let mut calls = Vec::new();
        for c in [
            "Content before ",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test",
            "<｜tool▁sep｜>{}",
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            " after",
        ] {
            let r = p.add(c, false).expect("add");
            content.push_str(&r.content);
            calls.extend(r.calls);
        }
        assert_eq!(content, "Content before  after");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "test");
        assert!(calls[0].function.arguments.is_empty());
    }

    /// Upstream "streaming_unicode_content".
    #[test]
    fn unicode_content_streams_unharmed() {
        let mut p = parser(false);
        let mut content = String::new();
        for c in ["مرحبا ", "بالعالم! ", "你好", "世界!"] {
            content.push_str(&p.add(c, false).expect("add").content);
        }
        assert_eq!(content, "مرحبا بالعالم! 你好世界!");
    }

    /// Upstream `TestDeepSeek3Parser_parseToolCallContent`: a body with no
    /// separator is invalid, and both halves get trimmed.
    #[test]
    fn a_tool_call_body_needs_the_separator_and_gets_trimmed() {
        assert!(parse_tool_call_content("no separator here").is_err());
        let call = parse_tool_call_content("  get_weather  <｜tool▁sep｜>  {\"a\":1}  ")
            .expect("parse");
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments.get("a"), Some(&json!(1)));
    }

    /// A malformed call must not wedge the stream: upstream warns and keeps
    /// looking for the calls-end tag.
    #[test]
    fn a_malformed_tool_call_is_skipped_rather_than_wedging_the_stream() {
        let mut p = parser(false);
        let got = p
            .add(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>bad json<｜tool▁sep｜>{oops<｜tool▁call▁end｜><｜tool▁calls▁end｜>after",
                true,
            )
            .expect("add");
        assert!(got.calls.is_empty());
        assert_eq!(got.content, "after");
    }

    /// Thinking must be BOTH supported by the model and asked for by the caller.
    /// `None` counts as not asked for -- unlike qwen3.5, where `None` means on.
    #[test]
    fn thinking_needs_both_model_support_and_an_explicit_request() {
        let mut unsupported = DeepSeek3Parser::new(false);
        unsupported.init(Vec::new(), None, Some(&ThinkValue::Bool(true)));
        assert_eq!(
            unsupported.add("abc</think>def", true).expect("add").content,
            "abc</think>def"
        );

        let mut unasked = DeepSeek3Parser::new(true);
        unasked.init(Vec::new(), None, None);
        assert_eq!(
            unasked.add("abc</think>def", true).expect("add").content,
            "abc</think>def"
        );
    }

    #[test]
    fn a_non_empty_assistant_prefill_starts_in_content_mode() {
        let mut p = DeepSeek3Parser::new(true);
        let last = Message::new("assistant", "So far:");
        p.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = p.add("carrying on", true).expect("add");
        assert_eq!(got.content, "carrying on");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn parallel_tool_calls_are_indexed_in_order() {
        let mut p = parser(false);
        let got = p
            .add(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>a<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁call▁begin｜>b<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                true,
            )
            .expect("add");
        let names: Vec<_> = got
            .calls
            .iter()
            .map(|c| (c.function.name.as_str(), c.function.index))
            .collect();
        assert_eq!(names, [("a", 0), ("b", 1)]);
    }
}
