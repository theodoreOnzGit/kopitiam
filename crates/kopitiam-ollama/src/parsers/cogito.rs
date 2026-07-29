//! Cogito response parser.
//!
//! **Upstream:** `model/parsers/cogito.go` (ollama, MIT, Copyright (c) Ollama).
//! Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b` (2026-07-28).
//!
//! ## Those tags are NOT ASCII -- look closely
//!
//! Cogito inherits DeepSeek's special-token family, and they are built from
//! **full-width** characters:
//!
//! * `\u{FF5C}` FULLWIDTH VERTICAL LINE -- the bars, **not** ASCII `|`;
//! * `\u{2581}` LOWER ONE EIGHTH BLOCK -- the "underscores" between words, which
//!   is SentencePiece's word-boundary marker, **not** ASCII `_`.
//!
//! Retype either one with an ASCII pipe or underscore and every tag silently
//! stops matching, the parser never leaves content mode, and the raw markup goes
//! to the user. The consts below are therefore written with explicit `\u{...}`
//! escapes -- unambiguous in any editor, and impossible to "tidy" into ASCII by
//! accident.
//!
//! So `<｜tool▁calls▁begin｜>` is **20 characters but 28 bytes**, and every index
//! in this file is a byte index. That is fine because the matching is all
//! `str::find`, which is byte-exact and never lands mid-character on a match.
//!
//! ## Tools switch thinking OFF -- and that is deliberate
//!
//! [`CogitoParser::set_initial_state`] has a branch you will not find in any
//! other family: **if any tools were offered, the stream starts in content
//! mode**, even with thinking explicitly enabled. Upstream's comment is
//! `"Note: for cogito, if there are tools, then we don't want to be thinking"`.
//! Kept as-is; it is a property of how these checkpoints were trained, not a
//! bug to iron out.
//!
//! Also note `thinkingEnabled := thinkValue != nil && thinkValue.Bool()` --
//! **`None` means OFF** here, same as lfm2, opposite of cohere.
//!
//! ## Tool-call payload shape
//!
//! (Outer fence is four backticks so the inner three-backtick JSON fence, which
//! is part of the wire format, survives.)
//!
//! ````text
//! <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
//! ```json
//! {"location":"Paris"}
//! ```<｜tool▁call▁end｜>
//! ````
//!
//! The fenced block is **required, exactly as spelled** -- see
//! [`JSON_FENCE_OPEN`] and [`JSON_FENCE_CLOSE`], and
//! [`parse_tool_call_content`] for how it is pulled apart.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, overlap, trailing_whitespace_len};

/// **Upstream:** `cogitoThinkingCloseTag`. Plain ASCII, unlike the tool tags --
/// note there is no *open* tag, because the prompt injects it.
const THINKING_CLOSE_TAG: &str = "</think>";

// The full-width tag family. `\u{FF5C}` is FULLWIDTH VERTICAL LINE and
// `\u{2581}` is LOWER ONE EIGHTH BLOCK -- see the module docs before touching
// any of these.
/// **Upstream:** `cogitoToolCallsBeginTag`.
const TOOL_CALLS_BEGIN_TAG: &str = "<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>";
/// **Upstream:** `cogitoToolCallsEndTag`.
const TOOL_CALLS_END_TAG: &str = "<\u{FF5C}tool\u{2581}calls\u{2581}end\u{FF5C}>";
/// **Upstream:** `cogitoToolCallBeginTag`. Singular -- wraps ONE call.
const TOOL_CALL_BEGIN_TAG: &str = "<\u{FF5C}tool\u{2581}call\u{2581}begin\u{FF5C}>";
/// **Upstream:** `cogitoToolCallEndTag`.
const TOOL_CALL_END_TAG: &str = "<\u{FF5C}tool\u{2581}call\u{2581}end\u{FF5C}>";
/// **Upstream:** `cogitoToolSepTag`. Separates `function` from the tool name.
const TOOL_SEP_TAG: &str = "<\u{FF5C}tool\u{2581}sep\u{FF5C}>";
/// **Upstream:** `cogitoToolOutputBeginTag`.
const TOOL_OUTPUT_BEGIN_TAG: &str = "<\u{FF5C}tool\u{2581}output\u{2581}begin\u{FF5C}>";
/// **Upstream:** `cogitoToolOutputEndTag`.
const TOOL_OUTPUT_END_TAG: &str = "<\u{FF5C}tool\u{2581}output\u{2581}end\u{FF5C}>";
/// **Upstream:** `cogitoToolOutputsBeginTag`.
const TOOL_OUTPUTS_BEGIN_TAG: &str = "<\u{FF5C}tool\u{2581}outputs\u{2581}begin\u{FF5C}>";
/// **Upstream:** `cogitoToolOutputsEndTag`.
const TOOL_OUTPUTS_END_TAG: &str = "<\u{FF5C}tool\u{2581}outputs\u{2581}end\u{FF5C}>";

/// Opens the JSON fence inside a tool call. **Upstream:** the literal
/// `"\n```json\n"` in `parseToolCallContent`.
const JSON_FENCE_OPEN: &str = "\n```json\n";
/// Closes it. **Upstream:** the literal `"\n```"`.
const JSON_FENCE_CLOSE: &str = "\n```";

/// **Upstream:** `CogitoParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    Thinking,
    Content,
    ToolCalls,
    /// Tool *results* echoed back into the stream. Everything here is
    /// **swallowed** -- see the note on [`CogitoParser::eat`].
    ToolOutput,
}

/// **Upstream:** `cogitoEvent`.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Thinking(String),
    Content(String),
    ToolCall(Box<ToolCall>),
}

/// **Upstream:** `CogitoParser`.
#[derive(Debug, Default)]
pub struct CogitoParser {
    state: State,
    buffer: String,
    call_index: usize,
}

impl CogitoParser {
    /// **Upstream:** `setInitialState`. Three ways to skip thinking, and the
    /// third one is unique to this family -- see the module docs.
    fn set_initial_state(
        &mut self,
        last_message: Option<&Message>,
        tools: &[Tool],
        think: Option<&ThinkValue>,
    ) {
        // `None` means OFF for cogito.
        let thinking_enabled = think.is_some_and(|t| t.enabled());
        if !thinking_enabled {
            self.state = State::Content;
            return;
        }

        let prefill = last_message.is_some_and(|m| m.role == "assistant");
        if prefill && last_message.is_some_and(|m| !m.content.is_empty()) {
            self.state = State::Content;
            return;
        }

        // Tools present -> no thinking. Upstream's own note.
        if !tools.is_empty() {
            self.state = State::Content;
            return;
        }

        self.state = State::Thinking;
    }

    /// **Upstream:** `(*CogitoParser).parseEvents`.
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

    /// **Upstream:** `(*CogitoParser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        if self.buffer.is_empty() {
            return (Vec::new(), false);
        }

        match self.state {
            State::Thinking => {
                if let Some(idx) = self.buffer.find(THINKING_CLOSE_TAG) {
                    let thinking = self.buffer[..idx].trim_end().to_string();
                    self.buffer = self.buffer[idx + THINKING_CLOSE_TAG.len()..]
                        .trim_start()
                        .to_string();
                    self.state = State::Content;
                    let mut events = Vec::new();
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    return (events, true);
                }

                // Hold back a partial `</think>` plus the whitespace in front of
                // it. With no partial tag this degrades to holding only the
                // trailing whitespace, which is still required -- `</think>`
                // may be the next thing to land, and it trims that whitespace.
                let overlap_len = overlap(&self.buffer, THINKING_CLOSE_TAG);
                let split = self.buffer.len() - overlap_len;
                let (before_partial_tag, _) = super::chop(&self.buffer, split);
                let ambiguous_start =
                    before_partial_tag.len() - trailing_whitespace_len(before_partial_tag);
                let (unambiguous, ambiguous) = super::chop(&self.buffer, ambiguous_start);
                let (unambiguous, ambiguous) = (unambiguous.to_string(), ambiguous.to_string());
                self.buffer = ambiguous;

                let mut events = Vec::new();
                if !unambiguous.is_empty() {
                    events.push(Event::Thinking(unambiguous));
                }
                (events, false)
            }

            State::Content => {
                if let Some(idx) = self.buffer.find(TOOL_CALLS_BEGIN_TAG) {
                    let content_before = self.buffer[..idx].trim_end().to_string();
                    self.buffer = self.buffer[idx + TOOL_CALLS_BEGIN_TAG.len()..].to_string();
                    self.state = State::ToolCalls;
                    let mut events = Vec::new();
                    if !content_before.is_empty() {
                        events.push(Event::Content(content_before));
                    }
                    return (events, true);
                }

                if let Some(idx) = self.buffer.find(TOOL_OUTPUTS_BEGIN_TAG) {
                    let content_before = self.buffer[..idx].trim_end().to_string();
                    self.buffer = self.buffer[idx + TOOL_OUTPUTS_BEGIN_TAG.len()..].to_string();
                    self.state = State::ToolOutput;
                    let mut events = Vec::new();
                    if !content_before.is_empty() {
                        events.push(Event::Content(content_before));
                    }
                    return (events, true);
                }

                // NOTE: no partial-tag buffering here -- upstream dumps the whole
                // buffer as content. A `<\u{FF5C}tool...` split across chunks can
                // therefore leak. Upstream's behaviour, kept: in practice the
                // tokeniser emits each of these as ONE special token, so they
                // cannot split.
                let content = std::mem::take(&mut self.buffer);
                (vec![Event::Content(content)], false)
            }

            State::ToolCalls => {
                if let Some(idx) = self.buffer.find(TOOL_CALL_BEGIN_TAG) {
                    let start = idx + TOOL_CALL_BEGIN_TAG.len();
                    if let Some(end) = self.buffer[start..].find(TOOL_CALL_END_TAG) {
                        let body = self.buffer[start..start + end].to_string();
                        if let Ok(call) = parse_tool_call_content(&body) {
                            self.buffer = self.buffer
                                [start + end + TOOL_CALL_END_TAG.len()..]
                                .trim_start()
                                .to_string();
                            return (vec![Event::ToolCall(Box::new(call))], true);
                        }
                        // Parse failed (upstream logs a warning). Fall through
                        // to the calls-end check rather than erroring out -- one
                        // bad call should not sink the rest of the block.
                    }
                }

                if let Some(idx) = self.buffer.find(TOOL_CALLS_END_TAG) {
                    self.buffer = self.buffer[idx + TOOL_CALLS_END_TAG.len()..]
                        .trim_start()
                        .to_string();
                    self.state = State::Content;
                    return (Vec::new(), true);
                }

                (Vec::new(), false)
            }

            // Tool OUTPUT is the tool's own result echoed back into the stream.
            // It is emitted NOWHERE -- not content, not thinking. The user
            // already has the result; showing it again as assistant text would
            // duplicate it. So this state only skips forward.
            State::ToolOutput => {
                if let Some(idx) = self.buffer.find(TOOL_OUTPUT_BEGIN_TAG) {
                    let start = idx + TOOL_OUTPUT_BEGIN_TAG.len();
                    if let Some(end) = self.buffer[start..].find(TOOL_OUTPUT_END_TAG) {
                        self.buffer = self.buffer[start + end + TOOL_OUTPUT_END_TAG.len()..]
                            .trim_start()
                            .to_string();
                        return (Vec::new(), true);
                    }
                }

                if let Some(idx) = self.buffer.find(TOOL_OUTPUTS_END_TAG) {
                    self.buffer = self.buffer[idx + TOOL_OUTPUTS_END_TAG.len()..]
                        .trim_start()
                        .to_string();
                    self.state = State::Content;
                    return (Vec::new(), true);
                }

                (Vec::new(), false)
            }
        }
    }
}

impl Parser for CogitoParser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.call_index = 0;
        // Upstream does not clear the buffer in Init (it always builds a fresh
        // parser). Cleared here so a re-`init` cannot inherit half a tag.
        // Stated divergence; no behaviour change for the documented lifecycle.
        self.buffer.clear();
        self.set_initial_state(last_message, &tools, think);
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);

        let mut out = Parsed::default();
        for event in self.parse_events() {
            match event {
                Event::ToolCall(tc) => out.calls.push(*tc),
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
            TOOL_OUTPUTS_BEGIN_TAG,
            TOOL_OUTPUTS_END_TAG,
        ]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        true
    }
}

/// Pull one tool call out of a `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` body.
///
/// **Upstream:** `parseToolCallContent`. Expected shape, and every piece is
/// mandatory (the outer fence below is four backticks so the inner three-backtick
/// JSON fence -- which is part of the data -- survives):
///
/// ````text
/// function<｜tool▁sep｜>get_weather
/// ```json
/// {"location":"Paris"}
/// ```
/// ````
///
/// Steps: split once on the separator (the `function` keyword in front of it is
/// discarded); find [`JSON_FENCE_OPEN`]; the tool name is everything before it,
/// trimmed; the arguments are the fenced JSON up to the next
/// [`JSON_FENCE_CLOSE`].
///
/// Any missing piece is a plain "invalid format" error, and the caller
/// *swallows* it -- a malformed call is dropped, never guessed at.
fn parse_tool_call_content(content: &str) -> Result<ToolCall, ParserError> {
    let invalid = || ParserError::MalformedToolCall("invalid format".into());

    let (_function_kw, name_and_args) = content.split_once(TOOL_SEP_TAG).ok_or_else(invalid)?;

    let json_start = name_and_args.find(JSON_FENCE_OPEN).ok_or_else(invalid)?;
    let tool_name = name_and_args[..json_start].trim().to_string();
    let json_content = &name_and_args[json_start + JSON_FENCE_OPEN.len()..];

    let json_end = json_content.find(JSON_FENCE_CLOSE).ok_or_else(invalid)?;
    let args_json = &json_content[..json_end];

    let arguments: ToolCallArguments = serde_json::from_str(args_json)?;

    Ok(ToolCall {
        function: ToolCallFunction {
            name: tool_name,
            arguments,
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolFunction;
    use serde_json::json;

    fn tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn cogito(think: bool) -> CogitoParser {
        let mut p = CogitoParser::default();
        p.init(Vec::new(), None, Some(&ThinkValue::Bool(think)));
        p
    }

    /// One call, spelled out with the real full-width tags.
    fn one_call(name: &str, args_json: &str) -> String {
        format!(
            "{TOOL_CALLS_BEGIN_TAG}{TOOL_CALL_BEGIN_TAG}function{TOOL_SEP_TAG}{name}\n```json\n{args_json}\n```{TOOL_CALL_END_TAG}{TOOL_CALLS_END_TAG}"
        )
    }

    /// The tags really are full-width. If someone "tidies" them into ASCII this
    /// is the test that screams.
    #[test]
    fn the_special_tags_are_full_width_not_ascii() {
        assert!(TOOL_CALLS_BEGIN_TAG.contains('\u{FF5C}'), "needs FULLWIDTH VERTICAL LINE");
        assert!(TOOL_CALLS_BEGIN_TAG.contains('\u{2581}'), "needs LOWER ONE EIGHTH BLOCK");
        assert!(!TOOL_CALLS_BEGIN_TAG.contains('|'), "must NOT contain ASCII pipe");
        assert!(!TOOL_CALLS_BEGIN_TAG.contains('_'), "must NOT contain ASCII underscore");
        // 20 characters but 28 bytes -- the gap is the whole point. The three
        // full-width characters cost 3 bytes each instead of 1.
        assert_eq!(TOOL_CALLS_BEGIN_TAG.chars().count(), 20);
        assert_eq!(TOOL_CALLS_BEGIN_TAG.len(), 28);
    }

    /// Upstream `TestCogitoParser`, the plain cases.
    #[test]
    fn cogito_splits_thinking_from_content() {
        let mut p = cogito(false);
        let got = p.add("This is a simple response.", true).expect("add");
        assert_eq!(got.content, "This is a simple response.");
        assert!(got.thinking.is_empty());

        let mut p = cogito(true);
        let got = p
            .add("This is thinking content.</think>This is response content.", true)
            .expect("add");
        assert_eq!(got.thinking, "This is thinking content.");
        assert_eq!(got.content, "This is response content.");
    }

    #[test]
    fn a_fenced_json_tool_call_is_extracted() {
        let mut p = cogito(false);
        let got = p
            .add(&one_call("get_weather", r#"{"location":"Paris"}"#), true)
            .expect("add");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(
            got.calls[0].function.arguments.get("location"),
            Some(&json!("Paris"))
        );
    }

    #[test]
    fn thinking_then_a_tool_call_both_come_out() {
        let mut p = cogito(true);
        let input = format!(
            "I need to check the weather.</think>{}",
            one_call("get_weather", r#"{"location":"Paris"}"#)
        );
        let got = p.add(&input, true).expect("add");
        assert_eq!(got.thinking, "I need to check the weather.");
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
    }

    /// Upstream `multiple_tool_calls` -- two calls in one block, indexed.
    #[test]
    fn two_calls_in_one_block_are_both_found_and_indexed() {
        let input = format!(
            "{TOOL_CALLS_BEGIN_TAG}\
             {TOOL_CALL_BEGIN_TAG}function{TOOL_SEP_TAG}get_weather\n```json\n{{\"location\":\"Paris\"}}\n```{TOOL_CALL_END_TAG}\n\
             {TOOL_CALL_BEGIN_TAG}function{TOOL_SEP_TAG}get_weather\n```json\n{{\"location\":\"London\"}}\n```{TOOL_CALL_END_TAG}\
             {TOOL_CALLS_END_TAG}"
        );
        let mut p = cogito(false);
        let got = p.add(&input, true).expect("add");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.index, 1);
        assert_eq!(
            got.calls[1].function.arguments.get("location"),
            Some(&json!("London"))
        );
    }

    /// Upstream `complex_tool_arguments`.
    ///
    /// **Representation note:** upstream's fixture expects `count: 42.0`,
    /// because Go's `json.Unmarshal` into `any` makes every number a `float64`.
    /// We deserialise into `serde_json::Value`, which keeps `42` as an integer.
    /// Same number, different Go/Rust convention -- and ours round-trips back to
    /// `42` rather than `42.0`, which is closer to what the model wrote.
    #[test]
    fn complex_json_arguments_survive_intact_and_in_order() {
        let mut p = cogito(false);
        let got = p
            .add(
                &one_call(
                    "process_data",
                    r#"{"items":["item1","item2"],"config":{"enabled":true,"threshold":0.95},"count":42}"#,
                ),
                true,
            )
            .expect("add");
        let a = &got.calls[0].function.arguments;
        assert_eq!(a.get("items"), Some(&json!(["item1", "item2"])));
        assert_eq!(a.get("config"), Some(&json!({"enabled": true, "threshold": 0.95})));
        assert_eq!(a.get("count"), Some(&json!(42)));
        let keys: Vec<&str> = a.0.keys().map(String::as_str).collect();
        assert_eq!(keys, ["items", "config", "count"]);
    }

    /// Upstream `tool_output_parsing`. A tool-output block is swallowed whole --
    /// it must reach NEITHER content nor thinking.
    #[test]
    fn a_tool_output_block_is_swallowed_and_never_shown_to_the_user() {
        let mut p = cogito(false);
        let input = format!(
            "{TOOL_OUTPUTS_BEGIN_TAG}{TOOL_OUTPUT_BEGIN_TAG}{{\"temperature\": 22}}{TOOL_OUTPUT_END_TAG}{TOOL_OUTPUTS_END_TAG}"
        );
        let got = p.add(&input, true).expect("add");
        assert!(got.content.is_empty(), "tool output must not become content");
        assert!(got.thinking.is_empty());
        assert!(got.calls.is_empty());
    }

    /// **The branch unique to cogito:** offering tools turns thinking off, even
    /// with `think = true`.
    #[test]
    fn offering_tools_switches_thinking_off_for_this_family() {
        let mut p = CogitoParser::default();
        p.init(vec![tool("get_weather")], None, Some(&ThinkValue::Bool(true)));
        let got = p.add("this looks like thinking</think>and this content", true).expect("add");
        assert!(got.thinking.is_empty(), "tools present -> no thinking mode");
        assert_eq!(got.content, "this looks like thinking</think>and this content");
    }

    /// `None` means thinking OFF for cogito.
    #[test]
    fn an_unspecified_think_value_means_thinking_off_for_this_family() {
        let mut p = CogitoParser::default();
        p.init(Vec::new(), None, None);
        let got = p.add("reasoning</think>answer", true).expect("add");
        assert!(got.thinking.is_empty());
        assert_eq!(got.content, "reasoning</think>answer");
    }

    /// Upstream's edge cases: only the FIRST `</think>` closes thinking.
    #[test]
    fn only_the_first_think_close_tag_closes_thinking() {
        let mut p = cogito(true);
        let got = p
            .add("I'm thinking <think>nested</think> more thinking</think>Final content.", true)
            .expect("add");
        assert_eq!(got.thinking, "I'm thinking <think>nested");
        assert_eq!(got.content, "more thinking</think>Final content.");

        let mut p = cogito(true);
        let got = p
            .add("First thinking</think>Content</think>More content.", true)
            .expect("add");
        assert_eq!(got.thinking, "First thinking");
        assert_eq!(got.content, "Content</think>More content.");
    }

    /// An assistant prefill with content skips thinking.
    #[test]
    fn an_assistant_content_prefill_starts_the_stream_in_content_mode() {
        let mut p = CogitoParser::default();
        let last = Message::new("assistant", "existing");
        p.init(Vec::new(), Some(&last), Some(&ThinkValue::Bool(true)));
        let got = p
            .add("Content with </think> tags should be treated as content.", true)
            .expect("add");
        assert!(got.thinking.is_empty());
        assert_eq!(got.content, "Content with </think> tags should be treated as content.");
    }

    #[test]
    fn a_thinking_close_tag_split_across_chunks_never_leaks() {
        let mut p = cogito(true);
        let a = p.add("thought</thi", false).expect("add");
        assert_eq!(a.thinking, "thought");
        assert!(a.content.is_empty());
        let b = p.add("nk>visible", true).expect("add");
        assert!(b.thinking.is_empty());
        assert_eq!(b.content, "visible");
    }

    /// Thinking survives character-at-a-time feeding -- that is the state that
    /// actually buffers.
    #[test]
    fn thinking_fed_one_character_at_a_time_gives_the_same_answer() {
        let input = "let me reckon</think>The answer lah.";

        let mut whole = cogito(true);
        let want = whole.add(input, true).expect("add");

        let mut p = cogito(true);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
        }

        assert_eq!(got.thinking, want.thinking);
        assert_eq!(got.content, want.content);
        assert_eq!(got.thinking, "let me reckon");
        assert_eq!(got.content, "The answer lah.");
    }

    /// **A REAL UPSTREAM LIMITATION, pinned so nobody "fixes" it by accident.**
    ///
    /// The content state does **no partial-tag buffering** -- it dumps its whole
    /// buffer as content every pass. So a `<｜tool▁calls▁begin｜>` split across
    /// chunks never accumulates, and the tool call is lost entirely: the raw
    /// markup goes to the user as content instead.
    ///
    /// Same shape as lfm2's limitation, and safe for the same reason: the
    /// tokeniser emits each of these full-width tags as ONE special token, so it
    /// cannot split in practice. Upstream has no fixture for a split one either.
    ///
    /// Do not "fix" this unilaterally -- it would make this port and ollama
    /// disagree on the same byte stream. Take it upstream first.
    #[test]
    fn the_tool_calls_begin_tag_is_not_buffered_across_chunks() {
        let input = format!("Checking now.{}", one_call("get_weather", r#"{"location":"SG"}"#));

        // One chunk: the call is found.
        let mut whole = cogito(false);
        let want = whole.add(&input, true).expect("add");
        assert_eq!(want.content, "Checking now.");
        assert_eq!(want.calls.len(), 1);

        // One character at a time: the tag leaks and NO call is found.
        let mut p = cogito(false);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.calls.extend(part.calls);
        }
        assert_eq!(got.content, input, "the whole thing leaks as content");
        assert!(got.calls.is_empty(), "and no tool call is recognised");
    }

    #[test]
    fn a_malformed_tool_call_body_is_dropped_rather_than_failing_the_generation() {
        // No `\n```json\n` fence at all.
        let input = format!(
            "{TOOL_CALLS_BEGIN_TAG}{TOOL_CALL_BEGIN_TAG}function{TOOL_SEP_TAG}f no fence here{TOOL_CALL_END_TAG}{TOOL_CALLS_END_TAG}tail"
        );
        let mut p = cogito(false);
        let got = p.add(&input, true).expect("add");
        assert!(got.calls.is_empty());
        assert_eq!(got.content, "tail");
    }

    #[test]
    fn parse_tool_call_content_rejects_every_missing_piece() {
        // No separator.
        assert!(parse_tool_call_content("function get_weather").is_err());
        // Separator but no JSON fence.
        assert!(parse_tool_call_content(&format!("function{TOOL_SEP_TAG}get_weather")).is_err());
        // Fence opened but never closed.
        assert!(
            parse_tool_call_content(&format!("function{TOOL_SEP_TAG}f\n```json\n{{}}")).is_err()
        );
        // Fence closed but the body is not JSON.
        assert!(
            parse_tool_call_content(&format!("function{TOOL_SEP_TAG}f\n```json\nnope\n```"))
                .is_err()
        );
    }

    #[test]
    fn cogito_advertises_all_ten_of_its_special_tokens() {
        let p = cogito(true);
        let toks = p.preserved_tokens();
        assert_eq!(toks.len(), 10);
        for t in [
            THINKING_CLOSE_TAG,
            TOOL_CALLS_BEGIN_TAG,
            TOOL_CALL_BEGIN_TAG,
            TOOL_SEP_TAG,
            TOOL_OUTPUTS_END_TAG,
        ] {
            assert!(toks.contains(&t), "missing {t}");
        }
        assert!(p.has_tool_support());
        assert!(p.has_thinking_support());
    }
}
