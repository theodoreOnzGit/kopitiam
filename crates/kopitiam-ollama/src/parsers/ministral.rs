//! Ministral (Mistral) response parser.
//!
//! **Upstream:** `model/parsers/ministral.go` (ollama, MIT, Copyright (c)
//! Ollama). Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b`
//! (2026-07-28).
//!
//! ## Bracket tags, not angle brackets
//!
//! Mistral frames everything with `[TOOL_CALLS]`, `[THINK]`, `[/THINK]` and
//! `[ARGS]`. A tool call is **three tags with no JSON wrapper around the name**:
//!
//! ```text
//! [TOOL_CALLS]get_weather[ARGS]{"city":"Paris"}
//! ```
//!
//! The name is simply everything between `[TOOL_CALLS]` and `[ARGS]`, and the
//! arguments are the JSON object that follows -- terminated by
//! [`find_json_end`], because nothing marks the end of the call. That is why
//! this family needs a brace counter and most others do not.
//!
//! ## An unknown tool name is a HARD ERROR here
//!
//! [`MinistralParser::add`] looks the name up in the tools the caller offered
//! and **returns an error** if it is not there -- see [`tool_by_name`]. Almost
//! every other family in this port either drops a bad call quietly or lets it
//! through. Ministral does not, and the difference is deliberate upstream: the
//! name is unquoted free text between two tags, so a hallucinated one is
//! indistinguishable from a typo, and silently inventing a tool call for a tool
//! that does not exist is worse than failing loudly.
//!
//! Note what that means for the caller: `add` can return `Err` **after** having
//! already produced content and thinking in earlier chunks. The error is the
//! last word, not a rollback.
//!
//! ## `[THINK]` works even when thinking support is off
//!
//! `parser_for_name("ministral")` builds this with `has_thinking_support =
//! false`, so the stream starts in content mode. But the content state still
//! watches for `[THINK]` and switches into thinking when it sees one. Upstream
//! does exactly this, and it is not obviously an accident -- the tag is
//! unambiguous when it appears, and leaking `[THINK]` into user-visible content
//! would be worse than honouring it.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, overlap, trailing_whitespace_len};

/// **Upstream:** the `ministral*Tag` consts.
const TOOL_CALLS_TAG: &str = "[TOOL_CALLS]";
const THINK_TAG: &str = "[THINK]";
const THINK_END_TAG: &str = "[/THINK]";
const ARGS_TAG: &str = "[ARGS]";

/// **Upstream:** `ministralParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    Content,
    Thinking,
    /// Between `[TOOL_CALLS]` and `[ARGS]` -- the bare tool name.
    ToolName,
    /// After `[ARGS]` -- the JSON object, ended by [`find_json_end`].
    ToolArgs,
}

/// **Upstream:** `ministralEvent`. The tool-call variant carries the **raw**
/// name and JSON, because validation happens up in `add` where the tool list is
/// -- `eat` has no business knowing which tools exist.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Content(String),
    Thinking(String),
    ToolCall { name: String, args: String },
}

/// **Upstream:** `MinistralParser`.
#[derive(Debug, Default)]
pub struct MinistralParser {
    state: State,
    buffer: String,
    tools: Vec<Tool>,
    call_index: usize,
    has_thinking_support: bool,
    /// Holds the tool name while the arguments are still arriving.
    pending_tool_name: String,
}

impl MinistralParser {
    /// `parser_for_name("ministral")` uses `false`. **Upstream:**
    /// `&MinistralParser{hasThinkingSupport: false}`.
    pub fn new(has_thinking_support: bool) -> Self {
        Self {
            has_thinking_support,
            ..Default::default()
        }
    }

    /// **Upstream:** `setInitialState`. Note it never consults the `think`
    /// value at all -- only the model's own capability flag and the prefill.
    fn set_initial_state(&mut self, last_message: Option<&Message>) {
        if !self.has_thinking_support {
            self.state = State::Content;
            return;
        }
        let prefill = last_message.is_some_and(|m| m.role == "assistant");
        if prefill && last_message.is_some_and(|m| !m.content.is_empty()) {
            self.state = State::Content;
            return;
        }
        self.state = State::Thinking;
    }

    /// **Upstream:** `(*MinistralParser).parseEvents`.
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

    /// **Upstream:** `(*MinistralParser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        match self.state {
            State::Content => {
                if let Some(idx) = self.buffer.find(TOOL_CALLS_TAG) {
                    let before = self.buffer[..idx].trim_end().to_string();
                    self.buffer = self.buffer[idx + TOOL_CALLS_TAG.len()..].to_string();
                    self.state = State::ToolName;
                    let mut events = Vec::new();
                    if !before.is_empty() {
                        events.push(Event::Content(before));
                    }
                    return (events, true);
                }

                // `[THINK]` is honoured even when `has_thinking_support` is
                // false -- see the module docs.
                if let Some(idx) = self.buffer.find(THINK_TAG) {
                    let before = self.buffer[..idx].trim_end().to_string();
                    self.buffer = self.buffer[idx + THINK_TAG.len()..].to_string();
                    self.state = State::Thinking;
                    let mut events = Vec::new();
                    if !before.is_empty() {
                        events.push(Event::Content(before));
                    }
                    return (events, true);
                }

                // Hold back whichever partial tag is longer, plus the
                // whitespace in front of it. With no partial tag at all this
                // still holds the trailing whitespace, because a tag may be the
                // very next thing to land and both tags trim it.
                let max_overlap = overlap(&self.buffer, TOOL_CALLS_TAG)
                    .max(overlap(&self.buffer, THINK_TAG));
                let split = self.buffer.len() - max_overlap;
                let (before_partial_tag, _) = super::chop(&self.buffer, split);
                let ambiguous_start =
                    before_partial_tag.len() - trailing_whitespace_len(before_partial_tag);
                let (unambiguous, ambiguous) = super::chop(&self.buffer, ambiguous_start);
                let (unambiguous, ambiguous) = (unambiguous.to_string(), ambiguous.to_string());
                self.buffer = ambiguous;

                let mut events = Vec::new();
                if !unambiguous.is_empty() {
                    events.push(Event::Content(unambiguous));
                }
                (events, false)
            }

            State::Thinking => {
                if let Some(idx) = self.buffer.find(THINK_END_TAG) {
                    let thinking = self.buffer[..idx].to_string();
                    self.buffer = self.buffer[idx + THINK_END_TAG.len()..]
                        .trim_start()
                        .to_string();
                    self.state = State::Content;
                    let mut events = Vec::new();
                    if !thinking.is_empty() {
                        events.push(Event::Thinking(thinking));
                    }
                    return (events, true);
                }

                // NOTE: unlike the content state above, this one does NOT widen
                // the held-back region over trailing whitespace, and it does not
                // trim the thinking text either. Upstream keeps thinking
                // verbatim -- `[/THINK]` only trims what comes AFTER it. So a
                // trailing space inside a thinking block survives, on purpose.
                let overlap_len = overlap(&self.buffer, THINK_END_TAG);
                if overlap_len > 0 {
                    let (unambiguous, ambiguous) =
                        super::chop(&self.buffer, self.buffer.len() - overlap_len);
                    let (unambiguous, ambiguous) = (unambiguous.to_string(), ambiguous.to_string());
                    self.buffer = ambiguous;
                    let mut events = Vec::new();
                    if !unambiguous.is_empty() {
                        events.push(Event::Thinking(unambiguous));
                    }
                    return (events, false);
                }

                let thinking = std::mem::take(&mut self.buffer);
                let mut events = Vec::new();
                if !thinking.is_empty() {
                    events.push(Event::Thinking(thinking));
                }
                (events, false)
            }

            State::ToolName => {
                let Some(idx) = self.buffer.find(ARGS_TAG) else {
                    // The name is bare text with no terminator of its own, so
                    // nothing can be decided until `[ARGS]` shows up.
                    return (Vec::new(), false);
                };
                // NOT trimmed -- upstream takes the name exactly as written.
                self.pending_tool_name = self.buffer[..idx].to_string();
                self.buffer = self.buffer[idx + ARGS_TAG.len()..].to_string();
                self.state = State::ToolArgs;
                (Vec::new(), true)
            }

            State::ToolArgs => {
                let Some(json_end) = find_json_end(&self.buffer) else {
                    // Incomplete JSON. Wait -- there is no closing tag to lean
                    // on, so the brace counter is the only end marker.
                    return (Vec::new(), false);
                };
                let json_str = self.buffer[..json_end + 1].to_string();
                self.buffer = self.buffer[json_end + 1..].to_string();
                let name = std::mem::take(&mut self.pending_tool_name);
                self.state = State::Content;
                (vec![Event::ToolCall { name, args: json_str }], true)
            }
        }
    }
}

impl Parser for MinistralParser {
    fn init(
        &mut self,
        tools: Vec<Tool>,
        last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.tools = tools.clone();
        self.call_index = 0;
        // Upstream does not clear these in Init; cleared here so a re-`init`
        // cannot inherit half a tag. Stated divergence, no behaviour change for
        // the documented lifecycle.
        self.buffer.clear();
        self.pending_tool_name.clear();
        self.set_initial_state(last_message);
        tools
    }

    /// **Upstream:** `(*MinistralParser).Add`.
    ///
    /// Returns `Err` on an unknown tool name or unparseable arguments. Upstream
    /// returns the content and thinking gathered *so far* alongside the error;
    /// Rust's `Result` cannot carry both, so the partial output is dropped and
    /// the error wins. The caller must treat the turn as failed either way, and
    /// showing half a reply next to a hard error is worse than showing none.
    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);

        let mut out = Parsed::default();
        for event in self.parse_events() {
            match event {
                Event::Content(c) => out.content.push_str(&c),
                Event::Thinking(t) => out.thinking.push_str(&t),
                Event::ToolCall { name, args } => {
                    let tool = tool_by_name(&self.tools, &name)?;
                    let arguments: ToolCallArguments = serde_json::from_str(&args)?;
                    out.calls.push(ToolCall {
                        function: ToolCallFunction {
                            name: tool.function.name.clone(),
                            arguments,
                            ..Default::default()
                        },
                        ..Default::default()
                    });
                }
            }
        }

        for call in &mut out.calls {
            call.function.index = self.call_index;
            self.call_index += 1;
        }

        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![TOOL_CALLS_TAG, THINK_TAG, THINK_END_TAG, ARGS_TAG]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        self.has_thinking_support
    }
}

/// Find an offered tool by exact name. **Upstream:** `toolByName`.
///
/// Exact match, no normalisation -- upstream compares with `==`, and inventing a
/// case-insensitive or trimmed match here would let through names the real
/// ollama rejects.
fn tool_by_name<'a>(tools: &'a [Tool], name: &str) -> Result<&'a Tool, ParserError> {
    tools
        .iter()
        .find(|t| t.function.name == name)
        .ok_or_else(|| ParserError::MalformedToolCall(format!("tool '{name}' not found")))
}

/// Byte index of the brace/bracket that closes the root JSON value, or `None`
/// if it has not arrived yet.
///
/// **Upstream:** `findJSONEnd`. This exists because a ministral tool call has
/// **no closing tag** -- the arguments simply end when the JSON does, so the
/// only way to know is to count.
///
/// It tracks string literals and backslash escapes, which is the part that
/// matters: `{"path": "a\"}b"}` contains a `}` inside a string, and a naive
/// counter stops on it and hands back invalid JSON.
///
/// **Faithful quirk:** `{` and `[` both increment, `}` and `]` both decrement,
/// with no check that they match. So `{"a": 1]` is accepted as complete. That is
/// upstream's, and it does not matter in practice -- the resulting string still
/// has to survive `serde_json`, which rejects it right afterwards.
fn find_json_end(s: &str) -> Option<usize> {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, r) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if r == '\\' {
                escaped = true;
            } else if r == '"' {
                in_string = false;
            }
            continue;
        }

        match r {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
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

    /// What `parser_for_name("ministral")` builds: thinking support OFF.
    fn ministral(tools: Vec<Tool>) -> MinistralParser {
        let mut p = MinistralParser::new(false);
        p.init(tools, None, None);
        p
    }

    #[test]
    fn plain_text_with_no_tags_is_all_content() {
        let mut p = ministral(Vec::new());
        let got = p.add("Hello, how can I help?", true).expect("add");
        assert_eq!(got.content, "Hello, how can I help?");
        assert!(got.thinking.is_empty());
        assert!(got.calls.is_empty());
    }

    #[test]
    fn a_tool_call_is_name_then_args_with_no_closing_tag() {
        let mut p = ministral(vec![tool("get_weather")]);
        let got = p
            .add(r#"Checking.[TOOL_CALLS]get_weather[ARGS]{"city":"Paris"}"#, true)
            .expect("add");
        assert_eq!(got.content, "Checking.");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[0].function.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn two_calls_in_a_row_are_both_found_and_indexed() {
        let mut p = ministral(vec![tool("f"), tool("g")]);
        let got = p
            .add(r#"[TOOL_CALLS]f[ARGS]{"a":1}[TOOL_CALLS]g[ARGS]{"b":2}"#, true)
            .expect("add");
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.name, "f");
        assert_eq!(got.calls[1].function.name, "g");
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.index, 1);
    }

    #[test]
    fn arguments_keep_the_order_the_model_wrote_them() {
        let mut p = ministral(vec![tool("f")]);
        let got = p
            .add(r#"[TOOL_CALLS]f[ARGS]{"zebra":1,"apple":2,"mango":3}"#, true)
            .expect("add");
        let keys: Vec<&str> = got.calls[0]
            .function
            .arguments
            .0
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["zebra", "apple", "mango"]);
    }

    /// The behaviour unique to this family: an unknown tool name is a **hard
    /// error**, not a silent drop.
    #[test]
    fn an_unknown_tool_name_is_a_hard_error() {
        let mut p = ministral(vec![tool("get_weather")]);
        let err = p
            .add(r#"[TOOL_CALLS]not_a_real_tool[ARGS]{"x":1}"#, true)
            .expect_err("should fail");
        assert!(matches!(err, ParserError::MalformedToolCall(ref m) if m.contains("not_a_real_tool")));
    }

    #[test]
    fn unparseable_arguments_are_a_hard_error_too() {
        let mut p = ministral(vec![tool("f")]);
        // `{"a": 1]` passes the naive brace counter but is not valid JSON.
        let err = p.add(r#"[TOOL_CALLS]f[ARGS]{"a": 1]"#, true).expect_err("should fail");
        assert!(matches!(err, ParserError::Json(_)));
    }

    /// `[THINK]` is honoured even though `has_thinking_support` is false.
    #[test]
    fn a_think_tag_switches_to_thinking_even_with_thinking_support_off() {
        let mut p = ministral(Vec::new());
        assert!(!p.has_thinking_support());
        let got = p
            .add("before[THINK]reasoning[/THINK]after", true)
            .expect("add");
        assert_eq!(got.thinking, "reasoning");
        assert_eq!(got.content, "beforeafter");
    }

    /// With thinking support ON the stream STARTS inside thinking -- no opening
    /// `[THINK]` needed, because the prompt supplied it.
    #[test]
    fn with_thinking_support_the_stream_starts_inside_the_think_block() {
        let mut p = MinistralParser::new(true);
        p.init(Vec::new(), None, None);
        let got = p.add("reasoning[/THINK]the answer", true).expect("add");
        assert_eq!(got.thinking, "reasoning");
        assert_eq!(got.content, "the answer");
    }

    /// An assistant prefill with content skips thinking even when supported.
    #[test]
    fn an_assistant_content_prefill_starts_the_stream_in_content_mode() {
        let mut p = MinistralParser::new(true);
        let last = Message::new("assistant", "Sure:");
        p.init(Vec::new(), Some(&last), None);
        let got = p.add(" here you go", true).expect("add");
        assert_eq!(got.content, " here you go");
        assert!(got.thinking.is_empty());
    }

    #[test]
    fn a_tool_calls_tag_split_across_chunks_never_leaks() {
        let mut p = ministral(vec![tool("f")]);
        let a = p.add("text[TOOL_", false).expect("add");
        assert_eq!(a.content, "text");
        let b = p.add(r#"CALLS]f[ARGS]{"a":1}"#, true).expect("add");
        assert!(b.content.is_empty());
        assert_eq!(b.calls.len(), 1);
    }

    #[test]
    fn a_think_end_tag_split_across_chunks_never_leaks() {
        let mut p = MinistralParser::new(true);
        p.init(Vec::new(), None, None);
        let a = p.add("thought[/THI", false).expect("add");
        assert_eq!(a.thinking, "thought");
        let b = p.add("NK]visible", true).expect("add");
        assert!(b.thinking.is_empty());
        assert_eq!(b.content, "visible");
    }

    /// One byte at a time must agree with one big chunk -- and unlike lfm2 and
    /// cogito, ministral's content state DOES buffer its partial tags, so the
    /// tool call survives.
    #[test]
    fn feeding_one_byte_at_a_time_gives_the_same_answer_as_one_big_chunk() {
        let input = r#"Checking now.[TOOL_CALLS]get_weather[ARGS]{"city":"SG"}"#;

        let mut whole = ministral(vec![tool("get_weather")]);
        let want = whole.add(input, true).expect("add");

        let mut p = ministral(vec![tool("get_weather")]);
        let mut got = Parsed::default();
        for (i, ch) in input.char_indices() {
            let part = p
                .add(&input[i..i + ch.len_utf8()], i + ch.len_utf8() == input.len())
                .expect("add");
            got.content.push_str(&part.content);
            got.thinking.push_str(&part.thinking);
            got.calls.extend(part.calls);
        }

        assert_eq!(got.content, want.content);
        assert_eq!(got.content, "Checking now.");
        assert_eq!(got.calls.len(), want.calls.len());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.arguments.get("city"), Some(&json!("SG")));
    }

    /// [`find_json_end`] has to survive braces inside string literals and
    /// escaped quotes -- that is the whole reason it is not a two-line counter.
    #[test]
    fn find_json_end_ignores_braces_inside_strings_and_escapes() {
        assert_eq!(find_json_end(r#"{"a":1}"#), Some(6));
        // A `}` inside a string must not end the object.
        let s = r#"{"a":"}"}tail"#;
        assert_eq!(find_json_end(s), Some(8));
        assert_eq!(&s[..9], r#"{"a":"}"}"#);
        // An escaped quote does not close the string, so the `}` inside stays
        // inside.
        let s = r#"{"a":"x\"}y"}"#;
        assert_eq!(find_json_end(s).map(|i| &s[..i + 1]), Some(s));
        // Nested structures.
        assert_eq!(find_json_end(r#"{"a":{"b":[1,2]}}"#), Some(16));
        // Incomplete -> None, so the parser waits instead of guessing.
        assert_eq!(find_json_end(r#"{"a":1"#), None);
        assert_eq!(find_json_end(""), None);
    }

    /// Arguments arriving in dribs and drabs must not be cut short by the
    /// brace counter.
    #[test]
    fn incomplete_json_arguments_are_buffered_until_the_object_closes() {
        let mut p = ministral(vec![tool("f")]);
        let a = p.add(r#"[TOOL_CALLS]f[ARGS]{"a":{"b":"#, false).expect("add");
        assert!(a.calls.is_empty(), "must not fire on a half-written object");
        let b = p.add("1}}", true).expect("add");
        assert_eq!(b.calls.len(), 1);
        assert_eq!(b.calls[0].function.arguments.get("a"), Some(&json!({"b": 1})));
    }

    #[test]
    fn ministral_advertises_its_four_bracket_tags() {
        let p = ministral(Vec::new());
        assert_eq!(
            p.preserved_tokens(),
            vec!["[TOOL_CALLS]", "[THINK]", "[/THINK]", "[ARGS]"]
        );
        assert!(p.has_tool_support());
        assert!(!p.has_thinking_support());
        assert!(MinistralParser::new(true).has_thinking_support());
    }
}
