//! FunctionGemma response parser.
//!
//! **Upstream:** `model/parsers/functiongemma.go` (ollama, MIT, Copyright (c)
//! Ollama). Ported against `4713800b08b2ddf5e14acf8398953cf7b12f169b`
//! (2026-07-28).
//!
//! ## A format of its own -- not JSON, not Python
//!
//! ```text
//! <start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>
//! ```
//!
//! Arguments are `key:value` pairs separated by commas. Values are bare. There
//! are **no quotes at all** -- a string is wrapped in `<escape>` ... `<escape>`
//! instead, using the *same* marker at both ends (it toggles).
//!
//! That toggle is what makes the marker work: inside a pair of `<escape>` tags,
//! commas, braces and colons are literal data, so
//! `note:<escape>a, b{c}<escape>` is ONE argument whose value is `a, b{c}`.
//! Treat `<escape>` as decoration and you split that into three broken
//! arguments.
//!
//! ## Thinking: none
//!
//! [`has_thinking_support`](Parser::has_thinking_support) is `false` and there
//! are no thinking tags. Content and tool calls only.

use crate::api::{Message, ThinkValue, Tool, ToolCall, ToolCallArguments, ToolCallFunction};

use super::{Parsed, Parser, ParserError, overlap};

/// **Upstream:** `functionGemmaFunctionCallOpen` / `...Close`.
const FUNCTION_CALL_OPEN: &str = "<start_function_call>";
const FUNCTION_CALL_CLOSE: &str = "<end_function_call>";
/// The toggling string marker. **Upstream:** the literal `"<escape>"` in
/// `splitArguments` and `parseValue`.
const ESCAPE_TAG: &str = "<escape>";
/// **Upstream:** the `call:` prefix in `functionGemmaCallRegex`.
const CALL_PREFIX: &str = "call:";

/// **Upstream:** `FunctionGemmaParserState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    CollectingContent,
    CollectingToolCalls,
}

/// **Upstream:** `functionGemmaEvent`.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Content(String),
    ToolCall(Box<ToolCall>),
}

/// **Upstream:** `FunctionGemmaParser`.
#[derive(Debug, Default)]
pub struct FunctionGemmaParser {
    state: State,
    buffer: String,
    call_index: usize,
}

impl FunctionGemmaParser {
    /// **Upstream:** `(*FunctionGemmaParser).parseEvents`.
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

    /// **Upstream:** `(*FunctionGemmaParser).eat`.
    fn eat(&mut self) -> (Vec<Event>, bool) {
        if self.buffer.is_empty() {
            return (Vec::new(), false);
        }

        match self.state {
            State::CollectingContent => {
                if let Some(idx) = self.buffer.find(FUNCTION_CALL_OPEN) {
                    // NOTE: content is taken RAW -- upstream does not trim the
                    // whitespace in front of the open tag here, unlike most
                    // other families. Upstream's `unicode_content_and_arguments`
                    // fixture pins it: the expected content is
                    // `"\u{3053}\u{3093}\u{306B}\u{3061}\u{306F} "`, trailing
                    // space and all.
                    let content = self.buffer[..idx].to_string();
                    self.buffer = self.buffer[idx + FUNCTION_CALL_OPEN.len()..].to_string();
                    self.state = State::CollectingToolCalls;
                    let mut events = Vec::new();
                    if !content.is_empty() {
                        events.push(Event::Content(content));
                    }
                    return (events, true);
                }

                // Hold back only the ambiguous tail -- and, again, NO
                // whitespace widening. This is what lets `"a <value> tag"`
                // survive: the `<` is held, then released intact when `value`
                // proves it was never `<start_function_call>`.
                let (unambiguous, ambiguous) = self.emit_with_partial_check(FUNCTION_CALL_OPEN);
                self.buffer = ambiguous;
                let mut events = Vec::new();
                if !unambiguous.is_empty() {
                    events.push(Event::Content(unambiguous));
                }
                (events, false)
            }

            State::CollectingToolCalls => {
                let Some(idx) = self.buffer.find(FUNCTION_CALL_CLOSE) else {
                    // A half-written call body cannot be parsed. Wait.
                    return (Vec::new(), false);
                };
                let body = self.buffer[..idx].to_string();
                let remaining = self.buffer[idx + FUNCTION_CALL_CLOSE.len()..].to_string();
                self.buffer = remaining.clone();

                let mut events = Vec::new();
                if let Ok(tc) = parse_tool_call(&body) {
                    events.push(Event::ToolCall(Box::new(tc)));
                }

                // Stay in the tool-call state when another opener is ALREADY in
                // the buffer -- back-to-back calls then never bounce through
                // content and never emit an empty content event.
                if !remaining.contains(FUNCTION_CALL_OPEN) {
                    self.state = State::CollectingContent;
                }
                (events, true)
            }
        }
    }

    /// **Upstream:** `emitWithPartialCheck`. Split the buffer into what is safe
    /// to emit and what could still become `tag`.
    fn emit_with_partial_check(&self, tag: &str) -> (String, String) {
        let overlap_len = overlap(&self.buffer, tag);
        if overlap_len > 0 {
            let (before, after) = super::chop(&self.buffer, self.buffer.len() - overlap_len);
            return (before.to_string(), after.to_string());
        }
        (self.buffer.clone(), String::new())
    }
}

impl Parser for FunctionGemmaParser {
    /// **Upstream:** `(*FunctionGemmaParser).Init`. Tools are stored upstream
    /// but **never read** -- `parseToolCall` does no validation against them.
    /// Left out here rather than kept as a dead field; noted so a reader diffing
    /// against Go does not think it was missed.
    fn init(
        &mut self,
        tools: Vec<Tool>,
        _last_message: Option<&Message>,
        _think: Option<&ThinkValue>,
    ) -> Vec<Tool> {
        self.state = State::CollectingContent;
        self.call_index = 0;
        self.buffer.clear();
        tools
    }

    fn add(&mut self, s: &str, _done: bool) -> Result<Parsed, ParserError> {
        self.buffer.push_str(s);

        let mut out = Parsed::default();
        for event in self.parse_events() {
            match event {
                Event::Content(c) => out.content.push_str(&c),
                Event::ToolCall(tc) => out.calls.push(*tc),
            }
        }

        for call in &mut out.calls {
            call.function.index = self.call_index;
            self.call_index += 1;
        }

        Ok(out)
    }

    fn preserved_tokens(&self) -> Vec<&'static str> {
        vec![FUNCTION_CALL_OPEN, FUNCTION_CALL_CLOSE]
    }

    fn has_tool_support(&self) -> bool {
        true
    }

    fn has_thinking_support(&self) -> bool {
        false
    }
}

/// Pull `call:name{args}` out of one call body.
///
/// **Upstream:** `parseToolCall` plus its
/// `regexp.MustCompile(`call:([^{]+)\{(.*)\}`)`.
///
/// Hand-rolled rather than pulling in `regex`. The equivalence, spelled out:
///
/// * the pattern is **not anchored**, so it finds the first `call:`;
/// * `[^{]+` is greedy but cannot contain `{`, so the name runs to the **first**
///   `{` and must be at least one character;
/// * `(.*)` is greedy, so the arguments run to the **last** `}` -- which is what
///   makes a nested `{a:{b:1}}` come back whole instead of stopping at the inner
///   brace;
/// * **`.` does not match `\n` in Go's regexp**, so that last `}` must be on the
///   same line as the opening `{`. Reproduced here by clipping at the first
///   newline. Every upstream fixture is single-line, so this path is untested
///   upstream -- it is matched anyway rather than quietly widened.
///
/// **DELIBERATE DIVERGENCE, and the only one in this file.** Upstream's
/// signature returns an error but **no branch ever sets it**: a body that does
/// not match the regex falls through to `return toolCall, nil` with a
/// **zero-valued** `api.ToolCall`, and the caller's `if err == nil` guard --
/// dead code -- then appends it. The result is a tool call with an **empty
/// name** handed to the application.
///
/// We return `Err` instead, so the caller's existing `if let Ok` drops it. The
/// reasons: a nameless call cannot be dispatched by anybody, this crate already
/// has [`ParserError::EmptyFunctionName`] for precisely this, and the vestigial
/// `err == nil` check upstream reads like the filter that was intended. No
/// upstream fixture pins the empty-call behaviour, so nothing is contradicted.
///
/// **What would make this wrong:** if some caller downstream actually relies on
/// seeing a zero-valued call as a signal that a malformed block occurred. If
/// that ever turns up, this is the place to revert -- emit
/// `ToolCall::default()` instead of erroring.
fn parse_tool_call(content: &str) -> Result<ToolCall, ParserError> {
    let malformed =
        || ParserError::MalformedToolCall(format!("no call:name{{args}} in {content:?}"));

    let call_at = content.find(CALL_PREFIX).ok_or_else(malformed)?;
    let after_call = &content[call_at + CALL_PREFIX.len()..];

    let open = after_call.find('{').ok_or_else(malformed)?;
    let name = &after_call[..open];
    if name.is_empty() {
        // `[^{]+` demands at least one character.
        return Err(ParserError::EmptyFunctionName);
    }

    // `(.*)` cannot cross a newline; clip there, then take the LAST `}`.
    let rest = &after_call[open + 1..];
    let line_end = rest.find('\n').unwrap_or(rest.len());
    let line = &rest[..line_end];
    let close = line.rfind('}').ok_or_else(malformed)?;
    let args_str = &line[..close];

    Ok(ToolCall {
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: parse_arguments(args_str),
            ..Default::default()
        },
        ..Default::default()
    })
}

/// Parse `key:value,key:value`. **Upstream:** `parseArguments`.
///
/// A part with no `:` is **skipped silently**, and the key is taken raw -- no
/// trimming. Both are upstream's.
fn parse_arguments(args_str: &str) -> ToolCallArguments {
    let mut args = ToolCallArguments::new();
    if args_str.is_empty() {
        return args;
    }

    for part in split_arguments(args_str) {
        let Some(colon) = part.find(':') else {
            continue;
        };
        let key = &part[..colon];
        let value = &part[colon + 1..];
        args.set(key, parse_value(value));
    }

    args
}

/// Split on commas that are neither nested nor inside an `<escape>` region.
///
/// **Upstream:** `(*FunctionGemmaParser).splitArguments`.
///
/// `<escape>` **toggles** -- the same marker opens and closes -- so the first
/// one turns literal mode on and the second turns it off. While on, commas and
/// braces are data. The marker itself is **kept in the output**, because
/// [`parse_value`] is what strips it.
///
/// **Divergence, and why it is safe:** upstream indexes raw bytes; we walk
/// `char_indices`. Every character it treats specially (`{`, `}`, `[`, `]`, `,`)
/// and the whole `<escape>` literal are ASCII, so both see the same delimiters,
/// and char boundaries mean a multi-byte value can never be split mid-character.
///
/// Note an empty part is **dropped**, not kept: upstream only pushes when
/// `current.Len() > 0`, so `a:1,,b:2` yields two parts.
fn split_arguments(args_str: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth: i32 = 0;
    let mut in_escape = false;

    let mut it = args_str.char_indices();
    while let Some((i, ch)) = it.next() {
        if args_str[i..].starts_with(ESCAPE_TAG) {
            in_escape = !in_escape;
            current.push_str(ESCAPE_TAG);
            // Skip the remaining characters of the marker.
            for _ in 1..ESCAPE_TAG.chars().count() {
                it.next();
            }
            continue;
        }

        if in_escape {
            current.push(ch);
            continue;
        }

        match ch {
            '{' | '[' => {
                depth += 1;
                current.push(ch);
            }
            '}' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Parse one bare value. **Upstream:** `(*FunctionGemmaParser).parseValue`.
///
/// Order is upstream's: `<escape>`-wrapped string first (so
/// `<escape>true<escape>` stays the **string** `"true"`), then booleans, then
/// numbers, then `[...]`, then `{...}`, then a bare string as the fallback.
///
/// Note there is no null/None case at all -- this format has no null.
fn parse_value(value: &str) -> serde_json::Value {
    use serde_json::Value;

    // `<escape>x<escape>` -> the string `x`. Needs BOTH markers and enough
    // length that they are not the same one counted twice.
    if value.len() >= 2 * ESCAPE_TAG.len()
        && value.starts_with(ESCAPE_TAG)
        && value.ends_with(ESCAPE_TAG)
    {
        return Value::String(value[ESCAPE_TAG.len()..value.len() - ESCAPE_TAG.len()].to_string());
    }

    if value == "true" {
        return Value::Bool(true);
    }
    if value == "false" {
        return Value::Bool(false);
    }

    if let Some(n) = parse_number(value) {
        return n;
    }

    if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        return Value::Array(split_arguments(inner).iter().map(|p| parse_value(p)).collect());
    }

    if let Some(inner) = value.strip_prefix('{').and_then(|v| v.strip_suffix('}')) {
        let mut obj = serde_json::Map::new();
        for part in split_arguments(inner) {
            let Some(colon) = part.find(':') else {
                continue;
            };
            obj.insert(part[..colon].to_string(), parse_value(&part[colon + 1..]));
        }
        return Value::Object(obj);
    }

    Value::String(value.to_string())
}

/// Try to read `s` as a number. **Upstream:** `parseNumber`.
///
/// Upstream uses `fmt.Sscanf`, and the two branches behave very differently:
///
/// * the **integer** branch is strict -- it re-formats what it read and requires
///   the result to equal the whole input, so `"12abc"` and `"+5"` both fail;
/// * the **float** branch is NOT -- `Sscanf` happily reads the longest valid
///   float *prefix*, so `"12abc"` comes back as the float `12.0`.
///
/// That asymmetry is upstream's and is reproduced here: exact `i64` first, then
/// longest-`f64`-prefix. It looks like an oversight, and diverging would still
/// be wrong -- this port and ollama must agree on the same bytes.
///
/// The prefix scan walks lengths down from the whole string, so it finds the
/// longest match, exactly like `Sscanf`.
fn parse_number(s: &str) -> Option<serde_json::Value> {
    use serde_json::Value;

    if let Ok(i) = s.parse::<i64>()
        && i.to_string() == s
    {
        return Some(Value::Number(i.into()));
    }

    // Longest valid f64 prefix, mirroring `Sscanf("%f")`.
    for end in (1..=s.len()).rev() {
        let Some(prefix) = s.get(..end) else {
            continue;
        };
        if let Ok(f) = prefix.parse::<f64>()
            && let Some(n) = serde_json::Number::from_f64(f)
        {
            return Some(Value::Number(n));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gemma() -> FunctionGemmaParser {
        let mut p = FunctionGemmaParser::default();
        p.init(Vec::new(), None, None);
        p
    }

    /// Upstream's fixtures stream **one tiny piece at a time** -- the tags are
    /// deliberately shredded (`"<", "start", "_", "function", ...`). This helper
    /// reproduces that, and it is the real test of the buffering.
    fn run_chunks(chunks: &[&str]) -> Parsed {
        let mut p = gemma();
        let mut got = Parsed::default();
        for (i, c) in chunks.iter().enumerate() {
            let part = p.add(c, i == chunks.len() - 1).expect("add");
            got.content.push_str(&part.content);
            got.calls.extend(part.calls);
        }
        got
    }

    /// Upstream `plain_content`, streamed one character at a time.
    #[test]
    fn plain_content_streamed_one_character_at_a_time_comes_out_whole() {
        let got = run_chunks(&["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"]);
        assert_eq!(got.content, "Hello, world!");
        assert!(got.calls.is_empty());
    }

    /// Upstream `simple_tool_call` -- every tag shredded across chunks.
    #[test]
    fn a_tool_call_survives_having_every_tag_split_across_chunks() {
        let got = run_chunks(&[
            "<", "start", "_", "function", "_", "call", ">", "call", ":", "get", "_", "weather",
            "{", "city", ":", "<", "escape", ">", "Paris", "<", "escape", ">", "}", "<", "end",
            "_", "function", "_", "call", ">",
        ]);
        assert!(got.content.is_empty());
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[0].function.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn content_before_a_tool_call_is_kept() {
        let got = run_chunks(&[
            "L", "et", " ", "me", " ", "check", ".", "<start_function_call>",
            "call:get_weather{city:<escape>Paris<escape>}", "<end_function_call>",
        ]);
        assert_eq!(got.content, "Let me check.");
        assert_eq!(got.calls.len(), 1);
    }

    #[test]
    fn content_after_a_tool_call_is_kept() {
        let got = run_chunks(&[
            "<start_function_call>call:test{}<end_function_call>", "Done", "!",
        ]);
        assert_eq!(got.content, "Done!");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "test");
        assert!(got.calls[0].function.arguments.is_empty());
    }

    /// Upstream `content_with_angle_brackets` -- the regression the partial-tag
    /// buffering exists for. A lone `<` must be held, then released intact.
    #[test]
    fn a_lone_angle_bracket_in_content_is_held_then_released_intact() {
        let got = run_chunks(&[
            "The", " ", "result", " ", "is", " ", "a", " ", "<", "value", ">", " ", "tag",
        ]);
        assert_eq!(got.content, "The result is a <value> tag");
        assert!(got.calls.is_empty());
    }

    /// Upstream `unicode_content_and_arguments`. Note the expected content keeps
    /// its **trailing space** -- this family does not trim in front of the open
    /// tag.
    #[test]
    fn unicode_content_and_arguments_survive_and_whitespace_is_not_trimmed() {
        let got = run_chunks(&[
            "\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}", " ", "<start_function_call>",
            "call:greet{name:<escape>\u{65E5}\u{672C}\u{8A9E}<escape>}", "<end_function_call>",
        ]);
        assert_eq!(got.content, "\u{3053}\u{3093}\u{306B}\u{3061}\u{306F} ");
        assert_eq!(got.calls[0].function.name, "greet");
        assert_eq!(
            got.calls[0].function.arguments.get("name"),
            Some(&json!("\u{65E5}\u{672C}\u{8A9E}"))
        );
    }

    /// Upstream `numeric_arguments`, `boolean_arguments`, `float_argument`.
    ///
    /// `approx_constant` is allowed because upstream's float fixture is
    /// literally `3.14`; it is test data, not a botched `PI`.
    #[expect(clippy::approx_constant)]
    #[test]
    fn bare_values_parse_to_numbers_and_booleans() {
        let got = run_chunks(&["<start_function_call>call:add{a:1,b:2}<end_function_call>"]);
        assert_eq!(got.calls[0].function.arguments.get("a"), Some(&json!(1)));
        assert_eq!(got.calls[0].function.arguments.get("b"), Some(&json!(2)));

        let got = run_chunks(&[
            "<start_function_call>call:set_flag{enabled:true,verbose:false}<end_function_call>",
        ]);
        assert_eq!(got.calls[0].function.arguments.get("enabled"), Some(&json!(true)));
        assert_eq!(got.calls[0].function.arguments.get("verbose"), Some(&json!(false)));

        let got =
            run_chunks(&["<start_function_call>call:set_temp{value:3.14}<end_function_call>"]);
        assert_eq!(got.calls[0].function.arguments.get("value"), Some(&json!(3.14)));
    }

    /// Upstream `multiple_params_sorted` -- and our ordered map keeps the
    /// model's order, which Go's map could not.
    #[test]
    fn arguments_keep_the_order_the_model_wrote_them() {
        let got = run_chunks(&[
            "<start_function_call>call:search{query:<escape>test<escape>,limit:10,offset:0}<end_function_call>",
        ]);
        let a = &got.calls[0].function.arguments;
        assert_eq!(a.get("query"), Some(&json!("test")));
        assert_eq!(a.get("limit"), Some(&json!(10)));
        assert_eq!(a.get("offset"), Some(&json!(0)));
        let keys: Vec<&str> = a.0.keys().map(String::as_str).collect();
        assert_eq!(keys, ["query", "limit", "offset"]);
    }

    /// Upstream `nested_object_argument`.
    #[test]
    fn nested_objects_are_parsed_recursively() {
        let got = run_chunks(&[
            "<start_function_call>call:create{config:{settings:{enabled:true,name:<escape>test<escape>}}}<end_function_call>",
        ]);
        assert_eq!(
            got.calls[0].function.arguments.get("config"),
            Some(&json!({"settings": {"enabled": true, "name": "test"}}))
        );
    }

    /// Upstream `multiple_tool_calls` -- back-to-back, indexed in order.
    #[test]
    fn back_to_back_tool_calls_are_both_found_and_indexed() {
        let got = run_chunks(&[
            "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>",
            "<start_function_call>call:get_time{tz:<escape>UTC<escape>}<end_function_call>",
        ]);
        assert_eq!(got.calls.len(), 2);
        assert_eq!(got.calls[0].function.name, "get_weather");
        assert_eq!(got.calls[1].function.name, "get_time");
        assert_eq!(got.calls[0].function.index, 0);
        assert_eq!(got.calls[1].function.index, 1);
    }

    #[test]
    fn empty_input_produces_nothing() {
        let mut p = gemma();
        let got = p.add("", true).expect("add");
        assert!(got.content.is_empty());
        assert!(got.calls.is_empty());
    }

    /// The `<escape>` toggle is what protects a comma or brace inside a string
    /// value from being read as structure.
    #[test]
    fn commas_and_braces_inside_an_escape_region_are_literal_data() {
        let got = run_chunks(&[
            "<start_function_call>call:note{text:<escape>a, b{c}<escape>,n:1}<end_function_call>",
        ]);
        let a = &got.calls[0].function.arguments;
        assert_eq!(a.len(), 2, "the comma inside <escape> must not split");
        assert_eq!(a.get("text"), Some(&json!("a, b{c}")));
        assert_eq!(a.get("n"), Some(&json!(1)));
    }

    /// An escaped value that LOOKS like a keyword stays a string.
    #[test]
    fn an_escaped_value_that_looks_like_a_keyword_stays_a_string() {
        assert_eq!(parse_value("<escape>true<escape>"), json!("true"));
        assert_eq!(parse_value("<escape>42<escape>"), json!("42"));
        assert_eq!(parse_value("true"), json!(true));
        assert_eq!(parse_value("42"), json!(42));
    }

    /// Upstream's asymmetric number parsing, pinned. Integers are exact; floats
    /// take the longest valid prefix, so `"12abc"` really does become `12.0`.
    #[test]
    fn integers_are_exact_but_floats_take_the_longest_valid_prefix() {
        assert_eq!(parse_number("42"), Some(json!(42)));
        assert_eq!(parse_number("0"), Some(json!(0)));
        assert_eq!(parse_number("-7"), Some(json!(-7)));
        // Not an exact integer, so it falls to the float branch.
        assert_eq!(parse_number("3.5"), Some(json!(3.5)));
        // The asymmetry: a numeric PREFIX is enough for the float branch.
        assert_eq!(parse_number("12abc"), Some(json!(12.0)));
        // `+5` is a valid i64 parse in Rust but re-formats as `5`, so the exact
        // check rejects it -- exactly as Go's re-format check does.
        assert_eq!(parse_number("+5"), Some(json!(5.0)));
        // Nothing numeric at all.
        assert_eq!(parse_number("Paris"), None);
        assert_eq!(parse_number(""), None);
    }

    /// **The deliberate divergence.** Upstream hands back a zero-valued
    /// `api.ToolCall` (empty name!) for a body that does not match; we drop it.
    /// See [`parse_tool_call`] for the full reasoning and how to revert.
    #[test]
    fn a_body_with_no_call_pattern_is_dropped_rather_than_emitted_nameless() {
        assert!(parse_tool_call("no call pattern at all").is_err());
        assert!(parse_tool_call("call:no_braces_here").is_err());
        assert!(parse_tool_call("call:{no_name}").is_err());

        // End to end: a malformed block yields NO call, and the surrounding
        // content still comes through.
        let got = run_chunks(&["before<start_function_call>rubbish<end_function_call>after"]);
        assert!(
            got.calls.is_empty(),
            "upstream would emit a nameless call here; we drop it"
        );
        assert_eq!(got.content, "beforeafter");
    }

    /// The arguments run to the LAST `}` on the line, which is what keeps a
    /// nested object whole.
    #[test]
    fn the_arguments_run_to_the_last_closing_brace_not_the_first() {
        let tc = parse_tool_call("call:f{a:{b:1}}").expect("parse");
        assert_eq!(tc.function.name, "f");
        assert_eq!(tc.function.arguments.get("a"), Some(&json!({"b": 1})));
    }

    #[test]
    fn functiongemma_advertises_its_tags_and_has_no_thinking() {
        let p = gemma();
        assert_eq!(
            p.preserved_tokens(),
            vec!["<start_function_call>", "<end_function_call>"]
        );
        assert!(p.has_tool_support());
        assert!(!p.has_thinking_support());
    }
}
